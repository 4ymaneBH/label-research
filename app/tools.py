from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.corpus import Evidence
from app.retrieval import RetrievalIndex
from app.schemas import ToolTrace


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=400)
    drug: str | None = Field(default=None, max_length=100)
    source_type: Literal["dailymed", "medlineplus", "glossary", "other"] | None = None
    section: str | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=8)


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str = Field(min_length=1, max_length=200)
    page: int = Field(default=1, ge=1, le=100)


TOOLS = {
    "search_corpus": (
        SearchArgs,
        "Find relevant passages. Optional filters are generic drug name, source type, and a "
        "section heading substring. Returned section IDs and page numbers can be opened "
        "with read_section. For comparisons, search for each drug separately.",
    ),
    "read_section": (
        ReadArgs,
        "Read the original corpus text around a search match. Use the section ID and page "
        "from search. Inspect other pages when completeness matters. Only this tool's "
        "evidence IDs can support final claims.",
    ),
}
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }
    for name, (model, description) in TOOLS.items()
]


def describe(evidence: Evidence) -> dict:
    return {
        "evidence_id": evidence.id,
        "section_id": evidence.section_id,
        "filename": evidence.filename,
        "section": evidence.section,
        "start_line": evidence.start_line,
        "end_line": evidence.end_line,
        "text": evidence.text,
    }


def run_tool(
    index: RetrievalIndex, name: str, raw_args: str
) -> tuple[dict, ToolTrace, Evidence | None]:
    started = monotonic()
    args = {}
    ids = []
    read_evidence = None
    try:
        if name not in TOOLS:
            result = {"error": "Unknown tool. Choose one of the available tools."}
        else:
            parsed = TOOLS[name][0].model_validate_json(raw_args)
            args = parsed.model_dump(exclude_none=True)
            if isinstance(parsed, SearchArgs):
                hits = index.search(**args)
                ids = [hit.evidence.id for hit in hits]
                result = {
                    "hits": [
                        {
                            **describe(hit.evidence),
                            "page": hit.page,
                            "total_pages": hit.total_pages,
                            "score": round(hit.score, 4),
                        }
                        for hit in hits
                    ]
                }
                if not hits:
                    result["suggestion"] = "Try a broader search or a different filter."
            else:
                page = index.read_section(**args)
                read_evidence = page.evidence
                ids = [read_evidence.id]
                result = {
                    "evidence": describe(read_evidence),
                    "page": page.page,
                    "total_pages": page.total_pages,
                }
    except ValidationError as exc:
        result = {
            "error": "Invalid tool arguments.",
            "fields": [
                ".".join(str(part) for part in error["loc"]) for error in exc.errors()
            ],
        }
    except (KeyError, ValueError):
        result = {"error": "The requested section, page, or filter was not usable."}

    trace = ToolTrace(
        tool=name if name in TOOLS else "unknown",
        arguments=args,
        evidence_ids=ids,
        elapsed_ms=int((monotonic() - started) * 1000),
    )
    return result, trace, read_evidence
