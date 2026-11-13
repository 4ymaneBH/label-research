from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)
]
AnswerStatus = Literal["answered", "partial", "insufficient_evidence"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question


class Citation(BaseModel):
    id: int
    filename: str
    section: str
    start_line: int
    end_line: int
    excerpt: str
    source_url: str | None


class ToolTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    evidence_ids: list[str]
    elapsed_ms: int


class ReviewSummary(BaseModel):
    outcome: Literal["passed", "repaired", "limited"]
    findings: list[str]


class AskResponse(BaseModel):
    answer: str
    status: AnswerStatus
    citations: list[Citation]
    plan: list[str]
    trace: list[ToolTrace]
    review: ReviewSummary
    disclaimer: str
    request_id: str


class Plan(BaseModel):
    subquestions: list[
        Annotated[str, StringConstraints(min_length=1, max_length=400)]
    ] = Field(min_length=1, max_length=6)


class Claim(BaseModel):
    text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1100)
    ]
    evidence_ids: list[str] = Field(max_length=8)


class Draft(BaseModel):
    claims: list[Claim] = Field(max_length=14)


class Verdict(BaseModel):
    claim_id: int
    verdict: Literal["supported", "unsupported", "contradicted", "ambiguous"]
    reason: str = Field(max_length=500)


class Critique(BaseModel):
    verdicts: list[Verdict] = Field(max_length=20)
    unanswered: list[Annotated[str, StringConstraints(max_length=500)]] = Field(
        max_length=6
    )
