"""Read the supplied corpus and assign stable, source-backed passage IDs."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_SOURCE = re.compile(r"^Source:\s+(https?://\S+)\s*$", re.IGNORECASE)
_PAGE_CHARS = 2_400


@dataclass(frozen=True)
class Evidence:
    id: str
    section_id: str
    filename: str
    section: str
    start_line: int
    end_line: int
    text: str
    source_url: str | None
    drug: str | None
    source_type: str

    def preview(self, max_chars: int = 1_000) -> "Evidence":
        """Return a shorter search preview while keeping its line range exact."""
        if len(self.text) <= max_chars:
            return self
        lines = self.text.split("\n")
        end = 1
        size = len(lines[0])
        while end < len(lines) and size + 1 + len(lines[end]) <= max_chars:
            size += 1 + len(lines[end])
            end += 1
        if end == len(lines):
            return self
        text = "\n".join(lines[:end])
        return replace(
            self,
            id=_passage_id(self.section_id, self.start_line - 1, text),
            end_line=self.start_line + end - 1,
            text=text,
        )


@dataclass(frozen=True)
class Corpus:
    chunks: tuple[Evidence, ...]
    sections: dict[str, tuple[Evidence, ...]]
    document_count: int

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        if not path.is_dir():
            raise ValueError(f"Corpus directory does not exist: {path}")

        sections: dict[str, tuple[Evidence, ...]] = {}
        document_count = 0
        for file in sorted(path.iterdir()):
            if not file.is_file() or file.suffix.lower() not in {".md", ".txt"}:
                continue
            document_sections = _read_document(file)
            if document_sections:
                sections.update(document_sections)
                document_count += 1

        if not sections:
            raise ValueError(f"No readable passages in corpus directory: {path}")
        return cls(
            chunks=tuple(page for pages in sections.values() for page in pages),
            sections=sections,
            document_count=document_count,
        )


def _read_document(file: Path) -> dict[str, tuple[Evidence, ...]]:
    lines = file.read_text(encoding="utf-8-sig").split("\n")
    stem = file.stem.casefold()
    prefix, _, subject = stem.partition("-")
    if prefix in {"dailymed", "medlineplus"} and subject:
        source_type, drug = prefix, subject.replace("-", " ")
    else:
        source_type, drug = ("glossary" if stem == "glossary" else "other"), None
    source_url = next(
        (match.group(1) for line in lines if (match := _SOURCE.fullmatch(line))),
        None,
    )
    headings = [
        (index, len(match.group(1)), match.group(2))
        for index, line in enumerate(lines)
        if (match := _HEADING.fullmatch(line))
    ]
    if not headings or headings[0][0] > 0:
        headings.insert(0, (0, 0, file.stem))

    sections = {}
    parents: list[tuple[int, str]] = []
    for index, (start, level, title) in enumerate(headings):
        while parents and parents[-1][0] >= level:
            parents.pop()
        parents.append((level, title))
        heading_path = " > ".join(name for _, name in parents)
        stop = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        while stop > start and not lines[stop - 1].strip():
            stop -= 1
        body_start = start + int(level > 0)
        while body_start < stop and not lines[body_start].strip():
            body_start += 1
        if not any(
            line.strip() and not _SOURCE.fullmatch(line)
            for line in lines[body_start:stop]
        ):
            continue

        section_id = _passage_id(file.name, start, "\n".join(lines[start:stop]))
        pages = []
        for page_start, page_stop in _page_ranges(lines, body_start, stop):
            text = "\n".join(lines[page_start:page_stop])
            pages.append(
                Evidence(
                    id=_passage_id(section_id, page_start, text),
                    section_id=section_id,
                    filename=file.name,
                    section=heading_path,
                    start_line=page_start + 1,
                    end_line=page_stop,
                    text=text,
                    source_url=source_url,
                    drug=drug,
                    source_type=source_type,
                )
            )
        sections[section_id] = tuple(pages)
    return sections


def _page_ranges(lines: list[str], start: int, stop: int) -> Iterator[tuple[int, int]]:
    """Split at a nearby paragraph break where possible and keep source lines intact."""
    while start < stop:
        end = start
        size = 0
        paragraph_break = None
        while end < stop:
            line_size = len(lines[end]) + 1
            if end > start and size + line_size > _PAGE_CHARS:
                break
            size += line_size
            if not lines[end].strip() and size >= _PAGE_CHARS // 2:
                paragraph_break = end
            end += 1
        if end < stop and paragraph_break is not None:
            end = paragraph_break
        if any(line.strip() for line in lines[start:end]):
            yield start, end
        start = end
        while start < stop and not lines[start].strip():
            start += 1


def _passage_id(identity: str, offset: int, text: str) -> str:
    return hashlib.blake2s(
        f"{identity}\n{offset}\n{text}".encode(), digest_size=12
    ).hexdigest()
