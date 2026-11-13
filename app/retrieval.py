"""Rank corpus passages by cosine similarity and read exact source sections."""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from app.corpus import Corpus, Evidence
from app.llm import ProviderError

Embed = Callable[[list[str]], list[list[float]]]
_BATCH_SIZE = 48
_WORD = re.compile(r"[^\W_]+")
_STOP_WORDS = frozenset(
    "a an and are as at be by do for from how in is it of on or that the to was were what when which with".split()
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    evidence: Evidence
    score: float
    page: int
    total_pages: int


@dataclass(frozen=True)
class SectionPage:
    evidence: Evidence
    page: int
    total_pages: int


class RetrievalIndex:
    def __init__(self, corpus: Corpus, embed: Embed | None, matrix: np.ndarray | None):
        self._corpus = corpus
        self._embed = embed
        self._matrix = matrix
        # Give section headings extra weight in the local keyword index.
        counts = [
            _terms(f"{item.filename} {item.section} {item.section} {item.text}")
            for item in corpus.chunks
        ]
        frequencies = Counter(term for passage in counts for term in passage)
        self._idf = {
            term: 1 + log((1 + len(counts)) / (1 + count))
            for term, count in frequencies.items()
        }
        self._lexical = [_weighted_terms(passage, self._idf) for passage in counts]
        self._page_by_id = {
            evidence.id: SectionPage(evidence, page, len(pages))
            for pages in corpus.sections.values()
            for page, evidence in enumerate(pages, start=1)
        }

    @classmethod
    def build(cls, corpus: Corpus, embed: Embed | None = None) -> "RetrievalIndex":
        if embed is None:
            return cls(corpus, None, None)
        batches = []
        try:
            for start in range(0, len(corpus.chunks), _BATCH_SIZE):
                texts = [
                    f"{evidence.filename}\n{evidence.section}\n{evidence.text}"
                    for evidence in corpus.chunks[start : start + _BATCH_SIZE]
                ]
                batches.append(_normalized(embed(texts), len(texts)))
        except ProviderError:
            logger.warning(
                "Embedding service unavailable at startup. Using local text search."
            )
            return cls(corpus, None, None)
        matrix = np.concatenate(batches)
        matrix.setflags(write=False)
        return cls(corpus, embed, matrix)

    def search(
        self,
        query: str,
        *,
        drug: str | None = None,
        source_type: str | None = None,
        section: str | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query or not 1 <= top_k <= 10:
            raise ValueError("Search needs a query and top_k between 1 and 10.")
        drug = drug.strip().casefold() if drug else None
        source_type = source_type.strip().casefold() if source_type else None
        section = section.strip().casefold() if section else None
        candidates = [
            i
            for i, evidence in enumerate(self._corpus.chunks)
            if (not drug or drug == evidence.drug)
            and (not source_type or source_type == evidence.source_type)
            and (not section or section in evidence.section.casefold())
        ]
        if not candidates:
            return []

        scores = self._scores(query, candidates)
        hits = []
        for position in np.argsort(-scores, kind="stable")[:top_k]:
            if not np.isfinite(scores[position]):
                break
            evidence = self._corpus.chunks[candidates[position]]
            page = self._page_by_id[evidence.id]
            hits.append(
                SearchHit(
                    evidence.preview(),
                    float(scores[position]),
                    page.page,
                    page.total_pages,
                )
            )
        return hits

    def _scores(self, query: str, candidates: list[int]) -> np.ndarray:
        if self._embed is not None and self._matrix is not None:
            try:
                vector = _normalized(self._embed([query]), 1)[0]
            except ProviderError:
                logger.warning(
                    "Query embedding unavailable. Using local text search for this query."
                )
            else:
                if vector.size != self._matrix.shape[1]:
                    raise ValueError(
                        "The query and corpus embedding dimensions differ."
                    )
                return self._matrix[candidates] @ vector

        terms = _weighted_terms(_terms(query), self._idf)
        scores = np.asarray(
            [
                sum(
                    weight * self._lexical[i].get(term, 0.0)
                    for term, weight in terms.items()
                )
                for i in candidates
            ],
            dtype=float,
        )
        scores[scores == 0] = -np.inf
        return scores

    def read_section(self, section_id: str, page: int = 1) -> SectionPage:
        pages = self._corpus.sections[section_id]
        if not 1 <= page <= len(pages):
            raise ValueError(f"Page must be between 1 and {len(pages)}.")
        return SectionPage(pages[page - 1], page, len(pages))


def _terms(text: str) -> Counter[str]:
    return Counter(
        word for word in _WORD.findall(text.casefold()) if word not in _STOP_WORDS
    )


def _weighted_terms(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weights = {
        term: (1 + log(count)) * idf[term]
        for term, count in counts.items()
        if term in idf
    }
    norm = sqrt(sum(weight * weight for weight in weights.values()))
    return {term: weight / norm for term, weight in weights.items()} if norm else {}


def _normalized(vectors: list[list[float]], count: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0:
        raise ValueError("The embedding provider returned an unexpected shape.")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or (lengths == 0).any():
        raise ValueError("The embedding provider returned an invalid vector.")
    return matrix / lengths
