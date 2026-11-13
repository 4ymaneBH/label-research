from pathlib import Path

import pytest

from app.corpus import Corpus
from app.llm import ProviderError
from app.retrieval import RetrievalIndex


def test_search_ranks_by_cosine_and_honors_source_filters(tmp_path):
    (tmp_path / "dailymed-alpha.md").write_text(
        "# Alpha\n\n## Warnings\n\nAlpha warning text.\n", encoding="utf-8"
    )
    (tmp_path / "medlineplus-beta.md").write_text(
        "# Beta\n\n## Precautions\n\nBeta precaution text.\n", encoding="utf-8"
    )

    def embed(texts):
        # Alpha's larger vector wins by dot product alone. Cosine should prefer Beta.
        return [[10, 10] if "dailymed-alpha" in text else [1, 0] for text in texts]

    index = RetrievalIndex.build(Corpus.load(tmp_path), embed)
    hits = index.search("warnings", top_k=2)
    assert [hit.evidence.drug for hit in hits] == ["beta", "alpha"]
    assert hits[0].score == pytest.approx(1)
    assert hits[1].score == pytest.approx(2**-0.5)

    filtered = index.search("warnings", source_type="DAILYMED", section="warnings")
    assert [hit.evidence.drug for hit in filtered] == ["alpha"]
    assert index.search("warnings", drug="unavailable") == []


def test_read_opens_exact_source_page_from_search_hit(tmp_path):
    paragraphs = [
        "This opening paragraph gives context. " * 30,
        "A later paragraph includes an exception. " * 38,
        "A very long unsplittable line. " * 120,
    ]
    text = (
        "# Alpha\n\nSource: https://example.test/alpha\n\n## Warnings\n"
        "\n### Exceptions\n\n" + "\n\n".join(paragraphs) + "\n"
    )
    (tmp_path / "dailymed-alpha.md").write_text(text, encoding="utf-8")
    corpus = Corpus.load(tmp_path)
    index = RetrievalIndex.build(corpus, lambda texts: [[1, 0] for _ in texts])
    hits = index.search("exceptions", section="exceptions")
    assert len(hits) >= 2
    assert hits[0].evidence.section == "Alpha > Warnings > Exceptions"
    assert hits[0].evidence.source_url == "https://example.test/alpha"

    for hit in hits:
        page = index.read_section(hit.evidence.section_id, hit.page)
        assert page.total_pages == len(hits)
        assert page.evidence.text.startswith(hit.evidence.text)
        for evidence in (hit.evidence, page.evidence):
            assert evidence.text == "\n".join(
                text.split("\n")[evidence.start_line - 1 : evidence.end_line]
            )
        if page.evidence.text != hit.evidence.text:
            assert page.evidence.id != hit.evidence.id

    assert any(paragraphs[-1] in page.text for page in corpus.chunks)
    with pytest.raises(KeyError):
        index.read_section("unknown")
    with pytest.raises(ValueError):
        index.read_section(hits[0].evidence.section_id, page=len(hits) + 1)


@pytest.mark.parametrize("fails_at_startup", [True, False])
def test_embedding_failure_falls_back_to_filtered_text_search(fails_at_startup):
    corpus = Corpus.load(Path(__file__).resolve().parents[1] / "corpus")
    calls = []

    def embed(texts):
        calls.append(texts)
        if fails_at_startup or len(texts) == 1:
            raise ProviderError("Embedding rate limit")
        return [[1.0, 0.0] for _ in texts]

    index = RetrievalIndex.build(corpus, embed)
    hits = index.search(
        "contraindications hypersensitivity",
        drug="apixaban",
        source_type="dailymed",
        section="contraindications",
    )
    assert hits
    hit = hits[0]
    assert hit.evidence.filename == "dailymed-apixaban.md"
    assert hit.score > 0
    assert index.read_section(hit.evidence.section_id, hit.page).evidence.text
    assert index.search("qzxmissingterm") == []
    if fails_at_startup:
        assert len(calls) == 1


def test_supplied_passages_match_their_source_line_ranges():
    path = Path(__file__).resolve().parents[1] / "corpus"
    corpus = Corpus.load(path)
    assert corpus.document_count == 21
    sources = {
        file.name: file.read_text(encoding="utf-8-sig").split("\n")
        for file in path.glob("*.md")
    }
    for chunk in corpus.chunks:
        assert chunk.text == "\n".join(
            sources[chunk.filename][chunk.start_line - 1 : chunk.end_line]
        )
    assert any(
        chunk.filename == "dailymed-warfarin.md"
        and "CONTRAINDICATIONS" in chunk.section
        for chunk in corpus.chunks
    )
