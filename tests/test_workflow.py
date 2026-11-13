import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import RateLimitError
from openai.types.chat import ChatCompletionMessage

from app.agent import DISCLAIMER, ResearchAgent
from app.corpus import Corpus
from app.llm import ProviderError, RequestTimeout
from app.main import create_app
from app.retrieval import RetrievalIndex
from app.schemas import Critique, Draft, Plan


def _tool_reply(name, arguments):
    return ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
    )


class ScriptedModel:
    """Only the model is faked; graph, tools, index, source text and API are real."""

    def __init__(self, *, repair=False, always_reject=False):
        self.repair = repair
        self.always_reject = always_reject
        self.reviewed = []

    def research(self, messages, tools, deadline):
        payload = json.loads(messages[1]["content"])
        if "review" in payload:
            return ChatCompletionMessage(
                role="assistant", content="Use the evidence already read."
            )
        results = [
            json.loads(item["content"]) for item in messages if item["role"] == "tool"
        ]
        if not results:
            return _tool_reply(
                "search_corpus",
                {
                    "query": payload["question"],
                    "section": "contraindications",
                    "top_k": 1,
                    "drug": payload["question"].split()[0].lower(),
                },
            )
        if len(results) == 1 and results[0].get("hits"):
            hit = results[0]["hits"][0]
            return _tool_reply(
                "read_section", {"section_id": hit["section_id"], "page": hit["page"]}
            )
        return ChatCompletionMessage(role="assistant", content="Research complete.")

    def structured(self, schema, messages, deadline):
        payload = json.loads(messages[1]["content"])
        if schema is Plan:
            return schema(subquestions=["Inspect the requested label section."])
        if schema is Draft:
            evidence = payload["evidence"][0]
            claims = [
                {
                    "text": "The source lists hypersensitivity among its contraindications.",
                    "evidence_ids": [evidence["evidence_id"]],
                }
            ]
            if self.repair and "review" not in payload:
                claims.extend(
                    [
                        {
                            "text": "No monitoring is needed.",
                            "evidence_ids": [evidence["evidence_id"]],
                        },
                        {
                            "text": "The risk is exactly 27%.",
                            "evidence_ids": ["not-returned"],
                        },
                    ]
                )
            return schema.model_validate({"claims": claims})
        assert schema is Critique
        claims = payload["claims"]
        self.reviewed.append(len(claims))
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "claim_id": claim["claim_id"],
                        # The deliberately forged citation gets a false pass from the fake critic.
                        "verdict": "unsupported"
                        if self.always_reject or claim["claim_id"] == 1
                        else "supported",
                        "reason": "Scripted decision.",
                    }
                    for claim in claims
                ],
                "unanswered": [],
            }
        )


def _index():
    corpus = Corpus.load(Path("corpus"))
    return RetrievalIndex.build(corpus, lambda texts: [[1.0, 0.0] for _ in texts])


def _app_with(agent):
    @asynccontextmanager
    async def factory():
        yield agent

    return create_app(factory)


def test_api_research_repairs_and_reviews_before_returning():
    model = ScriptedModel(repair=True)
    with TestClient(_app_with(ResearchAgent(model, _index()))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "/static/app.js" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.post("/ask", json={"question": " "}).status_code == 422
        response = client.post("/ask", json={"question": "Warfarin contraindications?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["review"]["outcome"] == "repaired"
    assert model.reviewed == [3, 1]
    assert [item["tool"] for item in body["trace"]] == ["search_corpus", "read_section"]
    assert "27%" not in body["answer"] and "No monitoring" not in body["answer"]
    assert "[1]" in body["answer"] and body["answer"].endswith(DISCLAIMER)
    (citation,) = body["citations"]
    assert citation["filename"] == "dailymed-warfarin.md"
    lines = (
        (Path("corpus") / citation["filename"]).read_text(encoding="utf-8").splitlines()
    )
    assert citation["excerpt"] == "\n".join(
        lines[citation["start_line"] - 1 : citation["end_line"]]
    )


def test_unresolved_claims_are_withheld():
    model = ScriptedModel(always_reject=True)
    result = ResearchAgent(model, _index()).ask(
        "Warfarin contraindications?", "request-a"
    )
    assert model.reviewed == [1, 1]
    assert result.status == "insufficient_evidence"
    assert result.review.outcome == "limited"
    assert result.citations == []
    assert "hypersensitivity" not in result.answer
    assert result.answer.endswith(DISCLAIMER)


@pytest.mark.parametrize("valid_draft", [True, False])
def test_research_draft_saves_a_model_call_and_still_gets_reviewed(valid_draft):
    class DraftingModel(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.calls = []

        def research(self, messages, tools, deadline):
            self.calls.append("research")
            reply = super().research(messages, tools, deadline)
            if reply.tool_calls or not valid_draft:
                return reply
            passage = json.loads(messages[-1]["content"])["evidence"]
            reply.content = json.dumps(
                {
                    "claims": [
                        {
                            "text": "The label lists hypersensitivity as a contraindication.",
                            "evidence_ids": [passage["evidence_id"]],
                        }
                    ]
                }
            )
            return reply

        def structured(self, schema, messages, deadline):
            self.calls.append(schema.__name__)
            return super().structured(schema, messages, deadline)

    model = DraftingModel()
    result = ResearchAgent(model, _index()).ask(
        "Apixaban contraindications?", "latency-test"
    )
    expected = ["Plan", "research", "research", "research"]
    if not valid_draft:
        expected.append("Draft")
    assert model.calls == [*expected, "Critique"]
    assert result.status == "answered"
    assert result.citations[0].filename == "dailymed-apixaban.md"
    assert result.answer.endswith(DISCLAIMER)


def test_partial_answer_identifies_unresolved_question_topic():
    class PartialModel(ScriptedModel):
        def structured(self, schema, messages, deadline):
            result = super().structured(schema, messages, deadline)
            if schema is Critique:
                result.unanswered = [
                    "semaglutide in pregnancy",
                    "invented clinical claim",
                ]
            return result

    result = ResearchAgent(PartialModel(), _index()).ask(
        "Apixaban contraindications and semaglutide in pregnancy?", "request-partial"
    )
    assert result.status == "partial"
    assert result.citations[0].filename == "dailymed-apixaban.md"
    assert "semaglutide in pregnancy" in result.answer
    assert "invented clinical claim" not in result.answer


def test_follow_up_research_revises_with_the_original_draft_and_evidence():
    class FollowUpModel(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.evidence_id = None

        def research(self, messages, tools, deadline):
            if "review" in json.loads(messages[1]["content"]):
                return ChatCompletionMessage(
                    role="assistant",
                    content=json.dumps(
                        {
                            "claims": [
                                {
                                    "text": "Follow-up draft omitted the original claim.",
                                    "evidence_ids": [self.evidence_id],
                                }
                            ]
                        }
                    ),
                )
            return super().research(messages, tools, deadline)

        def structured(self, schema, messages, deadline):
            result = super().structured(schema, messages, deadline)
            payload = json.loads(messages[1]["content"])
            if schema is Draft:
                self.evidence_id = payload["evidence"][0]["evidence_id"]
                if "review" in payload:
                    assert payload["previous_draft"]["claims"][0]["evidence_ids"] == [
                        self.evidence_id
                    ]
            if schema is Critique and len(self.reviewed) == 1:
                result.unanswered = ["bleeding"]
            return result

    result = ResearchAgent(FollowUpModel(), _index()).ask(
        "Apixaban contraindications and bleeding?", "request-follow-up"
    )
    assert "hypersensitivity" in result.answer
    assert "omitted the original claim" not in result.answer
    assert result.review.outcome == "repaired"


def test_concurrent_requests_keep_evidence_separate():
    agent = ResearchAgent(ScriptedModel(), _index())
    questions = ("Warfarin contraindications?", "Apixaban contraindications?")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda q: agent.ask(q, q), questions)
    assert [item.filename for item in first.citations] == ["dailymed-warfarin.md"]
    assert [item.filename for item in second.citations] == ["dailymed-apixaban.md"]


def test_compound_question_cites_both_sources():
    class ComparisonModel(ScriptedModel):
        def research(self, messages, tools, deadline):
            results = [
                json.loads(item["content"])
                for item in messages
                if item["role"] == "tool"
            ]
            if not results:
                calls = [
                    _tool_reply(
                        "search_corpus",
                        {
                            "query": f"{drug} contraindications",
                            "drug": drug,
                            "section": "contraindications",
                            "source_type": "dailymed",
                            "top_k": 1,
                        },
                    ).tool_calls[0]
                    for drug in ("warfarin", "apixaban")
                ]
            elif len(results) == 2:
                calls = [
                    _tool_reply(
                        "read_section",
                        {
                            "section_id": item["hits"][0]["section_id"],
                        },
                    ).tool_calls[0]
                    for item in results
                ]
            else:
                return ChatCompletionMessage(role="assistant", content="Done.")
            for i, call in enumerate(calls):
                call.id = f"call_{len(results)}_{i}"
            return ChatCompletionMessage(role="assistant", tool_calls=calls)

        def structured(self, schema, messages, deadline):
            if schema is Draft:
                evidence = json.loads(messages[1]["content"])["evidence"]
                return schema.model_validate(
                    {
                        "claims": [
                            {
                                "text": "Both labels list drug-specific hypersensitivity.",
                                "evidence_ids": [
                                    item["evidence_id"] for item in evidence
                                ],
                            }
                        ],
                    }
                )
            return super().structured(schema, messages, deadline)

    result = ResearchAgent(ComparisonModel(), _index()).ask(
        "Compare warfarin and apixaban contraindications.", "request-c"
    )
    assert result.status == "answered"
    assert {item.filename for item in result.citations} == {
        "dailymed-warfarin.md",
        "dailymed-apixaban.md",
    }
    assert "[1] [2]" in result.answer
    assert [event.tool for event in result.trace] == [
        "search_corpus",
        "search_corpus",
        "read_section",
        "read_section",
    ]


def test_missing_corpus_evidence_abstains_without_drafting_claims():
    model = ScriptedModel()
    result = ResearchAgent(model, _index()).ask(
        "Missingdrug contraindications?", "request-b"
    )
    assert result.status == "insufficient_evidence"
    assert model.reviewed == [0, 0]
    assert result.citations == []
    assert result.trace[0].tool == "search_corpus"


def test_upstream_failures_have_actionable_status():
    class BrokenAgent:
        def ask(self, question, request_id):
            if question == "timeout":
                raise RequestTimeout("The research request timed out.")
            raise ProviderError("private upstream details")

    with TestClient(_app_with(BrokenAgent())) as client:
        failed = client.post("/ask", json={"question": "test"})
        timed_out = client.post("/ask", json={"question": "timeout"})
    assert failed.status_code == 502 and "private" not in failed.text
    assert timed_out.status_code == 504


def test_app_starts_when_embedding_api_is_rate_limited(monkeypatch):
    from app import llm
    from app.config import Settings

    def rate_limit(*args, **kwargs):
        response = httpx.Response(
            429, request=httpx.Request("POST", "https://provider.test/v1/embeddings")
        )
        raise RateLimitError("rate limit", response=response, body=None)

    class FakeProvider:
        def __init__(self, **kwargs):
            from types import SimpleNamespace

            self.embeddings = SimpleNamespace(create=rate_limit)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=rate_limit))

        def close(self):
            pass

    monkeypatch.setattr(llm, "OpenAI", FakeProvider)
    monkeypatch.setattr(
        Settings, "from_env", lambda: Settings("unused", None, "chat", "embedding")
    )
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/").status_code == 200
        response = client.post("/ask", json={"question": "Apixaban contraindications?"})
        assert response.status_code == 429
        assert "chat model is rate-limited" in response.json()["detail"]
