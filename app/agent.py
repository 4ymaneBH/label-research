import json
import re
from time import monotonic
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app import prompts
from app.corpus import Evidence
from app.llm import ModelClient, remaining_time
from app.retrieval import RetrievalIndex
from app.schemas import (
    AskResponse,
    Citation,
    Critique,
    Draft,
    Plan,
    ReviewSummary,
    ToolTrace,
)
from app.tools import TOOL_SCHEMAS, describe, run_tool

DISCLAIMER = (
    "This output is for informational research purposes only and is not medical advice."
)
MAX_TOOL_CALLS = 12
MAX_RESEARCH_ROUNDS = 7
REQUEST_SECONDS = 120
INLINE_CITATION = re.compile(r"\[\d+\]")


class State(TypedDict, total=False):
    question: str
    request_id: str
    deadline: float
    plan: Plan
    evidence: dict[str, Evidence]
    trace: list[ToolTrace]
    rounds: int
    repair: int
    research_draft: Draft | None
    draft: Draft
    critique: Critique
    approved: list[int]
    first_review: list[str]
    response: AskResponse


def _messages(prompt: str, payload: dict) -> list[dict]:
    return [
        {"role": "system", "content": prompts.GROUNDING + "\n\n" + prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


class ResearchAgent:
    def __init__(self, model: ModelClient, index: RetrievalIndex):
        self.model = model
        self.index = index
        graph = StateGraph(State)
        graph.add_node("plan", self._plan)
        graph.add_node("research", self._research)
        graph.add_node("draft", self._draft)
        graph.add_node("review", self._review)
        graph.add_node("repair", self._repair)
        graph.add_node("render", self._render)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "draft")
        graph.add_edge("draft", "review")
        graph.add_conditional_edges(
            "review", self._after_review, {"repair": "repair", "render": "render"}
        )
        graph.add_conditional_edges(
            "repair", self._after_repair, {"research": "research", "draft": "draft"}
        )
        graph.add_edge("render", END)
        self.graph = graph.compile()

    def ask(self, question: str, request_id: str) -> AskResponse:
        result = self.graph.invoke(
            {
                "question": question,
                "request_id": request_id,
                "deadline": monotonic() + REQUEST_SECONDS,
                "evidence": {},
                "trace": [],
                "rounds": 0,
                "repair": 0,
            }
        )
        return result["response"]

    def _plan(self, state: State) -> dict:
        return {
            "plan": self.model.structured(
                Plan,
                _messages(prompts.PLAN, {"question": state["question"]}),
                state["deadline"],
            )
        }

    def _research(self, state: State) -> dict:
        evidence = dict(state["evidence"])
        trace = list(state["trace"])
        rounds = state["rounds"]
        research_draft = None
        payload = {"question": state["question"], "plan": state["plan"].subquestions}
        if state["repair"]:
            payload["review"] = state["critique"].model_dump()
            payload["sections_already_read"] = [
                {"filename": item.filename, "section": item.section}
                for item in evidence.values()
            ]
        if state["repair"]:
            task = (
                "Find evidence for the gaps flagged by the critic. Use the research tools "
                "as needed, then stop calling them. The full draft will be revised separately."
            )
        else:
            task = prompts.DRAFT
        messages = _messages(prompts.RESEARCH + "\n\n" + task, payload)
        nudged = False
        # Reserve part of the call budget for the critic's follow-up on the first pass.
        round_limit = (
            MAX_RESEARCH_ROUNDS if state["repair"] else MAX_RESEARCH_ROUNDS - 2
        )
        tool_limit = MAX_TOOL_CALLS if state["repair"] else MAX_TOOL_CALLS - 2
        while rounds < round_limit and len(trace) < tool_limit:
            # Leave time for drafting and review once there is some evidence.
            if evidence and remaining_time(state["deadline"]) < 30:
                break
            reply = self.model.research(messages, TOOL_SCHEMAS, state["deadline"])
            rounds += 1
            calls = reply.tool_calls or []
            if not calls:
                if not nudged and (
                    not trace
                    or (not evidence and any(event.evidence_ids for event in trace))
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": "Choose a research tool. If search found a relevant passage, "
                            "inspect its source section before stopping.",
                        }
                    )
                    nudged = True
                    continue
                if not state["repair"] and evidence and reply.content:
                    try:
                        candidate = Draft.model_validate_json(reply.content)
                    except ValidationError:
                        pass
                    else:
                        if candidate.claims and all(
                            claim.evidence_ids
                            and all(ref in evidence for ref in claim.evidence_ids)
                            for claim in candidate.claims
                        ):
                            research_draft = candidate
                break
            selected = [call for call in calls if call.type == "function"][
                : tool_limit - len(trace)
            ]
            if not selected:
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [call.model_dump() for call in selected],
                }
            )
            for call in selected:
                remaining_time(state["deadline"])
                result, event, passage = run_tool(
                    self.index, call.function.name, call.function.arguments
                )
                trace.append(event)
                if passage is not None:
                    evidence[passage.id] = passage
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return {
            "evidence": evidence,
            "trace": trace,
            "rounds": rounds,
            "research_draft": research_draft,
        }

    def _draft(self, state: State) -> dict:
        if state.get("research_draft") is not None:
            return {"draft": state["research_draft"]}
        if not state["evidence"]:
            return {"draft": Draft(claims=[])}
        payload = {
            "question": state["question"],
            "plan": state["plan"].subquestions,
            "evidence": [describe(item) for item in state["evidence"].values()],
        }
        if state["repair"]:
            payload["previous_draft"] = state["draft"].model_dump()
            payload["review"] = state["critique"].model_dump()
        draft = self.model.structured(
            Draft, _messages(prompts.DRAFT, payload), state["deadline"]
        )
        return {"draft": draft}

    def _review(self, state: State) -> dict:
        payload = {
            "question": state["question"],
            "claims": [
                {"claim_id": idx, **claim.model_dump()}
                for idx, claim in enumerate(state["draft"].claims)
            ],
            "evidence": [describe(item) for item in state["evidence"].values()],
        }
        critique = self.model.structured(
            Critique, _messages(prompts.REVIEW, payload), state["deadline"]
        )
        verdicts: dict[int, list[str]] = {}
        for verdict in critique.verdicts:
            verdicts.setdefault(verdict.claim_id, []).append(verdict.verdict)
        approved = [
            idx
            for idx, claim in enumerate(state["draft"].claims)
            if verdicts.get(idx) == ["supported"]
            and claim.evidence_ids
            and all(ref in state["evidence"] for ref in claim.evidence_ids)
        ]
        return {"critique": critique, "approved": approved}

    @staticmethod
    def _after_review(state: State) -> str:
        has_issues = (
            len(state["approved"]) != len(state["draft"].claims)
            or bool(state["critique"].unanswered)
            or not state["approved"]
        )
        if has_issues and not state["repair"] and monotonic() + 35 < state["deadline"]:
            return "repair"
        return "render"

    @staticmethod
    def _repair(state: State) -> dict:
        flags = []
        withheld = len(state["draft"].claims) - len(state["approved"])
        if withheld:
            flags.append(
                f"The first review flagged {withheld} claim(s) for insufficient support."
            )
        if state["critique"].unanswered or not state["approved"]:
            flags.append("The first review found incomplete coverage.")
        return {"repair": 1, "first_review": flags, "research_draft": None}

    @staticmethod
    def _after_repair(state: State) -> str:
        if not state["evidence"] or state["critique"].unanswered:
            return "research"
        return "draft"

    @staticmethod
    def _render(state: State) -> dict:
        citations = []
        ids: dict[str, int] = {}
        lines = []
        for idx in state["approved"]:
            claim = state["draft"].claims[idx]
            for ref in dict.fromkeys(claim.evidence_ids):
                if ref not in ids:
                    item = state["evidence"][ref]
                    ids[ref] = len(ids) + 1
                    citations.append(
                        Citation(
                            id=ids[ref],
                            filename=item.filename,
                            section=item.section,
                            start_line=item.start_line,
                            end_line=item.end_line,
                            excerpt=item.text,
                            source_url=item.source_url,
                        )
                    )
            markers = " ".join(
                f"[{ids[ref]}]" for ref in dict.fromkeys(claim.evidence_ids)
            )
            lines.append(f"{INLINE_CITATION.sub('', claim.text).strip()} {markers}")

        findings = []
        withheld = len(state["draft"].claims) - len(state["approved"])
        if withheld:
            findings.append(
                f"Withheld {withheld} claim(s) without adequate source support."
            )
        unanswered = state["critique"].unanswered
        question_text = " ".join(state["question"].casefold().split())
        gaps = []
        for item in unanswered:
            topic = " ".join(INLINE_CITATION.sub("", item).split())
            if topic and topic.casefold() in question_text and topic not in gaps:
                gaps.append(topic)
        if unanswered:
            findings.append(
                "The supplied evidence did not resolve every part of the question."
            )
            findings.extend(f"Unresolved question topic: {gap}" for gap in gaps)
        if not lines:
            status = "insufficient_evidence"
            lines = ["I couldn't verify an answer from the supplied corpus."]
            if not findings:
                findings.append("No claim had enough evidence to include.")
        elif findings:
            status = "partial"
            if gaps:
                lines.append(
                    "The supplied evidence did not resolve these topics:\n"
                    + "\n".join(f"- {gap}" for gap in gaps)
                )
            else:
                lines.append(
                    "I couldn't verify the remaining part of the question from the supplied evidence."
                )
        else:
            status = "answered"
            findings.append("Each included claim passed the source review.")
        if state["repair"]:
            findings[:0] = [
                *state["first_review"],
                "The draft was revised and reviewed again.",
            ]

        return {
            "response": AskResponse(
                answer="\n\n".join([*lines, DISCLAIMER]),
                status=status,
                citations=citations,
                plan=state["plan"].subquestions,
                trace=state["trace"],
                review=ReviewSummary(
                    outcome="limited"
                    if status != "answered"
                    else "repaired"
                    if state["repair"]
                    else "passed",
                    findings=findings,
                ),
                disclaimer=DISCLAIMER,
                request_id=state["request_id"],
            )
        }
