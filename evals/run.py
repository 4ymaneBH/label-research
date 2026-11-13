"""Run the opt-in live evaluation against a running assistant API."""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR.parent / "corpus"


def check_response(case, response, corpus):
    errors = []
    if not isinstance(response, dict):
        return ["Expected a JSON object"]

    if response.get("status") != case["status"]:
        errors.append(
            f"Expected status {case['status']}, got {response.get('status')!r}"
        )

    answer = response.get("answer")
    disclaimer = response.get("disclaimer")
    if (
        not isinstance(answer, str)
        or not isinstance(disclaimer, str)
        or "not medical advice" not in disclaimer.lower()
        or disclaimer not in answer
    ):
        errors.append("Missing medical disclaimer in the answer")

    review = response.get("review")
    if (
        not isinstance(review, dict)
        or review.get("outcome") not in ("passed", "repaired", "limited")
        or not isinstance(review.get("findings"), list)
    ):
        errors.append("Missing completed review")

    trace = response.get("trace")
    if not isinstance(trace, list) or any(
        not isinstance(event, dict) for event in trace
    ):
        errors.append("Missing tool trace")
    elif case["sources"]:
        successful_tools = {
            event.get("tool") for event in trace if event.get("evidence_ids")
        }
        for tool in ("search_corpus", "read_section"):
            if tool not in successful_tools:
                errors.append(f"No evidence returned by {tool}")

    citations = response.get("citations")
    if not isinstance(citations, list) or any(
        not isinstance(item, dict) for item in citations
    ):
        return [*errors, "Missing citation list"]

    filenames = set()
    ids = set()
    for citation in citations:
        filename = citation.get("filename")
        citation_id = citation.get("id")
        start = citation.get("start_line")
        end = citation.get("end_line")
        if type(citation_id) is not int or citation_id < 1 or citation_id in ids:
            errors.append(f"Invalid or duplicate citation ID {citation_id!r}")
        else:
            ids.add(citation_id)
        if not isinstance(filename, str) or filename not in corpus:
            errors.append(f"Unknown citation source {filename!r}")
            continue
        filenames.add(filename)
        lines = corpus[filename]
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(lines)
        ):
            errors.append(f"Invalid citation range for {filename}")
            continue
        if citation.get("excerpt") != "\n".join(lines[start - 1 : end]):
            errors.append(
                f"Citation excerpt does not match {filename}, lines {start} to {end}"
            )

    for filename in sorted(set(case["sources"]) - filenames):
        errors.append(f"Missing expected source {filename}")
    if case["status"] == "insufficient_evidence" and citations:
        errors.append("Out-of-corpus answer should have no citations")
    if isinstance(answer, str):
        markers = {int(marker) for marker in re.findall(r"\[(\d+)\]", answer)}
        if markers != ids:
            errors.append("Inline citation markers and citation records disagree")
    return errors


def evaluate(case, endpoint, timeout, corpus):
    request = Request(
        endpoint,
        data=json.dumps({"question": case["question"]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    response = None
    http_status = None
    try:
        with urlopen(request, timeout=timeout) as reply:
            http_status = reply.status
            response = json.load(reply)
        errors = check_response(case, response, corpus)
        if http_status != 200:
            errors.append(f"Expected HTTP 200, got {http_status}")
    except HTTPError as exc:
        http_status = exc.code
        with exc:
            response = exc.read(4096).decode("utf-8", errors="replace")
        errors = [f"HTTP {exc.code}"]
    except (URLError, TimeoutError, ValueError, UnicodeError) as exc:
        errors = [f"Request or response error: {exc}"]

    return {
        "id": case["id"],
        "question": case["question"],
        "http_status": http_status,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "errors": errors,
        "semantic_rubric": case["rubric"],
        "response": response,
    }


def main():
    cases = json.loads((EVAL_DIR / "cases.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--group", choices=("public", "extra", "all"), default="public"
    )
    selection.add_argument("--case", choices=[case["id"] for case in cases])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--timeout", type=float, default=150, help="Per-request timeout in seconds"
    )
    parser.add_argument(
        "--output", type=Path, help="Write raw answers and checks to this JSON file"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    selected = [
        case
        for case in cases
        if (
            case["id"] == args.case
            if args.case
            else args.group in (case["group"], "all")
        )
    ]
    corpus = {
        path.name: path.read_text(encoding="utf-8-sig").split("\n")
        for path in CORPUS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in (".md", ".txt")
    }
    endpoint = args.base_url.rstrip("/") + "/ask"
    results = []
    for case in selected:
        result = evaluate(case, endpoint, args.timeout, corpus)
        results.append(result)
        status = "FAIL" if result["errors"] else "PASS"
        print(f"{status} {case['id']} ({result['elapsed_seconds']:.2f}s)", flush=True)
        for error in result["errors"]:
            print(f"  {error}", flush=True)

    if args.output:
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote raw responses and rubrics to {args.output}")
    passed = sum(not result["errors"] for result in results)
    print(
        f"{passed}/{len(results)} passed automated checks. Review answer meaning with the case rubrics."
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
