const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const submitButton = form.querySelector("button[type=submit]");
const examples = document.querySelectorAll("[data-question]");
const answerHeading = document.querySelector("#answer-heading");
const answerPanel = document.querySelector("#answer-panel");
const answerStatus = document.querySelector("#answer-status");
const requestStatus = document.querySelector("#request-status");
const waitAnnouncement = document.querySelector("#wait-announcement");
const requestError = document.querySelector("#request-error");
const emptyAnswer = document.querySelector("#empty-answer");
const answerLoading = document.querySelector("#answer-loading");
const waitTime = document.querySelector("#wait-time");
const waitArt = new WaitArt(
  document.querySelector("#wait-art"),
  document.querySelector("#toggle-motion"),
);
const resultPanel = document.querySelector("#answer-result");
const answerText = document.querySelector("#answer-text");
const answerDisclaimer = document.querySelector("#answer-disclaimer");
const evidencePanel = document.querySelector("#evidence-panel");
const sourceList = document.querySelector("#source-list");
const planList = document.querySelector("#plan-list");
const traceList = document.querySelector("#trace-list");
const reviewOutcome = document.querySelector("#review-outcome");
const reviewList = document.querySelector("#review-list");
let waitTimer;

const answerLabels = new Map([
  ["partial", "Partial answer"],
  ["insufficient_evidence", "Insufficient evidence"],
]);

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setLoading(loading) {
  questionInput.readOnly = loading;
  submitButton.disabled = loading;
  submitButton.textContent = loading ? "Researching…" : "Ask question";
  for (const button of examples) button.disabled = loading;
  answerLoading.hidden = !loading;
  if (!loading) {
    clearInterval(waitTimer);
    waitArt.stop();
    waitAnnouncement.textContent = "";
    return;
  }

  const started = performance.now();
  const status = "Researching your question…";
  requestStatus.textContent = status;
  waitAnnouncement.textContent = status;
  waitTime.textContent = "0:00";
  waitArt.start();
  waitTimer = setInterval(() => {
    const elapsed = Math.floor((performance.now() - started) / 1000);
    const seconds = String(elapsed % 60).padStart(2, "0");
    waitTime.textContent = `${Math.floor(elapsed / 60)}:${seconds}`;
    if (elapsed >= 15) requestStatus.textContent = "Still researching your question…";
  }, 250);
}

function renderSources(citations) {
  const sourcesById = new Map();
  sourceList.replaceChildren();

  for (const source of citations) {
    const id = Number(source.id);
    if (!Number.isSafeInteger(id) || id < 1 || sourcesById.has(id)) continue;

    const disclosure = element("details", undefined, "source");
    disclosure.id = `source-${id}`;
    const summary = element("summary", `[${id}] ${source.filename}`);
    const body = element("div", undefined, "source-body");
    const lineRange = source.start_line === source.end_line
      ? `line ${source.start_line}`
      : `lines ${source.start_line}–${source.end_line}`;
    body.append(
      element("p", `${source.section} · ${lineRange}`, "source-meta"),
      element("blockquote", source.excerpt, "source-excerpt"),
    );

    if (typeof source.source_url === "string") {
      try {
        const url = new URL(source.source_url);
        if (url.protocol === "https:" || url.protocol === "http:") {
          const link = element("a", "Open original source");
          link.href = url.href;
          body.append(link);
        }
      } catch {
        // The passage remains available when the source URL is malformed.
      }
    }

    disclosure.append(summary, body);
    sourceList.append(disclosure);
    sourcesById.set(id, { disclosure, summary });
  }

  evidencePanel.hidden = sourcesById.size === 0;
  return sourcesById;
}

function renderAnswerText(answer, sourcesById) {
  answerText.replaceChildren();
  let position = 0;

  for (const marker of answer.matchAll(/\[(\d+)\]/g)) {
    const id = Number(marker[1]);
    const source = sourcesById.get(id);
    if (!source) continue;

    answerText.append(document.createTextNode(answer.slice(position, marker.index)));
    const link = element("a", marker[0], "citation");
    link.href = `#${source.disclosure.id}`;
    link.setAttribute("aria-label", `Show source ${id}`);
    link.addEventListener("click", () => {
      source.disclosure.open = true;
      source.summary.focus({ preventScroll: true });
    });
    answerText.append(link);
    position = marker.index + marker[0].length;
  }

  answerText.append(document.createTextNode(answer.slice(position)));
}

function renderResearchDetails(result) {
  const plan = result.plan?.length ? result.plan : ["No research steps recorded."];
  planList.replaceChildren(...plan.map((step) => element("li", step)));

  traceList.replaceChildren();
  for (const event of result.trace ?? []) {
    const item = element("li");
    const duration = Number.isFinite(event.elapsed_ms) ? ` · ${Math.round(event.elapsed_ms)} ms` : "";
    item.append(element("strong", `${event.tool}${duration}`));
    if (event.arguments && Object.keys(event.arguments).length) {
      item.append(element("pre", JSON.stringify(event.arguments, null, 2)));
    }
    const evidenceIds = event.evidence_ids ?? [];
    const evidenceText = evidenceIds.length
      ? `Evidence IDs returned: ${evidenceIds.join(", ")}`
      : "No evidence IDs returned.";
    item.append(element("span", evidenceText, "trace-meta"));
    traceList.append(item);
  }
  if (!traceList.childElementCount) traceList.append(element("li", "No tool calls recorded."));

  const review = result.review;
  reviewOutcome.textContent = review?.outcome
    ? review.outcome.replaceAll("_", " ").replace(/^./, (first) => first.toUpperCase())
    : "No review outcome recorded.";
  reviewList.replaceChildren(...(review?.findings ?? []).map((finding) => element("li", finding)));
  reviewList.hidden = !reviewList.childElementCount;
}

function renderResult(result) {
  answerStatus.textContent = answerLabels.get(result.status) ?? "";
  answerStatus.hidden = !answerStatus.textContent;
  const disclaimer = typeof result.disclaimer === "string" ? result.disclaimer.trim() : "";
  const answer = result.answer.trimEnd();
  const hasSeparateDisclaimer = disclaimer && answer.endsWith(disclaimer);
  const text = hasSeparateDisclaimer ? answer.slice(0, -disclaimer.length).trimEnd() : answer;
  answerDisclaimer.textContent = hasSeparateDisclaimer ? disclaimer : "";
  answerDisclaimer.hidden = !hasSeparateDisclaimer;
  renderAnswerText(text, renderSources(result.citations ?? []));
  renderResearchDetails(result);
  resultPanel.hidden = false;
  answerHeading.focus();
}

function errorMessage(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "Check the question and try again.";
  if (typeof detail?.message === "string") return detail.message;
  return "The request failed. Try again in a moment.";
}

questionInput.addEventListener("input", () => questionInput.setCustomValidity(""));

for (const button of examples) {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.setCustomValidity("");
    questionInput.focus();
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitButton.disabled) return;
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.setCustomValidity("Enter a research question.");
    questionInput.reportValidity();
    return;
  }

  requestError.hidden = true;
  emptyAnswer.hidden = true;
  resultPanel.hidden = true;
  answerStatus.hidden = true;
  setLoading(true);
  if (answerPanel.getBoundingClientRect().top > window.innerHeight * 0.75) {
    answerPanel.scrollIntoView({ block: "start" });
  }

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await response.json().catch(() => null);
    if (!response.ok) throw new Error(errorMessage(result?.detail));
    if (typeof result?.answer !== "string") throw new Error("The response was incomplete. Try again.");

    renderResult(result);
  } catch (error) {
    requestError.textContent = error instanceof TypeError
      ? "Could not complete the request. Try again."
      : error.message;
    requestError.hidden = false;
  } finally {
    setLoading(false);
  }
});
