# Ten minute assessment demo

Before the presentation, start the container and confirm `/health` returns 200. Run the public evaluation cases. Save the actual responses and check them against the manual rubrics in [cases.json](../evals/cases.json).

```sh
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

From a second terminal, run the four public cases and save the response data.

```sh
python evals/run.py --group public --output evals/report.json
```

The runner checks HTTP status, answer status, citations against the source text, tool trace, review and disclaimer. It records actual elapsed time and keeps each answer beside its rubric for manual review. Use `--group extra` to run additional cases or `--case CASE_ID` to run a single case.

| Time | Show |
| --- | --- |
| 0:00–1:00 | Introduce the assignment and the supplied corpus. Explain why comparison questions need evidence for each drug. |
| 1:00–2:00 | Explain the workflow from question planning through local tools to the critic. Show where citations come from. |
| 2:00–4:00 | Ask the public warfarin and apixaban comparison. Open both citations and show how the answer preserves qualifications. |
| 4:00–5:30 | Ask the warfarin and sertraline interaction question. Open the trace and show the searches and section reads the model chose. |
| 5:30–6:30 | Ask whether the labels give an exact general adult bleeding risk percentage. Show how the answer handles the limits of the evidence. |
| 6:30–8:00 | Walk through the tool dispatcher, citation checks, and critic routing. Explain how the final response uses reviewed claims. |
| 8:00–9:00 | Show actual evaluation results, a manually reviewed example, and the Docker command. |
| 9:00–10:00 | Describe one failure observed during development, the change made to address it, and any remaining limitations. |

Keep a saved API response from a real evaluation run available as a fallback. Tell the audience when you are showing a recorded result.
