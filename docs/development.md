# Development notes

OpenAI Codex assisted with planning, implementation, and testing.

## Project instructions used with Codex

- Keep the assistant in one process and one Docker container.
- Preserve the root Dockerfile contract and the `GET /health` and `POST /ask` routes.
- Use the bundled corpus for evidence. Keep model-selected tool calls, exact
  source citations and claim review observable.
- Use scripted providers for routine tests. Live evaluation sends corpus
  text and questions to the configured endpoint and is opt-in.
- Keep credentials and the assignment PDF out of new public commits. Never
  print secrets or include them in application logs.
