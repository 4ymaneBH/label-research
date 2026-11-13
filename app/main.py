import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import ResearchAgent
from app.config import ROOT, Settings
from app.corpus import Corpus
from app.llm import ChatRateLimit, ModelClient, ProviderError, RequestTimeout
from app.retrieval import RetrievalIndex
from app.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)
STATIC = ROOT / "app" / "static"


@asynccontextmanager
async def load_agent():
    settings = Settings.from_env()
    corpus = Corpus.load(settings.corpus_dir)
    logger.info(
        "Indexing %d corpus files (%d passages)",
        corpus.document_count,
        len(corpus.chunks),
    )
    model = ModelClient(settings)
    try:
        yield ResearchAgent(model, RetrievalIndex.build(corpus, model.embed))
    finally:
        model.close()


def create_app(agent_factory: Callable = load_agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with agent_factory() as agent:
            app.state.agent = agent
            yield

    app = FastAPI(title="Pharmacology Research Assistant", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        request_id = uuid4().hex
        try:
            return app.state.agent.ask(request.question, request_id)
        except ChatRateLimit as exc:
            logger.warning("Chat model rate-limited: %s", request_id)
            raise HTTPException(status_code=429, detail=str(exc)) from None
        except RequestTimeout as exc:
            logger.warning("Research request timed out: %s", request_id)
            raise HTTPException(status_code=504, detail=str(exc)) from None
        except ProviderError:
            logger.warning("Provider request failed: %s", request_id)
            raise HTTPException(
                status_code=502,
                detail="The model service could not complete the request. Please try again.",
            ) from None

    return app


app = create_app()
