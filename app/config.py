from dataclasses import dataclass, field
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str | None
    model: str
    embedding_model: str
    corpus_dir: Path = ROOT / "corpus"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        api_key = getenv("OPENAI_API_KEY", "").strip()
        model = getenv("OPENAI_MODEL", "").strip()
        if not api_key or not model:
            raise RuntimeError(
                "Set OPENAI_API_KEY and OPENAI_MODEL before starting the app."
            )
        # The supplied configuration spells the variable OPENAI_BASE_UR.
        base_url = (
            getenv("OPENAI_BASE_URL", "").strip()
            or getenv("OPENAI_BASE_UR", "").strip()
        )
        return cls(
            api_key=api_key,
            base_url=base_url or None,
            model=model,
            embedding_model=getenv("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small",
        )
