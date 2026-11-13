import logging
from time import monotonic
from typing import TypeVar

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class RequestTimeout(ProviderError):
    pass


class ChatRateLimit(ProviderError):
    pass


def remaining_time(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise RequestTimeout("The research request timed out.")
    return remaining


class ModelClient:
    def __init__(self, settings: Settings):
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=30.0,
            max_retries=0,
        )
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
            )
        except APITimeoutError:
            raise RequestTimeout("The embedding service timed out.") from None
        except APIError as exc:
            logger.warning("Embedding request failed: %s", type(exc).__name__)
            raise ProviderError("The embedding service is unavailable.") from None
        return [
            item.embedding for item in sorted(result.data, key=lambda item: item.index)
        ]

    def research(self, messages: list[dict], tools: list[dict], deadline: float):
        return self._chat(messages, deadline, tools=tools, tool_choice="auto")

    def structured(self, schema: type[T], messages: list[dict], deadline: float) -> T:
        messages = list(messages)
        for attempt in range(2):
            output = self._chat(
                messages, deadline, response_format={"type": "json_object"}
            )
            try:
                return schema.model_validate_json(output.content or "")
            except ValidationError:
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "The previous response did not validate. Return only the requested JSON.",
                        }
                    )
        raise ProviderError("The model returned an invalid response. Please try again.")

    def _chat(self, messages: list[dict], deadline: float, **kwargs):
        timeout = min(45.0, remaining_time(deadline))
        try:
            result = self.client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )
        except APITimeoutError:
            raise RequestTimeout("The language model timed out.") from None
        except RateLimitError:
            raise ChatRateLimit(
                "The chat model is rate-limited. Try again shortly or check your provider quota."
            ) from None
        except APIError as exc:
            logger.warning("Chat request failed: %s", type(exc).__name__)
            raise ProviderError("The language model is unavailable.") from None
        if not result.choices:
            raise ProviderError("The language model returned no response.")
        return result.choices[0].message

    def close(self) -> None:
        self.client.close()
