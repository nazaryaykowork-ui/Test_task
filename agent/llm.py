"""LLM gateway: every model call goes through here. Per-request timeouts within the turn
budget, one retry on 429/5xx, then the fallback model, which has its own free-tier quota."""

import math
import time
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from agent.config import Config
from agent.resilience import ServiceUnavailable, TurnBudget, with_backoff

T = TypeVar("T", bound=BaseModel)

ATTEMPTS_PER_MODEL = 2  # with the fallback, at most 4 requests per LLM step


class NoAnswer(Exception):
    """The model returned no text, e.g. its safety filters blocked the prompt or the reply."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"model returned no answer: {reason}")
        self.reason = reason


def is_transient(e: Exception) -> bool:
    if isinstance(e, errors.APIError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, httpx.TimeoutException | httpx.NetworkError)


def is_daily_quota(e: Exception) -> bool:
    return isinstance(e, errors.APIError) and e.code == 429 and "PerDay" in str(e.details)


def is_timeout(e: Exception) -> bool:
    # 504 DEADLINE_EXCEEDED is the server-side timeout (seen live after ~30 s).
    return isinstance(e, httpx.TimeoutException) or (isinstance(e, errors.APIError) and e.code == 504)


def worth_retrying_same_model(e: Exception) -> bool:
    """A timeout or a used-up daily quota will not clear in seconds: go to the fallback."""
    return is_transient(e) and not is_daily_quota(e) and not is_timeout(e)


class LLM:
    def __init__(
        self,
        cfg: Config,
        client: genai.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_call: Callable[..., None] = lambda **_: None,
        budget: TurnBudget | None = None,
    ) -> None:
        # The client-level timeout also covers direct client use (scripts/smoke_test.py).
        self.client = client or genai.Client(
            api_key=cfg.gemini_api_key,
            http_options=types.HttpOptions(timeout=cfg.llm_timeout_s * 1000),
        )
        self.models = (cfg.model, cfg.fallback_model)
        self.timeout_s = cfg.llm_timeout_s
        self.budget = budget or TurnBudget(math.inf)
        self.sleep = sleep
        self.on_call = on_call  # every request, failed ones included, is reported to the tracer

    def generate(self, prompt: str) -> str:
        return self._call(prompt).text

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Temperature 0: routing, plans and SQL should not vary between runs of a question.
        Output that does not match the schema is asked for once more."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=0.0
        )
        try:
            return schema.model_validate_json(self._call(prompt, config).text)
        except ValidationError:
            return schema.model_validate_json(self._call(prompt, config).text)

    def _call(self, prompt: str, config: types.GenerateContentConfig | None = None):
        for model in self.models:
            try:
                return with_backoff(
                    partial(self._request, model, prompt, config),
                    worth_retrying_same_model,
                    ATTEMPTS_PER_MODEL,
                    self.sleep,
                )
            except Exception as e:
                if not is_transient(e):
                    raise
                last_error = e
        raise ServiceUnavailable("Gemini", last_error)

    def _request(self, model: str, prompt: str, config: types.GenerateContentConfig | None):
        # Never wait longer than the turn has left. The SDK takes milliseconds.
        self.budget.check("Gemini")
        timeout_ms = int(min(self.timeout_s, self.budget.remaining()) * 1000)
        config = (config or types.GenerateContentConfig()).model_copy(
            update={"http_options": types.HttpOptions(timeout=timeout_ms)}
        )
        try:
            response = self.client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as e:
            code = getattr(e, "code", None)
            self.on_call(model=model, error=f"{type(e).__name__}_{code}" if code else type(e).__name__)
            raise
        usage = response.usage_metadata
        self.on_call(
            model=model,
            tokens_in=(usage and usage.prompt_token_count) or 0,
            tokens_out=(usage and usage.candidates_token_count) or 0,
        )
        if not response.text:
            raise NoAnswer(_no_text_reason(response))
        return response


def _no_text_reason(response) -> str:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback and feedback.block_reason:
        return f"prompt blocked ({getattr(feedback.block_reason, 'value', feedback.block_reason)})"
    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
        if reason and reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            return f"response stopped ({reason})"
    return "empty response"
