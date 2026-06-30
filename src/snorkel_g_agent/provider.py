from __future__ import annotations

import asyncio
import random
from typing import Any

import certifi
import httpx

from snorkel_g_agent.config import route_headers
from snorkel_g_agent.schema import ModelMessage, ModelResponse, RouteConfig, Usage


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(
        self,
        route: RouteConfig,
        request_timeout_seconds: int,
        request_retries: int = 3,
    ) -> None:
        self.route = route
        self.request_timeout_seconds = request_timeout_seconds
        self.request_retries = request_retries

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        payload = {
            "model": self.route.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
        }
        url = self.route.base_url.rstrip("/") + "/chat/completions"
        timeout = httpx.Timeout(self.request_timeout_seconds)
        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=certifi.where()) as client:
                    response = await client.post(
                        url,
                        headers=route_headers(self.route),
                        json=payload,
                    )
                if response.status_code < 400:
                    break
                body = response.text[:2000]
                message = f"provider returned {response.status_code}: {body}"
                retryable = response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.request_retries:
                    raise ProviderError(message)
                last_error = ProviderError(message)
            except (httpx.HTTPError, ProviderError) as exc:
                last_error = exc
                if attempt >= self.request_retries:
                    raise ProviderError(str(exc)) from exc
            await asyncio.sleep(min(2**attempt, 30) + random.uniform(0, 1.5))
        else:
            raise ProviderError(str(last_error) if last_error else "provider request failed")

        raw: dict[str, Any] = response.json()
        try:
            choice = raw["choices"][0]
            content = choice.get("message", {}).get("content") or choice.get("text") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected provider response shape: {raw}") from exc
        usage = raw.get("usage") or {}
        cached_tokens = None
        prompt_details = usage.get("prompt_tokens_details") or {}
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")
        return ModelResponse(
            content=content,
            model=raw.get("model") or self.route.model,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cached_tokens=cached_tokens,
            ),
            raw=raw,
        )
