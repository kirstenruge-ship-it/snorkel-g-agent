from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from typing import Any

import certifi
import httpx

from snorkel_g_agent.config import route_headers
from snorkel_g_agent.schema import ModelMessage, ModelResponse, RouteConfig, Usage


class ProviderError(RuntimeError):
    pass


def _decode_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return {"command": stripped}
        if isinstance(decoded, dict):
            return decoded
        return {"command": stripped}
    return {}


def _content_from_choice(choice: dict[str, Any]) -> str:
    message = choice.get("message")
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            call = tool_calls[0]
            if isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        return json.dumps(
                            {
                                "name": name,
                                "arguments": _decode_tool_arguments(
                                    function.get("arguments")
                                ),
                            }
                        )
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _chat_completions_url(route: RouteConfig) -> str:
    base_url = route.base_url.rstrip("/")
    if route.provider == "modal" and not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url + "/chat/completions"


class OpenAICompatibleProvider:
    def __init__(
        self,
        route: RouteConfig,
        request_timeout_seconds: int,
        request_retries: int = 3,
        request_retry_base_seconds: float = 2.0,
        request_retry_max_seconds: float = 30.0,
        max_model_tokens: int = 4096,
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.route = route
        self.request_timeout_seconds = request_timeout_seconds
        self.request_retries = request_retries
        self.request_retry_base_seconds = request_retry_base_seconds
        self.request_retry_max_seconds = request_retry_max_seconds
        self.max_model_tokens = max_model_tokens
        self.on_retry = on_retry

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        payload = {
            "model": self.route.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
            "max_tokens": self.max_model_tokens,
        }
        url = _chat_completions_url(self.route)
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
            sleep_seconds = min(
                self.request_retry_base_seconds * (2**attempt),
                self.request_retry_max_seconds,
            ) + random.uniform(0, 1.5)
            if self.on_retry is not None:
                self.on_retry(
                    {
                        "attempt": attempt + 1,
                        "max_attempts": self.request_retries + 1,
                        "sleep_seconds": round(sleep_seconds, 3),
                        "message": str(last_error) if last_error else "provider request failed",
                    }
                )
            await asyncio.sleep(sleep_seconds)
        else:
            raise ProviderError(str(last_error) if last_error else "provider request failed")

        raw: dict[str, Any] = response.json()
        try:
            choice = raw["choices"][0]
            content = _content_from_choice(choice)
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
