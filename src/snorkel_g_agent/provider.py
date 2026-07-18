from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import certifi
import httpx

from snorkel_g_agent.config import route_headers
from snorkel_g_agent.schema import ModelMessage, ModelResponse, RouteConfig, Usage
from snorkel_g_agent.time_utils import utc_now
from snorkel_g_agent.tool_definitions import TOOL_DEFINITIONS


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: str = "unknown",
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable


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


def _provider_error_kind(status_code: int) -> str:
    if status_code in {401, 403}:
        return "auth"
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if status_code == 400:
        return "bad_request"
    if status_code == 413:
        return "context_length"
    if status_code in {500, 502, 503, 504}:
        return "server"
    return "http"


def _looks_like_tool_schema_rejection(status_code: int, body: str) -> bool:
    if status_code not in {400, 404, 422}:
        return False
    lowered = body.lower()
    return any(token in lowered for token in ("tool_choice", "tools", "tool_calls", "function"))


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
        request_log_path: Path | None = None,
    ) -> None:
        self.route = route
        self.request_timeout_seconds = request_timeout_seconds
        self.request_retries = request_retries
        self.request_retry_base_seconds = request_retry_base_seconds
        self.request_retry_max_seconds = request_retry_max_seconds
        self.max_model_tokens = max_model_tokens
        self.on_retry = on_retry
        self.request_log_path = request_log_path

    async def complete(
        self,
        messages: list[ModelMessage],
        *,
        deadline: float | None = None,
    ) -> ModelResponse:
        payload = {
            "model": self.route.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
            "max_tokens": self.max_model_tokens,
        }
        url = _chat_completions_url(self.route)
        last_error: Exception | None = None
        tools_enabled = self.route.use_native_tools
        raw: dict[str, Any] | None = None
        content = ""

        for attempt in range(self.request_retries + 1):
            remaining = self._remaining_seconds(deadline)
            if remaining is not None and remaining <= 0:
                raise ProviderError(
                    "task deadline reached before provider request",
                    kind="task_deadline",
                    retryable=False,
                )
            attempt_timeout = min(
                float(self.request_timeout_seconds),
                remaining if remaining is not None else float(self.request_timeout_seconds),
            )
            timeout = httpx.Timeout(max(attempt_timeout, 0.001))
            request_payload = dict(payload)
            if tools_enabled:
                request_payload["tools"] = TOOL_DEFINITIONS
                request_payload["tool_choice"] = "auto"
            self._log_request(
                {
                    "event": "request",
                    "attempt": attempt + 1,
                    "max_attempts": self.request_retries + 1,
                    "url": url,
                    "route": self._route_log_data(),
                    "native_tools_enabled": tools_enabled,
                    "attempt_timeout_seconds": round(attempt_timeout, 3),
                    "payload": request_payload,
                }
            )
            response: httpx.Response | None = None
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=certifi.where()) as client:
                    response = await asyncio.wait_for(
                        client.post(
                            url,
                            headers=route_headers(self.route),
                            json=request_payload,
                        ),
                        timeout=max(attempt_timeout, 0.001),
                    )
                if response.status_code >= 400:
                    body = response.text[:2000]
                    retryable = response.status_code in {
                        408,
                        409,
                        425,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    kind = _provider_error_kind(response.status_code)
                    self._log_request(
                        {
                            "event": "response_error",
                            "attempt": attempt + 1,
                            "status_code": response.status_code,
                            "kind": kind,
                            "retryable": retryable,
                            "native_tools_enabled": tools_enabled,
                            "body": body,
                        }
                    )
                    if (
                        tools_enabled
                        and self.route.native_tool_fallback
                        and _looks_like_tool_schema_rejection(response.status_code, body)
                    ):
                        tools_enabled = False
                        last_error = ProviderError(
                            "provider rejected native tool schema; retrying without tools",
                            kind="tool_schema_rejected",
                            status_code=response.status_code,
                            retryable=True,
                        )
                        self._log_request(
                            {
                                "event": "tool_schema_fallback",
                                "attempt": attempt + 1,
                                "status_code": response.status_code,
                                "kind": last_error.kind,
                                "native_tools_enabled": tools_enabled,
                                "message": str(last_error),
                            }
                        )
                        self._notify_retry(attempt, last_error, sleep_seconds=0)
                        continue
                    last_error = ProviderError(
                        f"provider returned {response.status_code}: {body}",
                        kind=kind,
                        status_code=response.status_code,
                        retryable=retryable,
                    )
                    if not retryable or attempt >= self.request_retries:
                        raise last_error
                else:
                    try:
                        decoded = response.json()
                        if not isinstance(decoded, dict):
                            raise TypeError("response JSON is not an object")
                        choice = decoded["choices"][0]
                        content = _content_from_choice(choice)
                        if not content.strip():
                            raise ValueError("provider returned an empty assistant message")
                        raw = decoded
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        last_error = ProviderError(
                            f"unexpected provider response shape: {exc}",
                            kind="response_shape",
                            retryable=True,
                        )
                        self._log_request(
                            {
                                "event": "response_error",
                                "attempt": attempt + 1,
                                "status_code": response.status_code,
                                "kind": last_error.kind,
                                "retryable": True,
                                "native_tools_enabled": tools_enabled,
                                "body": response.text[:2000],
                            }
                        )
                        if attempt >= self.request_retries:
                            raise last_error from exc
                    else:
                        self._log_request(
                            {
                                "event": "response",
                                "attempt": attempt + 1,
                                "status_code": response.status_code,
                                "native_tools_enabled": tools_enabled,
                                "body": raw,
                            }
                        )
                        break
            except ProviderError:
                raise
            except (httpx.HTTPError, TimeoutError) as exc:
                kind = "timeout" if isinstance(exc, TimeoutError) else "network"
                last_error = ProviderError(
                    str(exc) or type(exc).__name__,
                    kind=kind,
                    retryable=True,
                )
                self._log_request(
                    {
                        "event": "exception",
                        "attempt": attempt + 1,
                        "kind": kind,
                        "status_code": None,
                        "retryable": True,
                        "native_tools_enabled": tools_enabled,
                        "message": str(exc),
                    }
                )
                if attempt >= self.request_retries:
                    raise last_error from exc

            sleep_seconds = min(
                self.request_retry_base_seconds * (2**attempt),
                self.request_retry_max_seconds,
            ) + random.uniform(0, 1.5)
            remaining = self._remaining_seconds(deadline)
            if remaining is not None:
                sleep_seconds = min(sleep_seconds, max(0.0, remaining))
            self._notify_retry(attempt, last_error, sleep_seconds=sleep_seconds)
            if sleep_seconds <= 0:
                raise ProviderError(
                    "task deadline reached during provider retry",
                    kind="task_deadline",
                    retryable=False,
                )
            await asyncio.sleep(sleep_seconds)
        else:
            raise ProviderError(str(last_error) if last_error else "provider request failed")

        if raw is None:
            raise ProviderError(
                str(last_error) if last_error else "provider produced no response",
                kind="response_shape",
                retryable=True,
            )
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

    @staticmethod
    def _remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - asyncio.get_running_loop().time()

    def _notify_retry(
        self,
        attempt: int,
        error: Exception | None,
        *,
        sleep_seconds: float,
    ) -> None:
        if self.on_retry is None:
            return
        self.on_retry(
            {
                "attempt": attempt + 1,
                "max_attempts": self.request_retries + 1,
                "sleep_seconds": round(sleep_seconds, 3),
                "message": str(error) if error else "provider request failed",
                "kind": getattr(error, "kind", "unknown"),
            }
        )

    def _route_log_data(self) -> dict[str, Any]:
        return {
            "provider": self.route.provider,
            "model": self.route.model,
            "base_url": self.route.base_url,
            "use_native_tools": self.route.use_native_tools,
            "native_tool_fallback": self.route.native_tool_fallback,
        }

    def _log_request(self, record: dict[str, Any]) -> None:
        if self.request_log_path is None:
            return
        self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": utc_now(), **record}
        with self.request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
