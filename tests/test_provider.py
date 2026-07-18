from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from snorkel_g_agent.provider import OpenAICompatibleProvider, ProviderError
from snorkel_g_agent.schema import ModelMessage, RouteConfig


class _CaptureClient:
    payload: dict[str, Any] | None = None
    url: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _CaptureClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        self.__class__.payload = json
        self.__class__.url = url
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )


class _NativeToolCallClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _NativeToolCallClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"pytest -q"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )


class _RejectToolsThenAcceptClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _RejectToolsThenAcceptClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        self.__class__.calls.append(json)
        if len(self.__class__.calls) == 1:
            return httpx.Response(400, text="unknown field: tool_choice")
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )


class _UnauthorizedClient:
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _UnauthorizedClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        self.__class__.calls += 1
        return httpx.Response(401, text="unauthorized")


class _ReadTimeoutThenAcceptClient:
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _ReadTimeoutThenAcceptClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        self.__class__.calls += 1
        if self.__class__.calls == 1:
            raise httpx.ReadTimeout("slow endpoint")
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {},
            },
        )


class _HangingClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _HangingClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        await asyncio.sleep(60)
        raise AssertionError("outer request deadline did not fire")


@pytest.mark.asyncio
async def test_provider_sends_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _CaptureClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
        max_model_tokens=1234,
    )

    await provider.complete([ModelMessage(role="user", content="hello")])

    assert _CaptureClient.payload is not None
    assert _CaptureClient.payload["max_tokens"] == 1234
    assert _CaptureClient.payload["tool_choice"] == "auto"
    assert any(
        tool["function"]["name"] == "exec" for tool in _CaptureClient.payload["tools"]
    )


@pytest.mark.asyncio
async def test_provider_can_disable_native_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _CaptureClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
            use_native_tools=False,
        ),
        request_timeout_seconds=30,
    )

    await provider.complete([ModelMessage(role="user", content="hello")])

    assert _CaptureClient.payload is not None
    assert "tools" not in _CaptureClient.payload
    assert "tool_choice" not in _CaptureClient.payload


@pytest.mark.asyncio
async def test_provider_normalizes_modal_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _CaptureClient)
    monkeypatch.setenv("MODAL_KEY", "key")
    monkeypatch.setenv("MODAL_SECRET", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="modal",
            model="glm-5.2",
            base_url="https://example.modal.direct",
            modal_key_env="MODAL_KEY",
            modal_secret_env="MODAL_SECRET",
        ),
        request_timeout_seconds=30,
    )

    await provider.complete([ModelMessage(role="user", content="hello")])

    assert _CaptureClient.url == "https://example.modal.direct/v1/chat/completions"


@pytest.mark.asyncio
async def test_provider_converts_native_tool_calls_to_action_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _NativeToolCallClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
    )

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    assert response.content == '{"name": "bash", "arguments": {"command": "pytest -q"}}'
    assert response.raw["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_provider_falls_back_when_tool_schema_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _RejectToolsThenAcceptClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _RejectToolsThenAcceptClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    request_log_path = tmp_path / "llm_requests.jsonl"
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
        request_retries=1,
        request_log_path=request_log_path,
    )

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    assert response.content == '{"action":"finish"}'
    assert "tools" in _RejectToolsThenAcceptClient.calls[0]
    assert "tools" not in _RejectToolsThenAcceptClient.calls[1]
    log_text = request_log_path.read_text()
    assert "tool_schema_rejected" in log_text
    assert "FAKE_API_KEY" not in log_text
    assert "secret" not in log_text


@pytest.mark.asyncio
async def test_provider_does_not_retry_nonretryable_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _UnauthorizedClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _UnauthorizedClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
        request_retries=4,
    )

    with pytest.raises(ProviderError) as caught:
        await provider.complete([ModelMessage(role="user", content="hello")])

    assert caught.value.kind == "auth"
    assert caught.value.retryable is False
    assert _UnauthorizedClient.calls == 1


@pytest.mark.asyncio
async def test_provider_retries_only_the_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ReadTimeoutThenAcceptClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _ReadTimeoutThenAcceptClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    retries: list[dict[str, Any]] = []
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
        request_retries=1,
        request_retry_base_seconds=0.1,
        request_retry_max_seconds=0.1,
        on_retry=retries.append,
    )

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    assert response.content == '{"action":"finish"}'
    assert _ReadTimeoutThenAcceptClient.calls == 2
    assert retries[0]["kind"] == "network"


@pytest.mark.asyncio
async def test_provider_hard_deadline_caps_hanging_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _HangingClient)
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    provider = OpenAICompatibleProvider(
        RouteConfig(
            provider="openai-compatible",
            model="glm-5.2",
            base_url="https://example.invalid/v1",
            api_key_env="FAKE_API_KEY",
        ),
        request_timeout_seconds=30,
        request_retries=0,
    )
    deadline = asyncio.get_running_loop().time() + 0.02

    with pytest.raises(ProviderError) as caught:
        await provider.complete(
            [ModelMessage(role="user", content="hello")],
            deadline=deadline,
        )

    assert caught.value.kind == "timeout"
    assert caught.value.retryable is True
