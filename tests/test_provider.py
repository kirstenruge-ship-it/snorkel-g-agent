from __future__ import annotations

from typing import Any

import httpx
import pytest

from snorkel_g_agent.provider import OpenAICompatibleProvider
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
