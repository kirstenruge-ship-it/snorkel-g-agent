from __future__ import annotations

import json
from pathlib import Path

import pytest

from snorkel_g_agent.agent import BenchmarkAgent
from snorkel_g_agent.provider import ProviderError
from snorkel_g_agent.schema import (
    AgentAction,
    AgentConfig,
    AppConfig,
    ModelResponse,
    RouteConfig,
    RunConfig,
    TaskSpec,
    Usage,
)
from snorkel_g_agent.tools import ToolExecutor


class CountingProvider:
    def __init__(self, finish_after: int) -> None:
        self.finish_after = finish_after
        self.calls = 0

    async def complete(self, messages, *, deadline=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls >= self.finish_after:
            content = '{"action":"finish","summary":"done after many steps"}'
        else:
            content = '{"action":"append_file","path":"scratch.txt","content":"tick\\n"}'
        return ModelResponse(content=content, model="fake", usage=Usage())


class ParseErrorThenFinishProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, deadline=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            content = "I should inspect first, but this is not a JSON action."
        else:
            content = '{"action":"finish","summary":"recovered"}'
        return ModelResponse(content=content, model="fake", usage=Usage())


class AlwaysParseErrorProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, deadline=None):  # noqa: ANN001, ARG002
        self.calls += 1
        return ModelResponse(content="not json", model="fake", usage=Usage())


class RetryableProviderFailureThenFinish:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, deadline=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("temporary 503", kind="server", retryable=True)
        return ModelResponse(
            content='{"action":"finish","summary":"resumed"}',
            model="fake",
            usage=Usage(),
        )


class FailedToolThenPrematureFinishProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, deadline=None):  # noqa: ANN001, ARG002
        self.calls += 1
        responses = [
            '{"action":"exec","cmd":"false"}',
            '{"action":"finish","summary":"giving up"}',
            '{"action":"exec","cmd":"true"}',
            '{"action":"finish","summary":"recovered"}',
        ]
        return ModelResponse(
            content=responses[self.calls - 1],
            model="fake",
            usage=Usage(),
        )


@pytest.mark.asyncio
async def test_agent_has_no_default_step_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(default_route="fake", command_timeout_seconds=5),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = CountingProvider(finish_after=85)
    agent.provider = provider  # type: ignore[assignment]

    result = await agent.run(TaskSpec(task_id="long-task", instruction="do it", workdir=tmp_path))

    assert result.status == "completed"
    assert provider.calls == 85
    assert result.steps == 86


def test_agent_uses_builtin_prompt_when_configured_prompt_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    config = AppConfig(
        run=RunConfig(default_route="fake"),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=tmp_path / "missing-system-prompt.md"),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")

    prompt = agent._load_system_prompt()

    assert "replace_in_file" in prompt
    assert "headless coding benchmark agent" in prompt
    assert "Passing tests on an unmodified tree is not task completion" in prompt
    assert "Avoid recursive searches from `/`" in prompt
    assert "visible tests are stale" in prompt


@pytest.mark.asyncio
async def test_agent_parse_error_prompts_for_repair_without_safe_exec_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(default_route="fake", command_timeout_seconds=5),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = ParseErrorThenFinishProvider()
    agent.provider = provider  # type: ignore[assignment]

    result = await agent.run(
        TaskSpec(task_id="parse-repair", instruction="do it", workdir=tmp_path)
    )

    assert result.status == "completed"
    assert provider.calls == 2
    trajectory = json.loads((result.output_dir / "agent" / "trajectory.json").read_text())
    observations = json.dumps(trajectory["steps"])
    assert "Action parse error" in observations
    assert "Return exactly one JSON object" in observations
    assert "safe inspection fallback" not in observations
    assert "rg --files" not in observations


@pytest.mark.asyncio
async def test_agent_stops_after_repeated_parse_repair_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(
            default_route="fake",
            command_timeout_seconds=5,
            max_parse_repair_attempts=2,
        ),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = AlwaysParseErrorProvider()
    agent.provider = provider  # type: ignore[assignment]

    result = await agent.run(
        TaskSpec(task_id="parse-repair-limit", instruction="do it", workdir=tmp_path)
    )

    assert result.status == "failed"
    assert provider.calls == 3
    assert result.error == "too many consecutive action parse errors (3)"


@pytest.mark.asyncio
async def test_agent_resumes_after_exhausted_retryable_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(
            default_route="fake",
            request_retry_base_seconds=0.1,
            request_retry_max_seconds=0.1,
        ),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = RetryableProviderFailureThenFinish()
    agent.provider = provider  # type: ignore[assignment]

    result = await agent.run(
        TaskSpec(task_id="provider-recovery", instruction="do it", workdir=tmp_path)
    )

    assert result.status == "completed"
    assert provider.calls == 2
    assert "Temporary model endpoint failure" in (tmp_path / "STATE_FILE.md").read_text()


@pytest.mark.asyncio
async def test_agent_retries_tool_runtime_exception_and_preserves_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(
            default_route="fake",
            tool_exception_retries=1,
            tool_retry_base_seconds=0,
        ),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = CountingProvider(finish_after=2)
    agent.provider = provider  # type: ignore[assignment]
    original_run = ToolExecutor.run
    calls = 0

    async def flaky_run(self, action: AgentAction):  # noqa: ANN001
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient tool transport failure")
        return await original_run(self, action)

    monkeypatch.setattr(ToolExecutor, "run", flaky_run)

    result = await agent.run(
        TaskSpec(task_id="tool-recovery", instruction="do it", workdir=tmp_path)
    )

    assert result.status == "completed"
    assert calls == 3
    assert (tmp_path / "scratch.txt").read_text() == "tick\n"


@pytest.mark.asyncio
async def test_agent_rejects_finish_immediately_after_failed_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(default_route="fake", command_timeout_seconds=5),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="fake-model",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    agent = BenchmarkAgent(config, "fake", config.routes["fake"], tmp_path / "out")
    provider = FailedToolThenPrematureFinishProvider()
    agent.provider = provider  # type: ignore[assignment]

    result = await agent.run(
        TaskSpec(task_id="finish-recovery", instruction="do it", workdir=tmp_path)
    )

    assert result.status == "completed"
    assert result.summary == "recovered"
    assert provider.calls == 4
    trajectory = (result.output_dir / "agent" / "trajectory.json").read_text()
    assert "Finish rejected" in trajectory
