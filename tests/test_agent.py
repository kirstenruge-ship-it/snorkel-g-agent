from __future__ import annotations

import json
from pathlib import Path

import pytest

from snorkel_g_agent.agent import BenchmarkAgent
from snorkel_g_agent.schema import (
    AgentConfig,
    AppConfig,
    ModelResponse,
    RouteConfig,
    RunConfig,
    TaskSpec,
    Usage,
)


class CountingProvider:
    def __init__(self, finish_after: int) -> None:
        self.finish_after = finish_after
        self.calls = 0

    async def complete(self, messages):  # noqa: ANN001
        self.calls += 1
        if self.calls >= self.finish_after:
            content = '{"action":"finish","summary":"done after many steps"}'
        else:
            content = '{"action":"append_file","path":"scratch.txt","content":"tick\\n"}'
        return ModelResponse(content=content, model="fake", usage=Usage())


class ParseErrorThenFinishProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            content = "I should inspect first, but this is not a JSON action."
        else:
            content = '{"action":"finish","summary":"recovered"}'
        return ModelResponse(content=content, model="fake", usage=Usage())


class AlwaysParseErrorProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages):  # noqa: ANN001
        self.calls += 1
        return ModelResponse(content="not json", model="fake", usage=Usage())


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
