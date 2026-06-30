import json
from pathlib import Path

import pytest

from snorkel_g_agent.batch import run_batch
from snorkel_g_agent.schema import AgentConfig, AppConfig, RouteConfig, RunConfig, TaskSpec


@pytest.mark.asyncio
async def test_batch_resume_skips_completed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "secret")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    config = AppConfig(
        run=RunConfig(default_route="fake"),
        routes={
            "fake": RouteConfig(
                provider="openai-compatible",
                model="glm-5.2",
                base_url="http://127.0.0.1:1/v1",
                api_key_env="FAKE_API_KEY",
            )
        },
        agent=AgentConfig(system_prompt_path=system_prompt),
    )
    out = tmp_path / "runs"
    result_dir = out / "done-task"
    result_dir.mkdir(parents=True)
    expected = {
        "task_id": "done-task",
        "benchmark": "generic",
        "status": "completed",
        "steps": 2,
        "output_dir": str(result_dir),
        "summary": "already done",
        "error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    (result_dir / "result.json").write_text(json.dumps(expected))

    results = await run_batch(
        config,
        [TaskSpec(task_id="done-task", instruction="do it", workdir=tmp_path)],
        out,
        concurrency=1,
        resume=True,
    )

    assert results[0].status == "completed"
    assert results[0].summary == "already done"
