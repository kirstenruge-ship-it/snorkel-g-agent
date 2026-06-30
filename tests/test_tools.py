from pathlib import Path

import pytest

from snorkel_g_agent.schema import AgentAction
from snorkel_g_agent.tools import ToolExecutor


@pytest.mark.asyncio
async def test_exec_runs_in_workdir(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(AgentAction(action="exec", cmd="pwd"))

    assert result.ok
    assert str(tmp_path) in result.content


@pytest.mark.asyncio
async def test_write_and_read_file(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    written = await executor.run(
        AgentAction(action="write_file", path="nested/answer.txt", content="ready")
    )
    read = await executor.run(AgentAction(action="read_file", path="nested/answer.txt"))

    assert written.ok
    assert read.ok
    assert "ready" in read.content


@pytest.mark.asyncio
async def test_rejects_path_escape(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(AgentAction(action="read_file", path="../secret.txt"))

    assert not result.ok
    assert "escapes workdir" in result.content
