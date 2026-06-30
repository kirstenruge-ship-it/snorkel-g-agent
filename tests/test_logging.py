from __future__ import annotations

import json
from pathlib import Path

from snorkel_g_agent.logging import AgentLogger
from snorkel_g_agent.schema import AgentAction, ToolResult


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def test_agent_logger_writes_codex_style_jsonl(tmp_path: Path) -> None:
    logger = AgentLogger(tmp_path)
    logger.agent_message('{"action":"read_file","path":"main.py"}')
    item_id = logger.tool_started(AgentAction(action="read_file", path="main.py"))
    logger.tool_completed(
        item_id,
        AgentAction(action="read_file", path="main.py"),
        ToolResult(ok=True, content="READ main.py\n\nprint('secret code')\n", exit_code=None),
    )
    logger.turn_completed(10, 3)

    text = (tmp_path / "agent.log.txt").read_text()
    assert (tmp_path / "codex.txt").read_text() == text
    assert "print('secret code')" not in text
    assert "full file content is in trajectory.json" in text

    lines = text.splitlines()
    assert lines[0] == "Reading additional input from stdin..."
    events = _json_lines(tmp_path / "agent.log.txt")
    assert events[0]["type"] == "thread.started"
    assert events[1]["type"] == "turn.started"
    assert events[-1]["type"] == "turn.completed"
    assert all("\n" not in line for line in lines)


def test_agent_logger_represents_file_writes_as_file_change(tmp_path: Path) -> None:
    logger = AgentLogger(tmp_path)
    action = AgentAction(action="write_file", path="nested/out.txt", content="hello")

    item_id = logger.tool_started(action)
    logger.tool_completed(item_id, action, ToolResult(ok=True, content="WROTE nested/out.txt"))

    events = _json_lines(tmp_path / "agent.log.txt")
    completed = events[-1]
    assert completed["type"] == "item.completed"
    assert completed["item"]["type"] == "file_change"
    assert completed["item"]["changes"] == [{"path": "nested/out.txt", "kind": "update"}]
