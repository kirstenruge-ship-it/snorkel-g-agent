from pathlib import Path

from snorkel_g_agent.context import ContextWindow, initialize_state_file
from snorkel_g_agent.schema import TaskSpec


def test_context_compacts_to_state_file(tmp_path: Path) -> None:
    task = TaskSpec(task_id="t1", benchmark="generic", instruction="do it", workdir=tmp_path)
    state_file = tmp_path / "STATE_FILE.md"
    initialize_state_file(state_file, task)
    context = ContextWindow(limit_tokens=10, state_file=state_file)
    context.add("system", "x" * 100)
    context.add("user", "recent")

    compacted = context.compact_if_needed(task, keep_recent=1)

    assert compacted
    assert context.compactions == 1
    assert "STATE_FILE.md" in context.messages[1].content
    assert context.messages[-1].content == "recent"
