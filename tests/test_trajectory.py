from pathlib import Path

from snorkel_g_agent.schema import AgentAction, ModelResponse, TaskSpec, ToolResult, Usage
from snorkel_g_agent.trajectory import TrajectoryWriter


def test_trajectory_writer_emits_harbor_atif(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="task-a",
        benchmark="terminal-bench",
        instruction="do it",
        workdir=tmp_path,
    )
    writer = TrajectoryWriter(tmp_path / "out", task, "snorkel-g-agent", "0.0", "glm-5.2")
    writer.add_user_step("hello")
    response = ModelResponse(
        content='{"action":"exec","cmd":"pwd"}',
        model="glm-5.2",
        usage=Usage(),
        raw={
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "exec", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        },
    )
    writer.add_agent_step(
        response,
        AgentAction(action="exec", cmd="pwd"),
        ToolResult(ok=True, content="ok", exit_code=0),
    )
    writer.finalize(10, 5)

    data = writer.json_path.read_text()

    assert '"schema_version": "ATIF-v1.7"' in data
    assert '"tool_calls"' in data
    assert '"name": "replace_in_file"' in data
    assert '"name": "search_text"' in data
    assert '"name": "scratchpad"' in data
    assert '"provider_response"' in data
    assert '"native_tool_calls"' in data
    assert (tmp_path / "out" / "trajectory.harbor.jsonl").exists()
