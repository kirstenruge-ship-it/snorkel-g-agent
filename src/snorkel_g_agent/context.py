from __future__ import annotations

from pathlib import Path

from snorkel_g_agent.schema import ModelMessage, TaskSpec
from snorkel_g_agent.time_utils import utc_now


def estimate_tokens(text: str) -> int:
    # Conservative approximation for code-heavy benchmark prompts.
    return max(1, len(text) // 3)


class ContextWindow:
    def __init__(self, limit_tokens: int, state_file: Path) -> None:
        self.limit_tokens = limit_tokens
        self.state_file = state_file
        self.messages: list[ModelMessage] = []
        self.compactions = 0

    def add(self, role: str, content: str) -> None:
        self.messages.append(ModelMessage(role=role, content=content))  # type: ignore[arg-type]

    def token_estimate(self) -> int:
        return sum(estimate_tokens(message.content) for message in self.messages)

    def should_compact(self) -> bool:
        return self.token_estimate() >= self.limit_tokens

    def compact_if_needed(self, task: TaskSpec, keep_recent: int = 8) -> bool:
        if not self.should_compact():
            return False
        self.compactions += 1
        state = self.state_file.read_text() if self.state_file.exists() else ""
        recent = self.messages[-keep_recent:]
        self.messages = [
            ModelMessage(
                role="system",
                content=(
                    f"Context was compacted at {utc_now()} for task {task.task_id}. "
                    "Continue from STATE_FILE.md and the recent interaction tail."
                ),
            ),
            ModelMessage(
                role="user",
                content=f"Current STATE_FILE.md:\n\n{state}\n\nContinue the task.",
            ),
            *recent,
        ]
        return True


def initialize_state_file(path: Path, task: TaskSpec) -> None:
    if path.exists():
        return
    path.write_text(
        "\n".join(
            [
                f"# STATE_FILE for {task.task_id}",
                "",
                f"- Created: {utc_now()}",
                f"- Benchmark: {task.benchmark}",
                "- Objective: solve the task accurately using shell/file tools.",
                "- Files touched: none yet.",
                "- Commands run: none yet.",
                "- Test status: not run yet.",
                "- Current blockers: none known.",
                "- Next step: inspect the workspace and task instructions.",
                "",
            ]
        )
    )


def append_state_note(path: Path, note: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Runtime Note {utc_now()}\n\n{note.strip()}\n")
