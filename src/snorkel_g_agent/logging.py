from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from snorkel_g_agent.schema import AgentAction, ToolResult
from snorkel_g_agent.time_utils import utc_now


class AgentLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.jsonl_path = output_dir / "agent.log.jsonl"
        self.text_path = output_dir / "agent.log.txt"
        self.codex_path = output_dir / "codex.txt"
        self.thread_id = f"thread-{uuid4().hex}"
        self._item_index = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_text("Reading additional input from stdin...")
        self._write_codex_event({"type": "thread.started", "thread_id": self.thread_id})
        self._write_codex_event({"type": "turn.started"})

    def event(self, event_type: str, **payload: Any) -> None:
        record = {"timestamp": utc_now(), "event": event_type, **payload}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if event_type == "compact":
            self.agent_message(f"Context compacted: {payload.get('message', '')}")
        elif event_type == "provider_retry":
            self.agent_message(
                "Provider retry "
                f"step={payload.get('step')} "
                f"attempt={payload.get('attempt')}/{payload.get('max_attempts')} "
                f"sleep={payload.get('sleep_seconds')}s: {payload.get('message', '')}"
            )
        elif event_type == "error":
            self.agent_message(f"Error: {payload.get('message', '')}")

    def agent_message(self, text: str) -> None:
        self._write_item_completed({"type": "agent_message", "text": text})

    def tool_started(self, action: AgentAction) -> str:
        item_id = self._next_item_id()
        self._write_codex_event(
            {
                "type": "item.started",
                "item": self._tool_item(item_id, action, status="in_progress"),
            }
        )
        return item_id

    def tool_completed(self, item_id: str, action: AgentAction, result: ToolResult) -> None:
        self._write_codex_event(
            {
                "type": "item.completed",
                "item": self._tool_item(item_id, action, status="completed", result=result),
            }
        )

    def turn_completed(self, input_tokens: int, output_tokens: int) -> None:
        self._write_codex_event(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": None,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": None,
                },
            }
        )

    def _write_item_completed(self, item: dict[str, Any]) -> None:
        item["id"] = self._next_item_id()
        self._write_codex_event({"type": "item.completed", "item": item})

    def _tool_item(
        self,
        item_id: str,
        action: AgentAction,
        *,
        status: str,
        result: ToolResult | None = None,
    ) -> dict[str, Any]:
        if action.action in {"write_file", "append_file", "replace_in_file", "scratchpad"}:
            return {
                "id": item_id,
                "type": "file_change",
                "changes": [self._file_change(action)],
                "status": status,
            }

        item: dict[str, Any] = {
            "id": item_id,
            "type": "command_execution",
            "command": self._tool_command(action),
            "aggregated_output": "" if result is None else self._summarize_output(action, result),
            "exit_code": None if result is None else result.exit_code,
            "status": status,
        }
        if result is not None and action.action != "exec":
            item["exit_code"] = 0 if result.ok else 1
        return item

    @staticmethod
    def _file_change(action: AgentAction) -> dict[str, str]:
        if action.action == "scratchpad":
            return {"path": "STATE_FILE.md", "kind": "update"}
        return {"path": action.path or "", "kind": "update"}

    @staticmethod
    def _tool_command(action: AgentAction) -> str:
        if action.action == "exec":
            return f"/bin/bash -lc {shlex.quote(action.cmd or '')}"
        if action.action == "read_file":
            return f"read_file {action.path}"
        if action.action == "list_files":
            path = action.path or "."
            glob = f" --glob {action.glob}" if action.glob else ""
            return f"list_files {path}{glob}"
        if action.action == "search_text":
            path = action.path or "."
            glob = f" --glob {action.glob}" if action.glob else ""
            return f"search_text {shlex.quote(action.pattern or '')} {path}{glob}"
        if action.action == "finish":
            return "finish"
        return action.action

    @staticmethod
    def _summarize_output(action: AgentAction, result: ToolResult) -> str:
        if action.action == "read_file":
            _, _, body = result.content.partition("\n\n")
            return (
                f"READ {action.path} ({len(body)} chars; full file content is in trajectory.json)"
            )
        if action.action == "finish":
            return result.content
        if action.action in {"list_files", "search_text", "exec"}:
            return AgentLogger._compact_output(result.content, limit=4000)
        return AgentLogger._compact_output(result.content, limit=1200)

    @staticmethod
    def _compact_output(text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        marker = "... <human log output truncated; full output is in trajectory.json> ..."
        return f"{head}\n\n{marker}\n\n{tail}"

    def _next_item_id(self) -> str:
        item_id = f"item_{self._item_index}"
        self._item_index += 1
        return item_id

    def _write_codex_event(self, payload: dict[str, Any]) -> None:
        self._write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def _write_text(self, line: str) -> None:
        for path in (self.text_path, self.codex_path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(line, flush=True)
