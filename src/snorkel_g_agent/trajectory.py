from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from snorkel_g_agent.schema import AgentAction, ModelResponse, TaskSpec, ToolResult
from snorkel_g_agent.time_utils import utc_now

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a path in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search workspace text files with a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "context_lines": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad",
            "description": "Append a durable task note to STATE_FILE.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append UTF-8 text to a file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace text in a UTF-8 file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string"},
                    "replacement": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "whitespace_flexible": {"type": "boolean"},
                },
                "required": ["path", "find", "replacement"],
            },
        },
    },
]


class TrajectoryWriter:
    def __init__(
        self,
        output_dir: Path,
        task: TaskSpec,
        agent_name: str,
        agent_version: str,
        model_name: str,
    ) -> None:
        self.output_dir = output_dir
        self.agent_dir = output_dir / "agent"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.agent_dir / "trajectory.json"
        self.jsonl_path = output_dir / "trajectory.harbor.jsonl"
        self.session_id = f"{task.task_id}-{uuid4().hex[:12]}"
        self.model_name = model_name
        self.trajectory: dict[str, Any] = {
            "schema_version": "ATIF-v1.7",
            "session_id": self.session_id,
            "trajectory_id": self.session_id,
            "agent": {
                "name": agent_name,
                "version": agent_version,
                "model_name": model_name,
                "tool_definitions": TOOL_DEFINITIONS,
                "extra": {"benchmark": task.benchmark, "task_id": task.task_id},
            },
            "steps": [],
            "notes": "Generated directly by snorkel-g-agent during task execution.",
            "extra": {"workdir": str(task.workdir)},
        }

    def add_user_step(self, message: str) -> None:
        self._add_step({"source": "user", "message": message})

    def add_system_step(self, message: str, observation: str | None = None) -> None:
        step: dict[str, Any] = {"source": "system", "message": message}
        if observation is not None:
            step["observation"] = {"results": [{"content": observation}]}
        self._add_step(step)

    def add_agent_step(
        self,
        response: ModelResponse,
        action: AgentAction,
        result: ToolResult | None = None,
    ) -> None:
        tool_call_id = f"call_{len(self.trajectory['steps']) + 1}_1"
        step: dict[str, Any] = {
            "source": "agent",
            "model_name": response.model,
            "message": response.content,
            "llm_call_count": 1,
            "metrics": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cached_tokens": response.usage.cached_tokens,
            },
        }
        if action.action != "finish":
            step["tool_calls"] = [
                {
                    "tool_call_id": tool_call_id,
                    "function_name": action.action,
                    "arguments": action.model_dump(exclude_none=True),
                }
            ]
        if result is not None:
            step["observation"] = {
                "results": [
                    {
                        "source_call_id": tool_call_id if action.action != "finish" else None,
                        "content": result.content,
                        "extra": {
                            "ok": result.ok,
                            "exit_code": result.exit_code,
                            "timed_out": result.timed_out,
                            "truncated": result.truncated,
                            **result.extra,
                        },
                    }
                ]
            }
        self._add_step(step)

    def finalize(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.trajectory["final_metrics"] = {
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_steps": len(self.trajectory["steps"]),
        }
        self._write_json()

    def _add_step(self, step: dict[str, Any]) -> None:
        step["step_id"] = len(self.trajectory["steps"]) + 1
        step["timestamp"] = utc_now()
        self.trajectory["steps"].append(step)
        self._write_json()
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")

    def _write_json(self) -> None:
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.trajectory, indent=2, ensure_ascii=False))
        tmp.replace(self.json_path)
