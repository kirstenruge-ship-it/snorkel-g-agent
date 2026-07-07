from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

BenchmarkName = Literal["terminal-bench", "swe-bench", "generic"]


class RouteConfig(BaseModel):
    provider: Literal["modal", "portkey", "openai-compatible"]
    model: str
    base_url: str
    api_key_env: str | None = None
    modal_key_env: str = "MODAL_TOKEN_ID"
    modal_secret_env: str = "MODAL_TOKEN_SECRET"
    priority: int = 0
    headers: dict[str, str] = Field(default_factory=dict)


class RunConfig(BaseModel):
    default_route: str
    max_concurrency: int = Field(default=50, ge=1)
    context_limit_tokens: int = Field(default=600_000, ge=1)
    request_timeout_seconds: int = Field(default=180, ge=1)
    request_retries: int = Field(default=8, ge=0)
    request_retry_base_seconds: float = Field(default=2.0, ge=0.1)
    request_retry_max_seconds: float = Field(default=30.0, ge=0.1)
    max_model_tokens: int = Field(default=4096, ge=1)
    command_timeout_seconds: int = Field(default=600, ge=1)
    task_timeout_seconds: int = Field(default=7200, ge=1)
    max_parse_repair_attempts: int = Field(default=5, ge=0)
    max_tool_output_chars: int = Field(default=24_000, ge=1000)
    state_file_name: str = "STATE_FILE.md"


class AgentConfig(BaseModel):
    name: str = "snorkel-g-agent"
    system_prompt_path: Path
    finish_markers: list[str] = Field(default_factory=lambda: ["FINAL_ANSWER", "TASK_COMPLETE"])


class AppConfig(BaseModel):
    run: RunConfig
    routes: dict[str, RouteConfig]
    agent: AgentConfig

    @field_validator("routes")
    @classmethod
    def require_routes(cls, routes: dict[str, RouteConfig]) -> dict[str, RouteConfig]:
        if not routes:
            raise ValueError("at least one route must be configured")
        return routes


class TaskSpec(BaseModel):
    task_id: str
    benchmark: BenchmarkName = "generic"
    instruction: str
    workdir: Path
    timeout_seconds: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def clean_task_id(cls, task_id: str) -> str:
        cleaned = task_id.strip()
        if not cleaned:
            raise ValueError("task_id cannot be empty")
        return cleaned


class ModelMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


class ModelResponse(BaseModel):
    content: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    action: Literal[
        "exec",
        "list_files",
        "search_text",
        "scratchpad",
        "read_file",
        "write_file",
        "append_file",
        "replace_in_file",
        "finish",
    ]
    cmd: str | None = None
    path: str | None = None
    content: str | None = None
    pattern: str | None = None
    glob: str | None = None
    max_results: int = Field(default=200, ge=1)
    context_lines: int = Field(default=0, ge=0, le=5)
    title: str | None = None
    find: str | None = None
    within: str | None = None
    replacement: str | None = None
    regex: bool = False
    count: int | None = Field(default=1, ge=0)
    whitespace_flexible: bool = False
    timeout_seconds: int | None = None
    summary: str | None = None
    tests: str | None = None


class ToolResult(BaseModel):
    ok: bool
    content: str
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    benchmark: BenchmarkName
    status: Literal["completed", "failed", "timeout"]
    steps: int
    output_dir: Path
    summary: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
