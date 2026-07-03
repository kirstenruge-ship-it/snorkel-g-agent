from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from snorkel_g_agent import __version__
from snorkel_g_agent.actions import ActionParseError, parse_action
from snorkel_g_agent.context import ContextWindow, append_state_note, initialize_state_file
from snorkel_g_agent.logging import AgentLogger
from snorkel_g_agent.prompts import DEFAULT_SYSTEM_PROMPT
from snorkel_g_agent.provider import OpenAICompatibleProvider
from snorkel_g_agent.schema import (
    AgentAction,
    AppConfig,
    RouteConfig,
    TaskResult,
    TaskSpec,
    ToolResult,
)
from snorkel_g_agent.tools import ToolExecutor
from snorkel_g_agent.trajectory import TrajectoryWriter


class BenchmarkAgent:
    def __init__(
        self,
        config: AppConfig,
        route_name: str,
        route: RouteConfig,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.route_name = route_name
        self.route = route
        self.output_dir = output_dir
        self.provider = OpenAICompatibleProvider(
            route,
            config.run.request_timeout_seconds,
            config.run.request_retries,
            config.run.request_retry_base_seconds,
            config.run.request_retry_max_seconds,
            config.run.max_model_tokens,
        )

    async def run(self, task: TaskSpec) -> TaskResult:
        task_dir = self.output_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        logger = AgentLogger(task_dir)
        task_workdir = task.workdir.resolve()
        state_file = task_workdir / self.config.run.state_file_name
        initialize_state_file(state_file, task)

        system_prompt = self._load_system_prompt()
        initial_user = self._initial_user_prompt(task, state_file)
        context = ContextWindow(self.config.run.context_limit_tokens, state_file)
        context.add("system", system_prompt)
        context.add("user", initial_user)

        trajectory = TrajectoryWriter(
            output_dir=task_dir,
            task=task,
            agent_name=self.config.agent.name,
            agent_version=__version__,
            model_name=self.route.model,
        )
        trajectory.add_user_step(initial_user)
        executor = ToolExecutor(
            task_workdir,
            self.config.run.command_timeout_seconds,
            self.config.run.max_tool_output_chars,
        )

        prompt_tokens = 0
        completion_tokens = 0
        summary: str | None = None
        status = "timeout"
        error: str | None = None
        deadline = asyncio.get_running_loop().time() + (
            task.timeout_seconds or self.config.run.task_timeout_seconds
        )

        logger.event("start", task_id=task.task_id, benchmark=task.benchmark, route=self.route_name)
        try:
            step = 0
            while True:
                step += 1
                if asyncio.get_running_loop().time() >= deadline:
                    status = "timeout"
                    error = "task timeout reached"
                    break
                if context.compact_if_needed(task):
                    message = f"Compacted context at approx {context.token_estimate()} tokens."
                    logger.event("compact", step=step, message=message)
                    trajectory.add_system_step("Context compacted.", message)

                self.provider.on_retry = lambda payload, current_step=step: logger.event(
                    "provider_retry",
                    step=current_step,
                    route=self.route_name,
                    **payload,
                )
                response = await self.provider.complete(context.messages)
                prompt_tokens += response.usage.prompt_tokens or 0
                completion_tokens += response.usage.completion_tokens or 0
                logger.event(
                    "model",
                    step=step,
                    route=self.route_name,
                    content=response.content,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
                logger.agent_message(response.content)
                context.add("assistant", response.content)

                try:
                    action = parse_action(response.content)
                except ActionParseError as exc:
                    action = AgentAction(
                        action="scratchpad",
                        title="Action parse error",
                        content=response.content[:2000],
                    )
                    result = ToolResult(
                        ok=False,
                        content=(
                            "Action parse error: "
                            f"{exc}\n\n"
                            "Your previous response did not contain a valid action. "
                            "Return exactly one JSON object and no prose, no markdown, "
                            "and no <tool_call> wrapper. Examples:\n"
                            '{"action":"exec","cmd":"pwd","timeout_seconds":10}\n'
                            '{"action":"scratchpad","title":"note","content":"what you learned"}\n'
                            '{"action":"finish","summary":"done","tests":"not run"}'
                        ),
                        extra={"parse_error": True},
                    )
                    trajectory.add_agent_step(response, action, result)
                    context.add("user", result.content)
                    logger.event("error", step=step, message=str(exc))
                    logger.event(
                        "tool",
                        step=step,
                        action="parse_error",
                        ok=result.ok,
                        exit_code=result.exit_code,
                        content=result.content,
                    )
                    continue

                tool_item_id = logger.tool_started(action)
                result = await executor.run(action)
                logger.event(
                    "tool",
                    step=step,
                    action=action.action,
                    ok=result.ok,
                    exit_code=result.exit_code,
                    content=result.content,
                )
                logger.tool_completed(tool_item_id, action, result)
                trajectory.add_agent_step(response, action, result)
                context.add("user", result.content)
                self._update_state(state_file, action, result)

                if action.action == "finish":
                    status = "completed" if result.ok else "failed"
                    summary = action.summary or result.content
                    break
        except Exception as exc:
            status = "failed"
            error = str(exc)
            logger.event("error", message=error)

        trajectory.finalize(prompt_tokens, completion_tokens)
        result = TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            status=status,  # type: ignore[arg-type]
            steps=len(trajectory.trajectory["steps"]),
            output_dir=task_dir,
            summary=summary,
            error=error,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        (task_dir / "result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2))
        logger.event("finish", **result.model_dump(mode="json"))
        logger.turn_completed(prompt_tokens, completion_tokens)
        return result

    def _initial_user_prompt(self, task: TaskSpec, state_file: Path) -> str:
        return "\n".join(
            [
                f"Task ID: {task.task_id}",
                f"Benchmark: {task.benchmark}",
                f"Workspace: {task.workdir.resolve()}",
                f"State file: {state_file}",
                "",
                "Task instruction:",
                task.instruction,
                "",
                "Start by inspecting the workspace. Return exactly one JSON action.",
            ]
        )

    def _load_system_prompt(self) -> str:
        path = self.config.agent.system_prompt_path
        if path.exists():
            return path.read_text()
        return DEFAULT_SYSTEM_PROMPT

    def _update_state(self, state_file: Path, action: AgentAction, result: ToolResult) -> None:
        action_data: dict[str, Any] = action.model_dump(exclude_none=True)
        append_state_note(
            state_file,
            "\n".join(
                [
                    f"- Action: `{action.action}`",
                    f"- Args: `{json.dumps(action_data, ensure_ascii=False)[:1000]}`",
                    f"- OK: {result.ok}",
                    f"- Exit code: {result.exit_code}",
                    f"- Timed out: {result.timed_out}",
                ]
            ),
        )
