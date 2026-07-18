from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from snorkel_g_agent import __version__
from snorkel_g_agent.actions import ActionParseError, parse_action
from snorkel_g_agent.context import ContextWindow, append_state_note, initialize_state_file
from snorkel_g_agent.logging import AgentLogger
from snorkel_g_agent.prompts import DEFAULT_SYSTEM_PROMPT
from snorkel_g_agent.provider import OpenAICompatibleProvider, ProviderError
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
        self.provider.request_log_path = task_dir / "llm_requests.jsonl"
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
            consecutive_parse_errors = 0
            consecutive_provider_failures = 0
            consecutive_tool_failures = 0
            last_failed_action: str | None = None
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
                try:
                    response = await self.provider.complete(
                        context.messages,
                        deadline=deadline,
                    )
                except ProviderError as exc:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if not exc.retryable or remaining <= 0:
                        status = "timeout" if exc.kind == "task_deadline" else "failed"
                        error = str(exc)
                        logger.event("error", step=step, message=error, kind=exc.kind)
                        break
                    consecutive_provider_failures += 1
                    sleep_seconds = min(
                        self.config.run.request_retry_base_seconds
                        * (2 ** min(consecutive_provider_failures - 1, 6)),
                        self.config.run.request_retry_max_seconds,
                        max(0.0, remaining),
                    ) + random.uniform(0, min(1.0, max(0.0, remaining)))
                    sleep_seconds = min(sleep_seconds, max(0.0, remaining))
                    message = (
                        "Temporary model endpoint failure. The runtime preserved the task "
                        "workspace, context, and STATE_FILE.md. Resume from the exact prior "
                        "step when service returns; do not restart broad inspection. "
                        f"Failure kind: {exc.kind}."
                    )
                    append_state_note(state_file, message)
                    context.add("user", message)
                    logger.event(
                        "provider_recovery",
                        step=step,
                        failure=consecutive_provider_failures,
                        sleep_seconds=round(sleep_seconds, 3),
                        kind=exc.kind,
                        message=str(exc),
                    )
                    if sleep_seconds > 0:
                        await asyncio.sleep(sleep_seconds)
                    continue
                consecutive_provider_failures = 0
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
                    consecutive_parse_errors += 1
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
                    if (
                        consecutive_parse_errors
                        > self.config.run.max_parse_repair_attempts
                    ):
                        status = "failed"
                        error = (
                            "too many consecutive action parse errors "
                            f"({consecutive_parse_errors})"
                        )
                        logger.event("error", step=step, message=error)
                        break
                    continue

                consecutive_parse_errors = 0
                tool_item_id = logger.tool_started(action)
                action_key = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
                if action.action == "finish" and consecutive_tool_failures:
                    result = ToolResult(
                        ok=False,
                        content=(
                            "Finish rejected because the immediately preceding tool operation "
                            "failed. Repair or replace that operation, verify the task outcome, "
                            "and only then finish. A failed tool call is evidence, not completion."
                        ),
                        extra={"finish_rejected_after_tool_failure": True},
                    )
                else:
                    result = await self._run_tool_with_retries(
                        executor,
                        action,
                        logger,
                        step,
                        deadline,
                    )
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
                feedback = result.content
                if not result.ok:
                    consecutive_tool_failures += 1
                    repeated = action_key == last_failed_action
                    last_failed_action = action_key
                    feedback += (
                        "\n\nTOOL FAILURE RECOVERY: Continue working. Inspect the concrete "
                        "failure, preserve any successful partial work, and retry the individual "
                        "operation with corrected arguments or a smaller deterministic command. "
                        "Do not abandon the task and do not repeat the identical failed action "
                        "unchanged."
                    )
                    if repeated:
                        feedback += (
                            " This exact action already failed previously; change the approach now."
                        )
                else:
                    consecutive_tool_failures = 0
                    last_failed_action = None
                context.add("user", feedback)
                self._update_state(state_file, action, result)

                if action.action == "finish" and result.ok:
                    status = "completed"
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

    async def _run_tool_with_retries(
        self,
        executor: ToolExecutor,
        action: AgentAction,
        logger: AgentLogger,
        step: int,
        deadline: float,
    ) -> ToolResult:
        for attempt in range(self.config.run.tool_exception_retries + 1):
            try:
                return await executor.run(action)
            except Exception as exc:
                remaining = deadline - asyncio.get_running_loop().time()
                if attempt >= self.config.run.tool_exception_retries or remaining <= 0:
                    return ToolResult(
                        ok=False,
                        content=(
                            f"Tool runtime raised {type(exc).__name__}: {exc}. "
                            "The task workspace and prior progress were preserved."
                        ),
                        extra={
                            "tool_exception": type(exc).__name__,
                            "attempts": attempt + 1,
                        },
                    )
                sleep_seconds = min(
                    self.config.run.tool_retry_base_seconds * (2**attempt)
                    + random.uniform(0, 0.25),
                    max(0.0, remaining),
                )
                logger.event(
                    "tool_retry",
                    step=step,
                    action=action.action,
                    attempt=attempt + 1,
                    max_attempts=self.config.run.tool_exception_retries + 1,
                    sleep_seconds=round(sleep_seconds, 3),
                    message=str(exc),
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
        raise AssertionError("unreachable")

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
                "Continue until the requested artifacts or code are implemented and checked. "
                "Endpoint retries and individual tool failures are recoverable events, not reasons "
                "to stop or restart the task.",
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
