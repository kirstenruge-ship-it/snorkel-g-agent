from __future__ import annotations

import asyncio
import json
from pathlib import Path

from snorkel_g_agent.agent import BenchmarkAgent
from snorkel_g_agent.config import resolve_route
from snorkel_g_agent.schema import AppConfig, TaskResult, TaskSpec


def read_manifest(path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(TaskSpec.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid manifest row {line_no}: {exc}") from exc
    return tasks


async def run_batch(
    config: AppConfig,
    tasks: list[TaskSpec],
    output_dir: Path,
    concurrency: int | None = None,
    route_name: str | None = None,
    resume: bool = True,
) -> list[TaskResult]:
    selected_route_name = route_name or config.run.default_route
    route = resolve_route(config, selected_route_name)
    limit = concurrency or config.run.max_concurrency
    semaphore = asyncio.Semaphore(limit)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

    async def run_one(task: TaskSpec) -> TaskResult:
        async with semaphore:
            result_path = output_dir / task.task_id / "result.json"
            if resume and result_path.exists():
                raw = await asyncio.to_thread(result_path.read_text)
                result = TaskResult.model_validate_json(raw)
                if result.status == "completed":
                    return result
            agent = BenchmarkAgent(config, selected_route_name, route, output_dir)
            return await agent.run(task)

    results = await asyncio.gather(*(run_one(task) for task in tasks))
    summary_path = output_dir / "batch_summary.json"
    summary = json.dumps([result.model_dump(mode="json") for result in results], indent=2)
    await asyncio.to_thread(summary_path.write_text, summary)
    return list(results)
