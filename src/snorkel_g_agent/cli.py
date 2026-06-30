from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from snorkel_g_agent.agent import BenchmarkAgent
from snorkel_g_agent.batch import read_manifest, run_batch
from snorkel_g_agent.config import load_config, resolve_route
from snorkel_g_agent.schema import TaskSpec

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
    route: Annotated[str | None, typer.Option("--route")] = None,
) -> None:
    app_config = load_config(config)
    selected = route or app_config.run.default_route
    resolved = resolve_route(app_config, selected)
    console.print(f"OK route={selected} provider={resolved.provider} model={resolved.model}")


@app.command("run-task")
def run_task(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
    task: Annotated[Path, typer.Option("--task", "-t", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("runs/single"),
    route: Annotated[str | None, typer.Option("--route")] = None,
) -> None:
    app_config = load_config(config)
    task_spec = TaskSpec.model_validate_json(task.read_text())
    if not task_spec.workdir.is_absolute():
        task_spec.workdir = (task.parent / task_spec.workdir).resolve()
    selected = route or app_config.run.default_route
    route_config = resolve_route(app_config, selected)
    agent = BenchmarkAgent(app_config, selected, route_config, out)
    result = asyncio.run(agent.run(task_spec))
    console.print_json(json.dumps(result.model_dump(mode="json")))


@app.command("run-batch")
def run_batch_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option("--manifest", "-m", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o")],
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1)] = None,
    route: Annotated[str | None, typer.Option("--route")] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    app_config = load_config(config)
    tasks = read_manifest(manifest)
    base = manifest.parent
    for task in tasks:
        if not task.workdir.is_absolute():
            task.workdir = (base / task.workdir).resolve()
    results = asyncio.run(run_batch(app_config, tasks, out, concurrency, route, resume))
    table = Table(title="snorkel-g-agent batch")
    table.add_column("task_id")
    table.add_column("status")
    table.add_column("steps", justify="right")
    table.add_column("output")
    for result in results:
        table.add_row(result.task_id, result.status, str(result.steps), str(result.output_dir))
    console.print(table)


if __name__ == "__main__":
    app()
