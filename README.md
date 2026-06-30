# snorkel-g-agent

Dedicated GLM-5.2 benchmark agent for private SWE-bench and Terminal-Bench style sweeps.

The design target is a benchmark runner that:

- treats GLM-5.2 as a first-class coding agent model, not a chat bot wedged into a shell loop;
- supports Portkey/Fireworks and Modal dedicated OpenAI-compatible endpoints;
- caps active context at 600k tokens with a Codex-style `STATE_FILE.md` reset;
- writes readable Claude Code-like agent logs as JSONL plus plain text;
- emits Harbor-native trajectory JSONL during the run, not as a post-hoc conversion;
- can run large task batches with explicit concurrency, timeouts, and resumable outputs.

## Quick Start

```bash
uv sync --extra dev
uv run snorkel-g-agent validate-config --config configs/glm52.example.yaml
uv run snorkel-g-agent run-task --config configs/glm52.example.yaml --task examples/tbench_task.json
```

For a large private sweep:

```bash
uv run snorkel-g-agent run-batch \
  --config configs/glm52.example.yaml \
  --manifest /path/to/tasks.jsonl \
  --out runs/glm52-private \
  --concurrency 50 \
  --resume
```

Each task gets:

- `agent.log.jsonl`
- `agent.log.txt`
- `trajectory.harbor.jsonl`
- `STATE_FILE.md`
- `result.json`

## Task Manifest Format

Each JSONL row should match:

```json
{
  "task_id": "task-001",
  "benchmark": "terminal-bench",
  "instruction": "Solve the task...",
  "workdir": "/absolute/path/to/task/workdir",
  "timeout_seconds": 7200
}
```

`benchmark` can be `terminal-bench`, `swe-bench`, or `generic`.

## Provider Routes

`configs/glm52.example.yaml` includes two routes:

- `modal_glm52`: direct dedicated endpoint, preferred for throughput. Uses Modal proxy auth
  headers from `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
- `portkey_fireworks_glm52`: Portkey into Fireworks priority GLM-5.2.

Secrets are read from environment variables. Do not put API keys in YAML.

The batch runner resumes completed task directories by default. Delete a task output directory or pass
`--no-resume` to force a rerun.

## Quality Gates

Before declaring changes finished:

```bash
uv run ruff check --fix .
uv run pytest -q
```
