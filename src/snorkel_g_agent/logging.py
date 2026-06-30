from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snorkel_g_agent.time_utils import utc_now


class AgentLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.jsonl_path = output_dir / "agent.log.jsonl"
        self.text_path = output_dir / "agent.log.txt"
        output_dir.mkdir(parents=True, exist_ok=True)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {"timestamp": utc_now(), "event": event_type, **payload}
        formatted = self._format_text(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(formatted + "\n")
        print(formatted, flush=True)

    def _format_text(self, record: dict[str, Any]) -> str:
        event = record["event"]
        timestamp = record["timestamp"]
        if event == "model":
            return (
                f"[{timestamp}] MODEL step={record.get('step')} route={record.get('route')}\n"
                f"{record.get('content', '')}"
            )
        if event == "tool":
            return (
                f"[{timestamp}] TOOL step={record.get('step')} action={record.get('action')} "
                f"ok={record.get('ok')} exit={record.get('exit_code')}\n{record.get('content', '')}"
            )
        if event == "compact":
            return f"[{timestamp}] COMPACT {record.get('message', '')}"
        if event == "provider_retry":
            return (
                f"[{timestamp}] PROVIDER_RETRY step={record.get('step')} "
                f"route={record.get('route')} "
                f"attempt={record.get('attempt')}/{record.get('max_attempts')} "
                f"sleep={record.get('sleep_seconds')}s {record.get('message', '')}"
            )
        if event == "error":
            return f"[{timestamp}] ERROR {record.get('message', '')}"
        return f"[{timestamp}] {event.upper()} {json.dumps(record, ensure_ascii=False)}"
