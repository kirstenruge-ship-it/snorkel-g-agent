from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from snorkel_g_agent.schema import AgentAction

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ActionParseError(ValueError):
    pass


def _candidate_json_blobs(text: str) -> list[str]:
    candidates = [match.group(1) for match in _FENCED_JSON.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.insert(0, stripped)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def parse_action(text: str) -> AgentAction:
    errors: list[str] = []
    for blob in _candidate_json_blobs(text):
        try:
            raw: dict[str, Any] = json.loads(blob)
            return AgentAction.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(str(exc))
    joined = "; ".join(errors[-2:]) if errors else "no JSON object found"
    raise ActionParseError(f"could not parse agent action: {joined}")
