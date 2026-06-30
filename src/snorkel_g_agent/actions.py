from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from snorkel_g_agent.schema import AgentAction

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


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


def _json_loads_with_repair(blob: str) -> dict[str, Any]:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        repaired = _INVALID_JSON_ESCAPE.sub(r"\\\\", blob)
        if repaired == blob:
            raise
        return json.loads(repaired)


def parse_action(text: str) -> AgentAction:
    errors: list[str] = []
    for blob in _candidate_json_blobs(text):
        try:
            raw = _json_loads_with_repair(blob)
            return AgentAction.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(str(exc))
    joined = "; ".join(errors[-2:]) if errors else "no JSON object found"
    raise ActionParseError(f"could not parse agent action: {joined}")
