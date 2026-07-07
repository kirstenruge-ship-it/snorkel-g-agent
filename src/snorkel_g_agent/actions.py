from __future__ import annotations

import json
import re
import shlex
from typing import Any

from pydantic import ValidationError

from snorkel_g_agent.schema import AgentAction

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)(?:</tool_call>|$)", re.DOTALL)
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


class ActionParseError(ValueError):
    pass


def _candidate_json_blobs(text: str) -> list[str]:
    candidates = [match.group(1) for match in _FENCED_JSON.finditer(text)]
    candidates.extend(_tool_call_candidates(text))
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.insert(0, stripped)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    return _dedupe(candidates)


def _tool_call_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        blob = match.group(1).strip()
        if not blob:
            continue
        candidates.append(blob)
        repaired = _repair_tool_call_blob(blob)
        if repaired != blob:
            candidates.append(repaired)
    return candidates


def _repair_tool_call_blob(blob: str) -> str:
    repaired = blob.strip()
    if repaired.startswith("{"):
        return repaired

    if repaired.startswith('"action"'):
        repaired = "{" + repaired
    elif repaired.startswith("action\""):
        repaired = "{\"" + repaired
    elif repaired.startswith("action:"):
        repaired = '{"action"' + repaired.removeprefix("action")

    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
    return repaired


def _dedupe(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _json_loads_with_repair(blob: str) -> dict[str, Any]:
    variants = [blob]
    escaped_controls = _escape_control_chars_in_strings(blob)
    if escaped_controls != blob:
        variants.append(escaped_controls)
    invalid_escape_repaired = _INVALID_JSON_ESCAPE.sub(r"\\\\", blob)
    if invalid_escape_repaired != blob:
        variants.append(invalid_escape_repaired)
    both_repaired = _INVALID_JSON_ESCAPE.sub(r"\\\\", escaped_controls)
    if both_repaired not in variants:
        variants.append(both_repaired)

    last_error: json.JSONDecodeError | None = None
    for variant in variants:
        try:
            return json.loads(variant)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("no JSON candidate", blob, 0)


def _escape_control_chars_in_strings(blob: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for char in blob:
        if not in_string:
            if char == '"':
                in_string = True
            out.append(char)
            continue

        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            out.append(char)
            escaped = True
        elif char == '"':
            out.append(char)
            in_string = False
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)

def _normalize_action_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    arguments = normalized.get("arguments")
    if isinstance(arguments, dict):
        merged = dict(arguments)
        for key, value in normalized.items():
            if key != "arguments":
                merged.setdefault(key, value)
        normalized = merged

    if "action" not in normalized:
        for key in ("tool", "name"):
            value = normalized.get(key)
            if isinstance(value, str) and value:
                normalized["action"] = value
                break

    action = normalized.get("action")
    if isinstance(action, str):
        action_aliases = {
            "bash": "exec",
            "shell": "exec",
            "sh": "exec",
            "terminal": "exec",
            "command": "exec",
            "run": "exec",
            "run_command": "exec",
            "execute": "exec",
            "execute_bash": "exec",
        }
        normalized["action"] = action_aliases.get(action.strip().lower(), action)

    if "cmd" not in normalized and "command" in normalized:
        normalized["cmd"] = normalized["command"]
    if "action" not in normalized and "cmd" in normalized:
        normalized["action"] = "exec"
    if isinstance(normalized.get("cmd"), list):
        normalized["cmd"] = shlex.join(str(part) for part in normalized["cmd"])
    if "replacement" not in normalized and "replace" in normalized:
        normalized["replacement"] = normalized["replace"]
    if "find" not in normalized and "old_string" in normalized:
        normalized["find"] = normalized["old_string"]
    if "replacement" not in normalized and "new_string" in normalized:
        normalized["replacement"] = normalized["new_string"]
    return normalized


def parse_action(text: str) -> AgentAction:
    errors: list[str] = []
    for blob in _candidate_json_blobs(text):
        try:
            raw = _json_loads_with_repair(blob)
            return AgentAction.model_validate(_normalize_action_aliases(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(str(exc))
    joined = "; ".join(errors[-2:]) if errors else "no JSON object found"
    raise ActionParseError(f"could not parse agent action: {joined}")
