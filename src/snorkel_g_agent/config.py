from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from snorkel_g_agent.schema import AppConfig, RouteConfig

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if match:
            return os.environ.get(match.group(1), "")
    if isinstance(value, dict):
        return {key: _expand_env_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(child) for child in value]
    return value


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    expanded = _expand_env_value(raw)
    config = AppConfig.model_validate(expanded)
    if not config.agent.system_prompt_path.is_absolute():
        config.agent.system_prompt_path = (path.parent / config.agent.system_prompt_path).resolve()
    return config


def resolve_route(config: AppConfig, route_name: str | None = None) -> RouteConfig:
    selected = route_name or config.run.default_route
    try:
        route = config.routes[selected]
    except KeyError as exc:
        available = ", ".join(sorted(config.routes))
        raise ValueError(f"unknown route {selected!r}; available routes: {available}") from exc
    if not route.base_url:
        raise ValueError(f"route {selected!r} has an empty base_url")
    if route.provider == "modal":
        missing = [
            env_name
            for env_name in [route.modal_key_env, route.modal_secret_env]
            if not os.environ.get(env_name)
        ]
        if missing:
            joined = ", ".join(f"${name}" for name in missing)
            raise ValueError(f"route {selected!r} requires {joined}")
    elif route.api_key_env is None or not os.environ.get(route.api_key_env):
        raise ValueError(f"route {selected!r} requires ${route.api_key_env}")
    return route


def route_headers(route: RouteConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if route.provider == "modal":
        headers["Modal-Key"] = os.environ[route.modal_key_env]
        headers["Modal-Secret"] = os.environ[route.modal_secret_env]
    elif route.provider == "portkey":
        if route.api_key_env is None:
            raise ValueError("portkey routes require api_key_env")
        headers["x-portkey-api-key"] = os.environ[route.api_key_env]
    else:
        if route.api_key_env is None:
            raise ValueError("openai-compatible routes require api_key_env")
        headers["Authorization"] = f"Bearer {os.environ[route.api_key_env]}"
    headers.update(route.headers)
    return headers
