from __future__ import annotations

import asyncio
import re
from pathlib import Path

from snorkel_g_agent.schema import AgentAction, ToolResult


def _safe_path(workdir: Path, user_path: str) -> Path:
    root = workdir.resolve()
    path = (root / user_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path escapes workdir: {user_path}")
    return path


def _truncate(content: str, limit: int) -> tuple[str, bool]:
    if len(content) <= limit:
        return content, False
    head = content[: limit // 2]
    tail = content[-limit // 2 :]
    return f"{head}\n\n... <tool output truncated> ...\n\n{tail}", True


def _whitespace_flexible_pattern(text: str) -> str:
    parts = re.split(r"(\s+)", text)
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)


class ToolExecutor:
    def __init__(self, workdir: Path, default_timeout: int, max_output_chars: int) -> None:
        self.workdir = workdir.resolve()
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars

    async def run(self, action: AgentAction) -> ToolResult:
        if action.action == "exec":
            return await self._exec(action)
        if action.action == "read_file":
            return self._read_file(action)
        if action.action == "write_file":
            return self._write_file(action, append=False)
        if action.action == "append_file":
            return self._write_file(action, append=True)
        if action.action == "replace_in_file":
            return self._replace_in_file(action)
        if action.action == "finish":
            return ToolResult(
                ok=True,
                content=f"Finished: {action.summary or ''}\nTests: {action.tests or 'not stated'}",
            )
        raise ValueError(f"unsupported action: {action.action}")

    async def _exec(self, action: AgentAction) -> ToolResult:
        if not action.cmd:
            return ToolResult(ok=False, content="exec action missing cmd")
        timeout = action.timeout_seconds or self.default_timeout
        process = await asyncio.create_subprocess_shell(
            action.cmd,
            cwd=self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()
        content = (
            f"$ {action.cmd}\n"
            f"exit_code={process.returncode}\n\n"
            f"STDOUT:\n{stdout.decode(errors='replace')}\n\n"
            f"STDERR:\n{stderr.decode(errors='replace')}"
        )
        if timed_out:
            content = f"Timed out after {timeout}s.\n\n{content}"
        truncated_content, truncated = _truncate(content, self.max_output_chars)
        return ToolResult(
            ok=(process.returncode == 0 and not timed_out),
            content=truncated_content,
            exit_code=process.returncode,
            timed_out=timed_out,
            truncated=truncated,
            extra={"timeout_seconds": timeout},
        )

    def _read_file(self, action: AgentAction) -> ToolResult:
        if not action.path:
            return ToolResult(ok=False, content="read_file action missing path")
        try:
            path = _safe_path(self.workdir, action.path)
            content = path.read_text(encoding="utf-8", errors="replace")
            truncated_content, truncated = _truncate(content, self.max_output_chars)
            return ToolResult(
                ok=True,
                content=f"READ {action.path}\n\n{truncated_content}",
                truncated=truncated,
            )
        except Exception as exc:
            return ToolResult(ok=False, content=f"read_file failed: {exc}")

    def _write_file(self, action: AgentAction, append: bool) -> ToolResult:
        if not action.path:
            return ToolResult(ok=False, content=f"{action.action} action missing path")
        if action.content is None:
            return ToolResult(ok=False, content=f"{action.action} action missing content")
        try:
            path = _safe_path(self.workdir, action.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(action.content)
            verb = "APPENDED" if append else "WROTE"
            return ToolResult(
                ok=True,
                content=f"{verb} {action.path} ({len(action.content)} chars)",
            )
        except Exception as exc:
            return ToolResult(ok=False, content=f"{action.action} failed: {exc}")

    def _replace_in_file(self, action: AgentAction) -> ToolResult:
        if not action.path:
            return ToolResult(ok=False, content="replace_in_file action missing path")
        if action.find is None:
            return ToolResult(ok=False, content="replace_in_file action missing find")
        if action.replacement is None:
            return ToolResult(ok=False, content="replace_in_file action missing replacement")
        try:
            path = _safe_path(self.workdir, action.path)
            original = path.read_text(encoding="utf-8")
            count = 1 if action.count is None else action.count

            if action.regex:
                pattern = action.find
                matches = list(re.finditer(pattern, original, flags=re.MULTILINE | re.DOTALL))
                if count > 0 and len(matches) != count:
                    return ToolResult(
                        ok=False,
                        content=(
                            f"replace_in_file expected {count} regex match(es) but found "
                            f"{len(matches)} in {action.path}"
                        ),
                        extra={"matches": len(matches), "expected": count},
                    )
                if count == 0 and not matches:
                    return ToolResult(
                        ok=False,
                        content=f"replace_in_file found 0 regex matches in {action.path}",
                        extra={"matches": 0},
                    )
                new_text, replacements = re.subn(
                    pattern,
                    action.replacement,
                    original,
                    count=0 if count == 0 else count,
                    flags=re.MULTILINE | re.DOTALL,
                )
            elif action.whitespace_flexible:
                pattern = _whitespace_flexible_pattern(action.find)
                matches = list(re.finditer(pattern, original, flags=re.MULTILINE | re.DOTALL))
                if count > 0 and len(matches) != count:
                    return ToolResult(
                        ok=False,
                        content=(
                            f"replace_in_file expected {count} whitespace-flexible match(es) "
                            f"but found {len(matches)} in {action.path}"
                        ),
                        extra={"matches": len(matches), "expected": count},
                    )
                if count == 0 and not matches:
                    return ToolResult(
                        ok=False,
                        content=(
                            "replace_in_file found 0 whitespace-flexible matches "
                            f"in {action.path}"
                        ),
                        extra={"matches": 0},
                    )
                new_text, replacements = re.subn(
                    pattern,
                    lambda _match: action.replacement or "",
                    original,
                    count=0 if count == 0 else count,
                    flags=re.MULTILINE | re.DOTALL,
                )
            else:
                matches = original.count(action.find)
                if count > 0 and matches != count:
                    return ToolResult(
                        ok=False,
                        content=(
                            f"replace_in_file expected {count} literal match(es) but found "
                            f"{matches} in {action.path}"
                        ),
                        extra={"matches": matches, "expected": count},
                    )
                if count == 0 and matches == 0:
                    return ToolResult(
                        ok=False,
                        content=f"replace_in_file found 0 literal matches in {action.path}",
                        extra={"matches": 0},
                    )
                new_text = original.replace(
                    action.find,
                    action.replacement,
                    -1 if count == 0 else count,
                )
                replacements = matches if count == 0 else count

            path.write_text(new_text, encoding="utf-8")
            return ToolResult(
                ok=True,
                content=f"REPLACED {replacements} match(es) in {action.path}",
                extra={"replacements": replacements},
            )
        except re.error as exc:
            return ToolResult(ok=False, content=f"replace_in_file regex failed: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, content=f"replace_in_file failed: {exc}")
