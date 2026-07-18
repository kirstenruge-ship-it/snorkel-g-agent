from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import signal
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


def _find_spans(text: str, needle: str, *, whitespace_flexible: bool) -> list[tuple[int, int]]:
    if whitespace_flexible:
        pattern = _whitespace_flexible_pattern(needle)
        return [
            (match.start(), match.end())
            for match in re.finditer(pattern, text, flags=re.MULTILINE | re.DOTALL)
        ]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found == -1:
            return spans
        spans.append((found, found + len(needle)))
        start = found + max(len(needle), 1)


def _effective_timeout(cmd: str, requested: int | None, default_timeout: int) -> int:
    if requested:
        return requested
    inspection_commands = {
        "rg",
        "grep",
        "find",
        "ls",
        "sed",
        "awk",
        "head",
        "tail",
        "cat",
        "pwd",
        "wc",
    }
    for segment in re.split(r"\s*(?:&&|;)\s*", cmd):
        segment = segment.strip()
        if not segment:
            continue
        try:
            parts = shlex.split(segment)
        except ValueError:
            return default_timeout
        if not parts:
            continue
        executable = Path(parts[0]).name
        if executable in {"cd", "export"}:
            continue
        if executable in inspection_commands:
            return min(default_timeout, 60)
        return default_timeout
    return default_timeout


class ToolExecutor:
    def __init__(self, workdir: Path, default_timeout: int, max_output_chars: int) -> None:
        self.workdir = workdir.resolve()
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars

    async def run(self, action: AgentAction) -> ToolResult:
        if action.action == "exec":
            return await self._exec(action)
        if action.action == "list_files":
            return self._list_files(action)
        if action.action == "search_text":
            return self._search_text(action)
        if action.action == "scratchpad":
            return self._scratchpad(action)
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
        timeout = _effective_timeout(action.cmd, action.timeout_seconds, self.default_timeout)
        process = await asyncio.create_subprocess_shell(
            action.cmd,
            cwd=self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
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
            extra={
                "timeout_seconds": timeout,
                "failure_kind": "timeout" if timed_out else None,
                "process_group_terminated": timed_out,
            },
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

    def _list_files(self, action: AgentAction) -> ToolResult:
        try:
            base = _safe_path(self.workdir, action.path or ".")
            if not base.exists():
                return ToolResult(
                    ok=False,
                    content=f"list_files path does not exist: {action.path}",
                )
            pattern = action.glob or "**/*"
            files = [
                path.relative_to(self.workdir).as_posix()
                for path in sorted(base.glob(pattern))
                if path.is_file()
            ]
            limited = files[: action.max_results]
            truncated = len(files) > len(limited)
            suffix = (
                f"\n... truncated {len(files) - len(limited)} more file(s)"
                if truncated
                else ""
            )
            return ToolResult(
                ok=True,
                content="\n".join(limited) + suffix,
                truncated=truncated,
                extra={"matches": len(files), "returned": len(limited)},
            )
        except Exception as exc:
            return ToolResult(ok=False, content=f"list_files failed: {exc}")

    def _search_text(self, action: AgentAction) -> ToolResult:
        if not action.pattern:
            return ToolResult(ok=False, content="search_text action missing pattern")
        try:
            base = _safe_path(self.workdir, action.path or ".")
            if not base.exists():
                return ToolResult(
                    ok=False,
                    content=f"search_text path does not exist: {action.path}",
                )
            regex = re.compile(action.pattern)
            file_pattern = action.glob or "**/*"
            results: list[str] = []
            for path in sorted(base.glob(file_pattern)):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                lines = text.splitlines()
                for idx, line in enumerate(lines, start=1):
                    if not regex.search(line):
                        continue
                    start = max(1, idx - action.context_lines)
                    end = min(len(lines), idx + action.context_lines)
                    if action.context_lines:
                        for line_no in range(start, end + 1):
                            prefix = ">" if line_no == idx else " "
                            results.append(
                                f"{path.relative_to(self.workdir).as_posix()}:{line_no}:{prefix}"
                                f"{lines[line_no - 1]}"
                            )
                    else:
                        results.append(f"{path.relative_to(self.workdir).as_posix()}:{idx}:{line}")
                    if len(results) >= action.max_results:
                        content = "\n".join(results) + "\n... truncated more match(es)"
                        return ToolResult(
                            ok=True,
                            content=content,
                            truncated=True,
                            extra={"returned": len(results)},
                        )
            return ToolResult(
                ok=True,
                content="\n".join(results) if results else "No matches",
                extra={"returned": len(results)},
            )
        except re.error as exc:
            return ToolResult(ok=False, content=f"search_text regex failed: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, content=f"search_text failed: {exc}")

    def _scratchpad(self, action: AgentAction) -> ToolResult:
        if action.content is None:
            return ToolResult(ok=False, content="scratchpad action missing content")
        try:
            path = _safe_path(self.workdir, "STATE_FILE.md")
            title = action.title or "Note"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n\n## {title}\n\n{action.content.strip()}\n")
            return ToolResult(
                ok=True,
                content=f"UPDATED scratchpad {path.relative_to(self.workdir).as_posix()}",
            )
        except Exception as exc:
            return ToolResult(ok=False, content=f"scratchpad failed: {exc}")

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
            target_text = original
            target_offset = 0

            if action.within is not None:
                within_spans = _find_spans(
                    original,
                    action.within,
                    whitespace_flexible=action.whitespace_flexible,
                )
                if len(within_spans) != 1:
                    return ToolResult(
                        ok=False,
                        content=(
                            "replace_in_file expected 1 context block match but found "
                            f"{len(within_spans)} in {action.path}"
                        ),
                        extra={"context_matches": len(within_spans)},
                    )
                target_offset, target_end = within_spans[0]
                target_text = original[target_offset:target_end]

            if action.regex:
                pattern = action.find
                matches = list(re.finditer(pattern, target_text, flags=re.MULTILINE | re.DOTALL))
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
                    target_text,
                    count=0 if count == 0 else count,
                    flags=re.MULTILINE | re.DOTALL,
                )
            elif action.whitespace_flexible:
                pattern = _whitespace_flexible_pattern(action.find)
                matches = list(re.finditer(pattern, target_text, flags=re.MULTILINE | re.DOTALL))
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
                    target_text,
                    count=0 if count == 0 else count,
                    flags=re.MULTILINE | re.DOTALL,
                )
            else:
                matches = target_text.count(action.find)
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
                new_text = target_text.replace(
                    action.find,
                    action.replacement,
                    -1 if count == 0 else count,
                )
                replacements = matches if count == 0 else count

            if action.within is not None:
                new_text = (
                    original[:target_offset]
                    + new_text
                    + original[target_offset + len(target_text) :]
                )
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
