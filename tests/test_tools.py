from pathlib import Path

import pytest

from snorkel_g_agent.schema import AgentAction
from snorkel_g_agent.tools import ToolExecutor


@pytest.mark.asyncio
async def test_exec_runs_in_workdir(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(AgentAction(action="exec", cmd="pwd"))

    assert result.ok
    assert str(tmp_path) in result.content


@pytest.mark.asyncio
async def test_write_and_read_file(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    written = await executor.run(
        AgentAction(action="write_file", path="nested/answer.txt", content="ready")
    )
    read = await executor.run(AgentAction(action="read_file", path="nested/answer.txt"))

    assert written.ok
    assert read.ok
    assert "ready" in read.content


@pytest.mark.asyncio
async def test_list_files_and_search_text(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "item.go").write_text("func main() {\n\tMonthYear()\n}\n")
    (tmp_path / "pkg" / "item.txt").write_text("MonthYear in docs\n")
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    listed = await executor.run(AgentAction(action="list_files", glob="**/*.go"))
    searched = await executor.run(
        AgentAction(
            action="search_text",
            pattern="MonthYear",
            glob="**/*.go",
            context_lines=1,
        )
    )

    assert listed.ok
    assert "pkg/item.go" in listed.content
    assert "pkg/item.txt" not in listed.content
    assert searched.ok
    assert "pkg/item.go:2:>" in searched.content
    assert "func main" in searched.content


@pytest.mark.asyncio
async def test_scratchpad_appends_to_state_file(tmp_path: Path) -> None:
    (tmp_path / "STATE_FILE.md").write_text("# STATE_FILE\n")
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(
        AgentAction(
            action="scratchpad",
            title="Evidence",
            content="Task 3 needs whitespace-flexible edits.",
        )
    )

    assert result.ok
    state = (tmp_path / "STATE_FILE.md").read_text()
    assert "## Evidence" in state
    assert "whitespace-flexible" in state


@pytest.mark.asyncio
async def test_rejects_path_escape(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(AgentAction(action="read_file", path="../secret.txt"))

    assert not result.ok
    assert "escapes workdir" in result.content


@pytest.mark.asyncio
async def test_replace_in_file_supports_whitespace_flexible_match(tmp_path: Path) -> None:
    target = tmp_path / "item.go"
    target.write_text(
        "func convert() {\n"
        "\tif field.Type == MonthYear {\n"
        "\t\treturn value\n"
        "\t}\n"
        "}\n"
    )
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(
        AgentAction(
            action="replace_in_file",
            path="item.go",
            find="if field.Type == MonthYear {\n    return value\n}",
            replacement=(
                "if field.Type == MonthYear {\n"
                "\t\tparts := strings.Split(value, \"/\")\n"
                "\t\treturn parts[1] + parts[0]\n"
                "\t}"
            ),
            whitespace_flexible=True,
        )
    )

    assert result.ok
    assert "parts := strings.Split" in target.read_text()


@pytest.mark.asyncio
async def test_replace_in_file_rejects_ambiguous_literal_match(tmp_path: Path) -> None:
    target = tmp_path / "main.go"
    target.write_text("return nil\nreturn nil\n")
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(
        AgentAction(
            action="replace_in_file",
            path="main.go",
            find="return nil",
            replacement="return err",
        )
    )

    assert not result.ok
    assert "expected 1 literal match(es) but found 2" in result.content


@pytest.mark.asyncio
async def test_replace_in_file_can_replace_all_matches(tmp_path: Path) -> None:
    target = tmp_path / "main.go"
    target.write_text("return nil\nreturn nil\n")
    executor = ToolExecutor(tmp_path, default_timeout=5, max_output_chars=2000)

    result = await executor.run(
        AgentAction(
            action="replace_in_file",
            path="main.go",
            find="return nil",
            replacement="return err",
            count=0,
        )
    )

    assert result.ok
    assert target.read_text() == "return err\nreturn err\n"
