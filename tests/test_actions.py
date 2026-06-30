import pytest

from snorkel_g_agent.actions import ActionParseError, parse_action


def test_parse_fenced_json_action() -> None:
    action = parse_action('```json\n{"action":"exec","cmd":"pytest -q"}\n```')

    assert action.action == "exec"
    assert action.cmd == "pytest -q"


def test_parse_loose_json_action() -> None:
    action = parse_action('Sure.\n{"action":"finish","summary":"done","tests":"pytest"}\nbye')

    assert action.action == "finish"
    assert action.summary == "done"


def test_parse_replace_in_file_action() -> None:
    action = parse_action(
        """
        {
          "action": "replace_in_file",
          "path": "main.go",
          "find": "if x {\\n    return nil\\n}",
          "within": "func target() {\\n    if x {\\n        return nil\\n    }\\n}",
          "replacement": "if x {\\n\\treturn value\\n}",
          "whitespace_flexible": true
        }
        """
    )

    assert action.action == "replace_in_file"
    assert action.whitespace_flexible is True
    assert action.count == 1
    assert action.within is not None


def test_parse_search_text_action() -> None:
    action = parse_action(
        '{"action":"search_text","pattern":"MonthYear","glob":"**/*.go","context_lines":2}'
    )

    assert action.action == "search_text"
    assert action.pattern == "MonthYear"
    assert action.glob == "**/*.go"
    assert action.context_lines == 2


def test_parse_repairs_invalid_json_escape_in_shell_command() -> None:
    action = parse_action(
        r'''{
          "action": "exec",
          "cmd": "grep -n 'testuser\|GetItemsByTitle' pkg/onepassword/items.go"
        }'''
    )

    assert action.action == "exec"
    assert action.cmd == r"grep -n 'testuser\|GetItemsByTitle' pkg/onepassword/items.go"


def test_parse_scratchpad_action() -> None:
    action = parse_action(
        '{"action":"scratchpad","title":"Root cause","content":"MonthYear conversion is wrong"}'
    )

    assert action.action == "scratchpad"
    assert action.title == "Root cause"
    assert "MonthYear" in (action.content or "")


def test_parse_action_rejects_missing_json() -> None:
    with pytest.raises(ActionParseError):
        parse_action("run pytest please")
