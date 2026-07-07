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


def test_parse_tool_call_wrapper_json_action() -> None:
    action = parse_action(
        'Let me inspect.\n<tool_call>{"action":"exec","cmd":"pwd","timeout_seconds":10}</tool_call>'
    )

    assert action.action == "exec"
    assert action.cmd == "pwd"
    assert action.timeout_seconds == 10


def test_parse_tool_call_missing_opening_brace() -> None:
    action = parse_action(
        "I need a note."
        '<tool_call>action":"scratchpad","title":"Finding","content":"looping"}</tool_call>'
    )

    assert action.action == "scratchpad"
    assert action.title == "Finding"
    assert action.content == "looping"


def test_parse_unclosed_tool_call_missing_opening_brace() -> None:
    action = parse_action(
        'Trying direct SQLi now.<tool_call>action":"exec","cmd":"echo test","timeout_seconds":10}'
    )

    assert action.action == "exec"
    assert action.cmd == "echo test"


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


def test_parse_replace_in_file_accepts_common_aliases() -> None:
    action = parse_action(
        '{"action":"replace_in_file","path":"main.go","find":"old","replace":"new"}'
    )
    alternate = parse_action(
        '{"action":"replace_in_file","path":"main.go","old_string":"left","new_string":"right"}'
    )

    assert action.replacement == "new"
    assert alternate.find == "left"
    assert alternate.replacement == "right"


def test_parse_exec_action_from_tool_and_command_aliases() -> None:
    action = parse_action('{"tool":"bash","command":"cat /app/STATE_FILE.md"}')

    assert action.action == "exec"
    assert action.cmd == "cat /app/STATE_FILE.md"


def test_parse_exec_action_from_shell_alias() -> None:
    action = parse_action('{"action":"shell","cmd":"ls -la"}')

    assert action.action == "exec"
    assert action.cmd == "ls -la"


def test_parse_exec_action_from_terminal_command_alias() -> None:
    action = parse_action('{"action":"terminal","command":"pwd"}')

    assert action.action == "exec"
    assert action.cmd == "pwd"


def test_parse_exec_action_joins_list_command() -> None:
    action = parse_action('{"action":"exec","cmd":["ls","-la","/app"]}')

    assert action.action == "exec"
    assert action.cmd == "ls -la /app"


def test_parse_exec_action_from_native_tool_call_arguments() -> None:
    action = parse_action(
        '{"name":"bash","arguments":{"command":["python","-m","pytest","-q"]}}'
    )

    assert action.action == "exec"
    assert action.cmd == "python -m pytest -q"


def test_parse_exec_action_from_bare_command_object() -> None:
    action = parse_action('<tool_call>exec\n{"cmd":"go test ./..."}')

    assert action.action == "exec"
    assert action.cmd == "go test ./..."


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


def test_parse_repairs_literal_control_chars_in_json_string() -> None:
    action = parse_action(
        """{
          "action": "write_file",
          "path": "main.go",
          "content": "package main
	func main() {
	}
"
        }"""
    )

    assert action.action == "write_file"
    assert action.content == "package main\n\tfunc main() {\n\t}\n"


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
