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


def test_parse_action_rejects_missing_json() -> None:
    with pytest.raises(ActionParseError):
        parse_action("run pytest please")
