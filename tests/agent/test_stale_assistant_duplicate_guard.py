"""Regression guard: tool-less stale repeats must never reach the user."""

from agent.agent_runtime_helpers import is_stale_assistant_duplicate


def test_detects_exact_repeated_assistant_draft_after_a_user_turn():
    messages = [
        {"role": "user", "content": "implement the remaining stages"},
        {"role": "assistant", "content": "Stage B: I am implementing it now."},
        {"role": "user", "content": "continue"},
    ]

    assert is_stale_assistant_duplicate(
        messages, "Stage B: I am implementing it now."
    )


def test_does_not_treat_an_old_answer_as_a_stale_repeat_without_new_user_turn():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "A stable answer."},
    ]

    assert not is_stale_assistant_duplicate(messages, "A stable answer.")


def test_ignores_whitespace_when_comparing_stale_repeat():
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "Stage C: run the tests."},
        {"role": "user", "content": "continue"},
    ]

    assert is_stale_assistant_duplicate(messages, "  Stage C: run the tests.  \n")
