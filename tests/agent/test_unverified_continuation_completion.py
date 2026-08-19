"""Regression: a phase-completion claim after a continuation needs tool evidence."""

from agent.agent_runtime_helpers import is_unverified_continuation_completion


CONTINUE = {
    "role": "user",
    "content": "[System: Continue now. Execute the required tool calls and only send your final answer after completing the task.]",
    "_intent_ack_continuation_synthetic": True,
}


def test_detects_phase_completion_after_synthetic_continuation_without_tool():
    messages = [
        {"role": "user", "content": "Делай K"},
        {"role": "assistant", "content": "Phase K запущен."},
        CONTINUE,
    ]

    assert is_unverified_continuation_completion(
        messages,
        "**Phase K завершён.** Я выполнил все шаги и получил реальный результат.",
    )


def test_allows_completion_after_a_tool_result():
    messages = [
        {"role": "user", "content": "Делай K"},
        {"role": "assistant", "content": "Phase K запущен."},
        CONTINUE,
        {"role": "tool", "content": '{"exit_code": 0, "output": "build passed"}'},
    ]

    assert not is_unverified_continuation_completion(
        messages,
        "**Phase K завершён.** Я выполнил все шаги и получил реальный результат.",
    )


def test_does_not_block_non_completion_reply_after_continuation():
    assert not is_unverified_continuation_completion(
        [{"role": "user", "content": "continue", "_intent_ack_continuation_synthetic": True}],
        "Нужна ваша авторизация в Telegram, чтобы продолжить.",
    )


def test_detects_english_completion_claim_too():
    assert is_unverified_continuation_completion(
        [CONTINUE],
        "Phase K complete. I completed all steps and production verification passed.",
    )
