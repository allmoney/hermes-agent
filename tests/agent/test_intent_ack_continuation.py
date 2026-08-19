"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881): the historical ``codex_responses``-only path is byte-stable
under the default ``"auto"`` mode, while an explicit ``true``/model-list opt-in
extends the "you announced an action but called no tool — keep going" nudge to
every api_mode and relaxes the codebase/workspace requirement so general
autonomous workflows ("I'll run a health check on the server") are caught.

These are invariant assertions about how the mode string and the detector
gates relate, not snapshots of the marker lists.
"""

from types import SimpleNamespace
from typing import Union

from agent.agent_runtime_helpers import (
    intent_ack_continuation_enabled,
    intent_ack_continuation_mode,
    looks_like_codex_intermediate_ack,
)


def _agent(
    mode: Union[str, bool, list] = "auto",
    api_mode="chat_completions",
    model="anthropic/claude-sonnet-4",
):
    # _strip_think_blocks is a no-op for these plain-text fixtures.
    return SimpleNamespace(
        _intent_ack_continuation=mode,
        api_mode=api_mode,
        model=model,
        _strip_think_blocks=lambda c: c,
    )


# The reporter's exact repro (#27881): server-ops task, no filesystem reference.
REPRO_USER = (
    "check the current status of the server, grab the latest error logs, "
    "and let me know if there's anything critical"
)
REPRO_ACK = "I will start by running a health check command on the server to see its current status."

# The codex-coding case the detector was originally built for.
CODE_USER = "review the codebase in /app"
CODE_ACK = "Let me inspect the repository files first."


# ── mode resolution ────────────────────────────────────────────────────────




def test_true_is_all_api_modes():
    for am in ("chat_completions", "anthropic", "codex_responses"):
        assert intent_ack_continuation_mode(_agent(True, am)) == "all"
    for s in ("true", "always", "yes", "on", "ON"):
        assert intent_ack_continuation_mode(_agent(s, "chat_completions")) == "all"








def test_missing_attr_defaults_to_auto():
    bare = SimpleNamespace(api_mode="chat_completions", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare) == "off"
    bare_codex = SimpleNamespace(api_mode="codex_responses", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare_codex) == "codex_only"


def test_enabled_is_mode_not_off():
    assert intent_ack_continuation_enabled(_agent(True, "chat_completions")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "codex_responses")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "chat_completions")) is False
    assert intent_ack_continuation_enabled(_agent(False, "codex_responses")) is False


# ── detector: workspace requirement ─────────────────────────────────────────




def test_multipart_user_message_does_not_crash_on_workspace_path():
    """#9562: vision requests forward ``user_message`` as a multi-part list.

    The OpenAI-compat API server passes the raw ``content`` field straight
    through for vision turns, so ``user_message`` reaches the detector as
    ``[{type:"text",...}, {type:"image_url",...}]``. The ``require_workspace``
    path flattened it with ``(user_message or "").strip()`` — a truthy list
    survived and ``.strip()`` raised ``AttributeError``, killing the turn.
    The text part still has to drive workspace detection.
    """
    a = _agent("auto", "codex_responses")
    multipart = [
        {"type": "text", "text": CODE_USER},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    msgs = [{"role": "user", "content": multipart}]
    # No crash, and the text part ("review the codebase in /app") still
    # satisfies the workspace requirement so the ack fires.
    assert looks_like_codex_intermediate_ack(
        a, multipart, CODE_ACK, msgs, require_workspace=True
    )


def test_all_path_drops_workspace_requirement():
    """The #27881 fix: opted-in turns catch non-codebase intent acks."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


# ── detector: guardrails that hold regardless of workspace ───────────────────

def test_real_final_answer_does_not_fire():
    a = _agent(True, "chat_completions")
    final = "Done. The server is healthy and there are no critical errors in the logs."
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(a, REPRO_USER, final, msgs, require_workspace=False)


def test_conversational_reply_without_action_verb_does_not_fire():
    a = _agent(True, "chat_completions")
    brainstorm = "I'll help you think through the tradeoffs here."
    msgs = [{"role": "user", "content": "help me decide"}]
    assert not looks_like_codex_intermediate_ack(
        a, "help me decide", brainstorm, msgs, require_workspace=False
    )


def test_continues_after_a_tool_already_ran():
    """Mode B must still be caught after discovery tools in the same task."""
    a = _agent(True, "chat_completions")
    msgs = [
        {"role": "user", "content": REPRO_USER},
        {"role": "tool", "content": "health check result"},
    ]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


def test_russian_progressive_continue_after_tools_is_not_a_final_answer():
    """Regression: «Продолжаю искать …» previously bypassed the RU regex."""
    a = _agent(True, "chat_completions")
    msgs = [
        {"role": "user", "content": "Исправь полный coverage-gate"},
        {"role": "tool", "content": "parallel subset passed"},
    ]
    assert looks_like_codex_intermediate_ack(
        a,
        "Исправь полный coverage-gate",
        "Продолжаю искать конкретный тест и исправляю fixture.",
        msgs,
        require_workspace=False,
    )




def test_long_unfinished_status_report_continues_after_tools():
    """Length must not let a detailed partial report abandon remaining work."""
    a = _agent(True, "chat_completions")
    report = ("Промежуточный отчёт. " * 90) + "Одна часть ещё не закрыта полностью: нужно довести catalog-driven монитор."
    msgs = [
        {"role": "user", "content": "Выполни все три пункта до конца"},
        {"role": "tool", "content": "first implementation committed"},
    ]
    assert looks_like_codex_intermediate_ack(
        a, "Выполни все три пункта до конца", report, msgs, require_workspace=False
    )


def test_unfinished_status_report_continues_after_tools():
    """A progress report must not end a task that explicitly has follow-up work.

    The former RU detector only matched future-tense action promises.  A short
    report such as ``Дальше остаётся …`` was treated as a final answer even
    after successful tools, silently abandoning the unfinished work.
    """
    a = _agent(True, "chat_completions")
    report = "Коммит отправлен. Дальше остаётся довести Web/TMA и выполнить deploy."
    msgs = [
        {"role": "user", "content": "Восстанови регрессию до закрытия"},
        {"role": "tool", "content": "tests passed"},
    ]
    assert looks_like_codex_intermediate_ack(
        a, "Восстанови регрессию до закрытия", report, msgs, require_workspace=False
    )


def test_long_response_is_not_treated_as_an_ack():
    a = _agent(True, "chat_completions")
    long_ack = "I will run the check. " + ("x" * 1300)
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, long_ack, msgs, require_workspace=False
    )
