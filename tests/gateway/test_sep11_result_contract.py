
"""SEP11 RESULT CONTRACT regression: no /q item may be acked without a real
completed turn result. Provider-failure exits that still produce completed=True
(missing keys, all retries exhausted, local processing error, synthetic
compaction fallback, max-iterations fallback) must leave the item queued.
Also covers the gateway run_sync() passthrough of turn_exit_reason (the field
the ack predicate depends on) and live-spool behavior on a tmp state file.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path("/usr/local/lib/hermes-agent")
sys.path.insert(0, str(REPO))

from gateway import queued_prompt_spool as spool  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
from gateway.platforms.base import MessageEvent  # noqa: E402
from gateway.session import SessionSource  # noqa: E402


def _ok_text_response(text="Done — report attached."):
    return {
        "final_response": text,
        "completed": True,
        "failed": False,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": "text_response(finish_reason=stop)",
    }


def test_sanitized_provider_error_with_text_response_must_not_ack():
    # Gateway sanitization can make an exhausted-pool failure look like a
    # normal delivered text response. It must stay durable/pending for retry.
    result = _ok_text_response("Provider error:\n\nAPI call failed after 5 retries: 502 All keys exhausted")
    assert spool.queued_turn_succeeded(result) is False


def test_sanitized_provider_error_is_retryable():
    result = _ok_text_response("Provider error:\n\nHTTP 502: All keys exhausted")
    assert spool.queued_turn_should_retry(result) is True


def test_provider_error_still_completed_must_not_ack():
    # The sep11 incident shape: no keys in pool -> retries exhausted ->
    # loop exited without a model response, final_response synthesized.
    result = {
        "final_response": "I apologize, but I encountered repeated errors: "
                          "502 All keys exhausted",
        "completed": True,          # legacy ack treated this as success
        "failed": False,
        "interrupted": False,
        "partial": False,
        "turn_exit_reason": "all_retries_exhausted_no_response",
    }
    assert spool.queued_turn_succeeded(result) is False


@pytest.mark.parametrize("result", [
    # synthetic compaction fallback (the 13:26 disappearance class)
    {"final_response": "handoff", "completed": True, "failed": False,
     "turn_exit_reason": "compaction_handoff_not_actionable"},
    # max-iterations fallback (summary request may still fail later)
    {"final_response": "summary", "completed": True, "failed": False,
     "turn_exit_reason": "max_iterations_reached(10/10)"},
    # local processing error finalized with an apology
    {"final_response": "I apologize, but I encountered an error while "
                       "processing the model response: boom",
     "completed": True, "failed": False,
     "turn_exit_reason": "local_processing_error(unmarshal)"},
    # interrupted turn
    {"final_response": "partial answer", "completed": True, "failed": False,
     "interrupted": True,
     "turn_exit_reason": "interrupted_by_user"},
    # invalid tool calls -> partial
    {"final_response": "fragment", "completed": True, "failed": False,
     "partial": True,
     "turn_exit_reason": "text_response(finish_reason=stop)"},
    # explicitly failed
    {"final_response": None, "completed": False, "failed": True,
     "turn_exit_reason": "session_persistence_failed"},
    # empty final response despite text_response reason
    {"final_response": "", "completed": True, "failed": False,
     "turn_exit_reason": "text_response(finish_reason=stop)"},
    # whitespace-only response
    {"final_response": "   \n  ", "completed": True, "failed": False,
     "turn_exit_reason": "text_response(finish_reason=stop)"},
    # missing provenance entirely (proxy-style dicts) must not ack
    {"final_response": "looks fine", "completed": True, "failed": False},
    # non-dict payloads
    None,
    "completed",
    42,
])
def test_non_success_terminal_states_must_not_ack(result):
    assert spool.queued_turn_succeeded(result) is False


@pytest.mark.parametrize("reason", [
    "text_response(finish_reason=stop)",
    "text_response(finish_reason=length)",
    "text_response(finish_reason=tool_calls)",
])
def test_real_text_response_acks(reason):
    result = _ok_text_response()
    result["turn_exit_reason"] = reason
    assert spool.queued_turn_succeeded(result) is True


def test_durable_item_survives_failed_turn_and_acks_only_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    monkeypatch.setattr(spool, "_audit_path", lambda: tmp_path / "queued_prompts.audit.jsonl")
    item_id = spool.enqueue(
        session_key="agent:main:telegram:dm:210540672",
        text="do the thing",
        source={"platform": "telegram", "chat_id": "210540672"},
    )
    assert spool.pending() and spool.pending()[0]["id"] == item_id

    # provider-error turn (completed=True, legacy shape) must NOT drain queue
    assert spool.queued_turn_succeeded({
        "completed": True, "failed": False,
        "turn_exit_reason": "all_retries_exhausted_no_response",
    }) is False
    assert [i["id"] for i in spool.pending()] == [item_id]

    # real success acks and drains
    spool.ack(item_id)  # ack is only called by gateway after predicate passes
    assert spool.pending() == []


def test_failed_provider_error_remains_retryable():
    """The runner must classify the failed=True envelope before its success gate."""
    result = {
        "final_response": "Provider error:\n\nHTTP 502: All keys exhausted",
        "completed": False,
        "failed": True,
        "turn_exit_reason": "all_retries_exhausted_no_response",
    }
    assert spool.queued_turn_should_retry(result) is True
    assert spool.queued_turn_succeeded(result) is False


def test_run_sync_result_passthrough_preserves_turn_exit_reason():
    # The reconstructed result dict in gateway/run.py (run_sync) must carry
    # turn_exit_reason through, or the ack predicate can never fire.
    src = (REPO / "gateway" / "run.py").read_text()
    # 1) passthrough exists inside the reconstructed dict (after result_holder)
    assert '"turn_exit_reason": (' in src, "run_sync passthrough missing"
    # 2) source dict in turn_finalizer carries the field
    fin = (REPO / "agent" / "turn_finalizer.py").read_text()
    assert '"turn_exit_reason": _turn_exit_reason,' in fin
    # 3) the ack call site actually consults the predicate
    assert "queued_turn_succeeded(followup_result)" in src


def test_restore_preserves_original_message_id_for_reply_anchor(tmp_path, monkeypatch):
    """Restart recovery must keep Telegram's clickable task-message anchor."""
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    monkeypatch.setattr(spool, "_audit_path", lambda: tmp_path / "queued_prompts.audit.jsonl")
    item_id = spool.enqueue(
        session_key="agent:main:telegram:dm:210540672",
        text="task from Telegram",
        source={
            "platform": "telegram",
            "chat_id": "210540672",
            "chat_type": "dm",
            "message_id": "123456",
        },
    )

    class Adapter:
        _pending_messages = {}

    class Runner:
        def __init__(self):
            self.event = None

        def _adapter_for_source(self, source):
            return Adapter()

        def _enqueue_fifo(self, session_key, event, adapter):
            self.event = event

    runner = Runner()
    assert spool.restore_into_runner(runner) == 1
    assert runner.event.message_id == "123456"
    assert runner.event.source.message_id == "123456"
    assert runner.event.metadata["queue_id"] == item_id


def test_restore_is_idempotent_and_does_not_duplicate_live_projection(tmp_path, monkeypatch):
    """Repeated restore must not turn two durable items into four live ones."""
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    monkeypatch.setattr(spool, "_audit_path", lambda: tmp_path / "queued_prompts.audit.jsonl")
    for n in (1, 2):
        spool.enqueue(
            session_key="agent:main:telegram:dm:210540672",
            text=f"task {n}",
            source={"platform": "telegram", "chat_id": "210540672", "message_id": str(n)},
        )

    class Adapter:
        _pending_messages = {}

    class Runner:
        def __init__(self):
            self.events = []

        def _adapter_for_source(self, source):
            return Adapter()

        def _enqueue_fifo(self, session_key, event, adapter):
            self.events.append(event)

    runner = Runner()
    assert spool.restore_into_runner(runner) == 2
    assert spool.restore_into_runner(runner) == 0
    assert len(runner.events) == 2


def test_queued_delivery_args_use_item_anchor_not_outer_anchor():
    runner = object.__new__(GatewayRunner)
    source = SessionSource.from_dict({
        "platform": "telegram", "chat_id": "210540672", "message_id": "outer",
    })
    queued_source = SessionSource.from_dict({
        "platform": "telegram", "chat_id": "210540672", "message_id": "item",
    })
    event = MessageEvent(
        text="queued task", source=queued_source, message_id="item",
        metadata={"queue_id": "q-1", "queue_reply_anchor": "item"},
    )
    actual_source, metadata, anchor = runner._queued_delivery_args(
        event, source=source, event_message_id="outer",
    )
    assert actual_source is queued_source
    assert anchor == "item"
    assert metadata["queue_id"] == "q-1"
    assert metadata.get("telegram_reply_to_message_id") != "outer"


def test_queued_streaming_uses_durable_item_anchor():
    src = (REPO / "gateway" / "run.py").read_text()
    marker = "_queued_item_metadata = getattr(pending_event, \"metadata\", None) or {}"
    start = src.index(marker)
    block = src[start:start + 900]
    assert 'get("queue_reply_anchor")' in block
    assert "or self._reply_anchor_for_event(pending_event)" in block
    assert "event_message_id=next_message_id" in src[start:]


def test_queued_delivery_args_cannot_be_overwritten_by_legacy_item_route():
    runner = object.__new__(GatewayRunner)
    source = SessionSource.from_dict({
        "platform": "telegram", "chat_id": "210540672", "message_id": "outer",
    })
    queued_source = SessionSource.from_dict({
        "platform": "telegram", "chat_id": "210540672", "message_id": "item",
    })
    event = MessageEvent(
        text="queued task", source=queued_source, message_id="item",
        metadata={
            "queue_id": "q-2",
            "queue_reply_anchor": "item",
            "telegram_reply_to_message_id": "neighbour",
            "reply_to_message_id": "neighbour",
            "message_id": "neighbour",
        },
    )
    _, metadata, anchor = runner._queued_delivery_args(
        event, source=source, event_message_id="outer",
    )
    assert anchor == "item"
    assert metadata["queue_reply_anchor"] == "item"
    assert metadata.get("telegram_reply_to_message_id") != "neighbour"
    assert metadata.get("reply_to_message_id") != "neighbour"
    assert metadata.get("message_id") != "neighbour"


    # sanity: current live spool parses; audit file readable; no content leak
    live = json.loads(Path("/root/.hermes/queued_prompts.json").read_text())
    assert isinstance(live, list)
    for item in live:
        assert {"id", "session_key", "text"} <= set(item.keys())


def test_outer_handler_retains_durable_queue_event_before_recursive_drain():
    """Any pre-drain exception must restore a queue item to the live FIFO."""
    src = (REPO / "gateway" / "run.py").read_text()
    marker = "except TurnLeaseTimeoutError as exc:"
    start = src.index(marker)
    end = src.index("    def _restore_moa_one_shot", start)
    block = src[start:end]
    assert "except BaseException:" in block
    assert "_retain_queued_event_after_abort(" in block
    assert '_queued_meta.get("queue_id")' in block
