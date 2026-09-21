"""SEP20 P3 split-tail rescue — unit tests for _rescue_split_turn_tail.

Covers the HANDOFF_PATCH_SEP20_SPLIT_TAIL_RESCUE surface:
- dedupe: one unserved split_tail_resume item per session_key;
- durable-only enqueue when no live adapter is available;
- live FIFO injection via _enqueue_fifo with reply anchor preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway import queued_prompt_spool


class _FakeSource:
    """Minimal SessionSource double: to_dict + message_id."""

    def __init__(self, message_id: str = "555"):
        self.message_id = message_id
        self.platform = "telegram"

    def to_dict(self) -> dict:
        return {"platform": "telegram", "message_id": self.message_id}


class _FakeAdapter:
    pass


class _RescueRunner:
    """Duck-typed GatewayRunner: only what _rescue_split_turn_tail touches."""

    def __init__(self, adapter, fifo=None):
        self._adapter = adapter
        self._fifo = fifo if fifo is not None else []

    def _adapter_for_source(self, source):
        return self._adapter

    def _enqueue_fifo(self, session_key, event, adapter):
        self._fifo.append((session_key, event, adapter))


def _make_runner(monkeypatch, tmp_path, adapter):
    """Point the spool at a temp ledger and return a fake runner."""
    monkeypatch.setattr(queued_prompt_spool, "_path",
                        lambda: tmp_path / "queued_prompts.json")
    monkeypatch.setattr(queued_prompt_spool, "_audit_path",
                        lambda: tmp_path / "audit.jsonl")
    runner = _RescueRunner(adapter)
    # Bind the REAL methods from GatewayRunner onto the fake: this tests the
    # production implementation, not a copy of it.
    from gateway.run import GatewayRunner
    runner._rescue_split_turn_tail = GatewayRunner._rescue_split_turn_tail.__get__(runner)
    runner._split_tail_resume_prompt = GatewayRunner._split_tail_resume_prompt.__get__(runner)
    return runner


def test_live_adapter_injects_fifo_event(monkeypatch, tmp_path):
    runner = _make_runner(monkeypatch, tmp_path, _FakeAdapter())
    ok = runner._rescue_split_turn_tail(
        session_key="agent:main:telegram:dm:1",
        new_session_id="20260920_x",
        source=_FakeSource("555"),
    )
    assert ok is True
    assert len(runner._fifo) == 1
    session_key, event, adapter = runner._fifo[0]
    assert event.metadata.get("split_tail_resume") is True
    assert event.metadata["split_tail_resume_session"] == "20260920_x"
    assert event.message_id == "555"  # reply anchor preserved
    assert "восстановление прерванного хода" in event.text


def test_no_adapter_enqueues_durable_only(monkeypatch, tmp_path):
    runner = _make_runner(monkeypatch, tmp_path, None)
    ok = runner._rescue_split_turn_tail(
        session_key="agent:main:telegram:dm:1",
        new_session_id="20260920_x",
        source=_FakeSource("556"),
    )
    assert ok is True
    items = queued_prompt_spool.pending()
    assert len(items) == 1
    item = items[0]
    assert item["metadata"]["split_tail_resume"] is True
    assert item["session_key"] == "agent:main:telegram:dm:1"
    # Reply anchor persisted as metadata (Telegram route).
    assert item["metadata"]["queue_reply_anchor"] == "556"


def test_dedupe_one_resume_per_session(monkeypatch, tmp_path):
    runner = _make_runner(monkeypatch, tmp_path, None)
    src = _FakeSource("557")
    first = runner._rescue_split_turn_tail(
        session_key="agent:main:telegram:dm:1",
        new_session_id="20260920_a",
        source=src,
    )
    assert first is True
    # Second call with an unserved item pending: must NOT enqueue another.
    second = runner._rescue_split_turn_tail(
        session_key="agent:main:telegram:dm:1",
        new_session_id="20260920_b",
        source=src,
    )
    assert second is False
    items = [i for i in queued_prompt_spool.pending()
             if i["metadata"].get("split_tail_resume")]
    assert len(items) == 1


def test_empty_inputs_return_false(monkeypatch, tmp_path):
    runner = _make_runner(monkeypatch, tmp_path, None)
    assert runner._rescue_split_turn_tail(
        session_key="", new_session_id="x", source=_FakeSource(),
    ) is False
    assert runner._rescue_split_turn_tail(
        session_key="agent:main:telegram:dm:1", new_session_id="x", source=None,
    ) is False
