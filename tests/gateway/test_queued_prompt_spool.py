from gateway import queued_prompt_spool as spool
from gateway.session import SessionSource
from gateway.config import Platform


class _Adapter:
    pass


class _Runner:
    def __init__(self):
        self.adapter = _Adapter()
        self.restored = []

    def _adapter_for_source(self, source):
        return self.adapter if source.platform == Platform.TELEGRAM else None

    def _enqueue_fifo(self, session_key, event, adapter):
        self.restored.append((session_key, event.text, event.metadata["queue_id"]))


def test_queue_ledger_restores_fifo_and_acks(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="210540672")
    q1 = spool.enqueue(session_key="agent:main:telegram:dm:210540672", text="one",
                      source=source.to_dict())
    q2 = spool.enqueue(session_key="agent:main:telegram:dm:210540672", text="two",
                      source=source.to_dict())

    runner = _Runner()
    assert spool.restore_into_runner(runner) == 2
    assert runner.restored == [
        ("agent:main:telegram:dm:210540672", "one", q1),
        ("agent:main:telegram:dm:210540672", "two", q2),
    ]

    spool.ack(q1)
    assert [item["id"] for item in spool.pending()] == [q2]
    spool.clear_session("agent:main:telegram:dm:210540672")
    assert spool.pending() == []


def test_each_telegram_item_persists_its_own_reply_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    session_key = "agent:main:telegram:dm:210540672"
    first = spool.enqueue(
        session_key=session_key,
        text="first",
        source={"platform": "telegram", "chat_id": "210540672", "message_id": "111"},
    )
    second = spool.enqueue(
        session_key=session_key,
        text="second",
        source={"platform": "telegram", "chat_id": "210540672", "message_id": "222"},
    )
    by_id = {item["id"]: item for item in spool.pending()}
    assert by_id[first]["metadata"]["queue_reply_anchor"] == "111"
    assert by_id[second]["metadata"]["queue_reply_anchor"] == "222"

    # Legacy entries without metadata are repaired from their own source, not
    # from session-level delivery state or a neighbouring FIFO item.
    by_id[second]["metadata"].pop("queue_reply_anchor")
    spool._write(list(by_id.values()))
    assert spool.backfill_reply_anchors() == 1
    by_id = {item["id"]: item for item in spool.pending()}
    assert by_id[second]["metadata"]["queue_reply_anchor"] == "222"


def test_queue_ack_requires_real_turn_completion():
    assert not spool.queued_turn_succeeded({
        "completed": True, "failed": False,
        "turn_exit_reason": "compaction_handoff_not_actionable",
    })
    assert not spool.queued_turn_succeeded({
        "completed": True, "failed": False,
        "turn_exit_reason": "max_iterations_reached(10/10)",
    })
    assert spool.queued_turn_succeeded({
        "completed": True, "failed": False,
        "final_response": "Real model answer delivered to the user.",
        "interrupted": False, "partial": False,
        "turn_exit_reason": "text_response(finish_reason=stop)",
    })


def test_queue_lifecycle_audit_excludes_message_text(tmp_path, monkeypatch):
    monkeypatch.setattr(spool, "_path", lambda: tmp_path / "queued_prompts.json")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="210540672")
    item_id = spool.enqueue(session_key="session", text="secret user text",
                            source=source.to_dict())
    spool.ack(item_id)
    audit = (tmp_path / "queued_prompts.audit.jsonl").read_text()
    assert '"event":"enqueued"' in audit
    assert '"event":"acknowledged"' in audit
    assert "secret user text" not in audit
