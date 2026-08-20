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
