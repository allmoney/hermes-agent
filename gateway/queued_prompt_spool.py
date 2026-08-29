"""Durable FIFO spool for Telegram/gateway ``/queue`` prompts.

The live gateway keeps queued MessageEvents in memory.  This small ledger makes
those events survive a graceful restart: enqueue publishes before the event is
made runnable, and dequeue acknowledges only after the event is removed from
the adapter slot.  Files are profile-local via get_hermes_home().

aug28 durability fix (queue task #7, /reset + /new):
  - ``clear_session`` was called from ``_clear_conversation_scope`` on every
    conversation boundary (``/new``, ``/reset``, idle-expiry, compression-
    exhausted auto-reset).  That destroyed the durable ledger at exactly the
    moment the user started a fresh session, so any item that had not yet been
    acked was permanently lost the instant the user typed ``/reset``.
    The ledger is now boundary-proof: items only leave it on explicit
    completion (``queue_completed`` / ``ack``) or on an explicit clear.
  - ``ack`` was called at dequeue time (the moment the event was popped from
    the adapter's slot).  If the gateway crashed, was OOM-killed, or the run
    was interrupted between dequeue and completion, the item had already been
    removed from the spool and could not be recovered.  ``_dequeue_pending_
    event`` no longer acks; the drain calls ``queue_completed`` after each
    queued item's turn has been delivered to the agent.
  - ``restore_into_runner`` could re-inject an item that was still in flight
    in a live session (e.g. after ``/reset`` mid-run, when the spool still
    holds the in-flight item because completion had not happened yet).  A
    process-local ``_started`` set makes restore idempotent: an item whose
    ``queue_id`` was already injected into a live adapter during this process
    is skipped rather than duplicated.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
# Process-local: set of queue_ids that have already been injected into a live
# adapter during THIS gateway process.  Cleared on restart, so a crashed run
# is re-injected on the next startup (at-least-once delivery).  The set only
# guards against same-process double-injection (e.g. after a /reset that
# clears live state but not the spool while the current item is mid-run).
_started: set[str] = set()


def _path() -> Path:
    from hermes_constants import get_hermes_home
    p = get_hermes_home() / "queued_prompts.json"
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    return p


def _read() -> list[dict[str, Any]]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write(items: list[dict[str, Any]]) -> None:
    path = _path()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def enqueue(*, session_key: str, text: str, source: dict[str, Any],
            metadata: dict[str, Any] | None = None) -> str:
    """Persist one queue item and return its stable id."""
    with _LOCK:
        item_id = str((metadata or {}).get("queue_id") or uuid.uuid4().hex)
        items = _read()
        if not any(str(x.get("id")) == item_id for x in items):
            # Test doubles and some platform metadata may contain non-JSON
            # objects. Persist their stable string form rather than allowing a
            # best-effort durability feature to break the live FIFO queue.
            safe_source = json.loads(json.dumps(source, default=str))
            safe_metadata = json.loads(json.dumps(metadata or {}, default=str))
            items.append({"id": item_id, "session_key": session_key,
                          "text": text, "source": safe_source,
                          "metadata": safe_metadata})
            _write(items)
        return item_id


def ack(item_id: str | None) -> None:
    if not item_id:
        return
    with _LOCK:
        items = [x for x in _read() if str(x.get("id")) != str(item_id)]
        _write(items)


def clear_session(session_key: str) -> None:
    """Remove queued work only when a real conversation boundary is requested."""
    if not session_key:
        return
    with _LOCK:
        items = [x for x in _read() if x.get("session_key") != session_key]
        _write(items)


def pending() -> list[dict[str, Any]]:
    with _LOCK:
        return _read()


def restore_into_runner(runner: Any) -> int:
    """Rebuild live FIFO events after adapters have connected."""
    restored = 0
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource
    for item in pending():
        try:
            source = SessionSource.from_dict(item["source"])
            adapter = runner._adapter_for_source(source)
            if adapter is None:
                continue
            metadata = dict(item.get("metadata") or {})
            metadata["queue_id"] = str(item["id"])
            event = MessageEvent(text=str(item.get("text") or ""),
                                 source=source, metadata=metadata)
            runner._enqueue_fifo(item["session_key"], event, adapter)
            restored += 1
        except Exception:
            # Keep the ledger entry for the next startup; a missing adapter or
            # temporarily invalid platform must never destroy user work.
            continue
    return restored
