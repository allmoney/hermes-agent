"""Proactive auto-handoff at 80% context (jun17).

When ``agent.session_input_tokens`` crosses a threshold (default 80% of the
model's ``context_length``), this module writes a handoff memo to
``~/.hermes/handoff/<session_id>.md`` so the NEXT session can pick up where
we left off.  The user's `/new` cycle is preserved — we just make sure the
next session has the context.

The module is **fail-open**: any error during memo writing is logged at
``DEBUG`` and swallowed.  The agent's normal response flow must not be
disrupted by a handoff glitch.

Public API
----------
- :func:`check_and_maybe_handoff` — call from conversation_loop after each
  successful API response, passing the agent and the canonical_usage.
- :func:`inject_pending_handoff_into_prompt` — call from agent init (or
  ephemeral-system-prompt loader) to surface the latest pending memo into
  the system prompt.
- :func:`format_handoff_warning_footer` — render the ``⚠️ /new`` footer
  field when threshold is crossed.
- :func:`acknowledge_handoff` — mark a memo as read (delete or move to
  ``~/.hermes/handoff/archive/``).

Threshold
---------
Default 80% of model context.  Override via env var
``HERMES_HANDOFF_THRESHOLD_PCT`` (int 1..100) for individual runs, or
``agent.handoff_threshold_pct`` in ``~/.hermes/config.yaml`` for a
permanent change.  See ``references/jun17-auto-handoff-design.md`` for the
threshold-tuning rationale.

File layout
-----------
::

    ~/.hermes/handoff/
    ├── auto-handoff-implementation.md   (this design doc, generated once)
    ├── <session_id>.md                  (active pending memos)
    └── archive/                         (acknowledged memos)

Why ``~/.hermes/handoff/`` and not ``/tmp/``: the dir survives reboots and
is under the user's git/version-controlled state, so memos don't get
silently lost.  The user has been trained to look there (it was where the
last handoff landed).

Why not the existing ``handoff_state`` column in ``state.db``: that column
is for CLI→gateway handoffs (different concern — re-binding a CLI session
to a running gateway).  We're adding a NEW signal: "this session is at 80%
context, the NEXT session should read the memo".  We use the filesystem
for memo body (markdown is more useful than a DB blob for the human
reading it during /new).

Marker
------
This module is patched into Hermes bundled code; to survive
``hermes update`` it is detected by the watchdog via a unique marker
in the conversation_loop patch (``HANDOFF_PATCH_JUN17_AUTO_HANDOFF``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HANDOFF_DIRNAME = "handoff"
_ARCHIVE_DIRNAME = "archive"
_MARKER = "HANDOFF_PATCH_JUN17_AUTO_HANDOFF"
_DEFAULT_THRESHOLD_PCT = 80


# --------------------------------------------------------------------- paths

def _hermes_home() -> Path:
    """Return ``$HERMES_HOME`` or ``~/.hermes`` (matches the rest of Hermes)."""
    raw = os.environ.get("HERMES_HOME")
    if raw:
        return Path(raw)
    return Path(os.path.expanduser("~/.hermes"))


def handoff_dir() -> Path:
    """Return ``~/.hermes/handoff``, creating it on first use."""
    d = _hermes_home() / _HANDOFF_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def handoff_archive_dir() -> Path:
    """Return ``~/.hermes/handoff/archive``, creating it on first use."""
    d = handoff_dir() / _ARCHIVE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- threshold

def resolve_threshold_pct() -> int:
    """Return the effective threshold percent for auto-handoff.

    Order of precedence (later wins):
        1. Default (80)
        2. Env var ``HERMES_HANDOFF_THRESHOLD_PCT``
        3. Config ``agent.handoff_threshold_pct`` in ``~/.hermes/config.yaml``

    Clamped to [1, 100].  Returns 80 on any error (fail-open).
    """
    pct = _DEFAULT_THRESHOLD_PCT
    # 1. env var
    env_val = os.environ.get("HERMES_HANDOFF_THRESHOLD_PCT", "").strip()
    if env_val:
        try:
            pct = int(env_val)
        except ValueError:
            logger.debug("auto_handoff: invalid HERMES_HANDOFF_THRESHOLD_PCT=%r, ignoring", env_val)
    # 2. config.yaml
    try:
        cfg_path = _hermes_home() / "config.yaml"
        if cfg_path.exists():
            import yaml  # type: ignore
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg_val = (cfg.get("agent") or {}).get("handoff_threshold_pct")
            if cfg_val is not None:
                pct = int(cfg_val)
    except Exception as e:
        logger.debug("auto_handoff: config.yaml read failed: %s", e)
    # clamp
    return max(1, min(100, pct))


# ---------------------------------------------------------------- db helpers

def _state_db_path() -> Optional[Path]:
    """Return the path to ``state.db`` (or None if Hermes is in a non-standard layout)."""
    p = _hermes_home() / "state.db"
    return p if p.exists() else None


def _fetch_session_meta(session_id: str) -> Optional[Dict[str, Any]]:
    """Read the most recent relevant columns for a session row.

    Returns None if the session doesn't exist in state.db.  All other
    errors fail-open (return None, log DEBUG).
    """
    db = _state_db_path()
    if not db:
        return None
    try:
        con = sqlite3.connect(str(db), timeout=1.0)
        try:
            row = con.execute(
                """
                SELECT id, source, user_id, model, started_at, ended_at,
                       input_tokens, output_tokens, cache_read_tokens,
                       tool_call_count, message_count, title
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return None
        keys = ("id", "source", "user_id", "model", "started_at", "ended_at",
                "input_tokens", "output_tokens", "cache_read_tokens",
                "tool_call_count", "message_count", "title")
        return dict(zip(keys, row))
    except Exception as e:
        logger.debug("auto_handoff: state.db read failed: %s", e)
        return None


def _fetch_recent_user_messages(session_id: str, limit: int = 5) -> List[str]:
    """Return the last ``limit`` user messages for the session (text only, trimmed).

    Best-effort: any DB error returns ``[]``.  Used to seed the memo's
    "recent user asks" section so the next session has the recent intent.
    """
    db = _state_db_path()
    if not db:
        return []
    try:
        con = sqlite3.connect(str(db), timeout=1.0)
        try:
            rows = con.execute(
                """
                SELECT content FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            con.close()
        out: List[str] = []
        for (content,) in rows:
            if not content:
                continue
            text = str(content).strip()
            if len(text) > 280:
                text = text[:277] + "..."
            out.append(text)
        return list(reversed(out))  # chronological order
    except Exception as e:
        logger.debug("auto_handoff: messages read failed: %s", e)
        return []


# ---------------------------------------------------------------- memo write

def memo_path_for(session_id: str) -> Path:
    """Path where the pending memo for ``session_id`` would live."""
    return handoff_dir() / f"{session_id}.md"


def memo_exists(session_id: str) -> bool:
    """True if a pending memo already exists for this session (idempotency)."""
    return memo_path_for(session_id).exists()


def build_memo(
    *,
    session_id: str,
    context_pct: int,
    input_tokens: int,
    cache_read_tokens: int,
    context_length: int,
    model: str,
    source: str,
    user_id: Optional[str],
    recent_user_messages: List[str],
) -> str:
    """Build the markdown body of the handoff memo.

    This is a pure function — no I/O, no DB access.  Easy to unit-test.
    The format is tuned for: (a) human reading during /new, (b) the next
    session's system prompt auto-injection (it has to be ~50kB or less).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = [
        f"# Auto-handoff from session `{session_id}`",
        "",
        f"**Created:** {now}  ",
        f"**Reason:** context at **{context_pct}%** ({input_tokens:,} tokens "
        f"of {context_length:,} limit, cache_read={cache_read_tokens:,}).  ",
        f"**Model:** `{model}`  ",
        f"**Source:** `{source}`  ",
    ]
    if user_id:
        lines.append(f"**User:** `{user_id}`  ")
    lines.append("")

    # --- Recent user asks ---
    if recent_user_messages:
        lines.append("## Recent user asks (last %d)" % len(recent_user_messages))
        lines.append("")
        for i, msg in enumerate(recent_user_messages, 1):
            lines.append(f"{i}. {msg}")
        lines.append("")

    # --- What to do ---
    lines.extend([
        "## What to do",
        "",
        "The previous session was approaching its context limit.  The user has",
        "issued `/new` (or the system is auto-rotating).  Pick up the work from",
        "the most recent user ask above and continue without re-asking for",
        "context the previous session already had.",
        "",
        "If a tool/file path is mentioned in the recent asks, prefer to use it",
        "directly.  If you need more context, ask the user briefly and proceed.",
        "",
    ])
    return "\n".join(lines)


def write_memo(session_id: str, body: str) -> Path:
    """Write ``body`` to the pending memo file (atomic via temp+rename).

    Returns the path.  If a memo already exists, it is overwritten (a
    re-trigger after additional context growth is fine; the latest
    snapshot is the most useful).
    """
    target = memo_path_for(session_id)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, target)
    return target


# -------------------------------------------------------------- public hook

def check_and_maybe_handoff(
    *,
    session_id: Optional[str],
    input_tokens: int,
    cache_read_tokens: int,
    context_length: Optional[int],
    model: Optional[str] = None,
    source: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """If threshold is crossed, write the handoff memo.  Returns metadata.

    ``context_length`` may be 0/None (e.g. before the model has reported
    its limit).  In that case we use 1M as a conservative default — for
    the user's MiniMax-M3 / 1.0M context that's accurate; for larger
    models we'd underestimate and trigger late (acceptable fail-open).

    Returns None if no handoff was written.  Returns a small dict with
    keys ``memo_path``, ``context_pct``, ``threshold_pct`` when written.
    Never raises — all errors are swallowed at DEBUG.
    """
    if not session_id:
        return None
    if context_length is None or context_length <= 0:
        context_length = 1_000_000  # conservative default
    threshold_pct = resolve_threshold_pct()
    threshold_tokens = (context_length * threshold_pct) // 100
    # Use the per-call `canonical_usage.input_tokens` (== `prompt_tokens` for
    # the API call) as the "current context size" — that's what the model
    # actually sees.  NOT the cumulative `session_input_tokens`, and NOT
    # the max-with-cache_read (which is misleading because cache_read can
    # be huge after a few turns even when the model still has plenty of
    # room in this turn's prompt).
    #
    # Caller should pass `input_tokens=caller_input_tokens` (this call's
    # input_tokens from canonical_usage), not the agent.session_input_tokens
    # running total.  We fall back to a meaningful number if the caller
    # passed the cumulative: detect by checking if the value is much larger
    # than the context_length (a sure sign it's cumulative, not per-call).
    effective = int(input_tokens or 0)
    if effective > context_length:
        # Likely cumulative, not per-call.  Try cache_read as a hint of
        # the current turn size; if that's also > context, treat as a
        # genuinely huge current turn (e.g. enormous code dump).
        if cache_read_tokens and cache_read_tokens < context_length:
            effective = int(cache_read_tokens)
        # else: trust the value, even if it seems too big — better a false
        # positive than missing a real overflow
    if effective < threshold_tokens:
        return None

    # Idempotency: if we already wrote a memo for this session, don't
    # write another (the existing one is the latest snapshot).
    if memo_exists(session_id):
        return {
            "memo_path": str(memo_path_for(session_id)),
            "context_pct": round((effective / context_length) * 100),
            "threshold_pct": threshold_pct,
            "already_written": True,
        }

    meta = _fetch_session_meta(session_id) or {}
    recent = _fetch_recent_user_messages(session_id, limit=5)
    body = build_memo(
        session_id=session_id,
        context_pct=round((effective / context_length) * 100),
        input_tokens=int(input_tokens or 0),
        cache_read_tokens=int(cache_read_tokens or 0),
        context_length=int(context_length),
        model=model or meta.get("model") or "unknown",
        source=source or meta.get("source") or "unknown",
        user_id=user_id or meta.get("user_id"),
        recent_user_messages=recent,
    )
    try:
        path = write_memo(session_id, body)
        # Backlog #1: write a one-shot alert marker so a separate cron can
        # send the user a Telegram message.  Decoupling the alert from the
        # API call means the user's experience is "fast response + immediate
        # alert" instead of "alert inside the response text".  The marker
        # is a tiny JSON file with just enough context for the alert.
        try:
            import json as _json
            alert_payload = {
                "session_id": session_id,
                "context_pct": round((effective / context_length) * 100),
                "threshold_pct": threshold_pct,
                "memo_path": str(path),
                "model": model or meta.get("model") or "unknown",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            alert_marker = handoff_dir() / ".alert_pending"
            # Atomic write: temp + rename, so the alert cron never sees a
            # half-written marker.
            tmp_marker = alert_marker.with_suffix(alert_marker.suffix + ".tmp")
            tmp_marker.write_text(_json.dumps(alert_payload, ensure_ascii=False),
                                  encoding="utf-8")
            os.replace(tmp_marker, alert_marker)
        except Exception as _alert_err:
            logger.debug("auto_handoff: alert marker write failed (ignored): %s", _alert_err)

        logger.info(
            "auto_handoff: wrote memo for session=%s context_pct=%d threshold=%d path=%s",
            session_id, round((effective / context_length) * 100),
            threshold_pct, path,
        )
        return {
            "memo_path": str(path),
            "context_pct": round((effective / context_length) * 100),
            "threshold_pct": threshold_pct,
            "already_written": False,
        }
    except Exception as e:
        logger.debug("auto_handoff: write failed for session=%s: %s", session_id, e)
        return None


# ----------------------------------------------------- injection on session start

def list_pending_memos() -> List[Path]:
    """Return all pending handoff memos sorted by mtime (newest first)."""
    d = handoff_dir()
    if not d.exists():
        return []
    out = [p for p in d.glob("*.md") if p.is_file()]
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def acknowledge_memo(session_id: str) -> bool:
    """Move a pending memo to the archive dir.  Returns True on success.

    The next session will not see an archived memo.  We keep the archive
    for one release cycle (~30 days) so the user can still reference
    "what was the previous session doing" if needed.
    """
    src = memo_path_for(session_id)
    if not src.exists():
        return False
    try:
        dest = handoff_archive_dir() / src.name
        if dest.exists():
            # If a prior archive exists, overwrite (we want the latest).
            dest.unlink()
        os.replace(src, dest)
        return True
    except Exception as e:
        logger.debug("auto_handoff: acknowledge failed: %s", e)
        return False


def inject_pending_handoff_into_prompt(
    max_memos: int = 1,
    max_chars_per_memo: int = 6000,
) -> str:
    """Return a system-prompt block summarising the latest pending memo.

    Returns ``""`` if no pending memo.  Used by
    :func:`gateway.run.GatewayRunner._load_ephemeral_system_prompt` to
    surface the handoff into the next session's system prompt.

    The block is designed to be:
    - Recognisable by the model (starts with ``## Auto-handoff from ...``)
    - Self-contained (the next session doesn't need anything else)
    - Compact (truncated to ``max_chars_per_memo`` to bound context use)
    """
    pending = list_pending_memos()
    if not pending:
        return ""
    picked = pending[:max_memos]
    blocks: List[str] = ["## Auto-handoff memo (from previous session)"]
    blocks.append("")
    blocks.append(
        "The previous session wrote a handoff memo because it was "
        "approaching its context limit.  Treat this as authoritative "
        "context for the work in progress.  Acknowledge it briefly in "
        "your first reply, then continue the task."
    )
    blocks.append("")
    for p in picked:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(text) > max_chars_per_memo:
            text = text[:max_chars_per_memo] + "\n\n[...truncated...]"
        blocks.append(text)
        blocks.append("")
    blocks.append("---")
    blocks.append("")
    return "\n".join(blocks)


# -------------------------------------------------------------- footer field

_HANDOFF_WARNING_FOOTER = "⚠️ /new"


def format_handoff_warning_footer(
    *,
    context_tokens: int,
    context_length: Optional[int],
    threshold_pct: Optional[int] = None,
) -> str:
    """Return ``"⚠️ /new"`` when threshold is crossed, else ``""``.

    Used as an extra field in the runtime footer.  Returns just the warning
    string — caller is responsible for spacing/joining.
    """
    if not context_length or context_length <= 0:
        return ""
    if context_tokens <= 0:
        return ""
    pct = threshold_pct if threshold_pct is not None else resolve_threshold_pct()
    if context_tokens < (context_length * pct) // 100:
        return ""
    return _HANDOFF_WARNING_FOOTER


# ------------------------------------------------------------------ self-test

if __name__ == "__main__":
    # Minimal smoke test: write a fake memo, list, archive, delete.
    fake_id = f"_smoke_{int(time.time())}"
    body = build_memo(
        session_id=fake_id,
        context_pct=87,
        input_tokens=870_000,
        cache_read_tokens=120_000,
        context_length=1_000_000,
        model="MiniMax-M3",
        source="telegram",
        user_id="210540672",
        recent_user_messages=["продолжай работу", "сделай X"],
    )
    p = write_memo(fake_id, body)
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print(f"list_pending: {[str(x) for x in list_pending_memos()]}")
    print(f"inject: {len(inject_pending_handoff_into_prompt())} chars")
    print(f"ack: {acknowledge_memo(fake_id)}")
    print(f"list_pending after ack: {[str(x) for x in list_pending_memos()]}")
    print(f"footer at 87%: {format_handoff_warning_footer(context_tokens=870_000, context_length=1_000_000)!r}")
    print(f"footer at 50%: {format_handoff_warning_footer(context_tokens=500_000, context_length=1_000_000)!r}")
    # cleanup
    archived = handoff_archive_dir() / f"{fake_id}.md"
    if archived.exists():
        archived.unlink()
        print("cleaned up archive")
