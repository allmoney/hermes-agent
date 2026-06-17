"""
Tests for the 3-state last_status feature and [HANDOFF] marker detection.

Added as part of the upstream PR for hermes-agent v0.16.0+:
  - cron.scheduler.HANDOFF_MARKER constant + detection
  - cron.jobs.mark_job_run(..., handoff=False) parameter
  - hermes_cli.cron yellow display for last_status="handoff"

These tests use hermes_constants.set_hermes_home_override to redirect
~/.hermes/ to a temp dir, so they don't touch the real config.
"""
import pytest

from hermes_constants import set_hermes_home_override

import cron.jobs as jobs


@pytest.fixture
def fresh_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp dir for test isolation."""
    set_hermes_home_override(str(tmp_path))
    yield tmp_path


def _seed_job(tmp_path, job_id="t1"):
    """Write a minimal test job to the temp jobs.json."""
    jobs.save_jobs([{
        "id": job_id,
        "name": "t",
        "prompt": "p",
        "skill": "s",
        "schedule": {
            "kind": "cron",
            "expr": "0 23 * * *",
            "display": "0 23 * * *",
        },
        "enabled": True,
        "state": "scheduled",
        "created_at": "2026-06-17T00:00:00+00:00",
        "next_run_at": None,
        "last_status": None,
        "last_delivery_error": None,
        "deliver": "local",
        "origin": None,
        "repeat": None,
        "enabled_toolsets": [],
    }])


# --- mark_job_run tests ---

def test_mark_job_run_success_ok(fresh_hermes_home):
    """Backward compat: success=True (no handoff) → last_status='ok'."""
    _seed_job(fresh_hermes_home)
    jobs.mark_job_run("t1", success=True)
    assert jobs.load_jobs()[0]["last_status"] == "ok"


def test_mark_job_run_failure_error(fresh_hermes_home):
    """Backward compat: success=False → last_status='error'."""
    _seed_job(fresh_hermes_home)
    jobs.mark_job_run("t1", success=False, error="boom")
    assert jobs.load_jobs()[0]["last_status"] == "error"


def test_mark_job_run_handoff(fresh_hermes_home):
    """NEW: success=True + handoff=True → last_status='handoff'."""
    _seed_job(fresh_hermes_home)
    jobs.mark_job_run("t1", success=True, handoff=True)
    assert jobs.load_jobs()[0]["last_status"] == "handoff"


def test_mark_job_run_handoff_with_failure_is_error(fresh_hermes_home):
    """Defensive: handoff=True + success=False stays 'error', not 'handoff'.

    Caller shouldn't pass both, but if they do, we should NOT classify
    a real failure as a handoff. Better to have a false 'error' than
    to hide a real one.
    """
    _seed_job(fresh_hermes_home)
    jobs.mark_job_run("t1", success=False, handoff=True, error="boom")
    assert jobs.load_jobs()[0]["last_status"] == "error"


def test_mark_job_run_default_handoff_false(fresh_hermes_home):
    """Backward compat: omitting handoff param → behaves as handoff=False."""
    _seed_job(fresh_hermes_home)
    jobs.mark_job_run("t1", success=True)  # no handoff kwarg
    assert jobs.load_jobs()[0]["last_status"] == "ok"


# --- HANDOFF_MARKER constant tests ---

def test_handoff_marker_constant_exists():
    """Regression: HANDOFF_MARKER constant must be defined in scheduler."""
    from cron.scheduler import HANDOFF_MARKER
    assert isinstance(HANDOFF_MARKER, str)
    assert HANDOFF_MARKER.startswith("[")
    assert HANDOFF_MARKER.endswith("]")
    # Should be a short, distinctive token — not a long sentence
    assert len(HANDOFF_MARKER) < 32


def test_handoff_marker_distinctive():
    """HANDOFF_MARKER should not collide with common agent responses.

    We test against a sample of common first-line responses to ensure the
    marker is unlikely to appear by accident. If this test fails, the
    marker is too generic and a real agent response could be misclassified.
    """
    from cron.scheduler import HANDOFF_MARKER
    common_first_lines = [
        "Here's the report:",
        "Done. Summary:",
        "I've completed the task.",
        "All fixes applied.",
        "Summary of work:",
        "The sweep finished with:",
    ]
    for line in common_first_lines:
        assert not line.startswith(HANDOFF_MARKER), \
            f"HANDOFF_MARKER too generic: collides with {line!r}"


# --- _process_job integration test (skeleton — adjust to upstream signature) ---

@pytest.mark.skip(reason="Pending _process_job signature inspection in upstream")
def test_process_job_strips_handoff_marker():
    """Scheduler strips [HANDOFF] prefix and signals handoff=True to mark_job_run.

    Full integration test depends on _process_job's signature in the
    upstream repo. Skeleton provided; expand once the signature is
    confirmed.
    """
    from cron.scheduler import _process_job
    # 1. Mock the agent runner to return "[HANDOFF]\n\nhandoff memo..."
    # 2. Capture mark_job_run calls (via monkeypatch)
    # 3. Assert: (a) marker stripped, (b) handoff=True passed,
    #    (c) last_status='handoff' in jobs.json
    # See PR description for the full test outline.
    pass
