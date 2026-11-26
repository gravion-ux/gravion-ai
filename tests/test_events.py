import pytest

from gravion_ai.events import RunEventLog


def test_run_event_log_persists_filters_and_summarises(tmp_path):
    path = tmp_path / "state" / "run-events.jsonl"
    log = RunEventLog(path)
    run_id = RunEventLog.new_run_id()
    session_id = RunEventLog.new_session_id()

    log.append(
        "run.started",
        run_id=run_id,
        session_id=session_id,
        module="chat",
        model="gravion-core",
        details={"input_chars": 5},
    )
    log.append(
        "run.completed",
        run_id=run_id,
        session_id=session_id,
        module="chat",
        model="gravion-core",
        details={"output_chars": 12},
    )

    reloaded = RunEventLog(path)
    entries = reloaded.entries(run_id=run_id)
    assert [e.event for e in entries] == ["run.started", "run.completed"]
    assert [e.seq for e in entries] == [1, 2]
    assert reloaded.entries(session_id=session_id)[0].module == "chat"
    summary = reloaded.latest_runs()[0]
    assert summary["run_id"] == run_id
    assert summary["last_event"] == "run.completed"
    assert summary["event_count"] == 2


def test_run_event_log_redacts_sensitive_details(tmp_path):
    log = RunEventLog(tmp_path / "run-events.jsonl")
    event = log.append(
        "run.failed",
        module="chat",
        details={
            "prompt": "private prompt",
            "api_key": "secret-key",
            "nested": {"authorization": "Bearer secret"},
            "error_type": "ProviderError",
        },
    )
    assert event.details["prompt"] == "[redacted]"
    assert event.details["api_key"] == "[redacted]"
    assert event.details["nested"]["authorization"] == "[redacted]"
    assert event.details["error_type"] == "ProviderError"
    assert "private prompt" not in (tmp_path / "run-events.jsonl").read_text(
        encoding="utf-8"
    )


def test_run_event_log_rejects_unknown_event(tmp_path):
    log = RunEventLog(tmp_path / "run-events.jsonl")
    with pytest.raises(ValueError, match="unknown run event"):
        log.append("run.exploded")
