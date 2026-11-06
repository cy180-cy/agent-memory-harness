"""Local persistence faults only; HTTP responses come from MockTransport."""

from pathlib import Path

import pytest

import memory_support as support
from memory_comparison import spec
from memory_comparison.execution import run_pair
from test_comparison_runtime_fix import plan, folder, complete_frames
from test_comparison_user_output import synthetic_builder


def sharing_error():
    error = PermissionError("synthetic Windows sharing conflict")
    error.winerror = 5
    return error


def test_normal_atomic_write_uses_independent_temporary_files(folder):
    target = folder / "journal.json"
    spec.atomic_json(target, {"generation": 1})
    spec.atomic_json(target, {"generation": 2})
    assert spec.read_json(target) == {"generation": 2}
    assert not list(folder.glob("*.next")) and not list(folder.glob("*.tmp"))


def test_one_windows_sharing_conflict_retries_only_the_file_replace(folder, monkeypatch):
    target = folder / "journal.json"
    spec.atomic_json(target, {"original": True})
    original = spec.os.replace
    calls = []

    def replace(source, destination):
        calls.append(str(source))
        if len(calls) == 1:
            assert spec.read_json(target) == {"original": True}
            raise sharing_error()
        return original(source, destination)

    monkeypatch.setattr(spec.os, "replace", replace)
    spec.atomic_json(target, {"updated": True})
    assert len(calls) == 2 and calls[0] == calls[1]
    assert spec.read_json(target) == {"updated": True}


def test_persistent_post_response_failure_keeps_count_and_response_and_blocks_B(
        plan, folder, synthetic_builder, monkeypatch):
    original = spec.os.replace
    failed = []

    def replace(source, destination):
        evidence = folder / "A/received_responses/0001.json"
        if Path(destination).name == "comparison_request_ledger.json" and evidence.exists():
            failed.append(str(source))
            raise sharing_error()
        return original(source, destination)

    monkeypatch.setattr(spec.os, "replace", replace)
    result = run_pair(plan, folder, support.RUNTIME[0], frames_by_group={
        "A": complete_frames("unused-A"), "B": complete_frames("unused-B")})
    assert synthetic_builder == ["A"] and result["groups"][1]["requests"] == 0
    assert result["real_api_requests"] == 0 and result["mock_http_requests"] == 1
    assert not result["a_to_b_gate"]["allowed"] and len(failed) >= 4
    journal = spec.read_json(folder / "comparison_request_ledger.json")
    assert len(journal["attempts"]) == 1  # The pre-I/O reservation remains durable.
    received = spec.read_json(folder / "A/received_responses/0001.json")
    assert received["http_status"] == 200 and received["raw_response"]["usage"]["completion_tokens"] == 600
    assert list(folder.glob(".comparison_request_ledger.json.*.tmp"))
    before = (folder / "comparison_request_ledger.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_pair(plan, folder, support.RUNTIME[0])
    assert before == (folder / "comparison_request_ledger.json").read_bytes()


def test_monitor_reads_immutable_publication_while_active_ledger_updates(folder):
    ledger = spec.RequestLedger(folder, "synthetic-monitor")
    event = {"ordinal": 1, "role": "user", "mode": "mock", "status": "reserved_before_io"}
    ledger.sync("A", {"events": [event]})
    first = spec.read_monitor_snapshot(folder)
    path = Path(first["snapshot_file"])
    original = path.read_bytes()
    # Holding a Windows reader on this immutable publication cannot block
    # replacement of the different, authoritative ledger path.
    with path.open("rb") as reader:
        event.update(status="response_received", http_status=200)
        ledger.sync("A", {"events": [event]})
        assert reader.read() == original
    latest = spec.read_monitor_snapshot(folder)
    assert latest["snapshot_file"] != first["snapshot_file"]
    assert latest["latest_attempt"]["status"] == "response_received"
    assert path.read_bytes() == original
