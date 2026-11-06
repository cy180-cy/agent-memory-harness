"""Focused test launcher using the existing offline guard and Memory fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if os.environ.get("AMH_STRUCTURED_CHECK_CHILD") != "1":
        names = ("SystemRoot", "WINDIR", "PATH", "PATHEXT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP")
        clean = {n: os.environ[n] for n in names if n in os.environ}
        clean.update(AMH_STRUCTURED_CHECK_CHILD="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        raise SystemExit(subprocess.run([sys.executable, "-X", "utf8", "-B", __file__, *sys.argv[1:]],
                                       cwd=ROOT, env=clean).returncode)
    sys.path[:0] = [str(ROOT / "validation"), str(ROOT / "tests"),
                   str(ROOT / "validation/runtime_support")]
    from run_memory_comparison import repository, fresh_output
    repo, out = repository(), fresh_output()
    (out / "repository_state.json").write_text(json.dumps(repo, ensure_ascii=False), encoding="utf-8")
    import offline_safety
    offline_safety.OUT = out
    safety = offline_safety.Safety(out)
    safety.install()
    offline_safety.initialize_models()
    import memory_support
    memory_support.configure(safety, out)
    if "--capture" in sys.argv:
        from test_structured_context_budget import replay_obstruction
        measurement, _, _ = replay_obstruction()
        (out / "obstruction_before.json").write_text(json.dumps(measurement, ensure_ascii=False, indent=2), encoding="utf-8")
        code = 0
    else:
        import pytest
        targets = [str(ROOT / "tests/test_structured_context_budget.py")]
        for name in ("test_budgeted_injection_keeps_provenance_and_original_history_unchanged",
                     "test_task_trial_isolation_and_state_json_roundtrip",
                     "test_yes_is_not_permission_and_explicit_confirmation_is_exact_and_single_use",
                     "test_conflicting_same_batch_is_preserved_then_later_observation_resolves",
                     "test_actual_agent_http_mock_tool_loop_uses_injected_memory",
                     "test_existing_full_history_snapshot_and_budget_window_shared_interface"):
            targets.append(str(ROOT / "tests/test_structured_memory.py") + "::" + name)
        code = pytest.main([*targets, "-q", "-p", "no:cacheprovider", "--junitxml", str(out / "pytest.xml")])
    ledgers = [json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("request_ledger.json")]
    report = {"exit_code": int(code), "real_api_requests": sum(l.get("actual_api_requests", 0) for l in ledgers),
              "mock_http_requests": sum(l.get("mock_http_requests", 0) for l in ledgers), "guard_events": safety.events}
    (out / "test_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifacts": str(out), **report}, ensure_ascii=False))
    raise SystemExit(code)
