"""Prepare by default; Mock/tests are offline. Live requires fresh bound approval."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "validation/runtime_support"


def repository():
    """Verify the current development commit and retain the upstream identity."""
    from repository_state import repository as read_repository
    return read_repository(ROOT)


def fresh_output():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f+0800")
    out = ROOT / "validation/phase1_comparison" / (stamp + "_" + uuid.uuid4().hex[:10])
    out.mkdir(parents=True,exist_ok=False)
    return out


def live_preflight(plan, authorization, folder, repo, key_reader, *, mode="live"):
    """Check approval, source/data identity and consumed state BEFORE Key access."""
    from memory_comparison.spec import require_authorization
    auth = require_authorization(plan,authorization,mode=mode)
    code = plan["groups"][0]["code"]
    if code.get("development_head", code.get("upstream_head")) != repo["head"]:
        raise PermissionError("Development commit changed; prepare fresh approval; no Key read")
    if plan["groups"][0]["code"]["tracked_patch_sha256"] != repo["tracked_patch_sha256"]:
        raise PermissionError("Tracked patch identity changed; no Key read")
    if (folder/"execution_started.lock").exists():
        raise PermissionError("Comparison authorization already consumed; no Key read")
    if plan.get("authorization_id"):
        from memory_comparison.spec import write_new
        ident = plan["authorization_id"]
        if not isinstance(ident, str) or not ident.replace("-", "").isalnum():
            raise PermissionError("Invalid authorization ID; no Key read")
        bindings = ROOT/"validation/phase1_comparison/authorization_bindings"
        bindings.mkdir(exist_ok=True)
        # Exclusive creation binds the ID across all run directories/processes.
        write_new(bindings/(ident+".json"), {"authorization_id":ident,
            "comparison_id":plan["comparison_id"], "identity_sha256":plan["identity_sha256"],
            "status":"claimed_no_automatic_resume", "plan_folder":str(folder.parent)})
    return auth,key_reader("AMH_QIANWEN_API_KEY")


def read_project_key(name):
    """Read only the project's named process/current-user variable, after approval."""
    if name != "AMH_QIANWEN_API_KEY":
        raise PermissionError("Unexpected credential name")
    value = os.environ.get(name)
    if value:
        return value
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, kind = winreg.QueryValueEx(key, name)
            return value if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str) else None
        except FileNotFoundError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mock",action="store_true")
    modes.add_argument("--tests",action="store_true")
    modes.add_argument("--execute",action="store_true")
    modes.add_argument("--status",type=Path,help="Read an immutable monitor snapshot from a run's live_execution folder")
    modes.add_argument("--export", dest="export_paths", type=Path, nargs="+", help="Export saved runs to CSV/JSON/Chinese Markdown; no model calls")
    modes.add_argument("--demo", type=Path, help="Read-only Chinese CLI replay of saved results")
    parser.add_argument("--suite", action="store_true", help="Prepare or Mock the fixed three-strategy Retail matrix; old A/B default is retained")
    parser.add_argument("--phase1-07", action="store_true", help="Prepare the approved task scope with CNY reservations; does not authorize live execution")
    parser.add_argument("--task-ids", nargs="+", help="With --suite, subset of fixed train tasks 34 69 73 93")
    parser.add_argument("--task", help="Task ID to display with --demo (defaults to first saved task)")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--authorization",type=Path)
    parser.add_argument("--gate-only",action="store_true",help="With --tests, run only A-to-B gate tests")
    parser.add_argument("--user-output-only",action="store_true",help="With --tests, run the User output profile regressions only")
    parser.add_argument("--runtime-fix-only",action="store_true",help="With --tests, run mode and thinking-budget regressions only")
    parser.add_argument("--development-tests",choices=("ledger","structured","evaluation","portable","fees"),help="With --tests, run only the selected development tests")
    args = parser.parse_args()
    if args.phase1_07 and (not args.suite or args.execute or args.task_ids != ["69", "73", "93"]):
        parser.error("--phase1-07 requires offline --suite --task-ids 69 73 93")
    if args.task_ids and (not args.suite or args.execute):
        parser.error("--task-ids requires offline --suite preparation; execute uses the frozen plan")
    if args.task and args.demo is None:
        parser.error("--task requires --demo")
    if args.gate_only and not args.tests:
        parser.error("--gate-only requires --tests")
    if args.user_output_only and (not args.tests or args.gate_only):
        parser.error("--user-output-only requires --tests and excludes --gate-only")
    if args.runtime_fix_only and (not args.tests or args.gate_only or args.user_output_only):
        parser.error("--runtime-fix-only requires --tests and excludes other test selectors")
    if args.development_tests and (not args.tests or args.gate_only or args.user_output_only or args.runtime_fix_only):
        parser.error("--development-tests requires --tests and excludes other selectors")
    if not args.execute and os.environ.get("AMH_COMPARISON_OFFLINE_CHILD") != "1":
        # Leave the caller's credential variable untouched. This explicit
        # allowlist never reads/passes AMH_QIANWEN_API_KEY to the offline child.
        names = ("SystemRoot","WINDIR","PATH","PATHEXT","USERPROFILE",
            "APPDATA","LOCALAPPDATA","TEMP","TMP")
        clean_env = {n:os.environ[n] for n in names if n in os.environ}
        clean_env.update(AMH_COMPARISON_OFFLINE_CHILD="1",PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable,"-X","utf8","-B",str(Path(__file__).resolve()),
            *sys.argv[1:]],cwd=ROOT,env=clean_env,check=False).returncode
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(BUDGET))
    sys.path.insert(0,str(ROOT/"validation"))
    from memory_comparison.spec import atomic_json, read_json, write_new, read_monitor_snapshot
    if args.status is not None:
        if not args.status.resolve().is_relative_to(ROOT/"validation"):
            raise PermissionError("Status is limited to project B validation outputs")
        print(json.dumps(read_monitor_snapshot(args.status),ensure_ascii=False,indent=2))
        return 0
    if args.demo is not None or args.export_paths is not None:
        from memory_comparison.results import export_results, render_demo
        paths = [args.demo] if args.demo is not None else args.export_paths
        if any(not p.resolve().is_relative_to(ROOT/"validation") for p in paths):
            raise PermissionError("Replay/export is limited to project B validation evidence")
        if args.demo is not None:
            print(render_demo(args.demo, args.task))
        else:
            out = fresh_output()
            report = export_results(paths, out/"exports")
            print(json.dumps({"artifacts":str(out/"exports"),"rows":len(report["rows"]),"real_api_requests":0},ensure_ascii=False))
        return 0
    repo = repository()
    secret,auth,plan = None,None,None
    if args.execute:
        if args.plan is None or not args.plan.resolve().is_relative_to(ROOT/"validation/phase1_comparison"):
            raise PermissionError("A newly prepared comparison plan is required; no Key read")
        plan = read_json(args.plan)
        out = args.plan.resolve().parent / "live_execution"
        auth,secret = live_preflight(plan,args.authorization,out,repo,
            read_project_key)
        if not secret or not secret.strip():
            raise RuntimeError("Missing private process credential; no request sent")
        out.mkdir(exist_ok=True)
    else:
        if args.authorization is not None or args.plan is not None:
            raise ValueError("Offline preparation always creates a fresh independent plan")
        out = fresh_output()
    if args.tests:
        write_new(out/"repository_state.json",repo)
    import offline_safety
    offline_safety.OUT = out
    safety = offline_safety.Safety(out)
    safety.install()
    offline_safety.initialize_models()
    from memory_comparison.calibration import calibrate, markdown_table
    from memory_comparison.execution import build_group, retail_task, run_pair
    from memory_comparison.spec import make_pair, validate_pair
    from tau2.runner.build import build_environment
    if args.execute:
        safety.live_authorized = True
        try:
            if plan.get("schema") == "memory-suite-v1":
                from memory_comparison.evaluation import run_suite
                result = run_suite(plan,out,safety,live=True,secret=secret,authorization=args.authorization)
            else:
                result = run_pair(plan,out,safety,live=True,secret=secret,authorization=args.authorization)
        finally:
            secret = None
        print(json.dumps({"artifacts":str(out),"real_api_requests":result["real_api_requests"]},ensure_ascii=False))
        return 0
    if args.tests:
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        sys.path.insert(0,str(ROOT/"tests"))
        import memory_support
        memory_support.configure(safety,out)
        import pytest
        targets = [str(ROOT/"tests/test_comparison_gate.py")]
        if args.development_tests:
            targets = [str(ROOT/{"ledger":"tests/test_ledger_persistence.py", "structured":"tests/test_structured_memory.py",
                                "evaluation":"tests/test_memory_evaluation.py",
                                "fees":"tests/test_suite_fees.py",
                                "portable":"tests/test_repository_state.py"}[args.development_tests])]
            if args.development_tests == "evaluation":
                targets += [str(ROOT/"tests/test_comparison_gate.py") + "::test_actual_run_pair_constructs_B_for_valid_reward_zero_or_one",
                            str(ROOT/"tests/test_memory_comparison.py") + "::test_group_and_global_ledger_caps_are_durable_and_never_refund",
                            str(ROOT/"tests/test_ledger_persistence.py") + "::test_one_windows_sharing_conflict_retries_only_the_file_replace"]
            elif args.development_tests == "portable":
                selections = {
                    "test_memory_agent.py": ["test_frozen_fixture_integrity",
                        "test_each_request_matches_independent_head_snapshot"],
                    "test_memory_comparison.py": [
                        "test_retail_calibration_uses_real_input_fields_and_safe_fixed_budget",
                        "test_reconstructed_prefix_matches_only_captured_visible_past",
                        "test_historical_real_usage_is_not_relabelled_as_mock_usage",
                        "test_new_authorization_gate_precedes_any_key_or_network",
                        "test_changed_source_manifest_is_rejected_before_key"],
                    "test_comparison_runtime_fix.py": [
                        "test_agent_truncation_remains_blocked_before_tools_or_B",
                        "test_existing_transport_stops_before_second_dispatch_and_global_count_is_durable"],
                }
                targets += [str(ROOT/"tests"/file)+"::"+name
                            for file, names in selections.items() for name in names]
        elif args.runtime_fix_only:
            targets = [str(ROOT/"tests/test_comparison_runtime_fix.py")]
        elif args.user_output_only:
            targets = [str(ROOT/"tests/test_comparison_user_output.py")]
        elif not args.gate_only:
            targets += [str(ROOT/"tests/test_memory_comparison.py"),str(ROOT/"tests/test_memory_agent.py"),
                str(ROOT/"tests/test_budget_window.py")]
        code = pytest.main([*targets,"-q","-p","no:cacheprovider","--junitxml",str(out/"pytest.xml")])
        ledgers = [read_json(p) for p in out.rglob("request_ledger.json")]
        result = {"exit_code":int(code),"real_api_requests":sum(l["actual_api_requests"] for l in ledgers if not l.get("synthetic_no_http")),
            "mock_http_requests":sum(l["mock_http_requests"] for l in ledgers if not l.get("synthetic_no_http")),
            "synthetic_request_slots":sum(len(l.get("events", [])) for l in ledgers if l.get("synthetic_no_http")),
            "guard_events":safety.events}
        write_new(out/"test_summary.json",result)
        print(json.dumps({"artifacts":str(out),**result},ensure_ascii=False,indent=2))
        return code
    if args.suite:
        from memory_comparison.evaluation import task_manifest, make_suite, validate_suite, run_suite
        manifest = task_manifest() if args.task_ids is None else task_manifest(args.task_ids)
        plan = make_suite(out.name, repo, manifest, phase1_07=args.phase1_07)
        write_new(out/"task_manifest.json",manifest)
        write_new(out/"comparison_plan.json",plan)
        checks = []
        for group in plan["groups"]:
            orch = build_group(group)
            checks.append({"group":group["group"], "agent":type(orch.agent).__name__,
                           "strategy":orch.agent.memory_strategy.name, "user":type(orch.user).__name__})
        write_new(out/"preparation_summary.json", {"mode":"offline_preparation", "real_api_requests":0,
            "configuration_check":validate_suite(plan), "construction_checks":checks,
            "paid_authorization_created":False, "budget_target_cny":plan["budget_target_cny"],
            "price_status":plan["price_status"]})
        write_new(out/"tracked_patch.diff.json", {"diff":repo["tracked_patch"]})
        if args.mock:
            execution = out/"mock_execution"
            execution.mkdir()
            result = run_suite(plan,execution,safety)
            from memory_comparison.results import export_results
            export_results([execution],out/"exports")
            atomic_json(out/"mock_summary.json",result)
        write_new(out/"safety_events.json",safety.events)
        print(json.dumps({"artifacts":str(out), "real_api_requests":0, "groups":len(plan["groups"]),
                          "mode":"mock" if args.mock else "offline_preparation"},ensure_ascii=False,indent=2))
        return 0
    task,env = retail_task(),build_environment("retail")
    calibration = calibrate(task,env)
    write_new(out/"calibration.json",calibration)
    (out/"calibration_table.md").write_text(markdown_table(calibration),encoding="utf-8")
    plan = make_pair(out.name,repo,calibration,task,env)
    write_new(out/"comparison_plan.json",plan)
    write_new(out/"authorization.pending.json",{"approved":False,"phase":"Phase1-memory-comparison-live",
        "comparison_id":plan["comparison_id"],"identity_sha256":plan["identity_sha256"],
        "group_request_caps":{"A":20,"B":20},"total_request_cap":40,"budget_target_cny":None,
        "note":"Not an effective paid authorization. Requires separate user approval."})
    write_new(out/"tracked_patch.diff.json",{"diff":repo["tracked_patch"]})
    checks = []
    for group in plan["groups"]:
        orch = build_group(group)
        checks.append({"group":group["group"],"agent_class":type(orch.agent).__name__,
            "strategy":orch.agent.memory_strategy.name,"user_class":type(orch.user).__name__,
            "agent_state":orch.agent_state,"user_state":orch.user_state,"database_id":id(orch.environment.tools.db)})
    write_new(out/"preparation_summary.json",{"mode":"offline_preparation","real_api_requests":0,
        "comparison_check":validate_pair(plan),"construction_checks":checks,
        "input_budget_selection":calibration["selection"],"paid_authorization_created":False})
    if args.mock:
        execution = out/"mock_execution"
        execution.mkdir()
        result = run_pair(plan,execution,safety)
        atomic_json(out/"mock_summary.json",result)
    write_new(out/"safety_events.json",safety.events)
    print(json.dumps({"artifacts":str(out),"real_api_requests":0,
        "selection":calibration["selection"],"mode":"mock" if args.mock else "offline_preparation"},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
