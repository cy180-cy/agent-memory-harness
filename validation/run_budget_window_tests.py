"""Guarded BudgetWindow tests; no baseline update or live execution mode."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "validation/runtime_support"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--node-only", action="store_true")
    mode.add_argument("--integration-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    out = ROOT / "validation/phase1_budget_window" / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f+0800")
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BUDGET))
    sys.path.insert(0, str(ROOT / "tests"))
    import offline_safety
    offline_safety.OUT = out
    safety = offline_safety.Safety(out)
    safety.install()
    offline_safety.initialize_models()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import memory_support
    memory_support.configure(safety, out)
    import pytest
    test = str(ROOT / "tests/test_budget_window.py")
    if args.node_only:
        targets = [test + "::TestBudgetWindowNodes"]
    elif args.integration_only:
        targets = [str(ROOT / "tests/test_memory_agent.py"), test + "::TestBudgetWindowIntegration"]
    else:
        targets = [str(ROOT / "tests/test_memory_agent.py"), test]
    code = pytest.main([*targets, "-q", "-p", "no:cacheprovider", "--junitxml", str(out / "pytest.xml")])
    result = {"exit_code": int(code), "targets": targets, "real_api_requests": 0,
              "mock_http_requests": memory_support.HTTP_MOCK_REQUESTS, "guard_events": safety.events}
    offline_safety.dump(out / "run_summary.json", result)
    print(json.dumps({"artifacts": str(out), **result}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
