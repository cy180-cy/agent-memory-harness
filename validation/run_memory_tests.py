"""Guarded, zero-live-API Memory tests; capture mode only before LLMAgent edits."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "validation/runtime_support"
from repository_state import UPSTREAM_HEAD, repository

HEAD = UPSTREAM_HEAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline",action="store_true")
    parser.add_argument("--artifacts",type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    prefix=[shutil.which("git"),"-c",f"safe.directory={ROOT.as_posix()}","-c","core.quotepath=false"]
    def git(*command):
        return subprocess.check_output(prefix+list(command),cwd=ROOT,
            env={"SystemRoot":os.environ.get("SystemRoot",r"C:\Windows"),"GIT_OPTIONAL_LOCKS":"0"},text=True,encoding="utf-8").strip()
    repo = repository(ROOT)
    if args.capture_baseline and git("diff",HEAD,"--","src/tau2/agent/llm_agent.py"):
        raise RuntimeError("Baseline capture refused: LLMAgent differs from frozen HEAD")
    git_blob=git("rev-parse",HEAD+":src/tau2/agent/llm_agent.py")
    artifact_root=ROOT/"validation/phase1_memory"
    out=args.artifacts.resolve() if args.artifacts else artifact_root/datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f+0800")
    if not out.is_relative_to(artifact_root):
        raise RuntimeError("Artifacts must remain under validation/phase1_memory")
    out.mkdir(parents=True,exist_ok=args.artifacts is not None)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(BUDGET))
    sys.path.insert(0,str(ROOT/"tests"))
    import offline_safety
    offline_safety.OUT=out
    safety=offline_safety.Safety(out)
    safety.install()
    offline_safety.initialize_models()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1"
    import memory_support
    memory_support.configure(safety,out)
    from offline_safety import dump
    dump(out/"repository_state.json",repo)
    if args.capture_baseline:
        snapshot={"head":HEAD,"llm_agent_git_blob":git_blob,
            "llm_agent_source_sha256":hashlib.sha256((ROOT/"src/tau2/agent/llm_agent.py").read_bytes()).hexdigest(),
            "scenario_source_sha256":hashlib.sha256((ROOT/"tests/memory_support.py").read_bytes()).hexdigest(),
            "versions":{n:metadata.version(n) for n in ("tau2","litellm","httpx","openai")},
            "cases":{name:memory_support.run_scenario("llm_agent",name) for name in memory_support.SCENARIOS}}
        dump(out/"frozen_requests.json",snapshot)
        result={"status":"captured_before_source_edits","cases":len(snapshot["cases"]),"mock_http_requests":memory_support.HTTP_MOCK_REQUESTS,"real_api_requests":0,"guard_events":safety.events}
        code=0
    else:
        import pytest
        code=pytest.main([str(ROOT/"tests/test_memory_agent.py"),"-q","-p","no:cacheprovider","--junitxml",str(out/"pytest.xml")])
        result={"exit_code":int(code),"mock_http_requests":memory_support.HTTP_MOCK_REQUESTS,"real_api_requests":0,"guard_events":safety.events}
    dump(out/"run_summary.json",result)
    print(json.dumps({"artifacts":str(out),**result},ensure_ascii=False,indent=2))
    return code


if __name__=="__main__":
    raise SystemExit(main())
