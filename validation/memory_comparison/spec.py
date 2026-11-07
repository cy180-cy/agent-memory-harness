"""Frozen experiment identity, strict pair validation and durable request limits."""

import hashlib
import json
import math
import os
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from qianwen_adapter import role_args as _base_role_args
from repository_state import UPSTREAM_HEAD

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "validation/runtime_support"
HISTORY = ROOT / "tests/fixtures/memory/train34_replay"
HEAD = UPSTREAM_HEAD  # Historical source identity, not a required current HEAD.
BRANCH = "feature/agent-memory-phase1-validation-gate"
PROXY = "http://127.0.0.1:7890"
LEGACY_OUTPUT_CONFIG = "comparison-output-legacy-1024-512"
USER_OUTPUT_CONFIG = "comparison-output-user1024-v1"
OUTPUT_CONFIG_VERSION = "comparison-output-agent3072-thinking1536-v1"
_JSON_WRITER = threading.RLock()


def normalize_mode(mode):
    """Use live/mock internally; accept only the historical real alias."""
    if not isinstance(mode, str) or mode not in ("live", "real", "mock"):
        raise ValueError("Unknown execution mode")
    return "live" if mode == "real" else mode


def execution_mode(*, live=False, mode=None):
    """Resolve an explicit mode without treating arbitrary strings as booleans."""
    if type(live) is not bool:
        raise ValueError("live must be a boolean")
    resolved = normalize_mode(mode) if mode is not None else ("live" if live else "mock")
    if live and resolved != "live":
        raise ValueError("Conflicting live flag and execution mode")
    return resolved


def configured_role_args(version=OUTPUT_CONFIG_VERSION):
    """Select a frozen comparison profile without changing the historical client."""
    profiles = {
        LEGACY_OUTPUT_CONFIG: {"agent": 1024, "user": 512},
        USER_OUTPUT_CONFIG: {"agent": 1024, "user": 1024},
        OUTPUT_CONFIG_VERSION: {"agent": 3072, "user": 1024},
    }
    if version not in profiles:
        raise ValueError("Unknown comparison output configuration")
    # Keep a reference to the historical factory so scoped role binding cannot
    # recursively feed one profile's extra_body into another profile.
    args = {role: {**_base_role_args(role), "max_completion_tokens": cap}
            for role, cap in profiles[version].items()}
    if version == OUTPUT_CONFIG_VERSION:
        args["agent"]["extra_body"] = {"thinking_budget": 1536}
    return args


def group_output_limits(group):
    """Validate the complete role parameters before binding any model calls."""
    version = group.get("output_config_version", LEGACY_OUTPUT_CONFIG)
    expected = configured_role_args(version)
    if group["role_args"] != expected:
        raise ValueError("Role parameters differ from frozen output configuration")
    return {role: args["max_completion_tokens"] for role, args in expected.items()}


def digest(value):
    """Hash canonical JSON, including every configuration field."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    """Never overwrite an existing preparation artifact."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, value):
    """Serialize local commits; retry only short Windows sharing failures.

    Never retry transport/model operations. A failed unique temporary file is
    retained as recovery evidence; the previous published JSON remains valid.
    """
    path = Path(path)
    with _JSON_WRITER:
        fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(4):
            try:
                os.replace(name, path)
                return
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 3:
                    raise
                time.sleep(0.025 * (attempt + 1))


def read_monitor_snapshot(folder):
    """Read only immutable published progress, never the active request ledger."""
    paths = sorted((Path(folder) / "monitor").glob("*.json"))
    if not paths:
        return {"status": "no_published_snapshot"}
    return {"snapshot_file": str(paths[-1]), **read_json(paths[-1])}


def code_identity(repo):
    """Include untracked memory code and the exact reused execution wrappers."""
    paths = list((ROOT / "src/tau2").rglob("*.py"))
    paths += list((ROOT / "validation/memory_comparison").glob("*.py"))
    paths += [ROOT / "validation/run_memory_comparison.py"]
    paths += list(BUDGET.glob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += [ROOT / "validation/repository_state.py"]
    # Public replay inputs are part of offline experiment identity as well.
    paths += [HISTORY / name for name in ("captured_agent_inputs.json",
        "trajectory.json", "source.json", "mock_frames.json")]
    return {"upstream_head":repo.get("upstream_head", UPSTREAM_HEAD),
        "development_head":repo["head"], "branch":repo["branch"],
        "tracked_patch_sha256":repo["tracked_patch_sha256"],
        "source_hashes":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
        "dependencies":{n:version(n) for n in ("tau2","litellm","httpx","openai","tiktoken")}}


def task_identity(task, environment):
    from tau2.domains.retail import utils
    paths = [utils.RETAIL_DB_PATH, utils.RETAIL_TASK_SET_PATH, utils.RETAIL_POLICY_PATH,
        utils.RETAIL_DATA_DIR / "split_tasks.json"]
    return {"task_sha256":digest(task.model_dump(mode="json")),
        "policy_sha256":digest(environment.get_policy()),
        "tools_sha256":digest([t.openai_schema for t in environment.get_tools()]),
        "data_files":{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        "user_guidelines_files":{p.relative_to(ROOT).as_posix():sha(p)
            for p in sorted((ROOT/"data/tau2/user_simulator").glob("*.md"))}}


def make_pair(comparison_id, repo, calibration, task, environment, *, output_config_version=OUTPUT_CONFIG_VERSION):
    import qianwen_adapter as adapter
    from output_budget import CONFIG_VERSION
    from tau2.agent.input_tokens import LocalJsonTokenCounter

    common = {"comparison_id":comparison_id, "domain":"retail", "split":"train",
        "task_id":"34", "trial":0, "num_trials":1, "concurrency":1,
        "agent":"memory_agent", "user":"user_simulator", "user_persona_config":None,
        "code":code_identity(repo), "inputs":task_identity(task,environment),
        "models":deepcopy(adapter.MODELS), "api_base":adapter.BASE_URL, "proxy":PROXY,
        "role_args":configured_role_args(output_config_version),
        "output_budget_version":CONFIG_VERSION,
        "output_config_version":output_config_version,
        "seed_rule":{"orchestrator_seed":42,"agent_and_user_state_seed":42,
            "provider_seed_parameter":"not_forwarded_by_existing_adapter",
            "paired_responses":False},
        "limits":{"group_requests":20,"comparison_requests":40,"max_steps":30,
            "max_errors":1,"task_timeout_seconds":300,"http_phase_timeout_seconds":30,
            "retries":{"sdk":0,"http":0,"task":0}},
        "evaluation":{"type":"ALL","reward_basis":["DB"],"strict_replay":True,
            "nl_judge":False,"review":False,"communication_validation":True},
        "input_counter":LocalJsonTokenCounter().describe(),
        "calibration_sha256":digest(calibration),
        "comparison_interpretation":"full-history reference versus input-constrained window tradeoff",
        "supplier_total_output_v2_verified":False}
    a = {**deepcopy(common),"group":"A","run_id":comparison_id+"_A",
        "memory":{"name":"full_history","version":"1","config":{}}}
    budget = calibration["selection"]["recommended_max_input_tokens"]
    b = {**deepcopy(common),"group":"B","run_id":comparison_id+"_B",
        "memory":{"name":"budget_window","version":"1","config":{
            "max_input_tokens":budget,"safety_margin_tokens":calibration["selection"]["safety_margin_tokens"]}}}
    plan = {"schema":"memory-comparison-v1","comparison_id":comparison_id,"groups":[a,b],
        "budget_target_cny":None,"price_status":"unknown","currency_hard_cap":False,
        "authorization_status":"pending_not_approved","live_executed":False}
    validate_pair(plan)
    plan["identity_sha256"] = digest(plan)
    return plan


def validate_pair(plan, *, task_ids=("34",), total_requests=40, reward_basis=("DB",)):
    """Fail on ANY non-strategy difference, rather than maintaining a partial allowlist."""
    if len(plan.get("groups",[])) != 2:
        raise ValueError("Exactly two groups required")
    a,b = plan["groups"]
    if (a["group"],b["group"]) != ("A","B"):
        raise ValueError("Group order must be A then B")
    if a["memory"] != {"name":"full_history","version":"1","config":{}}:
        raise ValueError("FullHistory must have no clipping target")
    if b["memory"]["name"] != "budget_window" or b["memory"]["version"] != "1":
        raise ValueError("Second group must use BudgetWindow v1")
    config = b["memory"]["config"]
    if (set(config) != {"max_input_tokens","safety_margin_tokens"}
            or type(config["max_input_tokens"]) is not int or config["max_input_tokens"] <= 0
            or type(config["safety_margin_tokens"]) is not int or config["safety_margin_tokens"] < 0):
        raise ValueError("Explicit valid BudgetWindow configuration required")
    remove = {"group","run_id","memory"}
    left = {k:v for k,v in a.items() if k not in remove}
    right = {k:v for k,v in b.items() if k not in remove}
    if left != right:
        raise ValueError("Non-strategy group conditions differ")
    if any(g["comparison_id"] != plan["comparison_id"] or
            g["run_id"] != plan["comparison_id"]+"_"+g["group"] for g in (a,b)):
        raise ValueError("Run identity mismatch")
    import qianwen_adapter as adapter
    from output_budget import CONFIG_VERSION
    if a["domain"] != "retail" or a["split"] != "train" or a["task_id"] not in task_ids:
        raise ValueError("Task is outside the frozen Retail train selection")
    if a["agent"] != "memory_agent" or a["models"] != adapter.MODELS:
        raise ValueError("Model/agent scope mismatch")
    if (a["user"] != "user_simulator" or a["user_persona_config"] is not None
            or a["api_base"] != adapter.BASE_URL or a["proxy"] != PROXY
            or a["output_budget_version"] != CONFIG_VERSION
            or (a["trial"],a["num_trials"],a["concurrency"]) != (0,1,1)
            or a["seed_rule"] != {"orchestrator_seed":42,"agent_and_user_state_seed":42,
                "provider_seed_parameter":"not_forwarded_by_existing_adapter","paired_responses":False}):
        raise ValueError("Unimplemented or changed effective run conditions")
    group_output_limits(a)
    if a["limits"] != {"group_requests":20,"comparison_requests":total_requests,"max_steps":30,
            "max_errors":1,"task_timeout_seconds":300,"http_phase_timeout_seconds":30,
            "retries":{"sdk":0,"http":0,"task":0}}:
        raise ValueError("Execution limits changed")
    if a["evaluation"] != {"type":"ALL","reward_basis":list(reward_basis),"strict_replay":True,
            "nl_judge":False,"review":False,"communication_validation":True}:
        raise ValueError("Scoring/Judge/Review conditions changed")
    return {"status":"passed","allowed_differences":["group","run_id","memory"]}


def check_identity(plan):
    if plan.get("schema") == "memory-suite-v1":
        from memory_comparison.evaluation import validate_suite
        validate_suite(plan)
    else:
        validate_pair(plan)
    payload = {k:v for k,v in plan.items() if k != "identity_sha256"}
    if digest(payload) != plan["identity_sha256"]:
        raise ValueError("Frozen comparison changed")
    for rel, expected in plan["groups"][0]["code"]["source_hashes"].items():
        if sha(ROOT/rel) != expected:
            raise ValueError("Source identity changed: " + rel)
    for rel, expected in plan["groups"][0]["inputs"]["data_files"].items():
        if sha(ROOT/rel) != expected:
            raise ValueError("Frozen data identity changed: " + rel)
    for rel, expected in plan["groups"][0]["inputs"]["user_guidelines_files"].items():
        if sha(ROOT/rel) != expected:
            raise ValueError("Frozen UserSimulator guidelines changed: " + rel)


def require_authorization(plan, authorization, *, mode="live"):
    """Run BEFORE any credential reader or real transport construction."""
    if normalize_mode(mode) != "live":
        raise PermissionError("Paid authorization is only valid for live execution")
    check_identity(plan)
    if authorization is None:
        raise PermissionError("New comparison-bound paid authorization required; no Key read")
    auth = read_json(authorization)
    if "currency_policy" in plan and (auth.get("budget_target_cny") != 10 or
            auth.get("authorization_id") != plan.get("authorization_id")):
        raise PermissionError("Currency scope or authorization ID mismatch; no Key read")
    if (auth.get("approved") is not True or auth.get("phase") != "Phase1-memory-comparison-live"
            or auth.get("comparison_id") != plan["comparison_id"]
            or auth.get("identity_sha256") != plan["identity_sha256"]
            or auth.get("group_request_caps") != {g["group"]:g["limits"]["group_requests"] for g in plan["groups"]}
            or auth.get("total_request_cap") != plan["groups"][0]["limits"]["comparison_requests"]
            or type(auth.get("budget_target_cny")) not in (int,float)
            or not math.isfinite(auth["budget_target_cny"])
            or auth["budget_target_cny"] <= 0):
        raise PermissionError("Missing fresh identity/count/monetary approval; no Key read")
    return auth


class RequestLimitStop(RuntimeError):
    """A persistent pre-I/O request resource stop."""


class RequestLedger:
    """Reopen counts without resetting; each execution may be started only once."""

    def __init__(self, folder, identity, *, per_group=20, total=40, mode="mock", local_context_groups=()):
        mode = normalize_mode(mode)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.path = self.folder / "comparison_request_ledger.json"
        self.identity = identity
        self.last_refusal = None
        self.fees = None
        self.local_context_groups = tuple(local_context_groups)
        if self.path.exists():
            self.data = read_json(self.path)
            if (self.data["identity"],self.data["per_group"],self.data["total"],normalize_mode(self.data["mode"])) != (identity,per_group,total,mode):
                raise ValueError("Cannot reset or rebind an existing ledger")
            if self.data.get("local_context_groups", []) != list(local_context_groups):
                raise ValueError("Cannot change the recorded continuation policy")
        else:
            self.data = {"identity":identity,"per_group":per_group,"total":total,
                "mode":mode,"attempts":[],"groups":{},"shared_stop":None}
            if local_context_groups:
                self.data["local_context_groups"] = list(local_context_groups)
            write_new(self.path,self.data)

    def start_once(self):
        if (self.data["attempts"] or self.data["groups"] or self.data["shared_stop"]
                or self.data.get("a_to_b_gate") is not None):
            raise FileExistsError("Historical execution/pause state must not be restarted")
        with (self.folder / "execution_started.lock").open("x",encoding="utf-8") as handle:
            handle.write(self.identity)

    def _commit(self):
        """The existing ledger is authoritative; monitoring has immutable files."""
        atomic_json(self.path, self.data)
        monitor = self.folder / "monitor"
        monitor.mkdir(exist_ok=True)
        latest = self.data["attempts"][-1] if self.data["attempts"] else {}
        snapshot = {"identity": self.identity, "mode": self.data["mode"],
            "requests": len(self.data["attempts"]), "shared_stop": self.data["shared_stop"],
            "latest_attempt": {k: latest.get(k) for k in ("group", "ordinal", "role", "status", "http_status")},
            "groups": {g: {k: v.get(k) for k in ("status", "requests", "official_reward", "stop_reason")}
                       for g, v in self.data["groups"].items()}}
        if self.fees is not None:
            snapshot["currency_budget"] = self.fees.totals()
        atomic_json(monitor / (f"{time.time_ns():020d}-" + uuid.uuid4().hex[:8] + ".json"), snapshot)

    def record_gate(self, decision, *, key="a_to_b_gate"):
        """Commit and read back the decision before B can be constructed."""
        if not key.replace("_", "").isalnum():
            raise ValueError("Invalid local gate identifier")
        if self.data.get(key) is not None:
            raise FileExistsError("Existing gate decision must not be overwritten")
        self.data[key] = deepcopy(decision)
        if not decision["allowed"] and not self.data["shared_stop"]:
            self.data["shared_stop"] = decision["primary_reason"]
        self._commit()
        gate_path = self.folder / (key + ".json")
        atomic_json(gate_path,decision)
        if (read_json(self.path).get(key) != decision
                or read_json(gate_path) != decision):
            raise RuntimeError("Gate persistence readback failed")

    def sync(self, group, value):
        """Called by existing CountedTransport's pre-I/O persistence callback."""
        for event in value["events"]:
            key = [group,event["ordinal"]]
            existing = next((e for e in self.data["attempts"] if e["key"] == key),None)
            if existing is None:
                used = sum(e["group"] == group for e in self.data["attempts"])
                if used >= self.data["per_group"] or len(self.data["attempts"]) >= self.data["total"]:
                    self.last_refusal = "persistent_request_limit_reached"
                    raise RequestLimitStop("persistent_request_limit_reached")
                if event["status"] != "reserved_before_io":
                    raise RuntimeError("Attempt was not persisted before I/O")
                if self.data["shared_stop"]:
                    raise RequestLimitStop("shared_execution_paused")
                if normalize_mode(event.get("mode")) != normalize_mode(self.data["mode"]):
                    raise ValueError("Request mode differs from execution ledger")
                existing = {"key":key,"group":group}
                self.data["attempts"].append(existing)
            else:
                if normalize_mode(event.get("mode")) != normalize_mode(self.data["mode"]):
                    raise ValueError("Request mode differs from execution ledger")
                for field in ("role","endpoint","parameters","mode","sdk"):
                    previous, current = existing.get(field), event.get(field)
                    if field == "mode":
                        previous, current = normalize_mode(previous), normalize_mode(current)
                    if field in existing and field in event and previous != current:
                        raise ValueError("Reserved request identity cannot be changed")
            existing.update(deepcopy(event))
        self._commit()

    def finish_group(self, group, result):
        self.data["groups"][group] = deepcopy(result)
        local_stop = (group in self.local_context_groups and result.get("status") == "resource_stop"
                      and result.get("stop_reason") == "context_budget_exceeded"
                      and result.get("error_type") == "ContextBudgetError"
                      and result.get("official_reward") is None)
        if result.get("pause_comparison") and not local_stop:
            self.data["shared_stop"] = result["stop_reason"]
        self._commit()
