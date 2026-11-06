"""Portable fixture setup; independent of the frozen original scenario module."""
import json
from pathlib import Path
import uuid
import memory_support as support

FIXTURES = Path(__file__).parent / "fixtures/memory"

def fixture(name):
    """Read a small provenance-labelled public regression fixture."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

def synthetic_plan(profile=None, *, omit_legacy_version=False):
    """Build a fresh unapproved test identity, never replay a paid approval."""
    from memory_comparison import spec
    from memory_comparison.execution import retail_task
    from tau2.runner.build import build_environment
    if support.RUNTIME is None:
        raise RuntimeError("Use a guarded offline entry for test configuration")
    repo = json.loads((support.RUNTIME[1] / "repository_state.json").read_text(encoding="utf-8"))
    value = spec.make_pair("synthetic_fixture_" + uuid.uuid4().hex[:12], repo,
        {"selection": {"recommended_max_input_tokens": 7168, "safety_margin_tokens": 128}},
        retail_task(), build_environment("retail"),
        output_config_version=profile or spec.OUTPUT_CONFIG_VERSION)
    if omit_legacy_version:
        if profile != spec.LEGACY_OUTPUT_CONFIG:
            raise ValueError("Only the explicit legacy fixture omits its version")
        for group in value["groups"]:
            del group["output_config_version"]
        value["identity_sha256"] = spec.digest({k:v for k,v in value.items() if k != "identity_sha256"})
    return value

def stale_authorization(folder):
    """Invalid synthetic scope: cannot authorize any live request."""
    from memory_comparison import spec
    path = Path(folder) / ("invalid_synthetic_authorization_" + uuid.uuid4().hex[:8] + ".json")
    spec.write_new(path, {"approved": False, "phase": "synthetic_expired_not_live",
                          "comparison_id": "not-this-comparison", "source_kind": "synthetic"})
    return path

def stage_result_excerpt(folder, name):
    """Materialize an allowlisted result-reader fixture only inside test output."""
    from memory_comparison import spec
    folder = Path(folder)
    folder.mkdir()
    value = fixture(name)
    if name == "legacy_metrics_excerpt.json":
        spec.write_new(folder / "summary.json", value["summary"])
        spec.write_new(folder / "effective_config.json", {"task_ids": [value["task_id"]]})
        spec.write_new(folder / "request_ledger.json", {"actual_api_requests": 0, "mock_http_requests": 0,
            "synthetic_no_http": True, "source_kind": "historical_metrics_fixture_materialization", "events": [
            {k:e[k] for k in ("ordinal", "role", "http_status")} for e in value["events"]]})
        spec.write_new(folder / "model_responses.json", [
            {"evidence": {"raw_response": {"usage": e["usage"]}}} for e in value["events"]])
    else:
        for key in ("effective_group", "summary", "official_reward", "agent_state", "trajectory"):
            spec.write_new(folder / (key + ".json"), value[key])
    spec.write_new(folder / "fixture_provenance.json", {k:value[k] for k in ("source_kind", "source_sha256")})
    return folder
