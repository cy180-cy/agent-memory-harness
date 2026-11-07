"""Small offline regressions for the Phase1-07 reservation/continuation repair."""

from copy import deepcopy
from decimal import Decimal
import uuid

import pytest

import memory_support as support
from memory_comparison import evaluation, execution, spec
from memory_comparison.fees import FeeBudget, FeeStop, PRICE_POLICY, token_cost
from test_comparison_user_output import synthetic_builder, normal_frames


@pytest.fixture
def folder():
    if support.RUNTIME is None:
        pytest.skip("Use --tests --development-tests fees")
    path = support.RUNTIME[1] / ("fees_" + uuid.uuid4().hex[:10])
    path.mkdir()
    return path


@pytest.fixture
def plan():
    repo = spec.read_json(support.RUNTIME[1] / "repository_state.json")
    return evaluation.make_suite("synthetic_fees_" + uuid.uuid4().hex[:10], repo,
                                 evaluation.task_manifest(("69", "73")), phase1_07=True)


class Counter:
    def count_request(self, body):
        return 1000

    def describe(self):
        return {"name": "synthetic-fixed-1000"}


def usage(prompt=800, completion=500):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "completion_tokens_details": {"reasoning_tokens": 400}}


def budget(folder, policy=None):
    ledger = spec.RequestLedger(folder, "synthetic", total=180)
    ledger.fees = FeeBudget(ledger, policy or PRICE_POLICY, Counter())
    return ledger, ledger.fees


def test_reservation_settlement_tier_and_reasoning_no_double_charge(folder):
    ledger, fees = budget(folder)
    fees.reserve("A", 1, "agent", {"max_completion_tokens": 3072})
    durable = spec.read_json(ledger.path)["currency_budget"]["entries"][0]
    assert (durable["reserved_input_tokens"], durable["reserved_output_tokens"]) == (1200, 3072)
    assert Decimal(durable["reserved_cny"]) == Decimal("0.0215808")
    fees.settle("A", 1, "agent", usage(), 200)
    assert fees.totals()["estimated_spent_cny"] == "0.00448"
    assert fees.totals()["held_reservation_cny"] == "0"
    assert fees.data["entries"][0]["raw_usage"] == usage()
    assert token_cost(PRICE_POLICY, "agent", 256001, 500) == Decimal("1.2384048")
    reopened = spec.RequestLedger(folder, "synthetic", total=180)
    reloaded = FeeBudget(reopened, PRICE_POLICY, Counter())
    assert reloaded.totals() == fees.totals()
    with pytest.raises(FileExistsError):
        reopened.data["shared_stop"] = "synthetic_consumed"
        reopened.start_once()


def test_fee_limit_stops_before_io_and_preserves_previous_spend(folder):
    policy = deepcopy(PRICE_POLICY)
    policy["limit_cny"] = "0.03"
    ledger, fees = budget(folder, policy)
    fees.reserve("A", 1, "agent", {"max_completion_tokens": 3072})
    fees.settle("A", 1, "agent", usage(completion=2000), 200)
    with pytest.raises(FeeStop, match="cost_budget_exceeded_before_send"):
        fees.reserve("A", 2, "agent", {"max_completion_tokens": 3072})
    assert len(fees.data["entries"]) == 1
    assert fees.totals()["estimated_spent_cny"] == "0.01408"
    assert spec.read_json(ledger.path)["shared_stop"] == "cost_budget_exceeded_before_send"


@pytest.mark.parametrize("raw", [None, {"prompt_tokens": "800", "completion_tokens": 10, "total_tokens": 810}])
def test_unknown_usage_retains_reservation_and_pauses(raw, folder):
    ledger, fees = budget(folder)
    fees.reserve("A", 1, "agent", {"max_completion_tokens": 3072})
    fees.settle("A", 1, "agent", raw, 200)
    assert fees.group_record("A")["amount_cny"] is None
    assert fees.totals()["unknown_attempts"] == 1
    assert Decimal(fees.totals()["held_reservation_cny"]) > 0
    with pytest.raises(FeeStop, match="cost_verification_unknown"):
        fees.reserve("B", 1, "agent", {"max_completion_tokens": 3072})


def test_failed_reservation_persistence_cannot_reach_dispatch(folder, monkeypatch):
    ledger, fees = budget(folder)
    def fail():
        raise PermissionError("synthetic persistent local write failure")
    monkeypatch.setattr(ledger, "_commit", fail)
    sent = []
    with pytest.raises(PermissionError):
        fees.reserve("A", 1, "user", {"max_completion_tokens": 1024})
        sent.append(True)
    assert not sent and not ledger.data["attempts"]


def test_real_transport_boundary_reserves_before_mock_and_denies_excess(plan, folder, synthetic_builder):
    policy = deepcopy(PRICE_POLICY)
    policy["limit_cny"] = "0.000001"
    ledger, fees = budget(folder, policy)
    result = execution.run_group(plan["groups"][0], folder / "group", ledger,
                                 support.RUNTIME[0], frames=normal_frames())
    assert result["requests"] == result["real_api_requests"] == result["mock_http_requests"] == 0
    assert result["stop_reason"] == "cost_budget_exceeded_before_send"
    assert list((folder / "group/received_responses").iterdir()) == []


@pytest.mark.parametrize("fault", ["none", "context", "unknown_usage", "timeout"])
def test_actual_suite_continuation_and_shared_faults(plan, folder, synthetic_builder, monkeypatch, fault):
    from tau2.agent.budget_window import ContextBudgetError
    from tau2.agent.memory_agent import MemoryAgent
    original = MemoryAgent._build_model_context
    if fault == "context":
        def limited(self, state):
            if self.memory_strategy.name == "structured_memory":
                diagnostic = {"pause_reason": "context_budget_exceeded", "status": "paused",
                              "source": "synthetic_injected_minimum_set_limit"}
                state.context_selections.append(diagnostic)
                raise ContextBudgetError("context_budget_exceeded", diagnostic)
            return original(self, state)
        monkeypatch.setattr(MemoryAgent, "_build_model_context", limited)
    frames = {g["group"]: normal_frames() for g in plan["groups"]}
    first = plan["groups"][0]["group"]
    if fault == "unknown_usage":
        frames[first][0]["body"].pop("usage")
    elif fault == "timeout":
        frames[first][0]["transport_error"] = "timeout"
    result = evaluation.run_suite(plan, folder, support.RUNTIME[0], frames_by_group=frames)
    durable = spec.read_json(folder / "comparison_request_ledger.json")
    assert result["real_api_requests"] == 0
    assert len(durable["attempts"]) == result["mock_http_requests"]
    if fault in ("none", "context"):
        assert len(synthetic_builder) == 6 and result["stop_reason"] is None
        if fault == "context":
            assert result["gates"][2]["allowed"] and result["gates"][2]["local_context_stop"]
            for pos in (2, 5):
                assert result["groups"][pos]["official_reward"] is None
                assert result["groups"][pos]["stop_reason"] == "context_budget_exceeded"
        else:
            assert all(g["status"] == "completed" for g in result["groups"])
            assert all(g["official_reward"] == 1 for g in result["groups"])
        assert all(r["status"] == "settled" for r in durable["currency_budget"]["entries"])
    else:
        assert len(synthetic_builder) == 1
        assert result["mock_http_requests"] == 1
        assert all(g["status"] == "not_executed" and g["requests"] == 0 for g in result["groups"][1:])
        assert durable["currency_budget"]["entries"][0]["status"] == "unknown"
        if fault == "unknown_usage":
            raw = spec.read_json(folder / first / "received_responses/0001.json")
            assert raw["http_status"] == 200 and "usage" not in raw["raw_response"]
    before = (folder / "comparison_request_ledger.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.run_suite(plan, folder, support.RUNTIME[0], frames_by_group=frames)
    assert before == (folder / "comparison_request_ledger.json").read_bytes()


def test_new_authorization_identity_and_limit_before_key(plan, folder, monkeypatch):
    import run_memory_comparison as entry
    repo = spec.read_json(support.RUNTIME[1] / "repository_state.json")
    reads = []
    with pytest.raises(PermissionError):
        entry.live_preflight(plan, None, folder / "live", repo, lambda name: reads.append(name))
    assert not reads
    # Exercise the same exclusive authorization-ID binding without creating a
    # real authorization file or ever reading an actual credential.
    monkeypatch.setattr(entry, "ROOT", folder)
    (folder / "validation/phase1_comparison").mkdir(parents=True)
    monkeypatch.setattr(spec, "require_authorization", lambda *a, **k: {"synthetic_only": True})
    entry.live_preflight(plan, None, folder / "live", repo, lambda name: "MOCK-NOT-A-KEY")
    with pytest.raises(FileExistsError):
        entry.live_preflight(plan, None, folder / "different_run", repo, lambda name: reads.append(name))
    assert not reads
