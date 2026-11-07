"""Local CNY reservation in the existing request ledger; no billing API."""

from copy import deepcopy
from decimal import Decimal
import math

PRICE_POLICY = {
    "version": "phase1-07-user-quoted-cny-v1",
    "limit_cny": "10", "input_reserve_multiplier": "1.20",
    "agent_tier_threshold_tokens": 256000,
    "agent": {"input": "1.60", "output": "6.40",
              "large_input": "4.80", "large_output": "19.20"},
    "user": {"input": "0.80", "output": "2.70"},
    "basis": "user-provided console rates per million tokens; no cache/free discounts",
    "supplier_bill_verified": False,
}


class FeeStop(RuntimeError):
    def __init__(self, reason):
        self.stop_reason = reason
        super().__init__(reason)


def token_cost(policy, role, prompt_tokens, completion_tokens):
    """Reasoning is already included in completion_tokens, never added twice."""
    prices = policy[role]
    large = role == "agent" and prompt_tokens > policy["agent_tier_threshold_tokens"]
    input_price = Decimal(prices["large_input" if large else "input"])
    output_price = Decimal(prices["large_output" if large else "output"])
    return (prompt_tokens * input_price + completion_tokens * output_price) / Decimal(1000000)


class FeeBudget:
    """One sequential suite; reservations and settlements use its durable ledger."""
    def __init__(self, ledger, policy, counter=None):
        from tau2.agent.input_tokens import LocalJsonTokenCounter
        self.ledger = ledger
        self.counter = counter or LocalJsonTokenCounter()
        self.policy = deepcopy(policy)
        existing = ledger.data.get("currency_budget")
        if existing is not None and existing["policy"] != policy:
            raise ValueError("Cannot reset or reprice an existing currency ledger")
        if existing is None:
            ledger.data["currency_budget"] = {"policy": deepcopy(policy), "entries": [],
                "stop_reason": None, "cost_status": "estimated_from_quoted_prices",
                "supplier_bill_cny": None}
            ledger._commit()
        self.data = ledger.data["currency_budget"]

    def totals(self, group=None):
        rows = [r for r in self.data["entries"] if group is None or r["group"] == group]
        spent = sum((Decimal(r["estimated_cost_cny"]) for r in rows if r["status"] == "settled"), Decimal(0))
        held = sum((Decimal(r["reserved_cny"]) for r in rows if r["status"] != "settled"), Decimal(0))
        return {"estimated_spent_cny": str(spent), "held_reservation_cny": str(held),
                "unknown_attempts": sum(r["status"] == "unknown" for r in rows),
                "pending_attempts": sum(r["status"] == "reserved" for r in rows)}

    def stop(self, reason):
        self.data["stop_reason"] = self.data["stop_reason"] or reason
        self.ledger.last_refusal = self.data["stop_reason"]
        self.ledger.data["shared_stop"] = self.ledger.data["shared_stop"] or reason
        self.ledger._commit()

    def reserve(self, group, ordinal, role, body):
        """Called after HTTP validation but before request-slot allocation or I/O."""
        if self.data["stop_reason"] or self.ledger.data["shared_stop"]:
            raise FeeStop(self.data["stop_reason"] or self.ledger.data["shared_stop"])
        if any(r["status"] != "settled" for r in self.data["entries"]):
            self.stop("cost_verification_unknown")
            raise FeeStop("cost_verification_unknown")
        if any((r["group"], r["ordinal"]) == (group, ordinal) for r in self.data["entries"]):
            raise FeeStop("cost_reservation_already_exists")
        try:
            estimate = self.counter.count_request(body)
            if type(estimate) is not int or estimate < 0:
                raise ValueError("Invalid input estimate")
            reserved_input = math.ceil(Decimal(estimate) * Decimal(self.policy["input_reserve_multiplier"]))
            output = body["max_completion_tokens"]
            if type(output) is not int or output <= 0:
                raise ValueError("Invalid output budget")
            reserve = token_cost(self.policy, role, reserved_input, output)
        except Exception:
            self.stop("cost_estimation_unknown")
            raise FeeStop("cost_estimation_unknown") from None
        total = self.totals()
        if Decimal(total["estimated_spent_cny"]) + Decimal(total["held_reservation_cny"]) + reserve > Decimal(self.policy["limit_cny"]):
            self.stop("cost_budget_exceeded_before_send")
            raise FeeStop("cost_budget_exceeded_before_send")
        self.data["entries"].append({"group": group, "ordinal": ordinal, "role": role,
            "status": "reserved", "input_estimate": estimate, "reserved_input_tokens": reserved_input,
            "reserved_output_tokens": output, "counter": self.counter.describe(),
            "reserved_cny": str(reserve), "estimated_cost_cny": None})
        self.ledger._commit()  # Persistence failure propagates; no request is sent.

    def settle(self, group, ordinal, role, usage, http_status):
        """Called only after raw response evidence was saved, before message release."""
        from output_budget import verify_usage
        row = next(r for r in self.data["entries"] if (r["group"], r["ordinal"]) == (group, ordinal))
        if row["role"] != role or row["status"] != "reserved":
            raise FeeStop("cost_settlement_identity_error")
        row.update(raw_usage=deepcopy(usage), http_status=http_status)
        decision = verify_usage(usage, row["reserved_output_tokens"])
        if decision["status"] == "budget_verification_unknown":
            row["status"] = "unknown"  # Keep full reservation; never substitute zero.
            self.stop("cost_verification_unknown")
            return
        value = token_cost(self.policy, role, usage["prompt_tokens"], usage["completion_tokens"])
        row.update(status="settled", estimated_cost_cny=str(value),
                   released_reservation_cny=str(Decimal(row["reserved_cny"]) - value))
        self.ledger._commit()
        if Decimal(self.totals()["estimated_spent_cny"]) >= Decimal(self.policy["limit_cny"]):
            self.stop("cost_budget_reached_after_response")

    def transport_failed(self, group, ordinal):
        row = next(r for r in self.data["entries"] if (r["group"], r["ordinal"]) == (group, ordinal))
        if row["status"] == "reserved":
            row["status"] = "unknown"
            self.stop("cost_verification_unknown")

    def group_record(self, group):
        totals = self.totals(group)
        unknown = totals["unknown_attempts"] or totals["pending_attempts"]
        return {"status": "unknown" if unknown else "known", **totals,
                "amount_cny": None if unknown else float(Decimal(totals["estimated_spent_cny"])),
                "basis": self.policy["basis"], "kind": "usage_based_estimate_not_supplier_bill",
                "supplier_bill_cny": None, "hard_currency_cap": False}
