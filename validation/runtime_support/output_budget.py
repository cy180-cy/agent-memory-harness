"""Strict local total-output verification; never rewrite supplier usage."""
from copy import deepcopy

TOTAL_OUTPUT_LIMITS = {"agent": 1024, "user": 512}
CONFIG_VERSION = "total-output-v2"
DOCUMENTED_PROVIDER_TOLERANCE = 10


class ResourceStop(RuntimeError):
    """Resource pause, not a business failure or an official reward."""
    def __init__(self, decision):
        self.decision = deepcopy(decision)
        self.stop_reason = decision["status"]
        super().__init__(self.stop_reason)


def verify_usage(raw, budget):
    result = {"status": "budget_verification_unknown", "budget": budget,
        "budget_semantics": "reasoning_and_final_output", "local_tolerance": 0,
        "supplier_protocol_fault": None, "raw_usage": deepcopy(raw),
        "official_reward": None, "cost": {"status": "unknown", "amount": None}}
    if not isinstance(raw, dict):
        result["reason"] = "usage_missing_or_not_object"
        return result
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(type(raw.get(k)) is not int or raw[k] < 0 for k in keys):
        result["reason"] = "missing_negative_or_non_integer_token_count"
        return result
    if raw["prompt_tokens"] + raw["completion_tokens"] != raw["total_tokens"]:
        result["reason"] = "prompt_plus_completion_does_not_equal_total"
        return result
    details = raw.get("completion_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            result["reason"] = "completion_details_not_object"
            return result
        for key in ("reasoning_tokens", "text_tokens", "audio_tokens"):
            value = details.get(key)
            if value is not None and (type(value) is not int or not 0 <= value <= raw["completion_tokens"]):
                result["reason"] = "invalid_completion_detail_" + key
                return result
        reasoning, text = details.get("reasoning_tokens"), details.get("text_tokens")
        if reasoning is not None and text is not None:
            if reasoning > text:
                result["reason"] = "reasoning_not_subset_of_text"
                return result
            result["derived_visible_text_tokens"] = text - reasoning
    prompt_details = raw.get("prompt_tokens_details")
    if prompt_details is not None:
        if not isinstance(prompt_details, dict):
            result["reason"] = "prompt_details_not_object"
            return result
        for key in ("cached_tokens", "text_tokens"):
            value = prompt_details.get(key)
            if value is not None and (type(value) is not int or not 0 <= value <= raw["prompt_tokens"]):
                result["reason"] = "invalid_prompt_detail_" + key
                return result
    # completion_tokens already includes reasoning. Do not add it again.
    result["accounted_total_output_tokens"] = raw["completion_tokens"]
    excess = raw["completion_tokens"] - budget
    if excess > 0:
        result.update(status="output_budget_exceeded", excess_tokens=excess,
            provider_tolerance_classification="within_documented_10_token_range" if excess <= DOCUMENTED_PROVIDER_TOLERANCE else "outside_documented_10_token_range_not_a_fault_verdict")
    else:
        result.update(status="budget_verified", excess_tokens=0)
    return result


class BudgetSession:
    """One sticky pause shared by both roles and the HTTP boundary."""
    def __init__(self):
        self.decisions = []
        self.stop_decision = None

    def before_request(self):
        if self.stop_decision is not None:
            raise ResourceStop(self.stop_decision)

    def after_saved_response(self, role, raw_usage, budget):
        decision = verify_usage(raw_usage, budget)
        decision["role"] = role
        self.decisions.append(decision)
        if decision["status"] != "budget_verified" and self.stop_decision is None:
            self.stop_decision = decision
        return decision

    def persistence_failed(self):
        self.stop_decision = {"status": "budget_verification_unknown", "reason": "response_evidence_not_durably_saved", "official_reward": None}
