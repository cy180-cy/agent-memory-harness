"""Classify raw protocol and actual adapted behavior without rewriting either."""
import json


def assess(raw, adapted, tools, *, tool_loop_verified=False):
    reasons = []
    result = {"raw_protocol_check": {"status": "failed"},
              "adapter_compatibility_check": {"status": "failed"},
              "kind": "invalid", "tool_candidate_valid": False, "reasons": reasons}
    if not isinstance(raw, dict) or not all(k in raw for k in ("id", "model", "choices")):
        reasons.append("incomplete_response")
        return result
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        reasons.append("incomplete_or_ambiguous_choices")
        return result
    choice = choices[0]
    message = choice.get("message")
    finish = choice.get("finish_reason")
    result["raw_protocol_check"]["finish_reason"] = finish
    if not isinstance(message, dict) or message.get("role") != "assistant":
        reasons.append("missing_assistant_message")
        return result
    if finish not in ("stop", "tool_calls") or message.get("refusal"):
        reasons.append("unsafe_finish_or_refusal")
        return result
    calls = message.get("tool_calls")
    if not calls:
        # Ordinary text is not a tool call, even if it claims to call a tool.
        if finish == "stop" and isinstance(message.get("content"), str) and message["content"].strip() and adapted:
            target = adapted.get("message", {})
            if not target.get("tool_calls") and target.get("content") == message["content"]:
                result.update(kind="text")
                result["raw_protocol_check"]["status"] = "passed"
                result["adapter_compatibility_check"] = {"status": "passed", "tool_exception_applied": False}
                return result
        reasons.append("no_structured_tool_calls_or_valid_text")
        return result
    if not isinstance(calls, list) or not adapted:
        reasons.append("tool_structure_or_conversion_error")
        return result
    available = {tool.name: tool for tool in tools}
    identities, expected = set(), []
    try:
        for call in calls:
            ident = call.get("id")
            function = call.get("function", {})
            name = function.get("name")
            if not isinstance(ident, str) or not ident.strip() or ident in identities:
                raise ValueError("missing_or_duplicate_call_id")
            identities.add(ident)
            if call.get("type") != "function" or name not in available:
                raise ValueError("unknown_or_invalid_tool")
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("arguments_not_object")
            tool = available[name]
            schema = tool.openai_schema["function"]["parameters"]
            if set(arguments) - set(schema.get("properties", {})):
                raise ValueError("unexpected_argument")
            tool.params.model_validate(arguments, strict=True)
            expected.append({"id": ident, "name": name, "arguments": arguments})
        normalized = adapted["litellm_response"]["choices"][0]
        result["adapter_compatibility_check"]["finish_reason"] = normalized.get("finish_reason")
        target = adapted["message"].get("tool_calls") or []
        actual = [{k: item[k] for k in ("id", "name", "arguments")} for item in target]
        norm_calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])} for c in normalized["message"]["tool_calls"]]
        if actual != expected or norm_calls != expected or normalized.get("finish_reason") != "tool_calls":
            raise ValueError("conversion_changed_tool_or_finish")
        if adapted["message"].get("content"):
            raise ValueError("mixed_text_and_tools")
    except Exception as error:
        reasons.append(str(error) if isinstance(error, ValueError) else type(error).__name__)
        return result
    result.update(kind="tools", tool_candidate_valid=True, verified_calls=expected)
    result["raw_protocol_check"].update(status="passed" if finish == "tool_calls" else "failed", difference="stop_with_structured_tools" if finish == "stop" else None)
    result["adapter_compatibility_check"].update(status="passed" if tool_loop_verified else "awaiting_loop", tool_exception_applied=finish == "stop", loop_verified=tool_loop_verified)
    return result
