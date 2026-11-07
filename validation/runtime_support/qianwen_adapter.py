"""Phase0-06.1 copy: total-output parameter and saved-response resource gate.

No finish_reason, thinking mode, model, temperature, or usage rewriting.
"""
import contextlib
import io
import json
import logging
import os
import sys
import time
from copy import deepcopy
from output_budget import TOTAL_OUTPUT_LIMITS, CONFIG_VERSION, ResourceStop

BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
KEY_ENV = "AMH_QIANWEN_API_KEY"
MODELS = {"agent": "openai/qwen3.7-plus", "user": "openai/qwen3.8-flash"}
LIMITS = TOTAL_OUTPUT_LIMITS
TRANSPORT_TOKEN = object()
ECHO_TOOL = {"type": "function", "function": {"name": "echo", "description": "Echo a short text for interface-schema validation only.",
    "parameters": {"type": "object", "title": "parameters", "properties": {"text": {"title": "Text", "type": "string"}}, "required": ["text"]}}}


class MissingCredentialError(RuntimeError):
    pass


class InterfaceRequestError(RuntimeError):
    pass


def role_args(role):
    if role not in MODELS:
        raise ValueError("Only agent and user roles are enabled; Judge and Review disabled")
    return {"api_base": BASE_URL, "temperature": 0.0, "max_completion_tokens": LIMITS[role],
            "timeout": 30.0, "num_retries": 0, "max_retries": 0, "drop_params": False, "stream": False}


def plan():
    return {"phase": "Phase0-06.1", "config_version": CONFIG_VERSION,
        "live_execution_enabled": False, "real_api_requests_authorized": 0,
        "key_env_name": KEY_ENV, "models": MODELS, "api_base": BASE_URL,
        "answer_only_limits": {"agent": None, "user": None},
        "total_output_limits": LIMITS, "parameters": {r: role_args(r) for r in MODELS},
        "supplier_enforcement_verified": False, "local_output_tolerance": 0,
        "provider_cost": {"status": "unknown", "amount": None}}


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {k: "[REDACTED]" if any(s in k.lower() for s in ("api_key", "authorization", "password")) else redact(v, secret) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secret) for v in value]
    return value


class RedactingStream(io.TextIOBase):
    def __init__(self, target, secret):
        self.target, self.secret = target, secret

    def write(self, text):
        self.target.write(redact(text, self.secret))
        return len(text)

    def flush(self):
        self.target.flush()


class SecretFilter(logging.Filter):
    def __init__(self, secret):
        super().__init__()
        self.secret = secret

    def filter(self, record):
        record.msg = redact(record.getMessage(), self.secret)
        record.args = ()
        record.exc_info = record.exc_text = None
        return True


def usage_record(raw):
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    counts = {n: raw.get(n) if isinstance(raw, dict) and type(raw.get(n)) is int and raw[n] >= 0 else None for n in names}
    status = "known" if all(v is not None for v in counts.values()) else "partial" if any(v is not None for v in counts.values()) else "unknown"
    return {"status": status, **counts, "source": "supplier_http_body" if isinstance(raw, dict) else "absent"}


def invoke(role, messages, *, client, key_reader, tools=None, tool_choice=None,
           evidence=None, message_observer=None, budget_session, persist_evidence,
           total_output_budget=None):
    """Same upstream/LiteLLM path, restricted to the counted Phase0-05 transport."""
    kwargs = role_args(role)
    budget_session.before_request()
    if not callable(persist_evidence):
        raise ValueError("A durable response evidence sink is required before generation")
    if getattr(client._transport, "budget_session", None) is not budget_session:
        raise InterfaceRequestError("HTTP transport must share the same budget session")
    if total_output_budget is not None:
        if type(total_output_budget) is not int or total_output_budget != LIMITS[role]:
            raise ValueError("Only this draft's explicit role-specific total budget is enabled")
        kwargs["max_completion_tokens"] = total_output_budget
    key = key_reader(KEY_ENV)
    if not key or not key.strip():
        raise MissingCredentialError(f"Missing process variable {KEY_ENV}; request not started")
    import httpx
    if getattr(client._transport, "approval_token", None) is not TRANSPORT_TOKEN:
        raise InterfaceRequestError("A verified Phase0-05 transport is required")
    import litellm
    from tau2.utils import llm_utils
    from litellm.litellm_core_utils import logging_utils as timing_logs
    from types import SimpleNamespace
    from litellm.llms.openai.openai import OpenAIChatCompletion

    observed = evidence if evidence is not None else {}

    def persist(stage):
        snapshot = redact(deepcopy(observed), key)
        snapshot.update(stage=stage, role=role, parameters=kwargs,
            source_mode="real" if client._transport.live else "mock")
        try:
            persist_evidence(snapshot)
        except Exception:
            budget_session.persistence_failed()
            budget_session.before_request()

    def observe(response):
        observed["http_status"] = response.status_code
        observed["supplier_request_ids"] = {k: redact(response.headers[k], key) for k in ("x-request-id", "x-dashscope-request-id", "request-id", "trace-id") if k in response.headers}
        if any(key in response.headers[k] for k in ("x-request-id", "x-dashscope-request-id", "request-id", "trace-id") if k in response.headers):
            observed["credential_echo_detected"] = True
        response.read()
        if str(response.request.url) != BASE_URL + "/chat/completions":
            raise InterfaceRequestError("Unexpected endpoint")
        try:
            body = response.json()
        except (ValueError, TypeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        observed["http_status"] = response.status_code
        observed["response_model"] = body.get("model")
        observed["response_id"] = body.get("id")
        observed["raw_response"] = redact({k: body[k] for k in ("id", "object", "created", "model", "choices", "usage") if k in body}, key)
        observed["supplier_request_ids"] = {k: redact(response.headers[k], key) for k in ("x-request-id", "x-dashscope-request-id", "request-id", "trace-id") if k in response.headers}
        if key in response.text:
            observed["credential_echo_detected"] = True
        observed["usage"] = redact(deepcopy(body.get("usage")), key)
        choices = body.get("choices") or []
        observed["finish_reason"] = choices[0].get("finish_reason") if choices else None
        # The unmodified body, usage, request ID and attempt evidence are saved first.
        persist("raw_response_saved_before_budget_check")
        observed["resource_check"] = budget_session.after_saved_response(role,
            observed["usage"], kwargs["max_completion_tokens"])
        persist("resource_decision_saved")

    backend = litellm.completion

    def guarded_completion(**request):
        budget_session.before_request()
        if request.get("model") != MODELS[role] or request.get("api_base") != BASE_URL:
            raise InterfaceRequestError("Unapproved model/endpoint; Judge and Review blocked")
        if any(k in request for k in ("api_key", "headers", "extra_headers", "fallbacks")):
            raise InterfaceRequestError("Credentials/headers/fallbacks forbidden in request kwargs")
        if request.get("num_retries") != 0 or request.get("max_retries") != 0:
            raise InterfaceRequestError("Retries must remain zero")
        return backend(**request)

    old = {n: getattr(litellm, n) for n in ("api_key", "openai_key", "client_session", "in_memory_llm_clients_cache", "turn_off_message_logging")}
    old_timing_logger = timing_logs._service_logger
    old_completion, old_cost, old_logger = llm_utils.completion, llm_utils.get_response_cost, llm_utils.logger
    old_usage = llm_utils.get_response_usage
    old_log_dir, old_log_mode = llm_utils.llm_log_dir.get(), llm_utils.llm_log_mode.get()
    old_factory = OpenAIChatCompletion._get_openai_client

    def checked_factory(instance, *args, **options):
        if options.get("max_retries") != 0:
            raise InterfaceRequestError("SDK retries not disabled before request")
        sdk_client = old_factory(instance, *args, **options)
        client._transport.verify_sdk(sdk_client)
        observed["sdk_verification"] = dict(client._transport.sdk_verification)
        return sdk_client
    hooks = list(client.event_hooks["response"])
    handlers = {h for obj in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()] if isinstance(obj, logging.Logger) for h in obj.handlers}
    filter_ = SecretFilter(key)

    def patch_log(record):
        record["message"] = redact(record["message"], key)
        record["exception"] = None

    start = time.perf_counter()
    try:
        litellm.api_key, litellm.openai_key, litellm.client_session = key, None, client
        litellm.in_memory_llm_clients_cache = type(old["in_memory_llm_clients_cache"])()
        litellm.turn_off_message_logging = True
        # Optional service telemetry creates an event-loop socket pair even with
        # no callbacks. Disable only these process-local timing sinks, retaining
        # all network guards and the explicit upstream request audit logs.
        timing_logs._service_logger = SimpleNamespace(service_success_hook=lambda **k: None, service_failure_hook=lambda **k: None)
        llm_utils.completion = guarded_completion
        # No known gateway pricing: do not call the upstream zero-on-error estimator.
        llm_utils.get_response_cost = lambda response: None
        def supplier_usage(response):
            counts = usage_record(observed.get("usage"))
            return None if counts["status"] != "known" else {k: counts[k] for k in ("prompt_tokens", "completion_tokens")}
        llm_utils.get_response_usage = supplier_usage
        llm_utils.logger = old_logger.patch(patch_log)
        llm_utils.set_llm_log_dir(None)  # Curated runner evidence only; never raw SDK/error logs.
        llm_utils.set_llm_log_mode("all")
        OpenAIChatCompletion._get_openai_client = checked_factory
        client.event_hooks["response"] = hooks + [observe]
        for handler in handlers:
            handler.addFilter(filter_)
        with contextlib.redirect_stdout(RedactingStream(sys.stdout, key)), contextlib.redirect_stderr(RedactingStream(sys.stderr, key)):
            try:
                message = llm_utils.generate(model=MODELS[role], messages=messages, tools=tools,
                    tool_choice=tool_choice, call_name=f"qianwen_{role}_interface", **kwargs)
            except Exception as error:
                observed["conversion_error_type"] = type(error).__name__
                persist("conversion_error_saved")
                budget_session.before_request()
                if not client._transport.live:
                    observed["preflight_error_detail"] = redact(str(error), key)
                # No provider error body, headers, credential, or chained exception in public errors.
                raise InterfaceRequestError(f"{role} interface request failed ({type(error).__name__}); provider details redacted") from None
        usage = usage_record(observed.get("usage"))
        observed["litellm_usage"] = deepcopy((message.raw_data or {}).get("usage"))
        persist("conversion_saved_before_message_release")
        if observed.get("credential_echo_detected"):
            client._transport.stopped = True
            raise InterfaceRequestError("Credential echo detected; run stopped")
        # Never release an over-budget/unknown-budget tool call to Agent/Environment.
        budget_session.before_request()
        message.cost = None
        message.usage = None if usage["status"] != "known" else {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}
        converted = {"status": "incomplete" if observed.get("finish_reason") == "length" else "received",
            "role": role, "model": MODELS[role], "parameters": kwargs,
            "message": redact(message.model_dump(mode="json", exclude={"raw_data"}), key),
            "litellm_response": redact({k: message.raw_data[k] for k in ("id", "object", "created", "model", "choices", "usage") if message.raw_data and k in message.raw_data}, key),
            "finish_reason": observed.get("finish_reason"), "usage": usage,
            "resource_check": deepcopy(observed["resource_check"]),
            "provider_cost": {"status": "unknown", "amount": None, "currency": None},
            "elapsed_seconds": round(time.perf_counter() - start, 6), "task_success": None,
            "transport": "phase0_counted_transport", "real_model_request": client._transport.live}
        from compatibility import normalize_mixed_response
        normalization = normalize_mixed_response(observed.get("raw_response"), converted, tools or [], messages)
        if normalization is not None:
            observed["mixed_content_normalization"] = normalization
            converted["mixed_content_normalization"] = redact(normalization, key)
            persist("mixed_content_decision_saved_before_message_release")
            if normalization["status"] != "normalized":
                client._transport.stopped = True
                raise InterfaceRequestError("Mixed response cannot be safely normalized; see saved audit")
            # Only this copy reaches tau2. raw_data and supplier evidence retain
            # the original text, finish reason, usage and ordered tool calls.
            message = message.model_copy(deep=True)
            message.content = None
            converted["message"] = redact(message.model_dump(mode="json", exclude={"raw_data"}), key)
        if message_observer is not None:
            message_observer(message)
        return converted
    finally:
        client.event_hooks["response"] = hooks
        for name, value in old.items():
            setattr(litellm, name, value)
        timing_logs._service_logger = old_timing_logger
        llm_utils.completion, llm_utils.get_response_cost, llm_utils.logger = old_completion, old_cost, old_logger
        llm_utils.get_response_usage = old_usage
        llm_utils.set_llm_log_dir(old_log_dir)
        llm_utils.set_llm_log_mode(old_log_mode)
        OpenAIChatCompletion._get_openai_client = old_factory
        for handler in handlers:
            handler.removeFilter(filter_)


if __name__ == "__main__":
    # No credential lookup, imports of model libraries, or live execution here.
    print(json.dumps(plan(), ensure_ascii=False, indent=2))
