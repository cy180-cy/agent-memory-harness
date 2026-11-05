"""Phase0-06 counted HTTP transport copy; future total-budget mapping only."""
import json
import time
from offline_safety import require
import qianwen_adapter as adapter

# Bound by the comparison runner only after its proxy/configuration checks.
PROXY: str

class CountedTransport:
    """Construct an HTTPX transport with the Phase0-05 pre-dispatch SDK checks."""
    @staticmethod
    def create(inner, *, safety, max_requests, caps, live=False, persist=None, secret="", budget_session=None):
        import httpx
        require(isinstance(inner, httpx.MockTransport) if not live else safety.live_authorized and isinstance(inner, httpx.HTTPTransport), "Unapproved transport")

        class Transport(httpx.BaseTransport):
            approval_token = adapter.TRANSPORT_TOKEN

            def __init__(self):
                self.live, self.role, self.sdk_verification = live, None, {}
                self.events, self.blocked = [], []
                self.budget_session = budget_session
                self.deadline, self.stopped = None, False

            def verify_sdk(self, sdk):
                require(sdk.max_retries == 0, "SDK retries not zero")
                require(str(sdk.base_url).rstrip("/") == adapter.BASE_URL, "SDK base mismatch")
                timeout = sdk.timeout
                require(timeout == 30 if isinstance(timeout, (int, float)) else all(getattr(timeout, n) == 30 for n in ("connect", "read", "write", "pool")), "SDK timeout mismatch")
                require(sdk._client.follow_redirects is False and sdk._client._trust_env is False, "Unsafe HTTP client")
                self.sdk_verification = {"max_retries": 0, "http_retries": 0, "task_retries": 0, "timeout_seconds": 30, "trust_env": False, "follow_redirects": False}

            def save(self):
                if persist:
                    persist(adapter.redact({"actual_api_requests": len(self.events) if live else 0,
                                             "mock_http_requests": 0 if live else len(self.events),
                                             "events": self.events, "blocked": self.blocked}, secret))

            def handle_request(self, request):
                try:
                    if budget_session is not None:
                        budget_session.before_request()
                    require(not self.stopped, "Run already stopped; no additional requests")
                    require(self.deadline is None or time.perf_counter() < self.deadline, "Task deadline reached before dispatch")
                    require(self.role in adapter.MODELS and self.sdk_verification, "Unknown role or missing SDK verification; Judge blocked")
                    require(len(self.events) < max_requests, "Hard model request count reached")
                    require(request.method == "POST" and str(request.url) == adapter.BASE_URL + "/chat/completions", "Unapproved endpoint")
                    body = json.loads(request.content)
                    require(body.get("model") == adapter.MODELS[self.role].split("/", 1)[1], "Role/model mismatch")
                    require(body.get("stream", False) is False and body.get("temperature") == 0, "Unexpected stream/temperature")
                    require("max_tokens" not in body, "Conflicting answer-only max_tokens is forbidden")
                    require(type(body.get("max_completion_tokens")) is int and body["max_completion_tokens"] == caps[self.role], "Total-output budget mapping mismatch")
                    require(request.headers.get("x-stainless-retry-count") == "0", "Retry header mismatch")
                    require(not live or inner._pool._retries == 0, "HTTP retries not zero")
                except Exception as error:
                    self.stopped = True
                    self.blocked.append({"reason": str(error), "role": self.role})
                    self.save()
                    raise
                # Reserve and persist BEFORE I/O; uncertainty and errors keep their slot.
                event = {"ordinal": len(self.events) + 1, "role": self.role,
                         "endpoint": str(request.url), "parameters": adapter.redact(body, secret),
                         "sdk": dict(self.sdk_verification), "status": "reserved_before_io",
                         "mode": "real" if live else "mock", "proxy": PROXY if live else None}
                self.events.append(event)
                self.save()
                start = time.perf_counter()
                try:
                    safety.proxy_dispatch_active = live
                    response = inner.handle_request(request)
                    response.read()
                    event.update(status="response_received", http_status=response.status_code)
                    if response.status_code >= 400:
                        self.stopped = True
                    return response
                except Exception as error:
                    self.stopped = True
                    event.update(status="transport_error", error_type=type(error).__name__)
                    raise
                finally:
                    safety.proxy_dispatch_active = False
                    event["seconds"] = round(time.perf_counter() - start, 6)
                    self.save()

            def close(self):
                inner.close()

        return Transport()
