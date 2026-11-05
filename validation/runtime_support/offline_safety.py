"""Reusable write/credential guards; deny all real I/O by default."""
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GUARD_SOURCE = Path(__file__).with_name("guard_functions.py")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Safety:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.proxy_dispatch_active = False
        self.live_authorized = False
        self.events = {"attempts": {k: [] for k in ("network", "model", "write_outside", "credential_read", "subprocess")}, "compatibility_events": []}
        self.namespace = {"Path": Path, "OUT": self.out, "os": os, "sys": sys,
                          "report": self.events, "current": "offline", "expected_denial": None}

    def install(self):
        require(self.out == OUT or OUT in self.out.parents, "Output must remain in this new run")
        names = {"violation", "confined", "windows_nul_device", "windows_version_probe", "audit"}
        tree = ast.parse(GUARD_SOURCE.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == names, "Audit guard source changed")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GUARD_SOURCE), "exec"), self.namespace)
        base = self.namespace["audit"]

        def audit(event, values):
            # Future approved live entry: only a counted dispatch to the pinned proxy.
            # This branch remains disabled throughout Phase0-05.1.
            if self.live_authorized and self.proxy_dispatch_active:
                if event == "socket.getaddrinfo" and values[:2] == ("127.0.0.1", 7890):
                    return
                if event == "socket.connect" and values[1] == ("127.0.0.1", 7890):
                    return
            base(event, values)

        sys.dont_write_bytecode = True
        sys.addaudithook(audit)
        # Remove values by name only: no inherited real credentials are read offline.
        for name in list(os.environ):
            upper = name.upper()
            if any(x in upper for x in ("API_KEY", "SECRET", "PASSWORD", "CREDENTIAL")) or upper.endswith("_TOKEN"):
                del os.environ[name]
        os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "DO_NOT_TRACK": "1",
                           "HF_HUB_OFFLINE": "1", "OTEL_SDK_DISABLED": "true",
                           "AWS_EC2_METADATA_DISABLED": "true", "TMP": str(self.out),
                           "TEMP": str(self.out), "TMPDIR": str(self.out)})
        import dotenv
        import dotenv.main
        dotenv.load_dotenv = dotenv.main.load_dotenv = lambda *a, **k: False
        dotenv.dotenv_values = dotenv.main.dotenv_values = lambda *a, **k: {}
        from loguru import logger
        logger.remove()
        logging.disable(logging.CRITICAL)

    def expected(self, kind, operation):
        old = self.namespace["expected_denial"]
        self.namespace["expected_denial"] = kind
        count = len(self.events["attempts"][kind])
        try:
            operation()
        except RuntimeError as error:
            require(str(error).startswith("Offline guard blocked " + kind), "Wrong rejection")
        else:
            raise AssertionError("Guard did not reject operation")
        finally:
            self.namespace["expected_denial"] = old
        require(len(self.events["attempts"][kind]) == count + 1, "Missing guard event")


def initialize_models():
    import litellm
    from types import SimpleNamespace
    from litellm.litellm_core_utils import logging_utils
    litellm.telemetry = False
    litellm.turn_off_message_logging = True
    logging_utils._service_logger = SimpleNamespace(service_success_hook=lambda **k: None, service_failure_hook=lambda **k: None)
    # All LiteLLM generation must use our verified synchronous transport.
    original = litellm.completion

    def guarded(**kwargs):
        import qianwen_adapter as adapter
        client = litellm.client_session
        if getattr(getattr(client, "_transport", None), "approval_token", None) is not adapter.TRANSPORT_TOKEN:
            raise RuntimeError("Offline guard blocked model: unverified transport")
        return original(**kwargs)

    litellm.completion = guarded
    for name in ("acompletion", "text_completion", "atext_completion", "embedding", "aembedding", "responses", "aresponses"):
        if hasattr(litellm, name):
            setattr(litellm, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Offline guard blocked model: disabled entry")))
    return litellm


def verify_protected():
    baseline = json.loads((OUT / "baseline.json").read_text(encoding="utf-8-sig"))
    differences = []
    for entry in baseline["protected"]:
        path = Path(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or path.stat().st_mtime_ns != entry["mtime_ns"] or sha(path).lower() != entry["sha256"].lower():
            differences.append(entry["relativePath"])
    require(not differences, "Protected historical files changed")
    return {"status": "passed", "protected_files": len(baseline["protected"]), "differences": differences}
