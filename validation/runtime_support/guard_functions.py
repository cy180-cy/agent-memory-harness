"""Original audit functions extracted without semantic changes.

Safety loads these functions into its per-run namespace; OUT, report, current,
expected_denial, Path, os and sys are deliberately supplied by that namespace.
Historical source: phase0_offline/20261102T1450137452726+0800/offline_smoke.py.
"""

import os
import sys
from pathlib import Path

# Injected by Safety into its per-run execution namespace.
OUT: Path
report: dict
current: str
expected_denial: str | None


def violation(kind, entry):
    report["attempts"][kind].append({"stage": current, "entry": entry, "intent": "regression" if expected_denial == kind else "unexpected"})
    raise RuntimeError(f"Offline guard blocked {kind}: {entry}")


def confined(path):
    if isinstance(path, int) or path is None:
        return True
    resolved = Path(os.fsdecode(path)).resolve()
    return resolved == OUT or OUT in resolved.parents


def windows_nul_device(path):
    # Match raw device spellings only: do not strip, resolve, or match substrings.
    return os.name == "nt" and isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).casefold() in {"nul", "\\\\.\\nul"}


def windows_version_probe():
    platform_module = sys.modules.get("platform")
    probe = getattr(platform_module, "_syscmd_ver", None)
    frame = sys._getframe(1)
    while frame is not None:
        if probe is not None and frame.f_code is probe.__code__:
            return frame.f_locals.get("cmd") in {"ver", "command /c ver", "cmd /c ver"}
        frame = frame.f_back
    return False


def audit(event, values):
    if event == "open":
        path, mode, flags = values
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if write and not windows_nul_device(path) and not confined(path):
            violation("write_outside", str(path))
        if not write and not isinstance(path, int):
            parts = [p.lower() for p in Path(os.fsdecode(path)).parts]
            if any(p.startswith(".env") or p in {".aws", ".azure", ".netrc", "_netrc", ".git-credentials"} for p in parts):
                violation("credential_read", str(path))
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate", "os.chown"}:
        if not confined(values[0]):
            violation("write_outside", f"{event}: {values[0]}")
    elif event in {"os.rename", "os.link", "os.symlink"}:
        if not all(confined(v) for v in values[:2]):
            violation("write_outside", event)
    elif event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"}:
        if os.name == "nt" and event == "subprocess.Popen" and windows_version_probe():
            # Still deny process creation. OSError makes this stdlib probe use its
            # existing native Windows fallback instead of aborting the import.
            report["compatibility_events"].append({"stage": current, "event": event, "action": "denied Windows version shell probe; native fallback requested"})
            raise OSError("Offline validation: Windows version shell probe disabled")
        violation("subprocess", event)
    elif event in {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}:
        violation("network", event)
