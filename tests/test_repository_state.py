"""Commit portability and pre-credential authorization regression checks."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import repository_state
import run_memory_comparison as entry
from memory_comparison import spec


def git_result(monkeypatch, *, ancestor=0, top=None, staged="", conflicts=""):
    root = Path(__file__).resolve().parents[1]
    calls = []
    current = "a" * 40  # Synthetic descendant returned by the mocked Git boundary.

    def run(argv, **kwargs):
        args = argv[4:]
        calls.append((args, kwargs))
        outputs = {
            ("rev-parse", "--show-toplevel"): str(top or root),
            ("rev-parse", "--verify", "HEAD^{commit}"): current,
            ("branch", "--show-current"): "feature/later-development",
            ("ls-files", "--unmerged"): conflicts,
            ("diff", "--cached", "--name-only"): staged,
            ("diff", "HEAD", "--binary"): "synthetic tracked patch\n",
        }
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return SimpleNamespace(stdout="", returncode=ancestor)
        return SimpleNamespace(stdout=outputs[tuple(args)], returncode=0)

    monkeypatch.setattr(repository_state.shutil, "which", lambda name: "git")
    monkeypatch.setattr(repository_state.subprocess, "run", run)
    return root, current, calls


def test_descendant_commit_and_new_branch_are_recorded(monkeypatch):
    root, current, calls = git_result(monkeypatch)
    result = repository_state.repository(root)
    assert result["head"] == current != result["upstream_head"]
    assert result["upstream_head"] == repository_state.UPSTREAM_HEAD
    assert result["branch"] == "feature/later-development"
    assert result["tracked_patch_sha256"] == hashlib.sha256(
        result["tracked_patch"].encode()).hexdigest()
    assert all(set(options["env"]) == {"SystemRoot", "GIT_OPTIONAL_LOCKS"}
               for _, options in calls)
    assert ["merge-base", "--is-ancestor", repository_state.UPSTREAM_HEAD, current] in [
        args for args, _ in calls]


@pytest.mark.parametrize("options,reason", [
    ({"ancestor": 1}, "does not descend"),
    ({"top": Path(__file__).parent}, "root mismatch"),
    ({"staged": "pending.py"}, "staged changes"),
    ({"conflicts": "unmerged entry"}, "Unresolved Git conflicts"),
])
def test_invalid_repository_is_still_rejected(monkeypatch, options, reason):
    root, _, _ = git_result(monkeypatch, **options)
    with pytest.raises(RuntimeError, match=reason):
        repository_state.repository(root)


def test_old_commit_approval_is_rejected_before_key_read(monkeypatch):
    # Only replace the earlier authorization boundary to isolate the new check.
    monkeypatch.setattr(spec, "require_authorization", lambda *a, **k: {"synthetic": True})
    plan = {"groups": [{"code": {
        "development_head": "b" * 40, "tracked_patch_sha256": "same",
    }}]}
    reads = []
    with pytest.raises(PermissionError, match="Development commit changed"):
        entry.live_preflight(plan, None, Path(__file__).parent,
                             {"head": "a" * 40, "tracked_patch_sha256": "same"},
                             lambda name: reads.append(name))
    assert reads == []


def test_missing_approval_does_not_read_credentials(monkeypatch):
    # Run the actual require_authorization branch, bypassing only plan validation.
    monkeypatch.setattr(spec, "check_identity", lambda plan: None)
    reads = []
    with pytest.raises(PermissionError, match="paid authorization required"):
        entry.live_preflight({}, None, Path(__file__).parent, {},
                             lambda name: reads.append(name))
    assert reads == []
