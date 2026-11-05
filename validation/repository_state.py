"""Record the current checkout separately from its immutable upstream ancestor."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

UPSTREAM_HEAD = "fc0055dc4e0a316c3f83133267fbd6faaa770992"


def repository(root):
    """Verify repository ownership/ancestry and capture the current patch.

    Normal descendant commits and branch names are allowed. Paid execution still
    requires a fresh authorization bound to the complete source/data identity.
    Git receives only its required OS setting, never inherited model credentials.
    """
    root = Path(root).resolve()
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to verify the experiment identity")
    env = {
        "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
        "GIT_OPTIONAL_LOCKS": "0",
    }

    def read(*args, check=True):
        return subprocess.run(
            [git, "-c", f"safe.directory={root.as_posix()}",
             "--no-optional-locks", *args],
            cwd=root, env=env, check=check, capture_output=True, encoding="utf-8",
        )

    top = read("rev-parse", "--show-toplevel").stdout.strip()
    if Path(top).resolve() != root:
        raise RuntimeError("Repository root mismatch")
    head = read("rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if read("merge-base", "--is-ancestor", UPSTREAM_HEAD, head,
            check=False).returncode != 0:
        raise RuntimeError("Current commit does not descend from the upstream source")
    if read("ls-files", "--unmerged").stdout.strip():
        raise RuntimeError("Unresolved Git conflicts prevent identity capture")
    if read("diff", "--cached", "--name-only").stdout.strip():
        raise RuntimeError("Unexpected staged changes; finish staging before a run")
    branch = read("branch", "--show-current").stdout.strip()
    patch = read("diff", "HEAD", "--binary").stdout
    return {
        "upstream_head": UPSTREAM_HEAD,
        "head": head,
        "branch": branch,
        "tracked_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "tracked_patch": patch,
    }
