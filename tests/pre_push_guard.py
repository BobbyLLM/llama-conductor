"""Pre-push guard: compile + consistency checks.

Fails push when:
- Python sources fail to compile
- pre-release consistency checks fail
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_consistency_check() -> int:
    script = ROOT / "tests" / "pre_release_consistency_check.py"
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT))


def _run_compile_check() -> bool:
    # Deterministic and dependency-light compile pass.
    ok_pkg = compileall.compile_dir(
        str(ROOT / "llama_conductor"),
        force=False,
        quiet=1,
    )
    ok_cli = compileall.compile_file(
        str(ROOT / "llama_conductor_cli.py"),
        force=False,
        quiet=1,
    )
    return bool(ok_pkg and ok_cli)


def main() -> int:
    print("[pre-push] running compile check...")
    if not _run_compile_check():
        print("[pre-push] FAIL: compile check failed")
        return 1
    print("[pre-push] compile check: PASS")

    print("[pre-push] running consistency check...")
    rc = _run_consistency_check()
    if rc != 0:
        print("[pre-push] FAIL: consistency check failed")
        return rc

    print("[pre-push] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

