"""Pre-release consistency checks.

Checks:
1) Version consistency: pyproject.toml == llama_conductor/__about__.py and router uses __version__ in /healthz.
2) License consistency: pyproject.toml, LICENSE, README.md, FAQ.md agree on AGPL family.
3) DOCS-TRUTH-MAP active doc references exist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ABOUT = ROOT / "llama_conductor" / "__about__.py"
ROUTER = ROOT / "llama_conductor" / "router_fastapi.py"
LICENSE = ROOT / "LICENSE"
README = ROOT / "README.md"
FAQ = ROOT / "FAQ.md"
TRUTH_MAP_CANDIDATES = (
    ROOT / "docs" / "index" / "DOCS-TRUTH-MAP.md",
    ROOT / "DOCS-TRUTH-MAP.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_truth_map() -> Optional[Path]:
    for candidate in TRUTH_MAP_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _extract_pyproject_field(text: str, key: str) -> str:
    m = re.search(rf'^\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "")


def _extract_about_version(text: str) -> str:
    m = re.search(r'^\s*__version__\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "")


def _is_agpl_text(text: str) -> bool:
    low = text.lower()
    return "gnu affero general public license" in low or "agpl" in low


def _active_doc_refs(text: str) -> list[str]:
    refs: list[str] = []
    in_active = False
    for line in text.splitlines():
        if line.strip().lower().startswith("## active docs"):
            in_active = True
            continue
        if in_active and line.strip().startswith("## "):
            break
        if not in_active:
            continue
        refs.extend(re.findall(r"`([^`]+)`", line))
    return refs


def _is_public_split_repo() -> bool:
    # Public split intentionally omits Cliniko pipeline/docs/assets.
    return not (ROOT / "llama_conductor" / "commands" / "cliniko.py").exists()


def _is_expected_missing_active_ref(ref: str, *, public_split: bool) -> bool:
    if not public_split:
        return False
    expected_prefixes = (
        "llama_conductor/test tools/",
        "ChatGPT/",
    )
    expected_exact = {
        "FIX-THIS-CLINIKO.md",
        "docs/cliniko/FIX-THIS-CLINIKO.md",
        "llama_conductor/CLINIKO-GOLD.md",
        "SMOKE-SCRATCHPAD.md",
        "docs/validation/SMOKE-SCRATCHPAD.md",
    }
    if ref in expected_exact:
        return True
    return any(ref.startswith(p) for p in expected_prefixes)


def main() -> int:
    errors: list[str] = []

    pyproject_text = _read(PYPROJECT)
    about_text = _read(ABOUT)
    router_text = _read(ROUTER)
    license_text = _read(LICENSE)
    readme_text = _read(README)
    faq_text = _read(FAQ)
    truth_map_path = _resolve_truth_map()

    # 1) Version consistency
    pyproject_version = _extract_pyproject_field(pyproject_text, "version")
    about_version = _extract_about_version(about_text)
    if not pyproject_version:
        errors.append("Missing [project].version in pyproject.toml")
    if not about_version:
        errors.append("Missing __version__ in llama_conductor/__about__.py")
    if pyproject_version and about_version and pyproject_version != about_version:
        errors.append(
            f"Version mismatch: pyproject.toml={pyproject_version} vs __about__.py={about_version}"
        )
    if "from .__about__ import __version__" not in router_text:
        errors.append("router_fastapi.py does not import __version__ from __about__.py")
    if 'return {"ok": True, "version": __version__}' not in router_text:
        errors.append("router_fastapi.py /healthz is not wired to __version__")

    # 2) License consistency (AGPL family)
    pyproject_license = _extract_pyproject_field(pyproject_text, "license")
    if not pyproject_license:
        errors.append("Missing [project].license in pyproject.toml")
    elif "agpl" not in pyproject_license.lower():
        errors.append(f"pyproject.toml license is not AGPL-family: {pyproject_license}")

    if not _is_agpl_text(license_text):
        errors.append("LICENSE file does not appear to be AGPL-family text")
    if "agpl-3.0-or-later" not in readme_text.lower():
        errors.append("README.md missing AGPL-3.0-or-later declaration")
    if "agpl-3.0-or-later" not in faq_text.lower():
        errors.append("FAQ.md missing AGPL-3.0-or-later declaration")

    # 3) Active-doc references must exist (if truth map is present)
    public_split = _is_public_split_repo()
    if truth_map_path is not None:
        truth_map_text = _read(truth_map_path)
        for ref in _active_doc_refs(truth_map_text):
            ref_path = ROOT / ref
            if not ref_path.exists() and not _is_expected_missing_active_ref(ref, public_split=public_split):
                errors.append(f"{truth_map_path.relative_to(ROOT)} active doc reference missing on disk: {ref}")
    elif not public_split:
        errors.append("Missing DOCS-TRUTH-MAP.md in non-public repo")

    if errors:
        print("[pre-release] FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[pre-release] PASS")
    print(f"- version: {pyproject_version}")
    print(f"- license: {pyproject_license}")
    if truth_map_path is None:
        print("- docs-truth-map: not present (skipped)")
    else:
        print("- docs-truth-map active references: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
