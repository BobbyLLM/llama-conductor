#!/usr/bin/env python3
"""Print a scalar value from router_config.yaml using dot-path keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _get_path(cfg: Any, path: str, default: Any) -> Any:
    cur = cfg
    for part in (path or "").split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _to_text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read one value from router_config.yaml")
    parser.add_argument("--config", required=True, help="Path to router_config.yaml")
    parser.add_argument("--key", required=True, help="Dot-path key, e.g. frontend.shim.enabled")
    parser.add_argument("--default", default="", help="Fallback value if key is missing")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(_to_text(args.default))
        return 0

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        print(_to_text(args.default))
        return 0

    value = _get_path(cfg, args.key, args.default)
    print(_to_text(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
