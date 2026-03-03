#!/usr/bin/env python3
"""Generate llama-server --models-preset INI from router_config.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def _as_ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalize_path_like(value: Any, base_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if lowered.startswith(("http://", "https://", "hf://")):
        return raw
    p = Path(raw)
    if p.is_absolute():
        return p.as_posix()
    if "\\" in raw or "/" in raw or raw.startswith("."):
        return (base_dir / p).resolve().as_posix()
    return raw


def _render_ini(cfg: Dict[str, Any], cfg_path: Path, ctx_size: int) -> str:
    section = (
        cfg.get("llama_server", {})
        .get("models_preset", {})
    )
    if not isinstance(section, dict):
        raise ValueError("llama_server.models_preset must be a mapping")

    enabled = bool(section.get("enabled", True))
    if not enabled:
        raise ValueError("llama_server.models_preset.enabled is false")

    defaults = section.get("defaults", {})
    models = section.get("models", {})

    if not isinstance(defaults, dict):
        raise ValueError("llama_server.models_preset.defaults must be a mapping")
    if not isinstance(models, dict) or not models:
        raise ValueError("llama_server.models_preset.models must be a non-empty mapping")

    cfg_dir = cfg_path.parent
    lines = ["version = 1", "", "[*]"]

    merged_defaults = dict(defaults)
    merged_defaults.setdefault("c", int(ctx_size))
    for key, value in merged_defaults.items():
        if value is None:
            continue
        value = _normalize_path_like(value, cfg_dir)
        lines.append(f"{key} = {_as_ini_value(value)}")

    lines.append("")

    for model_id, opts in models.items():
        name = str(model_id).strip()
        if not name:
            continue
        lines.append(f"[{name}]")
        if isinstance(opts, dict):
            for key, value in opts.items():
                if value is None:
                    continue
                value = _normalize_path_like(value, cfg_dir)
                lines.append(f"{key} = {_as_ini_value(value)}")
        elif isinstance(opts, str):
            value = _normalize_path_like(opts, cfg_dir)
            lines.append(f"mmproj = {_as_ini_value(value)}")
        else:
            raise ValueError(f"models.{name} must be mapping or string")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate llama-server models preset INI from router_config.yaml"
    )
    parser.add_argument("--config", required=True, help="Path to router_config.yaml")
    parser.add_argument("--out", required=True, help="Output path for generated INI")
    parser.add_argument("--ctx-size", type=int, default=8192, help="Default context size for [*].c")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    out_path = Path(args.out).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("router config root must be a mapping")

    ini_text = _render_ini(cfg, cfg_path=cfg_path, ctx_size=int(args.ctx_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ini_text, encoding="ascii", newline="\n")
    print(f"[preset-gen] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
