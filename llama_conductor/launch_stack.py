#!/usr/bin/env python3
"""
Python-first stack launcher for llama-conductor.

Goal: replace fragile batch orchestration with a portable command flow.
This module does not alter existing launch behavior unless explicitly used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional
from urllib.error import URLError
from urllib.request import urlopen

import yaml


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _get_path(cfg: Mapping[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _http_ready(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout_s) as resp:  # nosec B310 - local health probes only
            return 200 <= int(getattr(resp, "status", 0)) < 500
    except URLError:
        return False
    except Exception:
        return False


def _wait_http_ready(url: str, timeout_s: float, label: str) -> bool:
    deadline = time.monotonic() + max(0.5, float(timeout_s))
    while time.monotonic() < deadline:
        if _http_ready(url):
            return True
        time.sleep(0.5)
    print(f"[launch] ERROR: {label} not ready within {timeout_s:.1f}s -> {url}")
    return False


def _spawn(cmd: List[str], cwd: Path, env: Mapping[str, str], dry_run: bool) -> Optional[subprocess.Popen]:
    cmd_str = " ".join(cmd)
    print(f"[launch] spawn: {cmd_str}")
    if dry_run:
        return None

    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": dict(env),
    }
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **kwargs)  # noqa: S603


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("router_config root must be a mapping")
    return cfg


@dataclass
class LaunchPlan:
    provider: str
    router_port: int
    router_host: str
    upstream_base_url: str
    upstream_chat_url: str
    shim_enabled: bool
    shim_host: str
    shim_port: int
    ready_timeout_s: float


def _build_plan(cfg: Dict[str, Any], args: argparse.Namespace) -> LaunchPlan:
    provider = str(_get_path(cfg, "backend.provider", "llama_cpp")).strip().lower()
    if not provider:
        provider = "llama_cpp"

    router_port = _as_int(args.router_port if args.router_port is not None else _get_path(cfg, "port", 9000), 9000)
    router_host = str(
        args.router_host
        or _get_path(cfg, "launcher.router_host", "0.0.0.0")
        or "0.0.0.0"
    )

    local_llama_port = _as_int(
        args.llama_port if args.llama_port is not None else _get_path(cfg, "backend.llama_cpp.port", _get_path(cfg, "backend.local_llama_port", 8010)),
        8010,
    )
    upstream_base_url = str(_get_path(cfg, "backend.upstream_base_url", f"http://127.0.0.1:{local_llama_port}")).rstrip("/")
    upstream_chat_url = str(_get_path(cfg, "backend.upstream_chat_url", f"{upstream_base_url}/v1/chat/completions")).rstrip("/")
    if provider == "llama_cpp":
        upstream_base_url = f"http://127.0.0.1:{local_llama_port}"
        upstream_chat_url = f"{upstream_base_url}/v1/chat/completions"

    shim_enabled_cfg = _as_bool(_get_path(cfg, "frontend.shim.enabled", True), default=True)
    shim_enabled = bool(shim_enabled_cfg and not args.no_shim)
    shim_host = str(args.shim_host or _get_path(cfg, "frontend.shim.host", "0.0.0.0") or "0.0.0.0")
    shim_port = _as_int(
        args.shim_port if args.shim_port is not None else _get_path(cfg, "frontend.shim.port", os.getenv("MOA_CHAT_PORT", "8088")),
        8088,
    )
    ready_timeout_arg = getattr(args, "ready_timeout", None)
    ready_timeout_s = _as_float(
        ready_timeout_arg if ready_timeout_arg is not None else _get_path(cfg, "launcher.ready_timeout_s", 22.0),
        22.0,
    )

    return LaunchPlan(
        provider=provider,
        router_port=router_port,
        router_host=router_host,
        upstream_base_url=upstream_base_url,
        upstream_chat_url=upstream_chat_url,
        shim_enabled=shim_enabled,
        shim_host=shim_host,
        shim_port=shim_port,
        ready_timeout_s=ready_timeout_s,
    )


def _generate_models_preset(
    python_exe: str,
    router_dir: Path,
    config_path: Path,
    out_path: Path,
    ctx_size: int,
    dry_run: bool,
) -> bool:
    script = router_dir / "llama_conductor" / "generate_models_preset.py"
    if not script.exists():
        print(f"[launch] ERROR: missing preset generator: {script}")
        return False
    cmd = [
        python_exe,
        str(script),
        "--config",
        str(config_path),
        "--out",
        str(out_path),
        "--ctx-size",
        str(int(ctx_size)),
    ]
    print(f"[launch] preset-gen: {' '.join(cmd)}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(router_dir), check=False)
    if proc.returncode != 0:
        print(f"[launch] ERROR: preset generation failed (exit={proc.returncode})")
        return False
    return out_path.exists()


def _run_up(args: argparse.Namespace) -> int:
    router_dir = Path(args.router_dir).resolve()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[launch] ERROR: config not found: {config_path}")
        return 2
    if not router_dir.exists():
        print(f"[launch] ERROR: router dir not found: {router_dir}")
        return 2

    cfg = _load_yaml(config_path)
    plan = _build_plan(cfg, args)

    py = args.python_exe or sys.executable
    router_cli = router_dir / "llama_conductor_cli.py"
    shim_dir = router_dir / "llama_conductor" / "shim"
    shim_py = shim_dir / "shim_server.py"

    if not router_cli.exists():
        print(f"[launch] ERROR: router launcher missing: {router_cli}")
        return 2

    env_router: MutableMapping[str, str] = dict(os.environ)
    env_router["BACKEND_PROVIDER"] = plan.provider
    env_router["BACKEND_UPSTREAM_BASE_URL"] = plan.upstream_base_url
    env_router["BACKEND_UPSTREAM_CHAT_URL"] = plan.upstream_chat_url

    llama_port = _as_int(
        args.llama_port if args.llama_port is not None else _get_path(cfg, "backend.llama_cpp.port", _get_path(cfg, "backend.local_llama_port", 8010)),
        8010,
    )
    if plan.provider == "llama_cpp":
        llama_exe_raw = (
            args.llama_exe
            or os.getenv("LLAMA_SERVER_EXE", "")
            or str(_get_path(cfg, "backend.llama_cpp.exe_path", "") or "")
        )
        models_dir_raw = (
            args.models_dir
            or os.getenv("LLAMA_SERVER_MODELS_DIR", "")
            or str(_get_path(cfg, "backend.llama_cpp.models_dir", "") or "")
        )
        llama_exe = Path(llama_exe_raw).resolve() if llama_exe_raw else None
        models_dir = Path(models_dir_raw).resolve() if models_dir_raw else None
        if llama_exe is None or not llama_exe.exists():
            print("[launch] ERROR: llama_cpp provider requires backend.llama_cpp.exe_path or --llama-exe/LLAMA_SERVER_EXE.")
            return 2
        if models_dir is None or not models_dir.exists():
            print("[launch] ERROR: llama_cpp provider requires backend.llama_cpp.models_dir or --models-dir/LLAMA_SERVER_MODELS_DIR.")
            return 2

        models_preset = (
            args.models_preset
            or os.getenv("LLAMA_SERVER_MODELS_PRESET", "")
            or str(_get_path(cfg, "backend.llama_cpp.models_preset_path", "") or "")
        )
        if not models_preset:
            models_preset = str(models_dir / "llama-router-models.ini")
        models_preset_path = Path(models_preset).resolve()

        llama_host = str(args.llama_host or _get_path(cfg, "backend.llama_cpp.host", "0.0.0.0") or "0.0.0.0")
        llama_ctx = _as_int(args.llama_ctx if args.llama_ctx is not None else _get_path(cfg, "backend.llama_cpp.ctx_size", 8192), 8192)
        threads = _as_int(args.threads if args.threads is not None else _get_path(cfg, "backend.llama_cpp.threads", 12), 12)
        threads_batch = _as_int(args.threads_batch if args.threads_batch is not None else _get_path(cfg, "backend.llama_cpp.threads_batch", 12), 12)
        keep = _as_int(args.keep if args.keep is not None else _get_path(cfg, "backend.llama_cpp.keep", 96), 96)
        batch_size = _as_int(args.batch_size if args.batch_size is not None else _get_path(cfg, "backend.llama_cpp.batch_size", 512), 512)
        ubatch_size = _as_int(args.ubatch_size if args.ubatch_size is not None else _get_path(cfg, "backend.llama_cpp.ubatch_size", 256), 256)

        if not _generate_models_preset(
            python_exe=py,
            router_dir=router_dir,
            config_path=config_path,
            out_path=models_preset_path,
            ctx_size=llama_ctx,
            dry_run=args.dry_run,
        ):
            return 2

        llama_cmd = [
            str(llama_exe),
            "--models-dir",
            str(models_dir),
            "--models-preset",
            str(models_preset_path),
            "-t",
            str(threads),
            "--threads-batch",
            str(threads_batch),
            "-c",
            str(llama_ctx),
            "--keep",
            str(keep),
            "--context-shift",
            "-b",
            str(batch_size),
            "-ub",
            str(ubatch_size),
            "--metrics",
            "--port",
            str(llama_port),
            "--host",
            llama_host,
        ]
        _spawn(llama_cmd, cwd=llama_exe.parent, env=env_router, dry_run=args.dry_run)
    else:
        print(f"[launch] provider={plan.provider}; skipping local llama-server launch.")

    router_cmd = [py, str(router_cli), "serve", "--host", plan.router_host, "--port", str(plan.router_port)]
    _spawn(router_cmd, cwd=router_dir, env=env_router, dry_run=args.dry_run)

    if not args.dry_run:
        if plan.provider != "custom":
            if not _wait_http_ready(f"{plan.upstream_base_url}/v1/models", plan.ready_timeout_s, "upstream backend"):
                return 3
        if not _wait_http_ready(f"http://127.0.0.1:{plan.router_port}/v1/models", plan.ready_timeout_s, "llama-conductor"):
            return 3

    if plan.shim_enabled:
        if not shim_py.exists():
            print(f"[launch] ERROR: shim enabled but missing: {shim_py}")
            return 2
        env_shim: MutableMapping[str, str] = dict(os.environ)
        env_shim["MOA_CHAT_HOST"] = plan.shim_host
        env_shim["MOA_CHAT_PORT"] = str(plan.shim_port)
        env_shim["MOA_CHAT_LLAMASERVER_URL"] = plan.upstream_base_url
        env_shim["MOA_CHAT_ROUTER_URL"] = f"http://127.0.0.1:{plan.router_port}"
        env_shim["MOA_CHAT_INJECT_UI"] = "1" if _as_bool(_get_path(cfg, "frontend.shim.inject_ui", True), True) else "0"
        env_shim["MOA_CHAT_AUTH_USER"] = str(_get_path(cfg, "frontend.shim.auth_user", "moa") or "moa")
        env_shim["MOA_CHAT_PASSWORD"] = str(_get_path(cfg, "frontend.shim.auth_password", "") or "")
        env_shim["MOA_CHAT_FORCE_VISION"] = "1" if _as_bool(_get_path(cfg, "frontend.shim.force_vision", False), False) else "0"
        env_shim["MOA_CHAT_RENDER_MODE"] = str(_get_path(cfg, "frontend.shim.render_mode", "stream") or "stream")
        env_shim["MOA_CHAT_UI_MODEL_ALIAS"] = str(_get_path(cfg, "frontend.shim.ui_model_alias", "") or "")
        env_shim["MOA_CHAT_ROUTER_MODEL_ID"] = str(_get_path(cfg, "frontend.shim.router_model_id", "moa-router") or "moa-router")
        env_shim["MOA_CHAT_SINGLE_INSTANCE"] = "1" if _as_bool(_get_path(cfg, "frontend.shim.single_instance", True), True) else "0"
        env_shim["MOA_CHAT_TAKEOVER_EXISTING"] = "1" if _as_bool(_get_path(cfg, "frontend.shim.takeover_existing", True), True) else "0"
        env_shim["MOA_CHAT_TAKEOVER_FOREIGN"] = "1" if _as_bool(_get_path(cfg, "frontend.shim.takeover_foreign", True), True) else "0"
        env_shim["MOA_CHAT_TAKEOVER_TIMEOUT_S"] = str(_as_float(_get_path(cfg, "frontend.shim.takeover_timeout_s", 5.0), 5.0))
        shim_cmd = [py, str(shim_py)]
        _spawn(shim_cmd, cwd=shim_dir, env=env_shim, dry_run=args.dry_run)
        if not args.dry_run:
            if not _wait_http_ready(f"http://127.0.0.1:{plan.shim_port}/shim/healthz", plan.ready_timeout_s, "shim"):
                return 3
    else:
        print("[launch] shim disabled by config/flag; skipping shim launch.")

    print("[launch] launched.")
    print(f"[launch] provider:       {plan.provider}")
    print(f"[launch] upstream chat:  {plan.upstream_chat_url}")
    print(f"[launch] router:         http://127.0.0.1:{plan.router_port}")
    if plan.shim_enabled:
        print(f"[launch] shim:           http://127.0.0.1:{plan.shim_port}")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[doctor] ERROR: missing config: {config_path}")
        return 2
    cfg = _load_yaml(config_path)
    plan = _build_plan(cfg, args)
    print(f"[doctor] provider={plan.provider}")
    print(f"[doctor] router=http://127.0.0.1:{plan.router_port}")
    print(f"[doctor] upstream={plan.upstream_base_url}")
    print(f"[doctor] shim_enabled={plan.shim_enabled} shim=http://127.0.0.1:{plan.shim_port}")
    print(f"[doctor] ready_timeout_s={plan.ready_timeout_s:.1f}")
    if plan.provider == "llama_cpp":
        print(f"[doctor] llama_cpp.exe_path={_get_path(cfg, 'backend.llama_cpp.exe_path', '')}")
        print(f"[doctor] llama_cpp.models_dir={_get_path(cfg, 'backend.llama_cpp.models_dir', '')}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Portable launcher for llama-conductor stack")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="Launch backend/router/shim stack")
    up.add_argument("--router-dir", default=str(repo_root), help="Workspace root containing llama_conductor_cli.py")
    up.add_argument("--config", default=str(repo_root / "llama_conductor" / "router_config.yaml"))
    up.add_argument("--python-exe", default=sys.executable, help="Python executable to launch child processes")
    up.add_argument("--router-host", default=None)
    up.add_argument("--router-port", type=int, default=None)
    up.add_argument("--llama-host", default=None)
    up.add_argument("--llama-port", type=int, default=None)
    up.add_argument("--llama-exe", default=None)
    up.add_argument("--models-dir", default=None)
    up.add_argument("--models-preset", default=None)
    up.add_argument("--llama-ctx", type=int, default=None)
    up.add_argument("--threads", type=int, default=None)
    up.add_argument("--threads-batch", type=int, default=None)
    up.add_argument("--keep", type=int, default=None)
    up.add_argument("--batch-size", type=int, default=None)
    up.add_argument("--ubatch-size", type=int, default=None)
    up.add_argument("--shim-host", default=None)
    up.add_argument("--shim-port", type=int, default=None)
    up.add_argument("--no-shim", action="store_true", help="Disable shim launch even if config enables it")
    up.add_argument("--ready-timeout", type=float, default=None)
    up.add_argument("--dry-run", action="store_true")

    doctor = sub.add_parser("doctor", help="Print computed launch configuration")
    doctor.add_argument("--config", default=str(repo_root / "llama_conductor" / "router_config.yaml"))
    doctor.add_argument("--router-port", type=int, default=None)
    doctor.add_argument("--router-host", default="0.0.0.0")
    doctor.add_argument("--llama-port", type=int, default=None)
    doctor.add_argument("--shim-port", type=int, default=None)
    doctor.add_argument("--shim-host", default="0.0.0.0")
    doctor.add_argument("--no-shim", action="store_true")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "up":
        return _run_up(args)
    if args.cmd == "doctor":
        return _run_doctor(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
