#!/usr/bin/env python3
"""
llama-conductor CLI
Launcher for package structure
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _local_port_is_listening(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.35)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _existing_is_this_router(port: int) -> bool:
    try:
        import httpx

        with httpx.Client(timeout=0.6, follow_redirects=False) as client:
            resp = client.get(f"http://127.0.0.1:{int(port)}/healthz")
            if resp.status_code != 200:
                return False
            payload = (
                resp.json()
                if "application/json" in (resp.headers.get("content-type") or "").lower()
                else {}
            )
            return bool(
                isinstance(payload, dict)
                and payload.get("ok") is True
                and "version" in payload
            )
    except Exception:
        return False


def _find_listener_pids(port: int) -> List[int]:
    found: List[int] = []
    port_i = int(port)

    try:
        import psutil  # type: ignore

        pids = set()
        for conn in psutil.net_connections(kind="inet"):
            laddr = getattr(conn, "laddr", None)
            cpid = getattr(conn, "pid", None)
            status = str(getattr(conn, "status", "")).upper()
            if not laddr or not cpid:
                continue
            if int(getattr(laddr, "port", -1)) != port_i:
                continue
            if status != "LISTEN":
                continue
            pids.add(int(cpid))
        found = sorted(pids)
        if found:
            return found
    except Exception:
        pass

    if os.name == "nt":
        try:
            out = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            pids = set()
            for line in out.splitlines():
                if f":{port_i}" not in line:
                    continue
                if "LISTENING" not in line.upper():
                    continue
                cols = line.split()
                if not cols:
                    continue
                pid_raw = cols[-1].strip()
                if pid_raw.isdigit():
                    pids.add(int(pid_raw))
            found = sorted(pids)
        except Exception:
            found = []
    else:
        try:
            out = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port_i}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            pids = set()
            for line in out.splitlines():
                raw = line.strip()
                if raw.isdigit():
                    pids.add(int(raw))
            found = sorted(pids)
            if found:
                return found
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["ss", "-ltnp", f"sport = :{port_i}"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            pids = set()
            for match in re.finditer(r"pid=(\d+)", out or ""):
                pids.add(int(match.group(1)))
            found = sorted(pids)
        except Exception:
            found = []
    return found


def _terminate_pid(pid: int) -> bool:
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.returncode == 0
        except Exception:
            return False
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except Exception:
        return False


def _take_over_existing_instance(port: int, timeout_s: float) -> bool:
    pids = _find_listener_pids(port)
    if not pids:
        return not _local_port_is_listening(port)

    for pid in pids:
        _terminate_pid(pid)

    deadline = time.monotonic() + max(0.5, float(timeout_s or 5.0))
    while time.monotonic() < deadline:
        if not _local_port_is_listening(port):
            return True
        time.sleep(0.1)
    return not _local_port_is_listening(port)


def _format_pid_list(port: int) -> str:
    pids = _find_listener_pids(port)
    if not pids:
        return "(unknown)"
    return ", ".join(str(pid) for pid in pids)


def _print_port_blocked_help(port: int) -> None:
    pids = _format_pid_list(port)
    print(f"[llama-conductor] Port {port} is busy (PID(s): {pids}).")
    if os.name == "nt":
        print(f"[llama-conductor] Fix: close stale windows, or run `netstat -ano | findstr :{port}` then `taskkill /PID <pid> /T /F`.")
    else:
        print(f"[llama-conductor] Fix: run `lsof -nP -iTCP:{port} -sTCP:LISTEN` then kill stale PID(s).")


def serve(host="0.0.0.0", port=9000):
    """Launch the router"""
    single_instance = _env_bool("MOA_ROUTER_SINGLE_INSTANCE", True)
    takeover_existing = _env_bool("MOA_ROUTER_TAKEOVER_EXISTING", True)
    takeover_foreign = _env_bool("MOA_ROUTER_TAKEOVER_FOREIGN", True)
    takeover_timeout_s = _env_float("MOA_ROUTER_TAKEOVER_TIMEOUT_S", 5.0)

    try:
        if single_instance and _local_port_is_listening(port):
            if _existing_is_this_router(port):
                if takeover_existing:
                    print(f"[llama-conductor] Existing router found on 127.0.0.1:{port}; taking over...")
                    if not _take_over_existing_instance(port, takeover_timeout_s):
                        print(f"[llama-conductor] ERROR: takeover failed; port {port} still busy.")
                        return False
                    print("[llama-conductor] Takeover complete; starting fresh router instance.")
                else:
                    print(f"[llama-conductor] Router already running on 127.0.0.1:{port}; exiting duplicate launch.")
                    return True
            elif _local_port_is_listening(port):
                if takeover_existing and takeover_foreign:
                    print(f"[llama-conductor] Port {port} is in use by a non-responsive process; attempting forced takeover...")
                    if not _take_over_existing_instance(port, takeover_timeout_s):
                        print(f"[llama-conductor] ERROR: forced takeover failed; port {port} still busy.")
                        _print_port_blocked_help(port)
                        return False
                    print("[llama-conductor] Forced takeover complete; starting fresh router instance.")
                else:
                    print(f"[llama-conductor] ERROR: port {port} already in use by another process.")
                    _print_port_blocked_help(port)
                    return False

        import uvicorn
        print(f"[llama-conductor] Starting router on {host}:{port}")
        uvicorn.run(
            "llama_conductor.router_fastapi:app",  # NOTE: package.module format
            host=host,
            port=port,
            reload=False,
        )
    except KeyboardInterrupt:
        print("\n[llama-conductor] Shutting down...")
    except Exception as e:
        print(f"[llama-conductor] ERROR: {e}")
        if "10048" in str(e) or "Address already in use" in str(e):
            _print_port_blocked_help(port)
        import traceback
        traceback.print_exc()
        return False
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="llama-conductor: LLM harness",
        epilog="Example: llama-conductor serve --port 9000"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    serve_parser = subparsers.add_parser("serve", help="Start the router")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=9000, help="Port (default: 9000)")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        success = serve(host=args.host, port=args.port)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
