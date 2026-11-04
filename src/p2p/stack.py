"""Boot / stop the whole microservice stack locally:  python -m p2p.stack up|down|status

Starts 10 processes (3 MCP servers, 3 A2A agents, scoring, audit, case, gateway), waits for /health,
and records PIDs in $DATA_DIR/run/. Inherits the environment, so `LLM_PROVIDER=ollama python -m p2p.stack up` just works."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from p2p.common.config import get_settings
from p2p.serve import SERVICES

ORDER = ["audit", "mcp-erp", "mcp-graph", "mcp-knowledge", "scoring", "graph-analyst", "doc-auditor", "investigator", "case", "gateway"]
WAVES = [ORDER[:7], ORDER[7:8], ORDER[8:]]  # start order matters for warm-up: tools+specialists → investigator → gateway


def run_dir() -> Path:
    d = get_settings().DATA_DIR / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wait(names: list[str], timeout: float) -> list[str]:
    t0, pending = time.time(), set(names)
    while pending and time.time() - t0 < timeout:
        for name in list(pending):
            try:
                if httpx.get(f"http://127.0.0.1:{SERVICES[name][1]}/health", timeout=1.0).status_code == 200:
                    pending.discard(name)
            except httpx.HTTPError:
                pass
        time.sleep(0.4)
    return sorted(pending)


def up(timeout: float = 90.0) -> int:
    """Start in dependency waves (tools + specialists → investigator → case + gateway) so each service can warm its upstreams."""
    env = {**os.environ, "HOST": "127.0.0.1"}
    t0 = time.time()
    for wave in WAVES:
        for name in wave:
            log = open(run_dir() / f"{name}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "p2p.serve", name, "--host", "127.0.0.1"], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            (run_dir() / f"{name}.pid").write_text(str(p.pid))
        if pending := _wait(wave, timeout):
            print("FAILED to become healthy:", pending, "— see", run_dir())
            return 1
    print(f"stack healthy in {time.time() - t0:.1f}s  (gateway :8100, console :8102, logs in {run_dir()})")
    return 0


def down() -> None:
    for pidf in run_dir().glob("*.pid"):
        try:
            os.killpg(int(pidf.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pidf.unlink(missing_ok=True)
    print("stack stopped")


def status() -> None:
    for name in ORDER:
        try:
            ok = httpx.get(f"http://127.0.0.1:{SERVICES[name][1]}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            ok = False
        print(f"{name:14} {'UP' if ok else 'down'}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    {"up": lambda: sys.exit(up()), "down": down, "status": status}[cmd]()


if __name__ == "__main__":
    main()
