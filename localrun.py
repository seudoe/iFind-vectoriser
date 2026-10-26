"""
Local-run helpers (same file in vectoriser and internScraper): load the env files, pick a free port, say clearly what is running.

    load_env(Path(__file__).parent)                       # at the top of the entrypoint, before anything reads os.environ
    port = pick_port("internScraper", 5000, "ifind and vectoriser (INTERNSCRAPER_URL)")
    banner("internScraper", port, {"Talks to vectoriser": os.environ.get("VECTORISER_URL")})

Fixed local ports (deliberately off the usual 3000 / 8000 / 8080): ifind 4000, internScraper 5000, vectoriser 6000.
Only used by `python app.py`; the Docker images start uvicorn themselves on $PORT and never touch this.
"""

import os
import socket
import sys
from pathlib import Path
from typing import Dict, Optional

# the banner goes to the real stdout, not through the print-to-logger capture some services install
OUT = sys.__stdout__ or sys.stdout
RED = "\x1b[1;97;41m"   # bold white on red
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[1;32m"
OFF = "\x1b[0m"

if os.name == "nt":
    os.system("")   # switches the Windows console to ANSI escape handling


def load_env(base: Path) -> None:
    """`.env` first (production values; variables already set in the process win), then `.env.local` over it for local runs.
    `.env.local` is gitignored and `.dockerignore`d, so a deployed image never sees it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(base / ".env")
    local = base / ".env.local"
    if local.exists():
        load_dotenv(local, override=True)


def port_in_use(port: int) -> bool:
    """True when something already listens on `port` (checked on localhost and on all interfaces)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def pick_port(name: str, preferred: int, dependents: str = "the other components") -> int:
    """`preferred` when it is free; otherwise the next free port, with a big red warning on the console."""
    port = preferred
    while port_in_use(port):
        port += 1
    if port != preferred:
        width = 78
        lines = [
            "",
            f"  WARNING: PORT {preferred} IS ALREADY IN USE",
            f"  {name} will start on port {port} instead.",
            f"  You may need to reconfigure {dependents}",
            f"  so they point to http://localhost:{port}",
            "",
        ]
        print("", file=sys.stderr)
        for line in lines:
            print(f"{RED}{line.ljust(width)}{OFF}", file=sys.stderr)
        print("", file=sys.stderr, flush=True)
    return port


def banner(name: str, port: int, extra: Optional[Dict[str, Optional[str]]] = None) -> None:
    """What is about to run and where (printed before uvicorn starts, so it is the first thing you see)."""
    base = f"http://localhost:{port}"
    rows = [("API", base), ("Docs", f"{base}/docs"), ("Health", f"{base}/health")]
    rows += [(k, v or "(not set)") for k, v in (extra or {}).items()]
    try:
        "─".encode(getattr(OUT, "encoding", None) or "ascii")
        rule = "─"
    except (UnicodeEncodeError, LookupError):
        rule = "-"   # redirected output on a legacy Windows code page
    bar = rule * 66
    lines = [f"\n{DIM}{bar}{OFF}", f"  {BOLD}{name}{OFF}  starting on {GREEN}port {port}{OFF}"]
    lines += [f"  {key.ljust(24)} {value}" for key, value in rows]
    lines.append(f"{DIM}{bar}{OFF}\n")
    OUT.write("\n".join(lines) + "\n")   # not print(): services that capture print() would wrap it in log lines
    OUT.flush()
