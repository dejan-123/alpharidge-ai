#!/usr/bin/env python3
"""
SN45 Miner Fleet Manager

Reads miners.yaml, manages all miner processes in a tmux session, and shows a
live status dashboard.

Commands
  start  [--name NAME]          start all miners (or just NAME)
  stop   [--name NAME]          stop all miners (or just NAME)
  restart [--name NAME]         stop then start
  status [--watch N] [--chain]  show status table; --watch N refreshes every N s;
                                --chain queries the subtensor for on-chain registration
  logs   --name NAME            attach tmux to that miner's window
  attach                        attach to the full miner tmux session

Config  : miners.yaml  (copy from miners.yaml.example, git-ignored)
Logs    : logs/miners/<name>.log
Session : tmux session  sn45-miners,  one window per miner
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed.  Run: pip install pyyaml")
    sys.exit(1)

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: rich not installed.  Run: pip install rich")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "miners.yaml"
LOG_DIR = REPO_ROOT / "logs" / "miners"
TMUX_SESSION = "sn45-miners"

console = Console()


# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        console.print(f"[red]ERROR[/red]: {CONFIG_FILE} not found.")
        console.print(f"Copy [bold]miners.yaml.example[/bold] → [bold]miners.yaml[/bold] and fill in your values.")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def resolve_miners(cfg: Dict, name_filter: Optional[str] = None) -> List[Dict]:
    g = cfg.get("global", {})
    results = []
    for m in cfg.get("miners", []):
        merged = {**g, **m}
        if name_filter and merged.get("name") != name_filter:
            continue
        results.append(merged)
    if name_filter and not results:
        console.print(f"[red]ERROR[/red]: no miner named '{name_filter}' in {CONFIG_FILE}")
        sys.exit(1)
    return results


def detect_python(cfg: Dict) -> str:
    explicit = cfg.get("global", {}).get("python", "")
    if explicit:
        return explicit
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    conda_env = cfg.get("global", {}).get("conda_env", "alpharidge_ai")
    try:
        r = subprocess.run(
            ["conda", "run", "-n", conda_env, "which", "python"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "python3"


# ── tmux helpers ──────────────────────────────────────────────────────────────

def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux"] + list(args), capture_output=True, text=True)


def session_exists() -> bool:
    return _tmux("has-session", "-t", TMUX_SESSION).returncode == 0


def window_exists(name: str) -> bool:
    r = _tmux("list-windows", "-t", TMUX_SESSION, "-F", "#{window_name}")
    return name in r.stdout.splitlines()


def ensure_session() -> None:
    if not session_exists():
        _tmux("new-session", "-d", "-s", TMUX_SESSION, "-n", "_manager")


# ── Build the shell command for one miner ────────────────────────────────────

def build_launch_cmd(m: Dict, python: str) -> str:
    name = m["name"]
    log_file = LOG_DIR / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    netuid  = m.get("netuid", 45)
    network = m.get("subtensor_network", "finney")
    wallet  = m.get("wallet_name", "default")
    hotkey  = m["hotkey"]
    port    = m["axon_port"]
    api_key = m["api_key"]
    nlp_url = m.get("nlp_server_url", "")
    ext_ip  = m.get("axon_external_ip", "")
    auto_restart  = m.get("auto_restart", True)
    restart_delay = m.get("restart_delay", 30)

    env = (
        f"API_KEY={api_key}"
        f" CUBLAS_WORKSPACE_CONFIG=:4096:8"
        f" BT_NO_PARSE_CLI_ARGS=0"
    )
    if nlp_url:
        env += f" NLP_SERVER_URL={nlp_url}"

    miner_args = (
        f"--netuid {netuid}"
        f" --subtensor.network {network}"
        f" --wallet.name {wallet} --wallet.hotkey {hotkey}"
        f" --axon.port {port}"
    )
    if ext_ip:
        ext_port = m.get("axon_external_port", port)
        miner_args += f" --axon.external_ip {ext_ip} --axon.external_port {ext_port}"
    miner_args += " --logging.info"

    run_line = f"{env} {python} -m neurons.miner {miner_args} 2>&1 | tee -a {log_file}"

    if auto_restart:
        return (
            f"while true; do\n"
            f"  echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [FLEET] {name}: starting\" | tee -a {log_file}\n"
            f"  {run_line}\n"
            f"  echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [FLEET] {name}: exited — restarting in {restart_delay}s\" | tee -a {log_file}\n"
            f"  sleep {restart_delay}\n"
            f"done"
        )
    return run_line


# ── Start / Stop ──────────────────────────────────────────────────────────────

def start_miner(m: Dict, python: str, force: bool = False) -> None:
    name = m["name"]
    ensure_session()
    if window_exists(name):
        if force:
            _stop_window(name)
            time.sleep(1)
        else:
            console.print(f"  [yellow]SKIP[/yellow]    {name}  (already running — use restart)")
            return
    cmd = build_launch_cmd(m, python)
    _tmux("new-window", "-t", TMUX_SESSION, "-n", name, "-d")
    _tmux("send-keys", "-t", f"{TMUX_SESSION}:{name}",
          f"cd {REPO_ROOT} && {cmd}", "Enter")
    console.print(f"  [green]STARTED[/green]  {name}  port={m['axon_port']}  hotkey={m['hotkey'][:12]}…")


def _stop_window(name: str) -> None:
    _tmux("kill-window", "-t", f"{TMUX_SESSION}:{name}")


def stop_miner(m: Dict) -> None:
    name = m["name"]
    if window_exists(name):
        _stop_window(name)
        console.print(f"  [red]STOPPED[/red]  {name}")
    else:
        console.print(f"  [dim]SKIP[/dim]     {name}  (not running)")


# ── Log parsing ───────────────────────────────────────────────────────────────

def log_tail(name: str, n: int = 80) -> List[str]:
    log_file = LOG_DIR / f"{name}.log"
    if not log_file.exists():
        return []
    try:
        r = subprocess.run(["tail", "-n", str(n), str(log_file)],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.splitlines()
    except Exception:
        return []


_TS_RE = re.compile(r"\d{2}:\d{2}:\d{2}")
_ERR_RE = re.compile(r"\bERROR\b|\bCRITICAL\b|Traceback|Exception", re.I)
_ACT_RE = re.compile(r"request|synapse|forward|process|article|tweet|telegram", re.I)


def parse_log(lines: List[str]) -> Dict[str, Any]:
    last_ts = "—"
    errors = 0
    last_activity = "—"
    for line in reversed(lines):
        if last_ts == "—":
            m = _TS_RE.search(line)
            if m:
                last_ts = m.group()
        if _ERR_RE.search(line):
            errors += 1
        if last_activity == "—" and _ACT_RE.search(line):
            last_activity = line.strip()[-55:]
    return {"last_ts": last_ts, "errors": errors, "last_activity": last_activity}


# ── Port check ────────────────────────────────────────────────────────────────

def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ── Chain data (batched, cached) ─────────────────────────────────────────────

@dataclass
class MinerChainData:
    uid:        Optional[int]   = None
    incentive:  Optional[float] = None
    stake:      Optional[float] = None
    emission:   Optional[float] = None   # TAO per block (raw)

_chain_cache: Dict[str, MinerChainData] = {}
_chain_cache_ts: float = 0.0
_CHAIN_TTL = 120.0   # refresh chain data at most every 2 minutes


def fetch_chain_data(miners: List[Dict]) -> Dict[str, MinerChainData]:
    """Fetch metagraph once and return per-miner chain stats.

    Bittensor prints to stdout on init; we redirect that to /dev/null so it
    does not corrupt Rich's Live alternate-screen buffer.
    """
    global _chain_cache, _chain_cache_ts
    if time.time() - _chain_cache_ts < _CHAIN_TTL:
        return _chain_cache

    try:
        import bittensor as bt
        import io

        # Resolve hotkey names → SS58 addresses (reads local key files, fast)
        ss58_map: Dict[str, str] = {}   # name → ss58
        for m in miners:
            try:
                w = bt.Wallet(name=m.get("wallet_name", "default"), hotkey=m["hotkey"])
                ss58_map[m["name"]] = w.hotkey.ss58_address
            except Exception:
                pass

        # Suppress bittensor's verbose stdout/stderr during Subtensor init
        devnull = open(os.devnull, "w")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            first = miners[0]
            sub  = bt.Subtensor(network=first.get("subtensor_network", "finney"))
            meta = sub.metagraph(first.get("netuid", 45))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()

        # Build ss58 → uid index from metagraph
        hk_to_uid = {hk: uid for uid, hk in enumerate(meta.hotkeys)}

        result: Dict[str, MinerChainData] = {}
        for m in miners:
            name = m["name"]
            ss58 = ss58_map.get(name)
            if ss58 is None or ss58 not in hk_to_uid:
                result[name] = MinerChainData()
                continue
            uid = hk_to_uid[ss58]
            result[name] = MinerChainData(
                uid       = uid,
                incentive = float(meta.incentive[uid]),
                stake     = float(meta.S[uid]),
                emission  = float(meta.emission[uid]),
            )

        _chain_cache    = result
        _chain_cache_ts = time.time()
        return result

    except Exception:
        return {m["name"]: MinerChainData() for m in miners}


# ── Status table ──────────────────────────────────────────────────────────────

def build_table(miners: List[Dict], chain: Optional[Dict[str, MinerChainData]] = None) -> Table:
    sess_ok = session_exists()
    ts = time.strftime("%H:%M:%S")
    show_chain = chain is not None

    table = Table(
        box=box.ROUNDED,
        title=f"[bold cyan]SN45 Miner Fleet[/bold cyan]  [dim]{ts}[/dim]",
        expand=True,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Name",       style="bold",    min_width=10)
    table.add_column("Process",    justify="center", min_width=8)
    table.add_column("Axon",       justify="center", min_width=6)
    table.add_column("Port",       justify="right",  min_width=5)
    table.add_column("Hotkey",     min_width=12)
    table.add_column("Last seen",  justify="center", min_width=8)
    table.add_column("Errors",     justify="right",  min_width=6)
    table.add_column("Last activity", min_width=28)
    if show_chain:
        table.add_column("UID",       justify="right", min_width=5)
        table.add_column("Incentive", justify="right", min_width=10)
        table.add_column("Stake τ",   justify="right", min_width=9)
        table.add_column("Emission",  justify="right", min_width=10)

    for m in miners:
        name    = m["name"]
        port    = int(m["axon_port"])
        hotkey  = m["hotkey"]
        hk_str  = hotkey[:8] + "…" + hotkey[-4:] if len(hotkey) > 12 else hotkey

        running = sess_ok and window_exists(name)
        axon_up = port_open(port) if running else False

        lines = log_tail(name)
        stats = parse_log(lines)

        proc_cell = (
            Text("● RUN",  style="green bold") if running
            else Text("✗ DOWN", style="red bold")
        )
        axon_cell = (
            Text("✓ open", style="green") if axon_up
            else (Text("… wait", style="yellow") if running else Text("—", style="dim"))
        )
        err_cell = (
            Text(str(stats["errors"]), style="red bold") if stats["errors"] > 0
            else Text("0", style="dim")
        )

        row: List[Any] = [
            name, proc_cell, axon_cell, str(port), hk_str,
            stats["last_ts"], err_cell, stats["last_activity"],
        ]

        if show_chain:
            cd = chain.get(name, MinerChainData())
            if cd.uid is not None:
                uid_cell  = Text(str(cd.uid), style="green")
                inc_cell  = Text(f"{cd.incentive:.6f}", style="cyan")
                stk_cell  = Text(f"{cd.stake:.4f}",    style="yellow")
                emi_cell  = Text(f"{cd.emission:.6f}", style="dim cyan")
            else:
                uid_cell  = Text("unreg", style="red")
                inc_cell  = Text("—", style="dim")
                stk_cell  = Text("—", style="dim")
                emi_cell  = Text("—", style="dim")
            row.extend([uid_cell, inc_cell, stk_cell, emi_cell])

        table.add_row(*row)

    if show_chain and chain:
        reg = [chain.get(m["name"], MinerChainData()) for m in miners
               if chain.get(m["name"], MinerChainData()).uid is not None]
        n_reg = len(reg)
        n_tot = len(miners)
        if n_reg:
            tot_inc = sum(cd.incentive for cd in reg)
            tot_stk = sum(cd.stake     for cd in reg)
            tot_emi = sum(cd.emission  for cd in reg)
            table.add_section()
            summary_row: List[Any] = [
                Text(f"TOTAL ({n_reg}/{n_tot})", style="bold"),
                "", "", "", "", "", "", "",
                Text("", style="dim"),
                Text(f"{tot_inc:.6f}", style="bold cyan"),
                Text(f"{tot_stk:.4f} τ", style="bold yellow"),
                Text(f"{tot_emi:.6f}", style="bold dim cyan"),
            ]
            table.add_row(*summary_row)

    chain_age = ""
    if show_chain and _chain_cache_ts > 0:
        age = int(time.time() - _chain_cache_ts)
        chain_age = f"  | chain data: [dim]{age}s ago[/dim]"
    table.caption = (
        f"[dim]tmux: [bold]{TMUX_SESSION}[/bold]"
        f"  | attach: [bold]tmux attach -t {TMUX_SESSION}[/bold]"
        f"  | logs: [bold]mine.py logs --name NAME[/bold]"
        f"{chain_age}[/dim]"
    )
    return table


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_start(args: argparse.Namespace, cfg: Dict) -> None:
    python = detect_python(cfg)
    miners = resolve_miners(cfg, getattr(args, "name", None))
    console.print(f"\n[bold]Starting {len(miners)} miner(s) in tmux session '[cyan]{TMUX_SESSION}[/cyan]'…[/bold]\n")
    for m in miners:
        start_miner(m, python)
    console.print(f"\n[dim]Attach: tmux attach -t {TMUX_SESSION}[/dim]")
    console.print(f"[dim]Status: python scripts/mine.py status --watch 5[/dim]\n")


def cmd_stop(args: argparse.Namespace, cfg: Dict) -> None:
    miners = resolve_miners(cfg, getattr(args, "name", None))
    console.print(f"\n[bold]Stopping {len(miners)} miner(s)…[/bold]\n")
    for m in miners:
        stop_miner(m)
    console.print()


def cmd_restart(args: argparse.Namespace, cfg: Dict) -> None:
    python = detect_python(cfg)
    miners = resolve_miners(cfg, getattr(args, "name", None))
    console.print(f"\n[bold]Restarting {len(miners)} miner(s)…[/bold]\n")
    for m in miners:
        stop_miner(m)
    console.print()
    time.sleep(2)
    for m in miners:
        start_miner(m, python)
    console.print()


def cmd_status(args: argparse.Namespace, cfg: Dict) -> None:
    miners = resolve_miners(cfg)
    check_chain = getattr(args, "chain", False)
    watch = getattr(args, "watch", None)

    def _chain() -> Optional[Dict[str, MinerChainData]]:
        """Fetch chain data outside the Live render loop (stdout suppressed inside)."""
        return fetch_chain_data(miners) if check_chain else None

    if watch:
        chain = _chain()
        with Live(build_table(miners, chain), console=console,
                  refresh_per_second=1, screen=True) as live:
            last_chain_refresh = time.time()
            while True:
                time.sleep(min(watch, 5))
                # Refresh chain data in background on its own interval
                if check_chain and time.time() - last_chain_refresh >= _CHAIN_TTL:
                    chain = _chain()
                    last_chain_refresh = time.time()
                live.update(build_table(miners, chain))
    else:
        console.print(build_table(miners, _chain()))


def cmd_logs(args: argparse.Namespace, cfg: Dict) -> None:
    miners = resolve_miners(cfg, args.name)
    m = miners[0]
    name = m["name"]
    if session_exists() and window_exists(name):
        os.execvp("tmux", ["tmux", "attach", "-t", f"{TMUX_SESSION}:{name}"])
    else:
        log_file = LOG_DIR / f"{name}.log"
        if log_file.exists():
            console.print(f"[yellow]Miner '{name}' is not running.[/yellow]  Showing log tail:\n")
            os.execvp("tail", ["tail", "-f", str(log_file)])
        else:
            console.print(f"[red]Miner '{name}' is not running and has no log file.[/red]")
            sys.exit(1)


def cmd_attach(_args: argparse.Namespace, _cfg: Dict) -> None:
    if not session_exists():
        console.print(f"[red]Session '{TMUX_SESSION}' not found. Run 'start' first.[/red]")
        sys.exit(1)
    os.execvp("tmux", ["tmux", "attach", "-t", TMUX_SESSION])


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="mine.py",
        description="SN45 miner fleet manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/mine.py start
  python scripts/mine.py start --name miner-01
  python scripts/mine.py stop
  python scripts/mine.py restart --name miner-02
  python scripts/mine.py status
  python scripts/mine.py status --watch 5
  python scripts/mine.py status --watch 5 --chain
  python scripts/mine.py logs --name miner-01
  python scripts/mine.py attach
        """,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _name(sp):
        sp.add_argument("--name", metavar="NAME", help="Specific miner name (default: all)")

    s = sub.add_parser("start",   help="Start all miners (or --name one)")
    _name(s)

    s = sub.add_parser("stop",    help="Stop all miners (or --name one)")
    _name(s)

    s = sub.add_parser("restart", help="Restart all miners (or --name one)")
    _name(s)

    s = sub.add_parser("status",  help="Show live status table")
    s.add_argument("--watch", metavar="SECS", type=int,
                   help="Auto-refresh every N seconds (fullscreen)")
    s.add_argument("--chain", action="store_true",
                   help="Query subtensor for on-chain UID (slow ~10s)")

    s = sub.add_parser("logs",    help="Attach to miner window (or tail log file)")
    s.add_argument("--name", required=True, metavar="NAME")

    sub.add_parser("attach", help="Attach to the full miner tmux session")

    args = p.parse_args()
    cfg  = load_config()

    {
        "start":   cmd_start,
        "stop":    cmd_stop,
        "restart": cmd_restart,
        "status":  cmd_status,
        "logs":    cmd_logs,
        "attach":  cmd_attach,
    }[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
