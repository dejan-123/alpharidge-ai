#!/usr/bin/env python3
"""
Unstake all alpha stakes for every miner defined in miners.yaml.

Usage:
    python scripts/unstake_all.py
    python scripts/unstake_all.py --netuid 45
    python scripts/unstake_all.py --hotkeys miner11 miner12   # specific hotkeys only
    python scripts/unstake_all.py --dry-run                   # preview without spending
"""
import argparse
import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml not installed — run: pip install pyyaml")

try:
    import bittensor as bt
except ImportError:
    raise SystemExit("bittensor not installed — activate the venv first")

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    raise SystemExit("rich not installed — run: pip install rich")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "miners.yaml"
NETUID_DEFAULT = 45
NETWORK_DEFAULT = "finney"

console = Console()


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> Dict:
    if not CONFIG_FILE.exists():
        raise SystemExit(f"miners.yaml not found at {CONFIG_FILE}")
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def get_miners(cfg: Dict, hotkey_filter: Optional[List[str]] = None) -> List[Dict]:
    g = cfg.get("global", {})
    miners = []
    for m in cfg.get("miners", []):
        merged = {**g, **m}
        if hotkey_filter and merged.get("hotkey") not in hotkey_filter:
            continue
        miners.append(merged)
    return miners


# ── Unstake ───────────────────────────────────────────────────────────────────

def _parse_received_tao(output: str) -> float:
    """Extract the TAO amount received from btcli stake remove output."""
    import re
    # Pattern 1: 'receiving   0.001234 τ'  (btcli v10 table row)
    m = re.search(r'receiv\w*\s+([\d.]+)\s*τ', output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Pattern 2: '→ 0.001234 τ' or '➡ 0.001234 τ'  (balance change arrows)
    m = re.search(r'[→➡]\s*([\d.]+)\s*τ', output)
    if m:
        return float(m.group(1))
    # Pattern 3: '+0.001234 τ' (positive balance delta)
    m = re.search(r'\+([\d.]+)\s*τ', output)
    if m:
        return float(m.group(1))
    # Pattern 4: 'amount   0.001234 τ'
    m = re.search(r'amount\s+([\d.]+)\s*τ', output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def unstake_one(
    sub: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    dry_run: bool,
    debug: bool = False,
) -> Dict:
    """
    Unstake all alpha for one hotkey via the Bittensor SDK.
    Returns a result dict with keys: hotkey, wallet, status, received_tao, error.
    """
    hotkey = wallet.hotkey.ss58_address
    result = {"hotkey": hotkey, "wallet": wallet.name, "status": "skipped",
              "received_tao": 0.0, "error": ""}

    try:
        pre_stake = sub.get_stake_for_hotkey(hotkey, netuid)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Could not read stake: {e}"
        return result

    stake_tao = pre_stake.tao if pre_stake is not None else 0.0
    if dry_run:
        result["status"] = "dry-run"
        result["received_tao"] = stake_tao
        return result

    if stake_tao < 1e-9:
        result["status"] = "no_stake"
        return result

    try:
        response = sub.unstake_all(
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey,
            rate_tolerance=0.05,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            raise_error=False,
        )
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    if debug:
        console.print(f"\n[dim]--- ExtrinsicResponse for {hotkey} ---[/dim]")
        console.print(f"[dim]{response}[/dim]")
        console.print(f"[dim]--- end ---[/dim]\n")

    if not response.success:
        err = response.error
        msg = err.message if hasattr(err, "message") else str(err)
        result["status"] = "error"
        result["error"] = str(msg) if msg else getattr(response, "message", "Unstake failed")
        return result

    try:
        post_stake = sub.get_stake_for_hotkey(hotkey, netuid)
        post_tao = post_stake.tao if post_stake is not None else 0.0
    except Exception:
        post_tao = 0.0

    received = max(0.0, stake_tao - post_tao)
    result["status"] = "ok"
    result["received_tao"] = received
    return result


# ── Display ───────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]) -> None:
    t = Table(title="Unstake Summary", box=None, show_lines=True)
    t.add_column("Hotkey",       style="cyan")
    t.add_column("Wallet",       style="dim")
    t.add_column("Status")
    t.add_column("Received (τ)", justify="right")
    t.add_column("Note",         style="dim")

    total = 0.0
    for r in results:
        status = r["status"]
        if status == "ok":
            status_col = "[green]✓ unstaked[/green]"
        elif status == "no_stake":
            status_col = "[dim]— no stake[/dim]"
        elif status == "dry-run":
            status_col = "[yellow]DRY RUN[/yellow]"
        elif status == "skipped":
            status_col = "[dim]skipped[/dim]"
        else:
            status_col = f"[red]✗ {status}[/red]"

        received = f"{r['received_tao']:.4f}" if r["received_tao"] else "—"
        total += r["received_tao"]
        t.add_row(r["hotkey"], r["wallet"], status_col, received, r["error"][:80])

    console.print()
    console.print(t)
    if total:
        console.print(f"\n[bold green]Total TAO returned to coldkey: {total:.4f} τ[/bold green]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Unstake all alpha stakes for miners in miners.yaml")
    ap.add_argument("--netuid",   type=int, default=NETUID_DEFAULT,
                    help=f"Subnet UID (default: {NETUID_DEFAULT})")
    ap.add_argument("--network",  default=NETWORK_DEFAULT,
                    help=f"Subtensor network (default: {NETWORK_DEFAULT})")
    ap.add_argument("--hotkeys",  nargs="+", default=None, metavar="HOTKEY",
                    help="Unstake only these hotkeys (default: all miners in miners.yaml)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Show what would be unstaked without submitting any transactions")
    ap.add_argument("--debug",    action="store_true",
                    help="Print raw btcli output for each hotkey (useful to fix TAO parsing)")
    ap.add_argument("--password", default=None, metavar="PASS",
                    help="Common coldkey password for all wallets (or set BT_WALLET_PASSWORD env var). "
                         "If omitted and env var not set, you will be prompted once for each distinct wallet.")
    args = ap.parse_args()

    console.print(f"\n[bold]Unstake All[/bold]  netuid=[cyan]{args.netuid}[/cyan]  "
                  f"network=[cyan]{args.network}[/cyan]")
    if args.dry_run:
        console.print("[yellow]DRY RUN — no transactions will be submitted[/yellow]")
    console.print()

    cfg     = load_config()
    miners  = get_miners(cfg, args.hotkeys)

    if not miners:
        raise SystemExit("No matching miners found in miners.yaml")

    console.print(f"Found [cyan]{len(miners)}[/cyan] miner(s) to process:\n")
    for m in miners:
        console.print(f"  [dim]{m.get('wallet_name','?')}[/dim] / [cyan]{m.get('hotkey','?')}[/cyan]")
    console.print()

    # ── Unlock coldkeys so btcli reads the password from env ──────────────────
    # One password per distinct coldkey wallet. Prompt once for each.
    distinct_wallets = sorted({m.get("wallet_name", "default") for m in miners})
    wallet_passwords: Dict[str, str] = {}
    env_pass = args.password or os.environ.get("BT_WALLET_PASSWORD")
    for wname in distinct_wallets:
        if env_pass:
            wallet_passwords[wname] = env_pass
        else:
            wallet_passwords[wname] = getpass(f"Password for coldkey wallet '{wname}': ")

    # Create one unlocked wallet per hotkey
    wallet_objs: Dict[str, bt.Wallet] = {}
    for m in miners:
        wname = m.get("wallet_name", "default")
        hkey  = m.get("hotkey", "")
        if hkey in wallet_objs:
            continue
        pw = wallet_passwords[wname]
        try:
            w  = bt.Wallet(name=wname, hotkey=hkey)
            w.coldkey_file.save_password_to_env(password=pw)
            w.unlock_coldkey()
            wallet_objs[hkey] = w
            console.print(f"  [green]✓[/green] unlocked {wname}/{hkey}")
        except Exception as e:
            console.print(f"  [red]✗ cannot unlock {wname}/{hkey}: {e}[/red]")
            sys.exit(1)

    # Wipe password from environment and local variable — no longer needed
    os.environ.pop("BT_WALLET_PASSWORD", None)
    wallet_passwords.clear()
    console.print()

    # ── Unstake each miner ────────────────────────────────────────────────────
    sub = bt.Subtensor(network=args.network)
    results: List[Dict] = []
    for m in miners:
        wallet_name = m.get("wallet_name", "default")
        hkey   = m.get("hotkey", "")
        wallet = wallet_objs[hkey]
        console.print(f"Unstaking [cyan]{wallet.hotkey_str}[/cyan] ({wallet_name}/{hkey}) …", end=" ")
        r = unstake_one(sub, wallet, args.netuid, args.dry_run, args.debug)
        if r["status"] == "ok":
            console.print(f"[green]✓[/green] received {r['received_tao']:.4f} τ")
        elif r["status"] in ("no_stake", "dry-run"):
            console.print(f"[dim]{r['status']}[/dim]")
        else:
            console.print(f"[red]✗ {r['status']}[/red]")
            if r["error"]:
                console.print(f"  [red]{r['error'][:200]}[/red]")
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
