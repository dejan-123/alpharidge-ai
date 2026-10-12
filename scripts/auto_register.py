#!/usr/bin/env python3
"""
SN45 auto-registration script.

Polls the on-chain burn cost every N seconds and registers hotkeys
automatically when the cost drops below your threshold.

Usage:
    python scripts/auto_register.py \\
        --wallet sn45mining \\
        --hotkeys miner13 miner14 miner15 \\
        --threshold 0.04          # max TAO you are willing to pay
        --interval 60             # seconds between polls (default 60)

    # Dry-run — show cost but never register:
    python scripts/auto_register.py --wallet sn45mining --hotkeys miner13 --dry-run
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import bittensor as bt
except ImportError:
    raise SystemExit("bittensor not installed — activate the venv first")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    raise SystemExit("rich not installed — run: pip install rich")

NETUID_DEFAULT = 45
NETWORK_DEFAULT = "finney"

PROTECTED_COLDKEYS = [
    "5EZW7aNRY4TsLGdyDsMozTiUBQ2zxNDJ1wMmgLfrtWBD92XU",
]

console = Console()


def get_burn_cost(sub: bt.Subtensor, netuid: int) -> float:
    """Return current burn/recycle cost in TAO as a float."""
    try:
        cost = sub.recycle(netuid=netuid)
        return float(cost)
    except Exception:
        hp = sub.get_subnet_hyperparameters(netuid=netuid)
        return float(hp.burn)


def is_registered(sub: bt.Subtensor, netuid: int, hotkey_ss58: str) -> bool:
    try:
        return sub.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58)
    except Exception:
        return False


def lowest_not_immune_miner_coldkey(sub: bt.Subtensor, netuid: int) -> Optional[str]:
    """Return the coldkey of the lowest-incentive non-immune miner, or None."""
    try:
        met = sub.metagraph(netuid=netuid)
        current_block = int(met.block)
        immunity_period = int(sub.immunity_period(netuid=netuid))
        not_immune = [
            uid
            for uid in range(met.num_uids)
            if int(met.block_at_registration[uid]) > 0
            and current_block - int(met.block_at_registration[uid]) >= immunity_period
        ]
        miners = [uid for uid in not_immune if float(met.validator_trust[uid]) < 0.1]
        if not miners:
            return None
        lowest = min(miners, key=lambda uid: float(met.incentive[uid]))
        return met.coldkeys[lowest]
    except Exception:
        return None


def load_wallet(wallet_name: str, hotkey_name: str) -> bt.Wallet:
    return bt.Wallet(name=wallet_name, hotkey=hotkey_name)


def build_table(
    cost: float,
    threshold: float,
    balances: Dict[str, float],
    states: Dict[str, str],
    reg_counts: Dict[str, int],
    interval: int,
    dry_run: bool,
) -> Table:
    now = datetime.now().strftime("%H:%M:%S")
    title = f"SN45 Auto-Register  {now}  [dim](monitoring forever)[/dim]"
    if dry_run:
        title += "  [yellow][DRY RUN][/yellow]"

    t = Table(title=title, border_style="dim")
    t.add_column("Hotkey",        style="bold cyan", no_wrap=True)
    t.add_column("Wallet",        style="dim", no_wrap=True)
    t.add_column("Status",        justify="center")
    t.add_column("Regs",          justify="center")
    t.add_column("Cost (τ)",      justify="right")
    t.add_column("Threshold (τ)", justify="right")
    t.add_column("Balance (τ)",   justify="right")
    t.add_column("Note")

    cost_str = f"{cost:.4f}"
    thr_str  = f"{threshold:.4f}"
    cost_col = f"[green]{cost_str}[/green]" if cost <= threshold else f"[red]{cost_str}[/red]"

    for hotkey, state in states.items():
        if hotkey.startswith("_wallet_"):
            continue
        bal = balances.get(hotkey, 0.0)
        bal_str = f"{bal:.4f}"
        bal_col = f"[green]{bal_str}[/green]" if bal >= cost else f"[red]{bal_str}[/red]"
        wallet_label = states.get(f"_wallet_{hotkey}", "")
        regs = str(reg_counts.get(hotkey, 0))
        if state == "registered":
            status = "[green]✓ on-chain[/green]"
            note   = "Registered"
        elif state == "just_registered":
            status = "[bold green]✓ REGISTERED[/bold green]"
            note   = "Just registered — monitoring"
        elif state == "failed":
            status = "[red]✗ failed[/red]"
            note   = "Registration failed — will retry next poll"
        elif state == "insufficient":
            status = "[yellow]⚡ waiting[/yellow]"
            note   = f"Balance {bal_str} τ < cost {cost_str} τ"
        elif state == "protected":
            status = "[yellow]🛡 protected[/yellow]"
            note   = "Would deregister protected coldkey — waiting"
        elif state == "above_threshold":
            status = "[yellow]⏳ waiting[/yellow]"
            note   = f"Cost {cost_str} τ > threshold {thr_str} τ  (poll every {interval}s)"
        elif state == "ready":
            status = "[cyan]→ registering[/cyan]"
            note   = "Will register this poll"
        else:
            status = f"[dim]{state}[/dim]"
            note   = ""

        t.add_row(hotkey, wallet_label, status, regs, cost_col, thr_str, bal_col, note)

    return t


def try_register(
    sub: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    dry_run: bool,
) -> bool:
    if dry_run:
        console.print(f"  [yellow][DRY RUN][/yellow] Would register {wallet.hotkey_str} — skipping.")
        return False
    console.print(f"  [bold]Registering[/bold] {wallet.hotkey_str} on netuid {netuid} …")
    try:
        success = sub.burned_register(wallet=wallet, netuid=netuid)
        return bool(success)
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "miners.yaml"


def _hotkey_to_yaml_info() -> Dict[str, Dict]:
    """Read miners.yaml and return {hotkey_name: {name, wallet_name}} for all miners."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        import yaml
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
        g = cfg.get("global", {})
        result = {}
        for m in cfg.get("miners", []):
            merged = {**g, **m}
            if "hotkey" in merged and "name" in merged:
                result[merged["hotkey"]] = {
                    "name": merged["name"],
                    "wallet_name": merged.get("wallet_name", g.get("wallet_name", "default")),
                }
        return result
    except Exception:
        return {}


def _hotkey_to_miner_map() -> Dict[str, str]:
    """Read miners.yaml and return {hotkey_name: miner_name} for all miners."""
    return {hk: info["name"] for hk, info in _hotkey_to_yaml_info().items()}


def _start_miner(hotkey: str, hotkey_to_miner: Dict[str, str], dry_run: bool) -> None:
    """Start or restart the miner that corresponds to this hotkey."""
    miner_name = hotkey_to_miner.get(hotkey, hotkey)
    mine_script = REPO_ROOT / "scripts" / "mine.py"

    if dry_run:
        console.print(f"  [yellow][DRY RUN][/yellow] Would run: mine.py restart --name {miner_name}")
        return

    console.print(f"  [bold]Starting/restarting[/bold] miner [cyan]{miner_name}[/cyan] …")
    result = subprocess.run(
        [sys.executable, str(mine_script), "restart", "--name", miner_name],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"  [dim]{line}[/dim]")
    if result.returncode != 0:
        console.print(f"  [yellow]⚠ mine.py restart exited {result.returncode}[/yellow]")
        if result.stderr:
            console.print(f"  [red]{result.stderr.strip()}[/red]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-register SN45 hotkeys when burn cost < threshold")
    ap.add_argument("--wallet",    default=None, metavar="NAME",
                    help="Fallback coldkey wallet name (overridden per miner by miners.yaml wallet_name)")
    ap.add_argument("--hotkeys",   nargs="+", metavar="HOTKEY", default=None,
                    help="One or more hotkey names to register")
    ap.add_argument("--all",        action="store_true",
                    help="Register all unregistered hotkeys found in miners.yaml")
    ap.add_argument("--threshold", type=float, required=True, metavar="TAO",
                    help="Max burn cost (TAO) — register only when cost ≤ this value")
    ap.add_argument("--netuid",    type=int, default=NETUID_DEFAULT,
                    help=f"Subnet UID (default: {NETUID_DEFAULT})")
    ap.add_argument("--network",   default=NETWORK_DEFAULT,
                    help=f"Subtensor network (default: {NETWORK_DEFAULT})")
    ap.add_argument("--protected-coldkeys", nargs="+", metavar="COLDKEY", default=PROTECTED_COLDKEYS,
                    help="Do not register if the lowest-incentive not-immune miner coldkey is in this list")
    ap.add_argument("--interval",  type=int, default=30, metavar="S",
                    help="Seconds between polls (default: 30)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Show cost/status but never actually register")
    ap.add_argument("--password",  default=None, metavar="PASS",
                    help="Coldkey password (or set BT_WALLET_PASSWORD env var). "
                         "If omitted and env var not set, you will be prompted once at startup.")
    ap.add_argument("--no-start",  action="store_true",
                    help="Skip starting/restarting the miner after successful registration")
    args = ap.parse_args()

    # Resolve hotkey list — either from --hotkeys or --all (reads miners.yaml)
    yaml_info = _hotkey_to_yaml_info()
    if args.all:
        if not yaml_info:
            raise SystemExit("--all: no miners found in miners.yaml")
        args.hotkeys = list(yaml_info.keys())
        console.print(f"[dim]--all: loaded {len(args.hotkeys)} hotkeys from miners.yaml[/dim]")
    elif not args.hotkeys:
        raise SystemExit("Provide --hotkeys or --all")

    # Build hotkey → miner-name and hotkey → wallet-name mappings
    hotkey_to_miner:  Dict[str, str] = {}
    hotkey_to_wallet: Dict[str, str] = {}
    for hk in args.hotkeys:
        info = yaml_info.get(hk, {})
        hotkey_to_miner[hk]  = info.get("name", hk)
        hotkey_to_wallet[hk] = info.get("wallet_name") or args.wallet or "default"

    console.print(f"\n[bold]SN45 Auto-Register[/bold]  "
                  f"netuid=[cyan]{args.netuid}[/cyan]  threshold=[cyan]{args.threshold} τ[/cyan]  "
                  f"network=[cyan]{args.network}[/cyan]")
    if args.dry_run:
        console.print("[yellow]DRY RUN — no transactions will be submitted[/yellow]")
    console.print()

    if not args.no_start:
        console.print("[dim]Miner/wallet mapping:[/dim]")
        for hk in args.hotkeys:
            console.print(f"  hotkey [cyan]{hk}[/cyan] → miner [cyan]{hotkey_to_miner[hk]}[/cyan]  wallet [cyan]{hotkey_to_wallet[hk]}[/cyan]")
        console.print()

    # ── Password handling ─────────────────────────────────────────────────────
    # One password per distinct coldkey wallet name. Prompt once for each.
    distinct_wallets = sorted(set(hotkey_to_wallet.values()))
    wallet_passwords: Dict[str, str] = {}
    env_pass = args.password or os.environ.get("BT_WALLET_PASSWORD")
    for wname in distinct_wallets:
        if env_pass:
            wallet_passwords[wname] = env_pass
        else:
            wallet_passwords[wname] = getpass(f"Password for coldkey wallet '{wname}': ")

    console.print("[dim]Connecting to subtensor …[/dim]")
    sub = bt.Subtensor(network=args.network)

    # Build wallet objects per hotkey using the correct coldkey per hotkey
    wallets: Dict[str, bt.Wallet] = {}
    for hk in args.hotkeys:
        wname = hotkey_to_wallet[hk]
        try:
            wallets[hk] = load_wallet(wname, hk)
        except Exception as e:
            console.print(f"[red]Cannot load wallet {wname}/{hk}: {e}[/red]")
            sys.exit(1)

    # Pre-unlock all coldkeys so the loop never blocks on input
    console.print("[dim]Unlocking coldkeys …[/dim]")
    for hk, w in wallets.items():
        wname = hotkey_to_wallet[hk]
        pw = wallet_passwords[wname]
        try:
            os.environ["BT_WALLET_PASSWORD"] = pw
            w.coldkey_file.save_password_to_env(password=pw)
            w.unlock_coldkey()
            console.print(f"  [green]✓[/green] {hk}  ([dim]{wname}[/dim])")
        except Exception as e:
            console.print(f"  [red]✗ {hk} ({wname}): {e}[/red]")
            sys.exit(1)

    os.environ.pop("BT_WALLET_PASSWORD", None)
    wallet_passwords.clear()
    console.print()

    # Initial state — check on-chain status for every hotkey
    states: Dict[str, str] = {}
    reg_counts: Dict[str, int] = {hk: 0 for hk in wallets}
    for hk, w in wallets.items():
        try:
            if is_registered(sub, args.netuid, w.hotkey.ss58_address):
                states[hk] = "registered"
                reg_counts[hk] = 1
            else:
                states[hk] = "above_threshold"
        except Exception:
            states[hk] = "above_threshold"

    # Map coldkey ss58 → wallet object (for balance lookups)
    coldkey_wallets: Dict[str, bt.Wallet] = {}
    for hk, w in wallets.items():
        coldkey_wallets[w.coldkeypub.ss58_address] = w

    # Store wallet label per hotkey for display
    for hk, w in wallets.items():
        states[f"_wallet_{hk}"] = hotkey_to_wallet[hk]

    with Live(console=console, refresh_per_second=2, screen=False) as live:
        while True:
            try:
                cost = get_burn_cost(sub, args.netuid)
                # Fetch balance for each distinct coldkey
                ck_balances: Dict[str, float] = {}
                for ck_ss58, w in coldkey_wallets.items():
                    ck_balances[ck_ss58] = float(sub.get_balance(ck_ss58))
                # Build per-hotkey balance dict
                hk_balances: Dict[str, float] = {
                    hk: ck_balances.get(wallets[hk].coldkeypub.ss58_address, 0.0)
                    for hk in wallets
                }
                lowest_coldkey = lowest_not_immune_miner_coldkey(sub, args.netuid)
                is_protected = lowest_coldkey in args.protected_coldkeys if lowest_coldkey else False
            except Exception as e:
                # Connection dropped — reconnect once, then retry next poll
                live.update(f"[yellow]Connection error: {e} — reconnecting…[/yellow]")
                try:
                    sub = bt.Subtensor(network=args.network)
                except Exception:
                    pass
                time.sleep(args.interval)
                continue

            # ── Re-check ALL hotkeys every poll ───────────────────────────────
            for hk, w in wallets.items():
                if hk.startswith("_wallet_"):
                    continue
                on_chain = is_registered(sub, args.netuid, w.hotkey.ss58_address)
                bal = hk_balances.get(hk, 0.0)

                if on_chain:
                    if states[hk] not in ("registered", "just_registered"):
                        states[hk] = "registered"
                    elif states[hk] == "just_registered":
                        states[hk] = "registered"
                else:
                    if states[hk] in ("registered", "just_registered"):
                        console.log(f"[bold red]⚠ {hk} deregistered — waiting for cost to drop below threshold[/bold red]")
                    if is_protected:
                        states[hk] = "protected"
                    elif cost > args.threshold:
                        states[hk] = "above_threshold"
                    elif bal < cost:
                        states[hk] = "insufficient"
                    else:
                        states[hk] = "ready"

            live.update(build_table(cost, args.threshold, hk_balances, states, reg_counts, args.interval, args.dry_run))

            # ── Attempt registration for ONE ready hotkey ─────────────────────
            for hk, w in wallets.items():
                if hk.startswith("_wallet_"):
                    continue
                if states[hk] == "ready":
                    live.stop()
                    ok = try_register(sub, w, args.netuid, args.dry_run)
                    live.start()
                    if ok:
                        states[hk] = "just_registered"
                        reg_counts[hk] += 1
                        if not args.no_start:
                            _start_miner(hk, hotkey_to_miner, args.dry_run)
                        console.print(
                            f"  [dim]Fee spiked after registration — waiting {args.interval}s "
                            f"for it to decay below {args.threshold} τ before next hotkey.[/dim]"
                        )
                    else:
                        states[hk] = "failed"
                    # Refresh balances after spending
                    try:
                        for ck_ss58, ckw in coldkey_wallets.items():
                            ck_balances[ck_ss58] = float(sub.get_balance(ck_ss58))
                        hk_balances = {
                            hk2: ck_balances.get(wallets[hk2].coldkeypub.ss58_address, 0.0)
                            for hk2 in wallets
                        }
                    except Exception:
                        pass
                    live.update(build_table(cost, args.threshold, hk_balances, states, reg_counts, args.interval, args.dry_run))
                    break

            # Reset "failed" each cycle so it retries next poll
            for hk in wallets:
                if not hk.startswith("_wallet_") and states[hk] == "failed":
                    states[hk] = "above_threshold"

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
