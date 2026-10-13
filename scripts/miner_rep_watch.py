#!/usr/bin/env python3
"""SN45 miner reputation monitor — Phase 0 measurement.

Polls api.alpharidge.ai for a miner's aggregate reputation, live mechanism
profile, and per-epoch batch outcomes; scans the miner's log for LLM failure
signatures; decomposes the implied audit_v2 channel score; and appends a JSONL
snapshot each run so recovery can be tracked across epochs.

The public API does not expose per-channel reputation. Under profile v17 the
only weighted channels are triage (w=1, default .97), floor (w=.5, default
.953) and audit_v2 (w=3, default .41), so the audit_v2 score is solved as

    a = (r * (w_t + w_f + w_a) - w_t*t - w_f*f) / w_a

assuming triage/floor sit at their defaults — printed with a sensitivity band.

Run:
  .venv/bin/python scripts/miner_rep_watch.py miner-01
  .venv/bin/python scripts/miner_rep_watch.py miner-01 --watch 1200
  .venv/bin/python scripts/miner_rep_watch.py miner-01 --compare   # metagraph scan
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "miners.yaml"
API_BASE = "https://api.alpharidge.ai"
WALLETS_DIR = Path.home() / ".bittensor" / "wallets"
LOGS_DIR = REPO_ROOT / "logs" / "miners"
SNAPSHOT_FILE = REPO_ROOT / "logs" / "miner_rep_history.jsonl"

NETUID = 45
ANSI = re.compile(r"\x1b\[[0-9;]*m")

FAIL_PATTERNS = {
    "truncated": "cut off at max_tokens",
    "empty_retries": "empty after retries",
    "parse_err": "failed: ",
    "analysis_none": "analysis returned None",
    "degraded": "degraded analysis",
    "submitted_anyway": "submitting anyway",
}


def fleet_entry(name: str) -> dict:
    cfg = yaml.safe_load(CONFIG_FILE.read_text())
    default_wallet = (cfg.get("global") or {}).get("wallet_name")
    for m in cfg.get("miners", []):
        if m.get("name") != name:
            continue
        wallet = m.get("wallet_name") or default_wallet
        keyfile = WALLETS_DIR / wallet / "hotkeys" / m["hotkey"]
        ss58 = None
        try:
            ss58 = json.loads(keyfile.read_text()).get("ss58Address")
        except Exception:
            pass
        return {"name": name, "ss58": ss58,
                "log": LOGS_DIR / f"{name}.log"}
    raise SystemExit(f"miner {name!r} not in {CONFIG_FILE}")


def api_get(path: str) -> dict:
    req = urllib.request.Request(API_BASE + path, headers={"User-Agent": "curl/8.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def log_failures(path: Path) -> dict:
    """Per-UTC-hour failure signature counts, plus V2 completions."""
    per_hour = defaultdict(Counter)
    totals = Counter()
    if not path.exists():
        return {}
    for line in path.open(errors="replace"):
        line = ANSI.sub("", line)
        m = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}):\d{2}", line)
        if not m:
            continue
        hour = f"{m.group(1)}T{m.group(2)}"
        for key, pat in FAIL_PATTERNS.items():
            if pat in line:
                per_hour[hour][key] += 1
                totals[key] += 1
                break
        if "V2 analysis complete" in line:
            per_hour[hour]["v2_complete"] += 1
            totals["v2_complete"] += 1
    return {"per_hour": {h: dict(c) for h, c in sorted(per_hour.items())},
            "totals": dict(totals)}


def implied_audit_v2(rep: float, weights: dict, defaults: dict,
                     triage: float, floor: float) -> float | None:
    """Solve the weighted mean for audit_v2 given assumed triage/floor values."""
    w_a = float(weights.get("audit_v2", 0.0))
    if w_a <= 0.0:
        return None
    w_t = float(weights.get("triage", 0.0))
    w_f = float(weights.get("floor", 0.0))
    total_w = w_t + w_f + w_a
    return (rep * total_w - w_t * triage - w_f * floor) / w_a


def emission_mult(rep: float, mid: float, gain: float, bonus_start: float,
                  bonus_full: float, ceiling: float) -> float:
    gate = 1.0 / (1.0 + math.exp(-gain * (rep - mid)))
    ramp = min(1.0, max(0.0, (rep - bonus_start) / max(1e-9, bonus_full - bonus_start)))
    return gate * (1.0 + ceiling * ramp)


def snapshot(miner: dict) -> dict:
    rep = api_get(f"/dashboard/miners/{miner['ss58']}/reputation")
    batches = api_get(f"/dashboard/miners/{miner['ss58']}/batches").get("batches", [])
    latest = rep.get("latest") or {}
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": miner["name"], "hotkey": miner["ss58"],
        "epoch": latest.get("epoch"),
        "reputation": latest.get("reputation"),
        "samples": latest.get("samples"),
        "emission_mult": latest.get("emission_mult"),
        "profile_version": rep.get("profile_version"),
        "channel_weights": rep.get("channel_weights"),
        "channel_defaults": rep.get("channel_defaults"),
        "emission_midpoint": rep.get("emission_midpoint"),
        "emission_gain": rep.get("emission_gain"),
        "bonus_start": rep.get("bonus_start"),
        "bonus_full": rep.get("bonus_full"),
        "bonus_ceiling": rep.get("bonus_ceiling"),
        "history": rep.get("history") or [],
        "batches": batches,
        "log": log_failures(miner["log"]),
    }


def subnet_rank(hotkey: str) -> tuple[int | None, int | None]:
    """Rank among all metagraph hotkeys by API reputation. None when unavailable."""
    try:
        import bittensor as bt
        mg = bt.Subtensor(network="finney").metagraph(NETUID)
        reps = {}
        from concurrent.futures import ThreadPoolExecutor

        def one(hk):
            try:
                r = api_get(f"/dashboard/miners/{hk}/reputation")
                v = ((r or {}).get("latest") or {}).get("reputation")
                if v is not None:
                    reps[hk] = float(v)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=24) as ex:
            list(ex.map(one, list(mg.hotkeys)))
        order = sorted(reps.values(), reverse=True)
        if hotkey not in reps:
            return None, len(order)
        return order.index(reps[hotkey]) + 1, len(order)
    except Exception as e:
        print(f"subnet scan unavailable: {e}", file=sys.stderr)
        return None, None


def report(snap: dict) -> None:
    w = snap["channel_weights"] or {}
    d = snap["channel_defaults"] or {}
    r = snap["reputation"]
    mid = float(snap.get("emission_midpoint") or 0.0)
    gain = float(snap.get("emission_gain") or 0.0)

    print(f"== {snap['name']}  {snap['hotkey'][:12]}…  epoch {snap['epoch']}  "
          f"profile v{snap.get('profile_version')} ({snap['ts'][:19]}Z)")
    print(f"reputation {r:.4f}   samples {snap['samples']}   "
          f"emission_mult x{snap['emission_mult']:.3f}  "
          f"(gate mid {mid} gain {gain:g}, bonus {snap.get('bonus_start')}→"
          f"{snap.get('bonus_full')}, cap x{1 + float(snap.get('bonus_ceiling') or 0):.1f})")
    print(f"channel weights: {w}   defaults: {d}")

    t_def, f_def = float(d.get("triage", 0.0)), float(d.get("floor", 0.0))
    print("implied audit_v2 = (r·Σw − w_t·triage − w_f·floor) / w_a:")
    for t in (t_def, 0.90, 0.80):
        for f in (f_def, 0.85):
            a = implied_audit_v2(r, w, d, t, f)
            if a is not None:
                print(f"   triage={t:.3f} floor={f:.3f} -> audit_v2 ≈ {a:.3f}")
    need_bonus = implied_audit_v2(float(snap.get("bonus_start") or 0), w, d, t_def, f_def)
    need_full = implied_audit_v2(float(snap.get("bonus_full") or 0), w, d, t_def, f_def)
    if need_bonus is not None:
        print(f"   audit_v2 needed for bonus start {snap.get('bonus_start')}: "
              f"≈ {need_bonus:.3f}   for full {snap.get('bonus_full')}: ≈ {need_full:.3f}")

    hist = [h for h in snap["history"] if h.get("reputation") is not None]
    if hist:
        print(f"reputation last {len(hist)} epochs: "
              + " ".join(f"{h['reputation']:.3f}" for h in hist[:12]))
        mult_now = emission_mult(r, mid, gain, float(snap.get("bonus_start") or 1),
                                 float(snap.get("bonus_full") or 1),
                                 float(snap.get("bonus_ceiling") or 0))
        print(f"   fitted mult now {mult_now:.3f}; +0.01 rep -> "
              f"{emission_mult(r + 0.01, mid, gain, float(snap.get('bonus_start') or 1), float(snap.get('bonus_full') or 1), float(snap.get('bonus_ceiling') or 0)):.3f}")

    print("recent batches:")
    for b in (snap["batches"] or [])[:14]:
        pen = b.get("penalty_breakdown") or {}
        print(f"   e{b.get('epoch')}  items={b.get('earned_items')} "
              f"pen={b.get('penalty_items')} {pen if pen else ''} "
              f"pts={b.get('reward_points_max')} "
              f"flags={b.get('flagged_by_validators')} "
              f"zeroed={b.get('was_zeroed')}")

    lg = snap.get("log") or {}
    totals = lg.get("totals") or {}
    if totals:
        done = totals.get("v2_complete", 0)
        fail = sum(totals.get(k, 0) for k in
                   ("truncated", "empty_retries", "parse_err"))
        print(f"log totals: v2_complete={done}  llm_fails={fail}  "
              f"analysis_none={totals.get('analysis_none', 0)}  "
              f"degraded={totals.get('degraded', 0)}  "
              f"submitted_anyway={totals.get('submitted_anyway', 0)}")
        print("failures by hour (last 24h of log):")
        for h, c in list(lg.get("per_hour", {}).items())[-24:]:
            f_h = sum(c.get(k, 0) for k in ("truncated", "empty_retries", "parse_err"))
            print(f"   {h}  llm_fails={f_h} none={c.get('analysis_none', 0)} "
                  f"degraded={c.get('degraded', 0)} anyway={c.get('submitted_anyway', 0)} "
                  f"complete={c.get('v2_complete', 0)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("miner", help="miner name from miners.yaml (e.g. miner-01)")
    p.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                   help="re-poll every N seconds, appending to %s" % SNAPSHOT_FILE)
    p.add_argument("--compare", action="store_true",
                   help="scan the subnet metagraph and print reputation rank")
    args = p.parse_args()

    miner = fleet_entry(args.miner)
    if not miner["ss58"]:
        raise SystemExit(f"no ss58 for {args.miner} (hotkey file missing?)")

    while True:
        snap = snapshot(miner)
        report(snap)
        with SNAPSHOT_FILE.open("a") as fh:
            fh.write(json.dumps({k: v for k, v in snap.items()
                                 if k not in ("history", "batches", "log")}) + "\n")
        if args.compare:
            rank, n = subnet_rank(miner["ss58"])
            print(f"subnet rank: {rank}/{n}" if rank else "no reputation row on API")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
