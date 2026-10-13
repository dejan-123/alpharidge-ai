#!/usr/bin/env python3
"""
SN45 Miner Fleet Dashboard

Serves a single-page web dashboard with per-miner reputation, emission
multiplier, recent rewards/penalties and on-chain incentive.

Data sources:
  - miners.yaml                    fleet list (name, wallet, hotkey name)
  - ~/.bittensor/wallets/*/hotkeys  hotkey name -> ss58 address
  - api.alpharidge.ai              reputation + per-epoch batch outcomes
  - finney metagraph               uid / registered / incentive (optional)

Run:
  .venv/bin/python scripts/miner_dashboard.py [--port 8099] [--no-chain]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "miners.yaml"
API_BASE = "https://api.alpharidge.ai"
WALLETS_DIR = Path.home() / ".bittensor" / "wallets"

DASH_TTL_S = 60      # api.alpharidge.ai poll interval
CHAIN_TTL_S = 300    # metagraph refresh interval
SUBNET_TTL_S = 600   # subnet-wide reputation scan interval

_state = {"dash": {}, "chain": {}, "subnet": {},
          "updated": {"dash": 0.0, "chain": 0.0, "subnet": 0.0}}
_lock = threading.Lock()
_mg_hotkeys: list = []  # populated by _refresh_chain for the subnet scan


def _fleet():
    cfg = yaml.safe_load(CONFIG_FILE.read_text())
    default_wallet = (cfg.get("global") or {}).get("wallet_name")
    out = []
    for m in cfg.get("miners", []):
        keyfile = WALLETS_DIR / (m.get("wallet_name") or default_wallet) / "hotkeys" / m["hotkey"]
        ss58 = None
        try:
            ss58 = json.loads(keyfile.read_text()).get("ss58Address")
        except Exception:
            pass
        out.append({"name": m["name"], "hotkey_name": m["hotkey"], "ss58": ss58})
    return out


def _api_get(path):
    req = urllib.request.Request(API_BASE + path, headers={"User-Agent": "curl/8.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def _refresh_dash():
    rows = {}
    for m in _fleet():
        if not m["ss58"]:
            continue
        row = {"name": m["name"], "ss58": m["ss58"]}
        try:
            rep = _api_get(f"/dashboard/miners/{m['ss58']}/reputation")
            row["emission_midpoint"] = rep.get("emission_midpoint")
            row["latest"] = rep.get("latest") or {}
            row["rep_history"] = [
                {"epoch": h.get("epoch"), "reputation": h.get("reputation"),
                 "emission_mult": h.get("emission_mult")}
                for h in (rep.get("history") or [])[:30]
            ]
        except Exception:
            row["latest"] = {}
            row["rep_history"] = []
        try:
            batches = _api_get(f"/dashboard/miners/{m['ss58']}/batches")["batches"][:10]
            row["batches"] = [
                {k: b.get(k) for k in
                 ("epoch", "earned_items", "penalty_items", "penalty_breakdown",
                  "reward_points_max", "was_zeroed", "flagged_by_validators",
                  "diagnosis")}
                for b in batches
            ]
        except Exception:
            row["batches"] = []
        rows[m["name"]] = row
    with _lock:
        _state["dash"] = rows
        _state["updated"]["dash"] = time.time()


def _refresh_chain():
    global _mg_hotkeys
    try:
        import bittensor as bt
        st = bt.Subtensor(network="finney")
        mg = st.metagraph(45)
        _mg_hotkeys = list(mg.hotkeys)
        pos = {hk: i for i, hk in enumerate(mg.hotkeys)}
        rows = {}
        for m in _fleet():
            if not m["ss58"]:
                continue
            uid = pos.get(m["ss58"])
            rows[m["name"]] = {
                "registered": uid is not None,
                "uid": uid,
                "incentive": float(mg.incentive[uid]) if uid is not None else 0.0,
            }
        with _lock:
            _state["chain"] = rows
            _state["updated"]["chain"] = time.time()
    except Exception:
        pass


def _refresh_subnet():
    """Reputation of every miner on the subnet — pooled fetch, summarized into
    top / middle / bottom slices so the fleet can be compared to the field."""
    from concurrent.futures import ThreadPoolExecutor
    if not _mg_hotkeys:
        _refresh_chain()
    hotkeys = list(_mg_hotkeys)
    reps = {}

    def one(hk):
        try:
            r = _api_get(f"/dashboard/miners/{hk}/reputation")
            lat = (r or {}).get("latest") or {}
            rep_v, n = lat.get("reputation"), lat.get("samples", 0)
            if rep_v is not None:
                reps[hk] = {"reputation": float(rep_v), "samples": int(n or 0),
                            "emission_mult": lat.get("emission_mult")}
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(one, hotkeys))

    order = sorted(reps.items(), key=lambda kv: -kv[1]["reputation"])
    n = len(order)
    mid = order[max(0, n // 2 - 2):n // 2 + 3] if n else []
    subnet = {
        "count": n,
        "top": [{"hotkey": hk[:10] + "…", **v} for hk, v in order[:5]],
        "middle": [{"hotkey": hk[:10] + "…", **v} for hk, v in mid],
        "bottom": [{"hotkey": hk[:10] + "…", **v} for hk, v in order[-5:]],
        "ours": [
            {"hotkey": hk[:10] + "…", "name": name, **reps[hk]}
            for name, hk in ((m["name"], m["ss58"]) for m in _fleet())
            if hk in reps
        ],
    }
    # rank of each of our miners inside the subnet ordering
    rank = {hk: i + 1 for i, (hk, _) in enumerate(order)}
    for m in _fleet():
        if m["ss58"] in rank:
            subnet.setdefault("our_ranks", {})[m["name"]] = rank[m["ss58"]]
    with _lock:
        _state["subnet"] = subnet
        _state["updated"]["subnet"] = time.time()


def _loop(kind, fn, ttl):
    while True:
        try:
            fn()
        except Exception:
            pass
        time.sleep(ttl)


app = FastAPI(title="SN45 Miner Dashboard")


@app.get("/api/fleet")
def api_fleet():
    with _lock:
        return JSONResponse({
            "miners": _state["dash"],
            "chain": _state["chain"],
            "subnet": _state["subnet"],
            "updated": _state["updated"],
        })


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SN45 Miner Fleet</title>
<style>
  body { font-family: ui-monospace, monospace; background:#0d1117; color:#c9d1d9; margin:24px; }
  h1 { font-size:18px; } .sub { color:#8b949e; font-size:12px; margin-bottom:16px; }
  table { border-collapse:collapse; width:100%; }
  th,td { border-bottom:1px solid #21262d; padding:6px 8px; text-align:right; font-size:13px; }
  th { color:#8b949e; text-align:right; } td:first-child,th:first-child { text-align:left; }
  .good { color:#3fb950; } .warn { color:#d29922; } .bad { color:#f85149; }
  .bar { background:#21262d; border-radius:3px; height:10px; width:120px; position:relative; display:inline-block; vertical-align:middle; }
  .fill { background:#388bfd; height:100%; border-radius:3px; }
  .mid { position:absolute; top:-2px; bottom:-2px; width:2px; background:#d29922; }
  .pen { color:#f85149; font-size:12px; }
  details { font-size:12px; color:#8b949e; } summary { cursor:pointer; }
  .ep { display:inline-block; margin-right:8px; }
</style></head><body>
<h1>SN45 Miner Fleet</h1>
<div class="sub" id="upd">loading…</div>
<table><thead><tr>
  <th>miner</th><th>uid</th><th>reputation</th><th>rep</th><th>mult</th>
  <th>samples</th><th>incentive</th><th>rank</th><th>last epochs (points [items] / penalties)</th>
</tr></thead><tbody id="rows"></tbody></table>

<h1 style="margin-top:32px">Subnet reputation <span class="sub" id="subnet-upd"></span></h1>
<table><thead><tr>
  <th>band</th><th>miners (hotkey / reputation / mult / samples)</th>
</tr></thead><tbody id="subnet-rows"></tbody></table>
<script>
async function load() {
  const d = await (await fetch('/api/fleet')).json();
  const tb = document.getElementById('rows'); tb.innerHTML = '';
  // Fit the live gate curve from served (reputation, emission_mult) pairs:
  // logit(m) = gain*(r - mid). The API's emission_midpoint field has proven
  // stale vs the curve the snapshots were computed with, so derive it.
  let mid = 0.505, gain = 32;
  {
    const xs = [], ys = [];
    for (const m of Object.values(d.miners)) {
      for (const h of (m.rep_history || [])) {
        const r = h.reputation, mu = h.emission_mult;
        if (r != null && mu != null && mu > 1e-6 && mu < 1 - 1e-6) {
          xs.push(r); ys.push(Math.log(mu / (1 - mu)));
        }
      }
    }
    if (xs.length >= 5) {
      const n = xs.length, mx = xs.reduce((a,b)=>a+b)/n, my = ys.reduce((a,b)=>a+b)/n;
      let sxx = 0, sxy = 0;
      for (let i = 0; i < n; i++) { sxx += (xs[i]-mx)*(xs[i]-mx); sxy += (xs[i]-mx)*(ys[i]-my); }
      if (sxx > 1e-9) { gain = sxy / sxx; mid = mx - my / gain; }
    }
  }
  const names = Object.keys(d.miners).sort();
  for (const n of names) {
    const m = d.miners[n], c = (d.chain||{})[n] || {};
    const lat = m.latest || {};
    const rep = lat.reputation, mult = lat.emission_mult;
    const repCls = rep == null ? '' : (rep >= mid ? 'good' : (rep >= mid-0.1 ? 'warn' : 'bad'));
    const bar = rep == null ? '' :
      `<span class="bar"><span class="fill" style="width:${Math.min(100,rep*100)}%"></span>` +
      `<span class="mid" style="left:${mid*100}%"></span></span>`;
    const eps = (m.batches||[]).map(b => {
      const pen = b.penalty_items ? `<span class="pen">-${b.penalty_items} ${Object.keys(b.penalty_breakdown||{}).join(',')}</span>` : '0';
      const sev = ((b.diagnosis||{}).severity) || '';
      const sevCls = sev === 'ok' ? 'good' : (sev === 'warn' ? 'warn' : (sev ? 'bad' : ''));
      const pts = b.reward_points_max == null ? '—' :
        `<b class="${sevCls}">${Math.round(b.reward_points_max)}pts</b>`;
      const flags = (b.was_zeroed ? ' <b class="bad">ZEROED</b>' : '') +
        (b.flagged_by_validators ? ` <span class="warn">flagged×${b.flagged_by_validators}</span>` : '');
      return `<span class="ep">${b.epoch}: ${pts} <span class="sub">(${b.earned_items} it)</span> / ${pen}${flags}</span>`;
    }).join('');
    tb.innerHTML += `<tr>
      <td>${n}${c.registered===false?' <span class="bad">(unreg)</span>':''}</td>
      <td>${c.uid ?? '…'}</td>
      <td>${bar}</td>
      <td class="${repCls}">${rep==null?'—':rep.toFixed(3)}</td>
      <td>${mult==null?'—':mult.toFixed(2)}x</td>
      <td>${lat.samples ?? '—'}</td>
      <td>${c.incentive==null?'…':c.incentive.toFixed(6)}</td>
      <td>${((d.subnet||{}).our_ranks||{})[n] ? '#'+d.subnet.our_ranks[n]+'/'+d.subnet.count : '—'}</td>
      <td style="text-align:left"><details><summary>${(m.batches||[]).length} epochs</summary>${eps}</details></td>
    </tr>`;
  }
  const t = d.updated && d.updated.dash;
  document.getElementById('upd').textContent =
    'updated ' + (t ? new Date(t*1000).toLocaleTimeString() : '…') +
    ' — gate mid ' + mid.toFixed(3) + ' gain ' + gain.toFixed(0) +
    ' (fitted) — refreshes every 60s';

  const sn = d.subnet || {};
  const bands = [['top 5', sn.top], ['median', sn.middle], ['bottom 5', sn.bottom]];
  const srows = document.getElementById('subnet-rows'); srows.innerHTML = '';
  for (const [label, list] of bands) {
    const cells = (list||[]).map(x =>
      `<span class="ep">${x.hotkey} <b class="${x.reputation>=mid?'good':(x.reputation>=mid-0.1?'warn':'bad')}">${x.reputation.toFixed(3)}</b>` +
      ` ${x.emission_mult==null?'':x.emission_mult.toFixed(2)+'x'} n=${x.samples}</span>`).join('');
    srows.innerHTML += `<tr><td>${label}</td><td style="text-align:left">${cells||'—'}</td></tr>`;
  }
  const ours = (sn.ours||[]).map(x =>
    `<span class="ep">${x.name} <b>${x.reputation.toFixed(3)}</b></span>`).join('');
  srows.innerHTML += `<tr><td>ours</td><td style="text-align:left">${ours||'—'}</td></tr>`;
  document.getElementById('subnet-upd').textContent =
    sn.count ? `— ${sn.count} miners scanned` : '';
}
load(); setInterval(load, 60000);
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--no-chain", action="store_true",
                   help="skip metagraph queries (faster, no uid/incentive)")
    args = p.parse_args()

    threading.Thread(target=_loop, args=("dash", _refresh_dash, DASH_TTL_S), daemon=True).start()
    if not args.no_chain:
        threading.Thread(target=_loop, args=("chain", _refresh_chain, CHAIN_TTL_S), daemon=True).start()
        threading.Thread(target=_loop, args=("subnet", _refresh_subnet, SUBNET_TTL_S), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
