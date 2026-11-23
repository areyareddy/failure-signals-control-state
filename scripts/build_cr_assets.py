#!/usr/bin/env python3
"""Emit macros and tables for the added control analyses (s61, s62).

Inputs (read-only, produced by the experiment-root scripts):
  audit/cr_s1_controls.json    -- scripts/s61_cr_s1_controls.py
  audit/cr_s2s4_controls.json  -- scripts/s62_cr_s2s4_controls.py
Outputs: cr_macros.tex and tables/cr_*.tex in this paper directory.
No number in the paper that comes from these analyses is typed by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
ROOT = PAPER.parents[1]
S1 = json.loads((ROOT / "audit/cr_s1_controls.json").read_text(encoding="utf-8"))
S24 = json.loads((ROOT / "audit/cr_s2s4_controls.json").read_text(encoding="utf-8"))

macros: dict[str, str] = {}


def pct(x, d=1):
    s = f"{100 * x:.{d}f}"
    return ("$-$" + s[1:] if s.startswith("-") else s) + "\\%"


def num(n):
    return f"{n:,}".replace(",", "{,}")


def m(name, val):
    assert name not in macros, name
    macros[name] = val


# ------------------------------------------------------------------ S1 ledgers
L = S1["ledgers"]
ho = L["historical_heldout40"]
m("CrHistHeldTruncRescues", str(ho["truncated"]["rescues"]))
m("CrHistHeldTruncAttempts", str(ho["truncated"]["attempts"]))
m("CrHistHeldTruncRho", f"{ho['truncated']['rho_e5']:.2f}")
m("CrHistHeldAllRescues", str(ho["all"]["rescues"]))
m("CrHistHeldAllAttempts", str(ho["all"]["attempts"]))
m("CrHistHeldAllRho", f"{ho['all']['rho_e5']:.2f}")
dev_rho = L["historical_dev43"]["truncated"]["rho_e5"]
m("CrTruncRhoGap", f"{ho['truncated']['rho_e5'] / dev_rho:.0f}$\\times$")
for b, w in (("block1", "One"), ("block2", "Two")):
    lb = L[b]
    m(f"CrB{w}TruncRescues", str(lb["truncated"]["rescues"]))
    m(f"CrB{w}TruncAttempts", str(lb["truncated"]["attempts"]))
    m(f"CrB{w}TruncRho", f"{lb['truncated']['rho_e5']:.2f}")
    m(f"CrB{w}AllRho", f"{lb['all']['rho_e5']:.2f}")
    p = S1["persistence"][b]
    m(f"CrB{w}TruncPersist", pct(p["stage1_truncated"]["rate"], 0))
    m(f"CrB{w}OtherPersist", pct(p["stage1_other"]["rate"], 0))

# held-out ledger table: class, dev rho, historical held-out, block 1, block 2
CLASS_ORDER = S1["class_order_formula"]["frozen_order"]
NAMES = {c: c.replace("_", " ") for c in CLASS_ORDER}
rows = []
for c in CLASS_ORDER + ["all"]:
    cells = [NAMES.get(c, "all")]
    d = L["historical_dev43"][c]
    cells.append(f"{d['rescues']}/{d['attempts']}")
    cells.append(f"{d['rho_e5']:.2f}")
    for k in ("historical_heldout40", "block1", "block2"):
        r = L[k][c]
        cells.append(f"{r['rescues']}/{r['attempts']}")
        cells.append(f"{r['rho_e5']:.2f}")
    line = " & ".join(cells) + " \\\\"
    if c == "all":
        line = "\\midrule\n" + line
    rows.append(line)
(PAPER / "tables/cr_heldout_ledger.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

# class-order formula values
cof = S1["class_order_formula"]
m("CrGlobalRate", f"{cof['global_rate']:.3f}")
rows = []
for i, c in enumerate(CLASS_ORDER, 1):
    d = L["historical_dev43"][c]
    rows.append(f"{i} & {NAMES[c]} & {d['rescues']}/{d['attempts']} & {cof['p_shrunk_values'][c]:.3f} & "
                f"{num(round(cof['mean_tokens'][c]))} & {cof['rho_e5'][c]:.2f} \\\\")
(PAPER / "tables/cr_class_order_formula.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ S1 controls
VARIANTS = [
    ("timeout_only", "Skip timeout roots", "targeted"),
    ("trunccap_only", "Cap truncated roots at 2", "targeted"),
    ("coarse", "Both coarse clauses", "targeted"),
    ("rules", "Full frozen rules", "targeted"),
    ("uniform_cap2", "Every root capped at 2", "class-blind"),
    ("uniform_cap3", "Every root capped at 3", "class-blind"),
    ("timeout_plus_uniform_cap3", "Skip timeout, others at 3", "partly"),
]


def maxdef(s):
    return max(int(k) for k in s["deficit_histogram_all24"])


rows = []
prev_group = None
for key, label, group in VARIANTS:
    if prev_group and group != prev_group:
        rows.append("\\midrule")
    prev_group = group
    cells = [label]
    for b in ("block1", "block2"):
        s = S1["blocks"][b]["summary"][key]
        cells += [f"{s['parity_orderings']}/24", str(maxdef(s)), pct(s["median_exhaustion_saving_all24"])]
    rows.append(" & ".join(cells) + " \\\\")
(PAPER / "tables/cr_s1_controls.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

for b, w in (("block1", "One"), ("block2", "Two")):
    sm = S1["blocks"][b]["summary"]
    m(f"CrB{w}RulesExhMedian", pct(sm["rules"]["median_exhaustion_saving_all24"]))
    m(f"CrB{w}CapTwoExhMedian", pct(sm["uniform_cap2"]["median_exhaustion_saving_all24"]))
    m(f"CrB{w}CapTwoMaxLoss", str(maxdef(sm["uniform_cap2"])))
    m(f"CrB{w}CapTwoMinLoss", str(min(int(k) for k in sm["uniform_cap2"]["deficit_histogram_all24"])))
    m(f"CrB{w}ToCapThreeExhMedian", pct(sm["timeout_plus_uniform_cap3"]["median_exhaustion_saving_all24"]))
    m(f"CrB{w}ToCapThreeMaxLoss", str(maxdef(sm["timeout_plus_uniform_cap3"])))
    m(f"CrB{w}RulesMaxLoss", str(maxdef(sm["rules"])))
    m(f"CrB{w}CoarsePrimary", pct(sm["coarse"]["primary_saving"]))
    m(f"CrB{w}CoarseParityMedian", pct(sm["coarse"]["median_saving_over_parity_orderings"]))
    bt = S1["blocks"][b]["bootstrap_primary_ordering"]["rules"]
    m(f"CrB{w}Boot", f"{pct(bt['median'])} [{pct(bt['ci_low'])}, {pct(bt['ci_high'])}]")
    if b == "block1":
        se = (bt["ci_high"] - bt["ci_low"]) / (2 * 1.959964)
        need = 40 * (se / (bt["median"] / (1.959964 + 0.841621))) ** 2
        m("CrPowerRoots", str(int(math.ceil(need / 10.0) * 10)))

# ------------------------------------------------------------------ S2
br = S24["s2_breakdown"]
COH = {"primary_eval": "Held-out, block 1", "eval_block2": "Held-out, block 2", "full_pool_dev": "Development"}
rows = []
for coh in ("primary_eval", "eval_block2", "full_pool_dev"):
    cells = [COH[coh]]
    for arm in "ABCDE":
        r = next((x for x in br if x["cohort"] == coh and x["arm"] == arm), None)
        cells.append(f"{r['n']} ({r['capped']})" if r else "---")
    cells.append(str(sum(x["n"] for x in br if x["cohort"] == coh)))
    rows.append(" & ".join(cells) + " \\\\")
(PAPER / "tables/cr_s2_breakdown.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
m("CrSTwoBlockOneBranches", str(sum(x["n"] for x in br if x["cohort"] == "primary_eval")))
m("CrSTwoBlockTwoBranches", str(sum(x["n"] for x in br if x["cohort"] == "eval_block2")))
m("CrSTwoDevBranches", str(sum(x["n"] for x in br if x["cohort"] == "full_pool_dev")))

lp = S24["s2_loops"]
m("CrLoopN", str(lp["loop_heuristic_positive"]))
m("CrLoopOf", str(lp["capped_generations"]))
m("CrLoopRate", pct(lp["rate"], 0))

fs = S24["four_slot"]
for arm in ("A", "C"):
    f = fs[arm]
    m(f"CrFour{arm}Lower", pct(f["prefill_share_gpu_time_lower"], 2))
    m(f"CrFour{arm}Upper", pct(f["prefill_share_gpu_time_upper"], 2))
    m(f"CrFour{arm}AggDecode", f"{f['aggregate_decode_tok_s']:.0f}")
    m(f"CrFour{arm}ReqDecode", f"{f['per_request_decode_tok_s']:.1f}")
    m(f"CrFour{arm}Concurrency", f"{f['mean_concurrency']:.1f}")

# ------------------------------------------------------------------ S3/S4 controls
gc = S24["s4_global_cap"]
blocks = sorted(gc)
lo = min(gc[b]["global_saving"] for b in blocks)
hi = max(gc[b]["global_saving"] for b in blocks)
m("CrGlobalCapSavedRange", f"{100 * lo:.1f}--{100 * hi:.1f}\\%")
losses = [gc[b]["long_verified"] - gc[b]["global_verified"] for b in blocks]
m("CrGlobalCapLosses", str(sum(losses)))
m("CrGlobalCapLossBlocks", str(sum(1 for x in losses if x)))
lost = [x for b in blocks for x in gc[b]["lost_by_global_cap"]]
assert len(lost) == 1 and not lost[0]["affected"]
m("CrGlobalCapLostTok", num(lost[0]["tokens"]))
sd = S24["s3_global_cap"]["dev"]
m("CrSThreeDevLongVerified", str(sd["long_verified"]))
m("CrSThreeDevGlobalLoss", str(sd["long_verified"] - sd["global_verified"]))
rows = []
for b in blocks:
    g = gc[b]
    rows.append(f"S4 block {b} & {g['long_verified']} & {g['targeted_verified']} & {g['global_verified']} & "
                f"{pct(g['targeted_saving_counterfactual'])} & {pct(g['global_saving'])} \\\\")
rows.append("\\midrule")
for k, lab in (("eval", "S3 held-out"), ("dev", "S3 development")):
    g = S24["s3_global_cap"][k]
    rows.append(f"{lab} & {g['long_verified']} & {g['targeted_verified']} & {g['global_verified']} & "
                f"{pct(g['targeted_saving_counterfactual'])} & {pct(g['global_saving'])} \\\\")
(PAPER / "tables/cr_global_cap.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

tp = S24["s4_truncation_prediction"]
m("CrAffOverCap", f"{tp['affected']['over_8192']}/{tp['affected']['n']}")
m("CrAffOverCapPct", pct(tp["affected"]["over_8192_rate"], 0))
m("CrUnaffOverCap", f"{tp['unaffected']['over_8192']}/{tp['unaffected']['n']}")
m("CrUnaffOverCapPct", pct(tp["unaffected"]["over_8192_rate"], 0))
m("CrAffVerifiedDraws", str(len(tp["affected_verified_long_lengths"])))
m("CrAffVerifiedRoots", str(tp["affected_verified_distinct_roots"]))
m("CrAffVerifiedMaxTok", num(max(tp["affected_verified_long_lengths"])))

bs = S24["s4_bootstrap"]
for key, w in (("joules", "Energy"), ("tokens", "Tokens")):
    p, ci = bs[key]["point"]["pooled"], bs[key]["ci95"]["pooled"]
    m(f"CrCap{w}Pooled", pct(p))
    m(f"CrCap{w}PooledCI", f"[{pct(ci[0])}, {pct(ci[1])}]")
    for b, bw in (("1", "One"), ("2", "Two"), ("3", "Three")):
        c = bs[key]["ci95"][b]
        m(f"CrCap{w}B{bw}CI", f"[{pct(c[0])}, {pct(c[1])}]")

ps = S24["s4_bootstrap_prespecified"]
for key, w in (("joules", "Energy"), ("tokens", "Tokens")):
    lo, hi = ps[key]["ci95"]
    m(f"CrCap{w}PrespecCI", f"[{pct(lo)}, {pct(hi)}]")

# S4 run record: canary, stop limit, freeze timing.  Read from the run's own logs.
import csv
import re
import sqlite3
from datetime import datetime

_mk = list(csv.DictReader((ROOT / "audit/telemetry/cap_policy.markers.csv").open(encoding="utf-8")))
_t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
_gen = next(r for r in _mk if r["label"] == "generation_start")
_ends = [r for r in _mk if r["label"].startswith("cap_eval_") and r["label"].endswith("_end")]
_secs = lambda r: float(re.search(r"seconds=([\d.]+)", r["note"]).group(1))
_canary_s = sum(_secs(r) for r in _ends if _t(r["iso"]) < _t(_gen["iso"]))
_run_s = sum(_secs(r) for r in _ends)
_first = min(_t(r["iso"]) for r in _mk if r["label"].startswith("cap_eval_") and r["label"].endswith("_start"))
_con = sqlite3.connect(f"file:{(ROOT / 'data/cap_policy.sqlite').as_posix()}?mode=ro", uri=True)
_canary_cohorts = [r["label"][:-len("_start")] for r in _mk
                   if r["label"].startswith("cap_eval_") and r["label"].endswith("_start")
                   and _t(r["iso"]) < _t(_gen["iso"])]
_n_canary = sum(_con.execute("select count(*) from branches where cohort = ?", (c,)).fetchone()[0]
                for c in _canary_cohorts)
_n_total = _con.execute("select count(*) from branches").fetchone()[0]
_con.close()
_spec = (ROOT / "spec/CAP_POLICY_SPEC.md").read_text(encoding="utf-8")
_limit = re.search(r"Hard limit (\d+(?:\.\d+)?) GPU-hours", _spec).group(1)
_raised = re.search(r"budget raised to ([\d.]+) GPU-h", _gen["note"]).group(1)
_hashes = json.loads((ROOT / "audit/cap_policy_hashes.json").read_text(encoding="utf-8"))
_delay_min = (_t(_hashes["captured_utc"]) - _first).total_seconds() / 60
m("CapCanaryBranches", str(_n_canary))
m("CapCanaryProjHours", f"{_canary_s / _n_canary * _n_total / 3600:.2f}")
m("CapBudgetFrozen", _limit)
m("CapBudgetRaised", _raised)
m("CapRunGPUHours", f"{_run_s / 3600:.1f}")
m("CapHashDelayMin", f"{_delay_min:.0f}")

out = ["% Generated by scripts/build_cr_assets.py from audit/cr_*.json. Do not edit."]
out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
(PAPER / "cr_macros.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {len(macros)} macros and 5 tables")
for k, v in macros.items():
    print(f"  {k} = {v}")
