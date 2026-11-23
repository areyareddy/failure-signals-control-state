"""Step s56 -- frozen analysis for the targeted-8K cap-policy replication.

Governed by spec/CAP_POLICY_SPEC.md. Written and hashed BEFORE any outcome was opened:
the run is blinded (s12_run.py --blind withholds outcomes, tokens and wall times from
stdout) and no query against cap_policy.sqlite outcomes has been made. The hash record in
audit/cap_policy_hashes.json states the timing honestly -- this file was frozen after
generation BEGAN, not before it, which is weaker than the ideal and is recorded as such
rather than glossed.

WHAT THE POLICIES ARE, in branch terms:

    Long        = the 40 `long` branches of a block                    (cap 12,288)
    Targeted-8K = the 25 unaffected `long` branches
                + the 15 `short` branches on stage-one-truncated roots (cap 8,192)

So 25 of 40 roots are the SAME branch under both policies and are ties by construction.
That is why the primary endpoint is reported twice: policy-level over 40 roots (what a
deployer experiences, structurally diluted) and conditional on the 15 affected roots
(where the policies actually differ). Reporting only one of them misleads in one
direction or the other.

NEVER POOLED. Three blocks over the same 40 theorems are 120 observations on 40 theorems.
Per-block figures are primary; any aggregate uses a root-cluster bootstrap over the 40
theorems with blocks as repeated measures.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import stats as st  # noqa: E402
from s17_statistical_tests import mcnemar_exact  # noqa: E402
import s55_nvml_logger as nv  # noqa: E402

DB = ROOT / "data/cap_policy.sqlite"
PLAN = ROOT / "audit/cap_policy_plan.json"
OUT = ROOT / "audit/cap_policy_analysis.json"
TAG = "cap_policy"
BOOT_REPS = 20000
BOOT_SEED = 20260828
SHORT_CAP = 8192


def load() -> tuple[dict, dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = list(con.execute(
        "SELECT branch_key, root_id, cohort, seed_block, seed, verified, outcome_state, "
        "completion_tokens, request_wall_seconds, verifier_wall_seconds, model_calls, "
        "completion, claimed_at, finished_at, status FROM branches"))
    con.close()
    by: dict = defaultdict(dict)
    incomplete = []
    for (bk, rid, cohort, blk, seed, ver, state, toks, req, vfy, calls,
         comp, t0, t1, status) in rows:
        if status != "done":
            incomplete.append(bk)
            continue
        pol = "short" if "_short_" in cohort else "long"
        by[(blk, pol)][rid] = {
            "verified": bool(ver), "state": state, "tokens": toks or 0,
            "request_s": req or 0.0, "verifier_s": vfy or 0.0,
            "calls": calls or 0, "completion": comp or "",
            "seed": seed, "t0": t0, "t1": t1, "cohort": cohort}
    return by, {"incomplete": incomplete, "n_rows": len(rows)}


def energy_for(branch: dict) -> dict | None:
    """Trapezoidal joules over this branch's claimed->finished window."""
    if not branch["t0"] or not branch["t1"]:
        return None
    try:
        a = nv.parse_ts(branch["t0"].replace("Z", "+00:00"))
        b = nv.parse_ts(branch["t1"].replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return nv.integrate(TAG, a, b)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    affected = set(plan["truncated_roots"])
    unaffected = set(plan["unaffected_roots"])
    by, meta = load()

    print("[1/5] completeness gate")
    blocks = sorted({b for (b, _p) in by})
    complete_blocks = []
    for b in blocks:
        n_long = len(by.get((b, "long"), {}))
        n_short = len(by.get((b, "short"), {}))
        ok = (n_long == 40 and n_short == 15)
        print(f"      block {b}: long {n_long}/40  short {n_short}/15  "
              f"{'COMPLETE' if ok else 'incomplete -> reported as NOT RUN'}")
        if ok:
            complete_blocks.append(b)
    if not complete_blocks:
        print("\nNo complete block. Per the specification a partly-generated block is "
              "reported as not run, never as a subset. Nothing further is computed.")
        OUT.write_text(json.dumps({"complete_blocks": [], "reason": "no complete block"},
                                  indent=2) + "\n", encoding="utf-8")
        return 0

    print("\n[2/5] integrity gate: the byte-prefix property on matched pairs")
    prefix = {"checked": 0, "prefix_ok": 0, "equal_ok": 0, "violations": []}
    for b in complete_blocks:
        L, S = by[(b, "long")], by[(b, "short")]
        for rid in sorted(S):
            lo, sh = L[rid], S[rid]
            prefix["checked"] += 1
            # spec 9.4: equality when the cap never bound, prefix otherwise
            if lo["tokens"] <= SHORT_CAP:
                if sh["completion"] == lo["completion"]:
                    prefix["equal_ok"] += 1
                else:
                    prefix["violations"].append(
                        {"block": b, "root": rid, "kind": "should_be_equal",
                         "long_tokens": lo["tokens"]})
            else:
                if lo["completion"].startswith(sh["completion"]):
                    prefix["prefix_ok"] += 1
                else:
                    prefix["violations"].append(
                        {"block": b, "root": rid, "kind": "not_a_prefix",
                         "long_tokens": lo["tokens"], "short_tokens": sh["tokens"]})
    print(f"      {prefix['checked']} matched pairs: {prefix['prefix_ok']} prefix, "
          f"{prefix['equal_ok']} equal, {len(prefix['violations'])} violations")
    if prefix["violations"]:
        print("      VIOLATIONS ARE A FINDING, not just an error: the byte-prefix "
              "property underpins the cap-schedule result elsewhere in W2.")

    print("\n[3/5] yield, per block, both framings")

    def policy_view(b: str) -> tuple[dict, dict]:
        """Return (long_by_root, targeted_by_root) over all 40 roots."""
        L, S = by[(b, "long")], by[(b, "short")]
        longv = {r: L[r] for r in L}
        targ = {r: (S[r] if r in affected else L[r]) for r in L}
        return longv, targ

    per_block = {}
    for b in complete_blocks:
        longv, targ = policy_view(b)
        roots_all = sorted(longv)
        roots_aff = sorted(affected & set(longv))
        entry = {}
        for name, rs in (("all40", roots_all), ("affected15", roots_aff)):
            x = [targ[r]["verified"] for r in rs]
            y = [longv[r]["verified"] for r in rs]
            x_only = sum(1 for i in range(len(rs)) if x[i] and not y[i])
            y_only = sum(1 for i in range(len(rs)) if y[i] and not x[i])
            diffs = [float(x[i]) - float(y[i]) for i in range(len(rs))]
            lo, hi = st.bootstrap_ci(diffs, reps=BOOT_REPS, seed=BOOT_SEED)
            entry[name] = {
                "n": len(rs),
                "targeted_verified": sum(x), "long_verified": sum(y),
                "x_only": x_only, "y_only": y_only,
                "forced_ties": len(rs) - len(roots_aff) if name == "all40" else 0,
                "effect": round(sum(diffs) / len(rs), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "p_exact_mcnemar": round(mcnemar_exact(x_only, y_only), 6),
            }
        # cost side
        for name, rs in (("all40", roots_all),):
            t_tok = sum(targ[r]["tokens"] for r in rs)
            l_tok = sum(longv[r]["tokens"] for r in rs)
            t_wall = sum(targ[r]["request_s"] for r in rs)
            l_wall = sum(longv[r]["request_s"] for r in rs)
            entry["cost"] = {
                "targeted_tokens": t_tok, "long_tokens": l_tok,
                "tokens_saved": l_tok - t_tok,
                "tokens_saved_pct": round(100 * (l_tok - t_tok) / l_tok, 2) if l_tok else None,
                "targeted_request_s": round(t_wall, 1), "long_request_s": round(l_wall, 1),
                "wall_saved_pct": round(100 * (l_wall - t_wall) / l_wall, 2) if l_wall else None,
                "targeted_calls": sum(targ[r]["calls"] for r in rs),
                "long_calls": sum(longv[r]["calls"] for r in rs),
            }
        states = defaultdict(int)
        for r in roots_all:
            states[targ[r]["state"]] += 1
        entry["targeted_states"] = dict(states)
        states = defaultdict(int)
        for r in roots_all:
            states[longv[r]["state"]] += 1
        entry["long_states"] = dict(states)
        per_block[b] = entry
        a40, a15 = entry["all40"], entry["affected15"]
        print(f"      block {b}: all-40 {a40['targeted_verified']}/40 vs "
              f"{a40['long_verified']}/40 ({100 * a40['effect']:+.1f} pts, "
              f"p={a40['p_exact_mcnemar']:.3f})   affected-15 "
              f"{a15['targeted_verified']}/15 vs {a15['long_verified']}/15")
        print(f"               tokens saved {entry['cost']['tokens_saved_pct']}%")

    print("\n[4/5] energy, measured per branch")
    energy = {"per_block": {}, "coverage_failures": 0, "branches_measured": 0}
    idle_w = None
    try:
        base = nv.integrate(TAG, *[
            v[0] for v in [[0]]][0:0] or [0, 0])  # placeholder, replaced below
    except Exception:
        base = None
    # idle baseline from the markers, if both exist
    mp = ROOT / "audit/telemetry" / f"{TAG}.markers.csv"
    if mp.exists():
        import csv as _csv
        marks = defaultdict(list)
        with mp.open(encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                marks[r["label"]].append(nv.parse_ts(r["iso"]))
        if "idle_start" in marks and "idle_end" in marks:
            ib = nv.integrate(TAG, marks["idle_start"][0], marks["idle_end"][-1])
            idle_w = ib.get("mean_power_w")
            energy["idle_baseline"] = ib
    for b in complete_blocks:
        longv, targ = policy_view(b)
        acc = {"long_j": 0.0, "targeted_j": 0.0, "long_s": 0.0, "targeted_s": 0.0,
               "n_ok": 0, "n_fail": 0}
        for r in sorted(longv):
            for which, src in (("long", longv), ("targeted", targ)):
                e = energy_for(src[r])
                energy["branches_measured"] += 1
                if not e or not e.get("coverage_ok") or e.get("joules") is None:
                    acc["n_fail"] += 1
                    energy["coverage_failures"] += 1
                    continue
                acc["n_ok"] += 1
                acc[f"{which}_j"] += e["joules"]
                acc[f"{which}_s"] += e["interval_seconds"]
        saved = acc["long_j"] - acc["targeted_j"]
        acc["joules_saved"] = round(saved, 1)
        acc["joules_saved_pct"] = (round(100 * saved / acc["long_j"], 2)
                                   if acc["long_j"] else None)
        if idle_w is not None:
            acc["idle_adjusted_saved_j_estimate"] = round(
                saved - idle_w * (acc["long_s"] - acc["targeted_s"]), 1)
        acc["long_j"] = round(acc["long_j"], 1)
        acc["targeted_j"] = round(acc["targeted_j"], 1)
        energy["per_block"][b] = acc
        print(f"      block {b}: long {acc['long_j']} J vs targeted "
              f"{acc['targeted_j']} J  saved {acc['joules_saved_pct']}%  "
              f"(coverage failures {acc['n_fail']})")
    if energy["coverage_failures"]:
        print("      NOTE: some branches lack complete telemetry; per the specification "
              "their energy claim FAILS and is not interpolated.")

    print("\n[5/5] aggregate across blocks -- root-cluster bootstrap, blocks as repeats")
    agg = {}
    if len(complete_blocks) > 1:
        import random
        roots_all = sorted(by[(complete_blocks[0], "long")])
        for name, rs in (("all40", roots_all),
                         ("affected15", sorted(affected & set(roots_all)))):
            per_root = {}
            for r in rs:
                d = []
                for b in complete_blocks:
                    longv, targ = policy_view(b)
                    d.append(float(targ[r]["verified"]) - float(longv[r]["verified"]))
                per_root[r] = statistics.mean(d)
            vals = [per_root[r] for r in rs]
            rng = random.Random(BOOT_SEED)
            reps = []
            n = len(vals)
            for _ in range(BOOT_REPS):
                reps.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
            reps.sort()
            agg[name] = {
                "n_theorems": n, "n_blocks": len(complete_blocks),
                "mean_effect": round(sum(vals) / n, 4),
                "ci95": [round(reps[int(0.025 * BOOT_REPS)], 4),
                         round(reps[int(0.975 * BOOT_REPS)], 4)],
                "note": ("clustered on theorem; blocks are repeated measures on the same "
                         "theorems and are never counted as independent samples"),
            }
            print(f"      {name}: mean effect {100 * agg[name]['mean_effect']:+.2f} pts "
                  f"[{100 * agg[name]['ci95'][0]:+.2f}, {100 * agg[name]['ci95'][1]:+.2f}]"
                  f" over {n} theorems x {len(complete_blocks)} blocks")

    result = {
        "spec": "spec/CAP_POLICY_SPEC.md",
        "complete_blocks": complete_blocks,
        "n_rows_seen": meta["n_rows"],
        "n_incomplete": len(meta["incomplete"]),
        "prefix_gate": prefix,
        "per_block": per_block,
        "aggregate": agg,
        "energy": energy,
        "equivalence_claim": False,
        "reading": ("Prospective seed replication on the existing 40-theorem cohort, not "
                    "validation on new theorems. An interval crossing zero is unresolved, "
                    "never evidence of equality."),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
