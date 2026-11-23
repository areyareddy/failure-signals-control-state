#!/usr/bin/env python3
"""Added S2--S4 control analyses for W2 (CPU only, read-only).

All inputs are opened read-only; the only output is audit/cr_s2s4_controls.json.

  1. S2 branch breakdown by cohort, seed block and arm.
  2. Repetition loops among output-capped S2 generations.  The detector
     is a heuristic and is reported as such.
  3. Four-slot serving check under GPU-time accounting.
     Per-request phase intervals overlap under continuous batching, so summing them
     double-counts GPU time.  We report two bounds on prefill's share of busy GPU time:
       - lower: each request's prefill interval weighted by 1/k(t), k = active requests;
       - upper: the union of all prefill intervals over the regime's busy time (every
         instant at which any request is prefilling is charged entirely to prefill).
     Aggregate decode throughput is total decode tokens over busy time.
  4. A class-independent control for S3/S4: a *global* 8,192-token cap on every root,
     evaluated on executed Long draws through the byte-prefix property that S4 verified
     on all 45 matched pairs.  Tokens and yield only; energy is not inferred.
  5. Whether first-attempt truncation predicts a long retry (S4 Long draws).
  6. A theorem-clustered bootstrap for S4's energy and token savings.
"""
from __future__ import annotations

import collections
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import s55_nvml_logger as nv  # noqa: E402

OUT = ROOT / "audit" / "cr_s2s4_controls.json"
CAP = 8192
LONG = 12288
N_BOOT = 10_000


def ro(name):
    con = sqlite3.connect(f"file:{ROOT / 'data' / name}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def roots_meta():
    return {r["root_id"]: r for r in (json.loads(l) for l in
            (ROOT / "data/roots.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}


# ---------------------------------------------------------------- 1. S2 breakdown
def s2_breakdown():
    con = ro("timing_replay.sqlite")
    rows = list(con.execute(
        "SELECT cohort, seed_block, arm, COUNT(*) n, SUM(finish_reason='length') capped "
        "FROM branch_timings GROUP BY 1,2,3 ORDER BY 1,2,3"))
    out = [dict(r) for r in rows]
    assert sum(r["n"] for r in out) == 536
    return out


# ---------------------------------------------------------------- 2. repetition loops
def is_loop(text: str) -> bool:
    """Heuristic: in the final quarter of the completion, at least half of the
    non-blank lines repeat a line seen earlier in that window, or the tail is a
    short string repeated many times."""
    tail = text[-max(2000, len(text) // 4):]
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if len(lines) >= 8:
        seen, rep = set(), 0
        for l in lines:
            rep += l in seen
            seen.add(l)
        if rep / len(lines) >= 0.5:
            return True
    # character-level period check on the last 600 chars
    t = text[-600:]
    for period in range(4, 200):
        unit = t[-period:]
        if unit.strip() and t.count(unit) * period >= 0.8 * len(t):
            return True
    return False


def s2_loops():
    con = ro("runs.sqlite")
    rows = list(con.execute(
        "SELECT branch_key, cohort, arm, completion, completion_tokens FROM branches "
        "WHERE status='done' AND finish_reason='length'"))
    flags = [is_loop(r["completion"] or "") for r in rows]
    by = collections.Counter((r["cohort"], f) for r, f in zip(rows, flags))
    return {"capped_generations": len(rows), "loop_heuristic_positive": sum(flags),
            "rate": sum(flags) / len(rows) if rows else None,
            "by_cohort": {f"{c}|{'loop' if f else 'no_loop'}": n for (c, f), n in sorted(by.items())},
            "detector": "final-quarter line repetition >= 50% or a repeated unit covering >= 80% of the last 600 chars"}


# ---------------------------------------------------------------- 3. four-slot accounting
def interval_bounds(branches):
    """branches: list of (start, prefill_s, decode_s, end)."""
    events = []
    for s, p, d, e in branches:
        events.append((s, e, s, s + p))
    # breakpoints
    pts = sorted({x for s, e, ps, pe in events for x in (s, e, ps, min(pe, e))})
    busy = lower = upper = 0.0
    for a, b in zip(pts, pts[1:]):
        mid = (a + b) / 2
        active = [ev for ev in events if ev[0] <= mid < ev[1]]
        if not active:
            continue
        w = b - a
        busy += w
        pre = [ev for ev in active if ev[2] <= mid < ev[3]]
        if pre:
            upper += w
            lower += w * len(pre) / len(active)
    return busy, lower, upper


def four_slot():
    con = ro("serving_regime.sqlite")
    rows = [dict(r) for r in con.execute(
        "SELECT arm, started_at, finished_at, prefill_ms, decode_ms, prefill_tokens, decode_tokens "
        "FROM regime_branches WHERE regime='p4'")]
    out = {}
    allb = [(r["started_at"], r["prefill_ms"] / 1000, r["decode_ms"] / 1000, r["finished_at"]) for r in rows]
    for arm in ("A", "C"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        t0 = min(r["started_at"] for r in arm_rows)
        t1 = max(r["finished_at"] for r in arm_rows)
        # concurrency is computed against every request active in the window, including
        # the other arm's requests during the short hand-over overlap
        win = [b for b in allb if b[3] > t0 and b[0] < t1]
        own = [(r["started_at"], r["prefill_ms"] / 1000, r["decode_ms"] / 1000, r["finished_at"]) for r in arm_rows]
        busy, lo_own, up_own = interval_bounds_restricted(win, own)
        npre = sum(r["prefill_tokens"] for r in arm_rows)
        ndec = sum(r["decode_tokens"] for r in arm_rows)
        sum_pre = sum(r["prefill_ms"] for r in arm_rows) / 1000
        sum_dec = sum(r["decode_ms"] for r in arm_rows) / 1000
        out[arm] = {
            "n": len(arm_rows), "window_s": t1 - t0, "busy_s": busy,
            "prefill_tokens": npre, "decode_tokens": ndec, "rho": npre / ndec,
            "per_request_sum_prefill_s": sum_pre, "per_request_sum_decode_s": sum_dec,
            "per_request_prefill_share": sum_pre / (sum_pre + sum_dec),
            "per_request_decode_tok_s": ndec / sum_dec,
            "aggregate_decode_tok_s": ndec / busy,
            "mean_concurrency": (sum_pre + sum_dec) / busy,
            "prefill_share_gpu_time_lower": lo_own / busy,
            "prefill_share_gpu_time_upper": up_own / busy,
        }
    return out


def interval_bounds_restricted(all_win, own):
    """Like interval_bounds, but only `own` requests' prefill counts as prefill;
    concurrency k(t) counts every request active in the window."""
    ev_all = [(s, e) for s, p, d, e in all_win]
    ev_pre = [(s, min(s + p, e)) for s, p, d, e in own]
    t0 = min(s for s, _, _, _ in own)
    t1 = max(e for _, _, _, e in own)
    pts = sorted({x for s, e in ev_all for x in (s, e)} | {x for s, e in ev_pre for x in (s, e)} | {t0, t1})
    busy = lower = upper = 0.0
    for a, b in zip(pts, pts[1:]):
        if b <= t0 or a >= t1:
            continue
        mid = (a + b) / 2
        k = sum(1 for s, e in ev_all if s <= mid < e)
        if not k:
            continue
        w = b - a
        busy += w
        npre = sum(1 for s, e in ev_pre if s <= mid < e)
        if npre:
            upper += w
            lower += w * npre / k
    return busy, lower, upper


# ---------------------------------------------------------------- 4/5/6. S3/S4
def s4_rows():
    plan = json.loads((ROOT / "audit/cap_policy_plan.json").read_text(encoding="utf-8"))
    affected = set(plan["truncated_roots"])
    con = ro("cap_policy.sqlite")
    by = collections.defaultdict(dict)
    for r in con.execute("SELECT root_id, cohort, seed_block, verified, outcome_state, completion_tokens, "
                         "claimed_at, finished_at, status FROM branches"):
        assert r["status"] == "done"
        pol = "short" if "_short_" in r["cohort"] else "long"
        e = nv.integrate("cap_policy", nv.parse_ts(r["claimed_at"].replace("Z", "+00:00")),
                         nv.parse_ts(r["finished_at"].replace("Z", "+00:00")))
        assert e.get("coverage_ok"), r["root_id"]
        by[(r["seed_block"], pol)][r["root_id"]] = {
            "verified": bool(r["verified"]), "state": r["outcome_state"],
            "tokens": r["completion_tokens"] or 0, "joules": e["joules"]}
    return affected, by


def global_cap_from_long(long_rows, affected):
    """Counterfactual global 8,192 cap via the byte-prefix property."""
    res = {"all": {"long_verified": 0, "global_verified": 0, "targeted_verified": 0,
                   "long_tokens": 0, "global_tokens": 0, "targeted_tokens": 0}}
    lost = []
    for rid, r in long_rows.items():
        g_ver = r["verified"] and r["tokens"] <= CAP
        g_tok = min(r["tokens"], CAP)
        t_ver = g_ver if rid in affected else r["verified"]
        t_tok = g_tok if rid in affected else r["tokens"]
        a = res["all"]
        a["long_verified"] += r["verified"]
        a["global_verified"] += g_ver
        a["targeted_verified"] += t_ver
        a["long_tokens"] += r["tokens"]
        a["global_tokens"] += g_tok
        a["targeted_tokens"] += t_tok
        if r["verified"] and not g_ver:
            lost.append({"root": rid, "tokens": r["tokens"], "affected": rid in affected})
    a = res["all"]
    a["global_saving"] = 1 - a["global_tokens"] / a["long_tokens"]
    a["targeted_saving_counterfactual"] = 1 - a["targeted_tokens"] / a["long_tokens"]
    a["lost_by_global_cap"] = lost
    return a


def s3_long_rows(cohort):
    con = ro("runs.sqlite")
    return {r["root_id"]: {"verified": bool(r["verified"]), "tokens": r["completion_tokens"] or 0}
            for r in con.execute("SELECT root_id, verified, completion_tokens FROM branches "
                                 "WHERE cohort=? AND arm='A' AND status='done'", (cohort,))}


def truncation_prediction(by, affected):
    cell = {"affected": [0, 0, 0, 0], "unaffected": [0, 0, 0, 0]}  # >8192, truncated@12288, n, verified
    for (blk, pol), rows in by.items():
        if pol != "long":
            continue
        for rid, r in rows.items():
            c = cell["affected" if rid in affected else "unaffected"]
            c[0] += r["tokens"] > CAP
            c[1] += r["state"] == "truncated"
            c[2] += 1
            c[3] += r["verified"]
    out = {}
    for k, (gt, tr, n, v) in cell.items():
        out[k] = {"n": n, "over_8192": gt, "over_8192_rate": gt / n, "truncated_at_12288": tr,
                  "truncated_rate": tr / n, "verified": v}
    # verified proofs on affected roots: how long are they?
    lens = sorted(r["tokens"] for (b, p), rows in by.items() if p == "long"
                  for rid, r in rows.items() if rid in affected and r["verified"])
    out["affected_verified_long_lengths"] = lens
    out["affected_verified_distinct_roots"] = len({rid for (b, p), rows in by.items() if p == "long"
                                                   for rid, r in rows.items() if rid in affected and r["verified"]})
    return out


def energy_bootstrap(by, affected):
    blocks = sorted({b for b, _ in by})
    roots = sorted(by[(blocks[0], "long")])
    aff = [r for r in roots if r in affected]
    una = [r for r in roots if r not in affected]

    def saving(sample_aff, sample_una, blks, key):
        long_tot = targ_tot = 0.0
        for b in blks:
            L, S = by[(b, "long")], by[(b, "short")]
            for r in sample_aff:
                long_tot += L[r][key]
                targ_tot += S[r][key]
            for r in sample_una:
                long_tot += L[r][key]
                targ_tot += L[r][key]
        return 1 - targ_tot / long_tot

    out = {}
    rng = random.Random(20261122)
    for key in ("joules", "tokens"):
        point = {str(b): saving(aff, una, [b], key) for b in blocks}
        point["pooled"] = saving(aff, una, blocks, key)
        boots = collections.defaultdict(list)
        for _ in range(N_BOOT):
            sa = [rng.choice(aff) for _ in aff]
            su = [rng.choice(una) for _ in una]
            for b in blocks:
                boots[str(b)].append(saving(sa, su, [b], key))
            # pooled: a root's three blocks move together (theorem clusters)
            boots["pooled"].append(saving(sa, su, blocks, key))
        ci = {}
        for k, v in boots.items():
            v.sort()
            ci[k] = [v[int(0.025 * N_BOOT)], v[int(0.975 * N_BOOT) - 1]]
        out[key] = {"point": point, "ci95": ci}
    return out


def main():
    out = {"status": "added control analyses; post hoc and descriptive"}
    out["s2_breakdown"] = s2_breakdown()
    out["s2_loops"] = s2_loops()
    out["four_slot"] = four_slot()
    affected, by = s4_rows()
    out["s4_global_cap"] = {str(b): global_cap_from_long(by[(b, "long")], affected)
                            for b in sorted({b for b, _ in by})}
    meta = roots_meta()
    s3_aff = {r for r, m in meta.items() if m["fine_failure_class"] == "truncated"}
    out["s3_global_cap"] = {
        "eval": global_cap_from_long(s3_long_rows("primary_eval"), s3_aff),
        "dev": global_cap_from_long(s3_long_rows("full_pool_dev"), s3_aff),
    }
    out["s4_truncation_prediction"] = truncation_prediction(by, affected)
    out["s4_bootstrap"] = energy_bootstrap(by, affected)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
