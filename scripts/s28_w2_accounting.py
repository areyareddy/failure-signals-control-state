"""Step s28 -- W2 resource accounting. CPU only; reads both databases read-only.

    python scripts\\s28_w2_accounting.py

TWO PASSES OVER ONE DETERMINISTIC ARTIFACT. Serving phase times come from the timing
replay (data/timing_replay.sqlite); verifier wall time comes from the original run
(data/runs.sqlite). Those were separate passes. They are combinable only because the replay
regenerated every completion byte-for-byte under the frozen serving policy -- 536/536 hash
matches, zero cache reuse, and zero token disagreement with the original run on all four
token columns. This script re-asserts all of that before reporting a single number, and
refuses to continue otherwise. The paper must disclose the two-pass construction.

WHY THE PLANNED FRAMING DOES NOT SURVIVE. The W2 plan is titled around a "hidden prefill
cost". Prefill is not hidden and it is not the cost: it is 0.12-2.04% of measured serving
time in every arm, and carrying a full prior attempt buys thousands of prefill tokens for
well under a second.

It is not a saving either, and an earlier version of this file said it was. That claim --
"prior-attempt arms are cheaper per call" -- does not replicate. Paired C-A measured
serving time is negative in block 1, POSITIVE in block 2, and negative on dev-43, and the
block-1 interval crosses zero. No per-call direction is asserted anywhere below; the
`per_call_direction` block records the sign, the interval, and whether it is resolved for
every cohort, so the non-replication stays on the record instead of being narrated away.

WHAT SURVIVES is phase-resolved and outcome-aware accounting: cost per call is flat while
success collapses, and over half of all measured compute is discarded before the verifier
ever runs. Cost-per-success is therefore a first-class currency here, with a root bootstrap
of the RATIO and explicit zero-success accounting -- an arm that verifies nothing has an
infinite cost per success, and a resample that verifies nothing has an infinite one too.
Those infinities are carried through the percentiles rather than dropped, because dropping
them would silently report a finite interval for an arm that may well be unboundedly
expensive.

CURRENCIES (per the action plan's required accounting table):
    model calls, prefill tokens, decode tokens, completion tokens, total tokens,
    prefill ms, decode ms, measured serving time, replay request wall, original request
    wall, verifier wall, serving+verification, request-wall+verification, and cost per
    verified theorem.

Request wall: the REPLAY wall is primary, because it belongs to the same pass as the phase
times it sits beside. The original wall is retained as a provenance column only.

ARM E is excluded from the main A-D economic table. E ran on its own eligibility subset
(21 of 40 roots in block 1, 23 of 43 on dev), so an E row printed beside an A-D row
computed on all roots compares two different sets of theorems. E appears only in the
`e_matched` block, where A-D are recomputed on exactly E's roots.

Blocks are reported separately before any pooled summary. The 43-root development cohort is
descriptive. All contrasts are paired within root with root-bootstrap intervals, matching
W1's inference throughout.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import receipts  # noqa: E402
import stats as st  # noqa: E402

RUNS = ROOT / "data/runs.sqlite"
REPLAY = ROOT / "data/timing_replay.sqlite"
OUT_JSON = ROOT / "audit/w2_accounting.json"
OUT_CSV = ROOT / "audit/w2_accounting_rows.csv"

COHORTS = ("primary_eval", "eval_block2", "full_pool_dev")
LABEL = {"primary_eval": "Block 1", "eval_block2": "Block 2",
         "full_pool_dev": "Dev-43"}

MAIN_ARMS = "ABCD"          # the economic table
BOOT_REPS = 20000           # matches every other interval in the paper
BOOT_SEED = 20260828


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{RUNS}?mode=ro", uri=True)
    con.execute("ATTACH DATABASE ? AS T", (f"file:{REPLAY}?mode=ro",))
    return con


def integrity(con: sqlite3.Connection) -> dict:
    """Refuse to report accounting unless the two passes provably describe one artifact."""
    g = lambda q: con.execute(q).fetchone()[0]
    run = con.execute("SELECT replay_run_id, status, finished_utc FROM T.replay_runs"
                      ).fetchone()
    checks = {
        "replay_status": run[1],
        "runs_done": g("SELECT count(*) FROM branches WHERE status='done'"),
        "replay_rows": g("SELECT count(*) FROM T.branch_timings"),
        "joined": g("SELECT count(*) FROM branches b JOIN T.branch_timings t "
                    "ON b.branch_key=t.branch_key WHERE b.status='done'"),
        "hash_mismatch": g("SELECT count(*) FROM T.branch_timings WHERE hash_match!=1"),
        "cache_reuse": g("SELECT count(*) FROM T.branch_timings WHERE cached_tokens>0"),
        "nonpositive_prefill": g("SELECT count(*) FROM T.branch_timings WHERE prefill_ms<=0"),
        "nonpositive_decode": g("SELECT count(*) FROM T.branch_timings WHERE decode_ms<=0"),
        "nonpositive_replay_wall": g("SELECT count(*) FROM T.branch_timings "
                                     "WHERE request_wall_seconds<=0"),
        "segments": g("SELECT count(*) FROM T.replay_segments"),
        "canary_rows_quarantined": g("SELECT count(*) FROM T.canary_timings"),
    }
    for col in ("prefill_tokens", "decode_tokens", "completion_tokens", "total_tokens"):
        checks[f"token_mismatch_{col}"] = g(
            f"SELECT count(*) FROM branches b JOIN T.branch_timings t "
            f"ON b.branch_key=t.branch_key WHERE b.status='done' AND b.{col}!=t.{col}")
    # every analytical row must belong to a segment that exists
    checks["orphan_segment_rows"] = g(
        "SELECT count(*) FROM T.branch_timings t LEFT JOIN T.replay_segments s "
        "ON t.segment_id=s.segment_id WHERE s.segment_id IS NULL")
    # completion vs decode: the action plan asks for both. Under this serving policy they
    # are the SAME column -- assert it, so no later sentence can treat them as two
    # independent pieces of evidence.
    checks["completion_ne_decode"] = g(
        "SELECT count(*) FROM T.branch_timings WHERE completion_tokens!=decode_tokens")
    checks["prefill_plus_completion_ne_total"] = g(
        "SELECT count(*) FROM T.branch_timings "
        "WHERE prefill_tokens+completion_tokens!=total_tokens")

    print("[1/8] integrity of the two-pass join")
    for k, v in checks.items():
        print(f"      {k:32} {v}")
    assert checks["replay_status"] == "complete", "replay is not complete"
    assert checks["runs_done"] == checks["replay_rows"] == checks["joined"], \
        "replay and run do not cover the same branches"
    for k in ("hash_mismatch", "cache_reuse", "nonpositive_prefill", "nonpositive_decode",
              "nonpositive_replay_wall", "orphan_segment_rows",
              "completion_ne_decode", "prefill_plus_completion_ne_total"):
        assert checks[k] == 0, f"{k} = {checks[k]}"
    for col in ("prefill_tokens", "decode_tokens", "completion_tokens", "total_tokens"):
        assert checks[f"token_mismatch_{col}"] == 0, f"{col} disagrees between passes"
    print("      -> the replay reproduced the original token counts exactly; the two")
    print("         passes describe one artifact and may be combined.")
    print("      -> completion_tokens == decode_tokens on every row; they are reported")
    print("         as separate currencies but carry the same information.")
    checks["replay_run_id"] = run[0]
    checks["replay_finished_utc"] = run[2]
    return checks


def load(con: sqlite3.Connection) -> list[dict]:
    rows = []
    for r in con.execute(
            "SELECT b.cohort, b.arm, b.root_id, b.seed_block, b.verified, b.outcome_state, "
            "       b.model_calls, b.verifier_wall_seconds, b.request_wall_seconds, "
            "       t.prefill_tokens, t.decode_tokens, t.completion_tokens, t.total_tokens, "
            "       t.prefill_ms, t.decode_ms, t.request_wall_seconds "
            "FROM branches b JOIN T.branch_timings t ON b.branch_key=t.branch_key "
            "WHERE b.status='done'"):
        d = dict(zip(
            ("cohort", "arm", "root_id", "seed_block", "verified", "outcome_state",
             "model_calls", "verifier_s", "orig_wall_s", "prefill_tok", "decode_tok",
             "completion_tok", "total_tok", "prefill_ms", "decode_ms", "replay_wall_s"), r))
        d["serving_s"] = (d["prefill_ms"] + d["decode_ms"]) / 1000.0
        d["total_s"] = d["serving_s"] + d["verifier_s"]
        # request wall + verification: the end-to-end cost of one branch as a client sees
        # it, distinct from total_s, which sums only the two measured phases.
        d["wall_total_s"] = d["replay_wall_s"] + d["verifier_s"]
        rows.append(d)
    return rows


# key, printed label, divisor, format
CURRENCIES = [
    ("model_calls", "calls", 1, "{:.2f}"),
    ("prefill_tok", "prefill tok", 1, "{:.0f}"),
    ("decode_tok", "decode tok", 1, "{:.0f}"),
    ("completion_tok", "compl tok", 1, "{:.0f}"),
    ("total_tok", "total tok", 1, "{:.0f}"),
    ("prefill_ms", "prefill s", 1000, "{:.2f}"),
    ("decode_ms", "decode s", 1000, "{:.1f}"),
    ("serving_s", "serving s", 1, "{:.1f}"),
    ("replay_wall_s", "req wall s", 1, "{:.1f}"),
    ("orig_wall_s", "orig wall s", 1, "{:.1f}"),
    ("verifier_s", "verifier s", 1, "{:.1f}"),
    ("total_s", "serve+ver s", 1, "{:.1f}"),
    ("wall_total_s", "wall+ver s", 1, "{:.1f}"),
]

# printed in the narrow per-arm table; the JSON carries all of CURRENCIES
NARROW = ("prefill_tok", "decode_tok", "total_tok", "prefill_ms", "decode_ms",
          "serving_s", "replay_wall_s", "verifier_s", "total_s", "wall_total_s")


def ratio_bootstrap(pairs: list[tuple[float, int]], reps: int = BOOT_REPS,
                    alpha: float = 0.05, seed: int = BOOT_SEED) -> dict:
    """Percentile CI for a RATIO of sums -- cost per verified theorem -- resampling ROOTS.

    `pairs` is one (cost, verified) per root. Cost per success is sum(cost)/sum(verified),
    a ratio of two sums over the same resample, NOT a mean of per-root ratios: a root that
    verifies nothing has no per-root ratio at all.

    Zero-success accounting. A resample in which no root verifies has an infinite cost per
    success. That is a real outcome, not a failed computation, so it is carried into the
    sort as +inf rather than discarded. If more than alpha/2 of resamples are zero-success
    the upper limit is genuinely infinite, and the interval says so. `p_zero_success` is
    reported either way, because a finite interval built on 97% finite resamples still
    means something different from one built on 100%.
    """
    n = len(pairs)
    obs_cost = sum(c for c, _ in pairs)
    obs_ok = sum(v for _, v in pairs)
    point = obs_cost / obs_ok if obs_ok else math.inf
    if n == 0:
        return {"point": None, "ci95": None, "p_zero_success": None, "reps": 0}
    rng = random.Random(seed)
    vals, zeros = [], 0
    for _ in range(reps):
        c = k = 0.0
        for _ in range(n):
            cc, vv = pairs[rng.randrange(n)]
            c += cc
            k += vv
        if k == 0:
            zeros += 1
            vals.append(math.inf)
        else:
            vals.append(c / k)
    vals.sort()
    lo = vals[int((alpha / 2) * reps)]
    hi = vals[min(reps - 1, int((1 - alpha / 2) * reps))]
    j = lambda x: None if math.isinf(x) else round(x, 1)
    return {"point": j(point) if math.isfinite(point) else None,
            "point_infinite": not math.isfinite(point),
            "ci95": [j(lo), j(hi)],
            "ci95_upper_infinite": math.isinf(hi),
            "ci95_lower_infinite": math.isinf(lo),
            "p_zero_success": round(zeros / reps, 4),
            "reps": reps}


def fmt_ci(b: dict) -> str:
    if b["point_infinite"]:
        return f"{'inf':>12}  [inf, inf]"
    lo = "inf" if b["ci95_lower_infinite"] else f"{b['ci95'][0]:.0f}"
    hi = "inf" if b["ci95_upper_infinite"] else f"{b['ci95'][1]:.0f}"
    return f"{b['point']:12.0f}  [{lo}, {hi}]"


def cost_rows(rows: list[dict], cohort: str, arms: str,
              roots: set[str] | None = None) -> list[dict]:
    """Cost per verified theorem, per arm, with a root bootstrap of the ratio."""
    out = []
    for arm in arms:
        sub = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm
               and (roots is None or r["root_id"] in roots)]
        if not sub:
            continue
        serve = sum(r["serving_s"] for r in sub)
        ver = sum(r["verifier_s"] for r in sub)
        wall = sum(r["replay_wall_s"] for r in sub)
        tot = serve + ver
        ok = sum(r["verified"] for r in sub)
        toks = sum(r["total_tok"] for r in sub)
        sec = ratio_bootstrap([(r["total_s"], r["verified"]) for r in sub])
        tok = ratio_bootstrap([(float(r["total_tok"]), r["verified"]) for r in sub])
        out.append({
            "cohort": cohort, "arm": arm, "n": len(sub),
            "matched_subset": roots is not None,
            "serving_s": round(serve, 1), "verifier_s": round(ver, 1),
            "wall_s": round(wall, 1), "total_s": round(tot, 1), "verified": ok,
            "seconds_per_verified": round(tot / ok, 1) if ok else None,
            "tokens_per_verified": round(toks / ok, 0) if ok else None,
            "seconds_per_verified_boot": sec,
            "tokens_per_verified_boot": tok})
    return out


def main() -> int:
    receipts.force_utf8()
    con = connect()
    checks = integrity(con)
    rows = load(con)
    con.close()

    out = {"integrity": checks, "per_arm": [], "contrasts": [], "cost_per_success": [],
           "failure_state": [], "phase_split": [], "phase_global": {},
           "per_call_direction": [], "e_matched": {}}

    # ---------------------------------------------------------------- per-arm means
    print("\n[2/8] per-arm accounting (means per branch), arms A-D")
    print("      E is excluded here -- it ran on its own eligibility subset; see [7/8].")
    for cohort in COHORTS:
        print(f"\n  {LABEL[cohort]}")
        print("    arm   n " + " ".join(f"{lab:>11}" for k, lab, _, _ in CURRENCIES
                                        if k in NARROW) + f"{'verified':>10}")
        for arm in MAIN_ARMS:
            sub = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
            if not sub:
                continue
            rec = {"cohort": cohort, "arm": arm, "n": len(sub),
                   "verified": sum(r["verified"] for r in sub)}
            cells = []
            for key, lab, div, fmt in CURRENCIES:
                m = statistics.mean(r[key] for r in sub) / div
                rec[key] = round(m, 4)
                if key in NARROW:
                    cells.append(f"{fmt.format(m):>11}")
            out["per_arm"].append(rec)
            print(f"    {arm:3} {len(sub):3d} " + " ".join(cells)
                  + f"{rec['verified']:>10}")

    # ---------------------------------------------------------------- phase split
    print("\n[3/8] where measured serving time actually goes")
    print("    the planned framing assumed prefill; it is under 3% of serving everywhere")
    for cohort in COHORTS:
        for arm in "ABCDE":
            sub = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
            if not sub:
                continue
            pf = sum(r["prefill_ms"] for r in sub)
            dc = sum(r["decode_ms"] for r in sub)
            share = 100.0 * pf / (pf + dc)
            pf_tps = sum(r["prefill_tok"] for r in sub) / (pf / 1000.0)
            dc_tps = sum(r["decode_tok"] for r in sub) / (dc / 1000.0)
            out["phase_split"].append({
                "cohort": cohort, "arm": arm,
                "prefill_share_pct": round(share, 3),
                "prefill_tok_per_s": round(pf_tps, 1),
                "decode_tok_per_s": round(dc_tps, 1),
                "throughput_ratio": round(pf_tps / dc_tps, 1)})
            if cohort == "primary_eval":
                print(f"      {arm}  prefill {share:5.2f}% of serving   "
                      f"{pf_tps:7.0f} tok/s prefill vs {dc_tps:6.1f} tok/s decode "
                      f"({pf_tps / dc_tps:.0f}x)")

    # global / pooled phase accounting -- the aggregate figure the paper quotes
    pf = sum(r["prefill_ms"] for r in rows)
    dc = sum(r["decode_ms"] for r in rows)
    pft = sum(r["prefill_tok"] for r in rows)
    dct = sum(r["decode_tok"] for r in rows)
    pooled_pf_tps = pft / (pf / 1000.0)
    pooled_dc_tps = dct / (dc / 1000.0)
    # steady state = the long-prefill arms, where prefill is not dominated by fixed overhead
    long_arms = [r for r in rows if r["arm"] in "CDE"]
    lpf = sum(r["prefill_ms"] for r in long_arms)
    lpft = sum(r["prefill_tok"] for r in long_arms)
    ldc = sum(r["decode_ms"] for r in long_arms)
    ldct = sum(r["decode_tok"] for r in long_arms)
    ratios = [p["throughput_ratio"] for p in out["phase_split"] if p["arm"] in "CDE"]
    shares = [p["prefill_share_pct"] for p in out["phase_split"]]
    out["phase_global"] = {
        "n_branches": len(rows),
        "prefill_ms_total": round(pf, 1), "decode_ms_total": round(dc, 1),
        "prefill_share_pct": round(100.0 * pf / (pf + dc), 3),
        "prefill_share_pct_min": round(min(shares), 3),
        "prefill_share_pct_max": round(max(shares), 3),
        "prefill_tok_per_s": round(pooled_pf_tps, 1),
        "decode_tok_per_s": round(pooled_dc_tps, 1),
        "throughput_ratio_pooled": round(pooled_pf_tps / pooled_dc_tps, 1),
        "throughput_ratio_steady_state": round(
            (lpft / (lpf / 1000.0)) / (ldct / (ldc / 1000.0)), 1),
        "throughput_ratio_steady_state_min": round(min(ratios), 1),
        "throughput_ratio_steady_state_max": round(max(ratios), 1),
        "note": ("pooled ratio mixes short-prefill arms (A/B), where a ~194-token prefill "
                 "is dominated by fixed per-request overhead, with the long-prefill arms. "
                 "Quote the pooled figure and the steady-state range as two numbers; they "
                 "are not the same measurement."),
    }
    g = out["phase_global"]
    print(f"\n    POOLED over all {g['n_branches']} branches: prefill is "
          f"{g['prefill_share_pct']:.2f}% of serving "
          f"(range {g['prefill_share_pct_min']:.2f}-{g['prefill_share_pct_max']:.2f}% "
          f"by cohort-arm)")
    print(f"    pooled throughput ratio {g['throughput_ratio_pooled']:.0f}x   "
          f"steady state (C/D/E) {g['throughput_ratio_steady_state']:.0f}x "
          f"(arm range {g['throughput_ratio_steady_state_min']:.0f}-"
          f"{g['throughput_ratio_steady_state_max']:.0f}x)")

    # ---------------------------------------------------------------- paired contrasts
    print("\n[4/8] paired within-root contrasts (root bootstrap, 95%)")
    PAIRS = (("C", "A"), ("D", "B"), ("D", "A"), ("B", "A"), ("D", "C"))
    for cohort in COHORTS:
        by = {}
        for r in rows:
            if r["cohort"] == cohort:
                by.setdefault(r["root_id"], {})[r["arm"]] = r
        for x, y in PAIRS:
            shared = [v for v in by.values() if x in v and y in v]
            if len(shared) < 5:
                continue
            for key, lab, div, fmt in CURRENCIES:
                diffs = [(v[x][key] - v[y][key]) / div for v in shared]
                lo, hi = st.bootstrap_ci(diffs)
                out["contrasts"].append({
                    "cohort": cohort, "contrast": f"{x}-{y}", "currency": key,
                    "n": len(diffs), "mean_diff": round(statistics.mean(diffs), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "resolved": (lo > 0) == (hi > 0)})
        if cohort == "primary_eval":
            print(f"\n    {LABEL[cohort]}, C-A (prior attempt, no guidance):")
            for c in out["contrasts"]:
                if (c["cohort"] == cohort and c["contrast"] == "C-A"
                        and c["currency"] in ("prefill_tok", "decode_tok", "prefill_ms",
                                              "decode_ms", "serving_s", "replay_wall_s",
                                              "total_s", "wall_total_s")):
                    lab = dict((k, l) for k, l, _, _ in CURRENCIES)[c["currency"]]
                    flag = "resolved" if c["resolved"] else "UNRESOLVED"
                    print(f"      {lab:12} {c['mean_diff']:+10.2f}  "
                          f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  {flag}")

    # ------------------------------------------- per-call direction: does it replicate?
    print("\n[5/8] is a prior-attempt arm cheaper PER CALL? (the retracted claim)")
    print("    an earlier version of this script asserted yes. It does not replicate.")
    for key in ("serving_s", "replay_wall_s", "total_s"):
        for cohort in COHORTS:
            c = next((c for c in out["contrasts"] if c["cohort"] == cohort
                      and c["contrast"] == "C-A" and c["currency"] == key), None)
            if not c:
                continue
            out["per_call_direction"].append({
                "cohort": cohort, "contrast": "C-A", "currency": key,
                "mean_diff": c["mean_diff"], "ci95": c["ci95"],
                "sign": "negative" if c["mean_diff"] < 0 else "positive",
                "resolved": c["resolved"]})
    signs = {d["sign"] for d in out["per_call_direction"] if d["currency"] == "serving_s"}
    out["per_call_direction_replicates"] = len(signs) == 1
    for d in out["per_call_direction"]:
        print(f"      {LABEL[d['cohort']]:9} {d['currency']:14} C-A "
              f"{d['mean_diff']:+8.2f}  [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  "
              f"{'resolved' if d['resolved'] else 'UNRESOLVED'}")
    assert not out["per_call_direction_replicates"], (
        "C-A per-call sign is now consistent across cohorts; the docstring's retraction "
        "of the 'cheaper per call' claim must be revisited rather than left standing")
    print("      -> sign flips across blocks. No per-call direction is claimed.")

    # ---------------------------------------------------------------- cost per success
    print("\n[6/8] the currency that reverses the ranking: cost per verified theorem")
    print("    ratio bootstrap over roots, 20,000 reps; inf carried through, not dropped")
    for cohort in COHORTS:
        recs = cost_rows(rows, cohort, MAIN_ARMS)
        out["cost_per_success"].extend(recs)
        print(f"\n  {LABEL[cohort]}")
        print(f"    {'arm':4}{'n':>4}{'serve+ver':>11}{'verified':>10}"
              f"{'s/verified':>14}  95% CI{'':>10}{'P(0 ok)':>9}")
        for rec in recs:
            b = rec["seconds_per_verified_boot"]
            print(f"    {rec['arm']:4}{rec['n']:4d}{rec['total_s']:11.0f}"
                  f"{rec['verified']:10d}" + fmt_ci(b)
                  + f"{b['p_zero_success']:>9.3f}")

    # ---------------------------------------------------------------- arm E, matched
    print("\n[7/8] arm E, against A-D recomputed on E's OWN eligibility subset")
    for cohort in COHORTS:
        eroots = {r["root_id"] for r in rows if r["cohort"] == cohort and r["arm"] == "E"}
        if not eroots:
            continue
        recs = cost_rows(rows, cohort, "ABCDE", roots=eroots)
        per_arm = []
        for arm in "ABCDE":
            sub = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm
                   and r["root_id"] in eroots]
            if not sub:
                continue
            rec = {"cohort": cohort, "arm": arm, "n": len(sub),
                   "verified": sum(r["verified"] for r in sub)}
            for key, lab, div, fmt in CURRENCIES:
                rec[key] = round(statistics.mean(r[key] for r in sub) / div, 4)
            per_arm.append(rec)
        out["e_matched"][cohort] = {"n_roots": len(eroots),
                                    "cost_per_success": recs, "per_arm": per_arm}
        print(f"\n  {LABEL[cohort]}, E's {len(eroots)} eligible roots only")
        print(f"    {'arm':4}{'n':>4}{'serve+ver':>11}{'verified':>10}"
              f"{'s/verified':>14}  95% CI")
        for rec in recs:
            print(f"    {rec['arm']:4}{rec['n']:4d}{rec['total_s']:11.0f}"
                  f"{rec['verified']:10d}" + fmt_ci(rec["seconds_per_verified_boot"]))

    # ---------------------------------------------------------------- failure states
    print("\n[8/8] cost decomposition by terminal state")
    print("    truncated branches consume a full decode budget and skip the verifier")
    grand = sum(r["serving_s"] for r in rows)
    print(f"    {'state':14}{'n':>5}{'decode_tok':>12}{'serving_s':>11}"
          f"{'verifier_s':>11}{'total_serve':>13}{'share':>8}")
    for state in ("verified", "lean_failed", "truncated"):
        sub = [r for r in rows if r["outcome_state"] == state]
        if not sub:
            continue
        tot = sum(r["serving_s"] for r in sub)
        rec = {"state": state, "n": len(sub),
               "mean_decode_tok": round(statistics.mean(r["decode_tok"] for r in sub), 0),
               "mean_serving_s": round(statistics.mean(r["serving_s"] for r in sub), 1),
               "mean_verifier_s": round(statistics.mean(r["verifier_s"] for r in sub), 1),
               "total_serving_s": round(tot, 1),
               "share_of_serving_pct": round(100.0 * tot / grand, 1)}
        out["failure_state"].append(rec)
        print(f"    {state:14}{len(sub):5d}{rec['mean_decode_tok']:12.0f}"
              f"{rec['mean_serving_s']:11.1f}{rec['mean_verifier_s']:11.1f}"
              f"{tot:13.0f}{rec['share_of_serving_pct']:7.1f}%")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {OUT_CSV.relative_to(ROOT).as_posix()} and "
          f"{OUT_JSON.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
