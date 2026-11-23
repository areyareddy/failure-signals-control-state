"""Step s41 -- exact budget-matched portfolio frontier over the allocation draws.

THE QUESTION. Section 5.5 reports that the three allocation policies verify partly
disjoint root sets -- 13/40 by some policy against 6/40 by all three -- and then concedes
that no portfolio policy was evaluated. This step removes that concession.

WHY IT NEEDS NO NEW GENERATION. Per ALLOCATION_SPEC section 3, draw j of a policy uses
base_seed + (j-1). So the executed draws occupy a seed lattice:

    seed+0   executed at caps 12288 (Long), 6144 (Split-2 d1), 4096 (Split-3 d1)
    seed+1   executed at caps        6144 (Split-2 d2), 4096 (Split-3 d2)
    seed+2   executed at cap                            4096 (Split-3 d3)

Greedy decoding at a fixed seed makes a shorter draw a byte-exact PREFIX of the longer
draw at the same seed. This script CHECKS that property rather than assuming it. Where it
holds, the outcome of a seed under ANY cap c is determined exactly: the trajectory either
finished within c tokens, in which case its recorded outcome stands, or it did not, in
which case it is truncated and reaches no verifier. Every cap schedule inside the executed
range is therefore computable exactly, with zero new draws.

WHAT IS AND IS NOT CLAIMED. Yield is a step function of each cap, changing only when a cap
crosses the completion length of a verified draw. Enumerating caps at those breakpoints is
therefore exhaustive over the CONTINUOUS schedule space, not a sample of it -- which makes
the maximum a CEILING, established by arithmetic rather than by inference. That distinction
matters at n=40: a ceiling has no minimum detectable effect and no multiplicity, because
nothing is being tested.

The schedule that attains the ceiling is a different matter. It is selected by searching
against the evaluation outcomes, so it is POST HOC and is reported as an upper bound on
what cap scheduling could buy -- never as a recommended policy. Presenting it as one would
repeat exactly the outcome-dependent selection that already cost this study its allocation
router (ALLOCATION_SPEC section 7.1).

    python scripts\\s41_portfolio_frontier.py
    python scripts\\s41_portfolio_frontier.py --json audit/portfolio_frontier.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

BUDGET = 12288                      # the frozen per-root nominal cap budget
MAXCAP = {0: 12288, 1: 6144, 2: 4096}   # largest executed cap at each seed offset
ROUND_STEP = 1024                   # granularity a cap schedule could be pre-registered on


def rows(db: str, sql: str, args: tuple = ()) -> list[dict]:
    con = sqlite3.connect("file:%s?mode=ro" % (DATA / db), uri=True)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args)]
    con.close()
    return out


def load_seed_draws() -> dict:
    """The longest executed draw at each seed offset, keyed by root."""
    q = ("SELECT root_id, seed, completion, completion_tokens, verified, outcome_state "
         "FROM branches WHERE cohort=? AND status='done'")
    seed0 = {r["root_id"]: r for r in rows(
        "runs.sqlite", q.replace("cohort=?", "cohort='primary_eval' AND arm='A'"), ())}
    seed1 = {r["root_id"]: r for r in rows(
        "alloc_split2.sqlite", q + " AND draw_index=2", ("alloc_split2_eval",))}
    seed2 = {r["root_id"]: r for r in rows(
        "alloc_split3.sqlite", q + " AND draw_index=3", ("alloc_split3_eval",))}
    return {0: seed0, 1: seed1, 2: seed2}


def load_short_draws() -> dict:
    """The shorter executed draws, used only to CHECK the prefix property."""
    q = ("SELECT root_id, seed, completion, completion_tokens, verified, outcome_state "
         "FROM branches WHERE cohort=? AND status='done' AND draw_index=?")
    return {
        ("s2", 1): {r["root_id"]: r for r in rows("alloc_split2.sqlite", q,
                                                  ("alloc_split2_eval", 1))},
        ("s3", 1): {r["root_id"]: r for r in rows("alloc_split3.sqlite", q,
                                                  ("alloc_split3_eval", 1))},
        ("s3", 2): {r["root_id"]: r for r in rows("alloc_split3.sqlite", q,
                                                  ("alloc_split3_eval", 2))},
    }


def check_lattice(seed: dict, short: dict, roots: list[str]) -> dict:
    """The enabling precondition: shorter draws are byte-exact prefixes at equal seed."""
    checks = []

    def one(name, a, b, note):
        n_seed = n_prefix = n_pairs = 0
        for r in roots:
            if r not in a or r not in b:
                continue
            n_pairs += 1
            if a[r]["seed"] == b[r]["seed"]:
                n_seed += 1
            sa, sb = a[r]["completion"] or "", b[r]["completion"] or ""
            if sb.startswith(sa):
                n_prefix += 1
        checks.append({"pair": name, "note": note, "n": n_pairs,
                       "seeds_equal": n_seed, "exact_prefix": n_prefix,
                       "ok": n_pairs > 0 and n_seed == n_pairs == n_prefix})

    one("split3_d1_in_split2_d1", short[("s3", 1)], short[("s2", 1)],
        "seed+0, cap 4096 inside cap 6144")
    one("split2_d1_in_long", short[("s2", 1)], seed[0],
        "seed+0, cap 6144 inside cap 12288")
    one("split3_d2_in_split2_d2", short[("s3", 2)], seed[1],
        "seed+1, cap 4096 inside cap 6144")

    offsets = []
    for name, d, off in (("split2_d2", seed[1], 1), ("split3_d3", seed[2], 2)):
        n = sum(1 for r in roots
                if r in d and r in seed[0] and d[r]["seed"] == seed[0][r]["seed"] + off)
        offsets.append({"draw": name, "expected_offset": off,
                        "n_matching": n, "of": len(roots), "ok": n == len(roots)})

    return {"prefix_checks": checks, "seed_offsets": offsets,
            "ok": all(c["ok"] for c in checks) and all(o["ok"] for o in offsets)}


def build_need(seed: dict, roots: list[str]) -> dict:
    """need[o][root] = tokens at which seed offset o verifies that root, if it ever does.

    A draw recorded as truncated never finished, so no shorter cap finishes either; it is
    absent from need at every cap. A draw that finished but failed Lean is likewise absent,
    because shortening the cap cannot turn a rejected proof into an accepted one."""
    need = {}
    for o in (0, 1, 2):
        d = {}
        for r in roots:
            rec = seed[o].get(r)
            if rec and rec["outcome_state"] != "truncated" and rec["verified"]:
                d[r] = rec["completion_tokens"]
        need[o] = d
    return need


def yield_of(need: dict, caps: tuple) -> int:
    hit = set()
    for o in (0, 1, 2):
        if caps[o] > 0:
            hit |= {r for r, t in need[o].items() if t <= caps[o]}
    return len(hit)


def sweep(need: dict, breakpoints: dict, budget: int) -> dict:
    """Exhaustive over the given cap sets, subject to the nominal budget."""
    best_at_yield = {}
    n = 0
    for c0 in breakpoints[0]:
        for c1 in breakpoints[1]:
            for c2 in breakpoints[2]:
                spend = c0 + c1 + c2
                if spend > budget or spend == 0:
                    continue
                n += 1
                y = yield_of(need, (c0, c1, c2))
                if y not in best_at_yield or spend < best_at_yield[y]["nominal"]:
                    best_at_yield[y] = {"caps": [c0, c1, c2], "nominal": spend}
    return {"n_schedules": n, "best_at_yield": best_at_yield}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="audit/portfolio_frontier.json")
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args(argv)

    seed = load_seed_draws()
    short = load_short_draws()
    roots = sorted(seed[0])
    n_roots = len(roots)
    print("roots: %d" % n_roots)

    lattice = check_lattice(seed, short, roots)
    print("\nSEED LATTICE (the enabling precondition)")
    for c in lattice["prefix_checks"]:
        print("  %-26s seeds %2d/%-2d  exact prefix %2d/%-2d  %s   [%s]"
              % (c["pair"], c["seeds_equal"], c["n"], c["exact_prefix"], c["n"],
                 "OK" if c["ok"] else "FAIL", c["note"]))
    for o in lattice["seed_offsets"]:
        print("  %-26s seed == base + %d : %2d/%d  %s"
              % (o["draw"], o["expected_offset"], o["n_matching"], o["of"],
                 "OK" if o["ok"] else "FAIL"))
    if not lattice["ok"]:
        print("\nABORT: the prefix property does not hold; cap schedules are NOT "
              "simulable from the executed draws and nothing below would be valid.")
        return 2

    need = build_need(seed, roots)

    # ---- validity: the simulator must reproduce the three executed policies exactly ----
    executed = {
        "long": {"caps": (12288, 0, 0)},
        "split2": {"caps": (6144, 6144, 0)},
        "split3": {"caps": (4096, 4096, 4096)},
    }
    recorded = {
        "long": sum(1 for r in roots
                    if seed[0][r]["verified"] and seed[0][r]["outcome_state"] != "truncated"),
        "split2": None, "split3": None,
    }
    # recorded split counts come straight from the executed cohorts, not from the lattice
    for pol, db, cohort, ndraw in (("split2", "alloc_split2.sqlite", "alloc_split2_eval", 2),
                                   ("split3", "alloc_split3.sqlite", "alloc_split3_eval", 3)):
        got = rows(db, "SELECT root_id, MAX(verified) v FROM branches WHERE cohort=? "
                       "AND status='done' GROUP BY root_id", (cohort,))
        recorded[pol] = sum(1 for r in got if r["v"])

    print("\nVALIDITY -- simulator against the recorded cohorts")
    validity = []
    ok = True
    for pol, d in executed.items():
        got = yield_of(need, d["caps"])
        match = got == recorded[pol]
        ok &= match
        validity.append({"policy": pol, "caps": list(d["caps"]),
                         "simulated": got, "recorded": recorded[pol], "ok": match})
        print("  %-8s caps=%-22s simulated %2d/%d   recorded %2d   %s"
              % (pol, str(d["caps"]), got, n_roots, recorded[pol],
                 "OK" if match else "MISMATCH"))
    if not ok:
        print("\nABORT: the simulator disagrees with an executed cohort.")
        return 2

    union_caps = (MAXCAP[0], MAXCAP[1], MAXCAP[2])
    union = yield_of(need, union_caps)
    print("  %-8s caps=%-22s simulated %2d/%d   (all executed draws; nominal %d)"
          % ("union", str(union_caps), union, n_roots, sum(union_caps)))

    # ---- the two exhaustive sweeps ----
    oracle_bp = {o: [0] + sorted(set(need[o].values())) for o in (0, 1, 2)}
    round_bp = {o: [0] + [ROUND_STEP * i
                          for i in range(1, MAXCAP[o] // ROUND_STEP + 1)]
                for o in (0, 1, 2)}

    oracle = sweep(need, oracle_bp, args.budget)
    rnd = sweep(need, round_bp, args.budget)
    # unbounded budget, oracle caps: what the union actually costs
    unbounded = sweep(need, oracle_bp, sum(MAXCAP.values()))

    ceiling = max(oracle["best_at_yield"])
    round_best = max(rnd["best_at_yield"])
    pre_registered_best = max(recorded.values())

    print("\nEXHAUSTIVE SWEEPS at nominal budget %d" % args.budget)
    print("  oracle caps (breakpoints) : %4d schedules -> ceiling %2d/%d  caps=%s "
          "(nominal %d)"
          % (oracle["n_schedules"], ceiling, n_roots,
             oracle["best_at_yield"][ceiling]["caps"],
             oracle["best_at_yield"][ceiling]["nominal"]))
    print("  round caps (%d-multiples) : %4d schedules -> best    %2d/%d  caps=%s "
          "(nominal %d)"
          % (ROUND_STEP, rnd["n_schedules"], round_best, n_roots,
             rnd["best_at_yield"][round_best]["caps"],
             rnd["best_at_yield"][round_best]["nominal"]))
    print("  best pre-registered policy: %2d/%d (Split-2)" % (pre_registered_best, n_roots))
    print("  headroom over pre-registered: %+d root(s) = %+.1f points"
          % (ceiling - pre_registered_best,
             100.0 * (ceiling - pre_registered_best) / n_roots))

    # how many round schedules attain the maximum -- a knife-edge is worth disclosing
    attain = [list(k) for k in
              (kk for kk in _all_round(need, round_bp, args.budget, round_best))]
    print("  round schedules attaining the maximum: %d of %d"
          % (len(attain), rnd["n_schedules"]))
    for k in attain:
        print("      caps=%s (nominal %d)" % (k, sum(k)))

    # yield by number of draws actually used
    by_draws = {}
    for c0 in round_bp[0]:
        for c1 in round_bp[1]:
            for c2 in round_bp[2]:
                caps = (c0, c1, c2)
                spend = sum(caps)
                if spend > args.budget or spend == 0:
                    continue
                nd = sum(1 for x in caps if x > 0)
                y = yield_of(need, caps)
                if nd not in by_draws or y > by_draws[nd]["yield"]:
                    by_draws[nd] = {"yield": y, "caps": list(caps)}
    print("\n  best round-cap yield by number of draws used")
    for nd in sorted(by_draws):
        print("    %d draw(s): %2d/%d  caps=%s"
              % (nd, by_draws[nd]["yield"], n_roots, by_draws[nd]["caps"]))

    # price of the union
    union_min = unbounded["best_at_yield"].get(union)
    print("\n  the %d/%d union costs %d nominal tokens = %.3fx the budget"
          % (union, n_roots, union_min["nominal"],
             union_min["nominal"] / float(args.budget)))

    print("\n  round-cap yield distribution")
    dist = Counter()
    for c0 in round_bp[0]:
        for c1 in round_bp[1]:
            for c2 in round_bp[2]:
                if 0 < sum((c0, c1, c2)) <= args.budget:
                    dist[yield_of(need, (c0, c1, c2))] += 1
    for y, n in sorted(dist.items()):
        print("    %2d/%d : %3d schedule(s)" % (y, n_roots, n))

    result = {
        "n_roots": n_roots,
        "budget_nominal_tokens": args.budget,
        "max_executed_cap_per_seed_offset": MAXCAP,
        "round_step": ROUND_STEP,
        "seed_lattice": lattice,
        "validity": validity,
        "recorded_verified": recorded,
        "union": {"verified": union, "caps": list(union_caps),
                  "nominal_all_executed_draws": sum(union_caps),
                  "min_nominal_tokens": union_min["nominal"],
                  "min_caps": union_min["caps"],
                  "budget_multiple": union_min["nominal"] / float(args.budget)},
        "ceiling": {"verified": ceiling,
                    "caps": oracle["best_at_yield"][ceiling]["caps"],
                    "nominal": oracle["best_at_yield"][ceiling]["nominal"],
                    "n_schedules_searched": oracle["n_schedules"]},
        "round_best": {"verified": round_best,
                       "caps": rnd["best_at_yield"][round_best]["caps"],
                       "nominal": rnd["best_at_yield"][round_best]["nominal"],
                       "n_schedules_searched": rnd["n_schedules"],
                       "n_attaining": len(attain),
                       "attaining_caps": attain},
        "best_pre_registered": pre_registered_best,
        "headroom_roots": ceiling - pre_registered_best,
        "headroom_points": 100.0 * (ceiling - pre_registered_best) / n_roots,
        "by_draw_count": by_draws,
        "round_yield_distribution": {str(k): v for k, v in sorted(dist.items())},
        "frontier_oracle": {str(y): v for y, v in
                            sorted(unbounded["best_at_yield"].items())},
        "interpretation": {
            "ceiling_is_arithmetic": (
                "The maximum is exhaustive over the continuous cap space, because yield "
                "changes only at verified-completion breakpoints. It is not an estimate: "
                "it has no minimum detectable effect and no multiplicity correction, "
                "because no hypothesis is tested."),
            "attaining_schedule_is_post_hoc": (
                "The schedule attaining the maximum was selected against the evaluation "
                "outcomes. It is an upper bound on what cap scheduling could buy, not a "
                "recommended policy, and no p-value is computed for it."),
            "reachable_space": (
                "Seed+1 is executed only to 6144 tokens and seed+2 only to 4096, so "
                "schedules requiring longer draws at those offsets are outside the "
                "reachable space and are not evaluated."),
        },
    }

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


def _all_round(need, bp, budget, target):
    for c0 in bp[0]:
        for c1 in bp[1]:
            for c2 in bp[2]:
                caps = (c0, c1, c2)
                if 0 < sum(caps) <= budget and yield_of(need, caps) == target:
                    yield caps


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
