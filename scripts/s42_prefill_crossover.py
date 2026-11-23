"""Step s42 -- where prefill would start to matter, and in which serving regimes.

THE OBJECTION THIS ANSWERS. Section 5.2 measures prefill in one configuration: a single
slot, no continuous batching, no prompt cache, cold. That is arguably the one regime in
which prefill cannot matter, so "prefill is not the tax" reads as a property of the
configuration rather than a finding.

THE DECOMPOSITION. Write P and D for prefill and decode tokens and r_p, r_d for the
measured phase throughputs. The prefill share of serving time is

    S = (P/r_p) / (P/r_p + D/r_d) = rho / (rho + R),
    rho = P/D    the WORKLOAD ratio  (prompt tokens per completion token)
    R   = r_p/r_d the STACK ratio    (prefill throughput per decode throughput)

This is an identity, not a fitted model: substituting the definitions returns
pms/(pms+dms) exactly. It is reported precisely because it is exact -- it does not predict
anything new, it separates the measured share into the part a different WORKLOAD would
change and the part a different STACK would change, which is what makes the claim
transferable off this machine. The script asserts the identity holds to within floating
point, so a future refactor cannot quietly break it.

The crossover follows: S >= s requires rho >= R * s / (1 - s). Prefill becomes half of
serving time only when the prompt:completion ratio reaches the prefill:decode throughput
ratio.

REGIMES. Frozen phase timings come from the replay store; alternative serving regimes come
from s40's side-measurement, which runs the same prompts and seeds under a different slot
and batching geometry. Regimes are compared on r_p, r_d and R -- never on outcomes, which
s40 does not record.

    python scripts\\s42_prefill_crossover.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

TARGET_SHARES = (0.05, 0.10, 0.25, 0.50)


def q(db: str, sql: str, args: tuple = ()) -> list[dict]:
    con = sqlite3.connect("file:%s?mode=ro" % (DATA / db), uri=True)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args)]
    con.close()
    return out


def phase_block(P: float, D: float, pms: float, dms: float, n: int) -> dict:
    """The exact decomposition, with the identity checked rather than trusted."""
    r_p = 1000.0 * P / pms if pms else 0.0
    r_d = 1000.0 * D / dms if dms else 0.0
    share = pms / (pms + dms) if (pms + dms) else 0.0
    rho = P / D if D else float("inf")
    R = r_p / r_d if r_d else float("inf")
    predicted = rho / (rho + R) if (rho + R) else 0.0
    assert abs(predicted - share) < 1e-9, (
        "S = rho/(rho+R) is an identity and must reproduce pms/(pms+dms) exactly; "
        "got %.12f vs %.12f" % (predicted, share))
    return {"n": n, "prefill_tokens": P, "decode_tokens": D,
            "prefill_ms": pms, "decode_ms": dms,
            "prefill_tok_per_s": r_p, "decode_tok_per_s": r_d,
            "rho_workload": rho, "R_stack": R,
            "prefill_serving_share": share,
            "prefill_serving_share_pct": 100.0 * share}


def frozen_regime(arms: tuple, cohort: str) -> dict:
    out = {}
    qs = ",".join("?" * len(arms))
    rows = q("timing_replay.sqlite",
             "SELECT arm, COUNT(*) n, SUM(prefill_tokens) P, SUM(decode_tokens) D, "
             "SUM(prefill_ms) pms, SUM(decode_ms) dms FROM branch_timings "
             "WHERE cohort=? AND arm IN (%s) GROUP BY arm" % qs, (cohort,) + arms)
    for r in rows:
        out[r["arm"]] = phase_block(r["P"], r["D"], r["pms"], r["dms"], r["n"])
    if rows:
        tot = q("timing_replay.sqlite",
                "SELECT COUNT(*) n, SUM(prefill_tokens) P, SUM(decode_tokens) D, "
                "SUM(prefill_ms) pms, SUM(decode_ms) dms FROM branch_timings "
                "WHERE cohort=? AND arm IN (%s)" % qs, (cohort,) + arms)[0]
        out["pooled"] = phase_block(tot["P"], tot["D"], tot["pms"], tot["dms"], tot["n"])
    return out


def measured_regimes(arms: tuple) -> dict:
    db = DATA / "serving_regime.sqlite"
    if not db.exists():
        return {}
    qs = ",".join("?" * len(arms))
    try:
        regs = q("serving_regime.sqlite",
                 "SELECT DISTINCT regime FROM regime_branches WHERE regime NOT LIKE "
                 "'smoke%'")
    except sqlite3.OperationalError:
        return {}
    out = {}
    for rr in regs:
        reg = rr["regime"]
        rows = q("serving_regime.sqlite",
                 "SELECT arm, COUNT(*) n, SUM(prefill_tokens) P, SUM(decode_tokens) D, "
                 "SUM(prefill_ms) pms, SUM(decode_ms) dms FROM regime_branches "
                 "WHERE regime=? AND arm IN (%s) GROUP BY arm" % qs, (reg,) + arms)
        blk = {r["arm"]: phase_block(r["P"], r["D"], r["pms"], r["dms"], r["n"])
               for r in rows if r["n"]}
        tot = q("serving_regime.sqlite",
                "SELECT COUNT(*) n, SUM(prefill_tokens) P, SUM(decode_tokens) D, "
                "SUM(prefill_ms) pms, SUM(decode_ms) dms FROM regime_branches "
                "WHERE regime=? AND arm IN (%s)" % qs, (reg,) + arms)[0]
        if tot["n"]:
            blk["pooled"] = phase_block(tot["P"], tot["D"], tot["pms"], tot["dms"],
                                        tot["n"])
        meta = q("serving_regime.sqlite",
                 "SELECT workers, parallel_slots, n_ctx, started_at, finished_at, "
                 "n_branches FROM regime_runs WHERE regime=?", (reg,))
        out[reg] = {"arms": blk, "run": meta[0] if meta else None,
                    "complete": tot["n"] == 2 * 40 if len(arms) == 2 else None,
                    "determinism_vs_frozen": determinism_vs_frozen(reg, arms)}
    return out


def crossover(R: float) -> dict:
    return {"%.2f" % s: R * s / (1.0 - s) for s in TARGET_SHARES}


def determinism_vs_frozen(regime: str, arms: tuple) -> dict:
    """Does the alternative regime reproduce the frozen trajectories byte for byte?

    It matters for what may be compared. If the completions differ, then rho differs
    between regimes for a reason that has nothing to do with the serving stack, and the
    raw shares S are not comparable -- only R is. s06 predicted this: slot geometry and
    batching both perturb the numerics.
    """
    qs = ",".join("?" * len(arms))
    frozen = {r["branch_key"]: r["completion_sha256"] for r in q(
        "runs.sqlite",
        "SELECT branch_key, completion_sha256 FROM branches WHERE cohort='primary_eval' "
        "AND arm IN (%s) AND status='done'" % qs, arms)}
    got = q("serving_regime.sqlite",
            "SELECT branch_key, completion_sha256 FROM regime_branches WHERE regime=?",
            (regime,))
    n = len(got)
    same = sum(1 for r in got if frozen.get(r["branch_key"]) == r["completion_sha256"])
    return {"n_compared": n, "n_identical": same,
            "all_identical": bool(n) and same == n}


def counterfactual_share(rho_frozen: float, R_regime: float) -> float:
    """The share this regime would show ON THE FROZEN WORKLOAD.

    Because the regimes do not reproduce each other's completions, their measured rho
    differ, and a raw comparison of S would mix a workload change into a stack change.
    Holding rho at the frozen value isolates the stack term -- which is the one thing the
    identity is for."""
    return rho_frozen / (rho_frozen + R_regime)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="audit/prefill_crossover.json")
    ap.add_argument("--cohort", default="primary_eval")
    ap.add_argument("--arms", default="A,C")
    args = ap.parse_args(argv)
    arms = tuple(a.strip() for a in args.arms.split(","))

    frozen = frozen_regime(arms, args.cohort)
    regimes = measured_regimes(arms)

    def show(title: str, blk: dict) -> None:
        print("\n%s" % title)
        print("  %-8s %5s %11s %11s %10s %10s %8s %9s"
              % ("arm", "n", "prefill tok", "decode tok", "r_p tok/s", "r_d tok/s",
                 "R", "share %"))
        for k in sorted(blk):
            b = blk[k]
            print("  %-8s %5d %11.0f %11.0f %10.1f %10.1f %8.1f %9.2f"
                  % (k, b["n"], b["prefill_tokens"], b["decode_tokens"],
                     b["prefill_tok_per_s"], b["decode_tok_per_s"], b["R_stack"],
                     b["prefill_serving_share_pct"]))

    show("FROZEN REGIME (1 slot, no continuous batching, no prompt cache)", frozen)
    for reg, d in sorted(regimes.items()):
        run = d["run"] or {}
        show("REGIME %s (workers=%s, slots=%s, per-request n_ctx=%s)"
             % (reg, run.get("workers"), run.get("parallel_slots"), run.get("n_ctx")),
             d["arms"])

    print("\nIDENTITY  S = rho/(rho+R)  -- checked exactly for every block above")

    for reg, d in sorted(regimes.items()):
        det = d["determinism_vs_frozen"]
        print("\nREGIME %s vs frozen: %d/%d completions byte-identical%s"
              % (reg, det["n_identical"], det["n_compared"],
                 "" if det["all_identical"] else
                 "  -> rho differs for non-stack reasons; compare R, not S"))
        for arm in ("A", "C"):
            if arm in d["arms"] and arm in frozen:
                rho_f = frozen[arm]["rho_workload"]
                R_r = d["arms"][arm]["R_stack"]
                cf = counterfactual_share(rho_f, R_r)
                print("  arm %s: frozen R=%6.1f -> %5.2f%% ; %s R=%6.1f -> %5.2f%% "
                      "on the frozen workload (rho=%.2f held fixed)"
                      % (arm, frozen[arm]["R_stack"],
                         frozen[arm]["prefill_serving_share_pct"], reg, R_r,
                         100 * cf, rho_f))
                d["arms"][arm]["counterfactual_share_at_frozen_rho_pct"] = 100 * cf

    if "C" in frozen:
        R = frozen["C"]["R_stack"]
        rho = frozen["C"]["rho_workload"]
        print("\nCROSSOVER on the frozen stack (R = %.1f, measured on the long-prompt "
              "arm C)" % R)
        print("  the workload's prompt:completion ratio is rho = %.2f" % rho)
        for s in TARGET_SHARES:
            need = R * s / (1.0 - s)
            print("    prefill share %3.0f%% requires rho >= %7.1f  (%.0fx the observed "
                  "rho)" % (100 * s, need, need / rho))

    result = {
        "identity": "S = rho/(rho+R); rho = prefill/decode tokens, R = r_prefill/r_decode",
        "identity_is_exact": True,
        "cohort": args.cohort,
        "arms": list(arms),
        "frozen_regime": frozen,
        "measured_regimes": regimes,
        "target_shares": list(TARGET_SHARES),
        "crossover_rho_frozen_armC": crossover(frozen["C"]["R_stack"]) if "C" in frozen
        else None,
        "interpretation": {
            "not_a_model": (
                "The decomposition is an algebraic identity, not a fitted or predictive "
                "model. Its use is to separate the workload term from the stack term so "
                "the measured share can be restated for a different deployment."),
            "regime_scope": (
                "R is a property of the serving stack and changes with slot count, "
                "batching, caching and hardware. Each regime is measured, not assumed."),
        },
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
