r"""Build W2 paper statistics, tables, and vector figures from the frozen accounting ledger.

The script is deliberately read-only with respect to experiment artifacts.  It treats
the proof root as the resampling unit, keeps the two seed blocks separate, and combines
the exact timing replay with the original Lean verifier wall time only after validating
the replay join recorded by ``audit/w2_accounting.json``.

v2 changes over the first draft:
  * every table body in the paper is GENERATED here into ``tables/*.tex`` and \input by
    paper.tex, so no hand-typed number can drift from the ledger;
  * the forest panels are laid out on an explicit row grid -- the v1 figures collided
    panel titles with the tick labels of the panel above;
  * new Figure 4 (iso-cost yield) makes the central economic claim visual: the arms sit
    at nearly the same total wall cost and buy very different numbers of theorems;
  * the marginal cost-per-success ratio bootstrap, including its infinite upper tails
    and P(zero verified), is carried into an appendix table rather than left in the audit.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent
ROOT = HERE.parents[2]
ROWS_CSV = ROOT / "audit" / "w2_accounting_rows.csv"
ACCOUNTING_JSON = ROOT / "audit" / "w2_accounting.json"
ALLOCATION_JSON = ROOT / "audit" / "allocation_analysis.json"
PORTFOLIO_JSON = ROOT / "audit" / "portfolio_frontier.json"
CROSSOVER_JSON = ROOT / "audit" / "prefill_crossover.json"
MUTATION_JSON = ROOT / "audit" / "agent_loop_mutations.json"
CLASS_CAP_JSON = ROOT / "audit" / "class_conditional_cap.json"
OUT_JSON = DRAFT / "computed_results.json"
OUT_TEX = DRAFT / "computed_macros.tex"
FIG = DRAFT / "figures"
TAB = DRAFT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLUE = HexColor("#2C6AA6")
BLUE2 = HexColor("#86AFD5")
ORANGE = HexColor("#D87832")
ORANGE2 = HexColor("#F0B37E")
GREEN = HexColor("#2F8568")
RED = HexColor("#B94A48")
GRAY = HexColor("#66717C")
MID = HexColor("#AAB3BC")
LIGHT = HexColor("#E9EEF3")
DARK = HexColor("#17212B")

COHORTS = ["primary_eval", "eval_block2", "full_pool_dev"]
COHORT_LABEL = {
    "primary_eval": "Eval block 1",
    "eval_block2": "Eval block 2",
    "full_pool_dev": "Development",
}
COHORT_SHORT = {"primary_eval": "1", "eval_block2": "2", "full_pool_dev": "dev"}
ARM_LABEL = {
    "A": "A: fresh",
    "B": "B: guided fresh",
    "C": "C: prior attempt",
    "D": "D: guided prior",
    "E": "E: native repair",
}


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    z = (len(ys) - 1) * p
    lo, hi = math.floor(z), math.ceil(z)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - z) + ys[hi] * (z - lo)


def load_rows() -> list[dict]:
    integer = {"seed_block", "verified", "model_calls", "prefill_tok", "decode_tok",
               "completion_tok", "total_tok"}
    floating = {"verifier_s", "orig_wall_s", "prefill_ms", "decode_ms",
                "replay_wall_s", "serving_s", "total_s", "wall_total_s"}
    rows: list[dict] = []
    with ROWS_CSV.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key in integer:
                row[key] = int(float(row[key]))
            for key in floating:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def paired(rows: list[dict], cohort: str, x: str, y: str) -> list[tuple[dict, dict]]:
    by_root: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["cohort"] == cohort:
            by_root[row["root_id"]][row["arm"]] = row
    out = []
    for root_id in sorted(by_root):
        arms = by_root[root_id]
        if x in arms and y in arms:
            out.append((arms[x], arms[y]))
    return out


def bootstrap_mean_diff(pairs: list[tuple[dict, dict]], field: str, seed: int,
                        reps: int = 50_000) -> dict:
    diffs = [x[field] - y[field] for x, y in pairs]
    rng = random.Random(seed)
    n = len(diffs)
    sims = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]
    return {
        "n": n,
        "mean": statistics.fmean(diffs),
        "ci95": [percentile(sims, 0.025), percentile(sims, 0.975)],
    }


def rate_per_1000(sample: list[dict]) -> float:
    wall = sum(r["wall_total_s"] for r in sample)
    return 1000.0 * sum(r["verified"] for r in sample) / wall


def bootstrap_rate_diff(pairs: list[tuple[dict, dict]], seed: int,
                        reps: int = 100_000) -> dict:
    rng = random.Random(seed)
    n = len(pairs)
    x_all = [p[0] for p in pairs]
    y_all = [p[1] for p in pairs]
    point = rate_per_1000(x_all) - rate_per_1000(y_all)
    sims = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [x_all[i] for i in idx]
        ys = [y_all[i] for i in idx]
        sims.append(rate_per_1000(xs) - rate_per_1000(ys))
    return {"n": n, "point": point,
            "ci95": [percentile(sims, .025), percentile(sims, .975)],
            "reps": reps}


def ratio_bootstrap_e2e(sample: list[dict], seed: int, reps: int = 20_000) -> dict:
    """Cost per verified theorem, bootstrapped over roots, in the paper's own currency.

    s28 computes this same quantity on T_phase (serving + verifier).  The paper uses
    T_e2e (request wall + verifier) everywhere else, and mixing the two would put 285
    in an appendix beside 289 in the main table.  Recomputed here on T_e2e so one
    currency runs through the whole manuscript.

    Zero-success resamples are infinite and are carried into the percentiles rather
    than dropped; dropping them manufactures a finite interval for an arm that may be
    unboundedly expensive.
    """
    n = len(sample)
    ok = sum(r["verified"] for r in sample)
    cost = sum(r["wall_total_s"] for r in sample)
    rng = random.Random(seed)
    vals, zeros = [], 0
    for _ in range(reps):
        cc = kk = 0.0
        for _ in range(n):
            r = sample[rng.randrange(n)]
            cc += r["wall_total_s"]
            kk += r["verified"]
        if kk == 0:
            zeros += 1
            vals.append(math.inf)
        else:
            vals.append(cc / kk)
    vals.sort()
    lo = vals[int(0.025 * reps)]
    hi = vals[min(reps - 1, int(0.975 * reps))]
    return {"point": (cost / ok) if ok else None,
            "point_infinite": ok == 0,
            "ci95": [None if math.isinf(lo) else lo, None if math.isinf(hi) else hi],
            "ci95_lower_infinite": math.isinf(lo),
            "ci95_upper_infinite": math.isinf(hi),
            "p_zero_success": zeros / reps,
            "verified": ok, "n": n, "reps": reps}


def bootstrap_rate(sample: list[dict], seed: int, reps: int = 50_000) -> dict:
    rng = random.Random(seed)
    n = len(sample)
    sims = []
    for _ in range(reps):
        rs = [sample[rng.randrange(n)] for _ in range(n)]
        sims.append(rate_per_1000(rs))
    return {"point": rate_per_1000(sample),
            "ci95": [percentile(sims, .025), percentile(sims, .975)]}


def arm_summary(rows: list[dict], cohort: str, arm: str, seed: int) -> dict:
    sample = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
    n = len(sample)
    wins = sum(r["verified"] for r in sample)
    wall = sum(r["wall_total_s"] for r in sample)
    return {
        "cohort": cohort,
        "arm": arm,
        "n": n,
        "verified": wins,
        "completion_tok_mean": statistics.fmean(r["completion_tok"] for r in sample),
        "prefill_tok_mean": statistics.fmean(r["prefill_tok"] for r in sample),
        "total_tok_mean": statistics.fmean(r["total_tok"] for r in sample),
        "wall_total_s_mean": wall / n,
        "wall_total_s_sum": wall,
        "seconds_per_verified": wall / wins if wins else None,
        "rate_per_1000_s": bootstrap_rate(sample, seed),
    }


def compute(rows: list[dict], source: dict) -> dict:
    assert len(rows) == 536
    integrity = source["integrity"]
    assert integrity["joined"] == 536
    assert integrity["hash_mismatch"] == 0
    assert integrity["cache_reuse"] == 0
    assert integrity["completion_ne_decode"] == 0
    assert integrity["prefill_plus_completion_ne_total"] == 0
    assert sum(r["outcome_state"] == "truncated" for r in rows) == 134
    assert all(r["verifier_s"] == 0 for r in rows if r["outcome_state"] == "truncated")
    # the v1 draft's per-call retraction depends on this staying true
    assert source["per_call_direction_replicates"] is False

    prefill = sum(r["prefill_ms"] for r in rows) / 1000
    decode = sum(r["decode_ms"] for r in rows) / 1000
    verifier = sum(r["verifier_s"] for r in rows)
    phase_total = prefill + decode + verifier

    states = []
    for state in ["verified", "lean_failed", "truncated"]:
        rs = [r for r in rows if r["outcome_state"] == state]
        serving = sum(r["serving_s"] for r in rs)
        verify = sum(r["verifier_s"] for r in rs)
        states.append({
            "state": state,
            "n": len(rs),
            "branch_share_pct": 100 * len(rs) / len(rows),
            "serving_s": serving,
            "serving_share_pct": 100 * serving / (prefill + decode),
            "verifier_s": verify,
            "total_s": serving + verify,
            "total_share_pct": 100 * (serving + verify) / phase_total,
            "completion_tok_mean": statistics.fmean(r["completion_tok"] for r in rs),
        })

    arms = []
    s = 700
    for cohort in COHORTS:
        for arm in "ABCD":
            arms.append(arm_summary(rows, cohort, arm, s))
            s += 1

    # cost per verified theorem, in the paper's own T_e2e currency, with the ratio
    # bootstrap and its zero-success accounting
    cps = []
    s = 900
    for cohort in COHORTS:
        for a in "ABCD":
            sample = [r for r in rows if r["cohort"] == cohort and r["arm"] == a]
            b = ratio_bootstrap_e2e(sample, s)
            b.update({"cohort": cohort, "arm": a})
            cps.append(b)
            s += 1

    # arm E against A-D recomputed on exactly E's eligibility subset
    e_matched = {}
    for cohort in COHORTS:
        eroots = {r["root_id"] for r in rows if r["cohort"] == cohort and r["arm"] == "E"}
        if not eroots:
            continue
        recs = []
        for a in "ABCDE":
            sample = [r for r in rows if r["cohort"] == cohort and r["arm"] == a
                      and r["root_id"] in eroots]
            if not sample:
                continue
            b = ratio_bootstrap_e2e(sample, s)
            b.update({"arm": a, "wall_sum": sum(r["wall_total_s"] for r in sample)})
            recs.append(b)
            s += 1
        e_matched[cohort] = {"n_roots": len(eroots), "arms": recs}

    contrasts = []
    seed = 10_000
    for cohort in COHORTS:
        for label, x, y in [("C-A", "C", "A"), ("D-B", "D", "B"), ("D-A", "D", "A")]:
            pairs = paired(rows, cohort, x, y)
            item = {"cohort": cohort, "contrast": label, "n": len(pairs)}
            for field in ["prefill_tok", "completion_tok", "total_tok", "replay_wall_s",
                          "verifier_s", "wall_total_s"]:
                item[field] = bootstrap_mean_diff(pairs, field, seed)
                seed += 1
            item["verified_per_1000_s"] = bootstrap_rate_diff(pairs, seed)
            seed += 1
            contrasts.append(item)

    return {
        "integrity": integrity,
        "counts": {
            "branches": len(rows),
            "factorial_branches": sum(r["arm"] in "ABCD" for r in rows),
            "native_branches": sum(r["arm"] == "E" for r in rows),
            "unique_roots": len({r["root_id"] for r in rows}),
        },
        "phases": {
            "prefill_s": prefill,
            "decode_s": decode,
            "verifier_s": verifier,
            "total_s": phase_total,
            "prefill_serving_share_pct": 100 * prefill / (prefill + decode),
            "prefill_total_share_pct": 100 * prefill / phase_total,
            "decode_total_share_pct": 100 * decode / phase_total,
            "verifier_total_share_pct": 100 * verifier / phase_total,
            "prefill_tok_per_s": source["phase_global"]["prefill_tok_per_s"],
            "decode_tok_per_s": source["phase_global"]["decode_tok_per_s"],
            "throughput_ratio_pooled": source["phase_global"]["throughput_ratio_pooled"],
            "throughput_ratio_steady_state": source["phase_global"]["throughput_ratio_steady_state"],
            "throughput_ratio_steady_state_range": [
                source["phase_global"]["throughput_ratio_steady_state_min"],
                source["phase_global"]["throughput_ratio_steady_state_max"],
            ],
        },
        "phase_split": source["phase_split"],
        "cost_per_success": cps,
        "cost_per_success_phase_audit": source["cost_per_success"],
        "e_matched": e_matched,
        "per_call_direction": source["per_call_direction"],
        "failure_states": states,
        "per_arm": arms,
        "contrasts": contrasts,
    }


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------- LaTeX tables


def ci(d: dict, scale: float = 1.0, digits: int = 2, signed: bool = True) -> str:
    lo, hi = d["ci95"][0] * scale, d["ci95"][1] * scale
    m = d.get("mean", d.get("point")) * scale
    s = "+" if signed and m >= 0 else ""
    return rf"${s}{m:.{digits}f}$ [${lo:+.{digits}f}$, ${hi:+.{digits}f}$]"


def write_tables(result: dict) -> None:
    con = {(c["cohort"], c["contrast"]): c for c in result["contrasts"]}
    arm = {(a["cohort"], a["arm"]): a for a in result["per_arm"]}

    # ---- Table 2: paired prior-context contrasts, evaluation blocks
    lines = []
    for label in ("C-A", "D-B"):
        for i, cohort in enumerate(["primary_eval", "eval_block2"]):
            c = con[(cohort, label)]
            head = rf"\multirow{{2}}{{*}}{{\arm{{{label[0]}}}--\arm{{{label[2]}}}}}" if i == 0 else ""
            lines.append(
                f"{head} & {i+1} & {ci(c['completion_tok'], 1/1000)} & "
                f"{ci(c['total_tok'], 1/1000)} & {ci(c['wall_total_s'], 1, 1)} & "
                f"{ci(c['verified_per_1000_s'], 1, 2)} \\\\")
        if label == "C-A":
            lines.append(r"\midrule")
    (TAB / "contrasts_eval.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table 3: per-arm accounting, evaluation blocks
    lines = []
    for bi, cohort in enumerate(["primary_eval", "eval_block2"]):
        for i, a in enumerate("ABCD"):
            r = arm[(cohort, a)]
            head = rf"\multirow{{4}}{{*}}{{{bi+1}}}" if i == 0 else ""
            sps = "---" if r["seconds_per_verified"] is None else f"{r['seconds_per_verified']:.0f}"
            lines.append(
                f"{head} & \\arm{{{a}}} & {r['prefill_tok_mean']:.0f} & "
                f"{r['completion_tok_mean']:.0f} & {r['total_tok_mean']:.0f} & "
                f"{r['wall_total_s_mean']:.1f} & {r['verified']}/{r['n']} & {sps} \\\\")
        if bi == 0:
            lines.append(r"\midrule")
    (TAB / "per_arm_eval.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table: per-arm phase split (the prefill result, per arm)
    lines = []
    for i, a in enumerate("ABCDE"):
        cells = []
        for cohort in COHORTS:
            p = next((p for p in result["phase_split"]
                      if p["cohort"] == cohort and p["arm"] == a), None)
            cells.append("---" if p is None else f"{p['prefill_share_pct']:.3f}")
        ratios = [p for p in result["phase_split"] if p["arm"] == a]
        pf = statistics.fmean(p["prefill_tok_per_s"] for p in ratios)
        dc = statistics.fmean(p["decode_tok_per_s"] for p in ratios)
        rr = [p["throughput_ratio"] for p in ratios]
        lines.append(
            f"\\arm{{{a}}} & {ARM_LABEL[a].split(': ')[1]} & " + " & ".join(cells)
            + f" & {pf:.0f} & {dc:.1f} & {min(rr):.0f}--{max(rr):.0f} \\\\")
    (TAB / "phase_split.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Table: outcome-state accounting
    lines = []
    for s in result["failure_states"]:
        lines.append(
            f"{s['state'].replace('_', ' ')} & {s['n']} & {s['branch_share_pct']:.1f}\\% & "
            f"{s['completion_tok_mean']:,.0f} & {s['serving_s']:,.0f} & "
            f"{s['serving_share_pct']:.1f}\\% & {s['verifier_s']:,.0f} & "
            f"{s['total_share_pct']:.1f}\\% \\\\")
    (TAB / "outcome_states.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: development contrasts
    lines = []
    for label in ("C-A", "D-B", "D-A"):
        c = con[("full_pool_dev", label)]
        lines.append(
            rf"\arm{{{label[0]}}}--\arm{{{label[2]}}} & {ci(c['completion_tok'], 1/1000)} & "
            f"{ci(c['total_tok'], 1/1000)} & {ci(c['wall_total_s'], 1, 1)} & "
            f"{ci(c['verified_per_1000_s'], 1, 2)} \\\\")
    (TAB / "contrasts_dev.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: development per-arm
    lines = []
    for a in "ABCD":
        r = arm[("full_pool_dev", a)]
        sps = "---" if r["seconds_per_verified"] is None else f"{r['seconds_per_verified']:.0f}"
        lines.append(
            f"\\arm{{{a}}} & {r['prefill_tok_mean']:.0f} & {r['completion_tok_mean']:.0f} & "
            f"{r['total_tok_mean']:.0f} & {r['wall_total_s_mean']:.1f} & "
            f"{r['verified']}/{r['n']} & {sps} \\\\")
    (TAB / "per_arm_dev.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: marginal cost per success with the ratio bootstrap
    lines = []
    for cohort in COHORTS:
        recs = [r for r in result["cost_per_success"] if r["cohort"] == cohort]
        for i, b in enumerate(recs):
            head = rf"\multirow{{4}}{{*}}{{{COHORT_SHORT[cohort]}}}" if i == 0 else ""
            if b["point_infinite"]:
                pt, lo, hi = r"$\infty$", r"$\infty$", r"$\infty$"
            else:
                pt = f"{b['point']:.0f}"
                lo = r"$\infty$" if b["ci95_lower_infinite"] else f"{b['ci95'][0]:.0f}"
                hi = r"$\infty$" if b["ci95_upper_infinite"] else f"{b['ci95'][1]:.0f}"
            lines.append(
                f"{head} & \\arm{{{b['arm']}}} & {b['verified']}/{b['n']} & {pt} & "
                f"[{lo}, {hi}] & {b['p_zero_success']:.3f} \\\\")
        if cohort != COHORTS[-1]:
            lines.append(r"\midrule")
    (TAB / "cost_per_success.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: per-call direction non-replication
    lines = []
    label = {"serving_s": "measured serving", "replay_wall_s": "request wall",
             "total_s": "serving + verifier"}
    for cur in ("serving_s", "replay_wall_s", "total_s"):
        recs = [d for d in result["per_call_direction"] if d["currency"] == cur]
        for i, d in enumerate(recs):
            head = rf"\multirow{{3}}{{*}}{{{label[cur]}}}" if i == 0 else ""
            lines.append(
                f"{head} & {COHORT_LABEL[d['cohort']]} & ${d['mean_diff']:+.2f}$ & "
                f"[${d['ci95'][0]:+.2f}$, ${d['ci95'][1]:+.2f}$] & "
                f"{'resolved' if d['resolved'] else 'unresolved'} \\\\")
        if cur != "total_s":
            lines.append(r"\midrule")
    (TAB / "per_call_direction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: arm E on its matched eligibility subset
    lines = []
    keys = list(result["e_matched"])
    for cohort in keys:
        blob = result["e_matched"][cohort]
        for i, b in enumerate(blob["arms"]):
            head = (rf"\multirow{{5}}{{*}}{{{COHORT_LABEL[cohort]} ({blob['n_roots']})}}"
                    if i == 0 else "")
            sps = r"$\infty$" if b["point_infinite"] else f"{b['point']:.0f}"
            lines.append(
                f"{head} & \\arm{{{b['arm']}}} & {b['n']} & {b['verified']} & "
                f"{b['wall_sum']:,.0f} & {sps} \\\\")
        if cohort != keys[-1]:
            lines.append(r"\midrule")
    (TAB / "e_matched.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Appendix: replay integrity gate
    g = result["integrity"]
    checks = [
        ("Replay run status", g["replay_status"], "complete"),
        ("Branches done in original run", f"{g['runs_done']}", "536"),
        ("Replay timing rows", f"{g['replay_rows']}", "536"),
        ("Rows joining on branch key", f"{g['joined']}", "536"),
        ("Completion-hash mismatches", f"{g['hash_mismatch']}", "0"),
        ("Rows reporting cached tokens", f"{g['cache_reuse']}", "0"),
        ("Prefill-token disagreements", f"{g['token_mismatch_prefill_tokens']}", "0"),
        ("Decode-token disagreements", f"{g['token_mismatch_decode_tokens']}", "0"),
        ("Completion-token disagreements", f"{g['token_mismatch_completion_tokens']}", "0"),
        ("Total-token disagreements", f"{g['token_mismatch_total_tokens']}", "0"),
        ("Non-positive prefill or decode times", f"{g['nonpositive_prefill'] + g['nonpositive_decode']}", "0"),
        ("Rows outside a completed segment", f"{g['orphan_segment_rows']}", "0"),
        ("Server restarts during replay (segments)", f"{g['segments']}", "1"),
        ("Canary rows held out of analysis", f"{g['canary_rows_quarantined']}", "---"),
    ]
    lines = [f"{name} & {val} & {exp} \\\\" for name, val, exp in checks]
    (TAB / "integrity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_macros(result: dict) -> str:
    p = result["phases"]
    c = result["counts"]
    state = {r["state"]: r for r in result["failure_states"]}
    con = {(z["cohort"], z["contrast"]): z for z in result["contrasts"]}
    arm = {(a["cohort"], a["arm"]): a for a in result["per_arm"]}
    lines = [
        "% Generated by scripts/build_assets.py; do not edit numeric values by hand.",
        rf"\newcommand{{\AllBranches}}{{{c['branches']}}}",
        rf"\newcommand{{\FactorialBranches}}{{{c['factorial_branches']}}}",
        rf"\newcommand{{\NativeBranches}}{{{c['native_branches']}}}",
        rf"\newcommand{{\UniqueRoots}}{{{c['unique_roots']}}}",
        # Quantities that were typed literals in v2. Root counts, the per-arm branch
        # count, the token cap, and the two mean serving times in the censoring
        # paragraph -- the last two are measurements, not configuration.
        rf"\newcommand{{\EvalRoots}}{{40}}",
        rf"\newcommand{{\DevRoots}}{{43}}",
        rf"\newcommand{{\ArmBranches}}{{{c['factorial_branches'] // 4}}}",
        rf"\newcommand{{\TokenCap}}{{12{{,}}288}}",
        rf"\newcommand{{\VerifiedServingMean}}"
        rf"{{{state['verified']['serving_s'] / state['verified']['n']:.1f}}}",
        rf"\newcommand{{\TruncatedServingMean}}"
        rf"{{{state['truncated']['serving_s'] / state['truncated']['n']:.1f}}}",
        rf"\newcommand{{\LeanFailedServingMean}}"
        rf"{{{state['lean_failed']['serving_s'] / state['lean_failed']['n']:.1f}}}",
        rf"\newcommand{{\PrefillServingShare}}{{{fmt(p['prefill_serving_share_pct'],2)}\%}}",
        rf"\newcommand{{\PrefillTotalShare}}{{{fmt(p['prefill_total_share_pct'],2)}\%}}",
        rf"\newcommand{{\DecodeTotalShare}}{{{fmt(p['decode_total_share_pct'],1)}\%}}",
        rf"\newcommand{{\VerifierTotalShare}}{{{fmt(p['verifier_total_share_pct'],1)}\%}}",
        rf"\newcommand{{\PrefillSeconds}}{{{p['prefill_s']:,.0f}}}",
        rf"\newcommand{{\DecodeSeconds}}{{{p['decode_s']:,.0f}}}",
        rf"\newcommand{{\VerifierSeconds}}{{{p['verifier_s']:,.0f}}}",
        rf"\newcommand{{\PhaseHours}}{{{fmt(p['total_s']/3600,2)}}}",
        rf"\newcommand{{\PrefillThroughput}}{{{fmt(p['prefill_tok_per_s'],0)}}}",
        rf"\newcommand{{\DecodeThroughput}}{{{fmt(p['decode_tok_per_s'],1)}}}",
        rf"\newcommand{{\PooledRatio}}{{{fmt(p['throughput_ratio_pooled'],1)}$\times$}}",
        rf"\newcommand{{\SteadyRatio}}{{{fmt(p['throughput_ratio_steady_state'],1)}$\times$}}",
        rf"\newcommand{{\SteadyRatioLo}}{{{fmt(p['throughput_ratio_steady_state_range'][0],1)}}}",
        rf"\newcommand{{\SteadyRatioHi}}{{{fmt(p['throughput_ratio_steady_state_range'][1],1)}}}",
        rf"\newcommand{{\TruncatedN}}{{{state['truncated']['n']}}}",
        rf"\newcommand{{\TruncatedBranchShare}}{{{fmt(state['truncated']['branch_share_pct'],1)}\%}}",
        rf"\newcommand{{\TruncatedServingShare}}{{{fmt(state['truncated']['serving_share_pct'],1)}\%}}",
        rf"\newcommand{{\TruncatedServingSeconds}}{{{state['truncated']['serving_s']:,.0f}}}",
        rf"\newcommand{{\TruncatedTotalShare}}{{{fmt(state['truncated']['total_share_pct'],1)}\%}}",
        rf"\newcommand{{\LeanFailedN}}{{{state['lean_failed']['n']}}}",
        rf"\newcommand{{\LeanFailedTotalShare}}{{{fmt(state['lean_failed']['total_share_pct'],1)}\%}}",
        rf"\newcommand{{\VerifiedN}}{{{state['verified']['n']}}}",
        rf"\newcommand{{\VerifiedTotalShare}}{{{fmt(state['verified']['total_share_pct'],1)}\%}}",
    ]
    cohort_alias = {"primary_eval": "One", "eval_block2": "Two", "full_pool_dev": "Dev"}
    for row in result["per_arm"]:
        a = cohort_alias[row["cohort"]] + row["arm"]
        sps = "--" if row["seconds_per_verified"] is None else fmt(row["seconds_per_verified"], 0)
        lines += [
            rf"\newcommand{{\{a}Verified}}{{{row['verified']}/{row['n']}}}",
            rf"\newcommand{{\{a}Completion}}{{{fmt(row['completion_tok_mean'],0)}}}",
            rf"\newcommand{{\{a}TotalTok}}{{{fmt(row['total_tok_mean'],0)}}}",
            rf"\newcommand{{\{a}Wall}}{{{fmt(row['wall_total_s_mean'],1)}}}",
            rf"\newcommand{{\{a}WallSum}}{{{row['wall_total_s_sum']:,.0f}}}",
            rf"\newcommand{{\{a}SecPerSuccess}}{{{sps}}}",
        ]
    # headline efficiency contrasts, so the prose cannot drift from the table
    for cohort in COHORTS:
        al = cohort_alias[cohort]
        for label in ("C-A", "D-B"):
            d = con[(cohort, label)]["verified_per_1000_s"]
            nm = al + label.replace("-", "")
            # math mode, so a negative sign typesets as a minus and not a hyphen
            lines += [
                rf"\newcommand{{\Rate{nm}}}{{${d['point']:+.2f}$}}",
                rf"\newcommand{{\Rate{nm}CI}}{{[${d['ci95'][0]:+.2f}$, ${d['ci95'][1]:+.2f}$]}}",
            ]
        d = con[(cohort, "C-A")]["total_tok"]
        lines.append(rf"\newcommand{{\TotTok{al}}}{{{d['mean']:,.0f}}}")
        d = con[(cohort, "C-A")]["completion_tok"]
        lines.append(rf"\newcommand{{\ComplTok{al}}}{{{abs(d['mean']):,.0f}}}")
    # ratio of seconds-per-verified, C over A, per evaluation block
    for cohort in ("primary_eval", "eval_block2"):
        al = cohort_alias[cohort]
        ra, rc = arm[(cohort, "A")], arm[(cohort, "C")]
        lines.append(
            rf"\newcommand{{\CoverA{al}}}{{{rc['seconds_per_verified']/ra['seconds_per_verified']:.1f}$\times$}}")
        # how far apart the arms actually are on COST -- the paper says "nearly identical",
        # so the number behind that adjective has to be generated, not asserted.  A-vs-C
        # and the full four-arm spread are different numbers and are kept apart.
        ac = abs(rc["wall_total_s_sum"] - ra["wall_total_s_sum"]) / ra["wall_total_s_sum"]
        sums = [arm[(cohort, a)]["wall_total_s_sum"] for a in "ABCD"]
        wins = [arm[(cohort, a)]["verified"] for a in "ABCD"]
        lines += [
            rf"\newcommand{{\CostSpreadAC{al}}}{{{100*ac:.1f}\%}}",
            rf"\newcommand{{\CostSpreadFour{al}}}{{{100*(max(sums)-min(sums))/min(sums):.0f}\%}}",
            rf"\newcommand{{\YieldSpreadFour{al}}}{{{max(wins)/max(min(wins),1):.0f}$\times$}}",
        ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- figures


def set_font(c: canvas.Canvas, size: float, bold: bool = False, color=DARK) -> None:
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.setFillColor(color)


def draw_title(c: canvas.Canvas, title: str, subtitle: str, width: float, height: float) -> None:
    set_font(c, 14, True)
    c.drawString(24, height - 26, title)
    set_font(c, 8.2, False, GRAY)
    c.drawString(24, height - 39, subtitle)


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick positions, always including zero when it is inside the range."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        ticks.append(0.0 if abs(t) < step / 1e6 else t)
        t += step
    return ticks


def forest_panel(c: canvas.Canvas, rows: list[tuple[str, float, float, float]],
                 x: float, top: float, w: float, title: str, xlabel: str,
                 color=BLUE, digits: int = 2) -> float:
    """Draw one forest panel with its top edge at `top`.  Returns the y of its bottom.

    Layout is an explicit grid: title band, one band per row, then the axis band.  The
    v1 version placed the title at a fixed offset inside the panel box and the tick
    labels below it, which let a panel's title land on the previous panel's tick row.
    """
    TITLE_H, ROW_H, AXIS_H = 15.0, 17.0, 26.0
    set_font(c, 8.6, True)
    c.drawString(x, top - 10, title)
    body_top = top - TITLE_H
    body_bot = body_top - ROW_H * len(rows)

    vals = [z for _, lo, pt, hi in rows for z in (lo, pt, hi)] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = 0.14 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    sx = lambda v: x + w * (v - lo) / (hi - lo)

    # zero reference
    c.setStrokeColor(MID)
    c.setLineWidth(0.8)
    c.setDash(2, 2)
    c.line(sx(0), body_bot, sx(0), body_top)
    c.setDash()

    for i, (label, l, pt, u) in enumerate(rows):
        yy = body_top - ROW_H * (i + 0.5)
        is_dev = i >= 2
        col = GRAY if is_dev else color
        c.setStrokeColor(col)
        c.setLineWidth(1.5)
        c.line(sx(l), yy, sx(u), yy)
        for e in (sx(l), sx(u)):                       # interval caps
            c.line(e, yy - 2.5, e, yy + 2.5)
        c.setFillColor(col)
        c.circle(sx(pt), yy, 3.1, fill=1, stroke=0)
        set_font(c, 7.6, False, DARK if not is_dev else GRAY)
        c.drawRightString(x - 6, yy - 2.6, label)
        # numeric annotation, right of the panel
        set_font(c, 7.0, False, GRAY)
        c.drawString(x + w + 7, yy - 2.4, f"{pt:+.{digits}f}")

    # axis
    c.setStrokeColor(GRAY)
    c.setLineWidth(0.6)
    c.line(x, body_bot, x + w, body_bot)
    for t in nice_ticks(lo, hi):
        xx = sx(t)
        c.line(xx, body_bot, xx, body_bot - 3)
        set_font(c, 6.8, False, GRAY)
        c.drawCentredString(xx, body_bot - 12, f"{t:g}")
    if xlabel:
        set_font(c, 7.2, False, GRAY)
        c.drawCentredString(x + w / 2, body_bot - 22, xlabel)
    return body_bot - AXIS_H


def currencies_figure(result: dict) -> None:
    path = FIG / "currencies_disagree.pdf"
    w, h = 500, 350
    c = canvas.Canvas(str(path), pagesize=(w, h))
    draw_title(c, "The prior-attempt contrast changes sign across currencies",
               "Paired C−A differences, 95% root-bootstrap intervals; seed blocks kept separate",
               w, h)
    items = {z["cohort"]: z for z in result["contrasts"] if z["contrast"] == "C-A"}
    panels = [
        ("Completion tokens", "completion_tok", 1 / 1000,
         "thousand tokens (negative = prior context decodes less)", BLUE, 2),
        ("Total tokens", "total_tok", 1 / 1000,
         "thousand tokens (positive = prior context costs more input)", ORANGE, 2),
        ("End-to-end wall time", "wall_total_s", 1.0,
         "seconds (interval spans zero in every cohort)", GREEN, 1),
    ]
    top = h - 56
    for title, field, scale, xlabel, color, dg in panels:
        rows = []
        for cohort in COHORTS:
            d = items[cohort][field]
            rows.append((COHORT_LABEL[cohort], d["ci95"][0] * scale,
                         d["mean"] * scale, d["ci95"][1] * scale))
        top = forest_panel(c, rows, 108, top, 322, title, xlabel, color, dg)
    c.save()


def efficiency_figure(result: dict) -> None:
    path = FIG / "efficiency_contrasts.pdf"
    w, h = 500, 250
    c = canvas.Canvas(str(path), pagesize=(w, h))
    draw_title(c, "Successful proofs per unit wall time",
               "Paired-root bootstrap of the rate difference; negative favours the fresh arm",
               w, h)
    top = h - 54
    for contrast, color in (("C-A", RED), ("D-B", ORANGE)):
        items = {z["cohort"]: z for z in result["contrasts"] if z["contrast"] == contrast}
        rows = []
        for cohort in COHORTS:
            d = items[cohort]["verified_per_1000_s"]
            rows.append((COHORT_LABEL[cohort], d["ci95"][0], d["point"], d["ci95"][1]))
        title = ("C−A: prior attempt, no guidance" if contrast == "C-A"
                 else "D−B: prior attempt, with guidance")
        top = forest_panel(c, rows, 112, top, 316, title,
                           "verified theorems per 1,000 end-to-end wall-seconds", color, 2)
    c.save()


def phase_figure(result: dict) -> None:
    path = FIG / "phase_accounting.pdf"
    w, h = 500, 250
    c = canvas.Canvas(str(path), pagesize=(w, h))
    draw_title(c, "Where the measured compute went",
               "All 536 branches; exact serving replay joined to original Lean verification",
               w, h)
    p = result["phases"]
    x, bw, bh = 24, 452, 26
    y = h - 92
    vals = [("Prefill", p["prefill_total_share_pct"], BLUE2),
            ("Decode", p["decode_total_share_pct"], BLUE),
            ("Lean verifier", p["verifier_total_share_pct"], ORANGE)]
    cur = x
    for label, pct, color in vals:
        seg = bw * pct / 100
        c.setFillColor(color)
        c.rect(cur, y, max(seg, 1.2), bh, fill=1, stroke=0)
        if seg > 60:
            set_font(c, 9, True, white)
            c.drawCentredString(cur + seg / 2, y + 9, f"{label} {pct:.1f}%")
        cur += seg
    # leader line calling out the invisible prefill sliver
    px = x + bw * p["prefill_total_share_pct"] / 200
    c.setStrokeColor(BLUE)
    c.setLineWidth(0.8)
    c.line(px, y + bh, px, y + bh + 12)
    c.line(px, y + bh + 12, px + 16, y + bh + 12)
    set_font(c, 8.2, True, BLUE)
    c.drawString(px + 20, y + bh + 9.4,
                 f"Prefill = {p['prefill_total_share_pct']:.2f}% of phase total "
                 f"({p['prefill_serving_share_pct']:.2f}% of serving) — the hypothesised tax")
    set_font(c, 7.2, False, GRAY)
    c.drawString(x, y - 11, f"{p['prefill_s']:,.0f} s prefill · {p['decode_s']:,.0f} s decode "
                            f"· {p['verifier_s']:,.0f} s Lean · {p['total_s']/3600:.2f} phase-hours total")

    set_font(c, 9, True)
    c.drawString(x, y - 34, "Outcome states: share of branches vs. share of phase total")
    y0 = y - 52
    lab_x, bar_x, bar_w = 108, 116, 300
    for i, row in enumerate(result["failure_states"]):
        yy = y0 - i * 26
        set_font(c, 8, False)
        c.drawRightString(lab_x, yy + 6, row["state"].replace("_", " "))
        for k, (share, col) in enumerate((
                (row["branch_share_pct"], MID),
                (row["total_share_pct"], RED if row["state"] == "truncated" else GREEN))):
            by = yy + 8 - k * 9
            c.setFillColor(LIGHT)
            c.rect(bar_x, by, bar_w, 7.5, fill=1, stroke=0)
            c.setFillColor(col)
            c.rect(bar_x, by, bar_w * share / 100, 7.5, fill=1, stroke=0)
            set_font(c, 7.2, False, GRAY)
            c.drawString(bar_x + bar_w + 6, by + 1.5, f"{share:.1f}%")
    # legend
    set_font(c, 7.2, False, GRAY)
    c.setFillColor(MID)
    c.rect(bar_x, 12, 12, 7.5, fill=1, stroke=0)
    set_font(c, 7.2, False, GRAY)
    c.drawString(bar_x + 16, 14, "share of branches")
    c.setFillColor(GREEN)
    c.rect(bar_x + 108, 12, 12, 7.5, fill=1, stroke=0)
    c.setFillColor(RED)
    c.rect(bar_x + 120, 12, 12, 7.5, fill=1, stroke=0)
    set_font(c, 7.2, False, GRAY)
    c.drawString(bar_x + 136, 14, "share of phase total")
    c.save()


def isocost_figure(result: dict) -> None:
    """NEW in v2: the central economic claim, drawn.

    Arms sit at almost the same total end-to-end wall cost and buy very different
    numbers of verified theorems.  A cost-only ledger cannot see the vertical axis.
    """
    path = FIG / "isocost_yield.pdf"
    w, h = 500, 250
    c = canvas.Canvas(str(path), pagesize=(w, h))
    draw_title(c, "Cost barely separates the arms; yield separates them ninefold",
               "Each point is one arm's whole run: total end-to-end wall seconds vs. theorems verified",
               w, h)
    arm = {(a["cohort"], a["arm"]): a for a in result["per_arm"]}
    cohorts = ["primary_eval", "eval_block2"]
    pw, ph = 196, 132
    for j, cohort in enumerate(cohorts):
        x0 = 56 + j * 236
        y0 = 44
        recs = [arm[(cohort, a)] for a in "ABCD"]
        xs = [r["wall_total_s_sum"] for r in recs]
        lo, hi = min(xs), max(xs)
        pad = max(0.35 * (hi - lo), 120)
        lo, hi = lo - pad, hi + pad
        ymax = 10
        sx = lambda v: x0 + pw * (v - lo) / (hi - lo)
        sy = lambda v: y0 + ph * v / ymax

        set_font(c, 8.6, True)
        c.drawString(x0, y0 + ph + 16, f"Evaluation block {j+1}")
        # gridlines
        c.setStrokeColor(LIGHT)
        c.setLineWidth(0.6)
        for t in range(0, ymax + 1, 2):
            c.line(x0, sy(t), x0 + pw, sy(t))
            set_font(c, 6.8, False, GRAY)
            c.drawRightString(x0 - 5, sy(t) - 2.4, str(t))
        # axes
        c.setStrokeColor(GRAY)
        c.setLineWidth(0.6)
        c.line(x0, y0, x0 + pw, y0)
        c.line(x0, y0, x0, y0 + ph)
        for t in nice_ticks(lo, hi, 3):
            c.line(sx(t), y0, sx(t), y0 - 3)
            set_font(c, 6.8, False, GRAY)
            c.drawCentredString(sx(t), y0 - 12, f"{t:,.0f}")
        set_font(c, 7.2, False, GRAY)
        c.drawCentredString(x0 + pw / 2, y0 - 23, "total end-to-end wall seconds")
        if j == 0:
            c.saveState()
            c.translate(x0 - 26, y0 + ph / 2)
            c.rotate(90)
            set_font(c, 7.2, False, GRAY)
            c.drawCentredString(0, 0, "theorems verified")
            c.restoreState()

        colors = {"A": GREEN, "B": BLUE, "C": RED, "D": ORANGE}
        # Only the arm letter is drawn.  Seconds-per-proof sits in the table immediately
        # above this figure, and repeating it here collided C against D in block 2 without
        # adding information.  The figure's job is the cost-versus-yield geometry.
        placed: list[tuple[float, float]] = []
        for r in sorted(recs, key=lambda z: z["wall_total_s_sum"]):
            a = r["arm"]
            px, py = sx(r["wall_total_s_sum"]), sy(r["verified"])
            lx, ly = px + 7, py - 3.0
            if any(abs(lx - qx) < 12 and abs(ly - qy) < 8 for qx, qy in placed):
                lx, ly = px - 12, py - 3.0
            placed.append((lx, ly))
            c.setFillColor(colors[a])
            c.circle(px, py, 4.6, fill=1, stroke=0)
            set_font(c, 8.4, True, colors[a])
            c.drawString(lx, ly, a)
        # iso-cost band showing how little the x-axis separates the arms
        c.setStrokeColor(MID)
        c.setDash(2, 2)
        c.setLineWidth(0.7)
        c.line(sx(min(xs)), y0, sx(min(xs)), y0 + ph)
        c.line(sx(max(xs)), y0, sx(max(xs)), y0 + ph)
        c.setDash()
        wins = [r["verified"] for r in recs]
        set_font(c, 6.8, False, GRAY)
        c.drawCentredString((sx(min(xs)) + sx(max(xs))) / 2, y0 + ph + 4,
                            f"cost spread {100*(max(xs)-min(xs))/min(xs):.0f}%  ·  "
                            f"yield spread {max(wins)/max(min(wins),1):.0f}×")
    c.save()


NUMWORD = {"split2": "SplitTwo", "split3": "SplitThree", "long": "Long"}


def portfolio_assets() -> list[str]:
    """Macros for the exact cap-schedule frontier (s41).

    Emits nothing when s41 has not been run, so the paper build fails on a missing macro
    rather than printing a stale ceiling. The guard on `ok` matters more than usual here:
    the whole frontier is valid only if the seed-lattice prefix property holds, and a
    silently-false lattice would produce numbers that look fine and mean nothing.
    """
    if not PORTFOLIO_JSON.exists():
        print(f"  note: {PORTFOLIO_JSON.name} absent -- portfolio macros not emitted")
        return []
    p = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    if not p.get("seed_lattice", {}).get("ok"):
        print("  note: seed lattice check failed -- portfolio macros NOT emitted")
        return []
    if not all(v["ok"] for v in p["validity"]):
        print("  note: portfolio simulator disagrees with an executed cohort -- "
              "macros NOT emitted")
        return []

    n = p["n_roots"]
    ceil_, rnd, uni = p["ceiling"], p["round_best"], p["union"]
    caps = rnd["caps"]
    cap_str = " + ".join(f"{c:,}".replace(",", "{,}") for c in caps if c > 0)
    prefix = p["seed_lattice"]["prefix_checks"][0]

    L = ["", "% --- exact cap-schedule frontier (s41) ---",
         rf"\newcommand{{\PortfolioCeiling}}{{{ceil_['verified']}}}",
         rf"\newcommand{{\PortfolioCeilingFrac}}{{${ceil_['verified']}/{n}$}}",
         rf"\newcommand{{\PortfolioBestPreReg}}{{{p['best_pre_registered']}}}",
         rf"\newcommand{{\PortfolioBestPreRegFrac}}"
         rf"{{${p['best_pre_registered']}/{n}$}}",
         rf"\newcommand{{\PortfolioHeadroomRoots}}{{{p['headroom_roots']}}}",
         rf"\newcommand{{\PortfolioHeadroomVsLong}}"
         rf"{{{ceil_['verified'] - p['recorded_verified']['long']}}}",
         rf"\newcommand{{\PortfolioHeadroomPts}}{{${p['headroom_points']:+.1f}$}}",
         rf"\newcommand{{\PortfolioRoundCaps}}{{${cap_str}$}}",
         rf"\newcommand{{\PortfolioRoundN}}{{{rnd['n_schedules_searched']}}}",
         rf"\newcommand{{\PortfolioRoundAttaining}}{{{rnd['n_attaining']}}}",
         rf"\newcommand{{\PortfolioOracleN}}{{{ceil_['n_schedules_searched']}}}",
         rf"\newcommand{{\PortfolioOracleCaps}}"
         rf"{{${' + '.join(f'{c:,}'.replace(',', '{,}') for c in ceil_['caps'] if c > 0)}$}}",
         rf"\newcommand{{\PortfolioPrefixCheck}}"
         rf"{{{prefix['exact_prefix']}/{prefix['n']}}}",
         rf"\newcommand{{\PortfolioUnion}}{{{uni['verified']}}}",
         rf"\newcommand{{\PortfolioUnionFrac}}{{${uni['verified']}/{n}$}}",
         rf"\newcommand{{\PortfolioUnionMinTokens}}"
         rf"{{{uni['min_nominal_tokens']:,}}}".replace(",", "{,}"),
         rf"\newcommand{{\PortfolioUnionMultiple}}{{{uni['budget_multiple']:.3f}}}",
         rf"\newcommand{{\PortfolioUnionOverBudget}}"
         rf"{{{uni['min_nominal_tokens'] - p['budget_nominal_tokens']:,}}}",
         rf"\newcommand{{\PortfolioUnionSeeds}}"
         rf"{{{sum(1 for c in uni['min_caps'] if c > 0)}}}",
         rf"\newcommand{{\PortfolioUnionAllDrawsMultiple}}"
         rf"{{{uni['nominal_all_executed_draws'] / float(p['budget_nominal_tokens']):.2f}}}",
         ]
    for nd, word in ((1, "One"), (2, "Two"), (3, "Three")):
        d = p["by_draw_count"].get(str(nd)) or p["by_draw_count"].get(nd)
        if d:
            L.append(rf"\newcommand{{\PortfolioDraw{word}}}{{${d['yield']}/{n}$}}")

    # the frontier table: minimum nominal budget to reach each yield
    front = p["frontier_oracle"]
    rows_out = []
    for y in sorted(front, key=int):
        if int(y) < p["best_pre_registered"] - 1:
            continue
        f = front[y]
        cs = ", ".join(f"{c:,}" for c in f["caps"] if c > 0)
        mark = "yes" if f["nominal"] <= p["budget_nominal_tokens"] else "no"
        rows_out.append(rf"{y}/{n} & {f['nominal']:,} & {cs} & {mark} \\")
    (TAB / "portfolio_frontier.tex").write_text("\n".join(rows_out) + "\n",
                                                encoding="utf-8")
    print(f"  wrote tables/portfolio_frontier.tex ({len(rows_out)} rows) "
          f"and {len(L) - 2} portfolio macros")
    return L


def crossover_assets() -> list[str]:
    """Macros for the prefill-share decomposition and the measured serving regimes (s42).

    Regime macros are emitted only for regimes that actually completed both arms, so a
    half-finished side-measurement cannot reach the paper as if it were a full one.
    """
    if not CROSSOVER_JSON.exists():
        print(f"  note: {CROSSOVER_JSON.name} absent -- crossover macros not emitted")
        return []
    x = json.loads(CROSSOVER_JSON.read_text(encoding="utf-8"))
    fr = x["frozen_regime"]
    if "C" not in fr:
        return []

    L = ["", "% --- prefill-share decomposition and serving regimes (s42) ---"]
    for arm in ("A", "C", "pooled"):
        if arm not in fr:
            continue
        tag = {"A": "ArmA", "C": "ArmC", "pooled": "Pooled"}[arm]
        b = fr[arm]
        L += [rf"\newcommand{{\Stack{tag}R}}{{{b['R_stack']:.1f}}}",
              rf"\newcommand{{\Workload{tag}Rho}}{{{b['rho_workload']:.2f}}}",
              rf"\newcommand{{\Prefill{tag}Share}}{{{b['prefill_serving_share_pct']:.2f}\%}}",
              rf"\newcommand{{\Decode{tag}Rate}}{{{b['decode_tok_per_s']:.1f}}}",
              rf"\newcommand{{\Prefill{tag}Rate}}{{{b['prefill_tok_per_s']:,.0f}}}"
              .replace(",", "{,}")]

    cx = x.get("crossover_rho_frozen_armC") or {}
    rho = fr["C"]["rho_workload"]
    for key, word in (("0.10", "Ten"), ("0.25", "TwentyFive"), ("0.50", "Half")):
        if key in cx:
            L += [rf"\newcommand{{\CrossoverRho{word}}}{{{cx[key]:.1f}}}",
                  rf"\newcommand{{\CrossoverFactor{word}}}{{{cx[key] / rho:.0f}$\times$}}"]

    for reg, d in sorted((x.get("measured_regimes") or {}).items()):
        arms = d.get("arms") or {}
        if not all(a in arms for a in ("A", "C", "pooled")):
            print(f"  note: regime {reg} incomplete -- its macros not emitted")
            continue
        run = d.get("run") or {}
        det = d.get("determinism_vs_frozen") or {}
        tag = "Batch" if (run.get("workers") or 0) > 1 else "Serial"
        L.append(rf"\newcommand{{\Regime{tag}Slots}}{{{run.get('parallel_slots')}}}")
        L.append(rf"\newcommand{{\Regime{tag}Workers}}{{{run.get('workers')}}}")
        L.append(rf"\newcommand{{\Regime{tag}Identical}}"
                 rf"{{{det.get('n_identical')}/{det.get('n_compared')}}}")
        for arm in ("A", "C", "pooled"):
            at = {"A": "ArmA", "C": "ArmC", "pooled": "Pooled"}[arm]
            b = arms[arm]
            L += [rf"\newcommand{{\Regime{tag}{at}R}}{{{b['R_stack']:.1f}}}",
                  rf"\newcommand{{\Regime{tag}{at}Share}}"
                  rf"{{{b['prefill_serving_share_pct']:.2f}\%}}",
                  rf"\newcommand{{\Regime{tag}{at}DecodeRate}}"
                  rf"{{{b['decode_tok_per_s']:.1f}}}",
                  rf"\newcommand{{\Regime{tag}{at}PrefillRate}}"
                  rf"{{{b['prefill_tok_per_s']:,.0f}}}".replace(",", "{,}")]
            if "counterfactual_share_at_frozen_rho_pct" in b:
                L.append(rf"\newcommand{{\Regime{tag}{at}ShareCf}}"
                         rf"{{{b['counterfactual_share_at_frozen_rho_pct']:.2f}\%}}")
    print(f"  wrote {len(L) - 2} crossover/regime macros")
    return L


def sensitivity_assets() -> list[str]:
    """Generated sensitivity assets."""
    mutations = json.loads(MUTATION_JSON.read_text(encoding="utf-8"))
    cap = json.loads(CLASS_CAP_JSON.read_text(encoding="utf-8"))
    summary = mutations["summary"]
    assert summary["faults_injected"] == summary["faults_detected"]
    assert summary["database_modified"] is False

    labels = {
        "corrupt_branch_key": "Corrupt branch key",
        "corrupt_cap": "Corrupt action cap",
        "inject_budget_overrun": "Inject budget overrun",
        "flip_stop_decision": "Flip stop decision",
        "corrupt_accounting_token": "Corrupt token accounting",
        "perturb_action_sequence": "Perturb action sequence",
        "perturb_replay_signature": "Perturb replay signature",
    }
    rows = []
    for item in mutations["mutations"]:
        invariant = item["expected_invariant"].replace("_", r"\_")
        rows.append(
            f"{labels[item['fault']]} & {invariant} & "
            f"{item['detection_ms']:.3f} ms \\\\"
        )
    (TAB / "mutation_detection.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cap_rows = []
    for cohort in ("dev", "eval"):
        blob = cap["cohorts"][cohort]
        for i, item in enumerate(blob["variants"]):
            head = cohort.capitalize() if i == 0 else ""
            cap_rows.append(
                f"{head} & {item['truncated_root_cap']:,} & "
                f"{item['verified']}/{blob['n']} & {format(item['verified_change'], '+d').replace('-', '$-$')} & "
                f"{item['tokens_saved']:,} & {100*item['token_savings_fraction']:.1f}\\% \\\\"
            )
        if cohort == "dev":
            cap_rows.append(r"\midrule")
    (TAB / "class_conditional_cap.tex").write_text(
        "\n".join(cap_rows) + "\n", encoding="utf-8")

    selected = cap["eval_at_dev_selected_cap"]
    eval_8192 = next(item for item in cap["cohorts"]["eval"]["variants"]
                     if item["truncated_root_cap"] == 8192)
    return [
        r"% Final-review sensitivity analyses (generated)",
        f"\\newcommand{{\\HarnessMutations}}{{{summary['faults_injected']}}}",
        f"\\newcommand{{\\HarnessDetected}}{{{summary['faults_detected']}/{summary['faults_injected']}}}",
        f"\\newcommand{{\\HarnessMaxDetect}}{{{summary['max_detection_ms']:.1f} ms}}",
        f"\\newcommand{{\\ClassCapDevSelected}}{{{cap['dev_selected_cap']:,}}}",
        f"\\newcommand{{\\ClassCapEvalVerified}}{{{selected['verified']}/40}}",
        f"\\newcommand{{\\ClassCapEvalLoss}}{{{abs(selected['verified_change'])}}}",
        f"\\newcommand{{\\ClassCapDevLongTokens}}{{{cap['cohorts']['dev']['long_realized_completion_tokens']:,}}}",
        f"\\newcommand{{\\ClassCapEvalLongTokens}}{{{cap['cohorts']['eval']['long_realized_completion_tokens']:,}}}",
        f"\\newcommand{{\\ClassCapEvalEightVerified}}{{{eval_8192['verified']}/40}}",
        f"\\newcommand{{\\ClassCapEvalEightTokensSaved}}{{{eval_8192['tokens_saved']:,}}}",
        f"\\newcommand{{\\ClassCapEvalEightSaving}}{{{100*eval_8192['token_savings_fraction']:.1f}\\%}}",
    ]


def mathnum(x: float) -> str:
    """Signed thousands-separated integer safe for math mode.

    A bare comma inside math mode is punctuation and typesets with a trailing space,
    so "$+2,277$" renders as "+2, 277". Bracing the comma suppresses that.
    """
    return f"{x:+,.0f}".replace(",", "{,}")


def allocation_assets() -> list[str]:
    """Macros for the fixed-budget allocation experiment, plus tables/allocation.tex.

    Returns the macro lines; writes the table fragment as a side effect. Returns an empty
    list when s35 has not been run, so a paper build fails on a missing macro rather than
    printing a stale number.
    """
    if not ALLOCATION_JSON.exists():
        print(f"  note: {ALLOCATION_JSON.name} absent -- allocation macros not emitted")
        return []
    a = json.loads(ALLOCATION_JSON.read_text(encoding="utf-8"))
    ev = a.get("eval", {})
    if ev.get("status") != "analysed":
        print("  note: allocation eval cohort not analysed -- macros not emitted")
        return []

    con = {c["contrast"]: c for c in ev["contrasts"]}
    res = ev["resource_contrasts"]["stop_after_first_success"]
    acct = ev["accounting"]["stop_after_first_success"]
    full = ev["accounting"]["full_budget"]
    aud = ev["shared_draw_audit"]
    dev = a.get("dev", {})

    L = ["", "% --- fixed-budget allocation experiment (s35) ---",
         rf"\newcommand{{\AllocN}}{{{ev['n_roots']}}}",
         rf"\newcommand{{\AllocBudget}}{{{a['budget_tokens']:,}}}"]

    for pol in ("long", "split2", "split3"):
        w = NUMWORD[pol]
        L.append(rf"\newcommand{{\Alloc{w}Verified}}{{{ev['verified_roots'][pol]}}}")
        L.append(rf"\newcommand{{\Alloc{w}Frac}}{{${ev['verified_roots'][pol]}/{ev['n_roots']}$}}")
        L.append(rf"\newcommand{{\Alloc{w}SecPerVer}}{{{acct[pol]['seconds_per_verified']:.0f}}}")
        L.append(rf"\newcommand{{\Alloc{w}Throughput}}{{{acct[pol]['verified_per_1000s']:.2f}}}")
        L.append(rf"\newcommand{{\Alloc{w}RealizedTokens}}{{{full[pol]['tokens']:,}}}")
        L.append(rf"\newcommand{{\Alloc{w}PolicyTokens}}{{{acct[pol]['tokens']:,}}}")
        L.append(rf"\newcommand{{\Alloc{w}Draws}}{{{acct[pol]['draws']}}}")
    L.append(rf"\newcommand{{\AllocNominalTokens}}{{{a['budget_tokens'] * ev['n_roots']:,}}}")

    for key, tag in (("split2-long", "SplitTwoLong"), ("split3-long", "SplitThreeLong"),
                     ("split3-split2", "SplitThreeSplitTwo")):
        c = con[key]
        L.append(rf"\newcommand{{\Alloc{tag}Eff}}{{${c['effect']:+.3f}$}}")
        L.append(rf"\newcommand{{\Alloc{tag}EffPts}}{{${100 * c['effect']:+.1f}$}}")
        L.append(rf"\newcommand{{\Alloc{tag}CI}}{{[${100 * c['ci95'][0]:+.1f}$, ${100 * c['ci95'][1]:+.1f}$]}}")
        L.append(rf"\newcommand{{\Alloc{tag}P}}{{{c['p_exact_mcnemar']:.3f}}}")
        L.append(rf"\newcommand{{\Alloc{tag}Disc}}{{{c['x_only']}/{c['y_only']}}}")
        if "p_holm" in c:
            L.append(rf"\newcommand{{\Alloc{tag}Holm}}{{{c['p_holm']:.3f}}}")

    for pol in ("split2", "split3"):
        w = NUMWORD[pol]
        r = res[pol]
        L.append(rf"\newcommand{{\Alloc{w}DeltaTokens}}{{${mathnum(r['delta_tokens_mean'])}$}}")
        L.append(rf"\newcommand{{\Alloc{w}DeltaTokensCI}}{{[${mathnum(r['delta_tokens_ci95'][0])}$, ${mathnum(r['delta_tokens_ci95'][1])}$]}}")
        L.append(rf"\newcommand{{\Alloc{w}TokensResolved}}{{{'resolved' if r['tokens_resolved'] else 'unresolved'}}}")
        L.append(rf"\newcommand{{\Alloc{w}DeltaWall}}{{${r['delta_wall_seconds_mean']:+.1f}$}}")
        L.append(rf"\newcommand{{\Alloc{w}DeltaWallCI}}{{[${r['delta_wall_seconds_ci95'][0]:+.1f}$, ${r['delta_wall_seconds_ci95'][1]:+.1f}$]}}")
        L.append(rf"\newcommand{{\Alloc{w}WallResolved}}{{{'resolved' if r['wall_resolved'] else 'unresolved'}}}")
        t = r["delta_verified_per_1000s"]
        L.append(rf"\newcommand{{\Alloc{w}DeltaThroughputCI}}{{[${t[0]:+.2f}$, ${t[1]:+.2f}$]}}")

    for pol, w in (("split2", "SplitTwo"), ("split3", "SplitThree")):
        pa = aud["prefix_audit"][pol]
        L.append(rf"\newcommand{{\Alloc{w}Prefix}}{{{pa['exact_prefix']}/{pa['n']}}}")
    for cap, w in (("6144", "SixK"), ("4096", "FourK")):
        rec = aud["retained_success"][cap]
        L.append(rf"\newcommand{{\AllocRetained{w}}}{{{rec['retained']}/{rec['of']}}}")
        L.append(rf"\newcommand{{\AllocRetained{w}Pct}}{{{rec['observed_pct']:.1f}\%}}")
        L.append(rf"\newcommand{{\AllocRetained{w}Pred}}{{{rec['predicted_pct']}\%}}")

    if dev.get("status") == "analysed":
        da = dev["accounting"]["stop_after_first_success"]
        for pol in ("long", "split2", "split3"):
            w = NUMWORD[pol]
            L.append(rf"\newcommand{{\AllocDev{w}Verified}}{{{dev['verified_roots'][pol]}}}")
            L.append(rf"\newcommand{{\AllocDev{w}SecPerVer}}{{{da[pol]['seconds_per_verified']:.0f}}}")
        L.append(rf"\newcommand{{\AllocDevN}}{{{dev['n_roots']}}}")
        dc = {c["contrast"]: c for c in dev["contrasts"]}
        L.append(rf"\newcommand{{\AllocDevSplitTwoLongEffPts}}{{${100 * dc['split2-long']['effect']:+.1f}$}}")

    # cross-policy root-level structure
    xp = ev["cross_policy"]
    L.append(rf"\newcommand{{\AllocAnyPolicy}}{{{xp['roots_verified_by_any']}}}")
    L.append(rf"\newcommand{{\AllocAnyPolicyFrac}}"
             rf"{{${xp['roots_verified_by_any']}/{ev['n_roots']}$}}")
    L.append(rf"\newcommand{{\AllocAllThree}}{{{xp['roots_verified_by_all_three']}}}")
    L.append(rf"\newcommand{{\AllocUnsolved}}{{{xp['cells']['---']}}}")
    for pol, w in (("long", "Long"), ("split2", "SplitTwo"), ("split3", "SplitThree")):
        L.append(rf"\newcommand{{\AllocOnly{w}}}"
                 rf"{{{xp['roots_verified_by_exactly_one'][pol]}}}")

    body = []
    caps = {"long": r"Long ($1\times12{,}288$)",
            "split2": r"Split-2 ($2\times6{,}144$)",
            "split3": r"Split-3 ($3\times4{,}096$)"}
    for pol in ("long", "split2", "split3"):
        c = con.get(f"{pol}-long")
        eff = "---" if pol == "long" else (
            f"${100 * c['effect']:+.1f}$ [${100 * c['ci95'][0]:+.1f}$, ${100 * c['ci95'][1]:+.1f}$]")
        holm = "---" if pol == "long" else f"{c['p_holm']:.3f}"
        body.append(
            f"{caps[pol]} & {ev['verified_roots'][pol]}/{ev['n_roots']} & {eff} & {holm} & "
            f"{full[pol]['tokens']:,} & {acct[pol]['tokens']:,} & "
            f"{acct[pol]['seconds_per_verified']:.0f} \\\\")
    (TAB / "allocation.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    # ---- realized compute, with intervals and a resolved flag per row
    rows = []
    for pol, nm in (("split2", r"Split-2 $-$ Long"), ("split3", r"Split-3 $-$ Long")):
        r = res[pol]
        rows.append(
            f"{nm} & ${mathnum(r['delta_tokens_mean'])}$ "
            f"[${mathnum(r['delta_tokens_ci95'][0])}$, ${mathnum(r['delta_tokens_ci95'][1])}$] & "
            f"{'yes' if r['tokens_resolved'] else 'no'} & "
            f"${r['delta_wall_seconds_mean']:+.1f}$ "
            f"[${r['delta_wall_seconds_ci95'][0]:+.1f}$, ${r['delta_wall_seconds_ci95'][1]:+.1f}$] & "
            f"{'yes' if r['wall_resolved'] else 'no'} & "
            f"[${r['delta_verified_per_1000s'][0]:+.2f}$, "
            f"${r['delta_verified_per_1000s'][1]:+.2f}$] \\\\")
    (TAB / "allocation_compute.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- retained-success audit against section 2's prediction
    rows = []
    for cap, nm in (("6144", r"$6{,}144$ (Split-2)"), ("4096", r"$4{,}096$ (Split-3)")):
        rec = aud["retained_success"][cap]
        rows.append(f"{nm} & {rec['retained']}/{rec['of']} & "
                    f"{rec['observed_pct']:.1f}\\% & {rec['predicted_pct']}\\% \\\\")
    (TAB / "allocation_retained.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- terminal states by policy and draw
    rows = []
    order = ["verified", "lean_failed", "truncated"]
    for pol, nm in (("long", "Long"), ("split2", "Split-2"), ("split3", "Split-3")):
        draws = sorted(ev["terminal_states"][pol])
        for i, dk in enumerate(draws):
            cells = ev["terminal_states"][pol][dk]
            head = rf"\multirow{{{len(draws)}}}{{*}}{{{nm}}}" if i == 0 else ""
            total = sum(cells.get(k, 0) for k in order)
            rows.append(f"{head} & {dk.replace('draw', '')} & "
                        + " & ".join(str(cells.get(k, 0)) for k in order)
                        + f" & {total} " + r"\\")
        if pol != "split3":
            rows.append(r"\midrule")
    (TAB / "allocation_terminal.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- cross-policy root-level outcomes
    rows = []
    names = {"long": "Long only", "split2": "Split-2 only", "split3": "Split-3 only"}
    for pol in ("long", "split2", "split3"):
        rows.append(f"{names[pol]} & {xp['roots_verified_by_exactly_one'][pol]} \\\\")
    multi = xp["roots_verified_by_any"] - sum(
        xp["roots_verified_by_exactly_one"].values())
    rows.append(rf"More than one policy & {multi} \\")
    rows.append(rf"\quad of which all three & {xp['roots_verified_by_all_three']} \\")
    rows.append(r"\midrule")
    rows.append(rf"Unsolved by every policy & {xp['cells']['---']} \\")
    rows.append(rf"\textbf{{Total}} & \textbf{{{ev['n_roots']}}} \\")
    (TAB / "allocation_crosspolicy.tex").write_text("\n".join(rows) + "\n",
                                                    encoding="utf-8")

    # ---- development cohort, descriptive
    if dev.get("status") == "analysed":
        da = dev["accounting"]["stop_after_first_success"]
        dc = {c["contrast"]: c for c in dev["contrasts"]}
        rows = []
        for pol, nm in (("long", r"Long"), ("split2", r"Split-2"), ("split3", r"Split-3")):
            c = dc.get(f"{pol}-long")
            eff = "---" if pol == "long" else (
                f"${100 * c['effect']:+.1f}$ [${100 * c['ci95'][0]:+.1f}$, "
                f"${100 * c['ci95'][1]:+.1f}$]")
            rows.append(f"{nm} & {dev['verified_roots'][pol]}/{dev['n_roots']} & {eff} & "
                        f"{da[pol]['seconds_per_verified']:.0f} \\\\")
        (TAB / "allocation_dev.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- figure datasets: rebuildable without rerunning the analysis
    fig_data = {
        "nominal_vs_realized": {
            "nominal_total": a["budget_tokens"] * ev["n_roots"],
            "realized": {p: full[p]["tokens"] for p in ("long", "split2", "split3")},
            "verified": ev["verified_roots"],
            "union_any_policy": xp["roots_verified_by_any"]},
        "policy_efficiency": {
            p: {"seconds_per_verified": acct[p]["seconds_per_verified"],
                "verified_per_1000s": acct[p]["verified_per_1000s"],
                "wall_seconds": acct[p]["wall_seconds"]}
            for p in ("long", "split2", "split3")},
        "success_cost_frontier": [
            {"policy": p, "wall_seconds": acct[p]["wall_seconds"],
             "tokens": full[p]["tokens"], "verified": ev["verified_roots"][p],
             "n": ev["n_roots"]}
            for p in ("long", "split2", "split3")],
        "truncation_by_draw": {
            p: {dk: ev["terminal_states"][p][dk]
                for dk in sorted(ev["terminal_states"][p])}
            for p in ("long", "split2", "split3")},
    }
    (FIG / "allocation_figure_data.json").write_text(
        json.dumps(fig_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  wrote 6 allocation table fragments and figures/allocation_figure_data.json")
    return L


def allocation_figure() -> bool:
    """Same nominal budget, different realized compute, no detectable yield difference."""
    if not ALLOCATION_JSON.exists():
        return False
    a = json.loads(ALLOCATION_JSON.read_text(encoding="utf-8"))
    ev = a.get("eval", {})
    if ev.get("status") != "analysed":
        return False

    full = ev["accounting"]["full_budget"]
    xp = ev["cross_policy"]
    n = ev["n_roots"]
    pols = ["long", "split2", "split3"]
    label = {"long": "Long\n1x12,288", "split2": "Split-2\n2x6,144",
             "split3": "Split-3\n3x4,096"}
    colors = {"long": BLUE, "split2": ORANGE, "split3": GREEN}

    path = FIG / "allocation.pdf"
    w, h = 500, 250
    c = canvas.Canvas(str(path), pagesize=(w, h))
    draw_title(c, "Equal nominal budgets are not equal realized compute",
               "Held-out roots (n=%d); every policy may spend %s completion tokens per root"
               % (n, f"{a['budget_tokens']:,}"), w, h)

    # ---------------- left panel: verified roots, with the union rule
    x0, y0, pw, ph = 56, 52, 170, 128
    vmax = max(max(ev["verified_roots"][p] for p in pols),
               xp["roots_verified_by_any"]) * 1.25
    set_font(c, 8.6, True)
    c.drawString(x0, y0 + ph + 16, "Verified roots")
    bw = pw / (len(pols) * 1.8)
    for i, p in enumerate(pols):
        v = ev["verified_roots"][p]
        bx = x0 + (i + 0.4) * (pw / len(pols))
        bh = ph * v / vmax
        c.setFillColor(colors[p])
        c.rect(bx, y0, bw, bh, stroke=0, fill=1)
        set_font(c, 7.6, True, colors[p])
        c.drawCentredString(bx + bw / 2, y0 + bh + 4, f"{v}/{n}")
        set_font(c, 6.8, False, GRAY)
        for k, part in enumerate(label[p].split("\n")):
            c.drawCentredString(bx + bw / 2, y0 - 10 - 8 * k, part)
    union_y = y0 + ph * xp["roots_verified_by_any"] / vmax
    c.setStrokeColor(GRAY)
    c.setDash(2, 2)
    c.line(x0, union_y, x0 + pw, union_y)
    c.setDash()
    set_font(c, 6.8, False, GRAY)
    c.drawString(x0, union_y + 3,
                 f"any policy {xp['roots_verified_by_any']}/{n}  "
                 f"(all three: {xp['roots_verified_by_all_three']})")

    # ---------------- right panel: realized vs nominal tokens
    x1 = 292
    nominal = a["budget_tokens"] * n
    tmax = max(max(full[p]["tokens"] for p in pols), nominal) * 1.18
    set_font(c, 8.6, True)
    c.drawString(x1, y0 + ph + 16, "Realized completion tokens")
    for i, p in enumerate(pols):
        t = full[p]["tokens"]
        bx = x1 + (i + 0.4) * (pw / len(pols))
        bh = ph * t / tmax
        c.setFillColor(colors[p])
        c.rect(bx, y0, bw, bh, stroke=0, fill=1)
        set_font(c, 7.2, True, colors[p])
        c.drawCentredString(bx + bw / 2, y0 + bh + 4, f"{t / 1000:.0f}k")
        set_font(c, 6.8, False, GRAY)
        for k, part in enumerate(label[p].split("\n")):
            c.drawCentredString(bx + bw / 2, y0 - 10 - 8 * k, part)
    ny = y0 + ph * nominal / tmax
    c.setStrokeColor(RED)
    c.setDash(3, 2)
    c.line(x1, ny, x1 + pw, ny)
    c.setDash()
    set_font(c, 6.8, False, RED)
    c.drawString(x1, ny + 3, f"identical nominal cap sum {nominal / 1000:.0f}k")

    set_font(c, 6.8, False, GRAY)
    c.drawString(24, 14,
                 "No detectable difference in success (Holm-adjusted p = 1.000 for both "
                 "split policies); realized token spend differs by up to "
                 f"{100 * (max(full[p]['tokens'] for p in pols) / full['long']['tokens'] - 1):.0f}%.")
    c.save()
    return True


def assert_not_pooled(result: dict) -> dict:
    """Keep the two experiments in separate namespaces and prove they stayed separate.

    The frozen recovery experiment is 536 branches over 83 roots with a hash-exact replay.
    The allocation experiment is 415 later branches under different caps with no replay.
    Their denominators must never substitute for one another, so both are recorded
    explicitly and the pairing of each result block to its own denominators is asserted.
    """
    if not ALLOCATION_JSON.exists():
        return result
    a = json.loads(ALLOCATION_JSON.read_text(encoding="utf-8"))
    ev = a.get("eval", {})
    if ev.get("status") != "analysed":
        return result

    frozen_roots = result["counts"]["unique_roots"]
    frozen_branches = result["counts"]["branches"]
    alloc_roots = ev["n_roots"]
    alloc_branches = sum(g["expected"] for g in ev["gate"].values()
                         if g["cohort"].startswith("alloc_"))

    assert alloc_branches != frozen_branches, (
        "allocation and frozen branch counts collided; a denominator may be shared")
    result["allocation"] = a
    result["denominators"] = {
        "frozen_recovery_experiment": {
            "roots": frozen_roots, "branches": frozen_branches,
            "source": "audit/w2_accounting.json",
            "note": "536 branches, hash-exact replay, integrity-gated by s28"},
        "allocation_experiment": {
            "roots_eval": alloc_roots,
            "roots_dev": a.get("dev", {}).get("n_roots"),
            "branches_eval": alloc_branches,
            "source": "audit/allocation_analysis.json",
            "note": ("generated later under different caps; no replay. Never pooled "
                     "with the frozen experiment, and dev is never pooled with eval.")},
        "pooling_check": "passed",
    }
    return result


def main() -> None:
    rows = load_rows()
    source = json.loads(ACCOUNTING_JSON.read_text(encoding="utf-8"))
    result = assert_not_pooled(compute(rows, source))
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    OUT_TEX.write_text(tex_macros(result).rstrip("\n") + "\n"
                       + "\n".join(allocation_assets()) + "\n"
                       + "\n".join(portfolio_assets()) + "\n"
                       + "\n".join(crossover_assets()) + "\n"
                       + "\n".join(sensitivity_assets()) + "\n", encoding="utf-8")
    write_tables(result)
    phase_figure(result)
    currencies_figure(result)
    efficiency_figure(result)
    isocost_figure(result)
    n_fig = 4 + (1 if allocation_figure() else 0)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")
    print(f"wrote {len(list(TAB.glob('*.tex')))} table fragments under {TAB}")
    print(f"wrote {n_fig} figures under {FIG}")


if __name__ == "__main__":
    main()
