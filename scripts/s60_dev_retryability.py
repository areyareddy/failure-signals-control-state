#!/usr/bin/env python3
"""Recompute the failure-class retryability ledger split by dev/held-out.

Motivation. v8's Table 3 is captioned "Development retryability ledger used to
motivate the frozen class order", but its cells sum to 83 theorems / 332 attempts
-- the whole failure pool, not the 43 development roots. The table was transcribed
from the predecessor manuscript rather than recomputed. This script settles whether
that is a labelling error or an actual held-out leak, by recomputing the ledger from
the predecessor's recorded draws under each root set, and by refitting the class
ordering the frozen rules use.

Read-only with respect to the predecessor tree: it opens four JSONL draw files and
the split file, and writes only into this project's audit/.

The fitting logic (SHRINK_PSEUDO, rho = p_class / max(1, c_class), descending order)
is transcribed from the predecessor's scripts/simulate_scheduler.py:fit_params so
that the recomputed ordering is comparable to the frozen one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# Location of the predecessor project, which holds the four recorded retry-draw files and
# the frozen dev/held-out split. Read-only: this script never writes there. Override with
# --predecessor or the PREDECESSOR_DIR environment variable; the default assumes the
# predecessor sits beside this project, which is how the original run was laid out.
DEFAULT_PRED = HERE.parent / "CoLM_Jul2026_MoSpe_Navi"
PRED = Path(os.environ.get("PREDECESSOR_DIR", DEFAULT_PRED))

SHRINK_PSEUDO = 8.0

PLAIN_DRAWS = [
    ("R0a", "runs/epp_confirmatory_202/evaluation/cascade_attempts.jsonl"),
    ("R0b", "runs/epp_confirmatory_202_matched_seeds/evaluation/cascade_attempts.jsonl"),
    ("R0c", "runs/epp_confirmatory_202_router_seed540/evaluation/cascade_attempts.jsonl"),
    ("R0d", "runs/epp_confirmatory_202_seed550_confirm/evaluation/cascade_attempts.jsonl"),
]

# v8 tables/retryability.tex, for the reproduction check.
PUBLISHED_83 = {
    "unknown identifier": (2, 8, 4), "other": (6, 24, 9), "unsolved goals": (18, 72, 14),
    "linarith failed": (17, 68, 12), "simp failed": (2, 8, 1), "truncated": (32, 128, 15),
    "timeout": (6, 24, 0),
}


def tokens_of(row):
    return ((row.get("response") or {}).get("usage") or {}).get("completion_tokens") or 0


def load_attempts():
    out = collections.defaultdict(dict)
    for label, rel in PLAIN_DRAWS:
        path = PRED / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["arm"] != "R0" or r.get("deduplicated_from"):
                continue
            out[r["problem_id"]][label] = {
                "verified": bool(r["verified"]),
                "tokens": tokens_of(r),
                "cls": r["stage1_error_class"],
            }
    return dict(out)


def ledger(attempts, roots):
    """Class-level ledger over the given roots, all four recorded draws."""
    per = collections.defaultdict(lambda: {"theorems": set(), "attempts": 0, "rescues": 0, "tokens": 0})
    for pid in roots:
        for a in attempts.get(pid, {}).values():
            c = per[a["cls"]]
            c["theorems"].add(pid)
            c["attempts"] += 1
            c["rescues"] += int(a["verified"])
            c["tokens"] += a["tokens"]
    rows = []
    for cls, c in per.items():
        n, tok = c["attempts"], c["tokens"]
        rows.append({
            "class": cls,
            "theorems": len(c["theorems"]),
            "attempts": n,
            "rescues": c["rescues"],
            "rescue_probability": round(c["rescues"] / n, 3) if n else 0.0,
            "mean_completion_tokens": round(tok / n) if n else 0,
            "rescues_per_token_e5": round(c["rescues"] / tok * 1e5, 2) if tok else 0.0,
        })
    rows.sort(key=lambda r: -r["rescues_per_token_e5"])
    tot_n = sum(r["attempts"] for r in rows)
    tot_v = sum(r["rescues"] for r in rows)
    tot_t = sum(per[r["class"]]["tokens"] for r in rows)
    tot_thm = sum(r["theorems"] for r in rows)
    rows.append({
        "class": "all", "theorems": tot_thm, "attempts": tot_n,
        "rescues": tot_v,
        "rescue_probability": round(tot_v / tot_n, 3) if tot_n else 0.0,
        "mean_completion_tokens": round(tot_t / tot_n) if tot_n else 0,
        "rescues_per_token_e5": round(tot_v / tot_t * 1e5, 2) if tot_t else 0.0,
    })
    return rows


def wilson_upper(v: int, n: int, z: float = 1.959963984540054) -> float:
    """Upper Wilson bound on an attempt-level rate. Attempts cluster within theorems,
    so the paper reports this as descriptive and leans on draw-order parity instead."""
    if n == 0:
        return 1.0
    p = v / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return centre + margin


def class_order(attempts, roots):
    """Frozen-rule class ordering refit on `roots` (predecessor fit_params logic)."""
    per = collections.defaultdict(lambda: [0, 0, 0])
    for pid in roots:
        for a in attempts.get(pid, {}).values():
            cell = per[a["cls"]]
            cell[0] += a["verified"]
            cell[1] += 1
            cell[2] += a["tokens"]
    g_v = sum(c[0] for c in per.values())
    g_n = sum(c[1] for c in per.values())
    g_p = g_v / g_n if g_n else 0.1
    p_class = {c: (v + SHRINK_PSEUDO * g_p) / (n + SHRINK_PSEUDO) for c, (v, n, _) in per.items()}
    c_class = {c: (t / n if n else 0.0) for c, (v, n, t) in per.items()}
    rho = {c: p_class[c] / max(1.0, c_class[c]) for c in p_class}
    return sorted(rho, key=lambda c: -rho[c]), rho


def main():
    global PRED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predecessor", type=Path, default=PRED,
                    help="path to the predecessor project holding the recorded draws")
    PRED = ap.parse_args().predecessor
    if not (PRED / "data/scheduler_split.json").exists():
        raise SystemExit(f"predecessor project not found at {PRED}; pass --predecessor")

    split = json.loads((PRED / "data/scheduler_split.json").read_text(encoding="utf-8"))
    dev, held = split["dev"], split["held_out"]
    assert len(dev) == 43 and len(held) == 40, (len(dev), len(held))
    all83 = dev + held
    assert len(set(all83)) == 83

    attempts = load_attempts()
    missing = [p for p in all83 if len(attempts.get(p, {})) != 4]
    assert not missing, f"roots without 4 recorded draws: {missing[:5]}"

    led_all = ledger(attempts, all83)
    led_dev = ledger(attempts, dev)
    led_held = ledger(attempts, held)

    # Reproduction check against the published all-83 table.
    repro = {}
    for row in led_all:
        if row["class"] in PUBLISHED_83:
            want = PUBLISHED_83[row["class"]]
            got = (row["theorems"], row["attempts"], row["rescues"])
            repro[row["class"]] = {"published": list(want), "recomputed": list(got),
                                   "match": want == got}
    all_match = all(v["match"] for v in repro.values())

    ord_dev, rho_dev = class_order(attempts, dev)
    ord_all, rho_all = class_order(attempts, all83)

    # Derived quantities the prose needs, so no number is ever hand-typed.
    def row(led, cls):
        return next(r for r in led if r["class"] == cls)

    def rho_exact(r):
        tok = r["attempts"] * r["mean_completion_tokens"]
        return r["rescues"] / tok * 1e5 if tok else 0.0

    dev_to = row(led_dev, "timeout")
    all_to = row(led_all, "timeout")
    derived = {
        "dev_timeout_wilson_upper": wilson_upper(dev_to["rescues"], dev_to["attempts"]),
        "all_timeout_wilson_upper": wilson_upper(all_to["rescues"], all_to["attempts"]),
        "dev_timeout_roots": dev_to["theorems"],
        "dev_retry_spread": rho_exact(row(led_dev, "unknown_identifier")) / rho_exact(row(led_dev, "truncated")),
        "all_retry_spread": rho_exact(row(led_all, "unknown_identifier")) / rho_exact(row(led_all, "truncated")),
        # The class whose rank moves furthest between the two fits -- the sharpest
        # evidence that the all-root ledger did not select the frozen order.
        "rank_disagreement_class": max(
            ord_dev, key=lambda c: abs(ord_dev.index(c) - ord_all.index(c))),
        "n_classes_reranked": sum(
            1 for c in ord_dev if ord_dev.index(c) != ord_all.index(c)),
        "rank_in_dev_fit": None,
        "rank_in_all_fit": None,
    }
    dcls = derived["rank_disagreement_class"]
    derived["rank_in_dev_fit"] = ord_dev.index(dcls) + 1
    derived["rank_in_all_fit"] = ord_all.index(dcls) + 1

    # The decisive check: the order the predecessor actually froze and shipped must be
    # the dev-only refit, and must NOT be the all-root refit. If this ever fails, the
    # held-out characterisation is wrong and the paper must be reframed, not relabelled.
    frozen = json.loads(
        (PRED / "reports" / "transfer_analysis_validation_block1.json").read_text(encoding="utf-8")
    )["frozen_class_order"]
    derived["frozen_class_order"] = frozen
    derived["frozen_matches_dev_fit"] = (frozen == ord_dev)
    derived["frozen_matches_all_fit"] = (frozen == ord_all)
    assert derived["frozen_matches_dev_fit"], (
        "shipped frozen order does not match the development-only refit", frozen, ord_dev)
    assert not derived["frozen_matches_all_fit"], (
        "development and all-root fits agree, so this evidence does not separate them")

    out = {
        "derived": derived,
        "purpose": "settle whether v8 Table 3 is a labelling error or a held-out leak",
        "split": {"source": "predecessor data/scheduler_split.json", "seed": split.get("seed"),
                  "stratified_by": split.get("stratified_by"),
                  "n_dev": len(dev), "n_held_out": len(held)},
        "reproduces_published_83_root_table": all_match,
        "reproduction_detail": repro,
        "ledger_all83": led_all,
        "ledger_dev43": led_dev,
        "ledger_heldout40": led_held,
        "class_order_fit_on_dev43": ord_dev,
        "class_order_fit_on_all83": ord_all,
        "class_order_identical": ord_dev == ord_all,
        "rho_dev43": {k: round(v, 5) for k, v in rho_dev.items()},
        "rho_all83": {k: round(v, 5) for k, v in rho_all.items()},
    }
    dest = HERE / "audit" / "dev_retryability.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"reproduces published 83-root table: {all_match}")
    for cls, d in repro.items():
        flag = "ok" if d["match"] else "MISMATCH"
        print(f"  {cls:20s} published={d['published']} recomputed={d['recomputed']} {flag}")
    print()
    for name, led in (("DEV 43", led_dev), ("HELD-OUT 40", led_held), ("ALL 83", led_all)):
        print(f"--- {name} ---")
        for r in led:
            print(f"  {r['class']:20s} thm={r['theorems']:3d} att={r['attempts']:4d} "
                  f"resc={r['rescues']:3d} p={r['rescue_probability']:.3f} "
                  f"tok={r['mean_completion_tokens']:5d} rho={r['rescues_per_token_e5']:5.1f}")
    print()
    print("class order fit on dev-43 :", ord_dev)
    print("class order fit on all-83 :", ord_all)
    print("identical:", ord_dev == ord_all)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
