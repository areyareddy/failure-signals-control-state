#!/usr/bin/env python3
"""Added S1 control analyses for W2 (CPU only, read-only).

Everything here replays the predecessor's recorded draws through its own frozen
simulator (`simulate_scheduler.py`) and ablation engine (`ablate_rules.py`); nothing
is refit and no draw is generated.  Outputs go to audit/cr_s1_controls.json.

It computes:
  1. the held-out per-class ledger (historical draws, and each evaluation seed block);
  2. class-independent controls: a uniform two-retry limit for every root;
  3. the definition of savings in orderings that never reach uniform's solve count,
     and whether those orderings enter the ordering medians;
  4. how often a first-attempt-truncated root truncates again on retry.
We also recompute the theorem-cluster bootstrap that the predecessor reported, so the
paper can cite a recomputed rather than transcribed interval.
"""
from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The predecessor's recorded draws and simulator.  In the released artifact they sit in
# predecessor/; in the original working tree the predecessor project is a sibling folder.
# PREDECESSOR_DIR overrides both.
_bundled = ROOT / "predecessor"
PRED = Path(os.environ.get("PREDECESSOR_DIR",
                           _bundled if _bundled.exists() else ROOT.parent / "CoLM_Jul2026_MoSpe_Navi"))
OUT = ROOT / "audit" / "cr_s1_controls.json"
N_BOOT = 10_000

BLOCKS = {
    "block1": [
        ("F0", "runs/epp_prospective40_seed56/evaluation/cascade_attempts.jsonl"),
        ("F1", "runs/epp_prospective40_seed57/evaluation/cascade_attempts.jsonl"),
        ("F2", "runs/epp_prospective40_seed58/evaluation/cascade_attempts.jsonl"),
        ("F3", "runs/epp_prospective40_seed59/evaluation/cascade_attempts.jsonl"),
    ],
    "block2": [
        ("D0", "runs/epp_replication40_d0/evaluation/cascade_attempts.jsonl"),
        ("D1", "runs/epp_replication40_d1/evaluation/cascade_attempts.jsonl"),
        ("D2", "runs/epp_replication40_d2/evaluation/cascade_attempts.jsonl"),
        ("D3", "runs/epp_replication40_d3/evaluation/cascade_attempts.jsonl"),
    ],
}


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PRED / "scripts" / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PRED / "scripts"))
    spec.loader.exec_module(mod)
    return mod


sim = load("pred_sim", "simulate_scheduler.py")
abl = load("pred_abl", "ablate_rules.py")


def outcome_states(draws):
    """problem_id -> label -> outcome_state, for truncation persistence."""
    out = collections.defaultdict(dict)
    for label, rel in draws:
        for line in (PRED / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["arm"] != "R0" or r.get("deduplicated_from"):
                continue
            out[r["problem_id"]][label] = r["outcome_state"]
    return out


def ledger(attempts, roots):
    per = collections.defaultdict(lambda: {"theorems": set(), "attempts": 0, "rescues": 0, "tokens": 0})
    for pid in roots:
        for a in attempts.get(pid, {}).values():
            c = per[a["cls"]]
            c["theorems"].add(pid)
            c["attempts"] += 1
            c["rescues"] += int(a["verified"])
            c["tokens"] += a["tokens"]
    rows = {}
    tot = {"theorems": 0, "attempts": 0, "rescues": 0, "tokens": 0}
    for cls, c in per.items():
        rows[cls] = {"theorems": len(c["theorems"]), "attempts": c["attempts"], "rescues": c["rescues"],
                     "tokens": c["tokens"],
                     "p": c["rescues"] / c["attempts"],
                     "rho_e5": c["rescues"] / c["tokens"] * 1e5 if c["tokens"] else 0.0}
        tot["theorems"] += len(c["theorems"])
        for k in ("attempts", "rescues", "tokens"):
            tot[k] += c[k]
    tot["p"] = tot["rescues"] / tot["attempts"]
    tot["rho_e5"] = tot["rescues"] / tot["tokens"] * 1e5
    rows["all"] = tot
    return rows


def persistence(attempts, states, roots):
    """Retry truncation rate split by first-attempt class (truncated vs. not)."""
    cell = {"stage1_truncated": [0, 0, set()], "stage1_other": [0, 0, set()]}
    for pid in roots:
        for lab, a in attempts.get(pid, {}).items():
            key = "stage1_truncated" if a["cls"] == "truncated" else "stage1_other"
            trunc = states[pid][lab] == "truncated"
            cell[key][0] += trunc
            cell[key][1] += 1
            if trunc:
                cell[key][2].add(pid)
    out = {}
    for k, (t, n, s) in cell.items():
        out[k] = {"truncated_attempts": t, "attempts": n, "rate": t / n if n else None,
                  "roots_with_any_truncated_retry": len(s)}
    return out


# Class-independent controls.  ablate_rules caps every class at 4 when caps_on is True,
# so a uniform k-retry limit is expressed with a cap function wrapper.
def replay_uniform_cap(attempts, theorems, labels, k, timeout_skip=False):
    state = {t: {"next": 0, "solved": False} for t in theorems}
    cls = {t: next(iter(attempts[t].values()))["cls"] for t in theorems}
    seq = {t: [l for l in labels if l in attempts[t]] for t in theorems}
    order = sorted(theorems)
    idx = {t: i for i, t in enumerate(order)}

    def cap(t):
        return 0 if (timeout_skip and cls[t] == "timeout") else k

    points = [(0, 0)]
    spent = solves = 0
    while True:
        cands = [t for t in order if not state[t]["solved"] and state[t]["next"] < min(len(seq[t]), cap(t))]
        if not cands:
            break
        pick = min(cands, key=lambda t: (state[t]["next"], idx[t]))
        lab = seq[pick][state[pick]["next"]]
        state[pick]["next"] += 1
        a = attempts[pick][lab]
        spent += a["tokens"]
        if a["verified"]:
            state[pick]["solved"] = True
            solves += 1
        points.append((spent, solves))
    return points


def variant_fns(params):
    corder = params["class_order"]
    key_none = lambda c: 0  # noqa: E731
    key_in = lambda c: corder.index(c) if c in corder else 99  # noqa: E731

    def mk(**cfg):
        return lambda att, th, lab: abl.replay_variant(att, th, lab, params, **cfg)

    return {
        "uniform": mk(caps_on=False, timeout_skip=False, trunc_cap=None, order_key=key_none),
        "timeout_only": mk(caps_on=True, timeout_skip=True, trunc_cap=None, order_key=key_none),
        "trunccap_only": mk(caps_on=True, timeout_skip=False, trunc_cap=2, order_key=key_none),
        "coarse": mk(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_none),
        "rules": mk(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_in),
        # class-independent controls
        "uniform_cap1": lambda att, th, lab: replay_uniform_cap(att, th, lab, 1),
        "uniform_cap2": lambda att, th, lab: replay_uniform_cap(att, th, lab, 2),
        "uniform_cap3": lambda att, th, lab: replay_uniform_cap(att, th, lab, 3),
        "timeout_plus_uniform_cap2": lambda att, th, lab: replay_uniform_cap(att, th, lab, 2, timeout_skip=True),
        "timeout_plus_uniform_cap3": lambda att, th, lab: replay_uniform_cap(att, th, lab, 3, timeout_skip=True),
    }


def per_ordering(fns, attempts, theorems, labels):
    rows = []
    for sigma in itertools.permutations(labels):
        sigma = list(sigma)
        cu = fns["uniform"](attempts, theorems, sigma)
        target = cu[-1][1]
        tu = sim.tokens_to_reach(cu, target)
        u_full = cu[-1][0]
        row = {"ordering": "".join(s[-1] for s in sigma), "primary": sigma == labels,
               "uniform_target": target, "uniform_tokens_to_target": tu, "uniform_full_spend": u_full,
               "variants": {}}
        for name, fn in fns.items():
            if name == "uniform":
                continue
            cv = fn(attempts, theorems, sigma)
            tv = sim.tokens_to_reach(cv, target)
            row["variants"][name] = {
                "final_solves": cv[-1][1],
                "deficit": target - cv[-1][1],
                "tokens_to_target": tv,
                "saving_at_target": (1 - tv / tu) if tv is not None else None,
                "exhaustion_tokens": cv[-1][0],
                "exhaustion_saving_vs_uniform_full": 1 - cv[-1][0] / u_full,
                "solves_at_uniform_tokens_to_target": sim.solves_at(cv, tu),
            }
        rows.append(row)
    return rows


def summarise(rows, name):
    prim = next(r for r in rows if r["primary"])["variants"][name]
    parity = [r for r in rows if r["variants"][name]["tokens_to_target"] is not None]
    sav_parity = [r["variants"][name]["saving_at_target"] for r in parity]
    exh = [r["variants"][name]["exhaustion_saving_vs_uniform_full"] for r in rows]
    deficits = collections.Counter(r["variants"][name]["deficit"] for r in rows)
    return {
        "primary_saving": prim["saving_at_target"],
        "primary_deficit": prim["deficit"],
        "primary_exhaustion_saving": prim["exhaustion_saving_vs_uniform_full"],
        "parity_orderings": len(parity),
        "median_saving_over_parity_orderings": statistics.median(sav_parity) if sav_parity else None,
        "min_saving_over_parity_orderings": min(sav_parity) if sav_parity else None,
        "median_exhaustion_saving_all24": statistics.median(exh),
        "deficit_histogram_all24": {str(k): v for k, v in sorted(deficits.items())},
        "mean_final_solves_all24": statistics.mean(r["variants"][name]["final_solves"] for r in rows),
        "mean_uniform_target_all24": statistics.mean(r["uniform_target"] for r in rows),
    }


def bootstrap(attempts, theorems, labels, params, policies, seed=30):
    """Theorem-cluster bootstrap on the primary ordering, using the instance-keyed
    simulator so that duplicated theorems are independent instances."""
    rng = random.Random(seed)
    res = {p: [] for p in policies}
    unreached = {p: 0 for p in policies}
    for _ in range(N_BOOT):
        sample = [rng.choice(theorems) for _ in theorems]
        cu = sim.replay(attempts, sample, labels, "uniform", params)
        tgt = cu[-1][1]
        tu = sim.tokens_to_reach(cu, tgt)
        for p in policies:
            cp = sim.replay(attempts, sample, labels, p, params)
            tp = sim.tokens_to_reach(cp, tgt)
            if tp is None or not tu:
                unreached[p] += 1
                continue
            res[p].append(1 - tp / tu)
    out = {}
    for p, v in res.items():
        v.sort()
        n = len(v)
        out[p] = {"n_reached": n, "n_unreached": unreached[p],
                  "median": statistics.median(v), "ci_low": v[int(0.025 * n)], "ci_high": v[int(0.975 * n) - 1]}
    return out


def main():
    split = json.loads((PRED / "data/scheduler_split.json").read_text(encoding="utf-8"))
    dev, held = split["dev"], split["held_out"]
    hist = sim.load_attempts(sim.PLAIN_DRAWS)
    hist_states = outcome_states(sim.PLAIN_DRAWS)
    params = sim.fit_params(hist, dev, [l for l, _ in sim.PLAIN_DRAWS])

    out = {
        "status": "added control analyses; post hoc and descriptive unless stated",
        "class_order_formula": {
            "rho_class": "p_shrunk / max(1, mean completion tokens of class)",
            "p_shrunk": "(rescues + 8 * global_rate) / (attempts + 8), fit on 43 dev roots x 4 historical draws",
            "frozen_order": params["class_order"],
            "p_shrunk_values": params["p_class"],
            "mean_tokens": params["c_class"],
            "global_rate": params["g_p"],
            "rho_e5": {c: params["p_class"][c] / max(1.0, params["c_class"][c]) * 1e5 for c in params["p_class"]},
        },
        "ledgers": {
            "historical_dev43": ledger(hist, dev),
            "historical_heldout40": ledger(hist, held),
        },
        "persistence": {"historical_dev43": persistence(hist, hist_states, dev),
                        "historical_heldout40": persistence(hist, hist_states, held)},
        "blocks": {},
    }

    fns = variant_fns(params)
    for bname, draws in BLOCKS.items():
        att = sim.load_attempts(draws)
        st = outcome_states(draws)
        labels = [l for l, _ in draws]
        theorems = sorted(att)
        assert len(theorems) == 40 and set(theorems) == set(held)
        out["ledgers"][bname] = ledger(att, theorems)
        out["persistence"][bname] = persistence(att, st, theorems)
        rows = per_ordering(fns, att, theorems, labels)
        # self-check against the inherited ablation table
        rep = json.loads((PRED / f"reports/ablation_goedel_{bname}.json").read_text(encoding="utf-8"))
        for v in ("timeout_only", "trunccap_only", "coarse", "rules"):
            s = summarise(rows, v)
            want = rep["variants"][v]
            assert abs((s["primary_saving"] or 0) - (want["primary_savings"] or 0)) < 1e-12, (bname, v)
            assert s["parity_orderings"] == want["parity_orderings"], (bname, v)
            assert abs(s["median_saving_over_parity_orderings"] - want["median_savings"]) < 1e-12, (bname, v)
        out["blocks"][bname] = {
            "summary": {v: summarise(rows, v) for v in fns if v != "uniform"},
            "per_ordering": rows,
            "bootstrap_primary_ordering": bootstrap(att, theorems, labels, params, ["rules"]),
        }
        print(f"== {bname}")
        for v, s in out["blocks"][bname]["summary"].items():
            ps = s["primary_saving"]
            md = s["median_saving_over_parity_orderings"]
            print(f"  {v:28s} prim={'-' if ps is None else f'{ps:6.1%}'} def={s['primary_deficit']} "
                  f"parity={s['parity_orderings']:2d}/24 med|par={'-' if md is None else f'{md:6.1%}'} "
                  f"exh_med24={s['median_exhaustion_saving_all24']:6.1%} deficits={s['deficit_histogram_all24']}")
        print("  bootstrap:", out["blocks"][bname]["bootstrap_primary_ordering"])

    for k, v in out["ledgers"].items():
        print(f"-- ledger {k}")
        for c, r in v.items():
            print(f"   {c:20s} thm={r['theorems']:3d} att={r['attempts']:4d} resc={r['rescues']:3d} "
                  f"p={r['p']:.3f} rho={r['rho_e5']:.2f}")
    print("persistence:", json.dumps(out["persistence"], indent=1))
    print("class order:", out["class_order_formula"]["frozen_order"])
    print({c: round(v, 3) for c, v in out["class_order_formula"]["rho_e5"].items()})
    OUT.write_text(json.dumps(out, indent=2, default=list), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
