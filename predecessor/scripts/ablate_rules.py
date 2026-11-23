"""Component ablation of the frozen rules scheduler (offline, CPU-only).

Decomposes the three rules and adds ranking-only baselines, replayed on the
same recorded draws as the main analysis. All variants are allocation-only
and share the frozen dev-fit parameters (p_class, c_class, class order) —
nothing is refit. Self-check: the 'rules' variant must reproduce the
validated tokens-to-count numbers.

Variants:
  uniform        pass@k baseline (no caps, no priority)
  timeout_only   uniform + never retry timeout class
  trunccap_only  uniform + truncation retries capped at 2
  coarse         timeout skip + truncation cap, NO class priority
  classorder     class-priority tie-break only, NO caps (rho = p/c ranking)
  p_only         priority by class success rate p alone, full caps
  cost_only      priority by ascending class cost c alone, full caps
  rules          the full frozen scheduler (timeout skip + trunc cap + rho order)

Usage: python scripts/ablate_rules.py --name goedel_block1 --draws L0=path ... L3=path
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_scheduler import (  # noqa: E402
    PLAIN_DRAWS, ROOT, load_attempts, fit_params, tokens_to_reach,
)


def replay_variant(attempts, theorems, labels, params, caps_on, timeout_skip, trunc_cap, order_key):
    state = {t: {"next": 0, "solved": False} for t in theorems}
    cls = {t: next(iter(attempts[t].values()))["cls"] for t in theorems}
    seq = {t: [l for l in labels if l in attempts[t]] for t in theorems}
    order = sorted(theorems)
    idx = {t: i for i, t in enumerate(order)}

    def cap(t):
        c = cls[t]
        if timeout_skip and c == "timeout":
            return 0
        if trunc_cap is not None and c == "truncated":
            return trunc_cap
        return 4

    points = [(0, 0)]
    spent = solves = 0
    while True:
        cands = [t for t in order if not state[t]["solved"]
                 and state[t]["next"] < min(len(seq[t]), cap(t) if caps_on else 4)]
        if not cands:
            break
        pick = min(cands, key=lambda t: (state[t]["next"], order_key(cls[t]), idx[t]))
        lab = seq[pick][state[pick]["next"]]
        state[pick]["next"] += 1
        a = attempts[pick][lab]
        spent += a["tokens"]
        if a["verified"]:
            state[pick]["solved"] = True
            solves += 1
        points.append((spent, solves))
    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--draws", nargs=4, required=True)
    args = ap.parse_args()

    draw_defs = [tuple(x.split("=", 1)) for x in args.draws]
    labels = [l for l, _ in draw_defs]
    attempts = load_attempts(draw_defs)
    theorems = sorted(attempts)
    split = json.loads((ROOT / "data/scheduler_split.json").read_text(encoding="utf-8"))
    params = fit_params(load_attempts(PLAIN_DRAWS), split["dev"], [l for l, _ in PLAIN_DRAWS])
    corder = params["class_order"]
    p_rank = sorted(params["p_class"], key=lambda c: -params["p_class"][c])
    c_rank = sorted(params["c_class"], key=lambda c: params["c_class"][c])

    def key_none(c):
        return 0

    def key_in(rank):
        return lambda c: rank.index(c) if c in rank else 99

    variants = {
        "uniform":       dict(caps_on=False, timeout_skip=False, trunc_cap=None, order_key=key_none),
        "timeout_only":  dict(caps_on=True, timeout_skip=True, trunc_cap=None, order_key=key_none),
        "trunccap_only": dict(caps_on=True, timeout_skip=False, trunc_cap=2, order_key=key_none),
        "coarse":        dict(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_none),
        "classorder":    dict(caps_on=False, timeout_skip=False, trunc_cap=None, order_key=key_in(corder)),
        "p_only":        dict(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_in(p_rank)),
        "cost_only":     dict(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_in(c_rank)),
        "rules":         dict(caps_on=True, timeout_skip=True, trunc_cap=2, order_key=key_in(corder)),
    }

    # primary + all 24 orderings
    out = {"name": args.name, "variants": {}}
    u_primary = replay_variant(attempts, theorems, labels, params, **variants["uniform"])
    target = u_primary[-1][1]
    Tu = tokens_to_reach(u_primary, target)
    print(f"===== ablation {args.name}: uniform primary = {target} solves, Tu = {Tu:,} =====")
    print(f"{'variant':<15}{'primary tokens':>15}{'deficit':>8}{'savings':>9}{'parity/24':>11}{'med savings':>13}")
    for name, cfg in variants.items():
        prim = replay_variant(attempts, theorems, labels, params, **cfg)
        tv = tokens_to_reach(prim, target)
        deficit = target - prim[-1][1]
        sav = (1 - tv / Tu) if tv is not None else None
        parities, savs = 0, []
        for sigma in itertools.permutations(labels):
            cu = replay_variant(attempts, theorems, list(sigma), params, **variants["uniform"])
            cv = replay_variant(attempts, theorems, list(sigma), params, **cfg)
            tgt = cu[-1][1]
            tu = tokens_to_reach(cu, tgt)
            tvv = tokens_to_reach(cv, tgt)
            if tvv is not None:
                parities += 1
                if tu:
                    savs.append(1 - tvv / tu)
        med = statistics.median(savs) if savs else None
        tv_txt = f"{tv:,}" if tv is not None else "never"
        print(f"{name:<15}{tv_txt:>15}{deficit:>8}{(f'{sav:.1%}' if sav is not None else '-'):>9}"
              f"{parities:>8}/24{(f'{med:.1%}' if med is not None else '-'):>13}")
        out["variants"][name] = {"primary_tokens": tv, "primary_deficit": deficit,
                                 "primary_savings": sav, "parity_orderings": parities,
                                 "median_savings": med}
    out["uniform_target"] = target
    out["Tu"] = Tu
    path = ROOT / f"reports/ablation_{args.name}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
