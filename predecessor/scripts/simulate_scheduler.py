#!/usr/bin/env python3
"""Budget-aware retry scheduling: cross-fitted replay over recorded plain-retry draws.

Question: given a fixed retry token budget, does allocating the next attempt by a
retryability index rho = p_hat / c_hat (class-level, with attempt-number decay and a
repeated-identical-failure penalty) beat sequential pass@k (uniform round-robin)?

Data: the four genuine plain (R0) draws at offsets +500k/+530k/+540k/+550k. Replay:
policies choose WHICH unsolved theorem consumes its next recorded attempt; attempt
order within a theorem follows a seed ordering sigma (all 24 permutations reported).
Replay can spend at most 4 attempts per theorem -- recorded-data ceiling; the
hindsight oracle shows how binding that is.

Policies:
  uniform   -- sequential pass@k: draw 1 for every unsolved theorem, then draw 2, ...
  random    -- uniformly random unsolved theorem each step (5 RNG seeds averaged)
  rules     -- frozen class rules: never retry timeout; truncated capped at 2
               attempts; others up to 4; priority by dev-fit class rho
  scheduler -- calibrated dynamic rho with hierarchical shrinkage (class -> global),
               attempt-number decay and repeated-error-hash penalty, cross-fitted
  oracle    -- hindsight: knows recorded outcomes, spends only on winning prefixes
               (greedy by cost; upper bound)

Freeze discipline: parameters are fit ONLY on the dev split (data/scheduler_split.json,
stratified 43/40 by failure class, seed 30). Curves are reported for dev (leave-one-
seed-out cross-fitted) and for the held-out 40 (parameters from dev; theorems unseen).
The held-out numbers on RECORDED draws are still retrospective data -- the prospective
test requires fresh draws on the held-out 40 (see EXPERIMENT_LOG).

Usage:
  python scripts/simulate_scheduler.py            # dev + held-out replay on 4 draws
  python scripts/simulate_scheduler.py --draws-dir runs/... (prospective fresh draws)
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PLAIN_DRAWS = [
    ("R0a", "runs/epp_confirmatory_202/evaluation/cascade_attempts.jsonl"),
    ("R0b", "runs/epp_confirmatory_202_matched_seeds/evaluation/cascade_attempts.jsonl"),
    ("R0c", "runs/epp_confirmatory_202_router_seed540/evaluation/cascade_attempts.jsonl"),
    ("R0d", "runs/epp_confirmatory_202_seed550_confirm/evaluation/cascade_attempts.jsonl"),
]
SHRINK_PSEUDO = 8.0          # class p_hat shrinkage toward global (pseudo-attempts)
REPEAT_HASH_PENALTY_FLOOR = 0.25


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tokens_of(row: dict) -> int:
    return ((row.get("response") or {}).get("usage") or {}).get("completion_tokens") or 0


def failure_hash(row: dict) -> str:
    # identical failed proof OR identical error text head counts as "same failure"
    err = ((row.get("lean_stdout") or "") + (row.get("lean_stderr") or ""))[:400]
    return f"{row.get('proof_sha256')}|{hash(err)}"


def load_attempts(draws: list[tuple[str, str]]) -> dict[str, dict[str, dict]]:
    """theorem -> draw_label -> attempt record (plain R0 rows only, no dedup rows)."""
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for label, rel in draws:
        for r in read_jsonl(ROOT / rel):
            if r["arm"] != "R0" or r.get("deduplicated_from"):
                continue
            out[r["problem_id"]][label] = {
                "verified": bool(r["verified"]),
                "tokens": tokens_of(r),
                "cls": r["stage1_error_class"],
                "fhash": failure_hash(r),
            }
    return dict(out)


# ---------------- parameter fitting (dev theorems, chosen draws only) ----------------

def fit_params(attempts: dict[str, dict[str, dict]], theorems: list[str], fit_draws: list[str]) -> dict:
    per_class = collections.defaultdict(lambda: [0, 0, 0])   # v, n, tok
    per_attemptno = collections.defaultdict(lambda: [0, 0])  # v, n by attempt index
    repeat = [0, 0]     # after repeated identical failure: v, n
    fresh = [0, 0]      # after a changed failure: v, n
    for pid in theorems:
        seen_hashes: list[str] = []
        for k, d in enumerate(fit_draws):
            a = attempts.get(pid, {}).get(d)
            if not a:
                continue
            cell = per_class[a["cls"]]
            cell[0] += a["verified"]; cell[1] += 1; cell[2] += a["tokens"]
            pa = per_attemptno[k]
            pa[0] += a["verified"]; pa[1] += 1
            if k > 0 and seen_hashes:
                bucket = repeat if a["fhash"] == seen_hashes[-1] else fresh
                # outcome of THIS attempt conditioned on relation to previous failure
                bucket[0] += a["verified"]; bucket[1] += 1
            if not a["verified"]:
                seen_hashes.append(a["fhash"])
    g_v = sum(c[0] for c in per_class.values())
    g_n = sum(c[1] for c in per_class.values())
    g_p = g_v / g_n if g_n else 0.1
    g_c = sum(c[2] for c in per_class.values()) / max(1, g_n)
    p_class = {c: (v + SHRINK_PSEUDO * g_p) / (n + SHRINK_PSEUDO) for c, (v, n, _) in per_class.items()}
    c_class = {c: (t / n if n else g_c) for c, (v, n, t) in per_class.items()}
    # attempt-number decay relative to attempt 0, shrunk toward 1
    base = per_attemptno[0][0] / per_attemptno[0][1] if per_attemptno[0][1] else g_p
    decay = {}
    for k, (v, n) in per_attemptno.items():
        raw = (v + SHRINK_PSEUDO * base) / (n + SHRINK_PSEUDO)
        decay[k] = max(0.05, raw / base) if base else 1.0
    rep_rate = (repeat[0] + SHRINK_PSEUDO * g_p) / (repeat[1] + SHRINK_PSEUDO)
    fresh_rate = (fresh[0] + SHRINK_PSEUDO * g_p) / (fresh[1] + SHRINK_PSEUDO)
    repeat_penalty = max(REPEAT_HASH_PENALTY_FLOOR, rep_rate / fresh_rate if fresh_rate else 1.0)
    return {"p_class": p_class, "c_class": c_class, "g_p": g_p, "g_c": g_c,
            "decay": decay, "repeat_penalty": min(1.0, repeat_penalty),
            "class_order": sorted(p_class, key=lambda c: -(p_class[c] / max(1.0, c_class[c])))}


# ---------------- replay engine ----------------

def replay(attempts, theorems, sigma, policy, params, rng=None):
    """Returns list of (cumulative_tokens, cumulative_solves) step points.

    Internal state is keyed by INSTANCE INDEX, not theorem ID, so `theorems`
    may contain duplicates (bootstrap resamples with replacement): each copy
    is an independent instance replaying the same recorded attempts.
    """
    ids = list(theorems)
    instances = list(range(len(ids)))
    state = {i: {"next": 0, "solved": False, "hashes": [], "cls": None} for i in instances}
    for i in instances:
        for d in sigma:
            a = attempts.get(ids[i], {}).get(d)
            if a:
                state[i]["cls"] = a["cls"]
                break
    seq = {i: [attempts[ids[i]][d] for d in sigma if d in attempts.get(ids[i], {})] for i in instances}

    def available(p):
        return not state[p]["solved"] and state[p]["next"] < len(seq[p])

    def rho(p):
        st = state[p]
        cls = st["cls"]
        pr = params["p_class"].get(cls, params["g_p"]) * params["decay"].get(st["next"], 1.0)
        if len(st["hashes"]) >= 2 and st["hashes"][-1] == st["hashes"][-2]:
            pr *= params["repeat_penalty"]
        return pr / max(1.0, params["c_class"].get(cls, params["g_c"]))

    points = [(0, 0)]
    spent = solves = 0

    if policy == "oracle":
        costs = []
        for p in instances:
            win = None
            acc = 0
            for a in seq[p]:
                acc += a["tokens"]
                if a["verified"]:
                    win = acc
                    break
            if win is not None:
                costs.append(win)
        for c in sorted(costs):
            spent += c; solves += 1
            points.append((spent, solves))
        return points

    order = instances
    while True:
        if policy == "uniform":
            cands = [p for p in order if available(p)]
            if not cands:
                break
            pick = min(cands, key=lambda p: (state[p]["next"], p))
        elif policy == "random":
            cands = [p for p in order if available(p)]
            if not cands:
                break
            pick = rng.choice(cands)
        elif policy == "rules":
            def cap(p):
                cls = state[p]["cls"]
                return 0 if cls == "timeout" else (2 if cls == "truncated" else 4)
            cands = [p for p in order if available(p) and state[p]["next"] < cap(p)]
            if not cands:
                break
            corder = params["class_order"]
            pick = min(cands, key=lambda p: (state[p]["next"], corder.index(state[p]["cls"]) if state[p]["cls"] in corder else 99))
        elif policy == "scheduler":
            cands = [p for p in order if available(p)]
            if not cands:
                break
            pick = max(cands, key=rho)
        else:
            raise ValueError(policy)
        a = seq[pick][state[pick]["next"]]
        state[pick]["next"] += 1
        spent += a["tokens"]
        if a["verified"]:
            state[pick]["solved"] = True
            solves += 1
        else:
            state[pick]["hashes"].append(a["fhash"])
        points.append((spent, solves))
    return points


def solves_at(points, budget):
    s = 0
    for t, v in points:
        if t <= budget:
            s = v
        else:
            break
    return s


def tokens_to_reach(points, target):
    for t, v in points:
        if v >= target:
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-draws", nargs="*", default=None,
                    help="label=path pairs of fresh prospective draws to replay instead of the recorded four")
    ap.add_argument("--split", default="data/scheduler_split.json")
    args = ap.parse_args()

    split = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
    dev, held = split["dev"], split["held_out"]

    draw_defs = PLAIN_DRAWS if not args.fresh_draws else [tuple(x.split("=", 1)) for x in args.fresh_draws]
    attempts = load_attempts(draw_defs)
    labels = [l for l, _ in draw_defs]
    prospective = bool(args.fresh_draws)

    # In prospective mode the frozen parameters MUST come from the recorded four
    # draws on dev theorems -- the fresh files contain only held-out theorems.
    fit_attempts = attempts if not prospective else load_attempts(PLAIN_DRAWS)
    fit_labels = labels if not prospective else [l for l, _ in PLAIN_DRAWS]

    def run_block(theorems, fit_on_dev_all_draws, block_name):
        print(f"\n===== {block_name}: {len(theorems)} theorems, draws {labels} =====")
        # budget axis: fraction of the uniform policy's full spend
        results = collections.defaultdict(list)   # policy -> per-sigma curves
        for sigma in itertools.permutations(labels):
            if fit_on_dev_all_draws:
                params = fit_params(fit_attempts, dev, list(fit_labels))  # frozen: dev, recorded draws
            else:
                params = fit_params(attempts, dev, [d for d in labels if d != sigma[0]])  # LOSO on replayed first draw
            for pol in ("uniform", "random", "rules", "scheduler", "oracle"):
                if pol == "random":
                    for rs in range(5):
                        results[pol].append(replay(attempts, theorems, sigma, pol, params, rng=random.Random(1000 + rs)))
                else:
                    results[pol].append(replay(attempts, theorems, sigma, pol, params))
        full_uniform = [max(t for t, _ in c) for c in results["uniform"]]
        u_solves = [c[-1][1] for c in results["uniform"]]
        B = statistics.mean(full_uniform)
        target = statistics.mean(u_solves)
        print(f"uniform full spend B = {B:,.0f} retry tokens; uniform solves = {target:.2f} (mean over {len(results['uniform'])} orderings)")
        print(f"{'policy':<11}{'solves@1.00B':>13}{'solves@0.75B':>13}{'solves@0.50B':>13}{'tok to match uniform':>22}")
        for pol in ("uniform", "random", "rules", "scheduler", "oracle"):
            curves = results[pol]
            s100 = statistics.mean(solves_at(c, B) for c in curves)
            s75 = statistics.mean(solves_at(c, 0.75 * B) for c in curves)
            s50 = statistics.mean(solves_at(c, 0.50 * B) for c in curves)
            reach = [tokens_to_reach(c, round(target)) for c in curves]
            hit = [r for r in reach if r is not None]
            frac = len(hit) / len(reach)
            reach_txt = f"{statistics.mean(hit):,.0f} ({frac:.0%} reach)" if hit else "never"
            print(f"{pol:<11}{s100:>13.2f}{s75:>13.2f}{s50:>13.2f}{reach_txt:>22}")
        # paired bootstrap at parity: scheduler tokens-to-target vs uniform, resampling theorems
        rng = random.Random(30)
        params_frozen = fit_params(fit_attempts, dev, list(fit_labels))
        savings = []
        for _ in range(1000):
            sample = [rng.choice(theorems) for _ in theorems]
            sig = labels  # canonical ordering for bootstrap
            cu = replay(attempts, sample, sig, "uniform", params_frozen)
            cs = replay(attempts, sample, sig, "scheduler", params_frozen)
            tgt = cu[-1][1]
            tu, ts = cu[-1][0], tokens_to_reach(cs, tgt)
            if ts is not None and tu:
                savings.append(1 - ts / tu)
        if savings:
            savings.sort()
            lo, mid, hi = savings[25], statistics.median(savings), savings[974]
            reachp = len(savings) / 1000
            print(f"scheduler token savings at uniform-parity: median {mid:.1%} [95% CI {lo:.1%}, {hi:.1%}] "
                  f"(parity reached in {reachp:.0%} of resamples)")
        # pre-registered equivalence check: paired bootstrap CI of (uniform - policy)
        # solves at uniform's full budget, on the primary (canonical) ordering.
        for pol in ("rules", "scheduler"):
            diffs = []
            for _ in range(1000):
                sample = [rng.choice(theorems) for _ in theorems]
                cu = replay(attempts, sample, labels, "uniform", params_frozen)
                cp = replay(attempts, sample, labels, pol, params_frozen)
                b = cu[-1][0]
                diffs.append(solves_at(cu, b) - solves_at(cp, b))
            diffs.sort()
            lo_d, hi_d = diffs[25], diffs[974]
            verdict = "NEAR-PARITY (CI < 1.0 margin)" if hi_d < 1.0 else "no parity claim"
            print(f"equivalence {pol} vs uniform at full budget: (uniform - {pol}) solves "
                  f"mean {statistics.mean(diffs):+.2f}, 95% CI [{lo_d:+.2f}, {hi_d:+.2f}] -> {verdict}")

    if not prospective:
        run_block(dev, fit_on_dev_all_draws=False, block_name="DEV (cross-fitted LOSO)")
    run_block(held, fit_on_dev_all_draws=True,
              block_name=("HELD-OUT 40, FRESH PROSPECTIVE DRAWS" if prospective else "HELD-OUT 40 (recorded draws, params frozen from dev)"))


if __name__ == "__main__":
    main()
