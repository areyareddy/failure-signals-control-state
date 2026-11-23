"""Descriptive matched-budget slices for the inherited frozen scheduler result.

The predecessor's registered endpoint is tokens-to-uniform-yield.  This CPU-only audit
adds the complementary practitioner view: at the same token budget, how many held-out
roots have uniform and the frozen class rules solved?  It does not relabel this post-hoc
analysis as confirmatory or co-primary.

    python scripts\s44_scheduler_budget_slices.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLM = ROOT.parent / "CoLM_Jul2026_MoSpe_Navi"
SIM_PATH = COLM / "scripts" / "simulate_scheduler.py"
OUT = ROOT / "audit" / "scheduler_budget_slices.json"

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


def load_simulator():
    spec = importlib.util.spec_from_file_location("colm_simulate_scheduler", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spend_at(points: list[tuple[int, int]], budget: int) -> int:
    return max(token for token, _ in points if token <= budget)


def main() -> int:
    if not SIM_PATH.exists():
        raise SystemExit(f"accepted predecessor checkout not found: {COLM}")
    sim = load_simulator()

    # simulate_scheduler.py resolves its ROOT at import time, so the historical
    # development fit and both prospective blocks are read from the accepted checkout.
    split = json.loads((COLM / "data" / "scheduler_split.json").read_text(encoding="utf-8"))
    held = split["held_out"]
    historical = sim.load_attempts(sim.PLAIN_DRAWS)
    historical_labels = [label for label, _ in sim.PLAIN_DRAWS]
    params = sim.fit_params(historical, split["dev"], historical_labels)

    result = {
        "status": "descriptive_post_hoc_complement_to_registered_equal_yield_endpoint",
        "root_count": len(held),
        "blocks": {},
    }
    for name, draw_defs in BLOCKS.items():
        attempts = sim.load_attempts(draw_defs)
        labels = [label for label, _ in draw_defs]
        uniform = sim.replay(attempts, held, labels, "uniform", params)
        rules = sim.replay(attempts, held, labels, "rules", params)
        calibrated = sim.replay(attempts, held, labels, "scheduler", params)
        target = uniform[-1][1]
        tu = sim.tokens_to_reach(uniform, target)
        tr = sim.tokens_to_reach(rules, target)
        tc = sim.tokens_to_reach(calibrated, target)
        if tu is None or tr is None or tc is None:
            raise AssertionError(f"{name}: primary ordering does not reach parity")
        full_budget = uniform[-1][0]
        slices = []
        for fraction in (0.25, 0.50, 0.75, 1.00):
            budget = round(full_budget * fraction)
            slices.append({
                "fraction_uniform_full_spend": fraction,
                "budget_tokens": budget,
                "uniform_solves": sim.solves_at(uniform, budget),
                "rules_solves": sim.solves_at(rules, budget),
                "calibrated_solves": sim.solves_at(calibrated, budget),
                "rules_tokens_issued": spend_at(rules, budget),
            })
        result["blocks"][name] = {
            "draw_labels_primary_order": labels,
            "uniform_full_spend_tokens": full_budget,
            "uniform_full_spend_solves": target,
            "rules_exhausted_spend_tokens": rules[-1][0],
            "rules_exhausted_solves": rules[-1][1],
            "calibrated_tokens_to_target": tc,
            "calibrated_savings_vs_uniform_tokens_to_target": 1 - tc / tu,
            "unissued_at_uniform_full_spend": full_budget - rules[-1][0],
            "registered_equal_yield": {
                "uniform_tokens_to_target": tu,
                "rules_tokens_to_target": tr,
                "target_solves": target,
                "savings": 1 - tr / tu,
            },
            "matched_budget_slices": slices,
        }

    # Guard against provenance drift from the accepted manuscript's published endpoint.
    assert result["blocks"]["block1"]["registered_equal_yield"]["uniform_tokens_to_target"] == 734095
    assert result["blocks"]["block1"]["registered_equal_yield"]["rules_tokens_to_target"] == 532270
    assert result["blocks"]["block2"]["registered_equal_yield"]["uniform_tokens_to_target"] == 996402
    assert result["blocks"]["block2"]["registered_equal_yield"]["rules_tokens_to_target"] == 658880

    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    for name, block in result["blocks"].items():
        print(f"{name}: full-budget rules/uniform solves "
              f"{block['rules_exhausted_solves']}/{block['uniform_full_spend_solves']}; "
              f"unissued {block['unissued_at_uniform_full_spend']:,} tokens")
        for row in block["matched_budget_slices"]:
            print(f"  {row['fraction_uniform_full_spend']:.0%}: "
                  f"rules {row['rules_solves']} vs uniform {row['uniform_solves']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
