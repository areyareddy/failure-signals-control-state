"""Retrospective class-conditional cap sensitivity using exact same-seed prefixes.

For roots whose *first* attempt was truncated, cap the one-draw Long retry at a shorter
length; leave every other failure class at 12,288.  The allocation audit established that
same-seed shorter draws are exact prefixes, so success can only be retained when Long's
verified completion already ended before the shorter cap.  This is an exploratory CPU
counterfactual, not a prospective or blinded policy evaluation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs.sqlite"
ROOTS = ROOT / "data" / "roots.jsonl"
OUT = ROOT / "audit" / "class_conditional_cap.json"
COHORTS = {"dev": "full_pool_dev", "eval": "primary_eval"}
CAPS = (4096, 6144, 8192)


def main() -> int:
    meta = {
        row["root_id"]: row
        for row in (
            json.loads(line)
            for line in ROOTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    con = sqlite3.connect(f"file:{RUNS}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = {
        "status": "retrospective_exploratory_counterfactual_not_preregistered",
        "policy": "cap stage1-truncated roots only; leave all other roots at 12288",
        "prefix_basis": "allocation_analysis.json exact-prefix audit",
        "cohorts": {},
    }
    for label, cohort in COHORTS.items():
        rows = list(con.execute(
            "SELECT root_id, completion_tokens, verified FROM branches "
            "WHERE cohort=? AND arm='A' AND status='done' ORDER BY root_id",
            (cohort,),
        ))
        if len(rows) not in (40, 43):
            raise AssertionError(f"{cohort}: expected 40 or 43 rows, found {len(rows)}")
        base_tokens = sum(row["completion_tokens"] or 0 for row in rows)
        base_verified = sum(row["verified"] for row in rows)
        n_truncated_roots = sum(
            meta[row["root_id"]]["fine_failure_class"] == "truncated" for row in rows
        )
        variants = []
        for cap in CAPS:
            tokens = 0
            verified = 0
            for row in rows:
                is_trunc = meta[row["root_id"]]["fine_failure_class"] == "truncated"
                effective_cap = cap if is_trunc else 12288
                used = min(row["completion_tokens"] or 0, effective_cap)
                tokens += used
                verified += int(bool(row["verified"]) and
                                (row["completion_tokens"] or 0) <= effective_cap)
            variants.append({
                "truncated_root_cap": cap,
                "verified": verified,
                "verified_change": verified - base_verified,
                "realized_completion_tokens": tokens,
                "tokens_saved": base_tokens - tokens,
                "token_savings_fraction": (base_tokens - tokens) / base_tokens,
            })
        out["cohorts"][label] = {
            "n": len(rows),
            "stage1_truncated_roots": n_truncated_roots,
            "long_verified": base_verified,
            "long_realized_completion_tokens": base_tokens,
            "variants": variants,
        }
    con.close()

    # Mechanically choose the smallest candidate preserving Long's dev solve count.
    dev = out["cohorts"]["dev"]
    eligible = [row for row in dev["variants"] if row["verified_change"] == 0]
    out["dev_selected_cap"] = min(row["truncated_root_cap"] for row in eligible) if eligible else None
    selected = out["dev_selected_cap"]
    out["eval_at_dev_selected_cap"] = next(
        row for row in out["cohorts"]["eval"]["variants"]
        if row["truncated_root_cap"] == selected
    ) if selected is not None else None

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    print(f"dev-selected cap: {selected}")
    print(json.dumps(out["eval_at_dev_selected_cap"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
