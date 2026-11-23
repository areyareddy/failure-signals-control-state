"""Step s57 -- drive the cap-policy run in the frozen execution order, with markers.

Reads audit/cap_policy_plan.json and invokes the FROZEN scripts/s12_run.py once per
cohort, passing that cohort's cap via --max-tokens (the runner takes one cap per
invocation, s12_run.py:69, and is not edited). A telemetry marker brackets every cohort so
energy can be attributed later without assuming anything about ordering.

Always --blind: cohort names contain `eval`, outcomes must not reach stdout before the
frozen analysis opens them.

    python scripts/s57_run_cap_policy.py --canary            # first Long+Short pair only
    python scripts/s57_run_cap_policy.py --from-step 2       # resume the remainder
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = ROOT / "audit/cap_policy_plan.json"
DB = ROOT / "data/cap_policy.sqlite"
PROMPTS = ROOT / "data/cap_policy_prompts.jsonl"
TAG = "cap_policy"


def mark(label: str, note: str = "") -> None:
    subprocess.run([sys.executable, str(HERE / "s55_nvml_logger.py"), "mark",
                    "--tag", TAG, "--label", label, "--note", note],
                   check=False, capture_output=True)


def done_count() -> int:
    import sqlite3
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = con.execute("SELECT count(*) FROM branches WHERE status='done'").fetchone()[0]
    con.close()
    return n


def pending_count(cohort: str, block: int) -> int:
    import sqlite3
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = con.execute("SELECT count(*) FROM branches WHERE cohort=? AND "
                    "seed_block=? AND status!='done'", (cohort, block)).fetchone()[0]
    con.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true",
                    help="run only the first long/short chunk pair, then stop")
    ap.add_argument("--from-step", type=int, default=0)
    ap.add_argument("--to-step", type=int, default=10**6)
    args = ap.parse_args()

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    steps = plan["execution_order"]

    if args.canary:
        # the first chunk pair: one long cohort and one short cohort on the same 5 roots
        sel = [s for s in steps if s["cohort"].endswith(("_c1",))][:2]
    else:
        sel = [s for i, s in enumerate(steps)
               if args.from_step <= i <= args.to_step]

    print(f"{len(sel)} cohort(s) to run:")
    for s in sel:
        print(f"  {s['cohort']:28} cap={s['cap']:>5}  n={s['n_roots']}")

    t_all = time.time()
    n0 = done_count()
    for s in sel:
        cohort, cap, block = s["cohort"], s["cap"], s["block"]
        mark(f"{cohort}_start", f"cap={cap}")
        t0 = time.time()
        cmd = [sys.executable, str(HERE / "s12_run.py"),
               "--db", str(DB), "--prompts", str(PROMPTS),
               "--cohort", cohort, "--block", str(block),
               "--max-tokens", str(cap), "--blind"]
        before = pending_count(cohort, block)
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        after = pending_count(cohort, block)
        if before and after == before:
            print(f"  {cohort:28} SILENT NO-OP: {before} pending before and "
                  f"after (block={block}). Refusing to continue.")
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            return 3
        mark(f"{cohort}_end", f"cap={cap} seconds={dt:.1f}")
        ok = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"  {cohort:28} {ok}  {dt / 60:.1f} min")
        if r.returncode != 0:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            return r.returncode

    total = time.time() - t_all
    n1 = done_count()
    made = n1 - n0
    print(f"\n{made} branches in {total / 60:.1f} min"
          + (f"  ({total / made:.1f} s/branch)" if made else ""))
    if args.canary and made:
        remaining = 165 - n1
        proj = remaining * (total / made)
        print(f"\nPROJECTION")
        print(f"  measured           : {total / made:.1f} s/branch over {made} branches")
        print(f"  remaining branches : {remaining}")
        print(f"  projected remaining: {proj / 3600:.2f} GPU-h")
        print(f"  projected TOTAL    : {(total + proj) / 3600:.2f} GPU-h"
              f"  (hard limit 3.00)")
        if (total + proj) / 3600 > 3.0:
            print("  VERDICT: EXCEEDS the 3 GPU-hour limit. Stop and report; do not "
                  "reduce roots or blocks to fit.")
            return 2
        print("  VERDICT: within budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
