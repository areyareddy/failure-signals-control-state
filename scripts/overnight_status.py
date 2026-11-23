"""Read-only progress view for the overnight queue. Safe to run at any time.

    python scripts\\overnight_status.py
    python scripts\\overnight_status.py --watch 120

Reads the databases directly rather than trusting the driver's state file, so it stays
truthful even if the driver has died. It opens every database read-only and never writes,
so it cannot disturb a running job.

DELIBERATELY DOES NOT SHOW OUTCOMES. Counts of done/pending/failed only -- no verified
counts, no per-arm rates. The allocation experiment's dev thresholds must be frozen before
its eval outcomes are seen (spec/ALLOCATION_SPEC.md), and a progress monitor that quietly
reports success rates would make that impossible to honour. Use the analysis scripts, after
the queue finishes, to look at outcomes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import receipts  # noqa: E402
import overnight_queue as q  # noqa: E402

STATE = ROOT / "audit/overnight_queue.json"


def bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return " " * width
    n = int(width * done / total)
    return "#" * n + "." * (width - n)


def render() -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    alive = q.server_alive(3.0)
    print(f"\n  overnight queue @ {now}    server: "
          f"{'UP' if alive else 'DOWN'}")
    if state.get("deadline_utc"):
        left = state.get("seconds_left", 0)
        print(f"  deadline {state['deadline_utc']}   "
              f"driver last wrote {state.get('updated_utc', '?')}")
    print()
    print(f"  {'job':20} {'cap':>6}  {'progress':26} {'done':>9}  status")
    print("  " + "-" * 74)

    grand_done = grand_total = 0
    for jid, db, _, cohort, cap, expected, note in q.JOBS:
        p = ROOT / db
        c = {}
        if p.exists():
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                c = dict(con.execute(
                    "SELECT status, count(*) FROM branches WHERE cohort=? "
                    "GROUP BY status", (cohort,)))
            finally:
                con.close()
        done = c.get("done", 0)
        run = c.get("running", 0)
        failed = c.get("failed", 0)
        grand_done += done
        grand_total += expected
        if done >= expected:
            status = "complete"
        elif run:
            status = "RUNNING"
        elif done:
            status = "partial"
        else:
            status = "queued"
        if failed:
            status += f"  ({failed} failed)"
        print(f"  {jid:20} {cap:6d}  [{bar(done, expected)}] "
              f"{done:4d}/{expected:<4d}  {status}")

    print("  " + "-" * 74)
    pct = 100 * grand_done / grand_total if grand_total else 0
    print(f"  {'TOTAL':20} {'':6}  [{bar(grand_done, grand_total)}] "
          f"{grand_done:4d}/{grand_total:<4d}  {pct:.0f}%")

    hist = state.get("history") or []
    if hist:
        print("\n  finished jobs:")
        for h in hist:
            print(f"    {h['job']:20} {h['outcome']:16} {h.get('elapsed', '')}")
    print("\n  (outcomes intentionally not shown -- see the spec's inspection rule)\n")


def main(argv: list[str]) -> int:
    receipts.force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds")
    args = ap.parse_args(argv)
    while True:
        render()
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
