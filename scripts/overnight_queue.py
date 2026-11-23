"""Unattended GPU queue for the four staged jobs. One slot, strictly sequential.

    python scripts\\overnight_queue.py --hours 10
    python scripts\\overnight_queue.py --hours 10 --dry-run

WHAT IT RUNS, IN ORDER. Every job is a cohort in an already-staged database, so this
script chooses nothing scientific -- the design was frozen by s30/s31/s32 before any call.

    1  cprime_b2        C' seed block 2               40 calls   W1, pre-registered trigger
    2  alloc *_eval     fixed-budget allocation      200 calls   W2, primary endpoint
    3  alloc *_dev      same policies, dev roots     215 calls   W2, router thresholds
    4  restart_*        restart-permissive arms      120 calls   W1, limitation (7)

Ordering rules, and why:
  * C' block 2 is first because it is the cheapest and the most certain to be wanted; if
    everything after it fails, the highest-value result is still banked.
  * Eval cohorts precede dev cohorts. Generation order does not threaten validity --
    inspection order does -- so eval goes first purely so that a truncated window costs
    the router analysis rather than the primary endpoint.
  * Within the restart block, C-R and C'-R precede A-R, which is the rate anchor and the
    intended casualty if time runs out.

RESUMABILITY IS THE WHOLE SAFETY MODEL. s12_run.py claims one branch at a time in its own
transaction and returns an in-flight branch to 'pending' on the next start, so a crash, a
power cut or this script being killed costs at most one branch. Every job below is
therefore safe to re-enter, and this driver simply re-enters it.

WHAT IT WILL NOT DO. It never rebuilds a database, never edits a spec, and never runs an
analysis. When the queue finishes or the budget expires it stops, leaving the databases to
be analysed deliberately.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import receipts  # noqa: E402

STATE = ROOT / "audit/overnight_queue.json"
LOGDIR = ROOT / "audit/overnight"
BASE_URL = "http://127.0.0.1:8080"
MAX_ATTEMPTS = 4            # per job, across the whole window
SERVER_GRACE_S = 180        # how long to wait for the server to come back

# Cohorts under an inspection rule: their OUTCOMES must not reach a human-readable
# artifact before that cohort's analysis is frozen. s12 runs with --blind for these, so
# the job log under audit/overnight/ carries progress only.
#
# This set exists because it was once empty. On 2026-08-29 the allocation eval cohorts
# were generated unblinded and their per-arm verified counts were written to
# audit/overnight/alloc_split2_eval.log and alloc_split3_eval.log, which voided the
# hash-before-inspection guarantee of spec/ALLOCATION_SPEC.md section 7 and cost the
# router baseline. Add a cohort here BEFORE it is generated, not after.
BLIND_COHORTS = {
    "alloc_split2_eval",
    "alloc_split3_eval",
}

# A cohort whose name contains "eval" but which is deliberately NOT blinded must be
# named here with the reason. Anything unclassified aborts the queue at startup --
# see require_blinding_classification(). Default-deny, because what actually failed on
# 2026-08-29 was not a wrong decision but a forgotten one.
UNBLINDED_OK = {
    "eval_cprime_b2":
        "spec/CPRIME_SPEC.md froze the contrasts, the tests, the Holm family and the "
        "interpretation table before block 1 existed, and block 2 reuses that spec "
        "unchanged. There is no threshold to derive and nothing left to choose after "
        "seeing the outcome, so a plaintext log leaks no discretion.",
}


def require_blinding_classification() -> None:
    """Abort unless every eval cohort in JOBS is explicitly blinded or excused."""
    unclassified = sorted(
        c for _, _, _, c, _, _, _ in JOBS
        if "eval" in c and c not in BLIND_COHORTS and c not in UNBLINDED_OK)
    if unclassified:
        raise SystemExit(
            "refusing to start: these eval cohorts are neither in BLIND_COHORTS nor "
            "in UNBLINDED_OK:\n  " + "\n  ".join(unclassified) +
            "\n\nAdd each to BLIND_COHORTS, or to UNBLINDED_OK with the reason it "
            "needs no blinding. An eval cohort's outcomes reach a plaintext log the "
            "moment it runs, and that cannot be undone afterwards.")
    src = (HERE / "s12_run.py").read_text(encoding="utf-8")
    if BLIND_COHORTS and ('"--blind"' not in src or "args.blind" not in src):
        raise SystemExit(
            "refusing to start: BLIND_COHORTS is non-empty but scripts/s12_run.py does "
            "not implement --blind. Run scripts/s33_blinding_regression.py.")

# id, db, prompts, cohort, max_tokens, expected branches, note
JOBS = [
    ("cprime_b2", "data/cprime_b2.sqlite", "data/cprime_b2_prompts.jsonl",
     "eval_cprime_b2", 12288, 40, "W1 C' seed block 2 (pre-registered trigger)"),

    ("alloc_split2_eval", "data/alloc_split2.sqlite", "data/alloc_split2_prompts.jsonl",
     "alloc_split2_eval", 6144, 80, "W2 allocation 2x6144, held-out"),
    ("alloc_split3_eval", "data/alloc_split3.sqlite", "data/alloc_split3_prompts.jsonl",
     "alloc_split3_eval", 4096, 120, "W2 allocation 3x4096, held-out"),

    ("alloc_split2_dev", "data/alloc_split2.sqlite", "data/alloc_split2_prompts.jsonl",
     "alloc_split2_dev", 6144, 86, "W2 allocation 2x6144, dev"),
    ("alloc_split3_dev", "data/alloc_split3.sqlite", "data/alloc_split3_prompts.jsonl",
     "alloc_split3_dev", 4096, 129, "W2 allocation 3x4096, dev"),

    ("restart_C", "data/restart.sqlite", "data/restart_prompts.jsonl",
     "restart_C", 12288, 40, "W1 restart-permissive C-R"),
    ("restart_Cp", "data/restart.sqlite", "data/restart_prompts.jsonl",
     "restart_Cp", 12288, 40, "W1 restart-permissive C'-R"),
    ("restart_A", "data/restart.sqlite", "data/restart_prompts.jsonl",
     "restart_A", 12288, 40, "W1 restart-permissive A-R (drop first if short)"),
]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hms(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with (LOGDIR / "queue.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def server_alive(timeout: float = 10.0) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/tokenize", method="POST",
                data=json.dumps({"content": "x"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False


def wait_for_server(budget_s: float = SERVER_GRACE_S) -> bool:
    """The watchdog owns restarts; this only waits for one to take effect."""
    t0 = time.time()
    while time.time() - t0 < budget_s:
        if server_alive():
            return True
        log(f"    server down; waiting ({hms(budget_s - (time.time() - t0))} left)")
        time.sleep(15)
    return server_alive()


def counts(db: Path, cohort: str) -> dict[str, int]:
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return dict(con.execute(
            "SELECT status, count(*) FROM branches WHERE cohort=? GROUP BY status",
            (cohort,)))
    finally:
        con.close()


def remaining(db: Path, cohort: str) -> int:
    c = counts(db, cohort)
    return sum(v for k, v in c.items() if k != "done")


def write_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")


def snapshot(deadline: float, current: str | None, history: list) -> dict:
    jobs = []
    for jid, db, _, cohort, cap, expected, note in JOBS:
        c = counts(ROOT / db, cohort)
        jobs.append({"id": jid, "cohort": cohort, "max_tokens": cap,
                     "expected": expected, "done": c.get("done", 0),
                     "pending": c.get("pending", 0), "running": c.get("running", 0),
                     "failed": c.get("failed", 0), "note": note})
    return {"updated_utc": utc(), "current_job": current,
            "seconds_left": round(deadline - time.time(), 1),
            "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "server_alive": server_alive(2.0),
            "jobs": jobs, "history": history,
            "total_done": sum(j["done"] for j in jobs),
            "total_expected": sum(j["expected"] for j in jobs)}


def run_job(job, deadline: float, history: list) -> str:
    jid, db, prompts, cohort, cap, expected, note = job
    dbp = ROOT / db
    left = remaining(dbp, cohort)
    if left == 0:
        log(f"  {jid}: already complete ({expected}/{expected})")
        return "complete"

    cmd = [sys.executable, "-u", str(HERE / "s12_run.py"),
           "--db", db, "--prompts", prompts, "--cohort", cohort,
           "--max-tokens", str(cap)]
    if cohort.startswith("eval_cprime"):
        cmd += ["--block", "2"]
    if cohort in BLIND_COHORTS:
        # s12's stdout is redirected into a plaintext log below, so this is the only
        # thing standing between a blinded cohort and a readable result.
        cmd += ["--blind"]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if time.time() >= deadline:
            return "budget_expired"
        if not wait_for_server():
            log(f"  {jid}: server unavailable after {SERVER_GRACE_S}s; abandoning job")
            return "server_down"

        left = remaining(dbp, cohort)
        log(f"  {jid} attempt {attempt}/{MAX_ATTEMPTS}: {left} branch(es) to go "
            f"({hms(deadline - time.time())} of budget left)"
            + ("  [BLINDED]" if cohort in BLIND_COHORTS else ""))
        LOGDIR.mkdir(parents=True, exist_ok=True)
        logf = LOGDIR / f"{jid}.log"
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== attempt {attempt} at {utc()} =====\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                                    stderr=subprocess.STDOUT, env=env)
            while proc.poll() is None:
                time.sleep(20)
                write_state(snapshot(deadline, jid, history))
                if time.time() >= deadline:
                    log(f"  {jid}: budget expired mid-job; stopping after the "
                        f"in-flight branch")
                    proc.terminate()
                    try:
                        proc.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return "budget_expired"
            rc = proc.returncode

        left = remaining(dbp, cohort)
        if left == 0:
            log(f"  {jid}: COMPLETE ({expected}/{expected})")
            return "complete"
        log(f"  {jid}: exited rc={rc} with {left} branch(es) left; retrying")
        time.sleep(10)

    return "incomplete"


def main(argv: list[str]) -> int:
    receipts.force_utf8()
    # Before anything is generated: an eval cohort's outcomes reach a plaintext log the
    # moment it runs, and no later decision can put them back.
    require_blinding_classification()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="comma-separated job ids")
    args = ap.parse_args(argv)

    jobs = JOBS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        jobs = [j for j in JOBS if j[0] in want]

    deadline = time.time() + args.hours * 3600
    log("=" * 78)
    log(f"overnight queue: {len(jobs)} job(s), budget {args.hours}h, "
        f"deadline {datetime.fromtimestamp(deadline, timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    total_left = 0
    for jid, db, _, cohort, cap, expected, note in jobs:
        left = remaining(ROOT / db, cohort)
        total_left += left
        log(f"  {jid:20} cap {cap:5d}  {left:3d}/{expected:3d} to run   {note}")
    log(f"  total branches to generate: {total_left}")
    blinded = sorted({c for _, _, _, c, _, _, _ in JOBS if c in BLIND_COHORTS})
    log(f"  blinded cohorts (outcomes withheld from the logs): "
        f"{', '.join(blinded) if blinded else 'NONE'}")
    for c in sorted({c for _, _, _, c, _, _, _ in JOBS if c in UNBLINDED_OK}):
        log(f"  unblinded by decision: {c}")

    if args.dry_run:
        log("dry run -- nothing launched")
        write_state(snapshot(deadline, None, []))
        return 0

    history: list = []
    write_state(snapshot(deadline, None, history))
    for job in jobs:
        jid = job[0]
        if time.time() >= deadline:
            log(f"budget expired before {jid}")
            history.append({"job": jid, "outcome": "not_started", "at": utc()})
            continue
        log(f"--- {jid} ---")
        t0 = time.time()
        outcome = run_job(job, deadline, history)
        history.append({"job": jid, "outcome": outcome, "at": utc(),
                        "elapsed": hms(time.time() - t0)})
        write_state(snapshot(deadline, None, history))
        if outcome in ("budget_expired", "server_down"):
            log(f"stopping the queue: {outcome}")
            break

    log("=" * 78)
    log("queue finished. summary:")
    snap = snapshot(deadline, None, history)
    for j in snap["jobs"]:
        mark = "OK " if j["done"] >= j["expected"] else "-- "
        log(f"  {mark}{j['id']:20} {j['done']:3d}/{j['expected']:3d}")
    log(f"  generated {snap['total_done']}/{snap['total_expected']} branches overall")
    write_state(snap)
    log(f"state: {STATE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
