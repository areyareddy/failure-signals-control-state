"""Step s40 -- serving-regime side-measurement. NOT a frozen cohort.

WHAT THIS IS. The W2 prefill result is measured in one serving configuration: a single
slot, no continuous batching, no prompt cache, cold. It is fair to object that this is the
one regime in which prefill cannot matter. This script re-measures the SAME frozen arm-A
and arm-C prompts, at the SAME seeds, under 4-slot continuous batching, so the prefill
share can be reported as a function of the serving regime rather than as a property of one
machine.

WHAT THIS IS NOT. It runs no Lean verifier and stores no completion text -- only a
SHA-256 of the completion, which is enough to detect whether batching changed the sampled
trajectory without materialising any outcome. The cohort therefore produces no
verification result and can enter no confirmatory claim. It is a phase-timing measurement
only. This is also why it needs no blinding declaration: there is no outcome to blind.

WHY IT DOES NOT USE src/runner.py. That path is frozen, and it is serial by construction
(one claim, one branch, an explicit single-runner lock). Driving four concurrent requests
through it would mean editing the frozen execution path. This script talks to the same
src/generate.py client instead, so the recorded phase fields are the identical
llama.cpp timings keys (prompt_n/prompt_ms, predicted_n/predicted_ms) that the frozen
runner recorded, and the two regimes stay commensurable.

    python scripts\\s40_serving_regime.py --regime p1 --workers 1   # control
    python scripts\\s40_serving_regime.py --regime p4 --workers 4   # treatment
    python scripts\\s40_serving_regime.py --regime p4 --workers 4 --limit 4  # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import generate  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS regime_branches (
    regime            TEXT NOT NULL,
    branch_key        TEXT NOT NULL,
    root_id           TEXT NOT NULL,
    arm               TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    workers           INTEGER NOT NULL,
    parallel_slots    INTEGER,
    n_ctx             INTEGER,
    prefill_tokens    INTEGER,
    decode_tokens     INTEGER,
    prefill_ms        REAL,
    decode_ms         REAL,
    request_wall_seconds REAL,
    finish_reason     TEXT,
    cache_n           INTEGER,
    completion_sha256 TEXT,
    started_at        REAL,
    finished_at       REAL,
    raw_timings_json  TEXT,
    PRIMARY KEY (regime, branch_key)
);
CREATE TABLE IF NOT EXISTS regime_runs (
    regime            TEXT PRIMARY KEY,
    workers           INTEGER,
    parallel_slots    INTEGER,
    n_ctx             INTEGER,
    started_at        REAL,
    finished_at       REAL,
    n_branches        INTEGER,
    server_props_json TEXT
);
"""


def eval_jobs(runs_db, arms):
    """The frozen primary_eval arm-A/arm-C branches: same roots, same seeds."""
    con = sqlite3.connect("file:%s?mode=ro" % runs_db, uri=True)
    con.row_factory = sqlite3.Row
    qs = ",".join("?" * len(arms))
    rows = [dict(r) for r in con.execute(
        "SELECT branch_key, root_id, arm, seed FROM branches "
        "WHERE cohort='primary_eval' AND arm IN (%s) AND status='done' "
        "ORDER BY arm, root_id" % qs, arms)]
    con.close()
    return rows


def payloads(prompts):
    out = {}
    with prompts.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["branch_key"]] = d["messages"]
    return out


def server_props(url):
    try:
        with urllib.request.urlopen(url + "/props", timeout=10) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except Exception as exc:
        print("  warn: /props unavailable (%s)" % exc)
        return {}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, help="label, e.g. p1 or p4")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--arms", default="A,C")
    ap.add_argument("--config", default="configs/serving_regime_p4.json")
    ap.add_argument("--runs-db", default="data/runs.sqlite")
    ap.add_argument("--db", default="data/serving_regime.sqlite")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    url = cfg["server"]["url"]
    max_tokens = cfg["generation"]["max_tokens"]
    arms = tuple(a.strip() for a in args.arms.split(","))

    con = sqlite3.connect(ROOT / args.db)
    con.executescript(SCHEMA)
    con.commit()

    generate.Client(url).wait_until_ready()
    props = server_props(url)
    slots = props.get("total_slots")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    print("server: total_slots=%s per-request n_ctx=%s" % (slots, n_ctx))
    if slots is not None and slots < args.workers:
        print("  NOTE: workers=%d exceeds %s slot(s); the excess queues rather than "
              "batches." % (args.workers, slots))

    jobs = eval_jobs(ROOT / args.runs_db, arms)
    msgs = payloads(ROOT / args.prompts)
    done = {r[0] for r in con.execute(
        "SELECT branch_key FROM regime_branches WHERE regime=?", (args.regime,))}
    jobs = [j for j in jobs if j["branch_key"] not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print("regime=%s workers=%d arms=%s pending=%d (already done %d)"
          % (args.regime, args.workers, arms, len(jobs), len(done)))
    if not jobs:
        return 0

    local = threading.local()
    lock = threading.Lock()
    t_run0 = time.time()

    def client():
        if not hasattr(local, "c"):
            local.c = generate.Client(url)
        return local.c

    def one(job):
        t0 = time.time()
        res = client().chat(msgs[job["branch_key"]], seed=job["seed"],
                            max_tokens=max_tokens)
        t1 = time.time()
        return {
            "regime": args.regime, "branch_key": job["branch_key"],
            "root_id": job["root_id"], "arm": job["arm"], "seed": job["seed"],
            "workers": args.workers, "parallel_slots": slots, "n_ctx": n_ctx,
            "prefill_tokens": res.prefill_tokens, "decode_tokens": res.decode_tokens,
            "prefill_ms": res.prefill_ms, "decode_ms": res.decode_ms,
            "request_wall_seconds": res.request_wall_seconds,
            "finish_reason": res.finish_reason,
            "cache_n": res.raw_timings.get("cache_n", 0),
            "completion_sha256": hashlib.sha256(
                res.completion.encode("utf-8")).hexdigest(),
            "started_at": t0, "finished_at": t1,
            "raw_timings_json": json.dumps(res.raw_timings, sort_keys=True),
        }

    cols = ("regime", "branch_key", "root_id", "arm", "seed", "workers",
            "parallel_slots", "n_ctx", "prefill_tokens", "decode_tokens",
            "prefill_ms", "decode_ms", "request_wall_seconds", "finish_reason",
            "cache_n", "completion_sha256", "started_at", "finished_at",
            "raw_timings_json")
    sql = ("INSERT OR REPLACE INTO regime_branches (%s) VALUES (%s)"
           % (",".join(cols), ",".join("?" * len(cols))))

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                n_err += 1
                print("  ERROR %s: %s" % (job["branch_key"], exc), flush=True)
                continue
            with lock:
                con.execute(sql, tuple(rec[c] for c in cols))
                con.commit()
                n_ok += 1
                el = (time.time() - t_run0) / 60.0
                print("  [%3d/%3d] %-46s pf=%6d (%8.1fms)  dec=%6d (%9.1fms)  "
                      "wall=%6.1fs  elapsed=%.1fm"
                      % (n_ok + n_err, len(jobs), rec["branch_key"],
                         rec["prefill_tokens"], rec["prefill_ms"],
                         rec["decode_tokens"], rec["decode_ms"],
                         rec["request_wall_seconds"], el), flush=True)

    t_run1 = time.time()
    con.execute(
        "INSERT OR REPLACE INTO regime_runs (regime, workers, parallel_slots, n_ctx, "
        "started_at, finished_at, n_branches, server_props_json) VALUES (?,?,?,?,?,?,?,?)",
        (args.regime, args.workers, slots, n_ctx, t_run0, t_run1, n_ok,
         json.dumps({k: v for k, v in props.items() if k != "chat_template"},
                    sort_keys=True)))
    con.commit()
    con.close()
    print("\ndone: %d ok, %d error, wall %.1f min"
          % (n_ok, n_err, (t_run1 - t_run0) / 60.0))
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
