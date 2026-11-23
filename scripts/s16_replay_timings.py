"""Step s16 -- re-meter per-branch prefill/decode TIME. No Lean. No writes to runs.sqlite.

    python scripts\\s16_replay_timings.py --canary-only    preflight + canary, then stop
    python scripts\\s16_replay_timings.py                  full replay, resumable

NOT a re-run of the experiment. Every completion must be byte-identical to the one already
stored, asserted per branch with ZERO tolerance. runs.sqlite is opened read-only;
everything lands in data/timing_replay.sqlite.

WHY: generate.Result carried prefill_ms and decode_ms on all 536 original calls and
nothing consumed them. audit/timing_regression.json shows they cannot be reconstructed
from stored token counts and wall time -- the fit needs a negative intercept and misplaces
prefill by an order of magnitude.

CONTROLS

  zero tolerance        the first hash mismatch aborts.

  frozen identity       spec, prompts, roots, sampler, max_tokens, build, model file,
                        chat template and server args are hashed into a run identity. A
                        resume may only join a run with an identical identity, and every
                        segment re-checks LIVE provenance against it before measuring.

  canary is DIAGNOSTIC  canary rows go to canary_timings and are never analytical. A
                        canary measures unmatched branches whose paired A/C and B/D
                        partners are measured hours later, so pairing them would compare
                        across time rather than across arm. The main run measures all 536
                        itself, including the canary branches.

  every segment tested  a segment measures nothing until its OWN canary passes. Resuming
                        never inherits a previous segment's canary.

  restart detection     the llama-server (pid, create_time) is captured per segment and
                        checked before EVERY branch, so a restart that happens between
                        two requests -- with nothing failing -- is still caught before the
                        next measurement lands.

  counterbalanced order within each root the four A-D arms are measured in a rotation
                        keyed by root index, so each arm occupies each ordinal position
                        in a quarter of roots. A fixed A,B,C,D order would alias thermal
                        and scheduling drift directly onto the arm effect, which is the
                        very quantity being measured.

ORDER: 160 primary A-D, 160 block-2 A-D, 172 dev A-D, then 44 E. W2's isolating contrasts
need the 492 A-D rows; arm E is a positive control outside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import generate  # noqa: E402
import psutil  # noqa: E402
import receipts  # noqa: E402
import replay_store  # noqa: E402

SOURCE_DB = ROOT / "data/runs.sqlite"
REPLAY_DB = ROOT / "data/timing_replay.sqlite"
BACKUP_DB = ROOT / "data/runs.backup.sqlite"
SPEC = ROOT / "spec/frozen_spec.yaml"
SPEC_SHA = ROOT / "spec/frozen_spec.sha256"

# Bump this whenever the measurement protocol changes in a way that makes new rows
# non-comparable with old ones. It is part of the run identity, so a resume under a
# changed protocol opens a NEW run instead of pooling into the old one.
MEASUREMENT_PROTOCOL = "timing_replay_v2"

REQUIRED_GREEN = ["s01", "s02", "s03", "s05", "s06"]
WARMUP_CALLS = 2
COHORT_ORDER = {"primary_eval": 0, "eval_block2": 1, "full_pool_dev": 2}

# Latin-square rotation over the four factorial arms.
ARM_ROTATIONS = [("A", "B", "C", "D"), ("B", "C", "D", "A"),
                 ("C", "D", "A", "B"), ("D", "A", "B", "C")]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def server_identity() -> tuple[int | None, float | None]:
    """(pid, create_time) of the llama-server process. A restart changes both."""
    for proc in psutil.process_iter(["name", "pid", "create_time"]):
        if "llama-server" in (proc.info.get("name") or ""):
            return proc.info["pid"], proc.info["create_time"]
    return None, None


def props(base_url: str) -> dict:
    with urllib.request.urlopen(base_url + "/props", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def provenance(base_url: str) -> dict:
    p = props(base_url)
    gen = p.get("default_generation_settings", {})
    return {
        "llama_build": p.get("build_info", ""),
        "model_path": p.get("model_path", ""),
        "model_ftype": p.get("model_ftype", ""),
        "chat_template_sha256": hashlib.sha256(
            (p.get("chat_template") or "").encode("utf-8")).hexdigest(),
        "n_ctx": gen.get("n_ctx"),
        "total_slots": p.get("total_slots"),
        "bos_token": p.get("bos_token"),
        "eos_token": p.get("eos_token"),
        "server_args": {"n_ctx": gen.get("n_ctx"), "slots": p.get("total_slots"),
                        "cache_prompt": gen.get("cache_prompt", False)},
    }


def preflight(base_url: str) -> dict:
    print("[1/5] preflight")
    recomputed = receipts.sha_text(SPEC.read_text(encoding="utf-8"))
    stored = SPEC_SHA.read_text(encoding="utf-8").split()[0]
    print(f"      spec hash recomputed  "
          f"{'MATCH' if recomputed == stored else 'MISMATCH'}  {recomputed[:16]}")
    if recomputed != stored:
        raise SystemExit("frozen_spec.yaml no longer hashes to its recorded sha256")

    for step in REQUIRED_GREEN:
        status, detail = receipts.verify(step)
        print(f"      gate {step}  {status}  {detail}")
        if status != "done":
            raise SystemExit(f"{step} is {status}; refusing to start")

    con = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    n_done = con.execute("SELECT count(*) FROM branches WHERE status='done'").fetchone()[0]
    n_all = con.execute("SELECT count(*) FROM branches").fetchone()[0]
    hashes = [r[0] for r in con.execute(
        "SELECT DISTINCT spec_sha256 FROM branches WHERE status='done'")]
    # The fingerprint must cover every field validate() compares against, or a change
    # in one of them would leave the identity unchanged and a resume would check later
    # segments against different originals than the earlier ones used.
    fingerprint = receipts.sha_text(chr(10).join(
        f"{k}|{sha}|{seed}|{pt}|{dt}|{ct}|{tt}|{fr}"
        for k, sha, seed, pt, dt, ct, tt, fr in con.execute(
            "SELECT branch_key, completion_sha256, seed, prefill_tokens, decode_tokens, "
            "completion_tokens, total_tokens, finish_reason "
            "FROM branches ORDER BY branch_key")))
    con.close()
    print(f"      source rows {n_done}/{n_all}, {len(hashes)} spec hash, "
          f"fingerprint {fingerprint[:16]}")
    if n_done != n_all or hashes != [stored]:
        raise SystemExit("source database is not in the expected state")

    rows, integrity = replay_store.backup(SOURCE_DB, BACKUP_DB)
    print(f"[2/5] backup: {rows} rows, integrity_check={integrity}")
    if integrity != "ok" or rows != n_all:
        raise SystemExit("backup failed verification")

    prov = provenance(base_url)
    pid, started = server_identity()
    when = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
            if started else "unknown")
    print(f"[3/5] server: build {prov['llama_build']}, n_ctx {prov['n_ctx']}, "
          f"pid {pid} up since {when}")

    identity = {
        "spec_sha256": stored,
        "prompts_sha256": receipts.sha_file(ROOT / "data/prompts.jsonl"),
        "roots_sha256": receipts.sha_file(ROOT / "data/roots.jsonl"),
        "source_content_fingerprint": fingerprint,
        "sampler": generate.SAMPLER,
        "max_tokens": generate.MAX_TOKENS,
        # the schedule is a CONTROL, not an implementation detail: counterbalancing is
        # what stops drift from aliasing onto the arm effect. Two runs scheduled
        # differently are not poolable, so the schedule belongs in the identity.
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "arm_rotations": [list(r) for r in ARM_ROTATIONS],
        "cohort_order": COHORT_ORDER,
        **{k: v for k, v in prov.items() if k != "server_args"},
        "server_args": prov["server_args"],
    }
    identity_sha = receipts.sha_text(json.dumps(identity, sort_keys=True))
    print(f"      run identity {identity_sha[:16]}")
    return {"identity": identity, "identity_sha": identity_sha, "prov": prov}


def load_branches() -> list[dict]:
    """Cohort priority, counterbalanced arm rotation within each root, arm E last."""
    con = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    cols = ["branch_key", "cohort", "root_id", "arm", "seed_block", "seed",
            "completion_sha256", "prefill_tokens", "decode_tokens",
            "completion_tokens", "total_tokens", "finish_reason"]
    rows = [dict(zip(cols, r)) for r in con.execute(
        "SELECT " + ", ".join(cols) + " FROM branches WHERE status='done'")]
    con.close()

    ordered: list[dict] = []
    for cohort in sorted(COHORT_ORDER, key=lambda c: COHORT_ORDER[c]):
        by_root: dict[str, dict[str, dict]] = {}
        for r in rows:
            if r["cohort"] == cohort and r["arm"] != "E":
                by_root.setdefault(r["root_id"], {})[r["arm"]] = r
        # index WITHIN the cohort, not globally: a 40-root cohort then divides exactly
        # 10/10/10/10 across the four rotations. A global index leaves the eval cohorts
        # at 8/10/11/11, which is balanced enough to be unobjectionable but not exactly
        # balanced, and exact costs nothing.
        for local, rid in enumerate(sorted(by_root)):
            for arm in ARM_ROTATIONS[local % len(ARM_ROTATIONS)]:
                if arm in by_root[rid]:
                    ordered.append(by_root[rid][arm])
    ordered += sorted((r for r in rows if r["arm"] == "E"),
                      key=lambda r: (COHORT_ORDER[r["cohort"]], r["branch_key"]))
    return ordered


def select_canary(branches: list[dict]) -> list[dict]:
    """Representative: every arm A-E, both seed blocks, longest-prefill C and D."""
    def pick(**kw):
        return [b for b in branches if all(b[k] == v for k, v in kw.items())]

    def longest(rows):
        return max(rows, key=lambda r: r["prefill_tokens"]) if rows else None

    def first(rows):
        return sorted(rows, key=lambda r: r["branch_key"])[0] if rows else None

    candidates = [
        longest(pick(cohort="primary_eval", arm="C")),
        longest(pick(cohort="primary_eval", arm="D")),
        longest(pick(cohort="eval_block2", arm="C")),
        longest(pick(cohort="eval_block2", arm="D")),
        first(pick(cohort="primary_eval", arm="A")),
        first(pick(cohort="primary_eval", arm="B")),
        first(pick(cohort="eval_block2", arm="A")),
        first(pick(cohort="eval_block2", arm="B")),
        first(pick(cohort="primary_eval", arm="E")),
        first(pick(cohort="full_pool_dev", arm="D")),
    ]
    out, seen = [], set()
    for c in candidates:
        if c and c["branch_key"] not in seen:
            seen.add(c["branch_key"])
            out.append(c)
    return out


def payload_for(payloads: dict, branch: dict) -> dict:
    """A block-2 row reuses block 1's payload byte for byte; '#b2' identifies the ROW."""
    return payloads[branch["branch_key"].split("#", 1)[0]]


def validate(res, branch: dict) -> list[str]:
    bad = []
    for name, v in (("prefill_ms", res.prefill_ms), ("decode_ms", res.decode_ms)):
        if v is None or not math.isfinite(v) or v <= 0:
            bad.append(f"{name}={v!r} not finite and positive")
    if res.cached_tokens or res.raw_timings.get("cache_n", 0):
        bad.append(f"cache reuse: cached={res.cached_tokens}, "
                   f"cache_n={res.raw_timings.get('cache_n')}")
    if res.prefill_tokens != branch["prefill_tokens"]:
        bad.append(f"prefill_tokens {res.prefill_tokens} != {branch['prefill_tokens']}")
    if res.completion_tokens != branch["completion_tokens"]:
        bad.append(f"completion_tokens {res.completion_tokens} != "
                   f"{branch['completion_tokens']}")
    # every token count that gets STORED must also be CHECKED. decode_tokens and
    # total_tokens were being written into the analytical table without ever being
    # compared to the original, so a silent divergence in either would have survived
    # into the W2 accounting unremarked.
    if res.decode_tokens != branch["decode_tokens"]:
        bad.append(f"decode_tokens {res.decode_tokens} != {branch['decode_tokens']}")
    if branch.get("total_tokens") is not None and res.total_tokens != branch["total_tokens"]:
        bad.append(f"total_tokens {res.total_tokens} != {branch['total_tokens']}")
    if res.finish_reason != branch["finish_reason"]:
        bad.append(f"finish_reason {res.finish_reason!r} != {branch['finish_reason']!r}")
    return bad


class Segment:
    """A contiguous measurement period under one verified server process."""

    def __init__(self, con, run_id, pre, client, base_url, reason):
        self.con, self.run_id, self.pre = con, run_id, pre
        self.client, self.base_url = client, base_url
        prov = provenance(base_url)
        drift = [k for k in ("llama_build", "model_path", "chat_template_sha256",
                             "n_ctx", "total_slots")
                 if prov[k] != pre["identity"][k]]
        if drift:
            raise SystemExit(
                f"live server provenance no longer matches the frozen run identity: "
                f"{drift}. Measurements from a different configuration must not be "
                f"pooled with this run.")
        self.pid, self.started = server_identity()
        cur = con.execute(
            "INSERT INTO replay_segments (replay_run_id, started_utc, open_reason, "
            "llama_build, provenance_json, server_pid, server_started) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, utc(), reason, prov["llama_build"],
             json.dumps(prov, sort_keys=True), self.pid, self.started))
        self.id = cur.lastrowid

    def server_unchanged(self) -> bool:
        return server_identity() == (self.pid, self.started)

    def close(self, reason: str) -> None:
        self.con.execute("UPDATE replay_segments SET ended_utc=?, close_reason=? "
                         "WHERE segment_id=?", (utc(), reason[:300], self.id))

    def warm_up(self) -> None:
        for _ in range(WARMUP_CALLS):
            try:
                self.client.chat([{"role": "user", "content": "warmup"}], seed=1,
                                 max_tokens=16, attempts=1)
            except Exception:  # noqa: BLE001
                pass
        self.con.execute("UPDATE replay_segments SET warmup_calls=? WHERE segment_id=?",
                         (WARMUP_CALLS, self.id))

    def run_canary(self, canary: list[dict], payloads: dict) -> bool:
        """Every segment earns its own canary. Results are DIAGNOSTIC only."""
        print(f"      canary for segment {self.id}: {len(canary)} branches, "
              f"zero tolerance", flush=True)
        for b in canary:
            res = self.client.chat(payload_for(payloads, b)["messages"], seed=b["seed"],
                                   max_tokens=generate.MAX_TOKENS, attempts=1)
            sha = receipts.sha_text(res.completion)
            match = sha == b["completion_sha256"]
            problems = validate(res, b)
            self.con.execute(
                "INSERT OR REPLACE INTO canary_timings (replay_run_id, segment_id, "
                "branch_key, cohort, root_id, arm, seed_block, prefill_ms, decode_ms, "
                "prefill_tokens, decode_tokens, completion_tokens, request_wall_seconds, "
                "completion_sha256, original_sha256, hash_match, raw_timings_json, "
                "measured_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, self.id, b["branch_key"], b["cohort"], b["root_id"],
                 b["arm"], b["seed_block"], res.prefill_ms, res.decode_ms,
                 res.prefill_tokens, res.decode_tokens, res.completion_tokens,
                 res.request_wall_seconds, sha, b["completion_sha256"],
                 1 if match else 0, json.dumps(res.raw_timings, sort_keys=True), utc()))
            print(f"      {'ok  ' if match and not problems else 'FAIL'} "
                  f"{b['branch_key']:46} {b['arm']} blk{b['seed_block']} "
                  f"{res.prefill_ms:7.1f}ms + {res.decode_ms:9.1f}ms", flush=True)
            if not match or problems:
                replay_store.event(self.con, self.run_id, self.id, "canary_failed",
                                   f"{b['branch_key']}: match={match} {problems}")
                self.close("canary failed")
                return False
        self.con.execute("UPDATE replay_segments SET canary_passed=1 WHERE segment_id=?",
                         (self.id,))
        return True

    def record(self, b: dict, res, sha: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO branch_timings (replay_run_id, branch_key, "
            "segment_id, cohort, root_id, arm, seed_block, seed, max_tokens, prefill_ms, "
            "decode_ms, prefill_tokens, decode_tokens, completion_tokens, total_tokens, "
            "request_wall_seconds, cache_n, cached_tokens, finish_reason, "
            "completion_sha256, original_sha256, hash_match, raw_timings_json, "
            "replayed_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, b["branch_key"], self.id, b["cohort"], b["root_id"], b["arm"],
             b["seed_block"], b["seed"], generate.MAX_TOKENS, res.prefill_ms,
             res.decode_ms, res.prefill_tokens, res.decode_tokens, res.completion_tokens,
             res.total_tokens, res.request_wall_seconds,
             res.raw_timings.get("cache_n", 0), res.cached_tokens, res.finish_reason,
             sha, b["completion_sha256"], 1, json.dumps(res.raw_timings, sort_keys=True),
             utc()))
        self.con.execute("UPDATE replay_segments SET branches_measured = "
                         "branches_measured + 1 WHERE segment_id=?", (self.id,))


def new_segment(con, run_id, pre, client, base_url, canary, payloads, reason):
    """Open a segment, warm it, and make it earn its own canary before it measures."""
    seg = Segment(con, run_id, pre, client, base_url, reason)
    print(f"      segment {seg.id} open (server pid {seg.pid}): {reason}", flush=True)
    seg.warm_up()
    if not seg.run_canary(canary, payloads):
        return None
    print(f"      segment {seg.id} canary {len(canary)}/{len(canary)} exact "
          f"(diagnostic rows only)", flush=True)
    return seg


def main(argv: list[str]) -> int:
    receipts.force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary.json")
    ap.add_argument("--canary-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    base_url = cfg["server"]["url"]
    client = generate.Client(base_url)

    pre = preflight(base_url)
    branches = load_branches()
    payloads = {}
    for line in (ROOT / "data/prompts.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        payloads[rec["branch_key"]] = rec
    ad = sum(1 for b in branches if b["arm"] != "E")
    print(f"[4/5] {len(branches)} branches queued ({ad} A-D counterbalanced, "
          f"{len(branches) - ad} E last)")

    con = replay_store.connect(REPLAY_DB)
    row = con.execute("SELECT replay_run_id FROM replay_runs WHERE identity_sha256=? "
                      "AND status='open'", (pre["identity_sha"],)).fetchone()
    if row:
        run_id = row[0]
        print(f"      resuming replay run {run_id}")
        n = con.execute("UPDATE replay_segments SET ended_utc=?, "
                        "close_reason='orphaned by a previous process' "
                        "WHERE replay_run_id=? AND ended_utc IS NULL",
                        (utc(), run_id)).rowcount
        if n:
            print(f"      closed {n} orphaned segment(s); the new one runs its own canary")
    else:
        run_id = f"replay-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        i = pre["identity"]
        con.execute(
            "INSERT INTO replay_runs (replay_run_id, identity_sha256, started_utc, "
            "spec_sha256, prompts_sha256, roots_sha256, source_db_sha256, sampler_json, "
            "max_tokens, llama_build, model_path, model_ftype, chat_template_sha256, "
            "n_ctx, total_slots, server_args_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, pre["identity_sha"], utc(), i["spec_sha256"], i["prompts_sha256"],
             i["roots_sha256"], i["source_content_fingerprint"],
             json.dumps(i["sampler"], sort_keys=True), i["max_tokens"], i["llama_build"],
             i["model_path"], i["model_ftype"], i["chat_template_sha256"], i["n_ctx"],
             i["total_slots"], json.dumps(i["server_args"], sort_keys=True)))
        print(f"      opened replay run {run_id}")

    done = {r[0] for r in con.execute(
        "SELECT branch_key FROM branch_timings WHERE replay_run_id=?", (run_id,))}
    if done:
        print(f"      {len(done)} branch(es) already measured analytically; skipping")

    canary = select_canary(branches)
    print("\n[5/5] measurement")
    seg = new_segment(con, run_id, pre, client, base_url, canary, payloads, "run start")
    if seg is None:
        print("\n  CANARY FAILED -- stopping. A mismatch means determinism or the")
        print("  serving configuration changed. Investigate; do not re-run blindly.")
        return 1

    if args.canary_only:
        seg.close("canary-only run")
        con.close()
        print("\n  --canary-only: segment closed cleanly. Re-run without the flag to")
        print("  measure -- the main run re-measures every branch itself, including")
        print("  these, so no canary row ever reaches the analytical set.")
        return 0

    todo = [b for b in branches if b["branch_key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"\n  measuring {len(todo)} branches", flush=True)
    t0 = time.time()
    for i, b in enumerate(todo, 1):
        if not seg.server_unchanged():
            print(f"      SERVER RESTART DETECTED before {b['branch_key']}", flush=True)
            seg.close("server restarted between requests")
            replay_store.event(con, run_id, seg.id, "server_restart_detected",
                               b["branch_key"])
            if not client.wait_until_ready(deadline_seconds=900, poll=10):
                print("      server did not return; stopping.")
                return 1
            seg = new_segment(con, run_id, pre, client, base_url, canary, payloads,
                              "server restart")
            if seg is None:
                print("      POST-RESTART CANARY FAILED. Stopping.")
                return 1

        try:
            res = client.chat(payload_for(payloads, b)["messages"], seed=b["seed"],
                              max_tokens=generate.MAX_TOKENS, attempts=1)
        except Exception as exc:  # noqa: BLE001
            print(f"      transport failure on {b['branch_key']}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            seg.close(f"transport failure: {exc}")
            if not client.wait_until_ready(deadline_seconds=900, poll=10):
                print("      server did not return; stopping.")
                return 1
            seg = new_segment(con, run_id, pre, client, base_url, canary, payloads,
                              "transport failure")
            if seg is None:
                print("      POST-RESTART CANARY FAILED. Stopping.")
                return 1
            res = client.chat(payload_for(payloads, b)["messages"], seed=b["seed"],
                              max_tokens=generate.MAX_TOKENS, attempts=1)

        # Re-check AFTER the response and BEFORE recording. The pre-request check leaves
        # a race: a restart between that check and the response means this measurement
        # came from an indeterminate regime. The TEXT would still hash correctly --
        # determinism guarantees that -- so a hash check cannot catch it. The timing is
        # what is unusable, and timing is the entire point of this run. Discard and
        # re-measure in a fresh, canary-tested segment.
        if not seg.server_unchanged():
            print(f"      server changed DURING {b['branch_key']}; discarding that "
                  f"measurement and re-measuring in a new segment", flush=True)
            replay_store.event(con, run_id, seg.id, "restart_during_request",
                               b["branch_key"])
            seg.close("server restarted during a request")
            if not client.wait_until_ready(deadline_seconds=900, poll=10):
                print("      server did not return; stopping.")
                return 1
            seg = new_segment(con, run_id, pre, client, base_url, canary, payloads,
                              "restart during request")
            if seg is None:
                print("      POST-RESTART CANARY FAILED. Stopping.")
                return 1
            res = client.chat(payload_for(payloads, b)["messages"], seed=b["seed"],
                              max_tokens=generate.MAX_TOKENS, attempts=1)
            if not seg.server_unchanged():
                print("      server changed again during the re-measurement; stopping.")
                return 1

        sha = receipts.sha_text(res.completion)
        if sha != b["completion_sha256"]:
            replay_store.event(con, run_id, seg.id, "hash_mismatch", b["branch_key"])
            print(f"\n  HASH MISMATCH on {b['branch_key']}. Zero tolerance -- aborting.")
            seg.close("hash mismatch")
            return 1
        problems = validate(res, b)
        if problems:
            replay_store.event(con, run_id, seg.id, "validation_failed",
                               f"{b['branch_key']}: {problems}")
            print(f"\n  VALIDATION FAILED on {b['branch_key']}: {problems}")
            seg.close("validation failed")
            return 1
        seg.record(b, res, sha)

        if i % 10 == 0 or i == len(todo):
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"      {i:3d}/{len(todo)} ({100 * i // len(todo):3d}%)  "
                  f"{b['arm']} {b['cohort']:14} {res.prefill_ms:7.1f}ms + "
                  f"{res.decode_ms:9.1f}ms   ETA {eta / 3600:.2f}h", flush=True)

    seg.close("run complete")
    total = con.execute("SELECT count(*) FROM branch_timings WHERE replay_run_id=?",
                        (run_id,)).fetchone()[0]
    if total == len(branches):
        con.execute("UPDATE replay_runs SET finished_utc=?, status='complete' "
                    "WHERE replay_run_id=?", (utc(), run_id))
        print(f"\n  REPLAY COMPLETE: {total}/{len(branches)}, all hashes exact.")
    else:
        print(f"\n  {total}/{len(branches)} measured; re-run to continue.")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
