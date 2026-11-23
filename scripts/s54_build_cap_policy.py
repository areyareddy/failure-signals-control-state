"""Step s54 -- stage the targeted-8K cap-policy replication. CPU only.

Governed by spec/CAP_POLICY_SPEC.md, frozen before this script staged anything.

    Long        12,288 for all 40 roots
    Targeted-8K  8,192 for the 15 stage-one `truncated` roots, 12,288 for the other 25

so the policies differ on 15 roots per block. 120 Long + 45 Short = 165 generations.

WHY COHORT CHUNKS. scripts/s12_run.py takes ONE max_tokens for a whole invocation
(s12_run.py:69) and it is a frozen surface, so mixed caps cannot run in a single pass and
the matched Long/Short branches cannot be interleaved branch-by-branch. Running each pair
as its own invocation would work but costs a Mathlib REPL spin-up (~8.6 s) per pair.

The compromise the spec's section 7 asks for is temporal PROXIMITY of the two measurements
being differenced, not literal adjacency. So the 15 affected roots in each block are split
into chunks of five, and each chunk's Long and Short cohorts are run back to back with the
leading policy alternating. Matched measurements then sit ~4 minutes apart instead of
~2 hours, which is what keeps thermal drift out of the energy difference. Order cannot
touch verification at all: generation is deterministic given seed, prompt and cap on a
single uncached slot.

The 25 unaffected roots are generated once per block at 12,288 and their energy counts
toward BOTH policies, so their placement cannot bias the difference.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import policy  # noqa: E402
import receipts  # noqa: E402
import s08_init_db  # noqa: E402

DB = ROOT / "data/cap_policy.sqlite"
PROMPTS_OUT = ROOT / "data/cap_policy_prompts.jsonl"
PLAN_OUT = ROOT / "audit/cap_policy_plan.json"
REGISTRY = ROOT / "audit/seed_registry.json"
LONG_CAP = 12288
SHORT_CAP = 8192
CHUNK = 5
BASES = (4_000_000, 5_000_000, 6_000_000)


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    receipts.force_utf8()

    print("[1/5] gates: seed registry, prompts, failure classes")
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert reg["clean"], "seed registry is not clean -- refusing to stage"
    assert list(reg["proposed_bases"]) == list(BASES), "bases differ from the proved set"
    known = set()
    for b in BASES:
        known |= set(range(b, b + 40))
    print(f"      seed registry clean; {len(known)} proposed seeds proved disjoint")

    roots = {r["root_id"]: r for r in read_jsonl(ROOT / "data/roots.jsonl")
             if r["split"] == "held_out"}
    assert len(roots) == 40, f"expected 40 held-out roots, got {len(roots)}"

    frozen = {(r["root_id"], r["arm"]): r for r in read_jsonl(ROOT / "data/prompts.jsonl")}
    n_ok = 0
    for rid, rec in sorted(roots.items()):
        base = policy.official_prompt(rec["theorem_text"])
        ref = frozen[(rid, "A")]["messages"][0]["content"]
        assert ref == base, (f"{rid}: arm-A prompt does not re-derive byte-identically; "
                             "the shared prompt surface changed")
        n_ok += 1
    print(f"      {n_ok} arm-A prompts re-derive byte-identically")

    truncated = sorted(r for r, rec in roots.items() if rec["outcome_state"] == "truncated")
    unaffected = sorted(r for r in roots if r not in truncated)
    assert len(truncated) == 15, f"expected 15 truncated roots, got {len(truncated)}"
    assert len(unaffected) == 25
    print(f"      {len(truncated)} truncated (capped at {SHORT_CAP}), "
          f"{len(unaffected)} unaffected (stay at {LONG_CAP})")
    print("      classification reads frozen roots.jsonl, never this run's outcomes")

    print("\n[2/5] building the execution plan")
    order = sorted(roots)                      # deterministic root order
    seed_of = {}
    for bi, base_seed in enumerate(BASES, start=1):
        for i, rid in enumerate(order):
            seed_of[(bi, rid)] = base_seed + i

    chunks = [truncated[i:i + CHUNK] for i in range(0, len(truncated), CHUNK)]
    plan: list[dict] = []
    for bi in (1, 2, 3):
        plan.append({"cohort": f"cap_eval_b{bi}_long_main", "policy": "long",
                     "cap": LONG_CAP, "block": bi, "roots": unaffected})
        for ci, ch in enumerate(chunks, start=1):
            # alternate which policy leads, by chunk and by block, so no policy is
            # systematically measured on a hotter GPU
            long_first = ((bi + ci) % 2 == 0)
            pair = [
                {"cohort": f"cap_eval_b{bi}_long_c{ci}", "policy": "long",
                 "cap": LONG_CAP, "block": bi, "roots": ch},
                {"cohort": f"cap_eval_b{bi}_short_c{ci}", "policy": "short",
                 "cap": SHORT_CAP, "block": bi, "roots": ch},
            ]
            plan.extend(pair if long_first else pair[::-1])
    n_branches = sum(len(p["roots"]) for p in plan)
    assert n_branches == 165, f"expected 165 branches, planned {n_branches}"
    print(f"      {len(plan)} cohorts, {n_branches} branches "
          f"({3 * 40} long + {3 * 15} short)")

    print("\n[3/5] building prompts")
    records = []
    for step, entry in enumerate(plan):
        for rid in entry["roots"]:
            rec = roots[rid]
            base = policy.official_prompt(rec["theorem_text"])
            messages = [{"role": "user", "content": base}]
            seed = seed_of[(entry["block"], rid)]
            assert seed in known, f"{rid}: seed {seed} outside the proved block"
            records.append({
                "branch_key": f"{rid}|{entry['policy']}|b{entry['block']}",
                "root_id": rid, "split": "held_out", "arm": "A",
                "cohort": entry["cohort"], "seed": seed,
                "seed_block": entry["block"],
                "policy": entry["policy"], "max_tokens": entry["cap"],
                "exec_step": step,
                "messages": messages, "conversation_turns": 1,
                "payload_chars": len(base),
                "payload_sha256": receipts.sha_text(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                "base_prompt_sha256": receipts.sha_text(base),
                "fine_failure_class": rec["fine_failure_class"],
                "coarse_failure_group": rec["coarse_failure_group"],
                "stage_one_state": rec["outcome_state"],
            })
    assert len({r["branch_key"] for r in records}) == len(records), "duplicate branch_key"
    print(f"      {len(records)} prompts, all arm A, all seeds inside the proved blocks")

    print("\n[4/5] writing prompts and plan")
    with PROMPTS_OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    PLAN_OUT.write_text(json.dumps(
        {"spec": "spec/CAP_POLICY_SPEC.md", "chunk_size": CHUNK,
         "long_cap": LONG_CAP, "short_cap": SHORT_CAP, "bases": list(BASES),
         "truncated_roots": truncated, "unaffected_roots": unaffected,
         "execution_order": [{k: v for k, v in p.items() if k != "roots"} |
                             {"n_roots": len(p["roots"])} for p in plan]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[5/5] staging the job store (new database)")
    if DB.exists():
        chk = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        done = chk.execute(
            "SELECT count(*) FROM branches WHERE status != 'pending'").fetchone()[0]
        chk.close()
        assert done == 0, (f"{DB.name} holds {done} non-pending branches; refusing to "
                           "rebuild and discard generated data")
        stamp = __import__("time").strftime("%Y%m%dT%H%M%S")
        DB.replace(ROOT / "Backup" / f"cap_policy.sqlite.{stamp}.prerebuild")
        for suf in ("-wal", "-shm"):
            side = DB.with_name(DB.name + suf)
            if side.exists():
                side.unlink()

    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(s08_init_db.SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(branches)")}
    has_mt = "max_tokens" in cols
    for r in records:
        if has_mt:
            con.execute(
                "INSERT OR IGNORE INTO branches (branch_key, root_id, split, arm, cohort, "
                "seed_block, seed, payload_sha256, conversation_turns, "
                "fine_failure_class, coarse_failure_group, max_tokens) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["branch_key"], r["root_id"], r["split"], "A", r["cohort"],
                 r["seed_block"], r["seed"], r["payload_sha256"], 1,
                 r["fine_failure_class"], r["coarse_failure_group"], r["max_tokens"]))
        else:
            con.execute(
                "INSERT OR IGNORE INTO branches (branch_key, root_id, split, arm, cohort, "
                "seed_block, seed, payload_sha256, conversation_turns, "
                "fine_failure_class, coarse_failure_group) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (r["branch_key"], r["root_id"], r["split"], "A", r["cohort"],
                 r["seed_block"], r["seed"], r["payload_sha256"], 1,
                 r["fine_failure_class"], r["coarse_failure_group"]))
    con.commit()
    n = con.execute("SELECT count(*) FROM branches").fetchone()[0]
    con.close()
    assert n == 165, f"expected 165 staged branches, got {n}"
    print(f"      {DB.relative_to(ROOT).as_posix()}: {n} branches pending"
          + ("" if has_mt else "  (no max_tokens column; cap carried per invocation)"))
    print(f"      {PROMPTS_OUT.relative_to(ROOT).as_posix()}")
    print(f"      {PLAN_OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
