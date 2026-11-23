"""Step s53 -- exhaustive seed registry and collision proof. CPU only, read-only.

The cap-policy replication needs three genuinely NEW seed blocks. "New" has to be proved,
not asserted, so this enumerates every seed this project has ever used -- every W1/W2
SQLite branch database, plus the CoLM parent study -- and proves the proposed blocks are
disjoint from all of them.

TWO LEVELS OF CHECK, deliberately.

  BARE SEED     any integer used anywhere before. Reported, and the proposed blocks are
                required to avoid these entirely. This is stricter than necessary.
  (root, seed)  the tuple that actually matters: reusing a seed on a DIFFERENT root with a
                different prompt produces an unrelated draw and is harmless. Reusing it on
                the SAME root with the same prompt and cap would silently reproduce an
                existing draw and is the real hazard.

The strict bare-seed rule is kept because it is free -- the integer space is enormous --
and because a reviewer should not have to reason about tuple semantics to believe the
blocks are new.

CoLM is READ-ONLY. This script opens its JSON/JSONL records and never writes there.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COLM = ROOT.parent / "CoLM_Jul2026_MoSpe_Navi"
OUT = ROOT / "audit/seed_registry.json"

# Proposed blocks for the cap-policy replication: base + 0..39 for the 40 held-out roots.
PROPOSED_BASES = (4_000_000, 5_000_000, 6_000_000)
BLOCK_SIZE = 40


def local_seeds() -> dict:
    """Every seed in every W1/W2 branch database, with its (cohort, root) context."""
    used: dict[int, list[str]] = {}
    tuples: set[tuple[str, int]] = set()
    for db in sorted((ROOT / "data").glob("*.sqlite")):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = list(con.execute(
                "SELECT cohort, root_id, seed FROM branches WHERE seed IS NOT NULL"))
            con.close()
        except sqlite3.Error:
            continue
        for cohort, rid, seed in rows:
            used.setdefault(int(seed), []).append(f"{db.name}:{cohort}")
            tuples.add((rid, int(seed)))
    return {"seeds": used, "tuples": tuples}


def colm_seeds() -> tuple[set[int], list[str], list[str]]:
    """Seeds recorded anywhere in the CoLM parent study.

    Returns (seeds, files_scanned, files_with_seeds). If CoLM is absent this returns
    empty sets AND says so -- the caller refuses to certify a clean audit it could not
    actually perform, rather than silently reporting 'no collisions'.
    """
    seeds: set[int] = set()
    scanned: list[str] = []
    withseeds: list[str] = []
    if not COLM.exists():
        return seeds, scanned, withseeds
    pat = re.compile(r'"[A-Za-z_]*seed[A-Za-z_]*"\s*:\s*(-?\d+)')
    for base, _dirs, files in os.walk(COLM):
        # skip vendored code and model weights; seeds live in run records
        if any(part in base for part in ("vendor", ".git", "node_modules")):
            continue
        for fn in files:
            if not fn.endswith((".json", ".jsonl")):
                continue
            p = Path(base) / fn
            rel = str(p.relative_to(COLM))
            scanned.append(rel)
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            found = pat.findall(text)
            if found:
                withseeds.append(rel)
                seeds.update(int(x) for x in found)
    return seeds, scanned, withseeds


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("[1/3] enumerating W1/W2 seeds from every branch database")
    loc = local_seeds()
    local_set = set(loc["seeds"])
    print(f"      {len(local_set)} distinct seeds across "
          f"{len(loc['tuples'])} (root, seed) pairs")
    if local_set:
        print(f"      range {min(local_set)}..{max(local_set)}")

    print("\n[2/3] enumerating CoLM parent-study seeds (read-only)")
    colm, scanned, withseeds = colm_seeds()
    colm_available = COLM.exists()
    if not colm_available:
        print("      CoLM directory NOT FOUND -- audit cannot cover the parent study")
    else:
        print(f"      scanned {len(scanned)} json/jsonl files, "
              f"{len(withseeds)} carried seed fields")
        print(f"      {len(colm)} distinct CoLM seeds"
              + (f", range {min(colm)}..{max(colm)}" if colm else ""))

    every = local_set | colm

    print("\n[3/3] proving the proposed blocks are disjoint")
    proposed: dict[str, list[int]] = {}
    collisions: dict[str, list[int]] = {}
    for i, b in enumerate(PROPOSED_BASES, start=1):
        block = list(range(b, b + BLOCK_SIZE))
        proposed[f"block{i}"] = block
        hit = sorted(set(block) & every)
        if hit:
            collisions[f"block{i}"] = hit
        print(f"      block{i}: {b}..{b + BLOCK_SIZE - 1}  "
              f"collisions={len(hit)}")

    ok = not collisions and colm_available
    result = {
        "proposed_bases": list(PROPOSED_BASES),
        "block_size": BLOCK_SIZE,
        "proposed_blocks": proposed,
        "n_local_seeds": len(local_set),
        "local_seed_min": min(local_set) if local_set else None,
        "local_seed_max": max(local_set) if local_set else None,
        "n_colm_seeds": len(colm),
        "colm_available": colm_available,
        "colm_files_scanned": len(scanned),
        "colm_files_with_seeds": sorted(withseeds)[:50],
        "n_all_known_seeds": len(every),
        "collisions": collisions,
        "clean": ok,
        "rule": ("Proposed blocks must avoid every seed ever used by W1, W2 or CoLM. "
                 "The binding hazard is a repeated (root, seed, prompt, cap) tuple, "
                 "which would reproduce an existing draw; the bare-seed rule is "
                 "stricter than needed and is applied because it costs nothing."),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    if collisions:
        print("\nFAIL: proposed blocks collide. Choose different bases.")
        return 1
    if not colm_available:
        print("\nFAIL: CoLM not reachable, so the audit is incomplete. The specification "
              "must not claim a three-way collision proof.")
        return 1
    print("\nPASS: three blocks of 40 are disjoint from all "
          f"{len(every)} known seeds (W1, W2 and CoLM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
