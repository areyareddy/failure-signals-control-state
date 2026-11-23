"""Read-only integrity check for the W2 artifact.

1. Every file matches FILE_MANIFEST_SHA256.csv.
2. Every SQLite database passes PRAGMA integrity_check.
3. The S4 files frozen before generation match audit/cap_policy_hashes.json: the frozen
   digest, or, for the files changed by the recorded amendment, the post-amendment digest.
4. The copied predecessor results match provenance/predecessor/DIGESTS.json.

Exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FROZEN_FILES = {
    "analysis": "scripts/s56_cap_policy_analysis.py",
    "nvml_logger": "scripts/s55_nvml_logger.py",
    "plan": "audit/cap_policy_plan.json",
    "seed_registry": "audit/seed_registry.json",
    "seed_registry_script": "scripts/s53_seed_registry.py",
    "runner_driver": "scripts/s57_run_cap_policy.py",
    "spec": "spec/CAP_POLICY_SPEC.md",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures: list[str] = []

    manifest = ROOT / "FILE_MANIFEST_SHA256.csv"
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = ROOT / r["path"]
        if not p.exists():
            failures.append(f"missing: {r['path']}")
        elif sha(p) != r["sha256"]:
            failures.append(f"digest mismatch: {r['path']}")
    print(f"[1] manifest: {len(rows)} files checked")

    dbs = sorted((ROOT / "data").glob("*.sqlite"))
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
        if ok != "ok":
            failures.append(f"integrity_check failed: {db.name}")
    print(f"[2] databases: {len(dbs)} checked")

    hashes = json.loads((ROOT / "audit/cap_policy_hashes.json").read_text(encoding="utf-8"))
    after: dict[str, str] = {}
    for a in hashes.get("amendments", []):
        after.update(a.get("sha256_after", {}))
    for key, rel in FROZEN_FILES.items():
        p = ROOT / rel
        if not p.exists():
            print(f"    {key:22s} {rel:40s} not shipped")
            continue
        got = sha(p)
        if got == hashes["sha256"].get(key):
            status = "MATCH (frozen)"
        elif got == after.get(key):
            status = "MATCH (recorded amendment)"
        else:
            status = "MISMATCH"
            failures.append(f"frozen digest: {rel}")
        print(f"    {key:22s} {rel:40s} {status}")
    print("[3] S4 frozen digests checked")

    prov = ROOT / "provenance/predecessor"
    for d in json.loads((prov / "DIGESTS.json").read_text(encoding="utf-8")):
        if sha(prov / d["file"]) != d["sha256"]:
            failures.append(f"predecessor digest: {d['file']}")
    print("[4] predecessor digests checked")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
