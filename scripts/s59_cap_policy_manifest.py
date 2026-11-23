"""Step s59 -- experiment manifest and hash record for the cap-policy replication.

Captures what a reader needs to believe the run: server identity and effective argv,
executable and model digests, GPU and driver identity, and SHA-256 of the specification,
the builder, the analysis and the plan.

HONESTY ABOUT TIMING. The brief asks for the analysis to be frozen and hashed BEFORE
outcomes open. The specification, builder and plan were written before generation began.
The analysis script was not: it was written after generation started, while the run was
blinded and before any outcome was inspected. That is weaker than the ideal, and this
manifest records the distinction explicitly instead of implying a stronger guarantee.
Blindness is what makes the weaker version still meaningful -- s12_run.py --blind
withholds outcomes, token counts and wall times from stdout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_MANIFEST = ROOT / "audit/cap_policy_manifest.json"
OUT_HASHES = ROOT / "audit/cap_policy_hashes.json"


def sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ps(cmd: str) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True)
    return (r.stdout or "").strip()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("[1/3] server identity")
    argv = ps("(Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\")"
              ".CommandLine")
    exe = ps("(Get-Process llama-server -ErrorAction SilentlyContinue).Path")
    started = ps("(Get-Process llama-server -ErrorAction SilentlyContinue).StartTime")
    props = ""
    try:
        import urllib.request
        props = urllib.request.urlopen(
            "http://127.0.0.1:8080/props", timeout=10).read().decode()
    except Exception as e:  # noqa: BLE001
        props = f"unavailable: {e}"
    print(f"      exe: {exe or 'not running'}")

    print("[2/3] gpu and driver")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"      {gpu}")

    print("[3/3] digests")
    model_path = None
    if exe:
        for tok in (argv or "").split():
            if tok.endswith(".gguf"):
                model_path = tok
    files = {
        "spec": ROOT / "spec/CAP_POLICY_SPEC.md",
        "builder": ROOT / "scripts/s54_build_cap_policy.py",
        "runner_driver": ROOT / "scripts/s57_run_cap_policy.py",
        "analysis": ROOT / "scripts/s56_cap_policy_analysis.py",
        "nvml_logger": ROOT / "scripts/s55_nvml_logger.py",
        "seed_registry_script": ROOT / "scripts/s53_seed_registry.py",
        "plan": ROOT / "audit/cap_policy_plan.json",
        "seed_registry": ROOT / "audit/seed_registry.json",
        "prompts": ROOT / "data/cap_policy_prompts.jsonl",
    }
    hashes = {k: sha_file(v) for k, v in files.items()}
    exe_digest = sha_file(Path(exe)) if exe else None
    model_digest = sha_file(Path(model_path)) if model_path else None
    for k, v in hashes.items():
        print(f"      {k:22} {str(v)[:16]}")
    print(f"      {'llama-server.exe':22} {str(exe_digest)[:16]}")
    print(f"      {'model.gguf':22} {str(model_digest)[:16]}")

    manifest = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "spec/CAP_POLICY_SPEC.md",
        "server": {"argv": argv, "exe": exe, "started": started,
                   "exe_sha256": exe_digest, "model_path": model_path,
                   "model_sha256": model_digest, "props": props},
        "gpu": gpu,
        "config": "configs/primary.json",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashes,
        "freeze_timing": {
            "before_generation": ["spec", "builder", "plan", "seed_registry",
                                  "prompts", "nvml_logger", "seed_registry_script",
                                  "runner_driver"],
            "after_generation_started_before_outcomes_opened": ["analysis"],
            "statement": (
                "The specification, builder, execution plan, prompts and seed collision "
                "proof were written and fixed before any branch was generated. The "
                "analysis script was written after generation began but before any "
                "outcome was inspected: the run is blinded (s12_run.py --blind withholds "
                "outcomes, token counts and wall times) and no outcome query was made "
                "against cap_policy.sqlite before this hash was recorded. This is weaker "
                "than freezing the analysis pre-generation and is recorded as such."),
        },
    }
    OUT_HASHES.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"\nwrote {OUT_MANIFEST.relative_to(ROOT)}")
    print(f"wrote {OUT_HASHES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
