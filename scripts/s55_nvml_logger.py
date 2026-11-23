"""Step s55 -- Windows-host GPU telemetry logger. Governed by spec/CAP_POLICY_SPEC.md s10.

Wraps `nvidia-smi --query-gpu=... -lms 200` and writes IMMUTABLE CSV segments. A new
segment file per invocation; nothing is ever overwritten or ambiguously appended, so a
restart can never blur two thermal histories into one series.

WHY THE WINDOWS HOST AND NOT WSL. The WSL nvidia-smi reports through a passthrough layer
whose sampling and timestamps are not the host's. The spec makes the host logger the
authority; this file is that authority.

MARKERS. The logger also owns a marker file. Anything that wants to bound an interval --
idle baseline, server-ready, a block, a policy, a branch -- appends a marker line with a
high-resolution timestamp. Energy for an interval is then a trapezoidal integral over the
samples inside it, with no interpolation across gaps.

USAGE
    python scripts/s55_nvml_logger.py start   --tag cap_policy      # background sampler
    python scripts/s55_nvml_logger.py mark    --tag cap_policy --label idle_start
    python scripts/s55_nvml_logger.py stop    --tag cap_policy
    python scripts/s55_nvml_logger.py integrate --tag cap_policy \
        --from-marker idle_start --to-marker idle_end
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TELEM = ROOT / "audit" / "telemetry"
TELEM.mkdir(parents=True, exist_ok=True)

FIELDS = ("timestamp", "uuid", "power.draw", "utilization.gpu",
          "memory.used", "temperature.gpu", "clocks.sm")
INTERVAL_MS = 200


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def seg_path(tag: str, idx: int) -> Path:
    return TELEM / f"{tag}.seg{idx:03d}.csv"


def marker_path(tag: str) -> Path:
    return TELEM / f"{tag}.markers.csv"


def pid_path(tag: str) -> Path:
    return TELEM / f"{tag}.sampler.pid"


def next_segment(tag: str) -> int:
    n = 0
    while seg_path(tag, n).exists():
        n += 1
    return n


# ------------------------------------------------------------------ start / stop


def cmd_start(tag: str) -> int:
    idx = next_segment(tag)
    out = seg_path(tag, idx)
    cmd = ["nvidia-smi",
           "--query-gpu=" + ",".join(FIELDS),
           "--format=csv,noheader,nounits",
           "-lms", str(INTERVAL_MS)]
    # header written by us, so the segment is self-describing even though nvidia-smi
    # is invoked with noheader
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("host_recv_iso," + ",".join(f.replace(".", "_") for f in FIELDS) + "\n")
    fh_out = out.open("a", encoding="utf-8", newline="")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    pid_path(tag).write_text(json.dumps(
        {"pid": proc.pid, "segment": idx, "started": now_iso(),
         "interval_ms": INTERVAL_MS}) + "\n", encoding="utf-8")
    print(f"sampler pid {proc.pid} -> {out.relative_to(ROOT)}")

    # Stamp the host receive time on every line. nvidia-smi's own timestamp has
    # second-ish formatting; the host clock is what the markers use, so intervals and
    # samples must share one clock.
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            fh_out.write(now_iso() + "," + line + "\n")
            fh_out.flush()
    except KeyboardInterrupt:
        pass
    finally:
        fh_out.close()
        try:
            proc.terminate()
        except Exception:
            pass
    return 0


def cmd_stop(tag: str) -> int:
    p = pid_path(tag)
    if not p.exists():
        print("no sampler pid file; nothing to stop")
        return 0
    info = json.loads(p.read_text(encoding="utf-8"))
    # Windows has no SIGTERM for an unrelated process: os.kill raises WinError 87.
    # taskkill /T also reaps the nvidia-smi child, which is the thing actually sampling.
    r = subprocess.run(["taskkill", "/PID", str(info["pid"]), "/T", "/F"],
                       capture_output=True, text=True)
    print(f"stop sampler pid {info['pid']}: rc={r.returncode} "
          f"{(r.stdout or r.stderr).strip()[:80]}")
    p.unlink(missing_ok=True)
    return 0


def cmd_mark(tag: str, label: str, note: str = "") -> int:
    mp = marker_path(tag)
    new = not mp.exists()
    with mp.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["iso", "label", "note"])
        w.writerow([now_iso(), label, note])
    return 0


# ------------------------------------------------------------------ integration


def load_samples(tag: str) -> list[dict]:
    rows: list[dict] = []
    idx = 0
    while seg_path(tag, idx).exists():
        with seg_path(tag, idx).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r | {"_segment": idx})
        idx += 1
    return rows


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def integrate(tag: str, t0: float, t1: float) -> dict:
    """Trapezoidal joules over [t0, t1]. Reports coverage rather than interpolating.

    A gap is never bridged with an assumed power level: the caller is told the gap exists
    and the spec fails the energy claim for that interval.
    """
    rows = load_samples(tag)
    pts: list[tuple[float, float, dict]] = []
    invalid = 0
    for r in rows:
        try:
            t = parse_ts(r["host_recv_iso"])
            p = float(r["power_draw"])
        except (ValueError, KeyError, TypeError):
            invalid += 1
            continue
        if t0 <= t <= t1:
            pts.append((t, p, r))
    if len(pts) < 2:
        return {"joules": None, "n_samples": len(pts), "invalid_samples": invalid,
                "coverage_ok": False,
                "reason": "fewer than two valid samples inside the interval"}
    pts.sort(key=lambda x: x[0])
    joules = 0.0
    gaps: list[dict] = []
    deltas: list[float] = []
    for (ta, pa, _), (tb, pb, _) in zip(pts, pts[1:]):
        dt = tb - ta
        deltas.append(dt)
        joules += 0.5 * (pa + pb) * dt
        if dt > 5 * INTERVAL_MS / 1000.0:
            gaps.append({"from": ta, "to": tb, "seconds": round(dt, 3)})
    span = pts[-1][0] - pts[0][0]
    lead = pts[0][0] - t0
    tail = t1 - pts[-1][0]
    temps = [float(r["temperature_gpu"]) for _, _, r in pts if r["temperature_gpu"].strip()]
    utils = [float(r["utilization_gpu"]) for _, _, r in pts if r["utilization_gpu"].strip()]
    vram = [float(r["memory_used"]) for _, _, r in pts if r["memory_used"].strip()]
    deltas.sort()
    return {
        "joules": round(joules, 3),
        "interval_seconds": round(t1 - t0, 3),
        "covered_seconds": round(span, 3),
        "n_samples": len(pts),
        "invalid_samples": invalid,
        "mean_power_w": round(joules / span, 2) if span > 0 else None,
        "gaps": gaps,
        "lead_uncovered_s": round(lead, 3),
        "tail_uncovered_s": round(tail, 3),
        "sample_interval_s": {
            "p50": round(deltas[len(deltas) // 2], 4) if deltas else None,
            "p95": round(deltas[int(0.95 * (len(deltas) - 1))], 4) if deltas else None,
            "max": round(deltas[-1], 4) if deltas else None,
        },
        "temperature_c": {"min": min(temps), "max": max(temps)} if temps else None,
        "utilization_pct": {"mean": round(sum(utils) / len(utils), 1),
                            "max": max(utils)} if utils else None,
        "peak_vram_mib": max(vram) if vram else None,
        # Coverage is strict on purpose: the spec fails an energy claim rather than
        # quietly reporting a number that skipped a hole in the series.
        "coverage_ok": (not gaps and lead <= 1.0 and tail <= 1.0),
    }


def cmd_integrate(tag: str, a: str, b: str) -> int:
    mp = marker_path(tag)
    if not mp.exists():
        print("no markers recorded")
        return 1
    marks = {}
    with mp.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            marks.setdefault(r["label"], []).append(parse_ts(r["iso"]))
    if a not in marks or b not in marks:
        print(f"missing marker: {a if a not in marks else b}")
        return 1
    res = integrate(tag, marks[a][0], marks[b][-1])
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action",
                    choices=("start", "stop", "mark", "integrate", "selftest"))
    ap.add_argument("--tag", default="cap_policy")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--from-marker", dest="a", default="")
    ap.add_argument("--to-marker", dest="b", default="")
    args = ap.parse_args()

    if args.action == "start":
        return cmd_start(args.tag)
    if args.action == "stop":
        return cmd_stop(args.tag)
    if args.action == "mark":
        assert args.label, "--label required"
        return cmd_mark(args.tag, args.label, args.note)
    if args.action == "integrate":
        return cmd_integrate(args.tag, args.a, args.b)
    if args.action == "selftest":
        # 3 s of live sampling, then integrate it. Proves nvidia-smi -lms works on this
        # driver before a two-hour run depends on it.
        import threading
        t = threading.Thread(target=cmd_start, args=(args.tag,), daemon=True)
        t.start()
        time.sleep(1.0)
        cmd_mark(args.tag, "selftest_start")
        time.sleep(3.0)
        cmd_mark(args.tag, "selftest_end")
        time.sleep(0.5)
        cmd_stop(args.tag)
        time.sleep(0.5)
        return cmd_integrate(args.tag, "selftest_start", "selftest_end")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
