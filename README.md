# Resource-aware retry scheduling for Lean agents

Code-and-data artifact for **Failure Signals Are Control State: Resource-Aware Retry
Scheduling for Lean Agents** (Workshop on Resource-Aware Agentic AI,
NeurIPS 2026). The paper is in `paper/W2_camera_ready.pdf`, with its LaTeX source.

The paper asks how a Lean prover should spend retries after a failed first attempt, using
the verifier's failure class as input to the retry policy. It has four studies:

| study | question | status | records here |
|---|---|---|---|
| S1 | which failed theorem gets the next retry? | inherited from the predecessor, re-audited | `predecessor/`, `provenance/predecessor/`, `audit/cr_s1_controls.json` |
| S2 | where does serving time go? | new | `data/runs.sqlite`, `data/timing_replay.sqlite`, `data/serving_regime.sqlite` |
| S3 | do more, shorter attempts help at a fixed budget? | new, prospective | `data/runs.sqlite` (Long), `data/alloc_split2.sqlite`, `data/alloc_split3.sqlite` |
| S4 | does an 8,192-token cap on truncated roots lose proofs? | new, prospective | `data/cap_policy.sqlite`, `audit/telemetry/` |

S1 comes from the earlier workshop paper *Proof Yahtzee: From Prompting to Scheduling — Failure
Signals for Token-Efficient Retries in Frozen Lean Provers*, a non-archival paper at the
COLM 2026 Workshop on Efficient Reasoning (https://openreview.net/forum?id=KJ2nFJmMbh).
The S2 recovery attempts were generated for the companion paper *Failed Proofs Are Sticky:
Rejected-Proof Perseveration in Lean Recovery* (Workshop on Who Verifies the Agents?,
NeurIPS 2026), whose artifact is https://github.com/areyareddy/w1-recovery-interfaces-artifact.

## Start here

1. `REPRODUCE.md` gives CPU-only commands that regenerate every number in the paper.
2. `python scripts/verify_artifact.py` runs the read-only integrity check.
3. `FILE_MANIFEST_SHA256.csv` lists the SHA-256 of every file.

## Layout

| path | contents |
|---|---|
| `src/` | the agent loop (`agent_loop.py`), its conformance monitor, and the frozen policy, runner and generation code |
| `scripts/` | run drivers, analyses and the four paper asset builders |
| `spec/` | frozen specifications and their digests |
| `data/` | immutable branch-level SQLite databases and the 83 frozen roots |
| `audit/` | analysis outputs, hash manifests and raw NVML telemetry |
| `predecessor/` | the predecessor's recorded S1 retry draws (4 historical + 2 x 4 evaluation draws), its scheduler simulator and ablation script, the dev/held-out split, and the reports the S1 scripts check against |
| `provenance/predecessor/` | JSON results copied from the predecessor, with SHA-256 digests |
| `paper/` | the paper PDF and its LaTeX source |

## Added control analyses

Two scripts add control analyses to the paper. Both read only existing records.

- `scripts/s61_cr_s1_controls.py`: held-out per-class ledgers, class-blind retry caps,
  truncation persistence, the class-order formula, and a recomputed theorem bootstrap. It
  asserts that its replays reproduce the predecessor's ablation table exactly.
- `scripts/s62_cr_s2s4_controls.py`: the S2 branch breakdown, repetition loops among capped
  generations, GPU-time bounds for the four-slot check, a global 8,192-token cap, and a
  theorem-clustered bootstrap for S4's energy and token savings.

## Frozen digests

`audit/cap_policy_hashes.json` records the SHA-256 of the nine files fixed before the S4 run
(its `freeze_timing` field says which were written before generation; the digests themselves
were recorded 39 minutes into block 1, together with the analysis script, before any outcome
was queried). All nine are shipped. Seven match their frozen digest exactly. Two differ
through the one recorded amendment: the frozen driver `scripts/s57_run_cap_policy.py` never
passed `--block` to the generation step, so blocks 2 and 3 would have silently generated
nothing. It was patched to pass the block from the frozen plan and to fail when a cohort
makes no progress, and `spec/CAP_POLICY_SPEC.md` gained a dated change-log row describing it.
The amendment, its impact (no branch was generated under wrong settings) and the
post-amendment digests are in the `amendments` array of that file, and the pre-amendment
copies are in `audit/*.as_frozen_20260830`. The analysis script `s56` was not modified.
`python scripts/verify_artifact.py` checks all of this.

A second departure is recorded only in `audit/telemetry/cap_policy.markers.csv`: the
specification's stop rule was a 3 GPU-hour limit, the 10-attempt canary projected 3.03
GPU-hours, and the limit was raised to 3.5 before the main run (marker `generation_start`).
The decision used cost information only; the run used 2.9 GPU-hours.

## Release adjustments

Recorded files (databases, run logs, prompts, specifications, telemetry) are the bytes
written during the studies, with one exception: a local checkout path was redacted from
`audit/cap_policy_manifest.json` and from the `model_path` column of `timing_replay.sqlite`
and `serving_regime.sqlite` (followed by `VACUUM`). No digest reported in the paper covers
those three files. The databases' `worker` column and the earlier paper's run logs under
`predecessor/runs/` are unmodified and still contain the host name and local paths of the
machine that produced them.

## Scope

Everything here runs on a CPU. Regenerating completions needs the GPU, model weights and
Lean environment listed in `ENVIRONMENT.md`; no model weights are included.
