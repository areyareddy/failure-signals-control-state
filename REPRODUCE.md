# Reproducing the reported numbers

All commands run on a CPU from the repository root with Python 3.10+ and only the standard
library. They read the frozen records and write only into `audit/`.

## 1. Integrity check

```bash
python scripts/verify_artifact.py
```

This checks every file against `FILE_MANIFEST_SHA256.csv`, runs `PRAGMA integrity_check` on
every database, checks the S4 frozen digests in `audit/cap_policy_hashes.json`, and checks
the predecessor digests in `provenance/predecessor/DIGESTS.json`.

## 2. Added control analyses

```bash
python scripts/s61_cr_s1_controls.py    # about 3 minutes; writes audit/cr_s1_controls.json
python scripts/s62_cr_s2s4_controls.py  # about 10 minutes; writes audit/cr_s2s4_controls.json
```

Both outputs should be byte-identical to the committed files. `s61` stops with an assertion
error if its replays do not reproduce the predecessor's published ablation values.

## 2b. S2 accounting and prefill crossover

```bash
python scripts/s28_w2_accounting.py      # writes audit/w2_accounting.json and audit/w2_accounting_rows.csv
python scripts/s42_prefill_crossover.py  # writes audit/prefill_crossover.json
```

Both outputs should be byte-identical to the committed files. They give the S2 phase
accounting (Tables 13 and 14) and the prefill crossover in Section 4.

## 3. S1 audits

```bash
PREDECESSOR_DIR=predecessor python scripts/s60_dev_retryability.py
```

This recomputes the development and all-root ledgers and asserts that the shipped class
order equals the development-only refit and differs from the all-root refit.

## 4. S4 analysis

```bash
python scripts/s56_cap_policy_analysis.py
```

This reruns the frozen S4 analysis: prefix gate, per-block yield, cost, and energy integrated
from `audit/telemetry/`.

## 5. Agent harness

```bash
python scripts/s34_agent_loop_replay.py
python scripts/s43_agent_loop_mutations.py
```

These replay the S2/S3 fixed per-root plans through `src/agent_loop.py`, and inject seven
faults one at a time. The harness does not execute the S1 scheduler; S1 runs in
`predecessor/scripts/simulate_scheduler.py`.

## 6. Paper

`paper/source/` holds the LaTeX source, the generated macro files and table fragments, and
the figures. It compiles with any TeX distribution that has the NeurIPS 2026 style file
(included). The asset builders in `scripts/` were run from the original working tree, whose
layout differs from this repository; their outputs are the committed `paper/source/*.tex`
macro files and tables.
