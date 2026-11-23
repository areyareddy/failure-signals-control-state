# Targeted 8,192-token cap policy — prospective new-seed replication. Frozen before generation.

**Status when this file was written:** `data/cap_policy.sqlite` does not exist. No branch has
been staged and no completion exists. Every rule below — roots, seeds, arms, caps, order,
endpoints, tests, energy definition and interpretation — was fixed before any outcome could
be seen.

| | |
|---|---|
| seed registry / collision proof | `audit/seed_registry.json` (`scripts/s53_seed_registry.py`) |
| builder | `scripts/s54_build_cap_policy.py` |
| telemetry | `scripts/s55_nvml_logger.py` |
| analysis | `scripts/s56_cap_policy_analysis.py` |
| runner | `scripts/s12_run.py --blind` |
| database | `data/cap_policy.sqlite` — new; no frozen database is read-write or mutated |
| server config | `configs/primary.json`: 1 slot, ctx 32,768, no cont-batching, no prompt cache, port 8080 |

---

## 1. Question

Does a policy that caps generation at 8,192 tokens **only for roots whose frozen stage-one
failure class is `truncated`** preserve verified yield while reducing completion tokens,
wall time and directly measured GPU energy, on seed blocks never used before?

## 2. What this is and is not

This is a **prospective seed replication on the existing 40-root held-out theorem cohort**.
It is *not* validation on new theorems. Three new seed blocks reduce sampling uncertainty
about these 40 theorems and do not broaden theorem-level generalization. Any write-up must
say so in those words.

## 3. Arm

**Arm A only** — fresh generation. Arm A's frozen payload and prompt, byte-identical to
`data/prompts.jsonl`, verified by a builder gate. No guidance block, no prior-attempt
wrapper. Naming the arm here because leaving it implicit is how two analyses come to mean
different things.

## 4. Roots and failure class

All **40 held-out evaluation roots**. The cap decision reads `outcome_state` from the frozen
`data/roots.jsonl`, which records the **stage-one** result and is fixed long before this
experiment. On the held-out 40 that is `truncated` for **15** roots and `lean_failed` for 25.

The policy therefore conditions only on information frozen before this run. It never
consults an outcome produced by this run.

## 5. Seeds

Three new blocks, bases **4,000,000 / 5,000,000 / 6,000,000**, each covering base+0..39,
one seed per root per block.

`audit/seed_registry.json` proves disjointness against every seed this project has used:
**138** distinct W1/W2 seeds over 302 (root, seed) pairs, and **820** CoLM seeds recovered
from 287 of 2,353 scanned records — **958 known seeds, zero collisions**. The audit fails
closed: if CoLM were unreachable it returns non-zero and this specification may not claim a
three-way proof.

Within a `(root, seed block)` the **same seed is used for the matched Long and Short
generation**. That is what makes the prefix gate of §9 meaningful.

## 6. Policies

| policy | cap |
|---|---|
| **Long** | 12,288 tokens for every root |
| **Targeted-8K** | 8,192 for the 15 `truncated` roots; 12,288 for the other 25 |

The two policies are **identical on 25 of 40 roots by construction**. Only 15 roots per
block differ. Consequences, stated in advance:

- 120 Long branches (40 roots × 3 blocks) are generated once.
- 45 Short branches (15 roots × 3 blocks) are generated **separately and really executed**,
  so their energy is measured rather than inferred. Extracting an 8,192-token prefix from a
  12,288-token generation would give the right completion and the wrong energy, because the
  GPU did in fact decode to 12,288.
- Total **165 generations**.
- The Targeted-8K policy's cost is `Long(25 unaffected) + Short(15 affected)`, all measured.

## 7. Execution order

Within each block, the 15 affected roots are run as **adjacent matched pairs** — that root's
Long and Short branches back to back — with the order alternating root by root, and the
leading policy alternating across blocks.

Rationale, and it is about energy rather than outcomes. Generation is deterministic given
seed, prompt and cap on a single uncached slot, so execution order **cannot** bias
verification. It can bias thermal state, and the headline energy quantity is a *difference*
between two measurements on the same root. Separating those two measurements by hours would
put thermal drift directly inside the difference. Adjacency plus alternation removes most of
it; GPU temperature and SM clocks are logged so the residual is measured, not assumed.

## 8. Endpoints

### Primary

Paired verified-yield difference, **Targeted-8K − Long**, reported **two ways, both
pre-specified**:

1. **Policy-level, over all 40 roots.** What a deployer would experience. Note that 25 roots
   are forced ties by construction, so this is a deliberately diluted quantity.
2. **Conditional, on the 15 affected roots.** The effect where the policies actually differ.

Reporting only (1) would understate the per-affected-root effect; only (2) would overstate
the policy's impact. Both, always, with the tie structure stated.

**Reported separately per seed block.** Three blocks on the same 40 theorems are 120
observations on 40 theorems, never 120 independent units. Any aggregate uses a root-cluster
bootstrap over the 40 theorems with blocks as repeated measures, 20,000 replicates,
seed 20260828, matching every other interval in both papers.

### Secondary

Realized completion tokens and percentage saved; **directly measured joules and percentage
saved**; request wall time; model calls; capped/truncated branch counts; discordant root
counts; exact McNemar with interval; heterogeneity across blocks; energy per verified
theorem; and, where a bootstrap replicate contains zero successes, the probability mass of
those replicates rather than a silently dropped ratio.

### The expected result is a cost result

Truncated roots historically verify rarely — the failure-state table shows 2 arm-A-only
rescues across 30 observations in the incomplete group. The **expected yield difference is
approximately zero**, and the informative quantity is cost. This is recorded here so that a
null yield difference is read as the anticipated outcome rather than discovered afterwards.

**No equivalence claim.** An interval crossing zero is unresolved, never evidence of
equality. Preserving yield cannot be concluded from a null at this n.

## 9. Integrity gates

1. Every proposed seed disjoint from `audit/seed_registry.json`; abort otherwise.
2. Arm-A prompts re-derive byte-identically from `data/prompts.jsonl`.
3. Exactly 15 roots classed `truncated`, from frozen `roots.jsonl`, never from this run.
4. **Prefix gate.** For every matched pair at the same root and seed:
   - if the Long completion's `predicted_n <= 8192`, the Short completion must be
     **byte-identical** to it (the cap never bound);
   - otherwise the Short completion must be an **exact byte prefix** of the Long one.
   A violation is recorded as a **finding and reported**, not merely aborted: it would also
   undermine the byte-prefix property the portfolio/cap-schedule result depends on.
5. Zero cached tokens on every server response.
6. Server identity — executable digest, model digest, GPU identity, driver, build, effective
   argv — captured in the manifest before generation.
7. No frozen database opened read-write.

## 10. Energy definition

Windows-host NVML logger sampling `power.draw`, `utilization.gpu`, `memory.used`,
`temperature.gpu`, `clocks.sm` at ~200 ms into immutable CSV segments, a new segment per
restart, with high-resolution markers for idle baseline, server-ready, and every
block/policy/branch boundary. Sixty seconds of idle baseline before and after.

- Joules by **trapezoidal integration** over each branch's `claimed_at`→`finished_at` window.
- **Raw GPU board energy is the primary value.**
- **Idle-adjusted energy is secondary and always labelled an estimate.** The server holds the
  model resident at roughly 21 W, so idle attribution is a modelling choice, not a
  measurement.
- Telemetry gaps, invalid samples, the sampling-interval distribution, temperature,
  utilization and peak VRAM are all reported.
- **If telemetry coverage is incomplete for a branch, the energy claim for it fails.** A
  gap is not interpolated.
- The WSL `nvidia-smi` is never the authority; the Windows host logger is.

## 11. Compute limit and stopping rules

- **Hard limit 3 GPU-hours**, including canary and restart margin.
- A canary runs first; its measured rate projects the full run.
- **If the projection exceeds 3 GPU-hours, stop and report.** Do not reduce roots or blocks
  to fit — that silently converts this into a different, partial design.
- A block is analysed only if **all** of its branches are `done`. A partly-generated block is
  reported as not run, never as a subset.
- No cohort is extended, re-seeded or topped up to replace failed branches.

## 12. Blinding

Cohort names contain `eval`, so generation runs under `s12_run.py --blind`: outcomes, token
counts and wall times are withheld from stdout, since a branch stopping exactly at the cap is
a truncation tell. This specification and the analysis code are hashed **before** any outcome
is opened. Unblinding happens only in the frozen analysis.

## 13. Registration tier

**Prospective and pre-specified.** Written and hashed before generation, on seeds proved
unused. It is not part of either paper's originally registered design, so it is reported as a
prospective replication with its own frozen specification — the same standing as
`ALLOCATION_SPEC.md`, and explicitly not a confirmatory endpoint of W1 or W2.

## 14. Interpretation, fixed in advance

| observed | reading |
|---|---|
| yield difference ≈ 0 with measured token/energy savings | targeted capping buys cost at no detected yield price on these theorems and seeds; report as unresolved-on-yield with the interval, never as equivalence |
| Targeted-8K clearly below Long on yield | the 8,192 cap truncates proofs that would have verified; the candidate recommendation is withdrawn or weakened |
| Targeted-8K above Long on yield | almost certainly noise at this n; report the interval and do not build a story on it |
| savings absent or negative | the targeted policy does not pay even on cost; report plainly |

A null is reported as a null. No outcome licenses a deployment recommendation from three
seed blocks on 40 theorems.

## 15. Change log

| when | what |
|---|---|
| 2026-08-30, before generation | First frozen version, written after the seed collision proof passed and before `scripts/s54_build_cap_policy.py` staged any branch. |
| 2026-08-30, mid-generation | **Execution defect, no design change.** `scripts/s57_run_cap_policy.py` invoked the frozen `scripts/s12_run.py` without `--block`, which defaults to 1, so `runner.claim()` filtered `seed_block=1` and returned no rows for all 14 block-2 and block-3 cohorts. The driver reported `rc=0` in 0.0 min for each and exited normally after completing block 1 (55/165 branches). No branch was generated under wrong settings; blocks 2 and 3 were simply never started. s57 was patched to pass `--block` from the already-frozen plan (each step carries `block`), plus a fail-fast guard that aborts if a cohort's pending count is unchanged after its runner returns. The design, caps, seeds, execution order, prompts and analysis are untouched. Frozen copy of the pre-fix driver: `audit/s57_run_cap_policy.py.as_frozen_20260830`. |
