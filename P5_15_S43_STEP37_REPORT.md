# P5.15 Addendum 24 — Step 3.7 fails its cycle gate (109 > 80); Anderson acceleration stays off; the campaign runs D

**Planner report, 2026-09-19; stop for review.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 23–24. Frozen specs:
v12 `data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` (Step 3.7 method, gate, adoption) and v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json` (order and predictions).

## 1. Step 3.7 verdict

| gate item (spec v12) | result | verdict |
|---|---|---|
| flag off: two-cycle bitwise identity, AA code never touches the iterate | 0 genuine diffs vs D and vs the lightweight reference; **0 AA calls** (instrumented) | **PASS** (`7f675637`) |
| (a) flag on: certification ≤ 80 cycles at C\* | **certified at cycle 109** | **FAIL** |
| (b) certified cost within 1.5e-4 of D | 650,974,431.52 — \|Q − D\| = **7,456** against 97,645 | PASS |
| (c) cost difference reconciles to generation + internal flexibility | +14,828 / −7,372; residual 3.5e-7; every other priced component identically 0 | PASS |
| (d) hull polish at the AA certified point | 48/48 solved; Δ = −2,257 (TSO −30, DSO −2,227); **0.000347 %** | PASS |

**Adoption (Addendum 23 (vi)): fail on cycles → AA stays off; the campaign runs D.** The case file is unchanged. AA
remains in the code, default off, flag-off bitwise-verified.

**Predictions.** Addendum 23's gate of ≤ 80 cycles is not met. My separately recorded expectation (spec v12: central 90
cycles, range 70–115; the gate "judged uncertain before the run") contains the observed 109.

## 2. What Anderson acceleration did

| | D (AA off) | AA on |
|---|---|---|
| certification cycle | 139 | **109** (−22 %) |
| first pass V / ESS / PF | 48 / 75 / 125 | 43 / **63** / **99** |
| ρ_ess schedule | 0.01 → 0.015 (32) → 0.0225 (33), frozen 43 | 0.01 → 0.015 (30) → 0.0225 (32) → **0.0337 (42)** |
| certified cost | 650,966,975 | 650,974,431 (+7,456; 1.1e-5 of cost) |
| wall clock (run + hull polish) | 4,893 s (run only) | 3,325 s |
| peak RSS | — | **2.547 GB** |

**Per-cycle AA record** (`aa_per_cycle.jsonl`, 109 rows):

| outcome | cycles |
|---|---|
| step accepted | 50 |
| rejected by the safeguard | 22 |
| no memory to extrapolate from (m = 0, after a reset) | 27 |
| off — all channels inside tolerance (the 10 certifying cycles) | 10 |

- **Why the gain was capped at 22%: the safeguard kept resetting the memory.** It accepts an AA step only while the
  combined residual is below its mark from before the last accepted step. That residual is not monotone here, because
  the storage channel hovers near tolerance and recovered network failures perturb single cycles. Every rejection
  clears the memory (Addendum 23 (iii)), so AA spent 27 cycles, a quarter of the run, rebuilding memory, and rarely held
  its full memory of 5 for long.
- **AA accelerated the walk.** Storage passed 12 cycles earlier and PF 26 earlier. The certifying run then began once
  every channel was inside tolerance.
- **It is a different trajectory.** ρ_ess took one extra step (to 0.0337), and the certified point lies +7,456 from
  D's, within the configuration band and reconciled to the same two components. Under the adopted rules this cost is
  never mixed with D's.
- **Resource figure for Step 5.** Peak RSS is 2.55 GB for one serial certified evaluation (macOS reports bytes). That
  bounds candidate-level parallelism on 32 GB at about **10–12 concurrent evaluations**, before OS and headroom.

## 3. The rest of Addendum 24's order

### 3.1 Helper fix and exact-fix re-run

`2051309c`, `63a4d7b5`; details in `P5_15_S42_EXACT_FIX_RERUN_NOTE.md`.
- **The fix.** `p56a_oracle._interface_expression` now returns the model's own `pc_adn` / `qc_adn`. The regression check
  is exact: 1,728/1,728 entries match, and the old helper's error equals `interface_delta` exactly. All preserved
  fixtures still unpickle.
- **The re-run at D's point:**
  - **The prediction scored 5 of 12.** All six spring/summer TSO blocks solved, although they were predicted
    infeasible; five of the six autumn/winter blocks solved; TSO 2025 Autumn failed near-feasibly, at 1.3e-6 pu in
    reactive node-balance rows. The same five DSO-2025 blocks failed numerically.
  - **The void v11 run is explained.** With the fixed helper 11 of 12 TSO blocks solve, against 0 of 12, so the stale
    helper caused that run's failures.
  - **The rating-midpoint check measured a DSO-side quantity.** The 100 MVA figure is the DSO node-7 interface branch
    rating (`network.py:78-88`, which refuses a transmission network), not a TSO row. In the DSO7 blocks, where it does
    apply, the midpoint excess (~5e-6 pu) is inside IPOPT's tolerance.
  - **Manuscript consequence.** Addendum 24's conditional clause ("exact fixing is infeasible where the node 7
    interface is at its rating") **is not supported and must not be used.** The principle sentence stands.
- **Certified D models are now persisted** (165.5 MB, hash-recorded). Future polish variants need no 80-minute
  reproduction.

### 3.2 Persistent-worker bounded task (the path is now paused)

`16a19456`, `ff16e766`.
- **DSO node-7 per-cycle clone removed.** The lightweight capture is now the default, with the bitwise gate passing and
  clones going from 12 to 0 per cycle. With it, **no whole-model clone remains on the serial per-cycle path.**
- **Bound-restore fix.** W1001 warnings are now 0.
- **Parallel (8) vs serial: one field still differs** (`esso_models_pickle.bytes`). The cause is Pyomo's process-wide
  stale-flag generation counter, which also differs, masked, in the earlier evidence. Numerics are identical.
- **Where the projected 5× went.** Over five cycles, IPOPT time summed across 8 workers (127 s) roughly equals the
  wall-clock dispatch time (121 s). Each worker's round trip — apply captured state, solve, serialize and return the
  model — costs about as much as the solve it parallelizes. The screening's 97% / 5.09× is recorded as mis-accounted.
- **Safety of the changes:** every run on the new code reproduces D's first two cycles exactly (298 numeric trajectory
  fields, 0 diffs), including the parallel arm.

### 3.3 Hull-polish reporting (Addendum 24)

**At D's point:**
- block-objective change **−2,012** (3.1e-6), the construction guarantee;
- system-cost change **+11,967** (1.8e-5, settlement excluded), the R2.5 headline;
- the settlement's non-cancelling remainder at the certified tolerance, **36,679** (5.6e-5), excluded from Q by
  construction;
- **no degenerate hull intervals** at D's point, so the active counts stand: V 564, PF_P 458, PF_Q 197, ESS_P 27,
  ESS_Q 1,267 of 1,728 each.

**At the AA point, for comparison:** −2,257 / +12,267 / 39,432.

## 4. Process notes

- **Gate (c) tolerance fixed before launch** (`d5327668`). The Worker's harness passed it at "5 % of the headline
  difference or 1,000". I replaced that with a residual ≤ 1.0 and every non-dominant priced component identically zero,
  matching every earlier decomposition (~1e-7).
- **A known harness non-determinism was classified.** The recourse-jump sidecar sorts two always-identical aliases
  (`generation_cost`, `economic_market_cost`) with a stable sort over a Python set, so their order depends on hash
  randomization. It is non-gating and did not fire in the flag-off gate. It is not fixed, and it affects every harness
  in this family that reports a top-10 block-delta list.
- **One background run at a time throughout.** Each run launched from committed code with its preconditions checked.

## 5. Questions for review

1. **Step 3.7:** confirm AA stays off and D is the Step 5 configuration. Or, since AA reached a 22% reduction with a
   safeguard that discards its memory on every rejection, is a bounded variant wanted — e.g. a safeguard that rejects the
   step without clearing the memory, or the Fu–Zhang–Boyd envelope form, which spec v12 permitted? That would be a new
   arm under the same gate, not a re-run of this one.
2. **Step 5 sizing:** with 2.55 GB per evaluation, plan candidate-level parallelism at about 10 concurrent evaluations
   on the 32 GB machine?
3. **The alias tie-break non-determinism** in the harness family: fix it in a small bounded task, or leave it documented?

## Evidence

| item | commit |
|---|---|
| spec v13 | `cbdedbb4` |
| helper fix, checks, re-run harness, smoke; hull counts; exact-fix re-run and note | `2051309c`, `5c537a9c`, `7a9d6d7f`, `52b31276`, `0abc08f2`, `76e486f9`, `63a4d7b5` |
| persistent-worker bounded task | `16a19456`, `ff16e766` |
| AA integration (cherry-picked) | `9a965494`, `9ced0ad4`, `84257415` |
| flag-off gate | `dbfd204a`, `c8e642c2`, `7f675637` |
| flag-on harness, smoke, tolerance fix | `1c9e74c0`, `00123a77`, `c1c0a4d8`, `d5327668`, `de2d2a40` |
| AA run | `76095561` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted certified models and per-entry strides are
hash-recorded, not committed.
