# P5.15 Addendum 25 — configuration-selection run: AA-on meets the adoption rule; D certifies across the design space; the paper-scale build does not fit 32 GB as built

**Planner report, 2026-09-19; stop for review.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 25–26;
`STEP4_DFO_METHOD.md` v1 (orientation). Frozen spec v14 `data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`
(`cdeaa940`). Cost convention: `gross_operational_cost`, settlement excluded.

## 1. Selection-run verdict

**D certifies at every candidate, and AA-on (the `keep_memory` variant) certifies faster than D at all four with
(b)–(d) passing at each. The adoption rule reads "AA adopted".** The case-file update is held until this review:
Addendum 25 places it "before Phase A".

| candidate | D certifies at | AA certifies at | cycles saved | (b) \|Q_AA − Q_D\| ≤ 1.5e-4·Q_D | (c) reconciles | (d) hull polish, 48/48 |
|---|---|---|---|---|---|---|
| C\* (0.96875 / 3.875 at 5, 7, 9) | 139 | **107** | 23 % | +15,965 (band 97,645) | residual −8.8e-8 | 0.00025 % |
| paper's plan (node 7 only, 1.62 / 3.24) | 139 | **116** | 17 % | −312 (97,954) | 3.1e-7 | 0.00027 % |
| node-7-empty (C\* at 5, 9) | 136 | **107** | 21 % | +18,526 (97,785) | 1.0e-7 | 0.00032 % |
| 2×C\* (1.9375 / 7.75 at 5, 7, 9) | 187 | **180** | **4 %** | +12,893 (97,221) | −3.6e-7 | 0.000025 % |

- **Every reconciliation holds exactly.** In each case the difference is carried entirely by generation cost plus
  internal flexibility cost; every other priced component is identically 0.
- **The margin at 2×C\* is thin:** 7 cycles. The rule is met, but at the largest candidate the saving is small.
- **Robustness question answered.** Addendum 25's question for Step 4 — "does D certify across the design space?" —
  is answered yes at these four points: 136–187 cycles, cap 500, 0 local-solve failures. 2×C\* has the most recovered
  network failures and the slowest storage channel.
- **Oracle costs at the same configuration.** These are comparable within D:

  | candidate | Q_D | Q_D − Q_D(C\*) |
  |---|---|---|
  | C\* | 650,966,975 | — |
  | paper's plan | 653,029,767 | +2,062,792 |
  | node-7-empty | 651,900,014 | +933,039 |
  | 2×C\* | 648,138,277 | −2,828,699 |

  I(x) is not included: see §5 and the author decisions it raises.

## 2. The AA variant arm (item 2)

- **The change.** A safeguard rejection now takes the plain iterate, keeps the mark, and **retains** the (w, g) pair.
  Memory is cleared only when a ρ value changes, or on a failure cycle.
- **Default path unchanged.** It is bitwise identical to the committed Step 3.7 behaviour on 24 randomized 200-cycle
  sequences.
- **Flag-off gate passes again:** 0 AA calls, 0 differences against D.
- **At C\*:** certified at **107**, against 109 for the Step 3.7 arm, with (b)–(d) passing, so it goes forward under
  the choice rule. **Both predictions missed**: expert 80–95, Planner 78–100. Keeping the memory gained only 2 cycles.
- **One judgement call.** A freeze that leaves ρ unchanged does not clear the memory, which is the committed Step 3.7
  behaviour. The ADMM map does not change at such a freeze, so I kept it.

## 3. Campaign harness (item 1) — gate passed on the Planner's ruling

- **The harness.** It is `p515_s44_campaign_harness.py`:
  - one fresh process per evaluation, single-threaded BLAS/OMP;
  - write-once directories, heartbeats, exit codes, per-cycle records and the §2.5 record;
  - one campaign lock, with the legacy lock and the campaign lock now refusing each other;
  - a frozen spec per campaign, hashed into every record;
  - an optional post-certification step: decomposition and cost band against a reference, persisted models, hull
    polish.
- **The gate.** C\* through the harness, concurrent with two other evaluations, reproduced D **exactly** on every
  numeric trajectory field, terminal artifact, numeric sidecar and cost: cycle 139, 650,966,975.2943751, 7,179 solves.
- **The strict comparator's 43 differences** are tie **order** in one diagnostic top-10 list, caused by my deliberate
  sort-key fix at `8682cfdd`. A committed analysis explains all 139 rows. My ruling is in `P5_15_S44_GATE_RULING.md`
  (`8ce0b872`).
- **Alias tie-break fixed:** both top-k lists are sorted by name, deterministic across hash seeds. A reusable tie
  classifier handles older sidecars.
- **Per-node zero storage is evaluable.** The paper's plan and node-7-empty certified with 0 local failures, and each
  zero node publishes exactly 0.0 capacity.
- **Peak memory per evaluation:**
  - 2.2 GiB for the plain run;
  - 3.4–3.5 GB when the post-certification step pickles models and runs the hull polish.

  About 10 D evaluations fit on 32 GB, or about 7 with the post-certification step.
- **Recorded deviation from spec v14:** the D arm has a hull polish only at 2×C\*. The adoption rule gates (d) on the
  AA arm only.

## 4. Paper-scale build measurement (item 4) — watchdog abort

**What "paper scale" is in this code base** (`p515_s44_scale_measurement.py`, `5f6625db`):
- **The instance:** 5 representative years {2025, 2028, 2031, 2034, 2037}, each standing for 3 calendar years (the same
  15-year horizon), × 4 days × 25 scenario combinations (5 market × 5 operation).
- **Its source:** `EXPERT_REVIEW.md` and the one historical SRP1 configuration of that shape (`SRP1.json` at
  `0198407e`). It is built from a derived case copy with `RandomSeed` kept, from data that already exist in the
  repository.

**The scaling mechanism differs from the expert's assumption.** Scenarios are indexed **inside** each block. So the
instance is **80 network blocks instead of 48, each about 25× larger**: a DSO block has about 269k variables and 191k
constraints. It is not about 2,000 blocks.

**Result** (build only, zero solves, run alone; `37e5d86f`):

| stage | process memory (footprint) | wall |
|---|---|---|
| read planning data + scenario generation | 0.44 GiB | 13 s |
| 60 DSO blocks | **16.2 GiB** | 200 s |
| 20 TSO blocks, ESSO models, ADMM preparation | **18.3 GiB** | 21 s |
| pristine TSO/DSO clones kept for failure snapshots | **> 24 GiB → watchdog abort** at 407 s | — |

The SRP1 calibration of the same script is 1.09 GiB.

**Reading:**
- **The expert's ~100 GB estimate is too high.** It assumed memory scales with block count. The model state for one
  paper-scale evaluation is about **18 GiB before any solve**.
- **What breaks the 24 GiB line is the snapshot machinery.** The per-run pristine clones, a convenience for writing
  failure snapshots, roughly double the TSO/DSO model state.
- **A single serial paper-scale evaluation is plausibly feasible on 32 GB** if those clones are disabled, or rebuilt on
  demand from the case data rather than kept. Candidate-level parallelism at paper scale would be 1 slot.
- **Cycle time was not measured,** since the build did not fit. With each network NLP about 25× larger, it cannot be
  inferred from SRP1's 35 s.
- **Code paths that do not yet support a multi-scenario instance:**
  - `p56a_oracle.load_baseline` is hard-wired to `SRP1.json` and its checksum;
  - `run_admm_arm` assumes 51 solves per cycle;
  - some code reads only scenario (0, 0);
  - the ESS workbook's investment-cost scenario probabilities overwrite `prob_market_scenarios`. This one is
    uninvestigated.

## 5. Addendum 26 confirmations (zero solves; `6aab57d8`)

1. **Cost file — the committed file cannot be confirmed as the author's corrected one.**
   - The committed file is `data/SRP1/SharedESS/SRP1_ESS.xlsx`, sha256 `1458147446e9b70190465f42cef761af9cfd8b209e91035d858e2e63941d1414`.
     Its last commit on this branch is `072b1310` (2026-02-03, a folder rename); its content was last changed on
     2025-06-12.
   - A later version exists only on `paper_revisions`: commit `7ce1d1ab`, "Costs updated.", 2026-07-29. It multiplies
     every energy cost by 1.25. It is **not** in this branch's history.
   - STEP4's quoted unit costs (≈ €256k/MVA, €203k/MWh) match the **committed, older** file.
   - **Which one is corrected is the author's call.**
2. **I(x) and budget slack** (committed file; €; 2025; B = 1e6; salvage excluded):

   | point | I(x) | B − I(x) |
   |---|---|---|
   | paper's plan | 1,073,285 | **−73,285** |
   | C\* | 3,105,985 | **−2,105,985** |
   | lattice plan (1.5 / 3.0 at node 7) | 993,782 | +6,218 |
   | lattice C\* (1.0 / 4.0 at all nodes) | 3,206,178 | −2,206,178 |

   With the `paper_revisions` file every point is over budget. `budget` and `max_capacity` are active rows of the
   Benders master only (`shared_energy_storage_data.py:338`, `:360`), not on the oracle's path.

   **`max_capacity` caps energy (5 MWh), not power** as STEP4 §1.2 reads it. **2×C\* (7.75 MWh) exceeds it at every
   node-year**; the oracle does not check it.
3. **x = 0 at every node:** construction, variables, normalization and every capture work. Solve behaviour at
   all-zero is untested under D. Per-node zeros certified fine (§3).
4. **ESSO capacity multipliers are available without extra solves.**
   - Duals are imported on every ESSO solve and already preserved: in `esso_models_*.pkl` and the persisted certified
     models, on the `rated_s/e_capacity_unit` and degradation rows.
   - They are in the ESSO's scaled ADMM objective units, and their sign convention is unestablished.
   - The energy duals are about 0 in the ESSO, because energy's value sits in the networks' state-of-charge rows.
   - They are not yet shown to be ∂Q/∂x.

## 6. Node 7 wording (recorded, per Addendum 25)

"The DN's interface branch at node 7 is the only active interface constraint." The 100 MVA rating belongs to the
distribution network's interface branch at node 7, not to any TSO row. The Addendum 24 conditional clause is struck.

## 7. Questions for review

1. **AA adoption:** the rule reads "AA adopted". Confirm, and I will write `anderson_acceleration = {enabled,
   keep_memory}` into the case file before Phase A, re-verify the case-file-alone reproduction of the AA C\* run, and
   supersede D. The 2×C\* margin is 7 cycles.
2. **Cost file:** the committed `SRP1_ESS.xlsx`, or `7ce1d1ab` from `paper_revisions`? This decides I(x) everywhere
   and whether the lattice plan is within budget.
3. **`max_capacity`:** energy ≤ 5 MWh as production has it, or power ≤ 5 MVA as STEP4 reads it? This decides the
   lattice's upper bound (2×C\* is outside production's).
4. **Paper scale:**
   - (a) disable or rebuild-on-demand the pristine snapshot clones, then re-measure the build and time one cycle;
   - (b) choose the campaign instance (SRP1 as the search instance with paper-scale verification of the final
     incumbent, or a reduced scenario count);
   - (c) authorize the multi-scenario code paths listed in §4 as a bounded task.

   The measured 18 GiB model state makes a serial paper-scale evaluation plausible on 32 GB, not a parallel campaign.
5. **ESSO multipliers:** is a solve-based validation of their sign and units wanted before §5.6's optional search step
   relies on them?

## Evidence

| item | commit |
|---|---|
| spec v14 | `cdeaa940` |
| alias fix | `8682cfdd` |
| harness, checks, gate, tie analysis, ruling | `a60ea791`…`f825b8b7`, `8ce0b872` |
| follow-ups, AA variant, flag-off, harness post-certification step, smoke | `631d4183`, `42f48b92`, `1fdf3142`, `8f2b2736`, `92ddc722`, `1371dd73` |
| variant at C\* + D at 2×C\* | `d178c5c6` |
| selection AA campaign script and dry run | `68ec56ac`, `a72dd1f4` |
| selection AA campaign | `7b7a6078` |
| Addendum 26 confirmations | `acb74b5b`, `c3960b68`, `96fa8cc8`, `6aab57d8` |
| scale measurement script, calibration, paper build | `5f6625db`, `5fad6334`, `37e5d86f` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted certified models, per-entry strides,
`esso_capture/` and `results/` are hash-recorded in the campaign manifests, not committed.
