# P5.15 Addendum 27: Step 4 Phase A complete; x = 0 minimises F on SRP1. Stop for review

**Planner report, 2026-09-21.** This report is written for the External Expert and the Author and is self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 27, with the author's decisions, and `STEP4_DFO_METHOD.md`.
- **Frozen spec:** v15, `data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`.
- **Evidence tables:** `data/SRP1/Results/P515S45/phase_a_tables/phase_a_tables.md` (`c1b64278`). Every statistic below is
  computed there by a committed script, and its formula is stated.
- **Independent review:** the Advisor reviewed the interpretation (§6).
- **Objective convention, used on every table:**
  - Q = certified `gross_operational_cost`, with the settlement excluded.
  - F = I(x) + Q, where I(x) uses the corrected cost file.
  - Salvage and the settlement remainder are reported but excluded from F.
- **Configuration:** AA-on from the case file throughout. Nothing is running.

## 1. Verdict

1. **Phase A is complete: 68 of 68 evaluations certified.** Every evaluation met the 10-cycle Boyd bar within cap 500. There were no barrier points
   and no failed evaluations.
2. **x = 0 has the lowest F of all 64 distinct candidates.** It is lower than every size (E from 0.5 to 5 MWh), both durations (2 h and 4 h), all three nodes, all three investment
   years, and combinations of two or three nodes.
   - F(x = 0) = 653,859,461.
   - The smallest margin is **+52,801**, at node 5 with 0.25 MVA / 0.5 MWh. That is 3.4× the combined resolution of the two bars
     (15,379) and 158× their summed terminal steps (220 + 115).
   - The deficits grow with size, to +849,076 at the lattice C\*.
3. **The objective is linear and separable within resolution. That, not noise, is why x = 0 wins.**
   - **Linear.** Operational value rises linearly with energy: 227,727 ± 3,714 EUR per MWh. Power adds only
     51,714 ± 8,307 EUR per MVA. The unit costs are 253,878 EUR per MWh and 256,317 EUR per MVA.
   - **Additive across nodes.** Three nodes give 1.000 × the sum of their single-node values, a 13 EUR difference. Every pair is within its
     resolution.
   - **The mechanism is in the model structure.** The shared storage injects at the DSO's reference (interface) bus
     (`model_construction_helpers.py:1211-1216`). It therefore cannot relieve the DN interface branch that binds at node 7. What it sees is a
     system-wide price signal.
4. **Storage falls about 20 % short of paying for itself at 2025 prices, and investing later makes it worse.**
   - At 2025 the value/cost ratio is 0.78–0.82 at 4 h and 0.64–0.68 at 2 h.
   - Moving the best unit to 2030 cuts I by 26 % but its value by 35 %.
   - x = 0 wins under the gross convention, and also under a net-of-salvage one.
5. **Paper-scale timing:**
   - **The build now fits.** Snapshot clones off, peak 19.6 GiB.
   - **One cycle was not timed.** Both attempts ran out of memory during initialization.
   - **Measured per-block solve times** give a projection of about **1,272 s per cycle (lower bound)** and **32.6–53.8 h per certified evaluation** (central
     40.9 h).
   - **Memory is the binding constraint.** Memory reached ~36 GiB after 75 of 83 initialization blocks. A paper-scale evaluation does not fit this 32 GiB
     machine as built.

## 2. Case file, cost file, audit (Addendum 27 items 1–3, 5b)

| item | result | commit |
|---|---|---|
| AA-on (`keep_memory`) in `SRP1_params.json`; the loader reads it | Case files without the key load unchanged (checked on 5 case files) | `b5629311` |
| Case-file-alone re-verification at C\* | **107 cycles, 650,982,939.9389359**, identical on every trajectory field. The only differing field is `terminal_salvage_value` (×1.25 exactly, ≤ 8.4e-35 EUR), because salvage is priced from the corrected cost file. Ruled PASS. | `b2c86a1a`, `P5_15_S45_REVERIFY_RULING.md` |
| AA saving falls with storage size | 23 / 17 / 21 / 4 % at C\* / paper plan / node-7-empty / 2×C\* (recorded with the adoption) | — |
| Corrected cost file | `SRP1_ESS.xlsx` from `7ce1d1ab`, sha256 `e17bd588…e39cd6`. Energy costs are ×1.25 exactly; power costs are unchanged (cell-level diff). | `2cada62b`, `9e623dd3` |
| I(x) of the reference candidates (new file) | paper plan 1,237,798 (−238k vs budget); C\* 3,696,250; lattice plan 1.5 / 3.0: 1,146,109 (**now over budget**) | `9e623dd3` |
| Budget frontier: largest feasible E | 2 h: 2 / 3 / 4 MWh at 2025 / 2030 / 2035. 4 h: 3 / 4 / 5 MWh. The same at every node. | `9e623dd3` |
| Probability audit | **No defect** at SRP1 or at paper scale. Every operational term uses the networks' own probabilities. The workbook's investment-cost scenario probabilities reach only I(x), the budget and salvage. The attribute name is misleading and is left as it is, because preserved pickles carry it. | `6b343bb2`, `P5_15_S45_PROBABILITY_AUDIT_NOTE.md` |
| Salvage | A reporting expression only: not in the ESSO objective, not in I(x). It is ~0 for 2025 cohorts and material for 2030/2035 (up to 350,752). Excluded from F. | — |
| Node 7 wording | "The DN's interface branch at node 7 is the only active interface constraint". "Wear cost" is replaced in `REVISION_CONTEXT.md`. | `610d3b71` |

## 3. Phase A results

### 3.1 A0: robustness batch, 8 points, all certified (`750f96de`)

- **Certification cycles:**
  - x = 0 certifies at 132.
  - The 0.25 / 0.5 units at nodes 5 / 7 / 9 certify at 134 / 139 / 140.
  - The 0.25 / 1.0 units certify at 128 / 125 / 125.
  - The lattice plan (1.5 / 3.0 at node 7) certifies at 112.
- **Batching:** A0 ran as 7 + 1 at concurrency 7. The eighth point was launched only after all seven certified, which is the reading of "a non-certified point stops".

### 3.2 A1a: 30 single-node ladders at 2025, all certified (`d0dbd186`)

Each node alone, with E ∈ {1..5} MWh at 2 h and at 4 h. **F > F(x = 0) at every point.**

| node 7, 4 h | 1 MWh | 2 MWh | 3 MWh | 4 MWh | 5 MWh |
|---|---|---|---|---|---|
| Q reduction | 261,808 | 486,622 | 735,864 | 985,614 | 1,208,712 |
| I(x) | 317,957 | 635,914 | 953,871 | 1,271,828 | 1,589,785 |
| F − F(x=0) | +56,150 | +149,292 | +218,007 | +286,214 | +381,073 |

- **4 h dominates 2 h** at every energy and node: it gets the same energy with less power.
- **Node differences at equal size are 10–17k**, which is within resolution.

### 3.3 A1b: year ladder at node 7, 20 points, all certified (`fcffdb35`)

The best node by F over A1a is node 7. The ladder covers both durations and 2030 / 2035.

| 4 h, 1 MWh at node 7 | I(x) | Q reduction | salvage | F − F(x=0) (gross) | net of salvage |
|---|---|---|---|---|---|
| 2025 | 317,957 | 261,808 | 0 | +56,150 | +56,150 |
| 2030 | 236,168 | 170,354 | 22,605 | +65,814 | +43,209 |
| 2035 | 197,990 | 83,217 | 66,750 | +114,773 | +48,023 |

- **x = 0 wins under both conventions.**
- **Crediting salvage would reorder the ladder**, putting 2030 first, but it would not change the verdict.

### 3.4 A2: presence design and lattice C\*, 5 points, all certified (`8f3f2099`)

The per-node best setting is 0.25 / 1.0 at every node (computed, not assumed).

| combination | measured value | sum of singles | ratio | difference vs its resolution |
|---|---|---|---|---|
| 5 + 7 | 496,281 | 506,551 | 0.980 | −10,270 (resolution 34,380) |
| 5 + 9 | 506,764 | 501,763 | 1.010 | +5,002 (31,020) |
| 7 + 9 | 485,213 | 518,827 | 0.935 | −33,615 (53,316) |
| 5 + 7 + 9 | 763,583 | 763,570 | **1.000** | +13 (65,755) |
| lattice C\* 1.0 / 4.0 at all three nodes | 2,966,408 | 2,998,915 | 0.989 | −32,508 (58,158) |

- **No difference is determinate against its bars.**
- **Against terminal steps alone**, which is a tighter resolution, 7 + 9 and C\* are about 5× theirs. There is a hint of mild sub-additivity, but it is too small to matter.
- **Staging is not evaluable yet.** The candidate form carries one year. The A2 staging probe is deferred (§7 Q4).

### 3.5 A3: resolution probe, 5 points, all certified (`5aef8cf6`)

Five half-MWh lattice points at node 7, in the budget-feasible region. They are fitted together with the 11 other node-7 2025 points (n = 16):

**value(E, P) = 14,295 (± 6,194) + 227,727 (± 3,714)·E + 51,714 (± 8,307)·P, residual rms 10,285, max 18,450.**

- **σ_Q ≈ 10–18k EUR, or 1.6–2.8e-5 of Q.** That is 4–7× finer than the provisional 1.1e-4, and below one lattice step's cost (0.25 MVA
  = 64,079; 0.5 MWh = 126,939).
- **§5.2's degradation clause does not trigger.**
- **Scope of the probe:** the 9 budget-infeasible half-step candidates were excluded by the Planner, because they lie above 3.5 MWh, out of Phase B's reach.

### 3.6 Break-even, from the fitted surface (zero solves; T3)

| quantity | corrected costs | original costs |
|---|---|---|
| Energy cost at which the smallest 4 h unit pays (measured value) | 197,728 EUR/MWh (0.78 × current) | 0.97 × old |
| Energy cost at which a marginal 4 h MWh pays | **176,576 ± 2,345** (0.70 × current) | 0.87 × old |
| Value multiplier needed: 0.25 / 1.0 unit | ×1.21 | ×1.02 (**indeterminate**) |
| Value multiplier needed: 1.25 / 5.0 unit | ×1.32 | ×1.10 |

Marginal value/cost ratio by duration, (b + c/h) / (e + p/h). Values above 4 h are extrapolated beyond the evaluated range.

| duration | 2 h | 4 h | 6 h\* | 8 h\* | 10 h\* |
|---|---|---|---|---|---|
| corrected costs | 0.66 | 0.76 | 0.80 | 0.82 | 0.83 |
| original costs | 0.77 | 0.90 | 0.96 | 1.00 | 1.02 |

\* extrapolated

**Readings:**
- **The ×1.25 correction decides the sign at the smallest unit.** Under the original file the 0.25 / 1.0 unit is 5.4k short, which is inside σ_Q.
- **Under the corrected costs, no duration within the method's 2–4 h bound pays.** Even extrapolating to 10 h, the ratio reaches only 0.83.

## 4. Paper-scale bounded task (item 5)

| part | status |
|---|---|
| (a) Snapshot clones switchable | **Done** (`b1272593`). A new mode `'off'` is added; the default is unchanged. A zero-solve gate is at 0 solves. Bitwise gates at SRP1: 2 cycles (`31f1ba6e`), and 8 cycles through real local-solve failures and a real snapshot write (`f1136563`). In both, `off` has 0 clones and 0 captures, against 24–26 / 48–192 for `on`. **SRP1 campaigns keep snapshots on** (see below). |
| (b) Four single-scenario paths | **Change plan only; not implemented.** The timed cycle touches none of the paths (surveyed). They matter only for a full paper-scale evaluation, which memory currently rules out. The survey found the hull-polish helpers and the s31c settlement reporting read scenario (0, 0) only, which is wrong at paper scale. Proposed: fail loudly above 1×1 until generalized. |
| (c) Build re-measured; one cycle timed, run alone | Build **fits** (19.6 GiB against the old abort). The cycle was **not timed**; the two attempts are described below. |

**Why SRP1 campaigns keep snapshots on.** This deviates from "off for campaign and verification runs". At SRP1 the saving is ~0.25 GiB, and snapshots preserve
first-failure forensics. The two settings are bitwise-identical in results.

**Attempt r1** (`eb46be7e`) aborted on macOS `phys_footprint`, which counts compressed pages. Actual residency was 15.9 GB, and 12.1 GB was still available.

**Attempt r2** (`288cbe71`, `d35dbb17`) gated on true residency and added a swap-thrashing guard. It passed r1's abort point and finished all 60 DSO initialization solves. It then stopped on genuine swap growth (+1.15 GB in 60 s) during the TSO solves.

**Measured (T5):**
- Per-block solve time: DSO 18.95 s (n = 100), TSO 6.76 s (n = 14). That is ~29× the SRP1 block.
- Memory (`footprint_self`): +0.22 GiB per solve within a network, plus 5–6 GiB each time a network's models are built. The total reached ~36 GiB after
  75 of 83 initialization blocks.
- At SRP1 the second pass over the blocks adds a third of the first, so paper-scale memory would plateau at roughly **40–50 GiB**.

**Projection:** T_cycle ≥ 60 · 18.95 + 20 · 6.76 = **1,272 s**. T_eval = build + (K + 1) · T_cycle. With K from the SRP1 range (91–151), T_eval = **32.6–53.8 h**, central 40.9 h.

**Reading:** the provisional campaign design is not executable on this machine as built. That design searches on SRP1 and evaluates the final incumbent at paper scale.

## 5. Process record: deviations and their reasons

- **Concurrency 7, not 8 or 10.**
  - Measured peak is 2.3–2.55 GiB per evaluation, with ~21 GiB non-reclaimable-free.
  - A0's first spec (concurrency 8) was superseded before any run, with its predecessor hash recorded.
  - Running 7 at once costs no per-cycle time relative to serial (~37 s/cycle).
- **Harness fixes before A0** (`c1469fab`):
  - The bar was labelled "gross step" but computed on the net recourse. It is now the gross step. Every committed bar is unchanged (max |Δ| 0.0).
  - A case-file-AA evaluation could pass as a D reference. That is fixed.
- **Harness year support** (`85c147fa`). The harness accepted only 2025, so the year ladder was blocked. Single-cohort any-year support was added; every committed key is unchanged. Multi-cohort staging is not supported.
- **Operational stop rule for A1–A3.** Stop if ≥ 2 non-certified points fall in one ladder, or ≥ 3 overall. It never fired.
- **The year ladder runs both durations (20 points).** The spec's "best node and duration only: 20" is internally inconsistent: one duration gives 10 points.
- **The A3 anchor with x = 0 as incumbent** is the best non-zero point over A1's ladders (node 7, 0.25 / 1.0). The all-priors argmin (node 5,
  0.25 / 0.5) differs from it by 3,349 EUR, which is indeterminate. Both are recorded.
- **Determinism, a fifth reproduction.** Four candidates shared between A0 and A1a reproduce bitwise across campaigns, on every per-cycle field.

## 6. Independent review (Advisor) and the caveats it attaches

The Advisor confirms the aggregate claim and the linear/separable reading, with high confidence, and attaches the following caveats:

1. **Scope.** "x = 0 minimises F" is a claim **within the frozen AA-on configuration**. The smallest margin (+52.8k) is below the 0.011 % ≈ 72k
   spread between configurations at C\*. The 2–5 MWh deficits are 10–60× every noise figure.
2. **The end-of-block SoH convention is conservative against storage.** Available energy in each 5-year block uses the **end-of-block** SoH
   (`shared_energy_storage_data.py:584, 665-668`): 0.83 / 0.71 / 0.62 at the best point. A mid-block convention would raise available energy by
   ~9 %. This is the Advisor's arithmetic, not a solve. It is a formulation choice, not a defect, but it is the size that matters at the smallest unit.
3. **Single scenario.** SRP1 has 1 × 1 scenarios, so there is no option value under uncertainty. Storage value is convex in the price spread, so the expected value over
   25 scenarios could exceed SRP1's. The size of that effect is **unquantified**, and it is the only unmeasured factor of the right order to close a ~20 % gap
   at small sizes.
4. **Settling.** 4 of 68 points have a terminal step ≥ 3,000 with a monotone window. None is within 150k of x = 0. In the decision-relevant
   subset, every terminal step is < 3.1k, so a size-dependent bias from unfinished descent is excluded where it matters.
5. **Settlement remainder.** It ranges from −39.5k to +55.1k across evaluations. It is correctly excluded from Q, but it is of the same order as σ_Q.

## 7. Questions for the Expert and the Author

1. **Phase B.**
   - **What it would do as specified.** From x = 0 under the €1M budget, every feasible poll neighbour (0.25 / 0.5 and 0.25 / 1.0 at each
     node) is already evaluated and worse. Phase B would terminate at x = 0 with essentially zero new solves.
   - **Recommendation:**
     - run it only as the formal poll record, which is cache-served;
     - spend the evaluation budget on items 2 and 3 instead.
   - **Question:** do you agree?
2. **SoH convention sensitivity (proposed; changes the model, so needs authorization).** One labelled sensitivity arm at the
   smallest node-7 unit, with available capacity at start-of-block SoH, or ageing neutralized as an upper bound. A value rise under 10 % would exclude ageing as the decider. A rise over 20 % would need your ruling on the convention before "x = 0" is reported as a property of the
   system.
3. **Option value (proposed).** A reduced-scenario SRP1 variant (e.g. 5 market × 1 operation, 48 blocks) at x = 0 and the smallest node-7 unit.
   - Estimated cost: ~12–16 GiB memory, one evaluation at a time, ~5–10 h per certified evaluation. That is 1–3 days for two or three points.
   - A scenario-mean / SRP1 value ratio above 1.25 would make paper scale decisive and qualify the SRP1 claim.
   - Do you want this, and at which scenario split?
4. **Staging.** It needs a multi-cohort candidate form; the current form carries one year. Given that delay lowers value faster than it lowers cost, is staging still wanted?
5. **Paper scale.** A paper-scale evaluation needs ~40–50 GiB and ~33–54 h, and does not fit this machine. The options are:
   - (a) a machine with ≥ 64 GiB;
   - (b) reduced scenarios (item 3);
   - (c) memory work on the per-block retained state, a bounded diagnostic;
   - (d) SRP1-only results with the single-scenario caveat stated.

   Which do you want?
6. **Manuscript.** Should the result be stated as the break-even table (§3.6) as well as the optimum? The break-even table is the transferable finding: the
   energy cost at which storage pays, 176.6k EUR/MWh at the margin and 197.7k for the first unit.
7. **Method document.** STEP4 §1.4's step costs predate the correction; the energy step is now 126,939. STEP4 line 34 says
   `max_energy_to_power_factor` goes to 4, but the case file still reads 10. This is harmless, because the harness enforces 2–4 h. Please update.

## 8. Evidence index

| item | commit |
|---|---|
| cost file; spec v15 | `2cada62b`; `0a188005` |
| AA-on case file; comparator; re-verification spec, run and ruling | `b5629311`; `11f5ed1e`; `25470496`, `b2c86a1a` |
| I(x) with the corrected file; probability audit | `9e623dd3`; `6b343bb2` |
| harness fixes and A0 launcher; A0 specs (superseded, then used); A0 run | `c1469fab`; `6d8f4720`, `8bd0102c`, `99b81181`; `750f96de` |
| snapshot switch; bitwise gates at 2 and 8 cycles | `b1272593`; `31f1ba6e`, `f1136563` |
| scale script under AA-on and SRP1 calibration; watchdog change; paper r2; paper r1 | `e6d7ddb9`; `288cbe71`; `d35dbb17`; `eb46be7e` |
| year support; A1 launcher; A1a spec and run; A1b spec and run | `85c147fa`; `fbb1296f`; `0f8aa240`, `d0dbd186`; `ccd248b4`, `fcffdb35` |
| second I(x) table; a2/a3 specs; A2 run; A3 run | `1f2ca6f7`; `43eb2c0f`, `aea5182e`; `8f3f2099`; `5aef8cf6` |
| Phase A tables T1–T5 | `c1b64278` |

**Evidence conventions:**
- Every zero-solve claim is backed by an armed `SolveProfileGuard`.
- Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the campaign manifests, not committed.
