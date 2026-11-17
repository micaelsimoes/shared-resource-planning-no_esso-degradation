# P5.15 Addendum 28: the ageing convention does not decide the sign. Stop for review

**Planner report, 2026-09-21.** Written for the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 28–29.
- **Frozen spec:** v16, `data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json` (`912af388`).
  All predictions were recorded in it before any run.
- **Objective convention on every table:** Q is the certified `gross_operational_cost`, settlement excluded. Value = Q(0) − Q(x).
  I(x) = 317,957 at the smallest node-7 4 h unit (0.25 MVA / 1.0 MWh, 2025, candidate `db77e154…`).
- **Resolution rule:** a difference smaller than the sum of the two bars, or smaller than σ_Q (10–18k, from `c1b64278` T3), is
  indeterminate.
- **Status:** nothing is running. The Phase B record and the memory task wait for this review.

## 1. Verdict

1. **The ageing batch ran 5 model variants and all certified** (118–123 cycles). Under none of them does the smallest unit pay determinately:
   - with no ageing at all it reaches break-even (+485, indeterminate);
   - under the datasheet convention C2 it is −23,084, which sits at the edge of resolution.

   **So Addendum 28's reading, "the sign is decided by the ageing convention", is not supported.** The convention moves
   the smallest unit from clearly negative (C3, −56k) towards indeterminate. It does not make it positive. Larger units need
   ×1.32 and stay clearly negative under every variant.
2. **The prediction erred in one assumption, not in its arithmetic.**
   - Available energy rose almost exactly as predicted (×1.22 under C2, ×1.37 under no ageing).
   - **Value is sub-proportional to available energy.** Its elasticity is 0.60–0.62 over the two resolvable variants.
   - At a fixed 0.25 MVA, the extra energy is cycled into lower-value hours. EFC/day rises from 1.01 to 1.28, while value per available MWh
     falls from 359k to 318k.
3. **Zero-solve reports:**
   - **Structural finding confirmed, with a precise wording** (Z1).
   - **Discount-rate sensitivity** matches the expert's estimate to within 0.1 point (Z3).
   - **The captured spread** is 52.1 €/MWh per full cycle (Z4).

## 2. Ageing batch (`1545571e`; mechanism table `5e1e3562`). MODEL VARIANTS, not the baseline

Q(0) = 653,859,461 is unchanged by ageing, and x = 0 was not re-run. The C3 row is the baseline, shown for reference.

| variant | cycles | value | ×C3 measured | ×C3 predicted | value − I | bars (variant + x0) | verdict |
|---|---|---|---|---|---|---|---|
| C3 baseline (EOL 0.50, k 11,542) | 125 | 261,808 | 1.000 | — | −56,150 | 20,970 | negative |
| C2 (EOL 0.80, k 35,851) | 119 | 294,873 | **1.126** | 1.23 | **−23,084** | 23,899 | indeterminate by bars (by 815); negative vs σ_Q |
| C4 (EOL 0.70, k 22,429) | 119 | 276,953 | 1.058 | 1.16 | −41,004 | 23,610 | negative |
| C2 + φ_cal 0.985 | 121 | 273,764 | 1.046 | ≈ 1.10 | −44,193 | 21,147 | negative |
| C3 with mid-block health | 123 | 264,113 | 1.009 | ≈ 1.09 | −53,844 | 24,533 | negative |
| no ageing | 118 | 318,442 | **1.216** | 1.36 | **+485** | 18,811 | **break-even (indeterminate)** |

- **Floor:** soh_min is 0.50 in every variant, and no floor row was active.
- **Read-back:** each variant was read back from its built model before its run (k, φ, the SoH point).
- **Defaults are unchanged:** the model digests are identical to before, and a two-cycle bitwise gate passed (`e5721b6d`, `8ed12e30`).

**Mechanism** (committed formulas, `p515_s46_ageing_mechanism.py`). AE is the present-value-weighted available energy per rated MWh, with the 2 % per-representative-year factors.

| | AE | ×C3 | EFC/day | value per AE (€) | elasticity ln(value ratio)/ln(AE ratio) |
|---|---|---|---|---|---|
| C3 | 0.729 | 1.000 | 1.010 | 359,080 | — |
| C2 | 0.889 | 1.219 | 1.175 | 331,765 | 0.60 |
| C4 | 0.835 | 1.146 | 1.120 | 331,523 | 0.41\* |
| C2 + fade | 0.779 | 1.068 | 1.061 | 351,467 | 0.68\* |
| mid-block | 0.780 | 1.070 | 1.064 | 338,616 | 0.13\* |
| no ageing | 1.000 | 1.372 | 1.284 | 318,442 | 0.62 |

\* not resolvable: the value difference against C3 (3–15k) is the size of σ_Q.

**Readings:**
- **The expert's multipliers are the available-energy ratios, and they were right** (1.219 against 1.23; 1.372 against 1.36). What fails is "value ∝
  available energy". It fails at fixed power: this unit has 0.25 MVA, so more energy means a longer effective duration. That
  duration is spent on hours whose spread is smaller. This is consistent with the Phase A surface, where power, not only energy, carries value.
- **The mid-block variant moves almost nothing** (×1.009, against the Advisor's ~9 % estimate).
  - With more available energy, the storage cycles more (EFC 1.06 against 1.01).
  - So its end-of-block health is lower (0.824 against 0.834 in 2025).
  - The dispatch response offsets most of the convention change. That is endogenous, and a static estimate cannot see it.
- **Break-even needs no ageing at all.** Value then equals I within 485 €. Any physical fade leaves the smallest unit negative or
  indeterminate.

**Known limitation:** in mid-block mode, the capacity-sensitivity chain rule (`shared_energy_storage_data.py` ≈1458) still uses
end-of-block health. It enters neither Q nor any campaign field, and was left unchanged.

## 3. Zero-solve reports (`dd3afa6e`)

**Z1: structural finding.** The node-7 DN interface is **one branch**, bus 1 (the DN reference bus) to bus 2, a 100 MVA transformer.
- The binding row is that branch's apparent-power limit at the bus-1 end. All 23 binding slots sit on the bound, and x = 0 has the same slot set.
- The storage injects at **DN bus 1**, and at TN bus 7 in the TSO model. Bus 1 has no other branch, no load and no shunt.
- So the interface flow is identically the DN's net draw through branch 1–2. The storage term cancels out of it.
- **Wording for the manuscript:** the storage is sited at the upstream terminal of the single constrained branch. Its
  injection enters the interface balance, not the branch, so it cannot relieve the branch. That is a consequence of the interface-siting
  design. Future work: DN-internal siting.
- **Correction to Addendum 23's wording:** the storage is **not idle** in the binding slots. It is at ≥ 99 % of its rating in 2 of 23 and at 22–79 % in
  several others. It simply cannot change the constrained flow.
- **Not evaluated:** TN-side branches at bus 7, which are not captured.

**Z2: ageing trajectory (C3 baseline).** φ_cal = 1.0 as consumed (the key is absent, so the default applies) and k = 11,541.56.

| block | EFC/day | health used (end of block) |
|---|---|---|
| 2025 | 1.145 | 0.834 |
| 2030 | 1.027 | 0.709 |
| 2035 | 0.826 | 0.622 |

Available energy in each block uses the end-of-block health.

**Z3: discount rate.** Production applies one factor per representative year to all 5 years of each block. The reconstruction at 2 % reproduces the certified
value to 8.9e-8 €.

| rate | value | vs 2 % | value-to-cost | expert |
|---|---|---|---|---|
| 0 % | 287,841 | +9.94 % | 0.905 | +≤ 10 % |
| 2 % | 261,808 | — | 0.823 | — |
| 5 % | 230,738 | −11.87 % | 0.726 | −12 % |
| 8 % | 206,911 | −20.97 % | 0.651 | −21 % |

Even at 0 % the unit falls short. The margin is 30,116, 1.18× the conservative bar, so it is determinate. The rate stays at 2 %.

**Z4: price spread.**

| measure | €/MWh |
|---|---|
| Market daily spread, max − min (weighted) | 116.9 |
| Market daily spread, top-4 h mean − bottom-4 h mean | 98.0 |
| Captured spread, value and throughput discounted consistently (reference) | **52.1** per full cycle |
| Captured spread, literal | 47.9 per full cycle |
| Captured spread, 2025 / 2030 / 2035 | 49.5 / 44.4 / 67.1 |
| Expert's estimate | 55–60 |

The storage captures a little over half of the available 4 h spread.

## 4. Other items done

- **`max_energy_to_power_factor` 10 → 4** in `SRP1_ESS_Params.json` (`cb165d4e`).
  - Readers: the Benders master, the Benders feasibility check and bootstrap, the p56a cache key, and diagnostics. None is on the oracle path.
  - Oracle model digests at C\* and at x = 0 are identical before and after (51 blocks each).
  - No campaign spec pins this file.
- **STEP4 §1.4** already carried the corrected step costs; there was nothing to change.
- **Concurrency 5** was in force for the ageing batch: one wave of 5, 21.8 GiB available against 13.75 required.

## 5. Questions for the Expert and the Author

1. **Baseline calibration and floor.** Addendum 28 leaves this to the author, with a datasheet citation. The batch shows the choice matters less than expected:
   - C2 against C3 moves the smallest unit from −56k to −23k, but not across zero.
   - If the baseline changes, one re-certification check at C\* under AA-on precedes further campaigns.

   Which calibration and floor do you choose?
2. **Phase B.** Under C3, Addendum 28 has it as the formal record. The addendum's condition for re-running it, "if the author's calibration decision makes storage pay", is
   not met by any tested variant except no ageing, which gives break-even. Proceed with the formal record under the chosen baseline, as the next step after this review?
3. **Manuscript framing.** The transferable result stays the break-even energy cost. The mechanism is an addition: value rises with available energy at an
   elasticity of about 0.6, not proportionally, at fixed power. Should the ageing-sensitivity band be reported as measured? Under C2 the smallest unit needs
   ×1.08 more value; under no ageing it breaks even.
4. **Memory task (Addendum 29).** Next after the Phase B record, run alone. The decision rule is recorded in spec v16: resident memory after initialization ≤ 26 GiB, flat across cycles, and one cycle ≤ 25 min. Confirm the order is unchanged.

## 6. Evidence

| item | commit |
|---|---|
| spec v16 | `912af388` |
| zero-solve reports Z1–Z4 | `dd3afa6e` |
| production switches, variant mechanism and zero-solve checks; case-file edit and gate; two-cycle bitwise gate; ageing launcher; frozen spec | `e5721b6d`; `cb165d4e`; `8ed12e30`; `e127daa7`; `3e0cd105` |
| ageing batch run; mechanism table | `1545571e`; `5e1e3562` |

**Evidence conventions:**
- Every zero-solve claim is backed by an armed `SolveProfileGuard`.
- Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests, not committed.
