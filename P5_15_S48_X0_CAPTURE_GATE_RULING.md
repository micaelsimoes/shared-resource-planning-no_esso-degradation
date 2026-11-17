# P5.15 Addendum 32 Q4: x = 0 capture reproduction gate. Strict FAIL on echoed declared parameters; ruled PASS

**Planner ruling, 2026-09-22.**
- **Spec:** v18, `data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json`, section `Q4_terminal_tso_capture.gate`.
- **Run:** `s48_x0_capture` (spec `4a50c0e2`, evidence `c23a6cd8`).
- **Gate script:** `p515_s48_x0_capture_gate.py` (`49b3a9c3`), committed before the run.
- **Reference:** A0 x0, `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/`, C3-era. At x = 0 the result is independent of ageing (W21 NL identity, `2466401d`).

## What reproduced exactly
- **Certification:** cycle 132.
- **Gross cost:** 653,859,461.2279255, the same float.
- **Identical files (0 differences):** `per_cycle_record` (all 132 rows), `pf_entry_stride`, `ess_entry_stride`, the recourse-jump sidecar, `aa_per_cycle`, `component_levels_terminal`, both interface files, `ess_exempt_until_state`, and `g_s39_D.cycle_trajectory`.
- **Rule-ten ratio:** 0.00336, identical.
- **SoH-floor sidecar:** identical once `soh_min` is set aside. SoH values, floor duals and EFC are all unchanged.

## What the strict classifier flagged (2,425 differences)

| field | A0 (C3) | this run (baseline) | count |
|---|---|---|---|
| `soh_min` in the SoH-floor sidecar | 0.5 | 0.7 | 2,376 |
| `soh_min` in `boyd_terminal.json` | 0.5 | 0.7 | 18 |
| `k_in_force` in `g_s39_D.json` | 11,541.56 | 35,851.36 | 3 |
| `degradation_fraction_per_year_per_cohort_year` | 0 | 0.0727835 = 1 − 0.985⁵ | 27 |
| `esso_models_pickle.bytes` | 2,656,218 | 2,656,632 | 1 |

Every one of these is a **declared ageing parameter echoed into a diagnostic file**. None is a solved quantity. At x = 0 the storage rows are inactive, so their parameters cannot influence the solution. The unchanged trajectory, strides, component levels and SoH values confirm it. The 414-byte difference in the ESSO pickle comes from the parameter objects it carries.

## Ruling
**PASS.** The oracle reproduces x = 0 bitwise under the baseline declaration, as predicted in spec v18.
- **Why these are not genuine differences.** The flagged fields belong to the same class as the "baseline declaration" that the task instruction listed as non-gating. The Worker's classifier narrowed that class to six evaluation-record keys before the run. The Worker was right not to reclassify after seeing the result.
- **What stands.** The strict FAIL stays in the gate output and is not overridden there. This note is the ruling.
- **For future gates across a baseline change,** the gate script declares a class in advance: `declared_parameter_echo`, covering soh_min, k_in_force, degradation_fraction and the ESSO pickle size. It is non-gating only when the reference differs from the run by exactly the declared parameters.

**Step 2 (the zero-solve bus-7 analysis) is authorized** on the persisted models, with `p515_s48_x0_capture_analysis.py` as written.
