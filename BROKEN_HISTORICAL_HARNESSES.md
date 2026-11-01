# Broken-historical diagnostic harnesses (P5.15 Step 1)

Authority: `PLANNER_BRIEF_2026-09-13.md`, Addendum 2 — "repair only
`p514_n_instrumented_cstar.py` and `p514_l_capacity_ladder.py`; list the other seven as
broken-historical."

**Count discrepancy, flagged rather than resolved silently.** Addendum 2 says *seven*.
`P5_15_PLANNER_STEP1B_REPORT.md` §4.5 listed *nine*, and re-verification below confirms nine
by name. The list is authoritative by name, not by count; the author should confirm whether
the intent was "all of them" (nine) or whether two of the nine were meant to be repaired.
Until that is answered, **none of the nine is repaired** — which is the conservative reading
of "repair only the two".

None of these is production code. None is on the path of any gate G1–G5. Each is preserved
because its **output artifacts** remain valid evidence for the stage that produced them; it is
the harness, not the evidence, that no longer runs.

## Broken by the Step 1 ESSO reformulation (`b03c9b14`)

Retired symbols: `es_degradation_per_unit`, `es_degradation_per_unit_cumul`,
`es_soh_per_unit`, `pch_hat` / `pdch_hat`, `energy_storage_complementarity`,
`energy_storage_normalization`.

| harness | retired symbols referenced |
|---|---|
| `p513_e_gated_capture.py` | `es_degradation_per_unit`, `es_degradation_per_unit_cumul`, `es_soh_per_unit`, `pch_hat`, `pdch_hat` |
| `p54h1_gate.py` | `pch_hat`, `pdch_hat` |
| `p54f_admm_net_pq.py` | `pch_hat`, `pdch_hat` |
| `p54d2_sensitivity_root_cause.py` | `pch_hat`, `pdch_hat` |
| `p54d2p_validation.py` | `pch_hat`, `pdch_hat` |

## Broken by Step 1b Candidate 2 (`feb7b5fe`)

Capacity `Var`s replaced by `_fixed` `Param`s; `network_data._get_sensitivities` deactivated
behind `_BENDERS_SENSITIVITY_CHANNEL_RETIRED`.

| harness | cause |
|---|---|
| `p55c_c0_traces.py` | reads `shared_es_s_rated` / `shared_es_e_rated` as `Var`s; consumes sensitivities |
| `audit_p3_snapshots.py` | reads `shared_es_s_rated` / `shared_es_e_rated` as `Var`s |
| `p54d_lifecycle_sensitivity_audit.py` | consumes the retired sensitivity channel |
| `validate_vmag_refactor.py` | reads the capacity `Var`s; consumes sensitivities |

## Degrades gracefully — NOT broken

`p57_fingerprint.py` references `pch_hat` / `pdch_hat` but tolerates their absence; it runs
and reports a fingerprint over the surviving components. Its fingerprints are therefore **not
comparable across the reformulation boundary** — a pre-`b03c9b14` fingerprint and a post-
`b03c9b14` fingerprint differ because the component set differs, not because the model drifted.

## Repaired, because gates G1–G3 depend on them

`p514_n_instrumented_cstar.py` and `p514_l_capacity_ladder.py` — see `WORKER_REPORT.md`.

## Not repaired, by decision

`convex_oracle.py` (P5.5-C) — marked historical in its own module docstring. Reviving it
requires a modelling decision about the shared-ESS reactive limit in W-space, not a repair.

## Broken by the Step 3.1 signed-table implementation (2026-09-15)

| harness | cause |
|---|---|
| `p53b3_active_power_ess.py` | asserts `model.penalty_ess_usage` is zeroed under ADMM; row 8 split the weight into shared (zeroed) and local (kept), so the local Param is no longer zero |

Not repaired, per scope discipline; its historical artifacts remain valid for the stage that produced them.
