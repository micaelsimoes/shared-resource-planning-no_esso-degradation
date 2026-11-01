# P5.15 W183f (gate part): serial SRP1 two-cycle bitwise gate re-run on the W183a/W183c code

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 83, W183 order item W183f, gate part only. The spec v1 freeze is
not part of this task. Planner task W183f (gate part).

## Blocked on Planner

1. **The gate returned `GATE_PASS=False`, as it is frozen. One item failed:
   `vs_committed_w48_arm_zero_genuine_diffs`.** The recorded prediction "the two-cycle gate passes" therefore failed
   as stated. Per the task I did not retry and did not change anything.
   - **All 44 "genuine" diffs are in `per_cycle_record.jsonl`.** They are 22 keys × 2 rows. Each key is absent in the
     committed W48 arm and has the value `null` in this arm.
   - **The 22 keys are `p515_s44_campaign_harness.PER_CYCLE_RESPONSE_FIELDS`.** W64 (`70950e8c`, 2026-09-24) added
     them to `PER_CYCLE_RECORD_FIELDS`, and `W10.run_arm` writes that tuple.
   - **The W48 arm predates W64.** It ran at `fe0319d6` and was committed at `e75a575e`, and it carries 29 keys.
   - **The 29 shared keys are bitwise equal on both rows.**
   - **This is not a W183 effect.** This arm's `per_cycle_record.jsonl` is byte-identical (sha256 `f8506905…`) to
     the committed W83 gate arm `P515S53/tight_tail_w83/srp1_bitwise_gate/arm/per_cycle_record.jsonl`. That arm ran
     after W64 and before W183, and it failed the same item with the same 44 diffs (commit `eb8cf81f`, "GATE_PASS=False
     AS FROZEN … harness field-list drift since r2").
   - **No ruling on that W83 failure was found.** I searched the brief, `REVISION_CONTEXT.md`, `TASKS.md` and the
     `P5_15_*.md` reports for W83, `tight_tail_w83`, `PER_CYCLE_RESPONSE_FIELDS` and "field-list". The W83 result
     appears only in its commit message.

   **Decision needed (Planner, or expert if it touches a gate's scope).** How should this result be read? Options
   (the Worker does not choose):
   - (a) Accept it as the known field-list drift, recorded with the evidence above. This is a scope reading of the
     gate, so CLAUDE.md routes it to the expert.
   - (b) Re-run with a declared whole-arm reference substitution to a committed post-W64 arm. The W83 arm is one
     candidate; it ran with the tail enabled but inert. The gate's convention allows this substitution, and W83 used
     it. It costs 153 solves, about 2 min.
   - (c) A declared comparator rule for keys absent in the reference and `null` in the new arm.

   Under CLAUDE.md, a gate failure other than G6/G27 stops the campaign chain (W184 onwards) until this is ruled.
2. **Scope note.** The W183 order line in Addendum 83 reads "re-run on the changed code **at both DSO builders**".
   The Planner's task scoped this gate to the sequential builder only. The arm runs with `ParallelExecution: false`,
   so the parallel builder was not exercised (see Not confirmed).

## Changed

- **`17b4c1cf`**: the wrapper `p515_s55_w183f_bitwise_gate.py` (sha256 `8fb660be…f7521b47c`), committed before the
  run. It imports the committed `p515_s53_srp1_bitwise_gate.py` (sha256 `b63b1fb9…`, unchanged) and calls its
  `main()` unchanged. These substitutions were declared, and no others were made:
  1. `STAGE` / `SCHEMA` (`p515_s55_w183f_bitwise_gate_v1`) / `ARM` (`s55w183fgate`) / `OUT_REL` →
     `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate` (write-once, fresh).
  2. `EXTRA_CLEAN_FILES`: the gate's list plus `network_parameters.py` and the wrapper.
     `shared_resources_planning.py`, `p515_s44_campaign_harness.py`, `model_construction_helpers.py`, `network.py`
     and every imported gate script were already in the list.
  3. `EXTRA_FORBIDDEN`: the gate's list plus `p515_s55_`.

  The wrapper also does two things that change nothing the gate compares:
  - a refuse-only precondition that the arm-named `P56A/evals` directories do not exist;
  - after the gate returns, it writes `w183f_wrapper_record.json` (code sha256 before/after, HEAD, substitutions,
    verdict) and `w183f_manifest_sha256.json`.

  The reference (`W32.BASELINE_REFERENCE`, sha256 `ced90e47…`), the W48 whole-arm reference (manifest `9a5d6b78…`),
  tolerances, presence assertions, the W51 counter, the declared solve count and the compared fields are all the
  gate's own.
- **This commit**: this report, the gate outputs and the launch log (listed under Evidence).
- **Code in force.** HEAD `17b4c1cf`, the source equal to production head `6f43479e` plus the wrapper.

  | file | sha256 |
  |---|---|
  | `shared_resources_planning.py` | `ade9c843bc7c5b6514ab91ecc54f327330a5249709766699ff2f50c8288473d6` |
  | `network_parameters.py` | `30ea1b23982c8839f593e98030f99d764704912dfb6b344dd8a2454d06438a92` |
  | `network.py` | `18acaa84af7a043898a7b831223b4eab1d91f9bcf2c199f49beecfeb3bf91fb4` |
  | `model_construction_helpers.py` | `a20a224f…` (= the gate's file pin) |
  | `admm_parameters.py` | `40d0e972dcadfb4de099a0203682ba2e93badc62d9ae989f87be40b1ce3b75fa` |
  | `p515_s44_campaign_harness.py` | `c34ebdc5463a1fe0fd6398c6acad4dcfb61dbfbd142ef3de517ca517168d77be` |
  | `p515_g_g1_g4_admm_gates.py` | `6f4631e10a2f8a7786390c0f96e45332706779ae9cdf6a87ddd716d9ad9ad346` |

  The code was unchanged during the run (`code_unchanged_during_run: true`).

## Found

**Prediction "the two-cycle gate passes": failed as frozen, on one item.** See Blocked on Planner 1 for the
diagnosis.

**Instance.** C\* is 0.96875 MVA / 3.875 MWh at nodes 5, 7 and 9, investment year 2025, with candidate_key
`578636da…6350c` (pin matches). The configuration is baseline, cap 2, snapshots on, NRF off on every network (read
back in `rule_eleven_checklist.w183a_nrf_option`), tail off (production default). Objective convention:
`gross_operational_cost` (settlement-excluded gross) = 740,922,817.2425401. This equals the W48 arm's value, and
here it equals the recourse.

**Rule ten (terminal step / threshold).** The ratio is 845,932,257.61 / 158,685.51 = 5,330.87. The run is stopped by
cap 2 by design, not converged. The W48 arm's ratio is identical.

**Solves.** Declared 153 = 51 × (cap 2 + 1), with 0 network-failure events, 0 retries and 0 ESSO recovery events.
- Observed 153; event-level reconciled 153; `GUARD.verify(153) == []`.
- Guard counts: permitted_solve 153, permitted_exec 153, blocked 0 / 0.
- No solves outside the arm. Wall clock 100.6 s (arm 87.4 s).

**W35 layer (`gate.json`): 12/12 true.**
- ran_two_cycles
- reproduces_committed_reference_zero_diffs (n_diffs 0 vs C\* `g_s39_D.json` rows[:2] + derived fields)
- trajectory_fields_identical_to_reference (29 fields, 0 mismatches)
- solve_reconciliation_event_level_identity_holds
- arm_instance_derived_solve_profile_agrees
- planning_derivation_agrees_with_case_file
- process_guard_verified_exactly
- no_solves_outside_the_arm
- no_blocked_solver_calls
- shared_frozen_smopf_untouched
- ess_ageing_declaration_verified_in_child
- w35_code_present_in_the_modules_that_ran

**Other layers.**
- **W39 layer.** The retired-quadratic tripwire recorded 0 calls.
- **W51 layer (`w51_gate_addendum.json`).**
  - Presence: 21/21 before the run.
  - The armed counter recorded exactly 36 calls of each of `_set_row18_inactive_for_initialisation` and
    `_activate_row18_with_settlement`, 0 acting.
  - Gate items: w39_w35_gate_pass True, w51_code_present True, w51_counter_exact_and_never_acting True,
    vs_committed_w48_arm_all_files_present True, **vs_committed_w48_arm_zero_genuine_diffs False**.
- **Pre-run presence.** The W35 layer reported 57/57 combined checks (row 18 + W47 + W51) true, matching
  W183c point 5.

**Whole arm vs the committed W48 arm** (comparator `W10.classify`; files hash-checked against the W48 manifest first):

| file | provenance | tie_order | genuine |
|---|---|---|---|
| g_s39_D.json | 3 | 0 | 0 |
| boyd_terminal.json | 0 | 0 | 0 |
| component_levels_terminal.json | 0 | 0 | 0 |
| interface_settlement_detail_s31c.json | 0 | 0 | 0 |
| interface_voltage_terminal.json | 0 | 0 | 0 |
| recourse_jump_sidecar_baseline.jsonl | 0 | 0 | 0 |
| ess_entry_stride_baseline.jsonl | 0 | 0 | 0 |
| soh_floor_sidecar_baseline.jsonl | 0 | 0 | 0 |
| pf_entry_stride_s39_D.jsonl | 0 | 0 | 0 |
| ess_exempt_until_state_s39_D.jsonl | 0 | 0 | 0 |
| aa_per_cycle.jsonl | 0 | 0 | 0 |
| per_cycle_record.jsonl | 0 | 0 | **44** |

- **The three provenance diffs** are all under `rule_eleven_checklist`: `w10_arm` (arm name), and the new
  `w183a_nrf_option` (every network flag off) and `w84_convergence_depth_tail` (enabled false) read-backs.
- **The 44 genuine diffs** are all `<MISSING>` (W48) vs `null` (this arm), for 22 keys × rows 0 and 1:
  - E_abs_d_p_mwh_{unweighted,weighted}
  - covariance_dso_weighted
  - curtailed_res_{dso,tso}_mwh_weighted
  - cycle_wall_s
  - market_part_mw2h_{unweighted,weighted}
  - max_abs_d_p_mw
  - n_non_optimal_block_terminations
  - non_optimal_blocks
  - operation_part_mw2h_{unweighted,weighted}
  - response_capture_error, response_capture_s, response_captured, response_cycle
  - row18_charge_weighted
  - rss_bytes, ru_maxrss_bytes
  - sum_omega_d2_p_mw2h_{unweighted,weighted}

  None of the 29 shared keys differs, and none of the new keys is non-null.
- **Sidecar hashes.** The six other JSONL sidecars are byte-identical to the W48 arm's (sha256 equal), and
  `per_cycle_record.jsonl` is byte-identical to the W83 arm's (`f85069053693b154…` in both).
- **ESSO pickle sha differs from W48.** This is informational and not gated. It was also unequal in the committed r2
  and W83 runs of this gate; I did not check the earlier gates.

**Known caveat, unchanged.** `row18_gate_addendum.json` records 51/57 presence checks true. Six `w51:` checks read
the counter's wrappers because the committed S51G layer writes that file while the counter is installed. This is
identical to the committed r2 run (`P515S53/srp1_bitwise_gate`, 51/57) and the W83 run, and it is the caveat in
`P5_15_ADDENDUM40_42_REPORT.md` item 4. It is not gating; the gating evaluations are 57/57 (W35) and 21/21 (W51).

## Commands

Pre-run checks:
- `ps aux | grep python`: no project process.
- Neither `.p515_g_gate.lock` nor `.p515_s44_campaign.lock` exists.
- The output root and the arm-named eval directories were absent.
- Every gated file was clean, so the gate's own git check passed.

Run, attached, alone, both streams captured; the log is a sibling of the write-once root because the gate refuses an
existing root:

```
set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate.py \
  > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_launch.log 2>&1   # EXIT=1
```

Zero-solve reads after the run:
- the per-key characterisation of the 44 diffs (all `<MISSING>` → `null`, the 29 shared keys equal);
- `git log -S` on the harness, which places the 22 keys at `70950e8c` and shows it is not an ancestor of the r2 run
  `2167c0d9` or of `d5a00bd9`;
- sha256 of the JSONL sidecars across this arm, the W83 arm and the W48 arm.

## Evidence (committed with this report)

- `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate/`: everything except `arm/results/` (empty). Key hashes:
  - `gate.json` `68d69d29…`
  - `w51_gate_addendum.json` `425d7ca5…`
  - `w183f_wrapper_record.json` `10832bf3…`
  - `w183f_manifest_sha256.json` `e7cd233e…` (35 files)
- `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_launch.log` (sha256 `7317b55a…`).
- `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_arm_logs_sha256.txt`: 57 file hashes of the untracked arm eval
  directories `data/SRP1/Results/P56A/evals/p515s44_s45_w10_snapoff_s55w183fgate_578636daa6d6360d_{run,precheck}`
  (IPOPT logs), following the W83 convention.

## Not confirmed

- **The parallel DSO builder (`create_distribution_network_model`) was not exercised.** This gate covers the
  sequential builder only. Addendum 83's "at both DSO builders" is met for the parallel builder only by W183c's
  zero-solve NL identity (`P5_15_W183C_NRF.md`), not by a solve-bearing gate.
- **NRF on was not exercised;** every network flag was off and read back as off.
- **Whether the 44 diffs are acceptable is a ruling, not a measurement.** The byte identity with the post-W64 W83 arm
  shows that the diffs come from the harness field list. It does not by itself change the gate's frozen verdict.
- **No prior ruling on the identical W83 failure was found.** Searched: `PLANNER_BRIEF_2026-09-13.md`,
  `REVISION_CONTEXT.md`, `TASKS.md`, `P5_15_*.md`, for "W83", "tight_tail_w83", "s53w83tailgate",
  "PER_CYCLE_RESPONSE_FIELDS", "field-list" and "eb8cf81f". The record is in the commit message of `eb8cf81f` only.
