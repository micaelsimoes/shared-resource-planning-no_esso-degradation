# P5.15 Addendum 34 W33 item 3 -- two-cycle bitwise gate: flexibility-price override PRESENT at m = 1.0 vs committed baseline C*

git HEAD at run 2e82a23ff4a6d8400e768c9343ecbac80c6c9e18; GATE_PASS = **True**

Instance: C* key 578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c (pin matches: True).
Reference: data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json (sha256 ced90e47873925b5bb87360984d348558d11ea11fd004e93f3a2668fdc389971); cycle_trajectory[:2].
Override: flex_price_multiplier = 1.0 PRESENT; read-back 36 probe DSO blocks, 55296/55296 flex coefficients bitwise equal.

| gate item | holds |
|---|---|
| ran_two_cycles | True |
| reproduces_committed_reference_zero_diffs | True |
| trajectory_fields_identical_to_reference | True |
| solve_reconciliation_event_level_identity_holds | True |
| planning_derivation_agrees_with_case_file | True |
| process_guard_verified_exactly | True |
| no_solves_outside_the_arm | True |
| no_blocked_solver_calls | True |
| shared_frozen_smopf_untouched | True |
| ess_ageing_declaration_verified_in_child | True |
| override_present_at_m_1_0 | True |
| override_applied_checks_all_true | True |
| override_read_back_36_blocks_bitwise | True |
| harness_commit_in_head | True |

Solves (event-level, gating): base 153 + retries credited 0 = 153; observed 153; events by class {'recovered_tier1': 0, 'recovered_tier2': 0, 'unrecovered': 0, 'not_attempted': 0, 'indeterminate': 0}; W10 recovered-only identity 153 (reported); guard {'permitted_solve': 153, 'permitted_exec': 153, 'blocked_solve': 0, 'blocked_exec': 0}; verify failures [].
Reference comparison: n_diffs 0, first None; field-table mismatches 0 over 29 fields.
Arm: cycles 2, gross_operational_cost 740922817.2425401 (gross convention), recourse 740922817.2425401, terminal step / threshold 5330.87281262313.
Peak RSS (ru_maxrss, bytes): {'units': 'bytes on macOS/BSD (ru_maxrss)', 'process_ru_maxrss_before_arm': 460128256, 'process_ru_maxrss_after_arm': 2268790784, 'production_state_peak_rss_ru_maxrss': 2265481216, 'production_state_peak_rss_platform_units': 'bytes on macOS/BSD, kilobytes on Linux (Python resource module, RUSAGE_SELF.ru_maxrss)', 'solver_subprocesses_ru_maxrss_before': 13598720, 'solver_subprocesses_ru_maxrss_after': 52002816, 'semantics': 'ru_maxrss is a process high-water mark, so "after arm 2" is the max over both arms; the production_state figure is taken by production at the end of run_operational_planning and is per-arm'}
