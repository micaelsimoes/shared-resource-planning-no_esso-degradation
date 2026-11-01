# P5.15 Addendum 29 W32 M2 -- two-cycle bitwise gate: release_solution_bookkeeping ON vs committed baseline C*

git HEAD at run f5c2d87c884b0fa2fced65e42e7d0707dd051d94; GATE_PASS = **True**

Instance: C* key 578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c (pin matches: True).
Reference: data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json (sha256 ced90e47873925b5bb87360984d348558d11ea11fd004e93f3a2668fdc389971); cycle_trajectory[:2].

| gate item | holds |
|---|---|
| ran_two_cycles | True |
| reproduces_committed_reference_zero_diffs | True |
| trajectory_fields_identical_to_reference | True |
| solve_reconciliation_identity_holds | True |
| planning_derivation_agrees_with_case_file | True |
| process_guard_verified_exactly | True |
| no_solves_outside_the_arm | True |
| no_blocked_solver_calls | True |
| shared_frozen_smopf_untouched | True |
| switch_read_back_on_for_tso_and_every_dso | True |
| release_called_once_per_successful_network_solve | True |
| network_run_smopf_calls_equal_declared_network_solves | True |
| ess_ageing_declaration_verified_in_child | True |

Solves: base 153 + tier1 0 + 2 x tier2 0 = 153; observed 153; guard {'permitted_solve': 153, 'permitted_exec': 153, 'blocked_solve': 0, 'blocked_exec': 0}; verify failures [].
Switch: run_smopf calls 144 (declared network solves 144), succeeded 144, release calls 144.
Reference comparison: n_diffs 0, first None; field-table mismatches 0 over 29 fields.
Arm: cycles 2, gross_operational_cost 740922817.2425401 (gross convention), recourse 740922817.2425401, terminal step / threshold 5330.87281262313.
Peak RSS (ru_maxrss, bytes): {'units': 'bytes on macOS/BSD (ru_maxrss)', 'process_ru_maxrss_before_arm': 474038272, 'process_ru_maxrss_after_arm': 1654751232, 'production_state_peak_rss_ru_maxrss': 1654751232, 'production_state_peak_rss_platform_units': 'bytes on macOS/BSD, kilobytes on Linux (Python resource module, RUSAGE_SELF.ru_maxrss)', 'solver_subprocesses_ru_maxrss_before': 13516800, 'solver_subprocesses_ru_maxrss_after': 51871744, 'semantics': 'ru_maxrss is a process high-water mark, so "after arm 2" is the max over both arms; the production_state figure is taken by production at the end of run_operational_planning and is per-arm'}
