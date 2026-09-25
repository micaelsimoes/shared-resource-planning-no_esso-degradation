# P5.15 Addendum 46 ruling 7 W83 -- SRP1 two-cycle bitwise identity with the convergence-depth tight tail ENABLED (inert while the AA-off predicate has not held), vs committed baseline C* and the committed r2 arm

git HEAD at run f9634f62652d99375140c29096b07ea0a7d1bdd2; GATE_PASS = **True**

Instance: C* key 578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c (pin matches: True).
Reference: data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json (sha256 ced90e47873925b5bb87360984d348558d11ea11fd004e93f3a2668fdc389971); cycle_trajectory[:2].
Configuration: baseline -- no flex-price override, no release_solution_bookkeeping, snapshots on, apply_rho False, cap 2.

| gate item | holds |
|---|---|
| ran_two_cycles | True |
| reproduces_committed_reference_zero_diffs | True |
| trajectory_fields_identical_to_reference | True |
| solve_reconciliation_event_level_identity_holds | True |
| arm_instance_derived_solve_profile_agrees | True |
| planning_derivation_agrees_with_case_file | True |
| process_guard_verified_exactly | True |
| no_solves_outside_the_arm | True |
| no_blocked_solver_calls | True |
| shared_frozen_smopf_untouched | True |
| ess_ageing_declaration_verified_in_child | True |
| w35_code_present_in_the_modules_that_ran | True |

Solves (event-level, gating): base 153 + retries credited 0 = 153; observed 153; events by class {'recovered_tier1': 0, 'recovered_tier2': 0, 'unrecovered': 0, 'not_attempted': 0, 'indeterminate': 0}; recovered-only identity 153 (reported); guard {'permitted_solve': 153, 'permitted_exec': 153, 'blocked_solve': 0, 'blocked_exec': 0}; verify failures [].
Arm's own instance-derived profile: {'solves_per_cycle': 51, 'derivation': '(1 + n_dso) * n_years * n_days + n_esso_nodes, per cycle, from the planning object', 'n_dso': 3, 'n_years': 3, 'n_days': 4, 'n_esso_nodes': 3, 'rounds': 3, 'base_solves': 153, 'retry_solves_credited': 0, 'reconciliation_supported': True, 'expected_solves': 153, 'identity': "observed == base + sum over network-failure events of [recovery_attempted] + [tier2_attempted]; unsupported when an ESSO recovery event or an 'indeterminate' network event exists", 'identity_holds': True, 'recovered_only_identity': {'expected': 153, 'holds': True, 'note': "pre-W35 rule, REPORTED not gated: credits retries only to RECOVERED blocks, so an unrecovered block's attempts go uncounted (78a9b230)"}}
Reference comparison: n_diffs 0, first None; field-table mismatches 0 over 29 fields.
Arm: cycles 2, gross_operational_cost 740922817.2425401 (gross convention), recourse 740922817.2425401, terminal step / threshold 5330.87281262313.
Peak RSS (ru_maxrss, bytes): {'units': 'bytes on macOS/BSD (ru_maxrss)', 'process_ru_maxrss_before_arm': 495599616, 'process_ru_maxrss_after_arm': 2230370304, 'production_state_peak_rss_ru_maxrss': 2230370304, 'production_state_peak_rss_platform_units': 'bytes on macOS/BSD, kilobytes on Linux (Python resource module, RUSAGE_SELF.ru_maxrss)', 'solver_subprocesses_ru_maxrss_before': 14417920, 'solver_subprocesses_ru_maxrss_after': 51576832, 'semantics': 'ru_maxrss is a process high-water mark, so "after arm 2" is the max over both arms; the production_state figure is taken by production at the end of run_operational_planning and is per-arm'}

NOT covered by this gate: the hull polish and the S31C settlement writer are post-certification steps and do not run inside a two-cycle arm; their SRP1 invariance rests on the legacy branch running verbatim below 2 scenarios and on p515_s50_generalization_checks.py (864 committed C* period entries reproduced bit for bit)
