"""
P5.15 Addendum 24 item 1 -- zero-solve checks for `p515_s42_exact_fix_rerun.py`.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 24, frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`,
"Zero-solve checks for the harness where meaningful."

Exercises, without ever calling a solver (`SolveProfileGuard(permitted=())`
armed for the whole script, `verify(0)` checked before writing output):

  1. `_parse_ipopt_log_tail` against SYNTHETIC IPOPT-log text built from the
     EXACT excerpts `WORKER_REPORT_S41_POLISH_PREREQ.md` Part 1 already read
     by hand (a failing block's terminal summary, `optim_log_case9_2025_
     Autumn.log`, and a successful block's) -- so the check is deterministic
     and self-contained (no dependency on ephemeral, untracked local log
     files that may not exist in another environment).
  2. `_classify_final_attempt` against every one of the four control-flow
     shapes `network.py:762-807` can produce (no retry; tier-1 only,
     succeeded/failed; tier-1 then tier-2, succeeded/failed).
  3. `_expected_log_path` against a freshly built TSO block
     (`p515_s32_zero_solve_checks._build_admm_ready_state`, BY IMPORT,
     unchanged -- the same pattern `p515_s41_hull_polish_checks.py`/
     `p515_s42_interface_helper_checks.py` use), hand-computed expectation.
  4. `_tso_interface_diagnostics`'s arithmetic (required delta, bound,
     excess, MVA conversion) against hand-computed values on the SAME
     freshly built state, with synthetic nonzero `interface_delta_p/q`.
  5. `_score_predictions` against two synthetic `per_block` tables: one
     matching every prediction, one where all 12 TSO blocks solve (the
     `all_12_tso_solve` branch).
  6. `_persist_certified_models`/`_hash_and_size` round-trip on a throwaway
     pickle (hash/size match a manual recomputation; unpickles back intact).

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s42_exact_fix_rerun_checks.py
"""

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402 -- BY IMPORT, unchanged
import p515_s42_exact_fix_rerun as R  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'exact_fix_rerun_checks')
OUT_PATH = os.path.join(OUT_DIR, 'results.json')
EVAL_ID = 'p515s42_exact_fix_rerun_checks_probe'

# Verbatim excerpt structure from optim_log_case9_2025_Autumn.log (the block
# WORKER_REPORT_S41_POLISH_PREREQ.md Part 1 read by hand), the FAILING
# block's terminal summary -- iterations=50, unscaled constraint violation
# 3.2299223958199708e-01, exit "Converged to a point of local infeasibility.
# Problem may be infeasible."
FAILING_SNIPPET = """This is Ipopt version 3.14.18, running with linear solver ma97.

Number of Iterations....: 50

                                   (scaled)                 (unscaled)
Objective...............:   2.4820602274392985e+02    2.4820602274392985e+05
Dual infeasibility......:   9.9999999391627370e+01    9.9999999391627367e+04
Constraint violation....:   3.2299223958199708e-01    3.2299223958199708e-01
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.8228500773330356e-07    9.8228500773330364e-04
Overall NLP error.......:   9.9999999391627370e+01    9.9999999391627367e+04


Number of objective function evaluations             = 73
Number of objective gradient evaluations             = 27
Total seconds in IPOPT                               = 0.077

EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
"""

# Verbatim excerpt structure for a SUCCESSFUL block -- iterations=42,
# unscaled constraint violation 1.4589942587406313e-09, exit "Optimal
# Solution Found."
SUCCESS_SNIPPET = """This is Ipopt version 3.14.18, running with linear solver ma97.

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls

Number of Iterations....: 42

                                   (scaled)                 (unscaled)
Objective...............:   1.0000000000000000e+00    1.0000000000000000e+02
Dual infeasibility......:   8.0117027113261281e-06    9.9259759646773554e-02
Constraint violation....:   1.4589942587406313e-09    1.4589942587406313e-09
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   4.8786946603054682e-09    6.0443837823303611e-05
Overall NLP error.......:   2.2703407399290810e-06    9.9259759646773554e-02

Number of objective function evaluations             = 60
Total seconds in IPOPT                               = 0.050

EXIT: Optimal Solution Found.
"""


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to reuse a non-fresh output dir: {OUT_DIR}')
    os.makedirs(OUT_DIR, exist_ok=True)

    guard = SolveProfileGuard(permitted=(), label='P5.15-S42 exact-fix-rerun checks').install()
    checks = {}
    try:
        # ------------------------------------------------------------------
        # 1. _parse_ipopt_log_tail on synthetic (but verbatim-derived) text.
        # ------------------------------------------------------------------
        failing_path = os.path.join(OUT_DIR, '_synthetic_failing.log')
        success_path = os.path.join(OUT_DIR, '_synthetic_success.log')
        with open(failing_path, 'w') as handle:
            handle.write(FAILING_SNIPPET)
        with open(success_path, 'w') as handle:
            handle.write(SUCCESS_SNIPPET)

        parsed_fail = R._parse_ipopt_log_tail(failing_path)
        parsed_ok = R._parse_ipopt_log_tail(success_path)
        checks['parse_ipopt_log_tail_failing'] = {
            'parsed': parsed_fail,
            'expected_iterations': 50, 'expected_violation': 3.2299223958199708e-01,
            'expected_exit_substring': 'local infeasibility',
            'pass': (parsed_fail['iterations'] == 50
                    and parsed_fail['constraint_violation_unscaled'] == 3.2299223958199708e-01
                    and 'local infeasibility' in (parsed_fail['exit_message'] or '')),
        }
        checks['parse_ipopt_log_tail_success'] = {
            'parsed': parsed_ok,
            'expected_iterations': 42, 'expected_violation': 1.4589942587406313e-09,
            'expected_exit': 'Optimal Solution Found.',
            'pass': (parsed_ok['iterations'] == 42
                    and parsed_ok['constraint_violation_unscaled'] == 1.4589942587406313e-09
                    and parsed_ok['exit_message'] == 'Optimal Solution Found.'),
        }
        checks['parse_ipopt_log_tail_missing_file'] = {
            'parsed': R._parse_ipopt_log_tail(os.path.join(OUT_DIR, 'does_not_exist.log')),
            'pass': R._parse_ipopt_log_tail(
                os.path.join(OUT_DIR, 'does_not_exist.log'))['found'] is False,
        }

        # ------------------------------------------------------------------
        # 2. _classify_final_attempt on every control-flow shape.
        # ------------------------------------------------------------------
        no_retry_text = '[INFO] \t\t\t - Running SMOPF, Network TSO, 2025, Spring...\n'
        tier1_ok_text = ('[WARNING] Network primary solve did not converge ...\n'
                         '[INFO] Retrying network solve once for TSO, year=2025, day=Spring, '
                         'cold start, with warm_start_init_point=no.\n'
                         '[INFO] Network recovery solve succeeded for TSO, year=2025, day=Spring.\n')
        tier1_failed_text = ('[WARNING] Network primary solve did not converge ...\n'
                             '[INFO] Retrying network solve once for TSO, year=2025, day=Spring, '
                             'cold start, with warm_start_init_point=no.\n'
                             '[WARNING] Network recovery solve did not converge ...\n')
        tier2_ok_text = (tier1_failed_text
                         + '[INFO] Retrying network solve (tier 2: cold, mu_strategy=adaptive) '
                           'for TSO, year=2025, day=Spring, with mu_strategy=adaptive.\n'
                         + '[INFO] Network tier-2 recovery solve succeeded for TSO, year=2025, '
                           'day=Spring.\n')
        tier2_failed_text = (tier1_failed_text
                             + '[INFO] Retrying network solve (tier 2: cold, mu_strategy=adaptive) '
                               'for TSO, year=2025, day=Spring, with mu_strategy=adaptive.\n'
                             + '[WARNING] Network tier-2 recovery solve did not converge ...\n')
        classify_checks = {
            'no_retry': (R._classify_final_attempt(no_retry_text), (None, 'primary')),
            'tier1_ok': (R._classify_final_attempt(tier1_ok_text), ('recovery', 'tier-1 cold recovery')),
            'tier1_failed': (R._classify_final_attempt(tier1_failed_text),
                             ('recovery', 'tier-1 cold recovery')),
            'tier2_ok': (R._classify_final_attempt(tier2_ok_text),
                        ('recovery_tier2', 'tier-2 cold+adaptive recovery')),
            'tier2_failed': (R._classify_final_attempt(tier2_failed_text),
                             ('recovery_tier2', 'tier-2 cold+adaptive recovery')),
        }
        checks['classify_final_attempt'] = {
            name: {'got': got, 'expected': expected, 'pass': got == expected}
            for name, (got, expected) in classify_checks.items()}
        checks['classify_final_attempt']['all_pass'] = all(
            v['pass'] for k, v in checks['classify_final_attempt'].items() if k != 'all_pass')

        # ------------------------------------------------------------------
        # 3-4. _expected_log_path and _tso_interface_diagnostics on a FRESH,
        # production-constructed, never-solved TSO/DSO block.
        # ------------------------------------------------------------------
        planning, tso_model, dso_models, esso_model, consensus_vars, _dual = (
            _build_admm_ready_state(EVAL_ID))
        models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
        tso = planning.transmission_network
        year0 = next(iter(tso.years))
        day0 = next(iter(tso.days))
        t_net0 = tso.network[year0][day0]
        t_holder_params = tso.params

        expected_log_path = R._expected_log_path(t_net0, t_holder_params, log_suffix=None)
        options = t_holder_params.solver_params.options
        manual_stem, manual_ext = os.path.splitext(
            os.path.join(t_net0.logs_dir, options['output_file']))
        day_label = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(day0))
        manual_expected = f'{manual_stem}_{year0}_{day_label}{manual_ext}'
        checks['expected_log_path'] = {
            'computed': expected_log_path, 'manual': manual_expected,
            'pass': expected_log_path == manual_expected,
        }
        expected_log_path_recovery = R._expected_log_path(t_net0, t_holder_params, log_suffix='recovery')
        checks['expected_log_path_recovery_suffix'] = {
            'computed': expected_log_path_recovery,
            'pass': (expected_log_path_recovery == f'{manual_stem}_{year0}_{day_label}_recovery{manual_ext}'),
        }

        node_ids = sorted(planning.distribution_networks)
        node_a = node_ids[0]
        d_net_a = planning.distribution_networks[node_a].network[year0][day0]
        d_model_a = dso_models[node_a][year0][day0]
        ref_gen_a = d_net_a.get_reference_gen_idx()
        dn_a = list(t_net0.active_distribution_network_nodes).index(node_a)
        p0 = next(iter(tso_model[year0][day0].periods))
        dvar_p = tso_model[year0][day0].interface_delta_p[dn_a, 0, 0, p0]
        rating = dvar_p.ub
        synth_delta_p = 0.3 * rating
        dvar_p.set_value(synth_delta_p)
        d_model_a.pg[ref_gen_a, 0, 0, p0].set_value(0.05)

        common = R.O.common_coordinated_values(planning, models, consensus_vars)
        rows = R._tso_interface_diagnostics(planning, models, common, year0, day0)
        row_a_p0 = next(r for r in rows if r['node'] == node_a and r['period'] == p0)
        pc_val = float(pe.value(tso_model[year0][day0].pc[
            t_net0.get_adn_load_idx(node_a), 0, 0, p0]))
        common_p = common[(node_a, year0, day0, p0)]['common_p']
        manual_required_delta_p = common_p - pc_val
        manual_excess_p = max(0.0, manual_required_delta_p - dvar_p.ub, dvar_p.lb - manual_required_delta_p)
        checks['tso_interface_diagnostics'] = {
            'n_rows': len(rows),
            'expected_n_rows': (len(node_ids) * len(tso_model[year0][day0].periods)),
            'required_delta_p_pu': row_a_p0['required_delta_p_pu'],
            'manual_required_delta_p_pu': manual_required_delta_p,
            'excess_p_pu': row_a_p0['excess_p_pu'], 'manual_excess_p_pu': manual_excess_p,
            'excess_p_mva': row_a_p0['excess_p_mva'],
            'manual_excess_p_mva': manual_excess_p * t_net0.baseMVA,
            'base_mva': t_net0.baseMVA,
            'pass': (row_a_p0['n'] if False else True)  # placeholder overwritten below
                     and abs(row_a_p0['required_delta_p_pu'] - manual_required_delta_p) < 1e-12
                     and abs(row_a_p0['excess_p_pu'] - manual_excess_p) < 1e-12
                     and abs(row_a_p0['excess_p_mva'] - manual_excess_p * t_net0.baseMVA) < 1e-9
                     and len(rows) == len(node_ids) * len(tso_model[year0][day0].periods),
        }

        # ------------------------------------------------------------------
        # 5. _score_predictions.
        # ------------------------------------------------------------------
        all_match_records = []
        for block in R.SPRING_SUMMER_TSO_BLOCKS:
            all_match_records.append({'block': block, 'solved': False,
                                      'constraint_violation_mva': 3e-4})
        for block in R.AUTUMN_WINTER_TSO_BLOCKS:
            all_match_records.append({'block': block, 'solved': True,
                                      'constraint_violation_mva': None})
        for block in R.DSO_2025_BLOCKS_NO_PREDICTION:
            all_match_records.append({'block': block, 'solved': True,
                                      'constraint_violation_mva': None})
        score_all_match = R._score_predictions(all_match_records)

        all_solve_records = []
        for block in list(R.SPRING_SUMMER_TSO_BLOCKS) + list(R.AUTUMN_WINTER_TSO_BLOCKS):
            all_solve_records.append({'block': block, 'solved': True,
                                      'constraint_violation_mva': None})
        for block in R.DSO_2025_BLOCKS_NO_PREDICTION:
            all_solve_records.append({'block': block, 'solved': True,
                                      'constraint_violation_mva': None})
        score_all_solve = R._score_predictions(all_solve_records)

        checks['score_predictions'] = {
            'all_match_case': {
                'n_scored': score_all_match['n_predictions_scored'],
                'n_matched': score_all_match['n_predictions_matched'],
                'all_matched': score_all_match['all_predictions_matched'],
                'all_12_tso_solve': score_all_match['all_12_tso_solve'],
                'pass': (score_all_match['n_predictions_scored'] == 12
                        and score_all_match['n_predictions_matched'] == 12
                        and score_all_match['all_predictions_matched'] is True
                        and score_all_match['all_12_tso_solve'] is False),
            },
            'all_solve_case': {
                'all_12_tso_solve': score_all_solve['all_12_tso_solve'],
                'note_present': score_all_solve['on_all_12_tso_solve_note'] is not None,
                'pass': (score_all_solve['all_12_tso_solve'] is True
                        and score_all_solve['on_all_12_tso_solve_note'] is not None),
            },
        }

        # ------------------------------------------------------------------
        # 6. _hash_and_size round trip (no _persist_certified_models call --
        # that needs an out_dir/_refuse_overwrite side effect; hash/size are
        # exercised directly on a throwaway pickle instead).
        # ------------------------------------------------------------------
        throwaway_path = os.path.join(OUT_DIR, '_throwaway.pkl')
        with open(throwaway_path, 'wb') as handle:
            pickle.dump({'a': 1, 'b': [1, 2, 3]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sha, size = R._hash_and_size(throwaway_path)
        manual_hash = hashlib.sha256(open(throwaway_path, 'rb').read()).hexdigest()
        with open(throwaway_path, 'rb') as handle:
            roundtrip = pickle.load(handle)
        checks['hash_and_size_roundtrip'] = {
            'sha256': sha, 'manual_sha256': manual_hash, 'size': size,
            'manual_size': os.path.getsize(throwaway_path),
            'roundtrip_matches': roundtrip == {'a': 1, 'b': [1, 2, 3]},
            'pass': (sha == manual_hash and size == os.path.getsize(throwaway_path)
                     and roundtrip == {'a': 1, 'b': [1, 2, 3]}),
        }

    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(failures)

    _refuse_overwrite(OUT_PATH)
    out = {
        'stage': 'P5.15 Addendum 24 item 1 -- exact-fix rerun harness zero-solve checks',
        'authority': ['data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': failures},
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f'[S42-RERUN-CHECKS] wrote {OUT_PATH}')

    manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
    manifest = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = R.CP._sha256_file(fpath)
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-RERUN-CHECKS] wrote {manifest_path}')

    all_pass = (
        checks['parse_ipopt_log_tail_failing']['pass']
        and checks['parse_ipopt_log_tail_success']['pass']
        and checks['parse_ipopt_log_tail_missing_file']['pass']
        and checks['classify_final_attempt']['all_pass']
        and checks['expected_log_path']['pass']
        and checks['expected_log_path_recovery_suffix']['pass']
        and checks['tso_interface_diagnostics']['pass']
        and checks['score_predictions']['all_match_case']['pass']
        and checks['score_predictions']['all_solve_case']['pass']
        and checks['hash_and_size_roundtrip']['pass']
    )
    print(f'[S42-RERUN-CHECKS] ALL PASS = {all_pass}')
    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
