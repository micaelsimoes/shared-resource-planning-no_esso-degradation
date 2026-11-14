"""
P5.15 Addendum 29 (task W32, STEP M2) -- checks for the solution-bookkeeping release switch
(`SolverParameters.release_solution_bookkeeping`, default False; network.py `_release_solution_bookkeeping`,
called by `_run_smopf` after a successful `model.solutions.load_from(result)` only when the switch is on).

Armed `SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED)` for the whole script, installed before any
model import. PART Z makes ZERO solves (the guard's count is asserted 0 at its end); PART S makes exactly
4 declared solves (+ recovery attempts, counted by a pass-through wrapper on
`network._run_smopf_solver_attempt`), verified EXACTLY at the end.

PART Z (zero solves)
  Z1 a fresh `SolverParameters()` has release_solution_bookkeeping False.
  Z2 `read_solver_parameters`: key absent -> False; true -> True; false -> False (bool()).
  Z3 the SRP1 case as loaded by the oracle: the TSO's and every DSO's solver_params carry False
     (the case file does not declare the key -> today's behaviour).
  Z4 `p515_s44_scale_measurement.set_release_solution_bookkeeping(planning, True)` sets the TSO's and
     every DSO's flag, reads it back, leaves the shared-ESS solver_params untouched; (planning, False)
     restores; a planning whose flag cannot be set raises.
  Z5 source: `_release_solution_bookkeeping` clears exactly `model.solutions` and `result.solution`;
     `_run_smopf` calls it only inside the `solver_result_succeeded(result)` branch, after load_from and
     only when `result is not None` and the switch is on; the ESSO solve path is unchanged.
  Z6 preserved fixtures still unpickle (the three FrozenSMOPF pickles and the P512R cycle-21 anchor)
     -- nothing was removed, this is the standing rule's check.
PART S (4 solves) -- block-level A/B, switch OFF vs ON, same block, same code path:
  Two arm planning objects from `_construct_arm_planning` (C*), each restricted to node 7 / 2025 /
  Spring; the block built and cold-solved by production's `create_distribution_networks_models_sequential`,
  then re-solved warm (`NetworkData.optimize(..., from_warm_start=True)`). Arm OFF with the default,
  arm ON with the switch. Compared BITWISE (type and value): every Var value; every entry of the
  dual / ipopt_zL_out / ipopt_zU_out / ipopt_zL_in / ipopt_zU_in suffixes (by component name); the
  consensus values the initialization extracted for node 7; result.solver status / termination /
  message; the runtime parse production's `_get_info_from_results(result, 'Time:')` performs (parses to a
  number on both). Post-state: ON has len(model.solutions.solutions) == 0 and len(result.solution) == 0;
  OFF has 1 and 1.

Output (write-once): data/SRP1/Results/P515S49/memory_fix_checks/{memory_fix_checks.json, manifest_sha256.json}
Launch (attached, alone, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_memory_fix_checks.py \\
        > data/SRP1/Results/P515S49/memory_fix_checks_launch.log 2>&1
"""

import inspect
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402 -- PERMITTED only

GUARD = SolveProfileGuard(N.PERMITTED, label='P5.15 W32 memory fix checks (Z: 0 solves; S: 4 declared)').install()

import network as NET  # noqa: E402
import helper_functions as HF  # noqa: E402
import solver_parameters as SP  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s44_scale_measurement as S44  # noqa: E402

STAGE = 'P5.15 Addendum 29 W32 M2 -- release_solution_bookkeeping switch checks (zero-solve + 4-solve block A/B)'
SCHEMA = 'p515_s49_memory_fix_checks_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'memory_fix_checks')
DECLARED_SOLVES_S = 4
NODE, YEAR, DAY = 7, '2025', 'Spring'
FIXTURES = ('data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl',
            'data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
            'data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
            'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl')
SUFFIXES = ('dual', 'ipopt_zL_out', 'ipopt_zU_out', 'ipopt_zL_in', 'ipopt_zU_in')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W32-checks] {msg}', flush=True)


def part_z():
    res = {}
    sp = SP.SolverParameters()
    res['Z1_default_false'] = sp.release_solution_bookkeeping is False
    base = {'name': 'ipopt', 'verbose': False, 'options': {}}
    reads = {}
    for label, extra in (('absent', {}), ('true', {'release_solution_bookkeeping': True}),
                         ('false', {'release_solution_bookkeeping': False})):
        p = SP.SolverParameters()
        p.read_solver_parameters(dict(base, **extra))
        reads[label] = p.release_solution_bookkeeping
    res['Z2_reads'] = reads
    res['Z2_ok'] = reads == {'absent': False, 'true': True, 'false': False}
    planning = G.O.fresh_planning('p515s49_memfix_checks_r2_z')
    flags = {'tso': planning.transmission_network.params.solver_params.release_solution_bookkeeping,
             **{f'dso_{n}': d.params.solver_params.release_solution_bookkeeping
                for n, d in planning.distribution_networks.items()}}
    res['Z3_case_flags'] = flags
    res['Z3_ok'] = all(v is False for v in flags.values())
    esso_sp = planning.shared_ess_data.params.solver_params
    esso_before = getattr(esso_sp, 'release_solution_bookkeeping', None)
    on = S44.set_release_solution_bookkeeping(planning, True)
    esso_after_on = getattr(esso_sp, 'release_solution_bookkeeping', None)
    off = S44.set_release_solution_bookkeeping(planning, False)
    res['Z4_on'] = on
    res['Z4_off'] = off
    res['Z4_esso_untouched'] = esso_before == esso_after_on
    try:
        class _Broken:
            transmission_network = None
            distribution_networks = {}
        S44.set_release_solution_bookkeeping(_Broken(), True)
        res['Z4_broken_raises'] = False
    except Exception as error:  # noqa: BLE001
        res['Z4_broken_raises'] = f'{type(error).__name__}: {error}'
    res['Z4_ok'] = (on['took_effect'] and off['took_effect'] and res['Z4_esso_untouched']
                    and res['Z4_broken_raises'] is not False
                    and all(v is True for v in on['read_back'].values())
                    and all(v is False for v in off['read_back'].values()))
    rel_src = inspect.getsource(NET._release_solution_bookkeeping)
    run_src = inspect.getsource(NET._run_smopf)
    body = [ln.strip() for ln in rel_src.splitlines()
            if ln.strip() and not ln.strip().startswith(('#', 'def ', '"""', "'")) and '"""' not in ln]
    res['Z5_release_body_statements'] = body
    idx_succ = run_src.find('if solver_result_succeeded(result):')
    idx_load = run_src.find('model.solutions.load_from(result)')
    idx_call = run_src.find('_release_solution_bookkeeping(model, result)')
    idx_else = run_src.find('else:\n        _print_network_failure_context')
    res['Z5_call_positions'] = {'succeeded_branch': idx_succ, 'load_from': idx_load, 'release_call': idx_call,
                                'failure_else': idx_else}
    esso_src = inspect.getsource(sys.modules['shared_energy_storage_data'])
    res['Z5_ok'] = (body == ['model.solutions.clear()', 'result.solution.clear()']
                    and 0 < idx_succ < idx_load < idx_call < idx_else
                    and "getattr(params.solver_params, 'release_solution_bookkeeping', False)" in run_src
                    and '_release_solution_bookkeeping' not in esso_src)
    fixtures = {}
    for rel in FIXTURES:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            fixtures[rel] = 'absent'
            continue
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        fixtures[rel] = f'loaded {type(obj).__name__}'
        del obj
    res['Z6_fixtures'] = fixtures
    res['Z6_ok'] = all(v.startswith('loaded') for v in fixtures.values())
    res['Z_guard_counts_at_end'] = dict(GUARD.counts)
    res['Z_zero_solves'] = GUARD.counts['permitted_solve'] == 0 and GUARD.counts['permitted_exec'] == 0
    return res


def _snapshot_block(model, result):
    import pyomo.environ as pe
    out = {'vars': {}, 'suffixes': {}}
    for v in model.component_data_objects(pe.Var):
        out['vars'][v.name] = v.value
    for name in SUFFIXES:
        comp = getattr(model, name)
        out['suffixes'][name] = {c.name: val for c, val in comp.items()}
    out['solver'] = {'status': str(result.solver.status), 'termination': str(result.solver.termination_condition),
                     'message': str(getattr(result.solver, 'message', None))}
    runtime = NET._get_info_from_results(result, 'Time:').strip()
    out['runtime_parse'] = {'string': runtime, 'is_number': HF.is_number(runtime)}
    out['post_state'] = {'model_solutions': len(model.solutions.solutions),
                         'model_symbol_maps': len(model.solutions.symbol_map),
                         'result_solutions': len(result.solution)}
    return out


def _bitwise_diff(a, b, path=''):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append({'path': f'{path}.{k}', 'missing_in': 'off' if k not in a else 'on'})
            else:
                diffs += _bitwise_diff(a[k], b[k], f'{path}.{k}')
    elif not (type(a) is type(b) and a == b):
        diffs.append({'path': path, 'off': a, 'on': b})
    return diffs


def part_s():
    attempts = []
    orig_attempt = NET._run_smopf_solver_attempt

    def counting_attempt(net, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):
        attempts.append({'network': net.name, 'year': net.year, 'day': net.day, 'warm': from_warm_start,
                         'log_suffix': log_suffix})
        return orig_attempt(net, model, params, from_warm_start=from_warm_start,
                            option_overrides=option_overrides, log_suffix=log_suffix)

    NET._run_smopf_solver_attempt = counting_attempt
    arms = {}
    try:
        for arm, switch in (('off', False), ('on', True)):
            report = {}
            planning, sed, candidate = G._construct_arm_planning(
                f's49_memfix_checks_{arm}', os.path.join(REPO, OUT_REL, f'arm_{arm}'), report,
                investment_map=S44.C_STAR, eval_id=f'p515s49_memfix_checks_r2_{arm}',
                num_max_iters_override=S44.BUILD_CAP, apply_rho=False)
            applied = S44.set_release_solution_bookkeeping(planning, switch)
            dn = planning.distribution_networks[NODE]
            year = next(y for y in dn.years if str(y) == YEAR)   # keys are the loader's type
            day = next(d for d in dn.days if str(d) == DAY)
            dn.years = {year: dn.years[year]}
            dn.days = {day: dn.days[day]}
            cv, _dv = srp.create_admm_variables(planning)
            models, res_init = srp.create_distribution_networks_models_sequential({NODE: dn}, cv,
                                                                                  candidate['total_capacity'])
            block = models[NODE][year][day]
            snap_cold = _snapshot_block(block, res_init[NODE][year][day])
            cons = {k: (((cv.get(k) or {}).get('dso') or {}).get('current') or {}).get(NODE, {}).get(year, {}).get(day)
                    for k in ('vmag', 'pf', 'ess')}
            if not any(cons.values()):
                raise RuntimeError('no node-7 consensus values found to compare')
            res_warm = dn.optimize(models[NODE], from_warm_start=True)
            snap_warm = _snapshot_block(block, res_warm[year][day])
            arms[arm] = {'switch_applied': applied, 'cold': snap_cold, 'warm': snap_warm,
                         'consensus_node7': json.loads(json.dumps(cons, default=str)),
                         'succeeded': [HF.solver_result_succeeded(res_init[NODE][year][day]),
                                       HF.solver_result_succeeded(res_warm[year][day])]}
            del models, res_init, res_warm, block, planning
    finally:
        NET._run_smopf_solver_attempt = orig_attempt
    out = {'attempts': attempts}
    for phase in ('cold', 'warm'):
        a, b = arms['off'][phase], arms['on'][phase]
        out[f'{phase}_n_vars'] = len(a['vars'])
        out[f'{phase}_suffix_lengths'] = {k: [len(a['suffixes'][k]), len(b['suffixes'][k])] for k in SUFFIXES}
        out[f'{phase}_var_diffs'] = _bitwise_diff(a['vars'], b['vars'])[:50]
        out[f'{phase}_suffix_diffs'] = _bitwise_diff(a['suffixes'], b['suffixes'])[:50]
        out[f'{phase}_solver_diffs'] = _bitwise_diff(a['solver'], b['solver'])
        out[f'{phase}_runtime_parse'] = {'off': a['runtime_parse'], 'on': b['runtime_parse']}
        out[f'{phase}_post_state'] = {'off': a['post_state'], 'on': b['post_state']}
    out['consensus_diffs'] = _bitwise_diff(arms['off']['consensus_node7'], arms['on']['consensus_node7'])
    out['succeeded'] = {k: v['succeeded'] for k, v in arms.items()}
    out['switch_applied'] = {k: v['switch_applied'] for k, v in arms.items()}
    ok = all(not out[f'{p}_{k}'] for p in ('cold', 'warm') for k in ('var_diffs', 'suffix_diffs', 'solver_diffs'))
    ok = ok and not out['consensus_diffs'] and all(all(v) for v in out['succeeded'].values())
    ok = ok and all(out[f'{p}_post_state']['off'] == {'model_solutions': 1, 'model_symbol_maps': 0, 'result_solutions': 1}
                    and out[f'{p}_post_state']['on'] == {'model_solutions': 0, 'model_symbol_maps': 0, 'result_solutions': 0}
                    for p in ('cold', 'warm'))
    ok = ok and all(out[f'{p}_runtime_parse'][s]['is_number'] is not False
                    and out[f'{p}_runtime_parse'][s]['string'] != '' for p in ('cold', 'warm') for s in ('off', 'on'))
    out['S_ok'] = ok
    return out


def main():
    started = time.time()
    out_dir = os.path.join(REPO, OUT_REL)
    if os.path.exists(out_dir):
        raise SystemExit(f'refusing: output exists (write-once): {out_dir}')
    os.makedirs(out_dir)
    os.environ.update(H.THREAD_CAP_ENV)
    _log(STAGE)
    z = part_z()
    _log(f"PART Z: { {k: v for k, v in z.items() if k.endswith('_ok') or k == 'Z1_default_false' or k == 'Z_zero_solves'} }")
    s = part_s()
    recovery = sum(1 for a in s['attempts'] if a['log_suffix'])
    reconciled = DECLARED_SOLVES_S + recovery
    failures = GUARD.verify(reconciled)
    items = {'Z1': z['Z1_default_false'], 'Z2': z['Z2_ok'], 'Z3': z['Z3_ok'], 'Z4': z['Z4_ok'], 'Z5': z['Z5_ok'],
             'Z6': z['Z6_ok'], 'Z_zero_solves': z['Z_zero_solves'], 'S_block_ab_bitwise': s['S_ok'],
             'guard_verified_exactly': not failures}
    payload = {'schema': SCHEMA, 'stage': STAGE, 'utc': datetime.now(timezone.utc).isoformat(),
               'git_head': S44._git(['rev-parse', 'HEAD']),
               'script_sha256': S44.sha256_file(os.path.abspath(__file__)),
               'network_py_sha256': S44.sha256_file(os.path.join(REPO, 'network.py')),
               'solver_parameters_py_sha256': S44.sha256_file(os.path.join(REPO, 'solver_parameters.py')),
               'instance': {'candidate': 'C* (0.96875/3.875 at 5, 7, 9; 2025)', 'block': [NODE, YEAR, DAY]},
               'part_z': z, 'part_s': s,
               'guard': {'declared_S': DECLARED_SOLVES_S, 'recovery_attempts': recovery, 'reconciled': reconciled,
                         'counts': dict(GUARD.counts), 'verify_failures': failures},
               'items': items, 'all_pass': all(items.values()), 'wall_s': time.time() - started}
    with open(os.path.join(out_dir, 'memory_fix_checks.json'), 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = S44.sha256_file(fpath)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    for k, v in items.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    _log(f"solves {GUARD.counts['permitted_solve']}/{reconciled}; ALL_PASS={payload['all_pass']}")
    sys.exit(0 if payload['all_pass'] else 1)


if __name__ == '__main__':
    main()
