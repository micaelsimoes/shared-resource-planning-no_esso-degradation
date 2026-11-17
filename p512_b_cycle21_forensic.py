"""
P5.12-B -- forensic audit of the FIRST local NLP failure in the cold RESCALED
trajectory, at cycle 21.

INSTRUMENTATION STRATEGY, AND WHY IT NEEDS NO PRODUCTION EDIT.
`NetworkData.optimize` already accepts `failure_snapshot_callback` and
`pre_solve_snapshot_callback`, and already clones the pre-solve model when
either is supplied (`network_data.py:54-66`). Both coordination entry points --
`update_distribution_coordination_models_and_solve` and
`update_transmission_coordination_model_and_solve` -- already pass their own
callbacks and carry `cycle=iter`. So the whole audit is reachable by wrapping:

  * the two coordination functions, to learn the cycle and to inspect results
    immediately after the solves and BEFORE any consensus/dual/rho/recourse
    update;
  * `NetworkData.optimize`, to attach our callbacks at cycles 20 and 21.

Production's own callbacks are CHAINED, never replaced, so production's failure
snapshots still happen exactly as they would unobserved.

Stopping at the first cycle-21 failure is done by raising `Cycle21Failure` from
the coordination wrapper. That propagates out of `run_operational_planning`
before `update_and_check_convergence` runs, so no consensus, dual, rho or
recourse update is applied and cycle 22 never begins.

KNOWN LIMITATION, RECORDED RATHER THAN WORKED AROUND. The DSO blocks are solved
before the TSO block within a cycle, and production applies the DSO consensus
update between the two. If the first failure is a TSO block, that DSO-side
update will already have been applied when we abort. It is recorded in the
report; suppressing it would require editing production.

Execution is sequential (`parallel_execution = False` for SRP1), so no task
cancellation semantics arise.

    python p512_b_cycle21_forensic.py
"""

import hashlib
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pyomo.environ as pe  # noqa: E402
import network_data  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import p510_oracle as OR  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from helper_functions import solver_result_succeeded, solver_result_summary  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P512B')
TARGET_CYCLE = 21
AUDIT_CYCLES = (20, 21)
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}

VAR_FAMILIES = ('e', 'f', 'vmag_sqr', 'pg', 'qg', 'pc', 'qc',
                'shared_es_pch', 'shared_es_pdch', 'shared_es_soc',
                'flex_p_up', 'flex_p_down', 'r_sqr')
CONSENSUS_PARAMS = ('vmag_req', 'p_pf_req', 'q_pf_req', 'p_ess_req', 'q_ess_req',
                    'dual_vmag_req', 'dual_pf_p_req', 'dual_pf_q_req',
                    'dual_ess_p_req', 'dual_ess_q_req')


class Cycle21Failure(Exception):
    """Raised to abort before any cycle-level update is applied."""


STATE = {'cycle': None, 'captures': {20: {}, 21: {}}, 'failure': None,
         'order': []}


# ===========================================================================
#  block state capture
# ===========================================================================
def _vector_summary(model, name):
    comp = getattr(model, name, None)
    if comp is None:
        return None
    values, at_bound, nonfinite = [], 0, 0
    for idx in comp:
        data = comp[idx]
        raw = getattr(data, 'value', None)
        if raw is None:
            continue
        v = float(raw)
        if not math.isfinite(v):
            nonfinite += 1
            continue
        values.append(v)
        lb, ub = getattr(data, 'lb', None), getattr(data, 'ub', None)
        if lb is not None and abs(v - float(lb)) <= 1e-8:
            at_bound += 1
        elif ub is not None and abs(v - float(ub)) <= 1e-8:
            at_bound += 1
    if not values:
        return {'n': 0, 'nonfinite': nonfinite}
    digest = hashlib.sha256(
        ''.join(f'{v:.17g};' for v in values).encode()).hexdigest()[:16]
    return {'n': len(values), 'min': min(values), 'max': max(values),
            'absmax': max(abs(v) for v in values),
            'l1': sum(abs(v) for v in values),
            'at_bound': at_bound, 'nonfinite': nonfinite, 'sha16': digest,
            'values_head': values[:4]}


def _param_summary(model, name):
    comp = getattr(model, name, None)
    if comp is None:
        return None
    values = []
    for idx in comp:
        try:
            values.append(float(pe.value(comp[idx])))
        except Exception:
            continue
    if not values:
        return None
    return {'n': len(values), 'absmax': max(abs(v) for v in values),
            'l1': sum(abs(v) for v in values),
            'sha16': hashlib.sha256(
                ''.join(f'{v:.17g};' for v in values).encode()).hexdigest()[:16]}


def _suffix_summary(model, name):
    suf = getattr(model, name, None)
    if suf is None:
        return None
    try:
        items = [float(v) for v in suf.values() if v is not None]
    except Exception:
        return {'present': True, 'unreadable': True}
    if not items:
        return {'present': True, 'n': 0}
    finite = [v for v in items if math.isfinite(v)]
    return {'present': True, 'n': len(items),
            'nonfinite': len(items) - len(finite),
            'absmax': max(abs(v) for v in finite) if finite else None,
            'l1': sum(abs(v) for v in finite) if finite else None}


def capture_block(network_name, agent, year, day, pre_model, result, cycle):
    """Everything Gate B2 asks for, from production's own pre-solve clone."""
    started = time.time()
    out = {'agent': agent, 'network': network_name, 'year': year, 'day': str(day),
           'cycle': cycle}
    try:
        out['n_variables'] = int(sum(1 for _ in pre_model.component_data_objects(
            pe.Var, active=True)))
        out['n_constraints'] = int(sum(1 for _ in pre_model.component_data_objects(
            pe.Constraint, active=True)))
    except Exception as error:
        out['structure_error'] = f'{type(error).__name__}: {error}'

    active_obj = None
    for obj in pre_model.component_data_objects(pe.Objective, active=True):
        active_obj = obj
        break
    out['active_objective'] = getattr(active_obj, 'name', None)
    try:
        out['objective_value_at_start'] = float(pe.value(active_obj))
    except Exception:
        out['objective_value_at_start'] = None
    try:
        out['admm_objective_scale'] = float(pe.value(pre_model.admm_objective_scale))
    except Exception:
        out['admm_objective_scale'] = None
    for name in ('rho_v', 'rho_pf', 'rho_ess'):
        try:
            out[name] = float(pe.value(getattr(pre_model, name)))
        except Exception:
            out[name] = None

    out['primal'] = {f: _vector_summary(pre_model, f) for f in VAR_FAMILIES}
    out['primal'] = {k: v for k, v in out['primal'].items() if v}
    out['consensus'] = {p: _param_summary(pre_model, p) for p in CONSENSUS_PARAMS}
    out['consensus'] = {k: v for k, v in out['consensus'].items() if v}
    out['multipliers'] = {n: _suffix_summary(pre_model, n)
                          for n in ('ipopt_zL_in', 'ipopt_zU_in', 'dual')}

    # starting feasibility, using production's own constraint scan
    try:
        families = O.scan_constraints(pre_model, descend=False)
        worst = max(((f['max_violation'], name) for name, f in families.items()),
                    default=(0.0, None))
        out['start_max_violation'] = worst[0]
        out['start_worst_family'] = worst[1]
        out['start_violating_families'] = sorted(
            ((name, f['max_violation']) for name, f in families.items()
             if f['max_violation'] > 1e-8), key=lambda kv: -kv[1])[:8]
    except Exception as error:
        out['constraint_scan_error'] = f'{type(error).__name__}: {error}'

    # solver outcome
    out['solver_succeeded'] = bool(solver_result_succeeded(result))
    try:
        out['solver_summary'] = str(solver_result_summary(result))
    except Exception:
        out['solver_summary'] = None
    for attr, key in (('solver', 'solver'), ('problem', 'problem')):
        try:
            out[f'result_{key}'] = json.loads(json.dumps(
                getattr(result, attr)[0].__dict__, default=str))
        except Exception:
            pass
    out['capture_seconds'] = time.time() - started
    return out


def read_block_log(network, params):
    try:
        path = R.block_log_path(network, params)
        if not path or not os.path.exists(path):
            return {'log_path': path, 'available': False}
        info = R.read_log_since(path, 0)
        info['log_path'] = path
        info['available'] = True
        with open(path, errors='replace') as handle:
            tail = handle.read()[-6000:]
        info['tail'] = tail
        return info
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}


# ===========================================================================
#  instrumentation
# ===========================================================================
def install(planning):
    """Wrap the coordination entry points and NetworkData.optimize."""
    orig_dso = srp.update_distribution_coordination_models_and_solve
    orig_tso = srp.update_transmission_coordination_model_and_solve
    orig_optimize = network_data.NetworkData.optimize

    def _agent_of(holder):
        name = getattr(holder, 'name', '') or ''
        return 'TSO' if name == getattr(
            planning.transmission_network, 'name', None) else f'DSO:{name}'

    def patched_optimize(self, model, from_warm_start=False, print_header=True,
                         failure_snapshot_callback=None,
                         pre_solve_snapshot_callback=None):
        cycle = STATE['cycle']
        if cycle in AUDIT_CYCLES:
            prod_fail, prod_pre = failure_snapshot_callback, pre_solve_snapshot_callback

            def pre_cb(pre_model, year, day, result):
                if prod_pre is not None:
                    prod_pre(pre_model, year, day, result)
                key = f'{_agent_of(self)}|{year}|{day}'
                rec = capture_block(getattr(self, 'name', '?'), _agent_of(self),
                                    year, day, pre_model, result, cycle)
                try:
                    rec['ipopt_log'] = read_block_log(
                        self.network[year][day], self.params)
                except Exception as error:
                    rec['ipopt_log'] = {'error': str(error)}
                STATE['captures'][cycle][key] = rec
                STATE['order'].append((cycle, key, rec['solver_succeeded']))
                if not rec['solver_succeeded'] and cycle == TARGET_CYCLE \
                        and STATE['failure'] is None:
                    STATE['failure'] = {'key': key, 'record': rec}
                    print(f'[P5.12-B] FIRST CYCLE-21 FAILURE: {key} '
                          f'-- {rec.get("solver_summary")}', flush=True)

            def fail_cb(pre_model, year, day, result):
                if prod_fail is not None:
                    prod_fail(pre_model, year, day, result)

            failure_snapshot_callback, pre_solve_snapshot_callback = fail_cb, pre_cb
        return orig_optimize(self, model, from_warm_start=from_warm_start,
                             print_header=print_header,
                             failure_snapshot_callback=failure_snapshot_callback,
                             pre_solve_snapshot_callback=pre_solve_snapshot_callback)

    def _abort_if_failed(stage):
        if STATE['cycle'] == TARGET_CYCLE and STATE['failure'] is not None:
            raise Cycle21Failure(
                f'first local NLP failure at cycle {TARGET_CYCLE} during the '
                f'{stage} solves: {STATE["failure"]["key"]}')

    def patched_dso(*args, **kwargs):
        STATE['cycle'] = kwargs.get('cycle')
        result = orig_dso(*args, **kwargs)
        _abort_if_failed('DSO')
        return result

    def patched_tso(*args, **kwargs):
        STATE['cycle'] = kwargs.get('cycle')
        result = orig_tso(*args, **kwargs)
        _abort_if_failed('TSO')
        return result

    srp.update_distribution_coordination_models_and_solve = patched_dso
    srp.update_transmission_coordination_model_and_solve = patched_tso
    network_data.NetworkData.optimize = patched_optimize
    return (orig_dso, orig_tso, orig_optimize)


def restore(originals):
    srp.update_distribution_coordination_models_and_solve = originals[0]
    srp.update_transmission_coordination_model_and_solve = originals[1]
    network_data.NetworkData.optimize = originals[2]


# ===========================================================================
#  matched audit
# ===========================================================================
def matched_audit(before, after):
    """Cycle 20 versus cycle 21 for the same block."""
    out = {}
    for section in ('primal', 'consensus'):
        rows = {}
        for family in sorted(set(before.get(section, {})) | set(after.get(section, {}))):
            b = (before.get(section) or {}).get(family)
            a = (after.get(section) or {}).get(family)
            if not b or not a:
                rows[family] = {'present_cycle20': bool(b), 'present_cycle21': bool(a)}
                continue
            rows[family] = {
                'sha_changed': b.get('sha16') != a.get('sha16'),
                'absmax_20': b.get('absmax'), 'absmax_21': a.get('absmax'),
                'l1_20': b.get('l1'), 'l1_21': a.get('l1'),
                'l1_delta': (a.get('l1') - b.get('l1'))
                if (a.get('l1') is not None and b.get('l1') is not None) else None,
                'at_bound_20': b.get('at_bound'), 'at_bound_21': a.get('at_bound'),
                'nonfinite_21': a.get('nonfinite')}
        out[section] = rows
    out['structure'] = {
        'n_variables': (before.get('n_variables'), after.get('n_variables')),
        'n_constraints': (before.get('n_constraints'), after.get('n_constraints')),
        'changed': (before.get('n_variables') != after.get('n_variables')
                    or before.get('n_constraints') != after.get('n_constraints'))}
    out['objective'] = {
        'value_at_start_20': before.get('objective_value_at_start'),
        'value_at_start_21': after.get('objective_value_at_start'),
        'scale_20': before.get('admm_objective_scale'),
        'scale_21': after.get('admm_objective_scale'),
        'active_objective_20': before.get('active_objective'),
        'active_objective_21': after.get('active_objective')}
    out['rho'] = {k: (before.get(k), after.get(k)) for k in ('rho_v', 'rho_pf', 'rho_ess')}
    out['feasibility'] = {
        'start_max_violation_20': before.get('start_max_violation'),
        'start_max_violation_21': after.get('start_max_violation'),
        'worst_family_20': before.get('start_worst_family'),
        'worst_family_21': after.get('start_worst_family'),
        'violating_families_21': after.get('start_violating_families')}
    out['multipliers'] = {
        name: {'cycle20': (before.get('multipliers') or {}).get(name),
               'cycle21': (after.get('multipliers') or {}).get(name)}
        for name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual')}
    out['solver'] = {
        'succeeded_20': before.get('solver_succeeded'),
        'succeeded_21': after.get('solver_succeeded'),
        'summary_20': before.get('solver_summary'),
        'summary_21': after.get('solver_summary'),
        'iterations_20': (before.get('ipopt_log') or {}).get('iterations'),
        'iterations_21': (after.get('ipopt_log') or {}).get('iterations'),
        'exit_20': (before.get('ipopt_log') or {}).get('exit'),
        'exit_21': (after.get('ipopt_log') or {}).get('exit'),
        'unscaled_20': (before.get('ipopt_log') or {}).get('unscaled'),
        'unscaled_21': (after.get('ipopt_log') or {}).get('unscaled')}
    return out


# ===========================================================================
#  main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'p512b_cycle21_forensic.json')
    try:
        provenance, _ = gate('P5.12-B cycle-21 forensic', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.12-B] ABORTED\n{error}')
        sys.exit(1)

    cfg = OR.OracleConfig(
        scaling_mode=OR.SCALING_RESCALED, rho_v=RHO['v'], rho_pf=RHO['pf'],
        rho_ess=RHO['ess'], adaptive_penalty=False, neutralize_history=True,
        template_id='P512B-COLD-RESCALED',
        initialization_policy='original cold initialization',
        notes='P5.12-B cycle-21 forensic, capped at cycle 21')

    report = {'stage': 'P5.12-B', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'config': cfg.as_dict(), 'config_hash': cfg.config_hash,
              'target_cycle': TARGET_CYCLE, 'rho_fixed': RHO,
              'sequential_execution': True}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    planning = O.fresh_planning('p512b_forensic')
    report['rho_params_replaced'] = RH.apply_rho_to_params(planning, RHO)
    report['adaptive_before'] = RH.set_adaptive_penalty(planning, False)
    report['num_max_iters_before'] = planning.params.admm.num_max_iters
    planning.params.admm.num_max_iters = TARGET_CYCLE
    report['num_max_iters_on_deep_copy'] = TARGET_CYCLE
    report['parallel_execution'] = planning.parallel_execution
    persist()

    originals = install(planning)
    aborted = None
    started = time.time()
    console = io.StringIO()
    try:
        with redirect_stdout(console):
            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            with R.patched_admm_objectives() as applied:
                planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
        report['blocks_rescaled_at_build'] = len([v for v in applied.values() if v])
    except Cycle21Failure as error:
        aborted = str(error)
    finally:
        restore(originals)
    report['wall_clock_s'] = time.time() - started
    report['aborted_with'] = aborted
    report['console_tail'] = console.getvalue()[-4000:]

    report['solve_order'] = [
        {'cycle': c, 'block': k, 'succeeded': s} for c, k, s in STATE['order']]
    report['cycle20_blocks'] = STATE['captures'][20]
    report['cycle21_blocks'] = STATE['captures'][21]
    report['cycle21_failure'] = STATE['failure']
    report['cycle21_blocks_attempted'] = len(STATE['captures'][21])
    report['cycle21_blocks_succeeded'] = sum(
        1 for r in STATE['captures'][21].values() if r['solver_succeeded'])
    persist()

    if STATE['failure'] is not None:
        key = STATE['failure']['key']
        before = STATE['captures'][20].get(key)
        after = STATE['captures'][21].get(key)
        report['matched_audit'] = (matched_audit(before, after)
                                   if before and after else
                                   {'error': 'matching cycle-20 capture unavailable',
                                    'cycle20_present': bool(before)})
    persist()

    print('\n[P5.12-B] summary')
    print(f"    aborted_with          : {aborted}")
    print(f"    cycle-20 blocks captured: {len(STATE['captures'][20])}")
    print(f"    cycle-21 attempted/ok   : {report['cycle21_blocks_attempted']}"
          f"/{report['cycle21_blocks_succeeded']}")
    if STATE['failure']:
        rec = STATE['failure']['record']
        print(f"    FIRST FAILURE          : {STATE['failure']['key']}")
        print(f"    solver summary         : {rec.get('solver_summary')}")
        log = rec.get('ipopt_log') or {}
        print(f"    IPOPT exit / iters     : {log.get('exit')} / {log.get('iterations')}")
    print(f'\n[P5.12-B] report -> {out_path}')


if __name__ == '__main__':
    main()
