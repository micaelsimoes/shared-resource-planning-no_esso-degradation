"""
Stage P5.3-B3 -- active-power shared-ESS structural prototype.

    A (production) : sch/sdch geometry, sess_snet_def, sess_s_limit,
                     sess_pch_link, sess_pdch_link, apparent-power SOC,
                     sch*sdch complementarity
    B (prototype)  : active-power only --
                     pnet = pch - pdch                       (unchanged production row)
                     SOC_t = SOC_{t-1} + eta_ch*pch*dt - pdch*dt/eta_dch
                     pnet^2 + qnet^2 <= S_rated^2            (converter capability)
                     pch + pdch <= S_rated                   (derived, see below)
                     pch*pdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2

B3.4 derivation of the active envelope, from the production feasible set:
    pch <= sch,  pdch <= sdch,  sch + sdch <= S_rated
  =>  pch + pdch <= sch + sdch <= S_rated
so `pch + pdch <= S_rated` is IMPLIED by production and is used as the baseline
active envelope. The box bounds 0 <= pch,pdch <= S_rated and
|pnet|,|qnet| <= S_rated are likewise implied by production, so they are
redundant-but-explicit, not new restrictions.

B3.2 time basis: production has no explicit time step anywhere; `num_instants`
is 24 over one representative day, so dt = 24h/24 = 1 h. Power is p.u. on
baseMVA and energy p.u. on the same base (results scale SOC by baseMVA to MWh),
so the numeric factor is exactly 1 and the production SOC recursion implicitly
assumes dt = 1 h. dt is DERIVED here from num_instants, not hardcoded.

Objective handling: production's ESS utilization and complementarity penalties
are written on sch/sdch. Fixing sch/sdch to zero would silently delete them, so
the prototype adds the exact active-power analogues back, keeping the objective
comparable.

Diagnostic only -- production source is never modified; the conversion is
applied in memory to already-built production models.

    python p53b3_active_power_ess.py [--quick]
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import network as network_module  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import (ESS_COMPLEMENTARITY_TOLERANCE, ESS_MODEL_BILINEAR_RELAXATION,  # noqa: E402
                         PENALTY_ESS_COMPLEMENTARITY, SHARED_ESS_ZERO_CAPACITY_TOLERANCE)
from model_construction_helpers import period_duration_hours  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p52a2_epsilon_sensitivity import parse_ipopt_blocks  # noqa: E402  (reuse)
from p53a2_jacobian_correction import analyse_model, singular_spectrum  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53B3')
LOGS_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'Logs')

JACOBIAN_TAGS = {'case33_1/2030/Winter', 'case9/2025/Winter'}
SMOPF_HEADER = re.compile(r'Running SMOPF, Network (\S+?), (\S+?), (\S+?)\.\.\.')

DEACTIVATE = ('sess_snet_def', 'sess_pch_link', 'sess_pdch_link',
              'sess_s_limit', 'sess_soc_def', 'sess_comp')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def clear_logs():
    if os.path.isdir(LOGS_DIR):
        for name in os.listdir(LOGS_DIR):
            if name.startswith('optim_log_'):
                try:
                    os.remove(os.path.join(LOGS_DIR, name))
                except OSError:
                    pass


# ---------------------------------------------------------------------------
#  The prototype conversion
# ---------------------------------------------------------------------------
def convert_to_active_power(model, network, params):
    """Convert the shared-ESS block of an already-built, already-configured
    production model to the active-power formulation, in place."""
    info = {'converted_indices': [], 'skipped_zero_capacity': [], 'dt_hours': None}

    n_periods = len(list(model.periods))
    dt = 24.0 / n_periods          # derived, not assumed
    info['dt_hours'] = dt
    info['n_periods'] = n_periods
    base = network.baseMVA

    live = []
    for e in model.shared_energy_storages:
        s_rated = float(pe.value(model.shared_es_s_rated_fixed[e]))
        rows_active = any(model.sess_pnet_def[i].active
                          for i in model.sess_pnet_def if i[0] == e)
        if s_rated <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE or not rows_active:
            info['skipped_zero_capacity'].append({'index': e, 's_rated': s_rated})
            continue
        live.append((e, s_rated))
        info['converted_indices'].append({'index': e, 's_rated': s_rated})

    # 1. deactivate the apparent-power geometry for the live indices
    deactivated = defaultdict(int)
    for name in DEACTIVATE:
        comp = getattr(model, name, None)
        if comp is None:
            continue
        for index in comp:
            e = index[0] if isinstance(index, tuple) else index
            if any(e == le for le, _ in live) and comp[index].active:
                comp[index].deactivate()
                deactivated[name] += 1
    info['deactivated_rows'] = dict(deactivated)

    # 2. fix sch/sdch out of the problem for the live indices
    n_fixed = 0
    for e, _ in live:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.shared_es_sch[e, s_m, s_o, p].fix(0.0)
                    model.shared_es_sdch[e, s_m, s_o, p].fix(0.0)
                    n_fixed += 2
    info['n_sch_sdch_fixed'] = n_fixed

    # 3. explicit (production-implied) box bounds
    for e, s_rated in live:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.shared_es_pch[e, s_m, s_o, p].setlb(0.0)
                    model.shared_es_pch[e, s_m, s_o, p].setub(s_rated)
                    model.shared_es_pdch[e, s_m, s_o, p].setlb(0.0)
                    model.shared_es_pdch[e, s_m, s_o, p].setub(s_rated)
                    model.shared_es_pnet[e, s_m, s_o, p].setlb(-s_rated)
                    model.shared_es_pnet[e, s_m, s_o, p].setub(s_rated)
                    model.shared_es_qnet[e, s_m, s_o, p].setlb(-s_rated)
                    model.shared_es_qnet[e, s_m, s_o, p].setub(s_rated)

    idx = [(e, s_m, s_o, p) for e, _ in live
           for s_m in model.scenarios_market
           for s_o in model.scenarios_operation
           for p in model.periods]
    rated = {e: s for e, s in live}

    # 4. active-power SOC
    def soc_rule(m, e, s_m, s_o, p):
        sess = network.shared_energy_storages[e]
        if p == 0:
            from definitions import ENERGY_STORAGE_RELATIVE_INIT_SOC
            soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        else:
            soc_prev = m.shared_es_soc[e, s_m, s_o, p - 1]
        delta = (sess.eff_ch * m.shared_es_pch[e, s_m, s_o, p] * dt
                 - m.shared_es_pdch[e, s_m, s_o, p] * dt / sess.eff_dch)
        return m.shared_es_soc[e, s_m, s_o, p] == soc_prev + delta

    model.b3_sess_soc_active = pe.Constraint(idx, rule=soc_rule)

    # 5. converter capability
    def capability_rule(m, e, s_m, s_o, p):
        return (m.shared_es_pnet[e, s_m, s_o, p] ** 2
                + m.shared_es_qnet[e, s_m, s_o, p] ** 2) <= rated[e] ** 2
    model.b3_sess_converter_capability = pe.Constraint(idx, rule=capability_rule)

    # 6. derived active envelope  pch + pdch <= S_rated
    def envelope_rule(m, e, s_m, s_o, p):
        return (m.shared_es_pch[e, s_m, s_o, p]
                + m.shared_es_pdch[e, s_m, s_o, p]) <= rated[e]
    model.b3_sess_active_sum_limit = pe.Constraint(idx, rule=envelope_rule)

    # 7. complementarity moved to active power, tolerance preserved exactly
    def comp_rule(m, e, s_m, s_o, p):
        return (m.shared_es_pch[e, s_m, s_o, p] * m.shared_es_pdch[e, s_m, s_o, p]
                <= ESS_COMPLEMENTARITY_TOLERANCE * rated[e] ** 2)
    model.b3_sess_comp_active = pe.Constraint(idx, rule=comp_rule)

    # 8. objective: restore the active-power analogues of the sch/sdch penalties
    obj = list(model.component_data_objects(pe.Objective, active=True))[0]
    added = 0.0
    penalty_usage = pe.value(model.penalty_ess_usage) if hasattr(model, 'penalty_ess_usage') else None
    if penalty_usage is not None:
        for (e, s_m, s_o, p) in idx:
            added += (model.penalty_ess_usage * base
                      * (model.shared_es_pch[e, s_m, s_o, p]
                         + model.shared_es_pdch[e, s_m, s_o, p]))
    if params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:
        for (e, s_m, s_o, p) in idx:
            added += (base * PENALTY_ESS_COMPLEMENTARITY
                      * model.shared_es_pch[e, s_m, s_o, p]
                      * model.shared_es_pdch[e, s_m, s_o, p])
    if not isinstance(added, float):
        expr = obj.expr + added
        obj.deactivate()
        model.b3_objective = pe.Objective(expr=expr, sense=obj.sense)
        info['objective_replaced'] = True
    else:
        info['objective_replaced'] = False

    info['n_new_rows'] = 4 * len(idx)
    return info


# ---------------------------------------------------------------------------
#  Solve harness
# ---------------------------------------------------------------------------
def parse_console(text):
    events = {}
    positions = [(m.start(), m.group(1), m.group(2), m.group(3))
                 for m in SMOPF_HEADER.finditer(text)]
    for i, (start, name, year, day) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk = text[start:end]
        events[f'{name}/{year}/{day}'] = {
            'primary_failed': ('Network primary solve did not converge' in chunk
                               or 'Network solver did not converge' in chunk),
            'recovery_attempted': 'Retrying network solve once' in chunk,
            'recovery_succeeded': 'Network recovery solve succeeded' in chunk}
    return events


def log_blocks(name, year, day):
    path = os.path.join(LOGS_DIR, f'optim_log_{name}_{year}_{day}.log')
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as handle:
        blocks = parse_ipopt_blocks(handle.read().decode('utf-8', errors='replace'))
    def pack(b):
        if not b:
            return None
        out = {'iterations': int(b['iterations']) if 'iterations' in b else None,
               'exit': b.get('exit_status')}
        for k in ('objective', 'primal_infeasibility', 'dual_infeasibility',
                  'complementarity', 'overall_nlp_error'):
            v = b.get(k)
            if v is not None:
                out[k] = v[0] if isinstance(v, (list, tuple)) else v
        return out
    return pack(blocks[0] if blocks else None), pack(blocks[1] if len(blocks) > 1 else None)


def enumerate_results(container, prefix, out):
    if hasattr(container, 'solver'):
        out[prefix] = {'status': str(getattr(container.solver, 'status', None)),
                       'termination': str(getattr(container.solver, 'termination_condition', None)),
                       'succeeded': bool(srp._solver_result_succeeded(container))}
        return out
    try:
        items = container.items()
    except AttributeError:
        return out
    for key, value in items:
        enumerate_results(value, f'{prefix}/{key}', out)
    return out


def run_branch(prototype, capture_tags):
    clear_logs()
    captured, conv_info = {}, {}
    original = network_module.Network.run_smopf

    def patched(self, model, params, from_warm_start=False, print_header=True):
        tag = f'{self.name}/{self.year}/{self.day}'
        if prototype:
            conv_info[tag] = convert_to_active_power(model, self, params)
        if tag in capture_tags:
            captured[tag] = model.clone()
        return original(self, model, params, from_warm_start=from_warm_start,
                        print_header=print_header)

    console = io.StringIO()
    started = time.time()
    try:
        network_module.Network.run_smopf = patched
        with redirect_stdout(console):
            planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
            planning.read_planning_problem()
            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            consensus_vars, _dual = srp.create_admm_variables(planning)
            res = {'tso': dict(), 'dso': dict(), 'esso': dict()}
            dso_models, res['dso'] = srp.create_distribution_networks_models(
                planning.distribution_networks, consensus_vars,
                candidate['total_capacity'],
                parallel_execution=planning.parallel_execution)
            tso_models, res['tso'] = srp.create_transmission_network_model(
                planning, consensus_vars, candidate['total_capacity'])
            _em, res['esso'] = srp.create_shared_energy_storage_model(
                planning.shared_ess_data, consensus_vars, candidate['investment'])
    finally:
        network_module.Network.run_smopf = original
    runtime = time.time() - started
    text = console.getvalue()

    finals = {}
    for agent in ('dso', 'tso', 'esso'):
        enumerate_results(res[agent], agent, finals)
    events = parse_console(text)
    per_solve = {}
    for tag, ev in events.items():
        name, year, day = tag.split('/')
        primary, recovery = log_blocks(name, year, day)
        per_solve[tag] = {**ev, 'primary_block': primary, 'recovery_block': recovery}

    iters = [v['primary_block']['iterations'] for v in per_solve.values()
             if v['primary_block'] and v['primary_block'].get('iterations')]
    agg = {
        'total_local_solves': len(finals),
        'per_agent': {},
        'primary_failures': sum(1 for v in per_solve.values() if v['primary_failed']),
        'recovery_attempts': sum(1 for v in per_solve.values() if v['recovery_attempted']),
        'recovery_successes': sum(1 for v in per_solve.values() if v['recovery_succeeded']),
        'persistent_failures': sum(1 for v in finals.values() if not v['succeeded']),
        'persistent_failure_ids': [k for k, v in finals.items() if not v['succeeded']],
        'runtime_s': runtime,
        'iterations': {'n': len(iters),
                       'mean': float(np.mean(iters)) if iters else None,
                       'median': float(np.median(iters)) if iters else None,
                       'max': int(np.max(iters)) if iters else None,
                       'total': int(np.sum(iters)) if iters else None},
    }
    for key, value in finals.items():
        agent = key.split('/')[0]
        b = agg['per_agent'].setdefault(agent, {'total': 0, 'succeeded': 0})
        b['total'] += 1
        b['succeeded'] += 1 if value['succeeded'] else 0

    # physics/solution extraction on successful DSO models
    physics = {}
    for node_id, per_year in dso_models.items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                if not srp._solver_result_succeeded(res['dso'][node_id][year][day]):
                    continue
                net = planning.distribution_networks[node_id].network[year][day]
                e_idx = net.get_shared_energy_storage_idx(net.get_reference_node_id())
                s_rated = float(pe.value(model.shared_es_s_rated_fixed[e_idx]))
                if s_rated <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                    continue
                # real production parameters, read from the live model/network --
                # never defaulted, since P5.4-E2 converts these into MW/MWh
                sess = net.shared_energy_storages[e_idx]
                rec = {'s_rated': s_rated, 'pch': [], 'pdch': [], 'pnet': [],
                       'qnet': [], 'soc': [], 'cap_margin': [], 'comp_margin': [],
                       'e_rated': float(pe.value(model.shared_es_e_rated_fixed[e_idx])),
                       'eff_ch': float(sess.eff_ch), 'eff_dch': float(sess.eff_dch),
                       'dt': float(period_duration_hours(model)),
                       'base_mva': float(net.baseMVA)}
                for p in model.periods:
                    pch = float(pe.value(model.shared_es_pch[e_idx, 0, 0, p]))
                    pdch = float(pe.value(model.shared_es_pdch[e_idx, 0, 0, p]))
                    pnet = float(pe.value(model.shared_es_pnet[e_idx, 0, 0, p]))
                    qnet = float(pe.value(model.shared_es_qnet[e_idx, 0, 0, p]))
                    soc = float(pe.value(model.shared_es_soc[e_idx, 0, 0, p]))
                    rec['pch'].append(pch); rec['pdch'].append(pdch)
                    rec['pnet'].append(pnet); rec['qnet'].append(qnet)
                    rec['soc'].append(soc)
                    rec['cap_margin'].append(s_rated ** 2 - (pnet ** 2 + qnet ** 2))
                    rec['comp_margin'].append(
                        ESS_COMPLEMENTARITY_TOLERANCE * s_rated ** 2 - pch * pdch)
                try:
                    rec['objective'] = float(pe.value(
                        model.b3_objective if hasattr(model, 'b3_objective')
                        else model.objective))
                except Exception:
                    pass
                physics[f'dso/{node_id}/{year}/{day}'] = rec

    # the TSO model carries one shared ESS per active distribution network node,
    # so its rows must be audited too rather than assumed identical to the DSOs'
    for year, per_day in tso_models.items():
        for day, model in per_day.items():
            if not srp._solver_result_succeeded(res['tso'][year][day]):
                continue
            net = planning.transmission_network.network[year][day]
            for e_idx, sess in enumerate(net.shared_energy_storages):
                s_rated = float(pe.value(model.shared_es_s_rated_fixed[e_idx]))
                if s_rated <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                    continue
                rec = {'s_rated': s_rated, 'pch': [], 'pdch': [], 'pnet': [],
                       'qnet': [], 'soc': [], 'cap_margin': [], 'comp_margin': [],
                       'e_rated': float(pe.value(model.shared_es_e_rated_fixed[e_idx])),
                       'eff_ch': float(sess.eff_ch), 'eff_dch': float(sess.eff_dch),
                       'dt': float(period_duration_hours(model)),
                       'base_mva': float(net.baseMVA)}
                for p in model.periods:
                    pch = float(pe.value(model.shared_es_pch[e_idx, 0, 0, p]))
                    pdch = float(pe.value(model.shared_es_pdch[e_idx, 0, 0, p]))
                    pnet = float(pe.value(model.shared_es_pnet[e_idx, 0, 0, p]))
                    qnet = float(pe.value(model.shared_es_qnet[e_idx, 0, 0, p]))
                    rec['pch'].append(pch); rec['pdch'].append(pdch)
                    rec['pnet'].append(pnet); rec['qnet'].append(qnet)
                    rec['soc'].append(float(pe.value(model.shared_es_soc[e_idx, 0, 0, p])))
                    rec['cap_margin'].append(s_rated ** 2 - (pnet ** 2 + qnet ** 2))
                    rec['comp_margin'].append(
                        ESS_COMPLEMENTARITY_TOLERANCE * s_rated ** 2 - pch * pdch)
                physics[f'tso/{e_idx}/{year}/{day}'] = rec

    return {'aggregates': agg, 'per_solve': per_solve, 'finals': finals,
            'physics': physics, 'conversion': dict(list(conv_info.items())[:2]),
            'captured': captured}


def jacobian_for(model, tag):
    work = model.clone()
    res = analyse_model(work, tag, repair_uninitialized=True)
    spec = singular_spectrum(res['rows'], res['n_columns_seen'])
    owners = defaultdict(int)
    for i, cols in enumerate(res['rows']):
        if not cols:
            owners[res['row_meta'][i]['component']] += 1
    spec['zero_row_components'] = dict(owners)
    spec['n_derivative_failures'] = res['n_derivative_failures']
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    report = {'stage': 'P5.3-B3', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'complementarity_tolerance': ESS_COMPLEMENTARITY_TOLERANCE}

    print('[B3] branch A (production) ...', flush=True)
    A = run_branch(prototype=False, capture_tags=JACOBIAN_TAGS)
    print(f"    {A['aggregates']['per_agent']} persistent={A['aggregates']['persistent_failures']} "
          f"{A['aggregates']['persistent_failure_ids']}", flush=True)

    print('[B3] branch B (active-power prototype) ...', flush=True)
    B = run_branch(prototype=True, capture_tags=JACOBIAN_TAGS)
    print(f"    {B['aggregates']['per_agent']} persistent={B['aggregates']['persistent_failures']} "
          f"{B['aggregates']['persistent_failure_ids']}", flush=True)
    report['conversion_sample'] = B['conversion']

    if not args.quick:
        jac = {}
        for tag in sorted(JACOBIAN_TAGS):
            for label, branch in (('A', A), ('B', B)):
                if tag in branch['captured']:
                    print(f'[B3] jacobian {label} {tag} ...', flush=True)
                    jac[f'{label}:{tag}'] = jacobian_for(branch['captured'][tag], tag)
        report['jacobian'] = jac

    for label, branch in (('A', A), ('B', B)):
        branch.pop('captured', None)
        report[f'branch_{label}'] = branch

    out_path = os.path.join(OUT_DIR, 'p53b3_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[B3] report -> {out_path}')
    for label in ('A', 'B'):
        a = report[f'branch_{label}']['aggregates']
        print(f"  {label}: {a['per_agent']} primary_fail={a['primary_failures']} "
              f"recov={a['recovery_attempts']} persistent={a['persistent_failures']} "
              f"{a['persistent_failure_ids']} iters={a['iterations']} rt={a['runtime_s']:.0f}s")
    for key, spec in report.get('jacobian', {}).items():
        f, r = spec['full'], spec['reduced']
        print(f"  {key:26s} zeroRows={f['n_exactly_zero_rows']} owners={spec['zero_row_components']} "
              f"smin_full={f['sigma_min']:.3e} rank={f['numerical_rank']}/{f['n_rows']} "
              f"reduced_smin={r['sigma_min_nonzero_subspace']:.4e} cond={r['condition_number']:.3e}")


if __name__ == '__main__':
    main()
