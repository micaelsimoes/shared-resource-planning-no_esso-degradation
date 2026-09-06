"""
Stage P5.3-B1 -- exact reference-angle gauge A/B.

    A (production) : f_ref in [-EQUALITY_TOLERANCE, +EQUALITY_TOLERANCE]
    B (diagnostic) : f_ref fixed to exactly 0

Nothing else changes. The hard production `sess_snet_def` equality is retained;
no P5.2 narrow-band formulation is used. Both branches run the complete
positive-bootstrap initialization (36 DSO + 12 TSO + 3 ESSO) from a fresh
production model set, with identical candidate, scenario realization, IPOPT
options, MA97, exact Hessian and cold start.

`f_ref` is fixed by hooking `Network.run_smopf` and fixing the reference-node
imaginary voltage on the model immediately before the real solve is delegated to
production -- so the model is otherwise exactly the production cold start.

Derivative diagnostics are recomputed on representative models using the
corrected P5.3-A2 method (uninitialized variables given a nominal value first,
exact singular values from the eigenvalues of J J^T).

Diagnostic only. No production file is modified.

    python p53b1_reference_gauge.py
"""

import io
import json
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
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p52a2_epsilon_sensitivity import parse_ipopt_blocks  # noqa: E402  (reuse)
from p53a2_jacobian_correction import (  # noqa: E402  (reuse corrected method)
    analyse_model,
    singular_spectrum,
)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53B1')
LOGS_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'Logs')

# representative models for the derivative re-audit
JACOBIAN_TAGS = {'case33_1/2030/Winter', 'case9/2025/Winter'}
SMOPF_HEADER = re.compile(r'Running SMOPF, Network (\S+?), (\S+?), (\S+?)\.\.\.')


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


def log_attempts(name, year, day):
    path = os.path.join(LOGS_DIR, f'optim_log_{name}_{year}_{day}.log')
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as handle:
        text = handle.read().decode('utf-8', errors='replace')
    blocks = parse_ipopt_blocks(text)
    def pack(b):
        if not b:
            return None
        out = {'iterations': int(b['iterations']) if 'iterations' in b else None,
               'exit': b.get('exit_status')}
        for key in ('objective', 'primal_infeasibility', 'dual_infeasibility',
                    'complementarity', 'overall_nlp_error'):
            v = b.get(key)
            if v is not None:
                out[key] = v[0] if isinstance(v, (list, tuple)) else v
        return out
    return pack(blocks[0] if blocks else None), pack(blocks[1] if len(blocks) > 1 else None)


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
            'recovery_succeeded': 'Network recovery solve succeeded' in chunk,
        }
    return events


def enumerate_results(container, prefix, out):
    if hasattr(container, 'solver'):
        out[prefix] = {
            'status': str(getattr(container.solver, 'status', None)),
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


def run_branch(fix_f_ref, capture_tags):
    """Full positive-bootstrap initialization; returns outcomes + captured models."""
    clear_logs()
    captured, gauge_info = {}, {}
    original = network_module.Network.run_smopf

    def patched(self, model, params, from_warm_start=False, print_header=True):
        tag = f'{self.name}/{self.year}/{self.day}'
        ref_idx = self.get_node_idx(self.get_reference_node_id())
        if fix_f_ref:
            n = 0
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.f[ref_idx, s_m, s_o, p].fix(0.0)
                        n += 1
            gauge_info[tag] = {'ref_node_idx': ref_idx, 'n_fixed': n}
        else:
            sample = model.f[ref_idx, 0, 0, 0]
            gauge_info[tag] = {'ref_node_idx': ref_idx,
                               'lb': sample.lb, 'ub': sample.ub, 'fixed': sample.fixed}
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
            all_ok = srp._admm_local_solves_succeeded(planning, res)
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
        primary, recovery = log_attempts(name, year, day)
        per_solve[tag] = {**ev, 'primary_block': primary, 'recovery_block': recovery}

    # interface quantities from the consensus variables
    interface = {}
    for node_id, per_year in consensus_vars['vmag']['dso']['current'].items():
        for year, per_day in per_year.items():
            for day, vals in per_day.items():
                key = f'{node_id}/{year}/{day}'
                interface[key] = {
                    'vmag': list(vals),
                    'pf_p': list(consensus_vars['pf']['dso']['current'][node_id][year][day]['p']),
                    'pf_q': list(consensus_vars['pf']['dso']['current'][node_id][year][day]['q']),
                }

    objectives = {}
    for node_id, per_year in dso_models.items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                if srp._solver_result_succeeded(res['dso'][node_id][year][day]):
                    try:
                        objectives[f'dso/{node_id}/{year}/{day}'] = float(pe.value(model.objective))
                    except Exception:
                        pass
    for year, per_day in tso_models.items():
        for day, model in per_day.items():
            if srp._solver_result_succeeded(res['tso'][year][day]):
                try:
                    objectives[f'tso/{year}/{day}'] = float(pe.value(model.objective))
                except Exception:
                    pass

    agg = {
        'total_local_solves': len(finals),
        'per_agent': {},
        'primary_failures': sum(1 for v in per_solve.values() if v['primary_failed']),
        'recovery_attempts': sum(1 for v in per_solve.values() if v['recovery_attempted']),
        'clean_primary_successes': sum(
            1 for tag, v in per_solve.items() if not v['primary_failed']),
        'persistent_failures': sum(1 for v in finals.values() if not v['succeeded']),
        'persistent_failure_ids': [k for k, v in finals.items() if not v['succeeded']],
        'admm_local_solves_succeeded': bool(all_ok),
        'runtime_s': runtime,
    }
    for key, value in finals.items():
        agent = key.split('/')[0]
        b = agg['per_agent'].setdefault(agent, {'total': 0, 'succeeded': 0})
        b['total'] += 1
        b['succeeded'] += 1 if value['succeeded'] else 0

    iters = [v['primary_block']['iterations'] for v in per_solve.values()
             if v['primary_block'] and v['primary_block'].get('iterations')]
    agg['iterations'] = {
        'n': len(iters), 'mean': float(np.mean(iters)) if iters else None,
        'median': float(np.median(iters)) if iters else None,
        'max': int(np.max(iters)) if iters else None,
        'total': int(np.sum(iters)) if iters else None}

    return {'aggregates': agg, 'per_solve': per_solve, 'finals': finals,
            'interface': interface, 'objectives': objectives,
            'gauge_info': {k: gauge_info[k] for k in list(gauge_info)[:3]},
            'captured': captured}


def jacobian_for(model, tag):
    work = model.clone()
    res = analyse_model(work, tag, repair_uninitialized=True)
    spec = singular_spectrum(res['rows'], res['n_columns_seen'])
    zero_owners = defaultdict(int)
    for i, cols in enumerate(res['rows']):
        if not cols:
            zero_owners[res['row_meta'][i]['component']] += 1
    spec['zero_row_components'] = dict(zero_owners)
    # f_ref column norm / presence
    fcol = {}
    for key, norm in res['col_norms'].items():
        name = res['var_of_id'][key].name
        if name.startswith('f[0,0,0,0]'):
            fcol[name] = norm
    spec['f_ref_column_norm'] = fcol
    spec['n_derivative_failures'] = res['n_derivative_failures']
    return spec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.3-B1', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'design': {'A': 'production f_ref in [-1e-5, +1e-5]',
                         'B': 'f_ref fixed to exactly 0',
                         'sess_snet_def': 'hard production equality retained in both'}}

    print('[B1] branch A (production gauge) ...', flush=True)
    A = run_branch(fix_f_ref=False, capture_tags=JACOBIAN_TAGS)
    print(f"    {A['aggregates']['per_agent']} persistent={A['aggregates']['persistent_failures']}", flush=True)

    print('[B1] branch B (f_ref = 0) ...', flush=True)
    B = run_branch(fix_f_ref=True, capture_tags=JACOBIAN_TAGS)
    print(f"    {B['aggregates']['per_agent']} persistent={B['aggregates']['persistent_failures']}", flush=True)

    jac = {}
    for tag in sorted(JACOBIAN_TAGS):
        for label, branch in (('A', A), ('B', B)):
            if tag in branch['captured']:
                print(f'[B1] jacobian {label} {tag} ...', flush=True)
                jac[f'{label}:{tag}'] = jacobian_for(branch['captured'][tag], tag)
    report['jacobian'] = jac

    # physical/economic comparison
    common = sorted(set(A['objectives']) & set(B['objectives']))
    obj_delta = {k: B['objectives'][k] - A['objectives'][k] for k in common}
    iface_delta = {}
    for key in sorted(set(A['interface']) & set(B['interface'])):
        for field in ('vmag', 'pf_p', 'pf_q'):
            a = np.array(A['interface'][key][field], dtype=float)
            b = np.array(B['interface'][key][field], dtype=float)
            if a.shape == b.shape and a.size:
                iface_delta.setdefault(field, []).append(float(np.max(np.abs(b - a))))
    report['comparison'] = {
        'n_common_objectives': len(common),
        'max_abs_objective_delta': max((abs(v) for v in obj_delta.values()), default=None),
        'max_rel_objective_delta': max(
            (abs(v) / max(abs(A['objectives'][k]), 1e-12) for k, v in obj_delta.items()),
            default=None),
        'interface_max_abs_delta': {k: max(v) for k, v in iface_delta.items()},
        'objective_deltas_sample': dict(list(obj_delta.items())[:10]),
    }

    for label, branch in (('A', A), ('B', B)):
        branch.pop('captured', None)
        report[f'branch_{label}'] = branch

    out_path = os.path.join(OUT_DIR, 'p53b1_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[B1] report -> {out_path}')
    for label in ('A', 'B'):
        a = report[f'branch_{label}']['aggregates']
        print(f"  {label}: solves={a['total_local_solves']} {a['per_agent']} "
              f"primary_failures={a['primary_failures']} recovery={a['recovery_attempts']} "
              f"persistent={a['persistent_failures']} {a['persistent_failure_ids']} "
              f"enter_admm={a['admm_local_solves_succeeded']} runtime={a['runtime_s']:.0f}s")
        print(f"     iterations: {a['iterations']}")
    print(f"  comparison: {json.dumps(report['comparison'], default=str)[:400]}")
    for key, spec in jac.items():
        f, r = spec['full'], spec['reduced']
        print(f"  {key:28s} zeroRows={f['n_exactly_zero_rows']} smin_full={f['sigma_min']:.2e} "
              f"reduced_smin={r['sigma_min_nonzero_subspace']:.4e} cond={r['condition_number']:.3e} "
              f"f_ref_col={spec['f_ref_column_norm']}")


if __name__ == '__main__':
    main()
