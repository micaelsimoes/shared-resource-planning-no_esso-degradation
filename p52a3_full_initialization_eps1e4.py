"""
Stage P5.2-A3 -- full positive-bootstrap initialization at eps_rel = 1e-4.

Runs the COMPLETE production operational initialization for the exact P5
iteration-2 positive-bootstrap candidate, with the shared-ESS `sess_snet_def`
rows converted in memory to the two-sided band

    -kappa*eps*S_rated^2 <= kappa*g <= +kappa*eps*S_rated^2,  eps_rel = 1e-4

and kappa = 1/S_rated left untouched (no cap). ADMM and the outer planning loop
are NOT entered.

Unlike the P5.2-A gate, every local solve is instrumented for its PRIMARY and
RECOVERY attempts separately:

  * per-solve primary/recovery status comes from the exact INFO/WARNING lines
    `_run_smopf` (network.py) and the ESSO solver already print;
  * iteration counts come from the IPOPT logs, parsed with a summary-anchored
    parser (these logs use file_print_level=6, so the metric names also appear
    on ~1000 per-iteration diagnostic lines and must not be parsed naively);
  * the Logs directory is cleared first so each log holds only this run.

Note: the ESSO subproblem uses MA57 and defines no `output_file`, so it writes
no IPOPT log; its attempts are instrumented from console markers only and its
iteration counts are unavailable.

Diagnostic only. No production file is modified.

    python p52a3_full_initialization_eps1e4.py
"""

import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p52a_narrow_band_diagnostic import apply_narrow_band  # noqa: E402  (reuse)
from p52a2_epsilon_sensitivity import parse_ipopt_blocks  # noqa: E402  (reuse)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P52A3')
LOGS_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'Logs')
EPSILON_REL = 1e-4

SMOPF_HEADER = re.compile(r'Running SMOPF, Network (\S+?), (\S+?), (\S+?)\.\.\.')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def parse_console_solves(text):
    """Segment the console by `Running SMOPF` headers and classify each solve's
    primary/recovery path from the exact lines production prints."""
    events = []
    positions = [(m.start(), m.group(1), m.group(2), m.group(3))
                 for m in SMOPF_HEADER.finditer(text)]
    for i, (start, name, year, day) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk = text[start:end]
        primary_failed_recovery = 'Network primary solve did not converge' in chunk
        no_recovery_failure = 'Network solver did not converge' in chunk
        recovery_attempted = 'Retrying network solve once' in chunk
        recovery_succeeded = 'Network recovery solve succeeded' in chunk
        recovery_failed = 'Network recovery solve did not converge' in chunk
        events.append({
            'network': name, 'year': year, 'day': day,
            'primary_failed': primary_failed_recovery or no_recovery_failure,
            'recovery_attempted': recovery_attempted,
            'recovery_succeeded': recovery_succeeded,
            'recovery_failed': recovery_failed,
            'failed_without_recovery_attempt': no_recovery_failure,
        })
    esso = {
        'primary_failures': text.count('Shared ESS primary solve did not converge'),
        'recovery_attempts': text.count('Retrying Shared ESS solve once'),
        'solver_execution_failures': text.count('Shared ESS solver execution failed'),
    }
    return events, esso


def log_iterations(network_name, year, day):
    """Return (primary_iters, recovery_iters) from that solve's IPOPT log."""
    path = os.path.join(LOGS_DIR, f'optim_log_{network_name}_{year}_{day}.log')
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'rb') as handle:
        text = handle.read().decode('utf-8', errors='replace')
    blocks = parse_ipopt_blocks(text)
    if not blocks:
        return None, None, 0

    def it(block):
        return int(block['iterations']) if block and 'iterations' in block else None

    def exitstr(block):
        return block.get('exit_status') if block else None

    return ({'iterations': it(blocks[0]), 'exit': exitstr(blocks[0])},
            ({'iterations': it(blocks[1]), 'exit': exitstr(blocks[1])}
             if len(blocks) > 1 else None),
            len(blocks))


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
        out[prefix] = {'unparsed': type(container).__name__}
        return out
    for key, value in items:
        enumerate_results(value, f'{prefix}/{key}', out)
    return out


def audit_model_rows(model, network, tag, rows_out):
    """Collect the original unscaled g and apparent-power mismatch for every
    ACTIVE shared-ESS row of a solved model."""
    if not hasattr(model, 'sess_snet_def'):
        return
    base_mva = network.baseMVA
    for index in model.sess_snet_def:
        con = model.sess_snet_def[index]
        if not con.active:
            continue
        e = index[0]
        s_rated = float(pe.value(model.shared_es_s_rated_fixed[e]))
        if s_rated <= 0:
            continue
        _e, s_m, s_o, p = index
        sch = pe.value(model.shared_es_sch[e, s_m, s_o, p])
        sdch = pe.value(model.shared_es_sdch[e, s_m, s_o, p])
        pnet = pe.value(model.shared_es_pnet[e, s_m, s_o, p])
        qnet = pe.value(model.shared_es_qnet[e, s_m, s_o, p])
        if None in (sch, sdch, pnet, qnet):
            continue
        g = (sch - sdch) ** 2 - pnet ** 2 - qnet ** 2
        normalized = abs(g) / (s_rated ** 2)
        rows_out.append({
            'tag': tag, 'shared_ess_index': e, 'period': p,
            's_rated_pu': s_rated,
            'abs_g': abs(g),
            'normalized': normalized,
            'band_utilization': normalized / EPSILON_REL,
            'delta_S_pu': abs(abs(sch - sdch) - math.sqrt(pnet ** 2 + qnet ** 2)),
            'delta_S_MVA': abs(abs(sch - sdch) - math.sqrt(pnet ** 2 + qnet ** 2)) * base_mva,
            'delta_S_over_s_rated': abs(abs(sch - sdch) - math.sqrt(pnet ** 2 + qnet ** 2)) / s_rated,
        })


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.2-A3', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'epsilon_rel': EPSILON_REL,
              'kappa_policy': 'untouched production 1/S_rated (no cap)',
              'applied_in_memory_only': True}

    # clear logs so each file holds only this run
    if os.path.isdir(LOGS_DIR):
        for name in os.listdir(LOGS_DIR):
            if name.startswith('optim_log_'):
                try:
                    os.remove(os.path.join(LOGS_DIR, name))
                except OSError:
                    pass

    original_configure = srp.configure_shared_ess_operational_state

    def configure_then_band(model, shared_ess_idx, s_capacity, e_capacity, *args, **kwargs):
        out = original_configure(model, shared_ess_idx, s_capacity, e_capacity, *args, **kwargs)
        active = any(model.sess_snet_def[i].active
                     for i in model.sess_snet_def if i[0] == shared_ess_idx)
        if active and s_capacity and s_capacity > 0:
            apply_narrow_band(model, shared_ess_idx, EPSILON_REL, float(s_capacity))
        return out

    console_buffer = io.StringIO()
    started = time.time()
    try:
        srp.configure_shared_ess_operational_state = configure_then_band
        with redirect_stdout(console_buffer):
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
            tso_model, res['tso'] = srp.create_transmission_network_model(
                planning, consensus_vars, candidate['total_capacity'])
            esso_model, res['esso'] = srp.create_shared_energy_storage_model(
                planning.shared_ess_data, consensus_vars, candidate['investment'])
            all_ok = srp._admm_local_solves_succeeded(planning, res)
    finally:
        srp.configure_shared_ess_operational_state = original_configure
    wall = time.time() - started
    console = console_buffer.getvalue()

    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', console) or [None])[-1]
    report['wall_clock_s'] = wall

    # ---- authoritative final results ----
    finals = {}
    for agent in ('dso', 'tso', 'esso'):
        enumerate_results(res[agent], agent, finals)

    # ---- per-solve primary/recovery instrumentation ----
    console_events, esso_console = parse_console_solves(console)
    per_solve = []
    for event in console_events:
        primary, recovery, n_blocks = log_iterations(
            event['network'], event['year'], event['day'])
        # map back to the authoritative final result
        final = None
        for key, value in finals.items():
            parts = key.split('/')
            if len(parts) >= 3 and parts[-2] == str(event['year']) and parts[-1] == str(event['day']):
                agent = parts[0]
                if agent == 'tso' and event['network'] == planning.transmission_network.name:
                    final = (key, value)
                    break
                if agent == 'dso':
                    node = parts[1]
                    dn = planning.distribution_networks.get(int(node))
                    if dn is not None and dn.name == event['network']:
                        final = (key, value)
                        break
        entry = {
            'network': event['network'], 'year': event['year'], 'day': event['day'],
            'agent': (final[0].split('/')[0] if final else 'unknown'),
            'node': (final[0].split('/')[1] if final and final[0].startswith('dso') else None),
            'primary_failed': event['primary_failed'],
            'primary_iterations': primary['iterations'] if primary else None,
            'primary_exit': primary['exit'] if primary else None,
            'recovery_attempted': event['recovery_attempted'],
            'recovery_iterations': recovery['iterations'] if recovery else None,
            'recovery_exit': recovery['exit'] if recovery else None,
            'recovery_succeeded': event['recovery_succeeded'],
            'failed_without_recovery_attempt': event['failed_without_recovery_attempt'],
            'final_status': final[1]['status'] if final else None,
            'final_termination': final[1]['termination'] if final else None,
            'final_succeeded': final[1]['succeeded'] if final else None,
            'clean_primary_success': (final[1]['succeeded'] if final else False)
                                     and not event['recovery_attempted'],
            'ipopt_attempt_blocks': n_blocks,
        }
        per_solve.append(entry)
    report['per_solve'] = per_solve
    report['esso_console_markers'] = esso_console

    # ---- aggregates ----
    n_total = len(finals)
    per_agent = {}
    for key, value in finals.items():
        agent = key.split('/')[0]
        bucket = per_agent.setdefault(agent, {'total': 0, 'succeeded': 0, 'failed': 0})
        bucket['total'] += 1
        bucket['succeeded' if value.get('succeeded') else 'failed'] += 1
    persistent = {k: v for k, v in finals.items() if not v.get('succeeded')}

    report['aggregates'] = {
        'total_local_solves': n_total,
        'per_agent': per_agent,
        'network_solves_instrumented': len(per_solve),
        'primary_successes': sum(1 for e in per_solve if not e['primary_failed']),
        'primary_failures': sum(1 for e in per_solve if e['primary_failed']),
        'recovery_attempts': sum(1 for e in per_solve if e['recovery_attempted']),
        'recovery_successes': sum(1 for e in per_solve if e['recovery_succeeded']),
        'clean_primary_successes': sum(1 for e in per_solve if e['clean_primary_success']),
        'persistent_failures': len(persistent),
        'persistent_failure_identities': persistent,
        'admm_local_solves_succeeded': bool(all_ok),
        'would_enter_admm': bool(all_ok),
    }

    # ---- physical-relaxation audit over every active shared-ESS row ----
    rows = []
    for node_id, per_year in dso_models.items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                result = res['dso'][node_id][year][day]
                if not srp._solver_result_succeeded(result):
                    continue
                network = planning.distribution_networks[node_id].network[year][day]
                audit_model_rows(model, network, f'dso/{node_id}/{year}/{day}', rows)
    for year, per_day in tso_model.items():
        for day, model in per_day.items():
            result = res['tso'][year][day]
            if not srp._solver_result_succeeded(result):
                continue
            network = planning.transmission_network.network[year][day]
            audit_model_rows(model, network, f'tso/{year}/{day}', rows)

    if rows:
        norms = [r['normalized'] for r in rows]
        utils = [r['band_utilization'] for r in rows]
        report['physical_audit'] = {
            'n_active_shared_ess_rows': len(rows),
            'max_normalized_g': max(norms),
            'mean_normalized_g': statistics.fmean(norms),
            'p95_normalized_g': percentile(norms, 0.95),
            'max_band_utilization': max(utils),
            'mean_band_utilization': statistics.fmean(utils),
            'p95_band_utilization': percentile(utils, 0.95),
            'n_rows_above_0p5_utilization': sum(1 for u in utils if u > 0.5),
            'frac_rows_above_0p5_utilization': sum(1 for u in utils if u > 0.5) / len(utils),
            'n_rows_above_0p9_utilization': sum(1 for u in utils if u > 0.9),
            'frac_rows_above_0p9_utilization': sum(1 for u in utils if u > 0.9) / len(utils),
            'n_rows_active_at_boundary': sum(1 for u in utils if u >= 1.0 - 1e-9),
            'max_delta_S_pu': max(r['delta_S_pu'] for r in rows),
            'max_delta_S_MVA': max(r['delta_S_MVA'] for r in rows),
            'max_delta_S_over_s_rated': max(r['delta_S_over_s_rated'] for r in rows),
            'worst_10_rows_by_utilization': sorted(
                rows, key=lambda r: -r['band_utilization'])[:10],
        }
    else:
        report['physical_audit'] = {'n_active_shared_ess_rows': 0}

    out_path = os.path.join(OUT_DIR, 'p52a3_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    a = report['aggregates']
    print(f'[P5.2-A3] report -> {out_path}')
    print(f"[P5.2-A3] checksum {report['scenario_checksum']} eps_rel={EPSILON_REL} wall={wall:.1f}s")
    print(f"[P5.2-A3] total local solves = {a['total_local_solves']} | per agent = {a['per_agent']}")
    print(f"[P5.2-A3] network solves instrumented = {a['network_solves_instrumented']}")
    print(f"[P5.2-A3] primary successes = {a['primary_successes']} | primary failures = {a['primary_failures']}")
    print(f"[P5.2-A3] recovery attempts = {a['recovery_attempts']} | recovery successes = {a['recovery_successes']}")
    print(f"[P5.2-A3] clean primary successes = {a['clean_primary_successes']}")
    print(f"[P5.2-A3] PERSISTENT FAILURES = {a['persistent_failures']} {list(a['persistent_failure_identities'])}")
    print(f"[P5.2-A3] _admm_local_solves_succeeded = {a['admm_local_solves_succeeded']} | would enter ADMM = {a['would_enter_admm']}")
    for e in per_solve:
        if e['primary_failed'] or e['recovery_attempted'] or not e['final_succeeded']:
            print(f"    NOTE {e['agent']}/{e.get('node')}/{e['network']}/{e['year']}/{e['day']}: "
                  f"primary_failed={e['primary_failed']} exit='{e['primary_exit']}' "
                  f"recovery={e['recovery_attempted']}/{e['recovery_succeeded']} "
                  f"final={e['final_status']}/{e['final_termination']}")
    ph = report['physical_audit']
    if ph.get('n_active_shared_ess_rows'):
        print(f"[P5.2-A3] physical: rows={ph['n_active_shared_ess_rows']} "
              f"max|g|/S^2={ph['max_normalized_g']:.3e} mean={ph['mean_normalized_g']:.3e} "
              f"p95={ph['p95_normalized_g']:.3e}")
        print(f"[P5.2-A3] band util: max={ph['max_band_utilization']:.3e} p95={ph['p95_band_utilization']:.3e} "
              f">0.5: {ph['n_rows_above_0p5_utilization']} ({ph['frac_rows_above_0p5_utilization']:.2%}) "
              f">0.9: {ph['n_rows_above_0p9_utilization']} active={ph['n_rows_active_at_boundary']}")
        print(f"[P5.2-A3] max dS={ph['max_delta_S_pu']:.3e} pu = {ph['max_delta_S_MVA']:.3e} MVA ; "
              f"dS/S_rated={ph['max_delta_S_over_s_rated']:.3e}")


if __name__ == '__main__':
    main()
