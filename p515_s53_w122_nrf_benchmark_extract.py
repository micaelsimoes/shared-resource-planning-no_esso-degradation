"""P5.15 Addendum 57, Planner task W122 -- ZERO-SOLVE extraction from the SRP1 NRF benchmark outputs (spec v5 bca69f97).

Reads ONLY the write-once stage outputs of W122 under data/SRP1/Results/P515S53/w116_benchmark_nrf/ (the six NRF arm
runs *_r2, the two passive tie-breaker runs *_r2, report_v3) and the DN case files (bus ids / Vmax), and writes ONE
write-once JSON (through gate_result_io) with the per-run figures the W122 handoff reports. Imports: stdlib and
gate_result_io only (no pyomo, no solver, no production module) -- no solve can be issued from this file.

WHAT IT DERIVES (every formula is here, not only in prose):
  manifests      each run dir's manifest_sha256.json re-verified: every listed file's sha256 recomputed.
  q              phase-A Q and final Q (= arm_cost.gross_operational_cost, its source), Q = gross_operational_cost,
                 settlement EXCLUDED (the stage's own evaluate_common_q output, read, not recomputed).
  t_sum          the priced interface gap = recourse_components.interface_settlement_total (= t_TSO + sum_n t_DSO,n),
                 read at phase A and (if run) phase C; recomputed here as interface_settlement_tso + sum of
                 interface_settlement_dso for the check.
  solves         per phase: blocks, attempts (ledger attempts_recorded), retried blocks (attempts > 1), every record
                 succeeded, TSO records' termination conditions; guard final counts.
  feasibility    TSO: every TSO block's record in phase A (and C) succeeded with termination 'optimal'.
  reevaluation   per DSO block: trigger, and the items ABOVE their tolerance (hard vmag_sqr_band, hard
                 reference_generator_bound, hard no_reverse_flow; soft voltage excess; thermal), each with the bus
                 (model index i -> bus_i of data/SRP1/<case>/<case>_<year>.json nodes[i]), the hour (= period + 1),
                 the scenario and the amount; the no_reverse_flow excess in MW = excess_pu x DN baseMVA (from the case
                 file).
  curtailment    report_v3.curtailment_table rows, and DSO(arm) - DSO(coordinated) per convention (net, positive part)
                 and weighting (eur_at_1_block_weighted, mwh_day_weighted, mwh_rep_day_sum).
  tie_breaker    per entry (node, year, day, hour) |P_v - P_1| and |Q_v - Q_1| of the passive DSO interface schedule
                 at V in {0.1, 10} against the passive cold _r2 run (V = 1), and 0.1 against 10; the max and its
                 address; counts above 1e-3 MW / MVAr.
  walls          from each launch log: the harness's elapsed seconds at its exit line; the solve wall = sum of the
                 per_solve_record wall_s.

COMMAND (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
      p515_s53_w122_nrf_benchmark_extract.py > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w122_extract.log 2>&1
Output: data/SRP1/Results/P515S53/w116_benchmark_nrf/w122_checks/w122_nrf_extract.json (write-once).
"""

import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import gate_result_io as GRIO  # noqa: E402 -- stdlib only

ROOT = 'data/SRP1/Results/P515S53/w116_benchmark_nrf'
OUT = os.path.join(ROOT, 'w122_checks', 'w122_nrf_extract.json')
ARMS = [f'nrf_arm_{arm}_{start}_r2' for arm in ('passive', 'price_taker')
        for start in ('cold', 'warm_from_certified', 'perturbed')]
VARIANTS = ['nrf_passive_tie_breaker_0p1_r2', 'nrf_passive_tie_breaker_10_r2']
BASE_V1 = 'nrf_arm_passive_cold_r2'
REPORT = 'report_v3'
DN_CASE = {'5': 'case33_1', '7': 'case33_2', '9': 'case33_3'}   # data/SRP1/SRP1.json connection_node_id
SCHEDULE_TOL = 1e-3
INDEX_RE = re.compile(r'^([A-Za-z_]+)\[([^\]]*)\]$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


_CASES = {}


def case(node_id, year):
    key = (str(node_id), str(year))
    if key not in _CASES:
        name = DN_CASE[str(node_id)]
        _CASES[key] = load(os.path.join('data/SRP1', name, f'{name}_{year}.json'))
    return _CASES[key]


def parse_index(name):
    match = INDEX_RE.match(name)
    return match.group(1), [int(x) for x in match.group(2).split(',')]


def verify_manifest(run_dir):
    manifest_path = os.path.join(run_dir, 'manifest_sha256.json')
    manifest = load(manifest_path)
    bad = {path: {'listed': sha, 'now': sha256_file(path)} for path, sha in manifest.items()
           if sha256_file(path) != sha}
    return {'manifest': manifest_path, 'manifest_sha256': sha256_file(manifest_path), 'n_files': len(manifest),
            'all_match': not bad, 'mismatches': bad}


def launch_log_exit(run_id):
    path = os.path.join(ROOT, 'launch_logs', f'{run_id}.log')
    exit_line = None
    with open(path) as handle:
        for line in handle:
            if f'{run_id}: exit ' in line:
                exit_line = line.strip()
    match = re.search(r'\+\s*([\d.]+)s\] .*: exit (\d+); guard counts (\{.*\})', exit_line or '')
    return {'log': path, 'log_sha256': sha256_file(path), 'exit_line': exit_line,
            'elapsed_s_at_exit': float(match.group(1)) if match else None,
            'exit_code': int(match.group(2)) if match else None}


def solve_facts(run_dir, result):
    records = read_jsonl(os.path.join(run_dir, 'per_solve_record.jsonl'))
    ledger = result['solve_accounting']['ledger']
    phases = {}
    for row in ledger:
        entry = phases.setdefault(row['phase'], {'blocks': 0, 'attempts': 0, 'retried_blocks': {},
                                                 'all_succeeded': True})
        entry['blocks'] += 1
        entry['attempts'] += row['attempts_recorded']
        if row['attempts_recorded'] > 1:
            entry['retried_blocks'][row['block']] = row['attempt_tiers']
        entry['all_succeeded'] = entry['all_succeeded'] and row['succeeded'] is True
    tso = {}
    for rec in records:
        if rec['kind'] != 'TSO':
            continue
        entry = tso.setdefault(rec['phase'], {'n': 0, 'terminations': {}, 'all_succeeded_optimal': True,
                                              'blocks': []})
        entry['n'] += 1
        entry['blocks'].append(rec['block'])
        entry['terminations'][rec['termination_condition']] = entry['terminations'].get(
            rec['termination_condition'], 0) + 1
        entry['all_succeeded_optimal'] = (entry['all_succeeded_optimal'] and rec['succeeded'] is True
                                          and rec['termination_condition'] == 'optimal')
    return {'n_records': len(records), 'launches': sum(r['n_attempts'] for r in records),
            'solve_wall_s_sum': sum(r['wall_s'] for r in records), 'per_phase_ledger': phases,
            'tso_records_by_phase': tso, 'guard_final': result['solve_profile_guard']['final'],
            'accounting_observed_total': result['solve_accounting']['observed_total'],
            'accounting_entered_but_aborted_before_record': result['solve_accounting'][
                'entered_but_aborted_before_record']}


def t_sum(components):
    tso = components['interface_settlement_tso']
    dso = sum(components['interface_settlement_dso'].values())
    return {'interface_settlement_total': components['interface_settlement_total'],
            'recomputed_tso_plus_dso': tso + dso, 'interface_settlement_tso': tso,
            'interface_settlement_dso': components['interface_settlement_dso'],
            'interface_settlement_deviation_total': components['interface_settlement_deviation_total']}


def reevaluation_items(blocks):
    """Items above tolerance per block, with bus_i / hour / scenario."""
    out, summary = {}, {}
    for label, block in blocks.items():
        _kind, node_id, year, day = label.split('|')
        net = case(node_id, year)
        base = float(net['baseMVA'])
        v = block['violations']
        tol = v['tolerances']
        items = []
        for h in v['hard']:
            if h['excess_pu2'] <= tol['hard_tol_pu2']:
                continue
            if h['kind'] == 'no_reverse_flow':
                _n, (s_m, s_o, p) = parse_index(h['row'])
                items.append({'kind': 'no_reverse_flow', 'hour': p + 1, 's_m': s_m, 's_o': s_o,
                              'pg_adn_pu': h['pg_adn_pu'], 'excess_pu': h['excess_pu2'],
                              'excess_mw': h['excess_pu2'] * base})
            elif h['kind'] == 'vmag_sqr_band':
                _n, (i, s_m, s_o, p) = parse_index(h['var'])
                items.append({'kind': 'vmag_sqr_band', 'bus_index': i, 'bus_i': net['nodes'][i]['bus_i'],
                              'hour': p + 1, 's_m': s_m, 's_o': s_o, 'vmag_pu': h['vmag_sqr'] ** 0.5,
                              'bounds_pu2': h['bounds'], 'excess_pu2': h['excess_pu2']})
            else:
                items.append({'kind': h['kind'], 'var': h.get('var'), 'value_pu': h.get('value_pu'),
                              'bounds': h.get('bounds'), 'excess': h['excess_pu2']})
        for s in v['soft']:
            if s['soft_excess_pu2'] <= tol['soft_excess_tol_pu2']:
                continue
            family, (i, s_m, s_o, p) = parse_index(s['row'])
            node = net['nodes'][i]
            items.append({'kind': 'soft_' + family, 'bus_index': i, 'bus_i': node['bus_i'],
                          'vmax_pu': node['Vmax'], 'vmin_pu': node['Vmin'], 'hour': p + 1, 's_m': s_m, 's_o': s_o,
                          'violation_pu2': s['violation_pu2'], 'arm_own_slack_pu2': s['arm_own_slack_pu2'],
                          'soft_excess_pu2': s['soft_excess_pu2']})
        for t in v['thermal']:
            if t['violation_pu2'] > tol['thermal_tol_pu2']:
                items.append({'kind': 'thermal', 'row': t['row'], 'violation_pu2': t['violation_pu2']})
        out[label] = {'trigger': block['violations']['trigger_sequential_pass'], 'items_above_tol': items,
                      'interface_p_change_mw_max': block['interface_p_change_mw_max'],
                      'interface_q_change_mvar_max': block['interface_q_change_mvar_max'],
                      'solve': block['solve']}
        for it in items:
            s = summary.setdefault(it['kind'], {'n_items': 0, 'blocks': set(), 'buses': set(), 'hours': set(),
                                                'max': 0.0})
            s['n_items'] += 1
            s['blocks'].add(label)
            if 'bus_i' in it:
                s['buses'].add(f"{node_id}:{it['bus_i']}")
            if 'hour' in it:
                s['hours'].add(it['hour'])
            amount = it.get('excess_mw', it.get('soft_excess_pu2', it.get('excess_pu2', it.get('excess',
                                                                                                 it.get('violation_pu2')))))
            s['max'] = max(s['max'], amount)
    for s in summary.values():
        s['n_blocks'] = len(s['blocks'])
        s['blocks'] = sorted(s['blocks'])
        s['buses'] = sorted(s['buses'])
        s['hours'] = sorted(s['hours'])
    n_trigger = sum(1 for b in out.values() if b['trigger'])
    return out, {'n_blocks': len(out), 'n_blocks_triggering': n_trigger, 'by_kind': summary,
                 'max_unit': {'no_reverse_flow': 'MW', 'vmag_sqr_band': 'p.u.^2',
                              'soft_voltage_magnitude_upper_cons': 'p.u.^2 (excess over the arm own slack)',
                              'soft_voltage_magnitude_lower_cons': 'p.u.^2 (excess over the arm own slack)',
                              'reference_generator_bound': 'p.u.', 'thermal': 'p.u.^2'}}


def schedule_diff(a, b):
    worst = {'p_mw': (0.0, None), 'q_mvar': (0.0, None)}
    counts = {'p_mw': 0, 'q_mvar': 0}
    n = 0
    for node_id in a:
        for year in a[node_id]:
            for day in a[node_id][year]:
                for key in worst:
                    for p, (x, y) in enumerate(zip(a[node_id][year][day][key], b[node_id][year][day][key])):
                        d = abs(x - y)
                        if key == 'p_mw':
                            n += 1
                        if d > SCHEDULE_TOL:
                            counts[key] += 1
                        if d > worst[key][0]:
                            worst[key] = (d, {'node_id': node_id, 'year': year, 'day': day, 'hour': p + 1,
                                              'a': x, 'b': y})
    return {'n_entries': n, 'tol': SCHEDULE_TOL,
            'max_abs': {k: {'value': v[0], 'at': v[1]} for k, v in worst.items()},
            'n_entries_above_tol': counts}


def curtailment_rows(report):
    table = report['curtailment_table']
    coord = table['coordinated']['settled_cycle_181_signed_parts_from_models']
    rows = {}
    for run_id, row in table['arms_and_variants_phase_A'].items():
        parts = row['signed_parts']
        entry = {'row_phase': row['row_phase'], 'arm_cost_source': row['arm_cost_source']}
        for conv in ('net', 'positive_part', 'negative_part'):
            entry[conv] = {'total': parts[conv]['eur_at_1_block_weighted'],
                           'TSO': parts[conv]['by_agent']['TSO'], 'DSO': parts[conv]['by_agent']['DSO'],
                           'total_mwh_day_weighted': parts[conv]['mwh_day_weighted'],
                           'total_mwh_rep_day_sum': parts[conv]['mwh_rep_day_sum']}
        entry['dso_minus_coordinated_dso'] = {
            conv: {w: parts[conv]['by_agent']['DSO'][w] - coord[conv]['by_agent']['DSO'][w]
                   for w in ('eur_at_1_block_weighted', 'mwh_day_weighted', 'mwh_rep_day_sum')}
            for conv in ('net', 'positive_part')}
        rows[run_id] = entry
    return {'coordinated': coord, 'rows': rows,
            'weighting': table['weighting'], 'convention_status': table['convention_status']}


def main():
    if os.path.exists(OUT):
        print(f'refusing to overwrite {OUT}')
        return 2
    runs = {}
    for run_id in ARMS + VARIANTS + [REPORT]:
        run_dir = os.path.join(ROOT, run_id)
        result_path = os.path.join(run_dir, f'{run_id}.json')
        entry = {'result': {'path': result_path, 'sha256': sha256_file(result_path)},
                 'failure_json_present': os.path.exists(os.path.join(run_dir, 'failure.json')),
                 'manifest_check': verify_manifest(run_dir), 'launch': launch_log_exit(run_id)}
        runs[run_id] = entry
    out = {'task': 'Planner task W122 (spec v5 bca69f97): zero-solve extraction', 'script': os.path.basename(__file__),
           'script_sha256': sha256_file(__file__),
           'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED; salvage separate (0 at x = 0)',
           'instance': None, 'runs': runs, 'per_run': {}}
    for run_id in ARMS + VARIANTS:
        run_dir = os.path.join(ROOT, run_id)
        result = load(os.path.join(run_dir, f'{run_id}.json'))
        if out['instance'] is None:
            out['instance'] = result.get('instance')
        pa = result['phase_A']
        nrf = pa['no_reverse_flow']['dso_reverse_flow_count']
        row = {'arm': result['arm'], 'start': result['start'], 'decision_tie_breaker': result['decision_tie_breaker'],
               'q_phase_A': pa['evaluation']['gross_operational_cost'],
               'q_final': result['arm_cost']['gross_operational_cost'], 'q_final_source': result['arm_cost']['source'],
               't_sum_phase_A': t_sum(pa['evaluation']['recourse_components']),
               'terminal_salvage_value_phase_A': pa['evaluation']['recourse_components']['terminal_salvage_value'],
               'phase_A_nrf_held': {'totals': nrf['totals'], 'min_entry': nrf['min_entry'],
                                    'n_entries_at_zero_within_material_tol': nrf[
                                        'n_entries_at_zero_within_material_tol'],
                                    'any_reverse_flow_material': nrf['any_reverse_flow_material']},
               'tso_vs_dso_schedule_max_abs': pa['tso_vs_dso_schedule_max_abs'],
               'dso_schedule_vs_coordinated_max_abs': pa['dso_schedule_vs_coordinated_max_abs'],
               'solves': solve_facts(run_dir, result)}
        b = result.get('phase_B_consistency')
        if b is not None:
            per_block, summary = reevaluation_items(b['reevaluation']['blocks'])
            row['phase_B'] = {'max_abs_dv_dn_pu': b['max_abs_dv_dn_pu'],
                              'trigger': b['reevaluation']['trigger_sequential_pass'],
                              'nrf_violations_at_actual_voltage': b['no_reverse_flow_violations_at_actual_voltage'],
                              'summary_above_tol': summary, 'per_block': per_block}
        c = result.get('phase_C_sequential_pass')
        if c is not None:
            nc = c['no_reverse_flow']['dso_reverse_flow_count']
            row['phase_C'] = {'q': c['evaluation']['gross_operational_cost'], 'effect_on_q_eur': c['effect_on_q_eur'],
                              't_sum': t_sum(c['evaluation']['recourse_components']),
                              'max_abs_dv_dn_pu_after_pass': c['max_abs_dv_dn_pu_after_pass'],
                              'dso_voltage_shift_max_pu': c['dso_voltage_shift_max_pu'],
                              'tso_target_move_max': c['tso_target_move_max'],
                              'nrf_held_after_pass': {'totals': nc['totals'], 'min_entry': nc['min_entry'],
                                                      'any_reverse_flow_material': nc['any_reverse_flow_material']}}
        out['per_run'][run_id] = row
    report = load(os.path.join(ROOT, REPORT, f'{REPORT}.json'))
    out['curtailment'] = curtailment_rows(report)
    base = load(os.path.join(ROOT, BASE_V1, f'{BASE_V1}.json'))['phase_A']['dso_interface_schedule']
    sched = {v: load(os.path.join(ROOT, v, f'{v}.json'))['phase_A']['dso_interface_schedule'] for v in VARIANTS}
    out['tie_breaker_schedules'] = {
        'base_v1': BASE_V1,
        '0p1_vs_1': schedule_diff(sched['nrf_passive_tie_breaker_0p1_r2'], base),
        '10_vs_1': schedule_diff(sched['nrf_passive_tie_breaker_10_r2'], base),
        '0p1_vs_10': schedule_diff(sched['nrf_passive_tie_breaker_0p1_r2'], sched['nrf_passive_tie_breaker_10_r2'])}
    out['report_v3'] = {'claim': report['claim'], 'per_arm_nrf': report['per_arm_nrf'],
                        'predictions_scored_w119': report['predictions_scored'].get('planner_predictions_w119'),
                        'coordinated_reverse_flow_totals': report['coordinated_reverse_flow_count']['totals'],
                        'coordinated_reverse_flow_per_node': report['coordinated_reverse_flow_count']['per_node'],
                        'report_capture_check': report['report_capture_check'],
                        'solve_profile_guard': report['solve_profile_guard']}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'x') as handle:
        GRIO.dump(out, handle, indent=1, sort_keys=True)
    print(f'wrote {OUT} sha256 {sha256_file(OUT)}')
    bad = [k for k, v in runs.items() if not v['manifest_check']['all_match']]
    print('manifests all match:', not bad, bad)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
