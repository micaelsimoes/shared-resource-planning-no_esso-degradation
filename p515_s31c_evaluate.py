"""P5.15 row 3-prime economic baseline (Addendum 12) - gate evaluation. Zero solves.

Reads the s31c campaign artifacts and reports, in the order the report leads with:
  1. the cancellation residual at the terminal point: T_TSO + sum T_DSO, the price-weighted interface consensus
     residual sum_dso sum pi*baseMVA*(p_TSO - p_DSO), and their closure (the settlement sum should equal minus the
     priced residual exactly), with the settlement magnitudes for scale;
  2. cycles, convergence, rule ten, local-solve failures, network failures by class;
  3. the system-cost recourse (settlements excluded), the value including settlements, the D rows and the economic
     recourse fields;
  4. per DSO: the interface settlement sum_t pi_t * p_int,t, the flexibility volumes delta_P / delta_Q, and the
     maximum absolute interface consensus residual.
Usage: python p515_s31c_evaluate.py [RUN_DIR] [--dry-run]
Without --dry-run it writes RUN_DIR/s31c_evaluation.json (refuses to overwrite).
"""
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected one file for {pattern}, found {hits}')
    return hits[0]


def _max_abs_residual_mw(periods):
    best = 0.0
    fields_seen = set()
    for row in periods.values():
        if isinstance(row, dict):
            for key, value in row.items():
                if 'residual' in key.lower() and 'mw' in key.lower() and isinstance(value, (int, float)):
                    fields_seen.add(key)
                    best = max(best, abs(value))
    return best, sorted(fields_seen)


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run_dir = os.path.abspath(args[0]) if args else os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C_run')
    g = json.load(open(_one(os.path.join(run_dir, 'g_*.json'))))
    levels = json.load(open(os.path.join(run_dir, 'component_levels_terminal.json')))
    detail = json.load(open(_one(os.path.join(run_dir, 'interface_settlement_detail_*.json'))))
    rc = levels['recourse_components']

    t_tso = detail['t_tso_total']
    t_dso = detail['t_dso_by_node']
    t_sum = detail['t_tso_plus_t_dso_terminal']
    residual = detail['interface_consensus_residual_per_dso']
    priced_residual = sum(v['sum_pi_baseMVA_residual_weighted'] for v in residual.values())
    scale = abs(t_tso) + sum(abs(v) for v in t_dso.values())

    failures_path = glob.glob(os.path.join(run_dir, 'network_failures_*.jsonl'))
    classes = collections.Counter()
    if failures_path:
        with open(failures_path[0]) as handle:
            for line in handle:
                if line.strip():
                    classes[json.loads(line).get('primary_termination')] += 1

    per_dso = {}
    for node, rows in residual.items():
        max_mw, fields = _max_abs_residual_mw(rows.get('periods', {}))
        per_dso[node] = {
            'interface_settlement_weighted': rows['dso_settlement_sum_pi_p_int_weighted'],
            'priced_consensus_residual_weighted': rows['sum_pi_baseMVA_residual_weighted'],
            'max_abs_interface_consensus_residual_mw': max_mw,
            'residual_fields_used': fields,
            'flexibility_volumes': detail['flexibility_volumes_per_dso'].get(node),
        }

    out = {
        'stage': 'P5.15 row 3-prime economic baseline - gate evaluation',
        'run_dir': os.path.relpath(run_dir, REPO),
        '1_cancellation': {
            't_tso': t_tso, 't_dso_by_node': t_dso, 't_tso_plus_t_dso_terminal': t_sum,
            'priced_consensus_residual_total': priced_residual,
            'closure_t_sum_plus_priced_residual': t_sum + priced_residual,
            'settlement_magnitude_scale': scale,
            't_sum_relative_to_settlement_scale': (t_sum / scale) if scale else None,
            't_sum_relative_to_system_cost': (t_sum / rc['gross_operational_cost']) if rc['gross_operational_cost'] else None,
            # P5.15 Addendum 38 (C): with the TSO pinned the settlement no longer cancels
            # to the priced consensus residual alone -- its residual IS the price-deviation
            # covariance. Addendum 13's cancellation gate therefore becomes
            #     T_TSO + sum_DSO T_DSO  ==  covariance term   (by construction),
            # and BOTH sides are reported, together with what is left over
            # (`identity_residual_t_sum_minus_covariance`), which is the same priced
            # consensus residual the old gate measured. All four keys are `.get`: an
            # artifact written before Addendum 38 reports them as None, and at one market
            # scenario the covariance is identically zero, so this row reproduces the old
            # cancellation statement exactly.
            'contracted_total': rc.get('interface_settlement_contracted_total'),
            'covariance_term_total': rc.get('interface_settlement_covariance_total'),
            'identity_residual_t_sum_minus_covariance': rc.get('interface_settlement_identity_residual'),
            'closure_identity_residual_plus_priced_residual': (
                (rc['interface_settlement_identity_residual'] + priced_residual)
                if rc.get('interface_settlement_identity_residual') is not None else None),
            'voltage_pin_total_excluded_from_Qx': rc.get('voltage_pin_total'),
        },
        '2_convergence': {
            'cycles_run': g.get('cycles_run'), 'converged_at_cycle': g.get('converged_at_cycle'),
            'rule_ten': g.get('rule_ten_terminal_step_over_threshold'),
            'terminal_objective_step': g.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': g.get('terminal_objective_tolerance'),
            'local_solve_failures': g.get('local_solve_failures'),
            'network_failures_summary': g.get('network_failures_summary'),
            'network_failure_records_by_termination': dict(classes),
            'wall_clock_s': g.get('wall_clock_s'),
            'solves': g.get('solve_profile', {}).get('observed', {}).get('permitted_solve'),
            'shared_frozen_smopf_modified': g.get('shared_frozen_smopf_modified'),
        },
        '3_recourse': {
            'system_cost_recourse_gross': rc['gross_operational_cost'],
            'net_operational_recourse': rc['net_operational_recourse'],
            'gross_including_settlement': rc['gross_operational_cost_including_settlement'],
            'detector_penalty_total': rc['detector_penalty_total'],
            'detector_components': rc['detector_components'],
            'economic_recourse_all_D_excluded': rc['economic_recourse_all_D_excluded'],
            'economic_recourse_voltage_excluded': rc['economic_recourse_voltage_excluded'],
            'economic_components_weighted': {k: levels['totals_weighted'].get(k) for k in (
                'generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost',
                'res_curtailment_penalty', 'ess_usage_cost')},
        },
        '4_per_dso': per_dso,
        'esso_feasibility_violation_D3': levels.get('esso_feasibility_violation_D3'),
    }
    print(json.dumps(out, indent=1, default=str))
    if not dry:
        path = os.path.join(run_dir, 's31c_evaluation.json')
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite {path}')
        with open(path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
