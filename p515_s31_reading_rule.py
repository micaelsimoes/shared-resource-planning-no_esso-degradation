"""P5.15 Step 3.1 reading rule (signed table §5, Addendum 10). Zero solves.

Compares the post-signature baseline campaign (data/SRP1/Results/P515S31_run) with the Step 3.0 baseline
(data/SRP1/Results/P515S30, bitwise identical to P515G1B). Reports:
  1. whether Q(x) moved: Δ = gross_S31 − recourse_S30 with the rule-nine bar (sum of the two terminal steps);
     the bar is valid only if S31 converged and rule ten < 1 (otherwise INDETERMINATE);
  2. which rows account for it, at the S31 terminal point:
       definitional effect  = Σ values the removed/zeroed terms would take there under pre-signature weights
                              (row 3 ADN-interface flexibility charge, row 5 DSO RES curtailment at weight 1,
                               row 9 bilinear complementarity, row 14 orphan slacks);
       Q_old_at_S31_point   = gross_S31 + definitional effect;
       path effect          = recourse_S30 − Q_old_at_S31_point   (the optimum moved because the objective changed);
     so  recourse_S30 − gross_S31 = definitional effect + path effect, exactly;
  3. the terminal value of every D row, and the reported Q(x) under the signed D semantics
     (economic_recourse_all_D_excluded) and with only row 12 excluded (economic_recourse_voltage_excluded);
  4. the ESSO D3 violation (aggregate slacks).
Writes data/SRP1/Results/P515S31_run/s31_reading_rule.json.
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
S30 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S30')
S31 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31_run')
DEFINITIONAL = {
    'row_3_tso_adn_interface_flexibility_charge': 'flexibility_cost_tso_adn_interface_definitional',
    'row_5_dso_res_curtailment_at_weight_1': 'res_curtailment_definitional_at_weight_1',
    'row_9_bilinear_ess_complementarity': 'ess_complementarity_bilinear_definitional',
    'row_14_orphan_flex_q_day_balance': 'orphan_flex_q_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional',
    'row_14_orphan_tso_adn_flex_p_day_balance': 'orphan_tso_adn_flex_p_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional',
}
D_ROWS = {
    'row_10_local_ess_day_balance': 'local_ess_day_balance_slack',
    'row_11_shared_ess_day_balance': 'shared_ess_day_balance_slack',
    'row_12_voltage': 'voltage_slack',
    'row_13_flexibility_p_day_balance': 'flexibility_p_day_balance_slack',
    'row_15_node_balance': 'node_balance_slack',
    'row_16_branch_flow': 'branch_flow_slack',
}


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected one file for {pattern}, found {hits}')
    return hits[0]


def main():
    s30 = json.load(open(_one(os.path.join(S30, 'g_*.json'))))
    s31 = json.load(open(_one(os.path.join(S31, 'g_*.json'))))
    levels = json.load(open(os.path.join(S31, 'component_levels_terminal.json')))
    totals = levels['totals_weighted']
    rc = levels['recourse_components']

    converged = s31.get('converged_at_cycle') is not None
    rule_ten = s31.get('rule_ten_terminal_step_over_threshold')
    valid = converged and rule_ten is not None and rule_ten < 1.0
    gross_s31 = rc['gross_operational_cost']
    recourse_s30 = s30['recourse']
    delta = gross_s31 - recourse_s30
    s31_step = s31.get('terminal_objective_change_abs')
    if s31_step is None and s31.get('cycle_trajectory'):
        steps = [r.get('objective_change_abs') for r in s31['cycle_trajectory'] if r.get('objective_change_abs') is not None]
        s31_step = steps[-1] if steps else None
    bar = (s30['terminal_objective_change_abs'] + s31_step) if s31_step is not None else None

    definitional = {row: totals.get(key) for row, key in DEFINITIONAL.items()}
    missing = [row for row, value in definitional.items() if value is None]
    definitional_total = sum(v for v in definitional.values() if v is not None)
    q_old_at_s31 = gross_s31 + definitional_total
    path_effect = recourse_s30 - q_old_at_s31

    out = {
        'stage': 'P5.15 Step 3.1 reading rule (signed table section 5)',
        'baseline_step_3_0': {'recourse': recourse_s30, 'terminal_step': s30['terminal_objective_change_abs'],
                              'cycles': s30['cycles_run']},
        'post_signature_s31': {'gross_operational_cost': gross_s31, 'net_operational_recourse': rc['net_operational_recourse'],
                               'cycles_run': s31['cycles_run'], 'converged_at_cycle': s31.get('converged_at_cycle'),
                               'rule_ten': rule_ten, 'terminal_step': s31_step,
                               'local_solve_failures': s31.get('local_solve_failures'),
                               'levels_cycles_run': levels.get('cycles_run')},
        'q_moved': {'delta_gross_s31_minus_s30': delta, 'relative_to_s30': delta / recourse_s30,
                    'rule_nine_bar': bar, 'delta_over_bar': (delta / bar) if bar else None,
                    'bar_valid': valid,
                    'verdict': ('INDETERMINATE (S31 not converged or not settled)' if not valid
                                else ('MOVED beyond stopping slack' if abs(delta) > bar else 'NOT DISTINGUISHABLE from stopping slack'))},
        'attribution': {'definitional_by_row': definitional, 'definitional_rows_missing': missing,
                        'definitional_total': definitional_total,
                        'q_old_definition_at_s31_point': q_old_at_s31,
                        'path_effect': path_effect,
                        'identity_check_s30_minus_s31_equals_def_plus_path': (recourse_s30 - gross_s31) - (definitional_total + path_effect)},
        'd_rows_terminal_weighted': {row: rc['detector_components'].get(key) for row, key in D_ROWS.items()},
        'detector_penalty_total': rc['detector_penalty_total'],
        'reported_q': {'economic_recourse_all_D_excluded': rc['economic_recourse_all_D_excluded'],
                       'economic_recourse_voltage_excluded': rc['economic_recourse_voltage_excluded']},
        'esso_feasibility_violation_D3': levels.get('esso_feasibility_violation_D3'),
        'economic_components_weighted': {k: totals.get(k) for k in ('generation_cost', 'flexibility_cost_internal',
                                                                     'load_curtailment_cost', 'res_curtailment_penalty',
                                                                     'ess_usage_cost')},
    }
    path = os.path.join(S31, 's31_reading_rule.json')
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
