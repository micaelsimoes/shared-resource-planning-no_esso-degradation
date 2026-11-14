"""
Stage P5.10-C -- what is the oracle's endpoint: the ADMM output, or the polish?

Two definitions, evaluated on the same run so the comparison is exact:

    Oracle A   the rescaled ADMM output, taken directly. No polish.
    Oracle B   the rescaled ADMM output followed by the exact-consensus polish.

THE COMPARISON IS NOT SYMMETRIC AND THE REPORT MUST NOT PRETEND IT IS.  Oracle B
returns ONE operating point: the polish pins every agent to a common interface
value, so TSO and DSO agree exactly and the coordinated residual is ~1e-16.
Oracle A returns a set of per-agent solutions that DISAGREE at the interface by
the ADMM's terminal primal residual.  Each agent's own SMOPF is feasible -- IPOPT
solved it -- but there is no single physical operating point until someone
reconciles them.  So the question is not merely "which objective is lower".  It
is whether Oracle A's disagreement is small enough to call the result a solution.

Measured for each candidate:

  * objective, Oracle A (pre-polish net recourse) and Oracle B (polished);
  * the interface disagreement Oracle A leaves, in p.u. and in MW;
  * per-agent nonlinear feasibility of the UNPOLISHED models, using production's
    own `audit_networks`;
  * Oracle B's coordinated residual, ESSO feasibility and network violation;
  * repeatability: each candidate evaluated twice under the same declared
    configuration, after different call histories;
  * cost: ADMM runtime against polish runtime;
  * ranking under each definition.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p510_c_endpoint.py <rho_pf>
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p510_oracle as OR  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

CANDIDATES = ['base', 'se|node5|2025|-10%', 'se|node9|2025|-10%']
S_BASE_MVA = 100.0          # per-unit base, for reporting disagreement in MW


def summarise(record, repeat_record=None):
    cycles = record.get('cycle_detail') or []
    last = cycles[-1] if cycles else {}
    admm = record.get('admm') or {}
    unpolished = record.get('unpolished_network_audit') or {}
    residuals = record.get('coordination_residuals') or {}
    audit = record.get('network_audit') or {}
    esso = record.get('esso_audit') or {}
    out = {
        'status': record.get('status'),
        # ---- Oracle A
        'A_objective_net_recourse': record.get('admm_net_recourse_before_polish'),
        'A_base_objective': record.get('admm_base_objective_before_polish'),
        'A_interface_disagreement_pu': last.get('primal_pf'),
        'A_interface_disagreement_MW': (
            last.get('worst_pf_primal_difference')),
        'A_voltage_disagreement_pu': last.get('primal_v'),
        'A_unpolished_max_network_violation': unpolished.get('max_violation'),
        'A_unpolished_worst_block': unpolished.get('worst_where'),
        'A_unpolished_h1_violation': unpolished.get(
            'max_h1_complementarity_violation'),
        'A_runtime_s': admm.get('runtime_s'),
        'A_cycles': admm.get('cycles'),
        # ---- Oracle B
        'B_objective_total': record.get('total_objective'),
        'B_net_recourse': record.get('net_operational_recourse'),
        'B_max_coordinated_residual': residuals.get('max_coordinated'),
        'B_max_network_violation': audit.get('max_violation'),
        'B_esso_production_feasible': esso.get('production_feasible'),
        'B_failed_blocks': record.get('failed_blocks'),
        'B_runtime_s': record.get('polish_runtime_s'),
        # ---- the polish effect
        'polish_effect_on_recourse': record.get('admm_to_polish_improvement'),
    }
    if repeat_record is not None:
        out['repeat_A_objective'] = repeat_record.get(
            'admm_net_recourse_before_polish')
        out['repeat_B_objective'] = repeat_record.get('total_objective')
        out['A_bit_identical'] = (
            out['repeat_A_objective'] == out['A_objective_net_recourse'])
        out['B_bit_identical'] = (
            out['repeat_B_objective'] == out['B_objective_total'])
    return out


def main():
    os.makedirs(OR.OUT_DIR, exist_ok=True)
    rho_pf = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    out_path = os.path.join(OR.OUT_DIR, f'p510_c_endpoint_pf{rho_pf:g}.json')
    try:
        provenance, planning_gate = gate('P5.10-C oracle endpoint', OR.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.10] ABORTED\n{error}')
        sys.exit(1)

    config = OR.OracleConfig(
        scaling_mode=OR.SCALING_RESCALED, rho_v=1.5, rho_pf=rho_pf,
        rho_ess=1.0, adaptive_penalty=False, neutralize_history=True,
        notes='P5.10-C endpoint comparison')

    population = dict(BC.population(planning_gate))
    report = {'stage': 'P5.10-C', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'config': config.as_dict(), 'config_hash': config.config_hash,
              'per_unit_base_MVA': S_BASE_MVA,
              'asymmetry_note': (
                  'Oracle B returns one operating point; Oracle A returns '
                  'per-agent solutions that disagree at the interface by the '
                  'terminal primal residual. Each agent is individually '
                  'feasible; there is no single physical point until they are '
                  'reconciled.'),
              'candidates': {}}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for name in CANDIDATES:
        safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
        print(f'[P5.10-C] {name} ...', flush=True)
        record, _ = OR.evaluate(population[name], config,
                                case_id=f'c_{safe}', audit_unpolished=True)
        print(f'[P5.10-C] {name} (repeat, different call history) ...', flush=True)
        repeat, _ = OR.evaluate(population[name], config,
                                case_id=f'c_{safe}_repeat', audit_unpolished=True)
        entry = summarise(record, repeat)
        report['candidates'][name] = entry
        persist()
        print(f"          A: {entry['A_objective_net_recourse']}  "
              f"disagreement {entry['A_interface_disagreement_pu']} p.u.  "
              f"unpolished violation {entry['A_unpolished_max_network_violation']}")
        print(f"          B: {entry['B_objective_total']}  "
              f"coordinated residual {entry['B_max_coordinated_residual']}  "
              f"failed {len(entry['B_failed_blocks'] or [])}")
        print(f"          repeatable: A={entry.get('A_bit_identical')} "
              f"B={entry.get('B_bit_identical')}", flush=True)

    # ---- ranking under each definition -------------------------------------
    for key, field in (('ranking_oracle_A', 'A_objective_net_recourse'),
                       ('ranking_oracle_B', 'B_objective_total')):
        values = {n: e[field] for n, e in report['candidates'].items()
                  if e.get(field) is not None}
        if len(values) >= 2:
            base = values.get('base')
            report[key] = {
                'values': values,
                'deltas_from_base': ({n: v - base for n, v in values.items()}
                                     if base is not None else None),
                'order': [n for n, _ in sorted(values.items(), key=lambda kv: kv[1])]}
    persist()
    print(f'\n[P5.10-C] report -> {out_path}')


if __name__ == '__main__':
    main()
