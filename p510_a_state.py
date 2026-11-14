"""
Stage P5.10-A -- make the oracle state explicit, and measure what was implicit.

Four experiments, in order:

  A0  reproduction gates.  The explicit-configuration oracle, told to reproduce
      production's implicit configuration, must return the accepted values --
      CURRENT at the template's own rho reproduces P5.6-D generation 1, and
      RESCALED at the same rho reproduces P5.8-C generation 1.  If it does not,
      the abstraction is not faithful and nothing after it is trustworthy.

  A1  the inheritance channels, read directly off the frozen template.

  A2  the base-candidate asymmetry.  `objective_convergence` is only computed
      when `previous_recourse is not None`, and that is inherited only when the
      evaluated candidate equals the template's candidate.  T0's candidate is
      the base.  So the base is evaluated under a different stopping rule from
      every other candidate -- inside the quantity the planning problem ranks,
      `Delta(x) = Q(x) - Q(base)`.  Measured with history inherited and again
      with it neutralised.  Run under CURRENT because the asymmetry is a
      property of the warm-start logic, not of the scaling, and CURRENT is
      cheaper.

  A3  purity.  The same candidate under the same declared configuration, after
      a different call history, must be bit-identical.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p510_a_state.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p59_rho as RH  # noqa: E402
import p510_oracle as OR  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(OR.OUT_DIR, 'p510_a_state.json')

P56D_GEN1 = 828021090.360850          # accepted P5.6-D base chain, generation 1
P58C_GEN1_PREPOLISH = 825814074.49    # accepted P5.8-C rescaled, generation 1
GATE_TOL = 1.0

INHERITED = dict(rho_v=1.5, rho_pf=2.25, rho_ess=1.0)


def main():
    os.makedirs(OR.OUT_DIR, exist_ok=True)
    os.makedirs(OR.ARCHIVE_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.10-A explicit state', OR.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.10] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    x0 = population['base']
    x5 = population['se|node5|2025|-10%']

    report = {'stage': 'P5.10-A', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'inheritance_channels': {}, 'gates': {}, 'A2': [], 'A3': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    # ---------------- A1: read the channels off the template ----------------
    import pickle
    import p57_eval as E7
    with open(E7.T0_CACHE, 'rb') as handle:
        raw = pickle.load(handle)
    observed = RH.observed_rho(raw)
    distinct = {tuple(sorted(v.items())) for v in observed.values()}
    report['inheritance_channels'] = {
        '1_rho_in_template': {
            'blocks': len(observed),
            'distinct_values': [dict(t) for t in distinct],
            'parameter_file_says': {'v': 1.0, 'pf': 1.0, 'ess': 1.0}},
        '2_objective_scaling_in_template': (
            'carried by the cloned models; CURRENT vs RESCALED is a property '
            'of the template, not of a flag'),
        '3_consecutive_converged_cycles': raw.get('consecutive_converged_cycles'),
        '4_last_recourse': raw.get('last_recourse'),
        '4_candidate_solution_is_base': raw.get('candidate_solution') is not None,
        'note': ('channel 4 is inherited only when the evaluated candidate '
                 'equals the template candidate, which is the base; see A2')}
    persist()
    print('[P5.10-A] inheritance channels on the frozen T0 template:')
    print(f"   rho in template      : {[dict(t) for t in distinct]}  "
          f"(parameter file says 1.0/1.0/1.0)")
    print(f"   consecutive counters : {raw.get('consecutive_converged_cycles')}")
    print(f"   last_recourse        : {raw.get('last_recourse')}")
    print(f"   candidate_solution   : "
          f"{'present (the base candidate)' if raw.get('candidate_solution') else 'absent'}\n")

    # ---------------- A0: reproduction gates --------------------------------
    gates = [
        ('gate_CURRENT', OR.OracleConfig(
            scaling_mode=OR.SCALING_CURRENT, adaptive_penalty=True,
            neutralize_history=False, notes='production implicit configuration',
            **INHERITED), 'total_objective', P56D_GEN1),
        ('gate_RESCALED', OR.OracleConfig(
            scaling_mode=OR.SCALING_RESCALED, adaptive_penalty=True,
            neutralize_history=False, notes='P5.8-C generation 1',
            **INHERITED), 'admm_net_recourse_before_polish', P58C_GEN1_PREPOLISH),
    ]
    for name, config, field, expected in gates:
        print(f'[P5.10-A] {name}: {config.label} ...', flush=True)
        record, _ = OR.evaluate(x0, config, case_id=name)
        observed_value = record.get(field)
        delta = (observed_value - expected) if observed_value is not None else None
        report['gates'][name] = {
            'config': config.as_dict(), 'field': field, 'expected': expected,
            'observed': observed_value, 'delta': delta,
            'passed': delta is not None and abs(delta) <= GATE_TOL,
            'admm_cycles': (record.get('admm') or {}).get('cycles'),
            'status': record.get('status')}
        persist()
        print(f"           {field} = {observed_value}  expected {expected}  "
              f"delta {delta}", flush=True)
        if not report['gates'][name]['passed']:
            report['aborted'] = f'{name} did not reproduce the accepted value'
            persist()
            print(f'\n[P5.10-A] REPRODUCTION GATE FAILED. Not continuing.')
            sys.exit(1)
        print('           GATE PASSED\n', flush=True)

    # ---------------- A2: the base-candidate asymmetry ----------------------
    print('[P5.10-A] A2 -- stopping rule, base versus non-base candidate\n',
          flush=True)
    for neutralize in (False, True):
        for label, x in (('base', x0), ('se|node5|2025|-10%', x5)):
            config = OR.OracleConfig(
                scaling_mode=OR.SCALING_CURRENT, adaptive_penalty=True,
                neutralize_history=neutralize, **INHERITED)
            case_id = f'a2_{"neutral" if neutralize else "inherited"}_{label[:12]}'
            record, _ = OR.evaluate(x, config, case_id=case_id.replace('|', '_'))
            cycles = record.get('cycle_detail') or []
            first = cycles[0] if cycles else {}
            row = {
                'neutralize_history': neutralize, 'candidate': label,
                'admm_cycles': (record.get('admm') or {}).get('cycles'),
                'cycle1_objective_change_abs': first.get('objective_change_abs'),
                'cycle1_objective_convergence': first.get('objective_convergence'),
                'cycle1_residual_convergence': first.get('residual_convergence'),
                'cycle1_cycle_convergence': first.get('cycle_convergence'),
                'total_objective': record.get('total_objective'),
                'status': record.get('status')}
            report['A2'].append(row)
            persist()
            print(f"   neutralize={str(neutralize):5s} {label:20s} "
                  f"cycles={row['admm_cycles']}  "
                  f"cycle1 objective_change={row['cycle1_objective_change_abs']}  "
                  f"objective_converged={row['cycle1_objective_convergence']}",
                  flush=True)

    # ---------------- A3: purity under the explicit configuration -----------
    print('\n[P5.10-A] A3 -- purity: same configuration, different call history',
          flush=True)
    config = OR.OracleConfig(scaling_mode=OR.SCALING_CURRENT,
                             adaptive_penalty=True, neutralize_history=True,
                             **INHERITED)
    record, _ = OR.evaluate(x0, config, case_id='a3_repeat')
    first_run = next((r for r in report['A2']
                      if r['neutralize_history'] and r['candidate'] == 'base'), {})
    report['A3'] = {
        'config_hash': config.config_hash,
        'first_total_objective': first_run.get('total_objective'),
        'repeat_total_objective': record.get('total_objective'),
        'difference': ((record.get('total_objective') or 0)
                       - (first_run.get('total_objective') or 0)),
        'bit_identical': record.get('total_objective') == first_run.get('total_objective')}
    persist()
    print(f"   first  = {report['A3']['first_total_objective']}")
    print(f"   repeat = {report['A3']['repeat_total_objective']}")
    print(f"   bit-identical = {report['A3']['bit_identical']}")
    print(f'\n[P5.10-A] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
