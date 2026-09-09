"""
Stage P5.9-A -- rescaled ADMM plus penalty calibration.

Every case runs ONE generation from the same frozen T0 primal state under the
RESCALED objective, differing only in rho.  Production's IPOPT options, ADMM
tolerances and stopping criteria are untouched, and `adaptive_penalty` keeps its
production value (true) so that what is swept is the INITIAL penalty of the real
oracle, not a hypothetical fixed-rho one.  Stage B then separates the initial
value from the adaptation.

THE BASELINE IS NOT THE PARAMETER FILE.  `data/SRP1/SRP1_params.json` sets
rho_v = rho_pf = rho_ess = 1.0, but the frozen T0 template carries
rho_v = 1.5, rho_pf = 2.25, rho_ess = 1.0 -- the values production's own
adaptive rule left at the end of T0's cold build (1.5^1 and 1.5^2).  Production's
warm-start path clones `initial_state['models']` and never rebuilds the
augmented objectives, so every warm evaluation from P5.6-B onward inherits those
penalties and the parameter file's rho is dead on that path.  This is the same
inherited-state channel P5.8-A0 documented for `consecutive_converged_cycles`
(`shared_resources_planning.py:2093-2094`).

A0 is therefore a reproduction gate run with the template's rho left EXACTLY as
inherited: RESCALED at the oracle's real recurring penalties must return
P5.8-C's generation 1.  If it does not, nothing below is trustworthy and the
stage aborts.  A0b pins rho to the parameter-file values instead, as a
documented control that measures what the inherited penalties are worth.

Sweep values are ABSOLUTE, as specified.  Each case therefore also reports its
ratio to the inherited baseline, which is what it is actually a multiple of.

THE SELECTION RULE IS DECLARED HERE, BEFORE THE RESULTS EXIST, so that the
choice of "best" cannot be fitted to them afterwards:

    1. prefer zero failed polish blocks;
    2. among those, the lowest POLISHED total objective;
    3. if no case achieves zero, the fewest failed blocks, tie-broken by the
       lowest PRE-POLISH net recourse.

The goal is not the lowest objective alone.  It is a high-quality economic
solution WITH acceptable consensus AND a successful exact-consensus polish.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_a_sweep.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_a_sweep.json')

# P5.8-C generation 1, RESCALED at production rho -- the reproduction gate
P58C_GEN1_PREPOLISH = 825814074.49
P58C_GEN1_POLISH_FAILED = ['DSO5|2025|Spring', 'DSO5|2030|Summer']
GATE_TOLERANCE = 1.0            # currency units on 8.3e8

# P5.8-C generation 1, CURRENT formulation, for reference only
CURRENT_GEN1_PREPOLISH = 837188510.90
CURRENT_GEN1_POLISHED = 828021090.360850

INHERITED_RHO = {'v': 1.5, 'pf': 2.25, 'ess': 1.0}

# rho=None means "leave the template's inherited penalties untouched"
CASES = [
    ('A0_inherited_rho',      None),
    ('A0b_paramfile_rho1',    {'v': 1.0,    'pf': 1.0}),
    ('A1_rho_v_10',           {'v': 10.0,   'pf': 2.25}),
    ('A1_rho_v_100',          {'v': 100.0,  'pf': 2.25}),
    ('A1_rho_v_1000',         {'v': 1000.0, 'pf': 2.25}),
    ('A2_rho_pf_10',          {'v': 1.5,    'pf': 10.0}),
    ('A2_rho_pf_100',         {'v': 1.5,    'pf': 100.0}),
    ('A2_rho_pf_1000',        {'v': 1.5,    'pf': 1000.0}),
    ('A3_rho_v_pf_10',        {'v': 10.0,   'pf': 10.0}),
    ('A3_rho_v_pf_100',       {'v': 100.0,  'pf': 100.0}),
    ('A3_rho_v_pf_1000',      {'v': 1000.0, 'pf': 1000.0}),
]


def select_best(rows):
    """The declared selection rule, applied mechanically."""
    usable = [r for r in rows if r.get('admm_cycles') is not None]
    clean = [r for r in usable if r.get('n_failed_polish_blocks') == 0
             and r.get('total_objective') is not None]
    if clean:
        best = min(clean, key=lambda r: r['total_objective'])
        return best, 'zero failed polish blocks; lowest polished total objective'
    if not usable:
        return None, 'no case produced a usable ADMM result'
    best = min(usable, key=lambda r: (
        r.get('n_failed_polish_blocks', 99),
        r.get('admm_net_recourse_before_polish') or float('inf')))
    return best, ('no case achieved a clean polish; fewest failed blocks, '
                  'tie-broken by lowest pre-polish recourse')


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    os.makedirs(EV.ARCHIVE_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.9-A rho sweep', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {
        'stage': 'P5.9-A', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': 'canonical positive-bootstrap base x0',
        'formulation': 'RESCALED (base_objective + effective_scale * ADMM terms)',
        'generations_per_case': 1,
        'adaptive_penalty': 'production value (true), unchanged',
        'held_fixed': ['IPOPT options', 'ADMM tolerances', 'stopping criteria',
                       'anchor policy (MIDPOINT)', 'rho_ess = 1.0 (inherited)'],
        'inherited_rho_in_T0': INHERITED_RHO,
        'parameter_file_rho': {'v': 1.0, 'pf': 1.0, 'ess': 1.0},
        'baseline_note': ('A1 holds rho_pf at its inherited 2.25 and A2 holds '
                          'rho_v at its inherited 1.5, so each arm varies one '
                          'group away from the configuration the accepted '
                          'evidence actually ran'),
        'selection_rule': ('1) zero failed polish blocks; 2) lowest polished '
                           'total objective; 3) else fewest failed blocks, '
                           'tie-broken by lowest pre-polish recourse'),
        'reference': {'p58c_gen1_rescaled_prepolish': P58C_GEN1_PREPOLISH,
                      'p58c_gen1_rescaled_polish_failed': P58C_GEN1_POLISH_FAILED,
                      'current_gen1_prepolish': CURRENT_GEN1_PREPOLISH,
                      'current_gen1_polished': CURRENT_GEN1_POLISHED},
        'cases': []}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print(f'[P5.9-A] {len(CASES)} cases, one generation each, from frozen T0\n',
          flush=True)

    for case_id, rho in CASES:
        shown = rho if rho else INHERITED_RHO
        print(f'[P5.9-A] {case_id}  rho_v={shown["v"]} rho_pf={shown["pf"]} '
              f'rho_ess=1.0'
              f'{"  (inherited from T0, untouched)" if rho is None else ""} ...',
              flush=True)
        state, info = EV.load_template(rescale=True, rho=rho)
        record, _ = EV.evaluate(x0, template_state=state, case_id=case_id,
                                rho=rho, archive_label=case_id)
        row = EV.row(record)
        row['template_info'] = info
        row['rho_ratio_to_inherited'] = (
            {g: round(rho[g] / INHERITED_RHO[g], 4) for g in rho}
            if rho else {g: 1.0 for g in ('v', 'pf')})
        row['cycle_detail'] = record.get('cycle_detail')
        report['cases'].append(row)
        persist()

        print(f"           status={row['status']}  cycles={row['admm_cycles']}  "
              f"pre-polish={row['admm_net_recourse_before_polish']}  "
              f"polished={row['total_objective']}  "
              f"failed_blocks={row['n_failed_polish_blocks']}", flush=True)
        print(f"           primal ratios v/pf/ess = {row['primal_v_ratio']}/"
              f"{row['primal_pf_ratio']}/{row['primal_ess_ratio']}  "
              f"rho_final={row['rho_observed_final']}", flush=True)

        if case_id == 'A0_inherited_rho':
            observed = row['admm_net_recourse_before_polish']
            delta = (observed - P58C_GEN1_PREPOLISH) if observed is not None else None
            report['reproduction_gate'] = {
                'expected_prepolish': P58C_GEN1_PREPOLISH,
                'observed_prepolish': observed,
                'delta': delta,
                'tolerance': GATE_TOLERANCE,
                'expected_failed_blocks': P58C_GEN1_POLISH_FAILED,
                'observed_failed_blocks': row.get('failed_blocks'),
                'passed': delta is not None and abs(delta) <= GATE_TOLERANCE}
            persist()
            if not report['reproduction_gate']['passed']:
                report['aborted'] = ('A0 did not reproduce P5.8-C generation 1; '
                                     'the sweep below would not be comparable '
                                     'to the accepted evidence')
                persist()
                print(f"\n[P5.9-A] REPRODUCTION GATE FAILED: observed "
                      f"{observed}, expected {P58C_GEN1_PREPOLISH} "
                      f"(delta {delta}). Not continuing.", flush=True)
                sys.exit(1)
            print(f"           GATE PASSED (delta {delta:+.2f} on "
                  f"{P58C_GEN1_PREPOLISH})\n", flush=True)

    best, reason = select_best(report['cases'])
    report['best_case'] = best['case_id'] if best else None
    report['best_case_reason'] = reason
    report['best_case_row'] = best
    persist()

    print('\n[P5.9-A] summary')
    header = (f"      {'case':22} {'cyc':>4} {'pre-polish':>18} "
              f"{'polished':>18} {'fail':>5} {'pf_ratio':>9}")
    print(header)
    for r in report['cases']:
        print(f"      {r['case_id']:22} {str(r['admm_cycles']):>4} "
              f"{r['admm_net_recourse_before_polish'] or float('nan'):18.2f} "
              f"{r['total_objective'] or float('nan'):18.2f} "
              f"{r['n_failed_polish_blocks']:5d} "
              f"{r['primal_pf_ratio'] or float('nan'):9.4f}")
    print(f"\n[P5.9-A] best: {report['best_case']}  ({reason})")
    print(f'[P5.9-A] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
