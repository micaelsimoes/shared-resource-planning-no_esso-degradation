"""P5.15 Addendum 16 item 3 - gate s35pt (price-taker initialization, frozen spec v6) - INDEPENDENT evaluation.

Zero solves, SolveProfileGuard armed. Recomputes every gate-3 criterion from the two runs' own per-cycle trajectories
(g_baseline.json) instead of trusting any summary field: the s35ref writer's `stopped_by` defect showed why.

Stop cause (both runs): 'boyd' iff the last R cycles are consecutive, all have boyd_all_pass, and the run ended before
its cap (R = required_consecutive_cycles = 3; cap 150 for s35pt, 500 for s35ref); else 'cap'.

Gate 3 (spec v6 gate_s35pt.pass_iff), all three required:
  (a) s35pt stopped under Boyd within 150 and no channel frozen at a rho clamp;
  (b) |cost_pt - cost_ref| <= bar, bar = terminal objective_change_abs(pt) + terminal objective_change_abs(ref)
      (rule nine; cost = gross_operational_cost at each run's terminal cycle);
  (c) |EFC_pt - EFC_ref| <= 0.02 * EFC_ref, EFC = efc_per_day_max at each terminal cycle.
Validity: (b) and (c) are well posed only if the reference stopped under Boyd (it did: cycle 477, run 475-477).

Also reported, not gated:
  * bracket analysis - the reference approached the storage equilibrium from BELOW (EFC rising at its stop); the
    price-taker start approaches from ABOVE; the sign of each run's late EFC slope and whether the two terminal EFC
    values bracket a common limit;
  * per-channel terminal ratios (settling quality) for both runs;
  * EFC per cohort-year for both runs against the price-taker bound (Z2: 1.1918 / 1.1888 / 0.9647);
  * SoH floor block (active rows, floor multiplier) from each run's floor sidecar;
  * network failures per cycle for both runs; cycles to stop; system cost and EFC trajectories at matched cycles.
Usage: python p515_s35pt_evaluate.py [PT_RUN_DIR] [--dry-run]; writes PT_RUN_DIR/s35pt_evaluation_v2.json (write-once; v1 retained, its hash recorded).
"""
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')
CH = ('v', 'pf', 'ess')
PRICE_TAKER = {'y0': 1.1918, 'y1': 1.1888, 'y2': 0.9647}
MATCH = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150)


def _rows(run):
    p = glob.glob(os.path.join(run, 'g_*.json'))
    if len(p) != 1:
        raise RuntimeError(f'expected one g_*.json in {run}, found {p}')
    return json.load(open(p[0]))['cycle_trajectory'], p[0]


def _stop(rows, cap, required):
    allpass = {r['cycle'] for r in rows if r.get('boyd_all_pass')}
    tail = [r['cycle'] for r in rows[-required:]]
    ok = (len(tail) == required and all(c in allpass for c in tail)
          and all(tail[i] + 1 == tail[i + 1] for i in range(len(tail) - 1)) and len(rows) < cap)
    return {'stopped_by': 'boyd' if ok else 'cap', 'stop_run_cycles': tail if ok else None, 'cycles': len(rows)}


def _floor(run):
    side = glob.glob(os.path.join(run, 'soh_floor_sidecar_*.jsonl'))
    if not side:
        return {'available': False}
    last = None
    for line in open(side[0]):
        if line.strip():
            last = json.loads(line)
    if last is None:
        return {'available': False}
    ent = last.get('entries', [])
    act = [e for e in ent if e.get('active')]
    efc_cy = {}
    for e in ent:
        if e.get('efc_per_day') is not None:
            efc_cy.setdefault(f"y{e['y']}", []).append(e['efc_per_day'])
    return {'available': True, 'cycle': last.get('cycle'), 'n_rows': len(ent), 'n_active': len(act),
            'floor_multiplier': max((abs(e['dual']) for e in act if e.get('dual') is not None), default=0.0),
            'min_soh': min((e['es_soh_per_unit_cumul'] for e in ent if e.get('es_soh_per_unit_cumul') is not None), default=None),
            'efc_per_day_by_year_mean_over_nodes': {k: sum(v) / len(v) for k, v in efc_cy.items()}}


def _late_slope(rows, window=20):
    e = [(r['cycle'], r.get('efc_per_day_max')) for r in rows if r.get('efc_per_day_max') is not None]
    if len(e) <= window:
        return None
    return (e[-1][1] - e[-1 - window][1]) / (e[-1][0] - e[-1 - window][0])


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    pt_run = os.path.abspath(args[0]) if args else os.path.join(RES, 'P515S35_PT_run')
    out_path = os.path.join(pt_run, 's35pt_evaluation_v2.json')
    v1_path = os.path.join(pt_run, 's35pt_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s35pt independent evaluation').install()
    try:
        pt, pt_path = _rows(pt_run)
        ref, ref_path = _rows(REF_RUN)
        req = pt[-1].get('required_consecutive_cycles') or 3
        s_pt, s_ref = _stop(pt, 150, req), _stop(ref, 500, req)
        clamp_pt = any(r.get(f'rho_at_clamp_{c}') for r in pt for c in CH)
        lp, lr = pt[-1], ref[-1]
        cost_pt, cost_ref = lp['gross_operational_cost'], lr['gross_operational_cost']
        step_pt, step_ref = lp.get('objective_change_abs'), lr.get('objective_change_abs')
        bar = (step_pt + step_ref) if (step_pt is not None and step_ref is not None) else None
        efc_pt, efc_ref = lp.get('efc_per_day_max'), lr.get('efc_per_day_max')
        crit_a = (s_pt['stopped_by'] == 'boyd') and not clamp_pt
        crit_b = (abs(cost_pt - cost_ref) <= bar) if bar is not None else None
        crit_c = (abs(efc_pt - efc_ref) <= 0.02 * abs(efc_ref)) if (efc_pt is not None and efc_ref) else None
        # v2: (b) and (c) test reproducibility of a fixed point only if BOTH runs stopped under Boyd. v1 checked the
        # reference alone, so a cap-stopped initialized run was wrongly treated as a valid comparator.
        valid_bc = (s_ref['stopped_by'] == 'boyd') and (s_pt['stopped_by'] == 'boyd')

        slope_pt, slope_ref = _late_slope(pt), _late_slope(ref)
        bracket = {
            'ref_terminal_efc': efc_ref, 'pt_terminal_efc': efc_pt,
            'ref_late_slope_per_cycle': slope_ref, 'pt_late_slope_per_cycle': slope_pt,
            'ref_approach': ('from below' if (slope_ref or 0) > 0 else 'from above or flat'),
            'pt_approach': ('from above' if (slope_pt or 0) < 0 else 'from below or flat'),
            'terminal_values_bracket_a_common_limit': (efc_pt is not None and efc_ref is not None and
                                                        (slope_ref or 0) > 0 and (slope_pt or 0) < 0 and efc_pt >= efc_ref),
            'relative_gap': (abs(efc_pt - efc_ref) / efc_ref) if (efc_pt is not None and efc_ref) else None}

        def settling(r):
            return {c: max(r.get(f'boyd_{c}_primal_ratio') or 0.0, r.get(f'boyd_{c}_dual_ratio') or 0.0) for c in CH}

        def fails_per_cycle(run, rows):
            p = glob.glob(os.path.join(run, 'network_failures_*.jsonl'))
            n = sum(1 for l in open(p[0]) if l.strip()) if p else None
            return {'failures': n, 'per_cycle': (n / len(rows)) if n is not None else None}

        refd = {r['cycle']: r for r in ref}
        matched = [{'cycle': k, 'pt_cost': pt[k - 1]['gross_operational_cost'], 'ref_cost': refd[k]['gross_operational_cost'],
                    'pt_efc': pt[k - 1].get('efc_per_day_max'), 'ref_efc': refd[k].get('efc_per_day_max')}
                   for k in MATCH if k <= len(pt) and k in refd]

        out = {
            'stage': 'P5.15 Addendum 16 item 3 - gate s35pt - independent evaluation v2',
            'predecessor': ({'path': os.path.relpath(v1_path, REPO), 'sha256': hashlib.sha256(open(v1_path, 'rb').read()).hexdigest(),
                             'why_superseded': 'v1 marked criteria (b)/(c) well posed when only the reference had stopped under Boyd'}
                            if os.path.exists(v1_path) else None),
            'sources': {'pt': {'path': os.path.relpath(pt_path, REPO), 'sha256': hashlib.sha256(open(pt_path, 'rb').read()).hexdigest()},
                        'ref': {'path': os.path.relpath(ref_path, REPO), 'sha256': hashlib.sha256(open(ref_path, 'rb').read()).hexdigest()}},
            'stop': {'pt': s_pt, 'ref': s_ref, 'pt_rho_at_clamp_any': clamp_pt},
            'GATE3': {
                'criterion_a_boyd_stop_no_clamp': crit_a,
                'criterion_b_cost_within_rule_nine_bar': {'pt_cost': cost_pt, 'ref_cost': cost_ref, 'abs_diff': abs(cost_pt - cost_ref),
                                                          'pt_terminal_step': step_pt, 'ref_terminal_step': step_ref, 'bar': bar,
                                                          'pass': crit_b},
                'criterion_c_efc_within_2pct': {'pt_efc': efc_pt, 'ref_efc': efc_ref,
                                                'rel_diff': (abs(efc_pt - efc_ref) / efc_ref) if (efc_pt is not None and efc_ref) else None,
                                                'bound': 0.02, 'pass': crit_c},
                'criteria_b_c_well_posed': valid_bc,
                'criterion_b_status': ('evaluated' if valid_bc else 'INDETERMINATE - the initialized run did not settle, so the rule-nine bar does not apply'),
                'criterion_c_status': ('evaluated' if valid_bc else 'NOT A FIXED-POINT TEST here - the two runs approach from opposite directions and one did not stop'),
                'PASS': bool(crit_a and crit_b and crit_c and valid_bc)},
            'bracket_analysis': bracket,
            'settling_quality': {'pt': settling(lp), 'ref': settling(lr),
                                 'pt_objective_rule_ten': lp.get('objective_change_ratio'),
                                 'ref_objective_rule_ten': lr.get('objective_change_ratio')},
            'efc_per_cohort_year': {'pt': _floor(pt_run).get('efc_per_day_by_year_mean_over_nodes'),
                                    'ref': _floor(REF_RUN).get('efc_per_day_by_year_mean_over_nodes'),
                                    'price_taker_bound': PRICE_TAKER},
            'soh_floor': {'pt': _floor(pt_run), 'ref': _floor(REF_RUN)},
            'failures': {'pt': fails_per_cycle(pt_run, pt), 'ref': fails_per_cycle(REF_RUN, ref)},
            'matched_cycles': matched,
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    print(json.dumps({k: out[k] for k in ('stop', 'GATE3', 'bracket_analysis', 'settling_quality', 'failures')}, indent=1, default=str))
    if not dry:
        with open(out_path, 'w') as h:
            json.dump(out, h, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
