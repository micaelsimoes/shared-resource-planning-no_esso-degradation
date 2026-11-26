"""P5.15 S56 -- D0 post-check (ZERO SOLVES; no unpickling). Reads the D0 run record; never re-runs it.

Why this exists: the D0 harness (`p515_s56_tn_realized_resolve.py`, run_v1) recorded `solution_loaded = result is not
None` -- the central check's test, true for every solve that returns a SolverResults, including the failed ones
(production `_run_smopf` returns the result without loading it when `solver_result_succeeded` is False). For a failed
solve the harness therefore extracted costs at the model's UNLOADED (initial) values, and its own `statistics` block
includes those rows for the unit cell. This script recomputes the statistics from the per-block records with the
production success test, leaves run_v1 untouched, and adds two readings the report needs:

  A. statistics_v2: a block enters D1 only when BOTH arms loaded a solution (production `solver_result_succeeded`:
     solver status ok and termination optimal / locallyOptimal / globallyOptimal -- IPOPT "Solved To Acceptable Level"
     is loaded by production and therefore enters, flagged); a 'strict' variant keeps only blocks whose both arms exit
     "Optimal Solution Found.". Same formulas as the harness (spec 5d35b048 'statistics'); coverage reported as the
     share of sum_b w_b. Pair statistic D1(x0) - D1(unit) on the blocks valid in BOTH cells.
  B. D1 attribution (every valid block): per (scenario, hour)
        c[s,t] = w_b omega_s [ (gen_cost_B - gen_cost_A)[s,t] - sum_DN pi[s_m,t] (P_DSO[s,t] - Pbar_DSO[t]) ]
     plus the block-level TN flexibility-cost change w_b sum_s omega_s (flex_B - flex_A)[s]; sum == D1_b checked.
     P_DSO = the Arm B fixed ADN load (pc, the DN realized draw), Pbar_DSO its probability-weighted mean. Hours are
     classed by the TN renewable curtailment change (B - A > 1e-3 MW: 'B curtails more'; < -1e-3: 'A curtails more';
     else 'no curtailment change'), and by whether the set of controllable units at a bound changed.
  C. unit-cell failure diagnosis: the unit cell's shared ESS (node 7) pnet/qnet were fixed at the TSO copy's terminal
     schedule. For every unit block the Arm A model is rebuilt by the harness's own builder (zero solves) and the
     shared-ESS rows are evaluated at the fixed schedule with the natural split (pch = max(pnet, 0), pdch =
     max(-pnet, 0), hats = power / S, soc by the model's own recursion, day-balance slack at the residual clipped to its
     bounds): max violation per row family, set beside each block's solve status in both arms.
Guards: SolveProfileGuard(permitted=()) installed before any project import, verified 0 (the harness module imported for
C installs its own guard on top; any solve would hit both); pickle.load/loads blocked, verified 0.
Usage: <canonical python> -u p515_s56_tn_realized_postcheck.py --run-dir RUN_DIR --spec SPEC --out-dir NEW_DIR
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 D0 post-check (never solves)').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

ACCEPTED_TC = ('optimal', 'locallyOptimal', 'globallyOptimal')
CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [D0-post] {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:300]}' if detail is not None and not ok else ''))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def status(arm_rec):
    s = arm_rec.get('solve') or {}
    ex = (s.get('ipopt_records') or [{}])[-1]
    loaded = s.get('solver_status') == 'ok' and s.get('termination_condition') in ACCEPTED_TC
    return {'loaded_by_production': loaded, 'strict_optimal': loaded and ex.get('exit') == 'Optimal Solution Found.',
            'termination_condition': s.get('termination_condition'), 'solver_status': s.get('solver_status'),
            'exit': ex.get('exit'), 'iterations': ex.get('iterations'), 'wall_s': s.get('wall_s')}


def block_terms(blk, dn_ids):
    """D1 attribution for one block (both arms loaded). Returns (rows, totals)."""
    iu = blk['inputs_used']
    w, omega, prices = iu['w'], iu['omega'], iu['prices_eur_per_mwh_by_hour_by_market']
    A, B = blk['A'], blk['B']
    pbar = {dn: [sum(omega[k] * B['hours'][k][t]['adn'][str(dn)]['pc_mw'] for k in omega) for t in range(24)]
            for dn in dn_ids}
    rows, cls_tot = [], {}
    for k in omega:
        s_m = int(k.split('_')[0])
        for t in range(24):
            ha, hb = A['hours'][k][t], B['hours'][k][t]
            dgen = hb['gen_cost_eur'] - ha['gen_cost_eur']
            cov = sum(prices[t][s_m] * (hb['adn'][str(dn)]['pc_mw'] - pbar[dn][t]) for dn in dn_ids)
            dcurt = hb['res_curt_mw'] - ha['res_curt_mw']
            cls = 'B curtails more' if dcurt > 1e-3 else ('A curtails more' if dcurt < -1e-3 else 'no curtailment change')
            lb_changed = ha['gens_at_lb'] != hb['gens_at_lb'] or ha['gens_at_ub'] != hb['gens_at_ub']
            c = w * omega[k] * (dgen - cov)
            rows.append({'scenario': k, 'hour': t, 'contribution_eur_weighted': c, 'class': cls,
                         'controllable_bound_set_changed': lb_changed, 'd_curt_mw': dcurt,
                         'd_draw_mw': sum(hb['adn'][str(dn)]['pc_mw'] - ha['adn'][str(dn)]['pc_mw'] for dn in dn_ids),
                         'gens_at_lb_A': ha['gens_at_lb'], 'gens_at_lb_B': hb['gens_at_lb'],
                         'gens_at_ub_A': ha['gens_at_ub'], 'gens_at_ub_B': hb['gens_at_ub'],
                         'price': prices[t][s_m],
                         'mc_p_A': {dn: ha['adn'][dn].get('mc_p_eur_per_mwh') for dn in ha['adn']},
                         'mc_p_B': {dn: hb['adn'][dn].get('mc_p_eur_per_mwh') for dn in hb['adn']}})
            key = cls + (' | bound set changed' if lb_changed else '')
            cls_tot[key] = cls_tot.get(key, 0.0) + c
    flex = w * sum(omega[k] * (B['per_scenario'][k]['flex_cost'] - A['per_scenario'][k]['flex_cost']) for k in omega)
    total = sum(r['contribution_eur_weighted'] for r in rows) + flex
    return rows, {'by_class': cls_tot, 'tn_flex_change': flex, 'sum': total}


def stats_v2(res, dn_ids):
    out = {}
    for lab, cc in res['cells'].items():
        blocks, tot = {}, {}
        for bkey, blk in cc['blocks'].items():
            sa, sb = status(blk['A']), status(blk['B'])
            iu = blk['inputs_used']
            w, omega, prices = iu['w'], iu['omega'], iu['prices_eur_per_mwh_by_hour_by_market']
            B = blk['B']
            # V_b from the Arm B fixed ADN loads (set before the solve, so valid whatever the solve did)
            V = sum(omega[k] * sum(prices[t][int(k.split('_')[0])]
                                   * (B['hours'][k][t]['adn'][str(dn)]['pc_mw'] - iu['tso_p_common_mw'][str(dn)][t])
                                   for t in range(24)) for dn in dn_ids for k in omega)
            row = {'w': w, 'A': sa, 'B': sb, 'cov': iu['cov_block'], 'V_price_valued_draw_change': V,
                   'valid': sa['loaded_by_production'] and sb['loaded_by_production'],
                   'strict': sa['strict_optimal'] and sb['strict_optimal']}
            if row['valid']:
                cA, cB = blk['A']['costs']['C_gen_plus_flex'], blk['B']['costs']['C_gen_plus_flex']
                row.update({'C_A': cA, 'C_B': cB, 'dC': cB - cA, 'D1': (cB - cA) - iu['cov_block'],
                            'D1prime': (cB - cA) - V,
                            'C_A_minus_saved_gen': None,
                            'max_violation_A': blk['A']['max_violation']['max'],
                            'max_violation_B': blk['B']['max_violation']['max'],
                            'max_slack_var_B': max((v['max'] for v in blk['B']['slack_var_extremes'].values()
                                                    if v['max'] is not None), default=None),
                            'min_slack_var_B': min((v['min'] for v in blk['B']['slack_var_extremes'].values()
                                                    if v['min'] is not None), default=None),
                            'detector_A': blk['A']['detector_components']['detector_penalty_total'],
                            'detector_B': blk['B']['detector_components']['detector_penalty_total']})
                _rows, terms = block_terms(blk, dn_ids)
                row['attribution'] = terms
                _ck(f'{lab} {bkey}: attribution sums to w_b D1_b (abs <= 1e-6 max(1,|x|))',
                    abs(terms['sum'] - w * row['D1']) <= 1e-6 * max(1.0, abs(w * row['D1'])), [terms['sum'], w * row['D1']])
            blocks[bkey] = row
        for variant in ('valid', 'strict'):
            sel = [b for b in blocks.values() if b[variant]]
            wsum_all = sum(b['w'] for b in blocks.values())
            tot[variant] = {'n_blocks': len(sel), 'weight_share': sum(b['w'] for b in sel) / wsum_all,
                            'blocks': sorted(k for k, b in blocks.items() if b[variant]),
                            'C_A_w': sum(b['w'] * b['C_A'] for b in sel), 'C_B_w': sum(b['w'] * b['C_B'] for b in sel),
                            'dC_w': sum(b['w'] * b['dC'] for b in sel), 'cov_w': sum(b['w'] * b['cov'] for b in sel),
                            'D1_w': sum(b['w'] * b['D1'] for b in sel), 'D1prime_w': sum(b['w'] * b['D1prime'] for b in sel),
                            'V_w': sum(b['w'] * b['V_price_valued_draw_change'] for b in sel)}
        out[lab] = {'blocks': blocks, 'totals': tot,
                    'n_loaded': {arm: sum(1 for b in blocks.values() if b[arm]['loaded_by_production'])
                                 for arm in ('A', 'B')},
                    'n_strict': {arm: sum(1 for b in blocks.values() if b[arm]['strict_optimal']) for arm in ('A', 'B')},
                    'cov_w_all_blocks': sum(b['w'] * b['cov'] for b in blocks.values())}
    pair = {}
    if 'x0' in out and 'unit' in out:
        for variant in ('valid', 'strict'):
            common = sorted(set(out['x0']['totals'][variant]['blocks']) & set(out['unit']['totals'][variant]['blocks']))
            d = {lab: sum(out[lab]['blocks'][k]['w'] * out[lab]['blocks'][k]['D1'] for k in common) for lab in ('x0', 'unit')}
            pair[variant] = {'common_blocks': common, 'n': len(common),
                             'weight_share': sum(out['x0']['blocks'][k]['w'] for k in common)
                             / sum(b['w'] for b in out['x0']['blocks'].values()),
                             'D1_x0_on_common': d['x0'], 'D1_unit_on_common': d['unit'],
                             'pair_D1_x0_minus_unit': d['x0'] - d['unit']}
    return out, pair


def attribution_detail(res, dn_ids, lab, top=20):
    allrows = []
    for bkey, blk in res['cells'][lab]['blocks'].items():
        if not (status(blk['A'])['loaded_by_production'] and status(blk['B'])['loaded_by_production']):
            continue
        rows, _t = block_terms(blk, dn_ids)
        for r in rows:
            r['block'] = bkey
        allrows += rows
    by_class = {}
    for r in allrows:
        key = r['class'] + (' | bound set changed' if r['controllable_bound_set_changed'] else '')
        d = by_class.setdefault(key, {'n_scenario_hours': 0, 'sum_eur_weighted': 0.0})
        d['n_scenario_hours'] += 1
        d['sum_eur_weighted'] += r['contribution_eur_weighted']
    allrows.sort(key=lambda r: -abs(r['contribution_eur_weighted']))
    return {'by_class': by_class, 'top_scenario_hours': allrows[:top], 'n_scenario_hours': len(allrows)}


def unit_failure_diagnosis(spec, run_res):
    """C: zero-solve evaluation of the shared-ESS rows at the fixed schedule, unit cell, Arm A builds."""
    import p515_s56_tn_realized_resolve as H   # installs its own (permitted _solve_one) guard; ours stays underneath
    import model_construction_helpers as mch
    planning = H.S.read_planning_from_derived_case({'derived_case': {'path': spec['instance']['case_path']}},
                                                   ARGS.out_dir, H._Stages())
    cell = next(c for c in spec['cells'] if c['label'] == 'unit')
    cd = H.load_cell(cell)
    pc = deepcopy(planning)
    tn = pc.transmission_network
    dn_ids = list(tn.active_distribution_network_nodes)
    cand = pc.get_initial_candidate_solution()
    iy = H._year_key(pc.shared_ess_data.years, cell['candidate_canonical']['investment_year'])
    for n, (s_val, e_val) in cell['candidate_canonical']['nodes'].items():
        cand['investment'][int(n)][iy]['s'] = float(s_val)
        cand['investment'][int(n)][iy]['e'] = float(e_val)
    H.srp._rebuild_candidate_total_capacities(pc, cand)
    tn.update_data_with_candidate_solution(cand['total_capacity'])
    avail = {n: {y: {'s': cd['er']['storage_per_node'][str(n)]['published_available_capacity_terminal'][str(int(y))]['s_available'],
                     'e': cd['er']['storage_per_node'][str(n)]['published_available_capacity_terminal'][str(int(y))]['e_available']}
                 for y in tn.years} for n in dn_ids}
    bdata = {(str(int(y)), d): H.block_data('unit', cd['ms'], tn, y, d, dn_ids) for y in tn.years for d in tn.days}
    models, _recs = H.build_arm(tn, cand, avail, 'unit', 'A', bdata, dn_ids)
    out = {}
    for year in tn.years:
        for day in tn.days:
            m, net = models[year][day], tn.network[year][day]
            e = net.get_shared_energy_storage_idx(7)
            s0m, s0o = mch.sess_na_scenario(m)
            S_r = pe.value(m.shared_es_s_rated_fixed[e])
            E_r = pe.value(m.shared_es_e_rated_fixed[e])
            sess = net.shared_energy_storages[e]
            soc_prev = E_r * mch.ENERGY_STORAGE_RELATIVE_INIT_SOC
            for p in m.periods:
                pn = m.shared_es_pnet[e, s0m, s0o, p].value
                pch, pdch = max(pn, 0.0), max(-pn, 0.0)
                m.shared_es_pch[e, s0m, s0o, p].set_value(pch, skip_validation=True)
                m.shared_es_pdch[e, s0m, s0o, p].set_value(pdch, skip_validation=True)
                m.shared_es_pch_hat[e, s0m, s0o, p].set_value(pch / S_r if S_r else 0.0, skip_validation=True)
                m.shared_es_pdch_hat[e, s0m, s0o, p].set_value(pdch / S_r if S_r else 0.0, skip_validation=True)
                dt = mch.period_duration_hours(m)
                soc_prev = soc_prev + sess.eff_ch * pch * dt - pdch * dt / sess.eff_dch
                m.shared_es_soc[e, s0m, s0o, p].set_value(soc_prev, skip_validation=True)
            final = E_r * mch.ENERGY_STORAGE_RELATIVE_INIT_SOC
            resid = soc_prev - final
            for name, val in (('slack_shared_es_soc_final_up', max(resid, 0.0)),
                              ('slack_shared_es_soc_final_down', max(-resid, 0.0))):
                v = getattr(m, name, None)
                if v is not None:
                    x = v[e, s0m, s0o]
                    x.set_value(min(max(val, x.lb if x.lb is not None else val), x.ub if x.ub is not None else val),
                                skip_validation=True)
            fam = {}
            for c in m.component_objects(pe.Constraint, active=True):
                if not c.name.startswith('sess_'):
                    continue
                worst = 0.0
                for idx in c:
                    if (idx[0] if isinstance(idx, tuple) else idx) != e:
                        continue
                    cd_ = c[idx]
                    if not cd_.active:
                        continue
                    body = pe.value(cd_.body)
                    lo = pe.value(cd_.lower) if cd_.has_lb() else None
                    up = pe.value(cd_.upper) if cd_.has_ub() else None
                    worst = max(worst, (lo - body) if lo is not None else 0.0, (body - up) if up is not None else 0.0)
                fam[c.name] = worst
            soc_bounds = (E_r * mch.ENERGY_STORAGE_MIN_ENERGY_STORED, E_r * mch.ENERGY_STORAGE_MAX_ENERGY_STORED)
            socs = [m.shared_es_soc[e, s0m, s0o, p].value for p in m.periods]
            key = f'{int(year)}|{day}'
            rb = run_res['cells']['unit']['blocks'][key]
            out[key] = {'S_rated_pu': S_r, 'E_rated_pu': E_r, 'max_violation_by_sess_row': fam,
                        'capability_max_ratio': max(((m.shared_es_pnet[e, s0m, s0o, p].value ** 2
                                                      + m.shared_es_qnet[e, s0m, s0o, p].value ** 2) ** 0.5) / S_r
                                                    for p in m.periods),
                        'soc_min_minus_lb_pu': min(socs) - soc_bounds[0], 'soc_ub_minus_max_pu': soc_bounds[1] - max(socs),
                        'final_soc_residual_pu': resid,
                        'day_balance_slack_ub_pu': (getattr(m, 'slack_shared_es_soc_final_up')[e, s0m, s0o].ub
                                                    if hasattr(m, 'slack_shared_es_soc_final_up') else None),
                        'status_A': status(rb['A']), 'status_B': status(rb['B'])}
    return out


ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ARGS = ap.parse_args()
    if os.path.exists(ARGS.out_dir):
        print(f'[D0-post] REFUSE: {ARGS.out_dir} exists', flush=True)
        return 1
    t0 = time.time()
    rpath = os.path.join(ARGS.run_dir, 'tn_realized_resolve_results.json')
    man = json.load(open(os.path.join(ARGS.run_dir, 'manifest_sha256.json')))
    _ck('run record sha256 == run manifest', _sha(rpath) == man['outputs']['tn_realized_resolve_results.json'])
    spec = json.load(open(ARGS.spec))
    _ck('spec sha256 == run record spec sha256', _sha(ARGS.spec) == man['spec']['sha256'])
    res = json.load(open(rpath))
    os.makedirs(ARGS.out_dir)
    dn_ids = [5, 7, 9]
    harness_loaded_flag = {lab: sorted(k for k, b in cc['blocks'].items() for arm in ('A', 'B')
                                       if (b[arm].get('solve') or {}).get('solution_loaded')
                                       and not status(b[arm])['loaded_by_production'])
                           for lab, cc in res['cells'].items()}
    v2, pair = stats_v2(res, dn_ids)
    attrib = {lab: attribution_detail(res, dn_ids, lab) for lab in res['cells']}
    diag = unit_failure_diagnosis(spec, res)
    out = {'stage': 'P5.15 S56 D0 post-check (zero solves)', 'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
           'run_record': {'path': rpath, 'sha256': _sha(rpath)}, 'spec': {'path': ARGS.spec, 'sha256': _sha(ARGS.spec)},
           'utc': datetime.now(timezone.utc).isoformat(),
           'harness_defect': ('run_v1 solution_loaded = (result is not None): true for failed solves, so the harness '
                              'extracted costs at unloaded values and its statistics include them; blocks/arms affected '
                              'listed in blocks_marked_loaded_but_not_loaded_by_production'),
           'blocks_marked_loaded_but_not_loaded_by_production': harness_loaded_flag,
           'objective_convention': res['objective_convention'], 'weighting': res['weighting'],
           'statistics_v2': v2, 'pair_v2': pair, 'attribution': attrib, 'unit_failure_diagnosis': diag}
    gv = GUARD.verify(0, 0)
    out['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': gv}
    out['pickle'] = dict(PK)
    out['guards_ok'] = (not gv) and PK == {'load': 0, 'loads': 0}
    out['checks'] = CHECKS
    out['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    out['wall_s'] = time.time() - t0
    path = os.path.join(ARGS.out_dir, 'tn_realized_postcheck.json')
    with open(path, 'x', encoding='utf-8') as h:
        GRIO.dump(out, h, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(ARGS.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump({'outputs': {'tn_realized_postcheck.json': _sha(path)}, 'inputs': {rpath: _sha(rpath),
                   ARGS.spec: _sha(ARGS.spec)}}, h, indent=1)
    _log(f"wrote {path}; guards_ok {out['guards_ok']}; failed {out['failed_checks']}")
    return 0 if out['guards_ok'] and not out['failed_checks'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
