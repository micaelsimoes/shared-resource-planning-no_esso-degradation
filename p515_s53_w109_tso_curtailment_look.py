"""P5.15 W109 (Addendum 55) -- records-only look at the negative TSO RES curtailment. ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any Pyomo / production import and verified at exactly
0 at the end. Nothing is built for solving and nothing is solved; the persisted models are only read.

INSTANCE: SRP1, x = 0 (candidate_key 8435c71859ddde68...), the settled cycle-181 certificate of campaign
s53_w101_srp1_cont_x0, eval d110bd1a5977df1e...; terminal models `certified_models.pkl` (sha256 99ab1070..., verified
before unpickling). The planning problem (network data, block weights) is the canonical SRP1 baseline
(p56a_oracle.load_baseline, scenario checksum asserted there).

QUESTION (Addendum 55): the coordinated curtailment figure 589.43 EUR at 1 EUR/MWh, block-weighted, is DSO 608.67 plus
TSO -19.24. Where is the negative term (block, unit, hour), and is it the helper or the model?

FORMULAS (B = network.baseMVA; omega = prob_market[s_m] * prob_operation[s_o]; w_b = srp._get_admm_block_weight,
          d_b = years[year] * days[day] (undiscounted))
  c[g,s_m,s_o,p]   = B * (pg_avail[g,s_o,p] - pg[g,s_m,s_o,p])   [MWh per generator-hour]; the exact term of
                     model_construction_helpers.gen_curtailment_definitional_value (and gen_curtailment_penalty).
  block value      E_b = sum_{s_m,s_o} omega * sum_{g curtaillable, p} c          (= curtailment_report per_block)
  EUR at 1, block-weighted = sum_b w_b E_b ; MWh day-weighted = sum_b d_b E_b.
  tol band         TOL_MW = EQUALITY_TOLERANCE * B: pg's declared upper bound is pg_avail + EQUALITY_TOLERANCE
                     (model_construction_helpers.pg_bounds), so c >= -TOL_MW is admissible at every generator-hour.
  priced (first order, NOT a solve): sum_b w_b sum_s omega * cost_energy_p[s_m][p] * c   [EUR, Q units]
  floor            the most negative value the declared bounds admit: -sum_b w_b sum_s omega * N_free_b * TOL_MW
                     (N_free_b = curtaillable generator-hours with ub > 0).
Apparent-power row (network.py sg_capability, model_construction_helpers.sg_avail_rule): pg^2 + qg^2 <= sg_avail^2
  in p.u.^2; recorded per negative entry as its violation (UB.constraint_violation) -- near-zero availability makes
  this row too weak (violation ~ 2 * pg_avail * delta) to enforce pg <= pg_avail within IPOPT's feasibility tolerance.
Checks per entry: the pickled pg_avail Param equals the raw availability data gen.pg[s_o][p] of the planning
problem's network (same scenario/hour index), pg's pickled ub equals pg_avail + EQUALITY_TOLERANCE, and pg - ub (the
IPOPT bound-relaxation overshoot, if any).

Output (write-once, new directory): data/SRP1/Results/P515S53/w109_tso_curtailment/
    w109_tso_curtailment_look.json, launch.log (captured by the launcher), manifest_sha256.json (--manifest)

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w109_tso_curtailment
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w109_tso_curtailment_look.py \\
        > data/SRP1/Results/P515S53/w109_tso_curtailment/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w109_tso_curtailment_look.py --manifest
"""
import hashlib
import inspect
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W109 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'w109_tso_curtailment')
EVAL_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                        'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0')
MODELS = {'path': os.path.join(EVAL_DIR, 'certified_models.pkl'),
          'sha256': '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'}
INSTANCE = {'case': 'SRP1', 'candidate': 'x = 0',
            'candidate_key': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57',
            'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',
            'campaign': 's53_w101_srp1_cont_x0', 'certification_cycle': 181,
            'certified_models_sha256': MODELS['sha256']}
RECORD_TOTAL_EUR = 589.425936658749          # evaluation_record component_decomposition_totals_weighted


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _cite(module, needle):
    """file:line of the unique line containing `needle` in `module`'s source (resolved at run time)."""
    path = inspect.getsourcefile(module)
    with open(path) as handle:
        lines = handle.readlines()
    hits = [i + 1 for i, line in enumerate(lines) if needle in line]
    return {'needle': needle, 'where': [f'{os.path.relpath(path, REPO)}:{i}' for i in hits],
            'text': [lines[i - 1].rstrip() for i in hits]}


def look_agent(kind, planning, models, UB, srp, mch, defs, pe, component_blocks, helper_per_block):
    """Per-entry breakdown of the definitional curtailment for one agent kind ('TSO' or 'DSO')."""
    tol = float(defs.EQUALITY_TOLERANCE)
    negatives, per_block = [], {}
    agg = {'eur_at_1_block_weighted': 0.0, 'mwh_day_weighted': 0.0, 'mwh_rep_day_sum': 0.0,
           'neg_part_eur_at_1_block_weighted': 0.0, 'pos_part_eur_at_1_block_weighted': 0.0,
           'floor_eur_at_1_block_weighted': 0.0, 'n_entries': 0, 'n_negative': 0, 'n_below_minus_tol': 0,
           'n_pg_above_ub': 0, 'max_pg_minus_ub_pu': -float('inf'), 'n_avail_param_ne_data': 0,
           'max_abs_avail_param_minus_data_pu': 0.0, 'n_ub_ne_avail_plus_tol': 0, 'min_c_mwh': float('inf'),
           'max_c_mwh': -float('inf'), 'baseMVA': set(), 'max_abs_block_vs_helper': 0.0,
           'max_abs_block_vs_production_record': 0.0, 'n_blocks_compared_to_production_record': 0,
           'priced_net_c_eur_block_weighted': 0.0, 'priced_neg_part_eur_block_weighted': 0.0}
    for k, node_id, year, day, network_data, network, block in UB.iter_network_blocks(planning, models):
        if k != kind:
            continue
        label = UB.block_label(k, node_id, year, day)
        base = float(network.baseMVA)
        agg['baseMVA'].add(base)
        w_b = srp._get_admm_block_weight(network_data, year, day)
        d_b = float(network_data.years[year]) * float(network_data.days[day])
        rg_curt = bool(network_data.params.rg_curt)
        e_b, neg_b, pos_b, n_free = 0.0, 0.0, 0.0, 0
        gen_ids = {}
        for g in block.generators:
            gen = network.generators[g]
            if not gen.is_curtaillable():
                continue
            gen_ids[g] = gen.gen_id
            for s_m in block.scenarios_market:
                for s_o in block.scenarios_operation:
                    omega = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                    for p in block.periods:
                        var = block.pg[g, s_m, s_o, p]
                        pg = float(pe.value(var))
                        avail = float(pe.value(block.pg_avail[g, s_o, p]))
                        data = float(gen.pg[s_o][p]) if (gen.status[p] and not mch.renewable_generation_is_unavailable(gen, s_o, p)) else 0.0
                        ub = var.ub
                        lb = var.lb
                        c = base * (avail - pg) if rg_curt else 0.0
                        agg['n_entries'] += 1
                        if ub is not None and ub > 0.0:
                            n_free += 1
                        if abs(avail - data) > 0.0:
                            agg['n_avail_param_ne_data'] += 1
                        agg['max_abs_avail_param_minus_data_pu'] = max(agg['max_abs_avail_param_minus_data_pu'],
                                                                       abs(avail - data))
                        if ub is not None and ub > 0.0 and abs(ub - (avail + tol)) > 1e-15:
                            agg['n_ub_ne_avail_plus_tol'] += 1
                        if ub is not None:
                            agg['max_pg_minus_ub_pu'] = max(agg['max_pg_minus_ub_pu'], pg - ub)
                            if pg > ub:
                                agg['n_pg_above_ub'] += 1
                        price = float(network.cost_energy_p[s_m][p])
                        agg['min_c_mwh'] = min(agg['min_c_mwh'], c)
                        agg['max_c_mwh'] = max(agg['max_c_mwh'], c)
                        e_b += omega * c
                        agg['priced_net_c_eur_block_weighted'] += w_b * omega * price * c
                        if c < 0.0:
                            agg['priced_neg_part_eur_block_weighted'] += w_b * omega * price * c
                        if c < 0.0:
                            neg_b += omega * c
                            agg['n_negative'] += 1
                            if c < -tol * base * (1.0 + 1e-9):
                                agg['n_below_minus_tol'] += 1
                            if kind == 'TSO':
                                negatives.append({
                                    'block': label, 'year': year, 'day': day, 'unit_index': g,
                                    'gen_id': gen.gen_id, 'bus': gen.bus, 's_m': s_m, 's_o': s_o, 'hour': p,
                                    'omega': omega, 'pg_pu': pg, 'pg_avail_param_pu': avail,
                                    'availability_data_pu': data, 'pg_ub_pu': ub, 'pg_lb_pu': lb,
                                    'pg_minus_avail_pu': pg - avail, 'pg_minus_ub_pu': (pg - ub) if ub is not None else None,
                                    'c_mwh': c, 'c_over_tol_mw': c / (tol * base), 'price_eur_mwh': price,
                                    'qg_pu': float(pe.value(block.qg[g, s_m, s_o, p])),
                                    'sg_avail_pu': float(pe.value(block.sg_avail[g, s_o, p])),
                                    'sg_capability_row_present': (g, s_m, s_o, p) in block.sg_capability,
                                    'sg_capability_violation_pu2': (UB.constraint_violation(
                                        block.sg_capability[g, s_m, s_o, p])
                                        if (g, s_m, s_o, p) in block.sg_capability else None),
                                    'pg2_plus_qg2_minus_sg_avail2_pu2': (
                                        pg ** 2 + float(pe.value(block.qg[g, s_m, s_o, p])) ** 2
                                        - float(pe.value(block.sg_avail[g, s_o, p])) ** 2),
                                    'block_weight': w_b, 'eur_at_1_block_weighted': w_b * omega * c})
                        else:
                            pos_b += omega * c
        floor_b = -n_free * tol * base  # omega sums to 1 over scenarios; n_free counted over all scenarios
        n_scen = len(block.scenarios_market) * len(block.scenarios_operation)
        floor_b = floor_b / n_scen if n_scen else floor_b
        prod_label = f'{k}|{year}|{day}' if node_id is None else f'{k}|{node_id}|{year}|{day}'
        if prod_label not in component_blocks:
            raise RuntimeError(f'{prod_label} missing from component_levels_terminal.json (vacuous comparison)')
        rec_b = component_blocks[prod_label]['unweighted']['res_curtailment_definitional_at_weight_1']
        agg['n_blocks_compared_to_production_record'] += 1
        per_block[label] = {'mwh_rep_day': e_b, 'neg_part_mwh_rep_day': neg_b, 'pos_part_mwh_rep_day': pos_b,
                            'block_weight': w_b, 'day_weight': d_b, 'eur_at_1_block_weighted': w_b * e_b,
                            'n_curtaillable_units': len(gen_ids), 'curtaillable_gen_ids': gen_ids,
                            'baseMVA': base, 'rg_curt': rg_curt,
                            'penalty_gen_curtailment_param': float(pe.value(block.penalty_gen_curtailment)),
                            'total_gen_curt_penalty_expr': float(pe.value(block.total_gen_curt_penalty)),
                            'tol_floor_mwh_rep_day_approx': floor_b,
                            'production_record_unweighted': rec_b}
        if label not in helper_per_block:
            raise RuntimeError(f'{label} missing from curtailment_report per_block (vacuous comparison)')
        agg['max_abs_block_vs_helper'] = max(agg['max_abs_block_vs_helper'],
                                             abs(e_b - helper_per_block[label]['mwh_rep_day']))
        agg['max_abs_block_vs_production_record'] = max(agg['max_abs_block_vs_production_record'], abs(e_b - rec_b))
        agg['eur_at_1_block_weighted'] += w_b * e_b
        agg['mwh_day_weighted'] += d_b * e_b
        agg['mwh_rep_day_sum'] += e_b
        agg['neg_part_eur_at_1_block_weighted'] += w_b * neg_b
        agg['pos_part_eur_at_1_block_weighted'] += w_b * pos_b
        agg['floor_eur_at_1_block_weighted'] += w_b * floor_b
    agg['baseMVA'] = sorted(agg['baseMVA'])
    return agg, per_block, negatives


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, 'w109_tso_curtailment_look.json')
    if os.path.exists(out_json):
        raise RuntimeError(f'{out_json} exists; write-once')
    got = _sha(os.path.join(REPO, MODELS['path']))
    if got != MODELS['sha256']:
        raise RuntimeError(f"certified_models.pkl sha256 {got} != declared {MODELS['sha256']}")
    _log(f'certified_models.pkl sha256 verified {got}')

    import pyomo.environ as pe
    import definitions as defs
    import model_construction_helpers as mch
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    import network as network_module
    import json

    planning = O.load_baseline()['planning']
    _log('baseline planning loaded')
    with open(os.path.join(REPO, MODELS['path']), 'rb') as handle:
        payload = pickle.load(handle)
    _log('certified models unpickled')
    models = {'tso': payload['tso'], 'dso': payload['dso']}
    with open(os.path.join(REPO, EVAL_DIR, 'component_levels_terminal.json')) as handle:
        component_blocks = json.load(handle)['blocks']

    helper = UB.curtailment_report(planning, models)
    tso, tso_blocks, negatives = look_agent('TSO', planning, models, UB, srp, mch, defs, pe, component_blocks,
                                      helper['per_block'])
    dso, dso_blocks, _ = look_agent('DSO', planning, models, UB, srp, mch, defs, pe, component_blocks, helper['per_block'])
    _log(f"TSO {tso['eur_at_1_block_weighted']!r} EUR (helper {helper['totals']['TSO']['eur_at_1_block_weighted']!r}); "
         f"DSO {dso['eur_at_1_block_weighted']!r}; negatives {tso['n_negative']}, below -tol {tso['n_below_minus_tol']}")

    # production's own quantity: the certified record's per-block res_curtailment_definitional_at_weight_1 (the
    # campaign harness, p515_g_g1_g4_admm_gates._s31_block_components) and the Q term total_gen_curt_penalty
    prod_tso = sum(b['admm_block_weight'] * b['unweighted']['res_curtailment_definitional_at_weight_1']
                   for b in component_blocks.values() if b['kind'] == 'TSO')
    prod_dso = sum(b['admm_block_weight'] * b['unweighted']['res_curtailment_definitional_at_weight_1']
                   for b in component_blocks.values() if b['kind'] == 'DSO')
    prod_q_term = sum(b['admm_block_weight'] * b['unweighted']['res_curtailment_penalty']
                      for b in component_blocks.values())

    negatives.sort(key=lambda r: r['eur_at_1_block_weighted'])
    by_unit, by_block, by_hour = {}, {}, {}
    for r in negatives:
        by_unit[str(r['gen_id'])] = by_unit.get(str(r['gen_id']), 0.0) + r['eur_at_1_block_weighted']
        by_block[r['block']] = by_block.get(r['block'], 0.0) + r['eur_at_1_block_weighted']
        by_hour[str(r['hour'])] = by_hour.get(str(r['hour']), 0.0) + r['eur_at_1_block_weighted']

    citations = {
        'pg_bounds_curtaillable_ub': _cite(mch, 'return (0.0, gen.pg[s_o][p] + EQUALITY_TOLERANCE)'),
        'pg_avail_init': _cite(mch, 'pg_av = gen.pg[s_o][p]'),
        'pg_avail_param_decl': _cite(network_module, 'model.pg_avail = pe.Param('),
        'definitional_term': _cite(mch, 'total += weight * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])'),
        'penalty_term_in_objective': _cite(mch, 'gen_curt_penalty += penalty * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])'),
        'EQUALITY_TOLERANCE': _cite(defs, 'EQUALITY_TOLERANCE ='),
        'helper_call': _cite(UB, 'value += probability * float(pe.value(gen_curtailment_definitional_value('),
    }

    guard_failures = _GUARD.verify(0)
    result = {
        'stage': 'P5.15 W109 (Addendum 55) records-only look at the negative TSO curtailment', 'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script_sha256': _sha(THIS), 'instance': INSTANCE,
        'solve_profile_guard': {'permitted': [], 'verify_0': guard_failures, 'counts': dict(_GUARD.counts)},
        'objective_convention': 'curtailment in EUR at 1 EUR/MWh, block-weighted by admm_block_weight (years x days x '
                                'discount); MWh day-weighted = years x days undiscounted; not part of Q (see q_effect)',
        'equality_tolerance_pu': float(defs.EQUALITY_TOLERANCE),
        'helper_totals': helper['totals'],
        'record_total_eur_at_1_block_weighted': RECORD_TOTAL_EUR,
        'recomputed_total_minus_record_eur': tso['eur_at_1_block_weighted'] + dso['eur_at_1_block_weighted'] - RECORD_TOTAL_EUR,
        'tso': tso, 'dso': dso,
        'production_quantity': {
            'source': f'{EVAL_DIR}/component_levels_terminal.json blocks[*].unweighted.res_curtailment_definitional_at_weight_1 x admm_block_weight',
            'tso_eur_at_1_block_weighted': prod_tso, 'dso_eur_at_1_block_weighted': prod_dso,
            'q_term_res_curtailment_penalty_block_weighted': prod_q_term},
        'q_effect': {'tso_penalty_gen_curtailment_params': sorted({b['penalty_gen_curtailment_param'] for b in tso_blocks.values()}),
                     'dso_penalty_gen_curtailment_params': sorted({b['penalty_gen_curtailment_param'] for b in dso_blocks.values()}),
                     'tso_total_gen_curt_penalty_block_weighted': sum(b['block_weight'] * b['total_gen_curt_penalty_expr'] for b in tso_blocks.values()),
                     'dso_total_gen_curt_penalty_block_weighted': sum(b['block_weight'] * b['total_gen_curt_penalty_expr'] for b in dso_blocks.values())},
        'negatives_by_unit_eur': by_unit, 'negatives_by_block_eur': by_block, 'negatives_by_hour_eur': by_hour,
        'tso_per_block': tso_blocks, 'dso_per_block': dso_blocks,
        'tso_negative_entries': negatives,
        'citations': citations,
        'wall_s': time.time() - t0,
    }
    with open(out_json, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'wrote {out_json}; guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {result["wall_s"]:.1f} s')
    return 0 if not guard_failures else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    entries[MODELS['path'] + ' (hash-recorded input, not committed)'] = MODELS['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
