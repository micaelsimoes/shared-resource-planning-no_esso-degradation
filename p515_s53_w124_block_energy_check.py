"""P5.15 W124 (Addendum 57 order, benchmark follow-up) -- records-only check of the benchmark mechanism, all 12 blocks.
ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Nothing is built, unpickled or solved; only committed, hash-recorded JSON records are read (each pinned against the
manifest that hash-records it, and required git-tracked and clean against HEAD).

QUESTION. The W122 claim (price-taker NRF Q - coordinated Q181 = +90.9 MEUR) decomposes into TSO +70.17 M and DSO
+20.73 M. Advisor hypothesis H1: the price-taker DSOs import the same daily energy as coordinated but vacate the night /
early-morning hours; TN RES is then curtailed and the same energy re-procured from conventional units at pi_t. Checked
here block by block (year x representative day), from records only.

INPUTS (every value in the output carries its file and key):
  PT   = w116_benchmark_nrf/nrf_arm_price_taker_warm_from_certified_r2/nrf_arm_price_taker_warm_from_certified_r2.json
           phase_C_sequential_pass.dso_interface_schedule[node][year][day].p_mw[p]      (MW, DN side, hour = p + 1)
           phase_C_sequential_pass.evaluation.block_components['TSO|-|year|day']        (EUR, block-weighted)
           phase_C_sequential_pass.curtailment_signed_parts.per_block['TSO|-|year|day']
               .positive_part_mwh_rep_day, .net_mwh_rep_day, .block_weight, .day_weight
           phase_A.dso_interface_schedule / phase_A.dso_schedule_vs_coordinated_max_abs  (sign / unit cross-check only)
  LL   = w106_uncoordinated_settled/lambda_look/lambda_look.json
           table_coordinated[*].raw.E_dso_pu x table_coordinated[*].raw.B_dn  (coordinated DN-side interface P, MW)
           table_coordinated[*].pi                                            (pi_t, EUR/MWh; one value per hour)
  TCC  = w106_uncoordinated_settled/tso_coupling_check/tso_coupling_check.json
           per_block['TSO|-|year|day'].certified_coordinated.block_gross_cost_weighted  (EUR, block-weighted)
  W114 = w114_passive_infeasibility/w114_passive_infeasibility.json
           schedule_2035_summer_passive_vs_coordinated.totals[*].coord_dso_p_total_mw and .per_node[n][*].coord_dso_p_mw
           (unit check of E_dso_pu x B_dn only)
  R3   = w116_benchmark_nrf/report_v3/report_v3.json
           curtailment_table.coordinated.settled_cycle_181_signed_parts_from_models  (coordinated TN curtailment totals;
           context only -- the coordinated per-block TN curtailment is not in these records)

FORMULAS (all per block b = (year, day); n in {5, 7, 9}; h = 1..24):
  E_pt[n,b]   = sum_h PT p_mw[n,b,h]                          MWh per representative day
  E_co[n,b]   = sum_h LL E_dso_pu[n,b,h] x B_dn[n,b,h]        MWh per representative day
  E_x[b]      = sum_n E_x[n,b]
  rel[.]      = (E_pt - E_co) / E_co
  dP[n,b,h]   = PT p_mw - LL E_dso_pu x B_dn  (MW);  dP[b,h] = sum_n dP[n,b,h]; hours with |dP[b,h]| > 1 MW listed with sign
  dTSO[b]     = PT block_components['TSO|-|b'] - TCC certified_coordinated.block_gross_cost_weighted     (EUR)
  C_pt[b]     = PT curtailment per_block['TSO|-|b'].positive_part_mwh_rep_day x block_weight   (MWh-equivalent at
                1 EUR/MWh block-weighted -- the same weighting as the EUR block costs)
  ratio[b]    = dTSO[b] / C_pt[b]                              EUR/MWh (undefined if C_pt[b] <= 0)
  pi range[b] = [min_h pi, max_h pi] over LL rows of block b
  night[b]    = sum_{h in 1..6, 22..24} (E_co - E_pt)[b,h]    MWh per rep day (positive = price-taker imports LESS)
  day_rest[b] = the same over h = 7..21
  aggregates  = plain sum over the 12 representative days, and day-weighted (x per_block['TSO|-|b'].day_weight)

VERDICT RULE (Planner W124, fixed here before anything is computed; see VERDICT_RULE below).
  Reading R_total (PRIMARY): "import energies agree" = |rel(E_pt[b], E_co[b])| <= 1 % on the three-node total.
  Reading R_node (beside it): every node's |rel(E_pt[n,b], E_co[n,b])| <= 1 %.  Blocks whose verdict differs are listed.
  H1            energies agree AND C_pt[b] > 0 AND pi_min[b] <= ratio[b] <= pi_max[b]
  energy/loss   energies differ by more than 1 %
  undetermined  energies agree but the ratio is undefined (a TSO-cost / curtailment record missing, or C_pt[b] <= 0).
                The import energies themselves are always computable here (864 LL rows and the PT schedule asserted).
  outside_rule  energies agree, ratio defined, ratio outside the pi range -- a case the Planner rule does not name; it is
                reported as such, not forced into a class.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w124_block_energy_check/):
  w124_block_energy_check.json, launch.log, manifest_sha256.json
Smoke (no file written): add --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w124_block_energy_check && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w124_block_energy_check.py \\
        > data/SRP1/Results/P515S53/w124_block_energy_check/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w124_block_energy_check.py --manifest
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W124 block energy check zero-solve').install()

import gate_result_io as GRIO  # noqa: E402 -- stdlib only

# ---- the verdict rule, fixed before any computation --------------------------------------------------------------
ENERGY_REL_TOL = 0.01          # 1 %
HOUR_DIFF_MW = 1.0             # hourly profile difference reported above 1 MW
NIGHT_HOURS = [1, 2, 3, 4, 5, 6, 22, 23, 24]
VERDICT_RULE = {
    'source': 'Planner task W124, item 4',
    'energy_agree': f'|E_pt - E_co| / E_co <= {ENERGY_REL_TOL} (reading R_total, PRIMARY: three-node total; reading '
                    'R_node: every node)',
    'H1': 'energies agree AND C_pt > 0 AND pi_min <= dTSO / C_pt <= pi_max',
    'energy/loss': 'energies differ by more than 1 %',
    'undetermined': 'energies agree but the ratio is undefined (a TSO-cost / curtailment record missing, or C_pt <= 0)',
    'outside_rule': 'energies agree, ratio defined, ratio outside [pi_min, pi_max] -- not named by the Planner rule',
}

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(REPO, RES, 'w124_block_energy_check')
OUT_JSON = os.path.join(OUT_DIR, 'w124_block_energy_check.json')

PT_RUN = 'nrf_arm_price_taker_warm_from_certified_r2'
PT = os.path.join(RES, 'w116_benchmark_nrf', PT_RUN, f'{PT_RUN}.json')
LL = os.path.join(RES, 'w106_uncoordinated_settled', 'lambda_look', 'lambda_look.json')
TCC = os.path.join(RES, 'w106_uncoordinated_settled', 'tso_coupling_check', 'tso_coupling_check.json')
W114 = os.path.join(RES, 'w114_passive_infeasibility', 'w114_passive_infeasibility.json')
R3 = os.path.join(RES, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
Q181 = 653873702.1876609       # LL coordinated_cell.certified_gross (asserted below)
NODES = ['5', '7', '9']
YEARS = ['2025', '2030', '2035']
DAYS = ['Winter', 'Spring', 'Summer', 'Autumn']


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


INPUTS = {}


def _pin(rel):
    """sha256 of an input against its directory's manifest_sha256.json (flat or {files: {...}} form); it must also be
    git-tracked and clean against HEAD. Raise otherwise."""
    manifest_rel = os.path.join(os.path.dirname(rel), 'manifest_sha256.json')
    now = _sha(os.path.join(REPO, rel))
    man = _load(manifest_rel)
    pinned = man.get(rel)
    if pinned is None and isinstance(man.get('files'), dict):
        pinned = man['files'].get(rel)
        pinned = pinned.get('sha256') if isinstance(pinned, dict) else pinned
    if pinned is None:
        raise RuntimeError(f'{rel}: not hash-recorded in {manifest_rel}')
    if pinned != now:
        raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned} ({manifest_rel})')
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], capture_output=True, cwd=REPO).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise RuntimeError(f'{rel}: tracked {tracked}, clean {clean}')
    commit = subprocess.run(['git', 'log', '-1', '--format=%H', '--', rel], capture_output=True, text=True,
                            cwd=REPO).stdout.strip()
    INPUTS[rel] = {'sha256': now, 'manifest': manifest_rel, 'git_last_commit': commit}


def _rel(a, b):
    return (a - b) / b if b != 0 else None


def main():
    dry = '--dry' in sys.argv
    t0 = time.time()
    if not dry and os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    for rel in (PT, LL, TCC, W114, R3):
        _pin(rel)
    _log(f'{len(INPUTS)} inputs pinned')

    pt = _load(PT)
    ll = _load(LL)
    tcc = _load(TCC)
    w114 = _load(W114)
    r3 = _load(R3)

    # ---- coordinated rows (LL) -------------------------------------------------------------------------------------
    assert ll['coordinated_cell']['certified_gross'] == Q181, ll['coordinated_cell']['certified_gross']
    rows = ll['table_coordinated']
    assert len(rows) == 864, len(rows)
    co = {}     # (node, year, day, hour) -> MW
    pi = {}     # (year, day, hour) -> EUR/MWh
    for r in rows:
        key = (str(r['node_id']), str(r['year']), r['day'], int(r['hour']))
        assert key not in co, key
        assert int(r['hour']) == int(r['period']) + 1
        co[key] = r['raw']['E_dso_pu'] * r['raw']['B_dn']
        pk = (str(r['year']), r['day'], int(r['hour']))
        if pk in pi:
            assert pi[pk] == r['pi'], ('pi differs across nodes', pk)
        pi[pk] = r['pi']
    assert len(co) == 864 and len(pi) == 288

    # ---- unit check against W114 (2035 Summer coordinated) ----------------------------------------------------------
    s114 = w114['schedule_2035_summer_passive_vs_coordinated']
    w114_total = sum(x['coord_dso_p_total_mw'] for x in s114['totals'])
    ll_total = sum(co[(n, '2035', 'Summer', h)] for n in NODES for h in range(1, 25))
    per_node_max = {n: max(abs(x['coord_dso_p_mw'] - co[(n, '2035', 'Summer', int(x['hour']))])
                           for x in s114['per_node'][n]) for n in NODES}
    unit_check = {
        'w114_coord_dso_p_total_mwh_day': w114_total,
        'w114_source': f'{W114} schedule_2035_summer_passive_vs_coordinated.totals[*].coord_dso_p_total_mw (sum)',
        'll_E_dso_pu_x_B_dn_mwh_day': ll_total,
        'll_source': f'{LL} table_coordinated[year 2035, day Summer].raw.E_dso_pu x raw.B_dn (sum)',
        'abs_diff_mwh_day': abs(w114_total - ll_total),
        'per_node_hourly_max_abs_diff_mw': per_node_max,
        'B_dn_values': sorted({r['raw']['B_dn'] for r in rows}),
        'passed': abs(w114_total - ll_total) <= 1e-9 * abs(w114_total) and max(per_node_max.values()) <= 1e-9,
    }
    _log(f'unit check: W114 {w114_total:.6f} vs LL {ll_total:.6f} MWh/day, per-node hourly max {per_node_max}; '
         f'passed {unit_check["passed"]}')

    # ---- price-taker schedules, sign cross-check on phase A --------------------------------------------------------
    pc = pt['phase_C_sequential_pass']
    sch_c = pc['dso_interface_schedule']
    sch_a = pt['phase_A']['dso_interface_schedule']
    pa_max = max(abs(sch_a[n][y][d]['p_mw'][h - 1] - co[(n, y, d, h)])
                 for n in NODES for y in YEARS for d in DAYS for h in range(1, 25))
    pa_rec = pt['phase_A']['dso_schedule_vs_coordinated_max_abs']['p_mw']
    sign_check = {
        'recomputed_max_abs_phaseA_p_mw_minus_ll_mw': pa_max,
        'recorded': pa_rec,
        'recorded_source': f'{PT} phase_A.dso_schedule_vs_coordinated_max_abs.p_mw',
        'abs_diff': abs(pa_max - pa_rec),
        'passed': abs(pa_max - pa_rec) <= 1e-9 * max(1.0, abs(pa_rec)),
        'meaning': 'the price-taker DN-side p_mw and LL E_dso_pu x B_dn share sign convention and unit iff the '
                   'recomputed max equals the stage-recorded one',
    }
    _log(f'sign/unit cross-check phase A: recomputed {pa_max!r} recorded {pa_rec!r} passed {sign_check["passed"]}')

    bc = pc['evaluation']['block_components']
    cpb = pc['curtailment_signed_parts']['per_block']
    tcc_pb = tcc['per_block']

    blocks = {}
    agg_night = {'plain_mwh': 0.0, 'day_weighted_mwh': 0.0}
    agg_rest = {'plain_mwh': 0.0, 'day_weighted_mwh': 0.0}
    agg_hour = {h: 0.0 for h in range(1, 25)}
    agg_hour_node = {n: {h: 0.0 for h in NIGHT_HOURS} for n in NODES}
    d_tso_sum = 0.0
    for y in YEARS:
        for d in DAYS:
            b = f'{y}|{d}'
            tkey = f'TSO|-|{y}|{d}'
            missing = []
            e_pt = {n: sum(sch_c[n][y][d]['p_mw']) for n in NODES}
            e_co = {n: sum(co[(n, y, d, h)] for h in range(1, 25)) for n in NODES}
            e_pt_t, e_co_t = sum(e_pt.values()), sum(e_co.values())
            dp_node = {n: [sch_c[n][y][d]['p_mw'][h - 1] - co[(n, y, d, h)] for h in range(1, 25)] for n in NODES}
            dp = [sum(dp_node[n][h - 1] for n in NODES) for h in range(1, 25)]
            hours_over = [{'hour': h, 'pt_minus_co_mw': dp[h - 1],
                           'direction': 'price_taker_imports_more' if dp[h - 1] > 0 else 'price_taker_imports_less'}
                          for h in range(1, 25) if abs(dp[h - 1]) > HOUR_DIFF_MW]
            hours_over_node = {n: [{'hour': h, 'pt_minus_co_mw': dp_node[n][h - 1]} for h in range(1, 25)
                                   if abs(dp_node[n][h - 1]) > HOUR_DIFF_MW] for n in NODES}
            night = sum(-dp[h - 1] for h in NIGHT_HOURS)
            rest = sum(-dp[h - 1] for h in range(1, 25) if h not in NIGHT_HOURS)
            night_node = {n: sum(-dp_node[n][h - 1] for h in NIGHT_HOURS) for n in NODES}
            pis = [pi[(y, d, h)] for h in range(1, 25)]
            pi_min, pi_max = min(pis), max(pis)

            cost_pt = bc.get(tkey)
            cost_co = (tcc_pb.get(tkey) or {}).get('certified_coordinated', {}).get('block_gross_cost_weighted')
            curt = cpb.get(tkey)
            if cost_pt is None:
                missing.append('PT block_components')
            if cost_co is None:
                missing.append('TCC block_gross_cost_weighted')
            if curt is None:
                missing.append('PT curtailment per_block')
            d_tso = cost_pt - cost_co if (cost_pt is not None and cost_co is not None) else None
            if d_tso is not None:
                d_tso_sum += d_tso
            c_pt = curt['positive_part_mwh_rep_day'] * curt['block_weight'] if curt is not None else None
            ratio = d_tso / c_pt if (d_tso is not None and c_pt is not None and c_pt > 0) else None
            day_weight = curt['day_weight'] if curt is not None else None

            rel_t = _rel(e_pt_t, e_co_t)
            rel_n = {n: _rel(e_pt[n], e_co[n]) for n in NODES}
            agree_total = rel_t is not None and abs(rel_t) <= ENERGY_REL_TOL
            agree_node = all(v is not None and abs(v) <= ENERGY_REL_TOL for v in rel_n.values())

            def verdict(agree):
                if not agree:
                    return 'energy/loss'
                if ratio is None:          # a TSO-cost or curtailment record missing, or C_pt <= 0
                    return 'undetermined'
                if pi_min <= ratio <= pi_max:
                    return 'H1'
                return 'outside_rule'

            blocks[b] = {
                'year': y, 'day': d,
                'import_energy_mwh_rep_day': {
                    'price_taker_phase_C': {'per_node': e_pt, 'total': e_pt_t},
                    'coordinated': {'per_node': e_co, 'total': e_co_t},
                    'rel_diff_total': rel_t, 'rel_diff_per_node': rel_n,
                    'pt_minus_co_total_mwh': e_pt_t - e_co_t},
                'hours_abs_diff_over_1mw_total': hours_over,
                'hours_abs_diff_over_1mw_per_node': hours_over_node,
                'hourly_pt_minus_co_total_mw': dp,
                'night_vacating_co_minus_pt_mwh_rep_day': {'total': night, 'per_node': night_node},
                'rest_of_day_co_minus_pt_mwh_rep_day': rest,
                'pi_range_eur_mwh': [pi_min, pi_max],
                'tso_block_cost_eur': {'price_taker_phase_C': cost_pt, 'coordinated': cost_co, 'delta': d_tso},
                'tn_curtailment_price_taker_phase_C': None if curt is None else {
                    'positive_part_mwh_rep_day': curt['positive_part_mwh_rep_day'],
                    'net_mwh_rep_day': curt['net_mwh_rep_day'], 'block_weight': curt['block_weight'],
                    'day_weight': curt['day_weight'], 'C_pt_block_weighted_mwh_eq': c_pt},
                'ratio_dtso_per_curtailed_mwh_eq_eur_mwh': ratio,
                'energy_agree_R_total': agree_total, 'energy_agree_R_node': agree_node,
                'verdict_R_total': verdict(agree_total), 'verdict_R_node': verdict(agree_node),
                'missing_records': missing,
            }
            agg_night['plain_mwh'] += night
            agg_rest['plain_mwh'] += rest
            if day_weight is not None:
                agg_night['day_weighted_mwh'] += night * day_weight
                agg_rest['day_weighted_mwh'] += rest * day_weight
            for h in range(1, 25):
                agg_hour[h] += -dp[h - 1]
            for n in NODES:
                for h in NIGHT_HOURS:
                    agg_hour_node[n][h] += -dp_node[n][h - 1]

    q_c = pc['evaluation']['gross_operational_cost']
    decomposition = {
        'q_price_taker_phase_C': q_c, 'q_source': f'{PT} phase_C_sequential_pass.evaluation.gross_operational_cost',
        'q181_coordinated': Q181, 'q181_source': f'{LL} coordinated_cell.certified_gross',
        'claim': q_c - Q181, 'dTSO_sum_12_blocks': d_tso_sum, 'dDSO_residual': q_c - Q181 - d_tso_sum,
    }
    counts = {}
    for reading in ('verdict_R_total', 'verdict_R_node'):
        counts[reading] = {}
        for v in blocks.values():
            counts[reading][v[reading]] = counts[reading].get(v[reading], 0) + 1
    differ = [b for b, v in blocks.items() if v['verdict_R_total'] != v['verdict_R_node']]
    coord_tn = r3['curtailment_table']['coordinated']['settled_cycle_181_signed_parts_from_models']
    summary = {
        'n_blocks': len(blocks), 'verdict_counts': counts, 'blocks_where_readings_differ': differ,
        'night_hours': NIGHT_HOURS,
        'night_vacating_co_minus_pt_mwh': agg_night,
        'rest_of_day_co_minus_pt_mwh': agg_rest,
        'per_hour_of_day_co_minus_pt_mwh_sum_12_rep_days': agg_hour,
        'per_node_night_hour_co_minus_pt_mwh_sum_12_rep_days': agg_hour_node,
        'aggregation': 'plain = sum over the 12 representative days (MWh); day_weighted = x PT curtailment '
                       'per_block[TSO|-|b].day_weight (undiscounted days represented)',
        'coordinated_tn_curtailment_context': {
            'positive_part_TSO': coord_tn['positive_part']['by_agent']['TSO'],
            'net_TSO': coord_tn['net']['by_agent']['TSO'],
            'source': f'{R3} curtailment_table.coordinated.settled_cycle_181_signed_parts_from_models.<part>.by_agent.TSO',
            'note': 'totals only; the coordinated per-block TN curtailment is not in the records read here'},
        'price_taker_tn_curtailment_totals': pc['curtailment_signed_parts']['totals']['TSO'],
    }

    guard_failures = _GUARD.verify(0)
    out = {
        'stage': 'P5.15 W124 -- records-only block energy / curtailment check of the NRF benchmark mechanism '
                 '(zero solves, no model loads)',
        'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script_sha256': _sha(THIS),
        'instance': {'price_taker': pt.get('instance'), 'coordinated': ll.get('instance'),
                     'coordinated_eval_key': ll['coordinated_cell']['eval_key'],
                     'coordinated_certification_cycle': ll['coordinated_cell']['certification_cycle']},
        'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED; TSO block costs block-weighted EUR',
        'verdict_rule': VERDICT_RULE,
        'sources': {
            'E_pt': f'{PT} phase_C_sequential_pass.dso_interface_schedule[node][year][day].p_mw',
            'E_co': f'{LL} table_coordinated[*].raw.E_dso_pu x raw.B_dn',
            'pi': f'{LL} table_coordinated[*].pi',
            'tso_cost_pt': f'{PT} phase_C_sequential_pass.evaluation.block_components[TSO|-|year|day]',
            'tso_cost_co': f'{TCC} per_block[TSO|-|year|day].certified_coordinated.block_gross_cost_weighted',
            'C_pt': f'{PT} phase_C_sequential_pass.curtailment_signed_parts.per_block[TSO|-|year|day]'
                    '.positive_part_mwh_rep_day x .block_weight',
        },
        'inputs_sha256': INPUTS,
        'unit_check_w114': unit_check,
        'sign_unit_check_phase_A': sign_check,
        'decomposition': decomposition,
        'blocks': blocks,
        'summary': summary,
        'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures, 'counts': dict(_GUARD.counts)},
        'wall_s': time.time() - t0,
    }
    if dry:
        GRIO.dumps(out, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log('--dry: nothing written')
    else:
        with open(OUT_JSON, 'x') as handle:
            GRIO.dump(out, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log(f'wrote {OUT_JSON} sha256 {_sha(OUT_JSON)}')

    # ---- log table ---------------------------------------------------------------------------------------------------
    _log(f'decomposition: {json.dumps(decomposition)}')
    _log('block        E_pt(MWh)   E_co(MWh)   rel_tot   rel_n5    rel_n7    rel_n9    dTSO(EUR)      C_pt(MWh-eq) '
         'ratio  pi_min  pi_max  night(MWh) hrs>1MW  R_total      R_node')
    for b, v in blocks.items():
        ie = v['import_energy_mwh_rep_day']
        rn = ie['rel_diff_per_node']
        c = v['tn_curtailment_price_taker_phase_C']
        ratio = v['ratio_dtso_per_curtailed_mwh_eq_eur_mwh']
        _log(f"{b:12s} {ie['price_taker_phase_C']['total']:10.3f} {ie['coordinated']['total']:10.3f} "
             f"{ie['rel_diff_total']:+.4%} {rn['5']:+.4%} {rn['7']:+.4%} {rn['9']:+.4%} "
             f"{v['tso_block_cost_eur']['delta']:14.2f} {c['C_pt_block_weighted_mwh_eq']:12.2f} "
             f"{ratio if ratio is None else round(ratio, 2)!s:>7} {v['pi_range_eur_mwh'][0]:7.2f} "
             f"{v['pi_range_eur_mwh'][1]:7.2f} {v['night_vacating_co_minus_pt_mwh_rep_day']['total']:10.3f} "
             f"{len(v['hours_abs_diff_over_1mw_total']):3d}      {v['verdict_R_total']:12s} {v['verdict_R_node']}")
        _log(f"    hours |dP|>1MW (h: pt-co MW): "
             + ', '.join(f"{x['hour']}:{x['pt_minus_co_mw']:+.2f}" for x in v['hours_abs_diff_over_1mw_total']))
    _log(f'verdict counts {counts}; readings differ on {differ}')
    _log(f'night vacating (co - pt) {agg_night}; rest of day {agg_rest}')
    _log('per hour of day (co - pt, MWh, sum 12 rep days): '
         + ', '.join(f'{h}:{agg_hour[h]:+.2f}' for h in range(1, 25)))
    _log(f'guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {time.time() - t0:.1f} s')
    return 0 if not guard_failures and unit_check['passed'] and sign_check['passed'] else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    res = json.load(open(OUT_JSON))
    for rel, v in res['inputs_sha256'].items():
        entries[rel + ' (input)'] = v['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else main())
