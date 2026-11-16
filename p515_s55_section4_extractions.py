"""P5.15 Section 4 revision -- zero-solve extractions (author reopened the numbers freeze for these items only).

Requested by the Expert's handback of 2026-11-16 (Overleaf e315b07): placeholders a1, a2, b1, b2; the penalty-sensitivity
cost levels (c); checks (d)-(f) on numbers already in the text. RECORDS ONLY: this script reads committed JSON/CSV
records. It imports no production module, builds no model, loads no pickle (pickle.load/loads are blocked for the whole
run and the block is checked at exit: permitted 0, observed 0).

Every input is sha256-hashed and its git-tracked status recorded. Every value carries its source file and convention.

Conventions
  Q                gross_operational_cost (contracted interface settlement excluded), discounted, block-weighted:
                   recourse_components.gross_operational_cost of component_levels_terminal.json.
  component diffs  totals_weighted[key](x=0) - totals_weighted[key](x); 'weighted' = per-block scenario-probability
                   weighted value x admm_block_weight (years x days x discount annualisation) -- the weight Q uses.
                   Identity checked: Q = generation + flexibility_internal + load_curtailment + res_curtailment_penalty
                   + ess_usage + detector_penalty_total (detector = voltage slack + day-balance slacks).
  by-agent diffs   sum of the per-block weighted Q parts over TSO blocks / DSO blocks (the W124 / Section 4.1 77.2 %
                   convention, which is by agent).
  curtailment      W184 / Appendix B.3 formula: per agent, sum over its blocks of the UNWEIGHTED
                   res_curtailment_definitional_at_weight_1 (MWh per representative day, scenario-probability
                   weighted, ACTIVE energy: (pg_avail - pg) x baseMVA per 1 h period, net signed) x Years[y] x Days[d]
                   from data/SRP1/SRP1.json; undiscounted; / 1000 -> GWh.
  tolerance band   RES output is bounded by availability + EQUALITY_TOLERANCE (1e-5 pu = 1e-3 MW per generator-hour)
                   (Addendum 56). Its day-weighted floor per agent is taken from the W109 record (x = 0, per-block
                   tol_floor_mwh_rep_day_approx x day weight).

Usage: <canonical python> -u p515_s55_section4_extractions.py --out <new json path>   (refuses to overwrite)
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone

# ------------------------------------------------------------------ pickle block (armed for the whole run)
_PICKLE_CALLS = {'load': 0, 'loads': 0}


def _blocked(kind):
    def f(*_a, **_k):
        _PICKLE_CALLS[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked: this extraction is records-only')
    return f


pickle.load = _blocked('load')
pickle.loads = _blocked('loads')

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
R53 = 'data/SRP1/Results/P515S53'
R55 = 'data/SRP1/Results/P515S55'
FROZEN = f'{R53}/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json'
SRP1 = 'data/SRP1/SRP1.json'
W109 = f'{R53}/w109_tso_curtailment/w109_tso_curtailment_look.json'
W155 = f'{R53}/w155_a64_cells/w155_summary_after_04_e_soh050.json'
W188D_SOH = f'{R55}/w188d_operational_figure/soh_per_block_calibrations.csv'

EVALS = {
    # base flexibility price, default initial penalties (frozen cells ref:7aa017f0 / ref:bd504ecf)
    'x0_1x': (f'{R53}/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0',
              'ref:7aa017f0'),
    'unit_1x': (f'{R53}/w101_srp1_continuation/campaign_s53_w101_srp1_cont_n7_4h_e1/evals/3f084f2ffaeef2b7_n7_4h_e1',
                'ref:bd504ecf'),
    # twice the base flexibility price, default initial penalties (frozen cells h_50dea31c / h_74eda68d; claim H:m2)
    'x0_2x': (f'{R53}/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_50dea31c/evals/354b1bdae2b86bce_h_50dea31c',
              'h_50dea31c'),
    'unit_2x': (f'{R53}/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_74eda68d/evals/dca0862812873604_h_74eda68d',
                'h_74eda68d'),
}
for _s in ('1x', '2x'):
    for _r in ('0p5', '2p0'):
        for _c in ('x0', 'unit'):
            _camp = f'{R55}/e2_rho/campaign_s55_step7_e2_{_c}_{_s}_rho{_r}'
            _ev = os.listdir(f'{_camp}/evals')
            assert len(_ev) == 1, (_camp, _ev)
            EVALS[f'{_c}_{_s}_rho{_r}'] = (f'{_camp}/evals/{_ev[0]}', None)

Q_PARTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
           'ess_usage_cost', 'detector_penalty_total')
DETECTOR_SUB = ('voltage_slack', 'node_balance_slack', 'branch_flow_slack', 'flexibility_p_day_balance_slack',
                'local_ess_day_balance_slack', 'shared_ess_day_balance_slack')
INPUTS = {}


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _tracked(p):
    r = subprocess.run(['git', 'ls-files', '--error-unmatch', p], capture_output=True, text=True)
    return r.returncode == 0


def _reg(p):
    if p not in INPUTS:
        INPUTS[p] = {'sha256': _sha(p), 'git_tracked': _tracked(p)}
    return p


def _json(p):
    with open(_reg(p), encoding='utf-8') as f:
        return json.load(f)


def load_eval(name):
    d, frozen_cell = EVALS[name]
    cl = _json(f'{d}/component_levels_terminal.json')
    er = _json(f'{d}/evaluation_record.json')
    rc = cl['recourse_components']
    tw = cl['totals_weighted']
    q = rc['gross_operational_cost']
    ident = sum(tw[k] for k in Q_PARTS)
    det = sum(tw[k] for k in DETECTOR_SUB)
    agent = {'TSO': 0.0, 'DSO': 0.0}
    for b in cl['blocks'].values():
        agent[b['kind']] += sum(b['weighted'][k] for k in Q_PARTS)
    return {'dir': d, 'frozen_cell': frozen_cell, 'cl': cl, 'er': er, 'Q': q,
            'Q_identity_residual': q - ident, 'detector_sub_residual': tw['detector_penalty_total'] - det,
            'agent_Q': agent, 'agent_residual': q - agent['TSO'] - agent['DSO'],
            'status': er.get('status'), 'certified_cost': er.get('certified_cost'),
            'certification_cycle': er.get('certification_cycle'), 'cycles_run_record': er.get('cycles_run'),
            'cycles_run_components': cl.get('cycles_run'), 'terminal_gross': er.get('terminal_gross_operational_cost'),
            'eval_key': er.get('eval_key') or os.path.basename(d).split('_')[0]}


def frozen_cell(frozen, cell):
    t = frozen['tables']
    return t['cells'].get(cell) or t['cells_appended_w160'].get(cell)


def part_a(E, frozen):
    out = {}
    for price in ('1x', '2x'):
        x0, un = E[f'x0_{price}'], E[f'unit_{price}']
        tw0, tw1 = x0['cl']['totals_weighted'], un['cl']['totals_weighted']
        keys = [k for k in tw0 if k != 'res_curtailment_definitional_at_weight_1']
        diff = {k: tw0[k] - tw1[k] for k in keys}
        dq = x0['Q'] - un['Q']
        parts_sum = sum(diff[k] for k in Q_PARTS)
        nonzero = {k: v for k, v in diff.items() if v != 0.0}
        dagent = {a: x0['agent_Q'][a] - un['agent_Q'][a] for a in ('TSO', 'DSO')}
        out[price] = {
            'x0': x0['dir'], 'unit': un['dir'],
            'Q_x0': x0['Q'], 'Q_unit': un['Q'], 'operating_value_dQ': dq,
            'component_diff_weighted_eur': diff, 'nonzero_component_diffs_eur': nonzero,
            'sum_of_Q_parts_diff_eur': parts_sum, 'residual_dQ_minus_parts_eur': dq - parts_sum,
            'share_generation_pct': 100.0 * diff['generation_cost'] / dq,
            'share_flexibility_internal_pct': 100.0 * diff['flexibility_cost_internal'] / dq,
            'share_other_pct': 100.0 * (dq - diff['generation_cost'] - diff['flexibility_cost_internal']) / dq,
            'by_agent_diff_eur': dagent,
            'share_TSO_agent_pct': 100.0 * dagent['TSO'] / dq, 'share_DSO_agent_pct': 100.0 * dagent['DSO'] / dq,
            'frozen_Q_x0': frozen_cell(frozen, x0['frozen_cell'])['Q'],
            'frozen_Q_unit': frozen_cell(frozen, un['frozen_cell'])['Q'],
        }
        out[price]['Q_equals_frozen'] = (out[price]['frozen_Q_x0'] == x0['Q'] and out[price]['frozen_Q_unit'] == un['Q'])
        sens = {}
        for s_tag, tag in (('0.5', f'{price}_rho0p5'), ('1.0', price), ('2.0', f'{price}_rho2p0')):
            a0, a1 = E[f'x0_{tag}']['cl']['totals_weighted'], E[f'unit_{tag}']['cl']['totals_weighted']
            dqs = E[f'x0_{tag}']['Q'] - E[f'unit_{tag}']['Q']
            g = a0['generation_cost'] - a1['generation_cost']
            fl = a0['flexibility_cost_internal'] - a1['flexibility_cost_internal']
            sens[s_tag] = {'dQ': dqs, 'generation_diff': g, 'flexibility_diff': fl, 'other_diff': dqs - g - fl,
                           'share_generation_pct': 100.0 * g / dqs, 'share_flexibility_pct': 100.0 * fl / dqs}
        out[price]['split_across_initial_penalty_scalings_W185'] = sens
        gs = [v['share_generation_pct'] for v in sens.values()]
        out[price]['split_range_generation_pct'] = [min(gs), max(gs)]
    return out


def curtailment(cl, srp1):
    gwh = {'TSO': 0.0, 'DSO': 0.0}
    for b in cl['blocks'].values():
        u = b['unweighted']['res_curtailment_definitional_at_weight_1']
        gwh[b['kind']] += u * srp1['Years'][str(b['year'])] * srp1['Days'][str(b['day'])] / 1000.0
    return gwh


def part_b(E, srp1, w109):
    floor = {'TSO': 0.0, 'DSO': 0.0}
    for agent, key in (('TSO', 'tso_per_block'), ('DSO', 'dso_per_block')):
        for bk, b in w109[key].items():
            parts = bk.split('|')
            y, d = parts[2], parts[3]
            floor[agent] += b['tol_floor_mwh_rep_day_approx'] * srp1['Years'][y] * srp1['Days'][d] / 1000.0
    parts = {}
    for agent, key in (('TSO', 'tso_per_block'), ('DSO', 'dso_per_block')):
        pos = neg = 0.0
        for bk, b in w109[key].items():
            y, d = bk.split('|')[2], bk.split('|')[3]
            wgt = srp1['Years'][y] * srp1['Days'][d] / 1000.0
            pos += b['pos_part_mwh_rep_day'] * wgt
            neg += b['neg_part_mwh_rep_day'] * wgt
        parts[agent] = {'positive_part_gwh': pos, 'negative_part_gwh': neg, 'net_gwh': pos + neg}
    out = {'tolerance_floor_gwh_day_weighted_x0_W109': floor,
           'x0_positive_negative_parts_W109': parts,
           'w109_check_mwh_day_weighted': {'TSO': w109['tso']['mwh_day_weighted'],
                                           'DSO': w109['dso']['mwh_day_weighted']}}
    for price in ('1x', '2x'):
        c0 = curtailment(E[f'x0_{price}']['cl'], srp1)
        c1 = curtailment(E[f'unit_{price}']['cl'], srp1)
        out[price] = {'x0_gwh': c0, 'unit_gwh': c1,
                      'diff_x0_minus_unit_gwh': {a: c0[a] - c1[a] for a in c0}}
    return out


def part_c(E):
    out = {}
    for price in ('1x', '2x'):
        rows = {}
        for s, tag in (('0.5', f'{price}_rho0p5'), ('1.0', price), ('2.0', f'{price}_rho2p0')):
            x0, un = E[f'x0_{tag}'], E[f'unit_{tag}']
            rows[s] = {'Q_x0': x0['Q'], 'Q_unit': un['Q'], 'dQ': x0['Q'] - un['Q'],
                       'status_x0': x0['status'], 'status_unit': un['status'],
                       'dir_x0': x0['dir'], 'dir_unit': un['dir']}
        qs = [r['Q_x0'] for r in rows.values()]
        qu = [r['Q_unit'] for r in rows.values()]
        dq = [r['dQ'] for r in rows.values()]
        out[price] = {'rows': rows,
                      'spread_Q_x0_eur': max(qs) - min(qs), 'spread_Q_x0_rel': (max(qs) - min(qs)) / min(qs),
                      'spread_Q_unit_eur': max(qu) - min(qu), 'spread_Q_unit_rel': (max(qu) - min(qu)) / min(qu),
                      'spread_dQ_eur': max(dq) - min(dq)}
    return out


def part_d(frozen):
    rows = {r['cell']: r for r in frozen['tables']['ageing']['rows']}
    v_no, v_cyc, v_base = rows['e_no_ageing']['value'], rows['e_c2']['value'], rows['e_c2_calfade']['value']
    return {'value_no_ageing': v_no, 'value_cycling_only': v_cyc, 'value_baseline': v_base,
            'reduction_cycling_only_pct': 100.0 * (v_no - v_cyc) / v_no,
            'reduction_cycling_calendar_pct': 100.0 * (v_no - v_base) / v_no,
            'I': rows['e_c2_calfade']['I'],
            'source': f'{FROZEN} tables.ageing.rows[*].value (= Q(0) - Q(x), gross)'}


def part_e(frozen, w155):
    rows = {r['cell']: r for r in frozen['tables']['ageing']['rows']}
    rep = w155['reports']['e_soh050']
    soh = {}
    with open(_reg(W188D_SOH), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['cell'] in ('e_c2_calfade', 'e_soh050'):
                soh.setdefault(r['cell'], []).append(
                    {k: r[k] for k in ('block_year', 'soh_start_of_block', 'soh_end_of_block', 'e_available_mwh',
                                       'efc_per_day', 'soh_min_applied', 'floor_row_active_at_terminal_flag')})
    return {'AE_baseline': rows['e_c2_calfade']['AE'], 'AE_floor050': rep.get('AE'),
            'EFC_baseline': rows['e_c2_calfade']['EFC'], 'EFC_floor050': rep.get('EFC'),
            'per_block_W188d': soh}


MS_REPO = 'manuscript/6a67305f25e8348fb71380c3'
MS_NEW, MS_OLD = 'e315b07', '3ad0764'


def _tex(commit):
    b = subprocess.run(['git', '-C', MS_REPO, 'show', f'{commit}:main.tex'], capture_output=True, check=True).stdout
    INPUTS[f'{MS_REPO}@{commit}:main.tex'] = {'sha256': hashlib.sha256(b).hexdigest(), 'git_tracked': True}
    return b.decode('utf-8')


def _table_body(tex, label):
    import re
    i = tex.index('\\label{' + label + '}')
    body = tex[tex.index('\\midrule', i):tex.index('\\bottomrule', i)]
    return [r for r in re.split(r'\\\\', body) if '&' in r]


def printed_tables():
    import re
    num = r'\$?(\$-\$|-)?\s*([0-9]+\.[0-9])'
    out = {}
    for commit in (MS_NEW, MS_OLD):
        tex = _tex(commit)
        for lab in ('tab:res_value_ladder', 'tab:res_ageing'):
            rows = []
            for r in _table_body(tex, lab):
                cells = [c.strip() for c in r.replace('\\midrule', '').split('&')]
                rows.append(cells)
            out[f'{commit}:{lab}'] = rows
    return out


def part_f(frozen):
    claims = {c['claim_id']: c for c in frozen['tables']['claims']}
    a64 = {r['claim_id']: r for r in frozen['tables']['a64']['rows']}
    claims.update(a64)
    table_rows = {
        'tab:res_value_ladder': [
            ('5, 0.25, 0.5', 'C:y2025__n5_p0.25_e0.5', 49.7, 12.6),
            ('7 (reference), 0.25, 1', 'B:n7_4h_e1', 64.4, 13.1),
            ('7, 0.5, 1', 'B:n7_2h_e1', 114.1, 12.6),
            ('7, 0.5, 2', 'B:n7_4h_e2', 135.5, 13.4),
            ('5 and 9', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', 102.1, 12.6),
            ('5 and 7 (2030)', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 93.9, 13.4)],
        'tab:res_ageing': [
            ('Baseline (cycling + calendar)', 'E:n7_4h_e1_C2_calfade:value_minus_I', -64.4, 13.1),
            ('Cycling only', 'E:n7_4h_e1_C2:value_minus_I', -31.6, 12.6),
            ('Retention 50 %', 'E:C3_unit_value_minus_I', -73.7, 12.8),
            ('Retention 50 %, mid-block', 'E:n7_4h_e1_C3_midblock:value_minus_I', -65.9, 12.9),
            ('Datasheet 8 000 cycles', 'E:n7_4h_e1_C4:value_minus_I', -45.2, 12.6),
            ('No ageing', 'E:n7_4h_e1_no_ageing:value_minus_I', -4.1, 12.9),
            ('Baseline, 0.50 floor', 'E:soh050:value_minus_I', -59.5, 12.6)],
    }
    out = {}
    for tab, rows in table_rows.items():
        res = []
        for label, cid, printed_v, printed_t in rows:
            c = claims[cid]
            cells = c.get('instance') or c.get('cells')
            stat = {}
            for cell in cells:
                rec = frozen_cell(frozen, cell) if not isinstance(cells, dict) or 'status' not in cells[cell] \
                    else cells[cell]
                if rec is None:
                    rec = frozen_cell(frozen, cell)
                if rec is None or 'status' not in rec:
                    rec2 = frozen_cell(frozen, cell) or {}
                    stat[cell] = rec2.get('status')
                else:
                    stat[cell] = rec['status']
            if 'ref_cell' in c:   # A64 rows name the reference by short id
                ref = c['ref_cell'] if c['ref_cell'].startswith('ref:') else 'ref:' + c['ref_cell']
                stat.setdefault(ref, (frozen_cell(frozen, ref) or {}).get('status'))
            d = c['d_gross']
            v = d / 1000.0
            thr = c['gross_threshold_or_bar'] / 1000.0
            res.append({'row': label, 'claim': cid, 'd_gross_eur': d, 'threshold_eur': c['gross_threshold_or_bar'],
                        'rule': c.get('gross_rule'), 'verdict': c.get('gross_verdict'),
                        'printed_value_k': printed_v, 'printed_threshold_k': printed_t,
                        'value_rounds_to_printed': round(abs(v), 1) == abs(printed_v),
                        'threshold_rounds_to_printed': round(thr, 1) == printed_t,
                        'cell_status': stat, 'all_certified': all(s == 'certified' for s in stat.values())})
        out[tab] = res
    out['printed_rows_new_and_predecessor'] = printed_tables()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'refusing to overwrite {a.out}')
    frozen = _json(FROZEN)
    srp1 = _json(SRP1)
    w109 = _json(W109)
    w155 = _json(W155)
    E = {k: load_eval(k) for k in EVALS}
    checks = {}
    for k, e in E.items():
        checks[k] = {'status': e['status'], 'certified_cost': e['certified_cost'], 'Q_components': e['Q'],
                     'certified_cost_equals_Q_components': e['certified_cost'] == e['Q'],
                     'certification_cycle': e['certification_cycle'], 'cycles_run_record': e['cycles_run_record'],
                     'cycles_run_components': e['cycles_run_components'],
                     'Q_identity_residual_eur': e['Q_identity_residual'],
                     'detector_sub_residual_eur': e['detector_sub_residual'],
                     'agent_residual_eur': e['agent_residual'], 'dir': e['dir']}
    result = {
        'stage': 'P5.15 Section 4 revision -- zero-solve extractions (handback 2026-11-16, Overleaf e315b07)',
        'utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'script_sha256': _sha(__file__),
        'objective_convention': 'Q = gross_operational_cost, contracted interface settlement excluded; discounted, '
                                'block-weighted; salvage not included (0 at these cells)',
        'per_eval_checks': checks,
        'a_operating_value_composition': part_a(E, frozen),
        'b_res_curtailment': part_b(E, srp1, w109),
        'c_penalty_sensitivity': part_c(E),
        'd_ageing_percentages': part_d(frozen),
        'e_floor_050': part_e(frozen, w155),
        'f_table_thresholds': part_f(frozen),
    }
    if _PICKLE_CALLS != {'load': 0, 'loads': 0}:
        sys.exit(f'pickle guard: observed {_PICKLE_CALLS}, permitted 0')
    result['pickle_guard'] = {'permitted': 0, 'observed': dict(_PICKLE_CALLS)}
    result['production_modules_imported'] = sorted(m for m in sys.modules if m in (
        'pyomo', 'pyomo.environ', 'network', 'shared_resources_planning', 'model_construction_helpers'))
    result['inputs'] = INPUTS
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'x', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print('wrote', a.out, _sha(a.out))


if __name__ == '__main__':
    main()
