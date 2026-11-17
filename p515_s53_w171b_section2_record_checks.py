"""
P5.15 Addendum 68, Planner task W171b -- record checks behind two rows of the section-2 audit
(P5_15_W171_SECTION2_AUDIT.md, section 2.2.7). ZERO SOLVES, NO MODEL, NO PRODUCTION MODULE.

What it reads: committed JSON / JSONL run records only (`resettle_decision.json`, `resettle_cycle_record.jsonl`), the
frozen v6 stage spec (P_MAX) and the frozen Step-6 tables (cell list, slack cross-check). The only code it imports is
the stdlib-only settling-rule modules `settling_criterion*.py` (no Pyomo, no model code), to replay the committed
per-cycle inputs.

Check A -- "P-hat measured between the first and third turning points" (main.tex l. 624-625) against the code
  (`settling_criterion_v2.SettlingRuleV2.evaluate`: P_hat = T[-1].t - T[-3].t, the three MOST RECENT turning points).
  A1. Gate: the unmodified `settling_criterion_v6.SettlingRuleV6`, fed each cell's committed per-cycle inputs
      (Q, boyd_k, t_sum, all_clean_k as the rule observed them), must reproduce the committed decision (status, k*,
      branch, window). A cell that does not reproduce is reported and excluded from A2.
  A2. The same replay with ONE change: P_hat = T[2].t - T[0].t (the first and third turning points since k0, the
      text's definition); everything else is version 6 unchanged. Reported per cell: decision under each reading.
Check B -- "the objective's movement since the residual pass (its settling slack)" (main.tex l. 638, 643) against
  the scorer (`p515_s53_w132_resettle_v3_campaign.view_from_report`: slack = |Q(end) - Q(N_old)|, N_old = the
  original run's certification cycle, Q(N_old) from the ORIGINAL record; undefined for an ungated cell).
  For every uncertified cell: slack_code (recomputed and cross-checked against the frozen table), slack_text =
  |Q(end) - Q(k0_run)| (k0_run = this run's first residual pass), gap = |t_sum(end)|, and the two uncertified bars
  3 x max(gap, slack). The two uncertified F2 references of the tables (ref:5ca4f86c = W118 f2_incumbent,
  ref:e28de4ac = W118 f2_challenger) are included from their W118 records.
Check C -- every frozen-table claim scored under the uncertified form, re-scored with slack_text in place of
  slack_code (bar = 3 x max over the uncertified cells' gap and slack; determinate iff |d_gross| > bar AND the
  same-sign Q_cc margin > bar, as `p515_s53_w132_resettle_v3_campaign.resolve`). The code bar is recomputed too and
  must equal the frozen table's bar (gate).

Guards: `pickle` is blocked before any import (sys.modules['pickle'] = None); at the end the script asserts that no
production module (pyomo, shared_resources_planning, network*, shared_energy_storage*, model_construction_helpers,
admm_*) was imported. There is no solve path to guard.

Output: data/SRP1/Results/P515S53/w171b_section2_audit/w171b_record_checks.json and manifest_sha256.json (sha256 of
the output and of every input read). Refuses to overwrite an existing output.

Command (repo root, canonical interpreter, attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w171b_section2_record_checks.py \\
      > data/SRP1/Results/P515S53/w171b_section2_audit/launch.log 2>&1
"""
import sys

sys.modules['pickle'] = None  # block pickle before anything else is imported

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import settling_criterion_v2 as SC2  # noqa: E402  (stdlib only)
import settling_criterion_v6 as SC6  # noqa: E402  (stdlib only)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w171b_section2_audit')
OUT_JSON = os.path.join(OUT_DIR, 'w171b_record_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
SPEC_V6 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json')
FROZEN_TABLES = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
ROOTS = (os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6'),
         os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_ext_v6'),
         os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w155_a64_cells'))
# d_c52e1670: certified under v6 FROM RECORDS on its v5 run (frozen table certifying_spec); its run records are the v5 run.
EXTRA_V6_FROM_RECORDS = {
    'd_c52e1670': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w139_resettle_v5',
                               'campaign_s53_w139_resettle_v5_d_c52e1670', 'evals', 'af42a163b14f0895_d_c52e1670'),
}
# The uncertified F2 references of the tables, from their W118 runs (frozen table cells ref:5ca4f86c / ref:e28de4ac).
W118_REFS = {
    'ref:5ca4f86c': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle',
                                 'campaign_s53_w118_resettle_r2_f2_incumbent', 'evals', '24c5ccb6f285219f_f2_incumbent'),
    'ref:e28de4ac': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle',
                                 'campaign_s53_w118_resettle_r2_f2_challenger', 'evals', '1fe91e86f11e76af_f2_challenger'),
}
FORBIDDEN_MODULE_PREFIXES = ('pyomo', 'shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

INPUTS = {}


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    h = hashlib.sha256()
    with open(_abs(rel), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(rel):
    INPUTS[rel] = _sha(rel)
    with open(_abs(rel)) as f:
        return json.load(f)


def _jsonl(rel):
    INPUTS[rel] = _sha(rel)
    with open(_abs(rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


class TextPhatRuleV6(SC6.SettlingRuleV6):
    """Version 6 with ONE change: P_hat measured between the FIRST and THIRD turning points since k0 (the text),
    instead of the three most recent. Implemented by presenting T[:3] to version 2's evaluate (which computes
    P_hat = T[-1].t - T[-3].t and uses T only for that and for the len(T) >= 3 test); the sign state is restored
    before anything else reads it."""

    def evaluate(self, k):
        sign = self.sign
        t_full = sign.T
        if len(t_full) >= 3:
            sign.T = t_full[:3]
        try:
            return super().evaluate(k)
        finally:
            sign.T = t_full


def _cells():
    """[(cell, eval_dir_rel)] for every v6 decision under the three roots, plus d_c52e1670's v5 run."""
    out = []
    for root in ROOTS:
        for path in sorted(glob.glob(os.path.join(_abs(root), 'campaign_*', 'evals', '*', 'resettle_decision.json'))):
            ed = os.path.relpath(os.path.dirname(path), REPO)
            out.append((os.path.basename(ed).split('_', 1)[1], ed))
    for cell, ed in EXTRA_V6_FROM_RECORDS.items():
        out.append((cell, ed))
    return out


def _inputs_by_cycle(rows):
    q, boyd, t, clean = {}, {}, {}, {}
    for r in rows:
        s = r.get('settling') or {}
        k = r['cycle']
        q[k] = s.get('Q')
        boyd[k] = bool(s.get('boyd_k', False))
        t[k] = s.get('t_sum')
        clean[k] = bool(s.get('all_clean_k', False))
    return q, boyd, t, clean


def _rule_kwargs(dec):
    if dec.get('cap_mode') == 'dynamic':
        return {'cap_after_first_k0': SC2.CAP_AFTER_K0, 'cap_ceiling': dec['cap_ceiling']}
    return {'cap': dec['cap'], 'cap_ceiling': dec['cap_ceiling']}


def _summ(dec):
    if dec is None:
        return None
    return {'status': dec.get('status'), 'k_star': dec.get('k_star'), 'k_cap': dec.get('k_cap'),
            'branch': dec.get('branch'), 'window': dec.get('window'), 'W': dec.get('W'), 'P_hat': dec.get('P_hat'),
            'band_width': dec.get('band_width'), 'Q_k_star': dec.get('Q_k_star'), 'reasons': dec.get('reasons'),
            'T_at_decision': dec.get('T')}


def check_a(p_max):
    rows_out = []
    for cell, ed in _cells():
        dec_rel = os.path.join(ed, 'resettle_decision.json')
        rec_rel = os.path.join(ed, 'resettle_cycle_record.jsonl')
        committed = _load(dec_rel)
        if cell in EXTRA_V6_FROM_RECORDS:
            cert_rel = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6', 'v6_from_records',
                                    f'{cell}_v6_from_records_certificate.json')
            committed_v6 = _load(cert_rel)
            committed_v6 = dict(committed_v6, status='certified' if committed_v6.get('k_star') else None)
            kwargs = _rule_kwargs(committed)       # caps of the run the records come from
        else:
            committed_v6 = committed
            kwargs = _rule_kwargs(committed)
        rows = _jsonl(rec_rel)
        q, boyd, t, clean = _inputs_by_cycle(rows)
        last = max(q)
        _r, dec_code, _rule = SC6.replay(q, boyd, t, clean, p_max, last=last, **kwargs)
        _r2, dec_text, _rule2 = SC6.replay(q, boyd, t, clean, p_max, last=last, rule_class=TextPhatRuleV6, **kwargs)
        reproduces = (dec_code is not None and dec_code.get('status') == committed_v6.get('status')
                      and dec_code.get('k_star') == committed_v6.get('k_star')
                      and (dec_code.get('status') != 'certified'
                           or (dec_code.get('branch') == committed_v6.get('branch')
                               and list(dec_code.get('window') or []) == list(committed_v6.get('window') or []))))
        same = (dec_text is not None and dec_code is not None
                and dec_text.get('status') == dec_code.get('status') and dec_text.get('k_star') == dec_code.get('k_star')
                and dec_text.get('branch') == dec_code.get('branch')
                and list(dec_text.get('window') or []) == list(dec_code.get('window') or []))
        dq = None
        if dec_text and dec_code and dec_text.get('Q_k_star') is not None and dec_code.get('Q_k_star') is not None:
            dq = dec_text['Q_k_star'] - dec_code['Q_k_star']
        rows_out.append({'cell': cell, 'eval_dir': ed, 'records_last_cycle': last, 'rule_kwargs': kwargs,
                         'committed': _summ(committed_v6), 'replay_code_v6': _summ(dec_code),
                         'replay_reproduces_committed': bool(reproduces), 'replay_text_phat': _summ(dec_text),
                         'text_reading_changes_decision': (not same) if reproduces else None,
                         'Q_k_star_text_minus_code': dq,
                         'Q_k_star_text_minus_code_over_tau': (dq / SC6.TAU) if dq is not None else None,
                         'text_decision_beyond_records': (dec_text is None)})
    return rows_out


def check_b(table_cells):
    out = []
    for cell, ed in _cells() + sorted(W118_REFS.items()):
        if cell in EXTRA_V6_FROM_RECORDS:
            continue
        dec = _load(os.path.join(ed, 'resettle_decision.json'))
        if dec.get('status') == 'certified':
            continue
        rows = _jsonl(os.path.join(ed, 'resettle_cycle_record.jsonl'))
        by = {r['cycle']: (r.get('settling') or {}) for r in rows}
        end = max(by)
        q_end, t_end = by[end].get('Q'), by[end].get('t_sum')
        k0_run = dec.get('first_residual_pass_run') or dec.get('N')
        q_k0 = by.get(k0_run, {}).get('Q') if k0_run is not None else None
        q_nold = dec.get('Q_N_old_recorded')
        slack_code = abs(q_end - q_nold) if (q_end is not None and q_nold is not None) else None
        slack_text = abs(q_end - q_k0) if (q_end is not None and q_k0 is not None) else None
        gap = abs(t_end) if t_end is not None else None
        bar = (lambda s: 3.0 * max(gap, s) if (gap is not None and s is not None) else None)
        tab = table_cells.get(cell, {})
        out.append({'cell': cell, 'eval_dir': ed, 'gated': dec.get('gated'), 'end_cycle': end, 'k0_run': k0_run,
                    'N_old': dec.get('N_old'), 'Q_end': q_end, 'Q_k0_run': q_k0, 'Q_N_old_original_record': q_nold,
                    'gap_abs_t_sum_end': gap, 'slack_code_abs_Q_end_minus_Q_N_old': slack_code,
                    'slack_text_abs_Q_end_minus_Q_k0': slack_text,
                    'bar_code_3max_gap_slack': bar(slack_code), 'bar_text_3max_gap_slack': bar(slack_text),
                    'bar_text_over_bar_code': (bar(slack_text) / bar(slack_code))
                    if (bar(slack_text) and bar(slack_code)) else None,
                    'frozen_table_slack': tab.get('slack'), 'frozen_table_gap': tab.get('gap'),
                    'slack_code_equals_frozen_table': (tab.get('slack') is not None and slack_code is not None
                                                       and math.isclose(tab['slack'], slack_code, rel_tol=0, abs_tol=1e-6)),
                    'gap_equals_frozen_table': (tab.get('gap') is not None and gap is not None
                                                and math.isclose(tab['gap'], gap, rel_tol=0, abs_tol=1e-6))})
    return out


def check_c(claims, b_rows):
    """Re-score every uncertified-form claim with slack_text. Pure."""
    by = {r['cell']: r for r in b_rows}
    out = []
    for c in claims:
        if not str(c.get('gross_rule', '')).startswith('uncertified'):
            continue
        cells = [c['ref_cell'], c['other_cell']]
        unc = [x for x in cells if x in by]
        missing = [x for x in cells if x not in by and c.get('ref_status' if x == c['ref_cell'] else 'other_status')
                   != 'certified']
        comps_code, comps_text = [], []
        for x in unc:
            comps_code += [by[x]['gap_abs_t_sum_end'], by[x]['slack_code_abs_Q_end_minus_Q_N_old']]
            comps_text += [by[x]['gap_abs_t_sum_end'], by[x]['slack_text_abs_Q_end_minus_Q_k0']]
        d_q, d_cc = c['d_gross'], c['d_Qcc_report_only']
        m_cc = abs(d_cc) if (d_q > 0) == (d_cc > 0) else -abs(d_cc)

        def verdict(bar):
            return 'determinate' if (abs(d_q) > bar and m_cc > bar) else 'within the uncertified bar'
        bar_code = 3.0 * max(comps_code) if comps_code and None not in comps_code else None
        bar_text = 3.0 * max(comps_text) if comps_text and None not in comps_text else None
        out.append({'claim_id': c['claim_id'], 'cells': cells, 'uncertified_cells': unc, 'unresolved_cells': missing,
                    'd_gross': d_q, 'd_Qcc': d_cc, 'frozen_bar': c.get('gross_threshold_or_bar'),
                    'frozen_verdict': c.get('gross_verdict'), 'bar_code_recomputed': bar_code,
                    'bar_code_equals_frozen': (bar_code is not None and c.get('gross_threshold_or_bar') is not None
                                               and math.isclose(bar_code, c['gross_threshold_or_bar'], rel_tol=0,
                                                                abs_tol=1e-6)),
                    'verdict_code_recomputed': verdict(bar_code) if bar_code is not None else None,
                    'bar_text_slack': bar_text,
                    'verdict_text_slack': verdict(bar_text) if bar_text is not None else None,
                    'verdict_changes': (verdict(bar_text) != verdict(bar_code))
                    if (bar_text is not None and bar_code is not None) else None})
    return out


def main():
    if os.path.exists(_abs(OUT_JSON)) or os.path.exists(_abs(OUT_MANIFEST)):
        raise SystemExit(f'refusing to overwrite {OUT_JSON} / {OUT_MANIFEST}')
    os.makedirs(_abs(OUT_DIR), exist_ok=True)
    spec = _load(SPEC_V6)
    p_max = spec['stop_rule']['constants']['P_MAX']['value']
    tables = _load(FROZEN_TABLES)
    table_cells = tables['tables']['cells']
    a = check_a(p_max)
    b = check_b(table_cells)
    cc = check_c(tables['tables']['claims'], b)
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    if bad:
        raise SystemExit(f'production module(s) imported: {bad}')
    if sys.modules.get('pickle', 'absent') is not None:
        raise SystemExit('pickle block was lifted')
    n_rep = sum(r['replay_reproduces_committed'] for r in a)
    summary = {
        'A_cells': len(a), 'A_replay_reproduces_committed': n_rep,
        'A_not_reproduced': [r['cell'] for r in a if not r['replay_reproduces_committed']],
        'A_text_reading_changes_decision': [r['cell'] for r in a if r['text_reading_changes_decision']],
        'A_certified_cells_with_more_than_3_turning_points_at_k_star': [
            r['cell'] for r in a if r['committed'] and r['committed']['status'] == 'certified'
            and r['committed']['T_at_decision'] and len(r['committed']['T_at_decision']) > 3],
        'B_uncertified_cells': len(b),
        'B_slack_code_equals_frozen_table_all': all(r['slack_code_equals_frozen_table'] for r in b),
        'C_uncertified_form_claims': len(cc),
        'C_bar_code_equals_frozen_all': all(r['bar_code_equals_frozen'] for r in cc),
        'C_verdict_code_equals_frozen_all': all(r['verdict_code_recomputed'] == r['frozen_verdict'] for r in cc),
        'C_claims_whose_verdict_changes_under_text_slack': [r['claim_id'] for r in cc if r['verdict_changes']],
        'C_claims_whose_bar_changes_under_text_slack': [
            r['claim_id'] for r in cc if r['bar_text_slack'] is not None and r['bar_code_recomputed'] is not None
            and not math.isclose(r['bar_text_slack'], r['bar_code_recomputed'], rel_tol=0, abs_tol=1e-6)],
    }
    payload = {
        'schema': 'p515_s53_w171b_record_checks_v1', 'utc': datetime.now(timezone.utc).isoformat(),
        'zero_solves': True, 'pickle_blocked': True, 'production_modules_imported': bad,
        'imported_rule_modules': sorted(m for m in sys.modules if m.startswith('settling_criterion')),
        'tau': SC6.TAU, 'p_max_from_spec_v6': p_max,
        'definitions': {
            'phat_code': 'settling_criterion_v2.SettlingRuleV2.evaluate: P_hat = T[-1].t - T[-3].t (three most recent)',
            'phat_text': 'main.tex l. 624-625: measured between the first and third turning points -> T[2].t - T[0].t',
            'slack_code': ('p515_s53_w132_resettle_v3_campaign.view_from_report: |s_signed|, s_signed = Q(end) - '
                           'Q(N_old) from the ORIGINAL record (gated cells only)'),
            'slack_text': 'main.tex l. 638: the objective\'s movement since the residual pass -> |Q(end) - Q(k0_run)|',
            'uncertified_bar': '3 x max(|t_sum(end)|, slack) (p515_s53_w132_resettle_v3_campaign.resolve)'},
        'summary': summary, 'check_A_phat': a, 'check_B_slack': b, 'check_C_claims_uncertified_form': cc,
        'inputs_sha256': dict(sorted(INPUTS.items())),
    }
    with open(_abs(OUT_JSON), 'x') as f:
        json.dump(payload, f, indent=1, sort_keys=False)
    manifest = {'files': {OUT_JSON: _sha(OUT_JSON),
                          os.path.relpath(os.path.abspath(__file__), REPO): _sha(os.path.relpath(os.path.abspath(__file__), REPO))},
                'inputs': dict(sorted(INPUTS.items()))}
    with open(_abs(OUT_MANIFEST), 'x') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(summary, indent=1))
    for r in a:
        c, x = r['replay_code_v6'] or {}, r['replay_text_phat'] or {}
        print(f"A {r['cell']:>18} reproduces={r['replay_reproduces_committed']} code=({c.get('status')},{c.get('k_star')},"
              f"{c.get('branch')},W={c.get('W')},P={c.get('P_hat')}) text=({x.get('status')},{x.get('k_star')},"
              f"{x.get('branch')},W={x.get('W')},P={x.get('P_hat')}) dQ/tau={r['Q_k_star_text_minus_code_over_tau']}")
    for r in b:
        print(f"B {r['cell']:>12} gap={r['gap_abs_t_sum_end']:.1f} slack_code={r['slack_code_abs_Q_end_minus_Q_N_old']} "
              f"slack_text={r['slack_text_abs_Q_end_minus_Q_k0']} bar_code={r['bar_code_3max_gap_slack']} "
              f"bar_text={r['bar_text_3max_gap_slack']} table_eq={r['slack_code_equals_frozen_table']}")
    for r in cc:
        print(f"C {r['claim_id']:>52} d={r['d_gross']:.0f} bar_code={r['bar_code_recomputed']} (frozen "
              f"{r['frozen_bar']}) {r['verdict_code_recomputed']} | bar_text={r['bar_text_slack']} "
              f"{r['verdict_text_slack']} changes={r['verdict_changes']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
