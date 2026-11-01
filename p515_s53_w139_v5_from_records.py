"""P5.15 Addendum 60, Planner task W139 items 5 and 7 -- CRITERION v5 FROM RECORDS, NO RE-RUNS; the recurring-Acceptable
report with the clean / non-clean classification. ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (every imported module's own zero-permit guard is verified at 0 as well). Only JSON / JSONL records and IPOPT text
logs are read.

WHAT IS COMPUTED, per committed re-settle record (16): the three v4 runs (W138: cell 1 b_2a0ba8b2 903657de, cell 2
b_0dd237f0 653e4e5e, cell 3 b_4649234b 0734f103), cell 1's v3 run (W133, ed71177e), the ten W118 r2 cells and the two
settled references (W101):
  * per cycle, per block (12 TSO + 36 DSO + 3 ESSO) the FINAL ACCEPTED ATTEMPT and its exit: the network blocks from the
    committed `network_ipopt_solve_records.jsonl` (the last record of a (round, agent, network, year, day) chain; every
    record's EXIT cross-checked against its log byte range -- W131's reader, unchanged), the ESSO blocks from the
    per-solve ESSO logs (W131's reader: the primary / _recovery / _recovery_tier2 file chain);
  * for every final exit that is NOT "Optimal Solution Found" the four IPOPT metrics of that final iterate, parsed from
    its own bytes (network: the record's byte range; ESSO: the attempt's own file) with the SAME parser the v5 in-cycle
    capture uses (`p515_s53_w139_resettle_v5_hooks.parse_final_summary`), cross-checked against W137's reader
    (`p515_s53_w137_recurring_acceptable.ipopt_stats`, the full segment) value by value;
  * the v5 classification of every block (`settling_criterion_v5.classify_block_exit`; an Optimal final exit is clean on
    any tier and needs no metrics) and all_clean_k;
  * the rule replayed from the records (Q = gross_operational_cost, boyd_k = all_boyd_pass AND local_solves_ok, t_sum
    as W137's replay inputs) under v4 (all_optimal_k) and v5 (all_clean_k) with the cell's cap rule, over its recorded
    cycles; the certifying spec of the committed run (the spec it certified under, if any) beside;
  * the Planner's cell-3 prediction input: the v5 certifying cycle computed from cell 3's v4 records;
  * pb_y2025_n5's cycle-167 exit (TSO 2035 Spring, a recovery Acceptable at ~2,234x complementarity) still vetoes;
  * a sensitivity for the Planner's reading of the task text ("a recovery attempt of any exit" vetoes): every final
    Optimal exit on a recovery tier, and whether any lies in a v5 certifying window (the expert's ruling, implemented:
    an Optimal exit is clean on any tier -- v5 is strictly more permissive than v4).
ITEM 7 (the recurring-Acceptable report, updated): every non-Optimal final exit at or after the cell's k0 (the first
residual pass), grouped by block across cells, with its tier, its metrics, its ratios and max ratio, and CLEAN /
NON-CLEAN under the 10x rule.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w139_resettle_v5/v5_from_records/):
  w139_v5_from_records.json, w139_recurring_acceptable_v5.json, w139_v5_from_records_log_inventory.json,
  manifest_sha256.json
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w139_resettle_v5/v5_from_records && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w139_v5_from_records.py \\
        > data/SRP1/Results/P515S53/w139_resettle_v5/v5_from_records/launch.log 2>&1
"""
import collections
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W139 v5 from records (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v4 as SC4  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w131_prefreeze_diagnostics as W131  # noqa: E402 -- W131's readers (arms its own guard)
import p515_s53_w137_recurring_acceptable as RA  # noqa: E402 -- W137's log reader (arms its own guard)
import p515_s53_w137_resettle_v4_checks as K137  # noqa: E402 -- the replay inputs (arms its own guard)
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w139_v5_from_records', GUARD), ('w131_imported', W131._GUARD), ('w137_ra_imported', RA.GUARD))
                 + tuple(K137.GUARDS))
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w139_resettle_v5', 'v5_from_records')
OUT_JSON = 'w139_v5_from_records.json'
OUT_RA = 'w139_recurring_acceptable_v5.json'
OUT_INV = 'w139_v5_from_records_log_inventory.json'
OUT_MAN = 'manifest_sha256.json'
W137_ROOT = os.path.join(S53, 'w137_resettle_v4')
W132_ROOT = os.path.join(S53, 'w132_resettle_v3')
V4_STAGE_SPEC = {'path': os.path.join(W137_ROOT, 'frozen_s53_resettle_spec_v4_e11fbc89.json'), 'sha8': 'e11fbc89'}
V3_STAGE_SPEC = {'path': os.path.join(W132_ROOT, 'frozen_s53_resettle_spec_v3_139d1e62.json'), 'sha8': '139d1e62'}
V2_STAGE_SPEC = {'path': os.path.join(S53, 'w118_resettle', 'frozen_s53_resettle_spec_v2_fc791891.json'),
                 'sha8': 'fc791891'}
V1_STAGE_SPEC = {'path': os.path.join(S53, 'frozen_s53_spec_v39_8a612429.json'), 'sha8': '8a612429'}

V4_RUNS = {
    'b_2a0ba8b2@v4': {'cell': 'b_2a0ba8b2', 'commit': '903657de',
                      'eval_dir': os.path.join(W137_ROOT, 'campaign_s53_w137_resettle_v4_b_2a0ba8b2', 'evals',
                                               '33447912dab48fff_b_2a0ba8b2')},
    'b_0dd237f0@v4': {'cell': 'b_0dd237f0', 'commit': '653e4e5e',
                      'eval_dir': os.path.join(W137_ROOT, 'campaign_s53_w137_resettle_v4_b_0dd237f0', 'evals',
                                               '38d09af2c857755f_b_0dd237f0')},
    'b_4649234b@v4': {'cell': 'b_4649234b', 'commit': '0734f103',
                      'eval_dir': os.path.join(W137_ROOT, 'campaign_s53_w137_resettle_v4_b_4649234b', 'evals',
                                               'edc95b9bc4e5e395_b_4649234b')},
}
# The Planner's recorded cell-3 prediction is computed from this run's records.
CELL3_RUN = 'b_4649234b@v4'
PB_N5 = {'record': 'pb_y2025_n5', 'cycle': 167, 'block': 'TSO|2035|Spring'}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _net_key(agent, year, day):
    return f'TSO|{year}|{day}' if agent == 'TSO' else f'DSO|{int(agent[3:])}|{year}|{day}'


# ======================================================================================================================
#  the records
# ======================================================================================================================
def record_inputs():
    """{record id: {cell, eval_dir, q, b, t, n, kw, run_status, certifying}} for the 16 records."""
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        ins = K137.replay_inputs()
    w131 = json.load(open(os.path.join(REPO, K137.W131['path'])))['task1']
    for cell, (q, b, t, _ao, n, kw, committed, inputs, _src) in ins.items():
        if cell == K137.CELL1:
            rid = f'{cell}@v3'
            eval_dir = K137.CELL1_V3['eval_dir']
            run = {'spec': V3_STAGE_SPEC, 'criterion': 'v3 (reading alpha)', 'commit': K137.CELL1_V3['commit'],
                   'status': committed.get('status'), 'k_star': committed.get('k_star'), 'k_cap': committed.get('k_cap')}
            certifying = None
        else:
            rid = cell
            eval_dir = w131[cell]['eval_dir']
            if cell in ('x0', 'unit_n7_4h_e1'):
                spec, crit = V1_STAGE_SPEC, 'v1 (settling_criterion.py; W101)'
            else:
                spec, crit = V2_STAGE_SPEC, 'v2 (settling_criterion_v2.py; W118 r2)'
            run = {'spec': spec, 'criterion': crit, 'campaign_spec': w131[cell]['campaign_spec'],
                   'status': committed.get('status'), 'k_star': committed.get('k_star'), 'k_cap': committed.get('k_cap')}
            certifying = ({'spec': spec, 'criterion': crit, 'k_star': committed.get('k_star')}
                          if committed.get('status') == 'certified' else None)
            if cell == 'pb_y2025_n5':
                certifying = dict(certifying, excluded=('Addendum 58 Ruling 2 / Addendum 59: the certificate stays '
                                                        'EXCLUDED (the Acceptable recovery at k* 167 lies in its '
                                                        'window); the re-run pb_y2025_n5 is queued (W135 extension)'))
        out[rid] = {'cell': cell, 'eval_dir': eval_dir, 'q': q, 'b': b, 't': t, 'n': n, 'kw': kw, 'run': run,
                    'certifying': certifying, 'inputs_sha256': inputs}
    for rid, v in V4_RUNS.items():
        ev = v['eval_dir']
        man = json.load(open(os.path.join(REPO, ev.split(os.sep + 'evals' + os.sep)[0], 'campaign_manifest_sha256.json')))
        inputs = {}
        for f in ('per_cycle_record.jsonl', V4.CYCLE_FILE, V4.DECISION_FILE):
            rel = os.path.join(ev, f)
            now = _sha(os.path.join(REPO, rel))
            if man.get(rel) != now:
                raise RuntimeError(f'{rel}: sha256 {now} != manifest {man.get(rel)}')
            inputs[rel] = now
        rows = _jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))
        lines = {x['cycle']: x for x in _jsonl(os.path.join(ev, V4.CYCLE_FILE))}
        dec = json.load(open(os.path.join(REPO, ev, V4.DECISION_FILE)))
        q = {r['cycle']: r['gross_operational_cost'] for r in rows}
        b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
        t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
        cr = V4.declaration_for(v['cell'])['cap_rule']
        kw = ({'cap': cr['cap'], 'cap_ceiling': cr['ceiling']} if cr['kind'] == 'fixed' else
              {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
        run = {'spec': V4_STAGE_SPEC, 'criterion': 'v4 (reading gamma, window a)', 'commit': v['commit'],
               'status': dec.get('status'), 'k_star': dec.get('k_star'), 'k_cap': dec.get('k_cap')}
        certifying = ({'spec': V4_STAGE_SPEC, 'criterion': 'v4', 'k_star': dec.get('k_star')}
                      if dec.get('status') == 'certified' else None)
        out[rid] = {'cell': v['cell'], 'eval_dir': ev, 'q': q, 'b': b, 't': t, 'n': len(rows), 'kw': kw, 'run': run,
                    'certifying': certifying, 'inputs_sha256': inputs,
                    'in_run_all_optimal': {k: lines[k]['all_optimal_k'] for k in q}}
    return out


def block_finals(eval_rel, n):
    """{cycle: {block: {class, attempt, attempts, exit, (network) record}}} for cycles 1..n from the records (W131's
    readers) and the meta (coverage, log cross-check)."""
    net, net_meta = W131.network_final_attempts(eval_rel, n)
    recs = _jsonl(os.path.join(eval_rel, 'network_ipopt_solve_records.jsonl'))
    last = {}
    for r in recs:
        last[(r['round'], r['agent'], r['network'], str(r['year']), r['day'])] = r
    logs_dir = os.path.dirname(recs[0]['log_path'])
    esso, esso_meta = W131.esso_final_attempts(eval_rel, logs_dir, n)
    out = collections.defaultdict(dict)
    for (rnd, agent, network, year, day), v in net.items():
        if rnd == 0:
            continue
        out[rnd][_net_key(agent, year, day)] = {
            'family': 'network', 'class': H.ipopt_exit_class(v['final_exit']), 'attempt': v['final_attempt'],
            'attempts': v['attempts'], 'exit': v['final_exit'], 'network': network,
            'record': last[(rnd, agent, network, str(year), day)]}
    for (k, node), v in esso.items():
        if k == 0:
            continue
        suffix = {'primary': '', 'recovery': '_recovery', 'recovery_tier2': '_recovery_tier2'}[v['final_attempt']]
        out[k][f'ESSO|{node}'] = {
            'family': 'esso', 'class': H.ipopt_exit_class(v['final_exit']), 'attempt': v['final_attempt'],
            'attempts': None, 'exit': v['final_exit'],
            'log_path': os.path.join(logs_dir, f'optim_log_esso_node{node}_cycle{k:03d}{suffix}.txt')}
    coverage = all(len(out[k]) == 51 for k in range(1, n + 1))
    meta = {'coverage_51_every_cycle': coverage, 'network_log_byte_crosscheck_disagreements':
            net_meta['log_byte_crosscheck']['n_disagree'], 'network_attempt_counts': net_meta['attempt_counts'],
            'esso_missing': esso_meta['missing'], 'esso_exit_counts_final': esso_meta['exit_counts_final'],
            'logs_dir': os.path.relpath(logs_dir, REPO)}
    return out, meta


def classify_cycle(finals):
    """{block: v5 entry} for one cycle's 51 final exits; metrics parsed only for non-Optimal exits (an Optimal exit is
    clean on any tier)."""
    out = {}
    for key, f in finals.items():
        metrics, parse = None, None
        crosscheck = None
        if f['class'] != SC5.OPTIMAL_CLASS:
            if f['family'] == 'network':
                r = f['record']
                parse = V5.parse_final_summary(V5.read_segment_tail(r['log_path'], r['log_bytes'][0], r['log_bytes'][1]))
                ra = RA.ipopt_stats(RA._network_log_text(r))
                path = r['log_path']
            else:
                path = f['log_path']
                size = os.path.getsize(path)
                parse = V5.parse_final_summary(V5.read_segment_tail(path, 0, size, tail_bytes=size + 1))
                with open(path, errors='replace') as handle:
                    ra = RA.ipopt_stats(handle.read())
            metrics = parse['metrics']
            ra_m = {'overall_nlp_error': ra.get('overall_nlp_error_scaled'),
                    'dual_infeasibility': ra.get('dual_infeasibility_unscaled'),
                    'constraint_violation': ra.get('constraint_violation_unscaled'),
                    'complementarity': ra.get('complementarity_unscaled')}
            crosscheck = {'w137_reader_metrics': ra_m, 'equal': ra_m == metrics,
                          'log_exit_equals_record_exit': (parse.get('exit') or '').strip() == (f['exit'] or '').strip()}
            W131.INVENTORY.setdefault(os.path.relpath(path, REPO), {'sha256': _sha(path), 'bytes': os.path.getsize(path),
                                                                    'kind': f'{f["family"]}_log'})
        c = SC5.classify_block_exit(f['class'], f['attempt'], metrics, f['family'])
        e = {'family': f['family'], 'class': f['class'], 'attempt': f['attempt'], 'attempts': f.get('attempts'),
             'exit': f['exit'], 'clean': c['clean'], 'reason': c['reason']}
        if f['class'] != SC5.OPTIMAL_CLASS:
            e.update({'metrics': metrics, 'ratios': c['ratios'], 'max_ratio': c['max_ratio'],
                      'max_ratio_metric': c['max_ratio_metric'], 'iterations': parse.get('iterations'),
                      'parse_reason': parse.get('parse_reason'), 'reader_crosscheck': crosscheck,
                      'log': (os.path.relpath(f['record']['log_path'], REPO) if f['family'] == 'network'
                              else os.path.relpath(f['log_path'], REPO)),
                      'log_bytes': f['record']['log_bytes'] if f['family'] == 'network' else None,
                      'network': f.get('network')})
        out[key] = e
    return out


def _summ(dec, rule, n):
    if dec is None:
        return {'status': f'not certified within the recorded cycles 1..{n}', 'k_star': None, 'k_cap': None,
                'k0_at_end': rule.k0, 'n_vetoes': len(rule.vetoes), 'vetoes': list(rule.vetoes)}
    out = {k: dec.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W', 'window', 'band_width',
                                   't_sum_k_star', 'reasons')}
    out['n_vetoes'] = len(rule.vetoes)
    out['vetoes'] = list(rule.vetoes)
    return out


def record_report(rid, rec):
    finals, meta = block_finals(rec['eval_dir'], rec['n'])
    per_cycle = {k: classify_cycle(finals[k]) for k in range(1, rec['n'] + 1)}
    all_opt = {k: all(e['class'] == SC5.OPTIMAL_CLASS for e in per_cycle[k].values()) for k in per_cycle}
    all_clean = {k: all(e['clean'] for e in per_cycle[k].values()) for k in per_cycle}
    q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
    _o4, d4, r4 = SC4.replay(q, b, t, all_opt, V5.P_MAX, last=n, **kw)
    _o5, d5, r5 = SC5.replay(q, b, t, all_clean, V5.P_MAX, last=n, **kw)
    v4s, v5s = _summ(d4, r4, n), _summ(d5, r5, n)
    first_pass = next((k for k in sorted(q) if q[k] is not None and b.get(k)), None)
    non_opt = []
    for k in sorted(per_cycle):
        for key, e in per_cycle[k].items():
            if e['class'] != SC5.OPTIMAL_CLASS:
                non_opt.append({'cycle': k, 'block': key, **{x: e.get(x) for x in (
                    'network', 'attempt', 'attempts', 'exit', 'clean', 'reason', 'metrics', 'ratios', 'max_ratio',
                    'max_ratio_metric', 'iterations', 'log', 'log_bytes', 'reader_crosscheck')}})
    rec_opt = [{'cycle': k, 'block': key, 'attempt': e['attempt']} for k in sorted(per_cycle)
               for key, e in per_cycle[k].items() if e['class'] == SC5.OPTIMAL_CLASS and e['attempt'] != 'primary']
    win5 = v5s.get('window') if v5s.get('status') == 'certified' else None
    rec_opt_in_window = [x for x in rec_opt if win5 and win5[0] <= x['cycle'] <= win5[1]]
    cert = rec['certifying']
    cert_k = (cert or {}).get('k_star')
    v5k = v5s.get('k_star')
    rep = {
        'record': rid, 'cell': rec['cell'], 'eval_dir': rec['eval_dir'], 'cycles_recorded': n,
        'first_residual_pass': first_pass, 'cap_rule_replayed': kw,
        'committed_run': rec['run'],
        'certifying_spec': ({'spec': cert['spec'], 'criterion': cert['criterion'], 'k_star': cert['k_star'],
                             'excluded': cert.get('excluded')} if cert else None),
        'v4_from_records': v4s, 'v5_from_records': v5s,
        'v5_certifies_earlier_than_the_certifying_spec': bool(v5k is not None and cert_k is not None and v5k < cert_k),
        'v5_cycle_where_earlier': v5k if (v5k is not None and cert_k is not None and v5k < cert_k) else None,
        'v5_certifies_where_no_certificate': bool(v5k is not None and (cert is None or cert.get('excluded'))),
        'v5_equals_v4_from_records': (v5s.get('status'), v5s.get('k_star')) == (v4s.get('status'), v4s.get('k_star')),
        'non_optimal_cycles': sorted(k for k, v in all_opt.items() if not v),
        'non_clean_cycles': sorted(k for k, v in all_clean.items() if not v),
        'non_clean_cycles_at_or_after_first_pass': sorted(k for k, v in all_clean.items()
                                                          if not v and first_pass is not None and k >= first_pass),
        'non_optimal_final_exits': non_opt,
        'recovery_tier_optimal_final_exits': rec_opt,
        'planner_literal_reading_sensitivity': {
            'reading': ('the task text "a recovery attempt of any exit" read literally: an OPTIMAL recovery would also '
                        'veto (NOT implemented: the expert\'s ruling -- v5 strictly more permissive than v4 -- makes an '
                        'Optimal exit clean on any tier)'),
            'recovery_optimal_in_the_v5_certifying_window': rec_opt_in_window,
            'would_change_the_v5_outcome': bool(rec_opt_in_window)},
        'sources': meta, 'inputs_sha256': rec['inputs_sha256'],
    }
    if 'in_run_all_optimal' in rec:
        rep['in_run_all_optimal_equals_records'] = rec['in_run_all_optimal'] == all_opt
    return rep, per_cycle


def recurring(reports):
    """Item 7: every non-Optimal final exit at or after the record's first residual pass, grouped by block."""
    by_block = collections.OrderedDict()
    for rid, rep in reports.items():
        fp = rep['first_residual_pass']
        for e in rep['non_optimal_final_exits']:
            if fp is None or e['cycle'] < fp:
                continue
            b = by_block.setdefault(e['block'], {'block': e['block'], 'network': e.get('network'), 'records': {},
                                                 'n_exits': 0, 'n_clean': 0, 'n_non_clean': 0, 'max_ratios': [],
                                                 'tiers': collections.Counter(), 'reasons': collections.Counter()})
            b['records'].setdefault(rid, []).append({'cycle': e['cycle'], 'attempt': e['attempt'], 'clean': e['clean'],
                                                     'reason': e['reason'], 'max_ratio': e['max_ratio'],
                                                     'max_ratio_metric': e['max_ratio_metric'],
                                                     'ratios': e['ratios'], 'metrics': e['metrics'],
                                                     'iterations': e['iterations']})
            b['n_exits'] += 1
            b['n_clean' if e['clean'] else 'n_non_clean'] += 1
            if e['max_ratio'] is not None:
                b['max_ratios'].append(e['max_ratio'])
            b['tiers'][e['attempt']] += 1
            b['reasons'][e['reason']] += 1
    out = []
    for b in by_block.values():
        mr = b.pop('max_ratios')
        b['max_ratio_range'] = [min(mr), max(mr)] if mr else None
        b['tiers'] = dict(b['tiers'])
        b['reasons'] = dict(b['reasons'])
        b['n_records'] = len(b['records'])
        b['classification'] = ('clean' if b['n_non_clean'] == 0 else 'non-clean' if b['n_clean'] == 0 else 'mixed')
        out.append(b)
    return sorted(out, key=lambda b: (-b['n_records'], -b['n_exits'], b['block']))


def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_RA, OUT_INV, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    recs = record_inputs()
    reports = {}
    for rid in list(V4_RUNS) + [r for r in recs if r not in V4_RUNS]:
        rep, _pc = record_report(rid, recs[rid])
        reports[rid] = rep
        v4s, v5s = rep['v4_from_records'], rep['v5_from_records']
        _log(f"{rid}: certifying {((rep['certifying_spec'] or {}).get('spec') or {}).get('sha8')} "
             f"k* {(rep['certifying_spec'] or {}).get('k_star')}; v4 from records {v4s['status']} {v4s.get('k_star')}; "
             f"v5 from records {v5s['status']} {v5s.get('k_star')} window {v5s.get('window')}; non-clean at/after first "
             f"pass {rep['non_clean_cycles_at_or_after_first_pass']}; non-Optimal "
             + '; '.join(f"c{e['cycle']} {e['block']} {e['attempt']} {e['reason']} {e['max_ratio'] and round(e['max_ratio'], 3)}"
                         for e in rep['non_optimal_final_exits'] if rep['first_residual_pass'] is None
                         or e['cycle'] >= rep['first_residual_pass']))
    cell3 = reports[CELL3_RUN]['v5_from_records']
    pb = reports[PB_N5['record']]
    pb_exit = next((e for e in pb['non_optimal_final_exits'] if e['cycle'] == PB_N5['cycle']
                    and e['block'] == PB_N5['block']), None)
    pb_check = {'exit': pb_exit, 'non_clean': bool(pb_exit and pb_exit['clean'] is False),
                'reason': (pb_exit or {}).get('reason'), 'max_ratio': (pb_exit or {}).get('max_ratio'),
                'v5_from_records': pb['v5_from_records'],
                'still_vetoes': bool(pb_exit and pb_exit['clean'] is False
                                     and pb['v5_from_records'].get('k_star') != PB_N5['cycle'])}
    crosscheck_all = all(e['reader_crosscheck'] is None or (e['reader_crosscheck']['equal']
                                                            and e['reader_crosscheck']['log_exit_equals_record_exit'])
                         for rep in reports.values() for e in rep['non_optimal_final_exits'])
    table = []
    for rid, rep in reports.items():
        cs = rep['certifying_spec']
        table.append({'record': rid, 'cell': rep['cell'], 'committed_run': {k: rep['committed_run'].get(k) for k in (
            'criterion', 'commit', 'status', 'k_star', 'k_cap')} | {'spec': rep['committed_run']['spec']['sha8']},
            'certifying_spec': (cs['spec']['sha8'] if cs else None), 'certifying_criterion': (cs or {}).get('criterion'),
            'certified_k_star': (cs or {}).get('k_star'), 'certificate_excluded': (cs or {}).get('excluded'),
            'v4_from_records': {k: rep['v4_from_records'].get(k) for k in ('status', 'k_star', 'k_cap', 'window')},
            'v5_from_records': {k: rep['v5_from_records'].get(k) for k in ('status', 'k_star', 'k_cap', 'window',
                                                                           'branch', 'n_vetoes')},
            'v5_cycle_where_earlier': rep['v5_cycle_where_earlier'],
            'v5_certifies_where_no_certificate': rep['v5_certifies_where_no_certificate'],
            'non_clean_cycles_at_or_after_first_pass': rep['non_clean_cycles_at_or_after_first_pass'],
            'recovery_optimal_in_v5_window': rep['planner_literal_reading_sensitivity'][
                'recovery_optimal_in_the_v5_certifying_window']})
    ra_table = recurring(reports)
    guards = {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    code = {rel: _sha(os.path.join(REPO, rel)) for rel in (
        os.path.basename(__file__), 'settling_criterion_v5.py', 'settling_criterion_v4.py',
        'p515_s53_w139_resettle_v5_hooks.py', 'p515_s53_w131_prefreeze_diagnostics.py',
        'p515_s53_w137_recurring_acceptable.py', 'p515_s53_w137_resettle_v4_checks.py')}
    doc = {'schema': 'p515_s53_w139_v5_from_records_v1',
           'task': 'W139 item 5 (PLANNER_BRIEF_2026-09-13.md Addendum 60: v5 from records, no re-runs)',
           'utc': _utc(), 'git_head': git_head, 'code_sha256': code, 'definition': __doc__,
           'clean_rule': {'factor': SC5.CLEAN_FACTOR, 'metric_table': SC5.METRIC_TABLE, 'tolerances': SC5.TOLERANCES,
                          'tolerance_sources': SC5.TOLERANCE_SOURCES, 'reasons': SC5.CLEAN_REASONS},
           'per_record_table': table,
           'planner_cell3_prediction_input': {
               'record': CELL3_RUN, 'v4_run': V4_RUNS[CELL3_RUN],
               'v5_from_its_v4_records': {k: cell3.get(k) for k in ('status', 'k_star', 'window', 'branch', 'k0', 'W',
                                                                   'band_width', 't_sum_k_star', 'n_vetoes')},
               'prediction_cycle': cell3.get('k_star') if cell3.get('status') == 'certified' else None,
               'statement': ('cell 3 under v5 is bitwise identical to its v4 run through the v5 certifying cycle computed '
                             'from its v4 records, and certifies there')},
           'pb_y2025_n5_cycle_167_check': pb_check,
           'reader_crosscheck_all_equal': crosscheck_all,
           'reports': reports, 'n_logs_inventoried': len(W131.INVENTORY), 'guards': guards, 'wall_s': time.time() - t0}
    ra_doc = {'schema': 'p515_s53_w139_recurring_acceptable_v5_v1',
              'task': 'W139 item 7 (Addendum 59 recurring block; Addendum 60 reproducibility note): clean / non-clean',
              'utc': _utc(), 'git_head': git_head, 'code_sha256': code,
              'clean_rule': doc['clean_rule'],
              'predecessor_report': {'path': os.path.join(W137_ROOT, 'recurring_acceptable',
                                                          'w137_recurring_acceptable.json'),
                                     'sha256': _sha(os.path.join(REPO, W137_ROOT, 'recurring_acceptable',
                                                                 'w137_recurring_acceptable.json'))},
              'records': list(reports), 'by_block': ra_table,
              'reproducibility_note_blocks': {
                  'DSO|7|2025|Winter': next((b for b in ra_table if b['block'] == 'DSO|7|2025|Winter'), None),
                  'TSO|2035|Spring': next((b for b in ra_table if b['block'] == 'TSO|2035|Spring'), None),
                  'DSO|5|2030|Winter': next((b for b in ra_table if b['block'] == 'DSO|5|2030|Winter'), None)},
              'hours_note': RA.ipopt_stats('')['hours_note'], 'guards': guards}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    rp = os.path.join(out_dir, OUT_RA)
    with open(rp, 'x') as handle:
        GRIO.dump(ra_doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    ip = os.path.join(out_dir, OUT_INV)
    with open(ip, 'x') as handle:
        GRIO.dump(W131.INVENTORY, handle, indent=1, sort_keys=True)
    man = {os.path.relpath(p, REPO): _sha(p) for p in (jp, rp, ip)}
    for rep in reports.values():
        for rel, v in rep['inputs_sha256'].items():
            if isinstance(v, str):
                man[rel] = v
            elif isinstance(v, dict) and isinstance(v.get('sha256'), str):
                man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    for row in table:
        _log(f"TABLE {row['record']}: certifying {row['certifying_spec']} k* {row['certified_k_star']}"
             f"{' (EXCLUDED)' if row['certificate_excluded'] else ''}; v5 {row['v5_from_records']['status']} "
             f"{row['v5_from_records']['k_star']}; earlier {row['v5_cycle_where_earlier']}")
    for b in ra_table:
        _log(f"BLOCK {b['block']} ({b['network'] or 'ESSO'}): {b['n_exits']} exits in {b['n_records']} records, "
             f"{b['classification']} (clean {b['n_clean']}, non-clean {b['n_non_clean']}), tiers {b['tiers']}, max "
             f"ratio range {b['max_ratio_range']}")
    _log(f"cell 3 v5 from its v4 records: {doc['planner_cell3_prediction_input']['v5_from_its_v4_records']}")
    _log(f"pb_y2025_n5 c167: reason {pb_check['reason']} max ratio {pb_check['max_ratio']} still vetoes "
         f"{pb_check['still_vetoes']}; reader cross-check all equal {crosscheck_all}")
    _log(f'wrote {os.path.relpath(jp, REPO)}, {os.path.relpath(rp, REPO)} ({len(W131.INVENTORY)} logs inventoried); '
         f'guards {guards}; wall {time.time() - t0:.1f} s')
    for _n, g in reversed(GUARDS):
        g.uninstall()
    ok = all(not v['verify_0_failures'] for v in guards.values()) and crosscheck_all and pb_check['still_vetoes']
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
