"""
P5.15 Addendum 53, Planner task W103 -- ZERO SOLVES. The literal two-sign-change reading of the settling rule, and an
independent re-check of the post-run settling decision ((k) / G17), on the COMMITTED records of the SRP1 continuation
cells run under spec v39 (8a612429).

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0. No model is built;
the only inputs are committed JSON / JSONL records.

WHAT IT COMPUTES, per cell (x0, n7_4h_e1; N, CAP and P_MAX read from the cell's frozen campaign spec and asserted equal
to the task's values: x0 132 / 232, n7_4h_e1 112 / 212, P_MAX 22):
  L  the literal reading. `settling_criterion.SettlingRule` is fed the cell's committed per_cycle_record.jsonl exactly
     as the harness's post-run (k) check feeds it (Q = gross_operational_cost; boyd = boyd_all_pass and
     local_solves_ok), and after EACH `observe` `settling_criterion.literal_two_sign_change_report(rule, k)` is
     evaluated (REPORT-ONLY; never decides). Recorded: the per-cycle literal report; the first cycle k > N at which it
     certifies (the literal k*), with Q there and Q - Q_N; and, for the record, the first cycle at any k (including
     k <= N, where the adopted rule takes no decision).
  K  the re-check of the post-run decision. (K1) the pure rule's final decision equals the run's settling_decision.json
     key by key (JSON text, the harness's key list plus Q_k_star, range_over_tau, lapse_events) and every per-cycle rule
     record equals the in-cycle `settling` record of settling_continuation_cycle_record.jsonl; (K2) an INDEPENDENT
     re-derivation from the raw Q column that does not use settling_criterion's sign state: k0 = the first cycle of
     the final uninterrupted boyd-pass run; the sign sequence from k0 + K_EXCL with the EPS0 floor; turning points by
     arg-extremum over each same-sign run (ties -> earliest); half-swings; P_hat, W, the window, its range and band;
     s = Q_k* - Q_N; c = A[-1]/A[-2], c A[-1]/(1-c), mid(band) - Q_N -- each compared with the decision file and the
     harness's settling_report; (K3) Q_N bitwise (float.hex) equal to the recert's row N (the replay reference).

Output (write-once; refuses to overwrite): data/SRP1/Results/P515S53/w103_literal_reading/w103_literal_reading.json
plus its manifest w103_literal_reading_manifest_sha256.json (output + every input + this script + settling_criterion).

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w103_literal_reading.py \\
      > data/SRP1/Results/P515S53/w103_literal_reading/w103_literal_reading_launch.log 2>&1
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W103 literal reading (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion as SC  # noqa: E402

ROOT = 'data/SRP1/Results/P515S53/w101_srp1_continuation'
OUT_DIR = 'data/SRP1/Results/P515S53/w103_literal_reading'
OUT = os.path.join(OUT_DIR, 'w103_literal_reading.json')
MANIFEST = os.path.join(OUT_DIR, 'w103_literal_reading_manifest_sha256.json')
STAGE_SPEC = 'data/SRP1/Results/P515S53/frozen_s53_spec_v39_8a612429.json'
P_MAX_TASK = 22
CELLS = {
    'x0': {'campaign_spec': f'{ROOT}/campaign_s53_w101_srp1_cont_x0/campaign_spec_s53_w101_srp1_cont_x0_d1d5c3bc.json',
           'eval_dir': f'{ROOT}/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0',
           'campaign_results': f'{ROOT}/campaign_s53_w101_srp1_cont_x0/campaign_results.json',
           'N_task': 132, 'CAP_task': 232},
    'n7_4h_e1': {'campaign_spec': (f'{ROOT}/campaign_s53_w101_srp1_cont_n7_4h_e1/'
                                   'campaign_spec_s53_w101_srp1_cont_n7_4h_e1_434f46fc.json'),
                 'eval_dir': f'{ROOT}/campaign_s53_w101_srp1_cont_n7_4h_e1/evals/3f084f2ffaeef2b7_n7_4h_e1',
                 'campaign_results': f'{ROOT}/campaign_s53_w101_srp1_cont_n7_4h_e1/campaign_results.json',
                 'N_task': 112, 'CAP_task': 212},
}
DECISION_KEYS = ('status', 'k_star', 'branch', 'k0', 'T', 'A', 'P_hat', 'W', 'window', 'band', 'band_width', 'k_cap',
                 'reasons', 'range', 'Q_k_star', 'range_over_tau', 'lapse_events')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def committed_at_head(path):
    """True iff the working-tree file equals its blob at HEAD (git hash-object == HEAD:path)."""
    wt = subprocess.run(['git', 'hash-object', path], capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(['git', 'rev-parse', f'HEAD:{path}'], capture_output=True, text=True)
    return head.returncode == 0 and head.stdout.strip() == wt


def jtext(x):
    return json.dumps(x, sort_keys=True)


def independent_rederivation(q, boyd, n, k_last):
    """K2: from the raw Q column, WITHOUT settling_criterion's sign state. Assumes the decision is taken at k_last."""
    run_start = None
    for k in range(1, k_last + 1):
        ok = q.get(k) is not None and boyd[k]
        if not ok:
            run_start = None
        elif run_start is None:
            run_start = k
    k0 = run_start
    lapses_after_n = [k for k in range(n + 1, k_last + 1) if q.get(k) is None or not boyd[k]]
    signs = []  # (k, s)
    for k in range(k0 + SC.K_EXCL, k_last + 1):
        dq = q[k] - q[k - 1]
        s = 0 if abs(dq) < SC.EPS0 else (1 if dq > 0 else -1)
        signs.append((k, s))
    nz = [(k, s) for k, s in signs if s != 0]
    T = []
    i = 0
    while i < len(nz):
        j = i
        while j + 1 < len(nz) and nz[j + 1][1] == nz[i][1]:
            j += 1
        if j + 1 < len(nz):  # a sign change at nz[j+1]; the run of sign nz[i][1] spans cycles nz[j][0] .. nz[j+1][0]-1
            lo, hi = nz[j][0], nz[j + 1][0] - 1
            kind = 'max' if nz[i][1] == 1 else 'min'
            cands = list(range(lo, hi + 1))
            best = cands[0]
            for c in cands:
                if (q[c] > q[best]) if kind == 'max' else (q[c] < q[best]):
                    best = c
            T.append([best, kind, q[best]])
        i = j + 1
    A = [abs(T[m][2] - T[m - 1][2]) for m in range(1, len(T))]
    out = {'k0': k0, 'lapse_cycles_after_N': lapses_after_n, 'T': T, 'A': A}
    if len(T) >= 3:
        p_hat = T[-1][0] - T[-3][0]
        w = max(SC.W_MIN, int(math.ceil(SC.W_FACTOR * p_hat)))
        lo = k_last - w + 1
        vals = [q[c] for c in range(lo, k_last + 1)]
        out.update({'P_hat': p_hat, 'W': w, 'window': [lo, k_last], 'band': [min(vals), max(vals)],
                    'range': max(vals) - min(vals), 'range_over_tau': (max(vals) - min(vals)) / SC.TAU,
                    'swings_non_increasing': all(A[m + 1] <= A[m] for m in range(len(A) - 1)),
                    'window_inside_run': lo >= k0})
    return out


def cell_reading(cell, spec_info):
    cs = json.load(open(spec_info['campaign_spec']))
    cand = cs['candidates'][0]['settling_continuation']
    n = cand['hold_after_cycle']
    cap = cs['cap']
    p_max = cand['settling_rule']['p_max']
    assert (n, cap, p_max) == (spec_info['N_task'], spec_info['CAP_task'], P_MAX_TASK), (cell, n, cap, p_max)
    assert cap == n + SC.CAP_AFTER_N
    ev = spec_info['eval_dir']
    pcr = os.path.join(ev, 'per_cycle_record.jsonl')
    dec_path = os.path.join(ev, 'settling_decision.json')
    lines_path = os.path.join(ev, 'settling_continuation_cycle_record.jsonl')
    ref = cand['replay_reference']
    inputs = {name: {'path': p, 'sha256': sha256_file(p), 'committed_at_HEAD': committed_at_head(p)}
              for name, p in (('campaign_spec', spec_info['campaign_spec']), ('per_cycle_record', pcr),
                              ('settling_decision', dec_path), ('settling_continuation_cycle_record', lines_path),
                              ('campaign_results', spec_info['campaign_results']),
                              ('replay_reference_recert_per_cycle_record', ref['per_cycle_record']))}
    not_committed = [k for k, v in inputs.items() if not v['committed_at_HEAD']]
    if not_committed:
        raise RuntimeError(f'{cell}: inputs not committed at HEAD: {not_committed}')
    assert inputs['replay_reference_recert_per_cycle_record']['sha256'] == ref['sha256']

    rows = read_jsonl(pcr)
    assert [r['cycle'] for r in rows] == list(range(1, len(rows) + 1))
    rule = SC.SettlingRule(n, cap, p_max)
    per_cycle = []
    pure_recs = []
    lit_first_after_n = None
    lit_first_any = None
    for r in rows:
        k = r['cycle']
        rec = rule.observe(k, r['gross_operational_cost'], bool(r['boyd_all_pass'] and r['local_solves_ok']))
        pure_recs.append(rec)
        lit = SC.literal_two_sign_change_report(rule, k)
        per_cycle.append({'k': k, 'phase': rec['phase'], 'Q': rec['Q'], 'boyd_k': rec['boyd_k'],
                          'rule_decision': rec['decision'], 'literal': lit})
        if lit['literal_certifies'] and lit_first_any is None:
            lit_first_any = k
        if lit['literal_certifies'] and k > n and lit_first_after_n is None:
            lit_first_after_n = k
    q = {r['cycle']: r['gross_operational_cost'] for r in rows}
    boyd = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
    ref_rows = read_jsonl(ref['per_cycle_record'])
    q_n_ref = {r['cycle']: r['gross_operational_cost'] for r in ref_rows}[n]
    q_n = q[n]
    by_k = {p['k']: p for p in per_cycle}

    def lit_summary(k):
        if k is None:
            return None
        return {'k': k, 'Q': q[k], 'Q_minus_Q_N': q[k] - q_n, 'report': by_k[k]['literal'],
                'cycles_before_rule_k_star': (rule.decision or {}).get('k_star', k) - k
                if (rule.decision or {}).get('status') == 'certified' else None}

    # K1 -- the pure rule vs the run's decision file and in-cycle records
    dec_run = json.load(open(dec_path))
    dec_pure = rule.decision
    key_eq = {k: jtext(dec_run.get(k)) == jtext((dec_pure or {}).get(k)) for k in DECISION_KEYS}
    in_cycle = [x.get('settling') for x in read_jsonl(lines_path)]
    lines_eq = [jtext(a) for a in in_cycle] == [jtext(b) for b in pure_recs]
    early = [p['k'] for p in pure_recs if p['k'] <= n and str(p.get('decision') or '').startswith('certified')]
    k1 = {'decision_keys_equal': key_eq, 'decision_reproduced_exactly': all(key_eq.values()),
          'every_cycle_record_reproduced': lines_eq, 'n_in_cycle_records': len(in_cycle),
          'never_certifies_at_k_le_N': not early, 'decision_pure': dec_pure}

    # K2 -- independent re-derivation from raw Q
    k_last = rows[-1]['cycle']
    ind = independent_rederivation(q, boyd, n, k_last)
    sr = json.load(open(spec_info['campaign_results']))['settling_report']
    k2 = {'independent': ind}
    if dec_run.get('status') == 'certified':
        A = dec_run['A']
        c = A[-1] / A[-2] if len(A) >= 2 else None
        band = dec_run['band']
        derived = {'s_signed': q[k_last] - q_n, 'abs_s': abs(q[k_last] - q_n), 'c': c,
                   'c_A_over_1_minus_c': (c * A[-1] / (1 - c)) if (c is not None and c < 1) else None,
                   'mid_band_minus_Q_N': (band[0] + band[1]) / 2 - q_n}
        comp = {
            'k_star_is_last_cycle': dec_run['k_star'] == k_last,
            'k0': ind['k0'] == dec_run['k0'],
            'T': jtext(ind['T']) == jtext(dec_run['T']),
            'A': jtext(ind['A']) == jtext(dec_run['A']),
            'P_hat': ind.get('P_hat') == dec_run['P_hat'],
            'W': ind.get('W') == dec_run['W'],
            'window': ind.get('window') == dec_run['window'],
            'band': jtext(ind.get('band')) == jtext(dec_run['band']),
            'range': ind.get('range') == dec_run['range'],
            'range_le_tau': ind.get('range') is not None and ind['range'] <= SC.TAU,
            'swings_non_increasing': ind.get('swings_non_increasing') is True,
            'window_inside_run': ind.get('window_inside_run') is True,
            'Q_k_star': q[k_last] == dec_run['Q_k_star'],
            's_signed_eq_report': derived['s_signed'] == sr['s_signed'],
            'c_eq_report': derived['c'] == sr['report_only']['c_last_swing_ratio'],
            'c_A_over_1_minus_c_eq_report': derived['c_A_over_1_minus_c'] == sr['report_only']['c_A_last_over_1_minus_c'],
            'mid_band_minus_Q_N_eq_report': derived['mid_band_minus_Q_N'] == sr['report_only']['mid_band_minus_Q_N'],
            'Q_N_eq_report': q_n == sr['Q_cert_old_Q_N'],
            'no_lapse_after_N': not ind['lapse_cycles_after_N'],
        }
        if dec_run.get('branch') != 'oscillatory':
            for key in ('P_hat', 'W', 'window', 'band', 'range', 'swings_non_increasing', 'window_inside_run',
                        'range_le_tau'):
                comp[key] = None  # the monotone branch uses L_MONO; not re-derived here
        k2.update({'derived': derived, 'comparisons': comp,
                   'all_agree': all(v is not False for v in comp.values())})
    else:
        k2.update({'derived': None, 'comparisons': None, 'all_agree': None,
                   'note': 'decision not certified: independent re-derivation reported, not compared'})
    k3 = {'Q_N': q_n, 'Q_N_hex': float.hex(q_n), 'recert_row_N': q_n_ref, 'recert_row_N_hex': float.hex(q_n_ref),
          'bitwise_equal': float.hex(q_n) == float.hex(q_n_ref)}

    return {'cell': cell, 'N': n, 'CAP': cap, 'P_MAX': p_max, 'TAU': SC.TAU, 'EPS0': SC.EPS0, 'K_EXCL': SC.K_EXCL,
            'inputs': inputs, 'cycles_in_record': len(rows),
            'rule_decision_status': (dec_pure or {}).get('status'), 'rule_k_star': (dec_pure or {}).get('k_star'),
            'literal_first_certifying_k_after_N': lit_summary(lit_first_after_n),
            'literal_first_certifying_k_any_report_only': lit_summary(lit_first_any),
            'literal_at_last_cycle': by_k[rows[-1]['cycle']]['literal'],
            'K1_pure_rule_vs_run': k1, 'K2_independent_rederivation': k2, 'K3_Q_N_vs_recert': k3,
            'per_cycle': per_cycle}


def main():
    for p in (OUT, MANIFEST):
        if os.path.exists(p):
            raise SystemExit(f'REFUSED: {p} exists (write-once)')
    os.makedirs(OUT_DIR, exist_ok=True)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    doc = {'stage': 'P5.15 Addendum 53, W103 -- literal two-sign-change reading (REPORT-ONLY) + re-check of the '
                    'post-run settling decision; ZERO SOLVES',
           'utc': _utc(), 'git_head': head, 'stage_spec': {'path': STAGE_SPEC, 'sha256': sha256_file(STAGE_SPEC)},
           'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED (as spec v39)',
           'definitions': {
               'literal_reading': ('settling_criterion.literal_two_sign_change_report(rule, k) evaluated after each '
                                   'SettlingRule.observe(k, Q_k, boyd_k) over the committed per_cycle_record.jsonl; '
                                   'Q_k = gross_operational_cost, boyd_k = boyd_all_pass and local_solves_ok (as the '
                                   'harness post-run (k) check); literal k* = first k > N with literal_certifies'),
               'literal_function_doc': SC.literal_two_sign_change_report.__doc__,
               'c': 'A[-1] / A[-2]', 'c_A_over_1_minus_c': 'c A[-1] / (1 - c) (c < 1 only)',
               'mid_band_minus_Q_N': '(band_lo + band_hi) / 2 - Q_N', 's': 'Q_k* - Q_N'},
           'code_sha256': {f: sha256_file(f) for f in ('p515_s53_w103_literal_reading.py', 'settling_criterion.py',
                                                       'gate_result_io.py', 'p513_solve_profile_guard.py')},
           'cells': {}}
    for cell, info in CELLS.items():
        doc['cells'][cell] = cell_reading(cell, info)
        c = doc['cells'][cell]
        lit = c['literal_first_certifying_k_after_N']
        print(f'[W103] {cell}: N {c["N"]} CAP {c["CAP"]} rule {c["rule_decision_status"]} k* {c["rule_k_star"]}; '
              f'literal k* {None if lit is None else lit["k"]} '
              f'(Q-Q_N {None if lit is None else lit["Q_minus_Q_N"]}); '
              f'K1 {c["K1_pure_rule_vs_run"]["decision_reproduced_exactly"]}/'
              f'{c["K1_pure_rule_vs_run"]["every_cycle_record_reproduced"]}; K2 {c["K2_independent_rederivation"]["all_agree"]}; '
              f'K3 {c["K3_Q_N_vs_recert"]["bitwise_equal"]}', flush=True)
    doc['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    if doc['guard']['verify_0_failures']:
        raise RuntimeError(f'guard verify(0) failed: {doc["guard"]}')
    with open(OUT, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    files = [OUT, 'p515_s53_w103_literal_reading.py', 'settling_criterion.py', STAGE_SPEC]
    for c in doc['cells'].values():
        files += [v['path'] for v in c['inputs'].values()]
    manifest = {f: sha256_file(f) for f in sorted(set(files))}
    with open(MANIFEST, 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    print(f'[W103] wrote {OUT} and {MANIFEST} ({len(manifest)} entries); guard {doc["guard"]}', flush=True)


if __name__ == '__main__':
    main()
