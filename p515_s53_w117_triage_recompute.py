"""P5.15 W117 (Addendum 57 Decision 2) -- the SRP1 triage recomputed from records under the new determinacy rule.
ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Nothing is built, unpickled or solved; only committed / hash-recorded JSON(L) records are read.

RULE (Addendum 57 Decision 2, old certificates not re-run). Q = gross_operational_cost (settlement excluded);
t_sum = t_tso_plus_t_dso_terminal (interface_settlement_detail_s31c.json); Q_cc = Q + t_sum. A difference is DETERMINATE
only if its margin exceeds BAR = 3 x max(S, |t_sum| of the cell) in BOTH Q and Q_cc terms, else PENDING.
S = 20,806.323693990707 EUR (the settled unit slack, W112 task5 s_gross; = triage threshold / 3).
  "The cell" = the non-reference cell of the difference. When BOTH cells are old certificates (both carry a gap) this
  script uses the LARGER |t_sum| of the two (reading R1, primary, the Planner's W117 instruction). Reading R0 (the
  non-reference cell only) is computed beside it for every claim, and the verdicts that differ are listed.
  OLD verdict (Addendum 53 Ruling 2): flagged iff margin in Q terms < 3 x S = 62,418.971 EUR.

MARGIN. margin_Q = |d_Q|. margin_Qcc = |d_Qcc| if sign(d_Qcc) == sign(d_Q), else -|d_Qcc| (a sign that flips between the
two measures is never determinate). Claim types (recorded per claim):
  'sign'      -- the sign of a difference of F = I + Q (or of value); margin = |difference|;
  'threshold' -- a value difference against an investment cost (value - I, or delta-value - delta-I); margin = distance
                 of the value difference to the threshold I.
FORMULAS (d_Qcc: every Q replaced by Q + t_sum):
  F-type      d = [Q(o) + I(o) - SV(o)] - [Q(r) + I(r) - SV(r)],   SV = terminal salvage if the claim is net of salvage
                (year ladder, net convention) else 0;
  value-type  d = [Q(r) - Q(o)] - [I(o) - I(r)]                   (value - I; delta-value - delta-I).
I(x) is read from the record the committed report cites for that table (per claim: `I_source`); nothing is recomputed.

Item D (break-even) is a 10-point OLS coefficient, not a two-cell difference: it gets no verdict. Its fit is recomputed
in Q and in Q_cc with the formula of p515_s47_baseline_tables.py (reproduction of the committed Q figures asserted).

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w117_triage_recompute/):
  w117_triage_recompute.json, launch.log, manifest_sha256.json
Smoke (no file written): add --dry. Launch (attached, alone, both streams captured), then the manifest:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w117_triage_recompute.py \\
        > data/SRP1/Results/P515S53/w117_triage_recompute/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w117_triage_recompute.py --manifest
"""
import glob
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

_GUARD = SolveProfileGuard((), label='P5.15 W117 triage recompute zero-solve').install()

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results')
OUT_DIR = os.path.join(REPO, RES, 'P515S53', 'w117_triage_recompute')
OUT_JSON = os.path.join(OUT_DIR, 'w117_triage_recompute.json')

TRIAGE = os.path.join(RES, 'P515S53', 'triage_list_gap_join', 'triage_list_gap_join.json')
W112 = os.path.join(RES, 'P515S53', 'w112_consensus_gap', 'w112_consensus_gap.json')
W2 = os.path.join(RES, 'P515S45', 'investment_cost', 'investment_cost_results.json')

CAMPAIGNS = {
    'S45_a0_c7': os.path.join(RES, 'P515S45', 'campaign_s45_a0_c7'),
    'S45_a1a': os.path.join(RES, 'P515S45', 'campaign_s45_a1a'),
    'S45_a1b': os.path.join(RES, 'P515S45', 'campaign_s45_a1b'),
    'S46_ageing': os.path.join(RES, 'P515S46', 'campaign_s46_ageing'),
    'S47_a1a': os.path.join(RES, 'P515S47', 'campaign_s47_a1a_baseline'),
    'S47_recert': os.path.join(RES, 'P515S47', 'campaign_s47_recert'),
    'S47_phase_b': os.path.join(RES, 'P515S47', 'campaign_s47_phase_b'),
    'S49_flex': os.path.join(RES, 'P515S49', 'campaign_s49_flex_ladder'),
    'S50_marginal': os.path.join(RES, 'P515S50', 'campaign_s50_marginal'),
    'S51_f2_ladder': os.path.join(RES, 'P515S51', 'campaign_s51_f2_ladder'),
    'S51_f2_phase_b': os.path.join(RES, 'P515S51', 'campaign_s51_f2_phase_b'),
    'S53_f2_cert': os.path.join(RES, 'P515S53', 'campaign_s53_f2_certificate_r1'),
    'W101_x0': os.path.join(RES, 'P515S53', 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0'),
    'W101_unit': os.path.join(RES, 'P515S53', 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1'),
}
# where an eval key has copies in several campaigns, the one used (copies asserted identical in Q, t_sum, salvage)
PREFERRED = {'7eb1ce62': 'S45_a1a', 'bd504ecf': 'S47_a1a'}

CAMPAIGN_10 = ['a30a9faf', 'd7030f59', '4a852725', 'd0c1f160', '1bff3ed2', '10c73abd',
               '549476cd', 'dab6a8a2', 'e28de4ac', '5ca4f86c']
SETTLED_COUNTERPART = {
    '7aa017f0': 'x0 settled Q181 (W101 d110bd1a5977df1e_x0, cert 181; a continuation of the tight-tail x0 re-cert '
                '5cfe69a6, same candidate 8435c718)',
    'bd504ecf': 'unit settled Q172 (W101 3f084f2ffaeef2b7_n7_4h_e1, cert 172; a continuation of the tight-tail unit '
                're-cert ca8927e7, same candidate db77e154)',
}
X0_OLD = '7aa017f0'
EVALREC = 'evaluation_record.json'
DETAIL = 'interface_settlement_detail_s31c.json'
PCR = 'per_cycle_record.jsonl'


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


def _pin(rel, manifest_rel):
    """sha256 of an input against the manifest that hash-records it; raise on a mismatch or a missing entry."""
    now = _sha(os.path.join(REPO, rel))
    man = _load(manifest_rel)
    pinned = man.get(rel)
    if pinned is None and isinstance(man.get('files'), dict):  # the W2-era manifest form {files: {rel: {sha256}}}
        pinned = (man['files'].get(rel) or {}).get('sha256')
    if pinned is None:
        raise RuntimeError(f'{rel}: not hash-recorded in {manifest_rel}')
    if pinned != now:
        raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned} ({manifest_rel})')
    INPUTS[rel] = {'sha256': now, 'manifest': manifest_rel}


def _pin_git(rel):
    """an input with no manifest: it must be tracked and unmodified against HEAD; the last commit touching it is recorded."""
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], capture_output=True, cwd=REPO).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise RuntimeError(f'{rel}: tracked {tracked}, clean {clean}')
    commit = subprocess.run(['git', 'log', '-1', '--format=%H', '--', rel], capture_output=True, text=True,
                            cwd=REPO).stdout.strip()
    INPUTS[rel] = {'sha256': _sha(os.path.join(REPO, rel)), 'manifest': None, 'git_tracked_clean_commit': commit}


def _points(res):
    pts = res['points']
    return list(pts.values()) if isinstance(pts, dict) else pts


def _walk_I(obj, acc):
    """every dict carrying an eval_key and an investment cost (I_x_eur, or I beside z) -> acc[eval_key] = set(I)."""
    if isinstance(obj, dict):
        key = obj.get('eval_key')
        val = obj.get('I_x_eur')
        if val is None and 'z' in obj:
            val = obj.get('I')
        if key and val is not None:
            acc.setdefault(key, set()).add(float(val))
        for v in obj.values():
            _walk_I(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_I(v, acc)


# ---------------------------------------------------------------------------------------------------------------------
def _read_cell(camp_name, eval_dir_rel):
    man = os.path.join(CAMPAIGNS[camp_name], 'campaign_manifest_sha256.json')
    for name in (EVALREC, DETAIL, PCR):
        _pin(os.path.join(eval_dir_rel, name), man)
    rec = _load(os.path.join(eval_dir_rel, EVALREC))
    det = _load(os.path.join(eval_dir_rel, DETAIL))
    last = None
    with open(os.path.join(REPO, eval_dir_rel, PCR)) as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    k = rec['certification_cycle']
    q = rec['terminal_gross_operational_cost']
    assert rec['status'] == 'certified', eval_dir_rel
    assert rec['certified_cost'] == q, eval_dir_rel
    assert last['cycle'] == k == rec['cycles_run'], (eval_dir_rel, last['cycle'], k)
    assert last['gross_operational_cost'] == q, eval_dir_rel
    assert det['cycles_run'] == k, (eval_dir_rel, det['cycles_run'], k)
    t = det['t_tso_plus_t_dso_terminal']
    t_rec = rec['recourse_components']['interface_settlement_total']
    assert abs(t - t_rec) <= 1e-6 * max(1.0, abs(t)), (eval_dir_rel, t, t_rec)
    sv = rec['recourse_components']['terminal_salvage_value']
    assert rec['recourse_components']['gross_operational_cost'] == q, eval_dir_rel
    assert abs((q - sv) - rec['terminal_net_operational_recourse']) <= 1e-6, eval_dir_rel
    return {'campaign': camp_name, 'eval_dir': eval_dir_rel, 'eval_key': rec['eval_key'],
            'candidate_key': rec['candidate_key'], 'candidate_canonical': rec['candidate_canonical'],
            'label': rec.get('candidate_label'), 'status': rec['status'], 'certification_cycle': k,
            'Q': q, 't_sum': t, 't_sum_record_minus_detail': t_rec - t, 'Q_cc': q + t,
            'salvage': sv, 'net_operational_recourse': rec['terminal_net_operational_recourse'],
            'bar_gross_step': (rec.get('bar') or {}).get('value')}


def build_cells():
    """index every eval dir of the listed campaigns by eval_key (first 8 hex)."""
    idx = {}
    for name, root in CAMPAIGNS.items():
        for d in sorted(glob.glob(os.path.join(REPO, root, 'evals', '*'))):
            if not os.path.isfile(os.path.join(d, EVALREC)):
                continue
            idx.setdefault(os.path.basename(d)[:8], []).append((name, os.path.relpath(d, REPO)))
    return idx


CELL_CACHE = {}


def cell(prefix, idx):
    if prefix in CELL_CACHE:
        return CELL_CACHE[prefix]
    hits = idx.get(prefix, [])
    if not hits:
        raise RuntimeError(f'no eval dir for {prefix}')
    reads = [_read_cell(c, d) for c, d in hits]
    for r in reads[1:]:
        for f in ('eval_key', 'Q', 't_sum', 'salvage', 'certification_cycle'):
            assert r[f] == reads[0][f], (prefix, f, r[f], reads[0][f])
    if len(hits) > 1 and prefix not in PREFERRED:
        raise RuntimeError(f'{prefix}: several eval dirs {hits} and no preferred campaign declared')
    pick = PREFERRED[prefix] if len(hits) > 1 else hits[0][0]
    out = next(r for r in reads if r['campaign'] == pick)
    out['copies'] = [r['eval_dir'] for r in reads]
    CELL_CACHE[prefix] = out
    return out


# ---------------------------------------------------------------------------------------------------------------------
def I_index():
    """eval_key -> I(x) from the campaign_results the committed reports cite (all copies must agree)."""
    acc = {}
    for name, root in CAMPAIGNS.items():
        rel = os.path.join(root, 'campaign_results.json')
        if not os.path.exists(os.path.join(REPO, rel)):
            continue
        _pin(rel, os.path.join(root, 'campaign_manifest_sha256.json'))
        _walk_I(_load(rel), acc)
    out = {}
    for k, vals in acc.items():
        lo, hi = min(vals), max(vals)
        assert hi - lo <= 1e-6, (k, vals)
        out[k[:8]] = lo
    return out


def w2_I(label, canonical):
    w2 = _load(W2)
    c = w2['candidates'][label]
    assert c['candidate_canonical'] == canonical, (label, c['candidate_canonical'], canonical)
    return c['I_new_eur']


# ---------------------------------------------------------------------------------------------------------------------
def evaluate(claim, S, thr_old):
    r, o = claim['ref'], claim['other']
    Ir, Io = claim['I_ref'], claim['I_other']
    net = claim.get('net_of_salvage', False)
    svr = r['salvage'] if net else 0.0
    svo = o['salvage'] if net else 0.0

    def d(qr, qo):
        if claim['form'] == 'F':
            return (qo + Io - svo) - (qr + Ir - svr)
        return (qr - qo) - (Io - Ir)

    dq = d(r['Q'], o['Q'])
    dcc = d(r['Q_cc'], o['Q_cc'])
    m_q = abs(dq)
    m_cc = abs(dcc) if (dq > 0) == (dcc > 0) else -abs(dcc)
    both_old = claim['ref_kind'] == 'old' and claim['other_kind'] == 'old'
    t_r1 = max(abs(r['t_sum']), abs(o['t_sum'])) if both_old else abs(o['t_sum'])
    t_r0 = abs(o['t_sum'])
    bar_r1 = 3.0 * max(S, t_r1)
    bar_r0 = 3.0 * max(S, t_r0)
    det_r1 = m_q > bar_r1 and m_cc > bar_r1
    det_r0 = m_q > bar_r0 and m_cc > bar_r0
    out = {k: v for k, v in claim.items() if k not in ('ref', 'other')}
    out.update({
        'ref': {k: r[k] for k in ('eval_key', 'eval_dir', 'candidate_key', 'label', 'certification_cycle', 'Q', 't_sum',
                                  'Q_cc', 'salvage')},
        'other': {k: o[k] for k in ('eval_key', 'eval_dir', 'candidate_key', 'label', 'certification_cycle', 'Q',
                                    't_sum', 'Q_cc', 'salvage')},
        'd_Q': dq, 'd_Qcc': dcc, 'd_Qcc_minus_d_Q': dcc - dq, 'margin_Q': m_q, 'margin_Qcc': m_cc,
        'sign_same_in_Q_and_Qcc': (dq > 0) == (dcc > 0),
        't_used_R1': t_r1, 'bar_R1': bar_r1, 'verdict_R1': 'determinate' if det_r1 else 'pending',
        't_used_R0': t_r0, 'bar_R0': bar_r0, 'verdict_R0': 'determinate' if det_r0 else 'pending',
        'old_verdict_add53': 'flagged' if m_q < thr_old else 'not flagged',
        'margin_Q_over_bar_R1': m_q / bar_r1, 'margin_Qcc_over_bar_R1': m_cc / bar_r1,
    })
    if claim.get('recorded') is not None:
        out['recorded_minus_recomputed_d_Q'] = claim['recorded'] - dq
    return out


def main():
    t0 = time.time()
    dry = '--dry' in sys.argv
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    _pin(TRIAGE, os.path.join(os.path.dirname(TRIAGE), 'manifest_sha256.json'))
    _pin(W112, os.path.join(os.path.dirname(W112), 'manifest_sha256.json'))
    _pin(W2, os.path.join(os.path.dirname(W2), 'manifest_sha256.json'))
    triage = _load(TRIAGE)
    w112 = _load(W112)
    S = w112['task5']['references']['unit']['s_gross']
    thr_old = triage['threshold_eur']
    assert abs(3.0 * S - thr_old) < 1e-6, (S, thr_old)
    idx = build_cells()
    Ix = I_index()
    claims = []

    def add(item, cid, statement, ctype, form, ref, oth, I_ref, I_oth, I_source, recorded=None, recorded_source=None,
            net=False, ref_kind='old', other_kind='old', note=None):
        claims.append({'item': item, 'claim_id': cid, 'statement': statement, 'claim_type': ctype, 'form': form,
                       'ref': ref, 'other': oth, 'I_ref': I_ref, 'I_other': I_oth, 'I_source': I_source,
                       'net_of_salvage': net, 'ref_kind': ref_kind, 'other_kind': other_kind,
                       'recorded': recorded, 'recorded_source': recorded_source, 'note': note})

    x0 = cell(X0_OLD, idx)
    assert abs(x0['Q'] - 653859461.2279255) < 1e-6

    # ---- B: Phase A ladder under the baseline (S47 A1a re-run, 30 points): F(x) - F(0) > 0 ------------------------
    a1a = _load(os.path.join(CAMPAIGNS['S47_a1a'], 'campaign_results.json'))
    for p in _points(a1a):
        c = cell(p['eval_key'][:8], idx)
        assert abs(p['Q0_eur'] - x0['Q']) < 1e-6
        add('B', f"B:{p['label']}", f"x = 0 minimises F: F({p['label']}) - F(0) > 0 (S47 A1a baseline ladder)",
            'sign', 'F', x0, c, 0.0, Ix[c['eval_key'][:8]],
            'S47 A1a campaign_results.json points[].I_x_eur (the W2 I(x) table 9e623dd3)',
            recorded=p['F_eur'] - x0['Q'], recorded_source='S47 A1a campaign_results F_eur - Q(0)')

    # ---- C: Phase B formal record, 14 neighbours of x = 0 ----------------------------------------------------------
    pb = _load(os.path.join(CAMPAIGNS['S47_phase_b'], 'campaign_results.json'))
    for lab, p in pb['points'].items():
        c = cell(p['eval_key'][:8], idx)
        add('C', f'C:{lab}', f'Phase B certificate at x = 0: F({lab}) - F(0) > 0', 'sign', 'F', x0, c, 0.0,
            Ix[c['eval_key'][:8]], 'S47 Phase B campaign_results.json poll_history[].candidates[].I_x_eur',
            recorded=Ix[c['eval_key'][:8]] + p['certified_cost_gross_settlement_excluded'] - x0['Q'],
            recorded_source='S47 Phase B campaign_results certified_cost + I - Q(0)')

    # ---- E: ageing variants (S46) and the C3 unit ------------------------------------------------------------------
    c3 = cell('7eb1ce62', idx)
    I_unit = Ix['7eb1ce62']
    s45a1a = {p['eval_key'][:8]: p for p in _points(_load(os.path.join(CAMPAIGNS['S45_a1a'], 'campaign_results.json')))}
    add('E', 'E:C3_unit_value_minus_I', 'C3 unit does not pay: value - I < 0', 'threshold', 'value', x0, c3, 0.0,
        I_unit, 'S45 A1a campaign_results.json points[n7_4h_e1].I_x_eur (W2 table 9e623dd3)',
        recorded=-(s45a1a['7eb1ce62']['F_eur'] - x0['Q']), recorded_source='S45 A1a campaign_results -(F_eur - Q(0))')
    ag = _load(os.path.join(CAMPAIGNS['S46_ageing'], 'campaign_results.json'))
    for p in _points(ag):
        c = cell(p['eval_key'][:8], idx)
        add('E', f"E:{p['label']}:value_minus_I", f"ageing variant {p['label']}: sign of value - I", 'threshold',
            'value', x0, c, 0.0, p['I_x_eur'], 'S46 ageing campaign_results.json points[].I_x_eur',
            recorded=p['value_minus_I_eur'], recorded_source='S46 campaign_results value_minus_I_eur')
        add('E', f"E:{p['label']}:vs_C3", f"ageing variant {p['label']} vs the C3 unit: sign of value(variant) - value(C3)",
            'sign', 'value', c3, c, 0.0, 0.0, 'none (same candidate, I cancels)',
            recorded=p['value_eur'] - p['value_C3_eur'], recorded_source='S46 campaign_results value_eur - value_C3_eur')

    # ---- G: year ladder (S45 A1b, C3-era), gross and net of salvage ------------------------------------------------
    a1b = _load(os.path.join(CAMPAIGNS['S45_a1b'], 'campaign_results.json'))
    for p in _points(a1b):
        c = cell(p['eval_key'][:8], idx)
        for net in (False, True):
            conv = 'net of salvage' if net else 'gross'
            add('G', f"G:{p['label']}:{'net' if net else 'gross'}",
                f"year ladder, x = 0 wins ({conv}): F({p['label']}) - F(0) > 0", 'sign', 'F', x0, c, 0.0,
                p['I_x_eur'], 'S45 A1b campaign_results.json points[].I_x_eur (W2 table 9e623dd3)',
                recorded=(p['F_eur'] - x0['Q'] - (p['terminal_salvage_value'] if net else 0.0)),
                recorded_source='S45 A1b campaign_results F_eur - Q(0) (- terminal_salvage_value if net)', net=net)

    # ---- H: flexibility ladder, value - I at m = 1.5, 2, 3 --------------------------------------------------------
    for m, xk, uk in (('1.5', 'aa8a76d7', 'f9eae48f'), ('2', '50dea31c', '74eda68d'), ('3', '75c65e73', '05130541')):
        xr, u = cell(xk, idx), cell(uk, idx)
        add('H', f'H:m{m}:value_minus_I', f'flexibility ladder m = {m}: sign of value - I (node 7, 0.25 MVA / 1 MWh)',
            'threshold', 'value', xr, u, 0.0, w2_I('n7_4h_e1', u['candidate_canonical']),
            'W2 I(x) table investment_cost_results.json candidates[n7_4h_e1].I_new_eur (9e623dd3)')

    # ---- I: the second MWh at x2 (and x3), and value - I of the 2 MWh cell ----------------------------------------
    for m, xk, e1k, e2k in (('2', '50dea31c', '74eda68d', '5a6a88b4'), ('3', '75c65e73', '05130541', '9aae65d5')):
        xr, e1, e2 = cell(xk, idx), cell(e1k, idx), cell(e2k, idx)
        I1 = w2_I('n7_4h_e1', e1['candidate_canonical'])
        I2 = w2_I('n7_4h_e2', e2['candidate_canonical'])
        add('I', f'I:m{m}:second_MWh', f'm = {m}: the second MWh pays: [Q(e1) - Q(e2)] - [I(e2) - I(e1)] > 0',
            'threshold', 'value', e1, e2, I1, I2, 'W2 I(x) table candidates[n7_4h_e1/e2].I_new_eur (9e623dd3)')
        add('I', f'I:m{m}:e2_value_minus_I', f'm = {m}: node 7 0.5 MVA / 2 MWh pays: value - I > 0', 'threshold',
            'value', xr, e2, 0.0, I2, 'W2 I(x) table candidates[n7_4h_e2].I_new_eur (9e623dd3)')

    # ---- J: F2 ladder (m = 2), marginal MWh e2->e3, e3->e4, e4->e5, and value - I of e3..e5 -------------------------
    xm2 = cell('50dea31c', idx)
    lad = [('e2', '5a6a88b4'), ('e3', 'f3aa335e'), ('e4', 'a11d7966'), ('e5', '5f3cccb4')]
    for (ea, ka), (eb, kb) in zip(lad[:-1], lad[1:]):
        ca, cb = cell(ka, idx), cell(kb, idx)
        Ia = w2_I(f'n7_4h_{ea}', ca['candidate_canonical'])
        Ib = w2_I(f'n7_4h_{eb}', cb['candidate_canonical'])
        add('J', f'J:{ea}_to_{eb}', f'F2 ladder: the marginal MWh {ea} -> {eb} pays', 'threshold', 'value', ca, cb, Ia,
            Ib, f'W2 I(x) table candidates[n7_4h_{ea}/{eb}].I_new_eur (9e623dd3)')
    for eb, kb in lad[1:]:
        cb = cell(kb, idx)
        add('J', f'J:{eb}_value_minus_I', f'F2 ladder: node 7 4 h {eb} pays: value - I > 0', 'threshold', 'value', xm2,
            cb, 0.0, w2_I(f'n7_4h_{eb}', cb['candidate_canonical']),
            f'W2 I(x) table candidates[n7_4h_{eb}].I_new_eur (9e623dd3)')

    # ---- L: F2 certificate (S53 r1): poll set (10) + cached box neighbours (7) vs the incumbent 5ca4f86c ------------
    cert = _load(os.path.join(CAMPAIGNS['S53_f2_cert'], 'campaign_results.json'))
    tc = cert['termination_certificate']
    inc = cell('5ca4f86c', idx)
    I_inc = cert['final_incumbent']['I']
    assert cert['final_incumbent']['eval_key'].startswith('5ca4f86c') and abs(cert['final_incumbent']['Q'] - inc['Q']) < 1e-6
    lab2key = {}
    for ph in cert['poll_history']:
        for cand in ph['candidates']:
            if cand.get('eval_key'):
                lab2key[cand['label']] = cand['eval_key']
    rows = [('poll_set', r['label'], lab2key[r['label']], r['F_eur'] - cert['final_incumbent']['F'])
            for r in tc['poll_set']]
    rows += [('cached_box_neighbour', r['label'], r['eval_key'], r['F_minus_F_inc_eur'])
             for r in tc['cached_box_neighbours']['rows']]
    for part, lab, key, rec in rows:
        c = cell(key[:8], idx)
        add('L', f'L:{lab}', f'F2 certificate: F({lab}) - F(incumbent) (positive = worse than the incumbent; the '
            'certificate needs no determinate improvement)', 'sign', 'F', inc, c, I_inc, Ix[key[:8]],
            'S53 F2 certificate campaign_results.json (final_incumbent.I; candidates[].I_x_eur / cached rows I_x_eur)',
            recorded=rec, recorded_source=f'S53 certificate {part} F - F_inc', note=part)
    # df1a5525: named by the Addendum 53 triage inventory for L, but NOT a unit neighbour of the incumbent
    c = cell('df1a5525', idx)
    add('L', 'L:df1a5525_not_a_neighbour', 'F(y2030__n5_p0.25_e0.5__n7_p0.75_e2.5) - F(incumbent): listed by the '
        'Addendum 53 inventory under L; NOT in the certificate poll set or its cached box neighbours (zE7 differs by '
        '2 lattice units)', 'sign', 'F', inc, c, I_inc, Ix['df1a5525'],
        'S53 F2 certificate campaign_results.json budget_facts.cache_rows I_x_eur', note='triage_listed_non_neighbour')

    # ---- CHECK ROW: headline sign from the settled references --------------------------------------------------------
    xs, us = cell('d110bd1a', idx), cell('3f084f2f', idx)
    add('CHECK', 'CHECK:headline_V_minus_I_settled', 'headline: x = 0 optimal at SRP1, V - I < 0 (settled references)',
        'threshold', 'value', xs, us, 0.0, w2_I('n7_4h_e1', us['candidate_canonical']),
        'W2 I(x) table candidates[n7_4h_e1].I_new_eur (9e623dd3)', recorded=253539.62 - 317957.01,
        recorded_source='P5_15_ADDENDUM53 report: V settled 253,539.62, I 317,957.01', ref_kind='settled',
        other_kind='settled')

    results = [evaluate(c, S, thr_old) for c in claims]

    # ---- D: break-even fit (no verdict) -----------------------------------------------------------------------------
    unit = _load(W2)['candidates']['n7_4h_e1']
    p_cost, e_cost = unit['I_new_power_eur'] / 0.25, unit['I_new_energy_eur'] / 1.0
    n7 = [p for p in _points(a1a) if p['label'].startswith('n7_')]
    X = np.array([[1.0, p['candidate_canonical']['nodes']['7'][1], p['candidate_canonical']['nodes']['7'][0]]
                  for p in n7])
    fits = {}
    for conv in ('Q', 'Q_cc'):
        q0 = x0['Q'] if conv == 'Q' else x0['Q_cc']
        y = np.array([q0 - cell(p['eval_key'][:8], idx)[conv] for p in n7])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        s2 = float(res @ res) / (len(y) - 3)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        a, b, cc = (float(v) for v in coef)
        e_star = b + cc / 4.0 - p_cost / 4.0
        se_e = float(np.sqrt(se[1] ** 2 + (se[2] / 4.0) ** 2))
        fits[conv] = {'a': a, 'b_per_MWh': b, 'c_per_MVA': cc, 'se': [float(v) for v in se],
                      'breakeven_marginal_4h_energy_cost': e_star, 'margin_to_energy_cost_per_MWh': e_cost - e_star,
                      'se_breakeven_independent_terms_approx': se_e, 'residual_rms': float(np.sqrt(res @ res / len(y)))}
    bt_rel = os.path.join(RES, 'P515S47', 'baseline_tables', 'baseline_tables.json')
    _pin_git(bt_rel)
    bt = _load(bt_rel)
    assert abs(fits['Q']['b_per_MWh'] - bt['node7_fit']['b']) < 1e-6, 'D fit does not reproduce the committed b'
    assert abs(fits['Q']['breakeven_marginal_4h_energy_cost'] - bt['breakeven']['marginal_4h_eur_per_mwh']) < 1e-6
    assert abs(fits['Q']['se_breakeven_independent_terms_approx'] - bt['breakeven']['marginal_4h_se_eur_per_mwh']) < 1e-6
    D = {'status': 'NOT A DIFFERENCE: the break-even is a 10-point OLS coefficient (value = a + b E + c P over the S47 A1a '
                   'node-7 points), a linear combination of 11 cells, not a difference of two; no verdict under the '
                   'Decision 2 rule. The fit is recomputed in Q and in Q_cc terms as supplementary information.',
         'formula': 'value = Q(0) - Q(x) (Q_cc: + t_sum); e* = b + c/4 - p_cost/4; margin = e_cost - e* '
                    '(p515_s47_baseline_tables.py)',
         'p_cost_per_MVA': p_cost, 'e_cost_per_MWh': e_cost, 'fits': fits,
         'committed_baseline_tables_breakeven': bt.get('breakeven'),
         'cells': [{'label': p['label'], 'eval_key': p['eval_key'],
                    't_sum': cell(p['eval_key'][:8], idx)['t_sum']} for p in n7]}

    # ---- summaries ---------------------------------------------------------------------------------------------------
    def cells_of(rs):
        out = set()
        for r in rs:
            out.add(r['ref']['eval_key'][:8])
            out.add(r['other']['eval_key'][:8])
        return out

    triage33 = []
    for item, v in triage['items'].items():
        if v['group'] == 'flagged_62k':
            triage33 += [c['prefix'] for c in v['cells']]
    triage9 = [c['prefix'] for c in triage['items']['D_break_even_fit']['cells']]
    graded = [r for r in results if r['item'] != 'CHECK']
    old_flag = [r for r in graded if r['old_verdict_add53'] == 'flagged']
    new_pend = [r for r in graded if r['verdict_R1'] == 'pending']
    new_pend_r0 = [r for r in graded if r['verdict_R0'] == 'pending']
    movement = {
        'n_claims_graded': len(graded),
        'claims_old_flagged': len(old_flag),
        'claims_new_pending_R1': len(new_pend), 'claims_new_pending_R0': len(new_pend_r0),
        'claims_newly_pending_R1': [r['claim_id'] for r in graded
                                    if r['old_verdict_add53'] == 'not flagged' and r['verdict_R1'] == 'pending'],
        'claims_newly_determinate_R1': [r['claim_id'] for r in graded
                                        if r['old_verdict_add53'] == 'flagged' and r['verdict_R1'] == 'determinate'],
        'claims_newly_pending_R0': [r['claim_id'] for r in graded
                                    if r['old_verdict_add53'] == 'not flagged' and r['verdict_R0'] == 'pending'],
        'claims_newly_determinate_R0': [r['claim_id'] for r in graded
                                        if r['old_verdict_add53'] == 'flagged' and r['verdict_R0'] == 'determinate'],
        'claims_R1_vs_R0_differ': [r['claim_id'] for r in graded if r['verdict_R1'] != r['verdict_R0']],
    }
    newly_pending_claims = [r for r in graded if r['old_verdict_add53'] == 'not flagged' and r['verdict_R1'] == 'pending']
    old_cells = cells_of(old_flag)
    new_cells = cells_of(new_pend)
    new_cells_r0 = cells_of(new_pend_r0)
    t33 = set(triage33)
    rerun = {
        'triage_33': triage33, 'triage_extra_9_D': triage9,
        'old_rule_recomputed_cells': sorted(old_cells),
        'old_rule_recomputed_vs_triage33': {'in_recomputed_not_in_33': sorted(old_cells - t33),
                                            'in_33_not_in_recomputed': sorted(t33 - old_cells)},
        'new_rule_R1_cells': sorted(new_cells), 'n_new_rule_R1_cells': len(new_cells),
        'new_R1_cells_not_in_33': sorted(new_cells - t33),
        'of_which_old_rule_flagged_but_not_in_33 (inventory scope, not the bar)': sorted((new_cells - t33) & old_cells),
        'of_which_newly_pending_by_the_bar (in a claim flagged under neither rule before)':
            sorted((new_cells - t33) - old_cells),
        'cells_of_newly_pending_claims_R1 (claim-level, whether or not in the 33)': sorted(cells_of(newly_pending_claims)),
        'note_newly_determinate': 'impossible by construction under R1 and R0: every bar is >= 3 x S = the old threshold, '
                                  'so a claim flagged under Addendum 53 (margin_Q < 3 x S) cannot clear the new bar',
        'newly_determinate_cells_R1 (in the 33, in no pending claim)': sorted(t33 - new_cells),
        'new_rule_R0_cells': sorted(new_cells_r0), 'n_new_rule_R0_cells': len(new_cells_r0),
        'newly_pending_cells_R0': sorted(new_cells_r0 - t33),
        'newly_determinate_cells_R0': sorted(t33 - new_cells_r0),
        'settled_counterparts': SETTLED_COUNTERPART,
        'campaign_10': {k: {'on_new_R1_list': k in new_cells, 'on_new_R0_list': k in new_cells_r0,
                            'in_triage_33': k in t33} for k in CAMPAIGN_10},
        'new_R1_cells_not_in_campaign_10': sorted(new_cells - set(CAMPAIGN_10)),
    }
    cell_table = {k: {f: v[f] for f in ('eval_dir', 'label', 'eval_key', 'candidate_key', 'certification_cycle', 'Q',
                                        't_sum', 'Q_cc', 'salvage', 'bar_gross_step', 'copies')}
                  for k, v in sorted(CELL_CACHE.items())}

    guard_failures = _GUARD.verify(0)
    out = {
        'stage': 'P5.15 W117 -- Addendum 57 Decision 2: SRP1 triage recomputed from records under the new determinacy '
                 'rule (zero solves)',
        'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script_sha256': _sha(THIS),
        'objective_convention': 'Q = gross_operational_cost (settlement excluded); t_sum = t_tso_plus_t_dso_terminal; '
                                'Q_cc = Q + t_sum. F = I + Q; value = Q(ref) - Q(x). Salvage excluded unless the claim '
                                'is stated net of salvage (year ladder, net rows).',
        'rule': {'S_eur': S, 'S_source': 'w112_consensus_gap.json task5.references.unit.s_gross',
                 'bar': '3 x max(S, |t_sum|)', 'old_threshold_eur': thr_old,
                 'reading_R1_primary': 'two old cells: |t_sum| = the larger of the two; old vs settled or settled vs '
                                       'settled: the non-reference cell',
                 'reading_R0': 'the non-reference cell only, always',
                 'determinate': 'margin_Q > bar AND margin_Qcc > bar (margin_Qcc negative when the sign flips)'},
        'inputs_sha256': INPUTS,
        'claims': results,
        'D_break_even': D,
        'movement': movement,
        'rerun_list': rerun,
        'cells': cell_table,
        'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures, 'counts': dict(_GUARD.counts)},
        'wall_s': time.time() - t0,
    }
    if dry:
        GRIO.dumps(out, indent=1, sort_keys=True, default=GRIO.json_default_item)  # the writer's typing check, no file
        _log('--dry: nothing written')
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_JSON, 'x') as handle:
            GRIO.dump(out, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)

    # ---- log table -----------------------------------------------------------------------------------------------
    _log(f'S = {S}; old threshold {thr_old}; {len(INPUTS)} inputs pinned; {len(CELL_CACHE)} cells; {len(results)} claims')
    for r in results:
        _log(f"{r['claim_id']:62s} ref {r['ref']['eval_key'][:8]} (t {r['ref']['t_sum']:>10.1f}) other "
             f"{r['other']['eval_key'][:8]} (t {r['other']['t_sum']:>10.1f}) dQ {r['d_Q']:>13.2f} dQcc {r['d_Qcc']:>13.2f}"
             f" bar {r['bar_R1']:>10.1f} R1 {r['verdict_R1']:11s} R0 {r['verdict_R0']:11s} old {r['old_verdict_add53']}"
             + (f" rec-recomp {r['recorded_minus_recomputed_d_Q']:.3g}" if 'recorded_minus_recomputed_d_Q' in r else ''))
    _log(f'D fits: {json.dumps(fits)}')
    _log(f'movement: {json.dumps(movement)}')
    _log(f'rerun: {json.dumps(rerun)}')
    _log(f'guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {time.time() - t0:.1f} s')
    return 0 if not guard_failures else 1


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
