"""P5.15 Addendum 62 item 3, Planner task W145 -- THE BANDED BREAK-EVEN FIT SCORER (pre-registered).
ZERO SOLVES, NO MODEL LOADS.

Committed BEFORE any of the five remaining D-cell results (d_d3709599, d_a12d95a2, d_f759dd48, d_c7fee8be, d_9246ed01)
exist, so the formula is fixed before the data it scores. `--score` refuses to run unless this file is committed clean.

GUARDS. An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly
0 at the end, together with every imported module's own zero-permit guard; `pickle.load` / `pickle.loads` are blocked
for the whole run (W141's pattern) and their call counts verified at 0. Only JSON / JSONL records are read.

THE FIT IS W132's, CALLED, NOT COPIED. Every fit below is `p515_s53_w132_resettle_v3_campaign.d_fit` (the function the
v3 / v6 summarizers use, which reproduces the committed Phase A fit `baseline_tables.json` node7_fit within 1e-6):
    value_i = Q(0) - Q(x_i);  value = a + b E + c P, OLS (np.linalg.lstsq);  SE = sqrt(diag(sigma^2 (X^T X)^-1)),
    sigma^2 = RSS / (n - 3);  e* = b + c/4 - p_cost/4 (the marginal-4h break-even, EUR/MWh);
    se_e* = sqrt(se_b^2 + (se_c/4)^2);  margin = e_cost - e*;
    stopping-slack bound (Addendum 61, report-only) = sum_i |w_i| band_i + |sum_i w_i| band_0, w = row b + row c / 4 of
    (X^T X)^-1 X^T, band_0 = the band of Q(0).
`d_fit` reads its point set from the module global `D_FIT_NODE7_LABELS`; a fit on a SUBSET (the certified-only fit)
rebinds that global for the duration of the call only (`_point_set`, restored in `finally`, asserted restored) -- the
code that fits is W132's line for line. p_cost = I_new_power(n7_4h_e1) / 0.25, e_cost = I_new_energy(n7_4h_e1) / 1.0
from the W2 table (as in W132).

INPUTS PER POINT: label (n7_{2h,4h}_e{1..5}; E, P from `L132._ep_of`), Q (gross_operational_cost, settlement EXCLUDED),
Q_cc (= Q + t_sum, report-only), I (W2 table I_new_eur, power / energy split; recorded, report-only check that it is
linear in the unit costs), status. A CERTIFIED point carries its band width. An UNCERTIFIED point carries Q at its cap
and its uncertified bar = 3 x max(|gap|, |slack|), computed by `L132.resolve` (the formula's single home) on the cell's
scorer view. Q(0) must be certified.

OUTPUT 1 -- CERTIFIED-ONLY FIT: `d_fit` on the certified points only (n_c >= 4 and rank 3, else 'undefined'): a, b, c,
SE, e*, se_e*, margin to cost, slack bound (bands of the certified points and of Q(0)).

OUTPUT 2 -- BANDED FIT WITH EVERY POINT: each uncertified point j enters as the INTERVAL
    Q_j in [Q_cap_j - h_j, Q_cap_j + h_j],   h_j = bar_j + 2 TAU,   2 TAU = SC6.DETERMINACY_TAU_MULTIPLE * SC6.TAU
    (= 9,078.14 EUR; Addendum 61's determinacy floor).
OLS coefficients are linear in the data, so over the box of intervals each of b, c, b + c/4, e*, margin attains its min
and max at a vertex: ALL 2^k vertices (k = number of uncertified points, k <= 12) are fitted with `d_fit` and the exact
min / max (and the vertex attaining each) reported. Cross-check: e*_max - e*_mid = sum_j |w_j| h_j (w from the midpoint
fit's own slack-bound weights). Also the fit at the interval MIDPOINTS (Q = Q_cap, i.e. every point at face value; Q_cc
beside, report-only) with its slack bound twice: certified bands only (uncertified bands 0), and extended (uncertified
'bands' = h_j, which adds exactly half the e* range).

OUTPUT 3 -- PREDICTION (expert, Addendum 62): "the two fits agree on the slope within 5 %":
    rel_mid = |b_banded_mid - b_certified| / |b_certified| <= 0.05  -> held, else failed.
    THE SLOPE is the fit's energy coefficient b (EUR/MWh) -- Worker operationalisation, for Planner confirmation; the
    same ratio for the marginal-4h slope b + c/4 (= e* + p_cost/4) is reported beside, report-only.
    Extremes reported: |b_min - b_cert| / |b_cert| and |b_max - b_cert| / |b_cert| (and whether 5 % holds over the whole
    interval, report-only). With no uncertified point the prediction is vacuous (rel_mid = 0) and is so labelled.

OUTPUT 4 -- CONCLUSION CHECK: the break-even stays below the energy cost under both fits across the whole interval:
    e*_certified < e_cost  AND  max over the box of e*_banded < e_cost.

Report-only beside: per point, the certified-only fit's prediction and whether it lies inside each uncertified point's
interval.

MODES (attached, alone, both streams captured; output files opened 'x', never overwritten):
  --self-test   synthetic / committed-old-value self-tests only -> data/SRP1/Results/P515S53/w145_banded_fit_selftest/
  --score       claim point #12 (after d_9246ed01): self-tests, then the fit on the committed v6 records ->
                data/SRP1/Results/P515S53/w145_banded_fit/
    mkdir -p data/SRP1/Results/P515S53/w145_banded_fit && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w145_banded_breakeven_fit.py --score \\
        > data/SRP1/Results/P515S53/w145_banded_fit/launch.log 2>&1
Inputs of --score (committed, sha256 recorded; the pins of stage spec v6 96c23404 checked as its summarizer checks them):
  Q(0) = settled x0 d110bd1a (Q181) and n7_4h_e1 = settled unit 3f084f2f (Q172), via `L132.reference_views`;
  n7_4h_e2 = d_c52e1670 v6 from records (ab289125), via `L142.decided_reports`;
  the other eight = the v6 cell runs `w142_resettle_v6/campaign_s53_w142_resettle_v6_<cell>/campaign_results.json`.
Exit: 0 = self-tests pass, guards 0, prediction held and conclusion holds; 3 = the 5 % prediction failed or the
conclusion check failed (results written); 1 = harness fault / self-test failure / precondition (nothing written).
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W145 banded break-even fit (never solves)').install()

# ---- the no-model-load guard: pickle.load / pickle.loads raise for the whole run (W141's pattern) ---------------------
PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W145: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W145: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w142_resettle_v6_campaign as L142  # noqa: E402 -- the v6 launcher (arms its zero-permit guards)

L132 = L142.L132
V5 = L142.V5


def pickle_state():
    """The no-model-load guard's state. W141's module (imported through L142) installs its own blocking counters at
    import, over these; both sets of counters are read, and pickle.load / loads must still be a blocker."""
    w141 = sys.modules.get('p515_s53_w141_swing_variants')
    counts = {'w145': dict(PICKLE_COUNTS)}
    if w141 is not None and hasattr(w141, 'PICKLE_COUNTS'):
        counts['w141_imported'] = dict(w141.PICKLE_COUNTS)
    blocked = pickle.load is not _PICKLE_ORIG[0] and pickle.loads is not _PICKLE_ORIG[1]
    ok = blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())
    return {'counts': counts, 'pickle_load_and_loads_blocked': blocked, 'ok': ok}


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w145_banded_fit', GUARD),) + tuple(L142.GUARDS))

TAU = SC6.TAU
TWO_TAU = SC6.DETERMINACY_TAU_MULTIPLE * TAU
SLOPE_AGREEMENT_BOUND = 0.05
MAX_UNCERTIFIED = 12
LABELS = tuple(L132.D_FIT_NODE7_LABELS)
Q0_REF, UNIT_REF, UNIT_LABEL = '7aa017f0', 'bd504ecf', 'n7_4h_e1'
STAGE_SPEC_SHA_PREFIX = '96c23404'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_SCORE = os.path.join(S53, 'w145_banded_fit')
OUT_DIR_SELFTEST = os.path.join(S53, 'w145_banded_fit_selftest')
OUT_JSON_SCORE = 'w145_banded_fit.json'
OUT_JSON_SELFTEST = 'w145_banded_fit_selftest.json'
OUT_MAN = 'manifest_sha256.json'
SCRIPT_REL = os.path.basename(__file__)
OBJECTIVE_CONVENTION = ('Q = gross_operational_cost, settlement EXCLUDED (EUR); value = Q(0) - Q(x); Q_cc = Q + t_sum, '
                        'report-only')
SLOPE_DEFINITION = ('the slope = the fit coefficient b (EUR/MWh) of value = a + b E + c P (Worker operationalisation, for '
                    'Planner confirmation); the marginal-4h slope b + c/4 is reported beside, report-only')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for blk in iter(lambda: handle.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ======================================================================================================================
#  the fit (W132's d_fit, called) on a point set
# ======================================================================================================================
@contextlib.contextmanager
def _point_set(labels):
    """Rebind `L132.D_FIT_NODE7_LABELS` (the global `d_fit` reads its point set from) for one call; restored always."""
    saved = L132.D_FIT_NODE7_LABELS
    L132.D_FIT_NODE7_LABELS = tuple(labels)
    try:
        yield
    finally:
        L132.D_FIT_NODE7_LABELS = saved
    if L132.D_FIT_NODE7_LABELS is not saved:
        raise RuntimeError('W145: L132.D_FIT_NODE7_LABELS not restored')


def fit_on(labels, y_q, y_qcc, bands, q0_band, p_cost, e_cost, ep):
    """`L132.d_fit` on `labels` (values y = Q(0) - Q(x)). Returns d_fit's dict, or None when the design is degenerate
    (n < 4 or rank(X) < 3: the SE needs n - 3 > 0 and (X^T X)^-1)."""
    labels = list(labels)
    X_ = np.array([[1.0, ep[lb][0], ep[lb][1]] for lb in labels])
    if len(labels) < 4 or np.linalg.matrix_rank(X_) < 3:
        return None
    with _point_set(labels):
        return L132.d_fit({'Q': {lb: y_q[lb] for lb in labels}, 'Q_cc': {lb: y_qcc[lb] for lb in labels}}, q0_band,
                          {lb: bands[lb] for lb in labels}, p_cost, e_cost, ep)


def _summary(fit, labels):
    q = fit['Q']
    return {'n': len(labels), 'labels': list(labels), 'a': q['a'], 'b_per_MWh': q['b_per_MWh'], 'c_per_MVA': q['c_per_MVA'],
            'marginal_4h_slope_b_plus_c_over_4': q['b_per_MWh'] + q['c_per_MVA'] / 4.0,
            'se_a_b_c': q['se'], 'breakeven_marginal_4h_energy_cost': q['breakeven_marginal_4h_energy_cost'],
            'se_breakeven_independent_terms_approx': q['se_breakeven_independent_terms_approx'],
            'margin_to_energy_cost_per_MWh': q['margin_to_energy_cost_per_MWh'],
            'residual_rms': q['residual_rms'], 'residual_max_abs': q['residual_max_abs'],
            'Q_cc_report_only': {k: fit['Q_cc'][k] for k in ('a', 'b_per_MWh', 'c_per_MVA',
                                                              'breakeven_marginal_4h_energy_cost',
                                                              'margin_to_energy_cost_per_MWh')},
            'slack_bound_report_only': fit['slack_bound_report_only']['value'],
            'slack_bound_weights': fit['slack_bound_report_only']['weights']}


def uncertified_bar(view):
    """3 x max(|gap|, |slack|) via `L132.resolve` (the uncertified-form rule's single home); None if undefined."""
    if view.get('status') == 'certified':
        raise ValueError('uncertified_bar called on a certified view')
    return L132.resolve(1.0, 1.0, (view,)).get('bar')


QUANTITIES = ('b_per_MWh', 'c_per_MVA', 'marginal_4h_slope_b_plus_c_over_4', 'breakeven_marginal_4h_energy_cost',
              'margin_to_energy_cost_per_MWh')


def banded_breakeven_fit(points, q0, p_cost, e_cost, two_tau=TWO_TAU):
    """THE SCORER. Pure (apart from calling W132's d_fit / resolve).
    points: [{'label', 'status', 'Q', 'Q_cc', 'band', 'gap', 'slack', ...}] (scorer views, `L132.view_from_report` shape);
    q0: the view of Q(0) (must be certified)."""
    if q0.get('status') != 'certified':
        raise ValueError('Q(0) is not certified: the scorer needs a certified Q(0)')
    labels = [p['label'] for p in points]
    if len(set(labels)) != len(labels):
        raise ValueError(f'duplicate labels {labels}')
    by = {p['label']: p for p in points}
    ep = {lb: L132._ep_of(lb) for lb in labels}
    cert = [lb for lb in labels if by[lb]['status'] == 'certified']
    unc = [lb for lb in labels if by[lb]['status'] != 'certified']
    if len(unc) > MAX_UNCERTIFIED:
        raise ValueError(f'{len(unc)} uncertified points > {MAX_UNCERTIFIED}: vertex enumeration refused')
    for lb in labels:
        if by[lb].get('Q') is None or by[lb].get('Q_cc') is None:
            raise ValueError(f'{lb}: no Q / Q_cc')
    intervals = {}
    for lb in unc:
        bar = uncertified_bar(by[lb])
        if bar is None:
            raise ValueError(f'{lb}: uncertified bar undefined (gap / slack missing: an ungated uncertified cell)')
        h = bar + two_tau
        intervals[lb] = {'Q_cap': by[lb]['Q'], 'gap': by[lb]['gap'], 'slack': by[lb]['slack'], 'bar': bar,
                         'two_tau': two_tau, 'half_width': h, 'Q_lo': by[lb]['Q'] - h, 'Q_hi': by[lb]['Q'] + h}
    y_q = {lb: q0['Q'] - by[lb]['Q'] for lb in labels}
    y_qcc = {lb: q0['Q_cc'] - by[lb]['Q_cc'] for lb in labels}
    bands_cert = {lb: (by[lb]['band'] if lb in cert else 0.0) for lb in labels}
    out = {'n_points': len(labels), 'certified': cert, 'uncertified': unc, 'intervals': intervals,
           'p_cost': p_cost, 'e_cost': e_cost, 'two_tau': two_tau, 'q0': {k: q0.get(k) for k in ('Q', 'Q_cc', 'band')}}

    # (1) the certified-only fit
    fc = fit_on(cert, y_q, y_qcc, bands_cert, q0['band'], p_cost, e_cost, ep)
    out['certified_only_fit'] = _summary(fc, cert) if fc else {'undefined': True, 'n': len(cert),
                                                                'reason': 'n < 4 or rank(X) < 3'}
    # (2) the banded fit: midpoints, then every vertex of the box
    fm = fit_on(labels, y_q, y_qcc, bands_cert, q0['band'], p_cost, e_cost, ep)
    if fm is None:
        raise ValueError('the all-points design is degenerate')
    mid = _summary(fm, labels)
    fext = fit_on(labels, y_q, y_qcc, {**bands_cert, **{lb: intervals[lb]['half_width'] for lb in unc}}, q0['band'],
                  p_cost, e_cost, ep)
    mid['slack_bound_certified_bands_only'] = mid.pop('slack_bound_report_only')
    mid['slack_bound_extended_with_interval_half_widths'] = fext['slack_bound_report_only']['value']
    ext = {qn: {'min': None, 'max': None, 'argmin_vertex': None, 'argmax_vertex': None} for qn in QUANTITIES}
    n_vertices = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(unc)):
        yv = dict(y_q)
        for lb, s in zip(unc, signs):
            yv[lb] = q0['Q'] - (intervals[lb]['Q_cap'] + s * intervals[lb]['half_width'])
        fv = _summary(fit_on(labels, yv, yv, bands_cert, q0['band'], p_cost, e_cost, ep), labels)
        n_vertices += 1
        vtx = {lb: ('Q_cap + h' if s > 0 else 'Q_cap - h') for lb, s in zip(unc, signs)}
        for qn in QUANTITIES:
            v = fv[qn]
            if ext[qn]['min'] is None or v < ext[qn]['min']:
                ext[qn]['min'], ext[qn]['argmin_vertex'] = v, vtx
            if ext[qn]['max'] is None or v > ext[qn]['max']:
                ext[qn]['max'], ext[qn]['argmax_vertex'] = v, vtx
    w = mid['slack_bound_weights']
    half_closed = float(sum(abs(w[lb]) * intervals[lb]['half_width'] for lb in unc))
    e_mid = mid['breakeven_marginal_4h_energy_cost']
    out['banded_fit'] = {
        'midpoint_fit': mid, 'n_vertices_enumerated': n_vertices, 'range_over_the_box': ext,
        'breakeven_range': [ext['breakeven_marginal_4h_energy_cost']['min'], ext['breakeven_marginal_4h_energy_cost']['max']],
        'margin_to_cost_range': [ext['margin_to_energy_cost_per_MWh']['min'], ext['margin_to_energy_cost_per_MWh']['max']],
        'slope_range_b': [ext['b_per_MWh']['min'], ext['b_per_MWh']['max']],
        'closed_form_crosscheck': {
            'sum_abs_w_h': half_closed, 'enum_max_minus_mid': ext['breakeven_marginal_4h_energy_cost']['max'] - e_mid,
            'mid_minus_enum_min': e_mid - ext['breakeven_marginal_4h_energy_cost']['min'],
            'max_abs_diff': max(abs(ext['breakeven_marginal_4h_energy_cost']['max'] - e_mid - half_closed),
                                abs(e_mid - ext['breakeven_marginal_4h_energy_cost']['min'] - half_closed))}}

    # (3) the prediction: slope agreement within 5 %
    if fc is None:
        out['prediction_slope_within_5pct'] = {'held': None, 'scorable': False,
                                               'reason': 'the certified-only fit is undefined'}
    else:
        bc = out['certified_only_fit']['b_per_MWh']
        sc = out['certified_only_fit']['marginal_4h_slope_b_plus_c_over_4']
        rel_mid = abs(mid['b_per_MWh'] - bc) / abs(bc)
        rel_lo = abs(ext['b_per_MWh']['min'] - bc) / abs(bc)
        rel_hi = abs(ext['b_per_MWh']['max'] - bc) / abs(bc)
        m4 = ext['marginal_4h_slope_b_plus_c_over_4']
        out['prediction_slope_within_5pct'] = {
            'statement': 'expert, Addendum 62: the certified-only and banded fits agree on the slope within 5 %',
            'slope_definition': SLOPE_DEFINITION, 'bound': SLOPE_AGREEMENT_BOUND, 'scorable': True,
            'vacuous_no_uncertified_point': not unc,
            'b_certified': bc, 'b_banded_mid': mid['b_per_MWh'], 'rel_mid': rel_mid,
            'held': bool(rel_mid <= SLOPE_AGREEMENT_BOUND),
            'b_banded_min': ext['b_per_MWh']['min'], 'b_banded_max': ext['b_per_MWh']['max'],
            'rel_at_b_min': rel_lo, 'rel_at_b_max': rel_hi,
            'holds_over_the_whole_interval_report_only': bool(max(rel_lo, rel_hi) <= SLOPE_AGREEMENT_BOUND),
            'marginal_4h_slope_report_only': {
                'certified': sc, 'banded_mid': mid['marginal_4h_slope_b_plus_c_over_4'],
                'rel_mid': abs(mid['marginal_4h_slope_b_plus_c_over_4'] - sc) / abs(sc),
                'rel_at_min': abs(m4['min'] - sc) / abs(sc), 'rel_at_max': abs(m4['max'] - sc) / abs(sc)}}

    # (4) the conclusion check
    e_cert = out['certified_only_fit'].get('breakeven_marginal_4h_energy_cost')
    e_band_max = ext['breakeven_marginal_4h_energy_cost']['max']
    cert_ok = (e_cert < e_cost) if e_cert is not None else None
    out['conclusion_breakeven_below_energy_cost'] = {
        'e_cost': e_cost, 'e_star_certified_only': e_cert, 'e_star_banded_max_over_box': e_band_max,
        'e_star_banded_mid': e_mid, 'holds_certified_only': cert_ok, 'holds_banded_whole_interval': bool(e_band_max < e_cost),
        'holds': (bool(cert_ok and e_band_max < e_cost) if cert_ok is not None else None),
        'min_margin_certified_only': (e_cost - e_cert) if e_cert is not None else None,
        'min_margin_banded': e_cost - e_band_max}

    # report-only: the certified-only fit's prediction at every point
    if fc is not None:
        a, b, c = (out['certified_only_fit'][k] for k in ('a', 'b_per_MWh', 'c_per_MVA'))
        per = {}
        for lb in labels:
            yhat = a + b * ep[lb][0] + c * ep[lb][1]
            q_hat = q0['Q'] - yhat
            rec = {'E': ep[lb][0], 'P': ep[lb][1], 'status': by[lb]['status'], 'value': y_q[lb], 'value_hat': yhat,
                   'residual': y_q[lb] - yhat}
            if lb in intervals:
                rec['Q_hat_inside_interval'] = bool(intervals[lb]['Q_lo'] <= q_hat <= intervals[lb]['Q_hi'])
                rec['Q_cap_minus_Q_hat_over_half_width'] = (by[lb]['Q'] - q_hat) / intervals[lb]['half_width']
            per[lb] = rec
        out['per_point_vs_certified_only_fit_report_only'] = per
    return out


# ======================================================================================================================
#  self-tests (zero solves; committed old values + synthetic)
# ======================================================================================================================
def _costs():
    unit = L132._load(L132.W2_TABLE)['candidates'][UNIT_LABEL]
    return unit['I_new_power_eur'] / 0.25, unit['I_new_energy_eur'] / 1.0


def _phase_a_points():
    """The committed Phase A values (W117's cells, as `L132.scorer_self_tests` (2) builds them), all certified, band 0."""
    w117, _ = L132.K._pinned_json(L132.K.W117)
    cells = w117['cells']
    by_label = {c['label']: c for c in cells.values() if 'campaign_s47_a1a_baseline' in c['eval_dir']}
    x0 = cells[Q0_REF]
    pts = [{'label': lb, 'status': 'certified', 'Q': by_label[lb]['Q'], 'Q_cc': by_label[lb]['Q_cc'], 'band': 0.0,
            'gap': None, 'slack': None} for lb in LABELS]
    return pts, {'status': 'certified', 'Q': x0['Q'], 'Q_cc': x0['Q_cc'], 'band': 0.0}, w117


def _synthetic(a=10000.0, b=233000.0, c=52700.0, q0=650_000_000.0):
    """Exactly affine values: value_i = a + b E + c P; all certified, band 1,000 (+ 0 for Q(0))."""
    pts = []
    for lb in LABELS:
        e, p = L132._ep_of(lb)
        q = q0 - (a + b * e + c * p)
        pts.append({'label': lb, 'status': 'certified', 'Q': q, 'Q_cc': q + 500.0, 'band': 1000.0, 'gap': None,
                    'slack': None})
    return pts, {'status': 'certified', 'Q': q0, 'Q_cc': q0 + 500.0, 'band': 0.0}


def _uncertify(pts, label, gap, slack, shift=0.0):
    out = []
    for p in pts:
        p = dict(p)
        if p['label'] == label:
            p.update({'status': 'uncertified', 'gap': gap, 'slack': slack, 'Q': p['Q'] + shift,
                      'Q_cc': p['Q_cc'] + shift})
        out.append(p)
    return out


def self_tests():
    res = {}
    p_cost, e_cost = _costs()
    bt = L132._load(L132.BASELINE_TABLES)
    n7, be = bt['node7_fit'], bt['breakeven']

    # S0: the bar formula and the 2 TAU constant
    v1 = {'status': 'uncertified', 'gap': 1082.2324382210093, 'slack': 7973.221109390259}
    v2 = {'status': 'uncertified', 'gap': 7973.221109390259, 'slack': 1082.2324382210093}
    bar1, bar2 = uncertified_bar(v1), uncertified_bar(v2)
    undefined_none = uncertified_bar({'status': 'uncertified', 'gap': 1.0, 'slack': None}) is None
    res['S0_bar_and_two_tau'] = {
        'ok': bool(bar1 == bar2 == 3.0 * 7973.221109390259 and undefined_none and abs(TWO_TAU - 9078.14) < 0.005),
        'bar': bar1, 'two_tau': TWO_TAU, 'undefined_slack_gives_None': undefined_none}

    # S1: every point certified with its old values -> the committed Phase A fit within 1e-6 (certified-only AND banded),
    #     and bitwise the same numbers as L132.d_fit called directly on its own (unrebound) label set
    pts, q0, w117 = _phase_a_points()
    r = banded_breakeven_fit(pts, q0, p_cost, e_cost)
    cf, mf = r['certified_only_fit'], r['banded_fit']['midpoint_fit']
    direct = L132.d_fit({'Q': {lb: q0['Q'] - p['Q'] for lb, p in zip(LABELS, pts)},
                         'Q_cc': {lb: q0['Q_cc'] - p['Q_cc'] for lb, p in zip(LABELS, pts)}},
                        0.0, {lb: 0.0 for lb in LABELS}, p_cost, e_cost, {lb: L132._ep_of(lb) for lb in LABELS})
    diffs = {'a': cf['a'] - n7['a'], 'b': cf['b_per_MWh'] - n7['b'], 'c': cf['c_per_MVA'] - n7['c'],
             'se_a': cf['se_a_b_c'][0] - n7['se']['a'], 'se_b': cf['se_a_b_c'][1] - n7['se']['b'],
             'se_c': cf['se_a_b_c'][2] - n7['se']['c'],
             'breakeven': cf['breakeven_marginal_4h_energy_cost'] - be['marginal_4h_eur_per_mwh'],
             'se_breakeven': cf['se_breakeven_independent_terms_approx'] - be['marginal_4h_se_eur_per_mwh'],
             'residual_rms': cf['residual_rms'] - n7['residual_rms'],
             'residual_max_abs': cf['residual_max_abs'] - n7['residual_max_abs']}
    bitwise_direct = all(direct['Q'][k] == r_k for k, r_k in (
        ('a', cf['a']), ('b_per_MWh', cf['b_per_MWh']), ('c_per_MVA', cf['c_per_MVA']),
        ('breakeven_marginal_4h_energy_cost', cf['breakeven_marginal_4h_energy_cost'])))
    same_mid = all(cf[k] == mf[k] for k in ('a', 'b_per_MWh', 'c_per_MVA', 'breakeven_marginal_4h_energy_cost'))
    rng = r['banded_fit']['breakeven_range']
    res['S1_reproduces_committed_phase_a_fit'] = {
        'ok': bool(all(abs(v) < 1e-6 for v in diffs.values()) and bitwise_direct and same_mid and rng[0] == rng[1]
                   and r['banded_fit']['n_vertices_enumerated'] == 1
                   and r['prediction_slope_within_5pct']['vacuous_no_uncertified_point']
                   and r['prediction_slope_within_5pct']['rel_mid'] == 0.0
                   and L132.D_FIT_NODE7_LABELS == LABELS),
        'diffs_vs_committed': diffs, 'bitwise_equal_to_direct_d_fit_call': bitwise_direct,
        'certified_only_equals_midpoint_bitwise': same_mid, 'breakeven': cf['breakeven_marginal_4h_energy_cost'],
        'se_breakeven': cf['se_breakeven_independent_terms_approx'], 'margin_to_cost': cf['margin_to_energy_cost_per_MWh'],
        'committed_breakeven': be['marginal_4h_eur_per_mwh'], 'committed_se': be['marginal_4h_se_eur_per_mwh'],
        'w117_sha256': L132.K.W117['sha256']}

    # S2: an interval point gives slope / break-even bounds that bracket the midpoint fit; closed form = enumeration;
    #     certified-only fit has n = 9; two interval points -> 4 vertices
    p1 = _uncertify(pts, 'n7_4h_e5', 1082.2324382210093, 7973.221109390259)
    r1 = banded_breakeven_fit(p1, q0, p_cost, e_cost)
    p2 = _uncertify(p1, 'n7_2h_e3', 3000.0, 500.0)
    r2 = banded_breakeven_fit(p2, q0, p_cost, e_cost)

    def brackets(rr):
        ext, mid = rr['banded_fit']['range_over_the_box'], rr['banded_fit']['midpoint_fit']
        return all(ext[q]['min'] < mid[q] < ext[q]['max'] for q in QUANTITIES)
    s2 = {'one_point': {'brackets_strictly': brackets(r1), 'n_vertices': r1['banded_fit']['n_vertices_enumerated'],
                        'certified_only_n': r1['certified_only_fit']['n'],
                        'half_width': r1['intervals']['n7_4h_e5']['half_width'],
                        'b_range': r1['banded_fit']['slope_range_b'],
                        'b_mid': r1['banded_fit']['midpoint_fit']['b_per_MWh'],
                        'breakeven_range': r1['banded_fit']['breakeven_range'],
                        'closed_form_max_abs_diff': r1['banded_fit']['closed_form_crosscheck']['max_abs_diff']},
          'two_points': {'brackets_strictly': brackets(r2), 'n_vertices': r2['banded_fit']['n_vertices_enumerated'],
                         'certified_only_n': r2['certified_only_fit']['n'],
                         'b_range': r2['banded_fit']['slope_range_b'],
                         'closed_form_max_abs_diff': r2['banded_fit']['closed_form_crosscheck']['max_abs_diff']}}
    s2['ok'] = bool(s2['one_point']['brackets_strictly'] and s2['two_points']['brackets_strictly']
                    and s2['one_point']['n_vertices'] == 2 and s2['two_points']['n_vertices'] == 4
                    and s2['one_point']['certified_only_n'] == 9 and s2['two_points']['certified_only_n'] == 8
                    and abs(s2['one_point']['half_width'] - (3.0 * 7973.221109390259 + TWO_TAU)) < 1e-9
                    and s2['one_point']['closed_form_max_abs_diff'] < 1e-6
                    and s2['two_points']['closed_form_max_abs_diff'] < 1e-6)
    res['S2_interval_brackets_midpoint'] = s2

    # S3: the 5 % check on a planted case. Exactly affine certified data (certified-only fit recovers b exactly); the
    #     uncertified point n7_4h_e5 is displaced by Delta so the midpoint slope moves by exactly 4 % (held) / 6 % (failed)
    sp, sq0 = _synthetic()
    zero = {lb: 0.0 for lb in LABELS}
    unit_y = dict(zero, n7_4h_e5=1.0)
    ep = {lb: L132._ep_of(lb) for lb in LABELS}
    v_j = fit_on(LABELS, unit_y, unit_y, zero, 0.0, p_cost, e_cost, ep)['Q']['b_per_MWh']   # db / dy_j
    s3 = {'v_j_db_dy': v_j}
    for target, expect in ((0.04, True), (0.06, False), (-0.04, True), (-0.06, False)):
        dy = target * 233000.0 / v_j          # value shift; Q shift = -dy
        rr = banded_breakeven_fit(_uncertify(sp, 'n7_4h_e5', 50.0, 100.0, shift=-dy), sq0, p_cost, e_cost)
        pr = rr['prediction_slope_within_5pct']
        s3[f'planted_{target:+.2f}'] = {'rel_mid': pr['rel_mid'], 'held': pr['held'], 'expected_held': expect,
                                        'b_certified': pr['b_certified'], 'b_banded_mid': pr['b_banded_mid'],
                                        'rel_at_b_min': pr['rel_at_b_min'], 'rel_at_b_max': pr['rel_at_b_max']}
    s3['ok'] = bool(all(s3[k]['held'] is s3[k]['expected_held'] and abs(s3[k]['rel_mid'] - abs(float(k.split('_')[1])))
                        < 1e-9 and abs(s3[k]['b_certified'] - 233000.0) < 1e-6
                        for k in s3 if k.startswith('planted_')))
    res['S3_five_percent_check_planted'] = s3

    # S4: the conclusion check on a planted case: a small interval holds; an interval wide enough to push e*_max over
    #     e_cost fails under the banded fit while the certified-only fit still holds
    w_j = fit_on(LABELS, unit_y, unit_y, zero, 0.0, p_cost, e_cost, ep)['slack_bound_report_only']['weights']['n7_4h_e5']
    e_star_true = 233000.0 + 52700.0 / 4.0 - p_cost / 4.0
    need_h = 1.5 * (e_cost - e_star_true) / abs(w_j)
    small = banded_breakeven_fit(_uncertify(sp, 'n7_4h_e5', 50.0, 100.0), sq0, p_cost, e_cost)
    big = banded_breakeven_fit(_uncertify(sp, 'n7_4h_e5', (need_h - TWO_TAU) / 3.0, 1.0), sq0, p_cost, e_cost)
    cs, cb = small['conclusion_breakeven_below_energy_cost'], big['conclusion_breakeven_below_energy_cost']
    res['S4_conclusion_check_planted'] = {
        'ok': bool(cs['holds'] is True and cb['holds'] is False and cb['holds_certified_only'] is True
                   and cb['holds_banded_whole_interval'] is False and abs(cs['e_star_certified_only'] - e_star_true) < 1e-6),
        'small': cs, 'big': cb, 'e_star_true': e_star_true, 'w_j': w_j, 'planted_half_width': need_h}

    # S5: degenerate certified-only design -> 'undefined', prediction unscorable (all 2h points uncertified: P = E/4 only)
    pd = sp
    for lb in LABELS:
        if '_2h_' in lb:
            pd = _uncertify(pd, lb, 10.0, 10.0)
    rd = banded_breakeven_fit(pd, sq0, p_cost, e_cost)
    res['S5_degenerate_certified_design'] = {
        'ok': bool(rd['certified_only_fit'].get('undefined') is True
                   and rd['prediction_slope_within_5pct']['scorable'] is False
                   and rd['conclusion_breakeven_below_energy_cost']['holds'] is None
                   and rd['banded_fit']['n_vertices_enumerated'] == 32),
        'certified_only_fit': rd['certified_only_fit'], 'n_vertices': rd['banded_fit']['n_vertices_enumerated']}

    # S6: the point-set rebinding is restored, also on an exception inside the call
    try:
        with _point_set(LABELS[:4]):
            raise KeyError('planted')
    except KeyError:
        pass
    res['S6_point_set_restored'] = {'ok': L132.D_FIT_NODE7_LABELS == LABELS}
    res['all_ok'] = all(v['ok'] for k, v in res.items() if k.startswith('S'))
    return res, {L132.K.W117['path']: L132.K.W117['sha256'], L132.BASELINE_TABLES: _sha(L132.BASELINE_TABLES),
                 L132.W2_TABLE: _sha(L132.W2_TABLE)}


# ======================================================================================================================
#  claim point #12: the points from committed records
# ======================================================================================================================
def committed_points():
    """The ten D-fit points and Q(0) as scorer views, from committed records; preconditions raise."""
    ss_rel, ss_sha, ss = L142.load_stage_spec()
    if not ss_sha.startswith(STAGE_SPEC_SHA_PREFIX):
        raise RuntimeError(f'stage spec {ss_rel} sha {ss_sha} is not {STAGE_SPEC_SHA_PREFIX}')
    for key in ('baseline_tables', 'w2_table'):
        pin = ss['pins'][key]
        if _sha(pin['path']) != pin['sha256'] or not L142._committed_clean(pin['path']):
            raise RuntimeError(f'{pin["path"]} differs from its stage-spec pin or is not committed clean')
    decided, decided_inputs = L142.decided_reports()
    if decided_inputs != ss['pins']['decided_inputs_sha256']:
        raise RuntimeError("the decided cells' records changed since the stage spec froze")
    refs, ref_inputs = L132.reference_views()
    if ref_inputs != ss['pins']['references_inputs_sha256']:
        raise RuntimeError('the references changed since the stage spec froze')
    d_cells = {V5.CELLS[c]['orig_label']: c for c in V5.CELL_ORDER if V5.CELLS[c]['item'] == 'D'}
    if set(d_cells) | {UNIT_LABEL} != set(LABELS) or UNIT_LABEL in d_cells:
        raise RuntimeError(f'D cell labels {sorted(d_cells)} + {UNIT_LABEL} != {LABELS}')
    w2 = L132._load(L132.W2_TABLE)['candidates']
    inputs = {**ref_inputs, **{k: v for k, v in decided_inputs.items()}}
    inputs[ss_rel] = ss_sha
    inputs[L132.W2_TABLE] = _sha(L132.W2_TABLE)
    inputs[L132.BASELINE_TABLES] = _sha(L132.BASELINE_TABLES)
    points, missing, provenance = [], [], {}
    for lb in LABELS:
        if lb == UNIT_LABEL:
            view, src = dict(refs[UNIT_REF]), {'source': f'reference {UNIT_REF}: {refs[UNIT_REF]["name"]}',
                                               'k_star': refs[UNIT_REF].get('k_star')}
        else:
            cell = d_cells[lb]
            if cell in decided:
                rep = decided[cell]
                src = {'source': f'{cell}: decided before v6 ({L142.DECIDED[cell]["kind"]})',
                       'record': L142.DECIDED[cell].get('certificate_record')}
            else:
                rel = os.path.join(L142.campaign_root_rel(cell), L142.RESULTS_FILE)
                if not os.path.isfile(os.path.join(REPO, rel)) or not L142._committed_clean(rel):
                    missing.append(cell)
                    continue
                inputs[rel] = _sha(rel)
                rep = L132._load(rel).get('cell_report') or {}
                src = {'source': f'{cell}: v6 cell run', 'record': rel}
            view = dict(rep.get('view') or L132.view_from_report(rep))
            src.update({k: rep.get(k) for k in ('cell', 'eval_key', 'candidate_key', 'k_star', 'k_cap', 'cycles_run',
                                                'band_width', 'certifying_spec', 'label')})
        view['label'] = lb
        e, p = L132._ep_of(lb)
        cand = w2[lb]
        src['I'] = {'I_new_eur': cand['I_new_eur'], 'I_new_power_eur': cand['I_new_power_eur'],
                    'I_new_energy_eur': cand['I_new_energy_eur'], 'w2_candidate_key': cand['candidate_key'],
                    'canonical_node7_P_E': cand['candidate_canonical']['nodes']['7']}
        provenance[lb] = src
        points.append(view)
    if missing:
        raise RuntimeError(f'D cells without committed results: {missing}')
    q0 = refs[Q0_REF]
    return points, q0, provenance, inputs, {'path': ss_rel, 'sha256': ss_sha}


def i_linearity(provenance, p_cost, e_cost):
    """Report-only: I_new_power = p_cost P and I_new_energy = e_cost E for every point (the unit costs the break-even
    uses), with the canonical (P, E) equal to `_ep_of`."""
    out = {}
    for lb, src in provenance.items():
        e, p = L132._ep_of(lb)
        i = src['I']
        out[lb] = {'E': e, 'P': p, 'canonical_P_E': i['canonical_node7_P_E'],
                   'canonical_equals_ep': [float(x) for x in i['canonical_node7_P_E']] == [p, e],
                   'power_minus_p_cost_P': i['I_new_power_eur'] - p_cost * p,
                   'energy_minus_e_cost_E': i['I_new_energy_eur'] - e_cost * e}
    return out


# ======================================================================================================================
def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _write(out_dir_rel, json_name, doc, extra_inputs):
    out_dir = os.path.join(REPO, out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, json_name)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    man = {os.path.relpath(jp, REPO): _sha(os.path.relpath(jp, REPO)), **extra_inputs}
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    return os.path.relpath(jp, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--score', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    tag = 'W145-SCORE' if args.score else 'W145-SELFTEST'
    out_dir_rel = OUT_DIR_SCORE if args.score else OUT_DIR_SELFTEST
    json_name = OUT_JSON_SCORE if args.score else OUT_JSON_SELFTEST
    if os.path.exists(os.path.join(REPO, out_dir_rel, json_name)):
        _log(f'[{tag} PRECONDITION FAILED] {out_dir_rel}/{json_name} exists (write-once)')
        sys.exit(1)
    script_clean = L142._committed_clean(SCRIPT_REL)
    script_commit = _git('log', '-1', '--format=%H', '--', SCRIPT_REL) or None
    if args.score and not script_clean:
        _log(f'[{tag} PRECONDITION FAILED] {SCRIPT_REL} is not committed clean: the scorer must be pre-registered')
        sys.exit(1)
    _log(f'[{tag}] script sha256 {_sha(SCRIPT_REL)} committed clean {script_clean} last commit {script_commit}')
    st, st_inputs = self_tests()
    for k, v in st.items():
        if k.startswith('S'):
            _log(f'[{tag}] self-test {k}: {"PASS" if v["ok"] else "FAIL"}')
    p_cost, e_cost = _costs()
    doc = {'schema': 'p515_s53_w145_banded_breakeven_fit_v1', 'stage': tag, 'utc': _utc(),
           'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 62 item 3 (determinacy floor: Addendum 61 ruling 2)',
           'definition': __doc__, 'objective_convention': OBJECTIVE_CONVENTION, 'slope_definition': SLOPE_DEFINITION,
           'git_head': _git('rev-parse', 'HEAD'), 'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL),
                                                            'committed_clean': script_clean,
                                                            'last_commit': script_commit},
           'code_sha256': {rel: _sha(rel) for rel in ('p515_s53_w132_resettle_v3_campaign.py',
                                                      'p515_s53_w142_resettle_v6_campaign.py', 'settling_criterion_v6.py',
                                                      'gate_result_io.py')},
           'constants': {'TAU': TAU, 'TWO_TAU': TWO_TAU, 'SLOPE_AGREEMENT_BOUND': SLOPE_AGREEMENT_BOUND,
                         'p_cost_eur_per_MVA': p_cost, 'e_cost_eur_per_MWh': e_cost, 'labels': list(LABELS)},
           'self_tests': st}
    inputs = dict(st_inputs)
    code = 0 if st['all_ok'] else 1
    if args.score and st['all_ok']:
        try:
            points, q0, prov, p_inputs, ss = committed_points()
        except Exception as exc:  # noqa: BLE001
            _log(f'[{tag} PRECONDITION FAILED] {type(exc).__name__}: {exc}')
            sys.exit(1)
        inputs.update(p_inputs)
        res = banded_breakeven_fit(points, q0, p_cost, e_cost)
        doc.update({'stage_spec': ss, 'q0_reference': {'ref': Q0_REF, 'name': q0.get('name'), 'k_star': q0.get('k_star'),
                                                       'Q': q0['Q'], 'Q_cc': q0['Q_cc'], 'band': q0['band'],
                                                       'status': q0['status']},
                    'points': {p['label']: {k: p.get(k) for k in ('status', 'Q', 'Q_cc', 't', 'band', 'gap', 'slack')}
                               for p in points},
                    'provenance': prov, 'I_linearity_report_only': i_linearity(prov, p_cost, e_cost),
                    'result': res, 'inputs_sha256': inputs})
        pr, cc = res['prediction_slope_within_5pct'], res['conclusion_breakeven_below_energy_cost']
        cf, bf = res['certified_only_fit'], res['banded_fit']
        _log(f"[{tag}] certified {res['certified']} uncertified {res['uncertified']}")
        for lb, iv in res['intervals'].items():
            _log(f"[{tag}] interval {lb}: Q_cap {iv['Q_cap']:.2f} bar {iv['bar']:.2f} + 2tau {iv['two_tau']:.2f} = "
                 f"h {iv['half_width']:.2f}")
        if not cf.get('undefined'):
            _log(f"[{tag}] certified-only (n {cf['n']}): b {cf['b_per_MWh']:.2f} c {cf['c_per_MVA']:.2f} e* "
                 f"{cf['breakeven_marginal_4h_energy_cost']:.2f} +- {cf['se_breakeven_independent_terms_approx']:.2f} "
                 f"margin {cf['margin_to_energy_cost_per_MWh']:.2f} slack bound {cf['slack_bound_report_only']:.2f}")
        mf = bf['midpoint_fit']
        _log(f"[{tag}] banded mid (n {mf['n']}): b {mf['b_per_MWh']:.2f} e* {mf['breakeven_marginal_4h_energy_cost']:.2f}; "
             f"b range {bf['slope_range_b']}; e* range {bf['breakeven_range']}; margin range {bf['margin_to_cost_range']}")
        _log(f"[{tag}] PREDICTION slope within 5 %: held {pr.get('held')} rel_mid {pr.get('rel_mid')} extremes "
             f"{pr.get('rel_at_b_min')} / {pr.get('rel_at_b_max')}")
        _log(f"[{tag}] CONCLUSION e* < e_cost: {cc['holds']} (certified-only {cc['holds_certified_only']}, banded whole "
             f"interval {cc['holds_banded_whole_interval']}; min margin banded {cc['min_margin_banded']:.2f})")
        if pr.get('held') is not True or cc['holds'] is not True:
            code = 3
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    doc['guards'] = guards
    pk = pickle_state()
    doc['pickle_guard'] = pk
    doc['wall_s'] = time.time() - t0
    guards_ok = all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']
    if not guards_ok:
        code = 1
    doc['exit_code'] = code
    if not args.score or st['all_ok']:
        rel = _write(out_dir_rel, json_name, doc, inputs)
        _log(f'[{tag}] wrote {rel}')
    _log(f"[{tag}] self-tests all_ok {st['all_ok']}; guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle {pk}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
