"""P5.15 Addendum 45 item 3 close-out, task W82 -- the barrier-gap question closed on three pairs. READ-ONLY, ZERO SOLVES:
nothing is built, solved or re-run; no committed artifact is modified or re-written. SolveProfileGuard(permitted=()) is armed at
import (before W81's module is imported, which arms its own identical guard); both are verified at exactly 0 on every exit
path (try/finally) and uninstalled in reverse order.

ITEMS (Planner task W82):
  1. The COMPLETE 2x2 figure: W74's committed 2x2 gap difference (15,078.91) is TSO-only. Extend it to the 60 DSO blocks per
     cell of x0_a0p50 and n7_4h_e1_a0p50 (terminal ADMM solves) and report Delta G_total / the two-cell bar-sum 16,551.50.
  2. The mu-floor test (W81 finding 3): per 2x2 terminal solve, at the floor or above it and by how much; the gap difference
     recomputed under the counterfactual that every terminal solve reached its floor; the frozen three-way decomposition
     Delta G = P (structural pair count) + S (scaling regime) + C (above-floor excess = incomplete convergence).
  3. The SRP1 C2-baseline pair that governs R: value 259,427.76775527 = Q(0) - Q(unit), Q(0) pinned from the Phase A x0
     record (campaign_s45_a0_c7 7aa017f09989b56d_x0 -- the SAME x0 run W81 used), Q(unit) from campaign_s47_recert
     bd504ecf5a288d44_n7_4h_e1; against its own bars (x0_bar + unit bar = bar_sum_with_x0 34,734.63).
  Phase A (W81) is re-derived from its re-verified logs as the third pair, with the DSO floor reclassified (W81 recorded
  at_floor null on every DSO block because compl_inf_tol is not in the DSO options list; IPOPT then uses its default 1e-4).

CELLS / PROBLEM INSTANCES (candidate_key recorded per cell in the output):
  2x2  x0_a0p50        alpha_row v25 evals/7d53b6f21b686a44_x0_a0p50         (s52_pilot_2x2 instance, C2 baseline, alpha 0.5)
       n7_4h_e1_a0p50  alpha_row v25 evals/711fce9aa74d6878_n7_4h_e1_a0p50
  SRP1 Phase A   x0 a0_c7:x0, unit a0_c7:n7_p0.25_e1.0, unit_dup a1a:n7_4h_e1 (W81 cells, C3 configuration, spec v15)
  SRP1 C2 base   x0 = Phase A x0 (pinned Q(0)); unit = s47_recert:n7_4h_e1 (C2 + phi_cal 0.985 + soh_min 0.70)

FORMULAS (preserved here and in the output's `formulas`):
  n_pairs, mu_last, s, terminal ADMM solve (solve K + 1), g_b = w_b n_pairs mu_last / s: identical to W73/W74/W81 (the
  parser and identification rule are W81's, imported: p515_s53_w81_srp1_gap_estimate.parse_log / identify).
  mu_floor_b = min(tol, cit x s) / (btf + 1)  -- IPOPT 3.14.18 IpMonotoneMuUpdate: new_mu = Max(new_mu, mu_target,
               Min(tol, apply_obj_scaling(compl_inf_tol)) / (barrier_tol_factor + 1)). tol and cit are read from the options
               list printed before the run; cit = 1e-4 (IPOPT default) when absent from it; btf = 10 and mu_target = 0 (IPOPT
               defaults) -- the harness ASSERTS that barrier_tol_factor, mu_target, mu_min and mu_strategy are absent from the
               list of every terminal solve, so the defaults (monotone mu) are in force.
  at_floor iff |mu_last / mu_floor - 1| <= 1e-3; above iff mu_last / mu_floor > 1 + 1e-3; below iff < 1 - 1e-3 (flagged).
  g_floor_b  = w_b n_pairs mu_floor_b / s   (the at-floor counterfactual; = w_b n_pairs cit / 11 when cit s <= tol: s cancels)
  excess_b   = g_b - g_floor_b              (the above-floor excess; 0 at the floor)
  Delta X    = X(x0) - X(unit) summed over a group of blocks (TSO, DSO, DSO5/7/9, total)
  P          = Delta of sum_b w_b n_pairs cit_b / (btf + 1)       (structural pair count; the at-floor gap if every block were in
                                                                   the s-independent regime)
  S          = Delta G_floor - P                                   (non-zero only for blocks with cit s > tol: scaling regime)
  C          = Delta G - Delta G_floor = Delta excess              (incomplete convergence: stopping above the floor)
  Delta G    = P + S + C exactly (identity, asserted).
  R          = |Delta G_total| / bar_sum;  R_floor = |Delta G_floor_total| / bar_sum.
  share_X    = X / Delta G_total (signed).  Frozen rule (predictions_w82.json): the supported explanation is the component whose
               signed share is >= 0.5 (C -> incomplete convergence, S -> scaling artifact, P -> structural pair count); else
               'mixed, none dominant'.
  bar_sum    2x2: W72 resolution_two_cell = bar(x0_a0p50) + bar(n7_4h_e1_a0p50) (alpha_row_recompute.json);
             C2 baseline: s47_recert campaign_results points.n7_4h_e1.bar_sum_with_x0 (= x0_bar + unit bar, asserted);
             Phase A: W81 bar_sum (phase_a_tables T1).
  w_b        2x2: multiscenario_terminal.json admm_block_weight (W74's weight), cross-checked against component_levels_terminal,
             the settlement ratio and production shared_resources_planning._get_admm_block_weight on the s52_pilot_2x2 case
             JSON; SRP1: component_levels_terminal admm_block_weight, cross-checked as W81 on SRP1.json.
DERIVATION of the floor reduction: at the floor with cit s <= tol, mu = cit s / 11, so n mu / s = n cit / 11: the objective
  scaling factor cancels, and two cells whose solves all end at their floors differ in g only through n_pairs (and w_b, which
  is equal block-for-block). A solve that stops ABOVE its floor carries mu / s with mu drawn from the s-independent monotone
  sequence (mu <- min(0.2 mu, mu^1.5)), so its g is inflated by mu_last / mu_floor over the at-floor value.
LIMITATIONS (also in the output): every W81 limitation stands (order-of-magnitude estimate, nonconvex AC-OPF, n mu is a scale
  not a bound, central-path departure, whole-subproblem objective, per-block, ESSO outside Q). In addition: (a) the
  decomposition is an ACCOUNTING identity on the estimate; it does not decide whether a solve's stopping above its floor is
  itself caused by its scaling factor -- that needs solves and none is made; (b) the counterfactual assumes the pair count
  and weight would be unchanged at the floor; (c) the 2x2 subproblem objective at alpha 0.5 carries the row-18 premium, so
  the units ratio there is reported, not gated; (d) the C2-baseline x0 is a C3-era run pinned by the S2 Q(0) statement (x = 0
  has no storage; W21 case-file gate) -- this harness does not re-examine that pin.

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w82_gap_closeout.py --run \
      > data/SRP1/Results/P515S53/gap_closeout_w82_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w82_gap_closeout.py --manifest \
      > data/SRP1/Results/P515S53/gap_closeout_w82_manifest_launch.log 2>&1
"""

import glob
import json
import os
import re
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W82 barrier-gap close-out (read-only)').install()

import p515_s53_w81_srp1_gap_estimate as w81  # noqa: E402  (arms w81.GUARD, permitted=(); verified + uninstalled below)

_sha, _load, _git, _abs, Verifier = w81._sha, w81._load, w81._git, w81._abs, w81.Verifier

S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT = os.path.join(S53, 'gap_closeout_w82')
PREDICTIONS = os.path.join(OUT, 'predictions_w82.json')
PREDICTIONS_SHA256 = '35874c444a20eedec1a8f7a4673f29347d7221d935d6b32909f22ba9529476e9'
PREDICTIONS_COMMIT = '0cd0ab86'
OUT_JSON = os.path.join(OUT, 'gap_closeout_w82.json')
OUT_MANIFEST = os.path.join(OUT, 'gap_closeout_w82_manifest_sha256.json')
LAUNCH_LOG = os.path.join(S53, 'gap_closeout_w82_launch.log')

# --- 2x2 (items 1, 2) ---------------------------------------------------------------------------------------------------
AROW = os.path.join(S53, 'alpha_row', 'campaign_s53_alpha_row_v25')
PAIR1_MANIFEST = os.path.join(AROW, 'pair_1_manifest_sha256.json')
W72_JSON = os.path.join(S53, 'alpha_row', 'recompute_w72', 'alpha_row_recompute.json')
W72_MANIFEST = os.path.join(S53, 'alpha_row', 'recompute_w72', 'recompute_w72_manifest_sha256.json')
CASE_2X2 = os.path.join('data', 'SRP1', 'Results', 'P515S52', 'pilot_instance', 'SRP1__s52_pilot_2x2.json')
CELLS_2X2 = {'x0': {'label': 'x0_a0p50', 'eval_dir': os.path.join(AROW, 'evals', '7d53b6f21b686a44_x0_a0p50')},
             'unit': {'label': 'n7_4h_e1_a0p50', 'eval_dir': os.path.join(AROW, 'evals', '711fce9aa74d6878_n7_4h_e1_a0p50')}}
EVAL_FILES_2X2 = ('evaluation_record.json', 'component_levels_terminal.json', 'interface_settlement_detail_s31c.json',
                  'multiscenario_terminal.json', 'network_failures_s39_D.jsonl')
W74_TSO_TERMINAL = {'x0_a0p50': 18637.754597773586, 'n7_4h_e1_a0p50': 3558.841539982312}
W74_DIFF = 15078.913057791275
BAR_SUM_2X2_EXPECTED = 16551.504955768585

# --- SRP1 C2 baseline (item 3) ------------------------------------------------------------------------------------------
S47 = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert')
S47_MANIFEST = os.path.join(S47, 'campaign_manifest_sha256.json')
S47_RESULTS = os.path.join(S47, 'campaign_results.json')
S47_UNIT_DIR = os.path.join(S47, 'evals', 'bd504ecf5a288d44_n7_4h_e1')
S47_RUN_LOGS = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', 'p515s44_s47_recert_bd504ecf5a288d44_run', 'logs')
EVAL_FILES_SRP1 = ('evaluation_record.json', 'component_levels_terminal.json', 'interface_settlement_detail_s31c.json',
                   'network_failures_s39_D.jsonl')
C2_VALUE_EXPECTED = 259427.76775527

# --- Phase A (W81 re-derived) -------------------------------------------------------------------------------------------
W81_JSON = w81.OUT_JSON
W81_MANIFEST = w81.OUT_MANIFEST

IPOPT_DEFAULTS = {'compl_inf_tol': 1e-4, 'barrier_tol_factor': 10.0, 'mu_target': 0.0,
                  'source': 'IPOPT 3.14.18 option defaults (coin-or.github.io/Ipopt/OPTIONS.html); floor formula from '
                            'src/Algorithm/IpMonotoneMuUpdate.cpp tag releases/3.14.18'}
FORBIDDEN_MU_OPTIONS = ('barrier_tol_factor', 'mu_target', 'mu_min', 'mu_strategy', 'mu_linear_decrease_factor',
                        'mu_superlinear_decrease_power')
FLOOR_RTOL = 1e-3

FORMULAS = dict(w81.FORMULAS)
FORMULAS.update({
    'mu_floor': 'min(tol, cit x s) / (btf + 1); tol, cit from the options list printed before the run; cit = 1e-4 (IPOPT '
                'default) if absent; btf = 10, mu_target = 0 (defaults; barrier_tol_factor, mu_target, mu_min, mu_strategy, '
                'mu_linear_decrease_factor, mu_superlinear_decrease_power asserted ABSENT from the terminal solve\'s list). '
                'IPOPT 3.14.18 IpMonotoneMuUpdate: new_mu = Max(new_mu, mu_target, Min(tol, apply_obj_scaling(cit)) / '
                '(barrier_tol_factor + 1))',
    'floor_status': 'at iff |mu_last / mu_floor - 1| <= 1e-3; above iff > 1 + 1e-3; below iff < 1 - 1e-3',
    'regime': 's-independent iff cit x s <= tol (then g_floor = w n cit / 11); else scaling regime (g_floor = w n tol / (11 s))',
    'g_floor_b': 'w_b x n_pairs x mu_floor / s (at-floor counterfactual)',
    'excess_b': 'g_b - g_floor_b',
    'P': 'Delta (x0 - unit) of sum_b w_b n_pairs cit_b / (btf + 1)',
    'S': 'Delta G_floor - P',
    'C': 'Delta G - Delta G_floor (= Delta excess)',
    'identity': 'Delta G == P + S + C (asserted to 1e-9 relative + 1e-9 absolute)',
    'R': '|Delta G_total| / bar_sum; R_floor = |Delta G_floor_total| / bar_sum',
    'share': 'X / Delta G_total, X in {P, S, C}; frozen rule: component with share >= 0.5 is the supported explanation',
    'bar_sum_2x2': 'alpha_row_recompute.json cells[x0_a0p50].bar + cells[n7_4h_e1_a0p50].bar == row.unit.resolution_two_cell',
    'bar_sum_c2': 's47_recert campaign_results points.n7_4h_e1.bar_sum_with_x0 == x0_bar + bar.value (asserted)',
    'recovery_cycle_to_main_log_index': 'a network-failure event at ADMM cycle c is the main-log solve c + 1 (1-based; solve 1 '
                                        '= initialisation); asserted by that solve\'s EXIT not being "Optimal Solution Found."',
})
LIMITATIONS = list(w81.LIMITATIONS[:-1]) + [
    'The P/S/C decomposition is an accounting identity on the estimate: it does not decide whether a solve stopping above its '
    'floor is itself caused by its objective scaling factor (that would need solves; none is made).',
    'The at-floor counterfactual keeps each block\'s n_pairs and w_b unchanged.',
    '2x2 at alpha 0.5: the subproblem objective carries the row-18 premium, so the f / base units ratio is reported, not gated.',
    'The C2-baseline x0 is the C3-era Phase A x0 run pinned as Q(0) by the S2 statement (x = 0 has no storage; W21 case-file '
    'gate); this harness does not re-examine the pin.',
    'ESSO blocks are outside Q and are not estimated; ESSO logs are only counted (and hashed) for the solve-count identity.',
]


# ----------------------------------------------------------------------------------------------------------------------
def options_per_run(rel):
    """The FULL options list attributed to each IPOPT run, same attribution rule as w81.parse_log (the list printed in the
    gap before the run's banner)."""
    with open(_abs(rel), errors='replace') as handle:
        txt = handle.read()
    starts = [m.start() for m in re.finditer(r'This is Ipopt version', txt)]
    opt_starts = [m.start() for m in re.finditer(r'List of options:', txt)]
    out = []
    for i, st in enumerate(starts):
        prev = starts[i - 1] if i else -1
        opts_at = [o for o in opt_starts if prev < o < st]
        if not opts_at:
            out.append(None)
            continue
        block = txt[opts_at[-1]:st]
        block = block.split('*' * 20)[0]
        d = {}
        for m in re.finditer(r'^\s+(\w+) = (\S+)\s+\d+\s*$', block, re.M):
            d[m.group(1)] = m.group(2)
        out.append(d)
    return out


def floor_record(t, opts):
    if opts is None:
        raise SystemExit('terminal solve has no options list')
    bad = [k for k in FORBIDDEN_MU_OPTIONS if k in opts]
    if bad:
        raise SystemExit(f'mu-update option(s) set, defaults not in force: {bad}')
    tol = float(opts['tol'])
    if t['options_tol'] != tol:
        raise SystemExit(f'options parse disagreement: tol {tol} vs w81 parse {t["options_tol"]}')
    if 'compl_inf_tol' in opts:
        cit, cit_src = float(opts['compl_inf_tol']), 'options list'
    else:
        cit, cit_src = IPOPT_DEFAULTS['compl_inf_tol'], 'IPOPT default (absent from options list)'
    s, mu = t['obj_scale'], t['mu_last']
    btf = IPOPT_DEFAULTS['barrier_tol_factor']
    floor = min(tol, cit * s) / (btf + 1.0)
    ratio = mu / floor
    status = 'at' if abs(ratio - 1.0) <= FLOOR_RTOL else ('above' if ratio > 1.0 else 'below')
    return {'tol': tol, 'compl_inf_tol': cit, 'compl_inf_tol_source': cit_src, 'barrier_tol_factor': btf,
            'mu_floor': floor, 'mu_over_floor': ratio, 'floor_status': status,
            'regime': 's-independent' if cit * s <= tol else 'scaling (tol binds)',
            'compl_unscaled_over_cit': (t['compl_unscaled_final'] / cit if t['compl_unscaled_final'] is not None else None)}


def block_record(rel, K, pc_present, w, key):
    runs = w81.parse_log(rel)
    opts = options_per_run(rel)
    if len(opts) != len(runs):
        raise SystemExit(f'{rel}: options/run count mismatch')
    ident = w81.identify(runs, K, pc_present)
    if len(runs) <= K:
        raise SystemExit(f'{key}: terminal solve K + 1 = {K + 1} absent in {rel} ({len(runs)} runs)')
    t = runs[K]
    for f in ('n_pairs', 'mu_last', 'obj_scale'):
        if t[f] is None:
            raise SystemExit(f'{key}: terminal solve field {f} missing in {rel}')
    fl = floor_record(t, opts[K])
    cit = fl['compl_inf_tol']
    g = w * t['n_pairs'] * t['mu_last'] / t['obj_scale']
    g_floor = w * t['n_pairs'] * fl['mu_floor'] / t['obj_scale']
    return {'log': rel, 'sha256_at_read': _sha(rel), **ident, 'terminal_admm_solve': t, 'terminal_options': opts[K],
            'weight': w, **fl, 'g_primary_eur': g, 'g_floor_eur': g_floor, 'excess_eur': g - g_floor,
            'p_term_eur': w * t['n_pairs'] * cit / (IPOPT_DEFAULTS['barrier_tol_factor'] + 1.0),
            '_runs': runs}


def recovery_events(ed):
    ev = []
    with open(_abs(os.path.join(ed, 'network_failures_s39_D.jsonl'))) as handle:
        for line in handle:
            if line.strip():
                ev.append(json.loads(line))
    return ev


def check_recoveries(events, blocks, K, key_of):
    """Every recovery event must be at a non-terminal cycle; its main-log solve (cycle + 1) must be the failed primary."""
    out = []
    for e in events:
        key = key_of(e)
        b = blocks[key]
        idx = e['cycle'] + 1
        failed_exit = b['_runs'][idx - 1]['exit'] if idx <= len(b['_runs']) else None
        out.append({'block': key, 'cycle': e['cycle'], 'class': e.get('class'), 'termination': e.get('termination'),
                    'primary_termination': e.get('primary_termination'), 'main_log_index_1based': idx,
                    'main_log_exit_at_index': failed_exit,
                    'index_exit_not_optimal': failed_exit is not None and failed_exit != 'Optimal Solution Found.',
                    'not_terminal_cycle': e['cycle'] != K})
    return out


def aggregates(blocks, dso_node_of):
    groups = {'TSO': lambda k, b: k.startswith('TSO'), 'DSO': lambda k, b: not k.startswith('TSO'),
              **{f'DSO{n}': (lambda n: lambda k, b: dso_node_of(k) == n)(n) for n in (5, 7, 9)},
              'total': lambda k, b: True}
    agg = {}
    for g, sel in groups.items():
        bl = [b for k, b in blocks.items() if sel(k, b)]
        ts = [b['terminal_admm_solve'] for b in bl]
        agg[g] = {'n_blocks': len(bl), 'G_primary_eur': sum(b['g_primary_eur'] for b in bl),
                  'G_floor_eur': sum(b['g_floor_eur'] for b in bl), 'excess_eur': sum(b['excess_eur'] for b in bl),
                  'P_sum_eur': sum(b['p_term_eur'] for b in bl),
                  'n_at_floor': sum(b['floor_status'] == 'at' for b in bl),
                  'n_above_floor': sum(b['floor_status'] == 'above' for b in bl),
                  'n_below_floor': sum(b['floor_status'] == 'below' for b in bl),
                  'n_scaling_regime': sum(b['regime'] != 's-independent' for b in bl),
                  'mu_over_floor_max': max(b['mu_over_floor'] for b in bl),
                  'mu_over_floor_median': statistics.median(b['mu_over_floor'] for b in bl),
                  'obj_scale_min': min(t['obj_scale'] for t in ts), 'obj_scale_max': max(t['obj_scale'] for t in ts),
                  'n_pairs_sum': sum(t['n_pairs'] for t in ts),
                  'cit_sources': sorted({b['compl_inf_tol_source'] for b in bl}),
                  'exits': sorted({t['exit'] for t in ts})}
    return agg


def decompose(ax, au, bar_sum):
    out = {}
    for g in ax:
        dG = ax[g]['G_primary_eur'] - au[g]['G_primary_eur']
        dGf = ax[g]['G_floor_eur'] - au[g]['G_floor_eur']
        P = ax[g]['P_sum_eur'] - au[g]['P_sum_eur']
        S = dGf - P
        C = dG - dGf
        ok = abs(dG - (P + S + C)) <= 1e-9 * max(1.0, abs(dG)) + 1e-9
        out[g] = {'x0_G': ax[g]['G_primary_eur'], 'unit_G': au[g]['G_primary_eur'], 'delta_G': dG,
                  'delta_G_over_bar_sum': dG / bar_sum, 'x0_G_floor': ax[g]['G_floor_eur'],
                  'unit_G_floor': au[g]['G_floor_eur'], 'delta_G_floor': dGf, 'delta_G_floor_over_bar_sum': dGf / bar_sum,
                  'P_structural_pair_count': P, 'S_scaling_regime': S, 'C_incomplete_convergence': C,
                  'identity_holds': ok,
                  'share_P': P / dG if dG else None, 'share_S': S / dG if dG else None, 'share_C': C / dG if dG else None}
    return out


def explanation(dec_total):
    shares = {'incomplete convergence': dec_total['share_C'], 'scaling artifact': dec_total['share_S'],
              'structural pair-count difference': dec_total['share_P']}
    hits = [k for k, v in shares.items() if v is not None and v >= 0.5]
    return {'shares': shares, 'supported': hits[0] if len(hits) == 1 else ('mixed, none dominant' if not hits else hits)}


def strip_runs(blocks):
    for b in blocks.values():
        b.pop('_runs', None)
    return blocks


# ----------------------------------------------------------------------------------------------------------------------
def analyse_2x2(ver, srp):
    w74e = ver.manifest(w81.W74_MANIFEST)
    ver.against(w81.W74_JSON, w81.W74_MANIFEST, w74e)
    w74 = _load(w81.W74_JSON)
    p1e = ver.manifest(PAIR1_MANIFEST)
    w72e = ver.manifest(W72_MANIFEST)
    ver.against(W72_JSON, W72_MANIFEST, w72e)
    ver.tracked_only(CASE_2X2, 's52_pilot_2x2 case (years, days, discount) for the production weight cross-check')
    case = _load(CASE_2X2)
    sim = SimpleNamespace(years=dict(case['Years']), days=dict(case['Days']), discount_factor=case['DiscountFactor'])
    w72 = _load(W72_JSON)
    cells = {}
    for lab, spec in CELLS_2X2.items():
        ed = spec['eval_dir']
        for f in EVAL_FILES_2X2:
            ver.against(os.path.join(ed, f), PAIR1_MANIFEST, p1e)
        rec = _load(os.path.join(ed, 'evaluation_record.json'))
        comp = _load(os.path.join(ed, 'component_levels_terminal.json'))['blocks']
        ms = _load(os.path.join(ed, 'multiscenario_terminal.json'))
        settl = _load(os.path.join(ed, 'interface_settlement_detail_s31c.json'))['per_block_interface_settlement']
        wcell = w74['cells'][spec['label']]
        K = rec['cycles_run']
        if K != wcell['cycles_run'] or rec['eval_key'] != wcell['eval_key']:
            raise SystemExit(f'2x2 {lab}: evaluation_record disagrees with W74 (K or eval_key)')
        pc_present = rec.get('post_certification') is not None
        blocks = {}
        for key74, lb in wcell['log_blocks'].items():
            rel = lb['log']
            ver.against(rel, w81.W74_MANIFEST, w74e)
            if _sha(rel) != lb['sha256_at_read']:
                ver.failures.append(f'{rel}: differs from W74 sha256_at_read')
            if key74.startswith('TSO'):
                key = key74
            else:
                node, y, d = key74[3:].split('|')
                key = f'DSO|{node}|{y}|{d}'
            _, y, d = key74.split('|')
            w = ms['blocks'][key]['admm_block_weight']
            b = block_record(rel, K, pc_present, w, key)
            s = settl[key]
            base = (sum(comp[key]['unweighted'][n] for n in w81.BASE_COMPONENTS) + s['interface_settlement_unweighted'])
            w_prod = srp._get_admm_block_weight(sim, y, d)
            b.update({'block_w74': key74, 'weight_component_levels': comp[key]['admm_block_weight'],
                      'weight_production_fn': w_prod,
                      'weight_from_settlement': s['interface_settlement_weighted'] / s['interface_settlement_unweighted'],
                      'base_unweighted_eur_per_rep_day': base,
                      'units_ratio_f_over_base_informational': b['terminal_admm_solve']['objective_unscaled_final'] / base})
            b['weight_crosscheck_ok'] = all(abs(v / w - 1) <= 1e-9 for v in
                                            (b['weight_component_levels'], w_prod, b['weight_from_settlement']))
            # the W74 JSON's terminal fields must equal this parse
            t74 = lb['terminal_admm_solve']
            b['equals_w74_terminal_fields'] = all(t74[f] == b['terminal_admm_solve'][f] for f in
                                                  ('obj_scale', 'mu_last', 'n_pairs', 'iterations', 'objective_unscaled_final'))
            blocks[key] = b
        rec_events = recovery_events(ed)
        rec_check = check_recoveries(
            rec_events, blocks, K,
            lambda e: (f"TSO|{e['year']}|{e['day']}" if e['agent'] == 'TSO'
                       else f"DSO|{e['node_id']}|{e['year']}|{e['day']}"))
        agg = aggregates(blocks, lambda k: int(k.split('|')[1]) if k.startswith('DSO') else None)
        cells[lab] = {'label': spec['label'], 'eval_dir': ed, 'candidate_key': rec['candidate_key'],
                      'eval_key': rec['eval_key'], 'campaign_spec_sha256': rec['campaign_spec_sha256'],
                      'status': rec['status'], 'cycles_run': K, 'post_certification_present': pc_present,
                      'alpha_in_force': ms['summary'].get('alpha_in_force'),
                      'identification_holds_all_blocks': all(b['identification_holds'] for b in blocks.values()),
                      'weights_crosscheck_all': all(b['weight_crosscheck_ok'] for b in blocks.values()),
                      'equals_w74_terminal_fields_all': all(b['equals_w74_terminal_fields'] for b in blocks.values()),
                      'recovery_events': rec_check,
                      'recoveries_all_non_terminal_and_located': all(r['not_terminal_cycle'] and r['index_exit_not_optimal']
                                                                     for r in rec_check),
                      'units_ratio_informational_min': min(b['units_ratio_f_over_base_informational'] for b in blocks.values()),
                      'units_ratio_informational_max': max(b['units_ratio_f_over_base_informational'] for b in blocks.values()),
                      'aggregates': agg, 'blocks': strip_runs(blocks)}
    bx = w72['cells']['x0_a0p50']['bar']
    bu = w72['cells']['n7_4h_e1_a0p50']['bar']
    bar_sum = w72['row']['unit']['resolution_two_cell']
    if abs(bx + bu - bar_sum) > 1e-6 or abs(bar_sum - BAR_SUM_2X2_EXPECTED) > 1e-9:
        raise SystemExit(f'2x2 bar-sum check failed: {bx} + {bu} vs {bar_sum}')
    # control: W74's TSO-only terminal figures with W74's weights
    ctl = {}
    for lab, spec in CELLS_2X2.items():
        g = cells[lab]['aggregates']['TSO']['G_primary_eur']
        ctl[spec['label']] = {'G_TSO': g, 'w74': W74_TSO_TERMINAL[spec['label']],
                              'reproduces': abs(g - W74_TSO_TERMINAL[spec['label']]) <= 1e-9 * W74_TSO_TERMINAL[spec['label']]}
    dT = cells['x0']['aggregates']['TSO']['G_primary_eur'] - cells['unit']['aggregates']['TSO']['G_primary_eur']
    ctl['difference'] = {'value': dT, 'w74': W74_DIFF, 'reproduces': abs(dT - W74_DIFF) <= 1e-9 * W74_DIFF}
    dec = decompose(cells['x0']['aggregates'], cells['unit']['aggregates'], bar_sum)
    return {'cells': cells, 'bars': {'x0_a0p50': bx, 'n7_4h_e1_a0p50': bu, 'bar_sum': bar_sum, 'source': W72_JSON},
            'control_w74': ctl, 'differences_x0_minus_unit': dec,
            'R': abs(dec['total']['delta_G']) / bar_sum, 'R_floor': abs(dec['total']['delta_G_floor']) / bar_sum,
            'R_TSO_only_w74_convention': abs(dec['TSO']['delta_G']) / bar_sum,
            'explanation': explanation(dec['total']), 'explanation_TSO_only': explanation(dec['TSO'])}


def srp1_cell(ver, label, ed, man_rel, entries, logs_dir, case, srp):
    for f in EVAL_FILES_SRP1:
        p = os.path.join(ed, f)
        if f == 'network_failures_s39_D.jsonl' and p not in entries:
            if os.path.exists(_abs(p)):
                ver.failures.append(f'{p} exists but is not in {man_rel}')
            continue
        ver.against(p, man_rel, entries)
    rec = _load(os.path.join(ed, 'evaluation_record.json'))
    comp = _load(os.path.join(ed, 'component_levels_terminal.json'))['blocks']
    settl = _load(os.path.join(ed, 'interface_settlement_detail_s31c.json'))['per_block_interface_settlement']
    K = rec['cycles_run']
    pc_present = rec.get('post_certification') is not None
    sim = SimpleNamespace(years=dict(case['Years']), days=dict(case['Days']), discount_factor=case['DiscountFactor'])
    blocks = {}
    for kind, node, net, y, d in w81.block_list(case):
        key = f'TSO|{y}|{d}' if kind == 'TSO' else f'DSO|{node}|{y}|{d}'
        rel = os.path.join(logs_dir, f'optim_log_{net}_{y}_{d}.log')
        c = comp[key]
        w = c['admm_block_weight']
        b = block_record(rel, K, pc_present, w, key)
        s = settl[key]
        base = sum(c['unweighted'][n] for n in w81.BASE_COMPONENTS) + s['interface_settlement_unweighted']
        w_prod = srp._get_admm_block_weight(sim, y, d)
        w_settl = s['interface_settlement_weighted'] / s['interface_settlement_unweighted']
        b.update({'agent': kind, 'node': node, 'network': net, 'year': y, 'day': d,
                  'weight_production_fn': w_prod, 'weight_from_settlement': w_settl,
                  'weight_crosscheck_ok': abs(w_prod / w - 1) <= 1e-9 and abs(w_settl / w - 1) <= 1e-9,
                  'base_unweighted_eur_per_rep_day': base,
                  'units_ratio_f_over_base': b['terminal_admm_solve']['objective_unscaled_final'] / base})
        blocks[key] = b
    fail_path = os.path.join(ed, 'network_failures_s39_D.jsonl')
    events = recovery_events(ed) if os.path.exists(_abs(fail_path)) else []
    rec_check = check_recoveries(events, blocks, K,
                                 lambda e: (f"TSO|{e['year']}|{e['day']}" if e['agent'] == 'TSO'
                                            else f"DSO|{e['node_id']}|{e['year']}|{e['day']}"))
    recovery_logs = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(_abs(logs_dir), '*recovery*')))
    n_recovery_runs = 0
    for p in recovery_logs:
        with open(_abs(p), errors='replace') as handle:
            n_recovery_runs += handle.read().count('This is Ipopt version')
    esso_logs = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(_abs(logs_dir), 'optim_log_esso_node*.txt')))
    n_esso_runs = 0
    for p in esso_logs:
        with open(_abs(p), errors='replace') as handle:
            n_esso_runs += handle.read().count('This is Ipopt version')
    n_net = len(blocks)
    net_runs = sum(b['n_solves'] for b in blocks.values())
    identity = {'n_network_blocks': n_net, 'n_network_ipopt_runs': net_runs, 'n_recovery_logs': len(recovery_logs),
                'n_recovery_ipopt_runs': n_recovery_runs, 'n_recovery_events': len(events), 'n_esso_logs': len(esso_logs),
                'n_esso_ipopt_runs': n_esso_runs,
                'solve_profile_permitted_solve': rec['solve_profile']['observed']['permitted_solve'],
                'record_identity_holds_field': rec['solve_profile'].get('identity_holds'),
                'expected': n_net * (K + 1 + (1 if pc_present else 0)) + n_recovery_runs + n_esso_runs,
                'definition': 'permitted_solve == 48 x (K + 1) [network main logs: init + K cycles, no polish] + IPOPT runs '
                              'in every *recovery* log + IPOPT runs in every ESSO log'}
    identity['holds'] = ((not pc_present) and net_runs == n_net * (K + 1)
                         and identity['expected'] == identity['solve_profile_permitted_solve']
                         and n_recovery_runs == len(events))
    agg = aggregates(blocks, lambda k: int(k.split('|')[1]) if k.startswith('DSO') else None)
    return {'identity': label, 'eval_dir': ed, 'candidate_key': rec['candidate_key'], 'eval_key': rec['eval_key'],
            'campaign_spec_sha256': rec['campaign_spec_sha256'], 'status': rec['status'], 'cycles_run': K,
            'certified_gross_operational_cost': rec['certified_cost'], 'post_certification_present': pc_present,
            'logs_dir': logs_dir, 'recovery_logs': recovery_logs,
            'recovery_logs_sha256_at_read': {p: _sha(p) for p in recovery_logs},
            'esso_logs_sha256_at_read': {p: _sha(p) for p in esso_logs},
            'recovery_events': rec_check,
            'recoveries_all_non_terminal_and_located': all(r['not_terminal_cycle'] and r['index_exit_not_optimal']
                                                           for r in rec_check),
            'solve_count_identity': identity,
            'identification_holds_all_blocks': all(b['identification_holds'] for b in blocks.values()),
            'weights_crosscheck_all': all(b['weight_crosscheck_ok'] for b in blocks.values()),
            'units_ratio_min': min(b['units_ratio_f_over_base'] for b in blocks.values()),
            'units_ratio_max': max(b['units_ratio_f_over_base'] for b in blocks.values()),
            'units_abs_dev_median': statistics.median(abs(b['units_ratio_f_over_base'] - 1) for b in blocks.values()),
            'aggregates': agg, 'blocks': strip_runs(blocks)}


def analyse_srp1(ver, srp):
    w81e = ver.manifest(W81_MANIFEST)
    ver.against(W81_JSON, W81_MANIFEST, w81e)
    w = _load(W81_JSON)
    ver.tracked_only(w81.CASE_JSON, 'SRP1 case definition')
    case = _load(w81.CASE_JSON)
    # ---- Phase A: re-derived from its re-verified logs (hashes against W81's committed manifest)
    phase_a = {}
    for lab, spec in w81.CELLS.items():
        man = w81.CAMPAIGN_MANIFEST[spec['campaign']]
        entries = ver.manifest(man)
        c81 = w['cells'][lab]
        for b81 in c81['blocks'].values():
            ver.against(b81['log'], W81_MANIFEST, w81e)
        phase_a[lab] = srp1_cell(ver, spec['identity'], spec['eval_dir'], man, entries, c81['logs_dir'], case, srp)
        phase_a[lab]['equals_w81_terminal_fields_all'] = all(
            phase_a[lab]['blocks'][k]['terminal_admm_solve'] == c81['blocks'][k]['terminal_admm_solve'] for k in c81['blocks'])
        phase_a[lab]['equals_w81_G'] = all(
            abs(phase_a[lab]['blocks'][k]['g_primary_eur'] - c81['blocks'][k]['g_primary_eur']) <= 1e-12 * max(
                1.0, abs(c81['blocks'][k]['g_primary_eur'])) for k in c81['blocks'])
    bar_sum_a = w['phase_a']['bar_sum']
    dec_a = decompose(phase_a['x0']['aggregates'], phase_a['unit']['aggregates'], bar_sum_a)
    res_a = {'cells': phase_a, 'bar_sum': bar_sum_a, 'differences_x0_minus_unit': dec_a,
             'R': abs(dec_a['total']['delta_G']) / bar_sum_a, 'R_floor': abs(dec_a['total']['delta_G_floor']) / bar_sum_a,
             'R_w81_committed': w['verdict']['R'],
             'reproduces_w81_R': abs(abs(dec_a['total']['delta_G']) / bar_sum_a - w['verdict']['R']) <= 1e-12,
             'explanation_of_residual': explanation(dec_a['total'])}
    # ---- C2 baseline: x0 = Phase A x0 (pinned Q(0)), unit = s47_recert n7_4h_e1
    s47e = ver.manifest(S47_MANIFEST)
    ver.against(S47_RESULTS, S47_MANIFEST, s47e)
    cr = _load(S47_RESULTS)
    pt = cr['points']['n7_4h_e1']
    bi = cr['baseline_inputs']
    pin = {'Q0_eval_dir': bi['Q0_eval_dir'], 'x0_candidate_key': bi['x0_candidate_key'],
           'Q0_certification_cycle': bi['Q0_certification_cycle'], 'Q0_bar_eur': bi['Q0_bar_eur'], 'Q0_eur': bi['Q0_eur']}
    pin['x0_is_phase_a_x0'] = (bi['Q0_eval_dir'] == w81.CELLS['x0']['eval_dir']
                               and bi['x0_candidate_key'] == phase_a['x0']['candidate_key']
                               and bi['Q0_certification_cycle'] == phase_a['x0']['cycles_run'])
    unit = srp1_cell(ver, 's47_recert:n7_4h_e1', S47_UNIT_DIR, S47_MANIFEST, s47e, S47_RUN_LOGS, case, srp)
    if pt['eval_dir'] != S47_UNIT_DIR or pt['candidate_key'] != unit['candidate_key']:
        raise SystemExit('s47_recert point record disagrees with the unit eval dir')
    bar_x0, bar_u, bar_sum_c2 = pt['x0_bar'], pt['bar']['value'], pt['bar_sum_with_x0']
    if abs(bar_x0 + bar_u - bar_sum_c2) > 1e-6 or abs(bar_x0 - w['phase_a']['rows']['x0']['bar_eur']) > 1e-9:
        raise SystemExit('C2-baseline bar-sum check failed')
    if abs(pt['value_eur'] - C2_VALUE_EXPECTED) > 1e-6:
        raise SystemExit(f'C2 value {pt["value_eur"]} != {C2_VALUE_EXPECTED}')
    dec_c = decompose(phase_a['x0']['aggregates'], unit['aggregates'], bar_sum_c2)
    res_c = {'x0_pin': pin, 'unit': unit, 'bars': {'x0_bar': bar_x0, 'unit_bar': bar_u, 'bar_sum': bar_sum_c2,
                                                   'source': S47_RESULTS},
             'value_eur': pt['value_eur'], 'differences_x0_minus_unit': dec_c,
             'R': abs(dec_c['total']['delta_G']) / bar_sum_c2, 'R_floor': abs(dec_c['total']['delta_G_floor']) / bar_sum_c2,
             'delta_G_over_value': dec_c['total']['delta_G'] / pt['value_eur'],
             'explanation_of_residual': explanation(dec_c['total']),
             'unit_TSO_obj_scale_lower_than_x0_blocks': sum(
                 1 for k in unit['blocks'] if k.startswith('TSO') and unit['blocks'][k]['terminal_admm_solve']['obj_scale']
                 < phase_a['x0']['blocks'][k]['terminal_admm_solve']['obj_scale'])}
    return res_a, res_c, w


def score(two, pa, c2):
    x, u = two['cells']['x0'], two['cells']['unit']
    d = two['differences_x0_minus_unit']
    cells_a = pa['cells']
    all_at = lambda cell: cell['aggregates']['total']['n_at_floor'] == cell['aggregates']['total']['n_blocks']  # noqa: E731
    return {
        'P1_control_w74': all(v['reproduces'] for v in two['control_w74'].values()),
        'P2_identification_2x2': (x['identification_holds_all_blocks'] and u['identification_holds_all_blocks']
                                  and x['recoveries_all_non_terminal_and_located']
                                  and u['recoveries_all_non_terminal_and_located']),
        'P3_dso_floor_2x2': all(c['aggregates']['DSO']['n_at_floor'] == 60 for c in (x, u)),
        'P4_dso_diff_2x2': d['DSO']['delta_G'] <= 0 and abs(d['DSO']['delta_G']) <= 500.0,
        'P5_complete_ratio_2x2': 0.85 <= two['R'] <= 0.92,
        'P6_unit_tso_floor_2x2': u['aggregates']['TSO']['n_at_floor'] == 20,
        'P7_x0_tso_above_2x2': x['aggregates']['TSO']['n_above_floor'] >= 14,
        'P8_counterfactual_2x2': (d['total']['delta_G_floor'] <= 0 and two['R_floor'] < 0.05
                                  and x['aggregates']['total']['n_scaling_regime'] == 0
                                  and u['aggregates']['total']['n_scaling_regime'] == 0),
        'P9_explanation_2x2': d['total']['share_C'] is not None and d['total']['share_C'] >= 1.0,
        'P10_c2_x0_identical': c2['x0_pin']['x0_is_phase_a_x0'] and cells_a['x0']['equals_w81_terminal_fields_all'],
        'P11_recert_identification': (c2['unit']['identification_holds_all_blocks']
                                      and c2['unit']['solve_count_identity']['holds']
                                      and not c2['unit']['post_certification_present']
                                      and c2['unit']['recoveries_all_non_terminal_and_located']),
        'P12_recert_floor': all_at(c2['unit']),
        'P13_c2_ratio': c2['R'] < 0.05,
        'P14_phase_a_floor': all(all_at(c) for c in cells_a.values()),
        'P15_recert_units': (all(abs(b['units_ratio_f_over_base'] - 1) <= 0.05 for b in c2['unit']['blocks'].values())
                             and c2['unit']['units_abs_dev_median'] <= 0.01),
    }


def run():
    started = time.time()
    if os.path.exists(_abs(OUT_JSON)):
        raise SystemExit(f'REFUSED: output exists (write-once): {OUT_JSON}')
    ver = Verifier()
    if _sha(PREDICTIONS) != PREDICTIONS_SHA256:
        raise SystemExit('predictions file hash differs from the recorded one')
    ver.tracked_only(PREDICTIONS, f'predictions (committed at {PREDICTIONS_COMMIT}, before the harness)')
    # capture-path assertion BEFORE any analysis
    missing = []
    w74 = _load(w81.W74_JSON)
    for spec in CELLS_2X2.values():
        for f in EVAL_FILES_2X2:
            if not os.path.isfile(_abs(os.path.join(spec['eval_dir'], f))):
                missing.append(os.path.join(spec['eval_dir'], f))
        for lb in w74['cells'][spec['label']]['log_blocks'].values():
            if not os.path.isfile(_abs(lb['log'])):
                missing.append(lb['log'])
    case = _load(w81.CASE_JSON)
    for f in ('evaluation_record.json', 'component_levels_terminal.json', 'interface_settlement_detail_s31c.json'):
        if not os.path.isfile(_abs(os.path.join(S47_UNIT_DIR, f))):
            missing.append(os.path.join(S47_UNIT_DIR, f))
    for kind, node, net, y, d in w81.block_list(case):
        p = os.path.join(S47_RUN_LOGS, f'optim_log_{net}_{y}_{d}.log')
        if not os.path.isfile(_abs(p)):
            missing.append(p)
    for p in (PAIR1_MANIFEST, W72_JSON, W72_MANIFEST, CASE_2X2, S47_MANIFEST, S47_RESULTS, W81_JSON, W81_MANIFEST,
              w81.W74_JSON, w81.W74_MANIFEST):
        if not os.path.isfile(_abs(p)):
            missing.append(p)
    if missing:
        raise SystemExit(f'CAPTURE PATH MISSING ({len(missing)}): {missing[:10]}')
    print('[W82] capture paths present: 2x2 2 cells x 80 blocks, s47_recert unit 48 blocks, W81/W74/W72/pair_1/s47 inputs',
          flush=True)

    import shared_resources_planning as srp   # production _get_admm_block_weight; import only, nothing built or solved
    two = analyse_2x2(ver, srp)
    pa, c2, w81_json = analyse_srp1(ver, srp)
    if ver.failures:
        raise SystemExit(f'INPUT VERIFICATION FAILED: {ver.failures[:10]}')
    for name, dec in (('2x2', two['differences_x0_minus_unit']), ('PhaseA', pa['differences_x0_minus_unit']),
                      ('C2', c2['differences_x0_minus_unit'])):
        if not all(v['identity_holds'] for v in dec.values()):
            raise SystemExit(f'{name}: decomposition identity failed')
    for lab, c in two['cells'].items():
        a = c['aggregates']
        print(f"[W82] 2x2 {c['label']} K={c['cycles_run']} ident={c['identification_holds_all_blocks']} "
              f"w74-fields={c['equals_w74_terminal_fields_all']} w-xcheck={c['weights_crosscheck_all']} "
              f"recoveries={len(c['recovery_events'])} ok={c['recoveries_all_non_terminal_and_located']} | "
              f"TSO G {a['TSO']['G_primary_eur']:.2f} floor {a['TSO']['G_floor_eur']:.2f} at/above/below "
              f"{a['TSO']['n_at_floor']}/{a['TSO']['n_above_floor']}/{a['TSO']['n_below_floor']} max mu/floor "
              f"{a['TSO']['mu_over_floor_max']:.3f} | DSO G {a['DSO']['G_primary_eur']:.2f} floor {a['DSO']['G_floor_eur']:.2f}"
              f" at/above/below {a['DSO']['n_at_floor']}/{a['DSO']['n_above_floor']}/{a['DSO']['n_below_floor']}", flush=True)
    print(f"[W82] 2x2 control W74: {two['control_w74']}", flush=True)
    for lab, c in list(pa['cells'].items()) + [('c2_unit', c2['unit'])]:
        a = c['aggregates']
        print(f"[W82] SRP1 {c['identity']} K={c['cycles_run']} ident={c['identification_holds_all_blocks']} solve-identity="
              f"{c['solve_count_identity']['holds']} ({c['solve_count_identity']['expected']} vs "
              f"{c['solve_count_identity']['solve_profile_permitted_solve']}) recoveries={len(c['recovery_events'])} "
              f"w-xcheck={c['weights_crosscheck_all']} units [{c['units_ratio_min']:.6f}, {c['units_ratio_max']:.6f}] | "
              f"TSO G {a['TSO']['G_primary_eur']:.2f} at {a['TSO']['n_at_floor']}/12 | DSO G {a['DSO']['G_primary_eur']:.2f} "
              f"at {a['DSO']['n_at_floor']}/36 | total at/above/below {a['total']['n_at_floor']}/{a['total']['n_above_floor']}"
              f"/{a['total']['n_below_floor']}", flush=True)

    outcomes = score(two, pa, c2)
    preds = _load(PREDICTIONS)['predictions']
    scored = {k: {'statement': v['statement'], 'expected': v['expected'], 'observed': outcomes[k],
                  'result': 'CONFIRMED' if outcomes[k] == v['expected'] else 'REFUTED'} for k, v in preds.items()}
    sub_everywhere = two['R'] < 1 and pa['R'] < 1 and c2['R'] < 1
    verdict = {
        'pairs': {'SRP1 Phase A (a0_c7:x0 vs a0_c7:n7_p0.25_e1.0)': {'R': pa['R'], 'R_floor': pa['R_floor'],
                                                                      'delta_G': pa['differences_x0_minus_unit']['total']['delta_G'],
                                                                      'bar_sum': pa['bar_sum']},
                  'SRP1 C2 baseline (pinned a0_c7:x0 vs s47_recert:n7_4h_e1)': {
                      'R': c2['R'], 'R_floor': c2['R_floor'], 'delta_G': c2['differences_x0_minus_unit']['total']['delta_G'],
                      'bar_sum': c2['bars']['bar_sum']},
                  '2x2 complete (x0_a0p50 vs n7_4h_e1_a0p50)': {
                      'R': two['R'], 'R_floor': two['R_floor'], 'delta_G': two['differences_x0_minus_unit']['total']['delta_G'],
                      'bar_sum': two['bars']['bar_sum'], 'R_TSO_only': two['R_TSO_only_w74_convention']}},
        'sub_resolution_everywhere_measured': sub_everywhere,
        'explanation_2x2': two['explanation'],
        'rule': _load(PREDICTIONS)['frozen_verdict_rule']}
    manifest_scan = w81.scan_manifests_for_logs(sorted({S47_RUN_LOGS}))
    payload = {
        'task': 'P5.15 Addendum 45 item 3 close-out / W82: complete 2x2 gap difference, mu-floor test and at-floor '
                'counterfactual, SRP1 C2-baseline pair against its own bars (read-only, zero solves)',
        'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD').stdout.strip(),
        'script_sha256': _sha(os.path.basename(__file__)), 'w81_module_sha256': _sha('p515_s53_w81_srp1_gap_estimate.py'),
        'objective_convention': 'Q = certified gross_operational_cost (settlement-excluded, gross of salvage); gap figures '
                                'are EUR of Q (block weight applied); value = Q(0) - Q(unit)',
        'formulas': FORMULAS, 'ipopt_defaults_used': IPOPT_DEFAULTS, 'limitations': LIMITATIONS,
        'claim_scope': {
            'licenses': 'order-of-magnitude comparison of estimated interior-point barrier offsets on terminal ADMM solves, '
                        'for three named pairs, against each pair\'s bar-sum; the at-floor counterfactual and the P/S/C '
                        'accounting decomposition of each difference',
            'does_not_license': ['a measured change in Q or value', 'a causal statement that the scaling factor caused '
                                 'above-floor stopping', 'any other cell, alpha or instance', 'the ESSO'],
            'searched': ['W74 JSON + manifest and the 160 2x2 x0_a0p50/n7_4h_e1_a0p50 network logs',
                         'pair_1_manifest-verified 2x2 eval artifacts incl. network_failures', 'W72 recompute (2x2 bars)',
                         'W81 JSON + manifest and the 144 Phase A logs (re-verified, re-parsed)',
                         's47_recert campaign_results + manifest, the unit eval artifacts, its 48 main logs, recovery '
                         'log(s) and ESSO logs', 'every tracked *manifest*.json for coverage of the s47_recert run-log '
                         'directory (manifest_scan)']},
        'item1_item2_2x2': two, 'item3_srp1_c2_baseline': c2, 'srp1_phase_a_rederived': pa, 'verdict': verdict,
        'predictions': {'file': PREDICTIONS, 'sha256': PREDICTIONS_SHA256, 'committed_at': PREDICTIONS_COMMIT,
                        'scored': scored},
        'manifest_scan_for_s47_recert_logs': manifest_scan, 'inputs': ver.records}
    return payload, started


def _print_summary(p):
    for name, sec in (('2x2', p['item1_item2_2x2']), ('PhaseA', p['srp1_phase_a_rederived']),
                      ('C2', p['item3_srp1_c2_baseline'])):
        for g, dd in sec['differences_x0_minus_unit'].items():
            print(f"[W82] {name} {g}: x0 {dd['x0_G']:.2f} unit {dd['unit_G']:.2f} dG {dd['delta_G']:.2f} "
                  f"({dd['delta_G_over_bar_sum']:.4f} x bar) | floor x0 {dd['x0_G_floor']:.2f} unit {dd['unit_G_floor']:.2f} "
                  f"dG_floor {dd['delta_G_floor']:.2f} | P {dd['P_structural_pair_count']:.2f} S {dd['S_scaling_regime']:.2f}"
                  f" C {dd['C_incomplete_convergence']:.2f}", flush=True)
    for k, v in p['verdict']['pairs'].items():
        print(f"[W82] VERDICT {k}: R = {v['R']:.4f} (R_floor {v['R_floor']:.4f}), dG {v['delta_G']:.2f}, bar-sum "
              f"{v['bar_sum']:.2f}", flush=True)
    print(f"[W82] sub-resolution everywhere measured: {p['verdict']['sub_resolution_everywhere_measured']}; 2x2 "
          f"explanation {p['verdict']['explanation_2x2']}", flush=True)
    for k, s in p['predictions']['scored'].items():
        print(f"[W82] {k}: expected {s['expected']} observed {s['observed']} -> {s['result']}", flush=True)


def _finish(label):
    failures = GUARD.verify(0)
    failures81 = w81.GUARD.verify(0)
    print(f'[W82] {label} guard counts {dict(GUARD.counts)} verify(0) {failures}; W81-module guard counts '
          f'{dict(w81.GUARD.counts)} verify(0) {failures81}', flush=True)
    w81.GUARD.uninstall()
    GUARD.uninstall()
    return failures, failures81


def main_run():
    payload, exit_code = None, 1
    try:
        payload, started = run()
        exit_code = 0
    except SystemExit as exc:
        print(f'[W82] STOP: {exc}', flush=True)
    except Exception:                                   # noqa: BLE001 -- reported, then verify still runs
        traceback.print_exc()
    finally:
        failures = GUARD.verify(0)
        failures81 = w81.GUARD.verify(0)
        if payload is not None:
            payload['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': failures,
                                'w81_module_guard_counts': dict(w81.GUARD.counts),
                                'w81_module_guard_verify_0_failures': failures81}
            payload['wall_s'] = time.time() - started
            with open(_abs(OUT_JSON), 'x') as handle:
                json.dump(payload, handle, indent=1)
            _print_summary(payload)
            print(f'[W82] wrote {OUT_JSON}; wall {payload["wall_s"]:.1f}s', flush=True)
        failures, failures81 = _finish('run')
        sys.exit(exit_code if not (failures or failures81) else 1)


def main_manifest():
    exit_code = 1
    try:
        if os.path.exists(_abs(OUT_MANIFEST)):
            raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
        out = _load(OUT_JSON)
        me = os.path.basename(__file__)
        m = {OUT_JSON: _sha(OUT_JSON), LAUNCH_LOG: _sha(LAUNCH_LOG), me: _sha(me),
             'p515_s53_w81_srp1_gap_estimate.py': _sha('p515_s53_w81_srp1_gap_estimate.py')}
        bad = [] if out['script_sha256'] == m[me] else [me]
        if out['w81_module_sha256'] != m['p515_s53_w81_srp1_gap_estimate.py']:
            bad.append('p515_s53_w81_srp1_gap_estimate.py')
        for rel, r in out['inputs'].items():
            if _sha(rel) != r['sha256']:
                bad.append(rel)
            m[rel] = r['sha256']
        logs = {}
        for c in out['item1_item2_2x2']['cells'].values():
            logs.update({b['log']: b['sha256_at_read'] for b in c['blocks'].values()})
        for c in list(out['srp1_phase_a_rederived']['cells'].values()) + [out['item3_srp1_c2_baseline']['unit']]:
            logs.update({b['log']: b['sha256_at_read'] for b in c['blocks'].values()})
            logs.update(c['recovery_logs_sha256_at_read'])
            logs.update(c['esso_logs_sha256_at_read'])
        for rel, h in logs.items():
            if _sha(rel) != h:
                bad.append(rel)
            m[rel] = h
        if bad:
            raise SystemExit(f'REFUSED: changed since the run: {bad[:5]}')
        with open(_abs(OUT_MANIFEST), 'x') as handle:
            json.dump(m, handle, indent=1)
        print(f'[W82] wrote {OUT_MANIFEST}: {len(m)} entries', flush=True)
        exit_code = 0
    except SystemExit as exc:
        print(f'[W82] STOP: {exc}', flush=True)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
    finally:
        failures, failures81 = _finish('manifest')
        sys.exit(exit_code if not (failures or failures81) else 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--run']:
        main_run()
    elif sys.argv[1:] == ['--manifest']:
        main_manifest()
    else:
        _finish('usage')
        raise SystemExit('usage: --run | --manifest')
