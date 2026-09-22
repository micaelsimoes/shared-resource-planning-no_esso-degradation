"""
P5.15 Addendum 38 (task W38) -- F2, the DEMONSTRATION CASE of the planning method: the node-7 4 h CAPACITY LADDER
at 2025 under the flexibility-price multiplier m = 2, through the campaign harness
(`p515_s44_campaign_harness.py`) with the evaluation option `flex_price_multiplier` (W33 item 1, 2e82a23f;
zero-solve checks there; two-cycle bitwise gate at m = 1.0 PASS, 51e74279).

  MODEL VARIANT -- flexibility price x m   (m = 2; the DSO `cost_flex` profile, every year / day / hour, growth
  included, multiplied uniformly in the child before the DSO models are built; no workbook edit). m = 2 is a
  FLEXIBILITY-PRICE SCENARIO and is NEVER the baseline; every table in this stage is labelled as such.
  Ageing: the BASELINE (C2 + phi_cal 0.985 + soh_min 0.70), declared (`configuration.ess_ageing_baseline`).

THE LADDER is E in {1, 2, 3, 4, 5} MWh at node 7, 4 h, 2025 -- (0.25, 1.0), (0.5, 2.0), (0.75, 3.0), (1.0, 4.0),
(1.25, 5.0) MVA / MWh; nodes 5 and 9 empty.

THREE OF THE SIX POINTS ARE ALREADY COMMITTED AT m = 2 AND ARE NOT RE-RUN:
  x = 0    -- the flexibility-price ladder, dc468ab4 (`P515S49/campaign_s49_flex_ladder/campaign_results.json`)
  E = 1    -- the same ladder, dc468ab4
  E = 2    -- the marginal-MWh test, 0e36ec0a (`P515S50/campaign_s50_marginal/campaign_results.json`)
Each is pinned by path and sha256, re-checked at --freeze and --run, cross-checked against its own
`evaluation_record.json` and its campaign manifest, and its eval key is RECOMPUTED here and asserted equal to the
committed one; the committed value / resolution rows are recomputed from the pinned Q and bar by the ladder
launcher's own `per_m_row` and asserted bitwise equal to the committed rows.

SO THIS CAMPAIGN EVALUATES THREE NEW POINTS: E = 3, 4, 5 MWh at m = 2, in ONE wave at concurrency 3, in the order
    n7_4h_e3_m2, n7_4h_e4_m2, n7_4h_e5_m2
AA-on case file (declared), cap 500, 10 consecutive all-pass cycles, NO overrides, NO ageing model variant, NO
post-certification. Stop rule as a1a (STOP_RULE), evaluated after the wave (there is no later wave to stop;
recorded for form and for the exit code -- all three points are ONE region, so two non-certified points raise it).

RECORDED PER POINT AND IN THE LADDER TABLE (`table`, `ladder`; objective convention on every table: Q = certified
GROSS operational cost, settlement excluded): status (non-certified with cause), cycles, certification cycle, Q,
the bar, the rule-ten terminal-step-to-threshold ratios (production and gross), Boyd terminal ratios, the
multiplier and its label, the harness read-back, the flexibility cost per DSO, the solve reconciliation, wall time,
peak RSS, AA action counts, per-cycle trajectory path + sha256 -- all by L._point_result (the ladder launcher's own
function, BY IMPORT) -- and then, per rung:
  value = Q_m(0) - Q_m(x)             (Q_m(0) PINNED from dc468ab4)
  I(x)                                from the pinned W2 table, matched BY CANDIDATE KEY (asserted unique)
  value - I, and the resolution       bar_x + bar_x0 (the x0 bar pinned), with the reading
  value per MWh of ENERGY             value / E, against the 2025 new energy unit cost
  the TRUE MARGINAL MWh               delta value against the previous rung vs delta I = 317,957.0085035586 per MWh
                                      at 4 h (recomputed from the W2 table, its linearity checked), with the
                                      resolution of that difference = resolution_E + resolution_{E-1}
  value per full cycle                value / T, T = W25's T4 r2 cell-side full-cycle throughput (W25's
                                      `case_structure` / `_esso_capture` BY IMPORT, reproducing W25's committed T
                                      for the S2 record bitwise at --freeze and --run)
  budget                              I, slack at B = 1,000,000 EUR and feasibility; the BUDGET CORNER (3 MWh) is
                                      marked explicitly (`is_budget_corner`, `budget_corner`)
  per-cycle trajectory                path + sha256 (pinned rungs: the committed path and its manifest hash)

Two modes, attached, both streams captured, never detached:
  --freeze                    ZERO SOLVES: preconditions, pins, the three pinned rows, the W2 inputs and budget
                              corner, the W25 reproduction, rule eleven, `freeze_campaign_spec` + validation.
                              Prints the --run command.
  --run --spec-sha256 <sha>   loads THAT spec (the root must hold only it), re-checks everything plus the harness /
                              case-file / ESS-params / script / ladder-script / marginal-script sha256, the memory
                              preflight (refusing), takes the campaign lock, evaluates the wave, writes
                              campaign_results.json and campaign_manifest_sha256.json.
The parent never solves: this module's SolveProfileGuard(permitted=()) is installed on top of the ladder module's
parent guard and the marginal module's parent guard (both installed at import, BEFORE the harness and hence before
any model module) and of W25's module guard; all four are verified at exactly 0. Exit codes (--run): 0 every point
certified; 3 STOP_FOR_REVIEW (stop rule fired, harness clean); 2 a non-certified point (harness clean); 1 harness /
guard / read-back / precondition failure.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_f2_ladder_campaign.py --freeze \\
      > data/SRP1/Results/P515S51/campaign_s51_f2_ladder_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_f2_ladder_campaign.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S51/campaign_s51_f2_ladder_launch.log 2>&1
"""

import argparse
import os
import subprocess
import sys
import time

from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The marginal launcher imports the ladder launcher, which installs ITS parent guard (permitted=()) at import,
# before it imports the harness and hence before any model module is imported; the marginal launcher's own guard is
# installed immediately on top, and this module's on top of both. All are verified at 0. Every derived figure this
# stage shares with those stages is used BY IMPORT (L.*, M.*), so the value, resolution, throughput,
# flexibility-cost, budget-corner and solve-reconciliation definitions are literally the same code.
import p515_s50_marginal_campaign as M  # noqa: E402  (imports p515_s49_flex_ladder_campaign as M.L)
import p515_s49_flex_ladder_campaign as L  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S51 F2 ladder campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = L.LABEL
FLEX_LABEL = H.FLEX_PRICE_LABEL
SCENARIO_LABEL = ('FLEXIBILITY-PRICE SCENARIO m = 2 (MODEL VARIANT -- flexibility price x 2); NOT the baseline. '
                  'Every figure in this stage is a figure of that scenario.')
_P51 = os.path.join('data', 'SRP1', 'Results', 'P515S51')
_P50 = os.path.join('data', 'SRP1', 'Results', 'P515S50')
_P49 = os.path.join('data', 'SRP1', 'Results', 'P515S49')
CAMPAIGN_ID = 's51_f2_ladder'
CAMPAIGN_ROOT = os.path.join(REPO, _P51, 'campaign_s51_f2_ladder')
CASE_FILE_AA = dict(L.CASE_FILE_AA)
CAP = 500
CONCURRENCY = 3
REQUIRED_CONSECUTIVE_CYCLES = 10
ACTIVE_NODES = L.ACTIVE_NODES
YEAR = L.YEAR
UNIT_NODE = 7
M_FLEX = 2.0                      # the single flexibility-price multiplier of F2
DURATION_H = 4.0                  # the ladder's energy-to-power ratio
RUNGS = (1, 2, 3, 4, 5)           # E in MWh; (0.25 e, 1.0 e) MVA / MWh
PINNED_RUNGS = (1, 2)             # committed at m = 2 -- NOT re-run
NEW_RUNGS = (3, 4, 5)             # this campaign's evaluations
CASE_JSON_REL = L.CASE_JSON_REL
ESS_AGEING_BASELINE = dict(L.ESS_AGEING_BASELINE)

SPEC_V21 = {'path': os.path.join(_P51, 'frozen_s51_spec_v21_13cb828c.json'),
            'sha256': '13cb828c988dfaa5613675a4d34607ca0f74bcc885fc5fb2397904baf19e0530',
            'item': 'F2'}
LADDER_RESULTS = dict(M.LADDER_RESULTS)          # dc468ab4 -- x = 0 and E = 1 at m = 2
LADDER_SPEC = dict(M.LADDER_SPEC)
LADDER_MANIFEST = dict(M.LADDER_MANIFEST)
LADDER_SCRIPT = dict(M.LADDER_SCRIPT)
MARGINAL_RESULTS = {'path': os.path.join(_P50, 'campaign_s50_marginal', 'campaign_results.json'),
                    'sha256': 'a083768d1b5911ea9c45cdf3f67cc3dd09c1190edbea57aa92fd24076631118c',
                    'commit': '0e36ec0a',
                    'campaign': 's50_marginal (spec campaign_spec_s50_marginal_fa4b30ee)'}
MARGINAL_SPEC = {'path': os.path.join(_P50, 'campaign_s50_marginal', 'campaign_spec_s50_marginal_fa4b30ee.json'),
                 'sha256': 'fa4b30eee63d9b137df559379ed5cdd13a1b93e2a7b5622fd681b643618cb8a0', 'commit': '0e36ec0a'}
MARGINAL_MANIFEST = {'path': os.path.join(_P50, 'campaign_s50_marginal', 'campaign_manifest_sha256.json')}
MARGINAL_SCRIPT = {'path': 'p515_s50_marginal_campaign.py', 'commit': '2389d79b'}
ESS_PARAMS_FILE = dict(L.ESS_PARAMS_FILE)
A0_SPEC = dict(L.A0_SPEC)
INVESTMENT_COST_RESULTS = dict(L.INVESTMENT_COST_RESULTS)
W25_RESULTS = dict(L.W25_RESULTS)
PRIOR_W33 = dict(L.PRIOR_W33)

# Expected values, ASSERTED (never trusted from the task text): the pinned Q and bar at m = 2, I per MWh at 4 h,
# the 2025 new energy unit cost, the budget and the master capacity / ratio constraints.
Q_PINNED_EXPECTED = {0: 811016062.203051, 1: 810649709.6671975, 2: 810288986.5544469}
BAR_X0_M2_EXPECTED = 1296.2551001310349
CYCLES_PINNED_EXPECTED = {0: 134, 1: 132, 2: 144}
I_PER_MWH_4H_EXPECTED = 317957.0085035586
I_LINEARITY_TOL_EUR = 1e-6
ENERGY_UNIT_COST_2025_EXPECTED = M.ENERGY_UNIT_COST_2025_EXPECTED
BUDGET_EUR_EXPECTED = M.BUDGET_EUR_EXPECTED
MAX_CAPACITY_MWH_EXPECTED = 5.0
ENERGY_TO_POWER_RATIO_BOUNDS_EXPECTED = (2.0, 10.0)
BUDGET_CORNER_E_MWH_EXPECTED = 3.0

OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); value(x) = Q_m(0) - '
                        'Q_m(x) at m = 2 with Q_m(0) PINNED from the committed ladder (dc468ab4); terminal salvage '
                        'and net_operational_recourse = gross - salvage reported, excluded. At m = 2 the DSO '
                        'flexibility cost inside Q is priced at 2 x cost_flex -- a SCENARIO figure, not a baseline '
                        'figure.')
VALUE_DEFINITION = ('value(x) = Q_m(0) - Q_m(x) at m = 2; value - I with I = I(x) from the pinned W2 table matched '
                    'by candidate key; resolution = bar_x + bar_x0 (record.bar of each, same m, the x0 bar pinned): '
                    '|value - I| <= resolution is INDETERMINATE (CLAUDE.md: a difference smaller than the stopping '
                    'error is not a result; the bar is local and bounds stopping slack only); value per MWh of '
                    'ENERGY = value / E, compared with the 2025 new energy unit cost 253,877.6774385931 EUR/MWh '
                    '(I / E = 317,957.0085 EUR/MWh includes the 4 h power component); the TRUE MARGINAL MWh = this '
                    'rung minus the previous rung, delta value against delta I = 317,957.0085035586 per MWh at 4 h, '
                    'with resolution_E + resolution_{E-1} as the resolution of that difference; value per full '
                    'cycle = value / T, T = W25 T4 r2 cell-side full-cycle throughput of THIS cell at m = 2; '
                    'resolution per cycle = resolution / T.')
BUDGET_DEFINITION = M.BUDGET_DEFINITION
STOP_RULE = L.STOP_RULE
REGION_NOTE = ('regions as a1a (L._region_key: the node+duration ladder): all three evaluated points are ONE region '
               '(n7_4h_y2025), so two non-certified points would trigger (a) -- there is no later wave to stop, and '
               'the flag is recorded and returned as exit 3')

GIB = L.GIB
MEMORY_PER_CHILD_BUDGET_BYTES = L.MEMORY_PER_CHILD_BUDGET_BYTES  # 2.75 GiB (A0's per-child measure)
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE = L.MEMORY_RULE_TEMPLATE.format(concurrency=CONCURRENCY,
                                            per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
MEMORY_RULE_RATIONALE = L.MEMORY_RULE_RATIONALE

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 38 (F2, the demonstration case)',
    'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json F2',
    'Planner task W38 (3 evaluations: node 7 4 h E in {3, 4, 5} MWh at 2025, m = 2, concurrency 3, one wave, cap '
    '500, 10 cycles, AA-on case file, baseline ageing declared, no post-certification, stop rule as a1a; x = 0 and '
    'E = 1 pinned from the committed ladder dc468ab4 and E = 2 from the committed marginal test 0e36ec0a, not '
    're-run)',
    '2e82a23f (harness override + zero-solve checks), 51e74279 (two-cycle bitwise gate at m = 1.0), 812ee116 / '
    '55ff2cca / dc468ab4 (the flexibility-price ladder launcher, spec and results), 2389d79b / 0e36ec0a (the '
    'marginal-MWh launcher and results)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), LADDER_SCRIPT['path'], MARGINAL_SCRIPT['path'],
                     H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py', 'shared_energy_storage.py',
                     'p515_s49_flex_price_checks.py', 'p515_s49_flex_price_gate.py',
                     'p515_s47_tso_marginal_cost.py', CASE_JSON_REL)

_log = L._log
_banner = L._banner
_load = L._load


def _nodes_full(partial):
    return L._nodes_full(partial)


def _rung_nodes(e):
    """The node-7 4 h rung of E MWh: (0.25 E MVA, 1.0 E MWh); E = 0 is the empty candidate."""
    return _nodes_full({} if e == 0 else {UNIT_NODE: (e / DURATION_H, float(e))})


def _key_of(nodes):
    return H.candidate_key(H.canonical_candidate(nodes, investment_year=YEAR))


def _eval_key(nodes, m):
    return H.evaluation_key(_key_of(nodes), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            flex_price_multiplier=m)


def _m_tag(m):
    return L._m_tag(m)


_ORDINALS = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}


def _rung_label(e):
    return f'x0_m{_m_tag(M_FLEX)}' if e == 0 else f'n7_4h_e{e}_m{_m_tag(M_FLEX)}'


def points():
    """(label, nodes, m, instance) in launch order: ascending E."""
    return tuple((_rung_label(e), _rung_nodes(e), M_FLEX, f'e{e}') for e in NEW_RUNGS)


def waves(labels):
    return [labels[i:i + CONCURRENCY] for i in range(0, len(labels), CONCURRENCY)]


# ======================================================================================================================
#  pinned inputs (zero solves): the committed ladder (x = 0, E = 1) and marginal test (E = 2) rows at m = 2
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v21', SPEC_V21), ('ladder_results', LADDER_RESULTS), ('ladder_spec', LADDER_SPEC),
                      ('marginal_results', MARGINAL_RESULTS), ('marginal_spec', MARGINAL_SPEC),
                      ('ess_params_file', ESS_PARAMS_FILE), ('a0_spec', A0_SPEC),
                      ('investment_cost_results', INVESTMENT_COST_RESULTS), ('w25_results', W25_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == pin['sha256'] and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for name, pin in (('ladder_manifest', LADDER_MANIFEST), ('ladder_script', LADDER_SCRIPT),
                      ('marginal_manifest', MARGINAL_MANIFEST), ('marginal_script', MARGINAL_SCRIPT)):
        path = os.path.join(REPO, pin['path'])
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_on_disk': H.sha256_file(path) if os.path.isfile(path) else None,
                     'git_tracked': tracked, 'git_clean': not dirty, 'commit': pin.get('commit')}
        if not (os.path.isfile(path) and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for name, pin in PRIOR_W33.items():
        path = os.path.join(REPO, pin['path'])
        present = os.path.isfile(path)
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        payload = _load(pin['path']) if present else {}
        ok_flag = payload.get('all_ok') if name == 'checks' else payload.get('gate_pass')
        entry = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': in_head, 'git_tracked': tracked,
                 'sha256': H.sha256_file(path) if present else None, 'flag': ok_flag, 'expect': pin['expect']}
        out[f'w33_{name}'] = entry
        if not (present and tracked and in_head and ok_flag is True):
            failures.append(f'w33_{name}: {entry}')
    for name, commit in (('ladder_results_commit_in_HEAD', LADDER_RESULTS['commit']),
                         ('marginal_results_commit_in_HEAD', MARGINAL_RESULTS['commit'])):
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        out[name] = {'commit': commit, 'in_HEAD': in_head}
        if not in_head:
            failures.append(f'{name}: commit {commit} is not an ancestor of HEAD')
    return out, failures


def _cross_check_committed_point(checks, prefix, point, manifest, expected_e, expected_cycles):
    """The committed point's own evaluation_record.json must be present, tracked, hashed in its campaign manifest
    and agree with the campaign_results row on Q, the bar and the cycle count; the eval key must RECOMPUTE here."""
    checks[f'{prefix}_certified'] = point.get('status') == 'certified'
    checks[f'{prefix}_multiplier'] = point.get('flex_price_multiplier') == M_FLEX
    checks[f'{prefix}_cycles'] = point.get('cycles_run') == expected_cycles
    checks[f'{prefix}_eval_key_recomputes'] = point.get('eval_key') == _eval_key(_rung_nodes(expected_e), M_FLEX)
    checks[f'{prefix}_candidate_key_recomputes'] = point.get('candidate_key') == _key_of(_rung_nodes(expected_e))
    checks[f'{prefix}_bar_present'] = isinstance(point.get('bar'), float)
    checks[f'{prefix}_flex_cost_present'] = bool((point.get('flex_cost') or {}).get('per_dso'))
    rec_rel = os.path.join(point.get('eval_dir') or '', 'evaluation_record.json')
    present = bool(point.get('eval_dir')) and os.path.isfile(os.path.join(REPO, rec_rel))
    checks[f'{prefix}_record_present'] = present
    if present:
        rec = _load(rec_rel)
        checks[f'{prefix}_record_sha256_in_campaign_manifest'] = (
            manifest.get(rec_rel) == H.sha256_file(os.path.join(REPO, rec_rel)))
        checks[f'{prefix}_record_Q_matches'] = rec.get('certified_cost') == point.get('Q')
        checks[f'{prefix}_record_bar_matches'] = (rec.get('bar') or {}).get('value') == point.get('bar')
        checks[f'{prefix}_record_cycles_matches'] = rec.get('cycles_run') == point.get('cycles_run')
        checks[f'{prefix}_record_tracked'] = bool(H._git(['ls-files', '--', rec_rel]).strip())
    traj = (point.get('per_cycle_trajectory') or {})
    checks[f'{prefix}_trajectory_hash_in_campaign_manifest'] = (
        bool(traj.get('path')) and manifest.get(traj['path']) == traj.get('sha256'))


def pinned_rows():
    """Q_m(0), bar_x0 and the committed E = 1 / E = 2 rows at m = 2, from the two committed campaign results
    (each pinned by sha256), cross-checked against their own records and manifests. ZERO SOLVES."""
    failures, checks = [], {}
    ladder = _load(LADDER_RESULTS['path'])
    ladder_manifest = _load(LADDER_MANIFEST['path'])
    marginal = _load(MARGINAL_RESULTS['path'])
    marginal_manifest = _load(MARGINAL_MANIFEST['path'])
    checks['ladder_spec_sha256_matches_results'] = ladder.get('campaign_spec_sha256') == LADDER_SPEC['sha256']
    checks['ladder_no_stop_for_review'] = ladder.get('STOP_FOR_REVIEW') is False
    checks['ladder_no_non_certified_points'] = ladder.get('non_certified_points') == []
    checks['ladder_no_readback_mismatch'] = ladder.get('readback_mismatch_points') == []
    checks['ladder_no_harness_errors'] = ladder.get('harness_errors') == []
    checks['marginal_spec_sha256_matches_results'] = marginal.get('campaign_spec_sha256') == MARGINAL_SPEC['sha256']
    checks['marginal_no_stop_for_review'] = marginal.get('STOP_FOR_REVIEW') is False
    checks['marginal_no_non_certified_points'] = marginal.get('non_certified_points') == []
    checks['marginal_no_readback_mismatch'] = marginal.get('readback_mismatch_points') == []
    checks['marginal_no_harness_errors'] = marginal.get('harness_errors') == []

    x0 = (ladder['points'] or {}).get(_rung_label(0)) or {}
    committed = {0: {'point': x0, 'source': f"committed flexibility-price ladder ({LADDER_RESULTS['commit']})",
                     'results_path': LADDER_RESULTS['path'], 'campaign_row': None},
                 1: {'point': (ladder['points'] or {}).get(_rung_label(1)) or {},
                     'source': f"committed flexibility-price ladder ({LADDER_RESULTS['commit']})",
                     'results_path': LADDER_RESULTS['path'],
                     'campaign_row': (ladder.get('per_m') or {}).get(f'{M_FLEX:g}') or {}},
                 2: {'point': (marginal['points'] or {}).get(_rung_label(2)) or {},
                     'source': f"committed marginal-MWh test ({MARGINAL_RESULTS['commit']})",
                     'results_path': MARGINAL_RESULTS['path'],
                     'campaign_row': (marginal.get('per_m') or {}).get(f'{M_FLEX:g}') or {}}}
    _cross_check_committed_point(checks, 'pinned_x0', x0, ladder_manifest, 0, CYCLES_PINNED_EXPECTED[0])
    _cross_check_committed_point(checks, 'pinned_e1', committed[1]['point'], ladder_manifest, 1,
                                 CYCLES_PINNED_EXPECTED[1])
    _cross_check_committed_point(checks, 'pinned_e2', committed[2]['point'], marginal_manifest, 2,
                                 CYCLES_PINNED_EXPECTED[2])
    for e in (0, 1, 2):
        checks[f'pinned_e{e}_Q_expected'] = committed[e]['point'].get('Q') == Q_PINNED_EXPECTED[e]
    checks['pinned_x0_bar_expected'] = x0.get('bar') == BAR_X0_M2_EXPECTED
    # The marginal test's x0 is the SAME pinned ladder x0 -- the two committed stages must agree.
    checks['marginal_pinned_the_same_x0'] = (
        ((marginal.get('pinned_ladder') or {}).get('per_m') or {}).get(f'{M_FLEX:g}', {}).get('x0', {}).get('Q')
        == x0.get('Q'))
    failures += [f'pinned row check failed: {k}' for k, v in checks.items() if not v]
    return {'sources': {'ladder': dict(LADDER_RESULTS), 'marginal': dict(MARGINAL_RESULTS)},
            'x0': {k: x0.get(k) for k in ('status', 'cycles_run', 'certification_cycle', 'Q', 'bar', 'flex_cost',
                                          'rule_ten', 'eval_dir', 'eval_key', 'candidate_key',
                                          'flex_price_multiplier', 'per_cycle_trajectory')},
            'committed': {str(e): {'source': committed[e]['source'], 'results_path': committed[e]['results_path'],
                                   'campaign_row': committed[e]['campaign_row'],
                                   'point': {k: committed[e]['point'].get(k)
                                             for k in ('status', 'cycles_run', 'certification_cycle', 'Q', 'bar',
                                                       'flex_cost', 'rule_ten', 'throughput', 'eval_dir', 'eval_key',
                                                       'candidate_key', 'flex_price_multiplier',
                                                       'per_cycle_trajectory', 'solve_reconciliation')}}
                          for e in (1, 2)},
            'checks': checks}, failures


def investment_inputs():
    """I(E) for every rung matched BY CANDIDATE KEY, the 2025 unit costs, the budget, the master capacity / ratio
    constraints and the budget corner -- all from the pinned W2 table (M.budget_corner BY IMPORT). ZERO SOLVES."""
    failures, checks = [], {}
    w2 = _load(INVESTMENT_COST_RESULTS['path'])
    table = w2['candidates']
    i_by_rung, keys = {}, {}
    for e in RUNGS:
        key = _key_of(_rung_nodes(e))
        keys[str(e)] = key
        vals = sorted({c.get('I_new_eur') for c in table.values() if c.get('candidate_key') == key}, key=repr)
        checks[f'I_e{e}_unique_in_w2_table'] = len(vals) == 1 and vals[0] is not None
        i_by_rung[str(e)] = vals[0] if len(vals) == 1 else None
    checks['I_e1_is_317957_0085035586'] = i_by_rung['1'] == I_PER_MWH_4H_EXPECTED
    checks['I_e2_is_635914_0170071172'] = i_by_rung['2'] == M.I_UNIT2_EXPECTED
    delta_i = {str(e): (i_by_rung[str(e)] - i_by_rung[str(e - 1)]) if e > 1 else i_by_rung['1'] for e in RUNGS}
    checks['I_linear_in_energy_at_4h'] = all(
        abs(delta_i[str(e)] - I_PER_MWH_4H_EXPECTED) <= I_LINEARITY_TOL_EUR for e in RUNGS)
    unit_costs = w2['expected_unit_costs_per_case_year']['new'][str(YEAR)]
    energy_cost = unit_costs['energy_eur_per_mwh_discounted']
    checks['energy_unit_cost_2025_is_253877_6774385931'] = energy_cost == ENERGY_UNIT_COST_2025_EXPECTED
    master = w2['master_facts']
    checks['budget_is_1e6'] = master['budget_eur'] == BUDGET_EUR_EXPECTED
    checks['max_capacity_is_5_mwh'] = master['max_capacity_mwh'] == MAX_CAPACITY_MWH_EXPECTED
    checks['ratio_bounds_are_2_to_10'] = (master['min_energy_to_power_ratio'],
                                          master['max_energy_to_power_ratio']) == ENERGY_TO_POWER_RATIO_BOUNDS_EXPECTED
    checks['every_rung_within_max_capacity'] = max(RUNGS) <= master['max_capacity_mwh']
    checks['ladder_duration_within_ratio_bounds'] = (
        master['min_energy_to_power_ratio'] <= DURATION_H <= master['max_energy_to_power_ratio'])
    corner, more = M.budget_corner(w2)
    failures += more
    # M.budget_corner's own `statement` is worded for the marginal-MWh stage ("this campaign evaluates 2 MWh ...
    # the corner itself is NOT evaluated here"), which is false here: F2 evaluates the corner (E = 3). The
    # inherited wording is kept, clearly attributed, and the stage's own statement replaces it.
    corner['statement_as_worded_by_the_s50_marginal_launcher'] = corner['statement']
    corner['statement'] = (f"under the EUR {corner['budget_eur']:,.0f} budget the node-7 4 h ladder at {YEAR} is "
                           f"feasible up to {corner['corner_e_mwh']} MWh (I = {corner['corner_I_eur']}, slack "
                           f"{corner['corner_slack_eur']}); the next rung is infeasible. The BUDGET CORNER is that "
                           f"point, E = 3 MWh, and THIS campaign evaluates it, together with E = 4 and E = 5 (both "
                           f"budget-infeasible, evaluated to trace the ladder up to the max_capacity cap of "
                           f"{w2['master_facts']['max_capacity_mwh']} MWh). E = 1 and E = 2 are committed "
                           f"({LADDER_RESULTS['commit']}, {MARGINAL_RESULTS['commit']}) and are not re-run.")
    checks['budget_corner_is_3_mwh'] = corner['corner_e_mwh'] == BUDGET_CORNER_E_MWH_EXPECTED
    by_e = {p['e_mwh']: p for p in corner['ladder']}
    checks['budget_ladder_I_matches_by_key'] = all(by_e[float(e)]['I_eur'] == i_by_rung[str(e)] for e in RUNGS
                                                   if float(e) in by_e)
    failures += [f'investment input check failed: {k}' for k, v in checks.items() if not v]
    return {'I_by_rung_eur': i_by_rung, 'delta_I_by_rung_eur': delta_i,
            'I_per_mwh_energy_at_4h_eur': I_PER_MWH_4H_EXPECTED, 'I_source': dict(INVESTMENT_COST_RESULTS),
            'candidate_keys': keys, 'energy_unit_cost_2025_eur_per_mwh': energy_cost,
            'power_unit_cost_2025_eur_per_mva': unit_costs['power_eur_per_mva_discounted'],
            'budget_eur': master['budget_eur'], 'max_capacity_mwh': master['max_capacity_mwh'],
            'energy_to_power_ratio_bounds': [master['min_energy_to_power_ratio'],
                                             master['max_energy_to_power_ratio']],
            'budget_corner': corner, 'budget_definition': BUDGET_DEFINITION, 'checks': checks}, failures


# ======================================================================================================================
#  the ladder table (one row per rung)
# ======================================================================================================================
def rung_row(e, x0, point, investment, prev_row, source):
    """L.per_m_row for this rung against the PINNED x = 0, plus the per-MWh figures, the true marginal MWh against
    the previous rung, and the budget figures. `point` is a point-result dict (this campaign's or a committed one)."""
    i_e = investment['I_by_rung_eur'][str(e)]
    row = L.per_m_row(M_FLEX, x0, point, i_e)
    row['e_mwh'] = float(e)
    row['s_mva'] = e / DURATION_H
    row['label'] = _rung_label(e)
    row['source'] = source
    row['candidate_key'] = investment['candidate_keys'][str(e)]
    row['eval_key'] = point.get('eval_key')
    row['cycles_run'] = point.get('cycles_run')
    row['certification_cycle'] = point.get('certification_cycle')
    row['per_cycle_trajectory'] = point.get('per_cycle_trajectory')
    value = row['value_eur']
    row['value_per_mwh_energy_eur'] = (value / float(e)) if value is not None else None
    row['I_per_mwh_energy_eur'] = i_e / float(e)
    row['energy_unit_cost_2025_eur_per_mwh'] = investment['energy_unit_cost_2025_eur_per_mwh']
    row['resolution_per_mwh_energy_eur'] = ((row['resolution_eur'] / float(e))
                                            if row['resolution_eur'] is not None else None)
    # the TRUE MARGINAL MWh: this rung minus the previous one (E = 1's "previous rung" is x = 0, i.e. the rung
    # itself, so its marginal figures are its own value - I row).
    prev_value = prev_row.get('value_eur') if prev_row else 0.0
    prev_res = prev_row.get('resolution_eur') if prev_row else 0.0
    d_value = (value - prev_value) if (value is not None and prev_value is not None) else None
    d_res = ((row['resolution_eur'] + prev_res)
             if (row['resolution_eur'] is not None and prev_res is not None) else None)
    d_i = investment['delta_I_by_rung_eur'][str(e)]
    if d_value is None or d_res is None:
        reading = 'not available (this rung or the previous one is not certified)'
    elif abs(d_value - d_i) <= d_res:
        reading = 'INDETERMINATE (|delta value - delta I| <= resolution of the difference)'
    else:
        ordinal = _ORDINALS.get(e, f'{e}th')
        reading = (f'the {ordinal} MWh pays' if d_value - d_i > 0 else f'the {ordinal} MWh does not pay')
    row['marginal_mwh'] = {
        'definition': ('the E-th MWh = this rung minus the previous rung at the same m (for E = 1 the previous rung '
                       'is x = 0, so value_eur and resolution_eur are its own); resolution of the difference = '
                       'resolution_E + resolution_{E-1} (both are differences of iteratively-computed quantities, '
                       'so the bars add)'),
        'previous_rung_e_mwh': float(e - 1), 'delta_value_eur': d_value, 'delta_I_eur': d_i,
        'delta_value_minus_delta_I_eur': (d_value - d_i) if d_value is not None else None,
        'resolution_of_difference_eur': d_res,
        'delta_value_minus_delta_I_over_resolution': ((d_value - d_i) / d_res)
        if (d_value is not None and d_res) else None,
        'reading': reading,
        'delta_value_per_mwh_energy_eur': d_value,      # one MWh per rung
        'delta_I_per_mwh_energy_eur': d_i,
        'energy_unit_cost_2025_eur_per_mwh': investment['energy_unit_cost_2025_eur_per_mwh']}
    corner = investment['budget_corner']
    by_e = {p['e_mwh']: p for p in corner['ladder']}
    entry = by_e.get(float(e), {})
    row['budget'] = {'budget_eur': investment['budget_eur'], 'I_eur': i_e,
                     'slack_eur': entry.get('slack_eur'), 'budget_feasible': entry.get('budget_feasible'),
                     'w2_budget_feasible_new': entry.get('w2_budget_feasible_new'),
                     'is_budget_corner': float(e) == corner['corner_e_mwh'],
                     'max_capacity_mwh': investment['max_capacity_mwh'],
                     'at_max_capacity_cap': float(e) == investment['max_capacity_mwh']}
    row['value_minus_I_if_budget_feasible'] = (row['value_minus_I_eur'] if entry.get('budget_feasible') else None)
    return row


def build_ladder(x0, by_rung, investment, rungs=RUNGS):
    """One row per rung, in ascending E (the marginal figures chain from the previous rung). The pinned rungs are
    recomputed here from their committed Q / bar; `_assert_pinned_rows_reproduce` then checks them bitwise against
    the committed rows."""
    rows, prev = {}, None
    for e in rungs:
        if e in PINNED_RUNGS:
            src = by_rung[str(e)]['point']
            source = by_rung[str(e)]['source']
        else:
            src = by_rung[str(e)]
            source = 'this campaign'
        row = rung_row(e, x0, src, investment, prev, source)
        rows[str(e)] = row
        prev = row
    return rows


def _assert_pinned_rows_reproduce(rows, pinned):
    """The recomputed pinned rows must equal the committed ones bitwise on value, value - I and the resolution."""
    checks = {}
    for e in PINNED_RUNGS:
        committed = pinned['committed'][str(e)]['campaign_row'] or {}
        row = rows[str(e)]
        checks[f'pinned_e{e}_value_reproduces'] = row['value_eur'] == committed.get('value_eur')
        checks[f'pinned_e{e}_value_minus_I_reproduces'] = row['value_minus_I_eur'] == committed.get('value_minus_I_eur')
        checks[f'pinned_e{e}_resolution_reproduces'] = row['resolution_eur'] == committed.get('resolution_eur')
        checks[f'pinned_e{e}_T_reproduces'] = (row['T_cell_throughput_full_cycle_mwh']
                                               == committed.get('T_cell_throughput_full_cycle_mwh'))
        checks[f'pinned_e{e}_reading_reproduces'] = row['reading'] == committed.get('reading')
    # the committed marginal-MWh figure of E = 2 (0e36ec0a) must be the E = 2 row's marginal figure here
    committed_second = ((pinned['committed']['2']['campaign_row'] or {}).get('second_mwh') or {})
    marg = rows['2']['marginal_mwh']
    checks['committed_second_mwh_delta_value_reproduces'] = (marg['delta_value_eur']
                                                             == committed_second.get('delta_value_eur'))
    checks['committed_second_mwh_resolution_reproduces'] = (marg['resolution_of_difference_eur']
                                                            == committed_second.get('resolution_of_difference_eur'))
    return checks


# ======================================================================================================================
#  memory preflight (A0's measure, restated at this campaign's concurrency)
# ======================================================================================================================
def memory_preflight():
    """L.memory_preflight's vm_stat measure (concurrency-independent) with THIS campaign's threshold (3 children)."""
    out = L.memory_preflight()
    out.update({'required_bytes': MEMORY_REQUIRED_BYTES, 'required_gib': MEMORY_REQUIRED_BYTES / GIB,
                'concurrency': CONCURRENCY, 'rule': MEMORY_RULE})
    if out.get('available_bytes') is not None:
        out['free_plus_inactive_would_pass_same_threshold'] = out['free_plus_inactive_bytes'] >= MEMORY_REQUIRED_BYTES
        out['pass'] = out['available_bytes'] >= MEMORY_REQUIRED_BYTES
    else:
        out['pass'] = False
    return out


# ======================================================================================================================
#  rule eleven -- every quantity the frozen specification requires must have a capture path BEFORE the run
# ======================================================================================================================
CAMPAIGN_RESULT_FIELDS = (
    'status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'Q', 'bar', 'rule_ten', 'flex_price_multiplier',
    'flex_price_label', 'flex_price_readback', 'flex_cost', 'throughput', 'solve_reconciliation', 'wall_time',
    'peak_rss', 'aa_action_counts', 'per_cycle_trajectory',
    'ladder.value_eur', 'ladder.I_eur', 'ladder.value_minus_I_eur', 'ladder.resolution_eur', 'ladder.reading',
    'ladder.value_per_mwh_energy_eur', 'ladder.value_per_full_cycle_eur_per_mwh',
    'ladder.T_cell_throughput_full_cycle_mwh', 'ladder.marginal_mwh', 'ladder.budget',
    'ladder.per_cycle_trajectory', 'budget_corner')


def rule_eleven(pinned, investment):
    """The run-side capture paths (L.rule_eleven, the ladder launcher's checklist over the harness sources) plus the
    PINNED ones, which must already resolve to a value in the committed artifacts."""
    run_side = L.rule_eleven()
    checks = dict(run_side['checks'])
    checks['pinned_x0_Q'] = isinstance(pinned['x0'].get('Q'), float)
    checks['pinned_x0_bar'] = isinstance(pinned['x0'].get('bar'), float)
    checks['pinned_x0_flex_cost'] = bool((pinned['x0'].get('flex_cost') or {}).get('per_dso'))
    checks['pinned_x0_trajectory'] = bool((pinned['x0'].get('per_cycle_trajectory') or {}).get('path'))
    for e in PINNED_RUNGS:
        pin = pinned['committed'][str(e)]
        checks[f'pinned_e{e}_Q'] = isinstance(pin['point'].get('Q'), float)
        checks[f'pinned_e{e}_bar'] = isinstance(pin['point'].get('bar'), float)
        checks[f'pinned_e{e}_cycles'] = isinstance(pin['point'].get('cycles_run'), int)
        checks[f'pinned_e{e}_flex_cost'] = bool((pin['point'].get('flex_cost') or {}).get('per_dso'))
        checks[f'pinned_e{e}_T'] = isinstance((pin['point'].get('throughput') or {}).get(
            'T_cell_throughput_full_cycle_mwh'), float)
        checks[f'pinned_e{e}_trajectory'] = bool((pin['point'].get('per_cycle_trajectory') or {}).get('path'))
        checks[f'pinned_e{e}_campaign_row_value'] = isinstance((pin['campaign_row'] or {}).get('value_eur'), float)
        checks[f'pinned_e{e}_campaign_row_resolution'] = isinstance((pin['campaign_row'] or {}).get('resolution_eur'),
                                                                    float)
    for e in RUNGS:
        checks[f'I_e{e}_available'] = isinstance(investment['I_by_rung_eur'][str(e)], float)
        checks[f'delta_I_e{e}_available'] = isinstance(investment['delta_I_by_rung_eur'][str(e)], float)
    checks['energy_unit_cost_available'] = isinstance(investment['energy_unit_cost_2025_eur_per_mwh'], float)
    checks['budget_available'] = isinstance(investment['budget_eur'], float)
    checks['budget_corner_available'] = investment['budget_corner']['corner_e_mwh'] is not None
    checks['budget_slack_available_for_every_rung'] = all(
        isinstance(p.get('slack_eur'), float) for p in investment['budget_corner']['ladder'])
    checks['throughput_capture_for_new_cells'] = callable(getattr(L, 'unit_throughput_w25', None))
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s51 F2 ladder campaign_results): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': run_side['harness_record_capture_checklist']}


# ======================================================================================================================
#  checks shared by --freeze and --run
# ======================================================================================================================
def _common_checks():
    failures, evidence = [], {}
    case_file = L._check_case_file_loads_to_declaration()
    evidence['case_file_aa'] = case_file
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    loaded = H.ess_ageing_canonical_text(H.load_ess_ageing_parameters(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)))
    evidence['ess_file_loads_to_declaration'] = loaded == H.ess_ageing_canonical_text(ESS_AGEING_BASELINE)
    if not evidence['ess_file_loads_to_declaration']:
        failures.append('ESS parameters file does not load to the declaration')
    shared = {'label': L.LABEL == LABEL, 'case_file_aa': L.CASE_FILE_AA == CASE_FILE_AA,
              'ess_ageing_baseline': L.ESS_AGEING_BASELINE == ESS_AGEING_BASELINE, 'year': L.YEAR == YEAR,
              'active_nodes': L.ACTIVE_NODES == ACTIVE_NODES, 'flex_label': L.FLEX_LABEL == FLEX_LABEL,
              'cap': L.CAP == CAP, 'required_consecutive_cycles': L.REQUIRED_CONSECUTIVE_CYCLES
              == REQUIRED_CONSECUTIVE_CYCLES, 'per_child_memory_budget': L.MEMORY_PER_CHILD_BUDGET_BYTES
              == MEMORY_PER_CHILD_BUDGET_BYTES, 'marginal_cap': M.CAP == CAP,
              'marginal_label': M.LABEL == LABEL, 'marginal_year': M.YEAR == YEAR,
              'marginal_unit2_is_rung_2': M._nodes_full({M.UNIT_NODE: M.UNIT2}) == _rung_nodes(2),
              'marginal_unit1_is_rung_1': M._nodes_full({M.UNIT_NODE: M.UNIT1}) == _rung_nodes(1),
              'ladder_unit_is_rung_1': L._nodes_full({L.UNIT_NODE: L.UNIT}) == _rung_nodes(1),
              'm_is_a_ladder_multiplier': M_FLEX in L.MULTIPLIERS and M_FLEX in M.MULTIPLIERS}
    evidence['shared_constants_with_prior_stages'] = shared
    failures += [f'constant differs from a prior stage launcher: {k}' for k, v in shared.items() if not v]
    pins, more = _check_pins()
    evidence['pins'] = pins
    failures += more
    memory_rule = L._memory_rule_matches_a0()
    evidence['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f"memory preflight rule differs from A0's measure: {memory_rule}")
    pinned, more = pinned_rows()
    evidence['pinned'] = pinned
    failures += more
    investment, more = investment_inputs()
    evidence['investment'] = investment
    failures += more
    # W25 reproduction (the T definition this stage uses), exactly as the ladder checks it
    refs, more = L.reference_inputs()
    evidence['w25_reproduction'] = {'checks': {k: v for k, v in refs['checks'].items() if k.startswith('w25')},
                                    'w25_committed': refs['w25_committed']}
    failures += [f'W25/reference check failed: {k}' for k, v in refs['checks'].items() if not v]
    per_cycle = L.solves_per_cycle_from_case_file()
    evidence['solves_per_cycle_from_case_file'] = per_cycle
    if per_cycle != 51:
        failures.append(f'solves per cycle from the case file {per_cycle} != 51 (the W10/W32/W33 gates)')
    pts = points()
    labels = [p[0] for p in pts]
    keys = [_eval_key(n, m) for _l, n, m, _i in pts]
    pinned_keys = {pinned['x0'].get('eval_key')} | {pinned['committed'][str(e)]['point'].get('eval_key')
                                                    for e in PINNED_RUNGS}
    if len(pts) != 3 or len(set(labels)) != 3 or len(set(keys)) != 3 or set(keys) & pinned_keys:
        failures.append('points: expected 3 unique labels / eval keys, all distinct from the pinned keys')
    evidence['points'] = [{'label': label, 'instance': inst, 'flex_price_multiplier': m,
                           'nodes': {str(n): list(v) for n, v in nodes.items()}, 'investment_year': YEAR,
                           'candidate_key': _key_of(nodes), 'eval_key': _eval_key(nodes, m),
                           'energy_mwh': float(int(inst[1:])), 's_mva': int(inst[1:]) / DURATION_H}
                          for label, nodes, m, inst in pts]
    evidence['waves'] = waves(labels)
    spec21 = _load(SPEC_V21['path'])
    evidence['spec_v21_item'] = spec21[SPEC_V21['item']]
    # The pinned rungs are recomputed here (zero solves) and must reproduce their committed rows bitwise, BEFORE
    # any evaluation is launched.
    if all(pinned['checks'].values()) and all(investment['checks'].values()):
        pinned_rows_recomputed = build_ladder(pinned['x0'], pinned['committed'], investment, rungs=PINNED_RUNGS)
        reproduction = _assert_pinned_rows_reproduce(pinned_rows_recomputed, pinned)
        evidence['pinned_rows_reproduce_committed'] = reproduction
        evidence['pinned_rows_recomputed'] = pinned_rows_recomputed
        failures += [f'pinned row does not reproduce the committed one: {k}'
                     for k, v in reproduction.items() if not v]
    try:
        evidence['rule_eleven'] = rule_eleven(pinned, investment)
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, pts


def _validate_spec(spec, pts, evidence):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    extra = spec.get('extra') or {}
    cfg = spec['configuration']
    checks = {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'entries_in_order': [e['label'] for e in entries] == [p[0] for p in pts],
        'n_entries_3': len(entries) == 3,
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': ((cfg.get('ess_params_file') or {}).get('path') == ESS_PARAMS_FILE['path']
                                   and (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256']),
        'no_campaign_overrides': cfg.get('overrides') == {},
        'no_ageing_model_variant_anywhere': ('model_variant_label' not in spec
                                             and not any('model_variant' in e for e in entries)),
        'flex_label_at_spec_level': spec.get('flex_price_label') == FLEX_LABEL,
        'scenario_label_recorded': extra.get('scenario_label') == SCENARIO_LABEL,
        'cap': spec.get('cap') == CAP,
        'concurrency_3': spec.get('concurrency') == CONCURRENCY == 3,
        'one_wave': extra.get('waves') == [[p[0] for p in pts]],
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': cfg.get('arm_label') == 's39_D',
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'post_certification_none': all(e.get('post_certification') is None for e in entries),
        'labels_recorded': extra.get('label') == LABEL and extra.get('flex_label') == FLEX_LABEL,
        'spec_v21_recorded': extra.get('spec_v21') == SPEC_V21,
        'predictions_recorded': (extra.get('predictions_recorded_before_run')
                                 == evidence['spec_v21_item'].get('predictions_recorded_before_run')),
        'stop_rule_recorded': extra.get('stop_rule') == STOP_RULE,
        'pinned_x0_frozen': ((extra.get('pinned') or {}).get('x0') or {}).get('Q') == Q_PINNED_EXPECTED[0],
        'pinned_e1_frozen': (((extra.get('pinned') or {}).get('committed') or {}).get('1') or {}).get(
            'point', {}).get('Q') == Q_PINNED_EXPECTED[1],
        'pinned_e2_frozen': (((extra.get('pinned') or {}).get('committed') or {}).get('2') or {}).get(
            'point', {}).get('Q') == Q_PINNED_EXPECTED[2],
        'pinned_sources_recorded': ((extra.get('pinned') or {}).get('sources')
                                    == {'ladder': dict(LADDER_RESULTS), 'marginal': dict(MARGINAL_RESULTS)}),
        'pinned_rows_reproduce_committed': (bool(extra.get('pinned_rows_reproduce_committed'))
                                            and all(extra['pinned_rows_reproduce_committed'].values())),
        'I_by_rung_frozen': extra.get('I_by_rung_eur') == evidence['investment']['I_by_rung_eur'],
        'budget_corner_frozen': ((extra.get('budget_corner') or {}).get('corner_e_mwh')
                                 == BUDGET_CORNER_E_MWH_EXPECTED),
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'value_definition_recorded': extra.get('value_definition') == VALUE_DEFINITION,
        'memory_rule_recorded': extra.get('memory_preflight_rule') == MEMORY_RULE,
    }
    for label, nodes, m, _inst in pts:
        entry = by_label.get(label) or {}
        canon = H.canonical_candidate(nodes, investment_year=YEAR)
        checks[f'{label}:canonical_key'] = entry.get('canonical') == canon and entry.get('key') == H.candidate_key(canon)
        checks[f'{label}:no_overrides'] = entry.get('overrides') == {}
        checks[f'{label}:effective_aa_is_declaration'] = entry.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:flex_multiplier'] = entry.get('flex_price_multiplier') == m
        checks[f'{label}:flex_label'] = entry.get('flex_price_label') == FLEX_LABEL
        checks[f'{label}:eval_key_recomputes'] = entry.get('eval_key') == _eval_key(nodes, m)
        checks[f'{label}:eval_key_not_the_baseline_key'] = entry.get('eval_key') != _eval_key(nodes, None)
    return checks


def _guard_verifications():
    return {'parent': PARENT_GUARD.verify(0), 'marginal_parent': M.PARENT_GUARD.verify(0),
            'ladder_parent': L.PARENT_GUARD.verify(0), 'w25_module': L._w25().GUARD.verify(0)}


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(started):
    tag = 'S51-F2'
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, pts = _common_checks()
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    pinned, investment = evidence['pinned'], evidence['investment']
    extra = {'campaign_script': os.path.basename(__file__),
             'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'ladder_script': dict(LADDER_SCRIPT, sha256=H.sha256_file(os.path.join(REPO, LADDER_SCRIPT['path']))),
             'marginal_script': dict(MARGINAL_SCRIPT,
                                     sha256=H.sha256_file(os.path.join(REPO, MARGINAL_SCRIPT['path']))),
             'label': LABEL, 'flex_label': FLEX_LABEL, 'scenario_label': SCENARIO_LABEL, 'mode': 'full',
             'spec_v21': dict(SPEC_V21), 'spec_v21_item': evidence['spec_v21_item'],
             'predictions_recorded_before_run': evidence['spec_v21_item'].get('predictions_recorded_before_run'),
             'ess_params_file': dict(ESS_PARAMS_FILE), 'a0_spec': dict(A0_SPEC), 'pins': evidence['pins'],
             'flex_price_multiplier': M_FLEX, 'rungs_mwh': list(RUNGS), 'pinned_rungs_mwh': list(PINNED_RUNGS),
             'evaluated_rungs_mwh': list(NEW_RUNGS), 'duration_h': DURATION_H,
             'points': evidence['points'], 'waves': evidence['waves'],
             'stop_rule': STOP_RULE, 'region_note': REGION_NOTE,
             'pinned': pinned,
             'I_by_rung_eur': investment['I_by_rung_eur'], 'delta_I_by_rung_eur': investment['delta_I_by_rung_eur'],
             'I_per_mwh_energy_at_4h_eur': investment['I_per_mwh_energy_at_4h_eur'],
             'I_source': investment['I_source'], 'candidate_keys': investment['candidate_keys'],
             'energy_unit_cost_2025_eur_per_mwh': investment['energy_unit_cost_2025_eur_per_mwh'],
             'power_unit_cost_2025_eur_per_mva': investment['power_unit_cost_2025_eur_per_mva'],
             'budget_eur': investment['budget_eur'], 'max_capacity_mwh': investment['max_capacity_mwh'],
             'energy_to_power_ratio_bounds': investment['energy_to_power_ratio_bounds'],
             'budget_corner': investment['budget_corner'], 'budget_definition': BUDGET_DEFINITION,
             'investment_checks': investment['checks'],
             'pinned_rows_recomputed': evidence.get('pinned_rows_recomputed'),
             'pinned_rows_reproduce_committed': evidence.get('pinned_rows_reproduce_committed'),
             'w25_reproduction': evidence['w25_reproduction'],
             'shared_constants_with_prior_stages': evidence['shared_constants_with_prior_stages'],
             'solves_per_cycle_from_case_file': evidence['solves_per_cycle_from_case_file'],
             'post_certification': 'none', 'model_variant': 'none (ageing); flexibility-price multiplier per entry',
             'objective_convention': OBJECTIVE_CONVENTION, 'value_definition': VALUE_DEFINITION,
             'flex_cost_definition': L.FLEX_COST_DEFINITION,
             'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                'gross_operational_cost[k-1]| (harness W5)'),
             'memory_preflight_rule': MEMORY_RULE, 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
             'memory_preflight_refusing_at': '--run (non-gating at --freeze)', 'memory_at_freeze_non_gating': memory,
             'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID,
        [(label, nodes, {'investment_year': YEAR, 'flex_price_multiplier': m}) for label, nodes, m, _i in pts],
        configuration={'name': (f'F2 demonstration case -- {FLEX_LABEL} (m = {M_FLEX:g}) under {LABEL}: the node-7 '
                                f'4 h capacity ladder at {YEAR}, rungs E = {list(NEW_RUNGS)} MWh (E = 1, 2 and x = 0 '
                                'pinned from committed campaigns); the case file (AA keep_memory in '
                                'data/SRP1/SRP1_params.json) with the ESS ageing parameters declared'),
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('no overrides; no ageing model variant; no post-certification; the flexibility-price '
                                'multiplier is applied per entry in the child configuration hook and read back from '
                                'probe DSO blocks (pre-run) and the run\'s own DSO models (post-run); num_max_iters '
                                ':= cap; x = 0, E = 1 and E = 2 at m = 2 are NOT evaluated -- they are pinned from '
                                'the committed ladder (dc468ab4) and marginal test (0e36ec0a)')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=extra)
    checks = _validate_spec(spec, pts, evidence)
    guards = _guard_verifications()
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] {FLEX_LABEL} (m = {M_FLEX:g}) under {LABEL} -- F2 node-7 4 h ladder at {YEAR}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for entry in spec['candidates']:
        _log(f"[{tag}]   {entry['label']}: m={entry['flex_price_multiplier']} key={entry['key'][:16]} "
             f"eval_key={entry['eval_key'][:16]} eval_dir={entry['eval_dir']}")
    _log(f"[{tag}] pinned x = 0 (m = {M_FLEX:g}): Q(0)={pinned['x0']['Q']} bar={pinned['x0']['bar']} "
         f"cycles={pinned['x0']['cycles_run']} eval_key={(pinned['x0']['eval_key'] or '')[:16]} "
         f"source={LADDER_RESULTS['commit']}")
    for e in PINNED_RUNGS:
        pin = pinned['committed'][str(e)]
        row = pin['campaign_row'] or {}
        _log(f"[{tag}] pinned E = {e} MWh: Q={pin['point']['Q']} bar={pin['point']['bar']} "
             f"cycles={pin['point']['cycles_run']} value={row.get('value_eur')} "
             f"value-I={row.get('value_minus_I_eur')} resolution={row.get('resolution_eur')} "
             f"T={row.get('T_cell_throughput_full_cycle_mwh')} reading={row.get('reading')}; {pin['source']}")
    _log(f"[{tag}] I by rung (W2, by candidate key): {investment['I_by_rung_eur']}; delta I per MWh: "
         f"{investment['delta_I_by_rung_eur']} (expected {I_PER_MWH_4H_EXPECTED} at 4 h)")
    _log(f"[{tag}] 2025 unit costs: energy {investment['energy_unit_cost_2025_eur_per_mwh']} EUR/MWh, power "
         f"{investment['power_unit_cost_2025_eur_per_mva']} EUR/MVA; budget {investment['budget_eur']}; "
         f"max capacity {investment['max_capacity_mwh']} MWh; ratio bounds "
         f"{investment['energy_to_power_ratio_bounds']}")
    _log(f"[{tag}] BUDGET CORNER: {investment['budget_corner']['statement']}")
    ladder_brief = [{k: p[k] for k in ('labels', 'e_mwh', 'I_eur', 'slack_eur', 'budget_feasible')}
                    for p in investment['budget_corner']['ladder']]
    _log(f'[{tag}] budget ladder: {ladder_brief}')
    _log(f"[{tag}] pinned checks: all={all(pinned['checks'].values())} "
         f"failing={[k for k, v in pinned['checks'].items() if not v]}")
    _log(f"[{tag}] investment checks: {investment['checks']}; budget-corner checks: "
         f"{investment['budget_corner']['checks']}")
    _log(f"[{tag}] W25 reproduction: {evidence['w25_reproduction']['checks']}")
    _log(f"[{tag}] pinned rows reproduce the committed ones: "
         f"{evidence.get('pinned_rows_reproduce_committed')}")
    _log(f"[{tag}] waves: {evidence['waves']}; stop rule: {STOP_RULE}")
    _log(f'[{tag}] region note: {REGION_NOTE}')
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] pins: {evidence['pins']}")
    _log(f"[{tag}] shared constants with prior stages: {evidence['shared_constants_with_prior_stages']}")
    _log(f"[{tag}] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[{tag}] predictions recorded before the run: {extra['predictions_recorded_before_run']}")
    _log(f"[{tag}] memory at freeze (non-gating): {L._memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] guards: parent {PARENT_GUARD.counts} / marginal parent {M.PARENT_GUARD.counts} / ladder parent '
         f'{L.PARENT_GUARD.counts} / W25 module {L._w25().GUARD.counts}; verify0 failures={guards}; '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not any(guards.values())
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _point_result(label, rec, spec_point, per_cycle):
    """L._point_result (the ladder's own record reader). Its W25-throughput branch keys on instance == 'unit', so it
    is called with that instance and the real instance label is restored afterwards."""
    res = L._point_result(label, rec, dict(spec_point, instance='unit'), per_cycle)
    res['instance'] = spec_point['instance']
    res['energy_mwh'] = spec_point.get('energy_mwh')
    res['s_mva'] = spec_point.get('s_mva')
    return res


def stop_rule_state(records):
    return L.stop_rule_state(records)


def run(started, spec_sha256):
    tag = 'S51-F2'
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence, pts = _common_checks()
    failures += more
    checks = _validate_spec(spec, pts, evidence)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['configuration']['ess_params_file']['sha256'] != H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)):
        failures.append('ESS params file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    for name, pin in (('ladder', LADDER_SCRIPT), ('marginal', MARGINAL_SCRIPT)):
        if (spec['extra'].get(f'{name}_script') or {}).get('sha256') != H.sha256_file(
                os.path.join(REPO, pin['path'])):
            failures.append(f'the {name} launcher sha256 differs from the frozen spec')
    memory = memory_preflight()
    _log(f"[{tag}] memory preflight: {L._memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}; "
         f"vm_stat pages {memory['vm_stat_pages']}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {L._memory_line(memory)}')
    if failures:
        for failure in failures:
            _log(f'[{tag} PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    per_cycle = evidence['solves_per_cycle_from_case_file']
    pinned, investment = evidence['pinned'], evidence['investment']
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] {FLEX_LABEL} (m = {M_FLEX:g}) under {LABEL} -- F2 node-7 4 h ladder, evaluating E = '
         f'{list(NEW_RUNGS)} MWh')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    labels = [p[0] for p in pts]
    wave_list = waves(labels)
    _log(f'[{tag}] {len(labels)} points in {len(wave_list)} wave(s) {wave_list}; stop rule: {STOP_RULE}')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    records, wave_info, not_launched = [], [], []
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        for number, wave in enumerate(wave_list, start=1):
            state = stop_rule_state(records)
            if state['triggered']:
                not_launched += wave
                _log(f'[{tag}] STOP RULE: not launching wave {number} {wave}; {state["reasons"]}')
                wave_info.append({'wave': number, 'labels': wave, 'launched': False,
                                  'reason': f'stop rule: {state["reasons"]}'})
                continue
            _log(f'[{tag}] launching wave {number}/{len(wave_list)}: {wave} (concurrency {ctx.concurrency})')
            H.evaluate.last_batch_info = {}
            records += H.evaluate(wave, ctx)
            wave_info.append({'wave': number, 'labels': wave, 'launched': True,
                              'stop_rule_after_wave': stop_rule_state(records),
                              **getattr(H.evaluate, 'last_batch_info', {})})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    stop_state = stop_rule_state(records)
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    spec_points = {p['label']: p for p in spec['extra']['points']}
    point_results = {}
    for label in labels:
        if label in not_launched:
            point_results[label] = {'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'label': label,
                                    'instance': spec_points[label]['instance'],
                                    'flex_price_multiplier': spec_points[label]['flex_price_multiplier'],
                                    'status': 'not_launched_stop_rule'}
        else:
            point_results[label] = _point_result(label, by_label.get(label), spec_points[label], per_cycle)
    by_rung = {str(e): pinned['committed'][str(e)] for e in PINNED_RUNGS}
    for e in NEW_RUNGS:
        by_rung[str(e)] = point_results[_rung_label(e)]
    ladder_rows = build_ladder(pinned['x0'], by_rung, investment)
    pinned_reproduction = _assert_pinned_rows_reproduce(ladder_rows, pinned)
    launched = [label for label in labels if label not in not_launched]
    non_certified = [label for label in launched if point_results[label]['status'] != 'certified']
    harness_errors = [label for label in launched
                      if point_results[label]['status'] not in ('certified', 'not_certified')
                      or point_results[label]['exit_code'] != 0]
    readback_mismatch = [label for label in launched if point_results[label]['status'] in ('certified', 'not_certified')
                         and not (point_results[label]['flex_price_readback']['all_match']
                                  and point_results[label]['ess_ageing_readback_all_match']['pre_run'] is True
                                  and point_results[label]['ess_ageing_readback_all_match']['post_run'] is True)]
    solve_mismatch = [label for label in launched if point_results[label]['status'] in ('certified', 'not_certified')
                      and not point_results[label]['solve_reconciliation']['holds']]
    guards = _guard_verifications()
    table = [{'E_mwh': row['e_mwh'], 'S_mva': row['s_mva'], 'label': row['label'], 'source': row['source'],
              'status': row['status_unit'], 'cycles': row['cycles_run'],
              'Q_x0_pinned': row['Q_x0'], 'Q': row['Q_unit'], 'bar': row['bar_unit'],
              'value': row['value_eur'], 'I': row['I_eur'], 'value_minus_I': row['value_minus_I_eur'],
              'resolution': row['resolution_eur'], 'value_minus_I_over_resolution': row['value_minus_I_over_resolution'],
              'reading': row['reading'],
              'value_per_mwh_energy': row['value_per_mwh_energy_eur'],
              'I_per_mwh_energy': row['I_per_mwh_energy_eur'],
              'energy_unit_cost': row['energy_unit_cost_2025_eur_per_mwh'],
              'T_mwh': row['T_cell_throughput_full_cycle_mwh'],
              'value_per_full_cycle': row['value_per_full_cycle_eur_per_mwh'],
              'resolution_per_full_cycle': row['resolution_per_full_cycle_eur_per_mwh'],
              'rule_ten_gross': row['rule_ten_gross_ratio_unit'],
              'rule_ten_gross_x0_pinned': row['rule_ten_gross_ratio_x0'],
              'marginal_delta_value': row['marginal_mwh']['delta_value_eur'],
              'marginal_delta_I': row['marginal_mwh']['delta_I_eur'],
              'marginal_delta_value_minus_delta_I': row['marginal_mwh']['delta_value_minus_delta_I_eur'],
              'marginal_resolution': row['marginal_mwh']['resolution_of_difference_eur'],
              'marginal_reading': row['marginal_mwh']['reading'],
              'I_budget_slack': row['budget']['slack_eur'], 'budget_feasible': row['budget']['budget_feasible'],
              'is_budget_corner': row['budget']['is_budget_corner'],
              'at_max_capacity_cap': row['budget']['at_max_capacity_cap'],
              'dso_flex_saving_total': row['dso_flexibility_saving_total_weighted_r2_eur'],
              'per_cycle_trajectory_path': (row['per_cycle_trajectory'] or {}).get('path')}
             for row in (ladder_rows[str(e)] for e in RUNGS)]
    results = {
        'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'SCENARIO_LABEL': SCENARIO_LABEL,
        'STOP_FOR_REVIEW': stop_state['triggered'], 'stop_rule_state': stop_state, 'stop_rule': STOP_RULE,
        'region_note': REGION_NOTE,
        'non_certified_points': non_certified, 'not_launched_points': not_launched, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch, 'solve_reconciliation_mismatch_points': solve_mismatch,
        'stage': 'P5.15 Addendum 38 -- F2 demonstration case: the node-7 4 h ladder at m = 2 (W38, spec v21 F2)',
        'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'value_definition': VALUE_DEFINITION,
        'flex_cost_definition': L.FLEX_COST_DEFINITION, 'budget_definition': BUDGET_DEFINITION,
        'predictions_recorded_before_run': spec['extra'].get('predictions_recorded_before_run'),
        'table_objective_convention': ('Q gross (settlement excluded), at the m = 2 FLEXIBILITY-PRICE SCENARIO; '
                                       'value = Q_m(0) - Q_m(x) with Q_m(0) pinned from the committed ladder; '
                                       'I = I(x) from the W2 table by candidate key; value per MWh of energy = '
                                       'value / E; value per full cycle = value / T (W25 T4 r2); the marginal MWh '
                                       'is this rung minus the previous one'),
        'table': table, 'ladder': ladder_rows, 'points': point_results,
        'pinned': pinned, 'pinned_rows_reproduce_committed': pinned_reproduction,
        'investment_inputs': investment, 'budget_corner': investment['budget_corner'],
        'budget_corner_statement': investment['budget_corner']['statement'],
        'w25_reproduction': evidence['w25_reproduction'],
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pins', 'memory_rule_vs_a0_spec',
                                                      'shared_constants_with_prior_stages',
                                                      'solves_per_cycle_from_case_file')},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'wave_info': wave_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts),
                                       'marginal_parent_counts': dict(M.PARENT_GUARD.counts),
                                       'ladder_parent_counts': dict(L.PARENT_GUARD.counts),
                                       'w25_module_guard_counts': dict(L._w25().GUARD.counts),
                                       'verify_0_failures': guards},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(CAMPAIGN_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] objective convention: {OBJECTIVE_CONVENTION}')
    for row in table:
        _log(f'[{tag}] {row}')
    for label in labels:
        p = point_results[label]
        _log(f"[{tag}] {label}: status={p['status']} cycles={p.get('cycles_run')} Q={p.get('Q')} bar={p.get('bar')} "
             f"rule_ten_gross={(p.get('rule_ten') or {}).get('terminal_gross_step_over_threshold')} "
             f"solves={(p.get('solve_reconciliation') or {}).get('observed')}/"
             f"{(p.get('solve_reconciliation') or {}).get('expected')}")
    _log(f"[{tag}] BUDGET CORNER: {investment['budget_corner']['statement']}")
    _log(f'[{tag}] pinned rows reproduce the committed ones: all={all(pinned_reproduction.values())} '
         f'failing={[k for k, v in pinned_reproduction.items() if not v]}')
    if non_certified:
        _log(f'[{tag}] non-certified points (reported with cause): '
             f"{[(l, point_results[l].get('barrier_cause')) for l in non_certified]}")
    if readback_mismatch:
        _log(f'[{tag}] READ-BACK MISMATCH: {readback_mismatch}')
    if solve_mismatch:
        _log(f'[{tag}] SOLVE RECONCILIATION MISMATCH (reported): {solve_mismatch}')
    _log(f'[{tag}] guards: parent {PARENT_GUARD.counts} / marginal parent {M.PARENT_GUARD.counts} / ladder parent '
         f'{L.PARENT_GUARD.counts} / W25 module {L._w25().GUARD.counts}; verify0 failures={guards}')
    if stop_state['triggered']:
        _banner([f'STOP_FOR_REVIEW: the stop rule fired: {stop_state["reasons"]}', f'not launched: {not_launched}'])
    if any(guards.values()) or harness_errors or readback_mismatch or not all(pinned_reproduction.values()):
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    if stop_state['triggered']:
        sys.exit(3)
    if non_certified:
        _log(f'[{tag}] harness clean; {len(non_certified)} non-certified point(s)')
        sys.exit(2)
    _log(f'[{tag}] OK: all {len(labels)} points certified')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='evaluate the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    os.chdir(REPO)
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(started, args.spec_sha256)


if __name__ == '__main__':
    main()
