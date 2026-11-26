"""
P5.15 Addendum 59 and its Supplement, Planner task W137 -- the v4 RE-SETTLING CAMPAIGN (claim groups 1, 2 and 4; the
same 38 cells, order, caps, configuration and captures as W132): the 38 per-cell campaign freezes, the frozen stage spec
`frozen_s53_resettle_spec_v4_<sha8>.json` (the resettle series, predecessor v3 139d1e62), the per-cell run (one cell per
call, priority order enforced), and the zero-solve scorer. BUILT AND FROZEN IN W137; NO RUN IS LAUNCHED BY THE WORKER.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 59 (reading (gamma): no reset; G8 on production's certificate; W135's
router patch in the same freeze; cell 1 re-run under v4 with its trajectory predicted bitwise identical to the v3 run
through cycle 173) and its Supplement (the certifying window = reading (a), the last W cycles the test reads; v4 states
W); Addendum 58 (Rulings 1-3, the order); TASKS.md Addendum 58/59 sections (the pre-registered predictions; W136; the
Planner note on the sub-test reads); Planner task W137.

WHAT CHANGED FROM THE v3 CAMPAIGN (and nothing else): the settling rule (settling_criterion_v4 via
p515_s53_w137_resettle_v4_hooks; new declaration schema -> new eval keys and campaign roots); G8 follows production's
certificate (status_production_trajectory); G17 / G18 replay and check the v4 rule records; the cell report carries the
v4 fields (vetoes, out-of-window reads); cell #1 (b_2a0ba8b2, a re-run: the v3 run ed71177e stays cited) carries the
Planner's recorded prediction, scored post-run by G26 (a failure exits 3: stop for the Planner). The cells, the order,
the caps and ceilings, the configuration, the captures, the gated replay, the holds, the claim-completion points and
the scorer are W132's (imported).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-cells                        ZERO SOLVES. The 38 campaign specs (write-once), in the v4 stage root.
  --freeze-spec                         ZERO SOLVES. The stage spec (write-once, named by its sha256).
  --run --cell C --spec-sha256 S        THE RUN OF ONE CELL (NOT RUN IN W137). Priority order enforced.
  --run ... --preconditions-only        ZERO SOLVES. Every --run precondition, then STOP before the lock and the child.
  --summarize --after-cell C            ZERO SOLVES. The scorer over every cell with committed results up to C.

Exit codes: 0 done (every stopping gate holds; a G6-only failure does not stop, Planner ruling at W128); 1 any other
gate / harness / guard / precondition failure; 3 every stopping gate holds but a PREDICTION gate (G26, cell 1) failed --
a recorded prediction failed: STOP FOR THE PLANNER before the next cell.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W137 v4 re-settling launcher (never solves)').install()

import p515_s53_w132_resettle_v3_campaign as L132  # noqa: E402 -- generic gates / scorer (arms its guards)
import p515_s53_w137_resettle_v4_checks as K  # noqa: E402 -- the zero-solve checks (arms its guard)
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import settling_criterion_v4 as SC4  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L, W101L, L118, W131 = L132.H, L132.L, L132.X, L132.W9, L132.W98L, L132.W101L, L132.L118, L132.W131
V = L132.V
R = V.R


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(L132.GUARDS) + (('w137_parent', PARENT_GUARD),))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w137_resettle_v4_campaign', 'p515_s53_w135_resettle_ext_campaign') \
    + L132.OWN_PROCESS_SUBSTRINGS
STAGE_TEXT = ('P5.15 Addendum 59 and Supplement, W137 -- v4 re-settling campaign (38 cells: claim groups 1, 2, 4) under '
              'the current production configuration: gated cells replayed bitwise against their original record through '
              'the first residual pass k0 (abort on the first divergence), the G cells as first C2 evaluations; the '
              'certifying regime held after the run\'s first residual pass (AA off, tight tail on, rho frozen); settling '
              'rule v4 (reading gamma: no reset on a non-Optimal accepted solve; certification vetoed while a non-Optimal '
              'cycle lies in the last W cycles the test reads; gap clause; amended monotone branch) until it certifies or '
              'the cap; W105 captures, per-cycle t_sum and Q_cc, and the IPOPT exit of every block of every cycle')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W137_ROOT_REL
SPEC_PREFIX = 'frozen_s53_resettle_spec_v4_'
SPEC_SERIES = 'frozen_s53_resettle_spec'
SPEC_VERSION = 4
PREDECESSOR_REL = K.W132_STAGE_SPEC['path']
PREDECESSOR_SHA256 = K.W132_STAGE_SPEC['sha256']
CAMPAIGN_IDS = {cell: f'{K.CAMPAIGN_ID_PREFIX}{cell}' for cell in V4.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
TASKS = 'TASKS.md'
SOLVER_PATH = L132.SOLVER_PATH
PYTHON = L132.PYTHON
N_NETWORK_BLOCKS = 48
RECURRING_REL = os.path.join(ROOT_REL, 'recurring_acceptable', 'w137_recurring_acceptable.json')
RECURRING_MANIFEST_REL = os.path.join(ROOT_REL, 'recurring_acceptable', 'manifest_sha256.json')
EXTRA_CLEAN_FILES = tuple(dict.fromkeys(
    (SCRIPT_NAME, 'p515_s53_w137_resettle_v4_hooks.py', 'p515_s53_w137_resettle_v4_checks.py',
     'settling_criterion_v4.py', 'p515_s53_w137_recurring_acceptable.py', 'p515_s53_w132_resettle_v3_campaign.py',
     'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w132_resettle_v3_checks.py',
     'p515_s53_w135_resettle_ext_hooks.py', K.W135_PATCH_REL) + tuple(L132.EXTRA_CLEAN_FILES)))
CODE_PINNED = tuple(dict.fromkeys(K.CODE_PINNED_BY_CHECKS + (SCRIPT_NAME, 'p515_s53_w137_recurring_acceptable.py')
                                  + tuple(L132.CODE_PINNED)))

# ---- verbatim text (checked against the committed brief / TASKS.md, whitespace-normalised, at every freeze) ------------
TASKS_PREDICTION_LINES = {
    'group_1': ('Group 1 (12 cells, m = 1): certify oscillatory at k0 + [50, 75]; s ∈ [+8, +30] k€; \\|t_sum(k\\*)\\| ≤ '
                'τ/2. Failure: any uncertified, or s < 0'),
    'b_margins': ('B margins M = I + Q_k\\* − Q181 — **Advisor:** n5_4h_e1 [75, 100] k€, n9_4h_e1 [77, 102], n9_4h_e3 '
                  '[200, 225]. **Planner (formula):** M_new = M_old + s_cell − 14,240.96 (Q181 − old-tail x0 Q) with s ∈ '
                  '[8, 30] k → n5_4h_e1 [47.4, 69.4], n9_4h_e1 [49.6, 71.6], n9_4h_e3 [171.5, 193.5] k€; all '
                  'determinate. (The two sets are disjoint for n5/n9_e1 — the outcome discriminates)'),
    'd_fit': ('D fit: settled 4 h marginal break-even within ±6 k€/MWh of 182.2 k€ (Q); margin to cost stays > 60 '
              'k€/MWh. Watch: E ≥ 4 MWh single-node cells (3632b0ae, 36686489, c7fee8be, 9246ed01) — ≥ 2 of 4 '
              'uncertified by creep → stop'),
    'group_2': ('Group 2: cells with P ≤ 1.0 MVA certify, gap ≤ τ/2; s ∈ [+8, +30] k (m 1.5), [−5, +20] k (m 2). H '
                'm1.5 value − I negative, \\|margin\\| [15, 45] k; H m2 positive [35, 60] k; I second MWh positive [30, '
                '55] k; J e2→e3 [25, 45] k, e3→e4 [35, 60] k; **5f3cccb4 dead zone** — uncertified by the gap clause, '
                't_sum ≈ −8…−10 k stationary, J e4→e5 indeterminate'),
    'group_4': ('Group 4: C 156ce2d1 F − F(0) [120, 150] k, 6597a79d [150, 180] k, both certify. G certify at k0_run + '
                '[50, 75]; gross margins 37b5c499 [115, 145], 48749148 [160, 190], 47dce43c [190, 220], 9abf31d4 [235, '
                '265] k; net ≥ 90 k; determinate both. L small cells (7b199ef9, e1da0984, df1a5525, 8e4c220e, '
                '7db09f6c, 76c78064, 195156fa) certify, s ∈ [−5, +10] k, margins move ≤ 10 k; **L dead zone** '
                '0ee93aca, 45aa25a6, 7c455554, b2251bc5 uncertified by the gap clause; 2ab0ce2d borderline (recorded, '
                'not scored)'),
}
VERBATIM = {
    (BRIEF, 'ruling2_future_specs'): L132.VERBATIM[(BRIEF, 'ruling2_future_specs')],
    (BRIEF, 'ruling1_form'): L132.VERBATIM[(BRIEF, 'ruling1_form')],
    (BRIEF, 'ruling3_net_beside'): L132.VERBATIM[(BRIEF, 'ruling3_net_beside')],
    (BRIEF, 'priority_order'): L132.VERBATIM[(BRIEF, 'priority_order')],
    (BRIEF, 'reports_at_claims'): L132.VERBATIM[(BRIEF, 'reports_at_claims')],
    (BRIEF, 'a59_no_reset'): ('**No reset** of the count: an Acceptable exit outside the window is history the clean '
                              'window re-establishes over;'),
    (BRIEF, 'a59_g8'): "gate G8 persistence follows production's certificate, not the settling label",
    (BRIEF, 'a59_cell1_rerun'): '**Cell 1 re-run** under v4 rather than re-read from v3 records',
    (BRIEF, 'a59_bitwise_prediction'): ('the trajectory to cycle 173 is predicted bitwise identical (record that '
                                        'prediction)'),
    (BRIEF, 'a59s_window_a'): ('the certifying window is **exactly the set of cycles the certification test reads at '
                               'k\\*, as implemented**'),
    (BRIEF, 'a59s_states_w'): 'Spec v4 states W and asserts that no sub-test reads outside it',
    (TASKS, 'planner_note_enumerate'): ("v4 enumerates each sub-test's reads and records the out-of-window ones instead "
                                        "of asserting none"),
    (TASKS, 'ruling_iii_caps'): L132.VERBATIM[(TASKS, 'ruling_iii_caps')],
    (TASKS, 'lattice_scope'): L132.VERBATIM[(TASKS, 'lattice_scope')],
    (TASKS, 'dead_zone'): L132.VERBATIM[(TASKS, 'dead_zone')],
    **{(TASKS, f'prediction_{k}'): v for k, v in TASKS_PREDICTION_LINES.items()},
}

REFERENCES = L132.REFERENCES
GAP_REFUSED_LABEL = L132.GAP_REFUSED_LABEL
PF_SLOPE_WINDOW = L132.PF_SLOPE_WINDOW
CELL1 = K.CELL1
CELL1_PREDICTION_THROUGH = 173

DEFINITIONS = dict(L132.DEFINITIONS)
DEFINITIONS['per_cell'] = {k: v for k, v in L132.DEFINITIONS['per_cell'].items()
                           if k not in ('first_k0_alpha', 'gamma_report_only')}
DEFINITIONS['per_cell'].update({
    'k_star': 'the cycle the settling rule v4 certified; None when uncertified at its cap',
    'first_k0_v2': ('the run\'s first residual pass under the version-2 definition (N; holds and dynamic cap keyed on '
                    'it; with no non-Optimal reset it is also v4\'s first k0)'),
    'vetoes': ('every cycle at which a branch verdict was vetoed (certification_vetoed_non_optimal): the branch, its '
               'window, the non-Optimal cycles in it, k0'),
    'out_of_window_reads': ('at k*: the reads of the certification test outside the certifying window '
                            '(settling_criterion_v4.SUB_TEST_READS applied): turning points, cycles, whether any is '
                            'non-Optimal'),
    'non_optimal_cycles': ('cycles whose all_optimal_k is False (every block\'s final exit; ESSO included); a non-Optimal '
                           'cycle is not a lapse under v4'),
})

# ---- predictions, recorded BEFORE any v4 run (each with its source) ----------------------------------------------------
PREDICTIONS = {
    'tasks_pre_registered': {
        'statements': dict(TASKS_PREDICTION_LINES),
        'source': ('TASKS.md, Addendum 58 section, "PREDICTIONS, pre-registered here before any v3 cell runs" (Advisor '
                   'design review section 7 unless marked Planner); carried verbatim (checked at freeze)'),
        'scoring': ('scored by the Planner at each claim-completion point from the summarize output (claims scored, D '
                    'fit, per-cell reports); the dead-zone and gate-ability parts also by score_predictions (W132, '
                    'unchanged); the group-1 per-cell criteria by score_group1 (Worker operationalisation below)'),
        'status_note': ('the v3 campaign\'s cell 1 FAILED the group-1 statement "all certify" under reading alpha (W133); '
                        'Addendum 59 ruled the reading and ordered the re-run: the statements are carried unchanged')},
    'group1_operationalisation': {
        'per_cell': ('certified, branch oscillatory, k* - k0_run in [50, 75], s = Q(k*) - Q_N_old in [+8,000, '
                     '+30,000] EUR, |t_sum(k*)| <= tau/2'),
        'failure_clause': 'any group-1 cell uncertified, or s < 0',
        'source': 'Worker operationalisation of TASKS_PREDICTION_LINES.group_1 (W137), for Planner confirmation'},
    'planner_addendum59_cell1': {
        'statement': ('cell 1 under v4 is bitwise identical to the v3 cell-1 run through cycle 173 and certifies at 173'),
        'source': 'Planner, Addendum 59 ("the trajectory to cycle 173 is predicted bitwise identical (record that '
                  'prediction)"); Planner task W137',
        'cell': CELL1, 'v3_run': {'commit': K.CELL1_V3['commit'], 'eval_dir': K.CELL1_V3['eval_dir']},
        'operationalisation': ('G26 (cell 1 only): (i) for every cycle 1..173 the resettle_cycle_record gross_hex of the '
                               'v4 run equals the v3 run\'s, and (ii) every REPLAY-GATED field of per_cycle_record rows '
                               '1..173 (p515_s53_w101_settling_continuation_hooks.REPLAY_GATED_FIELDS) is equal as JSON '
                               'text, and (iii) the v4 decision is certified with k* = 173'),
        'scorer': 'cell1_bitwise_prediction (this module)',
        'on_failure': 'exit 3: stop for the Planner before cell 2'},
    'dead_zone_candidates': L132.PREDICTIONS['dead_zone_candidates'],
    'dead_zone_borderline': L132.PREDICTIONS['dead_zone_borderline'],
    'gate_ability': L132.PREDICTIONS['gate_ability'],
    'walls': L132.PREDICTIONS['walls'],
}


# ======================================================================================================================
#  utilities (W132's)
# ======================================================================================================================
_utc = L132._utc
_log = L132._log
_abs = L132._abs
_load = L132._load
_sha = L132._sha
_read_jsonl = L132._read_jsonl
_committed_clean = L132._committed_clean
_norm = L132._norm
_write_once_text = L132._write_once_text


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W137] guards {g} {extra_msg}')
    for _n, guard in reversed(GUARDS):
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    import subprocess
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


# ======================================================================================================================
#  configuration, entries, keys
# ======================================================================================================================
def configuration(cell):
    cfg = L132.configuration(cell)
    cfg['name'] = cfg['name'].replace('W132 SRP1 RE-SETTLING v3 (Addendum 58)',
                                      'W137 SRP1 RE-SETTLING v4 (Addendum 59 and Supplement)')
    cfg['note'] = cfg['note'].replace('settling_resettle (v3 schema; keyed)', 'settling_resettle (v4 schema; keyed)')
    return cfg


def orig_entry(cell):
    return L132.orig_entry(cell)


def entries(cell):
    (label, nodes, opts), = L132.entries(cell)
    opts = dict(opts)
    opts['settling_resettle'] = V4.declaration_for(cell)
    return [(label, nodes, opts)]


def expected_keys(cell):
    spec, e, kw = K.K132.resettle_kwargs(cell)
    ocfg = spec['configuration']
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V4.declaration_for(cell), **kw)
    key_v3 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V.declaration_for(cell), **kw)
    base_orig = H.evaluation_key(e['key'], e['overrides'], case_file_aa=ocfg.get('case_file_anderson_acceleration'),
                                 ess_ageing_baseline=ocfg.get('ess_ageing_baseline'),
                                 flex_price_multiplier=e.get('flex_price_multiplier'),
                                 convergence_depth_tail=ocfg.get('convergence_depth_tail'))
    return {'base_key_current_configuration': base, 'resettle_key': key, 'resettle_key_v3': key_v3,
            'base_key_original_configuration': base_orig, 'original_eval_key': e['eval_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The v4 re-settling key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it
    appears in no committed campaign spec OUTSIDE the v4 stage root (the rule: a pre-run check that scans committed
    artefacts excludes the run's own); it differs from the v3 key of the cell; the original configuration's key
    reproduces the original eval key."""
    k = expected_keys(cell)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    e = orig_entry(cell)
    eval_dir_name = H.eval_dir_name(k['resettle_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['resettle_key'])
    work = L._work_dir()
    entry = next((x for x in (spec or {}).get('candidates') or [] if x['label'] == cell), None)
    parts = {
        'base_key_original_configuration_equals_original_eval_key': k['base_key_original_configuration']
        == e['eval_key'] == V4.CELLS[cell]['orig_eval_key'],
        'resettle_key_differs_from_original_and_base': k['resettle_key'] not in (e['eval_key'],
                                                                                 k['base_key_current_configuration']),
        'resettle_key_differs_from_the_v3_key': k['resettle_key'] != k['resettle_key_v3'],
        'resettle_key_absent_from_committed_specs_outside_the_v4_root': k['resettle_key'] not in committed,
        'campaign_root_differs_from_original_and_v3': os.path.abspath(campaign_root(cell)) not in (
            os.path.abspath(_abs(V4.CELLS[cell]['orig_root'])), os.path.abspath(L132.campaign_root(cell))),
        'eval_dir_name_differs_from_original': eval_dir_name != e['eval_dir'],
        'working_dir_ids_differ_from_original': not (set(ids.values()) & set(e['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['resettle_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(cell, spec):
    """W132's field-by-field campaign-spec check, the declaration and the script being this campaign's."""
    checks = L132.validate_campaign_spec(cell, spec)
    e = (spec['candidates'] or [{}])[0] if len(spec['candidates']) == 1 else {}
    checks.pop('entry_resettle_is_the_v3_declaration', None)
    checks['entry_resettle_is_the_v4_declaration'] = e.get('settling_resettle') == V4.declaration_for(cell)
    checks['campaign_id'] = spec.get('campaign_id') == CAMPAIGN_IDS[cell]
    checks['script_recorded'] = (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME
    return checks


def parent_capture_checklist(cell, spec):
    """The child's capture checklist, asserted in the PARENT too (before the lock)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = V4.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        return True, checks
    except Exception as error:  # noqa: BLE001
        return False, {'error': f'{type(error).__name__}: {error}'}


def launch_command(cell, spec_sha, preconditions_only=False):
    log = f'run_{cell}_launch.log' if not preconditions_only else f'run_{cell}_preconditions_only.log'
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run --cell {cell} --spec-sha256 {spec_sha}'
            + (' --preconditions-only' if preconditions_only else '')
            + f' > {os.path.join(ROOT_REL, log)} 2>&1')


def summarize_command(cell):
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'{PYTHON} -u {SCRIPT_NAME} --summarize --after-cell {cell} > '
            f'{os.path.join(ROOT_REL, f"summarize_after_{cell}_launch.log")} 2>&1')


# ======================================================================================================================
#  post-run gates (W132's generic ones reused; the rule-dependent ones restated for v4)
# ======================================================================================================================
LINE_FIELDS_REQUIRED = L132.LINE_FIELDS_REQUIRED
SETTLING_FIELDS_REQUIRED = L118.SETTLING_FIELDS_REQUIRED + ('all_optimal_k', 'vetoed')
_decision = L132._decision
hold_checks = L132.hold_checks
stopping_check = L132.stopping_check
replay_gate_full = L132.replay_gate_full
overlap_check = L132.overlap_check
exit_crosscheck = L132.exit_crosscheck
status_label_check = L132.status_label_check
DECISION_KEYS_REPLAYED = ('status', 'k_star', 'branch', 'k0', 'N', 'T', 'A', 'P_hat', 'W', 'window', 'band', 'band_width',
                          'k_cap', 'reasons', 'range', 't_sum_k_star', 'gap_refusals', 'drift_rate_mean_dQ_last_25',
                          'dQ_cc_rate_mean_last_25', 'version', 'reading', 'non_optimal_cycles', 'lapse_events',
                          'vetoes', 'n_vetoes', 'window_all_optimal', 'out_of_window_reads',
                          'certification_vetoed_at_cap')


def settling_replay_check(cell, eval_dir):
    """The pure rule v4 replayed on the run's per_cycle_record (Q, boyd) with the in-cycle t_sum and all_optimal_k
    reproduces every in-cycle rule record and the decision."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V4.CYCLE_FILE))}
    rule = K.pure_rule(V4.declaration_for(cell))
    pure = []
    for r in rows:
        ln = lines.get(r['cycle']) or {}
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok']), ln.get('t_sum'),
                                 bool(ln.get('all_optimal_k'))))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k))
                                                      for k in DECISION_KEYS_REPLAYED),
             'decision_file_present': bool(dec), 'decision_is_version_4': dec.get('version') == 4}
    return all(parts.values()), {'parts': parts}


def line_fields_check(eval_dir):
    lines = _read_jsonl(os.path.join(eval_dir, V4.CYCLE_FILE))
    missing = {}
    for x in lines:
        m = [f for f in LINE_FIELDS_REQUIRED if f not in x]
        if x.get('gross') is not None:
            m += [f for f in ('blocks_captured',) if f not in x]
        s = x.get('settling') or {}
        m += [f'settling.{f}' for f in SETTLING_FIELDS_REQUIRED if f not in s]
        m += [f'boyd_ratios.{f}' for f in R.BOYD_RATIO_FIELDS if f not in (x.get('boyd_ratios') or {})]
        if len(x.get('ipopt_exit_by_block') or {}) != 51:
            m.append('ipopt_exit_by_block(51)')
        if m:
            missing[str(x['cycle'])] = m
    return not missing and bool(lines), {'n_lines': len(lines), 'missing_by_cycle': dict(list(missing.items())[:10])}


def cell1_bitwise_prediction(eval_dir, v3_eval_dir=None, through=CELL1_PREDICTION_THROUGH):
    """G26 -- the Planner's Addendum 59 prediction for cell 1, scored post-run: (i) per-cycle gross_hex (the
    resettle_cycle_record line) equal to the v3 run's for every cycle 1..through; (ii) every REPLAY-GATED field of
    per_cycle_record rows 1..through equal as JSON text; (iii) the v4 decision certified with k* = through. Pure reads."""
    v3 = _abs(v3_eval_dir or K.CELL1_V3['eval_dir'])
    lines4 = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V4.CYCLE_FILE))}
    lines3 = {x['cycle']: x for x in _read_jsonl(os.path.join(v3, V4.CYCLE_FILE))}
    rows4 = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    rows3 = {x['cycle']: x for x in _read_jsonl(os.path.join(v3, 'per_cycle_record.jsonl'))}
    first_hex, first_field = None, None
    for k in range(1, through + 1):
        a, b = lines4.get(k), lines3.get(k)
        if first_hex is None and (a is None or b is None or a.get('gross_hex') != b.get('gross_hex')):
            first_hex = {'cycle': k, 'v4': (a or {}).get('gross_hex'), 'v3': (b or {}).get('gross_hex'),
                         'missing': a is None or b is None}
        ra, rb = rows4.get(k), rows3.get(k)
        if first_field is None:
            if ra is None or rb is None:
                first_field = {'cycle': k, 'missing': True}
            else:
                diff = [f for f in R.REPLAY_GATED_FIELDS if json.dumps(ra.get(f), sort_keys=True)
                        != json.dumps(rb.get(f), sort_keys=True)]
                if diff:
                    first_field = {'cycle': k, 'fields_differing': diff}
        if first_hex is not None and first_field is not None:
            break
    dec = _decision(eval_dir) or {}
    parts = {'gross_hex_equal_every_cycle_1_to_173': first_hex is None,
             'gated_fields_equal_every_cycle_1_to_173': first_field is None,
             'certified_at_173': dec.get('status') == 'certified' and dec.get('k_star') == through}
    return all(parts.values()), {'parts': parts, 'first_gross_hex_difference': first_hex,
                                 'first_gated_field_difference': first_field, 'decision_status': dec.get('status'),
                                 'k_star': dec.get('k_star'), 'through': through, 'v3_eval_dir': os.path.relpath(v3, REPO),
                                 'prediction': PREDICTIONS['planner_addendum59_cell1']['statement']}


GATE_SCOPE = dict(L132.GATE_SCOPE)
GATE_SCOPE['G8_persistence_on_production_certificate'] = ('every cell: post-certification persistence follows '
                                                         'status_production_trajectory (Addendum 59)')
GATE_SCOPE['G26_cell1_prediction_bitwise_v3_through_173_and_certifies_at_173'] = (
    'cell 1 (b_2a0ba8b2) ONLY -- the Planner\'s recorded prediction; SKIPPED for every other cell (it is designed for '
    'the re-run only); a failure exits 3 (stop for the Planner)')
NON_STOPPING_GATES = ('G6_v37_optimal_and_four_metrics', 'G26_cell1_prediction_bitwise_v3_through_173_and_certifies_at_173')
PREDICTION_GATES = ('G26_cell1_prediction_bitwise_v3_through_173_and_certifies_at_173',)


def cell_gates(cell, entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code, 'gate_scope': GATE_SCOPE}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_resettle_summary'] = rec.get('settling_resettle_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys(cell)['resettle_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W101L.solve_profile_check(rec, len(records))
    g6 = W98L.g6_v37_evaluate_records(records, rec.get('cycles_run'), b=N_NETWORK_BLOCKS)
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_on_production_certificate'], detail['G8'] = V4.persistence_check_production_certificate(
        rec, eval_dir)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_first_pass_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_v4_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    if V4.CELLS[cell]['gated']:
        rg = replay_gate_full(cell, eval_dir)
        gates['G19_replay_bitwise_1_k0_every_field'] = rg['bitwise_through_k0']
        detail['G19'] = rg
    else:
        detail['G19'] = {'skipped': 'ungated cell (first C2 evaluation): no replay reference'}
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = L118.creep_capture_check(eval_dir, rec, rows)
    gates['G22_t_sum_in_cycle_equals_stride_and_terminal'], detail['G22'] = L118.t_sum_check(eval_dir)
    gates['G23_overlap_recorded'], detail['G23'] = overlap_check(cell, rec)
    try:
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'], detail['G24'] = exit_crosscheck(
            eval_dir, rec=rec)
    except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'] = False
        detail['G24'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    gates['G25_record_status_follows_the_settling_decision'], detail['G25'] = status_label_check(rec, eval_dir)
    if cell == CELL1:
        try:
            gates[PREDICTION_GATES[0]], detail['G26'] = cell1_bitwise_prediction(eval_dir)
        except Exception as error:  # noqa: BLE001 -- recorded; the prediction gate FAILS
            gates[PREDICTION_GATES[0]] = False
            detail['G26'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    else:
        detail['G26'] = {'skipped': 'cell 1 only (the Planner\'s prediction for the re-run)'}
    return gates, detail, rec


def cell_report(cell, eval_dir, rec):
    """W132's per-cell report with the v4 fields in place of the v3 alpha / gamma fields."""
    rep = L132.cell_report(cell, eval_dir, rec)
    dec = _decision(eval_dir) or {}
    for k in ('first_k0_alpha', 'gamma_report_only'):
        rep.pop(k, None)
    rep.update({'criterion_version': dec.get('version'), 'reading': dec.get('reading'), 'W': dec.get('W'),
                'window': dec.get('window'), 'vetoes': dec.get('vetoes'), 'n_vetoes': dec.get('n_vetoes'),
                'window_all_optimal': dec.get('window_all_optimal'),
                'out_of_window_reads': dec.get('out_of_window_reads'),
                'certification_vetoed_at_cap': dec.get('certification_vetoed_at_cap')})
    rep['view'] = L132.view_from_report(rep)
    return rep


def score_group1(reports):
    """The group-1 per-cell criteria (PREDICTIONS.group1_operationalisation); report-only."""
    out = {}
    for cell, r in reports.items():
        if V4.CELLS[cell]['item'] not in ('B', 'D'):
            continue
        cert = r.get('status') == 'certified'
        k0 = r.get('k0_run')
        s = r.get('s_signed')
        t = r.get('t_sum_k_star')
        out[cell] = {'certified': cert, 'branch_oscillatory': r.get('branch') == 'oscillatory',
                     'k_star_minus_k0_run': (r['k_star'] - k0) if (cert and k0 is not None) else None,
                     'k_star_in_k0_plus_50_75': bool(cert and k0 is not None and 50 <= r['k_star'] - k0 <= 75),
                     's': s, 's_in_8k_30k': s is not None and 8000.0 <= s <= 30000.0,
                     'abs_t_sum_le_tau_over_2': t is not None and abs(t) <= SC4.GAP_BOUND,
                     'failure_clause_uncertified_or_s_negative': (not cert) or (s is not None and s < 0)}
    return out


# ======================================================================================================================
#  post-run evaluator self-tests (synthetic eval dirs from the REAL wrappers; tampered negative controls; G8; G26)
# ======================================================================================================================
def _synthetic_run_dir(tmp, cell, variant):
    """W132's `_synthetic_run_dir` restated on the v4 drive (K.drive) and the v4 rule."""
    d = K.drive(cell, 'creep' if variant == 'creep' else 'certify',
                nonopt=({V4.CELLS[cell]['N_old'] + 2: ('ESSO|9',)} if variant == 'nonopt' else None))
    files, st = d['files'], d['state']
    lines = {x['cycle']: x for x in files[V4.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[V4.CREEP_FILE]}
    if variant == 'tamper_t_sum_30':
        lines[30]['t_sum'] = lines[30]['t_sum'] + 1.0
    if variant == 'tamper_hold_flag':
        lines[max(lines) - 5]['aa']['hold'] = False
    if variant == 'tamper_decision':
        files[V4.DECISION_FILE][0]['k_star'] = files[V4.DECISION_FILE][0]['k_star'] - 1
    if variant == 'tamper_exit':
        c = max(lines) - 1
        lines[c]['all_optimal_k'] = False
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == V4.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            elif fname == V4.CYCLE_FILE:
                for c in sorted(lines):
                    handle.write(GRIO.dumps(lines[c], default=GRIO.json_default) + '\n')
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    ref = st.reference if st.gated else {}
    rows, g_rows = [], []
    g_orig = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(V4.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if st.gated else {})
    for c in sorted(lines):
        x = lines[c]
        if st.gated and c <= V4.CELLS[cell]['N_old']:
            row = dict(ref[c])
            g = dict(g_orig[c])
        else:
            row = {'cycle': c, 'local_solves_ok': True, 'recourse': x['net_operational_recourse'],
                   'gross_operational_cost': x['gross'], 'terminal_salvage_value': x['terminal_salvage_value'],
                   'objective_change_abs': abs(x['step'] or 0.0), 'objective_tolerance': 65000.0,
                   'objective_change_ratio': abs(x['step'] or 0.0) / 65000.0, 'cycle_convergence': x['boyd_k'],
                   'consecutive_converged_cycles': x['consecutive_converged_cycles_tracked'],
                   'boyd_all_pass': x['boyd_k'], 'boyd_stop': x['boyd_k'],
                   **{f: x['boyd_ratios'][f] for f in R.BOYD_RATIO_FIELDS},
                   **{f'boyd_{g_}_channel_pass': creep[c]['boyd'][g_]['channel_pass'] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_after': x['rho']['rho_after'][g_] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_action': x['rho']['actions'][g_] for g_ in R.CHANNELS},
                   'rho_freeze_active': True, 'efc_per_day_max': 1.0}
            g = {'cycle': c}
        for g_ in R.CHANNELS:
            for f, v in creep[c]['boyd'][g_].items():
                g[f'boyd_{g_}_{f}'] = v
        rows.append(row)
        g_rows.append(g)
    if variant == 'tamper_row_40':
        rows[39] = dict(rows[39], objective_change_abs=(rows[39]['objective_change_abs'] or 0.0) + 1.0)
    with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
        for r in rows:
            handle.write(GRIO.dumps(r, default=GRIO.json_default) + '\n')
    with open(os.path.join(tmp, 'g_s39_D.json'), 'w') as handle:
        json.dump({'cycle_trajectory': g_rows}, handle)
    kk = K.K132.K118
    n_e = len(kk.NODES) * len(kk.YEARS) * len(kk.DAYS) * kk.PERIODS
    with open(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), 'w') as handle:
        for c in sorted(lines):
            t = lines[c]['t_sum'] if variant != 'tamper_t_sum_30' or c != 30 else lines[c]['t_sum'] - 1.0
            gap = t / n_e
            ents = [{'node_id': n, 'year': str(y), 'day': d_, 'power_type': 'p', 'period': p, 'x_dso': 10.0 + gap,
                     'z_tso_current': 10.0, 'lambda_dso': 0.0, 's_base_dso': 100.0, 'rho_pf': 1.0, 'r': 0.0}
                    for n in kk.NODES for y in kk.YEARS for d_ in kk.DAYS for p in range(kk.PERIODS)]
            handle.write(json.dumps({'cycle': c, 'identity_holds': True, 'production_boyd_pf_r': 0.0,
                                     'entries': ents}) + '\n')
    last = max(lines)
    detail = {'t_tso_plus_t_dso_terminal': lines[last]['t_sum'], 'cycles_run': last,
              'interface_reporting_detail': {str(n): {str(y): {d_: {'periods': {str(p): {'price_per_mwh': 1.0}
                                                                                  for p in range(kk.PERIODS)}}
                                                               for d_ in kk.DAYS} for y in kk.YEARS}
                                             for n in kk.NODES},
              'interface_consensus_residual_per_dso': {str(n): {'periods': {f'{y}|{d_}|{p}': {'admm_block_weight': 1.0}
                                                                            for y in kk.YEARS for d_ in kk.DAYS
                                                                            for p in range(kk.PERIODS)},
                                                                'sum_pi_baseMVA_residual_weighted': 0.0}
                                                       for n in kk.NODES}}
    with open(os.path.join(tmp, 'interface_settlement_detail_s31c.json'), 'w') as handle:
        json.dump(detail, handle)
    rec = H._apply_settling_resettle_status({'cycles_run': last, 'settling_resettle_summary': st.summary(),
                                             'status': 'certified', 'barrier': False, 'barrier_cause': None,
                                             'certified_cost': 1.0, 'certification_cycle': last,
                                             'terminal_gross_operational_cost': 1.0})
    return rec, rows


def _g26_self_tests():
    """G26 on committed data: the v3 run against itself (equal through 173 but NOT certified at 173 -> the prediction
    scorer fails, as it must on an uncertified record); a copy truncated to 173 with a v4-certified decision (passes); the
    same copy with one gross_hex changed at 150 (fails, first difference 150)."""
    v3 = _abs(K.CELL1_V3['eval_dir'])
    out = {}
    ok_self, d_self = cell1_bitwise_prediction(v3)
    out['v3_against_itself'] = {'ok': (not ok_self) and d_self['parts']['gross_hex_equal_every_cycle_1_to_173']
                                and d_self['parts']['gated_fields_equal_every_cycle_1_to_173']
                                and not d_self['parts']['certified_at_173'], 'parts': d_self['parts']}
    for name, tamper in (('certified_copy_passes', False), ('gross_hex_tampered_at_150_fails', True)):
        tmp = tempfile.mkdtemp(prefix='w137_g26_')
        try:
            lines = _read_jsonl(os.path.join(v3, V4.CYCLE_FILE))[:CELL1_PREDICTION_THROUGH]
            rows = _read_jsonl(os.path.join(v3, 'per_cycle_record.jsonl'))[:CELL1_PREDICTION_THROUGH]
            if tamper:
                lines[149]['gross_hex'] = float.hex(float.fromhex(lines[149]['gross_hex']) + 1.0)
            with open(os.path.join(tmp, V4.CYCLE_FILE), 'w') as handle:
                for x in lines:
                    handle.write(json.dumps(x) + '\n')
            with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
                for x in rows:
                    handle.write(json.dumps(x) + '\n')
            with open(os.path.join(tmp, V4.DECISION_FILE), 'w') as handle:
                json.dump({'status': 'certified', 'k_star': CELL1_PREDICTION_THROUGH, 'version': 4}, handle)
            ok, dd = cell1_bitwise_prediction(tmp)
            out[name] = {'ok': (ok is True) if not tamper else (ok is False and dd['first_gross_hex_difference']['cycle']
                                                                == 150),
                         'parts': dd['parts'], 'first_gross_hex_difference': dd['first_gross_hex_difference']}
        finally:
            shutil.rmtree(tmp)
    return out, all(v['ok'] for v in out.values())


def _g8_self_tests():
    """G8 on W133's committed cell-1 record: the W101 formula (record status) FAILS on it, as W133 recorded; the v4 G8
    (production's certificate) PASSES; the pkl hashes to the recorded sha256."""
    ev = _abs(K.CELL1_V3['eval_dir'])
    rec = json.load(open(os.path.join(ev, 'evaluation_record.json')))
    old_ok, old = W101L.persistence_check(rec, ev)
    new_ok, new = V4.persistence_check_production_certificate(rec, ev)
    return {'ok': bool((not old_ok) and new_ok and new['pkl_sha256_matches_recorded_report_only'] is True),
            'w101_g8_on_the_v3_cell1_record': {'ok': old_ok, 'detail': old},
            'v4_g8_on_the_v3_cell1_record': {'ok': new_ok, 'detail': new}}


def post_run_evaluator_self_tests():
    """The v4 post-run evaluators (G13, G15, G17, G18, G19, G22, G23, G25, the cell report) on synthetic eval dirs built
    by the real wrappers on the v4 state, with tampered negative controls; G8 and G26 on W133's committed cell 1; W132's
    G24 self-test and scorer self-tests (reused)."""
    out = {}
    cases = (('b_2a0ba8b2', 'pass', None, True), ('b_2a0ba8b2', 'tampered_row_40', 'tamper_row_40', False),
             ('b_2a0ba8b2', 'tampered_t_sum_30', 'tamper_t_sum_30', False),
             ('b_2a0ba8b2', 'tampered_hold_flag', 'tamper_hold_flag', False),
             ('b_2a0ba8b2', 'tampered_decision', 'tamper_decision', False),
             ('h_74eda68d', 'non_optimal_exit_is_not_a_lapse', 'nonopt', True),
             ('h_74eda68d', 'tampered_exit_flag', 'tamper_exit', False),
             ('g_37b5c499', 'ungated_rule_cap', 'creep', True))
    for cell, name, variant, expect in cases:
        tmp = tempfile.mkdtemp(prefix='w137_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, cell, variant)
            h_ok, _h = hold_checks(cell, tmp, rec)
            s_ok, _s = stopping_check(cell, rec, tmp)
            k_ok, _k = settling_replay_check(cell, tmp)
            f_ok, _f = line_fields_check(tmp)
            c_ok, _c = L118.creep_capture_check(tmp, rec, rows)
            t_ok, t_d = L118.t_sum_check(tmp, os.path.relpath(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), REPO),
                                         os.path.relpath(os.path.join(tmp, 'interface_settlement_detail_s31c.json'),
                                                         REPO))
            o_ok, _o = overlap_check(cell, rec)
            l_ok, _l = status_label_check(rec, tmp)
            rg = replay_gate_full(cell, tmp) if V4.CELLS[cell]['gated'] else {'bitwise_through_k0': True}
            allg = {'holds': h_ok, 'stopping': s_ok, 'rule_replay': k_ok, 'line_fields': f_ok, 'creep': c_ok,
                    't_sum': t_ok, 'overlap': o_ok, 'status_label': l_ok, 'replay_full': rg['bitwise_through_k0']}
            if expect:
                ok = all(allg.values())
            elif variant == 'tamper_row_40':
                ok = (not rg['bitwise_through_k0']) and rg['first_divergence_cycle'] == 40 and all(
                    v for k_, v in allg.items() if k_ != 'replay_full')
            elif variant == 'tamper_t_sum_30':
                ok = (not t_ok) and t_d['max_abs_diff_vs_stride'] >= 0.99
            elif variant == 'tamper_hold_flag':
                ok = (not h_ok) and all(v for k_, v in allg.items() if k_ != 'holds')
            elif variant == 'tamper_exit':
                ok = (not k_ok) and all(v for k_, v in allg.items() if k_ != 'rule_replay')
            else:
                ok = (not k_ok) and (not s_ok)
            rep = cell_report(cell, tmp, rec) if expect else None
            out[name] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                         'report_on_synthetic': ({k: rep.get(k) for k in ('status', 'k_star', 'branch', 'band_width',
                                                                          's_signed', 't_sum_end', 'k0_run',
                                                                          'non_optimal_cycles', 'n_vetoes',
                                                                          'window_all_optimal', 'label')}
                                                 if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    try:
        out['G8_on_the_committed_v3_cell1_record'] = _g8_self_tests()
    except Exception as error:  # noqa: BLE001
        out['G8_on_the_committed_v3_cell1_record'] = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    try:
        g26, g26_ok = _g26_self_tests()
        out['G26_cell1_prediction_scorer'] = {'ok': g26_ok, 'tests': g26}
    except Exception as error:  # noqa: BLE001
        out['G26_cell1_prediction_scorer'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                              'traceback': traceback.format_exc()}
    # W132's own post-run self-tests (unchanged, pinned 139d1e62): they carry the G24 comparison self-test on a
    # committed W118 cell (G24 is reused unchanged here) and W132's scorer self-tests (the scorer is reused unchanged)
    w132, w132_ok = L132.post_run_evaluator_self_tests()
    out['w132_post_run_and_scorer_self_tests_reused'] = {
        'ok': bool(w132_ok), 'per_test': {k: v.get('ok') for k, v in w132.items()},
        'g24': w132.get('G24_exit_crosscheck_on_committed_w118_cell'), 'scorer': w132.get('scorer')}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  checks state, verbatim, common preconditions
# ======================================================================================================================
def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST),
            'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'), 'all_hold_including_typing_test': doc.get('all_hold_including_typing_test'),
            'typing_test': doc.get('W_bool_typing_test'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256')}


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks()


_failing_check_items = L132._failing_check_items


def verbatim_check():
    texts = {f: _norm(open(_abs(f), encoding='utf-8').read()) for f in (BRIEF, TASKS)}
    found = {f'{f}:{k}': _norm(v) in texts[f] for (f, k), v in VERBATIM.items()}
    return {'files': {f: {'sha256_at_freeze': _sha(f), 'git_state': L._git_state(f)} for f in (BRIEF, TASKS)},
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def recurring_state():
    ok = os.path.isfile(_abs(RECURRING_REL)) and os.path.isfile(_abs(RECURRING_MANIFEST_REL))
    if not ok:
        return {'ok': False, 'path': RECURRING_REL}
    man = _load(RECURRING_MANIFEST_REL)
    sha = _sha(RECURRING_REL)
    doc = _load(RECURRING_REL)
    return {'ok': bool(man.get(RECURRING_REL) == sha and _committed_clean(RECURRING_REL)
                       and _committed_clean(RECURRING_MANIFEST_REL) and doc.get('all_seeded_as_seeded') is True),
            'path': RECURRING_REL, 'sha256': sha, 'manifest': RECURRING_MANIFEST_REL,
            'all_seeded_as_seeded': doc.get('all_seeded_as_seeded'),
            'recurring_blocks': [{k: b[k] for k in ('block', 'network', 'n_cells', 'n_solves', 'cells')}
                                 for b in doc.get('recurring_blocks') or []]}


def _common_checks():
    failures = []
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    for cell in V4.CELL_ORDER:
        rel = V4.reference_path(cell)
        if _sha(rel) != V4.CELLS[cell]['per_cycle_record_sha256'] or not _committed_clean(rel):
            failures.append(f'{cell} original record not as committed: {rel}')
    for rel in (PREDECESSOR_REL, K.W131['path'], K.K132.W117['path'], L132.W118_SUMMARY, L132.BASELINE_TABLES,
                L132.W2_TABLE, os.path.join(K.CELL1_V3['eval_dir'], 'per_cycle_record.jsonl'),
                os.path.join(K.CELL1_V3['eval_dir'], V4.CYCLE_FILE)):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    if _sha(PREDECESSOR_REL) != PREDECESSOR_SHA256:
        failures.append('the predecessor stage spec v3 is not 139d1e62')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a W137 / W135 / W132 / W118 / W105 / W101 / W98 launcher is alive: {others}')
    return failures


def _checks_output_ok(failures):
    cf = _checks_file_state()
    if not (cf['all_hold_including_typing_test'] is True and cf['committed_clean']
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in K.CODE_PINNED_BY_CHECKS:
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    return cf


# ======================================================================================================================
#  --freeze-cells
# ======================================================================================================================
def freeze_cells(started):
    tag = 'W137-FREEZE-CELLS'
    failures = _common_checks()
    cf = _checks_output_ok(failures)
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    for cell in V4.CELL_ORDER:
        failures += H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)
        pre = pre_launch_assertion(cell)
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    if os.path.isdir(_abs(ROOT_REL)) and any(f.startswith(SPEC_PREFIX) for f in os.listdir(_abs(ROOT_REL))):
        failures.append('a stage spec already exists: the cell specs are frozen BEFORE the stage spec')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    all_ok = True
    checks_pin = {'path': cf['path'], 'sha256': cf['sha256']}
    for cell in V4.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        c = V4.CELLS[cell]
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ('frozen_s53_resettle_spec_v4 (frozen AFTER this campaign spec) pins this spec by its '
                                'sha256 and holds the exact launch command'),
                 'label': V4.LABEL, 'cell': cell, 'item': c['item'], 'claim_group': V4.GROUP_OF_ITEM[c['item']],
                 'original': {'campaign_id': c['orig_campaign_id'], 'eval_key': c['orig_eval_key'],
                              'eval_dir': V4.original_eval_dir(cell), 'N_old': c['N_old'], 'k0': c['k0']},
                 'v3_cell': {'campaign_root': L132.campaign_root_rel(cell), 'eval_key_v3': pre['resettle_key_v3'],
                             'stage_spec_v3': {'path': PREDECESSOR_REL, 'sha256': PREDECESSOR_SHA256}},
                 'expected_eval_key': pre['resettle_key'], 'objective_convention': DEFINITIONS['objective_convention'],
                 'solve_claim': {'parent': 'never solves (every launcher guard permitted=(), verify(0))',
                                 'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
                 'zero_solve_checks_output': checks_pin, 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(V4.CELL_ORDER), 'code_sha256': {rel: _sha(rel) for rel in CODE_PINNED}}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=V4.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 59 and its Supplement', f'{BRIEF} Addendum 58', 'Planner task W137'],
            required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec)
        pre_frozen = pre_launch_assertion(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds']
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha} eval_key={e["eval_key"]} '
             f'(v3 {pre["resettle_key_v3"][:16]}) cap={spec["cap"]} checks={all(checks.values())} '
             f'failing={[k for k, v in checks.items() if not v]} pre-launch={pre_frozen["holds"]}')
    _finish(0 if all_ok else 1, f'freeze-cells {"OK" if all_ok else "NOT OK"} -- next: commit, then --freeze-spec')


def cell_spec_state():
    """{cell: {path, sha256, committed_clean, checks, pre_launch}} of the 38 frozen v4 campaign specs."""
    out = {}
    for cell in V4.CELL_ORDER:
        root = campaign_root(cell)
        files = sorted(f for f in os.listdir(root) if f.startswith('campaign_spec_')) if os.path.isdir(root) else []
        if len(files) != 1:
            out[cell] = {'error': f'campaign specs in {root}: {files}'}
            continue
        rel = os.path.relpath(os.path.join(root, files[0]), REPO)
        sha = _sha(rel)
        spec = _load(rel)
        checks = validate_campaign_spec(cell, spec)
        pre = pre_launch_assertion(cell, spec)
        out[cell] = {'path': rel, 'sha256': sha, 'name_carries_sha': files[0].endswith(f'_{sha[:8]}.json'),
                     'committed_clean': _committed_clean(rel), 'checks_all': all(checks.values()),
                     'checks_failing': [k for k, v in checks.items() if not v], 'pre_launch_holds': pre['holds'],
                     'eval_key': spec['candidates'][0]['eval_key'], 'eval_key_v3': pre['resettle_key_v3'],
                     'eval_dir': spec['candidates'][0]['eval_dir'],
                     'harness_sha256': spec['harness']['sha256'], 'git_head': spec.get('git_head'),
                     'root_holds_only_the_spec': sorted(os.listdir(root)) == [files[0]]}
    return out


# ======================================================================================================================
#  --freeze-spec
# ======================================================================================================================
def _find_stage_spec():
    hits = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX) and f.endswith('.json')) \
        if os.path.isdir(_abs(ROOT_REL)) else []
    if len(hits) != 1:
        return None, None
    rel = os.path.join(ROOT_REL, hits[0])
    sha = _sha(rel)
    if hits[0] != f'{SPEC_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'stage spec file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def load_stage_spec():
    rel, sha = _find_stage_spec()
    if rel is None:
        raise RuntimeError('frozen v4 re-settling stage spec not found')
    return rel, sha, _load(rel)


def stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section, dataset, points,
                       refs, ref_inputs, recurring):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    v3_ss = _load(PREDECESSOR_REL)
    r_section = checks_inline['sections']['R']['result']
    cells = {}
    for i, cell in enumerate(V4.CELL_ORDER):
        c = V4.CELLS[cell]
        o = o_section['cells'][cell]
        s = specs[cell]
        cells[cell] = {
            'launch_index_1_based': i + 1, 'item': c['item'], 'claim_group': V4.GROUP_OF_ITEM[c['item']],
            'gated': c['gated'], 'prefix': c['orig_eval_key'][:8],
            'original': {**o['original'], 'label': c['orig_label']},
            'm_flex_price_multiplier': c['flex_price_multiplier'] if c['flex_price_multiplier'] is not None else 1.0,
            'investment_year': o['investment_year'], 'canonical': o['canonical'], 'candidate_key': o['candidate_key'],
            'N_old': c['N_old'], 'k0_original_first_residual_pass': c['k0'],
            'original_lapses_after_k0': c['original_lapses_after_k0'], 'Q_N_old': o['Q_N_old'],
            'cap_rule': V4.cap_rule(cell), 'spec_cap': V4.spec_cap(cell), 'cap_ceiling': c['cap_ceiling'],
            'e_over_p': o['e_over_p'], 'lattice_e_over_p_legal': o['parts']['lattice_e_over_p_in_2_4_every_storage_node'],
            'I_cited': o['I'], 'I_source': o['I_source'], 't_sum_terminal_original': o['t_sum_terminal_original'],
            'dead_zone_candidate': o['dead_zone_candidate'], 'dead_zone_borderline': o['dead_zone_borderline'],
            'identity_vs_original': o['identity_vs_original'],
            'declaration': V4.declaration_for(cell), 'campaign_id': CAMPAIGN_IDS[cell],
            'campaign_root': campaign_root_rel(cell), 'configuration': configuration(cell), 'keys': expected_keys(cell),
            'v3': {'campaign_root': L132.campaign_root_rel(cell),
                   'campaign_spec': v3_ss['pins']['campaign_specs'][cell],
                   'eval_key': v3_ss['cells'][cell]['campaign_spec']['eval_key'],
                   'run': ({'commit': K.CELL1_V3['commit'], 'eval_dir': K.CELL1_V3['eval_dir'],
                            'status': 'cited (W133): uncertified at the cap 213 under v3 alpha; gamma would certify '
                                      'at 173; G8 failed on the label interaction'} if cell == CELL1 else None)},
            'rerun_of_v3_cell': cell == CELL1,
            'campaign_spec': {k: s[k] for k in ('path', 'sha256', 'eval_key', 'eval_dir', 'harness_sha256', 'git_head')},
            'launch_command': launch_command(cell, s['sha256']),
            'preconditions_only_command': launch_command(cell, s['sha256'], preconditions_only=True),
            'expected_wall_time': wall['per_cell'][cell],
            'prediction_gate': PREDICTIONS['planner_addendum59_cell1'] if cell == CELL1 else None}
    return {
        'schema': 'p515_s53_resettle_spec_v4', 'series': SPEC_SERIES, 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 3, 'path': PREDECESSOR_REL, 'sha256': _sha(PREDECESSOR_REL),
                        'status': ('the W132 v3 campaign spec; cell 1 ran under it (W133, ed71177e, cited); superseded '
                                   'for the remaining cells by this v4 spec (Addendum 59: v4 re-freeze)')},
        'authority': [f'{BRIEF} Addendum 59 (reading gamma; G8; W135 router patch in the same freeze; cell 1 re-run '
                      f'with its bitwise prediction)', f'{BRIEF} Addendum 59 Supplement (window = reading (a))',
                      f'{BRIEF} Addendum 58 (Rulings 1-3; order)', 'TASKS.md Addendum 58 / 59 sections',
                      'W136 (window replay, scratch; reused in committed code)', 'Planner task W137'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'pins': {'code_sha256': code, 'production_sha256': production, 'solver': solver,
                 'zero_solve_checks_output': {'path': cf['path'], 'sha256': cf['sha256'], 'manifest': cf['manifest']},
                 'w131': {'path': K.W131['path'], 'sha256': K.W131['sha256']},
                 'w117': {'path': K.K132.W117['path'], 'sha256': K.K132.W117['sha256']},
                 'references_inputs_sha256': ref_inputs,
                 'baseline_tables': {'path': L132.BASELINE_TABLES, 'sha256': _sha(L132.BASELINE_TABLES)},
                 'w2_table': {'path': L132.W2_TABLE, 'sha256': _sha(L132.W2_TABLE)},
                 'recurring_acceptable_report': {k: recurring[k] for k in ('path', 'sha256', 'manifest')},
                 'cell1_v3_run': {'commit': K.CELL1_V3['commit'], 'files_sha256': r_section['cells'][CELL1][
                     'inputs_sha256']},
                 'campaign_specs': {c: {'path': specs[c]['path'], 'sha256': specs[c]['sha256']} for c in V4.CELL_ORDER}},
        'production_since_originals': prov,
        'cells': cells, 'cell_order': list(V4.CELL_ORDER),
        'same_as_v3': {'cells_order_caps_ceilings_configuration_captures': True,
                       'checked': ('cell table imported from p515_s53_w132_resettle_v3_hooks (pinned 139d1e62); '
                                   'configuration and entry checked field by field against W132\'s (validate_campaign_'
                                   'spec); the declaration differs only in schema, label and settling_rule')},
        'not_in_this_spec': {'group_3_ageing_E_and_pb_y2025_n5': ('frozen_s53_resettle_ext_spec_v1 (W135 extension, '
                                                                  'frozen against this spec; after cell #38)'),
                             'covered_by_settled_references': {'7aa017f0': 'x0 settled d110bd1a',
                                                               'bd504ecf': 'unit settled 3f084f2f'}},
        'launch_order': {'order': list(V4.CELL_ORDER), 'enforced': 'every earlier cell has results before --run',
                         'one_cell_per_call': True},
        'launch_commands': {c: cells[c]['launch_command'] for c in V4.CELL_ORDER},
        'summarize_commands_at_the_completion_points': {p['after_cell']: summarize_command(p['after_cell'])
                                                        for p in points['report_points']},
        'claim_completion_points': points,
        'inputs_in_force_now': o_section['inputs_now'], 'references': refs,
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '+ tight tail {enabled True, compl_inf_tol 1e-6} declared'),
                          'tail_rule': 'production: the tail acts from AA-off + 1 (next-state = this cycle converged)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as W101/W118/'
                                         'W132)',
                          'concurrency': CONCURRENCY, 'option_b_release_solution_bookkeeping': 'absent',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze-cells, --freeze-spec and --run'},
        'stop_rule': {
            'module': 'settling_criterion_v4', 'class': 'settling_criterion_v4.SettlingRuleV4', 'version': SC4.VERSION,
            'versions_1_2_3_unchanged': ('settling_criterion.py, settling_criterion_v2.py and settling_criterion_v3.py '
                                         'byte-identical (checks V0)'),
            'constants': SC4.constants(V4.P_MAX), 'readings': SC4.READINGS, 'algorithm': SC4.__doc__,
            'failing_reasons': list(SC4.FAILING_REASONS),
            'p_max': {'value': V4.P_MAX, 'L': V4.L_MONO, 'source': 'W102 x0 P_hat 29, W103 unit P_hat 30 (as fc791891)'},
            'W': {'oscillatory': 'W = max(W_MIN, ceil(W_FACTOR * P_hat)) = max(20, ceil(1.1 * (T[-1].t - T[-3].t)))',
                  'monotone': 'L = L_MONO = 2 * P_MAX = 60',
                  'certifying_window': 'oscillatory [k - W + 1, k]; monotone [k - L + 1, k] (reading (a))'},
            'reading_gamma': 'the decision: no reset on a non-Optimal cycle; the certification veto on the window',
            'retry_tier': None,
            'sub_test_reads': SC4.SUB_TEST_READS,
            'sub_test_reads_assertion': ('NOT ASSERTED: the Addendum 59 Supplement says the test reads only the last W '
                                         'cycles; as implemented the oscillatory branch reads the swing amplitudes of '
                                         'every turning point since k0 and P_hat reads T[-1] and T[-3], and the monotone '
                                         'branch reads Q_(lo-1) and the sign of the last non-zero step before lo. Every '
                                         'certification records its out-of-window reads (decision out_of_window_reads); '
                                         'flagged by the Planner for the expert (TASKS.md); the algorithm is unchanged'),
            'validation_replay_13_cells': r_section['summary'],
            'N_and_holds_and_dynamic_cap': 'keyed on the first residual pass under the version-2 definition',
            'caps': {'gated': 'N_old + 100 (fixed; the per-cell ceiling in the cell table)',
                     'ungated': 'min(k0_run + 109, 300) (dynamic)',
                     'above_300': {'l_2ab0ce2d': 437, 'l_b2251bc5': 320}},
            'all_optimal_k_capture': 'as 139d1e62 (W132\'s ninth pass-through wrapper, reused unchanged; G24)',
            'boyd_k': 'boyd_metrics[\'all_boyd_pass\'] AND local_solves_ok, from THIS cycle (AA wrapper)',
            'Q_k': 'gross_operational_cost (recourse wrapper); None when any local solve failed',
            't_sum_k': 'as fc791891 / 139d1e62 (in-cycle; validated post-run, G22)',
            'mechanism': ('decision in the recourse wrapper; certification (or the dynamic cap of an ungated cell below '
                          'the spec cap) sets the certificate length to 0 -> production\'s own exit test ends the loop '
                          'at the end of THIS cycle; the decision file is written once'),
            'early_stop': 'ABSENT (validator refuses the key; checklist)'},
        'replay_gate': {
            'applies_to': list(V4.GATED_CELLS), 'skipped_for': list(V4.UNGATED_CELLS),
            'in_cycle': ('every cycle k <= k0 (the original first residual pass), at the end of the cycle: '
                         + ', '.join(R.REPLAY_GATED_FIELDS) + ' as JSON text against the ORIGINAL record row k; the first '
                         'difference writes the cycle line and ABORTS the cell; at k0 the run\'s first residual pass '
                         'must be k0'),
            'post_run': 'G19: every field of per_cycle_record rows 1..k0 bitwise',
            'overlap_report_only': 'k0+1..N_old: Q_new - Q_old and relative (G23)',
            'j_5f3cccb4': 'original first residual pass 159, Boyd lapse at 160, passes 161..170: gate 1..159, holds from 160'},
        'holds_after_first_residual_pass': {
            'AA': 'off (production\'s own off branch), every cycle > k0_run, even across a lapse',
            'tail': 'on', 'rho': 'frozen', 'same_as': 'W101 / W105 / W118 / W132'},
        'captures': {'as_139d1e62': True,
                     'ipopt_exit_by_block': 'resettle_cycle_record.jsonl (51 entries) + all_optimal_k',
                     'v4_rule_record': 'resettle_cycle_record.jsonl settling (+ all_optimal_k, vetoed); decision '
                                       'resettle_decision.json (+ vetoes, out_of_window_reads)',
                     'asserted_before_any_solve': 'assert_resettle_preconditions (child) and parent_capture_checklist'},
        'record_status_label': ('p515_s44_campaign_harness._apply_settling_resettle_status (unchanged since 139d1e62): '
                                'the record status follows the settling decision; production\'s view kept in '
                                'status_production_trajectory'),
        'definitions': DEFINITIONS,
        'scorer': {'claims_dataset': dataset, 'formulas': DEFINITIONS['claims'], 'D_fit': DEFINITIONS['D_fit'],
                   'uncertified_form': DEFINITIONS['uncertified_form'],
                   'functions': ['L132.score_claim', 'L132.resolve', 'L132.d_fit', 'L132.score_predictions',
                                 'score_group1', 'L132.view_from_report', 'L132.reference_views', 'cell_report'],
                   'post_run_prediction_scorer_cell1': 'cell1_bitwise_prediction (G26)'},
        'predictions_recorded_before_any_run': PREDICTIONS,
        'recurring_acceptable_report': recurring,
        'gates': {
            'G1-G7, G9, G11, G14, G16': 'as W101 / W118 / W132 (W86 evaluation_checks; solve profile; G6 v37; lambda)',
            'G8': 'persistence on PRODUCTION\'s certificate (status_production_trajectory), Addendum 59',
            'G13': 'holds inert through the first residual pass, held after', 'G15': 'stopping consistent',
            'G17': 'the pure rule v4 replays the in-cycle records and the decision', 'G18': 'line fields (+ exits, v4)',
            'G19': 'gated: replay 1..k0 bitwise every field', 'G20': 'production counters',
            'G21': 'creep captures complete', 'G22': 't_sum in-cycle == stride formula; terminal identity',
            'G23': 'overlap recorded (gated) / absent (ungated)',
            'G24': 'the 51 exits of every cycle == the network solve records\' final attempts and the ESSO logs',
            'G25': 'the record status follows the settling decision',
            'G26': 'cell 1 only: the Planner\'s Addendum 59 prediction (a failure exits 3)', 'scope': GATE_SCOPE,
            'non_stopping': list(NON_STOPPING_GATES), 'prediction_gates': list(PREDICTION_GATES),
            'post_run_evaluator_and_scorer_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w137_resettle_v4_checks.py', 'committed_output': cf,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()}},
                              'rerun_before_launch': ('--run re-runs every section and refuses unless all hold (the W100 '
                                                      'typing test is in the committed output only)')},
        'verbatim_text': {'quotes': {f'{f}:{k}': v for (f, k), v in VERBATIM.items()}, 'check': verb},
        'labelling_and_identity': {'label': V4.LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key, settling_resettle (v4 declaration)})'
                                                      '; every other key byte-identical to the pre-W137 harness '
                                                      '(checks K), the 38 v3 keys included')},
        'harness_change': ('p515_s44_campaign_harness.py: resettle_hooks_module -- W135\'s prepared branch (router '
                           'commit 1) and the v4 branch (router commit 2); nothing else (checks D); every committed key '
                           'unchanged (checks K)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'walls': {'this_spec_expected_h': wall['total_expected_h'], 'this_spec_worst_case_h': wall['total_worst_case_h'],
                  'basis': 'W132\'s estimate (the same 38 cells and caps), recomputed at this freeze',
                  'cell1_measured_v3_run': {'wall_s': 5605, 'cycles': 213, 'source': 'W133 ed71177e'},
                  'single_run_over_4h': {c: wall['per_cell'][c]['worst_case_h'] for c in V4.CELL_ORDER
                                         if wall['per_cell'][c]['worst_case_h'] > 4.0}},
        'dry_run_command': launch_command(V4.CELL_ORDER[0], specs[V4.CELL_ORDER[0]]['sha256'], preconditions_only=True),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised by checks H (the real wrappers on the v4 state; the real install; the four-way '
                             'dispatch; W132\'s exit wrapper on real production); THE FIRST REAL CYCLE OF CELL 1 IS THE '
                             'SMOKE'),
    }


def freeze_spec(started):
    tag = 'W137-SPEC'
    failures = _common_checks()
    os.makedirs(_abs(ROOT_REL), exist_ok=True)
    existing = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'the stage spec already exists (write-once): {existing}')
    cf = _checks_output_ok(failures)
    specs = cell_spec_state()
    for cell, s in specs.items():
        if 'error' in s or not (s['name_carries_sha'] and s['committed_clean'] and s['checks_all']
                                and s['pre_launch_holds'] and s['root_holds_only_the_spec']):
            failures.append(f'campaign spec of {cell} not frozen / committed / valid: {s}')
        elif s['harness_sha256'] != H.sha256_file(H.HARNESS_PATH):
            failures.append(f'{cell}: harness changed since its campaign spec froze')
        elif _load(s['path'])['extra'].get('code_sha256') != {rel: _sha(rel) for rel in CODE_PINNED}:
            failures.append(f'{cell}: code changed since its campaign spec froze')
    prov = K.K132.production_since_originals(CODE_PINNED)
    if not prov['ok']:
        failures.append(f'uncommitted files this run uses: {prov["uncommitted_files_this_run_uses"]}')
    recurring = recurring_state()
    if not recurring['ok']:
        failures.append(f'the recurring-Acceptable report must be committed, manifest-consistent and as seeded: '
                        f'{recurring}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found: {verb["found_whitespace_normalised"]}')
    solver = W101L.solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    o_section = checks_inline['sections']['O']['result']
    dataset = L132.claims_dataset()
    points = L132.completion_points(dataset)
    refs, ref_inputs = L132.reference_views()
    mem = L.memory_preflight(1)
    wall = L132.wall_time_estimate()
    content = stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section,
                                 dataset, points, refs, ref_inputs, recurring)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor v3 {_sha(PREDECESSOR_REL)[:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator / scorer self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f'[{tag}] claims scored: {dataset["n_claims"]} (skipped, not in the settled set: '
         f'{len(dataset["skipped_not_in_the_settled_set"])})')
    for p in points['report_points']:
        _log(f"[{tag}] completion point after #{p['after_cell_index_1_based']} {p['after_cell']}: {p['items_complete']}")
    for cell in V4.CELL_ORDER:
        w = wall['per_cell'][cell]
        _log(f"[{tag}] {cell}: cap {V4.spec_cap(cell)} expected {w['expected_h']:.2f} h worst {w['worst_case_h']:.2f} h; "
             f"LAUNCH: {content['cells'][cell]['launch_command']}")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h; memory "
         f"at freeze (non-gating): available {mem.get('available_gib')} GiB")
    _finish(0, '-- next: commit, then the dry run on the first cell')


# ======================================================================================================================
#  --run
# ======================================================================================================================
def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W137-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('the stage spec is not committed / clean')
    if ss['pins']['code_sha256'] != {rel: _sha(rel) for rel in ss['pins']['code_sha256']}:
        failures.append('code changed since the stage spec froze')
    pin = ss['pins']['campaign_specs'].get(cell) or {}
    if pin.get('sha256') != spec_sha256:
        failures.append(f'the stage spec pins {pin.get("sha256")} for {cell}, not {spec_sha256}')
    idx = V4.CELL_ORDER.index(cell)
    for prev in V4.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'priority order: {prev} (#{V4.CELL_ORDER.index(prev) + 1}) has no results yet')
    root = campaign_root(cell)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign spec not committed / clean')
    checks = validate_campaign_spec(cell, spec)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'the zero-solve checks do not all hold now -- failing: {_failing_check_items(checks_inline)}')
    pre = pre_launch_assertion(cell, spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["parts"]}')
    cap_ok, cap_checks = parent_capture_checklist(cell, spec)
    if not cap_ok:
        failures.append(f'parent-side capture checklist fails: {cap_checks}')
    solver = W101L.solver_check()
    if not solver['ok'] or solver['sha256'] != ss['pins']['solver']['sha256']:
        failures.append(f'solver path / binary differs: {solver}')
    mem = L.memory_preflight(1)
    _log(f"[{tag}] memory preflight: available {mem.get('available_gib')} GiB required {mem['required_gib']:.2f} GiB -> "
         f"{'PASS' if mem['pass'] else 'REFUSE'}")
    if not mem['pass']:
        failures.append(f"memory preflight REFUSED: {mem.get('available_gib')} < {mem['required_gib']}")
    if failures:
        for fl in failures:
            _log(f'[{tag} PRECONDITION FAILED] {fl}')
        _finish(1)
    entry = spec['candidates'][0]
    if preconditions_only:
        _log(f"[{tag}] every precondition holds: campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256} "
             f"(pinned by the stage spec {ss_rel} sha256={ss_sha}); launch index {idx + 1} of {len(V4.CELL_ORDER)}; "
             f"checks per section {({k: v['holds'] for k, v in checks_inline['sections'].items()})}; pre-launch parts "
             f"{pre['parts']}; parent capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; "
             f"cap {spec['cap']}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} sha256={solver['sha256']}")
        _log(f'[{tag}] STOPPED before the campaign lock and the child (no lock taken, no evaluation, zero solves)')
        _finish(0, 'preconditions-only OK')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cell {cell} (#{idx + 1}) eval_key {entry['eval_key']}; cap {spec['cap']}; lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    try:
        gates, detail, rec = cell_gates(cell, entry, eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
        gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}, {}
    report = None
    try:
        if _decision(eval_dir) is not None and os.path.isfile(os.path.join(eval_dir, 'per_cycle_record.jsonl')):
            report = cell_report(cell, eval_dir, rec)
    except Exception as error:  # noqa: BLE001
        report = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    summ = (rec or {}).get('settling_resettle_summary') or {}
    points = ss['claim_completion_points']['report_points']
    point = next((p for p in points if p['after_cell'] == cell), None)
    stopping = [k for k, v in gates.items() if not v and k not in NON_STOPPING_GATES]
    predictions_failed = [k for k in PREDICTION_GATES if k in gates and not gates[k]]
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'launch_index_1_based': idx + 1, 'utc': _utc(),
               'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'],
               'original_eval_key': V4.CELLS[cell]['orig_eval_key'],
               'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'stopping_gate_failures': stopping,
               'prediction_gate_failures': predictions_failed, 'gate_detail': detail, 'cell_report': report,
               'group1_prediction_criteria_report_only': (score_group1({cell: report}) if isinstance(report, dict)
                                                          and 'error' not in report else None),
               'claim_completion_point': point,
               'pre_launch_assertion': pre, 'parent_capture_checklist': cap_checks, 'memory_preflight_at_run': mem,
               'solver': solver, 'batch_info': batch, 'parent_view': (records[0] or {}).get('parent_view'),
               'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    if summ.get('replay_first_divergence'):
        _log(f"[{tag}] REPLAY DIVERGED -- cell ABORTED: {summ['replay_first_divergence']}")
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] settling v4: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"window {report.get('window')} band_width {report.get('band_width')} s {report.get('s_signed')} "
             f"t_sum(end) {report.get('t_sum_end')} k0_run {report.get('k0_run')} non-Optimal cycles "
             f"{report.get('non_optimal_cycles')} vetoes {report.get('n_vetoes')} cycles {report.get('cycles_run')} "
             f"record status {report.get('record_status')}")
    if predictions_failed:
        _log(f'[{tag}] RECORDED PREDICTION FAILED: {predictions_failed} {detail.get("G26")} -- STOP FOR THE PLANNER')
    if point is not None:
        _log(f"[{tag}] CLAIM-COMPLETION POINT after #{point['after_cell_index_1_based']} {cell}: items "
             f"{point['items_complete']} complete -- report (Addendum 58). Scorer: {summarize_command(cell)}")
    ok = not stopping and _guards_ok(g) and isinstance(report, dict) and 'error' not in report
    code = (3 if predictions_failed else 0) if ok else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  --summarize
# ======================================================================================================================
def summarize(started, after_cell):
    tag = f'W137-SUMMARY-after-{after_cell}'
    ss_rel, ss_sha, ss = load_stage_spec()
    idx = V4.CELL_ORDER.index(after_cell)
    reports, missing = {}, []
    inputs = {}
    for cell in V4.CELL_ORDER[:idx + 1]:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        rel = os.path.relpath(path, REPO)
        if not os.path.isfile(path) or not _committed_clean(rel):
            missing.append(cell)
            continue
        inputs[rel] = _sha(rel)
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, f'w137_summary_after_{idx + 1:02d}_{after_cell}.json')
    if missing or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] cells without committed results {missing} or the summary exists')
        _finish(1)
    refs, ref_inputs = L132.reference_views()
    if ref_inputs != ss['pins']['references_inputs_sha256']:
        _log(f'[{tag} PRECONDITION FAILED] the references changed since the stage spec froze')
        _finish(1)
    dataset = ss['scorer']['claims_dataset']
    views = {c: r.get('view') or L132.view_from_report(r) for c, r in reports.items()}
    scored = []
    for cl in dataset['claims']:
        def view(side):
            s = cl[side]
            return refs[s['ref']] if s['source'] == 'reference' else views.get(s['cell'], {})
        scored.append(L132.score_claim(cl, view('ref'), view('other')))
    d_cells = {V4.CELLS[c]['orig_label']: c for c in V4.CELL_ORDER if V4.CELLS[c]['item'] == 'D'}
    d_out = None
    if all(c in views for c in d_cells.values()):
        unit = _load(L132.W2_TABLE)['candidates']['n7_4h_e1']
        p_cost, e_cost = unit['I_new_power_eur'] / 0.25, unit['I_new_energy_eur'] / 1.0
        x0, un = refs['7aa017f0'], refs['bd504ecf']
        pts = {lb: (un if lb == 'n7_4h_e1' else views[d_cells[lb]]) for lb in L132.D_FIT_NODE7_LABELS}
        vals = {'Q': {lb: x0['Q'] - pts[lb]['Q'] for lb in L132.D_FIT_NODE7_LABELS},
                'Q_cc': {lb: x0['Q_cc'] - pts[lb]['Q_cc'] for lb in L132.D_FIT_NODE7_LABELS}}
        d_out = L132.d_fit(vals, x0['band'], {lb: pts[lb]['band'] for lb in L132.D_FIT_NODE7_LABELS}, p_cost, e_cost,
                           {lb: L132._ep_of(lb) for lb in L132.D_FIT_NODE7_LABELS})
        d_out['statuses'] = {lb: pts[lb]['status'] for lb in L132.D_FIT_NODE7_LABELS}
        d_out['all_points_certified'] = all(v == 'certified' for v in d_out['statuses'].values())
        d_out['first_unit_n7_eur_per_mwh'] = (x0['Q'] - un['Q'] - 0.25 * p_cost) / 1.0
        b = views.get('b_2a0ba8b2')
        d_out['first_unit_best_node_n5_4h_e1_eur_per_mwh'] = ((x0['Q'] - b['Q'] - 0.25 * p_cost) / 1.0) if b else None
        d_out['committed_baseline_tables'] = {k: _load(L132.BASELINE_TABLES)[k] for k in ('node7_fit', 'breakeven')}
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'after_cell': after_cell, 'after_cell_index_1_based': idx + 1,
           'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': DEFINITIONS['objective_convention'],
           'reports': reports, 'references': refs, 'claims_scored': scored,
           'claims_complete': [s['claim_id'] for s in scored if not str(s.get('verdict', '')).startswith('not scored')],
           'D_fit': d_out, 'predictions_scored': L132.score_predictions(reports),
           'group1_prediction_criteria': score_group1(reports), 'definitions': DEFINITIONS,
           'predictions': PREDICTIONS, 'inputs_sha256': {**inputs, **ref_inputs}}
    H._write_once_json(_abs(out_rel), doc)
    man = os.path.join(ROOT_REL, f'w137_summary_after_{idx + 1:02d}_{after_cell}_manifest_sha256.json')
    H._write_once_json(_abs(man), {out_rel: _sha(out_rel), **doc['inputs_sha256']})
    _log(f'[{tag}] wrote {out_rel}; claims complete {len(doc["claims_complete"])}; D fit '
         f'{"in" if d_out else "not yet"}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-cells', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=V4.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--after-cell', choices=V4.CELL_ORDER, default=None)
    parser.add_argument('--preconditions-only', action='store_true',
                        help='with --run: every --run precondition, then stop before the lock and the child')
    args = parser.parse_args()
    if args.preconditions_only and not args.run:
        parser.error('--preconditions-only requires --run')
    started = time.time()
    try:
        if args.freeze_cells:
            freeze_cells(started)
        elif args.freeze_spec:
            freeze_spec(started)
        elif args.summarize:
            if not args.after_cell:
                parser.error('--summarize requires --after-cell')
            summarize(started, args.after_cell)
        else:
            if not (args.spec_sha256 and args.cell):
                parser.error('--run requires --cell and --spec-sha256')
            run(started, args.cell, args.spec_sha256, preconditions_only=args.preconditions_only)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
