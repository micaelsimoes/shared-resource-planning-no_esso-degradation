"""
P5.15 Addendum 61, Planner task W142 -- the v6 RE-SETTLING CAMPAIGN (the 34 cells not yet decided: the eight remaining
break-even-fit D cells first, then claim groups 2 and 4 in the v5 order): the 34 per-cell campaign freezes, the frozen
stage spec `frozen_s53_resettle_spec_v6_<sha8>.json` (the resettle series, predecessor v5 ab32ffc9), the per-cell run
(one cell per call, priority order enforced), and the zero-solve scorer with the report-stage determinacy floor.
BUILT AND FROZEN IN W142; NO RUN IS LAUNCHED BY THE WORKER.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 61 (the swing noise floor tau/10 in the growth test, the turning-point
floor carried iff invariant on replay; the determinacy floor for differences between certified cells; the measured
post-certification movement stated; the window reads enumerated per certificate; the order: v6 freeze -> the eight
remaining break-even-fit cells -> remainder in priority order), Addenda 58-60 (the order, readings gamma and (a), the 10x
clean rule); TASKS.md Addendum 58-61 sections (the pre-registered predictions; the Planner reading that d_c52e1670 is
certified from its committed records under v6); Planner task W142.

WHAT CHANGED FROM THE v5 CAMPAIGN (and nothing else): the settling rule (settling_criterion_v6 via
p515_s53_w142_resettle_v6_hooks; new declaration schema -> new eval keys and campaign roots); G17 replays the v6 rule;
the cell report carries the v6 fields (floor rejections) and names the certifying spec v6; the scorer resolves
differences between certified cells on the determinacy floor (p515_s53_w142_determinacy); the two cells decided since v5
froze are read from their committed records (b_4649234b certified under v5, run 4b3be392; d_c52e1670 certified under v6
FROM RECORDS, W142 item 2) beside the two kept under v4; the expert's Addendum 60 D prediction is recorded FAILED under v5
and the Planner's Addendum 61 prediction "the eight remaining D cells certify under v6" is scored post-run by G27 (D cells
only; a failure exits 3: stop for the Planner). G26 (the cell-3 prediction) is retired with its cell. The cells' order,
caps and ceilings, the configuration, the captures (the v5 clean capture unchanged), the gated replay, the holds and the
rule-independent gates are W132's / W139's (imported).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-cells                        ZERO SOLVES. The 34 campaign specs (write-once), in the v6 stage root.
  --freeze-spec                         ZERO SOLVES. The stage spec (write-once, named by its sha256).
  --run --cell C --spec-sha256 S        THE RUN OF ONE CELL (NOT RUN IN W142). Priority order enforced.
  --run ... --preconditions-only        ZERO SOLVES. Every --run precondition, then STOP before the lock and the child.
  --summarize --after-cell C            ZERO SOLVES. The scorer over every cell with committed results up to C (the four
                                        decided cells read from their committed records).

Exit codes: 0 done (every stopping gate holds; a G6-only failure does not stop); 1 any other gate / harness / guard /
precondition failure; 3 every stopping gate holds but a PREDICTION gate (G27, D cells) failed -- a recorded prediction
failed: STOP FOR THE PLANNER before the next cell.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W142 v6 re-settling launcher (never solves)').install()

import p515_s53_w139_resettle_v5_campaign as L139  # noqa: E402 -- the v5 launcher (rule-independent gates; arms guards)
import p515_s53_w142_resettle_v6_checks as K  # noqa: E402 -- the zero-solve checks (arms its guard)
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402
import p515_s53_w142_determinacy as DET  # noqa: E402 -- the v6 scorer
import settling_criterion_v6 as SC6  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

L132, L137 = L139.L132, L139.L137
H, L, X, W9, W98L, W101L, L118, W131 = L132.H, L132.L, L132.X, L132.W9, L132.W98L, L132.W101L, L132.L118, L132.W131
V = L132.V
V4 = L137.V4
V5 = V6.V5
R = V.R


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(L139.GUARDS) + (('w142_parent', PARENT_GUARD),))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w142_resettle_v6_campaign', 'p515_s53_w142_resettle_ext_v6_campaign') \
    + L139.OWN_PROCESS_SUBSTRINGS
STAGE_TEXT = ('P5.15 Addendum 61, W142 -- v6 re-settling campaign (34 cells: the eight remaining break-even-fit D cells, '
              'claim groups 2 and 4) under the current production configuration: gated cells replayed bitwise against '
              'their original record through the first residual pass k0 (abort on the first divergence), the G cells as '
              'first C2 evaluations; the certifying regime held after the run\'s first residual pass (AA off, tight tail '
              'on, rho frozen); settling rule v6 (v5 -- reading gamma, window (a), certification vetoed while a NON-CLEAN '
              'cycle lies in the last W cycles the test reads, clean within 10x the tail tolerances -- with the swing '
              'noise floor tau/10: swings below it excluded from the "not growing" comparison, and a sign change closing '
              'a swing below it registering no turning point; gap clause; amended monotone branch) until it certifies or '
              'the cap; W105 captures, per-cycle t_sum and Q_cc, the IPOPT exit, final attempt tier and final metrics '
              'of every block of every cycle')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W142_ROOT_REL
SPEC_PREFIX = 'frozen_s53_resettle_spec_v6_'
SPEC_SERIES = 'frozen_s53_resettle_spec'
SPEC_VERSION = 6
PREDECESSOR_REL = K.V5_STAGE_SPEC['path']
PREDECESSOR_SHA256 = 'ab32ffc98d9eb8419871c95178923b40bbf06f5c3672edd9d96c1d0dc99a9072'
CAMPAIGN_IDS = {cell: f'{K.CAMPAIGN_ID_PREFIX}{cell}' for cell in V6.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
TASKS = 'TASKS.md'
SOLVER_PATH = L132.SOLVER_PATH
PYTHON = L132.PYTHON
N_NETWORK_BLOCKS = 48
FROM_RECORDS = K.FROM_RECORDS
FLOOR_REPLAY = K.FLOOR_REPLAY
D_CELL_FROM_RECORDS = K.D_CELL
EXTRA_CLEAN_FILES = tuple(dict.fromkeys(
    (SCRIPT_NAME, 'p515_s53_w142_resettle_v6_hooks.py', 'p515_s53_w142_resettle_v6_checks.py',
     'p515_s53_w142_resettle_ext_v6_hooks.py', 'settling_criterion_v6.py', 'p515_s53_w142_v6_from_records.py',
     'p515_s53_w142_floor_replay.py', 'p515_s53_w142_determinacy.py', 'p515_s53_w141_swing_variants.py',
     'p515_s53_w139_resettle_v5_campaign.py', 'p515_s53_w139_resettle_v5_hooks.py',
     'p515_s53_w139_resettle_v5_checks.py', 'settling_criterion_v5.py') + tuple(L139.EXTRA_CLEAN_FILES)))
CODE_PINNED = tuple(dict.fromkeys(K.CODE_PINNED_BY_CHECKS + (SCRIPT_NAME, 'p515_s53_w139_resettle_v5_campaign.py')
                                  + tuple(L139.CODE_PINNED)))

# ---- verbatim text (checked against the committed brief / TASKS.md, whitespace-normalised, at every freeze) ------------
VERBATIM = dict(L139.VERBATIM)
VERBATIM.update({
    (BRIEF, 'a61_title'): 'swing noise floor; what τ bounds; window reads enumerated',
    (BRIEF, 'a61_floor'): ('Swings below τ/10 are not swings of the oscillation being measured and are excluded from the '
                           'growth comparison'),
    (BRIEF, 'a61_refinement'): ('If that replay changes no certificate, v6 carries both; if it changes any, v6 carries the '
                                'swing floor only and the inconsistency is recorded for the cleanup.'),
    (BRIEF, 'a61_failed'): ('The break-even prediction failed on its first cell for a rule reason, not a solver one — '
                            'recorded as failed under v5.'),
    (BRIEF, 'a61_movement'): '"cells continued past certification moved at most 0.9 τ."',
    (BRIEF, 'a61_determinacy'): ('**Determinacy rule for differences between certified cells: margin ≥ max(3 × the larger '
                                 'bar, 2τ)**'),
    (BRIEF, 'a61_order'): ('v6 freeze (swing floor; turning-point floor if invariant on replay; determinacy floor in the '
                           'report stage) → eight remaining break-even-fit cells (≈ 11 h)'),
    (TASKS, 'a61_planner_reading'): ('**Planner reading:** the order names only the *eight remaining* break-even-fit cells '
                                     '→ **d_c52e1670 is certified from its committed records under v6**'),
})

REFERENCES = L132.REFERENCES
GAP_REFUSED_LABEL = L132.GAP_REFUSED_LABEL
DECIDED = {
    'b_2a0ba8b2': {'kind': 'kept under v4', **V6.KEPT_UNDER_V4['b_2a0ba8b2']},
    'b_0dd237f0': {'kind': 'kept under v4', **V6.KEPT_UNDER_V4['b_0dd237f0']},
    'b_4649234b': {'kind': 'certified under v5 (its run)', **V6.CERTIFIED_UNDER_V5['b_4649234b']},
    'd_c52e1670': {'kind': 'certified under v6 FROM RECORDS', **V6.CERTIFIED_V6_FROM_RECORDS['d_c52e1670']},
}

DEFINITIONS = copy.deepcopy(L139.DEFINITIONS)
DEFINITIONS['per_cell'].update({
    'k_star': 'the cycle the settling rule v6 certified; None when uncertified at its cap',
    'out_of_window_reads': ('at k*: the reads of the certification test outside the certifying window '
                            '(settling_criterion_v6.SUB_TEST_READS applied, the two floor reads included): turning '
                            'points, cycles, whether any is non-clean'),
    'turning_point_floor_rejections': ('every sign change whose closing swing was below TAU / 10 (no turning point '
                                       'registered): the cycle, the candidate, the swing, the point un-registered'),
    'certifying_spec': ('the stage spec under which the cell certified: this spec (v6) for a v6 run; v6 FROM RECORDS for '
                        'd_c52e1670; the v5 spec ab32ffc9 for b_4649234b; the v4 spec e11fbc89 for the two cells kept '
                        'under v4'),
})
DEFINITIONS['swing_floor'] = {'F': SC6.SWING_FLOOR, 'carries': list(SC6.CARRIES), 'evidence': SC6.CARRY_EVIDENCE,
                              'readings': {k: SC6.READINGS[k] for k in ('growth_test_floor_A', 'turning_point_floor_B',
                                                                         'floor_comparison')}}
DEFINITIONS['claims']['resolution_settled_vs_settled'] = (
    'v6 (Addendum 61 ruling 2): both cells certified -- DETERMINATE iff |margin| >= max(3 x the larger of the two band '
    'widths, 2 TAU) (gross verdict; Q_cc beside, report-only, on the same threshold); the superseded v3-v5 rule (the sum '
    'of the two band widths, strict >) is recorded beside every verdict, report-only (p515_s53_w142_determinacy)')
DEFINITIONS['claims']['resolution_with_an_uncertified_cell'] = (
    'the determinacy rule of the uncertified form, UNCHANGED (3 x max(|gap|, |slack|) over the uncertified cell(s), both '
    'terms)')
DEFINITIONS['determinacy_floor'] = {'rule': DET.RULE_TEXT, 'formula': 'settling_criterion_v6.determinate_certified',
                                    'two_tau': 2.0 * SC6.TAU,
                                    'movement_statement': 'cells continued past certification moved at most 0.9 tau'}


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
    _log(f'[W142] guards {g} {extra_msg}')
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
#  predictions (recorded before any v6 run)
# ======================================================================================================================
def _d_cells_v6():
    return [c for c in V6.CELL_ORDER if V6.CELLS[c]['item'] == 'D']


def predictions():
    """The predictions recorded BEFORE any v6 run (each with its source): W139's carried (the expert's Addendum 60 and
    the Planner's cell-3 prediction with their outcomes), and the Planner's Addendum 61 prediction for the eight."""
    out = L139.predictions()
    out['planner_addendum60_cell3'] = dict(out['planner_addendum60_cell3'],
                                           outcome='HELD (W140, 4b3be392: G26 PASS; certified at 148)')
    out['expert_addendum60'] = dict(
        out['expert_addendum60'],
        outcome=('cell 3: HELD (certified under v5 at 148, 4b3be392); the nine D cells: FAILED UNDER v5 on the first '
                 '(d_c52e1670, 51280961: uncertified at its cap 198, 0 vetoes, reasons swings_growing -- a three-cycle '
                 'blip at 111-113 left swings of 226 and 125 EUR and the all-pairs test failed from then on; a rule '
                 'reason, not a solver one; DSO7 2025 Winter did not appear on that cell)'),
        recorded_as='FAILED under v5 (PLANNER_BRIEF_2026-09-13.md Addendum 61: "recorded as failed under v5")')
    out['planner_addendum61_d_cells'] = {
        'statement': 'the eight remaining break-even-fit cells certify under v6',
        'source': 'Planner task W142 item 5 (recorded before any v6 run)',
        'cells': _d_cells_v6(),
        'operationalisation': ('G27 (D cells only): the cell\'s v6 decision is certified (resettle_decision.json status '
                               'certified, version 6); SKIPPED for every other cell; a failure exits 3 (stop for the '
                               'Planner before the next cell)'),
        'scorer': 'd_cell_prediction (this module); score_planner_a61 at the D claim point (report)',
        'context': {'d_c52e1670': 'certified under v6 FROM RECORDS at 150 (W142 item 2) -- not a v6 run, not scored here'}}
    return out


# ======================================================================================================================
#  configuration, entries, keys
# ======================================================================================================================
def configuration(cell):
    cfg = L132.configuration(cell)
    cfg['name'] = cfg['name'].replace('W132 SRP1 RE-SETTLING v3 (Addendum 58)', 'W142 SRP1 RE-SETTLING v6 (Addendum 61)')
    cfg['note'] = cfg['note'].replace('settling_resettle (v3 schema; keyed)', 'settling_resettle (v6 schema; keyed)')
    return cfg


def orig_entry(cell):
    return L132.orig_entry(cell)


def entries(cell):
    (label, nodes, opts), = L132.entries(cell)
    opts = dict(opts)
    opts['settling_resettle'] = V6.declaration_for(cell)
    return [(label, nodes, opts)]


def expected_keys(cell):
    spec, e, kw = K.K132.resettle_kwargs(cell)
    ocfg = spec['configuration']
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V6.declaration_for(cell), **kw)
    key_v5 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V5.declaration_for(cell), **kw)
    key_v4 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V4.declaration_for(cell), **kw)
    key_v3 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V.declaration_for(cell), **kw)
    base_orig = H.evaluation_key(e['key'], e['overrides'], case_file_aa=ocfg.get('case_file_anderson_acceleration'),
                                 ess_ageing_baseline=ocfg.get('ess_ageing_baseline'),
                                 flex_price_multiplier=e.get('flex_price_multiplier'),
                                 convergence_depth_tail=ocfg.get('convergence_depth_tail'))
    return {'base_key_current_configuration': base, 'resettle_key': key, 'resettle_key_v5': key_v5,
            'resettle_key_v4': key_v4, 'resettle_key_v3': key_v3, 'base_key_original_configuration': base_orig,
            'original_eval_key': e['eval_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The v6 re-settling key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it
    appears in no committed campaign spec OUTSIDE the v6 stage root (the rule: a pre-run check that scans committed
    artefacts excludes the run's own); it differs from the v5, v4 and v3 keys of the cell; the original configuration's
    key reproduces the original eval key."""
    k = expected_keys(cell)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    e = orig_entry(cell)
    eval_dir_name = H.eval_dir_name(k['resettle_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['resettle_key'])
    work = L._work_dir()
    entry = next((x for x in (spec or {}).get('candidates') or [] if x['label'] == cell), None)
    parts = {
        'base_key_original_configuration_equals_original_eval_key': k['base_key_original_configuration']
        == e['eval_key'] == V6.CELLS[cell]['orig_eval_key'],
        'resettle_key_differs_from_original_and_base': k['resettle_key'] not in (e['eval_key'],
                                                                                 k['base_key_current_configuration']),
        'resettle_key_differs_from_the_v5_v4_and_v3_keys': k['resettle_key'] not in (
            k['resettle_key_v5'], k['resettle_key_v4'], k['resettle_key_v3']),
        'resettle_key_absent_from_committed_specs_outside_the_v6_root': k['resettle_key'] not in committed,
        'campaign_root_differs_from_original_v3_v4_and_v5': os.path.abspath(campaign_root(cell)) not in (
            os.path.abspath(_abs(V6.CELLS[cell]['orig_root'])), os.path.abspath(L132.campaign_root(cell)),
            os.path.abspath(L137.campaign_root(cell)), os.path.abspath(L139.campaign_root(cell))),
        'eval_dir_name_differs_from_original': eval_dir_name != e['eval_dir'],
        'working_dir_ids_differ_from_original': not (set(ids.values()) & set(e['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['resettle_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    failing = [p for p, v in parts.items() if not v]
    return {'holds': all(parts.values()), 'parts': parts, 'failing': failing, **k, 'eval_dir_name': eval_dir_name,
            'working_dir_ids': ids, 'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(cell, spec):
    """W132's field-by-field campaign-spec check, the declaration and the script being this campaign's."""
    checks = L132.validate_campaign_spec(cell, spec)
    e = (spec['candidates'] or [{}])[0] if len(spec['candidates']) == 1 else {}
    checks.pop('entry_resettle_is_the_v3_declaration', None)
    checks['entry_resettle_is_the_v6_declaration'] = e.get('settling_resettle') == V6.declaration_for(cell)
    checks['campaign_id'] = spec.get('campaign_id') == CAMPAIGN_IDS[cell]
    checks['script_recorded'] = (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME
    return checks


def parent_capture_checklist(cell, spec):
    """The child's capture checklist, asserted in the PARENT too (before the lock)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = V6.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
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
#  walls and the claim-completion points (W132's, on the 34 v6 cells; the decided cells are complete)
# ======================================================================================================================
def wall_time_estimate():
    w = L132.wall_time_estimate()
    per = {c: w['per_cell'][c] for c in V6.CELL_ORDER}
    exp = sum(v['expected_h'] for v in per.values())
    worst = sum(v['worst_case_h'] for v in per.values())
    measured = {}
    for c, d in DECIDED.items():
        if 'source_v5_run_commit' in d:
            continue
        res = _load(os.path.join(d['campaign_root'], RESULTS_FILE))
        measured[c] = {'wall_clock_s': res['wall_clock_s'], 'cycles_run': (res.get('cell_report') or {}).get('cycles_run'),
                       'run_commit': d.get('v4_run_commit') or d.get('v5_run_commit')}
    return {'basis': ('W132\'s estimate (p515_s53_w132_resettle_v3_campaign.wall_time_estimate), restricted to the 34 v6 '
                      'cells; the runs of the decided cells measured beside'),
            'kappa_concurrency_1_over_3': w['kappa_concurrency_1_over_3'], 'overhead_s_per_cell': w['overhead_s_per_cell'],
            'per_cell': per, 'total_expected_h': exp, 'total_worst_case_h': worst,
            'decided_runs_measured': measured, 'w118_calibration_measured_over_estimate':
                w['w118_calibration_measured_over_estimate']}


def completion_points(dataset):
    """W132's rule on the v6 order: a claim is complete after the last of its v6 cells (a claim whose cells are all
    decided -- kept under v4, certified under v5, certified v6 from records -- or references is complete already); an item
    after the last cell of any of its claims (D: after the last D cell); each point also carries its label in the v3
    numbering (the D point is claim point #12)."""
    pos = {c: i for i, c in enumerate(V6.CELL_ORDER)}
    v3_pos = {c: i for i, c in enumerate(V.CELL_ORDER)}
    per_claim = {}
    for cl in dataset['claims']:
        cells = [s['cell'] for s in (cl['ref'], cl['other']) if s['source'] == 'w132' and s['cell'] in pos]
        decided = [s['cell'] for s in (cl['ref'], cl['other']) if s['source'] == 'w132' and s['cell'] in DECIDED]
        per_claim[cl['claim_id']] = {'item': cl['item'], 'cells': cells, 'decided_cells': decided,
                                     'complete_after': (max(cells, key=pos.get) if cells else None)}
    items = {}
    for cid, v in per_claim.items():
        if v['complete_after'] is None:
            continue
        cur = items.get(v['item'])
        if cur is None or pos[v['complete_after']] > pos[cur]:
            items[v['item']] = v['complete_after']
    items['D'] = max(_d_cells_v6(), key=pos.get)
    pending = {}
    for cl in dataset['claims']:
        v = per_claim[cl['claim_id']]
        if cl['old_verdict_R1_w117'] == 'pending' and v['complete_after'] is not None:
            cur = pending.get(cl['item'])
            if cur is None or pos[v['complete_after']] > pos[cur]:
                pending[cl['item']] = v['complete_after']
    labelled = dict(items)
    for item, cell in pending.items():
        if cell != items.get(item):
            labelled[f'{item} (its W117-pending claims)'] = cell
    points = {}
    for item, cell in labelled.items():
        points.setdefault(cell, []).append(item)
    ordered = [{'after_cell_index_1_based': pos[c] + 1, 'after_cell': c, 'items_complete': sorted(points[c]),
                'claim_point_label_v3_numbering': f'#{v3_pos[c] + 1}'}
               for c in sorted(points, key=pos.get)]
    return {'rule': ('a claim is complete after the last of its v6 cells in the launch order (the four decided cells count '
                     'as complete); an item after the last cell of any of its claims (D: after the last D cell, the '
                     'claim point numbered #12 in the v3 order, whose D fit includes d_c52e1670\'s v6-from-records '
                     'certificate); the Planner reports at each point (Addendum 58: "reports at each claim\'s '
                     'completion")'),
            'report_points': ordered, 'per_item': {k: {'after_cell': v, 'index_1_based': pos[v] + 1,
                                                       'claim_point_label_v3_numbering': f'#{v3_pos[v] + 1}'}
                                                   for k, v in labelled.items()},
            'per_claim': per_claim,
            'complete_already': sorted(cid for cid, v in per_claim.items() if v['complete_after'] is None
                                       and v['decided_cells']),
            'd_fit_points': {'d_c52e1670': 'the v6-from-records certificate (W142 item 2)',
                             'n7_4h_e1': 'the settled unit 3f084f2f (reference)',
                             'others': 'the eight v6 D cells'}}


# ======================================================================================================================
#  post-run gates (W132's / W139's rule-independent ones reused; the rule-dependent ones restated for v6)
# ======================================================================================================================
LINE_FIELDS_REQUIRED = L139.LINE_FIELDS_REQUIRED
_decision = L132._decision
hold_checks = L132.hold_checks
stopping_check = L132.stopping_check
replay_gate_full = L132.replay_gate_full
overlap_check = L132.overlap_check
exit_crosscheck = L132.exit_crosscheck
status_label_check = L132.status_label_check
clean_crosscheck = L139.clean_crosscheck
_acceptable_clean_exits = L139._acceptable_clean_exits
DECISION_KEYS_REPLAYED = L139.DECISION_KEYS_REPLAYED + ('swing_floor', 'turning_point_floor_rejections')


def settling_replay_check(cell, eval_dir):
    """The pure rule v6 replayed on the run's per_cycle_record (Q, boyd) with the in-cycle t_sum and all_clean_k
    reproduces every in-cycle rule record and the decision."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V6.CYCLE_FILE))}
    rule = K.pure_rule(V6.declaration_for(cell))
    pure = []
    for r in rows:
        ln = lines.get(r['cycle']) or {}
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok']), ln.get('t_sum'),
                                 bool(ln.get('all_clean_k'))))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k))
                                                      for k in DECISION_KEYS_REPLAYED),
             'decision_file_present': bool(dec), 'decision_is_version_6': dec.get('version') == 6}
    return all(parts.values()), {'parts': parts}


def line_fields_check(eval_dir):
    ok, d = L139.line_fields_check(eval_dir)
    lines = _read_jsonl(os.path.join(eval_dir, V6.CYCLE_FILE))
    miss = [x['cycle'] for x in lines if x.get('settling') is not None and x['settling'].get('eligible')
            and 'turning_point_floor_rejection' not in x['settling']]
    d = dict(d, v6_floor_field_missing_cycles=miss[:10])
    return bool(ok and not miss), d


def d_cell_prediction(cell, eval_dir):
    """G27 -- the Planner's Addendum 61 prediction for a D cell, scored post-run: the v6 decision is certified."""
    dec = _decision(eval_dir) or {}
    parts = {'decision_is_version_6': dec.get('version') == 6, 'certified': dec.get('status') == 'certified'}
    return all(parts.values()), {'parts': parts, 'status': dec.get('status'), 'k_star': dec.get('k_star'),
                                 'reasons': dec.get('reasons'), 'cell': cell}


GATE_SCOPE = {k: v for k, v in L139.GATE_SCOPE.items() if not k.startswith('G26')}
PREDICTION_GATE = 'G27_planner_a61_d_cell_certifies_under_v6'
GATE_SCOPE[PREDICTION_GATE] = ('the eight D cells ONLY -- the Planner\'s recorded Addendum 61 prediction; SKIPPED for '
                               'every other cell; a failure exits 3 (stop for the Planner)')
PREDICTION_GATES = (PREDICTION_GATE,)
NON_STOPPING_GATES = ('G6_v37_optimal_and_four_metrics',) + PREDICTION_GATES


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
    gates['G8_persistence_on_production_certificate'], detail['G8'] = V6.persistence_check_production_certificate(
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
    gates['G17_rule_v6_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    if V6.CELLS[cell]['gated']:
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
    for name, fn in (('G24_exit_capture_equals_the_solve_records_and_esso_logs', lambda: exit_crosscheck(eval_dir,
                                                                                                          rec=rec)),
                     ('G24b_clean_capture_equals_the_solve_records_and_esso_logs', lambda: clean_crosscheck(eval_dir))):
        try:
            gates[name], detail[name.split('_')[0]] = fn()
        except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
            gates[name] = False
            detail[name.split('_')[0]] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    gates['G25_record_status_follows_the_settling_decision'], detail['G25'] = status_label_check(rec, eval_dir)
    if V6.CELLS[cell]['item'] == 'D':
        try:
            gates[PREDICTION_GATE], detail['G27'] = d_cell_prediction(cell, eval_dir)
        except Exception as error:  # noqa: BLE001 -- recorded; the prediction gate FAILS
            gates[PREDICTION_GATE] = False
            detail['G27'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    else:
        detail['G27'] = {'skipped': 'D cells only (the Planner\'s Addendum 61 prediction)'}
    return gates, detail, rec


def cell_report(cell, eval_dir, rec):
    """W139's per-cell report (on W132's) with the v6 fields and the certifying spec v6."""
    rep = L139.cell_report(cell, eval_dir, rec)
    dec = _decision(eval_dir) or {}
    rep.update({'criterion_version': dec.get('version'), 'reading': dec.get('reading'),
                'swing_floor': dec.get('swing_floor'),
                'turning_point_floor_rejections': dec.get('turning_point_floor_rejections'),
                'certifying_spec': ({'series': SPEC_SERIES, 'version': SPEC_VERSION}
                                    if dec.get('status') == 'certified' else None)})
    rep['view'] = L132.view_from_report(rep)
    return rep


score_group1 = L139.score_group1


def score_planner_a61(reports):
    """The Planner's Addendum 61 prediction (report at the D point): each of the eight D cells certified under v6."""
    out = {}
    for cell in _d_cells_v6():
        r = reports.get(cell)
        out[cell] = ({'held': None, 'status': 'no result yet'} if r is None else
                     {'held': r.get('status') == 'certified', 'status': r.get('status'), 'k_star': r.get('k_star'),
                      'n_floor_rejections': len(r.get('turning_point_floor_rejections') or [])})
    done = [v for v in out.values() if v['held'] is not None]
    return {'per_cell': out, 'held_so_far': all(v['held'] for v in done) if done else None,
            'complete': len(done) == len(out)}


# ======================================================================================================================
#  post-run evaluator self-tests
# ======================================================================================================================
def _synthetic_run_dir(tmp, cell, variant):
    """W139's `_synthetic_run_dir` on the v6 drive (K.drive: the real nine wrappers on the v6 state) and the v6 rule."""
    plan = None
    if variant == 'acceptable_clean':
        plan = {V6.CELLS[cell]['N_old'] + 2: {'ESSO|9': 'esso_acc_within'}}
    elif variant == 'recovery_non_clean':
        plan = {V6.CELLS[cell]['N_old'] + 2: {'ESSO|9': 'esso_acc_within@recovery'}}
    d = K.drive(cell, 'creep' if variant == 'creep' else 'certify', plan=plan)
    files, st = d['files'], d['state']
    lines = {x['cycle']: x for x in files[V6.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[V6.CREEP_FILE]}
    if variant == 'tamper_t_sum_30':
        lines[30]['t_sum'] = lines[30]['t_sum'] + 1.0
    if variant == 'tamper_hold_flag':
        lines[max(lines) - 5]['aa']['hold'] = False
    if variant == 'tamper_decision':
        files[V6.DECISION_FILE][0]['k_star'] = files[V6.DECISION_FILE][0]['k_star'] - 1
    if variant == 'tamper_clean':
        lines[max(lines) - 1]['all_clean_k'] = False
    if variant == 'decision_uncertified':
        files[V6.DECISION_FILE][0]['status'] = 'uncertified'
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == V6.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            elif fname == V6.CYCLE_FILE:
                for c in sorted(lines):
                    handle.write(GRIO.dumps(lines[c], default=GRIO.json_default) + '\n')
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    ref = st.reference if st.gated else {}
    rows, g_rows = [], []
    g_orig = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(V6.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if st.gated else {})
    for c in sorted(lines):
        x = lines[c]
        if st.gated and c <= V6.CELLS[cell]['N_old']:
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


def _determinacy_self_tests():
    """The v6 scorer on synthetic views: the 2 TAU term binds (a margin that the superseded sum-of-bands rule calls
    determinate is within resolution under v6); the 3 x larger-bar term binds; an uncertified cell keeps the
    uncertified form exactly (resolve_v6 == L132.resolve)."""
    a = {'status': 'certified', 'band': 1000.0, 'Q': 1.0, 'Q_cc': 1.0}
    b = {'status': 'certified', 'band': 1500.0, 'Q': 1.0, 'Q_cc': 1.0}
    r1 = DET.resolve_v6(8000.0, 8000.0, (a, b))
    c = {'status': 'certified', 'band': 4400.0}
    r2 = DET.resolve_v6(13000.0, 13000.0, (c, dict(c, band=4200.0)))
    r3 = DET.resolve_v6(14000.0, 14000.0, (c, dict(c, band=4200.0)))
    u = {'status': 'uncertified', 'band': 400.0, 'gap': 9234.42, 'slack': 1461.49}
    r4 = DET.resolve_v6(103578.5, 112885.7, (c, u))
    r4_old = L132.resolve(103578.5, 112885.7, (c, u))
    out = {'two_tau_binds': {'ok': r1['verdict'] == 'within resolution' and r1['binding_term'] == '2 TAU'
                             and r1['superseded_rule_report_only']['verdict'] == 'determinate', 'r': r1},
           'three_x_bar_binds_below': {'ok': r2['verdict'] == 'within resolution'
                                       and r2['binding_term'] == '3 x larger bar', 'threshold': r2['threshold']},
           'three_x_bar_binds_above': {'ok': r3['verdict'] == 'determinate', 'threshold': r3['threshold']},
           'uncertified_form_unchanged': {'ok': {k: v for k, v in r4.items() if k != 'rule_v6'} == r4_old,
                                          'r': r4}}
    return out, all(v['ok'] for v in out.values())


def post_run_evaluator_self_tests():
    """The v6 post-run evaluators (G13, G15, G17, G18, G19, G22, G23, G25, G27, the cell report) on synthetic eval dirs
    built by the real wrappers on the v6 state, with tampered negative controls; G8 (W137's self-test on W133's committed
    cell 1); G24b on the committed v4 cell-3 run (W139's, the capture unchanged); the determinacy scorer; W132's G24 and
    scorer self-tests."""
    out = {}
    d_cell = _d_cells_v6()[0]
    cases = ((d_cell, 'pass', None, True), (d_cell, 'tampered_row_40', 'tamper_row_40', False),
             (d_cell, 'tampered_t_sum_30', 'tamper_t_sum_30', False),
             (d_cell, 'tampered_hold_flag', 'tamper_hold_flag', False),
             (d_cell, 'tampered_decision', 'tamper_decision', False),
             (d_cell, 'g27_fails_on_an_uncertified_decision', 'decision_uncertified', False),
             ('h_74eda68d', 'acceptable_clean_exit_passes', 'acceptable_clean', True),
             ('h_74eda68d', 'recovery_non_clean_exit_passes', 'recovery_non_clean', True),
             ('h_74eda68d', 'tampered_clean_flag', 'tamper_clean', False),
             ('g_37b5c499', 'ungated_rule_cap', 'creep', True))
    for cell, name, variant, expect in cases:
        tmp = tempfile.mkdtemp(prefix='w142_selftest_')
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
            rg = replay_gate_full(cell, tmp) if V6.CELLS[cell]['gated'] else {'bitwise_through_k0': True}
            p_ok = d_cell_prediction(cell, tmp)[0] if V6.CELLS[cell]['item'] == 'D' else True
            allg = {'holds': h_ok, 'stopping': s_ok, 'rule_replay': k_ok, 'line_fields': f_ok, 'creep': c_ok,
                    't_sum': t_ok, 'overlap': o_ok, 'status_label': l_ok, 'replay_full': rg['bitwise_through_k0'],
                    'g27_prediction': p_ok}
            if expect:
                ok = all(allg.values())
            elif variant == 'tamper_row_40':
                ok = (not rg['bitwise_through_k0']) and rg['first_divergence_cycle'] == 40 and all(
                    v for k_, v in allg.items() if k_ != 'replay_full')
            elif variant == 'tamper_t_sum_30':
                ok = (not t_ok) and t_d['max_abs_diff_vs_stride'] >= 0.99
            elif variant == 'tamper_hold_flag':
                ok = (not h_ok) and all(v for k_, v in allg.items() if k_ != 'holds')
            elif variant == 'tamper_clean':
                ok = (not k_ok) and all(v for k_, v in allg.items() if k_ != 'rule_replay')
            elif variant == 'decision_uncertified':
                ok = (not p_ok) and (not k_ok)
            else:
                ok = (not k_ok) and (not s_ok)
            rep = cell_report(cell, tmp, rec) if expect else None
            out[name] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                         'report_on_synthetic': ({k: rep.get(k) for k in ('status', 'k_star', 'branch', 'band_width',
                                                                          's_signed', 't_sum_end', 'k0_run',
                                                                          'non_clean_cycles', 'n_vetoes',
                                                                          'criterion_version', 'certifying_spec',
                                                                          'turning_point_floor_rejections', 'label')}
                                                 if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    for name, fn in (('G8_on_the_committed_v3_cell1_record', L137._g8_self_tests),
                     ('G24b_on_the_committed_v4_cell3_run', L139._g24b_self_tests),
                     ('determinacy_scorer', _determinacy_self_tests)):
        try:
            r = fn()
            out[name] = {'ok': r[1], 'tests': r[0]} if isinstance(r, tuple) else r
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    w132, w132_ok = L132.post_run_evaluator_self_tests()
    out['w132_post_run_and_scorer_self_tests_reused'] = {
        'ok': bool(w132_ok), 'per_test': {k: v.get('ok') for k, v in w132.items()},
        'g24': w132.get('G24_exit_crosscheck_on_committed_w118_cell'), 'scorer': w132.get('scorer')}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  checks state, verbatim, the from-records outputs, the decided cells, common preconditions
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


def from_records_state():
    rels = (FROM_RECORDS['path'], FROM_RECORDS['certificate'], FROM_RECORDS['manifest'], FLOOR_REPLAY['path'],
            FLOOR_REPLAY['manifest'])
    if not all(os.path.isfile(_abs(r)) for r in rels):
        return {'ok': False, 'missing': [r for r in rels if not os.path.isfile(_abs(r))]}
    man = _load(FROM_RECORDS['manifest'])
    rman = _load(FLOOR_REPLAY['manifest'])
    sha, sha_c, sha_r = _sha(FROM_RECORDS['path']), _sha(FROM_RECORDS['certificate']), _sha(FLOOR_REPLAY['path'])
    doc, cert, replay = _load(FROM_RECORDS['path']), _load(FROM_RECORDS['certificate']), _load(FLOOR_REPLAY['path'])
    return {'ok': bool(man.get(FROM_RECORDS['path']) == sha and man.get(FROM_RECORDS['certificate']) == sha_c
                       and rman.get(FLOOR_REPLAY['path']) == sha_r and all(_committed_clean(r) for r in rels)
                       and replay['decision']['v6_carries'] == 'A_and_B'
                       and doc.get('item1_v6_equals_replay_arm_AB_every_record') is True
                       and cert.get('k_star') is not None and cert.get('label') == 'v6 from records'
                       and doc['item3_determinacy_rescore']['not_reproduced'] == []),
            'from_records': {'path': FROM_RECORDS['path'], 'sha256': sha, 'manifest': FROM_RECORDS['manifest']},
            'certificate': {'path': FROM_RECORDS['certificate'], 'sha256': sha_c},
            'floor_replay': {'path': FLOOR_REPLAY['path'], 'sha256': sha_r, 'manifest': FLOOR_REPLAY['manifest']},
            'replay_decision': replay['decision'], 'certificate_summary': {
                k: cert.get(k) for k in ('label', 'k_star', 'window', 'W', 'P_hat', 'band', 'band_width',
                                         'range_over_tau', 't_sum_k_star', 'Q_k_star', 'turning_point_floor_rejections',
                                         'all_clean_crosscheck')},
            'certificate_out_of_window_reads': cert.get('out_of_window_reads'),
            'determinacy_rescore': doc['item3_determinacy_rescore'], 'movement_record': doc['item3_movement_record'],
            'v6_on_every_record': {r: v['v6'] for r, v in doc['item1_v6_on_every_record'].items()}}


def decided_reports():
    """The committed reports of the four decided cells: the two kept under v4 (their v4 results), b_4649234b (its v5
    run's results), d_c52e1670 (its v6-from-records certificate record)."""
    out, inputs = {}, {}
    for cell, d in DECIDED.items():
        if cell == D_CELL_FROM_RECORDS:
            rel = d['certificate_record']
            cert = _load(rel)
            rep = dict(cert['report'])
            rep['certifying_spec'] = dict(rep['certifying_spec'], certificate_record=rel)
            inputs[rel] = _sha(rel)
            out[cell] = rep
            continue
        rel = os.path.join(d['campaign_root'], RESULTS_FILE)
        res = _load(rel)
        rep = dict(res.get('cell_report') or {})
        version = 4 if d['kind'] == 'kept under v4' else 5
        rep['certifying_spec'] = ({'series': SPEC_SERIES, 'version': version, 'stage_spec': res.get('stage_spec')}
                                  if rep.get('status') == 'certified' else None)
        rep['run_commit'] = d.get('v4_run_commit') or d.get('v5_run_commit')
        out[cell] = rep
        inputs[rel] = _sha(rel)
    return out, inputs


def _common_checks():
    failures = []
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    for cell in V6.CELL_ORDER:
        rel = V6.reference_path(cell)
        if _sha(rel) != V6.CELLS[cell]['per_cycle_record_sha256'] or not _committed_clean(rel):
            failures.append(f'{cell} original record not as committed: {rel}')
    decided_files = []
    for cell, d in DECIDED.items():
        if cell == D_CELL_FROM_RECORDS:
            decided_files.append(d['certificate_record'])
            ev = os.path.join(d['campaign_root'], 'evals', d['eval_dir'])
        else:
            ev = os.path.join(d['campaign_root'], 'evals', d['eval_dir'])
        decided_files += [os.path.join(d['campaign_root'], RESULTS_FILE), os.path.join(ev, 'per_cycle_record.jsonl'),
                          os.path.join(ev, V6.CYCLE_FILE), os.path.join(ev, V6.DECISION_FILE)]
    for rel in (PREDECESSOR_REL, K.K137.W131['path'], K.K132.W117['path'], L132.W118_SUMMARY, L132.BASELINE_TABLES,
                L132.W2_TABLE) + tuple(decided_files):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    if _sha(PREDECESSOR_REL) != PREDECESSOR_SHA256:
        failures.append('the predecessor stage spec v5 is not ab32ffc9')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a W142 / W139 / W137 / W135 / W132 / W118 / W105 / W101 / W98 launcher is '
                        f'alive: {others}')
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
    tag = 'W142-FREEZE-CELLS'
    failures = _common_checks()
    cf = _checks_output_ok(failures)
    fr = from_records_state()
    if not fr['ok']:
        failures.append(f'the v6-from-records outputs must be committed and consistent: {fr}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    for cell in V6.CELL_ORDER:
        failures += H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)
        pre = pre_launch_assertion(cell)
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["failing"]}')
    if os.path.isdir(_abs(ROOT_REL)) and any(f.startswith(SPEC_PREFIX) for f in os.listdir(_abs(ROOT_REL))):
        failures.append('a stage spec already exists: the cell specs are frozen BEFORE the stage spec')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    all_ok = True
    checks_pin = {'path': cf['path'], 'sha256': cf['sha256']}
    for cell in V6.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        c = V6.CELLS[cell]
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ('frozen_s53_resettle_spec_v6 (frozen AFTER this campaign spec) pins this spec by its '
                                'sha256 and holds the exact launch command'),
                 'label': V6.LABEL, 'cell': cell, 'item': c['item'], 'claim_group': V6.GROUP_OF_ITEM[c['item']],
                 'original': {'campaign_id': c['orig_campaign_id'], 'eval_key': c['orig_eval_key'],
                              'eval_dir': V6.original_eval_dir(cell), 'N_old': c['N_old'], 'k0': c['k0']},
                 'v5_cell': {'campaign_root': L139.campaign_root_rel(cell), 'eval_key_v5': pre['resettle_key_v5'],
                             'stage_spec_v5': {'path': PREDECESSOR_REL, 'sha256': PREDECESSOR_SHA256}},
                 'v4_cell': {'campaign_root': L137.campaign_root_rel(cell), 'eval_key_v4': pre['resettle_key_v4']},
                 'v3_cell': {'campaign_root': L132.campaign_root_rel(cell), 'eval_key_v3': pre['resettle_key_v3']},
                 'expected_eval_key': pre['resettle_key'], 'objective_convention': DEFINITIONS['objective_convention'],
                 'solve_claim': {'parent': 'never solves (every launcher guard permitted=(), verify(0))',
                                 'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
                 'zero_solve_checks_output': checks_pin, 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(V6.CELL_ORDER), 'decided_before_v6': sorted(DECIDED),
                 'code_sha256': {rel: _sha(rel) for rel in CODE_PINNED}}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=V6.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 61', f'{BRIEF} Addenda 58-60', 'Planner task W142'],
            required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec)
        pre_frozen = pre_launch_assertion(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds']
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha} eval_key={e["eval_key"]} '
             f'(v5 {pre["resettle_key_v5"][:16]}) cap={spec["cap"]} checks={all(checks.values())} '
             f'failing={[k for k, v in checks.items() if not v]} pre-launch={pre_frozen["holds"]}')
    _finish(0 if all_ok else 1, f'freeze-cells {"OK" if all_ok else "NOT OK"} -- next: commit, then --freeze-spec')


def cell_spec_state():
    """{cell: {path, sha256, committed_clean, checks, pre_launch}} of the 34 frozen v6 campaign specs."""
    out = {}
    for cell in V6.CELL_ORDER:
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
                     'eval_key': spec['candidates'][0]['eval_key'], 'eval_key_v5': pre['resettle_key_v5'],
                     'eval_key_v4': pre['resettle_key_v4'], 'eval_key_v3': pre['resettle_key_v3'],
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
        raise RuntimeError('frozen v6 re-settling stage spec not found')
    return rel, sha, _load(rel)


def stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section, dataset, points,
                       refs, ref_inputs, fr, preds, decided, decided_inputs):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    v5_ss = _load(PREDECESSOR_REL)
    cells = {}
    for i, cell in enumerate(V6.CELL_ORDER):
        c = V6.CELLS[cell]
        o = o_section['cells'][cell]
        s = specs[cell]
        cells[cell] = {
            'launch_index_1_based': i + 1, 'v5_launch_index_1_based': v5_ss['cells'][cell]['launch_index_1_based'],
            'item': c['item'], 'claim_group': V6.GROUP_OF_ITEM[c['item']],
            'gated': c['gated'], 'prefix': c['orig_eval_key'][:8],
            'original': {**o['original'], 'label': c['orig_label']},
            'm_flex_price_multiplier': c['flex_price_multiplier'] if c['flex_price_multiplier'] is not None else 1.0,
            'investment_year': o['investment_year'], 'canonical': o['canonical'], 'candidate_key': o['candidate_key'],
            'N_old': c['N_old'], 'k0_original_first_residual_pass': c['k0'],
            'original_lapses_after_k0': c['original_lapses_after_k0'], 'Q_N_old': o['Q_N_old'],
            'cap_rule': V6.cap_rule(cell), 'spec_cap': V6.spec_cap(cell), 'cap_ceiling': c['cap_ceiling'],
            'e_over_p': o['e_over_p'], 'lattice_e_over_p_legal': o['parts']['lattice_e_over_p_in_2_4_every_storage_node'],
            'I_cited': o['I'], 'I_source': o['I_source'], 't_sum_terminal_original': o['t_sum_terminal_original'],
            'dead_zone_candidate': o['dead_zone_candidate'], 'dead_zone_borderline': o['dead_zone_borderline'],
            'identity_vs_original': o['identity_vs_original'],
            'declaration': V6.declaration_for(cell), 'campaign_id': CAMPAIGN_IDS[cell],
            'campaign_root': campaign_root_rel(cell), 'configuration': configuration(cell), 'keys': expected_keys(cell),
            'v5': {'campaign_root': L139.campaign_root_rel(cell), 'campaign_spec': v5_ss['pins']['campaign_specs'][cell],
                   'eval_key': v5_ss['cells'][cell]['campaign_spec']['eval_key'], 'run': None},
            'campaign_spec': {k: s[k] for k in ('path', 'sha256', 'eval_key', 'eval_dir', 'harness_sha256', 'git_head')},
            'launch_command': launch_command(cell, s['sha256']),
            'preconditions_only_command': launch_command(cell, s['sha256'], preconditions_only=True),
            'expected_wall_time': wall['per_cell'][cell],
            'prediction_gate': preds['planner_addendum61_d_cells'] if c['item'] == 'D' else None}
    return {
        'schema': 'p515_s53_resettle_spec_v6', 'series': SPEC_SERIES, 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 5, 'path': PREDECESSOR_REL, 'sha256': _sha(PREDECESSOR_REL),
                        'status': ('the W139 v5 campaign spec; cells 1-2 ran under it (W140: 4b3be392 cell 3 CERTIFIED at '
                                   '148; 51280961 d_c52e1670 UNCERTIFIED at its cap 198 -- certified under v6 FROM '
                                   'RECORDS at 150, W142 item 2); superseded for the remaining 34 cells by this v6 spec '
                                   '(Addendum 61)')},
        'authority': [f'{BRIEF} Addendum 61 (the swing noise floor tau/10; the turning-point floor carried iff invariant on '
                      f'replay; the determinacy floor; the movement statement; the window reads enumerated; the order)',
                      f'{BRIEF} Addenda 58-60 (the order; reading gamma; window (a); the 10x clean rule)',
                      'TASKS.md Addendum 58-61 sections (the Planner reading: d_c52e1670 certified from records)',
                      'Planner task W142'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'pins': {'code_sha256': code, 'production_sha256': production, 'solver': solver,
                 'zero_solve_checks_output': {'path': cf['path'], 'sha256': cf['sha256'], 'manifest': cf['manifest']},
                 'v6_from_records': fr['from_records'], 'd_c52e1670_certificate': fr['certificate'],
                 'floor_replay': fr['floor_replay'],
                 'w131': {'path': K.K137.W131['path'], 'sha256': K.K137.W131['sha256']},
                 'w117': {'path': K.K132.W117['path'], 'sha256': K.K132.W117['sha256']},
                 'references_inputs_sha256': ref_inputs, 'decided_inputs_sha256': decided_inputs,
                 'baseline_tables': {'path': L132.BASELINE_TABLES, 'sha256': _sha(L132.BASELINE_TABLES)},
                 'w2_table': {'path': L132.W2_TABLE, 'sha256': _sha(L132.W2_TABLE)},
                 'campaign_specs': {c: {'path': specs[c]['path'], 'sha256': specs[c]['sha256']} for c in V6.CELL_ORDER}},
        'production_since_originals': prov,
        'cells': cells, 'cell_order': list(V6.CELL_ORDER),
        'cell_count_note': ('34 cells = the v5 campaign\'s 36 less the two decided since v5 froze: b_4649234b (certified '
                            'under v5 by its run, 4b3be392) and d_c52e1670 (certified under v6 FROM RECORDS, W142 item 2); '
                            'the eight remaining D cells first, then group 2 (8) and group 4 (18) in the v5 order'),
        'decided_before_v6': {
            c: {**{k: v for k, v in d.items()}, 'report_committed': {k: decided[c].get(k) for k in (
                'status', 'k_star', 'band_width', 's_signed', 'cycles_run', 'label', 'certifying_spec')}}
            for c, d in DECIDED.items()},
        'd_c52e1670_v6_from_records': {'certificate': fr['certificate'], 'summary': fr['certificate_summary'],
                                       'out_of_window_reads': fr['certificate_out_of_window_reads'],
                                       'label': 'v6 from records',
                                       'reading': ('Planner reading of Addendum 61 (TASKS.md): the order names only the '
                                                   'eight remaining D cells; the cell is certified from its committed v5 '
                                                   'run (51280961) under v6, not re-run; it enters the D fit at claim '
                                                   'point #12')},
        'swing_floor_decision': {'replay': fr['floor_replay'], 'decision': fr['replay_decision'],
                                 'v6_on_every_committed_record': fr['v6_on_every_record']},
        'determinacy_floor': {**DEFINITIONS['determinacy_floor'], 'rescore_of_every_claim_scored_so_far':
                              fr['determinacy_rescore'], 'movement_record': fr['movement_record']},
        'same_as_v5': {'cells_order_caps_ceilings_configuration_captures': True,
                       'checked': ('cell table imported from p515_s53_w139_resettle_v5_hooks (pinned ab32ffc9) less the '
                                   'two decided cells; configuration and entry checked field by field against W132\'s '
                                   '(validate_campaign_spec); the declaration differs only in schema, label and '
                                   'settling_rule (module, class, version, the swing floor)')},
        'not_in_this_spec': {'group_3_ageing_E_and_pb_y2025_n5': ('frozen_s53_resettle_ext_spec_v3 (the W142 extension, '
                                                                  'frozen against this spec; after its last cell)'),
                             'decided_before_v6': sorted(DECIDED),
                             'covered_by_settled_references': {'7aa017f0': 'x0 settled d110bd1a',
                                                               'bd504ecf': 'unit settled 3f084f2f'}},
        'launch_order': {'order': list(V6.CELL_ORDER), 'enforced': 'every earlier cell has results before --run',
                         'one_cell_per_call': True},
        'launch_commands': {c: cells[c]['launch_command'] for c in V6.CELL_ORDER},
        'summarize_commands_at_the_completion_points': {p['after_cell']: summarize_command(p['after_cell'])
                                                        for p in points['report_points']},
        'claim_completion_points': points,
        'inputs_in_force_now': o_section['inputs_now'], 'references': refs,
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '+ tight tail {enabled True, compl_inf_tol 1e-6} declared'),
                          'tail_rule': 'production: the tail acts from AA-off + 1 (next-state = this cycle converged)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as W101/W118/'
                                         'W132/W137/W139)',
                          'concurrency': CONCURRENCY, 'option_b_release_solution_bookkeeping': 'absent',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze-cells, --freeze-spec and --run'},
        'stop_rule': {
            'module': 'settling_criterion_v6', 'class': 'settling_criterion_v6.SettlingRuleV6', 'version': SC6.VERSION,
            'versions_1_to_5_unchanged': ('settling_criterion.py, _v2.py .. _v5.py byte-identical (checks V0)'),
            'constants': SC6.constants(V6.P_MAX), 'readings': SC6.READINGS, 'algorithm': SC6.__doc__,
            'failing_reasons': list(SC6.FAILING_REASONS), 'swing_floor': DEFINITIONS['swing_floor'],
            'clean_rule': DEFINITIONS['clean_rule'],
            'p_max': {'value': V6.P_MAX, 'L': V6.L_MONO, 'source': 'W102 x0 P_hat 29, W103 unit P_hat 30 (as fc791891)'},
            'W': {'oscillatory': 'W = max(W_MIN, ceil(W_FACTOR * P_hat)) = max(20, ceil(1.1 * (T[-1].t - T[-3].t)))',
                  'monotone': 'L = L_MONO = 2 * P_MAX = 60',
                  'certifying_window': 'oscillatory [k - W + 1, k]; monotone [k - L + 1, k] (reading (a), unchanged)'},
            'retry_tier': None,
            'sub_test_reads': SC6.SUB_TEST_READS,
            'sub_test_reads_assertion': ('NOT ASSERTED (as v4 / v5; Addendum 61 ruling 3 accepts the enumeration): every '
                                         'certification records its out-of-window reads, the two floor reads included'),
            'N_and_holds_and_dynamic_cap': 'keyed on the first residual pass under the version-2 definition',
            'caps': {'gated': 'N_old + 100 (fixed; the per-cell ceiling in the cell table)',
                     'ungated': 'min(k0_run + 109, 300) (dynamic)',
                     'above_300': {'l_2ab0ce2d': 437, 'l_b2251bc5': 320}},
            'all_clean_k_capture': 'the v5 exit wrapper (p515_s53_w139_resettle_v5_hooks.make_exit_wrapper), unchanged',
            'boyd_k': 'boyd_metrics[\'all_boyd_pass\'] AND local_solves_ok, from THIS cycle (AA wrapper)',
            'Q_k': 'gross_operational_cost (recourse wrapper); None when any local solve failed',
            't_sum_k': 'as fc791891 / 139d1e62 / e11fbc89 / ab32ffc9 (in-cycle; validated post-run, G22)',
            'mechanism': ('decision in the recourse wrapper; certification (or the dynamic cap of an ungated cell below '
                          'the spec cap) sets the certificate length to 0 -> production\'s own exit test ends the loop '
                          'at the end of THIS cycle; the decision file is written once'),
            'early_stop': 'ABSENT (validator refuses the key; checklist)'},
        'replay_gate': {
            'applies_to': list(V6.GATED_CELLS), 'skipped_for': list(V6.UNGATED_CELLS),
            'in_cycle': ('every cycle k <= k0 (the original first residual pass), at the end of the cycle: '
                         + ', '.join(R.REPLAY_GATED_FIELDS) + ' as JSON text against the ORIGINAL record row k; the first '
                         'difference writes the cycle line and ABORTS the cell; at k0 the run\'s first residual pass '
                         'must be k0'),
            'post_run': 'G19: every field of per_cycle_record rows 1..k0 bitwise',
            'overlap_report_only': 'k0+1..N_old: Q_new - Q_old and relative (G23)'},
        'holds_after_first_residual_pass': {
            'AA': 'off (production\'s own off branch), every cycle > k0_run, even across a lapse',
            'tail': 'on', 'rho': 'frozen', 'same_as': 'W101 / W105 / W118 / W132 / W137 / W139'},
        'captures': {'as_ab32ffc9': True,
                     'v6_rule_record': ('resettle_cycle_record.jsonl settling (+ all_clean_k, vetoed, '
                                        'turning_point_floor_rejection); decision resettle_decision.json (+ vetoes, '
                                        'out_of_window_reads, clean_rule, swing_floor, turning_point_floor_rejections)'),
                     'asserted_before_any_solve': 'assert_resettle_preconditions (child) and parent_capture_checklist'},
        'record_status_label': ('p515_s44_campaign_harness._apply_settling_resettle_status (unchanged since 139d1e62)'),
        'definitions': DEFINITIONS,
        'scorer': {'claims_dataset': dataset, 'formulas': DEFINITIONS['claims'], 'D_fit': DEFINITIONS['D_fit'],
                   'uncertified_form': DEFINITIONS['uncertified_form'],
                   'determinacy': DEFINITIONS['determinacy_floor'],
                   'functions': ['p515_s53_w142_determinacy.score_claim_v6', 'p515_s53_w142_determinacy.resolve_v6',
                                 'L132.d_fit', 'L132.score_predictions', 'score_group1', 'score_planner_a61',
                                 'L132.view_from_report', 'L132.reference_views', 'cell_report', 'decided_reports'],
                   'decided_reports': ('b_2a0ba8b2, b_0dd237f0 (v4 results), b_4649234b (v5 results), d_c52e1670 (the '
                                       'v6-from-records certificate record) -- pinned'),
                   'post_run_prediction_scorer': 'd_cell_prediction (G27)'},
        'predictions_recorded_before_any_run': preds,
        'gates': {
            'G1-G7, G9, G11, G14, G16': 'as W101 / W118 / W132 / W137 / W139',
            'G8': 'persistence on PRODUCTION\'s certificate (status_production_trajectory), Addendum 59',
            'G13': 'holds inert through the first residual pass, held after', 'G15': 'stopping consistent',
            'G17': 'the pure rule v6 replays the in-cycle records and the decision',
            'G18': 'line fields (+ clean, v5; + the floor field, v6)',
            'G19': 'gated: replay 1..k0 bitwise every field', 'G20': 'production counters',
            'G21': 'creep captures complete', 'G22': 't_sum in-cycle == stride formula; terminal identity',
            'G23': 'overlap recorded (gated) / absent (ungated)',
            'G24': 'the 51 exit classes of every cycle == the network solve records\' final attempts and the ESSO logs',
            'G24b': ('the 51 clean entries of every cycle (tier, chain, metrics re-parsed, clean, reason) == the records '
                     'and the ESSO logs; all_clean_k the conjunction'),
            'G25': 'the record status follows the settling decision',
            'G27': 'D cells only: the Planner\'s Addendum 61 prediction (certified under v6; a failure exits 3)',
            'scope': GATE_SCOPE, 'non_stopping': list(NON_STOPPING_GATES), 'prediction_gates': list(PREDICTION_GATES),
            'post_run_evaluator_and_scorer_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w142_resettle_v6_checks.py', 'committed_output': cf,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()}},
                              'rerun_before_launch': ('--run re-runs every section and refuses unless all hold (the W100 '
                                                      'typing test is in the committed output only)')},
        'verbatim_text': {'quotes': {f'{f}:{k}': v for (f, k), v in VERBATIM.items()}, 'check': verb},
        'labelling_and_identity': {'label': V6.LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key, settling_resettle (v6 declaration)})'
                                                      '; every other key byte-identical to the pre-W142 harness '
                                                      '(checks K), the 36 v5, 38 v4 and 38 v3 keys included')},
        'harness_change': ('p515_s44_campaign_harness.py: resettle_hooks_module -- the v6 branch (3 lines; it routes the v6 '
                           'and the v6-extension declarations through p515_s53_w142_resettle_v6_hooks.hooks_module); '
                           'nothing else (checks D); every committed key unchanged (checks K)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'walls': {'this_spec_expected_h': wall['total_expected_h'], 'this_spec_worst_case_h': wall['total_worst_case_h'],
                  'basis': wall['basis'],
                  'single_run_over_4h': {c: wall['per_cell'][c]['worst_case_h'] for c in V6.CELL_ORDER
                                         if wall['per_cell'][c]['worst_case_h'] > 4.0}},
        'dry_run_command': launch_command(V6.CELL_ORDER[0], specs[V6.CELL_ORDER[0]]['sha256'], preconditions_only=True),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised by checks H (the real wrappers on the v6 state; the real install; the eight-way '
                             'dispatch; the exit wrappers on REAL production structures, H12 / H13) and C2 (the capture '
                             'replayed on the committed v4 cell-3 run); THE FIRST REAL CYCLE OF CELL #1 IS THE SMOKE'),
    }


def freeze_spec(started):
    tag = 'W142-SPEC'
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
    fr = from_records_state()
    if not fr['ok']:
        failures.append(f'the v6-from-records outputs must be committed and consistent: {fr}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    preds = predictions()
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found: {[k for k, v in verb["found_whitespace_normalised"].items() if not v]}')
    solver = W101L.solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    o_section = checks_inline['sections']['O']['result']
    dataset = L132.claims_dataset()
    points = completion_points(dataset)
    refs, ref_inputs = L132.reference_views()
    decided, decided_inputs = decided_reports()
    mem = L.memory_preflight(1)
    wall = wall_time_estimate()
    content = stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section,
                                 dataset, points, refs, ref_inputs, fr, preds, decided, decided_inputs)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor v5 {_sha(PREDECESSOR_REL)[:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator / scorer self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f'[{tag}] Planner A61 prediction: {preds["planner_addendum61_d_cells"]["statement"]}; expert A60: '
         f'{preds["expert_addendum60"]["recorded_as"]}')
    _log(f'[{tag}] claims scored: {dataset["n_claims"]} (skipped, not in the settled set: '
         f'{len(dataset["skipped_not_in_the_settled_set"])})')
    for p in points['report_points']:
        _log(f"[{tag}] completion point after #{p['after_cell_index_1_based']} {p['after_cell']} "
             f"(v3 numbering {p['claim_point_label_v3_numbering']}): {p['items_complete']}")
    for cell in V6.CELL_ORDER:
        w = wall['per_cell'][cell]
        _log(f"[{tag}] {cell}: cap {V6.spec_cap(cell)} expected {w['expected_h']:.2f} h worst {w['worst_case_h']:.2f} h; "
             f"LAUNCH: {content['cells'][cell]['launch_command']}")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h; memory "
         f"at freeze (non-gating): available {mem.get('available_gib')} GiB")
    _finish(0, '-- next: commit, then the dry run on the first cell')


# ======================================================================================================================
#  --run
# ======================================================================================================================
def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W142-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('the stage spec is not committed / clean')
    if ss['pins']['code_sha256'] != {rel: _sha(rel) for rel in ss['pins']['code_sha256']}:
        failures.append('code changed since the stage spec froze')
    pin = ss['pins']['campaign_specs'].get(cell) or {}
    if pin.get('sha256') != spec_sha256:
        failures.append(f'the stage spec pins {pin.get("sha256")} for {cell}, not {spec_sha256}')
    for key in ('v6_from_records', 'd_c52e1670_certificate', 'floor_replay'):
        p = ss['pins'][key]
        if _sha(p['path']) != p['sha256']:
            failures.append(f'{key} changed since the stage spec froze')
    idx = V6.CELL_ORDER.index(cell)
    for prev in V6.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'priority order: {prev} (#{V6.CELL_ORDER.index(prev) + 1}) has no results yet')
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
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["failing"]}')
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
             f"(pinned by the stage spec {ss_rel} sha256={ss_sha}); launch index {idx + 1} of {len(V6.CELL_ORDER)}; "
             f"checks per section {({k: v['holds'] for k, v in checks_inline['sections'].items()})}; pre-launch parts "
             f"{pre['parts']}; parent capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; "
             f"cap {spec['cap']}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} sha256={solver['sha256']}"
             + (f"; G27 prediction gate armed (D cell)" if V6.CELLS[cell]['item'] == 'D' else ''))
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
               'original_eval_key': V6.CELLS[cell]['orig_eval_key'],
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
        _log(f"[{tag}] settling v6: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"window {report.get('window')} band_width {report.get('band_width')} s {report.get('s_signed')} "
             f"t_sum(end) {report.get('t_sum_end')} k0_run {report.get('k0_run')} non-clean cycles "
             f"{report.get('non_clean_cycles')} floor rejections {len(report.get('turning_point_floor_rejections') or [])} "
             f"vetoes {report.get('n_vetoes')} cycles {report.get('cycles_run')} record status "
             f"{report.get('record_status')}")
    if predictions_failed:
        _log(f'[{tag}] RECORDED PREDICTION FAILED: {predictions_failed} {detail.get("G27")} -- STOP FOR THE PLANNER')
    if point is not None:
        _log(f"[{tag}] CLAIM-COMPLETION POINT after #{point['after_cell_index_1_based']} {cell} "
             f"({point['claim_point_label_v3_numbering']} in the v3 numbering): items {point['items_complete']} complete "
             f"-- report (Addendum 58). Scorer: {summarize_command(cell)}")
    ok = not stopping and _guards_ok(g) and isinstance(report, dict) and 'error' not in report
    code = (3 if predictions_failed else 0) if ok else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  --summarize
# ======================================================================================================================
def summarize(started, after_cell):
    tag = f'W142-SUMMARY-after-{after_cell}'
    ss_rel, ss_sha, ss = load_stage_spec()
    idx = V6.CELL_ORDER.index(after_cell)
    reports, missing = {}, []
    inputs = {}
    decided, decided_inputs = decided_reports()
    if decided_inputs != ss['pins']['decided_inputs_sha256']:
        _log(f'[{tag} PRECONDITION FAILED] the decided cells\' records changed since the stage spec froze')
        _finish(1)
    reports.update(decided)
    for cell in V6.CELL_ORDER[:idx + 1]:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        rel = os.path.relpath(path, REPO)
        if not os.path.isfile(path) or not _committed_clean(rel):
            missing.append(cell)
            continue
        inputs[rel] = _sha(rel)
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, f'w142_summary_after_{idx + 1:02d}_{after_cell}.json')
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
        scored.append(DET.score_claim_v6(cl, view('ref'), view('other')))
    d_cells = {V5.CELLS[c]['orig_label']: c for c in V5.CELL_ORDER if V5.CELLS[c]['item'] == 'D'}
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
        d_out['certifying_spec_per_point'] = {lb: (reports.get(d_cells.get(lb), {}) or {}).get('certifying_spec')
                                              for lb in L132.D_FIT_NODE7_LABELS if lb != 'n7_4h_e1'}
        d_out['all_points_certified'] = all(v == 'certified' for v in d_out['statuses'].values())
        d_out['first_unit_n7_eur_per_mwh'] = (x0['Q'] - un['Q'] - 0.25 * p_cost) / 1.0
        b = views.get('b_2a0ba8b2')
        d_out['first_unit_best_node_n5_4h_e1_eur_per_mwh'] = ((x0['Q'] - b['Q'] - 0.25 * p_cost) / 1.0) if b else None
        d_out['committed_baseline_tables'] = {k: _load(L132.BASELINE_TABLES)[k] for k in ('node7_fit', 'breakeven')}
    run_reports = {c: r for c, r in reports.items() if c not in DECIDED}
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'after_cell': after_cell, 'after_cell_index_1_based': idx + 1,
           'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': DEFINITIONS['objective_convention'],
           'reports': reports, 'decided_before_v6': sorted(DECIDED),
           'certifying_spec_per_cell': {c: r.get('certifying_spec') for c, r in reports.items()},
           'references': refs, 'claims_scored': scored,
           'claims_complete': [s['claim_id'] for s in scored if not str(s.get('verdict', '')).startswith('not scored')],
           'verdicts_changed_by_the_v6_determinacy_floor': [s['claim_id'] for s in scored if s.get('verdict_changed_by_v6')],
           'D_fit': d_out, 'predictions_scored': L132.score_predictions(run_reports),
           'group1_prediction_criteria': score_group1(reports), 'planner_addendum61': score_planner_a61(run_reports),
           'definitions': DEFINITIONS, 'predictions': ss['predictions_recorded_before_any_run'],
           'inputs_sha256': {**inputs, **ref_inputs, **decided_inputs}}
    H._write_once_json(_abs(out_rel), doc)
    man = os.path.join(ROOT_REL, f'w142_summary_after_{idx + 1:02d}_{after_cell}_manifest_sha256.json')
    H._write_once_json(_abs(man), {out_rel: _sha(out_rel), **doc['inputs_sha256']})
    _log(f'[{tag}] wrote {out_rel}; claims complete {len(doc["claims_complete"])}; D fit '
         f'{"in" if d_out else "not yet"}; verdicts changed by the v6 floor {doc["verdicts_changed_by_the_v6_determinacy_floor"]}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-cells', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=V6.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--after-cell', choices=V6.CELL_ORDER, default=None)
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
