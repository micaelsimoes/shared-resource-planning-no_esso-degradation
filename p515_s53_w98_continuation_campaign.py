"""
P5.15 Addendum 51, Planner task W98 -- STAGE 1 of the post-certification continuation (route A, x = 0): frozen stage
spec v37 (predecessor v36 14bbddc7, NOT edited), the stage-1 campaign freeze, and the stage-1 run. BUILT AND FROZEN IN
W98; THE RUN IS NOT LAUNCHED IN W98 (the Planner launches it).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 51 (route A staged; G6 restated; the expert's predictions and the
competing outcomes H1 / H2 / H3; the stage-2 decision rule); Planner task W98; P5_15_ADDENDUM48_50_3X3_REPORT.md; the
W97 diagnostics (b736e4e5).

THE RUN. One evaluation, x = 0, on the 3 x 3 instance under the IDENTICAL configuration the certified cell ran
(campaign s53_w91_3x3_pair, spec 231558f0: derived instance s53_3x3, case-file AA declared, ESS ageing baseline, row-18
premium alpha 0.5, tight tail {True, 1e-6}, option (b) on, persistence off, concurrency 1), plus the entry option
`certification_continuation` (p515_s53_w98_continuation_hooks; it enters the eval key): the certification rule
disabled, the certifying regime HELD for every cycle > 72 (AA off, tail on, rho frozen), cap 72 + 30 = 102, early stop
when |dQ| < 500 EUR for 3 consecutive post-certification cycles, every block's recourse recorded every cycle.

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-spec                 ZERO SOLVES. v37 (write-once, named by its sha256): re-runs the W98 zero-solve checks
                                inline and pins their committed output; G6 v37 and settling-analysis self-tests.
  --freeze                      ZERO SOLVES. The stage-1 campaign spec (s53_w98_x0_continuation), pinning v37; the
                                pre-launch key / collision assertion on the frozen spec.
  --run --spec-sha256 S         THE RUN (NOT RUN IN W98). Preconditions (the zero-solve checks re-run, the pre-launch
                                assertion, the memory preflight, the solver path), H.evaluate on the one entry, then the
                                gates, the replay gate, the hold / capture checks, G6 v37, the settling analysis and the
                                stage-2 decision; results + manifest, write-once.

Exit codes: freezes 0 done, 1 precondition / check / guard failure; --run 0 every gate holds (whatever the replay
label), 1 a gate / harness / guard / precondition failure.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W98 continuation launcher (never solves)').install()

# The W90 3 x 3 launcher (and through it W89, W86, W87, W88, the alpha-row launcher -- each arms its own permitted=()
# guard); the W98 zero-solve checks (arms its own guard); the continuation hooks (stdlib at import).
import p515_s53_w90_3x3_campaign as W0  # noqa: E402
import p515_s53_w98_continuation_checks as K  # noqa: E402
import p515_s53_w98_continuation_hooks as C  # noqa: E402
import p515_s53_w95_x0_drift_diagnostics as W95  # noqa: E402 -- its IPOPT log parsers (stdlib only), BY IMPORT

H, W9, L, X, A = W0.H, W0.W9, W0.L, W0.X, W0.A
# Uninstall order (last installed first): the checks module's guard (imported last), W90's chain in its own LIFO order,
# then this launcher's (installed first, above).
GUARDS_LIFO = (K.GUARD,) + tuple(W0.GUARDS_LIFO) + (PARENT_GUARD,)
GUARD_NAMES = ('w98_checks',) + tuple(W0.GUARD_NAMES) + ('w98_parent',)

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s53_w98_continuation_campaign'
STAGE_TEXT = ('P5.15 Addendum 51, W98 -- stage 1 of the post-certification continuation (route A): x = 0 replayed under '
              'the certifying configuration (bitwise gate against the 72 recorded cycles), then up to 30 further cycles '
              'with the certification rule disabled and the certifying regime held (AA off, tight tail on, rho frozen); '
              'early stop at |dQ| < 500 EUR for 3 cycles; every block recorded every cycle')
_P53 = W9._P53
ROOT_REL = os.path.join(_P53, 'w98_continuation')
SPEC_V36 = {'path': os.path.join(_P53, 'frozen_s53_spec_v36_14bbddc7.json'),
            'sha256': '14bbddc7593c6ee3ee56854b12cb146d6770a27953e3f36bd88c3e0417c40110'}
SPEC_PREFIX = 'frozen_s53_spec_v37_'
SPEC_VERSION = 37
CAMPAIGN_ID = 's53_w98_x0_continuation'
ENTRY_LABEL = 'x0_cont'
CAP = C.STAGE1_N + C.STAGE1_CONTINUATION_CYCLES
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
GIB = 1 << 30
N = C.STAGE1_N
CHECKS_REL = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
CHECKS_MANIFEST_REL = os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST)
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
SOLVER_PATH = '/usr/local/bin/ipopt'

# ---- the certified cell this run replays (committed evidence; every pin re-hashed at every freeze and at the run) ----
_PAIR = os.path.join(_P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair')
_X0 = os.path.join(_PAIR, 'evals', 'f6e9cd53fdbb8ee8_x0')
_N7 = os.path.join(_PAIR, 'evals', 'c82522f470b35b58_n7_4h_e1')
CERTIFIED = {
    'campaign_id': 's53_w91_3x3_pair', 'label': 'x0',
    'campaign_spec': {'path': os.path.join(_PAIR, 'campaign_spec_s53_w91_3x3_pair_231558f0.json'),
                      'sha256': '231558f0705797327c10d8e0e82ff878e19ed3a4620766bd5b113995bb920e99'},
    'campaign_results': {'path': os.path.join(_PAIR, 'campaign_results.json'),
                         'sha256': '587d3f1afb658e8a501d34b51bb8b269dd0c229391c1a26469f01001a8aad566'},
    'eval_dir': _X0,
    'eval_key': K.PAIR_KEYS['x0'],
    'candidate_key': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57',
    'certification_cycle': 72, 'Q': 842832534.7623764, 'bar': 13954.375690460205,
    'per_cycle_record': {'path': os.path.join(_X0, 'per_cycle_record.jsonl'), 'sha256': C.STAGE1_REPLAY_REFERENCE['sha256']},
    'evaluation_record': {'path': os.path.join(_X0, 'evaluation_record.json'),
                          'sha256': '61bd5232e645112b05ad66f5683ff72760937098410dd9a1d34df728ee029d80'},
    'g_trajectory': {'path': os.path.join(_X0, 'g_s39_D.json'),
                     'sha256': '6ea6003e4cd1505fb67e2e13cb0f6a08d41e583c29f95db2eeb18fd292aa0c4a'},
    'initialisation_identity': {'path': os.path.join(_X0, 'initialisation_identity.json'),
                                'sha256': 'f5451e1d0d4767d16ad8decddc9015ff89b9255ddafb9c0e4cf904a20ded1b10'},
    'x0_footprint_peak_bytes': 20517044400,
}
NODE7 = {'eval_dir': _N7, 'eval_key': K.PAIR_KEYS['n7_4h_e1'], 'certification_cycle': 69, 'Q': 842595839.5131028,
         'bar': 4010.5701702833176}
VALUE = {'V_eur': 236695.2492736578, 'bar_sum_resolution_eur': 17964.945860743523, 'R_ref': 259375.33,
         'I_eur': 317957.0085035586, 'R_at_certification': 0.9125588483309005,
         'source': os.path.join(_PAIR, 'campaign_results.json') + ' value_and_R'}
W97 = {'path': os.path.join(_P53, 'w97_diagnostics', 'w97_consolidated_diagnostics.json'),
       'sha256': '9ce7e6e7b8bc0b68380eb9f8f5c87b8c2d2b1937cee429458dc77b91f967391c', 'commit': 'b736e4e5'}
EXTRA_CLEAN_FILES = tuple(W0.EXTRA_CLEAN_FILES) + (SCRIPT_NAME, 'p515_s53_w98_continuation_hooks.py',
                                                  'p515_s53_w98_continuation_checks.py', 'p515_s53_w90_3x3_campaign.py',
                                                  'p515_s53_w95_x0_drift_diagnostics.py')

# ---- the replay gate: fields and tolerance ---------------------------------------------------------------------------
REPLAY_EXCLUDED_FIELDS = tuple(W0.NON_DETERMINISTIC_CYCLE_FIELDS)    # wall / RSS / capture timing
# The raw trajectory row (g_s39_D.json) is compared too, REPORTED, not gated; its `required_consecutive_cycles` field
# records the certificate length in force, which the continuation raises to 10**9 by design (hooks module).
RAW_ROW_EXPECTED_DIFFERENT = ('required_consecutive_cycles',)

# ---- G6 v37 (Addendum 51, "adopted as the Planner states it") --------------------------------------------------------
G6_OPTIMAL_EXIT = 'Optimal Solution Found.'
LADDER = X.LADDER

# ---- the settling analysis constants (frozen in v37) -----------------------------------------------------------------
FIT_MIN_STEPS = 4                  # the peak plus at least three steps after it
H2_BAND = (0.95, 1.05)             # "ratio ~ 1" (Worker operationalisation, frozen before the run)
PEAK_WITHIN = 5                    # "peak within a few cycles of certification" (Addendum 50: "within ~ 5 cycles")
PRED_RATIO = (0.80, 0.95)
PRED_D = (20000.0, 60000.0)
PRED_R_SETTLED_MIN = 0.80

# ---- verbatim text (Addendum 51; checked against the brief, whitespace-normalised, at every freeze) ------------------
VERBATIM = {
    'stage_1': ("**Stage 1: x = 0** — re-run under the identical spec to certification, **checked bitwise against the 72 "
                "recorded cycles** (this replay is also the instance's reproducibility measurement for the manuscript; a "
                "divergence at cycle k is reported with its magnitude and the run is then labelled separately, still "
                "informative), then continue **30 cycles** with the certification rule disabled and the certifying "
                "regime unchanged (tail on, AA off, ρ frozen) — a separately labelled run under spec v37. Stop early at "
                "|ΔQ| < 500 €/cycle for 3 cycles."),
    'analysis': ("**Analysis:** fit the post-certification steps to a geometric sequence; report the ratio and the "
                 "extrapolated remaining descent D = step/(1 − ratio) with the fit's validity (increasing steps → no "
                 "extrapolation)."),
    'stage_2_decision_rule': ("**Stage 2 decision rule, recorded now:** the storage cell's continuation runs if D_x0 "
                              "(measured plus extrapolated) exceeds the value's resolution (bar-sum ≈ 18 k€); otherwise "
                              "R is reported as the range [(V − D_x0)/R_ref, V/R_ref] with the storage cell's descent "
                              "bounded above by D_x0."),
    'expert_predictions': ("**Expert's predictions, recorded:** replay bitwise through cycle 72; steps peak within a few "
                           "cycles of certification and then decay geometrically with ratio 0.80–0.95; D_x0 = 20–60 k€; "
                           "stage 2 triggered; R_settled ≥ 0.80."),
    'competing_outcomes': ("**Competing outcomes and what each means:** (H1) hump then geometric decay → AA-off "
                           "transient plus a slow mode; the settling criterion is a step bound at the end of the window; "
                           "(H2) near-constant steps through cycle 102 (ratio ≈ 1) → the residual tolerances are too "
                           "loose for this instance; the certified Q's are reported as upper bounds and the criterion "
                           "must be objective-based (or ε_rel tightened) before any further 3 × 3 result is used; (H3) "
                           "growing steps or a jump → basin transition; Advisor review before anything else."),
    'g6': ("**G6 for future specs — adopted as the Planner states it:** `Optimal Solution Found` plus the four error "
           "metrics within the tail tolerances for the final accepted attempt; μ/floor reported per solve, never gated — "
           "the floor test measured scaling, not depth. Frozen into v37."),
}
TASK_W98_STAGE2_TEXT = ("the storage cell's continuation runs if D_x0 (measured plus extrapolated) exceeds the value's "
                        "resolution — the bar-sum, 17,964.945860743523 — otherwise R is reported as the range "
                        "[(V − D_x0)/R_ref, V/R_ref] with V = 236,695.2492736578 and R_ref = 259,375.33.")


# ======================================================================================================================
#  utilities
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _committed_clean(rel):
    st = L._git_state(rel)
    return bool(st.get('git_tracked') and st.get('git_clean'))


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in zip(GUARD_NAMES, GUARDS_LIFO)}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    """EVERY exit path: verify every guard at exactly 0, uninstall them LIFO, exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W98] guards {g} {extra_msg}')
    for guard in GUARDS_LIFO:
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and OWN_PROCESS_SUBSTRING in parts[1]:
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


def _norm(text):
    return ' '.join(text.split())


def _pins_ok():
    failures = []
    for name, pin in (('certified campaign spec', CERTIFIED['campaign_spec']),
                      ('certified campaign results', CERTIFIED['campaign_results']),
                      ('certified per-cycle record', CERTIFIED['per_cycle_record']),
                      ('certified evaluation record', CERTIFIED['evaluation_record']),
                      ('certified g trajectory', CERTIFIED['g_trajectory']),
                      ('certified initialisation identity', CERTIFIED['initialisation_identity']),
                      ('spec v36', SPEC_V36), ('W97 diagnostics', W97)):
        if _sha(pin['path']) != pin['sha256'] or not _committed_clean(pin['path']):
            failures.append(f'{name} not as committed: {pin["path"]}')
    return failures


def _common_checks():
    failures, ev = W0._common_checks()
    failures += _pins_ok()
    for rel in (SCRIPT_NAME, 'p515_s53_w98_continuation_hooks.py', 'p515_s53_w98_continuation_checks.py',
                os.path.relpath(H.HARNESS_PATH, REPO)):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of the W98 launcher is alive: {others}')
    return failures, ev


def solver_check():
    """What the child will resolve for NLP_SOLVER_PATH (the harness's own resolver) and the binary's sha256."""
    res = H._resolve_solver_path_from_dotenv()
    path = res.get('NLP_SOLVER_PATH')
    sha = H.sha256_file(path) if path and os.path.isfile(path) else None
    return {'resolved': res, 'expected': SOLVER_PATH, 'sha256': sha,
            'certified_cell_child_path': _load(CERTIFIED['evaluation_record']['path']).get('nlp_solver_path_in_child'),
            'ok': path == SOLVER_PATH and sha is not None}


# ======================================================================================================================
#  configuration and entry: the certified cell's, plus the continuation declaration
# ======================================================================================================================
def certified_spec():
    return _load(CERTIFIED['campaign_spec']['path'])


def certified_entry():
    return next(e for e in certified_spec()['candidates'] if e['label'] == CERTIFIED['label'])


def stage1_configuration():
    cfg = certified_spec()['configuration']
    return {'name': ('W98 CONTINUATION of the certified 3 x 3 x = 0 cell -- ' + cfg['name']),
            'arm_label': cfg['arm_label'], 'overrides': dict(cfg['overrides']),
            'case_file_anderson_acceleration': dict(cfg['case_file_anderson_acceleration']),
            'ess_ageing_baseline': json.loads(json.dumps(cfg['ess_ageing_baseline'])),
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'derived_instance': json.loads(json.dumps(cfg['derived_instance'])),
            'convergence_depth_tail': dict(cfg['convergence_depth_tail']),
            'note': ('the certified cell\'s configuration (spec 231558f0) unchanged; the entry adds '
                     'certification_continuation (keyed); cap N + 30; persistence off; option (b) on')}


def stage1_entries():
    e = certified_entry()
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    opts = {'investment_year': e['canonical']['investment_year'],
            'interface_deviation_premium': dict(e['interface_deviation_premium']),
            'release_solution_bookkeeping': e['release_solution_bookkeeping'],
            'certification_continuation': C.stage1_declaration()}
    return [(ENTRY_LABEL, nodes, opts)]


def expected_keys():
    e = certified_entry()
    cfg = certified_spec()['configuration']
    kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
              derived_instance=cfg['derived_instance'], interface_deviation_premium=e['interface_deviation_premium'],
              convergence_depth_tail=cfg['convergence_depth_tail'])
    base = H.evaluation_key(e['key'], {}, **kw)
    cont = H.evaluation_key(e['key'], {}, certification_continuation=C.stage1_declaration(), **kw)
    return {'base_key_without_continuation': base, 'continuation_key': cont}


def campaign_root():
    return _abs(os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_ID}'))


def pre_launch_assertion(spec=None):
    """Identity and collision, recomputed now: (a) the key WITHOUT the declaration equals the certified cell's key (the
    same candidate x configuration -- the replay's identity); (b) the continuation key differs from it and follows the
    declared formula; (c) it appears in no committed campaign spec outside this launcher's root; (d) this campaign's
    root, eval dir and working-dir ids differ from the certified cell's and do not exist yet (before the freeze) / hold
    only the frozen spec (after); (e) a frozen spec's entry carries exactly the recomputed key."""
    k = expected_keys()
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    cert = certified_entry()
    eval_dir_name = H.eval_dir_name(k['continuation_key'], ENTRY_LABEL)
    ids = H.eval_ids(CAMPAIGN_ID, k['continuation_key'])
    work = L._work_dir()
    entry = next((e for e in (spec or {}).get('candidates') or [] if e['label'] == ENTRY_LABEL), None)
    parts = {
        'base_key_equals_certified_key': k['base_key_without_continuation'] == CERTIFIED['eval_key'] == cert['eval_key'],
        'continuation_key_differs_from_certified_key': k['continuation_key'] != CERTIFIED['eval_key'],
        'continuation_key_absent_from_committed_specs': k['continuation_key'] not in committed,
        'campaign_root_differs_from_certified': os.path.abspath(campaign_root()) != os.path.abspath(_abs(_PAIR)),
        'eval_dir_name_differs_from_certified': eval_dir_name != os.path.basename(CERTIFIED['eval_dir']),
        'working_dir_ids_differ_from_certified': not (set(ids.values()) & set(cert['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
        'certified_eval_dir_untouched_by_this_campaign': not os.path.abspath(
            os.path.join(campaign_root(), 'evals', eval_dir_name)).startswith(os.path.abspath(_abs(_PAIR))),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['continuation_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name,
            'working_dir_ids': ids, 'certified_working_dir_ids': cert['working_dir_ids'],
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(spec, ss_pin):
    """The frozen campaign spec runs the certified configuration exactly, plus the continuation."""
    cert = certified_spec()
    ce, cc = certified_entry(), cert['configuration']
    cfg = spec['configuration']
    entries = spec['candidates']
    e = entries[0] if len(entries) == 1 else {}
    same_cfg_keys = ('arm_label', 'case_file', 'case_file_sha256', 'overrides', 'apply_rho', 'full_diagnostics_in_rows',
                     'case_file_anderson_acceleration', 'ess_ageing_baseline', 'ess_ageing_baseline_label',
                     'derived_instance', 'convergence_depth_tail')
    checks = {f'configuration:{k}': cfg.get(k) == cc.get(k) for k in same_cfg_keys}
    checks['configuration:ess_params_file_sha256'] = (cfg.get('ess_params_file') or {}).get('sha256') == \
        (cc.get('ess_params_file') or {}).get('sha256')
    same_entry_keys = ('canonical', 'key', 'overrides', 'effective_anderson_acceleration', 'interface_deviation_premium',
                       'release_solution_bookkeeping', 'post_certification')
    checks.update({f'entry:{k}': e.get(k) == ce.get(k) for k in same_entry_keys})
    checks.update({
        'one_entry': len(entries) == 1, 'entry_label': e.get('label') == ENTRY_LABEL,
        'entry_continuation_is_stage1_declaration': e.get('certification_continuation') == C.stage1_declaration(),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID, 'cap': spec.get('cap') == CAP,
        'concurrency': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_as_certified': spec.get('required_consecutive_cycles')
        == cert['required_consecutive_cycles'] == 10,
        'thread_caps_as_certified': spec.get('thread_caps') == cert['thread_caps'],
        'interpreter_as_certified': spec.get('interpreter') == cert['interpreter'],
        'solver_path_as_certified': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH')
        == (cert.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'stage_spec_pinned': (spec.get('extra') or {}).get('stage_spec') == ss_pin,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
        'persistence_off': e.get('post_certification') is None,
    })
    return checks


# ======================================================================================================================
#  G6 v37: `Optimal Solution Found` + the four error metrics within the tail tolerances, final accepted attempt
# ======================================================================================================================
def _block_key(r):
    return (r.get('network'), r.get('year'), r.get('day'))


def _read_segment(path, start, end):
    with open(path, 'rb') as handle:
        handle.seek(start)
        seg = handle.read(end - start)
    return seg.decode('utf-8', errors='replace'), hashlib.sha256(seg).hexdigest()


def g6_v37_evaluate_records(records, terminal_round, b=W9.BLOCKS_PER_ROUND, tail_tol=W9.TAIL['compl_inf_tol'],
                            segment_reader=_read_segment):
    """G6 v37 on one round (the run's terminal round). Per block: the attempts of that round ordered by the production
    ladder (primary, recovery, recovery_tier2); the FINAL attempt must be the accepted one (no superseded attempt
    accepted; a well-formed ladder); its exit must be `Optimal Solution Found.`; `compl_inf_tol` in force must be the
    tail value; and the four terminal error metrics parsed from that attempt's own IPOPT log byte range must meet the
    tolerances in force (scaled overall NLP error <= tol; unscaled dual infeasibility <= dual_inf_tol, constraint
    violation <= constr_viol_tol, complementarity <= compl_inf_tol; options not in the printed list take IPOPT's
    documented defaults -- W95's parser and defaults, by import). Gate: exactly B blocks and every block passes. mu/floor
    REPORTED per solve, never gated."""
    term = [r for r in records if r.get('round') == terminal_round]
    by_block = defaultdict(list)
    for r in term:
        by_block[_block_key(r)].append(r)
    per_block, failing = [], []
    for key in sorted(by_block, key=lambda k: tuple(str(x) for x in k)):
        attempts = by_block[key]
        labels = [a.get('attempt') for a in attempts]
        known = all(lab in LADDER for lab in labels)
        ordered = sorted(attempts, key=lambda a: LADDER.index(a.get('attempt'))) if known else list(attempts)
        final, superseded = ordered[-1], ordered[:-1]
        fails = []
        if not known or [a.get('attempt') for a in ordered] != list(LADDER[:len(ordered)]):
            fails.append('ladder_malformed')
        if any(X.accepted(a) for a in superseded):
            fails.append('superseded_attempt_accepted')
        if final.get('exit') != G6_OPTIMAL_EXIT:
            fails.append(f"final_exit_not_optimal: {final.get('exit')!r}")
        if final.get('compl_inf_tol_in_force') != tail_tol:
            fails.append(f"compl_inf_tol_in_force {final.get('compl_inf_tol_in_force')!r} != tail {tail_tol!r}")
        metrics, tols, margins, checks, seg_sha = None, {}, None, None, None
        try:
            text, seg_sha = segment_reader(final['log_path'], final['log_bytes'][0], final['log_bytes'][1])
            opts, _merged = W95.parse_options(text)
            _n_it, metrics = W95.parse_terminal_metrics(text)
            for name in ('tol', 'dual_inf_tol', 'constr_viol_tol', 'compl_inf_tol'):
                tols[name], tols[f'{name}_source'] = W95._num(opts, name)
        except Exception as error:  # noqa: BLE001 -- recorded as a failing block
            fails.append(f'log_unreadable: {type(error).__name__}: {error}')
        if metrics and all(m in metrics for m in ('overall_nlp_error', 'dual_infeasibility', 'constraint_violation',
                                                  'complementarity')):
            checks = {'overall_nlp_error_scaled_le_tol': metrics['overall_nlp_error']['scaled'] <= tols['tol'],
                      'dual_infeasibility_unscaled_le_dual_inf_tol':
                          metrics['dual_infeasibility']['unscaled'] <= tols['dual_inf_tol'],
                      'constraint_violation_unscaled_le_constr_viol_tol':
                          metrics['constraint_violation']['unscaled'] <= tols['constr_viol_tol'],
                      'complementarity_unscaled_le_compl_inf_tol':
                          metrics['complementarity']['unscaled'] <= tols['compl_inf_tol']}
            margins = {'overall_nlp_error_over_tol': metrics['overall_nlp_error']['scaled'] / tols['tol'],
                       'dual_infeasibility_over_dual_inf_tol': metrics['dual_infeasibility']['unscaled'] / tols['dual_inf_tol'],
                       'constraint_violation_over_constr_viol_tol':
                           metrics['constraint_violation']['unscaled'] / tols['constr_viol_tol'],
                       'complementarity_over_compl_inf_tol': metrics['complementarity']['unscaled'] / tols['compl_inf_tol']}
            if tols.get('compl_inf_tol') != tail_tol:
                fails.append(f"logged compl_inf_tol {tols.get('compl_inf_tol')!r} != tail {tail_tol!r}")
            if not all(checks.values()):
                fails.append(f'error_metrics_outside_tolerance: {sorted(k for k, v in checks.items() if not v)}')
        elif not any(f.startswith('log_unreadable') for f in fails):
            fails.append('terminal_error_metrics_not_parsed')
        entry = {'block': list(key), 'agent': final.get('agent'), 'attempts': labels, 'final_attempt': final.get('attempt'),
                 'final_exit': final.get('exit'), 'compl_inf_tol_in_force': final.get('compl_inf_tol_in_force'),
                 'tolerances': tols, 'metrics': metrics, 'checks': checks, 'metric_over_tolerance': margins,
                 'log_segment_sha256': seg_sha,
                 'mu_final': final.get('mu_final'), 'mu_floor': final.get('mu_floor'),
                 'mu_over_floor_reported_not_gated': final.get('mu_over_floor'),
                 'floor_status_reported_not_gated': final.get('floor_status'), 'failures': fails}
        per_block.append(entry)
        if fails:
            failing.append({'block': list(key), 'failures': fails})
    worst = {}
    for name in ('overall_nlp_error_over_tol', 'dual_infeasibility_over_dual_inf_tol',
                 'constraint_violation_over_constr_viol_tol', 'complementarity_over_compl_inf_tol'):
        vals = [e['metric_over_tolerance'][name] for e in per_block if e['metric_over_tolerance']]
        worst[name] = max(vals) if vals else None
    mu = [e['mu_over_floor_reported_not_gated'] for e in per_block if e['mu_over_floor_reported_not_gated'] is not None]
    return {'definition': VERBATIM['g6'], 'terminal_round': terminal_round, 'B': b, 'n_blocks': len(by_block),
            'n_records_in_round': len(term), 'n_failing_blocks': len(failing), 'failing_blocks': failing,
            'worst_metric_over_tolerance': worst,
            'mu_over_floor_reported_not_gated': {'n': len(mu), 'min': min(mu) if mu else None,
                                                 'max': max(mu) if mu else None,
                                                 'n_at_floor': sum(1 for e in per_block
                                                                   if e['floor_status_reported_not_gated'] == 'at')},
            'per_block': per_block, 'gate_pass': len(by_block) == b and not failing}


def g6_v37_evaluate(eval_dir, terminal_round):
    return g6_v37_evaluate_records(_read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)),
                                   terminal_round)


def g6_v37_self_tests():
    """On committed data (the IPOPT logs are gitignored and read in place; each segment's sha256 is recorded):
    the certified x = 0 terminal round 72 must PASS (W95: all 80 Optimal, all four metrics inside); the certified
    node-7 terminal round 69 must FAIL (its one 'Solved To Acceptable Level' exit on DSO7|2034|Autumn)."""
    out = {}
    for name, eval_dir, rnd, expect in (('x0_round_72', CERTIFIED['eval_dir'], 72, True),
                                        ('n7_round_69', NODE7['eval_dir'], 69, False)):
        try:
            r = g6_v37_evaluate(_abs(eval_dir), rnd)
            out[name] = {'expect_pass': expect, 'observed_pass': r['gate_pass'], 'ok': r['gate_pass'] is expect,
                         'n_blocks': r['n_blocks'], 'failing_blocks': r['failing_blocks'],
                         'worst_metric_over_tolerance': r['worst_metric_over_tolerance'],
                         'mu_over_floor_reported_not_gated': r['mu_over_floor_reported_not_gated'],
                         'segment_sha256_by_block': {'|'.join(map(str, e['block'])): e['log_segment_sha256']
                                                     for e in r['per_block']}}
        except Exception as error:  # noqa: BLE001
            out[name] = {'expect_pass': expect, 'ok': False, 'error': f'{type(error).__name__}: {error}'}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  the settling analysis (formula frozen in v37): steps, geometric fit, D, stage-2 decision, predictions, H1/H2/H3
# ======================================================================================================================
SETTLING_DEFINITION = {
    'Q': 'gross_operational_cost per cycle (settlement excluded), the run\'s own per_cycle_record.jsonl',
    'steps': 's_k = Q_k - Q_(k-1) for k = N+1 .. K (K = the run\'s last cycle; a step is unavailable if either Q is '
             'None); descent d_k = -s_k (positive while Q falls)',
    'D_measured': 'Q_N - Q_K of THIS run (its own cycle-N value; equal to the certified Q when the replay is bitwise)',
    'resolution_of_D_measured': '|s_K| (the terminal step: the stopping slack of this run, CLAUDE.md rule "report a '
                                'difference with its resolution")',
    'peak': 'k_peak = the first k in N+1..K maximising d_k',
    'fit_window': ('k = k_peak .. K (the peak and every later step). Justification: a geometric law describes a '
                   'monotone decay; the rise to the peak is the transient (the AA-off swing, H1) that no geometric law '
                   'describes, and including it biases the ratio upward. W97 found the step increments at cycle 72 had '
                   'shrunk to -147 (steps -3,858 then -4,006), so the peak is expected within a cycle or two of '
                   'certification; starting at the peak, not at N+1, keeps the fit on the decaying part whatever that '
                   'turns out to be. All post-N steps are fitted too, REPORTED, never deciding.'),
    'fit': ('ordinary least squares of ln d_k on (k - k_peak) over the fit window; r = exp(slope); the fitted step '
            'at K, d_hat_K = exp(intercept + slope (K - k_peak)), reported'),
    'validity': {'V1': f'at least {FIT_MIN_STEPS} steps in the fit window (the peak plus three after it)',
                 'V2': 'every step in the fit window is a descent (d_k > 0)', 'V3': 'r < 1',
                 'rule': 'valid iff V1 and V2 and V3; otherwise NO extrapolation ("increasing steps -> no '
                         'extrapolation": a peak within the last three steps fails V1, a rising fit fails V3)'},
    'D_extrap': ('r d_K / (1 - r), valid fit only: Addendum 51\'s D = step / (1 - ratio) with step = the FIRST '
                 'UNMEASURED step, r d_K (d_K = the measured last step), so D_extrap is the geometric tail beyond '
                 'the run and D_measured + D_extrap does not count d_K twice. The fitted-endpoint variant '
                 'r d_hat_K / (1 - r) is reported, not deciding.'),
    'D_x0': 'D_measured + D_extrap when the fit is valid; otherwise D_measured, flagged as a LOWER BOUND',
    'stage_2_decision': (
        'valid fit: stage 2 runs iff D_x0 > bar_sum (17,964.945860743523). Invalid fit: if D_measured alone exceeds '
        'bar_sum, stage 2 runs (the lower bound already does); else if the early-stop rule ended the run, D_x0 = '
        'D_measured and stage 2 runs iff D_x0 > bar_sum (the run met the settling rule; no extrapolation is '
        'claimed); else the decision is UNDETERMINED and goes to the Planner (a lower bound below the resolution with '
        'steps that do not decay is H2 / H3 territory, which Addendum 51 sends to review). When stage 2 does not run: '
        'R is reported as the range [(V - D_x0)/R_ref, V/R_ref] (V = 236,695.2492736578, R_ref = 259,375.33), the '
        'storage cell\'s descent bounded above by D_x0.'),
    'R_settled': ('stage 2 run: R_settled = (V - D_x0 + D_unit) / R_ref, D_unit the storage cell\'s D from stage 2 '
                  '(not computable in stage 1); stage 2 not run: the range above'),
    'classification_worker_operationalisation_frozen_before_the_run': {
        'H3': (f'a jump -- any post-N step with |s_k| > the certified bar {CERTIFIED["bar"]!r} -- or growing steps: '
               f'r > {H2_BAND[1]}, or the peak within the last three steps without an early stop'),
        'H2': f'not H3, a fit window of >= {FIT_MIN_STEPS} descending steps with {H2_BAND[0]} <= r <= {H2_BAND[1]}',
        'H1': f'not H3 / H2, a valid fit with r < {H2_BAND[0]}',
        'settled_by_early_stop': 'none of the above and the early-stop rule ended the run (steps below 500 EUR)',
        'unclassified': 'otherwise (reported with the step series)',
        'precedence': 'H3 > H2 > H1 > settled_by_early_stop > unclassified'},
    'prediction_scoring_worker_operationalisation_frozen_before_the_run': {
        'replay_bitwise_through_72': 'the replay gate reads bitwise_through_N',
        'peak_within_a_few_cycles': f'k_peak <= N + {PEAK_WITHIN} (Addendum 50: "peak within ~ 5 cycles")',
        'geometric_decay_ratio_0.80_0.95': f'valid fit and {PRED_RATIO[0]} <= r <= {PRED_RATIO[1]}',
        'D_x0_20_60_k': f'{PRED_D[0]:.0f} <= D_x0 <= {PRED_D[1]:.0f} (on the stated basis)',
        'stage_2_triggered': 'the stage-2 decision is "run"',
        'R_settled_ge_0.80': ('stage 2 not run: (V - D_x0)/R_ref >= 0.80; stage 2 run: pending its D_unit')},
}


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def _fit(window, d):
    k0 = window[0]
    xs = [k - k0 for k in window]
    ys = [math.log(d[k]) for k in window]
    slope, icpt = _ols(xs, ys)
    r = math.exp(slope)
    return {'r': r, 'ln_intercept': icpt, 'slope': slope, 'n_steps': len(window),
            'd_hat_last': math.exp(icpt + slope * xs[-1]),
            'endpoint_ratio': (d[window[-1]] / d[window[0]]) ** (1.0 / (window[-1] - window[0]))
            if window[-1] > window[0] and d[window[0]] > 0 and d[window[-1]] > 0 else None,
            'per_step_ratios': [d[b] / d[a] if d[a] else None for a, b in zip(window, window[1:])]}


def settling_analysis(q_by_cycle, n=N, early_stop_cycle=None, cap=CAP, bar_cert=CERTIFIED['bar'], value=VALUE):
    """The frozen formula (SETTLING_DEFINITION), a pure function of the run's per-cycle Q."""
    cycles = sorted(q_by_cycle)
    K = cycles[-1] if cycles else None
    out = {'definition': SETTLING_DEFINITION, 'N': n, 'K': K, 'early_stop_cycle': early_stop_cycle,
           'reached_cap': K == cap}
    if K is None or K <= n or q_by_cycle.get(n) is None:
        out.update({'status': 'no_post_certification_steps', 'stage_2': {'decision': 'undetermined'}})
        return out
    s = {k: (q_by_cycle[k] - q_by_cycle[k - 1]) for k in range(n + 1, K + 1)
         if q_by_cycle.get(k) is not None and q_by_cycle.get(k - 1) is not None}
    d = {k: -v for k, v in s.items()}
    D_measured = (q_by_cycle[n] - q_by_cycle[K]) if q_by_cycle.get(K) is not None else None
    out.update({'steps': {str(k): s[k] for k in sorted(s)}, 'D_measured': D_measured,
                'resolution_terminal_step_abs': abs(s[K]) if K in s else None,
                'post_N_steps_available': len(s), 'post_N_cycles': K - n})
    if not s:
        out.update({'status': 'no_available_steps', 'stage_2': {'decision': 'undetermined'}})
        return out
    k_peak = max(sorted(d), key=lambda k: (d[k], -k))
    window = [k for k in range(k_peak, K + 1) if k in d]
    contiguous = window == list(range(k_peak, K + 1))
    v1 = len(window) >= FIT_MIN_STEPS and contiguous
    v2 = all(d[k] > 0 for k in window)
    fit = _fit(window, d) if (len(window) >= 2 and v2 and contiguous) else None
    v3 = fit is not None and fit['r'] < 1.0
    valid = v1 and v2 and v3
    all_post = [k for k in range(n + 1, K + 1) if k in d]
    fit_all = (_fit(all_post, d) if (len(all_post) >= 2 and all(d[k] > 0 for k in all_post)
                                     and all_post == list(range(n + 1, K + 1))) else None)
    D_extrap = (fit['r'] * d[K] / (1.0 - fit['r'])) if valid else None
    D_extrap_fitted_endpoint = (fit['r'] * fit['d_hat_last'] / (1.0 - fit['r'])) if valid else None
    D_x0 = (D_measured + D_extrap) if valid else D_measured
    bar_sum = value['bar_sum_resolution_eur']
    if valid:
        decision, basis = ('run' if D_x0 > bar_sum else 'do_not_run'), 'measured_plus_extrapolated'
    elif D_measured is not None and D_measured > bar_sum:
        decision, basis = 'run', 'measured_lower_bound_already_exceeds_resolution'
    elif early_stop_cycle is not None:
        decision, basis = ('run' if (D_measured or 0.0) > bar_sum else 'do_not_run'), 'measured_early_stop_no_extrapolation'
    else:
        decision, basis = 'undetermined', 'lower_bound_below_resolution_without_valid_decay_to_planner'
    V, R_ref = value['V_eur'], value['R_ref']
    R_range = None
    if decision == 'do_not_run' and D_x0 is not None:
        lo, hi = (V - D_x0) / R_ref, V / R_ref
        R_range = [min(lo, hi), max(lo, hi)]
    jump = [k for k in s if abs(s[k]) > bar_cert]
    peak_late = (K - k_peak) < (FIT_MIN_STEPS - 1)
    r = fit['r'] if fit else None
    if jump or (r is not None and r > H2_BAND[1]) or (peak_late and early_stop_cycle is None):
        cls = 'H3'
    elif fit is not None and len(window) >= FIT_MIN_STEPS and H2_BAND[0] <= r <= H2_BAND[1]:
        cls = 'H2'
    elif valid and r < H2_BAND[0]:
        cls = 'H1'
    elif early_stop_cycle is not None:
        cls = 'settled_by_early_stop'
    else:
        cls = 'unclassified'
    preds = {
        'peak_within_a_few_cycles': k_peak <= n + PEAK_WITHIN,
        'geometric_decay_ratio_0.80_0.95': bool(valid and PRED_RATIO[0] <= r <= PRED_RATIO[1]),
        'D_x0_20_60_k': bool(D_x0 is not None and PRED_D[0] <= D_x0 <= PRED_D[1]),
        'stage_2_triggered': decision == 'run',
        'R_settled_ge_0.80': (('pending stage 2' if decision == 'run' else None) if R_range is None
                              else R_range[0] >= PRED_R_SETTLED_MIN),
    }
    out.update({
        'status': 'computed', 'k_peak': k_peak, 'd_peak': d[k_peak], 'fit_window': [window[0], window[-1]],
        'fit': fit, 'fit_all_post_N_reported_not_deciding': fit_all,
        'validity': {'V1_at_least_4_contiguous_steps': v1, 'V2_all_descending': v2, 'V3_r_lt_1': v3, 'valid': valid},
        'D_extrap': D_extrap, 'D_extrap_fitted_endpoint_reported': D_extrap_fitted_endpoint, 'D_x0': D_x0,
        'D_x0_basis': basis, 'D_x0_is_lower_bound': not valid,
        'stage_2': {'decision': decision, 'basis': basis, 'bar_sum': bar_sum, 'D_x0': D_x0,
                    'R_range_if_not_run': R_range, 'V': V, 'R_ref': R_ref},
        'jump_cycles': jump, 'classification': cls, 'predictions_scored': preds,
        'manuscript_statement': (f'certified on residuals at cycle {n}; the objective descended a further '
                                 f'{D_measured:,.2f} EUR over {K - n} cycles'
                                 + (f' (ratio {r:.4f})' if valid else ' (no valid geometric fit)')
                                 if D_measured is not None else None),
    })
    return out


def settling_self_tests():
    """Synthetic Q series through the frozen formula: each outcome class and the stage-2 branches."""
    q0 = CERTIFIED['Q']

    def series(steps, n=N):
        q = {k: q0 + 100.0 * (n - k) for k in range(1, n + 1)}
        q[n] = q0
        for i, st in enumerate(steps):
            q[n + 1 + i] = q[n + i] + st
        return q

    geo = [-4000.0 * (0.85 ** i) for i in range(30)]
    hump = [-4100.0, -4300.0, -4200.0] + [-4200.0 * 0.85 ** (i + 1) for i in range(27)]
    tests = {
        'H1_decay_r0.85_cap': (series(geo), None, lambda a: a['classification'] == 'H1' and a['validity']['valid']
                               and abs(a['fit']['r'] - 0.85) < 1e-9 and a['stage_2']['decision'] == 'run'),
        'H1_hump_then_decay': (series(hump), None, lambda a: a['classification'] == 'H1' and a['k_peak'] == N + 2
                               and a['fit_window'][0] == N + 2),
        'H2_flat_steps': (series([-4000.0 * (0.999 ** i) for i in range(30)]), None,
                          lambda a: a['classification'] == 'H2' and a['stage_2']['decision'] == 'run'),
        'H3_growing': (series([-1000.0 * 1.05 ** i for i in range(30)]), None,
                       lambda a: a['classification'] == 'H3' and not a['validity']['valid']),
        'H3_jump': (series([-3000.0, -20000.0, -2000.0, -1500.0, -1000.0, -800.0]), None,
                    lambda a: a['classification'] == 'H3' and a['jump_cycles'] == [N + 2]),
        'early_stop_small_do_not_run': (series([-900.0, -600.0, -400.0, -300.0, -200.0]), N + 5,
                                        lambda a: a['stage_2']['decision'] == 'do_not_run'
                                        and a['stage_2']['R_range_if_not_run'] is not None
                                        and a['D_x0'] < VALUE['bar_sum_resolution_eur']),
        'early_stop_noisy_signs': (series([-900.0, 300.0, -400.0, 200.0, -100.0]), N + 5,
                                   lambda a: a['classification'] == 'settled_by_early_stop'
                                   and a['stage_2']['basis'] == 'measured_early_stop_no_extrapolation'),
    }
    out = {}
    for name, (q, es, pred) in tests.items():
        a = settling_analysis(q, early_stop_cycle=es, cap=max(q))
        try:
            ok = bool(pred(a))
        except Exception:  # noqa: BLE001
            ok = False
        out[name] = {'ok': ok, 'classification': a.get('classification'), 'k_peak': a.get('k_peak'),
                     'r': (a.get('fit') or {}).get('r'), 'valid': (a.get('validity') or {}).get('valid'),
                     'D_measured': a.get('D_measured'), 'D_x0': a.get('D_x0'), 'stage_2': a.get('stage_2')}
    return out, all(v['ok'] for v in out.values())


# ======================================================================================================================
#  post-run: the replay gate, the hold / capture checks, the cell gates
# ======================================================================================================================
def replay_gate(eval_dir, rec):
    """Rows 1..N of the run's per_cycle_record.jsonl against the certified cell's committed rows: EVERY field except
    REPLAY_EXCLUDED_FIELDS, compared as JSON text (floats by their shortest repr, i.e. bit for bit -- the W90 S16
    rule); plus the initialisation (cycle 0) gross hex. The first differing cycle is k (0 = the initialisation);
    magnitude = the gross difference at k and the largest relative difference over the differing numeric fields.
    NOT a stop: a divergence relabels the run (Addendum 51)."""
    ref = _read_jsonl(_abs(CERTIFIED['per_cycle_record']['path']))
    run = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    ref_by = {r['cycle']: {k: v for k, v in r.items() if k not in REPLAY_EXCLUDED_FIELDS} for r in ref}
    run_by = {r['cycle']: {k: v for k, v in r.items() if k not in REPLAY_EXCLUDED_FIELDS} for r in run}
    ref_ii = (_load(CERTIFIED['initialisation_identity']['path']).get('record') or {}).get('gross_operational_cost_hex')
    run_ii = (rec.get('initialisation_identity') or {}).get('gross_operational_cost_hex')
    first, detail = None, None
    if run_ii != ref_ii:
        first, detail = 0, {'initialisation_gross_hex': [run_ii, ref_ii]}
    per_cycle = []
    for c in range(1, N + 1):
        a, b = run_by.get(c), ref_by.get(c)
        if a is None:
            per_cycle.append({'cycle': c, 'equal': False, 'missing_in_run': True})
            if first is None:
                first, detail = c, {'missing_in_run': True}
            continue
        diff = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True)
                      != json.dumps(b.get(k), sort_keys=True))
        per_cycle.append({'cycle': c, 'equal': not diff, 'fields_differing': diff})
        if diff and first is None:
            rel = {}
            for k in diff:
                x, y = a.get(k), b.get(k)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                    rel[k] = abs(x - y) / max(abs(y), 1e-300)
            ga, gb = a.get('gross_operational_cost'), b.get('gross_operational_cost')
            first, detail = c, {'fields_differing': diff,
                                'gross_difference_run_minus_recorded': (ga - gb) if (ga is not None and gb is not None)
                                else None, 'max_relative_difference': max(rel.values()) if rel else None,
                                'relative_difference_by_field': rel}
    # REPORTED, not gated: the raw trajectory rows
    raw_ref = {r['cycle']: r for r in _load(CERTIFIED['g_trajectory']['path'])['cycle_trajectory']}
    g_run_path = os.path.join(eval_dir, f"g_{certified_spec()['configuration']['arm_label']}.json")
    raw_diff = None
    if os.path.isfile(g_run_path):
        raw_run = {r['cycle']: r for r in json.load(open(g_run_path))['cycle_trajectory']}
        raw_diff = {}
        for c in range(1, N + 1):
            a, b = raw_run.get(c) or {}, raw_ref.get(c) or {}
            d = sorted(k for k in set(a) | set(b) if k not in RAW_ROW_EXPECTED_DIFFERENT
                       and json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True,
                                                                                            default=str))
            if d:
                raw_diff[str(c)] = d
    bitwise = first is None
    return {'status': 'bitwise_through_N' if bitwise else f'diverged_at_cycle_{first}', 'bitwise_through_N': bitwise,
            'first_divergence_cycle': first, 'divergence_magnitude': detail, 'N': N,
            'gated_fields': sorted(next(iter(ref_by.values()))), 'excluded_fields': list(REPLAY_EXCLUDED_FIELDS),
            'tolerance': 'bitwise (JSON text of each field, floats by shortest repr)',
            'initialisation_gross_hex': {'run': run_ii, 'recorded': ref_ii},
            'per_cycle': per_cycle,
            'raw_trajectory_rows_reported_not_gated': {'excluded': list(RAW_ROW_EXPECTED_DIFFERENT),
                                                       'cycles_with_differences': raw_diff},
            'label': ('W98 continuation of the certified x = 0 cell -- replay BITWISE through cycle 72' if bitwise else
                      f'W98 continuation -- SEPARATELY LABELLED RUN: the replay diverged from the certified record at '
                      f'cycle {first}; still informative (Addendum 51)')}


def hold_checks(eval_dir, rec):
    """From continuation_cycle_record.jsonl: no hold acted for cycle <= N (all three flags False); for every cycle > N
    all three held (AA 'off' wherever AA ran, tail applied ON and next-state True, rho unchanged and equal to the
    cycle-N value); one line per cycle; the certificate length 10**9 through the run (0 only on an early-stop cycle)."""
    lines = _read_jsonl(os.path.join(eval_dir, C.CYCLE_FILE))
    summ = rec.get('certification_continuation_summary') or {}
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    replay = [by[c] for c in cycles if c <= N]
    cont = [by[c] for c in cycles if c > N]
    rho_n = (by.get(N) or {}).get('rho', {}).get('rho_after')
    parts = {
        'one_line_per_cycle_contiguous': cycles == list(range(1, (rec.get('cycles_run') or 0) + 1)),
        'no_hold_acted_through_N': all((x.get('aa') or {}).get('hold') in (None, False)
                                       and (x.get('tail_apply') or {}).get('hold') is False
                                       and (x.get('tail_next') or {}).get('hold') is False
                                       and (x.get('rho') or {}).get('hold') is False for x in replay),
        'aa_held_off_after_N': all((x.get('aa') is None and x.get('gross') is None)
                                   or ((x.get('aa') or {}).get('hold') is True
                                       and (x.get('aa') or {}).get('action') == C.AA_OFF_ACTION) for x in cont),
        'tail_held_on_after_N': all((x.get('tail_apply') or {}).get('active_passed') is True
                                    and (x.get('tail_next') or {}).get('returned') is True for x in cont),
        'rho_frozen_after_N_at_cycle_N_values': bool(rho_n) and all(
            (x.get('rho') or {}).get('hold') is True and (x.get('rho') or {}).get('rho_after') == rho_n
            and not (x.get('rho') or {}).get('changed_channels') for x in cont),
        'certificate_length_disabled': all(
            x.get('certificate_length_in_force_at_cycle_end') == C.CERTIFICATION_DISABLED_THRESHOLD
            or (x.get('early_stop') or {}).get('fired') for x in lines),
        'summary_ok': summ.get('ok') is True,
    }
    n_changed = {'aa': sum(1 for x in cont if (x.get('aa') or {}).get('hold_changed_value')),
                 'tail_apply': sum(1 for x in cont if (x.get('tail_apply') or {}).get('hold_changed_value')),
                 'tail_next': sum(1 for x in cont if (x.get('tail_next') or {}).get('hold_changed_value'))}
    return all(parts.values()), {'parts': parts, 'n_cycles_where_the_hold_changed_a_value': n_changed,
                                 'natural_boyd_pass_after_N': {str(x['cycle']): (x.get('aa') or {}).get(
                                     'natural_all_boyd_pass') for x in cont},
                                 'rho_at_N': rho_n, 'summary': summ}


def stopping_check(rec, lines):
    summ = rec.get('certification_continuation_summary') or {}
    K = rec.get('cycles_run')
    es = summ.get('early_stop_cycle')
    if es is not None:
        streak = [(x.get('early_stop') or {}).get('streak') for x in lines if x['cycle'] > N]
        ok = K == es and streak and streak[-1] == C.STAGE1_EARLY_STOP['consecutive_cycles'] \
            and summ.get('stopped_by') == 'early_stop'
        return bool(ok), {'stopped_by': 'early_stop', 'cycle': es, 'cycles_run': K}
    ok = K == CAP and summ.get('stopped_by') == 'cap'
    return ok, {'stopped_by': summ.get('stopped_by'), 'cycles_run': K, 'cap': CAP}


def block_capture(eval_dir, rec, rows):
    """recourse_blocks_all.jsonl: one line per cycle with a gross (81 blocks: 80 network + SALVAGE), reconciling to
    the net recourse; and Addendum 50 item (i) recovered IN FULL for every cycle: per cycle, sum of the block deltas vs
    dQ, sum |delta|, and the five largest |delta| blocks (the complete table stays in the sidecar)."""
    lines = _read_jsonl(os.path.join(eval_dir, C.BLOCKS_FILE))
    ok_cycles = [r['cycle'] for r in rows if r.get('gross_operational_cost') is not None]
    by = {x['cycle']: x for x in lines}
    per_cycle = {}
    for c in sorted(by):
        x = by[c]
        deltas = x.get('deltas_vs_previous_cycle')
        if not deltas:
            continue
        ds = [e for e in deltas if e.get('delta') is not None]
        tot = sum(e['delta'] for e in ds)
        dq = (x['gross_operational_cost'] - by[c - 1]['gross_operational_cost']) if (c - 1) in by else None
        top = sorted(ds, key=lambda e: (-abs(e['delta']), str(e['agent']), str(e['node_id']), str(e['year']),
                                        str(e['day'])))[:5]
        per_cycle[str(c)] = {'dQ': dq, 'sum_block_deltas': tot,
                             'sum_block_deltas_minus_dQ_net': (tot - (x['net_operational_recourse']
                                                                      - by[c - 1]['net_operational_recourse'])),
                             'sum_abs_block_deltas': sum(abs(e['delta']) for e in ds), 'n_blocks': len(ds),
                             'top5': [{k: e[k] for k in ('agent', 'node_id', 'year', 'day', 'delta')} for e in top]}
    parts = {'one_line_per_successful_cycle': sorted(by) == ok_cycles,
             'all_blocks_every_line': all(x.get('n_blocks') == W9.BLOCKS_PER_ROUND + 1 for x in lines),
             'every_line_reconciles_to_net': all(x.get('reconciles_to_net') is True for x in lines)}
    return all(parts.values()), {'parts': parts, 'per_cycle_item_i': per_cycle}


def cell_gates(entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec:
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys()['continuation_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d, 'note': 'SRP1-specific items superseded by G5 / G6'}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W9._solve_profile_check(rec, len(records))
    g6 = g6_v37_evaluate(eval_dir, rec.get('cycles_run'))
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_off'] = (rec.get('post_certification') is None
                                   and not os.path.exists(os.path.join(eval_dir, 'certified_models.pkl')))
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    gates['G10_alpha_row_capture'], detail['G10'] = W9.alpha_row_capture_check(rec, eval_dir, require_compared=False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G12_b_in_force'], detail['G12'] = W0.b_plumbing_check(rec, records, True)
    gates['G13_holds_inert_through_N_and_held_after'], detail['G13'] = hold_checks(eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(rec, _read_jsonl(os.path.join(eval_dir,
                                                                                                   C.CYCLE_FILE)))
    return gates, detail, rec


def _synthetic_run_dir(tmp, steps_after, tamper_cycle=None):
    """A synthetic eval dir produced by the REAL wrappers (make_wrappers) driven through production's call order with
    stand-in originals (the checks module's), including the all-block capture through a stand-in block function."""
    from types import SimpleNamespace
    st, sink = K._fresh_state()
    orig, ret = K.standin_originals(K.Recorder(), penalty_change=False)
    net_by_cycle = {}

    def blocks_fn(pp, models):
        c = models['models']
        net = net_by_cycle[c]
        out = {('TSO', None, 2025 + i // 4, f'd{i % 4}'): net / 80.0 for i in range(20)}
        out.update({('DSO', nid, 2025 + i // 4, f'd{i % 4}'): net / 80.0 for nid in (5, 7, 9) for i in range(20)})
        out[('SALVAGE', None, None, None)] = -0.0
        return out

    fake = SimpleNamespace(_get_operational_recourse_block_components=blocks_fn,
                           _get_operational_objective_component_blocks=lambda pp, m: {})
    w = C.make_wrappers(st, orig, srp_module=fake)
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    w['_capture_convergence_depth_tail_baseline'](object(), admm)
    q, rows = CERTIFIED['Q'] + 100.0 * N, []
    steps = [-100.0] * N + list(steps_after)
    for c, s in enumerate(steps, start=1):
        q += s
        net_by_cycle[c] = q
        ret['gross_script'].append({'gross_operational_cost': q, 'net_operational_recourse': q,
                                    'terminal_salvage_value': 0.0})
        K._drive_cycle(w, st, c, q, active=c > 63, boyd_pass=True, cycle_convergence=True, allow_update=True,
                       sentinels={'admm': admm})
        rows.append({'cycle': c, 'gross_operational_cost': q})
        if st.early_stop_cycle is not None:
            break
    w['_apply_convergence_depth_tail'](object(), admm, False, object(), None)
    files = defaultdict(list)
    for fname, obj in sink:
        if tamper_cycle is not None and fname == C.CYCLE_FILE and obj['cycle'] == tamper_cycle:
            obj = json.loads(json.dumps(obj, default=str))
            obj['tail_apply']['hold'] = True
        files[fname].append(obj)
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            for o in objs:
                handle.write(json.dumps(o, default=str) + '\n')
    rec = {'cycles_run': len(rows), 'certification_continuation_summary': st.summary()}
    return rec, rows


def post_run_evaluator_self_tests():
    """Rule eleven for the post-run evaluators, BEFORE any run: the replay gate on the certified record against itself
    (bitwise), with one gross moved by one ulp at cycle 40 (diverged at 40) and with the initialisation hex changed
    (diverged at 0); the hold / stopping / block-capture checks on a synthetic run produced by the real wrappers
    (pass), and with one cycle-50 hold flag tampered (the hold check fails)."""
    import shutil
    import tempfile
    out = {}
    cert_dir = _abs(CERTIFIED['eval_dir'])
    cert_rec = _load(CERTIFIED['evaluation_record']['path'])
    r = replay_gate(cert_dir, cert_rec)
    out['replay_self_identity'] = {'ok': r['bitwise_through_N'] is True
                                   and r['raw_trajectory_rows_reported_not_gated']['cycles_with_differences'] == {},
                                   'status': r['status']}
    tmp = tempfile.mkdtemp(prefix='w98_replay_selftest_')
    try:
        rows = _read_jsonl(_abs(CERTIFIED['per_cycle_record']['path']))
        rows[39]['gross_operational_cost'] = math.nextafter(rows[39]['gross_operational_cost'], math.inf)
        with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        r = replay_gate(tmp, cert_rec)
        out['replay_one_ulp_at_40'] = {'ok': r['first_divergence_cycle'] == 40 and not r['bitwise_through_N']
                                       and r['divergence_magnitude']['fields_differing'] == ['gross_operational_cost'],
                                       'status': r['status'], 'magnitude': r['divergence_magnitude']}
        bad_rec = json.loads(json.dumps(cert_rec))
        bad_rec['initialisation_identity']['gross_operational_cost_hex'] = '0x1.0p+0'
        r = replay_gate(cert_dir, bad_rec)
        out['replay_init_hex_changed'] = {'ok': r['first_divergence_cycle'] == 0, 'status': r['status']}
    finally:
        shutil.rmtree(tmp)
    for name, steps, tamper, expect in (('synthetic_early_stop_at_75', [-400.0, -300.0, -200.0], None, True),
                                        ('synthetic_tampered_hold_flag', [-400.0, -300.0, -200.0], 50, False)):
        tmp = tempfile.mkdtemp(prefix='w98_hold_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, steps, tamper_cycle=tamper)
            h_ok, h_d = hold_checks(tmp, rec)
            s_ok, s_d = stopping_check(rec, _read_jsonl(os.path.join(tmp, C.CYCLE_FILE)))
            b_ok, b_d = block_capture(tmp, rec, rows)
            out[name] = {'ok': (h_ok is expect) and s_ok and b_ok, 'hold_checks': h_d['parts'], 'stopping': s_d,
                         'block_capture': b_d['parts'], 'expect_hold_checks_pass': expect}
        finally:
            shutil.rmtree(tmp)
    return out, all(v['ok'] for v in out.values())


# ======================================================================================================================
#  the stage spec v37
# ======================================================================================================================
def wall_time_estimate():
    rows = _read_jsonl(_abs(CERTIFIED['per_cycle_record']['path']))
    walls = [r['cycle_wall_s'] for r in rows]
    rec = _load(CERTIFIED['evaluation_record']['path'])
    child = rec['wall_time_s']['child_process_s']
    overhead = child - sum(walls)
    tail = [r['cycle_wall_s'] for r in rows if r['cycle'] >= 64]
    per_tail = sum(tail) / len(tail)
    full = sum(walls) + C.STAGE1_CONTINUATION_CYCLES * per_tail + overhead
    earliest = sum(walls) + C.STAGE1_EARLY_STOP['consecutive_cycles'] * per_tail + overhead
    return {'basis': 'the certified x = 0 cell\'s own per-cycle walls (per_cycle_record cycle_wall_s) and child wall',
            'replay_cycles_1_72_s': sum(walls), 'init_and_terminal_overhead_s': overhead,
            'tail_cycle_mean_s_cycles_64_72': per_tail,
            'cap_102_s': full, 'cap_102_h': full / 3600.0,
            'earliest_early_stop_cycle_75_s': earliest, 'earliest_early_stop_h': earliest / 3600.0,
            'not_included': ('the all-block capture per cycle (two production read functions, the same the recourse-jump '
                             'sidecar already calls every cycle; not measured) and any retry beyond the certified '
                             'run\'s own; the replay repeats the certified run\'s retries if bitwise')}


def verbatim_check():
    text = _norm(open(_abs(BRIEF), encoding='utf-8').read())
    found = {k: _norm(v) in text for k, v in VERBATIM.items()}
    st = L._git_state(BRIEF)
    return {'brief': BRIEF, 'brief_sha256_at_freeze': _sha(BRIEF),
            'brief_git_state': st, 'note': ('Addendum 51 is in the working tree of the brief, NOT committed, at freeze '
                                            '(the Planner\'s file; left unstaged per the task); the quotes are therefore '
                                            'fixed here verbatim, with the brief\'s sha256 at freeze'),
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def launch_command(spec_sha=None):
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u '
            f'{SCRIPT_NAME} --run --spec-sha256 {spec_sha or "<campaign spec sha256>"} '
            f'> {os.path.join(ROOT_REL, "stage1_run_v37_launch.log")} 2>&1')


def code_since_certified_run():
    """Every tracked .py changed between the certified run's freeze commit (the pair spec's git_head) and HEAD, plus
    any uncommitted tracked change: the only MODIFIED file allowed is the harness (the W98 option, C7); ADDED files
    are allowed only when the run's import path cannot reach them (the W93-W98 scripts, uncoordinated_benchmark.py --
    asserted absent from the harness, production and the gates module source)."""
    base = certified_spec()['git_head']
    diff = H._git(['diff', '--name-status', base, 'HEAD', '--', '*.py']).splitlines()
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    allowed_modified = {os.path.relpath(H.HARNESS_PATH, REPO)}
    allowed_added_prefixes = ('p515_s53_w93_', 'p515_s53_w95_', 'p515_s53_w96_', 'p515_s53_w97_', 'p515_s53_w98_')
    changed, bad = [], []
    for line in diff:
        st, rel = line.split('\t', 1)[0], line.split('\t')[-1]
        changed.append({'status': st, 'path': rel})
        if st.startswith('M') and rel in allowed_modified:
            continue
        if st.startswith('A') and (rel.startswith(allowed_added_prefixes) or rel == 'uncoordinated_benchmark.py'):
            continue
        bad.append({'status': st, 'path': rel})
    srcs = ''.join(open(_abs(r)).read() for r in ('p515_s44_campaign_harness.py', 'shared_resources_planning.py',
                                                  'p515_g_g1_g4_admm_gates.py', 'network.py'))
    unreached = 'uncoordinated_benchmark' not in srcs
    return {'certified_run_freeze_commit': base, 'head': H._git(['rev-parse', 'HEAD']), 'changed_py': changed,
            'not_allowed': bad, 'uncommitted_tracked_py': dirty,
            'uncoordinated_benchmark_not_imported_by_the_run_path': unreached,
            'ok': not bad and not dirty and unreached}


def stage_spec_content(checks_inline, checks_file, g6_tests, settle_tests, verb, mem, solver, pre, code_since,
                       post_tests):
    code = {rel: _sha(rel) for rel in (SCRIPT_NAME, 'p515_s53_w98_continuation_hooks.py',
                                       'p515_s53_w98_continuation_checks.py', os.path.relpath(H.HARNESS_PATH, REPO),
                                       'p515_s53_w90_3x3_campaign.py', 'p515_s53_w89_3x3_campaign.py',
                                       'p515_s53_w95_x0_drift_diagnostics.py')}
    keys = expected_keys()
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))
                  and rel != os.path.relpath(H.HARNESS_PATH, REPO)}
    return {
        'schema': 'p515_s53_stage_spec_v37', 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 36, **SPEC_V36},
        'authority': [f'{BRIEF} Addendum 51 (working tree, uncommitted at freeze; sha256 in verbatim_text)',
                      'Planner task W98', 'P5_15_ADDENDUM48_50_3X3_REPORT.md', f"W97 diagnostics {W97['commit']}"],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (terminal salvage 0 at x = 0, so gross '
                                 '= net here); every value in this spec and in the run\'s results is on this convention'),
        'code_sha256': code, 'production_sha256': production, 'code_since_certified_run': code_since,
        'instance_and_certified_cell': {**CERTIFIED, 'node7_cell_reference': NODE7, 'value_and_R': VALUE,
                                        'candidate': certified_entry()['canonical']},
        'run': {
            'stage': 'stage 1 (x = 0) only; stage 2 (the storage cell) only by the decision rule below',
            'campaign_id': CAMPAIGN_ID, 'campaign_root': os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_ID}'),
            'entry_label': ENTRY_LABEL, 'cap': CAP, 'N': N, 'continuation_cycles': C.STAGE1_CONTINUATION_CYCLES,
            'concurrency': CONCURRENCY, 'configuration': stage1_configuration(),
            'entry_options': {k: v for k, v in stage1_entries()[0][2].items()},
            'configuration_identity': ('identical to the certified cell (spec 231558f0): case file, derived instance, '
                                       'AA declaration, ESS ageing baseline and params file, tail {True, 1e-6}, premium '
                                       'alpha 0.5, option (b) on, persistence OFF, thread caps, interpreter, solver path; '
                                       'checked field by field at --freeze and at --run (validate_campaign_spec). The '
                                       'harness differs from the certified run\'s (252996a1) by the W98 option only; '
                                       'C7 proves every committed key unchanged under it'),
            'one_run_rule': ('stands (Addendum 51): nothing else runs on the Mac while stage 1 runs; the SRP1 '
                             'benchmark arms only with measured headroom >= 3 GiB above the continuation\'s peak, the '
                             'author\'s call'),
            'captures': ('per-cycle state (per_cycle_record + g trajectory + the continuation cycle record), per-attempt '
                         'IPOPT floor records (network_ipopt_solve_records*), the per-cycle response capture and the '
                         'terminal captures ON, as for the pair; persistence OFF (the v36 margin rule refused it)'),
        },
        'replay_gate': {
            'reference': CERTIFIED['per_cycle_record'], 'cycles': [1, N],
            'fields': 'EVERY field of per_cycle_record.jsonl except ' + ', '.join(REPLAY_EXCLUDED_FIELDS)
                      + ' (wall-clock, RSS and capture timing) -- i.e. per-cycle gross, net, salvage, objective change and '
                        'tolerance, every Boyd primal / dual ratio and channel pass, all_boyd_pass, cycle_convergence, '
                        'consecutive_converged_cycles, rho after / action / freeze, EFC, and the row-18 response fields; '
                        'plus the initialisation (cycle 0) gross hex',
            'tolerance': 'bitwise (JSON text of each field; floats by shortest repr, i.e. bit for bit -- the W90 S16 rule)',
            'on_divergence': ('record k (0 = initialisation) and the magnitude (the gross difference at k, the largest '
                              'relative difference over the differing numeric fields), RELABEL the run ("SEPARATELY '
                              'LABELLED RUN: the replay diverged at cycle k"), and CONTINUE -- the run is not stopped; '
                              'the post-72 hold stays keyed to cycle N = 72, the recorded certification cycle, not '
                              're-derived from the replay'),
            'reported_not_gated': ('the raw trajectory rows (g_s39_D.json) cycles 1..72, excluding '
                                   'required_consecutive_cycles, which records the raised certificate length by design'),
            'also': 'this replay is the instance\'s reproducibility measurement for the manuscript (Addendum 51)',
            'live_check_in_child': ('non-gating: the child compares each cycle\'s gross hex with the reference as it '
                                    'runs and prints it (continuation_cycle_record.jsonl, [W98-CONTINUATION] lines)'),
        },
        'certification_rule': {
            'through_N': ('evaluated exactly as production (consecutive Boyd-pass cycles; production\'s per-cycle count '
                          'is untouched and recorded); the loop is not allowed to EXIT on it: the certificate length is '
                          'raised to 10**9 at the tail baseline, before cycle 1 (so a divergent replay that certifies '
                          'early still continues). Its only reads are the exit test, the raw row\'s '
                          'required_consecutive_cycles and the INFO print (asserted before any solve); it feeds no '
                          'computed quantity. The loop exit restores 10'),
            'after_N': 'disabled; the run ends at the early stop or at the cap (102)',
        },
        'post_N_hold': {
            'N': N, 'acts_for': 'cycle > 72 only; inert for cycle <= 72',
            'AA_off': ('production\'s own AA step is called with all_boyd_pass forced True on a copy: its "off" branch, '
                       'no extrapolation, no write-back; asserted per cycle'),
            'tail_on': ('the tail is applied ON at the top of every cycle > 72 and the end-of-cycle next-state returns '
                        'True (production\'s AA-off == predicate check is bypassed only there, because the AA hold '
                        'makes AA off by construction; the natural predicate is recorded)'),
            'rho_frozen': ('production\'s penalty update is called with allow_update=False for cycle > 72 (its scaling '
                           'loop, the only place rho / gamma are written, does not run); rho and gamma before == after '
                           'asserted exactly; at x = 0 all three channels are already frozen from cycle 63 (W97), so '
                           'the hold changes nothing on a bitwise replay and guarantees it on a divergent one'),
            'mechanism': ('harness-side: pass-through wrappers on six production functions, installed first in the '
                          'child (p515_s53_w98_continuation_hooks); NO production module is edited'),
            'inseparability': ('W97 established that AA-off and tail-on coincide by construction (tail on at AA-off + 1). '
                               'This run holds BOTH and cannot separate them: the continuation discriminates by the '
                               'SHAPE of the post-certification steps (hump then geometric decay, constant, or growing / '
                               'a jump), not by separating AA from the tail.'),
        },
        'inertness_proof': {
            'checks_script': 'p515_s53_w98_continuation_checks.py',
            'committed_output': checks_file,
            'inline_rerun_at_freeze': {'all_hold': checks_inline['all_hold'],
                                       'inertness_proof': checks_inline['inertness_proof'],
                                       'per_check_holds': {k: v['holds'] for k, v in checks_inline['checks'].items()}},
            'rerun_before_launch': 'the --run mode re-runs every check and refuses unless all hold',
        },
        'early_stop_rule': {
            'rule': ('for cycle k > 72: s_k = Q_k - Q_(k-1) (gross); the run stops at the end of the first cycle k at '
                     'which |s_j| < 500.0 EUR for the 3 consecutive post-certification cycles j = k-2, k-1, k (the '
                     'earliest possible stop is k = 75); a step >= 500 EUR resets the count; a cycle whose local solves '
                     'failed has no Q, resets the count, and makes the next step unavailable'),
            'threshold_eur': C.STAGE1_EARLY_STOP['abs_gross_step_below_eur'],
            'consecutive_cycles': C.STAGE1_EARLY_STOP['consecutive_cycles'],
            'mechanism': ('the certificate length is set to 0 on that cycle, so production\'s own exit test ends the '
                          'loop at its end (production\'s only exit besides the cap); otherwise the run ends at the cap, '
                          'cycle 102'),
            'no_action_through_N': 'the rule is not evaluated for cycle <= 72 (proved by check C5)',
        },
        'instrumentation': {
            'all_blocks_every_cycle': ('recourse_blocks_all.jsonl: every block of production\'s '
                                       '_get_operational_recourse_block_components (80 network blocks + SALVAGE) and '
                                       'every objective component of _get_operational_objective_component_blocks, every '
                                       'cycle, with the deltas against the previous cycle -- the recourse-jump sidecar '
                                       'keeps only the top 10 (W96). This recovers Addendum 50 item (i) IN FULL for the '
                                       'historical window 63-72 (via the replay) and for the continuation'),
            'continuation_cycle_record': ('continuation_cycle_record.jsonl: per cycle the hold state (natural and held '
                                          'values), gross, step, early-stop streak, certificate length in force, the '
                                          'live replay comparison'),
        },
        'g6_v37': {
            'verbatim': VERBATIM['g6'],
            'operational': ('terminal round T = the run\'s last cycle; per block (80): attempts ordered by the ladder '
                            '(primary, recovery, recovery_tier2), the FINAL attempt is the accepted one (no superseded '
                            'attempt accepted, ladder well formed); its exit is exactly "Optimal Solution Found."; '
                            'compl_inf_tol in force = the tail value 1e-6; its four terminal error metrics (parsed from '
                            'its own IPOPT log byte range, W95\'s parser by import) within the tolerances in force: '
                            'scaled overall NLP error <= tol, unscaled dual infeasibility <= dual_inf_tol, constraint '
                            'violation <= constr_viol_tol, complementarity <= compl_inf_tol (IPOPT defaults when not '
                            'listed). Gate: 80 blocks and every block passes. mu_final / mu_floor and floor_status '
                            'REPORTED per solve, NEVER gated.'),
            'self_tests_on_committed_data': g6_tests,
        },
        'gates': {
            'G1-G5, G7, G9-G12': 'as the pair (W90 cell_gates), G10 without the identity reference (the replay gate owns it)',
            'G6': 'v37 as above (replaces v36\'s floor test)', 'G8': 'persistence off: no post-certification, no pickle',
            'G13': 'the holds: inert through 72 (every flag False), held after 72 (AA off, tail on, rho = the cycle-72 '
                   'values), certificate length 10**9',
            'G14': 'all-block capture: one line per successful cycle, 81 blocks, reconciling to net recourse',
            'G15': 'stopping consistent: early stop at k with the streak complete, or the cap',
            'replay_gate': 'not a pass/fail gate: it sets the run\'s label (bitwise vs separately labelled)',
            'applies_to': 'the one stage-1 entry',
            'post_run_evaluator_self_tests': post_tests,
        },
        'settling_analysis': {'definition': SETTLING_DEFINITION, 'self_tests_on_synthetic_series': settle_tests,
                              'implementation': f'{SCRIPT_NAME}:settling_analysis (pure function of the per-cycle Q)'},
        'stage_2_decision_rule': {'verbatim_addendum_51': VERBATIM['stage_2_decision_rule'],
                                  'verbatim_planner_task_w98': TASK_W98_STAGE2_TEXT,
                                  'operational': SETTLING_DEFINITION['stage_2_decision'],
                                  'constants': VALUE},
        'expert_predictions_verbatim': VERBATIM['expert_predictions'],
        'competing_outcomes_verbatim': VERBATIM['competing_outcomes'],
        'addendum_51_stage_1_and_analysis_verbatim': {'stage_1': VERBATIM['stage_1'], 'analysis': VERBATIM['analysis']},
        'verbatim_text': verb,
        'labelling_and_identity': {
            'label': C.LABEL,
            'evaluation_key_analysis': (
                'the cap and the certificate length (minimum_consecutive_converged_cycles / spec '
                'required_consecutive_cycles) are spec-level and do NOT enter evaluation_key: without a keyed '
                'declaration the continuation would carry the certified key f6e9cd53 exactly. W98 adds the entry option '
                'certification_continuation, which DOES enter the key: key = sha256({base_evaluation_key, '
                'certification_continuation}); every key without it is byte-identical (C7: 7,032 committed entries)'),
            'keys': keys, 'pre_launch_assertion_at_freeze': pre,
        },
        'memory_preflight': {
            'required_bytes': float(certified_spec()['extra']['required_pair_bytes']),
            'required_gib': float(certified_spec()['extra']['required_pair_bytes']) / GIB,
            'basis': ('the pair\'s own preflight requirement (g x P_on, footprint; spec 231558f0 required_pair_bytes); '
                      'the certified x = 0 child\'s measured footprint peak was 20,517,044,400 B (19.108 GiB), below it; '
                      'per-cycle RSS was flat from cycle 7 (ru_maxrss 15.12-15.16 GiB), so 30 more cycles are not '
                      'expected to raise the peak (not measured)'),
            'rule': 'available (hw.memsize - wired - anonymous - compressor-occupied) >= required, measured at --run',
            'measured_at_freeze_non_gating': mem,
            'persistence': 'OFF (the v36 margin rule refused it; no certified_models.pkl)',
        },
        'solver': solver,
        'solve_profile_declared': {
            'parent': 'never solves: every launcher guard permitted=(), verify(0)',
            'child': ('RECONCILED PER EVENT (G5), as the pair: 83 x (cycles_run + 1) + every retry attempted (80 network '
                      'blocks + 3 ESSO per round; round 0 = initialisation); cycles_run in [75, 102] -> base 6,308 to '
                      '8,549'),
        },
        'expected_wall_time': wall_time_estimate(),
        'launch_command_template': launch_command(),
    }


def _find_stage_spec():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    sha = _sha(rel)
    if hits[0] != f'{SPEC_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'v{SPEC_VERSION} file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def load_stage_spec():
    rel, sha = _find_stage_spec()
    if rel is None:
        raise RuntimeError(f'frozen stage spec v{SPEC_VERSION} not found')
    return rel, sha, _load(rel)


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks()


def _checks_file_state():
    doc = _load(CHECKS_REL) if os.path.isfile(_abs(CHECKS_REL)) else {}
    return {'path': CHECKS_REL, 'sha256': _sha(CHECKS_REL) if doc else None,
            'manifest': CHECKS_MANIFEST_REL, 'committed_clean': _committed_clean(CHECKS_REL) if doc else False,
            'all_hold': doc.get('all_hold'), 'inertness_proof': doc.get('inertness_proof'),
            'guard_verify_0_failures': (doc.get('guard') or {}).get('verify_0_failures'),
            'code_sha256_at_check': doc.get('code_sha256')}


def freeze_spec(started):
    tag = f'W98-V{SPEC_VERSION}'
    failures, _ev = _common_checks()
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'v{SPEC_VERSION} already exists (write-once): {existing}')
    cf = _checks_file_state()
    if not (cf['all_hold'] is True and cf['committed_clean'] and cf['guard_verify_0_failures'] == []):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in ('p515_s53_w98_continuation_hooks.py', 'p515_s53_w98_continuation_checks.py',
                os.path.relpath(H.HARNESS_PATH, REPO), 'shared_resources_planning.py', 'admm_anderson_acceleration.py'):
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    code_since = code_since_certified_run()
    if not code_since['ok']:
        failures.append(f'code changed since the certified run beyond the allowed set: {code_since}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f"inline re-run of the zero-solve checks fails: "
                        f"{[k for k, v in checks_inline['checks'].items() if not v['holds']]}")
    g6_tests, g6_ok = g6_v37_self_tests()
    if not g6_ok:
        failures.append(f'G6 v37 self-tests not as declared: {g6_tests}')
    settle_tests, settle_ok = settling_self_tests()
    if not settle_ok:
        failures.append(f'settling-analysis self-tests fail: {[k for k, v in settle_tests.items() if not v["ok"]]}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator self-tests fail: {[k for k, v in post_tests.items() if not v["ok"]]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found in the brief: {verb["found_whitespace_normalised"]}')
    solver = solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    pre = pre_launch_assertion()
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails: {pre["parts"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    mem = L.memory_preflight(1)
    content = stage_spec_content(checks_inline, cf, g6_tests, settle_tests, verb, mem, solver, pre, code_since,
                                 post_tests)
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError(f'v{SPEC_VERSION} written bytes do not hash to the name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] wrote {rel} sha256={sha} (predecessor v36 {SPEC_V36['sha256']})")
    _log(f"[{tag}] inertness proof (inline): {json.dumps(checks_inline['inertness_proof'])}")
    _log(f"[{tag}] checks per id: { {k: v['holds'] for k, v in checks_inline['checks'].items()} }")
    _log(f"[{tag}] G6 v37 self-tests: { {k: (v.get('observed_pass'), v['ok']) for k, v in g6_tests.items()} }")
    _log(f"[{tag}] settling self-tests: { {k: v['ok'] for k, v in settle_tests.items()} }")
    _log(f"[{tag}] post-run evaluator self-tests: { {k: v['ok'] for k, v in post_tests.items()} }")
    _log(f"[{tag}] keys: base {pre['base_key_without_continuation']} (== certified {CERTIFIED['eval_key'][:16]}: "
         f"{pre['parts']['base_key_equals_certified_key']}); continuation {pre['continuation_key']}")
    _log(f"[{tag}] memory at freeze (non-gating): available {mem.get('available_gib')} GiB; required at run "
         f"{content['memory_preflight']['required_gib']:.4f} GiB")
    w = content['expected_wall_time']
    _log(f"[{tag}] expected wall: cap {w['cap_102_h']:.2f} h; earliest early stop {w['earliest_early_stop_h']:.2f} h")
    _finish(0, '-- next: --freeze')


# ======================================================================================================================
#  campaign freeze and run
# ======================================================================================================================
def freeze(started):
    tag = 'W98-FREEZE'
    failures = H.check_campaign_preconditions(campaign_root(), extra_clean_files=EXTRA_CLEAN_FILES)
    more, _ev = _common_checks()
    failures += more
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        if not _committed_clean(ss_rel):
            failures.append('the stage spec v37 is not committed / clean')
        if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
            failures.append('the launcher / hooks / checks / harness changed since v37 froze')
    except RuntimeError as error:
        failures.append(str(error))
        ss = None
    pre = pre_launch_assertion()
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails: {pre["parts"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
             'stage_spec': ss_pin, 'label': C.LABEL, 'certified_cell': {k: CERTIFIED[k] for k in (
                 'campaign_id', 'eval_dir', 'eval_key', 'certification_cycle', 'Q', 'per_cycle_record')},
             'expected_eval_key': pre['continuation_key'], 'base_key_without_continuation': pre['base_key_without_continuation'],
             'required_bytes': ss['memory_preflight']['required_bytes'],
             'objective_convention': ss['objective_convention'],
             'solve_claim': ss['solve_profile_declared'], 'pre_launch_assertion_at_freeze': pre}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        campaign_root(), CAMPAIGN_ID, stage1_entries(), configuration=stage1_configuration(), cap=CAP,
        concurrency=CONCURRENCY, authority=[f'{BRIEF} Addendum 51', 'Planner task W98', ss_rel],
        required_consecutive_cycles=10, extra=extra)
    checks = validate_campaign_spec(spec, ss_pin)
    pre_frozen = pre_launch_assertion(spec)
    _log(f'[{tag}] frozen campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for e in spec['candidates']:
        _log(f"[{tag}]   {e['label']}: eval_key={e['eval_key']} eval_dir={e['eval_dir']} working_dir_ids="
             f"{e['working_dir_ids']} post_certification={e['post_certification']}")
    _log(f'[{tag}] spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] pre-launch assertion on the FROZEN spec: holds={pre_frozen['holds']} parts={pre_frozen['parts']} "
         f"(committed keys scanned {pre_frozen['n_committed_keys_scanned']})")
    _log(f'[{tag}] launch command: {launch_command(spec_sha)}')
    ok = all(checks.values()) and pre_frozen['holds']
    _finish(0 if ok else 1, f'freeze {"OK" if ok else "NOT OK"}')


def run(started, spec_sha256):
    tag = 'W98-RUN'
    failures, _ev = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('stage spec v37 not committed / clean')
    if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
        failures.append('the launcher / hooks / checks / harness changed since v37 froze')
    root = campaign_root()
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign spec not committed / clean')
    checks = validate_campaign_spec(spec, {'path': ss_rel, 'sha256': ss_sha})
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append('the zero-solve checks (inertness proof) do not all hold now')
    pre = pre_launch_assertion(spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["parts"]}')
    solver = solver_check()
    if not solver['ok'] or solver['sha256'] != ss['solver']['sha256']:
        failures.append(f'solver path / binary differs: {solver}')
    mem = W0._arm_preflight(int(math.ceil(ss['memory_preflight']['required_bytes'])), 'W98 stage 1')
    _log(f"[{tag}] memory preflight: available {mem.get('available_gib')} GiB required {mem['required_gib']:.4f} GiB -> "
         f"{'PASS' if mem['pass'] else 'REFUSE'}")
    if not mem['pass']:
        failures.append(f"memory preflight REFUSED: {mem.get('available_gib')} < {mem['required_gib']}")
    if failures:
        for fl in failures:
            _log(f'[{tag} PRECONDITION FAILED] {fl}')
        _finish(1)
    entry = spec['candidates'][0]
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; entry {entry['label']} eval_key {entry['eval_key']}; cap {spec['cap']}; lock {lock}")
    mon = W0.ChildFootprintMonitor(os.path.join(root, 'evals', entry['eval_dir'], 'launch.json'), entry['eval_key'])
    mon.start()
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        footprint = mon.stop()
        H.release_campaign_lock(expected_pid=os.getpid())
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    try:
        gates, detail, rec = cell_gates(entry, eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
        gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}, {}
    analysis, replay = {}, {}
    try:
        replay = replay_gate(eval_dir, rec)
        rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
        q = {r['cycle']: r.get('gross_operational_cost') for r in rows}
        summ = rec.get('certification_continuation_summary') or {}
        analysis = settling_analysis(q, early_stop_cycle=summ.get('early_stop_cycle'))
        analysis['predictions_scored']['replay_bitwise_through_72'] = replay.get('bitwise_through_N')
        analysis['rule_ten_per_post_N_cycle'] = {
            str(r['cycle']): (r['objective_change_abs'] / r['objective_tolerance']
                              if r.get('objective_change_abs') is not None and r.get('objective_tolerance') else None)
            for r in rows if r['cycle'] > N}
    except Exception as error:  # noqa: BLE001
        analysis = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'label': replay.get('label'), 'replay_gate': replay, 'gates': gates, 'gates_pass': all(gates.values()),
               'gate_detail': detail, 'settling_analysis': analysis, 'pre_launch_assertion': pre,
               'inertness_proof_rerun_before_launch': checks_inline['inertness_proof'],
               'memory_preflight_at_run': mem, 'footprint_peak_reported_not_gated': footprint,
               'solver': solver, 'batch_info': batch, 'parent_view': (records[0] or {}).get('parent_view'),
               'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    _log(f"[{tag}] label: {replay.get('label')}")
    _log(f"[{tag}] gates {gates}")
    _log(f"[{tag}] settling: status {analysis.get('status')} K {analysis.get('K')} D_measured "
         f"{analysis.get('D_measured')} r {(analysis.get('fit') or {}).get('r')} valid "
         f"{(analysis.get('validity') or {}).get('valid')} D_x0 {analysis.get('D_x0')} class "
         f"{analysis.get('classification')} stage 2 {(analysis.get('stage_2') or {}).get('decision')}")
    code = 0 if (all(gates.values()) and _guards_ok(g) and 'error' not in analysis) else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.freeze_spec:
            freeze_spec(started)
        elif args.freeze:
            freeze(started)
        else:
            if not args.spec_sha256:
                parser.error('--run requires --spec-sha256')
            run(started, args.spec_sha256)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
