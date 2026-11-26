"""
P5.15 Addendum 54 Ruling 1, Planner task W105 -- the C* SETTLING EXTENSION diagnostic: frozen stage spec v41
(predecessor v40 9bc1779d, superseded before any run, NOT edited; v40's predecessor v39 8a612429), the campaign freeze,
the run, and the zero-solve summary / scorer. BUILT AND FROZEN IN W105, RE-FROZEN IN W108; NO RUN IS LAUNCHED BY THE
WORKER (the Planner launches).

WHY v41 (Planner task W108, before any run). v40 (9bc1779d) and its campaign spec s53_w105_c_star_ext (1a9f1f24,
commit 4461e077) were frozen and committed; W107's launch was then refused at the preconditions with zero solves
(run_c_star_ext_v40_launch.log, 53f01ba7): the zero-solve checks' K required the extension key 8864266d to be absent
from EVERY committed campaign spec, which the committed campaign spec 1a9f1f24 itself falsifies, so the checks the
--run mode re-runs could never hold. The checks module (r2) now excludes the W105 stage root, exactly as
pre_launch_assertion does; a refusal now logs the failing sections and items; --preconditions-only runs every --run
precondition and stops before the campaign lock and the child. v40 pins the r1 module, so v40, 1a9f1f24 and the r1
checks output are superseded (kept, never run); v41 carries every v40 content item unchanged except the code pins,
the defect note and what the new campaign root / id and the new checks output imply. Same declaration, same eval key
8864266d, new campaign id s53_w105_c_star_ext_r2.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 54 Ruling 1 (option (a): the 100-cycle diagnostic extension; H_ess-flat;
"Record per cycle"; the predictions; the outcomes); TASKS.md Addendum 54 order (W105); Planner task W105 (the operational
definitions P_a-P_d, the refutation triggers, the outcome mapping).

THE CELL. c_star_ext = the SRP1 corner plan C* (recert cell c_star, eval key 96c5aa50; W104 eval key 4bf36c15, evidence
82dcebb7), run under EXACTLY W104's configuration (= the recert's: case file + AA keep_memory declaration, ESS ageing
baseline C2, tight tail {True, 1e-6}, persist_certified_models, no overrides, no option (b)), plus the keyed entry option
`settling_extension` (`p515_s53_w105_settling_extension_hooks`): cycles 1..187 replayed and gated bitwise against W104's
committed records with ABORT on the first divergence, W104's holds after cycle 87 (AA off, tail on, rho frozen), a
FIXED length of 287 cycles (cap 287; nothing ends the run earlier), the settling rule evaluated report-only, and the
per-cycle creep captures (Q by block and component, ESS schedule movement, all six Boyd residuals).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-spec                   ZERO SOLVES. v41 (write-once, named by its sha256): re-runs the W105 zero-solve checks
                                  inline and pins their committed output; the scorer and post-run evaluator self-tests;
                                  the pre-launch assertion; the W104 reference slopes; the wall estimate.
  --freeze                        ZERO SOLVES. The campaign spec (s53_w105_c_star_ext_r2), pinning v41; the pre-launch
                                  assertion on the frozen spec; the exact launch command.
  --run --spec-sha256 S           THE RUN (NOT RUN IN W105). Preconditions (the zero-solve checks re-run, the pre-launch
                                  assertion, the memory preflight, the solver path, the run-lock), H.evaluate on the one
                                  entry, the gates, the diagnostic report (predictions scored); results + manifest.
  --run --spec-sha256 S --preconditions-only
                                  ZERO SOLVES (W108). Every --run precondition above, in the same order and with the
                                  same code; then STOPS before the campaign lock and the child (no lock, no evaluation,
                                  nothing written but the log). Exit 0 iff every precondition holds.
  --summarize                     ZERO SOLVES. Re-scores the committed results with the frozen scorer.

Exit codes: 0 done / every gate holds; 1 a gate / harness / guard / precondition failure (a replay divergence aborts
the run: the child exits 1 with its barrier record; the launcher reports the cycle and magnitude and exits 1).
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W105 C* settling-extension launcher (never solves)').install()

# The W101 launcher (and through it the W98 / W90 / W89 / W86 chain, each arming its own permitted=() guard): the
# generic gates and helpers. The W105 checks (arms its own guard). The W105 hooks.
import p515_s53_w101_srp1_continuation_campaign as W101L  # noqa: E402
import p515_s53_w105_extension_checks as K  # noqa: E402
import p515_s53_w105_settling_extension_hooks as E  # noqa: E402
import settling_criterion as SC  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L = W101L.H, W101L.L, W101L.X, W101L.W9, W101L.W98L
def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


# every guard armed by this process (each verified at exactly 0 and uninstalled once)
GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(W101L.GUARDS) + (('w105_parent', PARENT_GUARD),))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w105_c_star_extension_campaign', 'p515_s53_w101_srp1_continuation_campaign',
                          'p515_s53_w98_continuation_campaign')
STAGE_TEXT = ('P5.15 Addendum 54 Ruling 1, W105 -- C* settling extension diagnostic: C* replayed under W104\'s '
              'configuration, gated bitwise per cycle against W104\'s committed records through 187 (abort on the first '
              'divergence), W104\'s holds after 87 (AA off, tight tail on, rho frozen), then exactly 100 more cycles '
              '188-287 under the same holds; the settling rule report-only; per cycle: Q by block and component, '
              'sum |dp_ess| per node and side, all six Boyd residuals')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W105_ROOT_REL
SPEC_PREFIX = 'frozen_s53_spec_v41_'
SPEC_VERSION = 41
SPEC_V39 = {'path': os.path.join(_P53, 'frozen_s53_spec_v39_8a612429.json'),
            'sha256': '8a61242924cee1690c0d9f9f25a0ef6bdc90a519d089415898b44d5983e4c756'}
SPEC_V40 = {'path': os.path.join(_P53, 'frozen_s53_spec_v40_9bc1779d.json'),
            'sha256': '9bc1779da9a44fddc99308701185f62b5d58a29b82577473a750b8f6f0b68785'}
# W108: superseded before any run (kept, never edited, never run); each re-verified committed and unchanged
SUPERSEDED = {
    'stage_spec_v40': SPEC_V40,
    'campaign_spec_s53_w105_c_star_ext': {'path': K.V40_CAMPAIGN_SPEC_REL,
                                          'sha256': '1a9f1f24233f6fe9905a40fc185e09898aa6b2f237566636fe5fb7da261cc8c2'},
    'zero_solve_checks_r1': {'path': os.path.join(ROOT_REL, 'zero_solve_checks', 'w105_zero_solve_checks.json'),
                             'sha256': '9688a018dae16cf57cfe06ce87dafacaae9a027d82057ad32c7904b2774b159a'},
    'refused_v40_launch_log': {'path': os.path.join(ROOT_REL, 'run_c_star_ext_v40_launch.log'),
                               'sha256': '53f01ba73608627451a87b59eb4f3b276bc86ae1998c474d4643f024aec3529d'},
}
DEFECT_NOTE = {
    'task': 'Planner task W108 (W107 launch refused with zero solves)',
    'defect': ('p515_s53_w105_extension_checks.py tests_K (r1, 49342e8c) required the extension eval key 8864266d to '
               'appear in NO committed campaign spec; W105 then committed its own campaign spec 1a9f1f24 (4461e077), '
               'which carries it, so K failed and the --run inline checks (_run_checks_inline) refused every launch. '
               'Reproduced zero-solve in W108: K the only failing section; the only holder the v40 campaign spec'),
    'fix': ('tests_K (r2) excludes the W105 stage root, exactly as pre_launch_assertion does '
            '(L.committed_eval_keys(exclude_roots=(ROOT_REL,))): the key must appear in no committed campaign spec '
            'outside its own root; negative controls (a planted spec outside the root; a planted spec in a sibling '
            'directory sharing the root name prefix) refused; positive controls (a planted spec in the r2 campaign '
            'root; the committed v40 campaign spec) accepted. Everything else in K unchanged'),
    'logging': ('a failing _run_checks_inline now logs the failing sections and, within each, the failing items '
                '(_failing_check_items; logging only)'),
    'preconditions_only_mode': ('--run --spec-sha256 S --preconditions-only: every --run precondition, then a stop '
                                'before the campaign lock and the child (zero solves); added in W108 because the '
                                'launcher had no dry-run mode'),
    'unchanged': ('the declaration, the eval key 8864266d, the cell, the configuration, the holds, cap 287, the '
                  'captures, the definitions, the predictions, the scorer, the gates'),
    'changed': ('schema / version / predecessor; the code pins; the campaign id and root (s53_w105_c_star_ext_r2; the '
                'v40 root is write-once) and what derives from them (working-dir ids, launch command, log name); the '
                'committed zero-solve checks output (r2) and its pre-launch / inline results; freeze-time records (git '
                'head, UTC, code_since_w104_launch, memory, the measured capture overhead and the wall estimate '
                'derived from it)'),
    'superseded_before_any_run': SUPERSEDED,
}
W104_PINS = {
    'campaign_spec': {'path': os.path.join(E.W104_ROOT, 'campaign_spec_s53_w101_srp1_cont_c_star_1a7483ef.json'),
                      'sha256': '1a7483ef17ef8e6054bc18c42fc9a0dd219ff93217f6a7ad6bb9a1130b45c7ea'},
    'campaign_results': {'path': os.path.join(E.W104_ROOT, 'campaign_results.json'),
                         'sha256': '20832469af3a7145b2d47c6db83451c4c0cd6564e69883b09a0cc750ed8fa059'},
    'campaign_manifest': {'path': os.path.join(E.W104_ROOT, 'campaign_manifest_sha256.json'),
                          'sha256': '5b8a7a5edd3d0b1a099ec5c1cc7e2ed5906e54a6889f403ca021326a49c12fe9'},
}
CAMPAIGN_ID = K.EXT_CAMPAIGN_ID  # W108: s53_w105_c_star_ext_r2 (the v40 id's root is write-once)
CELL = E.CELL
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
SUMMARY_FILE = 'w105_extension_summary.json'
SUMMARY_MANIFEST = 'w105_extension_summary_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
SOLVER_PATH = '/usr/local/bin/ipopt'
PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
N_NETWORK_BLOCKS = 48
N_ESSO = 3
SOLVES_PER_ROUND = N_NETWORK_BLOCKS + N_ESSO
EXTRA_CLEAN_FILES = (SCRIPT_NAME, 'p515_s53_w105_settling_extension_hooks.py', 'p515_s53_w105_extension_checks.py',
                     'p515_s53_w101_srp1_continuation_campaign.py', 'p515_s53_w101_settling_continuation_hooks.py',
                     'p515_s53_w101_continuation_checks.py', 'settling_criterion.py', 'interface_dual_capture.py',
                     'gate_result_io.py', 'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
                     'p515_s53_w98_continuation_checks.py', 'p515_s53_w90_3x3_campaign.py',
                     'p515_s53_w89_3x3_campaign.py', 'p515_s53_w95_x0_drift_diagnostics.py',
                     'p515_s53_w86_tail_recert_campaign.py', H.ESS_PARAMS_FILE_REL,
                     'shared_energy_storage_parameters.py', 'shared_energy_storage_data.py', 'network.py')
CODE_PINNED = ('p515_s53_w105_c_star_extension_campaign.py', 'p515_s53_w105_settling_extension_hooks.py',
               'p515_s53_w105_extension_checks.py', 'p515_s53_w101_srp1_continuation_campaign.py',
               'p515_s53_w101_settling_continuation_hooks.py', 'p515_s53_w101_continuation_checks.py',
               'settling_criterion.py', 'interface_dual_capture.py', 'gate_result_io.py', 'p515_s44_campaign_harness.py',
               'p515_g_g1_g4_admm_gates.py', 'p515_s53_w98_continuation_campaign.py',
               'p515_s53_w98_continuation_hooks.py', 'p515_s53_w98_continuation_checks.py',
               'p515_s53_w86_tail_recert_campaign.py', 'p515_s53_w90_3x3_campaign.py', 'p515_s53_w89_3x3_campaign.py',
               'p515_s53_w89_g6_final_attempt_reeval.py', 'p515_s53_w95_x0_drift_diagnostics.py',
               'p515_gate_result_bool_typing_test.py', 'shared_energy_storage_data.py', 'network.py')
PRODUCTION_UNCHANGED_SINCE_W104 = ('shared_resources_planning.py', 'network.py', 'admm_parameters.py',
                                   'admm_anderson_acceleration.py', 'model_construction_helpers.py',
                                   'shared_energy_storage_data.py')
W104_LAUNCH_GIT = '953b9bcd'

# ---- verbatim text (Addendum 54; checked against the committed brief, whitespace-normalised, at every freeze) --------
VERBATIM = {
    'ruling_1': '**Ruling 1 — C\\*: option (a), 100-cycle diagnostic extension (≈ 2.7 h), with a hypothesis to test.**',
    'h_ess_flat': '**H_ess-flat:** the ESSO carries no economic term (ε-throughput regularizer only)',
    'record_per_cycle': ('**Record per cycle:** per-block ΔQ (TSO, DSO, ESSO), Σ|Δp_ess| per node, the channel whose '
                         'primal residual is rising.'),
    'predictions': ('**Predictions:** the creep sits in TSO generation cost with ESS schedules still moving (Σ|Δp_ess| '
                    'not decaying) and the rising residual is the ESS channel; the rate stays within a factor 2 of '
                    '−265 €/cycle over 100 cycles.'),
    'refuted': 'refuted (creep in DSO blocks, or decays, or the residual leaves tolerance) → Advisor review before the '
               'campaign is sized.',
}

# ---- the operational definitions and predictions, recorded BEFORE any run ---------------------------------------------
WINDOW = (188, 287)
QUARTERS = ((188, 212), (213, 237), (238, 262), (263, 287))
FIRST_Q, LAST_Q = QUARTERS[0], QUARTERS[-1]
P_D_BOUNDS = (-530.0, -132.5)
DEFINITIONS = {
    'objective': 'Q_k = gross_operational_cost of cycle k (settlement excluded), per_cycle_record.jsonl',
    'dQ_k': 'Q_k - Q_(k-1)',
    'window': 'the extension window 188..287 (100 cycles); Q_187 is the last replayed (W104) cycle',
    'total_change': 'DQ = Q_287 - Q_187 (= the sum of dQ_k over 188..287)',
    'agent_component_change': ('D[a, c] = agents[a][c] at cycle 287 - at cycle 187, from creep_diagnostic_per_cycle.jsonl '
                               'q_decomposition.agents (a in TSO, DSO_5, DSO_7, DSO_9, ESSO; c in the components + '
                               'other + value); the components of a block sum to its contribution to Q'),
    'P_a': ('the creep sits in TSO generation cost: share_TSO_gen = D[TSO, generation_cost] / DQ >= 0.5 (a positive '
            'ratio: the same sign). Reported beside: every D[a, c] / DQ, the DSO share (sum of D[DSO_n, value]) / DQ, '
            'the ESSO share (0 by construction), the reconciliation residual'),
    'P_b': ('the ESS schedules are still moving: with S_k = sum over ESS nodes of sum over years, days, periods of '
            '|p(k) - p(k-1)| on the ESSO side (consensus_vars ess esso = the ESSO\'s es_pnet, MW), '
            'ratio_b = mean(S_k, 263..287) / mean(S_k, 188..212) >= 0.5. Reported beside, per node and per side (esso, '
            'tso, dso, z; p and q; charge / discharge per side) with the same ratio. PRIMARY SIDE = ESSO (Worker '
            'reading of "Sigma |dp_ess|": the ESSO\'s own schedule; the network-side copies differ from it only by the '
            'ESS primal residual)'),
    'P_c': ('the rising residual is the ESS channel: slope_g = the ordinary least-squares slope of '
            'boyd_{g}_primal_ratio on k over 188..287 (per_cycle_record), g in v, pf, ess; P_c holds iff argmax_g '
            'slope_g == ess AND slope_ess > 0. All six slopes (primal and dual) reported. PRE-REGISTERED AGAINST: W104 '
            'found pf_primal rising 0.013 -> 0.127 over 152-187 (w104_reference_slopes_152_187)'),
    'P_d': 'the rate stays within a factor 2 of -265 EUR/cycle: mean(dQ_k, 188..287) = DQ / 100 in [-530, -132.5]; the '
           'four 25-cycle quarter means reported',
    'triggers': {
        'T1_dso_share_gt_half': 'the DSO blocks carry more than 50 % of the change: DSO share > 0.5',
        'T2_creep_decays': '|mean dQ over 263..287| < 0.25 x |mean dQ over 188..212|',
        'T3_boyd_lapse': ('any cycle k in 188..287 with NOT (boyd_all_pass AND local_solves_ok); cycles 88..187 are '
                          'W104\'s, replayed bitwise, and had none'),
    },
    'outcome': ('confirmed: P_a, P_b, P_c, P_d all hold and no trigger -> the expert adds a creep branch; refuted: any '
                'trigger -> Advisor review before the campaign is sized; mixed: anything else, reported as such; '
                'not_scoreable: the run did not reach 287 (aborted)'),
    'resolution_note': ('DQ, the shares and the means are statistics of ONE trajectory (exact per-cycle values), not '
                        'differences of two settled quantities: no stopping-slack bar applies. The terminal-step-to-'
                        'threshold ratios are reported (|dQ_287| / EPS0; production rule ten of cycle 287)'),
    'report_only_rule': ('SettlingRule(N = 87, cap = 287, P_MAX = 22) observes every cycle, non-latching; the first '
                         'cycle at which it would certify (if any), every would-certify cycle, and its status at 287 are '
                         'reported; it never ends the run'),
    'objective_convention': 'gross_operational_cost (settlement excluded); net and salvage reported beside',
}
PREDICTIONS = {
    'expert_H_ess_flat': {
        'P_a': {'statement': 'D[TSO, generation_cost] / DQ >= 0.5', 'threshold': 0.5},
        'P_b': {'statement': 'mean S_k(263..287) / mean S_k(188..212) >= 0.5 (ESSO side, all nodes)', 'threshold': 0.5},
        'P_c': {'statement': 'argmax_g OLS slope of boyd_g_primal_ratio over 188..287 == ess, slope > 0'},
        'P_d': {'statement': 'mean dQ over 188..287 in [-530, -132.5] EUR/cycle', 'bounds': list(P_D_BOUNDS)},
        'refutation_triggers': ['T1_dso_share_gt_half', 'T2_creep_decays', 'T3_boyd_lapse'],
        'outcome_mapping': DEFINITIONS['outcome'],
    },
    'pre_registration_against_P_c': ('W104 found the pf_primal ratio rising 0.013 -> 0.127 over cycles 152-187 '
                                     '(committed per_cycle_record of 82dcebb7)'),
}


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
    return W101L._committed_clean(rel)


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W105] guards {g} {extra_msg}')
    for _n, guard in GUARDS:
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


def _norm(text):
    return ' '.join(text.split())


# ======================================================================================================================
#  the configuration and the entry
# ======================================================================================================================
def w104_spec():
    return _load(W104_PINS['campaign_spec']['path'])


def w104_entry():
    return w104_spec()['candidates'][0]


def recert_entry():
    return next(e for e in _load(K.RECERT_SPEC_REL)['candidates'] if e['label'] == E.BASE_CELL)


def configuration():
    cfg = w104_spec()['configuration']
    return {'name': 'W105 C* SETTLING EXTENSION diagnostic -- ' + cfg['name'].split(' -- ', 1)[-1],
            'arm_label': cfg['arm_label'], 'overrides': dict(cfg['overrides']),
            'case_file_anderson_acceleration': dict(cfg['case_file_anderson_acceleration']),
            'ess_ageing_baseline': json.loads(json.dumps(cfg['ess_ageing_baseline'])),
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': dict(cfg['convergence_depth_tail']),
            'note': ('W104\'s configuration (spec 1a7483ef = the recert ddd6cd44) unchanged; the entry declares '
                     'settling_extension (keyed) instead of settling_continuation; cap 287; persist_certified_models '
                     'as W104; no option (b); concurrency 1')}


def entries():
    e = w104_entry()
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    pc = e['post_certification']
    opts = {'investment_year': e['canonical']['investment_year'],
            'post_certification': {'persist_certified_models': pc['persist_certified_models'],
                                   'hull_polish': pc['hull_polish'], 'reference': pc['reference']},
            'settling_extension': E.declaration()}
    return [(CELL, nodes, opts)]


def expected_keys():
    e = recert_entry()
    cfg = w104_spec()['configuration']
    kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
              convergence_depth_tail=cfg['convergence_depth_tail'])
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    ext = H.evaluation_key(e['key'], e['overrides'], settling_extension=E.declaration(), **kw)
    return {'base_key_without_extension': base, 'extension_key': ext}


def campaign_root_rel():
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_ID}')


def campaign_root():
    return _abs(campaign_root_rel())


def pre_launch_assertion(spec=None):
    k = expected_keys()
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    rec, w104 = recert_entry(), w104_entry()
    eval_dir_name = H.eval_dir_name(k['extension_key'], CELL)
    ids = H.eval_ids(CAMPAIGN_ID, k['extension_key'])
    work = L._work_dir()
    entry = ((spec or {}).get('candidates') or [None])[0]
    parts = {
        'base_key_equals_recert_key': k['base_key_without_extension'] == rec['eval_key'],
        'extension_key_differs_from_recert_and_w104': k['extension_key'] not in (rec['eval_key'], w104['eval_key']),
        'extension_key_absent_from_committed_specs_outside_w105_root': k['extension_key'] not in committed,
        'campaign_root_differs_from_w104': os.path.abspath(campaign_root()) != os.path.abspath(_abs(E.W104_ROOT)),
        'eval_dir_name_differs_from_w104_and_recert': eval_dir_name not in (w104['eval_dir'], rec['eval_dir']),
        'working_dir_ids_differ_from_w104_and_recert': not (set(ids.values()) & (set(w104['working_dir_ids'].values())
                                                                                  | set(rec['working_dir_ids'].values()))),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
        'candidate_key_equals_w104_and_recert': w104['key'] == rec['key'],
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['extension_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(spec, ss_pin):
    ws = w104_spec()
    we, wc = w104_entry(), ws['configuration']
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    same_cfg = ('arm_label', 'case_file', 'case_file_sha256', 'overrides', 'apply_rho', 'full_diagnostics_in_rows',
                'case_file_anderson_acceleration', 'ess_ageing_baseline', 'ess_ageing_baseline_label',
                'convergence_depth_tail', 'ess_params_file')
    checks = {f'configuration:{k}': (cfg.get(k) == wc.get(k) if k != 'ess_params_file' else
                                     (cfg.get(k) or {}).get('sha256') == (wc.get(k) or {}).get('sha256'))
              for k in same_cfg}
    same_entry = ('canonical', 'key', 'overrides', 'effective_anderson_acceleration', 'post_certification')
    checks.update({f'entry:{k}': e.get(k) == we.get(k) for k in same_entry})
    checks.update({
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == CELL,
        'entry_extension_is_the_declaration': e.get('settling_extension') == E.declaration(),
        'entry_has_no_continuation': 'settling_continuation' not in e and 'certification_continuation' not in e,
        'entry_has_no_release_solution_bookkeeping_as_w104': ('release_solution_bookkeeping' not in e
                                                               and 'release_solution_bookkeeping' not in we),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'cap_287': spec.get('cap') == E.CAP == 287,
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_as_w104': spec.get('required_consecutive_cycles') == ws['required_consecutive_cycles']
        == 10,
        'bar_window_as_w104': spec.get('bar_window_cycles') == ws['bar_window_cycles'],
        'thread_caps_as_w104': spec.get('thread_caps') == ws['thread_caps'],
        'interpreter_as_w104': spec.get('interpreter') == ws['interpreter'],
        'solver_path_as_w104': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH')
        == (ws.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'stage_spec_pinned': (spec.get('extra') or {}).get('stage_spec') == ss_pin,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
    })
    return checks


# ======================================================================================================================
#  provenance
# ======================================================================================================================
def code_since_w104():
    """Production byte-identical between W104's launch commit and HEAD; no uncommitted change to any tracked .py."""
    diff = H._git(['diff', '--name-status', W104_LAUNCH_GIT, 'HEAD', '--', '*.py']).splitlines()
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    changed = [{'status': line.split('\t', 1)[0], 'path': line.split('\t')[-1]} for line in diff]
    prod_changed = [c for c in changed if c['path'] in PRODUCTION_UNCHANGED_SINCE_W104]
    return {'w104_launch_commit': W104_LAUNCH_GIT, 'head': H._git(['rev-parse', 'HEAD']), 'changed_py': changed,
            'production_changed': prod_changed, 'uncommitted_tracked_py': dirty,
            'note': 'the harness changed (W105 keyed option settling_extension); production unchanged',
            'ok': not prod_changed and not dirty}


def _w104_pins_ok():
    failures = []
    for name, pin in list(W104_PINS.items()) + [('spec v39', SPEC_V39)] + [(f'superseded {k}', v)
                                                                             for k, v in SUPERSEDED.items()]:
        if _sha(pin['path']) != pin['sha256'] or not _committed_clean(pin['path']):
            failures.append(f'{name} not as committed: {pin["path"]}')
    manifest = _load(W104_PINS['campaign_manifest']['path'])
    for key in ('per_cycle_record', 'cycle_record', 'decision', 'recourse_blocks_all', 'g_s39_D'):
        pin = E.W104[key]
        if manifest.get(pin['path']) != pin['sha256'] or _sha(pin['path']) != pin['sha256'] \
                or not _committed_clean(pin['path']):
            failures.append(f'W104 {key} not as committed / manifest: {pin["path"]}')
    return failures


def _common_checks():
    failures = _w104_pins_ok()
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a continuation / extension launcher is alive: {others}')
    return failures


def verbatim_check():
    text = _norm(open(_abs(BRIEF), encoding='utf-8').read())
    found = {k: _norm(v) in text for k, v in VERBATIM.items()}
    return {'brief': BRIEF, 'brief_sha256_at_freeze': _sha(BRIEF), 'brief_git_state': L._git_state(BRIEF),
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def ols_slope(pairs):
    """Ordinary least-squares slope of y on x over [(x, y)]; None with fewer than two distinct x."""
    n = len(pairs)
    if n < 2:
        return None
    mx = sum(x for x, _y in pairs) / n
    my = sum(y for _x, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _y in pairs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / sxx


def w104_reference_slopes():
    rows = {r['cycle']: r for r in _read_jsonl(_abs(E.W104['per_cycle_record']['path']))}
    out = {}
    for g in E.CHANNELS:
        for kind in ('primal', 'dual'):
            f = f'boyd_{g}_{kind}_ratio'
            out[f] = {'slope_152_187': ols_slope([(k, rows[k][f]) for k in range(152, 188)]),
                      'at_152': rows[152][f], 'at_187': rows[187][f],
                      'slope_143_187': ols_slope([(k, rows[k][f]) for k in range(143, 188)])}
    q = {k: rows[k]['gross_operational_cost'] for k in rows}
    out['mean_dQ_143_187'] = (q[187] - q[142]) / 45.0
    out['mean_dQ_163_187'] = (q[187] - q[162]) / 25.0
    return out


def wall_time_estimate(overhead_s_per_cycle):
    lines = _read_jsonl(_abs(E.W104['cycle_record']['path']))
    walls = [x['t_end_s'] - x['t_start_s'] for x in lines]
    rec = _load(os.path.join(E.W104_EVAL_DIR, 'evaluation_record.json'))
    child = rec['wall_time_s']['child_process_s']
    campaign = _load(W104_PINS['campaign_results']['path'])['wall_clock_s']
    held = walls[87:]
    per_held = sum(held) / len(held)
    per_held_max = max(held)
    overhead = child - sum(walls)
    launcher = campaign - child
    est = sum(walls) + 100 * per_held + overhead + launcher + E.CAP * overhead_s_per_cycle
    return {'basis': ('W104\'s own per-cycle walls (t_end_s - t_start_s of its cycle lines, concurrency 1), its child '
                      'wall and its campaign wall: replay 1..187 = W104\'s sum; 188..287 = 100 x W104\'s mean over '
                      '88..187; + W104\'s init / terminal / persistence overhead and launcher overhead; + the measured '
                      'new capture overhead x 287'),
            'w104_cycles': len(walls), 'w104_sum_cycle_walls_s': sum(walls), 'w104_mean_cycle_s_1_187': sum(walls) / len(walls),
            'w104_mean_cycle_s_88_187': per_held, 'w104_max_cycle_s_88_187': per_held_max,
            'w104_child_overhead_s': overhead, 'w104_launcher_overhead_s': launcher,
            'new_capture_overhead_s_per_cycle_measured': overhead_s_per_cycle,
            'expected_wall_s': est, 'expected_wall_h': est / 3600.0,
            'task_statement': 'about 287 x 25-28 s = 2.0-2.3 h plus the capture overhead'}


# ======================================================================================================================
#  the scorer (frozen definitions; pure)
# ======================================================================================================================
def _mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def _ratio(a, b):
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return a / b


def score_extension(rows, creep, run_reached_cap=True):
    """The recorded predictions scored with the frozen definitions (DEFINITIONS). `rows` = {cycle: per_cycle_record
    row}, `creep` = {cycle: creep line}. Pure function."""
    lo, hi = WINDOW
    out = {'definitions_version': SPEC_VERSION, 'window': list(WINDOW)}
    if not run_reached_cap or any(k not in rows for k in range(lo - 1, hi + 1)):
        out.update({'outcome': 'not_scoreable', 'reason': 'the run did not reach 287 (or rows are missing)'})
        return out
    q = {k: rows[k]['gross_operational_cost'] for k in range(lo - 1, hi + 1)}
    if any(v is None for v in q.values()):
        out.update({'outcome': 'not_scoreable', 'reason': 'a failed cycle in 187..287 has no Q'})
        return out
    dq = {k: q[k] - q[k - 1] for k in range(lo, hi + 1)}
    total = q[hi] - q[lo - 1]
    quarter_means = {f'{a}_{b}': _mean(dq[k] for k in range(a, b + 1)) for a, b in QUARTERS}
    mean_all = _mean(dq[k] for k in range(lo, hi + 1))
    out.update({'Q_187': q[lo - 1], 'Q_287': q[hi], 'DQ_total': total, 'mean_dQ_188_287': mean_all,
                'mean_dQ_quarters': quarter_means})
    # ---- P_a / T1 ------------------------------------------------------------------------------------------------
    a0 = ((creep.get(lo - 1) or {}).get('q_decomposition') or {}).get('agents')
    a1 = ((creep.get(hi) or {}).get('q_decomposition') or {}).get('agents')
    shares, pa, t1 = None, None, None
    if a0 and a1 and total != 0:
        comps = E.Q_COMPONENTS_ALL + ('ess_terms', 'value')
        D = {a: {c: a1[a][c] - a0[a][c] for c in comps} for a in a1}
        shares = {a: {c: D[a][c] / total for c in comps} for a in D}
        dso = sum(D[a]['value'] for a in D if a.startswith('DSO_'))
        tso_gen = D['TSO']['generation_cost']
        recon = total - sum(D[a]['value'] for a in D)
        pa = {'share_TSO_generation': tso_gen / total, 'D_TSO_generation': tso_gen, 'threshold': 0.5,
              'holds': (tso_gen / total) >= 0.5}
        t1 = {'dso_share': dso / total, 'D_DSO': dso, 'fires': (dso / total) > 0.5}
        out['decomposition'] = {'D': D, 'shares': shares, 'tso_share': D['TSO']['value'] / total,
                                'dso_share': dso / total, 'esso_share': D['ESSO']['value'] / total,
                                'reconciliation_DQ_minus_sum_agents': recon}
    out['P_a'] = pa or {'holds': None, 'reason': 'decomposition unavailable at 187 / 287 or DQ == 0'}
    # ---- P_b ------------------------------------------------------------------------------------------------------
    def fam_series(node, fam):
        vals = {}
        for k in range(lo, hi + 1):
            mv = (creep.get(k) or {}).get('ess_movement') or {}
            if not mv.get('available'):
                return None
            v = ((mv.get('all_nodes') or {}) if node is None else ((mv.get('per_node') or {}).get(node) or {})).get(fam)
            if v is None:
                return None
            vals[k] = v
        return vals
    pb_detail = {}
    families = [f'{kind}_{side}' for kind in ('p', 'q') for side in E.ESS_SIDES] + \
               [f'{kind}_{side}' for kind in ('charge', 'discharge') for side in E.ESS_CHARGE_SIDES]
    nodes = sorted(((creep.get(hi) or {}).get('ess_movement') or {}).get('per_node') or {})
    for node in [None] + nodes:
        for fam in families:
            s = fam_series(node, fam)
            if s is None:
                continue
            m0 = _mean(s[k] for k in range(FIRST_Q[0], FIRST_Q[1] + 1))
            m1 = _mean(s[k] for k in range(LAST_Q[0], LAST_Q[1] + 1))
            pb_detail[f'{node or "all"}:{fam}'] = {'mean_188_212': m0, 'mean_263_287': m1, 'ratio': _ratio(m1, m0)}
    prim = pb_detail.get('all:p_esso')
    out['P_b'] = ({'primary': 'all:p_esso', **prim, 'threshold': 0.5,
                   'holds': (prim['ratio'] is not None and prim['ratio'] >= 0.5)} if prim
                  else {'holds': None, 'reason': 'ESS movement unavailable'})
    out['P_b_detail'] = pb_detail
    # ---- P_c ------------------------------------------------------------------------------------------------------
    slopes = {}
    for g in E.CHANNELS:
        for kind in ('primal', 'dual'):
            f = f'boyd_{g}_{kind}_ratio'
            slopes[f] = ols_slope([(k, rows[k][f]) for k in range(lo, hi + 1)])
    prim_sl = {g: slopes[f'boyd_{g}_primal_ratio'] for g in E.CHANNELS}
    arg = max(prim_sl, key=lambda g: prim_sl[g])
    out['P_c'] = {'slopes': slopes, 'argmax_primal': arg, 'argmax_slope': prim_sl[arg],
                  'holds': arg == 'ess' and prim_sl['ess'] > 0,
                  'primal_ratio_at_188_and_287': {g: [rows[lo][f'boyd_{g}_primal_ratio'],
                                                      rows[hi][f'boyd_{g}_primal_ratio']] for g in E.CHANNELS}}
    # ---- P_d / T2 / T3 --------------------------------------------------------------------------------------------
    out['P_d'] = {'mean_dQ_188_287': mean_all, 'bounds': list(P_D_BOUNDS),
                  'holds': P_D_BOUNDS[0] <= mean_all <= P_D_BOUNDS[1], 'quarter_means': quarter_means}
    m_first, m_last = quarter_means[f'{FIRST_Q[0]}_{FIRST_Q[1]}'], quarter_means[f'{LAST_Q[0]}_{LAST_Q[1]}']
    t2 = {'abs_mean_263_287': abs(m_last), 'abs_mean_188_212': abs(m_first),
          'fires': abs(m_last) < 0.25 * abs(m_first)}
    lapses = [k for k in range(lo, hi + 1) if not (rows[k]['boyd_all_pass'] and rows[k]['local_solves_ok'])]
    t3 = {'lapse_cycles': lapses, 'fires': bool(lapses),
          'max_ratio_188_287': {f'boyd_{g}_{kind}_ratio': max(rows[k][f'boyd_{g}_{kind}_ratio'] for k in range(lo, hi + 1))
                                for g in E.CHANNELS for kind in ('primal', 'dual')}}
    out['triggers'] = {'T1_dso_share_gt_half': t1 or {'fires': None, 'reason': 'decomposition unavailable'},
                       'T2_creep_decays': t2, 'T3_boyd_lapse': t3}
    fired = [n for n, t in out['triggers'].items() if t.get('fires') is True]
    held = {p: out[p].get('holds') for p in ('P_a', 'P_b', 'P_c', 'P_d')}
    if fired:
        outcome = 'refuted'
    elif all(v is True for v in held.values()):
        outcome = 'confirmed'
    else:
        outcome = 'mixed'
    out.update({'predictions_held': held, 'triggers_fired': fired, 'outcome': outcome})
    last_row = rows[hi]
    out['terminal_step_to_threshold'] = {
        'abs_dQ_287_over_EPS0': abs(dq[hi]) / SC.EPS0,
        'rule_ten_cycle_287': ((last_row['objective_change_abs'] / last_row['objective_tolerance'])
                               if last_row.get('objective_change_abs') is not None and last_row.get('objective_tolerance')
                               else None)}
    return out


def scorer_self_tests():
    """The scorer on synthetic trajectories with known verdicts: confirmed / refuted by each trigger / mixed."""
    def make(rate_fn, tso_gen_frac, dso_frac, ess_move_fn, slope_ess, slope_pf, lapse_at=None):
        rows, creep = {}, {}
        q = 6.5e8
        agents_base = {a: {c: 0.0 for c in E.Q_COMPONENTS_ALL + ('ess_terms', 'value')}
                       for a in ('TSO', 'DSO_5', 'DSO_7', 'DSO_9', 'ESSO')}
        cum = 0.0
        for k in range(180, 288):
            if k >= 188:
                step = rate_fn(k)
                q += step
                cum += step
            rows[k] = {'cycle': k, 'gross_operational_cost': q, 'boyd_all_pass': k != lapse_at,
                       'local_solves_ok': True, 'objective_change_abs': 1.0, 'objective_tolerance': 65000.0,
                       'boyd_v_primal_ratio': 0.1, 'boyd_v_dual_ratio': 0.2,
                       'boyd_pf_primal_ratio': 0.1 + slope_pf * k, 'boyd_pf_dual_ratio': 0.2,
                       'boyd_ess_primal_ratio': 0.1 + slope_ess * k, 'boyd_ess_dual_ratio': 0.3}
            ag = json.loads(json.dumps(agents_base))
            ag['TSO']['generation_cost'] = tso_gen_frac * cum
            ag['TSO']['value'] = (1.0 - dso_frac) * cum
            ag['TSO']['slack_penalties'] = (1.0 - dso_frac - tso_gen_frac) * cum
            for n in (5, 7, 9):
                ag[f'DSO_{n}']['flexibility_cost'] = dso_frac * cum / 3.0
                ag[f'DSO_{n}']['value'] = dso_frac * cum / 3.0
            s = ess_move_fn(k)
            creep[k] = {'cycle': k, 'q_decomposition': {'agents': ag},
                        'ess_movement': {'available': True,
                                         'all_nodes': {f: s for f in ['p_esso', 'p_tso', 'p_dso', 'p_z']},
                                         'per_node': {str(n): {f: s / 3.0 for f in ['p_esso', 'p_tso', 'p_dso', 'p_z']}
                                                      for n in (5, 7, 9)}}}
        return rows, creep
    cases = {
        'confirmed': (make(lambda k: -265.0, 0.8, 0.1, lambda k: 5.0, 1e-4, 1e-5), 'confirmed'),
        'refuted_T1_dso': (make(lambda k: -265.0, 0.3, 0.6, lambda k: 5.0, 1e-4, 1e-5), 'refuted'),
        'refuted_T2_decay': (make(lambda k: -600.0 * 0.95 ** (k - 188), 0.8, 0.1, lambda k: 5.0, 1e-4, 1e-5),
                             'refuted'),
        'refuted_T3_lapse': (make(lambda k: -265.0, 0.8, 0.1, lambda k: 5.0, 1e-4, 1e-5, lapse_at=240), 'refuted'),
        'mixed_pf_rises': (make(lambda k: -265.0, 0.8, 0.1, lambda k: 5.0, 1e-5, 1e-4), 'mixed'),
        'mixed_ess_stops': (make(lambda k: -265.0, 0.8, 0.1, lambda k: 5.0 * 0.95 ** (k - 188), 1e-4, 1e-5), 'mixed'),
        'mixed_rate_high': (make(lambda k: -700.0, 0.8, 0.1, lambda k: 5.0, 1e-4, 1e-5), 'mixed'),
    }
    out = {}
    for name, ((rows, creep), want) in cases.items():
        s = score_extension(rows, creep)
        out[name] = {'ok': s['outcome'] == want, 'want': want, 'got': s['outcome'], 'held': s['predictions_held'],
                     'fired': s['triggers_fired']}
    s_ab = score_extension({k: v for k, v in cases['confirmed'][0][0].items() if k < 250}, {}, run_reached_cap=False)
    out['aborted_not_scoreable'] = {'ok': s_ab['outcome'] == 'not_scoreable', 'got': s_ab['outcome']}
    return out, all(v['ok'] for v in out.values())


# ======================================================================================================================
#  post-run gates
# ======================================================================================================================
LINE_FIELDS_REQUIRED = ('cycle', 'phase', 'regime', 'gross', 'gross_hex', 'net_operational_recourse',
                        'terminal_salvage_value', 'boyd_k', 'local_solves_ok', 'boyd_ratios', 'boyd_pf_primal_ratio',
                        'settling', 'report_only_would_certify', 'holds', 'replay_equal',
                        'certificate_length_in_force_at_cycle_end', 't_start_s', 't_end_s')
CREEP_FIELDS_REQUIRED = ('cycle', 'q_decomposition', 'ess_movement', 'boyd', 'capture_s')


def hold_checks(eval_dir, rec):
    n = E.N_HOLD
    lines = _read_jsonl(os.path.join(eval_dir, E.CYCLE_FILE))
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    pre = [by[c] for c in cycles if c <= n]
    held = [by[c] for c in cycles if c > n]
    rho_n = ((by.get(n) or {}).get('rho') or {}).get('rho_after')
    parts = {
        'one_line_per_cycle_1_287': cycles == list(range(1, E.CAP + 1)) and (rec.get('cycles_run') == E.CAP),
        'no_hold_acted_through_87': all((x.get('holds') or {}) == {'aa': False, 'tail_apply': False, 'tail_next': False,
                                                                   'rho': False}
                                        or ((x.get('holds') or {}).get('aa') is None and x.get('gross') is None)
                                        for x in pre),
        'aa_held_off_after_87': all((x.get('aa') is None and x.get('gross') is None)
                                    or ((x.get('aa') or {}).get('hold') is True
                                        and (x.get('aa') or {}).get('action') == E.AA_OFF_ACTION) for x in held),
        'tail_held_on_after_87': all((x.get('tail_apply') or {}).get('active_passed') is True
                                     and (x.get('tail_next') or {}).get('returned') is True for x in held),
        'rho_frozen_after_87_at_cycle_87_values': bool(rho_n) and all(
            (x.get('rho') or {}).get('hold') is True and (x.get('rho') or {}).get('rho_after') == rho_n
            and not (x.get('rho') or {}).get('changed_channels') for x in held),
        'certificate_length_disabled_every_cycle': all(
            x.get('certificate_length_in_force_at_cycle_end') == E.CERTIFICATION_DISABLED_THRESHOLD for x in lines),
        'replay_equal_every_cycle_1_187': (all(by[c].get('replay_equal') is True for c in range(1, E.REPLAY_THROUGH + 1)
                                               if c in by) and all(c in by for c in range(1, E.REPLAY_THROUGH + 1))),
        'summary_ok': (rec.get('settling_extension_summary') or {}).get('ok') is True,
    }
    return all(parts.values()), {'parts': parts, 'rho_at_87': rho_n}


def fixed_length_check(eval_dir, rec):
    summ = rec.get('settling_extension_summary') or {}
    dec_path = os.path.join(eval_dir, E.DECISION_FILE)
    dec = json.load(open(dec_path)) if os.path.isfile(dec_path) else None
    ok = (rec.get('cycles_run') == E.CAP and summ.get('stopped_by') == 'cap' and dec is not None
          and dec.get('report_only') is True and dec.get('w104_mirror_decision_equals_w104') is True)
    return bool(ok), {'cycles_run': rec.get('cycles_run'), 'stopped_by': summ.get('stopped_by'),
                      'decision_file_present': dec is not None,
                      'first_would_certify': ((dec or {}).get('first_would_certify') or {}).get('k_star')}


def settling_replay_check(eval_dir):
    """The pure rules replayed on the run's per_cycle_record reproduce the in-cycle records: the report-only rule
    (every cycle, non-latching) and the W104 mirror (1..187); the decision file's first_would_certify and would-certify
    cycles likewise."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = _read_jsonl(os.path.join(eval_dir, E.CYCLE_FILE))
    rule = E.ReportOnlyRule()
    mirror = SC.SettlingRule(E.N_HOLD, E.REPLAY_THROUGH, E.P_MAX)
    pure, pure_m = [], []
    for r in rows:
        b = bool(r['boyd_all_pass'] and r['local_solves_ok'])
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'], b))
        if r['cycle'] <= E.REPLAY_THROUGH:
            pure_m.append(mirror.observe(r['cycle'], r['gross_operational_cost'], b))
    in_cycle = [x.get('settling') for x in lines]
    in_mirror = [x.get('settling_w104_mirror') for x in lines if x['cycle'] <= E.REPLAY_THROUGH]
    dec_path = os.path.join(eval_dir, E.DECISION_FILE)
    dec = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}
    parts = {'report_only_records_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'mirror_records_reproduced': [K._jt(a) for a in in_mirror] == [K._jt(b) for b in pure_m],
             'first_would_certify_reproduced': K._jt(dec.get('first_would_certify')) == K._jt(rule.first),
             'would_certify_cycles_reproduced': dec.get('would_certify_cycles') == [c['cycle'] for c in
                                                                                    rule.certifications],
             'decision_file_present': bool(dec)}
    return all(parts.values()), {'parts': parts}


def replay_gate_full(eval_dir):
    """Rows 1..187 of the run's per_cycle_record.jsonl against W104's committed rows: EVERY field, as JSON text."""
    ref = {r['cycle']: r for r in _read_jsonl(_abs(E.W104['per_cycle_record']['path']))}
    path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    run = {r['cycle']: r for r in _read_jsonl(path)} if os.path.isfile(path) else {}
    first, detail = None, None
    for c in range(1, E.REPLAY_THROUGH + 1):
        a, b = run.get(c), ref.get(c)
        if a is None:
            first, detail = c, {'missing_in_run': True}
            break
        diff = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k),
                                                                                                        sort_keys=True))
        if diff:
            ga, gb = a.get('gross_operational_cost'), b.get('gross_operational_cost')
            first, detail = c, {'fields_differing': diff, 'gross_difference_run_minus_recorded':
                                (ga - gb) if (ga is not None and gb is not None) else None}
            break
    return {'bitwise_through_187': first is None, 'first_divergence_cycle': first, 'divergence': detail}


def w104_lines_and_blocks_check(eval_dir):
    """Post-run re-check of cycles 1..187: the cycle-line gated fields and the mirror's settling record against W104's
    cycle lines, and the W101 all-block lines (except capture_s) against W104's recourse_blocks_all."""
    ref_lines = {r['cycle']: r for r in _read_jsonl(_abs(E.W104['cycle_record']['path']))}
    ref_blocks = {r['cycle']: r for r in _read_jsonl(_abs(E.W104['recourse_blocks_all']['path']))}
    lines = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, E.CYCLE_FILE))}
    blines = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, E.BLOCKS_FILE))}
    bad_l = [c for c in range(1, E.REPLAY_THROUGH + 1) if c not in lines or E.line_compare(lines[c], ref_lines[c])]
    strip = (lambda d: {k: v for k, v in d.items() if k != 'capture_s'})
    bad_b = [c for c in range(1, E.REPLAY_THROUGH + 1)
             if c not in blines or K._jt(strip(blines[c])) != K._jt(strip(ref_blocks[c]))]
    return {'cycle_lines_equal': not bad_l, 'blocks_lines_equal': not bad_b, 'cycle_lines_differing_first10': bad_l[:10],
            'blocks_lines_differing_first10': bad_b[:10]}


def creep_capture_check(eval_dir, rec, rows):
    creep = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, E.CREEP_FILE))}
    ess_path = os.path.join(eval_dir, E.ESS_SCHEDULE_FILE)
    ess = _read_jsonl(ess_path) if os.path.isfile(ess_path) else []
    ok_cycles = [r['cycle'] for r in rows if r.get('gross_operational_cost') is not None]
    g = json.load(open(os.path.join(eval_dir, f"g_{configuration()['arm_label']}.json")))
    g_rows = {r['cycle']: r for r in g['cycle_trajectory']}
    boyd_bad = []
    for c, x in creep.items():
        gr = g_rows.get(c)
        b = x.get('boyd') or {}
        for ch in E.CHANNELS:
            for f in ('r', 's', 'eps_pri', 'eps_dual', 'primal_ratio', 'dual_ratio'):
                if gr is None or (b.get(ch) or {}).get(f) != gr.get(f'boyd_{ch}_{f}'):
                    boyd_bad.append((c, ch, f))
    parts = {
        'one_creep_line_per_cycle': sorted(creep) == list(range(1, E.CAP + 1)),
        'fields_every_line': all(all(f in x for f in CREEP_FIELDS_REQUIRED) for x in creep.values()),
        'q_decomposition_every_successful_cycle_reconciles': all(
            ((creep.get(c) or {}).get('q_decomposition') or {}).get('reconciliation', {}).get('reconciles') is True
            for c in ok_cycles),
        'ess_movement_available_from_cycle_2': all(((creep.get(c) or {}).get('ess_movement') or {}).get('available')
                                                   is True for c in range(2, E.CAP + 1)),
        'boyd_captured_every_cycle_equals_production_rows': not boyd_bad,
        'ess_schedule_header_plus_287_lines': (len(ess) == E.CAP + 1 and ess[0].get('header') is True
                                               and [x['cycle'] for x in ess[1:]] == list(range(1, E.CAP + 1))),
        'no_capture_errors': (rec.get('settling_extension_summary') or {}).get('n_capture_errors') == 0,
    }
    return all(parts.values()), {'parts': parts, 'boyd_mismatch_first10': boyd_bad[:10],
                                 'bytes': {'creep': os.path.getsize(os.path.join(eval_dir, E.CREEP_FILE)),
                                           'ess_schedule': os.path.getsize(ess_path) if os.path.isfile(ess_path) else None}}


def line_fields_check(eval_dir):
    lines = _read_jsonl(os.path.join(eval_dir, E.CYCLE_FILE))
    missing = {}
    for x in lines:
        m = [f for f in LINE_FIELDS_REQUIRED if f not in x]
        if x.get('gross') is not None:
            m += [f for f in ('blocks_captured',) if f not in x]
        if x['cycle'] <= E.REPLAY_THROUGH and 'settling_w104_mirror' not in x:
            m.append('settling_w104_mirror')
        m += [f'boyd_ratios.{f}' for f in E.BOYD_RATIO_FIELDS if f not in (x.get('boyd_ratios') or {})]
        if m:
            missing[str(x['cycle'])] = m
    return not missing and bool(lines), {'n_lines': len(lines), 'missing_by_cycle': missing}


def cell_gates(entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('barrier') and rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_extension_summary'] = rec.get('settling_extension_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys()['extension_key']
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
    gates['G8_persistence_as_w104'], detail['G8'] = W101L.persistence_check(rec, eval_dir)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_87_held_after'], detail['G13'] = hold_checks(eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_fixed_length_287'], detail['G15'] = fixed_length_check(eval_dir, rec)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_replays_reproduce_in_cycle'], detail['G17'] = settling_replay_check(eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    rg = replay_gate_full(eval_dir)
    gates['G19_replay_bitwise_1_187_every_field'] = rg['bitwise_through_187']
    detail['G19'] = rg
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = creep_capture_check(eval_dir, rec, rows)
    lb = w104_lines_and_blocks_check(eval_dir)
    gates['G22_w104_cycle_lines_and_block_lines_1_187'] = lb['cycle_lines_equal'] and lb['blocks_lines_equal']
    detail['G22'] = lb
    return gates, detail, rec


def cell_report(eval_dir, rec):
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    creep = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, E.CREEP_FILE))}
    dec = json.load(open(os.path.join(eval_dir, E.DECISION_FILE)))
    rep = score_extension(rows, creep, run_reached_cap=rec.get('cycles_run') == E.CAP)
    rep.update({'cell': CELL, 'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
                'candidate_canonical': rec.get('candidate_canonical'), 'cycles_run': rec.get('cycles_run'),
                'report_only_settling': {'first_would_certify': dec.get('first_would_certify'),
                                         'would_certify_cycles': dec.get('would_certify_cycles'),
                                         'status_at_cap': dec.get('status_at_cap'),
                                         'lapse_events': dec.get('lapse_events')},
                'net_operational_recourse_287': (rows.get(E.CAP) or {}).get('recourse'),
                'terminal_salvage_value_287': (rows.get(E.CAP) or {}).get('terminal_salvage_value'),
                'objective_convention': DEFINITIONS['objective_convention']})
    return rep


def _synthetic_run_dir(tmp, variant):
    """A synthetic eval dir produced by the REAL wrappers (the checks' recorded drive), plus a per_cycle_record rebuilt
    from W104's rows (1..187) and the drive's values (188..287), a g_s39_D.json with the recorded / scripted Boyd rows,
    and the record summary; `variant` tampers one artifact."""
    fx = K.w104_fixtures()
    d = K.drive(fx, 'bitwise')
    files = K._sink_files(d['sink'])
    st = d['state']
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == E.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    lines = {x['cycle']: x for x in files[E.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[E.CREEP_FILE]}
    rows, g_rows = [], []
    for c in range(1, E.CAP + 1):
        if c <= E.REPLAY_THROUGH:
            row = dict(fx['rows'][c])
            g = dict(fx['g'][c])
        else:
            x = lines[c]
            row = {'cycle': c, 'local_solves_ok': True, 'recourse': x['gross'], 'gross_operational_cost': x['gross'],
                   'terminal_salvage_value': 0.0, 'objective_change_abs': abs(x['step']), 'objective_tolerance': 65000.0,
                   'objective_change_ratio': abs(x['step']) / 65000.0, 'cycle_convergence': x['boyd_k'],
                   'consecutive_converged_cycles': x['consecutive_converged_cycles_tracked'],
                   'boyd_all_pass': x['boyd_k'], 'boyd_stop': x['boyd_k'],
                   **{f: x['boyd_ratios'][f] for f in E.BOYD_RATIO_FIELDS},
                   **{f'boyd_{g_}_channel_pass': creep[c]['boyd'][g_]['channel_pass'] for g_ in E.CHANNELS},
                   **{f'rho_{g_}_after': x['rho']['rho_after'][g_] for g_ in E.CHANNELS},
                   **{f'rho_{g_}_action': x['rho']['actions'][g_] for g_ in E.CHANNELS},
                   'rho_freeze_active': True, 'efc_per_day_max': 1.0}
            g = {'cycle': c}
            for g_ in E.CHANNELS:
                for f, v in creep[c]['boyd'][g_].items():
                    g[f'boyd_{g_}_{f}'] = v
        rows.append(row)
        g_rows.append(g)
    if variant == 'tamper_row_150':
        rows[149] = dict(rows[149], objective_change_abs=(rows[149]['objective_change_abs'] or 0.0) + 1.0)
    if variant == 'tamper_boyd_capture_250':
        c250 = creep[250]
        c250['boyd']['ess']['r'] = c250['boyd']['ess']['r'] * 2.0
        with open(os.path.join(tmp, E.CREEP_FILE), 'w') as handle:
            for c in range(1, E.CAP + 1):
                handle.write(GRIO.dumps(creep[c], default=GRIO.json_default) + '\n')
    with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
        for r in rows:
            handle.write(GRIO.dumps(r, default=GRIO.json_default) + '\n')
    arm = configuration()['arm_label']
    with open(os.path.join(tmp, f'g_{arm}.json'), 'w') as handle:
        json.dump({'cycle_trajectory': g_rows}, handle)
    rec = {'cycles_run': E.CAP, 'settling_extension_summary': st.summary(), 'status': 'certified'}
    return rec, rows


def post_run_evaluator_self_tests():
    out = {}
    for name, variant, expect in (('pass', None, True), ('tampered_per_cycle_row_150', 'tamper_row_150', False),
                                  ('tampered_boyd_capture_250', 'tamper_boyd_capture_250', False)):
        tmp = tempfile.mkdtemp(prefix='w105_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, variant)
            h_ok, _h = hold_checks(tmp, rec)
            f_ok, _f = fixed_length_check(tmp, rec)
            s_ok, _s = settling_replay_check(tmp)
            l_ok, _l = line_fields_check(tmp)
            rg = replay_gate_full(tmp)
            c_ok, c_d = creep_capture_check(tmp, rec, rows)
            lb = w104_lines_and_blocks_check(tmp)
            allg = {'holds': h_ok, 'fixed_length': f_ok, 'rule_replays': s_ok, 'line_fields': l_ok,
                    'replay_full_1_187': rg['bitwise_through_187'], 'creep_capture': c_ok,
                    'w104_lines_blocks': lb['cycle_lines_equal'] and lb['blocks_lines_equal']}
            if expect:
                ok = all(allg.values())
            elif variant == 'tamper_row_150':
                ok = (not rg['bitwise_through_187']) and rg['first_divergence_cycle'] == 150 and all(
                    v for k_, v in allg.items() if k_ != 'replay_full_1_187')
            else:
                ok = (not c_ok) and c_d['boyd_mismatch_first10'][:1] == [(250, 'ess', 'r')] and all(
                    v for k_, v in allg.items() if k_ != 'creep_capture')
            rep = None
            if expect:
                rep = score_extension({r['cycle']: r for r in rows},
                                      {x['cycle']: x for x in _read_jsonl(os.path.join(tmp, E.CREEP_FILE))})
            out[name] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                         'score_on_synthetic': ({k: rep.get(k) for k in ('outcome', 'predictions_held', 'triggers_fired',
                                                                         'mean_dQ_188_287', 'DQ_total')}
                                                if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    # the synthetic drive: -265 EUR/cycle + a lapse at 200, 100 % TSO generation -> P_a holds, T3 fires -> refuted
    syn = (out.get('pass') or {}).get('score_on_synthetic') or {}
    out['score_on_synthetic_drive_is_refuted_by_T3'] = {
        'ok': syn.get('outcome') == 'refuted' and syn.get('triggers_fired') == ['T3_boyd_lapse']
        and (syn.get('predictions_held') or {}).get('P_a') is True, 'got': syn}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  the stage spec v41 (v40 superseded before any run)
# ======================================================================================================================
def launch_command(spec_sha=None):
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run '
            f'--spec-sha256 {spec_sha or "<campaign spec sha256>"} '
            f'> {os.path.join(ROOT_REL, "run_c_star_ext_v41_launch.log")} 2>&1')


def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST), 'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256'),
            'L4_overhead': (((doc.get('sections') or {}).get('L') or {}).get('result') or {}).get(
                'L4_shapes_sizes_overhead', {}).get('new_overhead_s_per_cycle_estimate')}


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks()


def _failing_check_items(checks):
    """W108 (logging only): {failing section: [failing items]} of a `_run_checks_inline` result. Items: a section
    error; `parts` entries not True; `tests` entries whose holds / ok is not True; top-level entries that are False or
    carry a holds / ok that is not True."""
    out = {}
    for sid, sec in ((checks or {}).get('sections') or {}).items():
        if (sec or {}).get('holds') is True:
            continue
        r = (sec or {}).get('result')
        items = []
        if not isinstance(r, dict):
            out[sid] = [f'result is {type(r).__name__}']
            continue
        if 'error' in r:
            items.append(f"error: {r['error']}")
        for k, v in (r.get('parts') or {}).items():
            if v is not True:
                items.append(f'parts.{k}')
        for k, v in (r.get('tests') or {}).items():
            if isinstance(v, dict) and v.get('holds', v.get('ok')) is not True:
                items.append(f'tests.{k}')
        for k, v in r.items():
            if k in ('holds', 'parts', 'tests', 'error'):
                continue
            if v is False or (isinstance(v, dict) and ('holds' in v or 'ok' in v)
                              and v.get('holds', v.get('ok')) is not True):
                items.append(k)
        out[sid] = items or ['holds is not True (no itemised field)']
    return out


def _checks_failure_message(checks):
    return f'the zero-solve checks do not all hold now -- failing sections and items: {_failing_check_items(checks)}'


def stage_spec_content(checks_inline, checks_file, post_tests, scorer_tests, verb, solver, pre, code_since, mem,
                       ref_slopes, wall):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    return {
        'schema': 'p515_s53_stage_spec_v41', 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 40, **SPEC_V40}, 'defect_note': DEFECT_NOTE,
        'authority': [f'{BRIEF} Addendum 54 Ruling 1 (committed 271e9325)', 'TASKS.md Addendum 54 order (W105)',
                      'Planner task W105'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED; net_operational_recourse and the '
                                 'terminal salvage reported beside it; every value in this spec and in the run\'s '
                                 'results is on this convention'),
        'code_sha256': code, 'production_sha256': production, 'code_since_w104_launch': code_since,
        'w104': {'pins': W104_PINS, 'records': E.W104, 'launch_git': W104_LAUNCH_GIT},
        'cell': {'label': CELL, 'base_cell': E.BASE_CELL, 'candidate_key': w104_entry()['key'],
                 'canonical': w104_entry()['canonical'], 'recert_eval_key': recert_entry()['eval_key'],
                 'w104_eval_key': w104_entry()['eval_key'], 'N_hold': E.N_HOLD, 'replay_through': E.REPLAY_THROUGH,
                 'extension_cycles': E.EXTENSION_CYCLES, 'cap': E.CAP, 'declaration': E.declaration(),
                 'campaign_id': CAMPAIGN_ID, 'campaign_root': campaign_root_rel(), 'keys': expected_keys(),
                 'pre_launch_assertion_at_freeze': pre, 'launch_command_template': launch_command()},
        'configuration': {'identical_to_w104': configuration(), 'concurrency': CONCURRENCY,
                          'option_b_release_solution_bookkeeping': 'absent, as in W104 (never keyed)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as W104)',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze and at --run'},
        'replay_gate': {
            'in_cycle': ('every cycle k <= 187, at the end of the cycle: the per_cycle_record fields '
                         + ', '.join(E.REPLAY_GATED_FIELDS) + '; the W104 cycle-line fields '
                         + ', '.join(E.CYCLE_LINE_GATED_FIELDS) + '; the mirror rule\'s settling record against W104\'s '
                         '\'settling\'; at 187 the mirror\'s decision against W104\'s settling_decision.json -- all as '
                         'JSON text against W104\'s committed records; the FIRST difference writes the cycle line with '
                         'the cycle and magnitude (fields, gross difference, largest relative difference) and ABORTS '
                         'the run (no relabelled continuation)'),
            'post_run': ('G19 every field of per_cycle_record rows 1..187 bitwise (adds '
                         + ', '.join(E.REPLAY_POST_RUN_ONLY_FIELDS) + '); G22 the cycle lines and the W101 all-block '
                         'lines 1..187 (except capture_s) against W104\'s'),
            'reference_hashes': {k: E.W104[k]['sha256'] for k in ('per_cycle_record', 'cycle_record', 'decision',
                                                                   'recourse_blocks_all', 'g_s39_D')},
            'expected': 'bitwise: production unchanged since W104\'s launch commit (code_since_w104_launch)'},
        'holds_after_87': {'AA': 'off (production\'s own off branch via all_boyd_pass forced True on a copy, every '
                                 'cycle > 87, even when Boyd lapses)',
                           'tail': 'on (applied True at the top of every cycle > 87; next-state True)',
                           'rho': 'frozen (allow_update False; before == after asserted)',
                           'boyd_lapse_behaviour': 'recorded (the rule\'s k0 resets); the run continues to 287',
                           'exactly_as_w104': 'the same wrapper code as W101\'s for these three holds (checks H1-H3)'},
        'fixed_length': {'cap': E.CAP, 'certificate_length': '10**9 for the whole run (restored at exit)',
                         'stops': 'only production\'s num_max_iters = 287 (or an abort)', 'early_stop': 'ABSENT'},
        'report_only_rule': {'module': 'settling_criterion', 'class': 'settling_criterion.SettlingRule',
                             'adapter': 'p515_s53_w105_settling_extension_hooks.ReportOnlyRule (non-latching)',
                             'n': E.N_HOLD, 'cap': E.CAP, 'p_max': E.P_MAX, 'constants': SC.constants(E.P_MAX),
                             'readings': SC.READINGS, 'definition': DEFINITIONS['report_only_rule'],
                             'w104_mirror': {'n': E.N_HOLD, 'cap': E.REPLAY_THROUGH, 'p_max': E.P_MAX,
                                             'use': 'gate only (cycles <= 187)'}},
        'captures': {
            'a_q_by_block_and_component': {
                'file': E.CREEP_FILE, 'key': 'q_decomposition',
                'definition': ('per TSO / DSO block (48): the weighted components of production\'s '
                               '_get_local_objective_components ' + ', '.join(E.Q_COMPONENTS) + ' and other = block '
                               'value (production _get_operational_recourse_block_components) - classified_total; the '
                               'components of a block sum to its contribution to gross Q; aggregates per agent (TSO, '
                               'DSO_5, DSO_7, DSO_9, ESSO) with ess_terms = ' + ' + '.join(E.ESS_TERMS) + '; deltas vs '
                               'the previous cycle; reconciliation with gross Q every cycle'),
                'q_by_agent_from_source': ('gross_operational_cost = sum over TSO blocks + sum over DSO blocks of '
                                           'weight x (objective_function_rule - contracted settlement - voltage pin); '
                                           'weight = years x days x annualisation. The ESSO does NOT enter gross Q '
                                           '(its only term in the NET recourse is the terminal salvage); its row is 0 '
                                           'by construction; its salvage and feasibility penalty are reported beside'),
                'shape': '48 blocks x 9 values (+ deltas) + 5 agents per cycle'},
            'b_ess_movement': {
                'file': E.CREEP_FILE, 'key': 'ess_movement', 'raw_file': E.ESS_SCHEDULE_FILE,
                'definition': ('per ESS node n (5, 7, 9) and family f: S_f,n(k) = sum over years, days, periods of '
                               '|x_f(k) - x_f(k-1)|, x read at production\'s get_admm_boyd_residual_metrics call of '
                               'cycle k (after the cycle\'s ESS consensus / dual update, before AA). Families: p and q '
                               'of the consensus copies esso (= the ESSO\'s es_pnet), tso (= TSO expected_shared_ess x '
                               's_base), dso (= DSO expected_shared_ess x s_base), z (the consensus value); charge and '
                               'discharge of esso (es_pch / es_pdch summed over cohorts), tso and dso '
                               '(probability-weighted shared_es_pch / shared_es_pdch x s_base) -- charge and discharge '
                               'are separate variables on every side, so they are recorded separately. Units MW / Mvar. '
                               'Unavailable (None) on the first cycle'),
                'shape': '36 blocks x 24 periods; 14 families; 12,096 floats per raw cycle line'},
            'c_boyd_full': {'file': E.CREEP_FILE, 'key': 'boyd',
                            'definition': ('every field production\'s get_admm_boyd_residual_metrics returns per '
                                           'channel (v, pf, ess): r, s (the raw primal and dual residual norms), '
                                           'eps_pri, eps_dual, primal_ratio, dual_ratio, and the rest; plus '
                                           'all_boyd_pass, eps_abs, eps_rel')},
            'lambda_t': 'interface_duals_per_cycle.jsonl (harness default, unchanged)',
            'write_only': 'checks L1 (real W104 cycle-187 models: fingerprints before == after)',
            'capture_path_asserted_before_any_solve': 'assert_extension_preconditions (checklist incl. capture:*)'},
        'definitions': DEFINITIONS,
        'predictions_recorded_before_any_run': PREDICTIONS,
        'w104_reference_slopes_152_187': ref_slopes,
        'scorer': {'function': f'{SCRIPT_NAME}:score_extension', 'self_tests': scorer_tests},
        'gates': {
            'G1-G5, G7, G9, G11': 'as W104 (W86 evaluation_checks; solve profile 51 x (287 + 1) + retries)',
            'G6': 'v37 (Optimal Solution Found + four metrics within the tail tolerances, final accepted attempt; B = 48)',
            'G8': 'persistence as W104', 'G13': 'holds inert through 87, held 88..287; certificate length 10**9 every cycle',
            'G14': 'all-block capture', 'G15': 'fixed length: cycles_run == 287, stopped by the cap',
            'G16': 'lambda_t sidecar complete', 'G17': 'the pure rules replay the in-cycle records (report-only + mirror)',
            'G18': 'line fields', 'G19': 'replay 1..187 bitwise every field', 'G20': 'production counters',
            'G21': 'creep captures complete, reconciling, Boyd equal to production\'s g rows, no capture errors',
            'G22': 'cycle lines and all-block lines 1..187 equal W104\'s',
            'post_run_evaluator_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w105_extension_checks.py', 'committed_output': checks_file,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()}},
                              'rerun_before_launch': 'the --run mode re-runs every check and refuses unless all hold'},
        'verbatim_text': {'quotes': VERBATIM, 'check': verb},
        'labelling_and_identity': {
            'label': E.LABEL,
            'evaluation_key': ('sha256({base_evaluation_key, settling_extension}); every key without the declaration '
                               'byte-identical to the pre-W105 harness (checks K: all committed entries)')},
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solver': solver,
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (287 + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'launch_command_template': launch_command(),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised zero-solve by checks R (real wrappers, recorded values) and H4 (real install); '
                             'THE FIRST REAL CYCLE OF THE LAUNCH IS THE SMOKE (its in-cycle replay gate at cycle 1)'),
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


def freeze_spec(started):
    tag = f'W105-V{SPEC_VERSION}'
    failures = _common_checks()
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'v{SPEC_VERSION} already exists (write-once): {existing}')
    cf = _checks_file_state()
    if not (cf['all_hold'] is True and cf['committed_clean']
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in K.CODE_PINNED_BY_CHECKS:
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    code_since = code_since_w104()
    if not code_since['ok']:
        failures.append(f'production changed since W104\'s launch: {code_since}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append('inline re-run of the zero-solve checks fails -- failing sections and items: '
                        f'{_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    scorer_tests, scorer_ok = scorer_self_tests()
    if not scorer_ok:
        failures.append(f'scorer self-tests fail: {[k for k, v in scorer_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found in the brief: {verb["found_whitespace_normalised"]}')
    solver = W101L.solver_check()
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
    ref_slopes = w104_reference_slopes()
    overhead = cf.get('L4_overhead') or 0.0
    wall = wall_time_estimate(overhead)
    content = stage_spec_content(checks_inline, cf, post_tests, scorer_tests, verb, solver, pre, code_since, mem,
                                 ref_slopes, wall)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError(f'v{SPEC_VERSION} written bytes do not hash to the name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] wrote {rel} sha256={sha} (predecessor v40 {SPEC_V40['sha256']})")
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f"[{tag}] scorer self-tests: { {k: v.get('ok') for k, v in scorer_tests.items()} }")
    _log(f"[{tag}] keys: base {pre['base_key_without_extension'][:16]} == recert; extension {pre['extension_key']}")
    _log(f"[{tag}] W104 slopes 152-187 (pf_primal {ref_slopes['boyd_pf_primal_ratio']['slope_152_187']!r}, ess_primal "
         f"{ref_slopes['boyd_ess_primal_ratio']['slope_152_187']!r}); mean dQ 143-187 {ref_slopes['mean_dQ_143_187']!r}")
    _log(f"[{tag}] expected wall {wall['expected_wall_h']:.2f} h; memory at freeze (non-gating): available "
         f"{mem.get('available_gib')} GiB")
    _finish(0, '-- next: --freeze')


def freeze(started):
    tag = 'W105-FREEZE'
    failures = _common_checks()
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        if not _committed_clean(ss_rel):
            failures.append(f'the stage spec v{SPEC_VERSION} is not committed / clean')
        if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
            failures.append(f'code changed since v{SPEC_VERSION} froze')
    except RuntimeError as error:
        failures.append(str(error))
        ss = None
    failures += H.check_campaign_preconditions(campaign_root(), extra_clean_files=EXTRA_CLEAN_FILES)
    pre = pre_launch_assertion()
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails: {pre["parts"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
             'stage_spec': ss_pin, 'label': E.LABEL, 'cell': CELL, 'w104': W104_PINS,
             'w104_eval_key': w104_entry()['eval_key'], 'recert_eval_key': recert_entry()['eval_key'],
             'expected_eval_key': pre['extension_key'], 'objective_convention': ss['objective_convention'],
             'solve_claim': ss['solve_profile_declared'], 'pre_launch_assertion_at_freeze': pre,
             'supersedes_before_any_run': SUPERSEDED}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        campaign_root(), CAMPAIGN_ID, entries(), configuration=configuration(), cap=E.CAP, concurrency=CONCURRENCY,
        authority=[f'{BRIEF} Addendum 54 Ruling 1', 'Planner task W105', 'Planner task W108', ss_rel],
        required_consecutive_cycles=10,
        extra=extra)
    checks = validate_campaign_spec(spec, ss_pin)
    pre_frozen = pre_launch_assertion(spec)
    ok = all(checks.values()) and pre_frozen['holds']
    e = spec['candidates'][0]
    _log(f'[{tag}] {CELL}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    _log(f"[{tag}]   eval_key={e['eval_key']} eval_dir={e['eval_dir']} cap={spec['cap']} "
         f"post_certification={e['post_certification']}")
    _log(f"[{tag}]   spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}; "
         f"pre-launch on the frozen spec holds={pre_frozen['holds']}")
    _log(f'[{tag}]   LAUNCH: {launch_command(spec_sha)}')
    _finish(0 if ok else 1, f'freeze {"OK" if ok else "NOT OK"}')


def _campaign_spec_path():
    root = campaign_root()
    hits = sorted(f for f in os.listdir(root) if f.startswith('campaign_spec_')) if os.path.isdir(root) else []
    return os.path.join(root, hits[0]) if len(hits) == 1 else None


def run(started, spec_sha256, preconditions_only=False):
    tag = 'W105-RUN-PRECONDITIONS-ONLY' if preconditions_only else 'W105-RUN'
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append(f'stage spec v{SPEC_VERSION} not committed / clean')
    if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
        failures.append(f'code changed since v{SPEC_VERSION} froze')
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
        failures.append(_checks_failure_message(checks_inline))
    pre = pre_launch_assertion(spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["parts"]}')
    solver = W101L.solver_check()
    if not solver['ok'] or solver['sha256'] != ss['solver']['sha256']:
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
        # W108: stop here -- before the campaign lock, the evaluation and the child (zero solves, nothing written)
        _log(f"[{tag}] every precondition holds: campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; "
             f"stage spec {ss_rel} sha256={ss_sha}; checks per section "
             f"{ {k: v['holds'] for k, v in checks_inline['sections'].items()} }; pre-launch parts {pre['parts']}; "
             f"eval_key {entry['eval_key']}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} sha256={solver['sha256']}")
        _log(f'[{tag}] STOPPED before the campaign lock and the child (no lock taken, no evaluation, zero solves)')
        _finish(0, 'preconditions-only OK')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cell {CELL} eval_key {entry['eval_key']}; cap {spec['cap']}; lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    try:
        gates, detail, rec = cell_gates(entry, eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
        gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}, {}
    report = None
    try:
        if os.path.isfile(os.path.join(eval_dir, E.DECISION_FILE)) and os.path.isfile(
                os.path.join(eval_dir, 'per_cycle_record.jsonl')):
            report = cell_report(eval_dir, rec)
    except Exception as error:  # noqa: BLE001
        report = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    summ = (rec or {}).get('settling_extension_summary') or {}
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': CELL, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'],
               'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'gate_detail': detail, 'diagnostic_report': report,
               'pre_launch_assertion': pre, 'memory_preflight_at_run': mem, 'solver': solver, 'batch_info': batch,
               'parent_view': (records[0] or {}).get('parent_view'), 'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    if summ.get('replay_first_divergence'):
        _log(f"[{tag}] REPLAY DIVERGED -- run ABORTED: {summ['replay_first_divergence']}")
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] outcome {report.get('outcome')} held {report.get('predictions_held')} fired "
             f"{report.get('triggers_fired')} mean dQ {report.get('mean_dQ_188_287')} first would-certify "
             f"{((report.get('report_only_settling') or {}).get('first_would_certify') or {}).get('k_star')}")
    code = 0 if (all(gates.values()) and _guards_ok(g) and isinstance(report, dict) and 'error' not in report) else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


def summarize(started):
    tag = 'W105-SUMMARY'
    ss_rel, ss_sha, _ss = load_stage_spec()
    path = os.path.join(campaign_root(), RESULTS_FILE)
    out_rel = os.path.join(ROOT_REL, SUMMARY_FILE)
    if not os.path.isfile(path) or not _committed_clean(os.path.relpath(path, REPO)) or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] results not committed, or the summary exists')
        _finish(1)
    res = json.load(open(path))
    spec = _load(res['campaign_spec_path'])
    eval_dir = os.path.join(campaign_root(), 'evals', spec['candidates'][0]['eval_dir'])
    rec = json.load(open(os.path.join(eval_dir, 'evaluation_record.json')))
    rescored = cell_report(eval_dir, rec)
    same = K._jt(rescored) == K._jt(res.get('diagnostic_report'))
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'stage_spec': {'path': ss_rel, 'sha256': ss_sha},
           'objective_convention': DEFINITIONS['objective_convention'], 'rescored': rescored,
           'equals_the_run_report': same, 'definitions': DEFINITIONS, 'predictions': PREDICTIONS}
    H._write_once_json(_abs(out_rel), doc)
    H._write_once_json(_abs(os.path.join(ROOT_REL, SUMMARY_MANIFEST)), {out_rel: _sha(out_rel)})
    _log(f"[{tag}] outcome {rescored.get('outcome')} (equals the run's report: {same})")
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--preconditions-only', action='store_true',
                        help='with --run: every --run precondition, then stop before the lock and the child')
    args = parser.parse_args()
    if args.preconditions_only and not args.run:
        parser.error('--preconditions-only requires --run')
    started = time.time()
    try:
        if args.freeze_spec:
            freeze_spec(started)
        elif args.freeze:
            freeze(started)
        elif args.summarize:
            summarize(started)
        else:
            if not args.spec_sha256:
                parser.error('--run requires --spec-sha256')
            run(started, args.spec_sha256, preconditions_only=args.preconditions_only)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
