"""
P5.15 Addendum 53, Planner task W101 -- the SRP1 THREE-REFERENCE SETTLING CONTINUATION: frozen stage spec v39
(predecessor v38 8bc0ffa6, NOT edited), the three per-cell campaign freezes, the per-cell run, and the zero-solve
summary. BUILT AND FROZEN IN W101; NO RUN IS LAUNCHED IN W101 (the Planner launches, one cell per launch).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 53 (Ruling 1: the settling criterion, dR = 0.07, tau = 4,539;
Ruling 2: the SRP1 references first -- bitwise replay to certification, then the certifying regime until the criterion
certifies or 100 cycles; the expert's predictions; "Records": lambda_t per node in the default per-cycle records);
TASKS.md Addendum 53 order (the Advisor's exact algorithm adopted by the Planner, its readings, the projections and the
competing prediction); Planner task W101; P5_15_ADDENDUM52_SETTLING_REVIEW_NOTE.md option 1(a).

THE CELLS. The SRP1 tight-tail re-certification (campaign s53_w86_tail_recert, spec ddd6cd44, run at git 31a37efb,
evidence 0a4bf784) -- the basis of R_ref = 259,375.33:
    x0        eval key 5cfe69a6...  certified at N = 132
    n7_4h_e1  eval key ca8927e7...  certified at N = 112   (the unit)
    c_star    eval key 96c5aa50...  certified at N = 87
Each continuation runs EXACTLY the recert's configuration (case file + AA keep_memory declaration, ESS ageing baseline
C2, tight tail {True, 1e-6}, persist_certified_models, no overrides, no option (b) -- the recert declared none), plus
the keyed entry option `settling_continuation` (`p515_s53_w101_settling_continuation_hooks`): the bitwise replay gate
against the recert's committed per-cycle record with ABORT on the first divergence, the certifying regime held after N
(AA off, tail on, rho frozen), and the settling stop rule (`settling_criterion`) as the only exit before the cap N + 100.
One campaign (root, spec, lock) per cell; concurrency 1 (the recert ran the three at concurrency 3: concurrency is a
campaign-level setting and does not enter any per-cell setting).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-spec                   ZERO SOLVES. v39 (write-once, named by its sha256): re-runs the W101 zero-solve checks
                                  inline and pins their committed output; post-run evaluator self-tests; I from the
                                  SRP1 master expression; the code-since-recert check; the pre-launch assertions.
  --freeze                        ZERO SOLVES. The three per-cell campaign specs (s53_w101_srp1_cont_<cell>), each
                                  pinning v39; the per-cell pre-launch assertion on the frozen spec; the three exact
                                  launch commands.
  --run --cell C --spec-sha256 S  THE RUN OF ONE CELL (NOT RUN IN W101). Cell order x0 -> n7_4h_e1 -> c_star is
                                  enforced. Preconditions (the zero-solve checks re-run, the pre-launch assertion, the
                                  memory preflight, the solver path, the run-lock), H.evaluate on the one entry, then
                                  the gates, the settling report; results + manifest, write-once.
  --summarize                     ZERO SOLVES. Reads the three cells' committed results and scores the recorded
                                  predictions with the frozen definitions (P1-P3, dV, V_new - I, the projections).

Exit codes: freezes 0 done / 1 failure; --run 0 every gate holds, 1 a gate / harness / guard / precondition failure
(a replay divergence aborts the cell: the child exits 1 with its barrier record; the launcher reports the cycle and
magnitude and exits 1).
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W101 SRP1 settling-continuation launcher (never solves)'
                                 ).install()

# The W98 launcher (and through it the W90 / W89 / W86 chain, each arming its own permitted=() guard): its G6 v37
# evaluator, the W86 launcher helpers (L), the acceptance cross-check (X). The W101 checks (arms its own guard).
import p515_s53_w98_continuation_campaign as W98L  # noqa: E402
import p515_s53_w101_continuation_checks as K  # noqa: E402
import p515_s53_w101_settling_continuation_hooks as C  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import settling_criterion as SC  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9 = W98L.H, W98L.L, W98L.X, W98L.W9
GUARDS = ((('w101_checks', K.GUARD),) + tuple(zip(W98L.GUARD_NAMES, W98L.GUARDS_LIFO))
          + (('w101_parent', PARENT_GUARD),))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s53_w101_srp1_continuation_campaign'
STAGE_TEXT = ('P5.15 Addendum 53, W101 -- SRP1 three-reference settling continuation: each tight-tail reference replayed '
              'under its certifying configuration (bitwise gate per cycle against the recorded 1..N, abort on the first '
              'divergence), then the certifying regime held (AA off, tight tail on, rho frozen) until the settling stop '
              'rule certifies or the cap N + 100; lambda_t, all blocks and pf_primal recorded every cycle')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W101_ROOT_REL
SPEC_PREFIX = 'frozen_s53_spec_v39_'
SPEC_VERSION = 39
SPEC_V38 = {'path': os.path.join(_P53, 'frozen_s53_spec_v38_8bc0ffa6.json'),
            'sha256': '8bc0ffa688e3e95e83f31b3895c0689e347d6305d7f673899ab795c70b881103'}
RECERT = {
    'campaign_id': 's53_w86_tail_recert', 'root': C.RECERT_ROOT, 'run_git': '31a37efb', 'evidence_commit': '0a4bf784',
    'campaign_spec': {'path': os.path.join(C.RECERT_ROOT, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json'),
                      'sha256': 'ddd6cd4422b56f946eeeb427dff44ef572490bd34c606fad96eaf00fd4a84ec1'},
    'campaign_results': {'path': os.path.join(C.RECERT_ROOT, 'campaign_results.json'),
                         'sha256': '361402fed03c16c65e1502a587d0e21d279b79f83faeadea7b41a1c99b756480'},
    'campaign_manifest': {'path': os.path.join(C.RECERT_ROOT, 'campaign_manifest_sha256.json'),
                          'sha256': '83f8dab9997d239b671490331941edf285c16b0cf49179390c9eb35bd1cc0009'},
}
PRODUCTION_UNCHANGED_SINCE_RECERT = ('shared_resources_planning.py', 'network.py', 'admm_parameters.py',
                                     'admm_anderson_acceleration.py', 'model_construction_helpers.py')
CAMPAIGN_IDS = {cell: f's53_w101_srp1_cont_{cell}' for cell in C.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
SUMMARY_FILE = 'w101_three_reference_summary.json'
SUMMARY_MANIFEST = 'w101_three_reference_summary_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
SOLVER_PATH = '/usr/local/bin/ipopt'
PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
GIB = 1 << 30
N_NETWORK_BLOCKS = 48          # (1 TSO + 3 DSO) x 3 years x 4 days (SRP1)
N_ESSO = 3
SOLVES_PER_ROUND = N_NETWORK_BLOCKS + N_ESSO
V_SRP1 = 259375.32654094696    # recert campaign_results R.R_tail (R_ref = 259,375.33 rounded)
I_CITED = {'value_eur': 317957.0085035586, 'rounded': 317957.01,
           'cited': [{'path': 'P5_15_ADDENDUM28_AGEING_REPORT.md',
                      'sha256': 'a90dd1548d45571996304fbec320d66707be92954064cf028262cfa5921c4d17',
                      'text': 'I(x) = 317,957 at the smallest node-7 4 h unit (0.25 MVA / 1.0 MWh, 2025, candidate '
                              '`db77e154...`)'},
                     {'path': os.path.join(_P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'campaign_results.json'),
                      'field': 'value_and_R.I_eur', 'value': 317957.0085035586,
                      'source': 'p515_s53_w89_3x3_campaign.instance facts: pe.value(master.investment_cost), 3 x 3'}]}
EXTRA_CLEAN_FILES = (SCRIPT_NAME, 'p515_s53_w101_settling_continuation_hooks.py', 'p515_s53_w101_continuation_checks.py',
                     'settling_criterion.py', 'interface_dual_capture.py', 'gate_result_io.py',
                     'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
                     'p515_s53_w98_continuation_checks.py', 'p515_s53_w90_3x3_campaign.py',
                     'p515_s53_w89_3x3_campaign.py', 'p515_s53_w95_x0_drift_diagnostics.py',
                     'p515_s53_w86_tail_recert_campaign.py', H.ESS_PARAMS_FILE_REL,
                     'shared_energy_storage_parameters.py')
CODE_PINNED = ('p515_s53_w101_srp1_continuation_campaign.py', 'p515_s53_w101_settling_continuation_hooks.py',
               'p515_s53_w101_continuation_checks.py', 'settling_criterion.py', 'interface_dual_capture.py',
               'gate_result_io.py', 'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py',
               'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
               'p515_s53_w98_continuation_checks.py', 'p515_s53_w86_tail_recert_campaign.py',
               'p515_s53_w90_3x3_campaign.py', 'p515_s53_w89_3x3_campaign.py',
               'p515_s53_w89_g6_final_attempt_reeval.py', 'p515_s53_w95_x0_drift_diagnostics.py',
               'p515_gate_result_bool_typing_test.py')

# ---- verbatim text (Addendum 53; checked against the committed brief, whitespace-normalised, at every freeze) --------
VERBATIM = {
    'ruling_1_tau': '**δR = 0.07 → τ = δR·V_SRP1/4 = 4,539 € per cell.**',
    'ruling_1_criterion': ('after the residuals pass, at least two sign changes of the objective step (the period is '
                           'measured, not assumed); successive swings not growing; the range of Q over one full period ≤ '
                           'τ; no fit in the decision. **Add a monotone branch:** if no sign change occurs within 2× the '
                           'longest period seen on the instance, certify when the steps are decreasing and the range over '
                           'that window ≤ τ.'),
    'expert_predictions': ("**Expert's predictions:** settled slacks per cell within [5, 25] k€; the value's settled "
                           "change |ΔV| ≤ 20 k€, so the sign margin survives; the three slacks alike to within ≈ 5 k€"),
    'reopen_rule': '**If |ΔV| > 40 k€ the SRP1 sign conclusion is reopened.**',
    'records': ('**Records.** The per-period interface consensus duals (λ_t per node) join the default per-cycle records '
                'from now on — a write-only change'),
}

# ---- predictions, recorded BEFORE any run ----------------------------------------------------------------------------
PREDICTIONS = {
    'expert_P1': {'statement': '5,000 <= |s_i| <= 25,000 for each cell', 'lo': 5000.0, 'hi': 25000.0,
                  'indeterminate_if': 'a boundary (5,000 or 25,000) lies within band_i of |s_i|'},
    'expert_P2': {'statement': '|dV| <= 20,000, dV = s_x0 - s_unit, resolution band_x0 + band_unit', 'bound': 20000.0,
                  'reopen_rule': 'if |dV| > 40,000 the SRP1 sign conclusion is REOPENED', 'reopen_bound': 40000.0,
                  'also_reported': 'V_new - I, V_new = V_SRP1 + dV, I = the unit\'s investment cost (verified at freeze)',
                  'indeterminate_if': '| |dV| - bound | <= resolution'},
    'expert_P3': {'statement': 'spread = max(s_i) - min(s_i) <= ~5,000', 'bound': 5000.0,
                  'resolution': 'band of the argmax cell + band of the argmin cell (each <= tau when certified: up to 2 tau)',
                  'pre_declared': ('P3 sits AT the resolution (2 tau = 9,078 >= 5,000) and is LIKELY INDETERMINATE; '
                                   'recorded as such before any run'),
                  'indeterminate_if': '| spread - bound | <= resolution'},
    'advisor_projections_labelled_projections': {
        'k_star': {'x0': [162, 173], 'n7_4h_e1': [141, 151], 'c_star': [121, 131]},
        'earliest_possible_from_records': {'x0': 148, 'n7_4h_e1': 127, 'c_star': 106},
        'campaign_wall_h': [3.0, 3.8], 'worst_case_all_caps_h': [4.4, 5.3], 'per_cell_worst_h': 1.9},
    'advisor_competing_H_b': {
        'statement': ('the post-k0 regime settles MONOTONICALLY to a lower limit: only the monotone branch can fire '
                      '(earliest k0 + 46) and s_x0 < 0'),
        'plateau_warning': ('pf_primal creeps back and Boyd lapses; Q returns toward the pre-k0 band (+38 to +53 k EUR '
                            'above Q_N)'),
        'scored_by': ('H-b holds for a cell if it certifies on the monotone branch; s_x0 < 0 checked on x0; plateau: '
                      'the number of Boyd lapses after N, the largest pf_primal ratio after N, and whether Q at k* / the '
                      'cap lies in [Q_N + 38,000, Q_N + 53,000]')},
}

# ---- the per-cell report definitions (frozen; implemented by C.settling_report and score_predictions) ----------------
REPORT_DEFINITIONS = {
    'Q_cert_old': 'Q_N (the recorded certification cycle\'s gross_operational_cost; bitwise the recert\'s by the replay)',
    'Q_cert_new': 'Q_k* (the settling certification cycle)',
    'k_star': 'the cycle at which the settling rule certified (None when uncertified at the cap)',
    'branch': 'oscillatory | monotone', 'k0': 'the rule\'s k0 at k*', 'T': 'turning points [(t, kind, Q_t)]',
    'A': 'half-swings', 'P_hat': 'T[-1].t - T[-3].t', 'W': 'max(20, ceil(1.1 P_hat)) (monotone: L_MONO)',
    'band': '[min, max] Q over the certifying window (uncertified: over the last max(W_MIN, ceil(W_FACTOR P_hat)) '
            'cycles, or the last W_MIN cycles without P_hat)',
    'band_width': 'max - min of the band', 'range_over_tau': 'range of the certifying window / TAU',
    's': 's = Q_k* - Q_N (signed; resolution = band_width); uncertified: Q_cap - Q_N, flagged',
    'settled_slack': '|s|',
    'report_only': {'c': 'A[-1] / A[-2]', 'c_A_over_1_minus_c': 'c A[-1] / (1 - c) (c < 1 only)',
                    'mid_band_minus_Q_N': '(band_lo + band_hi) / 2 - Q_N'},
    'terminal_step_to_threshold': ('|dQ_last| / EPS0 and range_over_tau at k*; production\'s rule ten '
                                   '(objective_change_abs / objective_tolerance) of the last cycle, reported'),
    'objective_convention': 'gross_operational_cost (settlement excluded); net and salvage reported beside',
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
    st = L._git_state(rel)
    return bool(st.get('git_tracked') and st.get('git_clean'))


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W101] guards {g} {extra_msg}')
    for _n, guard in GUARDS:
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and (OWN_PROCESS_SUBSTRING in parts[1]
                                                             or 'p515_s53_w98_continuation_campaign' in parts[1]):
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


def _norm(text):
    return ' '.join(text.split())


# ======================================================================================================================
#  the recert configuration and the per-cell entries
# ======================================================================================================================
def recert_spec():
    return _load(RECERT['campaign_spec']['path'])


def recert_entry(cell):
    return next(e for e in recert_spec()['candidates'] if e['label'] == cell)


def configuration():
    cfg = recert_spec()['configuration']
    return {'name': 'W101 SETTLING CONTINUATION of an SRP1 tight-tail reference -- ' + cfg['name'],
            'arm_label': cfg['arm_label'], 'overrides': dict(cfg['overrides']),
            'case_file_anderson_acceleration': dict(cfg['case_file_anderson_acceleration']),
            'ess_ageing_baseline': json.loads(json.dumps(cfg['ess_ageing_baseline'])),
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': dict(cfg['convergence_depth_tail']),
            'note': ('the recert configuration (spec ddd6cd44) unchanged; the entry adds settling_continuation (keyed); '
                     'cap N + 100; persist_certified_models as the recert; no option (b) (the recert declared none); '
                     'concurrency 1')}


def entries(cell, p_max):
    e = recert_entry(cell)
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    pc = e['post_certification']
    opts = {'investment_year': e['canonical']['investment_year'],
            'post_certification': {'persist_certified_models': pc['persist_certified_models'],
                                   'hull_polish': pc['hull_polish'], 'reference': pc['reference']},
            'settling_continuation': C.declaration_for(cell, p_max)}
    return [(cell, nodes, opts)]


def expected_keys(cell, p_max):
    e = recert_entry(cell)
    cfg = recert_spec()['configuration']
    kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
              convergence_depth_tail=cfg['convergence_depth_tail'])
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    cont = H.evaluation_key(e['key'], e['overrides'], settling_continuation=C.declaration_for(cell, p_max), **kw)
    return {'base_key_without_continuation': base, 'continuation_key': cont}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, p_max, spec=None):
    k = expected_keys(cell, p_max)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    rec = recert_entry(cell)
    eval_dir_name = H.eval_dir_name(k['continuation_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['continuation_key'])
    work = L._work_dir()
    entry = next((e for e in (spec or {}).get('candidates') or [] if e['label'] == cell), None)
    parts = {
        'base_key_equals_recert_key': k['base_key_without_continuation'] == rec['eval_key'],
        'continuation_key_differs_from_recert_key': k['continuation_key'] != rec['eval_key'],
        'continuation_key_absent_from_committed_specs_outside_w101_root': k['continuation_key'] not in committed,
        'campaign_root_differs_from_recert': os.path.abspath(campaign_root(cell)) != os.path.abspath(_abs(C.RECERT_ROOT)),
        'eval_dir_name_differs_from_recert': eval_dir_name != rec['eval_dir'],
        'working_dir_ids_differ_from_recert': not (set(ids.values()) & set(rec['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['continuation_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'recert_working_dir_ids': rec['working_dir_ids'], 'n_committed_keys_scanned': len(committed),
            'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(cell, spec, ss_pin, p_max):
    rs = recert_spec()
    re_, rc = recert_entry(cell), rs['configuration']
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    same_cfg = ('arm_label', 'case_file', 'case_file_sha256', 'overrides', 'apply_rho', 'full_diagnostics_in_rows',
                'case_file_anderson_acceleration', 'ess_ageing_baseline', 'ess_ageing_baseline_label',
                'convergence_depth_tail', 'ess_params_file')
    checks = {f'configuration:{k}': (cfg.get(k) == rc.get(k) if k != 'ess_params_file' else
                                     (cfg.get(k) or {}).get('sha256') == (rc.get(k) or {}).get('sha256'))
              for k in same_cfg}
    same_entry = ('canonical', 'key', 'overrides', 'effective_anderson_acceleration', 'post_certification')
    checks.update({f'entry:{k}': e.get(k) == re_.get(k) for k in same_entry})
    checks.update({
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == cell,
        'entry_settling_is_the_declaration': e.get('settling_continuation') == C.declaration_for(cell, p_max),
        'entry_has_no_release_solution_bookkeeping_as_recert': ('release_solution_bookkeeping' not in e
                                                                 and 'release_solution_bookkeeping' not in re_),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_IDS[cell],
        'cap_N_plus_100': spec.get('cap') == C.CELLS[cell]['N'] + SC.CAP_AFTER_N,
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_as_recert': spec.get('required_consecutive_cycles') == rs['required_consecutive_cycles']
        == 10,
        'bar_window_as_recert': spec.get('bar_window_cycles') == rs['bar_window_cycles'],
        'thread_caps_as_recert': spec.get('thread_caps') == rs['thread_caps'],
        'interpreter_as_recert': spec.get('interpreter') == rs['interpreter'],
        'solver_path_as_recert': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH')
        == (rs.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'stage_spec_pinned': (spec.get('extra') or {}).get('stage_spec') == ss_pin,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
    })
    return checks


# ======================================================================================================================
#  provenance
# ======================================================================================================================
def code_since_recert():
    """Production (PRODUCTION_UNCHANGED_SINCE_RECERT) byte-identical between the recert's run commit and HEAD, and no
    uncommitted change to any tracked .py; every other changed tracked .py listed (harness / gates / new modules)."""
    base = RECERT['run_git']
    diff = H._git(['diff', '--name-status', base, 'HEAD', '--', '*.py']).splitlines()
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    changed = [{'status': line.split('\t', 1)[0], 'path': line.split('\t')[-1]} for line in diff]
    prod_changed = [c for c in changed if c['path'] in PRODUCTION_UNCHANGED_SINCE_RECERT]
    modified = sorted(c['path'] for c in changed if c['status'].startswith('M'))
    return {'recert_run_commit': base, 'head': H._git(['rev-parse', 'HEAD']), 'changed_py': changed,
            'modified_py': modified, 'production_changed': prod_changed, 'uncommitted_tracked_py': dirty,
            'note': ('the harness changed (W98 option, W100 writer, W101 option + lambda_t capture) and the gates module '
                     '(W100 writer migration, byte-identical output); production unchanged'),
            'ok': not prod_changed and not dirty}


def solver_check():
    res = H._resolve_solver_path_from_dotenv()
    path = res.get('NLP_SOLVER_PATH')
    sha = H.sha256_file(path) if path and os.path.isfile(path) else None
    rec = _load(os.path.join(C.RECERT_ROOT, 'evals', C.CELLS['x0']['eval_dir'], 'evaluation_record.json'))
    return {'resolved': res, 'expected': SOLVER_PATH, 'sha256': sha,
            'recert_x0_child_path': rec.get('nlp_solver_path_in_child'),
            'ok': path == SOLVER_PATH and sha is not None and rec.get('nlp_solver_path_in_child') == SOLVER_PATH}


def _recert_pins_ok():
    failures = []
    for name, pin in (('recert campaign spec', RECERT['campaign_spec']), ('recert results', RECERT['campaign_results']),
                      ('recert manifest', RECERT['campaign_manifest']), ('spec v38', SPEC_V38)):
        if _sha(pin['path']) != pin['sha256'] or not _committed_clean(pin['path']):
            failures.append(f'{name} not as committed: {pin["path"]}')
    manifest = _load(RECERT['campaign_manifest']['path'])
    for cell in C.CELL_ORDER:
        rel = C.reference_path(cell)
        if manifest.get(rel) != C.CELLS[cell]['per_cycle_record_sha256'] or _sha(rel) != manifest.get(rel) \
                or not _committed_clean(rel):
            failures.append(f'{cell} reference per-cycle record not as committed / manifest: {rel}')
    return failures


def _common_checks():
    failures = []
    failures += _recert_pins_ok()
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a continuation launcher is alive: {others}')
    return failures


def verbatim_check():
    text = _norm(open(_abs(BRIEF), encoding='utf-8').read())
    found = {k: _norm(v) in text for k, v in VERBATIM.items()}
    return {'brief': BRIEF, 'brief_sha256_at_freeze': _sha(BRIEF), 'brief_git_state': L._git_state(BRIEF),
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def investment_cost_srp1():
    """I = the unit's investment cost from the SRP1 planning's own master expression (no solve), beside the cited
    317,957.0085035586."""
    import pyomo.environ as pe
    import p56a_oracle as O
    eval_id = f'p515s53_w101_I_check_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)
    try:
        planning = O.fresh_planning(eval_id)
        sed = planning.shared_ess_data
        master = sed.build_master_problem()
        e = recert_entry('n7_4h_e1')
        x = {(n, y): {'s': 0.0, 'e': 0.0} for n in sed.active_distribution_network_nodes for y in sed.years}
        for n, (s_val, e_val) in e['canonical']['nodes'].items():
            if s_val or e_val:
                ykey = next(y for y in sed.years if int(y) == int(e['canonical']['investment_year']))
                x[(int(n), ykey)] = {'s': float(s_val), 'e': float(e_val)}
        cand = O.vector_to_candidate(planning, x)
        sed.load_candidate_solution_into_master_model(master, cand)
        i_master = float(pe.value(master.investment_cost))
        i_transcription = O.investment_cost(planning, cand)
    finally:
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)
    return {'I_srp1_master_expression': i_master, 'I_srp1_p56a_transcription': i_transcription,
            'candidate_key': recert_entry('n7_4h_e1')['key'], 'cited': I_CITED,
            'equals_cited_bitwise': i_master == I_CITED['value_eur'],
            'abs_difference_to_cited': abs(i_master - I_CITED['value_eur'])}


def wall_time_estimate():
    out = {}
    tot_min = tot_max = 0.0
    for cell in C.CELL_ORDER:
        d = os.path.join(C.RECERT_ROOT, 'evals', C.CELLS[cell]['eval_dir'])
        text = open(_abs(os.path.join(d, 'child_stdout.log')), errors='replace').read()
        walls = []
        for line in text.splitlines():
            if '[INFO] \t - Iteration ' in line and ': ' in line and line.rstrip().endswith(' s'):
                try:
                    walls.append(float(line.rsplit(': ', 1)[1].split()[0]))
                except ValueError:
                    pass
        rec = _load(os.path.join(d, 'evaluation_record.json'))
        child = rec['wall_time_s']['child_process_s']
        n = C.CELLS[cell]['N']
        per_tail = sum(walls[-12:]) / 12.0
        overhead = child - sum(walls)
        worst = sum(walls) + 100 * per_tail + overhead
        out[cell] = {'recert_cycles': len(walls), 'N': n, 'replay_s_at_recert_concurrency_3': sum(walls),
                     'mean_cycle_s': sum(walls) / len(walls), 'mean_last12_s': per_tail,
                     'init_terminal_persist_overhead_s': overhead, 'worst_case_cap_s': worst,
                     'worst_case_cap_h': worst / 3600.0, 'worst_case_cycles': n + 100}
        tot_max += worst
        tot_min += sum(walls) + overhead
    return {'basis': ('the recert\'s own per-cycle walls ("[INFO] - Iteration k: X s", child_stdout.log) at concurrency '
                      '3, and its child wall; a continuation cycle = the mean of the recert\'s last 12 cycles; not '
                      'included: the all-block and lambda_t captures per cycle (pure reads; not measured) and the '
                      'effect of concurrency 1 (expected faster)'),
            'per_cell': out, 'campaign_worst_case_all_caps_h': tot_max / 3600.0,
            'campaign_replays_only_h': tot_min / 3600.0,
            'task_statement': 'about 25-30 s/cycle; x0 <= 232 cycles ~ 1.9 h; campaign 4.4-5.3 h if every cell caps'}


# ======================================================================================================================
#  post-run gates
# ======================================================================================================================
def solve_profile_check(rec, n_records):
    sp = rec.get('solve_profile') or {}
    obs = sp.get('observed') or {}
    rounds = (rec.get('cycles_run') or 0) + 1
    ok = (sp.get('reconciliation_supported') is True and sp.get('identity_holds') is True
          and sp.get('solves_per_cycle') == SOLVES_PER_ROUND and sp.get('rounds') == rounds
          and sp.get('base_solves') == SOLVES_PER_ROUND * rounds and obs.get('blocked_solve') == 0
          and obs.get('blocked_exec') == 0 and obs.get('permitted_solve') == sp.get('expected_solves')
          and n_records == obs.get('permitted_solve') - N_ESSO * rounds)
    return ok, {'solve_profile': sp, 'n_network_records': n_records, 'rounds': rounds}


def replay_gate_full(cell, eval_dir, rec):
    """Rows 1..N of the run's per_cycle_record.jsonl against the recert's committed rows: EVERY field, as JSON text
    (floats by shortest repr: bit for bit). The in-cycle gate already aborts on a difference in REPLAY_GATED_FIELDS;
    this adds REPLAY_POST_RUN_ONLY_FIELDS and re-checks the rest from the written record."""
    n = C.CELLS[cell]['N']
    ref = {r['cycle']: r for r in _read_jsonl(_abs(C.reference_path(cell)))}
    path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    run = {r['cycle']: r for r in _read_jsonl(path)} if os.path.isfile(path) else {}
    first, detail, per = None, None, []
    for c in range(1, n + 1):
        a, b = run.get(c), ref.get(c)
        if a is None:
            per.append({'cycle': c, 'equal': False, 'missing_in_run': True})
            if first is None:
                first, detail = c, {'missing_in_run': True}
            continue
        diff = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k),
                                                                                                        sort_keys=True))
        per.append({'cycle': c, 'equal': not diff, 'fields_differing': diff})
        if diff and first is None:
            ga, gb = a.get('gross_operational_cost'), b.get('gross_operational_cost')
            first, detail = c, {'fields_differing': diff, 'gross_difference_run_minus_recorded':
                                (ga - gb) if (ga is not None and gb is not None) else None}
    return {'bitwise_through_N': first is None, 'first_divergence_cycle': first, 'divergence': detail, 'N': n,
            'fields': sorted(next(iter(ref.values()))), 'tolerance': 'bitwise (JSON text per field)',
            'per_cycle': per}


def hold_checks(cell, eval_dir, rec):
    n = C.CELLS[cell]['N']
    lines = _read_jsonl(os.path.join(eval_dir, C.CYCLE_FILE))
    summ = rec.get('settling_continuation_summary') or {}
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    replay = [by[c] for c in cycles if c <= n]
    cont = [by[c] for c in cycles if c > n]
    rho_n = ((by.get(n) or {}).get('rho') or {}).get('rho_after')
    parts = {
        'one_line_per_cycle_contiguous': cycles == list(range(1, (rec.get('cycles_run') or 0) + 1)),
        'no_hold_acted_through_N': all((x.get('holds') or {}) == {'aa': False, 'tail_apply': False, 'tail_next': False,
                                                                  'rho': False}
                                       or ((x.get('holds') or {}).get('aa') is None and x.get('gross') is None)
                                       for x in replay),
        'aa_held_off_after_N': all((x.get('aa') is None and x.get('gross') is None)
                                   or ((x.get('aa') or {}).get('hold') is True
                                       and (x.get('aa') or {}).get('action') == C.AA_OFF_ACTION) for x in cont),
        'tail_held_on_after_N': all((x.get('tail_apply') or {}).get('active_passed') is True
                                    and (x.get('tail_next') or {}).get('returned') is True for x in cont),
        'rho_frozen_after_N_at_cycle_N_values': bool(rho_n) and all(
            (x.get('rho') or {}).get('hold') is True and (x.get('rho') or {}).get('rho_after') == rho_n
            and not (x.get('rho') or {}).get('changed_channels') for x in cont),
        'certificate_length_disabled_except_the_settling_cycle': all(
            x.get('certificate_length_in_force_at_cycle_end') == C.CERTIFICATION_DISABLED_THRESHOLD
            or (x.get('certificate_length_in_force_at_cycle_end') == C.SETTLING_CERTIFIED_THRESHOLD
                and str((x.get('settling') or {}).get('decision') or '').startswith('certified')) for x in lines),
        'replay_equal_every_cycle_through_N': all(x.get('replay_equal') is True for x in replay) and len(replay) == n,
        'summary_ok': summ.get('ok') is True,
    }
    return all(parts.values()), {'parts': parts, 'rho_at_N': rho_n,
                                 'n_cycles_where_the_hold_changed_a_value': {
                                     'aa': sum(1 for x in cont if (x.get('aa') or {}).get('hold_changed_value')),
                                     'tail_apply': sum(1 for x in cont if (x.get('tail_apply') or {}).get('hold_changed_value')),
                                     'tail_next': sum(1 for x in cont if (x.get('tail_next') or {}).get('hold_changed_value'))},
                                 'summary': summ}


LINE_FIELDS_REQUIRED = ('cycle', 'phase', 'gross', 'gross_hex', 'net_operational_recourse', 'terminal_salvage_value',
                        'boyd_k', 'local_solves_ok', 'boyd_ratios', 'boyd_pf_primal_ratio', 'settling', 'holds',
                        'replay_equal', 'certificate_length_in_force_at_cycle_end', 't_start_s', 't_end_s')
SETTLING_FIELDS_REQUIRED = ('k0', 'eligible', 'dQ', 's_k', 'sign_change', 'turning_point', 'len_T', 'A', 'P_hat', 'W',
                            'window', 'range', 'range_over_tau', 'certA', 'certA_parts', 'certB', 'certB_parts',
                            'decision', 'reasons')


def line_fields_check(eval_dir):
    lines = _read_jsonl(os.path.join(eval_dir, C.CYCLE_FILE))
    missing = {}
    for x in lines:
        m = [f for f in LINE_FIELDS_REQUIRED if f not in x]
        if x.get('gross') is not None:
            m += [f for f in ('blocks_captured',) if f not in x]
        m += [f'settling.{f}' for f in SETTLING_FIELDS_REQUIRED if f not in (x.get('settling') or {})]
        m += [f'boyd_ratios.{f}' for f in C.BOYD_RATIO_FIELDS if f not in (x.get('boyd_ratios') or {})]
        if m:
            missing[str(x['cycle'])] = m
    return not missing and bool(lines), {'n_lines': len(lines), 'missing_by_cycle': missing}


def block_capture(eval_dir, rec, rows):
    lines = _read_jsonl(os.path.join(eval_dir, C.BLOCKS_FILE))
    ok_cycles = [r['cycle'] for r in rows if r.get('gross_operational_cost') is not None]
    by = {x['cycle']: x for x in lines}
    parts = {'one_line_per_successful_cycle': sorted(by) == ok_cycles,
             'all_blocks_every_line': all(x.get('n_blocks') == N_NETWORK_BLOCKS + 1 for x in lines),
             'every_line_reconciles_to_net': all(x.get('reconciles_to_net') is True for x in lines),
             'deltas_every_line_after_the_first': all('deltas_vs_previous_cycle' in by[c] for c in sorted(by)[1:]
                                                      if (c - 1) in by)}
    return all(parts.values()), {'parts': parts, 'n_lines': len(lines)}


def lambda_sidecar_check(eval_dir, rec):
    path = os.path.join(eval_dir, IDC.INTERFACE_DUAL_FILE)
    if not os.path.isfile(path):
        return False, {'error': 'sidecar missing'}
    header, lines = IDC.read_sidecar(path)
    summ = rec.get('interface_dual_capture') or {}
    cycles = rec.get('cycles_run') or 0
    parts = {'header': header is not None and header.get('n_blocks') == 36 and header.get('n_periods') == 24,
             'one_line_per_cycle': sorted(lines) == list(range(1, cycles + 1)),
             'every_line_captured': all(v.get('captured') is True for v in lines.values()),
             'shape_every_line': all(len(v.get(f) or []) == 36 and all(len(b) == 24 for b in v.get(f) or [])
                                     for v in lines.values() for _g, _a, _k, f in IDC.CHANNEL_FIELDS),
             'summary_ok': summ.get('ok') is True,
             'metadata_admm_objective_scale_non_mutable': header is not None and all(
                 b.get('admm_objective_scale_tso_mutable') is False and b.get('admm_objective_scale_dso_mutable') is False
                 for b in header['metadata']['per_block'])}
    return all(parts.values()), {'parts': parts, 'bytes': os.path.getsize(path),
                                 'bytes_per_cycle_line_max': summ.get('bytes_per_cycle_line_max')}


def settling_replay_check(cell, eval_dir, rec, p_max):
    """(k): the pure function replayed on the run's per_cycle_record (Q = gross_operational_cost; boyd = boyd_all_pass
    and local_solves_ok) with the same N, cap and P_MAX reproduces the in-cycle decision EXACTLY and every in-cycle
    per-cycle rule record, and never certifies at k <= N."""
    n = C.CELLS[cell]['N']
    cap = n + SC.CAP_AFTER_N
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = _read_jsonl(os.path.join(eval_dir, C.CYCLE_FILE))
    rule = SC.SettlingRule(n, cap, p_max)
    pure = []
    for r in rows:
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok'])))
    in_cycle = [x.get('settling') for x in lines]
    dec_path = os.path.join(eval_dir, C.DECISION_FILE)
    dec_run = json.load(open(dec_path)) if os.path.isfile(dec_path) else None
    dec_pure = rule.decision
    keys = ('status', 'k_star', 'branch', 'k0', 'T', 'A', 'P_hat', 'W', 'window', 'band', 'band_width', 'k_cap',
            'reasons', 'range')
    same_dec = dec_run is not None and dec_pure is not None and all(
        json.dumps(dec_run.get(k), sort_keys=True) == json.dumps(dec_pure.get(k), sort_keys=True) for k in keys)
    same_lines = [json.dumps(a, sort_keys=True) for a in in_cycle] == [json.dumps(b, sort_keys=True) for b in pure]
    early = [p['k'] for p in pure if p['k'] <= n and str(p.get('decision') or '').startswith('certified')]
    parts = {'decision_reproduced_exactly': same_dec, 'every_cycle_record_reproduced': same_lines,
             'never_certifies_at_k_le_N': not early, 'decision_file_present': dec_run is not None}
    return all(parts.values()), {'parts': parts, 'decision_run': dec_run, 'decision_pure': dec_pure}


def stopping_check(cell, rec, eval_dir):
    summ = rec.get('settling_continuation_summary') or {}
    k = rec.get('cycles_run')
    cap = C.CELLS[cell]['N'] + SC.CAP_AFTER_N
    dec_path = os.path.join(eval_dir, C.DECISION_FILE)
    dec = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}
    if dec.get('status') == 'certified':
        ok = k == dec.get('k_star') and summ.get('stopped_by') == 'settling_rule'
    else:
        ok = k == cap and dec.get('status') == 'uncertified' and summ.get('stopped_by') == 'cap'
    return bool(ok), {'cycles_run': k, 'cap': cap, 'decision_status': dec.get('status'), 'k_star': dec.get('k_star'),
                      'stopped_by': summ.get('stopped_by')}


def persistence_check(rec, eval_dir):
    pc = rec.get('post_certification') or {}
    certified = rec.get('status') == 'certified'
    has_pkl = os.path.exists(os.path.join(eval_dir, 'certified_models.pkl'))
    ok = (has_pkl and pc.get('status') not in (None, 'skipped', 'error')) if certified else (
        not has_pkl and pc.get('status') == 'skipped')
    return bool(ok), {'record_status': rec.get('status'), 'post_certification_status': pc.get('status'),
                      'certified_models_pkl': has_pkl,
                      'note': ('persist_certified_models as the recert: runs iff the harness\'s residual certificate '
                               'holds on the last row (consecutive Boyd-pass cycles >= 10), whatever the settling '
                               'verdict')}


def cell_gates(cell, entry, eval_dir, p_max):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('barrier') and rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_summary'] = rec.get('settling_continuation_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys(cell, p_max)['continuation_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = solve_profile_check(rec, len(records))
    g6 = W98L.g6_v37_evaluate_records(records, rec.get('cycles_run'), b=N_NETWORK_BLOCKS)
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_as_recert'], detail['G8'] = persistence_check(rec, eval_dir)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_N_and_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = lambda_sidecar_check(eval_dir, rec)
    gates['G17_settling_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir, rec, p_max)
    gates['G18_line_fields_i_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    rg = replay_gate_full(cell, eval_dir, rec)
    gates['G19_replay_bitwise_1_N_every_field'] = rg['bitwise_through_N']
    detail['G19'] = rg
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    return gates, detail, rec


def cell_report(cell, eval_dir, rec):
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    n = C.CELLS[cell]['N']
    q = {r['cycle']: r['gross_operational_cost'] for r in rows}
    dec = json.load(open(os.path.join(eval_dir, C.DECISION_FILE)))
    rep = C.settling_report(dec, q, n)
    lines = _read_jsonl(os.path.join(eval_dir, C.CYCLE_FILE))
    after = [x for x in lines if x['cycle'] > n]
    last = rows[-1]
    steps = [q[k] - q[k - 1] for k in sorted(q) if (k - 1) in q and q[k] is not None and q[k - 1] is not None]
    rep.update({
        'cell': cell, 'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
        'candidate_canonical': rec.get('candidate_canonical'), 'N': n, 'cycles_run': rec.get('cycles_run'),
        'net_operational_recourse_at_k': last.get('recourse'), 'terminal_salvage_value_at_k': last.get('terminal_salvage_value'),
        'terminal_step_abs': abs(steps[-1]) if steps else None,
        'terminal_step_over_EPS0': (abs(steps[-1]) / SC.EPS0) if steps else None,
        'rule_ten_last_cycle': ((last['objective_change_abs'] / last['objective_tolerance'])
                                if last.get('objective_change_abs') is not None and last.get('objective_tolerance')
                                else None),
        'boyd_lapses_after_N': sum(1 for x in after if not x.get('boyd_k')),
        'pf_primal_ratio_after_N': {'max': max((x.get('boyd_pf_primal_ratio') or 0.0) for x in after) if after else None,
                                    'last': after[-1].get('boyd_pf_primal_ratio') if after else None},
    })
    return rep


def score_predictions(reports, v_srp1=V_SRP1, i_eur=I_CITED['value_eur']):
    """The recorded predictions scored with the frozen definitions. `reports` = {cell: cell_report}. A cell that is not
    certified is scored 'not_scoreable_uncertified' (its values reported). Pure function."""
    out = {}
    p1 = {}
    for cell, r in reports.items():
        s, band = r.get('s_signed'), r.get('band_width')
        if r.get('status') != 'certified' or s is None:
            p1[cell] = {'verdict': 'not_scoreable_uncertified', 's': s, 'band_width': band}
            continue
        a = abs(s)
        near = any(abs(a - b) <= band for b in (PREDICTIONS['expert_P1']['lo'], PREDICTIONS['expert_P1']['hi']))
        inside = PREDICTIONS['expert_P1']['lo'] <= a <= PREDICTIONS['expert_P1']['hi']
        p1[cell] = {'abs_s': a, 'band_width': band, 'inside': inside,
                    'verdict': 'indeterminate' if near else ('held' if inside else 'missed')}
    out['expert_P1'] = p1
    x0, un = reports.get('x0') or {}, reports.get('n7_4h_e1') or {}
    if x0.get('status') == 'certified' and un.get('status') == 'certified':
        dv = x0['s_signed'] - un['s_signed']
        res = x0['band_width'] + un['band_width']
        bound, reopen = PREDICTIONS['expert_P2']['bound'], PREDICTIONS['expert_P2']['reopen_bound']
        v_new = v_srp1 + dv
        out['expert_P2'] = {'dV': dv, 'resolution': res, 'abs_dV': abs(dv),
                            'verdict': ('indeterminate' if abs(abs(dv) - bound) <= res else
                                        'held' if abs(dv) <= bound else 'missed'),
                            'reopen_rule_triggered': abs(dv) > reopen,
                            'reopen_rule_indeterminate': abs(abs(dv) - reopen) <= res,
                            'V_old': v_srp1, 'V_new': v_new, 'I': i_eur, 'V_new_minus_I': v_new - i_eur,
                            'V_new_minus_I_resolution': res,
                            'V_new_minus_I_determinate': abs(v_new - i_eur) > res}
    else:
        out['expert_P2'] = {'verdict': 'not_scoreable_uncertified'}
    cert = {c: r for c, r in reports.items() if r.get('status') == 'certified'}
    if len(cert) == 3:
        hi = max(cert, key=lambda c: cert[c]['s_signed'])
        lo = min(cert, key=lambda c: cert[c]['s_signed'])
        spread = cert[hi]['s_signed'] - cert[lo]['s_signed']
        res = cert[hi]['band_width'] + cert[lo]['band_width']
        out['expert_P3'] = {'spread': spread, 'argmax': hi, 'argmin': lo, 'resolution': res,
                            'verdict': ('indeterminate' if abs(spread - PREDICTIONS['expert_P3']['bound']) <= res else
                                        'held' if spread <= PREDICTIONS['expert_P3']['bound'] else 'missed'),
                            'pre_declared_likely_indeterminate': True}
    else:
        out['expert_P3'] = {'verdict': 'not_scoreable_uncertified'}
    proj = PREDICTIONS['advisor_projections_labelled_projections']
    out['advisor_projections'] = {c: {'k_star': r.get('k_star'), 'projected': proj['k_star'].get(c),
                                      'inside': (r.get('k_star') is not None
                                                 and proj['k_star'][c][0] <= r['k_star'] <= proj['k_star'][c][1]),
                                      'earliest_possible': proj['earliest_possible_from_records'].get(c)}
                                  for c, r in reports.items()}
    hb = {c: {'branch': r.get('branch'), 'monotone': r.get('branch') == 'monotone',
              'boyd_lapses_after_N': r.get('boyd_lapses_after_N'),
              'pf_primal_after_N_max': (r.get('pf_primal_ratio_after_N') or {}).get('max'),
              'Q_end_minus_Q_N': r.get('s_signed'),
              'in_plateau_band_38k_53k': (r.get('s_signed') is not None and 38000.0 <= r['s_signed'] <= 53000.0)}
          for c, r in reports.items()}
    out['advisor_H_b'] = {'per_cell': hb,
                          'only_monotone_fired': all(v['monotone'] for c, v in hb.items()
                                                     if reports[c].get('status') == 'certified'),
                          's_x0_negative': (x0.get('s_signed') is not None and x0['s_signed'] < 0)}
    return out


# ======================================================================================================================
#  post-run evaluator self-tests (rule eleven for the evaluators, BEFORE any run)
# ======================================================================================================================
def _synthetic_run_dir(tmp, cell, p_max, tamper=None):
    """A synthetic eval dir produced by the REAL wrappers driven through production's call order with the RECORDED
    production values of the cell (checks H5) and a synthetic continuation that the rule certifies; plus a synthetic
    per_cycle_record, recourse_blocks_all (stand-in block function), lambda sidecar (the real capture on a fake dual
    structure of the SRP1 shape) and record summaries."""
    from types import SimpleNamespace
    decl = C.declaration_for(cell, p_max)
    n = decl['hold_after_cycle']
    cap = n + 100
    reference = C.load_replay_reference(decl)
    raw = {r['cycle']: r for r in json.load(open(_abs(os.path.join(
        C.RECERT_ROOT, 'evals', C.CELLS[cell]['eval_dir'], 'g_s39_D.json'))))['cycle_trajectory']}
    scripts = {'pen': [], 'rc': [], 'efc': []}
    orig, _calls = K._standin_originals(scripts)
    sink = []
    st = C.ContinuationState(decl, None, cap, reference=reference, sink=sink)
    blocks_by_cycle = {}

    def blocks_fn(pp, models):
        net = blocks_by_cycle[models['cycle']]
        out = {('TSO', None, y, d): net / 48.0 for y in (2025, 2030, 2035) for d in ('Spring', 'Summer', 'Autumn', 'Winter')}
        out.update({('DSO', nd, y, d): net / 48.0 for nd in (5, 7, 9) for y in (2025, 2030, 2035)
                    for d in ('Spring', 'Summer', 'Autumn', 'Winter')})
        out[('SALVAGE', None, None, None)] = -0.0
        return out
    fake = SimpleNamespace(_get_operational_recourse_block_components=blocks_fn,
                           _get_operational_objective_component_blocks=lambda pp, m: {})
    w = C.make_wrappers(st, orig, srp_module=fake)
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    w['_capture_convergence_depth_tail_baseline'](object(), admm)
    q_n = raw[n]['gross_operational_cost']
    direction = 1.0 if q_n >= raw[n - 1]['gross_operational_cost'] else -1.0
    rows = []
    consecutive = 0
    for c in range(1, cap + 1):
        if c <= n:
            bm, rc, pen, efc = K._row_objects(raw[c])
            row = {f: reference[c].get(f) for f in reference[c]}
        else:
            j = c - n
            qv = q_n + direction * 2000.0 * (1.0 - 0.9 ** j)
            bm = {'all_boyd_pass': True, **{g: {'primal_ratio': 0.5, 'dual_ratio': 0.5, 'channel_pass': True,
                                                'r': 1.0, 's': 1.0} for g in C.CHANNELS}}
            rc = {'gross_operational_cost': qv, 'net_operational_recourse': qv, 'terminal_salvage_value': 0.0}
            after = K._row_objects(raw[n])[2][2]
            pen = ({g: 'held (frozen after 10 unchanged cycles)' for g in C.CHANNELS}, dict(after), dict(after),
                   {g: 0.0 for g in C.CHANNELS}, {g: 0.0 for g in C.CHANNELS}, True,
                   {g: {'frozen': True} for g in C.CHANNELS})
            efc = None
            consecutive_next = consecutive + 1
            row = {'cycle': c, 'local_solves_ok': True, 'recourse': qv, 'gross_operational_cost': qv,
                   'terminal_salvage_value': 0.0, 'objective_change_abs': None, 'objective_tolerance': None,
                   'objective_change_ratio': None, 'cycle_convergence': True,
                   'consecutive_converged_cycles': consecutive_next, 'boyd_all_pass': True, 'boyd_stop': True,
                   **{f'boyd_{g}_{k}_ratio': 0.5 for g in C.CHANNELS for k in ('primal', 'dual')},
                   **{f'boyd_{g}_channel_pass': True for g in C.CHANNELS},
                   'rho_v_after': after['v'], 'rho_pf_after': after['pf'], 'rho_ess_after': after['ess'],
                   'rho_v_action': 'held (frozen after 10 unchanged cycles)',
                   'rho_pf_action': 'held (frozen after 10 unchanged cycles)',
                   'rho_ess_action': 'held (frozen after 10 unchanged cycles)', 'rho_freeze_active': True,
                   'efc_per_day_max': None}
        consecutive = row['consecutive_converged_cycles']
        blocks_by_cycle[c] = rc['net_operational_recourse']
        scripts['pen'].append(pen)
        scripts['rc'].append(rc)
        scripts['efc'].append(efc)
        with contextlib.redirect_stdout(io.StringIO()):
            K._drive(w, c, boyd_metrics=bm, local_ok=True, next_conv=None, active=c > n, admm=admm)
        rows.append(row)
        if admm.minimum_consecutive_converged_cycles == C.SETTLING_CERTIFIED_THRESHOLD:
            break
    w['_apply_convergence_depth_tail'](object(), admm, False, object(), None)
    files = defaultdict(list)
    for fname, obj in sink:
        files[fname].append(obj)
    if tamper == 'hold_flag':
        files[C.CYCLE_FILE][40]['holds']['aa'] = True
    if tamper == 'decision':
        files[C.DECISION_FILE][0]['k_star'] = files[C.DECISION_FILE][0]['k_star'] - 1
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == C.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    if tamper == 'replay_row':
        rows[10] = dict(rows[10], objective_change_abs=(rows[10]['objective_change_abs'] or 0.0) + 1.0)
    with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
        for r in rows:
            handle.write(GRIO.dumps(r, default=GRIO.json_default) + '\n')
    # lambda sidecar: the real capture over a fake dual structure of the SRP1 shape (36 blocks x 24)
    pp = SimpleNamespace(active_distribution_network_nodes=[5, 7, 9], years={2025: 5, 2030: 5, 2035: 5},
                         days={'Spring': 92, 'Summer': 91, 'Autumn': 91, 'Winter': 91},
                         transmission_network=SimpleNamespace(network={y: {d: SimpleNamespace(baseMVA=100.0)
                                                                            for d in ('Spring', 'Summer', 'Autumn', 'Winter')}
                                                                       for y in (2025, 2030, 2035)}),
                         distribution_networks={nd: SimpleNamespace(network={y: {d: SimpleNamespace(
                             baseMVA=100.0, get_interface_branch_rating=lambda: 200.0)
                             for d in ('Spring', 'Summer', 'Autumn', 'Winter')} for y in (2025, 2030, 2035)})
                             for nd in (5, 7, 9)})
    dv = {'pf': {a: {'current': {nd: {y: {d: {'p': [0.1 * i for i in range(24)], 'q': [0.0] * 24}
                                          for d in ('Spring', 'Summer', 'Autumn', 'Winter')}
                                      for y in (2025, 2030, 2035)} for nd in (5, 7, 9)}} for a in ('tso', 'dso')}}
    param = SimpleNamespace(mutable=False)
    tso_m = {y: {d: SimpleNamespace(admm_objective_scale=param) for d in ('Spring', 'Summer', 'Autumn', 'Winter')}
             for y in (2025, 2030, 2035)}
    dso_m = {nd: tso_m for nd in (5, 7, 9)}
    cap_idc = IDC.InterfaceDualCapture(tmp)
    cap_idc_orig_meta = IDC.read_metadata   # the fake models carry no pyomo Params: metadata stubbed for this test only

    def meta(pp_, t, d_, a, b):
        return {'sigma_fixed_admm_parameters_objective_scale': 1.0,
                'per_block': [{'admm_objective_scale_tso_mutable': False, 'admm_objective_scale_dso_mutable': False}
                              for _ in b]}
    IDC.read_metadata = meta
    try:
        for _c in range(len(rows)):
            cap_idc.on_boyd(pp, tso_m, dso_m, dv, SimpleNamespace(objective_scale=1.0))
    finally:
        IDC.read_metadata = cap_idc_orig_meta
    rec = {'cycles_run': len(rows), 'settling_continuation_summary': st.summary(),
           'interface_dual_capture': cap_idc.summary(), 'status': 'certified', 'post_certification': {}}
    return rec, rows


def post_run_evaluator_self_tests(p_max):
    out = {}
    for cell in C.CELL_ORDER:
        for name, tamper, expect in (('pass', None, True), ('tampered_hold_flag', 'hold_flag', False),
                                     ('tampered_decision_k_star', 'decision', False),
                                     ('tampered_replay_row_objective_change', 'replay_row', False)):
            tmp = tempfile.mkdtemp(prefix='w101_selftest_')
            try:
                rec, rows = _synthetic_run_dir(tmp, cell, p_max, tamper=tamper)
                h_ok, h_d = hold_checks(cell, tmp, rec)
                b_ok, b_d = block_capture(tmp, rec, rows)
                s_ok, s_d = stopping_check(cell, rec, tmp)
                l_ok, l_d = lambda_sidecar_check(tmp, rec)
                k_ok, k_d = settling_replay_check(cell, tmp, rec, p_max)
                f_ok, f_d = line_fields_check(tmp)
                r = replay_gate_full(cell, tmp, rec)
                allg = {'hold': h_ok, 'blocks': b_ok, 'stopping': s_ok, 'lambda': l_ok, 'settling_replay': k_ok,
                        'line_fields': f_ok, 'replay_full': r['bitwise_through_N']}
                if expect:
                    ok = all(allg.values())
                elif tamper == 'hold_flag':
                    ok = (not h_ok) and all(v for k_, v in allg.items() if k_ != 'hold')
                elif tamper == 'decision':
                    ok = (not k_ok) and (not s_ok)
                else:
                    ok = (not r['bitwise_through_N']) and r['first_divergence_cycle'] == 11
                rep = None
                if expect:
                    q = {x['cycle']: x['gross_operational_cost'] for x in rows}
                    dec = json.load(open(os.path.join(tmp, C.DECISION_FILE)))
                    rep = C.settling_report(dec, q, C.CELLS[cell]['N'])
                out[f'{cell}:{name}'] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                                         'decision': (json.load(open(os.path.join(tmp, C.DECISION_FILE))).get('k_star')
                                                      if os.path.isfile(os.path.join(tmp, C.DECISION_FILE)) else None),
                                         'report_on_synthetic': rep}
            except Exception as error:  # noqa: BLE001
                out[f'{cell}:{name}'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                         'traceback': traceback.format_exc()}
            finally:
                shutil.rmtree(tmp)
    # score_predictions on synthetic reports: a held, a missed, an indeterminate, an uncertified case
    syn = {'x0': {'status': 'certified', 's_signed': -12000.0, 'band_width': 3000.0, 'k_star': 165, 'branch': 'oscillatory'},
           'n7_4h_e1': {'status': 'certified', 's_signed': -9000.0, 'band_width': 2500.0, 'k_star': 160,
                        'branch': 'monotone'},
           'c_star': {'status': 'uncertified', 's_signed': 41000.0, 'band_width': 6000.0, 'k_star': None}}
    sc = score_predictions(syn)
    ok_sc = (sc['expert_P1']['x0']['verdict'] == 'held' and sc['expert_P1']['c_star']['verdict']
             == 'not_scoreable_uncertified' and sc['expert_P2']['dV'] == -3000.0
             and sc['expert_P2']['verdict'] == 'held' and sc['expert_P2']['reopen_rule_triggered'] is False
             and sc['expert_P3']['verdict'] == 'not_scoreable_uncertified'
             and sc['advisor_projections']['x0']['inside'] is True and sc['advisor_projections']['n7_4h_e1']['inside']
             is False)
    syn2 = {c: dict(v, status='certified', s_signed=v['s_signed'] if c != 'c_star' else -8000.0, band_width=2000.0)
            for c, v in syn.items()}
    sc2 = score_predictions(syn2)
    ok_sc2 = sc2['expert_P3']['verdict'] == 'indeterminate' and sc2['expert_P3']['spread'] == 4000.0
    out['score_predictions_synthetic'] = {'ok': bool(ok_sc and ok_sc2), 'case1': sc, 'case2_P3': sc2['expert_P3']}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  the stage spec v39
# ======================================================================================================================
def launch_command(cell, spec_sha=None):
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run --cell {cell} '
            f'--spec-sha256 {spec_sha or "<campaign spec sha256 of " + cell + ">"} '
            f'> {os.path.join(ROOT_REL, f"run_{cell}_v39_launch.log")} 2>&1')


def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST), 'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'), 'p_max': doc.get('p_max'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256')}


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks()


def stage_spec_content(p_max, p_detail, checks_inline, checks_file, post_tests, verb, solver, pres, code_since,
                       i_check, mem):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    cells = {}
    for cell in C.CELL_ORDER:
        e = recert_entry(cell)
        cells[cell] = {'recert_eval_key': e['eval_key'], 'candidate_key': e['key'], 'canonical': e['canonical'],
                       'N': C.CELLS[cell]['N'], 'cap': C.CELLS[cell]['N'] + SC.CAP_AFTER_N,
                       'Q_N_recorded': _read_jsonl(_abs(C.reference_path(cell)))[-1]['gross_operational_cost'],
                       'replay_reference': C.declaration_for(cell, p_max)['replay_reference'],
                       'declaration': C.declaration_for(cell, p_max), 'campaign_id': CAMPAIGN_IDS[cell],
                       'campaign_root': campaign_root_rel(cell), 'keys': expected_keys(cell, p_max),
                       'pre_launch_assertion_at_freeze': pres[cell], 'launch_command_template': launch_command(cell)}
    return {
        'schema': 'p515_s53_stage_spec_v39', 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 38, **SPEC_V38},
        'note_on_version': ('Addendum 53 says "under spec v38"; v38 (8bc0ffa6) is the W98 3 x 3 stage-1 spec, so this is '
                            'v39 with v38 as its predecessor (TASKS.md)'),
        'authority': [f'{BRIEF} Addendum 53 (committed 4a80c3e2)', 'TASKS.md Addendum 53 order (Advisor algorithm adopted)',
                      'Planner task W101', 'P5_15_ADDENDUM52_SETTLING_REVIEW_NOTE.md option 1(a)'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED; net_operational_recourse and the '
                                 'terminal salvage reported beside it; every value in this spec and in the runs\' '
                                 'results is on this convention'),
        'code_sha256': code, 'production_sha256_post_ed5ba42a': production,
        'pins_note': 'post-W100 (ed5ba42a) module hashes, incl. gate_result_io.py and the W101 modules',
        'code_since_recert_run': code_since,
        'recert': RECERT, 'cells': cells, 'cell_order': list(C.CELL_ORDER),
        'configuration': {'identical_to_recert': configuration(), 'concurrency': CONCURRENCY,
                          'option_b_release_solution_bookkeeping': 'absent, as in the recert (never keyed)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as the recert)',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze and at --run'},
        'stop_rule': {
            'module': 'settling_criterion', 'class': 'settling_criterion.SettlingRule',
            'constants': SC.constants(p_max), 'readings_adopted_by_the_planner': SC.READINGS,
            'algorithm': SC.__doc__, 'failing_reasons': list(SC.FAILING_REASONS),
            'boyd_k': ('boyd_metrics[\'all_boyd_pass\'] AND local_solves_ok, from THIS cycle\'s boyd_metrics as the AA '
                       'wrapper receives it (production updates consecutive_converged_cycles after the recourse wrapper)'),
            'Q_k': 'gross_operational_cost from the recourse wrapper; None when any local solve failed',
            'mechanism': ('decision in the recourse wrapper (before production\'s convergence test); certification sets the '
                          'certificate length to 0 -> production\'s own exit test ends the loop at the end of THIS cycle '
                          '(W98\'s mechanism); the decision record is written (settling_decision.json)'),
            'only_exit_before_cap': True, 'early_stop': 'ABSENT (asserted: validator refuses the key; checklist (c))',
            'literal_two_sign_change_reading': 'settling_criterion.literal_two_sign_change_report -- REPORT-ONLY',
            'v38_settling_analysis': 'not used (no fit enters any decision)',
            'p_max': {'value': p_max, 'function': 'settling_criterion.p_max_from_records',
                      'records': [C.reference_path(c) for c in C.CELL_ORDER], 'cycles': '1..N of each',
                      'advisor_hand_derived': K.ADVISOR_P_MAX, 'advisor_without_floor': 25,
                      'this_function_without_floor': checks_inline.get('P_MAX', {}).get('without_floor_this_function'),
                      'detail': p_detail},
        },
        'replay_gate': {
            'in_cycle': ('every cycle k <= N, at the end of the cycle (after production\'s EFC read): the fields '
                         + ', '.join(C.REPLAY_GATED_FIELDS) + ' compared as JSON text with the recert\'s committed row k; '
                         'the FIRST difference writes the cycle line with the cycle and magnitude (fields, gross '
                         'difference, largest relative difference) and ABORTS the cell (no relabelled continuation)'),
            'post_run': ('every field of per_cycle_record.jsonl rows 1..N bitwise (G19), adding '
                         + ', '.join(C.REPLAY_POST_RUN_ONLY_FIELDS)),
            'reference_hashes': {c: C.CELLS[c]['per_cycle_record_sha256'] for c in C.CELL_ORDER},
            'expected': ('bitwise: production unchanged since the recert\'s run commit 31a37efb (code_since_recert_run); '
                         'the harness changes are writer-only / keyed options'),
        },
        'holds_after_N': {'AA': 'off -- production\'s own \'off\' branch via all_boyd_pass forced True on a copy, every '
                                'cycle > N, even when Boyd lapses (AA is non-latching in production)',
                          'tail': 'on (applied True at the top of every cycle > N; next-state True)',
                          'rho': 'frozen (allow_update False; before == after asserted)',
                          'boyd_lapse_behaviour': ('a lapse (or failed cycle) resets k0 strictly in the rule and is '
                                                   'recorded; the holds continue; only the rule or the cap ends the run')},
        'records': {
            'continuation_line_fields_i': list(LINE_FIELDS_REQUIRED) + [f'settling.{f}' for f in SETTLING_FIELDS_REQUIRED]
            + ['boyd_ratios (all six boyd_*_ratio; pf_primal first-class)', 'blocks_captured'],
            'per_block_dQ': 'recourse_blocks_all.jsonl: 48 network blocks + SALVAGE, with deltas, every cycle (as W98)',
            'lambda_t': ('interface_duals_per_cycle.jsonl (every evaluation, harness default): lambda_pf_{p,q}_{tso,dso} '
                         'per (node, year, day) x 24 periods, read after the cycle\'s dual updates (before AA), header '
                         'with the scaling metadata (rating, base MVA, admm_objective_scale, sigma); not converted'),
            'production_counters': 'consecutive_converged_cycles and boyd_all_pass stay in per_cycle_record.jsonl (j)'},
        'capture_checklist_asserted_before_any_solve': {
            'a': 'boyd_metrics[\'all_boyd_pass\'] exists; the AA wrapper installed with AA enabled',
            'b': 'source order _anderson_acceleration_cycle_step( < _get_operational_recourse_components( < convergence test',
            'c': 'no early_stop in the declaration; certificate length written only by disable / restore / settling',
            'd': 'CAP == N + 100', 'e': 'the replay reference hashes and holds exactly 1..N',
            'f': 'every constant with its formula and the P_MAX function name', 'g': 'tests 1-12 + the 2(b) replay (checks)',
            'h': 'the lambda_t capture path', 'i': 'the per-cycle line fields (G18, post-run)',
            'j': 'production counters in per_cycle_record (G20)', 'k': 'post-run replay of the pure function (G17)'},
        'zero_solve_checks': {'script': 'p515_s53_w101_continuation_checks.py', 'committed_output': checks_file,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()
                                                  if isinstance(v, dict) and 'holds' in v}},
                              'rerun_before_launch': 'the --run mode re-runs every check and refuses unless all hold'},
        'gates': {
            'G1-G5, G7, G9, G11': 'as the recert / W98 (W86 evaluation_checks; solve profile 51 x (cycles + 1) + retries)',
            'G6': 'v37 (Optimal Solution Found + four metrics within the tail tolerances, final accepted attempt; B = 48)',
            'G8': 'persistence as the recert (persist iff the harness\'s residual certificate holds on the last row)',
            'G13': 'holds inert through N, held after N; certificate length 10**9 except 0 on the settling cycle',
            'G14': 'all-block capture', 'G15': 'stopping consistent (settling k* == cycles_run, or the cap uncertified)',
            'G16': 'lambda_t sidecar complete (header + one captured line per cycle, 36 x 24)',
            'G17': 'post-run replay of the pure rule reproduces the in-cycle decision and every per-cycle record; never '
                   'certifies at k <= N', 'G18': 'the per-cycle line fields (i)', 'G19': 'replay 1..N bitwise every field',
            'G20': 'production counters in the per-cycle record', 'applies_to': 'every cell (one entry per campaign)',
            'post_run_evaluator_self_tests': post_tests},
        'report_definitions': REPORT_DEFINITIONS,
        'predictions_recorded_before_any_run': PREDICTIONS,
        'values': {'V_SRP1': V_SRP1, 'R_ref_rounded': SC.R_REF, 'I': i_check},
        'verbatim_text': {'quotes': VERBATIM, 'check': verb},
        'labelling_and_identity': {
            'label': C.LABEL,
            'evaluation_key': ('sha256({base_evaluation_key, settling_continuation}); every key without the declaration '
                               'byte-identical to the pre-W101 harness (checks K: all committed entries)')},
        'memory_preflight': {'rule': 'W86 rule at concurrency 1: available >= 3.5 GiB (per-child budget incl. persist)',
                             'measured_at_freeze_non_gating': mem},
        'solver': solver,
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': ('RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted; '
                                             'cycles_run in [N + 1, N + 100]')},
        'expected_wall_time': wall_time_estimate(),
        'launch_commands_templates': {c: launch_command(c) for c in C.CELL_ORDER},
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised zero-solve by checks H5 / H6 (real wrappers, real install, recorded values); '
                             'THE FIRST REAL CYCLE OF THE x0 LAUNCH IS THE SMOKE (its in-cycle replay gate at cycle 1)'),
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
    tag = f'W101-V{SPEC_VERSION}'
    failures = _common_checks()
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'v{SPEC_VERSION} already exists (write-once): {existing}')
    cf = _checks_file_state()
    if not (cf['all_hold'] is True and cf['committed_clean']
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in ('p515_s53_w101_continuation_checks.py', 'settling_criterion.py', 'interface_dual_capture.py',
                'p515_s53_w101_settling_continuation_hooks.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
                'shared_resources_planning.py', 'admm_anderson_acceleration.py'):
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    code_since = code_since_recert()
    if not code_since['ok']:
        failures.append(f'production changed since the recert run: {code_since}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append('inline re-run of the zero-solve checks fails')
    p_max = checks_inline.get('p_max')
    p_max2, p_detail = K.compute_p_max()
    if p_max != p_max2 or p_max != cf.get('p_max'):
        failures.append(f'P_MAX mismatch: inline {p_max} recomputed {p_max2} committed {cf.get("p_max")}')
    post_tests, post_ok = post_run_evaluator_self_tests(p_max)
    if not post_ok:
        failures.append(f'post-run evaluator self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found in the brief: {verb["found_whitespace_normalised"]}')
    solver = solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    pres = {cell: pre_launch_assertion(cell, p_max) for cell in C.CELL_ORDER}
    for cell, pre in pres.items():
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    try:
        i_check = investment_cost_srp1()
    except Exception as error:  # noqa: BLE001
        i_check = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        failures.append(f'I check failed: {i_check["error"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    mem = L.memory_preflight(1)
    content = stage_spec_content(p_max, p_detail, checks_inline, cf, post_tests, verb, solver, pres, code_since,
                                 i_check, mem)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError(f'v{SPEC_VERSION} written bytes do not hash to the name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] wrote {rel} sha256={sha} (predecessor v38 {SPEC_V38['sha256']})")
    _log(f"[{tag}] P_MAX {p_max} (settling_criterion.p_max_from_records; advisor 22); TAU {SC.TAU!r}; EPS0 {SC.EPS0!r}")
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items() if 'holds' in v} }")
    _log(f"[{tag}] post-run evaluator self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f"[{tag}] I: master {i_check.get('I_srp1_master_expression')!r} transcription "
         f"{i_check.get('I_srp1_p56a_transcription')!r} cited {I_CITED['value_eur']!r} bitwise "
         f"{i_check.get('equals_cited_bitwise')}")
    for cell in C.CELL_ORDER:
        _log(f"[{tag}] {cell}: base {pres[cell]['base_key_without_continuation'][:16]} == recert; continuation "
             f"{pres[cell]['continuation_key']}")
    w = content['expected_wall_time']
    _log(f"[{tag}] expected wall worst case (all caps) {w['campaign_worst_case_all_caps_h']:.2f} h; per cell "
         f"{ {c: round(v['worst_case_cap_h'], 2) for c, v in w['per_cell'].items()} }")
    _log(f"[{tag}] memory at freeze (non-gating): available {mem.get('available_gib')} GiB")
    _finish(0, '-- next: --freeze')


# ======================================================================================================================
#  campaign freeze, run, summarize
# ======================================================================================================================
def freeze(started):
    tag = 'W101-FREEZE'
    failures = _common_checks()
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        if not _committed_clean(ss_rel):
            failures.append(f'the stage spec v{SPEC_VERSION} is not committed / clean')
        if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
            failures.append(f'code changed since v{SPEC_VERSION} froze')
        p_max = ss['stop_rule']['p_max']['value']
    except RuntimeError as error:
        failures.append(str(error))
        ss, p_max = None, None
    for cell in C.CELL_ORDER:
        failures += H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)
        if p_max is not None:
            pre = pre_launch_assertion(cell, p_max)
            if not pre['holds']:
                failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    all_ok = True
    for cell in C.CELL_ORDER:
        pre = pre_launch_assertion(cell, p_max)
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ss_pin, 'label': C.LABEL, 'cell': cell, 'recert': RECERT,
                 'recert_eval_key': recert_entry(cell)['eval_key'], 'expected_eval_key': pre['continuation_key'],
                 'N': C.CELLS[cell]['N'], 'objective_convention': ss['objective_convention'],
                 'solve_claim': ss['solve_profile_declared'], 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(C.CELL_ORDER)}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell, p_max), configuration=configuration(),
            cap=C.CELLS[cell]['N'] + SC.CAP_AFTER_N, concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 53', 'Planner task W101', ss_rel], required_consecutive_cycles=10,
            extra=extra)
        checks = validate_campaign_spec(cell, spec, ss_pin, p_max)
        pre_frozen = pre_launch_assertion(cell, p_max, spec)
        ok = all(checks.values()) and pre_frozen['holds']
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
        _log(f"[{tag}]   eval_key={e['eval_key']} eval_dir={e['eval_dir']} cap={spec['cap']} "
             f"post_certification={e['post_certification']}")
        _log(f"[{tag}]   spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}; "
             f"pre-launch on the frozen spec holds={pre_frozen['holds']}")
        _log(f'[{tag}]   LAUNCH: {launch_command(cell, spec_sha)}')
    _finish(0 if all_ok else 1, f'freeze {"OK" if all_ok else "NOT OK"}')


def _campaign_spec_path(cell):
    root = campaign_root(cell)
    hits = sorted(f for f in os.listdir(root) if f.startswith('campaign_spec_')) if os.path.isdir(root) else []
    return os.path.join(root, hits[0]) if len(hits) == 1 else None


def run(started, cell, spec_sha256):
    tag = f'W101-RUN-{cell}'
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    p_max = ss['stop_rule']['p_max']['value']
    if not _committed_clean(ss_rel):
        failures.append(f'stage spec v{SPEC_VERSION} not committed / clean')
    if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
        failures.append(f'code changed since v{SPEC_VERSION} froze')
    idx = C.CELL_ORDER.index(cell)
    for prev in C.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'cell order x0 -> n7_4h_e1 -> c_star: {prev} has no results yet')
    root = campaign_root(cell)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign spec not committed / clean')
    checks = validate_campaign_spec(cell, spec, {'path': ss_rel, 'sha256': ss_sha}, p_max)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold'] or checks_inline.get('p_max') != p_max:
        failures.append('the zero-solve checks do not all hold now (or P_MAX moved)')
    pre = pre_launch_assertion(cell, p_max, spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["parts"]}')
    solver = solver_check()
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
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cell {cell} eval_key {entry['eval_key']}; N {C.CELLS[cell]['N']}; cap {spec['cap']}; "
         f"lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    try:
        gates, detail, rec = cell_gates(cell, entry, eval_dir, p_max)
    except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
        gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}, {}
    report = None
    try:
        if os.path.isfile(os.path.join(eval_dir, C.DECISION_FILE)) and os.path.isfile(
                os.path.join(eval_dir, 'per_cycle_record.jsonl')):
            report = cell_report(cell, eval_dir, rec)
    except Exception as error:  # noqa: BLE001
        report = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    summ = (rec or {}).get('settling_continuation_summary') or {}
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'], 'recert_eval_key': recert_entry(cell)['eval_key'],
               'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'gate_detail': detail, 'settling_report': report,
               'pre_launch_assertion': pre, 'memory_preflight_at_run': mem, 'solver': solver, 'batch_info': batch,
               'parent_view': (records[0] or {}).get('parent_view'), 'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    if summ.get('replay_first_divergence'):
        _log(f"[{tag}] REPLAY DIVERGED -- cell ABORTED: {summ['replay_first_divergence']}")
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] settling: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"s {report.get('s_signed')} band_width {report.get('band_width')} range/tau {report.get('range_over_tau')}")
    code = 0 if (all(gates.values()) and _guards_ok(g) and isinstance(report, dict) and 'error' not in report) else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


def summarize(started):
    tag = 'W101-SUMMARY'
    ss_rel, ss_sha, _ss = load_stage_spec()
    reports, missing = {}, []
    for cell in C.CELL_ORDER:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        if not os.path.isfile(path) or not _committed_clean(os.path.relpath(path, REPO)):
            missing.append(cell)
            continue
        reports[cell] = (json.load(open(path)).get('settling_report') or {})
    out_rel = os.path.join(ROOT_REL, SUMMARY_FILE)
    if missing or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] cells without committed results {missing} or summary exists')
        _finish(1)
    scored = score_predictions(reports)
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'stage_spec': {'path': ss_rel, 'sha256': ss_sha},
           'objective_convention': REPORT_DEFINITIONS['objective_convention'], 'reports': reports,
           'predictions_scored': scored, 'definitions': REPORT_DEFINITIONS, 'predictions': PREDICTIONS}
    H._write_once_json(_abs(out_rel), doc)
    H._write_once_json(_abs(os.path.join(ROOT_REL, SUMMARY_MANIFEST)), {out_rel: _sha(out_rel)})
    _log(f'[{tag}] {json.dumps(scored, default=str)[:4000]}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=C.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.freeze_spec:
            freeze_spec(started)
        elif args.freeze:
            freeze(started)
        elif args.summarize:
            summarize(started)
        else:
            if not (args.spec_sha256 and args.cell):
                parser.error('--run requires --cell and --spec-sha256')
            run(started, args.cell, args.spec_sha256)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
