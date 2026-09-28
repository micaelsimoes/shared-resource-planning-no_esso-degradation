"""
P5.15 Addendum 57 (Decisions 2 and 3) and Addendum 54 Ruling 2, Planner task W118 -- the SRP1 RE-SETTLING CAMPAIGN (ten
cells): the frozen stage spec `frozen_s53_resettle_spec_v1_<sha8>.json` (a new spec series; the stage series it builds on
is v41 fcea4b38 / v39 8a612429, recorded as its basis), the ten per-cell campaign freezes, the per-cell run, and the
zero-solve summary with the frozen difference formulas. BUILT AND FROZEN IN W118; NO RUN IS LAUNCHED BY THE WORKER.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 57 (Decision 2: the gap clause |t_sum| <= tau/2, the amended monotone
branch |last step| x L <= tau with L = 60, P_MAX and L instance-measured; Decision 3: the 2030 / 2035 year-ladder cells as
first C2 evaluations with a lattice-legality check, the same holds as the references after the first residual pass,
the F2 challenger first with the uncertified form frozen); Addendum 54 Ruling 2 (re-run, not pin; gate bitwise through the
first tail cycle; differences against the settled references); P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md Decision 3 and
"Design otherwise"; TASKS.md (the Advisor's eight-cell design review); Planner task W118.

THE CELLS (`p515_s53_w118_resettle_hooks.CELLS`, launch order CELL_ORDER):
  f2_challenger (e28de4ac, s53_f2_certificate_r1) -> f2_incumbent (5ca4f86c, s51_f2_phase_b)  -- m = 2, gated
  pb_y2030_n9 (a30a9faf), pb_y2030_n7 (d7030f59), pb_y2025_n5 (4a852725), pb_y2030_n5 (d0c1f160),
  pb_y2025_n9 (1bff3ed2), pb_y2025_n7 (10c73abd)                                              -- Phase B, m = 1, gated
  yl_y2030 (549476cd), yl_y2035 (dab6a8a2), both s45_a1b                                      -- first C2 evaluations
Each runs the CURRENT production configuration: the case file (AA keep_memory declared), the ESS ageing baseline C2
declared, the tight tail {True, 1e-6} declared (production's rule: from AA-off + 1), persist_certified_models as the W101
cells, the F2 cells at flexibility price x 2, plus the keyed entry option `settling_resettle`. One campaign (root, spec,
lock) per cell; concurrency 1.

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-spec                         ZERO SOLVES. The stage spec (write-once, named by its sha256).
  --freeze                              ZERO SOLVES. The ten campaign specs; the exact launch commands.
  --run --cell C --spec-sha256 S        THE RUN OF ONE CELL (NOT RUN IN W118). Order enforced. Preconditions (the
                                        zero-solve checks re-run inline, the pre-launch assertion, the parent-side
                                        capture checklist, the memory preflight, the solver path, the run-lock), then
                                        H.evaluate on the one entry, the gates, the cell report; results + manifest.
  --run ... --preconditions-only        ZERO SOLVES. Every --run precondition, then STOP before the lock and the child.
  --summarize                           ZERO SOLVES. The ten cell reports, the frozen differences, the predictions.

Exit codes: 0 done / every gate holds; 1 a gate / harness / guard / precondition failure (a replay divergence aborts the
cell: the child exits 1 with its barrier record; the launcher reports the cycle and magnitude and exits 1).
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import statistics
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W118 re-settling launcher (never solves)').install()

import p515_s53_w101_srp1_continuation_campaign as W101L  # noqa: E402 -- generic gates (arms its guards)
import p515_s53_w118_resettle_checks as K  # noqa: E402 -- the zero-solve checks (arms its guards)
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L = W101L.H, W101L.L, W101L.X, W101L.W9, W101L.W98L
W112 = K.W112


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(W101L.GUARDS) + (('w118_parent', PARENT_GUARD),))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w118_resettle_campaign', 'p515_s53_w105_c_star_extension_campaign',
                          'p515_s53_w101_srp1_continuation_campaign', 'p515_s53_w98_continuation_campaign')
STAGE_TEXT = ('P5.15 Addendum 57, W118 -- SRP1 re-settling campaign (10 cells) under the current production '
              'configuration: gated cells replayed bitwise against their original record through the first residual '
              'pass k0 (abort on the first divergence), the year-ladder cells as first C2 evaluations; the certifying '
              'regime held after the run\'s k0 (AA off, tight tail on, rho frozen); settling rule v2 (gap clause '
              '|t_sum| <= tau/2; monotone branch with |dQ| x 60 <= tau) until it certifies or the cap; W105 captures + '
              'per-cycle t_sum and Q_cc')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W118_ROOT_REL
SPEC_PREFIX = 'frozen_s53_resettle_spec_v1_'
SPEC_SERIES = 'frozen_s53_resettle_spec'
SPEC_VERSION = 1
BASIS_SPECS = {
    'stage_spec_v41': {'path': os.path.join(_P53, 'frozen_s53_spec_v41_fcea4b38.json'),
                       'sha256': 'fcea4b3837205b7134d308d200299f14f7cb1187ef43db5f1233091138004b99'},
    'stage_spec_v39': {'path': os.path.join(_P53, 'frozen_s53_spec_v39_8a612429.json'),
                       'sha256': '8a61242924cee1690c0d9f9f25a0ef6bdc90a519d089415898b44d5983e4c756'},
}
CAMPAIGN_IDS = {cell: f's53_w118_resettle_{cell}' for cell in R.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
SUMMARY_FILE = 'w118_resettle_summary.json'
SUMMARY_MANIFEST = 'w118_resettle_summary_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
SOLVER_PATH = '/usr/local/bin/ipopt'
PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
N_NETWORK_BLOCKS = 48
N_ESSO = 3
W101_X0_SPEC_REL = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                'campaign_spec_s53_w101_srp1_cont_x0_d1d5c3bc.json')
EXTRA_CLEAN_FILES = (SCRIPT_NAME, 'p515_s53_w118_resettle_hooks.py', 'p515_s53_w118_resettle_checks.py',
                     'settling_criterion_v2.py', 'settling_criterion.py', 'interface_dual_capture.py', 'gate_result_io.py',
                     'p515_s53_w105_settling_extension_hooks.py', 'p515_s53_w105_extension_checks.py',
                     'p515_s53_w101_srp1_continuation_campaign.py', 'p515_s53_w101_settling_continuation_hooks.py',
                     'p515_s53_w101_continuation_checks.py', 'p515_s53_w98_continuation_campaign.py',
                     'p515_s53_w98_continuation_hooks.py', 'p515_s53_w98_continuation_checks.py',
                     'p515_s53_w112_consensus_gap.py', 'p515_s53_w86_tail_recert_campaign.py', H.ESS_PARAMS_FILE_REL,
                     'shared_energy_storage_parameters.py', 'shared_energy_storage_data.py', 'network.py',
                     K.COST_FILE_REL)
CODE_PINNED = K.CODE_PINNED_BY_CHECKS + (SCRIPT_NAME, 'p515_s53_w101_srp1_continuation_campaign.py',
                                         'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
                                         'p515_s53_w86_tail_recert_campaign.py', 'p515_s53_w89_g6_final_attempt_reeval.py',
                                         'shared_energy_storage_data.py', 'network.py')
PRODUCTION_FILES = ('shared_resources_planning.py', 'network.py', 'admm_parameters.py', 'admm_anderson_acceleration.py',
                    'model_construction_helpers.py', 'shared_energy_storage_data.py')
P_MAX_SOURCES = {'x0': K.DECISIONS['x0'], 'unit': K.DECISIONS['unit']}

# ---- verbatim text (Addendum 57; checked against the committed brief, whitespace-normalised, at every freeze) --------
VERBATIM = {
    'gap_clause': '**Certification gains the clause |t_sum| ≤ τ/2** (2,270 €) on top of the settling rule',
    'monotone_amended': ('**Monotone branch amended** for the campaign spec: steps decreasing over the window **and** '
                         '|last step| × L ≤ τ, with L = 60 (2× the longest period measured on the instance)'),
    'p_max_instance_measured': 'P_MAX and L are instance-measured quantities recorded in each spec.',
    'year_ladder': ('(a) the 2030 and 2035 year-ladder cells are **first C2 evaluations**, no bitwise gate, both years so '
                    'the ladder is one configuration (+1 cell)'),
    'same_holds': '(b) **Same holds as the references** after the first residual pass (AA off, tail on, ρ frozen), for parity.',
    'f2_first': '(c) **F2 challenger first**, uncertified reporting form frozen in advance;',
}

# ---- the frozen definitions (per cell, the differences, the uncertified form) ----------------------------------------
DEFINITIONS = {
    'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED; t_sum = the priced interface-consensus gap '
                             'sum w * pi * (p_DSO - p_TSO) (EUR); Q_cc = Q + t_sum is a FIRST-ORDER, consensus-consistent '
                             'DIAGNOSTIC, reported beside gross, never a costed quantity; verdicts are read on GROSS'),
    'per_cell': {
        'k_star': 'the cycle the settling rule v2 certified (None when uncertified at its cap)',
        'branch': 'oscillatory | monotone',
        'band': '[min, max] Q over the certifying window (uncertified: over the last L = 60 cycles)',
        'band_width': 'max - min of the band', 'range_over_tau': 'range of the certifying window / tau',
        's': 'gated cells: s = Q_k* - Q_N_old (Q_N_old = the ORIGINAL record\'s last cycle; resolution band_width)',
        't_sum_k_star': 't_sum at k* (in-cycle, validated post-run against the stride and the terminal identity)',
        'Q_cc_k_star': 'Q_k* + t_sum_k*',
        'overlap': ('gated cells, REPORT-ONLY: Q_new(k) - Q_old(k) for k0+1..N_old; relative = (Q_new - Q_old) / Q_old; '
                    'at N_old scored against the recorded prediction'),
        'terminal_step_to_threshold': ('|dQ_last| / EPS0 and, for certified cells, range / tau at k*; production\'s '
                                       'rule ten (objective_change_abs / objective_tolerance) of the last row'),
    },
    'uncertified_form': ('frozen (settling_criterion_v2): band [min, max] over the last L = 60 cycles; drift rate = mean '
                         'dQ over the last 25 cycles; dQ_cc rate = mean (dQ + dt_sum) over the same cycles; t_sum and '
                         'Q_cc at the cap; reasons (including gap_clause); an uncertified cell\'s differences are '
                         'INDETERMINATE by construction'),
    'differences': {
        'phase_b_and_year_ladder_vs_settled_x0': (
            'M_j = I_j + Q_j(k*) - Q181, Q181 = 653,873,702.1876609 (settled x = 0, W102 cycle 181); resolution_j = '
            'band_j + band_x0 (band_x0 = 4,209.1713362932205); determinate iff |M_j| > resolution_j. Q_cc: M_j^cc = '
            'I_j + Q_j(k*) + t_j(k*) - (Q181 + t_x0), t_x0 = 142.52829384803772. I_j from the cited source '
            '(Phase B: phase_b_state.json I_x_eur; year ladder: W2 investment_cost_results.json I_new_eur)'),
        'f2_certificate': (
            'D = (I_c - I_i) + (Q_c(k*_c) - Q_i(k*_i)); resolution band_c + band_i; D_cc = D + (t_c(k*_c) - t_i(k*_i)). '
            'If either cell is uncertified D is INDETERMINATE by construction; the bands and rates are reported. '
            'Expert\'s note (Addendum 57 Decision 3(c)): the cited 6,338 EUR margin is < 2 tau, so "within resolution" '
            'is the expected verdict and it is acceptable'),
        'year_ladder_2035_minus_2030': (
            'D_yl = (I_2035 + Q_2035(k*)) - (I_2030 + Q_2030(k*)); resolution band_2030 + band_2035; D_yl^cc = D_yl + '
            '(t_2035 - t_2030); plus each year against x = 0 (M_j above)'),
        'verdict_words': 'determinate (|x| > resolution) | within resolution (|x| <= resolution) | indeterminate_uncertified',
    },
}

# ---- predictions, recorded BEFORE any run (each with its source) -------------------------------------------------------
PREDICTIONS = {
    'gate_outcome': {
        'statement': 'every gated cell replays BITWISE through k0 (the in-cycle gate never aborts; G19 holds)',
        'cells': list(R.GATED_CELLS),
        'source': ('Advisor design review (TASKS.md: "gate through k0 achievable (expected)"; F2 pair "gate expected but '
                   'never demonstrated on the multiplier path")')},
    'tail_overlap': {
        'statement': 'overlap relative (Q_new - Q_old) / Q_old at N_old in [-1.5e-6, -0.8e-6]', 'lo': -1.5e-6,
        'hi': -0.8e-6, 'source': 'Advisor design review (TASKS.md; Planner task W118)'},
    'phase_b': {
        'statement': ('oscillatory certification at k* in [k0 + 50, k0 + 70]; s_j in [+8, +30] kEUR; |s_j - 14,971| <= '
                      '10 kEUR; margins stay positive (M_j > 0: x = 0 stays better); |t_sum(k*)| < tau/2'),
        'k_star_after_k0': [50, 70], 's_range_eur': [8000.0, 30000.0], 's_x0_eur': 14970.61903166771,
        's_minus_s_x0_bound_eur': 10000.0,
        'source': ('Planner task W118 (attribution not stated in the task; searched: brief Addenda 53-57, the '
                   'consolidated note, TASKS.md, REVISION_CONTEXT.md -- recorded as given)')},
    'f2_challenger': {
        'statement': 'C*-like creep: UNCERTIFIED at its cap (N_old + 100 = 261), probability >= 0.5',
        'source': 'Advisor design review (competing prediction; TASKS.md "H_creep >= 0.5 probability")'},
    'f2_incumbent': {
        'statement': 'turns within ~15 cycles',
        'operationalisation': ('the first turning point of the rule after the run\'s k0 (T[0].t) is <= k0_run + 15 '
                               '(Worker operationalisation of "within ~15 cycles", for Planner confirmation)'),
        'bound_cycles': 15, 'source': 'Planner task W118 (as given)'},
    'year_ladder': {
        'statement': 'k* ~ k0 + 60',
        'operationalisation': 'k* in [k0_run + 50, k0_run + 70] (Worker operationalisation of "~", as Phase B)',
        'k_star_after_k0': [50, 70], 'source': 'Planner task W118 (as given)'},
    'walls': {'statement': 'per cell and total, the expected_wall_time block of this spec',
              'source': 'Worker (expected_wall_time basis)'},
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
    _log(f'[W118] guards {g} {extra_msg}')
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
#  configuration, entries, keys
# ======================================================================================================================
def configuration(cell):
    cfg = K.configuration_for(cell)
    return {'name': ('W118 SRP1 RE-SETTLING (Addendum 57) -- the current production configuration: the case file (AA '
                     'keep_memory declared), the ESS ageing baseline ' + cfg['ess_ageing_baseline_label'] + ' declared, '
                     'the convergence-depth tight tail DECLARED ENABLED (compl_inf_tol 1e-6, from AA-off + 1), '
                     'post-certification: persist the certified TSO/DSO models only'),
            'arm_label': 's39_D', 'overrides': {},
            'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': cfg['convergence_depth_tail'],
            'note': ('the entry adds settling_resettle (keyed); cap: gated N_old + 100, ungated 300 (the rule stops at '
                     'min(k0_run + 109, 300)); persist_certified_models as the W101 cells; no option (b); concurrency 1'
                     + ('; flexibility price x 2 per entry (F2)' if R.CELLS[cell]['group'] == 'f2' else ''))}


def orig_entry(cell):
    return K._spec_entry(cell)[1]


def entries(cell):
    e = orig_entry(cell)
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    opts = {'investment_year': e['canonical']['investment_year'],
            'post_certification': {'persist_certified_models': True, 'hull_polish': False, 'reference': None},
            'settling_resettle': R.declaration_for(cell)}
    if R.CELLS[cell]['flex_price_multiplier'] is not None:
        opts['flex_price_multiplier'] = R.CELLS[cell]['flex_price_multiplier']
    return [(cell, nodes, opts)]


def expected_keys(cell):
    spec, e = K._spec_entry(cell)
    ocfg = spec['configuration']
    cfg = K.configuration_for(cell)
    flex = e.get('flex_price_multiplier')
    kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
              flex_price_multiplier=flex, convergence_depth_tail=cfg['convergence_depth_tail'])
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=R.declaration_for(cell), **kw)
    base_orig = H.evaluation_key(e['key'], e['overrides'], case_file_aa=ocfg.get('case_file_anderson_acceleration'),
                                 ess_ageing_baseline=ocfg.get('ess_ageing_baseline'), flex_price_multiplier=flex,
                                 convergence_depth_tail=ocfg.get('convergence_depth_tail'))
    return {'base_key_current_configuration': base, 'resettle_key': key,
            'base_key_original_configuration': base_orig, 'original_eval_key': e['eval_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The re-settling key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it
    appears in no committed campaign spec OUTSIDE the W118 stage root (the rule: a pre-run check that scans committed
    artefacts excludes the run's own); the original configuration's key reproduces the original eval key."""
    k = expected_keys(cell)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    e = orig_entry(cell)
    eval_dir_name = H.eval_dir_name(k['resettle_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['resettle_key'])
    work = L._work_dir()
    entry = next((x for x in (spec or {}).get('candidates') or [] if x['label'] == cell), None)
    parts = {
        'base_key_original_configuration_equals_original_eval_key': k['base_key_original_configuration']
        == e['eval_key'] == R.CELLS[cell]['orig_eval_key'],
        'resettle_key_differs_from_original_and_base': k['resettle_key'] not in (e['eval_key'],
                                                                                 k['base_key_current_configuration']),
        'resettle_key_absent_from_committed_specs_outside_w118_root': k['resettle_key'] not in committed,
        'campaign_root_differs_from_original': os.path.abspath(campaign_root(cell)) != os.path.abspath(
            _abs(R.CELLS[cell]['orig_root'])),
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


def validate_campaign_spec(cell, spec, ss_pin):
    ref = _load(W101_X0_SPEC_REL)          # the W101 x0 cell: the same C2 + tight-tail configuration, concurrency 1
    want = configuration(cell)
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    oe = orig_entry(cell)
    checks = {f'configuration:{k}': cfg.get(k) == want[k] for k in
              ('arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
               'ess_ageing_baseline_label', 'convergence_depth_tail')}
    checks.update({
        'configuration:case_file_sha256': cfg.get('case_file_sha256') == K.CASE_FILE_SHA256,
        'configuration:ess_params_file_C2': (cfg.get('ess_params_file') or {}).get('sha256') == K.ESS_PARAMS_SHA256_C2,
        'configuration:as_the_w101_x0_cell': all(cfg.get(k) == ref['configuration'].get(k) for k in (
            'arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
            'ess_ageing_baseline_label', 'convergence_depth_tail', 'apply_rho', 'full_diagnostics_in_rows',
            'case_file_sha256')),
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == cell,
        'entry_canonical_and_key_as_original': e.get('canonical') == oe['canonical'] and e.get('key') == oe['key'],
        'entry_overrides_empty': e.get('overrides') == {},
        'entry_post_certification_persist_only': e.get('post_certification') == {
            'persist_certified_models': True, 'hull_polish': False, 'reference': None},
        'entry_flex_multiplier': e.get('flex_price_multiplier') == R.CELLS[cell]['flex_price_multiplier'],
        'entry_resettle_is_the_declaration': e.get('settling_resettle') == R.declaration_for(cell),
        'entry_has_no_other_continuation': not any(x in e for x in ('settling_continuation', 'certification_continuation',
                                                                    'settling_extension', 'release_solution_bookkeeping',
                                                                    'model_variant')),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_IDS[cell],
        'cap_is_the_cell_cap': spec.get('cap') == R.spec_cap(cell),
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_10': spec.get('required_consecutive_cycles') == 10,
        'bar_window_as_w101': spec.get('bar_window_cycles') == ref['bar_window_cycles'],
        'thread_caps_as_w101': spec.get('thread_caps') == ref['thread_caps'],
        'interpreter_as_w101': spec.get('interpreter') == ref['interpreter'],
        'solver_path_as_w101': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'stage_spec_pinned': (spec.get('extra') or {}).get('stage_spec') == ss_pin,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
    })
    return checks


def parent_capture_checklist(cell, spec):
    """The child's capture checklist, asserted in the PARENT too (before the lock), with the spec's declared tail and
    case-file AA (the child re-asserts it with its own tail checklist before any solve)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = R.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        return True, checks
    except Exception as error:  # noqa: BLE001
        return False, {'error': f'{type(error).__name__}: {error}'}


# ======================================================================================================================
#  provenance, P_MAX, walls
# ======================================================================================================================
def production_since_originals():
    """REPORT (the bitwise gate is the evidence): production .py changes between each original run's recorded git head
    and HEAD. GATE: no uncommitted change to any file the run uses (CODE_PINNED, the production files, the harness's
    clean list, EXTRA_CLEAN_FILES); other uncommitted tracked .py files are REPORTED (W118 runs beside parallel Workers
    whose in-progress edits touch files this run never imports)."""
    heads = {}
    for cell in R.CELL_ORDER:
        spec = K._spec_entry(cell)[0]
        heads.setdefault(spec.get('git_head'), []).append(cell)
    per = {}
    for head, cells in heads.items():
        diff = H._git(['diff', '--name-status', head, 'HEAD', '--', *PRODUCTION_FILES]).splitlines() if head else []
        per[head] = {'cells': cells, 'production_changed_since': diff}
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    used = set(CODE_PINNED) | set(PRODUCTION_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) | set(EXTRA_CLEAN_FILES)
    dirty_used = [d for d in dirty if d.split()[-1] in used]   # porcelain 'XY path' (the helper strips line 1)
    return {'per_original_git_head': per, 'uncommitted_tracked_py': dirty, 'uncommitted_files_this_run_uses': dirty_used,
            'head': H._git(['rev-parse', 'HEAD']),
            'note': ('production changed since the pre-tail originals (the tail, W84-W86, and later writer-only / keyed '
                     'changes); the W86 tail re-certification showed pre-tail runs replay bitwise with the tail enabled up '
                     'to the first tail cycle; the in-cycle gate through k0 is the evidence for each gated cell'),
            'ok': not dirty_used}


def p_max_from_decisions():
    """P_MAX = the longest post-certification period measured on the settled SRP1 references: max P_hat of the
    committed W102 (x0) and W103 (unit) settling decisions; L = 2 P_MAX."""
    vals = {}
    for name, rel in P_MAX_SOURCES.items():
        d = _load(rel)
        vals[name] = {'P_hat': d['P_hat'], 'k_star': d['k_star'], 'path': rel, 'sha256': _sha(rel)}
    p = max(v['P_hat'] for v in vals.values())
    return p, {'sources': vals, 'P_MAX': p, 'L': 2 * p, 'equals_hooks_constant': p == R.P_MAX and 2 * p == R.L_MONO,
               'citation': 'W102 (x0, evidence e4b5c992): P_hat 29; W103 (unit, evidence 3ee62863): P_hat 30'}


def _iteration_walls(eval_dir_rel):
    txt = open(_abs(os.path.join(eval_dir_rel, 'child_stdout.log')), errors='replace').read()
    return [float(m) for m in re.findall(r'\[INFO\] \t - Iteration \d+: ([0-9.]+) s', txt)]


def wall_time_estimate():
    """Per cell: s = the original run's mean per-cycle wall (its own campaign concurrency, 5 or 7) x kappa, kappa =
    W104's C* mean per-cycle wall at concurrency 1 (W101 captures) / the W86 recert's C* mean at concurrency 3 -- a
    conservative concurrency-1 factor (the originals ran at 5-7); plus W110's measured child + launcher overhead and one
    inline re-run of the zero-solve checks. Expected cycles: gated k0 + 60 (Phase B prediction), the F2 challenger its cap
    (the Advisor's prediction), the F2 incumbent k0 + 60, the year ladder the original k0 (110, a proxy: k0_run is
    unknown) + 60; worst = the cap."""
    c104 = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_c_star', 'evals',
                        '4bf36c151fd10613_c_star')
    c86 = os.path.join(_P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals', '96c5aa50cc229cc1_c_star')
    kappa = statistics.mean(_iteration_walls(c104)) / statistics.mean(_iteration_walls(c86))
    w110 = os.path.join(_P53, 'w105_c_star_extension', 'campaign_s53_w105_c_star_ext_r2')
    w110_eval = os.path.join(w110, 'evals', '8864266d3c064ed0_c_star_ext')
    lines = _read_jsonl(_abs(os.path.join(w110_eval, 'settling_extension_cycle_record.jsonl')))
    sum_walls = sum(x['t_end_s'] - x['t_start_s'] for x in lines)
    child = _load(os.path.join(w110_eval, 'evaluation_record.json'))['wall_time_s']['child_process_s']
    campaign = _load(os.path.join(w110, 'campaign_results.json'))['wall_clock_s']
    overhead = (child - sum_walls) + (campaign - child) + 120.0
    per, tot_exp, tot_worst = {}, 0.0, 0.0
    for cell in R.CELL_ORDER:
        c = R.CELLS[cell]
        walls = _iteration_walls(R.original_eval_dir(cell))
        s = statistics.mean(walls) * kappa
        cap = (c['N_old'] + 100) if c['gated'] else min(c['k0'] + R.CAP_AFTER_K0, R.CAP_CEILING)
        exp_cycles = cap if cell == 'f2_challenger' else min(c['k0'] + 60, cap)
        e_s, w_s = exp_cycles * s + overhead, cap * s + overhead
        per[cell] = {'original_mean_cycle_s': statistics.mean(walls), 'original_cycles': len(walls),
                     's_per_cycle_estimate': s, 'expected_cycles': exp_cycles, 'cap_cycles': cap,
                     'expected_h': e_s / 3600.0, 'worst_case_h': w_s / 3600.0}
        tot_exp += e_s
        tot_worst += w_s
    return {'basis': wall_time_estimate.__doc__, 'kappa_concurrency_1_over_3': kappa,
            'overhead_s_per_cell': overhead, 'per_cell': per, 'total_expected_h': tot_exp / 3600.0,
            'total_worst_case_h': tot_worst / 3600.0, 'brief_statement': '~ 14-17 h plus the added cell (Addendum 57)'}


# ======================================================================================================================
#  post-run gates
# ======================================================================================================================
LINE_FIELDS_REQUIRED = ('cycle', 'phase', 'regime', 'gross', 'gross_hex', 'net_operational_recourse',
                        'terminal_salvage_value', 'boyd_k', 'local_solves_ok', 'boyd_ratios', 'boyd_pf_primal_ratio',
                        'settling', 'holds', 'replay_equal', 'certificate_length_in_force_at_cycle_end', 't_start_s',
                        't_end_s', 't_sum', 't_by_node', 'Q_cc', 'first_residual_pass_run')
SETTLING_FIELDS_REQUIRED = ('k0', 'N', 'cap', 'eligible', 'dQ', 's_k', 'sign_change', 'len_T', 'certA', 'certB',
                            'certA_parts', 'certB_parts', 't_sum', 'Q_cc', 'gap_ok', 'decision', 'reasons')
CREEP_FIELDS_REQUIRED = ('cycle', 'q_decomposition', 'ess_movement', 'boyd', 't_sum', 'Q_cc', 'capture_s')


def _decision(eval_dir):
    p = os.path.join(eval_dir, R.DECISION_FILE)
    return json.load(open(p)) if os.path.isfile(p) else None


def hold_checks(cell, eval_dir, rec):
    lines = _read_jsonl(os.path.join(eval_dir, R.CYCLE_FILE))
    summ = rec.get('settling_resettle_summary') or {}
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    fp = summ.get('first_residual_pass_run')
    pre = [by[c] for c in cycles if fp is None or c <= fp]
    held = [by[c] for c in cycles if fp is not None and c > fp]
    rho_fp = ((by.get(fp) or {}).get('rho') or {}).get('rho_after') if fp else None
    dec = _decision(eval_dir) or {}
    last = cycles[-1] if cycles else None
    parts = {
        'one_line_per_cycle_contiguous': cycles == list(range(1, (rec.get('cycles_run') or 0) + 1)),
        'first_pass_as_declared_for_gated': (not R.CELLS[cell]['gated']) or fp == R.CELLS[cell]['k0'],
        'no_hold_acted_through_first_pass': all(
            (x.get('holds') or {}) == {'aa': False, 'tail_apply': False, 'tail_next': False, 'rho': False}
            or ((x.get('holds') or {}).get('aa') is None and x.get('gross') is None) for x in pre),
        'aa_held_off_after_first_pass': all((x.get('aa') is None and x.get('gross') is None)
                                            or ((x.get('aa') or {}).get('hold') is True
                                                and (x.get('aa') or {}).get('action') == R.AA_OFF_ACTION) for x in held),
        'tail_held_on_after_first_pass': all((x.get('tail_apply') or {}).get('active_passed') is True
                                             and (x.get('tail_next') or {}).get('returned') is True for x in held),
        'rho_frozen_after_first_pass_at_its_values': fp is None or (bool(rho_fp) and all(
            (x.get('rho') or {}).get('hold') is True and (x.get('rho') or {}).get('rho_after') == rho_fp
            and not (x.get('rho') or {}).get('changed_channels') for x in held)),
        'certificate_length_disabled_except_the_rule_end': all(
            x.get('certificate_length_in_force_at_cycle_end') == R.CERTIFICATION_DISABLED_THRESHOLD
            or (x['cycle'] == last and x.get('certificate_length_in_force_at_cycle_end') == R.SETTLING_END_THRESHOLD
                and summ.get('stopped_by') in ('settling_rule', 'rule_cap')) for x in lines),
        'replay_equal_every_gated_cycle': ((not R.CELLS[cell]['gated'])
                                           or (all(by[c].get('replay_equal') is True
                                                   for c in range(1, R.CELLS[cell]['k0'] + 1) if c in by)
                                               and all(c in by for c in range(1, R.CELLS[cell]['k0'] + 1)))),
        'summary_ok': summ.get('ok') is True,
        'decision_present': bool(dec),
    }
    return all(parts.values()), {'parts': parts, 'first_pass': fp, 'rho_at_first_pass': rho_fp,
                                 'n_cycles_where_the_hold_changed_a_value': {
                                     'aa': sum(1 for x in held if (x.get('aa') or {}).get('hold_changed_value')),
                                     'tail_apply': sum(1 for x in held if (x.get('tail_apply') or {}).get(
                                         'hold_changed_value')),
                                     'tail_next': sum(1 for x in held if (x.get('tail_next') or {}).get(
                                         'hold_changed_value'))}}


def stopping_check(cell, rec, eval_dir):
    summ = rec.get('settling_resettle_summary') or {}
    dec = _decision(eval_dir) or {}
    k = rec.get('cycles_run')
    if dec.get('status') == 'certified':
        ok = k == dec.get('k_star') and summ.get('stopped_by') == 'settling_rule'
    elif dec.get('status') == 'uncertified':
        spec_cap = R.spec_cap(cell)
        if R.CELLS[cell]['gated']:
            ok = k == dec.get('k_cap') == spec_cap and summ.get('stopped_by') == 'cap'
        else:
            ok = k == dec.get('k_cap') and summ.get('stopped_by') == ('rule_cap' if k < spec_cap else 'cap')
    else:
        ok = False
    return bool(ok), {'cycles_run': k, 'decision_status': dec.get('status'), 'k_star': dec.get('k_star'),
                      'k_cap': dec.get('k_cap'), 'stopped_by': summ.get('stopped_by'), 'spec_cap': R.spec_cap(cell)}


def settling_replay_check(cell, eval_dir):
    """The pure rule v2 replayed on the run's per_cycle_record (Q, boyd) and the in-cycle t_sum reproduces every
    in-cycle rule record and the decision."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, R.CYCLE_FILE))}
    cr = R.declaration_for(cell)['cap_rule']
    rule = (SC2.SettlingRuleV2(R.P_MAX, cap=cr['cap']) if cr['kind'] == 'fixed' else
            SC2.SettlingRuleV2(R.P_MAX, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling']))
    pure = []
    for r in rows:
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'], bool(r['boyd_all_pass'] and r['local_solves_ok']),
                                 (lines.get(r['cycle']) or {}).get('t_sum')))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    keys = ('status', 'k_star', 'branch', 'k0', 'N', 'T', 'A', 'P_hat', 'W', 'window', 'band', 'band_width', 'k_cap',
            'reasons', 'range', 't_sum_k_star', 'gap_refusals', 'drift_rate_mean_dQ_last_25', 'dQ_cc_rate_mean_last_25')
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k)) for k in keys),
             'decision_file_present': bool(dec)}
    return all(parts.values()), {'parts': parts}


def replay_gate_full(cell, eval_dir):
    """Gated cells: rows 1..k0 of the run's per_cycle_record.jsonl against the ORIGINAL record's rows: EVERY field, as
    JSON text (adds REPLAY_POST_RUN_ONLY_FIELDS to the in-cycle gate)."""
    c = R.CELLS[cell]
    ref = {r['cycle']: r for r in _read_jsonl(_abs(R.reference_path(cell)))}
    path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    run = {r['cycle']: r for r in _read_jsonl(path)} if os.path.isfile(path) else {}
    first, detail = None, None
    for k in range(1, c['k0'] + 1):
        a, b = run.get(k), ref.get(k)
        if a is None:
            first, detail = k, {'missing_in_run': True}
            break
        diff = sorted(f for f in set(a) | set(b) if json.dumps(a.get(f), sort_keys=True) != json.dumps(b.get(f),
                                                                                                        sort_keys=True))
        if diff:
            ga, gb = a.get('gross_operational_cost'), b.get('gross_operational_cost')
            first, detail = k, {'fields_differing': diff, 'gross_difference_run_minus_recorded':
                                (ga - gb) if (ga is not None and gb is not None) else None}
            break
    return {'bitwise_through_k0': first is None, 'k0': c['k0'], 'first_divergence_cycle': first, 'divergence': detail}


def line_fields_check(eval_dir):
    lines = _read_jsonl(os.path.join(eval_dir, R.CYCLE_FILE))
    missing = {}
    for x in lines:
        m = [f for f in LINE_FIELDS_REQUIRED if f not in x]
        if x.get('gross') is not None:
            m += [f for f in ('blocks_captured',) if f not in x]
        s = x.get('settling') or {}
        m += [f'settling.{f}' for f in SETTLING_FIELDS_REQUIRED if f not in s]
        m += [f'boyd_ratios.{f}' for f in R.BOYD_RATIO_FIELDS if f not in (x.get('boyd_ratios') or {})]
        if m:
            missing[str(x['cycle'])] = m
    return not missing and bool(lines), {'n_lines': len(lines), 'missing_by_cycle': dict(list(missing.items())[:10])}


def creep_capture_check(eval_dir, rec, rows):
    creep = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, R.CREEP_FILE))}
    ess_path = os.path.join(eval_dir, R.ESS_SCHEDULE_FILE)
    ess = _read_jsonl(ess_path) if os.path.isfile(ess_path) else []
    n = rec.get('cycles_run') or 0
    ok_cycles = [r['cycle'] for r in rows if r.get('gross_operational_cost') is not None]
    gpath = os.path.join(eval_dir, 'g_s39_D.json')
    g_rows = {r['cycle']: r for r in json.load(open(gpath))['cycle_trajectory']} if os.path.isfile(gpath) else {}
    boyd_bad = []
    for c, x in creep.items():
        gr = g_rows.get(c)
        b = x.get('boyd') or {}
        for ch in R.CHANNELS:
            for f in ('r', 's', 'eps_pri', 'eps_dual', 'primal_ratio', 'dual_ratio'):
                if gr is None or (b.get(ch) or {}).get(f) != gr.get(f'boyd_{ch}_{f}'):
                    boyd_bad.append((c, ch, f))
    parts = {
        'one_creep_line_per_cycle': sorted(creep) == list(range(1, n + 1)),
        'fields_every_line': all(all(f in x for f in CREEP_FIELDS_REQUIRED) for x in creep.values()),
        'q_decomposition_every_successful_cycle_reconciles': all(
            ((creep.get(c) or {}).get('q_decomposition') or {}).get('reconciliation', {}).get('reconciles') is True
            for c in ok_cycles),
        'ess_movement_available_from_cycle_2': all(((creep.get(c) or {}).get('ess_movement') or {}).get('available')
                                                   is True for c in range(2, n + 1)),
        'boyd_captured_every_cycle_equals_production_rows': not boyd_bad,
        'ess_schedule_header_plus_one_line_per_cycle': (len(ess) == n + 1 and bool(ess) and ess[0].get('header') is True
                                                        and [x['cycle'] for x in ess[1:]] == list(range(1, n + 1))),
        'no_capture_errors': (rec.get('settling_resettle_summary') or {}).get('n_capture_errors') == 0,
    }
    return all(parts.values()), {'parts': parts, 'boyd_mismatch_first10': boyd_bad[:10]}


def t_sum_check(eval_dir, stride_rel=None, detail_rel=None):
    """In-cycle t_sum == W112's formula on the run's own pf stride (pi, w from its terminal detail), every cycle, to
    1e-6 EUR (the bitwise count reported); the last cycle's t_sum against the terminal identity t_tso_plus_t_dso_terminal
    within 0.01 EUR (the W112 tolerance)."""
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, R.CYCLE_FILE))}
    stride_rel = stride_rel or os.path.relpath(os.path.join(eval_dir, 'pf_entry_stride_s39_D.jsonl'), REPO)
    detail_rel = detail_rel or os.path.relpath(os.path.join(eval_dir, 'interface_settlement_detail_s31c.json'), REPO)
    detail = json.load(open(_abs(detail_rel)))
    pi, w = W112._price_weight(detail)
    per = W112._stream_stride(stride_rel, pi, w)
    diffs = {c: abs(lines[c]['t_sum'] - per[c]['t_sum']) for c in lines if c in per and lines[c].get('t_sum') is not None}
    missing = [c for c in lines if c not in per or lines[c].get('t_sum') is None]
    last = max(lines)
    ident = W112._detail_identity(detail)
    term = abs(lines[last]['t_sum'] - ident['t_tso_plus_t_dso_terminal']) if lines[last].get('t_sum') is not None else None
    parts = {'every_cycle_has_t_sum_and_a_stride_line': not missing,
             'in_cycle_equals_stride_formula_to_1e-6': bool(diffs) and max(diffs.values()) <= 1e-6,
             'terminal_identity_within_0_01': term is not None and term <= 0.01 and ident['cycles_run'] == last}
    return all(parts.values()), {'parts': parts, 'max_abs_diff_vs_stride': max(diffs.values()) if diffs else None,
                                 'n_bitwise_equal': sum(1 for v in diffs.values() if v == 0.0), 'n_cycles': len(diffs),
                                 'missing_first10': missing[:10], 'terminal_abs_diff': term,
                                 'terminal_t_tso_plus_t_dso': ident['t_tso_plus_t_dso_terminal']}


def overlap_check(cell, rec):
    summ = rec.get('settling_resettle_summary') or {}
    ov = summ.get('overlap_k0_plus_1_to_N_old') or []
    c = R.CELLS[cell]
    if not c['gated']:
        return len(ov) == 0, {'n': len(ov)}
    ok = [o['cycle'] for o in ov] == list(range(c['k0'] + 1, c['N_old'] + 1)) and all(
        o.get('Q_new_minus_Q_old') is not None for o in ov)
    return bool(ok), {'n': len(ov)}


GATE_SCOPE = {
    'G19_replay_bitwise_1_k0_every_field': 'gated cells only (the year ladder has no replay reference: SKIPPED)',
    'G23_overlap_recorded': 'gated cells: k0+1..N_old recorded; ungated: none (asserted empty)',
    'all_other_gates': 'every cell',
}


def cell_gates(cell, entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code, 'gate_scope': GATE_SCOPE}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('barrier') and rec.get('status') == 'error':
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
    gates['G8_persistence_as_w101'], detail['G8'] = W101L.persistence_check(rec, eval_dir)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_first_pass_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_v2_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    if R.CELLS[cell]['gated']:
        rg = replay_gate_full(cell, eval_dir)
        gates['G19_replay_bitwise_1_k0_every_field'] = rg['bitwise_through_k0']
        detail['G19'] = rg
    else:
        detail['G19'] = {'skipped': 'ungated cell (first C2 evaluation): no replay reference'}
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = creep_capture_check(eval_dir, rec, rows)
    gates['G22_t_sum_in_cycle_equals_stride_and_terminal'], detail['G22'] = t_sum_check(eval_dir)
    gates['G23_overlap_recorded'], detail['G23'] = overlap_check(cell, rec)
    return gates, detail, rec


def cell_report(cell, eval_dir, rec):
    """The frozen per-cell report (DEFINITIONS['per_cell']) from the run's artefacts."""
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, R.CYCLE_FILE))}
    dec = _decision(eval_dir) or {}
    c = R.CELLS[cell]
    summ = rec.get('settling_resettle_summary') or {}
    q = {k: r['gross_operational_cost'] for k, r in rows.items()}
    last = max(rows)
    steps = [q[k] - q[k - 1] for k in sorted(q) if (k - 1) in q and q[k] is not None and q[k - 1] is not None]
    fp = summ.get('first_residual_pass_run')
    after = [x for k, x in lines.items() if fp is not None and k > fp]
    rep = {'cell': cell, 'group': c['group'], 'gated': c['gated'], 'eval_key': rec.get('eval_key'),
           'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
           'original_eval_key': c['orig_eval_key'], 'k0_run': fp, 'k0_original': c['k0'] if c['gated'] else None,
           'N_old': c['N_old'] if c['gated'] else None, 'rule_cap': (dec.get('cap') or summ.get('rule_cap')),
           'cycles_run': rec.get('cycles_run'), 'status': dec.get('status'), 'k_star': dec.get('k_star'),
           'branch': dec.get('branch'), 'band': dec.get('band'), 'band_width': dec.get('band_width'),
           'range_over_tau': dec.get('range_over_tau'), 'Q_k_star': dec.get('Q_k_star'),
           'T': dec.get('T'), 'A': dec.get('A'), 'P_hat': dec.get('P_hat'),
           't_sum_k_star': dec.get('t_sum_k_star'), 'Q_cc_k_star': dec.get('Q_cc_k_star'),
           'lapse_events': dec.get('lapse_events'), 'gap_refusals': dec.get('gap_refusals'),
           'net_operational_recourse_last': rows[last].get('recourse'),
           'terminal_salvage_value_last': rows[last].get('terminal_salvage_value'),
           'terminal_step_abs': abs(steps[-1]) if steps else None,
           'terminal_step_over_EPS0': (abs(steps[-1]) / SC2.EPS0) if steps else None,
           'rule_ten_last_cycle': ((rows[last]['objective_change_abs'] / rows[last]['objective_tolerance'])
                                   if rows[last].get('objective_change_abs') is not None
                                   and rows[last].get('objective_tolerance') else None),
           'boyd_lapses_after_k0': sum(1 for x in after if not x.get('boyd_k')),
           'pf_primal_ratio_after_k0': {'max': max((x.get('boyd_pf_primal_ratio') or 0.0) for x in after) if after
                                        else None, 'last': after[-1].get('boyd_pf_primal_ratio') if after else None},
           'objective_convention': DEFINITIONS['objective_convention']}
    if dec.get('status') == 'uncertified':
        rep.update({k: dec.get(k) for k in ('k_cap', 'reasons', 'band_window', 'drift_rate_mean_dQ_last_25',
                                            'dQ_cc_rate_mean_last_25', 'Q_at_cap', 't_sum_at_cap', 'Q_cc_at_cap',
                                            'gap_clause_refused_at_cap')})
    if c['gated']:
        ref = {r['cycle']: r for r in _read_jsonl(_abs(R.reference_path(cell)))}
        q_n_old = ref[c['N_old']]['gross_operational_cost']
        q_end = dec.get('Q_k_star') if dec.get('status') == 'certified' else dec.get('Q_at_cap')
        ov = summ.get('overlap_k0_plus_1_to_N_old') or []
        ov_n = next((o for o in ov if o['cycle'] == c['N_old']), None)
        rep.update({'Q_N_old': q_n_old, 's_signed': (q_end - q_n_old) if q_end is not None else None,
                    's_resolution_band_width': dec.get('band_width'),
                    's_flag': None if dec.get('status') == 'certified' else 'uncertified: Q_at_cap - Q_N_old',
                    'overlap': ov, 'overlap_at_N_old': ov_n,
                    'replay_bitwise_through': summ.get('replay_bitwise_through_cycle'),
                    'replay_first_divergence': summ.get('replay_first_divergence')})
    return rep


def _verdict(x, res):
    if x is None or res is None:
        return 'indeterminate_uncertified'
    return 'determinate' if abs(x) > res else 'within resolution'


def differences(reports, i_values, x0):
    """The frozen difference formulas (DEFINITIONS['differences']). `reports` = {cell: cell_report}; `i_values` =
    {cell: I_j}; `x0` = {'Q181', 'band_width', 't'}. Pure."""
    out = {'x0_comparator': x0, 'phase_b_and_year_ladder_vs_x0': {}, 'f2_certificate': None, 'year_ladder': None}
    for cell, r in reports.items():
        if R.CELLS[cell]['group'] == 'f2':
            continue
        i = i_values[cell]
        if r.get('status') != 'certified':
            out['phase_b_and_year_ladder_vs_x0'][cell] = {
                'status': r.get('status'), 'M': None, 'M_cc': None, 'verdict': 'indeterminate_uncertified',
                'band_at_cap': r.get('band'), 'band_width': r.get('band_width'),
                'drift_rate': r.get('drift_rate_mean_dQ_last_25'), 'dQ_cc_rate': r.get('dQ_cc_rate_mean_last_25'),
                'Q_at_cap': r.get('Q_at_cap'), 't_sum_at_cap': r.get('t_sum_at_cap'), 'I_j': i}
            continue
        m = i + r['Q_k_star'] - x0['Q181']
        m_cc = i + r['Q_k_star'] + r['t_sum_k_star'] - (x0['Q181'] + x0['t'])
        res = r['band_width'] + x0['band_width']
        out['phase_b_and_year_ladder_vs_x0'][cell] = {
            'status': 'certified', 'I_j': i, 'Q_k_star': r['Q_k_star'], 't_sum_k_star': r['t_sum_k_star'], 'M': m,
            'M_cc': m_cc, 'resolution': res, 'verdict': _verdict(m, res), 'verdict_cc_report_only': _verdict(m_cc, res),
            'margin_over_resolution': abs(m) / res}
    rc, ri = reports.get('f2_challenger'), reports.get('f2_incumbent')
    if rc is not None and ri is not None:
        if rc.get('status') == 'certified' and ri.get('status') == 'certified':
            dd = (i_values['f2_challenger'] - i_values['f2_incumbent']) + (rc['Q_k_star'] - ri['Q_k_star'])
            dd_cc = dd + (rc['t_sum_k_star'] - ri['t_sum_k_star'])
            res = rc['band_width'] + ri['band_width']
            out['f2_certificate'] = {'D': dd, 'D_cc': dd_cc, 'resolution': res, 'verdict': _verdict(dd, res),
                                     'verdict_cc_report_only': _verdict(dd_cc, res),
                                     'I_c_minus_I_i': i_values['f2_challenger'] - i_values['f2_incumbent']}
        else:
            out['f2_certificate'] = {'D': None, 'verdict': 'indeterminate_uncertified',
                                     'bands': {'challenger': rc.get('band'), 'incumbent': ri.get('band')},
                                     'band_widths': {'challenger': rc.get('band_width'), 'incumbent': ri.get('band_width')},
                                     'rates': {'challenger': {'drift': rc.get('drift_rate_mean_dQ_last_25'),
                                                              'dQ_cc': rc.get('dQ_cc_rate_mean_last_25')},
                                               'incumbent': {'drift': ri.get('drift_rate_mean_dQ_last_25'),
                                                             'dQ_cc': ri.get('dQ_cc_rate_mean_last_25')}},
                                     'status': {'challenger': rc.get('status'), 'incumbent': ri.get('status')}}
        out['f2_certificate']['expert_note'] = ('the cited 6,338 EUR margin is < 2 tau: "within resolution" is the '
                                                'expected verdict and it is acceptable (Addendum 57 Decision 3(c))')
    y30, y35 = reports.get('yl_y2030'), reports.get('yl_y2035')
    if y30 is not None and y35 is not None:
        if y30.get('status') == 'certified' and y35.get('status') == 'certified':
            dd = (i_values['yl_y2035'] + y35['Q_k_star']) - (i_values['yl_y2030'] + y30['Q_k_star'])
            dd_cc = dd + (y35['t_sum_k_star'] - y30['t_sum_k_star'])
            res = y30['band_width'] + y35['band_width']
            out['year_ladder'] = {'D_2035_minus_2030': dd, 'D_cc': dd_cc, 'resolution': res,
                                  'verdict': _verdict(dd, res), 'verdict_cc_report_only': _verdict(dd_cc, res)}
        else:
            out['year_ladder'] = {'D_2035_minus_2030': None, 'verdict': 'indeterminate_uncertified',
                                  'status': {'2030': y30.get('status'), '2035': y35.get('status')}}
    return out


def score_predictions(reports, diffs):
    """The recorded predictions scored with their frozen / operationalised definitions. Pure."""
    out = {}
    g = {}
    for cell in R.GATED_CELLS:
        r = reports.get(cell) or {}
        g[cell] = {'held': r.get('replay_bitwise_through') == R.CELLS[cell]['k0'] and not r.get('replay_first_divergence'),
                   'replay_bitwise_through': r.get('replay_bitwise_through')}
    out['gate_outcome'] = g
    p = PREDICTIONS['tail_overlap']
    out['tail_overlap'] = {cell: {'relative_at_N_old': ((reports.get(cell) or {}).get('overlap_at_N_old') or {}).get(
        'relative'), 'held': (lambda v: v is not None and p['lo'] <= v <= p['hi'])(
        ((reports.get(cell) or {}).get('overlap_at_N_old') or {}).get('relative'))} for cell in R.GATED_CELLS}
    pb = PREDICTIONS['phase_b']
    out['phase_b'] = {}
    for cell in R.GATED_CELLS:
        if R.CELLS[cell]['group'] != 'phase_b':
            continue
        r = reports.get(cell) or {}
        d = (diffs.get('phase_b_and_year_ladder_vs_x0') or {}).get(cell) or {}
        if r.get('status') != 'certified':
            out['phase_b'][cell] = {'verdict': 'not_scoreable_uncertified', 'status': r.get('status')}
            continue
        k_rel = r['k_star'] - r['k0_run']
        s = r.get('s_signed')
        out['phase_b'][cell] = {
            'oscillatory_in_window': r.get('branch') == 'oscillatory' and pb['k_star_after_k0'][0] <= k_rel
            <= pb['k_star_after_k0'][1], 'k_star_minus_k0': k_rel,
            's': s, 's_in_range': s is not None and pb['s_range_eur'][0] <= s <= pb['s_range_eur'][1],
            's_near_x0': s is not None and abs(s - pb['s_x0_eur']) <= pb['s_minus_s_x0_bound_eur'],
            'margin_positive': d.get('M') is not None and d['M'] > 0, 'M': d.get('M'),
            't_sum_below_gap': r.get('t_sum_k_star') is not None and abs(r['t_sum_k_star']) < SC2.GAP_BOUND,
            'resolution_s': r.get('band_width')}
    rc = reports.get('f2_challenger') or {}
    out['f2_challenger'] = {'uncertified_at_cap': rc.get('status') == 'uncertified', 'status': rc.get('status'),
                            'note': 'a probability (>= 0.5): the outcome is recorded, not scored held / missed'}
    ri = reports.get('f2_incumbent') or {}
    k0i = ri.get('k0_run')
    t_after = sorted(t[0] for t in (ri.get('T') or []) if k0i is not None and t[0] > k0i)
    t0 = t_after[0] if t_after else None
    out['f2_incumbent'] = {'first_turning_point_after_k0': t0, 'k0_run': k0i,
                           'held': t0 is not None and t0 <= k0i + PREDICTIONS['f2_incumbent']['bound_cycles'],
                           'operationalisation': PREDICTIONS['f2_incumbent']['operationalisation']}
    yl = PREDICTIONS['year_ladder']
    out['year_ladder'] = {cell: {'k_star_minus_k0': ((reports.get(cell) or {}).get('k_star') or 0)
                                 - ((reports.get(cell) or {}).get('k0_run') or 0)
                                 if (reports.get(cell) or {}).get('status') == 'certified' else None,
                                 'held': ((reports.get(cell) or {}).get('status') == 'certified'
                                          and yl['k_star_after_k0'][0] <= (reports[cell]['k_star'] - reports[cell]['k0_run'])
                                          <= yl['k_star_after_k0'][1])}
                          for cell in R.UNGATED_CELLS}
    return out


def differences_self_tests():
    x0 = {'Q181': 1000.0, 'band_width': 10.0, 't': 1.0}
    base = {'status': 'certified', 'band_width': 5.0, 't_sum_k_star': 2.0}
    reports = {c: dict(base, Q_k_star=900.0) for c in R.CELL_ORDER}
    reports['f2_challenger'] = dict(base, Q_k_star=500.0)
    reports['f2_incumbent'] = dict(base, Q_k_star=507.0, t_sum_k_star=0.0)
    reports['yl_y2035'] = dict(base, Q_k_star=950.0)
    iv = {c: 50.0 for c in R.CELL_ORDER}
    d = differences(reports, iv, x0)
    m = d['phase_b_and_year_ladder_vs_x0']['pb_y2030_n9']
    ok1 = (m['M'] == 50.0 + 900.0 - 1000.0 and m['M_cc'] == 50.0 + 900.0 + 2.0 - 1001.0 and m['resolution'] == 15.0
           and m['verdict'] == 'determinate')
    f2 = d['f2_certificate']
    ok2 = f2['D'] == -7.0 and f2['D_cc'] == -5.0 and f2['resolution'] == 10.0 and f2['verdict'] == 'within resolution'
    yl = d['year_ladder']
    ok3 = yl['D_2035_minus_2030'] == 50.0 and yl['verdict'] == 'determinate' and yl['resolution'] == 10.0
    reports2 = dict(reports, f2_incumbent={'status': 'uncertified', 'band': [1.0, 2.0], 'band_width': 1.0,
                                           'drift_rate_mean_dQ_last_25': -5.0, 'dQ_cc_rate_mean_last_25': -3.0})
    d2 = differences(reports2, iv, x0)
    ok4 = d2['f2_certificate']['D'] is None and d2['f2_certificate']['verdict'] == 'indeterminate_uncertified'
    return {'ok': bool(ok1 and ok2 and ok3 and ok4), 'phase_b': ok1, 'f2': ok2, 'year_ladder': ok3,
            'uncertified_indeterminate': ok4, 'example': d}


def _synthetic_run_dir(tmp, cell, variant):
    """A synthetic eval dir from the REAL wrappers' checks drive (K.drive): the cycle / creep / ess / blocks /
    decision files, a per_cycle_record (the original rows through N_old + the drive's continuation), a g_s39_D.json,
    a pf stride and an interface-settlement detail consistent with the drive's fake world (w = 1, pi = 1)."""
    d = K.drive(cell, 'certify' if variant != 'creep' else 'creep')
    files, st = d['files'], d['state']
    lines = {x['cycle']: x for x in files[R.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[R.CREEP_FILE]}
    if variant == 'tamper_t_sum_30':
        lines[30]['t_sum'] = lines[30]['t_sum'] + 1.0
    if variant == 'tamper_hold_flag':
        lines[max(lines) - 5]['aa']['hold'] = False
    if variant == 'tamper_decision':
        files[R.DECISION_FILE][0]['k_star'] = files[R.DECISION_FILE][0]['k_star'] - 1
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == R.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            elif fname == R.CYCLE_FILE:
                for c in sorted(lines):
                    handle.write(GRIO.dumps(lines[c], default=GRIO.json_default) + '\n')
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    ref = st.reference if st.gated else {}
    rows, g_rows = [], []
    g_orig = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(R.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if st.gated else {})
    for c in sorted(lines):
        x = lines[c]
        if st.gated and c <= R.CELLS[cell]['N_old']:
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
    # the pf stride and the terminal detail of the drive's fake world (w = 1, pi = 1; 864 active-power entries)
    n_e = len(K.NODES) * len(K.YEARS) * len(K.DAYS) * K.PERIODS
    with open(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), 'w') as handle:
        for c in sorted(lines):
            t = st.t_sum[c]
            gap = t / n_e
            ents = [{'node_id': n, 'year': str(y), 'day': d_, 'power_type': 'p', 'period': p, 'x_dso': 10.0 + gap,
                     'z_tso_current': 10.0, 'lambda_dso': 0.0, 's_base_dso': 100.0, 'rho_pf': 1.0, 'r': 0.0}
                    for n in K.NODES for y in K.YEARS for d_ in K.DAYS for p in range(K.PERIODS)]
            handle.write(json.dumps({'cycle': c, 'identity_holds': True, 'production_boyd_pf_r': 0.0,
                                     'entries': ents}) + '\n')
    last = max(lines)
    detail = {'t_tso_plus_t_dso_terminal': st.t_sum[last], 'cycles_run': last,
              'interface_reporting_detail': {str(n): {str(y): {d_: {'periods': {str(p): {'price_per_mwh': 1.0}
                                                                                  for p in range(K.PERIODS)}}
                                                               for d_ in K.DAYS} for y in K.YEARS} for n in K.NODES},
              'interface_consensus_residual_per_dso': {str(n): {'periods': {f'{y}|{d_}|{p}': {'admm_block_weight': 1.0}
                                                                            for y in K.YEARS for d_ in K.DAYS
                                                                            for p in range(K.PERIODS)},
                                                                'sum_pi_baseMVA_residual_weighted': 0.0} for n in K.NODES}}
    with open(os.path.join(tmp, 'interface_settlement_detail_s31c.json'), 'w') as handle:
        json.dump(detail, handle)
    rec = {'cycles_run': last, 'settling_resettle_summary': st.summary(), 'status': 'certified'}
    return rec, rows


def post_run_evaluator_self_tests():
    """The NEW / adapted post-run evaluators (G13, G15, G17, G18, G19, G21, G22, G23, the cell report) on synthetic
    eval dirs built by the real wrappers, with tampered negative controls. The reused W86 / W101 evaluators (G1-G11,
    G14, G16) are self-tested where they were built."""
    out = {}
    cases = (('pb_y2025_n5', 'pass', None, True), ('pb_y2025_n5', 'tampered_row_40', 'tamper_row_40', False),
             ('pb_y2025_n5', 'tampered_t_sum_30', 'tamper_t_sum_30', False),
             ('pb_y2025_n5', 'tampered_hold_flag', 'tamper_hold_flag', False),
             ('pb_y2025_n5', 'tampered_decision', 'tamper_decision', False),
             ('yl_y2030', 'ungated_rule_cap', 'creep', True))
    for cell, name, variant, expect in cases:
        tmp = tempfile.mkdtemp(prefix='w118_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, cell, variant)
            h_ok, _h = hold_checks(cell, tmp, rec)
            s_ok, _s = stopping_check(cell, rec, tmp)
            k_ok, _k = settling_replay_check(cell, tmp)
            f_ok, _f = line_fields_check(tmp)
            c_ok, _c = creep_capture_check(tmp, rec, rows)
            t_ok, t_d = t_sum_check(tmp, os.path.relpath(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), REPO),
                                    os.path.relpath(os.path.join(tmp, 'interface_settlement_detail_s31c.json'), REPO))
            o_ok, _o = overlap_check(cell, rec)
            rg = replay_gate_full(cell, tmp) if R.CELLS[cell]['gated'] else {'bitwise_through_k0': True}
            allg = {'holds': h_ok, 'stopping': s_ok, 'rule_replay': k_ok, 'line_fields': f_ok, 'creep': c_ok,
                    't_sum': t_ok, 'overlap': o_ok, 'replay_full': rg['bitwise_through_k0']}
            if expect:
                ok = all(allg.values())
            elif variant == 'tamper_row_40':
                ok = (not rg['bitwise_through_k0']) and rg['first_divergence_cycle'] == 40 and all(
                    v for k_, v in allg.items() if k_ != 'replay_full')
            elif variant == 'tamper_t_sum_30':
                ok = (not t_ok) and t_d['max_abs_diff_vs_stride'] >= 0.99
            elif variant == 'tamper_hold_flag':
                ok = (not h_ok) and all(v for k_, v in allg.items() if k_ != 'holds')
            else:
                ok = (not k_ok) and (not s_ok)
            rep = cell_report(cell, tmp, rec) if expect else None
            out[name] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                         'report_on_synthetic': ({k: rep.get(k) for k in ('status', 'k_star', 'branch', 'band_width',
                                                                          's_signed', 't_sum_k_star', 'k0_run',
                                                                          'drift_rate_mean_dQ_last_25')}
                                                 if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    out['differences_formulas'] = differences_self_tests()
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  the stage spec
# ======================================================================================================================
def launch_command(cell, spec_sha=None, preconditions_only=False):
    log = f"run_{cell}_launch.log" if not preconditions_only else f"run_{cell}_preconditions_only.log"
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run --cell {cell} '
            f'--spec-sha256 {spec_sha or "<campaign spec sha256 of " + cell + ">"}'
            + (' --preconditions-only' if preconditions_only else '')
            + f' > {os.path.join(ROOT_REL, log)} 2>&1')


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


def _failing_check_items(checks):
    out = {}
    for sid, sec in ((checks or {}).get('sections') or {}).items():
        if (sec or {}).get('holds') is True:
            continue
        r = (sec or {}).get('result') or {}
        items = []
        if 'error' in r:
            items.append(f"error: {r['error']}")
        for k, v in (r.get('parts') or {}).items():
            if v is not True:
                items.append(f'parts.{k}')
        for k, v in (r.get('tests') or {}).items():
            if isinstance(v, dict) and v.get('holds', v.get('ok')) is not True:
                items.append(f'tests.{k}')
        for k, v in (r.get('cells') or {}).items() if isinstance(r.get('cells'), dict) else []:
            if isinstance(v, dict) and v.get('ok') is False:
                items.append(f'cells.{k}')
        out[sid] = items or ['holds is not True (no itemised field)']
    return out


def verbatim_check():
    text = _norm(open(_abs(BRIEF), encoding='utf-8').read())
    found = {k: _norm(v) in text for k, v in VERBATIM.items()}
    return {'brief': BRIEF, 'brief_sha256_at_freeze': _sha(BRIEF), 'brief_git_state': L._git_state(BRIEF),
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def _basis_ok():
    return [f'{n} not as committed: {p["path"]}' for n, p in BASIS_SPECS.items()
            if _sha(p['path']) != p['sha256'] or not _committed_clean(p['path'])]


def _common_checks():
    failures = _basis_ok()
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    for cell in R.CELL_ORDER:
        rel = R.reference_path(cell)
        if _sha(rel) != R.CELLS[cell]['per_cycle_record_sha256'] or not _committed_clean(rel):
            failures.append(f'{cell} original record not as committed: {rel}')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a W118 / W105 / W101 / W98 launcher is alive: {others}')
    return failures


def stage_spec_content(checks_inline, checks_file, post_tests, verb, solver, pres, prov, p_max_detail, mem, wall,
                       o_section):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    cells = {}
    for cell in R.CELL_ORDER:
        c = R.CELLS[cell]
        o = o_section['cells'][cell]
        cells[cell] = {
            'group': c['group'], 'gated': c['gated'], 'prefix': c['orig_eval_key'][:8],
            'original': o['original'], 'original_label': c['orig_label'],
            'k0_first_residual_pass_original': c['k0'], 'N_old': c['N_old'], 'Q_N_old': o['Q_N_old'],
            'canonical': o['canonical'], 'candidate_key': o['candidate_key'], 'e_over_p': o['e_over_p'],
            'lattice_legal': o['parts']['lattice_legal'], 'lattice_rules': o['lattice'],
            'in_the_current_admissible_domain': o['parts']['in_the_current_admissible_domain'],
            'substituted_plan': None, 'I_j': o['I_j'], 'I_j_source': o['I_j_source'],
            't_sum_terminal_original': o['t_sum_terminal_original'],
            'flex_price_multiplier': c['flex_price_multiplier'],
            'cap_rule': R.cap_rule(cell), 'spec_cap': R.spec_cap(cell), 'declaration': R.declaration_for(cell),
            'campaign_id': CAMPAIGN_IDS[cell], 'campaign_root': campaign_root_rel(cell),
            'configuration': configuration(cell), 'keys': expected_keys(cell),
            'pre_launch_assertion_at_freeze': pres[cell], 'launch_command_template': launch_command(cell)}
    return {
        'schema': 'p515_s53_resettle_spec_v1', 'series': SPEC_SERIES, 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'in_this_series': None,
                        'note': ('a NEW spec series (Worker choice, W118): the stage-spec series v39 / v41 pins '
                                 'settling_criterion.py version 1 and single-cell stages; this campaign freezes the '
                                 'criterion version 2 and ten cells. It builds on:'),
                        'basis_specs': BASIS_SPECS},
        'authority': [f'{BRIEF} Addendum 57 Decisions 2 and 3 (committed 8120f6d1)', f'{BRIEF} Addendum 54 Ruling 2',
                      'P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md Decision 3 and "Design otherwise"',
                      'TASKS.md Advisor eight-cell design review', 'Planner task W118'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'code_sha256': code, 'production_sha256': production, 'production_since_originals': prov,
        'cells': cells, 'cell_order': list(R.CELL_ORDER),
        'launch_order': {'order': list(R.CELL_ORDER),
                         'worker_recommendation': ('as ordered (F2 challenger -> F2 incumbent -> Phase B x 6 -> 2030 -> '
                                                   '2035); no change recommended: the F2 challenger answers the '
                                                   'highest-risk question first (C*-like creep), the incumbent is '
                                                   'needed for D, and the year ladder has no gate to front-load')},
        'inputs_in_force_now': o_section['inputs_now'], 'x0_comparator': o_section['x0_comparator'],
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '+ tight tail {enabled True, compl_inf_tol 1e-6} declared'),
                          'tail_rule': 'production: the tail acts from AA-off + 1 (next-state = this cycle converged)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as W101)',
                          'concurrency': CONCURRENCY, 'option_b_release_solution_bookkeeping': 'absent (as W101)',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze and at --run'},
        'stop_rule': {
            'module': 'settling_criterion_v2', 'class': 'settling_criterion_v2.SettlingRuleV2',
            'version': SC2.VERSION, 'version_1_unchanged': ('settling_criterion.py byte-identical to the v39 / v41 pins '
                                                           '(checks V0); every version-1 certificate untouched'),
            'constants': SC2.constants(R.P_MAX), 'readings': SC2.READINGS, 'algorithm': SC2.__doc__,
            'failing_reasons': list(SC2.FAILING_REASONS),
            'p_max': {'value': R.P_MAX, 'L': R.L_MONO, 'instance_measured': p_max_detail},
            'N': 'the run\'s first residual pass k0_run (no decision before k0_run + K_EXCL; no "before N_old" rule)',
            'cap': {'gated': 'N_old + 100 (= the spec cap)', 'ungated': 'min(k0_run + 109, 300) (dynamic; spec cap 300)'},
            'boyd_k': 'boyd_metrics[\'all_boyd_pass\'] AND local_solves_ok, from THIS cycle\'s boyd_metrics (AA wrapper)',
            'Q_k': 'gross_operational_cost (recourse wrapper); None when any local solve failed',
            't_sum_k': ('in-cycle at production\'s Boyd call: sum w * pi * (p_DSO - p_TSO) over the active-power pf '
                        'consensus copies (MW), pi = _expected_market_price, w = _get_admm_block_weight (production); '
                        'validated post-run (G22) against W112\'s formula on the run\'s pf stride (1e-6 EUR) and the '
                        'terminal interface_settlement_detail_s31c.json identity (0.01 EUR)'),
            'mechanism': ('decision in the recourse wrapper; certification (or the dynamic cap of an ungated cell below '
                          'the spec cap) sets the certificate length to 0 -> production\'s own exit test ends the loop '
                          'at the end of THIS cycle; the decision file is written once'),
            'only_exit_before_cap': True, 'early_stop': 'ABSENT (validator refuses the key; checklist)',
            'uncertified_form_frozen': DEFINITIONS['uncertified_form']},
        'replay_gate': {
            'applies_to': list(R.GATED_CELLS), 'skipped_for': list(R.UNGATED_CELLS),
            'in_cycle': ('every cycle k <= k0 (the original first residual pass), at the end of the cycle: '
                         + ', '.join(R.REPLAY_GATED_FIELDS) + ' as JSON text against the ORIGINAL record row k; the '
                         'first difference writes the cycle line and ABORTS the cell (no relabelled continuation); at '
                         'k0 the run\'s first residual pass must be k0'),
            'post_run': 'G19: every field of per_cycle_record rows 1..k0 bitwise, adding '
                        + ', '.join(R.REPLAY_POST_RUN_ONLY_FIELDS),
            'overlap_report_only': 'k0+1..N_old: Q_new - Q_old and relative, recorded in-cycle (G23)',
            'expected': ('bitwise: the tail first acts at k0 + 1; production changes since the originals are recorded '
                         '(production_since_originals); the W86 recert showed the pre-tail -> tail replay holds')},
        'holds_after_first_residual_pass': {
            'AA': 'off (production\'s own off branch via all_boyd_pass forced True on a copy), every cycle > k0_run, '
                  'even across a Boyd lapse', 'tail': 'on (applied True; next-state True)',
            'rho': 'frozen (allow_update False; before == after asserted)',
            'boyd_lapse_behaviour': 'recorded; the rule\'s k0 resets strictly; the holds continue',
            'same_as': 'W101 / W105 (the references), from the run\'s first residual pass (Addendum 57 Decision 3(b))'},
        'captures': {
            'w105': ('Q by block and cost component (creep_diagnostic_per_cycle.jsonl q_decomposition); sum |dx| of the '
                     'ESS schedules for 14 families + the raw schedules (ess_schedule_per_cycle.jsonl); every raw Boyd '
                     'field (creep boyd)'),
            'w101_all_block_line': 'recourse_blocks_all.jsonl (48 blocks + SALVAGE, with deltas)',
            't_sum_and_Q_cc': 'resettle_cycle_record.jsonl t_sum, t_by_node, sum_abs_gap_mw_p, Q_cc (and creep lines)',
            'lambda_t': 'interface_duals_per_cycle.jsonl (harness default)',
            'asserted_before_any_solve': 'assert_resettle_preconditions (child) and parent_capture_checklist (launcher)'},
        'definitions': DEFINITIONS,
        'predictions_recorded_before_any_run': PREDICTIONS,
        'gates': {
            'G1-G9, G11, G14, G16': 'as W101 / W105 (W86 evaluation_checks; solve profile 51 x (cycles + 1) + retries; '
                                    'G6 v37; persistence; lambda sidecar)',
            'G13': 'holds inert through the first residual pass, held after; certificate length 10**9 except 0 on the '
                   'rule\'s last cycle', 'G15': 'stopping consistent', 'G17': 'the pure rule v2 replays the in-cycle '
                   'records and the decision', 'G18': 'line fields', 'G19': 'gated: replay 1..k0 bitwise every field',
            'G20': 'production counters', 'G21': 'creep captures complete', 'G22': 't_sum in-cycle == stride formula; '
                   'terminal identity', 'G23': 'overlap recorded (gated) / absent (ungated)', 'scope': GATE_SCOPE,
            'post_run_evaluator_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w118_resettle_checks.py', 'committed_output': checks_file,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()}},
                              'rerun_before_launch': ('the --run mode re-runs sections V, R, O, H, K, P and refuses '
                                                      'unless all hold (the W100 typing test is in the committed output '
                                                      'only)')},
        'verbatim_text': {'quotes': VERBATIM, 'check': verb},
        'labelling_and_identity': {'label': R.LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key, settling_resettle}); every key '
                                                      'without the declaration byte-identical to the pre-W118 harness '
                                                      '(checks K)')},
        'harness_change': ('p515_s44_campaign_harness.py: the keyed option settling_resettle (validator, key branch, '
                           'spec entry, child install / checklist / record / exit 2), minimal and mutually exclusive '
                           'with the three earlier options; every pre-W118 key byte-identical (checks K)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solver': solver,
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'launch_commands_templates': {c: launch_command(c) for c in R.CELL_ORDER},
        'dry_run_template': launch_command('f2_challenger', preconditions_only=True),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised by checks H (real wrappers with the original recorded values; real install); '
                             'THE FIRST REAL CYCLE OF THE F2 CHALLENGER LAUNCH IS THE SMOKE (its in-cycle gate at cycle 1)'),
    }


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
        raise RuntimeError('frozen re-settling stage spec not found')
    return rel, sha, _load(rel)


def freeze_spec(started):
    tag = 'W118-SPEC'
    failures = _common_checks()
    os.makedirs(_abs(ROOT_REL), exist_ok=True)
    existing = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'the stage spec already exists (write-once): {existing}')
    cf = _checks_file_state()
    if not (cf['all_hold_including_typing_test'] is True and cf['committed_clean']
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in K.CODE_PINNED_BY_CHECKS:
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    prov = production_since_originals()
    if not prov['ok']:
        failures.append(f'uncommitted files this run uses: {prov["uncommitted_files_this_run_uses"]}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found in the brief: {verb["found_whitespace_normalised"]}')
    solver = W101L.solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    p_max, p_detail = p_max_from_decisions()
    if not p_detail['equals_hooks_constant']:
        failures.append(f'P_MAX from the decisions {p_max} != the hooks constant {R.P_MAX}')
    pres = {cell: pre_launch_assertion(cell) for cell in R.CELL_ORDER}
    for cell, pre in pres.items():
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    o_section = checks_inline['sections']['O']['result']
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    mem = L.memory_preflight(1)
    wall = wall_time_estimate()
    content = stage_spec_content(checks_inline, cf, post_tests, verb, solver, pres, prov, p_detail, mem, wall, o_section)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (basis: v41 {BASIS_SPECS["stage_spec_v41"]["sha256"][:8]}, '
         f'v39 {BASIS_SPECS["stage_spec_v39"]["sha256"][:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f"[{tag}] P_MAX {p_max} (x0 P_hat {p_detail['sources']['x0']['P_hat']}, unit P_hat "
         f"{p_detail['sources']['unit']['P_hat']}); L {2 * p_max}; tau {SC2.TAU!r}; gap bound {SC2.GAP_BOUND!r}")
    for cell in R.CELL_ORDER:
        o = o_section['cells'][cell]
        _log(f"[{tag}] {cell}: original {o['original']['eval_key'][:8]} k0 {o['k0']} N_old {o['N_old']} cap "
             f"{R.spec_cap(cell)} lattice_legal {o['parts']['lattice_legal']} E/P {o['e_over_p']} I_j {o['I_j']!r} "
             f"key {pres[cell]['resettle_key'][:16]} expected {wall['per_cell'][cell]['expected_h']:.2f} h worst "
             f"{wall['per_cell'][cell]['worst_case_h']:.2f} h")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h; memory "
         f"at freeze (non-gating): available {mem.get('available_gib')} GiB")
    _finish(0, '-- next: --freeze')


def freeze(started):
    tag = 'W118-FREEZE'
    failures = _common_checks()
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        if not _committed_clean(ss_rel):
            failures.append('the stage spec is not committed / clean')
        if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
            failures.append('code changed since the stage spec froze')
    except RuntimeError as error:
        failures.append(str(error))
        ss = None
    for cell in R.CELL_ORDER:
        failures += H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)
        pre = pre_launch_assertion(cell)
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    all_ok = True
    for cell in R.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ss_pin, 'label': R.LABEL, 'cell': cell, 'group': R.CELLS[cell]['group'],
                 'original': ss['cells'][cell]['original'], 'expected_eval_key': pre['resettle_key'],
                 'objective_convention': ss['objective_convention'], 'solve_claim': ss['solve_profile_declared'],
                 'pre_launch_assertion_at_freeze': pre, 'cell_order': list(R.CELL_ORDER)}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=R.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 57 Decisions 2 and 3', f'{BRIEF} Addendum 54 Ruling 2', 'Planner task W118',
                       ss_rel], required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec, ss_pin)
        pre_frozen = pre_launch_assertion(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds']
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
        _log(f"[{tag}]   eval_key={e['eval_key']} eval_dir={e['eval_dir']} cap={spec['cap']}")
        _log(f"[{tag}]   spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}; "
             f"pre-launch on the frozen spec holds={pre_frozen['holds']}")
        _log(f'[{tag}]   LAUNCH: {launch_command(cell, spec_sha)}')
    _log(f"[{tag}]   DRY RUN (F2 challenger): run the LAUNCH line above for f2_challenger with --preconditions-only "
         f"and log run_f2_challenger_preconditions_only.log")
    _finish(0 if all_ok else 1, f'freeze {"OK" if all_ok else "NOT OK"}')


def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W118-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('the stage spec is not committed / clean')
    if ss['code_sha256'] != {rel: _sha(rel) for rel in ss['code_sha256']}:
        failures.append('code changed since the stage spec froze')
    idx = R.CELL_ORDER.index(cell)
    for prev in R.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'cell order {list(R.CELL_ORDER)}: {prev} has no results yet')
    root = campaign_root(cell)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign spec not committed / clean')
    checks = validate_campaign_spec(cell, spec, {'path': ss_rel, 'sha256': ss_sha})
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
        _log(f"[{tag}] every precondition holds: campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; "
             f"stage spec {ss_rel} sha256={ss_sha}; checks per section "
             f"{ {k: v['holds'] for k, v in checks_inline['sections'].items()} }; pre-launch parts {pre['parts']}; "
             f"parent capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; solver "
             f"{solver['resolved'].get('NLP_SOLVER_PATH')} sha256={solver['sha256']}")
        _log(f'[{tag}] STOPPED before the campaign lock and the child (no lock taken, no evaluation, zero solves)')
        _finish(0, 'preconditions-only OK')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cell {cell} eval_key {entry['eval_key']}; cap {spec['cap']}; lock {lock}")
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
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'], 'original_eval_key': R.CELLS[cell][
                   'orig_eval_key'], 'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'gate_detail': detail, 'cell_report': report,
               'pre_launch_assertion': pre, 'parent_capture_checklist': cap_checks, 'memory_preflight_at_run': mem,
               'solver': solver, 'batch_info': batch, 'parent_view': (records[0] or {}).get('parent_view'),
               'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    if summ.get('replay_first_divergence'):
        _log(f"[{tag}] REPLAY DIVERGED -- cell ABORTED: {summ['replay_first_divergence']}")
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] settling v2: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"band_width {report.get('band_width')} s {report.get('s_signed')} t_sum(k*) {report.get('t_sum_k_star')} "
             f"k0_run {report.get('k0_run')} cycles {report.get('cycles_run')}")
    code = 0 if (all(gates.values()) and _guards_ok(g) and isinstance(report, dict) and 'error' not in report) else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


def summarize(started):
    tag = 'W118-SUMMARY'
    ss_rel, ss_sha, ss = load_stage_spec()
    reports, missing = {}, []
    for cell in R.CELL_ORDER:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        if not os.path.isfile(path) or not _committed_clean(os.path.relpath(path, REPO)):
            missing.append(cell)
            continue
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, SUMMARY_FILE)
    if missing or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] cells without committed results {missing} or the summary exists')
        _finish(1)
    i_values = {cell: ss['cells'][cell]['I_j'] for cell in R.CELL_ORDER}
    x0 = {'Q181': ss['x0_comparator']['Q181'], 'band_width': ss['x0_comparator']['band_width'],
          't': ss['x0_comparator']['t_x0_terminal']}
    diffs = differences(reports, i_values, x0)
    scored = score_predictions(reports, diffs)
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'stage_spec': {'path': ss_rel, 'sha256': ss_sha},
           'objective_convention': DEFINITIONS['objective_convention'], 'reports': reports, 'differences': diffs,
           'predictions_scored': scored, 'definitions': DEFINITIONS, 'predictions': PREDICTIONS}
    H._write_once_json(_abs(out_rel), doc)
    H._write_once_json(_abs(os.path.join(ROOT_REL, SUMMARY_MANIFEST)), {out_rel: _sha(out_rel)})
    _log(f'[{tag}] {json.dumps(diffs, default=str)[:4000]}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=R.CELL_ORDER, default=None)
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
