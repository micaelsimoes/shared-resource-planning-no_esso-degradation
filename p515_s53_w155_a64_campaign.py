"""
P5.15 Addendum 64, Planner task W155 -- THE FOUR ADDENDUM 64 CELLS: the 4 per-cell campaign freezes, the frozen spec
`frozen_s53_a64_cells_spec_v1_<sha8>.json` (series frozen_s53_a64_cells_spec; predecessor: the extension-v6 stage spec
frozen_s53_resettle_ext_spec_v3_84775dc4, the last spec of the re-settling series, unchanged), the per-cell run (one cell
per call, launch order enforced), the preconditions-only dry run (every launcher precondition, then a ZERO-SOLVE probe of
the child path through the harness's own configuration hook -- past the floor-row identity check ~3003 -- and STOP), and
the zero-solve scorer (the m = 1.75 claim; the minimum-SoH row; the recorded predictions). A NEW FILE: the W142 launchers
are NOT edited. NOTHING IS RUN by writing this file.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 64 (the m = 1.75 cell; the soh_min 0.50 row: "variant key in new files,
Advisor design check, bitwise gate against the settled C2_calfade unit through 172, frozen spec"); Planner task W155 (the
Advisor's design check H1-H3, section M, the predictions adopted from the Advisor's ranges).

ORDER OF USE (W155):
  1. the router commit (one branch: a W155 declaration dispatches to p515_s53_w155_a64_hooks), its own commit;
  2. the zero-solve checks (`p515_s53_w155_a64_checks.py`) run, output committed;
  3. --freeze-cells, commit; --freeze-spec, commit;
  4. --run --cell e_soh050 --preconditions-only (the dry run; STOP for the Planner);
  5. (the Planner) --run per cell in the order g070_neutrality -> h_x0_m175 -> h_unit_m175 -> e_soh050.

THE CELLS: `p515_s53_w155_a64_hooks.CELLS` (all ungated first evaluations, dynamic cap min(k0_run + 109, 300), criterion
v6, gap clause tau/2, holds after the first residual pass).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-cells | --freeze-spec | --run --cell C --spec-sha256 S [--preconditions-only] | --summarize --after-cell C

Exit codes: 0 done (every stopping gate holds); 1 any other gate / harness / guard / precondition failure; 3 every
stopping gate holds but a recorded STOP condition fired (g070_neutrality not bitwise through 172; e_soh050 bitwise
identical to 3f084f2f past cycle 5; at --summarize: value - I negative at m = 1.75, or Delta value negative) -- STOP FOR
THE PLANNER before the next cell.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import pickle
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W155 Addendum 64 cells launcher (never solves)').install()

import p515_s53_w155_a64_checks as K  # noqa: E402 -- the zero-solve checks (arms its guard; blocks pickle at import)

# The checks module blocks pickle.load / loads at import (its own run). The launcher restores them at once and blocks
# them only where it claims "no model load" (the dry run's child-path probe), with its own verified counters.
pickle.load, pickle.loads = K._PICKLE_ORIG

import p515_s53_w155_a64_hooks as M  # noqa: E402
import p515_s53_w132_resettle_v3_campaign as L132  # noqa: E402 -- generic gates / scorer (arms its guards)
import p515_s53_w139_resettle_v5_campaign as L139  # noqa: E402 -- the v5 gates (arms its guards)
import p515_s53_w142_resettle_v6_campaign as L142  # noqa: E402 -- the v6 gates (arms its guards)
import p515_s53_w142_resettle_ext_v6_campaign as LX  # noqa: E402 -- the extension-v6 gates and readers (arms its guards)
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402
import p515_s53_w142_determinacy as DET  # noqa: E402 -- the v6 scorer
import p515_s46_ageing_mechanism as M46A  # noqa: E402 -- the committed AE / EFC formula `pv` (arms its own guard)
import settling_criterion_v6 as SC6  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L, W101L, L118 = L132.H, L132.L, L132.X, L132.W9, L132.W98L, L132.W101L, L132.L118
V = L132.V
R = V.R


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(LX.GUARDS) + tuple(L142.GUARDS) + tuple(L139.GUARDS)
                        + tuple(L132.GUARDS) + (('s46_ageing_mechanism_imported', M46A.GUARD),
                                                ('w155_parent', PARENT_GUARD)))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = K.OWN_PROCESS_SUBSTRINGS + tuple(L132.OWN_PROCESS_SUBSTRINGS)
STAGE_TEXT = ('P5.15 Addendum 64, W155 -- four cells: the m = 1.75 flexibility-ladder pair (x = 0 and the node-7 unit, '
              'multiplier keyed) and the minimum-SoH row (the C2_calfade unit at minimum_soh 0.50, keyed in the '
              'declaration and applied through production\'s apply_to after the standard variant, the shared floor-row '
              'dict carried to 0.50; and its neutrality cell at 0.70, gated bitwise against 3f084f2f through 172); ungated '
              'first evaluations; the certifying regime held after the run\'s first residual pass (AA off, tight tail on, '
              'rho frozen); settling rule v6 until it certifies or the cap; W105 captures, t_sum, Q_cc, the IPOPT exit, '
              'final attempt tier and final metrics of every block, the SoH-floor sidecar, the terminal ageing trajectory')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.ROOT_REL
SPEC_PREFIX = 'frozen_s53_a64_cells_spec_v1_'
SPEC_SERIES = 'frozen_s53_a64_cells_spec'
SPEC_VERSION = 1
PREDECESSOR = {'series': 'frozen_s53_resettle_ext_spec', 'version': 3, 'path': K.EXT_V6_STAGE_SPEC['path'],
               'sha256': '84775dc41428ba6a6b504c86de728f3d2b44d19354c977c18d3b33f197c76ee6',
               'status': ('the W142 extension v6 stage spec: its 7 cells ran (W152) and its files and specs are '
                          'unchanged; this spec is the first of a new series for the Addendum 64 cells (the v6 rule, the '
                          'v6 state and the extension readers reused by import)')}
CAMPAIGN_IDS = {cell: f'{K.CAMPAIGN_ID_PREFIX}{cell}' for cell in M.CELL_ORDER}
# r1 (superseded before any stage spec was frozen; committed 39859d1b, never run): its campaign specs stay committed and
# unchanged under their own roots; they carry the SAME eval keys as r2 (same candidate x configuration x declaration) and
# differ only in campaign id, working-dir ids and pinned code. Reason: the r1 checks (985c1558) section K counted the
# committed W155 entries without comparing them, so the inline re-run refused the stage-spec freeze.
SUPERSEDED_R1 = {
    'reason': ('r1 checks section K defect: once the r1 campaign specs were committed, key_regression_every_entry_equal '
               'failed in the inline re-run at --freeze-spec (W155-SPEC PRECONDITION FAILED, freeze_spec_launch.log); '
               'the checks were fixed (own W155 entries compared by the formula inside the own root) and everything '
               'downstream re-frozen under r2 names; nothing was overwritten or deleted'),
    'checks_output': K.SUPERSEDED_CHECKS_OUTPUT,
    'campaign_specs': {
        'g070_neutrality': {'path': os.path.join(ROOT_REL, 'campaign_s53_w155_a64_g070_neutrality',
                                                 'campaign_spec_s53_w155_a64_g070_neutrality_444dad64.json'),
                            'sha256': '444dad647adb73fe688a8ca1fb8eadbcc5f503c440b1c15fce6dc5d6520ffa8e'},
        'h_x0_m175': {'path': os.path.join(ROOT_REL, 'campaign_s53_w155_a64_h_x0_m175',
                                           'campaign_spec_s53_w155_a64_h_x0_m175_001b3991.json'),
                      'sha256': '001b399183cbc0828199009c5e2449aa35d7e981126c59d0765b7cc10eccf0af'},
        'h_unit_m175': {'path': os.path.join(ROOT_REL, 'campaign_s53_w155_a64_h_unit_m175',
                                             'campaign_spec_s53_w155_a64_h_unit_m175_fbd0a9a0.json'),
                        'sha256': 'fbd0a9a04a60a551ca218c11cc0f7e65b30d959794169334bcd7f4fb09553061'},
        'e_soh050': {'path': os.path.join(ROOT_REL, 'campaign_s53_w155_a64_e_soh050',
                                          'campaign_spec_s53_w155_a64_e_soh050_140c6bba.json'),
                     'sha256': '140c6bba6bd490dbab69b037a462589ae717aca77718f7172b7abb9629c24808'}},
    'commits': {'checks_output': '985c1558', 'campaign_specs': '39859d1b'},
    'status': 'superseded, never run; never to be launched (this launcher only knows the r2 campaign ids)'}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
SOLVER_PATH = L132.SOLVER_PATH
PYTHON = L132.PYTHON
N_NETWORK_BLOCKS = 48
WALL_LIMIT_H = 4.0
TAU = SC6.TAU
EXTRA_CLEAN_FILES = tuple(dict.fromkeys((SCRIPT_NAME, 'p515_s53_w155_a64_hooks.py', 'p515_s53_w155_a64_checks.py')
                                        + tuple(LX.EXTRA_CLEAN_FILES)))
CODE_PINNED = tuple(dict.fromkeys(K.CODE_PINNED_BY_CHECKS + (SCRIPT_NAME,) + tuple(LX.CODE_PINNED)
                                  + ('p515_s53_w142_resettle_ext_v6_campaign.py',)))
UNIT_REF_EVAL_DIR = M.UNIT_REF_EVAL_DIR
X0_REF = '7aa017f0'              # the settled x0 (W101 d110bd1a, certified at 181): Q(0) of the SoH row
UNIT_REF = 'bd504ecf'            # the settled unit (W101 3f084f2f, certified at 172): the 0.70 reference of the SoH row
E_C2_CALFADE_EVAL_DIR = os.path.join(_P53, 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_e_c2_calfade',
                                     'evals', 'bbd82994a7ddff94_e_c2_calfade')
V6_ROOT = os.path.join(_P53, 'w142_resettle_v6')
V6_H_EVAL_DIRS = {c: os.path.join(V6_ROOT, f'campaign_s53_w142_resettle_v6_{c}', 'evals')
                  for c in ('h_aa8a76d7', 'h_f9eae48f', 'h_50dea31c', 'h_74eda68d')}

# ---- the frozen definitions --------------------------------------------------------------------------------------------
DEFINITIONS = {
    'objective_convention': L132.DEFINITIONS['objective_convention'],
    'per_cell': dict(L132.DEFINITIONS['per_cell'], **{
        'kind': 'soh (the C2_calfade unit at a keyed minimum_soh) | flex (m = 1.75, keyed multiplier)',
        'floor_year': ('W.floor_year_reading at the end cycle (k* or the cap): the first block year whose node-7 floor '
                       'row (cohort 2025) is active in soh_floor_sidecar_baseline.jsonl (active = |SoH - soh_min| <= '
                       '1e-6); None when no floor row is active'),
        'AE_EFC': 'p515_s46_ageing_mechanism.pv over the per-block SoH_used / EFC/day of node 7 cohort 2025 (record)'}),
    'uncertified_form': L132.DEFINITIONS['uncertified_form'],
    'determinacy': L142.DEFINITIONS['determinacy_floor'],
    'claims': {
        'H:m1.75:value_minus_I': {'form': 'value', 'ref': 'h_x0_m175', 'other': 'h_unit_m175', 'I_ref': 0.0,
                                  'I_other': M.UNIT_I, 'net_of_salvage': False, 'claim_type': 'threshold',
                                  'statement': ('flexibility ladder m = 1.75: sign of value - I (node 7, 0.25 MVA / 1 '
                                                'MWh), value = Q(0)_m1.75 - Q(unit)_m1.75')},
        'E:soh050:delta_value_vs_070': {'form': 'value', 'ref': UNIT_REF, 'other': 'e_soh050', 'I_ref': 0.0,
                                        'I_other': 0.0, 'net_of_salvage': False, 'claim_type': 'sign',
                                        'statement': ('minimum SoH row: Delta value = value(0.50) - value(0.70) = '
                                                      'Q(3f084f2f, 0.70, k* 172) - Q(e_soh050); x = 0 is unaffected by '
                                                      'soh_min (Q(0) = the settled x0 Q181 d110bd1a on both sides)')},
        'E:soh050:value_minus_I': {'form': 'value', 'ref': X0_REF, 'other': 'e_soh050', 'I_ref': 0.0,
                                   'I_other': M.UNIT_I, 'net_of_salvage': False, 'claim_type': 'threshold',
                                   'statement': 'the C2_calfade unit at minimum_soh 0.50: sign of value - I'},
        'E:soh050:delta_value_vs_g070_REPORT_BESIDE': {
            'form': 'value', 'ref': 'g070_neutrality', 'other': 'e_soh050', 'I_ref': 0.0, 'I_other': 0.0,
            'net_of_salvage': False, 'claim_type': 'sign',
            'statement': 'Delta value against the neutrality cell (identical to 3f084f2f through 172 by its gate)'},
        'scorer': 'p515_s53_w142_determinacy.score_claim_v6 (W132 score_claim, Addendum 61 floor), views by '
                  'p515_s53_w132_resettle_v3_campaign.view_from_report / reference_views',
    },
}

# ---- predictions, recorded BEFORE any run (Planner task W155, adopting the Advisor's ranges) ----------------------------
PREDICTIONS = {
    'A_m175': {
        'source': 'Planner task W155 (adopting the Advisor\'s ranges)',
        'value_minus_I': {'sign': 'positive', 'point_eur': 15000.0, 'range_eur': [8000.0, 22000.0]},
        'x0_certifies': {'P': 0.9}, 'unit_certifies': {'P_range': [0.55, 0.6]},
        'determinate_only_if_both_certify': True, 'threshold_eur_range': [12400.0, 13200.0],
        'P_determinate_range': [0.35, 0.45],
        'k0_range': [100, 125], 'k_star_range': [130, 185],
        'fallback_sentence_if_within_resolution_or_unit_uncertified': (
            'pays at m = 2; at m = 1.5 and 1.75 within resolution; crossing in (1.5, 2)'),
        'STOP': 'a NEGATIVE value - I at m = 1.75 is a STOP (it contradicts envelope monotonicity)',
    },
    'B_soh050': {
        'source': 'Planner task W155 (adopting the Advisor\'s ranges)',
        'delta_value': {'definition': 'value(0.50) - value(0.70)', 'sign': 'positive', 'point_eur': 12000.0,
                        'range_eur': [4000.0, 22000.0]},
        'more_likely_within_resolution_than_not': True,
        'threshold_eur': 13054.0, 'threshold_formula': 'max(3 x 4,351.4, 2 tau) = 13,054 (4,351.4 = the 3f084f2f band)',
        'floor_never_binds': {'soh_end_2035_range': [0.64, 0.70], 'floor_duals_abs_below': 1e-6},
        'efc_per_day': {'2035': {'point': 0.88, 'range': [0.80, 1.00]}, '2025': {'range': [1.10, 1.30]},
                        '2030': {'range': [1.00, 1.20]}},
        'trajectory': 'diverges from 3f084f2f at cycle 1, and by more than tau/10 by cycle 5',
        'STOP': ['a NEGATIVE Delta value is a STOP',
                 'if the trajectory stays bitwise identical to 3f084f2f past cycle 5, STOP: the key did not reach the '
                 'model (gate G30, exit 3)'],
    },
    'gate_g070': {'statement': 'g070_neutrality reproduces 3f084f2f bitwise through 172 (gate G28)',
                  'STOP': 'a failure is a STOP (exit 3)',
                  'scope_note': ('g070_neutrality is the NEUTRALITY GATE of the router patch and the new wrappers; at '
                                 'the identity value 0.70 it CANNOT prove that 0.50 takes effect -- e_soh050\'s G9s '
                                 '(read-back floor 0.5), G29s (sidecar at 0.5, identity by equality) and G30 (divergence '
                                 'from 3f084f2f by cycle 5) do')},
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


_read_jsonl = L132._read_jsonl
_committed_clean = L132._committed_clean
_decision = L132._decision


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W155] guards {g} {extra_msg}')
    for _n, guard in reversed(GUARDS):
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


def _refuse(tag, failures):
    """Every refusal names the check that failed (one line each), then exits 1."""
    for f in failures:
        _log(f'[{tag} PRECONDITION FAILED] {f}')
    _finish(1, f'REFUSED: {len(failures)} failing check(s)')


# ======================================================================================================================
#  configuration, entries, keys
# ======================================================================================================================
def configuration(cell):
    cfg = K.configuration_now()
    c = M.CELLS[cell]
    return {'name': ('W155 SRP1 ADDENDUM 64 CELLS -- the current production configuration: the case file (AA keep_memory '
                     'declared), the ESS ageing baseline ' + cfg['ess_ageing_baseline_label'] + ' declared (minimum SoH '
                     '0.70 = the file, NOT overridden in the spec), the convergence-depth tight tail DECLARED ENABLED '
                     '(compl_inf_tol 1e-6), post-certification: persist the certified TSO/DSO models only'),
            'arm_label': 's39_D', 'overrides': {},
            'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': cfg['convergence_depth_tail'],
            'note': ('the entry adds settling_resettle (W155 schema; keyed)'
                     + (f"; model_variant = the {c['arm']} arm (keyed through the base key); minimum_soh "
                        f"{c['minimum_soh']} keyed in the declaration and applied after the variant by production's "
                        f"apply_to" if c['kind'] == 'soh' else
                        f"; flexibility price x {c['flex_price_multiplier']} (keyed through the base key)")
                     + '; cap 300 (the rule stops at min(k0_run + 109, 300)); persist_certified_models; concurrency 1')}


def entries(cell):
    c = M.CELLS[cell]
    opts = {'investment_year': c['investment_year'],
            'post_certification': {'persist_certified_models': True, 'hull_polish': False, 'reference': None},
            'settling_resettle': M.declaration_for(cell)}
    if c['kind'] == 'soh':
        opts['model_variant'] = M.arm_variant(cell)
    else:
        opts['flex_price_multiplier'] = c['flex_price_multiplier']
    return [(cell, dict(c['nodes']), opts)]


def expected_keys(cell):
    c = M.CELLS[cell]
    kw = K.key_kwargs(cell)
    base = H.evaluation_key(c['candidate_key'], {}, **kw)
    rkey = H.evaluation_key(c['candidate_key'], {}, settling_resettle=M.declaration_for(cell), **kw)
    return {'base_key_current_configuration': base, 'resettle_key': rkey, 'candidate_key': c['candidate_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The resettle key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it appears in
    no committed campaign spec OUTSIDE this stage root (CLAUDE.md: a pre-run check that scans committed artefacts excludes
    the run's own); it differs from the base key; the working dirs are absent."""
    k = expected_keys(cell)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    eval_dir_name = H.eval_dir_name(k['resettle_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['resettle_key'])
    work = L._work_dir()
    entry = next((x for x in (spec or {}).get('candidates') or [] if x['label'] == cell), None)
    parts = {
        'resettle_key_differs_from_base': k['resettle_key'] != k['base_key_current_configuration'],
        'resettle_key_absent_from_committed_specs_outside_the_own_root': k['resettle_key'] not in committed,
        'base_key_absent_from_committed_specs_outside_the_own_root': k['base_key_current_configuration'] not in committed,
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['resettle_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    detail = {}
    if not parts['resettle_key_absent_from_committed_specs_outside_the_own_root']:
        detail['holders_outside'] = committed.get(k['resettle_key'])
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL], **detail}


def validate_campaign_spec(cell, spec):
    ref = _load(L132.W101_X0_SPEC_REL)          # the W101 x0 cell: the same C2 + tight-tail configuration
    want = configuration(cell)
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    c = M.CELLS[cell]
    mv = M.arm_variant(cell)
    checks = {f'configuration:{k}': cfg.get(k) == want[k] for k in
              ('arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
               'ess_ageing_baseline_label', 'convergence_depth_tail')}
    checks.update({
        'configuration:case_file_sha256': cfg.get('case_file_sha256') == K.K132.CASE_FILE_SHA256,
        'configuration:ess_params_file_is_the_pinned_file': (cfg.get('ess_params_file') or {}).get('sha256')
        == M.ESS_PARAMS_SHA256,
        'configuration:minimum_soh_070_not_overridden': (cfg.get('ess_ageing_baseline') or {}).get('minimum_soh')
        == M.FILE_MINIMUM_SOH,
        'configuration:as_the_w101_x0_cell': all(cfg.get(k) == ref['configuration'].get(k) for k in (
            'arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
            'ess_ageing_baseline_label', 'convergence_depth_tail', 'apply_rho', 'full_diagnostics_in_rows',
            'case_file_sha256')),
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == cell,
        'entry_key_is_the_cell_candidate': e.get('key') == c['candidate_key'],
        'entry_canonical_nodes_and_year': (e.get('canonical') or {}).get('investment_year') == c['investment_year']
        and {int(k): tuple(map(float, v)) for k, v in ((e.get('canonical') or {}).get('nodes') or {}).items()}
        == c['nodes'],
        'entry_overrides_empty': e.get('overrides') == {},
        'entry_post_certification_persist_only': e.get('post_certification') == {
            'persist_certified_models': True, 'hull_polish': False, 'reference': None},
        'entry_resettle_is_the_w155_declaration': e.get('settling_resettle') == M.declaration_for(cell),
        'entry_model_variant_is_the_arm': (json.dumps(e.get('model_variant'), sort_keys=True)
                                           == json.dumps(mv, sort_keys=True)),
        'variant_label_at_spec_and_entry_iff_variant': (
            (mv is None and 'model_variant_label' not in e and 'model_variant_label' not in spec)
            or (mv is not None and e.get('model_variant_label') == H.MODEL_VARIANT_LABEL
                and spec.get('model_variant_label') == H.MODEL_VARIANT_LABEL)),
        'entry_flex_multiplier_iff_h_cell': ((c['kind'] == 'flex' and e.get('flex_price_multiplier') == M.FLEX_M
                                              and e.get('flex_price_label') == H.FLEX_PRICE_LABEL
                                              and spec.get('flex_price_label') == H.FLEX_PRICE_LABEL)
                                             or (c['kind'] == 'soh' and 'flex_price_multiplier' not in e
                                                 and 'flex_price_label' not in e and 'flex_price_label' not in spec)),
        'minimum_soh_only_in_a_soh_declaration': (('minimum_soh' in (e.get('settling_resettle') or {}))
                                                  == (c['kind'] == 'soh')
                                                  and 'minimum_soh' not in {k for k in e if k != 'settling_resettle'}),
        'entry_has_no_other_continuation': not any(x in e for x in ('settling_continuation', 'certification_continuation',
                                                                    'settling_extension', 'release_solution_bookkeeping',
                                                                    'interface_deviation_premium')),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_IDS[cell],
        'cap_is_the_cell_cap': spec.get('cap') == M.spec_cap(cell),
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_10': spec.get('required_consecutive_cycles') == 10,
        'bar_window_as_w101': spec.get('bar_window_cycles') == ref['bar_window_cycles'],
        'thread_caps_as_w101': spec.get('thread_caps') == ref['thread_caps'],
        'interpreter_as_w101': spec.get('interpreter') == ref['interpreter'],
        'solver_path_as_w101': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
        'cell_recorded': (spec.get('extra') or {}).get('cell') == cell,
    })
    return checks


def parent_capture_checklist(cell, spec):
    """The child's capture checklist (M.assert_resettle_preconditions), asserted in the PARENT too (before the lock)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = M.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        return True, checks
    except Exception as error:  # noqa: BLE001
        return False, {'error': f'{type(error).__name__}: {error}'}


def harness_routed():
    sha = H.sha256_file(H.HARNESS_PATH)
    try:
        routed = all(H.resettle_hooks_module(M.declaration_for(c)) is M for c in M.CELL_ORDER)
    except Exception:  # noqa: BLE001
        routed = False
    return {'ok': bool(sha == K.HARNESS_POST_W155_SHA256 and routed), 'harness_sha256': sha,
            'expected_post_w155_sha256': K.HARNESS_POST_W155_SHA256, 'routes_every_w155_declaration_here': routed}


# ======================================================================================================================
#  walls
# ======================================================================================================================
def _eval_dir_of(evals_root_rel):
    hits = sorted(os.listdir(_abs(evals_root_rel)))
    if len(hits) != 1:
        raise RuntimeError(f'{evals_root_rel}: expected one eval dir, found {hits}')
    return os.path.join(evals_root_rel, hits[0])


def wall_time_estimate():
    """Per cell. s = the mean per-cycle wall of the measured run of the same candidate under the same configuration at
    concurrency 1: the SoH cells -> the W152 e_c2_calfade run (the C2_calfade unit, bbd82994); h_x0_m175 -> the v6 x0
    runs at m 1.5 and 2 (h_aa8a76d7, h_50dea31c), the slower of the two; h_unit_m175 -> the v6 unit runs at m 1.5 and 2
    (h_f9eae48f, h_74eda68d), the slower. Expected cycles: the SoH cells 172 (e_c2_calfade's k*), the H cells the upper
    end of the predicted k* range (185). Worst = the CEILING 300 (no k0_run can raise the cap above it). Plus the measured
    overhead of the e_c2_calfade campaign (campaign wall - sum of iteration walls) and 300 s for the inline checks and
    the parent's preconditions. A cell whose worst case exceeds 4 h refuses the freeze."""
    def mean_walls(eval_dir_rel):
        w = L132._iteration_walls(eval_dir_rel)
        return sum(w) / len(w), len(w), sum(w)
    e_s, e_n, e_sum = mean_walls(E_C2_CALFADE_EVAL_DIR)
    e_root = os.path.dirname(os.path.dirname(E_C2_CALFADE_EVAL_DIR))
    overhead = (_load(os.path.join(e_root, RESULTS_FILE))['wall_clock_s'] - e_sum) + 300.0
    h = {c: mean_walls(_eval_dir_of(p)) for c, p in V6_H_EVAL_DIRS.items()}
    per = {}
    for cell in M.CELL_ORDER:
        c = M.CELLS[cell]
        if c['kind'] == 'soh':
            s, basis, exp_cycles = e_s, f'W152 e_c2_calfade ({E_C2_CALFADE_EVAL_DIR}): {e_n} cycles', 172
        elif cell == 'h_x0_m175':
            s = max(h['h_aa8a76d7'][0], h['h_50dea31c'][0])
            basis, exp_cycles = 'v6 x0 at m 1.5 (h_aa8a76d7) and m 2 (h_50dea31c), the slower', 185
        else:
            s = max(h['h_f9eae48f'][0], h['h_74eda68d'][0])
            basis, exp_cycles = 'v6 unit at m 1.5 (h_f9eae48f) and m 2 (h_74eda68d), the slower', 185
        cap = M.spec_cap(cell)
        per[cell] = {'basis': basis, 's_per_cycle_estimate': s, 'expected_cycles': exp_cycles, 'worst_cycles': cap,
                     'expected_h': (exp_cycles * s + overhead) / 3600.0, 'worst_case_h': (cap * s + overhead) / 3600.0}
    return {'basis': wall_time_estimate.__doc__, 'overhead_s_per_cell': overhead, 'per_cell': per,
            'h_cell_bases_s_per_cycle': {k: v[0] for k, v in h.items()},
            'total_expected_h': sum(v['expected_h'] for v in per.values()),
            'total_worst_case_h': sum(v['worst_case_h'] for v in per.values()),
            'single_run_over_4h': {c: v['worst_case_h'] for c, v in per.items() if v['worst_case_h'] > WALL_LIMIT_H}}


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
#  post-run gates
# ======================================================================================================================
def hold_checks(cell, eval_dir, rec):
    lines = _read_jsonl(os.path.join(eval_dir, M.CYCLE_FILE))
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
        'first_pass_equals_the_rule_first_k0_v2': fp == summ.get('first_k0_v2'),
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
            x.get('certificate_length_in_force_at_cycle_end') == M.CERTIFICATION_DISABLED_THRESHOLD
            or (x['cycle'] == last and x.get('certificate_length_in_force_at_cycle_end') == M.SETTLING_END_THRESHOLD
                and summ.get('stopped_by') in ('settling_rule', 'rule_cap')) for x in lines),
        'summary_ok': summ.get('ok') is True,
        'summary_schema_is_w155': summ.get('schema') == M.SCHEMA,
        'decision_present': bool(dec),
    }
    return all(parts.values()), {'parts': parts, 'first_pass': fp, 'rho_at_first_pass': rho_fp}


def stopping_check(cell, rec, eval_dir):
    summ = rec.get('settling_resettle_summary') or {}
    dec = _decision(eval_dir) or {}
    k = rec.get('cycles_run')
    cap = M.spec_cap(cell)
    if dec.get('status') == 'certified':
        ok = k == dec.get('k_star') and summ.get('stopped_by') == 'settling_rule'
    elif dec.get('status') == 'uncertified':
        ok = k == dec.get('k_cap') and summ.get('stopped_by') == ('rule_cap' if k < cap else 'cap')
    else:
        ok = False
    return bool(ok), {'cycles_run': k, 'decision_status': dec.get('status'), 'k_star': dec.get('k_star'),
                      'k_cap': dec.get('k_cap'), 'stopped_by': summ.get('stopped_by'), 'spec_cap': cap}


def settling_replay_check(cell, eval_dir):
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, M.CYCLE_FILE))}
    rule = K.K142.pure_rule(M.declaration_for(cell))
    pure = []
    for r in rows:
        ln = lines.get(r['cycle']) or {}
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok']), ln.get('t_sum'),
                                 bool(ln.get('all_clean_k'))))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    keys = L142.DECISION_KEYS_REPLAYED
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k)) for k in keys),
             'decision_file_present': bool(dec), 'decision_is_version_6': dec.get('version') == 6}
    return all(parts.values()), {'parts': parts}


def _end_cycle(eval_dir, rec):
    dec = _decision(eval_dir) or {}
    end = dec.get('k_star') if dec.get('status') == 'certified' else dec.get('k_cap')
    return end if end is not None else rec.get('cycles_run')


def floor_capture_check(cell, eval_dir, rec):
    """G26: the SoH-floor sidecar holds one line per cycle and the end cycle's line carries every block of the node-7
    2025 cohort (the floor-year reading)."""
    path = os.path.join(eval_dir, M.FLOOR_SIDECAR_FILE)
    if not os.path.isfile(path):
        return False, {'error': f'{M.FLOOR_SIDECAR_FILE} absent'}
    lines = _read_jsonl(path)
    reading, ok = M.floor_year_reading(lines, _end_cycle(eval_dir, rec), node=M.UNIT_NODE)
    return bool(ok and len(lines) == (rec.get('cycles_run') or -1)), {'reading': reading, 'n_lines': len(lines)}


def flex_readback_check(rec):
    """G10 (H cells): the multiplier recorded is 1.75, applied and read back pre-run (probe DSO blocks) and from the run's
    own DSO models (post-run), all_match both."""
    ap = rec.get('flex_price_applied_in_child') or {}
    pre = (ap.get('readback_pre_run') or {})
    post = rec.get('flex_price_readback_terminal') or {}
    parts = {'record_multiplier_1_75': rec.get('flex_price_multiplier') == M.FLEX_M,
             'applied_checks_all_true': bool(ap.get('checks')) and all(v is True for v in ap['checks'].values()),
             'pre_run_readback_all_match': pre.get('all_match') is True,
             'terminal_readback_all_match': post.get('all_match') is True}
    return all(parts.values()), {'parts': parts, 'n_flex_coefficients_pre': pre.get('n_flex_coefficients'),
                                 'max_rel_dev_pre': pre.get('max_rel_dev')}


def bitwise_vs_unit(eval_dir, through):
    """The 3f084f2f trajectory equality through `through` (W.trajectory_equality: gross_operational_cost compared as JSON
    text every cycle; every other shared field, report-only)."""
    run_rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    ref_rows = _read_jsonl(_abs(os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl')))
    eq = M.trajectory_equality(run_rows, ref_rows, through)
    return eq, run_rows, ref_rows


def g070_neutrality_gate(eval_dir):
    """G28 (g070_neutrality only): bitwise against 3f084f2f through 172 (the e_c2_calfade G28 precedent)."""
    eq, run_rows, ref_rows = bitwise_vs_unit(eval_dir, M.UNIT_REF['k_star'])
    return bool(eq['reproduced']), {**eq, 'reference': {'eval_dir': UNIT_REF_EVAL_DIR,
                                                        'per_cycle_record_sha256': _sha(os.path.join(
                                                            UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl'))},
                                    'run_last_cycle': run_rows[-1]['cycle'] if run_rows else None}


def soh050_divergence_gate(eval_dir):
    """G30 (e_soh050 only): the trajectory must NOT stay bitwise identical to 3f084f2f past cycle 5 (else the key did not
    reach the model: STOP). Reported beside: the first differing cycle and |Q_run - Q_ref| for cycles 1..5 against
    tau/10 (the prediction: diverges at cycle 1, by more than tau/10 by cycle 5)."""
    eq, run_rows, ref_rows = bitwise_vs_unit(eval_dir, 6)
    run = {r['cycle']: r for r in run_rows}
    ref = {r['cycle']: r for r in ref_rows}
    d = {k: (run[k]['gross_operational_cost'] - ref[k]['gross_operational_cost'])
         for k in range(1, 6) if k in run and k in ref}
    fd = eq['first_difference']
    diverged_by_5 = fd is not None and fd.get('cycle') is not None and fd['cycle'] <= 5
    return bool(diverged_by_5), {'first_difference': fd, 'dQ_cycles_1_5': d,
                                 'max_abs_dQ_1_5': max((abs(v) for v in d.values()), default=None),
                                 'tau_over_10': TAU / 10.0,
                                 'prediction_diverges_at_cycle_1': bool(fd and fd.get('cycle') == 1),
                                 'prediction_beyond_tau_over_10_by_cycle_5': bool(d) and max(abs(v) for v in d.values())
                                 > TAU / 10.0,
                                 'rule': 'STOP iff the run equals 3f084f2f bitwise through cycle 6 (past cycle 5)'}


GATE_SCOPE = {
    'G8_persistence_on_production_certificate': 'every cell (Addendum 59 rule, as the v6 campaigns)',
    'G9_ess_ageing_readback': 'the H cells only (no model variant: the harness reads the ESS ageing baseline back)',
    'G9s_soh_readback_at_the_declared_floor': 'the SoH cells only (M.soh_readback_gate)',
    'G10_flex_price_readback': 'the H cells only',
    'G19_replay': 'SKIPPED for every cell (ungated first evaluations)',
    'G23_overlap_recorded': 'every cell: none (asserted empty)',
    'G26_floor_sidecar_end_cycle': 'every cell (node 7, cohort 2025)',
    'G27_ageing_trajectory_terminal': 'the SoH cells (the record carries it; the H unit reports it beside)',
    'G29s_soh_sidecar_and_identity_by_equality': 'the SoH cells only (M.soh_sidecar_gate)',
    'G28_g070_bitwise_vs_3f084f2f_through_172': 'g070_neutrality only; failure = STOP (exit 3)',
    'G30_e_soh050_diverges_from_3f084f2f_by_cycle_5': 'e_soh050 only; bitwise identical past cycle 5 = STOP (exit 3)',
    'G6_v37_optimal_and_four_metrics': 'every cell; REPORTED, does NOT stop the campaign (Planner ruling at W128)',
    'all_other_gates': 'every cell',
}
NON_STOPPING_GATES = ('G6_v37_optimal_and_four_metrics', 'G28_g070_bitwise_vs_3f084f2f_through_172',
                      'G30_e_soh050_diverges_from_3f084f2f_by_cycle_5')
PREDICTION_GATES = ('G28_g070_bitwise_vs_3f084f2f_through_172', 'G30_e_soh050_diverges_from_3f084f2f_by_cycle_5')


def cell_gates(cell, entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    c = M.CELLS[cell]
    decl = M.declaration_for(cell)
    gates, detail = {}, {'exit_code': exit_code, 'gate_scope': GATE_SCOPE}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_resettle_summary'] = rec.get('settling_resettle_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys(cell)['resettle_key']
    ch, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': ch, 'detail': d}
    gates['G3_append_reconcile'] = ch.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = ch.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W101L.solve_profile_check(rec, len(records))
    g6 = W98L.g6_v37_evaluate_records(records, rec.get('cycles_run'), b=N_NETWORK_BLOCKS)
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = ch.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_on_production_certificate'], detail['G8'] = V6.persistence_check_production_certificate(
        rec, eval_dir)
    if c['kind'] == 'soh':
        gates['G9s_soh_readback_at_the_declared_floor'], detail['G9s'] = M.soh_readback_gate(rec, decl)
    else:
        gates['G9_ess_ageing_readback'] = ch.get('ess_ageing_readback_all_match', False)
        gates['G10_flex_price_readback'], detail['G10'] = flex_readback_check(rec)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_first_pass_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_v6_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = L142.line_fields_check(eval_dir)
    detail['G19'] = {'skipped': 'ungated first evaluation: no replay reference'}
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = L118.creep_capture_check(eval_dir, rec, rows)
    gates['G22_t_sum_in_cycle_equals_stride_and_terminal'], detail['G22'] = L118.t_sum_check(eval_dir)
    ov = (rec.get('settling_resettle_summary') or {}).get('overlap_k0_plus_1_to_N_old') or []
    gates['G23_overlap_recorded'] = len(ov) == 0
    try:
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'], detail['G24'] = L132.exit_crosscheck(
            eval_dir, rec=rec)
    except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'] = False
        detail['G24'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    try:
        gates['G24b_clean_capture_equals_the_solve_records_and_esso_logs'], detail['G24b'] = L139.clean_crosscheck(
            eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
        gates['G24b_clean_capture_equals_the_solve_records_and_esso_logs'] = False
        detail['G24b'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    gates['G25_record_status_follows_the_settling_decision'], detail['G25'] = L132.status_label_check(rec, eval_dir)
    gates['G26_floor_sidecar_end_cycle'], detail['G26'] = floor_capture_check(cell, eval_dir, rec)
    series, s_ok = LX.ageing_series(rec)
    detail['G27'] = series
    if c['kind'] == 'soh':
        gates['G27_ageing_trajectory_terminal'] = s_ok
        g_report = json.load(open(os.path.join(eval_dir, 'g_s39_D.json')))
        side = _read_jsonl(os.path.join(eval_dir, M.FLOOR_SIDECAR_FILE))
        gates['G29s_soh_sidecar_and_identity_by_equality'], detail['G29s'] = M.soh_sidecar_gate(
            side, g_report, decl, rec.get('cycles_run'), rec.get('settling_resettle_summary') or {})
    if cell == M.NEUTRALITY_CELL:
        try:
            gates['G28_g070_bitwise_vs_3f084f2f_through_172'], detail['G28'] = g070_neutrality_gate(eval_dir)
        except Exception as error:  # noqa: BLE001
            gates['G28_g070_bitwise_vs_3f084f2f_through_172'] = False
            detail['G28'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    if cell == M.SOH050_CELL:
        try:
            gates['G30_e_soh050_diverges_from_3f084f2f_by_cycle_5'], detail['G30'] = soh050_divergence_gate(eval_dir)
        except Exception as error:  # noqa: BLE001
            gates['G30_e_soh050_diverges_from_3f084f2f_by_cycle_5'] = False
            detail['G30'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return gates, detail, rec


# ======================================================================================================================
#  the per-cell report
# ======================================================================================================================
def cell_report(cell, eval_dir, rec):
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, M.CYCLE_FILE))}
    dec = _decision(eval_dir) or {}
    c = M.CELLS[cell]
    summ = rec.get('settling_resettle_summary') or {}
    q = {k: r['gross_operational_cost'] for k, r in rows.items()}
    last = max(rows)
    end = dec.get('k_star') if dec.get('status') == 'certified' else dec.get('k_cap')
    end = end if end is not None else last
    steps = [q[k] - q[k - 1] for k in sorted(q) if (k - 1) in q and q[k] is not None and q[k - 1] is not None]
    fp = summ.get('first_residual_pass_run')
    after = [x for k, x in lines.items() if fp is not None and k > fp]
    win = [k for k in range(end - L132.PF_SLOPE_WINDOW + 1, end + 1) if k in rows
           and rows[k].get('boyd_pf_primal_ratio') is not None]
    certified = dec.get('status') == 'certified'
    q_end = dec.get('Q_k_star') if certified else dec.get('Q_at_cap')
    t_end = dec.get('t_sum_k_star') if certified else dec.get('t_sum_at_cap')
    tol = (rows.get(end) or {}).get('objective_tolerance')
    rep = {'cell': cell, 'item': c['item'], 'kind': c['kind'], 'arm': c['arm'], 'minimum_soh': c['minimum_soh'],
           'flex_price_multiplier': c['flex_price_multiplier'], 'claim_group': M.GROUP_OF_ITEM[c['item']],
           'gated': False, 'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
           'candidate_canonical': rec.get('candidate_canonical'),
           'k0_run': fp, 'first_k0_v2': summ.get('first_k0_v2'), 'criterion_version': dec.get('version'),
           'rule_cap': dec.get('cap'), 'cap_ceiling': c['cap_ceiling'], 'cycles_run': rec.get('cycles_run'),
           'status': dec.get('status'), 'record_status': rec.get('status'), 'k_star': dec.get('k_star'),
           'end_cycle': end, 'branch': dec.get('branch'), 'band': dec.get('band'), 'band_width': dec.get('band_width'),
           'range_over_tau': dec.get('range_over_tau'), 'Q_k_star': dec.get('Q_k_star'),
           't_sum_k_star': dec.get('t_sum_k_star'), 'Q_cc_k_star': dec.get('Q_cc_k_star'),
           'Q_end': q_end, 't_sum_end': t_end, 'Q_cc_end': (q_end + t_end) if (q_end is not None and t_end is not None)
           else None, 'Q_net_end': (rows.get(end) or {}).get('recourse'),
           'terminal_salvage_value_end': (rows.get(end) or {}).get('terminal_salvage_value'),
           'T': dec.get('T'), 'A': dec.get('A'), 'P_hat': dec.get('P_hat'),
           'lapse_events': dec.get('lapse_events'), 'gap_refusals': dec.get('gap_refusals'),
           'non_clean_cycles': dec.get('non_clean_cycles'), 'non_optimal_cycles': summ.get('non_optimal_cycles'),
           'acceptable_clean_exits': L139._acceptable_clean_exits(eval_dir), 'W': dec.get('W'), 'window': dec.get('window'),
           'vetoes': dec.get('vetoes'), 'n_vetoes': dec.get('n_vetoes'), 'swing_floor': dec.get('swing_floor'),
           'turning_point_floor_rejections': dec.get('turning_point_floor_rejections'),
           'window_all_clean': dec.get('window_all_clean'), 'out_of_window_reads': dec.get('out_of_window_reads'),
           'terminal_step_abs': abs(steps[-1]) if steps else None,
           'terminal_step_over_EPS0': (abs(steps[-1]) / SC6.EPS0) if steps else None,
           'terminal_step_to_threshold_ratio_end_cycle': ((rows[end]['objective_change_abs'] / tol)
                                                          if (rows.get(end) or {}).get('objective_change_abs')
                                                          is not None and tol else None),
           'rule_ten_last_cycle': ((rows[last]['objective_change_abs'] / rows[last]['objective_tolerance'])
                                   if rows[last].get('objective_change_abs') is not None
                                   and rows[last].get('objective_tolerance') else None),
           'boyd_lapses_after_k0': sum(1 for x in after if not x.get('boyd_k')),
           'pf_primal_slope_last_50': L132._ols_slope(win, [rows[k]['boyd_pf_primal_ratio'] for k in win]),
           'pf_primal_slope_window': [win[0], win[-1]] if win else None,
           'objective_convention': DEFINITIONS['objective_convention']}
    if not certified:
        rep.update({k: dec.get(k) for k in ('k_cap', 'reasons', 'band_window', 'drift_rate_mean_dQ_last_25',
                                            'dQ_cc_rate_mean_last_25', 'Q_at_cap', 't_sum_at_cap', 'Q_cc_at_cap',
                                            'gap_clause_refused_at_cap')})
        rep['t_by_node_at_cap'] = (lines.get(end) or {}).get('t_by_node')
        rep['gap_refused'] = bool(dec.get('gap_refusals'))
        rep['label'] = L132.GAP_REFUSED_LABEL if rep['gap_refused'] else 'uncertified at the cap'
    floor_path = os.path.join(eval_dir, M.FLOOR_SIDECAR_FILE)
    if os.path.isfile(floor_path):
        rep['floor_reading'] = M.floor_year_reading(_read_jsonl(floor_path), end, node=M.UNIT_NODE)[0]
        rep['floor_year'] = rep['floor_reading'].get('floor_year')
        per = rep['floor_reading'].get('per_block') or []
        rep['floor_duals_abs_max_node7_cohort2025'] = max((abs(b['dual']) for b in per if b.get('dual') is not None),
                                                          default=None)
    series, ok = LX.ageing_series(rec)
    rep.update({'ageing_series': series, 'ageing_series_ok': ok,
                'AE': M46A.pv(series['soh_used']) if ok else None, 'EFC': M46A.pv(series['efc_per_day']) if ok else None})
    rep['view'] = L132.view_from_report(rep)
    return rep


# ======================================================================================================================
#  the scorer (pure below the loaders)
# ======================================================================================================================
def _claim(cid):
    d = DEFINITIONS['claims'][cid]
    return {'claim_id': cid, 'item': cid.split(':')[0], 'statement': d['statement'], 'form': d['form'],
            'claim_type': d['claim_type'], 'net_of_salvage': d['net_of_salvage'], 'I_ref': d['I_ref'],
            'I_other': d['I_other']}


def _view(side, views, refs):
    return refs[side] if side in refs else views.get(side, {})


def score_all(views, reports, refs):
    """The four claims (DEFINITIONS['claims']) and the recorded predictions A and B, with the STOP conditions. Pure."""
    claims = {}
    for cid, d in DEFINITIONS['claims'].items():
        if cid == 'scorer':
            continue
        claims[cid] = DET.score_claim_v6(_claim(cid), _view(d['ref'], views, refs), _view(d['other'], views, refs))
    out = {'claims': claims}
    # ---- A
    a = claims['H:m1.75:value_minus_I']
    vx, vu = views.get('h_x0_m175') or {}, views.get('h_unit_m175') or {}
    if 'd_Q' in a:
        pa = PREDICTIONS['A_m175']
        both = vx.get('status') == 'certified' and vu.get('status') == 'certified'
        gross = a.get('gross') or {}
        determinate = a.get('verdict') == 'determinate'
        rx, ru = reports.get('h_x0_m175') or {}, reports.get('h_unit_m175') or {}
        out['A'] = {
            'value_minus_I': a['d_Q'], 'value_minus_I_Qcc': a.get('d_Qcc'),
            'value': (vx['Q'] - vu['Q']) if (vx.get('Q') is not None and vu.get('Q') is not None) else None,
            'sign_positive_as_predicted': a['d_Q'] > 0,
            'within_predicted_range': pa['value_minus_I']['range_eur'][0] <= a['d_Q'] <= pa['value_minus_I']['range_eur'][1],
            'x0_certified': vx.get('status') == 'certified', 'unit_certified': vu.get('status') == 'certified',
            'both_certified': both, 'verdict': a.get('verdict'), 'threshold': gross.get('threshold') or gross.get('bar'),
            'threshold_within_predicted_range': (gross.get('threshold') is not None and pa['threshold_eur_range'][0]
                                                 <= gross['threshold'] <= pa['threshold_eur_range'][1]),
            'k0': {'x0': rx.get('k0_run'), 'unit': ru.get('k0_run'), 'range': pa['k0_range']},
            'k_star': {'x0': rx.get('k_star'), 'unit': ru.get('k_star'), 'range': pa['k_star_range']},
            'fallback_sentence_applies': (not determinate) or not (vu.get('status') == 'certified'),
            'fallback_sentence': pa['fallback_sentence_if_within_resolution_or_unit_uncertified'],
            'STOP_negative_value_minus_I': a['d_Q'] < 0}
    else:
        out['A'] = {'scored': False, 'reason': 'a cell has no result yet'}
    # ---- B
    b = claims['E:soh050:delta_value_vs_070']
    rs = reports.get('e_soh050') or {}
    if 'd_Q' in b:
        pb = PREDICTIONS['B_soh050']
        gross = b.get('gross') or {}
        series = rs.get('ageing_series') or {}
        efc = dict(zip([str(y) for y in series.get('block_years') or []], series.get('efc_per_day') or []))
        soh_end = dict(zip([str(y) for y in series.get('block_years') or []], series.get('soh_end') or []))
        out['B'] = {
            'delta_value': b['d_Q'], 'delta_value_Qcc': b.get('d_Qcc'),
            'sign_positive_as_predicted': b['d_Q'] > 0,
            'within_predicted_range': pb['delta_value']['range_eur'][0] <= b['d_Q'] <= pb['delta_value']['range_eur'][1],
            'verdict': b.get('verdict'), 'threshold': gross.get('threshold') or gross.get('bar'),
            'within_resolution_as_more_likely_predicted': b.get('verdict') != 'determinate',
            'floor_year': rs.get('floor_year'),
            'floor_duals_abs_max': rs.get('floor_duals_abs_max_node7_cohort2025'),
            'floor_never_binds_as_predicted': (rs.get('floor_year') is None and rs.get(
                'floor_duals_abs_max_node7_cohort2025') is not None
                and rs['floor_duals_abs_max_node7_cohort2025'] < pb['floor_never_binds']['floor_duals_abs_below']),
            'soh_end_2035': soh_end.get('2035'),
            'soh_end_2035_within_predicted_range': (soh_end.get('2035') is not None
                                                    and pb['floor_never_binds']['soh_end_2035_range'][0]
                                                    <= soh_end['2035'] <= pb['floor_never_binds']['soh_end_2035_range'][1]),
            'efc_per_day': efc,
            'efc_within_predicted_ranges': {y: (efc.get(y) is not None and r['range'][0] <= efc[y] <= r['range'][1])
                                            for y, r in pb['efc_per_day'].items()},
            'value_minus_I_soh050': {k: claims['E:soh050:value_minus_I'].get(k) for k in ('verdict', 'd_Q', 'd_Qcc')},
            'delta_vs_g070_report_beside': {k: claims['E:soh050:delta_value_vs_g070_REPORT_BESIDE'].get(k)
                                            for k in ('verdict', 'd_Q')},
            'STOP_negative_delta_value': b['d_Q'] < 0}
    else:
        out['B'] = {'scored': False, 'reason': 'a cell has no result yet'}
    out['STOP'] = sorted(k for k, v in (('A: value - I negative at m = 1.75', out['A'].get('STOP_negative_value_minus_I')),
                                        ('B: Delta value negative', out['B'].get('STOP_negative_delta_value'))) if v)
    return out


def reference_views():
    return L132.reference_views()


# ======================================================================================================================
#  self-tests of the evaluators (zero solves; committed inputs)
# ======================================================================================================================
def evaluator_self_tests():
    out = {}
    try:
        unit = _read_jsonl(_abs(os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl')))
        tmp = tempfile.mkdtemp(prefix='w155_selftest_')
        try:
            def write(rows):
                with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
                    for r in rows:
                        handle.write(json.dumps(r) + '\n')
            write(unit)
            g28_same, _ = g070_neutrality_gate(tmp)
            g30_same, _ = soh050_divergence_gate(tmp)
            t = copy.deepcopy(unit)
            t[150]['gross_operational_cost'] = math.nextafter(t[150]['gross_operational_cost'], -math.inf)
            write(t)
            g28_planted, d28 = g070_neutrality_gate(tmp)
            t = copy.deepcopy(unit)
            t[0]['gross_operational_cost'] += 1000.0
            write(t)
            g30_div, d30 = soh050_divergence_gate(tmp)
            t = copy.deepcopy(unit)
            t[6]['gross_operational_cost'] += 1000.0
            write(t)
            g30_late, _ = soh050_divergence_gate(tmp)
        finally:
            shutil.rmtree(tmp)
        out['G28_G30_evaluators_on_the_committed_unit'] = {
            'ok': bool(g28_same and not g30_same and not g28_planted and d28['first_difference']['cycle'] == 151
                       and g30_div and d30['prediction_diverges_at_cycle_1'] and not g30_late),
            'g28_unit_vs_itself': g28_same, 'g30_unit_vs_itself_STOP': not g30_same, 'g28_planted_ulp_at_151': d28[
                'first_difference'], 'g30_diverge_at_1': g30_div, 'g30_first_difference_at_7_is_STOP': not g30_late}
    except Exception as error:  # noqa: BLE001
        out['G28_G30_evaluators_on_the_committed_unit'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                                           'traceback': traceback.format_exc()}
    try:
        refs, _inp = reference_views()
        a_ = {'status': 'certified', 'Q': 100.0, 'Q_cc': 100.0, 'band': 4000.0}
        b_ = {'status': 'certified', 'Q': 100.0 - 20000.0, 'Q_cc': 100.0 - 20000.0, 'band': 3000.0}
        views = {'h_x0_m175': a_, 'h_unit_m175': dict(b_, Q=100.0 - (M.UNIT_I + 15000.0),
                                                       Q_cc=100.0 - (M.UNIT_I + 15000.0)),
                 'e_soh050': dict(b_, Q=refs[UNIT_REF]['Q'] - 12000.0, Q_cc=refs[UNIT_REF]['Q_cc'] - 12000.0),
                 'g070_neutrality': dict(refs[UNIT_REF])}
        sc = score_all(views, {}, refs)
        neg = score_all(dict(views, h_unit_m175=dict(b_, Q=100.0 - (M.UNIT_I - 5000.0), Q_cc=100.0 - (M.UNIT_I - 5000.0)),
                             e_soh050=dict(b_, Q=refs[UNIT_REF]['Q'] + 9000.0, Q_cc=refs[UNIT_REF]['Q_cc'] + 9000.0)),
                        {}, refs)
        out['scorer_synthetic'] = {
            'ok': bool(abs(sc['A']['value_minus_I'] - 15000.0) < 1e-6 and sc['A']['verdict'] == 'determinate'
                       and abs(sc['A']['threshold'] - 2 * TAU * max(1.0, 3 * 4000.0 / (2 * TAU))) < 1e-6
                       and abs(sc['B']['delta_value'] - 12000.0) < 1e-6
                       and sc['B']['verdict'] == 'within resolution' and not sc['STOP']
                       and neg['STOP'] == ['A: value - I negative at m = 1.75', 'B: Delta value negative']),
            'A': sc['A'], 'B_delta': sc['B'].get('delta_value'), 'B_threshold': sc['B'].get('threshold'),
            'negative_STOP': neg['STOP']}
    except Exception as error:  # noqa: BLE001
        out['scorer_synthetic'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                   'traceback': traceback.format_exc()}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  the checks state; common preconditions
# ======================================================================================================================
def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST),
            'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'), 'all_hold_including_typing_test': doc.get('all_hold_including_typing_test'),
            'section_M_holds': ((doc.get('sections') or {}).get('M') or {}).get('holds'),
            'pickle_guard': doc.get('pickle_guard'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256')}


INLINE_SECTIONS = tuple((sid, fn) for sid, fn in K.SECTIONS if sid != 'M')   # M builds models: committed output only


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks(INLINE_SECTIONS)


def _common_checks():
    failures = []
    hr = harness_routed()
    if not hr['ok']:
        failures.append(f'harness_routed: the harness is not the pre-W155 harness + the W155 branch, or does not route '
                        f'here: {hr}')
    if not (os.path.isfile(_abs(PREDECESSOR['path'])) and _sha(PREDECESSOR['path']) == PREDECESSOR['sha256']
            and _committed_clean(PREDECESSOR['path'])):
        failures.append(f'predecessor: the extension-v6 stage spec is not as pinned / committed: {PREDECESSOR["path"]}')
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'clean_files: {rel} not committed / clean')
    for rel in (os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl'), K.W117['path'], L132.W118_SUMMARY,
                os.path.join(K.X0_REF_EVAL_DIR, 'per_cycle_record.jsonl'),
                os.path.join(E_C2_CALFADE_EVAL_DIR, 'child_stdout.log')):
        if not _committed_clean(rel):
            failures.append(f'references: {rel} not committed / clean')
    if _sha(os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl')) != M.UNIT_REF['per_cycle_record_sha256']:
        failures.append('references: the 3f084f2f per_cycle_record does not hash to its pin')
    for c, pin in SUPERSEDED_R1['campaign_specs'].items():
        if not (os.path.isfile(_abs(pin['path'])) and _sha(pin['path']) == pin['sha256'] and _committed_clean(pin['path'])
                and sorted(os.listdir(os.path.dirname(_abs(pin['path'])))) == [os.path.basename(pin['path'])]):
            failures.append(f'superseded_r1: {c} r1 campaign spec not as committed, or its root holds more: {pin}')
    others = _own_process_alive()
    if others:
        failures.append(f'concurrency: another copy of a W155 / W142 / W139 / W137 / W135 / W132 / W118 launcher is '
                        f'alive: {others}')
    return failures


def _checks_output_ok(failures):
    cf = _checks_file_state()
    if not (cf['all_hold_including_typing_test'] is True and cf['committed_clean'] and cf['section_M_holds'] is True
            and cf['pickle_guard'] == {'load': 0, 'loads': 0}
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'checks_output: the committed zero-solve checks output must exist, be clean and all hold: '
                        f'{ {k: v for k, v in cf.items() if k != "code_sha256_at_check"} }')
    for rel in K.CODE_PINNED_BY_CHECKS:
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'checks_output: {rel} differs from the version the committed checks ran on')
    return cf


# ======================================================================================================================
#  --freeze-cells
# ======================================================================================================================
def freeze_cells(started):
    tag = 'W155-FREEZE-CELLS'
    failures = _common_checks()
    cf = _checks_output_ok(failures)
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline_checks: {K.failing_items(checks_inline)}')
    post_tests, post_ok = evaluator_self_tests()
    if not post_ok:
        failures.append(f'evaluator_self_tests: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    wall = wall_time_estimate()
    if wall['single_run_over_4h']:
        failures.append(f'walls: a cell\'s worst case exceeds {WALL_LIMIT_H} h: {wall["single_run_over_4h"]}')
    for cell in M.CELL_ORDER:
        failures += [f'campaign_preconditions[{cell}]: {f}'
                     for f in H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)]
        pre = pre_launch_assertion(cell)
        if not pre['holds']:
            failures.append(f'pre_launch_assertion[{cell}]: {[k for k, v in pre["parts"].items() if not v]}')
    if os.path.isdir(_abs(ROOT_REL)) and any(f.startswith(SPEC_PREFIX) for f in os.listdir(_abs(ROOT_REL))):
        failures.append('order: a stage spec already exists (the cell specs are frozen BEFORE the stage spec)')
    if failures:
        _refuse(tag, failures)
    all_ok = True
    checks_pin = {'path': cf['path'], 'sha256': cf['sha256']}
    for cell in M.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        c = M.CELLS[cell]
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': (f'{SPEC_SERIES} v{SPEC_VERSION} (frozen AFTER this campaign spec) pins this spec by its '
                                'sha256 and holds the exact launch command'),
                 'label': M.LABEL, 'cell': cell, 'item': c['item'], 'kind': c['kind'], 'arm': c['arm'],
                 'model_variant': M.arm_variant(cell), 'minimum_soh': c['minimum_soh'],
                 'flex_price_multiplier': c['flex_price_multiplier'], 'role': c['role'],
                 'candidate_source_provenance_only': c['source'],
                 'expected_eval_key': pre['resettle_key'], 'objective_convention': DEFINITIONS['objective_convention'],
                 'solve_claim': {'parent': 'never solves (every launcher guard permitted=(), verify(0))',
                                 'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
                 'zero_solve_checks_output': checks_pin, 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(M.CELL_ORDER), 'code_sha256': {rel: _sha(rel) for rel in CODE_PINNED}}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=M.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 64', 'Planner task W155 (the Advisor\'s design check H1-H3)'],
            required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec)
        pre_frozen = pre_launch_assertion(cell, spec)
        cap_ok, cap = parent_capture_checklist(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds'] and cap_ok
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha} eval_key={e["eval_key"]} '
             f'cap={spec["cap"]} checks={all(checks.values())} failing={[k for k, v in checks.items() if not v]} '
             f'pre-launch={pre_frozen["holds"]} capture-checklist={cap_ok}'
             + ('' if cap_ok else f' {cap}'))
    _finish(0 if all_ok else 1, f'freeze-cells {"OK" if all_ok else "NOT OK"} -- next: commit, then --freeze-spec')


def cell_spec_state():
    out = {}
    for cell in M.CELL_ORDER:
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
                     'pre_launch_failing': [k for k, v in pre['parts'].items() if not v],
                     'eval_key': spec['candidates'][0]['eval_key'], 'eval_dir': spec['candidates'][0]['eval_dir'],
                     'working_dir_ids': spec['candidates'][0]['working_dir_ids'],
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
        raise RuntimeError('frozen W155 stage spec not found')
    return rel, sha, _load(rel)


def stage_spec_content(checks_inline, cf, post_tests, solver, prov, mem, wall, specs, o_section, refs, ref_inputs):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    cells = {}
    for i, cell in enumerate(M.CELL_ORDER):
        c = M.CELLS[cell]
        s = specs[cell]
        cells[cell] = {
            'launch_index_1_based': i + 1, 'item': c['item'], 'kind': c['kind'], 'arm': c['arm'],
            'model_variant': M.arm_variant(cell), 'minimum_soh': c['minimum_soh'],
            'flex_price_multiplier': c['flex_price_multiplier'], 'role': c['role'],
            'candidate_key': c['candidate_key'], 'nodes': {str(k): list(v) for k, v in c['nodes'].items()},
            'investment_year': c['investment_year'], 'candidate_source_provenance_only': c['source'],
            'claim_group': M.GROUP_OF_ITEM[c['item']], 'gated': False, 'cap_rule': M.cap_rule(cell),
            'spec_cap': M.spec_cap(cell), 'floor_row_lower_expected': M.floor_expected(cell),
            'declaration': M.declaration_for(cell), 'campaign_id': CAMPAIGN_IDS[cell],
            'campaign_root': campaign_root_rel(cell), 'configuration': configuration(cell), 'keys': expected_keys(cell),
            'campaign_spec': {k: s[k] for k in ('path', 'sha256', 'eval_key', 'eval_dir', 'harness_sha256', 'git_head')},
            'launch_command': launch_command(cell, s['sha256']),
            'preconditions_only_command': launch_command(cell, s['sha256'], preconditions_only=True),
            'expected_wall_time': wall['per_cell'][cell]}
    return {
        'schema': 'p515_s53_a64_cells_spec_v1', 'series': SPEC_SERIES, 'version': SPEC_VERSION,
        'stage_text': STAGE_TEXT, 'predecessor': PREDECESSOR, 'superseded_r1_before_this_freeze': SUPERSEDED_R1,
        'authority': [f'{BRIEF} Addendum 64 (run the m = 1.75 cell; the soh_min 0.50 row: variant key in new files, '
                      'Advisor design check, bitwise gate against the settled C2_calfade unit through 172, frozen spec)',
                      'Planner task W155 (the Advisor\'s design check H1-H3; section M; the predictions)',
                      f'{BRIEF} Addendum 61 (criterion v6; the determinacy floor)', f'{BRIEF} Addendum 62'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'pins': {'code_sha256': code, 'production_sha256': production, 'solver': solver,
                 'harness': {'pre_w155': K.PRE_W155_HARNESS, 'post_w155_branch_sha256': K.HARNESS_POST_W155_SHA256,
                             'w155_branch_lines': list(K.W155_BRANCH_LINES)},
                 'zero_solve_checks_output': {'path': cf['path'], 'sha256': cf['sha256'], 'manifest': cf['manifest']},
                 'references_inputs_sha256': ref_inputs,
                 'unit_reference_3f084f2f': {'eval_dir': UNIT_REF_EVAL_DIR,
                                             'per_cycle_record_sha256': M.UNIT_REF['per_cycle_record_sha256']},
                 'ess_params_file': {'path': M.ESS_PARAMS_REL, 'sha256': M.ESS_PARAMS_SHA256},
                 'campaign_specs': {c: {'path': specs[c]['path'], 'sha256': specs[c]['sha256']} for c in M.CELL_ORDER}},
        'production_since': prov,
        'cells': cells, 'cell_order': list(M.CELL_ORDER),
        'launch_order': {'order': list(M.CELL_ORDER),
                         'enforced': ('--run refuses a cell until every earlier cell has committed results, and every '
                                      'cell after g070_neutrality until its G28 holds (the neutrality gate)'),
                         'one_cell_per_call': True},
        'launch_commands': {c: cells[c]['launch_command'] for c in M.CELL_ORDER},
        'summarize_commands': {c: summarize_command(c) for c in ('h_unit_m175', 'e_soh050')},
        'inputs_in_force_now': o_section['inputs_now'], 'references': refs,
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '(minimum SoH 0.70 = the file, NOT overridden in the spec) + tight tail '
                                                 '{enabled True, compl_inf_tol 1e-6} declared'),
                          'soh_cells': ('+ model_variant C2_calfade (the standard 4-key variant) + minimum_soh in the '
                                        'keyed settling_resettle declaration'),
                          'h_cells': '+ flex_price_multiplier 1.75 (keyed through the base key; label at spec and entry)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference',
                          'concurrency': CONCURRENCY,
                          'checked_field_by_field': 'validate_campaign_spec at --freeze-cells, --freeze-spec and --run'},
        'soh_floor_plumbing': {
            'design': 'Advisor H1-H3 (Planner task W155)',
            'H1_apply': M.apply_minimum_soh.__doc__, 'H2_floor_dict': M.update_floor_rows.__doc__,
            'install': M.soh_floor_plumbing.__doc__, 'fail_fast': M.SohFloorPlumbing.assert_in_force.__doc__,
            'source_facts': M.soh_plumbing_capture_checklist.__doc__,
            'neutrality_scope': PREDICTIONS['gate_g070']['scope_note']},
        'stop_rule': {'module': 'settling_criterion_v6', 'class': 'settling_criterion_v6.SettlingRuleV6',
                      'version': SC6.VERSION, 'constants': SC6.constants(M.P_MAX), 'readings': SC6.READINGS,
                      'as_v6': 'the v6 declaration settling_rule (V6.settling_rule_declaration()), unchanged',
                      'clean_rule': L142.DEFINITIONS['clean_rule'], 'swing_floor': L142.DEFINITIONS['swing_floor'],
                      'gap_clause': 'tau / 2 (the v6 rule)', 'caps': 'min(k0_run + 109, 300) (dynamic), every cell'},
        'replay_gate': {'applies_to': [], 'skipped_for': list(M.CELL_ORDER)},
        'holds_after_first_residual_pass': {'AA': 'off', 'tail': 'on', 'rho': 'frozen',
                                            'same_as': 'W101 / W118 / W132 / W139 / W142'},
        'definitions': DEFINITIONS,
        'predictions_recorded_before_any_run': PREDICTIONS,
        'gates': {'scope': GATE_SCOPE, 'non_stopping': list(NON_STOPPING_GATES),
                  'prediction_gates_exit_3': list(PREDICTION_GATES),
                  'G9s': M.soh_readback_gate.__doc__, 'G29s': M.soh_sidecar_gate.__doc__,
                  'G28': g070_neutrality_gate.__doc__, 'G30': soh050_divergence_gate.__doc__,
                  'G10': flex_readback_check.__doc__, 'evaluator_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w155_a64_checks.py', 'committed_output': cf,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()},
                                  'not_rerun_inline': ['M (builds models; in the committed output)', 'W (typing)']}},
        'labelling_and_identity': {'label': M.LABEL, 'model_variant_label': H.MODEL_VARIANT_LABEL,
                                   'flex_price_label': H.FLEX_PRICE_LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key (carries model_variant for the SoH '
                                                      'cells, flex_price_multiplier 1.75 for the H cells), '
                                                      'settling_resettle (the W155 declaration; minimum_soh inside it '
                                                      'for the SoH cells)}); no key formula change; every other key '
                                                      'byte-identical to the pre-W155 harness (checks K)')},
        'harness_change': ('p515_s44_campaign_harness.py: the W155 branch (three lines after the v6 branch, before the '
                           'W118 fallback): a W155 declaration dispatches to p515_s53_w155_a64_hooks; every existing '
                           'branch identical (checks D)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'walls': {'limit_h_per_run': WALL_LIMIT_H, 'single_run_over_4h': wall['single_run_over_4h'],
                  'total_expected_h': wall['total_expected_h'], 'total_worst_case_h': wall['total_worst_case_h']},
        'dry_run_command': launch_command('e_soh050', specs['e_soh050']['sha256'], preconditions_only=True),
        'dry_run_scope': ('--preconditions-only: every --run precondition (the launch order REPORTED, not enforced), then '
                          'the ZERO-SOLVE child-path probe: the harness-routed hooks installed (the nine wrappers and the '
                          'SoH plumbing), the harness\'s floor-row precheck dict, G.s38_pf_capture_hooks entered with it, '
                          'the planning object as run_admm_arm builds it, the harness\'s OWN configuration hook with the '
                          'FROZEN campaign spec (past the floor-row identity check ~3003), the fail-fast; guard at 0, '
                          'pickle blocked; no lock, no child, no campaign-root write'),
    }


def freeze_spec(started):
    tag = 'W155-SPEC'
    failures = _common_checks()
    os.makedirs(_abs(ROOT_REL), exist_ok=True)
    existing = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'write_once: the stage spec already exists: {existing}')
    cf = _checks_output_ok(failures)
    specs = cell_spec_state()
    for cell, s in specs.items():
        if 'error' in s or not (s['name_carries_sha'] and s['committed_clean'] and s['checks_all']
                                and s['pre_launch_holds'] and s['root_holds_only_the_spec']):
            failures.append(f'campaign_spec[{cell}]: not frozen / committed / valid: {s}')
        elif s['harness_sha256'] != H.sha256_file(H.HARNESS_PATH):
            failures.append(f'campaign_spec[{cell}]: harness changed since its campaign spec froze')
        elif _load(s['path'])['extra'].get('code_sha256') != {rel: _sha(rel) for rel in CODE_PINNED}:
            failures.append(f'campaign_spec[{cell}]: code changed since its campaign spec froze')
    prov = K.uncommitted_used(CODE_PINNED)
    if not prov['ok']:
        failures.append(f'uncommitted: files this run uses: {prov["uncommitted_files_this_run_uses"]}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline_checks: {K.failing_items(checks_inline)}')
    post_tests, post_ok = evaluator_self_tests()
    if not post_ok:
        failures.append(f'evaluator_self_tests: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    solver = W101L.solver_check()
    if not solver['ok']:
        failures.append(f'solver: {solver}')
    wall = wall_time_estimate()
    if wall['single_run_over_4h']:
        failures.append(f'walls: a cell\'s worst case exceeds {WALL_LIMIT_H} h: {wall["single_run_over_4h"]}')
    if failures:
        _refuse(tag, failures)
    o_section = checks_inline['sections']['O']['result']
    refs, ref_inputs = reference_views()
    mem = L.memory_preflight(1)
    content = stage_spec_content(checks_inline, cf, post_tests, solver, prov, mem, wall, specs, o_section, refs,
                                 ref_inputs)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor {PREDECESSOR["path"]} {PREDECESSOR["sha256"][:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] evaluator self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    for cell in M.CELL_ORDER:
        w = wall['per_cell'][cell]
        _log(f"[{tag}] {cell}: cap {M.spec_cap(cell)} expected {w['expected_h']:.2f} h worst {w['worst_case_h']:.2f} h; "
             f"LAUNCH: {content['cells'][cell]['launch_command']}")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h; "
         f"DRY RUN: {content['dry_run_command']}")
    _finish(0, '-- next: commit, then the dry run on e_soh050')


# ======================================================================================================================
#  the dry run's zero-solve child-path probe
# ======================================================================================================================
@contextlib.contextmanager
def _pickle_blocked():
    counts = {'load': 0, 'loads': 0}
    orig = (pickle.load, pickle.loads)

    def bl(*_a, **_k):
        counts['load'] += 1
        raise RuntimeError('W155 dry run: pickle.load called -- no model loads are permitted')

    def bls(*_a, **_k):
        counts['loads'] += 1
        raise RuntimeError('W155 dry run: pickle.loads called -- no model loads are permitted')
    pickle.load, pickle.loads = bl, bls
    try:
        yield counts
    finally:
        pickle.load, pickle.loads = orig


def child_path_probe(cell, spec):
    """ZERO SOLVES. The child's path up to (NOT including) `planning.run_operational_planning`, with the FROZEN campaign
    spec and its entry: the hooks module the HARNESS routes the declaration to, installed for the run
    (`settling_resettle_hooks`: the nine wrappers + the SoH plumbing); the floor-row precheck dict
    (`p515_s40_polish_gap._build_floor_rows`, as the child ~5378); `G.s38_pf_capture_hooks` entered with it (looked up on
    G after the install, as the child's `with`); the planning object as `run_admm_arm` builds it; the harness's OWN
    configuration hook (`_config_hook_factory`, the frozen spec, the entry's variant / multiplier, expected_floor_rows =
    the dict) -- the floor-row identity check ~3003 included; the plumbing's fail-fast (as run_admm_arm's
    esso_capture_hooks entry would call it). Working-dir ids are the probe's own (never the entry's). A separate
    SolveProfileGuard(permitted=()) and pickle blocking for the probe, both verified at 0."""
    import p515_g_g1_g4_admm_gates as G
    import p56a_oracle as O
    from p515_s40_polish_gap import _build_floor_rows
    entry = spec['candidates'][0]
    decl = entry['settling_resettle']
    mod = H.resettle_hooks_module(decl)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    ids = {'precheck': f'p515s53_w155_dryrun_{tag}_precheck', 'run': f'p515s53_w155_dryrun_{tag}_run'}
    scratch = tempfile.mkdtemp(prefix='p515s53_w155_dryrun_')
    holder, report, out = {}, {}, {'ids': ids, 'routed_module': mod.__name__}
    guard = SolveProfileGuard(permitted=(), label='P5.15 W155 dry-run child-path probe (never solves)').install()
    try:
        with _pickle_blocked() as pk:
            _cc, floor_rows_by_node, _fc = _build_floor_rows(ids['precheck'])
            out['floor_rows_precheck'] = M.floor_rows_summary(floor_rows_by_node)
            eff_overrides = H.validate_overrides(entry.get('overrides') or {})
            mv = H.validate_model_variant(entry.get('model_variant'))
            flex_m = H.validate_flex_price_multiplier(entry.get('flex_price_multiplier'))
            investment_year = H.investment_year_from_canonical(entry['canonical'])
            investment_map = H.investment_map_from_canonical(entry['canonical'])
            eval_scratch = os.path.join(scratch, 'eval')
            os.makedirs(eval_scratch)
            with mod.settling_resettle_hooks(eval_scratch, decl, holder, int(spec['cap'])) as st:
                with G.s38_pf_capture_hooks(*[os.path.join(scratch, f'{k}.jsonl') for k in
                                              ('recourse_jump', 'ess_stride', 'floor', 'pf_stride')],
                                            floor_rows_by_node, stride=1):
                    planning, sed, candidate = G._construct_arm_planning(
                        spec['configuration']['arm_label'], eval_scratch, report, k_override=None,
                        investment_map=investment_map, eval_id=ids['run'], num_max_iters_override=int(spec['cap']),
                        apply_rho=False, investment_year=investment_year)
                    hook = H._config_hook_factory(spec, holder, overrides=eff_overrides, model_variant=mv,
                                                  investment_year=investment_year,
                                                  expected_floor_rows=floor_rows_by_node, flex_price_multiplier=flex_m)
                    hook(planning=planning, sed=sed, candidate=candidate, report=report)
                    out['past_the_configuration_hook_floor_identity_check'] = True
                    pl = getattr(st, 'plumbing', None)
                    out['fail_fast'] = pl.assert_in_force(planning) if pl is not None else 'no plumbing (H cell)'
                    out['plumbing'] = pl.summary() if pl is not None else None
                    out['floor_rows_shared_after'] = M.floor_rows_summary(floor_rows_by_node)
                    out['ess_soh_min_in_force'] = sorted({e.soh_min for y in sed.years
                                                          for e in sed.shared_energy_storages[y]})
                    out['loaded_ageing_soh_min'] = sed.params.ageing.soh_min
            out['pickle_counts'] = dict(pk)
        rule11 = report.get('rule_eleven_checklist') or {}
        out['w20_model_variant'] = rule11.get('w20_model_variant')
        out['w21_ess_ageing_baseline'] = rule11.get('w21_ess_ageing_baseline')
        out['w33_flex_price_multiplier'] = rule11.get('w33_flex_price_multiplier')
        out['readback_pre_run_floor_row_lower'] = {
            n: v['readback']['floor_row_lower'] for n, v in
            ((holder.get('model_variant_readback_pre_run') or {}).get('per_node') or {}).items()}
        out['minimum_soh_applied_checks'] = ((holder.get('model_variant_applied') or {}).get('minimum_soh_applied')
                                             or {}).get('checks')
        out['summary'] = {k: (holder.get(M.SUMMARY_KEY) or {}).get(k) for k in (
            'schema', 'kind', 'minimum_soh_declared', 'flex_price_multiplier_expected', 'soh_floor_plumbing_ok')}
    finally:
        guard.uninstall()
        out['probe_guard'] = {'counts': dict(guard.counts), 'verify_0_failures': guard.verify(0)}
        shutil.rmtree(scratch, ignore_errors=True)
        left = {}
        for i in ids.values():
            work = os.path.join(O.WORK_DIR, i)
            if os.path.isdir(work):
                if not any(files for _r, _d, files in os.walk(work)):
                    shutil.rmtree(work)
                else:
                    left[i] = work
        out['working_dirs_with_files_left'] = left
    c = M.CELLS[cell]
    floor = M.floor_expected(cell)
    parts = {'routed_to_w155': mod is M,
             'past_the_configuration_hook_floor_identity_check': out.get('past_the_configuration_hook_floor_identity_check')
             is True,
             'zero_solves_probe_guard': not out['probe_guard']['verify_0_failures'],
             'zero_pickle_loads': out.get('pickle_counts') == {'load': 0, 'loads': 0},
             'no_working_dir_left': not out['working_dirs_with_files_left']}
    if c['kind'] == 'soh':
        parts.update({
            'identity_check_passed_by_equality_flag_true': (out.get('w20_model_variant') or {}).get(
                'floor_rows_identical_to_baseline_probe') is True,
            'shared_dict_rows_all_at_the_declared_floor_both_sides': all(
                v['soh_min_values'] == [floor] and v['n_rows'] == M.FLOOR_ROWS_PER_NODE
                for v in (out.get('floor_rows_shared_after') or {}).values()) and bool(out.get('floor_rows_shared_after')),
            'probe_readback_floor_row_lower_is_the_declared_floor': bool(out['readback_pre_run_floor_row_lower']) and all(
                v == floor for v in out['readback_pre_run_floor_row_lower'].values()),
            'every_ess_and_the_loaded_parameters_at_the_declared_floor': (out.get('ess_soh_min_in_force') == [floor]
                                                                          and out.get('loaded_ageing_soh_min') == floor),
            'fail_fast_held': isinstance(out.get('fail_fast'), dict) and all(out['fail_fast'].values()),
            'plumbing_one_apply_one_dict': ((out.get('plumbing') or {}).get('n_apply_calls') == 1
                                            and (out.get('plumbing') or {}).get('n_floor_dicts_captured') == 1),
            'summary_plumbing_ok': (out.get('summary') or {}).get('soh_floor_plumbing_ok') is True,
        })
    else:
        parts['flex_multiplier_applied_and_read_back'] = ((out.get('w33_flex_price_multiplier') or {}).get(
            'readback_all_match') is True and (out.get('w33_flex_price_multiplier') or {}).get(
            'flex_price_multiplier') == M.FLEX_M)
    return all(v is True for v in parts.values()), {'parts': parts, **out}


# ======================================================================================================================
#  --run
# ======================================================================================================================
def launch_order_state(cell):
    idx = M.CELL_ORDER.index(cell)
    missing = [prev for prev in M.CELL_ORDER[:idx] if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE))]
    g070 = None
    if idx > 0 and M.NEUTRALITY_CELL not in missing:
        res = json.load(open(os.path.join(campaign_root(M.NEUTRALITY_CELL), RESULTS_FILE)))
        g070 = (res.get('gates') or {}).get('G28_g070_bitwise_vs_3f084f2f_through_172')
    failures = [f'launch_order: {p} (#{M.CELL_ORDER.index(p) + 1}) has no results yet' for p in missing]
    if idx > 0 and M.NEUTRALITY_CELL not in missing and g070 is not True:
        failures.append(f'launch_order: g070_neutrality G28 (bitwise vs 3f084f2f through 172) is {g070!r}: STOP')
    return failures


def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W155-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('stage_spec: not committed / clean')
    if ss['pins']['code_sha256'] != {rel: _sha(rel) for rel in ss['pins']['code_sha256']}:
        failures.append('stage_spec: code changed since the stage spec froze')
    pin = ss['pins']['campaign_specs'].get(cell) or {}
    if pin.get('sha256') != spec_sha256:
        failures.append(f'stage_spec: it pins {pin.get("sha256")} for {cell}, not {spec_sha256}')
    order = launch_order_state(cell)
    if order and not preconditions_only:
        failures += order
    root = campaign_root(cell)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f'campaign_preconditions: {f}' for f in H.check_campaign_preconditions(
        root, extra_clean_files=EXTRA_CLEAN_FILES) if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign_root: must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign_spec: not committed / clean')
    checks = validate_campaign_spec(cell, spec)
    failures += [f'campaign_spec_check: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('ESS parameters file', (spec['configuration'].get('ess_params_file') or {}).get('sha256'),
                               _sha(M.ESS_PARAMS_REL))):
        if pinned != now:
            failures.append(f'pins: {what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline_checks: {K.failing_items(checks_inline)}')
    pre = pre_launch_assertion(cell, spec)
    if not pre['holds']:
        failures.append(f'pre_launch_assertion: {[k for k, v in pre["parts"].items() if not v]}')
    cap_ok, cap_checks = parent_capture_checklist(cell, spec)
    if not cap_ok:
        failures.append(f'parent_capture_checklist: {cap_checks}')
    solver = W101L.solver_check()
    if not solver['ok'] or solver['sha256'] != ss['pins']['solver']['sha256']:
        failures.append(f'solver: path / binary differs: {solver}')
    mem = L.memory_preflight(1)
    _log(f"[{tag}] memory preflight: available {mem.get('available_gib')} GiB required {mem['required_gib']:.2f} GiB -> "
         f"{'PASS' if mem['pass'] else 'REFUSE'}")
    if not mem['pass']:
        failures.append(f"memory_preflight: REFUSED {mem.get('available_gib')} < {mem['required_gib']}")
    if failures:
        _refuse(tag, failures)
    entry = spec['candidates'][0]
    idx = M.CELL_ORDER.index(cell)
    if preconditions_only:
        _log(f"[{tag}] every --run precondition holds: campaign spec {os.path.relpath(spec_path, REPO)} "
             f"sha256={spec_sha256} (pinned by the stage spec {ss_rel} sha256={ss_sha}); launch index {idx + 1} of "
             f"{len(M.CELL_ORDER)}; harness routed ({harness_routed()['harness_sha256'][:8]}); checks per section "
             f"{({k: v['holds'] for k, v in checks_inline['sections'].items()})}; pre-launch parts {pre['parts']}; parent "
             f"capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; cap {spec['cap']}; "
             f"model_variant {entry.get('model_variant')}; minimum_soh {entry['settling_resettle'].get('minimum_soh')}; "
             f"flex {entry.get('flex_price_multiplier')}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} "
             f"sha256={solver['sha256']}")
        _log(f"[{tag}] LAUNCH ORDER (reported; a real launch {'WOULD REFUSE' if order else 'would proceed'}): "
             f"{order or 'every earlier cell has results and G28 holds'}")
        _log(f'[{tag}] child-path probe (ZERO SOLVES): the harness-routed hooks, the floor precheck dict, '
             f'G.s38_pf_capture_hooks, the planning object, the harness configuration hook with the frozen spec ...')
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                ok, probe = child_path_probe(cell, spec)
            quiet = buf.getvalue()
        except Exception as error:  # noqa: BLE001
            _log(f'[{tag} PRECONDITION FAILED] child_path_probe: {type(error).__name__}: {error}')
            traceback.print_exc()
            _finish(1, 'dry run: child-path probe FAILED')
        _log(f"[{tag}] child-path probe parts: {probe['parts']}")
        _log(f"[{tag}] routed module {probe.get('routed_module')}; floor rows precheck {probe.get('floor_rows_precheck')}")
        _log(f"[{tag}] floor rows shared dict AFTER the apply (both sides of ~3003): "
             f"{probe.get('floor_rows_shared_after')}")
        _log(f"[{tag}] configuration hook w20_model_variant: {probe.get('w20_model_variant')}")
        _log(f"[{tag}] pre-run readback floor_row_lower by node: {probe.get('readback_pre_run_floor_row_lower')}; "
             f"ESS soh_min in force {probe.get('ess_soh_min_in_force')}; loaded ageing soh_min "
             f"{probe.get('loaded_ageing_soh_min')}; minimum_soh_applied checks {probe.get('minimum_soh_applied_checks')}")
        _log(f"[{tag}] plumbing {probe.get('plumbing')}; fail-fast {probe.get('fail_fast')}")
        _log(f"[{tag}] probe guard {probe.get('probe_guard')}; pickle {probe.get('pickle_counts')}; working dirs left "
             f"{probe.get('working_dirs_with_files_left')}; production stdout during the probe: {len(quiet)} chars "
             f"(suppressed)")
        if not ok:
            _log(f"[{tag} PRECONDITION FAILED] child_path_probe: {[k for k, v in probe['parts'].items() if v is not True]}")
            _finish(1, 'dry run: child-path probe NOT OK')
        _log(f'[{tag}] STOPPED before the campaign lock and the child (no lock taken, no evaluation, no campaign-root '
             f'write, zero solves)')
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
    stopping = [k for k, v in gates.items() if not v and k not in NON_STOPPING_GATES]
    prediction_failed = [k for k in PREDICTION_GATES if k in gates and not gates[k]]
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'launch_index_1_based': idx + 1, 'utc': _utc(),
               'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'], 'kind': M.CELLS[cell]['kind'],
               'model_variant': entry.get('model_variant'),
               'minimum_soh': entry['settling_resettle'].get('minimum_soh'),
               'flex_price_multiplier': entry.get('flex_price_multiplier'),
               'gates': gates, 'gates_pass': all(gates.values()), 'stopping_gate_failures': stopping,
               'prediction_gate_failures': prediction_failed,
               'gate_detail': detail, 'cell_report': report,
               'pre_launch_assertion': pre, 'parent_capture_checklist': cap_checks, 'memory_preflight_at_run': mem,
               'solver': solver, 'batch_info': batch, 'parent_view': (records[0] or {}).get('parent_view'),
               'settling_resettle_summary_soh_plumbing': summ.get('soh_floor_plumbing'),
               'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] settling v6: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"band_width {report.get('band_width')} Q_end {report.get('Q_end')} t_sum(end) {report.get('t_sum_end')} "
             f"k0_run {report.get('k0_run')} cycles {report.get('cycles_run')} floor_year {report.get('floor_year')} "
             f"terminal step / threshold {report.get('terminal_step_to_threshold_ratio_end_cycle')} record status "
             f"{report.get('record_status')}")
    if prediction_failed:
        _log(f'[{tag}] RECORDED STOP CONDITION: {prediction_failed} -- {detail.get("G28") or detail.get("G30")} -- STOP '
             f'FOR THE PLANNER before the next cell (exit 3)')
    if cell in ('h_unit_m175', 'e_soh050'):
        _log(f'[{tag}] scorer: {summarize_command(cell)}')
    ok_rest = not stopping and _guards_ok(g) and isinstance(report, dict) and 'error' not in report
    code = (3 if prediction_failed else 0) if ok_rest else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  --summarize
# ======================================================================================================================
def summarize(started, after_cell):
    tag = f'W155-SUMMARY-after-{after_cell}'
    ss_rel, ss_sha, ss = load_stage_spec()
    idx = M.CELL_ORDER.index(after_cell)
    reports, missing, inputs = {}, [], {}
    for cell in M.CELL_ORDER[:idx + 1]:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        rel = os.path.relpath(path, REPO)
        if not os.path.isfile(path) or not _committed_clean(rel):
            missing.append(cell)
            continue
        inputs[rel] = _sha(rel)
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, f'w155_summary_after_{idx + 1:02d}_{after_cell}.json')
    if missing or os.path.exists(_abs(out_rel)):
        _refuse(tag, [f'summary_inputs: cells without committed results {missing} or the summary exists ({out_rel})'])
    refs, ref_inputs = reference_views()
    if ref_inputs != ss['pins']['references_inputs_sha256']:
        _refuse(tag, ['references: changed since the stage spec froze'])
    views = {c: r.get('view') or L132.view_from_report(r) for c, r in reports.items()}
    scored = score_all(views, reports, refs)
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'after_cell': after_cell, 'after_cell_index_1_based': idx + 1,
           'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': DEFINITIONS['objective_convention'],
           'reports': reports, 'references': refs, 'scored': scored, 'definitions': DEFINITIONS,
           'predictions': PREDICTIONS, 'inputs_sha256': {**inputs, **ref_inputs}}
    H._write_once_json(_abs(out_rel), doc)
    man = os.path.join(ROOT_REL, f'w155_summary_after_{idx + 1:02d}_{after_cell}_manifest_sha256.json')
    H._write_once_json(_abs(man), {out_rel: _sha(out_rel), **doc['inputs_sha256']})
    _log(f'[{tag}] wrote {out_rel}; A {scored.get("A")}; B {scored.get("B")}')
    if scored['STOP']:
        _log(f'[{tag}] RECORDED STOP CONDITION: {scored["STOP"]} -- STOP FOR THE PLANNER (exit 3)')
        _finish(3)
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-cells', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=M.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--after-cell', choices=M.CELL_ORDER, default=None)
    parser.add_argument('--preconditions-only', action='store_true',
                        help='with --run: every --run precondition, the zero-solve child-path probe, then stop')
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
