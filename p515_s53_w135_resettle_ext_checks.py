"""
P5.15 Addendum 58 Supplement, Planner task W135 -- ZERO-SOLVE checks for the re-settling EXTENSION (ageing E, six
model-variant arms at minimum SoH 0.70, and pb_y2025_n5 under the settling rule v3; 7 cells).

WRITTEN IN W135, RUN LATER: only after W132's cell #38 has its results committed AND the prepared router edit
(`p515_s53_w135_harness_router.patch`) has been applied -- a separate task. Run before that, sections S (W132 #38) and
D / K / H (the dispatch) fail, which is the intended behaviour.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W132 checks
module, imported for its helpers, arms its own and the modules it imports arm theirs; every guard is verified at 0).
NOTHING IS SOLVED. Section M BUILDS ESSO models (production's `_build_subproblem`, through the campaign child's own
configuration hook) on fresh SRP1 planning objects -- builds, never solves; H6 / H8 read a fresh planning object.

WHAT IS CHECKED
  V   the stop rule v3 as W132 validated it (`p515_s53_w132_resettle_v3_checks.tests_V`, reused unchanged): W135 uses
      settling_criterion_v3 exactly as the 38 W132 cells do.
  O   the seven cells: the original record (PROVENANCE for the E arms, the replay reference for pb) committed, in its
      campaign manifest, sha as pinned; the original spec entry; the original configuration's base key reproduces the
      original eval key; every E arm on the unit candidate db77e154...; the six arm variants against W20's committed
      definitions (S46 spec entries; the C3 unit = W20's BASELINE_AS_VARIANT); pb: N_old 119, first residual pass 110,
      no lapse after it, configuration identity as W118 / W132 (the tail the one declared change); THE FILE IN FORCE:
      39106f93, minimum_soh 0.70, calibration (10000, 0.80, 0.80), phi_cal 0.985; the 0.50-era evidence
      (ageing_mechanism.json b5eca2a2) committed; the settled unit 3f084f2f: k* 172, first residual pass 103, holds from
      113 and -- the zero-solve evidence behind the C2_calfade consistency check -- its NATURAL regime on 104..112 is
      already the held regime (AA off by production, tail active passed, rho frozen, every cycle a residual pass).
  S   W132's cell #38 (l_195156fa) results and manifest committed and clean, no W132 / W135 launcher alive; no
      uncommitted change to a file this run uses; production changes since each original's git head (report).
  H   the hooks through W132's REAL nine wrappers (reused) with W135 declarations and stand-in production: H1 pb gated
      with its ORIGINAL values bitwise through k0 = 110, then a synthetic continuation that certifies; H2 a one-ulp
      divergence at cycle 40 aborts; H3 an E cell (ungated) certifies; H4 an E cell creeps to its dynamic rule cap
      k0_run + 109; H5 a non-Optimal ESSO exit is a lapse; H6 the REAL install with the PATCHED harness dispatch (W135
      -> this extension, W132 -> v3, W118 -> W118) and the child's order; H7 the summary carries this schema; H8-H11
      W132's real-production tests of the SAME wrappers (exit wrapper, AA / tail / rho holds), reused.
  M   per arm, ON FRESHLY BUILT MODELS: the campaign child's own configuration hook (`_config_hook_factory`, with the
      campaign configuration: case-file AA, the ESS ageing baseline declared and pinned, the tail) applies the arm and
      reads it back from probe models; ESSO models built as production builds them are read back on clones; THE G9
      REPLACEMENT (`variant_readback_gate`) holds on that record with the closed forms, floor_row_lower == 0.70 on every
      node and the file-vs-declaration checks; negative controls (floor 0.50, k off by 1e-9, label missing) fail it.
      C2_calfade = the baseline: its freshly built ESSO models are IDENTICAL (every constraint, variable, parameter,
      expression, objective and the cohort bookkeeping) to the baseline path's (no variant, ESS ageing readback).
  K   keys: every entry of every committed campaign spec keys identically under this harness and the pre-W135 harness
      (9fd91ded, sha256 pinned, from git), W118 and W132 settling_resettle entries included, except the campaign's OWN
      W135 entries, accepted only inside the W135 stage root and only if their frozen key follows the formula over the
      pre-W135 base key; the 7 keys follow the formula, are distinct, differ from every original and every W132 / W118
      key, and appear in no committed spec outside the own root (the rule: a pre-run scan excludes the run's own);
      planted controls (outside refused; sibling-prefix refused; own accepted); the pre-W135 harness refuses a W135
      declaration.
  P   `assert_resettle_preconditions` holds for every cell with its cap and spec-like entry; negative controls (cap,
      tail off, AA off, variant swapped, floor overridden, label missing) refused; the validator refuses malformed
      declarations.
  D   the harness is EXACTLY the pre-W135 harness plus the prepared patch (3 added lines in `resettle_hooks_module`,
      no line removed, every existing branch identical); its sha256 is the pinned post-patch sha256.
  X   the pure readers on committed records: the floor year of the settled unit 3f084f2f at k* 172 is 2035; an S46
      0.50-era record has no active floor row; the trajectory equality reproduces 3f084f2f against itself through 172
      and finds a planted one-ulp difference.
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured; LATER -- see above):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w135_resettle_ext_checks.py \\
      > data/SRP1/Results/P515S53/w135_resettle_ext/zero_solve_checks_launch.log 2>&1
"""
import contextlib
import copy
import hashlib
import inspect
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
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W135 re-settling extension zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w135_resettle_ext_hooks as W  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w132_resettle_v3_checks as K132  # noqa: E402 -- the v3 suite and helpers (arms its own guard)
import settling_criterion_v3 as SC3  # noqa: E402
import p515_s46_variant_checks as M46V  # noqa: E402 -- W20's variant definitions and model fingerprint (arms its own guard)

K118, K105, K98 = K132.K118, K132.K105, K132.K98


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w135_checks', GUARD),) + tuple(K132.GUARDS) + (('s46_variant_checks_imported', M46V.GUARD),))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W135_ROOT_REL = os.path.join(_P53, 'w135_resettle_ext')
OUT_DIR_REL = os.path.join(W135_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w135_zero_solve_checks.json'
OUT_MANIFEST = 'w135_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w135_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w135_resettle_ext_'
KEY_EXCLUDED_ROOTS = (W135_ROOT_REL,)
# The harness pinned by the W132 stage spec 139d1e62 (last changed by 9fd91ded) -- the harness BEFORE the router edit
PRE_W135_HARNESS = {'commit': '9fd91ded', 'sha256': 'bf0a757c0d36da406c5388f2af4f36aed84e6baa4f569f3658e3218b449a5c6a'}
PATCH_REL = 'p515_s53_w135_harness_router.patch'
HARNESS_POST_PATCH_SHA256 = '861d6070c107d9f4930ca3b6961945d99440d77a7d43b75631adee14ecc0442d'
UNIT_SETTLED_EVAL_DIR = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals',
                                     '3f084f2ffaeef2b7_n7_4h_e1')
UNIT_SETTLED = {'per_cycle_record_sha256': 'fd8aad17f8a8a40e1ca73a2af2d5edeeee83e2d3265f2bdad8c38c68fb41b324',
                'k_star': 172, 'first_residual_pass': 103, 'hold_after_cycle': 112}
AGEING_MECHANISM = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S46', 'ageing_mechanism',
                                         'ageing_mechanism.json'),
                    'sha256': 'b5eca2a238bb4383dfda026bdf675a3878feba9235c8eabef66d02c7bf6d1a1f', 'commit': '5e1e3562'}
S46_C2_EVAL_DIR = W.original_eval_dir('e_c2')
FILE_IN_FORCE = {'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                 'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.8}}
IDENTITY_KEYS = K132.IDENTITY_KEYS
GATED_IDENTITY_KEYS = K132.GATED_IDENTITY_KEYS
PRODUCTION_FILES = K132.PRODUCTION_FILES


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _clean(rel):
    return K118._git_tracked_clean(rel)


def _jt(x):
    return K132._jt(x)


def spec_entry(cell):
    """(original spec, its entry holding the original eval key)."""
    c = W.CELLS[cell]
    spec = json.load(open(_abs(os.path.join(c['orig_root'], c['orig_spec']))))
    hits = [e for e in spec['candidates'] if e.get('eval_key') == c['orig_eval_key']]
    return spec, (hits[0] if len(hits) == 1 else None)


def configuration_now():
    """The current production configuration (W132's: case-file AA, the C2 ESS ageing baseline declared, the tail on)
    plus the ESS parameters pin every W135 spec carries."""
    cfg = K132.configuration_now()
    cfg['ess_params_file'] = {'path': W.ESS_PARAMS_REL, 'sha256': W.ESS_PARAMS_SHA256}
    return cfg


def key_kwargs(cell):
    """The evaluation-key arguments of a W135 cell under the current configuration (no settling_resettle)."""
    cfg = configuration_now()
    return dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
                model_variant=W.arm_variant(cell), convergence_depth_tail=cfg['convergence_depth_tail'])


def candidate_of(cell):
    """(candidate key, overrides) of the cell's evaluation: the original entry's (E: the unit; pb: its own)."""
    _spec, e = spec_entry(cell)
    return e['key'], e['overrides']


def spec_like(cell, decl=None, entry_overrides=None, config_overrides=None, cap=None):
    """A campaign-spec-shaped dict for the preconditions (what the frozen spec will hold for the cell)."""
    decl = decl if decl is not None else W.declaration_for(cell)
    key, overrides = candidate_of(cell)
    cfg = configuration_now()
    cfg.update(config_overrides or {})
    entry = {'label': cell, 'key': key, 'overrides': overrides, 'settling_resettle': decl}
    mv = W.arm_variant(cell)
    out = {'cap': W.spec_cap(cell) if cap is None else cap, 'configuration': cfg, 'candidates': [entry]}
    if mv is not None:
        entry['model_variant'] = mv
        entry['model_variant_label'] = H.MODEL_VARIANT_LABEL
        out['model_variant_label'] = H.MODEL_VARIANT_LABEL
    entry.update(entry_overrides or {})
    return out


# ======================================================================================================================
#  V -- the stop rule v3 (W132's suite, reused)
# ======================================================================================================================
def tests_V():
    r = K132.tests_V()
    return {'holds': r['holds'] is True, 'source': 'p515_s53_w132_resettle_v3_checks.tests_V (reused unchanged)',
            'tests': {k: v.get('ok') for k, v in r['tests'].items()}}


# ======================================================================================================================
#  O -- the seven cells, the file in force, the references
# ======================================================================================================================
def _unit_natural_regime():
    """The settled unit 3f084f2f (W101): first residual pass 103, holds engaged after 112 (W101's N). On 104..112 its
    natural regime -- what production did with no hold -- is already the held regime: AA returned production's off
    action, the tail was passed active, rho was frozen (action 'held (frozen ...)', no value change), every cycle a
    residual pass. A W135-style run holds from k0_run + 1 = 104: on those cycles the holds change no value."""
    d = UNIT_SETTLED_EVAL_DIR
    rows = {r['cycle']: r for r in _read_jsonl(_abs(os.path.join(d, 'per_cycle_record.jsonl')))}
    lines = {x['cycle']: x for x in _read_jsonl(_abs(os.path.join(d, 'settling_continuation_cycle_record.jsonl')))}
    dec = json.load(open(_abs(os.path.join(d, 'settling_decision.json'))))
    fp = next(c for c in sorted(rows) if rows[c]['boyd_all_pass'] and rows[c]['local_solves_ok'])
    rng = range(UNIT_SETTLED['first_residual_pass'] + 1, UNIT_SETTLED['hold_after_cycle'] + 1)
    per = {}
    for c in rng:
        x, r = lines.get(c) or {}, rows.get(c) or {}
        per[c] = {
            'residual_pass': bool(r.get('boyd_all_pass') and r.get('local_solves_ok')),
            'aa_action_is_production_off': (x.get('aa') or {}).get('action') == R.AA_OFF_ACTION,
            'tail_active_passed': (x.get('tail_apply') or {}).get('active_passed') is True,
            'rho_frozen_unchanged': (r.get('rho_freeze_active') is True
                                     and all(str(r.get(f'rho_{g}_action', '')).startswith('held (frozen')
                                             for g in ('v', 'pf', 'ess'))
                                     and all(r.get(f'rho_{g}_after') == (rows.get(c - 1) or {}).get(f'rho_{g}_after')
                                             for g in ('v', 'pf', 'ess'))),
            'no_hold_in_w101_yet': (x.get('holds') or {}) == {'aa': False, 'tail_apply': False, 'tail_next': False,
                                                              'rho': False}}
    held_from = min((c for c, x in lines.items() if (x.get('holds') or {}).get('aa') is True), default=None)
    return {'first_residual_pass': fp, 'k_star': dec.get('k_star'), 'status': dec.get('status'),
            'w101_holds_from_cycle': held_from, 'cycles_checked': [rng.start, rng.stop - 1], 'per_cycle': per,
            'natural_regime_equals_held_regime_104_112': all(all(v.values()) for v in per.values()),
            'per_cycle_record_sha256': _sha(os.path.join(d, 'per_cycle_record.jsonl'))}


def tests_O():
    res, cells = {}, {}
    now = configuration_now()
    case_now = H.sha256_file(H.CASE_FILE)
    ess_now = _sha(W.ESS_PARAMS_REL)
    ess = json.load(open(_abs(W.ESS_PARAMS_REL)))
    for cell in W.CELL_ORDER:
        c = W.CELLS[cell]
        spec, entry = spec_entry(cell)
        man = json.load(open(_abs(os.path.join(c['orig_root'], 'campaign_manifest_sha256.json'))))
        rel = W.reference_path(cell)
        sha = _sha(rel)
        rows = _read_jsonl(_abs(rel))
        passes = [r['cycle'] for r in rows if r['boyd_all_pass'] and r['local_solves_ok']]
        n_rows = rows[-1]['cycle']
        cfg = spec['configuration']
        base_orig = H.evaluation_key(entry['key'], entry['overrides'],
                                     case_file_aa=cfg.get('case_file_anderson_acceleration'),
                                     model_variant=entry.get('model_variant'),
                                     ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
                                     flex_price_multiplier=entry.get('flex_price_multiplier'),
                                     convergence_depth_tail=cfg.get('convergence_depth_tail')) if entry else None
        rec = json.load(open(_abs(os.path.join(W.original_eval_dir(cell), 'evaluation_record.json'))))
        parts = {
            'original_spec_entry_found': entry is not None,
            'entry_eval_dir_and_label_as_pinned': bool(entry) and entry['eval_dir'] == c['orig_eval_dir']
            and entry['label'] == c['orig_label'],
            'per_cycle_record_sha256_as_pinned': sha == c['per_cycle_record_sha256'],
            'per_cycle_record_in_campaign_manifest': man.get(rel) == sha,
            'per_cycle_record_committed_clean': _clean(rel),
            'original_cycles_as_pinned': n_rows == c['orig_cycles'],
            'original_record_certified_at_its_last_cycle': rec.get('status') == 'certified'
            and rec.get('cycles_run') == n_rows,
            'base_key_original_configuration_equals_original_eval_key': base_orig == c['orig_eval_key'],
            'cap_within_ceiling': W.spec_cap(cell) <= c['cap_ceiling'],
        }
        if c['item'] == 'E':
            s46 = (M46V.VARIANTS.get(c['orig_model_variant']) if c['orig_model_variant'] else M46V.BASELINE_AS_VARIANT)
            parts.update({
                'e_candidate_is_the_unit': (entry or {}).get('key') == W.UNIT_CANDIDATE_KEY,
                'e_unit_nodes_and_year': (entry or {}).get('canonical', {}).get('investment_year') == W.UNIT_YEAR
                and {int(k): tuple(map(float, v)) for k, v in (entry or {}).get('canonical', {}).get('nodes', {}).items()}
                == W.UNIT_NODES,
                'e_arm_variant_equals_w20_definition': _jt(H.validate_model_variant(s46)) == _jt(W.arm_variant(cell)),
                'e_original_entry_variant_as_pinned': _jt((entry or {}).get('model_variant'))
                == _jt(H.validate_model_variant(M46V.VARIANTS[c['orig_model_variant']])
                       if c['orig_model_variant'] else None),
                'e_arm_variant_valid_in_the_harness': _jt(H.validate_model_variant(W.arm_variant(cell)))
                == _jt(W.arm_variant(cell)),
                'e_variant_has_no_key_that_could_change_the_floor': 'minimum_soh' not in H.MODEL_VARIANT_KEYS,
            })
        else:
            lapses = [k for k in range(passes[0], n_rows + 1) if k not in passes] if passes else None
            ident = {k: cfg.get(k) == (now[k] if k in now else K132._w101_ref_cfg().get(k)) for k in IDENTITY_KEYS}
            ident['case_file_sha256'] = cfg.get('case_file_sha256') == case_now == K132.CASE_FILE_SHA256
            ident.update({k: cfg.get(k) == now[k] for k in GATED_IDENTITY_KEYS})
            ident['ess_params_file_at_run_is_the_file_now'] = ((cfg.get('ess_params_file') or {}).get('sha256')
                                                               == W.ESS_PARAMS_SHA256 == ess_now)
            ident['tail_absent_at_run_the_one_declared_change'] = cfg.get('convergence_depth_tail') is None
            ident['entry_overrides_empty'] = (entry or {}).get('overrides') == {}
            ident['entry_no_model_variant_flex_premium'] = not any(
                k in (entry or {}) for k in ('model_variant', 'flex_price_multiplier', 'interface_deviation_premium'))
            parts.update({
                'pb_N_old_as_pinned': n_rows == c['N_old'],
                'pb_first_residual_pass_as_pinned': bool(passes) and passes[0] == c['k0'],
                'pb_lapses_after_k0_as_pinned': lapses == c['original_lapses_after_k0'],
                'pb_N_old_passes': bool(passes) and passes[-1] == n_rows,
                'pb_same_original_as_w118': (R.CELLS['pb_y2025_n5']['orig_eval_key'] == c['orig_eval_key']
                                             and R.CELLS['pb_y2025_n5']['per_cycle_record_sha256']
                                             == c['per_cycle_record_sha256']),
                **{f'identity:{k}': bool(v) for k, v in ident.items()}})
        cells[cell] = {'ok': all(parts.values()), 'parts': parts, 'item': c['item'], 'arm': c['arm'],
                       'model_variant': W.arm_variant(cell), 'orig_eval_key': c['orig_eval_key'],
                       'orig_cycles': n_rows, 'original_spec': os.path.join(c['orig_root'], c['orig_spec']),
                       'original_git_head': spec.get('git_head'), 'per_cycle_record': rel,
                       'per_cycle_record_sha256': sha, 'cap_rule': W.cap_rule(cell), 'spec_cap': W.spec_cap(cell)}
    res['cells'] = cells
    cal = ess['ageing']['calibration']
    k_by_arm = {a: W.closed_form_expected(mv, cal['cycles_n'], cal['reference_dod_d']) for a, mv in W.ARMS.items()}
    res['closed_forms_by_arm_report'] = k_by_arm
    unit = _unit_natural_regime()
    res['settled_unit_3f084f2f'] = unit
    am_sha = _sha(AGEING_MECHANISM['path'])
    res['inputs_now'] = {'case_file': {'path': H.CASE_FILE_REL, 'sha256': case_now},
                         'ess_params_file': {'path': W.ESS_PARAMS_REL, 'sha256': ess_now,
                                             'minimum_soh': ess['ageing']['minimum_soh'],
                                             'calendar_retention_per_year': ess['ageing']['calendar_retention_per_year'],
                                             'calibration': {k: cal[k] for k in ('status', 'cycles_n', 'reference_dod_d',
                                                                                 'eol_retention_r')}},
                         'configuration_now': now,
                         'ageing_mechanism_json': {**AGEING_MECHANISM, 'sha256_now': am_sha}}
    res['parts'] = {
        'every_cell_ok': all(v['ok'] for v in cells.values()), 'n_cells_7': len(cells) == 7,
        'case_file_now_as_pinned': case_now == K132.CASE_FILE_SHA256,
        'ess_params_now_is_the_pinned_file': ess_now == W.ESS_PARAMS_SHA256 == K132.ESS_PARAMS_SHA256_C2
        and _clean(W.ESS_PARAMS_REL),
        'file_in_force_minimum_soh_070_phi_0985_calibration_c2': (
            ess['ageing']['minimum_soh'] == FILE_IN_FORCE['minimum_soh'] == W.FLOOR_ROW_LOWER_EXPECTED
            and ess['ageing']['calendar_retention_per_year'] == FILE_IN_FORCE['calendar_retention_per_year']
            and {k: cal[k] for k in FILE_IN_FORCE['calibration']} == FILE_IN_FORCE['calibration']),
        'declared_ess_baseline_minimum_soh_070': now['ess_ageing_baseline']['minimum_soh'] == W.FLOOR_ROW_LOWER_EXPECTED,
        'c2_calfade_arm_equals_the_file_ageing_law': (
            W.ARMS[W.BASELINE_EQUIVALENT_ARM]['eol_retention_r'] == cal['eol_retention_r']
            and W.ARMS[W.BASELINE_EQUIVALENT_ARM]['calendar_retention_per_year']
            == ess['ageing']['calendar_retention_per_year']
            and W.ARMS[W.BASELINE_EQUIVALENT_ARM]['available_energy_soh_point'] == 'end'
            and W.ARMS[W.BASELINE_EQUIVALENT_ARM]['ageing_enabled'] is True),
        'ageing_mechanism_json_as_pinned_and_committed': am_sha == AGEING_MECHANISM['sha256']
        and _clean(AGEING_MECHANISM['path']),
        'settled_unit_record_as_pinned': unit['per_cycle_record_sha256'] == UNIT_SETTLED['per_cycle_record_sha256']
        and _clean(os.path.join(UNIT_SETTLED_EVAL_DIR, 'per_cycle_record.jsonl')),
        'settled_unit_k_star_172_first_pass_103_holds_from_113': (
            unit['k_star'] == UNIT_SETTLED['k_star'] and unit['first_residual_pass'] == UNIT_SETTLED['first_residual_pass']
            and unit['w101_holds_from_cycle'] == UNIT_SETTLED['hold_after_cycle'] + 1),
        'settled_unit_natural_regime_104_112_is_the_held_regime': unit['natural_regime_equals_held_regime_104_112'],
        'model_variant_label_and_rtol_as_the_harness': (W.MODEL_VARIANT_LABEL == H.MODEL_VARIANT_LABEL
                                                        and W.READBACK_RTOL == H.MODEL_VARIANT_READBACK_RTOL),
    }
    res['holds'] = all(res['parts'].values())
    return res


# ======================================================================================================================
#  S -- W132 #38 committed; production since the originals
# ======================================================================================================================
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w132_resettle_v3_campaign', 'p515_s53_w135_resettle_ext_campaign',
                          'p515_s53_w118_resettle_campaign')


def _launchers_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def w132_done():
    """W132's last cell (#38, l_195156fa) has its results and manifest committed and clean; the W132 stage spec is the
    pinned 139d1e62 and names that cell last."""
    res_rel, man_rel = W.w132_last_cell_files_rel()
    ss_ok = os.path.isfile(_abs(W.W132_STAGE_SPEC_REL)) and _sha(W.W132_STAGE_SPEC_REL) == W.W132_STAGE_SPEC_SHA256
    last = json.load(open(_abs(W.W132_STAGE_SPEC_REL)))['cell_order'][-1] if ss_ok else None
    parts = {'w132_stage_spec_as_pinned': ss_ok, 'w132_last_cell_is_l_195156fa': last == W.W132_LAST_CELL,
             'w132_38_results_committed_clean': os.path.isfile(_abs(res_rel)) and _clean(res_rel),
             'w132_38_manifest_committed_clean': os.path.isfile(_abs(man_rel)) and _clean(man_rel)}
    return all(parts.values()), {'parts': parts, 'results': res_rel, 'manifest': man_rel}


def production_since_originals(used_files=()):
    heads = {}
    for cell in W.CELL_ORDER:
        heads.setdefault(spec_entry(cell)[0].get('git_head'), []).append(cell)
    per = {}
    for head, cells in heads.items():
        diff = H._git(['diff', '--name-status', head, 'HEAD', '--', *PRODUCTION_FILES]).splitlines() if head else []
        per[head] = {'cells': cells, 'production_changed_since': diff}
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    used = set(used_files) | set(PRODUCTION_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN)
    dirty_used = [d for d in dirty if d.split()[-1] in used]
    return {'per_original_git_head': per, 'uncommitted_tracked_py': dirty, 'uncommitted_files_this_run_uses': dirty_used,
            'head': H._git(['rev-parse', 'HEAD']), 'ok': not dirty_used,
            'note': ('the E arms are first evaluations (nothing is replayed against their 0.50-era originals); pb is '
                     'gated in-cycle bitwise through k0 = 110 against its original, as W118 held it')}


def tests_S():
    done, d = w132_done()
    alive = _launchers_alive()
    prov = production_since_originals(CODE_PINNED_BY_CHECKS)
    parts = {'w132_cell_38_committed': done, 'no_w132_w135_w118_launcher_alive': not alive,
             'no_uncommitted_change_to_a_file_this_run_uses': prov['ok']}
    return {'holds': all(parts.values()), 'parts': parts, 'w132_38': d, 'launchers_alive': alive, 'production': prov}


# ======================================================================================================================
#  H -- the hooks through W132's real wrappers with W135 declarations
# ======================================================================================================================
def drive(cell, variant='certify', first_pass_at=50, nonopt=None):
    """W132's `drive` (p515_s53_w132_resettle_v3_checks.drive) restated for a W135 declaration: the REAL nine wrappers
    (`V.make_wrappers`) on the W135 state, driven in production's call order with stand-in production. pb: its ORIGINAL
    values through N_old, then a synthetic continuation; an E cell: a synthetic run (Boyd from `first_pass_at`).
    Variants: certify, ulp_at_40, gap, creep. `nonopt` = {cycle: [block keys]} injects Acceptable exits."""
    decl = W.declaration_for(cell)
    ref = W.load_replay_reference(decl)
    cap = W.spec_cap(cell)
    sink = []
    st = W.ResettleStateW135(decl, None, cap, reference=ref, sink=sink)
    scripts = {'boyd': [], 'aa': [], 'next': [], 'pen': [], 'rc': [], 'efc': []}
    orig, calls = K105._standins(None, scripts)
    local_script = []
    orig['_admm_local_solves_succeeded'] = lambda pp_, results: local_script.pop(0)
    blocks_by_cycle = {}
    gated = decl['replay_reference'] is not None
    c_info = W.CELLS[cell]
    g_rows = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(W.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if gated else {})
    n_old = c_info['N_old'] if gated else None
    q_anchor = ref[n_old]['gross_operational_cost'] if gated else 6.5e8
    nonopt = nonopt or {}

    def t_target(c):
        if gated and c <= n_old:
            return 12345.0
        j = c - (n_old if gated else first_pass_at)
        if variant == 'gap' and j <= 80:
            return 5000.0
        return 150.0
    pp, tso, dso, esso, cv, set_cycle = K118._fake_world(t_target)
    w = V.make_wrappers(st, orig, H.ipopt_exit_class, srp_module=K118._fake_srp(blocks_by_cycle, st),
                        classifier_label='p515_s44_campaign_harness.ipopt_exit_class (W135 checks)')
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    models = {'tso': tso, 'dso': dso, 'esso': esso}
    esso_terms_orig = K105.E.esso_side_terms
    K105.E.esso_side_terms = lambda pp_, em: {str(n): {'salvage_value': 0.0, 'feasibility_penalty': 0.0}
                                              for n in K118.NODES}
    raised = None
    template = g_rows.get(n_old) if gated else json.load(open(_abs(os.path.join(
        W.original_eval_dir('pb_y2025_n5_v3'), 'g_s39_D.json'))))['cycle_trajectory'][-1]
    pen_hold = K118._frozen_pen(g_rows[c_info['k0']] if gated else template)
    try:
        local_script.append(True)
        w['_admm_local_solves_succeeded'](pp, K132.fake_results(pp))       # the initialisation check (round 0)
        w['_capture_convergence_depth_tail_baseline'](pp, admm)
        for c in range(1, cap + 1):
            if gated and c <= n_old:
                row, g = ref[c], g_rows[c]
                bm = K105.boyd_from_g(g)
                rc = {'gross_operational_cost': row['gross_operational_cost'], 'net_operational_recourse': row['recourse'],
                      'terminal_salvage_value': row['terminal_salvage_value']}
                if variant == 'ulp_at_40' and c == 40:
                    rc['gross_operational_cost'] = math.nextafter(rc['gross_operational_cost'], math.inf)
                pen = K118._pen_row(g) if c <= c_info['k0'] else pen_hold
                efc = row['efc_per_day_max']
                local_ok = row['local_solves_ok']
                active = c > c_info['k0']
            else:
                base = n_old if gated else first_pass_at
                j = c - base
                if variant in ('certify', 'gap'):
                    qs = 6000.0 * (0.6 ** (j / 15.0)) * math.cos(2 * math.pi * j / 30.0)
                else:
                    qs = -300.0 * j
                bm = copy.deepcopy(K105.boyd_from_g(template))
                passing = gated or c >= first_pass_at
                if gated and variant in ('certify', 'gap') and c == n_old + 1:
                    passing = False
                bm['all_boyd_pass'] = bool(passing)
                q = q_anchor + qs
                rc = {'gross_operational_cost': q, 'net_operational_recourse': q, 'terminal_salvage_value': 0.0}
                pen = pen_hold
                efc = 1.0
                local_ok = True
                active = st.held(c)
            blocks_by_cycle[c] = K118._blocks_for(rc['gross_operational_cost'], rc['terminal_salvage_value'])
            set_cycle(c)
            scripts['boyd'].append(bm)
            scripts['aa'].append(None)
            scripts['next'].append(None)
            scripts['pen'].append(pen)
            if local_ok:
                scripts['rc'].append(rc)
            scripts['efc'].append(efc)
            local_script.append(bool(local_ok))
            with contextlib.redirect_stdout(io.StringIO()):
                w['_apply_convergence_depth_tail'](pp, admm, active, object(), c)
                w['get_admm_boyd_residual_metrics'](pp, tso, dso, esso, cv, {}, admm)
                w['_admm_local_solves_succeeded'](pp, K132.fake_results(pp, nonopt.get(c, ())))
                if local_ok:
                    aa_rec = w['_anderson_acceleration_cycle_step'](object(), None, cv, {}, None, {}, bm, c)
                    w['_get_operational_recourse_components'](pp, models)
                else:
                    aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
                w['_convergence_depth_tail_next_state'](bool(bm['all_boyd_pass'] and local_ok), True, aa_rec)
                w['_update_admm_penalties']({}, {}, {}, {}, bm, object(), iter=c, allow_update=local_ok,
                                            freeze_state={})
                w['_get_admm_efc_per_day_max'](esso)
            if admm.minimum_consecutive_converged_cycles == W.SETTLING_END_THRESHOLD:
                break
        w['_apply_convergence_depth_tail'](pp, admm, False, object(), None)
    except RuntimeError as error:
        raised = str(error)
    finally:
        K105.E.esso_side_terms = esso_terms_orig
    files = {}
    for fname, obj in sink:
        files.setdefault(fname, []).append(obj)
    return {'state': st, 'files': files, 'raised': raised, 'calls': calls, 'admm': admm}


def _h_layering(srp):
    """W132's real-install test (`K132._h_layering`) restated for a W135 cell and the PATCHED harness dispatch."""
    before = {name: getattr(srp, name) for name in W.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w135_layering_')
    cell = 'pb_y2025_n5_v3'
    n = W.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with W.settling_resettle_hooks(scratch, W.declaration_for(cell), holder, cap=W.spec_cap(cell)) as st:
            installed = {name: getattr(srp, name) for name in W.WRAPPED}
            all_installed = all(installed[k] is not before[k] for k in W.WRAPPED)
            st.first_pass = n
            installed_inner = srp.get_admm_boyd_residual_metrics
            with IDC.interface_dual_capture_hooks(scratch, holder):
                idc_outer = srp.get_admm_boyd_residual_metrics is not installed_inner
                with H.convergence_depth_append_hooks(stub):
                    base = srp._capture_convergence_depth_tail_baseline(pp, admm)
                    active = False
                    for c in range(1, n + 3):
                        if st.cur is not None:
                            st.cur['finalized'] = True
                        srp._apply_convergence_depth_tail(pp, admm, active, base, c)
                        conv = c <= n
                        active = srp._convergence_depth_tail_next_state(
                            conv if c <= n else False, True,
                            {'cycle': c, 'action': R.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
                    st.cur['finalized'] = True
                    srp._apply_convergence_depth_tail(pp, admm, active, base, None)
        files_written = sorted(os.listdir(scratch))
    finally:
        shutil.rmtree(scratch)
    after = {name: getattr(srp, name) for name in before}
    restored = all(after[k] is before[k] for k in before)
    next_events = [e[1]['value'] for e in stub.events if e[0] == 'next_state']
    apply_events = [e[1] for e in stub.events if e[0] == 'apply']
    appender_saw_held = (next_events[n] is True and apply_events[n + 1]['record']['active'] is True)
    child_src = inspect.getsource(H._child_real)
    i_disp = child_src.find('W118C = resettle_hooks_module(resettle)')
    i_pre = child_src.find('W118C.assert_resettle_preconditions(resettle, spec, tail_checklist, aa_on)')
    i_cm = child_src.find('W118C.settling_resettle_hooks(eval_dir, resettle, holder, int(spec[\'cap\']))')
    pre_ok = 0 < i_disp < i_pre < child_src.find('G.run_admm_arm(') and i_cm > i_pre
    w132_cell, w118_cell = V.CELL_ORDER[0], 'f2_challenger'
    dispatch = {
        'w135_to_w135': H.resettle_hooks_module(W.declaration_for(cell)) is W,
        'w135_e_cell_to_w135': H.resettle_hooks_module(W.declaration_for('e_c2')) is W,
        'w132_to_w132': H.resettle_hooks_module(V.declaration_for(w132_cell)) is V,
        'w118_to_w118': H.resettle_hooks_module(R.declaration_for(w118_cell)) is R,
        'harness_validates_w135': H.validate_settling_resettle(W.declaration_for(cell)) == W.declaration_for(cell),
        'harness_validates_w132_unchanged': H.validate_settling_resettle(V.declaration_for(w132_cell))
        == V.declaration_for(w132_cell),
        'harness_validates_w118_unchanged': H.validate_settling_resettle(R.declaration_for(w118_cell))
        == R.declaration_for(w118_cell)}
    summ = holder.get(W.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and pre_ok and idc_outer and all_installed
                          and all(dispatch.values()) and summ.get('phase') == 'ended' and summ.get('schema') == W.SCHEMA
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_dispatch_then_checklist_before_run_admm_arm': pre_ok, 'dispatch': dispatch,
            'files_written_by_install_without_cycles': files_written, 'summary_phase': summ.get('phase'),
            'summary_schema': summ.get('schema'), 'summary_errors': summ.get('errors')}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    cell = 'pb_y2025_n5_v3'
    c = W.CELLS[cell]
    s = K132._drive_summary(drive(cell, 'certify'))
    s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                   and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                   and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                   and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                   and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                   and s['exits_51_every_line'] and s['all_optimal_every_line_is_bool']
                   and s['non_optimal_cycles'] == [] and s['certificate_length_after'] == 10
                   and s['lines'] == s['k_star'] == s['creep_lines']
                   and s['holds_through_first_pass'] == [K132.HOLDS_OFF] and s['holds_after_first_pass'] == [K132.HOLDS_ON]
                   and s['t_sum_every_line'] and not s['capture_errors'])
    s.pop('summary')
    res['H1_pb_original_values_bitwise_through_k0_then_certify'] = {'holds': s['ok'], 'detail': s}
    s = K132._drive_summary(drive(cell, 'ulp_at_40'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] and 'REPLAY DIVERGED at cycle 40' in s['raised'] and s['lines'] == 40
                   and (s['first_divergence'] or {}).get('fields_differing') == ['gross_operational_cost']
                   and s['replay_bitwise_through'] == 39 and not s['summary_ok']
                   and s['stopped_by'] == 'replay_divergence_abort')
    res['H2_pb_one_ulp_at_40_aborts'] = {'holds': s['ok'], 'detail': s}
    s = K132._drive_summary(drive('e_c2_calfade', 'certify', first_pass_at=103))
    summ3 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 103 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule'
                   and s['n_overlap'] == 0 and s['exit_capture_complete'] and s['exits_51_every_line']
                   and s['holds_through_first_pass'] == [K132.HOLDS_OFF] and s['holds_after_first_pass'] == [K132.HOLDS_ON]
                   and summ3.get('schema') == W.SCHEMA and summ3.get('arm') == 'C2_calfade')
    res['H3_e_cell_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    s = K132._drive_summary(drive('e_c3_unit', 'creep', first_pass_at=60))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 60 + W.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 60
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0)
    res['H4_e_cell_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    base = K132._drive_summary(drive('e_c4', 'certify', first_pass_at=100))
    # k0 = 100 (ungated: the first residual pass); the non-Optimal cycle 18 cycles later, inside the certifying span
    # (W_MIN = 20), as W132's H3 placed it (N_old + 20 = its k0 + 18)
    s = K132._drive_summary(drive('e_c4', 'certify', first_pass_at=100, nonopt={118: ('ESSO|7',)}))
    s5 = {k: v for k, v in s.items() if k != 'summary'}
    s5['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['n_lapses'] == 1
                    and s['lapse_causes'][-1] == ['non_optimal'] and s['non_optimal_cycles'] == [118]
                    and s['k_star'] > base['k_star'] and s['in_cycle_rule_equals_pure_replay'] and s['summary_ok'])
    s5['k_star_all_optimal'] = base['k_star']
    res['H5_e_cell_non_optimal_esso_exit_is_a_lapse'] = {'holds': s5['ok'], 'detail': s5}
    res['H6_layering_real_install_patched_dispatch'] = _h_layering(srp)
    st = W.ResettleStateW135(W.declaration_for('e_no_ageing'), None, W.spec_cap('e_no_ageing'), reference={}, sink=[])
    summ = st.summary()
    res['H7_summary_schema_is_w135'] = {'holds': summ.get('schema') == W.SCHEMA and summ.get('arm') == 'no_ageing'
                                        and isinstance(st, V.ResettleStateV3), 'schema': summ.get('schema')}
    for name, fn in (('H8_w132_exit_wrapper_on_real_production_reused', K132._h_exit_real),
                     ('H9_aa_hold_real_production_reused', lambda: K132._h_real_v3(srp, 'aa')),
                     ('H10_tail_hold_real_production_reused', lambda: K132._h_real_v3(srp, 'tail')),
                     ('H11_rho_hold_real_production_reused', lambda: K132._h_real_v3(srp, 'rho'))):
        try:
            res[name] = fn()
        except Exception as error:  # noqa: BLE001
            res[name] = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  M -- per-arm variant readback on freshly built models; the G9 replacement; C2_calfade == the baseline
# ======================================================================================================================
def _campaign_spec_like():
    cfg = configuration_now()
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
                              'ess_ageing_baseline': cfg['ess_ageing_baseline'],
                              'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
                              'ess_params_file': cfg['ess_params_file'],
                              'convergence_depth_tail': cfg['convergence_depth_tail']},
            'cap': W.CAP_CEILING, 'required_consecutive_cycles': 10}


def _built(G, SED, M46V, name, run_tag, scratch, mv, floor_rows):
    """One arm (or the baseline, mv None): the planning object as `run_admm_arm` builds it, the child's OWN
    configuration hook (applies the variant; reads it back from probe models; refuses on mismatch), then ESSO models
    built as production builds them (NOT solved)."""
    report, holder = {}, {}
    eval_id = f'p515s53_w135_checks_{run_tag}_{name}'
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, name), report, investment_map=dict(W.UNIT_NODES), eval_id=eval_id,
        num_max_iters_override=W.CAP_CEILING, apply_rho=False, investment_year=W.UNIT_YEAR)
    hook = H._config_hook_factory(_campaign_spec_like(), holder, overrides={}, model_variant=mv,
                                  investment_year=W.UNIT_YEAR, expected_floor_rows=floor_rows)
    hook(planning=planning, sed=sed, candidate=candidate, report=report)
    models = M46V._build_esso(SED, sed, candidate)
    return planning, sed, candidate, holder, report, models, eval_id


def _synthetic_record(mv, holder, fresh_readback):
    cfg = configuration_now()
    return {'model_variant': H.validate_model_variant(mv), 'model_variant_label': H.MODEL_VARIANT_LABEL,
            'model_variant_applied_in_child': holder.get('model_variant_applied'),
            'model_variant_readback_pre_run': holder.get('model_variant_readback_pre_run'),
            'model_variant_readback_terminal': fresh_readback,
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_params_sha256_in_child': _sha(W.ESS_PARAMS_REL),
            'ess_ageing_verified_pre_run': holder.get('ess_ageing_verified_pre_run')}


def tests_M():
    import p515_g_g1_g4_admm_gates as G
    import shared_energy_storage_data as SED
    import p515_s40_polish_gap as PG
    import p56a_oracle as O
    scratch = tempfile.mkdtemp(prefix='p515s53_w135_checks_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    ids = []
    arms, fps = {}, {}
    try:
        _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s53_w135_checks_{run_tag}_precheck')
        ids.append(f'p515s53_w135_checks_{run_tag}_precheck')
        for cell in W.E_CELLS:
            decl = W.declaration_for(cell)
            mv = decl['model_variant']
            _p, sed, _cand, holder, report, models, eid = _built(G, SED, M46V, cell, run_tag, scratch, mv, floor_rows)
            ids.append(eid)
            fresh = H.model_variant_readback_models(models, sed, mv, W.UNIT_YEAR, clone=True)
            floor_fresh, _counts = G._identify_soh_floor_rows(models)
            rec = _synthetic_record(mv, holder, fresh)
            ok, det = W.variant_readback_gate(rec, decl)
            negatives = {}
            bad = copy.deepcopy(rec)
            bad['model_variant_readback_terminal']['per_node']['7']['readback']['floor_row_lower'] = 0.5
            negatives['floor_050_refused'] = not W.variant_readback_gate(bad, decl)[0]
            bad = copy.deepcopy(rec)
            k7 = bad['model_variant_readback_pre_run']['per_node']['7']['readback']
            if k7.get('k') is not None:
                k7['k'] = k7['k'] * (1.0 + 1e-9)
                negatives['k_off_by_1e-9_refused'] = not W.variant_readback_gate(bad, decl)[0]
            else:
                bad['model_variant_readback_pre_run']['per_node']['7']['readback']['d_row_form'] = 'D * 2kE == 365 n avg'
                negatives['d_row_form_changed_refused'] = not W.variant_readback_gate(bad, decl)[0]
            bad = copy.deepcopy(rec)
            bad['model_variant_label'] = None
            negatives['label_missing_refused'] = not W.variant_readback_gate(bad, decl)[0]
            bad = copy.deepcopy(rec)
            bad['ess_params_sha256_in_child'] = '0' * 64
            negatives['file_sha_mismatch_refused'] = not W.variant_readback_gate(bad, decl)[0]
            arms[cell] = {'ok': bool(ok and all(negatives.values()) and floor_fresh == floor_rows),
                          'g9_replacement_holds': ok, 'g9_parts_failing': sorted(k for k, v in det['parts'].items()
                                                                                  if v is not True),
                          'negative_controls': negatives,
                          'floor_rows_fresh_equal_the_baseline_precheck': floor_fresh == floor_rows,
                          'closed_form': det.get('closed_form'),
                          'readback_node7_pre_run': (rec['model_variant_readback_pre_run'] or {}).get(
                              'per_node', {}).get('7', {}).get('readback'),
                          'readback_node7_fresh': fresh['per_node']['7']['readback'],
                          'floor_row_lower_by_node': det['terminal']['floor_row_lower'],
                          'rule_eleven_w20': report.get('rule_eleven_checklist', {}).get('w20_model_variant'),
                          'rule_eleven_w21': report.get('rule_eleven_checklist', {}).get('w21_ess_ageing_baseline')}
            if cell == W.CELL_OF_ARM[W.BASELINE_EQUIVALENT_ARM]:
                fps['variant'] = {str(n): M46V._fingerprint(m) for n, m in models.items()}
                fps['variant_ess'] = [{k: repr(getattr(e, k)) for k in ('bus', 't_cal', 'cl_nom', 'dod_nom', 'soh_min',
                                                                        'cl_eff', 'phi_cal')}
                                      for y in sed.years for e in sed.shared_energy_storages[y]]
                fps['variant_settings'] = list(SED._esso_ageing_model_settings(sed))
            del models
        _p, sed, _cand, holder, _report, models, eid = _built(G, SED, M46V, 'baseline_no_variant', run_tag, scratch,
                                                              None, floor_rows)
        ids.append(eid)
        base_fp = {str(n): M46V._fingerprint(m) for n, m in models.items()}
        base_ess = [{k: repr(getattr(e, k)) for k in ('bus', 't_cal', 'cl_nom', 'dod_nom', 'soh_min', 'cl_eff', 'phi_cal')}
                    for y in sed.years for e in sed.shared_energy_storages[y]]
        base_readback_ok = ((holder.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run') or {}).get(
            'all_match') is True
        diffs = {n: sorted(k for k in set(base_fp[n]) | set(fps.get('variant', {}).get(n, {}))
                           if base_fp[n].get(k) != fps.get('variant', {}).get(n, {}).get(k))
                 for n in base_fp}
        identical = bool(fps.get('variant')) and all(not d for d in diffs.values())
        c2 = {'ok': bool(identical and base_ess == fps.get('variant_ess') and base_readback_ok
                         and fps.get('variant_settings') == list(SED._esso_ageing_model_settings(sed)) == ['end', True]),
              'esso_models_identical_every_node': identical,
              'n_items_by_node': {n: len(base_fp[n]) for n in base_fp},
              'digest_baseline': {n: M46V._fp_digest(base_fp[n]) for n in base_fp},
              'digest_c2_calfade_variant': {n: M46V._fp_digest(fps['variant'][n]) for n in fps.get('variant', {})},
              'first_differences_by_node': {n: d[:10] for n, d in diffs.items()},
              'ess_constants_identical_repr': base_ess == fps.get('variant_ess'),
              'baseline_path_ess_readback_all_match': base_readback_ok}
        del models
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        left = {}
        for i in ids:
            work = os.path.join(O.WORK_DIR, i)
            if os.path.isdir(work):
                if not any(files for _r, _d, files in os.walk(work)):
                    shutil.rmtree(work)
                else:
                    left[i] = work
    return {'holds': bool(arms and all(v['ok'] for v in arms.values()) and c2['ok']),
            'arms': arms, 'c2_calfade_equals_the_baseline_model': c2, 'working_dirs_with_files_left': left,
            'note': ('zero solves: models are BUILT (production _build_subproblem via the child hook and '
                     'update_model_with_candidate_solution), never solved; scratch outside the repository, removed')}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w135():
    import importlib.util
    src = subprocess.run(['git', 'show', f"{PRE_W135_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W135_HARNESS['sha256']:
        raise RuntimeError(f'pre-W135 harness sha256 {sha} != pinned {PRE_W135_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w135_pre_harness_')
    path = os.path.join(tmp, '_w135_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w135_pre_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod, sha, src


def resettle_keys(pre=None):
    out = {}
    for cell in W.CELL_ORDER:
        key, overrides = candidate_of(cell)
        kw = key_kwargs(cell)
        base = H.evaluation_key(key, overrides, **kw)
        decl = W.declaration_for(cell)
        rkey = H.evaluation_key(key, overrides, settling_resettle=decl, **kw)
        base_pre = pre.evaluation_key(key, overrides, **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        base_no_variant = H.evaluation_key(key, overrides, **dict(kw, model_variant=None))
        out[cell] = {'candidate_key': key, 'original_eval_key': W.CELLS[cell]['orig_eval_key'], 'base_key_now': base,
                     'resettle_key': rkey, 'formula_holds': rkey == formula, 'base_equals_pre_w135': base == base_pre,
                     'base_carries_the_variant': (W.arm_variant(cell) is None) == (base == base_no_variant),
                     'differs_from_original': rkey != W.CELLS[cell]['orig_eval_key']
                     and base != W.CELLS[cell]['orig_eval_key']}
    return out


def _key_holders(scanned, key):
    return sorted({rel for rel, spec in scanned for e in spec.get('candidates') or [] if H._entry_eval_key(e) == key})


def _outside_roots(rels, exclude_roots=KEY_EXCLUDED_ROOTS):
    return [r for r in rels if not any(r.startswith(root + os.sep) for root in exclude_roots)]


def _pre_refuses_w135(pre):
    try:
        pre.validate_settling_resettle(W.declaration_for(W.CELL_ORDER[0]))
        return False
    except ValueError:
        return True


def tests_K():
    pre, pre_sha, _src = _harness_pre_w135()
    n_specs = n_entries = n_equal = n_own = n_own_ok = n_w118 = n_w132 = n_frozen_equal = 0
    mismatch, errors, own_bad = [], [], []
    scanned = []
    all_keys = set()
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            all_keys.add(H._entry_eval_key(e))
            try:
                args, kw = K132._key_args(spec, e)
                rs = e.get('settling_resettle')
                if W.is_w135_declaration(rs):
                    n_own += 1
                    decl = W.validate_settling_resettle(rs)
                    formula = hashlib.sha256(json.dumps({'base_evaluation_key': pre.evaluation_key(*args, **kw),
                                                         'settling_resettle': decl}, sort_keys=True,
                                                        separators=(',', ':')).encode()).hexdigest()
                    now = H.evaluation_key(*args, settling_resettle=decl, **kw)
                    inside = rel.startswith(W135_ROOT_REL + os.sep)
                    if inside and now == formula == H._entry_eval_key(e):
                        n_own_ok += 1
                    else:
                        own_bad.append({'spec': rel, 'label': e.get('label'), 'inside_own_root': inside,
                                        'formula_equals_frozen': formula == H._entry_eval_key(e)})
                    continue
                if V.is_v3_declaration(rs):
                    n_w132 += 1
                elif rs is not None:
                    n_w118 += 1
                new = H.evaluation_key(*args, settling_resettle=rs, **kw)
                old = pre.evaluation_key(*args, settling_resettle=rs, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
                if H._entry_eval_key(e) == new:
                    n_frozen_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    holders = {cell: _key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: _outside_roots(h) for cell, h in holders.items()}
    probe = keys[W.CELL_ORDER[0]]['resettle_key']
    w132_keys = {v['resettle_key'] for v in K132.resettle_keys().values()}

    def planted(rel):
        return (rel, {'campaign_id': 'w135_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    planted_outside_rel = os.path.join(_P53, 'w135_planted_negative_control', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_sibling_rel = os.path.join(W135_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_own_rel = os.path.join(W135_ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{W.CELL_ORDER[0]}',
                                   f'campaign_spec_{CAMPAIGN_ID_PREFIX}{W.CELL_ORDER[0]}_00000000.json')
    ctl_outside = _outside_roots(_key_holders(scanned + [planted(planted_outside_rel)], probe))
    ctl_sibling = _outside_roots(_key_holders(scanned + [planted(planted_sibling_rel)], probe))
    own_holders = _key_holders(scanned + [planted(planted_own_rel)], probe)
    ctl_own = _outside_roots(own_holders)
    mine = {v['resettle_key'] for v in keys.values()}
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_entry_equal': n_equal == n_entries - n_own and n_entries > 0,
        'w118_resettle_entries_scanned_and_equal': n_w118 >= 10,
        'w132_resettle_entries_scanned_and_equal': n_w132 >= 38,
        'own_w135_entries_inside_root_follow_the_formula': n_own_ok == n_own and not own_bad,
        'resettle_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'resettle_base_equals_pre_w135_every_cell': all(v['base_equals_pre_w135'] for v in keys.values()),
        'base_key_carries_the_variant_for_e_cells_only': all(v['base_carries_the_variant'] for v in keys.values()),
        'resettle_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'resettle_keys_distinct': len(mine) == len(keys) == 7,
        'resettle_keys_differ_from_every_w132_key': not (mine & w132_keys) and len(w132_keys) == 38,
        'resettle_keys_absent_from_committed_specs_outside_own_root': not any(outside.values()),
        'control_planted_outside_root_refused': planted_outside_rel in ctl_outside,
        'control_planted_sibling_prefix_refused': planted_sibling_rel in ctl_sibling,
        'control_own_campaign_spec_accepted': planted_own_rel in own_holders and planted_own_rel not in ctl_own,
        'control_w135_declaration_refused_by_the_pre_w135_harness': _pre_refuses_w135(pre),
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w135_harness': {**PRE_W135_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'entries_whose_frozen_eval_key_equals_the_recomputed_REPORTED': n_frozen_equal,
            'w118_resettle_entries': n_w118, 'w132_resettle_entries': n_w132,
            'own_w135_entries': {'n': n_own, 'ok': n_own_ok, 'bad': own_bad[:20]},
            'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'resettle_keys_in_committed_specs_all_REPORTED': holders, 'key_excluded_roots': list(KEY_EXCLUDED_ROOTS),
            'controls': {'planted_outside': {'rel': planted_outside_rel, 'outside_found': ctl_outside},
                         'planted_sibling': {'rel': planted_sibling_rel, 'outside_found': ctl_sibling},
                         'planted_own': {'rel': planted_own_rel, 'holders': own_holders, 'outside_found': ctl_own}}}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    for cell in W.CELL_ORDER:
        decl = W.declaration_for(cell)
        cap = W.spec_cap(cell)
        tail_on = {'tail_enabled_for_this_run': True}
        try:
            good = W.assert_resettle_preconditions(decl, spec_like(cell), tail_on, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        other_arm = 'e_c4' if cell != 'e_c4' else 'e_c2'
        neg_specs = {'cap_wrong': (spec_like(cell, cap=cap - 1), tail_on, True),
                     'tail_off': (spec_like(cell), {'tail_enabled_for_this_run': False}, True),
                     'aa_off': (spec_like(cell), tail_on, False),
                     'floor_overridden_in_the_spec': (spec_like(cell, config_overrides={'ess_ageing_baseline': dict(
                         configuration_now()['ess_ageing_baseline'], minimum_soh=0.5)}), tail_on, True),
                     'ess_file_pin_other': (spec_like(cell, config_overrides={'ess_params_file': {
                         'path': W.ESS_PARAMS_REL, 'sha256': '0' * 64}}), tail_on, True)}
        if W.CELLS[cell]['arm'] is not None:
            neg_specs['entry_variant_swapped'] = (spec_like(cell, entry_overrides={
                'model_variant': W.arm_variant(other_arm)}), tail_on, True)
            neg_specs['variant_label_missing'] = (spec_like(cell, entry_overrides={'model_variant_label': None}),
                                                  tail_on, True)
        else:
            neg_specs['variant_on_the_pb_entry'] = (spec_like(cell, entry_overrides={
                'model_variant': W.arm_variant('e_c2')}), tail_on, True)
        negatives = {}
        for name, (sp, tail, aa) in neg_specs.items():
            try:
                W.assert_resettle_preconditions(decl, sp, tail, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'w132_schema': dict(decl, schema=V.DECLARATION_SCHEMA),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'rule_v2': dict(decl, settling_rule=dict(rule, module='settling_criterion_v2', version=2)),
               'retry_tier': dict(decl, settling_rule=dict(rule, retry_tier='tier1')),
               'cap_rule_changed': dict(decl, cap_rule=dict(decl['cap_rule'], ceiling=999)),
               'exit_capture_off': dict(decl, captures=dict(decl['captures'], ipopt_exit_by_block=False)),
               'floor_changed': dict(decl, floor_row_lower_expected=0.5),
               'arm_changed': dict(decl, arm='C4' if decl['arm'] != 'C4' else 'C2'),
               'model_variant_changed': dict(decl, model_variant=W.arm_variant(other_arm))}
        if decl['replay_reference'] is not None:
            bad.update({'wrong_k0': dict(decl, first_residual_pass_expected=decl['first_residual_pass_expected'] - 1),
                        'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], sha256='0' * 64)),
                        'abort_false': dict(decl, abort_on_replay_divergence=False),
                        'no_reference': dict(decl, replay_reference=None),
                        'lapses_changed': dict(decl, original_lapses_after_k0=[999])})
        else:
            bad.update({'invented_reference': dict(decl, replay_reference={'per_cycle_record': 'x', 'sha256': '0' * 64}),
                        'abort_true': dict(decl, abort_on_replay_divergence=True)})
        refused = {}
        for name, d in bad.items():
            try:
                W.validate_settling_resettle(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:160]
        cell_ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values())
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'n_checklist_items': len(good) if isinstance(good, dict) else None,
                     'checklist_failing': (sorted(k for k, v in good.items() if v is not True)
                                           if isinstance(good, dict) and 'error' not in good else good.get('error')),
                     'negative_controls': negatives, 'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
#  D -- the harness is the pre-W135 harness plus the prepared patch, exactly
# ======================================================================================================================
def _patch_blocks(patch_text):
    """(old block, new block, n added, n removed) of the patch's single hunk (context kept in both)."""
    lines = patch_text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith('@@'))
    old, new, added, removed = [], [], 0, 0
    for ln in lines[start + 1:]:
        if ln.startswith('@@') or ln.startswith('diff --git'):
            raise RuntimeError('the patch must hold exactly one hunk')
        tag, body = ln[:1], ln[1:]
        if tag == ' ':
            old.append(body)
            new.append(body)
        elif tag == '-':
            old.append(body)
            removed += 1
        elif tag == '+':
            new.append(body)
            added += 1
        elif ln.strip() == '':
            old.append(body or '\n')
            new.append(body or '\n')
        else:
            raise RuntimeError(f'unexpected patch line {ln!r}')
    return ''.join(old), ''.join(new), added, removed


def tests_D():
    _pre, pre_sha, pre_src = _harness_pre_w135()
    patch_text = open(_abs(PATCH_REL)).read()
    old_block, new_block, added, removed = _patch_blocks(patch_text)
    pre_text = pre_src.decode()
    patched = pre_text.replace(old_block, new_block) if pre_text.count(old_block) == 1 else None
    now_text = open(H.HARNESS_PATH).read()
    now_sha = H.sha256_file(H.HARNESS_PATH)
    fn_src = inspect.getsource(H.resettle_hooks_module)
    parts = {
        'pre_w135_harness_as_pinned': pre_sha == PRE_W135_HARNESS['sha256'],
        'patch_committed_clean': _clean(PATCH_REL),
        'patch_is_three_added_lines_no_removal': added == 3 and removed == 0,
        'patch_context_found_once_in_the_pre_w135_harness': patched is not None,
        'harness_now_equals_pre_w135_plus_the_patch_exactly': patched is not None and patched == now_text,
        'harness_sha256_is_the_pinned_post_patch_sha256': now_sha == HARNESS_POST_PATCH_SHA256,
        'harness_committed_clean': _clean('p515_s44_campaign_harness.py'),
        'router_w132_branch_first_then_w135_then_w118': (
            0 <= fn_src.find('W132C.is_v3_declaration(value)') < fn_src.find('W135C.is_w135_declaration(value)')
            < fn_src.find('import p515_s53_w118_resettle_hooks as W118C')),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'harness_sha256_now': now_sha,
            'patch_sha256': _sha(PATCH_REL), 'n_added': added, 'n_removed': removed}


# ======================================================================================================================
#  X -- the pure readers on committed records
# ======================================================================================================================
def tests_X():
    res = {}
    unit_lines = _read_jsonl(_abs(os.path.join(UNIT_SETTLED_EVAL_DIR, W.FLOOR_SIDECAR_FILE)))
    fy, ok = W.floor_year_reading(unit_lines, UNIT_SETTLED['k_star'])
    res['X1_unit_floor_binds_2035_at_172'] = {'holds': bool(ok and fy.get('floor_year') == 2035
                                                            and fy.get('active_block_years') == [2035]),
                                              'reading': fy}
    s46_lines = _read_jsonl(_abs(os.path.join(S46_C2_EVAL_DIR, W.FLOOR_SIDECAR_FILE)))
    last = max(x['cycle'] for x in s46_lines)
    fy2, ok2 = W.floor_year_reading(s46_lines, last)
    res['X2_s46_c2_at_050_no_floor_row_active'] = {'holds': bool(ok2 and fy2.get('floor_year') is None
                                                                 and all(b['soh_min'] == 0.5 for b in fy2['per_block'])),
                                                   'reading': fy2}
    fy3, ok3 = W.floor_year_reading(unit_lines, 10 ** 6)
    res['X3_missing_cycle_is_a_capture_failure'] = {'holds': ok3 is False and 'error' in fy3}
    rows = _read_jsonl(_abs(os.path.join(UNIT_SETTLED_EVAL_DIR, 'per_cycle_record.jsonl')))
    eq = W.trajectory_equality(rows, rows, UNIT_SETTLED['k_star'])
    tampered = copy.deepcopy(rows)
    tampered[149]['gross_operational_cost'] = math.nextafter(tampered[149]['gross_operational_cost'], math.inf)
    neq = W.trajectory_equality(tampered, rows, UNIT_SETTLED['k_star'])
    short = W.trajectory_equality(rows[:100], rows, UNIT_SETTLED['k_star'])
    res['X4_trajectory_equality'] = {
        'holds': bool(eq['reproduced'] and not neq['reproduced'] and neq['first_difference']['cycle'] == 150
                      and (neq['first_difference_any_shared_field_report_only'] or {}).get('cycle') == 150
                      and not short['reproduced'] and short['first_difference'].get('missing_in') == 'run'),
        'self': eq, 'one_ulp_at_150': neq['first_difference'], 'run_shorter': short['first_difference']}
    return {'holds': all(v['holds'] for v in res.values()), 'tests': res}


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('M', tests_M), ('K', tests_K),
            ('P', tests_P), ('D', tests_D), ('X', tests_X))

CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'p515_s53_w135_resettle_ext_hooks.py', PATCH_REL,
                         'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w132_resettle_v3_checks.py',
                         'settling_criterion_v3.py', 'settling_criterion_v2.py', 'settling_criterion.py',
                         'p515_s53_w118_resettle_hooks.py', 'p515_s53_w118_resettle_checks.py',
                         'p515_s44_campaign_harness.py', 'gate_result_io.py', 'interface_dual_capture.py',
                         'shared_resources_planning.py', 'admm_anderson_acceleration.py',
                         'shared_energy_storage_data.py', 'shared_energy_storage_parameters.py', 'network.py',
                         'helper_functions.py', 'p515_s53_w105_settling_extension_hooks.py',
                         'p515_s53_w101_settling_continuation_hooks.py', 'p515_s53_w105_extension_checks.py',
                         'p515_s53_w101_continuation_checks.py', 'p515_s53_w98_continuation_checks.py',
                         'p515_s53_w112_consensus_gap.py', 'p515_g_g1_g4_admm_gates.py',
                         'p515_s46_variant_checks.py', 'p515_s40_polish_gap.py', 'p515_gate_result_bool_typing_test.py')


def run_all_checks(sections=SECTIONS):
    out = {}
    ok = True
    for sid, fn in sections:
        t0 = time.time()
        try:
            r = fn()
        except Exception as error:  # noqa: BLE001 -- recorded as a failing section
            r = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        r_ok = r.get('holds') is True
        out[sid] = {'holds': r_ok, 'wall_s': time.time() - t0, 'result': r}
        ok = ok and r_ok
    return {'all_hold': ok, 'p_max': W.P_MAX, 'constants': SC3.constants(W.P_MAX), 'readings': SC3.READINGS,
            'sections': out}


def run_typing_test(out_dir):
    """W100's repository-wide boolean-typing test, output to a new write-once file (subprocess; stdlib only)."""
    out_path = os.path.join(out_dir, TYPING_OUT)
    if os.path.exists(out_path):
        raise SystemExit(f'refusing to overwrite existing artifact: {out_path}')
    t0 = time.time()
    proc = subprocess.run([sys.executable, '-u', 'p515_gate_result_bool_typing_test.py', '--out', out_path], cwd=REPO,
                          capture_output=True, text=True)
    doc = json.load(open(out_path)) if os.path.isfile(out_path) else None
    return {'exit_code': proc.returncode, 'pass': proc.returncode == 0, 'wall_s': time.time() - t0,
            'out': os.path.relpath(out_path, REPO), 'stdout_tail': proc.stdout[-2000:], 'stderr_tail': proc.stderr[-2000:],
            'verdict': (doc or {}).get('verdict'), 'n_files_scanned': (doc or {}).get('n_files_scanned')}


def main():
    started = _utc()
    out_dir = _abs(OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_FILE, OUT_MANIFEST, TYPING_OUT):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    res = run_all_checks()
    typing = run_typing_test(out_dir)
    code_pins = {rel: H.sha256_file(_abs(rel)) for rel in CODE_PINNED_BY_CHECKS}
    guards = {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}
    doc = {'schema': 'p515_s53_w135_zero_solve_checks_v1',
           'task': ('W135 (PLANNER_BRIEF_2026-09-13.md Addendum 58 Supplement; TASKS.md Addendum 58: the ageing arms at '
                    'minimum SoH 0.70, option (b); pb_y2025_n5 re-run under v3 at the end of the queue)'),
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'code_sha256': code_pins, 'guards': guards, 'W_bool_typing_test': typing, **res,
           'all_hold_including_typing_test': bool(res['all_hold'] and typing['pass'])}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    tpath = os.path.join(out_dir, TYPING_OUT)
    if os.path.isfile(tpath):
        manifest[os.path.relpath(tpath, REPO)] = H.sha256_file(tpath)
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    for sid, r in res['sections'].items():
        print(f"[W135-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W135-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W135-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W135-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
