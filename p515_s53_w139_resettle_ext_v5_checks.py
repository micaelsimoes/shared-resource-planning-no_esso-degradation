"""
P5.15 Addendum 60, Planner task W139 item 6 -- ZERO-SOLVE checks for the re-settling EXTENSION RE-FROZEN AGAINST v5
(ageing E, six model-variant arms at minimum SoH 0.70, and pb_y2025_n5_v5; 7 cells; frozen_s53_resettle_ext_spec_v2,
predecessor a9d6d1a7). W135's section set (as W137 ran it), re-targeted to v5; the W135 checks module is imported for
its rule-independent helpers and NOT edited.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the imported checks
modules arm their own; every guard is verified at 0). NOTHING IS SOLVED. Section M BUILDS ESSO models (production's
`_build_subproblem`, through the campaign child's own configuration hook) on fresh SRP1 planning objects -- builds, never
solves.

WHAT IS CHECKED
  V   the stop rule v5 as the v5 campaign validates it (`p515_s53_w139_resettle_v5_checks.tests_V`, reused).
  O   the seven cells (W135's section O restated for this module: the original records, the file in force, the settled
      unit 3f084f2f and its natural regime); the Phase B cell's identity renamed pb_y2025_n5_v5.
  S   the v5 stage spec frozen, committed and naming l_195156fa last (the v5 campaign's last cell's results / manifest
      state REPORTED, not required: the launcher refuses the cells until they are committed); no launcher alive; no
      uncommitted change to a file this run uses; production since the originals (report).
  H   the hooks through the REAL nine v5 wrappers (the v5 checks' `drive`, with this module's declarations and state):
      H1 pb gated with its ORIGINAL values bitwise through k0 = 110, then certifies; H2 a one-ulp divergence aborts; H3
      an E cell certifies; H4 an E cell creeps to its dynamic rule cap; H5 a primary Acceptable within 10x inside the
      window does not veto, a recovery Acceptable does; H6 the REAL install with the harness dispatch; H7 the summary
      schema; H8-H11 the real-production tests of the same wrappers (reused from the v5 checks).
  M   per arm, ON FRESHLY BUILT MODELS: W135's section M with this module's declarations (the arm variants are W135's).
  K   keys: the 7 extension-v5 keys follow the formula over the pre-W139 base key, are distinct, differ from every
      original, from the 7 W135 (v1 extension) keys, from the 38 v3, the 38 v4 and the 36 v5 keys, and appear in no
      committed spec outside the extension-v5 root; planted controls; the pre-W139 harness refuses the declaration. (The
      repository-wide key regression over every committed spec is the v5 checks' section K, which covers this family's
      own root.)
  P   `assert_resettle_preconditions` for every cell (W135's negatives, the v5 rule).
  D   the harness router: the v5 checks' section D (pre-W139 + the v5 branch exactly).
  X   the pure readers on committed records (W135's section X, reused unchanged).
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w139_resettle_ext_v5_checks.py \\
      > data/SRP1/Results/P515S53/w139_resettle_ext_v5/zero_solve_checks_launch.log 2>&1
"""
import contextlib
import copy
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
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W139 extension v5 zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w139_resettle_ext_v5_hooks as W  # noqa: E402
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w135_resettle_ext_hooks as W135  # noqa: E402
import p515_s53_w139_resettle_v5_checks as K139  # noqa: E402 -- the v5 suite and helpers (arms its own guard)
import p515_s53_w135_resettle_ext_checks as K135  # noqa: E402 -- W135's helpers (arms its own guard)
import settling_criterion_v5 as SC5  # noqa: E402
import p515_s46_variant_checks as M46V  # noqa: E402 -- W20's variant definitions and model fingerprint (arms its guard)

K137 = K139.K137
K132 = K139.K132
K118, K105, K98 = K132.K118, K132.K105, K132.K98


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w139_ext_checks', GUARD),) + tuple(K139.GUARDS) + tuple(K135.GUARDS)
                 + (('s46_variant_checks_imported', M46V.GUARD),))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K139.W139_EXT_ROOT_REL
OUT_DIR_REL = os.path.join(ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w139_ext_zero_solve_checks.json'
OUT_MANIFEST = 'w139_ext_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w139_ext_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = K139.EXT_CAMPAIGN_ID_PREFIX
KEY_EXCLUDED_ROOTS = (ROOT_REL,)
V5_ROOT_REL = W.V5_ROOT_REL
UNIT_SETTLED_EVAL_DIR = K135.UNIT_SETTLED_EVAL_DIR
UNIT_SETTLED = K135.UNIT_SETTLED
AGEING_MECHANISM = K135.AGEING_MECHANISM
FILE_IN_FORCE = K135.FILE_IN_FORCE
IDENTITY_KEYS = K132.IDENTITY_KEYS
GATED_IDENTITY_KEYS = K132.GATED_IDENTITY_KEYS
PRODUCTION_FILES = K132.PRODUCTION_FILES
PB = 'pb_y2025_n5_v5'
W135_EXT_SPEC = {'path': os.path.join(K137.W135_ROOT_REL, 'frozen_s53_resettle_ext_spec_v1_a9d6d1a7.json'),
                 'sha8': 'a9d6d1a7'}


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


configuration_now = K135.configuration_now


def key_kwargs(cell):
    cfg = configuration_now()
    return dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
                model_variant=W.arm_variant(cell), convergence_depth_tail=cfg['convergence_depth_tail'])


def candidate_of(cell):
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
#  V -- the stop rule v5 (the v5 suite, reused)
# ======================================================================================================================
def tests_V():
    r = K139.tests_V()
    return {'holds': r['holds'] is True, 'source': 'p515_s53_w139_resettle_v5_checks.tests_V (reused unchanged)',
            'tests': {k: v.get('ok') for k, v in r['tests'].items()}}


# ======================================================================================================================
#  O -- the seven cells, the file in force, the references (W135's section O on this module's cells)
# ======================================================================================================================
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
        w135_cell = {v: k for k, v in W.RENAMED.items()}.get(cell, cell)
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
            'cell_equals_the_w135_cell_but_the_rule': (
                {k: v for k, v in c.items() if k != 'w135_cell'} == W135.CELLS[w135_cell]
                and W.cap_rule(cell) == W135.cap_rule(w135_cell) and W.arm_variant(cell) == W135.arm_variant(w135_cell)),
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
                       'w135_cell': w135_cell, 'model_variant': W.arm_variant(cell), 'orig_eval_key': c['orig_eval_key'],
                       'orig_cycles': n_rows, 'original_spec': os.path.join(c['orig_root'], c['orig_spec']),
                       'original_git_head': spec.get('git_head'), 'per_cycle_record': rel,
                       'per_cycle_record_sha256': sha, 'cap_rule': W.cap_rule(cell), 'spec_cap': W.spec_cap(cell)}
    res['cells'] = cells
    cal = ess['ageing']['calibration']
    res['closed_forms_by_arm_report'] = {a: W.closed_form_expected(mv, cal['cycles_n'], cal['reference_dod_d'])
                                         for a, mv in W.ARMS.items()}
    unit = K135._unit_natural_regime()
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
        'cell_order_is_w135_with_pb_renamed': W.CELL_ORDER == tuple(W.RENAMED.get(c, c) for c in W135.CELL_ORDER),
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
        'w135_extension_spec_v1_committed_clean': _clean(W135_EXT_SPEC['path'])
        and _sha(W135_EXT_SPEC['path']).startswith(W135_EXT_SPEC['sha8']),
    }
    res['holds'] = all(res['parts'].values())
    return res


# ======================================================================================================================
#  S -- the v5 stage spec frozen (the v5 last cell reported); production since the originals
# ======================================================================================================================
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w139_resettle_v5_campaign', 'p515_s53_w139_resettle_ext_v5_campaign',
                          'p515_s53_w137_resettle_v4_campaign', 'p515_s53_w132_resettle_v3_campaign',
                          'p515_s53_w135_resettle_ext_campaign', 'p515_s53_w118_resettle_campaign')


def _launchers_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def v5_spec_state():
    """The v5 stage spec (found by its series prefix; its name carries its sha256) is committed clean and names
    l_195156fa as its last cell. (ok, detail)."""
    rel, sha = W.v5_stage_spec_rel()
    doc = json.load(open(_abs(rel))) if rel else {}
    parts = {'v5_stage_spec_found_named_by_its_sha256': rel is not None,
             'v5_stage_spec_committed_clean': bool(rel) and _clean(rel),
             'v5_stage_spec_series_version_5': doc.get('series') == 'frozen_s53_resettle_spec' and doc.get('version') == 5,
             'v5_last_cell_is_l_195156fa': (doc.get('cell_order') or [None])[-1] == W.V5_LAST_CELL == 'l_195156fa'}
    return all(parts.values()), {'parts': parts, 'path': rel, 'sha256': sha}


def v5_done():
    """The v5 campaign's last cell (l_195156fa) has its results and manifest committed and clean, and the v5 stage spec
    is frozen as v5_spec_state requires. The LAUNCH gate of this extension (the launcher requires it for --run)."""
    spec_ok, sd = v5_spec_state()
    res_rel, man_rel = W.v5_last_cell_files_rel()
    parts = {'v5_stage_spec': spec_ok,
             'v5_last_cell_results_committed_clean': os.path.isfile(_abs(res_rel)) and _clean(res_rel),
             'v5_last_cell_manifest_committed_clean': os.path.isfile(_abs(man_rel)) and _clean(man_rel)}
    return all(parts.values()), {'parts': parts, 'results': res_rel, 'manifest': man_rel, 'v5_stage_spec': sd}


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
    spec_ok, sd = v5_spec_state()
    done, d = v5_done()
    alive = _launchers_alive()
    prov = production_since_originals(CODE_PINNED_BY_CHECKS)
    parts = {'v5_stage_spec_frozen_committed_names_l_195156fa_last': spec_ok,
             'no_launcher_alive': not alive,
             'no_uncommitted_change_to_a_file_this_run_uses': prov['ok']}
    return {'holds': all(parts.values()), 'parts': parts, 'v5_stage_spec': sd,
            'v5_last_cell_committed_REPORTED_launch_gate_only': {'done': done, **d}, 'launchers_alive': alive,
            'production': prov}


# ======================================================================================================================
#  H -- the hooks through the real v5 wrappers with extension declarations
# ======================================================================================================================
def drive(cell, variant='certify', first_pass_at=50, plan=None):
    return K139.drive(cell, variant, first_pass_at=first_pass_at, plan=plan, module=W, state_cls=W.ResettleStateExtV5)


def _h_layering(srp):
    """The real install (the extension's context manager) and the harness dispatch of this family."""
    import p515_s53_w137_resettle_v4_hooks as V4
    before = {name: getattr(srp, name) for name in W.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w139_ext_layering_')
    cell = PB
    n = W.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    import interface_dual_capture as IDC
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
    w132_cell = V.CELL_ORDER[0]
    dispatch = {
        'ext_v5_to_this_extension': H.resettle_hooks_module(W.declaration_for(cell)) is W,
        'ext_v5_e_cell_to_this_extension': H.resettle_hooks_module(W.declaration_for('e_c2')) is W,
        'v5_to_w139': H.resettle_hooks_module(V5.declaration_for('b_4649234b')) is V5,
        'w135_v1_to_w135': H.resettle_hooks_module(W135.declaration_for('e_c2')) is W135,
        'v4_to_w137': H.resettle_hooks_module(V4.declaration_for(w132_cell)) is V4,
        'harness_validates_ext_v5': H.validate_settling_resettle(W.declaration_for(cell)) == W.declaration_for(cell),
        'harness_validates_w135_v1_unchanged': (H.validate_settling_resettle(W135.declaration_for('pb_y2025_n5_v4'))
                                                == W135.declaration_for('pb_y2025_n5_v4'))}
    summ = holder.get(W.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and idc_outer and all_installed
                          and all(dispatch.values()) and summ.get('phase') == 'ended' and summ.get('schema') == W.SCHEMA
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'dispatch': dispatch, 'files_written_by_install_without_cycles': files_written,
            'summary_phase': summ.get('phase'), 'summary_schema': summ.get('schema'),
            'summary_errors': summ.get('errors')}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    c = W.CELLS[PB]
    s = K139._drive_summary(drive(PB, 'certify'))
    s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                   and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                   and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                   and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                   and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                   and s['clean_capture_complete'] and s['exits_51_every_line'] and s['clean_51_every_line']
                   and s['non_clean_cycles'] == [] and s['certificate_length_after'] == 10
                   and s['lines'] == s['k_star'] == s['creep_lines'] and s['decision_version'] == 5
                   and s['holds_through_first_pass'] == [K132.HOLDS_OFF] and s['holds_after_first_pass'] == [K132.HOLDS_ON]
                   and s['t_sum_every_line'] and not s['capture_errors'])
    s.pop('summary')
    res['H1_pb_original_values_bitwise_through_k0_then_certify'] = {'holds': s['ok'], 'detail': s}
    s = K139._drive_summary(drive(PB, 'ulp_at_40'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] and 'REPLAY DIVERGED at cycle 40' in s['raised'] and s['lines'] == 40
                   and (s['first_divergence'] or {}).get('fields_differing') == ['gross_operational_cost']
                   and s['replay_bitwise_through'] == 39 and not s['summary_ok']
                   and s['stopped_by'] == 'replay_divergence_abort')
    res['H2_pb_one_ulp_at_40_aborts'] = {'holds': s['ok'], 'detail': s}
    s = K139._drive_summary(drive('e_c2_calfade', 'certify', first_pass_at=103))
    summ3 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 103 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule'
                   and s['n_overlap'] == 0 and s['exit_capture_complete'] and s['clean_capture_complete']
                   and s['exits_51_every_line'] and s['clean_51_every_line']
                   and s['holds_through_first_pass'] == [K132.HOLDS_OFF] and s['holds_after_first_pass'] == [K132.HOLDS_ON]
                   and summ3.get('schema') == W.SCHEMA and summ3.get('arm') == 'C2_calfade')
    res['H3_e_cell_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    s = K139._drive_summary(drive('e_c3_unit', 'creep', first_pass_at=60))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 60 + W.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 60
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0)
    res['H4_e_cell_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    base = K139._drive_summary(drive('e_c4', 'certify', first_pass_at=100))
    ks = base['k_star']
    probe = drive('e_c4', 'certify', first_pass_at=100)
    keys = list(probe['files'][V.CYCLE_FILE][0]['ipopt_exit_by_block'])
    dso7 = next(k for k in keys if k.startswith('DSO|7|') and k.endswith('|Winter'))
    tso = next(k for k in keys if k.startswith('TSO|'))
    s_in = K139._drive_summary(drive('e_c4', 'certify', first_pass_at=100, plan={ks - 1: {dso7: 'acc_2p51'}}))
    s_rec = K139._drive_summary(drive('e_c4', 'certify', first_pass_at=100, plan={ks - 1: {tso: 'acc_2p51@recovery'}}))
    s5 = {'primary_acceptable_2p51_inside_window': {k: v for k, v in s_in.items() if k != 'summary'},
          'recovery_acceptable_inside_window': {k: v for k, v in s_rec.items() if k != 'summary'}}
    s5['ok'] = bool(s_in['raised'] is None and s_in['status'] == 'certified' and s_in['k_star'] == ks
                    and s_in['n_vetoes'] == 0 and s_in['non_clean_cycles'] == [] and s_in['summary_ok']
                    and s_in['in_cycle_rule_equals_pure_replay']
                    and s_rec['raised'] is None and s_rec['n_lapses'] == base['n_lapses'] and s_rec['n_vetoes'] >= 1
                    and s_rec['vetoes'][0]['cycle'] == ks and s_rec['in_cycle_rule_equals_pure_replay']
                    and s_rec['summary_ok'] and (s_rec['status'] != 'certified' or s_rec['k_star'] > ks))
    s5['k_star_all_optimal'] = ks
    res['H5_e_cell_clean_rule_v5'] = {'holds': s5['ok'], 'detail': s5}
    res['H6_layering_real_install_and_dispatch'] = _h_layering(srp)
    st = W.ResettleStateExtV5(W.declaration_for('e_no_ageing'), None, W.spec_cap('e_no_ageing'), reference={}, sink=[])
    summ = st.summary()
    res['H7_summary_schema_is_the_extension_v5'] = {
        'holds': summ.get('schema') == W.SCHEMA and summ.get('arm') == 'no_ageing'
        and isinstance(st, V5.ResettleStateV5) and summ.get('criterion_version') == 5, 'schema': summ.get('schema')}
    for name, fn in (('H8_v5_exit_wrapper_on_real_production_structures_reused', K139._h_exit_real_v5),
                     ('H9_aa_hold_real_production_reused', lambda: K139._h_real_v5(srp, 'aa')),
                     ('H10_tail_hold_real_production_reused', lambda: K139._h_real_v5(srp, 'tail')),
                     ('H11_rho_hold_real_production_reused', lambda: K139._h_real_v5(srp, 'rho'))):
        try:
            res[name] = fn()
        except Exception as error:  # noqa: BLE001
            res[name] = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  M -- per-arm variant readback on freshly built models (W135's section M on this module's declarations)
# ======================================================================================================================
def tests_M():
    import p515_g_g1_g4_admm_gates as G
    import shared_energy_storage_data as SED
    import p515_s40_polish_gap as PG
    import p56a_oracle as O
    scratch = tempfile.mkdtemp(prefix='p515s53_w139_ext_checks_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    ids = []
    arms, fps = {}, {}
    try:
        _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s53_w139_ext_checks_{run_tag}_precheck')
        ids.append(f'p515s53_w139_ext_checks_{run_tag}_precheck')
        for cell in W.E_CELLS:
            decl = W.declaration_for(cell)
            mv = decl['model_variant']
            _p, sed, _cand, holder, report, models, eid = K135._built(G, SED, M46V, cell, run_tag, scratch, mv,
                                                                      floor_rows)
            ids.append(eid)
            fresh = H.model_variant_readback_models(models, sed, mv, W.UNIT_YEAR, clone=True)
            w20 = (report.get('rule_eleven_checklist') or {}).get('w20_model_variant') or {}
            floor_probe_ok = w20.get('floor_rows_identical_to_baseline_probe') is True
            try:
                floor_on_candidate_models, _counts = G._identify_soh_floor_rows(models)
                floor_on_candidate_models_note = {'identified': True,
                                                  'equal_to_precheck': floor_on_candidate_models == floor_rows}
            except RuntimeError as error:
                floor_on_candidate_models_note = {'identified': False, 'error': str(error)[:300]}
            rec = K135._synthetic_record(mv, holder, fresh)
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
            arms[cell] = {'ok': bool(ok and all(negatives.values()) and floor_probe_ok),
                          'g9_replacement_holds': ok, 'g9_parts_failing': sorted(k for k, v in det['parts'].items()
                                                                                  if v is not True),
                          'negative_controls': negatives,
                          'floor_rows_hook_probe_equal_the_precheck': floor_probe_ok,
                          'floor_identification_on_candidate_models_REPORT_ONLY': floor_on_candidate_models_note,
                          'closed_form': det.get('closed_form'),
                          'readback_node7_pre_run': (rec['model_variant_readback_pre_run'] or {}).get(
                              'per_node', {}).get('7', {}).get('readback'),
                          'readback_node7_fresh': fresh['per_node']['7']['readback'],
                          'floor_row_lower_by_node': det['terminal']['floor_row_lower']}
            if cell == W.CELL_OF_ARM[W.BASELINE_EQUIVALENT_ARM]:
                fps['variant'] = {str(n): M46V._fingerprint(m) for n, m in models.items()}
                fps['variant_ess'] = [{k: repr(getattr(e, k)) for k in ('bus', 't_cal', 'cl_nom', 'dod_nom', 'soh_min',
                                                                        'cl_eff', 'phi_cal')}
                                      for y in sed.years for e in sed.shared_energy_storages[y]]
                fps['variant_settings'] = list(SED._esso_ageing_model_settings(sed))
            del models
        _p, sed, _cand, holder, _report, models, eid = K135._built(G, SED, M46V, 'baseline_no_variant', run_tag,
                                                                   scratch, None, floor_rows)
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
#  K -- keys (this family; the repository-wide regression is the v5 checks' section K)
# ======================================================================================================================
def resettle_keys(pre=None):
    out = {}
    for cell in W.CELL_ORDER:
        key, overrides = candidate_of(cell)
        kw = key_kwargs(cell)
        base = H.evaluation_key(key, overrides, **kw)
        decl = W.declaration_for(cell)
        rkey = H.evaluation_key(key, overrides, settling_resettle=decl, **kw)
        w135_cell = {v: k for k, v in W.RENAMED.items()}.get(cell, cell)
        w135_key = H.evaluation_key(key, overrides, settling_resettle=W135.declaration_for(w135_cell), **kw)
        base_pre = pre.evaluation_key(key, overrides, **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        base_no_variant = H.evaluation_key(key, overrides, **dict(kw, model_variant=None))
        out[cell] = {'candidate_key': key, 'original_eval_key': W.CELLS[cell]['orig_eval_key'], 'base_key_now': base,
                     'resettle_key': rkey, 'w135_v1_key': w135_key, 'formula_holds': rkey == formula,
                     'base_equals_pre_w139': base == base_pre,
                     'base_carries_the_variant': (W.arm_variant(cell) is None) == (base == base_no_variant),
                     'differs_from_original': rkey != W.CELLS[cell]['orig_eval_key']
                     and base != W.CELLS[cell]['orig_eval_key'], 'differs_from_w135_v1': rkey != w135_key}
    return out


def tests_K():
    pre, pre_sha, _src = K139._harness_pre_w139()
    scanned = []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        scanned.append((rel, json.load(open(_abs(rel)))))
    keys = resettle_keys(pre)
    w135_frozen = {}
    ss1 = json.load(open(_abs(W135_EXT_SPEC['path'])))
    for c, pin in ss1['pins']['campaign_specs'].items():
        w135_frozen[c] = json.load(open(_abs(pin['path'])))['candidates'][0]['eval_key']
    v3_keys = {v['resettle_key'] for v in K132.resettle_keys().values()}
    v4_keys = {v['resettle_key'] for v in K137.resettle_keys().values()}
    v5_keys = {v['resettle_key'] for v in K139.resettle_keys().values()}
    mine = {v['resettle_key'] for v in keys.values()}
    holders = {cell: K135._key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: K135._outside_roots(h, KEY_EXCLUDED_ROOTS) for cell, h in holders.items()}
    probe = keys[W.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w139_ext_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                                 'eval_key': probe}]})
    ctl = {}
    for name, rel in (('outside', os.path.join(_P53, 'w139_ext_planted_negative_control', 'campaign_planted',
                                               'campaign_spec_planted_00000000.json')),
                      ('sibling_prefix', os.path.join(ROOT_REL + '_planted_sibling', 'campaign_planted',
                                                      'campaign_spec_planted_00000000.json')),
                      ('w135_root', os.path.join(K137.W135_ROOT_REL, 'campaign_planted',
                                                 'campaign_spec_planted_00000000.json')),
                      ('v5_root', os.path.join(K139.W139_ROOT_REL, 'campaign_planted',
                                               'campaign_spec_planted_00000000.json'))):
        ctl[name] = {'rel': rel, 'refused': rel in K135._outside_roots(
            K135._key_holders(scanned + [planted(rel)], probe), KEY_EXCLUDED_ROOTS)}
    own_rel = os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{W.CELL_ORDER[0]}',
                           f'campaign_spec_{CAMPAIGN_ID_PREFIX}{W.CELL_ORDER[0]}_00000000.json')
    own_holders = K135._key_holders(scanned + [planted(own_rel)], probe)
    try:
        pre.validate_settling_resettle(W.declaration_for(W.CELL_ORDER[0]))
        pre_refuses = False
    except ValueError:
        pre_refuses = True
    parts = {
        'resettle_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'resettle_base_equals_pre_w139_every_cell': all(v['base_equals_pre_w139'] for v in keys.values()),
        'base_key_carries_the_variant_for_e_cells_only': all(v['base_carries_the_variant'] for v in keys.values()),
        'resettle_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'resettle_keys_distinct': len(mine) == len(keys) == 7,
        'w135_v1_keys_recomputed_equal_the_frozen_a9d6d1a7_specs': all(
            w135_frozen[{v: k for k, v in W.RENAMED.items()}.get(c, c)] == keys[c]['w135_v1_key'] for c in W.CELL_ORDER),
        'resettle_keys_differ_from_every_w135_v1_key': not (mine & set(w135_frozen.values())),
        'resettle_keys_differ_from_every_v3_v4_v5_key': (not (mine & (v3_keys | v4_keys | v5_keys))
                                                         and len(v3_keys) == 38 and len(v4_keys) == 38
                                                         and len(v5_keys) == 36),
        'resettle_keys_absent_from_committed_specs_outside_own_root': not any(outside.values()),
        **{f'control_planted_{k}_refused': v['refused'] for k, v in ctl.items()},
        'control_own_campaign_spec_accepted': own_rel in own_holders and own_rel not in K135._outside_roots(
            own_holders, KEY_EXCLUDED_ROOTS),
        'control_declaration_refused_by_the_pre_w139_harness': pre_refuses,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w139_harness': {**K139.PRE_W139_HARNESS, 'sha256_loaded': pre_sha},
            'resettle_keys': keys, 'resettle_keys_in_committed_specs_all_REPORTED': holders,
            'key_excluded_roots': list(KEY_EXCLUDED_ROOTS), 'controls': ctl,
            'control_own': {'rel': own_rel, 'holders': own_holders},
            'repository_wide_regression': 'p515_s53_w139_resettle_v5_checks.tests_K (own family ext_v5, own root)'}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    import p515_s53_w137_resettle_v4_hooks as V4
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
                         'path': W.ESS_PARAMS_REL, 'sha256': '0' * 64}}), tail_on, True),
                     'tail_value_not_the_table': (spec_like(cell, config_overrides={'convergence_depth_tail': {
                         'enabled': True, 'compl_inf_tol': 1e-5}}), tail_on, True)}
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
        w135_cell = {v: k for k, v in W.RENAMED.items()}.get(cell, cell)
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'w135_v1_cell_name': dict(decl, cell='pb_y2025_n5_v4'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'w135_v1_schema': dict(decl, schema=W135.DECLARATION_SCHEMA),
               'v5_main_schema': dict(decl, schema=V5.DECLARATION_SCHEMA),
               'rule_v4': dict(decl, settling_rule=V4.settling_rule_declaration()),
               'the_w135_v1_declaration': W135.declaration_for(w135_cell),
               'reset_on_non_clean': dict(decl, settling_rule=dict(rule, reset_on_non_clean=True)),
               'clean_factor_changed': dict(decl, settling_rule=dict(rule, clean=dict(rule['clean'], factor=100.0))),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'retry_tier': dict(decl, settling_rule=dict(rule, retry_tier='tier1')),
               'cap_rule_changed': dict(decl, cap_rule=dict(decl['cap_rule'], ceiling=999)),
               'clean_capture_off': dict(decl, captures=dict(decl['captures'], exit_clean_by_block=False)),
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
#  D, X -- reused
# ======================================================================================================================
def tests_D():
    return K139.harness_router_check()


def tests_X():
    return K135.tests_X()


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('M', tests_M), ('K', tests_K),
            ('P', tests_P), ('D', tests_D), ('X', tests_X))

CODE_PINNED_BY_CHECKS = tuple(dict.fromkeys(
    (os.path.basename(__file__), 'p515_s53_w139_resettle_ext_v5_hooks.py', 'p515_s53_w135_resettle_ext_hooks.py',
     'p515_s53_w135_resettle_ext_checks.py', 'p515_s46_variant_checks.py', 'p515_s40_polish_gap.py',
     'shared_energy_storage_parameters.py') + K139.CODE_PINNED_BY_CHECKS))


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
    return {'all_hold': ok, 'p_max': W.P_MAX, 'constants': SC5.constants(W.P_MAX), 'readings': SC5.READINGS,
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
    doc = {'schema': 'p515_s53_w139_ext_zero_solve_checks_v1',
           'task': 'W139 item 6 (PLANNER_BRIEF_2026-09-13.md Addendum 60): the W135 extension re-frozen against v5',
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
        print(f"[W139-EXT-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W139-EXT-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} "
          f"wall={typing['wall_s']:.0f}s")
    print(f"[W139-EXT-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) "
          f"guards={guards}")
    print(f"[W139-EXT-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
