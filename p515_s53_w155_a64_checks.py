"""
P5.15 Addendum 64, Planner task W155 -- ZERO-SOLVE checks for the four Addendum 64 cells (g070_neutrality, h_x0_m175,
h_unit_m175, e_soh050; `p515_s53_w155_a64_hooks`), frozen under `frozen_s53_a64_cells_spec_v1_<sha8>.json`.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (every imported checks
module arms its own; all are verified at 0). `pickle.load` / `pickle.loads` are replaced by BLOCKING counters before any
project import, for the whole run, and verified at 0 (no model is loaded). NOTHING IS SOLVED. Section M BUILDS ESSO
models (production's `_build_subproblem` through the harness's own configuration hook, on fresh SRP1 planning objects) --
builds, never solves; scratch outside the repository, removed.

WHAT IS CHECKED
  V  the stop rule v6 (`p515_s53_w142_resettle_v6_checks.tests_V`, reused unchanged).
  O  the four cells: candidate provenance (S49 / S46 entries; keys and canonical forms), the file in force (minimum SoH
     0.70, C2 calibration, phi 0.985), the settled references (3f084f2f unit Q172, d110bd1a x0 Q181) committed as
     pinned, the soh arm == the file's own ageing law, I of the unit == W117's I_other.
  S  no launcher of this family (or the earlier ones) alive; no uncommitted change to a file this run uses.
  H  the hooks through the REAL nine wrappers on the W155 state (an E cell certifies, an H cell creeps to its dynamic
     rule cap), the REAL install (wrappers and SoH plumbing installed and restored; the harness dispatch).
  M  the SoH-floor plumbing ON FRESHLY BUILT MODELS (the Planner's section M):
       M1 `G._identify_soh_floor_rows` on probes built through the new path at 0.50: every row's soh_min is 0.5 exactly;
          constraint_idx and counts (6 per node) identical to the baseline; the harness's identity check (~3003)
          passed by equality (0.5 on both sides);
       M2 `H.model_variant_readback`: floor_row_lower 0.5, k 35,851.3609, phi 0.985, SoH point 'end';
       M3 the fingerprint diff (`M46V._fingerprint`) 0.50 vs baseline per node: EXACTLY the six floor-row bounds plus
          the salvage-expression constants differ (listed);
       M4 the identity control: the new path at 0.70 is fingerprint-identical to the baseline (the C2_calfade variant
          path without the plumbing, and the no-variant baseline path) and the ESS constants are repr-identical;
       M5 negative controls: the readback gate refuses floor_row_lower 0.7 on a 0.5 declaration and vice versa; the
          post-run sidecar gate passes 0.5 lines with the identity flag True and refuses a 0.7 entry / a False flag; the
          plumbing's fail-fast refuses a run where the apply did not run; the apply refuses without a captured dict;
       M6 (report-only) the salvage closed form uses min_soh 0.5.
  P  `assert_resettle_preconditions` for every cell; the ENTRY CHECKS: a flexibility multiplier allowed (= 1.75) on the
     H cells and forbidden on the SoH cells; minimum_soh forbidden on the H cells; the spec's minimum_soh is the file's
     0.70; validator refusals.
  K  keys: the formula over the pre-W155 base key; 4 distinct; differ from the extension-v6 e_c2_calfade key and the v6
     H keys; absent from every committed spec outside this root; planted controls; the pre-W155 harness refuses the
     W155 declaration; the repository-wide key regression (every committed entry's key equal under both harnesses).
  D  the router: the pre-W155 harness (ca46892b, 7914860f) plus exactly the three branch lines; dispatch of every family
     unchanged.
  W  (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w155_a64_checks.py \\
      > data/SRP1/Results/P515S53/w155_a64_cells/zero_solve_checks_launch.log 2>&1
"""
import pickle

# ---- the no-model-load guard: pickle.load / pickle.loads raise for the whole run (installed before any project import)
PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W155 checks: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W155 checks: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W155 Addendum 64 cells zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w155_a64_hooks as M  # noqa: E402
import p515_s53_w142_resettle_ext_v6_hooks as W  # noqa: E402
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w142_resettle_v6_checks as K142  # noqa: E402 -- the v6 suite and helpers (arms its own guard)
import p515_s53_w142_resettle_ext_v6_checks as KX  # noqa: E402 -- the extension-v6 helpers (arms its own guard)
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s46_variant_checks as M46V  # noqa: E402 -- W20's model fingerprint and ESSO build (arms its guard)

K139, K137, K135 = K142.K139, KX.K137, KX.K135
K132 = KX.K132
K118 = KX.K118


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w155_checks', GUARD),) + tuple(KX.GUARDS) + tuple(K142.GUARDS)
                 + (('s46_variant_checks_imported', M46V.GUARD),))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = os.path.join(_P53, 'w155_a64_cells')
OUT_DIR_REL = os.path.join(ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w155_zero_solve_checks.json'
OUT_MANIFEST = 'w155_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w155_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w155_a64_'
KEY_EXCLUDED_ROOTS = (ROOT_REL,)
# the harness before W155 (pinned by the v6 and extension-v6 stage specs 96c23404 / 84775dc4; last changed by 7914860f,
# W142's router commit)
PRE_W155_HARNESS = {'commit': '7914860f222b921106d3da7b7fed0666914a937b',
                    'sha256': 'ca46892ba1fb75913962c508f4a230ce5356ebcbbd1ef4b43031c8242ee98e8d'}
W155_BRANCH_LINES = ('    import p515_s53_w155_a64_hooks as W155C',
                     '    if W155C.is_w155_declaration(value):',
                     '        return W155C')
HARNESS_POST_W155_SHA256 = '9c6bda1969eca917c7875c1e4a84ab004a3b55edf52ad87179e12c816bbb7f21'  # pre-W155 + the branch
EXT_V6_STAGE_SPEC = {'path': os.path.join(_P53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
                     'sha8': '84775dc4'}
V6_STAGE_SPEC = {'path': os.path.join(_P53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'),
                 'sha8': '96c23404'}
X0_REF_EVAL_DIR = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                               'd110bd1a5977df1e_x0')
W117 = K132.W117
K_C2_EXPECTED_4DP = 35851.3609
PHI_EXPECTED = 0.985
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


configuration_now = K135.configuration_now   # W132's configuration + the ESS parameters pin


def source_entry(cell):
    """(source spec, its entry holding the source eval key) -- PROVENANCE of the candidate only."""
    s = M.CELLS[cell]['source']
    spec = json.load(open(_abs(os.path.join(s['root'], s['spec']))))
    hits = [e for e in spec['candidates'] if e.get('eval_key') == s['eval_key']]
    return spec, (hits[0] if len(hits) == 1 else None)


def key_kwargs(cell):
    cfg = configuration_now()
    return dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
                model_variant=M.arm_variant(cell), flex_price_multiplier=M.CELLS[cell]['flex_price_multiplier'],
                convergence_depth_tail=cfg['convergence_depth_tail'])


def spec_like(cell, decl=None, entry_overrides=None, config_overrides=None, cap=None, spec_overrides=None):
    """A campaign-spec-shaped dict for the preconditions (what the frozen spec will hold for the cell)."""
    decl = decl if decl is not None else M.declaration_for(cell)
    c = M.CELLS[cell]
    cfg = configuration_now()
    cfg.update(config_overrides or {})
    entry = {'label': cell, 'key': c['candidate_key'], 'overrides': {}, 'settling_resettle': decl}
    mv = M.arm_variant(cell)
    out = {'cap': M.spec_cap(cell) if cap is None else cap, 'configuration': cfg, 'candidates': [entry],
           'required_consecutive_cycles': 10}
    if mv is not None:
        entry['model_variant'] = mv
        entry['model_variant_label'] = H.MODEL_VARIANT_LABEL
        out['model_variant_label'] = H.MODEL_VARIANT_LABEL
    if c['flex_price_multiplier'] is not None:
        entry['flex_price_multiplier'] = c['flex_price_multiplier']
        entry['flex_price_label'] = H.FLEX_PRICE_LABEL
        out['flex_price_label'] = H.FLEX_PRICE_LABEL
    entry.update(entry_overrides or {})
    out.update(spec_overrides or {})
    return out


# ======================================================================================================================
#  V -- the stop rule v6 (reused)
# ======================================================================================================================
def tests_V():
    r = K142.tests_V()
    return {'holds': r['holds'] is True, 'source': 'p515_s53_w142_resettle_v6_checks.tests_V (reused unchanged)',
            'tests': {k: v.get('ok') for k, v in r['tests'].items()}}


# ======================================================================================================================
#  O -- the cells, the file in force, the references
# ======================================================================================================================
def unit_reference():
    ed = M.UNIT_REF_EVAL_DIR
    rows = _read_jsonl(_abs(os.path.join(ed, 'per_cycle_record.jsonl')))
    dec = json.load(open(_abs(os.path.join(ed, 'settling_decision.json'))))
    return {'eval_dir': ed, 'per_cycle_record_sha256': _sha(os.path.join(ed, 'per_cycle_record.jsonl')),
            'committed_clean': _clean(os.path.join(ed, 'per_cycle_record.jsonl')), 'n_rows': len(rows),
            'k_star': dec.get('k_star'), 'status': dec.get('status'), 'band_width': dec.get('band_width'),
            'Q_172': next((r['gross_operational_cost'] for r in rows if r['cycle'] == 172), None)}


def tests_O():
    cells = {}
    ess = json.load(open(_abs(M.ESS_PARAMS_REL)))
    cal = ess['ageing']['calibration']
    w117 = json.load(open(_abs(W117['path'])))
    unit_I = {c['I_other'] for c in w117['claims'] if c['claim_id'] in ('H:m1.5:value_minus_I', 'H:m2:value_minus_I',
                                                                       'E:n7_4h_e1_C2_calfade:value_minus_I')}
    for cell in M.CELL_ORDER:
        c = M.CELLS[cell]
        spec, e = source_entry(cell)
        nodes = {int(k): tuple(map(float, v)) for k, v in ((e or {}).get('canonical') or {}).get('nodes', {}).items()}
        parts = {
            'source_entry_found': e is not None,
            'source_entry_label_as_pinned': (e or {}).get('label') == c['source']['label'],
            'source_candidate_key_is_the_cell_candidate': (e or {}).get('key') == c['candidate_key'],
            'source_canonical_nodes_are_the_cell_nodes': nodes == c['nodes'],
            'source_investment_year_2025': ((e or {}).get('canonical') or {}).get('investment_year') == c['investment_year']
            == M.UNIT_YEAR,
            'candidate_key_recomputed_from_the_canonical_form': H.candidate_key(H.canonical_candidate(
                c['nodes'], investment_year=c['investment_year'])) == c['candidate_key'],
            'cap_within_ceiling': M.spec_cap(cell) <= c['cap_ceiling'] == 300,
            'ungated_first_evaluation': c['gated'] is False and M.declaration_for(cell)['replay_reference'] is None,
        }
        if c['kind'] == 'soh':
            parts.update({
                'soh_arm_is_c2_calfade': c['arm'] == 'C2_calfade',
                'soh_variant_valid_in_the_harness': _jt(H.validate_model_variant(M.arm_variant(cell)))
                == _jt(M.arm_variant(cell)),
                'soh_minimum_soh_float_in_0_1': isinstance(c['minimum_soh'], float) and 0.0 <= c['minimum_soh'] < 1.0,
                'soh_variant_has_no_key_that_could_change_the_floor': 'minimum_soh' not in H.MODEL_VARIANT_KEYS,
            })
        else:
            parts.update({
                'h_source_is_the_s49_flex_ladder_entry': (e or {}).get('flex_price_multiplier') == 1.5,
                'h_multiplier_1_75_valid_and_keyed': H.flex_price_multiplier_in_key(c['flex_price_multiplier']) == 1.75,
            })
        cells[cell] = {'ok': all(parts.values()), 'parts': parts, 'kind': c['kind'], 'source': c['source'],
                       'source_spec_git_head': spec.get('git_head'), 'candidate_key': c['candidate_key']}
    unit = unit_reference()
    x0_rows = _read_jsonl(_abs(os.path.join(X0_REF_EVAL_DIR, 'per_cycle_record.jsonl')))
    x0_dec = json.load(open(_abs(os.path.join(X0_REF_EVAL_DIR, 'settling_decision.json'))))
    parts = {
        'every_cell_ok': all(v['ok'] for v in cells.values()), 'n_cells_4': len(cells) == 4,
        'cell_order_as_the_task': M.CELL_ORDER == ('g070_neutrality', 'h_x0_m175', 'h_unit_m175', 'e_soh050'),
        'ess_params_now_is_the_pinned_file': _sha(M.ESS_PARAMS_REL) == M.ESS_PARAMS_SHA256 and _clean(M.ESS_PARAMS_REL),
        'file_minimum_soh_070_phi_0985_calibration_c2': (
            ess['ageing']['minimum_soh'] == M.FILE_MINIMUM_SOH == 0.7
            and ess['ageing']['calendar_retention_per_year'] == 0.985
            and {k: cal[k] for k in ('status', 'cycles_n', 'reference_dod_d', 'eol_retention_r')}
            == {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.8}),
        'declared_ess_baseline_minimum_soh_070': configuration_now()['ess_ageing_baseline']['minimum_soh'] == 0.7,
        'c2_calfade_arm_equals_the_file_ageing_law': (
            W.ARMS['C2_calfade'] == {'eol_retention_r': cal['eol_retention_r'],
                                     'calendar_retention_per_year': ess['ageing']['calendar_retention_per_year'],
                                     'available_energy_soh_point': 'end', 'ageing_enabled': True}),
        'unit_reference_3f084f2f_as_pinned_and_committed': (unit['per_cycle_record_sha256']
                                                            == M.UNIT_REF['per_cycle_record_sha256']
                                                            and unit['committed_clean']),
        'unit_reference_certified_at_172': unit['k_star'] == 172 == M.UNIT_REF['k_star'] and unit['status'] == 'certified',
        'x0_reference_d110bd1a_certified_at_181_committed': (x0_dec.get('k_star') == 181 == M.X0_REF['k_star']
                                                             and x0_dec.get('status') == 'certified'
                                                             and _clean(os.path.join(X0_REF_EVAL_DIR,
                                                                                     'per_cycle_record.jsonl'))
                                                             and max(r['cycle'] for r in x0_rows) == 181),
        'unit_I_is_the_w117_I_other': unit_I == {M.UNIT_I},
        'model_variant_label_and_rtol_as_the_harness': (M.MODEL_VARIANT_LABEL == H.MODEL_VARIANT_LABEL
                                                        and M.READBACK_RTOL == H.MODEL_VARIANT_READBACK_RTOL),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'cells': cells, 'unit_reference_3f084f2f': unit,
            'x0_reference_d110bd1a': {'eval_dir': X0_REF_EVAL_DIR, 'k_star': x0_dec.get('k_star'),
                                      'band_width': x0_dec.get('band_width')},
            'inputs_now': {'case_file': {'path': H.CASE_FILE_REL, 'sha256': H.sha256_file(H.CASE_FILE)},
                           'ess_params_file': {'path': M.ESS_PARAMS_REL, 'sha256': _sha(M.ESS_PARAMS_REL),
                                               'minimum_soh': ess['ageing']['minimum_soh']},
                           'configuration_now': configuration_now()}}


# ======================================================================================================================
#  S -- launchers, uncommitted files
# ======================================================================================================================
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w155_a64_campaign', 'p515_s53_w142_resettle_v6_campaign',
                          'p515_s53_w142_resettle_ext_v6_campaign', 'p515_s53_w139_resettle_v5_campaign',
                          'p515_s53_w139_resettle_ext_v5_campaign', 'p515_s53_w137_resettle_v4_campaign',
                          'p515_s53_w132_resettle_v3_campaign', 'p515_s53_w135_resettle_ext_campaign',
                          'p515_s53_w118_resettle_campaign')


def launchers_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def uncommitted_used(used_files=()):
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    used = set(used_files) | set(PRODUCTION_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN)
    dirty_used = [d for d in dirty if d.split()[-1] in used]
    return {'uncommitted_tracked_py': dirty, 'uncommitted_files_this_run_uses': dirty_used,
            'head': H._git(['rev-parse', 'HEAD']), 'ok': not dirty_used}


def tests_S():
    alive = launchers_alive()
    prov = uncommitted_used(CODE_PINNED_BY_CHECKS)
    parts = {'no_launcher_alive': not alive, 'no_uncommitted_change_to_a_file_this_run_uses': prov['ok']}
    return {'holds': all(parts.values()), 'parts': parts, 'launchers_alive': alive, 'production': prov}


# ======================================================================================================================
#  H -- the hooks through the real wrappers; the real install
# ======================================================================================================================
def drive(cell, variant='certify', first_pass_at=50, plan=None):
    return K142.drive(cell, variant, first_pass_at=first_pass_at, plan=plan, module=M, state_cls=M.ResettleStateW155)


def _install_and_restore(cell):
    """The REAL install (`M.settling_resettle_hooks`) for one cell, no cycles: the nine wrappers installed and, for a SoH
    cell, the plumbing's three kinds of attribute patched; every attribute restored on exit; the harness dispatches the
    declaration here and validates it unchanged."""
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    soh = M.CELLS[cell]['kind'] == 'soh'
    targets = [(srp, n) for n in M.WRAPPED] + [(G, 's38_pf_capture_hooks'), (G, 'esso_capture_hooks')]
    hm = M._harness_modules()
    targets += [(mod, 'apply_model_variant') for _n, mod in hm]
    before = {(id(o), n): getattr(o, n) for o, n in targets}
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w155_install_')
    try:
        with M.settling_resettle_hooks(scratch, M.declaration_for(cell), holder, cap=M.spec_cap(cell)) as st:
            during = {(id(o), n): getattr(o, n) for o, n in targets}
            nine = all(during[(id(srp), n)] is not before[(id(srp), n)] for n in M.WRAPPED)
            plumb = {f'{getattr(o, "__name__", "?")}.{n}': during[(id(o), n)] is not before[(id(o), n)]
                     for o, n in targets if o is not srp}
            st_ok = isinstance(st, M.ResettleStateW155) and ((st.plumbing is not None) == soh)
        files_written = sorted(os.listdir(scratch))
    finally:
        shutil.rmtree(scratch)
    after = {(id(o), n): getattr(o, n) for o, n in targets}
    restored = all(after[k] is before[k] for k in before)
    summ = holder.get(M.SUMMARY_KEY) or {}
    dispatch = {'harness_routes_here': H.resettle_hooks_module(M.declaration_for(cell)) is M,
                'harness_validates_unchanged': H.validate_settling_resettle(M.declaration_for(cell))
                == M.declaration_for(cell)}
    ok = bool(nine and restored and st_ok and all(dispatch.values()) and summ.get('schema') == M.SCHEMA
              and not summ.get('errors')
              and (all(plumb.values()) if soh else not any(plumb.values()))
              and (('soh_floor_plumbing' in summ) == soh))
    return {'holds': ok, 'nine_wrappers_installed': nine, 'plumbing_attributes_patched': plumb,
            'every_attribute_restored_on_exit': restored, 'state_ok': st_ok, 'dispatch': dispatch,
            'harness_modules_patched': [n for n, _m in hm], 'files_written_without_cycles': files_written,
            'summary_schema': summ.get('schema'), 'summary_phase': summ.get('phase'),
            'summary_soh_plumbing': summ.get('soh_floor_plumbing')}


def tests_H():
    res = {}
    s = K142.drive_summary(drive('e_soh050', 'certify', first_pass_at=103))
    summ = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 103 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule'
                   and s['n_overlap'] == 0 and s['exit_capture_complete'] and s['clean_capture_complete']
                   and s['exits_51_every_line'] and s['clean_51_every_line']
                   and s['holds_through_first_pass'] == [K132.HOLDS_OFF] and s['holds_after_first_pass'] == [K132.HOLDS_ON]
                   and summ.get('schema') == M.SCHEMA and summ.get('minimum_soh_declared') == 0.5)
    res['H1_soh_cell_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    s = K142.drive_summary(drive('h_unit_m175', 'creep', first_pass_at=110))
    summ = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 110 + M.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 110
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0
                   and summ.get('flex_price_multiplier_expected') == 1.75 and summ.get('minimum_soh_declared') is None)
    res['H2_h_cell_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    s = K142.drive_summary(drive('h_x0_m175', 'gap', first_pass_at=100))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['in_cycle_rule_equals_pure_replay'] and s['summary_ok']
                   and s['n_overlap'] == 0)
    res['H3_h_x0_gap_clause_variant_replays_purely'] = {'holds': s['ok'], 'detail': s}
    for cell in M.CELL_ORDER:
        try:
            res[f'H4_real_install_and_restore_{cell}'] = _install_and_restore(cell)
        except Exception as error:  # noqa: BLE001
            res[f'H4_real_install_and_restore_{cell}'] = {'holds': False, 'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}
    st = M.ResettleStateW155(M.declaration_for('h_x0_m175'), None, M.spec_cap('h_x0_m175'), reference={}, sink=[])
    summ = st.summary()
    res['H5_summary_schema'] = {'holds': summ.get('schema') == M.SCHEMA and summ.get('kind') == 'flex'
                                and isinstance(st, V6.ResettleStateV6) and summ.get('criterion_version') == 6
                                and 'soh_floor_plumbing' not in summ, 'schema': summ.get('schema')}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  M -- the SoH-floor plumbing on freshly built models
# ======================================================================================================================
UNCHANGED_ESS = ('bus', 't_cal', 'cl_nom', 'dod_nom', 'soh_min', 'cl_eff', 'phi_cal')


def _hook_spec(cell):
    """The campaign-spec-like dict the harness configuration hook reads (configuration, cap, the certificate length)."""
    cfg = configuration_now()
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
                              'ess_ageing_baseline': cfg['ess_ageing_baseline'],
                              'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
                              'ess_params_file': cfg['ess_params_file'],
                              'convergence_depth_tail': cfg['convergence_depth_tail']},
            'cap': M.spec_cap(cell), 'required_consecutive_cycles': 10}


def _side_paths(scratch, name):
    return [os.path.join(scratch, f'{name}_{k}.jsonl') for k in ('recourse_jump', 'ess_stride', 'floor', 'pf_stride')]


def built_path(G, SED, name, run_tag, scratch, mv, minimum_soh, floor_rows, ids):
    """One path, as the child runs it up to (NOT including) the first solve: (with the SoH plumbing when
    `minimum_soh` is given) G.s38_pf_capture_hooks entered with the shared floor dict, the planning object as
    `run_admm_arm` builds it, the harness's OWN configuration hook (variant apply -> [minimum_soh apply + dict update]
    -> probe readback -> floor-row identity check ~3003, raises on mismatch), the fail-fast; then probes and the
    candidate-applied ESSO models built as production builds them (NOT solved)."""
    report, holder = {}, {}
    eval_id = f'p515s53_w155_checks_{run_tag}_{name}'
    ids.append(eval_id)
    spec = _hook_spec('e_soh050')
    out = {'name': name, 'minimum_soh': minimum_soh, 'model_variant': mv}
    pcm = M.soh_floor_plumbing(minimum_soh) if minimum_soh is not None else contextlib.nullcontext()
    with pcm as pl:
        with G.s38_pf_capture_hooks(*_side_paths(scratch, name), floor_rows, stride=1):
            planning, sed, cand = G._construct_arm_planning(
                's39_D', os.path.join(scratch, name), report, investment_map=dict(M.UNIT_NODES), eval_id=eval_id,
                num_max_iters_override=M.CAP_CEILING, apply_rho=False, investment_year=M.UNIT_YEAR)
            hook = H._config_hook_factory(spec, holder, overrides={}, model_variant=mv, investment_year=M.UNIT_YEAR,
                                          expected_floor_rows=floor_rows)
            hook(planning=planning, sed=sed, candidate=cand, report=report)
            if pl is not None:
                out['fail_fast'] = pl.assert_in_force(planning)
                out['plumbing'] = pl.summary()
    probes = {n: SED._build_subproblem(sed, n) for n in sed.active_distribution_network_nodes}
    out['floor_rows_probe'], out['floor_counts_probe'] = G._identify_soh_floor_rows(probes)
    y_inv = [int(y) for y in sed.years].index(M.UNIT_YEAR)
    out['readback_probe'] = {str(n): H.model_variant_readback(m.clone(), sed, y_inv) for n, m in probes.items()}
    out['probe_node7'] = probes[M.UNIT_NODE]
    out['sed'] = sed
    models = M46V._build_esso(SED, sed, cand)
    out['fingerprint'] = {str(n): M46V._fingerprint(m) for n, m in models.items()}
    out['salvage_text'] = {str(n): str(SED._build_terminal_salvage_value_expression(
        sed, m, sed.get_shared_energy_storage_idx(n))) for n, m in models.items()}
    out['floor_idx'] = {str(n): sorted(r['constraint_idx'] for r in rows.values())
                        for n, rows in out['floor_rows_probe'].items()}
    out['ess'] = [{k: repr(getattr(e, k)) for k in UNCHANGED_ESS} for y in sed.years for e in sed.shared_energy_storages[y]]
    out['settings'] = list(SED._esso_ageing_model_settings(sed))
    out['holder'] = holder
    out['w20'] = (report.get('rule_eleven_checklist') or {}).get('w20_model_variant')
    out['w21'] = (report.get('rule_eleven_checklist') or {}).get('w21_ess_ageing_baseline')
    out['g_report_like'] = {'rule_eleven_checklist': {'w20_model_variant': out['w20']}}
    out['terminal_readback'] = (H.model_variant_readback_models(models, sed, mv, M.UNIT_YEAR, clone=True)
                                if mv is not None else None)
    out['floor_rows_shared_after'] = copy.deepcopy(floor_rows)
    del models
    return out


def _synthetic_record(path, mv):
    cfg = configuration_now()
    return {'model_variant': H.validate_model_variant(mv), 'model_variant_label': H.MODEL_VARIANT_LABEL,
            'model_variant_applied_in_child': path['holder'].get('model_variant_applied'),
            'model_variant_readback_pre_run': path['holder'].get('model_variant_readback_pre_run'),
            'model_variant_readback_terminal': path['terminal_readback'],
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_params_sha256_in_child': _sha(M.ESS_PARAMS_REL),
            'ess_ageing_verified_pre_run': path['holder'].get('ess_ageing_verified_pre_run')}


def _synthetic_sidecar(floor_rows, n_cycles=3):
    return [{'cycle': k, 'entries': [{'node_id': n, 'y_inv': str(yi), 'y': str(y), 'constraint_idx': r['constraint_idx'],
                                      'soh_min': r['soh_min'], 'active': False, 'dual': 0.0}
                                     for n, rows in floor_rows.items() for (yi, y), r in sorted(rows.items())]}
            for k in range(1, n_cycles + 1)]


def _salvage_closed_form(SED, sed, probe7, minimum_soh_values):
    """M6 (report-only): per cohort y_inv, production's own K = terminal_discount * energy_recovery_fraction *
    expected_unit_cost * remaining_life_fraction and the closed-form coefficients of e_available(y_inv, T) =
    K (1 - f) / (1 - m) and of e_rated(y_inv, T) = K [f - (1 - f) m / (1 - m)] at each m, against the linear coefficients of
    production's salvage expression built on the node-7 probe (standard repn)."""
    from pyomo.repn import generate_standard_repn
    params = sed.params.salvage_value
    years = list(sed.years)
    T = len(years) - 1
    idx = sed.get_shared_energy_storage_idx(M.UNIT_NODE)
    expr = SED._build_terminal_salvage_value_expression(sed, probe7, idx)
    repn = generate_standard_repn(expr, compute_values=True)
    coef = {id(v): c for v, c in zip(repn.linear_vars, repn.linear_coefs)}
    f = params.recycling_floor_fraction
    disc = SED._get_terminal_discount_factor(sed)
    out = {}
    for y_inv, year_inv in enumerate(years):
        ess = sed.shared_energy_storages[year_inv][idx]
        cost = SED._get_expected_energy_investment_cost(sed, year_inv)
        _age, _rem, frac = SED._get_remaining_calendar_life(sed, y_inv, ess)
        K = disc * params.energy_recovery_fraction * cost * frac
        ea, er = probe7.es_e_available_per_unit[y_inv, T], probe7.es_e_rated_per_unit[y_inv, T]
        row = {'year_inv': str(year_inv), 'ess_soh_min_in_force': ess.soh_min, 'K': K, 'remaining_life_fraction': frac,
               'recycling_floor_fraction': f,
               'repn_coef_e_available': coef.get(id(ea)), 'repn_coef_e_rated': coef.get(id(er)),
               'e_available_fixed': ea.fixed, 'e_rated_fixed': er.fixed}
        for m in minimum_soh_values:
            row[f'closed_form_at_{m}'] = {'e_available': K * (1.0 - f) / (1.0 - m),
                                          'e_rated': K * (f - (1.0 - f) * m / (1.0 - m))}
        out[str(y_inv)] = row
    return {'enabled': params.enabled, 'terminal_discount': disc, 'per_cohort_node7': out,
            'salvage_text_contains_the_constant_0_5': None}


def tests_M():
    import p515_g_g1_g4_admm_gates as G
    import shared_energy_storage_data as SED
    import p515_s40_polish_gap as PG
    import p56a_oracle as O
    scratch = tempfile.mkdtemp(prefix='p515s53_w155_checks_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    ids = []
    res = {}
    mv = M.arm_variant('e_soh050')
    try:
        _cc, base_rows, base_counts = PG._build_floor_rows(f'p515s53_w155_checks_{run_tag}_precheck')
        ids.append(f'p515s53_w155_checks_{run_tag}_precheck')
        paths = {}
        for name, variant, soh in (('new_050', mv, 0.5), ('new_070', mv, 0.7), ('variant_070_no_plumbing', mv, None),
                                   ('baseline_no_variant', None, None)):
            paths[name] = built_path(G, SED, name, run_tag, scratch, variant, soh, copy.deepcopy(base_rows), ids)
        p5, p7, pv, pb = paths['new_050'], paths['new_070'], paths['variant_070_no_plumbing'], paths['baseline_no_variant']
        # ---- M1
        rows5 = p5['floor_rows_probe']
        m1 = {
            'every_probe_floor_row_soh_min_is_0_5_exactly': all(r['soh_min'] == 0.5 for rows in rows5.values()
                                                                for r in rows.values()),
            'constraint_idx_identical_to_the_baseline': {n: sorted((k, r['constraint_idx']) for k, r in rows.items())
                                                         for n, rows in rows5.items()}
            == {n: sorted((k, r['constraint_idx']) for k, r in rows.items()) for n, rows in base_rows.items()},
            'counts_identical_to_the_baseline_6_per_node': (p5['floor_counts_probe'] == base_counts
                                                            and set(base_counts.values()) == {M.FLOOR_ROWS_PER_NODE}),
            'shared_dict_after_the_apply_equals_the_probe_rows_0_5_both_sides': p5['floor_rows_shared_after'] == rows5,
            'harness_identity_check_passed_by_equality': (p5['w20'] or {}).get('floor_rows_identical_to_baseline_probe')
            is True,
            'fail_fast_held': all((p5.get('fail_fast') or {}).values()) and bool(p5.get('fail_fast')),
            'plumbing_one_apply_one_dict_one_s38_entry_no_error': (p5['plumbing']['n_apply_calls'] == 1
                                                                   and p5['plumbing']['n_floor_dicts_captured'] == 1
                                                                   and p5['plumbing']['s38_entries'] == 1
                                                                   and not p5['plumbing']['errors']),
        }
        res['M1_floor_rows_at_0_5'] = {'holds': all(m1.values()), 'parts': m1,
                                       'floor_rows_probe_0_5': {str(n): {f'{k[0]},{k[1]}': r for k, r in rows.items()}
                                                                for n, rows in rows5.items()},
                                       'floor_rows_baseline': {str(n): {f'{k[0]},{k[1]}': r for k, r in rows.items()}
                                                               for n, rows in base_rows.items()},
                                       'counts': {str(n): c for n, c in p5['floor_counts_probe'].items()}}
        # ---- M2
        rb = p5['readback_probe']
        cfk = W.closed_form_expected(mv, 10000, 0.8)['k']
        m2 = {f'node_{n}': {'floor_row_lower_is_0_5': r['floor_row_lower'] == 0.5,
                            'k_closed_form_rtol_1e-12': math.isclose(r['k'], cfk, rel_tol=1e-12, abs_tol=0.0),
                            'k_rounds_to_35851_3609': round(r['k'], 4) == K_C2_EXPECTED_4DP,
                            'phi_0_985_rtol_1e-12': math.isclose(r['phi_cal_in_model'], PHI_EXPECTED, rel_tol=1e-12,
                                                                 abs_tol=0.0),
                            'soh_point_end': r['available_energy_soh_point'] == 'end'} for n, r in rb.items()}
        res['M2_model_variant_readback_at_0_5'] = {
            'holds': all(all(v.values()) for v in m2.values()), 'parts': m2,
            'readback': {n: {k: r[k] for k in ('floor_row_lower', 'k', 'phi_cal_in_model', 'available_energy_soh_point',
                                                'd_row_form')} for n, r in rb.items()},
            'readback_070_path_floor': {n: r['floor_row_lower'] for n, r in p7['readback_probe'].items()},
            'k_closed_form': cfk}
        # ---- M3
        m3, listed = {}, {}
        for n in pv['fingerprint']:
            a, b = pv['fingerprint'][n], p5['fingerprint'][n]
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            floor_keys = sorted(f'C:energy_storage_capacity_degradation[{i}]' for i in pv['floor_idx'][n])
            floor_diff = [k for k in diff if k in floor_keys]
            other = [k for k in diff if k not in floor_keys]
            floor_bounds = {k: {'baseline_lower': a[k][1], 'new_lower': b[k][1], 'upper_both': [a[k][2], b[k][2]]}
                            for k in floor_diff}
            m3[f'node_{n}'] = {
                'exactly_the_six_floor_rows_differ': floor_diff == floor_keys and len(floor_keys) == M.FLOOR_ROWS_PER_NODE,
                'floor_rows_differ_only_in_the_lower_bound_0_7_to_0_5': all(
                    v['baseline_lower'] == '0.7' and v['new_lower'] == '0.5' and v['upper_both'] == ['None', 'None']
                    for v in floor_bounds.values()),
                'the_only_other_differences_are_the_salvage_expression_and_its_credit': other == [
                    'E:salvage_credit[None]', 'E:salvage_value[None]'],
                'salvage_entry_is_production_salvage_text_each_side': (
                    a.get('E:salvage_value[None]') == pv['salvage_text'][n]
                    and b.get('E:salvage_value[None]') == p5['salvage_text'][n]),
                'salvage_credit_differs_only_by_the_salvage_text': (
                    pv['salvage_text'][n] in (a.get('E:salvage_credit[None]') or '')
                    and (a.get('E:salvage_credit[None]') or '').replace(pv['salvage_text'][n], p5['salvage_text'][n])
                    == b.get('E:salvage_credit[None]')),
                'salvage_text_differs_between_floors': pv['salvage_text'][n] != p5['salvage_text'][n]}
            listed[f'node_{n}'] = {'n_items': len(a), 'differing_keys': diff, 'floor_row_bounds': floor_bounds,
                                   'salvage_text_baseline': pv['salvage_text'][n][:600],
                                   'salvage_text_0_5': p5['salvage_text'][n][:600],
                                   'salvage_credit_text_baseline': (a.get('E:salvage_credit[None]') or '')[:600],
                                   'salvage_credit_text_0_5': (b.get('E:salvage_credit[None]') or '')[:600]}
        res['M3_fingerprint_diff_050_vs_baseline'] = {'holds': all(all(v.values()) for v in m3.values()), 'parts': m3,
                                                      'listed': listed,
                                                      'baseline_path': 'variant_070_no_plumbing (the e_c2_calfade path)'}
        # ---- M4
        m4 = {}
        for ref_name, ref in (('variant_070_no_plumbing', pv), ('baseline_no_variant', pb)):
            diffs = {n: sorted(k for k in set(ref['fingerprint'][n]) | set(p7['fingerprint'][n])
                               if ref['fingerprint'][n].get(k) != p7['fingerprint'][n].get(k)) for n in ref['fingerprint']}
            m4[f'new_070_fingerprint_identical_to_{ref_name}'] = all(not d for d in diffs.values())
            m4[f'new_070_ess_constants_repr_identical_to_{ref_name}'] = p7['ess'] == ref['ess']
            m4[f'new_070_floor_rows_equal_{ref_name}'] = p7['floor_rows_probe'] == ref['floor_rows_probe']
        m4['new_070_settings_end_on'] = p7['settings'] == pv['settings'] == ['end', True]
        m4['new_070_shared_dict_after_equals_the_baseline_precheck'] = p7['floor_rows_shared_after'] == base_rows
        m4['new_070_identity_check_passed'] = (p7['w20'] or {}).get('floor_rows_identical_to_baseline_probe') is True
        m4['new_070_plumbing_ran_once'] = p7['plumbing']['n_apply_calls'] == 1 and not p7['plumbing']['errors']
        m4['baseline_no_variant_readback_all_match'] = (pb['w21'] or {}).get('readback_all_match') is True
        res['M4_identity_control_070'] = {'holds': all(m4.values()), 'parts': m4,
                                          'digests': {nm: {n: M46V._fp_digest(fp) for n, fp in p['fingerprint'].items()}
                                                      for nm, p in paths.items()}}
        # ---- M5
        d050, d070 = M.declaration_for('e_soh050'), M.declaration_for('g070_neutrality')
        r050, r070 = _synthetic_record(p5, mv), _synthetic_record(p7, mv)
        g = {'readback_050_record_050_declaration_holds': M.soh_readback_gate(r050, d050)[0],
             'readback_070_record_070_declaration_holds': M.soh_readback_gate(r070, d070)[0]}
        bad = copy.deepcopy(r050)
        bad['model_variant_readback_terminal']['per_node']['7']['readback']['floor_row_lower'] = 0.7
        g['readback_refuses_floor_0_7_on_a_0_5_declaration'] = not M.soh_readback_gate(bad, d050)[0]
        g['readback_refuses_the_070_record_on_a_0_5_declaration'] = not M.soh_readback_gate(r070, d050)[0]
        bad = copy.deepcopy(r070)
        bad['model_variant_readback_pre_run']['per_node']['5']['readback']['floor_row_lower'] = 0.5
        g['readback_refuses_floor_0_5_on_a_0_7_declaration'] = not M.soh_readback_gate(bad, d070)[0]
        g['readback_refuses_the_050_record_on_a_0_7_declaration'] = not M.soh_readback_gate(r050, d070)[0]
        bad = copy.deepcopy(r050)
        bad['model_variant_applied_in_child'].pop('minimum_soh_applied')
        g['readback_refuses_a_record_without_the_minimum_soh_apply'] = not M.soh_readback_gate(bad, d050)[0]
        side = _synthetic_sidecar(p5['floor_rows_shared_after'])
        summ_ok = {'soh_floor_plumbing_ok': True, 'minimum_soh_declared': 0.5}
        g['sidecar_gate_holds_on_0_5_lines'] = M.soh_sidecar_gate(side, p5['g_report_like'], d050, 3, summ_ok)[0]
        bad = copy.deepcopy(side)
        bad[1]['entries'][4]['soh_min'] = 0.7
        g['sidecar_gate_refuses_one_0_7_entry'] = not M.soh_sidecar_gate(bad, p5['g_report_like'], d050, 3, summ_ok)[0]
        badr = copy.deepcopy(p5['g_report_like'])
        badr['rule_eleven_checklist']['w20_model_variant']['floor_rows_identical_to_baseline_probe'] = False
        g['sidecar_gate_refuses_identity_flag_false'] = not M.soh_sidecar_gate(side, badr, d050, 3, summ_ok)[0]
        g['sidecar_gate_refuses_0_7_lines_on_a_0_5_declaration'] = not M.soh_sidecar_gate(
            _synthetic_sidecar(base_rows), p5['g_report_like'], d050, 3, summ_ok)[0]
        # the plumbing's own refusals (zero solves; a fresh planning object)
        neg_id = f'p515s53_w155_checks_{run_tag}_negative'
        ids.append(neg_id)
        planning_n, sed_n, _cand_n = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'negative'), {}, investment_map=dict(M.UNIT_NODES), eval_id=neg_id,
            num_max_iters_override=M.CAP_CEILING, apply_rho=False, investment_year=M.UNIT_YEAR)
        originals = (G.esso_capture_hooks, G.s38_pf_capture_hooks, H.apply_model_variant)
        with M.soh_floor_plumbing(0.5):
            try:
                G.esso_capture_hooks(planning_n, {})
                g['fail_fast_refuses_when_the_apply_did_not_run'] = False
            except RuntimeError as error:
                g['fail_fast_refuses_when_the_apply_did_not_run'] = 'NOT in force before the first solve' in str(error)
            try:
                H.apply_model_variant(sed_n, mv)
                g['apply_refuses_without_a_captured_floor_dict'] = False
            except RuntimeError as error:
                g['apply_refuses_without_a_captured_floor_dict'] = 'was not captured before the apply' in str(error)
        g['every_patched_attribute_restored'] = (
            (G.esso_capture_hooks, G.s38_pf_capture_hooks, H.apply_model_variant) == originals)
        res['M5_negative_controls'] = {'holds': all(g.values()), 'parts': g,
                                       'gate_050_failing_parts': sorted(k for k, v in M.soh_readback_gate(
                                           r050, d050)[1]['parts'].items() if v is not True)}
        # ---- M6 (report-only)
        try:
            sal = _salvage_closed_form(SED, p5['sed'], p5['probe_node7'], (0.5, 0.7))
            for row in sal['per_cohort_node7'].values():
                cf = row['closed_form_at_0.5']
                ca = row['repn_coef_e_available'] if row['repn_coef_e_available'] is not None else 0.0
                cr = row['repn_coef_e_rated'] if row['repn_coef_e_rated'] is not None else 0.0
                row['repn_coef_absent_read_as_0'] = (row['repn_coef_e_available'] is None
                                                     or row['repn_coef_e_rated'] is None)
                row['repn_matches_closed_form_at_0_5'] = (
                    math.isclose(ca, cf['e_available'], rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(cr, cf['e_rated'], rel_tol=1e-12, abs_tol=1e-12))
                row['closed_forms_at_0_5_and_0_7_differ'] = row['closed_form_at_0.5'] != row['closed_form_at_0.7']
            sal['salvage_text_contains_the_constant_0_5'] = '0.5' in p5['salvage_text'][str(M.UNIT_NODE)]
            res['M6_salvage_closed_form_REPORT_ONLY'] = {'holds': True, 'report_only': True, **sal}
        except Exception as error:  # noqa: BLE001 -- report-only
            res['M6_salvage_closed_form_REPORT_ONLY'] = {'holds': True, 'report_only': True,
                                                         'error': f'{type(error).__name__}: {error}'}
        for p in paths.values():
            for k in ('probe_node7', 'sed'):
                p.pop(k, None)
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
    res['working_dirs_with_files_left'] = left
    gating = [k for k in res if k.startswith('M') and 'REPORT_ONLY' not in k]
    return {'holds': bool(all(res[k].get('holds') is True for k in gating) and not left), 'tests': res,
            'note': ('zero solves: models are BUILT (production _build_subproblem through the harness\'s own '
                     'configuration hook and update_model_with_candidate_solution), never solved; scratch outside the '
                     'repository, removed; pickle.load / loads blocked')}


# ======================================================================================================================
#  P -- preconditions, entry checks, validator
# ======================================================================================================================
NEGATIVE_EXPECTED_CHECK = {
    'cap_wrong': 'd_spec_cap_equals_the_cell_cap_within_its_ceiling',
    'tail_off': 'tail_enabled_for_this_run', 'aa_off': 'a_anderson_acceleration_on',
    'spec_minimum_soh_overridden_to_0_5': 'w155_spec_minimum_soh_is_the_file_070_not_overridden',
    'flex_multiplier_on_a_soh_cell': 'w155_flex_multiplier_forbidden_on_a_soh_cell',
    'soh_cell_without_the_variant': 'w155_entry_model_variant_is_the_declared_arm',
    'minimum_soh_anywhere_in_an_h_entry': 'w155_minimum_soh_forbidden_on_an_h_cell',
    'h_cell_without_the_multiplier': 'w155_flex_multiplier_on_the_h_cell_is_1_75',
    'h_cell_multiplier_2': 'w155_flex_multiplier_on_the_h_cell_is_1_75',
    'h_cell_with_a_model_variant': 'w155_h_cell_has_no_model_variant',
}


def tests_P():
    out = {}
    ok = True
    for cell in M.CELL_ORDER:
        c = M.CELLS[cell]
        decl = M.declaration_for(cell)
        good_spec = spec_like(cell)
        tl = {'tail_enabled_for_this_run': True}
        try:
            good = M.assert_resettle_preconditions(decl, good_spec, tl, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        entry_ok = all(M.spec_entry_checks(decl, good_spec).values())
        negatives = {
            'cap_wrong': (spec_like(cell, cap=M.spec_cap(cell) - 1), tl, True),
            'tail_off': (good_spec, {'tail_enabled_for_this_run': False}, True),
            'aa_off': (good_spec, tl, False),
            'spec_minimum_soh_overridden_to_0_5': (spec_like(cell, config_overrides={'ess_ageing_baseline': dict(
                configuration_now()['ess_ageing_baseline'], minimum_soh=0.5)}), tl, True),
        }
        if c['kind'] == 'soh':
            negatives['flex_multiplier_on_a_soh_cell'] = (spec_like(cell, entry_overrides={
                'flex_price_multiplier': 1.75, 'flex_price_label': H.FLEX_PRICE_LABEL},
                spec_overrides={'flex_price_label': H.FLEX_PRICE_LABEL}), tl, True)
            negatives['soh_cell_without_the_variant'] = (spec_like(cell, entry_overrides={'model_variant': None}), tl,
                                                         True)
        else:
            negatives['minimum_soh_anywhere_in_an_h_entry'] = (spec_like(cell, entry_overrides={'minimum_soh': 0.5}),
                                                               tl, True)
            negatives['h_cell_without_the_multiplier'] = (spec_like(cell, entry_overrides={
                'flex_price_multiplier': None}), tl, True)
            negatives['h_cell_multiplier_2'] = (spec_like(cell, entry_overrides={'flex_price_multiplier': 2.0}), tl,
                                                True)
            negatives['h_cell_with_a_model_variant'] = (spec_like(cell, entry_overrides={
                'model_variant': W.ARMS['C2_calfade'], 'model_variant_label': H.MODEL_VARIANT_LABEL}), tl, True)
        neg, neg_named = {}, {}
        for name, (spec, tlx, aa) in negatives.items():
            try:
                M.assert_resettle_preconditions(decl, spec, tlx, aa)
                neg[name] = 'NOT refused'
            except RuntimeError as error:
                neg[name] = f'refused: {str(error)[:1500]}'
            neg_named[name] = f"'{NEGATIVE_EXPECTED_CHECK[name]}'" in neg[name]
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1), 'unknown_cell': dict(decl, cell='x0'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'ext_v6_schema': dict(decl, schema=V6.EXT_DECLARATION_SCHEMA),
               'rule_v5': dict(decl, settling_rule=K142.V5.settling_rule_declaration()),
               'cap_rule_changed': dict(decl, cap_rule=dict(decl['cap_rule'], ceiling=999)),
               'replay_reference_added': dict(decl, replay_reference={'per_cycle_record': 'x'})}
        if c['kind'] == 'soh':
            bad['minimum_soh_other_value'] = dict(decl, minimum_soh=0.6)
            bad['minimum_soh_dropped'] = {k: v for k, v in decl.items() if k != 'minimum_soh'}
            bad['flex_on_a_soh_cell'] = dict(decl, flex_price_multiplier_expected=1.75)
            bad['floor_expected_070_on_0_5'] = dict(decl, floor_row_lower_expected=0.7 if decl['minimum_soh'] == 0.5
                                                    else 0.5)
        else:
            bad['minimum_soh_on_an_h_cell'] = dict(decl, minimum_soh=0.5)
            bad['flex_expected_2'] = dict(decl, flex_price_multiplier_expected=2.0)
        refused = {}
        for name, d in bad.items():
            try:
                M.validate_settling_resettle(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:200]
        cell_ok = (good_ok and entry_ok and all(v.startswith('refused') for v in neg.values())
                   and all(neg_named.values()) and all(refused.values()))
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'n_checklist_items': len(good) if isinstance(good, dict) else None,
                     'checklist_failing': (sorted(k for k, v in good.items() if v is not True) if not good_ok
                                           and 'error' not in good else good.get('error')),
                     'entry_checks': M.spec_entry_checks(decl, good_spec), 'negative_controls': neg,
                     'negative_controls_refused_by_the_named_check': neg_named,
                     'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w155():
    return K137._harness_at(PRE_W155_HARNESS['commit'], PRE_W155_HARNESS['sha256'], '_w155_pre_harness')


def resettle_keys(pre=None):
    out = {}
    for cell in M.CELL_ORDER:
        c = M.CELLS[cell]
        kw = key_kwargs(cell)
        base = H.evaluation_key(c['candidate_key'], {}, **kw)
        decl = M.declaration_for(cell)
        key = H.evaluation_key(c['candidate_key'], {}, settling_resettle=decl, **kw)
        base_pre = pre.evaluation_key(c['candidate_key'], {}, **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': c['candidate_key'], 'base_key_now': base, 'resettle_key': key,
                     'formula_holds': key == formula, 'base_equals_pre_w155': base == base_pre}
    return out


def tests_K():
    pre, pre_sha, _src = _harness_pre_w155()
    n_specs = n_entries = n_equal = 0
    mismatch, errors, scanned = [], [], []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            try:
                args, kw = K132._key_args(spec, e)
                rs = e.get('settling_resettle')
                if M.is_w155_declaration(rs):
                    continue
                new = H.evaluation_key(*args, settling_resettle=rs, **kw)
                old = pre.evaluation_key(*args, settling_resettle=rs, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    ext = json.load(open(_abs(EXT_V6_STAGE_SPEC['path'])))
    v6 = json.load(open(_abs(V6_STAGE_SPEC['path'])))
    ext_keys = {c: json.load(open(_abs(p['path'])))['candidates'][0]['eval_key']
                for c, p in ext['pins']['campaign_specs'].items()}
    v6_h_keys = {c: json.load(open(_abs(p['path'])))['candidates'][0]['eval_key']
                 for c, p in v6['pins']['campaign_specs'].items() if c.startswith('h_')}
    holders = {cell: K137._key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: K137._outside_root(h, ROOT_REL) for cell, h in holders.items()}
    probe = keys[M.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w155_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    ctl = {}
    for name, rel in (('outside', os.path.join(_P53, 'w155_planted_negative_control', 'campaign_planted',
                                               'campaign_spec_planted_00000000.json')),
                      ('sibling_prefix', os.path.join(ROOT_REL + '_planted_sibling', 'campaign_planted',
                                                      'campaign_spec_planted_00000000.json')),
                      ('ext_v6_root', os.path.join(_P53, 'w142_resettle_ext_v6', 'campaign_planted',
                                                   'campaign_spec_planted_00000000.json'))):
        ctl[name] = {'rel': rel, 'refused': rel in K137._outside_root(K137._key_holders(scanned + [planted(rel)], probe),
                                                                      ROOT_REL)}
    own_rel = os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{M.CELL_ORDER[0]}',
                           f'campaign_spec_{CAMPAIGN_ID_PREFIX}{M.CELL_ORDER[0]}_00000000.json')
    own_holders = K137._key_holders(scanned + [planted(own_rel)], probe)
    try:
        pre.validate_settling_resettle(M.declaration_for('e_soh050'))
        pre_refuses = False
    except ValueError:
        pre_refuses = True
    all_keys = {v['resettle_key'] for v in keys.values()}
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_entry_equal': n_equal == n_entries and n_entries > 0,
        'w155_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'w155_base_equals_pre_w155_every_cell': all(v['base_equals_pre_w155'] for v in keys.values()),
        'w155_keys_distinct': len(all_keys) == len(keys) == 4,
        'w155_keys_differ_from_every_extension_v6_key': not (all_keys & set(ext_keys.values())),
        'w155_keys_differ_from_every_v6_h_key': not (all_keys & set(v6_h_keys.values())),
        'soh_keys_differ_from_each_other_and_from_e_c2_calfade': len({keys['g070_neutrality']['resettle_key'],
                                                                       keys['e_soh050']['resettle_key'],
                                                                       ext_keys['e_c2_calfade']}) == 3,
        'h_keys_carry_the_multiplier_in_the_base_key': all(
            keys[c]['base_key_now'] != H.evaluation_key(M.CELLS[c]['candidate_key'], {},
                                                        **dict(key_kwargs(c), flex_price_multiplier=None))
            for c in M.FLEX_CELLS),
        'w155_keys_absent_from_committed_specs_outside_the_own_root': not any(outside.values()),
        **{f'control_planted_{k}_refused': v['refused'] for k, v in ctl.items()},
        'control_own_campaign_spec_accepted': own_rel in own_holders and own_rel not in K137._outside_root(
            own_holders, ROOT_REL),
        'control_w155_declaration_refused_by_the_pre_w155_harness': pre_refuses,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w155_harness': {**PRE_W155_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'extension_v6_keys': ext_keys, 'v6_h_keys': v6_h_keys, 'w155_keys_in_committed_specs_all_REPORTED': holders,
            'controls': ctl, 'control_own': {'rel': own_rel, 'holders': own_holders}}


# ======================================================================================================================
#  D -- the router
# ======================================================================================================================
def harness_router_check():
    import difflib
    import inspect
    _pre, _sha_pre, pre_src = _harness_pre_w155()
    now_src = open(_abs('p515_s44_campaign_harness.py')).read()
    diff = [ln for ln in difflib.unified_diff(pre_src.splitlines(), now_src.splitlines(), lineterm='', n=0)
            if not ln.startswith(('---', '+++', '@@'))]
    added = [ln[1:] for ln in diff if ln.startswith('+')]
    removed = [ln[1:] for ln in diff if ln.startswith('-')]
    fn_src = inspect.getsource(H.resettle_hooks_module)
    i142 = fn_src.find('W142C.is_v6_declaration(value)')
    i155 = fn_src.find('W155C.is_w155_declaration(value)')
    i118 = fn_src.find('import p515_s53_w118_resettle_hooks as W118C')
    sha_now = _sha('p515_s44_campaign_harness.py')
    last = H._git(['log', '--format=%H %s', '-1', '--', 'p515_s44_campaign_harness.py'])
    dispatch = dict(K142.dispatch_checks())
    dispatch['w155_to_w155'] = all(H.resettle_hooks_module(M.declaration_for(c)) is M for c in M.CELL_ORDER)
    dispatch['harness_validates_w155'] = all(H.validate_settling_resettle(M.declaration_for(c)) == M.declaration_for(c)
                                             for c in M.CELL_ORDER)
    parts = {
        'no_line_removed': removed == [],
        'exactly_three_lines_added': len(added) == 3,
        'the_three_added_are_the_w155_branch': tuple(added) == W155_BRANCH_LINES,
        'order_v6_then_w155_then_w118': 0 <= i142 < i155 < i118,
        'harness_committed_clean': _clean('p515_s44_campaign_harness.py'),
        'harness_sha256_is_the_pinned_post_w155_sha256': sha_now == HARNESS_POST_W155_SHA256,
        'last_harness_commit_message_carries_the_sha': HARNESS_POST_W155_SHA256 in last,
        'dispatch_every_family': all(dispatch.values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'added': added, 'removed': removed, 'dispatch': dispatch,
            'harness_sha256_now': sha_now, 'pre_w155_harness': PRE_W155_HARNESS,
            'post_w155_sha256_pinned': HARNESS_POST_W155_SHA256, 'last_harness_commit': last}


def tests_D():
    return harness_router_check()


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('M', tests_M), ('P', tests_P),
            ('K', tests_K), ('D', tests_D))

CODE_PINNED_BY_CHECKS = tuple(dict.fromkeys(
    (os.path.basename(__file__), 'p515_s53_w155_a64_hooks.py', 'p515_s53_w142_resettle_ext_v6_hooks.py',
     'p515_s53_w142_resettle_v6_hooks.py', 'p515_s53_w142_resettle_v6_checks.py',
     'p515_s53_w142_resettle_ext_v6_checks.py', 'settling_criterion_v6.py', 'p515_s46_variant_checks.py',
     'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py', 'p515_s40_polish_gap.py')
    + tuple(KX.CODE_PINNED_BY_CHECKS)))


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
        print(f"[W155-CHECKS] {sid}: holds={r_ok} wall={out[sid]['wall_s']:.1f}s"
              + (f" error={r.get('error')}" if isinstance(r, dict) and r.get('error') else ''), flush=True)
    return {'all_hold': ok, 'p_max': V6.P_MAX, 'constants': SC6.constants(V6.P_MAX), 'sections': out}


def failing_items(res):
    out = {}
    for sid, s in res['sections'].items():
        if s['holds']:
            continue
        r = s['result']
        out[sid] = (r.get('error') or sorted(k for k, v in (r.get('parts') or {}).items() if v is not True)
                    or sorted(k for k, v in (r.get('tests') or r.get('cells') or {}).items()
                              if isinstance(v, dict) and v.get('holds', v.get('ok')) is not True))
    return out


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


def guards_report():
    return {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}


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
    guards = guards_report()
    pickle_ok = PICKLE_COUNTS == {'load': 0, 'loads': 0}
    doc = {'schema': 'p515_s53_w155_zero_solve_checks_v1',
           'task': 'W155 (PLANNER_BRIEF_2026-09-13.md Addendum 64)',
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'code_sha256': code_pins, 'guards': guards, 'pickle_guard': dict(PICKLE_COUNTS),
           'W_bool_typing_test': typing, **res, 'failing_items': failing_items(res),
           'all_hold_including_typing_test': bool(res['all_hold'] and typing['pass'] and pickle_ok)}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    tpath = os.path.join(out_dir, TYPING_OUT)
    if os.path.isfile(tpath):
        manifest[os.path.relpath(tpath, REPO)] = H.sha256_file(tpath)
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    print(f"[W155-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W155-CHECKS] failing items: {doc['failing_items']}")
    print(f"[W155-CHECKS] all_hold={res['all_hold']} (with typing and pickle {doc['all_hold_including_typing_test']}) "
          f"pickle={PICKLE_COUNTS} guards={guards}")
    print(f"[W155-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
