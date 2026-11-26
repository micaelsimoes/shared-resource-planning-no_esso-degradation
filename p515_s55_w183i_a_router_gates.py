"""
P5.15 Addendum 86 ruling 1, Planner task W183i (a) -- ZERO-SOLVE gates (i)-(iii) of the one-branch router patch in
`p515_s44_campaign_harness.resettle_hooks_module` that routes the Step-7 declaration schema
(`p515_s55_step7_resettle_hooks.SCHEMA`) to the new hooks module `p515_s55_step7_resettle_hooks.py`.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 1 and W183i order item (a); Planner task W183i (a).

PRE-PATCH HARNESS. The harness blob at PRE_PATCH_COMMIT (dd9e46d1, the HEAD before W183i), written to a scratch copy
OUTSIDE the repo and loaded under its own module name; its sha256 must equal PRE_PATCH_HARNESS_SHA256 (c34ebdc5..., the
harness Addendum 85/86 name). The patched working-tree harness must differ from it by exactly the three added router
lines (difflib, recorded).

GATES
  (i)   Every committed eval key recomputes with the patched harness: the W183a T1 method, BY IMPORT
        (`p515_s53_w183a_option_tests.t1_old_keys`; that script is NOT run as a whole and nothing is written into its
        output dir) over every git-tracked `campaign_spec_*.json` under data/SRP1/Results with schema
        'p515_s44_campaign_spec_v2' -- each entry's key from its stored inputs equals the stored `eval_key` and the
        pre-patch harness's key. Expected 7,213 entries of 252 specs; skips listed with reasons.
  (ii)  Routing identity: every `settling_resettle` declaration (non-None) found (A) in the entries of every
        git-tracked `campaign_spec_*.json` under data/SRP1/Results and (B, widened scope) anywhere in every git-tracked
        `*.json` under data/SRP1/Results (any dict value under the key 'settling_resettle') is routed by the patched
        `resettle_hooks_module` to the SAME module object as by the pre-patch one; and the patched
        `validate_settling_resettle` gives the same outcome (equal canonical output, or the same exception text) as
        the pre-patch one. Listed per schema -> module with counts.
  (iii) Neutrality: no declaration of (A)/(B) is a Step-7 declaration nor is routed to the new module; the schema
        string occurs in no git-tracked file other than the new module and the W183i scripts. Planted (scratch
        freeze OUTSIDE the repo through the production `freeze_campaign_spec`, then read in memory) one spec per cell
        for each of the 9 cells: routed to the new module; validates (patched) and is refused by the pre-patch
        harness (the branch is needed); `spec_entry_checks` all True; the full capture checklist
        `assert_resettle_preconditions` passes; the eval key recomputes; the state constructs. Refusals: a renamed
        cell, a committed ext-v6 cell name under the new schema, the declaration under the ext-v6 schema, a mutated
        expected option, an int where a float is declared; an entry missing an expected option, missing its label at
        entry or spec level, carrying an undeclared option, or carrying a model variant -- each refused, with the
        named failing check.

GUARDS: `SolveProfileGuard(permitted=())` installed BEFORE any project import and verified at exactly 0 (and the
imported W183a module's own guard, installed at its import, also 0); `pickle.load` / `pickle.loads` blocked and counted
(0); pyomo `ConcreteModel.__init__` blocked and counted (0). JSON through `gate_result_io`; sha256 manifests.

OUTPUT (write-once; the directory may pre-exist holding only launch.log):
  data/SRP1/Results/P515S55/w183i_a_router_gates/{w183i_a_router_gates.json, manifest_sha256.json}
  `--post-run` adds manifest_post_run_sha256.json over the directory incl. launch.log.
EXACT COMMANDS (repo root; attached, alone, both streams captured; this script, the hooks module and the patched
harness committed first):
  mkdir -p data/SRP1/Results/P515S55/w183i_a_router_gates && set -o noclobber && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183i_a_router_gates.py \\
      --scratch <scratchpad>/w183i_a_gates > data/SRP1/Results/P515S55/w183i_a_router_gates/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183i_a_router_gates.py --post-run
Exit 0 iff every gate passes and every guard reads 0.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183i (a) router gates (zero solves)').install()

import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183i (a): pickle.load called -- nothing is unpickled')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183i (a): pickle.loads called -- nothing is unpickled')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError('W183i (a): a pyomo ConcreteModel was constructed -- no model build')


pe.ConcreteModel.__init__ = _blocked_concrete_init

import argparse  # noqa: E402
import copy  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w183a_option_tests as W183A  # noqa: E402 -- T1 code by import; its main() is never called
import p515_s53_w142_resettle_ext_v6_hooks as EXT6  # noqa: E402
import p515_s53_w155_a64_hooks as W155  # noqa: E402
import p515_s55_step7_resettle_hooks as S7  # noqa: E402

pickle.load, pickle.loads = _blocked_load, _blocked_loads   # re-assert after the imports (W183a installs its own)
os.chdir(REPO)

SCRIPT = os.path.basename(__file__)
SCHEMA = 'p515_s55_w183i_a_router_gates_v1'
STAGE = 'P5.15 Addendum 86 W183i (a) -- Step-7 hooks module router patch: gates (i)-(iii) (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 1; W183i order item (a)', 'Planner task W183i (a)']
HARNESS_REL = 'p515_s44_campaign_harness.py'
HOOKS_REL = 'p515_s55_step7_resettle_hooks.py'
WRAPPER_REL = 'p515_s55_w183i_bitwise_gate.py'
PRE_PATCH_COMMIT = 'dd9e46d1'
PRE_PATCH_HARNESS_SHA256 = 'c34ebdc5463a1fe0fd6398c6acad4dcfb61dbfbd142ef3de517ca517168d77be'
EXPECTED_ADDED_LINES = ['    import p515_s55_step7_resettle_hooks as S7C',
                        '    if S7C.is_step7_declaration(value):',
                        '        return S7C']
EXPECTED_T1 = {'n_v2_specs': 252, 'n_entries': 7213}
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183i_a_router_gates')
RESULTS_REL = os.path.join('data', 'SRP1', 'Results')
TEMPLATE_SPEC_REL = os.path.join(
    RESULTS_REL, 'P515S53', 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_e_no_ageing',
    'campaign_spec_s53_w142_resettle_ext_v6_e_no_ageing_10f4aa7a.json')
TEMPLATE_SPEC_SHA256_PREFIX = '10f4aa7a'
LOCK_NAME = '.w183i_a_router_gates.lock'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183i-a] {msg}', flush=True)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, binary=False):
    out = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _outcome(fn, *args, **kwargs):
    try:
        return {'ok': True, 'value': fn(*args, **kwargs)}
    except Exception as error:  # noqa: BLE001
        return {'ok': False, 'error': f'{type(error).__name__}: {error}'}


# ======================================================================================================================
#  the pre-patch harness
# ======================================================================================================================
def load_pre_patch_harness(scratch):
    blob = _git(['show', f'{PRE_PATCH_COMMIT}:{HARNESS_REL}'], binary=True)
    sha = hashlib.sha256(blob).hexdigest()
    if sha != PRE_PATCH_HARNESS_SHA256:
        raise RuntimeError(f'pre-patch harness blob at {PRE_PATCH_COMMIT} sha256 {sha} != {PRE_PATCH_HARNESS_SHA256}')
    path = os.path.join(scratch, 'pre_patch_harness', 'p515_s44_campaign_harness_pre_w183i.py')
    os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as handle:
        handle.write(blob)
    spec = importlib.util.spec_from_file_location('p515_s44_campaign_harness_pre_w183i', path)
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    os.chdir(REPO)
    with open(os.path.join(REPO, HARNESS_REL), 'rb') as handle:
        patched = handle.read()
    diff = list(difflib.unified_diff(blob.decode().splitlines(), patched.decode().splitlines(), lineterm='', n=0))
    added = [d[1:] for d in diff if d.startswith('+') and not d.startswith('+++')]
    removed = [d[1:] for d in diff if d.startswith('-') and not d.startswith('---')]
    hunks = [d for d in diff if d.startswith('@@')]
    info = {'commit': _git(['rev-parse', PRE_PATCH_COMMIT]).strip(), 'blob_sha256': sha,
            'path_in_scratch': path, 'patched_sha256': hashlib.sha256(patched).hexdigest(),
            'diff_hunks': hunks, 'diff_added_lines': added, 'diff_removed_lines': removed,
            'diff_is_exactly_the_three_router_lines': (added == EXPECTED_ADDED_LINES and not removed
                                                       and len(hunks) == 1)}
    return module, info


# ======================================================================================================================
#  declarations in committed artefacts
# ======================================================================================================================
def _walk_declarations(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}.{k}'
            if k == 'settling_resettle' and isinstance(v, dict):
                out.append((p, v))
            _walk_declarations(v, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_declarations(v, f'{path}[{i}]', out)


def collect_declarations():
    tracked = [f for f in _git(['ls-files', RESULTS_REL]).splitlines() if f.endswith('.json')]
    specs = [f for f in tracked if os.path.basename(f).startswith('campaign_spec_')]
    a, b, b_files, unreadable = [], [], 0, []
    for rel in sorted(specs):
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            unreadable.append({'file': rel, 'reason': 'tracked but missing from the working tree'})
            continue
        with open(path) as handle:
            spec = json.load(handle)
        for i, entry in enumerate(spec.get('candidates') or []):
            decl = entry.get('settling_resettle') if isinstance(entry, dict) else None
            if decl is not None:
                a.append({'file': rel, 'where': f'candidates[{i}] ({entry.get("label")})', 'decl': decl})
    for rel in sorted(tracked):
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as handle:
            raw = handle.read()
        if b'"settling_resettle"' not in raw:
            continue
        b_files += 1
        try:
            obj = json.loads(raw)
        except Exception as error:  # noqa: BLE001
            unreadable.append({'file': rel, 'reason': f'{type(error).__name__}: {error}'})
            continue
        found = []
        _walk_declarations(obj, '', found)
        for where, decl in found:
            b.append({'file': rel, 'where': where, 'decl': decl})
    return {'n_tracked_json': len(tracked), 'n_tracked_campaign_specs': len(specs), 'A': a, 'B': b,
            'n_files_B_mentioning_settling_resettle': b_files, 'unreadable': unreadable}


def gate_ii_iii_committed(H_pre, decls):
    by_schema, mismatches, validate_mismatches, step7_hits = {}, [], [], []
    for scope in ('A', 'B'):
        for item in decls[scope]:
            decl = item['decl']
            schema = decl.get('schema') or '<no schema key: W118 declaration>'
            m_new = H.resettle_hooks_module(decl)
            m_pre = H_pre.resettle_hooks_module(decl)
            same = m_new is m_pre
            row = by_schema.setdefault(scope, {}).setdefault(schema, {'n': 0, 'modules_patched': set(),
                                                                      'modules_pre_patch': set(), 'all_same': True,
                                                                      'validate_same': True})
            row['n'] += 1
            row['modules_patched'].add(m_new.__name__)
            row['modules_pre_patch'].add(m_pre.__name__)
            if not same:
                row['all_same'] = False
                mismatches.append({'scope': scope, 'file': item['file'], 'where': item['where'],
                                   'patched': m_new.__name__, 'pre': m_pre.__name__})
            if S7.is_step7_declaration(decl) or m_new is S7:
                step7_hits.append({'scope': scope, 'file': item['file'], 'where': item['where']})
            v_new = _outcome(H.validate_settling_resettle, decl)
            v_pre = _outcome(H_pre.validate_settling_resettle, decl)
            v_same = (v_new['ok'] == v_pre['ok']
                      and (_canon(v_new.get('value')) == _canon(v_pre.get('value')) if v_new['ok']
                           else v_new['error'] == v_pre['error']))
            if not v_same:
                row['validate_same'] = False
                validate_mismatches.append({'scope': scope, 'file': item['file'], 'where': item['where'],
                                            'patched': v_new.get('error', 'ok'), 'pre': v_pre.get('error', 'ok')})
            row.setdefault('n_validate_ok', 0)
            row.setdefault('n_validate_refused', 0)
            row['n_validate_ok' if v_new['ok'] else 'n_validate_refused'] += 1
    for scope_rows in by_schema.values():
        for row in scope_rows.values():
            row['modules_patched'] = sorted(row['modules_patched'])
            row['modules_pre_patch'] = sorted(row['modules_pre_patch'])
    n_a, n_b = len(decls['A']), len(decls['B'])
    ii = {'n_declarations_A_campaign_spec_entries': n_a, 'n_declarations_B_any_tracked_json': n_b,
          'routing_by_scope_and_schema': by_schema, 'routing_mismatches': mismatches,
          'validate_outcome_mismatches': validate_mismatches,
          'pass': n_a > 0 and n_b >= n_a and not mismatches and not validate_mismatches}
    proc = subprocess.run(['git', 'grep', '-l', '-F', S7.SCHEMA], cwd=REPO, capture_output=True, text=True)
    if proc.returncode not in (0, 1):   # 1 = no match
        raise RuntimeError(f'git grep failed: {proc.returncode} {proc.stderr}')
    grep = proc.stdout.split()
    allowed = {HOOKS_REL, SCRIPT, WRAPPER_REL}
    neutral = {'n_committed_declarations_routed_to_the_new_module_or_carrying_its_schema': len(step7_hits),
               'hits': step7_hits, 'tracked_files_containing_the_schema_string': grep,
               'only_the_new_module_and_w183i_scripts_contain_it': set(grep) <= allowed,
               'pass': not step7_hits and set(grep) <= allowed}
    return ii, neutral


# ======================================================================================================================
#  (iii) planted entries
# ======================================================================================================================
def _template_configuration():
    path = os.path.join(REPO, TEMPLATE_SPEC_REL)
    sha = _sha_file(path)
    if not sha.startswith(TEMPLATE_SPEC_SHA256_PREFIX):
        raise RuntimeError(f'{TEMPLATE_SPEC_REL} sha256 {sha} does not carry its name prefix')
    with open(path) as handle:
        t = json.load(handle)
    c = t['configuration']
    conf = {'name': 'W183i (a) planted Step-7 entry (scratch freeze outside the repo; never launched)',
            'arm_label': c['arm_label'], 'overrides': c['overrides'],
            'case_file_anderson_acceleration': c['case_file_anderson_acceleration'],
            'ess_ageing_baseline': c['ess_ageing_baseline'],
            'ess_ageing_baseline_label': c['ess_ageing_baseline_label'],
            'convergence_depth_tail': c['convergence_depth_tail'],
            'note': 'W183i (a) gate (iii) planted entry'}
    return conf, {'path': TEMPLATE_SPEC_REL, 'sha256': sha, 'cap': t['cap'],
                  'post_certification_of_template_entry': t['candidates'][0]['post_certification']}


def _cell_options(cell):
    c = S7.CELLS[cell]
    opts = {'settling_resettle': S7.declaration_for(cell),
            'post_certification': {'persist_certified_models': True, 'hull_polish': False, 'reference': None}}
    if c['nrf_option'] is not None:
        opts['nrf_option'] = c['nrf_option']
    if c['rho0_scale'] is not None:
        opts['rho0_scale'] = c['rho0_scale']
    if c['flex_price_multiplier'] is not None:
        opts['flex_price_multiplier'] = c['flex_price_multiplier']
    return opts


def _key_kwargs_full(entry, conf):
    kw = W183A._key_kwargs(entry, conf)
    kw.update(rho0_scale=entry.get('rho0_scale'), nrf_option=entry.get('nrf_option'))
    return kw


def _expect_refused(spec, decl, must_name):
    """The capture checklist must raise BEFORE any solve, naming every check in `must_name`."""
    out = _outcome(S7.assert_resettle_preconditions, decl, spec, {'tail_enabled_for_this_run': True}, True)
    named = (not out['ok']) and all(n in out['error'] for n in must_name)
    sec = S7.spec_entry_checks(decl, spec)
    failing = sorted(k for k, v in sec.items() if not v)
    return {'refused': not out['ok'], 'error': out.get('error'), 'must_name': list(must_name),
            'names_every_expected_check': named, 'spec_entry_checks_failing': failing,
            'ok': (not out['ok']) and named and set(must_name) <= set(failing)}


def _expect_decl_refused(decl, must_contain):
    out = _outcome(H.validate_settling_resettle, decl)
    return {'refused': not out['ok'], 'error': out.get('error'), 'must_contain': must_contain,
            'ok': (not out['ok']) and must_contain in out['error']}


def gate_iii_planted(H_pre, scratch, committed_keys):
    conf, template = _template_configuration()
    cap = S7.spec_cap(S7.CELL_ORDER[0])
    per_cell, keys = {}, {}
    labels = {'nrf': S7.NRF_OPTION_LABEL == H.NRF_OPTION_LABEL, 'rho0': S7.RHO0_SCALE_LABEL == H.RHO0_SCALE_LABEL,
              'flex': S7.FLEX_PRICE_LABEL == H.FLEX_PRICE_LABEL}
    cand_keys = {'x0': H.candidate_key(H.canonical_candidate(S7.X0_NODES, investment_year=S7.INVESTMENT_YEAR)),
                 'unit': H.candidate_key(H.canonical_candidate(S7.UNIT_NODES, investment_year=S7.INVESTMENT_YEAR))}
    constants = {
        'labels_equal_harness_constants': labels,
        'x0_candidate_key_recomputed': cand_keys['x0'], 'unit_candidate_key_recomputed': cand_keys['unit'],
        'x0_key_equals_module_and_w155': cand_keys['x0'] == S7.X0_CANDIDATE_KEY == W155.X0_CANDIDATE_KEY,
        'unit_key_equals_module_and_ext_v6': cand_keys['unit'] == S7.UNIT_CANDIDATE_KEY == EXT6.UNIT_CANDIDATE_KEY,
        'cap_rule_equals_ext_v6_ungated_cell': (S7.cap_rule(S7.CELL_ORDER[0])
                                                == EXT6.cap_rule('e_no_ageing')),
        'settling_rule_is_the_v6_declaration': (S7.declaration_for(S7.CELL_ORDER[0])['settling_rule']
                                                == EXT6.declaration_for('e_no_ageing')['settling_rule']),
        'captures_equal_e_no_ageing': all(S7.declaration_for(c)['captures']
                                          == EXT6.declaration_for('e_no_ageing')['captures'] for c in S7.CELL_ORDER),
        'expected_blocks_equal_e_no_ageing': all(S7.declaration_for(c)['expected_blocks']
                                                 == EXT6.declaration_for('e_no_ageing')['expected_blocks']
                                                 for c in S7.CELL_ORDER),
        'holds_after_equal_e_no_ageing': all(S7.declaration_for(c)['holds_after']
                                             == EXT6.declaration_for('e_no_ageing')['holds_after']
                                             for c in S7.CELL_ORDER),
        'spec_cap_equals_template_cap': cap == template['cap'],
        'n_cells': len(S7.CELL_ORDER), 'n_e3': len(S7.E3_CELLS), 'n_e2': len(S7.E2_CELLS),
        'cell_table_keys_equal_cell_order': sorted(S7.CELLS) == sorted(S7.CELL_ORDER),
    }
    constants['pass'] = (all(labels.values()) and constants['x0_key_equals_module_and_w155']
                         and constants['unit_key_equals_module_and_ext_v6']
                         and constants['cap_rule_equals_ext_v6_ungated_cell']
                         and constants['settling_rule_is_the_v6_declaration']
                         and constants['captures_equal_e_no_ageing'] and constants['expected_blocks_equal_e_no_ageing']
                         and constants['holds_after_equal_e_no_ageing']
                         and constants['spec_cap_equals_template_cap'] and constants['n_cells'] == 9
                         and constants['n_e3'] == 1 and constants['n_e2'] == 8
                         and constants['cell_table_keys_equal_cell_order'])
    specs = {}
    for cell in S7.CELL_ORDER:
        c = S7.CELLS[cell]
        decl = S7.declaration_for(cell)
        root = os.path.join(scratch, 'planted', cell)
        nodes = S7.X0_NODES if c['plan'] == 'x0' else S7.UNIT_NODES
        _path, digest, spec = H.freeze_campaign_spec(
            root, f'w183i_a_planted_{cell}', [(cell, nodes, _cell_options(cell))], conf, cap=cap, concurrency=1,
            authority=['W183i (a) gate (iii) planted Step-7 entry (scratch freeze outside the repo; never launched)'])
        spec = json.loads(json.dumps(spec, default=str))   # the in-memory form of the frozen text
        specs[cell] = spec
        entry = spec['candidates'][0]
        edecl = entry['settling_resettle']
        routed = H.resettle_hooks_module(edecl)
        v_new = _outcome(H.validate_settling_resettle, edecl)
        v_pre = _outcome(H_pre.validate_settling_resettle, edecl)
        sec = S7.spec_entry_checks(edecl, spec)
        pre = _outcome(S7.assert_resettle_preconditions, edecl, spec, {'tail_enabled_for_this_run': True}, True)
        recomputed = H.evaluation_key(entry['key'], entry['overrides'], **_key_kwargs_full(entry, spec['configuration']))
        state = _outcome(lambda d=edecl: S7.ResettleStateStep7(d, None, cap, reference=None, sink=[]))
        state_summary = _outcome(lambda: state['value'].summary()) if state['ok'] else {'ok': False}
        keys[cell] = entry['eval_key']
        row = {
            'spec_sha256_scratch': digest, 'eval_key': entry['eval_key'], 'candidate_key': entry['key'],
            'entry_keys': sorted(entry), 'spec_level_labels': sorted(k for k in spec if k.endswith('_label')),
            'entry_options': {k: entry.get(k) for k in ('nrf_option', 'rho0_scale', 'flex_price_multiplier')},
            'routed_module': routed.__name__, 'routed_to_the_new_module': routed is S7,
            'validates_patched_equal_to_the_declaration': v_new['ok'] and _canon(v_new['value']) == _canon(decl),
            'pre_patch_harness_refuses': (not v_pre['ok']), 'pre_patch_error': v_pre.get('error'),
            'spec_entry_checks': sec, 'spec_entry_checks_all_true': all(sec.values()),
            'capture_checklist_passes': pre['ok'], 'capture_checklist_error': pre.get('error'),
            'capture_checklist_n_checks': len(pre['value']) if pre['ok'] else None,
            'eval_key_recomputes': recomputed == entry['eval_key'],
            'state_constructs': state['ok'], 'state_error': state.get('error'),
            'state_summary_schema': (state_summary.get('value') or {}).get('schema') if state_summary.get('ok') else None,
            'eval_key_in_committed_v2_specs': entry['eval_key'] in committed_keys,
        }
        row['pass'] = (row['routed_to_the_new_module'] and row['validates_patched_equal_to_the_declaration']
                       and row['pre_patch_harness_refuses'] and row['spec_entry_checks_all_true']
                       and row['capture_checklist_passes'] and row['eval_key_recomputes'] and row['state_constructs']
                       and row['state_summary_schema'] == S7.SCHEMA)
        per_cell[cell] = row
        _log(f'   planted {cell}: key {entry["eval_key"][:16]} routed {routed.__name__} pass {row["pass"]} '
             f'(checklist {row["capture_checklist_n_checks"]} checks; pre-patch refuses {row["pre_patch_harness_refuses"]})')

    # ---- refusals ----
    e3, e2_1x, e2_2x = 'e3_cnrf_x0_1x', 'e2_x0_1x_rho0p5', 'e2_unit_2x_rho2p0'
    refusals = {}
    d = S7.declaration_for(e3)
    refusals['R1_renamed_cell'] = _expect_decl_refused(dict(d, cell=e3 + '_renamed'), 'cell must be one of')
    refusals['R2_committed_ext_v6_cell_name_under_the_step7_schema'] = _expect_decl_refused(
        dict(d, cell='e_no_ageing'), 'cell must be one of')
    refusals['R3_step7_declaration_under_the_ext_v6_schema'] = _expect_decl_refused(
        dict(d, schema=EXT6.DECLARATION_SCHEMA), 'cell must be one of')
    refusals['R4_mutated_expected_rho0_scale'] = _expect_decl_refused(
        dict(S7.declaration_for(e2_1x), rho0_scale_expected=2.0), "differs from the Step-7 declaration")
    refusals['R5_int_where_a_float_is_declared'] = _expect_decl_refused(
        dict(S7.declaration_for(e2_2x), flex_price_multiplier_expected=2), "differs from the Step-7 declaration")
    refusals['R6_nrf_option_expected_dropped'] = _expect_decl_refused(
        {k: v for k, v in d.items() if k != 'nrf_option_expected'}, 'must have exactly')
    refusals['R7_early_stop_key'] = _expect_decl_refused(dict(d, early_stop={'after': 1}), 'must NOT carry')

    def mutated(cell, fn):
        spec = copy.deepcopy(specs[cell])
        fn(spec, spec['candidates'][0])
        return spec

    def drop(*names, level='entry'):
        def fn(spec, entry):
            for n in names:
                (entry if level == 'entry' else spec).pop(n)
        return fn

    def add(**kv):
        def fn(_spec, entry):
            entry.update(kv)
        return fn

    cases = {
        'E1_e3_entry_missing_nrf_option_and_label': (e3, drop('nrf_option', 'nrf_option_label'),
                                                     ['step7_nrf_option_is_the_declared_value',
                                                      'step7_nrf_option_label_at_entry_and_spec_level']),
        'E2_e3_entry_missing_nrf_option_only': (e3, drop('nrf_option'),
                                                ['step7_nrf_option_is_the_declared_value']),
        'E3_e3_entry_missing_nrf_option_label': (e3, drop('nrf_option_label'),
                                                 ['step7_nrf_option_label_at_entry_and_spec_level']),
        'E4_e3_spec_missing_nrf_option_label': (e3, drop('nrf_option_label', level='spec'),
                                                ['step7_nrf_option_label_at_entry_and_spec_level']),
        'E5_e3_entry_nrf_option_False': (e3, add(nrf_option=False),
                                         ['step7_nrf_option_is_the_declared_value']),
        'E6_e2_entry_missing_rho0_scale': (e2_1x, drop('rho0_scale'), ['step7_rho0_scale_is_the_declared_value']),
        'E7_e2_entry_missing_rho0_scale_label': (e2_1x, drop('rho0_scale_label'),
                                                 ['step7_rho0_scale_label_at_entry_and_spec_level']),
        'E8_e2_spec_missing_rho0_scale_label': (e2_1x, drop('rho0_scale_label', level='spec'),
                                                ['step7_rho0_scale_label_at_entry_and_spec_level']),
        'E9_e2_entry_rho0_scale_other_value': (e2_1x, add(rho0_scale=2.0), ['step7_rho0_scale_is_the_declared_value']),
        'E10_e2_2x_entry_missing_flex_multiplier': (e2_2x, drop('flex_price_multiplier'),
                                                    ['step7_flex_price_multiplier_is_the_declared_value']),
        'E11_e2_2x_entry_missing_flex_label': (e2_2x, drop('flex_price_label'),
                                               ['step7_flex_price_multiplier_label_at_entry_and_spec_level']),
        'E12_e2_2x_spec_missing_flex_label': (e2_2x, drop('flex_price_label', level='spec'),
                                              ['step7_flex_price_multiplier_label_at_entry_and_spec_level']),
        'E13_e2_entry_carrying_an_undeclared_nrf_option': (
            e2_1x, add(nrf_option=True, nrf_option_label=S7.NRF_OPTION_LABEL),
            ['step7_nrf_option_absent_from_the_entry',
             'step7_entry_keys_are_exactly_the_base_and_the_declared_options']),
        'E14_e3_entry_carrying_an_undeclared_rho0_scale_1p0': (
            e3, add(rho0_scale=1.0), ['step7_rho0_scale_absent_from_the_entry',
                                      'step7_entry_keys_are_exactly_the_base_and_the_declared_options']),
        'E15_e2_1x_entry_carrying_flex_multiplier_1p0': (
            e2_1x, add(flex_price_multiplier=1.0),
            ['step7_flex_price_multiplier_absent_from_the_entry',
             'step7_entry_keys_are_exactly_the_base_and_the_declared_options']),
        'E16_entry_carrying_a_model_variant': (
            e3, add(model_variant=dict(EXT6.ARMS['no_ageing']), model_variant_label=EXT6.MODEL_VARIANT_LABEL),
            ['step7_no_model_variant_arm_a_by_absence',
             'step7_entry_keys_are_exactly_the_base_and_the_declared_options']),
        'E17_entry_label_not_the_cell': (e2_2x, add(label='e2_unit_2x_rho2p0_x'), ['step7_entry_label_is_the_cell']),
        'E18_entry_with_an_aa_override': (e2_2x, add(overrides={'anderson_acceleration': {'enabled': False}}),
                                          ['step7_entry_overrides_empty']),
    }
    for name, (cell, fn, must) in cases.items():
        spec = mutated(cell, fn)
        refusals[name] = _expect_refused(spec, spec['candidates'][0]['settling_resettle'], must)
    for name, r in refusals.items():
        _log(f'   refusal {name}: {"OK" if r["ok"] else "FAILED"} ({(r.get("error") or "")[:140]})')
    distinct = len(set(keys.values())) == len(keys)
    out = {'template': template, 'configuration_used': conf, 'constants': constants, 'per_cell': per_cell,
           'planted_eval_keys': keys, 'planted_keys_distinct': distinct,
           'n_planted_keys_found_in_committed_v2_specs': sum(1 for k in keys.values() if k in committed_keys),
           'refusals': refusals,
           'pass': (constants['pass'] and all(r['pass'] for r in per_cell.values()) and distinct
                    and all(r['ok'] for r in refusals.values()))}
    return out


def committed_v2_eval_keys():
    keys = set()
    for rel in _git(['ls-files', RESULTS_REL]).splitlines():
        if not (os.path.basename(rel).startswith('campaign_spec_') and rel.endswith('.json')):
            continue
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            spec = json.load(handle)
        if spec.get('schema') == W183A.SPEC_SCHEMA_V2:
            keys |= {e.get('eval_key') for e in spec.get('candidates') or []}
    return keys


# ======================================================================================================================
#  run
# ======================================================================================================================
def _guards():
    mine, w183a = GUARD.verify(0), W183A.GUARD.verify(0)
    return {'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': mine},
            'imported_w183a_module_guard': {'counts': dict(W183A.GUARD.counts), 'verify_0_failures': w183a},
            'pickle': dict(PICKLE_COUNTS), 'imported_w183a_pickle_counts': dict(W183A.PICKLE_COUNTS),
            'pickle_blocked_at_end': pickle.load is _blocked_load and pickle.loads is _blocked_loads,
            'model_build': dict(MODEL_BUILD_COUNTS),
            'all_zero': (not mine and not w183a and PICKLE_COUNTS == {'load': 0, 'loads': 0}
                         and W183A.PICKLE_COUNTS == {'load': 0, 'loads': 0}
                         and MODEL_BUILD_COUNTS['ConcreteModel.__init__'] == 0)}


def _preconditions(out_dir, scratch):
    failures = []
    if not os.path.isdir(out_dir):
        failures.append(f'output directory {out_dir} does not exist (create it with the launch-log redirect)')
    elif sorted(set(os.listdir(out_dir)) - {'launch.log'}):
        failures.append(f'output directory {out_dir} is not fresh: {sorted(os.listdir(out_dir))}')
    if os.path.exists(scratch):
        failures.append(f'scratch dir exists: {scratch}')
    if os.path.abspath(scratch).startswith(REPO + os.sep):
        failures.append(f'scratch dir must be outside the repo: {scratch}')
    for lock in (H.CAMPAIGN_LOCK_PATH, H.LEGACY_RUN_LOCK_PATH):
        if os.path.exists(lock):
            failures.append(f'lock present: {lock}')
    procs = {}
    for line in subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True,
                               text=True).stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    ancestors, pid = set(), os.getpid()
    while pid in procs and pid not in ancestors and pid > 1:
        ancestors.add(pid)
        pid = procs[pid][0]
    others = [f'{p} {cmd}' for p, (_pp, cmd) in procs.items()
              if ('p515_' in cmd and 'python' in cmd) and p not in ancestors]
    if others:
        failures.append(f'another project process is running (refusing to run concurrently): {others}')
    status = _git(['status', '--porcelain', '--', SCRIPT, HOOKS_REL, HARNESS_REL, WRAPPER_REL])
    tracked = {f: bool(_git(['ls-files', f]).strip()) for f in (SCRIPT, HOOKS_REL, HARNESS_REL, WRAPPER_REL)}
    if status.strip() or not all(tracked.values()):
        failures.append(f'gated files not committed and clean: {status!r} {tracked}')
    return failures


def run(scratch):
    started = time.time()
    out_dir = os.path.join(REPO, OUT_REL)
    failures = _preconditions(out_dir, scratch)
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        return 1
    lock = os.path.join(out_dir, LOCK_NAME)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.makedirs(scratch)
        return _body(out_dir, scratch, started)
    finally:
        os.close(fd)
        os.remove(lock)


def _body(out_dir, scratch, started):
    head = _git(['rev-parse', 'HEAD']).strip()
    hashes = {n: _sha_file(os.path.join(REPO, n)) for n in (HARNESS_REL, HOOKS_REL, SCRIPT, WRAPPER_REL,
                                                            'p515_s53_w183a_option_tests.py',
                                                            'p515_s53_w142_resettle_v6_hooks.py',
                                                            'p515_s53_w142_resettle_ext_v6_hooks.py',
                                                            'p515_s53_w155_a64_hooks.py', 'gate_result_io.py',
                                                            'p513_solve_profile_guard.py')}
    _log(f'{STAGE}; HEAD {head}')
    for n, s in hashes.items():
        _log(f'   sha256 {n}: {s}')
    result = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
              'argv': sys.argv, 'interpreter': sys.executable, 'code_sha256': hashes,
              'new_schema': S7.SCHEMA, 'new_module': S7.__name__,
              'objective_convention': 'no objective value is computed or reported (zero-solve key / routing gates)'}
    gates = {}
    try:
        H_pre, pre_info = load_pre_patch_harness(scratch)
        result['pre_patch_harness'] = pre_info
        _log(f'pre-patch harness {pre_info["commit"][:12]} blob {pre_info["blob_sha256"][:16]}; patched '
             f'{pre_info["patched_sha256"][:16]}; diff exactly the three router lines: '
             f'{pre_info["diff_is_exactly_the_three_router_lines"]} {pre_info["diff_added_lines"]}')
        t1 = W183A.t1_old_keys(H_pre)
        t1['expected'] = EXPECTED_T1
        t1['counts_as_expected'] = (t1['n_v2_specs'] == EXPECTED_T1['n_v2_specs']
                                    and t1['n_entries'] == EXPECTED_T1['n_entries'])
        t1['method'] = ('p515_s53_w183a_option_tests.t1_old_keys by import: H = the patched working-tree harness, '
                        f'H_pre = the harness blob at {PRE_PATCH_COMMIT} (sha256 {PRE_PATCH_HARNESS_SHA256})')
        t1['pass_with_patch_diff'] = bool(t1['pass'] and pre_info['diff_is_exactly_the_three_router_lines'])
        gates['i_keys_recompute'] = t1
        _log(f"(i) {t1['n_v2_specs']} v2 specs ({t1['n_tracked_spec_files']} tracked spec files), {t1['n_entries']} "
             f"entries; new == stored {t1['n_new_equal_stored']}; new == pre-patch {t1['n_new_equal_pre_change']}; "
             f"skips {t1['n_skips']}; errors {len(t1['errors'])}; PASS {t1['pass']}; counts as expected "
             f"{t1['counts_as_expected']}")
        decls = collect_declarations()
        ii, neutral = gate_ii_iii_committed(H_pre, decls)
        ii['scan'] = {k: v for k, v in decls.items() if k not in ('A', 'B')}
        ii['scope_searched'] = ('A: entries of every git-tracked campaign_spec_*.json under data/SRP1/Results (any '
                                'schema); B: every git-tracked *.json under data/SRP1/Results containing the text '
                                '"settling_resettle", parsed, every dict value under that key at any depth. Not '
                                'searched: untracked files, .jsonl / .json.gz, files outside data/SRP1/Results.')
        gates['ii_routing_identity'] = ii
        for scope, rows in ii['routing_by_scope_and_schema'].items():
            for schema, row in sorted(rows.items()):
                _log(f"(ii) {scope} {schema}: n {row['n']} -> {row['modules_patched']} (pre {row['modules_pre_patch']}) "
                     f"same {row['all_same']} validate same {row['validate_same']} "
                     f"(ok {row['n_validate_ok']} / refused {row['n_validate_refused']})")
        _log(f"(ii) A {ii['n_declarations_A_campaign_spec_entries']} B {ii['n_declarations_B_any_tracked_json']} "
             f"mismatches {len(ii['routing_mismatches'])} validate mismatches {len(ii['validate_outcome_mismatches'])} "
             f"PASS {ii['pass']}")
        committed_keys = committed_v2_eval_keys()
        planted = gate_iii_planted(H_pre, scratch, committed_keys)
        gates['iii_neutrality'] = {'committed': neutral, 'planted': planted,
                                   'pass': neutral['pass'] and planted['pass']}
        _log(f"(iii) committed hits {neutral['n_committed_declarations_routed_to_the_new_module_or_carrying_its_schema']}; "
             f"schema string in {neutral['tracked_files_containing_the_schema_string']}; planted pass "
             f"{planted['pass']} (constants {planted['constants']['pass']}, keys distinct "
             f"{planted['planted_keys_distinct']}, refusals {sum(r['ok'] for r in planted['refusals'].values())}/"
             f"{len(planted['refusals'])}); PASS {gates['iii_neutrality']['pass']}")
    except Exception as error:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        result['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': tb}
    result['gates'] = gates
    result['guards'] = _guards()
    result['summary'] = {
        'i_keys_recompute': (gates.get('i_keys_recompute') or {}).get('pass_with_patch_diff'),
        'i_counts_as_expected': (gates.get('i_keys_recompute') or {}).get('counts_as_expected'),
        'ii_routing_identity': (gates.get('ii_routing_identity') or {}).get('pass'),
        'iii_neutrality': (gates.get('iii_neutrality') or {}).get('pass'),
        'guards_all_zero': result['guards']['all_zero']}
    result['all_pass'] = bool(set(gates) == {'i_keys_recompute', 'ii_routing_identity', 'iii_neutrality'}
                              and all(v is True for v in result['summary'].values()) and 'error' not in result)
    result['wall_clock_s'] = time.time() - started
    json_path = os.path.join(out_dir, 'w183i_a_router_gates.json')
    W183A_refuse(json_path)
    with open(json_path, 'w') as handle:
        handle.write(GRIO.dumps(result, indent=1, sort_keys=True))
    manifest = {os.path.relpath(json_path, REPO): _sha_file(json_path), **{n: s for n, s in hashes.items()}}
    mpath = os.path.join(out_dir, 'manifest_sha256.json')
    W183A_refuse(mpath)
    with open(mpath, 'w') as handle:
        handle.write(GRIO.dumps(dict(sorted(manifest.items())), indent=1, sort_keys=True))
    _log(f"ALL_PASS={result['all_pass']} summary={result['summary']} guards={result['guards']['solve_profile_guard']} "
         f"pickle={PICKLE_COUNTS} build={MODEL_BUILD_COUNTS} wall={result['wall_clock_s']:.1f}s")
    return 0 if result['all_pass'] else 1


def W183A_refuse(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def mode_post_run():
    out_dir = os.path.join(REPO, OUT_REL)
    path = os.path.join(out_dir, 'manifest_post_run_sha256.json')
    W183A_refuse(path)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            manifest[os.path.relpath(os.path.join(root, fname), REPO)] = _sha_file(os.path.join(root, fname))
    with open(path, 'w') as handle:
        handle.write(GRIO.dumps(dict(sorted(manifest.items())), indent=1, sort_keys=True))
    guards = _guards()
    _log(f'wrote {os.path.relpath(path, REPO)} ({len(manifest)} files); guards all zero {guards["all_zero"]}')
    return 0 if guards['all_zero'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', help='scratch dir outside the repo; must not exist')
    parser.add_argument('--post-run', action='store_true')
    args = parser.parse_args(argv)
    if args.post_run:
        return mode_post_run()
    if not args.scratch:
        parser.error('--scratch is required')
    return run(os.path.abspath(args.scratch))


if __name__ == '__main__':
    sys.exit(main())
