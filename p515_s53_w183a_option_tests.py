"""
P5.15 Addendum 83 ruling 3 (ii) and the harness half of 3 (iv) (task W183a) -- ZERO-SOLVE tests of the two declared,
keyed evaluation options `rho0_scale` and `nrf_option` of `p515_s44_campaign_harness.py`.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 83 (ruling 3 (ii), (iv); W183 order, W183a: "a unit test that the key
changes and the S39 refusal stays for an undeclared rho"); P5_15_W182E_CODE_PATHS.md (Z6); P5_15_W183_0_NL_REFERENCE.md.

Nothing is solved: `SolveProfileGuard(permitted=())` is installed before any project import and verified at exactly 0
at the end; `pickle.load` / `pickle.loads` are blocked for the whole run and counted (0 expected). Model BUILDS happen
only in T3/T4 (the configuration hook's own ESSO probe builds, as in every campaign child) and in T5's subprocess (the
W183-0 NL-digest harness, its own guard verified at 0).

TESTS
  T1  Old keys unchanged. Every git-tracked `campaign_spec_*.json` under data/SRP1/Results: schema
      'p515_s44_campaign_spec_v2' -> every entry's eval key recomputed from the stored inputs (entry `key`,
      `overrides`, `model_variant`, `flex_price_multiplier`, `interface_deviation_premium`,
      `certification_continuation`, `settling_continuation` / `_extension` / `_resettle`; configuration
      `case_file_anderson_acceleration`, `ess_ageing_baseline`, `derived_instance`, `convergence_depth_tail`) with the
      CURRENT harness and no new kwargs, asserted equal to the stored `eval_key`; and, beside it, with the PRE-CHANGE
      harness (the blob at PRE_CHANGE_COMMIT, loaded from a scratch copy) -- new == pre on every entry. Every other
      file is a skip, with its reason.
  T2  Key algebra: rho0_scale 1.0 == absent; nrf_option False == absent; 0.5 != base; 2.0 != 0.5; True != base;
      0.5 + True distinct from both -- on several base configurations; the four wrappers (settling_resettle,
      certification_continuation, settling_continuation, settling_extension; declarations taken from committed specs)
      with rho0_scale 0.5 / nrf True differ from the stored key and equal sha256{base_evaluation_key: <base WITH the
      option>, <wrapper>: <declaration>}; validator refusals (True as the scale, 1 as the option, and others), at
      `evaluation_key` and at `freeze_campaign_spec` (nothing written); a scratch freeze (outside the repo) records
      the options and labels and refuses 1.0 / False beside the absent entry as duplicates; the child's label rule
      (`_child_real` refuses a non-default option without its label at spec or entry level; a sentinel on the entry's
      next read proves the labelled entry passes).
  T3  Configuration hook, zero solves (precedent `p515_s47_case_file_baseline_check.py`: `_construct_arm_planning` +
      `H._config_hook_factory(...)` on a spec-like ext-v6 configuration block, unit candidate node 7 0.25/1.0 2025):
      undeclared + case-file rho passes; undeclared + one rho entry mutated raises naming the failing S39 check (a DSO
      pf entry and the ESSO entry); declared 0.5 / 2.0 (and 1.0 explicit) -> every entry equals S39 x s exactly;
      declared 0.5 + a mutated case-file rho still raises and nothing is scaled.
  T4  nrf_option True -> the hook raises naming the missing production support (production has none before W183c).
      T4b (SIMULATED support, test-only monkeypatch of the marker; not production): the hook sets the flag on the TSO
      and every DSO and reads it back; an undeclared run with a flag on raises.
  T5  `p515_s53_w183_nl_digest.py` re-run on the changed harness for x0_A_1x and unit_A_2x (no determinism child),
      label post_w183a, into a fresh directory; every NL sha256 (plain, labelled .nl/.row/.col), block digest and
      component digest asserted byte-identical to the committed W183-0 reference (`w183_nl_reference_pre`, gz JSON).

OUTPUT (write-once; the directory may pre-exist holding only launch.log):
  <out>/w183a_option_tests.json   (gate_result_io)      <out>/manifest_sha256.json
  <t5-out>/...  the NL-digest harness's own output, + w183_nl_digest.json.gz (gzip mtime 0) and
  manifest_post_run_sha256.json
LAUNCH (attached, alone, both streams captured; this script and the harness committed first):
  mkdir -p data/SRP1/Results/P515S55/w183a_option_tests
  NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w183a_option_tests.py --out data/SRP1/Results/P515S55/w183a_option_tests \\
      --t5-out data/SRP1/Results/P515S55/w183_nl_post_w183a --scratch <scratchpad>/w183a_tests \\
      --expect-production-commit 575bfec4 > data/SRP1/Results/P515S55/w183a_option_tests/launch.log 2>&1
Exit 0 iff every test passes.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183a option tests (zero solves)').install()

import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183a: pickle.load called -- no pickle loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183a: pickle.loads called -- no pickle loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
HARNESS_REL = 'p515_s44_campaign_harness.py'
NL_DIGEST_REL = 'p515_s53_w183_nl_digest.py'
STAGE = 'P5.15 Addendum 83 W183a -- rho0_scale / nrf_option keyed evaluation options: tests T1-T5 (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (ii), (iv); W183 order (W183a)',
             'P5_15_W182E_CODE_PATHS.md Z6', 'P5_15_W183_0_NL_REFERENCE.md']
PRE_CHANGE_COMMIT = '33dbf690'   # HEAD before W183a: the last commit whose harness has no rho0_scale / nrf_option
SPEC_SCHEMA_V2 = 'p515_s44_campaign_spec_v2'
CAP = 300
REQUIRED_CONSECUTIVE = 10
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CONVERGENCE_DEPTH_TAIL = {'compl_inf_tol': 1e-06, 'enabled': True}
INVESTMENT_YEAR = 2025
UNIT = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
X0 = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
NO_AGEING = {'ageing_enabled': False, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0,
             'eol_retention_r': 0.5}
WRAPPERS = ('settling_resettle', 'certification_continuation', 'settling_continuation', 'settling_extension')
T5_CONFIGS = ('x0_A_1x', 'unit_A_2x')
REFERENCE_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183_nl_reference_pre')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[W183a {datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _git_status(paths):
    text = _git(['status', '--porcelain', '--'] + list(paths))
    return {line[3:]: line[:2] for line in text.splitlines()}


def _key_kwargs(entry, conf):
    """The stored inputs of one v2 entry, as `freeze_campaign_spec` passed them to `evaluation_key`."""
    return dict(case_file_aa=conf.get('case_file_anderson_acceleration'),
                model_variant=entry.get('model_variant'),
                ess_ageing_baseline=conf.get('ess_ageing_baseline'),
                flex_price_multiplier=entry.get('flex_price_multiplier'),
                derived_instance=conf.get('derived_instance'),
                interface_deviation_premium=entry.get('interface_deviation_premium'),
                convergence_depth_tail=conf.get('convergence_depth_tail'),
                certification_continuation=entry.get('certification_continuation'),
                settling_continuation=entry.get('settling_continuation'),
                settling_extension=entry.get('settling_extension'),
                settling_resettle=entry.get('settling_resettle'))


def _raises(fn, *args, **kwargs):
    """(raised?, 'Type: message')."""
    try:
        fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        return True, f'{type(error).__name__}: {error}'
    return False, None


# ======================================================================================================================
#  T1
# ======================================================================================================================
def load_pre_change_harness(scratch):
    blob = subprocess.run(['git', 'show', f'{PRE_CHANGE_COMMIT}:{HARNESS_REL}'], cwd=REPO, capture_output=True,
                          check=True).stdout
    path = os.path.join(scratch, 'pre_change_harness', 'p515_s44_campaign_harness_pre_w183a.py')
    os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as handle:
        handle.write(blob)
    spec = importlib.util.spec_from_file_location('p515_s44_campaign_harness_pre_w183a', path)
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved   # the module inserts its own directory; keep the repo first
    return module, {'commit': _git(['rev-parse', PRE_CHANGE_COMMIT]).strip(), 'path_in_scratch': path,
                    'blob_sha256': _sha_bytes(blob),
                    'has_rho0_scale_kwarg': 'rho0_scale' in __import__('inspect').signature(
                        module.evaluation_key).parameters}


def t1_old_keys(H_pre):
    files = [f for f in _git(['ls-files', 'data/SRP1/Results']).splitlines()
             if os.path.basename(f).startswith('campaign_spec_') and f.endswith('.json')]
    per_spec, skips, mismatches_stored, mismatches_pre, errors = [], [], [], [], []
    n_entries = n_specs = 0
    entries_with_new_options = []
    for rel in sorted(files):
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            skips.append({'file': rel, 'reason': 'tracked but missing from the working tree'})
            continue
        with open(path) as handle:
            spec = json.load(handle)
        schema = spec.get('schema')
        if schema != SPEC_SCHEMA_V2:
            skips.append({'file': rel, 'reason': f'schema {schema!r} != {SPEC_SCHEMA_V2!r} (v1 specs carry no '
                                                  f'eval_key: the eval key is the candidate key)',
                          'n_candidates': len(spec.get('candidates') or [])})
            continue
        conf = spec['configuration']
        n_specs += 1
        row = {'file': rel, 'sha256': _sha_file(path), 'n_entries': 0, 'n_equal_stored': 0, 'n_equal_pre': 0}
        for entry in spec['candidates']:
            n_entries += 1
            row['n_entries'] += 1
            if 'rho0_scale' in entry or 'nrf_option' in entry:
                entries_with_new_options.append({'file': rel, 'label': entry.get('label')})
            kwargs = _key_kwargs(entry, conf)
            try:
                new = H.evaluation_key(entry['key'], entry['overrides'], **kwargs)
                pre = H_pre.evaluation_key(entry['key'], entry['overrides'], **kwargs)
            except Exception as error:  # noqa: BLE001
                errors.append({'file': rel, 'label': entry.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == entry['eval_key']:
                row['n_equal_stored'] += 1
            else:
                mismatches_stored.append({'file': rel, 'label': entry.get('label'), 'stored': entry['eval_key'],
                                          'new': new, 'pre_change': pre, 'pre_equals_stored': pre == entry['eval_key']})
            if new == pre:
                row['n_equal_pre'] += 1
            else:
                mismatches_pre.append({'file': rel, 'label': entry.get('label'), 'new': new, 'pre_change': pre})
        per_spec.append(row)
    distinct_spec_sha = len({r['sha256'] for r in per_spec})
    out = {
        'n_tracked_spec_files': len(files), 'n_v2_specs': n_specs, 'n_v2_specs_distinct_content': distinct_spec_sha,
        'n_entries': n_entries,
        'n_new_equal_stored': sum(r['n_equal_stored'] for r in per_spec),
        'n_new_equal_pre_change': sum(r['n_equal_pre'] for r in per_spec),
        'mismatches_vs_stored': mismatches_stored, 'mismatches_vs_pre_change': mismatches_pre, 'errors': errors,
        'skips': skips, 'n_skips': len(skips),
        'entries_already_declaring_rho0_scale_or_nrf_option': entries_with_new_options,
        'per_spec': per_spec,
        'inputs_recomputed_from': ['entry.key', 'entry.overrides', 'entry.model_variant', 'entry.flex_price_multiplier',
                                   'entry.interface_deviation_premium', 'entry.certification_continuation',
                                   'entry.settling_continuation', 'entry.settling_extension', 'entry.settling_resettle',
                                   'configuration.case_file_anderson_acceleration',
                                   'configuration.ess_ageing_baseline', 'configuration.derived_instance',
                                   'configuration.convergence_depth_tail'],
    }
    out['pass'] = (n_specs > 0 and n_entries > 0 and not mismatches_stored and not mismatches_pre and not errors
                   and out['n_new_equal_stored'] == n_entries and out['n_new_equal_pre_change'] == n_entries
                   and not entries_with_new_options)
    return out


# ======================================================================================================================
#  T2
# ======================================================================================================================
def t2_key_algebra(scratch, declared_ageing):
    ck_unit = H.candidate_key(H.canonical_candidate(UNIT, investment_year=INVESTMENT_YEAR))
    ck_x0 = H.candidate_key(H.canonical_candidate(X0, investment_year=INVESTMENT_YEAR))
    bases = {
        'bare_case_file': (ck_unit, {}, {}),
        'aa_override': (ck_unit, {'anderson_acceleration': {'enabled': False}}, {}),
        'ext_v6_block': (ck_unit, {}, dict(case_file_aa=dict(CASE_FILE_AA), ess_ageing_baseline=declared_ageing,
                                           convergence_depth_tail=dict(CONVERGENCE_DEPTH_TAIL))),
        'ext_v6_block_flex2_no_ageing': (ck_unit, {}, dict(case_file_aa=dict(CASE_FILE_AA),
                                                           ess_ageing_baseline=declared_ageing,
                                                           convergence_depth_tail=dict(CONVERGENCE_DEPTH_TAIL),
                                                           model_variant=dict(NO_AGEING), flex_price_multiplier=2.0)),
        'flex2_only': (ck_unit, {}, dict(flex_price_multiplier=2.0)),
        'model_variant_only': (ck_unit, {}, dict(model_variant=dict(NO_AGEING))),
        'premium_only': (ck_unit, {}, dict(interface_deviation_premium={'alpha': 0.05, 'floor': 1.0})),
        'x0_ext_v6_block': (ck_x0, {}, dict(case_file_aa=dict(CASE_FILE_AA), ess_ageing_baseline=declared_ageing,
                                            convergence_depth_tail=dict(CONVERGENCE_DEPTH_TAIL))),
    }
    algebra, ok = {}, True
    for name, (ck, ov, kw) in bases.items():
        base = H.evaluation_key(ck, ov, **kw)
        k = {
            'base': base,
            'rho_1p0': H.evaluation_key(ck, ov, rho0_scale=1.0, **kw),
            'rho_int_1': H.evaluation_key(ck, ov, rho0_scale=1, **kw),
            'nrf_false': H.evaluation_key(ck, ov, nrf_option=False, **kw),
            'rho_1p0_nrf_false': H.evaluation_key(ck, ov, rho0_scale=1.0, nrf_option=False, **kw),
            'rho_0p5': H.evaluation_key(ck, ov, rho0_scale=0.5, **kw),
            'rho_2p0': H.evaluation_key(ck, ov, rho0_scale=2.0, **kw),
            'nrf_true': H.evaluation_key(ck, ov, nrf_option=True, **kw),
            'rho_0p5_nrf_true': H.evaluation_key(ck, ov, rho0_scale=0.5, nrf_option=True, **kw),
        }
        checks = {
            'rho_1p0_equals_absent': k['rho_1p0'] == base,
            'rho_int_1_equals_absent': k['rho_int_1'] == base,
            'nrf_false_equals_absent': k['nrf_false'] == base,
            'rho_1p0_and_nrf_false_equal_absent': k['rho_1p0_nrf_false'] == base,
            'rho_0p5_differs_from_base': k['rho_0p5'] != base,
            'rho_2p0_differs_from_0p5': k['rho_2p0'] != k['rho_0p5'],
            'rho_2p0_differs_from_base': k['rho_2p0'] != base,
            'nrf_true_differs_from_base': k['nrf_true'] != base,
            'both_distinct_from_each_alone_and_base': len({base, k['rho_0p5'], k['nrf_true'],
                                                           k['rho_0p5_nrf_true']}) == 4,
        }
        ok = ok and all(checks.values())
        algebra[name] = {'keys': {n: v[:16] for n, v in k.items()}, 'checks': checks}

    # wrappers: declarations from committed specs
    files = [f for f in _git(['ls-files', 'data/SRP1/Results']).splitlines()
             if os.path.basename(f).startswith('campaign_spec_') and f.endswith('.json')]
    found = {}
    for rel in sorted(files):
        with open(os.path.join(REPO, rel)) as handle:
            spec = json.load(handle)
        if spec.get('schema') != SPEC_SCHEMA_V2:
            continue
        for entry in spec['candidates']:
            for w in WRAPPERS:
                if w in entry and w not in found:
                    found[w] = (rel, entry, spec['configuration'])
    wrappers = {}
    validators = {'settling_resettle': H.validate_settling_resettle,
                  'certification_continuation': H.validate_certification_continuation,
                  'settling_continuation': H.validate_settling_continuation,
                  'settling_extension': H.validate_settling_extension}
    for w in WRAPPERS:
        if w not in found:
            wrappers[w] = {'found': False}
            ok = False
            continue
        rel, entry, conf = found[w]
        kwargs = _key_kwargs(entry, conf)
        without = H.evaluation_key(entry['key'], entry['overrides'], **kwargs)
        with_rho = H.evaluation_key(entry['key'], entry['overrides'], rho0_scale=0.5, **kwargs)
        with_nrf = H.evaluation_key(entry['key'], entry['overrides'], nrf_option=True, **kwargs)
        with_rho1 = H.evaluation_key(entry['key'], entry['overrides'], rho0_scale=1.0, nrf_option=False, **kwargs)
        base_kwargs = {k: v for k, v in kwargs.items() if k not in WRAPPERS}
        base_with_rho = H.evaluation_key(entry['key'], entry['overrides'], rho0_scale=0.5, **base_kwargs)
        expected = hashlib.sha256(json.dumps({'base_evaluation_key': base_with_rho,
                                              w: validators[w](entry[w])},
                                             sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        checks = {'without_equals_stored': without == entry['eval_key'],
                  'with_rho0_scale_0p5_differs': with_rho != without,
                  'with_nrf_true_differs': with_nrf != without,
                  'with_rho_1p0_nrf_false_equals_stored': with_rho1 == entry['eval_key'],
                  'with_rho_0p5_is_wrapper_of_base_with_rho_0p5': with_rho == expected}
        ok = ok and all(checks.values())
        wrappers[w] = {'found': True, 'spec': rel, 'label': entry.get('label'), 'stored': entry['eval_key'][:16],
                       'with_rho_0p5': with_rho[:16], 'with_nrf_true': with_nrf[:16], 'checks': checks}

    # validator refusals
    ck = ck_unit
    refusal_cases = {
        'evaluation_key_rho0_scale_True': lambda: H.evaluation_key(ck, {}, rho0_scale=True),
        'evaluation_key_rho0_scale_False': lambda: H.evaluation_key(ck, {}, rho0_scale=False),
        'evaluation_key_rho0_scale_0': lambda: H.evaluation_key(ck, {}, rho0_scale=0.0),
        'evaluation_key_rho0_scale_negative': lambda: H.evaluation_key(ck, {}, rho0_scale=-0.5),
        'evaluation_key_rho0_scale_nan': lambda: H.evaluation_key(ck, {}, rho0_scale=float('nan')),
        'evaluation_key_rho0_scale_inf': lambda: H.evaluation_key(ck, {}, rho0_scale=math.inf),
        'evaluation_key_rho0_scale_str': lambda: H.evaluation_key(ck, {}, rho0_scale='0.5'),
        'evaluation_key_nrf_option_1': lambda: H.evaluation_key(ck, {}, nrf_option=1),
        'evaluation_key_nrf_option_0': lambda: H.evaluation_key(ck, {}, nrf_option=0),
        'evaluation_key_nrf_option_str': lambda: H.evaluation_key(ck, {}, nrf_option='True '),
        'validate_rho0_scale_True': lambda: H.validate_rho0_scale(True),
        'validate_nrf_option_1': lambda: H.validate_nrf_option(1),
        'hook_factory_rho0_scale_True': lambda: H._config_hook_factory({'configuration': {}}, {}, rho0_scale=True),
        'hook_factory_nrf_option_1': lambda: H._config_hook_factory({'configuration': {}}, {}, nrf_option=1),
    }
    refusals = {}
    for name, fn in refusal_cases.items():
        raised, msg = _raises(fn)
        refusals[name] = {'raised': raised, 'error': msg}
        ok = ok and raised and msg.startswith('ValueError')
    # freeze-level refusals (nothing written)
    freeze_conf = {'name': 'w183a_t2_freeze_test', 'overrides': {}, 'arm_label': 's39_D',
                   'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                   'ess_ageing_baseline': declared_ageing,
                   'ess_ageing_baseline_label': 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)',
                   'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL)}

    def _freeze(root, cands):
        return H.freeze_campaign_spec(root, f'w183a_t2_{os.path.basename(root)}', cands, freeze_conf, cap=CAP,
                                      concurrency=1, authority=['W183a T2 scratch freeze (outside the repo; '
                                                                'never launched)'])
    freeze_refusals = {}
    for name, cands in {
        'rho0_scale_True': [('a', UNIT, {'rho0_scale': True})],
        'nrf_option_1': [('a', UNIT, {'nrf_option': 1})],
        'rho0_scale_1p0_beside_absent_is_duplicate': [('a', UNIT), ('b', UNIT, {'rho0_scale': 1.0})],
        'nrf_option_False_beside_absent_is_duplicate': [('a', UNIT), ('b', UNIT, {'nrf_option': False})],
        'unknown_option_refused': [('a', UNIT, {'rho0': 0.5})],
    }.items():
        root = os.path.join(scratch, 'freeze', name)
        raised, msg = _raises(_freeze, root, cands)
        written = os.path.exists(root)
        freeze_refusals[name] = {'raised': raised, 'error': msg, 'root_created': written}
        ok = ok and raised and msg.startswith('ValueError') and not written
    # a successful scratch freeze
    root = os.path.join(scratch, 'freeze', 'ok')
    cands = [('unit_base', UNIT), ('unit_rho0p5', UNIT, {'rho0_scale': 0.5}), ('unit_rho2', UNIT, {'rho0_scale': 2.0}),
             ('unit_nrf', UNIT, {'nrf_option': True}),
             ('unit_rho0p5_nrf', UNIT, {'rho0_scale': 0.5, 'nrf_option': True}),
             ('x0_rho1_nrf_false', X0, {'rho0_scale': 1.0, 'nrf_option': False}),
             ('unit_noageing_flex2_rho0p5', UNIT, {'rho0_scale': 0.5, 'model_variant': dict(NO_AGEING),
                                                   'flex_price_multiplier': 2.0})]
    spec_path, spec_sha, spec = _freeze(root, cands)
    entries = {e['label']: e for e in spec['candidates']}
    conf_kw = dict(case_file_aa=dict(CASE_FILE_AA), ess_ageing_baseline=declared_ageing,
                   convergence_depth_tail=dict(CONVERGENCE_DEPTH_TAIL))
    expect = {'unit_base': H.evaluation_key(ck_unit, {}, **conf_kw),
              'unit_rho0p5': H.evaluation_key(ck_unit, {}, rho0_scale=0.5, **conf_kw),
              'unit_rho2': H.evaluation_key(ck_unit, {}, rho0_scale=2.0, **conf_kw),
              'unit_nrf': H.evaluation_key(ck_unit, {}, nrf_option=True, **conf_kw),
              'unit_rho0p5_nrf': H.evaluation_key(ck_unit, {}, rho0_scale=0.5, nrf_option=True, **conf_kw),
              'x0_rho1_nrf_false': H.evaluation_key(ck_x0, {}, **conf_kw),
              'unit_noageing_flex2_rho0p5': H.evaluation_key(ck_unit, {}, rho0_scale=0.5,
                                                             model_variant=dict(NO_AGEING),
                                                             flex_price_multiplier=2.0, **conf_kw)}
    fchecks = {
        'every_entry_key_equals_direct_evaluation_key': all(entries[n]['eval_key'] == expect[n] for n in expect),
        'spec_rho0_scale_label': spec.get('rho0_scale_label') == H.RHO0_SCALE_LABEL,
        'spec_nrf_option_label': spec.get('nrf_option_label') == H.NRF_OPTION_LABEL,
        'base_entry_has_no_new_fields': not ({'rho0_scale', 'rho0_scale_label', 'nrf_option', 'nrf_option_label'}
                                             & set(entries['unit_base'])),
        'rho_entries_carry_value_and_label': all(entries[n].get('rho0_scale') == s
                                                 and entries[n].get('rho0_scale_label') == H.RHO0_SCALE_LABEL
                                                 for n, s in (('unit_rho0p5', 0.5), ('unit_rho2', 2.0),
                                                              ('unit_rho0p5_nrf', 0.5),
                                                              ('unit_noageing_flex2_rho0p5', 0.5))),
        'nrf_entries_carry_value_and_label': all(entries[n].get('nrf_option') is True
                                                 and entries[n].get('nrf_option_label') == H.NRF_OPTION_LABEL
                                                 for n in ('unit_nrf', 'unit_rho0p5_nrf')),
        'default_values_recorded_without_label': (entries['x0_rho1_nrf_false'].get('rho0_scale') == 1.0
                                                  and entries['x0_rho1_nrf_false'].get('nrf_option') is False
                                                  and 'rho0_scale_label' not in entries['x0_rho1_nrf_false']
                                                  and 'nrf_option_label' not in entries['x0_rho1_nrf_false']),
        'spec_outside_repo': not os.path.abspath(spec_path).startswith(REPO + os.sep),
    }
    ok = ok and all(fchecks.values())

    # the child's label rule (`_child_real`), stopped by a sentinel on the entry's first read after the label checks
    class _PastLabels(Exception):
        pass

    class _Entry(dict):
        def get(self, key, default=None):
            if key == 'post_certification':
                raise _PastLabels('label checks passed')
            return super().get(key, default)

    def _child(spec_obj, entry_obj):
        return H._child_real(None, spec_obj, spec_path, _Entry(entry_obj), None, None, None, time.time(), progress={})

    child_cases = {}
    for opt, lab in (('rho0_scale', 'rho0_scale_label'), ('nrf_option', 'nrf_option_label')):
        name = 'unit_rho0p5' if opt == 'rho0_scale' else 'unit_nrf'
        good = dict(entries[name])
        no_entry_label = {k: v for k, v in good.items() if k != lab}
        wrong_entry_label = dict(good, **{lab: 'WRONG'})
        spec_no_label = {k: v for k, v in spec.items() if k != lab}
        for case, (sp, en) in {f'{opt}_labelled_at_spec_and_entry': (spec, good),
                               f'{opt}_entry_label_missing': (spec, no_entry_label),
                               f'{opt}_entry_label_wrong': (spec, wrong_entry_label),
                               f'{opt}_spec_label_missing': (spec_no_label, good)}.items():
            try:
                _child(sp, en)
                outcome = {'outcome': 'returned (unexpected)'}
            except _PastLabels:
                outcome = {'outcome': 'passed the label checks (sentinel reached)'}
            except Exception as error:  # noqa: BLE001
                outcome = {'outcome': 'raised', 'error': f'{type(error).__name__}: {error}'}
            child_cases[case] = outcome
    default_entry = dict(entries['x0_rho1_nrf_false'])
    try:
        _child({k: v for k, v in spec.items() if k not in ('rho0_scale_label', 'nrf_option_label')}, default_entry)
        child_cases['defaults_without_any_label'] = {'outcome': 'returned (unexpected)'}
    except _PastLabels:
        child_cases['defaults_without_any_label'] = {'outcome': 'passed the label checks (sentinel reached)'}
    except Exception as error:  # noqa: BLE001
        child_cases['defaults_without_any_label'] = {'outcome': 'raised', 'error': f'{type(error).__name__}: {error}'}
    cchecks = {}
    for opt, label_const in (('rho0_scale', H.RHO0_SCALE_LABEL), ('nrf_option', H.NRF_OPTION_LABEL)):
        cchecks[f'{opt}_labelled_passes'] = child_cases[f'{opt}_labelled_at_spec_and_entry']['outcome'].startswith(
            'passed')
        for bad in ('entry_label_missing', 'entry_label_wrong', 'spec_label_missing'):
            c = child_cases[f'{opt}_{bad}']
            cchecks[f'{opt}_{bad}_refused'] = (c['outcome'] == 'raised' and c['error'].startswith('RuntimeError')
                                              and f'{opt} entry' in c['error'] and label_const in c['error'])
    cchecks['defaults_without_any_label_pass'] = child_cases['defaults_without_any_label']['outcome'].startswith(
        'passed')
    ok = ok and all(cchecks.values())
    return {'pass': ok, 'algebra': algebra, 'wrappers': wrappers, 'refusals': refusals,
            'freeze_refusals': freeze_refusals,
            'freeze_ok': {'spec_path_in_scratch': spec_path, 'spec_sha256': spec_sha, 'checks': fchecks,
                          'entries': {n: {k: e.get(k) for k in ('eval_key', 'rho0_scale', 'rho0_scale_label',
                                                                'nrf_option', 'nrf_option_label', 'model_variant',
                                                                'flex_price_multiplier', 'flex_price_label')}
                                      for n, e in entries.items()},
                          'spec_labels': {'rho0_scale_label': spec.get('rho0_scale_label'),
                                          'nrf_option_label': spec.get('nrf_option_label')}},
            'child_label_rule': {'cases': child_cases, 'checks': cchecks}}


# ======================================================================================================================
#  T3 / T4
# ======================================================================================================================
def _rho_snapshot(planning):
    return {g: dict(planning.params.admm.rho[g]) for g in ('v', 'pf', 'ess')}


def run_hook_case(G, ctx, case, mutate=None, **hook_kwargs):
    report, holder = {}, {}
    eval_id = f"p515s55_w183a_{ctx['run_tag']}_{case}"
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(ctx['scratch'], 'hook', case), report, investment_map=UNIT, eval_id=eval_id,
        num_max_iters_override=CAP, apply_rho=False, investment_year=INVESTMENT_YEAR)
    case_file_rho = _rho_snapshot(planning)
    mutation = mutate(planning) if mutate is not None else None
    before = _rho_snapshot(planning)
    hook = H._config_hook_factory(ctx['spec'], holder, overrides={}, investment_year=INVESTMENT_YEAR,
                                  expected_floor_rows=None, **hook_kwargs)
    try:
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        outcome, error = 'passed', None
    except Exception as exc:  # noqa: BLE001
        outcome, error = 'raised', f'{type(exc).__name__}: {exc}'
    after = _rho_snapshot(planning)
    checklist = report.get('rule_eleven_checklist') or {}
    nets = [('TSO', planning.transmission_network.params)] + [
        (f'DSO{n}', dn.params) for n, dn in sorted(planning.distribution_networks.items())]
    return {'case': case, 'eval_id': eval_id, 'hook_kwargs': hook_kwargs, 'mutation': mutation,
            'outcome': outcome, 'error': error, 'case_file_rho': case_file_rho, 'rho_before_hook': before,
            'rho_after_hook': after,
            's44_configuration_checks': checklist.get('s44_campaign_configuration_checks'),
            'w183a_rho0_scale': checklist.get('w183a_rho0_scale'),
            'w183a_nrf_option': checklist.get('w183a_nrf_option'),
            'holder_rho0_scale_applied': holder.get('rho0_scale_applied') is not None,
            'no_reverse_flow_after_hook': {n: getattr(p, 'no_reverse_flow', '<absent>') for n, p in nets}}


def _mutate(group, network, factor):
    def m(planning):
        old = planning.params.admm.rho[group][network]
        planning.params.admm.rho[group][network] = old * factor
        return {'group': group, 'network': network, 'old': old, 'new': planning.params.admm.rho[group][network]}
    return m


def _all_equal(snapshot, target):
    vals = [(g, n, v) for g in ('v', 'pf', 'ess') for n, v in snapshot[g].items()]
    bad = [(g, n, v, target[g]) for g, n, v in vals if not (isinstance(v, float) and v == target[g])]
    return len(vals), bad


def t3_t4_hook(G, ctx):
    S39 = {'v': G.S39_RHO_V, 'pf': G.S39_RHO_PF, 'ess': G.S39_RHO_ESS}
    cases = {}
    cases['undeclared_case_file'] = run_hook_case(G, ctx, 'undeclared_case_file')
    cases['undeclared_mutated_dso_pf'] = run_hook_case(G, ctx, 'undeclared_mutated_dso_pf',
                                                       mutate=_mutate('pf', 'case33_2', 1.01))
    cases['undeclared_mutated_esso'] = run_hook_case(G, ctx, 'undeclared_mutated_esso',
                                                     mutate=_mutate('ess', 'esso', 2.0))
    for s, tag in ((0.5, '0p5'), (2.0, '2p0'), (1.0, '1p0_explicit')):
        cases[f'declared_{tag}'] = run_hook_case(G, ctx, f'declared_{tag}', rho0_scale=s)
    cases['declared_0p5_mutated_tso_v'] = run_hook_case(G, ctx, 'declared_0p5_mutated_tso_v',
                                                        mutate=_mutate('v', 'case9', 1.5), rho0_scale=0.5)
    t3 = {}
    c = cases['undeclared_case_file']
    n, bad = _all_equal(c['rho_after_hook'], S39)
    t3['undeclared_case_file_passes_unchanged'] = (c['outcome'] == 'passed' and not bad and n == 13
                                                   and c['rho_after_hook'] == c['case_file_rho']
                                                   and c['w183a_rho0_scale'] is None)
    for name, check in (('undeclared_mutated_dso_pf', 'rho_pf_matches_D'),
                        ('undeclared_mutated_esso', 'rho_ess_matches_D'),
                        ('declared_0p5_mutated_tso_v', 'rho_v_matches_D')):
        c = cases[name]
        others = {'rho_v_matches_D', 'rho_pf_matches_D', 'rho_ess_matches_D'} - {check}
        t3[f'{name}_raises_naming_{check}'] = (c['outcome'] == 'raised' and c['error'].startswith('RuntimeError')
                                               and check in c['error']
                                               and not any(o in c['error'] for o in others)
                                               and c['rho_after_hook'] == c['rho_before_hook']
                                               and c['w183a_rho0_scale'] is None)
    counts = {}
    for tag, s in (('0p5', 0.5), ('2p0', 2.0), ('1p0_explicit', 1.0)):
        c = cases[f'declared_{tag}']
        target = {g: S39[g] * s for g in S39}
        n, bad = _all_equal(c['rho_after_hook'], target)
        rec = c['w183a_rho0_scale'] or {}
        counts[tag] = {'n_entries': n, 'n_not_equal': len(bad), 'target': target}
        t3[f'declared_{tag}_every_entry_equals_s39_times_s'] = (
            c['outcome'] == 'passed' and n == 13 and not bad and rec.get('n_entries') == 13
            and rec.get('n_mismatches') == 0 and all((rec.get('checks') or {}).values())
            and rec.get('before') == c['case_file_rho']
            and all((c['s44_configuration_checks'] or {}).get(k) for k in ('rho_v_matches_D', 'rho_pf_matches_D',
                                                                            'rho_ess_matches_D')))
    t3_pass = all(t3.values())

    # T4: production has no NRF support before W183c
    import network_parameters as NP
    marker_absent = not hasattr(NP.NetworkParameters(), H.NRF_SUPPORT_ATTRIBUTE)
    cases['nrf_true_unsupported'] = run_hook_case(G, ctx, 'nrf_true_unsupported', nrf_option=True)
    cases['nrf_false_unsupported'] = run_hook_case(G, ctx, 'nrf_false_unsupported', nrf_option=False)
    c = cases['nrf_true_unsupported']
    t4 = {'production_marker_absent': marker_absent,
          'nrf_true_raises_naming_missing_support': (
              c['outcome'] == 'raised' and c['error'].startswith('RuntimeError')
              and 'no-reverse-flow support' in c['error'] and H.NRF_SUPPORT_ATTRIBUTE in c['error']),
          'nrf_true_sets_nothing': all(v == '<absent>' for v in c['no_reverse_flow_after_hook'].values()),
          'nrf_false_passes': cases['nrf_false_unsupported']['outcome'] == 'passed'}
    t4_pass = all(t4.values())

    # T4b: SIMULATED support (test-only; restored afterwards)
    original_support = H.nrf_option_production_support
    had_class_attr = hasattr(NP.NetworkParameters, H.NRF_SUPPORT_ATTRIBUTE)
    try:
        setattr(NP.NetworkParameters, H.NRF_SUPPORT_ATTRIBUTE, False)
        H.nrf_option_production_support = lambda: (True, 'SIMULATED support marker (W183a test-only monkeypatch)')

        def _flag_on_dso(planning):
            node = sorted(planning.distribution_networks)[0]
            setattr(planning.distribution_networks[node].params, H.NRF_SUPPORT_ATTRIBUTE, True)
            return {'dso_flag_set_before_hook': node}
        cases['sim_nrf_true'] = run_hook_case(G, ctx, 'sim_nrf_true', nrf_option=True)
        cases['sim_undeclared_flag_on'] = run_hook_case(G, ctx, 'sim_undeclared_flag_on', mutate=_flag_on_dso)
        cases['sim_undeclared_off'] = run_hook_case(G, ctx, 'sim_undeclared_off')
    finally:
        H.nrf_option_production_support = original_support
        if not had_class_attr:
            delattr(NP.NetworkParameters, H.NRF_SUPPORT_ATTRIBUTE)
    restored = (H.nrf_option_production_support is original_support
                and hasattr(NP.NetworkParameters(), H.NRF_SUPPORT_ATTRIBUTE) == had_class_attr)
    c = cases['sim_nrf_true']
    rec = c['w183a_nrf_option'] or {}
    t4b = {'sim_nrf_true_sets_tso_and_every_dso': (c['outcome'] == 'passed'
                                                   and set(c['no_reverse_flow_after_hook']) == {'TSO', 'DSO5', 'DSO7',
                                                                                                 'DSO9'}
                                                   and all(v is True for v in c['no_reverse_flow_after_hook'].values())
                                                   and all((rec.get('checks') or {}).values())),
           'sim_undeclared_flag_on_raises': (cases['sim_undeclared_flag_on']['outcome'] == 'raised'
                                             and 'every_network_flag_is_off' in cases['sim_undeclared_flag_on']['error']),
           'sim_undeclared_off_passes': cases['sim_undeclared_off']['outcome'] == 'passed',
           'monkeypatch_restored': restored}
    t4b_pass = all(t4b.values())
    return ({'pass': t3_pass, 'checks': t3, 'counts': counts},
            {'pass': t4_pass, 'checks': t4},
            {'pass': t4b_pass, 'checks': t4b,
             'note': 'SIMULATED production support (test-only monkeypatch of the marker and a class attribute on '
                     'network_parameters.NetworkParameters, restored afterwards); exercises the hook path W183c '
                     'enables; says nothing about production'},
            cases)


# ======================================================================================================================
#  T5
# ======================================================================================================================
def t5_nl_identity(args, t5_out):
    os.makedirs(t5_out, exist_ok=True)
    present = sorted(os.listdir(t5_out))
    if present:
        raise RuntimeError(f'T5 output directory is not empty: {t5_out}: {present[:5]}')
    cmd = [sys.executable, '-u', os.path.join(REPO, NL_DIGEST_REL), '--out', t5_out, '--label', 'post_w183a',
           '--scratch', os.path.join(args.scratch, 'w183a_nl_build'), '--configs', ','.join(T5_CONFIGS),
           '--no-determinism']
    if args.expect_production_commit:
        cmd += ['--expect-production-commit', args.expect_production_commit]
    log_path = os.path.join(t5_out, 'launch.log')
    _log(f'T5: {cmd} (stdout + stderr -> {os.path.relpath(log_path, REPO)})')
    t0 = time.time()
    with open(log_path, 'w') as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ)).returncode
    wall = time.time() - t0
    post_json = os.path.join(t5_out, 'w183_nl_digest.json')
    out = {'command': cmd, 'returncode': rc, 'wall_s': wall, 'launch_log': os.path.relpath(log_path, REPO)}
    if not os.path.isfile(post_json):
        out['pass'] = False
        out['error'] = 'no result JSON'
        return out
    raw = open(post_json, 'rb').read()
    gz_path = post_json + '.gz'
    with open(gz_path, 'wb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    gz_roundtrip = _sha_bytes(gzip.open(gz_path).read()) == _sha_bytes(raw)
    post = json.loads(raw)
    ref_gz = os.path.join(REPO, REFERENCE_DIR_REL, 'w183_nl_digest.json.gz')
    ref = json.load(gzip.open(ref_gz))
    per = {}
    n_blocks = n_identical = 0
    for c in T5_CONFIGS:
        per[c] = {}
        for b in ('sequential', 'parallel'):
            ra, rb = ref['records'][c][b]['blocks'], post['records'][c][b]['blocks']
            diffs, items_gz_diff = [], []
            for key in sorted(set(ra) | set(rb)):
                a, bb = ra.get(key), rb.get(key)
                if a is None or bb is None:
                    diffs.append({'block': key, 'missing_in': 'reference' if a is None else 'post'})
                    continue
                files = {'plain': (a['nl']['plain'] or {}).get('sha256') == (bb['nl']['plain'] or {}).get('sha256')}
                for f in ('nl', 'row', 'col'):
                    files[f'symbolic_{f}'] = ((a['nl']['symbolic'] or {}).get(f, {}).get('sha256')
                                              == (bb['nl']['symbolic'] or {}).get(f, {}).get('sha256'))
                files['plain_sha256_not_null'] = (a['nl']['plain'] or {}).get('sha256') is not None
                files['block_digest'] = a['sha256'] == bb['sha256']
                files['component_digests'] = a['components'] == bb['components']
                n_blocks += 1
                if all(files.values()):
                    n_identical += 1
                else:
                    diffs.append({'block': key, 'files': files})
                if a.get('items_gz_sha256') != bb.get('items_gz_sha256'):
                    items_gz_diff.append(key)
            per[c][b] = {'n_blocks_reference': len(ra), 'n_blocks_post': len(rb), 'differences': diffs,
                         'byte_identical': not diffs, 'items_gz_sha256_differs_on': items_gz_diff}
    out.update({
        'post_json_sha256': _sha_bytes(raw), 'post_json_bytes': len(raw),
        'post_json_gz': os.path.relpath(gz_path, REPO), 'post_json_gz_sha256': _sha_file(gz_path),
        'post_json_gz_roundtrip_equal': gz_roundtrip,
        'reference_gz': os.path.relpath(ref_gz, REPO), 'reference_gz_sha256': _sha_file(ref_gz),
        'reference_gz_git_status': _git_status([os.path.relpath(ref_gz, REPO)]) or 'tracked, clean',
        'reference_git_head_at_run': ref.get('git_head_at_run'), 'post_git_head_at_run': post.get('git_head_at_run'),
        'post_all_ok': post.get('all_ok'), 'post_checks': post.get('checks'),
        'post_solve_profile_guard': post.get('solve_profile_guard'),
        'post_harness_sha256_imported': (post.get('repo_modules_imported_sha256_start') or {}).get(HARNESS_REL),
        'reference_harness_sha256_imported': (ref.get('repo_modules_imported_sha256_start') or {}).get(HARNESS_REL),
        'eval_key_base_equal': {c: ref['records'][c]['eval_key_base'] == post['records'][c]['eval_key_base']
                                for c in T5_CONFIGS},
        'n_blocks_compared': n_blocks, 'n_blocks_identical': n_identical, 'per_config': per,
    })
    guard = post.get('solve_profile_guard') or {}
    out['pass'] = (rc == 0 and post.get('all_ok') is True and gz_roundtrip and n_blocks > 0
                   and n_blocks == n_identical
                   and all(v['byte_identical'] for c in per.values() for v in c.values())
                   and all(out['eval_key_base_equal'].values())
                   and not guard.get('verify_0_failures') and guard.get('counts', {}).get('blocked_solve') == 0
                   and out['post_harness_sha256_imported'] != out['reference_harness_sha256_imported'])
    manifest = {}
    for root, _dirs, files in os.walk(t5_out):
        for fname in sorted(files):
            p = os.path.join(root, fname)
            if p == os.path.join(t5_out, 'manifest_post_run_sha256.json'):
                continue
            if fname in ('w183_nl_digest.json', 'w183_nl_digest.json.gz', 'launch.log', 'manifest_sha256.json'):
                manifest[os.path.relpath(p, REPO)] = _sha_file(p)
    with open(os.path.join(t5_out, 'manifest_post_run_sha256.json'), 'w') as handle:
        GRIO.dump(dict(sorted(manifest.items())), handle, indent=1, sort_keys=True)
    out['manifest_post_run'] = manifest
    return out


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--t5-out', default=None, help='T5 output dir (fresh); omit with --skip-t5')
    parser.add_argument('--scratch', required=True, help='scratch dir outside the repo; must not exist')
    parser.add_argument('--expect-production-commit', default=None)
    parser.add_argument('--skip-t5', action='store_true', help='smoke runs only')
    parser.add_argument('--allow-uncommitted', action='store_true', help='smoke runs only')
    args = parser.parse_args()
    started = time.time()
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    t5_out = None
    if not args.skip_t5:
        if not args.t5_out:
            raise SystemExit('--t5-out is required unless --skip-t5')
        t5_out = args.t5_out if os.path.isabs(args.t5_out) else os.path.join(REPO, args.t5_out)
    failures = []
    if os.path.exists(out) and [p for p in os.listdir(out) if p != 'launch.log']:
        failures.append(f'output directory exists and is not empty (write-once): {out}')
    if t5_out and os.path.exists(t5_out):
        failures.append(f'T5 output directory exists: {t5_out}')
    if os.path.exists(args.scratch):
        failures.append(f'scratch dir exists: {args.scratch}')
    if os.path.abspath(args.scratch).startswith(REPO + os.sep):
        failures.append(f'scratch dir must be outside the repo: {args.scratch}')
    if os.path.exists(H.CAMPAIGN_LOCK_PATH):
        failures.append(f'campaign lock present: {H.CAMPAIGN_LOCK_PATH}')
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
              if (SCRIPT_REL in cmd or NL_DIGEST_REL in cmd) and p not in ancestors]
    if others:
        failures.append(f'another copy is running (refusing to run concurrently): {others}')
    status = _git_status([SCRIPT_REL, HARNESS_REL])
    tracked = {f: bool(_git(['ls-files', f]).strip()) for f in (SCRIPT_REL, HARNESS_REL)}
    if not args.allow_uncommitted and (status or not all(tracked.values())):
        failures.append(f'{SCRIPT_REL} / {HARNESS_REL} not committed and clean: {status} {tracked}')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.scratch)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    head = _git(['rev-parse', 'HEAD']).strip()
    _log(f'{STAGE}; HEAD {head}; harness sha256 {_sha_file(os.path.join(REPO, HARNESS_REL))}')
    result = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'run_tag': run_tag, 'argv': sys.argv,
              'git_head_at_run': head, 'script': SCRIPT_REL, 'script_sha256': _sha_file(__file__),
              'harness': HARNESS_REL, 'harness_sha256': _sha_file(os.path.join(REPO, HARNESS_REL)),
              'git_status_at_start': status or 'clean', 'interpreter': sys.executable,
              'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH')}
    tests = {}
    try:
        H_pre, pre_info = load_pre_change_harness(args.scratch)
        result['pre_change_harness'] = pre_info
        _log(f'pre-change harness {pre_info["commit"][:12]} blob {pre_info["blob_sha256"][:16]} '
             f'(rho0_scale kwarg: {pre_info["has_rho0_scale_kwarg"]})')
        tests['T1'] = t1_old_keys(H_pre)
        _log(f"T1: {tests['T1']['n_v2_specs']} v2 specs, {tests['T1']['n_entries']} entries; new == stored "
             f"{tests['T1']['n_new_equal_stored']}; new == pre-change {tests['T1']['n_new_equal_pre_change']}; "
             f"skips {tests['T1']['n_skips']}; errors {len(tests['T1']['errors'])}; PASS {tests['T1']['pass']}")
        ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
        declared = H.load_ess_ageing_parameters(ess_path)
        pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': _sha_file(ess_path)}
        tests['T2'] = t2_key_algebra(args.scratch, declared)
        _log(f"T2: PASS {tests['T2']['pass']}")
        import p515_g_g1_g4_admm_gates as G
        ctx = {'scratch': args.scratch, 'run_tag': run_tag,
               'spec': {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                                          'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                                          'ess_ageing_baseline': declared, 'ess_params_file': pin,
                                          'ess_ageing_baseline_label': 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)',
                                          'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL)},
                        'cap': CAP, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE}}
        result['t3_spec_like'] = ctx['spec']
        t3, t4, t4b, cases = t3_t4_hook(G, ctx)
        tests['T3'], tests['T4'], tests['T4b_simulated_support'] = t3, t4, t4b
        result['hook_cases'] = cases
        _log(f"T3: PASS {t3['pass']} {t3['checks']} counts {t3['counts']}")
        _log(f"T4: PASS {t4['pass']} {t4['checks']}; T4b (simulated): PASS {t4b['pass']} {t4b['checks']}")
        if t5_out:
            tests['T5'] = t5_nl_identity(args, t5_out)
            _log(f"T5: PASS {tests['T5']['pass']}; blocks {tests['T5'].get('n_blocks_identical')} / "
                 f"{tests['T5'].get('n_blocks_compared')} identical; rc {tests['T5']['returncode']}")
    except Exception as error:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        result['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': tb}
    guard_failures = GUARD.verify(0)
    result['tests'] = tests
    result['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
    result['pickle_counts'] = dict(PICKLE_COUNTS)
    result['pickle_blocked'] = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    expected = {'T1', 'T2', 'T3', 'T4', 'T4b_simulated_support'} | ({'T5'} if t5_out else set())
    result['summary'] = {name: t.get('pass') for name, t in tests.items()}
    result['all_pass'] = (set(tests) == expected and all(t.get('pass') is True for t in tests.values())
                          and not guard_failures and PICKLE_COUNTS == {'load': 0, 'loads': 0}
                          and 'error' not in result)
    result['wall_clock_s'] = time.time() - started
    json_path = os.path.join(out, 'w183a_option_tests.json')
    with open(json_path, 'w') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True)
    manifest = {os.path.relpath(json_path, REPO): _sha_file(json_path),
                SCRIPT_REL: _sha_file(__file__), HARNESS_REL: _sha_file(os.path.join(REPO, HARNESS_REL))}
    if 'T5' in tests:
        manifest.update(tests['T5'].get('manifest_post_run') or {})
        mp = os.path.join(t5_out, 'manifest_post_run_sha256.json')
        if os.path.isfile(mp):
            manifest[os.path.relpath(mp, REPO)] = _sha_file(mp)
    with open(os.path.join(out, 'manifest_sha256.json'), 'w') as handle:
        GRIO.dump(dict(sorted(manifest.items())), handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    _log(f"ALL_PASS={result['all_pass']} summary={result['summary']} guard={dict(GUARD.counts)} "
         f"pickle={PICKLE_COUNTS} wall={result['wall_clock_s']:.1f}s")
    sys.exit(0 if result['all_pass'] else 1)


if __name__ == '__main__':
    main()
