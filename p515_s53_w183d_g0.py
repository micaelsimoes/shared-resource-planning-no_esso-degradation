"""
P5.15 Addendum 83 ruling 3 (vii), W183 order item W183d -- GATE G0 on the final production head, ZERO SOLVES, NO BUILD.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (vii): "at x = 0, the ESSO and network NL files with
`ageing_enabled` on and off are byte-identical; and the combination `model_variant` + `flex_price_multiplier` applies
both (NL of the unit at 2x under N differs from the unit at 2x under A only in the deactivated/altered ageing rows; key
contains both). Prediction: identical at x = 0."

WHY NO BUILD. W183c's gate 2 (`e717850f`) rebuilt the eight configurations {x0, unit} x {A, N} x {1x, 2x}, both DSO
builders, ON THE FINAL PRODUCTION HEAD `6f43479e` with `p515_s53_w183_nl_digest.py` into
`data/SRP1/Results/P515S55/w183_nl_post_w183c_off/` (gz JSON, manifests and ESSO NL gz committed; network NL gz and
per-block item gz on disk, hash-recorded in `manifest_sha256.json`). G0 is therefore read off that artifact; nothing
is rebuilt here.

WHAT IS CHECKED (every check mechanical; G0_PASS = all gating checks true)
  1 SOURCE. The source gz hashes to its pin and is git-tracked and clean; its decompression equals the raw JSON the
    manifests record; every file of `manifest_sha256.json` (main run and determinism child) and of
    `manifest_post_run_sha256.json` exists and hashes as recorded; the artifact's own checks / all_ok / guard (0) hold;
    its recorded production modules equal the CURRENT HEAD's (per module) and the current production is clean and its
    last commit is the expected one; every recorded raw NL sha256 of the G0 configurations equals the sha256 of the
    decompressed gz on disk (ties the hash-recorded network NL files to the identities the JSON states); the item gz
    files hash as the records state; per configuration and builder the hook read-backs: w33 (flex) true everywhere,
    w21 (ageing baseline) read-back true under A and its file/pin checks true under N, w20 (model variant) read-back
    true under N and absent under A (absent = the baseline in the harness).
  2a x = 0, A vs N, at 1x and 2x (recomputed with the digest harness's own `compare_blocks`, then checked equal to the
    stored comparison): ESSO NL (plain, labelled, .row, .col) identical per node; network NL (all four files) and
    component digests identical on all 48 sequential + 36 parallel blocks; ESSO component digests differ only in
    `Constraint:energy_storage_capacity_degradation`, every differing index inactive in both.
  2b unit at 2x, N vs A: network NL identical on all blocks of both builders; ESSO NL differs on node 7 only; node 7
    `.col` identical; the rows with differing segments (the digest harness's own `nl_row_analysis`, on the committed
    ESSO NL gz) are a subset of `energy_storage_capacity_degradation[*]` and equal the expected six [1],[2],[4],[5],
    [7],[8]; component digests differ exactly in {'Constraint:energy_storage_capacity_degradation'} (network: none).
    unit at 1x: the same quantities, INFORMATIVE (not gating).
  2c keys: recomputed with `p515_s44_campaign_harness.evaluation_key` (the digest harness's call): base, model_variant
    only, flex only, combined -- pairwise distinct (combined != each single != base), each equal to the artifact's
    recorded key, for x = 0 and the unit; and in ONE hook record of `unit_N_2x` (each builder) the w20 and w33
    read-backs are both true, with the declared variant and m = 2.0.

GUARDS. `SolveProfileGuard(permitted=())` installed before any other project import, verified at exactly 0 (and the
digest harness's import-time guard, also 0); `pickle.load` / `pickle.loads` blocked and counted (0); a
`pyomo.environ.ConcreteModel.__init__` tripwire counts model builds (must be 0).

OUTPUT (write-once; the directory may pre-exist holding only `launch.log`):
  <out>/g0.json                 the G0 artifact (schema `w183d_g0_v1`; written through `gate_result_io`)
  <out>/manifest_sha256.json    sha256 of g0.json

THE LAUNCHER. `p515_s55_e1_launcher.g0_check` reads the digest-run schema (`summary.i_x0_A_2x_vs_x0_N_2x`,
`summary.eval_keys`, `configuration`, `production.modules_sha256`), not this one. g0.json therefore names the file the
launcher must be pointed at (`launcher_g0_artifact`): the source digest gz and its sha256, whose G0 content this
artifact verifies.

LAUNCH (attached, alone, both streams captured; this script committed first):
  mkdir data/SRP1/Results/P515S55/w183d_g0 && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183d_g0.py \\
      --out data/SRP1/Results/P515S55/w183d_g0 --expect-production-commit 6f43479e \\
      > data/SRP1/Results/P515S55/w183d_g0/launch.log 2>&1
Exit 0 when G0_PASS, 1 otherwise.
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183d G0 (zero solves, no build)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183d G0: pickle.load called -- nothing is unpickled')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183d G0: pickle.loads called -- nothing is unpickled')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError('W183d G0: a pyomo ConcreteModel was constructed -- no model build')


pe.ConcreteModel.__init__ = _blocked_concrete_init

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w183_nl_digest as DG  # noqa: E402  (its import installs its own zero-solve guard; verified below)

pickle.load, pickle.loads = _blocked_load, _blocked_loads   # re-assert after the imports
os.chdir(REPO)

SCRIPT_REL = os.path.basename(__file__)
SCHEMA = 'w183d_g0_v1'
STAGE = 'P5.15 Addendum 83 ruling 3 (vii), W183d -- gate G0 on the final production head (zero solves, no build)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (vii); W183 order item W183d',
             'P5_15_W183_0_NL_REFERENCE.md', 'P5_15_W183C_NRF.md (gate 2: the off run on 6f43479e)',
             'P5_15_W183B_E1_LAUNCHER.md (Blocked 4: g0_check needs a G0 artifact on the current production)']
SOURCE_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183_nl_post_w183c_off')
SOURCE_GZ_REL = os.path.join(SOURCE_DIR_REL, 'w183_nl_digest.json.gz')
SOURCE_GZ_SHA256 = 'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a'
SOURCE_RAW_SHA256 = '69c11d07ba40516c4e60f26a04eca93d864bb3ad6b46188112fd528fbb3be821'
DEGRADATION = 'Constraint:energy_storage_capacity_degradation'
EXPECTED_UNIT_ROWS = sorted(f'energy_storage_capacity_degradation[{i}]' for i in (1, 2, 4, 5, 7, 8))
ESSO_NODES = ('ESSO|node5', 'ESSO|node7', 'ESSO|node9')
G0_CONFIGS = ('x0_A_1x', 'x0_N_1x', 'x0_A_2x', 'x0_N_2x', 'unit_A_1x', 'unit_N_1x', 'unit_A_2x', 'unit_N_2x')
BUILDERS = ('sequential', 'parallel')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_gunzip(path):
    h = hashlib.sha256()
    with gzip.open(os.path.join(REPO, path), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _tracked_clean(path):
    tracked = bool(_git(['ls-files', '--', path]).strip())
    clean = not _git(['status', '--porcelain', '--', path]).strip()
    return tracked, clean


def verify_manifest(manifest_rel):
    with open(os.path.join(REPO, manifest_rel)) as handle:
        manifest = json.load(handle)
    missing, mismatched = [], []
    for rel, sha in sorted(manifest.items()):
        if not os.path.isfile(os.path.join(REPO, rel)):
            missing.append(rel)
        elif _sha(rel) != sha:
            mismatched.append(rel)
    tracked = set(_git(['ls-files', '--', os.path.dirname(manifest_rel)]).splitlines())
    return {'manifest': manifest_rel, 'manifest_sha256': _sha(manifest_rel), 'n_files': len(manifest),
            'n_git_tracked': sum(1 for r in manifest if r in tracked),
            'n_untracked_hash_recorded': sum(1 for r in manifest if r not in tracked),
            'missing': missing, 'mismatched': mismatched, 'all_hold': not missing and not mismatched}


def production_now():
    mods = sorted(p for p in _git(['ls-files', '*.py']).splitlines()
                  if p.strip() and '/' not in p and not re.match(r'^p\d', p))
    status = _git(['status', '--porcelain', '--'] + mods).strip()
    return mods, {m: _sha(m) for m in mods}, status, _git(['log', '-1', '--format=%H', '--'] + mods).strip()


# ======================================================================================================================
#  1 source
# ======================================================================================================================
def check_source(d, expect_commit):
    out, checks = {}, {}
    out['gz'] = {'path': SOURCE_GZ_REL, 'sha256_pinned': SOURCE_GZ_SHA256, 'sha256_on_disk': _sha(SOURCE_GZ_REL)}
    out['gz']['git_tracked'], out['gz']['git_clean'] = _tracked_clean(SOURCE_GZ_REL)
    checks['source_gz_sha256_equals_pin'] = out['gz']['sha256_on_disk'] == SOURCE_GZ_SHA256
    checks['source_gz_git_tracked_and_clean'] = out['gz']['git_tracked'] and out['gz']['git_clean']
    out['gz']['decompressed_sha256'] = _sha_gunzip(SOURCE_GZ_REL)
    checks['source_decompressed_equals_recorded_raw_sha256'] = out['gz']['decompressed_sha256'] == SOURCE_RAW_SHA256
    # manifests
    out['manifests'] = {
        'main': verify_manifest(os.path.join(SOURCE_DIR_REL, 'manifest_sha256.json')),
        'determinism_child': verify_manifest(os.path.join(SOURCE_DIR_REL, 'determinism_child', 'manifest_sha256.json')),
        'post_run': verify_manifest(os.path.join(SOURCE_DIR_REL, 'manifest_post_run_sha256.json')),
    }
    for k, v in out['manifests'].items():
        checks[f'manifest_{k}_hashes_hold'] = v['all_hold']
    raw_rel = os.path.join(SOURCE_DIR_REL, 'w183_nl_digest.json')
    with open(os.path.join(REPO, out['manifests']['main']['manifest'])) as handle:
        checks['main_manifest_records_the_raw_json_sha256'] = json.load(handle).get(raw_rel) == SOURCE_RAW_SHA256
    # the artifact's own checks
    out['artifact_checks'] = d.get('checks')
    out['artifact_all_ok'] = d.get('all_ok')
    out['artifact_guard'] = d.get('solve_profile_guard')
    checks['artifact_all_ok'] = d.get('all_ok') is True and all(v is True for v in (d.get('checks') or {}).values())
    g = (d.get('solve_profile_guard') or {}).get('counts') or {}
    checks['artifact_guard_zero'] = bool(g) and all(v == 0 for v in g.values()) and not (
        d.get('solve_profile_guard') or {}).get('verify_0_failures')
    checks['artifact_determinism_byte_identical'] = (d.get('determinism') or {}).get('byte_identical_all') is True
    checks['artifact_nrf_option_off'] = (d.get('configuration') or {}).get('nrf_option') is None
    checks['artifact_configs_are_the_eight'] = sorted((d.get('configuration') or {}).get('configs') or []) == sorted(
        G0_CONFIGS)
    # production
    mods, now, status, last = production_now()
    art = (d.get('production') or {}).get('modules_sha256') or {}
    per_module = {m: {'artifact': art.get(m), 'current': now.get(m), 'equal': art.get(m) == now.get(m)}
                  for m in sorted(set(art) | set(now))}
    expect_full = _git(['rev-parse', expect_commit]).strip()
    diff_rc = subprocess.run(['git', 'diff', '--quiet', expect_full, '--'] + mods, cwd=REPO).returncode
    out['production'] = {'definition': 'tracked root-level *.py not matching ^p\\d', 'n_modules_current': len(now),
                         'n_modules_artifact': len(art), 'per_module': per_module,
                         'artifact_expected_commit': (d.get('production') or {}).get('expected_commit'),
                         'artifact_git_head_at_run': d.get('git_head_at_run'),
                         'artifact_last_commit_touching': (d.get('production') or {}).get('last_commit_touching'),
                         'current_last_commit_touching': last, 'current_git_status': status or 'clean',
                         'expected_commit': expect_full, 'git_head_now': _git(['rev-parse', 'HEAD']).strip()}
    checks['artifact_production_modules_equal_current_every_module'] = bool(per_module) and all(
        v['equal'] for v in per_module.values())
    checks['current_production_clean'] = not status
    checks['current_production_equals_expected_commit'] = diff_rc == 0
    checks['current_production_last_commit_is_expected'] = last == expect_full
    checks['artifact_expected_commit_is_expected'] = out['production']['artifact_expected_commit'] == expect_full
    # raw NL sha256 of the decompressed gz files equal the JSON's recorded sha256 (every G0 configuration)
    n_files, bad = 0, []
    item_bad = []
    for c in G0_CONFIGS:
        for b in BUILDERS:
            for key, blk in d['records'][c][b]['blocks'].items():
                nl = blk['nl']
                files = [nl['plain']] + [nl['symbolic'][f] for f in ('nl', 'row', 'col')]
                for f in files:
                    n_files += 1
                    if _sha_gunzip(f['gz']) != f['sha256']:
                        bad.append(f['gz'])
                if _sha(blk['items_gz']) != blk['items_gz_sha256']:
                    item_bad.append(blk['items_gz'])
    out['nl_raw_sha256_tie'] = {'n_files_checked': n_files, 'mismatched': bad,
                                'note': ('each NL gz on disk (ESSO committed; network untracked, hash-recorded) '
                                         'decompressed and hashed, equal to the raw sha256 the JSON records')}
    out['item_gz_tie'] = {'mismatched': item_bad}
    checks['nl_gz_decompressed_equals_recorded_raw_sha256_every_g0_file'] = n_files > 0 and not bad
    checks['item_gz_sha256_equal_records'] = not item_bad
    # read-backs
    rb_table, rb_ok = {}, {}
    for c in G0_CONFIGS:
        model = d['records'][c]['model']
        for b in BUILDERS:
            rb = d['records'][c][b]['readbacks']
            w20, w21, w33 = rb.get('w20_model_variant'), rb.get('w21_ess_ageing_baseline'), rb.get(
                'w33_flex_price_multiplier')
            row = {'w20_model_variant_readback_all_match': (w20 or {}).get('readback_all_match') if w20 else None,
                   'w20_present': w20 is not None,
                   'w21_ess_ageing_baseline_readback_all_match': (w21 or {}).get('readback_all_match'),
                   'w21_checks_all_true': bool(w21) and all(v is True for v in (w21.get('checks') or {}).values()),
                   'w33_flex_price_multiplier_readback_all_match': (w33 or {}).get('readback_all_match'),
                   'w33_multiplier': (w33 or {}).get('flex_price_multiplier'),
                   'harness_readback_checks_all_true': all(
                       v is True for v in d['records'][c][b]['readback_checks'].values())}
            if model == 'A':
                ok = (row['w20_present'] is False and row['w21_ess_ageing_baseline_readback_all_match'] is True
                      and row['w21_checks_all_true'])
            else:
                ok = row['w20_model_variant_readback_all_match'] is True and row['w21_checks_all_true']
            ok = ok and row['w33_flex_price_multiplier_readback_all_match'] is True and row[
                'harness_readback_checks_all_true'] and row['w33_multiplier'] == d['records'][c]['flex_price_multiplier']
            row['ok'] = ok
            rb_table[f'{c}|{b}'] = row
            rb_ok[f'{c}|{b}'] = ok
    out['readbacks'] = {'rule': ('w33 read-back true everywhere (multiplier as configured); A: w21 read-back true and '
                                 'w20 absent; N: w20 read-back true and the w21 file/pin checks true (the hook runs '
                                 'the w21 baseline read-back only when model_variant is None, '
                                 'p515_s44_campaign_harness `_config_hook_factory`, so it is null under N); the '
                                 "digest harness's own readback_checks all true"),
                        'per_config_builder': rb_table}
    checks['readbacks_w20_w21_w33_as_defined_every_config_and_builder'] = all(rb_ok.values())
    return out, checks


# ======================================================================================================================
#  2 G0 checks
# ======================================================================================================================
def _network_keys(blocks):
    return [k for k in blocks if not k.startswith('ESSO')]


def check_x0(d, flex):
    ca, cb = f'x0_A_{flex}', f'x0_N_{flex}'
    ra, rb = d['records'][ca], d['records'][cb]
    comp = {b: DG.compare_blocks(ra, rb, b) for b in BUILDERS}
    stored = d['comparisons'][f'i_{ca}_vs_{cb}']
    seq, par = comp['sequential']['per_block'], comp['parallel']['per_block']
    out = {'pair': [ca, cb], 'n_blocks': {b: comp[b]['blocks_compared'] for b in BUILDERS}}
    checks = {
        'recomputed_comparison_equals_stored': all(comp[b] == stored[b] for b in BUILDERS),
        'esso_nodes_present': sorted(k for k in seq if k.startswith('ESSO')) == list(ESSO_NODES),
        'esso_nl_four_files_identical_every_node': all(all(seq[k]['nl_identical'].values()) for k in ESSO_NODES),
        'network_blocks_48_sequential_36_parallel': (len(_network_keys(seq)) == 48 and len(_network_keys(par)) == 36
                                                     and not comp['sequential']['blocks_missing']
                                                     and not comp['parallel']['blocks_missing']),
        'network_nl_four_files_identical_every_block_both_builders': all(
            all(blocks[k]['nl_identical'].values()) for blocks in (seq, par) for k in _network_keys(blocks)),
        'network_component_digests_identical_every_block_both_builders': all(
            not blocks[k]['components_differing'] and blocks[k]['block_digest_identical']
            for blocks in (seq, par) for k in _network_keys(blocks)),
        'esso_components_differ_only_in_degradation': all(
            set(seq[k]['components_differing']) <= {DEGRADATION} for k in ESSO_NODES),
        'esso_every_differing_degradation_index_inactive_in_both': all(
            (seq[k].get('component_detail') or {}).get(DEGRADATION, {}).get('every_differing_index_inactive_in_both',
                                                                             True) is True
            and (seq[k].get('component_detail') or {}).get(DEGRADATION, {}).get('n_differing_active_in_a_or_b', 0) == 0
            for k in ESSO_NODES),
    }
    out['esso'] = {k: {'nl_identical': seq[k]['nl_identical'], 'plain_sha256_A': ra['sequential']['blocks'][k]['nl'][
        'plain']['sha256'], 'plain_sha256_N': rb['sequential']['blocks'][k]['nl']['plain']['sha256'],
                       'components_differing': seq[k]['components_differing'],
                       'degradation_n_items': (seq[k].get('component_detail') or {}).get(DEGRADATION, {}).get(
                           'n_items_a'),
                       'degradation_n_differing': (seq[k].get('component_detail') or {}).get(DEGRADATION, {}).get(
                           'n_differing'),
                       'degradation_n_differing_active': (seq[k].get('component_detail') or {}).get(
                           DEGRADATION, {}).get('n_differing_active_in_a_or_b')} for k in ESSO_NODES}
    out['network_components_differing_union'] = {b: comp[b]['components_differing_union'] for b in BUILDERS}
    return out, checks


def check_unit(d, flex):
    ca, cb = f'unit_A_{flex}', f'unit_N_{flex}'
    ra, rb = d['records'][ca], d['records'][cb]
    comp = {b: DG.compare_blocks(ra, rb, b) for b in BUILDERS}
    stored = d['comparisons'][f'ii_{ca}_vs_{cb}']
    seq, par = comp['sequential']['per_block'], comp['parallel']['per_block']
    row = DG.nl_row_analysis(ra, rb, 'sequential', 'ESSO|node7')
    rows = sorted(row['rows_with_differing_segments'])
    esso_nl_diff = sorted(k for k in ESSO_NODES if not seq[k]['nl_identical']['plain'])
    esso_comp_union = sorted({c for k in ESSO_NODES for c in seq[k]['components_differing']})
    checks = {
        'recomputed_comparison_equals_stored': all(comp[b] == stored[b] for b in BUILDERS),
        'recomputed_row_analysis_equals_stored': row == stored['esso_nl_row_analysis'].get('ESSO|node7'),
        'network_blocks_48_sequential_36_parallel': (len(_network_keys(seq)) == 48 and len(_network_keys(par)) == 36
                                                     and not comp['sequential']['blocks_missing']
                                                     and not comp['parallel']['blocks_missing']),
        'network_nl_four_files_identical_every_block_both_builders': all(
            all(blocks[k]['nl_identical'].values()) for blocks in (seq, par) for k in _network_keys(blocks)),
        'network_component_digests_identical_every_block_both_builders': all(
            not blocks[k]['components_differing'] for blocks in (seq, par) for k in _network_keys(blocks)),
        'esso_nl_differs_on_node7_only': esso_nl_diff == ['ESSO|node7'],
        'esso_nodes_5_9_nl_four_files_identical': all(all(seq[k]['nl_identical'].values())
                                                      for k in ('ESSO|node5', 'ESSO|node9')),
        'node7_col_identical': row['col_identical'] is True and seq['ESSO|node7']['nl_identical']['col'] is True,
        'node7_row_names_identical': row['row_names_identical'] is True,
        'node7_differing_rows_subset_of_degradation': bool(rows) and row[
            'every_differing_row_is_energy_storage_capacity_degradation'] is True,
        'node7_differing_rows_equal_expected_six': rows == EXPECTED_UNIT_ROWS,
        'esso_component_digests_differ_exactly_in_degradation': esso_comp_union == [DEGRADATION] and all(
            seq[k]['components_differing'] == [DEGRADATION] for k in ESSO_NODES),
    }
    out = {'pair': [ca, cb], 'n_blocks': {b: comp[b]['blocks_compared'] for b in BUILDERS},
           'esso_nl_differing_nodes': esso_nl_diff,
           'esso_plain_sha256': {k: {'A': ra['sequential']['blocks'][k]['nl']['plain']['sha256'],
                                     'N': rb['sequential']['blocks'][k]['nl']['plain']['sha256']} for k in ESSO_NODES},
           'esso_component_digests_differing_union': esso_comp_union,
           'esso_degradation_detail': {k: {f: (seq[k].get('component_detail') or {}).get(DEGRADATION, {}).get(f)
                                           for f in ('n_items_a', 'n_differing', 'n_differing_active_in_a_or_b')}
                                       for k in ESSO_NODES},
           'node7_row_analysis': {k: row[k] for k in (
               'col_identical', 'row_names_identical', 'n_rows_a', 'n_rows_b', 'n_cols_a', 'n_cols_b',
               'rows_with_differing_segments', 'segments_differing_by_row', 'non_row_segments_differing',
               'header_lines_differing', 'k_segment_detail')},
           'expected_rows': EXPECTED_UNIT_ROWS}
    return out, checks


def check_keys(d):
    declared = H.load_ess_ageing_parameters(os.path.join(REPO, H.ESS_PARAMS_FILE_REL))
    spec = d['configuration']['spec_like']['configuration']
    no_ageing = d['configuration']['models']['N']
    out, checks = {'per_instance': {}}, {
        'declared_ageing_equals_artifact': declared == spec['ess_ageing_baseline'],
        'ess_params_sha256_equals_artifact_pin': _sha(H.ESS_PARAMS_FILE_REL) == spec['ess_params_file']['sha256'],
        'artifact_model_N_equals_digest_harness_declaration': no_ageing == DG.NO_AGEING,
    }
    variants = {'base': (None, 1.0), 'model_variant_only': (no_ageing, 1.0), 'flex_only': (None, 2.0),
                'combined': (no_ageing, 2.0)}
    labels = {'base': '{i}_A_1x', 'model_variant_only': '{i}_N_1x', 'flex_only': '{i}_A_2x', 'combined': '{i}_N_2x'}
    for inst in ('x0', 'unit'):
        canonical = H.canonical_candidate(DG.INSTANCES[inst], investment_year=DG.INVESTMENT_YEAR)
        ck = H.candidate_key(canonical)
        keys = {name: H.evaluation_key(ck, {}, case_file_aa=spec['case_file_anderson_acceleration'],
                                       model_variant=mv, ess_ageing_baseline=declared, flex_price_multiplier=m,
                                       convergence_depth_tail=spec['convergence_depth_tail'], nrf_option=None)
                for name, (mv, m) in variants.items()}
        recorded = {name: d['summary']['eval_keys'][labels[name].format(i=inst)] for name in variants}
        out['per_instance'][inst] = {'candidate_key': ck,
                                     'candidate_key_equals_record': ck == d['records'][f'{inst}_A_1x'][
                                         'candidate_key'],
                                     'keys': keys, 'recorded': recorded,
                                     'record_labels': {n: labels[n].format(i=inst) for n in variants}}
        checks[f'{inst}_candidate_key_equals_record'] = out['per_instance'][inst]['candidate_key_equals_record']
        checks[f'{inst}_recomputed_keys_equal_recorded'] = keys == recorded
        checks[f'{inst}_four_keys_pairwise_distinct'] = len(set(keys.values())) == 4
        checks[f'{inst}_combined_differs_from_each_single_and_base'] = all(
            keys['combined'] != keys[n] for n in ('model_variant_only', 'flex_only', 'base'))
        checks[f'{inst}_each_single_differs_from_base'] = all(
            keys[n] != keys['base'] for n in ('model_variant_only', 'flex_only'))
    hook = {}
    for b in BUILDERS:
        rb = d['records']['unit_N_2x'][b]['readbacks']
        w20, w33 = rb.get('w20_model_variant') or {}, rb.get('w33_flex_price_multiplier') or {}
        hook[b] = {'w20_readback_all_match': w20.get('readback_all_match'), 'w20_model_variant': w20.get('model_variant'),
                   'w33_readback_all_match': w33.get('readback_all_match'),
                   'w33_flex_price_multiplier': w33.get('flex_price_multiplier'),
                   'w33_n_flex_coefficients': w33.get('n_flex_coefficients'), 'w33_max_rel_dev': w33.get('max_rel_dev')}
        checks[f'unit_N_2x_{b}_one_hook_record_w20_and_w33_readbacks_true'] = (
            w20.get('readback_all_match') is True and w33.get('readback_all_match') is True
            and w20.get('model_variant') == no_ageing and w33.get('flex_price_multiplier') == 2.0)
    out['unit_N_2x_hook_record'] = hook
    out['note'] = ('the key is a sha256 of a payload; "key contains both" is shown by the combined key differing from '
                   'each single-option key (each option changes the key in the presence of the other)')
    return out, checks


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-production-commit', required=True)
    parser.add_argument('--allow-uncommitted', action='store_true',
                        help='smoke runs only (output outside the repository): do not refuse an uncommitted script')
    args = parser.parse_args()
    started = time.time()
    out_abs = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    failures = []
    if os.path.exists(out_abs) and [p for p in os.listdir(out_abs) if p != 'launch.log']:
        failures.append(f'output directory not empty (write-once): {out_abs}')
    self_tracked, self_clean = _tracked_clean(SCRIPT_REL)
    if not (self_tracked and self_clean) and not args.allow_uncommitted:
        failures.append(f'{SCRIPT_REL} is not committed and clean')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out_abs, exist_ok=True)
    head = _git(['rev-parse', 'HEAD']).strip()
    _log(f'{STAGE}; HEAD {head}; source {SOURCE_GZ_REL}')

    with gzip.open(os.path.join(REPO, SOURCE_GZ_REL), 'rt') as handle:
        d = json.load(handle)

    results, gating = {}, {}
    t0 = time.time()
    results['1_source'], c = check_source(d, args.expect_production_commit)
    gating['1_source'] = c
    _log(f'1 source ({time.time() - t0:.0f}s): {c}')
    for flex in ('1x', '2x'):
        results[f'2a_x0_{flex}'], c = check_x0(d, flex)
        gating[f'2a_x0_{flex}'] = c
        _log(f'2a x0 {flex}: {c}')
    results['2b_unit_2x'], c = check_unit(d, '2x')
    gating['2b_unit_2x'] = c
    _log(f'2b unit 2x: {c}')
    informative = {}
    informative['2b_unit_1x'], c = check_unit(d, '1x')
    informative['2b_unit_1x_checks'] = c
    _log(f'2b unit 1x (informative): {c}')
    results['2c_keys'], c = check_keys(d)
    gating['2c_keys'] = c
    _log(f'2c keys: {c}')

    guard_failures = GUARD.verify(0)
    dg_guard_failures = DG.GUARD.verify(0)
    guards = {'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
              'digest_harness_import_guard': {'counts': dict(DG.GUARD.counts), 'verify_0_failures': dg_guard_failures},
              'pickle': dict(PICKLE_COUNTS), 'model_build': dict(MODEL_BUILD_COUNTS)}
    gating['guards'] = {'solve_guard_zero': not guard_failures, 'digest_harness_guard_zero': not dg_guard_failures,
                        'pickle_zero': PICKLE_COUNTS == {'load': 0, 'loads': 0},
                        'model_build_zero': MODEL_BUILD_COUNTS['ConcreteModel.__init__'] == 0}
    per_check = {grp: all(v is True for v in c.values()) for grp, c in gating.items()}
    g0_pass = all(per_check.values())
    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'argv': sys.argv,
        'git_head_at_run': head, 'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
        'script_git_tracked_and_clean': self_tracked and self_clean, 'allow_uncommitted': args.allow_uncommitted,
        'production_head': results['1_source']['production']['current_last_commit_touching'],
        'expected_production_commit': results['1_source']['production']['expected_commit'],
        'source_artifact': {'path': SOURCE_GZ_REL, 'sha256': SOURCE_GZ_SHA256, 'raw_json_sha256': SOURCE_RAW_SHA256,
                            'what': ('the W183c gate-2 NL-digest run (option off) on 6f43479e, '
                                     'p515_s53_w183_nl_digest.py; committed in e717850f')},
        'launcher_g0_artifact': {
            'g0_artifact': SOURCE_GZ_REL, 'g0_sha256': SOURCE_GZ_SHA256,
            'why': ('p515_s55_e1_launcher.g0_check reads the digest-run schema (summary.i_x0_A_2x_vs_x0_N_2x, '
                    'summary.eval_keys, configuration.spec_like / models, production.modules_sha256), not this '
                    'schema; this artifact verifies that file\'s G0 content on the final head')},
        'prediction': 'Addendum 83 ruling 3 (vii): identical at x = 0',
        'G0_PASS': g0_pass, 'per_check_pass': per_check, 'gating_checks': gating,
        'results': results, 'informative': informative, 'guards': guards,
        'no_build_statement': ('nothing was built or solved: G0 is read from the committed W183c off run on the final '
                               'production head; the NL identities rest on the recorded raw sha256 of each NL file '
                               '(network NL files hash-recorded, untracked; their gz hashes verified against the '
                               'manifest and their decompressed sha256 against the JSON)'),
        'wall_clock_s': time.time() - started}
    path = os.path.join(out_abs, 'g0.json')
    with open(path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True)
    rel = os.path.relpath(path, REPO)
    with open(os.path.join(out_abs, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump({rel: _sha(rel)}, handle, indent=1, sort_keys=True)
    for grp, ok in per_check.items():
        _log(f'  {"PASS" if ok else "FAIL"} {grp}' + ('' if ok else f': {[k for k, v in gating[grp].items() if v is not True]}'))
    _log(f'guards {guards}')
    _log(f'wrote {rel} sha256={_sha(rel)}')
    _log(f'G0_PASS={g0_pass} wall={time.time() - started:.1f}s')
    if not g0_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
