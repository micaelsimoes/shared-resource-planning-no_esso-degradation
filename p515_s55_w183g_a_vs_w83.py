"""
P5.15 Addendum 85 ruling 1 (b'), W183g (a) -- ZERO-SOLVE whole-arm comparison of the COMMITTED W183f gate arm against
the COMMITTED W83 gate arm (post-W64, pre-W183), with the gate's OWN comparator.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 1 (b') and W183g order item (a); Planner task W183g (a).

RECORDED PREDICTION (expert, Addendum 85 ruling 1 (b')): 0 genuine diffs on all 12 files; provenance diffs <= 3, all
under `rule_eleven_checklist` (`w10_arm`, `w183a_nrf_option`, `w84_convergence_depth_tail`).

WHAT IS COMPARED
  new arm   data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate/arm   (arm s55w183fgate, committed c5b292dd)
  reference data/SRP1/Results/P515S53/tight_tail_w83/srp1_bitwise_gate/arm   (arm s53w83tailgate, committed eb8cf81f;
            the W83 run with the tight tail enabled and inert, after W64 70950e8c, before W183)
COMPARATOR: the gate's own, imported and called unchanged -- `S52G.compare_against_w39_arm(out_root)` (the function
the committed gate `p515_s53_srp1_bitwise_gate.py` calls for its whole-arm item, after its substitution 5 of
`S52G.W39_ARM`), i.e. `W10.classify` per file over `W10.GATING_JSON + W10.GATING_JSONL` (12 files), `CP._diff` bitwise
with the declared exclusions, the rule_eleven_checklist subtree as provenance, arm-path tokens normalized, recourse-jump
tie order by TC. The only substitutions are the ones the gate itself makes: `S52G.W39_ARM` = the W83 arm record below
(as the gate sets it to its W48 arm), `S52G.ARM` = 's55w183fgate' (the W183f arm name, as the gate had set it). In the
comparator's own output 'w39' / 'legacy' = the REFERENCE (W83) side and 'w48' / 'lightweight' = the NEW (W183f) side
(labels hard-coded in S52G; recorded here so they are not misread).

PRECONDITIONS (refuse on any failure, before comparing): every file listed in every manifest of the W83 gate directory
(manifest_sha256.json, w51_manifest_sha256.json, w83_manifest_sha256.json; each manifest sha256 pinned below to its
committed blob) hashes to its entry, no file on disk is unlisted, and the directory is clean in git; the same for the
W183f gate directory (manifest_sha256.json, w51_manifest_sha256.json, w183f_manifest_sha256.json); and the gate's own
pin check (`S52G._check_w39_arm_pins`) passes on the W83 arm record. The output directory exists (created by the
launch-log redirect) and holds nothing but launch.log; an O_EXCL lock refuses a concurrent copy.

GUARDS: `SolveProfileGuard(permitted=())` installed BEFORE any project import and verified at exactly 0 (and the
imported chain's W10.GUARD, installed at its import, also 0); `pickle.load` / `pickle.loads` blocked and counted (0);
`pyomo ConcreteModel.__init__` blocked and counted (0). The ESSO pickle files are HASHED by the comparator
(informational), never unpickled. JSON written through `gate_result_io`. The gate is NOT re-run; its committed outputs
and frozen verdict are read only.

OUTPUT (write-once): data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83/{vs_w83.json, manifest_sha256.json};
`--post-run` adds manifest_post_run_sha256.json over the directory incl. launch.log.

EXACT COMMANDS (repo root; canonical interpreter; attached, alone, both streams captured; never detached):
  mkdir -p data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183g_a_vs_w83.py \\
      > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_vs_w83/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183g_a_vs_w83.py --post-run
Exit 0 when the comparison completed with all guards at 0 (the prediction's outcome is recorded, not encoded in the
exit status); 1 on a refusal.
"""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

ZERO_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183g (a) (b\') vs W83 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError("W183g (a) (b'): pickle.load called -- nothing is unpickled")


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError("W183g (a) (b'): pickle.loads called -- nothing is unpickled")


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError("W183g (a) (b'): a pyomo ConcreteModel was constructed -- no model build")


pe.ConcreteModel.__init__ = _blocked_concrete_init

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s53_srp1_bitwise_gate as S53G  # noqa: E402 -- the committed gate (-> S52G -> ... -> W10, W10.GUARD)

pickle.load, pickle.loads = _blocked_load, _blocked_loads   # re-assert after the imports
os.chdir(REPO)

S52G = S53G.S52G
W10 = S53G.W10
CP = W10.CP
GUARD = W10.GUARD

SCRIPT = os.path.basename(__file__)
SCHEMA = 'p515_s55_w183g_a_vs_w83_v1'
AUTHORITY = ["PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 1 (b'); W183g order item (a)", 'Planner task W183g (a)']
PREDICTION = ("0 genuine diffs on all 12 files; provenance diffs <= 3, all under rule_eleven_checklist (w10_arm, "
              "w183a_nrf_option, w84_convergence_depth_tail)")
PREDICTED_PROVENANCE_KEYS = ('w10_arm', 'w183a_nrf_option', 'w84_convergence_depth_tail')

_S55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
_W83 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w83', 'srp1_bitwise_gate')
W183F_GATE = os.path.join(_S55, 'w183f_srp1_bitwise_gate')
W183F_ARM_NAME = 's55w183fgate'
W183F_MANIFESTS = {'manifest_sha256.json': '140c33e1a5cfc73cd05c121f10a0f8d68837dafa002f08f542cadef0dd2f17a1',
                   'w51_manifest_sha256.json': 'a6c783394501190525ca2a91d0f281e5b3dd7deef91b9aeef8d2a3ffcaa63638',
                   'w183f_manifest_sha256.json': 'e7cd233e812998949def55b8f175389e5a546eb1d92783271c50515837f26943'}
W83_MANIFESTS = {'manifest_sha256.json': 'f4d3f7aa5fa2458cd664ad3a09dba394bb1c8164f7b62302c1926ca0df1b4d03',
                 'w51_manifest_sha256.json': '8754894d0333ceac84ce5f98345b965722263454df13d0bedf721e57b8f9f81b',
                 'w83_manifest_sha256.json': 'a794ce17e16b4d426858272665eec2aa376b0ca326fb29d5861c7abe9610a49f'}
# The committed W83 gate arm, in the gate's own reference-arm record format (S53G.W48_ARM / S52G.W39_ARM).
W83_ARM = {
    'gate_dir': _W83,
    'arm_dir': os.path.join(_W83, 'arm'),
    'manifest': os.path.join(_W83, 'manifest_sha256.json'),
    'manifest_sha256': W83_MANIFESTS['manifest_sha256.json'],
    'gate_json': os.path.join(_W83, 'gate.json'),
    'arm_name': 's53w83tailgate',
    'commit': 'eb8cf81f (W83 step 5, tight tail enabled; post-W64 70950e8c, pre-W183)',
}
OUT_REL = os.path.join(_S55, 'w183f_srp1_bitwise_gate_vs_w83')
LOCK_NAME = '.w183g_a_vs_w83.lock'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183g-a vs W83] {msg}', flush=True)


def _sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_gate_dir(gate_rel, manifests):
    problems, checked, listed = [], 0, set()
    for mname, pin in manifests.items():
        mpath = os.path.join(REPO, gate_rel, mname)
        got = _sha(mpath) if os.path.isfile(mpath) else None
        if got != pin:
            problems.append(f'{gate_rel}/{mname}: sha256 {got} != pin {pin}')
            continue
        with open(mpath) as handle:
            manifest = json.load(handle)
        for rel, sha in manifest.items():
            listed.add(rel)
            checked += 1
            path = os.path.join(REPO, rel)
            if not os.path.isfile(path) or _sha(path) != sha:
                problems.append(f'{rel} does not hash to {mname}')
    on_disk = {os.path.relpath(os.path.join(r, f), REPO) for r, _d, fs in os.walk(os.path.join(REPO, gate_rel))
               for f in fs}
    unlisted = sorted(on_disk - listed - {os.path.join(gate_rel, m) for m in manifests})
    if unlisted:
        problems.append(f'{gate_rel}: files on disk in no manifest: {unlisted}')
    status = W10._git(['status', '--porcelain', '--', gate_rel])
    if status.strip():
        problems.append(f'{gate_rel} not clean in git:\n{status}')
    return problems, {'gate_dir': gate_rel, 'manifests_pinned': manifests, 'file_entries_checked': checked,
                      'files_on_disk': len(on_disk), 'unlisted_files': unlisted, 'git_clean': not status.strip()}


def _guards():
    zero, w10 = ZERO_GUARD.verify(0), GUARD.verify(0)
    return {'zero_solve_guard': {'counts': dict(ZERO_GUARD.counts), 'verify_0_failures': zero},
            'imported_chain_W10_GUARD': {'counts': dict(GUARD.counts), 'verify_0_failures': w10},
            'pickle': dict(PICKLE_COUNTS), 'model_build': dict(MODEL_BUILD_COUNTS),
            'all_zero': (not zero and not w10 and PICKLE_COUNTS == {'load': 0, 'loads': 0}
                         and MODEL_BUILD_COUNTS['ConcreteModel.__init__'] == 0)}


def _write_json(path, payload):
    W10._refuse_overwrite(path)
    text = GRIO.dumps(payload, indent=1)
    with open(path, 'w') as handle:
        handle.write(text)


def _diff_table(comparison):
    """Every diff by file, class and key (field), from the comparator's own per-file output."""
    rows = []
    for fname, entry in comparison['per_file'].items():
        if 'error' in entry:
            rows.append({'file': fname, 'class': 'error', 'field': None, 'detail': entry['error']})
            continue
        for cls, key in (('provenance', 'provenance_diffs'), ('tie_order', 'tie_order_diffs'),
                         ('genuine', 'genuine_diffs_first')):
            for d in entry.get(key, []):
                rows.append({'file': fname, 'class': cls, 'field': d.get('field'),
                             'reference_w83': d.get('legacy'), 'new_w183f': d.get('lightweight'),
                             'reason': d.get('reason')})
    return rows


def run(out_dir):
    started = time.time()
    head = W10._git(['rev-parse', 'HEAD'])
    problems_new, chk_new = check_gate_dir(W183F_GATE, W183F_MANIFESTS)
    problems_ref, chk_ref = check_gate_dir(_W83, W83_MANIFESTS)
    S52G.W39_ARM = W83_ARM                     # the gate's substitution 5, pointed at the W83 arm
    S52G.ARM = W183F_ARM_NAME                  # the W183f arm name, as the gate had set it
    problems_pin, _m = S52G._check_w39_arm_pins()
    with open(os.path.join(REPO, W183F_GATE, 'w183f_wrapper_record.json')) as handle:
        wrapper = json.load(handle)
    with open(os.path.join(REPO, _W83, 'gate.json')) as handle:
        w83_gate = json.load(handle)
    arm_name_ok = (wrapper['declared_substitutions']['1_identifiers']['ARM'] == W183F_ARM_NAME
                   and all(W83_ARM['arm_name'] in v for v in w83_gate['arm_summary']['working_dir_ids'].values()))
    problems = problems_new + problems_ref + problems_pin + ([] if arm_name_ok else ['arm names do not match records'])
    _log(f'hash checks: W183f {chk_new["file_entries_checked"]} entries, W83 {chk_ref["file_entries_checked"]} entries; '
         f'gate pin check problems {len(problems_pin)}; arm names ok {arm_name_ok}')
    for p in problems:
        _log(f'[REFUSED] {p}')
    if problems:
        return 1

    comparison = S52G.compare_against_w39_arm(os.path.join(REPO, W183F_GATE))
    table = _diff_table(comparison)
    per_file = {f: v.get('n', {'error': v.get('error')}) for f, v in comparison['per_file'].items()}
    for fname, n in per_file.items():
        _log(f'   {fname}: {n}')
    for r in table:
        _log(f"   diff [{r['class']}] {r['file']} :: {r['field']}")
    totals = comparison['totals']
    prov_fields = [r['field'] for r in table if r['class'] == 'provenance']
    prefix = 'g_s39_D.json.rule_eleven_checklist.'
    prov_keys = [f[len(prefix):] if isinstance(f, str) and f.startswith(prefix) else None for f in prov_fields]
    outcome = {
        'n_files_compared': len(comparison['files_compared']),
        'all_12_files_present': comparison['all_files_present'] and len(comparison['files_compared']) == 12,
        'genuine_total': totals['genuine'],
        'tie_order_total': totals['tie_order'],
        'provenance_total': totals['provenance'],
        'zero_genuine_on_all_12_files': (totals['genuine'] == 0 and comparison['all_files_present']
                                         and len(comparison['files_compared']) == 12),
        'provenance_at_most_3': totals['provenance'] <= 3,
        'provenance_all_under_rule_eleven_checklist': all(k is not None for k in prov_keys),
        'provenance_keys': prov_keys,
        'provenance_keys_within_predicted_set': all(k in PREDICTED_PROVENANCE_KEYS for k in prov_keys),
    }
    outcome['prediction_part_genuine_held'] = outcome['zero_genuine_on_all_12_files'] and totals['tie_order'] == 0
    outcome['prediction_part_provenance_held'] = (outcome['provenance_at_most_3']
                                                  and outcome['provenance_all_under_rule_eleven_checklist']
                                                  and outcome['provenance_keys_within_predicted_set'])
    outcome['prediction_held'] = (outcome['prediction_part_genuine_held']
                                  and outcome['prediction_part_provenance_held'])
    guards = _guards()
    _log(f'totals {totals}; prediction held {outcome["prediction_held"]} (genuine part '
         f'{outcome["prediction_part_genuine_held"]}, provenance part {outcome["prediction_part_provenance_held"]}: '
         f'keys {prov_keys}); guards all zero {guards["all_zero"]}')

    payload = {
        'schema': SCHEMA, 'authority': AUTHORITY, 'recorded_prediction': PREDICTION,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'script': SCRIPT, 'script_sha256': _sha(os.path.abspath(__file__)),
        'imported_sha256': {n: _sha(os.path.join(REPO, n)) for n in (
            'p515_s53_srp1_bitwise_gate.py', 'p515_s52_srp1_bitwise_gate.py', 'p515_s45_snapshot_off_two_cycle_gate.py',
            'p515_s40_clone_capture_preflight.py', 'p515_s43_aa_flagoff_gate.py', 'p515_s44_tie_classifier.py',
            'gate_result_io.py', 'p513_solve_profile_guard.py')},
        'nothing_re_run': 'no solve; the gate is not re-run; both committed arms are read only',
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in the gate',
        'new_arm': {'gate_dir': W183F_GATE, 'arm_name': W183F_ARM_NAME, 'commit': 'c5b292dd', 'check': chk_new},
        'reference_arm': {**W83_ARM, 'check': chk_ref, 'gate_pin_check_problems': problems_pin},
        'label_mapping': {"comparator 'w39' / 'legacy'": 'reference = W83 arm',
                          "comparator 'w48' / 'lightweight'": 'new = W183f arm'},
        'comparator_call': 'S52G.compare_against_w39_arm(<W183f gate dir>) with S52G.W39_ARM = W83 arm record',
        'per_file_counts': per_file, 'totals': totals,
        'every_diff_by_file_class_key': table,
        'outcome_vs_prediction': outcome,
        'comparison': comparison,
        'guards': guards,
        'wall_clock_s': time.time() - started,
    }
    _write_json(os.path.join(out_dir, 'vs_w83.json'), payload)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            if fname in ('launch.log', LOCK_NAME):
                continue
            manifest[W10._rel(os.path.join(root, fname))] = _sha(os.path.join(root, fname))
    _write_json(os.path.join(out_dir, 'manifest_sha256.json'), manifest)
    _log(f'wrote {OUT_REL}/vs_w83.json, manifest_sha256.json ({len(manifest)} files)')
    return 0 if guards['all_zero'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-run', action='store_true')
    args = parser.parse_args(argv)
    out_dir = os.path.join(REPO, OUT_REL)
    if args.post_run:
        path = os.path.join(out_dir, 'manifest_post_run_sha256.json')
        manifest = {}
        for root, _dirs, fnames in os.walk(out_dir):
            for fname in sorted(fnames):
                manifest[W10._rel(os.path.join(root, fname))] = _sha(os.path.join(root, fname))
        _write_json(path, manifest)
        guards = _guards()
        _log(f'wrote {W10._rel(path)} ({len(manifest)} files); guards all zero {guards["all_zero"]}')
        return 0 if guards['all_zero'] else 1
    _log(f"{SCHEMA}: W183f arm vs W83 arm, the gate's own comparator (zero solves)")
    if not os.path.isdir(out_dir):
        _log(f'[REFUSED] output directory {OUT_REL} does not exist (create it with the launch-log redirect)')
        return 1
    extra = sorted(set(os.listdir(out_dir)) - {'launch.log'})
    if extra:
        _log(f'[REFUSED] output directory {OUT_REL} is not fresh (holds {extra}); write-once')
        return 1
    lock = os.path.join(out_dir, LOCK_NAME)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _log('[REFUSED] another copy holds the lock')
        return 1
    try:
        return run(out_dir)
    finally:
        os.close(fd)
        os.remove(lock)


if __name__ == '__main__':
    sys.exit(main())
