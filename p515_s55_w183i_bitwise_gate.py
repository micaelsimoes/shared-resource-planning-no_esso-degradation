"""
P5.15 Addendum 86 ruling 1, W183i (a) gate (iv) -- the SRP1 TWO-CYCLE BITWISE GATE re-run on the PATCHED harness (the
one-branch Step-7 router patch), through a THIN SUCCESSOR of wrapper v2 (`p515_s55_w183f_bitwise_gate_v2.py`, committed
4c0d8e1d, NOT edited), read under the ruled comparator (Addendum 85 ruling 1 (c)) against the W48 reference and,
beside it, the W83 arm.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 1 (gate (iv): "the two-cycle bitwise gate re-run on the
patched harness ... into a new write-once root, read under the ruled comparator against the W48 reference and the W83
arm"); Planner task W183i (a).

WHY A SUCCESSOR. Wrapper v2's `--run-gate --arm A --out-rel R` redirects to a fresh root, but its clean-file list is
fixed at `V1.EXTRA_CLEAN_FILES + (v2,)`: it cannot add the new hooks module `p515_s55_step7_resettle_hooks.py`, which
the task requires in the gate's clean-file list; and it reads only against the W48 arm. This file runs the committed
gate `p515_s53_srp1_bitwise_gate.main()` (imported, unchanged; -> W48 -> W39 -> W35 -> W32 -> W10, whose armed bounded
`SolveProfileGuard` is installed at import and verified EXACTLY against the per-event reconciled count) with ONLY the
declared substitutions of its module globals (the v1/v2 convention):
  1. identifiers STAGE / SCHEMA / ARM ('s55w183igate') / OUT_REL -> the fresh write-once root
     `data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate` (the arm-named eval dirs under P56A/evals are new);
  2. EXTRA_CLEAN_FILES = v1's list (the gate's + network_parameters.py + v1) + wrapper v2 + this file +
     `p515_s55_step7_resettle_hooks.py` (the harness is already in the gate's list);
  3. EXTRA_FORBIDDEN = v1's list (the gate's + 'p515_s55_'); this process and its launching shells are excluded as
     ancestors (the committed convention).
Nothing else: the C* reference, the W48 whole-arm reference (manifest-pinned in the gate), tolerances, presence
assertions, the armed W51 counter, the declared solve count and the compared fields are the gate's own.

AFTER THE GATE (zero-solve reads, no edit of anything the gate wrote):
  * GATE_PASS (as frozen: exit 0 and the W51 addendum's gate_pass) and GATE_PASS_RULED (v2's `ruled_comparison` +
    `ruled_verdict` against the W48 arm, imported unchanged);
  * vs the W83 arm (`P515S53/tight_tail_w83/srp1_bitwise_gate`, its three manifests hash-checked first with v2's
    `check_gate_dir`): the gate's comparator (v2 `ruled_comparison` with the W83 arm record) and rule (c); reported as
    counts (genuine / provenance / tie-order / drift), not as a gate verdict (the frozen verdict is vs W48);
  * Q = gross_operational_cost of the new arm against the W48 arm's 740,922,817.2425401.
RECORDED PREDICTION (Addendum 86 ruling 1): 0 genuine, GATE_PASS_RULED True, provenance <= 4.

DECLARED BEFORE THE RUN (the gate's own derivation): 51 solves/cycle x (cap 2 + 1) = 153, plus every attempted retry.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
        /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183i_bitwise_gate.py --run-gate \\
        > data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S55/w183i_srp1_bitwise_gate/{gate.json, gate.md, manifest_sha256.json,
row18_gate_addendum.json, w51_gate_addendum.json, w51_manifest_sha256.json, arm/, w183i_wrapper_record.json,
w183i_manifest_sha256.json}
Exit 0 iff GATE_PASS_RULED; 1 otherwise (or on a precondition refusal).
"""
import os
import sys

if '--run-gate' not in sys.argv:
    sys.exit('p515_s55_w183i_bitwise_gate.py: the only mode is --run-gate (see the module docstring)')

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s55_w183f_bitwise_gate_v2 as V2  # noqa: E402 -- '--run-gate' in argv: no zero-solve guard; imports V1
                                             # -> S53G -> ... -> W10 (installs the armed bounded W10.GUARD)

os.chdir(REPO)
V1 = V2.V1
S53G = V2.S53G
S52G = V2.S52G
W10 = V2.W10
CP = V2.CP
H = V2.H
GUARD = V2.GUARD

SCRIPT = os.path.basename(__file__)
HOOKS_REL = 'p515_s55_step7_resettle_hooks.py'
STAGE = ('P5.15 Addendum 86 W183i (a) gate (iv) -- SRP1 two-cycle bitwise identity (serial, sequential DSO builder) on '
         'the patched harness (Step-7 router branch), vs committed baseline C* and the committed W48 arm; read under '
         'the ruled comparator (Addendum 85 ruling 1 (c)) vs W48 and vs the W83 arm')
SCHEMA = 'p515_s55_w183i_bitwise_gate_v1'
ARM = 's55w183igate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183i_srp1_bitwise_gate')
EXTRA_CLEAN_FILES = tuple(V1.EXTRA_CLEAN_FILES) + (V2.SCRIPT, SCRIPT, HOOKS_REL)
EXTRA_FORBIDDEN = tuple(V1.EXTRA_FORBIDDEN)
CODE_FILES = tuple(V1.CODE_FILES) + (V2.SCRIPT, SCRIPT, HOOKS_REL)
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 86 ruling 1; W183i order item (a) gate (iv)',
             'Planner task W183i (a)']
PREDICTION = '0 genuine, GATE_PASS_RULED True, provenance <= 4'
W48_Q = 740922817.2425401

_W83 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w83', 'srp1_bitwise_gate')
W83_MANIFESTS = {'manifest_sha256.json': 'f4d3f7aa5fa2458cd664ad3a09dba394bb1c8164f7b62302c1926ca0df1b4d03',
                 'w51_manifest_sha256.json': '8754894d0333ceac84ce5f98345b965722263454df13d0bedf721e57b8f9f81b',
                 'w83_manifest_sha256.json': 'a794ce17e16b4d426858272665eec2aa376b0ca326fb29d5861c7abe9610a49f'}
# The committed W83 gate arm, in the gate's reference-arm record format (as p515_s55_w183g_a_vs_w83.W83_ARM).
W83_ARM = {
    'gate_dir': _W83,
    'arm_dir': os.path.join(_W83, 'arm'),
    'manifest': os.path.join(_W83, 'manifest_sha256.json'),
    'manifest_sha256': W83_MANIFESTS['manifest_sha256.json'],
    'gate_json': os.path.join(_W83, 'gate.json'),
    'arm_name': 's53w83tailgate',
    'commit': 'eb8cf81f (W83 step 5, tight tail enabled; post-W64 70950e8c, pre-W183)',
}


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183i-a gate iv] {msg}', flush=True)


def _code_sha():
    return {name: CP._sha256_file(os.path.join(REPO, name)) for name in CODE_FILES}


def _diff_rows(comparison):
    rows = []
    for fname, entry in comparison['per_file'].items():
        for cls, key in (('provenance', 'provenance_diffs'), ('tie_order', 'tie_order_diffs'),
                         ('genuine', 'genuine_diffs_first')):
            for d in entry.get(key, []) or []:
                rows.append({'file': fname, 'class': cls, 'field': d.get('field'), 'reference': d.get('legacy'),
                             'new': d.get('lightweight'), 'reason': d.get('reason')})
    return rows


def _q(arm_dir):
    with open(os.path.join(REPO, arm_dir, 'g_s39_D.json')) as handle:
        g = json.load(handle)
    return g.get('gross_operational_cost'), g.get('cycles_run'), g.get('rule_ten_terminal_step_over_threshold')


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)
    if os.path.exists(out_root):
        _log(f'[PRECONDITION FAILED] output root exists: {OUT_REL}')
        return 1
    head = W10._git(['rev-parse', 'HEAD'])
    code_before = _code_sha()
    _log(f'git HEAD {head}; output root {OUT_REL}')
    for name, sha in code_before.items():
        _log(f'   sha256 {name}: {sha}')
    eval_root = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals')
    stub = f'p515s44_s45_w10_snapoff_{ARM}_{W10.C_STAR_KEY_PIN[:16]}'
    existing = [d for d in (f'{stub}_run', f'{stub}_precheck') if os.path.exists(os.path.join(eval_root, d))]
    if existing:
        for d in existing:
            _log(f'[PRECONDITION FAILED] arm-named eval dir already exists: {d}')
        return 1
    # the W83 reference is hash-checked BEFORE anything solves (refuse on mismatch)
    problems_w83, chk_w83 = V2.check_gate_dir(_W83, W83_MANIFESTS)
    for p in problems_w83:
        _log(f'[PRECONDITION FAILED] W83 arm: {p}')
    if problems_w83:
        return 1
    _log(f'W83 arm hash check: {chk_w83["file_entries_checked"]} entries, 0 problems')

    # ---- the declared substitutions 1-3, and no others ----
    S53G.STAGE, S53G.SCHEMA, S53G.ARM, S53G.OUT_REL = STAGE, SCHEMA, ARM, OUT_REL
    S53G.EXTRA_CLEAN_FILES = EXTRA_CLEAN_FILES
    S53G.EXTRA_FORBIDDEN = EXTRA_FORBIDDEN
    status = S53G.main()
    _log(f'committed gate exit status: {status}')
    code_after = _code_sha()
    w51_path = os.path.join(out_root, 'w51_gate_addendum.json')
    if not os.path.isfile(w51_path):
        _log('the committed layers refused before the arm (no w51_gate_addendum.json); GATE_PASS=False; nothing ruled')
        return 1
    with open(w51_path) as handle:
        addendum = json.load(handle)
    gate_pass = status == 0 and addendum.get('gate_pass') is True

    # ---- ruled vs W48 (the whole-arm reference), v2's functions unchanged ----
    comparison, ruled, _loaded = V2.ruled_comparison(out_root, ARM, V2.REFERENCE_ARM)
    verdict = V2.ruled_verdict(addendum['gate_items'], gate_pass, comparison, ruled,
                               addendum['comparison_vs_committed_w48_arm_GATING'])
    rows_w48 = _diff_rows(comparison)
    for fname in V2.COMPARED_FILES:
        _log(f'   vs W48 {fname}: gate {comparison["per_file"][fname].get("n")} -> rule (c) {ruled[fname].get("n")}')
    _log(f'vs W48 totals {comparison["totals"]}; rule (c) {verdict["totals"]}; drift keys {verdict["n_drift_keys"]}')

    # ---- vs W83 (beside it), the same comparator and rule (c) ----
    comparison_w83, ruled_w83, _l83 = V2.ruled_comparison(out_root, ARM, W83_ARM)
    rows_w83 = _diff_rows(comparison_w83)
    totals_w83 = {'genuine_frozen': 0, 'field_list_drift': 0, 'genuine_ruled': 0}
    for r in ruled_w83.values():
        for k in totals_w83:
            totals_w83[k] += (r.get('n') or {}).get(k, 0)
    drift_w83 = sorted({k for r in ruled_w83.values() for k in r.get('drift_keys', [])})
    for fname in V2.COMPARED_FILES:
        _log(f'   vs W83 {fname}: gate {comparison_w83["per_file"][fname].get("n")} -> rule (c) '
             f'{ruled_w83[fname].get("n")}')
    _log(f'vs W83 totals {comparison_w83["totals"]}; rule (c) {totals_w83}; drift keys {len(drift_w83)}')
    q_new, cycles_new, r10_new = _q(os.path.join(OUT_REL, 'arm'))
    q_w48, _c48, r10_w48 = _q(V2.REFERENCE_ARM['arm_dir'])
    q_w83, _c83, r10_w83 = _q(W83_ARM['arm_dir'])
    prov_w48 = comparison['totals']['provenance']
    prov_w83 = comparison_w83['totals']['provenance']
    outcome = {
        'GATE_PASS_as_frozen': gate_pass, 'GATE_PASS_RULED_vs_W48': verdict['GATE_PASS_RULED'],
        'genuine_ruled_vs_W48': verdict['totals']['genuine_ruled'], 'genuine_ruled_vs_W83': totals_w83['genuine_ruled'],
        'provenance_vs_W48': prov_w48, 'provenance_vs_W83': prov_w83,
        'tie_order_vs_W48': comparison['totals']['tie_order'], 'tie_order_vs_W83': comparison_w83['totals']['tie_order'],
        'prediction_0_genuine_held': verdict['totals']['genuine_ruled'] == 0 and totals_w83['genuine_ruled'] == 0,
        'prediction_gate_pass_ruled_held': verdict['GATE_PASS_RULED'] is True,
        'prediction_provenance_le_4_held': prov_w48 <= 4 and prov_w83 <= 4,
        'Q_new': q_new, 'Q_W48': q_w48, 'Q_W83': q_w83, 'Q_new_equals_W48_bitwise': q_new == q_w48 == W48_Q,
        'cycles_run_new': cycles_new,
        'rule_ten_terminal_step_over_threshold': {'new': r10_new, 'W48': r10_w48, 'W83': r10_w83},
    }
    outcome['prediction_held'] = (outcome['prediction_0_genuine_held'] and outcome['prediction_gate_pass_ruled_held']
                                  and outcome['prediction_provenance_le_4_held'])
    record = {
        'schema': SCHEMA + '_wrapper_record', 'stage': STAGE, 'authority': AUTHORITY, 'rule': V2.RULE_TEXT,
        'recorded_prediction': PREDICTION,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'code_sha256_before_run': code_before, 'code_sha256_after_run': code_after,
        'code_unchanged_during_run': code_before == code_after,
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in gate.json',
        'scope': ('sequential DSO builder only (ParallelExecution false in the case file); the parallel DSO builder '
                  'is not exercised by this gate (Addendum 85 ruling 1, scope)'),
        'declared_substitutions': {'1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
                                   '2_extra_clean_files': list(EXTRA_CLEAN_FILES),
                                   '3_extra_forbidden': list(EXTRA_FORBIDDEN)},
        'committed_gate_exit_status': status,
        'GATE_PASS': gate_pass, 'GATE_PASS_RULED': verdict['GATE_PASS_RULED'],
        'outcome_vs_prediction': outcome,
        'vs_W48': {'reference_arm': V2.REFERENCE_ARM, 'drift_keys': verdict['drift_keys'], 'verdict': verdict,
                   'per_file_gate': {f: comparison['per_file'][f].get('n', comparison['per_file'][f].get('error'))
                                     for f in V2.COMPARED_FILES},
                   'per_file_rule_c': ruled, 'every_diff': rows_w48},
        'vs_W83': {'reference_arm': {**W83_ARM, 'gate_dir_check': chk_w83}, 'totals_gate': comparison_w83['totals'],
                   'totals_rule_c': totals_w83, 'drift_keys': drift_w83,
                   'per_file_gate': {f: comparison_w83['per_file'][f].get('n',
                                                                         comparison_w83['per_file'][f].get('error'))
                                     for f in V2.COMPARED_FILES},
                   'per_file_rule_c': ruled_w83, 'every_diff': rows_w83, 'comparison': comparison_w83},
        'guard_counts_at_end': dict(GUARD.counts),
    }
    V2._write_json(os.path.join(out_root, 'w183i_wrapper_record.json'), record)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            manifest[W10._rel(os.path.join(root, fname))] = CP._sha256_file(os.path.join(root, fname))
    V2._write_json(os.path.join(out_root, 'w183i_manifest_sha256.json'), manifest)
    _log(f'code unchanged during run: {code_before == code_after}; guard counts {dict(GUARD.counts)}')
    _log(f'Q new {q_new!r} vs W48 {q_w48!r} (equal {q_new == q_w48}); outcome {outcome}')
    _log(f'GATE_PASS={gate_pass} GATE_PASS_RULED={verdict["GATE_PASS_RULED"]}')
    return 0 if verdict['GATE_PASS_RULED'] else 1


if __name__ == '__main__':
    sys.exit(main())
