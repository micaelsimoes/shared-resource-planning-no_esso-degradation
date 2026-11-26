"""
P5.15 Addendum 83, W183 order item W183f (gate part) -- the SERIAL SRP1 TWO-CYCLE BITWISE GATE
(`p515_s53_srp1_bitwise_gate.py`, committed, run AS IT IS by import) re-run on the code changed by W183a/W183c.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 83, W183 order (W183f: "the serial two-cycle bitwise gate
(`p515_s53_srp1_bitwise_gate.py`) re-run on the changed code"); Planner task W183f (gate part). The spec v1 freeze is
NOT part of this script.

RECORDED PREDICTION (Addendum 83, W183 predictions): "the two-cycle gate passes".

WHAT CHANGED SINCE THE GATE LAST RAN. W183c (6f43479e): `shared_resources_planning.py` (NRF rows constructed only
when `no_reverse_flow` is declared; default off) and `network_parameters.py` (`no_reverse_flow`, default False);
W183a/W183c: `p515_s44_campaign_harness.py` (rho0_scale / nrf_option keyed options, NRF capture). The gate's arm
runs the SEQUENTIAL DSO builder only (ParallelExecution false in the case file); the parallel builder is NOT
exercised here (W183c's zero-solve NL identity covers it).

HOW IT IS BUILT. `p515_s53_srp1_bitwise_gate` BY IMPORT (-> W48 -> W39 -> W35 -> W32 -> W10, whose armed bounded
`SolveProfileGuard` is installed at import and verified EXACTLY against the per-event reconciled count). The
committed gate's `main()` is called unchanged, with exactly these declared substitutions of its module globals
(its own substitution convention) and no others:
  1. identifiers `STAGE` / `SCHEMA` / `ARM` ('s55w183fgate') / `OUT_REL` -> the fresh write-once root
     `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate` (the arm-named eval dirs under P56A/evals are new);
  2. `EXTRA_CLEAN_FILES` -- the gate's list (which already names shared_resources_planning.py,
     p515_s44_campaign_harness.py, model_construction_helpers.py, network.py and every imported gate script) plus
     `network_parameters.py` and this file, so the gate refuses an uncommitted edit of any of them;
  3. `EXTRA_FORBIDDEN` -- the gate's list plus 'p515_s55_', so a live E-campaign process refuses the gate (this
     process and its launching shells are excluded as ancestors, the committed convention).
Unchanged: the reference (`W32.BASELINE_REFERENCE`, C* g_s39_D.json), the whole-arm reference (the committed W48
arm `data/SRP1/Results/P515S52/srp1_bitwise_gate/arm`, manifest-pinned in the gate), tolerances, presence
assertions (row 18 + W47 + W51), the armed W51 counter, the declared solve count and the compared fields.

ONE WRAPPER-ONLY PRECONDITION (refuse-only; changes nothing the gate compares): the arm-named eval dirs
`P56A/evals/p515s44_s45_w10_snapoff_s55w183fgate_<key16>_{run,precheck}` must not exist before the run.
ONE WRAPPER-ONLY RECORD, written AFTER the gate returns: `w183f_wrapper_record.json` (wrapper / imported-gate /
code sha256 before and after the run, git HEAD, substitutions, the gate's exit status and verdict) and
`w183f_manifest_sha256.json` over the whole output root.

DECLARED BEFORE THE RUN (the gate's own derivation from data/SRP1/SRP1.json): 51 solves/cycle x (cap 2 + 1
initialisation) = 153, plus every attempted retry, verified exactly by the armed guard.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
        /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate.py \\
        > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate/{gate.json, gate.md, manifest_sha256.json,
row18_gate_addendum.json, w51_gate_addendum.json, w51_manifest_sha256.json, arm/, w183f_wrapper_record.json,
w183f_manifest_sha256.json}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s53_srp1_bitwise_gate as S53G  # noqa: E402 -- imports W48 -> W39 -> W35 -> W32 -> W10 (armed guard)

W10 = S53G.W10
GUARD = W10.GUARD
CP = W10.CP

STAGE = ('P5.15 Addendum 83 W183f -- SRP1 two-cycle bitwise identity (serial, sequential DSO builder) re-run on the '
         'W183a/W183c code (NRF off by default), vs committed baseline C* and the committed W48 arm')
SCHEMA = 'p515_s55_w183f_bitwise_gate_v1'
ARM = 's55w183fgate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183f_srp1_bitwise_gate')
EXTRA_CLEAN_FILES = tuple(S53G.EXTRA_CLEAN_FILES) + ('network_parameters.py', os.path.basename(__file__))
EXTRA_FORBIDDEN = tuple(S53G.EXTRA_FORBIDDEN) + ('p515_s55_',)

CODE_FILES = ('shared_resources_planning.py', 'network_parameters.py', 'network.py', 'model_construction_helpers.py',
              'admm_parameters.py', 'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py',
              'p515_s53_srp1_bitwise_gate.py', 'p515_s52_srp1_bitwise_gate.py', 'p515_s51_srp1_bitwise_gate.py',
              'p515_s50_generalization_gate.py', 'p515_s49_memory_fix_gate.py',
              'p515_s45_snapshot_off_two_cycle_gate.py', os.path.basename(__file__))


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183f-gate] {msg}', flush=True)


def _code_sha():
    return {name: CP._sha256_file(os.path.join(REPO, name)) for name in CODE_FILES}


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)
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

    # ---- the declared substitutions 1-3, and no others ----
    S53G.STAGE = STAGE
    S53G.SCHEMA = SCHEMA
    S53G.ARM = ARM
    S53G.OUT_REL = OUT_REL
    S53G.EXTRA_CLEAN_FILES = EXTRA_CLEAN_FILES
    S53G.EXTRA_FORBIDDEN = EXTRA_FORBIDDEN

    status = S53G.main()
    _log(f'committed gate exit status: {status}')

    code_after = _code_sha()
    if not os.path.isdir(out_root):
        _log('output root absent (the committed layers refused before the arm); nothing more to write')
        _log('GATE_PASS=False')
        return 1

    verdict = {}
    w51_path = os.path.join(out_root, 'w51_gate_addendum.json')
    if os.path.isfile(w51_path):
        with open(w51_path) as handle:
            w51 = json.load(handle)
        verdict = {'gate_items': w51.get('gate_items'), 'gate_pass': w51.get('gate_pass')}
    gate_pass = status == 0 and verdict.get('gate_pass') is True
    record = {
        'schema': SCHEMA + '_wrapper_record', 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 83, W183 order item W183f (gate part)',
                      'Planner task W183f (gate part)'],
        'recorded_prediction': 'the two-cycle gate passes',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head,
        'code_sha256_before_run': code_before, 'code_sha256_after_run': code_after,
        'code_unchanged_during_run': code_before == code_after,
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in gate.json',
        'scope': ('sequential DSO builder only (ParallelExecution false in the case file); the parallel DSO builder '
                  'is not exercised by this gate'),
        'declared_substitutions': {
            '1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
            '2_extra_clean_files': list(EXTRA_CLEAN_FILES),
            '3_extra_forbidden': list(EXTRA_FORBIDDEN)},
        'committed_gate_exit_status': status,
        'committed_gate_verdict': verdict,
        'guard_counts_at_end': dict(GUARD.counts),
        'gate_pass': gate_pass,
    }
    record_path = os.path.join(out_root, 'w183f_wrapper_record.json')
    manifest_path = os.path.join(out_root, 'w183f_manifest_sha256.json')
    W10._refuse_overwrite(record_path)
    W10._refuse_overwrite(manifest_path)
    with open(record_path, 'w') as handle:
        json.dump(record, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'code unchanged during run: {code_before == code_after}; guard counts {dict(GUARD.counts)}')
    _log(f'wrote {W10._rel(record_path)}, {W10._rel(manifest_path)} ({len(manifest)} files)')
    _log(f'GATE_PASS={gate_pass}')
    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
