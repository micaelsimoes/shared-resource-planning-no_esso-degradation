"""
P5.15 Addendum 36/39 (task W48) -- the SRP1 TWO-CYCLE BITWISE IDENTITY gate for the W47 harness extension and
the two scenario-indexing fixes, against the SAME committed baseline C* reference as the W35 and W39 gates.

Authority: Planner task W48 ruling 2 ("gate the harness change with the SRP1 two-cycle bitwise identity before the
pilot runs; reuse the committed gate conventions by import against the same committed C* reference, with the
declared solve count reconciled per event and an armed bounded guard; run it ONCE; if it does not pass, STOP").

WHAT IS UNDER TEST. W47 (298e58f0) changed, on the path an SRP1 ADMM evaluation executes:
  * `p515_s44_campaign_harness._config_hook_factory` -- new `interface_deviation_premium` argument and a
    `derived_instance` branch, both entered only when declared (neither is declared here);
  * `p515_g_g1_g4_admm_gates.run_admm_arm` -- passes `optimization_results` / `primal_evolution` only to a
    post-run hook that DECLARES them (signature-inspected); the gate's hook (W10's, by import) declares neither,
    so it is called exactly as before;
  * `p515_g_g1_g4_admm_gates._s31c_interface_detail` -- flexibility-volume sums probability-weighted over the
    scenario keys (weight 1.0 * 1.0 at one scenario), written by `run_admm_arm` into
    `interface_settlement_detail_s31c.json` inside the two-cycle arm;
  * `shared_resources_planning._process_scenario_dispersion_results` -- the workbook's shared-ESS dispersion
    read at `sess_na_scenario` (the pair itself at one scenario).
Each is claimed inert at SRP1 (1 x 1). The W47 zero-solve checks (ae951d31) test that without solving; this gate
is the solve-bearing test.

HOW IT IS BUILT. The committed W39 gate 1 (`p515_s51_srp1_bitwise_gate.py`, itself the committed W35 gate
`p515_s50_generalization_gate.py` on W32 / W10) BY IMPORT and RUN AS IT IS -- its armed `SolveProfileGuard`
(W10.GUARD, bounded, `p514_n_instrumented_cstar.PERMITTED` call sites, verified EXACTLY against the per-EVENT
reconciled count), its preconditions and lock refusals, its capture-path checklist, the per-event solve
reconciliation (`p515_s44_scale_measurement.event_level_solve_reconciliation`: 51 solves/cycle x (cap 2 + 1) =
153 + every attempted retry), the committed reference and its sha256 pin (`W32.BASELINE_REFERENCE`:
P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json, cycle_trajectory[:2] + the rows-derived
top-level fields, GATING), the trajectory field table, the row 18 presence checks and the ARMED tripwire on the
retired quadratic -- with exactly these declared substitutions and no others:
  1. `STAGE` / `SCHEMA` / `ARM` / `OUT_REL` -- this stage's identifiers and its own write-once output root;
  2. `EXTRA_CLEAN_FILES` -- extended with the W47 zero-solve checks and this file (the W47 modules and the W39
     gate script are already in the list), so the gate refuses to run against an uncommitted edit of any of them;
  3. the live-module presence assertion = the W39 row 18 checks PLUS `w47_code_presence` below (a stale import
     of pre-W47 code cannot pass);
  4. `EXTRA_FORBIDDEN` -- extended with 'p515_s51_' and 'p515_s52_' (the stages run since W35), so a live pilot
     campaign process refuses the gate (this process and its launching shells are excluded as ancestors, the
     committed convention).
PLUS ONE ADDITIONAL GATING COMPARISON, declared here before the run: the arm's whole output against the COMMITTED
W39 gate-1 arm (`data/SRP1/Results/P515S51/srp1_bitwise_gate/arm`, identical configuration -- C*, baseline, cap 2,
snapshots on -- run on the pre-W47 code at 4ce7447b), with W10's own comparator (`W10.classify`, BY IMPORT) over
W10's gating file lists (`CP.ARTIFACT_FILES` = g_s39_D.json, boyd_terminal.json, component_levels_terminal.json,
interface_settlement_detail_s31c.json, interface_voltage_terminal.json; and the seven sidecar / per-cycle JSONL
files). Every committed file compared is first checked against the W39 gate's committed manifest. Reason: the
reference comparison above covers the trajectory rows only, and the S31C writer W47 changed runs INSIDE the arm;
its output is compared here. Pre-run evidence that this comparison is well-defined across processes: the committed
W35 (P515S50) and W39 (P515S51) gate arms -- two processes, a day apart -- have byte-identical JSONL sidecars, and
their JSON artifacts differ only in timestamp_utc and arm-directory paths (plus the row 18 reporting fields W39
added to component_levels_terminal.json, which the W39 arm, the one compared here, already carries).

GATE = the W39 gate's verdict (every W35 gate item + row 18 presence + retired quadratic never called) AND the W47
presence checks AND zero genuine diffs against the committed W39 arm with every compared file present.

NOT COVERED, stated rather than implied: `_child_real`'s W47 branches (derived-instance install, terminal-phase
lock, multi-scenario capture, workbook) -- entered only when `derived_instance` or `interface_deviation_premium`
is declared; W10.run_arm reproduces the child's wiring rather than calling `_child_real`; the zero-solve checks C
and D exercised them on the pilot instance. `_process_scenario_dispersion_results` runs only in the workbook
writer, which a two-cycle arm does not call (zero-solve check F: at SRP1 the NA pair is the only pair).

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s52_srp1_bitwise_gate.py \\
        > data/SRP1/Results/P515S52/srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S52/srp1_bitwise_gate/{gate.json, gate.md, manifest_sha256.json,
row18_gate_addendum.json, w48_gate_addendum.json, w48_manifest_sha256.json, arm/}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import inspect
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s51_srp1_bitwise_gate as S51G  # noqa: E402 -- imports W35 -> W32 -> W10 (installs the armed guard)

W35G = S51G.W35G
W10 = S51G.W10
GUARD = W10.GUARD
H = W10.H
G = W10.G
CP = W10.CP

import shared_resources_planning as srp  # noqa: E402

STAGE = ('P5.15 Addendum 36/39 W48 -- SRP1 two-cycle bitwise identity: the W47 harness extension and the two '
         'scenario-indexing fixes are inert at one scenario, vs committed baseline C*')
SCHEMA = 'p515_s52_srp1_bitwise_gate_v1'
ARM = 's52w48gate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S52', 'srp1_bitwise_gate')
EXTRA_CLEAN_FILES = tuple(S51G.EXTRA_CLEAN_FILES) + (   # S51G's list already names the W39 gate script
    'p515_s52_pilot_checks.py', os.path.basename(__file__))
EXTRA_FORBIDDEN = tuple(W35G.EXTRA_FORBIDDEN) + ('p515_s51_', 'p515_s52_')

# The committed W39 gate-1 arm: same configuration, pre-W47 code (4ce7447b).
W39_ARM = {
    'gate_dir': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'srp1_bitwise_gate'),
    'arm_dir': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'srp1_bitwise_gate', 'arm'),
    'manifest': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'srp1_bitwise_gate', 'manifest_sha256.json'),
    'manifest_sha256': 'c832a64c0b0a0442f8bd0a96150f81b5ddfc3ae14963d05b4582db709c615146',
    'gate_json': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'srp1_bitwise_gate', 'gate.json'),
    'arm_name': S51G.ARM,
    'commit': '4ce7447b',
}
COMPARED_JSON = tuple(W10.GATING_JSON)
COMPARED_JSONL = tuple(W10.GATING_JSONL)


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W48-gate] {msg}', flush=True)


# ======================================================================================
#  substitution 3 -- the W47 code must be present in the LIVE modules (in addition to row 18's)
# ======================================================================================
def w47_code_presence():
    arm_src = inspect.getsource(G.run_admm_arm)
    s31c_src = inspect.getsource(G._s31c_interface_detail)
    disp_src = inspect.getsource(srp._process_scenario_dispersion_results)
    hook_src = inspect.getsource(H._config_hook_factory)
    hook_sig = inspect.signature(H._config_hook_factory).parameters
    child_src = inspect.getsource(H._child_real)
    w10_arm_src = inspect.getsource(W10.run_arm)
    return {
        'run_admm_arm_passes_optimization_results_only_to_declaring_hooks': (
            "if 'optimization_results' in inspect.signature(post_run_hook).parameters:" in arm_src),
        'run_admm_arm_passes_primal_evolution_only_to_declaring_hooks': (
            "if 'primal_evolution' in inspect.signature(post_run_hook).parameters:" in arm_src),
        'gate_post_run_hook_declares_neither_new_parameter': (
            'def post_run_hook(planning, sed, models, rows, report, out_dir, label, state=None):' in w10_arm_src),
        's31c_flexibility_volumes_probability_weighted': (
            'abs_delta_p_sum_mw += omega_w47 * abs(delta_p_mw)' in s31c_src
            and 'abs_delta_q_sum_mvar += omega_w47 * abs(delta_q_mvar)' in s31c_src),
        'dispersion_shared_ess_read_at_the_na_pair': (
            'na=sess_na_scenario(model)' in disp_src
            and 'model.shared_es_pnet[e, na[0], na[1], p]' in disp_src),
        'config_hook_factory_takes_the_premium_argument': 'interface_deviation_premium' in hook_sig,
        'config_hook_factory_premium_default_is_none': hook_sig['interface_deviation_premium'].default is None,
        'config_hook_premium_branch_guarded': 'if premium is not None:  # W47' in hook_src,
        'config_hook_derived_branch_guarded': 'if derived is not None:  # W47' in hook_src,
        'child_capture_only_when_declared': (
            'capture_multiscenario = derived is not None or premium is not None' in child_src),
        'harness_derived_instance_validator_present': callable(getattr(H, 'validate_derived_instance', None)),
        'harness_terminal_capture_present': callable(getattr(H, 'write_multiscenario_terminal', None)),
    }


def combined_code_presence():
    row18 = S51G_ROW18_PRESENCE()
    w47 = w47_code_presence()
    return {**{f'row18:{k}': v for k, v in row18.items()}, **{f'w47:{k}': v for k, v in w47.items()}}


S51G_ROW18_PRESENCE = S51G.row18_code_presence   # the committed W39 checks, captured before substitution


# ======================================================================================
#  the additional comparison -- the whole arm vs the committed W39 gate-1 arm
# ======================================================================================
def _check_w39_arm_pins():
    problems = []
    manifest_path = os.path.join(REPO, W39_ARM['manifest'])
    got = CP._sha256_file(manifest_path) if os.path.isfile(manifest_path) else None
    if got != W39_ARM['manifest_sha256']:
        problems.append(f"W39 gate manifest sha256 {got} != pin {W39_ARM['manifest_sha256']}")
        return problems, {}
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    for fname in COMPARED_JSON + COMPARED_JSONL:
        rel = os.path.join(W39_ARM['arm_dir'], fname)
        path = os.path.join(REPO, rel)
        if rel not in manifest:
            problems.append(f'{rel} not in the W39 manifest')
        elif not os.path.isfile(path) or CP._sha256_file(path) != manifest[rel]:
            problems.append(f'{rel} does not hash to the W39 manifest')
    status = W10._git(['status', '--porcelain', '--', W39_ARM['gate_dir']])
    if status.strip():
        problems.append(f'W39 gate directory not clean in git:\n{status}')
    return problems, manifest


def compare_against_w39_arm(out_root):
    mine_dir = os.path.join(out_root, 'arm')
    w39_dir = os.path.join(REPO, W39_ARM['arm_dir'])
    with open(os.path.join(out_root, 'gate.json')) as handle:
        mine_ids = json.load(handle)['arm_summary']['working_dir_ids']
    with open(os.path.join(REPO, W39_ARM['gate_json'])) as handle:
        w39_ids = json.load(handle)['arm_summary']['working_dir_ids']
    # Same order in both groups: W10._normalize maps position i of either group to <ARM_TOKEN_i>.
    token_groups = [
        [os.path.abspath(mine_dir), W10._rel(mine_dir), mine_ids['run'], mine_ids['precheck'], ARM],
        [os.path.abspath(w39_dir), W10._rel(w39_dir), w39_ids['run'], w39_ids['precheck'], W39_ARM['arm_name']],
    ]
    files = {}
    for fname in COMPARED_JSON:
        a = CP._load_json(os.path.join(w39_dir, fname))
        b = CP._load_json(os.path.join(mine_dir, fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: w39={a is not None} w48={b is not None}'}
            continue
        files[fname] = W10.classify(fname, a, b, token_groups)
    for fname in COMPARED_JSONL:
        a = CP._load_jsonl(os.path.join(w39_dir, fname))
        b = CP._load_jsonl(os.path.join(mine_dir, fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: w39={a is not None} w48={b is not None}'}
            continue
        entry = W10.classify(fname, a, b, token_groups)
        entry['n_rows'] = {'w39': len(a), 'w48': len(b)}
        files[fname] = entry
    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in files.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    pickles = {name: (CP._sha256_file(os.path.join(d, f'esso_models_{W10.ARM_LABEL}.pkl'))
                      if os.path.isfile(os.path.join(d, f'esso_models_{W10.ARM_LABEL}.pkl')) else None)
               for name, d in (('w39', w39_dir), ('w48', mine_dir))}
    pickles['sha256_equal'] = pickles['w39'] is not None and pickles['w39'] == pickles['w48']
    return {
        'reference_arm': W39_ARM,
        'comparator': ("W10.classify (BY IMPORT): CP._diff bitwise, CP.EXCLUDE_KEY_NAMES / EXCLUDE_DOTTED_SUFFIXES, "
                       "rule_eleven_checklist subtree = provenance, arm-path tokens normalized, recourse-jump "
                       "tie order reclassified by TC; everything else genuine. CP._diff's 'legacy' = the W39 arm, "
                       "'lightweight' = this arm"),
        'token_groups': token_groups,
        'files_compared': list(COMPARED_JSON) + list(COMPARED_JSONL),
        'per_file': files, 'totals': totals,
        'all_files_present': all('error' not in v for v in files.values()),
        'zero_genuine_diffs': totals['genuine'] == 0,
        'esso_models_pickle_sha256_informational': pickles,
    }


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)
    w39_problems, _manifest = _check_w39_arm_pins()
    if w39_problems:
        for p in w39_problems:
            _log(f'[PRECONDITION FAILED] {p}')
        return 1
    _log(f"committed W39 arm pinned: {W39_ARM['arm_dir']} (manifest sha256 {W39_ARM['manifest_sha256']})")
    presence_w47 = w47_code_presence()
    _log(f'W47 presence checks: {sum(presence_w47.values())}/{len(presence_w47)} true')

    # ---- the declared substitutions, and no others ----
    S51G.STAGE = STAGE
    S51G.SCHEMA = SCHEMA
    S51G.ARM = ARM
    S51G.OUT_REL = OUT_REL
    S51G.EXTRA_CLEAN_FILES = EXTRA_CLEAN_FILES
    S51G.row18_code_presence = combined_code_presence
    W35G.EXTRA_FORBIDDEN = EXTRA_FORBIDDEN

    status = S51G.main()   # W39 gate 1 as committed (-> W35 gate main); exits 1 on a precondition refusal
    _log(f'W39/W35 gate verdict (exit status): {status}')

    comparison = compare_against_w39_arm(out_root)
    _log(f"vs committed W39 arm: totals {comparison['totals']}; all files present "
         f"{comparison['all_files_present']}; esso pickle sha equal "
         f"{comparison['esso_models_pickle_sha256_informational']['sha256_equal']} (informational)")
    for fname, v in comparison['per_file'].items():
        _log(f"   {fname}: {v.get('n', v.get('error'))}")

    gate_items = {
        'w39_w35_gate_pass': status == 0,
        'w47_code_present_in_the_modules_that_ran': all(presence_w47.values()),
        'vs_committed_w39_arm_all_files_present': comparison['all_files_present'],
        'vs_committed_w39_arm_zero_genuine_diffs': comparison['zero_genuine_diffs'],
    }
    gate_pass = all(gate_items.values())
    payload = {
        'schema': SCHEMA + '_addendum', 'stage': STAGE,
        'authority': ['Planner task W48 ruling 2', 'PLANNER_BRIEF_2026-09-13.md Addenda 36 and 39'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'imported_gate_sha256': {name: CP._sha256_file(os.path.join(REPO, name)) for name in (
            'p515_s51_srp1_bitwise_gate.py', 'p515_s50_generalization_gate.py', 'p515_s49_memory_fix_gate.py',
            'p515_s45_snapshot_off_two_cycle_gate.py')},
        'w47_module_sha256': {name: CP._sha256_file(os.path.join(REPO, name)) for name in (
            'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py', 'shared_resources_planning.py')},
        'declared_substitutions': {
            '1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
            '2_extra_clean_files': list(EXTRA_CLEAN_FILES),
            '3_presence': 'W39 row 18 checks + w47_code_presence',
            '4_extra_forbidden': list(EXTRA_FORBIDDEN)},
        'w47_code_presence_asserted_before_run': presence_w47,
        'comparison_vs_committed_w39_arm_GATING': comparison,
        'not_covered_by_this_gate': (
            "_child_real's W47 branches (derived-instance install, terminal-phase lock, multi-scenario capture, "
            'workbook) are entered only when derived_instance or interface_deviation_premium is declared, and '
            'W10.run_arm reproduces the child wiring rather than calling _child_real; '
            '_process_scenario_dispersion_results runs only in the workbook writer, which a two-cycle arm does '
            'not call. Both rest on the W47 zero-solve checks (ae951d31: C, D, F).'),
        'guard_counts_at_end': dict(GUARD.counts),
        'gate_items': gate_items, 'gate_pass': gate_pass,
    }
    addendum_path = os.path.join(out_root, 'w48_gate_addendum.json')
    W10._refuse_overwrite(addendum_path)
    with open(addendum_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'w48_manifest_sha256.json')
    W10._refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    for key, value in gate_items.items():
        _log(f'   {key}: {value}')
    _log(f'wrote {W10._rel(addendum_path)}, {W10._rel(manifest_path)} ({len(manifest)} files)')
    _log(f'GATE_PASS={gate_pass}')
    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
