"""
P5.15 Addendum 40 ruling 2 (tasks W51 -> W56) -- the SRP1 TWO-CYCLE BITWISE IDENTITY gate for "row 18 INACTIVE at the
initialisation solve, activated with the settlement weight", against the SAME committed baseline C* reference as
the W35 / W39 / W48 gates, plus the whole arm against the committed W48 gate arm.

Authority: frozen spec v23 `data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json`, `ruling2_init_fix.gate`
("SRP1 two-cycle bitwise identity (inert at one scenario)"); PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2;
Planner task W51 ("reuse the committed gate conventions by import, with the declared solve count reconciled per
event and an armed bounded guard, plus code-presence assertions for this change").

RECORDED PREDICTION (spec v23 `ruling2_init_fix.predictions_recorded_before_run.planner[0]`): "the SRP1 bitwise gate
passes unchanged (nothing is wired at one scenario)".

WHAT IS UNDER TEST. W51 (d7962d4e) and W54 (d573e145, the STRUCTURAL form that replaced W51's Param-only form;
frozen spec `data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v2_5123e67b.json`) changed
`shared_resources_planning.py` only:
  * `_set_row18_inactive_for_initialisation` -- called by both DSO initialisation builders after
    `add_scenario_commitment_terms`, before the initialisation solve; per index deactivates `row18_dev_{p,q}_def`
    then fixes the deviation pair at 0; returns at once where `row18_alpha` does not exist;
  * `_activate_row18_with_settlement` -- called in `_prepare_distribution_objectives_for_admm` after the
    settlement weight is set to 1; per index sets the minimal split, unfixes the pair, then activates the row;
    returns at once where `row18_alpha` does not exist (W51's `row18_alpha_admm` marker no longer exists).
At SRP1 (ONE market x ONE operation scenario) `add_scenario_commitment_terms` constructs nothing, so both are
claimed to be no-ops on every block. The zero-solve checks (`p515_s53_init_fix_zero_solve_checks.py`, check D)
show it structurally at 1 x 1 without solving; this gate is the solve-bearing test.
W56 UPDATE (Planner task W56): substitution 3's `w51_code_presence` and the armed counter's acting markers are
re-targeted to the structural mechanism; nothing else in this gate changed. WRITTEN, NOT RUN at W56.
W57 UPDATE (Planner task W57 item 4): `_activate_row18_with_settlement` now evaluates each defining row's body after
activation and raises above `_ROW18_ACTIVATION_BODY_TOL` (W57 item 2; frozen spec v3). The 19 W56 presence checks
still hold on the live module unchanged; two are ADDED (the body check follows activation, in order; the tolerance
is the declared 1e-9). Nothing else in this gate changed. WRITTEN, NOT RUN at W57.

HOW IT IS BUILT. The committed W48 gate (`p515_s52_srp1_bitwise_gate.py`) BY IMPORT, which imports the W39 gate 1
(`p515_s51_srp1_bitwise_gate.py`) -> the W35 gate (`p515_s50_generalization_gate.py`) -> W32 -> W10, whose armed
`SolveProfileGuard` (W10.GUARD, bounded, `p514_n_instrumented_cstar.PERMITTED` call sites) is installed at import
and verified EXACTLY against the per-EVENT reconciled count (`p515_s44_scale_measurement.
event_level_solve_reconciliation`). DECLARED BEFORE THE RUN (W10.derive_solves_from_case_file on
data/SRP1/SRP1.json): 3 years x 4 days = 12 (year, day) blocks x (1 TSO + 3 DSO) = 48 network solves + 3 ESSO = 51
solves per cycle; base = 51 x (cap 2 + 1 initialisation) = 153; the gate requires observed == 153 + every attempted
retry (recovered or not) and GUARD.verify(that) exactly. Everything else -- preconditions and lock refusals (neither
`.p515_g_gate.lock` nor `.p515_s44_campaign.lock` may exist; no live forbidden process), the capture-path
checklist, the committed reference and its sha256 pin (`W32.BASELINE_REFERENCE`: P515S47/campaign_s47_recert/evals/
070f833e1e318f85_c_star/g_s39_D.json, cycle_trajectory[:2] + rows-derived top-level fields, GATING), the trajectory
field table, the ARMED tripwire on the retired quadratic (W39) -- is the committed gates', unchanged, with exactly
these declared substitutions and no others:
  1. identifiers `STAGE` / `SCHEMA` / `ARM` / `OUT_REL` (W39 and W48 module globals) -- this stage's own
     write-once output root, so no committed artifact can be overwritten;
  2. `EXTRA_CLEAN_FILES` -- W48's list plus the W51 zero-solve checks and this file (shared_resources_planning.py,
     model_construction_helpers.py and every imported gate script are already in it), so the gate refuses to run
     against an uncommitted edit of any of them;
  3. the live-module presence assertion = W39's row 18 checks + W48's W47 checks + `w51_code_presence` below
     (W56: the structural form; a stale import of pre-W51 code, or of W51's Param-only form, cannot pass);
  4. `EXTRA_FORBIDDEN` -- extended with 'p515_s53_' (the F2 certificate campaign), so a live campaign process
     refuses the gate (this process and its launching shells are excluded as ancestors, the committed convention);
  5. the reference arm of W48's whole-arm comparison -> the COMMITTED W48 gate arm
     (`data/SRP1/Results/P515S52/srp1_bitwise_gate/arm`, identical configuration -- C*, baseline, cap 2, snapshots
     on -- run at fe0319d6 and committed at e75a575e, manifest sha256 pinned below), compared with W10's own
     comparator (`W10.classify`) over W10's gating file lists, every committed file first checked against the W48
     manifest.
PLUS ONE ARMED COUNTER, declared here before the run (CLAUDE.md rule six: armed, never asserted): both new functions
are wrapped for the whole run (pass-through: the original is called unchanged); the gate requires EXACTLY
n_dso_blocks = 3 DSOs x 12 (year, day) = 36 calls of each (derived from the case file before the run -- too few
fails as loudly as too many, since it would mean the path under test did not run) and ZERO calls on a block
carrying `row18_alpha` (W56: the one marker both functions now test -- i.e. both returned without acting on every
block).

GATE = the W39/W35 gate verdict (every W35 item + row 18 presence + retired quadratic never called) AND all
presence checks (row 18, W47, W51) AND zero genuine diffs against the committed W48 arm with every compared file
present AND the armed W51 counter exact with zero acting calls.

NOT COVERED, stated rather than implied: the fix's effect above one scenario (the change is designed to alter the
initialisation solve there -- it is the subject of the alpha row, not of this gate; the W51 zero-solve checks A-C
cover its structure, per index in both phases); the parallel DSO builder (`create_distribution_network_model`) runs only with
`parallel_execution` true, which this arm does not use (zero-solve check A2 exercised it in-process).

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_srp1_bitwise_gate.py \\
        > data/SRP1/Results/P515S53/srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/srp1_bitwise_gate/{gate.json, gate.md, manifest_sha256.json,
row18_gate_addendum.json, w51_gate_addendum.json, w51_manifest_sha256.json, arm/}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import hashlib
import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s52_srp1_bitwise_gate as S52G  # noqa: E402 -- imports W39 -> W35 -> W32 -> W10 (installs the armed guard)

S51G = S52G.S51G
W35G = S52G.W35G
W10 = S52G.W10
GUARD = W10.GUARD
CP = W10.CP

import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

STAGE = ('P5.15 Addendum 40 ruling 2 W51 -- SRP1 two-cycle bitwise identity: row 18 inactive at the '
         'initialisation solve is inert at one scenario, vs committed baseline C* and the committed W48 arm')
SCHEMA = 'p515_s53_srp1_bitwise_gate_v1'
ARM = 's53w51gate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'srp1_bitwise_gate')
EXTRA_CLEAN_FILES = tuple(S52G.EXTRA_CLEAN_FILES) + (   # already names srp, MCH and every imported gate script
    'p515_s53_init_fix_zero_solve_checks.py', os.path.basename(__file__))
EXTRA_FORBIDDEN = tuple(S52G.EXTRA_FORBIDDEN) + ('p515_s53_',)

# The committed W48 gate arm: same configuration, run at fe0319d6 (pre-W51 code), committed at e75a575e.
W48_ARM = {
    'gate_dir': os.path.join('data', 'SRP1', 'Results', 'P515S52', 'srp1_bitwise_gate'),
    'arm_dir': os.path.join('data', 'SRP1', 'Results', 'P515S52', 'srp1_bitwise_gate', 'arm'),
    'manifest': os.path.join('data', 'SRP1', 'Results', 'P515S52', 'srp1_bitwise_gate', 'manifest_sha256.json'),
    'manifest_sha256': '9a5d6b78bfdb4738e83389878eba3236b298a4e5e5b05308a48b44ebe094ba69',
    'gate_json': os.path.join('data', 'SRP1', 'Results', 'P515S52', 'srp1_bitwise_gate', 'gate.json'),
    'arm_name': S52G.ARM,
    'commit': 'e75a575e (run at fe0319d6)',
}

# Source pins taken at the W51 parent (e1f98be8) and unchanged by W51 -- recorded before the run.
PIN_MODEL_CONSTRUCTION_HELPERS_FILE_SHA256 = 'a20a224f11008de8c0206484583b00253677267d616e1051bb1583791ed47f64'
PIN_BENCHMARK_SOURCE_SHA256 = '7b225fd08c70fee21d02b9f2eb428aa08f11bfba573c2d0edf0f580b1cf6cccf'
PIN_ADD_SCENARIO_COMMITMENT_TERMS_SOURCE_SHA256 = 'fb9e762472243af6f65ab3708a29cf11edc73c5314e7ce4ff3764d504b89f084'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W51-gate] {msg}', flush=True)


def _src_sha(fn):
    return hashlib.sha256(inspect.getsource(fn).rstrip('\n').encode()).hexdigest()


def _ordered(text, *needles):
    positions = [text.find(n) for n in needles]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


# ======================================================================================================================
#  substitution 3 -- the W51 change must be present in the LIVE modules
# ======================================================================================================================
# W56: the structural mechanism's family table (W54), declared here and compared with the live module.
EXPECTED_ROW18_DEVIATION_FAMILIES = (
    ('row18_dev_p_def', 'row18_dev_p_up', 'row18_dev_p_down', 'pg_adn', 'expected_interface_pf_p'),
    ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down', 'qg_adn', 'expected_interface_pf_q'),
)
# W57: the activation body-check tolerance (production `_ROW18_ACTIVATION_BODY_TOL`), declared here.
EXPECTED_ROW18_ACTIVATION_BODY_TOL = 1e-9


def w51_code_presence():
    """W51 -> W56: the row 18 initialisation fix in its STRUCTURAL form (W54) must be present in the live modules.
    (Name kept: it is substitution 3's hook and the addendum's key prefix.)"""
    inactive = getattr(srp, '_set_row18_inactive_for_initialisation', None)
    activate = getattr(srp, '_activate_row18_with_settlement', None)
    inactive_src = inspect.getsource(inactive) if callable(inactive) else ''
    activate_src = inspect.getsource(activate) if callable(activate) else ''
    seq_src = inspect.getsource(srp.create_distribution_networks_models_sequential)
    par_src = inspect.getsource(srp.create_distribution_network_model)
    prep_src = inspect.getsource(srp._prepare_distribution_objectives_for_admm)
    run_src = inspect.getsource(srp._run_operational_planning)
    srp_src = inspect.getsource(srp)
    return {
        'family_table_present_and_as_declared': (
            getattr(srp, '_ROW18_DEVIATION_FAMILIES', None) == EXPECTED_ROW18_DEVIATION_FAMILIES),
        'inactive_function_present': callable(inactive),
        'inactive_returns_where_row18_not_wired': "if not hasattr(model, 'row18_alpha'):\n        return" in inactive_src,
        'inactive_deactivates_row_before_fixing_pair_at_zero': _ordered(
            inactive_src, 'for row_name, up_name, down_name, _flow_name, _expectation_name in _ROW18_DEVIATION_FAMILIES:',
            'row[index].deactivate()', 'up[index].fix(0.0)', 'down[index].fix(0.0)'),
        'inactive_never_writes_alpha': ('row18_alpha.set_value' not in inactive_src
                                        and 'row18_alpha_admm' not in inactive_src),
        'activate_function_present': callable(activate),
        'activate_returns_where_row18_not_wired': "if not hasattr(model, 'row18_alpha'):\n        return" in activate_src,
        'activate_refuses_unless_initialisation_state': _ordered(
            activate_src, 'if row[index].active or not up[index].fixed or not down[index].fixed:',
            'raise RuntimeError(', 'e = pe.value(flow[s_m, s_o, p]) - pe.value(expectation[p])'),
        'activate_minimal_split_then_unfix_then_activate': _ordered(
            activate_src, 'e = pe.value(flow[s_m, s_o, p]) - pe.value(expectation[p])',
            'up[index].set_value(e if e > 0.0 else 0.0)', 'down[index].set_value(-e if e < 0.0 else 0.0)',
            'up[index].unfix()', 'down[index].unfix()', 'row[index].activate()'),
        'activate_never_writes_alpha': ('row18_alpha.set_value' not in activate_src
                                        and 'row18_alpha_admm' not in activate_src),
        'activate_checks_defining_row_bodies_after_activation': _ordered(
            activate_src, 'row[index].activate()',
            'body_residual = pe.value(row[index].body) - pe.value(row[index].upper)',
            'if not abs(body_residual) <= _ROW18_ACTIVATION_BODY_TOL:', 'defining-row body of'),
        'activation_body_tolerance_as_declared': (
            getattr(srp, '_ROW18_ACTIVATION_BODY_TOL', None) == EXPECTED_ROW18_ACTIVATION_BODY_TOL),
        'w51_param_retired_from_module': 'row18_alpha_admm' not in srp_src,
        'sequential_builder_deactivates_after_wiring_before_solve': _ordered(
            seq_src, 'add_scenario_commitment_terms(', '_set_row18_inactive_for_initialisation(dso_model[year][day])',
            'results[node_id] = distribution_network.optimize(dso_model)'),
        'parallel_builder_deactivates_after_wiring_before_solve': _ordered(
            par_src, 'add_scenario_commitment_terms(', '_set_row18_inactive_for_initialisation(dso_model[year][day])',
            'res = distribution_network.optimize(dso_model)'),
        'prepare_activates_with_the_settlement_weight': _ordered(
            prep_src, 'interface_settlement_weight.set_value(1.00)',
            '_activate_row18_with_settlement(dso_model[year][day])'),
        'run_passes_the_run_alpha_to_the_build': "premium_alpha=interface_premium['alpha']," in run_src,
        'run_prepares_before_scale_and_admm_update': _ordered(
            run_src, 'create_distribution_networks_models(',
            '_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)',
            '_compute_common_admm_objective_scale(', 'update_distribution_models_to_admm('),
        'model_construction_helpers_file_unchanged_by_w51_w54': (
            CP._sha256_file(os.path.join(REPO, 'model_construction_helpers.py'))
            == PIN_MODEL_CONSTRUCTION_HELPERS_FILE_SHA256),
        'add_scenario_commitment_terms_source_unchanged': (
            _src_sha(MCH.add_scenario_commitment_terms) == PIN_ADD_SCENARIO_COMMITMENT_TERMS_SOURCE_SHA256),
        'uncoordinated_benchmark_source_unchanged': (
            _src_sha(srp._run_operational_planning_without_coordination) == PIN_BENCHMARK_SOURCE_SHA256),
    }


W48_COMBINED_PRESENCE = S52G.combined_code_presence   # the committed W39 + W47 checks, captured before substitution


def combined_code_presence():
    prior = W48_COMBINED_PRESENCE()
    w51 = w51_code_presence()
    return {**prior, **{f'w51:{k}': v for k, v in w51.items()}}


# ======================================================================================================================
#  the armed W51 counter (pass-through; declared count checked exactly)
# ======================================================================================================================
class _W51CallCounter:
    NAMES = ('_set_row18_inactive_for_initialisation', '_activate_row18_with_settlement')
    # W56: both functions now act exactly where `row18_alpha` exists (W51's `row18_alpha_admm` is retired), so a call
    # on a block carrying `row18_alpha` is an ACTING call; at SRP1 (one scenario) the gate requires none.
    MARKERS = {'_set_row18_inactive_for_initialisation': 'row18_alpha',
               '_activate_row18_with_settlement': 'row18_alpha'}

    def __init__(self):
        self.calls = {n: 0 for n in self.NAMES}
        self.acting = {n: [] for n in self.NAMES}
        self._originals = {}

    def install(self):
        for name in self.NAMES:
            original = getattr(srp, name)
            self._originals[name] = original

            def wrapped(model, *args, __name=name, __original=original, **kwargs):
                self.calls[__name] += 1
                if hasattr(model, self.MARKERS[__name]):
                    self.acting[__name].append({'block': str(getattr(model, 'name', None)),
                                                'stack': ''.join(traceback.format_stack(limit=20))})
                return __original(model, *args, **kwargs)

            setattr(srp, name, wrapped)
        return self

    def uninstall(self):
        for name, original in self._originals.items():
            setattr(srp, name, original)


COUNTER = _W51CallCounter()


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)

    # ---- substitution 5: the reference arm of the whole-arm comparison ----
    S52G.W39_ARM = W48_ARM
    arm_problems, _manifest = S52G._check_w39_arm_pins()
    if arm_problems:
        for p in arm_problems:
            _log(f'[PRECONDITION FAILED] {p}')
        return 1
    _log(f"committed W48 arm pinned: {W48_ARM['arm_dir']} (manifest sha256 {W48_ARM['manifest_sha256']})")

    presence_w51 = w51_code_presence()
    _log(f'W51 presence checks: {sum(presence_w51.values())}/{len(presence_w51)} true')
    if not all(presence_w51.values()):
        for k, v in presence_w51.items():
            if not v:
                _log(f'[PRECONDITION FAILED] W51 code not present in the live module: {k}')
        return 1

    declared = W10.derive_solves_from_case_file()
    n_dso = declared['n_networks'] - 1
    expected_w51_calls = n_dso * declared['n_year_day_blocks']
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle, base "
         f"{declared['declared_base_solves_per_arm']} (cap {declared['cap']} + initialisation), per-event "
         f'reconciled; W51 counter: exactly {expected_w51_calls} calls of each new function, 0 acting')

    # ---- substitutions 1-4, and no others ----
    S52G.STAGE = STAGE
    S52G.SCHEMA = SCHEMA
    S52G.ARM = ARM
    S52G.OUT_REL = OUT_REL
    S51G.STAGE = STAGE
    S51G.SCHEMA = SCHEMA
    S51G.ARM = ARM
    S51G.OUT_REL = OUT_REL
    S51G.EXTRA_CLEAN_FILES = EXTRA_CLEAN_FILES
    S51G.row18_code_presence = combined_code_presence
    W35G.EXTRA_FORBIDDEN = EXTRA_FORBIDDEN

    COUNTER.install()
    try:
        status = S51G.main()   # W39 gate 1 as committed (-> W35 gate main); exits 1 on a precondition refusal
    finally:
        COUNTER.uninstall()
    _log(f'W39/W35 gate verdict (exit status): {status}')
    _log(f'W51 counter: calls {COUNTER.calls}; acting {({k: len(v) for k, v in COUNTER.acting.items()})}')

    comparison = S52G.compare_against_w39_arm(out_root)   # reference = W48_ARM (substitution 5)
    _log(f"vs committed W48 arm: totals {comparison['totals']}; all files present "
         f"{comparison['all_files_present']}; esso pickle sha equal "
         f"{comparison['esso_models_pickle_sha256_informational']['sha256_equal']} (informational)")
    for fname, v in comparison['per_file'].items():
        _log(f"   {fname}: {v.get('n', v.get('error'))}")

    counter_ok = (all(n == expected_w51_calls for n in COUNTER.calls.values())
                  and all(not v for v in COUNTER.acting.values()))
    gate_items = {
        'w39_w35_gate_pass': status == 0,
        'w51_code_present_in_the_modules_that_ran': all(w51_code_presence().values()),
        'w51_counter_exact_and_never_acting': counter_ok,
        'vs_committed_w48_arm_all_files_present': comparison['all_files_present'],
        'vs_committed_w48_arm_zero_genuine_diffs': comparison['zero_genuine_diffs'],
    }
    gate_pass = all(gate_items.values())
    payload = {
        'schema': SCHEMA + '_addendum', 'stage': STAGE,
        'authority': ['Planner task W51', 'Planner task W56 (structural form)',
                      'Planner task W57 (activation body check)',
                      'PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2',
                      'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json ruling2_init_fix',
                      'data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v2_5123e67b.json',
                      'data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v3_1064db50.json'],
        'recorded_prediction': 'the SRP1 bitwise gate passes unchanged (nothing is wired at one scenario)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'imported_gate_sha256': {name: CP._sha256_file(os.path.join(REPO, name)) for name in (
            'p515_s52_srp1_bitwise_gate.py', 'p515_s51_srp1_bitwise_gate.py', 'p515_s50_generalization_gate.py',
            'p515_s49_memory_fix_gate.py', 'p515_s45_snapshot_off_two_cycle_gate.py')},
        'module_sha256': {name: CP._sha256_file(os.path.join(REPO, name)) for name in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'admm_parameters.py',
            'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py')},
        'declared_substitutions': {
            '1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
            '2_extra_clean_files': list(EXTRA_CLEAN_FILES),
            '3_presence': 'W39 row 18 checks + W48 w47_code_presence + w51_code_presence',
            '4_extra_forbidden': list(EXTRA_FORBIDDEN),
            '5_reference_arm': W48_ARM},
        'declared_solve_profile': declared,
        'w51_code_presence_asserted_before_run': presence_w51,
        'w51_counter_ARMED': {
            'mechanism': ('pass-through wrappers on both new functions for the whole run (CLAUDE.md rule six: '
                          'armed, never asserted); count declared from the case file before the run'),
            'expected_calls_each': expected_w51_calls, 'calls': dict(COUNTER.calls),
            'acting_calls': COUNTER.acting, 'pass': counter_ok},
        'comparison_vs_committed_w48_arm_GATING': comparison,
        'not_covered_by_this_gate': (
            'the fix above one scenario (it is designed to change the initialisation solve there -- the subject of '
            'the alpha row; structure covered by the W51 zero-solve checks A-C); the parallel DSO builder, entered '
            'only with parallel_execution true (zero-solve check A2).'),
        'guard_counts_at_end': dict(GUARD.counts),
        'gate_items': gate_items, 'gate_pass': gate_pass,
    }
    addendum_path = os.path.join(out_root, 'w51_gate_addendum.json')
    W10._refuse_overwrite(addendum_path)
    os.makedirs(out_root, exist_ok=True)
    with open(addendum_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'w51_manifest_sha256.json')
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
