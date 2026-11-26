"""
P5.15 Addendum 38 (task W39) -- GATE 1 of 3: the SRP1 TWO-CYCLE BITWISE IDENTITY for
row 18 as signed, against the COMMITTED baseline C* trajectory.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[0]` -- "SRP1 two-cycle bitwise identity (row 18 vacuous at one scenario)";
PLANNER_BRIEF_2026-09-13.md Addenda 37 and 38.

WHAT IS UNDER TEST. The row 18 implementation changes five production files
(`model_construction_helpers.py`, `shared_resources_planning.py`, `network.py`,
`admm_parameters.py`, `p515_s31c_evaluate.py`). On SRP1 -- ONE market and ONE operation
scenario -- every one of those changes is claimed to be a no-op:

  * row 18 and the interface-voltage pin are NOT CONSTRUCTED at one scenario
    (`add_scenario_commitment_terms` returns before building anything), so the objective
    expression is the pre-change expression and the objective is never rebuilt;
  * the shared-ESS non-anticipativity alias resolves to `(s_m, s_o)` itself at one
    scenario, so every rule below `sess_na_scenario` is literally the pre-change rule;
  * the TSO interface-delta alias likewise resolves to the only scenario pair, and the
    freeing loop's new guard is entered on every iteration it used to be entered on;
  * the settlement SPLIT changes only REPORTING: at one market scenario the covariance
    part is identically zero and the contracted part IS the whole settlement, so the
    quantity Q(x) excludes is unchanged;
  * `ADMMParameters.interface_deviation_premium` defaults to alpha = 0, so no campaign
    or case file that omits the key can activate row 18.

This gate is the solve-bearing test of all five claims at once: two cycles at C* under
the baseline configuration must reproduce the committed baseline re-certification's
`cycle_trajectory[:2]` and the rows-derived top-level fields with ZERO diffs.

HOW IT IS BUILT. Everything is the COMMITTED W35 gate (`p515_s50_generalization_gate.py`,
itself built on W10/W32) BY IMPORT, with exactly four declared substitutions and no
others:
  1. `STAGE` / `SCHEMA` / `ARM` / `OUT_REL` -- this stage's identifiers and its own
     write-once output root (so no committed artifact can be overwritten; CLAUDE.md's
     "never re-run a harness onto an artifact a committed report cites");
  2. `EXTRA_CLEAN_FILES` -- extended with this stage's files, so the gate refuses to run
     against an uncommitted edit of anything row 18 touches;
  3. `w35_code_presence` -> `row18_code_presence` below: the LIVE modules are asserted to
     carry the Addendum 38 implementation (a stale import cannot pass this gate), and the
     retired quadratic is asserted to be present-but-unwired;
  4. an ARMED tripwire on `_add_tso/dso_scenario_deviation_penalty` for the whole run:
     the claim "the retired quadratic is called on no production path" is armed, never
     asserted (CLAUDE.md rule six). Any call is a gate failure with its stack.
Everything else -- the armed `SolveProfileGuard`, the preconditions and lock refusals,
the capture-path checklist, the per-EVENT solve reconciliation, the committed reference
and its sha256 pin, the trajectory field table, the manifest -- is the committed gate's,
unchanged.

GATE: every item of the W35 gate's `gate_items` (cycles_run == 2; zero diffs against the
committed reference; zero field mismatches; the event-level solve identity; the arm's own
instance-derived solve profile; GUARD.verify exactly; no blocked solver call; shared
FrozenSMOPF tree untouched; the ESS-ageing declaration verified in the child) PLUS
`row18_code_present_in_the_modules_that_ran` and `retired_quadratic_never_called`.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_srp1_bitwise_gate.py \\
        > data/SRP1/Results/P515S51/srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/srp1_bitwise_gate/{gate.json, gate.md,
manifest_sha256.json, arm/}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import inspect
import os
import sys
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s50_generalization_gate as W35G  # noqa: E402 -- imports W32 -> W10 (installs the armed guard)

W10 = W35G.W10
GUARD = W10.GUARD

import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402

STAGE = ('P5.15 Addendum 38 W39 gate 1 -- SRP1 two-cycle bitwise identity: row 18 as '
         'signed is vacuous at one scenario, vs committed baseline C*')
SCHEMA = 'p515_s51_srp1_bitwise_gate_v1'
ARM = 's51row18gate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'srp1_bitwise_gate')
EXTRA_CLEAN_FILES = tuple(W35G.EXTRA_CLEAN_FILES) + (
    'admm_parameters.py', 'p515_s31c_evaluate.py', 'p515_s51_row18_zero_solve_checks.py',
    os.path.basename(__file__))


# ======================================================================================
#  substitution 3 -- the Addendum 38 code must be present in the LIVE modules
# ======================================================================================
def row18_code_presence():
    wiring = inspect.getsource(MCH.add_scenario_commitment_terms)
    rule = inspect.getsource(MCH.objective_function_rule)
    sess = inspect.getsource(MCH.sess_na_scenario)
    soc = inspect.getsource(MCH.sess_soc_rule)
    delta = inspect.getsource(MCH.interface_pf_p_transmission_def)
    balance = inspect.getsource(MCH.compute_node_load)
    recourse = inspect.getsource(srp._get_operational_recourse_components)
    tso_build = inspect.getsource(srp.create_transmission_network_model)
    dso_build = inspect.getsource(srp.create_distribution_networks_models_sequential)
    defaults = ADMMParameters().interface_deviation_premium
    return {
        # row 18 itself
        'wiring_function_present': callable(getattr(MCH, 'add_scenario_commitment_terms', None)),
        'wiring_returns_before_building_at_one_scenario': 'if n_scenarios == 1:' in wiring,
        'wiring_builds_lp_pair': ('row18_dev_p_up' in wiring and 'row18_dev_p_down' in wiring
                                  and 'row18_dev_q_up' in wiring and 'row18_dev_q_down' in wiring),
        'charge_is_alpha_times_pibar_times_baseMVA': (
            'model.row18_alpha * model.row18_premium[p] * network.baseMVA' in wiring),
        'deviation_is_against_the_coupled_expectation_var': (
            'm.expected_interface_pf_p[p]' in inspect.getsource(MCH.row18_interface_deviation_p_rule)),
        'row18_is_inside_objective_function_rule': "hasattr(model, 'row18_deviation_charge')" in rule,
        'voltage_pin_is_inside_objective_function_rule': "hasattr(model, 'scenario_voltage_pin')" in rule,
        'wiring_rebuilds_the_objective': 'model.del_component(model.objective)' in wiring,
        'alpha_defaults_inactive': (defaults['alpha'] == 0.0 and defaults['floor'] is None
                                    and defaults['source'] == 'default'),
        # (A) TSO pinned
        'tso_delta_is_scenario_free_in_the_interface_expression': (
            'm.interface_delta_p[dn, s_m0, s_o0, p]' in delta),
        'tso_delta_is_scenario_free_in_the_node_balance': (
            'model.interface_delta_p[dn, s_m0, s_o0, p]' in balance),
        'tso_freeing_loop_is_guarded_by_the_first_scenario_pair': (
            '(s_m, s_o) == sess_na_scenario(tso_model[year][day])' in tso_build),
        # (B) storage non-anticipativity
        'sess_alias_present': 'next(iter(m.scenarios_market))' in sess,
        'sess_rows_skip_duplicates': 'sess_row_is_duplicate' in soc,
        'sess_balance_row_is_scenario_free': 'model.shared_es_pnet[e, s_m0, s_o0, p]' in balance,
        # (C) settlement split
        'contracted_expression_present': callable(
            getattr(MCH, 'interface_energy_settlement_contracted', None)),
        'recourse_excludes_only_the_contracted_part': (
            "part='contracted'" in recourse and 'interface_settlement_covariance_total' in recourse),
        'recourse_reports_the_identity_residual': (
            'interface_settlement_identity_residual' in recourse),
        # (D) voltage pin excluded from Q(x)
        'recourse_excludes_the_voltage_pin': '_get_operational_voltage_pin_blocks' in recourse,
        'per_scenario_voltage_mismatch_reported': callable(
            getattr(srp, '_get_local_scenario_voltage_mismatch', None)),
        # the retired quadratic: present but not wired
        'retired_quadratic_retained': (callable(getattr(srp, '_add_tso_scenario_deviation_penalty', None))
                                       and callable(getattr(srp, '_add_dso_scenario_deviation_penalty', None))),
        'retired_quadratic_unwired_in_the_tso_builder': (
            '_add_tso_scenario_deviation_penalty(' not in tso_build
            and 'add_scenario_commitment_terms(' in tso_build),
        'retired_quadratic_unwired_in_the_dso_builder': (
            '_add_dso_scenario_deviation_penalty(' not in dso_build
            and 'add_scenario_commitment_terms(' in dso_build),
        # alpha threading
        'alpha_threaded_from_admm_parameters': (
            'admm_parameters.interface_deviation_premium'
            in inspect.getsource(srp._run_operational_planning)),
    }


# ======================================================================================
#  substitution 4 -- ARMED tripwire on the retired quadratic (never asserted)
# ======================================================================================
class _RetiredQuadraticTripwire:
    def __init__(self):
        self.calls = []
        self._originals = {}

    def install(self):
        for name in ('_add_tso_scenario_deviation_penalty', '_add_dso_scenario_deviation_penalty'):
            original = getattr(srp, name)
            self._originals[name] = original

            def wrapped(*args, __name=name, __original=original, **kwargs):
                self.calls.append({'function': __name,
                                   'stack': ''.join(traceback.format_stack(limit=25))})
                return __original(*args, **kwargs)

            setattr(srp, name, wrapped)
        return self

    def uninstall(self):
        for name, original in self._originals.items():
            setattr(srp, name, original)


TRIPWIRE = _RetiredQuadraticTripwire()


def main():
    # ---- the four declared substitutions, and no others ----
    W35G.STAGE = STAGE
    W35G.SCHEMA = SCHEMA
    W35G.ARM = ARM
    W35G.OUT_REL = OUT_REL
    W35G.EXTRA_CLEAN_FILES = EXTRA_CLEAN_FILES
    W35G.w35_code_presence = row18_code_presence

    TRIPWIRE.install()
    try:
        status = W35G.main()
    finally:
        TRIPWIRE.uninstall()

    # The tripwire verdict is APPENDED to the committed gate's own verdict: the gate as a
    # whole passes only if the quadratic was never called.
    quadratic_never_called = not TRIPWIRE.calls
    print(f'[W39-gate1] retired_quadratic_never_called = {quadratic_never_called} '
          f'({len(TRIPWIRE.calls)} calls)')
    for call in TRIPWIRE.calls:
        print(f"[W39-gate1] UNEXPECTED CALL to {call['function']}:\n{call['stack']}")

    verdict_path = os.path.join(REPO, OUT_REL, 'row18_gate_addendum.json')
    if os.path.isdir(os.path.dirname(verdict_path)):
        import json
        W10._refuse_overwrite(verdict_path)
        with open(verdict_path, 'w') as handle:
            json.dump({
                'schema': SCHEMA + '_addendum',
                'stage': STAGE,
                'row18_code_presence_asserted_before_run': row18_code_presence(),
                'retired_quadratic_tripwire': {
                    'mechanism': ('ARMED for the whole run around both retired functions '
                                  '(CLAUDE.md rule six: armed, never asserted)'),
                    'n_calls': len(TRIPWIRE.calls), 'calls': TRIPWIRE.calls},
                'retired_quadratic_never_called': quadratic_never_called,
                'w35_gate_exit_status': status,
                'gate_pass': status == 0 and quadratic_never_called,
                'guard_counts': dict(GUARD.counts),
            }, handle, indent=1, default=str)
        print(f'[W39-gate1] wrote {os.path.relpath(verdict_path, REPO)}')

    return 0 if (status == 0 and quadratic_never_called) else 1


if __name__ == '__main__':
    sys.exit(main())
