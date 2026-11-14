"""
P5.15 Step 3.6, Worker task W3 -- zero-solve checks for `p515_s36_step36_timing.py`.

Verifies, WITHOUT performing a single real solve (armed with
`p513_solve_profile_guard.SolveProfileGuard(permitted=())` for the WHOLE
script, `verify(expected_solves=0, expected_execs=0)` at the end):

  1. Every wrap target `p515_s36_step36_timing.recorder_installed` patches
     resolves to the object the real production caller actually invokes --
     for module-level functions, by `dis`-scanning the caller's bytecode for
     a matching `LOAD_GLOBAL` (proving the call is an unqualified global
     lookup in that module's namespace, so patching the module attribute
     affects it); for class-level methods, by real (zero-solve) instantiation
     of the owning class and an identity check on `type(instance).<method>`.
  2. `recorder_installed(...)`'s install/uninstall restores every patched
     attribute to EXACT (`is`) identity.
  3. Every wrapper preserves the return value and the exception of whatever
     it wraps, using trivial stand-in callables (never a real Pyomo solve,
     clone, or file-parse) -- including the frame-walk attribution logic
     (`_find_ancestor_frame`), exercised against SYNTHESIZED ancestor frames
     whose `co_filename`/`co_name`/locals are constructed with `compile()`
     to exactly match the real call sites' (file, function, local variable
     names) without executing any real production code.
  4. `analyze_phase_timing` reproduces a hand-computed toy example.
  5. (P5.15 Step 3.6 follow-up, item 3) `analyze_phase_timing`'s `x_threshold`
     has no default and is required.
  6. (item 2) the degraded-mode hard non-verdict: `verdict_pass` is the
     literal string `'INDETERMINATE (NL-write share not separated)'`, never
     a bool, whenever `nl_write_seconds` is absent -- and a plain bool
     otherwise, on the same toy records.
  7. (item 1) the widened precondition function
     (`p515_s36_step36_timing_run._check_preconditions`) refuses when a
     LISTED file is dirty -- simulated by stubbing the git-status subprocess
     call only, never by dirtying a real file.
  8. (P5.15 Step 3.6 Worker task, defect D1) a derived phase that computes to
     a negative duration RAISES `ValueError` (never silently reported), AND
     a D1 regression guard: `clone` at the same block as a correctly-nested
     `block_total` is never subtracted from `bookkeeping`.
  9. (defect D2) totals exclude `cycle is None` (pre-ADMM-loop
     initialization) records; those are reported separately under
     `initialization_totals`.
 10. (defect D3) `solve_bundle_subtimes` (per-seq {nl_write, ipopt,
     sol_parse}) produces a DETERMINATE verdict (plain bool) and
     `overhead_total == wall_total - ipopt_total`; PARTIAL coverage still
     falls back to the degraded indeterminate path (never a partial
     verdict).

Output directory: `data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/`
by default, overridable via the `P515_S36_CHECKS_OUT_DIR` environment
variable (used for the follow-up re-runs so the original, committed
`zero_solve_checks/`/`zero_solve_checks_v2/` directories are never
overwritten) -- refuses to run if the selected directory already exists --
`results.json` plus a sha256 manifest.

Usage (original run, still reproducible):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py

Usage (follow-up re-runs -- write to a NEW directory each time):
    P515_S36_CHECKS_OUT_DIR=data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v2 \\
        /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py
    P515_S36_CHECKS_OUT_DIR=data/SRP1/Results/P515S36/step36_timing/zero_solve_checks_v3 \\
        /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_step36_timing_checks.py
"""

import dis
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Output directory: overridable via P515_S36_CHECKS_OUT_DIR so a re-run adding
# new checks (P5.15 Step 3.6 follow-up, item 5) never overwrites the committed
# `zero_solve_checks/` directory the original run wrote -- the write-once
# `os.makedirs(..., exist_ok=False)` guard below still applies to whichever
# directory is selected.
OUT_DIR = os.environ.get(
    'P515_S36_CHECKS_OUT_DIR',
    os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'step36_timing', 'zero_solve_checks'))

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s36_step36_timing as T  # noqa: E402
import p515_s36_step36_timing_run as R  # noqa: E402

FAILURES = []
RESULTS = {}


def _check(name, condition, detail=None):
    RESULTS[name] = {'pass': bool(condition), 'detail': detail}
    if not condition:
        FAILURES.append(name)
    return condition


def _make_ancestor_caller(filename, funcname, local_names, call_expr):
    """Compile a function whose `__code__.co_filename == filename` and
    `co_name == funcname`, with `local_names` bound as LOCAL variables (from
    keyword arguments the returned callable is invoked with) before
    `call_expr` runs and is returned. Used to synthesize an ancestor frame
    matching a real production call site's (file, function, locals) WITHOUT
    executing any real production code -- `target` (also a keyword argument)
    is the wrapped attribute under test.
    """
    lines = [f'def {funcname}(**_kwargs):', "    target = _kwargs['target']"]
    for name in local_names:
        lines.append(f"    {name} = _kwargs['{name}']")
    lines.append(f'    return {call_expr}')
    src = '\n'.join(lines) + '\n'
    code = compile(src, filename, 'exec')
    ns = {}
    exec(code, ns)
    return ns[funcname]


class _Sentinel:
    def __repr__(self):
        return '<sentinel>'


SENTINEL = _Sentinel()


class _BoomError(Exception):
    pass


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to run: output directory already exists: {OUT_DIR}')
    os.makedirs(OUT_DIR, exist_ok=False)

    guard = SolveProfileGuard(permitted=(), label='p515_s36_step36_timing_checks (zero solves)').install()
    try:
        import shared_resources_planning as srp
        import network
        import network_data  # noqa: F401 -- imported for parity/documentation; not called directly
        import shared_energy_storage_data as SED
        import pyomo.environ as pe
        import pyomo.opt as po
        from pyomo.opt.base.solvers import OptSolver
        from pyomo.opt.solver.shellcmd import SystemCallSolver
        from pyomo.core.base.block import BlockData
        from pyomo.core.base.PyomoModel import ModelSolutions

        # =================================================================
        # CHECK 1 -- module-level wrap targets resolve as unqualified
        # globals in the caller's OWN bytecode (dis-scan, zero-solve).
        # =================================================================
        loop_code = srp._run_operational_planning.__code__
        load_global_names = {instr.argval for instr in dis.get_instructions(loop_code)
                              if instr.opname == 'LOAD_GLOBAL'}
        expected_srp_wraps = {
            'update_distribution_coordination_models_and_solve',
            'update_transmission_coordination_model_and_solve',
            'update_shared_energy_storages_coordination_model_and_solve',
            'update_and_check_convergence',
            'get_admm_residual_metrics',
            'get_admm_boyd_residual_metrics',
            '_update_tso_proximal_centres_after_solve',
            '_update_admm_penalties',
        }
        missing = sorted(expected_srp_wraps - load_global_names)
        _check('C1_srp_stage_and_admm_global_wraps_are_LOAD_GLOBAL_in_run_operational_planning',
               not missing, {'missing': missing, 'observed_relevant': sorted(expected_srp_wraps & load_global_names)})

        optimize_code = SED.SharedEnergyStorageData.optimize.__code__
        optimize_globals = {instr.argval for instr in dis.get_instructions(optimize_code)
                            if instr.opname == 'LOAD_GLOBAL'}
        _check('C1_SED_optimize_calls__optimize_via_LOAD_GLOBAL',
               '_optimize' in optimize_globals, {'observed': sorted(optimize_globals)})

        underscore_optimize_code = SED._optimize.__code__
        underscore_optimize_globals = {instr.argval for instr in dis.get_instructions(underscore_optimize_code)
                                       if instr.opname == 'LOAD_GLOBAL'}
        _check('C1__optimize_calls_complementarity_diagnostics_via_LOAD_GLOBAL',
               '_get_esso_complementarity_diagnostics' in underscore_optimize_globals,
               {'observed': sorted(underscore_optimize_globals)})

        # sanity: the two filename suffixes used for ancestry matching below
        # must not be ambiguous with respect to `str.endswith`.
        _check('C1_network_py_suffix_does_not_match_network_data_py',
               not 'network_data.py'.endswith('network.py'))

        # =================================================================
        # CHECK 2 -- class-level wrap targets resolve via the class the real
        # instance's MRO actually uses (real, zero-solve instantiation).
        # =================================================================
        net_instance = network.Network()
        _check('C2_Network_instance_run_smopf_resolves_to_class_attribute',
               type(net_instance).run_smopf is network.Network.run_smopf)

        m = pe.ConcreteModel()
        _check('C2_ConcreteModel_clone_resolves_to_BlockData_clone',
               type(m).clone is BlockData.clone)
        _check('C2_ConcreteModel_solutions_load_from_resolves_to_ModelSolutions_load_from',
               type(m.solutions).load_from is ModelSolutions.load_from)

        solver_instance = po.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')
        _check('C2_ipopt_SolverFactory_instance_solve_resolves_to_OptSolver_solve',
               type(solver_instance).solve is OptSolver.solve,
               {'mro': [c.__name__ for c in type(solver_instance).__mro__]})
        _check('C2_ipopt_SolverFactory_instance_execute_command_resolves_to_SystemCallSolver',
               type(solver_instance)._execute_command is SystemCallSolver._execute_command,
               'same identity precedent p513_solve_profile_guard.py already relies on')

        # =================================================================
        # CHECK 3 -- install/uninstall exact-identity restoration.
        # =================================================================
        recorder = T.PhaseTimingRecorder()

        def _snapshot():
            return {
                'srp.update_distribution_coordination_models_and_solve': srp.update_distribution_coordination_models_and_solve,
                'srp.update_transmission_coordination_model_and_solve': srp.update_transmission_coordination_model_and_solve,
                'srp.update_shared_energy_storages_coordination_model_and_solve': srp.update_shared_energy_storages_coordination_model_and_solve,
                'srp.update_and_check_convergence': srp.update_and_check_convergence,
                'srp.get_admm_residual_metrics': srp.get_admm_residual_metrics,
                'srp.get_admm_boyd_residual_metrics': srp.get_admm_boyd_residual_metrics,
                'srp._update_tso_proximal_centres_after_solve': srp._update_tso_proximal_centres_after_solve,
                'srp._update_admm_penalties': srp._update_admm_penalties,
                'network.Network.run_smopf': network.Network.run_smopf,
                'SED._optimize': SED._optimize,
                'SED._get_esso_complementarity_diagnostics': SED._get_esso_complementarity_diagnostics,
                'BlockData.clone': BlockData.clone,
                'OptSolver.solve': OptSolver.solve,
                'ModelSolutions.load_from': ModelSolutions.load_from,
            }

        pre_originals = _snapshot()
        with T.recorder_installed(recorder):
            during = _snapshot()
        after = _snapshot()

        differs = {key: (during[key] is not pre_originals[key]) for key in pre_originals}
        _check('C3_every_wrap_target_changed_while_installed', all(differs.values()), differs)
        restored = {key: (after[key] is pre_originals[key]) for key in pre_originals}
        _check('C3_every_wrap_target_restored_to_exact_identity_after_uninstall', all(restored.values()), restored)

        # =================================================================
        # CHECK 4 -- behaviour preservation with trivial stand-ins.
        # =================================================================
        behaviour_results = {}

        # ---- (a) stage_total: exact real signatures, cycle bound by name ----
        def _dso_stand_in_ok(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf,
                             ess_req, dual_ess, params, sess_estimated_capacities,
                             from_warm_start=False, parallel_execution=False, cycle=None):
            return SENTINEL

        def _dso_stand_in_raises(*args, **kwargs):
            raise _BoomError('boom')

        def _run_stage_test(attr_owner, attr_name, stand_in_ok, stand_in_raises, call_kwargs, expected_cycle):
            original_real = getattr(attr_owner, attr_name)
            try:
                setattr(attr_owner, attr_name, stand_in_ok)
                local_recorder = T.PhaseTimingRecorder()
                with T.recorder_installed(local_recorder):
                    result = getattr(attr_owner, attr_name)(**call_kwargs)
                ok_pass = (result is SENTINEL)
                stage_records = [r for r in local_recorder.records if r['phase'] == 'stage_total']
                ok_recorded = (len(stage_records) == 1 and stage_records[0]['elapsed_s'] >= 0
                              and stage_records[0]['cycle'] == expected_cycle)

                setattr(attr_owner, attr_name, stand_in_raises)
                local_recorder2 = T.PhaseTimingRecorder()
                raised = None
                with T.recorder_installed(local_recorder2):
                    try:
                        getattr(attr_owner, attr_name)(**call_kwargs)
                    except _BoomError as error:
                        raised = error
                exc_pass = (raised is not None and str(raised) == 'boom')
                stage_records2 = [r for r in local_recorder2.records if r['phase'] == 'stage_total']
                exc_recorded = (len(stage_records2) == 1)

                return ok_pass and ok_recorded and exc_pass and exc_recorded, {
                    'return_value_preserved': ok_pass, 'record_appended_on_success': ok_recorded,
                    'exception_preserved': exc_pass, 'record_appended_on_exception': exc_recorded,
                }
            finally:
                setattr(attr_owner, attr_name, original_real)

        ok, detail = _run_stage_test(
            srp, 'update_distribution_coordination_models_and_solve',
            _dso_stand_in_ok, _dso_stand_in_raises,
            {'distribution_networks': {}, 'models': {}, 'vmag_req': {}, 'dual_vmag': {},
             'pf_req': {}, 'dual_pf': {}, 'ess_req': {}, 'dual_ess': {}, 'params': None,
             'sess_estimated_capacities': {}, 'from_warm_start': False,
             'parallel_execution': False, 'cycle': 7}, 7)
        behaviour_results['dso_stage_total'] = detail
        _check('C4_dso_stage_total_wrapper_preserves_behaviour', ok, detail)

        def _tso_stand_in_ok(transmission_network, model, vmag_req, dual_vmag, pf_req, dual_pf,
                             ess_req, dual_ess, params, sess_estimated_capacities,
                             from_warm_start=False, cycle=None):
            return SENTINEL

        ok, detail = _run_stage_test(
            srp, 'update_transmission_coordination_model_and_solve',
            _tso_stand_in_ok, _dso_stand_in_raises,
            {'transmission_network': None, 'model': {}, 'vmag_req': {}, 'dual_vmag': {},
             'pf_req': {}, 'dual_pf': {}, 'ess_req': {}, 'dual_ess': {}, 'params': None,
             'sess_estimated_capacities': {}, 'from_warm_start': False, 'cycle': 9}, 9)
        behaviour_results['tso_stage_total'] = detail
        _check('C4_tso_stage_total_wrapper_preserves_behaviour', ok, detail)

        def _esso_stand_in_ok(planning_problem, models, ess_req, dual_ess, params,
                              from_warm_start=False, cycle=None):
            return SENTINEL

        ok, detail = _run_stage_test(
            srp, 'update_shared_energy_storages_coordination_model_and_solve',
            _esso_stand_in_ok, _dso_stand_in_raises,
            {'planning_problem': None, 'models': {}, 'ess_req': {}, 'dual_ess': {},
             'params': None, 'from_warm_start': False, 'cycle': 3}, 3)
        behaviour_results['esso_stage_total'] = detail
        _check('C4_esso_stage_total_wrapper_preserves_behaviour', ok, detail)

        # ---- (b) admm_global, no cycle param -- falls back to recorder.current_cycle ----
        def _run_admm_global_test(attr_name, call_args, call_kwargs):
            original_real = getattr(srp, attr_name)
            try:
                def stand_in_ok(*args, **kwargs):
                    return SENTINEL

                setattr(srp, attr_name, stand_in_ok)
                local_recorder = T.PhaseTimingRecorder()
                local_recorder.set_current_cycle(42)
                with T.recorder_installed(local_recorder):
                    result = getattr(srp, attr_name)(*call_args, **call_kwargs)
                admm_records = [r for r in local_recorder.records if r['phase'] == 'admm_global'
                                and r['block'].get('call') == attr_name]
                return (result is SENTINEL and len(admm_records) == 1
                        and admm_records[0]['cycle'] == 42), {
                    'return_value_preserved': result is SENTINEL,
                    'records': admm_records,
                }
            finally:
                setattr(srp, attr_name, original_real)

        for name in ('update_and_check_convergence', 'get_admm_residual_metrics',
                     'get_admm_boyd_residual_metrics'):
            ok, detail = _run_admm_global_test(name, (), {})
            behaviour_results[f'admm_global_{name}'] = detail
            _check(f'C4_admm_global_{name}_wrapper_preserves_behaviour', ok, detail)

        # `_update_tso_proximal_centres_after_solve` carries its OWN `cycle` kwarg
        original_real = srp._update_tso_proximal_centres_after_solve
        try:
            def stand_in_ok(planning_problem, model, results, cycle=None):
                return SENTINEL

            srp._update_tso_proximal_centres_after_solve = stand_in_ok
            local_recorder = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder):
                result = srp._update_tso_proximal_centres_after_solve(None, None, None, cycle=11)
            recs = [r for r in local_recorder.records if r['phase'] == 'admm_global'
                   and r['block'].get('call') == '_update_tso_proximal_centres_after_solve']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['cycle'] == 11)
            detail = {'records': recs}
        finally:
            srp._update_tso_proximal_centres_after_solve = original_real
        behaviour_results['admm_global_tso_proximal_centres'] = detail
        _check('C4_admm_global_tso_proximal_centres_wrapper_binds_own_cycle_kwarg', ok, detail)

        # `_update_admm_penalties` binds `iter=` (a different parameter NAME for
        # the same value)
        original_real = srp._update_admm_penalties
        try:
            def stand_in_ok(tso_model, dso_models, esso_model, residual_metrics, boyd_metrics,
                            params, iter=None, allow_update=True, freeze_state=None):
                return SENTINEL

            srp._update_admm_penalties = stand_in_ok
            local_recorder = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder):
                result = srp._update_admm_penalties(None, None, None, None, None, None, iter=13)
            recs = [r for r in local_recorder.records if r['phase'] == 'admm_global'
                   and r['block'].get('call') == '_update_admm_penalties']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['cycle'] == 13)
            detail = {'records': recs}
        finally:
            srp._update_admm_penalties = original_real
        behaviour_results['admm_global_update_admm_penalties'] = detail
        _check('C4_admm_global_update_admm_penalties_wrapper_binds_iter_kwarg_as_cycle', ok, detail)

        # ---- (c) Network.run_smopf (block_total, self-derived key -- no frame walk) ----
        original_run_smopf = network.Network.run_smopf

        class _FakeNet:
            name = 'case33_1'
            year = 2025
            day = 'Summer'
            is_transmission = False

        def stand_in_run_smopf(self, *args, **kwargs):
            return SENTINEL

        try:
            network.Network.run_smopf = stand_in_run_smopf
            local_recorder = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder):
                result = network.Network.run_smopf(_FakeNet(), None, None, from_warm_start=False)
            recs = [r for r in local_recorder.records if r['phase'] == 'block_total' and r['agent'] == 'dso']
            ok = (result is SENTINEL and len(recs) == 1
                  and recs[0]['block'] == {'kind': 'dso', 'name': 'case33_1', 'year': 2025, 'day': 'Summer'})
            detail = {'records': recs}
        finally:
            network.Network.run_smopf = original_run_smopf
        behaviour_results['network_run_smopf_block_total'] = detail
        _check('C4_network_run_smopf_wrapper_derives_block_key_from_self', ok, detail)

        # ---- (d) SED._optimize (block_total, ESSO) ----
        original_sed_optimize = SED._optimize

        def stand_in_sed_optimize(model, params, from_warm_start=False, node_id=None,
                                   diagnostic_sink=None, option_overrides=None,
                                   complementarity_diagnostics_sink=None, cycle=None, logs_dir=None):
            return SENTINEL

        try:
            SED._optimize = stand_in_sed_optimize
            local_recorder = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder):
                result = SED._optimize(None, None, node_id=7, cycle=5)
            recs = [r for r in local_recorder.records if r['phase'] == 'block_total' and r['agent'] == 'esso']
            ok = (result is SENTINEL and len(recs) == 1
                  and recs[0]['block'] == {'kind': 'esso', 'node_id': 7} and recs[0]['cycle'] == 5)
            detail = {'records': recs}
        finally:
            SED._optimize = original_sed_optimize
        behaviour_results['sed_optimize_block_total'] = detail
        _check('C4_sed_optimize_wrapper_binds_node_id_and_cycle', ok, detail)

        # ---- (e) frame-walk-dependent wraps: synthesize ancestor frames ----

        # solve_bundle, network.py ancestry (_run_smopf_solver_attempt)
        original_solve = OptSolver.solve
        try:
            def stand_in_solve(self, model, **kwargs):
                return SENTINEL

            OptSolver.solve = stand_in_solve
            local_recorder = T.PhaseTimingRecorder()

            class _FakeNetworkTSO:
                name = 'case9'
                year = 2030
                day = 'Winter'
                is_transmission = True

            with T.recorder_installed(local_recorder):
                caller = _make_ancestor_caller(
                    os.path.join('some', 'path', 'network.py'), '_run_smopf_solver_attempt',
                    ['network', 'log_suffix'],
                    'target(None, None, tee=False, load_solutions=False)')
                result = caller(target=getattr(OptSolver, 'solve'), network=_FakeNetworkTSO(),
                                log_suffix='recovery')
            recs = [r for r in local_recorder.records if r['phase'] == 'solve_bundle']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['agent'] == 'tso'
                  and recs[0]['block'] == {'kind': 'tso', 'name': 'case9', 'year': 2030, 'day': 'Winter'}
                  and recs[0]['attempt'] == 'tier1_recovery')
            detail = {'records': recs}
        finally:
            OptSolver.solve = original_solve
        behaviour_results['solve_bundle_network_ancestry'] = detail
        _check('C4_solve_bundle_wrapper_attributes_via_synthesized_network_py_ancestor_frame', ok, detail)

        # solve_bundle, shared_energy_storage_data.py ancestry (_run_solver_attempt),
        # and exception passthrough
        try:
            def stand_in_solve_raises(self, model, **kwargs):
                raise _BoomError('esso boom')

            OptSolver.solve = stand_in_solve_raises
            local_recorder = T.PhaseTimingRecorder()
            raised = None
            with T.recorder_installed(local_recorder):
                caller = _make_ancestor_caller(
                    os.path.join('some', 'path', 'shared_energy_storage_data.py'), '_run_solver_attempt',
                    ['node_id', 'log_suffix', 'cycle'],
                    'target(None, None, tee=False, load_solutions=False)')
                try:
                    caller(target=getattr(OptSolver, 'solve'), node_id=9, log_suffix=None, cycle=4)
                except _BoomError as error:
                    raised = error
            recs = [r for r in local_recorder.records if r['phase'] == 'solve_bundle']
            ok = (raised is not None and str(raised) == 'esso boom' and len(recs) == 1
                  and recs[0]['agent'] == 'esso' and recs[0]['block'] == {'kind': 'esso', 'node_id': 9}
                  and recs[0]['attempt'] == 'primary')
            detail = {'records': recs}
        finally:
            OptSolver.solve = original_solve
        behaviour_results['solve_bundle_esso_ancestry_and_exception'] = detail
        _check('C4_solve_bundle_wrapper_attributes_via_synthesized_esso_ancestor_frame_and_preserves_exception',
               ok, detail)

        # load_solution, network.py ancestry (_run_smopf)
        original_load_from = ModelSolutions.load_from
        try:
            def stand_in_load_from(self, *args, **kwargs):
                return SENTINEL

            ModelSolutions.load_from = stand_in_load_from
            local_recorder = T.PhaseTimingRecorder()

            class _FakeNetworkDSO:
                name = 'case33_2'
                year = 2025
                day = 'Autumn'
                is_transmission = False

            with T.recorder_installed(local_recorder):
                caller = _make_ancestor_caller(
                    os.path.join('some', 'path', 'network.py'), '_run_smopf',
                    ['network', 'recovery_attempted', 'tier2_attempted'],
                    'target(None)')
                result = caller(target=getattr(ModelSolutions, 'load_from'), network=_FakeNetworkDSO(),
                                recovery_attempted=True, tier2_attempted=True)
            recs = [r for r in local_recorder.records if r['phase'] == 'load_solution']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['agent'] == 'dso'
                  and recs[0]['attempt'] == 'tier2_recovery')
            detail = {'records': recs}
        finally:
            ModelSolutions.load_from = original_load_from
        behaviour_results['load_solution_network_ancestry'] = detail
        _check('C4_load_solution_wrapper_attributes_via_synthesized_network_py_ancestor_frame', ok, detail)

        # diagnostics_parse, shared_energy_storage_data.py ancestry (_optimize)
        original_diag = SED._get_esso_complementarity_diagnostics
        try:
            def stand_in_diag(model, node_id, log_path):
                return SENTINEL

            SED._get_esso_complementarity_diagnostics = stand_in_diag
            local_recorder = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder):
                caller = _make_ancestor_caller(
                    os.path.join('some', 'path', 'shared_energy_storage_data.py'), '_optimize',
                    ['cycle'], 'target(None, 5, "irrelevant.log")')
                result = caller(target=getattr(SED, '_get_esso_complementarity_diagnostics'), cycle=17)
            recs = [r for r in local_recorder.records if r['phase'] == 'diagnostics_parse']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['cycle'] == 17
                  and recs[0]['block'] == {'kind': 'esso', 'node_id': 5})
            detail = {'records': recs}
        finally:
            SED._get_esso_complementarity_diagnostics = original_diag
        behaviour_results['diagnostics_parse_ancestry'] = detail
        _check('C4_diagnostics_parse_wrapper_attributes_via_synthesized_optimize_ancestor_frame', ok, detail)

        # clone, network_data.py ancestry (NetworkData.optimize)
        original_clone = BlockData.clone
        try:
            def stand_in_clone(self, *args, **kwargs):
                return SENTINEL

            BlockData.clone = stand_in_clone
            local_recorder = T.PhaseTimingRecorder()

            class _FakeHolder:
                name = 'case33_2'
                is_transmission = False

            with T.recorder_installed(local_recorder):
                caller = _make_ancestor_caller(
                    os.path.join('some', 'path', 'network_data.py'), 'optimize',
                    ['self', 'year', 'day'], 'target(None)')
                result = caller(target=getattr(BlockData, 'clone'), self=_FakeHolder(),
                                year=2025, day='Autumn')
            recs = [r for r in local_recorder.records if r['phase'] == 'clone']
            ok = (result is SENTINEL and len(recs) == 1 and recs[0]['agent'] == 'dso'
                  and recs[0]['block'] == {'kind': 'dso', 'name': 'case33_2', 'year': 2025, 'day': 'Autumn'})
            detail = {'records': recs}

            # Negative control: a clone call with NO matching ancestor frame
            # must be passed through UNTIMED (not recorded at all).
            local_recorder2 = T.PhaseTimingRecorder()
            with T.recorder_installed(local_recorder2):
                result2 = BlockData.clone(None)
            no_ancestor_ok = (result2 is SENTINEL and len(local_recorder2.records) == 0)
        finally:
            BlockData.clone = original_clone
        behaviour_results['clone_ancestry'] = detail
        behaviour_results['clone_no_ancestor_passthrough_untimed'] = {'pass': no_ancestor_ok}
        _check('C4_clone_wrapper_attributes_via_synthesized_network_data_py_ancestor_frame', ok, detail)
        _check('C4_clone_wrapper_passes_through_untimed_when_no_declared_ancestor', no_ancestor_ok)

        # =================================================================
        # CHECK 5 -- derive_param_update_and_bookkeeping and
        # analyze_phase_timing reproduce a hand-computed toy example.
        # =================================================================
        # Toy cycle 1, agent dso, two nodes ('n5' then 'n7'):
        #   stage_total:  t=0.000 -> 1.000   (elapsed 1.000)
        #   n5 block_total: t=0.200 -> 0.700  (elapsed 0.500)
        #     n5 solve_bundle: t=0.300 -> 0.600 (elapsed 0.300, primary)
        #     n5 load_solution: t=0.600 -> 0.650 (elapsed 0.050, primary)
        #     -> n5 bookkeeping = 0.500 - 0.300 - 0.050 = 0.150
        #   n7 block_total: t=0.750 -> 1.000  (elapsed 0.250)
        #     n7 solve_bundle: t=0.800 -> 0.950 (elapsed 0.150, primary)
        #     -> n7 bookkeeping = 0.250 - 0.150 = 0.100
        #   param_update(n5) = 0.200 - 0.000 (stage start) = 0.200
        #   param_update(n7) = 0.750 - 0.700 (end of n5 block_total) = 0.050
        toy = [
            {'seq': 1, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso'}, 'phase': 'stage_total',
             'attempt': 'n/a', 'start_perf': 0.000, 'end_perf': 1.000, 'elapsed_s': 1.000},
            {'seq': 2, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.200, 'end_perf': 0.700, 'elapsed_s': 0.500},
            {'seq': 3, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.300, 'end_perf': 0.600, 'elapsed_s': 0.300},
            {'seq': 4, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'load_solution', 'attempt': 'primary', 'start_perf': 0.600, 'end_perf': 0.650, 'elapsed_s': 0.050},
            {'seq': 5, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n7', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.750, 'end_perf': 1.000, 'elapsed_s': 0.250},
            {'seq': 6, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n7', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.800, 'end_perf': 0.950, 'elapsed_s': 0.150},
        ]
        derived = T.derive_param_update_and_bookkeeping(toy)
        pu = {d['block'].get('name'): round(d['elapsed_s'], 6) for d in derived if d['phase'] == 'param_update'}
        bk = {d['block'].get('name'): round(d['elapsed_s'], 6) for d in derived if d['phase'] == 'bookkeeping'}
        _check('C5_derive_param_update_matches_hand_computation',
               pu == {'n5': 0.200, 'n7': 0.050}, pu)
        _check('C5_derive_bookkeeping_matches_hand_computation',
               bk == {'n5': 0.150, 'n7': 0.100}, bk)

        all_records = toy + derived
        analysis = T.analyze_phase_timing(all_records, x_threshold=0.70,
                                          production_iter_wall_by_cycle={1: 1.100},
                                          projection_workers=2)
        # Hand computation:
        #   overhead_local = param_update(0.25) + clone(0) + nl_write(=solve_bundle upper
        #     bound, 0.45) + load_solution(0.05) + bookkeeping(0.25) + diagnostics_parse(0)
        #     = 0.25 + 0.45 + 0.05 + 0.25 = 1.00
        #   recorder span for cycle 1 = max(end_perf) - min(start_perf) = 1.000 - 0.000 = 1.000
        #   unattributed = production_iter_wall(1.100) - span(1.000) = 0.100
        #   overhead_serial = admm_global(0) + unattributed(0.100) + solve_bundle_remainder(0,
        #     since nl_write_seconds is absent -> the WHOLE solve_bundle is the NL-write
        #     upper bound, per the docstring, so the "remainder" is 0)
        #   verdict_ratio = 1.00 / (1.00 + 0.100) = 0.90909...  -> would PASS at X=0.70 on the
        #     ratio alone, BUT nl_write_seconds is absent here (degraded mode), so per item 2
        #     (Planner decision, P5.15 Step 3.6 follow-up) verdict_pass must be the hard
        #     non-verdict string, never True/False, even though the ratio clears the bar --
        #     see C7 below for the same formula exercised in BOTH modes explicitly.
        expected_overhead_local = round(0.25 + 0.45 + 0.05 + 0.25, 6)
        expected_overhead_serial = round(0.100, 6)
        _check('C5_overhead_local_total_matches_hand_computation',
               round(analysis['overhead_local_total'], 6) == expected_overhead_local,
               analysis['overhead_local_total'])
        _check('C5_overhead_serial_total_matches_hand_computation',
               round(analysis['overhead_serial_total'], 6) == expected_overhead_serial,
               analysis['overhead_serial_total'])
        _check('C5_verdict_ratio_clears_x_0p70_bar',
               analysis['verdict_ratio'] is not None and analysis['verdict_ratio'] >= 0.70,
               analysis['verdict_ratio'])
        _check('C5_verdict_pass_is_degraded_nonverdict_not_bool_true',
               analysis['verdict_pass'] == 'INDETERMINATE (NL-write share not separated)'
               and not isinstance(analysis['verdict_pass'], bool),
               analysis['verdict_pass'])
        _check('C5_nl_write_share_flagged_as_upper_bound', analysis['nl_write_share_is_upper_bound'] is True)
        _check('C5_unattributed_matches_hand_computation',
               round(analysis['per_cycle_unattributed'][1], 6) == 0.100,
               analysis['per_cycle_unattributed'])

        # =================================================================
        # CHECK 6 (P5.15 Step 3.6 follow-up, item 3) -- `x_threshold` has NO
        # default; it is a required positional-or-keyword argument.
        # =================================================================
        import inspect as _inspect
        sig = _inspect.signature(T.analyze_phase_timing)
        _check('C6_x_threshold_parameter_has_no_default',
               sig.parameters['x_threshold'].default is _inspect.Parameter.empty,
               str(sig.parameters['x_threshold']))
        try:
            T.analyze_phase_timing(all_records)
            raised_without_threshold = False
        except TypeError:
            raised_without_threshold = True
        _check('C6_calling_analyze_phase_timing_without_x_threshold_raises_TypeError',
               raised_without_threshold)
        _check('C6_run_entrypoint_states_x_threshold_as_module_constant_0p70',
               R.X_THRESHOLD == 0.70, R.X_THRESHOLD)

        # =================================================================
        # CHECK 7 (P5.15 Step 3.6 follow-up, item 2) -- degraded mode (no
        # NL-write sub-times) is a hard non-verdict; a run WITH sub-times
        # still returns a plain bool. Same toy records, both modes, on the
        # SAME x_threshold, so the only variable is `nl_write_seconds`.
        # =================================================================
        degraded_analysis = T.analyze_phase_timing(
            all_records, x_threshold=0.70, production_iter_wall_by_cycle={1: 1.100},
            projection_workers=2, nl_write_seconds=None)
        _check('C7_degraded_mode_verdict_pass_is_indeterminate_string',
               degraded_analysis['verdict_pass'] == 'INDETERMINATE (NL-write share not separated)',
               degraded_analysis['verdict_pass'])
        _check('C7_degraded_mode_verdict_is_indeterminate_via_run_helper',
               R.verdict_is_indeterminate(degraded_analysis) is True)

        # P5.15 Step 3.6 Worker task (defect D3) CORRECTION: an aggregate-only
        # `nl_write_seconds` (no per-record `solve_bundle_subtimes`) makes
        # `nl_write_share_is_upper_bound` False, but is STILL indeterminate --
        # D3 showed that knowing the NL-write share ALONE is not enough to
        # classify the rest of `solve_bundle` (the v1 defect was putting the
        # UNKNOWN ipopt+sol_parse remainder entirely into overhead_serial).
        # A determinate verdict now requires the FULL per-record b/c/d1 split
        # (`solve_bundle_subtimes`, exercised in Check 11) -- this check was
        # originally written expecting aggregate-only NL-write to be
        # sufficient; that expectation was itself part of the pre-D3-fix
        # picture and is corrected here, not merely re-asserted.
        aggregate_nl_write_analysis = T.analyze_phase_timing(
            all_records, x_threshold=0.70, production_iter_wall_by_cycle={1: 1.100},
            projection_workers=2, nl_write_seconds={'aggregate': 0.20})
        _check('C7_aggregate_nl_write_alone_sets_upper_bound_flag_false',
               aggregate_nl_write_analysis['nl_write_share_is_upper_bound'] is False)
        _check('C7_D3_aggregate_nl_write_alone_is_STILL_indeterminate_without_solve_bundle_subtimes',
               aggregate_nl_write_analysis['verdict_pass'] == 'INDETERMINATE (NL-write share not separated)'
               and aggregate_nl_write_analysis['solve_bundle_subtimes_fully_covered'] is False,
               aggregate_nl_write_analysis['verdict_pass'])
        _check('C7_D3_aggregate_nl_write_alone_indeterminate_via_run_helper',
               R.verdict_is_indeterminate(aggregate_nl_write_analysis) is True)

        # Sanity: the helper itself, on synthetic dicts, with no analyze_phase_timing
        # call at all (isolates the helper's own logic from the analysis function).
        _check('C7_run_helper_flags_string_verdict_as_indeterminate',
               R.verdict_is_indeterminate({'verdict_pass': 'INDETERMINATE (NL-write share not separated)'}) is True)
        _check('C7_run_helper_flags_bool_true_as_determinate',
               R.verdict_is_indeterminate({'verdict_pass': True}) is False)
        _check('C7_run_helper_flags_bool_false_as_determinate',
               R.verdict_is_indeterminate({'verdict_pass': False}) is False)

        # =================================================================
        # CHECK 8 (P5.15 Step 3.6 follow-up, item 1) -- the widened
        # precondition function (`p515_s36_step36_timing_run._check_preconditions`)
        # refuses when a LISTED file is dirty. Simulated by stubbing the
        # git-status subprocess call only (never by actually dirtying a real
        # file); the real `ps aux` scan and lock/output-dir checks still run
        # for real underneath, so this may ALSO surface real, unrelated
        # failures (e.g. the s38 numerical campaign's own live process) --
        # this check only asserts that the SIMULATED dirty-file failure is
        # present among whatever `_check_preconditions()` returns.
        # =================================================================
        _real_subprocess_run = R.subprocess.run

        def _stub_subprocess_run(args, *pos, **kwargs):
            if len(args) >= 2 and args[0] == 'git' and args[1] == 'status':
                class _FakeCompleted:
                    stdout = ' M admm_parameters.py\n'
                return _FakeCompleted()
            return _real_subprocess_run(args, *pos, **kwargs)

        try:
            R.subprocess.run = _stub_subprocess_run
            simulated_failures = R._check_preconditions()
        finally:
            R.subprocess.run = _real_subprocess_run
        dirty_failure_present = any(
            'production files are not clean in git' in f and 'admm_parameters.py' in f
            for f in simulated_failures)
        _check('C8_check_preconditions_refuses_on_simulated_dirty_listed_file',
               dirty_failure_present, simulated_failures)
        # Confirm the real (unstubbed) call is restored afterward (identity),
        # i.e. this check never leaves `subprocess.run` monkeypatched.
        _check('C8_check_preconditions_subprocess_run_restored_after_stub_removed',
               R.subprocess.run is _real_subprocess_run)

        # =================================================================
        # CHECK 9 (P5.15 Step 3.6 Worker task, defect D1) -- a derived phase
        # that computes to a negative duration RAISES `ValueError`, never
        # silently reports the negative value. Two toy examples:
        #   (a) a nesting-defect toy (`block_total` shorter than the
        #       `solve_bundle` it is supposed to enclose) -- must raise.
        #   (b) the SAME toy structurally, but with `clone` ALSO present at
        #       the same block, VALID (block_total properly encloses
        #       solve_bundle+load_solution; clone is a sibling, per the D1
        #       fix) -- must NOT raise, and `clone` must NOT appear anywhere
        #       in the bookkeeping subtraction (regression guard for D1
        #       itself: the exact defect that produced -8.37 s in v1).
        # =================================================================
        toy_negative = [
            {'seq': 1, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'nX', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.0, 'end_perf': 0.1, 'elapsed_s': 0.1},
            {'seq': 2, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'nX', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.0, 'end_perf': 0.3, 'elapsed_s': 0.3},
        ]
        raised_on_negative = False
        negative_error_detail = None
        try:
            T.derive_param_update_and_bookkeeping(toy_negative)
        except ValueError as error:
            raised_on_negative = True
            negative_error_detail = str(error)
        _check('C9_derive_raises_valueerror_on_negative_bookkeeping',
               raised_on_negative, negative_error_detail)

        # (b) D1 regression guard: clone present at the SAME block as a
        # correctly-nested block_total must NOT be subtracted -- bookkeeping
        # = block_total - solve_bundle - load_solution (clone excluded).
        #   block_total: 0.500 (n5), solve_bundle 0.300, load_solution 0.050,
        #   clone 0.180 (a SIBLING event, timestamps outside block_total's
        #   own span, exactly as network_data.py:61 vs :62-63) ->
        #   bookkeeping = 0.500 - 0.300 - 0.050 = 0.150 (clone NEVER
        #   subtracted; if it were, this would be 0.150 - 0.180 = -0.030,
        #   which per Check 9(a)'s own contract would raise).
        toy_clone_sibling = [
            {'seq': 1, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'clone', 'attempt': 'n/a', 'start_perf': 0.000, 'end_perf': 0.180, 'elapsed_s': 0.180},
            {'seq': 2, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.200, 'end_perf': 0.700, 'elapsed_s': 0.500},
            {'seq': 3, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.300, 'end_perf': 0.600, 'elapsed_s': 0.300},
            {'seq': 4, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'load_solution', 'attempt': 'primary', 'start_perf': 0.600, 'end_perf': 0.650, 'elapsed_s': 0.050},
        ]
        derived_clone_sibling = T.derive_param_update_and_bookkeeping(toy_clone_sibling)
        bk_clone_sibling = [d['elapsed_s'] for d in derived_clone_sibling if d['phase'] == 'bookkeeping']
        _check('C9_D1_regression_clone_is_never_subtracted_from_bookkeeping',
               len(bk_clone_sibling) == 1 and round(bk_clone_sibling[0], 6) == 0.150,
               bk_clone_sibling)

        # =================================================================
        # CHECK 10 (P5.15 Step 3.6 Worker task, defect D2) -- totals exclude
        # `cycle is None` (pre-ADMM-loop initialization) records; those are
        # reported separately under `initialization_totals`, never silently
        # mixed into the sampled-cycle totals.
        # =================================================================
        toy_cycle1 = [
            {'seq': 10, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso'}, 'phase': 'stage_total',
             'attempt': 'n/a', 'start_perf': 0.000, 'end_perf': 1.000, 'elapsed_s': 1.000},
            {'seq': 11, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.200, 'end_perf': 0.700, 'elapsed_s': 0.500},
            {'seq': 12, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.300, 'end_perf': 0.600, 'elapsed_s': 0.300},
            {'seq': 13, 'cycle': 1, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n5', 'year': 2025, 'day': 'Summer'},
             'phase': 'load_solution', 'attempt': 'primary', 'start_perf': 0.600, 'end_perf': 0.650, 'elapsed_s': 0.050},
        ]
        toy_init = [
            {'seq': 1, 'cycle': None, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n_init', 'year': 2025, 'day': 'Summer'},
             'phase': 'block_total', 'attempt': 'n/a', 'start_perf': 0.0, 'end_perf': 0.4, 'elapsed_s': 0.4},
            {'seq': 2, 'cycle': None, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n_init', 'year': 2025, 'day': 'Summer'},
             'phase': 'solve_bundle', 'attempt': 'primary', 'start_perf': 0.0, 'end_perf': 0.3, 'elapsed_s': 0.3},
            {'seq': 3, 'cycle': None, 'agent': 'dso', 'block': {'kind': 'dso', 'name': 'n_init', 'year': 2025, 'day': 'Summer'},
             'phase': 'load_solution', 'attempt': 'primary', 'start_perf': 0.3, 'end_perf': 0.35, 'elapsed_s': 0.05},
        ]
        toy_d2 = toy_cycle1 + toy_init
        derived_d2 = T.derive_param_update_and_bookkeeping(toy_d2)
        analysis_d2 = T.analyze_phase_timing(toy_d2 + derived_d2, x_threshold=0.70,
                                             production_iter_wall_by_cycle={1: 1.100},
                                             projection_workers=2)
        _check('C10_solve_bundle_count_excludes_init_cycle_none_records',
               analysis_d2['per_phase_by_agent']['dso']['solve_bundle']['count'] == 1,
               analysis_d2['per_phase_by_agent']['dso']['solve_bundle'])
        _check('C10_initialization_totals_captures_the_excluded_cycle_none_solve_bundle',
               analysis_d2['initialization_totals']['dso']['solve_bundle']['count'] == 1
               and round(analysis_d2['initialization_totals']['dso']['solve_bundle']['sum'], 6) == 0.300,
               analysis_d2['initialization_totals']['dso']['solve_bundle'])
        _check('C10_initialization_totals_captures_the_excluded_cycle_none_bookkeeping',
               analysis_d2['initialization_totals']['dso']['bookkeeping']['count'] == 1
               and round(analysis_d2['initialization_totals']['dso']['bookkeeping']['sum'], 6) == 0.050,
               analysis_d2['initialization_totals']['dso']['bookkeeping'])

        # =================================================================
        # CHECK 11 (P5.15 Step 3.6 Worker task, defect D3) -- `solve_bundle_subtimes`
        # (per-seq {nl_write, ipopt, sol_parse}) produces a DETERMINATE
        # verdict (plain bool), and `overhead_total` is computed as
        # `wall_total - ipopt_total`, cross-checked against
        # `overhead_local_total + overhead_serial_total`.
        #   Toy (same records as Check 5's n5/n7 solve_bundle events):
        #     seq 3 (n5, elapsed 0.300): nl_write=0.10, ipopt=0.15, sol_parse=0.03 (glue=0.02)
        #     seq 6 (n7, elapsed 0.150): nl_write=0.05, ipopt=0.08, sol_parse=0.01 (glue=0.01)
        #   nl_write_total=0.15, ipopt_total=0.23, sol_parse_total=0.04, glue_total=0.03
        #   overhead_local  = param_update(0.25) + clone(0) + nl_write(0.15)
        #                    + load_solution(0.05) + bookkeeping(0.25)
        #                    + diagnostics_parse(0) + sol_parse(0.04) + glue(0.03) = 0.77
        #   overhead_serial = admm_global(0) + unattributed(0.10) = 0.10
        #   overhead_total  = 0.87 == wall_total(1.10) - ipopt_total(0.23) = 0.87
        #   verdict_ratio   = 0.77 / 0.87 = 0.885... >= 0.70 -> verdict_pass is True (a bool)
        # =================================================================
        subtimes = {3: {'nl_write': 0.10, 'ipopt': 0.15, 'sol_parse': 0.03},
                    6: {'nl_write': 0.05, 'ipopt': 0.08, 'sol_parse': 0.01}}
        analysis_d3 = T.analyze_phase_timing(all_records, x_threshold=0.70,
                                             production_iter_wall_by_cycle={1: 1.100},
                                             projection_workers=2, solve_bundle_subtimes=subtimes)
        _check('C11_solve_bundle_subtimes_fully_covered_flag_true',
               analysis_d3['solve_bundle_subtimes_fully_covered'] is True)
        _check('C11_nl_write_share_not_flagged_as_upper_bound',
               analysis_d3['nl_write_share_is_upper_bound'] is False)
        _check('C11_overhead_local_total_matches_hand_computation',
               round(analysis_d3['overhead_local_total'], 6) == 0.77, analysis_d3['overhead_local_total'])
        _check('C11_overhead_serial_total_matches_hand_computation',
               round(analysis_d3['overhead_serial_total'], 6) == 0.10, analysis_d3['overhead_serial_total'])
        _check('C11_overhead_total_equals_wall_minus_ipopt_crosscheck',
               round(analysis_d3['overhead_total'], 6) == 0.87
               and round(analysis_d3['overhead_total_cross_check_(wall_minus_ipopt)'], 6) == 0.87,
               {'overhead_total': analysis_d3['overhead_total'],
                'cross_check': analysis_d3['overhead_total_cross_check_(wall_minus_ipopt)']})
        _check('C11_verdict_pass_is_plain_bool_true_when_fully_covered',
               analysis_d3['verdict_pass'] is True, analysis_d3['verdict_pass'])
        _check('C11_ipopt_total_matches_hand_computation',
               round(analysis_d3['ipopt_total'], 6) == 0.23, analysis_d3['ipopt_total'])

        # Partial coverage (only ONE of the two solve_bundle seqs has
        # subtimes) must NOT be treated as fully covered -- falls back to
        # the degraded path, never silently computes a partial verdict.
        partial_subtimes = {3: {'nl_write': 0.10, 'ipopt': 0.15, 'sol_parse': 0.03}}
        analysis_partial = T.analyze_phase_timing(all_records, x_threshold=0.70,
                                                   production_iter_wall_by_cycle={1: 1.100},
                                                   projection_workers=2,
                                                   solve_bundle_subtimes=partial_subtimes)
        _check('C11_partial_coverage_falls_back_to_degraded_indeterminate',
               analysis_partial['solve_bundle_subtimes_fully_covered'] is False
               and analysis_partial['verdict_pass'] == 'INDETERMINATE (NL-write share not separated)',
               analysis_partial['verdict_pass'])

        results_path = os.path.join(OUT_DIR, 'results.json')
        with open(results_path, 'w') as handle:
            json.dump({'checks': RESULTS, 'failures': FAILURES, 'behaviour_detail': behaviour_results,
                      'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                      handle, indent=2, sort_keys=True, default=str)

        manifest = {}
        with open(results_path, 'rb') as handle:
            manifest['results.json'] = hashlib.sha256(handle.read()).hexdigest()
        manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
        with open(manifest_path, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        print(f"[P5.15-S36-TIMING-CHECKS] {len(RESULTS) - len(FAILURES)}/{len(RESULTS)} checks passed.")
        if FAILURES:
            print(f"[FAIL] {FAILURES}")
        else:
            print('[OK] all checks passed.')
        print(f'[OK] wrote {results_path}')
        print(f'[OK] wrote {manifest_path}')

    finally:
        guard_failures = guard.verify(expected_solves=0, expected_execs=0)
        guard.uninstall()
        if guard_failures:
            raise RuntimeError(f'SolveProfileGuard failed at expected_solves=0: {guard_failures}')
        print('[OK] SolveProfileGuard verified: 0 permitted solves, 0 blocked solves (zero-solve checks).')

    if FAILURES:
        raise RuntimeError(f'{len(FAILURES)} check(s) failed: {FAILURES}')


if __name__ == '__main__':
    main()
