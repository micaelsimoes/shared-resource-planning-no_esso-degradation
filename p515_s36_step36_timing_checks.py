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

Output: `data/SRP1/Results/P515S36/step36_timing/zero_solve_checks/` (refuses
to run if it already exists) -- `results.json` plus a sha256 manifest.

Usage:
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

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'step36_timing', 'zero_solve_checks')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s36_step36_timing as T  # noqa: E402

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
        analysis = T.analyze_phase_timing(all_records, production_iter_wall_by_cycle={1: 1.100},
                                          x_threshold=0.70, projection_workers=2)
        # Hand computation:
        #   overhead_local = param_update(0.25) + clone(0) + nl_write(=solve_bundle upper
        #     bound, 0.45) + load_solution(0.05) + bookkeeping(0.25) + diagnostics_parse(0)
        #     = 0.25 + 0.45 + 0.05 + 0.25 = 1.00
        #   recorder span for cycle 1 = max(end_perf) - min(start_perf) = 1.000 - 0.000 = 1.000
        #   unattributed = production_iter_wall(1.100) - span(1.000) = 0.100
        #   overhead_serial = admm_global(0) + unattributed(0.100) + solve_bundle_remainder(0,
        #     since nl_write_seconds is absent -> the WHOLE solve_bundle is the NL-write
        #     upper bound, per the docstring, so the "remainder" is 0)
        #   verdict_ratio = 1.00 / (1.00 + 0.100) = 0.90909...  -> pass at X=0.70
        expected_overhead_local = round(0.25 + 0.45 + 0.05 + 0.25, 6)
        expected_overhead_serial = round(0.100, 6)
        _check('C5_overhead_local_total_matches_hand_computation',
               round(analysis['overhead_local_total'], 6) == expected_overhead_local,
               analysis['overhead_local_total'])
        _check('C5_overhead_serial_total_matches_hand_computation',
               round(analysis['overhead_serial_total'], 6) == expected_overhead_serial,
               analysis['overhead_serial_total'])
        _check('C5_verdict_pass_at_x_0p70', analysis['verdict_pass'] is True, analysis['verdict_ratio'])
        _check('C5_nl_write_share_flagged_as_upper_bound', analysis['nl_write_share_is_upper_bound'] is True)
        _check('C5_unattributed_matches_hand_computation',
               round(analysis['per_cycle_unattributed'][1], 6) == 0.100,
               analysis['per_cycle_unattributed'])

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
