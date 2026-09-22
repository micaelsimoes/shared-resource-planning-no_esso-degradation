import os
from dotenv import load_dotenv
from definitions import ERROR_PARAMS_FILE

DOTENV_FILE = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(DOTENV_FILE)

# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self, default_solver='ipopt', path_env_vars=('NLP_SOLVER_PATH', 'SOLVER_PATH'), label='NLP solver', require_path=True):

        self.solver = default_solver
        self.verbose = False
        self.options = None
        self.recovery_options = None
        # P5.15 Addendum 7 Part 1 item 1 (PLANNER_BRIEF_2026-09-13.md): explicit
        # recovery-eligibility policy, independent of whether `recovery_options`
        # happens to be populated. Defaults hold even if `read_solver_parameters`
        # is never called (e.g. a harness that builds `SolverParameters` directly).
        self.recovery_enabled = True
        self.recovery_tier2_enabled = True
        # P5.15 Addendum 29 (W32): after a successful network solve, release
        # Pyomo's solution bookkeeping (network._release_solution_bookkeeping).
        # Default False = the pre-W32 behaviour, byte for byte.
        self.release_solution_bookkeeping = False
        self.solver_path = next((os.getenv(var) for var in path_env_vars if os.getenv(var)), None)

        if require_path and not self.solver_path:
            env_vars = ' or '.join(path_env_vars)
            print(f'[ERROR] {label} path not found. Set {env_vars} in {DOTENV_FILE}. Exiting')
            exit(ERROR_PARAMS_FILE)

    def read_solver_parameters(self, solver_data):
        _read_solver_parameters(self, solver_data)


def _read_solver_parameters(parameters, solver_data):
    parameters.solver = solver_data['name']
    parameters.verbose = solver_data['verbose']
    parameters.options = solver_data['options']
    parameters.recovery_options = solver_data.get('recovery_options')
    # P5.15 Addendum 7 Part 1 item 1: optional nested `recovery` block --
    # `solver.recovery.enabled` / `solver.recovery.tier2_enabled` in the case/
    # ESSO params JSON (`solver_data` IS the "solver" object, see the callers in
    # network_parameters.py / shared_energy_storage_parameters.py). A case file
    # is never required to declare this block; absence means the default
    # (True/True) applies, and eligibility no longer depends on
    # `recovery_options` being non-empty.
    recovery_policy = solver_data.get('recovery') or {}
    parameters.recovery_enabled = bool(recovery_policy.get('enabled', True))
    parameters.recovery_tier2_enabled = bool(recovery_policy.get('tier2_enabled', True))
    # P5.15 Addendum 29 (W32): optional; absent means False (pre-W32 behaviour).
    parameters.release_solution_bookkeeping = bool(solver_data.get('release_solution_bookkeeping', False))
