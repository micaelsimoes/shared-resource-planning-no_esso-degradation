import os
from dotenv import load_dotenv
from definitions import ERROR_PARAMS_FILE
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self, default_solver='ipopt', path_env_vars=('NLP_SOLVER_PATH', 'SOLVER_PATH'), label='NLP solver', require_path=True):

        self.solver = default_solver
        self.verbose = False
        self.options = None
        self.solver_path = next(
            (os.getenv(var) for var in path_env_vars if os.getenv(var)),
            None,
        )

        if require_path and not self.solver_path:
            print(f'[ERROR] {label} path not found! Exiting')
            exit(ERROR_PARAMS_FILE)

    def read_solver_parameters(self, solver_data):
        _read_solver_parameters(self, solver_data)


def _read_solver_parameters(parameters, solver_data):
    parameters.solver = solver_data['name']
    parameters.verbose = solver_data['verbose']
    parameters.options = solver_data['options']
