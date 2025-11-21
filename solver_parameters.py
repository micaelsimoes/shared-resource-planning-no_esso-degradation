import os
from dotenv import load_dotenv
from definitions import ERROR_PARAMS_FILE
load_dotenv('.env')

# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self):
        self.solver = 'ipopt'
        self.verbose = False
        self.options = None
        self.solver_path = os.getenv('SOLVER_PATH')
        if not self.solver_path:
            print('[ERROR] Solver path not found! Exiting')
            exit(ERROR_PARAMS_FILE)

    def read_solver_parameters(self, solver_data):
        _read_solver_parameters(self, solver_data)


def _read_solver_parameters(parameters, solver_data):
    parameters.solver = solver_data['name']
    parameters.verbose = solver_data['verbose']
    parameters.options = solver_data['options']
