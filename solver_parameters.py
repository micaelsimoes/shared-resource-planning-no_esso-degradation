import os
from dotenv import load_dotenv
from definitions import ERROR_PARAMS_FILE
load_dotenv('.env')

# ============================================================================================
#   Class SolverParameters
# ============================================================================================
class SolverParameters:

    def __init__(self):

        # LP solver
        self.lp_solver = "clp"
        self.lp_solver_path = os.getenv('LP_SOLVER_PATH')
        self.lp_options = {}

        # NLP solver
        self.nlp_solver = "ipopt"
        self.nlp_solver_path = os.getenv('NLP_SOLVER_PATH')
        self.nlp_options = {}

        self.verbose = False

    def read_solver_parameters(self, solver_data):

        lp_data = solver_data.get("lp_solver", {})
        self.lp_solver = lp_data.get("name", self.lp_solver)
        self.lp_options = lp_data.get("options", {})
        if not isinstance(self.lp_options, dict):
            raise ValueError("'lp_solver.options' must be a dictionary.")

        nlp_data = solver_data.get("nlp_solver", {})
        self.nlp_solver = nlp_data.get("name", self.nlp_solver)
        self.nlp_options = nlp_data.get("options", {})
        if not isinstance(self.nlp_options, dict):
            raise ValueError("'nlp_solver.options' must be a dictionary.")

        self.verbose = bool(solver_data.get("verbose", False))
