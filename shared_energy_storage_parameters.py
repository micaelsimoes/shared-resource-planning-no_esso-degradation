from solver_parameters import SolverParameters
from helper_functions import *


# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                               # 1 M m.u.
        self.max_capacity = 2.50                        # Max energy capacity (related to space constraints)
        self.min_pe_factor = 0.10                       # Minimum S/E factor (related to the ESS technology)
        self.max_pe_factor = 4.00                       # Maximum S/E factor (related to the ESS technology)
        self.relax_equalities = False                   # Relax equality constraints
        self.slacks = Slacks()                          # Relax/use slack variables
        self.plot_results = False                       # Plot results
        self.verbose = False                            # Verbose -- Bool
        self.solver_params = SolverParameters()         # Solver Parameters
        self.print_results_to_file = False              # Write results to file

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


# ======================================================================================================================
#   Slack Classes
# ======================================================================================================================
class Slacks:

    def __init__(self):
        self.rated_power = False
        self.rated_capacity = False
        self.apparent_power_agg = False
        self.apparent_power_def = False
        self.complementarity = False
        self.soh = False

    def read_slacks_parameters(self, slacks_data):
        if 'rated_power' in slacks_data:
            self.rated_power = slacks_data['rated_power']
        if 'rated_capacity' in slacks_data:
            self.rated_power = slacks_data['rated_capacity']
        if 'apparent_power_agg' in slacks_data:
            self.apparent_power_agg = slacks_data['apparent_power_agg']
        if 'complementarity' in slacks_data:
            self.complementarity = slacks_data['complementarity']
        if 'soh' in slacks_data:
            self.complementarity = slacks_data['soh']


def _read_parameters_from_file(planning_parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    planning_parameters.budget = float(params_data['budget'])
    planning_parameters.max_capacity = float(params_data['max_capacity'])
    planning_parameters.min_pe_factor = float(params_data['min_pe_factor'])
    planning_parameters.max_pe_factor = float(params_data['max_pe_factor'])
    planning_parameters.relax_equalities = bool(params_data['relax_equalities'])
    if 'slacks' in params_data:
        planning_parameters.slacks.read_slacks_parameters(params_data['slacks'])
    planning_parameters.print_results_to_file = bool(params_data['print_results_to_file'])
    planning_parameters.solver_params.read_solver_parameters(params_data['solver'])
    planning_parameters.solver_params.read_solver_parameters(params_data['solver'])
