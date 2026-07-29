from solver_parameters import SolverParameters
from helper_functions import *


# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                               # 1 M m.u.
        self.max_capacity = 2.50                        # Max energy capacity (related to space constraints)
        self.min_energy_to_power_ratio = 2.00           # Minimum energy-to-power ratio (related to the ESS technology)
        self.max_energy_to_power_ratio = 10.00          # Maximum energy-to-power ratio (related to the ESS technology)
        self.calendar_ageing = 0.5e-2                   # Calendar ageing parameter
        self.slacks = False                             # Relax/use slack variables
        self.plot_results = False                       # Plot results
        self.print_results_to_file = False              # Write results to file
        self.verbose = False                            # Verbose -- Bool
        self.solver_params = SolverParameters()         # Solver Parameters

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(planning_parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    planning_parameters.budget = float(params_data['budget'])
    planning_parameters.max_capacity = float(params_data['max_capacity'])
    planning_parameters.min_energy_to_power_ratio = float(params_data['min_energy_to_power_factor'])
    planning_parameters.max_energy_to_power_ratio = float(params_data['max_energy_to_power_factor'])
    planning_parameters.calendar_ageing = float(params_data['calendar_ageing'])
    planning_parameters.slacks = bool(params_data['slacks'])
    planning_parameters.print_results_to_file = bool(params_data['print_results_to_file'])
    planning_parameters.solver_params.read_solver_parameters(params_data['solver'])
