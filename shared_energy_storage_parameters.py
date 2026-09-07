from solver_parameters import SolverParameters
from helper_functions import *


# ======================================================================================================================
#  Salvage Value Parameters
# ======================================================================================================================
class SalvageValueParameters:

    def __init__(self):
        self.enabled = False
        self.energy_recovery_fraction = 1.00
        self.recycling_floor_fraction = 0.00
        self.cost_basis = 'EXPECTED_INSTALLATION_ENERGY_COST'
        self.health_basis = 'NORMALIZED_ABOVE_MINIMUM_SOH'
        self.calendar_life_basis = 'REMAINING_FRACTION_AT_TERMINAL'

    def read_parameters(self, params_data):
        if not params_data:
            return

        self.enabled = bool(params_data.get('enabled', self.enabled))
        self.energy_recovery_fraction = float(
            params_data.get('energy_recovery_fraction', self.energy_recovery_fraction)
        )
        self.recycling_floor_fraction = float(
            params_data.get('recycling_floor_fraction', self.recycling_floor_fraction)
        )
        self.cost_basis = str(params_data.get('cost_basis', self.cost_basis)).upper()
        self.health_basis = str(params_data.get('health_basis', self.health_basis)).upper()
        self.calendar_life_basis = str(
            params_data.get('calendar_life_basis', self.calendar_life_basis)
        ).upper()

        if not 0.00 <= self.energy_recovery_fraction <= 1.00:
            raise ValueError('Salvage energy_recovery_fraction must be between 0 and 1.')
        if not 0.00 <= self.recycling_floor_fraction <= 1.00:
            raise ValueError('Salvage recycling_floor_fraction must be between 0 and 1.')
        if self.cost_basis != 'EXPECTED_INSTALLATION_ENERGY_COST':
            raise ValueError(f'Unsupported salvage cost basis: {self.cost_basis}.')
        if self.health_basis != 'NORMALIZED_ABOVE_MINIMUM_SOH':
            raise ValueError(f'Unsupported salvage health basis: {self.health_basis}.')
        if self.calendar_life_basis != 'REMAINING_FRACTION_AT_TERMINAL':
            raise ValueError(
                f'Unsupported salvage calendar-life basis: {self.calendar_life_basis}.'
            )


# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                               # 1 M m.u.
        self.max_capacity = 2.50                        # Max energy capacity (related to space constraints)
        self.min_energy_to_power_ratio = 2.00           # Minimum energy-to-power ratio (related to the ESS technology)
        self.max_energy_to_power_ratio = 10.00          # Maximum energy-to-power ratio (related to the ESS technology)
        self.slacks = False                             # Relax/use slack variables
        self.plot_results = False                       # Plot results
        self.print_results_to_file = False              # Write results to file
        self.verbose = False                            # Verbose -- Bool
        self.salvage_value = SalvageValueParameters()
        self.solver_params = SolverParameters(
            default_solver='ipopt',
            path_env_vars=('NLP_SOLVER_PATH', 'SOLVER_PATH'),
            label='NLP solver'
        )
        self.lp_solver_params = SolverParameters(
            default_solver='clp',
            path_env_vars=('LP_SOLVER_PATH',),
            label='LP solver'
        )

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(planning_parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    planning_parameters.budget = float(params_data['budget'])
    planning_parameters.max_capacity = float(params_data['max_capacity'])
    planning_parameters.min_energy_to_power_ratio = float(params_data['min_energy_to_power_factor'])
    planning_parameters.max_energy_to_power_ratio = float(params_data['max_energy_to_power_factor'])
    planning_parameters.slacks = bool(params_data['slacks'])
    planning_parameters.print_results_to_file = bool(params_data['print_results_to_file'])
    nlp_solver_data = params_data.get('nlp_solver')
    if nlp_solver_data is None:
        nlp_solver_data = params_data['solver']
    planning_parameters.solver_params.read_solver_parameters(nlp_solver_data)
    if 'lp_solver' in params_data:
        planning_parameters.lp_solver_params.read_solver_parameters(params_data['lp_solver'])
    planning_parameters.salvage_value.read_parameters(params_data.get('salvage_value'))
