from math import log

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
#  Degradation Calibration -- DECLARED, NOT CONSUMED
# ======================================================================================================================
class DegradationCalibrationParameters:
    """The (N, D, R) calibration triple for the cycling degradation law.

    N -- cycle count, D -- reference depth of discharge, R -- end-of-life
    retention. The law's characteristic constant is k = N * D / (-ln R), and the
    identity EFC_to_R = -ln(R) * k = N * D holds for every triple.

    P5.13-C declared this block without consuming it. P5.13-D activates it:
    with `status = ACTIVE` the degradation law consumes `k` instead of `cl_nom`
    (`shared_energy_storage_data.py`, the `energy_storage_capacity_degradation`
    rows). This CHANGES the degradation term and therefore the objective, and was
    authorized as calibration C3 = (10000, 0.80, 0.50), k = 11541.560327111707.
    With `status = DECLARED_NOT_CONSUMED` the law reverts to `cl_nom` exactly.
    See `P5_13_B_CYCLING_CALIBRATION.md` for the candidates and their impact.
    """

    STATUS_DECLARED = 'DECLARED_NOT_CONSUMED'
    STATUS_ACTIVE = 'ACTIVE'
    VALID_STATUS = (STATUS_DECLARED, STATUS_ACTIVE)

    def __init__(self):
        self.cycles_n = None
        self.reference_dod_d = None
        self.eol_retention_r = None
        self.status = self.STATUS_DECLARED

    def is_active(self):
        return self.status == self.STATUS_ACTIVE

    def characteristic_constant(self):
        """k = N * D / (-ln R), or None while the triple is undecided.

        Consumed by the degradation law when the calibration is ACTIVE; see
        `EnergyStorageAgeingParameters.effective_cycle_constant`.
        """
        if self.cycles_n is None or self.reference_dod_d is None or self.eol_retention_r is None:
            return None
        return self.cycles_n * self.reference_dod_d / (-log(self.eol_retention_r))

    def read_parameters(self, params_data):
        if not params_data:
            return

        self.cycles_n = _read_optional_number(params_data, 'cycles_n', self.cycles_n)
        self.reference_dod_d = _read_optional_number(params_data, 'reference_dod_d', self.reference_dod_d)
        self.eol_retention_r = _read_optional_number(params_data, 'eol_retention_r', self.eol_retention_r)
        self.status = str(params_data.get('status', self.status)).upper()

        if self.status not in self.VALID_STATUS:
            raise ValueError(
                f'Unsupported degradation calibration status: {self.status}. '
                f'Expected one of {self.VALID_STATUS}.')
        if self.status == self.STATUS_ACTIVE and None in (
                self.cycles_n, self.reference_dod_d, self.eol_retention_r):
            raise ValueError(
                'An ACTIVE degradation calibration requires all three of cycles_n, '
                'reference_dod_d and eol_retention_r. k = N*D/(-ln R) is undefined '
                'otherwise.')
        if self.reference_dod_d is not None and not 0.00 < self.reference_dod_d <= 1.00:
            raise ValueError('Calibration reference_dod_d must lie in (0, 1].')
        if self.eol_retention_r is not None and not 0.00 < self.eol_retention_r < 1.00:
            raise ValueError('Calibration eol_retention_r must lie in (0, 1).')
        if self.cycles_n is not None and self.cycles_n <= 0:
            raise ValueError('Calibration cycles_n must be positive.')


# ======================================================================================================================
#  Energy Storage Ageing Parameters
# ======================================================================================================================
class EnergyStorageAgeingParameters:
    """Ageing constants applied to every SharedEnergyStorage instance.

    P5.13-C moved these out of `SharedEnergyStorage.__init__`, where they were
    hard-coded and therefore identical for every case and invisible to review.
    The defaults below reproduce those hard-coded values EXACTLY, so an absent
    key leaves behaviour unchanged.
    """

    def __init__(self):
        self.t_cal = 15                                 # Calendar life, [years]
        self.cl_nom = 10000                             # Cycle life, nominal, [cycles]
        self.dod_nom = 0.80                             # Depth-of-Discharge, nominal, [0-1]
        self.soh_min = 0.50                             # Minimum SoH, [0-1]
        self.calibration = DegradationCalibrationParameters()

    def read_parameters(self, params_data):
        if not params_data:
            return

        self.t_cal = _read_optional_number(params_data, 'calendar_life_years', self.t_cal)
        self.cl_nom = _read_optional_number(params_data, 'cycle_life_nominal', self.cl_nom)
        self.dod_nom = _read_optional_number(params_data, 'depth_of_discharge_nominal', self.dod_nom)
        self.soh_min = _read_optional_number(params_data, 'minimum_soh', self.soh_min)
        self.calibration.read_parameters(params_data.get('calibration'))

        if self.t_cal <= 0:
            raise ValueError('ESS calendar_life_years must be positive.')
        if self.cl_nom <= 0:
            raise ValueError('ESS cycle_life_nominal must be positive.')
        if not 0.00 < self.dod_nom <= 1.00:
            raise ValueError('ESS depth_of_discharge_nominal must lie in (0, 1].')
        if not 0.00 <= self.soh_min < 1.00:
            raise ValueError('ESS minimum_soh must lie in [0, 1).')

    def apply_to(self, shared_energy_storage):
        """Apply the ageing constants to one SharedEnergyStorage instance.

        `cl_eff` is the constant the degradation law actually consumes. With the
        calibration inactive it is `cl_nom`, reproducing the historical
        behaviour exactly. With the calibration ACTIVE it is
        `k = N*D/(-ln R)`, so the reference depth and the end-of-life retention
        become load-bearing instead of vestigial -- which is what stops the
        (count, depth) pair drifting apart again (P5.13-B, F0).
        """
        shared_energy_storage.t_cal = self.t_cal
        shared_energy_storage.cl_nom = self.cl_nom
        shared_energy_storage.dod_nom = self.dod_nom
        shared_energy_storage.soh_min = self.soh_min
        shared_energy_storage.cl_eff = self.effective_cycle_constant()
        return shared_energy_storage

    def effective_cycle_constant(self):
        """The constant consumed by the degradation law: k, or cl_nom."""
        if not self.calibration.is_active():
            return self.cl_nom
        if self.calibration.cycles_n != self.cl_nom:
            raise ValueError(
                f'Calibration cycles_n ({self.calibration.cycles_n}) must equal '
                f'cycle_life_nominal ({self.cl_nom}); two sources for the same '
                'quantity are exactly how the count and the depth drifted apart.')
        if self.calibration.reference_dod_d != self.dod_nom:
            raise ValueError(
                f'Calibration reference_dod_d ({self.calibration.reference_dod_d}) '
                f'must equal depth_of_discharge_nominal ({self.dod_nom}).')
        return self.calibration.characteristic_constant()


def _read_optional_number(params_data, key, current):
    """Read a numeric key, preserving its JSON type (int stays int).

    Type preservation matters: `cl_nom` enters a constraint expression, and
    coercing 10000 to 10000.0 would change the rendered model.
    """
    if key not in params_data or params_data[key] is None:
        return current
    value = params_data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'ESS parameter {key} must be numeric.')
    return value


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
        self.ageing = EnergyStorageAgeingParameters()
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
    planning_parameters.ageing.read_parameters(params_data.get('ageing'))
