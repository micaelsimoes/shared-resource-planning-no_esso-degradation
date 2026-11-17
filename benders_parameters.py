# ======================================================================================================================
#  Class Benders' Parameters
# ======================================================================================================================
class FiniteDifferenceParameters:

    def __init__(self):
        self.enabled = False
        self.validate_after_heuristic_stop = False
        self.directions = ['fixed_ratio']
        self.relative_step_sizes = [1e-2, 5e-3]
        self.node_id = None
        self.year = None
        self.relative_error_tolerance = 1e-1
        self.replay_relative_tolerance = 1e-6
        self.replay_absolute_tolerance = 1.0
        self.max_replay_refinements = 10
        self.minimum_signal_to_noise_ratio = 10.0
        self.slope_consistency_tolerance = 1e-1
        self.soh_active_tolerance = 1e-4

    def read_parameters(self, params_data):
        if not params_data:
            return
        self.enabled = bool(params_data.get('enabled', self.enabled))
        self.validate_after_heuristic_stop = bool(
            params_data.get(
                'validate_after_heuristic_stop',
                self.validate_after_heuristic_stop,
            )
        )
        self.directions = [
            str(direction).lower() for direction in params_data.get('directions', self.directions)
        ]
        if not self.directions:
            raise ValueError('At least one finite-difference direction must be configured.')
        supported_directions = {'power_only', 'energy_only', 'fixed_ratio'}
        unsupported_directions = set(self.directions) - supported_directions
        if unsupported_directions:
            raise ValueError(
                f'Unsupported finite-difference directions: {sorted(unsupported_directions)}.'
            )
        step_sizes = params_data.get(
            'relative_step_sizes',
            params_data.get('step_sizes', self.relative_step_sizes),
        )
        self.relative_step_sizes = [float(value) for value in step_sizes]
        self.node_id = params_data.get('node_id', self.node_id)
        self.year = params_data.get('year', self.year)
        self.relative_error_tolerance = float(
            params_data.get('relative_error_tolerance', self.relative_error_tolerance)
        )
        self.replay_relative_tolerance = float(
            params_data.get('replay_relative_tolerance', self.replay_relative_tolerance)
        )
        self.replay_absolute_tolerance = float(
            params_data.get('replay_absolute_tolerance', self.replay_absolute_tolerance)
        )
        self.max_replay_refinements = int(
            params_data.get('max_replay_refinements', self.max_replay_refinements)
        )
        if self.max_replay_refinements < 1:
            raise ValueError('Finite-difference max_replay_refinements must be at least one.')
        self.minimum_signal_to_noise_ratio = float(
            params_data.get('minimum_signal_to_noise_ratio', self.minimum_signal_to_noise_ratio)
        )
        self.slope_consistency_tolerance = float(
            params_data.get('slope_consistency_tolerance', self.slope_consistency_tolerance)
        )
        self.soh_active_tolerance = float(
            params_data.get('soh_active_tolerance', self.soh_active_tolerance)
        )


class PositiveBootstrapParameters:

    def __init__(self):
        self.enabled = False
        self.budget_fraction = 5e-2
        self.energy_to_power_ratio = None

    def read_parameters(self, params_data):
        if not params_data:
            return

        self.enabled = bool(params_data.get('enabled', self.enabled))
        self.budget_fraction = float(
            params_data.get('budget_fraction', self.budget_fraction)
        )
        ratio = params_data.get('energy_to_power_ratio', self.energy_to_power_ratio)
        self.energy_to_power_ratio = None if ratio is None else float(ratio)

        if not 0.00 < self.budget_fraction <= 1.00:
            raise ValueError('Positive-bootstrap budget_fraction must be in (0, 1].')
        if self.energy_to_power_ratio is not None and self.energy_to_power_ratio <= 0.00:
            raise ValueError('Positive-bootstrap energy_to_power_ratio must be positive.')


class SensitivityProbeParameters:

    def __init__(self):
        self.enabled = False
        self.budget_fraction = 5e-2
        self.energy_to_power_ratio = None

    def read_parameters(self, params_data):
        if not params_data:
            return

        self.enabled = bool(params_data.get('enabled', self.enabled))
        self.budget_fraction = float(
            params_data.get('budget_fraction', self.budget_fraction)
        )
        ratio = params_data.get('energy_to_power_ratio', self.energy_to_power_ratio)
        self.energy_to_power_ratio = None if ratio is None else float(ratio)

        if not 0.00 < self.budget_fraction <= 1.00:
            raise ValueError('Sensitivity-probe budget_fraction must be in (0, 1].')
        if self.energy_to_power_ratio is not None and self.energy_to_power_ratio <= 0.00:
            raise ValueError('Sensitivity-probe energy_to_power_ratio must be positive.')


class BendersParameters:

    def __init__(self):
        self.tol_abs = 1e3
        self.tol_rel = 1e-2
        self.num_max_iters = 1000
        self.positive_bootstrap = PositiveBootstrapParameters()
        self.sensitivity_probe = SensitivityProbeParameters()
        self.finite_difference = FiniteDifferenceParameters()

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(benders_params, params_data):
    benders_params.tol_abs = float(params_data['tol_abs'])
    benders_params.tol_rel = float(params_data['tol_rel'])
    benders_params.num_max_iters = int(params_data['num_max_iters'])
    benders_params.positive_bootstrap.read_parameters(params_data.get('positive_bootstrap'))
    benders_params.sensitivity_probe.read_parameters(params_data.get('sensitivity_probe'))
    benders_params.finite_difference.read_parameters(params_data.get('finite_difference'))
