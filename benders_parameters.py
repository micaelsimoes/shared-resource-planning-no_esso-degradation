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
        self.minimum_signal_to_noise_ratio = float(
            params_data.get('minimum_signal_to_noise_ratio', self.minimum_signal_to_noise_ratio)
        )
        self.slope_consistency_tolerance = float(
            params_data.get('slope_consistency_tolerance', self.slope_consistency_tolerance)
        )
        self.soh_active_tolerance = float(
            params_data.get('soh_active_tolerance', self.soh_active_tolerance)
        )


class BendersParameters:

    def __init__(self):
        self.tol_abs = 1e3
        self.tol_rel = 1e-2
        self.num_max_iters = 1000
        self.finite_difference = FiniteDifferenceParameters()

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(benders_params, params_data):
    benders_params.tol_abs = float(params_data['tol_abs'])
    benders_params.tol_rel = float(params_data['tol_rel'])
    benders_params.num_max_iters = int(params_data['num_max_iters'])
    benders_params.finite_difference.read_parameters(params_data.get('finite_difference'))
