# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = {
            'consensus': {
                'v': 0.1e-2, 'v_mean': 1e-2,
                'pf': 0.1e-2, 'pf_mean': 1e-2,
                'ess': 0.1e-2, 'ess_mean': 1e-2
            },
            'stationarity': {'v': 0.5e-2, 'pf': 0.5e-2, 'ess': 5e-2},
            'objective': {'abs': 1e4, 'rel': 5e-4}}
        self.num_max_iters = 1000
        self.minimum_consecutive_converged_cycles = 2
        self.shared_ess_normalization_floor_mva = 0.10
        self.adaptive_penalty = False
        self.penalty_update = {
            'residual_balance_ratio': 5.0,
            'residual_balance_ratio_pf_decrease': 5.0,
            'increase_factor': 2.0,
            'decrease_factor': 2.0,
            'min': 1e-4,
            'max': 1e4,
        }
        self.rho = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.rho_previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(admm_params, params_data):

    consensus_tolerances = params_data['tol']['consensus']

    admm_params.tol['consensus']['v'] = float(consensus_tolerances['v'])
    admm_params.tol['consensus']['v_mean'] = float(consensus_tolerances.get('v_mean', consensus_tolerances['v']))
    admm_params.tol['consensus']['pf'] = float(consensus_tolerances['pf'])
    admm_params.tol['consensus']['pf_mean'] = float(consensus_tolerances.get('pf_mean', consensus_tolerances['pf']))
    admm_params.tol['consensus']['ess'] = float(consensus_tolerances['ess'])
    admm_params.tol['consensus']['ess_mean'] = float(consensus_tolerances.get('ess_mean', consensus_tolerances['ess']))

    admm_params.tol['stationarity']['v'] = float(params_data['tol']['stationarity']['v'])
    admm_params.tol['stationarity']['pf'] = float(params_data['tol']['stationarity']['pf'])
    admm_params.tol['stationarity']['ess'] = float(params_data['tol']['stationarity']['ess'])

    objective_tolerances = params_data['tol'].get('objective', {})
    admm_params.tol['objective']['abs'] = float(objective_tolerances.get('abs', admm_params.tol['objective']['abs']))
    admm_params.tol['objective']['rel'] = float(objective_tolerances.get('rel', admm_params.tol['objective']['rel']))
    admm_params.num_max_iters = int(params_data['num_max_iters'])
    admm_params.minimum_consecutive_converged_cycles = int(params_data.get('minimum_consecutive_converged_cycles', admm_params.minimum_consecutive_converged_cycles))
    admm_params.shared_ess_normalization_floor_mva = float(params_data.get('shared_ess_normalization_floor_mva', admm_params.shared_ess_normalization_floor_mva))
    admm_params.adaptive_penalty = bool(params_data['adaptive_penalty'])
    penalty_update = params_data.get('penalty_update', {})
    for key in admm_params.penalty_update:
        if key in penalty_update:
            if admm_params.penalty_update['residual_balance_ratio_pf_decrease'] <= 1.0:
                raise ValueError('ADMM residual_balance_ratio_pf_decrease must be greater than 1.')
            admm_params.penalty_update[key] = float(penalty_update[key])

    if admm_params.minimum_consecutive_converged_cycles < 1:
        raise ValueError('ADMM minimum_consecutive_converged_cycles must be at least 1.')
    if admm_params.shared_ess_normalization_floor_mva <= 0.00:
        raise ValueError('ADMM shared-ESS normalization floor must be positive.')
    if admm_params.penalty_update['residual_balance_ratio'] <= 1.0:
        raise ValueError('ADMM residual_balance_ratio must be greater than 1.')
    if admm_params.penalty_update['increase_factor'] <= 1.0:
        raise ValueError('ADMM increase_factor must be greater than 1.')
    if admm_params.penalty_update['decrease_factor'] <= 1.0:
        raise ValueError('ADMM decrease_factor must be greater than 1.')
    if admm_params.penalty_update['min'] <= 0.0:
        raise ValueError('ADMM minimum penalty must be positive.')
    if admm_params.penalty_update['max'] < admm_params.penalty_update['min']:
        raise ValueError('ADMM maximum penalty must not be smaller than the minimum penalty.')
    admm_params.rho['v'] = params_data['rho']['v']
    admm_params.rho['pf'] = params_data['rho']['pf']
    admm_params.rho['ess'] = params_data['rho']['ess']
    if 'v' in params_data['previous_iteration']:
        if bool(params_data['previous_iteration']['v']):
            print('[WARNING] Previous iteration interface voltage magnitude variables not implemented!')
    if 'pf' in params_data['previous_iteration']:
        if bool(params_data['previous_iteration']['pf']):
            print('[WARNING] Previous iteration interface power flow variables not implemented!')
    admm_params.previous_iter['ess']['tso'] = bool(params_data['previous_iteration']['ess']['tso'])
    admm_params.previous_iter['ess']['dso'] = bool(params_data['previous_iteration']['ess']['dso'])
    if admm_params.previous_iter['ess']['tso'] or admm_params.previous_iter['ess']['dso']:
        admm_params.rho_previous_iter['ess'] = params_data['rho_previous_iter']['ess']
