# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = {'consensus': {'v': 0.1e-2, 'pf': 0.1e-2, 'ess': 1e-2},
                    'stationarity': {'v': 0.5e-2, 'pf': 0.5e-2, 'ess': 5e-2}}
        self.num_max_iters = 1000
        self.adaptive_penalty = False
        self.previous_iter = {'v': False, 'pf': False, 'ess': False}
        self.rho = {'v': dict(), 'pf': dict(), 'ess': dict()}

    def read_parameters_from_file(self, params_data):
        _read_parameters_from_file(self, params_data)


def _read_parameters_from_file(admm_params, params_data):
    admm_params.tol['consensus']['v'] = float(params_data['tol']['consensus']['v'])
    admm_params.tol['consensus']['pf'] = float(params_data['tol']['consensus']['pf'])
    admm_params.tol['consensus']['ess'] = float(params_data['tol']['consensus']['ess'])
    admm_params.tol['stationarity']['v'] = float(params_data['tol']['stationarity']['v'])
    admm_params.tol['stationarity']['pf'] = float(params_data['tol']['stationarity']['pf'])
    admm_params.tol['stationarity']['ess'] = float(params_data['tol']['stationarity']['ess'])
    admm_params.num_max_iters = int(params_data['num_max_iters'])
    admm_params.adaptive_penalty = bool(params_data['adaptive_penalty'])
    admm_params.previous_iter['v'] = bool(params_data['previous_iteration']['v'])
    admm_params.previous_iter['pf'] = bool(params_data['previous_iteration']['pf'])
    admm_params.previous_iter['ess'] = bool(params_data['previous_iteration']['ess'])
    admm_params.rho['v'] = params_data['rho']['v']
    admm_params.rho['pf'] = params_data['rho']['pf']
    admm_params.rho['ess'] = params_data['rho']['ess']
