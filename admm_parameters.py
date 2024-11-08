# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = {'consensus': {'v': 1e-2, 'pf': 1e-3, 'ess': 1e-3},
                    'stationarity': {'v': 1e-2, 'pf': 1e-3, 'ess': 1e-3}}
        self.num_max_iters = 1000
        self.adaptive_penalty = False
        self.previous_iter = {'v': False, 'pf': False, 'ess': False}
        self.rho = {'current': {'v': dict(), 'pf': dict(), 'ess': dict()},
                    'prev': {'v': dict(), 'pf': dict(), 'ess': dict()}}

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
    admm_params.rho['current']['v'] = params_data['rho']['current']['v']
    admm_params.rho['current']['pf'] = params_data['rho']['current']['pf']
    admm_params.rho['current']['ess'] = params_data['rho']['current']['ess']
    admm_params.rho['prev']['v'] = params_data['rho']['prev']['v']
    admm_params.rho['prev']['pf'] = params_data['rho']['prev']['pf']
    admm_params.rho['prev']['ess'] = params_data['rho']['prev']['ess']
