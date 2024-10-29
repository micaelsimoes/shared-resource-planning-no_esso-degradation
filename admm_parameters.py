# ======================================================================================================================
#  Class ADMM Parameters
# ======================================================================================================================
class ADMMParameters:

    def __init__(self):
        self.tol = {'consensus': {'v': 1e-3, 'pf': 1e-4, 'ess': 1e-4},
                    'stationarity': {'v': 1e-3, 'pf': 1e-4, 'ess': 1e-4}}
        self.num_max_iters = 1000
        self.adaptive_penalty = False
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
    admm_params.rho['v'] = params_data['rho']['v']
    admm_params.rho['pf'] = params_data['rho']['pf']
    admm_params.rho['ess'] = params_data['rho']['ess']
