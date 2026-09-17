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
            'objective': {'abs': 1e4, 'rel': 5e-4},
            # P5.15 Step 3.2 (Boyd et al. 2011 Sec. 3.3.1) stopping-rule
            # tolerances: eps_pri = sqrt(p)*eps_abs + eps_rel*max(||x||,||z||),
            # eps_dual = sqrt(n)*eps_abs + eps_rel*||y||. Defaults used when the
            # case file has no `admm.tol.boyd` block (`boyd_eps_source`
            # records which applied).
            'boyd': {'eps_abs': 1e-5, 'eps_rel': 1e-4}}
        self.boyd_eps_source = 'default'
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
            # P5.15 Step 3.2 E2 (frozen spec v3,
            # data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json,
            # `balancing_rule_3_3a.freeze`): optional cycle index after
            # which rho (and, under the tied gamma policy, gamma) adaptation
            # is held on every channel. None (default) means never freeze --
            # other case studies are unaffected. Superseded for SRP1 by the
            # spec v4 per-channel rule below (kept wired for any case study
            # that still sets it and not the v4 keys).
            'freeze_after_cycle': None,
            # P5.15 Step 3.4 (frozen spec v4,
            # data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json,
            # `changes_from_v3.c_rho_policy.freeze`): a channel freezes once
            # rho has been unchanged for `freeze_after_unchanged_cycles`
            # consecutive cycles AND balancing has acted on that channel at
            # least once; `freeze_backstop_cycle` freezes every channel
            # regardless, from that cycle on. Both None (default) means
            # neither v4 mechanism is active -- other case studies are
            # unaffected. Additive to `freeze_after_cycle` above (whichever
            # fires first holds the channel); a channel frozen at the rho
            # clamp is recorded as `rho_at_clamp`, a gate-failure flag, not
            # an exception.
            'freeze_after_unchanged_cycles': None,
            'freeze_backstop_cycle': None,
            # P5.15 Addendum 19 (frozen spec v8,
            # data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json,
            # `production_change`): optional list of channels ('v', 'pf',
            # 'ess') permanently EXEMPT from residual balancing -- never
            # increased or decreased; `_update_admm_penalties` records the
            # action as "exempt (fixed)", freezes `freeze_state[channel]`
            # from the first call with `at_clamp` permanently False and an
            # `exempt` marker True, and never lets that channel trip the
            # clamp flag or the freeze-streak/backstop bookkeeping. Default
            # [] (empty list): no channel exempt, so every other case study
            # and every previous arm is unchanged.
            'balancing_exempt_channels': [],
        }
        self.balancing_exempt_channels_source = 'default'
        self.rho = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.rho_previous_iter = {'v': dict(), 'pf': dict(), 'ess': dict()}
        self.proximal_regularization = {
            'enabled': False,
            'tso': {
                'enabled': False,
                'gamma': {
                    'v': 1.0,
                    'pf': 1.0,
                    'ess': 1.0,
                },
                # P5.15 Step 3.2 E2 (frozen spec v3, `gamma_policy`):
                # 'fixed' (default) keeps gamma at the configured value
                # above, exactly as before this change. 'tied_to_rho' sets
                # gamma_c = tau * rho_c per channel (Deng and Yin, 2016).
                # `tau` is only used when tied.
                'gamma_policy': 'fixed',
                'tau': 1.0,
            },
            'dso': {
                'enabled': False,
                'gamma': {
                    'v': 1.0,
                    'pf': 1.0,
                    'ess': 1.0,
                },
            },
        }

        # ------------------------------------------------------------------------------------------------------------
        # P5.15 Step 3.4 (frozen spec v4, `changes_from_v3.b_sigma_fixed`,
        # `.a_D5_esso_scaling`, `.d_ess_reference_rating`): optional,
        # per-case-study keys. Every default below reproduces exactly the
        # pre-3.4 behaviour, so other case studies (and SRP1 itself, if the
        # keys were removed) load unchanged.
        # ------------------------------------------------------------------------------------------------------------
        # Fixed common ADMM objective scale (sigma). None (default) means
        # `_compute_common_admm_objective_scale` is used every run, exactly
        # as before. When set, that computed value is still obtained (the
        # function stays wired) and asserted within `objective_scale_assert_factor`
        # of the fixed value, failing loudly outside the calibration range.
        self.objective_scale = None
        self.objective_scale_source = 'default'
        self.objective_scale_assert_factor = 3.0

        # Fixed shared-ESS reference rating (S_ref, MVA), applied at every
        # ADMM shared-ESS normalization call site. None (default) means the
        # current per-agent `2*max(S, shared_ess_normalization_floor_mva)`
        # normalization, unchanged.
        self.shared_ess_reference_rating_mva = None
        self.shared_ess_reference_rating_source = 'default'

        # P5.15 Addendum 16 items 2-3 (frozen spec v6,
        # data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json,
        # `initialization.case_file`): how the shared-ESS ADMM consensus is
        # initialized on a FRESH candidate evaluation (`initial_state is
        # None`). 'standalone' (default, absent key) preserves current
        # behaviour exactly -- z starts at the average of the three agents'
        # zero-dispatch standalone solutions
        # (`_initialize_shared_ess_consensus`). 'price_taker' additionally
        # runs `_initialize_shared_ess_from_price_taker`
        # (`shared_resources_planning.py`), which sets INITIAL VALUES ONLY
        # (never fixes) for z, the agent copies, the TSO proximal centres
        # and the ESSO warm state from a history-free price-taking LP
        # (`shared_ess_price_taker.solve_price_taker_schedule`). Other case
        # studies are unaffected by this key's absence.
        self.shared_ess_initialization = 'standalone'
        self.shared_ess_initialization_source = 'default'

        # ESSO augmented-Lagrangian scale (D5, Addendum 15 item 5(a)):
        # multiplies ONLY the ESSO's AL terms (the two dual terms and the
        # two rho/2 terms), never the ESSO base objective. 'fixed' with
        # value 1.0 and source 'default' (the state below) reproduces the
        # pre-3.4 ESSO objective bit-for-bit -- the multiplication is not
        # even constructed in that case (see
        # `update_shared_energy_storage_model_to_admm`). 'mode' may also be
        # 'sigma_over_median_block_weight', in which case the case-file
        # value is ignored and the scale is derived at runtime as
        # sigma / median(TSO and DSO block weights).
        self.esso_al_scale = {
            'mode': 'fixed',
            'value': 1.0,
            'source': 'default',
        }

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

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Step 3.2 Boyd stopping-rule tolerances (optional; other case
    # studies keep loading unchanged and fall back to the defaults set in
    # __init__).
    boyd_tolerances = params_data['tol'].get('boyd')
    if boyd_tolerances is not None:
        admm_params.tol['boyd']['eps_abs'] = float(boyd_tolerances['eps_abs'])
        admm_params.tol['boyd']['eps_rel'] = float(boyd_tolerances['eps_rel'])
        admm_params.boyd_eps_source = 'case_file'
    else:
        admm_params.boyd_eps_source = 'default'
    if admm_params.tol['boyd']['eps_abs'] <= 0.0:
        raise ValueError('ADMM Boyd eps_abs must be positive.')
    if admm_params.tol['boyd']['eps_rel'] <= 0.0:
        raise ValueError('ADMM Boyd eps_rel must be positive.')

    admm_params.num_max_iters = int(params_data['num_max_iters'])
    admm_params.minimum_consecutive_converged_cycles = int(params_data.get('minimum_consecutive_converged_cycles', admm_params.minimum_consecutive_converged_cycles))
    admm_params.shared_ess_normalization_floor_mva = float(params_data.get('shared_ess_normalization_floor_mva', admm_params.shared_ess_normalization_floor_mva))
    admm_params.adaptive_penalty = bool(params_data['adaptive_penalty'])
    penalty_update = params_data.get('penalty_update', {})
    _cycle_index_keys = ('freeze_after_cycle', 'freeze_after_unchanged_cycles', 'freeze_backstop_cycle')
    # P5.15 Addendum 19 (frozen spec v8): 'balancing_exempt_channels' is a
    # list of channel names, not a float penalty-update coefficient; handled
    # separately below, alongside the freeze-cycle keys.
    _non_float_penalty_update_keys = _cycle_index_keys + ('balancing_exempt_channels',)
    for key in admm_params.penalty_update:
        # The three freeze-cycle keys are optional integer cycle indices (or
        # None), not float penalty-update coefficients; handled separately
        # below.
        if key in _non_float_penalty_update_keys:
            continue
        if key in penalty_update:
            admm_params.penalty_update[key] = float(penalty_update[key])
    if 'freeze_after_cycle' in penalty_update and penalty_update['freeze_after_cycle'] is not None:
        admm_params.penalty_update['freeze_after_cycle'] = int(penalty_update['freeze_after_cycle'])
    else:
        admm_params.penalty_update['freeze_after_cycle'] = None
    if (
            admm_params.penalty_update['freeze_after_cycle'] is not None and
            admm_params.penalty_update['freeze_after_cycle'] < 0
    ):
        raise ValueError('ADMM penalty_update.freeze_after_cycle must be non-negative.')

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Step 3.4 (frozen spec v4, `changes_from_v3.c_rho_policy.freeze`):
    # per-channel unchanged-streak freeze and the global backstop. Both
    # optional; None (default) means neither v4 mechanism is active.
    if 'freeze_after_unchanged_cycles' in penalty_update and penalty_update['freeze_after_unchanged_cycles'] is not None:
        admm_params.penalty_update['freeze_after_unchanged_cycles'] = int(penalty_update['freeze_after_unchanged_cycles'])
    else:
        admm_params.penalty_update['freeze_after_unchanged_cycles'] = None
    if (
            admm_params.penalty_update['freeze_after_unchanged_cycles'] is not None and
            admm_params.penalty_update['freeze_after_unchanged_cycles'] < 1
    ):
        raise ValueError('ADMM penalty_update.freeze_after_unchanged_cycles must be at least 1.')

    if 'freeze_backstop_cycle' in penalty_update and penalty_update['freeze_backstop_cycle'] is not None:
        admm_params.penalty_update['freeze_backstop_cycle'] = int(penalty_update['freeze_backstop_cycle'])
    else:
        admm_params.penalty_update['freeze_backstop_cycle'] = None
    if (
            admm_params.penalty_update['freeze_backstop_cycle'] is not None and
            admm_params.penalty_update['freeze_backstop_cycle'] < 1
    ):
        raise ValueError('ADMM penalty_update.freeze_backstop_cycle must be at least 1.')

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Addendum 19 (frozen spec v8, `production_change`): optional
    # per-channel balancing exemption. Absent key -> [] (default), source
    # 'default' -- no behaviour change for any case study that does not set
    # this key.
    if 'balancing_exempt_channels' in penalty_update and penalty_update['balancing_exempt_channels'] is not None:
        balancing_exempt_channels = penalty_update['balancing_exempt_channels']
        if not isinstance(balancing_exempt_channels, list):
            raise ValueError('ADMM penalty_update.balancing_exempt_channels must be a list.')
        invalid_channels = [c for c in balancing_exempt_channels if c not in ('v', 'pf', 'ess')]
        if invalid_channels:
            raise ValueError(
                "ADMM penalty_update.balancing_exempt_channels entries must be drawn from "
                f"{{'v', 'pf', 'ess'}}; got invalid entries: {invalid_channels}.")
        admm_params.penalty_update['balancing_exempt_channels'] = list(balancing_exempt_channels)
        admm_params.balancing_exempt_channels_source = 'case_file'
    else:
        admm_params.penalty_update['balancing_exempt_channels'] = []
        admm_params.balancing_exempt_channels_source = 'default'

    if admm_params.minimum_consecutive_converged_cycles < 1:
        raise ValueError('ADMM minimum_consecutive_converged_cycles must be at least 1.')
    if admm_params.shared_ess_normalization_floor_mva <= 0.00:
        raise ValueError('ADMM shared-ESS normalization floor must be positive.')
    if admm_params.penalty_update['residual_balance_ratio'] <= 1.0:
        raise ValueError('ADMM residual_balance_ratio must be greater than 1.')
    if admm_params.penalty_update['residual_balance_ratio_pf_decrease'] <= 1.0:
        raise ValueError('ADMM residual_balance_ratio_pf_decrease must be greater than 1.')
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

    # ------------------------------------------------------------------------------------------------------------------
    # Proximal regularization
    proximal_data = params_data.get('proximal_regularization', {})
    admm_params.proximal_regularization['enabled'] = bool(proximal_data.get('enabled', admm_params.proximal_regularization['enabled']))
    for agent in ('tso', 'dso'):
        agent_data = proximal_data.get(agent, {})
        admm_params.proximal_regularization[agent]['enabled'] = bool(agent_data.get('enabled', admm_params.proximal_regularization[agent]['enabled']))
        gamma_data = agent_data.get('gamma', {})
        for group in ('v', 'pf', 'ess'):
            gamma = float(gamma_data.get(group, admm_params.proximal_regularization[agent]['gamma'][group]))
            if gamma < 0.0:
                raise ValueError(f'ADMM proximal gamma for {agent.upper()} {group.upper()} must be non-negative.')
            admm_params.proximal_regularization[agent]['gamma'][group] = gamma

        # ------------------------------------------------------------------------------------------------------------
        # P5.15 Step 3.2 E2 (frozen spec v3): TSO-only gamma policy. Other
        # agents (DSO) and other case studies are unaffected -- optional,
        # defaults preserve current behaviour exactly.
        if agent == 'tso':
            gamma_policy = agent_data.get('gamma_policy', admm_params.proximal_regularization['tso']['gamma_policy'])
            if gamma_policy not in ('fixed', 'tied_to_rho'):
                raise ValueError("ADMM proximal_regularization.tso.gamma_policy must be 'fixed' or 'tied_to_rho'.")
            admm_params.proximal_regularization['tso']['gamma_policy'] = gamma_policy

            tau = float(agent_data.get('tau', admm_params.proximal_regularization['tso']['tau']))
            if tau <= 0.0:
                raise ValueError('ADMM proximal_regularization.tso.tau must be positive.')
            admm_params.proximal_regularization['tso']['tau'] = tau

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Step 3.4 (frozen spec v4): fixed sigma, S_ref and the ESSO AL
    # scale. All optional; absent means the pre-3.4 behaviour, unchanged.
    # ------------------------------------------------------------------------------------------------------------------
    if 'objective_scale' in params_data and params_data['objective_scale'] is not None:
        objective_scale = float(params_data['objective_scale'])
        if objective_scale <= 0.0:
            raise ValueError('ADMM objective_scale must be positive.')
        admm_params.objective_scale = objective_scale
        admm_params.objective_scale_source = 'case_file'
    else:
        admm_params.objective_scale = None
        admm_params.objective_scale_source = 'default'

    objective_scale_assert_factor = float(params_data.get('objective_scale_assert_factor', admm_params.objective_scale_assert_factor))
    if objective_scale_assert_factor < 1.0:
        raise ValueError('ADMM objective_scale_assert_factor must be at least 1.')
    admm_params.objective_scale_assert_factor = objective_scale_assert_factor

    if 'shared_ess_reference_rating_mva' in params_data and params_data['shared_ess_reference_rating_mva'] is not None:
        shared_ess_reference_rating_mva = float(params_data['shared_ess_reference_rating_mva'])
        if shared_ess_reference_rating_mva <= 0.0:
            raise ValueError('ADMM shared_ess_reference_rating_mva must be positive.')
        admm_params.shared_ess_reference_rating_mva = shared_ess_reference_rating_mva
        admm_params.shared_ess_reference_rating_source = 'case_file'
    else:
        admm_params.shared_ess_reference_rating_mva = None
        admm_params.shared_ess_reference_rating_source = 'default'

    esso_al_scale_data = params_data.get('esso_al_scale')
    if esso_al_scale_data is None:
        admm_params.esso_al_scale = {'mode': 'fixed', 'value': 1.0, 'source': 'default'}
    elif isinstance(esso_al_scale_data, str):
        if esso_al_scale_data != 'sigma_over_median_block_weight':
            raise ValueError(
                "ADMM esso_al_scale, when a string, must be 'sigma_over_median_block_weight'."
            )
        admm_params.esso_al_scale = {
            'mode': 'sigma_over_median_block_weight',
            'value': None,
            'source': 'case_file',
        }
    else:
        esso_al_scale_value = float(esso_al_scale_data)
        if esso_al_scale_value <= 0.0:
            raise ValueError('ADMM esso_al_scale must be positive.')
        admm_params.esso_al_scale = {
            'mode': 'fixed',
            'value': esso_al_scale_value,
            'source': 'case_file',
        }

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Addendum 16 items 2-3 (frozen spec v6): optional shared-ESS
    # consensus initialization mode. Absent key -> 'standalone', source
    # 'default', no behavioural change for any case study.
    if 'shared_ess_initialization' in params_data and params_data['shared_ess_initialization'] is not None:
        shared_ess_initialization = str(params_data['shared_ess_initialization'])
        if shared_ess_initialization not in ('standalone', 'price_taker'):
            raise ValueError(
                "ADMM shared_ess_initialization must be 'standalone' or 'price_taker'.")
        admm_params.shared_ess_initialization = shared_ess_initialization
        admm_params.shared_ess_initialization_source = 'case_file'
    else:
        admm_params.shared_ess_initialization = 'standalone'
        admm_params.shared_ess_initialization_source = 'default'
