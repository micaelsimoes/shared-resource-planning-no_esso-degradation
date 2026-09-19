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
            # P5.15 Addendum 21 (frozen spec v10,
            # data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json,
            # `production_change`): optional, per-channel CONDITIONAL
            # exemption -- a channel is held exempt ("exempt (fixed)")
            # until its own Boyd `dual_ratio` (full s/eps_dual) has been
            # strictly below a configured threshold for a configured
            # number of CONSECUTIVE cycles, at which point the exemption
            # lifts, one-way (never re-exempted), and the channel is
            # balanced by the standard rule from that same update on.
            # Dict keyed by channel ('v', 'pf', 'ess'), each value
            # `{'dual_ratio_below': <float > 0>, 'consecutive_cycles':
            # <int >= 1>}`. A channel may not appear in BOTH this dict and
            # `balancing_exempt_channels` above (validated below). Default
            # {} (empty dict): no channel conditionally exempt, so every
            # other case study, channel and arm that does not set this key
            # is byte-identical to pre-v10 behaviour.
            'balancing_exempt_until': {},
        }
        self.balancing_exempt_channels_source = 'default'
        self.balancing_exempt_until_source = 'default'
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

        # P5.15 Step 3.6 (PLANNER_BRIEF_2026-09-13.md Addendum 21 item 4,
        # WORKER_REPORT_S36_CLONE_CAPTURE.md): the TSO's FrozenSMOPF
        # pre-solve snapshot (failure diagnostics + the cycle-7 comparator)
        # used to be captured by `model.clone()`-ing the whole TSO block on
        # EVERY cycle, unconditionally (WORKER_REPORT_S36_TIMING_MEASUREMENT.md
        # D-Step). 'lightweight' (default): one clone per (year, day) TSO
        # block, taken ONCE per `_run_operational_planning` call before the
        # ADMM loop starts, then a `clone()`-free per-cycle capture of only
        # the mutable Param/Var/Suffix/active-constraint state
        # (`network.capture_block_mutable_state`); a snapshot is rebuilt
        # on demand -- only when a failure or the cycle-7 comparator
        # actually needs to be written -- by replaying that capture onto a
        # fresh clone of the pristine base
        # (`network.apply_block_mutable_state`). 'legacy_clone' restores
        # the pre-Step-3.6 behaviour exactly (clone every cycle, every TSO
        # block, via `NetworkData.optimize`'s `failure_snapshot_callback`/
        # `pre_solve_snapshot_callback`) -- kept for regression comparison
        # and as a fallback; not wired to any case-file key, set
        # programmatically only.
        # P5.15 Addendum 27 item 5(a) adds a THIRD legal value, 'off': no
        # pristine base is built at all and no per-cycle capture is taken,
        # so the FrozenSMOPF diagnostics (failure snapshots and the cycle-7
        # comparator) are NOT produced -- a failed block instead prints a
        # warning naming this mode and writes a small JSON marker file in
        # the FrozenSMOPF directory (never a `.pkl`, so no snapshot scanner
        # can mistake it for one). Motivation: at paper scale the pristine
        # clones alone added >= 5.7 GiB and were the stage that crossed the
        # 24 GiB build watchdog (`data/SRP1/Results/P515S44/
        # scale_measurement/paper_build/watchdog_abort_build.json`). DEFAULT
        # STAYS 'lightweight': campaign and verification runs keep today's
        # behaviour bit-for-bit. Legal values are validated once per
        # `_run_operational_planning` call
        # (`shared_resources_planning._validate_snapshot_capture_modes`),
        # which also refuses 'off' together with `persistent_workers`
        # enabled (that path clones a block per solve). Not wired to any
        # case-file key, set programmatically only.
        self.tso_snapshot_capture_mode = 'lightweight'

        # P5.15 Addendum 23/24, Step 3.6 persistent-worker bounded task item
        # 3 (`WORKER_REPORT_S36_CLONE_CAPTURE.md` Q2 / `WORKER_REPORT_
        # S36_TIMING_10CYC.md` Defect 1 finding): the SAME clone -> capture
        # replacement as `tso_snapshot_capture_mode` above, applied to the
        # DSO node-7 FrozenSMOPF pre-solve snapshot (failure diagnostics +
        # the cycle-7 comparator) -- the only DSO block that ever wires
        # `NetworkData.optimize`'s snapshot callbacks (every other node
        # passes `None`/`None` and already never clones). 'lightweight'
        # (default): one clone per (year, day) node-7 block, taken ONCE per
        # `_run_operational_planning` call, then a `clone()`-free per-cycle
        # capture (`network.capture_block_mutable_state`); a snapshot is
        # rebuilt on demand, only when a failure or the cycle-7 comparator
        # actually needs one written, by replaying the capture onto a fresh
        # clone of the pristine base (`network.apply_block_mutable_state`).
        # 'legacy_clone' restores the pre-task behaviour exactly (clone
        # every cycle for node 7, via `NetworkData.optimize`'s
        # `failure_snapshot_callback`/`pre_solve_snapshot_callback`) -- kept
        # for regression comparison and as a fallback; not wired to any
        # case-file key, set programmatically only.
        # P5.15 Addendum 27 item 5(a): the same third legal value 'off' as
        # `tso_snapshot_capture_mode` above (no pristine node-7 base, no
        # per-cycle capture, a warning plus a JSON marker instead of a
        # snapshot on a failed node-7 block), with the same default
        # ('lightweight'), the same validation, and the same refusal to run
        # with `persistent_workers` enabled.
        self.dso_snapshot_capture_mode = 'lightweight'

        # P5.15 Addendum 22 item (2), Step 3.6 (PLANNER_BRIEF_2026-09-13.md
        # Addendum 18, `admm_persistent_workers.py`): within-cycle block
        # parallelism via persistent, single-threaded worker processes (one
        # fixed subset of the 36 DSO / 12 TSO / 3 ESSO blocks per worker,
        # models built once, only consensus/dual state and solved blocks
        # exchanged per cycle -- see `admm_persistent_workers.py`'s module
        # docstring for the full design). DEFAULT OFF
        # (`enabled: False`): `_run_operational_planning` never constructs a
        # `PersistentWorkerPool` and every per-cycle dispatch call is
        # byte-for-byte the existing serial code path. Not wired to any
        # case-file key; set programmatically only, exactly like
        # `tso_snapshot_capture_mode` above.
        self.persistent_workers = {
            'enabled': False,
            'num_workers': 8,
        }

        # P5.15 Step 3.7 (PLANNER_BRIEF_2026-09-13.md Addendum 22/23; frozen
        # spec `data/SRP1/Results/P515S41/
        # frozen_s41_hull_aa_spec_v12_6e5a546f.json`, `item4_step_3_7_anderson`):
        # type-II Anderson acceleration on the joint (z, u) ADMM consensus/
        # dual iterate -- see `admm_anderson_acceleration.py`'s module
        # docstring for the full design. DEFAULT OFF (`enabled: False`):
        # `_run_operational_planning` never calls anything in
        # `admm_anderson_acceleration` (its `if aa_enabled:` guards), so the
        # existing serial ADMM cycle is byte-for-byte unchanged. Since P5.15
        # Addendum 27 item 1 this dict MAY be set from the case file: an
        # OPTIONAL `admm.anderson_acceleration` object (keys `enabled`,
        # `memory`, `regularization`, `reject_policy`), validated and merged
        # onto these defaults by `_read_parameters_from_file`; when the key
        # is absent the dict below is left exactly as is (no key added), so
        # every case study without it loads unchanged. It may also still be
        # set programmatically. `memory`
        # (m, number of secant columns) and `regularization` (Tikhonov
        # lambda on the least-squares solve) are the frozen-spec values
        # (5, 1e-10); kept configurable here only so a zero-solve check can
        # exercise small memories without constructing a full case study.
        # P5.15 Addendum 25 item 2: an OPTIONAL key `reject_policy`
        # ('clear_memory' default = the Step 3.7 behaviour, or 'keep_memory')
        # may be added to this dict (programmatically, or from the case
        # file); it is deliberately NOT a default key here, so every existing
        # settings dict keeps meaning the Step 3.7 behaviour (see
        # `admm_anderson_acceleration.py`).
        self.anderson_acceleration = {
            'enabled': False,
            'memory': 5,
            'regularization': 1e-10,
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
    _non_float_penalty_update_keys = _cycle_index_keys + ('balancing_exempt_channels', 'balancing_exempt_until')
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

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Addendum 21 (frozen spec v10, `production_change`): optional
    # per-channel CONDITIONAL balancing exemption. Absent/empty key -> {}
    # (default), source 'default' -- no behaviour change for any case
    # study that does not set this key.
    if 'balancing_exempt_until' in penalty_update and penalty_update['balancing_exempt_until']:
        balancing_exempt_until = penalty_update['balancing_exempt_until']
        if not isinstance(balancing_exempt_until, dict):
            raise ValueError('ADMM penalty_update.balancing_exempt_until must be a dict.')
        invalid_channels = [c for c in balancing_exempt_until if c not in ('v', 'pf', 'ess')]
        if invalid_channels:
            raise ValueError(
                "ADMM penalty_update.balancing_exempt_until keys must be drawn from "
                f"{{'v', 'pf', 'ess'}}; got invalid entries: {invalid_channels}.")
        overlap = sorted(set(balancing_exempt_until) & set(admm_params.penalty_update['balancing_exempt_channels']))
        if overlap:
            raise ValueError(
                "ADMM penalty_update: a channel may not be in BOTH balancing_exempt_channels "
                f"and balancing_exempt_until; overlap: {overlap}.")
        normalized_exempt_until = {}
        for channel, channel_cfg in balancing_exempt_until.items():
            if not isinstance(channel_cfg, dict):
                raise ValueError(
                    f"ADMM penalty_update.balancing_exempt_until['{channel}'] must be a dict "
                    "with keys 'dual_ratio_below' and 'consecutive_cycles'.")
            if 'dual_ratio_below' not in channel_cfg or 'consecutive_cycles' not in channel_cfg:
                raise ValueError(
                    f"ADMM penalty_update.balancing_exempt_until['{channel}'] must set both "
                    "'dual_ratio_below' and 'consecutive_cycles'.")
            dual_ratio_below = float(channel_cfg['dual_ratio_below'])
            consecutive_cycles = int(channel_cfg['consecutive_cycles'])
            if dual_ratio_below <= 0.0:
                raise ValueError(
                    f"ADMM penalty_update.balancing_exempt_until['{channel}'].dual_ratio_below "
                    "must be positive.")
            if consecutive_cycles < 1:
                raise ValueError(
                    f"ADMM penalty_update.balancing_exempt_until['{channel}'].consecutive_cycles "
                    "must be at least 1.")
            normalized_exempt_until[channel] = {
                'dual_ratio_below': dual_ratio_below, 'consecutive_cycles': consecutive_cycles,
            }
        admm_params.penalty_update['balancing_exempt_until'] = normalized_exempt_until
        admm_params.balancing_exempt_until_source = 'case_file'
    else:
        admm_params.penalty_update['balancing_exempt_until'] = {}
        admm_params.balancing_exempt_until_source = 'default'

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
            # P5.15 Addendum 20 / frozen spec v9 (data/SRP1/Results/P515S38/
            # frozen_s38_pf_pace_spec_v9_7a2b4ab7.json, arms s38_A_tau0 /
            # s38_C_combined): tau = 0.0 (proximal term off on every TSO
            # channel) is a deliberately authorized configuration, not an
            # error -- relaxed from "must be positive" to "must be
            # non-negative". Verified before this change: no code path
            # divides by tau or by any prox_gamma_* Param (gamma only
            # multiplies, e.g. `(model[year][day].prox_gamma_v / 2) *
            # proximal_v ** 2` in `update_transmission_model_to_admm`;
            # `proximal_share = s_proximal_part / s` in
            # `get_admm_boyd_residual_metrics` divides by the residual norm
            # `s`, not by gamma or tau, and is already guarded `if s > 0.0
            # else 0.0`), so tau/gamma == 0 is numerically safe.
            if tau < 0.0:
                raise ValueError('ADMM proximal_regularization.tso.tau must be non-negative.')
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

    # ------------------------------------------------------------------------------------------------------------------
    # P5.15 Addendum 27 item 1 (frozen spec v15 `data/SRP1/Results/P515S45/
    # frozen_s45_phaseA_spec_v15_5feefd7b.json`): optional Anderson
    # acceleration settings. Absent key -> nothing is touched (the __init__
    # default dict, with no `reject_policy` key, stays exactly as it is), so
    # every case study without this key loads unchanged. Present -> must be
    # an object with keys drawn from {enabled, memory, regularization,
    # reject_policy}, merged onto the current dict.
    if 'anderson_acceleration' in params_data:
        admm_params.anderson_acceleration = _read_anderson_acceleration(
            admm_params.anderson_acceleration, params_data['anderson_acceleration'])


def _read_anderson_acceleration(current_settings, aa_data):
    import admm_anderson_acceleration  # numpy only; imported only when the case file carries the key
    supported_keys = ('enabled', 'memory', 'regularization', 'reject_policy')
    if not isinstance(aa_data, dict):
        raise ValueError('ADMM anderson_acceleration must be an object (dict).')
    unknown_keys = sorted(set(aa_data) - set(supported_keys))
    if unknown_keys:
        raise ValueError(
            f'ADMM anderson_acceleration: unsupported keys {unknown_keys}; supported: {list(supported_keys)}.')
    settings = dict(current_settings)
    if 'enabled' in aa_data:
        if not isinstance(aa_data['enabled'], bool):
            raise ValueError('ADMM anderson_acceleration.enabled must be a bool.')
        settings['enabled'] = aa_data['enabled']
    if 'memory' in aa_data:
        if isinstance(aa_data['memory'], bool) or not isinstance(aa_data['memory'], int):
            raise ValueError('ADMM anderson_acceleration.memory must be an int.')
        if aa_data['memory'] < 1:
            raise ValueError('ADMM anderson_acceleration.memory must be at least 1.')
        settings['memory'] = aa_data['memory']
    if 'regularization' in aa_data:
        if isinstance(aa_data['regularization'], bool) or not isinstance(aa_data['regularization'], float):
            raise ValueError('ADMM anderson_acceleration.regularization must be a float.')
        if aa_data['regularization'] < 0.0:
            raise ValueError('ADMM anderson_acceleration.regularization must be non-negative.')
        settings['regularization'] = aa_data['regularization']
    if 'reject_policy' in aa_data:
        if aa_data['reject_policy'] not in admm_anderson_acceleration.REJECT_POLICIES:
            raise ValueError(
                'ADMM anderson_acceleration.reject_policy must be one of '
                f'{admm_anderson_acceleration.REJECT_POLICIES}; got {aa_data["reject_policy"]!r}.')
        settings['reject_policy'] = aa_data['reject_policy']
    return settings
