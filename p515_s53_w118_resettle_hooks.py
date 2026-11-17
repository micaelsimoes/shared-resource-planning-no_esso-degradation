"""
P5.15 Addendum 57 (Decisions 2 and 3) and Addendum 54 Ruling 2, Planner task W118 -- the SRP1 RE-SETTLING CAMPAIGN
hooks (frozen stage spec `frozen_s53_resettle_spec_v1_<sha8>.json`).

WHAT THIS MODULE IS. A child-side, harness-installed set of pass-through wrappers on EIGHT production functions of
`shared_resources_planning` (module globals resolved by `_run_operational_planning` at call time -- the W98 / W101 /
W105 technique; this module is W105's `p515_s53_w105_settling_extension_hooks` adapted, and reuses its stateless capture
helpers and W101's replay comparison). No production module is edited. It is installed ONLY for a campaign-spec entry
that declares `settling_resettle` (`p515_s44_campaign_harness`, W118); the declaration enters the entry's eval key.

THE RUN IT SERVES (Addendum 54 Ruling 2 "re-run, not pin"; Addendum 57 Decisions 2 and 3). Ten SRP1 cells re-run from
cycle 1 under the CURRENT production configuration (the case file with AA keep_memory, the ESS ageing baseline C2, the
tight tail {True, 1e-6} from AA-off + 1 -- production's rule), with
  * GATED cells (the six Phase B neighbours and the F2 pair): cycles 1..k0 (k0 = the original run's first residual
    pass = N_old - 9) REPLAYED and gated BITWISE against the original record (W101's REPLAY_GATED_FIELDS as JSON text,
    in-cycle; the FIRST difference writes the cycle line and ABORTS the run). The tail first acts at k0 + 1, so the
    original (pre-tail) record and this run must agree through k0. Cycles k0 + 1..N_old: the overlap Q_new - Q_old is
    recorded (REPORT-ONLY, no gate);
  * UNGATED cells (the 2030 / 2035 year ladder, first C2 evaluations): no replay reference; the full trajectory;
  * for every cycle AFTER THE RUN'S FIRST RESIDUAL PASS k0_run (boyd_k True for the first time) the certifying regime
    HELD exactly as W101 / W105 hold it: AA OFF (production's own 'off' branch, even across a Boyd lapse), the tight tail
    ON, rho FROZEN; a lapse is recorded;
  * the settling stop rule VERSION 2 (`settling_criterion_v2.SettlingRuleV2`: N := k0_run; the amended monotone branch
    with |dQ| x L <= tau, L = 60; the gap clause |t_sum| <= tau / 2) as the ONLY way to end before its cap: on
    certification the certificate length goes to 0 and production's own exit test ends the loop at the end of THIS
    cycle (W98 / W101 mechanism). CAP: gated N_old + 100 (= the spec cap = production's num_max_iters); ungated
    min(k0_run + 109, 300) (dynamic: when the rule reaches it uncertified before the spec cap 300 the certificate length
    goes to 0 in the same way, recorded as stopped_by 'rule_cap');
  * per-cycle captures, WRITE-ONLY (pure reads; a capture error is recorded, never raised): W105's (a) Q by block and
    cost component, (b) sum |dx| of the ESS schedules for 14 families (and the raw schedules), (c) every raw Boyd field;
    W101's all-block line; and the PRICED INTERFACE-CONSENSUS GAP t_sum(k) = sum over interface nodes, years, days and
    periods of w(y, d) * pi(n, y, d, t) * (p_DSO - p_TSO), p_DSO / p_TSO = consensus_vars['pf'][dso|tso]['current']
    [n][y][d]['p'][t] (MW; the copies W112 read from the pf stride), pi = production's `_expected_market_price`
    (tso_model[y][d], network, t), w = production's `_get_admm_block_weight`(transmission network, y, d) -- the
    functions production's terminal `interface_settlement_detail_s31c.json` identity uses; Q_cc = Q + t_sum.

THE WRAPPERS (each calls production; outside a hold each passes its arguments through as the SAME objects and returns
production's value UNCHANGED):
  1. `_capture_convergence_depth_tail_baseline` (once): certificate length -> 10**9.
  2. `_apply_convergence_depth_tail`: cycle tracker; tail hold (a) after k0_run; restore at loop exit.
  3. `get_admm_boyd_residual_metrics`: production's value returned unchanged (same object); captures (b), (c), t_sum.
  4. `_anderson_acceleration_cycle_step`: boyd_k; the first residual pass; AA hold after k0_run.
  5. `_convergence_depth_tail_next_state`: tail hold (b) after k0_run.
  6. `_get_operational_recourse_components`: Q_k; the all-block line; capture (a); the stop rule.
  7. `_update_admm_penalties`: rho hold after k0_run.
  8. `_get_admm_efc_per_day_max`: finalises the cycle -- the replay gate (gated, cycles <= k0), the overlap, the
     cycle line, the creep line.

Zero solves: nothing here solves or builds a model. Stdlib only at import (the harness parent imports it for keys).
"""
import copy
import inspect
import json
import os
import time
from contextlib import contextmanager

import gate_result_io as GRIO
import settling_criterion_v2 as SC2
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105

SCHEMA = 'p515_s53_w118_settling_resettle_v1'
OPTION_NAME = 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING RUN (W118, frozen_s53_resettle_spec_v1) -- current production configuration (C2, tight '
         'tail); a gated cell replayed bitwise against its original record through its first residual pass k0 (abort '
         'on divergence), an ungated cell recorded in full; the certifying regime held after the run\'s k0 (AA off, '
         'tight tail on, rho frozen); settling rule v2 (gap clause, amended monotone branch) until it certifies or the '
         'cap; W105 captures plus the per-cycle priced interface gap t_sum')
P_MAX = 30                  # instance-measured (W102 x0 P_hat 29, W103 unit P_hat 30); verified at freeze
L_MONO = 2 * P_MAX
CAP_AFTER_N_OLD = 100
CAP_AFTER_K0 = SC2.CAP_AFTER_K0
CAP_CEILING = SC2.CAP_CEILING
CERTIFICATION_DISABLED_THRESHOLD = C101.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = 0  # consecutive_converged_cycles >= 0 always: the loop exits at the end of this cycle
AA_OFF_ACTION = C101.AA_OFF_ACTION
CHANNELS = C101.CHANNELS
BOYD_RATIO_FIELDS = C101.BOYD_RATIO_FIELDS
REPLAY_GATED_FIELDS = C101.REPLAY_GATED_FIELDS
REPLAY_POST_RUN_ONLY_FIELDS = C101.REPLAY_POST_RUN_ONLY_FIELDS
WRAPPED = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail', 'get_admm_boyd_residual_metrics',
           '_anderson_acceleration_cycle_step', '_convergence_depth_tail_next_state', '_update_admm_penalties',
           '_get_operational_recourse_components', '_get_admm_efc_per_day_max')
CYCLE_FILE = 'resettle_cycle_record.jsonl'
BLOCKS_FILE = C101.BLOCKS_FILE                    # recourse_blocks_all.jsonl, the W101 line format unchanged
CREEP_FILE = E105.CREEP_FILE                      # creep_diagnostic_per_cycle.jsonl, the W105 line format
ESS_SCHEDULE_FILE = E105.ESS_SCHEDULE_FILE        # ess_schedule_per_cycle.jsonl, the W105 line format
DECISION_FILE = 'resettle_decision.json'
SUMMARY_KEY = 'settling_resettle'
FORBIDDEN_DECLARATION_KEYS = ('early_stop',)
# The ONLY three assignments of the certificate length in this module (asserted from source): the disable at the tail
# baseline, the restore at the loop exit, and the end of the run by the rule (certification, or its dynamic cap).
CERTIFICATE_LENGTH_WRITES = (
    'admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD',
    'st.params.minimum_consecutive_converged_cycles = st.threshold_original',
    'st.params.minimum_consecutive_converged_cycles = SETTLING_END_THRESHOLD',
)
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the ten cells (original records: committed, in their campaign manifests; verified at freeze and before the run) --
_RES = os.path.join('data', 'SRP1', 'Results')
_S47 = os.path.join(_RES, 'P515S47', 'campaign_s47_phase_b')
_S51 = os.path.join(_RES, 'P515S51', 'campaign_s51_f2_phase_b')
_S53F2 = os.path.join(_RES, 'P515S53', 'campaign_s53_f2_certificate_r1')
_S45 = os.path.join(_RES, 'P515S45', 'campaign_s45_a1b')
CELLS = {
    'f2_challenger': {
        'group': 'f2', 'gated': True, 'orig_root': _S53F2, 'orig_spec': 'campaign_spec_s53_f2_certificate_r1_803571c0.json',
        'orig_campaign_id': 's53_f2_certificate_r1',
        'orig_eval_key': 'e28de4acf80bbbe9f4b176296fc16b17929804f72255249aa2b139633aef6401',
        'orig_eval_dir': 'e28de4acf80bbbe9_y2030__n5_p0_25_e1__n7_p1_e3_m2', 'orig_label': 'y2030__n5_p0.25_e1__n7_p1_e3_m2',
        'N_old': 161, 'k0': 152, 'flex_price_multiplier': 2.0,
        'per_cycle_record_sha256': '10c14cbf11a4facfd31d09cd54abc94650f3dd74e2631925362ae65a7d3249d3'},
    'f2_incumbent': {
        'group': 'f2', 'gated': True, 'orig_root': _S51, 'orig_spec': 'campaign_spec_s51_f2_phase_b_5ce295e1.json',
        'orig_campaign_id': 's51_f2_phase_b',
        'orig_eval_key': '5ca4f86c3406c0424d3b63b8f39df4568f1be77f7f2d72642491539c0eb42663',
        'orig_eval_dir': '5ca4f86c3406c042_y2030__n5_p0_25_e0_5__n7_p1_e3_5_m2',
        'orig_label': 'y2030__n5_p0.25_e0.5__n7_p1_e3.5_m2', 'N_old': 181, 'k0': 172, 'flex_price_multiplier': 2.0,
        'per_cycle_record_sha256': '749949ae7b4d83bf753acf52a092d734d2037bf8daf5a2bfe457d8baee9f3176'},
    'pb_y2030_n9': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': 'a30a9faffdd74f9ebef7de4e72c12e9f4b0ca7e4cd63c425faf4c6c37278dd67',
        'orig_eval_dir': 'a30a9faffdd74f9e_y2030__n9_p0_25_e0_5', 'orig_label': 'y2030__n9_p0.25_e0.5',
        'N_old': 129, 'k0': 120, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': 'b7889025d1ad4ec2c010f5fc2327a9963060fe21bace8aeab56e0c96aa6566d2'},
    'pb_y2030_n7': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': 'd7030f598c2aeb4dbf00af404568ba801c0c02bd1c26c4b70e97f43eb30f0f66',
        'orig_eval_dir': 'd7030f598c2aeb4d_y2030__n7_p0_25_e0_5', 'orig_label': 'y2030__n7_p0.25_e0.5',
        'N_old': 134, 'k0': 125, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': 'ab250b6557d9548c04e4c0f24ea827ee026b4fd9593cc70fcc3f717881f97eb4'},
    'pb_y2025_n5': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': '4a8527254a85b9cca48e1eb872b35ed9f3d5209f84002e2c255516b8af5b304a',
        'orig_eval_dir': '4a8527254a85b9cc_y2025__n5_p0_25_e0_5', 'orig_label': 'y2025__n5_p0.25_e0.5',
        'N_old': 119, 'k0': 110, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': 'c84067449b6c5908b8c1187936d0516ba953c89bacc0255fb746c04437e2d9e7'},
    'pb_y2030_n5': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': 'd0c1f1605a44d5a86554dcf5ee169517122667cd2c93c2bdff81cbc4cf395b51',
        'orig_eval_dir': 'd0c1f1605a44d5a8_y2030__n5_p0_25_e0_5', 'orig_label': 'y2030__n5_p0.25_e0.5',
        'N_old': 131, 'k0': 122, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': '7d0dddd5eb23a0eda78572e37cdf701018c7e942a9bc02056596a335d315f5ca'},
    'pb_y2025_n9': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': '1bff3ed2fb98302a10e5eadaff8cdd1fda48d9178564b1ea0ee62824450aa163',
        'orig_eval_dir': '1bff3ed2fb98302a_y2025__n9_p0_25_e0_5', 'orig_label': 'y2025__n9_p0.25_e0.5',
        'N_old': 122, 'k0': 113, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': '129859c10482980cc48ca9826dc342e517e59da56d219409027fd9225bc99b09'},
    'pb_y2025_n7': {
        'group': 'phase_b', 'gated': True, 'orig_root': _S47, 'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json',
        'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': '10c73abd8511010de3962ea8be10693d51c411b7fca0a0dfb92d2730c6b9ec89',
        'orig_eval_dir': '10c73abd8511010d_y2025__n7_p0_25_e0_5', 'orig_label': 'y2025__n7_p0.25_e0.5',
        'N_old': 121, 'k0': 112, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': 'de4bc52b23884eaf71a5ccec616a4ffb9f4a3332dde57661a0ea386c8878ef3d'},
    'yl_y2030': {
        'group': 'year_ladder', 'gated': False, 'orig_root': _S45, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b',
        'orig_eval_key': '549476cd276a0350da9e5e02d92ce1548a2330c3f042dcce130c95da6af27cdb',
        'orig_eval_dir': '549476cd276a0350_n7_4h_e1_y2030', 'orig_label': 'n7_4h_e1_y2030',
        'N_old': 119, 'k0': 110, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': '9a08a3a21906c441072f16ad95e97a4a5c7b69ebf2e8cfafd0d7ab34c612e1e1'},
    'yl_y2035': {
        'group': 'year_ladder', 'gated': False, 'orig_root': _S45, 'orig_spec': 'campaign_spec_s45_a1b_bd3040a9.json',
        'orig_campaign_id': 's45_a1b',
        'orig_eval_key': 'dab6a8a2df221ba9b48daa43b30ca0088be9f9beb2e5a7709ffc9662c8b8e332',
        'orig_eval_dir': 'dab6a8a2df221ba9_n7_4h_e1_y2035', 'orig_label': 'n7_4h_e1_y2035',
        'N_old': 119, 'k0': 110, 'flex_price_multiplier': None,
        'per_cycle_record_sha256': 'cf72939e940021c82169194ba25f7a11960c6423cc7b166f7859106a7225545d'},
}
# Launch order (Addendum 57 Decision 3(c); Planner task W118): F2 challenger -> F2 incumbent -> Phase B x 6 -> 2030 ->
# 2035.
CELL_ORDER = ('f2_challenger', 'f2_incumbent', 'pb_y2030_n9', 'pb_y2030_n7', 'pb_y2025_n5', 'pb_y2030_n5',
              'pb_y2025_n9', 'pb_y2025_n7', 'yl_y2030', 'yl_y2035')
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])


def original_eval_dir(cell):
    return os.path.join(CELLS[cell]['orig_root'], 'evals', CELLS[cell]['orig_eval_dir'])


def reference_path(cell):
    return os.path.join(original_eval_dir(cell), 'per_cycle_record.jsonl')


def settling_rule_declaration():
    return {'module': 'settling_criterion_v2', 'class': 'settling_criterion_v2.SettlingRuleV2', 'version': SC2.VERSION,
            'tau': SC2.TAU, 'eps0': SC2.EPS0, 'k_excl': SC2.K_EXCL, 'w_min': SC2.W_MIN, 'w_factor': SC2.W_FACTOR,
            'p_max': P_MAX, 'l_mono': L_MONO, 'gap_bound': SC2.GAP_BOUND, 'drift_window': SC2.DRIFT_WINDOW}


def cap_rule(cell):
    c = CELLS[cell]
    if c['gated']:
        return {'kind': 'fixed', 'cap': c['N_old'] + CAP_AFTER_N_OLD, 'formula': 'N_old + 100'}
    return {'kind': 'dynamic', 'after_first_k0': CAP_AFTER_K0, 'ceiling': CAP_CEILING,
            'formula': 'min(k0_run + 109, 300)'}


def spec_cap(cell):
    """The harness cap (production's num_max_iters) for the cell: gated N_old + 100; ungated the ceiling 300."""
    r = cap_rule(cell)
    return r['cap'] if r['kind'] == 'fixed' else r['ceiling']


def declaration_for(cell):
    """The one valid declaration of a cell."""
    c = CELLS[cell]
    gated = c['gated']
    return {
        'label': LABEL, 'cell': cell, 'group': c['group'],
        'gate': 'bitwise_through_first_residual_pass' if gated else 'none_first_c2_evaluation',
        'first_residual_pass_expected': c['k0'] if gated else None,
        'N_old': c['N_old'] if gated else None,
        'holds_after': 'the run first residual pass k0_run: AA off, tight tail on, rho frozen for every later cycle',
        'cap_rule': cap_rule(cell),
        'settling_rule': settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True},
        'replay_reference': ({'per_cycle_record': reference_path(cell), 'sha256': c['per_cycle_record_sha256'],
                              'n_cycles': c['N_old'], 'gated_through_cycle': c['k0'],
                              'original_eval_key': c['orig_eval_key']} if gated else None),
        'abort_on_replay_divergence': bool(gated),
    }


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop` key
    is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} must NOT carry {forbidden} (the settling rule is the only stop before the cap)')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME}.cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} differs from the W118 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """{cycle: row} of the gated cell's original per_cycle_record.jsonl (cycles 1..N_old); refuses unless it hashes to
    the declaration, holds exactly 1..N_old, carries every gated field, and its first residual pass is the declared k0
    with Boyd passing on every cycle k0..N_old. None for an ungated cell."""
    ref = decl.get('replay_reference')
    if ref is None:
        return None
    path = os.path.join(REPO, ref['per_cycle_record'])
    got = C101._sha256(path)
    if got != ref['sha256']:
        raise RuntimeError(f'replay reference {ref["per_cycle_record"]} sha256 {got} != declared {ref["sha256"]}')
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[int(r['cycle'])] = r
    if sorted(out) != list(range(1, ref['n_cycles'] + 1)):
        raise RuntimeError(f'replay reference cycles != 1..{ref["n_cycles"]}')
    missing = [f for f in REPLAY_GATED_FIELDS + REPLAY_POST_RUN_ONLY_FIELDS if f not in out[1]]
    if missing:
        raise RuntimeError(f'replay reference rows lack gated fields {missing}')
    passes = [c for c in sorted(out) if out[c]['boyd_all_pass'] and out[c]['local_solves_ok']]
    k0 = ref['gated_through_cycle']
    if not passes or passes[0] != k0 or passes != list(range(k0, ref['n_cycles'] + 1)):
        raise RuntimeError(f'replay reference: first residual pass {passes[:1]} != declared k0 {k0}, or the passes '
                           f'are not contiguous to N_old')
    return out


def _certificate_length_writes_in_source():
    src = inspect.getsource(make_wrappers)
    return sorted(line.strip() for line in src.splitlines()
                  if 'minimum_consecutive_converged_cycles =' in line and not line.strip().startswith('#'))


# ======================================================================================================================
#  the priced interface-consensus gap t_sum (in-cycle; pure reads)
# ======================================================================================================================
T_SUM_SOURCE_FACTS = {
    # the terminal identity (interface_settlement_detail_s31c.json) prices with these two production functions
    'reporting_detail_price': ('_get_interface_reporting_detail',
                               ('price = _expected_market_price(local_tso_model, network, p)',
                                "p_int_tso_expected = pe.value(local_tso_model.expected_interface_pf_p[dn, p]) * s_base",
                                "p_int_dso_expected = pe.value(local_dso_model.expected_interface_pf_p[p]) * s_base")),
    'expected_price_definition': ('_expected_market_price',
                                  ('contribution = network.prob_market_scenarios[s_m] * network.cost_energy_p[s_m][p]',)),
    'block_weight_definition': ('_get_admm_block_weight',
                                ('annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - '
                                 'int(years[0])))',)),
    'pf_consensus_layout': ('create_admm_variables',
                            ("consensus_variables['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * "
                             "num_instants, 'q': [0.0] * num_instants}",
                             "consensus_variables['pf']['dso']['current'][node_id][year][day] = {'p': [0.0] * "
                             "num_instants, 'q': [0.0] * num_instants}")),
}
T_SUM_GATES_FACTS = ('block_weight = srp._get_admm_block_weight(transmission_network, year, day)',
                     "x_dso_pf = consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p]",
                     "z_tso_pf = consensus_vars['pf']['tso']['current'][node_id][year][day][power_type][p]")


def t_sum_capture_checklist():
    """The source facts the in-cycle t_sum relies on. Returns {name: bool}; raises nothing."""
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    out = {}
    for name, (fn, snippets) in T_SUM_SOURCE_FACTS.items():
        f = getattr(srp, fn, None)
        src = inspect.getsource(f) if callable(f) else ''
        out[f'source:{name}'] = bool(src) and all(s in src for s in snippets)
    out['signature:_expected_market_price'] = (list(inspect.signature(srp._expected_market_price).parameters)
                                               == ['model', 'network', 'p'])
    out['signature:_get_admm_block_weight'] = (list(inspect.signature(srp._get_admm_block_weight).parameters)
                                               == ['network_data', 'year', 'day'])
    gsrc = inspect.getsource(G)
    out['gates:terminal_identity_and_pf_stride_use_the_same_copies_and_weight'] = all(s in gsrc for s in T_SUM_GATES_FACTS)
    return out


def interface_price_weight(planning_problem, tso_model, srp_module):
    """{(node, year, day, period): (pi, w)} with production's own functions (pure reads of the network data and the TSO
    block's scenario set)."""
    tn = planning_problem.transmission_network
    out = {}
    for n in planning_problem.active_distribution_network_nodes:
        for y in planning_problem.years:
            for d in planning_problem.days:
                network = tn.network[y][d]
                w = srp_module._get_admm_block_weight(tn, y, d)
                for p in range(planning_problem.num_instants):
                    out[(n, y, d, p)] = (srp_module._expected_market_price(tso_model[y][d], network, p), w)
    return out


def t_sum_from_consensus(planning_problem, consensus_vars, price_weight):
    """t_sum = sum over (node, year, day, period) of w * pi * (p_DSO - p_TSO), the active-power pf consensus copies
    (MW), in the order node, year, day, period (W112's order). Returns (t_sum, {node: t}, sum |p_DSO - p_TSO| MW)."""
    pf = consensus_vars['pf']
    total, abs_gap = 0.0, 0.0
    by_node = {}
    for n in planning_problem.active_distribution_network_nodes:
        acc = 0.0
        for y in planning_problem.years:
            for d in planning_problem.days:
                xd = pf['dso']['current'][n][y][d]['p']
                zt = pf['tso']['current'][n][y][d]['p']
                for p in range(planning_problem.num_instants):
                    pi, w = price_weight[(n, y, d, p)]
                    term = w * pi * (xd[p] - zt[p])
                    total += term
                    acc += term
                    abs_gap += abs(xd[p] - zt[p])
        by_node[str(n)] = acc
    return total, by_node, abs_gap


def price_weight_digest(price_weight):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(price_weight, key=lambda k: tuple(str(x) for x in k)):
        pi, w = price_weight[key]
        h.update(f'{key[0]}|{key[1]}|{key[2]}|{key[3]}|{float(pi).hex()}|{float(w).hex()}\n'.encode())
    return h.hexdigest()


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve)
# ======================================================================================================================
def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child); fail fast: W101's items (a) Boyd key + AA on, (b) source order,
    (c) certificate length written only by disable / restore / rule end, no early stop, (d) the spec cap equals the
    cell's (gated N_old + 100; ungated 300), (e) the gated cell's original record hashes, holds 1..N_old and passes first
    at k0, (f) the rule constants equal settling_criterion_v2's with P_MAX 30 / L 60 / gap tau/2, (h) the lambda_t
    capture path; the signatures of the eight wrapped functions; the AA 'off' literal; the tail enabled; W105's capture
    paths; the t_sum source facts. Raises on any failure; returns the checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    decl = validate_settling_resettle(decl)
    cell = decl['cell']
    cap = int(spec['cap'])
    params = {
        '_capture_convergence_depth_tail_baseline': ['planning_problem', 'admm_parameters'],
        '_apply_convergence_depth_tail': ['planning_problem', 'admm_parameters', 'active', 'baseline', 'cycle'],
        'get_admm_boyd_residual_metrics': ['planning_problem', 'tso_model', 'dso_models', 'esso_model',
                                           'consensus_vars', 'dual_vars', 'admm_parameters'],
        '_anderson_acceleration_cycle_step': ['aa_state', 'aa_layout', 'consensus_vars', 'dual_vars', 'w_before',
                                              'rho_channel', 'boyd_metrics', 'iter'],
        '_convergence_depth_tail_next_state': ['cycle_convergence', 'aa_enabled', 'aa_record'],
        '_update_admm_penalties': ['tso_model', 'dso_models', 'esso_model', 'residual_metrics', 'boyd_metrics',
                                   'params', 'iter', 'allow_update', 'freeze_state'],
        '_get_operational_recourse_components': ['planning_problem', 'models'],
        '_get_admm_efc_per_day_max': ['esso_model'],
    }
    loop_src = inspect.getsource(srp._run_operational_planning)
    step_src = inspect.getsource(aam.AndersonAccelerationState.step)
    boyd_src = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    pos = [loop_src.find(s) for s in C101.SOURCE_ORDER]
    rule = decl['settling_rule']
    checks = {
        'a_boyd_all_pass_key_in_production': "'all_boyd_pass'" in boyd_src,
        'a_aa_step_called_with_boyd_metrics': ('aa_record = _anderson_acceleration_cycle_step(\n'
                                               '                    aa_state, aa_layout, consensus_vars, dual_vars,\n'
                                               '                    aa_w_before, aa_rho_before, boyd_metrics, iter,')
        in loop_src,
        'a_anderson_acceleration_on': bool(aa_on),
        'b_source_order_aa_step_lt_recourse_lt_convergence_test': all(p >= 0 for p in pos) and pos[0] < pos[1] < pos[2],
        'c_no_early_stop_in_declaration': not any(k in decl for k in FORBIDDEN_DECLARATION_KEYS),
        'c_certificate_length_written_only_by_disable_restore_rule_end': (
            _certificate_length_writes_in_source() == sorted(CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(C101.CERTIFICATE_LENGTH_READS)),
        'd_spec_cap_equals_the_cell_cap': cap == spec_cap(cell) and cap <= CAP_CEILING,
        'f_rule_constants_equal_settling_criterion_v2': (rule == settling_rule_declaration()
                                                         and rule['tau'] == SC2.TAU and rule['eps0'] == SC2.TAU / 100.0
                                                         and rule['gap_bound'] == SC2.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60),
        'f_rule_class_callable': callable(getattr(SC2, 'SettlingRuleV2', None)),
        'tail_enabled_for_this_run': bool((tail_checklist or {}).get('tail_enabled_for_this_run')),
        'aa_off_literal_is_production': (srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == AA_OFF_ACTION
                                         and repr(AA_OFF_ACTION)[1:-1] in step_src),
        'loop_exit_is_the_convergence_break': 'if convergence:\n            print(f"[INFO] \\t - ADMM converged' in loop_src,
        'block_functions_callable': all(callable(getattr(srp, nm, None)) for nm in (
            '_get_operational_recourse_block_components', '_get_operational_objective_component_blocks')),
        'efc_read_once_per_cycle_after_penalties_before_the_row': (
            loop_src.count('_get_admm_efc_per_day_max(esso_model)') == 1
            and 0 <= loop_src.find('= _update_admm_penalties(') < loop_src.find('_get_admm_efc_per_day_max(esso_model)')
            < loop_src.find('admm_diagnostics.append({')),
    }
    for name, expected in params.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    if decl['replay_reference'] is not None:
        try:
            load_replay_reference(decl)
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0'] = True
        except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in E105.capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    for k, v in t_sum_capture_checklist().items():
        checks[f't_sum:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W118 settling-resettle preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  state and wrappers
# ======================================================================================================================
class ResettleState:
    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        self.decl = decl
        self.cell = decl['cell']
        self.gated = decl['replay_reference'] is not None
        self.k0_expected = decl['first_residual_pass_expected']
        self.n_old = decl['N_old']
        self.cap = cap                                  # the harness cap (production's num_max_iters)
        self.eval_dir = eval_dir
        self.reference = reference or {}
        self.sink = sink
        cr = decl['cap_rule']
        p_max = decl['settling_rule']['p_max']
        if cr['kind'] == 'fixed':
            self.rule = SC2.SettlingRuleV2(p_max, cap=cr['cap'], cap_ceiling=CAP_CEILING)
        else:
            self.rule = SC2.SettlingRuleV2(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])
        self.first_pass = None
        self.phase = 'before'
        self.cycle = None
        self.params = None
        self.threshold_original = None
        self.threshold_restored = None
        self.gross = {}
        self.t_sum = {}
        self.price_weight = None
        self.price_weight_sha256 = None
        self.prev_blocks = None
        self.prev_blocks_cycle = None
        self.prev_q = None
        self.prev_q_cycle = None
        self.prev_ess = None
        self.prev_ess_cycle = None
        self.ess_blocks = None
        self.consecutive = 0
        self.cur = None
        self.lines = 0
        self.creep_lines = 0
        self.ess_lines = 0
        self.replay_equal_through = 0
        self.first_divergence = None
        self.overlap = []
        self.decision = None
        self.decision_written = False
        self.ended_by = None
        self.errors = []
        self.capture_errors = []
        self.events = []
        self.t0 = time.time()

    def held(self, c):
        return self.first_pass is not None and c > self.first_pass

    def _write(self, fname, obj):
        if self.sink is not None:
            self.sink.append((fname, obj))
            return
        with open(os.path.join(self.eval_dir, fname), 'a') as handle:
            handle.write(GRIO.dumps(obj, default=GRIO.json_default) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _write_decision(self, obj):
        if self.sink is not None:
            self.sink.append((DECISION_FILE, obj))
            self.decision_written = True
            return
        with open(os.path.join(self.eval_dir, DECISION_FILE), 'x') as handle:
            GRIO.dump(obj, handle, default=GRIO.json_default, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        self.decision_written = True

    def stopped_by(self):
        if self.first_divergence is not None:
            return 'replay_divergence_abort'
        dec = self.decision or {}
        if dec.get('status') == 'certified' and self.cycle == dec.get('k_star'):
            return 'settling_rule'
        if dec.get('status') == 'uncertified' and self.cycle == dec.get('k_cap'):
            return 'rule_cap' if self.ended_by == 'rule_cap' else 'cap'
        return 'other'

    def summary(self):
        dec = self.decision or {}
        stopped = self.stopped_by()
        gate_ok = (not self.gated) or (self.replay_equal_through == self.k0_expected
                                        and self.first_pass == self.k0_expected)
        return {
            'schema': SCHEMA, 'declaration': self.decl, 'cell': self.cell, 'gated': self.gated,
            'first_residual_pass_expected': self.k0_expected, 'first_residual_pass_run': self.first_pass,
            'N_old': self.n_old, 'spec_cap': self.cap, 'rule_cap': self.rule.cap, 'phase': self.phase,
            'certificate_length_original': self.threshold_original,
            'certificate_length_raised_to': CERTIFICATION_DISABLED_THRESHOLD,
            'certificate_length_restored_at_exit': self.threshold_restored,
            'last_cycle': self.cycle, 'cycle_lines_written': self.lines, 'creep_lines_written': self.creep_lines,
            'ess_schedule_lines_written': self.ess_lines,
            'replay_bitwise_through_cycle': self.replay_equal_through, 'replay_first_divergence': self.first_divergence,
            'overlap_k0_plus_1_to_N_old': list(self.overlap),
            'settling_status': dec.get('status'), 'k_star': dec.get('k_star'), 'branch': dec.get('branch'),
            'stopped_by': stopped, 'lapse_events': list(self.rule.lapses), 'gap_refusals': list(self.rule.gap_refusals),
            'price_weight_sha256': self.price_weight_sha256,
            'errors': list(self.errors), 'capture_errors': list(self.capture_errors[:50]),
            'n_capture_errors': len(self.capture_errors),
            'ok': (not self.errors and not self.capture_errors and self.phase == 'ended'
                   and self.threshold_restored == self.threshold_original and self.lines == (self.cycle or 0)
                   and self.creep_lines == (self.cycle or 0) and self.first_divergence is None and gate_ok
                   and self.decision is not None and self.decision_written
                   and stopped in ('settling_rule', 'rule_cap', 'cap')),
            'files': {'cycle_record': CYCLE_FILE, 'blocks_all': BLOCKS_FILE, 'creep': CREEP_FILE,
                      'ess_schedule': ESS_SCHEDULE_FILE, 'decision': DECISION_FILE},
        }


def _jtext(value):
    return GRIO.dumps(value, default=GRIO.json_default, sort_keys=True)


def make_wrappers(st, originals, srp_module=None):
    """The eight wrappers over `originals` (name -> callable). Separated from the context manager so the zero-solve
    checks can drive them with stand-in originals and recorded values. `srp_module` supplies the block functions and
    the two pricing functions (production's module in a run)."""

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W118 settling-resettle hook: {msg}')

    def _end_run(reason):
        """The rule ends the run at the end of THIS cycle (production's own exit test)."""
        st.ended_by = reason
        st.params.minimum_consecutive_converged_cycles = SETTLING_END_THRESHOLD
        st.events.append({'event': f'run_ended_by_{reason}', 'cycle': st.cycle})

    def w_baseline(planning_problem, admm_parameters):
        out = originals['_capture_convergence_depth_tail_baseline'](planning_problem, admm_parameters)
        if st.params is not None:
            raise_('tail baseline captured twice (a second ADMM call inside one re-settling run)')
        st.params = admm_parameters
        st.threshold_original = admm_parameters.minimum_consecutive_converged_cycles
        admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD
        st.events.append({'event': 'certificate_disabled', 'from': st.threshold_original,
                          'to': CERTIFICATION_DISABLED_THRESHOLD})
        return out

    def _phase(c):
        if st.gated:
            if c <= st.k0_expected:
                return 'gated'
            if c <= st.n_old:
                return 'overlap'
            return 'continuation'
        return 'held' if st.held(c) else 'pre_hold'

    def w_apply(planning_problem, admm_parameters, active, baseline, cycle):
        if cycle is None:   # the loop's exit restore
            out = originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
            if st.phase == 'in_cycle':
                if st.cur is not None and not st.cur.get('finalized'):
                    st.errors.append(f'cycle {st.cycle} was never finalised before the loop exit')
                if st.params is not None:
                    st.params.minimum_consecutive_converged_cycles = st.threshold_original
                    st.threshold_restored = st.params.minimum_consecutive_converged_cycles
                st.phase = 'ended'
                st.events.append({'event': 'loop_exit', 'last_cycle': st.cycle, 'restored_to': st.threshold_restored})
            return out
        if st.phase == 'ended':
            raise_(f'a cycle ({cycle}) after the loop exit')
        expected = (st.cycle or 0) + 1
        if cycle != expected:
            raise_(f'cycle tracker: got cycle {cycle!r}, expected {expected}')
        if st.cur is not None and not st.cur.get('finalized'):
            raise_(f'cycle {st.cycle} was never finalised (no EFC read after its penalty update)')
        st.cycle = cycle
        st.phase = 'in_cycle'
        st.cur = {'cycle': cycle, 'phase': _phase(cycle), 'regime': 'hold' if st.held(cycle) else 'natural',
                  't_start_s': time.time() - st.t0}
        if not st.held(cycle):
            st.cur['tail_apply'] = {'hold': False, 'active_passed': bool(active)}
            return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
        st.cur['tail_apply'] = {'hold': True, 'natural_active': bool(active), 'active_passed': True,
                                'hold_changed_value': not bool(active)}
        return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, True, baseline, cycle)

    def w_boyd(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):
        """Production's Boyd metrics, returned UNCHANGED (the same object); captures (b), (c) and t_sum are pure reads."""
        result = originals['get_admm_boyd_residual_metrics'](planning_problem, tso_model, dso_models, esso_model,
                                                             consensus_vars, dual_vars, admm_parameters)
        if st.phase != 'in_cycle' or st.cur is None or st.cur.get('finalized') or '_boyd_seen' in st.cur:
            return result
        st.cur['_boyd_seen'] = True
        t = time.time()
        cap = {'cycle': st.cycle}
        try:
            cap['boyd'] = E105.boyd_capture(result)
        except Exception as error:  # noqa: BLE001 -- write-only capture: recorded, never raised
            cap['boyd'] = None
            st.capture_errors.append({'cycle': st.cycle, 'capture': 'boyd', 'error': f'{type(error).__name__}: {error}'})
        try:
            if st.ess_blocks is None:
                st.ess_blocks = E105.ess_block_order(planning_problem)
            sched, errs = E105.read_ess_schedules(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                                                  st.ess_blocks)
            for e in errs[:5]:
                st.capture_errors.append({'cycle': st.cycle, 'capture': 'ess', 'error': e})
            prev = st.prev_ess if st.prev_ess_cycle == st.cycle - 1 else None
            nodes = list(planning_problem.active_distribution_network_nodes)
            cap['ess_movement'] = E105.ess_movement(sched, prev, st.ess_blocks, nodes)
            cap['ess_schedules'] = sched
            st.prev_ess, st.prev_ess_cycle = sched, st.cycle
        except Exception as error:  # noqa: BLE001
            cap['ess_movement'] = None
            cap['ess_schedules'] = None
            st.capture_errors.append({'cycle': st.cycle, 'capture': 'ess', 'error': f'{type(error).__name__}: {error}'})
        try:
            if st.price_weight is None:
                st.price_weight = interface_price_weight(planning_problem, tso_model, srp_module)
                st.price_weight_sha256 = price_weight_digest(st.price_weight)
            ts, by_node, abs_gap = t_sum_from_consensus(planning_problem, consensus_vars, st.price_weight)
            cap['t_sum'] = {'t_sum': ts, 't_by_node': by_node, 'sum_abs_gap_mw_p': abs_gap}
        except Exception as error:  # noqa: BLE001
            cap['t_sum'] = None
            st.capture_errors.append({'cycle': st.cycle, 'capture': 't_sum', 'error': f'{type(error).__name__}: {error}'})
        cap['capture_s'] = time.time() - t
        st.cur['_boyd_capture'] = cap
        return result

    def _cycle_t_sum():
        cap = (st.cur or {}).get('_boyd_capture') or {}
        return (cap.get('t_sum') or {}).get('t_sum')

    def w_aa(aa_state, aa_layout, consensus_vars, dual_vars, w_before, rho_channel, boyd_metrics, iter):
        if iter != st.cycle:
            raise_(f'AA step at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        st.cur['boyd_k'] = bool(boyd_metrics['all_boyd_pass'])
        st.cur['boyd_ratios_at_aa'] = {f'boyd_{g}_{kind}_ratio': boyd_metrics[g][f'{kind}_ratio']
                                       for g in CHANNELS for kind in ('primal', 'dual')}
        if not st.held(iter):
            rec = originals['_anderson_acceleration_cycle_step'](aa_state, aa_layout, consensus_vars, dual_vars,
                                                                 w_before, rho_channel, boyd_metrics, iter)
            st.cur['aa'] = {'hold': False, 'action': (rec or {}).get('action')}
            if st.cur['boyd_k'] and st.first_pass is None:
                st.first_pass = iter
                st.cur['first_residual_pass'] = True
                st.events.append({'event': 'first_residual_pass', 'cycle': iter,
                                  'expected': st.k0_expected, 'holds_from_cycle': iter + 1})
            return rec
        forced = dict(boyd_metrics)
        forced['all_boyd_pass'] = True
        rec = originals['_anderson_acceleration_cycle_step'](aa_state, aa_layout, consensus_vars, dual_vars,
                                                             w_before, rho_channel, forced, iter)
        st.cur['aa'] = {'hold': True, 'natural_all_boyd_pass': bool(boyd_metrics.get('all_boyd_pass')),
                        'action': (rec or {}).get('action'),
                        'hold_changed_value': not bool(boyd_metrics.get('all_boyd_pass'))}
        if (rec or {}).get('action') != AA_OFF_ACTION:
            raise_(f"AA hold at cycle {iter}: production's step returned {(rec or {}).get('action')!r}, not off")
        return rec

    def w_next(cycle_convergence, aa_enabled, aa_record):
        c = st.cycle
        if aa_record is not None and aa_record.get('cycle') is not None and aa_record.get('cycle') != c:
            raise_(f"tail next-state: AA record cycle {aa_record.get('cycle')!r} != tracker {c!r}")
        if c is None or not st.held(c):
            out = originals['_convergence_depth_tail_next_state'](cycle_convergence, aa_enabled, aa_record)
            if st.cur is not None:
                st.cur['tail_next'] = {'hold': False, 'value': bool(out), 'cycle_convergence': bool(cycle_convergence)}
            return out
        st.cur['tail_next'] = {'hold': True, 'natural_cycle_convergence': bool(cycle_convergence),
                               'aa_action': (aa_record or {}).get('action'), 'returned': True,
                               'hold_changed_value': not bool(cycle_convergence)}
        return True

    def w_efc(esso_model):
        out = originals['_get_admm_efc_per_day_max'](esso_model)
        if st.phase != 'in_cycle' or st.cur is None or st.cur.get('finalized'):
            return out     # outside the loop (terminal captures)
        if 'pending_penalties' not in st.cur:
            raise_(f'cycle {st.cycle}: EFC read before the penalty update (production order changed)')
        st.cur['efc_per_day_max'] = out
        boyd_metrics, local_solves_ok, result = st.cur.pop('pending_penalties')
        _finalize_cycle(boyd_metrics, local_solves_ok, result)
        return out

    def _observe(q, boyd, t_sum):
        """The stop rule v2 for this cycle; the only place the settling decision is taken."""
        c = st.cycle
        rec = st.rule.observe(c, q, boyd, t_sum)
        st.cur['settling'] = rec
        dec = st.rule.decision
        if dec is not None and st.decision is None:
            st.decision = dict(dec)
            st.decision.update({'cell': st.cell, 'gated': st.gated, 'N_old': st.n_old,
                                'first_residual_pass_run': st.first_pass, 'spec_cap': st.cap,
                                'decided_at_cycle': c, 'Q_N_old_recorded': (st.reference.get(st.n_old) or {}).get(
                                    'gross_operational_cost') if st.gated else None})
            if dec['status'] == 'certified':
                _end_run('settling_rule')
            elif c < st.cap:
                _end_run('rule_cap')
            else:
                st.ended_by = 'cap'
                st.events.append({'event': 'uncertified_at_cap', 'cycle': c})
            st._write_decision(st.decision)

    def w_recourse(planning_problem, models):
        rc = originals['_get_operational_recourse_components'](planning_problem, models)
        if st.phase != 'in_cycle' or st.cur is None or 'gross' in st.cur:
            return rc
        c = st.cycle
        gross = rc['gross_operational_cost']
        st.cur['gross'] = gross
        st.cur['gross_hex'] = float.hex(gross) if isinstance(gross, float) else None
        st.cur['net_operational_recourse'] = rc.get('net_operational_recourse')
        st.cur['terminal_salvage_value'] = rc.get('terminal_salvage_value')
        prev = st.gross.get(c - 1)
        st.cur['step'] = (gross - prev) if (prev is not None and gross is not None) else None
        st.gross[c] = gross
        if st.decl['record_all_blocks'] and srp_module is not None:
            _capture_blocks(planning_problem, models, c, rc)
        if 'boyd_k' not in st.cur:
            raise_(f'cycle {c}: recourse computed but the AA wrapper did not run this cycle (no boyd_k)')
        _observe(gross, st.cur['boyd_k'], _cycle_t_sum())
        return rc

    def _capture_blocks(planning_problem, models, c, rc):
        """W101's all-block line (the W105 format), then capture (a) from the SAME two block dicts."""
        t = time.time()
        blocks = srp_module._get_operational_recourse_block_components(planning_problem, models)
        obj = srp_module._get_operational_objective_component_blocks(planning_problem, models)
        net = rc.get('net_operational_recourse')
        total = sum(blocks.values())
        tol = max(1e-4, 1e-10 * max(abs(net), 1.0)) if net is not None else None
        line = {'cycle': c, 'phase': _phase(c),
                'gross_operational_cost': rc.get('gross_operational_cost'), 'net_operational_recourse': net,
                'terminal_salvage_value': rc.get('terminal_salvage_value'), 'n_blocks': len(blocks),
                'blocks': C101._block_rows(blocks), 'block_sum': total,
                'block_sum_minus_net': (total - net) if net is not None else None,
                'reconciles_to_net': bool(abs(total - net) <= tol) if tol is not None else None,
                'objective_component_blocks': [{'agent': a, 'node_id': nid, 'year': str(y), 'day': str(d), **comp}
                                               for (a, nid, y, d), comp in sorted(
                                                   obj.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]}
        if st.prev_blocks is not None and st.prev_blocks_cycle == c - 1:
            deltas = []
            for r in line['blocks']:
                key = (r['agent'], r['node_id'], r['year'], r['day'])
                previous = st.prev_blocks.get(key)
                deltas.append({'agent': r['agent'], 'node_id': r['node_id'], 'year': r['year'], 'day': r['day'],
                               'previous': previous, 'current': r['value'],
                               'delta': (r['value'] - previous) if previous is not None else None})
            line['deltas_vs_previous_cycle'] = deltas
        st.prev_blocks = {(r['agent'], r['node_id'], r['year'], r['day']): r['value'] for r in line['blocks']}
        st.prev_blocks_cycle = c
        line['capture_s'] = time.time() - t
        st.cur['blocks_captured'] = len(blocks)
        st._write(BLOCKS_FILE, line)
        t2 = time.time()
        try:
            prev = st.prev_q if st.prev_q_cycle == c - 1 else None
            qd = E105.q_decomposition(blocks, obj, rc.get('gross_operational_cost'), prev)
            if qd['reconciliation']['reconciles'] is not True:
                st.capture_errors.append({'cycle': c, 'capture': 'q_decomposition',
                                          'error': f"does not reconcile: {qd['reconciliation']}"})
            try:
                qd['esso_side_terms_not_in_gross_Q'] = E105.esso_side_terms(planning_problem, models['esso'])
            except Exception as error:  # noqa: BLE001
                qd['esso_side_terms_not_in_gross_Q'] = None
                st.capture_errors.append({'cycle': c, 'capture': 'esso_side_terms',
                                          'error': f'{type(error).__name__}: {error}'})
            st.prev_q, st.prev_q_cycle = qd, c
        except Exception as error:  # noqa: BLE001
            qd = None
            st.capture_errors.append({'cycle': c, 'capture': 'q_decomposition',
                                      'error': f'{type(error).__name__}: {error}'})
        st.cur['_q_decomposition'] = qd
        st.cur['_q_capture_s'] = time.time() - t2

    def w_penalties(tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params, iter=None,
                    allow_update=True, freeze_state=None):
        if iter != st.cycle:
            raise_(f'penalty update at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        hold = st.held(iter)
        result = originals['_update_admm_penalties'](tso_model, dso_models, esso_model, residual_metrics, boyd_metrics,
                                                     params, iter=iter,
                                                     allow_update=(False if hold else allow_update),
                                                     freeze_state=freeze_state)
        actions, before, after, bg, ag, rfa, fs = result
        rec = {'hold': hold, 'allow_update_natural': bool(allow_update),
               'allow_update_passed': False if hold else bool(allow_update),
               'rho_before': dict(before), 'rho_after': dict(after), 'gamma_before': dict(bg), 'gamma_after': dict(ag),
               'actions': dict(actions), 'rho_freeze_active': bool(rfa),
               'frozen': {g: bool((fs or {}).get(g, {}).get('frozen')) for g in CHANNELS}}
        st.cur['rho'] = rec
        if hold:
            changed = [g for g in CHANNELS if before[g] != after[g] or bg[g] != ag[g]]
            rec['changed_channels'] = changed
            if changed:
                raise_(f'rho hold at cycle {iter}: rho / gamma changed on {changed}')
        st.cur['pending_penalties'] = (boyd_metrics, allow_update, result)
        return result

    def _finalize_cycle(boyd_metrics, local_solves_ok, penalty_result):
        cur = st.cur
        c = cur['cycle']
        actions, _before, after, _bg, _ag, rfa, _fs = penalty_result
        t_sum = _cycle_t_sum()
        if 'gross' not in cur:          # a failed cycle: no Q, no AA step -> the rule sees (None, False)
            cur['gross'] = None
            cur['gross_hex'] = None
            cur['step'] = None
            cur['boyd_k'] = False
            _observe(None, False, t_sum)
        boyd_all_pass = boyd_metrics['all_boyd_pass']
        cycle_convergence = boyd_all_pass and local_solves_ok
        st.consecutive = (st.consecutive + 1) if cycle_convergence else 0
        run_values = {'cycle': c, 'local_solves_ok': local_solves_ok, 'recourse': cur.get('net_operational_recourse'),
                      'gross_operational_cost': cur.get('gross'),
                      'terminal_salvage_value': cur.get('terminal_salvage_value'),
                      'cycle_convergence': cycle_convergence, 'consecutive_converged_cycles': st.consecutive,
                      'boyd_all_pass': boyd_all_pass, 'boyd_stop': cycle_convergence,
                      'rho_v_after': after['v'], 'rho_pf_after': after['pf'], 'rho_ess_after': after['ess'],
                      'rho_v_action': actions['v'], 'rho_pf_action': actions['pf'], 'rho_ess_action': actions['ess'],
                      'rho_freeze_active': rfa, 'efc_per_day_max': cur.get('efc_per_day_max')}
        for g in CHANNELS:
            run_values[f'boyd_{g}_primal_ratio'] = boyd_metrics[g]['primal_ratio']
            run_values[f'boyd_{g}_dual_ratio'] = boyd_metrics[g]['dual_ratio']
            run_values[f'boyd_{g}_channel_pass'] = boyd_metrics[g]['channel_pass']
        cur['local_solves_ok'] = bool(local_solves_ok)
        cur['boyd_ratios'] = {f: run_values[f] for f in BOYD_RATIO_FIELDS}
        cur['boyd_pf_primal_ratio'] = run_values['boyd_pf_primal_ratio']
        cur['consecutive_converged_cycles_tracked'] = st.consecutive
        cur['holds'] = {'aa': (cur.get('aa') or {}).get('hold'), 'tail_apply': (cur.get('tail_apply') or {}).get('hold'),
                        'tail_next': (cur.get('tail_next') or {}).get('hold'), 'rho': (cur.get('rho') or {}).get('hold')}
        cur['certificate_length_in_force_at_cycle_end'] = (st.params.minimum_consecutive_converged_cycles
                                                          if st.params is not None else None)
        cur['first_residual_pass_run'] = st.first_pass
        boyd_cap = cur.pop('_boyd_capture', None)
        qd = cur.pop('_q_decomposition', None)
        q_s = cur.pop('_q_capture_s', None)
        cur.pop('_boyd_seen', None)
        if boyd_cap is None:
            st.capture_errors.append({'cycle': c, 'capture': 'boyd', 'error': 'no Boyd call observed this cycle'})
            boyd_cap = {}
        ts = boyd_cap.get('t_sum') or {}
        cur['t_sum'] = ts.get('t_sum')
        cur['t_by_node'] = ts.get('t_by_node')
        cur['sum_abs_gap_mw_p'] = ts.get('sum_abs_gap_mw_p')
        cur['Q_cc'] = (cur['gross'] + cur['t_sum']) if (cur.get('gross') is not None
                                                         and cur.get('t_sum') is not None) else None
        if cur['t_sum'] is not None:
            st.t_sum[c] = cur['t_sum']
        # the replay gate (gated cells, cycles <= k0) and the overlap (k0 < c <= N_old, report-only)
        divergence = None
        ref_row = st.reference.get(c) if st.gated else None
        if ref_row is not None and c <= st.k0_expected:
            diff, detail = C101.replay_compare(run_values, ref_row)
            cur['replay_equal'] = not diff
            if diff:
                divergence = {'cycle': c, **detail}
            elif st.replay_equal_through == c - 1:
                st.replay_equal_through = c
            if divergence is None and c == st.k0_expected and st.first_pass != c:
                divergence = {'cycle': c, 'first_residual_pass_mismatch': {'run': st.first_pass,
                                                                           'expected': st.k0_expected},
                              'fields_differing': [], 'gross_difference_run_minus_recorded': None,
                              'max_relative_difference': None}
            if divergence is not None:
                st.first_divergence = divergence
                cur['replay_divergence'] = divergence
        else:
            cur['replay_equal'] = None
        if ref_row is not None and c > st.k0_expected:
            q_old, q_new = ref_row.get('gross_operational_cost'), cur.get('gross')
            ov = {'cycle': c, 'Q_old': q_old, 'Q_new': q_new,
                  'Q_new_minus_Q_old': (q_new - q_old) if (q_new is not None and q_old is not None) else None,
                  'relative': ((q_new - q_old) / q_old) if (q_new is not None and q_old) else None}
            cur['overlap'] = ov
            st.overlap.append(ov)
        cur['t_end_s'] = time.time() - st.t0
        cur['finalized'] = True
        st._write(CYCLE_FILE, cur)
        st.lines += 1
        sched = boyd_cap.pop('ess_schedules', None)
        if sched is not None:
            if st.ess_lines == 0:
                st._write(ESS_SCHEDULE_FILE, {'header': True, 'schema': SCHEMA + '/ess_schedule',
                                              'blocks': [[n, str(y), str(d)] for n, y, d in st.ess_blocks],
                                              'n_blocks': len(st.ess_blocks),
                                              'families': {'p': list(E105.ESS_SIDES), 'q': list(E105.ESS_SIDES),
                                                           'charge': list(E105.ESS_CHARGE_SIDES),
                                                           'discharge': list(E105.ESS_CHARGE_SIDES)},
                                              'units': 'MW (p, charge, discharge) / Mvar (q)',
                                              'read_at': 'production get_admm_boyd_residual_metrics call (after the '
                                                         "cycle's ESSO consensus / dual update, before AA)"})
            st._write(ESS_SCHEDULE_FILE, {'cycle': c, **sched})
            st.ess_lines += 1
        elif st.ess_lines == 0:
            st.capture_errors.append({'cycle': c, 'capture': 'ess', 'error': 'no schedule this cycle'})
        creep = {'cycle': c, 'phase': cur['phase'], 'regime': cur['regime'], 'gross': cur.get('gross'),
                 'step': cur.get('step'), 'local_solves_ok': bool(local_solves_ok), 'boyd_k': cur.get('boyd_k'),
                 't_sum': cur['t_sum'], 'Q_cc': cur['Q_cc'],
                 'q_decomposition': qd, 'ess_movement': boyd_cap.get('ess_movement'), 'boyd': boyd_cap.get('boyd'),
                 'capture_s': {'boyd_ess_t_sum': boyd_cap.get('capture_s'), 'q_decomposition': q_s}}
        st._write(CREEP_FILE, creep)
        st.creep_lines += 1
        s = cur.get('settling') or {}
        print(f"[W118-RESETTLE] {st.cell} cycle {c} ({cur['phase']}/{cur['regime']}) Q={cur.get('gross')!r} "
              f"step={cur.get('step')} t_sum={cur.get('t_sum')} boyd_k={cur.get('boyd_k')} "
              f"pf_primal={run_values['boyd_pf_primal_ratio']} replay_equal={cur.get('replay_equal')} "
              f"k0={s.get('k0')} len_T={s.get('len_T')} range={s.get('range')} decision={s.get('decision')} "
              f"reasons={s.get('reasons')} holds={cur['holds']}", flush=True)
        if divergence is not None and st.decl['abort_on_replay_divergence']:
            raise_(f"REPLAY DIVERGED at cycle {c} (fields {divergence.get('fields_differing')}, gross difference "
                   f"{divergence.get('gross_difference_run_minus_recorded')!r}, max relative "
                   f"{divergence.get('max_relative_difference')!r}"
                   f"{', first-pass mismatch ' + repr(divergence['first_residual_pass_mismatch']) if 'first_residual_pass_mismatch' in divergence else ''}"
                   f") -- cell ABORTED (W118: no relabelled continuation)")

    return {'_capture_convergence_depth_tail_baseline': w_baseline, '_apply_convergence_depth_tail': w_apply,
            'get_admm_boyd_residual_metrics': w_boyd,
            '_anderson_acceleration_cycle_step': w_aa, '_convergence_depth_tail_next_state': w_next,
            '_update_admm_penalties': w_penalties, '_get_operational_recourse_components': w_recourse,
            '_get_admm_efc_per_day_max': w_efc}


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the wrappers for the run (the harness enters this FIRST, so these wrap production directly and every
    harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleState(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = make_wrappers(st, originals, srp_module=srp)
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()
