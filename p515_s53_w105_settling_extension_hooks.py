"""
P5.15 Addendum 54 Ruling 1, Planner task W105 -- the C* SETTLING EXTENSION diagnostic hooks (frozen stage spec v40).

WHAT THIS MODULE IS. A child-side, harness-installed set of pass-through wrappers on EIGHT production functions of
`shared_resources_planning` (module globals resolved by `_run_operational_planning` at call time -- the W98 / W101
technique; this module is W101's `p515_s53_w101_settling_continuation_hooks` adapted, and reuses its stateless helpers).
No production module is edited. It is installed ONLY for a campaign-spec entry that declares `settling_extension`
(`p515_s44_campaign_harness`, W105); the declaration enters the entry's eval key, so the extension never shares a key,
an eval dir or a working dir with W104's run or with the recert it replayed.

THE RUN IT SERVES (Addendum 54 Ruling 1, H_ess-flat). The SRP1 corner plan C* (the recert cell c_star, eval key 96c5aa50;
W104's continuation eval key 4bf36c15, evidence 82dcebb7) re-run from scratch under the IDENTICAL configuration:
  * cycles 1..187 REPLAYED and gated BITWISE, cycle by cycle, against W104's committed records: the per_cycle_record
    fields production hands the wrappers (W101's REPLAY_GATED_FIELDS), the W104 cycle-line fields
    (`CYCLE_LINE_GATED_FIELDS`), and W104's settling record reproduced by a MIRROR of W104's rule
    (SettlingRule(N = 87, cap = 187, P_MAX = 22)); at 187 the mirror's decision against W104's settling_decision.json.
    The FIRST difference writes the cycle line with the cycle and magnitude and ABORTS the run (RuntimeError -> the
    child's barrier record); never continued as a relabelled run;
  * the certification rule DISABLED for the whole run (certificate length 10**9 before cycle 1, restored at exit);
    the loop runs to the fixed cap 287 (production's num_max_iters) -- nothing else ends it;
  * for every cycle > 87 the certifying regime HELD exactly as W104 held it: AA OFF (production's own 'off' branch,
    even when Boyd lapses), the tight tail ON, rho FROZEN;
  * the settling rule (SettlingRule(N = 87, cap = 287, P_MAX = 22), W104's N = 87 state) evaluated EVERY cycle,
    REPORT-ONLY and NON-LATCHING: a certification is recorded (the first one is `first_would_certify`) and the latch is
    cleared, so the rule's sign / turning-point state keeps evolving exactly as if it had not certified; it never
    touches the certificate length;
  * per-cycle captures, WRITE-ONLY (each a pure read of objects production already holds; a capture error is recorded,
    never raised):
      (a) Q by block and cost component (`CREEP_FILE`, 'q_decomposition'): for every TSO and DSO block the weighted
          components of `_get_local_objective_components` (generation, flexibility, load curtailment, RES curtailment,
          ESS usage, ESS complementarity, slack penalties) plus `other` = block value - classified total, so the
          components sum to the block's contribution to gross Q; per-agent aggregates (TSO, DSO_5, DSO_7, DSO_9, ESSO)
          and deltas vs the previous cycle. The ESSO row is identically 0: the ESSO does not enter
          `gross_operational_cost` (asserted from source); its salvage and own objective terms are reported beside.
      (b) ESS schedule movement (`CREEP_FILE`, 'ess_movement'; raw schedules in `ESS_SCHEDULE_FILE`): per ESS node,
          sum over years, days and periods of |x(k) - x(k-1)| for the consensus copies the ESS channel sees at
          production's Boyd call (esso = the ESSO's es_pnet, tso / dso = the network-side expected ESS injections,
          z = the consensus value), P and Q; and the charge / discharge schedules (ESSO: es_pch / es_pdch summed over
          cohorts; TSO / DSO: probability-weighted shared_es_pch / shared_es_pdch x s_base).
      (c) all six Boyd residuals (`CREEP_FILE`, 'boyd'): per channel v / pf / ess the raw primal r and dual s norms,
          eps_pri / eps_dual, the ratios and every other field production returns.

THE WRAPPERS (each calls production; for cycle <= 87 each passes its arguments through as the SAME objects and returns
production's value UNCHANGED; after 87 the three holds act exactly as W101's):
  1. `_capture_convergence_depth_tail_baseline` (once): certificate length -> 10**9.
  2. `_apply_convergence_depth_tail`: cycle tracker; tail hold (a) after 87; restore at loop exit.
  3. `get_admm_boyd_residual_metrics` (NEW vs W101): production's value returned unchanged (same object); captures
     (b) and (c) from its arguments and its return value.
  4. `_anderson_acceleration_cycle_step`: boyd_k; AA hold after 87.
  5. `_convergence_depth_tail_next_state`: tail hold (b) after 87.
  6. `_get_operational_recourse_components`: Q_k; W101's all-block line (identical); capture (a); the two rules.
  7. `_update_admm_penalties`: rho hold after 87.
  8. `_get_admm_efc_per_day_max`: finalises the cycle -- the replay gates (cycle <= 187), the cycle line, the creep line.

Zero solves: nothing here solves or builds a model. Stdlib only at import (the harness parent imports it for keys).
"""
import copy
import inspect
import json
import os
import time
from contextlib import contextmanager

import gate_result_io as GRIO
import settling_criterion as SC
import p515_s53_w101_settling_continuation_hooks as C101

SCHEMA = 'p515_s53_w105_settling_extension_v1'
OPTION_NAME = 'settling_extension'
LABEL = ('SETTLING EXTENSION DIAGNOSTIC RUN (W105, spec v40) -- C* replayed bitwise through cycle 187 against W104 '
         '(abort on divergence; W104 holds after cycle 87), then exactly 100 more cycles 188-287 under the same holds; '
         'the settling rule evaluated report-only (never stops the run); per-block dQ by component, sum |dp_ess| per '
         'node and side, and all six Boyd residuals recorded every cycle')
CELL = 'c_star_ext'
BASE_CELL = 'c_star'
N_HOLD = 87                 # W104's N: the holds engage after this cycle, exactly as in W104
REPLAY_THROUGH = 187        # W104's last cycle (its cap): cycles 1..187 are gated bitwise against W104
EXTENSION_CYCLES = 100      # 188..287, a fixed-length diagnostic
CAP = REPLAY_THROUGH + EXTENSION_CYCLES
P_MAX = 22                  # W104's settling-rule declaration (spec v39: settling_criterion.p_max_from_records)
RULE_MODE = 'report_only_non_latching'
CERTIFICATION_DISABLED_THRESHOLD = C101.CERTIFICATION_DISABLED_THRESHOLD
AA_OFF_ACTION = C101.AA_OFF_ACTION
CHANNELS = C101.CHANNELS
BOYD_RATIO_FIELDS = C101.BOYD_RATIO_FIELDS
REPLAY_GATED_FIELDS = C101.REPLAY_GATED_FIELDS
REPLAY_POST_RUN_ONLY_FIELDS = C101.REPLAY_POST_RUN_ONLY_FIELDS
# W104's cycle-line fields reproduced bitwise for cycles 1..187 (JSON text): everything W101 wrote except the timing,
# the phase label, replay_equal (None after 87 in W104) and the settling record (gated separately against the mirror).
CYCLE_LINE_GATED_FIELDS = ('local_solves_ok', 'gross', 'gross_hex', 'net_operational_recourse', 'terminal_salvage_value',
                           'step', 'boyd_k', 'boyd_ratios_at_aa', 'boyd_ratios', 'boyd_pf_primal_ratio', 'aa',
                           'tail_apply', 'tail_next', 'rho', 'holds', 'efc_per_day_max',
                           'consecutive_converged_cycles_tracked', 'certificate_length_in_force_at_cycle_end',
                           'blocks_captured')
# The W104 decision record carries these fields added by W101's hooks on top of the rule's own decision.
W104_DECISION_EXTRA = {'cell': BASE_CELL, 'N': N_HOLD, 'cap': REPLAY_THROUGH, 'decided_at_cycle': REPLAY_THROUGH}
WRAPPED = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail', 'get_admm_boyd_residual_metrics',
           '_anderson_acceleration_cycle_step', '_convergence_depth_tail_next_state', '_update_admm_penalties',
           '_get_operational_recourse_components', '_get_admm_efc_per_day_max')
CYCLE_FILE = 'settling_extension_cycle_record.jsonl'
BLOCKS_FILE = C101.BLOCKS_FILE                    # recourse_blocks_all.jsonl, the W101 line format unchanged
CREEP_FILE = 'creep_diagnostic_per_cycle.jsonl'
ESS_SCHEDULE_FILE = 'ess_schedule_per_cycle.jsonl'
DECISION_FILE = 'settling_extension_decision.json'
SUMMARY_KEY = 'settling_extension'
FORBIDDEN_DECLARATION_KEYS = ('early_stop',)
# The ONLY two assignments of the certificate length in this module (asserted from source): the disable at the tail
# baseline and the restore at the loop exit. The report-only rule never writes it.
CERTIFICATE_LENGTH_WRITES = (
    'admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD',
    'st.params.minimum_consecutive_converged_cycles = st.threshold_original',
)
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- W104's committed C* evidence (82dcebb7): the replay reference ----------------------------------------------------
W104_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                         'campaign_s53_w101_srp1_cont_c_star')
W104_EVAL_DIR = os.path.join(W104_ROOT, 'evals', '4bf36c151fd10613_c_star')
W104 = {
    'evidence_commit': '82dcebb7', 'campaign_id': 's53_w101_srp1_cont_c_star',
    'eval_key': '4bf36c151fd106139782939724c6d888585cdb78e0dbff6516d2cea36a42d5a1',
    'per_cycle_record': {'path': os.path.join(W104_EVAL_DIR, 'per_cycle_record.jsonl'),
                         'sha256': 'f65c383366ea3e4132f5ba3a3941074689240851ffc2fcf6ed3daaa034f6a890'},
    'cycle_record': {'path': os.path.join(W104_EVAL_DIR, 'settling_continuation_cycle_record.jsonl'),
                     'sha256': '2ea4552bcf24adad79843426e3ea10669b43ad6d29fd59a9b2f10ef90f498b63'},
    'decision': {'path': os.path.join(W104_EVAL_DIR, 'settling_decision.json'),
                 'sha256': 'a51965ffdbd857805356fbdbad8adfb7c92fac77d6fd0904e7d109e8790af60e'},
    'recourse_blocks_all': {'path': os.path.join(W104_EVAL_DIR, 'recourse_blocks_all.jsonl'),
                            'sha256': '328b1b20f90224e18d073e996d3622622944269d989e6954733d380dc334c789'},
    'g_s39_D': {'path': os.path.join(W104_EVAL_DIR, 'g_s39_D.json'),
                'sha256': '0f9ba6dfb85cc101551acdedca0516bb6777a7e9a2b27654eefd8c35ec425136'},
}
# Q decomposition: the components `_get_local_objective_components` returns that sum to its classified total, plus
# `other` (= the block's contribution to gross Q minus that classified total: the settlement deviation part, row 18 --
# both 0 at one scenario -- and any float rounding). ESS_TERMS is a reporting aggregate.
Q_COMPONENTS = ('generation_cost', 'flexibility_cost', 'load_curtailment_cost', 'res_curtailment_penalty',
                'ess_usage_penalty', 'ess_complementarity_penalties', 'slack_penalties')
Q_COMPONENTS_ALL = Q_COMPONENTS + ('other',)
ESS_TERMS = ('ess_usage_penalty', 'ess_complementarity_penalties')
ESS_SIDES = ('esso', 'tso', 'dso', 'z')
ESS_CHARGE_SIDES = ('esso', 'tso', 'dso')


def declaration():
    """The one valid declaration (validated below)."""
    return {
        'label': LABEL, 'cell': CELL, 'base_cell': BASE_CELL, 'hold_after_cycle': N_HOLD,
        'replay_through_cycle': REPLAY_THROUGH, 'extension_cycles': EXTENSION_CYCLES, 'cap': CAP,
        'settling_rule': C101.settling_rule_declaration(P_MAX), 'settling_rule_mode': RULE_MODE,
        'w104_mirror_rule': {'n': N_HOLD, 'cap': REPLAY_THROUGH, 'p_max': P_MAX},
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True},
        'replay_reference': {
            'evidence_commit': W104['evidence_commit'], 'eval_key': W104['eval_key'], 'n_cycles': REPLAY_THROUGH,
            'per_cycle_record': W104['per_cycle_record']['path'],
            'per_cycle_record_sha256': W104['per_cycle_record']['sha256'],
            'cycle_record': W104['cycle_record']['path'], 'cycle_record_sha256': W104['cycle_record']['sha256'],
            'decision': W104['decision']['path'], 'decision_sha256': W104['decision']['sha256']},
        'abort_on_replay_divergence': True,
    }


def validate_settling_extension(value):
    """None = not declared. Otherwise the value must equal `declaration()` EXACTLY (an `early_stop` key is refused by
    name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} must NOT carry {forbidden} (the extension is fixed-length; nothing stops it)')
    want = declaration()
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} differs from the W105 declaration on {bad}')
    return copy.deepcopy(want)


def _sha256(path):
    return C101._sha256(path)


def _read_jsonl_by_cycle(path):
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[int(r['cycle'])] = r
    return out


def load_replay_reference(decl):
    """(rows, lines, decision): W104's per_cycle_record rows and cycle lines {cycle: ...} and its decision record;
    refuses unless each hashes to the declaration and the rows / lines hold exactly cycles 1..187."""
    ref = decl['replay_reference']
    out = []
    for key in ('per_cycle_record', 'cycle_record', 'decision'):
        path = os.path.join(REPO, ref[key])
        got = _sha256(path)
        if got != ref[f'{key}_sha256']:
            raise RuntimeError(f'replay reference {ref[key]} sha256 {got} != declared {ref[key + "_sha256"]}')
        if key == 'decision':
            with open(path) as handle:
                out.append(json.load(handle))
        else:
            by = _read_jsonl_by_cycle(path)
            if sorted(by) != list(range(1, ref['n_cycles'] + 1)):
                raise RuntimeError(f'replay reference {ref[key]} cycles != 1..{ref["n_cycles"]}')
            out.append(by)
    rows, lines, dec = out
    missing = [f for f in REPLAY_GATED_FIELDS + REPLAY_POST_RUN_ONLY_FIELDS if f not in rows[1]]
    missing += [f'line.{f}' for f in CYCLE_LINE_GATED_FIELDS + ('settling',) if f not in lines[100]]
    if missing:
        raise RuntimeError(f'replay reference lacks gated fields {missing}')
    return rows, lines, dec


def _certificate_length_writes_in_source():
    src = inspect.getsource(make_wrappers)
    return sorted(line.strip() for line in src.splitlines()
                  if 'minimum_consecutive_converged_cycles =' in line and not line.strip().startswith('#'))


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve)
# ======================================================================================================================
SOURCE_FACTS = {
    # (a) Q = TSO + DSO primal values minus the contracted settlement and the voltage pin: the ESSO is not in gross Q
    'q_gross_is_tso_plus_dso': (
        '_get_operational_recourse_components',
        ("gross_operational_cost_including_settlement = planning_problem.transmission_network.get_primal_value("
         "models['tso'])",
         "gross_operational_cost_including_settlement += distribution_network.get_primal_value(models['dso'][node_id])",
         'gross_operational_cost = (gross_operational_cost_including_settlement',
         "net_operational_recourse = gross_operational_cost - terminal_salvage_value")),
    'q_blocks_tso_dso_salvage': (
        '_get_operational_recourse_block_components',
        ("blocks[('TSO', None, year, day)] = float(weight * local_value)",
         "blocks[('DSO', node_id, year, day)] = float(weight * local_value)",
         "blocks[('SALVAGE', None, None, None)] = -float(terminal_salvage_value)")),
    'q_components_classified_total': (
        '_get_local_objective_components',
        ("'generation_cost': float(pe.value(model.total_gen_cost))",
         "'flexibility_cost': float(pe.value(model.total_flex_cost))",
         "'load_curtailment_cost': float(pe.value(model.total_load_curt_cost))",
         "'res_curtailment_penalty': float(pe.value(model.total_gen_curt_penalty))",
         "'ess_usage_penalty': float(pe.value(model.total_ess_utilization_cost_penalty))",
         "'slack_penalties': float(pe.value(model.total_slack_penalties))",
         "'ess_complementarity_penalties': float(pe.value(model.total_ess_complementarity_penalties))")),
    # (b) the ESS consensus copies and where production writes them
    'ess_consensus_four_copies': (
        'create_admm_variables',
        ("consensus_variables['ess']['esso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, "
         "'q': [0.0] * num_instants}",
         "consensus_variables['ess']['z']['current'][node_id][year][day] = {'p': [0.0] * num_instants, "
         "'q': [0.0] * num_instants}")),
    'ess_copies_written_from_models': (
        '_update_shared_energy_storage_variables',
        ("p_req = pe.value(sess_model[node_id].es_pnet[y, d, p])",
         "p_req = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base",
         "p_req = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base",
         "shared_ess_vars['z']['current'][node_id][year][day][power_type][p] = z_new")),
    # (c) the Boyd channel fields
    'boyd_channel_fields': (
        'get_admm_boyd_residual_metrics',
        ("'r': r,", "'s': s,", "'eps_pri': eps_pri,", "'eps_dual': eps_dual,", "'primal_ratio': primal_ratio,",
         "'dual_ratio': dual_ratio,", "channels['all_boyd_pass'] = all(")),
}
LOOP_FACTS = (
    'boyd_metrics = get_admm_boyd_residual_metrics(planning_problem, tso_model, dso_models, esso_model, '
    'consensus_vars, dual_vars, admm_parameters)',
)
MODEL_SOURCE_FACTS = {
    'shared_energy_storage_data': (
        'model.es_pnet = pe.Var(model.years, model.days, model.periods, domain=pe.Reals, initialize=0.0)',
        'model.es_pch_per_unit = pe.Var(model.years, model.years, model.days, model.periods, '
        'domain=pe.NonNegativeReals, initialize=0.00)',
        'model.es_pdch_per_unit = pe.Var(model.years, model.years, model.days, model.periods, '
        'domain=pe.NonNegativeReals, initialize=0.00)',
        'model.salvage_value = pe.Expression(expr=salvage_value)',
        'model.feasibility_penalty = pe.Expression(expr=slack_penalty)'),
    'network': (
        'model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, '
        'model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)',
        'model.shared_es_pdch = pe.Var(model.shared_energy_storages, model.scenarios_market, '
        'model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)'),
}


def capture_path_checklist():
    """The source facts every capture relies on (a, b, c), plus the loop facts. Returns {name: bool}; raises nothing."""
    import shared_resources_planning as srp
    import shared_energy_storage_data as sed_mod
    import network as net_mod
    out = {}
    for name, (fn, snippets) in SOURCE_FACTS.items():
        src = inspect.getsource(getattr(srp, fn))
        missing = [s for s in snippets if s not in src]
        out[f'source:{name}'] = not missing
    rec_src = inspect.getsource(srp._get_operational_recourse_components)
    salvage_line = "terminal_salvage_value = planning_problem.shared_ess_data.get_salvage_value(models['esso'])"
    out['source:esso_not_in_gross_Q'] = ('shared_ess_data.get_primal_value' not in rec_src
                                         and rec_src.count("models['esso']") == 1 and salvage_line in rec_src
                                         and rec_src.find('gross_operational_cost = (') < rec_src.find(salvage_line))
    loop_src = inspect.getsource(srp._run_operational_planning)
    out['loop:boyd_call_once_per_cycle'] = (loop_src.count('get_admm_boyd_residual_metrics(') == 1
                                            and all(s in loop_src for s in LOOP_FACTS))
    boyd_at = loop_src.find(LOOP_FACTS[0])
    loop_at = loop_src.find('for iter in range(1, admm_parameters.num_max_iters + 1):')
    out['loop:boyd_after_esso_update_before_aa'] = (
        0 <= loop_at < loop_src.rfind('"update_sess": True', 0, max(boyd_at, 0)) < boyd_at
        < loop_src.find('aa_record = _anderson_acceleration_cycle_step('))
    out['loop:boyd_metrics_passed_to_penalties'] = (
        'tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, admm_parameters,' in loop_src)
    for mod_name, mod in (('shared_energy_storage_data', sed_mod), ('network', net_mod)):
        src = inspect.getsource(mod)
        out[f'model_source:{mod_name}'] = all(s in src for s in MODEL_SOURCE_FACTS[mod_name])
    out['writer_is_gate_result_io'] = callable(getattr(GRIO, 'dumps', None))
    return out


def assert_extension_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child); fail fast: W101's items (a) Boyd key + AA on, (b) source order,
    (c) certificate length written only by disable / restore, no early stop, (d) cap == 287 == the declaration's,
    (e) the three W104 references hash and hold exactly 1..187, (f) the rule constants, (h) the lambda_t capture path;
    plus signatures of the eight wrapped functions, the AA 'off' literal, the tail enabled, and the new capture
    paths (`capture_path_checklist`). Raises on any failure; returns the checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    decl = validate_settling_extension(decl)
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
        'c_certificate_length_written_only_by_disable_restore': (
            _certificate_length_writes_in_source() == sorted(CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(C101.CERTIFICATE_LENGTH_READS)),
        'd_cap_equals_287': cap == CAP == decl['cap'] == decl['replay_through_cycle'] + decl['extension_cycles'],
        'f_constants_equal_the_module': (rule == C101.settling_rule_declaration(P_MAX) and rule['tau'] == SC.TAU
                                         and rule['eps0'] == SC.TAU / 100.0 and rule['p_max'] == P_MAX),
        'f_p_max_function_named': (rule['p_max_function'] == 'settling_criterion.p_max_from_records'
                                   and callable(getattr(SC, 'p_max_from_records', None))),
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
    try:
        load_replay_reference(decl)
        checks['e_w104_references_hash_and_hold_exactly_1_187'] = True
    except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
        checks['e_w104_references_hash_and_hold_exactly_1_187'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W105 settling-extension preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the captures (pure reads; each returns (payload, errors))
# ======================================================================================================================
def ess_block_order(planning_problem):
    """[(node_id, year, day)] in production's own iteration order (as interface_dual_capture.block_order)."""
    return [(n, y, d) for n in planning_problem.active_distribution_network_nodes
            for y in planning_problem.years for d in planning_problem.days]


def _fv(v):
    """A Var's value as a Python float (None stays None)."""
    val = v.value
    return None if val is None else float(val)


def read_ess_schedules(planning_problem, tso_model, dso_models, esso_model, consensus_vars, blocks):
    """The ESS schedules at production's Boyd call (MW / Mvar), one list of periods per block (node, year, day):
    p / q for the consensus copies esso, tso, dso, z (consensus_vars['ess'][side]['current']), and charge / discharge
    for the ESSO (es_pch_per_unit / es_pdch_per_unit summed over cohorts y_inv, in es_pnet's units) and the TSO / DSO
    (probability-weighted shared_es_pch / shared_es_pdch x s_base, the network's shared ESS at the interface node /
    the DN reference node). Pure read. Returns (schedules, errors)."""
    errors = []
    out = {'p': {}, 'q': {}, 'charge': {}, 'discharge': {}}
    ess = consensus_vars['ess']
    for side in ESS_SIDES:
        for kind in ('p', 'q'):
            out[kind][side] = [[float(v) for v in ess[side]['current'][n][y][d][kind]] for n, y, d in blocks]
    years, days = list(planning_problem.years), list(planning_problem.days)
    tn = planning_problem.transmission_network
    for side in ESS_CHARGE_SIDES:
        out['charge'][side], out['discharge'][side] = [], []
    for n, y, d in blocks:
        # ESSO
        m = esso_model[n]
        yi, di = years.index(y), days.index(d)
        ch, dch = [], []
        for p in m.periods:
            vals_c = [_fv(m.es_pch_per_unit[y_inv, yi, di, p]) for y_inv in m.years]
            vals_d = [_fv(m.es_pdch_per_unit[y_inv, yi, di, p]) for y_inv in m.years]
            if any(v is None for v in vals_c + vals_d):
                errors.append(f'esso node {n} {y} {d} period {p}: a charge / discharge Var has no value')
            ch.append(sum(v for v in vals_c if v is not None))
            dch.append(sum(v for v in vals_d if v is not None))
        out['charge']['esso'].append(ch)
        out['discharge']['esso'].append(dch)
        # TSO and DSO
        tnet = tn.network[y][d]
        dnet = planning_problem.distribution_networks[n].network[y][d]
        for side, net, blk, e in (
                ('tso', tnet, tso_model[y][d], tnet.get_shared_energy_storage_idx(n)),
                ('dso', dnet, dso_models[n][y][d], dnet.get_shared_energy_storage_idx(dnet.get_reference_node_id()))):
            base = float(net.baseMVA)
            ch, dch = [], []
            for p in blk.periods:
                c_sum = d_sum = 0.0
                for s_m in blk.scenarios_market:
                    for s_o in blk.scenarios_operation:
                        prob = float(net.prob_market_scenarios[s_m]) * float(net.prob_operation_scenarios[s_o])
                        vc, vd = _fv(blk.shared_es_pch[e, s_m, s_o, p]), _fv(blk.shared_es_pdch[e, s_m, s_o, p])
                        if vc is None or vd is None:
                            errors.append(f'{side} node {n} {y} {d} period {p}: shared_es_pch / pdch has no value')
                            continue
                        c_sum += prob * vc * base
                        d_sum += prob * vd * base
                ch.append(c_sum)
                dch.append(d_sum)
            out['charge'][side].append(ch)
            out['discharge'][side].append(dch)
    return out, errors


def _abs_diff_sum(cur, prev):
    return sum(abs(a - b) for a, b in zip(cur, prev))


def ess_movement(schedules, prev, blocks, nodes):
    """Sum over years, days and periods of |x(k) - x(k-1)| per ESS node, for every schedule family and side, and the
    all-node totals. `prev` None (first cycle) -> None everywhere. Pure."""
    fams = [(kind, side) for kind in ('p', 'q') for side in ESS_SIDES] + \
           [(kind, side) for kind in ('charge', 'discharge') for side in ESS_CHARGE_SIDES]
    per_node = {str(n): {} for n in nodes}
    totals = {}
    for kind, side in fams:
        name = f'{kind}_{side}'
        tot = 0.0 if prev is not None else None
        for n in nodes:
            if prev is None:
                per_node[str(n)][name] = None
                continue
            s = 0.0
            for i, (bn, _y, _d) in enumerate(blocks):
                if bn == n:
                    s += _abs_diff_sum(schedules[kind][side][i], prev[kind][side][i])
            per_node[str(n)][name] = s
            tot += s
        totals[name] = tot
    return {'available': prev is not None, 'per_node': per_node, 'all_nodes': totals}


def boyd_capture(channels):
    """Every field production returns per channel (v, pf, ess) plus the top-level keys; Python scalars only. Pure."""
    def scal(v):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return v
        if isinstance(v, int):
            return int(v)
        return float(v)
    out = {g: {k: scal(v) for k, v in channels[g].items()} for g in CHANNELS}
    for k in ('all_boyd_pass', 'eps_abs', 'eps_rel', 'boyd_eps_source'):
        out[k] = scal(channels.get(k))
    return out


def q_decomposition(blocks, obj, gross, prev):
    """Per TSO / DSO block the weighted components summing to the block's contribution to gross Q, per-agent
    aggregates, deltas vs the previous cycle (`prev` = the previous cycle's return value or None), the reconciliation
    with gross Q, and the ESSO row (0: not in gross Q). `blocks` / `obj` are production's two block dicts. Pure."""
    rows = []
    agents = {}
    for key in sorted((k for k in blocks if k[0] in ('TSO', 'DSO')), key=lambda k: tuple(str(x) for x in k)):
        agent, node_id, year, day = key
        comp = obj[key]
        value = float(blocks[key])
        entry = {'agent': agent, 'node_id': node_id, 'year': str(year), 'day': str(day), 'value': value}
        for c in Q_COMPONENTS:
            entry[c] = float(comp[c])
        entry['other'] = value - float(comp['classified_total'])
        rows.append(entry)
        a = 'TSO' if agent == 'TSO' else f'DSO_{node_id}'
        agg = agents.setdefault(a, {c: 0.0 for c in Q_COMPONENTS_ALL + ('value',)})
        for c in Q_COMPONENTS_ALL + ('value',):
            agg[c] += entry[c]
    for agg in agents.values():
        agg['ess_terms'] = sum(agg[c] for c in ESS_TERMS)
    agents['ESSO'] = {**{c: 0.0 for c in Q_COMPONENTS_ALL + ('value',)}, 'ess_terms': 0.0,
                      'note': 'the ESSO does not enter gross_operational_cost (production source, asserted)'}
    total = sum(r['value'] for r in rows)
    comp_total = sum(sum(r[c] for c in Q_COMPONENTS_ALL) for r in rows)
    tol = max(1e-4, 1e-10 * max(abs(gross), 1.0)) if gross is not None else None
    out = {'components': list(Q_COMPONENTS_ALL), 'ess_terms_are': list(ESS_TERMS), 'blocks': rows, 'agents': agents,
           'reconciliation': {'gross': gross, 'sum_block_values': total, 'sum_components': comp_total,
                              'gross_minus_sum_block_values': (gross - total) if gross is not None else None,
                              'gross_minus_sum_components': (gross - comp_total) if gross is not None else None,
                              'tolerance': tol,
                              'reconciles': (bool(abs(gross - total) <= tol and abs(gross - comp_total) <= tol)
                                             if tol is not None else None)}}
    if prev is not None:
        pb = {(r['agent'], r['node_id'], r['year'], r['day']): r for r in prev['blocks']}
        deltas = []
        for r in rows:
            p = pb.get((r['agent'], r['node_id'], r['year'], r['day']))
            deltas.append({'agent': r['agent'], 'node_id': r['node_id'], 'year': r['year'], 'day': r['day'],
                           **{c: (r[c] - p[c]) if p is not None else None for c in Q_COMPONENTS_ALL + ('value',)}})
        out['blocks_delta'] = deltas
        out['agents_delta'] = {a: {c: v[c] - prev['agents'][a][c] for c in Q_COMPONENTS_ALL + ('value', 'ess_terms')}
                               for a, v in agents.items() if a in prev['agents']}
    return out


def esso_side_terms(planning_problem, esso_model):
    """Per ESSO node: its salvage value (the only ESSO term in the NET recourse) and its feasibility penalty (the
    ESSO objective's own terms: slacks + the epsilon-throughput regulariser). Reported beside Q; neither is in gross Q."""
    import pyomo.environ as pe
    return {str(n): {'salvage_value': float(pe.value(esso_model[n].salvage_value)),
                     'feasibility_penalty': float(pe.value(esso_model[n].feasibility_penalty))}
            for n in planning_problem.active_distribution_network_nodes}


# ======================================================================================================================
#  state and wrappers
# ======================================================================================================================
class ReportOnlyRule:
    """SettlingRule(N_HOLD, CAP, P_MAX), REPORT-ONLY and NON-LATCHING: after a certification the decision is recorded
    and cleared, so the next observe() continues from the same sign / turning-point state (a certification changes no
    state of the rule except `decision`, which is what `observe` refuses to continue past)."""

    def __init__(self, n=N_HOLD, cap=CAP, p_max=P_MAX):
        self.rule = SC.SettlingRule(n, cap, p_max)
        self.certifications = []
        self.first = None
        self.cap_decision = None

    def observe(self, k, q, boyd):
        rec = self.rule.observe(k, q, boyd)
        dec = self.rule.decision
        if dec is not None:
            if dec.get('status') == 'certified':
                entry = {'cycle': k, 'branch': dec['branch'], 'Q_k': dec.get('Q_k_star'), 'band': dec.get('band'),
                         'band_width': dec.get('band_width'), 'range_over_tau': dec.get('range_over_tau'),
                         'k0': dec.get('k0'), 'P_hat': dec.get('P_hat'), 'W': dec.get('W'), 'window': dec.get('window')}
                self.certifications.append(entry)
                if self.first is None:
                    self.first = dict(dec)
            else:
                self.cap_decision = dict(dec)
            self.rule.decision = None      # non-latching: report-only
        return rec


class ExtensionState:
    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        self.decl = decl
        self.n = decl['hold_after_cycle']
        self.replay_through = decl['replay_through_cycle']
        self.cap = cap
        self.eval_dir = eval_dir
        rows, lines, dec = reference if reference is not None else ({}, {}, None)
        self.ref_rows, self.ref_lines, self.ref_decision = rows, lines, dec
        self.sink = sink
        self.mirror = SC.SettlingRule(self.n, self.replay_through, decl['w104_mirror_rule']['p_max'])
        self.mirror_decision = None
        self.rule = ReportOnlyRule(self.n, cap, decl['settling_rule']['p_max'])
        self.phase = 'before'
        self.cycle = None
        self.params = None
        self.threshold_original = None
        self.threshold_restored = None
        self.gross = {}
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
        self.errors = []
        self.capture_errors = []
        self.events = []
        self.decision_written = False
        self.t0 = time.time()

    def _write(self, fname, obj):
        if self.sink is not None:
            self.sink.append((fname, obj))
            return len(GRIO.dumps(obj, default=GRIO.json_default)) + 1
        text = GRIO.dumps(obj, default=GRIO.json_default) + '\n'
        with open(os.path.join(self.eval_dir, fname), 'a') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return len(text.encode())

    def _write_decision(self, obj):
        if self.sink is not None:
            self.sink.append((DECISION_FILE, obj))
            return
        with open(os.path.join(self.eval_dir, DECISION_FILE), 'x') as handle:
            GRIO.dump(obj, handle, default=GRIO.json_default, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self):
        first = self.rule.first
        return {
            'schema': SCHEMA, 'declaration': self.decl, 'cell': self.decl['cell'], 'N_hold': self.n,
            'replay_through': self.replay_through, 'cap': self.cap, 'phase': self.phase,
            'certificate_length_original': self.threshold_original,
            'certificate_length_raised_to': CERTIFICATION_DISABLED_THRESHOLD,
            'certificate_length_restored_at_exit': self.threshold_restored,
            'last_cycle': self.cycle, 'cycle_lines_written': self.lines, 'creep_lines_written': self.creep_lines,
            'ess_schedule_lines_written': self.ess_lines,
            'replay_bitwise_through_cycle': self.replay_equal_through,
            'replay_first_divergence': self.first_divergence,
            'mirror_decision_equals_w104': (None if self.mirror_decision is None else
                                            self.mirror_decision.get('equals_w104')),
            'report_only_first_would_certify_cycle': (first or {}).get('k_star'),
            'report_only_first_would_certify_branch': (first or {}).get('branch'),
            'report_only_would_certify_cycles': [c['cycle'] for c in self.rule.certifications],
            'report_only_status_at_cap': (self.rule.cap_decision or {}).get('status'),
            'stopped_by': ('replay_divergence_abort' if self.first_divergence is not None else
                           'cap' if self.cycle == self.cap else 'other'),
            'lapse_events': list(self.rule.rule.lapses),
            'errors': list(self.errors), 'capture_errors': list(self.capture_errors[:50]),
            'n_capture_errors': len(self.capture_errors),
            'ok': (not self.errors and not self.capture_errors and self.phase == 'ended'
                   and self.threshold_restored == self.threshold_original and self.cycle == self.cap
                   and self.lines == self.cap and self.creep_lines == self.cap and self.ess_lines == self.cap
                   and self.first_divergence is None and self.replay_equal_through == self.replay_through
                   and (self.mirror_decision or {}).get('equals_w104') is True and self.decision_written),
            'files': {'cycle_record': CYCLE_FILE, 'blocks_all': BLOCKS_FILE, 'creep': CREEP_FILE,
                      'ess_schedule': ESS_SCHEDULE_FILE, 'decision': DECISION_FILE},
        }


def _jtext(value):
    return GRIO.dumps(value, default=GRIO.json_default, sort_keys=True)


def line_compare(line, ref_line):
    """{field: {run, recorded}} for every CYCLE_LINE_GATED_FIELD (and the mirror's settling record vs W104's
    'settling') whose JSON text differs."""
    diff = {}
    for f in CYCLE_LINE_GATED_FIELDS:
        a, b = _jtext(line.get(f)), _jtext(ref_line.get(f))
        if a != b:
            diff[f] = {'run': a[:400], 'recorded': b[:400]}
    a, b = _jtext(line.get('settling_w104_mirror')), _jtext(ref_line.get('settling'))
    if a != b:
        diff['settling_w104_mirror_vs_settling'] = {'run': a[:400], 'recorded': b[:400]}
    return diff


def make_wrappers(st, originals, srp_module=None):
    """The eight wrappers over `originals` (name -> callable). Separated from the context manager so the zero-solve
    checks can drive them with stand-in originals and recorded values."""

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W105 settling-extension hook: {msg}')

    def w_baseline(planning_problem, admm_parameters):
        out = originals['_capture_convergence_depth_tail_baseline'](planning_problem, admm_parameters)
        if st.params is not None:
            raise_('tail baseline captured twice (a second ADMM call inside one extension run)')
        st.params = admm_parameters
        st.threshold_original = admm_parameters.minimum_consecutive_converged_cycles
        admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD
        st.events.append({'event': 'certificate_disabled', 'from': st.threshold_original,
                          'to': CERTIFICATION_DISABLED_THRESHOLD})
        return out

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
        st.cur = {'cycle': cycle, 'phase': 'replay' if cycle <= st.replay_through else 'extension',
                  'regime': 'pre_hold' if cycle <= st.n else 'hold', 't_start_s': time.time() - st.t0}
        if cycle <= st.n:
            st.cur['tail_apply'] = {'hold': False, 'active_passed': bool(active)}
            return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
        st.cur['tail_apply'] = {'hold': True, 'natural_active': bool(active), 'active_passed': True,
                                'hold_changed_value': not bool(active)}
        return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, True, baseline, cycle)

    def w_boyd(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):
        """Production's Boyd metrics, returned UNCHANGED (the same object); captures (b) and (c) are pure reads."""
        result = originals['get_admm_boyd_residual_metrics'](planning_problem, tso_model, dso_models, esso_model,
                                                             consensus_vars, dual_vars, admm_parameters)
        if st.phase != 'in_cycle' or st.cur is None or st.cur.get('finalized') or '_boyd_seen' in st.cur:
            return result
        st.cur['_boyd_seen'] = True
        t = time.time()
        cap = {'cycle': st.cycle}
        try:
            cap['boyd'] = boyd_capture(result)
        except Exception as error:  # noqa: BLE001 -- write-only capture: recorded, never raised
            cap['boyd'] = None
            st.capture_errors.append({'cycle': st.cycle, 'capture': 'boyd', 'error': f'{type(error).__name__}: {error}'})
        try:
            if st.ess_blocks is None:
                st.ess_blocks = ess_block_order(planning_problem)
            sched, errs = read_ess_schedules(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                                             st.ess_blocks)
            for e in errs[:5]:
                st.capture_errors.append({'cycle': st.cycle, 'capture': 'ess', 'error': e})
            prev = st.prev_ess if st.prev_ess_cycle == st.cycle - 1 else None
            nodes = list(planning_problem.active_distribution_network_nodes)
            cap['ess_movement'] = ess_movement(sched, prev, st.ess_blocks, nodes)
            cap['ess_schedules'] = sched
            st.prev_ess, st.prev_ess_cycle = sched, st.cycle
        except Exception as error:  # noqa: BLE001
            cap['ess_movement'] = None
            cap['ess_schedules'] = None
            st.capture_errors.append({'cycle': st.cycle, 'capture': 'ess', 'error': f'{type(error).__name__}: {error}'})
        cap['capture_s'] = time.time() - t
        st.cur['_boyd_capture'] = cap
        return result

    def w_aa(aa_state, aa_layout, consensus_vars, dual_vars, w_before, rho_channel, boyd_metrics, iter):
        if iter != st.cycle:
            raise_(f'AA step at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        st.cur['boyd_k'] = bool(boyd_metrics['all_boyd_pass'])
        st.cur['boyd_ratios_at_aa'] = {f'boyd_{g}_{kind}_ratio': boyd_metrics[g][f'{kind}_ratio']
                                       for g in CHANNELS for kind in ('primal', 'dual')}
        if iter <= st.n:
            rec = originals['_anderson_acceleration_cycle_step'](aa_state, aa_layout, consensus_vars, dual_vars,
                                                                 w_before, rho_channel, boyd_metrics, iter)
            st.cur['aa'] = {'hold': False, 'action': (rec or {}).get('action')}
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
        if c is None or c <= st.n:
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

    def _observe(q, boyd):
        """Both rules for this cycle: the W104 mirror (cycles <= 187, gated) and the report-only rule (every cycle).
        Neither writes the certificate length."""
        c = st.cycle
        if c <= st.replay_through:
            st.cur['settling_w104_mirror'] = st.mirror.observe(c, q, boyd)
            if st.mirror.decision is not None and st.mirror_decision is None:
                dec = dict(st.mirror.decision)
                dec.update({'cell': BASE_CELL, 'N': st.n, 'cap': st.replay_through, 'decided_at_cycle': c,
                            'Q_N': st.gross.get(st.n)})
                equal = st.ref_decision is not None and _jtext(dec) == _jtext(st.ref_decision)
                st.mirror_decision = {'decision': dec, 'equals_w104': bool(equal)}
                st.cur['mirror_decision_equals_w104'] = bool(equal)
        rec = st.rule.observe(c, q, boyd)
        st.cur['settling'] = rec
        st.cur['report_only_would_certify'] = str(rec.get('decision') or '').startswith('certified')

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
        _observe(gross, st.cur['boyd_k'])
        return rc

    def _capture_blocks(planning_problem, models, c, rc):
        """W101's all-block line, UNCHANGED (so cycles 1..187 compare bitwise with W104's), then capture (a) from the
        SAME two block dicts (no extra production call)."""
        t = time.time()
        blocks = srp_module._get_operational_recourse_block_components(planning_problem, models)
        obj = srp_module._get_operational_objective_component_blocks(planning_problem, models)
        net = rc.get('net_operational_recourse')
        total = sum(blocks.values())
        tol = max(1e-4, 1e-10 * max(abs(net), 1.0)) if net is not None else None
        line = {'cycle': c, 'phase': 'replay' if c <= st.n else 'continuation',
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
        # capture (a)
        t2 = time.time()
        try:
            prev = st.prev_q if st.prev_q_cycle == c - 1 else None
            qd = q_decomposition(blocks, obj, rc.get('gross_operational_cost'), prev)
            if qd['reconciliation']['reconciles'] is not True:
                st.capture_errors.append({'cycle': c, 'capture': 'q_decomposition',
                                          'error': f"does not reconcile: {qd['reconciliation']}"})
            try:
                qd['esso_side_terms_not_in_gross_Q'] = esso_side_terms(planning_problem, models['esso'])
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
        hold = iter > st.n
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
        if 'gross' not in cur:          # a failed cycle: no Q, no AA step -> the rules see (None, False)
            cur['gross'] = None
            cur['gross_hex'] = None
            cur['step'] = None
            cur['boyd_k'] = False
            _observe(None, False)
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
        # the creep line (captures a-c) -- pulled out of the cycle line so the cycle line keeps W101's size
        boyd_cap = cur.pop('_boyd_capture', None)
        qd = cur.pop('_q_decomposition', None)
        q_s = cur.pop('_q_capture_s', None)
        cur.pop('_boyd_seen', None)
        if boyd_cap is None:
            st.capture_errors.append({'cycle': c, 'capture': 'boyd', 'error': 'no Boyd call observed this cycle'})
            boyd_cap = {}
        # the replay gates (cycles <= 187)
        divergence = None
        ref_row = st.ref_rows.get(c)
        if ref_row is not None:
            diff, detail = C101.replay_compare(run_values, ref_row)
            ldiff = line_compare(cur, st.ref_lines.get(c) or {})
            if c == st.replay_through and not (st.mirror_decision or {}).get('equals_w104'):
                ldiff['mirror_decision_vs_w104_decision'] = {
                    'run': _jtext((st.mirror_decision or {}).get('decision'))[:400], 'recorded': _jtext(st.ref_decision)[:400]}
            cur['replay_equal'] = not diff and not ldiff
            if diff or ldiff:
                divergence = {'cycle': c, **detail, 'cycle_line_fields_differing': sorted(ldiff),
                              'cycle_line_values': ldiff}
                st.first_divergence = divergence
                cur['replay_divergence'] = divergence
            elif st.replay_equal_through == c - 1:
                st.replay_equal_through = c
        else:
            cur['replay_equal'] = None
        cur['t_end_s'] = time.time() - st.t0
        cur['finalized'] = True
        st._write(CYCLE_FILE, cur)
        st.lines += 1
        # capture (b) raw schedules and the creep line
        sched = boyd_cap.pop('ess_schedules', None)
        if sched is not None:
            if st.ess_lines == 0:
                st._write(ESS_SCHEDULE_FILE, {'header': True, 'schema': SCHEMA + '/ess_schedule',
                                              'blocks': [[n, str(y), str(d)] for n, y, d in st.ess_blocks],
                                              'n_blocks': len(st.ess_blocks),
                                              'families': {'p': list(ESS_SIDES), 'q': list(ESS_SIDES),
                                                           'charge': list(ESS_CHARGE_SIDES),
                                                           'discharge': list(ESS_CHARGE_SIDES)},
                                              'units': 'MW (p, charge, discharge) / Mvar (q)',
                                              'read_at': 'production get_admm_boyd_residual_metrics call (after the '
                                                         "cycle's ESSO consensus / dual update, before AA)"})
            st._write(ESS_SCHEDULE_FILE, {'cycle': c, **sched})
            st.ess_lines += 1
        elif st.ess_lines == 0:
            st.capture_errors.append({'cycle': c, 'capture': 'ess', 'error': 'no schedule this cycle'})
        creep = {'cycle': c, 'phase': cur['phase'], 'regime': cur['regime'], 'gross': cur.get('gross'),
                 'step': cur.get('step'), 'local_solves_ok': bool(local_solves_ok), 'boyd_k': cur.get('boyd_k'),
                 'q_decomposition': qd, 'ess_movement': boyd_cap.get('ess_movement'), 'boyd': boyd_cap.get('boyd'),
                 'capture_s': {'boyd_and_ess': boyd_cap.get('capture_s'), 'q_decomposition': q_s}}
        st._write(CREEP_FILE, creep)
        st.creep_lines += 1
        s = cur.get('settling') or {}
        print(f"[W105-EXT] cycle {c} ({cur['phase']}/{cur['regime']}) Q={cur.get('gross')!r} step={cur.get('step')} "
              f"boyd_k={cur.get('boyd_k')} pf_primal={run_values['boyd_pf_primal_ratio']} "
              f"ess_primal={run_values['boyd_ess_primal_ratio']} replay_equal={cur.get('replay_equal')} "
              f"k0={s.get('k0')} len_T={s.get('len_T')} range={s.get('range')} decision={s.get('decision')} "
              f"holds={cur['holds']}", flush=True)
        if c == st.cap:
            _write_extension_decision()
        if divergence is not None and st.decl['abort_on_replay_divergence']:
            raise_(f"REPLAY DIVERGED at cycle {c} (per-cycle fields {divergence['fields_differing']}, cycle-line "
                   f"fields {divergence['cycle_line_fields_differing']}, gross difference "
                   f"{divergence['gross_difference_run_minus_recorded']!r}, max relative "
                   f"{divergence['max_relative_difference']!r}) -- run ABORTED (W105: no relabelled continuation)")

    def _write_extension_decision():
        first = st.rule.first
        st._write_decision({
            'schema': SCHEMA + '/decision', 'mode': RULE_MODE, 'cell': st.decl['cell'], 'N_hold': st.n,
            'cap': st.cap, 'report_only': True,
            'first_would_certify': first,
            'would_certify_cycles': [x['cycle'] for x in st.rule.certifications],
            'would_certify_records': st.rule.certifications,
            'status_at_cap': st.rule.cap_decision,
            'w104_mirror_decision': (st.mirror_decision or {}).get('decision'),
            'w104_mirror_decision_equals_w104': (st.mirror_decision or {}).get('equals_w104'),
            'lapse_events': list(st.rule.rule.lapses)})
        st.decision_written = True

    return {'_capture_convergence_depth_tail_baseline': w_baseline, '_apply_convergence_depth_tail': w_apply,
            'get_admm_boyd_residual_metrics': w_boyd,
            '_anderson_acceleration_cycle_step': w_aa, '_convergence_depth_tail_next_state': w_next,
            '_update_admm_penalties': w_penalties, '_get_operational_recourse_components': w_recourse,
            '_get_admm_efc_per_day_max': w_efc}


@contextmanager
def settling_extension_hooks(eval_dir, decl, holder, cap):
    """Install the wrappers for the run (the harness enters this FIRST, so these wrap production directly and every
    harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    decl = validate_settling_extension(decl)
    if int(cap) != CAP:
        raise RuntimeError(f'settling extension: spec cap {cap} != {CAP}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ExtensionState(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
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
