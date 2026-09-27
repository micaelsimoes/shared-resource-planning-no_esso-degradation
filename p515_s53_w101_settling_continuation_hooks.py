"""
P5.15 Addendum 53, Planner task W101 -- the SRP1 SETTLING CONTINUATION hooks (frozen stage spec v39).

WHAT THIS MODULE IS. A child-side, harness-installed set of pass-through wrappers on SEVEN production functions of
`shared_resources_planning` (module globals resolved by `_run_operational_planning` at call time -- W98's technique;
this module is W98's `p515_s53_w98_continuation_hooks` adapted). No production module is edited. It is installed ONLY
for a campaign-spec entry that declares `settling_continuation` (`p515_s44_campaign_harness`, W101); the declaration
enters the entry's eval key, so a continuation never shares a key, an eval dir or a working dir with the certified
evaluation it replays.

THE RUN IT SERVES (Addendum 53 Ruling 2). One of the three SRP1 tight-tail references (campaign s53_w86_tail_recert,
spec ddd6cd44; x0 certified at N = 132, n7_4h_e1 at 112, c_star at 87) re-run from scratch under the IDENTICAL
configuration, with
  * cycles 1..N REPLAYED and gated BITWISE, cycle by cycle, against the committed per-cycle record of the recert: at
    the end of every cycle k <= N the fields production hands the wrappers (`REPLAY_GATED_FIELDS`) are compared as JSON
    text with the recorded row k; on the FIRST difference the cycle line is written with the cycle and magnitude and
    the run is ABORTED (RuntimeError -> the child's barrier record). It is never continued as a relabelled run;
  * the certification rule DISABLED (the certificate length is raised to 10**9 before cycle 1);
  * for every cycle > N the certifying regime HELD exactly as W98 did: Anderson acceleration OFF (production's own
    'off' branch, even when the Boyd test lapses -- AA is non-latching in production), the tight tail ON, rho FROZEN;
  * the settling stop rule (`settling_criterion.SettlingRule`, the pure function) as the ONLY way to end before the
    cap N + 100: it observes (Q_k, boyd_k) every cycle inside the recourse wrapper -- where production has just
    computed Q_k and before its convergence test -- and on certification sets the certificate length to 0, so
    production's own exit test ends the loop at the end of THIS cycle (W98's mechanism). There is NO early-stop rule.

boyd_k = `boyd_metrics['all_boyd_pass']` AND local_solves_ok, taken from THIS cycle's `boyd_metrics` as the AA wrapper
receives it (production calls the AA step only when every local solve succeeded; it updates
`consecutive_converged_cycles` only after the recourse wrapper). A cycle whose local solves failed has no Q (neither
the AA step nor the recourse function is called) and is observed as (None, False) at the end of the cycle.

THE WRAPPERS (each calls production; for cycle <= N each passes its arguments through as the SAME objects and returns
production's value UNCHANGED):
  1. `_capture_convergence_depth_tail_baseline` (once, before cycle 1): certificate length -> 10**9.
  2. `_apply_convergence_depth_tail` (top of every cycle; `cycle=None` once at loop exit): the CYCLE TRACKER; TAIL HOLD
     (a) for cycle > N; restores the certificate length at loop exit.
  3. `_anderson_acceleration_cycle_step`: records boyd_k and the six Boyd ratios; AA HOLD for cycle > N (production's
     step with `all_boyd_pass` forced True on a shallow COPY -> its 'off' branch; asserted).
  4. `_convergence_depth_tail_next_state`: TAIL HOLD (b) for cycle > N.
  5. `_get_operational_recourse_components`: Q_k and every block's recourse (with deltas); the SETTLING RULE.
  6. `_update_admm_penalties`: RHO HOLD for cycle > N (its result is kept for the cycle's finalisation).
  7. `_get_admm_efc_per_day_max` (production reads it once per cycle, AFTER the penalty update and right before the
     diagnostics row): records the value, then FINALISES the cycle -- the in-cycle REPLAY GATE for cycle <= N (abort on
     the first difference), the observation of a failed cycle by the rule, the cycle line.

Zero solves: nothing here solves or builds a model. Stdlib only at import (the harness parent imports it for keys).
"""
import inspect
import json
import math
import os
import time
from contextlib import contextmanager

import gate_result_io as GRIO
import settling_criterion as SC

SCHEMA = 'p515_s53_w101_settling_continuation_v1'
LABEL = ('SETTLING CONTINUATION RUN (W101, spec v39) -- replay of a certified SRP1 reference gated bitwise through '
         'cycle N (abort on divergence), then the certifying regime held (AA off, tight tail on, rho frozen) until the '
         'settling stop rule certifies or the cap N + 100')
OPTION_NAME = 'settling_continuation'
DECLARATION_KEYS = frozenset({'label', 'cell', 'hold_after_cycle', 'cap_after_hold', 'settling_rule',
                              'record_all_blocks', 'replay_reference', 'abort_on_replay_divergence'})
SETTLING_RULE_KEYS = frozenset({'module', 'function', 'tau', 'eps0', 'k_excl', 'w_min', 'w_factor', 'p_max', 'l_mono',
                                'p_max_function'})
REPLAY_REFERENCE_KEYS = frozenset({'per_cycle_record', 'sha256', 'n_cycles'})
FORBIDDEN_DECLARATION_KEYS = ('early_stop',)
CERTIFICATION_DISABLED_THRESHOLD = 10 ** 9   # > any cap: production's own certificate can never complete
SETTLING_CERTIFIED_THRESHOLD = 0             # consecutive_converged_cycles >= 0 always: the loop exits this cycle
AA_OFF_ACTION = 'off (all channels within Boyd tolerance)'
CERTIFICATE_LENGTH_READS = (
    'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)',
    "'required_consecutive_cycles': admm_parameters.minimum_consecutive_converged_cycles,",
    "f'{admm_parameters.minimum_consecutive_converged_cycles} | '",
)
# The ONLY three assignments of the certificate length in this module (checklist (c), asserted from source): the
# disable at the tail baseline, the restore at the loop exit, and the settling decision.
CERTIFICATE_LENGTH_WRITES = (
    'admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD',
    'st.params.minimum_consecutive_converged_cycles = st.threshold_original',
    'st.params.minimum_consecutive_converged_cycles = SETTLING_CERTIFIED_THRESHOLD',
)
SOURCE_ORDER = ('_anderson_acceleration_cycle_step(', '_get_operational_recourse_components(',
                'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)')
CYCLE_FILE = 'settling_continuation_cycle_record.jsonl'
BLOCKS_FILE = 'recourse_blocks_all.jsonl'
DECISION_FILE = 'settling_decision.json'
SUMMARY_KEY = 'settling_continuation'
CHANNELS = ('v', 'pf', 'ess')
BOYD_RATIO_FIELDS = tuple(f'boyd_{g}_{kind}_ratio' for g in CHANNELS for kind in ('primal', 'dual'))
# The per_cycle_record.jsonl fields the wrappers hold at the end of cycle k (gated in-cycle, bitwise as JSON text).
# The three remaining fields -- objective_change_abs / objective_tolerance / objective_change_ratio (functions of the
# net recourse of k and of the previous cycle, both gated here, and of the tolerance configuration) -- are compared
# post-run by the launcher's full-record replay gate.
REPLAY_GATED_FIELDS = (
    'cycle', 'local_solves_ok', 'recourse', 'gross_operational_cost', 'terminal_salvage_value',
    'cycle_convergence', 'consecutive_converged_cycles', 'boyd_all_pass', 'boyd_stop',
    'boyd_v_primal_ratio', 'boyd_v_dual_ratio', 'boyd_v_channel_pass',
    'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio', 'boyd_pf_channel_pass',
    'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_ess_channel_pass',
    'rho_v_after', 'rho_pf_after', 'rho_ess_after', 'rho_v_action', 'rho_pf_action', 'rho_ess_action',
    'rho_freeze_active', 'efc_per_day_max',
)
REPLAY_POST_RUN_ONLY_FIELDS = ('objective_change_abs', 'objective_tolerance', 'objective_change_ratio')
WRAPPED = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
           '_anderson_acceleration_cycle_step', '_convergence_depth_tail_next_state', '_update_admm_penalties',
           '_get_operational_recourse_components', '_get_admm_efc_per_day_max')
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the three SRP1 cells (the tight-tail re-certification, the basis of R_ref = 259,375.33) --------------------------
RECERT_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
CELLS = {
    'x0': {'eval_key16': '5cfe69a615ae3708', 'eval_dir': '5cfe69a615ae3708_x0', 'N': 132,
           'per_cycle_record_sha256': 'bd7e47057ee618486fed0a385e8f784ecfd0089e77d5da2aaef13e89fed2f4a2'},
    'n7_4h_e1': {'eval_key16': 'ca8927e75d628bd1', 'eval_dir': 'ca8927e75d628bd1_n7_4h_e1', 'N': 112,
                 'per_cycle_record_sha256': 'b5875bbe9908cc3dfd14add7afb168622698b5d6ff500d50acb1117962d58d1f'},
    'c_star': {'eval_key16': '96c5aa50cc229cc1', 'eval_dir': '96c5aa50cc229cc1_c_star', 'N': 87,
               'per_cycle_record_sha256': 'dd580b247b7f917638af6393fdbdebdbf96d37a0e520342849fb7d11975deeb8'},
}
CELL_ORDER = ('x0', 'n7_4h_e1', 'c_star')


def reference_path(cell):
    return os.path.join(RECERT_ROOT, 'evals', CELLS[cell]['eval_dir'], 'per_cycle_record.jsonl')


def settling_rule_declaration(p_max):
    return {'module': 'settling_criterion', 'function': 'settling_criterion.SettlingRule', 'tau': SC.TAU,
            'eps0': SC.EPS0, 'k_excl': SC.K_EXCL, 'w_min': SC.W_MIN, 'w_factor': SC.W_FACTOR, 'p_max': p_max,
            'l_mono': 2 * p_max, 'p_max_function': 'settling_criterion.p_max_from_records'}


def declaration_for(cell, p_max):
    c = CELLS[cell]
    return validate_settling_continuation({
        'label': LABEL, 'cell': cell, 'hold_after_cycle': c['N'], 'cap_after_hold': SC.CAP_AFTER_N,
        'settling_rule': settling_rule_declaration(p_max), 'record_all_blocks': True,
        'replay_reference': {'per_cycle_record': reference_path(cell), 'sha256': c['per_cycle_record_sha256'],
                             'n_cycles': c['N']},
        'abort_on_replay_divergence': True})


def _is_pos_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 1


def validate_settling_continuation(value):
    """None = not declared. Otherwise EXACTLY `DECLARATION_KEYS` (an `early_stop` key is refused by name): label ==
    LABEL; cell one of CELLS with hold_after_cycle == its N; cap_after_hold == 100; settling_rule == the module's own
    constants with an integer p_max >= 1 and l_mono == 2 p_max; record_all_blocks True; replay_reference = {the cell's
    committed per-cycle record, its sha256, n_cycles == N}; abort_on_replay_divergence True. Returns a new dict.
    Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} must NOT carry {forbidden} (the settling rule is the only stop before the cap)')
    if set(value) != DECLARATION_KEYS:
        raise ValueError(f'{OPTION_NAME} must have exactly {sorted(DECLARATION_KEYS)}; got {sorted(value)}')
    if value['label'] != LABEL:
        raise ValueError(f'{OPTION_NAME}.label must be {LABEL!r}')
    cell = value['cell']
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME}.cell must be one of {sorted(CELLS)}; got {cell!r}')
    n = value['hold_after_cycle']
    if not _is_pos_int(n) or n != CELLS[cell]['N']:
        raise ValueError(f'hold_after_cycle must be the recorded certification cycle {CELLS[cell]["N"]}; got {n!r}')
    if value['cap_after_hold'] != SC.CAP_AFTER_N or isinstance(value['cap_after_hold'], bool):
        raise ValueError(f'cap_after_hold must be {SC.CAP_AFTER_N}')
    rule = value['settling_rule']
    if not isinstance(rule, dict) or set(rule) != SETTLING_RULE_KEYS:
        raise ValueError(f'settling_rule must have exactly {sorted(SETTLING_RULE_KEYS)}; got {rule!r}')
    p_max = rule['p_max']
    if not _is_pos_int(p_max) or rule != settling_rule_declaration(p_max):
        raise ValueError(f'settling_rule must equal settling_criterion\'s constants with p_max >= 1; got {rule!r}')
    if value['record_all_blocks'] is not True or value['abort_on_replay_divergence'] is not True:
        raise ValueError('record_all_blocks and abort_on_replay_divergence must be True')
    ref = value['replay_reference']
    if not isinstance(ref, dict) or set(ref) != REPLAY_REFERENCE_KEYS:
        raise ValueError(f'replay_reference must have exactly {sorted(REPLAY_REFERENCE_KEYS)}; got {ref!r}')
    if (ref['per_cycle_record'] != reference_path(cell) or ref['sha256'] != CELLS[cell]['per_cycle_record_sha256']
            or ref['n_cycles'] != n):
        raise ValueError(f'replay_reference must be the {cell} recert record (sha256, n_cycles == N); got {ref!r}')
    return {'label': LABEL, 'cell': cell, 'hold_after_cycle': n, 'cap_after_hold': value['cap_after_hold'],
            'settling_rule': dict(rule), 'record_all_blocks': True,
            'replay_reference': {'per_cycle_record': ref['per_cycle_record'], 'sha256': ref['sha256'],
                                 'n_cycles': ref['n_cycles']},
            'abort_on_replay_divergence': True}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_replay_reference(declaration):
    """{cycle: row} from the certified cell's committed per_cycle_record.jsonl; refuses unless it hashes to the
    declaration and holds exactly cycles 1..N."""
    ref = declaration['replay_reference']
    path = os.path.join(REPO, ref['per_cycle_record'])
    got = _sha256(path)
    if got != ref['sha256']:
        raise RuntimeError(f'replay reference {ref["per_cycle_record"]} sha256 {got} != declared {ref["sha256"]}')
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[int(r['cycle'])] = r
    if sorted(out) != list(range(1, ref['n_cycles'] + 1)):
        raise RuntimeError(f'replay reference cycles {sorted(out)[:3]}..{sorted(out)[-3:]} != 1..{ref["n_cycles"]}')
    missing = [f for f in REPLAY_GATED_FIELDS + REPLAY_POST_RUN_ONLY_FIELDS if f not in out[1]]
    if missing:
        raise RuntimeError(f'replay reference rows lack gated fields {missing}')
    return out


def _certificate_length_writes_in_source():
    src = inspect.getsource(make_wrappers)
    return sorted(line.strip() for line in src.splitlines()
                  if 'minimum_consecutive_converged_cycles =' in line and not line.strip().startswith('#'))


def assert_settling_preconditions(declaration, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child); fail fast. (a) the Boyd key and the AA wrapper with AA on;
    (b) production's source order AA step < recourse < convergence test; (c) no early stop in the declaration and the
    certificate length written only by the disable / restore / settling decision; (d) cap == N + 100; (e) the replay
    reference hashes and holds exactly 1..N; (f) the rule's constants equal the module's with the P_MAX function
    named; (h) the lambda_t capture path (`interface_dual_capture.assert_capture_path`); plus signatures, the AA 'off'
    literal, the certificate-length reads, the tail enabled. Raises on any failure; returns the checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    n = declaration['hold_after_cycle']
    cap = int(spec['cap'])
    params = {
        '_capture_convergence_depth_tail_baseline': ['planning_problem', 'admm_parameters'],
        '_apply_convergence_depth_tail': ['planning_problem', 'admm_parameters', 'active', 'baseline', 'cycle'],
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
    pos = [loop_src.find(s) for s in SOURCE_ORDER]
    rule = declaration['settling_rule']
    checks = {
        'a_boyd_all_pass_key_in_production': "'all_boyd_pass'" in boyd_src,
        'a_aa_step_called_with_boyd_metrics': ('aa_record = _anderson_acceleration_cycle_step(\n'
                                               '                    aa_state, aa_layout, consensus_vars, dual_vars,\n'
                                               '                    aa_w_before, aa_rho_before, boyd_metrics, iter,')
        in loop_src,
        'a_anderson_acceleration_on': bool(aa_on),
        'b_source_order_aa_step_lt_recourse_lt_convergence_test': all(p >= 0 for p in pos) and pos[0] < pos[1] < pos[2],
        'c_no_early_stop_in_declaration': not any(k in declaration for k in FORBIDDEN_DECLARATION_KEYS),
        'c_certificate_length_written_only_by_disable_restore_settling': (
            _certificate_length_writes_in_source() == sorted(CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(CERTIFICATE_LENGTH_READS)),
        'd_cap_equals_N_plus_100': cap == n + SC.CAP_AFTER_N == n + declaration['cap_after_hold'],
        'f_constants_equal_the_module': (rule == settling_rule_declaration(rule['p_max'])
                                         and rule['tau'] == SC.TAU and rule['eps0'] == SC.TAU / 100.0),
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
        load_replay_reference(declaration)
        checks['e_replay_reference_hashes_and_holds_exactly_1_N'] = True
    except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
        checks['e_replay_reference_hashes_and_holds_exactly_1_N'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W101 settling-continuation preconditions fail (before any solve): {failing}')
    return checks


def _jtext(value):
    return GRIO.dumps(value, default=GRIO.json_default, sort_keys=True)


class ContinuationState:
    """Per-run bookkeeping. `n` = hold_after_cycle."""

    def __init__(self, declaration, eval_dir, cap, reference=None, sink=None):
        self.decl = declaration
        self.n = declaration['hold_after_cycle']
        self.cap = cap
        self.eval_dir = eval_dir
        self.reference = reference or {}
        self.sink = sink
        self.rule = SC.SettlingRule(self.n, cap, declaration['settling_rule']['p_max'])
        self.phase = 'before'
        self.cycle = None
        self.params = None
        self.threshold_original = None
        self.threshold_restored = None
        self.gross = {}
        self.prev_blocks = None
        self.prev_blocks_cycle = None
        self.consecutive = 0
        self.decision = None
        self.certified_cycle = None
        self.cur = None
        self.lines = 0
        self.replay_equal_through = 0
        self.first_divergence = None
        self.errors = []
        self.events = []
        self.t0 = time.time()

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
            return
        path = os.path.join(self.eval_dir, DECISION_FILE)
        with open(path, 'x') as handle:
            GRIO.dump(obj, handle, default=GRIO.json_default, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self):
        dec = self.decision or {}
        return {
            'schema': SCHEMA, 'declaration': self.decl, 'cell': self.decl['cell'], 'N': self.n, 'cap': self.cap,
            'phase': self.phase, 'certificate_length_original': self.threshold_original,
            'certificate_length_raised_to': CERTIFICATION_DISABLED_THRESHOLD,
            'certificate_length_restored_at_exit': self.threshold_restored,
            'last_cycle': self.cycle, 'cycle_lines_written': self.lines,
            'replay_bitwise_through_cycle': self.replay_equal_through,
            'replay_first_divergence': self.first_divergence,
            'settling_status': dec.get('status'), 'k_star': dec.get('k_star'), 'branch': dec.get('branch'),
            'stopped_by': ('settling_rule' if dec.get('status') == 'certified' else
                           'replay_divergence_abort' if self.first_divergence is not None else
                           'cap' if self.cycle == self.cap else 'other'),
            'lapse_events': list(self.rule.lapses),
            'errors': list(self.errors),
            'ok': (not self.errors and self.phase == 'ended' and self.threshold_restored == self.threshold_original
                   and self.lines == (self.cycle or 0) and self.first_divergence is None
                   and self.replay_equal_through == min(self.n, self.cycle or 0)
                   and self.decision is not None),
            'files': {'cycle_record': CYCLE_FILE, 'blocks_all': BLOCKS_FILE, 'decision': DECISION_FILE},
        }


def _block_rows(blocks):
    rows = []
    for (agent, node_id, year, day), value in blocks.items():
        rows.append({'agent': agent, 'node_id': node_id, 'year': None if year is None else str(year),
                     'day': None if day is None else str(day), 'value': value})
    rows.sort(key=lambda r: (str(r['agent']), str(r['node_id']), str(r['year']), str(r['day'])))
    return rows


def replay_compare(run_values, ref_row):
    """{field: (run_text, ref_text)} for every REPLAY_GATED_FIELD whose JSON text differs, plus the magnitude."""
    diff = {}
    for f in REPLAY_GATED_FIELDS:
        a, b = _jtext(run_values.get(f)), _jtext(ref_row.get(f))
        if a != b:
            diff[f] = (a, b)
    rel = {}
    for f in diff:
        x, y = run_values.get(f), ref_row.get(f)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) \
                and not isinstance(y, bool):
            rel[f] = abs(float(x) - float(y)) / max(abs(float(y)), 1e-300)
    ga, gb = run_values.get('gross_operational_cost'), ref_row.get('gross_operational_cost')
    return diff, {'fields_differing': sorted(diff), 'values': {f: {'run': diff[f][0], 'recorded': diff[f][1]}
                                                              for f in sorted(diff)},
                  'gross_difference_run_minus_recorded': (float(ga) - float(gb)) if (
                      isinstance(ga, (int, float)) and isinstance(gb, (int, float))) else None,
                  'max_relative_difference': max(rel.values()) if rel else None, 'relative_difference_by_field': rel}


def make_wrappers(st, originals, srp_module=None):
    """The seven wrappers over `originals` (name -> callable). Separated from the context manager so the zero-solve
    checks can drive them with stand-in originals and sentinel arguments."""

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W101 settling-continuation hook: {msg}')

    def w_baseline(planning_problem, admm_parameters):
        out = originals['_capture_convergence_depth_tail_baseline'](planning_problem, admm_parameters)
        if st.params is not None:
            raise_('tail baseline captured twice (a second ADMM call inside one continuation run)')
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
        st.cur = {'cycle': cycle, 'phase': 'replay' if cycle <= st.n else 'continuation',
                  't_start_s': time.time() - st.t0}
        if cycle <= st.n:
            st.cur['tail_apply'] = {'hold': False, 'active_passed': bool(active)}
            return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
        st.cur['tail_apply'] = {'hold': True, 'natural_active': bool(active), 'active_passed': True,
                                'hold_changed_value': not bool(active)}
        return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, True, baseline, cycle)

    def w_aa(aa_state, aa_layout, consensus_vars, dual_vars, w_before, rho_channel, boyd_metrics, iter):
        if iter != st.cycle:
            raise_(f'AA step at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        # boyd_k from THIS cycle's boyd_metrics as the AA step receives it (production calls it only when every local
        # solve succeeded, so local_solves_ok is True here)
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
        """Production reads EFC/day once per cycle, AFTER the penalty update and right before the diagnostics row:
        the cycle is finalised here (replay gate, cycle line), once the row's every gated field is known."""
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
        """The settling rule for this cycle; on certification the certificate length goes to 0 (the loop exits at the
        end of THIS cycle). The only place the settling decision is taken."""
        rec = st.rule.observe(st.cycle, q, boyd)
        st.cur['settling'] = rec
        dec = st.rule.decision
        if dec is not None and st.decision is None:
            st.decision = dict(dec)
            st.decision.update({'cell': st.decl['cell'], 'N': st.n, 'cap': st.cap, 'decided_at_cycle': st.cycle,
                                'Q_N': st.gross.get(st.n)})
            if dec['status'] == 'certified':
                st.certified_cycle = st.cycle
                st.params.minimum_consecutive_converged_cycles = SETTLING_CERTIFIED_THRESHOLD
                st.events.append({'event': 'settling_certified', 'cycle': st.cycle, 'branch': dec['branch']})
            else:
                st.events.append({'event': 'settling_uncertified_at_cap', 'cycle': st.cycle})
            st._write_decision(st.decision)
        if rec.get('decision', '') and str(rec.get('decision')).startswith('certified') and st.cycle <= st.n:
            raise_(f'settling rule certified at cycle {st.cycle} <= N {st.n} (impossible by construction)')

    def w_recourse(planning_problem, models):
        rc = originals['_get_operational_recourse_components'](planning_problem, models)
        if st.phase != 'in_cycle' or st.cur is None or 'gross' in st.cur:
            return rc          # outside the loop (terminal captures), or a second call in the cycle
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
        t = time.time()
        blocks = srp_module._get_operational_recourse_block_components(planning_problem, models)
        obj = srp_module._get_operational_objective_component_blocks(planning_problem, models)
        net = rc.get('net_operational_recourse')
        total = sum(blocks.values())
        tol = max(1e-4, 1e-10 * max(abs(net), 1.0)) if net is not None else None
        line = {'cycle': c, 'phase': 'replay' if c <= st.n else 'continuation',
                'gross_operational_cost': rc.get('gross_operational_cost'), 'net_operational_recourse': net,
                'terminal_salvage_value': rc.get('terminal_salvage_value'), 'n_blocks': len(blocks),
                'blocks': _block_rows(blocks), 'block_sum': total,
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
        st.cur['pending_penalties'] = (boyd_metrics, allow_update, result)   # finalised in the EFC wrapper
        return result

    def _finalize_cycle(boyd_metrics, local_solves_ok, penalty_result):
        cur = st.cur
        c = cur['cycle']
        actions, _before, after, _bg, _ag, rfa, _fs = penalty_result
        if 'gross' not in cur:          # a failed cycle: no Q, no AA step -> the rule sees (None, False)
            cur['gross'] = None
            cur['gross_hex'] = None
            cur['step'] = None
            cur['boyd_k'] = False
            _observe(None, False)
        # production's own consecutive count, tracked (cycle_convergence = boyd_all_pass and local_solves_ok)
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
        divergence = None
        ref_row = st.reference.get(c)
        if ref_row is not None:
            diff, detail = replay_compare(run_values, ref_row)
            cur['replay_equal'] = not diff
            if diff:
                divergence = {'cycle': c, **detail}
                st.first_divergence = divergence
                cur['replay_divergence'] = divergence
            elif st.replay_equal_through == c - 1:
                st.replay_equal_through = c
        else:
            cur['replay_equal'] = None
        cur['certificate_length_in_force_at_cycle_end'] = (st.params.minimum_consecutive_converged_cycles
                                                          if st.params is not None else None)
        cur['t_end_s'] = time.time() - st.t0
        cur['finalized'] = True
        st._write(CYCLE_FILE, cur)
        st.lines += 1
        s = cur.get('settling') or {}
        print(f"[W101-SETTLING] cell {st.decl['cell']} cycle {c} ({cur['phase']}) Q={cur.get('gross')!r} "
              f"step={cur.get('step')} boyd_k={cur.get('boyd_k')} pf_primal={run_values['boyd_pf_primal_ratio']} "
              f"replay_equal={cur.get('replay_equal')} k0={s.get('k0')} len_T={s.get('len_T')} P_hat={s.get('P_hat')} "
              f"range={s.get('range')} decision={s.get('decision')} reasons={s.get('reasons')} "
              f"holds={cur['holds']}", flush=True)
        if divergence is not None and st.decl['abort_on_replay_divergence']:
            raise_(f"REPLAY DIVERGED at cycle {c} (fields {divergence['fields_differing']}, gross difference "
                   f"{divergence['gross_difference_run_minus_recorded']!r}, max relative "
                   f"{divergence['max_relative_difference']!r}) -- cell ABORTED (W101: no relabelled continuation)")

    return {'_capture_convergence_depth_tail_baseline': w_baseline, '_apply_convergence_depth_tail': w_apply,
            '_anderson_acceleration_cycle_step': w_aa, '_convergence_depth_tail_next_state': w_next,
            '_update_admm_penalties': w_penalties, '_get_operational_recourse_components': w_recourse,
            '_get_admm_efc_per_day_max': w_efc}


@contextmanager
def settling_continuation_hooks(eval_dir, declaration, holder, cap):
    """Install the wrappers for the run (the harness enters this FIRST, so these wrap production directly and every
    harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    declaration = validate_settling_continuation(declaration)
    for fname in (CYCLE_FILE, BLOCKS_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ContinuationState(declaration, eval_dir, int(cap), reference=load_replay_reference(declaration))
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


def settling_report(decision, q_by_cycle, n):
    """The frozen report definitions (spec v39), per cell, from the decision record and the run's Q: Q_cert_old = Q_N;
    Q_cert_new = Q_k*; s = Q_k* - Q_N (resolution = band_width); settled slack |s|; report-only c = A[-1]/A[-2],
    c A[-1]/(1 - c), mid(band) - Q_N. Uncertified: Q at the cap, the band over the last window, flagged."""
    q_n = q_by_cycle.get(n)
    out = {'status': decision.get('status'), 'Q_cert_old_Q_N': q_n, 'k0': decision.get('k0'), 'T': decision.get('T'),
           'A': decision.get('A'), 'P_hat': decision.get('P_hat'), 'band': decision.get('band'),
           'band_width': decision.get('band_width')}
    if decision.get('status') == 'certified':
        k = decision['k_star']
        out.update({'k_star': k, 'branch': decision['branch'], 'W': decision.get('W'),
                    'window': decision.get('window'), 'range_over_tau': decision.get('range_over_tau'),
                    'Q_cert_new_Q_k_star': q_by_cycle.get(k)})
    else:
        k = decision.get('k_cap')
        out.update({'k_star': None, 'k_cap': k, 'branch': None, 'Q_at_cap': q_by_cycle.get(k),
                    'uncertified_reasons': decision.get('reasons')})
    q_k = q_by_cycle.get(k) if k is not None else None
    s = (q_k - q_n) if (q_k is not None and q_n is not None) else None
    out.update({'s_signed': s, 's_resolution_band_width': decision.get('band_width'),
                'settled_slack_abs_s': abs(s) if s is not None else None})
    A = decision.get('A') or []
    rep = {'c_last_swing_ratio': None, 'c_A_last_over_1_minus_c': None, 'mid_band_minus_Q_N': None}
    if len(A) >= 2 and A[-2]:
        c = A[-1] / A[-2]
        rep['c_last_swing_ratio'] = c
        rep['c_A_last_over_1_minus_c'] = (c * A[-1] / (1.0 - c)) if c < 1.0 else None
    band = decision.get('band')
    if band and q_n is not None:
        rep['mid_band_minus_Q_N'] = (band[0] + band[1]) / 2.0 - q_n
    out['report_only'] = rep
    if out.get('status') == 'certified' and s is not None and decision.get('band_width') is not None:
        out['s_determinate_vs_band'] = abs(s) > decision['band_width']
    if not (isinstance(s, float) and math.isfinite(s)):
        out['s_note'] = 's unavailable (Q_N or Q_k missing)'
    return out
