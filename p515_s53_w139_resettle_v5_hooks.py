"""
P5.15 Addendum 60, Planner task W139 -- the v5 RE-SETTLING CAMPAIGN hooks (the 36 cells of the v4 campaign not yet
certified under v4: cell 3 b_4649234b re-run first, then the 9 D cells, then claim groups 2 and 4 in the v4 order;
frozen stage spec `frozen_s53_resettle_spec_v5_<sha8>.json`, predecessor v4 e11fbc89).

WHAT THIS MODULE IS. The W132 / W137 re-settling machinery, REUSED BY IMPORT (`p515_s53_w132_resettle_v3_hooks`, pinned by
139d1e62, and `p515_s53_w137_resettle_v4_hooks`, pinned by e11fbc89; neither edited): the cell table (W132's, less the
two cells certified under v4), the nine pass-through wrappers (`make_wrappers`: W118's eight + W132's IPOPT-exit
wrapper), the replay-reference loader, the exit-capture source facts, G8 on production's certificate, and the state
(`ResettleStateV3`, subclassed here). What is NEW here is only:
  * the stop rule VERSION 5 (`settling_criterion_v5`: version 4 with the veto reading all_clean_k -- an Acceptable exit
    on a PRIMARY attempt with all four IPOPT metrics within 10x the tail tolerances counts as clean);
  * THE CLEAN CAPTURE, IN-CYCLE (Addendum 60 / W139 item 2): at the exit wrapper (production's
    `_admm_local_solves_succeeded`, called once per cycle after the DSO, TSO and ESSO solves and BEFORE
    `_drain_network_ipopt_solve_records` empties the per-network attempt records), per block:
      - network blocks (48): the cycle's attempt records still held in each `Network.ipopt_solve_records` deque
        (network.py `_append_ipopt_solve_record`: one record per attempt, `attempt` = primary | recovery |
        recovery_tier2, the attempt's own byte range of its IPOPT output file) are READ, NOT CONSUMED; the final
        attempt is the last record (network.py `_run_smopf`: the returned result is the last attempt made); its four
        metrics are parsed from the final summary inside that byte range;
      - ESSO blocks (3): the entries appended this cycle to `SharedEnergyStorageData.esso_complementarity_diagnostics`
        (one per accepted ESSO solve, carrying the log path of the attempt that produced the loaded solution) and to
        `solver_recovery_diagnostics` (one per ESSO solve that attempted a recovery, with its tier); the tier is read
        from both (they must agree) and the metrics from that attempt's own log file;
    written into the cycle line as `exit_clean_by_block` (51 entries: exit class, attempt tier and chain, the four
    metrics, their ratios to the tolerances, clean and the reason) and `all_clean_k`; the rule reads all_clean_k. The
    capture path is asserted BEFORE the first solve (`clean_capture_checklist`, with the tolerance table against
    source) and cross-checked post-run (the campaign's G24 extension, G24b);
  * the declaration schema that routes a cell to this module (the harness dispatches on it: the v5 router branch,
    which also routes the W139 re-targeted extension's declarations -- `hooks_module`);
  * the preconditions checklist restated for the v5 declaration.
The run is exactly W137's (gated replay through k0, the holds after the run's first residual pass keyed on the
version-2 definition, the captures, W132's exit capture); only the rule the recourse wrapper consults differs and the
exit wrapper additionally READS the attempt records and the logs. Nothing the capture reads is read by a solver or by
the ADMM iteration; nothing is consumed (the deques are copied, not popped): the capture cannot change a solve.

Zero solves: nothing here solves or builds a model. Stdlib (+ the W137 / W132 / W118 / W105 / W101 hooks modules,
themselves stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import inspect
import json
import os
import re
from contextlib import contextmanager

import settling_criterion_v5 as SC5
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V
import p515_s53_w137_resettle_v4_hooks as V4

SCHEMA = 'p515_s53_w139_settling_resettle_v5'
DECLARATION_SCHEMA = SCHEMA         # the declaration's 'schema' key: the harness dispatches on it
EXT_DECLARATION_SCHEMA = 'p515_s53_w139_settling_resettle_ext_v5'   # the W139 extension (its own module)
EXT_HOOKS_MODULE = 'p515_s53_w139_resettle_ext_v5_hooks'
OPTION_NAME = V.OPTION_NAME         # 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING RUN v5 (W139, frozen_s53_resettle_spec_v5) -- current production configuration (C2, tight '
         'tail); a gated cell replayed bitwise against its original record through its first residual pass k0 (abort '
         'on divergence), an ungated cell recorded in full; the certifying regime held after the run\'s first residual '
         'pass (AA off, tight tail on, rho frozen); settling rule v5 (reading gamma, window (a); certification vetoed '
         'while a NON-CLEAN cycle lies in the last W cycles the test reads; clean = Optimal, or Acceptable on the '
         'primary attempt with the four IPOPT metrics within 10x the tail tolerances) until it certifies or the cap; '
         'W105 captures, t_sum, the IPOPT exit, attempt tier and final metrics of every block of every cycle')
P_MAX = V.P_MAX
L_MONO = V.L_MONO
CAP_AFTER_N_OLD = V.CAP_AFTER_N_OLD     # 100
CAP_AFTER_K0 = SC5.CAP_AFTER_K0         # 109
UNGATED_CAP_CEILING = V.UNGATED_CAP_CEILING
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = V.N_TSO_BLOCKS, V.N_DSO_BLOCKS, V.N_ESSO
OPTIMAL = SC5.OPTIMAL_CLASS
WRAPPED = V.WRAPPED
CYCLE_FILE = V.CYCLE_FILE
BLOCKS_FILE = V.BLOCKS_FILE
CREEP_FILE = V.CREEP_FILE
ESS_SCHEDULE_FILE = V.ESS_SCHEDULE_FILE
DECISION_FILE = V.DECISION_FILE
SUMMARY_KEY = V.SUMMARY_KEY
FORBIDDEN_DECLARATION_KEYS = V.FORBIDDEN_DECLARATION_KEYS
CERTIFICATION_DISABLED_THRESHOLD = V.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = V.SETTLING_END_THRESHOLD
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the cells: W132's table less the two certified under v4 (Addendum 60: "Cells certified under v4 keep their
#      certificates"); order = the v4 order from cell 3 on (cell 3 b_4649234b first, then the 9 D cells, then groups 2
#      and 4 as before) ------------------------------------------------------------------------------------------------
KEPT_UNDER_V4 = {
    'b_2a0ba8b2': {'v4_run_commit': '903657de', 'k_star': 173, 'status': 'certified',
                   'campaign_root': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w137_resettle_v4',
                                                 'campaign_s53_w137_resettle_v4_b_2a0ba8b2'),
                   'eval_dir': '33447912dab48fff_b_2a0ba8b2', 'eval_key_prefix': '33447912'},
    'b_0dd237f0': {'v4_run_commit': '653e4e5e', 'k_star': 173, 'status': 'certified',
                   'campaign_root': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w137_resettle_v4',
                                                 'campaign_s53_w137_resettle_v4_b_0dd237f0'),
                   'eval_dir': '38d09af2c857755f_b_0dd237f0', 'eval_key_prefix': '38d09af2'},
}
V4_CELL3 = {'cell': 'b_4649234b', 'v4_run_commit': '0734f103', 'status': 'uncertified', 'k_cap': 193,
            'campaign_root': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w137_resettle_v4',
                                          'campaign_s53_w137_resettle_v4_b_4649234b'),
            'eval_dir': 'edc95b9bc4e5e395_b_4649234b', 'eval_key_prefix': 'edc95b9b'}
CELL_ORDER = tuple(c for c in V.CELL_ORDER if c not in KEPT_UNDER_V4)
CELLS = {c: V.CELLS[c] for c in CELL_ORDER}
GROUP_OF_ITEM = V.GROUP_OF_ITEM
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])
DEAD_ZONE_CANDIDATES = V.DEAD_ZONE_CANDIDATES
DEAD_ZONE_BORDERLINE = V.DEAD_ZONE_BORDERLINE
LAST_CELL = CELL_ORDER[-1]
original_eval_dir = V.original_eval_dir
reference_path = V.reference_path
cap_rule = V.cap_rule
spec_cap = V.spec_cap
load_replay_reference = V.load_replay_reference
exit_capture_checklist = V.exit_capture_checklist
exit_by_block = V.exit_by_block
exit_counts = V.exit_counts
block_keys = V.block_keys
_key_text = V._key_text
persistence_check_production_certificate = V4.persistence_check_production_certificate


def settling_rule_declaration():
    return {'module': 'settling_criterion_v5', 'class': 'settling_criterion_v5.SettlingRuleV5', 'version': SC5.VERSION,
            'reading': 'gamma', 'window': 'a', 'reset_on_non_clean': False,
            'veto': ('a branch verdict is vetoed at k while a NON-CLEAN cycle lies in its certifying window: '
                     'oscillatory [k - W + 1, k], monotone [k - L + 1, k]; state unchanged'),
            'veto_reason': SC5.VETO_REASON, 'retry_tier': None,
            'clean': {'rule': ('Optimal on any attempt tier, or Acceptable on the primary attempt with every metric <= '
                               'factor x tolerance'),
                      'factor': SC5.CLEAN_FACTOR, 'metric_table': SC5.METRIC_TABLE, 'tolerances': SC5.TOLERANCES},
            'n_and_dynamic_cap_keyed_on': 'the first residual pass under the version-2 definition',
            'sub_test_reads': 'enumerated in settling_criterion_v5.SUB_TEST_READS; out-of-window reads recorded',
            'tau': SC5.TAU, 'eps0': SC5.EPS0, 'k_excl': SC5.K_EXCL, 'w_min': SC5.W_MIN, 'w_factor': SC5.W_FACTOR,
            'p_max': P_MAX, 'l_mono': L_MONO, 'gap_bound': SC5.GAP_BOUND, 'drift_window': SC5.DRIFT_WINDOW}


def declaration_for(cell):
    """The one valid declaration of a cell: W132's declaration with this schema, label, the v5 rule and the clean
    capture declared."""
    if cell not in CELLS:
        raise KeyError(f'{cell} is not a v5 cell (the cells certified under v4 keep their v4 certificates)')
    out = V.declaration_for(cell)
    out.update({'schema': DECLARATION_SCHEMA, 'label': LABEL, 'settling_rule': settling_rule_declaration()})
    out['captures'] = dict(out['captures'], exit_clean_by_block=True, attempt_tier_and_final_metrics=True)
    return out


def is_v5_declaration(value):
    """A declaration the v5 router branch takes: this module's schema or the W139 extension's."""
    return isinstance(value, dict) and value.get('schema') in (DECLARATION_SCHEMA, EXT_DECLARATION_SCHEMA)


def hooks_module(value):
    """The module that implements a v5-family declaration (the router branch returns it): this module for the main
    schema, the W139 extension module for the extension schema. Stdlib-only (lazy import)."""
    if isinstance(value, dict) and value.get('schema') == EXT_DECLARATION_SCHEMA:
        import importlib
        return importlib.import_module(EXT_HOOKS_MODULE)
    import sys
    return sys.modules[__name__]


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (v5) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (v5) must NOT carry {forbidden} (the settling rule is the only stop before the '
                         f'cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (v5): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (v5).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (v5) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (v5) differs from the W139 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


# ======================================================================================================================
#  THE CLEAN CAPTURE (pure readers; the exit wrapper calls them in-cycle, the post-run gate G24b and the from-records
#  replay call the same parser on the same bytes)
# ======================================================================================================================
_SUMMARY_LINES = ('Objective', 'Dual infeasibility', 'Constraint violation', 'Variable bound violation',
                  'Complementarity', 'Overall NLP error')
_SUMMARY_RE = re.compile(r'^(Objective|Dual infeasibility|Constraint violation|Variable bound violation|Complementarity|'
                         r'Overall NLP error)\.*:\s+(\S+)\s+(\S+)\s*$', re.M)
_ITER_RE = re.compile(r'Number of Iterations\.+:\s*(\d+)')
_EXIT_RE = re.compile(r'EXIT: ([^\n]*)')
SEGMENT_TAIL_BYTES = 65536
ESSO_LOG_RE = re.compile(r'^optim_log_esso_node(?P<node>\d+)_(?P<stamp>init|cycle(?P<cycle>\d{3}))'
                         r'(?P<suffix>_recovery_tier2|_recovery)?(?P<dup>_dup\d+)?\.txt$')
_SUFFIX_TO_ATTEMPT = {None: 'primary', '_recovery': 'recovery', '_recovery_tier2': 'recovery_tier2'}


def _float_or_none(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v


def parse_final_summary(text):
    """The final summary of the LAST IPOPT solve in `text` (the lines after the last 'Number of Iterations'): the
    iterations, the six (scaled, unscaled) pairs, the EXIT text, and the four v5 metrics in METRIC_TABLE's columns
    (`metrics`; None when any is missing). Pure."""
    out = {'iterations': None, 'summary': None, 'exit': None, 'metrics': None, 'parse_reason': None}
    at = text.rfind('Number of Iterations')
    exits = _EXIT_RE.findall(text)
    out['exit'] = exits[-1].strip() if exits else None
    if at < 0:
        out['parse_reason'] = "no 'Number of Iterations' line (no final summary)"
        return out
    tail = text[at:]
    m = _ITER_RE.search(tail)
    out['iterations'] = int(m.group(1)) if m else None
    summ = {}
    for name, scaled, unscaled in _SUMMARY_RE.findall(tail):
        key = name.lower().replace(' ', '_')
        if key not in summ:
            summ[key] = {'scaled': _float_or_none(scaled), 'unscaled': _float_or_none(unscaled)}
    out['summary'] = summ
    metrics = {}
    for metric, row in SC5.METRIC_TABLE.items():
        key = row['log_line'].lower().replace(' ', '_')
        metrics[metric] = (summ.get(key) or {}).get(row['column'])
    if all(isinstance(v, float) for v in metrics.values()):
        out['metrics'] = metrics
    else:
        out['parse_reason'] = f'final summary incomplete: {sorted(k for k, v in metrics.items() if v is None)}'
        out['metrics_partial'] = metrics
    return out


def read_segment_tail(path, lo, hi, tail_bytes=SEGMENT_TAIL_BYTES):
    """The text of bytes [max(lo, hi - tail_bytes), hi) of `path` (a network attempt's own byte range, or a whole ESSO
    log with lo = 0, hi = size); if the tail holds no 'Number of Iterations' and the segment is longer, the whole
    segment [lo, hi)."""
    lo = int(lo or 0)
    hi = int(hi)
    with open(path, 'rb') as handle:
        start = max(lo, hi - tail_bytes)
        handle.seek(start)
        blob = handle.read(hi - start)
        if b'Number of Iterations' not in blob and start > lo:
            handle.seek(lo)
            blob = handle.read(hi - lo)
    return blob.decode(errors='replace')


def _rel(path):
    if not path:
        return path
    ap = os.path.abspath(path)
    return os.path.relpath(ap, REPO) if ap.startswith(REPO + os.sep) else ap


def network_attempt_records(planning_problem):
    """{block key text: [attempt record copies]} for the 48 network blocks: the records still held in each Network's
    `ipopt_solve_records` deque (READ, NOT CONSUMED). Keys as `V._key_text`."""
    import shared_resources_planning as srp
    out = {}
    for label, network_data in srp._convergence_depth_tail_holders(planning_problem):
        for year in network_data.years:
            for day in network_data.days:
                network = network_data.network[year][day]
                records = getattr(network, 'ipopt_solve_records', None)
                chain = [dict(r) for r in list(records or ())]
                key = f'TSO|{year}|{day}' if label == 'TSO' else f'DSO|{int(label[3:])}|{year}|{day}'
                out[key] = chain
    return out


def network_block_capture(chain, exit_class, tail_bytes=SEGMENT_TAIL_BYTES):
    """One network block's v5 entry from its attempt chain of THIS cycle (production's records, in order). Raises on a
    structural gap (no record, first not primary, a repeated / unknown tier, a final record without a log range, the
    log's EXIT class != the result's class, tol in force != the table)."""
    if not chain:
        raise RuntimeError('no attempt record for the block this cycle')
    attempts = [r.get('attempt') for r in chain]
    if attempts[0] != 'primary' or len(set(attempts)) != len(attempts) or any(a not in SC5.ATTEMPT_TIERS for a in attempts):
        raise RuntimeError(f'attempt chain {attempts} is not primary [, recovery [, recovery_tier2]]')
    final = chain[-1]
    lb = final.get('log_bytes')
    path = final.get('log_path')
    parsed = None
    if path and lb and lb[1] is not None:
        parsed = parse_final_summary(read_segment_tail(path, lb[0], lb[1], tail_bytes))
    import p515_s44_campaign_harness as HAR
    log_exit = (parsed or {}).get('exit')
    log_class = HAR.ipopt_exit_class(log_exit) if log_exit is not None else None
    rec_class = HAR.ipopt_exit_class(final.get('exit')) if final.get('exit') is not None else None
    if exit_class is not None and log_class is not None and log_class != exit_class:
        raise RuntimeError(f'final attempt {final.get("attempt")}: log EXIT class {log_class} != result class '
                           f'{exit_class}')
    tol_in_force = final.get('tol_in_force')
    if exit_class is not None and tol_in_force != SC5.TOLERANCES['network']['tol']:
        raise RuntimeError(f'tol in force {tol_in_force} != the table value {SC5.TOLERANCES["network"]["tol"]}')
    metrics = (parsed or {}).get('metrics')
    cls = SC5.classify_block_exit(exit_class, final.get('attempt'), metrics, 'network')
    return {'family': 'network', 'class': exit_class, 'attempt': final.get('attempt'), 'attempts': attempts,
            'metrics': metrics, 'ratios': cls['ratios'], 'max_ratio': cls['max_ratio'],
            'max_ratio_metric': cls['max_ratio_metric'], 'clean': cls['clean'], 'reason': cls['reason'],
            'iterations': (parsed or {}).get('iterations'), 'log_exit': log_exit, 'record_exit': final.get('exit'),
            'record_exit_class': rec_class, 'tol_in_force': tol_in_force,
            'compl_inf_tol_in_force': final.get('compl_inf_tol_in_force'),
            'variable_bound_violation_unscaled': (((parsed or {}).get('summary') or {}).get(
                'variable_bound_violation') or {}).get('unscaled'),
            'log': {'path': _rel(path), 'bytes': list(lb) if lb else None},
            'parse_reason': (parsed or {}).get('parse_reason') if parsed else 'no log byte range recorded'}


def esso_log_attempt(path):
    """(node, stamp, attempt) from an ESSO log file name (shared_energy_storage_data._create_solver's scheme)."""
    m = ESSO_LOG_RE.match(os.path.basename(path or ''))
    if not m:
        return None
    return int(m.group('node')), m.group('stamp'), _SUFFIX_TO_ATTEMPT[m.group('suffix')]


def esso_block_capture(node, cycle, exit_class, comp_entries, recovery_entries):
    """One ESSO block's v5 entry from the entries appended THIS cycle for `node` to the two ESSO lists. Raises on a
    structural gap (an accepted exit without its complementarity entry, more than one entry, a log name that is not this
    node / cycle, the recovery tier and the log suffix disagreeing, the log's EXIT class != the result's class)."""
    comp = [e for e in comp_entries if str(e.get('node_id')) == str(node)]
    recs = [e for e in recovery_entries if str(e.get('node_id')) == str(node)]
    if len(comp) > 1 or len(recs) > 1:
        raise RuntimeError(f'ESSO node {node}: {len(comp)} complementarity / {len(recs)} recovery entries this cycle')
    tier_from_recovery = None
    if recs:
        r = recs[0]
        tier_from_recovery = 'recovery_tier2' if r.get('tier2_attempted') else 'recovery'
    attempt = tier_from_recovery or 'primary'
    path = None
    parsed = None
    if exit_class in (SC5.OPTIMAL_CLASS, SC5.ACCEPTABLE_CLASS):
        if not comp:
            raise RuntimeError(f'ESSO node {node}: accepted exit {exit_class} without its complementarity entry')
        path = comp[0].get('log_path')
        got = esso_log_attempt(path)
        want_stamp = 'init' if cycle == 0 else f'cycle{cycle:03d}'
        if got is None or got[0] != int(node) or got[1] != want_stamp:
            raise RuntimeError(f'ESSO node {node}: log {path} is not this node / cycle ({want_stamp})')
        if got[2] != attempt:
            raise RuntimeError(f'ESSO node {node}: log suffix attempt {got[2]} != recovery entry tier {attempt}')
        size = os.path.getsize(path)
        parsed = parse_final_summary(read_segment_tail(path, 0, size, tail_bytes=size + 1))
        import p515_s44_campaign_harness as HAR
        log_exit = parsed.get('exit')
        if log_exit is not None and HAR.ipopt_exit_class(log_exit) != exit_class:
            raise RuntimeError(f'ESSO node {node}: log EXIT {log_exit!r} class != result class {exit_class}')
    elif recs:
        r = recs[0]
        path = r.get('tier2_log') if r.get('tier2_attempted') else r.get('recovery_log')
    metrics = (parsed or {}).get('metrics')
    cls = SC5.classify_block_exit(exit_class, attempt, metrics, 'esso')
    return {'family': 'esso', 'class': exit_class, 'attempt': attempt,
            'attempts': (['primary'] + (['recovery'] if recs else []) + (['recovery_tier2'] if recs and
                                                                         recs[0].get('tier2_attempted') else [])),
            'metrics': metrics, 'ratios': cls['ratios'], 'max_ratio': cls['max_ratio'],
            'max_ratio_metric': cls['max_ratio_metric'], 'clean': cls['clean'], 'reason': cls['reason'],
            'iterations': (parsed or {}).get('iterations'), 'log_exit': (parsed or {}).get('exit'),
            'variable_bound_violation_unscaled': (((parsed or {}).get('summary') or {}).get(
                'variable_bound_violation') or {}).get('unscaled'),
            'log': {'path': _rel(path), 'bytes': None},
            'parse_reason': (parsed or {}).get('parse_reason') if parsed else (
                'final result not accepted: metrics not needed' if exit_class not in (SC5.OPTIMAL_CLASS,
                                                                                      SC5.ACCEPTABLE_CLASS) else None)}


def esso_new_entries(cursor, planning_problem):
    """The entries appended to the two ESSO lists since the previous call (`cursor` = {name: (id(list), len)}, updated
    in place; a list replaced by production -- `solver_recovery_diagnostics = list()` before the initialisation --
    restarts at 0)."""
    sed = planning_problem.shared_ess_data
    out = {}
    for name, lst in (('complementarity', sed.esso_complementarity_diagnostics),
                      ('recovery', sed.solver_recovery_diagnostics)):
        prev_id, prev_len = cursor.get(name, (None, 0))
        start = prev_len if (prev_id == id(lst) and prev_len <= len(lst)) else 0
        out[name] = [dict(x) for x in lst[start:]]
        cursor[name] = (id(lst), len(lst))
    return out


def clean_by_block(planning_problem, by_block, cycle, cursor, net_chains=None):
    """{block: v5 entry} for the 51 blocks of this cycle (the order of `by_block`, production's). `by_block` is W132's
    exit capture of the same call (the classes); `net_chains` the network attempt records (default: read now)."""
    net_chains = network_attempt_records(planning_problem) if net_chains is None else net_chains
    esso = esso_new_entries(cursor, planning_problem)
    out = {}
    for key, v in by_block.items():
        kind = key.split('|')[0]
        try:
            if kind in ('TSO', 'DSO'):
                out[key] = network_block_capture(net_chains.get(key) or [], v['class'])
            else:
                out[key] = esso_block_capture(int(key.split('|')[1]), cycle, v['class'], esso['complementarity'],
                                              esso['recovery'])
        except (RuntimeError, OSError, ValueError) as error:
            raise RuntimeError(f'cycle {cycle} {key}: clean capture: {error}') from error
    return out, {'n_esso_complementarity_entries': len(esso['complementarity']),
                 'n_esso_recovery_entries': len(esso['recovery'])}


def tolerance_state_in_memory(planning_problem):
    """The solver options the planning object holds NOW, against the table (called at the initialisation check, before
    the first cycle): network tol == the table tol, no dual_inf_tol / constr_viol_tol passed on any holder (options or
    recovery_options), ESSO tol == the ESSO override, no ESSO compl_inf_tol / dual_inf_tol / constr_viol_tol passed."""
    import shared_resources_planning as srp
    import shared_energy_storage_data as sed
    parts = {}
    for label, nd in srp._convergence_depth_tail_holders(planning_problem):
        sp = nd.params.solver_params
        opts = dict(sp.options or {})
        rec = dict(sp.recovery_options or {})
        parts[f'{label}_tol_is_table'] = opts.get('tol') == SC5.TOLERANCES['network']['tol']
        parts[f'{label}_no_dual_inf_or_constr_viol_tol_passed'] = not any(
            k in d for d in (opts, rec) for k in ('dual_inf_tol', 'constr_viol_tol'))
        parts[f'{label}_recovery_does_not_pass_tol_or_compl_inf_tol'] = not any(k in rec for k in ('tol',
                                                                                                    'compl_inf_tol'))
    ep = planning_problem.shared_ess_data.params.solver_params
    eopts = dict(ep.options or {})
    erec = dict(ep.recovery_options or {})
    over = dict(sed.ESSO_TOL_OVERRIDES)
    parts['esso_tol_after_override_is_table'] = over.get('tol') == SC5.TOLERANCES['esso']['tol']
    parts['esso_no_compl_dual_constr_tol_passed'] = not any(
        k in d for d in (eopts, erec, over) for k in ('compl_inf_tol', 'dual_inf_tol', 'constr_viol_tol'))
    return all(parts.values()), parts


# ---- the source facts the clean capture relies on (asserted BEFORE the first solve) -----------------------------------
CLEAN_SOURCE_FACTS = {
    'loop_local_check_before_the_drain': None,           # computed (order in the loop source)
    'network_record_per_attempt': ('network', '_run_smopf_solver_attempt', (
        'log_offset = _ipopt_log_size(solver_log_path)',
        '_append_ipopt_solve_record(network, params, solver, solver_log_path, log_offset, log_suffix, from_warm_start)')),
    'network_record_fields': ('network', '_append_ipopt_solve_record', (
        "'attempt': log_suffix or 'primary'", "record['log_bytes'] = [log_offset, end]", 'records.append(record)',
        'fields = parse_ipopt_attempt_segment(text, passed)')),
    'network_record_tol_in_force': ('network', 'parse_ipopt_attempt_segment', (
        "record['tol_in_force'] = tol", "record['compl_inf_tol_in_force'] = cit")),
    'network_attempt_order': ('network', '_run_smopf', (
        "primary_result, primary_log_path = _run_smopf_solver_attempt(network, model, params, from_warm_start=from_warm_start)",
        "log_suffix='recovery')", "log_suffix='recovery_tier2')")),
    'network_log_appended_per_network': ('network', '_create_smopf_solver', ("solver.options['file_append'] = 'yes'",)),
    'drain_empties_the_deques': ('shared_resources_planning', '_drain_network_ipopt_solve_records', (
        'record = records.popleft()',)),
    'esso_sinks_passed': ('shared_energy_storage_data', 'SharedEnergyStorageData.optimize', (
        'diagnostic_sink=self.solver_recovery_diagnostics',
        'complementarity_diagnostics_sink=self.esso_complementarity_diagnostics', 'cycle=cycle')),
    'esso_complementarity_entry_is_the_loaded_attempt': ('shared_energy_storage_data', '_optimize', (
        'if tier2_attempted and tier2_result is not None:', 'diagnostics_log_path = tier2_log_path',
        'diagnostics_log_path = recovery_log_path', 'diagnostics_log_path = primary_log_path',
        'complementarity_diagnostics_sink.append(complementarity_diagnostics)',
        "'tier': 'tier2' if tier2_attempted else 'tier1'", "'tier2_attempted': tier2_attempted")),
    'esso_diagnostics_carry_the_log_path': ('shared_energy_storage_data', '_get_esso_complementarity_diagnostics', (
        "'log_path': log_path", "'node_id': node_id")),
    'esso_log_name_scheme': ('shared_energy_storage_data', '_create_solver', (
        "stamp = f'cycle{cycle:03d}' if isinstance(cycle, int) else 'init'",
        "f'optim_log_esso_node{node_id}_{stamp}.txt' if node_id is not None",
        "options['output_file'] = f'{path_stem}_{log_suffix}{path_extension}'", "options['file_append'] = 'no'")),
    'esso_update_passes_the_cycle': ('shared_resources_planning', 'update_shared_energy_storages_coordination_model_and_solve',
                                     ('res = shared_ess_data.optimize(models, from_warm_start=from_warm_start, cycle=cycle)',)),
    'esso_tol_override': ('shared_energy_storage_data', 'SharedEnergyStorageData.optimize', (
        'option_overrides=ESSO_TOL_OVERRIDES',)),
}


def _src(mod_name, qual):
    import importlib
    obj = importlib.import_module(mod_name)
    for part in qual.split('.'):
        obj = getattr(obj, part, None)
    return inspect.getsource(obj) if obj is not None else ''


def case_file_tolerances():
    """The tolerance table from the committed case files (the options production passes): every network params file
    SRP1.json names, the ESS params file, ESSO_TOL_OVERRIDES. Returns (ok, detail). Parent-safe (json + the ESSO
    module's constant)."""
    import shared_energy_storage_data as sed
    base = os.path.join(REPO, 'data', 'SRP1')
    top = json.load(open(os.path.join(base, 'SRP1.json')))
    files = {'TSO': os.path.join(top['TransmissionNetwork']['name'], top['TransmissionNetwork']['params_file'])}
    for d in top['DistributionNetworks']:
        files[f"DSO{d['connection_node_id']}"] = os.path.join(d['name'], d['params_file'])
    parts, detail = {}, {}
    for label, rel in files.items():
        s = json.load(open(os.path.join(base, rel)))['solver']
        opts, rec = s.get('options') or {}, s.get('recovery_options') or {}
        detail[label] = {'file': os.path.join('data', 'SRP1', rel), 'tol': opts.get('tol'),
                         'compl_inf_tol_production': opts.get('compl_inf_tol'),
                         'recovery_options': rec}
        parts[f'{label}_tol_{SC5.TOLERANCES["network"]["tol"]}'] = opts.get('tol') == SC5.TOLERANCES['network']['tol']
        parts[f'{label}_dual_inf_tol_constr_viol_tol_not_passed'] = not any(
            k in d for d in (opts, rec) for k in ('dual_inf_tol', 'constr_viol_tol'))
        parts[f'{label}_recovery_does_not_pass_tol_or_compl_inf_tol'] = not any(k in rec for k in ('tol',
                                                                                                    'compl_inf_tol'))
    ess_rel = os.path.join('SharedESS', top['SharedEnergyStorage']['params_file'])
    s = json.load(open(os.path.join(base, ess_rel)))['solver']
    eopts, erec = s.get('options') or {}, s.get('recovery_options') or {}
    over = dict(sed.ESSO_TOL_OVERRIDES)
    detail['ESSO'] = {'file': os.path.join('data', 'SRP1', ess_rel), 'tol_case_file': eopts.get('tol'),
                      'ESSO_TOL_OVERRIDES': over, 'recovery_options': erec}
    parts['ESSO_tol_after_override_is_table'] = over.get('tol') == SC5.TOLERANCES['esso']['tol']
    parts['ESSO_compl_dual_constr_tol_not_passed'] = not any(
        k in d for d in (eopts, erec, over) for k in ('compl_inf_tol', 'dual_inf_tol', 'constr_viol_tol'))
    parts['table_dual_inf_tol_is_ipopt_default_1'] = (SC5.TOLERANCES['network']['dual_inf_tol']
                                                      == SC5.TOLERANCES['esso']['dual_inf_tol'] == 1.0)
    parts['table_constr_viol_tol_is_ipopt_default_1e-4'] = (SC5.TOLERANCES['network']['constr_viol_tol']
                                                           == SC5.TOLERANCES['esso']['constr_viol_tol'] == 1e-4)
    parts['table_esso_compl_inf_tol_is_ipopt_default_1e-4'] = SC5.TOLERANCES['esso']['compl_inf_tol'] == 1e-4
    return all(parts.values()), {'parts': parts, 'files': detail}


def clean_capture_checklist(tail_compl_inf_tol=None):
    """The source facts and the tolerance table the clean capture relies on. Returns {name: bool}; raises nothing.
    `tail_compl_inf_tol`: the spec's convergence_depth_tail compl_inf_tol (must equal the table's network value)."""
    import shared_resources_planning as srp
    import network as net
    out = {}
    for name, fact in CLEAN_SOURCE_FACTS.items():
        if fact is None:
            continue
        mod, qual, snippets = fact
        try:
            src = _src(mod, qual)
        except Exception:  # noqa: BLE001
            src = ''
        out[f'clean_source:{name}'] = bool(src) and all(s in src for s in snippets)
    loop_src = inspect.getsource(srp._run_operational_planning)
    loop_at = loop_src.find('for iter in range(1, admm_parameters.num_max_iters + 1):')
    i_esso = loop_src.find("results['esso'] = update_shared_energy_storages_coordination_model_and_solve(", loop_at)
    i_check = loop_src.find('local_solves_ok = _admm_local_solves_succeeded(planning_problem, results)', loop_at)
    i_drain = loop_src.find('network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, '
                            'iter))', loop_at)
    out['clean_source:loop_esso_then_local_check_then_drain'] = 0 <= loop_at < i_esso < i_check < i_drain
    out['clean_source:loop_one_drain_per_cycle'] = (loop_src.count('_drain_network_ipopt_solve_records(planning_problem, '
                                                                   'iter)') == 1)
    out['clean_source:no_drain_between_cycle_start_and_the_check'] = (
        '_drain_network_ipopt_solve_records' not in loop_src[loop_at:i_check])
    out['clean_source:deque_holds_three_attempts'] = net.IPOPT_SOLVE_RECORDS_MAXLEN >= 3
    ok, tol = case_file_tolerances()
    for k, v in tol['parts'].items():
        out[f'tolerance_table:{k}'] = bool(v)
    if tail_compl_inf_tol is not None:
        out['tolerance_table:network_compl_inf_tol_is_the_spec_tail_value'] = (
            tail_compl_inf_tol == SC5.TOLERANCES['network']['compl_inf_tol'])
    out['classifier:unit_cases'] = _classifier_unit_cases()
    return out


def _classifier_unit_cases():
    t = SC5.TOLERANCES['network']
    at = {'overall_nlp_error': 2.0 * t['tol'], 'dual_infeasibility': 0.01, 'constraint_violation': 1e-13,
          'complementarity': 2.51 * t['compl_inf_tol']}
    over = dict(at, complementarity=11.0 * t['compl_inf_tol'])
    return (SC5.classify_block_exit('acceptable', 'primary', at, 'network')['clean'] is True
            and SC5.classify_block_exit('acceptable', 'primary', over, 'network')['clean'] is False
            and SC5.classify_block_exit('acceptable', 'recovery', at, 'network')['clean'] is False
            and SC5.classify_block_exit('optimal', 'recovery', None, 'network')['clean'] is True
            and SC5.classify_block_exit('other', 'primary', at, 'network')['clean'] is False)


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve) -- W137's items restated for the v5 declaration + the clean
#  capture
# ======================================================================================================================
PRODUCTION_SIGNATURES = V4.PRODUCTION_SIGNATURES


def rule_declaration_checks(rule):
    """The declared rule is this module's v5 declaration and its constants are settling_criterion_v5's. Pure."""
    return {
        'f_rule_is_the_v5_declaration': rule == settling_rule_declaration(),
        'f_rule_constants_equal_settling_criterion_v5': (rule['tau'] == SC5.TAU and rule['eps0'] == SC5.TAU / 100.0
                                                         and rule['gap_bound'] == SC5.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60
                                                         and rule['version'] == 5 and rule['retry_tier'] is None
                                                         and rule['reset_on_non_clean'] is False
                                                         and rule['veto_reason'] == SC5.VETO_REASON
                                                         and rule['clean']['factor'] == 10.0),
        'f_rule_class_callable': callable(getattr(SC5, 'SettlingRuleV5', None)),
    }


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): W137's items restated for the v5
    declaration, plus the clean-capture facts and the tolerance table (`clean_capture_checklist`). Raises on any
    failure; returns the checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    decl = validate_settling_resettle(decl)
    cell = decl['cell']
    cap = int(spec['cap'])
    loop_src = inspect.getsource(srp._run_operational_planning)
    step_src = inspect.getsource(aam.AndersonAccelerationState.step)
    boyd_src = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    pos = [loop_src.find(s) for s in C101.SOURCE_ORDER]
    cr = decl['cap_rule']
    tail_cfg = ((spec or {}).get('configuration') or {}).get('convergence_depth_tail') or {}
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
            V._certificate_length_writes_in_source() == sorted(R.CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(C101.CERTIFICATE_LENGTH_READS)),
        'd_spec_cap_equals_the_cell_cap_within_its_ceiling': (cap == spec_cap(cell) and cap <= cr['ceiling']
                                                              and cr['ceiling'] == CELLS[cell]['cap_ceiling']),
        **rule_declaration_checks(decl['settling_rule']),
        'tail_enabled_for_this_run': bool((tail_checklist or {}).get('tail_enabled_for_this_run')),
        'aa_off_literal_is_production': (srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == R.AA_OFF_ACTION
                                         and repr(R.AA_OFF_ACTION)[1:-1] in step_src),
        'loop_exit_is_the_convergence_break': 'if convergence:\n            print(f"[INFO] \\t - ADMM converged' in loop_src,
        'block_functions_callable': all(callable(getattr(srp, nm, None)) for nm in (
            '_get_operational_recourse_block_components', '_get_operational_objective_component_blocks')),
        'efc_read_once_per_cycle_after_penalties_before_the_row': (
            loop_src.count('_get_admm_efc_per_day_max(esso_model)') == 1
            and 0 <= loop_src.find('= _update_admm_penalties(') < loop_src.find('_get_admm_efc_per_day_max(esso_model)')
            < loop_src.find('admm_diagnostics.append({')),
        'state_carries_every_w118_state_attribute': _state_attribute_superset(),
        'state_class_is_the_v5_state_on_the_w132_state': issubclass(ResettleStateV5, V.ResettleStateV3),
        'rule_class_is_the_v5_adapter': issubclass(HookedRuleV5, SC5.SettlingRuleV5),
        'declaration_captures_exit_clean_by_block': decl['captures'].get('exit_clean_by_block') is True,
    }
    for name, expected in PRODUCTION_SIGNATURES.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    if decl['replay_reference'] is not None:
        try:
            load_replay_reference(decl)
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = True
        except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in E105.capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    for k, v in R.t_sum_capture_checklist().items():
        checks[f't_sum:{k}'] = bool(v)
    for k, v in exit_capture_checklist().items():
        checks[f'exit:{k}'] = bool(v)
    for k, v in clean_capture_checklist(tail_cfg.get('compl_inf_tol') if tail_cfg else None).items():
        checks[f'clean:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W139 settling-resettle v5 preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the v5 rule adapter, the exit wrapper and the state (W132's, subclassed)
# ======================================================================================================================
_MISSING = object()


class HookedRuleV5(SC5.SettlingRuleV5):
    """The v5 rule as W118's wrappers call it: `observe(k, q, boyd, t_sum)` reads all_clean_k from the state (the exit
    wrapper's clean capture of cycle k); a cycle whose exits were not captured RAISES (fail loudly)."""

    def bind(self, state):
        self._state = state
        return self

    def observe(self, k, q, boyd, t_sum, all_clean=_MISSING):
        if all_clean is _MISSING:
            got = self._state.all_clean.get(k, _MISSING)
            if got is _MISSING:
                self._state.errors.append(f'cycle {k}: all_clean_k not captured before the rule')
                raise RuntimeError(f'W139 settling-resettle v5 hook: cycle {k}: all_clean_k not captured (the local-'
                                   f'solve wrapper did not run this cycle) -- the rule cannot decide')
            all_clean = got
        return super().observe(k, q, boyd, t_sum, all_clean)


def _v5_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return HookedRuleV5(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return HookedRuleV5(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


class ResettleStateV5(V.ResettleStateV3):
    """W132's v3 state (constructor, exit capture, W118's attributes) with the v5 rule and the clean capture: the
    constructor is W132's (it mirrors W118's attribute by attribute; `_state_attribute_superset` asserts it before any
    solve), then the rule is replaced by the v5 adapter bound to this state BEFORE any cycle is observed. The summary is
    W118's plus W132's exit capture and the v5 fields."""

    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        super().__init__(decl, eval_dir, cap, reference=reference, sink=sink)
        self.rule = _v5_rule(decl).bind(self)
        self.all_clean = {}
        self.esso_cursor = {}
        self.init_clean = None
        self.tolerance_in_memory = None
        self.clean_counts = {'non_clean_blocks': 0, 'acceptable_clean_blocks': 0}

    def summary(self):
        s = R.ResettleState.summary(self)
        rule = self.rule
        n_cycles = self.cycle or 0
        exit_ok = (sorted(self.all_optimal) == list(range(1, n_cycles + 1)) and self.init_exit is not None
                   and self.local_check_calls == n_cycles + 1)
        clean_ok = sorted(self.all_clean) == list(range(1, n_cycles + 1))
        tol_ok = bool((self.tolerance_in_memory or {}).get('ok'))
        s.update({
            'schema': SCHEMA, 'criterion_version': SC5.VERSION, 'reading': SC5.READING,
            'first_k0_v2': rule.n, 'non_clean_cycles': list(rule.non_clean_cycles),
            'non_optimal_cycles': sorted(k for k, v in self.all_optimal.items() if not v),
            'vetoes': [dict(v) for v in rule.vetoes], 'n_vetoes': len(rule.vetoes),
            'n_cycles_exit_captured': len(self.all_optimal), 'n_cycles_clean_captured': len(self.all_clean),
            'local_check_calls': self.local_check_calls,
            'exit_capture_init_round_0': self.init_exit, 'exit_classifier': self.exit_classifier,
            'exit_capture_complete': exit_ok, 'clean_capture_complete': clean_ok,
            'clean_capture_init_round_0': self.init_clean, 'tolerance_state_in_memory_at_init': self.tolerance_in_memory,
            'clean_counts': dict(self.clean_counts), 'cap_ceiling': self.decl['cap_rule']['ceiling'],
            'state_class': ('p515_s53_w139_resettle_v5_hooks.ResettleStateV5 (on p515_s53_w132_resettle_v3_hooks.'
                            'ResettleStateV3, reused by import)'),
        })
        s['ok'] = bool(s['ok'] and exit_ok and clean_ok and tol_ok)
        return s


def _state_attribute_superset():
    """Every attribute a fresh W118 state carries is carried by a fresh v5 state, and the v5 state's rule is the v5
    adapter bound to it (drift guard for the reused constructor)."""
    w118 = R.ResettleState(R.declaration_for(R.CELL_ORDER[2]), None, R.spec_cap(R.CELL_ORDER[2]), reference={}, sink=[])
    cell = CELL_ORDER[0]
    v5 = ResettleStateV5(declaration_for(cell), None, spec_cap(cell), reference={}, sink=[])
    return (set(vars(w118)) <= set(vars(v5)) and isinstance(v5.rule, HookedRuleV5)
            and getattr(v5.rule, '_state', None) is v5)


def make_exit_wrapper(st, original, exit_class, classifier_label=None, net_chain_reader=None, tolerance_reader=None):
    """The ninth wrapper, v5: W132's exit wrapper (its value returned UNCHANGED; the 51 exits, all_optimal_k) and then,
    in-cycle, the clean capture of the 51 blocks (`exit_clean_by_block`, `all_clean_k`, `non_clean_blocks` into the
    cycle line; st.all_clean). At the initialisation call the network records are already drained by production (report:
    unavailable), the ESSO lists' cursor is set and the solver options in memory are checked against the table (a
    mismatch RAISES before the first cycle). `net_chain_reader` / `tolerance_reader`: stand-ins for the zero-solve
    checks (default: the real readers)."""
    w3 = V.make_exit_wrapper(st, original, exit_class, classifier_label)
    read_chains = net_chain_reader or network_attempt_records
    read_tol = tolerance_reader or tolerance_state_in_memory

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W139 settling-resettle v5 hook: {msg}')

    def w_local(planning_problem, results):
        was_init = st.phase == 'before' and st.cycle is None
        ok = w3(planning_problem, results)
        if was_init:
            tok, tparts = read_tol(planning_problem)
            st.tolerance_in_memory = {'ok': bool(tok), 'parts': tparts}
            if not tok:
                raise_(f'the solver options in memory disagree with the tolerance table: '
                       f'{sorted(k for k, v in tparts.items() if not v)}')
            esso = esso_new_entries(st.esso_cursor, planning_problem)
            st.init_clean = {'round': 0, 'network': ('not captured: production drains the network attempt records '
                                                     'before the initialisation check (report only)'),
                             'n_esso_complementarity_entries': len(esso['complementarity']),
                             'n_esso_recovery_entries': len(esso['recovery'])}
            return ok
        by_block = st.cur['ipopt_exit_by_block']
        try:
            entries, meta = clean_by_block(planning_problem, by_block, st.cycle, st.esso_cursor,
                                           net_chains=read_chains(planning_problem))
        except RuntimeError as error:
            raise_(str(error))
        if list(entries) != list(by_block):
            raise_(f'cycle {st.cycle}: clean capture keys differ from the exit capture')
        all_clean = all(v['clean'] for v in entries.values())
        st.cur['exit_clean_by_block'] = entries
        st.cur['all_clean_k'] = all_clean
        st.cur['non_clean_blocks'] = sorted(k for k, v in entries.items() if not v['clean'])
        st.cur['acceptable_clean_blocks'] = sorted(k for k, v in entries.items()
                                                   if v['reason'] == 'acceptable_primary_within_factor')
        st.cur['clean_capture_meta'] = meta
        st.clean_counts['non_clean_blocks'] += len(st.cur['non_clean_blocks'])
        st.clean_counts['acceptable_clean_blocks'] += len(st.cur['acceptable_clean_blocks'])
        st.all_clean[st.cycle] = all_clean
        return ok

    return w_local


def make_wrappers(st, originals, exit_class, srp_module=None, classifier_label=None, net_chain_reader=None,
                  tolerance_reader=None):
    """The nine wrappers over `originals`: W118's eight (`R.make_wrappers`, unchanged) and the v5 exit wrapper (W132's
    with the clean capture). Separated from the context manager so the zero-solve checks can drive them."""
    wrappers = R.make_wrappers(st, {n: originals[n] for n in R.WRAPPED}, srp_module=srp_module)
    wrappers['_admm_local_solves_succeeded'] = make_exit_wrapper(
        st, originals['_admm_local_solves_succeeded'], exit_class, classifier_label, net_chain_reader=net_chain_reader,
        tolerance_reader=tolerance_reader)
    return wrappers


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine wrappers for the run on the v5 state (the harness enters this FIRST, so these wrap production
    directly and every harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle v5: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateV5(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
                             classifier_label=f'p515_s44_campaign_harness.ipopt_exit_class ({HAR.__file__})')
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()
