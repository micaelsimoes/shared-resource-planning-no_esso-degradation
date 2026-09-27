"""Per-cycle interface consensus duals (lambda_t) -- a WRITE-ONLY default record of the campaign harness
(P5.15 Addendum 53 "Records"; Planner task W101).

WHAT IS RECORDED. Every cycle, the interface power-flow consensus duals exactly as production holds them in
`dual_vars['pf'][agent]['current'][node_id][year][day][kind][period]` (agent 'tso' / 'dso', kind 'p' / 'q'; the leaf is
the list `[0.0] * num_instants` built by `create_admm_variables`), read immediately after the cycle's DSO -> TSO -> ESSO
dual updates -- i.e. at production's once-per-cycle `get_admm_boyd_residual_metrics(...)` call, which receives
`dual_vars` after the last update of the cycle and BEFORE the Anderson step (an accepted AA step may overwrite
`dual_vars` afterwards; with AA held off, as in every post-certification continuation cycle, the two coincide). The
n-th call of that function is cycle n: production calls it exactly once per cycle and never at initialisation
(asserted from source before any solve, `assert_capture_path`, the precedent of W64's per-cycle response capture).

SCALING METADATA (recorded once, in the sidecar's header line; nothing is converted here). On the interface-P channel
only sigma and the interface rating scale lambda_t (TASKS.md, Addendum 49 lambda_t ruling and its W93 correction;
`uncoordinated_benchmark.interface_price_terms` holds the conversion): per block (node, year, day) the interface branch
rating (MVA, `get_interface_branch_rating()` of the DN at that year/day), the TSO and DSO base MVA, and each side's
`admm_objective_scale` Param (sigma / block weight; a NON-mutable Param, so it cannot change during the run -- asserted);
plus `admm_parameters.objective_scale` (sigma_fixed). The dual update is
`dual += rho_pf * (E - z) / interface_rating * s_base` (production, `update_admm_consensus_variables`).

SIDECAR `INTERFACE_DUAL_FILE` (JSONL; separate so `per_cycle_record.jsonl` lines stay small): line 1 = the header
(schema, block order, periods, metadata); then one line per cycle {'cycle', 'captured', 'lambda_pf_p_tso',
'lambda_pf_p_dso', 'lambda_pf_q_tso', 'lambda_pf_q_dso'} with each array ordered as the header's `blocks` (one list of
`n_periods` floats per block). Floats are written by `json` repr (shortest round-trip, i.e. exact). Every write goes
through `gate_result_io` (W100).

WRITE-ONLY. The wrapper calls production's function with the SAME argument objects and returns its return value
unchanged; the capture only READS (dict/list indexing, `pyomo.value` on Params, two network read methods) and copies
floats into new lists. A capture error is recorded in the line and never raised. Stdlib only at import.
"""

import os
import time
from contextlib import contextmanager

import gate_result_io as GRIO

SCHEMA = 'interface_dual_capture_v1'
INTERFACE_DUAL_FILE = 'interface_duals_per_cycle.jsonl'
SUMMARY_KEY = 'interface_dual_capture'
WRAPPED = ('get_admm_boyd_residual_metrics',)
CHANNEL_FIELDS = (('pf', 'tso', 'p', 'lambda_pf_p_tso'), ('pf', 'dso', 'p', 'lambda_pf_p_dso'),
                  ('pf', 'tso', 'q', 'lambda_pf_q_tso'), ('pf', 'dso', 'q', 'lambda_pf_q_dso'))
# Source facts asserted before any solve (`assert_capture_path`).
SOURCE_BOYD_CALL = ('boyd_metrics = get_admm_boyd_residual_metrics(planning_problem, tso_model, dso_models, esso_model, '
                    'consensus_vars, dual_vars, admm_parameters)')
SOURCE_DUAL_LEAF = "dual_variables['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}"
SOURCE_DUAL_UPDATE = ("dual_vars['pf']['tso']['current'][node_id][year][day]['p'][p] += rho_pf_tso * error_p_pf_req_tso "
                      "/ interface_rating * tso_s_base")


def block_order(planning_problem):
    """[(node_id, year, day)] in production's own iteration order (active DN nodes, planning years, planning days)."""
    return [(node_id, year, day) for node_id in planning_problem.active_distribution_network_nodes
            for year in planning_problem.years for day in planning_problem.days]


def read_duals(dual_vars, blocks):
    """The four interface-PF dual arrays (new lists of Python floats), one list per block. Pure read."""
    out = {}
    for group, agent, kind, field in CHANNEL_FIELDS:
        cur = dual_vars[group][agent]['current']
        out[field] = [[float(v) for v in cur[node_id][year][day][kind]] for node_id, year, day in blocks]
    return out


def read_metadata(planning_problem, tso_model, dso_models, admm_parameters, blocks):
    """The scaling metadata per block (pure reads). `admm_objective_scale` is read with pyomo's value(); its
    mutability is recorded (a non-mutable Param cannot change)."""
    import pyomo.environ as pe
    tn = planning_problem.transmission_network
    per_block = []
    for node_id, year, day in blocks:
        dnet = planning_problem.distribution_networks[node_id].network[year][day]
        tnet = tn.network[year][day]
        t_block = tso_model[year][day]
        d_block = dso_models[node_id][year][day]
        entry = {'node_id': node_id, 'year': year, 'day': day,
                 'interface_rating_mva': float(dnet.get_interface_branch_rating()),
                 'tso_base_mva': float(tnet.baseMVA), 'dso_base_mva': float(dnet.baseMVA),
                 'admm_objective_scale_tso': None, 'admm_objective_scale_dso': None,
                 'admm_objective_scale_tso_mutable': None, 'admm_objective_scale_dso_mutable': None}
        for side, blk in (('tso', t_block), ('dso', d_block)):
            param = getattr(blk, 'admm_objective_scale', None)
            if param is not None:
                entry[f'admm_objective_scale_{side}'] = float(pe.value(param))
                entry[f'admm_objective_scale_{side}_mutable'] = bool(param.mutable)
        per_block.append(entry)
    return {'sigma_fixed_admm_parameters_objective_scale': getattr(admm_parameters, 'objective_scale', None),
            'per_block': per_block,
            'conversion_note': ('not converted here. Interface-P channel: only sigma (through admm_objective_scale = '
                                'sigma / block weight) and the interface rating apply; see '
                                'uncoordinated_benchmark.interface_price_terms (lambda_dso_linear = pi + eff_dso * '
                                'dual_pf_p_req / (r_pu * B_dn), r_pu = rating / B). The dual_vars value is the model '
                                'Param dual_pf_p_req of the NEXT cycle\'s solve.')}


def assert_capture_path():
    """Rule eleven, BEFORE any solve (child): the capture's source facts hold in production. Raises on failure;
    returns the checklist."""
    import inspect
    import shared_resources_planning as srp
    run_src = inspect.getsource(srp._run_operational_planning)
    create_src = inspect.getsource(srp.create_admm_variables)
    boyd_at = run_src.find(SOURCE_BOYD_CALL)
    loop_at = run_src.find('for iter in range(1, admm_parameters.num_max_iters + 1):')
    aa_at = run_src.find('aa_record = _anderson_acceleration_cycle_step(')
    checks = {
        'boyd_metrics_call_once_in_run_source': run_src.count('get_admm_boyd_residual_metrics(') == 1,
        'boyd_metrics_call_inside_the_cycle_loop': 0 <= loop_at < boyd_at,
        'boyd_metrics_call_before_the_aa_step': 0 <= boyd_at < aa_at,
        'boyd_metrics_signature': list(inspect.signature(srp.get_admm_boyd_residual_metrics).parameters) == [
            'planning_problem', 'tso_model', 'dso_models', 'esso_model', 'consensus_vars', 'dual_vars',
            'admm_parameters'],
        'dual_leaf_is_p_q_period_lists': SOURCE_DUAL_LEAF in create_src,
        'dual_update_scaled_by_rating_and_s_base': SOURCE_DUAL_UPDATE in inspect.getsource(srp),
        'wrapped_names_exist': all(callable(getattr(srp, n, None)) for n in WRAPPED),
        'writer_is_gate_result_io': callable(getattr(GRIO, 'dumps', None)),
    }
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'interface-dual capture path missing (before any solve): {failing}')
    return checks


class InterfaceDualCapture:
    """The per-run state. `sink` (tests): a list collecting (fname, obj) instead of files."""

    def __init__(self, eval_dir, sink=None):
        self.eval_dir = eval_dir
        self.sink = sink
        self.calls = 0
        self.lines = 0
        self.errors = []
        self.header = None
        self.blocks = None
        self.closed = False
        self.bytes_written = 0
        self.per_line_bytes = []

    def _write(self, obj):
        text = GRIO.dumps(obj, default=GRIO.json_default) + '\n'
        if self.sink is not None:
            self.sink.append((INTERFACE_DUAL_FILE, obj))
        else:
            with open(os.path.join(self.eval_dir, INTERFACE_DUAL_FILE), 'a') as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        self.bytes_written += len(text.encode())
        return len(text.encode())

    def on_boyd(self, planning_problem, tso_model, dso_models, dual_vars, admm_parameters):
        """Called after production's get_admm_boyd_residual_metrics returned. Never raises."""
        if self.closed:
            return
        self.calls += 1
        cycle = self.calls
        t0 = time.time()
        line = {'cycle': cycle, 'captured': False, 'error': None}
        try:
            if self.header is None:
                self.blocks = block_order(planning_problem)
                n_periods = len(dual_vars['pf']['tso']['current'][self.blocks[0][0]][self.blocks[0][1]][
                    self.blocks[0][2]]['p'])
                self.header = {'schema': SCHEMA, 'header': True, 'first_cycle': cycle,
                               'blocks': [[n, y, d] for n, y, d in self.blocks], 'n_blocks': len(self.blocks),
                               'n_periods': n_periods,
                               'fields': [f for _g, _a, _k, f in CHANNEL_FIELDS],
                               'source': ("dual_vars['pf'][agent]['current'][node_id][year][day][kind][period], read at "
                                          'get_admm_boyd_residual_metrics (after the cycle\'s dual updates, before AA)'),
                               'metadata': read_metadata(planning_problem, tso_model, dso_models, admm_parameters,
                                                         self.blocks)}
                self._write(self.header)
            line.update(read_duals(dual_vars, self.blocks))
            line['captured'] = True
        except Exception as error:  # noqa: BLE001 -- write-only capture: recorded, never raised
            line['error'] = f'{type(error).__name__}: {error}'
            self.errors.append({'cycle': cycle, 'error': line['error']})
        line['capture_s'] = time.time() - t0
        try:
            self.per_line_bytes.append(self._write(line))
            self.lines += 1
        except Exception as error:  # noqa: BLE001
            self.errors.append({'cycle': cycle, 'error': f'write failed: {type(error).__name__}: {error}'})

    def close(self):
        self.closed = True

    def summary(self):
        return {'schema': SCHEMA, 'file': INTERFACE_DUAL_FILE, 'boyd_calls_captured': self.calls,
                'cycle_lines_written': self.lines, 'header_written': self.header is not None,
                'n_blocks': len(self.blocks) if self.blocks else None,
                'n_periods': (self.header or {}).get('n_periods'), 'bytes_written': self.bytes_written,
                'bytes_per_cycle_line_max': max(self.per_line_bytes) if self.per_line_bytes else None,
                'errors': list(self.errors),
                'ok': not self.errors and self.lines == self.calls and (self.calls == 0 or self.header is not None)}


def make_wrapper(capture, original):
    """The pass-through wrapper over production's `get_admm_boyd_residual_metrics` (separated for the checks)."""

    def w_boyd(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):
        result = original(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars,
                          admm_parameters)
        capture.on_boyd(planning_problem, tso_model, dso_models, dual_vars, admm_parameters)
        return result
    return w_boyd


@contextmanager
def interface_dual_capture_hooks(eval_dir, holder):
    """Installs the wrapper for the run; restores production on exit (even on error); `holder[SUMMARY_KEY]` gets the
    summary. Refuses to overwrite an existing sidecar."""
    import shared_resources_planning as srp
    path = os.path.join(eval_dir, INTERFACE_DUAL_FILE)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    capture = InterfaceDualCapture(eval_dir)
    original = srp.get_admm_boyd_residual_metrics
    srp.get_admm_boyd_residual_metrics = make_wrapper(capture, original)
    try:
        yield capture
    finally:
        srp.get_admm_boyd_residual_metrics = original
        holder[SUMMARY_KEY] = capture.summary()


def read_sidecar(path):
    """(header, {cycle: line}) from a sidecar."""
    import json
    header, lines = None, {}
    with open(path) as handle:
        for text in handle:
            if not text.strip():
                continue
            obj = json.loads(text)
            if obj.get('header'):
                header = obj
            else:
                lines[obj['cycle']] = obj
    return header, lines
