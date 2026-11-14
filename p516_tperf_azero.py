"""P5.16 T-perf A-zero -- two ZERO-SOLVE follow-ups to P5_16_TPERF_REPORT.md.

Authority: Planner task "P5.16 T-perf A-zero" (2026-10-31), follow-up to P5_16_TPERF_SPEC_v1.md / P5_16_TPERF_REPORT.md.
Not a production change; nothing here enters a table. No ADMM cycle, NO SOLVE: `SolveProfileGuard(())` (blocking, zero
permitted call sites) is installed BEFORE any production import and `verify(0)` must return [] at the end.

INSTANCE (both questions): campaign s53_w86_tail_recert (spec ddd6cd44), cell n7_4h_e1, eval key ca8927e75d628bd1...,
candidate_key db77e1549af855bc... (2025 cohort; node 7 0.25 MVA / 1.0 MWh; nodes 5, 9 empty), certified at cycle 112.
  certified_models.pkl sha256 fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04 (checked against the
  declared value AND both committed manifests BEFORE unpickling; re-hashed at the end: unchanged).

Q1 -- WHICH OBJECTIVE TERM HAS GRADIENT 1e5.
  Blocks: TSO 2025 Spring (case9) and DSO node 7 2025 Spring (case33_2). Each is CLONED before any evaluation; the loaded
  block is never evaluated or modified. The ACTIVE objective is differentiated at the certified (cycle-112) point with
  `pyomo.core.expr.calculus.derivatives.differentiate(expr, wrt_list=vars, mode=reverse_numeric)`, vars = the free
  (unfixed) Vars of the active objective. Attribution is EXACT by linearity, never by name-guessing:
    active = M * inner        (p58_rescaled_admm_objective: M = admm_objective_scale; asserted)
    inner  = base_copy / eff  + sum_k AL_k        (production `obj = copy(objective.expr) / effective_scale; obj += ...`)
    base_copy = sum_j c_j     (the top-level terms of `objective_function_rule`: named Expressions such as
                               total_gen_cost, total_slack_penalties, ..., interface_settlement_weight * interface_settlement)
  grad(active) = (M/eff) * sum_j grad(c_j) + M * sum_k grad(AL_k); each c_j and each AL_k is differentiated SEPARATELY
  and the closure |grad(active) - sum of parts| is reported. AL_k are classified by the mutable Params they contain
  (dual_* -> AL linear, rho_* -> AL quadratic, prox_gamma_* -> proximal; channel v / pf / ess; *_prev -> previous-iterate).
  Slack families are mapped to their penalty constant in definitions.py by the production penalty functions
  (model_construction_helpers.py voltage_slack_penalty .. shared_ess_day_balance_slack_penalty) and the gradient is
  CHECKED against prob * constant * (baseMVA if the function multiplies by it) * M/eff.
  Economic units are CHECKED: grad(pg) from total_gen_cost / (baseMVA * M/eff) == network.cost_energy_p[0][p], and
  grad(flex_*_down) from total_flex_cost == network.cost_flex[0][p] (the block's build-time `network`, from its pickled
  bound-rule partial, p515_s48 / p516_tperf precedent).
  Effective unscaled dual-infeasibility tolerance (IPOPT 3.14.18, gradient-based scaling, no x scaling):
    scaled stationarity  ||df*grad f + J~' lam~ - z~||_inf / s_d <= tol, and lam = lam~ * dc/df, z = z~/df, so
    unscaled ||grad L||_inf <= tol * s_d / df          (s_d >= 1; = 1 when the printed 'Overall NLP error' equals the
                                                         printed scaled dual infeasibility and dominates the others)
  with df = 100 / max|grad f| (IPOPT nlp_scaling_max_gradient 100). Read from the T-perf IPOPT logs (zero-solve): the
  printed objective factor, the final scaled / unscaled dual infeasibility, the overall NLP error, the user options.
  NL cross-check: the linear-part objective gradient (G0 segment) of the NL file IPOPT read in T-perf case i_warm.

Q2 -- WHERE THE ADMM CYCLE'S WALL TIME GOES (same cell, committed per-solve and per-cycle records).
  Per-cycle wall: production's own print `[INFO] \\t - Iteration N: X.XX s` (shared_resources_planning.py, iter_start
  = time.time() at the top of the cycle body .. iter_end at its end) in child_stdout.log; cross-checked against the
  harness 'drained' event utc stamps (convergence_depth_append_events.jsonl), one per round.
  Per-solve: network_ipopt_solve_records.jsonl carries NO wall time (fields enumerated and recorded); it carries the
  agent, network, attempt and IPOPT iteration count of every network solve. ESSO solves: counted from
  leak_classification_s39_D.jsonl (one record per (node, cycle) ESSO solve); no time recorded.
  PROXY (labelled as such; cross-machine): per-solve time = F + r * iterations, with r = IPOPT OverallAlgorithm /
  iterations and F = (run_smopf wall | IPOPT process wall) - OverallAlgorithm, both MEASURED in T-perf on THIS machine
  (Mac17,3) on the same cell's certified blocks (tperf_local.json). Calibration A = case i_warm, validated on the held-out
  case ii_cold; calibration B = ii_cold, validated on i_warm. TSO records use the case9 parameters, all DSO records the
  case33_2 parameters (nodes 5 and 9, case33_1 / case33_3, were not measured). ESSO solves are NOT in the proxy.
  non-solve share = 1 - sum(proxy) / cycle wall; break-even factor k* = 0.6 * wall / sum(proxy) = how many times slower
  per solve the Studio would have to be than this machine for the non-solve share to fall to 40 %.

PREDICTION (Planner, recorded here BEFORE the run and written to run.log first): on SRP1 the non-solve share of a cycle
is >= 40 % (35 s/cycle recorded vs ~15 s of solves measured in T-perf).

OUTPUT (write-once; refuses if the directory exists): data/SRP1/Results/P516TPERF/azero/
    q1_gradient.json, q2_cycle_time.json, run.log (this process's stdout + stderr, tee'd), manifest_sha256.json
LAUNCH (attached, alone; both streams are tee'd into run.log by the harness itself):
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_azero.py
Harness test into a scratch directory that must not exist (same guard, same computation):
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_azero.py --scratch /tmp/<new dir>
Concurrency: exclusive flock on data/SRP1/Results/P516TPERF/.p516_tperf_azero.lock plus a process scan.
Not touched: the D0 roots (data/SRP1/Results/P516D0), p516_tperf_profile.py, any production file.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

_GUARD = SolveProfileGuard((), label='P5.16 T-perf A-zero (ZERO solves)').install()

import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import psutil  # noqa: E402

THIS = os.path.abspath(__file__)
PY = '/opt/anaconda3/envs/opf_env_py311/bin/python'
STAGE = 'P5.16 T-perf A-zero -- objective-gradient attribution (Q1) and ADMM-cycle wall-time split (Q2); zero solves'
AUTHORITY = ['Planner task "P5.16 T-perf A-zero" (2026-10-31)', 'P5_16_TPERF_SPEC_v1.md', 'P5_16_TPERF_REPORT.md']
TPERF_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P516TPERF')
OUT_REL = os.path.join(TPERF_ROOT_REL, 'azero')
LOCK_REL = os.path.join(TPERF_ROOT_REL, '.p516_tperf_azero.lock')
D0_FORBIDDEN = os.path.join('data', 'SRP1', 'Results', 'P516D0')

PREDICTION = ('Q2 (Planner, recorded before the run): on SRP1 the non-solve share of a cycle is >= 40 % '
              '(35 s/cycle recorded vs ~15 s of solves measured in T-perf).')
PREDICTION_THRESHOLD = 0.40

CAMPAIGN_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
EVAL_REL = os.path.join(CAMPAIGN_REL, 'evals', 'ca8927e75d628bd1_n7_4h_e1')
CERTIFIED_SHA = 'fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04'
CELL_MANIFESTS = (os.path.join(CAMPAIGN_REL, 'campaign_manifest_sha256.json'),
                  os.path.join(EVAL_REL, 'child_manifest_sha256.json'))
CELL_INPUTS = {  # key: (relative path, declared sha256 or None) -- every one must match every manifest that pins it
    'certified_models': (os.path.join(EVAL_REL, 'certified_models.pkl'), CERTIFIED_SHA),
    'network_ipopt_solve_records': (os.path.join(EVAL_REL, 'network_ipopt_solve_records.jsonl'), None),
    'child_stdout': (os.path.join(EVAL_REL, 'child_stdout.log'), None),
    'convergence_depth_append_events': (os.path.join(EVAL_REL, 'convergence_depth_append_events.jsonl'), None),
    'per_cycle_record': (os.path.join(EVAL_REL, 'per_cycle_record.jsonl'), None),
    'evaluation_record': (os.path.join(EVAL_REL, 'evaluation_record.json'), None),
    'post_certification': (os.path.join(EVAL_REL, 'post_certification.json'), None),
    'leak_classification': (os.path.join(EVAL_REL, 'leak_classification_s39_D.jsonl'), None),
    'heartbeat': (os.path.join(EVAL_REL, 'heartbeat_s39_D.json'), None),
    'wait4_rusage': (os.path.join(EVAL_REL, 'wait4_rusage.json'), None),
    'g_report': (os.path.join(EVAL_REL, 'g_s39_D.json'), None),
}
TPERF_LOCAL_REL = os.path.join(TPERF_ROOT_REL, 'local')
TPERF_MANIFEST_REL = os.path.join(TPERF_LOCAL_REL, 'manifest_sha256.json')
TPERF_INPUTS = {
    'tperf_local_json': os.path.join(TPERF_LOCAL_REL, 'tperf_local.json'),
    'nl_tso_i_warm': os.path.join(TPERF_LOCAL_REL, 'nl', 'tso_i_warm.nl'),
    'nl_dso_i_warm': os.path.join(TPERF_LOCAL_REL, 'nl', 'dso_i_warm.nl'),
    'log_tso_i_warm': os.path.join(TPERF_LOCAL_REL, 'logs', 'tperf_tso_i_warm_2025_Spring.log'),
    'log_tso_ii_cold': os.path.join(TPERF_LOCAL_REL, 'logs', 'tperf_tso_ii_cold_2025_Spring.log'),
    'log_dso_i_warm': os.path.join(TPERF_LOCAL_REL, 'logs', 'tperf_dso_i_warm_2025_Spring.log'),
    'log_dso_ii_cold': os.path.join(TPERF_LOCAL_REL, 'logs', 'tperf_dso_ii_cold_2025_Spring.log'),
}
PRIOR_ART_REL = os.path.join('data', 'SRP1', 'Results', 'P515S36', 'parallel_audit')
PRIOR_ART_FILE = 'per_cycle_table_compact.json'

INSTANCE = {
    'campaign': 's53_w86_tail_recert', 'campaign_spec_sha256_prefix': 'ddd6cd44', 'cell_label': 'n7_4h_e1',
    'eval_key': 'ca8927e75d628bd188cf74f9d81436cbb3e0be53f64e02d9876ebb271a21951d',
    'candidate_key': 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a',
    'candidate_canonical': {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}},
    'certification_cycle': 112,
}
YEAR, DAY, DSO_NODE = 2025, 'Spring', 7
GRAD_THRESHOLD = 1e3
IPOPT_NLP_SCALING_MAX_GRADIENT = 100.0  # IPOPT 3.14.18 default (not set in the production options; asserted from the log)

# slack family -> (definitions.py constant, production penalty function multiplies by network.baseMVA?, source)
SLACK_FAMILIES = {
    'slack_v_sqr_up': ('PENALTY_VOLTAGE_SQUARED', False, 'mch.voltage_slack_penalty'),
    'slack_v_sqr_down': ('PENALTY_VOLTAGE_SQUARED', False, 'mch.voltage_slack_penalty'),
    'slack_node_balance_p_up': ('PENALTY_NODE_BALANCE', True, 'mch.node_balance_slack_penalty'),
    'slack_node_balance_p_down': ('PENALTY_NODE_BALANCE', True, 'mch.node_balance_slack_penalty'),
    'slack_node_balance_q_up': ('PENALTY_NODE_BALANCE', True, 'mch.node_balance_slack_penalty'),
    'slack_node_balance_q_down': ('PENALTY_NODE_BALANCE', True, 'mch.node_balance_slack_penalty'),
    'slack_flow_ij_sqr': ('PENALTY_CURRENT', True, 'mch.branch_flow_slack_penalty'),
    'slack_flow_ji_sqr': ('PENALTY_CURRENT', True, 'mch.branch_flow_slack_penalty'),
    'slack_flex_p_balance_up': ('PENALTY_FLEXIBILITY', True, 'mch.flexibility_p_day_balance_slack_penalty'),
    'slack_flex_p_balance_down': ('PENALTY_FLEXIBILITY', True, 'mch.flexibility_p_day_balance_slack_penalty'),
    'slack_es_soc_final_up': ('PENALTY_ESS_BALANCE', True, 'mch.local_ess_day_balance_slack_penalty'),
    'slack_es_soc_final_down': ('PENALTY_ESS_BALANCE', True, 'mch.local_ess_day_balance_slack_penalty'),
    'slack_shared_es_soc_final_up': ('PENALTY_SHARED_ESS_BALANCE', True, 'mch.shared_ess_day_balance_slack_penalty'),
    'slack_shared_es_soc_final_down': ('PENALTY_SHARED_ESS_BALANCE', True, 'mch.shared_ess_day_balance_slack_penalty'),
}
# top-level objective_function_rule term -> category
TERM_CATEGORY = {
    'total_gen_cost': 'economic_cost',
    'total_flex_cost': 'economic_cost',
    'total_load_curt_cost': 'economic_cost',
    'row18_deviation_charge': 'economic_cost',
    'interface_settlement': 'economic_transfer_settlement (excluded from Q by convention)',
    'total_gen_curt_penalty': 'usage_penalty',
    'total_ess_utilization_cost_penalty': 'usage_penalty',
    'total_load_curt_penalty': 'usage_penalty',
    'total_flex_penalty': 'usage_penalty',
    'total_slack_penalties': 'slack_penalty',
    'total_ess_complementarity_penalties': 'slack_penalty (ESS day-balance)',
    'scenario_voltage_pin': 'solver_only_pin',
}

T0 = time.time()


# ======================================================================================================================
#  small utilities
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [A-zero +{time.time() - T0:8.2f}s] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _write_json(path, obj):
    import gate_result_io as GRIO
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    GRIO.check(obj, default=GRIO.json_default_item)
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, default=GRIO.json_default_item, indent=1, sort_keys=False)


def _stats(values):
    values = [float(v) for v in values]
    if not values:
        return None
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values),
            'mean': statistics.fmean(values)}


def _decade_histogram(values):
    hist = Counter()
    zeros = 0
    for v in values:
        a = abs(v)
        if a == 0.0:
            zeros += 1
        else:
            hist[int(math.floor(math.log10(a)))] += 1
    return {'zero': zeros,
            'decades': {f'[1e{d}, 1e{d + 1})': hist[d] for d in sorted(hist)},
            'definition': 'count of free objective Vars with |dF/dx| in [10^d, 10^(d+1)); zero counted separately'}


class _Tee:
    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data):
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return False


class Lock:
    def __init__(self, path):
        self.path = path
        self.handle = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, 'a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f'refusing to run: another p516_tperf_azero.py holds {self.path}')
        me = {os.getpid(), os.getppid()}
        others = []
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if proc.info['pid'] not in me and 'p516_tperf_azero.py' in cmd and 'python' in cmd:
                others.append({'pid': proc.info['pid'], 'cmdline': cmd[:300]})
        if others:
            raise RuntimeError(f'refusing to run: other p516_tperf_azero.py processes {others}')
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f'{os.getpid()} {_utc()}\n')
        self.handle.flush()
        return {'path': self.path, 'pid': os.getpid(), 'other_copies_found': others}

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


# ======================================================================================================================
#  inputs
# ======================================================================================================================
def verify_cell_inputs():
    manifests = {rel: json.load(open(os.path.join(REPO, rel))) for rel in CELL_MANIFESTS}
    out = {}
    for key, (rel, declared) in CELL_INPUTS.items():
        now = _sha(os.path.join(REPO, rel))
        pins = {m: d.get(rel) for m, d in manifests.items()}
        problems = []
        if declared is not None and declared != now:
            problems.append(f'declared {declared}')
        for m, value in pins.items():
            if value is not None and value != now:
                problems.append(f'{m}: {value}')
        if not any(v is not None for v in pins.values()):
            problems.append('pinned by no manifest')
        if problems:
            raise RuntimeError(f'input hash mismatch for {rel}: now {now}; {problems}')
        out[key] = {'path': rel, 'sha256': now, 'declared_in_task': declared, 'manifest_pins': pins,
                    'git_tracked': bool(_git(['ls-files', '--', rel])), 'size_bytes': os.path.getsize(rel)}
    return out


def verify_tperf_inputs():
    manifest = json.load(open(os.path.join(REPO, TPERF_MANIFEST_REL)))['files']
    out = {'manifest': {'path': TPERF_MANIFEST_REL, 'sha256': _sha(TPERF_MANIFEST_REL),
                        'git_tracked': bool(_git(['ls-files', '--', TPERF_MANIFEST_REL]))}}
    for key, rel in TPERF_INPUTS.items():
        now = _sha(rel)
        pinned = manifest.get(rel)
        if pinned != now:
            raise RuntimeError(f'T-perf input hash mismatch for {rel}: now {now}, manifest {pinned}')
        out[key] = {'path': rel, 'sha256': now, 'manifest_pin': pinned,
                    'git_tracked': bool(_git(['ls-files', '--', rel]))}
    return out


def verify_prior_art():
    rel = os.path.join(PRIOR_ART_REL, PRIOR_ART_FILE)
    pinned = json.load(open(os.path.join(PRIOR_ART_REL, 'manifest_sha256.json'))).get(PRIOR_ART_FILE)
    now = _sha(rel)
    if pinned != now:
        raise RuntimeError(f'prior-art hash mismatch for {rel}: now {now}, manifest {pinned}')
    return {'path': rel, 'sha256': now, 'manifest_pin': pinned, 'git_tracked': bool(_git(['ls-files', '--', rel]))}


def _build_time_objects(block):
    """The `network` / `params` the block was BUILT with (pickled bound-rule partial; p515_s48 / p516_tperf)."""
    import functools
    import model_construction_helpers as mch
    fcn = block.slack_v_sqr_down._rule_bounds._initializer._fcn
    if not (isinstance(fcn, functools.partial) and fcn.func is mch.voltage_slack_down_bounds):
        raise RuntimeError(f'unexpected bound rule on {block.name}: {fcn!r}')
    return fcn.keywords['network'], fcn.keywords['params']


# ======================================================================================================================
#  IPOPT log readout (T-perf logs; zero solves)
# ======================================================================================================================
def parse_ipopt_log(path):
    text = open(path).read()
    out = {'path': path}
    m = re.search(r'Scaling parameter for objective function = ([0-9.eE+-]+)', text)
    out['objective_scaling_parameter'] = float(m.group(1)) if m else None
    opts = {}
    block = re.search(r'List of options:\n\n.*?\n(.*?)\n\n', text, re.S)
    if block:
        for line in block.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == '=':
                opts[parts[0]] = parts[2]
    out['user_options_listed'] = opts
    final = {}
    for label, key in (('Objective', 'objective'), ('Dual infeasibility', 'dual_infeasibility'),
                       ('Constraint violation', 'constraint_violation'),
                       ('Variable bound violation', 'variable_bound_violation'),
                       ('Complementarity', 'complementarity'), ('Overall NLP error', 'overall_nlp_error')):
        mm = re.findall(rf'^{label}\.*:\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$', text, re.M)
        if mm:
            final[key] = {'scaled': float(mm[-1][0]), 'unscaled': float(mm[-1][1])}
    out['final'] = final
    m = re.findall(r'Number of Iterations\.\.\.\.: (\d+)', text)
    out['iterations'] = int(m[-1]) if m else None
    m = re.findall(r'EXIT: (.*)', text)
    out['exit'] = m[-1].strip() if m else None
    return out


def parse_nl_g0(path):
    """Linear part of the objective gradient: the NL file's G0 segment ('G0 <nnz>' then '<j> <coef>' lines)."""
    coefs = []
    with open(path) as handle:
        lines = iter(handle)
        for line in lines:
            if line.startswith('G0 '):
                n = int(line.split()[1])
                for _ in range(n):
                    j, c = next(lines).split()
                    coefs.append(float(c))
                break
    a = [abs(c) for c in coefs]
    return {'path': path, 'nnz': len(coefs), 'max_abs': max(a) if a else None,
            'count_abs_eq_1e5': sum(1 for x in a if x == 1e5), 'count_abs_eq_5e4': sum(1 for x in a if x == 5e4),
            'count_abs_ge_1e3': sum(1 for x in a if x >= GRAD_THRESHOLD),
            'histogram': _decade_histogram(coefs),
            'note': 'linear part only (coefficient of each variable in the objective\'s linear segment); a variable '
                    'that also enters nonlinearly (the AL quadratics) has its nonlinear part elsewhere (O0 segment). '
                    'Variables are after Pyomo NL presolve; IPOPT further removes lb == ub columns.'}


# ======================================================================================================================
#  Q1
# ======================================================================================================================
def _al_class(term, identify_mutable_parameters):
    names = sorted({p.parent_component().local_name for p in identify_mutable_parameters(term)})
    joined = ' '.join(names)
    if 'vmag' in joined or 'rho_v' in names or 'prox_v_prev' in names or 'prox_gamma_v' in names:
        channel = 'v'
    elif '_pf' in joined:
        channel = 'pf'
    elif 'ess' in joined:
        channel = 'ess'
    else:
        channel = 'unknown'
    if any(n.startswith('prox_gamma') for n in names):
        kind = 'proximal'
    elif any(n.endswith('_prev') for n in names):
        kind = 'AL_previous_iterate_' + ('linear' if any(n.startswith('dual_') for n in names) else 'quadratic')
    elif any(n.startswith('dual_') for n in names):
        kind = 'AL_linear_dual'
    elif any(n.startswith('rho_') for n in names):
        kind = 'AL_quadratic_rho'
    else:
        kind = 'AL_unclassified'
    return f'{kind}:{channel}', names


def q1_block(label, orig, pe, defs):
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
    from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
    from pyomo.core.expr.numeric_expr import ProductExpression, SumExpression, DivisionExpression

    t = time.perf_counter()
    m = orig.clone()  # clone BEFORE any evaluation; `orig` is never evaluated
    clone_s = time.perf_counter() - t
    network, params = _build_time_objects(m)
    actives = list(m.component_data_objects(pe.Objective, active=True))
    if len(actives) != 1:
        raise RuntimeError(f'{label}: expected exactly one active objective, found {[o.name for o in actives]}')
    act = actives[0]
    eff = float(pe.value(m.admm_objective_scale))
    expr = act.expr
    structure = {'active_objective': act.name, 'admm_objective_scale_eff': eff,
                 'admm_common_objective_scale_sigma': float(pe.value(m.admm_common_objective_scale)),
                 'admm_block_weight': float(pe.value(m.admm_block_weight)),
                 'model_objective_active': bool(m.objective.active), 'admm_objective_active': bool(m.admm_objective.active)}
    if act.name == 'p58_rescaled_admm_objective':
        if not (isinstance(expr, ProductExpression) and expr.nargs() == 2 and not hasattr(expr.arg(0), 'is_expression_type')):
            raise RuntimeError(f'{label}: unexpected rescaled-objective structure {type(expr).__name__}')
        M = float(expr.arg(0))
        inner = expr.arg(1)
        structure['inner_is_admm_objective_expr'] = inner is m.admm_objective.expr
        if not structure['inner_is_admm_objective_expr']:
            raise RuntimeError(f'{label}: rescaled objective does not wrap admm_objective.expr')
    elif act.name == 'admm_objective':
        M, inner = 1.0, expr
    else:
        raise RuntimeError(f'{label}: unexpected active objective {act.name}')
    structure['outer_multiplier_M'] = M
    structure['M_equals_eff'] = (M == eff)
    if not isinstance(inner, SumExpression):
        raise RuntimeError(f'{label}: ADMM objective is not a SumExpression')
    base_div = inner.arg(0)
    if not (isinstance(base_div, DivisionExpression) and float(base_div.arg(1)) == eff):
        raise RuntimeError(f'{label}: first ADMM term is not copy(objective.expr) / effective_scale')
    base_copy = base_div.arg(0)
    base_mult = M / eff
    structure['base_multiplier_M_over_eff'] = base_mult
    obj_args = list(m.objective.expr.args) if isinstance(m.objective.expr, SumExpression) else [m.objective.expr]
    base_args = list(base_copy.args) if isinstance(base_copy, SumExpression) else [base_copy]
    structure['base_copy_shares_objective_terms'] = (len(obj_args) == len(base_args)
                                                     and all(a is b for a, b in zip(obj_args, base_args)))
    structure['n_base_terms'] = len(base_args)
    structure['n_admm_terms'] = inner.nargs() - 1
    structure['form'] = ('active = M * [ (sum_j c_j) / eff + sum_k AL_k ]; M = eff for p58_rescaled_admm_objective, so the '
                         'objective_function_rule terms enter at unit weight (EUR of the representative day) and the AL / '
                         'proximal terms are multiplied by eff')

    vs = sorted(identify_variables(expr, include_fixed=False), key=lambda v: v.name)
    n_fixed_in_obj = sum(1 for v in identify_variables(expr, include_fixed=True) if v.fixed)
    n_free_model = sum(1 for v in m.component_data_objects(pe.Var, descend_into=True) if not v.fixed)
    pos = {id(v): i for i, v in enumerate(vs)}
    t = time.perf_counter()
    g_total = [float(x) for x in differentiate(expr, wrt_list=vs, mode=Modes.reverse_numeric)]
    diff_s = time.perf_counter() - t

    parts = {}

    def _add(lab, term, mult):
        tv = list(identify_variables(term, include_fixed=False))
        if not tv:
            return
        g = differentiate(term, wrt_list=tv, mode=Modes.reverse_numeric)
        vec = parts.setdefault(lab, defaultdict(float))
        for v, d in zip(tv, g):
            vec[pos[id(v)]] += float(d) * mult

    base_labels = []
    for a in base_args:
        if hasattr(a, 'is_named_expression_type') and a.is_named_expression_type():
            lab = a.local_name if hasattr(a, 'local_name') else a.name
            weight = 1.0
        elif isinstance(a, ProductExpression):
            named = [x for x in a.args if hasattr(x, 'is_named_expression_type') and x.is_named_expression_type()]
            lab = named[0].local_name if named else f'unnamed_{type(a).__name__}'
            weight = None
        else:
            lab = f'unnamed_{type(a).__name__}'
            weight = None
        base_labels.append({'label': lab, 'type': type(a).__name__,
                            'value_at_certified_point': float(pe.value(a)), 'weight_note': weight})
        _add(lab, a, base_mult)
    al_param_names = {}
    for k in range(1, inner.nargs()):
        a = inner.arg(k)
        lab, names = _al_class(a, identify_mutable_parameters)
        al_param_names.setdefault(lab, set()).update(names)
        _add(lab, a, M)
    weights = {}
    for name in ('interface_settlement_weight', 'scenario_voltage_pin_weight', 'row18_alpha',
                 'penalty_gen_curtailment', 'penalty_shared_ess_usage', 'penalty_ess_usage',
                 'cost_load_curtailment', 'penalty_load_curtailment', 'penalty_flex_usage',
                 'rho_v', 'rho_pf', 'rho_ess', 'rho_ess_prev', 'prox_gamma_v', 'prox_gamma_pf', 'prox_gamma_ess'):
        if hasattr(m, name):
            weights[name] = float(pe.value(getattr(m, name)))

    # closure
    recon = [0.0] * len(vs)
    for vec in parts.values():
        for i, d in vec.items():
            recon[i] += d
    closure = max((abs(a - b) / max(1.0, abs(a)) for a, b in zip(g_total, recon)), default=0.0)
    if closure > 1e-9:
        raise RuntimeError(f'{label}: gradient attribution does not close: {closure}')

    def _category(lab):
        if lab in TERM_CATEGORY:
            return TERM_CATEGORY[lab]
        if lab.startswith('AL_') or lab.startswith('proximal'):
            return 'AL_or_proximal'
        return 'other'

    # per-variable table
    prob = float(network.prob_market_scenarios[0]) * float(network.prob_operation_scenarios[0])
    n_scen = len(m.scenarios_market) * len(m.scenarios_operation)
    if n_scen != 1:
        raise RuntimeError(f'{label}: expected a single scenario on SRP1, found {n_scen}')
    base_mva = float(network.baseMVA)
    rows = []
    for i, v in enumerate(vs):
        contrib = {lab: vec[i] for lab, vec in parts.items() if i in vec and vec[i] != 0.0}
        dominant = max(contrib, key=lambda x: abs(contrib[x])) if contrib else None
        rows.append({'i': i, 'var': v.name, 'family': v.parent_component().local_name,
                     'index': list(v.index()) if isinstance(v.index(), tuple) else v.index(),
                     'value': None if v.value is None else float(v.value),
                     'lb': None if v.lb is None else float(v.lb), 'ub': None if v.ub is None else float(v.ub),
                     'grad': g_total[i], 'contributions': contrib, 'dominant_term': dominant,
                     'dominant_category': _category(dominant) if dominant else 'zero_gradient'})

    # slack-family check against the definitions.py constants
    slack_check = {}
    for fam, (const_name, uses_base, source) in SLACK_FAMILIES.items():
        frows = [r for r in rows if r['family'] == fam]
        if not frows:
            continue
        const = float(getattr(defs, const_name))
        expected = prob * const * (base_mva if uses_base else 1.0) * base_mult
        mism = [r['var'] for r in frows if abs(r['grad'] - expected) > 1e-12 * abs(expected)]
        values = [r['value'] for r in frows if r['value'] is not None]
        slack_check[fam] = {'penalty_constant': const_name, 'penalty_constant_value': const,
                            'multiplied_by_baseMVA': uses_base, 'production_penalty_function': source,
                            'expected_grad = prob * const * (baseMVA) * M/eff': expected,
                            'n_free_vars': len(frows), 'n_grad_mismatch': len(mism), 'mismatch_examples': mism[:5],
                            'slack_value_at_certified_point': _stats(values),
                            'n_slack_value_gt_1e-6': sum(1 for x in values if x > 1e-6),
                            'max_value_over_ub': max((r['value'] / r['ub'] for r in frows
                                                      if r['value'] is not None and r['ub'] not in (None, 0.0)),
                                                     default=None)}
    unmapped_slack_families = sorted({r['family'] for r in rows if r['family'].startswith('slack_')
                                      and r['family'] not in SLACK_FAMILIES})

    # economic unit checks
    unit_checks = {}
    if 'total_gen_cost' in parts:
        diffs = []
        for r in rows:
            if r['family'] == 'pg' and 'total_gen_cost' in r['contributions']:
                p = r['index'][-1]
                price = float(network.cost_energy_p[0][p])
                diffs.append(abs(r['contributions']['total_gen_cost'] / (base_mva * base_mult * prob) - price)
                             / max(1.0, abs(price)))
        unit_checks['pg_total_gen_cost_over_baseMVA_equals_cost_energy_p'] = {
            'n': len(diffs), 'max_rel_diff': max(diffs) if diffs else None}
    if 'total_flex_cost' in parts:
        diffs = []
        for r in rows:
            if r['family'] in ('flex_p_down', 'flex_q_down') and 'total_flex_cost' in r['contributions']:
                p = r['index'][-1]
                price = float(network.cost_flex[0][p])
                diffs.append(abs(r['contributions']['total_flex_cost'] / (base_mva * base_mult * prob) - price)
                             / max(1.0, abs(price)))
        unit_checks['flex_down_total_flex_cost_over_baseMVA_equals_cost_flex'] = {
            'n': len(diffs), 'max_rel_diff': max(diffs) if diffs else None}
    unit_checks['periods'] = len(m.periods)
    unit_checks['note'] = ('cost coefficients are price [EUR/MWh] * baseMVA * pu power per period; with 24 periods '
                           'per day the period is 1 h, so |dF/dx| / baseMVA is a price in EUR/MWh for a pu power '
                           'variable (checked above against the block\'s own price arrays)')

    # large-gradient grouping
    large = [r for r in rows if abs(r['grad']) >= GRAD_THRESHOLD]
    by_family = defaultdict(list)
    for r in large:
        by_family[(r['dominant_term'], r['family'])].append(r)
    groups = []
    for (term, fam), rs in sorted(by_family.items(), key=lambda kv: -max(abs(x['grad']) for x in kv[1])):
        vals = [x['value'] for x in rs if x['value'] is not None]
        groups.append({'dominant_term': term, 'category': _category(term), 'family': fam, 'n_vars': len(rs),
                       'abs_grad': _stats([abs(x['grad']) for x in rs]),
                       'is_slack': fam.startswith('slack_'),
                       'penalty_constant': SLACK_FAMILIES.get(fam, (None,))[0],
                       'value_at_certified_point': _stats(vals),
                       'n_value_gt_1e-6': sum(1 for x in vals if x > 1e-6),
                       'other_terms_contributing': sorted({lab for x in rs for lab in x['contributions']} - {term}),
                       'examples': [{'var': x['var'], 'grad': x['grad'], 'value': x['value'], 'lb': x['lb'],
                                     'ub': x['ub'], 'contributions': x['contributions']} for x in rs[:3]]})

    # per-term and per-category gradient summaries
    per_term = {}
    for lab, vec in parts.items():
        nz = [abs(d) for d in vec.values() if d != 0.0]
        per_term[lab] = {'category': _category(lab), 'n_vars_nonzero': len(nz), 'abs_grad': _stats(nz),
                         'histogram': _decade_histogram(nz)}
        if lab in al_param_names:
            per_term[lab]['params'] = sorted(al_param_names[lab])
    per_cat = defaultdict(list)
    for r in rows:
        per_cat[r['dominant_category']].append(abs(r['grad']))
    per_category = {c: {'n_vars': len(v), 'abs_grad': _stats(v), 'histogram': _decade_histogram(v)}
                    for c, v in sorted(per_cat.items())}
    max_abs = max(abs(x) for x in g_total)
    argmax = [r['var'] for r in rows if abs(r['grad']) == max_abs]
    non_slack_max = max((abs(r['grad']) for r in rows if not r['dominant_category'].startswith('slack_penalty')),
                        default=None)
    out = {
        'block': label, 'model_name': m.name, 'network': network.name, 'year': network.year, 'day': network.day,
        'baseMVA': base_mva, 'scenario_probability': prob, 'clone_s': clone_s, 'differentiate_s': diff_s,
        'structure': structure, 'base_terms': base_labels, 'weights_and_params': weights,
        'n_free_vars_in_objective': len(vs), 'n_fixed_vars_in_objective': n_fixed_in_obj,
        'n_free_vars_in_model': n_free_model,
        'max_abs_grad': max_abs, 'argmax_count': len(argmax), 'argmax_examples': argmax[:10],
        'df_implied = min(1, 100/max|grad|)': min(1.0, IPOPT_NLP_SCALING_MAX_GRADIENT / max_abs),
        'max_abs_grad_excluding_slack_penalty_dominated_vars': non_slack_max,
        'df_implied_without_slack_penalty_vars (arithmetic only, no solve)': (
            min(1.0, IPOPT_NLP_SCALING_MAX_GRADIENT / non_slack_max) if non_slack_max else None),
        'attribution_closure_max_rel': closure,
        'histogram_all': _decade_histogram(g_total),
        'n_abs_grad_ge_1e3': len(large),
        'large_gradient_groups (|grad| >= 1e3)': groups,
        'per_term': per_term, 'per_dominant_category': per_category,
        'slack_family_check': slack_check, 'unmapped_slack_families': unmapped_slack_families,
        'economic_unit_checks': unit_checks,
        'solver_options_from_build_params': dict(params.solver_params.options or {}),
    }
    return out, rows


def tolerance_analysis(blocks, logs):
    out = {'derivation': (
        'IPOPT 3.14.18 gradient-based scaling with no x scaling: scaled problem min df*f s.t. dc*c, dd*d. The scaled '
        'stationarity test is ||df*grad f + sum dc_i lam~_i grad c_i - z~||_inf / s_d <= tol. Unscaled multipliers '
        'lam_i = lam~_i dc_i / df, z = z~/df, so the scaled residual equals df * ||grad L||_inf (unscaled). Hence the '
        'unscaled stationarity tolerance is tol * s_d / df. IPOPT also requires the unscaled ||grad L|| <= dual_inf_tol '
        '(default 1, not set here), which is looser whenever tol * s_d / df < 1.'),
        'blocks': {}}
    for b, info in blocks.items():
        opts = info['solver_options_from_build_params']
        tol = float(opts.get('tol'))
        acc = float(opts.get('acceptable_tol')) if 'acceptable_tol' in opts else None
        df = info['df_implied = min(1, 100/max|grad|)']
        lw, lc = logs[f'{b}_i_warm'], logs[f'{b}_ii_cold']
        sd_evidence = {}
        for case, lg in (('i_warm', lw), ('ii_cold', lc)):
            f = lg['final']
            others = max(f['constraint_violation']['scaled'], f['complementarity']['scaled'])
            sd_evidence[case] = {
                'dual_inf_scaled': f['dual_infeasibility']['scaled'],
                'dual_inf_unscaled': f['dual_infeasibility']['unscaled'],
                'unscaled_over_scaled': f['dual_infeasibility']['unscaled'] / f['dual_infeasibility']['scaled'],
                'overall_nlp_error_scaled': f['overall_nlp_error']['scaled'],
                'overall_equals_dual_inf_scaled': f['overall_nlp_error']['scaled'] == f['dual_infeasibility']['scaled'],
                'dual_inf_dominates_constraint_and_complementarity': f['dual_infeasibility']['scaled'] > others,
                'objective_unscaled': f['objective']['unscaled'],
                'iterations': lg['iterations'], 'exit': lg['exit']}
        sd_is_one = all(e['overall_equals_dual_inf_scaled'] and e['dual_inf_dominates_constraint_and_complementarity']
                        for e in sd_evidence.values())
        tau = tol / df
        econ = [r for r in info['_rows'] if r['dominant_category'] in ('economic_cost',)
                or r['dominant_category'].startswith('economic_transfer')]
        econ_g = [abs(r['grad']) for r in econ if r['grad'] != 0.0]
        al = [abs(r['grad']) for r in info['_rows'] if r['dominant_category'] == 'AL_or_proximal' and r['grad'] != 0.0]
        base_mva = info['baseMVA']
        out['blocks'][b] = {
            'tol': tol, 'acceptable_tol': acc,
            'dual_inf_tol_set_in_options': 'dual_inf_tol' in opts,
            'dual_inf_tol_listed_in_ipopt_user_options': 'dual_inf_tol' in lw['user_options_listed'],
            'df_from_pyomo_gradient': df, 'df_logged_by_ipopt': lw['objective_scaling_parameter'],
            'df_agrees': (lw['objective_scaling_parameter'] is not None
                          and abs(df - lw['objective_scaling_parameter']) <= 1e-12),
            's_d_evidence_from_tperf_logs': sd_evidence,
            's_d_equals_1_at_termination (inferred)': sd_is_one,
            'tau_unscaled = tol * s_d / df (s_d = 1)': tau,
            'tau_unscaled_acceptable = acceptable_tol / df': (acc / df) if acc else None,
            'units': ('objective units per unit of the variable: EUR (representative day, base terms at unit weight in '
                      'p58_rescaled_admm_objective) per pu'),
            'tau_price_equivalent_EUR_per_MWh = tau / baseMVA': tau / base_mva,
            'economic_vars': {'n': len(econ_g), 'abs_grad': _stats(econ_g),
                              'abs_grad_price_equivalent_EUR_per_MWh': _stats([g / base_mva for g in econ_g]),
                              'relative_tolerance tau/|grad|': _stats([tau / g for g in econ_g])},
            'AL_or_proximal_vars': {'n': len(al), 'abs_grad': _stats(al),
                                    'relative_tolerance tau/|grad|': _stats([tau / g for g in al])},
            'achieved_terminal_unscaled_dual_inf_in_tperf': {c: e['dual_inf_unscaled'] for c, e in sd_evidence.items()},
        }
    return out


# ======================================================================================================================
#  Q2
# ======================================================================================================================
def q2_cycle_time(tperf):
    rel = CELL_INPUTS
    lines = open(rel['child_stdout'][0]).read().splitlines()
    walls = {}
    for line in lines:
        mm = re.match(r'^\[INFO\] \t - Iteration (\d+): ([0-9.]+) s$', line)
        if mm:
            k = int(mm.group(1))
            if k in walls:
                raise RuntimeError(f'duplicate cycle-wall line for cycle {k}')
            walls[k] = float(mm.group(2))
    exec_time = [float(x) for x in re.findall(r'^\[INFO\] \t - Execution time: ([0-9.]+) s$', '\n'.join(lines), re.M)]
    events = [json.loads(x) for x in open(rel['convergence_depth_append_events'][0]) if x.strip()]
    drains = {e['round']: datetime.fromisoformat(e['utc']) for e in events if e.get('event') == 'drained'}
    records = [json.loads(x) for x in open(rel['network_ipopt_solve_records'][0]) if x.strip()]
    record_fields = sorted({k for r in records for k in r})
    time_like = [f for f in record_fields if any(t in f.lower() for t in ('time', 'wall', 'second', 'utc', 'elapsed'))]
    leak = [json.loads(x) for x in open(rel['leak_classification'][0]) if x.strip()]
    esso_labels = Counter(str(x['cycle']) for x in leak)
    esso_per_cycle = Counter(int(x['cycle']) for x in leak if str(x['cycle']).isdigit())
    evaluation = json.load(open(rel['evaluation_record'][0]))
    post = json.load(open(rel['post_certification'][0]))
    cycles = sorted(walls)
    if cycles != list(range(1, post['certification']['cycles_run'] + 1)):
        raise RuntimeError(f'cycle-wall lines {cycles[:3]}..{cycles[-3:]} != 1..cycles_run')

    # proxy parameters from T-perf (this machine)
    S = tperf['solves']
    params = {}
    for b in ('tso', 'dso'):
        for c in ('i_warm', 'ii_cold'):
            e = S[f'{b}_{c}']
            it = int(e['production_attempt_parse']['iterations'])
            oa = float(e['ipopt_log']['summary']['overall_algorithm_wall_s'])
            run = float(e['timing']['run_smopf_wall_s'])
            proc = float(e['timing']['ipopt_process_wall_s (_execute_command)'])
            params[(b, c)] = {'iterations': it, 'overall_algorithm_s': oa, 'run_smopf_wall_s': run,
                              'ipopt_process_wall_s': proc, 'r_s_per_iter': oa / it,
                              'F_run_s': run - oa, 'F_proc_s': proc - oa}
    calib = {'A': 'i_warm', 'B': 'ii_cold'}
    held = {'A': 'ii_cold', 'B': 'i_warm'}
    validation = {}
    for cal, c in calib.items():
        for b in ('tso', 'dso'):
            p, h = params[(b, c)], params[(b, held[cal])]
            pred_run = p['F_run_s'] + p['r_s_per_iter'] * h['iterations']
            pred_proc = p['F_proc_s'] + p['r_s_per_iter'] * h['iterations']
            validation[f'{cal}:{b}'] = {
                'calibrated_on': c, 'held_out': held[cal], 'held_out_iterations': h['iterations'],
                'run_smopf_pred_s': pred_run, 'run_smopf_meas_s': h['run_smopf_wall_s'],
                'run_smopf_rel_err': (pred_run - h['run_smopf_wall_s']) / h['run_smopf_wall_s'],
                'ipopt_process_pred_s': pred_proc, 'ipopt_process_meas_s': h['ipopt_process_wall_s'],
                'ipopt_process_rel_err': (pred_proc - h['ipopt_process_wall_s']) / h['ipopt_process_wall_s']}

    by_round = defaultdict(list)
    for r in records:
        by_round[r['round']].append(r)
    per_cycle = []
    for k in cycles:
        recs = by_round.get(k, [])
        row = {'cycle': k, 'wall_printed_s': walls[k],
               'drain_interval_s': ((drains[k] - drains[k - 1]).total_seconds()
                                    if k in drains and (k - 1) in drains else None),
               'n_network_solves': len(recs),
               'n_by_agent': dict(Counter(r['agent'] for r in recs)),
               'n_recovery_attempts': sum(1 for r in recs if r['attempt'] != 'primary'),
               'n_esso_solves': esso_per_cycle.get(k, 0),
               'iterations_tso': sum(int(r['iterations'] or 0) for r in recs if r['agent'] == 'TSO'),
               'iterations_dso': sum(int(r['iterations'] or 0) for r in recs if r['agent'] != 'TSO'),
               'n_iterations_missing': sum(1 for r in recs if r['iterations'] is None)}
        for cal, c in calib.items():
            run_sum = proc_sum = 0.0
            for r in recs:
                b = 'tso' if r['agent'] == 'TSO' else 'dso'
                p = params[(b, c)]
                it = int(r['iterations'] or 0)
                run_sum += p['F_run_s'] + p['r_s_per_iter'] * it
                proc_sum += p['F_proc_s'] + p['r_s_per_iter'] * it
            row[f'proxy_{cal}_sum_run_smopf_s'] = run_sum
            row[f'proxy_{cal}_sum_ipopt_process_s'] = proc_sum
            row[f'proxy_{cal}_non_solve_share_run_smopf'] = 1.0 - run_sum / walls[k]
            row[f'proxy_{cal}_non_solve_share_ipopt_process'] = 1.0 - proc_sum / walls[k]
            row[f'proxy_{cal}_break_even_factor_run_smopf'] = (1.0 - PREDICTION_THRESHOLD) * walls[k] / run_sum
            row[f'proxy_{cal}_break_even_factor_ipopt_process'] = (1.0 - PREDICTION_THRESHOLD) * walls[k] / proc_sum
        per_cycle.append(row)

    def col(key):
        return [r[key] for r in per_cycle if r[key] is not None]

    summary = {'cycle_wall_printed_s': _stats(col('wall_printed_s')),
               'drain_interval_s': _stats(col('drain_interval_s')),
               'printed_minus_drain_interval_s': _stats([r['wall_printed_s'] - r['drain_interval_s']
                                                        for r in per_cycle if r['drain_interval_s'] is not None]),
               'n_network_solves': _stats(col('n_network_solves')), 'n_esso_solves': _stats(col('n_esso_solves')),
               'iterations_tso': _stats(col('iterations_tso')), 'iterations_dso': _stats(col('iterations_dso')),
               'sum_printed_cycle_walls_s': sum(col('wall_printed_s')),
               'production_execution_time_s': exec_time,
               'evaluation_record_wall_time_s': evaluation.get('wall_time_s')}
    for cal in calib:
        for kind in ('run_smopf', 'ipopt_process'):
            summary[f'proxy_{cal}_sum_{kind}_s'] = _stats(col(f'proxy_{cal}_sum_{kind}_s'))
            summary[f'proxy_{cal}_non_solve_share_{kind}'] = _stats(col(f'proxy_{cal}_non_solve_share_{kind}'))
            summary[f'proxy_{cal}_break_even_factor_{kind}'] = _stats(col(f'proxy_{cal}_break_even_factor_{kind}'))
            summary[f'proxy_{cal}_n_cycles_non_solve_share_{kind}_below_0.40'] = sum(
                1 for x in col(f'proxy_{cal}_non_solve_share_{kind}') if x < PREDICTION_THRESHOLD)

    # prediction against the proxy (labelled as a proxy)
    medians_run = [summary[f'proxy_{cal}_non_solve_share_run_smopf']['median'] for cal in calib]
    mins_run = [summary[f'proxy_{cal}_non_solve_share_run_smopf']['min'] for cal in calib]
    medians_proc = [summary[f'proxy_{cal}_non_solve_share_ipopt_process']['median'] for cal in calib]
    verdict = {
        'prediction': PREDICTION,
        'premise_35s_per_cycle_vs_this_cell': summary['cycle_wall_printed_s'],
        'evaluated_on': 'PROXY (cross-machine: per-solve times from T-perf on this machine; cycle walls from the Studio)',
        'non_solve_share_run_smopf_definition_median_over_calibrations': medians_run,
        'non_solve_share_run_smopf_definition_min_cycle_over_calibrations': mins_run,
        'non_solve_share_ipopt_process_definition_median_over_calibrations': medians_proc,
        'holds_at_proxy_level_every_cycle_run_smopf_definition': all(x >= PREDICTION_THRESHOLD for x in mins_run),
        'holds_at_proxy_level_median_run_smopf_definition': all(x >= PREDICTION_THRESHOLD for x in medians_run),
        'break_even_factor_run_smopf_median_over_calibrations': [
            summary[f'proxy_{cal}_break_even_factor_run_smopf']['median'] for cal in calib],
        'break_even_factor_run_smopf_min_cycle_over_calibrations': [
            summary[f'proxy_{cal}_break_even_factor_run_smopf']['min'] for cal in calib],
    }
    return {
        'what_exists': {
            'per_solve_wall_time': False,
            'per_solve_record_fields': record_fields, 'per_solve_time_like_fields': time_like,
            'per_solve_records_n': len(records),
            'per_solve_attempts': dict(Counter(r['attempt'] for r in records)),
            'per_cycle_wall_time': True,
            'per_cycle_wall_source': ('child_stdout.log production print "[INFO] \\t - Iteration N: X.XX s" '
                                      '(shared_resources_planning.py ADMM loop: iter_start = time.time() at the top of '
                                      'the cycle body, iter_end at its end; 2-decimal resolution)'),
            'per_cycle_wall_lines_found': len(walls),
            'drain_events_found': len(drains),
            'drain_source': 'convergence_depth_append_events.jsonl event "drained" utc (harness, once per round)',
            'esso_solves_source': 'leak_classification_s39_D.jsonl (one record per (node, cycle) ESSO solve; no time)',
            'esso_records_per_cycle': _stats(list(esso_per_cycle.values())),
            'ipopt_logs_of_this_cell': ('not on this machine: records point to the Studio path '
                                        + os.path.dirname(records[0]['log_path'])),
            'evaluation_record_wall_time_s': evaluation.get('wall_time_s'),
            'evaluation_record_solve_profile_solves_per_cycle': (evaluation.get('solve_profile') or {}).get(
                'solves_per_cycle'),
            'esso_cycle_labels': {'n_labels': len(esso_labels), 'non_numeric': sorted(l for l in esso_labels if not l.isdigit()),
                                  'records_per_label': _stats(list(esso_labels.values()))},
        },
        'proxy_definition': {
            'per_solve_time': 'F + r * iterations; r = OverallAlgorithm / iterations; F = wall - OverallAlgorithm',
            'variants': {'run_smopf': 'F from the whole network.run_smopf wall (NL write + IPOPT process + result load)',
                         'ipopt_process': 'F from the IPOPT process wall only (excludes NL write and result load)'},
            'calibrations': calib, 'held_out': held,
            'tso_records_use': 'case9 (T-perf TSO block)',
            'dso_records_use': 'case33_2 (T-perf DSO node-7 block) for all three DSO networks; case33_1 / case33_3 '
                               'not measured',
            'esso': 'NOT included (no per-solve time anywhere for this cell); its time is inside the residual',
            'machine': 'per-solve parameters measured on this machine (Mac17,3); cycle walls from the Studio run',
            'parameters': {f'{b}:{c}': v for (b, c), v in params.items()},
            'held_out_validation': validation,
        },
        'per_cycle': per_cycle,
        'summary': summary,
        'prediction_vs_outcome': verdict,
    }


def prior_art_summary():
    rows = json.load(open(os.path.join(PRIOR_ART_REL, PRIOR_ART_FILE)))
    rows = rows if isinstance(rows, list) else rows.get('rows', rows)
    out = {'source': os.path.join(PRIOR_ART_REL, PRIOR_ART_FILE),
           'what': ('P5.15 Step 3.6 parallel audit (WORKER_REPORT_S36_PARALLEL_AUDIT.md): Studio run P515S35_REF, 12 '
                    'sampled cycles; IPOPT time = per-block "Total seconds in IPOPT" from that run\'s own logs; a '
                    'DIFFERENT run, configuration and candidate from this cell -- context only, not this cell'),
           'rows': rows}
    try:
        out['ipopt_esso_s'] = _stats([r['ipopt_esso_s'] for r in rows])
        out['ipopt_total_s'] = _stats([r['ipopt_total_s'] for r in rows])
        out['wall_s'] = _stats([r['wall_s'] for r in rows])
        out['overhead_fraction'] = _stats([r['overhead_fraction'] for r in rows])
    except (KeyError, TypeError):
        out['parse_note'] = 'compact table layout not as expected; rows recorded verbatim'
    return out


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description=STAGE)
    ap.add_argument('--scratch', metavar='DIR', default=None,
                    help='write to this new directory instead of the azero root (harness test; same zero-solve guard)')
    args = ap.parse_args()
    out_dir = os.path.abspath(args.scratch) if args.scratch else os.path.join(REPO, OUT_REL)
    out_rel = os.path.relpath(out_dir, REPO)
    if os.path.abspath(out_dir).startswith(os.path.abspath(os.path.join(REPO, D0_FORBIDDEN))):
        raise RuntimeError('refusing: output under the D0 root')
    lock = Lock(os.path.join(REPO, LOCK_REL))
    lock_info = lock.acquire()
    try:
        if os.path.exists(out_dir):
            raise RuntimeError(f'refusing: output directory exists: {out_dir}')
        os.makedirs(out_dir)
        log_handle = open(os.path.join(out_dir, 'run.log'), 'x')
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(orig_out, log_handle), _Tee(orig_err, log_handle)
        status = 'started'
        try:
            _log(STAGE)
            _log('PREDICTION (recorded before any input is read): ' + PREDICTION)
            _log(f'guard installed before production imports: {_GUARD.label}; permitted call sites: {_GUARD.permitted}')
            prov = {'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
                    'hw_model': subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True, text=True).stdout.strip(),
                    'git_head': _git(['rev-parse', 'HEAD']), 'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
                    'harness_sha256': _sha(THIS), 'lock': lock_info, 'utc_start': _utc()}
            if sys.executable != PY:
                raise RuntimeError(f'interpreter {sys.executable} != canonical {PY}')
            _log('verifying cell inputs (declared + campaign manifest + child manifest)')
            cell_inputs = verify_cell_inputs()
            _log('verifying T-perf inputs against the T-perf manifest; prior-art table against its manifest')
            tperf_inputs = verify_tperf_inputs()
            prior_art_input = verify_prior_art()

            import pyomo
            import pyomo.environ as pe
            import definitions as defs
            prov['pyomo_version'] = pyomo.version.version
            prov['definitions_sha256'] = _sha('definitions.py')
            prov['model_construction_helpers_sha256'] = _sha('model_construction_helpers.py')

            _log('unpickling certified_models.pkl (read-only)')
            with open(CELL_INPUTS['certified_models'][0], 'rb') as handle:
                payload = pickle.load(handle)
            originals = {'tso': payload['tso'][YEAR][DAY], 'dso': payload['dso'][DSO_NODE][YEAR][DAY]}

            logs = {}
            for k in ('tso_i_warm', 'tso_ii_cold', 'dso_i_warm', 'dso_ii_cold'):
                logs[k] = parse_ipopt_log(TPERF_INPUTS[f'log_{k}'])
            q1 = {'schema': 'p516_tperf_azero_q1_v1', 'stage': STAGE, 'authority': AUTHORITY, 'instance': INSTANCE,
                  'objective_convention': ('none reported as a cost: gradients of the ACTIVE solver objective '
                                           '(p58_rescaled_admm_objective) at the certified cycle-112 point'),
                  'blocks': {}, 'nl_cross_check': {}, 'tperf_ipopt_logs': logs}
            for b in ('tso', 'dso'):
                _log(f'Q1 {b}: clone, differentiate, attribute')
                info, rows = q1_block(b, originals[b], pe, defs)
                info['_rows'] = rows
                q1['blocks'][b] = info
                _log(f'  {b}: {info["n_free_vars_in_objective"]} free objective vars; max|grad| {info["max_abs_grad"]:.6g}'
                     f' ({info["argmax_count"]} vars); {info["n_abs_grad_ge_1e3"]} with |grad| >= 1e3; closure '
                     f'{info["attribution_closure_max_rel"]:.2e}')
                for g in info['large_gradient_groups (|grad| >= 1e3)'][:12]:
                    _log(f'    {g["family"]:<34} via {g["dominant_term"]:<36} n={g["n_vars"]:<5} '
                         f'|grad| {g["abs_grad"]["min"]:.4g}..{g["abs_grad"]["max"]:.4g}')
                q1['nl_cross_check'][b] = parse_nl_g0(TPERF_INPUTS[f'nl_{b}_i_warm'])
                q1['nl_cross_check'][b]['pyomo_max_abs_grad'] = info['max_abs_grad']
                q1['nl_cross_check'][b]['pyomo_count_abs_eq_1e5'] = sum(1 for r in rows if abs(r['grad']) == 1e5)
                q1['nl_cross_check'][b]['pyomo_count_abs_eq_5e4'] = sum(1 for r in rows if abs(r['grad']) == 5e4)
                q1['nl_cross_check'][b]['max_agrees'] = (q1['nl_cross_check'][b]['max_abs'] == info['max_abs_grad'])
                q1['nl_cross_check'][b]['tperf_replica_max_abs_obj_gradient'] = json.load(
                    open(TPERF_INPUTS['tperf_local_json']))['solves'][f'{b}_i_warm']['gradient_scaling']['max_abs_obj_gradient']
            del originals
            del payload
            q1['tolerance'] = tolerance_analysis(q1['blocks'], logs)
            for b in ('tso', 'dso'):
                rows = q1['blocks'][b].pop('_rows')
                q1['blocks'][b]['vars_abs_grad_ge_1e3'] = [
                    {k: r[k] for k in ('var', 'family', 'grad', 'value', 'lb', 'ub', 'dominant_term', 'contributions')}
                    for r in rows if abs(r['grad']) >= GRAD_THRESHOLD]
            for b, t in q1['tolerance']['blocks'].items():
                _log(f'  tolerance {b}: df {t["df_from_pyomo_gradient"]:.3g} (logged {t["df_logged_by_ipopt"]}); '
                     f'tau_unscaled {t["tau_unscaled = tol * s_d / df (s_d = 1)"]:.3g}; s_d=1 '
                     f'{t["s_d_equals_1_at_termination (inferred)"]}; econ |grad| '
                     f'{t["economic_vars"]["abs_grad"]}')

            _log('Q2: per-cycle reconstruction')
            tperf = json.load(open(TPERF_INPUTS['tperf_local_json']))
            q2 = {'schema': 'p516_tperf_azero_q2_v1', 'stage': STAGE, 'authority': AUTHORITY, 'instance': INSTANCE,
                  'objective_convention': 'none (times, counts and iterations only)'}
            q2.update(q2_cycle_time(tperf))
            q2['prior_art_context'] = prior_art_summary()
            s = q2['summary']
            _log(f'  cycle wall printed: {s["cycle_wall_printed_s"]}')
            for cal in ('A', 'B'):
                _log(f'  proxy {cal}: sum run_smopf {s[f"proxy_{cal}_sum_run_smopf_s"]["median"]:.2f} s median; '
                     f'non-solve share (run_smopf) median {s[f"proxy_{cal}_non_solve_share_run_smopf"]["median"]:.3f} '
                     f'min {s[f"proxy_{cal}_non_solve_share_run_smopf"]["min"]:.3f}; break-even k* median '
                     f'{s[f"proxy_{cal}_break_even_factor_run_smopf"]["median"]:.2f}')

            post_sha = _sha(CELL_INPUTS['certified_models'][0])
            if post_sha != CERTIFIED_SHA:
                raise RuntimeError(f'certified_models.pkl changed during the run: {post_sha}')
            failures = _GUARD.verify(0)
            guard = {'label': _GUARD.label, 'permitted': list(_GUARD.permitted), 'counts': dict(_GUARD.counts),
                     'verify_0': failures,
                     'armed_at_end': getattr(__import__('pyomo.opt.base.solvers', fromlist=['OptSolver']).OptSolver.solve,
                                             '__name__', '') == 'guarded_solve'}
            if failures or not guard['armed_at_end']:
                raise RuntimeError(f'solve guard failed: {guard}')
            _log(f'guard verify(0): {failures}; counts {_GUARD.counts}; armed at end {guard["armed_at_end"]}')
            prov['utc_end'] = _utc()
            prov['wall_s'] = time.time() - T0
            for rec in (q1, q2):
                rec['provenance'] = prov
                rec['inputs'] = {'cell': cell_inputs, 'tperf': tperf_inputs, 'prior_art': prior_art_input}
                rec['guard'] = guard
                rec['post_run_certified_models_sha256'] = post_sha
            _write_json(os.path.join(out_dir, 'q1_gradient.json'), q1)
            _write_json(os.path.join(out_dir, 'q2_cycle_time.json'), q2)
            status = 'completed'
            _log('status: completed; writing the manifest')
        except Exception:
            traceback.print_exc()
            status = 'failed'
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = orig_out, orig_err
            log_handle.close()
            files = {}
            for name in sorted(os.listdir(out_dir)):
                files[os.path.join(out_rel, name)] = _sha(os.path.join(out_dir, name))
            files[os.path.relpath(THIS, REPO)] = _sha(THIS)
            for rel, _ in CELL_INPUTS.values():
                files[rel] = _sha(rel)
            for rel in list(TPERF_INPUTS.values()) + [TPERF_MANIFEST_REL, os.path.join(PRIOR_ART_REL, PRIOR_ART_FILE)]:
                files[rel] = _sha(rel)
            with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
                json.dump({'schema': 'p516_tperf_azero_manifest_v1', 'utc': _utc(), 'status': status,
                           'files': files}, handle, indent=1)
            print(f'manifest written ({len(files)} files); status {status}', flush=True)
    finally:
        lock.release()


if __name__ == '__main__':
    main()
