"""
P5.15 S56 Audit C (zero solves): facts for the unified core rerun.

Authority: TASKS.md "S56 order" -- Audit C (author's 2nd request 2026-11-21): persistent workers vs the
tight-tail final-solve tolerances; inventory of existing core simulations and their records; grid facts;
runtimes of those run types.

WHAT THIS DOES
  Reads committed / hash-recorded records only (JSON files under data/SRP1/Results and the code files named in
  CODE_ANCHORS) and writes ONE new directory, refusing if it exists:
      data/SRP1/Results/P515S56/input_audit_C/
          audit_C_facts.json      -- extracted facts, every value with its source path; written via gate_result_io
          manifest_sha256.json    -- sha256 of every file read and of the facts file
          launch.log              -- this run's stdout + stderr (tee'd by the script itself)

  ZERO SOLVES, ZERO BUILDS, ZERO PICKLE LOADS:
    - p513_solve_profile_guard.SolveProfileGuard(permitted=()) installed before any other project import and
      verified at exactly 0 (all four counters) at the end;
    - pickle.load / pickle.loads replaced by raisers (call count must be 0);
    - no production module is imported (asserted at the end against sys.modules).
  No git command is run.

EXACT LAUNCH COMMAND (attached, both streams captured by the script's own tee):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_audit_C_extract.py
"""

import glob
import hashlib
import json
import os
import pickle
import re
import statistics
import sys
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_REL = 'data/SRP1/Results/P515S56/input_audit_C'
OUT_DIR = os.path.join(REPO, OUT_REL)
# Trial runs only: S56_AUDIT_C_TRIAL_OUT=<absolute scratch dir> redirects every write outside the repository.
if os.environ.get('S56_AUDIT_C_TRIAL_OUT'):
    OUT_DIR = os.environ['S56_AUDIT_C_TRIAL_OUT']
    OUT_REL = OUT_DIR

PRODUCTION_MODULES = ('shared_resources_planning', 'network', 'network_data', 'shared_energy_storage_data',
                      'admm_persistent_workers', 'admm_parameters', 'model_construction_helpers',
                      'admm_anderson_acceleration', 'p515_s44_campaign_harness', 'uncoordinated_benchmark')


# ---------------------------------------------------------------------------------------------------------------
# Output directory (refuse if it exists) and the self-tee'd launch log.
# ---------------------------------------------------------------------------------------------------------------
if os.path.exists(OUT_DIR):
    sys.stderr.write(f'REFUSED: {OUT_REL} already exists (write-once).\n')
    sys.exit(2)
os.makedirs(OUT_DIR)
_LOG = open(os.path.join(OUT_DIR, 'launch.log'), 'w', buffering=1)


class _Tee:
    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        _LOG.write(s)

    def flush(self):
        self.stream.flush()
        _LOG.flush()


sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)


def _utc():
    return datetime.now(timezone.utc).isoformat()


print(f'[{_utc()}] p515_s56_audit_C_extract start; out {OUT_REL}')

# ---------------------------------------------------------------------------------------------------------------
# Guards: solve guard (blocking), pickle block.
# ---------------------------------------------------------------------------------------------------------------
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (imports pyomo.opt only)
import gate_result_io  # noqa: E402  (imports json only)

GUARD = SolveProfileGuard(permitted=(), label='s56_audit_C')
GUARD.install()
_PICKLE_CALLS = {'load': 0, 'loads': 0}


def _blocked_load(*a, **k):
    _PICKLE_CALLS['load'] += 1
    raise RuntimeError('pickle.load blocked (zero-load audit)')


def _blocked_loads(*a, **k):
    _PICKLE_CALLS['loads'] += 1
    raise RuntimeError('pickle.loads blocked (zero-load audit)')


pickle.load = _blocked_load
pickle.loads = _blocked_loads

READ = {}  # rel path -> sha256


def _rel(p):
    return os.path.relpath(os.path.abspath(p), REPO)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    rel = _rel(path)
    READ[rel] = _sha(path)
    with open(path) as f:
        return json.load(f)


def read_text(path):
    rel = _rel(path)
    READ[rel] = _sha(path)
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------------------------------------------
# Q1: code anchors (line numbers located at run time, with the file's sha256).
# ---------------------------------------------------------------------------------------------------------------
CODE_ANCHORS = {
    'shared_resources_planning.py': [
        ('pool_constructed', 'persistent_pool = admm_persistent_workers.PersistentWorkerPool('),
        ('tail_refuses_persistent_pool', "raise ValueError('convergence_depth_tail is not supported together with persistent_workers.')"),
        ('tail_refusal_comment', '# Worker processes hold their own copies of the solver-option holders; the tail could not reach them.'),
        ('tail_apply_per_cycle_call', 'tail_cycle_record = _apply_convergence_depth_tail('),
        ('tail_next_state_call', 'convergence_depth_tail_active_next = _convergence_depth_tail_next_state('),
        ('tail_restore_at_exit', "convergence_depth_tail_state['restore_at_exit'] = _apply_convergence_depth_tail("),
        ('dso_stage_pool_dispatch', "results['dso'] = persistent_pool.run_dso_stage("),
        ('tso_stage_pool_dispatch', "results['tso'] = persistent_pool.run_tso_stage("),
        ('esso_stage_pool_dispatch', "results['esso'] = persistent_pool.run_esso_stage("),
        ('drain_records_per_cycle', 'network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, iter))'),
        ('tail_option_constant', "CONVERGENCE_DEPTH_TAIL_OPTION = 'compl_inf_tol'"),
        ('def_tail_holders', 'def _convergence_depth_tail_holders(planning_problem):'),
        ('def_tail_baseline', 'def _capture_convergence_depth_tail_baseline(planning_problem, admm_parameters):'),
        ('tail_baseline_refuses_recovery_override', "f'{CONVERGENCE_DEPTH_TAIL_OPTION!r}; the retry tiers would override the tail value.')"),
        ('def_tail_apply', 'def _apply_convergence_depth_tail(planning_problem, admm_parameters, active, baseline, cycle):'),
        ('tail_write', 'solver_params.options[CONVERGENCE_DEPTH_TAIL_OPTION] = tail_value'),
        ('def_tail_next_state', 'def _convergence_depth_tail_next_state(cycle_convergence, aa_enabled, aa_record):'),
        ('def_drain_records', 'def _drain_network_ipopt_solve_records(planning_problem, round_index):'),
        ('snapshot_off_refuses_persistent', "f'persistent_workers enabled: the persistent-worker path clones the block being '"),
        ('def_dso_parallel_path', 'def update_distribution_coordination_models_and_solve_parallel('),
        ('dso_parallel_pool', 'with ProcessPoolExecutor(max_workers=max_workers) as executor:'),
    ],
    'admm_persistent_workers.py': [
        ('spawn_context', "MP_CONTEXT = mp.get_context('spawn')"),
        ('dso_spec_network_params', "'network_params': distribution_network.params,"),
        ('tso_spec_network_params', "'network_params': transmission_network.params,"),
        ('esso_spec_solver_params', "'esso_solver_params': shared_ess_data.params.solver_params,"),
        ('worker_args_pickled_at_start', 'args=(worker_id, block_specs_by_worker[worker_id], task_q, result_q,'),
        ('solve_batch_message', "'cmd': 'solve_batch', 'cycle': cycle, 'from_warm_start': from_warm_start,"),
        ('worker_dso_tso_solve', 'result = network_yd.run_smopf(model, network_params, from_warm_start=from_warm_start, print_header=True)'),
        ('worker_esso_tol_overrides', 'option_overrides=ESSO_TOL_OVERRIDES,'),
        ('worker_returns_model_result_only', "'key': key, 'model': model, 'result': result,"),
        ('def_dispatch_stage', 'def _dispatch_stage(self, kind, canonical_keys, cycle, from_warm_start, captured_states):'),
    ],
    'network.py': [
        ('def_create_smopf_solver', 'def _create_smopf_solver(network, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):'),
        ('options_merge_case_then_overrides', '        options.update(solver_params.options)'),
        ('def_append_ipopt_solve_record', 'def _append_ipopt_solve_record(network, params, solver, solver_log_path, log_offset, log_suffix, from_warm_start):'),
        ('records_on_network_object', 'network.ipopt_solve_records = records'),
        ('def_run_smopf', 'def _run_smopf(network, model, params, from_warm_start=False):'),
        ('tier1_recovery_cold', "recovery_options['warm_start_init_point'] = 'no'"),
        ('tier2_adaptive', "tier2_options['mu_strategy'] = 'adaptive'"),
        ('tier2_flag', "getattr(params.solver_params, 'recovery_tier2_enabled', True)"),
    ],
    'admm_parameters.py': [
        ('persistent_workers_default', 'self.persistent_workers = {'),
        ('tail_default', 'self.convergence_depth_tail = {'),
    ],
    'shared_energy_storage_data.py': [
        ('esso_tol_overrides', "ESSO_TOL_OVERRIDES = {'tol': 1e-10, 'acceptable_tol': 1e-9}"),
    ],
    'p515_s44_campaign_harness.py': [
        ('harness_tail_declaration_validator', 'def validate_convergence_depth_tail(value):'),
        ('harness_concurrency_safety_premise', 'SOL files are uniquely named; `ParallelExecution` is false and persistent'),
        ('harness_requires_persistent_off', "checks['persistent_workers_off'] = not a.persistent_workers.get('enabled')"),
        ('harness_requires_parallel_execution_off', "checks['parallel_execution_off'] = not planning.parallel_execution"),
        ('harness_raises_if_not_as_frozen', "raise RuntimeError(f'S44 campaign child: configuration not as frozen (case file D + cap): {missing}')"),
    ],
    'uncoordinated_benchmark.py': [
        ('benchmark_single_scenario_only_doc', 'Single-scenario only (SRP1). Above one scenario the fixed-interface rows would pin an expectation the per-scenario'),
        ('benchmark_require_single_scenario', 'def require_single_scenario(planning_problem):'),
    ],
    'p515_s55_step7_resettle_hooks.py': [
        ('step7_refuses_derived_instance', "'step7_no_premium_or_derived_instance': ('interface_deviation_premium' not in entry"),
    ],
    'p515_s55_e1_launcher.py': [
        ('e1_key_derived_instance_none', "'derived_instance': None, 'interface_deviation_premium': None,"),
        ('e1_memory_per_slot', 'MEMORY_PER_SLOT_BYTES = 11 * GIB // 4'),
        ('e1_batch_concurrency', 'BATCH_CONCURRENCY = 4'),
        ('e1_memory_preflight', 'def memory_preflight(concurrency):'),
    ],
}

# Count-only probes (a string's occurrence count in a file; 0 is a scoped negative).
CODE_COUNTS = {
    'p515_s44_campaign_harness.py': ['persistent_workers', 'ParallelExecution', 'parallel_execution'],
    'p515_s55_step7_launcher.py': ['persistent_workers'],
    'p515_s55_e1_launcher.py': ['persistent_workers'],
    'admm_persistent_workers.py': ['convergence_depth_tail', 'compl_inf_tol', 'ipopt_solve_records',
                                   'recovery_tier2', 'option_overrides'],
}


def locate_anchors():
    out = {}
    for fname, anchors in CODE_ANCHORS.items():
        path = os.path.join(REPO, fname)
        text = read_text(path)
        lines = text.split('\n')
        entry = {'sha256': READ[fname], 'anchors': {}}
        for label, needle in anchors:
            hits = [i + 1 for i, line in enumerate(lines) if needle in line]
            entry['anchors'][label] = {'needle': needle, 'lines': hits, 'found': bool(hits)}
        out[fname] = entry
    counts = {}
    for fname, needles in CODE_COUNTS.items():
        text = read_text(os.path.join(REPO, fname))
        counts[fname] = {n: text.count(n) for n in needles}
    return out, counts


# ---------------------------------------------------------------------------------------------------------------
# Q2/Q4: evaluation inventory.
# ---------------------------------------------------------------------------------------------------------------
W188A = 'data/SRP1/Results/P515S55/w188a_slack_inventory/w188a_slack_inventory.json'
FROZEN_T = 'data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json'
SPEC_V28 = 'data/SRP1/Results/P515S53/frozen_s53_spec_v28_c8b5a715.json'

EXTRA_EVAL_GLOBS = [
    ('E3_cnrf', 'data/SRP1/Results/P515S55/e3_cnrf/campaign_*/evals/*'),
    ('E2_rho', 'data/SRP1/Results/P515S55/e2_rho/campaign_*/evals/*'),
    ('W187_screen_1x', 'data/SRP1/Results/P515S55/e1_screen_1x/campaign_*/evals/*'),
    ('pilot_2x2_s52', 'data/SRP1/Results/P515S52/campaign_s52_pilot*/evals/*'),
    ('3x3_smoke_s53', 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_smoke_*/evals/*'),
    ('search_s45', 'data/SRP1/Results/P515S45/campaign_s45_*/evals/*'),
    ('search_s47', 'data/SRP1/Results/P515S47/campaign_s47_*/evals/*'),
    ('search_s49_flex_ladder', 'data/SRP1/Results/P515S49/campaign_s49_*/evals/*'),
    ('search_s50_marginal', 'data/SRP1/Results/P515S50/campaign_s50_*/evals/*'),
    ('search_s51_f2', 'data/SRP1/Results/P515S51/campaign_s51_*/evals/*'),
    ('search_s53_f2_certificate', 'data/SRP1/Results/P515S53/campaign_s53_f2_certificate*/evals/*'),
]

BENCHMARK_ROOT = 'data/SRP1/Results/P515S53/w116_benchmark_nrf'


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _short(v, n=240):
    if isinstance(v, str) and len(v) > n:
        return v[:n] + ' ...'
    return v


def extract_eval(eval_dir, group, t2_status=None, t2_item=None, w188a_status=None):
    ed = os.path.join(REPO, eval_dir)
    out = {'group': group, 'eval_dir': _rel(ed), 'label': os.path.basename(ed.rstrip('/'))}
    rec_p = os.path.join(ed, 'evaluation_record.json')
    w4_p = os.path.join(ed, 'wait4_rusage.json')
    la_p = os.path.join(ed, 'launch.json')
    ex_p = os.path.join(ed, 'exit_code.txt')
    if os.path.exists(la_p):
        la = read_json(la_p)
        out['started_utc'] = la.get('started_utc')
        out['thread_caps_in_child_env'] = la.get('thread_caps_in_child_env')
    if os.path.exists(ex_p):
        out['exit_code_txt'] = read_text(ex_p).strip()
    if os.path.exists(w4_p):
        w4 = read_json(w4_p)
        out['wall_s_parent_view'] = w4.get('wall_s_parent_view')
        out['ru_maxrss_bytes'] = _get(w4, 'rusage', 'ru_maxrss')
        out['ru_utime_s'] = _get(w4, 'rusage', 'ru_utime')
        out['wait4_exit_code'] = w4.get('exit_code')
    if not os.path.exists(rec_p):
        out['record_present'] = False
        return out
    r = read_json(rec_p)
    out['record_present'] = True
    cfg = r.get('configuration') or {}
    sp = r.get('solve_profile') or {}
    cert = r.get('certification') or {}
    di = r.get('derived_instance')
    out.update({
        'eval_key': r.get('eval_key'),
        'candidate_key': r.get('candidate_key'),
        'candidate_label': r.get('candidate_label'),
        'candidate_canonical': r.get('candidate_canonical'),
        'campaign_spec_path': r.get('campaign_spec_path'),
        'campaign_spec_sha256': r.get('campaign_spec_sha256'),
        'config_name': _short(cfg.get('name')),
        'config_case_file': cfg.get('case_file'),
        'config_case_file_sha256': cfg.get('case_file_sha256'),
        'convergence_depth_tail': cfg.get('convergence_depth_tail'),
        'convergence_depth_tail_applied_in_child': r.get('convergence_depth_tail_applied_in_child'),
        'anderson_acceleration_effective_in_child': r.get('anderson_acceleration_effective_in_child'),
        'ess_ageing_baseline_label': r.get('ess_ageing_baseline_label') or cfg.get('ess_ageing_baseline_label'),
        'minimum_soh': _get(r, 'ess_ageing_baseline', 'minimum_soh'),
        'model_variant': r.get('model_variant'),
        'model_variant_label': _short(r.get('model_variant_label')),
        'flex_price_multiplier': r.get('flex_price_multiplier'),
        'flex_price_label': _short(r.get('flex_price_label')),
        'rho0_scale': r.get('rho0_scale'),
        'interface_deviation_premium': r.get('interface_deviation_premium'),
        'release_solution_bookkeeping': r.get('release_solution_bookkeeping'),
        'derived_instance_case_path': di.get('case_path') if isinstance(di, dict) else None,
        'derived_instance_case_sha256': di.get('case_sha256') if isinstance(di, dict) else None,
        'record_status': r.get('status'),
        'status_production_trajectory': r.get('status_production_trajectory'),
        'certification_cycle': r.get('certification_cycle'),
        'cycles_run': r.get('cycles_run'),
        'cap': r.get('cap'),
        'certified_by_production': cert.get('certified'),
        'stopped_by': _get(r, 'stopped_by_trajectory', 'stopped_by'),
        'solves_observed': _get(sp, 'observed', 'permitted_solve'),
        'solves_per_cycle': sp.get('solves_per_cycle'),
        'rounds': sp.get('rounds'),
        'n_years': sp.get('n_years'),
        'n_days': sp.get('n_days'),
        'wall_time_s': r.get('wall_time_s'),
        'peak_rss': {k: v for k, v in (r.get('peak_rss') or {}).items() if k != 'semantics'} or None,
        'gross_operational_cost_terminal': r.get('terminal_gross_operational_cost'),
        'certified_cost': r.get('certified_cost'),
        'local_solve_failures_total': cert.get('local_solve_failures_total'),
        'network_failure_classes': _get(r, 'network_failures_summary', 'classes'),
    })
    nrf = [k for k in r if 'nrf' in k.lower()]
    if nrf:
        out['nrf_fields'] = {k: _short(r[k]) if not isinstance(r[k], (dict, list)) else _short(json.dumps(r[k]), 300)
                             for k in nrf}
    ss = r.get('settling_resettle_summary') or r.get('settling_continuation_summary')
    if isinstance(ss, dict):
        out['settling_summary'] = {k: ss.get(k) for k in ('status', 'k_star', 'k0_run', 'end_cycle', 'verdict',
                                                          'certified', 'stopped_at', 'cycles_run') if k in ss}
    if t2_status is not None:
        out['frozen_T2_status'] = t2_status
        out['frozen_T2_item'] = t2_item
    if w188a_status is not None:
        out['w188a_status'] = w188a_status
    # Derived (formula stated in facts['formulas']).
    wall = out.get('wall_s_parent_view')
    if wall is None and isinstance(out.get('wall_time_s'), dict):
        wall = out['wall_time_s'].get('child_process_s')
    out['wall_h'] = wall / 3600.0 if wall else None
    out['wall_per_round_s'] = (wall / out['rounds']) if (wall and out.get('rounds')) else None
    rss = out.get('ru_maxrss_bytes') or _get(out, 'peak_rss', 'child_python_process_ru_maxrss')
    out['maxrss_gib'] = rss / (1 << 30) if rss else None
    return out


def extract_benchmark():
    out = {'root': BENCHMARK_ROOT, 'arms': {}}
    root = os.path.join(REPO, BENCHMARK_ROOT)
    for arm_dir in sorted(glob.glob(os.path.join(root, '*'))):
        name = os.path.basename(arm_dir)
        pc = os.path.join(arm_dir, 'phase_checkpoints.jsonl')
        if not os.path.isfile(pc):
            continue
        lines = [json.loads(l) for l in read_text(pc).split('\n') if l.strip()]
        ps = os.path.join(arm_dir, 'per_solve_record.jsonl')
        n_solve_lines = None
        if os.path.exists(ps):
            n_solve_lines = sum(1 for l in read_text(ps).split('\n') if l.strip())
        log_p = os.path.join(root, 'launch_logs', name + '.log')
        start_clock = end_clock = None
        if os.path.exists(log_p):
            log = read_text(log_p).split('\n')
            stamps = [re.match(r'^(\d\d:\d\d:\d\d) \[W\d+ \+\s*([\d.]+)s\]', l) for l in log]
            stamps = [m for m in stamps if m]
            if stamps:
                start_clock = stamps[0].group(1)
                end_clock = stamps[-1].group(1)
                out['arms'].setdefault(name, {})['harness_elapsed_s_last_stamp'] = float(stamps[-1].group(2))
        last = lines[-1] if lines else {}
        out['arms'].setdefault(name, {}).update({
            'phases': [l.get('phase') for l in lines],
            'solves_last_checkpoint': last.get('solves'),
            'per_solve_record_lines': n_solve_lines,
            'gross_operational_cost_last_phase': next((l.get('gross_operational_cost') for l in reversed(lines)
                                                       if 'gross_operational_cost' in l), None),
            'last_checkpoint_utc': last.get('utc'),
            'log_first_stamp_local': start_clock,
            'log_last_stamp_local': end_clock,
            'launch_log': _rel(log_p) if os.path.exists(log_p) else None,
        })
    rep = read_json(os.path.join(root, 'report_v3', 'report_v3.json'))
    out['report_v3_keys'] = sorted(rep.keys())[:60]
    return out


def overlap_counts(evals):
    iv = []
    for i, e in enumerate(evals):
        s = e.get('started_utc')
        w = e.get('wall_s_parent_view')
        if not s or not w:
            continue
        t0 = datetime.fromisoformat(s)
        iv.append((i, t0, t0 + timedelta(seconds=w)))
    for i, a0, a1 in iv:
        others = [evals[j]['eval_dir'] for j, b0, b1 in iv if j != i and b0 < a1 and a0 < b1]
        evals[i]['n_overlapping_evals_in_inventory'] = len(others)
        evals[i]['overlapping_eval_dirs'] = others[:8]


def summarize(evals, key_fn):
    groups = {}
    for e in evals:
        groups.setdefault(key_fn(e), []).append(e)
    out = {}
    for k, es in sorted(groups.items(), key=lambda kv: str(kv[0])):
        def vals(f):
            return [e[f] for e in es if isinstance(e.get(f), (int, float))]
        w = vals('wall_h')
        c = vals('rounds')
        pr = vals('wall_per_round_s')
        m = vals('maxrss_gib')
        out[str(k)] = {
            'n': len(es),
            'wall_h_min_median_max': [min(w), statistics.median(w), max(w)] if w else None,
            'wall_h_sum': sum(w) if w else None,
            'rounds_min_median_max': [min(c), statistics.median(c), max(c)] if c else None,
            'wall_per_round_s_min_median_max': [min(pr), statistics.median(pr), max(pr)] if pr else None,
            'maxrss_gib_max': max(m) if m else None,
            'n_with_overlap': sum(1 for e in es if e.get('n_overlapping_evals_in_inventory')),
            'tail_enabled_count': sum(1 for e in es if _get(e, 'convergence_depth_tail', 'enabled') is True),
            'flex_price_multipliers': sorted({str(e.get('flex_price_multiplier')) for e in es}),
            'model_variants': sorted({json.dumps(e.get('model_variant'), sort_keys=True) for e in es}),
            'record_status_counts': {s: sum(1 for e in es if e.get('record_status') == s)
                                     for s in sorted({str(e.get('record_status')) for e in es})},
        }
    return out


def main():
    facts = {
        'stage': 'P5.15 S56 Audit C (zero solves)',
        'started_utc': _utc(),
        'out_dir': OUT_REL,
        'formulas': {
            'wall_h': 'wait4_rusage.json wall_s_parent_view / 3600 (parent-side wall of the child evaluation process); '
                      'fallback evaluation_record.wall_time_s.child_process_s',
            'wall_per_round_s': 'wall_s_parent_view / solve_profile.rounds (rounds = 1 initialisation + cycles_run); '
                                'includes build, initialisation and post-certification time',
            'maxrss_gib': 'wait4_rusage.rusage.ru_maxrss (bytes on macOS) / 2**30; fallback '
                          'evaluation_record.peak_rss.child_python_process_ru_maxrss',
            'n_overlapping_evals_in_inventory': 'number of OTHER evaluations in this inventory whose '
                                                '[launch.started_utc, + wall_s_parent_view] interval overlaps this '
                                                "one's (only evaluations in this inventory are counted)",
        },
    }
    anchors, counts = locate_anchors()
    facts['Q1_code_anchors'] = anchors
    facts['Q1_code_counts'] = counts
    v28 = read_json(os.path.join(REPO, SPEC_V28))
    facts['Q1_spec_v28_implementation'] = {'path': SPEC_V28, 'sha256': READ[SPEC_V28],
                                           'implementation': v28.get('implementation')}

    w = read_json(os.path.join(REPO, W188A))
    ft = read_json(os.path.join(REPO, FROZEN_T))
    t2 = dict(ft['tables']['cells'])
    t2.update(ft['tables']['cells_appended_w160'])

    evals = []
    seen = set()
    for cell, c in w['cells'].items():
        ed = c['eval_dir']
        t = t2.get(cell, {})
        group = 'T2:' + str(t.get('item')) if t else ('3x3' if cell.startswith('t11_3x3') else 'w188a_other')
        e = extract_eval(ed, group, t2_status=t.get('status') if t else None, t2_item=t.get('item') if t else None,
                         w188a_status=c.get('status'))
        e['cell'] = cell
        evals.append(e)
        seen.add(os.path.normpath(ed))
    for group, pattern in EXTRA_EVAL_GLOBS:
        for ed in sorted(glob.glob(os.path.join(REPO, pattern))):
            if not os.path.isdir(ed):
                continue
            rel = os.path.normpath(_rel(ed))
            if rel in seen:
                continue
            seen.add(rel)
            e = extract_eval(rel, group)
            e['cell'] = None
            evals.append(e)
    overlap_counts(evals)
    facts['evaluations'] = evals
    facts['summary_by_group'] = summarize(evals, lambda e: e['group'])
    facts['summary_by_group_and_flex_variant'] = summarize(
        evals, lambda e: (e['group'], e.get('flex_price_multiplier'), json.dumps(e.get('model_variant'), sort_keys=True)))
    facts['benchmark_w116'] = extract_benchmark()
    facts['frozen_tables_benchmark'] = {k: ft['tables']['benchmark'].get(k) for k in
                                        ('benefit', 'benefit_relative', 'coordinated_Q', 'coordinated_source',
                                         'frozen_spec', 'arms', 'larger_band', 'multiple')}

    # Q3: case files (years, block lengths, scenario counts) -- read only.
    case_files = ['data/SRP1/SRP1.json', 'data/SRP1/SRP1_5x3y_1x1.json',
                  'data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json',
                  'data/SRP1/Results/P515S52/pilot_instance/SRP1__s52_pilot_2x2.json',
                  'data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json']
    cases = {}
    for cf in case_files:
        d = read_json(os.path.join(REPO, cf))
        cases[cf] = {'Years': d.get('Years'), 'NumMarketScenarios': d.get('NumMarketScenarios'),
                     'TN_num_operation_scenarios': _get(d, 'TransmissionNetwork', 'num_operation_scenarios'),
                     'DN_num_operation_scenarios': [n.get('num_operation_scenarios') for n in d.get('DistributionNetworks', [])],
                     'RandomSeed': d.get('RandomSeed'), 'DiscountFactor': d.get('DiscountFactor'),
                     'ess_params_file': _get(d, 'SharedEnergyStorage', 'params_file'),
                     'MarketData': d.get('MarketData'), 'sha256': READ[cf]}
    facts['Q3_case_files'] = cases
    inst = read_json(os.path.join(REPO, 'data/SRP1/Results/P515S53/w89_3x3/instance/instance_record.json'))
    facts['Q3_3x3_instance_record'] = {k: inst.get(k) for k in ('derive_base', 'derive_overrides', 'derive_description',
                                                                 'changes_vs_source', 'source_case', 'git_head')}
    facts['Q3_3x3_instance_record']['prefix_verification'] = {
        k: _get(inst, 'prefix_verification', k) for k in ('definition', 'n_blocks', 'n_arrays_compared', 'n_mismatches',
                                                           'probabilities_3x3_all_one_third')}
    for ess in ('data/SRP1/SharedESS/SRP1_ESS_Params.json', 'data/SRP1/SharedESS/SRP1_5x3y_ESS_Params.json'):
        facts.setdefault('Q3_ess_params', {})[ess] = read_json(os.path.join(REPO, ess))
    facts['Q3_ess_params_sha256'] = {k: READ[k] for k in facts['Q3_ess_params']}

    # ---- guards
    facts['finished_utc'] = _utc()
    failures = GUARD.verify(0, 0)
    facts['guards'] = {'solve_guard_counts': dict(GUARD.counts), 'solve_guard_verify_failures': failures,
                       'pickle_calls': dict(_PICKLE_CALLS),
                       'production_modules_imported': [m for m in PRODUCTION_MODULES if m in sys.modules]}
    if failures or any(_PICKLE_CALLS.values()) or facts['guards']['production_modules_imported']:
        raise SystemExit(f'GUARD FAILURE: {facts["guards"]}')

    facts_path = os.path.join(OUT_DIR, 'audit_C_facts.json')
    with open(facts_path, 'w') as f:
        gate_result_io.dump(facts, f, indent=1, sort_keys=False)
    manifest = {'inputs_sha256': dict(sorted(READ.items())), 'n_inputs': len(READ),
                'outputs_sha256': {_rel(facts_path): _sha(facts_path)},
                'script': {'path': 'p515_s56_audit_C_extract.py', 'sha256': _sha(os.path.abspath(__file__))}}
    with open(os.path.join(OUT_DIR, 'manifest_sha256.json'), 'w') as f:
        gate_result_io.dump(manifest, f, indent=1)
    print(f'[{_utc()}] wrote {len(evals)} evaluations; {len(READ)} inputs hashed; guards {facts["guards"]}')
    print(f'[{_utc()}] facts sha256 {manifest["outputs_sha256"][_rel(facts_path)]}')


if __name__ == '__main__':
    try:
        main()
    finally:
        GUARD.uninstall()
