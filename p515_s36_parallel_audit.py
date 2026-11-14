"""
P5.15 Step 3.6 (`PLANNER_BRIEF_2026-09-13.md` Addendum 18) -- bounded READ-ONLY
audit of the existing parallel solve path and of where the ~39 s ADMM cycle
wall time goes.

**Zero solves.** This script only parses already-written IPOPT log files,
`stdout` captures and `network_failures_*.jsonl` sidecars from a completed
production run (`data/SRP1/Results/P515S35_REF_run/`, evidence directory
`data/SRP1/Results/P56A/evals/p515s35ref_baseline/`). It does not import or
call any solver, does not build any Pyomo model, and does not touch
`shared_resources_planning.py`, `network.py` or `shared_energy_storage_data.py`
in any way other than reading them as text for the static code-audit section.

`p513_solve_profile_guard.SolveProfileGuard` is armed with an EMPTY permitted
list (i.e. zero permitted call sites) for the whole run and verified with
`expected_solves=0` at the end -- if any solve were ever attempted (by this
script or anything it imports), the guard raises immediately and the script
fails loudly rather than silently reporting a number computed after an
undeclared solve.

Output directory: `data/SRP1/Results/P515S36/parallel_audit/` (refuses to run
if it already exists -- this campaign is never re-run onto its own evidence).

Usage:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_parallel_audit.py
"""

import hashlib
import json
import os
import re
import sys
from collections import defaultdict

from p513_solve_profile_guard import SolveProfileGuard

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'parallel_audit')

REF_RUN_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S35_REF_run')
LOGS_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56A', 'evals', 'p515s35ref_baseline', 'logs')
STDOUT_LOG = os.path.join(REF_RUN_DIR, 'stdout_baseline.log')
FAILURES_JSONL = os.path.join(REF_RUN_DIR, 'network_failures_baseline.jsonl')
HEARTBEAT_JSON = os.path.join(REF_RUN_DIR, 'heartbeat_baseline.json')

# Case study topology (SRP1.json / case*_params.json), read manually, not
# re-derived by importing production code (zero-import-of-production rule for
# this bounded audit -- see the docstring).
DSO_NODES = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
YEARS = ['2025', '2030', '2035']
DAYS = ['Spring', 'Summer', 'Autumn', 'Winter']
TSO_NETWORK = 'case9'
ESSO_NODES = [5, 7, 9]

SAMPLE_CYCLES = [1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 477]

IPOPT_SECONDS_RE = re.compile(r'Total seconds in IPOPT\s*=\s*([0-9.eE+-]+)')
CPU_SECS_WO_FUNC_RE = re.compile(r'Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.eE+-]+)')
CPU_SECS_FUNC_RE = re.compile(r'Total CPU secs in NLP function evaluations\s*=\s*([0-9.eE+-]+)')
ITERATION_LINE_RE = re.compile(r'Iteration (\d+):\s*([0-9.]+)\s*s')


def parse_ipopt_solve_records(path):
    """Return, in file order, one dict per solve found in an (appended) IPOPT
    log: {'total_seconds': float, 'cpu_wo_func': float|None, 'cpu_func': float|None}.
    Zero solves are executed here -- this is a text scan of an existing file.
    """
    if not os.path.exists(path):
        return []
    records = []
    pending_wo_func = None
    pending_func = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = CPU_SECS_WO_FUNC_RE.search(line)
            if m:
                pending_wo_func = float(m.group(1))
                continue
            m = CPU_SECS_FUNC_RE.search(line)
            if m:
                pending_func = float(m.group(1))
                continue
            m = IPOPT_SECONDS_RE.search(line)
            if m:
                records.append({
                    'total_seconds': float(m.group(1)),
                    'cpu_wo_func': pending_wo_func,
                    'cpu_func': pending_func,
                })
                pending_wo_func = None
                pending_func = None
    return records


def parse_cycle_wall_times(path):
    """Return {cycle_int: wall_seconds} from the production
    `print(f"[INFO] \\t - Iteration {iter}: {iter_end - iter_start:.2f} s")`
    line (shared_resources_planning.py:3007), unmodified by any harness."""
    wall = {}
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = ITERATION_LINE_RE.search(line)
            if m:
                wall[int(m.group(1))] = float(m.group(2))
    return wall


def load_failure_records(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dso_log_stem(network_name, year, day):
    return os.path.join(LOGS_DIR, f'optim_log_{network_name}_{year}_{day}')


def esso_log_path(node_id, cycle):
    return os.path.join(LOGS_DIR, f'optim_log_esso_node{node_id}_cycle{cycle:03d}.txt')


def build_block_list():
    """The 51 blocks solved once per cycle: 36 DSO + 12 TSO + 3 ESSO."""
    blocks = []
    for node_id, network_name in sorted(DSO_NODES.items()):
        for year in YEARS:
            for day in DAYS:
                blocks.append({
                    'type': 'DSO', 'agent': 'DSO', 'node_id': node_id,
                    'network_name': network_name, 'year': year, 'day': day,
                })
    for year in YEARS:
        for day in DAYS:
            blocks.append({
                'type': 'TSO', 'agent': 'TSO', 'node_id': None,
                'network_name': TSO_NETWORK, 'year': year, 'day': day,
            })
    for node_id in ESSO_NODES:
        blocks.append({'type': 'ESSO', 'agent': 'ESSO', 'node_id': node_id,
                        'network_name': None, 'year': None, 'day': None})
    assert len(blocks) == 51, f'expected 51 blocks, got {len(blocks)}'
    return blocks


def failure_key(rec):
    return (rec['agent'], rec['node_id'], rec['network_name'], rec['year'], rec['day'])


def block_key(block):
    return (block['agent'], block['node_id'], block['network_name'], block['year'], block['day'])


def main():
    if os.path.exists(OUT_DIR):
        print(f'[REFUSE] Output directory already exists, not overwriting: {OUT_DIR}')
        sys.exit(1)
    for required in (REF_RUN_DIR, LOGS_DIR, STDOUT_LOG, FAILURES_JSONL):
        if not os.path.exists(required):
            print(f'[ERROR] Required input not found: {required}')
            sys.exit(1)

    guard = SolveProfileGuard(permitted=(), label='p515_s36_parallel_audit (zero solves)').install()

    try:
        blocks = build_block_list()

        # ---- Primary IPOPT-seconds series per (case,year,day)/(case9,year,day) ----
        dso_tso_series = {}  # key -> list of solve records, occurrence 0 = pre-loop init
        for block in blocks:
            if block['type'] in ('DSO', 'TSO'):
                key = block_key(block)
                stem = dso_log_stem(block['network_name'], block['year'], block['day'])
                dso_tso_series[key] = {
                    'primary': parse_ipopt_solve_records(stem + '.log'),
                    'recovery': parse_ipopt_solve_records(stem + '_recovery.log'),
                    'tier2': parse_ipopt_solve_records(stem + '_recovery_tier2.log'),
                }

        primary_len_report = {
            ':'.join(str(k) for k in key): len(series['primary'])
            for key, series in dso_tso_series.items()
        }
        distinct_primary_lengths = sorted(set(primary_len_report.values()))

        # ---- Failure/recovery event reconciliation ----
        failures = load_failure_records(FAILURES_JSONL)
        by_key = defaultdict(list)
        for rec in failures:
            by_key[failure_key(rec)].append(rec)
        for key in by_key:
            by_key[key].sort(key=lambda r: r['cycle'])
        # position of each (key,cycle) within its key's recovery-log occurrence
        # order, and (for the tier2 subset) within its key's tier2-log order.
        recovery_index = {}   # (key,cycle) -> index into series['recovery']
        tier2_index = {}      # (key,cycle) -> index into series['tier2']
        for key, recs in by_key.items():
            tier2_recs = [r for r in recs if r.get('tier2_attempted')]
            for i, r in enumerate(recs):
                recovery_index[(key, r['cycle'])] = i
            for j, r in enumerate(tier2_recs):
                tier2_index[(key, r['cycle'])] = j

        failure_cycle_set = {(k, c): rec for k, recs in by_key.items() for c, rec in
                              [(r['cycle'], r) for r in recs]}

        # ---- Cycle wall times ----
        wall_times = parse_cycle_wall_times(STDOUT_LOG)

        # ---- Per-sampled-cycle block-type IPOPT-second sums ----
        per_cycle_rows = []
        missing_data_notes = []

        def block_total_seconds(block, cycle):
            """Primary + (if this block failed at this cycle) recovery [+ tier2]
            IPOPT seconds actually spent servicing this block in this cycle."""
            if block['type'] == 'ESSO':
                path = esso_log_path(block['node_id'], cycle)
                recs = parse_ipopt_solve_records(path)
                if not recs:
                    missing_data_notes.append(f'ESSO node {block["node_id"]} cycle {cycle}: no log found at {path}')
                    return None
                if len(recs) != 1:
                    missing_data_notes.append(
                        f'ESSO node {block["node_id"]} cycle {cycle}: expected 1 solve record, found {len(recs)}')
                return sum(r['total_seconds'] for r in recs)

            key = block_key(block)
            series = dso_tso_series[key]
            primary = series['primary']
            if cycle >= len(primary):
                missing_data_notes.append(
                    f'{block["type"]} {key} cycle {cycle}: primary log has only {len(primary)} occurrences '
                    f'(occurrence 0 = pre-loop init, need index {cycle})')
                return None
            total = primary[cycle]['total_seconds']

            rec = failure_cycle_set.get((key, cycle))
            if rec is not None:
                r_idx = recovery_index.get((key, cycle))
                if r_idx is not None and r_idx < len(series['recovery']):
                    total += series['recovery'][r_idx]['total_seconds']
                else:
                    missing_data_notes.append(
                        f'{block["type"]} {key} cycle {cycle}: recovery_attempted but no matching recovery-log '
                        f'occurrence (index {r_idx}, recovery log has {len(series["recovery"])})')
                if rec.get('tier2_attempted'):
                    t_idx = tier2_index.get((key, cycle))
                    if t_idx is not None and t_idx < len(series['tier2']):
                        total += series['tier2'][t_idx]['total_seconds']
                    else:
                        missing_data_notes.append(
                            f'{block["type"]} {key} cycle {cycle}: tier2_attempted but no matching tier2-log '
                            f'occurrence (index {t_idx}, tier2 log has {len(series["tier2"])})')
            return total

        for cycle in SAMPLE_CYCLES:
            by_type_sum = {'DSO': 0.0, 'TSO': 0.0, 'ESSO': 0.0}
            by_type_count = {'DSO': 0, 'TSO': 0, 'ESSO': 0}
            per_block = []
            any_missing = False
            for block in blocks:
                secs = block_total_seconds(block, cycle)
                if secs is None:
                    any_missing = True
                    continue
                by_type_sum[block['type']] += secs
                by_type_count[block['type']] += 1
                per_block.append({**{k: v for k, v in block.items()}, 'ipopt_seconds': secs})

            total_ipopt = sum(by_type_sum.values())
            wall = wall_times.get(cycle)
            row = {
                'cycle': cycle,
                'wall_s': wall,
                'ipopt_dso_s': by_type_sum['DSO'],
                'ipopt_tso_s': by_type_sum['TSO'],
                'ipopt_esso_s': by_type_sum['ESSO'],
                'ipopt_total_s': total_ipopt,
                'overhead_s': (wall - total_ipopt) if wall is not None else None,
                'overhead_fraction': ((wall - total_ipopt) / wall) if wall else None,
                'dso_blocks_counted': by_type_count['DSO'],
                'tso_blocks_counted': by_type_count['TSO'],
                'esso_blocks_counted': by_type_count['ESSO'],
                'any_missing_data': any_missing,
                'failed_blocks_this_cycle': sorted(
                    f'{r["agent"]}|{r.get("node_id")}|{r["network_name"]}|{r["year"]}|{r["day"]}|{r["class"]}'
                    for r in failures if r['cycle'] == cycle
                ),
                'per_block': per_block,
            }
            per_cycle_rows.append(row)

        # ---- Theoretical parallel speed-up (greedy LPT partition per block type) ----
        def lpt_partition_makespan(times, workers):
            if workers <= 0 or not times:
                return 0.0
            loads = [0.0] * min(workers, max(1, len(times)))
            for t in sorted(times, reverse=True):
                i = loads.index(min(loads))
                loads[i] += t
            return max(loads)

        speedup_rows = []
        for row in per_cycle_rows:
            if row['any_missing_data'] or row['wall_s'] is None:
                continue
            dso_times = [b['ipopt_seconds'] for b in row['per_block'] if b['type'] == 'DSO']
            tso_times = [b['ipopt_seconds'] for b in row['per_block'] if b['type'] == 'TSO']
            esso_times = [b['ipopt_seconds'] for b in row['per_block'] if b['type'] == 'ESSO']
            overhead = row['overhead_s']
            for workers in (1, 3, 4, 8):
                dso_wall = lpt_partition_makespan(dso_times, workers)
                tso_wall = lpt_partition_makespan(tso_times, workers)
                esso_wall = lpt_partition_makespan(esso_times, workers)
                parallel_cycle_wall = dso_wall + tso_wall + esso_wall + overhead
                amdahl_f_serial = overhead / row['wall_s'] if row['wall_s'] else None
                amdahl_bound = (1.0 / (amdahl_f_serial + (1 - amdahl_f_serial) / workers)
                                if amdahl_f_serial is not None else None)
                speedup_rows.append({
                    'cycle': row['cycle'],
                    'workers': workers,
                    'dso_parallel_wall_s': dso_wall,
                    'tso_parallel_wall_s': tso_wall,
                    'esso_parallel_wall_s': esso_wall,
                    'overhead_s_unchanged': overhead,
                    'projected_parallel_cycle_wall_s': parallel_cycle_wall,
                    'projected_speedup': row['wall_s'] / parallel_cycle_wall if parallel_cycle_wall else None,
                    'amdahl_f_serial': amdahl_f_serial,
                    'amdahl_speedup_bound': amdahl_bound,
                })

        # ---- Aggregate summary across sampled cycles ----
        valid_rows = [r for r in per_cycle_rows if not r['any_missing_data'] and r['wall_s'] is not None]
        def _median(xs):
            xs = sorted(xs)
            n = len(xs)
            if n == 0:
                return None
            mid = n // 2
            return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])

        summary = {
            'sampled_cycles': SAMPLE_CYCLES,
            'valid_cycles': [r['cycle'] for r in valid_rows],
            'median_wall_s': _median([r['wall_s'] for r in valid_rows]),
            'median_ipopt_total_s': _median([r['ipopt_total_s'] for r in valid_rows]),
            'median_overhead_s': _median([r['overhead_s'] for r in valid_rows]),
            'median_overhead_fraction': _median([r['overhead_fraction'] for r in valid_rows]),
            'median_ipopt_dso_s': _median([r['ipopt_dso_s'] for r in valid_rows]),
            'median_ipopt_tso_s': _median([r['ipopt_tso_s'] for r in valid_rows]),
            'median_ipopt_esso_s': _median([r['ipopt_esso_s'] for r in valid_rows]),
            'run_total_cycles': max(wall_times) if wall_times else None,
            'run_total_dso_tso_failures': len(failures),
        }
        for workers in (1, 3, 4, 8):
            rows_w = [r for r in speedup_rows if r['workers'] == workers]
            summary[f'median_projected_speedup_{workers}w'] = _median(
                [r['projected_speedup'] for r in rows_w if r['projected_speedup'] is not None])
            summary[f'median_amdahl_bound_{workers}w'] = _median(
                [r['amdahl_speedup_bound'] for r in rows_w if r['amdahl_speedup_bound'] is not None])

        # ---- Provenance sanity: primary-log occurrence counts ----
        provenance = {
            'ref_run_dir': REF_RUN_DIR,
            'logs_dir': LOGS_DIR,
            'stdout_log': STDOUT_LOG,
            'failures_jsonl': FAILURES_JSONL,
            'heartbeat_json': HEARTBEAT_JSON,
            'primary_log_occurrence_counts_by_key': primary_len_report,
            'distinct_primary_lengths_observed': distinct_primary_lengths,
            'expected_primary_length': '478 = 1 pre-loop init solve + 477 ADMM cycles',
            'total_failure_records': len(failures),
            'failure_class_counts': {
                cls: sum(1 for r in failures if r['class'] == cls)
                for cls in sorted(set(r['class'] for r in failures))
            },
        }

        # ---------------------------------------------------------------
        # Static code-audit findings (Part 1) -- recorded as data, not
        # computed; every entry cites file:line in this repository, verified
        # by direct reading during this audit.
        # ---------------------------------------------------------------
        code_audit = {
            "dispatch_mechanism": {
                "answer": "concurrent.futures.ProcessPoolExecutor (not threads, not joblib). A NEW "
                          "executor is constructed inside each parallel function call and torn down "
                          "at the end of its `with` block -- worker processes are SPAWNED PER CALL "
                          "(i.e. per cycle, for update_and_solve_dso), not persistent across cycles. "
                          "Every argument (including the full per-node Pyomo model) is pickled to the "
                          "worker and the returned updated model is pickled back.",
                "citations": [
                    "shared_resources_planning.py:6 (from concurrent.futures import ProcessPoolExecutor, as_completed)",
                    "shared_resources_planning.py:3625-3644 (create_distribution_networks_models_parallel: "
                    "new ProcessPoolExecutor per call, max_workers=min(os.cpu_count()//2, len(distribution_networks)))",
                    "shared_resources_planning.py:5429-5449 (update_distribution_coordination_models_and_solve_parallel: "
                    "new ProcessPoolExecutor per call, max_workers=os.cpu_count()//2)",
                    "shared_resources_planning.py:5439-5442 (executor.submit(update_and_solve_dso, node_id, "
                    "distribution_networks[node_id], models[node_id], ...) -- the whole model object is an argument)",
                ],
            },
            "what_is_sent_and_returned": {
                "answer": "Sent to each worker: node_id, the DistributionNetwork/NetworkData object for "
                          "that node, the FULL per-(year,day) Pyomo model dict, the consensus/dual "
                          "dicts (vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess), the ADMM "
                          "params object and the node's estimated SESS capacity. Returned: (node_id, "
                          "results-dict, updated model dict). Results are assembled by COMPLETION ORDER "
                          "via as_completed(), then written into dicts keyed by node_id -- so the final "
                          "res/models dicts are order-independent (keyed, not appended), but nothing in "
                          "the parallel path enforces that the FIRST completing worker corresponds to "
                          "any particular deterministic tie-break if two updates raced on shared state "
                          "(they do not share state here, so this is currently benign).",
                "citations": [
                    "shared_resources_planning.py:5439-5442 (submit args)",
                    "shared_resources_planning.py:5444-5447 (for future in as_completed(tasks): node_id, "
                    "result, updated_model = future.result(); res[node_id] = result; models[node_id] = updated_model)",
                    "shared_resources_planning.py:5499 (return (node_id, res, model))",
                ],
            },
            "solver_invocation_per_worker": {
                "answer": "Same underlying solver path as sequential: network.py's `_create_smopf_solver` "
                          "-> po.SolverFactory('ipopt', executable=solver_params.solver_path) -- i.e. the "
                          "SAME NLP_SOLVER_PATH-configured executable production always uses; no "
                          "per-worker executable override exists. Linear solver and thread count come "
                          "from the case-file `solver.options` (e.g. `linear_solver: ma97` in "
                          "case33_3_params.json) -- there is NO OMP_NUM_THREADS or per-worker thread cap "
                          "set anywhere in production or in the parallel path, so MA97 uses whatever "
                          "default threading it has inside every worker process, and `W` workers means "
                          "`W` simultaneous multi-threaded IPOPT processes with no core-oversubscription "
                          "guard. Pyomo's TempfileManager is never explicitly configured (no `tmpdir=` "
                          "or `keepfiles=True` passed to solver.solve anywhere in network.py or "
                          "shared_energy_storage_data.py), so .nl/.sol scratch files use Pyomo's default "
                          "unique-per-call temp names -- these are NOT a collision risk under multiprocessing. "
                          "The IPOPT **log** path is the collision risk: it is built deterministically "
                          "from (case name, year, day[, log_suffix]) with NO node_id/worker/PID component "
                          "and `file_append='yes'` -- i.e. it silently appends rather than erroring on "
                          "collision. In the current 3-node SRP1 case each DSO node already has a "
                          "distinct case name (case33_1/2/3), so no live collision was observed in the "
                          "sampled evidence, but nothing in the code prevents one if two blocks sharing "
                          "a case name+year+day were ever solved concurrently by two workers.",
                "citations": [
                    "network.py:540-602 (_create_smopf_solver: SolverFactory(solver_params.solver, "
                    "executable=solver_params.solver_path); options['linear_solver'] etc. come only from "
                    "the case-file solver.options / option_overrides)",
                    "network.py:563-573 (deterministic log path: os.path.join(network.logs_dir, value), "
                    "suffix = f'_{year}_{day}'[+ '_' + log_suffix]; solver.options['file_append'] = 'yes')",
                    "shared_resources_planning.py:7348 (distribution_network.logs_dir = "
                    "planning_problem.logs_dir -- SAME logs_dir object for every DSO node)",
                    "shared_resources_planning.py:7392 (transmission_network.logs_dir = "
                    "planning_problem.logs_dir -- same directory again)",
                    "data/SRP1/case33_1/case33_1_params.json, data/SRP1/case33_2/case33_2_params.json, "
                    "data/SRP1/case9/case9_params.json: solver.options.linear_solver = 'ma97', no thread cap",
                    "No OMP_NUM_THREADS / MKL_NUM_THREADS / TempfileManager reference anywhere in "
                    "shared_resources_planning.py, network.py or shared_energy_storage_data.py (grep, scoped "
                    "to these three files plus network_parameters.py)",
                ],
            },
            "esso_log_path_is_already_collision_hardened_by_contrast": {
                "answer": "shared_energy_storage_data.py's `_create_solver` (Addendum 5 / P5.15-F) DOES "
                          "stamp the ESSO log by node_id and cycle, resolves it against its own logs_dir, "
                          "and additionally checks os.path.exists() before writing, renaming to a "
                          "'_dupN' suffix rather than silently appending -- a materially stronger "
                          "isolation policy than network.py's DSO/TSO path has. This asymmetry (the ESSO "
                          "was hardened after a documented defect, network.py was not) is a direct "
                          "candidate explanation for why any future within-cycle DSO/TSO parallelism is "
                          "more exposed than ESSO parallelism would be.",
                "citations": [
                    "shared_energy_storage_data.py:1058-1107 (cycle/node stamp, os.path.exists check, "
                    "'_dupN' fallback, file_append='no')",
                    "shared_energy_storage_data.py:1067-1080 comment block, referencing "
                    "P5_15_G1_G4_BLOCKED.md and the historical os.chdir defect",
                ],
            },
            "tso_and_esso_coverage": {
                "answer": "The existing `*_parallel` functions cover DSO ONLY. There is no "
                          "`update_transmission_..._parallel` or `update_shared_energy_storages_..._parallel` "
                          "function anywhere in the repository (grep for '_parallel' across "
                          "shared_resources_planning.py, network.py, shared_energy_storage_data.py finds "
                          "only the two DSO functions and their model-construction counterpart). The TSO "
                          "solve (`update_transmission_coordination_model_and_solve`, unparameterized by "
                          "parallel_execution) and the ESSO solve "
                          "(`update_shared_energy_storages_coordination_model_and_solve`, looping over "
                          "3 nodes) both run purely sequentially today, in the parent process, in every "
                          "configuration.",
                "citations": [
                    "shared_resources_planning.py:5311-5316 (update_distribution_coordination_models_and_solve "
                    "dispatches to _parallel only for the DSO step)",
                    "grep -n \"_parallel\" shared_resources_planning.py network.py shared_energy_storage_data.py "
                    "-- only create_distribution_networks_models_parallel and "
                    "update_distribution_coordination_models_and_solve_parallel exist",
                ],
            },
            "block_level_parallelism_achieved_today": {
                "answer": "Even where the parallel path exists, it parallelizes over DISTRIBUTION NODES "
                          "(3 for SRP1: node 5/7/9), not over the 36 independent (node,year,day) DSO "
                          "blocks. Each submitted task (`update_and_solve_dso`) internally loops over "
                          "all 12 (year,day) pairs for its one node SEQUENTIALLY inside "
                          "`distribution_network.optimize(model, ...)`. So today's parallel path gives at "
                          "most 3-way concurrency for SRP1's DSO step (bounded further by "
                          "`max_workers = os.cpu_count() // 2`), never the 36-way (or 51-way "
                          "DSO+TSO+ESSO) concurrency Addendum 18 targets; achieving that requires "
                          "restructuring the unit of work from 'one node's whole year/day loop' to 'one "
                          "(node,year,day) block'.",
                "citations": [
                    "shared_resources_planning.py:5452-5499 (update_and_solve_dso: one process per "
                    "node_id, internal `for year ... for day ...` loop, single "
                    "`distribution_network.optimize(model, ...)` call covering all 12 year/day pairs)",
                    "network_data.py:54-68 (NetworkData.optimize loops over self.years/self.days "
                    "SEQUENTIALLY within one call, solving each (year,day) SMOPF one at a time)",
                ],
            },
            "admm_update_order_preserved": {
                "answer": "Yes for the DSO step itself (results are written into node_id-keyed dicts, "
                          "and the sequential vs parallel DSO functions are drop-in return-compatible), "
                          "but the parallel DSO path DOES NOT preserve the ordering/observability "
                          "guarantees the sequential path gives the rest of the ADMM orchestration: it "
                          "drops the `cycle` argument entirely (so no FrozenSMOPF snapshot capture, see "
                          "below) and never threads `failure_snapshot_callback` / "
                          "`pre_solve_snapshot_callback` through to `distribution_network.optimize`.",
                "citations": [
                    "shared_resources_planning.py:5429 (update_distribution_coordination_models_and_solve_parallel "
                    "signature has no `cycle` parameter, unlike the sequential twin at :5318)",
                    "shared_resources_planning.py:5489 (`res = distribution_network.optimize(model, "
                    "from_warm_start=from_warm_start)` -- no cycle, no callbacks -- contrast with the "
                    "sequential call at shared_resources_planning.py:5407-5413 which passes "
                    "`failure_snapshot_callback=snapshot_callback, pre_solve_snapshot_callback=success_snapshot_callback`)",
                ],
            },
            "recovery_policy_across_process_boundary": {
                "answer": "The tier-1/tier-2 recovery logic itself (`network.py:_run_smopf`, "
                          "`_is_recoverable_network_failure`) is pure, function-local and reads only its "
                          "own arguments -- it runs IDENTICALLY inside a ProcessPoolExecutor worker, "
                          "since `update_and_solve_dso` calls the SAME `distribution_network.optimize()` "
                          "-> `network.run_smopf()` -> `_run_smopf()` chain as the sequential path. "
                          "Recovery retries therefore DO work across the process boundary today. What "
                          "does NOT work: (1) FrozenSMOPF snapshot writing -- entirely un-wired in the "
                          "parallel path (no callbacks passed, see above), so a parallel-path failure on "
                          "node 7 would never produce the frozen pre-solve snapshot the sequential path "
                          "captures; (2) any diagnostics SINK object mutated inside the worker (the ESSO "
                          "has `solver_recovery_diagnostics`, appended inside `_optimize`) would be lost "
                          "across a process boundary because only `(node_id, res, model)` is returned, "
                          "not the mutated network/diagnostics object -- this is not currently exercised "
                          "for DSO (no equivalent diagnostics-sink parameter exists on the DSO recovery "
                          "path today), but it is the exact failure mode a future DSO diagnostics sink "
                          "would hit if added without also being returned from the worker.",
                "citations": [
                    "network.py:673-763 (_run_smopf: tier-1 cold retry, tier-2 mu_strategy=adaptive retry, "
                    "both pure functions of (network, model, params))",
                    "shared_resources_planning.py:5311-5316 vs :5429 (parallel entry point has no cycle arg)",
                    "shared_resources_planning.py:5372-5413 (sequential path wires `snapshot_callback` / "
                    "`success_snapshot_callback` into `distribution_network.optimize`; the parallel twin "
                    "at :5489 does not)",
                    "shared_energy_storage_data.py:67-95 (ESSO's `optimize` mutates "
                    "`self.solver_recovery_diagnostics` in place -- would not survive a process "
                    "boundary if the ESSO were ever parallelized, since only `results` is returned)",
                ],
            },
            "solve_profile_guard_across_process_boundary": {
                "answer": "SolveProfileGuard.install() monkeypatches `pyomo.opt.base.solvers.OptSolver.solve` "
                          "and `pyomo.opt.solver.shellcmd.SystemCallSolver._execute_command` AT THE CLASS "
                          "OBJECT LEVEL, in the CURRENT process's in-memory copy of the pyomo module. "
                          "`ProcessPoolExecutor` on this machine (macOS/Darwin) uses the 'spawn' start "
                          "method by default (fork is not the default on Darwin since Python 3.8), so "
                          "every worker process re-executes the Python interpreter from scratch and "
                          "re-imports pyomo fresh -- it NEVER inherits the parent's monkeypatched class "
                          "objects, regardless of whether the guard was installed before or after the "
                          "pool was created. A solve executed inside a worker therefore calls the "
                          "ORIGINAL (unpatched) OptSolver.solve: it is invisible to the guard -- neither "
                          "counted as `permitted_solve` nor as `blocked_solve`. Consequences for a "
                          "bounded-solve claim: (a) a guard armed for 0 solves in the parent process "
                          "would silently PASS even if all 36 DSO blocks solved via the parallel path in "
                          "child processes -- a false 'no solve' claim; (b) a guard armed for the full "
                          "51-solve profile would UNDERCOUNT by exactly the number of DSO solves routed "
                          "through the parallel path, and `verify()` would report 'too few', but for the "
                          "wrong structural reason (an invisible call site, not a code path that failed "
                          "to run) -- so the failure message itself would misdiagnose the cause unless "
                          "the reader already knows the parallel path exists. Any future guard covering a "
                          "persistent-worker design must install the SAME monkeypatch inside every "
                          "worker process (e.g. at worker-initializer time) and aggregate each worker's "
                          "own counts back to the parent, since class patches do not cross process "
                          "boundaries.",
                "citations": [
                    "p513_solve_profile_guard.py:50-78 (install(): OptSolver.solve = guarded_solve; "
                    "SystemCallSolver._execute_command = guarded_exec -- class-attribute assignment)",
                    "p513_solve_profile_guard.py:39-48 (_matching_frame uses sys._getframe on the CURRENT "
                    "process's call stack only)",
                    "Python multiprocessing default start method on macOS is 'spawn' since Python 3.8 "
                    "(platform behaviour, not repository-specific; verified against the environment's "
                    "documented interpreter, /Users/micaelsimoes/miniconda3/envs/opf_env_py311)",
                ],
            },
            "p56b_concurrency_crash_attribution": {
                "scope_of_search": "Searched: REVISION_CONTEXT.md (all 'P5.6-B' occurrences), "
                                    "P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md in full, "
                                    "P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md ('P5.6-B7' occurrence), "
                                    "and every p56*.py script in the repository root (grep for "
                                    "'ProcessPoolExecutor', 'multiprocessing', 'concurrent', 'contention', "
                                    "'SOLVER_CRASH'). Did not search git history/log or any branch other "
                                    "than the current checkout.",
                "finding": "The P5.6-B crash is NOT the same code path as this audit's subject "
                           "(`*_parallel` within-cycle block solves in shared_resources_planning.py). It "
                           "is CANDIDATE-EVALUATION-level concurrency in the derivative-free search "
                           "harness (p56a_oracle.py etc.): multiple FULL, independent 48-block sequential "
                           "ADMM evaluations (one whole SRP1 planning run per worker) run concurrently, "
                           "one worker per candidate. The report records: 'Resource contention is an "
                           "observed risk... W workers means W simultaneous IPOPT processes on top of "
                           "whatever threading MA97 uses. On this 8-core machine, a SOLVER_CRASH "
                           "(ApplicationError: Solver (ipopt) did not exit normally) was actually observed "
                           "during P5.6-A while several heavy processes ran concurrently' "
                           "(P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md, B7 section). No script in "
                           "the repository actually orchestrates that concurrent launch via "
                           "ProcessPoolExecutor/multiprocessing -- grep for those across all p56*.py files "
                           "returns nothing, so the concurrent load described was very likely produced by "
                           "manually/shell-launched separate Python invocations, not a harness this audit "
                           "can re-inspect as code. The report does NOT attribute the crash to a specific "
                           "mechanism (no mention of shared temp files, shared log paths, or .nl/.sol "
                           "collisions in connection with this crash anywhere in the searched material); "
                           "it is described only as generic IPOPT/MA97 resource contention on an 8-core "
                           "machine running multiple full evaluations at once. The crash was SUBSEQUENTLY "
                           "found not reproducible: 'se|ALL|x19 crashed IPOPT under both starts... "
                           "Withdrawn by P5.6-C. That crash was not reproducible... reclassified as a "
                           "transient solver failure' (same report). Direct, contemporaneous evidence that "
                           "this class of harness DID worry about log-file collision under concurrency: "
                           "p56a_oracle.py's `fresh_planning(eval_id)` docstring states explicitly 'IPOPT "
                           "is configured with file_append=\"yes\" and would otherwise append every "
                           "evaluation's log into the same file' and mitigates it by giving every eval_id "
                           "its own private logs_dir -- i.e. a real, named log-collision hazard from the "
                           "exact same file_append='yes' + shared-logs_dir mechanism this audit's Part 1 "
                           "flags in network.py, but at the evaluation layer, not the block layer, and "
                           "pre-empted rather than diagnosed as the SOLVER_CRASH's cause.",
                "citations": [
                    "P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md, section 'B7 -- search-cost "
                    "estimate' (SOLVER_CRASH / ApplicationError quote, 8-core contention statement)",
                    "P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md, section 'Verdict' > 'What B did "
                    "not settle' (the se|ALL|x19 SOLVER_CRASH) and its 'Withdrawn by P5.6-C' correction",
                    "p56a_oracle.py: fresh_planning(eval_id) docstring and body (per-eval_id logs_dir, "
                    "explicit file_append='yes' collision comment)",
                    "grep -rn 'ProcessPoolExecutor|multiprocessing' p56*.py p511*.py -> no matches in this "
                    "repository checkout",
                ],
            },
            "global_or_module_level_mutable_state": {
                "answer": "No module-level mutable dict/list/counter touched by the DSO solve path was "
                          "found in shared_resources_planning.py, network.py or "
                          "shared_energy_storage_data.py (targeted grep for module-level "
                          "dict()/list()/set()/global assignments plus manual reading of the parallel "
                          "functions). No production `os.chdir()` calls exist (only comments describing a "
                          "historical harness-level os.chdir defect that P5.15-F removed the need for). "
                          "`p56a_oracle.py` keeps one PROCESS-level cache, `_BASELINE` (a module-level "
                          "global, lazily populated), explicitly read-only after first population and "
                          "documented as never mutated -- safe under `ProcessPoolExecutor`'s 'spawn' start "
                          "method because each spawned process gets its own fresh, empty `_BASELINE`, but "
                          "would be a hazard under a 'fork'-based or thread-based pool sharing the "
                          "already-populated object across workers that then mutated it (they do not). "
                          "No other global cache of this kind was found in the audited production files.",
                "citations": [
                    "grep -n \"^[A-Za-z_].*= *(dict|list|set|\\{\\}|\\[\\])\" shared_resources_planning.py "
                    "network.py shared_energy_storage_data.py model_construction_helpers.py -- no hazardous "
                    "module-level mutable state found",
                    "grep -rn os.chdir *.py -- only comments in shared_energy_storage_data.py:1072 and "
                    "shared_resources_planning.py:47 referencing a historical, already-removed harness defect",
                    "p56a_oracle.py:_BASELINE / load_baseline() -- process-level lazy cache, read-only "
                    "after population, safe only because ProcessPoolExecutor uses 'spawn' on this platform",
                ],
            },
        }

        # ---------------------------------------------------------------
        # Write outputs
        # ---------------------------------------------------------------
        os.makedirs(OUT_DIR, exist_ok=True)

        outputs = {
            'provenance.json': provenance,
            'per_cycle_table.json': per_cycle_rows,
            'speedup_projection.json': speedup_rows,
            'summary.json': summary,
            'code_audit_findings.json': code_audit,
            'missing_data_notes.json': missing_data_notes,
        }
        written_paths = []
        for filename, payload in outputs.items():
            path = os.path.join(OUT_DIR, filename)
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
            written_paths.append(path)

        # Compact per-cycle table without the (verbose) per-block breakdown,
        # for quick human reading.
        compact_rows = [{k: v for k, v in row.items() if k != 'per_block'} for row in per_cycle_rows]
        compact_path = os.path.join(OUT_DIR, 'per_cycle_table_compact.json')
        with open(compact_path, 'w') as f:
            json.dump(compact_rows, f, indent=2, sort_keys=True, default=str)
        written_paths.append(compact_path)

        print(f'[OK] Wrote {len(written_paths)} output files to {OUT_DIR}')
        print(json.dumps(summary, indent=2, sort_keys=True))
        if missing_data_notes:
            print(f'[WARNING] {len(missing_data_notes)} missing-data notes -- see missing_data_notes.json')

        # ---- sha256 manifest of everything written (excluding itself) ----
        manifest = {}
        for path in sorted(written_paths):
            with open(path, 'rb') as f:
                manifest[os.path.relpath(path, OUT_DIR)] = hashlib.sha256(f.read()).hexdigest()
        manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f'[OK] Wrote sha256 manifest: {manifest_path}')

    finally:
        failures_check = guard.verify(expected_solves=0, expected_execs=0)
        guard.uninstall()
        if failures_check:
            raise RuntimeError(f'SolveProfileGuard failed at expected_solves=0: {failures_check}')
        print('[OK] SolveProfileGuard verified: 0 permitted solves, 0 blocked solves (zero-solve audit).')


if __name__ == '__main__':
    main()
