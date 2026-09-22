"""
P5.15 Addendum 29 (spec v16 `after_review.memory_task`, repeated in spec v18 `memory_task`), task W32,
STEP M1 -- PROFILE the per-first-solve memory growth of ONE network block and classify it.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 29 ("resident memory before and after the first and
second solve of the same block, with load_solutions on/off and with NL symbol maps and solution objects
cleared after value extraction; classify the growth as one-time bookkeeping, inherent IPOPT/NL workspace,
or a per-solve leak"); data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json `memory_task`.
Evidence being explained: P515S44 scale_measurement paper_cycle_snapoff_r2 and Phase A table T5
(+0.221 GiB footprint per within-network first solve at paper scale).

WHAT THE PRODUCTION SOLVE PATH ALREADY DOES (read from the code, not assumed)
-----------------------------------------------------------------------------
`network._run_smopf_solver_attempt` calls `solver.solve(model, tee=..., load_solutions=False)`; with
`load_solutions=False` Pyomo 6.9.5 (`OptSolver.solve`) moves the NL writer's SymbolMap out of
`model.solutions.symbol_map` into `result._smap`. `network._run_smopf` then calls
`model.solutions.load_from(result)`, which (ModelSolutions.load_from / add_solution / select):
  * re-registers the SymbolMap, builds ONE `ModelSolution` whose `_entry` dicts hold, for EVERY
    variable and every active constraint, a tuple (component data, entry dict) -- the entry dict being
    the SAME dict object the SolverResults holds ({'Value': x, 'ipopt_zL_out': .., 'ipopt_zU_out': ..}
    per variable, {'Dual': y} per constraint) -- plus one entry per FIXED variable;
  * deletes the SymbolMap (delete_symbol_map=True) and sets `result._smap = None`;
  * `select(0)` copies values into the Vars and the import suffixes (dual, ipopt_zL_out, ipopt_zU_out).
The `ModelSolution` stays in `model.solutions.solutions` until the next load (clear=True replaces it),
and the SolverResults (with its full `solution` container) is RETURNED and kept by production in the
per-(year, day) results dict until the next cycle replaces it. Neither is read again by production:
the values live in the Vars and suffixes (repository search recorded in the report). So production is
ALREADY "load_solutions=False with values loaded explicitly"; the variants below are defined relative
to that.

VARIANTS (post-solve action applied by a pass-through wrapper around `network._run_smopf`, AFTER the
production call has returned -- i.e. after value extraction; production code is not modified)
  a  production as is (the returned SolverResults is kept by the caller, exactly as production keeps
     it in results[year][day]; `model.solutions` untouched).
  b  a + `result.solution.clear()` -- the housekeeping Pyomo's own load_solutions=True path performs
     after loading (`OptSolver.solve`: `_model.solutions.load_from(result, ...); result._smap_id = None;
     result.solution.clear()`). This is the closest faithful equivalent of "load_solutions on": the
     loading call is identical (`load_from` with select 0), only the SolverResults' solution container
     is dropped afterwards. `model.solutions` is untouched.
  c  a + `model.solutions.clear()` (drops the ModelSolution and any SymbolMap still registered) +
     `result.solution.clear()` + `result._smap = None`. Suffixes (dual, ipopt_z*_out, ipopt_z*_in) are
     NOT touched: the warm start needs them (`replace_warm_start_suffix`), recovery snapshots them, and
     the EXPORT suffixes are written into the next NL file.
  d  a + gc.collect() after each solve.
  e  c + gc.collect() after each solve.

MODES
  single    ONE DSO block (node 7, the first year x the first day) built by production's
            `create_distribution_networks_models_sequential` on the arm's planning object restricted to
            that node / year / day; solve 1 is that function's own (cold) initialization solve; solves 2
            and 3 are `NetworkData.optimize(models, from_warm_start=True)` on the SAME block -- the call
            production's cycle dispatch makes (`update_distribution_coordination_models_and_solve_
            sequential`), with the block unchanged between solves (no ADMM terms are added: the
            profiled quantity is the solve, not the ADMM preparation; declared limitation).
  sequence  FOUR DSO blocks (node 7, the first year x all four days): 4 cold first solves (production's
            initialization function), then two warm passes (`optimize(..., from_warm_start=True)`),
            12 solves -- the within-network first-solve increment of Phase A T5, and the
            second/third-solve increment of every block.
  long      ONE block as in `single`, 1 cold + 9 warm solves (variants a and e): the per-solve slope
            over solves 3..10 separates a leak from allocator noise.
  attribution  = single, variant a, plus a direct size walk of the bookkeeping objects after each solve
            (never used for the increment tables: the walk itself allocates).

MEASUREMENT (every checkpoint, 5 samples 0.1 s apart; median and range recorded)
  rss_self       psutil.Process().memory_info().rss (resident set, this process)
  footprint_self macOS phys_footprint via libproc proc_pid_rusage(RUSAGE_INFO_V0).ri_phys_footprint
                 (`p515_s44_scale_measurement.phys_footprint`, BY IMPORT; counts compressed pages)
  rss_children   RSS of live descendants (0 at every checkpoint: IPOPT has exited)
  allocated_blocks sys.getallocatedblocks() (live pymalloc blocks -- Python-object count proxy)
  IPOPT          a 20 ms sampler thread records the peak RSS of the IPOPT child during each solve;
                 RUSAGE_CHILDREN ru_maxrss is recorded after each solve.
A watchdog (`p515_s44_scale_measurement.Watchdog`, rss_tree, limit 20 GiB, thrashing guard) runs in
every child for machine protection; its samples are kept.

SOLVES: an armed `SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED)` per child, declared before
the child runs (single 3, sequence 12) and verified EXACTLY against declared + recovery attempts (every
`network._run_smopf_solver_attempt` call is counted by a pass-through wrapper, with its log suffix).

EXACT COMMAND (repo root; attached; both streams captured; alone):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_memory_profile.py \\
        > data/SRP1/Results/P515S49/memory_profile_launch.log 2>&1
Output (write-once): data/SRP1/Results/P515S49/memory_profile/
"""

import argparse
import gc
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import psutil

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s44_scale_measurement as S44  # noqa: E402 -- measurement + planning helpers, BY IMPORT (no guard at import)

STAGE = 'P5.15 Addendum 29 W32 M1 -- per-solve memory profile of one network block (classification)'
SCHEMA = 'p515_s49_memory_profile_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 29',
             'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json memory_task',
             'Planner task W32 (memory: profile, classify, fix bookkeeping, gate)']
OUT_PARENT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S49')
DEFAULT_LABEL = 'memory_profile'
OUT_DIR = os.path.join(OUT_PARENT, DEFAULT_LABEL)   # set from --label in main()
OWN_LOCK = os.path.join(REPO, '.p515_s49_memory_profile.lock')
OTHER_LOCKS = (os.path.join(REPO, '.p515_g_gate.lock'), os.path.join(REPO, '.p515_s44_campaign.lock'),
               os.path.join(REPO, '.p515_s44_scale_measurement.lock'))
PYTHON = sys.executable
GIB = 1 << 30
MIB = 1 << 20
NODE = 7
VARIANTS = ('a', 'b', 'c', 'd', 'e')
VARIANT_TEXT = {
    'a': 'production as is (SolverResults kept by the caller; model.solutions untouched)',
    'b': 'a + result.solution.clear() (Pyomo load_solutions=True housekeeping)',
    'c': 'a + model.solutions.clear() + result.solution.clear() + result._smap=None (suffixes kept)',
    'd': 'a + gc.collect()',
    'e': 'c + gc.collect()',
}
LABEL = [DEFAULT_LABEL]
DECLARED_SOLVES = {'single': 3, 'attribution': 3, 'sequence': 12, 'long': 10}
WARM_PASSES = {'single': 2, 'attribution': 2, 'sequence': 2, 'long': 9}
WATCHDOG_LIMIT_GIB = 20.0
N_SAMPLES = 5
SAMPLE_GAP_S = 0.1
IPOPT_SAMPLER_S = 0.02

# The run plan, fixed before anything runs (order: per instance, reps outer, variants inner).
PLAN = []
for _instance in ('srp1', 'paper'):
    for _rep in (1, 2, 3):
        for _variant in VARIANTS:
            PLAN.append({'instance': _instance, 'mode': 'single', 'variant': _variant, 'rep': _rep})
    PLAN.append({'instance': _instance, 'mode': 'attribution', 'variant': 'a', 'rep': 1})
    for _rep in (1, 2):
        for _variant in ('a', 'e'):
            PLAN.append({'instance': _instance, 'mode': 'sequence', 'variant': _variant, 'rep': _rep})
    for _variant in ('a', 'e'):
        PLAN.append({'instance': _instance, 'mode': 'long', 'variant': _variant, 'rep': 1})


def child_id(entry):
    return f"{entry['instance']}_{entry['mode']}_{entry['variant']}_r{entry['rep']}"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W32-M1] {msg}', flush=True)


# ======================================================================================
#  CHILD
# ======================================================================================
class IpoptSampler:
    """Peak RSS of this process's descendants (the IPOPT child) while a solve runs."""

    def __init__(self):
        self.peak = 0
        self.n = 0
        self._halt = threading.Event()
        self._thread = None
        self._me = psutil.Process(os.getpid())

    def _run(self):
        while not self._halt.is_set():
            total = 0
            try:
                for kid in self._me.children(recursive=True):
                    try:
                        total += kid.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            self.peak = max(self.peak, total)
            self.n += 1
            self._halt.wait(IPOPT_SAMPLER_S)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._thread.join(timeout=2)


def measure(label):
    samples = []
    for i in range(N_SAMPLES):
        mem = S44.tree_memory(os.getpid()) or {}
        samples.append({'rss_self': mem.get('rss_self'), 'footprint_self': mem.get('footprint_self'),
                        'rss_children': mem.get('rss_children'), 'n_children': mem.get('n_children'),
                        'allocated_blocks': sys.getallocatedblocks()})
        if i < N_SAMPLES - 1:
            time.sleep(SAMPLE_GAP_S)
    out = {'label': label, 't_utc': _utc(), 'samples': samples}
    for key in ('rss_self', 'footprint_self', 'rss_children', 'allocated_blocks'):
        vals = [s[key] for s in samples if s[key] is not None]
        out[key] = int(statistics.median(vals)) if vals else None
        out[f'{key}_range'] = (max(vals) - min(vals)) if vals else None
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    out['sys_available'] = vm.available
    out['swap_used'] = sw.used
    out['ru_maxrss_self'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['ru_maxrss_children'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return out


def _sizer(roots, seen, stop):
    """Bytes reachable from `roots` (sys.getsizeof), not crossing Pyomo components (`stop`), ids in
    `seen` counted once across calls. Floats/ints are reported separately (they are shared with Var
    values and suffix entries, so clearing the containers does not free them)."""
    containers = 0
    leaves = 0
    n = 0
    stack = list(roots)
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, stop):
            continue
        n += 1
        size = sys.getsizeof(obj)
        if isinstance(obj, (float, int, bool)) or obj is None:
            leaves += size
            continue
        containers += size
        if isinstance(obj, str):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        d = getattr(obj, '__dict__', None)
        if isinstance(d, dict):
            stack.append(d)
    return {'container_bytes': containers, 'leaf_bytes': leaves, 'n_objects': n}


def attribution(model, result):
    """Direct size of the bookkeeping objects (attribution mode only; allocates)."""
    from pyomo.core.base.component import ComponentBase
    import pyomo.environ as pe
    stop = (ComponentBase,)
    out = {}
    ms = model.solutions
    out['model_solutions'] = {
        'n_solutions': len(ms.solutions), 'n_symbol_maps': len(ms.symbol_map),
        'entries': ({k: len(v) for k, v in ms.solutions[0]._entry.items()} if ms.solutions else None)}
    out['result'] = {'present': result is not None}
    if result is not None:
        out['result']['n_solutions'] = len(result.solution)
        out['result']['smap_present'] = result.__dict__.get('_smap') is not None
        if len(result.solution):
            sol = result.solution(0)
            out['result']['entries'] = {k: len(getattr(sol, k)) for k in ('variable', 'constraint', 'objective',
                                                                          'problem')}
    seen = set()
    out['size_model_solutions_alone'] = _sizer([ms.solutions, ms.symbol_map], set(), stop)
    out['size_result_solution_alone'] = (_sizer([result.solution], set(), stop) if result is not None else None)
    out['size_union_model_solutions_then_result'] = {
        'model_solutions': _sizer([ms.solutions, ms.symbol_map], seen, stop),
        'result_solution_additional': (_sizer([result.solution], seen, stop) if result is not None else None)}
    del seen
    suffixes = {}
    for name in ('dual', 'ipopt_zL_out', 'ipopt_zU_out', 'ipopt_zL_in', 'ipopt_zU_in'):
        comp = getattr(model, name, None)
        suffixes[name] = len(comp) if comp is not None else None
    out['suffix_lengths'] = suffixes
    out['model_counts'] = {
        'var_data': sum(1 for _ in model.component_data_objects(pe.Var)),
        'constraint_data_active': sum(1 for _ in model.component_data_objects(pe.Constraint, active=True))}
    return out


def post_solve_action(variant, model, result):
    done = []
    if variant == 'b' and result is not None:
        result.solution.clear()
        done.append('result.solution.clear()')
    if variant in ('c', 'e'):
        model.solutions.clear()
        done.append('model.solutions.clear()')
        if result is not None:
            result.solution.clear()
            result.__dict__['_smap'] = None
            done.append('result.solution.clear(); result._smap = None')
    if variant in ('d', 'e'):
        done.append(f'gc.collect() -> {gc.collect()}')
    return done


def child(args):
    entry = {'instance': args.instance, 'mode': args.mode, 'variant': args.variant, 'rep': args.rep}
    cid = child_id(entry)
    cdir = os.path.join(OUT_DIR, 'children', cid)
    os.makedirs(cdir)
    S44._child_env_check()
    launch = json.load(open(os.path.join(OUT_DIR, 'launch.json')))
    wd = S44.Watchdog(cdir, cid, limit_bytes=int(WATCHDOG_LIMIT_GIB * GIB))
    wd.start()
    stages = S44.StageLog(os.path.join(cdir, f'stages_{cid}.jsonl'), wd)

    from p513_solve_profile_guard import SolveProfileGuard
    import p514_n_instrumented_cstar as N
    guard = SolveProfileGuard(N.PERMITTED, label=f'P5.15 W32 memory profile {cid}').install()
    declared = DECLARED_SOLVES[args.mode]
    record = {'schema': SCHEMA, 'stage': STAGE, 'child_id': cid, **entry,
              'variant_text': VARIANT_TEXT[args.variant], 'declared_solves_before_run': declared,
              'pid': os.getpid(), 'utc_start': _utc(), 'thread_caps': {k: os.environ.get(k) for k in S44.THREAD_CAP_ENV}}
    checkpoints = []
    solves = []
    attempts = []
    try:
        with stages.stage('import production modules'):
            import network as network_module
            import shared_resources_planning as srp
            import p515_g_g1_g4_admm_gates as G
            import p515_s44_campaign_harness as H  # noqa: F401 -- d_configuration_check uses it
            O = G.O
        inst_launch = {'derived_case': launch['derived_cases'][args.instance], 'instance': args.instance}
        planning0 = S44.read_planning_from_derived_case(inst_launch, cdir, stages)
        checksum = S44.inject_oracle_baseline(O, planning0, inst_launch)
        record['scenario_checksum'] = checksum
        record['planning_dimensions'] = S44.planning_dimensions(planning0)
        prov = S44.provenance_record(planning0, args.instance, checksum)
        record['provenance'] = prov
        if [f for f in prov['gate_failures'] if f['identity'] != 'scenario checksum']:
            raise RuntimeError(f'provenance: non-canonical solver identity: {prov["gate_failures"]}')
        construct_report = {}
        with stages.stage('_construct_arm_planning (C*)'):
            planning, sed, candidate = G._construct_arm_planning(
                f's49_mem_{cid}', os.path.join(cdir, 'arm'), construct_report, investment_map=S44.C_STAR,
                eval_id=f'p515s49_{LABEL[0]}_{cid}', num_max_iters_override=S44.BUILD_CAP, apply_rho=False)
        record['d_configuration_checks'] = S44.d_configuration_check(H, planning, sed, candidate, construct_report,
                                                                     S44.BUILD_CAP)
        record['snapshot_setting'] = S44.apply_snapshot_setting(planning, 'off', None)
        dn = planning.distribution_networks[NODE]
        year0 = list(dn.years)[0]
        days = list(dn.days)
        keep_days = days if args.mode == 'sequence' else days[:1]
        dn.years = {year0: dn.years[year0]}
        dn.days = {d: dn.days[d] for d in keep_days}
        record['block_selection'] = {'node': NODE, 'network': dn.name, 'year': year0, 'days': keep_days,
                                     'method': 'the arm planning object restricted to this node/year/days '
                                               '(NetworkData.years / .days), then production builders'}
        cv, _dv = srp.create_admm_variables(planning)
        checkpoints.append(measure('before build'))

        # ---- pass-through wrappers (never behaviour changes) --------------------------------------
        orig_run_smopf = network_module._run_smopf
        orig_attempt = network_module._run_smopf_solver_attempt
        solve_ctx = {'pass': 'init (cold)'}

        def counting_attempt(net, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):
            attempts.append({'network': net.name, 'year': net.year, 'day': net.day,
                             'from_warm_start': from_warm_start, 'log_suffix': log_suffix})
            return orig_attempt(net, model, params, from_warm_start=from_warm_start,
                                option_overrides=option_overrides, log_suffix=log_suffix)

        def wrapped_run_smopf(net, model, params, from_warm_start=False):
            k = len(solves) + 1
            pre = measure(f'solve {k} pre')
            children_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            t0 = time.time()
            with IpoptSampler() as sampler:
                result = orig_run_smopf(net, model, params, from_warm_start=from_warm_start)
            wall = time.time() - t0
            raw = measure(f'solve {k} post (raw)')
            attr = attribution(model, result) if args.mode == 'attribution' else None
            done = post_solve_action(args.variant, model, result)
            post = measure(f'solve {k} post (after action)')
            import helper_functions as HF
            solves.append({
                'k': k, 'pass': solve_ctx['pass'], 'network': net.name, 'year': net.year, 'day': net.day,
                'from_warm_start': from_warm_start, 'wall_s': round(wall, 3),
                'succeeded': HF.solver_result_succeeded(result), 'summary': HF.solver_result_summary(result),
                'ipopt_peak_rss_sampled': sampler.peak, 'ipopt_sampler_n': sampler.n,
                'ru_maxrss_children_before': children_before,
                'ru_maxrss_children_after': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                'action': done, 'pre': pre, 'post_raw': raw, 'post_action': post, 'attribution': attr})
            print(f'[W32-M1 {cid}] solve {k} {solve_ctx["pass"]} {net.name} {net.year} {net.day} warm={from_warm_start} '
                  f'ok={solves[-1]["succeeded"]} wall={wall:.1f}s rss {pre["rss_self"]/MIB:.1f}->{raw["rss_self"]/MIB:.1f}'
                  f'->{post["rss_self"]/MIB:.1f} MiB fp {pre["footprint_self"]/MIB:.1f}->{raw["footprint_self"]/MIB:.1f}'
                  f'->{post["footprint_self"]/MIB:.1f} MiB ipopt_peak {sampler.peak/MIB:.1f} MiB', flush=True)
            return result

        network_module._run_smopf = wrapped_run_smopf
        network_module._run_smopf_solver_attempt = counting_attempt
        try:
            with stages.stage('build + solve pass 1 (production create_distribution_networks_models_sequential)'):
                dso_models, res_init = srp.create_distribution_networks_models_sequential(
                    {NODE: dn}, cv, candidate['total_capacity'])
            holder = res_init[NODE]
            n_warm_passes = WARM_PASSES[args.mode]
            for p in range(n_warm_passes):
                solve_ctx['pass'] = f'warm pass {p + 1}'
                with stages.stage(f'warm pass {p + 1} (NetworkData.optimize from_warm_start=True)'):
                    holder = dn.optimize(dso_models[NODE], from_warm_start=True)  # replaces, as production does
        finally:
            network_module._run_smopf = orig_run_smopf
            network_module._run_smopf_solver_attempt = orig_attempt
        checkpoints.append(measure('after all solves (results and models alive)'))
        blocks = [dso_models[NODE][y][d] for y in dso_models[NODE] for d in dso_models[NODE][y]]
        record['block_sizes'] = [{'vars': sum(1 for _ in b.component_data_objects(__import__('pyomo.environ').environ.Var)),
                                  'constraints_active': sum(1 for _ in b.component_data_objects(
                                      __import__('pyomo.environ').environ.Constraint, active=True))} for b in blocks]
        del blocks
        del holder, res_init, dso_models
        gc.collect()
        checkpoints.append(measure('after del models/results + gc.collect'))
    except BaseException as error:  # noqa: BLE001 -- recorded
        guard.uninstall()
        wd.stop()
        record.update({'status': 'error', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc(), 'guard_counts': dict(guard.counts),
                       'checkpoints': checkpoints, 'solves': solves, 'attempts': attempts})
        with open(os.path.join(cdir, 'child_record.json'), 'w') as handle:
            json.dump(record, handle, indent=1, default=str)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    guard.uninstall()
    recovery_attempts = sum(1 for a in attempts if a['log_suffix'])
    reconciled = declared + recovery_attempts
    failures = guard.verify(reconciled)
    final = wd.stop()
    record.update({
        'status': 'complete' if not failures and len(solves) == declared else 'solve_count_mismatch',
        'guard': {'permitted': [list(p) for p in N.PERMITTED], 'counts': dict(guard.counts),
                  'declared': declared, 'recovery_attempts': recovery_attempts, 'reconciled': reconciled,
                  'verify_failures': failures, 'n_wrapped_run_smopf_calls': len(solves)},
        'attempts': attempts, 'checkpoints': checkpoints, 'solves': solves,
        'watchdog': {'peak': wd.peak, 'final': final, 'thresholds': wd.thresholds()},
        'utc_end': _utc()})
    with open(os.path.join(cdir, 'child_record.json'), 'w') as handle:
        json.dump(record, handle, indent=1, default=str)
    print(f'[W32-M1 {cid}] status={record["status"]} solves={guard.counts["permitted_solve"]}/{reconciled}', flush=True)
    return 0 if record['status'] == 'complete' else 1


# ======================================================================================
#  ANALYSIS
# ======================================================================================
def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {'n': len(vals), 'median': statistics.median(vals), 'min': min(vals), 'max': max(vals),
            'range': max(vals) - min(vals)}


def analyse(records):
    """Increment tables (MiB). Per solve k: d_raw = post_raw - pre (the solve itself, incl. production's
    load), d_action = post_action - post_raw (the variant's action), d_total = post_action - pre."""
    table = {}
    for rec in records:
        if rec.get('status') != 'complete' or rec['mode'] == 'attribution':
            continue
        key = (rec['instance'], rec['mode'], rec['variant'])
        tab = table.setdefault(key, {})
        build = rec['solves'][0]['pre']
        before = rec['checkpoints'][0]
        tab.setdefault('build', {}).setdefault('rss_self', []).append((build['rss_self'] - before['rss_self']) / MIB)
        tab['build'].setdefault('footprint_self', []).append(
            (build['footprint_self'] - before['footprint_self']) / MIB)
        for s in rec['solves']:
            slot = tab.setdefault(f"solve{s['k']}", {})
            for m in ('rss_self', 'footprint_self', 'allocated_blocks'):
                div = MIB if m != 'allocated_blocks' else 1
                slot.setdefault(f'{m}_raw', []).append((s['post_raw'][m] - s['pre'][m]) / div)
                slot.setdefault(f'{m}_action', []).append((s['post_action'][m] - s['post_raw'][m]) / div)
                slot.setdefault(f'{m}_total', []).append((s['post_action'][m] - s['pre'][m]) / div)
            slot.setdefault('ipopt_peak_rss_mib', []).append(s['ipopt_peak_rss_sampled'] / MIB)
            slot.setdefault('wall_s', []).append(s['wall_s'])
            slot.setdefault('within_checkpoint_rss_range_mib', []).append(s['post_action']['rss_self_range'] / MIB)
            slot.setdefault('within_checkpoint_fp_range_mib', []).append(s['post_action']['footprint_self_range'] / MIB)
        fin = rec['checkpoints'][-1]
        alive = rec['checkpoints'][-2]
        tab.setdefault('release_after_del_gc', {}).setdefault('rss_self', []).append(
            (fin['rss_self'] - alive['rss_self']) / MIB)
        tab['release_after_del_gc'].setdefault('footprint_self', []).append(
            (fin['footprint_self'] - alive['footprint_self']) / MIB)
    out = {}
    for key, tab in table.items():
        out['|'.join(key)] = {slot: {m: {'values': v, 'stats': _stats(v)} for m, v in d.items()}
                              for slot, d in tab.items()}
    return out


def leak_slopes(records):
    """Least-squares slope (MiB per solve) of the post-action checkpoint over solves 3..N of the
    `long` children -- a per-solve leak shows as a positive slope above the checkpoint noise."""
    out = {}
    for rec in records:
        if rec.get('status') != 'complete' or rec['mode'] != 'long':
            continue
        res = {}
        pts = [s for s in rec['solves'] if s['k'] >= 3]
        for m in ('rss_self', 'footprint_self', 'allocated_blocks'):
            xs = [s['k'] for s in pts]
            div = MIB if m != 'allocated_blocks' else 1
            ys = [s['post_action'][m] / div for s in pts]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
            res[m] = {'slope_per_solve': slope, 'n_points': len(xs), 'first_k': xs[0], 'last_k': xs[-1],
                      'residual_max_abs': max(abs(r) for r in resid), 'series': ys,
                      'total_change_k3_to_last': ys[-1] - ys[0]}
        out[f"{rec['instance']}|{rec['variant']}"] = res
    return out


def write_markdown(payload, path):
    L = [f'# {STAGE}', '', f"Generated {payload['utc_end']}; git HEAD {payload['git_head']}.", '',
         'Units: MiB (2^20 bytes); allocated_blocks = sys.getallocatedblocks() count. '
         'd_raw = after production solve+load minus before; d_action = the variant action alone; '
         'd_total = d_raw + d_action. Median over repeats [min, max].', '']
    L.append('## Variants')
    for v, t in VARIANT_TEXT.items():
        L.append(f'- **{v}**: {t}')
    L.append('')
    for key, tab in payload['increments'].items():
        L.append(f'### {key}')
        L.append('')
        L.append('| slot | rss d_raw | rss d_action | rss d_total | footprint d_raw | footprint d_action | '
                 'footprint d_total | alloc_blocks d_total | IPOPT peak RSS | wall s |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')

        def f(slot, m):
            st = (tab.get(slot, {}).get(m) or {}).get('stats')
            if not st:
                return '-'
            if 'blocks' in m:
                return f"{st['median']:.0f} [{st['min']:.0f}, {st['max']:.0f}]"
            return f"{st['median']:.1f} [{st['min']:.1f}, {st['max']:.1f}]"
        for slot in sorted(tab, key=lambda s: (not s.startswith('build'), s.startswith('release'),
                                               int(re.sub(r'\D', '', s) or 0))):
            if slot in ('build', 'release_after_del_gc'):
                L.append(f"| {slot} | - | - | {f(slot, 'rss_self')} | - | - | {f(slot, 'footprint_self')} | - | - | - |")
                continue
            L.append(f"| {slot} | {f(slot, 'rss_self_raw')} | {f(slot, 'rss_self_action')} | {f(slot, 'rss_self_total')} | "
                     f"{f(slot, 'footprint_self_raw')} | {f(slot, 'footprint_self_action')} | "
                     f"{f(slot, 'footprint_self_total')} | {f(slot, 'allocated_blocks_total')} | "
                     f"{f(slot, 'ipopt_peak_rss_mib')} | {f(slot, 'wall_s')} |")
        L.append('')
    L.append('## Leak slope (long mode: 1 cold + 9 warm solves of one block; slope over solves 3..10)')
    L.append('')
    L.append('| instance / variant | rss MiB/solve | footprint MiB/solve | alloc_blocks/solve | rss change k3->k10 MiB | rss residual max MiB |')
    L.append('|---|---|---|---|---|---|')
    for key, res in payload['leak_slopes_long_mode'].items():
        L.append(f"| {key} | {res['rss_self']['slope_per_solve']:.2f} | {res['footprint_self']['slope_per_solve']:.2f} | "
                 f"{res['allocated_blocks']['slope_per_solve']:.0f} | {res['rss_self']['total_change_k3_to_last']:.1f} | "
                 f"{res['rss_self']['residual_max_abs']:.1f} |")
    L.append('')
    L.append('## Attribution (direct size walk, variant a, one child per instance)')
    L.append('')
    for inst, att in payload['attribution'].items():
        L.append(f'### {inst}')
        L.append('```')
        L.append(json.dumps(att, indent=1, default=str)[:6000])
        L.append('```')
    L.append('')
    L.append('## Children')
    for c in payload['children']:
        L.append(f"- {c['child_id']}: exit {c['exit_code']}, status {c.get('status')}, solves "
                 f"{c.get('guard_observed')}/{c.get('guard_reconciled')}, wall {c['wall_s']:.1f} s")
    with open(path, 'w') as handle:
        handle.write('\n'.join(L) + '\n')


# ======================================================================================
#  PARENT
# ======================================================================================
def parent(only=None):
    from p513_solve_profile_guard import SolveProfileGuard
    SolveProfileGuard(permitted=(), label='W32 M1 parent (never solves)').install()
    if os.path.exists(OUT_DIR):
        print(f'REFUSED: output directory exists (write-once): {OUT_DIR}', file=sys.stderr)
        return 2
    held = [p for p in OTHER_LOCKS + (OWN_LOCK,) if os.path.exists(p)]
    others = S44._other_harness_processes()
    if held or others:
        print(f'REFUSED: machine not alone; locks held={held} harness processes={others}', file=sys.stderr)
        return 2
    fd = os.open(OWN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'started_utc': _utc()}, handle)
    started = time.time()
    try:
        os.makedirs(os.path.join(OUT_DIR, 'case'))
        os.makedirs(os.path.join(OUT_DIR, 'children'))
        derived = {}
        for instance in ('srp1', 'paper'):
            case, spec, changes = S44.derive_case(instance, {})
            path = os.path.join(OUT_DIR, 'case', f'SRP1__{instance}.json')
            with open(path, 'w') as handle:
                json.dump(case, handle, indent='\t')
            derived[instance] = {'path': os.path.relpath(path, REPO), 'sha256': S44.sha256_file(path),
                                 'source': S44.SOURCE_CASE_REL, 'source_sha256': S44.sha256_file(S44.SOURCE_CASE),
                                 'changes_vs_source': changes, 'instance_definition': spec}
        launch = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'utc': _utc(),
                  'git_head': S44._git(['rev-parse', 'HEAD']),
                  'git_tracked_changes': S44._git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
                  'script': os.path.basename(__file__), 'script_sha256': S44.sha256_file(os.path.abspath(__file__)),
                  'interpreter': PYTHON, 'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
                  'label': LABEL[0], 'only': sorted(only) if only else None,
                  'derived_cases': derived, 'plan': PLAN, 'declared_solves_per_child': DECLARED_SOLVES,
                  'variants': VARIANT_TEXT, 'node': NODE, 'watchdog_limit_gib': WATCHDOG_LIMIT_GIB,
                  'measurement': {'samples_per_checkpoint': N_SAMPLES, 'gap_s': SAMPLE_GAP_S,
                                  'ipopt_sampler_s': IPOPT_SAMPLER_S,
                                  'rss': 'psutil memory_info().rss', 'footprint': 'libproc ri_phys_footprint'},
                  'machine': {'total_memory_bytes': psutil.virtual_memory().total, 'cpu_count': os.cpu_count(),
                              'available_at_launch': psutil.virtual_memory().available,
                              'swap_used_at_launch': psutil.swap_memory().used}}
        S44._write_once_json(os.path.join(OUT_DIR, 'launch.json'), launch)
        env = dict(os.environ)
        env.update(S44.THREAD_CAP_ENV)
        children = []
        plan = [e for e in PLAN if (only is None or child_id(e) in only)]
        for entry in plan:
            cid = child_id(entry)
            others = S44._other_harness_processes()
            others = [o for o in others if 'p515_s49_memory_profile' not in o['cmdline']]
            if others:
                raise RuntimeError(f'another harness process appeared: {others}')
            _log(f'child {cid} ...')
            t0 = time.time()
            logdir = os.path.join(OUT_DIR, 'children_logs')
            os.makedirs(logdir, exist_ok=True)
            with open(os.path.join(logdir, f'{cid}_stdout.log'), 'w') as out, \
                    open(os.path.join(logdir, f'{cid}_stderr.log'), 'w') as err:
                rc = subprocess.run([PYTHON, '-u', os.path.abspath(__file__), '--child', '--label', LABEL[0],
                                     '--instance', entry['instance'], '--mode', entry['mode'],
                                     '--variant', entry['variant'], '--rep', str(entry['rep'])],
                                    cwd=REPO, env=env, stdout=out, stderr=err).returncode
            rec_path = os.path.join(OUT_DIR, 'children', cid, 'child_record.json')
            rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
            children.append({'child_id': cid, 'exit_code': rc, 'status': rec.get('status'),
                             'guard_observed': (rec.get('guard') or {}).get('counts', {}).get('permitted_solve'),
                             'guard_reconciled': (rec.get('guard') or {}).get('reconciled'),
                             'wall_s': time.time() - t0})
            _log(f'child {cid}: exit {rc} status {rec.get("status")} wall {time.time() - t0:.1f}s')
            if rc == 97:
                raise RuntimeError(f'watchdog abort in {cid}; stopping the plan')
        records = []
        for entry in plan:
            p = os.path.join(OUT_DIR, 'children', child_id(entry), 'child_record.json')
            if os.path.exists(p):
                records.append(json.load(open(p)))
        attribution_out = {}
        for rec in records:
            if rec['mode'] == 'attribution' and rec.get('status') == 'complete':
                attribution_out[rec['instance']] = {
                    'block_sizes': rec.get('block_sizes'),
                    'per_solve': [{'k': s['k'], 'from_warm_start': s['from_warm_start'], 'attribution': s['attribution'],
                                   'rss_raw_mib': (s['post_raw']['rss_self'] - s['pre']['rss_self']) / MIB,
                                   'fp_raw_mib': (s['post_raw']['footprint_self'] - s['pre']['footprint_self']) / MIB}
                                  for s in rec['solves']]}
        payload = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'launch': launch,
                   'git_head': launch['git_head'], 'children': children,
                   'all_children_complete': all(c['status'] == 'complete' for c in children),
                   'increments': analyse(records), 'leak_slopes_long_mode': leak_slopes(records),
                   'attribution': attribution_out,
                   'wall_s': time.time() - started, 'utc_end': _utc()}
        S44._write_once_json(os.path.join(OUT_DIR, 'memory_profile.json'), payload)
        write_markdown(payload, os.path.join(OUT_DIR, 'memory_profile.md'))
        manifest = {}
        for root, _dirs, files in os.walk(OUT_DIR):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                manifest[os.path.relpath(fpath, REPO)] = S44.sha256_file(fpath)
        S44._write_once_json(os.path.join(OUT_DIR, 'manifest_sha256.json'), manifest)
        _log(f'done: {len(children)} children, all complete={payload["all_children_complete"]}, '
             f'wall {time.time() - started:.1f}s')
        return 0 if payload['all_children_complete'] else 1
    finally:
        os.remove(OWN_LOCK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--label', default=DEFAULT_LABEL)
    parser.add_argument('--only', default=None, help='comma-separated child ids (smoke runs)')
    parser.add_argument('--instance', choices=('srp1', 'paper'))
    parser.add_argument('--mode', choices=tuple(DECLARED_SOLVES))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--rep', type=int)
    args = parser.parse_args()
    global OUT_DIR
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.label):
        raise SystemExit(f'bad label {args.label!r}')
    LABEL[0] = args.label
    OUT_DIR = os.path.join(OUT_PARENT, args.label)
    if args.child:
        sys.exit(child(args))
    sys.exit(parent(set(args.only.split(',')) if args.only else None))


if __name__ == '__main__':
    main()
