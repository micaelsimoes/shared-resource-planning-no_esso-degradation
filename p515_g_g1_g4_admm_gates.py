"""
P5.15 -- G1-G4 gate runner.

Authority: PLANNER_BRIEF_2026-09-13.md Step 1 "Gate" section, as amended by Addendum 1
(G5 reporting), Addendum 3 item 3 (G1 re-specified as a per-node reconciliation gate) and
Addendum 4 (per-cycle ESSO detector logging standing requirement, G1 confirmed as
specified).

This harness does NOT reimplement any production solve. It reuses, verbatim, the already
-repaired stage harnesses:
  * p514_n_instrumented_cstar.py  (control / perturbation arm: assert_capture_paths_exist,
    capture_esso, and every module-level constant -- S_INV, E_INV, INVEST_YEAR, BUDGET,
    REL, CAP, RHO, REFERENCE_RECOURSE, EFC_BINDING_THRESHOLD, PERMITTED);
  * p514_l_capacity_ladder.py     (ladder initialization-stage harness: main() called
    directly, with its module-level OUT redirected so it cannot collide with the
    committed P514L artifacts).

What this harness ADDS, because neither repaired harness captures it and the standing
requirement for ALL gates (Addendum 4) needs it:
  1. The FULL per-cycle ADMM trajectory (every row from p512_a_cold_rescaled_convergence
     .cycle_row), not only the terminal row -- "record the per-cycle trajectory by
     default".
  2. `shared_ess_data.esso_complementarity_diagnostics` -- production's OWN per-ESSO-solve
     log (shared_energy_storage_data._get_esso_complementarity_diagnostics, populated by
     _run_solver_attempt on every successful ESSO solve) of the ratio detector
     `max min(pch,pdch)/s_max` and the analytic leak estimate
     `2*N_periods*mu_final/(2*s_obj*eps)`, with mu_final/s_obj PARSED from that solve's
     own IPOPT log. Grouped into per-cycle rounds using the documented production fact
     that `update_shared_energy_storages_coordination_model_and_solve` is called exactly
     once per ADMM cycle (shared_resources_planning.py:2267) and once more before cycle 1
     for initialization (shared_resources_planning.py:~2126-2130), each call solving every
     active node once -- so consecutive blocks of `len(active_nodes)` entries in the flat
     diagnostics list correspond, in order, to: [init, cycle 1, cycle 2, ...]. This
     grouping is verified against the run's own `cycles_run` count before being reported,
     not assumed silently.

RULE ELEVEN: capture paths are asserted before any solve is attempted.

Writes ONLY new files, under data/SRP1/Results/P515G/ -- neither repaired harness's
default output directory is touched, because both collide with already-committed
artifacts (P514N: esso_models_control.pkl, n1_control.json, ...; P514L: ladder_s1.json,
...).

    python p515_g_g1_g4_admm_gates.py <gate>
    gate in {g1, g2, g4b, g3_init, g3_full}   (g4a == g1; run g1 twice for G4)
"""

import io
import itertools
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p512_a_cold_rescaled_convergence as A  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import p514_l_capacity_ladder as L  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G')
os.makedirs(OUT, exist_ok=True)


@contextmanager
def unique_esso_logs(log_dir):
    """Diagnostic-only monkeypatch, reverted on exit -- same convention as
    `p58_rescale.patched_admm_objectives()` (which this module already imports and
    uses). NOT a production-code change: `shared_energy_storage_data.py` on disk is
    untouched; this reassigns the module's `_create_solver` name for the lifetime of
    one process-local `with` block.

    WHY THIS IS NEEDED (found empirically, see the comment at its call site):
    `_create_solver` writes each node's IPOPT log to a bare relative filename with
    `file_append='yes'` and no per-solve uniqueness. Across a multi-cycle ADMM run the
    same node's log file therefore accumulates one "Objective"/"Complementarity"
    summary block per cycle, and `_parse_ipopt_barrier_terms`'s `re.search` (first
    match) silently returns the FIRST cycle's `mu_final`/`s_obj` for every later cycle.
    Verified on a 2-cycle probe before this fix: node 5's `mu_final` was IDENTICAL
    (3.131588719877726e-09) at the init round and both ADMM cycles, while the
    Var-based ratio detector correctly varied cycle to cycle.

    IMPLEMENTATION NOTE (superseding an earlier, WRONG attempt at this same fix):
    the first attempt isolated logs by `os.chdir`-ing the whole process into a
    private directory for the duration of the ADMM run. That crashed the run: TSO
    failure-snapshot capture (`shared_resources_planning.py:save_failed_tso_block` /
    `_save_frozen_network_block`, itself gated on `not solver_result_succeeded`, i.e.
    a genuine non-converged TSO block, which DID occur during the first G1 attempt)
    builds its own save directory from `transmission_network.results_dir`, a path
    that is RELATIVE in the deep-copied planning object and therefore only resolves
    correctly when the process CWD is still the repo root. `os.chdir` is therefore
    unsafe here, in EITHER direction, and is NOT used. Instead this passes each ESSO
    solve an ABSOLUTE `output_file` directly via `option_overrides`, which
    `_create_solver` merges into `options` BEFORE its own
    `if 'output_file' not in options:` default-assignment check
    (shared_energy_storage_data.py, `_create_solver`) -- so the override is honoured,
    the process CWD is never touched, and every node-round's IPOPT log is still a
    fresh, uniquely named file under `log_dir`.
    """
    original = SED._create_solver
    counter = itertools.count()
    os.makedirs(log_dir, exist_ok=True)

    def patched(model, params, from_warm_start=False, node_id=None, option_overrides=None,
                log_suffix=None):
        if node_id is not None:
            n = next(counter)
            suffix = f'{n:06d}' if not log_suffix else f'{n:06d}_{log_suffix}'
            option_overrides = dict(option_overrides) if option_overrides else {}
            option_overrides['output_file'] = os.path.join(
                log_dir, f'optim_log_node_{node_id}_{suffix}.txt')
        return original(model, params, from_warm_start=from_warm_start, node_id=node_id,
                         option_overrides=option_overrides, log_suffix=None)

    SED._create_solver = patched
    try:
        yield
    finally:
        SED._create_solver = original


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def assert_g_capture_paths(sed):
    """RULE ELEVEN, this gate's own addition: the per-cycle ESSO detector."""
    missing = []
    if not hasattr(sed, 'esso_complementarity_diagnostics'):
        missing.append('shared_ess_data.esso_complementarity_diagnostics missing')
    if not hasattr(SED, '_get_esso_complementarity_diagnostics'):
        missing.append('shared_energy_storage_data._get_esso_complementarity_diagnostics missing')
    if not hasattr(SED, 'EPS_ESSO_THROUGHPUT'):
        missing.append('shared_energy_storage_data.EPS_ESSO_THROUGHPUT missing')
    if missing:
        raise AssertionError('RULE ELEVEN (gate detector): capture paths missing -> '
                              + '; '.join(missing))
    return {'esso_complementarity_diagnostics_sink': True, 'asserted_before_run': True}


def _group_diagnostics_by_round(diagnostics, n_active_nodes, cycles_run):
    """Group the flat per-solve diagnostics list into [init, cycle 1, cycle 2, ...].

    Verifies the grouping rather than assuming it: reports whether
    len(diagnostics) == n_active_nodes * (cycles_run + 1) exactly (clean case, every
    ESSO solve on every node succeeded on every round) and falls back to reporting the
    flat list plus a per-node summary if it does not (e.g. a local ESSO failure skipped
    an append for one node-round).
    """
    expected_total = n_active_nodes * (cycles_run + 1)
    clean = (n_active_nodes > 0 and len(diagnostics) == expected_total)
    rounds = []
    if clean:
        for r in range(cycles_run + 1):
            block = diagnostics[r * n_active_nodes:(r + 1) * n_active_nodes]
            rounds.append({
                'round': 'init' if r == 0 else r,
                'entries': block,
                'ratio_max': max((e['complementarity_ratio_max'] for e in block
                                   if e['complementarity_ratio_max'] is not None), default=None),
                'bound_max': max((e['spurious_throughput_bound'] for e in block
                                   if e['spurious_throughput_bound'] is not None), default=None),
                'measured_max': max((e['spurious_throughput_measured'] for e in block
                                      if e['spurious_throughput_measured'] is not None), default=None),
            })
    per_node = {}
    for e in diagnostics:
        node = str(e['node_id'])
        acc = per_node.setdefault(node, {'ratio_max': None, 'bound_max': None,
                                          'measured_max': None, 'n_solves': 0})
        acc['n_solves'] += 1
        for key, src in (('ratio_max', 'complementarity_ratio_max'),
                          ('bound_max', 'spurious_throughput_bound'),
                          ('measured_max', 'spurious_throughput_measured')):
            v = e.get(src)
            if v is not None and (acc[key] is None or v > acc[key]):
                acc[key] = v
    return {
        'n_active_nodes': n_active_nodes, 'cycles_run': cycles_run,
        'expected_total_solves': expected_total, 'observed_total_solves': len(diagnostics),
        'grouping_clean': clean,
        'per_round': rounds if clean else None,
        'per_node_summary': per_node,
        'flat_diagnostics': diagnostics,
    }


def run_admm_arm(label, out_dir, k_override=None, investment_map=None):
    """One full cold ADMM arm through the production path, reusing p514_n's own
    module-level constants and capture helpers verbatim. `investment_map`, if given,
    overrides the uniform S_INV/E_INV assignment for specific node_ids (others left at
    N.S_INV/N.E_INV for the control/perturbation arms, or at the harness's default 0/0
    if this is a fresh candidate); pass a full dict {node_id: (s, e)} covering every
    active node to avoid ambiguity (this is what G3's full eval does)."""
    os.makedirs(out_dir, exist_ok=True)
    checklist = N.assert_capture_paths_exist()
    started = time.time()
    report = {'stage': 'P5.15 G1-G4', 'arm': label,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'rule_eleven_checklist': checklist}
    guard = SolveProfileGuard(N.PERMITTED, label=f'P5.15-G {label}').install()
    try:
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning(f'p515g_{label}')
            planning.params.admm.num_max_iters = N.CAP
            planning.params.admm.tol['objective']['rel'] = N.REL
            planning.shared_ess_data.params.budget = N.BUDGET
            RH.apply_rho_to_params(planning, N.RHO)
            RH.set_adaptive_penalty(planning, True)
            sed = planning.shared_ess_data
            detector_checklist = assert_g_capture_paths(sed)
            report['rule_eleven_checklist']['detector'] = detector_checklist

            if k_override is not None:
                for year in sed.years:
                    for ess in sed.shared_energy_storages[year]:
                        ess.cl_eff = k_override
            report['k_in_force'] = {str(y): getattr(sed.shared_energy_storages[y][0], 'cl_eff', None)
                                    for y in sed.years}

            candidate = planning.get_initial_candidate_solution()
            if investment_map is None:
                for node_id in sed.active_distribution_network_nodes:
                    candidate['investment'][node_id][N.INVEST_YEAR]['s'] = N.S_INV
                    candidate['investment'][node_id][N.INVEST_YEAR]['e'] = N.E_INV
                report['instance'] = {'s_mva': N.S_INV, 'e_mwh': N.E_INV, 'year': N.INVEST_YEAR,
                                       'assignment': 'uniform across active nodes (control/perturbation)'}
            else:
                for node_id, (s_val, e_val) in investment_map.items():
                    candidate['investment'][node_id][N.INVEST_YEAR]['s'] = s_val
                    candidate['investment'][node_id][N.INVEST_YEAR]['e'] = e_val
                report['instance'] = {'year': N.INVEST_YEAR, 'assignment': 'per-node',
                                       'investment_map': {str(k): v for k, v in investment_map.items()}}
            srp._rebuild_candidate_total_capacities(planning, candidate)

            n_active_nodes = len(sed.active_distribution_network_nodes)
            report['active_distribution_network_nodes'] = list(sed.active_distribution_network_nodes)

            # ---- IPOPT log isolation (NOT a production change) ----
            # `shared_energy_storage_data._create_solver` writes each node's IPOPT log to
            # a BARE RELATIVE filename (`optim_log_node_{node_id}.txt`, `file_append='yes'`)
            # with no `logs_dir` awareness -- unlike `network.py:520-521`, which resolves
            # `output_file` against `network.logs_dir` and is therefore already isolated.
            # `unique_esso_logs()` gives every ESSO solve an ABSOLUTE, unique `output_file`
            # via `option_overrides` (see its docstring for why `os.chdir` was tried first
            # and reverted: it crashed a genuine TSO non-convergence's failure-snapshot
            # capture, which builds its own save path from a RELATIVE
            # `transmission_network.results_dir`). The process CWD is never changed.
            ipopt_log_dir = os.path.join(out_dir, 'ipopt_logs', label)
            if os.path.exists(ipopt_log_dir):
                raise RuntimeError(f'refusing to reuse a non-fresh ipopt log dir: {ipopt_log_dir}')
            with R.patched_admm_objectives(), unique_esso_logs(ipopt_log_dir):
                _c, _results, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
    finally:
        guard.uninstall()
        report['wall_clock_s'] = time.time() - started

    rows = []
    prev_recourse = None
    for e in (state.get('admm_diagnostics') or []):
        row = A.cycle_row(e, prev_recourse)
        rows.append(row)
        prev_recourse = row.get('recourse')

    last = rows[-1] if rows else {}
    report['cycle_trajectory'] = rows  # full per-cycle table, not just the terminal row
    report.update({
        'cycles_run': len(rows), 'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'converged_at_cycle': next((r['cycle'] for r in rows if r['cycle_convergence']), None),
        'terminal_objective_change_abs': last.get('objective_change_abs'),
        'terminal_objective_tolerance': last.get('objective_tolerance'),
        'rule_ten_terminal_step_over_threshold': (
            last.get('objective_change_abs') / last.get('objective_tolerance')
            if last.get('objective_change_abs') and last.get('objective_tolerance') else None),
        'local_solve_failures': sum(1 for r in rows if r.get('local_solves_ok') is False),
    })
    report['esso_capture'] = N.capture_esso(models['esso'], sed)

    # ---- Addendum-4 standing requirement: per-cycle ESSO complementarity detector ----
    diagnostics = list(sed.esso_complementarity_diagnostics)
    report['esso_complementarity_diagnostics_by_round'] = _group_diagnostics_by_round(
        diagnostics, n_active_nodes, len(rows))

    pickle_path = os.path.join(out_dir, f'esso_models_{label}.pkl')
    _refuse_overwrite(pickle_path)
    try:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(models['esso'], handle)
        report['esso_models_pickle'] = {'path': os.path.relpath(pickle_path, REPO),
                                        'bytes': os.path.getsize(pickle_path)}
    except Exception as error:
        report['esso_models_pickle'] = {'error': f'{type(error).__name__}: {error}'}

    report['solve_profile'] = {'observed': dict(guard.counts),
                               'identity_holds': guard.counts['permitted_solve'] == 51 * len(rows) + 51}
    path = os.path.join(out_dir, f'g_{label}.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    efc_all = [v['efc_per_day_max'] for v in report['esso_capture'].values() if v['efc_per_day_max']]
    print(f"[P5.15-G {label}] recourse={report['recourse']} "
          f"cycles={report['cycles_run']} solves={guard.counts['permitted_solve']} "
          f"local_failures={report['local_solve_failures']} "
          f"wall={report['wall_clock_s']:.0f}s")
    print(f"   EFC/day max across nodes: {max(efc_all) if efc_all else None} "
          f"(threshold {N.EFC_BINDING_THRESHOLD})")
    print(f"   detector grouping clean: {report['esso_complementarity_diagnostics_by_round']['grouping_clean']} "
          f"(observed {len(diagnostics)} / expected {n_active_nodes * (len(rows) + 1)})")
    return report, path


def run_ladder_init(s_mva, out_dir):
    """Reuse p514_l_capacity_ladder.main() UNMODIFIED, redirecting its module-level OUT
    so the already-committed P514L artifacts (ladder_s1.json etc.) are never touched."""
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f'ladder_s{float(s_mva):g}.json')
    _refuse_overwrite(target)
    original_out = L.OUT
    L.OUT = out_dir
    try:
        L.main(s_mva)
    finally:
        L.OUT = original_out
    return target


def _acquire_exclusive_run_lock():
    """P5.15 Planner guard (harness-only, NOT production).

    This harness MUST NOT run concurrently with another copy of itself. The ESSO
    writes its IPOPT log to a BARE RELATIVE filename with `file_append='yes'`
    (`shared_energy_storage_data.py:1007-1014`), unlike `network.py:520-521`
    which resolves `output_file` against `network.logs_dir`. Two concurrent
    campaigns therefore interleave their solver output into the same file, and
    `_parse_ipopt_barrier_terms` (first match, not last) returns the WRONG
    `mu_final`/`s_obj` for every cycle -- silently producing a plausible but
    fabricated per-cycle detector trajectory.

    This has now happened three times: once destroying a G1 run, and twice when
    four gates (g1, g2, g4b, g3_full) were launched simultaneously. The results
    would have been contaminated, not merely slow. Fail loudly instead.
    """
    import atexit
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = open(lock_path).read().strip()
        except OSError:
            holder = 'unknown'
        raise SystemExit(
            f'REFUSING TO RUN: another copy of {os.path.basename(__file__)} holds '
            f'{lock_path} (pid/arg: {holder}).\n'
            'Concurrent runs corrupt the shared ESSO IPOPT logs and produce a '
            'FABRICATED per-cycle detector trajectory. Run the gates SEQUENTIALLY. '
            'If no such process exists, remove the lock file by hand.')
    os.write(fd, f'{os.getpid()} {" ".join(sys.argv[1:])}'.encode())
    os.close(fd)
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))


if __name__ == '__main__':
    _acquire_exclusive_run_lock()
    gate = sys.argv[1] if len(sys.argv) > 1 else None
    if gate == 'g1':
        run_admm_arm('control', OUT, k_override=None)
    elif gate == 'g2':
        run_admm_arm('k10000', OUT, k_override=10000.0)
    elif gate == 'g4b':
        run_admm_arm('control_rep2', OUT, k_override=None)
    elif gate == 'g3_init':
        for s in (1.00, 1.25, 1.62):
            run_ladder_init(s, os.path.join(OUT, 'ladder'))
    elif gate == 'g3_full':
        # node 7 only, others zero, per PLANNER task text.
        planning_probe = None  # active node ids discovered inside run_admm_arm via sed
        # discovered lazily: build investment_map after loading a fresh planning to read
        # the active node list, without solving anything.
        probe = O.fresh_planning('p515g_g3_full_probe')
        active_nodes = list(probe.shared_ess_data.active_distribution_network_nodes)
        del probe
        investment_map = {nid: (0.0, 0.0) for nid in active_nodes}
        if 7 not in investment_map:
            raise RuntimeError(f'node 7 not in active_distribution_network_nodes={active_nodes}')
        investment_map[7] = (1.62, 3.24)
        run_admm_arm('g3_full_node7', OUT, k_override=None, investment_map=investment_map)
    else:
        print(__doc__)
        sys.exit(1)
