"""
P5.15 Addendum 38 (task W39) -- GATE 3 of 3: the SINGLE-BLOCK 2 x 2 A/B at
alpha in {0, 0.5, large}, BEFORE the pilot, reporting alpha * pibar_t against each DSO's
INTERNAL FLEXIBILITY PRICE per hour.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[2]` -- "single-block 2x2 A/B at alpha in {0, 0.5, large} BEFORE the pilot; report
alpha*pibar_t against each DSO's internal flexibility price per hour";
PLANNER_BRIEF_2026-09-13.md Addendum 38 ("dispersion is a threshold outcome").

WHY THIS GATE EXISTS. Dispersion under row 18 is a THRESHOLD outcome: a DSO deviates from
its commitment when deviating is cheaper than the alternative it has, and its alternative
is its own internal flexibility, priced at `network.cost_flex[s_m][t]` (row 2). So the
decisive comparison is the PRICE RATIO alpha * pibar_t / c_flex_t, hour by hour and DSO by
DSO -- if alpha * pibar_t sits far above c_flex_t at every hour, the pilot will show no
dispersion whatever alpha is chosen, and the pilot would measure nothing. This gate
reports that ratio (zero-solve, for EVERY DSO and every hour) and, on ONE DSO block,
measures the realized dispersion at the three alphas, so the ratio's prediction can be
checked against behaviour before the 7-10 h pilot is spent.

WHAT IS SOLVED, and what is not. Each arm builds the SELECTED DSO's blocks through
production's own builder (`shared_resources_planning.create_distribution_networks_models`
with the arm's alpha) and lets that builder solve them -- the DSO's standalone
(uncoordinated) SMOPF, which is what the ADMM initialization solves. That is ONE solve per
(year, day) block per arm, declared before the run and guard-verified EXACTLY. NO ADMM
cycle runs here; no TSO or ESSO block is built; the dispersion measured is the DSO's own
response to the premium at its uncoordinated point, NOT a converged coordinated result.

ARMS: alpha = 0 (the free-deviation reference: row 18 is not constructed at all),
alpha = 0.50 (the pilot value), alpha = ALPHA_LARGE (the pinned limit).

GATE ITEMS ARE SCOPED PER ARM (CLAUDE.md stage template):
  * every arm: its blocks solved; row 18 present iff alpha > 0; solves reconciled.
  * alpha = 0 arm: row 18 structurally absent (no charge component at all).
  * alpha = large arm: dispersion below the declared tolerance.
  * across arms: dispersion non-increasing in alpha.
The PRICE TABLE is reported for every DSO and hour and is not gated -- it is the
measurement this gate exists to produce.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_single_block_ab.py --label <write-once-label> \\
        > data/SRP1/Results/P515S51/single_block_ab_launch_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/single_block_ab/<label>/
Exit 0 on PASS, 1 on FAIL, 2 on a precondition refusal.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import psutil  # noqa: E402

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402

PERMITTED = tuple(tuple(p) for p in N.PERMITTED)
GUARD = SolveProfileGuard(PERMITTED, label='P5.15 W39 gate 3 -- single-block 2x2 A/B').install()

import pyomo.environ as pe  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p56a_oracle as O  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from helper_functions import solver_result_succeeded  # noqa: E402

STAGE = ('P5.15 Addendum 38 W39 gate 3 -- single-block 2x2 A/B at alpha in {0, 0.5, large}; '
         'alpha*pibar_t against each DSO\'s internal flexibility price per hour')
SCHEMA = 'p515_s51_single_block_ab_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 38',
             'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json gates[2]']

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'single_block_ab')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s51_single_block_ab.lock')

OVERRIDE_YEARS = {'2025': 5}
OVERRIDE_MARKET_SCENARIOS = 2
OVERRIDE_OPERATION_SCENARIOS = 2
INSTANCE_LABEL = 's51_2x2_single_block'
SELECTED_NODE = 7                 # the node the F2 demonstration and the ladder use
ALPHA_LARGE = 1000.0
ALPHAS = (0.0, 0.50, ALPHA_LARGE)
DISPERSION_ZERO_TOL_MW = 1.0e-2   # declared BEFORE the run; the alpha = large arm only

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
THREAD_CAP_ENV = dict(S.THREAD_CAP_ENV)
PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'model_construction_helpers.py', 'shared_resources_planning.py', 'network.py',
    'admm_parameters.py', 'p56a_oracle.py', 'p515_s44_scale_measurement.py',
    os.path.basename(__file__))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(message):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W39-gate3] {message}', flush=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'<git failed: {error}>'


def check_preconditions(out_dir):
    failures = []
    for path in (OWN_LOCK_PATH, G.CAMPAIGN_LOCK_PATH, os.path.join(REPO, '.p515_g_gate.lock'),
                 os.path.join(REPO, '.p515_s44_scale_measurement.lock'),
                 os.path.join(REPO, '.p515_s51_2x2_limit_gate.lock')):
        if os.path.exists(path):
            failures.append(f'lock file exists: {path}')
    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')
    me = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me or 'python' not in cmd:
            continue
        if S.HARNESS_PATTERN.search(cmd):
            failures.append(f'another p51x/p514 harness is alive: pid={proc.info["pid"]} {cmd[:200]}')
    status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN))
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')
    bad_env = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        failures.append(f'thread caps not in force (export them before launching): {bad_env}')
    return failures


def capture_path_checklist(planning):
    """CLAUDE.md rule eleven: every quantity this gate's specification requires must have a
    capture path, verified BEFORE the run."""
    node = next(iter(planning.distribution_networks))
    network_data = planning.distribution_networks[node]
    year = next(iter(network_data.years))
    day = next(iter(network_data.days))
    network = network_data.network[year][day]
    return {
        'dispersion_metric_callable': callable(getattr(srp, '_get_local_interface_dispersion', None)),
        'row18_wiring_callable': callable(getattr(MCH, 'add_scenario_commitment_terms', None)),
        'expected_market_price_callable': callable(getattr(MCH, 'expected_market_price', None)),
        'network_carries_cost_flex': hasattr(network, 'cost_flex') and len(network.cost_flex) > 0,
        'network_carries_cost_energy_p': hasattr(network, 'cost_energy_p') and len(network.cost_energy_p) > 0,
        'builder_accepts_premium_alpha': 'premium_alpha' in __import__('inspect').signature(
            srp.create_distribution_networks_models).parameters,
        'selected_node_present': SELECTED_NODE in planning.distribution_networks,
    }


def price_table(planning):
    """ZERO-SOLVE, for EVERY DSO and every hour: alpha * pibar_t against the DSO's own
    internal flexibility price c_flex_t (row 2's price -- the alternative to deviating).
    Reported per market scenario AND probability-weighted, since both prices are
    scenario-dependent above one market scenario."""
    table = {}
    for node_id, network_data in planning.distribution_networks.items():
        for year in network_data.years:
            for day in network_data.days:
                network = network_data.network[year][day]
                n_periods = len(network.cost_energy_p[0])
                pibar = [MCH.expected_market_price(network, p) for p in range(n_periods)]
                flex_expected = []
                for p in range(n_periods):
                    total = 0.0
                    for s_m in range(len(network.prob_market_scenarios)):
                        total += network.prob_market_scenarios[s_m] * network.cost_flex[s_m][p]
                    flex_expected.append(total)
                rows = []
                for p in range(n_periods):
                    entry = {'period': p, 'pibar': pibar[p], 'c_flex_expected': flex_expected[p],
                             'c_flex_by_market_scenario': [network.cost_flex[s_m][p] for s_m in
                                                           range(len(network.prob_market_scenarios))],
                             'pi_by_market_scenario': [network.cost_energy_p[s_m][p] for s_m in
                                                       range(len(network.prob_market_scenarios))]}
                    for alpha in ALPHAS:
                        entry[f'alpha_{alpha}_premium'] = alpha * pibar[p]
                        entry[f'alpha_{alpha}_over_c_flex'] = (
                            (alpha * pibar[p]) / flex_expected[p] if flex_expected[p] else None)
                    rows.append(entry)
                table[f'DSO:{node_id}:{network.name}:{year}:{day}'] = {
                    'n_periods': n_periods,
                    'pibar_min': min(pibar), 'pibar_max': max(pibar),
                    'c_flex_expected_min': min(flex_expected), 'c_flex_expected_max': max(flex_expected),
                    'n_hours_premium_below_c_flex': {
                        str(alpha): sum(1 for p in range(n_periods)
                                        if alpha * pibar[p] < flex_expected[p])
                        for alpha in ALPHAS},
                    'rows': rows,
                }
    return table


def run_arm(alpha, out_root, holder):
    arm = f'alpha_{alpha}'
    record = holder.setdefault(arm, {'arm': arm, 'alpha': alpha})
    eval_id = f'p515s51_ab_{os.path.basename(out_root)}_{arm}'
    record['eval_id'] = eval_id
    before = GUARD.counts['permitted_solve']
    t0 = time.time()

    planning = O.fresh_planning(eval_id)
    planning.params.admm.interface_deviation_premium = {
        'alpha': float(alpha), 'floor': None, 'source': f'W39 gate 3 arm {arm!r}'}
    record['alpha_applied'] = dict(planning.params.admm.interface_deviation_premium)

    consensus_vars, _dual = srp.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    record['candidate_label'] = 'x = 0 (no shared-ESS investment) -- the initial candidate'

    network_data = planning.distribution_networks[SELECTED_NODE]
    models, results = srp.create_distribution_networks_models(
        {SELECTED_NODE: network_data}, consensus_vars, candidate['total_capacity'],
        parallel_execution=False,
        premium_alpha=planning.params.admm.interface_deviation_premium['alpha'],
        premium_floor=planning.params.admm.interface_deviation_premium['floor'])

    blocks = {}
    for year in network_data.years:
        for day in network_data.days:
            model = models[SELECTED_NODE][year][day]
            network = network_data.network[year][day]
            result = results[SELECTED_NODE][year][day]
            dispersion = srp._get_local_interface_dispersion(model, network)
            blocks[f'{year}:{day}'] = {
                'solved': bool(solver_result_succeeded(result)),
                'termination': str(getattr(getattr(result, 'solver', None), 'termination_condition', None)),
                'row18_wired': hasattr(model, 'row18_deviation_charge'),
                'row18_alpha_on_model': (float(pe.value(model.row18_alpha))
                                         if hasattr(model, 'row18_alpha') else None),
                'voltage_pin_wired': hasattr(model, 'scenario_voltage_pin'),
                'objective_function_rule_value': float(
                    pe.value(MCH.objective_function_rule(model, network_data.params))),
                'model_objective_value': float(pe.value(model.objective.expr)),
                'interface_settlement': float(pe.value(model.interface_settlement)),
                'interface_settlement_contracted': float(pe.value(model.interface_settlement_contracted)),
                'interface_settlement_deviation': float(pe.value(model.interface_settlement_deviation)),
                'voltage_mismatch': srp._get_local_scenario_voltage_mismatch(model, network),
                'dispersion': dispersion,
            }
    record['blocks'] = blocks
    record['n_blocks'] = len(blocks)
    record['all_solved'] = all(b['solved'] for b in blocks.values())
    record['rms_mw_max_over_blocks'] = max(b['dispersion']['p']['rms_mw'] for b in blocks.values())
    record['max_abs_mw_over_blocks'] = max(b['dispersion']['p']['max_abs_mw'] for b in blocks.values())
    record['total_charge'] = sum(b['dispersion']['row18_charge'] for b in blocks.values())
    record['solves'] = GUARD.counts['permitted_solve'] - before
    record['wall_s'] = time.time() - t0
    record['objective_convention'] = (
        'block-local, UNWEIGHTED by year/day/discount; `objective_function_rule` is the '
        'quantity Q(x) is built from (it carries row 18 and the voltage pin, and the '
        'voltage pin is subtracted back out at the recourse level, Addendum 38 (D))')
    return record


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    args = parser.parse_args()

    out_root = os.path.join(OUT_ROOT, args.label)
    failures = check_preconditions(out_root)
    if failures:
        for failure in failures:
            print(f'[W39-gate3 PRECONDITION FAILED] {failure}', file=sys.stderr)
        return EXIT_REFUSED
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: lock held {OWN_LOCK_PATH}', file=sys.stderr)
        return EXIT_REFUSED
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'label': args.label, 'started_utc': _utc()}, handle)

    started = time.time()
    holder = {}
    os.environ.update(THREAD_CAP_ENV)
    try:
        os.makedirs(out_root)
        case_dir = os.path.join(out_root, 'case')
        os.makedirs(case_dir)
        case, spec, changes = S.derive_case('srp1', {
            'years': OVERRIDE_YEARS,
            'num_market_scenarios': OVERRIDE_MARKET_SCENARIOS,
            'num_operation_scenarios': OVERRIDE_OPERATION_SCENARIOS})
        case_path = os.path.join(case_dir, 'SRP1__s51_2x2_ab.json')
        with open(case_path, 'w') as handle:
            json.dump(case, handle, indent='\t')

        launch = {
            'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
            'label': args.label, 'instance': INSTANCE_LABEL, 'instance_definition': spec,
            'derived_case': {'path': os.path.relpath(case_path, REPO),
                             'sha256': _sha256_file(case_path),
                             'source': os.path.relpath(S.SOURCE_CASE, REPO),
                             'source_sha256': _sha256_file(S.SOURCE_CASE),
                             'changes_vs_source': changes},
            'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__),
            'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
            'alphas': list(ALPHAS), 'selected_node': SELECTED_NODE,
            'dispersion_zero_tol_mw_declared_before_the_run': DISPERSION_ZERO_TOL_MW,
            'guard_permitted': [list(p) for p in PERMITTED],
            'started_utc': _utc(), 'pid': os.getpid(),
        }

        stages = S.StageLog(os.path.join(out_root, 'stages.jsonl'),
                            S.Watchdog(out_root, 's51ab', limit_bytes=int(12 * (1 << 30))))
        stages.wd.start()
        planning0 = S.read_planning_from_derived_case(launch, out_root, stages)
        launch['scenario_checksum'] = S.inject_oracle_baseline(O, planning0, launch)
        launch['planning_dimensions'] = S.planning_dimensions(planning0)
        provenance = S.provenance_record(planning0, INSTANCE_LABEL, launch['scenario_checksum'])
        launch['provenance'] = provenance
        non_checksum = [f for f in provenance['gate_failures'] if f['identity'] != 'scenario checksum']
        if non_checksum:
            raise RuntimeError(f'provenance: non-canonical identity: {non_checksum}')

        checklist = capture_path_checklist(planning0)
        launch['capture_path_checklist_asserted_before_run'] = checklist
        if not all(checklist.values()):
            raise RuntimeError(f'capture-path checklist failed: '
                               f'{[k for k, v in checklist.items() if not v]}')

        # declared BEFORE the run: one solve per (year, day) block of the selected DSO, per arm
        network_data = planning0.distribution_networks[SELECTED_NODE]
        n_blocks = len(list(network_data.years)) * len(list(network_data.days))
        declared_total = len(ALPHAS) * n_blocks
        launch['declared_solve_profile'] = {
            'n_blocks_per_arm': n_blocks, 'n_arms': len(ALPHAS),
            'declared_total_strict': declared_total,
            'derivation': 'one standalone DSO SMOPF solve per (year, day) block per arm',
        }

        # the price table is ZERO-SOLVE and is written BEFORE any arm runs
        launch['price_table'] = price_table(planning0)
        launch_path = os.path.join(out_root, 'launch.json')
        G._refuse_overwrite(launch_path)
        with open(launch_path, 'w') as handle:
            json.dump(launch, handle, indent=1, default=str)
        _log(f'declared {declared_total} solves ({len(ALPHAS)} arms x {n_blocks} blocks); '
             f'price table written for {len(launch["price_table"])} DSO blocks')

        # The legacy run lock (`.p515_g_gate.lock`). It is released by
        # `_acquire_exclusive_run_lock`'s own atexit handler -- the module exposes no
        # explicit release entry point, and the committed gates do not release it either.
        G._acquire_exclusive_run_lock()
        for alpha in ALPHAS:
            _log(f'arm alpha = {alpha}')
            record = run_arm(alpha, out_root, holder)
            _log(f"arm alpha={alpha}: solved={record['all_solved']} "
                 f"rms_mw={record['rms_mw_max_over_blocks']} charge={record['total_charge']}")

        guard_failures = GUARD.verify(declared_total)
        rms = {a: holder[f'alpha_{a}']['rms_mw_max_over_blocks'] for a in ALPHAS}
        monotone = all(rms[ALPHAS[i]] >= rms[ALPHAS[i + 1]] - DISPERSION_ZERO_TOL_MW
                       for i in range(len(ALPHAS) - 1))

        gate_items = {
            'every_arm_solved_every_block': all(r['all_solved'] for r in holder.values()),
            'solve_count_verified_exactly': not guard_failures,
            'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
            'row18_present_iff_alpha_positive': all(
                all(b['row18_wired'] == (r['alpha'] > 0.0) for b in r['blocks'].values())
                for r in holder.values()),
            'alpha_zero_arm_has_no_row18_component': all(
                b['row18_wired'] is False and b['dispersion']['row18_charge'] == 0.0
                for b in holder['alpha_0.0']['blocks'].values()),
            'large_arm_dispersion_below_the_declared_tolerance': (
                rms[ALPHA_LARGE] <= DISPERSION_ZERO_TOL_MW),
            'dispersion_non_increasing_in_alpha': monotone,
            'objective_equals_objective_function_rule_on_every_block': all(
                abs(b['model_objective_value'] - b['objective_function_rule_value'])
                <= 1e-9 * max(1.0, abs(b['objective_function_rule_value']))
                for r in holder.values() for b in r['blocks'].values()),
        }
        gate_pass = all(gate_items.values())

        payload = {
            **launch,
            'finished_utc': _utc(), 'wall_clock_s': time.time() - started,
            'arms': holder,
            'dispersion_by_alpha_mw': rms,
            'charge_by_alpha': {a: holder[f'alpha_{a}']['total_charge'] for a in ALPHAS},
            'gate_scope': ('the "no row 18 component" item applies to the alpha = 0 arm ONLY and '
                           'the "dispersion below tolerance" item to the alpha = large arm ONLY; '
                           'the alpha = 0.50 arm is the measurement, not a gated reference '
                           '(CLAUDE.md: scope a gate per arm)'),
            'not_a_result': ('these are UNCOORDINATED standalone DSO solves at a single candidate, '
                             'not ADMM results; they establish the threshold behaviour before the '
                             'pilot, and are not comparable with any coordinated figure'),
            'solve_profile': {'declared_total_strict': declared_total,
                              'observed': GUARD.counts['permitted_solve'],
                              'counts': dict(GUARD.counts), 'verify_failures': guard_failures},
            'gate_items': gate_items, 'gate_pass': gate_pass,
        }
        gate_path = os.path.join(out_root, 'gate.json')
        G._refuse_overwrite(gate_path)
        with open(gate_path, 'w') as handle:
            json.dump(payload, handle, indent=1, default=str)

        manifest = {}
        for root, _dirs, fnames in os.walk(out_root):
            for fname in sorted(fnames):
                fpath = os.path.join(root, fname)
                manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
        manifest_path = os.path.join(out_root, 'manifest_sha256.json')
        G._refuse_overwrite(manifest_path)
        with open(manifest_path, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        for key, value in gate_items.items():
            _log(f'   {key}: {value}')
        _log(f'GATE_PASS={gate_pass}; dispersion by alpha (MW) = {rms}')
        return EXIT_OK if gate_pass else EXIT_ERROR
    finally:
        GUARD.uninstall()
        if os.path.exists(OWN_LOCK_PATH):
            os.remove(OWN_LOCK_PATH)


if __name__ == '__main__':
    sys.exit(main())
