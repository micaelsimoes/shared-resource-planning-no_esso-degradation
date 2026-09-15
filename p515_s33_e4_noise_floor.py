"""P5.15 Step 3.2, Addendum 14, E4 -- local-solver noise floor.

Binding specification: data/SRP1/Results/P515S33/frozen_e4_noise_floor_spec_v1_35f54dfe.json
(commit 254048f4). Read it fully before touching this file; where this docstring and the
spec differ, the spec wins.

Purpose (spec `purpose`): measure the iterate difference produced by the local IPOPT
solver alone, at fixed inputs, in the Boyd normalized units of each channel (V, PF, ESS),
and derive `eps_abs` from it by the rule frozen in the spec's `derivation_rule`.

Precedent followed (task instruction): `p512_y_tso_replay.py` and
`p512_x_comparator_replay.py`. Both unpickle a preserved PRE-SOLVE Pyomo model, take
missing context (network object, params) from `p56a_oracle.fresh_planning`, and build the
solver with production `network._create_smopf_solver(net, model, params,
from_warm_start=...)`, then call `solver.solve(model, tee=False, load_solutions=False,
keepfiles=True)` directly. This harness does the same for each of the 8 permitted solves.

One process, not eight (spec `replay_method`: "fresh process state per replay: unpickle
the fixture anew for EVERY solve ... Running the replays in one process is fine, provided
each solve uses a freshly unpickled model and a freshly created solver"). Both conditions
hold here: `pickle.load` is called fresh from disk immediately before every solve (see
`run_variant`), and `network._create_smopf_solver` is called fresh every time, returning a
brand-new `pyomo.opt.SolverFactory` instance with its own `.options` dict. The actual NLP
solve is external-process work regardless of how many solves happen in the calling Python
process: `pyomo.opt.solver.shellcmd.SystemCallSolver._execute_command` launches a new
`/usr/local/bin/ipopt` subprocess for every `solver.solve()` call (verified below by the
`SolveProfileGuard`, which also counts `_execute_command` invocations and requires them to
equal the solve count exactly). There is therefore no in-process solver state that could
leak between solves in this one-process design; this is recorded as the justification for
not falling back to eight separate processes, per the spec's own conditional ("If one
process turns out not to be identical to separate processes, record that and use separate
processes.") -- no such non-identity was found or is mechanistically plausible here.

Objective convention: the value recorded and compared is the model's ACTIVE Pyomo
Objective at the loaded solution -- `p58_rescaled_admm_objective` for both fixtures (the
current production ADMM-rescaled formulation; confirmed identical active-objective name on
both fixtures before any solve). This is the quantity IPOPT actually minimizes; it is NOT
`model.objective` (present but inactive on both fixtures) and NOT a net/gross operational
recourse. The objective name actually found is recorded in every solve record.

No production, case-file, or spec edit. `SolveProfileGuard` (`p513_solve_profile_guard`)
is armed for the whole script at exactly 8 permitted solves (2 fixtures x {W1,W2,C1,T1}),
checked exactly via `guard.verify(8)`. Output goes to a NEW directory
`data/SRP1/Results/P515S33/E4/`, refused if it already exists.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_s33_e4_noise_floor.py
"""
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import pyomo.environ as pe  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import network as N  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as RESCALE  # noqa: E402 -- reused only for read_log_since (log parsing)
from helper_functions import solver_result_succeeded  # noqa: E402
from shared_resources_planning import _shared_ess_admm_normalization_mva  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

SPEC_PATH = os.path.join(REPO, 'data/SRP1/Results/P515S33/frozen_e4_noise_floor_spec_v1_35f54dfe.json')
OUT_DIR = os.path.join(REPO, 'data/SRP1/Results/P515S33/E4')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
RESULTS_PATH = os.path.join(OUT_DIR, 'e4_noise_floor_results.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'e4_manifest_sha256.json')

FIXTURE_KIND = {
    'matched_success_TSO_case9_2025_Summer_cycle7.pkl': 'TSO',
    'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl': 'DSO',
}

VARIANTS = ('W1', 'W2', 'C1', 'T1')
VARIANT_WARM_START = {'W1': True, 'W2': True, 'C1': False, 'T1': True}
PAIRS = (('W1', 'W2'), ('W1', 'C1'), ('W1', 'T1'))
CHANNELS = ('V', 'PF', 'ESS')
T1_DIVISOR = 10.0

PERMITTED = [(os.path.basename(__file__), '_do_solve')]
EXPECTED_SOLVES = 8

S32_REFERENCE_STEPS = {
    'V': 9.2e-05, 'PF': 1.7e-04, 'ESS': 5.5e-05,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# solve (the one permitted call site the guard allows)
# ===========================================================================
def _do_solve(solver, model):
    t0 = time.time()
    result = solver.solve(model, tee=False, load_solutions=False, keepfiles=True)
    return result, time.time() - t0


# ===========================================================================
# per-block channel extraction, in the Boyd normalized units the spec defines
# ===========================================================================
def extract_tso_channels(model, net, planning, floor_mva):
    s_base = net.baseMVA
    year, day = net.year, net.day
    channels = {c: [] for c in CHANNELS}
    detail = []
    for dn in range(len(net.active_distribution_network_nodes)):
        node_id = net.active_distribution_network_nodes[dn]
        dso_net = planning.distribution_networks[node_id].network[year][day]
        interface_rating = dso_net.get_interface_branch_rating()
        e_idx = net.get_shared_energy_storage_idx(node_id)
        s_rated_mva = pe.value(model.shared_es_s_rated_fixed[e_idx]) * s_base
        rating_mva = _shared_ess_admm_normalization_mva(s_rated_mva, floor_mva)
        for p in model.periods:
            v = pe.value(model.expected_interface_vmag[dn, p])
            pf_p = pe.value(model.expected_interface_pf_p[dn, p]) * s_base / interface_rating
            pf_q = pe.value(model.expected_interface_pf_q[dn, p]) * s_base / interface_rating
            ess_p = pe.value(model.expected_shared_ess_p[e_idx, p]) * s_base / (2.0 * rating_mva)
            ess_q = pe.value(model.expected_shared_ess_q[e_idx, p]) * s_base / (2.0 * rating_mva)
            channels['V'].append(v)
            channels['PF'].append(pf_p)
            channels['PF'].append(pf_q)
            channels['ESS'].append(ess_p)
            channels['ESS'].append(ess_q)
            detail.append({'node_id': node_id, 'dn': dn, 'period': p, 'V': v, 'PF_p': pf_p,
                            'PF_q': pf_q, 'ESS_p': ess_p, 'ESS_q': ess_q,
                            'interface_rating_mva': interface_rating,
                            'shared_ess_rating_mva': rating_mva, 's_rated_mva': s_rated_mva})
    return channels, detail


def extract_dso_channels(model, net, planning, floor_mva):
    s_base = net.baseMVA
    ref_node_id = net.get_reference_node_id()
    e_idx = net.get_shared_energy_storage_idx(ref_node_id)
    s_rated_mva = pe.value(model.shared_es_s_rated_fixed[e_idx]) * s_base
    rating_mva = _shared_ess_admm_normalization_mva(s_rated_mva, floor_mva)
    interface_rating = net.get_interface_branch_rating()
    channels = {c: [] for c in CHANNELS}
    detail = []
    for p in model.periods:
        v = pe.value(model.expected_interface_vmag[p])
        pf_p = pe.value(model.expected_interface_pf_p[p]) * s_base / interface_rating
        pf_q = pe.value(model.expected_interface_pf_q[p]) * s_base / interface_rating
        ess_p = pe.value(model.expected_shared_ess_p[p]) * s_base / (2.0 * rating_mva)
        ess_q = pe.value(model.expected_shared_ess_q[p]) * s_base / (2.0 * rating_mva)
        channels['V'].append(v)
        channels['PF'].append(pf_p)
        channels['PF'].append(pf_q)
        channels['ESS'].append(ess_p)
        channels['ESS'].append(ess_q)
        detail.append({'node_id': ref_node_id, 'period': p, 'V': v, 'PF_p': pf_p, 'PF_q': pf_q,
                        'ESS_p': ess_p, 'ESS_q': ess_q, 'interface_rating_mva': interface_rating,
                        'shared_ess_rating_mva': rating_mva, 's_rated_mva': s_rated_mva})
    return channels, detail


def _json_safe_options(options):
    safe = {}
    for k, v in dict(options).items():
        safe[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    return safe


# ===========================================================================
# one solve
# ===========================================================================
def run_variant(fixture_key, fixture_info, variant, planning, floor_mva):
    full_path = os.path.join(REPO, fixture_info['path'])
    with open(full_path, 'rb') as f:
        payload = pickle.load(f)
    md, model = payload['metadata'], payload['model']
    kind = FIXTURE_KIND[fixture_key]

    if kind == 'TSO':
        net = planning.transmission_network.network[md['year']][md['day']]
        params = planning.transmission_network.params
        identity_ok = (net.name == md['network_name'] and net.year == md['year']
                       and str(net.day) == str(md['day']) and net.is_transmission)
    else:
        net = planning.distribution_networks[md['node_id']].network[md['year']][md['day']]
        params = planning.distribution_networks[md['node_id']].params
        identity_ok = (net.name == md['network_name'] and net.year == md['year']
                       and str(net.day) == str(md['day']) and not net.is_transmission)
    require(identity_ok, f'{fixture_key} {variant}: identity check failed')

    net.logs_dir = LOG_DIR
    from_warm_start = VARIANT_WARM_START[variant]
    solver, log_path, _ctx = N._create_smopf_solver(
        net, model, params, from_warm_start=from_warm_start, log_suffix=variant)

    base_tol = float(solver.options['tol'])
    base_acceptable_tol = float(solver.options['acceptable_tol'])
    t1_applied = None
    if variant == 'T1':
        solver.options['tol'] = base_tol / T1_DIVISOR
        solver.options['acceptable_tol'] = base_acceptable_tol / T1_DIVISOR
        t1_applied = {'base_tol': base_tol, 'base_acceptable_tol': base_acceptable_tol,
                      'divisor': T1_DIVISOR,
                      'applied_tol': base_tol / T1_DIVISOR,
                      'applied_acceptable_tol': base_acceptable_tol / T1_DIVISOR}

    options_echo = _json_safe_options(solver.options)

    active_objectives_before = [c.name for c in model.component_objects(pe.Objective, active=True)]
    require(len(active_objectives_before) == 1,
            f'{fixture_key} {variant}: expected exactly one active objective, found {active_objectives_before}')
    objective_name = active_objectives_before[0]

    result, wall_seconds = _do_solve(solver, model)

    succeeded = solver_result_succeeded(result)
    status = str(result.solver.status)
    termination = str(result.solver.termination_condition)
    message = str(getattr(result.solver, 'message', None))
    solve_result_id = getattr(result.solver, 'id', None)

    loaded_solution = False
    objective_value = None
    channels, detail = None, None
    if succeeded:
        model.solutions.load_from(result)
        loaded_solution = True
        objective_value = pe.value(getattr(model, objective_name))
        if kind == 'TSO':
            channels, detail = extract_tso_channels(model, net, planning, floor_mva)
        else:
            channels, detail = extract_dso_channels(model, net, planning, floor_mva)

    log_info = {}
    if log_path and os.path.exists(log_path):
        log_info = RESCALE.read_log_since(log_path, 0)

    record = {
        'fixture_key': fixture_key, 'kind': kind, 'variant': variant,
        'from_warm_start': from_warm_start, 'identity_ok': identity_ok,
        'metadata': md, 'objective_name': objective_name,
        'objective_value': objective_value, 'wall_seconds': wall_seconds,
        'status': status, 'termination_condition': termination, 'message': message,
        'solve_result_id': solve_result_id, 'succeeded': succeeded,
        'loaded_solution': loaded_solution, 'log_path': log_path,
        'log_iterations': log_info.get('iterations'), 'log_exit': log_info.get('exit'),
        'log_unscaled_summary': log_info.get('unscaled'), 'log_scaled_summary': log_info.get('scaled'),
        'effective_options': options_echo, 't1_tolerance_override': t1_applied,
        'entry_counts': {c: len(channels[c]) for c in CHANNELS} if channels else None,
    }
    return record, channels, detail


# ===========================================================================
# statistics
# ===========================================================================
def rms_and_max(a, b):
    require(len(a) == len(b), 'entry-count mismatch between variants')
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return None, None, 0
    rms = math.sqrt(sum(d * d for d in diffs) / n)
    max_abs = max(abs(d) for d in diffs)
    return rms, max_abs, n


def bitwise_identical(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def round_up_1_or_3(x):
    require(x > 0.0, 'round_up_1_or_3 requires a positive value')
    k = math.floor(math.log10(x))
    candidates = sorted({coef * (10.0 ** exp) for exp in range(k - 3, k + 4) for coef in (1, 3)})
    for v in candidates:
        if v >= x * (1.0 - 1e-9):
            return v
    raise RuntimeError('round_up_1_or_3: no candidate found in search range')


def main():
    require(not os.path.exists(OUT_DIR), f'refuse: output directory already exists: {OUT_DIR}')
    os.makedirs(LOG_DIR)

    spec = json.load(open(SPEC_PATH))
    fixtures = spec['fixtures']['files']

    fixture_verification = {}
    for fixture_key, info in fixtures.items():
        full_path = os.path.join(REPO, info['path'])
        actual_bytes = os.path.getsize(full_path)
        actual_sha = sha256_of(full_path)
        ok = (actual_bytes == info['bytes'] and actual_sha == info['sha256'])
        fixture_verification[fixture_key] = {
            'path': info['path'], 'expected_bytes': info['bytes'], 'actual_bytes': actual_bytes,
            'expected_sha256': info['sha256'], 'actual_sha256': actual_sha, 'ok': ok,
        }
        require(ok, f'fixture hash/size mismatch for {fixture_key}: {fixture_verification[fixture_key]}')

    O.WORK_DIR = os.path.join(OUT_DIR, 'evals')
    planning = O.fresh_planning('p515s33e4')
    floor_mva = planning.params.admm.shared_ess_normalization_floor_mva

    guard = SolveProfileGuard(PERMITTED, label='P5.15 Step 3.2 E4').install()
    solve_records = {}
    channel_store = {}
    detail_store = {}
    try:
        for fixture_key, info in fixtures.items():
            solve_records[fixture_key] = {}
            channel_store[fixture_key] = {}
            detail_store[fixture_key] = {}
            for variant in VARIANTS:
                print(f'[E4] fixture={fixture_key} variant={variant} solving...', flush=True)
                record, channels, detail = run_variant(fixture_key, info, variant, planning, floor_mva)
                if not record['succeeded']:
                    print(f'[E4][WARNING] {fixture_key} {variant}: solve did not terminate '
                          f'optimal/acceptable (status={record["status"]}, '
                          f'termination={record["termination_condition"]}) -- recorded as an '
                          f'invalidity condition, run continues to complete all 8 solves', flush=True)
                solve_records[fixture_key][variant] = record
                channel_store[fixture_key][variant] = channels
                detail_store[fixture_key][variant] = detail
                print(f'  -> {record["termination_condition"]} iters={record["log_iterations"]} '
                      f'obj={record["objective_value"]} wall={record["wall_seconds"]:.2f}s', flush=True)
    finally:
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)

    # -----------------------------------------------------------------
    # validity checks
    # -----------------------------------------------------------------
    invalid_reasons = []
    for fixture_key in fixtures:
        for variant in VARIANTS:
            rec = solve_records[fixture_key][variant]
            if not rec['succeeded']:
                invalid_reasons.append(f'{fixture_key} {variant}: termination not optimal/acceptable '
                                        f'({rec["status"]}, {rec["termination_condition"]})')
    w1_w2_bitwise = {}
    for fixture_key in fixtures:
        w1_w2_bitwise[fixture_key] = {}
        w1_channels = channel_store[fixture_key]['W1']
        w2_channels = channel_store[fixture_key]['W2']
        for c in CHANNELS:
            if w1_channels is None or w2_channels is None:
                w1_w2_bitwise[fixture_key][c] = None
                invalid_reasons.append(f'{fixture_key}: W1 or W2 did not produce a loaded solution; '
                                        f'bitwise-identical check on channel {c} could not be performed')
                continue
            identical = bitwise_identical(w1_channels[c], w2_channels[c])
            w1_w2_bitwise[fixture_key][c] = identical
            if not identical:
                invalid_reasons.append(f'{fixture_key}: W1 vs W2 NOT bitwise identical on channel {c}')
    if guard_failures:
        invalid_reasons.append(f'guard failures: {guard_failures}')

    # -----------------------------------------------------------------
    # per-pair, per-channel statistics
    # -----------------------------------------------------------------
    pair_stats = {}
    for fixture_key in fixtures:
        pair_stats[fixture_key] = {}
        for a, b in PAIRS:
            pair_label = f'{a}_vs_{b}'
            pair_stats[fixture_key][pair_label] = {}
            obj_a = solve_records[fixture_key][a]['objective_value']
            obj_b = solve_records[fixture_key][b]['objective_value']
            both_objectives_present = obj_a is not None and obj_b is not None
            obj_diff_abs = (obj_a - obj_b) if both_objectives_present else None
            obj_diff_rel = (obj_diff_abs / abs(obj_a)) if (both_objectives_present and obj_a != 0) else None
            both_channels_present = (channel_store[fixture_key][a] is not None
                                      and channel_store[fixture_key][b] is not None)
            for c in CHANNELS:
                if both_channels_present:
                    rms, max_abs, n = rms_and_max(channel_store[fixture_key][a][c], channel_store[fixture_key][b][c])
                else:
                    rms, max_abs, n = None, None, None
                pair_stats[fixture_key][pair_label][c] = {
                    'rms_per_entry': rms, 'max_abs_per_entry': max_abs, 'entry_count': n,
                }
            pair_stats[fixture_key][pair_label]['objective'] = {
                'a_value': obj_a, 'b_value': obj_b,
                'abs_diff': obj_diff_abs, 'rel_diff': obj_diff_rel,
            }
            pair_stats[fixture_key][pair_label]['termination_a'] = solve_records[fixture_key][a]['termination_condition']
            pair_stats[fixture_key][pair_label]['termination_b'] = solve_records[fixture_key][b]['termination_condition']
            pair_stats[fixture_key][pair_label]['iterations_a'] = solve_records[fixture_key][a]['log_iterations']
            pair_stats[fixture_key][pair_label]['iterations_b'] = solve_records[fixture_key][b]['log_iterations']

    # -----------------------------------------------------------------
    # derivation rule (delta_c, eps_abs) -- only if valid
    # -----------------------------------------------------------------
    derivation = {'valid': not invalid_reasons, 'invalid_reasons': invalid_reasons}
    if not invalid_reasons:
        delta_c = {}
        for c in CHANNELS:
            per_fixture_max = []
            for fixture_key in fixtures:
                rms_c1 = pair_stats[fixture_key]['W1_vs_C1'][c]['rms_per_entry']
                rms_t1 = pair_stats[fixture_key]['W1_vs_T1'][c]['rms_per_entry']
                per_fixture_max.append(max(rms_c1, rms_t1))
            delta_c[c] = max(per_fixture_max)
        max_delta = max(delta_c.values())
        pre_round = max(1e-5, 3.0 * max_delta)
        eps_abs = round_up_1_or_3(pre_round)
        derivation.update({
            'delta_c': delta_c,
            'max_c_delta_c': max_delta,
            'three_times_max_delta_c': 3.0 * max_delta,
            'pre_round_value': pre_round,
            'eps_abs': eps_abs,
            's32_reference_steps': S32_REFERENCE_STEPS,
            'delta_c_vs_s32_reference_ratio': {
                c: (delta_c[c] / S32_REFERENCE_STEPS[c]) for c in CHANNELS
            },
        })

    # -----------------------------------------------------------------
    # write outputs
    # -----------------------------------------------------------------
    output = {
        'stage': 'P5.15 Step 3.2 E4', 'spec_path': os.path.relpath(SPEC_PATH, REPO),
        'spec_sha256': sha256_of(SPEC_PATH),
        'fixture_verification': fixture_verification,
        'guard': {
            'permitted_call_sites': PERMITTED, 'expected_solves_declared_in_advance': EXPECTED_SOLVES,
            'observed_counts': guard.counts, 'permitted_sites_hit': guard.permitted_sites,
            'verify_failures': guard_failures,
        },
        'solve_records': solve_records,
        'w1_w2_bitwise_identical': w1_w2_bitwise,
        'pair_statistics': pair_stats,
        'validity': {'invalid_reasons': invalid_reasons, 'valid': not invalid_reasons},
        'derivation': derivation,
        'floor_mva_used': floor_mva,
        'one_process_justification': (
            'All 8 solves ran in one Python process. Each solve unpickled the fixture fresh from '
            'disk (pickle.load immediately before use) and called network._create_smopf_solver fresh '
            '(new SolverFactory instance, new .options dict) immediately before solver.solve. The '
            'guard counted permitted_exec (SystemCallSolver._execute_command, i.e. actual ipopt '
            'subprocess launches) equal to permitted_solve (8 each); see guard.observed_counts below. '
            'No separate-process fallback was triggered.'
        ),
    }
    # channel detail kept separately (larger; still within the new E4 directory)
    detail_output = {'channel_arrays': channel_store, 'per_entry_detail': detail_store}

    with open(RESULTS_PATH, 'x') as f:
        json.dump(output, f, indent=1, default=str)
    detail_path = os.path.join(OUT_DIR, 'e4_channel_detail.json')
    with open(detail_path, 'x') as f:
        json.dump(detail_output, f, indent=1, default=str)

    # -----------------------------------------------------------------
    # sha256 manifest of everything written under OUT_DIR (including logs)
    # -----------------------------------------------------------------
    manifest = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in files:
            full = os.path.join(root, name)
            if full == MANIFEST_PATH:
                continue
            rel = os.path.relpath(full, REPO)
            manifest[rel] = {'sha256': sha256_of(full), 'bytes': os.path.getsize(full)}
    with open(MANIFEST_PATH, 'x') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)

    print(f'[E4] status: {"VALID" if not invalid_reasons else "INVALID"}', flush=True)
    if not invalid_reasons:
        print(f'[E4] delta_c={derivation["delta_c"]} eps_abs={derivation["eps_abs"]}', flush=True)
    else:
        print(f'[E4] invalid_reasons={invalid_reasons}', flush=True)
    print(f'[E4] guard counts: {guard.counts}', flush=True)
    return output


if __name__ == '__main__':
    require(sys.executable == '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python',
            'wrong interpreter')
    main()
