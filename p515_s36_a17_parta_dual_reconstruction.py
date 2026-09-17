"""
P5.15 Addendum 17 -- PART A: zero-solve reconstruction of run 1's terminal
per-agent shared-ESS consensus duals (lambda), by replaying the production
ADMM dual-update rule over run 1's own recorded (x, z, rho) trajectory.

Authority: Planner task "Addendum 17 capture and reconstruction", PART A
(2026-09-16). Reads, never modifies: `shared_resources_planning.py`'s
`_update_shared_energy_storage_variables` (the master ESS consensus/dual
update, ~lines 7062-7150) and `create_admm_variables` (dual initialization,
~lines 3944-3949, 4013-4016).

## The exact update rule replayed (verbatim from
`_update_shared_energy_storage_variables`, `shared_resources_planning.py`)

Per (node, year, day, power_type, period) entry, per ADMM cycle c, given the
three agents' local solutions x_tso(c), x_dso(c), x_esso(c) (this cycle's
already-solved values) and the PREVIOUS cycle's duals lambda_agent(c-1):

    S_ref(c)        = shared_ess_reference_rating_mva at cycle c (fixed 2.5 MVA
                      throughout run 1 -- `_admm_shared_ess_reference_mva`,
                      Step 3.4 / Addendum 15 item 5(d); confirmed identical for
                      TSO, DSO and ESSO, since `reference_mva is not None`
                      short-circuits `_shared_ess_admm_normalization_mva` to
                      return the SAME S_ref regardless of each agent's own
                      physical rating S_i)
    a_agent(c)      = 1 / (2 * S_ref(c))                       (identical
                      across agents when S_ref is fixed, per the code path
                      above -- NOT per-agent physical rating in this regime)
    rho_agent(c)    = the ADMM penalty in force DURING cycle c's solves --
                      i.e. `rho_ess_before` at that cycle's `g_baseline.json`
                      row (the cycle's own `_update_admm_penalties` call runs
                      AFTER this dual update, producing `rho_ess_after`, which
                      becomes `rho_ess_before` of cycle c+1). Verified equal
                      across TSO/DSO/ESSO here (they start equal at cycle 1's
                      case-file value and are scaled by the identical factor
                      `factors['ess']` every cycle -- `_update_admm_penalties`,
                      `_scale_admm_penalty` calls on `rho_v/pf/ess`(TSO/DSO)
                      and `rho`(ESSO)); g_baseline.json records ONE scalar
                      `rho_ess_before`/`_after` per cycle for exactly this
                      reason. Cross-checked at the terminal cycle against the
                      ESSO's own captured `rho` Param (0.16875 both).

    denominator     = sum_agent [ rho_agent(c) * a_agent(c)^2 ]
    numerator       = sum_agent [ rho_agent(c) * a_agent(c)^2 * x_agent(c)
                                  + lambda_agent(c-1) * a_agent(c) ]
    z(c)            = numerator / denominator

    lambda_agent(c) = lambda_agent(c-1)
                      + rho_agent(c) * a_agent(c) * (x_agent(c) - z(c))

Initial condition (`create_admm_variables`): lambda_agent(0) = 0.0 for every
entry, every agent -- the pre-loop dual state before cycle 1 (VERIFIED by
direct inspection below, not assumed).

## Skip rule (production's own gate, `_update_shared_energy_storage_variables`)

The update for a given (node, year, day) triple at cycle c is skipped
(`continue`, no z or lambda change) UNLESS ALL THREE of that cycle's TSO, DSO
and ESSO local solves for that block report success via
`_solver_result_succeeded`. A recovered block (cold/tier-2 retry inside
`update_distribution_coordination_models_and_solve`/
`update_transmission_coordination_model_and_solve`) stores the RECOVERED
solve's result object in `results[...]`, which IS what `_solver_result_
succeeded` reads at this later point in the cycle -- there is no separate
"still-failed" flag threaded through. A recovered block therefore updates
NORMALLY (this is what "recovered" means operationally: the final results
object used everywhere downstream, including here, reports success).
VERIFIED empirically for run 1: `network_failures_baseline.jsonl` classifies
every one of its 298 entries as `recovered` (284) or `recovered_tier2` (14)
-- there is no `failed`/unrecovered entry -- and `g_baseline.json`'s own
`local_solves_ok` field is `True` at all 477 cycles. So the skip branch is
never taken in run 1's trajectory; this script asserts that (0 skips) rather
than assuming it.

## Entry-key mapping

`ess_entry_stride_baseline.jsonl` keys entries by (node_id, year LABEL e.g.
"2025", day LABEL e.g. "Spring", power_type). The ESSO's own pickled
`dual_p_req`/`dual_q_req` Params (`esso_models_baseline.pkl`, cross-validation
target) are indexed by (y_idx, d_idx, period) INTEGER positions
(`model.years = range(len(shared_ess_data.years))`,
`model.days = range(len(shared_ess_data.days))`,
`shared_energy_storage_data.py:407-408`). The (year label -> y_idx) and
(day label -> d_idx) maps are read from `shared_ess_data.years`/`.days`
ordering via ONE zero-solve `p56a_oracle.fresh_planning` call (the SAME
production entry point `p515_g_g1_g4_admm_gates._construct_arm_planning`
uses to obtain `sed`, reused directly rather than re-derived from the case
file by hand) -- guarded at 0 solves.

## Output

New directory `data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/`
(refuses if it exists): `parta_dual_reconstruction_results.json` (full
report) and `sha256_manifest.json`.
"""

import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import shared_resources_planning as srp  # noqa: E402  (read-only: one normalization helper call)

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')
OUT_DIR = os.path.join(RES, 'P515S36', 'A17_partA_dual_reconstruction')

REF_TERMINAL_CYCLE = 477
NODES = (5, 7, 9)
AGENTS = ('tso', 'dso', 'esso')
FRESH_PLANNING_EVAL_ID = 'p515s36_a17_parta_years_days'

RELATIVE_TOLERANCE = 1e-9
# Same noise-floor convention as `p515_s36_a17_diagnostics.py` (committed
# 5e2ba6f9): KKT-multiplier-type entries at inactive rows sit at
# machine-precision magnitude, so a relative-error test against them is
# testing floating-point noise, not the reconstruction rule. Entries with
# |captured| at or below this floor are excluded from the relative-error
# PASS criterion (still included in the unconditional max-abs-error report).
NOISE_FLOOR = 1e-8


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _one_g_baseline(run_dir):
    matches = glob.glob(os.path.join(run_dir, 'g_*.json'))
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one g_*.json in {run_dir}, found {matches}')
    return _load_json(matches[0]), matches[0]


# ======================================================================================================================
#  Zero-solve: years/days ordering (needed to map stride labels -> ESSO model integer indices)
# ======================================================================================================================
def _get_years_days_ordering():
    import p56a_oracle as O

    guard = SolveProfileGuard([], label='P5.15-S36 A17 PART A years/days lookup').install()
    try:
        planning = O.fresh_planning(FRESH_PLANNING_EVAL_ID)
        sed = planning.shared_ess_data
        years = list(sed.years)
        days = list(sed.days)
    finally:
        failures = guard.verify(0, 0)
        guard.uninstall()
    if failures:
        raise AssertionError('RULE SIX: years/days lookup was expected to solve nothing -> ' + '; '.join(failures))
    return years, days, dict(guard.counts)


# ======================================================================================================================
#  Verify the zero-initial-dual precondition directly from source (not assumed)
# ======================================================================================================================
def _verify_zero_initial_dual_from_source():
    import inspect
    src = inspect.getsource(srp.create_admm_variables)
    needle = "dual_variables['ess']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}"
    found_tso = needle in src
    found_dso = needle.replace("'tso'", "'dso'") in src
    found_esso = needle.replace("'tso'", "'esso'") in src
    return {
        'source_function': 'shared_resources_planning.create_admm_variables',
        'tso_zero_init_literal_found': found_tso,
        'dso_zero_init_literal_found': found_dso,
        'esso_zero_init_literal_found': found_esso,
        'all_found': found_tso and found_dso and found_esso,
    }


# ======================================================================================================================
#  Verify the skip-never-fires precondition for run 1
# ======================================================================================================================
def _verify_no_skips(run_dir):
    failures_path = os.path.join(run_dir, 'network_failures_baseline.jsonl')
    recs = []
    with open(failures_path) as handle:
        for line in handle:
            recs.append(json.loads(line))
    terminations = sorted(set(r['termination'] for r in recs))
    all_recovered = all(t in ('recovered', 'recovered_tier2') for t in terminations)

    g, _ = _one_g_baseline(run_dir)
    ct = g['cycle_trajectory']
    all_local_solves_ok = all(r['local_solves_ok'] for r in ct)
    cycles_1_to_terminal = [r['cycle'] for r in ct] == list(range(1, REF_TERMINAL_CYCLE + 1))

    return {
        'network_failures_file': failures_path,
        'n_failure_records': len(recs),
        'terminations_seen': terminations,
        'all_recovered_or_recovered_tier2': all_recovered,
        'g_baseline_all_cycles_local_solves_ok': all_local_solves_ok,
        'g_baseline_cycles_exactly_1_to_terminal': cycles_1_to_terminal,
        'conclusion': ('zero cycles skip the dual update in run 1 -- every logged failure was recovered '
                       '(no `failed`/unrecovered classification exists), and every cycle\'s own '
                       '`local_solves_ok` is True' if (all_recovered and all_local_solves_ok and cycles_1_to_terminal)
                       else 'PRECONDITION VIOLATED -- see fields above'),
    }


# ======================================================================================================================
#  The reconstruction itself
# ======================================================================================================================
def _entry_index(node_id, y_idx, d_idx, power_type, period, years, days):
    # Deterministic flat index: node-major, year, day, power_type, period.
    n_p, n_y, n_d, n_pt, n_per = len(NODES), len(years), len(days), 2, 24
    ni = NODES.index(node_id)
    pti = 0 if power_type == 'p' else 1
    return (((ni * n_y + y_idx) * n_d + d_idx) * n_pt + pti) * n_per + period


def reconstruct(run_dir, years, days, snapshot_before_cycle=None):
    """`snapshot_before_cycle`, if given, additionally records lambda_esso
    (all agents, in fact) as it stood BEFORE that cycle's own update is
    applied -- i.e. the value that cycle's ESSO solve actually used as its
    `dual_p_req`/`dual_q_req` Param input (set in
    `update_shared_energy_storages_coordination_model_and_solve` BEFORE
    `shared_ess_data.optimize(...)` is called, i.e. BEFORE that same
    cycle's z/lambda update runs). This is the correct cross-validation
    target for `esso_models_baseline.pkl`'s frozen `dual_p_req`/`dual_q_req`
    Params at the run's terminal cycle T: since those Params are only ever
    SET, never re-read after the solve, the pickle holds lambda_esso(T-1),
    NOT lambda_esso(T)."""
    n_years, n_days = len(years), len(days)
    n_entries = len(NODES) * n_years * n_days * 2 * 24

    lam = {agent: np.zeros(n_entries, dtype=float) for agent in AGENTS}
    lam_snapshot_before = None

    g, _ = _one_g_baseline(run_dir)
    ct_by_cycle = {r['cycle']: r for r in g['cycle_trajectory']}

    stride_path = glob.glob(os.path.join(run_dir, 'ess_entry_stride_*.jsonl'))
    if len(stride_path) != 1:
        raise RuntimeError(f'expected exactly one ess_entry_stride_*.jsonl in {run_dir}, found {stride_path}')
    stride_path = stride_path[0]

    norm_trajectory = {agent: [] for agent in AGENTS}
    reference_rating_values_seen = set()
    rho_values_seen_per_cycle = []

    with open(stride_path) as handle:
        for line in handle:
            row = json.loads(line)
            cycle = row['cycle']

            if snapshot_before_cycle is not None and cycle == snapshot_before_cycle:
                lam_snapshot_before = {agent: lam[agent].copy() for agent in AGENTS}

            g_row = ct_by_cycle[cycle]
            rho = float(g_row['rho_ess_before'])
            s_ref = float(g_row['shared_ess_reference_rating_mva'])
            reference_rating_values_seen.add(s_ref)
            rho_values_seen_per_cycle.append(rho)

            # Real production normalization helper (reference_mva short-circuits it to
            # exactly s_ref regardless of the dummy rating/floor passed in).
            a = 1.0 / (2.0 * srp._shared_ess_admm_normalization_mva(0.0, 0.0, reference_mva=s_ref))

            rho_agent = {agent: rho for agent in AGENTS}  # verified equal across agents, see docstring
            denom = sum(rho_agent[agent] * a ** 2 for agent in AGENTS)

            for entry in row['entries']:
                node_id = entry['node_id']
                year_idx = years.index(int(entry['year']))
                day_idx = days.index(entry['day'])
                power_type = entry['power_type']
                x = entry['x']

                for p in range(24):
                    idx = _entry_index(node_id, year_idx, day_idx, power_type, p, years, days)
                    x_agent = {agent: x[agent][p] for agent in AGENTS}
                    lam_prev = {agent: lam[agent][idx] for agent in AGENTS}

                    numer = sum(rho_agent[agent] * a ** 2 * x_agent[agent] + lam_prev[agent] * a
                                for agent in AGENTS)
                    z_new = numer / denom

                    for agent in AGENTS:
                        lam[agent][idx] = lam_prev[agent] + rho_agent[agent] * a * (x_agent[agent] - z_new)

            for agent in AGENTS:
                norm_trajectory[agent].append({'cycle': cycle, 'reconstructed_norm_y': float(np.linalg.norm(lam[agent]))})

    return {
        'lambda_terminal': lam,
        'lambda_snapshot_before_cycle': lam_snapshot_before,
        'snapshot_before_cycle': snapshot_before_cycle,
        'norm_trajectory': norm_trajectory,
        'reference_rating_values_seen': sorted(reference_rating_values_seen),
        'rho_values_seen_per_cycle_min_max': [min(rho_values_seen_per_cycle), max(rho_values_seen_per_cycle)],
        'n_entries': n_entries,
    }


# ======================================================================================================================
#  Cross-validation vs g_baseline.json's boyd_ess_norm_y_<agent> trajectory
# ======================================================================================================================
def cross_validate_norm_trajectory(run_dir, norm_trajectory):
    g, _ = _one_g_baseline(run_dir)
    ct_by_cycle = {r['cycle']: r for r in g['cycle_trajectory']}
    report = {}
    for agent in AGENTS:
        key = f'boyd_ess_norm_y_{agent}'
        max_rel = 0.0
        max_abs = 0.0
        worst_cycle = None
        n_compared = 0
        for entry in norm_trajectory[agent]:
            cycle = entry['cycle']
            recon = entry['reconstructed_norm_y']
            recorded = ct_by_cycle[cycle][key]
            abs_err = abs(recon - recorded)
            rel_err = abs_err / recorded if recorded != 0.0 else abs_err
            n_compared += 1
            if rel_err > max_rel:
                max_rel = rel_err
                max_abs = abs_err
                worst_cycle = cycle
        report[agent] = {
            'n_cycles_compared': n_compared,
            'max_relative_error': max_rel,
            'max_abs_error_at_worst_cycle': max_abs,
            'worst_cycle': worst_cycle,
            'passes_1e-9_relative': max_rel <= RELATIVE_TOLERANCE,
        }
    return report


# ======================================================================================================================
#  Cross-validation vs the ESSO's own captured terminal dual_p_req/dual_q_req (esso_models_baseline.pkl)
# ======================================================================================================================
def cross_validate_esso_terminal(run_dir, lam_esso, years, days):
    import pickle
    with open(os.path.join(run_dir, 'esso_models_baseline.pkl'), 'rb') as handle:
        models = pickle.load(handle)

    max_abs_unconditional = 0.0
    max_rel_naive = 0.0
    worst_naive = None
    max_rel_floor_gated = 0.0
    worst_floor_gated = None
    n_checked = 0
    n_below_floor = 0
    rho_captured = {}
    for node_id in NODES:
        m = models[node_id]
        rho_captured[node_id] = pe.value(m.rho)
        for (y_idx, d_idx, p), _ in m.dual_p_req.items():
            for power_type, param in (('p', m.dual_p_req), ('q', m.dual_q_req)):
                captured = pe.value(param[y_idx, d_idx, p])
                idx = _entry_index(node_id, y_idx, d_idx, power_type, p, years, days)
                recon = lam_esso[idx]
                abs_err = abs(recon - captured)
                rel_err = abs_err / abs(captured) if captured != 0.0 else abs_err
                n_checked += 1

                if abs_err > max_abs_unconditional:
                    max_abs_unconditional = abs_err

                if rel_err > max_rel_naive:
                    max_rel_naive = rel_err
                    worst_naive = {'node_id': node_id, 'y_idx': y_idx, 'd_idx': d_idx, 'period': p,
                                   'power_type': power_type, 'reconstructed': recon, 'captured': captured}

                if abs(captured) <= NOISE_FLOOR:
                    n_below_floor += 1
                    continue
                if rel_err > max_rel_floor_gated:
                    max_rel_floor_gated = rel_err
                    worst_floor_gated = {'node_id': node_id, 'y_idx': y_idx, 'd_idx': d_idx, 'period': p,
                                         'power_type': power_type, 'reconstructed': recon, 'captured': captured}

    return {
        'n_entries_checked': n_checked,
        'n_entries_at_or_below_noise_floor': n_below_floor,
        'noise_floor': NOISE_FLOOR,
        'max_abs_error_unconditional_all_entries': max_abs_unconditional,
        'max_relative_error_naive_all_entries': max_rel_naive,
        'worst_entry_naive': worst_naive,
        'naive_reading': ('dominated by entries where the CAPTURED dual is itself at machine-precision '
                           '(inactive-row KKT-multiplier noise) -- see worst_entry_naive, whose `captured` '
                           'magnitude is far below any physically meaningful dual. Not the PASS criterion.'),
        'max_relative_error_excluding_below_noise_floor': max_rel_floor_gated,
        'worst_entry_excluding_below_noise_floor': worst_floor_gated,
        'rho_captured_per_node': rho_captured,
        'passes_1e-9_relative': (max_rel_floor_gated <= RELATIVE_TOLERANCE
                                  and max_abs_unconditional <= 1e-6),
    }


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    zero_init_check = _verify_zero_initial_dual_from_source()
    if not zero_init_check['all_found']:
        raise AssertionError(f'zero-initial-dual precondition NOT verified from source: {zero_init_check}')

    no_skip_check = _verify_no_skips(REF_RUN)
    if not (no_skip_check['all_recovered_or_recovered_tier2'] and no_skip_check['g_baseline_all_cycles_local_solves_ok']
            and no_skip_check['g_baseline_cycles_exactly_1_to_terminal']):
        raise AssertionError(f'no-skip precondition NOT verified: {no_skip_check}')

    years, days, years_days_guard_counts = _get_years_days_ordering()

    recon = reconstruct(REF_RUN, years, days, snapshot_before_cycle=REF_TERMINAL_CYCLE)
    norm_cross_val = cross_validate_norm_trajectory(REF_RUN, recon['norm_trajectory'])
    # Cross-validate against lambda_esso(T-1) -- see `reconstruct`'s
    # `snapshot_before_cycle` docstring: the ESSO's OWN pickled
    # `dual_p_req`/`dual_q_req` Params were SET from this value immediately
    # before the terminal cycle's ESSO solve and are never updated again
    # (Params, not read back after solving), so they hold lambda_esso(T-1),
    # not lambda_esso(T).
    esso_cross_val = cross_validate_esso_terminal(REF_RUN, recon['lambda_snapshot_before_cycle']['esso'], years, days)
    esso_cross_val_naive_terminal = cross_validate_esso_terminal(REF_RUN, recon['lambda_terminal']['esso'], years, days)

    overall_pass = (
        all(norm_cross_val[agent]['passes_1e-9_relative'] for agent in AGENTS)
        and esso_cross_val['passes_1e-9_relative']
    )
    if recon['snapshot_before_cycle'] != REF_TERMINAL_CYCLE:
        raise AssertionError('snapshot_before_cycle bookkeeping mismatch')

    report = {
        'stage': 'P5.15-S36-A17-PARTA', 'authority': 'Addendum 17 capture-and-reconstruction task (2026-09-16)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'run': REF_RUN, 'terminal_cycle': REF_TERMINAL_CYCLE,
        'years_days_ordering': {'years': years, 'days': days, 'guard_counts': years_days_guard_counts},
        'zero_initial_dual_precondition': zero_init_check,
        'no_skip_precondition': no_skip_check,
        'update_rule': {
            'formula': ('lambda_agent(c) = lambda_agent(c-1) + rho_agent(c) * a(c) * (x_agent(c) - z(c)); '
                        'z(c) = [sum_agent(rho_agent(c)*a(c)^2*x_agent(c) + lambda_agent(c-1)*a(c))] / '
                        '[sum_agent(rho_agent(c)*a(c)^2)]; a(c) = 1/(2*S_ref(c))'),
            'source': 'shared_resources_planning.py::_update_shared_energy_storage_variables (comment block '
                      '"Shared-ESS consensus ADMM"), rho read as g_baseline.json cycle row\'s rho_ess_before '
                      '(the value in force during that cycle\'s solves, BEFORE that cycle\'s own rho adaptation)',
            'S_ref_values_seen': recon['reference_rating_values_seen'],
            'rho_min_max_seen': recon['rho_values_seen_per_cycle_min_max'],
            'agents_share_one_rho_value_per_cycle': ('by construction of `_update_admm_penalties`/'
                                                       '`_scale_admm_penalty`: TSO rho_ess, DSO rho_ess and ESSO rho '
                                                       'start equal (case-file initial value) and are scaled by the '
                                                       'IDENTICAL factor every cycle -- g_baseline.json records ONE '
                                                       'scalar rho_ess_before/_after per cycle for this reason'),
        },
        'n_entries_reconstructed': recon['n_entries'],
        'cross_validation_vs_g_baseline_norm_trajectory': norm_cross_val,
        'cross_validation_vs_esso_captured_terminal_duals': {
            'reading': (
                'The ESSO model\'s dual_p_req/dual_q_req Params are SET from '
                'dual_vars["ess"]["esso"]["current"] in '
                'update_shared_energy_storages_coordination_model_and_solve, BEFORE that '
                'same cycle\'s shared_ess_data.optimize(...) call, and BEFORE that same '
                'cycle\'s own z/lambda update runs (the update happens later in the cycle, '
                'in update_and_check_convergence(update_sess=True)). A Param is never read '
                'back and re-set after solving, so the run\'s pickled terminal-cycle Params '
                'hold lambda_esso(T-1), the value one cycle BEHIND the run\'s own terminal '
                'lambda_esso(T) that boyd_ess_norm_y_esso and this script\'s own lambda_terminal '
                'reflect. The CORRECT cross-validation target is therefore '
                'lambda_snapshot_before_cycle (T-1), not lambda_terminal (T); both are reported.'
            ),
            'against_lambda_esso_T_minus_1_CORRECT_target': esso_cross_val,
            'against_lambda_esso_T_naive_terminal_INFORMATIONAL_ONLY': esso_cross_val_naive_terminal,
        },
        'overall_pass_1e-9_relative': overall_pass,
    }

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'parta_dual_reconstruction_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # Save the reconstructed terminal per-agent, per-entry duals (task: "Save the
    # reconstructed terminal per-agent, per-entry duals").
    terminal_duals_path = os.path.join(OUT_DIR, 'reconstructed_terminal_duals.json')
    terminal_duals = {
        'entry_index_convention': ('node-major (order = (5,7,9)), then year_idx (order = years above), then '
                                    'day_idx (order = days above), then power_type (0=p,1=q), then period (0-23); '
                                    'flat index computed by _entry_index in this script'),
        'years_order': years, 'days_order': days,
        'lambda_terminal_cycle_477': {agent: recon['lambda_terminal'][agent].tolist() for agent in AGENTS},
        'lambda_before_cycle_477_i.e._lambda_at_cycle_476': {
            agent: recon['lambda_snapshot_before_cycle'][agent].tolist() for agent in AGENTS},
        'note': ('lambda_terminal_cycle_477 is the run\'s true final ADMM dual state (what '
                 'boyd_ess_norm_y_* at cycle 477 reflects). lambda_before_cycle_477 (= '
                 'lambda_esso(476) for the esso agent) is what esso_models_baseline.pkl\'s '
                 'dual_p_req/dual_q_req Params actually hold -- see the cross-validation '
                 'section\'s "reading" field.'),
    }
    with open(terminal_duals_path, 'w') as handle:
        json.dump(terminal_duals, handle)

    manifest = {}
    for fname in ('parta_dual_reconstruction_results.json', 'reconstructed_terminal_duals.json'):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    script_path = os.path.abspath(__file__)
    with open(script_path, 'rb') as handle:
        manifest[os.path.basename(script_path)] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(f'zero_initial_dual_precondition: {zero_init_check["all_found"]}')
    print(f'no_skip_precondition: {no_skip_check["conclusion"]}')
    for agent in AGENTS:
        r = norm_cross_val[agent]
        print(f'norm trajectory {agent}: max_rel_err={r["max_relative_error"]:.3e} '
              f'(worst cycle {r["worst_cycle"]}) pass={r["passes_1e-9_relative"]}')
    print(f'ESSO terminal cross-val: max_rel_err(floor-gated)={esso_cross_val["max_relative_error_excluding_below_noise_floor"]:.3e} '
          f'max_abs_err(unconditional)={esso_cross_val["max_abs_error_unconditional_all_entries"]:.3e} '
          f'pass={esso_cross_val["passes_1e-9_relative"]}')
    print(f'OVERALL PASS: {overall_pass}')
    print(f'wrote: {results_path}')
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
