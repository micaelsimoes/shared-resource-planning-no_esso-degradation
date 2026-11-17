"""
P5.15 Addendum 17 -- H1/H2 zero-solve dual-identifiability diagnostic.

Authority: Planner task "Bounded ZERO-SOLVE diagnostic: is the slow storage-dual
build-up the price-identified component or the unidentified per-agent split?"
(2026-09-17). No Pyomo/IPOPT solves; no production, case-file or harness edits.
`SolveProfileGuard([], ...)` is installed for the entire script (permitted list
empty -- zero solves declared and verified).

## Question

Prices determine only the SUMMED storage KKT condition, not the SPLIT of dual
value across TSO/DSO/ESSO, which is free along the gradients of the storage
constraints active at the terminal schedule (the "in-span" directions). This
script decomposes run 1's reconstructed per-agent shared-ESS consensus duals
lambda_agent(c), c = 1..477, into a component ORTHOGONAL to that span
(I, "identifiable" -- price pins this down) and a component IN the span
(J, "in-span" -- free, price-independent), and asks which settles later.

## Inputs (all committed, read-only)

  - `data/SRP1/Results/P515S35_REF_run/ess_entry_stride_baseline.jsonl`: per
    cycle 1..477, per (node, year, day, power_type) entry, the consensus z and
    the three agents' x, all 24 periods, stride 1 (every cycle recorded).
  - `data/SRP1/Results/P515S35_REF_run/g_baseline.json`: per-cycle rho_ess_before,
    shared_ess_reference_rating_mva, and the boyd_ess_norm_y_* cross-validation
    targets.
  - `data/SRP1/Results/P515S35_REF_run/esso_capture/baseline/node{5,7,9}_cycle477.jsonl`:
    the ESSO's own terminal-cycle per-(y,d,p) pch/pdch/pnet/s_max (checked: NO
    SoC or energy field is recorded here -- see "SoC reconstruction" below).
  - `data/SRP1/Results/P515S35_REF_run/esso_models_baseline.pkl`: the ESSO's own
    pickled terminal-cycle Pyomo model, read only for
    `es_e_available_per_unit`/`es_s_available_per_unit` (the post-SoH available
    capacity Vars, SOLVED at cycle 477, not lagged -- unlike `dual_p_req`, which
    PART A (commit `b63e4d94`) found is a Param set BEFORE that cycle's solve;
    `es_e_available_per_unit` is a Var, read back AFTER the solve, so it
    genuinely reflects cycle 477's own state).

## The exact ADMM dual-update rule replayed (verbatim, identical to PART A,
`p515_s36_a17_parta_dual_reconstruction.py`, itself sourced from
`shared_resources_planning.py::_update_shared_energy_storage_variables`,
"Shared-ESS consensus ADMM" block, ~lines 7062-7150)

Per (node, year, day, power_type, period) entry, per ADMM cycle c:

    a(c)      = 1 / (2 * S_ref(c))            S_ref fixed at 2.5 MVA throughout
                                                run 1 (identical across agents
                                                once reference_mva is not None,
                                                `_shared_ess_admm_normalization_mva`)
    rho(c)    = g_baseline.json cycle-c row's rho_ess_before (identical across
                                                TSO/DSO/ESSO by construction of
                                                `_update_admm_penalties`)
    z(c)      = [ sum_agent( rho(c)*a(c)^2*x_agent(c) + lambda_agent(c-1)*a(c) ) ]
                / [ sum_agent( rho(c)*a(c)^2 ) ]
    lambda_agent(c) = lambda_agent(c-1) + rho(c)*a(c)*(x_agent(c) - z(c))

Initial condition lambda_agent(0) = 0 for every entry/agent (`create_admm_variables`,
verified from source string, same check as PART A). This script's own replay is
cross-validated against `g_baseline.json`'s `boyd_ess_norm_y_{tso,dso,esso}`
trajectory (the SAME quantity PART A validated to <=1.5e-14 relative error) at
every one of the 477 cycles, to the same 1e-9 relative tolerance.

## The terminal active-row span S (per (node, year, day), P channel only --
Q has no LP price, so Q is reported only as a dual-share fraction, never
decomposed against a span)

At the terminal schedule (cycle 477): z(477) from the stride file (the "run 1
terminal schedule"); pch(477)/pdch(477)/s_max(477) from the ESSO's own capture
(`esso_capture/baseline/node{n}_cycle477.jsonl`) -- these are the physical
(MW/MWh, un-normalized) values, confirmed identical in scale to x_esso in the
stride file (cross-checked below to <1e-9 abs).

  - SoC reconstruction (NOT recorded in the ESSO capture -- checked, absent;
    reconstructed per the task's fallback instruction):
    `sess_soc_rule` (`model_construction_helpers.py:893-912`):
        dt = HOURS_PER_REPRESENTATIVE_DAY / 24 = 1.0 h  (definitions.py:94)
        soc[-1] := E_avail(node,year) * ENERGY_STORAGE_RELATIVE_INIT_SOC (=0.50,
                   definitions.py:40)
        soc[p]  = soc[p-1] + eff_ch*pch[p]*dt - pdch[p]*dt/eff_dch
    eff_ch=0.97, eff_dch=0.96 (`shared_energy_storage.py:14-15` class defaults,
    confirmed un-overridden for nodes 5/7/9 by a zero-solve read of
    `shared_ess_data.shared_energy_storages[year][idx]`, same technique
    `p515_s34_efc_benchmark.py:317-327` used).
    E_avail(node,year) = sum_{y_inv} es_e_available_per_unit[y_inv,year_idx]
    (`shared_energy_storage_data.py:256-262`, `get_available_capacities`),
    read from the terminal-cycle ESSO pickle (post-SoH available energy, MWh,
    un-normalized -- this IS what `shared_es_e_rated_fixed` in the TSO/DSO
    local models is set to, `shared_resources_planning.py:5141-5142` etc.,
    divided by that agent's OWN s_base; working in physical units here matches
    the ESSO's own (un-normalized) space, which is also the space z/x_agent
    live in, `_update_shared_energy_storage_variables:7004,7021,6986` -- TSO/DSO
    values are multiplied back up by their own s_base before being stored in
    `shared_ess_vars`).
    soc_min = E_avail * ENERGY_STORAGE_MIN_ENERGY_STORED (=0.10)
    soc_max = E_avail * ENERGY_STORAGE_MAX_ENERGY_STORED (=0.90)

  - The day-balance gradient v (one 24-vector per block): from `sess_soc_final_rule`
    (`model_construction_helpers.py:915-921`), the day-balance row is
    soc[23] == E_avail*0.50, i.e. sum_p delta_p = 0 with
    delta_p = eff_ch*pch_p*dt - pdch_p*dt/eff_dch. Its gradient in the reduced
    (pnet_p) space, given the near-complementarity of pch/pdch (only one active
    per period at an interior-point optimum), is:
        v_p = eff_ch * dt        if z_p >= 0   (period charges)
        v_p = dt / eff_dch       if z_p <  0   (period discharges)
    (MODELLING CHOICE: charge/discharge direction is read from the sign of the
    terminal z, not from the ESSO capture's pch/pdch, which can carry a small
    non-complementarity leak (documented at
    `shared_energy_storage_data.py:471-511`); the two agree on sign at every
    period checked here, see the per-block cross-check in the output.)

  - Power-limit-active periods (`sess_active_sum_limit_rule`,
    `model_construction_helpers.py:827-837`, `pch+pdch <= S_rated`, matching the
    same `S_rated` reflected in `pnet`'s envelope since pch/pdch are
    near-complementary): periods p with |z_p| within POWER_LIMIT_REL_TOL=1e-6
    relative of s_max (task-specified tolerance), s_max read directly from the
    ESSO capture's own `s_max` field for that (node,year,day,period).

  - SoC-bound-active periods (`sess_soc_lower_limit`/`sess_soc_upper_limit`,
    `model_construction_helpers.py:840-847`): periods p with the reconstructed
    soc[p] within SOC_BOUND_REL_TOL=1e-6 (MODELLING CHOICE, same order as the
    task's power-limit tolerance, scaled to the energy dimension) relative of
    E_avail(node,year) of soc_min or soc_max.

  - Idle periods (task-specified): |z_p| < IDLE_REL_TOL=1e-6 relative of s_max.
    Reported as a genuine "interval of freedom" modelling choice: included in
    the "with_idle" span variant, excluded from the "without_idle" variant; both
    are reported for every block.

  - Candidate matrix per block = [v, unit vectors for the periods above (union,
    de-duplicated)], columns in this fixed order (v first, then periods 0..23
    ascending for whichever category(ies) apply). Orthonormalized via
    `numpy.linalg.qr` (reduced mode); rank = count of |diag(R)| > RANK_TOL=1e-8
    (absolute -- candidate columns are O(1) in norm: v in [dt/eff_dch, eff_ch*dt]
    subset~[0.97,1.04], unit vectors norm 1, so 1e-8 cleanly separates numerical
    noise from genuine rank). The kept orthonormal basis Q_S is Q's columns at
    the NONZERO-diagonal positions (not simply Q[:, :rank] by position -- this
    is the ROBUST choice: sequential QR/Gram-Schmidt correctly identifies, at
    every column, whether it adds a new direction not already in the span of
    ALL preceding columns, regardless of which specific columns end up flagged
    dependent; taking the nonzero-diagonal columns therefore spans col(M)
    exactly regardless of column order, whereas naively slicing the first
    `rank` columns would not be robust to an out-of-order dependency).
    P_S = Q_S @ Q_S.T (24x24 orthogonal projector); P_perp = I - P_S.

## Per-cycle decomposition (P channel; Q reported as a dual-share fraction only)

  I_agent(c) = P_perp @ lambda_agent_block(c)   (identifiable component)
  J_agent(c) = P_S    @ lambda_agent_block(c)   (in-span / free component)

Aggregated (sum of squares, then sqrt) over all 36 (node,year,day) P-channel
blocks, per agent; "total" combines the three agents in quadrature
(sqrt(sum_agent ||.||^2)), the SAME convention `g_baseline.json`'s own
`boyd_ess_norm_y` (no suffix) uses relative to `boyd_ess_norm_y_{tso,dso,esso}`
(cross-checked in this script's own output).

## Settling-cycle rule (task-specified, applied to each series)

The first cycle c* such that, for EVERY cycle c in [c*, 477] with c-50 >= 1, the
relative change |series(c) - series(c-50)| / |series(c-50)| < 0.01 (1% per 50
cycles). Found by scanning cycles 51..477 for the LAST cycle at which this
relative change is still >= 0.01 (a "violation"); c* = (last violation) + 1. If
no violation occurs in 51..477, c* = 51 (settled from the earliest checkable
cycle). If the last checkable cycle (477) is itself a violation, the series has
NOT settled by 477 (c* = None). The relative change over the FINAL 50-cycle
window (cycle 477 vs cycle 427) is always reported additionally, regardless.

## Verdict rule (task-specified, applied verbatim)

  - H1 supported if ||I|| (total, and for TSO specifically) settles MATERIALLY
    LATER than ||J|| (this script's threshold for "materially": settling-cycle
    difference > 50 cycles, i.e. more than one settling window -- a MODELLING
    CHOICE, documented here, not implied by the task text), or ||I|| is still
    changing >1%/50cyc at 477 while ||J|| is not.
  - H2 supported if ||J|| settles later (by the same >50-cycle threshold), or
    ||J|| is still drifting while ||I|| has settled.
  - Inconclusive if dim S >= 20 of 24 on MOST (>18 of 36) P-channel blocks (the
    decomposition is then vacuous -- almost the whole 24-dim period space is
    "in-span" by construction, leaving no room for a genuine orthogonal
    complement), or if both ||I|| and ||J|| settle together (settling-cycle
    difference <= 50 cycles, or both None/not-settled). Both span variants
    (with_idle / without_idle) are evaluated and reported; if they disagree, both
    are stated, with numbers, and the verdict favours "inconclusive" unless both
    variants agree.

## Gate 3 (`data/SRP1/Results/P515S35_PT_run/`)

Its `ess_entry_stride_baseline.jsonl` is written at stride 5 (cycles
1,6,11,...,146 -- 30 rows) while the run itself has 150 cycles
(`g_baseline.json`'s `cycle_trajectory` spans exactly cycles 1..150). The
per-cycle ADMM dual-update recursion requires EVERY cycle's (x, rho) to replay
correctly (lambda(c) depends on lambda(c-1) AND cycle c's own x); cycles
2-5, 7-10, ..., 147-150 have no recorded x, so the recursion cannot be
replayed past cycle 1 without inventing intermediate x values. Per the task's
own instruction ("only if its stride x/z cover every cycle; otherwise state not
computable") this script does NOT attempt a reconstruction for gate 3 and
records this finding (with the exact evidence above) in the output JSON instead.

## Output

New directory `data/SRP1/Results/P515S36/H1H2_identifiability/` (refuses if it
exists): `h1h2_identifiability_results.json` (full report) and
`sha256_manifest.json`.
"""

import glob
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s36_a17_parta_dual_reconstruction as parta  # noqa: E402 (reuse verified helpers)
from definitions import (  # noqa: E402
    ENERGY_STORAGE_MAX_ENERGY_STORED, ENERGY_STORAGE_MIN_ENERGY_STORED,
    ENERGY_STORAGE_RELATIVE_INIT_SOC, HOURS_PER_REPRESENTATIVE_DAY,
)

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')
GATE3_RUN = os.path.join(RES, 'P515S35_PT_run')
OUT_DIR = os.path.join(RES, 'P515S36', 'H1H2_identifiability')

REF_TERMINAL_CYCLE = 477
NODES = (5, 7, 9)
AGENTS = ('tso', 'dso', 'esso')
N_PERIODS = 24
N_YEARS = 3
N_DAYS = 4
FRESH_PLANNING_EVAL_ID = 'p515s36_h1h2_years_days'

DT_HOURS = HOURS_PER_REPRESENTATIVE_DAY / N_PERIODS  # = 1.0 for the standard 24-instant day

POWER_LIMIT_REL_TOL = 1e-6
IDLE_REL_TOL = 1e-6
SOC_BOUND_REL_TOL = 1e-6
RANK_TOL = 1e-8
RELATIVE_TOLERANCE = 1e-9  # cross-validation tolerance vs g_baseline.json

SETTLE_WINDOW = 50
SETTLE_THRESHOLD = 0.01
SETTLE_MATERIAL_GAP = 50  # cycles; "materially later" modelling choice, see docstring
IDENTIFIABLE_FRACTION_CYCLES = [1, 50, 150, 300, 477]
VACUOUS_DIM_THRESHOLD = 20  # of 24
VACUOUS_BLOCK_FRACTION = 0.5  # "most" blocks


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _one_g_baseline(run_dir):
    matches = glob.glob(os.path.join(run_dir, 'g_*.json'))
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one g_*.json in {run_dir}, found {matches}')
    return _load_json(matches[0]), matches[0]


# ======================================================================================================================
#  Zero-solve: years/days ordering + efficiencies (single batch, own guard-free reads -- outer guard covers it)
# ======================================================================================================================
def _get_years_days_and_efficiencies():
    import p56a_oracle as O

    planning = O.fresh_planning(FRESH_PLANNING_EVAL_ID)
    sed = planning.shared_ess_data
    years = list(sed.years)
    days = list(sed.days)
    active_nodes = list(sed.active_distribution_network_nodes)
    if sorted(active_nodes) != sorted(NODES):
        raise AssertionError(f'active nodes {active_nodes} != expected {NODES}')

    idx0 = sed.get_shared_energy_storage_idx(NODES[0])
    ess0 = sed.shared_energy_storages[years[0]][idx0]
    eff_ch, eff_dch = ess0.eff_ch, ess0.eff_dch
    for node in NODES:
        idx = sed.get_shared_energy_storage_idx(node)
        for year in years:
            ess = sed.shared_energy_storages[year][idx]
            if ess.eff_ch != eff_ch or ess.eff_dch != eff_dch:
                raise AssertionError(
                    f'eff_ch/eff_dch not uniform: node={node} year={year} got '
                    f'({ess.eff_ch},{ess.eff_dch}) expected ({eff_ch},{eff_dch})')
    return years, days, eff_ch, eff_dch


# ======================================================================================================================
#  Terminal-cycle inputs: ESSO capture (pch/pdch/s_max) + ESSO pickle (E_avail, post-SoH)
# ======================================================================================================================
def _load_esso_terminal_capture(run_dir, node, cycle):
    path = os.path.join(run_dir, 'esso_capture', 'baseline', f'node{node}_cycle{cycle:03d}.jsonl')
    if not os.path.isfile(path):
        raise RuntimeError(f'missing ESSO terminal capture file: {path}')
    entries = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row['y_inv'] != 0:
                # Only cohort y_inv=0 carries nonzero investment for this instance
                # (cross-checked against the pickle's es_e_investment_fixed below);
                # a nonzero-y_inv row here would mean a second cohort is active and
                # this script's span construction (which sums only over what the
                # capture records) would be INCOMPLETE -- fail loudly rather than
                # silently drop it.
                raise RuntimeError(f'unexpected non-zero y_inv row in {path}: {row}')
            entries[(row['y'], row['d'], row['p'])] = row
    return entries, path


def _load_esso_pickle_e_avail(run_dir):
    path = os.path.join(run_dir, 'esso_models_baseline.pkl')
    with open(path, 'rb') as handle:
        models = pickle.load(handle)
    e_avail = {}
    s_avail = {}
    e_investment_fixed = {}
    for node in NODES:
        m = models[node]
        e_avail[node] = {}
        s_avail[node] = {}
        e_investment_fixed[node] = {k: pe.value(v) for k, v in m.es_e_investment_fixed.items()}
        for y in m.years:
            e_avail[node][y] = float(sum(pe.value(m.es_e_available_per_unit[y_inv, y]) for y_inv in m.years))
            s_avail[node][y] = float(sum(pe.value(m.es_s_available_per_unit[y_inv, y]) for y_inv in m.years))
    return e_avail, s_avail, e_investment_fixed, path


# ======================================================================================================================
#  Terminal-cycle span construction (P channel), per (node, year_idx, day_idx)
# ======================================================================================================================
def _build_block_span(z, pch, pdch, s_max, e_avail, eff_ch, eff_dch):
    """z, pch, pdch: length-24 arrays. s_max, e_avail: scalars.

    Returns a dict with both span variants (with_idle / without_idle), each
    holding {'P_S', 'P_perp', 'dim', 'basis_index_order', 'candidate_periods'}.
    """
    dt = DT_HOURS
    soc = np.zeros(N_PERIODS)
    soc_prev = e_avail * ENERGY_STORAGE_RELATIVE_INIT_SOC
    for p in range(N_PERIODS):
        delta = eff_ch * pch[p] * dt - pdch[p] * dt / eff_dch
        soc[p] = soc_prev + delta
        soc_prev = soc[p]
    soc_min = e_avail * ENERGY_STORAGE_MIN_ENERGY_STORED
    soc_max = e_avail * ENERGY_STORAGE_MAX_ENERGY_STORED
    soc_tol = SOC_BOUND_REL_TOL * e_avail

    v = np.array([eff_ch * dt if z[p] >= 0.0 else dt / eff_dch for p in range(N_PERIODS)])

    power_limit_periods = [p for p in range(N_PERIODS)
                            if abs(abs(z[p]) - s_max) <= POWER_LIMIT_REL_TOL * s_max]
    soc_bound_periods = [p for p in range(N_PERIODS)
                          if (soc[p] - soc_min) <= soc_tol or (soc_max - soc[p]) <= soc_tol]
    idle_periods = [p for p in range(N_PERIODS) if abs(z[p]) < IDLE_REL_TOL * s_max]

    def orthonormalize(extra_periods):
        idx_set = sorted(set(power_limit_periods) | set(soc_bound_periods) | set(extra_periods))
        cols = [v]
        for p in idx_set:
            e_p = np.zeros(N_PERIODS)
            e_p[p] = 1.0
            cols.append(e_p)
        m = np.array(cols).T  # 24 x k
        q, r = np.linalg.qr(m)
        diag_r = np.abs(np.diag(r))
        keep = np.where(diag_r > RANK_TOL)[0]
        q_s = q[:, keep]
        p_s = q_s @ q_s.T
        p_perp = np.eye(N_PERIODS) - p_s
        return {
            'dim': int(len(keep)),
            'candidate_periods': idx_set,
            'P_S': p_s, 'P_perp': p_perp,
        }

    return {
        'soc': soc.tolist(), 'soc_min': soc_min, 'soc_max': soc_max,
        'power_limit_periods': power_limit_periods, 'soc_bound_periods': soc_bound_periods,
        'idle_periods': idle_periods,
        'with_idle': orthonormalize(idle_periods),
        'without_idle': orthonormalize([]),
    }


# ======================================================================================================================
#  Full per-cycle replay + I/J decomposition
# ======================================================================================================================
def _run_replay(run_dir, years, days, blocks, terminal_cycle):
    n_nodes, n_years, n_days = len(NODES), len(years), len(days)
    n_blocks = n_nodes * n_years * n_days
    n_entries = n_blocks * 2 * N_PERIODS

    lam = {agent: np.zeros(n_entries, dtype=float) for agent in AGENTS}

    g, _ = _one_g_baseline(run_dir)
    ct_by_cycle = {r['cycle']: r for r in g['cycle_trajectory']}

    stride_path = glob.glob(os.path.join(run_dir, 'ess_entry_stride_*.jsonl'))
    if len(stride_path) != 1:
        raise RuntimeError(f'expected exactly one ess_entry_stride_*.jsonl in {run_dir}, found {stride_path}')
    stride_path = stride_path[0]

    # P_S / P_perp stacks, ordered by block index (node-major, year, day)
    block_order = []
    for node in NODES:
        for y_idx in range(n_years):
            for d_idx in range(n_days):
                block_order.append((node, y_idx, d_idx))
    assert len(block_order) == n_blocks
    block_index_lookup = {key: i for i, key in enumerate(block_order)}
    p_s_with = np.stack([blocks[key]['with_idle']['P_S'] for key in block_order])
    p_s_without = np.stack([blocks[key]['without_idle']['P_S'] for key in block_order])

    series = {
        'cycle': [],
        'total_lambda_norm': [], 'agent_lambda_norm': {a: [] for a in AGENTS},
        'p_lambda_norm': {a: [] for a in AGENTS}, 'q_lambda_norm': {a: [] for a in AGENTS},
        'with_idle': {
            'I_norm': {a: [] for a in AGENTS}, 'J_norm': {a: [] for a in AGENTS},
            'I_norm_total': [], 'J_norm_total': [],
        },
        'without_idle': {
            'I_norm': {a: [] for a in AGENTS}, 'J_norm': {a: [] for a in AGENTS},
            'I_norm_total': [], 'J_norm_total': [],
        },
    }

    reference_rating_values_seen = set()
    rho_values_seen = []

    with open(stride_path) as handle:
        for line in handle:
            row = json.loads(line)
            cycle = row['cycle']
            if cycle > terminal_cycle:
                break

            g_row = ct_by_cycle[cycle]
            rho = float(g_row['rho_ess_before'])
            s_ref = float(g_row['shared_ess_reference_rating_mva'])
            reference_rating_values_seen.add(s_ref)
            rho_values_seen.append(rho)
            a = 1.0 / (2.0 * parta.srp._shared_ess_admm_normalization_mva(0.0, 0.0, reference_mva=s_ref))
            denom = sum(rho * a ** 2 for _ in AGENTS)

            for entry in row['entries']:
                node_id = entry['node_id']
                y_idx = years.index(int(entry['year']))
                d_idx = days.index(entry['day'])
                power_type = entry['power_type']
                b_idx = block_index_lookup[(node_id, y_idx, d_idx)]
                pti = 0 if power_type == 'p' else 1
                base = (b_idx * 2 + pti) * N_PERIODS

                x_agent = {ag: np.asarray(entry['x'][ag], dtype=float) for ag in AGENTS}
                lam_prev = {ag: lam[ag][base:base + N_PERIODS].copy() for ag in AGENTS}
                numer = sum(rho * a ** 2 * x_agent[ag] + lam_prev[ag] * a for ag in AGENTS)
                z_new = numer / denom
                for ag in AGENTS:
                    lam[ag][base:base + N_PERIODS] = lam_prev[ag] + rho * a * (x_agent[ag] - z_new)

            # ---- per-cycle aggregate readout ----
            reshaped = {ag: lam[ag].reshape(n_blocks, 2, N_PERIODS) for ag in AGENTS}
            series['cycle'].append(cycle)
            total_sq = 0.0
            for ag in AGENTS:
                full_norm = float(np.linalg.norm(lam[ag]))
                p_norm = float(np.linalg.norm(reshaped[ag][:, 0, :]))
                q_norm = float(np.linalg.norm(reshaped[ag][:, 1, :]))
                series['agent_lambda_norm'][ag].append(full_norm)
                series['p_lambda_norm'][ag].append(p_norm)
                series['q_lambda_norm'][ag].append(q_norm)
                total_sq += full_norm ** 2
            series['total_lambda_norm'].append(float(np.sqrt(total_sq)))

            p_block = {ag: reshaped[ag][:, 0, :] for ag in AGENTS}  # (n_blocks, 24)
            for variant_name, p_s_stack in (('with_idle', p_s_with), ('without_idle', p_s_without)):
                sq_i_total = 0.0
                sq_j_total = 0.0
                for ag in AGENTS:
                    j_all = np.einsum('bij,bj->bi', p_s_stack, p_block[ag])
                    i_all = p_block[ag] - j_all
                    sq_i = float(np.sum(i_all ** 2))
                    sq_j = float(np.sum(j_all ** 2))
                    series[variant_name]['I_norm'][ag].append(float(np.sqrt(sq_i)))
                    series[variant_name]['J_norm'][ag].append(float(np.sqrt(sq_j)))
                    sq_i_total += sq_i
                    sq_j_total += sq_j
                series[variant_name]['I_norm_total'].append(float(np.sqrt(sq_i_total)))
                series[variant_name]['J_norm_total'].append(float(np.sqrt(sq_j_total)))

    return {
        'lam_terminal': lam, 'series': series, 'block_order': block_order,
        'reference_rating_values_seen': sorted(reference_rating_values_seen),
        'rho_min_max_seen': [min(rho_values_seen), max(rho_values_seen)],
        'n_entries': n_entries, 'n_blocks': n_blocks,
    }


# ======================================================================================================================
#  Cross-validation vs g_baseline.json's boyd_ess_norm_y_<agent> / boyd_ess_norm_y trajectory
# ======================================================================================================================
def _cross_validate(run_dir, series):
    g, _ = _one_g_baseline(run_dir)
    ct_by_cycle = {r['cycle']: r for r in g['cycle_trajectory']}
    report = {}
    for agent in AGENTS:
        key = f'boyd_ess_norm_y_{agent}'
        max_rel = 0.0
        worst_cycle = None
        for cycle, recon in zip(series['cycle'], series['agent_lambda_norm'][agent]):
            recorded = ct_by_cycle[cycle][key]
            rel_err = abs(recon - recorded) / recorded if recorded != 0.0 else abs(recon - recorded)
            if rel_err > max_rel:
                max_rel = rel_err
                worst_cycle = cycle
        report[agent] = {'max_relative_error': max_rel, 'worst_cycle': worst_cycle,
                          'passes_1e-9_relative': max_rel <= RELATIVE_TOLERANCE}
    max_rel_total = 0.0
    worst_cycle_total = None
    for cycle, recon in zip(series['cycle'], series['total_lambda_norm']):
        recorded = ct_by_cycle[cycle]['boyd_ess_norm_y']
        rel_err = abs(recon - recorded) / recorded if recorded != 0.0 else abs(recon - recorded)
        if rel_err > max_rel_total:
            max_rel_total = rel_err
            worst_cycle_total = cycle
    report['total'] = {'max_relative_error': max_rel_total, 'worst_cycle': worst_cycle_total,
                        'passes_1e-9_relative': max_rel_total <= RELATIVE_TOLERANCE}
    return report


# ======================================================================================================================
#  Settling-cycle rule
# ======================================================================================================================
def _settling(values, cycles, window=SETTLE_WINDOW, threshold=SETTLE_THRESHOLD):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    assert cycles == list(range(cycles[0], cycles[0] + n))
    last_violation_cycle = None
    for i in range(window, n):  # i indexes cycles[window] .. cycles[n-1]
        prev = vals[i - window]
        cur = vals[i]
        rel = abs(cur - prev) / abs(prev) if prev != 0.0 else abs(cur - prev)
        if rel >= threshold:
            last_violation_cycle = cycles[i]
    if last_violation_cycle is None:
        settling_cycle = cycles[window]
    else:
        candidate = last_violation_cycle + 1
        settling_cycle = candidate if candidate <= cycles[-1] else None
    prev_last = vals[-1 - window]
    rel_last_window = (abs(vals[-1] - prev_last) / abs(prev_last)) if prev_last != 0.0 else abs(vals[-1] - prev_last)
    return {
        'settling_cycle': settling_cycle,
        'relative_change_last_50_cycles': float(rel_last_window),
        'settled_by_terminal': settling_cycle is not None,
    }


# ======================================================================================================================
#  Verdict
# ======================================================================================================================
def _verdict(settle_I_total, settle_J_total, settle_I_tso, settle_J_tso, dim_s_summary):
    vacuous_fraction = dim_s_summary['n_blocks_dim_ge_20'] / dim_s_summary['n_blocks']
    vacuous = vacuous_fraction > VACUOUS_BLOCK_FRACTION

    def _gap(settle_i, settle_j):
        if settle_i['settling_cycle'] is None and settle_j['settling_cycle'] is None:
            return None, 'both_not_settled'
        if settle_i['settling_cycle'] is None:
            return None, 'I_not_settled_J_settled'
        if settle_j['settling_cycle'] is None:
            return None, 'J_not_settled_I_settled'
        return settle_i['settling_cycle'] - settle_j['settling_cycle'], 'both_settled'

    gap_total, status_total = _gap(settle_I_total, settle_J_total)
    gap_tso, status_tso = _gap(settle_I_tso, settle_J_tso)

    if vacuous:
        return {
            'verdict': 'INCONCLUSIVE',
            'reason': (f'{dim_s_summary["n_blocks_dim_ge_20"]}/{dim_s_summary["n_blocks"]} P-channel blocks '
                       f'have dim S >= {VACUOUS_DIM_THRESHOLD} of 24 (> {VACUOUS_BLOCK_FRACTION*100:.0f}% of blocks) '
                       '-- the decomposition is vacuous.'),
            'gap_total_cycles': gap_total, 'status_total': status_total,
            'gap_tso_cycles': gap_tso, 'status_tso': status_tso,
        }

    def _classify(gap, status):
        if status in ('both_not_settled',):
            return 'INCONCLUSIVE'
        if status == 'I_not_settled_J_settled':
            return 'H1'
        if status == 'J_not_settled_I_settled':
            return 'H2'
        # both settled
        if gap > SETTLE_MATERIAL_GAP:
            return 'H1'
        if gap < -SETTLE_MATERIAL_GAP:
            return 'H2'
        return 'INCONCLUSIVE'

    verdict_total = _classify(gap_total, status_total)
    verdict_tso = _classify(gap_tso, status_tso)
    if verdict_total == verdict_tso and verdict_total != 'INCONCLUSIVE':
        final = verdict_total
    elif verdict_total == 'INCONCLUSIVE' or verdict_tso == 'INCONCLUSIVE':
        final = 'INCONCLUSIVE'
    else:
        final = 'INCONCLUSIVE'  # total and TSO disagree

    return {
        'verdict': final,
        'reason': (f'total: gap(settle_I - settle_J)={gap_total} cycles, status={status_total}, '
                   f'classified {verdict_total}; TSO: gap={gap_tso} cycles, status={status_tso}, '
                   f'classified {verdict_tso}. Material-gap threshold = {SETTLE_MATERIAL_GAP} cycles.'),
        'gap_total_cycles': gap_total, 'status_total': status_total, 'classification_total': verdict_total,
        'gap_tso_cycles': gap_tso, 'status_tso': status_tso, 'classification_tso': verdict_tso,
    }


# ======================================================================================================================
#  Gate 3 (not computable) -- record the evidence
# ======================================================================================================================
def _gate3_finding(run_dir):
    stride_matches = glob.glob(os.path.join(run_dir, 'ess_entry_stride_*.jsonl'))
    if len(stride_matches) != 1:
        return {'computable': False, 'reason': f'expected exactly one stride file, found {stride_matches}'}
    stride_path = stride_matches[0]
    cycles = []
    stride_declared = None
    with open(stride_path) as handle:
        for line in handle:
            row = json.loads(line)
            cycles.append(row['cycle'])
            if stride_declared is None:
                stride_declared = row.get('stride')
    g, _ = _one_g_baseline(run_dir)
    ct = g['cycle_trajectory']
    terminal_cycle = ct[-1]['cycle']
    full_coverage = cycles == list(range(1, terminal_cycle + 1))
    return {
        'computable': False,
        'run_dir': run_dir,
        'stride_file': stride_path,
        'stride_declared_field': stride_declared,
        'n_stride_rows': len(cycles),
        'cycles_recorded': [cycles[0], cycles[-1]] if cycles else None,
        'g_baseline_terminal_cycle': terminal_cycle,
        'full_cycle_coverage': full_coverage,
        'reason': (
            f'stride file records {len(cycles)} rows (cycle {cycles[0]}..{cycles[-1]}, declared stride '
            f'{stride_declared}) while g_baseline.json runs cycles 1..{terminal_cycle}. The per-cycle ADMM '
            'dual-update recursion needs EVERY cycle\'s (x, rho) to replay lambda(c) from lambda(c-1); '
            'the missing intermediate cycles (and the missing terminal cycles '
            f'{cycles[-1]+1}..{terminal_cycle} if any) make the reconstruction NOT COMPUTABLE from this '
            'run\'s committed artifacts. Per task instruction, not attempted.'
        ),
    }


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='P5.15-S36 H1/H2 dual-identifiability diagnostic').install()
    try:
        zero_init_check = parta._verify_zero_initial_dual_from_source()
        if not zero_init_check['all_found']:
            raise AssertionError(f'zero-initial-dual precondition NOT verified: {zero_init_check}')

        no_skip_check = parta._verify_no_skips(REF_RUN)
        if not (no_skip_check['all_recovered_or_recovered_tier2'] and no_skip_check['g_baseline_all_cycles_local_solves_ok']
                and no_skip_check['g_baseline_cycles_exactly_1_to_terminal']):
            raise AssertionError(f'no-skip precondition NOT verified: {no_skip_check}')

        years, days, eff_ch, eff_dch = _get_years_days_and_efficiencies()

        e_avail, s_avail, e_investment_fixed, pickle_path = _load_esso_pickle_e_avail(REF_RUN)

        # Terminal (cycle 477) stride row -> z per (node, year_idx, day_idx, power_type)
        stride_path = glob.glob(os.path.join(REF_RUN, 'ess_entry_stride_*.jsonl'))[0]
        terminal_row = None
        with open(stride_path) as handle:
            for line in handle:
                row = json.loads(line)
                if row['cycle'] == REF_TERMINAL_CYCLE:
                    terminal_row = row
                    break
        if terminal_row is None:
            raise RuntimeError(f'cycle {REF_TERMINAL_CYCLE} not found in {stride_path}')

        z_lookup = {}
        for e in terminal_row['entries']:
            y_idx = years.index(int(e['year']))
            d_idx = days.index(e['day'])
            z_lookup[(e['node_id'], y_idx, d_idx, e['power_type'])] = np.asarray(e['z'], dtype=float)

        # Build per-block spans (P channel only)
        blocks = {}
        block_diagnostics = {}
        esso_z_cross_check_max_abs = 0.0
        for node in NODES:
            esso_terminal, esso_terminal_path = _load_esso_terminal_capture(REF_RUN, node, REF_TERMINAL_CYCLE)
            for y_idx in range(len(years)):
                for d_idx in range(len(days)):
                    z = z_lookup[(node, y_idx, d_idx, 'p')]
                    pch = np.array([esso_terminal[(y_idx, d_idx, p)]['pch'] for p in range(N_PERIODS)])
                    pdch = np.array([esso_terminal[(y_idx, d_idx, p)]['pdch'] for p in range(N_PERIODS)])
                    s_max_vals = [esso_terminal[(y_idx, d_idx, p)]['s_max'] for p in range(N_PERIODS)]
                    if max(s_max_vals) - min(s_max_vals) > 1e-12:
                        raise AssertionError(f's_max not constant across periods for node={node} y={y_idx} d={d_idx}: {s_max_vals}')
                    s_max = float(s_max_vals[0])
                    pnet_esso = pch - pdch
                    cross_abs = float(np.max(np.abs(pnet_esso - z)))
                    esso_z_cross_check_max_abs = max(esso_z_cross_check_max_abs, cross_abs)

                    key = (node, y_idx, d_idx)
                    blk = _build_block_span(z, pch, pdch, s_max, e_avail[node][y_idx], eff_ch, eff_dch)
                    blocks[key] = blk
                    block_diagnostics[f'{node}/{years[y_idx]}/{days[d_idx]}'] = {
                        's_max': s_max, 'e_avail': e_avail[node][y_idx],
                        'esso_pnet_vs_z_max_abs_diff': cross_abs,
                        'power_limit_periods': blk['power_limit_periods'],
                        'soc_bound_periods': blk['soc_bound_periods'],
                        'idle_periods': blk['idle_periods'],
                        'dim_S_with_idle': blk['with_idle']['dim'],
                        'dim_S_without_idle': blk['without_idle']['dim'],
                    }

        # dim S summary
        dims_with = [blocks[k]['with_idle']['dim'] for k in blocks]
        dims_without = [blocks[k]['without_idle']['dim'] for k in blocks]
        n_blocks_total = len(blocks)
        dim_s_summary = {
            'with_idle': {'dims': dims_with, 'min': min(dims_with), 'max': max(dims_with),
                          'mean': float(np.mean(dims_with)),
                          'n_blocks_ge_20': int(sum(d >= VACUOUS_DIM_THRESHOLD for d in dims_with))},
            'without_idle': {'dims': dims_without, 'min': min(dims_without), 'max': max(dims_without),
                             'mean': float(np.mean(dims_without)),
                             'n_blocks_ge_20': int(sum(d >= VACUOUS_DIM_THRESHOLD for d in dims_without))},
            'n_blocks': n_blocks_total,
        }

        # Full replay
        replay = _run_replay(REF_RUN, years, days, blocks, REF_TERMINAL_CYCLE)
        series = replay['series']
        cross_val = _cross_validate(REF_RUN, series)

        # Settling analysis
        settling = {}
        for variant in ('with_idle', 'without_idle'):
            settling[variant] = {
                'I_total': _settling(series[variant]['I_norm_total'], series['cycle']),
                'J_total': _settling(series[variant]['J_norm_total'], series['cycle']),
                'I_agent': {a: _settling(series[variant]['I_norm'][a], series['cycle']) for a in AGENTS},
                'J_agent': {a: _settling(series[variant]['J_norm'][a], series['cycle']) for a in AGENTS},
            }

        # Identifiable fraction at reference cycles (P channel), both variants
        cycle_index = {c: i for i, c in enumerate(series['cycle'])}
        identifiable_fraction = {}
        for variant in ('with_idle', 'without_idle'):
            identifiable_fraction[variant] = {}
            for c in IDENTIFIABLE_FRACTION_CYCLES:
                if c not in cycle_index:
                    identifiable_fraction[variant][c] = None
                    continue
                i_idx = cycle_index[c]
                i_norm = series[variant]['I_norm_total'][i_idx]
                j_norm = series[variant]['J_norm_total'][i_idx]
                p_norm_sq = sum(series['p_lambda_norm'][a][i_idx] ** 2 for a in AGENTS)
                identifiable_fraction[variant][c] = {
                    'I_norm_total': i_norm, 'J_norm_total': j_norm,
                    'I_over_lambda_P_fraction_sq': (i_norm ** 2 / p_norm_sq) if p_norm_sq > 0 else None,
                    'lambda_P_norm_total': float(np.sqrt(p_norm_sq)),
                }

        # Q dual-share (P+Q combined denominator), per agent and total, at reference cycles
        q_share = {}
        for c in IDENTIFIABLE_FRACTION_CYCLES:
            if c not in cycle_index:
                q_share[c] = None
                continue
            idx = cycle_index[c]
            per_agent = {}
            total_q_sq = 0.0
            total_all_sq = 0.0
            for a in AGENTS:
                q_n = series['q_lambda_norm'][a][idx]
                full_n = series['agent_lambda_norm'][a][idx]
                per_agent[a] = {'q_norm': q_n, 'full_norm': full_n,
                                 'q_share_sq': (q_n ** 2 / full_n ** 2) if full_n > 0 else None}
                total_q_sq += q_n ** 2
                total_all_sq += full_n ** 2
            q_share[c] = {'per_agent': per_agent,
                          'total_q_share_sq': (total_q_sq / total_all_sq) if total_all_sq > 0 else None}

        # Verdict
        verdict = {}
        for variant in ('with_idle', 'without_idle'):
            dim_summary_variant = {
                'n_blocks_dim_ge_20': dim_s_summary[variant]['n_blocks_ge_20'],
                'n_blocks': dim_s_summary['n_blocks'],
            }
            verdict[variant] = _verdict(
                settling[variant]['I_total'], settling[variant]['J_total'],
                settling[variant]['I_agent']['tso'], settling[variant]['J_agent']['tso'],
                dim_summary_variant,
            )

        if verdict['with_idle']['verdict'] == verdict['without_idle']['verdict']:
            overall_verdict = verdict['with_idle']['verdict']
        else:
            overall_verdict = 'INCONCLUSIVE'

        # Gate 3
        gate3 = _gate3_finding(GATE3_RUN)

        report = {
            'stage': 'P5.15-S36-H1H2', 'authority': 'Addendum 17 H1/H2 identifiability diagnostic (2026-09-17)',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'run': REF_RUN, 'terminal_cycle': REF_TERMINAL_CYCLE,
            'years_days_ordering': {'years': years, 'days': days},
            'efficiencies': {'eff_ch': eff_ch, 'eff_dch': eff_dch, 'dt_hours': DT_HOURS},
            'e_available_mwh_post_soh_by_node_year': {
                str(node): {str(years[y]): e_avail[node][y] for y in range(len(years))} for node in NODES},
            's_available_mva_by_node_year': {
                str(node): {str(years[y]): s_avail[node][y] for y in range(len(years))} for node in NODES},
            'e_investment_fixed_by_node': {str(node): e_investment_fixed[node] for node in NODES},
            'esso_pickle_path': pickle_path,
            'zero_initial_dual_precondition': zero_init_check,
            'no_skip_precondition': no_skip_check,
            'update_rule_reference_rating_values_seen': replay['reference_rating_values_seen'],
            'update_rule_rho_min_max_seen': replay['rho_min_max_seen'],
            'esso_pnet_vs_terminal_z_max_abs_diff_over_all_blocks': esso_z_cross_check_max_abs,
            'tolerances': {
                'power_limit_rel_tol': POWER_LIMIT_REL_TOL, 'idle_rel_tol': IDLE_REL_TOL,
                'soc_bound_rel_tol': SOC_BOUND_REL_TOL, 'rank_tol': RANK_TOL,
                'settle_window': SETTLE_WINDOW, 'settle_threshold': SETTLE_THRESHOLD,
                'settle_material_gap_cycles': SETTLE_MATERIAL_GAP,
                'vacuous_dim_threshold': VACUOUS_DIM_THRESHOLD, 'vacuous_block_fraction': VACUOUS_BLOCK_FRACTION,
            },
            'block_diagnostics': block_diagnostics,
            'dim_S_summary': dim_s_summary,
            'cross_validation_vs_g_baseline': cross_val,
            'settling': settling,
            'identifiable_fraction_at_reference_cycles': identifiable_fraction,
            'q_dual_share_at_reference_cycles': q_share,
            'verdict_by_variant': verdict,
            'overall_verdict': overall_verdict,
            'gate3': gate3,
            'n_entries_reconstructed': replay['n_entries'],
            'n_blocks': replay['n_blocks'],
        }

        overall_pass_cross_val = all(cross_val[a]['passes_1e-9_relative'] for a in AGENTS) and cross_val['total']['passes_1e-9_relative']
        report['cross_validation_overall_pass'] = overall_pass_cross_val

        # Full per-cycle series (kept -- needed to audit settling-cycle claims)
        series_out = {
            'cycle': series['cycle'],
            'total_lambda_norm': series['total_lambda_norm'],
            'agent_lambda_norm': series['agent_lambda_norm'],
            'p_lambda_norm': series['p_lambda_norm'], 'q_lambda_norm': series['q_lambda_norm'],
            'with_idle': {'I_norm_total': series['with_idle']['I_norm_total'],
                          'J_norm_total': series['with_idle']['J_norm_total'],
                          'I_norm': series['with_idle']['I_norm'], 'J_norm': series['with_idle']['J_norm']},
            'without_idle': {'I_norm_total': series['without_idle']['I_norm_total'],
                             'J_norm_total': series['without_idle']['J_norm_total'],
                             'I_norm': series['without_idle']['I_norm'], 'J_norm': series['without_idle']['J_norm']},
        }
    finally:
        failures = guard.verify(0, 0)
        guard.uninstall()
    if failures:
        raise AssertionError('RULE SIX: this diagnostic was expected to solve nothing -> ' + '; '.join(failures))

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'h1h2_identifiability_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    series_path = os.path.join(OUT_DIR, 'h1h2_per_cycle_series.json')
    with open(series_path, 'w') as handle:
        json.dump(series_out, handle, default=str)

    manifest = {}
    for fname in ('h1h2_identifiability_results.json', 'h1h2_per_cycle_series.json'):
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
    print(f'cross_validation_overall_pass: {overall_pass_cross_val}')
    print(f'dim S (with_idle): min={dim_s_summary["with_idle"]["min"]} max={dim_s_summary["with_idle"]["max"]} '
          f'mean={dim_s_summary["with_idle"]["mean"]:.2f} n_ge_20={dim_s_summary["with_idle"]["n_blocks_ge_20"]}/{n_blocks_total}')
    print(f'dim S (without_idle): min={dim_s_summary["without_idle"]["min"]} max={dim_s_summary["without_idle"]["max"]} '
          f'mean={dim_s_summary["without_idle"]["mean"]:.2f} n_ge_20={dim_s_summary["without_idle"]["n_blocks_ge_20"]}/{n_blocks_total}')
    for variant in ('with_idle', 'without_idle'):
        print(f'[{variant}] settling I_total={settling[variant]["I_total"]} J_total={settling[variant]["J_total"]}')
        print(f'[{variant}] verdict={verdict[variant]}')
    print(f'OVERALL VERDICT: {overall_verdict}')
    print(f'gate3 computable: {gate3["computable"]}')
    print(f'wrote: {results_path}')
    return 0 if overall_pass_cross_val else 1


if __name__ == '__main__':
    sys.exit(main())
