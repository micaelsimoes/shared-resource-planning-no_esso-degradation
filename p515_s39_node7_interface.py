"""P5.15 Addendum 21 item (3) -- node 7 TSO-DSO interface, zero-solve look (S39).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 21; `P5_15_ADDENDUM20_EXPERT_REPORT.md`
section 3.4 and `P5_15_S38_PF_PACE_REPORT.md` section 3.3 (the late PF dual residual
concentrates >=75-95% at the node 7 interface, ~100% active power, 2030 dominant at the
stop). Frozen spec v10 (`data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json`,
`e2fd8e60`) lists this look under "parallel_work", extended to arms C and D when available.

Purpose: read ONLY already-committed / already-hash-recorded per-run artifacts for a given
v9/v10 arm run directory (no re-run, no solve) and assemble, for the node-7 TSO-DSO
interface at that arm's stop cycle (and over its last `STRIDE_LAST_N` cycles), the evidence
needed to assess whether the concentration is explained by:
  (a) the interface flow sitting near its branch rating;
  (b) a voltage bound at node 7 being active;
  (c) a DSO flexibility bound being active;
  (d) the shared storage dispatch at node 7 sitting at / cycling near a bound;
  (e) none of the above being observable from saved data.
Nodes 5 and 9 are carried throughout as comparators. This script computes and reports
observations only; it draws no conclusion about which explanation applies (that is done in
the accompanying worker report) and proposes no algorithm change.

Zero solves: `SolveProfileGuard` (permitted=()) is installed before ANY input file is
touched (including the `esso_models_*.pkl` unpickle, which is a plain `pickle.load` -- no
solver call -- but is done under the guard anyway as a belt-and-braces measure) and
verified with `guard.verify(0)` before any output is written.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_s39_node7_interface.py RUN_DIR LABEL [--dry-run]

RUN_DIR is the arm's run directory, e.g.
`data/SRP1/Results/P515S38_A_TAU0_run` (label `s38_A_tau0`) or
`data/SRP1/Results/P515S38_B_PFBAL_run` (label `s38_B_pfbal`), and later
`data/SRP1/Results/P515S39_C_run` / `P515S39_D_run` for arms C and D. LABEL is used only to
name the output subdirectory and to locate the run's own `g_<label>.json` /
`pf_entry_stride_<label>.jsonl` / `ess_entry_stride_baseline.jsonl` files inside RUN_DIR,
and (best-effort, optional) the arm's IPOPT log directory under
`data/SRP1/Results/P56A/evals/`.

Writes (write-once, refuses to overwrite):
    data/SRP1/Results/P515S39/node7_interface/<LABEL>/node7_interface_<LABEL>.json
    data/SRP1/Results/P515S39/node7_interface/<LABEL>/run_log_<LABEL>.txt
    data/SRP1/Results/P515S39/node7_interface/<LABEL>/evidence_manifest_sha256.json

The large per-cycle stride captures (`pf_entry_stride_*.jsonl`, `ess_entry_stride_baseline.jsonl`,
80-96 MB each) are hash-recorded in the manifest, not copied.
"""
import glob
import hashlib
import json
import math
import os
import pickle
import sys
from collections import deque
from datetime import datetime, timezone

import pyomo.environ as pe  # noqa: F401 -- required to unpickle the ESSO models' Vars

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

NODES = (5, 7, 9)
FOCUS_NODE = 7
FOCUS_YEAR = '2030'
STRIDE_LAST_N = 30
TOP_K_ALL = 15
TOP_K_FOCUS = 10
UTIL_NEAR_1PCT = 0.99
UTIL_NEAR_5PCT = 0.95
VOLT_NEAR_1PCT_PU = 0.01
VOLT_NEAR_5PCT_PU = 0.05

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39', 'node7_interface')
SCRIPT_REL = os.path.relpath(os.path.abspath(__file__), REPO)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _one(pattern, required=True):
    hits = sorted(glob.glob(pattern))
    if len(hits) == 1:
        return hits[0]
    if not hits and not required:
        return None
    raise RuntimeError(f'expected exactly one file for {pattern!r}, found {hits}')


def _read_last_n_jsonl(path, n):
    dq = deque(maxlen=n)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                dq.append(line)
    return [json.loads(item) for item in dq]


def _read_first_line_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _entry_key(e):
    return (e['node_id'], e['year'], e['day'], e['power_type'], e['period'])


def _sumsq(entries, pred=None):
    return sum(e['s'] ** 2 for e in entries if pred is None or pred(e))


def _node_decomposition(entries):
    total = _sumsq(entries)
    out = {'total_s2': total}
    if total <= 0:
        out['note'] = 'total_s2 is zero or negative; fractions undefined'
        return out
    out['p_frac'] = _sumsq(entries, lambda e: e['power_type'] == 'p') / total
    out['q_frac'] = _sumsq(entries, lambda e: e['power_type'] == 'q') / total
    out['by_node_frac'] = {str(n): _sumsq(entries, lambda e, n=n: e['node_id'] == n) / total for n in NODES}
    years = sorted(set(e['year'] for e in entries))
    out['by_year_frac'] = {y: _sumsq(entries, lambda e, y=y: e['year'] == y) / total for y in years}
    out['node7_p_2030_frac'] = _sumsq(
        entries,
        lambda e: e['node_id'] == FOCUS_NODE and e['power_type'] == 'p' and e['year'] == FOCUS_YEAR,
    ) / total
    return out


def _top_entries(entries, k, pred=None):
    pool = [e for e in entries if pred is None or pred(e)]
    pool = sorted(pool, key=lambda e: e['s'] ** 2, reverse=True)[:k]
    out = []
    for e in pool:
        rating = e['interface_rating']
        out.append({
            'node_id': e['node_id'], 'year': e['year'], 'day': e['day'],
            'power_type': e['power_type'], 'period': e['period'],
            's': e['s'], 'r': e['r'], 's2': e['s'] ** 2,
            'x_dso': e['x_dso'], 'z_tso_current': e['z_tso_current'], 'z_tso_prev': e['z_tso_prev'],
            'lambda_dso': e['lambda_dso'], 'interface_rating': rating,
            'rho_pf': e['rho_pf'], 'gamma_pf': e['gamma_pf'],
            'utilization_p_abs_over_rating': (abs(e['x_dso']) / rating) if rating else None,
        })
    return out


def _series_for_keys(rows, keys):
    keyset = set(keys)
    out = {k: {'cycle': [], 's': [], 'r': [], 'x_dso': [], 'z_tso_current': []} for k in keyset}
    for row in rows:
        cyc = row['cycle']
        for e in row['entries']:
            k = _entry_key(e)
            if k in keyset:
                slot = out[k]
                slot['cycle'].append(cyc)
                slot['s'].append(e['s'])
                slot['r'].append(e['r'])
                slot['x_dso'].append(e['x_dso'])
                slot['z_tso_current'].append(e['z_tso_current'])
    return out


def _monotonicity(values, tol=1e-12):
    if len(values) < 2:
        return {'classification': 'insufficient_data', 'n': len(values)}
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    signs = [1 if d > tol else (-1 if d < -tol else 0) for d in diffs]
    nonzero = [s for s in signs if s != 0]
    if not nonzero:
        return {'classification': 'flat', 'n_sign_changes': 0, 'n_steps': len(diffs)}
    sign_changes = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
    classification = 'oscillating' if sign_changes > 0 else (
        'monotone_decreasing' if nonzero[0] < 0 else 'monotone_increasing')
    return {'classification': classification, 'n_sign_changes': sign_changes, 'n_steps': len(diffs)}


def _key_label(k):
    node_id, year, day, power_type, period = k
    return f'node{node_id}|{year}|{day}|{power_type}|p{period}'


# ---------------------------------------------------------------------------
# PF interface channel
# ---------------------------------------------------------------------------

def analyze_pf_channel(pf_path, log):
    log(f'reading PF per-entry stride (last {STRIDE_LAST_N} cycles + full-file line count): {pf_path}')
    n_lines = 0
    with open(pf_path) as f:
        for _ in f:
            n_lines += 1
    log(f'  {n_lines} cycle-rows (stride 1, so {n_lines} cycles captured)')
    rows = _read_last_n_jsonl(pf_path, STRIDE_LAST_N)
    stop_row = rows[-1]
    stop_cycle = stop_row['cycle']
    identity_ok = all(r.get('identity_holds') for r in rows)
    max_rel_err_r = max(r.get('rel_err_r', 0.0) for r in rows)
    max_rel_err_s = max(r.get('rel_err_s', 0.0) for r in rows)
    log(f'  stop cycle (last captured row) = {stop_cycle}; identity holds on last '
        f'{len(rows)} rows = {identity_ok} (max rel_err_r={max_rel_err_r:.3e}, max rel_err_s={max_rel_err_s:.3e})')

    stop_entries = stop_row['entries']
    decomposition = _node_decomposition(stop_entries)
    top_all = _top_entries(stop_entries, TOP_K_ALL)
    top_focus = _top_entries(
        stop_entries, TOP_K_FOCUS,
        pred=lambda e: e['node_id'] == FOCUS_NODE and e['power_type'] == 'p' and e['year'] == FOCUS_YEAR,
    )
    key_union = {tuple((e['node_id'], e['year'], e['day'], e['power_type'], e['period'])) for e in top_all + top_focus}
    series = _series_for_keys(rows, key_union)
    series_out = {}
    for k, s in series.items():
        mono = _monotonicity([abs(v) for v in s['s']])
        r_signs = [1 if v > 1e-12 else (-1 if v < -1e-12 else 0) for v in s['r']]
        r_nonzero = [x for x in r_signs if x != 0]
        r_sign_flips = sum(1 for i in range(1, len(r_nonzero)) if r_nonzero[i] != r_nonzero[i - 1])
        series_out[_key_label(k)] = {
            'cycles': s['cycle'], 's': s['s'], 'r': s['r'],
            'abs_s_monotonicity_over_window': mono,
            'r_sign_flips_over_window': r_sign_flips,
        }

    # per-node interface_rating (should be node-constant; recorded, not assumed)
    ratings = {}
    for n in NODES:
        vals = sorted({e['interface_rating'] for e in stop_entries if e['node_id'] == n})
        ratings[str(n)] = vals

    return {
        'input_file': os.path.relpath(pf_path, REPO),
        'n_cycle_rows_in_file': n_lines,
        'stop_cycle': stop_cycle,
        'window_cycles_analyzed': [rows[0]['cycle'], rows[-1]['cycle']],
        'capture_identity_check': {
            'identity_holds_all_rows_in_window': identity_ok,
            'max_rel_err_r_in_window': max_rel_err_r,
            'max_rel_err_s_in_window': max_rel_err_s,
        },
        'interface_rating_mva_per_node': ratings,
        'stop_decomposition_of_s2': decomposition,
        'top_entries_by_s2_at_stop_all_nodes': top_all,
        'top_entries_by_s2_at_stop_node7_p_2030': top_focus,
        'per_entry_series_last_window': series_out,
    }, stop_cycle, stop_entries


# ---------------------------------------------------------------------------
# interface flow vs rating (all periods, not just the top-s entries)
# ---------------------------------------------------------------------------

def analyze_interface_flow(pf_stop_entries, log):
    log('computing interface-flow utilization (S = sqrt(P^2+Q^2)) vs interface_rating at the stop cycle')
    by_key = {}
    for e in pf_stop_entries:
        base_key = (e['node_id'], e['year'], e['day'], e['period'])
        slot = by_key.setdefault(base_key, {'rating': e['interface_rating']})
        slot[e['power_type']] = e['x_dso']
        slot[e['power_type'] + '_z'] = e['z_tso_current']

    per_node = {}
    for n in NODES:
        recs = []
        for (node_id, year, day, period), v in by_key.items():
            if node_id != n:
                continue
            p = v.get('p', 0.0)
            q = v.get('q', 0.0)
            s_flow = math.sqrt(p * p + q * q)
            rating = v['rating']
            util = s_flow / rating if rating else None
            recs.append({
                'year': year, 'day': day, 'period': period, 'p_mw': p, 'q_mvar': q,
                's_flow_mva': s_flow, 'rating_mva': rating, 'utilization_s': util,
                'utilization_p_only': abs(p) / rating if rating else None,
            })
        n_total = len(recs)
        n_1pct = sum(1 for r in recs if r['utilization_s'] is not None and r['utilization_s'] >= UTIL_NEAR_1PCT)
        n_5pct = sum(1 for r in recs if r['utilization_s'] is not None and r['utilization_s'] >= UTIL_NEAR_5PCT)
        recs_2030 = [r for r in recs if r['year'] == FOCUS_YEAR]
        n_2030_1pct = sum(1 for r in recs_2030 if r['utilization_s'] is not None and r['utilization_s'] >= UTIL_NEAR_1PCT)
        n_2030_5pct = sum(1 for r in recs_2030 if r['utilization_s'] is not None and r['utilization_s'] >= UTIL_NEAR_5PCT)
        top5 = sorted(recs, key=lambda r: r['utilization_s'] or 0.0, reverse=True)[:5]
        per_node[str(n)] = {
            'n_periods_total': n_total,
            'max_utilization_s': max((r['utilization_s'] for r in recs), default=None),
            'mean_utilization_s': (sum(r['utilization_s'] for r in recs) / n_total) if n_total else None,
            'n_within_1pct_of_rating': n_1pct,
            'n_within_5pct_of_rating': n_5pct,
            'n_periods_2030': len(recs_2030),
            'n_within_1pct_of_rating_2030': n_2030_1pct,
            'n_within_5pct_of_rating_2030': n_2030_5pct,
            'top5_periods_by_utilization': top5,
        }
    log(f"  node {FOCUS_NODE} max utilization_s = {per_node[str(FOCUS_NODE)]['max_utilization_s']}")
    return per_node


# ---------------------------------------------------------------------------
# interface voltage vs bounds (terminal-cycle capture)
# ---------------------------------------------------------------------------

def analyze_voltage(voltage_path, log):
    log(f'reading terminal interface voltage: {voltage_path}')
    d = json.load(open(voltage_path))
    entries = d['entries']
    per_node = {}
    for n in NODES:
        node_entries = [e for e in entries if e['node_id'] == n]
        n_total = len(node_entries)
        n_1pct = sum(1 for e in node_entries if e['distance_to_nearest_bound_pu'] <= VOLT_NEAR_1PCT_PU)
        n_5pct = sum(1 for e in node_entries if e['distance_to_nearest_bound_pu'] <= VOLT_NEAR_5PCT_PU)
        min_dist_entry = min(node_entries, key=lambda e: e['distance_to_nearest_bound_pu']) if node_entries else None
        by_year = {}
        for y in sorted({e['year'] for e in node_entries}):
            ye = [e for e in node_entries if e['year'] == y]
            by_year[y] = {
                'n_total': len(ye),
                'min_distance_to_bound_pu': min((e['distance_to_nearest_bound_pu'] for e in ye), default=None),
                'n_within_1pct_pu': sum(1 for e in ye if e['distance_to_nearest_bound_pu'] <= VOLT_NEAR_1PCT_PU),
                'n_within_5pct_pu': sum(1 for e in ye if e['distance_to_nearest_bound_pu'] <= VOLT_NEAR_5PCT_PU),
            }
        per_node[str(n)] = {
            'n_total': n_total,
            'n_within_1pct_pu': n_1pct,
            'n_within_5pct_pu': n_5pct,
            'min_distance_to_bound_pu': min_dist_entry['distance_to_nearest_bound_pu'] if min_dist_entry else None,
            'min_distance_entry': min_dist_entry,
            'by_year': by_year,
        }
    log(f"  node {FOCUS_NODE} min distance to voltage bound (pu) = {per_node[str(FOCUS_NODE)]['min_distance_to_bound_pu']}")
    return {
        'input_file': os.path.relpath(voltage_path, REPO),
        'cycle': d['summary'].get('cycle'),
        'source_summary': d['summary'],
        'per_node': per_node,
    }


# ---------------------------------------------------------------------------
# flexibility usage (bound NOT available in this run's terminal artifacts -- see note)
# ---------------------------------------------------------------------------

def analyze_flexibility(settlement_path, component_levels_path, log):
    log(f'reading interface settlement detail (flexibility usage): {settlement_path}')
    sd = json.load(open(settlement_path))
    flex_vol = {str(n): sd['flexibility_volumes_per_dso'].get(str(n)) for n in NODES}

    log(f'reading component levels terminal (per-block flexibility cost): {component_levels_path}')
    cl = json.load(open(component_levels_path))
    blocks = cl['blocks']
    per_node_cost = {}
    for n in NODES:
        node_blocks = {k: v for k, v in blocks.items() if k.startswith(f'DSO|{n}|')}
        by_year = {}
        for key, v in node_blocks.items():
            year = v['year']
            by_year.setdefault(year, []).append({
                'day': v['day'],
                'flexibility_cost_internal_unweighted': v['unweighted']['flexibility_cost_internal'],
                'flexibility_cost_internal_weighted': v['weighted']['flexibility_cost_internal'],
            })
        per_node_cost[str(n)] = by_year

    return {
        'input_files': [os.path.relpath(settlement_path, REPO), os.path.relpath(component_levels_path, REPO)],
        'usage_volumes_per_node': flex_vol,
        'usage_note': 'sum/max |delta P|,|delta Q| in MW/Mvar and pu -- ACTIVATION VOLUME (usage), not a bound',
        'flexibility_cost_internal_per_node_year_day': per_node_cost,
        'bound_not_captured_note': (
            'No terminal capture artifact of this run records a per-node/per-period flexibility BOUND '
            '(max/min flexible load or delta-P/Q limit). Searched: interface_settlement_detail_s31c.json '
            '(flexibility_volumes_per_dso and interface_reporting_detail carry usage/activation only), '
            'component_levels_terminal.json (cost only), boyd_terminal.json (no flexibility bound field), '
            'case33_2_params.json (no scalar flexibility-limit field). The underlying case data source for a '
            'device-level flexible-load bound is data/SRP1/case33_2/case33_2_operational_data.xlsx, sheet '
            '"Flex" (network-wide min/max flexible load per period, per season, per market scenario, before '
            'per-year growth-factor scaling); reconciling that source against the per-period usage above (with '
            'the correct year growth factor and market scenario) was NOT attempted in this stage -- scope '
            'limited to already-captured terminal/stride artifacts. This is a negative claim about node 7\'s '
            'DSO (case33_2) specifically; the same three JSON files and the params file were checked, not the '
            'full repository.'
        ),
    }


# ---------------------------------------------------------------------------
# storage dispatch at node 7 (and comparators) vs rating and SoH floor
# ---------------------------------------------------------------------------

def analyze_storage(ess_path, soh_path, g, esso_pkl_path, instance, log):
    log(f'reading ESS entry stride (stop row): {ess_path}')
    with open(ess_path) as f:
        n_ess_lines = sum(1 for _ in f)
    ess_stop = _read_last_n_jsonl(ess_path, 1)[0]
    stop_cycle = ess_stop['cycle']
    s_rated = instance['s_mva']

    by_key = {}
    for e in ess_stop['entries']:
        base_key = (e['node_id'], e['year'], e['day'])
        slot = by_key.setdefault(base_key, {})
        slot[e['power_type']] = e

    per_node = {}
    for n in NODES:
        period_recs = []
        for (node_id, year, day), v in by_key.items():
            if node_id != n:
                continue
            p_entry = v.get('p')
            q_entry = v.get('q')
            if p_entry is None:
                continue
            n_periods = len(p_entry['z'])
            for period in range(n_periods):
                p_z = p_entry['z'][period]
                q_z = q_entry['z'][period] if q_entry is not None else 0.0
                s_z = math.sqrt(p_z * p_z + q_z * q_z)
                util = s_z / s_rated if s_rated else None
                spread = max(
                    abs(p_entry['x']['tso'][period] - p_z),
                    abs(p_entry['x']['dso'][period] - p_z),
                    abs(p_entry['x']['esso'][period] - p_z),
                )
                period_recs.append({
                    'year': year, 'day': day, 'period': period,
                    'p_z_mw': p_z, 'q_z_mvar': q_z, 's_z_mva': s_z,
                    'utilization': util,
                    'consensus_spread_p_mw': spread,
                    'x_tso': p_entry['x']['tso'][period], 'x_dso': p_entry['x']['dso'][period],
                    'x_esso': p_entry['x']['esso'][period],
                })
        n_total = len(period_recs)
        n_1pct = sum(1 for r in period_recs if r['utilization'] is not None and r['utilization'] >= UTIL_NEAR_1PCT)
        n_5pct = sum(1 for r in period_recs if r['utilization'] is not None and r['utilization'] >= UTIL_NEAR_5PCT)
        recs_2030 = [r for r in period_recs if r['year'] == FOCUS_YEAR]
        top5 = sorted(period_recs, key=lambda r: r['utilization'] or 0.0, reverse=True)[:5]
        per_node[str(n)] = {
            'rated_s_mva': s_rated,
            'n_periods_total': n_total,
            'max_utilization': max((r['utilization'] for r in period_recs), default=None),
            'mean_consensus_spread_p_mw': (sum(r['consensus_spread_p_mw'] for r in period_recs) / n_total) if n_total else None,
            'max_consensus_spread_p_mw': max((r['consensus_spread_p_mw'] for r in period_recs), default=None),
            'n_within_1pct_of_rating': n_1pct,
            'n_within_5pct_of_rating': n_5pct,
            'n_periods_2030': len(recs_2030),
            'top5_periods_by_utilization': top5,
            'efc_per_day_at_stop': ess_stop['efc_per_day_per_node'].get(str(n)),
        }
    log(f"  node {FOCUS_NODE} max storage utilization (S/rated) at stop = {per_node[str(FOCUS_NODE)]['max_utilization']}")

    # SoH floor sidecar, terminal row
    log(f'reading SoH floor sidecar (stop row): {soh_path}')
    with open(soh_path) as f:
        n_soh_lines = sum(1 for _ in f)
    soh_stop = _read_last_n_jsonl(soh_path, 1)[0]
    soh_by_node = {}
    for n in NODES:
        rows = [r for r in soh_stop['entries'] if r['node_id'] == n]
        soh_by_node[str(n)] = [
            {
                'y_inv': r['y_inv'], 'y': r['y'], 'es_soh_per_unit_cumul': r['es_soh_per_unit_cumul'],
                'soh_min': r['soh_min'], 'active': r['active'],
                'margin_above_floor': r['es_soh_per_unit_cumul'] - r['soh_min'],
                'efc_per_day': r.get('efc_per_day'),
            }
            for r in rows
        ]

    # ESSO capture (cohort-level, terminal) from g_*.json, per node
    esso_capture_by_node = {str(n): g['esso_capture'].get(str(n)) for n in NODES}

    # ESSO models pickle: cross-check rating and SoH cumul; note absence of a per-period SoC var
    log(f'unpickling ESSO models (zero-solve, cross-check only): {esso_pkl_path}')
    with open(esso_pkl_path, 'rb') as f:
        esso_models = pickle.load(f)
    pkl_cross_check = {}
    var_names_seen = None
    for n in NODES:
        model = esso_models.get(n)
        if model is None:
            pkl_cross_check[str(n)] = None
            continue
        names = sorted(v.name for v in model.component_objects(pe.Var, active=None))
        if var_names_seen is None:
            var_names_seen = names
        s_rated_vals = {idx: pe.value(model.es_s_rated[idx], exception=False) for idx in model.es_s_rated}
        e_rated_vals = {idx: pe.value(model.es_e_rated[idx], exception=False) for idx in model.es_e_rated}
        pkl_cross_check[str(n)] = {'es_s_rated_by_cohort': s_rated_vals, 'es_e_rated_by_cohort': e_rated_vals}
    soc_var_present = any('soc' in name.lower() for name in (var_names_seen or []))

    return {
        'input_files': [os.path.relpath(ess_path, REPO), os.path.relpath(soh_path, REPO),
                         os.path.relpath(esso_pkl_path, REPO)],
        'n_ess_stride_lines': n_ess_lines,
        'n_soh_stride_lines': n_soh_lines,
        'stop_cycle': stop_cycle,
        'instance_rating': instance,
        'dispatch_vs_rating_per_node': per_node,
        'soh_floor_terminal_per_node': soh_by_node,
        'esso_capture_terminal_per_node_from_g_json': esso_capture_by_node,
        'esso_models_pkl_cross_check': pkl_cross_check,
        'esso_models_pkl_var_names': var_names_seen,
        'soc_variable_present_in_esso_model': soc_var_present,
        'soc_not_captured_note': (
            'No per-period (intra-day) state-of-charge / energy-state series for node 7\'s shared storage is '
            'available from saved data. Searched: ess_entry_stride_baseline.jsonl (per-period entries carry '
            'only power z/x, keyed by power_type in {p,q} -- no energy-state field), the ESSO capture embedded '
            'in g_<label>.json (cohort-level cumulative SoH / average-charge-discharge / EFC-per-day only, not '
            'per-period), and the unpickled esso_models_*.pkl Var list above (es_s_rated, es_e_rated, es_pnet, '
            'es_qnet, slack_es_pnet_up/down, es_s_rated_per_unit, es_e_rated_per_unit, es_s_available_per_unit, '
            'es_e_available_per_unit, es_pch_per_unit, es_pdch_per_unit, es_avg_ch_dch_per_unit, '
            'es_soh_per_unit_cumul, es_D_per_unit -- no per-period SoC/energy Var). The physical per-period '
            'energy state, if tracked at all, would live in the DSO (case33_2) network model\'s own shared-ESS '
            'block, which is not preserved for the terminal cycle in this run\'s artifacts (only two isolated '
            'FrozenSMOPF snapshots at cycle 7 are kept, under results/FrozenSMOPF/, not at the stop cycle and '
            'not loaded here as out of scope for this stage).'
        ),
    }


# ---------------------------------------------------------------------------
# IPOPT terminal-cycle log check (best-effort, read-only)
# ---------------------------------------------------------------------------

def analyze_ipopt_logs(label, stop_cycle, log):
    label_lower = label.lower()
    candidates = [
        d for d in glob.glob(os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals', f'*{label_lower}*arm', 'logs'))
        if 'preflight' not in d and 'probe' not in d and '__moved_aside' not in d
    ]
    if len(candidates) != 1:
        log(f'  IPOPT log directory: NOT uniquely found for label {label!r} ({len(candidates)} candidates); skipped')
        return {'logs_dir': None, 'candidates_found': candidates, 'per_node': {},
                'note': 'best-effort glob did not resolve to exactly one directory; not read'}
    logs_dir = candidates[0]
    log(f'  IPOPT log directory: {logs_dir}')
    per_node = {}
    for n in NODES:
        fpath = os.path.join(logs_dir, f'optim_log_esso_node{n}_cycle{stop_cycle:03d}.txt')
        if not os.path.exists(fpath):
            per_node[str(n)] = {'file': None, 'found': False}
            continue
        tail = open(fpath).readlines()[-40:]
        exit_line = next((ln.strip() for ln in tail if ln.strip().startswith('EXIT')), None)
        iter_line = next((ln.strip() for ln in tail if 'Number of Iterations' in ln), None)
        per_node[str(n)] = {
            'file': os.path.relpath(fpath, REPO), 'found': True,
            'exit_status': exit_line, 'iterations_line': iter_line,
        }
    return {'logs_dir': os.path.relpath(logs_dir, REPO), 'per_node': per_node}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    dry = '--dry-run' in argv
    pos = [a for a in argv if not a.startswith('--')]
    if len(pos) < 2:
        print(__doc__)
        return 2
    run_dir = os.path.abspath(pos[0])
    label = pos[1]

    out_dir = os.path.join(OUT_ROOT, label)
    out_json = os.path.join(out_dir, f'node7_interface_{label}.json')
    out_log = os.path.join(out_dir, f'run_log_{label}.txt')
    out_manifest = os.path.join(out_dir, 'evidence_manifest_sha256.json')
    for p in (out_json, out_log, out_manifest):
        if not dry and os.path.exists(p):
            raise RuntimeError(f'refusing to overwrite existing output {p}')

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f'P5.15 Addendum 21 item (3): node 7 interface zero-solve look')
    log(f'run_dir = {run_dir}')
    log(f'label   = {label}')
    log(f'timestamp_utc = {datetime.now(timezone.utc).isoformat()}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s39 node7 interface (zero-solve)').install()
    inputs_used = []
    try:
        g_path = _one(os.path.join(run_dir, f'g_{label}.json'))
        inputs_used.append(g_path)
        g = json.load(open(g_path))
        instance = g['instance']
        log(f'instance (from g_{label}.json): {instance}')
        log(f"cycles_run={g['cycles_run']}, converged_at_cycle={g.get('converged_at_cycle')}, "
            f"gross_operational_cost={g.get('gross_operational_cost')}")

        pf_path = _one(os.path.join(run_dir, f'pf_entry_stride_{label}.jsonl'))
        inputs_used.append(pf_path)
        pf_channel, stop_cycle, pf_stop_entries = analyze_pf_channel(pf_path, log)
        if stop_cycle != g['cycles_run']:
            log(f'WARNING: pf_entry_stride stop cycle {stop_cycle} != g_*.json cycles_run {g["cycles_run"]}')

        interface_flow = analyze_interface_flow(pf_stop_entries, log)

        voltage_path = _one(os.path.join(run_dir, 'interface_voltage_terminal.json'))
        inputs_used.append(voltage_path)
        voltage = analyze_voltage(voltage_path, log)

        settlement_path = _one(os.path.join(run_dir, 'interface_settlement_detail_s31c.json'))
        component_levels_path = _one(os.path.join(run_dir, 'component_levels_terminal.json'))
        inputs_used += [settlement_path, component_levels_path]
        flexibility = analyze_flexibility(settlement_path, component_levels_path, log)

        ess_path = _one(os.path.join(run_dir, 'ess_entry_stride_baseline.jsonl'))
        soh_path = _one(os.path.join(run_dir, 'soh_floor_sidecar_baseline.jsonl'))
        esso_pkl_path = _one(os.path.join(run_dir, f'esso_models_{label}.pkl'))
        inputs_used += [ess_path, soh_path, esso_pkl_path]
        storage = analyze_storage(ess_path, soh_path, g, esso_pkl_path, instance, log)
        if storage['stop_cycle'] != g['cycles_run']:
            log(f'WARNING: ess_entry_stride stop cycle {storage["stop_cycle"]} != g_*.json cycles_run {g["cycles_run"]}')

        log('IPOPT terminal-cycle log check (best-effort, read-only, not required for the analysis above):')
        ipopt_check = analyze_ipopt_logs(label, g['cycles_run'], log)

        boyd_path = _one(os.path.join(run_dir, 'boyd_terminal.json'), required=False)
        boyd_terminal_channels = None
        if boyd_path:
            inputs_used.append(boyd_path)
            boyd_terminal_channels = json.load(open(boyd_path)).get('binding_test_per_channel')

        inventory = {
            'files_found_in_run_dir': sorted(os.path.relpath(p, REPO) for p in inputs_used),
            'candidate_files_searched_but_not_used': [
                'results/FrozenSMOPF/*.pkl (isolated cycle-7 snapshots, not at the stop cycle; not loaded)',
                'recourse_jump_sidecar_baseline.jsonl (system-wide recourse deltas, not node/interface specific; not used)',
                'esso_recovery_events_*.jsonl, network_failures_*.jsonl, leak_classification_*.jsonl, '
                'frozen_snapshots_*.jsonl, heartbeat_*.json, stdout_*.log (monitoring/process artifacts, no '
                'interface/flow/voltage/flexibility/storage content; not used)',
                'data/SRP1/SRP1.json (used only to map node 7 -> case33_2, read separately from this script '
                'during preparation; DistributionNetworks[*].connection_node_id: 5->case33_1, 7->case33_2, '
                '9->case33_3)',
                'data/SRP1/case33_2/case33_2_operational_data.xlsx sheet "Flex" (see flexibility bound_not_captured_note)',
            ],
        }

        result = {
            'stage': 'P5.15 Addendum 21 item (3) -- node 7 TSO-DSO interface zero-solve look',
            'authority': [
                'PLANNER_BRIEF_2026-09-13.md Addendum 21',
                'P5_15_ADDENDUM20_EXPERT_REPORT.md section 3.4',
                'P5_15_S38_PF_PACE_REPORT.md section 3.3',
                'data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json',
            ],
            'run_dir': os.path.relpath(run_dir, REPO),
            'label': label,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'nodes': NODES,
            'focus_node': FOCUS_NODE,
            'focus_year': FOCUS_YEAR,
            'instance': instance,
            'stop_cycle': g['cycles_run'],
            'converged_at_cycle': g.get('converged_at_cycle'),
            'gross_operational_cost': g.get('gross_operational_cost'),
            'boyd_terminal_channels': boyd_terminal_channels,
            'pf_channel': pf_channel,
            'interface_flow_vs_rating': interface_flow,
            'voltage_vs_bounds': voltage,
            'flexibility': flexibility,
            'storage': storage,
            'ipopt_terminal_log_check': ipopt_check,
            'inventory': inventory,
        }

        failures = guard.verify(0)
        if failures:
            raise RuntimeError(f'SolveProfileGuard violation: {failures}')
        log('SolveProfileGuard.verify(0): OK -- zero solves for the whole script')
    finally:
        guard.uninstall()

    if dry:
        log('--dry-run: not writing outputs')
        return 0

    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log(f'wrote {out_json}')
    with open(out_log, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {out_log}')

    manifest_files = []
    all_paths = sorted(set(inputs_used)) + [out_json, out_log]
    for p in all_paths:
        manifest_files.append({
            'path': os.path.relpath(p, REPO),
            'bytes': os.path.getsize(p),
            'sha256': _sha256(p),
        })
    manifest = {
        'stage': f'P5.15 Addendum 21 item (3) -- node7 interface zero-solve look, arm {label}',
        'script': SCRIPT_REL,
        'note': ('Inputs are the arm run directory\'s own committed/hash-recorded artifacts, read-only; none '
                 'modified. The large per-cycle stride files (pf_entry_stride_*.jsonl, '
                 'ess_entry_stride_baseline.jsonl) are hashed here, not copied.'),
        'n_files': len(manifest_files),
        'total_bytes': sum(m['bytes'] for m in manifest_files),
        'files': manifest_files,
    }
    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote {out_manifest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
