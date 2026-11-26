"""P5.15 Addendum 22 item 1(b) -- node 7 mechanism cross-check and RESULT table (S40).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 22 ("Node 7 ... Report it as a result
... Run the period-by-period cross-check of PF-residual entries vs at-rating periods
(zero solves) to record the mechanism."); frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`, item
`1_zero_solve_analyses.b_node7_cross_check`.

This is a NEW script (does not edit `p515_s39_node7_interface.py`, whose committed
outputs under `data/SRP1/Results/P515S39/node7_interface/{s38_A_tau0,s38_B_pfbal}/` are
left untouched). It reads the same class of raw per-arm artifacts that script reads
(`pf_entry_stride_<label>.jsonl`, `ess_entry_stride_baseline.jsonl`, `g_<label>.json`),
recomputing the full period-slot population itself (that script only kept the top-15/
top-5 lists; this script needs the full 288-period-slot population per node to build
contingency tables and the RESULT table).

Arms analyzed: D (the adopted oracle), C, and, for continuity, v9 arms A and B.

--------------------------------------------------------------------------------------
Part 1 -- period-by-period cross-check (observation, not a proven causal mechanism)
--------------------------------------------------------------------------------------
At each arm's stop cycle, for node 7:
  - "at-rating" periods: period-slots (year, day, period) where
    S_flow = sqrt(P^2+Q^2) (DSO-side interface copy) satisfies S_flow/rating >= 0.99
    (set B99) or >= 0.95 (set B95), out of the 288 period-slots (3 years x 4 days x 24
    periods).
  - "top-s^2" PF-residual entries: node-7 PF consensus entries (power_type in {p,q},
    576 = 2 x 288 total) ranked by s^2 (s = x_dso - z_tso_current, the same field the
    production Boyd-residual capture writes). Two sets: top N=25 entries, and top
    ceil(10% x 576)=58 entries. Each entry set is mapped to its period-slot (year, day,
    period), DROPPING the power_type distinction and de-duplicating, to compare against
    the (period-slot-level) at-rating sets.
  - Contingency table (2x2) for each of the 4 combinations {top25, top10pct} x
    {>=99%, >=95%}, against the 288-period-slot population; coincidence rate =
    |top ^ at-rating| / |top|; base rate (expected under independence) =
    |top| * |at-rating| / 288; enrichment = observed / expected.

2030 concentration: for each at-rating set and each top-s^2 set, the fraction of its
members with year == '2030', compared with the 1/3 base rate (96/288 period-slots per
year, exactly one third).

--------------------------------------------------------------------------------------
Part 2 -- RESULT table (Addendum 22: the interface rating is NOT changed; the storage's
congestion relief at node 7 in the rating-binding periods is reported as a RESULT)
--------------------------------------------------------------------------------------
For node 7 (nodes 5 and 9 as comparators): every period-slot at/above 99% of the
interface rating (the "binding periods"), with P, Q, |S|, rating, utilization, and the
shared storage's net dispatch in the SAME period-slot (from `ess_entry_stride`'s
consensus `z`; sign convention: es_pnet = pch_per_unit - pdch_per_unit, i.e. POSITIVE =
net CHARGE, NEGATIVE = net DISCHARGE -- `shared_energy_storage_data.py:723`, read-only,
not modified here), plus a count of how many of those binding periods have the storage
itself at >=99% of ITS OWN rating (|z|/es_rated_mva >= 0.99).

Zero solves: `SolveProfileGuard(permitted=())` is installed before any input file is
touched and verified with `guard.verify(0)` before any output is written.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_s40_node7_crosscheck.py [--dry-run]

Writes (write-once, refuses to overwrite), per arm:
    data/SRP1/Results/P515S40/node7_result/<label>/node7_crosscheck_<label>.json
    data/SRP1/Results/P515S40/node7_result/<label>/run_log_<label>.txt
    data/SRP1/Results/P515S40/node7_result/<label>/evidence_manifest_sha256.json
"""
import glob
import hashlib
import json
import math
import os
import sys
from collections import deque
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

NODES = (5, 7, 9)
FOCUS_NODE = 7
FOCUS_YEAR = '2030'
UTIL_THRESHOLDS = (0.99, 0.95)
TOP_N = 25
TOP_PCT = 0.10

ARMS = [
    {'label': 's38_A_tau0', 'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38_A_TAU0_run'),
     'role': 'continuity (v9 arm A: tau=0, PF balancing off)'},
    {'label': 's38_B_pfbal', 'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38_B_PFBAL_run'),
     'role': 'continuity (v9 arm B: PF balancing live)'},
    {'label': 's39_C', 'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_C_run'),
     'role': 'v10 arm C'},
    {'label': 's39_D', 'dir': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run'),
     'role': 'v10 arm D -- the ADOPTED ORACLE (Addendum 22)'},
]

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'node7_result')
SCRIPT_REL = os.path.relpath(os.path.abspath(__file__), REPO)


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


def _read_last_line_jsonl(path):
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last is not None else None


def _round_half_up(x):
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------
# per-node period-slot interface table (all 288 slots, not just the top-K)
# ---------------------------------------------------------------------------

def build_interface_table(pf_stop_entries):
    """Return {node: {(year,day,period): {'p':.., 'q':.., 'rating':.., 's_flow':.., 'util':..}}}
    and, separately, the raw node-7 entry list (per power_type) with s^2 for the
    top-s^2 ranking."""
    by_node_slot = {n: {} for n in NODES}
    node7_entries = []
    for e in pf_stop_entries:
        n = e['node_id']
        slot = (e['year'], e['day'], e['period'])
        if n in by_node_slot:
            d = by_node_slot[n].setdefault(slot, {'rating': e['interface_rating']})
            d[e['power_type']] = e['x_dso']
        if n == FOCUS_NODE:
            node7_entries.append({
                'year': e['year'], 'day': e['day'], 'period': e['period'],
                'power_type': e['power_type'], 's': e['s'], 's2': e['s'] ** 2,
                'x_dso': e['x_dso'], 'z_tso_current': e['z_tso_current'], 'r': e['r'],
            })
    per_node_util = {}
    for n in NODES:
        recs = {}
        for slot, d in by_node_slot[n].items():
            p = d.get('p', 0.0)
            q = d.get('q', 0.0)
            rating = d['rating']
            s_flow = math.sqrt(p * p + q * q)
            util = s_flow / rating if rating else None
            recs[slot] = {'year': slot[0], 'day': slot[1], 'period': slot[2],
                          'p_mw': p, 'q_mvar': q, 's_flow_mva': s_flow,
                          'rating_mva': rating, 'utilization': util}
        per_node_util[n] = recs
    return per_node_util, node7_entries


# ---------------------------------------------------------------------------
# storage per-node period-slot table
# ---------------------------------------------------------------------------

def build_storage_table(ess_stop_row, s_rated_mva):
    by_key = {}
    for e in ess_stop_row['entries']:
        base_key = (e['node_id'], e['year'], e['day'])
        by_key.setdefault(base_key, {})[e['power_type']] = e
    per_node_slot = {n: {} for n in NODES}
    for (node_id, year, day), v in by_key.items():
        if node_id not in per_node_slot:
            continue
        p_entry = v.get('p')
        if p_entry is None:
            continue
        n_periods = len(p_entry['z'])
        for period in range(n_periods):
            p_z = p_entry['z'][period]
            util = abs(p_z) / s_rated_mva if s_rated_mva else None
            mode = 'charging' if p_z > 1e-9 else ('discharging' if p_z < -1e-9 else 'idle')
            per_node_slot[node_id][(year, day, period)] = {
                'p_z_mw': p_z, 'utilization_own_rating': util, 'mode': mode,
            }
    return per_node_slot


# ---------------------------------------------------------------------------
# Part 1: contingency / coincidence analysis
# ---------------------------------------------------------------------------

def top_s2_sets(node7_entries):
    ranked = sorted(node7_entries, key=lambda e: e['s2'], reverse=True)
    n_total = len(ranked)
    n_10pct = _round_half_up(TOP_PCT * n_total)
    top_n_entries = ranked[:TOP_N]
    top_pct_entries = ranked[:n_10pct]

    def to_slots(entries):
        slots = set()
        for e in entries:
            slots.add((e['year'], e['day'], e['period']))
        return slots

    return {
        'n_total_node7_entries': n_total,
        'top_n': {'k': TOP_N, 'entries': top_n_entries, 'slots': to_slots(top_n_entries)},
        'top_pct': {'k': n_10pct, 'pct_requested': TOP_PCT, 'entries': top_pct_entries,
                    'slots': to_slots(top_pct_entries)},
    }


def at_rating_sets(node7_util_table, threshold):
    return {slot for slot, d in node7_util_table.items()
            if d['utilization'] is not None and d['utilization'] >= threshold}


def contingency(top_slots, at_rating_slots, population_size):
    inter = top_slots & at_rating_slots
    a = len(inter)
    top_only = len(top_slots) - a
    rating_only = len(at_rating_slots) - a
    neither = population_size - a - top_only - rating_only
    n_top = len(top_slots)
    n_rating = len(at_rating_slots)
    expected = (n_top * n_rating / population_size) if population_size else None
    coincidence_rate = (a / n_top) if n_top else None
    base_rate_of_rating_set = (n_rating / population_size) if population_size else None
    enrichment = (a / expected) if expected else None
    return {
        'n_top': n_top, 'n_at_rating': n_rating, 'population_size': population_size,
        'contingency_table': {
            'top_and_at_rating': a, 'top_only': top_only,
            'at_rating_only': rating_only, 'neither': neither,
        },
        'coincidence_rate_top_and_at_rating_over_n_top': coincidence_rate,
        'base_rate_at_rating_over_population': base_rate_of_rating_set,
        'expected_count_under_independence': expected,
        'enrichment_observed_over_expected': enrichment,
    }


def year_concentration(slots, label):
    n_total = len(slots)
    n_2030 = sum(1 for s in slots if s[0] == FOCUS_YEAR)
    frac = (n_2030 / n_total) if n_total else None
    baseline = 1.0 / 3.0
    return {
        'set': label, 'n_total': n_total, 'n_2030': n_2030,
        'frac_2030': frac, 'baseline_one_third': baseline,
        'enrichment_over_baseline': (frac / baseline) if frac is not None else None,
    }


# ---------------------------------------------------------------------------
# Part 2: RESULT table
# ---------------------------------------------------------------------------

def result_table(node_util_table, storage_table, s_rated_mva):
    out = {}
    for n in NODES:
        util_table = node_util_table[n]
        stor_table = storage_table.get(n, {})
        binding = []
        n_storage_own_rating_coincident = {0.99: 0, 0.95: 0}
        n_binding_by_threshold = {th: 0 for th in UTIL_THRESHOLDS}
        for slot, d in sorted(util_table.items()):
            util = d['utilization']
            if util is None:
                continue
            is_binding_99 = util >= 0.99
            is_binding_95 = util >= 0.95
            if not is_binding_95:
                continue
            if is_binding_99:
                n_binding_by_threshold[0.99] += 1
            n_binding_by_threshold[0.95] += 1
            stor = stor_table.get(slot)
            row = {
                'year': slot[0], 'day': slot[1], 'period': slot[2],
                'p_mw': d['p_mw'], 'q_mvar': d['q_mvar'], 's_flow_mva': d['s_flow_mva'],
                'rating_mva': d['rating_mva'], 'utilization': util,
                'at_or_above_99pct': is_binding_99, 'at_or_above_95pct': is_binding_95,
                'storage_p_z_mw': stor['p_z_mw'] if stor else None,
                'storage_mode': stor['mode'] if stor else None,
                'storage_utilization_own_rating': stor['utilization_own_rating'] if stor else None,
                'storage_at_or_above_99pct_own_rating': (
                    stor['utilization_own_rating'] >= 0.99 if stor and stor['utilization_own_rating'] is not None else False
                ),
            }
            binding.append(row)
            if is_binding_99 and row['storage_at_or_above_99pct_own_rating']:
                n_storage_own_rating_coincident[0.99] += 1
            if row['storage_at_or_above_99pct_own_rating']:
                n_storage_own_rating_coincident[0.95] += 1
        out[str(n)] = {
            'rating_mva': next(iter(util_table.values()))['rating_mva'] if util_table else None,
            'storage_rated_mva': s_rated_mva,
            'n_periods_total': len(util_table),
            'n_binding_ge_99pct': n_binding_by_threshold[0.99],
            'n_binding_ge_95pct': n_binding_by_threshold[0.95],
            'n_binding_ge_95pct_where_storage_ge_99pct_own_rating': n_storage_own_rating_coincident[0.95],
            'n_binding_ge_99pct_where_storage_ge_99pct_own_rating': n_storage_own_rating_coincident[0.99],
            'binding_periods_ge_95pct_detail': binding,
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def analyze_arm(arm, log):
    label = arm['label']
    run_dir = arm['dir']
    log(f"=== arm {label} ({arm['role']}), run_dir={os.path.relpath(run_dir, REPO)} ===")

    inputs = []
    g_path = _one(os.path.join(run_dir, f'g_{label}.json'))
    inputs.append(g_path)
    g = json.load(open(g_path))
    instance = g['instance']
    s_rated_mva = instance['s_mva']
    log(f"instance: {instance}; cycles_run={g['cycles_run']}, "
        f"converged_at_cycle={g.get('converged_at_cycle')}, "
        f"gross_operational_cost={g.get('gross_operational_cost')}")

    pf_path = _one(os.path.join(run_dir, f'pf_entry_stride_{label}.jsonl'))
    inputs.append(pf_path)
    stop_row = _read_last_line_jsonl(pf_path)
    stop_cycle = stop_row['cycle']
    log(f"pf_entry_stride stop cycle = {stop_cycle} "
        f"({'matches' if stop_cycle == g['cycles_run'] else 'DOES NOT MATCH'} g.cycles_run)")
    node_util_table, node7_entries = build_interface_table(stop_row['entries'])
    log(f"node-7 entries collected: {len(node7_entries)} (expect 576 = 2 power_types x 3 years x 4 days x 24 periods)")

    ess_path = _one(os.path.join(run_dir, 'ess_entry_stride_baseline.jsonl'))
    inputs.append(ess_path)
    ess_stop_row = _read_last_line_jsonl(ess_path)
    if ess_stop_row['cycle'] != stop_cycle:
        log(f"WARNING: ess_entry_stride stop cycle {ess_stop_row['cycle']} != pf stop cycle {stop_cycle}")
    storage_table = build_storage_table(ess_stop_row, s_rated_mva)

    # --- Part 1: cross-check ---
    top_sets = top_s2_sets(node7_entries)
    at_rating = {th: at_rating_sets(node_util_table[FOCUS_NODE], th) for th in UTIL_THRESHOLDS}
    for th in UTIL_THRESHOLDS:
        log(f"node 7 at-rating (>= {th}) period-slots: {len(at_rating[th])} / 288")

    cross_check = {}
    for top_key, top_info in (('top_n', top_sets['top_n']), ('top_pct', top_sets['top_pct'])):
        for th in UTIL_THRESHOLDS:
            key = f'{top_key}_vs_ge{int(th*100)}pct'
            cont = contingency(top_info['slots'], at_rating[th], population_size=288)
            cross_check[key] = cont
            log(f"  {key}: n_top={cont['n_top']} n_at_rating={cont['n_at_rating']} "
                f"coincidence={cont['contingency_table']['top_and_at_rating']} "
                f"({cont['coincidence_rate_top_and_at_rating_over_n_top']:.3f}) "
                f"expected_under_independence={cont['expected_count_under_independence']:.3f} "
                f"enrichment={cont['enrichment_observed_over_expected']}")

    # 2030 concentration
    year_2030 = {
        'at_rating_ge99pct': year_concentration(at_rating[0.99], 'node7_at_rating_ge99pct'),
        'at_rating_ge95pct': year_concentration(at_rating[0.95], 'node7_at_rating_ge95pct'),
        'top_n_s2': year_concentration(top_sets['top_n']['slots'], f'node7_top{TOP_N}_s2'),
        'top_pct_s2': year_concentration(top_sets['top_pct']['slots'],
                                          f"node7_top{top_sets['top_pct']['k']}_s2_(10pct)"),
    }
    for k, v in year_2030.items():
        log(f"  2030 concentration [{k}]: {v['n_2030']}/{v['n_total']} = "
            f"{v['frac_2030']:.3f} (baseline 0.333, enrichment {v['enrichment_over_baseline']})")

    # --- Part 2: RESULT table ---
    result = result_table(node_util_table, storage_table, s_rated_mva)
    for n in NODES:
        r = result[str(n)]
        log(f"  RESULT node {n}: rating={r['rating_mva']} MVA, n_binding(>=99%)={r['n_binding_ge_99pct']}, "
            f"n_binding(>=95%)={r['n_binding_ge_95pct']}, "
            f"of which storage>=99%-own-rating (>=99% interface)={r['n_binding_ge_99pct_where_storage_ge_99pct_own_rating']}, "
            f"(>=95% interface)={r['n_binding_ge_95pct_where_storage_ge_99pct_own_rating']}")

    out = {
        'arm': label, 'role': arm['role'], 'run_dir': os.path.relpath(run_dir, REPO),
        'instance': instance, 'stop_cycle': stop_cycle,
        'gross_operational_cost': g.get('gross_operational_cost'),
        'methodology_note': (
            "top-s^2 sets are ranked over the 576 node-7 PF entries (power_type in "
            "{p,q} x 288 period-slots) at the arm's stop cycle, then mapped to their "
            "288-period-slot population (de-duplicated across power_type) for the "
            "contingency test against the (period-slot-level) at-rating sets. "
            "TOP_N=25 fixed count; TOP_PCT=10%% rounded to the nearest integer "
            f"({top_sets['top_pct']['k']} of {top_sets['n_total_node7_entries']})."
        ),
        'top_s2_sets_node7': {
            'n_total_node7_entries': top_sets['n_total_node7_entries'],
            'top_n': {'k': top_sets['top_n']['k'], 'n_distinct_period_slots': len(top_sets['top_n']['slots']),
                      'entries': top_sets['top_n']['entries']},
            'top_pct': {'k': top_sets['top_pct']['k'], 'pct_requested': TOP_PCT,
                        'n_distinct_period_slots': len(top_sets['top_pct']['slots']),
                        'entries': top_sets['top_pct']['entries']},
        },
        'at_rating_sets_node7_period_slot_counts': {str(th): len(at_rating[th]) for th in UTIL_THRESHOLDS},
        'cross_check_contingency': cross_check,
        'year_2030_concentration': year_2030,
        'result_table_by_node': result,
    }
    return out, inputs


def main(argv):
    dry = '--dry-run' in argv
    lines_all = []

    def log_top(msg):
        print(msg)
        lines_all.append(msg)

    log_top('P5.15 Addendum 22 item 1(b): node 7 cross-check + RESULT table (S40)')
    log_top(f'timestamp_utc = {datetime.now(timezone.utc).isoformat()}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s40 node7 crosscheck (zero-solve)').install()
    per_arm_outputs = {}
    try:
        for arm in ARMS:
            lines = []

            def log(msg, _lines=lines):
                print(msg)
                _lines.append(msg)

            out, inputs = analyze_arm(arm, log)
            per_arm_outputs[arm['label']] = {'result': out, 'inputs': inputs, 'lines': lines}

        failures = guard.verify(0)
        if failures:
            raise RuntimeError(f'SolveProfileGuard violation: {failures}')
        log_top('SolveProfileGuard.verify(0): OK -- zero solves for the whole script (all arms)')
    finally:
        guard.uninstall()

    if dry:
        log_top('--dry-run: not writing outputs')
        return 0

    for label, payload in per_arm_outputs.items():
        out_dir = os.path.join(OUT_ROOT, label)
        out_json = os.path.join(out_dir, f'node7_crosscheck_{label}.json')
        out_log = os.path.join(out_dir, f'run_log_{label}.txt')
        out_manifest = os.path.join(out_dir, 'evidence_manifest_sha256.json')
        for p in (out_json, out_log, out_manifest):
            if os.path.exists(p):
                raise RuntimeError(f'refusing to overwrite existing output {p}')
        os.makedirs(out_dir, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(payload['result'], f, indent=2, default=str)
        with open(out_log, 'w') as f:
            f.write('\n'.join(payload['lines']) + '\n')
        manifest_files = []
        all_paths = sorted(set(payload['inputs'])) + [out_json, out_log]
        for p in all_paths:
            manifest_files.append({'path': os.path.relpath(p, REPO), 'bytes': os.path.getsize(p),
                                    'sha256': _sha256(p)})
        manifest = {
            'stage': f'P5.15 Addendum 22 item 1(b) -- node7 crosscheck, arm {label}',
            'script': SCRIPT_REL,
            'note': ("Inputs are the arm run directory's own committed/hash-recorded "
                     "artifacts, read-only; none modified."),
            'n_files': len(manifest_files),
            'total_bytes': sum(m['bytes'] for m in manifest_files),
            'files': manifest_files,
        }
        with open(out_manifest, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'wrote {out_json}, {out_log}, {out_manifest}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
