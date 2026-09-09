"""
Stage P5.5-D4 -- ZERO-BINARY fixed-mode screen.

This is a rejection screen that needs no branch-and-bound at all.

D3 established F_H1 subset F_charge union F_discharge.  Fixing the mode of every
shared ESS selects ONE branch of that union, so the fixed-mode feasible set is a
SUBSET of the MISOCP feasible set.  For a minimisation,

    Q_MISOCP  <=  Q_fixed_mode.

Any primal-FEASIBLE point of the fixed-mode model with value V gives
Q_fixed_mode <= V, hence

    Q_MISOCP  <=  Q_fixed_mode  <=  V.

So if V is still far below the nonlinear feasible upper bound, the unrestricted
MISOCP optimum is at least as far below and therefore cannot close the gap.  The
argument needs only a feasible point -- no dual certificate, no enumeration --
which is exactly why it is worth running before any binaries are introduced.

The fixed-mode model is NOT itself claimed to be a lower bound on the original
nonlinear problem; fixing the mode restricts the set.  It is a screen.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55d_d4_fixedmode.py
"""

import io
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import (_sess_node_map, build_centralized_relaxation,  # noqa: E402
                            model_size)
from p55c_c7_solve import _grb, solve  # noqa: E402
from p55c_c8_tightness import cycle_basis  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55D')
P55C_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

DELTA = math.sqrt(float(ESS_COMPLEMENTARITY_TOLERANCE))
TINY = 1e-9
TIE_RULE = ('if max(pch, pdch) <= 1e-9 p.u. the period carries no meaningful '
            'dispatch; the mode is then set to charge-dominant deterministically')

CASES = [
    ('TSO only', dict(only_agents=['TSO'])),
    ('TSO + DSO5', dict(only_agents=['TSO', 'DSO5'])),
    ('full four-network oracle', dict()),
]
SETTINGS = [
    ('default', {}),
    ('homogeneous', {'BarHomogeneous': 1}),
    ('homogeneous+focus', {'BarHomogeneous': 1, 'NumericFocus': 3}),
    ('focus+scale', {'NumericFocus': 3, 'ScaleFlag': 2}),
    ('homogeneous+nopresolve', {'BarHomogeneous': 1, 'Presolve': 0}),
]
# Gurobi status codes that can carry a usable primal point
USABLE_STATUS = {2, 13}
# any |objective| beyond this is solver garbage, not a solution
GARBAGE = 1e12
# capacity multiplier for the labelled sensitivity variant (see report)
SENSITIVITY_MULTIPLIER = 100.0

Q_NONLINEAR_COLD = 838496830.813414
Q_NONLINEAR_BEST_RECOVERED = 836586463.43
TOL_CUT = 7.164e5


def build_mode_schedule(path):
    """charge-dominant vs discharge-dominant, per (node, year, day, period)."""
    with open(path) as handle:
        raw = json.load(handle)
    schedule, counts = {}, Counter()
    for key, entry in raw.items():
        node, year, day, period = key.split('|')
        pch, pdch = abs(entry['pch']), abs(entry['pdch'])
        if max(pch, pdch) <= TINY:
            mode = 'charge'
            counts['tie_or_tiny'] += 1
        else:
            mode = 'charge' if pch >= pdch else 'discharge'
        counts[mode] += 1
        schedule[(int(node), int(year), day, int(period))] = mode
    return schedule, dict(counts)


def scale_candidate_capacity(candidate, multiplier):
    """A LABELLED DIAGNOSTIC variant, not the canonical candidate.

    The canonical positive-bootstrap candidate carries ~0.0106 MVA per node-year,
    so "0.499 S" of circulation is ~5 kW in absolute terms.  Scaling the capacity
    tests whether the ESS-mode conclusion is an artefact of that near-zero
    capacity or holds when the storage is large enough to matter.
    """
    scaled = deepcopy(candidate)
    for node in scaled['total_capacity']:
        for year in scaled['total_capacity'][node]:
            for key in ('s', 'e'):
                scaled['total_capacity'][node][year][key] = (
                    abs(scaled['total_capacity'][node][year][key]) * multiplier)
    for node in scaled.get('investment', {}):
        for year in scaled['investment'][node]:
            for key in ('s', 'e'):
                scaled['investment'][node][year][key] = (
                    abs(scaled['investment'][node][year][key]) * multiplier)
    return scaled


def apply_fixed_modes(parent, schedule, delta):
    """Impose the selected branch of the D3 disjunction on every ESS copy.

    Every copy of one physical ESS -- the transmission block's and the
    distribution block's -- receives the SAME mode, because they represent the
    same device and the interface_ess_p coupling already forces their net powers
    to be equal.
    """
    applied, missing = 0, 0
    for key, blk in parent.blocks.items():
        tag, year, day, s_m, s_o = key
        node_id = None if tag == 'TSO' else int(tag[3:])
        sess_nodes = _sess_node_map(blk.network, node_id)
        rows = pe.ConstraintList()
        blk.add_component('fixed_mode_branch', rows)
        for e, esso_node in sess_nodes.items():
            for p in blk.periods:
                mode = schedule.get((esso_node, year, day, p))
                if mode is None:
                    missing += 1
                    continue
                if mode == 'charge':
                    rows.add(blk.shared_es_pdch[e, s_m, s_o, p]
                             <= delta * blk.S_av[e])
                else:
                    rows.add(blk.shared_es_pch[e, s_m, s_o, p]
                             <= delta * blk.S_av[e])
                applied += 1
    return applied, missing


def diagnostics(parent):
    """The P5.5-C8 tightness quantities, recomputed on this primal point."""
    rank_ac, rank_tap, cycle, circ = [], [], [], []
    for key, blk in parent.blocks.items():
        tag = '|'.join(str(k) for k in key)
        network = blk.network
        tap_set = set(blk.tap_branches) if hasattr(blk, 'tap_branches') else set()
        worst_ac, worst_tap = 0.0, 0.0
        for b in blk.branches:
            t_idx = network.get_node_idx(network.branches[b].tbus)
            for p in blk.periods:
                u = float(pe.value(blk.Ub[b, p]))
                w = float(pe.value(blk.vmag_sqr[t_idx, blk.s_m, blk.s_o, p]))
                c = float(pe.value(blk.Cb[b, p]))
                d = float(pe.value(blk.Db[b, p]))
                gap = (u * w - (c * c + d * d)) / max(u * w, 1e-12)
                if b in tap_set:
                    worst_tap = max(worst_tap, gap)
                else:
                    worst_ac = max(worst_ac, gap)
        rank_ac.append({'block': tag, 'max_relative_rank_gap': worst_ac})
        if tap_set:
            rank_tap.append({'block': tag, 'max_relative_rank_gap': worst_tap})

        cycles = cycle_basis(network, list(blk.branches))
        if cycles:
            worst = 0.0
            for cyc in cycles:
                for p in blk.periods:
                    total = 0.0
                    for b, head, tail in cyc:
                        c = float(pe.value(blk.Cb[b, p]))
                        d = float(pe.value(blk.Db[b, p]))
                        f_idx = network.get_node_idx(network.branches[b].fbus)
                        angle = math.atan2(d, c)
                        total += angle if head == f_idx else -angle
                    worst = max(worst, abs((total + math.pi) % (2 * math.pi) - math.pi))
            cycle.append({'block': tag, 'n_cycles': len(cycles),
                          'max_cycle_angle_residual_rad': worst})

        worst_circ, worst_circ_mw, worst_s_mva = 0.0, 0.0, 0.0
        for e in blk.shared_energy_storages:
            s_av = float(pe.value(blk.S_av[e]))
            worst_s_mva = max(worst_s_mva, s_av * network.baseMVA)
            if s_av <= 1e-12:
                continue
            for p in blk.periods:
                pch = float(pe.value(blk.shared_es_pch[e, blk.s_m, blk.s_o, p]))
                pdch = float(pe.value(blk.shared_es_pdch[e, blk.s_m, blk.s_o, p]))
                worst_circ = max(worst_circ, min(pch, pdch) / s_av)
                worst_circ_mw = max(worst_circ_mw,
                                    min(pch, pdch) * network.baseMVA)
        circ.append({'block': tag, 'max_circulation_over_S': worst_circ,
                     'max_circulation_MW': worst_circ_mw,
                     'S_available_MVA': worst_s_mva})

    esso = []
    for (node, year) in parent.E_rated:
        e_rated = float(pe.value(parent.E_rated[node, year]))
        e_avail = float(pe.value(parent.E_available[node, year]))
        esso.append({'node': node, 'year': year,
                     'E_available_over_rated':
                         (e_avail / e_rated) if e_rated > 1e-12 else None})
    return {
        'ac_rank_gap_max': max((r['max_relative_rank_gap'] for r in rank_ac),
                               default=0.0),
        'ac_rank_gap_per_block': rank_ac,
        'oltc_rank_gap_max': max((r['max_relative_rank_gap'] for r in rank_tap),
                                 default=0.0),
        'cycle_residual_max_rad': max(
            (r['max_cycle_angle_residual_rad'] for r in cycle), default=0.0),
        'cycle_per_block': cycle,
        'ess_circulation_max': max((r['max_circulation_over_S'] for r in circ),
                                   default=0.0),
        'ess_circulation_max_MW': max((r['max_circulation_MW'] for r in circ),
                                      default=0.0),
        'ess_S_available_max_MVA': max((r['S_available_MVA'] for r in circ),
                                       default=0.0),
        'ess_circulation_per_block': circ,
        'esso_min_available_over_rated': min(
            (r['E_available_over_rated'] for r in esso
             if r['E_available_over_rated'] is not None), default=None),
    }


def solve_with_ladder(parent):
    """Walk the settings ladder until a USABLE primal point appears.

    A Gurobi status outside {OPTIMAL, SUBOPTIMAL}, or an objective beyond the
    garbage threshold, is a solver failure and is recorded as such -- it is NOT
    a value.  P5.5-D4's rejection argument rests on having a genuinely feasible
    point, so a fabricated number would invalidate the whole screen.
    """
    runs, best = [], None
    for sname, extra in SETTINGS:
        started = time.time()
        with redirect_stdout(io.StringIO()) as log:
            try:
                opt, _ = solve(parent, extra=extra)
                failure = None
            except Exception as exc:
                opt, failure = None, f'{type(exc).__name__}: {exc}'
        runtime = time.time() - started
        text = log.getvalue()
        if opt is None:
            runs.append({'setting': sname, 'error': failure, 'runtime_s': runtime,
                         'usable': False})
            continue
        status = _grb(opt, 'Status')
        value, bound = _grb(opt, 'ObjVal'), _grb(opt, 'ObjBound')
        usable = (status in USABLE_STATUS and value is not None
                  and math.isfinite(value) and abs(value) < GARBAGE)
        runs.append({
            'setting': sname, 'params': extra, 'gurobi_status': status,
            'ObjVal': value, 'ObjBound': bound, 'usable': usable,
            'certified': (bound is not None and math.isfinite(bound)
                          and 'failed to compute QCP dual' not in text),
            'MaxVio': _grb(opt, 'MaxVio'), 'BarIterCount': _grb(opt, 'BarIterCount'),
            'runtime_s': runtime,
            'qcp_dual_warning': 'failed to compute QCP dual' in text,
            'suboptimal': 'Sub-optimal termination' in text,
            'numerical_trouble': 'Numerical trouble' in text})
        print(f"      {sname:24s} status={status} usable={usable} "
              f"ObjVal={value} MaxVio={_grb(opt, 'MaxVio')} t={runtime:.1f}s",
              flush=True)
        if usable and (best is None or value < best['ObjVal']):
            best = runs[-1]
    return runs, best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-D4 fixed-mode screen', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D4] ABORTED\n{error}')
        sys.exit(1)

    schedule_path = os.path.join(OUT_DIR, 'p55d_d1_ess_schedule.json')
    if not os.path.exists(schedule_path):
        print(f'[D4] ABORTED: {schedule_path} not found -- run P5.5-D1 first.')
        sys.exit(1)
    schedule, counts = build_mode_schedule(schedule_path)
    print(f'[D4] mode schedule from the D1 polished nonlinear solution: {counts}')
    print(f'[D4] delta = sqrt(eps) = {DELTA:g}; tie rule: {TIE_RULE}\n', flush=True)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    ub_polished = None
    d1_path = os.path.join(OUT_DIR, 'p55d_d1_polish.json')
    if os.path.exists(d1_path):
        with open(d1_path) as handle:
            ub_polished = json.load(handle).get('rigorous_feasible_UB_incumbent')

    report = {'stage': 'P5.5-D4', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'delta': DELTA, 'tie_rule': TIE_RULE, 'mode_counts': counts,
              'mode_source': 'D1 polished nonlinear feasible solution',
              'UB_polished_exact_consensus': ub_polished,
              'UB_cold': Q_NONLINEAR_COLD,
              'UB_best_recovered': Q_NONLINEAR_BEST_RECOVERED,
              'rejection_logic': ('fixed-mode set is a SUBSET of the MISOCP set, '
                                  'so Q_MISOCP <= Q_fixed_mode <= V for any '
                                  'primal-feasible V; a low V therefore bounds '
                                  'the MISOCP optimum from above'),
              'cases': []}

    variants = [('canonical candidate', candidate, 1.0)]
    if SENSITIVITY_MULTIPLIER and SENSITIVITY_MULTIPLIER != 1.0:
        variants.append((f'capacity x{SENSITIVITY_MULTIPLIER:g} SENSITIVITY',
                         scale_candidate_capacity(candidate,
                                                  SENSITIVITY_MULTIPLIER),
                         SENSITIVITY_MULTIPLIER))

    for vlabel, vcandidate, multiplier in variants:
        print(f'\n[D4] ################ {vlabel} ################', flush=True)
        for label, restriction in CASES:
            entry = {'variant': vlabel, 'capacity_multiplier': multiplier,
                     'case': label,
                     'restriction': {k: list(v) for k, v in restriction.items()}}

            # --- A: continuous baseline, no mode rows --------------------
            print(f'\n[D4] === {label} :: CONTINUOUS baseline', flush=True)
            parent = build_centralized_relaxation(planning, vcandidate,
                                                  **restriction)
            entry['model_size'] = model_size(parent)
            runs, best = solve_with_ladder(parent)
            entry['continuous_runs'] = runs
            if best is not None:
                with redirect_stdout(io.StringIO()):
                    solve(parent, extra=dict(SETTINGS)[best['setting']])
                entry['continuous_best'] = best
                entry['continuous_diagnostics'] = diagnostics(parent)
                print(f"    continuous ObjVal = {best['ObjVal']:.6f} "
                      f"({best['setting']})")
                print(f"    circulation {entry['continuous_diagnostics']['ess_circulation_max']:.4f} S "
                      f"= {entry['continuous_diagnostics']['ess_circulation_max_MW']:.6e} MW "
                      f"(S = {entry['continuous_diagnostics']['ess_S_available_max_MVA']:.6f} MVA)")
            else:
                entry['continuous_best'] = None
                print('    continuous: NO USABLE PRIMAL POINT', flush=True)

            # --- B: same model plus the fixed-mode branch ----------------
            print(f'\n[D4] === {label} :: FIXED-MODE', flush=True)
            parent_fm = build_centralized_relaxation(planning, vcandidate,
                                                     **restriction)
            applied, missing = apply_fixed_modes(parent_fm, schedule, DELTA)
            entry['mode_rows_applied'] = applied
            entry['mode_rows_missing'] = missing
            runs_fm, best_fm = solve_with_ladder(parent_fm)
            entry['fixed_mode_runs'] = runs_fm
            if best_fm is not None:
                with redirect_stdout(io.StringIO()):
                    solve(parent_fm, extra=dict(SETTINGS)[best_fm['setting']])
                entry['fixed_mode_best'] = best_fm
                entry['fixed_mode_diagnostics'] = diagnostics(parent_fm)
                print(f"    fixed-mode ObjVal = {best_fm['ObjVal']:.6f} "
                      f"({best_fm['setting']})")
                print(f"    circulation {entry['fixed_mode_diagnostics']['ess_circulation_max']:.4f} S "
                      f"= {entry['fixed_mode_diagnostics']['ess_circulation_max_MW']:.6e} MW")
            else:
                entry['fixed_mode_best'] = None
                print('    fixed-mode: NO USABLE PRIMAL POINT', flush=True)

            # --- the quantity the whole stage exists to measure ----------
            if best is not None and best_fm is not None:
                delta_obj = best_fm['ObjVal'] - best['ObjVal']
                entry['objective_change_from_mode_control'] = delta_obj
                entry['objective_change_relative'] = (
                    delta_obj / max(abs(best['ObjVal']), 1e-12))
                print(f"\n    >>> mode control moves the objective by "
                      f"{delta_obj:.6f} "
                      f"({100 * entry['objective_change_relative']:.6f} %)")
                if not restriction:
                    for name, ub in (('cold', Q_NONLINEAR_COLD),
                                     ('best_recovered', Q_NONLINEAR_BEST_RECOVERED),
                                     ('polished', ub_polished)):
                        if ub is None or multiplier != 1.0:
                            continue
                        gap_c = ub - best['ObjVal']
                        gap_f = ub - best_fm['ObjVal']
                        entry[f'gap_continuous_vs_{name}'] = gap_c
                        entry[f'gap_fixed_mode_vs_{name}'] = gap_f
                        entry[f'gap_closed_fraction_vs_{name}'] = (
                            (gap_c - gap_f) / gap_c if gap_c else None)
                        print(f"    gap vs {name:15s}: continuous {gap_c:.2f} "
                              f"({100 * gap_c / abs(ub):.4f} %) -> fixed-mode "
                              f"{gap_f:.2f} ({100 * gap_f / abs(ub):.4f} %), "
                              f"closed {100 * (gap_c - gap_f) / gap_c:.4f} %")
            report['cases'].append(entry)

    out = os.path.join(OUT_DIR, 'p55d_d4_fixedmode.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[D4] report -> {out}')


if __name__ == '__main__':
    main()
