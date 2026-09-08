"""
Stage P5.5-D8 -- what remains after ESS mode control, attributed by OBJECTIVE.

P5.5-C8 ranked the sources of looseness with rank/circulation statistics.  D8
requires the same question answered with objective values, because a statistic
can be large while the objective it moves is negligible -- which is exactly what
D4 found for the ESS.

The parent objective is a weighted sum over blocks,

    Q(parent) = sum_blocks weight(block) * block_objective(block),

so the gap between the relaxation optimum and the polished nonlinear feasible
point decomposes EXACTLY, block by block:

    gap = sum_blocks weight * [ block_objective(polished) -
                                block_objective(relaxation optimum) ].

That decomposition is what this script computes.  It needs the per-block values
of the polished point, which P5.5-D1 writes out.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55d_d8_attribution.py
"""

import io
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from convex_oracle import block_objective  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c7_solve import _grb, solve  # noqa: E402
from p55d_d4_fixedmode import (DELTA, SETTINGS, USABLE_STATUS, GARBAGE,  # noqa: E402
                               apply_fixed_modes, build_mode_schedule,
                               diagnostics, solve_with_ladder)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55D')
P55C_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

# P5.5-C8 reference values, for the required side-by-side comparison
C8_REFERENCE = {
    'ess_circulation_over_S': 4.989262e-01,
    'ac_rank_gap': 2.449850e-02,
    'cycle_residual_rad': 1.234422e-02,
    'oltc_rank_gap': 4.656229e-06,
    'esso_min_available_over_rated': 0.9975397287609732,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-D8 attribution', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D8] ABORTED\n{error}')
        sys.exit(1)

    d1_path = os.path.join(OUT_DIR, 'p55d_d1_polish.json')
    if not os.path.exists(d1_path):
        print(f'[D8] ABORTED: {d1_path} not found -- run P5.5-D1 first.')
        sys.exit(1)
    with open(d1_path) as handle:
        d1 = json.load(handle)
    polished_blocks = {k: v for k, v in d1['c6_global_exact'].get(
        'per_block', d1.get('per_block', {})).items()}
    if not polished_blocks or 'weighted_block_objective_at_polished_point' not in \
            next(iter(polished_blocks.values())):
        print('[D8] ABORTED: D1 report carries no per-block polished objectives; '
              're-run P5.5-D1 with the per-block dump enabled.')
        sys.exit(1)
    ub_polished = d1['polished_net_operational_recourse']
    polished_gross = d1['polished_gross_operational_cost']

    schedule, counts = build_mode_schedule(
        os.path.join(OUT_DIR, 'p55d_d1_ess_schedule.json'))
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-D8', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'polished_gross_operational_cost': polished_gross,
              'polished_net_operational_recourse': ub_polished,
              'C8_reference': C8_REFERENCE, 'variants': []}

    for label, use_modes in (('continuous', False), ('fixed-mode', True)):
        print(f'\n[D8] ===== {label} full model =====', flush=True)
        parent = build_centralized_relaxation(planning, candidate)
        if use_modes:
            applied, _ = apply_fixed_modes(parent, schedule, DELTA)
            print(f'[D8] {applied} mode rows applied', flush=True)
        runs, best = solve_with_ladder(parent)
        if best is None:
            report['variants'].append({'variant': label, 'runs': runs,
                                       'usable': False})
            print('[D8] no usable primal point', flush=True)
            continue
        with redirect_stdout(io.StringIO()):
            solve(parent, extra=dict(SETTINGS)[best['setting']])

        per_block, by_agent = {}, defaultdict(float)
        total_relaxed = 0.0
        for key, blk in parent.blocks.items():
            tag = '|'.join(str(k) for k in key)
            weight = parent.block_weights[key]
            relaxed = weight * float(pe.value(block_objective(blk)))
            polished = polished_blocks[tag][
                'weighted_block_objective_at_polished_point']
            total_relaxed += relaxed
            per_block[tag] = {
                'weight': weight,
                'weighted_objective_relaxed': relaxed,
                'weighted_objective_polished': polished,
                'gap_contribution': polished - relaxed,
            }
            by_agent[key[0]] += polished - relaxed

        total_gap = sum(v['gap_contribution'] for v in per_block.values())
        variant = {
            'variant': label, 'runs': runs, 'best': best, 'usable': True,
            'model_size': model_size(parent),
            'relaxation_objective': best['ObjVal'],
            'sum_of_weighted_block_objectives': total_relaxed,
            'total_gap_vs_polished': total_gap,
            'gap_by_agent': dict(by_agent),
            'gap_by_agent_share': {k: (v / total_gap if total_gap else None)
                                   for k, v in by_agent.items()},
            'per_block': per_block,
            'diagnostics': diagnostics(parent),
        }
        report['variants'].append(variant)

        print(f"\n[D8] relaxation objective        = {best['ObjVal']:.6f}")
        print(f"[D8] polished gross (feasible UB)= {polished_gross:.6f}")
        print(f"[D8] total gap                   = {total_gap:.6f}")
        print('\n[D8] gap attributed by agent (objective, not statistics)')
        for agent, value in sorted(by_agent.items(), key=lambda kv: -abs(kv[1])):
            share = 100 * value / total_gap if total_gap else float('nan')
            print(f'      {agent:8s} {value:18.4f}   {share:7.3f} %')
        print('\n[D8] worst individual blocks')
        for tag, entry in sorted(per_block.items(),
                                 key=lambda kv: -abs(kv[1]['gap_contribution']))[:8]:
            print(f"      {tag:26s} {entry['gap_contribution']:16.4f}")
        d = variant['diagnostics']
        print('\n[D8] diagnostics against P5.5-C8')
        print(f"      ESS circulation / S : {d['ess_circulation_max']:.6e}  "
              f"(C8 {C8_REFERENCE['ess_circulation_over_S']:.6e})")
        print(f"      ESS circulation MW  : {d['ess_circulation_max_MW']:.6e}  "
              f"(S = {d['ess_S_available_max_MVA']:.6f} MVA)")
        print(f"      AC rank gap         : {d['ac_rank_gap_max']:.6e}  "
              f"(C8 {C8_REFERENCE['ac_rank_gap']:.6e})")
        print(f"      cycle residual rad  : {d['cycle_residual_max_rad']:.6e}  "
              f"(C8 {C8_REFERENCE['cycle_residual_rad']:.6e})")
        print(f"      OLTC rank gap       : {d['oltc_rank_gap_max']:.6e}  "
              f"(C8 {C8_REFERENCE['oltc_rank_gap']:.6e})")
        print(f"      ESSO E_av / E_rated : {d['esso_min_available_over_rated']}")

    # how much of the gap did ESS mode control actually remove?
    usable = [v for v in report['variants'] if v.get('usable')]
    if len(usable) == 2:
        cont, fixed = usable[0], usable[1]
        removed = cont['total_gap_vs_polished'] - fixed['total_gap_vs_polished']
        report['gap_removed_by_ESS_mode_control'] = removed
        report['gap_removed_fraction'] = (
            removed / cont['total_gap_vs_polished']
            if cont['total_gap_vs_polished'] else None)
        print(f"\n[D8] gap removed by ESS mode control = {removed:.4f} "
              f"({100 * removed / cont['total_gap_vs_polished']:.4f} % of the gap)")

    out = os.path.join(OUT_DIR, 'p55d_d8_attribution.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[D8] report -> {out}')


if __name__ == '__main__':
    main()
