"""
Stage P5.5-C8 -- tightness diagnostics on the solved centralized relaxation.

The point is not to say how loose the bound is but to say WHERE the looseness
comes from, family by family:

  rho_ij   AC rank gap        U_b * W_jj - (C_b^2 + D_b^2) >= 0, zero iff that
                              branch's W submatrix is rank one.
  rho_tr   OLTC rank gap      the same quantity restricted to tap branches, where
                              U_b is additionally free inside the tap box.
  cycle    angle consistency  sum of atan2(W_ijI, W_ijR) around each independent
                              cycle.  A W solution comes from actual voltage
                              angles only if every such sum is a multiple of
                              2*pi.  Distribution networks here are radial and
                              have no cycles, so this is a transmission-only
                              diagnostic.
  ess      simultaneous       min(pch, pdch) / S: the circulating power the
           circulation        dropped complementarity condition would forbid.
  esso     energy headroom    E_available / E_rated: C5 relaxed the physical
                              E_available = E_rated * soh_cumul to an interval,
                              and this says how much of that interval the
                              relaxation actually exploits.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c8_tightness.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c7_solve import _grb, solve  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')
SOLVE_SETTINGS = {'BarHomogeneous': 1}


def cycle_basis(network, branch_ids):
    """Fundamental cycles as oriented branch lists, via a spanning forest."""
    parent, parent_edge = {}, {}
    adjacency = {}
    for b in branch_ids:
        branch = network.branches[b]
        f = network.get_node_idx(branch.fbus)
        t = network.get_node_idx(branch.tbus)
        adjacency.setdefault(f, []).append((t, b))
        adjacency.setdefault(t, []).append((f, b))

    seen, tree_edges = set(), set()
    order = []
    for start in list(adjacency):
        if start in seen:
            continue
        seen.add(start)
        parent[start] = None
        stack = [start]
        while stack:
            node = stack.pop()
            order.append(node)
            for other, b in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    parent[other] = node
                    parent_edge[other] = b
                    tree_edges.add(b)
                    stack.append(other)

    def path_to_root(node):
        path = []
        while parent.get(node) is not None:
            path.append((parent[node], node, parent_edge[node]))
            node = parent[node]
        return path, node

    cycles = []
    for b in branch_ids:
        if b in tree_edges:
            continue
        branch = network.branches[b]
        f = network.get_node_idx(branch.fbus)
        t = network.get_node_idx(branch.tbus)
        path_f, root_f = path_to_root(f)
        path_t, root_t = path_to_root(t)
        if root_f != root_t:
            continue
        nodes_f = {f: 0}
        for i, (up, down, _e) in enumerate(path_f, start=1):
            nodes_f[up] = i
        meet, depth_t = None, None
        if t in nodes_f:
            meet, depth_t = t, 0
        else:
            for i, (up, down, _e) in enumerate(path_t, start=1):
                if up in nodes_f:
                    meet, depth_t = up, i
                    break
        if meet is None:
            continue
        # oriented walk: f -> meet up the tree, then meet -> t down the tree,
        # then the closing chord t -> f
        walk = []
        for up, down, e in path_f[:nodes_f[meet]]:
            walk.append((e, down, up))          # traversed from `down` to `up`
        for up, down, e in reversed(path_t[:depth_t]):
            walk.append((e, up, down))
        walk.append((b, t, f))
        cycles.append(walk)
    return cycles


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C8 tightness', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C8] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    print('[C8] building and solving the full centralized relaxation ...', flush=True)
    parent = build_centralized_relaxation(planning, candidate)
    size = model_size(parent)
    opt, _ = solve(parent, extra=SOLVE_SETTINGS)
    obj_val, obj_bound = _grb(opt, 'ObjVal'), _grb(opt, 'ObjBound')
    certified = obj_bound is not None and math.isfinite(obj_bound)
    print(f'[C8] ObjVal={obj_val} ObjBound={obj_bound} certified={certified}',
          flush=True)

    report = {
        'stage': 'P5.5-C8', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_size': size,
        'solve': {'settings': SOLVE_SETTINGS, 'gurobi_status': _grb(opt, 'Status'),
                  'ObjVal': obj_val, 'ObjBound': obj_bound, 'certified': certified,
                  'BarIterCount': _grb(opt, 'BarIterCount'),
                  'Runtime': _grb(opt, 'Runtime')},
        'note': ('The primal point diagnosed here comes from a solve that did '
                 'not yield a dual certificate (see P5.5-C7).  The rank and '
                 'angle gaps below are properties of that primal point and are '
                 'reported as such; they are not attached to a proven bound.'),
    }

    rank_rows, tap_rows, cycle_rows, ess_rows = [], [], [], []
    for key, blk in parent.blocks.items():
        tag = '|'.join(str(k) for k in key)
        network = blk.network
        tap_set = set(blk.tap_branches) if hasattr(blk, 'tap_branches') else set()
        block_gap = {'branch': 0.0, 'tap': 0.0}
        for b in blk.branches:
            t_idx = network.get_node_idx(network.branches[b].tbus)
            for p in blk.periods:
                u = float(pe.value(blk.Ub[b, p]))
                w = float(pe.value(blk.vmag_sqr[t_idx, blk.s_m, blk.s_o, p]))
                c = float(pe.value(blk.Cb[b, p]))
                d = float(pe.value(blk.Db[b, p]))
                gap = u * w - (c * c + d * d)
                scale = max(u * w, 1e-12)
                target = 'tap' if b in tap_set else 'branch'
                block_gap[target] = max(block_gap[target], gap / scale)
        rank_rows.append({'block': tag, 'max_relative_rank_gap': block_gap['branch']})
        if tap_set:
            tap_rows.append({'block': tag, 'max_relative_rank_gap': block_gap['tap']})

        # cycle consistency (meshed networks only)
        cycles = cycle_basis(network, list(blk.branches))
        if cycles:
            worst = 0.0
            for cycle in cycles:
                for p in blk.periods:
                    total = 0.0
                    for b, head, tail in cycle:
                        c = float(pe.value(blk.Cb[b, p]))
                        d = float(pe.value(blk.Db[b, p]))
                        f_idx = network.get_node_idx(network.branches[b].fbus)
                        angle = math.atan2(d, c)
                        # atan2(D, C) is the angle of V_from * conj(V_to);
                        # traversing the branch the other way negates it
                        total += angle if head == f_idx else -angle
                    residual = abs((total + math.pi) % (2 * math.pi) - math.pi)
                    worst = max(worst, residual)
            cycle_rows.append({'block': tag, 'n_cycles': len(cycles),
                               'max_cycle_angle_residual_rad': worst})

        # ESS simultaneous circulation
        worst_circ = 0.0
        for e in blk.shared_energy_storages:
            s_av = float(pe.value(blk.S_av[e]))
            if s_av <= 1e-12:
                continue
            for p in blk.periods:
                pch = float(pe.value(blk.shared_es_pch[e, blk.s_m, blk.s_o, p]))
                pdch = float(pe.value(blk.shared_es_pdch[e, blk.s_m, blk.s_o, p]))
                worst_circ = max(worst_circ, min(pch, pdch) / s_av)
        ess_rows.append({'block': tag, 'max_circulation_over_S': worst_circ})

    esso_rows = []
    for (node, year) in parent.E_rated:
        e_rated = float(pe.value(parent.E_rated[node, year]))
        e_avail = float(pe.value(parent.E_available[node, year]))
        s_rated = float(pe.value(parent.S_rated[node, year]))
        s_avail = float(pe.value(parent.S_available[node, year]))
        esso_rows.append({
            'node': node, 'year': year, 'E_rated': e_rated, 'E_available': e_avail,
            'E_available_over_rated': (e_avail / e_rated) if e_rated > 1e-12 else None,
            'S_rated': s_rated, 'S_available': s_avail,
        })

    report['rho_ij_ac_rank_gap'] = {
        'per_block': rank_rows,
        'max_over_model': max((r['max_relative_rank_gap'] for r in rank_rows),
                              default=0.0)}
    report['rho_tr_oltc_rank_gap'] = {
        'per_block': tap_rows,
        'max_over_model': max((r['max_relative_rank_gap'] for r in tap_rows),
                              default=0.0)}
    report['cycle_consistency'] = {
        'per_block': cycle_rows,
        'max_over_model': max((r['max_cycle_angle_residual_rad'] for r in cycle_rows),
                              default=0.0)}
    report['ess_circulation'] = {
        'per_block': ess_rows,
        'max_over_model': max((r['max_circulation_over_S'] for r in ess_rows),
                              default=0.0)}
    report['esso_energy_headroom'] = {
        'per_node_year': esso_rows,
        'min_available_over_rated': min(
            (r['E_available_over_rated'] for r in esso_rows
             if r['E_available_over_rated'] is not None), default=None)}

    out = os.path.join(OUT_DIR, 'p55c_c8_tightness.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"\n[C8] rho_ij  max relative AC rank gap   = "
          f"{report['rho_ij_ac_rank_gap']['max_over_model']:.6e}")
    print(f"[C8] rho_tr  max relative OLTC rank gap = "
          f"{report['rho_tr_oltc_rank_gap']['max_over_model']:.6e}")
    print(f"[C8] cycle   max angle residual (rad)   = "
          f"{report['cycle_consistency']['max_over_model']:.6e} "
          f"over {len(cycle_rows)} meshed blocks")
    print(f"[C8] ess     max min(pch,pdch)/S        = "
          f"{report['ess_circulation']['max_over_model']:.6e}")
    print(f"[C8] esso    min E_available/E_rated    = "
          f"{report['esso_energy_headroom']['min_available_over_rated']}")
    print('\n[C8] worst blocks by AC rank gap')
    for row in sorted(rank_rows, key=lambda r: -r['max_relative_rank_gap'])[:8]:
        print(f"      {row['block']:26s} {row['max_relative_rank_gap']:.6e}")
    print(f'\n[C8] report -> {out}')


if __name__ == '__main__':
    main()
