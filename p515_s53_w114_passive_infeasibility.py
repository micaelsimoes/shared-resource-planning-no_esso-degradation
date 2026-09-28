"""P5.15 W114 -- which constraint makes the PASSIVE-arm TSO block 2035 Summer infeasible. DIAGNOSTIC ONLY.

Context (W113 2ee7a124; W114 Part A): under benchmark spec v2 bb659122 the passive arm's TSO block 2035 Summer is
"Converged to a point of local infeasibility" on all three production attempts at every start (cold,
warm_from_certified, perturbed); at the COORDINATED schedule the same block solves (W113 tso-coupling-check). This
script rebuilds that block at the passive schedule with the benchmark's own builders and solves elastic
(feasibility-diagnostic) COPIES of it. Nothing here feeds any arm value; no arm definition, spec, tolerance or
production file changes.

INSTANCE: SRP1, x = 0 (candidate_key 8435c71859ddde68...), block TSO 2035 Summer; the passive schedule is the
passive-arm cold DSO solve (decision tie-breaker 1 EUR/MWh); the coordinated schedule is the settled cycle-181 x = 0
cell d110bd1a5977df1e_x0 (certified_models.pkl sha256 99ab1070..., verified before unpickling).

REAL BUILDERS, NO REIMPLEMENTATION: planning = p56a_oracle.fresh_planning; solver options =
p515_s53_w93_uncoordinated_benchmark.apply_arm_solver_options (compl_inf_tol 1e-6, as every arm stage); candidate =
its _x0_candidate; DSO blocks = uncoordinated_benchmark.build_dso_arm_models(arm='passive', tie-breaker 1.0) +
check_arm_structures + apply_start('cold'); DSO solves = uncoordinated_benchmark._solve_block (production run_smopf
with its retry tiers), ONLY the three 2035 Summer blocks (the other 33 DSO blocks are built, never solved -- each block
is an independent NLP); TSO = build_tso_arm_model(coupling fixed_interface, pin False, tie-breaker 0) +
check_arm_structures + apply_start('cold'). The TSO build needs targets for every (year, day); the 2035 Summer targets
are the passive schedule and every OTHER (year, day) carries the coordinated DSO schedule as a placeholder (those TSO
blocks are never solved; build_tso_arm_model reads interface_targets[node][year][day] per block only).

STEPS AND SOLVE COUNT (declared in advance, SolveProfileGuard, checked EXACTLY at every boundary):
  1  3 DSO solves  -- passive, cold, 2035 Summer, nodes 5/7/9 (each expected to succeed at the primary attempt, as in
                      W113); reproduction checked against W113's per_solve_record (objective bitwise, iterations).
  2  3 TSO solves  -- the arm block itself through _solve_block: expected ArmSolveFailure after primary + tier-1 +
                      tier-2 (as W113); reproduction checked against W113's failure.json (constraint violation bitwise,
                      iterations).
  3  4 elastic solves, one attempt each (network._run_smopf_solver_attempt, the production solver factory and options,
                      obj_scaling_factor overridden to 1.0 for the slack objective -- declared), each on a clone() of
                      the arm block with pyomo's own `core.add_slack_variables` transformation (non-negative slacks on
                      the target rows; the original objective deactivated; objective = unweighted sum of slacks,
                      i.e. weight 1 per model unit: p.u. power on 100 MVA, p.u.^2 on squared-magnitude rows):
       E1 physical families elastic, interface schedule HARD: bus P balance, bus Q balance, voltage-magnitude rows and
          the vmag / vmag_sqr / e / f bounds, branch thermal rows, generator P bounds, generator Q bounds, the RES
          apparent-power and power-factor rows, the voltage-product sign row. Variable bounds are made elastic by
          moving them into explicit rows (block `w114_bound_rows`) and removing the Var bound (domain kept). Every
          active row and every finite free-Var bound of the block is CLASSIFIED (elastic or declared hard); an
          unclassified family raises before the solve.
       E2 interface P and Q elastic (uncoord_interface_p_fixed / _q_fixed rows), every physical row HARD: the nearest
          (L1, p.u.) interface schedule the TN can accept.
       E3 interface P only elastic (Q hard).   E4 interface Q only elastic (P hard).
  TOTAL 10 solves = 10 process launches. Guard permits only uncoordinated_benchmark.py:_solve_block and this file's
  elastic_solve.

The model's own voltage slacks slack_v_sqr_* (free, bounded, at non-ADN buses; fixed 0 at 5/7/9) stay as in the arm:
free within their bounds at zero cost in the elastic objective, so a zero elastic optimum = the arm's own feasible set;
their values are reported.

OUTPUT (write-once, new directory): data/SRP1/Results/P515S53/w114_passive_infeasibility/
    w114_passive_infeasibility.json, per_solve_record.jsonl, manifest_sha256.json (--manifest, after the run)
IPOPT logs: data/SRP1/Results/P56A/evals/p515s53w114_passive_infeasibility/logs (gitignored; hash-recorded by
--manifest).

LAUNCH (attached, alone, both streams captured), then the manifest:
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w114_passive_infeasibility.py > data/SRP1/Results/P515S53/w114_passive_infeasibility_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w114_passive_infeasibility.py --manifest
Zero-solve build check (blocking guard; writes only to the given scratch directory):
    ... p515_s53_w114_passive_infeasibility.py --dry-run <scratch dir>

Objective convention: no Q is computed here (Q = gross_operational_cost, settlement excluded, elsewhere); the
`active_objective_value` fields are the blocks' own decision objectives, the elastic objectives are slack sums.
"""
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

DRY_RUN = '--dry-run' in sys.argv
MANIFEST = '--manifest' in sys.argv

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard before any production import

THIS_FILE = os.path.basename(os.path.abspath(__file__))
PERMITTED = (('uncoordinated_benchmark.py', '_solve_block'), (THIS_FILE, 'elastic_solve'))
DECLARED_SOLVES = {'dso_passive_cold_2035_summer': 3, 'tso_arm_block_reproduction': 3, 'elastic': 4}
DECLARED_TOTAL = sum(DECLARED_SOLVES.values())       # 10 solves = 10 launches
_GUARD = None
if not MANIFEST:
    _GUARD = SolveProfileGuard(() if DRY_RUN else PERMITTED,
                               label='P5.15 W114 ' + ('dry-run zero-solve' if DRY_RUN else 'passive infeasibility')
                               ).install()

import gate_result_io as GRIO  # noqa: E402

OUT_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w114_passive_infeasibility')
LAUNCH_LOG_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w114_passive_infeasibility_launch.log')
EVAL_ID = 'p515s53w114_passive_infeasibility'
YEAR, DAY = 2035, 'Summer'
W113_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w106_uncoordinated_settled', 'arm_passive_cold')
ELASTIC_OPTION_OVERRIDES = {'obj_scaling_factor': 1.0}
SLACK_REPORT_THRESHOLD = 1e-6          # model units (p.u. / p.u.^2); slacks at or below are reported as zero

# E1 classification. Row families (Constraint components) and bound families (Var components) made elastic:
E1_ROW_FAMILIES = {
    'bus_P_balance': ('node_balance_p',),
    'bus_Q_balance': ('node_balance_q',),
    'voltage_magnitude_rows': ('voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons'),
    'branch_thermal': ('branch_flow_limit', 'branch_flow_limit_ji'),
    'generator_S_capability_and_pf': ('sg_capability', 'gen_pf_upper', 'gen_pf_lower', 'gen_pf_profile'),
    'voltage_product_sign': ('voltage_product_real_nonnegative',),
}
E1_BOUND_FAMILIES = {
    'generator_P_limits': ('pg',),
    'generator_Q_limits': ('qg',),
    'voltage_magnitude_bounds': ('vmag', 'vmag_sqr', 'e', 'f'),
}
# Declared HARD in E1 (definitions and the schedule under test); any other active row family raises.
E1_HARD_ROWS = ('voltage_mag_def', 'voltage_mag_sqr_def', 'voltage_product_real_def', 'voltage_product_imag_def',
                'expected_interface_vmag_def', 'expected_interface_pf_p_def', 'expected_interface_pf_q_def',
                'expected_shared_ess_p_def', 'expected_shared_ess_q_def',
                'uncoord_interface_p_fixed', 'uncoord_interface_q_fixed')
E1_HARD_BOUNDS = {
    'r': 'transformer ratio, +/- 1e-5 around 1 (device data)',
    'r_sqr': 'domain lower bound 0 only',
    'slack_v_sqr_down': "the arm's own soft voltage band (free, bounded, zero cost here; values reported)",
    'slack_v_sqr_up': "the arm's own soft voltage band (free, bounded, zero cost here; values reported)",
    'interface_delta_p': 'interface rating box; delta = 0 is forced by the hard fixed-interface rows in E1',
    'interface_delta_q': 'interface rating box; delta = 0 is forced by the hard fixed-interface rows in E1',
    'expected_interface_vmag': 'domain lower bound 0 only',
}
ELASTIC_VARIANTS = (
    ('E1_physical_elastic_schedule_hard', 'physical', None),
    ('E2_interface_PQ_elastic_physics_hard', 'rows', ('uncoord_interface_p_fixed', 'uncoord_interface_q_fixed')),
    ('E3_interface_P_elastic_Q_hard', 'rows', ('uncoord_interface_p_fixed',)),
    ('E4_interface_Q_elastic_P_hard', 'rows', ('uncoord_interface_q_fixed',)),
)
_LOG_T0 = time.time()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W114 +{time.time() - _LOG_T0:7.1f}s] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _check_guard(expected, where):
    failures = _GUARD.verify(expected)
    if failures:
        raise RuntimeError(f'SolveProfileGuard at {where}: expected exactly {expected}: {failures}; {_GUARD.counts}')
    return {'where': where, 'expected': expected, 'counts': dict(_GUARD.counts), 'verified': True}


def _key(keys, want):
    hits = [k for k in keys if str(k) == str(want)]
    if len(hits) != 1:
        raise RuntimeError(f'cannot resolve key {want!r} in {list(keys)}')
    return hits[0]


def _val(x):
    import pyomo.environ as pe
    v = pe.value(x, exception=False)
    return None if v is None else float(v)


# ======================================================================================================================
#  zero-solve readers
# ======================================================================================================================
def tn_state(block, network):
    """Per hour: generator P/Q vs bounds, bus voltage magnitude, branch apparent flow vs rating, interface P/Q/V at the
    ADN buses, and the model's own voltage-slack values. Pure reads."""
    base = network.baseMVA
    adn = list(network.active_distribution_network_nodes)
    sm, so = next(iter(block.scenarios_market)), next(iter(block.scenarios_operation))
    hours = []
    for p in block.periods:
        gens = []
        for g in block.generators:
            gen = network.generators[g]
            pg = block.pg[g, sm, so, p]
            qg = block.qg[g, sm, so, p]
            gens.append({'gen_idx': g, 'bus': gen.bus, 'gen_type': getattr(gen, 'gen_type', None),
                         'curtaillable': bool(gen.is_curtaillable()),
                         'pg_mw': None if pg.value is None else pg.value * base,
                         'pg_ub_mw': None if pg.ub is None else pg.ub * base,
                         'qg_mvar': None if qg.value is None else qg.value * base,
                         'qg_lb_mvar': None if qg.lb is None else qg.lb * base,
                         'qg_ub_mvar': None if qg.ub is None else qg.ub * base})
        buses = []
        for i in block.nodes:
            node = network.nodes[i]
            vs = _val(block.vmag_sqr[i, sm, so, p])
            row = {'bus': node.bus_i, 'vmag_pu': None if vs is None or vs < 0 else math.sqrt(vs),
                   'v_min': node.v_min, 'v_max': node.v_max}
            for name in ('slack_v_sqr_down', 'slack_v_sqr_up'):
                comp = getattr(block, name, None)
                if comp is not None and (i, sm, so, p) in comp:
                    row[name] = comp[i, sm, so, p].value
            buses.append(row)
        branches = []
        for b in block.branches:
            br = network.branches[b]
            fs = _val(block.flow_ij_sqr[b, sm, so, p])
            flow = None if fs is None or fs < 0 else math.sqrt(fs) * base
            branches.append({'branch_idx': b, 'fbus': br.fbus, 'tbus': br.tbus, 'rate_mva': br.rate,
                             'flow_ij_mva': flow,
                             'loading_pct': None if flow is None or not br.rate else 100.0 * flow / br.rate})
        interface = []
        for dn, node_id in enumerate(adn):
            interface.append({'node': node_id,
                              'p_mw': None if block.expected_interface_pf_p[dn, p].value is None
                              else block.expected_interface_pf_p[dn, p].value * base,
                              'q_mvar': None if block.expected_interface_pf_q[dn, p].value is None
                              else block.expected_interface_pf_q[dn, p].value * base,
                              'v_pu_tn': block.expected_interface_vmag[dn, p].value,
                              'delta_p_mw': None if block.interface_delta_p[dn, sm, so, p].value is None
                              else block.interface_delta_p[dn, sm, so, p].value * base,
                              'delta_q_mvar': None if block.interface_delta_q[dn, sm, so, p].value is None
                              else block.interface_delta_q[dn, sm, so, p].value * base})
        pg_sum = sum(x['pg_mw'] for x in gens if x['pg_mw'] is not None)
        hours.append({'hour': p + 1, 'period': p, 'generators': gens, 'buses': buses, 'branches': branches,
                      'interface': interface, 'pg_total_mw': pg_sum,
                      'pg_capacity_total_mw': sum(x['pg_ub_mw'] for x in gens if x['pg_ub_mw'] is not None),
                      'qg_capacity_total_mvar': [sum(x['qg_lb_mvar'] for x in gens if x['qg_lb_mvar'] is not None),
                                                 sum(x['qg_ub_mvar'] for x in gens if x['qg_ub_mvar'] is not None)],
                      'max_branch_loading_pct': max((x['loading_pct'] for x in branches
                                                     if x['loading_pct'] is not None), default=None)})
    return hours


def schedule_table(passive, coord_dso, coord_tso, y, d):
    """Per node and hour: the passive DSO schedule beside the settled coordinated one (DSO side and TSO side)."""
    out = {}
    for node in sorted(passive):
        a = passive[node][y][d]
        c = coord_dso[node][y][d]
        t = coord_tso[node][y][d]
        rows = []
        for p in range(len(a['p_mw'])):
            rows.append({'hour': p + 1,
                         'passive_p_mw': a['p_mw'][p], 'passive_q_mvar': a['q_mvar'][p],
                         'passive_v_pu_dn': a['v_pu_dn'][p], 'passive_v_kv': a['v_kv'][p],
                         'coord_dso_p_mw': c['p_mw'][p], 'coord_dso_q_mvar': c['q_mvar'][p],
                         'coord_dso_v_pu_dn': c['v_pu_dn'][p], 'coord_dso_v_kv': c['v_kv'][p],
                         'coord_tso_p_mw': t['p_mw'][p], 'coord_tso_q_mvar': t['q_mvar'][p],
                         'coord_tso_v_pu_tn': t['v_pu_tn'][p], 'coord_tso_v_kv': t['v_kv'][p],
                         'passive_minus_coord_dso_p_mw': a['p_mw'][p] - c['p_mw'][p],
                         'passive_minus_coord_dso_q_mvar': a['q_mvar'][p] - c['q_mvar'][p]})
        out[str(node)] = rows
    totals = []
    for p in range(len(next(iter(passive.values()))[y][d]['p_mw'])):
        totals.append({'hour': p + 1,
                       'passive_p_total_mw': sum(passive[n][y][d]['p_mw'][p] for n in passive),
                       'passive_q_total_mvar': sum(passive[n][y][d]['q_mvar'][p] for n in passive),
                       'coord_dso_p_total_mw': sum(coord_dso[n][y][d]['p_mw'][p] for n in coord_dso),
                       'coord_dso_q_total_mvar': sum(coord_dso[n][y][d]['q_mvar'][p] for n in coord_dso)})
    return {'per_node': out, 'totals': totals}


# ======================================================================================================================
#  the elastic copies (diagnostic only)
# ======================================================================================================================
def _classify_e1(block):
    """Every active row family and every finite free-Var bound family of the block must be elastic or declared hard."""
    import pyomo.environ as pe
    elastic_rows = {c for fam in E1_ROW_FAMILIES.values() for c in fam}
    elastic_bounds = {v for fam in E1_BOUND_FAMILIES.values() for v in fam}
    unclassified_rows, row_counts = [], {}
    for comp in block.component_objects(pe.Constraint, active=True, descend_into=True):
        n_active = sum(1 for d in comp.values() if d.active)
        if not n_active:
            continue
        row_counts[comp.name] = n_active
        if comp.name not in elastic_rows and comp.name not in E1_HARD_ROWS:
            unclassified_rows.append(comp.name)
    unclassified_bounds, bound_counts = [], {}
    for comp in block.component_objects(pe.Var, descend_into=True):
        n = sum(1 for d in comp.values() if not d.fixed and (d.lb is not None or d.ub is not None))
        if not n:
            continue
        bound_counts[comp.name] = n
        if comp.name not in elastic_bounds and comp.name not in E1_HARD_BOUNDS:
            unclassified_bounds.append(comp.name)
    if unclassified_rows or unclassified_bounds:
        raise RuntimeError(f'E1 classification incomplete: rows {unclassified_rows}, bounds {unclassified_bounds}')
    return {'active_row_counts': row_counts, 'free_bounded_var_counts': bound_counts,
            'elastic_row_families': E1_ROW_FAMILIES, 'elastic_bound_families': E1_BOUND_FAMILIES,
            'hard_rows': list(E1_HARD_ROWS), 'hard_bounds': E1_HARD_BOUNDS, 'complete': True}


def build_elastic_copy(block, kind, rows):
    """clone() of the arm block + pyomo core.add_slack_variables on the target rows. For E1 ('physical') the elastic
    bound families are first moved into explicit rows (block w114_bound_rows) and their Var bounds removed. Returns
    (copy, targets, bound_map, classification)."""
    import pyomo.environ as pe
    c = block.clone()
    classification = None
    bound_map = {}
    if kind == 'physical':
        classification = _classify_e1(c)
        c.w114_bound_rows = pe.Block()
        targets = [getattr(c, name) for fam in E1_ROW_FAMILIES.values() for name in fam
                   if hasattr(c, name) and any(d.active for d in getattr(c, name).values())]
        for fam in E1_BOUND_FAMILIES.values():
            for name in fam:
                var = getattr(c, name)
                for side in ('lower', 'upper'):
                    entries = []
                    for idx in var:
                        vd = var[idx]
                        if vd.fixed:
                            continue
                        bnd = vd.lb if side == 'lower' else vd.ub
                        if bnd is not None:
                            entries.append((idx, float(bnd)))
                    if not entries:
                        continue
                    cl = pe.ConstraintList()
                    c.w114_bound_rows.add_component(f'{name}_{side}', cl)
                    for idx, bnd in entries:
                        vd = var[idx]
                        row = cl.add(vd >= bnd if side == 'lower' else vd <= bnd)
                        bound_map[row.name] = {'var': vd.name, 'var_family': name, 'index': idx, 'side': side,
                                               'bound': bnd}
                    targets.append(cl)
                for idx in var:
                    if not var[idx].fixed:
                        var[idx].setlb(None)
                        var[idx].setub(None)
    else:
        targets = [getattr(c, name) for name in rows]
    pe.TransformationFactory('core.add_slack_variables').apply_to(c, targets=targets)
    return c, targets, bound_map, classification


def elastic_solve(planning, network_data, network, model, label):
    """ONE IPOPT attempt through production's solver factory (network._run_smopf_solver_attempt), the arm's options with
    ELASTIC_OPTION_OVERRIDES; the solution is loaded when it can be. The permitted call site for the elastic copies."""
    import network as NW
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    started = time.time()
    result, log_path = NW._run_smopf_solver_attempt(network, model, network_data.params, from_warm_start=False,
                                                    option_overrides=dict(ELASTIC_OPTION_OVERRIDES),
                                                    log_suffix=f'w114_{label}')
    wall = time.time() - started
    attempts = srp._drain_network_ipopt_solve_records(planning, f'w114:{label}')
    for a in attempts:
        a['final_summary'] = UB.read_ipopt_attempt_summary(a.get('log_path'), a.get('log_bytes'))
    loaded, load_error = False, None
    if result is not None:
        try:
            model.solutions.load_from(result)
            loaded = True
        except Exception as error:  # noqa: BLE001 -- recorded
            load_error = f'{type(error).__name__}: {error}'
    solver = getattr(result, 'solver', None)
    return {'label': label, 'log_path': log_path, 'wall_s': wall, 'n_attempts': len(attempts), 'attempts': attempts,
            'solver_status': str(getattr(solver, 'status', None)),
            'termination_condition': str(getattr(solver, 'termination_condition', None)),
            'message': str(getattr(solver, 'message', None)),
            'succeeded': bool(srp._solver_result_succeeded(result)) if result is not None else False,
            'solution_loaded': loaded, 'load_error': load_error,
            'option_overrides': dict(ELASTIC_OPTION_OVERRIDES)}


def _decode(con_data, network, base, bound_map):
    """Physical identity of a target row: family, bus/branch/generator/node, hour."""
    comp = con_data.parent_component().name
    idx = con_data.index()
    out = {'row': con_data.name, 'component': comp}
    if con_data.name in bound_map:
        bm = bound_map[con_data.name]
        vidx = bm['index'] if isinstance(bm['index'], tuple) else (bm['index'],)
        out.update({'var': bm['var'], 'var_family': bm['var_family'], 'side': bm['side'], 'bound': bm['bound'],
                    'hour': vidx[-1] + 1})
        if bm['var_family'] in ('pg', 'qg'):
            gen = network.generators[vidx[0]]
            out.update({'gen_idx': vidx[0], 'gen_bus': gen.bus, 'gen_type': getattr(gen, 'gen_type', None),
                        'curtaillable': bool(gen.is_curtaillable()), 'bound_physical': bm['bound'] * base,
                        'unit': 'MW' if bm['var_family'] == 'pg' else 'MVAr'})
        elif bm['var_family'] in ('vmag_sqr', 'e', 'f'):
            out['bus'] = network.nodes[vidx[0]].bus_i
        elif bm['var_family'] == 'vmag':
            out['vmag_index'] = vidx[0]
        return out
    idx = idx if isinstance(idx, tuple) else (idx,)
    out['hour'] = idx[-1] + 1
    if comp in ('node_balance_p', 'node_balance_q', 'voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons'):
        out['bus'] = network.nodes[idx[0]].bus_i
    elif comp in ('branch_flow_limit', 'branch_flow_limit_ji'):
        br = network.branches[idx[0]]
        out.update({'branch_idx': idx[0], 'fbus': br.fbus, 'tbus': br.tbus, 'rate_mva': br.rate})
    elif comp in ('sg_capability', 'gen_pf_upper', 'gen_pf_lower', 'gen_pf_profile'):
        gen = network.generators[idx[0]]
        out.update({'gen_idx': idx[0], 'gen_bus': gen.bus, 'gen_type': getattr(gen, 'gen_type', None)})
    elif comp in ('uncoord_interface_p_fixed', 'uncoord_interface_q_fixed'):
        out['node'] = list(network.active_distribution_network_nodes)[idx[0]]
    elif comp == 'voltage_product_real_nonnegative':
        out['voltage_product_index'] = list(idx)
    return out


def read_slacks(model, targets, network, bound_map, allow_none=False):
    """Every target row's slack pair; nonzero ones (> SLACK_REPORT_THRESHOLD) decoded with physical units."""
    base = network.baseMVA
    xblock = model.component('_core_add_slack_variables')
    family_of = {c: fam for fam, comps in E1_ROW_FAMILIES.items() for c in comps}
    family_of.update({'uncoord_interface_p_fixed': 'interface_P_schedule', 'uncoord_interface_q_fixed':
                      'interface_Q_schedule'})
    bound_family_of = {v: fam for fam, vs in E1_BOUND_FAMILIES.items() for v in vs}
    nonzero, per_family, n_rows, total = [], {}, 0, 0.0
    for comp in targets:
        for cd in comp.values():
            if not cd.active:
                continue
            n_rows += 1
            name = cd.getname(fully_qualified=True)
            sp = xblock.component('_slack_plus_' + name)
            sn = xblock.component('_slack_minus_' + name)
            vp = None if sp is None else sp.value
            vn = None if sn is None else sn.value
            if (sp is not None and vp is None) or (sn is not None and vn is None):
                if allow_none:
                    continue
                raise RuntimeError(f'slack without a value at {name}')
            vp, vn = (vp or 0.0), (vn or 0.0)
            total += vp + vn
            if cd.name in bound_map:
                fam = bound_family_of[bound_map[cd.name]['var_family']]
            else:
                fam = family_of.get(cd.parent_component().name, cd.parent_component().name)
            pf = per_family.setdefault(fam, {'n_rows': 0, 'n_nonzero': 0, 'slack_sum_model_units': 0.0,
                                             'slack_max_model_units': 0.0})
            pf['n_rows'] += 1
            pf['slack_sum_model_units'] += vp + vn
            pf['slack_max_model_units'] = max(pf['slack_max_model_units'], vp, vn)
            if max(vp, vn) > SLACK_REPORT_THRESHOLD:
                pf['n_nonzero'] += 1
                rec = _decode(cd, network, base, bound_map)
                rec.update({'family': fam, 'slack_plus_lower_side': vp, 'slack_minus_upper_side': vn,
                            'body_at_solution': _val(cd.body), 'lower': _val(cd.lower) if cd.lower is not None
                            else None, 'upper': _val(cd.upper) if cd.upper is not None else None})
                comp_name = cd.parent_component().name
                if comp_name in ('node_balance_p', 'node_balance_q', 'uncoord_interface_p_fixed',
                                 'uncoord_interface_q_fixed'):
                    rec['net_slack_physical'] = (vp - vn) * base
                    rec['physical_unit'] = 'MW' if comp_name.endswith('_p') or comp_name.endswith('p_fixed') \
                        else 'MVAr'
                elif comp_name in ('branch_flow_limit', 'branch_flow_limit_ji'):
                    rate_pu = rec['rate_mva'] / base
                    rec['overload_mva'] = (math.sqrt(rate_pu ** 2 + vn) - rate_pu) * base
                elif cd.name in bound_map and bound_map[cd.name]['var_family'] in ('pg', 'qg'):
                    rec['violation_physical'] = max(vp, vn) * base
                nonzero.append(rec)
    return {'n_target_rows': n_rows, 'slack_total_model_units': total, 'per_family': per_family,
            'nonzero': sorted(nonzero, key=lambda r: -(r['slack_plus_lower_side'] + r['slack_minus_upper_side']))}


def interface_moves(model, network, schedule_2035_summer):
    """E2-E4: the accepted interface P/Q (expected_interface_* at the solution) against the passive target."""
    base = network.baseMVA
    out = {}
    for dn, node in enumerate(network.active_distribution_network_nodes):
        rows = []
        tgt = schedule_2035_summer[node]
        for p in model.periods:
            ep = model.expected_interface_pf_p[dn, p].value
            eq = model.expected_interface_pf_q[dn, p].value
            rows.append({'hour': p + 1, 'target_p_mw': tgt['p_mw'][p], 'target_q_mvar': tgt['q_mvar'][p],
                         'accepted_p_mw': None if ep is None else ep * base,
                         'accepted_q_mvar': None if eq is None else eq * base,
                         'move_p_mw': None if ep is None else ep * base - tgt['p_mw'][p],
                         'move_q_mvar': None if eq is None else eq * base - tgt['q_mvar'][p],
                         'v_pu_tn': model.expected_interface_vmag[dn, p].value})
        out[str(node)] = rows
    return out


# ======================================================================================================================
#  reproduction checks against W113 (committed evidence 2ee7a124)
# ======================================================================================================================
def _w113_records():
    recs = {}
    with open(_abs(os.path.join(W113_DIR, 'per_solve_record.jsonl'))) as handle:
        for line in handle:
            r = json.loads(line)
            recs[r['block']] = r
    with open(_abs(os.path.join(W113_DIR, 'failure.json'))) as handle:
        failure = json.load(handle)
    return recs, failure


def _attempt_fingerprint(a):
    s = (a.get('final_summary') or {}).get('unscaled') or {}
    return {'attempt': a.get('attempt'), 'exit': a.get('exit'), 'iterations': a.get('iterations'),
            'constraint_violation_unscaled': s.get('constraint_violation'), 'objective_unscaled': s.get('objective'),
            'dual_infeasibility_unscaled': s.get('dual_infeasibility')}


def compare_to_w113(record, w113):
    mine = [_attempt_fingerprint(a) for a in record['attempts']]
    theirs = [_attempt_fingerprint(a) for a in w113['attempts']]
    return {'block': record['block'], 'w114': mine, 'w113': theirs,
            'active_objective_value_w114': record.get('active_objective_value'),
            'active_objective_value_w113': w113.get('active_objective_value'),
            'identical_bitwise': (mine == theirs and record.get('active_objective_value')
                                  == w113.get('active_objective_value'))}


# ======================================================================================================================
#  run
# ======================================================================================================================
def run(dry_run_dir=None):
    import pyomo.environ as pe  # noqa: F401
    import p515_s53_w93_uncoordinated_benchmark as B
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    import p56a_oracle as O

    out_dir = _abs(OUT_DIR_REL) if dry_run_dir is None else dry_run_dir
    eval_dir = os.path.join(O.WORK_DIR, EVAL_ID)
    lock = False
    if dry_run_dir is None:
        failures = B.check_preconditions(out_dir)
        if os.path.exists(eval_dir):
            failures.append(f'IPOPT eval dir already exists (write-once): {eval_dir}')
        if failures:
            print('REFUSING TO RUN w114_passive_infeasibility:', *failures, sep='\n  ', flush=True)
            return 2
        B.acquire_lock('w114_passive_infeasibility')
        lock = True
    try:
        os.makedirs(out_dir, exist_ok=False)
        _log(f'W114 passive infeasibility diagnostic ({"DRY RUN, zero solves" if dry_run_dir else "declared solves "}'
             f'{"" if dry_run_dir else DECLARED_SOLVES}); output {out_dir}')
        _check_guard(0, 'start')
        if dry_run_dir is None:
            planning = O.fresh_planning(EVAL_ID)
        else:
            planning = copy.deepcopy(O.load_baseline()['planning'])
        solver_options = B.apply_arm_solver_options(srp, planning)
        candidate = B._x0_candidate(srp, planning)
        tn = planning.transmission_network
        Y, D = _key(tn.years, YEAR), _key(tn.days, DAY)
        network = tn.network[Y][D]

        certified = B._load_pickle_verified(B.COORDINATED['certified_models'])
        reference = UB.coordinated_reference_structure(planning, certified)
        coord_dso = UB.get_dso_interface_schedule(planning, certified['dso'])
        coord_tso = UB.get_tso_interface_schedule(planning, certified['tso'])
        coord_state = tn_state(certified['tso'][Y][D], network)
        del certified

        dso_models, dso_build = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm=UB.ARM_PASSIVE,
                                                        curtailment_penalty=B.TIE_BREAKER['decision']['passive_dso'])
        structure = UB.check_arm_structures(planning, {'dso': dso_models}, reference, dso_build_record=dso_build)
        start_records = UB.apply_start(planning, {'tso': None, 'dso': dso_models}, start=UB.START_COLD,
                                       agents=('DSO',))
        result = {'stage': 'P5.15 W114 passive-arm TSO 2035 Summer infeasibility diagnostic', 'utc_start': _utc(),
                  'dry_run': dry_run_dir is not None,
                  'git_head': B.H._git(['rev-parse', 'HEAD']), 'script': THIS_FILE,
                  'script_sha256': B.H.sha256_file(os.path.abspath(__file__)),
                  'module_sha256': {n: B.H.sha256_file(_abs(n)) for n in B.FROZEN_SPEC_BOUND_FILES},
                  'frozen_benchmark_spec': B._frozen_spec_identity(),
                  'instance': {'problem': 'SRP1', 'label': 'x0', 'candidate_key': B.X0['candidate_key'],
                               'block': f'TSO|-|{YEAR}|{DAY}', 'arm': 'passive', 'start': 'cold',
                               'dso_decision_tie_breaker': B.TIE_BREAKER['decision']['passive_dso'],
                               'tso_decision_tie_breaker': B.TIE_BREAKER['decision']['tso'],
                               'coordinated_cell': B.COORDINATED['eval_key'],
                               'coordinated_models_sha256': B.COORDINATED['certified_models']['sha256']},
                  'diagnostic_only': ('elastic copies are diagnostics; they never feed any arm value and no arm '
                                      'definition changes'),
                  'declared_solves': DECLARED_SOLVES, 'declared_total': DECLARED_TOTAL,
                  'solver_options_arm': solver_options, 'elastic_option_overrides': ELASTIC_OPTION_OVERRIDES,
                  'slack_report_threshold_model_units': SLACK_REPORT_THRESHOLD,
                  'dso_structure_check_2035_summer': {k: v for k, v in structure.items()
                                                      if k.endswith(f'|{YEAR}|{DAY}')},
                  'dso_start_records_2035_summer': {k: v for k, v in start_records.items()
                                                    if k.endswith(f'|{YEAR}|{DAY}')},
                  'coordinated_tn_state_2035_summer': coord_state, 'guard': []}
        sink = None if dry_run_dir else B.SolveSink(out_dir)
        w113_recs, w113_failure = _w113_records()

        # ---- step 1: the three passive DSO blocks of 2035 Summer, cold
        dso_records = {}
        passive = copy.deepcopy(coord_dso)
        if dry_run_dir is None:
            for node_id in sorted(planning.distribution_networks):
                dn = planning.distribution_networks[node_id]
                y, d = _key(dn.years, YEAR), _key(dn.days, DAY)
                _res, rec = UB._solve_block(planning, dn, dn.network[y][d], dso_models[node_id][y][d], kind='DSO',
                                            node_id=node_id, year=y, day=d, phase='w114:passive:cold:dso',
                                            record_callback=sink)
                dso_records[rec['block']] = compare_to_w113(rec, w113_recs[rec['block']])
            result['guard'].append(_check_guard(DECLARED_SOLVES['dso_passive_cold_2035_summer'], 'after DSO'))
            sched = UB.get_dso_interface_schedule(planning, dso_models)
            for node_id in passive:
                passive[node_id][Y][D] = sched[node_id][_key(sched[node_id].keys(), YEAR)][DAY]
        result['dso_reproduction_vs_w113'] = dso_records
        passive_2035 = {n: passive[n][Y][D] for n in passive}
        result['targets_note'] = ('TSO build targets: 2035 Summer = the passive schedule (these three DSO solves); every '
                                  'other (year, day) = the settled coordinated DSO schedule as a placeholder (those '
                                  'TSO blocks are never solved)' if dry_run_dir is None else
                                  'DRY RUN: every (year, day) = the coordinated schedule (no DSO solve)')
        result['schedule_2035_summer_passive_vs_coordinated'] = schedule_table(passive, coord_dso, coord_tso, Y, D)

        # ---- step 2: the TSO arm block at that schedule, and its reproduction
        tso_model, tso_build = UB.build_tso_arm_model(planning, candidate['total_capacity'], passive,
                                                      curtailment_penalty=B.TIE_BREAKER['decision']['tso'],
                                                      coupling=UB.TSO_COUPLING_FIXED,
                                                      pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
        tso_structure = UB.check_arm_structures(planning, {'tso': tso_model}, reference, tso_build_record=tso_build,
                                                tso_coupling=UB.TSO_COUPLING_FIXED)
        UB.apply_start(planning, {'tso': tso_model, 'dso': None}, start=UB.START_COLD, agents=('TSO',))
        block = tso_model[Y][D]
        result['tso_structure_check_2035_summer'] = {k: v for k, v in tso_structure.items()
                                                     if k.endswith(f'|{YEAR}|{DAY}')}
        result['tso_build_point_capacity_2035_summer'] = [
            {'hour': h['hour'], 'pg_capacity_total_mw': h['pg_capacity_total_mw'],
             'qg_capacity_total_mvar': h['qg_capacity_total_mvar'],
             'generators_pg_ub_mw': {str(g['gen_idx']): g['pg_ub_mw'] for g in h['generators']}}
            for h in tn_state(block, network)]
        if dry_run_dir is None:
            try:
                _r, rec = UB._solve_block(planning, tn, network, block, kind='TSO', node_id=None, year=Y, day=D,
                                          phase='w114:passive:cold:tso', record_callback=sink)
                result['tso_reproduction'] = {'raised_arm_solve_failure': False, 'record_summary': rec['summary']}
            except UB.ArmSolveFailure as failure:
                result['tso_reproduction'] = {'raised_arm_solve_failure': True, 'error': str(failure),
                                              'vs_w113': compare_to_w113(failure.record, w113_failure['record'])}
            result['guard'].append(_check_guard(DECLARED_SOLVES['dso_passive_cold_2035_summer']
                                                + DECLARED_SOLVES['tso_arm_block_reproduction'], 'after TSO repro'))
            _log(f"TSO reproduction: {result['tso_reproduction'].get('vs_w113', {}).get('identical_bitwise')}")

        # ---- step 3: the elastic copies
        result['elastic'] = {}
        for label, kind, rows in ELASTIC_VARIANTS:
            c, targets, bound_map, classification = build_elastic_copy(block, kind, rows)
            entry = {'kind': kind, 'target_components': [t.name for t in targets],
                     'n_slack_vars': len(list(c.component('_core_add_slack_variables').component_objects(pe.Var))),
                     'classification': classification}
            if dry_run_dir is None:
                entry['solve'] = elastic_solve(planning, tn, network, c, label)
                entry['slacks'] = read_slacks(c, targets, network, bound_map)
                entry['elastic_objective_value'] = _val(c.component('_core_add_slack_variables')._slack_objective)
                entry['interface_accepted_vs_passive_target'] = interface_moves(c, network, passive_2035)
                entry['tn_state'] = tn_state(c, network)
                _log(f"{label}: {entry['solve']['termination_condition']} loaded {entry['solve']['solution_loaded']} "
                     f"slack total {entry['slacks']['slack_total_model_units']:.6g} nonzero "
                     f"{ {k: v['n_nonzero'] for k, v in entry['slacks']['per_family'].items()} }")
            else:
                entry['slacks_dry'] = read_slacks(c, targets, network, bound_map, allow_none=True)
                entry['tn_state_dry_hours'] = len(tn_state(c, network))
            result['elastic'][label] = entry
        if dry_run_dir is None:
            result['guard'].append(_check_guard(DECLARED_TOTAL, 'end'))
        else:
            result['guard'].append(_check_guard(0, 'dry-run end'))
        result['utc_end'] = _utc()
        path = os.path.join(out_dir, 'w114_passive_infeasibility.json')
        with open(path, 'x') as handle:
            GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log(f'wrote {path}; guard {dict(_GUARD.counts)}')
        return 0
    finally:
        if _GUARD is not None:
            _GUARD.uninstall()
        if lock:
            B.release_lock()


def manifest():
    import p515_s44_campaign_harness as H
    out_dir = _abs(OUT_DIR_REL)
    entries = {}
    for name in sorted(os.listdir(out_dir)):
        if name != 'manifest_sha256.json':
            entries[os.path.join(OUT_DIR_REL, name)] = H.sha256_file(os.path.join(out_dir, name))
    entries[LAUNCH_LOG_REL] = H.sha256_file(_abs(LAUNCH_LOG_REL))
    logs = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', EVAL_ID, 'logs')
    ipopt = {os.path.join(logs, n): H.sha256_file(_abs(os.path.join(logs, n))) for n in sorted(os.listdir(_abs(logs)))}
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump({'files': entries, 'ipopt_logs_gitignored_hash_record': ipopt}, handle, indent=1, sort_keys=True)
    print(f'manifest: {len(entries)} files, {len(ipopt)} IPOPT logs')
    return 0


if __name__ == '__main__':
    if MANIFEST:
        sys.exit(manifest())
    if DRY_RUN:
        sys.exit(run(dry_run_dir=sys.argv[sys.argv.index('--dry-run') + 1]))
    sys.exit(run())
