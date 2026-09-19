"""
P5.15 Addendum 25 item 1 (Task A) -- per-node zero-storage evaluability check,
ZERO SOLVES (armed `SolveProfileGuard(permitted=())` for the whole script,
`verify(0)` at the end).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item1_harness.per_node_zero_check`:

    "BEFORE any run, zero-solve: establish that a node with zero storage
    (s = e = 0) while other nodes carry storage is evaluable (construction,
    S_ref normalization, ESSO model, consensus entries, EFC/SoH captures,
    evaluator). If not, STOP and report; no structural substitute (e.g.
    removing the node) without authorization."

Scope: the PER-NODE case only (some nodes zero, others positive). The
all-zero case x = 0 is owned by another Worker
(`p515_s44_addendum26_confirmations.py`) and is NOT examined here.

Candidates (per-node (s_mva, e_mwh) at investment year 2025, the three the
gate evaluates):
  c_star       : 0.96875 / 3.875 at nodes 5, 7, 9 (positive-only reference)
  paper_plan   : node 7 at 1.62 / 3.24; nodes 5, 9 zero
  node7_empty  : 0.96875 / 3.875 at nodes 5, 9; node 7 zero

For each candidate, through the REAL production functions (never
re-implemented), with every `.optimize` replaced by a never-solving stub on
the TransmissionNetwork / DistributionNetwork / SharedEnergyStorageData
INSTANCES of this script's own planning objects (the technique of
`p515_s32_zero_solve_checks._build_admm_ready_state` /
`p515_g_g1_g4_admm_gates._s38_stub_tso_optimize`):

  1. CONSTRUCTION -- `p515_g_g1_g4_admm_gates._construct_arm_planning` with
     `investment_map` (the harness path), exactly as `run_admm_arm` calls
     it: the candidate's investment and total_capacity dicts are recorded
     per node and year; for C* the map path is compared, for exact
     equality, against the uniform (`investment_map=None`) path the
     reference arm D used.
  2. MODELS -- `create_admm_variables`, `create_distribution_networks_models`,
     `create_transmission_network_model`, `create_shared_energy_storage_model`
     (candidate applied by production: `update_data_with_candidate_solution`,
     `update_model_with_candidate_solution`, `configure_shared_ess_
     operational_state`, `_configure_esso_cohort_state`): per node, the
     inactive flags production sets (network shared-ESS gate per block,
     ESSO per-cohort `_esso_cohort_inactive`), active-constraint counts per
     ESSO family, and the ESSO converter-circle rows (whose right-hand side
     is forced to zero at a zero node -- recorded, a numerical property that
     only a solve can exercise).
  3. ADMM PREPARATION -- `_prepare_*_objectives_for_admm`,
     `_resolve_esso_al_scale` (with the case file's FIXED sigma, the value
     production uses when `objective_scale` is set; the computed-sigma
     calibration assertion needs solved objectives and is not exercised),
     `update_{distribution,transmission}_models_to_admm`,
     `update_shared_energy_storage_model_to_admm`,
     `_initialize_shared_ess_consensus`, `get_updated_capacities`.
  4. S_ref NORMALIZATION -- every one of production's shared-ESS
     normalization helpers (`_shared_ess_admm_normalization_pu` /
     `_mva`) evaluated with the EXACT per-node arguments the call sites
     pass (TSO, DSO and ESSO ratings, every node/year/day), recording the
     underlying per-node rating (zero at a zero node) and the resulting
     normalization (must be S_ref, finite, positive).
  5. CONSENSUS ENTRIES -- every `consensus_vars['ess'][agent]` and
     `dual_vars['ess']` entry exists and is finite for every node.
  6. PER-CYCLE EVALUATORS + CAPTURES -- `get_admm_residual_metrics` and
     `get_admm_boyd_residual_metrics` called ONCE, inside the SAME capture
     wrappers every oracle run installs (`s38_pf_capture_hooks` -> recourse-
     jump, ESS-entry-stride, SoH-floor and PF-entry-stride sidecars;
     `s39_exempt_until_capture_hooks` via one `_update_admm_penalties` call
     with `allow_update=False`); every numeric output checked finite;
     `_get_admm_efc_per_day_max`, `p514_n.capture_esso`,
     `_identify_soh_floor_rows`.
  7. TERMINAL EVALUATORS -- `write_boyd_terminal_s35ref` (which calls
     `write_interface_settlement_detail_s31c` -> `write_component_levels_
     terminal`, and `write_interface_voltage_terminal`) on the unsolved
     models, with one synthetic trajectory row (reported as synthetic).

Values at an unsolved point are NOT meaningful numbers and are not reported
as results; what this check establishes is STRUCTURAL: no exception, no
division by a node's zero rating, every entry present and finite, and the
zero node gated by production's own mechanisms.

Output (write-once): data/SRP1/Results/P515S44/per_node_zero_check/
    per_node_zero_check.json, manifest_sha256.json, plus the evaluator
    artifacts under evaluators/<candidate>/ and scratch/<candidate>/.

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s44_per_node_zero_check.py \\
        > data/SRP1/Results/P515S44/per_node_zero_check_launch.log 2>&1
"""

import hashlib
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 per-node zero-storage check (zero solves)').install()

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'per_node_zero_check')
SPEC_V14_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44',
                             'frozen_s44_selection_spec_v14_e4500e27.json')
EVAL_ID_PREFIX = 'p515s44_pnz'

CANDIDATES = {
    'c_star': {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)},
    'paper_plan': {5: (0.0, 0.0), 7: (1.62, 3.24), 9: (0.0, 0.0)},
    'node7_empty': {5: (0.96875, 3.875), 7: (0.0, 0.0), 9: (0.96875, 3.875)},
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _walk_numbers(obj, path=''):
    """Yield (path, value) for every numeric leaf (bools excluded)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f'{path}.{k}' if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, f'{path}[{i}]')
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


def _nonfinite_leaves(obj, limit=20):
    bad = [(p, v) for p, v in _walk_numbers(obj) if not math.isfinite(v)]
    return {'n_numeric_leaves': sum(1 for _ in _walk_numbers(obj)), 'n_nonfinite': len(bad),
            'first_nonfinite': [{'path': p, 'value': repr(v)} for p, v in bad[:limit]]}


def _stub_network_optimize(self, model, from_warm_start=False, print_header=True,
                           failure_snapshot_callback=None, pre_solve_snapshot_callback=None):
    """Never reaches a solver (same technique as p515_s32_zero_solve_checks._stub_optimize)."""
    return {year: {day: None for day in self.days} for year in self.years}


def _stub_esso_optimize(self, models, from_warm_start=False, cycle=None):
    """Never reaches a solver: SharedEnergyStorageData.optimize returns one result per node."""
    return {node_id: None for node_id in self.active_distribution_network_nodes}


def _install_stubs(planning):
    planning.transmission_network.optimize = _stub_network_optimize.__get__(
        planning.transmission_network, type(planning.transmission_network))
    for dn in planning.distribution_networks.values():
        dn.optimize = _stub_network_optimize.__get__(dn, type(dn))
    sed = planning.shared_ess_data
    sed.optimize = _stub_esso_optimize.__get__(sed, type(sed))


def _candidate_summary(candidate, nodes, years):
    return {
        str(n): {
            'investment': {str(y): dict(candidate['investment'][n][y]) for y in years},
            'total_capacity': {str(y): dict(candidate['total_capacity'][n][y]) for y in years},
        }
        for n in nodes
    }


def _check_construction(label, investment_map, out_dir):
    """Step 1: the harness's construction path, and (C* only) map-vs-uniform equality."""
    report = {}
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(out_dir, 'scratch', label), report, k_override=None,
        investment_map=investment_map, eval_id=f'{EVAL_ID_PREFIX}_{label}_construct',
        num_max_iters_override=500, apply_rho=False)
    nodes = list(sed.active_distribution_network_nodes)
    years = list(sed.years)
    summary = _candidate_summary(candidate, nodes, years)
    invest_year = N.INVEST_YEAR
    honoured = {}
    for n in nodes:
        s_val, e_val = investment_map[n]
        inv = candidate['investment'][n]
        honoured[str(n)] = {
            'investment_year_entry_equals_map': (inv[invest_year]['s'] == s_val and inv[invest_year]['e'] == e_val),
            'other_years_zero': all(inv[y]['s'] == 0.0 and inv[y]['e'] == 0.0 for y in years if y != invest_year),
            'zero_node': (s_val == 0.0 and e_val == 0.0),
            'total_capacity_all_years_zero': all(
                candidate['total_capacity'][n][y]['s'] == 0.0 and candidate['total_capacity'][n][y]['e'] == 0.0
                for y in years),
        }
    out = {
        'investment_year': invest_year,
        'active_nodes': nodes,
        'years': [str(y) for y in years],
        'report_instance': report.get('instance'),
        'candidate': summary,
        'map_honoured_per_node': honoured,
        'map_honoured_all': all(v['investment_year_entry_equals_map'] and v['other_years_zero']
                                for v in honoured.values()),
        'zero_nodes_have_zero_total_capacity_every_year': all(
            v['total_capacity_all_years_zero'] for v in honoured.values() if v['zero_node']),
    }
    del planning
    if label == 'c_star':
        report_u = {}
        planning_u, _sed_u, candidate_u = G._construct_arm_planning(
            's39_D', os.path.join(out_dir, 'scratch', 'c_star_uniform'), report_u, k_override=None,
            investment_map=None, eval_id=f'{EVAL_ID_PREFIX}_c_star_uniform_construct',
            num_max_iters_override=500, apply_rho=False)
        out['uniform_path_report_instance'] = report_u.get('instance')
        out['map_path_candidate_equals_uniform_path_candidate_exactly'] = (candidate_u == candidate)
        del planning_u
    return out, candidate


def _esso_family_counts(model):
    fams = ('rated_s_capacity_unit', 'rated_e_capacity_unit', 'rated_s_capacity', 'rated_e_capacity',
            'available_s_capacity_unit', 'available_e_capacity_unit', 'energy_storage_charging_discharging',
            'energy_storage_capacity_degradation', 'energy_storage_limits',
            'energy_storage_cohort_pnet_share_h3', 'energy_storage_operation_agg')
    out = {}
    for f in fams:
        comp = getattr(model, f)
        rows = list(comp.values())
        out[f] = {'rows': len(rows), 'active': sum(1 for c in rows if c.active)}
    unfixed = sum(1 for v in model.component_data_objects(pe.Var) if not v.fixed)
    total = sum(1 for _ in model.component_data_objects(pe.Var))
    out['_vars'] = {'total': total, 'unfixed': unfixed}
    return out


def _esso_circle_rows(model):
    """The converter-capability rows `es_pnet^2 + es_qnet^2 <= es_s_rated^2`
    (every second row of `energy_storage_operation_agg`, identified by body
    structure, not index arithmetic): count, active, and whether es_s_rated[y]
    is forced to zero by the linear rated-capacity equalities (investment
    Param zero for every cohort within its lifetime)."""
    circle = []
    for c in model.energy_storage_operation_agg.values():
        if c.upper is not None and not c.equality:
            circle.append(c)
    per_year_rhs_forced_zero = {}
    for y in model.years:
        contributing = [(y_inv, pe.value(model.es_s_investment_fixed[y_inv]))
                        for y_inv in model.years
                        if not model.es_s_rated_per_unit[y_inv, y].fixed]
        per_year_rhs_forced_zero[str(y)] = all(val == 0.0 for _yi, val in contributing)
    return {'n_rows': len(circle), 'n_active': sum(1 for c in circle if c.active),
            'es_s_rated_forced_zero_by_equalities_per_year': per_year_rhs_forced_zero}


def _check_models_and_evaluators(label, investment_map, candidate, out_dir):
    """Steps 2-7 on a SEPARATE fresh planning object (its own eval id)."""
    result = {}
    planning = G.O.fresh_planning(f'{EVAL_ID_PREFIX}_{label}_models')
    G._set_results_dir_for_arm(planning, os.path.join(out_dir, 'scratch', label + '_models', 'results'))
    _install_stubs(planning)
    admm_params = planning.params.admm
    sed = planning.shared_ess_data
    tn = planning.transmission_network
    dns = planning.distribution_networks
    nodes = list(sed.active_distribution_network_nodes)
    years = list(sed.years)
    days = list(sed.days)
    cand = deepcopy(candidate)

    # ---- step 2: models -------------------------------------------------------
    consensus_vars, dual_vars = srp.create_admm_variables(planning)
    dso_models, _r_dso = srp.create_distribution_networks_models(
        dns, consensus_vars, cand['total_capacity'], parallel_execution=planning.parallel_execution)
    tso_model, _r_tso = srp.create_transmission_network_model(planning, consensus_vars, cand['total_capacity'])
    esso_model, _r_esso = srp.create_shared_energy_storage_model(sed, consensus_vars, cand['investment'])

    per_node = {}
    for n in nodes:
        zero = (investment_map[n] == (0.0, 0.0))
        dso_gate = {}
        for y in years:
            for d in days:
                m = dso_models[n][y][d]
                net = dns[n].network[y][d]
                idx = net.get_shared_energy_storage_idx(net.get_reference_node_id())
                s_fix = pe.value(m.shared_es_s_rated_fixed[idx])
                e_fix = pe.value(m.shared_es_e_rated_fixed[idx])
                exp_p_fixed = all(m.expected_shared_ess_p[p].fixed for p in m.periods)
                dso_gate[f'{y}|{d}'] = {'s_rated_fixed_pu': s_fix, 'e_rated_fixed_pu': e_fix,
                                        'expected_shared_ess_p_all_fixed': exp_p_fixed}
        tso_gate = {}
        for y in years:
            for d in days:
                m = tso_model[y][d]
                net = tn.network[y][d]
                idx = net.get_shared_energy_storage_idx(n)
                tso_gate[f'{y}|{d}'] = {
                    's_rated_fixed_pu': pe.value(m.shared_es_s_rated_fixed[idx]),
                    'e_rated_fixed_pu': pe.value(m.shared_es_e_rated_fixed[idx]),
                    'expected_shared_ess_p_all_fixed': all(m.expected_shared_ess_p[idx, p].fixed for p in m.periods),
                }
        em = esso_model[n]
        per_node[str(n)] = {
            'zero_node': zero,
            'esso_cohort_inactive': {str(k): v for k, v in em._esso_cohort_inactive.items()},
            'esso_family_counts': _esso_family_counts(em),
            'esso_converter_circle_rows': _esso_circle_rows(em),
            'dso_shared_ess_gate_all_blocks_inactive': all(
                v['s_rated_fixed_pu'] == 0.0 and v['e_rated_fixed_pu'] == 0.0 and v['expected_shared_ess_p_all_fixed']
                for v in dso_gate.values()),
            'dso_shared_ess_gate_all_blocks_active': all(
                v['s_rated_fixed_pu'] > 0.0 and not v['expected_shared_ess_p_all_fixed'] for v in dso_gate.values()),
            'tso_shared_ess_gate_all_blocks_inactive': all(
                v['s_rated_fixed_pu'] == 0.0 and v['e_rated_fixed_pu'] == 0.0 and v['expected_shared_ess_p_all_fixed']
                for v in tso_gate.values()),
            'tso_shared_ess_gate_all_blocks_active': all(
                v['s_rated_fixed_pu'] > 0.0 and not v['expected_shared_ess_p_all_fixed'] for v in tso_gate.values()),
            'dso_blocks': dso_gate,
            'tso_blocks': tso_gate,
        }
    result['models_per_node'] = per_node

    # ---- step 3: ADMM preparation ------------------------------------------------
    srp._prepare_distribution_objectives_for_admm(dns, dso_models)
    srp._prepare_transmission_objectives_for_admm(tn, tso_model)
    sigma_fixed = admm_params.objective_scale
    if sigma_fixed is None:
        raise RuntimeError('case file objective_scale is None -- expected the fixed sigma of the D oracle')
    al_scale_esso, al_apply = srp._resolve_esso_al_scale(planning, admm_params, sigma_fixed)
    srp.update_distribution_models_to_admm(planning, dso_models, admm_params, sigma_fixed)
    srp.update_transmission_model_to_admm(planning, tso_model, admm_params, sigma_fixed)
    srp.update_shared_energy_storage_model_to_admm(planning, esso_model, admm_params, al_scale_esso=al_scale_esso)
    srp._initialize_shared_ess_consensus(planning, consensus_vars)
    sess_caps = sed.get_updated_capacities(esso_model)
    result['admm_preparation'] = {
        'sigma_fixed_used': sigma_fixed, 'al_scale_esso': al_scale_esso, 'al_scale_applied': al_apply,
        'esso_admm_objective_active_per_node': {str(n): bool(esso_model[n].admm_objective.active) for n in nodes},
        'get_updated_capacities_structure_ok': all(
            set(sess_caps[n].keys()) == set(years) and all(
                _finite(sess_caps[n][y]['s_available']) and _finite(sess_caps[n][y]['e_available']) for y in years)
            for n in nodes),
        'note': ('values read at the unsolved point (never solved); structure/finite only. '
                 '_compute_common_admm_objective_scale (the calibration assertion of the computed '
                 'sigma) needs solved block objectives and is NOT exercised here.'),
    }

    # ---- step 4: S_ref normalization, at every call-site argument ---------------
    reference = srp._admm_shared_ess_reference_mva(admm_params)
    floor = admm_params.shared_ess_normalization_floor_mva
    sref_rows = {}
    all_equal_sref = True
    for n in nodes:
        node_rows = {'tso_rating_arg_mva': [], 'dso_rating_arg_mva': [], 'esso_rating_arg_mva': [],
                     'tso_norm_pu_times_sbase': [], 'dso_norm_pu_times_sbase': [], 'tso_norm_mva': [],
                     'dso_norm_mva': [], 'esso_norm_mva': []}
        for y in years:
            esso_idx = sed.get_shared_energy_storage_idx(n)
            esso_s = sed.shared_energy_storages[y][esso_idx].s
            esso_norm = srp._shared_ess_admm_normalization_mva(esso_s, floor, reference_mva=reference)
            node_rows['esso_rating_arg_mva'].append(esso_s)
            node_rows['esso_norm_mva'].append(esso_norm)
            for d in days:
                tnet = tn.network[y][d]
                tidx = tnet.get_shared_energy_storage_idx(n)
                t_s_pu = tnet.shared_energy_storages[tidx].s
                t_base = tnet.baseMVA
                dnet = dns[n].network[y][d]
                didx = dnet.get_shared_energy_storage_idx(dnet.get_reference_node_id())
                d_s_pu = dnet.shared_energy_storages[didx].s
                d_base = dnet.baseMVA
                t_pu = srp._shared_ess_admm_normalization_pu(t_s_pu, t_base, floor, reference_mva=reference)
                d_pu = srp._shared_ess_admm_normalization_pu(d_s_pu, d_base, floor, reference_mva=reference)
                t_mva = srp._shared_ess_admm_normalization_mva(t_s_pu * t_base, floor, reference_mva=reference)
                d_mva = srp._shared_ess_admm_normalization_mva(d_s_pu * d_base, floor, reference_mva=reference)
                node_rows['tso_rating_arg_mva'].append(t_s_pu * t_base)
                node_rows['dso_rating_arg_mva'].append(d_s_pu * d_base)
                node_rows['tso_norm_pu_times_sbase'].append(t_pu * t_base)
                node_rows['dso_norm_pu_times_sbase'].append(d_pu * d_base)
                node_rows['tso_norm_mva'].append(t_mva)
                node_rows['dso_norm_mva'].append(d_mva)
        norms = (node_rows['tso_norm_pu_times_sbase'] + node_rows['dso_norm_pu_times_sbase']
                 + node_rows['tso_norm_mva'] + node_rows['dso_norm_mva'] + node_rows['esso_norm_mva'])
        ok = all(_finite(v) and v > 0 and math.isclose(v, reference, rel_tol=0, abs_tol=1e-12) for v in norms)
        all_equal_sref = all_equal_sref and ok
        sref_rows[str(n)] = {
            'rating_args_min_max_mva': {
                k: [min(v), max(v)] for k, v in node_rows.items() if k.endswith('_arg_mva')},
            'normalizations_min_max_mva': {
                k: [min(v), max(v)] for k, v in node_rows.items() if 'norm' in k},
            'all_normalizations_equal_S_ref_finite_positive': ok,
        }
    result['s_ref_normalization'] = {
        'S_ref_mva': reference, 'floor_mva': floor, 'per_node': sref_rows,
        'all_nodes_all_sites_equal_S_ref': all_equal_sref,
        'call_sites_covered': [
            'update_transmission_model_to_admm (_shared_ess_admm_normalization_pu, TSO network .s)',
            'update_distribution_models_to_admm (_shared_ess_admm_normalization_pu, DSO network .s)',
            'update_shared_energy_storage_model_to_admm (_shared_ess_admm_normalization_mva, ESSO .s)',
            'get_admm_residual_metrics / get_admm_boyd_residual_metrics (_mva, TSO/DSO/ESSO .s)',
            'TSO dual/proximal update (_pu, TSO network .s)',
        ],
    }

    # ---- step 5: consensus entries ------------------------------------------------
    cons = {}
    for n in nodes:
        entries = {}
        for agent in ('tso', 'dso', 'esso', 'z'):
            vals = []
            for y in years:
                for d in days:
                    for pt in ('p', 'q'):
                        vals.extend(consensus_vars['ess'][agent]['current'][n][y][d][pt])
            entries[agent] = {'n': len(vals), 'all_finite': all(_finite(v) for v in vals)}
        dvals = []
        for agent in dual_vars['ess']:
            for y in years:
                for d in days:
                    for pt in ('p', 'q'):
                        dvals.extend(dual_vars['ess'][agent]['current'][n][y][d][pt])
        entries['dual_ess_all_agents'] = {'n': len(dvals), 'all_finite': all(_finite(v) for v in dvals)}
        cons[str(n)] = entries
    result['consensus_entries'] = {
        'per_node': cons,
        'all_present_and_finite': all(e['n'] > 0 and e['all_finite'] for node in cons.values() for e in node.values()),
        'expected_entries_per_agent_per_node': len(years) * len(days) * 2 * planning.num_instants,
    }

    # ---- step 6: per-cycle evaluators inside the oracle's own capture wrappers --
    ev_dir = os.path.join(out_dir, 'evaluators', label)
    os.makedirs(ev_dir, exist_ok=True)
    probe = G.O.fresh_planning(f'{EVAL_ID_PREFIX}_{label}_floor_probe')
    import shared_energy_storage_data as SED
    probe_models = {nid: SED._build_subproblem(probe.shared_ess_data, nid)
                    for nid in probe.shared_ess_data.active_distribution_network_nodes}
    floor_rows_by_node, floor_counts = G._identify_soh_floor_rows(probe_models)
    del probe_models, probe
    paths = {k: os.path.join(ev_dir, f) for k, f in (
        ('recourse_jump', 'recourse_jump_sidecar_baseline.jsonl'),
        ('ess_stride', 'ess_entry_stride_baseline.jsonl'),
        ('floor', 'soh_floor_sidecar_baseline.jsonl'),
        ('pf_stride', 'pf_entry_stride_zero_check.jsonl'),
        ('exempt', 'ess_exempt_until_state_zero_check.jsonl'))}
    with G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                paths['pf_stride'], floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(paths['exempt']):
        residual_metrics = srp.get_admm_residual_metrics(planning, tso_model, dso_models, esso_model, consensus_vars)
        boyd_metrics = srp.get_admm_boyd_residual_metrics(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
        penalty_out = srp._update_admm_penalties(
            tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, admm_params,
            iter=1, allow_update=False, freeze_state=srp._init_admm_freeze_state())
    sidecar_lines = {}
    for k, p in paths.items():
        with open(p) as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        sidecar_lines[k] = {'n_lines': len(lines), 'nonfinite': _nonfinite_leaves(lines)}
    with open(paths['floor']) as handle:
        floor_entries = json.loads(handle.readline())['entries']
    floor_by_node = {}
    for e in floor_entries:
        nd = floor_by_node.setdefault(str(e['node_id']), {'n_entries': 0, 'efc_values': [], 'soh_values': []})
        nd['n_entries'] += 1
        nd['efc_values'].append(e['efc_per_day'])
        nd['soh_values'].append(e['es_soh_per_unit_cumul'])
    capture = N.capture_esso(esso_model, sed)
    efc_max = srp._get_admm_efc_per_day_max(esso_model)
    result['per_cycle_evaluators_and_captures'] = {
        'get_admm_residual_metrics_nonfinite': _nonfinite_leaves(residual_metrics),
        'get_admm_boyd_residual_metrics_nonfinite': _nonfinite_leaves(boyd_metrics),
        'update_admm_penalties_returned_tuple_len': len(penalty_out),
        'sidecars': sidecar_lines,
        'soh_floor_rows_per_node_structural': {str(k): v for k, v in floor_counts.items()},
        'soh_floor_sidecar_entries_per_node': {
            n: {'n_entries': v['n_entries'],
                'efc_per_day_all_none': all(x is None for x in v['efc_values']),
                'soh_values_distinct': sorted({x for x in v['soh_values'] if x is not None})}
            for n, v in floor_by_node.items()},
        'capture_esso_per_node_keys': {n: sorted(v.keys()) for n, v in capture.items()},
        'capture_esso_efc_per_day_max_per_node': {n: v['efc_per_day_max'] for n, v in capture.items()},
        'capture_esso_nonfinite': _nonfinite_leaves(capture),
        '_get_admm_efc_per_day_max': efc_max,
        'note': ('one call of each per-cycle function at the UNSOLVED point, inside the oracle run\'s '
                 'own capture wrappers; checks structure (no exception, no zero-rating division, '
                 'finite leaves), not values'),
    }

    # ---- step 7: terminal evaluators ------------------------------------------------
    models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
    synthetic_row = {'cycle': 1, 'cycle_convergence': False, 'consecutive_converged_cycles': 0,
                     'recourse': None, 'gross_operational_cost': None}
    report = {'converged_at_cycle': None, 'cycles_run': 1,
              's34_recourse_jump_sidecar_path': os.path.relpath(paths['recourse_jump'], REPO),
              's34_ess_entry_stride_sidecar_path': os.path.relpath(paths['ess_stride'], REPO),
              's35ref_soh_floor_sidecar_path': os.path.relpath(paths['floor'], REPO)}
    terminal_ok, terminal_err = True, None
    try:
        G.write_boyd_terminal_s35ref(planning, sed, models, [synthetic_row], report, ev_dir, 'zero_check',
                                     floor_rows_by_node=floor_rows_by_node, floor_sidecar_path=paths['floor'])
    except Exception as error:  # noqa: BLE001 -- recorded, the check then FAILS
        terminal_ok = False
        terminal_err = f'{type(error).__name__}: {error}\n{traceback.format_exc()}'
    terminal_files = {}
    for fname in ('boyd_terminal.json', 'component_levels_terminal.json',
                  'interface_settlement_detail_s31c.json', 'interface_voltage_terminal.json'):
        p = os.path.join(ev_dir, fname)
        if os.path.exists(p):
            with open(p) as handle:
                data = json.load(handle)
            terminal_files[fname] = {'exists': True, 'nonfinite': _nonfinite_leaves(data)}
        else:
            terminal_files[fname] = {'exists': False}
    cl_path = os.path.join(ev_dir, 'component_levels_terminal.json')
    rc_keys = None
    if os.path.exists(cl_path):
        with open(cl_path) as handle:
            rc_keys = sorted(json.load(handle).get('recourse_components', {}).keys())
    result['terminal_evaluators'] = {
        'write_boyd_terminal_s35ref_ok': terminal_ok, 'error': terminal_err,
        'files': terminal_files, 'recourse_components_keys': rc_keys,
        'synthetic_trajectory_row_used': synthetic_row,
    }
    return result


def main():
    if os.path.exists(OUT_ROOT):
        raise SystemExit(f'output root already exists (write-once): {OUT_ROOT}')
    for label in CANDIDATES:
        for suffix in ('construct', 'models', 'floor_probe'):
            eid = f'{EVAL_ID_PREFIX}_{label}_{suffix}'
            if os.path.exists(os.path.join(G.O.WORK_DIR, eid)):
                raise SystemExit(f'eval dir already exists (never reusable): {eid}')
    if os.path.exists(os.path.join(G.O.WORK_DIR, f'{EVAL_ID_PREFIX}_c_star_uniform_construct')):
        raise SystemExit('eval dir already exists: c_star_uniform_construct')
    os.makedirs(OUT_ROOT)
    started = time.time()
    payload = {
        'stage': 'P5.15 Addendum 25 item 1 Task A -- per-node zero-storage evaluability (zero solves)',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 25',
                      os.path.relpath(SPEC_V14_PATH, REPO) + ' item1_harness.per_node_zero_check'],
        'spec_v14_sha256': _sha256_file(SPEC_V14_PATH),
        'case_file_sha256': _sha256_file(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'per-node zero (some nodes zero, others positive); the all-zero x = 0 case is NOT examined here',
        'candidates': {k: {str(n): list(v) for n, v in m.items()} for k, m in CANDIDATES.items()},
        'per_candidate': {},
    }
    for label, investment_map in CANDIDATES.items():
        print(f'[S44-PNZ] ===== candidate {label}: {investment_map} =====', flush=True)
        entry = {}
        try:
            construction, candidate = _check_construction(label, investment_map, OUT_ROOT)
            entry['construction'] = construction
            entry.update(_check_models_and_evaluators(label, investment_map, candidate, OUT_ROOT))
            entry['exception'] = None
        except Exception as error:  # noqa: BLE001 -- a break IS the finding; recorded, not hidden
            entry['exception'] = f'{type(error).__name__}: {error}\n{traceback.format_exc()}'
            print(f'[S44-PNZ] *** candidate {label} RAISED: {error}', flush=True)
        payload['per_candidate'][label] = entry

    # ---- verdict (structural; zero-node-specific items) ------------------------------
    verdict_items = {}
    for label, entry in payload['per_candidate'].items():
        if entry.get('exception'):
            verdict_items[label] = {'no_exception': False}
            continue
        mpn = entry['models_per_node']
        zero_nodes = [n for n, v in mpn.items() if v['zero_node']]
        pos_nodes = [n for n, v in mpn.items() if not v['zero_node']]
        pce = entry['per_cycle_evaluators_and_captures']
        items = {
            'no_exception': True,
            'map_honoured': entry['construction']['map_honoured_all'],
            'zero_nodes_zero_total_capacity': entry['construction']['zero_nodes_have_zero_total_capacity_every_year'],
            'zero_nodes_esso_all_cohorts_inactive': all(
                all(mpn[n]['esso_cohort_inactive'].values()) for n in zero_nodes),
            'positive_nodes_esso_2025_cohort_active': all(
                not mpn[n]['esso_cohort_inactive']['0'] for n in pos_nodes),
            'zero_nodes_network_gate_inactive_every_block': all(
                mpn[n]['dso_shared_ess_gate_all_blocks_inactive'] and mpn[n]['tso_shared_ess_gate_all_blocks_inactive']
                for n in zero_nodes),
            'positive_nodes_network_gate_active_every_block': all(
                mpn[n]['dso_shared_ess_gate_all_blocks_active'] and mpn[n]['tso_shared_ess_gate_all_blocks_active']
                for n in pos_nodes),
            's_ref_all_sites_all_nodes': entry['s_ref_normalization']['all_nodes_all_sites_equal_S_ref'],
            'consensus_entries_present_finite': entry['consensus_entries']['all_present_and_finite'],
            'residual_metrics_finite': pce['get_admm_residual_metrics_nonfinite']['n_nonfinite'] == 0,
            'boyd_metrics_finite': pce['get_admm_boyd_residual_metrics_nonfinite']['n_nonfinite'] == 0,
            'sidecars_written_finite': all(v['n_lines'] >= 1 and v['nonfinite']['n_nonfinite'] == 0
                                           for v in pce['sidecars'].values()),
            'capture_esso_finite': pce['capture_esso_nonfinite']['n_nonfinite'] == 0,
            'zero_nodes_efc_none_not_error': all(
                pce['capture_esso_efc_per_day_max_per_node'][n] is None for n in zero_nodes),
            'terminal_evaluators_ok': (entry['terminal_evaluators']['write_boyd_terminal_s35ref_ok']
                                       and all(f.get('exists') and f['nonfinite']['n_nonfinite'] == 0
                                               for f in entry['terminal_evaluators']['files'].values())),
        }
        if label == 'c_star':
            items['c_star_map_equals_uniform'] = entry['construction'][
                'map_path_candidate_equals_uniform_path_candidate_exactly']
        verdict_items[label] = items
    structural_pass = all(all(v.values()) for v in verdict_items.values())
    guard_failures = GUARD.verify(0)
    payload['verdict'] = {
        'items': verdict_items,
        'structural_evaluability_pass': structural_pass,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'not_established_zero_solve': [
            'numerical behaviour of the ESSO at a zero node: the converter-circle rows '
            'es_pnet^2 + es_qnet^2 <= es_s_rated^2 stay ACTIVE with es_s_rated forced to 0 by the '
            'linear rated-capacity equalities -- a degenerate (LICQ-failing) feasible set {q = 0}; '
            'whether IPOPT solves it cleanly every cycle is observable only in a run',
            'the zero node\'s PUBLISHED available capacity each cycle (get_updated_capacities reads '
            'SOLVED es_s/e_available_per_unit); the networks gate on |cap| <= 1e-10 '
            '(SHARED_ESS_ZERO_CAPACITY_TOLERANCE), so a solver residue above 1e-10 would publish a '
            'tiny ACTIVE storage -- observable only in a run (the harness records the terminal '
            'published capacities per node)',
        ],
    }
    payload['wall_clock_s'] = time.time() - started
    out_path = os.path.join(OUT_ROOT, 'per_node_zero_check.json')
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, files in os.walk(OUT_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
    with open(os.path.join(OUT_ROOT, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    GUARD.uninstall()
    print(f'[S44-PNZ] verdict items: {json.dumps(verdict_items, indent=1)}')
    print(f'[S44-PNZ] structural_evaluability_pass={structural_pass} guard={GUARD.counts} '
          f'guard_verify_0_failures={guard_failures}')
    if not structural_pass or guard_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
