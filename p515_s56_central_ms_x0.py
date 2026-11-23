"""P5.15 S56 validation step 3 -- centralized comparator at x = 0, alpha = 0, no voltage pin, on the SRP1-grid 3 x 3.

Task (Planner, S56 plan v3 section 7 row 3): on the SRP1-grid 3 x 3 instance (instances_v1/SRP1__s56_srp1grid_3x3.json, read
through the derived-instance reader `p515_s44_scale_measurement.read_planning_from_derived_case`; combined scenario
checksum 73d0a230...), at x = 0 (no storage) and alpha = 0 (no row 18), with no voltage pin and settlement weight 0, the
merged stochastic OPF of one (year, day) block decouples into nine independent single-scenario merged OPFs, so
C_central(block) = sum_s omega_s C_s. Two blocks, chosen zero-solve BEFORE any solve (frozen spec, `block_choice`).

PER BLOCK (production functions only; nothing in production edited):
  merged   centralized_coordination.combine_networks(TN, DNs) -> day network md -> Network.build_model(params); every
           pc, qc Var fixed at load.pd / load.qd (the central check's ruling 1); scalar Params set as the ADMM path sets
           them (spec `objective_alignment`); interface_settlement_weight stays 0; row 18 / pin components absent
  single s a deepcopy of md with every scenario array sliced to scenario s = (s_m, s_o): network cost_energy_p and
           cost_flex row s_m; every element (load, load flexibility, generator) array of shape (n_o, T) row s_o;
           probabilities [1.0]; num_oper_scenarios 1. Same build / fixing / alignment.
  solves   merged from the default start (production initialisers); the 9 singles from the default start; then a
           freshly built merged model warm-started from the 9 single solutions (primal values only, no duals,
           from_warm_start=False). 11 per block, 22 in all, all through `_solve_one` (Network.run_smopf).

PRE-SOLVE GATES (phase A; any failure refuses the run with ZERO solves):
  instance case sha256 and scenario checksum; probabilities and price arrays (energy, flexibility) identical across TN
  and every DN for every block; block choice recomputed == spec; per chosen block on a separately built gate model:
  (i) NO COUPLING: every Var component's scenario positions are read from its declaration (AST of network.py /
  model_construction_helpers.py); every active constraint's free variables carry ONE scenario label; no scenario-common
  free Var; (ii) SLICE IDENTITY at a seeded random point: for each of the 9 scenarios, the single model's constraints
  correspond one-to-one to the merged model's constraints of that scenario, with equal bodies, bounds, variable bounds
  and fixings, equal per-scenario objective expressions, and merged objective == sum_s omega_s single objective;
  (iii) alpha = 0 / no pin / settlement 0; capture checklist; every repo module imported is committed clean;
  solver_params.verbose off (production exit()s after a failed solve otherwise); the block-choice premise (every TN
  conventional generator has P_min = 0).

REPORTED BESIDE THE SOLVES (capture only): per solve the wall time and peak RSS (IPOPT child sampled every 0.1 s, and
the exact running maximum ru_maxrss); per block the planning weight (NetworkData.get_primal_value's
years[y] x days[d] / (1 + r)^(y - y0), stated, not applied); the relaxations the merge makes against the coordinated
models (dropped DN reference-generator P/Q limits, DN reference-node voltage limits, |P| against the interface rating R
that bounds the ADMM TSO's interface_delta) as scenario-hours outside and smallest margins. Predictions P1-P5 of the
frozen spec are evaluated after the guards are verified (P1 reads them). A solution counts as loaded only when
production's solver_result_succeeded(result) holds (audit fix, 2026-11-23: `result is not None` read failed solves
as loaded, the S56 G harness defect).

GUARDS. SolveProfileGuard(permitted=[(this file, '_solve_one')]) installed before any project import, verified at
exactly 22 solves and 22 process launches (0 when refused before any solve, and in --trial); pickle.load/loads blocked;
single-instance lock; the repo-root campaign lock and any other running p5*.py python process refuse the launch; the
output directory must not exist. Stdout and stderr are captured by the caller (attached run).

Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_central_ms_x0.py --spec SPEC --out-dir DIR
       [--trial]   (--trial: zero solves; skips the committed-clean and spec-name checks; phase A only; a spec whose
                    block_choice.result is null is accepted and the computed choice is printed)
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=((SCRIPT_REL, '_solve_one'),),
                          label='P5.15 S56 step 3 centralized comparator x0').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (this stage reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
from centralized_coordination import combine_networks  # noqa: E402
from helper_functions import solver_result_succeeded  # noqa: E402
from model_construction_helpers import pg_avail_init  # noqa: E402
from network import _get_branch_power_flow, _compute_objective_function_value_per_scenario  # noqa: E402

CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
DECL_FILES = ('network.py', 'model_construction_helpers.py')
CHECKS = []
GiB = 1024.0 ** 3


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 step3] {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok else ''))
    return bool(ok)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def committed_clean(rel):
    return bool(_git('ls-files', '--', rel).strip()) and not _git('status', '--porcelain', '--', rel).strip()


def rss():
    """(python process peak RSS, largest waited-for child peak RSS) in bytes (macOS ru_maxrss is bytes)."""
    k = 1 if sys.platform == 'darwin' else 1024
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * k,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * k)


def other_project_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found = os.getpid(), []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(
                re.match(r'p5\S*\.py$', os.path.basename(t)) for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


class _Stages:
    @contextmanager
    def stage(self, name):
        t0 = time.time()
        _log(f'stage: {name}')
        yield
        _log(f'stage done ({time.time() - t0:.1f} s): {name}')


# ----------------------------------------------------------------------------------------------- scenario positions
def declared_positions():
    """{component name: [declaration...]} for every `model.NAME = pe.Var|pe.Constraint|pe.Expression(args)` of the
    two model-building modules, with the positional-argument index of model.scenarios_market / scenarios_operation."""
    out = {}
    for f in DECL_FILES:
        with open(f, encoding='utf-8') as h:
            tree = ast.parse(h.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call)):
                continue
            t = node.targets[0]
            if not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'model'):
                continue
            fn = ast.unparse(node.value.func)
            if fn not in ('pe.Var', 'pe.Constraint', 'pe.Expression'):
                continue
            args = [ast.unparse(a) for a in node.value.args]
            out.setdefault(t.attr, []).append({
                'kind': fn, 'file': f, 'line': node.lineno, 'nargs': len(args),
                'pm': [i for i, a in enumerate(args) if a == 'model.scenarios_market'],
                'po': [i for i, a in enumerate(args) if a == 'model.scenarios_operation']})
    return out


def component_positions(m, decl, ctype):
    """{name: (pm, po)} for every active component of ctype with entries; (None, None) for a scalar component;
    'unknown' when the declaration cannot be resolved unambiguously."""
    res = {}
    for comp in m.component_objects(ctype, active=True):
        if not len(comp):
            continue
        if not comp.is_indexed():
            res[comp.name] = (None, None)
            continue
        cands = [d for d in decl.get(comp.name, []) if d['nargs'] == comp.dim()]
        pos = {(tuple(d['pm']), tuple(d['po'])) for d in cands}
        if len(pos) != 1:
            res[comp.name] = 'unknown'
            continue
        pm, po = pos.pop()
        res[comp.name] = ((pm[0], po[0]) if (len(pm) == 1 and len(po) == 1) else
                          ((None, None) if (not pm and not po) else 'unknown'))
    return res


def _tup(idx):
    return idx if isinstance(idx, tuple) else (idx,)


def _strip(idx, pos):
    """index without its scenario positions (the key shared by merged and single models)."""
    pm, po = pos
    return tuple(x for i, x in enumerate(_tup(idx)) if i not in (pm, po))


def _with_scen(key, pos, s_m, s_o):
    """inverse of _strip: re-insert (s_m, s_o) at positions pos."""
    pm, po = pos
    out, it = [], iter(key)
    for i in range(len(key) + 2):
        out.append(s_m if i == pm else (s_o if i == po else next(it)))
    return tuple(out) if len(out) > 1 else out[0]


def var_labels(m, vpos):
    lab = {}
    for comp in m.component_objects(pe.Var, active=True):
        pos = vpos.get(comp.name)
        if pos is None or pos == 'unknown' or pos == (None, None):
            continue
        for idx, v in comp.items():
            t = _tup(idx)
            lab[id(v)] = (t[pos[0]], t[pos[1]])
    return lab


def coupling_check(m, vpos, cpos):
    """No constraint couples scenarios: each active constraint's free Vars carry one scenario label, which equals the
    constraint's own scenario index."""
    unknown_vars = sorted(n for n, p in vpos.items() if p == 'unknown')
    common_vars = sorted(n for n, p in vpos.items() if p == (None, None))
    unknown_cons = sorted(n for n, p in cpos.items() if p == 'unknown')
    lab = var_labels(m, vpos)
    r = {'n_constraints': 0, 'single_label': 0, 'no_free_vars': 0, 'multi_label': 0, 'label_ne_index': 0,
         'unlabelled_var_in_constraint': 0, 'examples_multi': [], 'unknown_var_components': unknown_vars,
         'scenario_common_var_components': common_vars, 'unknown_constraint_components': unknown_cons,
         'n_labelled_vars': len(lab)}
    for comp in m.component_objects(pe.Constraint, active=True):
        pos = cpos.get(comp.name)
        for idx, c in comp.items():
            if not c.active:
                continue
            r['n_constraints'] += 1
            labels, unl = set(), 0
            for v in identify_variables(c.body, include_fixed=False):
                lb = lab.get(id(v))
                if lb is None:
                    unl += 1
                else:
                    labels.add(lb)
            r['unlabelled_var_in_constraint'] += 1 if unl else 0
            if not labels:
                r['no_free_vars'] += 1
            elif len(labels) == 1:
                r['single_label'] += 1
                if pos not in (None, 'unknown', (None, None)):
                    t = _tup(idx)
                    if (t[pos[0]], t[pos[1]]) != next(iter(labels)):
                        r['label_ne_index'] += 1
            else:
                r['multi_label'] += 1
                if len(r['examples_multi']) < 5:
                    r['examples_multi'].append([c.name, sorted(labels)])
    r['ok'] = (not unknown_vars and not common_vars and not unknown_cons and r['multi_label'] == 0
               and r['label_ne_index'] == 0 and r['unlabelled_var_in_constraint'] == 0 and r['n_constraints'] > 0)
    return r


# ------------------------------------------------------------------------------------------------ network slicing
def _slice_obj(obj, n_o, T, s_o, path, rec, depth=0):
    if depth > 3 or not hasattr(obj, '__dict__'):
        return
    for k, v in list(vars(obj).items()):
        if isinstance(v, np.ndarray):
            if v.ndim == 2 and v.shape == (n_o, T):
                setattr(obj, k, v[s_o:s_o + 1].copy())
                rec.add(f'{path}.{k}')
        elif hasattr(v, '__dict__') and not isinstance(v, (str, bytes, int, float)):
            _slice_obj(v, n_o, T, s_o, f'{path}.{k}', rec, depth + 1)


def slice_network(md, s_m, s_o):
    n_m, n_o, T = len(md.prob_market_scenarios), len(md.prob_operation_scenarios), md.num_instants
    nd = deepcopy(md)
    nd.prob_market_scenarios = [1.0]
    nd.prob_operation_scenarios = [1.0]
    for k in ('cost_energy_p', 'cost_flex'):
        a = np.asarray(getattr(md, k))
        assert a.shape == (n_m, T), (k, a.shape)
        setattr(nd, k, a[s_m:s_m + 1].copy())
    nd.num_oper_scenarios = 1
    rec = set()
    for kind in ('loads', 'generators', 'nodes', 'branches', 'energy_storages'):
        for el in getattr(nd, kind):
            _slice_obj(el, n_o, T, s_o, kind, rec)
    unhandled = sorted(k for k, v in vars(md).items()
                       if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] in (n_m, n_o)
                       and k not in ('cost_energy_p', 'cost_flex'))
    return nd, {'element_arrays_sliced': sorted(rec), 'network_arrays_sliced': ['cost_energy_p', 'cost_flex'],
                'network_level_unhandled_scenario_shaped_arrays': unhandled}


# ------------------------------------------------------------------------------------------------ model preparation
def configure(params, spec):
    p = deepcopy(params)
    p.solver_params.options = dict(p.solver_params.options or {})
    p.solver_params.options.update(spec['solver']['option_overrides'])
    p.solver_params.recovery_enabled = False
    p.solver_params.recovery_tier2_enabled = False
    return p


def fix_loads(m, nd):
    n = 0
    for c in m.loads:
        load = nd.loads[c]
        for s_m in m.scenarios_market:
            for s_o in m.scenarios_operation:
                for p in m.periods:
                    m.pc[c, s_m, s_o, p].fix(load.pd[s_o][p])
                    m.qc[c, s_m, s_o, p].fix(load.qd[s_o][p])
                    n += 2
    return n


def align(m, spec):
    rep = {}
    for name, val in spec['objective_alignment']['set'].items():
        prm = getattr(m, name)
        before = float(pe.value(prm))
        prm.set_value(val)
        rep[name] = [before, float(pe.value(prm))]
    return rep


def formulation_record(m):
    present = {n: hasattr(m, n) for n in ('row18_deviation_charge', 'row18_dev_p_def', 'scenario_voltage_pin',
                                          'expected_interface_pf_p', 'expected_interface_vmag')}
    return {'present': present, 'interface_settlement_weight': float(pe.value(m.interface_settlement_weight)),
            'interface_settlement_value': float(pe.value(m.interface_settlement)),
            'n_adn_nodes': len(list(m.adn_nodes)) if hasattr(m, 'adn_nodes') else None,
            'n_shared_energy_storages': len(list(m.shared_energy_storages)),
            'n_energy_storages': len(list(m.energy_storages)),
            'scalar_mutable_params': {p.name: float(pe.value(p)) for p in m.component_objects(pe.Param, active=True)
                                      if not p.is_indexed() and p.mutable}}


def build(nd, params, spec):
    t0 = time.perf_counter()
    m = nd.build_model(params)
    rec = {'build_s': time.perf_counter() - t0, 'n_load_values_fixed': fix_loads(m, nd), 'alignment': align(m, spec)}
    rec['formulation'] = formulation_record(m)
    f = rec['formulation']
    rec['alpha0_nopin_settlement0_ok'] = (not any(f['present'].values()) and f['interface_settlement_weight'] == 0.0
                                          and f['interface_settlement_value'] == 0.0 and f['n_adn_nodes'] == 0
                                          and f['n_shared_energy_storages'] == 0)
    return m, rec


# ------------------------------------------------------------------------------------------------ slice-identity gate
SCEN_EXPRS = ('gen_cost_scenario', 'flex_cost_scenario', 'load_curt_cost_scenario', 'gen_curt_penalty_scenario',
              'ess_utilization_cost_penalty_scenario', 'slack_penalties_scenario', 'ess_complementarity_penalty_scenario')


def random_point(m, seed, box):
    rng = np.random.default_rng(seed)
    n = 0
    for v in m.component_data_objects(pe.Var, active=True):
        if v.fixed:
            continue
        lo = -box if v.lb is None else max(v.lb, -box)
        hi = box if v.ub is None else min(v.ub, box)
        if hi < lo:
            val = v.lb if v.lb is not None else v.ub
        else:
            val = float(rng.uniform(lo, hi))
        v.set_value(val, skip_validation=True)
        n += 1
    return n


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def slice_identity_gate(md, params, spec, mg, vpos, cpos, tol):
    """mg: the merged gate model, already at a random point. Builds each single model, maps the point, compares."""
    probs_m, probs_o = md.prob_market_scenarios, md.prob_operation_scenarios
    # merged constraints grouped by scenario: label -> {(name, key): condata}
    groups, unlabelled = {}, []
    for comp in mg.component_objects(pe.Constraint, active=True):
        if not len(comp):
            continue
        pos = cpos.get(comp.name)
        if pos in (None, 'unknown', (None, None)):
            unlabelled.append(comp.name)
            continue
        for idx, c in comp.items():
            if c.active:
                t = _tup(idx)
                groups.setdefault((t[pos[0]], t[pos[1]]), {})[(comp.name, _strip(idx, pos))] = c
    out = {'scenarios': {}, 'tol_rel': tol, 'unlabelled_merged_constraint_components': unlabelled}
    obj_sum = 0.0
    ok = not unlabelled
    for s_m in range(len(probs_m)):
        for s_o in range(len(probs_o)):
            nd, srec = slice_network(md, s_m, s_o)
            ms, brec = build(nd, params, spec)
            r = {'slice': srec, 'build_s': brec['build_s'], 'alignment': brec['alignment'],
                 'alpha0_nopin_settlement0_ok': brec['alpha0_nopin_settlement0_ok']}
            # map variable values / compare bounds and fixings
            nv, vb_mis, unmapped = 0, [], 0
            for comp in ms.component_objects(pe.Var, active=True):
                pos = vpos.get(comp.name)
                mc = getattr(mg, comp.name)
                for idx, v in comp.items():
                    if pos in (None, 'unknown', (None, None)):
                        unmapped += 1
                        continue
                    midx = _with_scen(_strip(idx, pos), pos, s_m, s_o)
                    mv = mc[midx]
                    if (mv.lb, mv.ub, mv.fixed) != (v.lb, v.ub, v.fixed) or (v.fixed and mv.value != v.value):
                        if len(vb_mis) < 5:
                            vb_mis.append([comp.name, str(idx), [mv.lb, mv.ub, mv.fixed], [v.lb, v.ub, v.fixed]])
                        r.setdefault('n_var_bound_mismatch', 0)
                        r['n_var_bound_mismatch'] = r.get('n_var_bound_mismatch', 0) + 1
                    if not v.fixed:
                        v.set_value(mv.value, skip_validation=True)
                    nv += 1
            r['n_vars_mapped'] = nv
            r['n_vars_unmapped'] = unmapped
            r['var_bound_mismatch_examples'] = vb_mis
            r.setdefault('n_var_bound_mismatch', 0)
            # constraints
            g = groups.get((s_m, s_o), {})
            seen, n_c, max_rel, n_exact, mism, missing = set(), 0, 0.0, 0, [], 0
            for comp in ms.component_objects(pe.Constraint, active=True):
                if not len(comp):
                    continue
                pos = cpos.get(comp.name)
                if pos in (None, 'unknown', (None, None)):
                    missing += len(comp)
                    continue
                for idx, c in comp.items():
                    if not c.active:
                        continue
                    key = (comp.name, _strip(idx, pos))
                    mc = g.get(key)
                    if mc is None:
                        missing += 1
                        continue
                    seen.add(key)
                    n_c += 1
                    bs, bm = pe.value(c.body), pe.value(mc.body)
                    lo_s = pe.value(c.lower) if c.has_lb() else None
                    lo_m = pe.value(mc.lower) if mc.has_lb() else None
                    up_s = pe.value(c.upper) if c.has_ub() else None
                    up_m = pe.value(mc.upper) if mc.has_ub() else None
                    d = _rel(bs, bm)
                    max_rel = max(max_rel, d)
                    n_exact += 1 if bs == bm else 0
                    if d > tol or lo_s != lo_m or up_s != up_m or c.equality != mc.equality:
                        if len(mism) < 5:
                            mism.append([key[0], str(idx), bs, bm, lo_s, lo_m, up_s, up_m])
            r['constraints'] = {'n_single': n_c + missing, 'n_matched': n_c, 'n_missing_in_merged': missing,
                                'n_merged_unmatched': len(g) - len(seen), 'max_rel_body_diff': max_rel,
                                'n_body_bitwise_equal': n_exact, 'mismatch_examples': mism}
            # per-scenario objective expressions and the single objective
            ex = {}
            for e in SCEN_EXPRS:
                if hasattr(ms, e):
                    a, b = float(pe.value(getattr(ms, e)[0, 0])), float(pe.value(getattr(mg, e)[s_m, s_o]))
                    ex[e] = {'single': a, 'merged': b, 'rel': _rel(a, b)}
            r['scenario_expressions'] = ex
            obj_s = float(pe.value(ms.objective))
            r['single_objective'] = obj_s
            obj_sum += probs_m[s_m] * probs_o[s_o] * obj_s
            r['ok'] = (r['n_var_bound_mismatch'] == 0 and unmapped == 0 and missing == 0
                       and r['constraints']['n_merged_unmatched'] == 0 and not mism and n_c > 0
                       and all(x['rel'] <= tol for x in ex.values()) and r['alpha0_nopin_settlement0_ok']
                       and not srec['network_level_unhandled_scenario_shaped_arrays'])
            ok &= r['ok']
            out['scenarios'][f'{s_m}{s_o}'] = r
            del ms, nd
    obj_m = float(pe.value(mg.objective))
    out['merged_objective'] = obj_m
    out['sum_omega_single_objective'] = obj_sum
    out['objective_rel'] = _rel(obj_m, obj_sum)
    out['ok'] = ok and out['objective_rel'] <= tol
    return out


# ------------------------------------------------------------------------------------------------ block choice
def block_choice(planning, params):
    """Zero-solve proxy (frozen in the spec): per block and scenario-hour, system surplus
    S = (TN RES available + DN RES available) - (TN non-ADN load + DN load), MW, at demand (flexibility and losses
    ignored). RES available = production pg_avail_init over curtaillable generators whose status[p] is on (pg_bounds
    sets pg to 0 otherwise; DN reference generators excluded). Conventional P_min = 0 on every TN generator, so S > 0 means some RES must be curtailed."""
    tn, dns = planning.transmission_network, planning.distribution_networks
    table = {}
    for y in tn.years:
        for d in tn.days:
            t = tn.network[y][d]
            n_m, n_o, T = len(t.prob_market_scenarios), len(t.prob_operation_scenarios), t.num_instants
            R_tn = np.zeros((n_o, T))
            L_tn = np.zeros((n_o, T))
            R_dn = np.zeros((n_o, T))
            L_dn = np.zeros((n_o, T))
            for s_o in range(n_o):
                for p in range(T):
                    R_tn[s_o, p] = sum(pg_avail_init(None, g, s_o, p, t, params) for g in range(len(t.generators))
                                       if t.generators[g].is_curtaillable() and t.generators[g].status[p]) * t.baseMVA
                    L_tn[s_o, p] = sum(ld.pd[s_o][p] for ld in t.loads
                                       if ld.bus not in t.active_distribution_network_nodes) * t.baseMVA
                    for dn in dns:
                        n = dns[dn].network[y][d]
                        ref = n.get_reference_gen_idx()
                        R_dn[s_o, p] += sum(pg_avail_init(None, g, s_o, p, n, params) for g in range(len(n.generators))
                                            if g != ref and n.generators[g].is_curtaillable()
                                            and n.generators[g].status[p]) * n.baseMVA
                        L_dn[s_o, p] += sum(ld.pd[s_o][p] for ld in n.loads) * n.baseMVA
            Sx = R_tn + R_dn - L_tn - L_dn
            w_o = np.asarray(t.prob_operation_scenarios)
            table[f'{y}|{d}'] = {
                'year': int(y), 'day': d,
                'surplus_scenario_hours': int(n_m * (Sx > 0).sum()),
                'surplus_operation_scenario_hours': int((Sx > 0).sum()),
                'weighted_positive_surplus_mwh': float((w_o[:, None] * np.maximum(Sx, 0.0)).sum()),
                'max_surplus_mw': float(Sx.max()),
                'R_tn_mw': R_tn.tolist(), 'R_dn_mw': R_dn.tolist(), 'L_tn_mw': L_tn.tolist(), 'L_dn_mw': L_dn.tolist()}
    order = {y: i for i, y in enumerate(tn.years)}
    dorder = {d: i for i, d in enumerate(tn.days)}
    keys = list(table)

    def tb(k):
        return (order[table[k]['year']] if table[k]['year'] in order else order[str(table[k]['year'])],
                dorder[table[k]['day']])
    fewest = min(keys, key=lambda k: (table[k]['surplus_scenario_hours'], table[k]['weighted_positive_surplus_mwh'],
                                      table[k]['max_surplus_mw'], tb(k)))
    most = min(keys, key=lambda k: (-table[k]['surplus_scenario_hours'], -table[k]['weighted_positive_surplus_mwh'],
                                    -table[k]['max_surplus_mw'], tb(k)))
    summary = {k: {kk: v[kk] for kk in ('surplus_scenario_hours', 'surplus_operation_scenario_hours',
                                         'weighted_positive_surplus_mwh', 'max_surplus_mw')} for k, v in table.items()}
    return {'fewest': fewest, 'most': most, 'summary': summary}, table


# ------------------------------------------------------------------------------------------------ results capture
def max_violation(m, top=5):
    viol = []
    for c in m.component_data_objects(pe.Constraint, active=True):
        body = pe.value(c.body, exception=False)
        if body is None:
            continue
        lo = pe.value(c.lower) if c.has_lb() else None
        up = pe.value(c.upper) if c.has_ub() else None
        v = max((lo - body) if lo is not None else 0.0, (body - up) if up is not None else 0.0, 0.0)
        if v > 0.0:
            viol.append((v, c.name))
    viol.sort(reverse=True)
    return {'max': viol[0][0] if viol else 0.0, 'top': viol[:top],
            'n_violated_gt_1e-6': sum(1 for v, _n in viol if v > 1e-6)}


def costs(m, nd, params):
    """Totals (the model's own Expressions) and, per scenario, the production per-scenario objective with its parts:
    generation split TN_ / ADN_ generators (generation_cost's own condition: controllable; the merged network is a
    transmission network), flexibility split P (flex_p_down) / Q (flex_q_down)."""
    out = {k: float(pe.value(getattr(m, k))) for k in ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost',
                                                        'total_gen_curt_penalty', 'total_slack_penalties',
                                                        'total_ess_utilization_cost_penalty',
                                                        'total_ess_complementarity_penalties')}
    out['C_gen_plus_flex'] = out['total_gen_cost'] + out['total_flex_cost']
    out['objective'] = float(pe.value(m.objective))
    base = nd.baseMVA
    per = {}
    for s_m in m.scenarios_market:
        c_p, c_f = nd.cost_energy_p[s_m], nd.cost_flex[s_m]
        for s_o in m.scenarios_operation:
            g_tn = g_adn = 0.0
            for g in m.generators:
                gen = nd.generators[g]
                if gen.is_controllable():
                    v = sum(c_p[p] * base * pe.value(m.pg[g, s_m, s_o, p]) for p in m.periods)
                    if str(gen.gen_id).startswith('TN_'):
                        g_tn += v
                    else:
                        g_adn += v
            f_p = f_q = 0.0
            if params.fl_reg:
                for c in m.loads:
                    if nd.loads[c].fl_reg:
                        for p in m.periods:
                            f_p += c_f[p] * base * pe.value(m.flex_p_down[c, s_m, s_o, p])
                            f_q += c_f[p] * base * pe.value(m.flex_q_down[c, s_m, s_o, p])
            gen_s = float(pe.value(m.gen_cost_scenario[s_m, s_o]))
            flex_s = float(pe.value(m.flex_cost_scenario[s_m, s_o]))
            per[f'{s_m}{s_o}'] = {
                'probability': nd.prob_market_scenarios[s_m] * nd.prob_operation_scenarios[s_o],
                'gen_cost': gen_s, 'gen_cost_TN_generators': g_tn, 'gen_cost_ADN_generators': g_adn,
                'flex_cost': flex_s, 'flex_cost_P': f_p, 'flex_cost_Q': f_q,
                'C_gen_plus_flex': gen_s + flex_s,
                'slack_penalties': float(pe.value(m.slack_penalties_scenario[s_m, s_o])),
                'objective_per_scenario': _compute_objective_function_value_per_scenario(nd, m, params, s_m, s_o),
                'split_check_rel': max(_rel(gen_s, g_tn + g_adn), _rel(flex_s, f_p + f_q))}
    out['per_scenario'] = per
    return out


def curtailment(m, nd, params, thr_mw):
    base = nd.baseMVA
    out = {}
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            tn_h, dn_h = [], []
            for p in m.periods:
                a = {'TN': 0.0, 'ADN': 0.0}
                for g in m.generators:
                    gen = nd.generators[g]
                    if not gen.is_curtaillable():
                        continue
                    avail = (pe.value(m.pg_avail[g, s_o, p]) if params.rg_curt
                             else pg_avail_init(None, g, s_o, p, nd, params))
                    a['TN' if str(gen.gen_id).startswith('TN_') else 'ADN'] += (avail - pe.value(m.pg[g, s_m, s_o, p])) * base
                tn_h.append(a['TN'])
                dn_h.append(a['ADN'])
            out[f'{s_m}{s_o}'] = {'TN_mwh': sum(tn_h), 'ADN_mwh': sum(dn_h),
                                  'TN_hours_gt_thr': sum(1 for x in tn_h if x > thr_mw),
                                  'ADN_hours_gt_thr': sum(1 for x in dn_h if x > thr_mw),
                                  'TN_hourly_mw': tn_h, 'ADN_hourly_mw': dn_h}
    out['threshold_mw'] = thr_mw
    out['TN_scenario_hours_gt_thr'] = sum(v['TN_hours_gt_thr'] for k, v in out.items() if isinstance(v, dict))
    return out


def interface(m, nd, params, dn_ids):
    out = {}
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            sc = {}
            for dn in dn_ids:
                tbus = f'TN_{dn}'
                node_idx = nd.get_node_idx(tbus)
                brs = [b for b in nd.branches if (b.fbus == tbus or b.tbus == tbus)
                       and str(b.branch_id).startswith(f'ADN_{dn}_')]
                P, Q, V = [], [], []
                for p in m.periods:
                    pp = qq = 0.0
                    for b in brs:
                        other = b.tbus if b.fbus == tbus else b.fbus
                        pij, qij = _get_branch_power_flow(nd, params, b, tbus, other, m, s_m, s_o, p)
                        pp += pij
                        qq += qij
                    P.append(pp)
                    Q.append(qq)
                    V.append(math.sqrt(pe.value(m.e[node_idx, s_m, s_o, p]) ** 2
                                       + pe.value(m.f[node_idx, s_m, s_o, p]) ** 2))
                sc[str(dn)] = {'branches': [str(b.branch_id) for b in brs], 'P_mw': P, 'Q_mvar': Q, 'V_pu': V}
            out[f'{s_m}{s_o}'] = sc
    return out


def block_weight(tn, y, d):
    """The (year, day) block's weight in production's planning sum, as NetworkData.get_primal_value applies it:
    years[y] x days[d] x 1 / (1 + discount_factor) ** (y - first year). Reported beside C_central (which is
    EUR per representative day); not applied to it."""
    years = list(tn.years)
    ann = 1 / ((1 + tn.discount_factor) ** (int(y) - int(years[0])))
    return {'years_represented': tn.years[y], 'days_represented': tn.days[d], 'discount_factor': tn.discount_factor,
            'first_year': int(years[0]), 'annualization': ann, 'weight': tn.years[y] * tn.days[d] * ann,
            'formula': 'years[y] * days[d] / (1 + discount_factor) ** (y - first_year)  (NetworkData.get_primal_value)'}


def relaxation_limits(tn, dns, y, d):
    """Zero-solve. Limits the coordinated (ADMM) models carry and the merged model does not (the S55 central check's
    `known_relaxation`, extended): per DN, the dropped DN reference generator's P/Q limits (gen.pmin/pmax/qmin/qmax x
    baseMVA; pg_bounds / qg_bounds add EQUALITY_TOLERANCE), the DN reference node's voltage limits (the merge joins the
    DN branches to the TN bus, whose limits are kept), and the interface rating R (Network.get_interface_branch_rating,
    MVA) that bounds the TSO's per-scenario interface_delta copies to +-R about its fixed pc in the ADMM path."""
    t = tn.network[y][d]
    out = {}
    for dn in dns:
        n = dns[dn].network[y][d]
        g = n.generators[n.get_reference_gen_idx()]
        ref = n.nodes[n.get_node_idx(n.get_reference_node_id())]
        tbus = t.nodes[t.get_node_idx(dn)]
        out[str(dn)] = {'ref_gen_p_mw': [g.pmin * n.baseMVA, g.pmax * n.baseMVA],
                        'ref_gen_q_mvar': [g.qmin * n.baseMVA, g.qmax * n.baseMVA],
                        'dn_ref_node_v_pu': [ref.v_min, ref.v_max], 'tn_bus_v_pu': [tbus.v_min, tbus.v_max],
                        'interface_rating_mva': n.get_interface_branch_rating()}
    return out


def relaxation_check(itf, limits):
    """Capture only: per DN, scenario-hours of the merged interface outside each limit of `relaxation_limits`, and the
    smallest margin (negative = outside)."""
    out = {}
    for dn, lim in limits.items():
        r = {'p_outside': 0, 'q_outside': 0, 'v_outside_dn_ref_limits': 0, 'abs_p_above_rating': 0,
             'min_margin_p_mw': math.inf, 'min_margin_q_mvar': math.inf, 'min_margin_v_dn_ref_pu': math.inf,
             'min_margin_rating_mw': math.inf}
        for sc in itf.values():
            x = sc[dn]
            for P, Q, V in zip(x['P_mw'], x['Q_mvar'], x['V_pu']):
                mp = min(P - lim['ref_gen_p_mw'][0], lim['ref_gen_p_mw'][1] - P)
                mq = min(Q - lim['ref_gen_q_mvar'][0], lim['ref_gen_q_mvar'][1] - Q)
                mv = min(V - lim['dn_ref_node_v_pu'][0], lim['dn_ref_node_v_pu'][1] - V)
                mr = lim['interface_rating_mva'] - abs(P)
                r['p_outside'] += mp < 0
                r['q_outside'] += mq < 0
                r['v_outside_dn_ref_limits'] += mv < 0
                r['abs_p_above_rating'] += mr < 0
                r['min_margin_p_mw'] = min(r['min_margin_p_mw'], mp)
                r['min_margin_q_mvar'] = min(r['min_margin_q_mvar'], mq)
                r['min_margin_v_dn_ref_pu'] = min(r['min_margin_v_dn_ref_pu'], mv)
                r['min_margin_rating_mw'] = min(r['min_margin_rating_mw'], mr)
        out[dn] = r
    return out


_SUMMARY_RE = {k: re.compile(rf'^{k}\.+:\s+(\S+)\s+(\S+)', re.M) for k in (
    'Objective', 'Dual infeasibility', 'Constraint violation', 'Complementarity', 'Overall NLP error')}


def ipopt_summary(rec):
    """the final (scaled, unscaled) summary rows of THIS attempt's byte range of its IPOPT output file."""
    try:
        with open(rec['log_path'], 'r', errors='replace') as h:
            h.seek(rec['log_bytes'][0])
            text = h.read(rec['log_bytes'][1] - rec['log_bytes'][0])
        out = {}
        for k, rx in _SUMMARY_RE.items():
            mm = rx.findall(text)
            if mm:
                out[k] = [float(mm[-1][0]), float(mm[-1][1])]
        return out
    except Exception as e:  # noqa: BLE001 -- capture only
        return {'error': f'{type(e).__name__}: {e}'}


class _RssSampler:
    """Capture only: samples, every `period_s`, the RSS of this process and of its descendants (the IPOPT child)
    while one solve runs; peak values are SAMPLED (a peak shorter than the period can be missed). The exact running
    maximum over all waited-for children (ru_maxrss) is recorded beside it by `_solve_one`."""

    def __init__(self, period_s=0.1):
        self.period_s, self.peak_children, self.peak_self, self.n = period_s, 0, 0, 0
        self._stop = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                kids = sum(c.memory_info().rss for c in self._proc.children(recursive=True))
                self.peak_children = max(self.peak_children, kids)
                self.peak_self = max(self.peak_self, self._proc.memory_info().rss)
                self.n += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        return False


def _solve_one(nd, m, params):
    """The ONE permitted solve site (SolveProfileGuard). Production Network.run_smopf, from the model's current values.
    A solution is loaded into the model iff production's `solver_result_succeeded(result)` (network._run_smopf loads
    only then, and returns None when the load itself fails); a failed solve returns a non-None result WITHOUT a
    loaded solution, so `result is not None` is not the loaded test (the S56 G harness defect)."""
    n_rec = len(getattr(nd, 'ipopt_solve_records', []) or [])
    child_before = rss()[1]
    t0 = time.perf_counter()
    with _RssSampler() as smp:
        result = nd.run_smopf(m, params, from_warm_start=False, print_header=True)
    wall = time.perf_counter() - t0
    recs = list(getattr(nd, 'ipopt_solve_records', []) or [])[n_rec:]
    self_rss, child_rss = rss()
    return {'wall_s': wall, 'ipopt_records': recs, 'ipopt_final_summary': [ipopt_summary(r) for r in recs],
            'termination_condition': str(result.solver.termination_condition) if result is not None else None,
            'solver_status': str(result.solver.status) if result is not None else None,
            'solution_loaded': bool(result is not None and solver_result_succeeded(result)),
            'peak_rss_self_bytes_after': self_rss, 'peak_rss_children_bytes_after': child_rss,
            'peak_rss_children_running_max_rose': child_rss > child_before,
            'sampled_peak_rss_ipopt_child_bytes': smp.peak_children, 'sampled_peak_rss_self_bytes': smp.peak_self,
            'rss_samples': smp.n, 'rss_sample_period_s': smp.period_s}


def optimal(st):
    return bool(st['solution_loaded'] and st['termination_condition'] == 'optimal' and st['ipopt_records']
                and st['ipopt_records'][-1].get('exit') == 'Optimal Solution Found.')


def capture_solution(m, nd, params, spec, dn_ids, limits):
    """capture only; a capture defect is RECORDED (never raised) so that it cannot abort the declared solve set."""
    out = {}
    for k, fn in (('costs', lambda: costs(m, nd, params)), ('max_violation_after', lambda: max_violation(m)),
                  ('curtailment', lambda: curtailment(m, nd, params, spec['reported']['curtailment_threshold_mw'])),
                  ('interface', lambda: interface(m, nd, params, dn_ids)),
                  ('relaxation', lambda: relaxation_check(out['interface'], limits))):
        try:
            out[k] = fn()
        except Exception:  # noqa: BLE001
            out.setdefault('capture_errors', {})[k] = traceback.format_exc()
    return out


def warm_map(mw, singles, vpos):
    """merged free Var <- single (s_m, s_o) Var at index with (0, 0) in the scenario positions."""
    r = {'set': 0, 'skipped_fixed': 0, 'unmapped': 0, 'none_value': 0, 'outside_bounds': 0}
    for comp in mw.component_objects(pe.Var, active=True):
        pos = vpos.get(comp.name)
        for idx, v in comp.items():
            if v.fixed:
                r['skipped_fixed'] += 1
                continue
            if pos in (None, 'unknown', (None, None)):
                r['unmapped'] += 1
                continue
            t = _tup(idx)
            src = singles.get((t[pos[0]], t[pos[1]]))
            if src is None:
                r['unmapped'] += 1
                continue
            sv = getattr(src, comp.name)[_with_scen(_strip(idx, pos), pos, 0, 0)]
            if sv.value is None:
                r['none_value'] += 1
                continue
            val = sv.value
            if (v.lb is not None and val < v.lb) or (v.ub is not None and val > v.ub):
                r['outside_bounds'] += 1
            v.set_value(val, skip_validation=True)
            r['set'] += 1
    return r


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    a = ap.parse_args()
    pre = []
    spec_rel = os.path.relpath(os.path.abspath(a.spec), REPO)
    spec_sha = _sha(a.spec)
    tag = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not a.trial:
        if not spec_sha.startswith(tag):
            pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
        for rel in (SCRIPT_REL, spec_rel):
            if not committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    if os.path.exists(CAMPAIGN_LOCK):
        pre.append(f'campaign lock {CAMPAIGN_LOCK} exists')
    others = other_project_processes()
    if others:
        pre.append(f'other project processes running: {others}')
    lock = os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), '.central_ms_x0.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[S56 step3] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        return 1
    try:
        return run(a, spec_rel, spec_sha)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def run(a, spec_rel, spec_sha):
    t_start = time.perf_counter()
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    inst = spec['instance']
    expected_solves = 0 if a.trial else spec['guards']['expected_solves']
    os.makedirs(a.out_dir)
    res = {'stage': spec['stage'], 'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'utc_start': datetime.now(timezone.utc).isoformat(), 'trial': a.trial, 'interpreter': sys.executable,
           'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'objective_convention': spec['objective_convention'],
           'instance': {'label': inst['label'], 'case_path': inst['case_path'], 'case_sha256': inst['case_sha256'],
                        'scenario_checksum': inst['scenario_checksum'], 'plan': inst['plan']}}
    # capture checklist, asserted before anything is read (CLAUDE.md rule eleven)
    for name in ('costs', 'curtailment', 'interface', 'max_violation', 'ipopt_summary', '_solve_one', 'warm_map',
                 'coupling_check', 'slice_identity_gate', 'block_choice', 'rss', 'relaxation_limits',
                 'relaxation_check', 'block_weight', 'evaluate'):
        assert callable(globals().get(name)), name
    for k in spec['reported']['required']:
        assert k in ('status', 'iterations', 'wall_time', 'memory', 'C_gen_plus_flex', 'slack_terms', 'objective',
                     'per_scenario_costs', 'interface_per_scenario', 'curtailment_per_scenario',
                     'ipopt_final_summary', 'block_weight', 'relaxation_margins', 'instance_checksum'), k
    # ---- read the instance (production reader), checksums
    _ck('instance case sha256 == declaration', _sha(inst['case_path']) == inst['case_sha256'])
    planning = S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                                 os.path.join(a.out_dir, 'instance_read'), _Stages())
    ck = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
    _ck('combined scenario checksum == declaration', ck == inst['scenario_checksum'], ck)
    res['instance']['scenario_checksum_read'] = ck
    tn, dns = planning.transmission_network, planning.distribution_networks
    dn_ids = list(dns)
    struct_ok, struct_bad = True, []
    for y in tn.years:
        for d in tn.days:
            t = tn.network[y][d]
            for dn in dns:
                n = dns[dn].network[y][d]
                same = (list(n.prob_market_scenarios) == list(t.prob_market_scenarios)
                        and list(n.prob_operation_scenarios) == list(t.prob_operation_scenarios)
                        and np.array_equal(np.asarray(n.cost_energy_p), np.asarray(t.cost_energy_p))
                        and np.array_equal(np.asarray(n.cost_flex), np.asarray(t.cost_flex)))
                if not same:
                    struct_ok = False
                    struct_bad.append(f'{y}|{d}|{dn}')
    _ck('probabilities, energy prices and flexibility prices identical across TN and every DN, every block '
        '(the merged model prices DN flexibility with the TN copy)', struct_ok, struct_bad)
    merged = combine_networks(tn, dns)
    params = configure(merged.params, spec)
    res['solver_options_in_force'] = dict(params.solver_params.options)
    res['recovery'] = [params.solver_params.recovery_enabled, params.solver_params.recovery_tier2_enabled]
    # production's _run_smopf calls exit() after a failed solve when solver_params.verbose is set: the record of the
    # declared solve set would be lost, so the run refuses unless it is off
    _ck('solver_params.verbose is False (a failed solve cannot exit() the process)',
        params.solver_params.verbose is False, params.solver_params.verbose)
    # the block-choice proxy's premise: every TN conventional (non-curtaillable) generator has P_min = 0
    pmin = {f'{y}|{d}': [g.pmin * tn.network[y][d].baseMVA for g in tn.network[y][d].generators
                         if not g.is_curtaillable()] for y in tn.years for d in tn.days}
    res['tn_conventional_pmin_mw'] = pmin
    _ck('block-choice premise: every TN conventional generator has P_min = 0 in every block',
        all(v == 0.0 for vals in pmin.values() for v in vals), pmin)
    # ---- block choice (zero solve), against the spec
    choice, table = block_choice(planning, params)
    res['block_choice'] = {'computed': choice, 'table': table}
    sc = spec['block_choice'].get('result')
    if sc is None:
        _ck('block choice: spec result present (trial accepts null)', a.trial, 'null in spec')
        _log(f"computed block choice: fewest {choice['fewest']}, most {choice['most']}")
    else:
        _ck('block choice recomputed == spec', sc == {'fewest': choice['fewest'], 'most': choice['most'],
                                                      'summary': choice['summary']},
            {'spec': {k: sc.get(k) for k in ('fewest', 'most')}, 'computed': [choice['fewest'], choice['most']]})
    blocks = [choice['fewest'], choice['most']]
    res['blocks'] = blocks
    # ---- phase A: gates per block (zero solves)
    decl = declared_positions()
    gate_tol = spec['pre_solve_gates']['slice_identity_rel_tol']
    res['phase_A'] = {}
    for bk in blocks:
        y_s, d = bk.split('|')
        y = next(yy for yy in tn.years if str(yy) == y_s)
        md = merged.network[y][d]
        mg, brec = build(md, params, spec)
        vpos = component_positions(mg, decl, pe.Var)
        cpos = component_positions(mg, decl, pe.Constraint)
        ra = {'build': brec, 'n_vars': sum(1 for _ in mg.component_data_objects(pe.Var, active=True)),
              'n_constraints': sum(1 for _ in mg.component_data_objects(pe.Constraint, active=True)),
              'var_positions': {k: list(v) if isinstance(v, tuple) else v for k, v in vpos.items()},
              'constraint_positions': {k: list(v) if isinstance(v, tuple) else v for k, v in cpos.items()},
              'n_scenarios': [len(md.prob_market_scenarios), len(md.prob_operation_scenarios)],
              'probabilities': [list(md.prob_market_scenarios), list(md.prob_operation_scenarios)]}
        _ck(f'{bk}: alpha = 0 (no row 18), no voltage pin, settlement weight 0 and value 0, no ADN / shared ESS',
            brec['alpha0_nopin_settlement0_ok'], brec['formulation'])
        _ck(f'{bk}: 3 market x 3 operation scenarios', ra['n_scenarios'] == [3, 3], ra['n_scenarios'])
        ra['block_weight'] = block_weight(tn, y, d)
        ra['relaxation_limits'] = relaxation_limits(tn, dns, y, d)
        t0 = time.perf_counter()
        ra['coupling'] = coupling_check(mg, vpos, cpos)
        ra['coupling']['wall_s'] = time.perf_counter() - t0
        _ck(f'{bk}: no constraint couples scenarios (every active constraint carries one scenario label = its index; '
            f'no scenario-common or unresolvable Var/Constraint component)', ra['coupling']['ok'],
            {k: v for k, v in ra['coupling'].items() if k != 'examples_multi'})
        ra['random_point_n_vars'] = random_point(mg, spec['pre_solve_gates']['random_point_seed'],
                                                 spec['pre_solve_gates']['random_point_box'])
        t0 = time.perf_counter()
        ra['slice_identity'] = slice_identity_gate(md, params, spec, mg, vpos, cpos, gate_tol)
        ra['slice_identity']['wall_s'] = time.perf_counter() - t0
        _ck(f'{bk}: slice identity at a random point (9 single models == the merged model restricted to each '
            f'scenario; merged objective == sum omega single objective, rel <= {gate_tol})',
            ra['slice_identity']['ok'], {'objective_rel': ra['slice_identity']['objective_rel'],
                                         'bad': [k for k, v in ra['slice_identity']['scenarios'].items()
                                                 if not v['ok']]})
        res['phase_A'][bk] = ra
        del mg
    # ---- code identity: every repo module imported is committed clean (non-trial: refusal)
    mods = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f).startswith(REPO + os.sep) and f.endswith('.py'):
            rel = os.path.relpath(os.path.abspath(f), REPO)
            mods[rel] = {'sha256': _sha(rel), 'committed_clean': committed_clean(rel)}
    res['imported_repo_modules'] = mods
    res['git_status_tracked'] = _git('status', '--porcelain', '-uno')
    res['git_diff_sha256'] = hashlib.sha256(_git('diff').encode()).hexdigest()
    dirty = sorted(k for k, v in mods.items() if not v['committed_clean'])
    if not a.trial:
        _ck('every imported repo module is committed clean', not dirty, dirty)
    else:
        res['trial_dirty_imported_modules'] = dirty
    failed = [c['check'] for c in CHECKS if not c['ok']]
    res['refused_before_any_solve'] = failed
    res['phase_A_peak_rss_bytes'] = list(rss())
    # ---- phase B: the solves
    res['solves'] = {}
    if failed:
        _log(f'REFUSED before any solve: {failed}')
    try:
        phase_b(res, a, spec, failed, blocks, tn, merged, params, decl, dn_ids)
    except Exception:  # noqa: BLE001 -- recorded; the results file and the guard verification are still written
        res['phase_B_exception'] = traceback.format_exc()
        _log('phase B exception:\n' + res['phase_B_exception'])
    res['total_wall_s'] = time.perf_counter() - t_start
    res['peak_rss_bytes_end'] = list(rss())
    # guards verified BEFORE the predictions (P1 reads them)
    gv = GUARD.verify(0 if (failed or a.trial) else expected_solves)
    res['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                    'expected': 0 if (failed or a.trial) else expected_solves, 'verify_failures': gv}
    res['pickle'] = dict(PK)
    res['guards_ok'] = (not gv) and PK == {'load': 0, 'loads': 0}
    res['checks'] = CHECKS
    res['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    # ---- predictions
    if not failed and not a.trial:
        try:
            res['predictions'] = evaluate(res, spec, blocks)
        except Exception:  # noqa: BLE001 -- recorded; the results file is still written
            res['predictions_error'] = traceback.format_exc()
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(a.out_dir, 'central_ms_x0_results.json')
    with open(path, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(a.out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, a.out_dir)] = _sha(p)
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump({'outputs': outputs, 'spec': {'path': spec_rel, 'sha256': spec_sha},
                   'script': {'path': SCRIPT_REL, 'sha256': res['script_sha256']},
                   'instance': {'path': inst['case_path'], 'sha256': _sha(inst['case_path']),
                                'scenario_checksum': res['instance'].get('scenario_checksum_read')},
                   'imported_repo_modules': {k: v['sha256'] for k, v in mods.items()}}, h, indent=1)
    _log(f"wrote {path}; guards_ok {res['guards_ok']}; failed checks {res['failed_checks']}; "
         f"predictions {({k: v.get('held') for k, v in res.get('predictions', {}).items()})}")
    if failed:
        return 3
    return 0 if res['guards_ok'] else 2


def phase_b(res, a, spec, failed, blocks, tn, merged, params, decl, dn_ids):
    for bk in ([] if (failed or a.trial) else blocks):
        y_s, d = bk.split('|')
        y = next(yy for yy in tn.years if str(yy) == y_s)
        md = merged.network[y][d]
        b = {}
        res['solves'][bk] = b
        logs = os.path.join(a.out_dir, 'logs', bk.replace('|', '_'))
        lim = res['phase_A'][bk]['relaxation_limits']
        # merged, default start
        m, brec = build(md, params, spec)
        vpos = component_positions(m, decl, pe.Var)
        md.logs_dir = os.path.join(logs, 'merged_default')
        os.makedirs(md.logs_dir)
        st = _solve_one(md, m, params)
        b['merged_default'] = {'build': brec, 'solve': st, 'optimal': optimal(st)}
        if st['solution_loaded']:
            b['merged_default'].update(capture_solution(m, md, params, spec, dn_ids, lim))
        _log(f"{bk} merged default: {st['termination_condition']} "
             f"{(st['ipopt_records'] or [{}])[-1].get('exit')} it {(st['ipopt_records'] or [{}])[-1].get('iterations')} "
             f"wall {st['wall_s']:.1f}s obj {b['merged_default'].get('costs', {}).get('objective')}")
        del m
        # singles
        singles, b['single'] = {}, {}
        for s_m in range(len(md.prob_market_scenarios)):
            for s_o in range(len(md.prob_operation_scenarios)):
                nd, srec = slice_network(md, s_m, s_o)
                ms, brec = build(nd, params, spec)
                nd.logs_dir = os.path.join(logs, f'single_{s_m}{s_o}')
                os.makedirs(nd.logs_dir)
                st = _solve_one(nd, ms, params)
                r = {'slice': srec, 'build': brec, 'solve': st, 'optimal': optimal(st),
                     'probability': md.prob_market_scenarios[s_m] * md.prob_operation_scenarios[s_o]}
                if st['solution_loaded']:
                    r.update(capture_solution(ms, nd, params, spec, dn_ids, lim))
                b['single'][f'{s_m}{s_o}'] = r
                singles[(s_m, s_o)] = ms
                _log(f"{bk} single {s_m}{s_o}: {st['termination_condition']} "
                     f"{(st['ipopt_records'] or [{}])[-1].get('exit')} wall {st['wall_s']:.1f}s "
                     f"obj {r.get('costs', {}).get('objective')}")
        # merged, warm start from the 9 single solutions
        mw, brec = build(md, params, spec)
        wrep = warm_map(mw, singles, vpos)
        try:  # capture only: a None-valued Var must not abort the declared solve set
            at = {'objective': float(pe.value(mw.objective)),
                  'C_gen_plus_flex': float(pe.value(mw.total_gen_cost) + pe.value(mw.total_flex_cost)),
                  'max_violation': max_violation(mw)}
        except Exception:  # noqa: BLE001
            at = {'capture_error': traceback.format_exc()}
        del singles
        md.logs_dir = os.path.join(logs, 'merged_warm')
        os.makedirs(md.logs_dir)
        st = _solve_one(md, mw, params)
        b['merged_warm'] = {'build': brec, 'warm_mapping': wrep, 'at_mapped_point': at, 'solve': st,
                            'optimal': optimal(st)}
        if st['solution_loaded']:
            b['merged_warm'].update(capture_solution(mw, md, params, spec, dn_ids, lim))
        _log(f"{bk} merged warm: {st['termination_condition']} {(st['ipopt_records'] or [{}])[-1].get('exit')} "
             f"wall {st['wall_s']:.1f}s obj {b['merged_warm'].get('costs', {}).get('objective')}")
        del mw


def evaluate(res, spec, blocks):
    """The frozen predictions P1-P5 (spec `predictions`, thresholds in `predictions_thresholds`). Objectives are the
    production `model.objective` values (settlement weight 0, no row 18, no pin: the plan's gross convention, slack
    terms included, no salvage), EUR per representative day, probability-weighted over the block's 9 scenarios."""
    th = spec['predictions_thresholds']
    pr = {}
    # P1: phase A passed; guards exact (solves and launches)
    g = res['guard']
    pr['P1'] = {'statement': spec['predictions']['P1'], 'phase_A_failed_checks': res['refused_before_any_solve'],
                'guard_counts': g['counts'], 'guard_verify_failures': g['verify_failures'], 'pickle': res['pickle'],
                'held': (not res['refused_before_any_solve']) and res['guards_ok']
                and g['counts']['permitted_solve'] == spec['guards']['expected_solves']
                and g['counts']['permitted_exec'] == spec['guards']['expected_solves']}
    # P2: every solve Optimal
    not_opt, n = [], 0
    for bk in blocks:
        b = res['solves'].get(bk, {})
        runs = [('merged_default', b.get('merged_default')), ('merged_warm', b.get('merged_warm'))] + \
               [(f'single_{k}', v) for k, v in b.get('single', {}).items()]
        for name, r in runs:
            n += 1
            if not (r and r['optimal']):
                not_opt.append(f'{bk}:{name}')
    pr['P2'] = {'statement': spec['predictions']['P2'], 'n_solves_seen': n, 'not_optimal': not_opt,
                'held': (not not_opt) and n == spec['guards']['expected_solves']}
    # P3 / P4: merged (warm / default start) objective against sum_s omega_s C_s
    p3, p4, p5 = {}, {}, {}
    for bk in blocks:
        b = res['solves'].get(bk, {})
        sing = b.get('single', {})
        md_, mw_ = b.get('merged_default') or {}, b.get('merged_warm') or {}
        have_s = len(sing) == 9 and all('costs' in v for v in sing.values())
        so = sum(v['probability'] * v['costs']['objective'] for v in sing.values()) if have_s else None
        sc = sum(v['probability'] * v['costs']['C_gen_plus_flex'] for v in sing.values()) if have_s else None
        ss = sum(v['probability'] * v['costs']['total_slack_penalties'] for v in sing.values()) if have_s else None
        for key, store, tol in (('merged_warm', p3, th['P3_rel']), ('merged_default', p4, th['P4_rel'])):
            r = b.get(key) or {}
            if have_s and 'costs' in r:
                mo = r['costs']['objective']
                rel = abs(mo - so) / abs(so) if so != 0.0 else (0.0 if mo == so else math.inf)
                store[bk] = {'merged_objective': mo, 'sum_omega_single_objective': so, 'diff_eur': mo - so,
                             'rel': rel, 'tol_rel': tol, 'held': rel <= tol,
                             'merged_C_gen_plus_flex': r['costs']['C_gen_plus_flex'],
                             'sum_omega_single_C_gen_plus_flex': sc}
            else:
                store[bk] = {'held': False, 'reason': 'a solution was not loaded (singles or merged)'}
        if not p4[bk]['held'] and 'merged_objective' in p4[bk]:
            p4[bk]['reading'] = ('local-optimum difference (declared before launch; not a stop): default-start '
                                 'merged objective and sum_s omega_s C_s both recorded')
        # P5: the central cost in the plan's convention, with the block weight
        w = res['phase_A'][bk]['block_weight']
        p5[bk] = {'C_central_eur_per_representative_day': so,
                  'C_central_definition': 'sum_s omega_s C_s, C_s = single-scenario model.objective',
                  'C_gen_plus_flex_sum_omega': sc, 'slack_terms_sum_omega': ss,
                  'merged_default_objective': md_.get('costs', {}).get('objective'),
                  'merged_warm_objective': mw_.get('costs', {}).get('objective'),
                  'block_weight': w['weight'], 'block_weight_record': w,
                  'C_central_times_block_weight_eur': (so * w['weight']) if so is not None else None,
                  'convention': spec['objective_convention'],
                  'instance_scenario_checksum': res['instance']['scenario_checksum']}
        p5[bk]['complete'] = so is not None and p5[bk]['merged_default_objective'] is not None \
            and p5[bk]['merged_warm_objective'] is not None
    pr['P3'] = {'statement': spec['predictions']['P3'], 'per_block': p3, 'held': all(v['held'] for v in p3.values())}
    pr['P4'] = {'statement': spec['predictions']['P4'], 'per_block': p4, 'held': all(v['held'] for v in p4.values()),
                'failure_is_a_stop': False}
    ck_ok = (res['instance']['scenario_checksum'] == spec['instance']['scenario_checksum']
             and res['instance'].get('scenario_checksum_read') == spec['instance']['scenario_checksum'])
    pr['P5'] = {'statement': spec['predictions']['P5'], 'per_block': p5,
                'instance_checksum_in': ['spec.instance', 'results.instance (read and declared)',
                                         'results.predictions.P5.per_block', 'manifest_sha256.json.instance'],
                'held': all(v['complete'] for v in p5.values()) and ck_ok}
    return pr


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
