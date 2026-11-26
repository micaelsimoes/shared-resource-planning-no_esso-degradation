"""P5.15 S56 -- D0 v2: TN realized-exchange re-solve, shared ESS as a fixed injection. Exactly 80 IPOPT solves.

Predecessor: `p515_s56_tn_realized_resolve.py` (spec v1 5d35b048, run_v1, post-check postcheck_v1). v1 fixed the unit
cell's shared-ESS net P/Q (node 7) at the TSO copy's terminal schedule while keeping the shared-ESS INTERNAL rows active;
that schedule violates its own rows by ~1e-8 p.u. (SoC limits) and up to 1.0008 (capability ratio), leaving no interior,
and 11/40 unit solves failed (postcheck_v1 `unit_failure_diagnosis`). v1 also read a failed solve as loaded
(`result is not None`).

v2 change (Planner ruling, a diagnostic configuration choice, not a production change). In the TN-only model of BOTH arms
of BOTH cells the shared ESS enters only as a FIXED INJECTION:
  * the variables through which it enters the TN node balance -- the scenario-free alias copies
    `shared_es_pnet/qnet[e, s_m0, s_o0, p]` (`mch.sess_na_scenario`, read by `mch.compute_node_load`) -- are fixed at
    the saved TSO terminal schedule (exactly as v1, via v1's `build_arm`);
  * EVERY shared-ESS internal row is deactivated for every shared ESS of the TSO block: production's own list
    `mch._SHARED_ESS_OPERATIONAL_CONSTRAINTS` (pnet definition, pch/pdch hat links, converter capability, active-sum
    limit, SoC dynamics, SoC upper/lower limits, day closure, complementarity) -- these are the only rows besides
    node_balance_p/q that reference any shared-ESS variable (gated per block, below);
  * a fixed injection lying outside the bounds of its own variable has that bound alone relaxed (recorded per block).
The day-balance slacks `slack_shared_es_soc_final_up/down` then appear in no active row; they stay in the objective
(production weight), bounded [0, ub], and are reported separately. The ESS's own feasibility is NOT re-checked here (it
belongs to the ESSO). Everything else is v1: production build, candidate application, ADN setup, objective, solver,
options, extraction, statistics formulas. In x0 every shared ESS is zero-capacity, its rows are already deactivated by
production (`configure_shared_ess_operational_state`), so the change is inert there (prediction P2).

Solve success is production's own test, applied inside the run: `helper_functions.solver_result_succeeded(result)` on
the result `Network.run_smopf` returns (production `_run_smopf` loads a solution exactly when that test holds, and
returns None when loading fails). Nothing is extracted from a solve that test rejects.

Frozen spec: data/SRP1/Results/P515S56/tn_realized_resolve/frozen_tn_realized_spec_v2_<sha8>.json (--spec; its sha256
must start with the <sha8> in its name; it and this script must be committed clean unless --trial).

GUARDS. This script's SolveProfileGuard(permitted=[(this file, '_solve_one')]) is installed before any project import.
v1 and the v1 post-check are then imported (for their functions); each installs its own guard on top at import, and both
are uninstalled immediately after the imports, in reverse order, which restores this script's guard (asserted by
identity). Verified at exactly 80 solves and 80 executions (0 under --trial or a pre-solve refusal), with the v1 and
post-check guards' counts all zero. pickle.load/loads blocked for the whole run (no pickle is read). Single-instance lock
(the same lock file as v1); refuses if the campaign lock exists or a campaign / concurrency-gate / D0 process runs; the
output directory must not exist; stdout and stderr captured by the caller (attached run).

Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_tn_realized_resolve_v2.py --spec SPEC \\
           --out-dir DIR [--trial]
       (--trial: skips the committed-clean checks and the solves -- reads, builds and gates only)
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from pyomo.opt.base.solvers import OptSolver  # noqa: E402
from pyomo.opt.solver.shellcmd import SystemCallSolver  # noqa: E402

GUARD = SolveProfileGuard(permitted=((SCRIPT_REL, '_solve_one'),),
                          label='P5.15 S56 D0 v2 TN realized re-solve (shared ESS fixed injection)').install()
_V2_SOLVE, _V2_EXEC = OptSolver.solve, SystemCallSolver._execute_command

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (D0 v2 reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import p515_s56_tn_realized_resolve as H  # noqa: E402  (installs H.GUARD on top: permits only v1's _solve_one)
import p515_s56_tn_realized_postcheck as PCK  # noqa: E402  (installs PCK.GUARD on top: permits nothing)

PCK.GUARD.uninstall()
H.GUARD.uninstall()
assert OptSolver.solve is _V2_SOLVE and SystemCallSolver._execute_command is _V2_EXEC, \
    'v2 guard not restored after uninstalling the imported v1 / post-check guards'
pickle.load, pickle.loads = _blk('load'), _blk('loads')   # the imports replaced the blockers: re-arm this script's

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402
from pyomo.repn import generate_standard_repn  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import helper_functions as HF  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

CHECKS = H.CHECKS            # one check list: v1's helpers (load_cell, block_data, model_gates) append to it
CAMPAIGN_LOCK = H.CAMPAIGN_LOCK
CONCURRENT_SCRIPTS = H.CONCURRENT_SCRIPTS + (SCRIPT_REL, 'p515_s56_tn_realized_postcheck.py')
SESS_PREFIXES = ('shared_es_', 'slack_shared_es_')
SESS_SLACKS = ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down')
GRID_SLACKS = tuple(n for n in H.SLACK_VARS if n not in SESS_SLACKS)


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [D0v2] {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))
    return bool(ok)


def concurrent_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(
                os.path.basename(t) in CONCURRENT_SCRIPTS for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


# ---------------------------------------------------------------------------------------- the v2 change (zero solves)
def isolate_shared_ess(m, net, bd, dn_ids, tag):
    """Deactivate every shared-ESS internal row (production's list), keep the fixed scenario-free pnet/qnet injection
    (fixed by v1's build_arm), relax only a bound the fixed injection violates; gate the resulting structure."""
    s_m0, s_o0 = mch.sess_na_scenario(m)
    rows = {}
    for cn in mch._SHARED_ESS_OPERATIONAL_CONSTRAINTS:
        comp = getattr(m, cn)
        rows[cn] = {'n_rows': len(comp), 'n_active_before': sum(1 for c in comp.values() if c.active)}
        for c in comp.values():
            c.deactivate()
    relaxed, fixed_ok, sched = [], True, {}
    for dn in dn_ids:
        e = net.get_shared_energy_storage_idx(dn)
        s_r = pe.value(m.shared_es_s_rated_fixed[e])
        pn, qn = [], []
        for p in m.periods:
            for var in (m.shared_es_pnet[e, s_m0, s_o0, p], m.shared_es_qnet[e, s_m0, s_o0, p]):
                fixed_ok &= bool(var.fixed)
                x = var.value
                if var.lb is not None and x < var.lb:
                    relaxed.append({'var': var.name, 'value': x, 'bound': 'lb', 'was': var.lb})
                    var.setlb(None)
                if var.ub is not None and x > var.ub:
                    relaxed.append({'var': var.name, 'value': x, 'bound': 'ub', 'was': var.ub})
                    var.setub(None)
            pn.append(m.shared_es_pnet[e, s_m0, s_o0, p].value)
            qn.append(m.shared_es_qnet[e, s_m0, s_o0, p].value)
        dt = mch.period_duration_hours(m)
        sched[str(dn)] = {
            'e_idx': e, 's_rated_pu': s_r, 'e_rated_pu': pe.value(m.shared_es_e_rated_fixed[e]),
            'max_abs_pnet_mw': max(abs(x) for x in pn) * net.baseMVA, 'max_abs_qnet_mvar': max(abs(x) for x in qn) * net.baseMVA,
            'net_energy_mwh': sum(pn) * dt * net.baseMVA,
            'charge_mwh': sum(max(x, 0.0) for x in pn) * dt * net.baseMVA,
            'discharge_mwh': sum(max(-x, 0.0) for x in pn) * dt * net.baseMVA,
            'capability_ratio_max': (max((a * a + b * b) ** 0.5 for a, b in zip(pn, qn)) / s_r) if s_r else None}
    # structural gate: the only active rows referencing any shared-ESS variable are node_balance_p/q, through fixed
    # pnet/qnet at the scenario-free copy
    refs, unfixed_in_balance, non_alias = {}, [], []
    for c in m.component_data_objects(pe.Constraint, active=True):
        vs = [v for v in identify_variables(c.body, include_fixed=True)
              if v.parent_component().name.startswith(SESS_PREFIXES)]
        if not vs:
            continue
        refs.setdefault(c.parent_component().name, set()).update(v.parent_component().name for v in vs)
        for v in vs:
            if not v.fixed:
                unfixed_in_balance.append(v.name)
            if v.parent_component().name not in ('shared_es_pnet', 'shared_es_qnet') or \
                    tuple(v.index()[1:3]) != (s_m0, s_o0):
                non_alias.append(v.name)
    rp = generate_standard_repn(m.objective.expr, compute_values=True)
    obj_sess = sorted({v.name for v, cf in zip(rp.linear_vars, rp.linear_coefs)
                       if cf != 0 and v.parent_component().name.startswith(SESS_PREFIXES)}
                      | {v.name for pair in (rp.quadratic_vars or []) for v in pair
                         if v.parent_component().name.startswith(SESS_PREFIXES)}
                      | {v.name for v in (rp.nonlinear_vars or []) if v.parent_component().name.startswith(SESS_PREFIXES)})
    obj_sess_unexpected = [n for n in obj_sess if not n.startswith(SESS_SLACKS)]
    slack_bounds = {}
    for name in SESS_SLACKS:
        comp = getattr(m, name, None)
        if comp is None:
            continue
        for dn in dn_ids:
            v = comp[net.get_shared_energy_storage_idx(dn), s_m0, s_o0]
            slack_bounds[f'{name}[{dn}]'] = {'fixed': bool(v.fixed), 'lb': v.lb, 'ub': v.ub}
    g = {'rows_deactivated': rows,
         'any_sess_row_active_after': any(c.active for cn in mch._SHARED_ESS_OPERATIONAL_CONSTRAINTS
                                          for c in getattr(m, cn).values()),
         'active_rows_referencing_shared_ess_vars': {k: sorted(v) for k, v in refs.items()},
         'unfixed_shared_ess_vars_in_active_rows': unfixed_in_balance[:10],
         'non_alias_shared_ess_vars_in_active_rows': non_alias[:10],
         'injection_fixed': fixed_ok, 'bounds_relaxed': relaxed,
         'objective_shared_ess_vars_nonzero_coef': obj_sess, 'objective_shared_ess_unexpected': obj_sess_unexpected,
         'day_balance_slack_bounds': slack_bounds, 'fixed_schedule': sched}
    ok = (not g['any_sess_row_active_after'] and set(refs) <= {'node_balance_p', 'node_balance_q'}
          and not unfixed_in_balance and not non_alias and fixed_ok and not obj_sess_unexpected)
    _ck(f'{tag}: shared ESS = fixed injection only (rows deactivated, only node balance references it, fixed)', ok,
        {k: g[k] for k in ('any_sess_row_active_after', 'active_rows_referencing_shared_ess_vars',
                           'unfixed_shared_ess_vars_in_active_rows', 'non_alias_shared_ess_vars_in_active_rows',
                           'injection_fixed', 'objective_shared_ess_unexpected')})
    g['ok'] = ok
    return g


def build_arm_v2(tn, candidate, avail, cell_label, arm, bdata, dn_ids):
    """v1's production build (unchanged), then the v2 shared-ESS isolation on every block."""
    models, recs = H.build_arm(tn, candidate, avail, cell_label, arm, bdata, dn_ids)
    iso = {}
    for year in tn.years:
        for day in tn.days:
            key = (str(int(year)), day)
            iso[key] = isolate_shared_ess(models[year][day], tn.network[year][day], bdata[key], dn_ids,
                                          f'{cell_label} {arm} {key[0]} {day}')
    return models, recs, iso


# --------------------------------------------------------------------------------------------------- solve + extract
def _solve_one(net, m, params):
    """The ONE permitted solve site (SolveProfileGuard). Production Network.run_smopf from the model's own values;
    success = production `solver_result_succeeded` on the returned result (None when production failed to load)."""
    n_rec = len(getattr(net, 'ipopt_solve_records', []) or [])
    t0 = time.perf_counter()
    result = net.run_smopf(m, params, from_warm_start=False, print_header=True)
    wall = time.perf_counter() - t0
    recs = list(getattr(net, 'ipopt_solve_records', []) or [])[n_rec:]
    loaded = bool(HF.solver_result_succeeded(result))
    ex = (recs or [{}])[-1]
    tc = str(result.solver.termination_condition) if result is not None and hasattr(result, 'solver') else None
    return {'wall_s': wall, 'ipopt_records': recs, 'termination_condition': tc,
            'solver_status': str(result.solver.status) if result is not None and hasattr(result, 'solver') else None,
            'result_returned': result is not None, 'solver_result_summary': HF.solver_result_summary(result),
            'loaded_by_production': loaded, 'solution_loaded': loaded,
            'strict_optimal': loaded and tc == 'optimal' and ex.get('exit') == 'Optimal Solution Found.'}


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    a = ap.parse_args()
    t_start = time.perf_counter()
    pre = []
    spec_sha = H._sha(a.spec)
    tag = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not spec_sha.startswith(tag):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
    if not a.trial:
        for rel in (SCRIPT_REL, os.path.relpath(os.path.abspath(a.spec), REPO)):
            if not H.committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    if os.path.exists(CAMPAIGN_LOCK):
        pre.append(f'campaign lock {CAMPAIGN_LOCK} exists')
    others = concurrent_processes()
    if others:
        pre.append(f'concurrent processes: {others}')
    lock = os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), '.tn_realized_resolve.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[D0v2] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        return 1
    try:
        return run(a, spec_sha, t_start)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def run(a, spec_sha, t_start):
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    # capture checklist, asserted before anything is read or solved (CLAUDE.md rule eleven)
    capture = {'C_A': 'H.extract', 'C_B': 'H.extract', 'status_iterations_time': '_solve_one',
               'production_success_test': '_solve_one', 'slack_detector_terms': 'H.extract',
               'max_constraint_violation': 'H.max_violation', 'dso_covariance_per_block': 'H.block_data',
               'D1': 'statistics', 'C_A_vs_saved_tso_cost': 'statistics', 'bus_duals_adn': 'H.extract',
               'curtailment_binding_limits_per_hour': 'H.extract', 'shared_ess_isolation_gate': 'isolate_shared_ess',
               'fixed_schedule_properties': 'isolate_shared_ess', 'x0_vs_v1_reproduction': 'statistics',
               'pair_full_and_v1_common_blocks': 'statistics', 'D1_attribution': 'PCK.block_terms',
               'storage_tn_side_effect': 'statistics', 'predictions': 'predictions'}
    missing = [k for k in spec['reported'] if k not in capture]
    assert not missing, f'spec quantities without a capture path: {missing}'
    for k, fn in capture.items():
        mod, _, name = fn.rpartition('.')
        assert hasattr({'H': H, 'PCK': PCK}[mod], name) if mod else name in globals(), (k, fn)
    os.makedirs(a.out_dir)
    expected_solves = 0 if a.trial else spec['guards']['expected_solves']
    res = {'stage': spec['stage'], 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script': SCRIPT_REL,
           'script_sha256': H._sha(SCRIPT_REL), 'git_head': H._git('rev-parse', 'HEAD').strip(),
           'utc_start': datetime.now(timezone.utc).isoformat(), 'trial': a.trial,
           'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'objective_convention': spec['objective']['reported_C'], 'weighting': spec['statistics']['weighting'],
           'success_test': spec['solver']['success_test'], 'cells': {}}
    modules = {p: H._input_record(p) for p in ('network.py', 'network_data.py', 'model_construction_helpers.py',
                                               'shared_resources_planning.py', 'helper_functions.py',
                                               'p515_s44_scale_measurement.py', 'p515_s56_tn_realized_resolve.py',
                                               'p515_s56_tn_realized_postcheck.py',
                                               spec['instance']['case_path'], H.CAMPAIGN_SPEC)}
    res['modules'] = modules
    v1in = {}
    for k, d in spec['v1_evidence'].items():
        r = H._input_record(d['path'])
        _ck(f'v1 evidence {k} sha256 == spec', r['sha256'] == d['sha256'], r)
        v1in[k] = r
    res['v1_evidence'] = v1in
    inst = spec['instance']
    _ck('instance case sha256 == spec', H._sha(inst['case_path']) == inst['case_sha256'])
    cspec = json.load(open(H.CAMPAIGN_SPEC))
    di = cspec['configuration']['derived_instance']
    _ck('instance == the campaign spec declaration (path, sha256, checksum)',
        di['case_path'] == inst['case_path'] and di['case_sha256'] == inst['case_sha256']
        and di['scenario_checksum'] == inst['scenario_checksum'])
    planning = H.S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}}, a.out_dir, H._Stages())
    checksum = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
    _ck('combined scenario checksum == declaration', checksum == inst['scenario_checksum'], checksum)
    dn_ids = list(planning.transmission_network.active_distribution_network_nodes)
    _ck('TN ADN nodes == [5, 7, 9]', dn_ids == [5, 7, 9], dn_ids)
    _ck('production shared-ESS row list == spec', list(mch._SHARED_ESS_OPERATIONAL_CONSTRAINTS)
        == spec['model']['shared_ess_rows_deactivated'], list(mch._SHARED_ESS_OPERATIONAL_CONSTRAINTS))
    params_solve = H.configure(planning.transmission_network.params, spec)
    res['solver_options_in_force'] = dict(params_solve.solver_params.options)
    res['recovery'] = [params_solve.solver_params.recovery_enabled, params_solve.solver_params.recovery_tier2_enabled]
    _ck('bound_relax_factor not overridden (IPOPT default 1e-8, the P5 threshold)',
        'bound_relax_factor' not in res['solver_options_in_force'])

    # ---- phase A (zero solves): inputs, block data, candidate application, model + isolation gates per (cell, arm)
    ctx = {}
    for cell in spec['cells']:
        lab = cell['label']
        cd = H.load_cell(cell)
        pc = deepcopy(planning)
        tn = pc.transmission_network
        cand = pc.get_initial_candidate_solution()
        iy = H._year_key(pc.shared_ess_data.years, cell['candidate_canonical']['investment_year'])
        for n, (s_val, e_val) in cell['candidate_canonical']['nodes'].items():
            cand['investment'][int(n)][iy]['s'] = float(s_val)
            cand['investment'][int(n)][iy]['e'] = float(e_val)
        srp._rebuild_candidate_total_capacities(pc, cand)
        tn.update_data_with_candidate_solution(cand['total_capacity'])
        avail = {}
        for n in dn_ids:
            pub = cd['er']['storage_per_node'][str(n)]['published_available_capacity_terminal']
            avail[n] = {y: {'s': pub[str(int(y))]['s_available'], 'e': pub[str(int(y))]['e_available']}
                        for y in tn.years}
        bdata = {}
        for year in tn.years:
            for day in tn.days:
                bdata[(str(int(year)), day)] = H.block_data(lab, cd['ms'], tn, year, day, dn_ids)
        cov_total = sum(bd['w'] * bd['cov_block'] for bd in bdata.values())
        saved_dso = cd['ms']['summary']['settlement_identity']['dso_deviation_part']
        _ck(f'{lab}: sum_b w_b cov_b == summary DSO deviation part (rel <= 1e-9)',
            H._rel(cov_total, sum(saved_dso.values())) <= 1e-9, [cov_total, sum(saved_dso.values())])
        arms_gate, arms_iso = {}, {}
        for arm in ('A', 'B'):
            models, recs, iso = build_arm_v2(tn, cand, avail, lab, arm, bdata, dn_ids)
            arms_gate[arm], arms_iso[arm] = recs, iso
            del models
        ctx[lab] = {'cell': cell, 'cd': cd, 'tn': tn, 'cand': cand, 'avail': avail, 'bdata': bdata}
        res['cells'][lab] = {
            'eval_dir': cd['edir'], 'inputs': cd['inputs'], 'candidate_canonical': cell['candidate_canonical'],
            'candidate_total_capacity': {str(n): {str(y): v for y, v in d.items()}
                                         for n, d in cand['total_capacity'].items()},
            'available_capacity_applied': {str(n): {str(y): v for y, v in d.items()} for n, d in avail.items()},
            'dso_covariance_weighted_total': cov_total, 'dso_covariance_saved_summary': saved_dso,
            'certification_bar_eur': cd['er']['bar']['value'],
            'terminal_gross_operational_cost': cd['er']['terminal_gross_operational_cost'],
            'pre_solve_model_gates': {arm: {f'{k[0]}|{k[1]}': v for k, v in r.items()} for arm, r in arms_gate.items()},
            'pre_solve_shared_ess_isolation': {arm: {f'{k[0]}|{k[1]}': v for k, v in r.items()}
                                               for arm, r in arms_iso.items()},
            'blocks': {}}
    failed = [c['check'] for c in CHECKS if not c['ok']]
    if failed:
        res['refused_before_any_solve'] = failed
        _log(f'REFUSED before any solve: {len(failed)} failed checks')
    # ---- phase B: the 80 solves
    for cell in ([] if (failed or a.trial) else spec['cells']):
        lab = cell['label']
        c = ctx[lab]
        tn, bdata = c['tn'], c['bdata']
        for arm in ('A', 'B'):
            models, recs, iso = build_arm_v2(tn, c['cand'], c['avail'], lab, arm, bdata, dn_ids)
            for year in tn.years:
                for day in tn.days:
                    key = (str(int(year)), day)
                    net, m = tn.network[year][day], models[year][day]
                    blk = res['cells'][lab]['blocks'].setdefault(f'{key[0]}|{day}', {})
                    r = {'model_gates_ok': recs[key]['ok'], 'shared_ess_isolation_ok': iso[key]['ok'],
                         'shared_ess_bounds_relaxed': iso[key]['bounds_relaxed']}
                    net.logs_dir = os.path.join(a.out_dir, 'logs', lab, arm)
                    os.makedirs(net.logs_dir, exist_ok=True)
                    st = _solve_one(net, m, params_solve)
                    r['solve'] = st
                    if st['loaded_by_production']:
                        try:
                            r.update(H.extract(m, net, tn.params, bdata[key], dn_ids))
                            s_m0, s_o0 = mch.sess_na_scenario(m)
                            r['shared_ess_day_balance_slack_values'] = {
                                f'{n}[{dn}]': getattr(m, n)[net.get_shared_energy_storage_idx(dn), s_m0, s_o0].value
                                for n in SESS_SLACKS if hasattr(m, n) for dn in dn_ids}
                        except Exception:  # noqa: BLE001 -- a capture failure is recorded, never hides the solve
                            r['extract_error'] = traceback.format_exc()
                            _ck(f'{lab} {arm} {key[0]} {day}: extraction', False, r['extract_error'][-400:])
                    blk[arm] = r
                    ex = (st['ipopt_records'] or [{}])[-1]
                    _log(f"{lab} {arm} {key[0]} {day}: {st['termination_condition']} {ex.get('exit')} "
                         f"loaded {st['loaded_by_production']} it {ex.get('iterations')} wall {st['wall_s']:.1f}s "
                         f"C {r.get('costs', {}).get('C_gen_plus_flex')}")
                    models[year][day] = None
            del models
        for year in tn.years:
            for day in tn.days:
                key = (str(int(year)), day)
                res['cells'][lab]['blocks'][f'{key[0]}|{day}']['inputs_used'] = {
                    'w': bdata[key]['w'], 'omega': bdata[key]['omega'], 'cov_block': bdata[key]['cov_block'],
                    'cov_saved_by_dn': {str(k): v for k, v in bdata[key]['cov_saved'].items()},
                    'cov_recomputed_by_dn': {str(k): v for k, v in bdata[key]['cov_recomputed'].items()},
                    'tso_cost_saved': bdata[key]['tso_cost_saved'],
                    'tso_p_common_mw': {str(k): v for k, v in bdata[key]['tso_p'].items()},
                    'prices_eur_per_mwh_by_hour_by_market': bdata[key]['prices']}
    if not (failed or a.trial):
        try:
            res['statistics'] = statistics(res, ctx, dn_ids, spec)
            res['predictions'] = predictions(res, spec)
        except Exception:  # noqa: BLE001 -- recorded; the per-block records are still written
            res['statistics_error'] = traceback.format_exc()
            _ck('statistics computed', False, res['statistics_error'][-400:])
    res['total_wall_s'] = time.perf_counter() - t_start
    gv = GUARD.verify(expected_solves, expected_solves)
    imported_guard_counts = {'v1': dict(H.GUARD.counts), 'postcheck': dict(PCK.GUARD.counts)}
    imported_zero = all(v == 0 for d in imported_guard_counts.values() for v in d.values())
    res['guard'] = {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                    'permitted_sites': dict(GUARD.permitted_sites), 'expected_solves': expected_solves,
                    'verify_failures': gv, 'imported_guards_uninstalled_after_import': imported_guard_counts,
                    'imported_guards_all_zero': imported_zero}
    res['pickle'] = {'v2': dict(PK), 'v1_module': dict(H.PK), 'postcheck_module': dict(PCK.PK)}
    res['guards_ok'] = (not gv) and imported_zero and all(
        d == {'load': 0, 'loads': 0} for d in res['pickle'].values())
    res['checks'] = CHECKS
    res['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    out = os.path.join(a.out_dir, 'tn_realized_resolve_v2_results.json')
    with open(out, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    manifest = {'outputs': {os.path.basename(out): H._sha(out)},
                'inputs': {**{f"{lab}/{k}": v for lab, cc in res['cells'].items() for k, v in cc['inputs'].items()},
                           **{f'module/{k}': v for k, v in modules.items()},
                           **{f'v1_evidence/{k}': v for k, v in v1in.items()}},
                'spec': {'path': a.spec, 'sha256': spec_sha},
                'logs_dir_files': {os.path.relpath(os.path.join(dp, f), a.out_dir): H._sha(os.path.join(dp, f))
                                   for dp, _d, fs in os.walk(os.path.join(a.out_dir, 'logs')) for f in fs},
                'note': 'the stdout/stderr launch logs are written while the script runs; hashed in the Worker report'}
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump(manifest, h, indent=1)
    _log(f"wrote {out}; guards_ok {res['guards_ok']}; failed checks {len(res['failed_checks'])}")
    if failed:
        return 3
    return 0 if res['guards_ok'] else 2


# ------------------------------------------------------------------------------------------------------- statistics
def _arm_status(r):
    s = (r or {}).get('solve') or {}
    ex = (s.get('ipopt_records') or [{}])[-1]
    return {'loaded_by_production': bool(s.get('loaded_by_production')), 'strict_optimal': bool(s.get('strict_optimal')),
            'termination_condition': s.get('termination_condition'), 'solver_status': s.get('solver_status'),
            'exit': ex.get('exit'), 'iterations': ex.get('iterations'), 'wall_s': s.get('wall_s')}


def _max_of(extremes, names, key):
    vals = [extremes[n][key] for n in names if n in extremes and extremes[n][key] is not None]
    if not vals:
        return None
    return max(vals) if key == 'max' else min(vals)


def statistics(res, ctx, dn_ids, spec):
    """Same formulas as v1 (spec 'statistics'); a block enters D1 only when both arms passed production's success
    test; 'strict' = both arms exit 'Optimal Solution Found.'."""
    out = {'per_cell': {}}
    for lab, cc in res['cells'].items():
        bd_all = ctx[lab]['bdata']
        rows = {}
        for bkey, blk in cc['blocks'].items():
            y, d = bkey.split('|')
            bd = bd_all[(y, d)]
            w = bd['w']
            A, B = blk.get('A', {}), blk.get('B', {})
            sa, sb = _arm_status(A), _arm_status(B)
            V = sum(bd['omega'][k] * sum(bd['prices'][t][int(k.split('_')[0])]
                                         * (bd['dso_p'][dn][k][t] - bd['tso_p'][dn][t]) for t in range(24))
                    for dn in dn_ids for k in bd['omega'])
            row = {'w': w, 'A': sa, 'B': sb, 'cov': bd['cov_block'], 'V_price_valued_draw_change': V,
                   'valid': sa['loaded_by_production'] and sb['loaded_by_production'] and 'costs' in A and 'costs' in B,
                   'strict': sa['strict_optimal'] and sb['strict_optimal'] and 'costs' in A and 'costs' in B}
            for arm, r in (('A', A), ('B', B)):
                if 'costs' in r:
                    row[f'C_{arm}'] = r['costs']['C_gen_plus_flex']
                    row[f'obj_{arm}'] = r['costs']['objective']
                    row[f'max_violation_{arm}'] = r['max_violation']['max']
                    row[f'max_violation_by_component_{arm}'] = r['max_violation']['max_by_component']
                    row[f'detector_{arm}'] = r['detector_components']['detector_penalty_total']
                    row[f'slack_penalties_{arm}'] = r['costs']['total_slack_penalties']
                    row[f'ess_day_balance_penalty_{arm}'] = r['costs']['total_ess_complementarity_penalties']
                    row[f'max_grid_slack_{arm}'] = _max_of(r['slack_var_extremes'], GRID_SLACKS, 'max')
                    row[f'min_grid_slack_{arm}'] = _max_of(r['slack_var_extremes'], GRID_SLACKS, 'min')
                    row[f'max_sess_slack_{arm}'] = _max_of(r['slack_var_extremes'], SESS_SLACKS, 'max')
                    row[f'res_curt_mwh_expected_{arm}'] = sum(h['res_curt_mw'] for k in r['hours']
                                                              for h in r['hours'][k]) / 9.0
            if row['valid']:
                cA, cB = row['C_A'], row['C_B']
                saved_gen = ctx[lab]['cd']['cl']['blocks'][f'TSO|{y}|{d}']['unweighted']['generation_cost']
                row.update({'dC': cB - cA, 'D1': (cB - cA) - bd['cov_block'], 'D1prime': (cB - cA) - V,
                            'saved_tso_generation_cost': saved_gen, 'C_A_minus_saved_gen': cA - saved_gen,
                            'obj_A_minus_saved_local': row['obj_A'] - bd['tso_cost_saved']['block_recourse_production_local']})
                _r, terms = PCK.block_terms(blk, dn_ids)
                row['attribution'] = terms
                _ck(f'{lab} {bkey}: attribution sums to w_b D1_b (abs <= 1e-6 max(1,|x|))',
                    abs(terms['sum'] - w * row['D1']) <= 1e-6 * max(1.0, abs(w * row['D1'])), [terms['sum'], w * row['D1']])
            rows[bkey] = row
        wsum_all = sum(r['w'] for r in rows.values())
        tot = {}
        for variant in ('valid', 'strict'):
            sel = [r for r in rows.values() if r[variant]]
            tot[variant] = {'n_blocks': len(sel), 'weight_share': sum(r['w'] for r in sel) / wsum_all,
                            'blocks': sorted(k for k, r in rows.items() if r[variant]),
                            **{f'{q}_w': sum(r['w'] * r[q] for r in sel)
                               for q in ('C_A', 'C_B', 'dC', 'cov', 'D1', 'D1prime', 'V_price_valued_draw_change',
                                         'C_A_minus_saved_gen', 'obj_A_minus_saved_local')}}
        # attribution by class over valid blocks (post-check formula)
        by_class = {}
        for bkey, blk in cc['blocks'].items():
            if not rows[bkey]['valid']:
                continue
            brows, _t = PCK.block_terms(blk, dn_ids)
            for r in brows:
                k = r['class'] + (' | bound set changed' if r['controllable_bound_set_changed'] else '')
                dd = by_class.setdefault(k, {'n_scenario_hours': 0, 'sum_eur_weighted': 0.0})
                dd['n_scenario_hours'] += 1
                dd['sum_eur_weighted'] += r['contribution_eur_weighted']
        out['per_cell'][lab] = {'blocks': rows, 'totals': tot, 'complete': tot['valid']['n_blocks'] == len(rows),
                                'n_loaded': {arm: sum(1 for r in rows.values() if r[arm]['loaded_by_production'])
                                             for arm in ('A', 'B')},
                                'n_strict': {arm: sum(1 for r in rows.values() if r[arm]['strict_optimal'])
                                             for arm in ('A', 'B')},
                                'cov_w_all_blocks': sum(r['w'] * r['cov'] for r in rows.values()),
                                'attribution_by_class': by_class,
                                'certification_bar_eur': res['cells'][lab]['certification_bar_eur']}
    pcl = out['per_cell']
    v1run = json.load(open(spec['v1_evidence']['run_v1_results']['path']))
    v1pc = json.load(open(spec['v1_evidence']['postcheck_v1']['path']))
    v1_common = v1pc['pair_v2']['valid']['common_blocks']
    _ck('v1 12-block common set == spec', v1_common == spec['statistics']['v1_common_blocks'], v1_common)
    if 'x0' in pcl and 'unit' in pcl:
        x0, un = pcl['x0']['blocks'], pcl['unit']['blocks']
        both_valid = sorted(k for k in x0 if x0[k]['valid'] and un[k]['valid'])
        full = len(both_valid) == len(x0)
        pair = {'full_horizon': {'complete': full, 'n_blocks': len(both_valid),
                                 'D1_x0_w': sum(x0[k]['w'] * x0[k]['D1'] for k in both_valid),
                                 'D1_unit_w': sum(un[k]['w'] * un[k]['D1'] for k in both_valid)}}
        pair['full_horizon']['pair_D1_x0_minus_unit'] = pair['full_horizon']['D1_x0_w'] - pair['full_horizon']['D1_unit_w']
        c12 = [k for k in v1_common if x0[k]['valid'] and un[k]['valid']]
        pair['v1_common_12'] = {'blocks': v1_common, 'n_valid_in_v2': len(c12),
                                'weight_share': sum(x0[k]['w'] for k in v1_common) / sum(r['w'] for r in x0.values()),
                                'D1_x0_w': sum(x0[k]['w'] * x0[k]['D1'] for k in c12),
                                'D1_unit_w': sum(un[k]['w'] * un[k]['D1'] for k in c12),
                                'v1_postcheck_pair_D1_x0_minus_unit': v1pc['pair_v2']['valid']['pair_D1_x0_minus_unit']}
        pair['v1_common_12']['pair_D1_x0_minus_unit'] = pair['v1_common_12']['D1_x0_w'] - pair['v1_common_12']['D1_unit_w']
        pair['per_block_D1_unit_minus_x0'] = {
            k: {'w': x0[k]['w'], 'D1_x0': x0[k]['D1'], 'D1_unit': un[k]['D1'], 'diff': un[k]['D1'] - x0[k]['D1'],
                'w_diff': x0[k]['w'] * (un[k]['D1'] - x0[k]['D1']),
                'dC_x0': x0[k]['dC'], 'dC_unit': un[k]['dC'], 'cov_x0': x0[k]['cov'], 'cov_unit': un[k]['cov']}
            for k in both_valid}
        # the storage's TN-side effect: unit minus x0 TN cost at each arm's fixed exchange (the exchanges differ too)
        pair['storage_tn_side'] = {
            k: {'w': x0[k]['w'], 'C_A_unit_minus_x0': un[k]['C_A'] - x0[k]['C_A'],
                'C_B_unit_minus_x0': un[k]['C_B'] - x0[k]['C_B'],
                'fixed_schedule_node7': res['cells']['unit']['pre_solve_shared_ess_isolation']['A'][k]['fixed_schedule']['7']}
            for k in both_valid}
        pair['storage_tn_side_weighted'] = {
            'C_A_unit_minus_x0_w': sum(x0[k]['w'] * (un[k]['C_A'] - x0[k]['C_A']) for k in both_valid),
            'C_B_unit_minus_x0_w': sum(x0[k]['w'] * (un[k]['C_B'] - x0[k]['C_B']) for k in both_valid)}
        out['pair'] = pair
    # x0 reproduction of v1 (P2) and the unit beside v1 on v1's loaded blocks
    repro = {}
    for lab in pcl:
        rr = {}
        for bkey, row in pcl[lab]['blocks'].items():
            v1b = v1run['cells'][lab]['blocks'][bkey]
            e = {}
            for arm in ('A', 'B'):
                v1s = v1b[arm].get('solve') or {}
                v1_loaded = v1s.get('solver_status') == 'ok' and v1s.get('termination_condition') in PCK.ACCEPTED_TC
                if f'C_{arm}' in row and v1_loaded and 'costs' in v1b[arm]:
                    c1 = v1b[arm]['costs']['C_gen_plus_flex']
                    e[arm] = {'v1': c1, 'v2': row[f'C_{arm}'], 'rel': H._rel(row[f'C_{arm}'], c1),
                              'abs': row[f'C_{arm}'] - c1, 'bitwise': row[f'C_{arm}'] == c1}
                else:
                    e[arm] = {'v1_loaded': v1_loaded, 'v2_has_costs': f'C_{arm}' in row}
            rr[bkey] = e
        repro[lab] = rr
    out['v1_comparison'] = repro
    return out


def predictions(res, spec):
    st = res['statistics']
    pcl = st['per_cell']
    tol2 = spec['predictions']['P2_tolerance_rel_per_block']
    thr5 = spec['predictions']['P5_threshold_pu']
    n_strict = sum(pcl[lab]['n_strict'][arm] for lab in pcl for arm in ('A', 'B'))
    out = {'P1': {'n_strict_optimal': n_strict, 'n_loaded_by_production': sum(pcl[lab]['n_loaded'][arm] for lab in pcl
                                                                              for arm in ('A', 'B')),
                  'held': n_strict == 80}}
    x0r = st['v1_comparison']['x0']
    rels = [x0r[b][arm].get('rel') for b in x0r for arm in ('A', 'B')]
    out['P2'] = {'n_compared': sum(1 for r in rels if r is not None), 'max_rel': max((r for r in rels if r is not None),
                                                                                     default=None),
                 'n_bitwise': sum(1 for b in x0r for arm in ('A', 'B') if x0r[b][arm].get('bitwise')),
                 'tolerance_rel_per_block': tol2,
                 'held': len(rels) == 40 and all(r is not None and r <= tol2 for r in rels)}
    u = pcl['unit']
    lo, hi = spec['predictions']['P3_range_eur']
    d1u = u['totals']['valid']['D1_w']
    out['P3'] = {'unit_D1_w_valid_blocks': d1u, 'complete': u['complete'], 'range_eur': [lo, hi],
                 'held': (lo <= d1u <= hi) if u['complete'] else None,
                 'note': None if u['complete'] else 'indeterminate: unit cell incomplete on the full horizon'}
    fh = st['pair']['full_horizon']
    out['P4'] = {'pair_D1_x0_minus_unit_full': fh['pair_D1_x0_minus_unit'] if fh['complete'] else None,
                 'complete': fh['complete'], 'bound_eur': spec['predictions']['P4_bound_eur'],
                 'held': (abs(fh['pair_D1_x0_minus_unit']) < spec['predictions']['P4_bound_eur']) if fh['complete'] else None,
                 'v1_common_12_pair': st['pair']['v1_common_12']['pair_D1_x0_minus_unit']}
    p5 = {}
    for lab in pcl:
        for bkey, row in pcl[lab]['blocks'].items():
            mx = row.get('max_grid_slack_B')
            p5[f'{lab}|{bkey}'] = {'loaded': row['B']['loaded_by_production'], 'max_grid_slack_B': mx,
                                   'max_sess_slack_B': row.get('max_sess_slack_B'),
                                   'held': row['B']['loaded_by_production'] and mx is not None and mx <= thr5}
    out['P5'] = {'threshold_pu': thr5, 'per_block': p5, 'n_held': sum(1 for v in p5.values() if v['held']),
                 'unit_2028_Spring': p5.get('unit|2028|Spring'),
                 'held': all(v['held'] for v in p5.values()),
                 'max_grid_slack_B_over_all': max((v['max_grid_slack_B'] for v in p5.values()
                                                   if v['max_grid_slack_B'] is not None), default=None),
                 'max_sess_slack_B_over_all': max((v['max_sess_slack_B'] for v in p5.values()
                                                   if v['max_sess_slack_B'] is not None), default=None)}
    return out


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
