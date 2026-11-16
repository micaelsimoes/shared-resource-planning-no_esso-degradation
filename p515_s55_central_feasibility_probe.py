"""P5.15 S55 -- centralised single-day check, STEP 1 (feasibility). ZERO SOLVES: build-only probe.

Question (Expert handback 2026-11-16): can the existing code build ONE merged model (TN + three DNs, interface variables
shared directly) for one representative day without new model development; at what build cost and size; and does the
merged model carry the same cost components and constraints as the coordinated network models, so that its optimum is
comparable with the coordinated Q of the plan without storage?

What it does (production functions only; nothing edited):
  - SolveProfileGuard(permitted=()) installed BEFORE any project import; every solve and solver exec raises; verified at
    exactly 0 at exit. pickle.load/loads blocked.
  - SharedResourcesPlanning('data/SRP1', 'SRP1.json').read_planning_problem()  (x = 0: no candidate is applied; the
    merged network drops shared ESS by construction, combine_networks l. 75).
  - centralized_coordination.combine_networks(TN, DNs) -- the production merge used by
    run_operational_planning(type='centralized').
  - for each 2025 representative day: Network.build_model(params) of the merged day network, of the TSO day network and
    of the three DSO day networks; wall time, peak RSS, variable / constraint / nonzero-free counts.
  - differences that decide comparability, read from the built objects:
      * params: every attribute of the TN params that differs from a DN's params (the merged model uses the TN's);
      * prices: cost_energy_p and cost_flex of the TN vs each DN, per day (the merged model prices DN flexibility and
        DN controllable generation at the TN's arrays);
      * DN generators kept in the merge that are controllable (priced by generation_cost) and their DN-model treatment;
      * voltage limits: each DN reference node (dropped by the merge) vs the TN interface node that replaces it;
      * constraint components present in the TSO or DSO models and absent from the merged model, by component name.

Usage: <canonical python> -u p515_s55_central_feasibility_probe.py --out <new json path>
"""
import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S55 centralised feasibility probe (never solves)').install()

import pickle  # noqa: E402

_PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        _PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import pyomo.environ as pe  # noqa: E402

from centralized_coordination import combine_networks  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

YEAR = 2025
DROP = ('constraint_components', 'rows_by_component', 'vars_by_component')


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == 'darwin' else r / 1024


def counts(model):
    nv = sum(1 for _ in model.component_data_objects(pe.Var, active=True))
    nv_free = sum(1 for v in model.component_data_objects(pe.Var, active=True) if not v.fixed)
    nc = sum(1 for _ in model.component_data_objects(pe.Constraint, active=True))
    comps = sorted(c.name for c in model.component_objects(pe.Constraint, active=True))
    rows = {c.name: sum(1 for _ in c.values()) for c in model.component_objects(pe.Constraint, active=True)}
    vrows = {v.name: sum(1 for _ in v.values()) for v in model.component_objects(pe.Var, active=True)}
    return {'vars': nv, 'vars_free': nv_free, 'constraints': nc, 'constraint_components': comps,
            'rows_by_component': rows, 'vars_by_component': vrows}


def build(network_day, params, label):
    t0 = time.perf_counter()
    m = network_day.build_model(params)
    dt = time.perf_counter() - t0
    c = counts(m)
    c.update({'label': label, 'build_s': dt, 'peak_rss_mb_after': rss_mb()})
    return m, c


def params_diff(pa, pb):
    out = {}
    da, db = vars(pa), vars(pb)
    for k in sorted(set(da) | set(db)):
        va, vb = da.get(k, '<absent>'), db.get(k, '<absent>')
        if hasattr(va, '__dict__') or hasattr(vb, '__dict__'):
            sub = params_diff(va, vb) if hasattr(va, '__dict__') and hasattr(vb, '__dict__') else {'': [repr(va)[:80], repr(vb)[:80]]}
            for kk, vv in sub.items():
                out[f'{k}.{kk}'] = vv
        elif repr(va) != repr(vb):
            out[k] = [repr(va)[:120], repr(vb)[:120]]
    return out


def arr(x):
    return [[float(v) for v in row] for row in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'refusing to overwrite {a.out}')
    t_all = time.perf_counter()
    planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    planning.read_planning_problem()
    tn, dns = planning.transmission_network, planning.distribution_networks
    t0 = time.perf_counter()
    merged = combine_networks(tn, dns)
    merge_s = time.perf_counter() - t0
    res = {'stage': 'P5.15 S55 centralised single-day check, step 1 (feasibility; build only)',
           'utc': datetime.now(timezone.utc).isoformat(), 'merge_s_all_blocks': merge_s,
           'read_and_merge_peak_rss_mb': rss_mb(), 'days': {}, 'params_tn_vs_dn': {}, 'prices': {},
           'dn_generators_kept': {}, 'voltage_limits_interface': {}}
    for dn_id, dn in dns.items():
        res['params_tn_vs_dn'][str(dn_id)] = params_diff(tn.params, dn.params)
    for day in tn.days:
        tnd = tn.network[YEAR][day]
        md = merged.network[YEAR][day]
        mm, cm = build(md, merged.params, f'merged {YEAR} {day}')
        _tm, ct = build(tnd, tn.params, f'TSO {YEAR} {day}')
        cds = {}
        for dn_id, dn in dns.items():
            _dm, cd = build(dn.network[YEAR][day], dn.params, f'DSO {dn_id} {YEAR} {day}')
            cds[str(dn_id)] = cd
        def _delta(key):
            tot = {}
            for c in [ct] + list(cds.values()):
                for k, v in c[key].items():
                    tot[k] = tot.get(k, 0) + v
            return {k: [tot.get(k, 0), cm[key].get(k, 0)] for k in sorted(set(tot) | set(cm[key]))
                    if tot.get(k, 0) != cm[key].get(k, 0)}
        row_delta, var_delta = _delta('rows_by_component'), _delta('vars_by_component')
        union = set(ct['constraint_components']).union(*[set(c['constraint_components']) for c in cds.values()])
        missing = sorted(union - set(cm['constraint_components']))
        extra = sorted(set(cm['constraint_components']) - union)
        res['days'][day] = {
            'merged': {k: v for k, v in cm.items() if k not in DROP},
            'tso': {k: v for k, v in ct.items() if k not in DROP},
            'dso': {k: {kk: vv for kk, vv in v.items() if kk not in DROP} for k, v in cds.items()},
            'sum_tso_dso_vars': ct['vars'] + sum(c['vars'] for c in cds.values()),
            'sum_tso_dso_constraints': ct['constraints'] + sum(c['constraints'] for c in cds.values()),
            'constraint_components_in_tso_or_dso_not_in_merged': missing,
            'constraint_components_in_merged_only': extra,
            'rows_tso_plus_dso_vs_merged_where_different': row_delta,
            'vars_tso_plus_dso_vs_merged_where_different': var_delta,
            'merged_nodes': len(md.nodes), 'merged_branches': len(md.branches),
            'merged_generators': len(md.generators), 'merged_loads': len(md.loads),
            'merged_objective_components_nonzero_capable': sorted(
                n for n in ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost', 'total_gen_curt_penalty',
                            'total_ess_utilization_cost_penalty', 'total_slack_penalties',
                            'total_ess_complementarity_penalties', 'interface_settlement') if hasattr(mm, n))}
        pr = {'tn': {'cost_energy_p': arr(tnd.cost_energy_p), 'cost_flex': arr(tnd.cost_flex),
                     'prob_market': list(map(float, tnd.prob_market_scenarios))}}
        for dn_id, dn in dns.items():
            dd = dn.network[YEAR][day]
            pr[str(dn_id)] = {'cost_energy_p_equal_tn': arr(dd.cost_energy_p) == arr(tnd.cost_energy_p),
                              'cost_flex_equal_tn': arr(dd.cost_flex) == arr(tnd.cost_flex),
                              'baseMVA': dd.baseMVA, 'tn_baseMVA': tnd.baseMVA,
                              'max_abs_cost_flex_diff': max(abs(x - y) for rx, ry in zip(arr(dd.cost_flex), arr(tnd.cost_flex)) for x, y in zip(rx, ry)) if len(dd.cost_flex) else None}
            ref = dd.get_reference_node_id()
            refn = [n for n in dd.nodes if n.bus_i == ref][0]
            tnn = [n for n in tnd.nodes if n.bus_i == dn_id][0]
            res['voltage_limits_interface'].setdefault(str(dn_id), {})[day] = {
                'dn_ref_node': [refn.v_min, refn.v_max], 'tn_interface_node': [tnn.v_min, tnn.v_max]}
            kept = [g for g in dd.generators if g.bus != ref]
            res['dn_generators_kept'].setdefault(str(dn_id), {})[day] = [
                {'gen_id': g.gen_id, 'type': g.gen_type, 'controllable': bool(g.is_controllable()),
                 'pmax': g.pmax, 'pmin': g.pmin} for g in kept]
        res['prices'][day] = pr
        # the merged model's demand band: TN-mode loads are Vars within +/- EQUALITY_TOLERANCE (pu) of demand
        # (model_construction_helpers.pc_bounds), the DSO models hold them as fixed Params. Bound on the cost the band
        # can remove: sum over loads and hours of EQUALITY_TOLERANCE * baseMVA * max(energy price, flexibility price).
        from definitions import EQUALITY_TOLERANCE
        n_loads = len(md.loads)
        ce, cf = arr(tnd.cost_energy_p)[0], arr(tnd.cost_flex)[0]
        res.setdefault('demand_band', {})[day] = {
            'equality_tolerance_pu': EQUALITY_TOLERANCE, 'n_loads_merged': n_loads,
            'mw_per_load_hour': EQUALITY_TOLERANCE * md.baseMVA,
            'bound_eur_at_energy_price': sum(EQUALITY_TOLERANCE * md.baseMVA * n_loads * x for x in ce),
            'bound_eur_at_max_energy_or_flex_price': sum(EQUALITY_TOLERANCE * md.baseMVA * n_loads * max(x, y)
                                                         for x, y in zip(ce, cf))}
    # comparator named by the handback: the coordinated x = 0 evaluation ref:7aa017f0, unweighted per-block levels
    cl_path = ('data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/'
               'd110bd1a5977df1e_x0/component_levels_terminal.json')
    with open(cl_path, encoding='utf-8') as h:
        cl = json.load(h)
    comp = {}
    for b in cl['blocks'].values():
        if str(b['year']) != str(YEAR):
            continue
        d = comp.setdefault(b['day'], {'generation_cost': 0.0, 'flexibility_cost_internal': 0.0,
                                       'detector_penalty_total': 0.0, 'other_q_parts': 0.0})
        u = b['unweighted']
        d['generation_cost'] += u['generation_cost']
        d['flexibility_cost_internal'] += u['flexibility_cost_internal']
        d['detector_penalty_total'] += u['detector_penalty_total']
        d['other_q_parts'] += u['load_curtailment_cost'] + u['res_curtailment_penalty'] + u['ess_usage_cost']
    for d in comp.values():
        d['gen_plus_flex'] = d['generation_cost'] + d['flexibility_cost_internal']
        d['q_day_unweighted'] = d['gen_plus_flex'] + d['detector_penalty_total'] + d['other_q_parts']
    import hashlib
    res['comparator'] = {'path': cl_path, 'sha256': hashlib.sha256(open(cl_path, 'rb').read()).hexdigest(),
                         'cell': 'ref:7aa017f0 (x = 0, W101 d110bd1a, k* 181)', 'days_2025_unweighted_eur': comp}
    res['total_wall_s'] = time.perf_counter() - t_all
    res['peak_rss_mb'] = rss_mb()
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    res['pickle'] = dict(_PK)
    ok = not res['guard']['verify_0_failures'] and _PK == {'load': 0, 'loads': 0}
    res['guards_ok'] = ok
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'x', encoding='utf-8') as h:
        json.dump(res, h, indent=1, default=str)
    print('wrote', a.out, 'guards_ok', ok)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
