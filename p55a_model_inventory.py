"""
Stage P5.5-A -- live inventory of the actual production SMOPF models.

Audit only. Builds one TSO (case9) and one DSO (case33_1) model through the real
production path and dumps every Var, Param, Expression, Constraint and Objective
with sizes and representative expressions, so the convexity map in A2 is derived
from the code that runs rather than from assumptions.

    python p55a_model_inventory.py
"""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55A')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def describe(model, label, sample_expr=True):
    out = {'label': label, 'vars': {}, 'params': {}, 'expressions': {},
           'constraints': {}, 'objectives': {}, 'sets': {}}
    for comp in model.component_objects(pe.Var, active=None):
        entries = list(comp.values())
        first = entries[0] if entries else None
        out['vars'][comp.local_name] = {
            'n': len(entries),
            'domain': str(first.domain) if first is not None else None,
            'sample_bounds': list(first.bounds) if first is not None else None,
            'n_fixed': sum(1 for e in entries if e.fixed),
        }
    for comp in model.component_objects(pe.Param, active=None):
        entries = list(comp.values()) if comp.is_indexed() else [comp]
        out['params'][comp.local_name] = {'n': len(entries),
                                          'mutable': bool(getattr(comp, 'mutable', False))}
    for comp in model.component_objects(pe.Expression, active=None):
        entries = list(comp.values())
        out['expressions'][comp.local_name] = {
            'n': len(entries),
            'sample': str(entries[0].expr)[:400] if entries and sample_expr else None}
    for comp in model.component_objects(pe.Constraint, active=None):
        entries = list(comp.values())
        active = [e for e in entries if e.active]
        sample = None
        if active and sample_expr:
            try:
                sample = str(active[0].expr)[:700]
            except Exception as error:
                sample = f'<unavailable: {error}>'
        out['constraints'][comp.local_name] = {
            'n': len(entries), 'n_active': len(active), 'sample': sample}
    for comp in model.component_objects(pe.Objective, active=None):
        out['objectives'][comp.local_name] = {
            'sense': 'min' if comp.sense == pe.minimize else 'max',
            'n_terms_str_len': len(str(comp.expr))}
    for comp in model.component_objects(pe.Set, active=None):
        try:
            out['sets'][comp.local_name] = len(comp)
        except Exception:
            pass
    return out


def topology(network):
    active = [b for b in network.branches if b.is_connected()]
    pairs = {tuple(sorted((b.fbus, b.tbus))) for b in active}
    buses = [n.bus_i for n in network.nodes]
    parent = {b: b for b in buses}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (a, b) in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = len({find(b) for b in buses})
    transformers = [b for b in active if b.is_transformer]
    return {
        'n_nodes': len(buses),
        'n_branches_after_preprocess': len(active),
        'n_distinct_pairs': len(pairs),
        'n_components': comps,
        'cyclomatic': len(pairs) - len(buses) + comps,
        'radial': (len(pairs) - len(buses) + comps) == 0,
        'n_transformers': len(transformers),
        'transformers': [{'branch_id': b.branch_id, 'fbus': b.fbus, 'tbus': b.tbus,
                          'ratio': b.ratio, 'is_transformer': b.is_transformer,
                          'vmag_reg': b.vmag_reg, 'rate': b.rate,
                          'angle_min': b.angle_min, 'angle_max': b.angle_max}
                         for b in transformers],
        'n_branches_with_line_charging': sum(1 for b in active if b.b_sh),
        'n_branches_with_shunt_conductance': sum(1 for b in active if b.g_sh),
        'n_nodes_with_fixed_shunt': sum(
            1 for n in network.nodes if getattr(n, 'gs', 0) or getattr(n, 'bs', 0)),
        'generator_types': {},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-A inventory', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    year, day = list(planning.years)[0], list(planning.days)[0]

    # ---- TSO ----
    tn = planning.transmission_network
    tso_net = tn.network[year][day]
    with redirect_stdout(io.StringIO()):
        tso_model = tso_net.build_model(tn.params)
    report['tso'] = {
        'name': tso_net.name, 'year': year, 'day': day,
        'baseMVA': tso_net.baseMVA,
        'num_instants': tso_net.num_instants,
        'topology': topology(tso_net),
        'params': {k: getattr(tn.params, k, None) for k in (
            'obj_type', 'transf_reg', 'es_reg', 'fl_reg', 'rg_curt', 'l_curt',
            'enforce_vg', 'branch_limit_type', 'ess_model', 'shared_ess_model')},
        'n_shared_ess': len(tso_net.shared_energy_storages),
        'n_ordinary_ess': len(tso_net.energy_storages),
        'n_generators': len(tso_net.generators),
        'n_loads': len(tso_net.loads),
        'model': describe(tso_model, 'TSO case9'),
    }

    # ---- DSO ----
    dso = planning.distribution_networks[5]
    dso_net = dso.network[year][day]
    with redirect_stdout(io.StringIO()):
        dso_model = dso_net.build_model(dso.params)
    report['dso'] = {
        'name': dso_net.name, 'year': year, 'day': day,
        'baseMVA': dso_net.baseMVA,
        'num_instants': dso_net.num_instants,
        'topology': topology(dso_net),
        'params': {k: getattr(dso.params, k, None) for k in (
            'obj_type', 'transf_reg', 'es_reg', 'fl_reg', 'rg_curt', 'l_curt',
            'enforce_vg', 'branch_limit_type', 'ess_model', 'shared_ess_model')},
        'n_shared_ess': len(dso_net.shared_energy_storages),
        'n_ordinary_ess': len(dso_net.energy_storages),
        'n_generators': len(dso_net.generators),
        'n_loads': len(dso_net.loads),
        'model': describe(dso_model, 'DSO case33_1'),
    }

    # ---- generator inventory, live ----
    for tag, net in (('tso', tso_net), ('dso', dso_net)):
        gens = []
        for g in net.generators:
            gens.append({
                'gen_id': g.gen_id, 'bus': g.bus, 'type': g.gen_type,
                'pmax': g.pmax, 'pmin': g.pmin, 'qmax': g.qmax, 'qmin': g.qmin,
                'vg': g.vg, 'status': g.status,
                'power_factor_control': getattr(g, 'power_factor_control', None),
                'max_pf': getattr(g, 'max_pf', None), 'min_pf': getattr(g, 'min_pf', None),
                'is_controllable': g.is_controllable() if hasattr(g, 'is_controllable') else None,
                'is_curtaillable': g.is_curtaillable() if hasattr(g, 'is_curtaillable') else None,
            })
        report[tag]['generators'] = gens
        report[tag]['loads_flexible'] = sum(1 for l in net.loads if l.fl_reg)
        report[tag]['loads_total'] = len(net.loads)

    # ---- ESSO ----
    with redirect_stdout(io.StringIO()):
        consensus_vars, _ = srp.create_admm_variables(planning)
        esso_models, _ = srp.create_shared_energy_storage_model(
            planning.shared_ess_data, consensus_vars, candidate['investment'])
    node0 = list(esso_models)[0]
    report['esso'] = {
        'nodes': list(esso_models),
        'model': describe(esso_models[node0], f'ESSO node {node0}'),
    }

    out = os.path.join(OUT_DIR, 'p55a_inventory.json')
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f'[P5.5-A] inventory -> {out}')
    for tag in ('tso', 'dso'):
        t = report[tag]['topology']
        print(f"\n{tag.upper()} {report[tag]['name']}: nodes={t['n_nodes']} "
              f"branches={t['n_branches_after_preprocess']} pairs={t['n_distinct_pairs']} "
              f"components={t['n_components']} cyclomatic={t['cyclomatic']} radial={t['radial']}")
        print(f"  transformers={t['n_transformers']} {t['transformers']}")
        print(f"  line charging branches={t['n_branches_with_line_charging']} "
              f"shunt-G branches={t['n_branches_with_shunt_conductance']} "
              f"fixed-shunt nodes={t['n_nodes_with_fixed_shunt']}")
        print(f"  params={report[tag]['params']}")
        print(f"  shared_ess={report[tag]['n_shared_ess']} ordinary_ess={report[tag]['n_ordinary_ess']} "
              f"gens={report[tag]['n_generators']} loads={report[tag]['loads_total']} "
              f"flexible_loads={report[tag]['loads_flexible']}")
        print(f"  constraint components ({len(report[tag]['model']['constraints'])}): "
              f"{sorted(report[tag]['model']['constraints'])}")
        print(f"  var components ({len(report[tag]['model']['vars'])}): "
              f"{sorted(report[tag]['model']['vars'])}")
    print(f"\nESSO constraint components: {sorted(report['esso']['model']['constraints'])}")
    print(f"ESSO var components: {sorted(report['esso']['model']['vars'])}")


if __name__ == '__main__':
    main()
