"""
Stage P5.13-C -- behaviour-neutrality gate for relocating the ESS ageing
constants (t_cal, cl_nom, dod_nom, soh_min) from the hard-coded defaults in
`shared_energy_storage.py` into `SRP1_ESS_Params.json`.

Frozen gate specification:
    data/SRP1/Results/P513C/frozen_param_move_gate_v1_bd8ab535.json

The gate is STRUCTURAL and runs NO SOLVE. It builds the three ESSO models via
the production path used by p54c_esso_active_energy_validation.py, and records

  I1  the ordered semantic state of each model (variables with bounds, fixed
      flags and values; Params; every constraint expression as a string)
  I2  the .nl export of each model
  I3  the four ageing constants on every SharedEnergyStorage object
  I4  objective / feasibility-penalty / salvage expression strings

Usage
    python p513_c_param_move_gate.py capture --label pre
    python p513_c_param_move_gate.py capture --label post
    python p513_c_param_move_gate.py compare --baseline <pre.json> --candidate <post.json>
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P513C')
AGEING = ('t_cal', 'cl_nom', 'dod_nom', 'soh_min')


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ['git', '--no-optional-locks', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def serialize_set(component):
    """Ordered rendering of a Pyomo Set, indexed or not."""
    try:
        if component.is_indexed():
            return {str(k): [str(v) for v in component[k]] for k in sorted(component.keys(), key=str)}
        return [str(v) for v in component]
    except Exception as error:  # pragma: no cover - defensive
        return f'<unavailable: {error}>'


def model_state(model):
    """Deterministic, ordered semantic state of a Pyomo model."""
    state = {'sets': {}, 'params': {}, 'vars': {}, 'constraints': {}, 'objectives': {}}

    for comp in model.component_objects(pe.Set, active=None):
        state['sets'][comp.local_name] = serialize_set(comp)

    for comp in model.component_objects(pe.Param, active=None):
        entries = {}
        for key in sorted(comp.keys(), key=str) if comp.is_indexed() else [None]:
            try:
                entries[str(key)] = repr(pe.value(comp[key] if key is not None else comp, exception=False))
            except Exception as error:
                entries[str(key)] = f'<unavailable: {error}>'
        state['params'][comp.local_name] = entries

    for comp in model.component_objects(pe.Var, active=None):
        entries = {}
        for key in sorted(comp.keys(), key=str):
            v = comp[key]
            entries[str(key)] = [repr(v.lb), repr(v.ub), repr(v.fixed), repr(v.value), str(v.domain)]
        state['vars'][comp.local_name] = entries

    for comp in model.component_objects(pe.Constraint, active=None):
        entries = {}
        for key in sorted(comp.keys(), key=str):
            entries[str(key)] = str(comp[key].expr)
        state['constraints'][comp.local_name] = {'active': bool(comp.active), 'rows': entries}

    for comp in model.component_objects(pe.Objective, active=None):
        state['objectives'][comp.local_name] = {'active': bool(comp.active),
                                                'sense': str(comp.sense),
                                                'expr': str(comp.expr)}

    for name in ('feasibility_penalty', 'salvage_value', 'salvage_credit', 'investment_cost'):
        if hasattr(model, name):
            try:
                state.setdefault('expressions', {})[name] = str(getattr(model, name).expr)
            except Exception as error:
                state.setdefault('expressions', {})[name] = f'<unavailable: {error}>'

    return state


def build_models():
    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        consensus_vars, _dual = srp.create_admm_variables(planning)
        esso_models, _results = srp.create_shared_energy_storage_model(
            planning.shared_ess_data, consensus_vars, candidate['investment'])
    return planning, esso_models


def capture(label):
    os.makedirs(OUT_DIR, exist_ok=True)
    planning, esso_models = build_models()
    shared_ess_data = planning.shared_ess_data

    manifest = {
        'stage': 'P5.13-C', 'label': label,
        'gate_spec': 'data/SRP1/Results/P513C/frozen_param_move_gate_v1_bd8ab535.json',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': git_head(),
        'harness_sha256': sha256_file(os.path.abspath(__file__)),
        'I3_ageing_constants': {},
        'I1_model_state_sha256': {},
        'I2_nl_sha256': {},
        'I4_expressions': {},
    }

    # I3 -- every SharedEnergyStorage object in every representative year
    for year in shared_ess_data.years:
        for idx, ess in enumerate(shared_ess_data.shared_energy_storages[year]):
            manifest['I3_ageing_constants'][f'{year}|{idx}|bus{ess.bus}'] = {
                name: repr(getattr(ess, name)) for name in AGEING}

    node_ids = sorted(shared_ess_data.active_distribution_network_nodes)
    tmpdir = tempfile.mkdtemp(prefix='p513c_')
    for node_id in node_ids:
        model = esso_models[node_id]
        state = model_state(model)
        state_text = json.dumps(state, sort_keys=True, indent=1)
        manifest['I1_model_state_sha256'][str(node_id)] = sha256_text(state_text)
        with open(os.path.join(OUT_DIR, f'state_{label}_node{node_id}.json'), 'w') as handle:
            handle.write(state_text)

        nl_path = os.path.join(tmpdir, f'esso_node{node_id}.nl')
        model.write(nl_path, io_options={'symbolic_solver_labels': False})
        manifest['I2_nl_sha256'][str(node_id)] = sha256_file(nl_path)

        manifest['I4_expressions'][str(node_id)] = {
            name: sha256_text(str(getattr(model, name).expr))
            for name in ('objective', 'feasibility_penalty', 'salvage_value')
            if hasattr(model, name)}

    out = os.path.join(OUT_DIR, f'manifest_{label}.json')
    with open(out, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[P5.13-C] capture "{label}" written to {out}')
    print(f'[P5.13-C] I3 constants: {json.dumps(sorted({json.dumps(v, sort_keys=True) for v in manifest["I3_ageing_constants"].values()}))}')
    return out


def compare(baseline_path, candidate_path):
    with open(baseline_path) as handle:
        base = json.load(handle)
    with open(candidate_path) as handle:
        cand = json.load(handle)

    failures = []
    for key, label in (('I1_model_state_sha256', 'I1 ordered semantic state'),
                       ('I2_nl_sha256', 'I2 .nl export'),
                       ('I3_ageing_constants', 'I3 ageing constants'),
                       ('I4_expressions', 'I4 expression strings')):
        if base[key] != cand[key]:
            failures.append(label)
            for sub in sorted(set(base[key]) | set(cand[key])):
                if base[key].get(sub) != cand[key].get(sub):
                    failures.append(f'    {label} differs at {sub}: {base[key].get(sub)} -> {cand[key].get(sub)}')

    verdict = {'stage': 'P5.13-C', 'baseline': os.path.basename(baseline_path),
               'candidate': os.path.basename(candidate_path),
               'baseline_git_head': base.get('git_head'), 'candidate_git_head': cand.get('git_head'),
               'failures': failures,
               'verdict': 'PASS — structurally neutral' if not failures else 'FAIL — NOT neutral'}
    out = os.path.join(OUT_DIR, 'verdict.json')
    with open(out, 'w') as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--label', required=True)
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('--baseline', required=True)
    cmp_.add_argument('--candidate', required=True)
    args = parser.parse_args()

    if args.mode == 'capture':
        capture(args.label)
        return 0
    return compare(args.baseline, args.candidate)


if __name__ == '__main__':
    sys.exit(main())
