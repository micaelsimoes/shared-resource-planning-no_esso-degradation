"""P5.15 S56 step 2 (S56_PLAN_v2.md, Expert item X7; Addendum 89 ruling 3, R8) -- ZERO-SOLVE gates of the ESSO
slack-penalty params key `esso_slack_penalty` (default unchanged: definitions.PENALTY_ESSO_SLACK = 1e3).

THE CHANGE UNDER TEST (working tree at launch; committed only after G1-G3 pass):
  shared_energy_storage_parameters.py  SharedEnergyStorageParameters.esso_slack_penalty (default PENALTY_ESSO_SLACK);
                                       optional top-level key 'esso_slack_penalty' read strictly (JSON number, not
                                       bool / string / null, finite, > 0; stored as float).
  shared_energy_storage_data.py        `_build_subproblem`: the slack_es_pnet_{up,down} objective weight is
                                       getattr(params, 'esso_slack_penalty', PENALTY_ESSO_SLACK) instead of the
                                       constant (the getattr keeps parameter objects pickled before the key loadable).

TREES. Both production trees are clean exports into `--scratch`, never the live working tree (another Worker edits
shared_resources_planning.py / model_construction_helpers.py / p515_s44_campaign_harness.py concurrently):
  before = `git archive <BASE>` of every tracked root-level *.py + the tracked data/SRP1 inputs (Results excluded);
  after  = the same export + the two files above overlaid from the working tree (sha256 recorded; nothing else).
<BASE> (`--base-commit`) is the commit that adds this script and the frozen spec; the script refuses unless the
spec and this script at BASE equal the files being run, BASE's two production files do not contain the key, and the
overlay files do. The five-block 3 x 3 case file (W89 instance) is copied from BASE into each tree's data/SRP1/.
Each tree is run in its OWN child process (`python <tree>/p515_s56_r8_key_gates.py --child ...`, sys.path[0] = tree;
every production module's __file__ is asserted under the tree).

GATES (all zero solves; `SolveProfileGuard(permitted=())` installed before any project import in the parent and in
every child, verified at exactly 0; read back from each child's JSON):
  G1  default path, bitwise: ESSO NL (plain .nl with labels off = the production form; labelled .nl, .row, .col)
      identical before vs after for every cell = grid {SRP1 2025/2030/2035, five-block 3 x 3 2025-2037} x plan
      {x0, reference unit n7 0.25 MVA / 1 MWh 2025} x salvage in_objective {False, True (in-memory override)} x
      stage {init: the model `create_shared_energy_storage_model` hands to the (stubbed) initial ESSO solve;
      admm: after `update_shared_energy_storage_model_to_admm` with production's al-scale and sigma, ADMM
      request/dual Params set to 0.0} x node {5, 7, 9}. Control G1c: before vs a second before child, bitwise.
  G2  key = 1e4 in memory (the value parsed by the production reader from a scratch params file, assigned to the
      read params object): per cell, after-default vs after-1e4 labelled NL: .row and .col identical; every
      differing line lies in the objective gradient segment G0, belongs to a slack_es_pnet_{up,down} column, has
      coefficient exactly 1000.0 -> exactly 10000.0; every slack column of the .col is among them (count declared
      in the spec); the plain NL differs at the same lines with the same values.
  G3  fixtures: the declared list (git-tracked *.pkl + W183c NAMED_FIXTURES + data/SRP1/Results/FrozenSMOPF/**),
      fixed before any load, unpickled under the before modules and under the after modules (one child each);
      load count = declared count exactly in each; per-path outcome identical (no fixture newly failing).
      G3c: a SharedEnergyStorageParameters pickled by the before child (no `esso_slack_penalty` attribute) is
      unpickled by the after child, lacks the attribute, and the SRP1 ESSO built with it is bitwise the after
      default build (init stage, both plans, in_objective False).
  G0  (reader, after tree, scratch files): absent -> 1e3; 1e4 / 10000 -> 10000.0 (float); invalid values raise.
  G4  (measurement with a prediction): per grid, the G-CC-3 (ii) slack-safety ratio = peak over cohorts of
      kappa * max(c_ch, c_dch) * SoH_cal(T) over the bar 1e-3 * (slack objective coefficient read from the built
      model), at 1e3 (before, after default) and at 1e4 (after, key set). kappa is read from the built ADMM model.

Exit 0 iff G0-G3 (and G1c, G3c) pass. G4 is reported against its prediction, not gated.

OUTPUT (write-once): `--out` must NOT exist (refused otherwise); the script creates it. Launch logs go to siblings.
LAUNCH (attached, alone, both streams captured; spec + this script committed first as BASE):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_r8_key_gates.py \\
      --spec data/SRP1/Results/P515S56/frozen_r8_key_gate_spec_v1_<h>.json \\
      --out data/SRP1/Results/P515S56/r8_key_v1 --scratch <scratchpad>/r8_key_v1 --base-commit <BASE> \\
      > data/SRP1/Results/P515S56/r8_key_v1_launch_stdout.log 2> data/SRP1/Results/P515S56/r8_key_v1_launch_stderr.log
then  ... p515_s56_r8_key_gates.py --manifest --out data/SRP1/Results/P515S56/r8_key_v1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = '--child' in sys.argv
if CHILD:  # the tree this file was exported into is the ONLY project source on sys.path
    sys.path[:] = [HERE] + [p for p in sys.path[1:] if os.path.abspath(p or '.') != HERE]
elif HERE not in sys.path:
    sys.path.insert(0, HERE)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label=f"P5.15 S56 R8 key gates ({'child' if CHILD else 'parent'})").install()

import argparse  # noqa: E402
import ast  # noqa: E402
import difflib  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

SCHEMA = 'p515_s56_r8_key_gates_v1'
SCRIPT = 'p515_s56_r8_key_gates.py'
PROD_FILES = ('shared_energy_storage_parameters.py', 'shared_energy_storage_data.py')
KEY = 'esso_slack_penalty'
PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
W183C = 'p515_s53_w183c_nrf_gates.py'
FROZEN_SMOPF_DIR = os.path.join('data', 'SRP1', 'Results', 'FrozenSMOPF')
NL_FILES = ('plain', 'sym', 'row', 'col')
SLACK_RE = re.compile(r'^slack_es_pnet_(up|down)\[')


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _log(msg):
    print(f'[r8 {datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    with open(path, 'rb') as handle:
        return _sha_bytes(handle.read())


def _dump(obj, path):
    import gate_result_io as GRIO
    with open(path, 'x') as handle:
        handle.write(GRIO.dumps(obj, indent=1, sort_keys=True))


# ======================================================================================================================
#  CHILD: builds (one tree per process)
# ======================================================================================================================
def _assert_modules_from_tree(mods):
    files = {m.__name__: os.path.abspath(m.__file__) for m in mods}
    bad = {k: v for k, v in files.items() if not v.startswith(HERE + os.sep)}
    if bad:
        raise SystemExit(f'refusing: modules not loaded from the tree {HERE}: {bad}')
    return files


def _read_planning(case_name, side):
    from shared_resources_planning import SharedResourcesPlanning
    for sub in ('Diagrams', os.path.join('Results', 'Logs')):
        os.makedirs(os.path.join(side, sub))
    planning = SharedResourcesPlanning(os.path.join(HERE, 'data', 'SRP1'), case_name)
    planning.diagrams_dir = os.path.join(side, 'Diagrams')
    planning.results_dir = os.path.join(side, 'Results')
    planning.logs_dir = os.path.join(side, 'Results', 'Logs')
    stream = io.StringIO()
    t0 = time.time()
    with redirect_stdout(stream):
        planning.read_planning_problem()
    planning.shared_ess_data.logs_dir = planning.logs_dir
    return planning, {'read_s': time.time() - t0, 'stdout_tail': stream.getvalue()[-2000:],
                      'scenario_checksum': (planning.scenario_metadata or {}).get('combined_scenario_checksum')}


def _candidate(planning, plan):
    import shared_resources_planning as srp
    cand = planning.get_initial_candidate_solution()
    for node, year, s, e in plan:
        cand['investment'][node][year]['s'] = s
        cand['investment'][node][year]['e'] = e
    srp._rebuild_candidate_total_capacities(planning, cand)
    return cand


def _write_nl(model, nl_dir, stem):
    paths = {'plain': os.path.join(nl_dir, f'{stem}.nl'), 'sym': os.path.join(nl_dir, f'{stem}.sym.nl'),
             'row': os.path.join(nl_dir, f'{stem}.sym.row'), 'col': os.path.join(nl_dir, f'{stem}.sym.col')}
    model.write(paths['plain'], format='nl', io_options={'symbolic_solver_labels': False})
    model.write(paths['sym'], format='nl', io_options={'symbolic_solver_labels': True})
    return {k: {'path': v, 'sha256': _sha_file(v), 'bytes': os.path.getsize(v)} for k, v in paths.items()}


def _build_cell(planning, plan, in_objective, penalty, nl_dir, cell, record_models=False):
    """One ESSO build through production: `create_shared_energy_storage_model` with the instance's `optimize`
    stubbed (NL written inside the stub = what the initial solve would receive), then
    `update_shared_energy_storage_model_to_admm` with production's al-scale and sigma (ADMM Params set to 0.0)."""
    import pyomo.environ as pe
    import shared_resources_planning as srp
    sed = planning.shared_ess_data
    sv = sed.params.salvage_value
    saved_obj = sv.in_objective
    saved_pen = sed.params.__dict__.get(KEY, '<absent>')
    sv.in_objective = in_objective
    if penalty is not None:
        setattr(sed.params, KEY, penalty)
    out = {'init': {}, 'admm': {}, 'stub_calls': 0}
    models_kept = {}
    try:
        cand = _candidate(planning, plan)
        consensus_vars, _dual = srp.create_admm_variables(planning)

        def stub(models, *a, **k):
            out['stub_calls'] += 1
            for node_id, model in models.items():
                out['init'][str(node_id)] = _write_nl(model, nl_dir, f'{cell}__init__n{node_id}')
            return {n: None for n in models}

        sed.optimize = stub
        try:
            with redirect_stdout(io.StringIO()):
                esso, _res = srp.create_shared_energy_storage_model(sed, consensus_vars, cand['investment'])
        finally:
            sed.__dict__.pop('optimize', None)  # back to the class method
        admm = planning.params.admm
        sigma = admm.objective_scale
        if sigma is None:
            raise SystemExit('refusing: admm.objective_scale (sigma_fixed) is None; the spec declares the fixed sigma')
        with redirect_stdout(io.StringIO()):
            a_scale, applied = srp._resolve_esso_al_scale(planning, admm, sigma)
        srp.update_shared_energy_storage_model_to_admm(planning, esso, admm, al_scale_esso=a_scale,
                                                       objective_scale=sigma)
        for node_id, model in esso.items():
            for name in ('p_req', 'q_req', 'dual_p_req', 'dual_q_req'):
                comp = getattr(model, name)
                for idx in comp.index_set():
                    comp[idx] = 0.0
            out['admm'][str(node_id)] = _write_nl(model, nl_dir, f'{cell}__admm__n{node_id}')
            if hasattr(model, 'salvage_objective_kappa'):
                out.setdefault('kappa', {})[str(node_id)] = pe.value(model.salvage_objective_kappa)
        out['sigma'] = sigma
        out['al_scale'] = {'value': a_scale, 'applied': applied}
        if record_models:
            models_kept = esso
    finally:
        sv.in_objective = saved_obj
        if penalty is not None:
            if saved_pen == '<absent>':
                delattr(sed.params, KEY)
            else:
                setattr(sed.params, KEY, saved_pen)
    return out, models_kept


def _penalty_in_force(model):
    from pyomo.repn import generate_standard_repn
    repn = generate_standard_repn(model.objective.expr, compute_values=True)
    coefs = {v.name: c for v, c in zip(repn.linear_vars, repn.linear_coefs)}
    slack = {k: v for k, v in coefs.items() if SLACK_RE.match(k)}
    vals = sorted(set(slack.values()))
    return {'n_slack_terms': len(slack), 'distinct_coefficients': vals}


def _g4(planning, models):
    """G-CC-3 (ii) slack safety on a built in_objective model: peak over cohorts of kappa * max(c) * SoH_cal(T)."""
    import pyomo.environ as pe
    import shared_energy_storage_data as SED
    sed = planning.shared_ess_data
    years = list(sed.years)
    per_node = {}
    for node_id, model in models.items():
        kappa = pe.value(model.salvage_objective_kappa)
        coeffs = model._salvage_throughput_coefficients
        idx = sed.get_shared_energy_storage_idx(node_id)
        cohorts = {}
        for y_inv, year in enumerate(years):
            cs = [max(v) for k, v in coeffs.items() if k[0] == y_inv]
            if not cs:
                continue
            ses = sed.shared_energy_storages[year][idx]
            age_t, _rem, rlf = SED._get_remaining_calendar_life(sed, y_inv, ses)
            soh_cal = ses.phi_cal ** age_t
            cohorts[str(year)] = {'max_c': max(cs), 'age_at_terminal_y': age_t, 'rlf': rlf, 'soh_cal_T': soh_cal,
                                  'marginal': kappa * max(cs) * soh_cal}
        pen = _penalty_in_force(model)
        if len(pen['distinct_coefficients']) != 1:
            raise SystemExit(f'G4: slack coefficients not unique on node {node_id}: {pen}')
        bar = 1e-3 * pen['distinct_coefficients'][0]
        peak_year = max(cohorts, key=lambda y: cohorts[y]['marginal'])
        per_node[str(node_id)] = {'kappa': kappa, 'penalty_in_force': pen['distinct_coefficients'][0],
                                  'n_slack_terms': pen['n_slack_terms'], 'bar': bar, 'cohorts': cohorts,
                                  'peak_cohort': peak_year, 'peak': cohorts[peak_year]['marginal'],
                                  'ratio': cohorts[peak_year]['marginal'] / bar}
    return per_node


def _reader_tests(scratch, base_params_path):
    import definitions as DEFS
    from shared_energy_storage_parameters import SharedEnergyStorageParameters
    with open(base_params_path) as handle:
        base = json.load(handle)
    cases = {'absent': ('<absent>', DEFS.PENALTY_ESSO_SLACK), 'float_1e4': (1e4, 10000.0), 'int_10000': (10000, 10000.0),
             'float_1': (1.0, 1.0)}
    bad = {'bool_true': True, 'string_1e4': '1e4', 'null': None, 'zero': 0, 'negative': -1e4, 'nan': float('nan'),
           'inf': float('inf'), 'list': [1e4], 'object': {}}
    res = {'ok': {}, 'raises': {}}
    os.makedirs(scratch)
    for name, (value, expected) in cases.items():
        data = dict(base)
        if value != '<absent>':
            data[KEY] = value
        path = os.path.join(scratch, f'params_{name}.json')
        with open(path, 'w') as handle:
            json.dump(data, handle)
        p = SharedEnergyStorageParameters()
        p.read_parameters_from_file(path)
        got = getattr(p, KEY)
        res['ok'][name] = {'value': got, 'type': type(got).__name__,
                           'pass': got == expected and type(got) is float}
    for name, value in bad.items():
        data = dict(base)
        data[KEY] = value
        path = os.path.join(scratch, f'params_{name}.json')
        with open(path, 'w') as handle:
            json.dump(data, handle)
        p = SharedEnergyStorageParameters()
        try:
            p.read_parameters_from_file(path)
            res['raises'][name] = {'raised': False, 'pass': False, 'value_read': repr(getattr(p, KEY))}
        except ValueError as error:
            res['raises'][name] = {'raised': True, 'pass': True, 'error': str(error)}
    res['default_attribute'] = getattr(SharedEnergyStorageParameters(), KEY)
    res['constant'] = DEFS.PENALTY_ESSO_SLACK
    res['pass'] = (all(v['pass'] for v in res['ok'].values()) and all(v['pass'] for v in res['raises'].values())
                   and res['default_attribute'] == 1e3 and res['constant'] == 1e3)
    return res


def child_build(args):
    spec = json.load(open(args.spec))
    import definitions as DEFS
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    import shared_resources_planning as SRP
    import model_construction_helpers as MCH
    mods = _assert_modules_from_tree([DEFS, SED, SEP, SRP, MCH])
    tag = args.tag
    work = args.work
    nl_root = os.path.join(work, 'nl')
    rec = {'schema': SCHEMA + '_child_build', 'tag': tag, 'tree': HERE, 'modules': mods, 'started_utc': _utc(),
           'interpreter': sys.executable, 'constant_PENALTY_ESSO_SLACK': DEFS.PENALTY_ESSO_SLACK,
           'params_has_key_attr_default': hasattr(SEP.SharedEnergyStorageParameters(), KEY),
           'grids': {}, 'g4': {}, 'errors': {}}
    is_after = tag.startswith('after')
    penalties = {'default': None}
    if is_after:
        rec['g0_reader'] = _reader_tests(os.path.join(work, 'g0_reader'),
                                         os.path.join(HERE, 'data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'))
        # the 1e4 value as the production reader parses it from a scratch params file (key added to a copy)
        p = SEP.SharedEnergyStorageParameters()
        src = os.path.join(HERE, 'data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
        data = json.load(open(src))
        data[KEY] = spec['g2']['key_value']
        scratch_params = os.path.join(work, 'params_key_1e4.json')
        with open(scratch_params, 'x') as handle:
            json.dump(data, handle)
        p.read_parameters_from_file(scratch_params)
        penalties['key_1e4'] = getattr(p, KEY)
        rec['key_1e4_parsed'] = penalties['key_1e4']
    for grid, gspec in spec['grids'].items():
        planning, info = _read_planning(gspec['case_file'], os.path.join(work, 'side', grid))
        sed = planning.shared_ess_data
        info['years'] = {str(y): sed.years[y] for y in sed.years}
        info['nodes'] = list(sed.active_distribution_network_nodes)
        info['slacks'] = sed.params.slacks
        info['params_file'] = sed.params_file
        info['salvage_enabled'] = sed.params.salvage_value.enabled
        info['salvage_in_objective_as_read'] = sed.params.salvage_value.in_objective
        info['soh_point'] = SED._esso_ageing_model_settings(sed)[0]
        info['params_key_attr'] = sed.params.__dict__.get(KEY, '<absent>')
        info['checksum_matches_spec'] = info['scenario_checksum'] == gspec['scenario_checksum']
        rec['grids'][grid] = {'read': info, 'cells': {}}
        for plan_name, plan in spec['plans'].items():
            plan_t = [tuple(x) for x in plan]
            for in_obj in (False, True):
                for pen_name, pen in penalties.items():
                    cell = f'{grid}__{plan_name}__inobj{int(in_obj)}__{pen_name}'
                    nl_dir = os.path.join(nl_root, cell)
                    os.makedirs(nl_dir)
                    keep = in_obj and plan_name == 'ref_unit'
                    out, models = _build_cell(planning, plan_t, in_obj, pen, nl_dir, cell, record_models=keep)
                    rec['grids'][grid]['cells'][cell] = out
                    if keep:
                        rec['g4'].setdefault(grid, {})[pen_name] = _g4(planning, models)
        if grid == spec['g3c']['grid']:
            if tag == 'before':
                path = os.path.join(args.g3c_dir, 'params_pickled_by_before.pkl')
                with open(path, 'xb') as handle:
                    pickle.dump(sed.params, handle)
                rec['g3c_pickle'] = {'path': path, 'sha256': _sha_file(path),
                                     'has_key_attr': KEY in sed.params.__dict__}
            elif tag == 'after':
                path = os.path.join(args.g3c_dir, 'params_pickled_by_before.pkl')
                with open(path, 'rb') as handle:
                    old = pickle.load(handle)
                g3c = {'loaded_type': f'{type(old).__module__}.{type(old).__name__}',
                       'class_module_file': os.path.abspath(sys.modules[type(old).__module__].__file__),
                       'has_key_attr_after_unpickle': KEY in old.__dict__, 'cells': {}}
                saved = sed.params
                sed.params = old
                try:
                    for plan_name, plan in spec['plans'].items():
                        cell = f'{grid}__{plan_name}__inobj0__g3c_unpickled_params'
                        nl_dir = os.path.join(nl_root, cell)
                        os.makedirs(nl_dir)
                        out, _m = _build_cell(planning, [tuple(x) for x in plan], False, None, nl_dir, cell)
                        g3c['cells'][cell] = out
                finally:
                    sed.params = saved
                rec['g3c'] = g3c
        del planning
    rec['finished_utc'] = _utc()
    rec['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    _dump(rec, os.path.join(work, f'child_build_{tag}.json'))
    return 0


def child_fixtures(args):
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    import shared_resources_planning as SRP
    import model_construction_helpers as MCH
    mods = _assert_modules_from_tree([SED, SEP, SRP, MCH])
    declared = json.load(open(args.fixture_list))
    counter = {'load': 0}
    real_load = pickle.load

    def counted_load(handle, *a, **k):
        counter['load'] += 1
        return real_load(handle, *a, **k)

    pickle.load = counted_load
    results = {}
    for rel in declared:
        path = os.path.join(args.repo, rel)
        if not os.path.isfile(path):
            results[rel] = {'outcome': 'ABSENT'}
            continue
        try:
            with open(path, 'rb') as handle:
                obj = pickle.load(handle)
            results[rel] = {'outcome': 'ok', 'type': type(obj).__name__, 'sha256': _sha_file(path)}
            del obj
        except Exception as error:  # noqa: BLE001 -- recorded per path
            msg = re.sub(r" from '[^']*'", " from '<module file>'", f'{type(error).__name__}: {error}')
            results[rel] = {'outcome': msg, 'sha256': _sha_file(path)}
    pickle.load = real_load
    n_present = sum(1 for r in results.values() if r['outcome'] != 'ABSENT')
    rec = {'schema': SCHEMA + '_child_fixtures', 'tag': args.tag, 'tree': HERE, 'modules': mods,
           'n_declared': len(declared), 'n_present': n_present, 'n_loads': counter['load'], 'results': results,
           'guard': {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}}
    _dump(rec, os.path.join(args.work, f'child_fixtures_{args.tag}.json'))
    return 0


# ======================================================================================================================
#  PARENT
# ======================================================================================================================
def _git(*a, check=True):
    return subprocess.run(['git', *a], cwd=HERE, capture_output=True, text=True, check=check).stdout


def _git_bytes(*a):
    return subprocess.run(['git', *a], cwd=HERE, capture_output=True, check=True).stdout


def _refuse_concurrent():
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout.splitlines()
    others = []
    for line in ps:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] == str(os.getpid()):
            continue
        if 'python' in os.path.basename(parts[1]) and any(os.path.basename(p) == SCRIPT for p in parts[2:]):
            if '--child' not in parts:
                others.append(line)
    if others:
        raise SystemExit(f'refusing: another copy is running: {others}')


def _named_fixtures_from_w183c(base):
    """W183c's NAMED_FIXTURES dict, read from its source at BASE by `ast` (not imported: it patches pickle)."""
    src = _git('show', f'{base}:{W183C}')
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'NAMED_FIXTURES' for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit('refusing: NAMED_FIXTURES not found in W183c source')


def _declare_fixtures(base):
    tracked = [p for p in _git('ls-files', '--', '*.pkl').splitlines() if p]
    named = _named_fixtures_from_w183c(base)
    globbed = []
    for root, _dirs, files in os.walk(os.path.join(HERE, FROZEN_SMOPF_DIR)):
        for f in sorted(files):
            if f.endswith('.pkl'):
                globbed.append(os.path.relpath(os.path.join(root, f), HERE))
    by = {}
    for rel in tracked:
        by.setdefault(rel, []).append('git ls-files *.pkl')
    for rel, checks in named.items():
        by.setdefault(rel, []).append('W183c NAMED_FIXTURES: ' + ', '.join(checks))
    for rel in globbed:
        by.setdefault(rel, []).append('FrozenSMOPF/** glob')
    return dict(sorted(by.items()))


def _export(base, dest, overlay=None):
    os.makedirs(dest)
    listing = [p for p in _git('ls-tree', '-r', '--name-only', base).splitlines()
               if (p.endswith('.py') and '/' not in p)
               or (p.startswith('data/SRP1/') and not p.startswith('data/SRP1/Results/'))]
    tar = dest.rstrip('/') + '.tar'
    subprocess.run(['git', 'archive', '-o', tar, base, '--', *listing], cwd=HERE, check=True)
    subprocess.run(['tar', '-x', '-f', tar, '-C', dest], check=True)
    os.remove(tar)
    overlaid = {}
    for rel in overlay or ():
        shutil.copyfile(os.path.join(HERE, rel), os.path.join(dest, rel))
        overlaid[rel] = _sha_file(os.path.join(dest, rel))
    return {'n_files': len(listing), 'overlay_sha256': overlaid,
            'prod_sha256': {rel: _sha_file(os.path.join(dest, rel)) for rel in PROD_FILES}}


def _run_child(tree, extra, log_stem, out, env):
    cmd = [PY, '-u', os.path.join(tree, SCRIPT), '--child', *extra]
    with open(os.path.join(out, f'{log_stem}_stdout.log'), 'x') as so, \
            open(os.path.join(out, f'{log_stem}_stderr.log'), 'x') as se:
        rc = subprocess.run(cmd, cwd=tree, stdout=so, stderr=se, env=env).returncode
    _log(f'child {log_stem}: rc {rc}')
    return rc


def _read_text(path):
    with open(path) as handle:
        return handle.read().splitlines()


def _g_segment(lines):
    """Line-index range [start, end) of the objective-0 gradient entries."""
    for i, line in enumerate(lines):
        if line.startswith('G0 '):
            n = int(line.split()[1].split('#')[0])
            return i + 1, i + 1 + n
    return None


def _g2_pair(dflt, key, expected_slack):
    """after-default vs after-key, one node of one cell/stage."""
    res = {}
    for f in ('row', 'col'):
        res[f'{f}_identical'] = dflt[f]['sha256'] == key[f]['sha256']
    cols = _read_text(dflt['col']['path'])
    slack_cols = {j for j, name in enumerate(cols) if SLACK_RE.match(name)}
    analysis = {}
    for form in ('sym', 'plain'):
        a, b = _read_text(dflt[form]['path']), _read_text(key[form]['path'])
        seg = _g_segment(a)
        diff_idx = [i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]]
        changed, bad = {}, []
        for i in diff_idx:
            if seg is None or not (seg[0] <= i < seg[1]) or i >= len(b):
                bad.append({'line': i, 'reason': 'outside G0 segment', 'a': a[i] if i < len(a) else None,
                            'b': b[i] if i < len(b) else None})
                continue
            ja, ca = a[i].split('#')[0].split()
            jb, cb = b[i].split('#')[0].split()
            ja, jb, ca, cb = int(ja), int(jb), float(ca), float(cb)
            ok = ja == jb and ja in slack_cols and ca == 1000.0 and cb == 10000.0 and cb == 10.0 * ca
            changed[ja] = (ca, cb)
            if not ok:
                bad.append({'line': i, 'reason': 'not a slack column x10 (1000 -> 10000)', 'col': ja,
                            'name': cols[ja] if ja < len(cols) else None, 'a': ca, 'b': cb})
        analysis[form] = {'n_lines': (len(a), len(b)), 'n_differing': len(diff_idx), 'n_bad': len(bad),
                          'bad_first': bad[:5], 'all_slack_columns_changed': set(changed) == slack_cols,
                          'changed_cols': sorted(changed)}
    res['n_slack_columns'] = len(slack_cols)
    res['expected_slack_columns'] = expected_slack
    res['sym'] = {k: v for k, v in analysis['sym'].items() if k != 'changed_cols'}
    res['plain'] = {k: v for k, v in analysis['plain'].items() if k != 'changed_cols'}
    res['plain_and_sym_same_columns'] = analysis['sym']['changed_cols'] == analysis['plain']['changed_cols']
    res['pass'] = (res['row_identical'] and res['col_identical'] and len(slack_cols) == expected_slack
                   and all(analysis[f]['n_bad'] == 0 and analysis[f]['all_slack_columns_changed']
                           and analysis[f]['n_differing'] == expected_slack for f in ('sym', 'plain'))
                   and res['plain_and_sym_same_columns'])
    return res


def _iter_nl(child, pen_name):
    """(cell_base, stage, node) -> NL record, for the cells built with penalty `pen_name`."""
    out = {}
    for grid, g in child['grids'].items():
        for cell, c in g['cells'].items():
            if not cell.endswith(f'__{pen_name}'):
                continue
            base = cell[: -len(f'__{pen_name}')]
            for stage in ('init', 'admm'):
                for node, files in c[stage].items():
                    out[(base, stage, node)] = files
    return out


def _bitwise(a, b):
    keys = sorted(set(a) | set(b))
    per, missing = {}, []
    for k in keys:
        if k not in a or k not in b:
            missing.append('|'.join(k))
            continue
        per['|'.join(k)] = {f: a[k][f]['sha256'] == b[k][f]['sha256'] for f in NL_FILES}
    return {'n_compared': len(per), 'n_files_compared': 4 * len(per), 'missing': missing,
            'n_identical_all_four': sum(all(v.values()) for v in per.values()),
            'differing': {k: v for k, v in per.items() if not all(v.values())},
            'pass': bool(per) and not missing and all(all(v.values()) for v in per.values())}


def _gzip_tree(src_root, dst_root):
    inventory = {}
    for root, _d, files in os.walk(src_root):
        for f in sorted(files):
            src = os.path.join(root, f)
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(dst_root, rel + '.gz')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'rb') as handle:
                data = handle.read()
            with open(dst, 'xb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz:
                    gz.write(data)
            inventory[rel] = _sha_bytes(data)
    return inventory


def parent(args):
    t_start = time.time()
    spec_path = os.path.abspath(args.spec)
    spec_bytes = open(spec_path, 'rb').read()
    spec = json.loads(spec_bytes)
    m = re.search(r'_v(\d+)_([0-9a-f]{8})\.json$', spec_path)
    if not m or _sha_bytes(spec_bytes)[:8] != m.group(2):
        raise SystemExit(f'refusing: spec filename hash does not match its content ({spec_path})')
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f'refusing: {out} exists')
    if args.rehearsal and out.startswith(HERE + os.sep):
        raise SystemExit('refusing: a rehearsal writes outside the repository only')
    scratch = os.path.abspath(args.scratch)
    if os.path.exists(scratch) or scratch.startswith(HERE + os.sep):
        raise SystemExit(f'refusing: scratch {scratch} exists or lies inside the repository')
    _refuse_concurrent()
    base = _git('rev-parse', args.base_commit).strip()
    # provenance refusals (each logs which check failed)
    checks = {}
    spec_rel = os.path.relpath(spec_path, HERE)
    if args.rehearsal:  # debugging only: script and spec not yet committed at BASE; never the evidence run
        checks['REHEARSAL_script_and_spec_not_checked_at_base'] = True
    else:
        checks['script_at_base_equals_running'] = (_sha_bytes(_git_bytes('show', f'{base}:{SCRIPT}'))
                                                   == _sha_file(os.path.join(HERE, SCRIPT)))
        checks['spec_at_base_equals_file'] = (_sha_bytes(_git_bytes('show', f'{base}:{spec_rel}'))
                                              == _sha_bytes(spec_bytes))
    base_prod = {rel: _git_bytes('show', f'{base}:{rel}') for rel in PROD_FILES}
    work_prod = {rel: open(os.path.join(HERE, rel), 'rb').read() for rel in PROD_FILES}
    checks['base_prod_files_lack_key'] = all(KEY.encode() not in v for v in base_prod.values())
    checks['overlay_files_carry_key'] = all(KEY.encode() in v for v in work_prod.values())
    # every committed JSON except the run's own spec (which names the key; it is not a params file)
    json_hits = [line.split(':', 1)[1] for line in
                 _git('grep', '-l', KEY, base, '--', '*.json', check=False).splitlines() if line.strip()]
    checks['no_committed_json_sets_key'] = not [h for h in json_hits if h != spec_rel]
    checks['definitions_constant_unchanged_at_base'] = (
        re.search(rb'^PENALTY_ESSO_SLACK = 1e3$', _git_bytes('show', f'{base}:definitions.py'), re.M) is not None)
    inst_bytes = _git_bytes('show', f"{base}:{spec['grids']['fiveblock_3x3']['source_case_rel']}")
    checks['fiveblock_case_sha256_matches_spec'] = (_sha_bytes(inst_bytes)
                                                    == spec['grids']['fiveblock_3x3']['source_case_sha256'])
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f'refusing: provenance checks failed: {failed}')
    os.makedirs(out)
    os.makedirs(scratch)
    _log(f'base {base}; out {out}; scratch {scratch}')
    diff_text = {rel: ''.join(difflib.unified_diff(base_prod[rel].decode().splitlines(True),
                                                   work_prod[rel].decode().splitlines(True),
                                                   f'{base[:8]}:{rel}', f'overlay:{rel}')) for rel in PROD_FILES}
    trees = {}
    for tag, overlay in (('before', None), ('after', PROD_FILES)):
        tree = os.path.join(scratch, f'tree_{tag}')
        trees[tag] = {'path': tree, **_export(base, tree, overlay)}
        if args.rehearsal:
            shutil.copyfile(os.path.join(HERE, SCRIPT), os.path.join(tree, SCRIPT))
        with open(os.path.join(tree, 'data', 'SRP1', spec['grids']['fiveblock_3x3']['case_file']), 'xb') as handle:
            handle.write(inst_bytes)
    checks['after_overlay_equals_working_tree_at_launch'] = all(
        trees['after']['overlay_sha256'][rel] == _sha_bytes(work_prod[rel]) for rel in PROD_FILES)
    checks['before_prod_equals_base'] = all(
        trees['before']['prod_sha256'][rel] == _sha_bytes(base_prod[rel]) for rel in PROD_FILES)
    env = dict(os.environ)
    env.update({'NLP_SOLVER_PATH': spec['environment']['NLP_SOLVER_PATH'],
                'LP_SOLVER_PATH': spec['environment']['LP_SOLVER_PATH']})
    spec_copy = os.path.join(scratch, 'spec.json')
    shutil.copyfile(spec_path, spec_copy)
    g3c_dir = os.path.join(scratch, 'g3c')
    os.makedirs(g3c_dir)
    # declared fixture list, fixed before any load
    fixtures = _declare_fixtures(base)
    fixture_list = os.path.join(out, 'fixtures_declared.json')
    _dump({'n': len(fixtures), 'fixtures': fixtures}, fixture_list)
    flist_scratch = os.path.join(scratch, 'fixtures_list.json')
    with open(flist_scratch, 'x') as handle:
        json.dump(sorted(fixtures), handle)
    rc = {}
    works = {}
    # reference first: the pre-change tree, then its determinism control, then the post-change tree
    for tag, tree_tag in (('before', 'before'), ('before_rerun', 'before'), ('after', 'after')):
        works[tag] = os.path.join(scratch, f'work_{tag}')
        os.makedirs(works[tag])
        rc[f'build_{tag}'] = _run_child(trees[tree_tag]['path'],
                                        ['build', '--tag', tag, '--spec', spec_copy, '--work', works[tag],
                                         '--g3c-dir', g3c_dir], f'child_build_{tag}', out, env)
    for tag in ('before', 'after'):
        w = os.path.join(scratch, f'work_fixtures_{tag}')
        os.makedirs(w)
        works[f'fixtures_{tag}'] = w
        rc[f'fixtures_{tag}'] = _run_child(trees[tag]['path'],
                                           ['fixtures', '--tag', tag, '--work', w, '--fixture-list', flist_scratch,
                                            '--repo', HERE], f'child_fixtures_{tag}', out, env)
    children = {}
    for tag in ('before', 'before_rerun', 'after'):
        p = os.path.join(works[tag], f'child_build_{tag}.json')
        children[tag] = json.load(open(p)) if os.path.isfile(p) else None
    fx = {}
    for tag in ('before', 'after'):
        p = os.path.join(works[f'fixtures_{tag}'], f'child_fixtures_{tag}.json')
        fx[tag] = json.load(open(p)) if os.path.isfile(p) else None
    result = {'schema': SCHEMA, 'script': SCRIPT, 'rehearsal': bool(args.rehearsal), 'spec': spec_rel, 'spec_sha256': _sha_bytes(spec_bytes),
              'base_commit': base, 'git_head_at_launch': _git('rev-parse', 'HEAD').strip(),
              'started_utc': datetime.fromtimestamp(t_start, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'interpreter': sys.executable, 'provenance_checks': checks, 'diff_under_test': diff_text,
              'trees': trees, 'child_returncodes': rc, 'gates': {}, 'errors': {}}
    if any(v != 0 for v in rc.values()) or any(v is None for v in children.values()) or any(
            v is None for v in fx.values()):
        result['errors']['children'] = 'a child failed or wrote no record; see child_*_stderr.log'
    try:
        gates = result['gates']
        # read-back + instance identity
        gates['children'] = {t: {'guard': c['guard'], 'modules_from_tree': all(
            v.startswith(c['tree']) for v in c['modules'].values()),
            'grids_read': {g: {k: v['read'][k] for k in ('scenario_checksum', 'checksum_matches_spec', 'years',
                                                         'slacks', 'params_file', 'salvage_in_objective_as_read',
                                                         'soh_point', 'params_key_attr', 'read_s')}
                           for g, v in c['grids'].items()},
            'stub_calls': {cell: v['stub_calls'] for g in c['grids'].values() for cell, v in g['cells'].items()}}
            for t, c in children.items() if c}
        child_ok = all(
            c['guard']['counts']['blocked_solve'] == 0 and c['guard']['counts']['permitted_solve'] == 0
            and not c['guard']['verify_0_failures'] and gates['children'][t]['modules_from_tree']
            and all(v['checksum_matches_spec'] for v in gates['children'][t]['grids_read'].values())
            and all(n == 1 for n in gates['children'][t]['stub_calls'].values())
            for t, c in children.items() if c)
        gates['children_ok'] = child_ok and all(c is not None for c in children.values())
        # G1 + control
        before = _iter_nl(children['before'], 'default')
        gates['G1'] = _bitwise(before, _iter_nl(children['after'], 'default'))
        gates['G1']['n_expected'] = spec['g1']['n_node_models_per_tree']
        gates['G1']['pass'] = gates['G1']['pass'] and gates['G1']['n_compared'] == spec['g1']['n_node_models_per_tree']
        gates['G1c_determinism'] = _bitwise(before, _iter_nl(children['before_rerun'], 'default'))
        # G2
        dflt, key = _iter_nl(children['after'], 'default'), _iter_nl(children['after'], 'key_1e4')
        per = {}
        for k in sorted(dflt):
            grid = k[0].split('__')[0]
            per['|'.join(k)] = _g2_pair(dflt[k], key[k], spec['grids'][grid]['expected_slack_columns_per_node'])
        gates['G2'] = {'key_value_parsed': children['after'].get('key_1e4_parsed'),
                       'n_pairs': len(per), 'n_pass': sum(v['pass'] for v in per.values()),
                       'failing': {k: v for k, v in per.items() if not v['pass']},
                       'example': per[sorted(per)[0]] if per else None,
                       'pass': bool(per) and len(per) == spec['g1']['n_node_models_per_tree']
                       and all(v['pass'] for v in per.values())
                       and children['after'].get('key_1e4_parsed') == 10000.0}
        # G0
        gates['G0_reader'] = children['after']['g0_reader']
        # G3
        fb, fa = fx['before'], fx['after']
        newly_failing = sorted(p for p in fb['results'] if fb['results'][p]['outcome'] == 'ok'
                               and fa['results'][p]['outcome'] != 'ok')
        differ = sorted(p for p in fb['results'] if fb['results'][p]['outcome'] != fa['results'][p]['outcome'])
        outcome_counts = {t: {} for t in ('before', 'after')}
        for t, r in (('before', fb), ('after', fa)):
            for v in r['results'].values():
                key_o = v['outcome'] if v['outcome'] in ('ok', 'ABSENT') else 'error: ' + v['outcome'][:100]
                outcome_counts[t][key_o] = outcome_counts[t].get(key_o, 0) + 1
        gates['G3'] = {'n_declared': len(fixtures), 'n_present': {'before': fb['n_present'], 'after': fa['n_present']},
                       'n_loads': {'before': fb['n_loads'], 'after': fa['n_loads']}, 'outcome_counts': outcome_counts,
                       'newly_failing': newly_failing, 'outcome_differs': differ,
                       'guards': {'before': fb['guard'], 'after': fa['guard']},
                       'modules_from_tree': {'before': all(v.startswith(fb['tree']) for v in fb['modules'].values()),
                                             'after': all(v.startswith(fa['tree']) for v in fa['modules'].values())},
                       'non_ok_paths_after': {p: v['outcome'] for p, v in fa['results'].items() if v['outcome'] != 'ok'}}
        gates['G3']['pass'] = (not newly_failing and not differ and fb['n_loads'] == fb['n_present']
                               and fa['n_loads'] == fa['n_present'] and fb['n_declared'] == len(fixtures)
                               and fa['n_declared'] == len(fixtures)
                               and all(gates['G3']['modules_from_tree'].values())
                               and not fb['guard']['verify_0_failures'] and not fa['guard']['verify_0_failures'])
        gates['G3']['all_declared_load_literal'] = all(
            v['outcome'] == 'ok' for v in fa['results'].values())
        # G3c
        g3c = children['after'].get('g3c') or {}
        g3c_cmp = {}
        for cell, c in (g3c.get('cells') or {}).items():
            base_cell = cell.replace('__g3c_unpickled_params', '')
            ref = children['after']['grids'][spec['g3c']['grid']]['cells'][base_cell + '__default']
            for node, files in c['init'].items():
                g3c_cmp[f'{base_cell}|init|{node}'] = all(files[f]['sha256'] == ref['init'][node][f]['sha256']
                                                          for f in NL_FILES)
        gates['G3c'] = {'pickled_by_before': children['before'].get('g3c_pickle'),
                        'loaded_type': g3c.get('loaded_type'), 'class_module_file': g3c.get('class_module_file'),
                        'has_key_attr_after_unpickle': g3c.get('has_key_attr_after_unpickle'),
                        'init_nl_bitwise_vs_after_default': g3c_cmp,
                        'pass': (g3c.get('has_key_attr_after_unpickle') is False and bool(g3c_cmp)
                                 and len(g3c_cmp) == spec['g3c']['n_node_models'] and all(g3c_cmp.values())
                                 and str(g3c.get('class_module_file', '')).startswith(trees['after']['path']))}
        # G4
        g4 = {}
        for grid in spec['grids']:
            row = {}
            for label, tag, pen in (('before_1e3', 'before', 'default'), ('after_1e3', 'after', 'default'),
                                    ('after_1e4', 'after', 'key_1e4')):
                nodes = children[tag]['g4'][grid][pen]
                ratios = {n: v['ratio'] for n, v in nodes.items()}
                peaks = {n: v['peak'] for n, v in nodes.items()}
                row[label] = {'penalty_in_force': sorted({v['penalty_in_force'] for v in nodes.values()}),
                              'bar': sorted({v['bar'] for v in nodes.values()}),
                              'kappa': sorted({v['kappa'] for v in nodes.values()}),
                              'peak_cohort': sorted({v['peak_cohort'] for v in nodes.values()}),
                              'peak': max(peaks.values()), 'ratio': max(ratios.values()),
                              'per_node_ratio': ratios,
                              'per_cohort_marginal_node7': {y: c['marginal'] for y, c in nodes['7']['cohorts'].items()}}
            ref = spec['g4']['reference_peaks'][grid]
            row['reference_peak'] = ref
            row['peak_rel_dev_vs_reference'] = abs(row['after_1e3']['peak'] - ref['value']) / ref['value']
            row['prediction_ratio_lt_1_at_1e4'] = row['after_1e4']['ratio'] < 1.0
            g4[grid] = row
        gates['G4'] = {'definition': spec['g4']['definition'], 'per_grid': g4,
                       'prediction_held': all(v['prediction_ratio_lt_1_at_1e4'] for v in g4.values())}
    except Exception:  # noqa: BLE001 -- recorded, never hidden
        import traceback
        result['errors']['parent_analysis'] = traceback.format_exc()
        print(result['errors']['parent_analysis'], file=sys.stderr, flush=True)
    gates = result['gates']
    gated = ('children_ok', 'G0_reader', 'G1', 'G1c_determinism', 'G2', 'G3', 'G3c')
    verdict = {}
    for g in gated:
        v = gates.get(g)
        verdict[g] = (v is True) if isinstance(v, bool) else bool(isinstance(v, dict) and v.get('pass') is True)
    result['verdict'] = verdict
    result['predictions'] = spec['predictions']
    result['all_gates_pass'] = all(verdict.values()) and not result['errors']
    # preserve: child JSONs, NL files (gzip, mtime 0), fixture records
    inv = {}
    for tag, w in works.items():
        for f in sorted(os.listdir(w)):
            if f.startswith('child_') and f.endswith('.json'):  # not the scratch params copy carrying the key
                shutil.copyfile(os.path.join(w, f), os.path.join(out, f))
        if os.path.isdir(os.path.join(w, 'nl')):
            inv[tag] = _gzip_tree(os.path.join(w, 'nl'), os.path.join(out, 'nl', tag))
    result['nl_inventory_sha256_uncompressed'] = inv
    result['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    result['finished_utc'] = _utc()
    result['wall_s'] = time.time() - t_start
    _dump(result, os.path.join(out, 'r8_key_gates.json'))
    _log(f"verdict {verdict}; all pass {result['all_gates_pass']}; errors {sorted(result['errors'])}; "
         f"guard {dict(GUARD.counts)}")
    if 'G4' in gates:
        for grid, row in gates['G4']['per_grid'].items():
            _log(f"G4 {grid}: ratio 1e3 {row['after_1e3']['ratio']:.6g} (before {row['before_1e3']['ratio']:.6g}), "
                 f"1e4 {row['after_1e4']['ratio']:.6g}; peak {row['after_1e3']['peak']:.10g} cohort "
                 f"{row['after_1e3']['peak_cohort']}")
    return 0 if result['all_gates_pass'] and not result['guard']['verify_0_failures'] else 1


def manifest(args):
    import gate_result_io as GRIO
    out = os.path.abspath(args.out)
    target = os.path.join(out, 'manifest_sha256.json')
    if os.path.exists(target):
        raise SystemExit(f'refusing to overwrite {target}')
    files = {}
    for root, _dirs, names in os.walk(out):
        for name in sorted(names):
            path = os.path.join(root, name)
            files[os.path.relpath(path, HERE)] = _sha_file(path)
    parent_dir = os.path.dirname(out)
    for name in sorted(os.listdir(parent_dir)):
        if name.startswith(os.path.basename(out) + '_launch'):
            files[os.path.relpath(os.path.join(parent_dir, name), HERE)] = _sha_file(os.path.join(parent_dir, name))
    files[SCRIPT] = _sha_file(os.path.join(HERE, SCRIPT))
    with open(target, 'x') as handle:
        handle.write(GRIO.dumps({'schema': SCHEMA + '_manifest', 'utc': _utc(), 'files': files}, indent=1,
                                sort_keys=True))
    print(f'manifest: {len(files)} files -> {target}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', choices=('build', 'fixtures'))
    ap.add_argument('--manifest', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--out')
    ap.add_argument('--scratch')
    ap.add_argument('--base-commit')
    ap.add_argument('--tag')
    ap.add_argument('--work')
    ap.add_argument('--g3c-dir')
    ap.add_argument('--fixture-list')
    ap.add_argument('--repo')
    ap.add_argument('--rehearsal', action='store_true')
    args = ap.parse_args()
    if args.manifest:
        return manifest(args)
    if args.child == 'build':
        return child_build(args)
    if args.child == 'fixtures':
        return child_fixtures(args)
    return parent(args)


if __name__ == '__main__':
    sys.exit(main())
