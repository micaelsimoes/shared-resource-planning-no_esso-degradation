"""P5.15 S56 -- isolated ESSO validation of the storage-cohort formulation (arrangement A; S56_COHORT_FORMULATION_PROPOSAL.md
rows 1-12, validation i-vii; Expert rulings E1-E7 recorded in TASKS.md S56). ESSO only: no network model is solved.

THE CHANGE UNDER TEST (working tree at launch; committed only after T-i passes):
  shared_energy_storage_parameters.py  optional top-level ESS-params key 'cohort_formulation' ('legacy_h3' default when
                                       absent | 'independent_units'), read strictly.
  shared_energy_storage_data.py        `_build_subproblem` + `_build_esso_cohort_units_rows` +
                                       `_configure_esso_cohort_units_rows` (arrangement A, built only under
                                       'independent_units'; H3 rows unwired there).
Detectors: `p515_s56_cohort_diagnostics.py` (committed with this script, so identical in both trees).

TREES (as `p515_s56_r8_key_gates.py`): `before` = `git archive <BASE>` of every tracked root-level *.py + tracked
data/SRP1 inputs (Results excluded); `after` = the same + the two production files overlaid from the working tree (sha256
recorded). The live working tree is never imported (another Worker edits five other production files there). The
five-block 3 x 3 case file is copied from BASE into each tree. Each tree runs in its own child process.

CHILDREN AND GUARDS. `SolveProfileGuard` installed before any project import in every process:
  parent, build children, fixture children: permitted=() and verified at exactly 0 solves;
  the solve child: permitted=(('shared_energy_storage_data.py', '_run_solver_attempt'),), verified at EXACTLY the
  spec's declared count (spec['solves']['declared_count']); a recovery retry would make it fail, loudly.

TESTS (spec = the frozen JSON; predictions are in the spec, recorded before launch):
  T-i   zero-solve NL bitwise (plain .nl labels off + labelled .nl/.row/.col): legacy before vs after (G1) and its
        determinism control (G1c); 'independent_units' vs legacy on the single-cohort plans (measurement + listing).
  G0    reader of the key (after tree, scratch params copies).
  G3    fixtures: every declared preserved pickle loads with the same outcome under before and after modules.
  G3c   a params object and a built ESSO model pickled by the before child are unpickled and used by the after child.
  T-ib  zero-solve row inventory, 2025 + 2030 plan.
  T-vii zero-solve expiry build, shortened calendar life of the 2025 cohort.
  T-ii, T-iii, T-iv, T-v  ESSO solves (node 7 only; the solve child sets active_distribution_network_nodes = [7]).

OUTPUT (write-once): `--out` must NOT exist. Launch (attached, alone, both streams captured; spec + script committed
first as BASE):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_cohort_esso_tests.py \\
      --spec data/SRP1/Results/P515S56/frozen_cohort_esso_spec_v1_<h>.json \\
      --out data/SRP1/Results/P515S56/cohort_esso_v1 --scratch <scratchpad>/cohort_esso_v1 --base-commit <BASE> \\
      > data/SRP1/Results/P515S56/cohort_esso_v1_launch_stdout.log 2> data/SRP1/Results/P515S56/cohort_esso_v1_launch_stderr.log
then  ... p515_s56_cohort_esso_tests.py --manifest --out data/SRP1/Results/P515S56/cohort_esso_v1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = '--child' in sys.argv
CHILD_KIND = sys.argv[sys.argv.index('--child') + 1] if CHILD else None
if CHILD:  # the tree this file was exported into is the ONLY project source on sys.path
    sys.path[:] = [HERE] + [p for p in sys.path[1:] if os.path.abspath(p or '.') != HERE]
elif HERE not in sys.path:
    sys.path.insert(0, HERE)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

SOLVE_SITE = ('shared_energy_storage_data.py', '_run_solver_attempt')
GUARD = SolveProfileGuard(permitted=(SOLVE_SITE,) if CHILD_KIND == 'solves' else (),
                          label=f"P5.15 S56 cohort ESSO tests ({CHILD_KIND or 'parent'})").install()

import argparse  # noqa: E402
import ast  # noqa: E402
import difflib  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

SCHEMA = 'p515_s56_cohort_esso_tests_v1'
SCRIPT = 'p515_s56_cohort_esso_tests.py'
DIAG = 'p515_s56_cohort_diagnostics.py'
PROD_FILES = ('shared_energy_storage_parameters.py', 'shared_energy_storage_data.py')
KEY = 'cohort_formulation'
INDEP = 'independent_units'
LEGACY = 'legacy_h3'
PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
W183C = 'p515_s53_w183c_nrf_gates.py'
FROZEN_SMOPF_DIR = os.path.join('data', 'SRP1', 'Results', 'FrozenSMOPF')
NL_FILES = ('plain', 'sym', 'row', 'col')
NEW_ROW_FAMILIES = ('energy_storage_cohort_soc', 'energy_storage_cohort_soc_window', 'energy_storage_cohort_soc_final',
                    'energy_storage_cohort_converter', 'energy_storage_operation_agg_q')
NEW_VAR_FAMILIES = ('es_soc_per_unit', 'es_q_per_unit', 'slack_es_soc_final_per_unit_up',
                    'slack_es_soc_final_per_unit_down', 'slack_es_qnet_up', 'slack_es_qnet_down')


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _log(msg):
    print(f'[cohort {datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


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
#  CHILD helpers
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
    return planning, {'read_s': time.time() - t0,
                      'scenario_checksum': (planning.scenario_metadata or {}).get('combined_scenario_checksum')}


def _candidate(planning, plan):
    import shared_resources_planning as srp
    cand = planning.get_initial_candidate_solution()
    for node, year, s, e in plan:
        cand['investment'][node][year]['s'] = s
        cand['investment'][node][year]['e'] = e
    srp._rebuild_candidate_total_capacities(planning, cand)
    return cand


def _plan_hash(plan):
    return _sha_bytes(json.dumps(sorted([list(x) for x in plan])).encode())[:16]


def _write_nl(model, nl_dir, stem):
    paths = {'plain': os.path.join(nl_dir, f'{stem}.nl'), 'sym': os.path.join(nl_dir, f'{stem}.sym.nl'),
             'row': os.path.join(nl_dir, f'{stem}.sym.row'), 'col': os.path.join(nl_dir, f'{stem}.sym.col')}
    model.write(paths['plain'], format='nl', io_options={'symbolic_solver_labels': False})
    model.write(paths['sym'], format='nl', io_options={'symbolic_solver_labels': True})
    return {k: {'path': v, 'sha256': _sha_file(v), 'bytes': os.path.getsize(v)} for k, v in paths.items()}


def _set_mode(sed, mode):
    """mode None = leave the params object as read (before tree: no attribute; after tree: the default)."""
    if mode is None:
        return '<as read>'
    sed.params.cohort_formulation = mode
    return mode


def _build_cell(planning, plan, in_objective, mode, nl_dir, cell):
    """One ESSO build through production (as R8 `_build_cell`): `create_shared_energy_storage_model` with the instance's
    `optimize` stubbed (NL written inside the stub = what the initial solve would receive), then
    `update_shared_energy_storage_model_to_admm` with production's al-scale and sigma (ADMM Params set to 0.0)."""
    import shared_resources_planning as srp
    sed = planning.shared_ess_data
    sv = sed.params.salvage_value
    saved_obj = sv.in_objective
    saved_mode = sed.params.__dict__.get(KEY, '<absent>')
    sv.in_objective = in_objective
    out = {'init': {}, 'admm': {}, 'stub_calls': 0, 'mode': _set_mode(sed, mode), 'plan': plan,
           'plan_hash': _plan_hash(plan)}
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
            sed.__dict__.pop('optimize', None)
        admm = planning.params.admm
        sigma = admm.objective_scale
        with redirect_stdout(io.StringIO()):
            a_scale, applied = srp._resolve_esso_al_scale(planning, admm, sigma)
        srp.update_shared_energy_storage_model_to_admm(planning, esso, admm, al_scale_esso=a_scale, objective_scale=sigma)
        for node_id, model in esso.items():
            for name in ('p_req', 'q_req', 'dual_p_req', 'dual_q_req'):
                comp = getattr(model, name)
                for idx in comp.index_set():
                    comp[idx] = 0.0
            out['admm'][str(node_id)] = _write_nl(model, nl_dir, f'{cell}__admm__n{node_id}')
            out.setdefault('has_registry', {})[str(node_id)] = hasattr(model, '_esso_cohort_units_rows')
            out.setdefault('multi_years', {})[str(node_id)] = getattr(model, '_esso_cohort_units_multi_years', None)
    finally:
        sv.in_objective = saved_obj
        if saved_mode == '<absent>':
            sed.params.__dict__.pop(KEY, None)
        else:
            sed.params.cohort_formulation = saved_mode
    return out


def _build_direct(planning, plan, mode):
    """build_subproblem + update_model_with_candidate_solution (the first two steps of
    `create_shared_energy_storage_model`); returns the node models."""
    sed = planning.shared_ess_data
    _set_mode(sed, mode)
    cand = _candidate(planning, plan)
    sed.update_data_with_candidate_solution(cand['investment'])
    models = sed.build_subproblem()
    sed.update_model_with_candidate_solution(models, cand['investment'])
    return models, cand


def _inventory(model, years):
    """Active rows per family (and per block for the new families / aggregate circle), free new Vars."""
    import pyomo.environ as pe
    fam = {}
    for con in model.component_data_objects(pe.Constraint, active=True):
        name = con.parent_component().name
        fam[name] = fam.get(name, 0) + 1
    out = {'active_rows_by_family': fam}
    reg = getattr(model, '_esso_cohort_units_rows', None)
    if reg is not None:
        by_year = {}
        for (y_inv, y), rows in reg['cohort'].items():
            for name, idx in rows:
                if getattr(model, name)[idx].active:
                    key = f'{name}|{years[y]}'
                    by_year[key] = by_year.get(key, 0) + 1
        for y, rows in reg['aggregate'].items():
            for name, idx in rows:
                if getattr(model, name)[idx].active:
                    key = f'{name}|{years[y]}'
                    by_year[key] = by_year.get(key, 0) + 1
        out['new_rows_active_by_block'] = by_year
        out['aggregate_circle_active_by_block'] = {
            str(years[y]): sum(1 for idx in idxs if model.energy_storage_operation_agg[idx].active)
            for y, idxs in reg['aggregate_circle'].items()}
        out['aggregate_circle_built_by_block'] = {str(years[y]): len(idxs) for y, idxs in reg['aggregate_circle'].items()}
        out['n_pairs_built'] = len(reg['pairs'])
    free = {}
    for name in NEW_VAR_FAMILIES:
        if hasattr(model, name):
            comp = getattr(model, name)
            free[name] = sum(1 for v in comp.values() if not v.fixed)
    out['new_vars_free'] = free
    out['h3_rows_built'] = len(model.energy_storage_cohort_pnet_share_h3)
    out['h3_rows_active'] = sum(1 for c in model.energy_storage_cohort_pnet_share_h3.values() if c.active)
    return out


def _reader_tests(work, base_params_path):
    from shared_energy_storage_parameters import SharedEnergyStorageParameters
    with open(base_params_path) as handle:
        base = json.load(handle)
    ok_cases = {'absent': ('<absent>', LEGACY), 'legacy_h3': (LEGACY, LEGACY), 'independent_units': (INDEP, INDEP)}
    bad = {'bool_true': True, 'null': None, 'int_1': 1, 'H3': 'H3', 'empty': '', 'upper': 'INDEPENDENT_UNITS',
           'list': [INDEP], 'object': {}}
    res = {'ok': {}, 'raises': {}}
    os.makedirs(work)
    for name, (value, expected) in ok_cases.items():
        data = dict(base)
        if value != '<absent>':
            data[KEY] = value
        path = os.path.join(work, f'params_{name}.json')
        with open(path, 'x') as handle:
            json.dump(data, handle)
        p = SharedEnergyStorageParameters()
        p.read_parameters_from_file(path)
        got = getattr(p, KEY)
        res['ok'][name] = {'value': got, 'pass': got == expected}
    for name, value in bad.items():
        data = dict(base)
        data[KEY] = value
        path = os.path.join(work, f'params_{name}.json')
        with open(path, 'x') as handle:
            json.dump(data, handle)
        p = SharedEnergyStorageParameters()
        try:
            p.read_parameters_from_file(path)
            res['raises'][name] = {'raised': False, 'pass': False, 'value_read': repr(getattr(p, KEY))}
        except ValueError as error:
            res['raises'][name] = {'raised': True, 'pass': True, 'error': str(error)}
    res['default_attribute'] = getattr(SharedEnergyStorageParameters(), KEY)
    res['pass'] = (all(v['pass'] for v in res['ok'].values()) and all(v['pass'] for v in res['raises'].values())
                   and res['default_attribute'] == LEGACY)
    return res


# ======================================================================================================================
#  CHILD: builds
# ======================================================================================================================
def child_build(args):
    spec = json.load(open(args.spec))
    import definitions as DEFS  # noqa: F401
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    import shared_resources_planning as SRP
    import model_construction_helpers as MCH
    mods = _assert_modules_from_tree([SED, SEP, SRP, MCH])
    tag = args.tag
    is_after = tag.startswith('after')
    work = args.work
    nl_root = os.path.join(work, 'nl')
    rec = {'schema': SCHEMA + '_child_build', 'tag': tag, 'tree': HERE, 'modules': mods, 'started_utc': _utc(),
           'interpreter': sys.executable,
           'params_default_has_key_attr': hasattr(SEP.SharedEnergyStorageParameters(), KEY),
           'grids': {}, 'errors': {}}
    modes = {'legacy': None}
    params_path = os.path.join(HERE, 'data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
    if is_after:
        rec['g0_reader'] = _reader_tests(os.path.join(work, 'g0_reader'), params_path)
        p = SEP.SharedEnergyStorageParameters()
        data = json.load(open(params_path))
        data[KEY] = INDEP
        scratch_params = os.path.join(work, 'params_key_independent_units.json')
        with open(scratch_params, 'x') as handle:
            json.dump(data, handle)
        p.read_parameters_from_file(scratch_params)
        rec['indep_parsed'] = getattr(p, KEY)
        modes['indep'] = rec['indep_parsed']
    for grid, gspec in spec['grids'].items():
        planning, info = _read_planning(gspec['case_file'], os.path.join(work, 'side', grid))
        sed = planning.shared_ess_data
        years = list(sed.years)
        info['years'] = {str(y): sed.years[y] for y in sed.years}
        info['nodes'] = list(sed.active_distribution_network_nodes)
        info['slacks'] = sed.params.slacks
        info['soh_point'] = SED._esso_ageing_model_settings(sed)[0]
        info['params_key_attr_as_read'] = sed.params.__dict__.get(KEY, '<absent>')
        info['checksum_matches_spec'] = info['scenario_checksum'] == gspec['scenario_checksum']
        rec['grids'][grid] = {'read': info, 'cells': {}}
        for plan_name, plan in gspec['plans'].items():
            plan_t = [tuple(x) for x in plan]
            for in_obj in (False, True):
                for mode_name, mode in modes.items():
                    cell = f'{grid}__{plan_name}__inobj{int(in_obj)}__{mode_name}'
                    nl_dir = os.path.join(nl_root, cell)
                    os.makedirs(nl_dir)
                    try:
                        rec['grids'][grid]['cells'][cell] = _build_cell(planning, plan_t, in_obj, mode, nl_dir, cell)
                    except Exception:  # noqa: BLE001 -- recorded
                        rec['errors'][cell] = traceback.format_exc()
        if grid == 'srp1':
            # G3c (before: pickle; after: unpickle and use)
            if tag == 'before':
                g3c = {}
                ppath = os.path.join(args.g3c_dir, 'params_pickled_by_before.pkl')
                with open(ppath, 'xb') as handle:
                    pickle.dump(sed.params, handle)
                g3c['params'] = {'path': ppath, 'sha256': _sha_file(ppath), 'has_key_attr': KEY in sed.params.__dict__}
                plan_t = [tuple(x) for x in spec['tib']['plan']]
                models, cand = _build_direct(planning, plan_t, None)
                mpath = os.path.join(args.g3c_dir, 'esso_model_n7_pickled_by_before.pkl')
                nl_dir = os.path.join(nl_root, 'g3c_model_before')
                os.makedirs(nl_dir)
                g3c['model_nl_before_pickle'] = _write_nl(models[7], nl_dir, 'g3c_model_before')
                with open(mpath, 'xb') as handle:
                    pickle.dump(models[7], handle)
                g3c['model'] = {'path': mpath, 'sha256': _sha_file(mpath), 'plan': plan_t}
                rec['g3c_pickles'] = g3c
            elif tag == 'after':
                g3c = {'cells': {}}
                try:
                    with open(os.path.join(args.g3c_dir, 'params_pickled_by_before.pkl'), 'rb') as handle:
                        old = pickle.load(handle)
                    g3c['params_class_file'] = os.path.abspath(sys.modules[type(old).__module__].__file__)
                    g3c['params_has_key_attr_after_unpickle'] = KEY in old.__dict__
                    g3c['formulation_resolved'] = SED._esso_cohort_formulation(type('X', (), {'params': old})())
                    saved = sed.params
                    sed.params = old
                    try:
                        for plan_name in ('x0', 'ref_unit', 'p2025_2030'):
                            plan = spec['g3c']['plans'][plan_name]
                            cell = f'srp1__{plan_name}__inobj0__g3c_unpickled_params'
                            nl_dir = os.path.join(nl_root, cell)
                            os.makedirs(nl_dir)
                            g3c['cells'][cell] = _build_cell(planning, [tuple(x) for x in plan], False, None, nl_dir, cell)
                    finally:
                        sed.params = saved
                    # the old ESSO model: update it with after modules, under independent_units in the params object
                    with open(os.path.join(args.g3c_dir, 'esso_model_n7_pickled_by_before.pkl'), 'rb') as handle:
                        old_model = pickle.load(handle)
                    g3c['model_has_registry_after_unpickle'] = hasattr(old_model, '_esso_cohort_units_rows')
                    sed.params.cohort_formulation = INDEP
                    try:
                        cand = _candidate(planning, [tuple(x) for x in spec['tib']['plan']])
                        SED._update_model_with_candidate_solution(sed, {7: old_model}, cand['investment'])
                        g3c['model_update_raised'] = None
                    except Exception:  # noqa: BLE001
                        g3c['model_update_raised'] = traceback.format_exc()
                    finally:
                        sed.params.cohort_formulation = LEGACY
                    nl_dir = os.path.join(nl_root, 'g3c_model_after_update')
                    os.makedirs(nl_dir)
                    g3c['model_nl_after_update'] = _write_nl(old_model, nl_dir, 'g3c_model_after_update')
                    g3c['model_has_registry_after_update'] = hasattr(old_model, '_esso_cohort_units_rows')
                except Exception:  # noqa: BLE001
                    rec['errors']['g3c'] = traceback.format_exc()
                rec['g3c'] = g3c
            if is_after:
                # T-ib inventory and T-vii expiry (independent_units), plus the legacy inventory of the same plan
                try:
                    tib = {}
                    for mode_name, mode in (('indep', INDEP), ('legacy', LEGACY)):
                        models, _c = _build_direct(planning, [tuple(x) for x in spec['tib']['plan']], mode)
                        tib[mode_name] = _inventory(models[7], years)
                    sed.params.cohort_formulation = LEGACY
                    rec['tib'] = tib
                except Exception:  # noqa: BLE001
                    rec['errors']['tib'] = traceback.format_exc()
                try:
                    rec['tvii'] = _t_vii(planning, spec, years)
                except Exception:  # noqa: BLE001
                    rec['errors']['tvii'] = traceback.format_exc()
        del planning
    rec['finished_utc'] = _utc()
    rec['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    _dump(rec, os.path.join(work, f'child_build_{tag}.json'))
    return 0


def _t_vii(planning, spec, years):
    """Expiry: the 2025 cohort's calendar life shortened in memory (t_cal on the cohort's SharedEnergyStorage objects
    for its installation year, all nodes), set after `update_data_with_candidate_solution` and before the build."""
    import shared_resources_planning as srp  # noqa: F401
    sed = planning.shared_ess_data
    tspec = spec['tvii']
    plan_t = [tuple(x) for x in tspec['plan']]
    sed.params.cohort_formulation = INDEP
    saved = {}
    try:
        cand = _candidate(planning, plan_t)
        sed.update_data_with_candidate_solution(cand['investment'])
        year = tspec['shortened_cohort_year']
        for ses in sed.shared_energy_storages[year]:
            saved[id(ses)] = (ses, ses.t_cal)
            ses.t_cal = tspec['t_cal_years']
        models = sed.build_subproblem()
        sed.update_model_with_candidate_solution(models, cand['investment'])
        m = models[7]
        out = _inventory(m, years)
        y_short = years.index(year)
        fixed_outside = {}
        for name in NEW_VAR_FAMILIES:
            if not hasattr(m, name):
                continue
            comp = getattr(m, name)
            n_free_outside = 0
            for idx, v in comp.items():
                if name in ('slack_es_qnet_up', 'slack_es_qnet_down'):
                    continue
                y_inv, y = idx[0], idx[1]
                lifetime = not m.es_s_rated_per_unit[y_inv, y].fixed
                if (not lifetime) and not v.fixed:
                    n_free_outside += 1
            fixed_outside[name] = n_free_outside
        out['new_vars_free_outside_lifetime'] = fixed_outside
        out['lifetime_pairs'] = sorted(f'{years[a]}|{years[b]}' for a in m.years for b in m.years
                                       if not m.es_s_rated_per_unit[a, b].fixed and pe_value(m.es_s_investment_fixed[a]) > 0)
        out['shortened_cohort_rated_fixed_by_block'] = {str(years[y]): m.es_s_rated_per_unit[y_short, y].fixed
                                                        for y in m.years}
        out['multi_years'] = [years[y] for y in getattr(m, '_esso_cohort_units_multi_years', [])]
    finally:
        for ses, tcal in saved.values():
            ses.t_cal = tcal
        sed.params.cohort_formulation = LEGACY
    return out


def pe_value(x):
    import pyomo.environ as pe
    return pe.value(x)


# ======================================================================================================================
#  CHILD: fixtures
# ======================================================================================================================
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
#  CHILD: solves (after tree only)
# ======================================================================================================================
def _phase_profile(phases, rate, start_p, n_periods):
    """Aggregate request (MW, + = charging) from consecutive phases [(sign, grid_energy_mwh), ...] delivered at `rate`
    MW per period of 1 h, each phase starting on a new period, the last period of a phase partial."""
    prof = {}
    p = start_p
    for sign, energy in phases:
        remaining = energy
        while remaining > 1e-15:
            step = min(rate, remaining)
            if p >= n_periods:
                raise ValueError('phase profile does not fit in the day')
            prof[p] = sign * step
            remaining -= step
            p += 1
    return prof


def _calendar_e(sed, node, year_inv, e_rated, year):
    """Calendar-only end-of-block available energy (an upper bound on E_av: cycling only lowers it)."""
    years = list(sed.years)
    idx = sed.get_shared_energy_storage_idx(node)
    ses = sed.shared_energy_storages[year_inv][idx]
    n = sed.years[year_inv]
    k = years.index(year) - years.index(year_inv) + 1
    return e_rated * ses.phi_cal ** (n * k)


def _result_record(res):
    if res is None:
        return {'result': None}
    try:
        return {'status': str(res.solver.status), 'termination': str(res.solver.termination_condition),
                'message': str(getattr(res.solver, 'message', '')),
                'succeeded': bool(_succeeded(res))}
    except Exception:  # noqa: BLE001
        return {'result_repr': repr(res)[:500]}


def _succeeded(res):
    from helper_functions import solver_result_succeeded
    return solver_result_succeeded(res)


def child_solves(args):
    spec = json.load(open(args.spec))
    import pyomo.environ as pe
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    import shared_resources_planning as SRP
    import model_construction_helpers as MCH
    import p515_s56_cohort_diagnostics as DG
    from helper_functions import fix_or_set
    mods = _assert_modules_from_tree([SED, SEP, SRP, MCH, DG])
    work = args.work
    sspec = spec['solves']
    rec = {'schema': SCHEMA + '_child_solves', 'tree': HERE, 'modules': mods, 'started_utc': _utc(),
           'interpreter': sys.executable, 'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
           'tests': {}, 'errors': {}, 'solve_log': []}
    planning, info = _read_planning('SRP1.json', os.path.join(work, 'side'))
    sed = planning.shared_ess_data
    rec['read'] = info
    rec['read']['checksum_matches_spec'] = info['scenario_checksum'] == spec['grids']['srp1']['scenario_checksum']
    node = sspec['node']
    sed.active_distribution_network_nodes = [node]
    sed.logs_dir = os.path.join(work, 'ipopt_logs')
    os.makedirs(sed.logs_dir)
    years = list(sed.years)
    days = list(sed.days)
    n_p = sed.num_instants
    idx = sed.get_shared_energy_storage_idx(node)
    rec['efficiencies'] = {str(y): [sed.shared_energy_storages[y][idx].eff_ch, sed.shared_energy_storages[y][idx].eff_dch]
                           for y in years}
    rec['solver_options'] = {'solver_params': {k: str(v) for k, v in vars(sed.params.solver_params).items()},
                             'esso_overrides': dict(SED.ESSO_TOL_OVERRIDES)}
    counter = {'cycle': 0}

    def solve(models, label, from_warm_start=False):
        counter['cycle'] += 1
        cyc = counter['cycle']
        n_before = GUARD.counts['permitted_solve']
        t0 = time.time()
        stream = io.StringIO()
        with redirect_stdout(stream):
            res = sed.optimize(models, from_warm_start=from_warm_start, cycle=cyc)
        entry = {'label': label, 'cycle_stamp': cyc, 'wall_s': time.time() - t0,
                 'solves_this_call': GUARD.counts['permitted_solve'] - n_before, **_result_record(res[node]),
                 'stdout_tail': stream.getvalue()[-1500:]}
        diag = list(getattr(sed, 'esso_complementarity_diagnostics', []) or [])
        if diag:
            entry['complementarity_diagnostics_last'] = diag[-1]
        rec['solve_log'].append(entry)
        return entry

    def init_models(plan, mode):
        models, cand = _build_direct(planning, [tuple(x) for x in plan], mode)
        m = models[node]
        for y in m.years:
            for d in m.days:
                for p in m.periods:
                    fix_or_set(m.es_pnet[y, d, p], 0.0)
                    fix_or_set(m.es_qnet[y, d, p], 0.0)
        return models, m

    def apply_request(m, y, d, fixed, free):
        for p, val in fixed.items():
            fix_or_set(m.es_pnet[y, d, p], val)
        for p in free:
            m.es_pnet[y, d, p].unfix()

    def values_at(m, y, d, p):
        out = {'es_pnet': pe.value(m.es_pnet[y, d, p]), 'es_qnet': pe.value(m.es_qnet[y, d, p]),
               'agg_net': sum(pe.value(m.es_pch_per_unit[c, y, d, p] - m.es_pdch_per_unit[c, y, d, p])
                              for c in DG.active_cohorts(m, y)),
               'per_cohort_net': {str(years[c]): pe.value(m.es_pch_per_unit[c, y, d, p] - m.es_pdch_per_unit[c, y, d, p])
                                  for c in DG.active_cohorts(m, y)}}
        out.update(DG.slack_at(m, y, d, p))
        return out

    def objective_record(m):
        out = {'objective': pe.value(m.objective) if m.objective.active else None,
               'feasibility_penalty': pe.value(m.feasibility_penalty)}
        if hasattr(m, 'admm_objective'):
            out['admm_objective'] = pe.value(m.admm_objective)
        return out

    def run(name, fn):
        try:
            rec['tests'][name] = fn()
        except Exception:  # noqa: BLE001 -- recorded, never hidden; the guard count then shows what ran
            rec['errors'][name] = traceback.format_exc()
            print(rec['errors'][name], file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- T-ii
    tii = sspec['t_ii']

    def t_ii_a(mode):
        def fn():
            y = years.index(tii['block'])
            d = days.index(tii['day'])
            models, m = init_models(tii['plan'], mode)
            p0 = tii['a']['discharge_period']
            apply_request(m, y, d, {p0: tii['a']['request_mw']}, [p for p in m.periods if p != p0])
            entry = solve(models, f't_ii_a_{mode}')
            at = values_at(m, y, d, p0)
            out = {'mode': mode, 'block': tii['block'], 'day': tii['day'], 'period': p0, 'solve': entry,
                   'at_period': at, 'delivered_mw': -at['agg_net'],
                   'p_slack_mw': at['p_up'] + at['p_down'], **objective_record(m),
                   'detectors': DG.node_report(m, sed, node, blocks=[(y, d)])}
            # hand formula on the solved E_av: sum_c min(s_c, (INIT - MIN) * E_av_c * eta_dch / dt)
            hand = 0.0
            for c in DG.active_cohorts(m, y):
                eta_dch = sed.shared_energy_storages[years[c]][idx].eff_dch
                hand += min(pe.value(m.es_s_rated_per_unit[c, y]),
                            (0.5 - 0.1) * pe.value(m.es_e_available_per_unit[c, y]) * eta_dch)
            out['hand_formula_on_solved_e_av_mw'] = hand
            return out
        return fn

    def t_ii_b():
        y = years.index(tii['block'])
        d = days.index(tii['day'])
        models, m = init_models(tii['plan'], INDEP)
        pk = tii['b']['discharge_period']
        apply_request(m, y, d, {pk: tii['b']['request_mw']}, [p for p in m.periods if p != pk])
        entry = solve(models, 't_ii_b')
        at = values_at(m, y, d, pk)
        rep = DG.node_report(m, sed, node, blocks=[(y, d)])
        bd = rep['block_day'][f'{tii["block"]}|{tii["day"]}']
        c2 = str(years.index(tii['b']['precharged_cohort']))
        soc = bd['per_cohort'][c2]['soc_mwh']
        e_av = bd['per_cohort'][c2]['e_available_mwh']
        eta_dch = sed.shared_energy_storages[tii['b']['precharged_cohort']][idx].eff_dch
        s_c = bd['per_cohort'][c2]['s_rated_mva']
        return {'solve': entry, 'at_period': at, 'delivered_mw': -at['agg_net'], 'p_slack_mw': at['p_up'] + at['p_down'],
                **objective_record(m), 'detectors': rep,
                'precharged_cohort_stored_rise_before_peak_mwh': soc[pk - 1] - 0.5 * e_av,
                'precharged_cohort_min_rise_needed_mwh': max(0.0, s_c / eta_dch - (0.5 - 0.1) * e_av),
                'precharged_cohort_window_headroom_mwh': (0.9 - 0.5) * e_av}

    # ---------------------------------------------------------------- T-iii
    tiii = sspec['t_iii']

    def t_iii(mode):
        def fn():
            y = years.index(tiii['block'])
            d = days.index(tiii['day'])
            plan = [tuple(x) for x in tiii['plan']]
            e_cal = sum(_calendar_e(sed, node, yi, e, tiii['block']) for (_n, yi, _s, e) in plan)
            etas = {(sed.shared_energy_storages[yi][idx].eff_ch, sed.shared_energy_storages[yi][idx].eff_dch)
                    for (_n, yi, _s, _e) in plan}
            if len(etas) != 1:
                raise ValueError(f'T-iii assumes common efficiencies, got {etas}')
            eta_ch, eta_dch = etas.pop()
            swing = (0.5 - 0.1) * tiii['fraction_f'] * e_cal
            prof = _phase_profile([(-1, swing * eta_dch), (+1, 2 * swing / eta_ch), (-1, swing * eta_dch)],
                                  tiii['rate_mw'], tiii['start_period'], n_p)
            models, m = init_models(tiii['plan'], mode)
            fixed = {p: prof.get(p, 0.0) for p in m.periods}
            apply_request(m, y, d, fixed, [])
            entry = solve(models, f't_iii_{mode}')
            rep = DG.node_report(m, sed, node, blocks=[(y, d)])
            bd = rep['block_day'][f'{tiii["block"]}|{tiii["day"]}']
            return {'mode': mode, 'solve': entry, 'calendar_sum_e_mwh': e_cal, 'swing_stored_mwh': swing,
                    'request_profile_mw': {str(p): v for p, v in sorted(fixed.items())}, **objective_record(m),
                    'detectors': rep,
                    'per_cohort_window_fraction': {c: [r['window_fraction_min'], r['window_fraction_max']]
                                                   for c, r in bd['per_cohort'].items()},
                    'p_slack_max_mw_block_day': max(DG.slack_at(m, y, d, p)['p_up'] + DG.slack_at(m, y, d, p)['p_down']
                                                    for p in m.periods)}
        return fn

    # ---------------------------------------------------------------- T-iv
    tiv = sspec['t_iv']

    def t_iv(kind):
        def fn():
            y = years.index(tiv['block'])
            d = days.index(tiv['day'])
            plan = [tuple(x) for x in tiv['plan']]
            # zero-solve peak (calendar-only E; formula recorded in the spec)
            (_n1, ya, sa, ea), (_n2, yb, sb, eb) = plan
            ea_cal = _calendar_e(sed, node, ya, ea, tiv['block'])
            eb_cal = _calendar_e(sed, node, yb, eb, tiv['block'])
            eta_ch = sed.shared_energy_storages[yb][idx].eff_ch
            eta_dch = sed.shared_energy_storages[ya][idx].eff_dch
            no_transfer_max = min(sa, 0.4 * ea_cal * eta_dch) + min(sb, 0.4 * eb_cal * eta_dch)
            spare_a = 0.4 * ea_cal - sa / eta_dch  # stored energy A can give beyond its own peak share
            gain_b = max(0.0, spare_a) * eta_dch * eta_ch
            with_transfer_max = min(sa, 0.4 * ea_cal * eta_dch) + min(sb, (0.4 * eb_cal + gain_b) * eta_dch)
            peak = 0.5 * (no_transfer_max + with_transfer_max)
            models, m = init_models(tiv['plan'], INDEP)
            pk = tiv['peak_period']
            if kind == 'peak':
                fixed = {p: 0.0 for p in m.periods if p < pk}
                fixed[pk] = -peak
                free = [p for p in m.periods if p > pk]
            else:
                fixed = {p: 0.0 for p in m.periods}
                free = []
            apply_request(m, y, d, fixed, free)
            entry = solve(models, f't_iv_{kind}')
            rep = DG.node_report(m, sed, node, blocks=[(y, d)])
            bd = rep['block_day'][f'{tiv["block"]}|{tiv["day"]}']
            out = {'kind': kind, 'solve': entry, 'request_fixed_mw': {str(p): v for p, v in sorted(fixed.items())},
                   'request_free_periods': free, 'zero_solve_bounds': {
                       'no_transfer_max_mw': no_transfer_max, 'with_transfer_max_mw': with_transfer_max,
                       'peak_mw': peak, 'ea_calendar_mwh': ea_cal, 'eb_calendar_mwh': eb_cal},
                   **objective_record(m), 'detectors': rep,
                   'inter_cohort_gross_transfer_mwh': bd['inter_cohort_gross_transfer_mwh'],
                   'conversion_loss_mwh': bd['conversion_loss_mwh'],
                   'within_cohort_simultaneity_mwh': {c: r['within_cohort_simultaneity_mwh']
                                                      for c, r in bd['per_cohort'].items()},
                   'closure_residual_mwh': {c: r['closure_residual_mwh'] for c, r in bd['per_cohort'].items()},
                   'energy_accounting': bd['energy_accounting'],
                   's_aggregate_mva': sum(r['s_rated_mva'] for r in bd['per_cohort'].values())}
            if kind == 'peak':
                out['at_peak'] = values_at(m, y, d, pk)
                out['p_slack_at_peak_mw'] = out['at_peak']['p_up'] + out['at_peak']['p_down']
                out['no_transfer_slack_bound_mw'] = peak - no_transfer_max
            return out
        return fn

    # ---------------------------------------------------------------- T-v
    tv = sspec['t_v']

    def t_v(tv_on, start):
        def fn():
            sv = sed.params.salvage_value
            saved_obj = sv.in_objective
            sv.in_objective = tv_on
            sed.params.cohort_formulation = INDEP
            try:
                cand = _candidate(planning, [tuple(x) for x in tv['plan']])
                consensus_vars, _dual = SRP.create_admm_variables(planning)
                stub_calls = {'n': 0}

                def stub(models, *a, **k):
                    stub_calls['n'] += 1
                    return {n: None for n in models}

                sed.optimize = stub
                try:
                    with redirect_stdout(io.StringIO()):
                        esso, _r = SRP.create_shared_energy_storage_model(sed, consensus_vars, cand['investment'])
                finally:
                    sed.__dict__.pop('optimize', None)
                admm = planning.params.admm
                sigma = admm.objective_scale
                with redirect_stdout(io.StringIO()):
                    a_scale, applied = SRP._resolve_esso_al_scale(planning, admm, sigma)
                SRP.update_shared_energy_storage_model_to_admm(planning, esso, admm, al_scale_esso=a_scale,
                                                              objective_scale=sigma)
            finally:
                sv.in_objective = saved_obj
            m = esso[node]
            prof = {int(k): v for k, v in tv['p_req_profile_mw'].items()}
            for y in m.years:
                for d in m.days:
                    for p in m.periods:
                        m.p_req[y, d, p] = prof.get(p, 0.0)
                        m.q_req[y, d, p] = 0.0
                        m.dual_p_req[y, d, p] = 0.0
                        m.dual_q_req[y, d, p] = 0.0
            kappa = pe.value(m.salvage_objective_kappa) if hasattr(m, 'salvage_objective_kappa') else None
            if start in ('oldest', 'newest'):
                for y in m.years:
                    act = DG.active_cohorts(m, y)
                    if len(act) < 2:
                        continue
                    pick = min(act) if start == 'oldest' else max(act)
                    for d in m.days:
                        for p in m.periods:
                            r = prof.get(p, 0.0)
                            for c in act:
                                m.es_pch_per_unit[c, y, d, p].set_value(max(r, 0.0) if c == pick else 0.0)
                                m.es_pdch_per_unit[c, y, d, p].set_value(max(-r, 0.0) if c == pick else 0.0)
                            m.es_pnet[y, d, p].set_value(r)
            entry = solve(esso, f't_v_tv{int(tv_on)}_{start}')
            rep = DG.node_report(m, sed, node)
            thr = {}
            for key, v in rep['degradation'].items():
                thr[key] = v['avg_throughput_mwh_per_day']
            return {'tv_on': tv_on, 'start': start, 'kappa': kappa, 'al_scale': {'value': a_scale, 'applied': applied},
                    'sigma': sigma, 'stub_calls': stub_calls['n'], 'solve': entry,
                    'admm_objective': pe.value(m.admm_objective), 'feasibility_penalty': pe.value(m.feasibility_penalty),
                    'salvage_credit_obj': pe.value(m.salvage_credit_obj) if hasattr(m, 'salvage_credit_obj') else None,
                    'detectors': rep, 'per_cohort_throughput_mwh_per_day': thr,
                    'D_max_abs_diff': max(v['abs_diff'] for v in rep['degradation'].values()),
                    'tracking_max_abs_mw': max(abs(pe.value(m.es_pnet[y, d, p]) - prof.get(p, 0.0))
                                               for y in m.years for d in m.days for p in m.periods)}
        return fn

    order = [('t_ii_a_indep', t_ii_a(INDEP)), ('t_ii_a_legacy', t_ii_a(LEGACY)), ('t_ii_b', t_ii_b),
             ('t_iii_indep', t_iii(INDEP)), ('t_iii_legacy', t_iii(LEGACY)),
             ('t_iv_peak', t_iv('peak')), ('t_iv_control', t_iv('control')),
             ('t_v_tv_on', t_v(True, 'as_built')), ('t_v_tv_off_oldest', t_v(False, 'oldest')),
             ('t_v_tv_off_newest', t_v(False, 'newest'))]
    if [n for n, _f in order] != sspec['order']:
        raise SystemExit(f'refusing: solve order differs from the spec: {[n for n, _f in order]}')
    for name, fn in order:
        _log(f'solve test {name}')
        run(name, fn)
    sed.params.cohort_formulation = LEGACY
    rec['finished_utc'] = _utc()
    declared = sspec['declared_count']
    rec['guard'] = {'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                    'declared': declared, 'verify_failures': GUARD.verify(declared)}
    _dump(rec, os.path.join(work, 'child_solves.json'))
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
            'prod_sha256': {rel: _sha_file(os.path.join(dest, rel)) for rel in PROD_FILES + (DIAG,)
                            if os.path.isfile(os.path.join(dest, rel))}}


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


def _nl_header_segments(lines):
    """Segment headers of a labelled .nl (lines starting with a letter code), as a list of (line_no, text)."""
    return [(i, ln) for i, ln in enumerate(lines) if ln[:1].isalpha()]


def _nl_difference(a, b):
    """a, b: the four-file records of two node models. Bitwise per file; when they differ, the row/col sets and the
    first differing lines of the labelled NL."""
    res = {f: a[f]['sha256'] == b[f]['sha256'] for f in NL_FILES}
    if all(res.values()):
        return {'identical': True}
    ra, rb = _read_text(a['row']['path']), _read_text(b['row']['path'])
    ca, cb = _read_text(a['col']['path']), _read_text(b['col']['path'])
    sa, sb = _read_text(a['sym']['path']), _read_text(b['sym']['path'])
    diff = list(difflib.unified_diff(sa, sb, 'a', 'b', n=0, lineterm=''))
    return {'identical': False, 'per_file': res,
            'rows_only_in_a': sorted(set(ra) - set(rb))[:50], 'rows_only_in_b': sorted(set(rb) - set(ra))[:50],
            'n_rows': [len(ra), len(rb)], 'cols_only_in_a': sorted(set(ca) - set(cb))[:50],
            'cols_only_in_b': sorted(set(cb) - set(ca))[:50], 'n_cols': [len(ca), len(cb)],
            'row_order_same': ra == rb, 'col_order_same': ca == cb,
            'sym_header_a': sa[:2], 'sym_header_b': sb[:2], 'sym_diff_first_lines': diff[:60],
            'sym_n_diff_lines': len(diff)}


def _iter_nl(child, mode_name):
    out = {}
    for grid, g in child['grids'].items():
        for cell, c in g['cells'].items():
            if not cell.endswith(f'__{mode_name}'):
                continue
            base = cell[: -len(f'__{mode_name}')]
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


def _check(pred, value):
    """A prediction {'op': ..., 'value' | 'lo'/'hi'} against a measured value -> bool."""
    op = pred['op']
    if value is None:
        return False
    if op == 'le':
        return value <= pred['value']
    if op == 'ge':
        return value >= pred['value']
    if op == 'lt':
        return value < pred['value']
    if op == 'gt':
        return value > pred['value']
    if op == 'eq':
        return value == pred['value']
    if op == 'within':
        return abs(value - pred['value']) <= pred['tol']
    if op == 'true':
        return value is True
    raise ValueError(op)


def _evaluate(spec, children, fx, solves):
    """The analysis: every gate / prediction of the spec, against the children's records."""
    g = {}
    after = children['after']
    # children read-back
    g['children'] = {t: {'guard': c['guard'], 'modules_from_tree': all(v.startswith(c['tree']) for v in c['modules'].values()),
                         'checksums': {gr: v['read']['checksum_matches_spec'] for gr, v in c['grids'].items()},
                         'stub_calls_all_1': all(v['stub_calls'] == 1 for gr in c['grids'].values()
                                                 for v in gr['cells'].values()),
                         'errors': sorted(c['errors'])}
                     for t, c in children.items() if c}
    g['children_ok'] = all(
        v['guard']['counts']['blocked_solve'] == 0 and v['guard']['counts']['permitted_solve'] == 0
        and not v['guard']['verify_0_failures'] and v['modules_from_tree'] and all(v['checksums'].values())
        and v['stub_calls_all_1'] and not v['errors'] for v in g['children'].values()) and all(
        c is not None for c in children.values())
    # T-i: G1 legacy before vs after, G1c control
    before = _iter_nl(children['before'], 'legacy')
    g['G1_legacy_bitwise'] = _bitwise(before, _iter_nl(after, 'legacy'))
    g['G1_legacy_bitwise']['n_expected'] = spec['t_i']['n_node_models_per_mode']
    g['G1_legacy_bitwise']['pass'] = (g['G1_legacy_bitwise']['pass']
                                      and g['G1_legacy_bitwise']['n_compared'] == spec['t_i']['n_node_models_per_mode'])
    g['G1c_determinism'] = _bitwise(before, _iter_nl(children['before_rerun'], 'legacy'))
    # T-i: independent_units vs legacy (after), single-cohort plans
    leg, ind = _iter_nl(after, 'legacy'), _iter_nl(after, 'indep')
    single = set(spec['t_i']['single_cohort_plans'])

    def _plan_of(base_cell):
        return base_cell.split('__')[1]

    per, per_multi = {}, {}
    for k in sorted(set(leg) | set(ind)):
        if k in leg and k in ind:
            (per if _plan_of(k[0]) in single else per_multi)['|'.join(k)] = _nl_difference(leg[k], ind[k])
    n_ident = sum(1 for v in per.values() if v['identical'])
    g['T_i_indep_vs_legacy'] = {'scope': 'single-cohort and x0 plans', 'n_compared': len(per), 'n_identical': n_ident,
                                'differing': {k: v for k, v in per.items() if not v['identical']},
                                'all_identical': n_ident == len(per) == spec['t_i']['n_single_cohort_node_models']}
    g['T_i_indep_vs_legacy_multi_cohort_informational'] = {
        'n_compared': len(per_multi), 'n_identical': sum(1 for v in per_multi.values() if v['identical']),
        'summary': {k: {kk: v.get(kk) for kk in ('identical', 'n_rows', 'n_cols', 'per_file')} for k, v in per_multi.items()}}
    reg = {}
    for gr in after['grids'].values():
        for cell, c in gr['cells'].items():
            if cell.endswith('__indep'):
                reg[cell] = {'has_registry': c.get('has_registry'), 'multi_years': c.get('multi_years')}
    g['T_i_indep_registry'] = reg
    g['T_i_indep_no_multi_years_on_single_cohort_plans'] = all(
        all(not v for v in r['multi_years'].values()) for cell, r in reg.items() if _plan_of(cell) in single)
    # G0
    g['G0_reader'] = after['g0_reader']
    # G3 fixtures
    fb, fa = fx['before'], fx['after']
    newly_failing = sorted(p for p in fb['results'] if fb['results'][p]['outcome'] == 'ok'
                           and fa['results'][p]['outcome'] != 'ok')
    differ = sorted(p for p in fb['results'] if fb['results'][p]['outcome'] != fa['results'][p]['outcome'])
    counts = {t: {} for t in ('before', 'after')}
    for t, r in (('before', fb), ('after', fa)):
        for v in r['results'].values():
            k = v['outcome'] if v['outcome'] in ('ok', 'ABSENT') else 'error: ' + v['outcome'][:100]
            counts[t][k] = counts[t].get(k, 0) + 1
    g['G3_fixtures'] = {'n_declared': fb['n_declared'], 'n_present': [fb['n_present'], fa['n_present']],
                        'n_loads': [fb['n_loads'], fa['n_loads']], 'outcome_counts': counts,
                        'newly_failing': newly_failing, 'outcome_differs': differ,
                        'pass': (not newly_failing and not differ and fb['n_loads'] == fb['n_present']
                                 and fa['n_loads'] == fa['n_present'] and not fb['guard']['verify_0_failures']
                                 and not fa['guard']['verify_0_failures'])}
    # G3c
    g3c = after.get('g3c') or {}
    cmp_ = {}
    for cell, c in (g3c.get('cells') or {}).items():
        base_cell = cell.replace('__g3c_unpickled_params', '')
        ref = after['grids']['srp1']['cells'].get(base_cell + '__legacy')
        if ref is None:
            cmp_[cell] = 'no reference cell'
            continue
        cmp_[cell] = all(files[f]['sha256'] == ref['init'][node][f]['sha256']
                         for node, files in c['init'].items() for f in NL_FILES) and all(
            files[f]['sha256'] == ref['admm'][node][f]['sha256'] for node, files in c['admm'].items() for f in NL_FILES)
    bp = children['before'].get('g3c_pickles') or {}
    model_nl_same = (bool(bp) and bool(g3c.get('model_nl_after_update'))
                     and all(bp['model_nl_before_pickle'][f]['sha256'] == g3c['model_nl_after_update'][f]['sha256']
                             for f in NL_FILES))
    g['G3c'] = {'params_has_key_attr_after_unpickle': g3c.get('params_has_key_attr_after_unpickle'),
                'formulation_resolved': g3c.get('formulation_resolved'), 'params_cells_bitwise': cmp_,
                'model_has_registry_after_unpickle': g3c.get('model_has_registry_after_unpickle'),
                'model_update_raised': g3c.get('model_update_raised'),
                'model_has_registry_after_update': g3c.get('model_has_registry_after_update'),
                'model_nl_bitwise_before_pickle_vs_after_update': model_nl_same}
    g['G3c']['pass'] = (g3c.get('params_has_key_attr_after_unpickle') is False
                        and g3c.get('formulation_resolved') == LEGACY
                        and len(cmp_) == 3 and all(v is True for v in cmp_.values())
                        and g3c.get('model_has_registry_after_unpickle') is False
                        and g3c.get('model_update_raised') is None
                        and g3c.get('model_has_registry_after_update') is False and model_nl_same)
    # T-ib
    tib = after.get('tib') or {}
    pr = spec['tib']['predictions']
    ind_inv = tib.get('indep', {})
    fam = ind_inv.get('active_rows_by_family', {})
    measured = {'soc_rows': fam.get('energy_storage_cohort_soc'), 'closure_rows': fam.get('energy_storage_cohort_soc_final'),
                'window_rows': fam.get('energy_storage_cohort_soc_window'),
                'cohort_circle_rows': fam.get('energy_storage_cohort_converter'),
                'q_rows': fam.get('energy_storage_operation_agg_q'),
                'aggregate_circle_active_by_block': ind_inv.get('aggregate_circle_active_by_block'),
                'h3_rows_active': ind_inv.get('h3_rows_active'), 'h3_rows_built': ind_inv.get('h3_rows_built'),
                'new_vars_free': ind_inv.get('new_vars_free'),
                'legacy_h3_rows_active': (tib.get('legacy') or {}).get('h3_rows_active')}
    g['T_ib'] = {'measured': measured, 'predicted': pr,
                 'held': {k: measured.get(k) == v for k, v in pr.items()}, 'inventory': tib}
    g['T_ib']['pass'] = all(g['T_ib']['held'].values())
    # T-vii
    tvii = after.get('tvii') or {}
    pr7 = spec['tvii']['predictions']
    fam7 = tvii.get('active_rows_by_family', {})
    m7 = {'soc_rows': fam7.get('energy_storage_cohort_soc'), 'closure_rows': fam7.get('energy_storage_cohort_soc_final'),
          'window_rows': fam7.get('energy_storage_cohort_soc_window'),
          'cohort_circle_rows': fam7.get('energy_storage_cohort_converter'),
          'q_rows': fam7.get('energy_storage_operation_agg_q'),
          'aggregate_circle_active_by_block': tvii.get('aggregate_circle_active_by_block'),
          'multi_years': tvii.get('multi_years'), 'new_vars_free': tvii.get('new_vars_free'),
          'new_vars_free_outside_lifetime': tvii.get('new_vars_free_outside_lifetime'),
          'shortened_cohort_rated_fixed_by_block': tvii.get('shortened_cohort_rated_fixed_by_block')}
    g['T_vii'] = {'measured': m7, 'predicted': pr7, 'held': {k: m7.get(k) == v for k, v in pr7.items()},
                  'inventory': tvii}
    g['T_vii']['pass'] = all(g['T_vii']['held'].values())
    # solves
    if solves is not None:
        g['solves'] = _evaluate_solves(spec, solves)
    return g


def _evaluate_solves(spec, s):
    sp = spec['solves']
    t = s['tests']
    out = {'guard': s['guard'], 'guard_exact': not s['guard']['verify_failures'], 'errors': sorted(s['errors']),
           'read_checksum_ok': s['read']['checksum_matches_spec'], 'predictions': {}}
    P = out['predictions']

    def put(name, pred, value, extra=None):
        P[name] = {'prediction': pred, 'measured': value, 'held': _check(pred, value), **(extra or {})}

    def ok(name):
        return name in t and t[name]['solve'].get('succeeded') is True

    for name in sp['order']:
        P[f'{name}__solver_succeeded'] = {'measured': t.get(name, {}).get('solve', {}).get('termination'),
                                          'held': ok(name)}
    pr = sp['predictions']
    if 't_ii_a_indep' in t:
        put('t_ii_a_indep_delivered', pr['t_ii_a_indep_delivered'], t['t_ii_a_indep']['delivered_mw'],
            {'p_slack_mw': t['t_ii_a_indep']['p_slack_mw'],
             'hand_formula_on_solved_e_av_mw': t['t_ii_a_indep']['hand_formula_on_solved_e_av_mw']})
        put('t_ii_a_indep_slack', pr['t_ii_a_indep_slack'], t['t_ii_a_indep']['p_slack_mw'])
    if 't_ii_a_legacy' in t:
        put('t_ii_a_legacy_delivered', pr['t_ii_a_legacy_delivered'], t['t_ii_a_legacy']['delivered_mw'],
            {'p_slack_mw': t['t_ii_a_legacy']['p_slack_mw']})
    if 't_ii_b' in t:
        put('t_ii_b_slack', pr['t_ii_b_slack'], t['t_ii_b']['p_slack_mw'],
            {'precharge_rise_mwh': t['t_ii_b']['precharged_cohort_stored_rise_before_peak_mwh'],
             'min_rise_needed_mwh': t['t_ii_b']['precharged_cohort_min_rise_needed_mwh'],
             'window_headroom_mwh': t['t_ii_b']['precharged_cohort_window_headroom_mwh']})
        put('t_ii_b_precharge_at_least_needed', pr['t_ii_b_precharge_at_least_needed'],
            t['t_ii_b']['precharged_cohort_stored_rise_before_peak_mwh']
            - t['t_ii_b']['precharged_cohort_min_rise_needed_mwh'])
    if 't_iii_indep' in t:
        smm = t['t_iii_indep']['detectors']['summary_multi_cohort']
        put('t_iii_indep_window_margin', pr['t_iii_indep_window_margin'], smm['window_margin_min_mwh'],
            {'per_cohort_window_fraction': t['t_iii_indep']['per_cohort_window_fraction']})
        put('t_iii_indep_p_slack', pr['t_iii_indep_p_slack'], t['t_iii_indep']['p_slack_max_mw_block_day'])
    if 't_iii_legacy' in t:
        fr = t['t_iii_legacy']['per_cohort_window_fraction']
        oldest = min(fr, key=int)
        put('t_iii_legacy_old_cohort_min_fraction', pr['t_iii_legacy_old_cohort_min_fraction'], fr[oldest][0],
            {'per_cohort_window_fraction_implied': fr})
        put('t_iii_legacy_old_cohort_max_fraction', pr['t_iii_legacy_old_cohort_max_fraction'], fr[oldest][1])
    if 't_iv_peak' in t:
        x = t['t_iv_peak']
        put('t_iv_peak_slack', pr['t_iv_peak_slack'], x['p_slack_at_peak_mw'],
            {'no_transfer_slack_bound_mw': x['no_transfer_slack_bound_mw'], 'zero_solve_bounds': x['zero_solve_bounds']})
        put('t_iv_peak_transfer_positive', pr['t_iv_peak_transfer_positive'], x['inter_cohort_gross_transfer_mwh'],
            {'conversion_loss_mwh': x['conversion_loss_mwh']})
        put('t_iv_peak_within_cohort', pr['t_iv_peak_within_cohort'], max(x['within_cohort_simultaneity_mwh'].values()))
        put('t_iv_peak_closure', pr['t_iv_peak_closure'], max(abs(v) for v in x['closure_residual_mwh'].values()))
        put('t_iv_peak_energy_accounting', pr['t_iv_peak_energy_accounting'],
            abs(x['energy_accounting']['residual_mwh']), {'energy_accounting': x['energy_accounting']})
    if 't_iv_control' in t:
        x = t['t_iv_control']
        put('t_iv_control_transfer_over_s', pr['t_iv_control_transfer_over_s'],
            x['inter_cohort_gross_transfer_mwh'] / x['s_aggregate_mva'],
            {'transfer_mwh': x['inter_cohort_gross_transfer_mwh'], 's_aggregate_mva': x['s_aggregate_mva']})
    if 't_v_tv_on' in t:
        x = t['t_v_tv_on']
        put('t_v_tv_on_kappa_positive', pr['t_v_tv_on_kappa_positive'], x['kappa'])
        put('t_v_tv_on_D_from_own_throughput', pr['t_v_D_from_own_throughput'], x['D_max_abs_diff'])
        thr = x['per_cohort_throughput_mwh_per_day']
        put('t_v_tv_on_oldest_cohort_cycles_more_than_energy_share', pr['t_v_tv_on_oldest_cycles_more_than_share'],
            _oldest_share_excess(sp['t_v'], thr), {'per_cohort_throughput_mwh_per_day': thr})
    if 't_v_tv_off_oldest' in t and 't_v_tv_off_newest' in t:
        a, b = t['t_v_tv_off_oldest'], t['t_v_tv_off_newest']
        for x, nm in ((a, 'oldest'), (b, 'newest')):
            put(f't_v_tv_off_{nm}_D_from_own_throughput', pr['t_v_D_from_own_throughput'], x['D_max_abs_diff'])
        dj = abs(a['admm_objective'] - b['admm_objective'])
        put('t_v_tv_off_objective_agreement', pr['t_v_tv_off_objective_agreement_rel'],
            dj / max(1.0, abs(a['admm_objective'])),
            {'admm_objective': [a['admm_objective'], b['admm_objective']], 'abs_diff': dj})
        split = {k: [a['per_cohort_throughput_mwh_per_day'][k], b['per_cohort_throughput_mwh_per_day'].get(k)]
                 for k in a['per_cohort_throughput_mwh_per_day']}
        out['t_v_tv_off_split'] = {'per_cohort_throughput_mwh_per_day_oldest_vs_newest_start': split,
                                   'max_abs_diff': max(abs(u - v) for u, v in split.values() if v is not None)}
    out['n_predictions'] = len(P)
    out['n_held'] = sum(1 for v in P.values() if v['held'])
    return out


def _oldest_share_excess(tvspec, thr):
    """(oldest cohort's share of the throughput summed over its multi-cohort blocks) minus (its share of the rated
    energy of the cohorts active there), averaged over the multi-cohort blocks; > 0 = cycles more than its share."""
    plan = tvspec['plan']
    e = {str(y): ee for (_n, y, _s, ee) in plan}
    oldest = str(min(int(y) for y in e))
    by_block = {}
    for key, v in thr.items():
        yi, yb = key.split('|')
        by_block.setdefault(yb, {})[yi] = v
    diffs = []
    for yb, d in by_block.items():
        if len(d) < 2 or oldest not in d:
            continue
        tot = sum(d.values())
        etot = sum(e[c] for c in d)
        if tot <= 0:
            continue
        diffs.append(d[oldest] / tot - e[oldest] / etot)
    return sum(diffs) / len(diffs) if diffs else None


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
    checks = {}
    spec_rel = os.path.relpath(spec_path, HERE)
    if args.rehearsal:  # harness debugging only (zero solves); never the evidence run
        checks['REHEARSAL_script_and_spec_not_checked_at_base'] = True
    else:
        checks['script_at_base_equals_running'] = (_sha_bytes(_git_bytes('show', f'{base}:{SCRIPT}'))
                                                   == _sha_file(os.path.join(HERE, SCRIPT)))
        checks['diag_at_base_equals_working'] = (_sha_bytes(_git_bytes('show', f'{base}:{DIAG}'))
                                                 == _sha_file(os.path.join(HERE, DIAG)))
        checks['spec_at_base_equals_file'] = (_sha_bytes(_git_bytes('show', f'{base}:{spec_rel}'))
                                              == _sha_bytes(spec_bytes))
    base_prod = {rel: _git_bytes('show', f'{base}:{rel}') for rel in PROD_FILES}
    work_prod = {rel: open(os.path.join(HERE, rel), 'rb').read() for rel in PROD_FILES}
    checks['base_prod_files_lack_key'] = all(KEY.encode() not in v for v in base_prod.values())
    checks['overlay_files_carry_key'] = all(KEY.encode() in v for v in work_prod.values())
    json_hits = [line.split(':', 1)[1] for line in
                 _git('grep', '-l', KEY, base, '--', '*.json', check=False).splitlines() if line.strip()]
    checks['no_committed_json_sets_key'] = not [h for h in json_hits if h != spec_rel]
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
            for f in (SCRIPT, DIAG):
                shutil.copyfile(os.path.join(HERE, f), os.path.join(tree, f))
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
    fixtures = _declare_fixtures(base)
    _dump({'n': len(fixtures), 'fixtures': fixtures}, os.path.join(out, 'fixtures_declared.json'))
    flist_scratch = os.path.join(scratch, 'fixtures_list.json')
    with open(flist_scratch, 'x') as handle:
        json.dump(sorted(fixtures), handle)
    rc, works = {}, {}
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
    result = {'schema': SCHEMA, 'script': SCRIPT, 'rehearsal': bool(args.rehearsal), 'spec': spec_rel,
              'spec_sha256': _sha_bytes(spec_bytes), 'base_commit': base,
              'git_head_at_launch': _git('rev-parse', 'HEAD').strip(),
              'started_utc': datetime.fromtimestamp(t_start, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'interpreter': sys.executable, 'provenance_checks': checks, 'diff_under_test': diff_text,
              'trees': trees, 'child_returncodes': rc, 'gates': {}, 'errors': {}}
    gates = {}
    try:
        gates = _evaluate(spec, children, fx, None)
    except Exception:  # noqa: BLE001 -- recorded, never hidden
        result['errors']['zero_solve_analysis'] = traceback.format_exc()
        print(result['errors']['zero_solve_analysis'], file=sys.stderr, flush=True)
    # the solve child runs only when the zero-solve production-safety gates hold (legacy bitwise, its control,
    # children ok, fixtures) and never in a rehearsal
    pre = all(isinstance(gates.get(k), dict) and gates[k].get('pass') is True
              for k in ('G1_legacy_bitwise', 'G1c_determinism', 'G3_fixtures')) and gates.get('children_ok') is True
    result['solves_launched'] = bool(pre and not args.rehearsal)
    solves = None
    if result['solves_launched']:
        works['solves'] = os.path.join(scratch, 'work_solves')
        os.makedirs(works['solves'])
        rc['solves'] = _run_child(trees['after']['path'], ['solves', '--spec', spec_copy, '--work', works['solves']],
                                  'child_solves', out, env)
        p = os.path.join(works['solves'], 'child_solves.json')
        solves = json.load(open(p)) if os.path.isfile(p) else None
        if solves is None:
            result['errors']['solves'] = 'the solve child wrote no record; see child_solves_stderr.log'
        else:
            try:
                gates['solves'] = _evaluate_solves(spec, solves)
            except Exception:  # noqa: BLE001
                result['errors']['solve_analysis'] = traceback.format_exc()
                print(result['errors']['solve_analysis'], file=sys.stderr, flush=True)
            logs = os.path.join(works['solves'], 'ipopt_logs')
            if os.path.isdir(logs):
                result['ipopt_logs_sha256'] = _gzip_tree(logs, os.path.join(out, 'ipopt_logs'))
    result['gates'] = gates
    verdict = {}
    for k in ('children_ok', 'G1_legacy_bitwise', 'G1c_determinism', 'G0_reader', 'G3_fixtures', 'G3c', 'T_ib', 'T_vii'):
        v = gates.get(k)
        verdict[k] = (v is True) if isinstance(v, bool) else bool(isinstance(v, dict) and v.get('pass') is True)
    verdict['T_i_indep_vs_legacy_all_identical'] = bool((gates.get('T_i_indep_vs_legacy') or {}).get('all_identical'))
    result['verdict'] = verdict
    result['t_i_passes'] = all(verdict[k] for k in ('children_ok', 'G1_legacy_bitwise', 'G1c_determinism', 'G0_reader',
                                                    'G3_fixtures', 'G3c'))
    result['predictions'] = spec.get('predictions_summary')
    for tag, w in works.items():
        for f in sorted(os.listdir(w)):
            if f.startswith('child_') and f.endswith('.json'):
                shutil.copyfile(os.path.join(w, f), os.path.join(out, f))
        if os.path.isdir(os.path.join(w, 'nl')):
            result.setdefault('nl_inventory_sha256_uncompressed', {})[tag] = _gzip_tree(
                os.path.join(w, 'nl'), os.path.join(out, 'nl', tag))
    result['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    result['finished_utc'] = _utc()
    result['wall_s'] = time.time() - t_start
    _dump(result, os.path.join(out, 'cohort_esso_tests.json'))
    _log(f"verdict {verdict}; T-i passes {result['t_i_passes']}; solves launched {result['solves_launched']}; "
         f"errors {sorted(result['errors'])}; guard {dict(GUARD.counts)}")
    if 'solves' in gates:
        s = gates['solves']
        _log(f"solves: guard exact {s['guard_exact']} ({s['guard']['counts']}); predictions held {s['n_held']}/"
             f"{s['n_predictions']}; errors {s['errors']}")
        for k, v in s['predictions'].items():
            _log(f"  {k}: held {v['held']} measured {v['measured']}")
    return 0 if result['t_i_passes'] and not result['errors'] else 1


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
    for f in (SCRIPT, DIAG):
        files[f] = _sha_file(os.path.join(HERE, f))
    with open(target, 'x') as handle:
        handle.write(GRIO.dumps({'schema': SCHEMA + '_manifest', 'utc': _utc(), 'files': files}, indent=1,
                                sort_keys=True))
    print(f'manifest: {len(files)} files -> {target}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--child', choices=('build', 'fixtures', 'solves'))
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
    if args.child == 'solves':
        return child_solves(args)
    return parent(args)


if __name__ == '__main__':
    sys.exit(main())
