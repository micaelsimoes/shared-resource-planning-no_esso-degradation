"""
P5.15 Addendum 30 (task W21, item 1) -- ZERO-SOLVE gate for the case-file edit that makes
the ageing BASELINE (C2 + phi_cal 0.985 + soh_min 0.70) the content of
`data/SRP1/SharedESS/SRP1_ESS_Params.json` (`ageing` block only: calibration
eol_retention_r 0.50 -> 0.80, calendar_retention_per_year 0.985 added, minimum_soh
0.50 -> 0.70, the calibration `_note` updated). Armed `SolveProfileGuard(permitted=())`
for the whole script (installed before any model import), `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 30; frozen spec v17
`data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json` (`baseline`, S1);
Planner task W21 item 1.

TWO PHASES, two processes (`p56a_oracle.load_baseline` reads the files once per process):
  --phase before   on the UNEDITED file (C3; sha256 ESS_PARAMS_SHA256_BEFORE)
  --phase after    on the EDITED file; loads the `before` record (sha256 given with
                   --before-sha256, checked) and compares.

WHAT EACH PHASE DOES
  (a) READ-BACK at the smallest node-7 4 h unit (0.25 MVA / 1.0 MWh at node 7, 2025): a
      planning object built by `_construct_arm_planning` (the code `run_admm_arm` calls),
      passed through the campaign child's configuration hook with the ESS-ageing
      declaration = what the file loads to (`_config_hook_factory`; it verifies the loaded
      parameters and reads k / phi / the floor bound back from PROBE ESSO models); then the
      ESSO models are built as `create_shared_energy_storage_model` builds them
      (`update_data_with_candidate_solution`, `build_subproblem`,
      `update_model_with_candidate_solution`) and k, phi, the SoH-point mode and the floor
      row's lower bound are read back numerically from a CLONE of node 7's model
      (`model_variant_readback`), against the closed forms of the file's own values.
  (b) ORACLE-PATH BUILD at x = 0 and at C* (0.96875 MVA / 3.875 MWh at 5, 7, 9; 2025): as in
      W20's gate (`p515_s46_case_file_edit_check.py`, cb165d4e) -- `_construct_arm_planning`,
      the child's configuration hook, then the production builders `run_operational_planning`
      calls at initialization (`create_admm_variables`, `create_distribution_networks_models`,
      `create_transmission_network_model`, `create_shared_energy_storage_model`) with each
      holder's `optimize` replaced by a stub that DIGESTS the blocks handed to it (nothing is
      solved). The digest is FINER than W20's: per component (every Param value, every Var
      (value, raw lb, raw ub, fixed), every Constraint (active, lower, upper AND the body
      expression string, so coefficient changes such as k in the D row are seen), every
      Objective (active, expression string), every Expression (expression string), every
      block's active flag). For ESSO blocks the per-index state is kept, so a differing item
      can be named. For ESSO blocks the NL file Pyomo writes for the block (the problem IPOPT
      would receive; written, never solved) is hashed as a second, solver-visible digest.
  (c) SALVAGE. The terminal salvage expression (`_build_terminal_salvage_value_expression`,
      health basis NORMALIZED_ABOVE_MINIMUM_SOH) is evaluated from the built model's own
      `salvage_value` Expression for a 2030 cohort (the 2025 cohort's remaining calendar life
      at the horizon is 0, so its salvage is 0 whatever soh_min is) at synthetic terminal SoH
      values, and the remaining-life fraction of each cohort year is read from production's
      `_get_remaining_calendar_life`.
  after only: per instance, which blocks differ and, inside differing blocks, which
      components / items; the JSON diff of the file (only the ageing block may change).

Output (write-once): data/SRP1/Results/P515S47/case_file_baseline/{before,after}/
    case_file_baseline_<phase>.json, manifest_sha256.json
Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_case_file_baseline_check.py \\
        --phase before > data/SRP1/Results/P515S47/case_file_baseline_before_launch.log 2>&1
    (edit the file)
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_case_file_baseline_check.py \\
        --phase after --before-sha256 <sha> > data/SRP1/Results/P515S47/case_file_baseline_after_launch.log 2>&1
"""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W21 case-file baseline gate (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = ('P5.15 Addendum 30 W21 item 1 -- SRP1_ESS_Params.json ageing baseline C2 + phi_cal 0.985 + soh_min 0.70: '
         'zero-solve gate')
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 30',
             'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json (baseline, S1)',
             'Planner task W21 item 1']
ESS_PARAMS_REL = H.ESS_PARAMS_FILE_REL
ESS_PARAMS_SHA256_BEFORE = 'fdce321ffe1bf0f4b81424cdf081d826c3c81c410a4eb9f3b3561675e26b80f0'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'case_file_baseline')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
C_STAR_KEY_PIN = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'
LOADED_BEFORE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                 'minimum_soh': 0.5, 'calendar_retention_per_year': 1.0,
                 'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                 'eol_retention_r': 0.5}}
LOADED_AFTER = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                'eol_retention_r': 0.8}}
# JSON leaf paths the edit may change (everything else in the file must be identical).
ALLOWED_CHANGED_PATHS = ['ageing.calendar_retention_per_year', 'ageing.calibration._note',
                         'ageing.calibration.eol_retention_r', 'ageing.minimum_soh']
UNIT_NODES = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
SALVAGE_SOH_PROBES = (1.0, 0.9, 0.8, 0.75, 0.7)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W21-casefile] {msg}', flush=True)


def _git(args, check=True):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=check).stdout


def _sha(path):
    return H.sha256_file(path)


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _leaves(obj, prefix=''):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, f'{prefix}.{k}' if prefix else str(k)))
        return out
    return {prefix: obj}


# ======================================================================================================================
#  digests
# ======================================================================================================================
def block_state(model, keep_items):
    """Per component: {index repr: state repr}. Returns (component digests, per-index items or None)."""
    import pyomo.environ as pe
    comps = {}
    for comp in model.component_objects(pe.Param, active=None, descend_into=True):
        comps[f'Param:{comp.name}'] = {repr(i): repr(pe.value(comp[i], exception=False)) for i in comp}
    for comp in model.component_objects(pe.Var, active=None, descend_into=True):
        comps[f'Var:{comp.name}'] = {repr(i): repr((v.value, v._lb, v._ub, v.fixed)) for i, v in comp.items()}
    for comp in model.component_objects(pe.Constraint, active=None, descend_into=True):
        comps[f'Constraint:{comp.name}'] = {
            repr(i): repr((c.active, None if c.lower is None else pe.value(c.lower),
                           None if c.upper is None else pe.value(c.upper), str(c.body))) for i, c in comp.items()}
    for comp in model.component_objects(pe.Objective, active=None, descend_into=True):
        comps[f'Objective:{comp.name}'] = {repr(i): repr((o.active, str(o.expr))) for i, o in comp.items()}
    for comp in model.component_objects(pe.Expression, active=None, descend_into=True):
        comps[f'Expression:{comp.name}'] = {repr(i): str(e.expr) for i, e in comp.items()}
    comps['Blocks:active'] = {blk.name or '<root>': repr(blk.active)
                              for blk in model.block_data_objects(active=None, descend_into=True)}
    comp_digests = {name: _h(json.dumps(items, sort_keys=True)) for name, items in comps.items()}
    block = _h(json.dumps(comp_digests, sort_keys=True))
    return block, comp_digests, (comps if keep_items else None)


def nl_digest(model, scratch, tag):
    """sha256 of the NL file Pyomo writes for the block (never solved); None + error text if the writer refuses."""
    path = os.path.join(scratch, f'{tag}.nl')
    try:
        model.write(path, format='nl', io_options={'symbolic_solver_labels': False})
        digest = _sha(path)
        size = os.path.getsize(path)
        os.remove(path)
        return {'sha256': digest, 'bytes': size, 'error': None}
    except Exception as error:  # noqa: BLE001
        return {'sha256': None, 'bytes': None, 'error': f'{type(error).__name__}: {error}'}


def _spec_like(declared, pin):
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                              'ess_ageing_baseline': declared, 'ess_params_file': pin,
                              'ess_ageing_baseline_label': 'GATE'},
            'cap': CAP, 'required_consecutive_cycles': 10}


def oracle_path_build(G, srp, label, investment_map, run_tag, phase, scratch, declared, pin, floor_rows):
    report = {}
    eval_id = f'p515s47_w21_casefile_{phase}_{run_tag}_{label}'
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, label), report, investment_map=investment_map, eval_id=eval_id,
        num_max_iters_override=CAP, apply_rho=False, investment_year=2025)
    holder = {}
    blocks, esso_items, esso_nl = {}, {}, {}

    def network_stub(tag):
        def stub(model, *args, **kwargs):
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    b, comps, _items = block_state(model[year][day], keep_items=False)
                    blocks[f'{tag}|{year}|{day}'] = {'sha256': b, 'components': comps}
                    results[year][day] = None
            return results
        return stub

    def esso_stub(models, *args, **kwargs):
        for node_id, model in models.items():
            key = f'ESSO|node{node_id}'
            b, comps, items = block_state(model, keep_items=True)
            blocks[key] = {'sha256': b, 'components': comps}
            esso_items[key] = items
            esso_nl[key] = nl_digest(model, scratch, f'{phase}_{label}_node{node_id}')
        return {node_id: None for node_id in models}

    H._config_hook_factory(_spec_like(declared, pin), holder, overrides={}, investment_year=2025,
                           expected_floor_rows=floor_rows)(planning=planning, sed=sed, candidate=candidate,
                                                           report=report)
    planning.transmission_network.optimize = network_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = network_stub(f'DSO{node_id}')
    sed.optimize = esso_stub
    consensus_vars, _dual_vars = srp.create_admm_variables(planning)
    srp.create_distribution_networks_models(planning.distribution_networks, consensus_vars,
                                            candidate['total_capacity'],
                                            parallel_execution=planning.parallel_execution)
    srp.create_transmission_network_model(planning, consensus_vars, candidate['total_capacity'])
    srp.create_shared_energy_storage_model(sed, consensus_vars, candidate['investment'])
    canonical = H.canonical_candidate(investment_map, investment_year=2025)
    return {'label': label, 'eval_id': eval_id, 'candidate_canonical': canonical,
            'candidate_key': H.candidate_key(canonical),
            'hook_ess_ageing_checks': (holder.get('ess_ageing_verified_pre_run') or {}).get('checks'),
            'hook_readback_all_match': ((holder.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run')
                                        or {}).get('all_match'),
            'n_blocks': len(blocks), 'blocks': dict(sorted(blocks.items())), 'esso_items': esso_items,
            'esso_nl': esso_nl}


# ======================================================================================================================
#  (a) read-back at the unit, (c) salvage
# ======================================================================================================================
def _build_esso(sed, candidate):
    """ESSO models exactly as `create_shared_energy_storage_model` builds them, minus the TSO request and solve."""
    sed.update_data_with_candidate_solution(candidate['investment'])
    models = sed.build_subproblem()
    sed.update_model_with_candidate_solution(models, candidate['investment'])
    return models


def readback_unit(G, run_tag, phase, scratch, declared, pin, floor_rows):
    report, holder = {}, {}
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, 'unit'), report, investment_map=UNIT_NODES,
        eval_id=f'p515s47_w21_casefile_{phase}_{run_tag}_unit', num_max_iters_override=CAP, apply_rho=False,
        investment_year=2025)
    H._config_hook_factory(_spec_like(declared, pin), holder, overrides={}, investment_year=2025,
                           expected_floor_rows=floor_rows)(planning=planning, sed=sed, candidate=candidate,
                                                           report=report)
    verified = holder['ess_ageing_verified_pre_run']
    models = _build_esso(sed, candidate)
    built = H.ess_ageing_readback_models({7: models[7]}, sed, declared, 2025, clone=True)
    rb = built['per_node']['7']['readback']
    probe7 = verified['readback_pre_run']['per_node']['7']['readback']
    return {'unit': {'nodes': {str(n): list(v) for n, v in UNIT_NODES.items()}, 'investment_year': 2025},
            'expected_closed_forms': built['expected'],
            'built_node7': {k: rb.get(k) for k in ('k', 'phi_cal_in_model', 'floor_row_lower', 'd_row_form',
                                                   'available_energy_soh_point', 'n_years_from_d_row', 'y_inv', 'y0')},
            'built_node7_checks': built['per_node']['7']['checks'], 'built_all_match': built['all_match'],
            'probe_node7': {k: probe7.get(k) for k in ('k', 'phi_cal_in_model', 'floor_row_lower')},
            'probe_all_nodes_match': verified['readback_pre_run']['all_match'],
            'loaded_ageing_in_child': verified['loaded'], 'k_production': verified['k_production'],
            'hook_checks': verified['checks'],
            'per_ess_constants': verified['per_ess']}


def salvage_probe(G, SED, run_tag, phase, scratch, declared, pin, floor_rows):
    """Evaluate the built model's own `salvage_value` Expression for a 2030 cohort (node 7, 1.0 MWh) at synthetic
    terminal SoH values; remaining calendar-life fraction per cohort from production's helper."""
    import pyomo.environ as pe
    report, holder = {}, {}
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, 'salvage2030'), report, investment_map=UNIT_NODES,
        eval_id=f'p515s47_w21_casefile_{phase}_{run_tag}_salvage2030', num_max_iters_override=CAP, apply_rho=False,
        investment_year=2030)
    H._config_hook_factory(_spec_like(declared, pin), holder, overrides={}, investment_year=2030,
                           expected_floor_rows=floor_rows)(planning=planning, sed=sed, candidate=candidate,
                                                           report=report)
    models = _build_esso(sed, candidate)
    years = [int(y) for y in sed.years]
    idx = sed.get_shared_energy_storage_idx(7)
    life = {}
    for y_inv, year in enumerate(sed.years):
        ess = sed.shared_energy_storages[year][idx]
        age, remaining, frac = SED._get_remaining_calendar_life(sed, y_inv, ess)
        life[str(year)] = {'age_at_terminal_years': age, 'remaining_life_years': remaining,
                           'remaining_life_fraction': frac}
    y_inv = years.index(2030)
    t_idx = len(years) - 1
    rows = []
    for soh in SALVAGE_SOH_PROBES:
        clone = models[7].clone()
        clone.es_e_rated_per_unit[y_inv, t_idx].set_value(1.0)
        clone.es_e_available_per_unit[y_inv, t_idx].set_value(soh)
        clone.es_soh_per_unit_cumul[y_inv, t_idx].set_value(soh)
        rows.append({'terminal_soh': soh, 'salvage_value_model_eur': pe.value(clone.salvage_value)})
        del clone
    src, first = inspect.getsourcelines(SED._build_terminal_salvage_value_expression)
    line_of = {}
    for i, text in enumerate(src):
        for tag in ('min_soh = shared_energy_storage.soh_min', 'usable_energy_above_eol = ', 'residual_energy = (',
                    'salvage_value += ('):
            if tag in text and tag not in line_of:
                line_of[tag] = first + i
    return {'cohort': '2030 at node 7, E = 1.0 MWh (0.25 MVA)',
            'soh_min_in_model': sed.shared_energy_storages[list(sed.years)[y_inv]][idx].soh_min,
            'salvage_by_terminal_soh': rows, 'remaining_calendar_life_by_cohort_year': life,
            'salvage_params': {k: getattr(sed.params.salvage_value, k) for k in (
                'enabled', 'energy_recovery_fraction', 'recycling_floor_fraction', 'cost_basis', 'health_basis',
                'calendar_life_basis')},
            'expression_source': {'file': 'shared_energy_storage_data.py',
                                  'function': '_build_terminal_salvage_value_expression',
                                  'first_line': first, 'lines': line_of},
            'formula': ('salvage = sum over cohorts of disc * f_rec * c_E(y_inv) * frac_life(y_inv) * '
                        '(f_floor * E_rated + (1 - f_floor) * (E_available - soh_min * E_rated) / (1 - soh_min)), '
                        'at the terminal block; f_floor = recycling_floor_fraction (0.0 here), so the health factor '
                        'is (SoH_T - soh_min) / (1 - soh_min)')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('before', 'after'))
    parser.add_argument('--before-sha256', default=None)
    parser.add_argument('--out', default=OUT_REL)
    args = parser.parse_args()
    started = time.time()
    out_root = os.path.join(REPO, args.out, args.phase)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    ess_path = os.path.join(REPO, ESS_PARAMS_REL)
    ess_sha = _sha(ess_path)
    with open(ess_path) as handle:
        ess_json_text = handle.read()
    loaded = H.load_ess_ageing_parameters(ess_path)
    expected_loaded = LOADED_BEFORE if args.phase == 'before' else LOADED_AFTER
    failures = []
    if args.phase == 'before' and ess_sha != ESS_PARAMS_SHA256_BEFORE:
        failures.append(f'phase before: {ESS_PARAMS_REL} sha256 {ess_sha} != pre-edit {ESS_PARAMS_SHA256_BEFORE}')
    if H.ess_ageing_canonical_text(loaded) != H.ess_ageing_canonical_text(expected_loaded):
        failures.append(f'phase {args.phase}: the file loads to {loaded}, expected {expected_loaded}')
    before_record = None
    if args.phase == 'after':
        before_path = os.path.join(REPO, args.out, 'before', 'case_file_baseline_before.json')
        if not args.before_sha256:
            failures.append('phase after needs --before-sha256')
        elif not os.path.isfile(before_path) or _sha(before_path) != args.before_sha256:
            failures.append(f'before record missing or sha256 != {args.before_sha256}: {before_path}')
        else:
            with open(before_path) as handle:
                before_record = json.load(handle)
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out_root)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    head = _git(['rev-parse', 'HEAD']).strip()
    _log(f'{STAGE}; phase {args.phase}; git HEAD {head}; {ESS_PARAMS_REL} sha256 {ess_sha}; loads to {loaded}')

    import p515_g_g1_g4_admm_gates as G
    import p514_n_instrumented_cstar as N
    import p515_s40_polish_gap as PG
    import shared_resources_planning as srp
    import shared_energy_storage_data as SED

    pin = {'path': ESS_PARAMS_REL, 'sha256': ess_sha}
    scratch = tempfile.mkdtemp(prefix='p515s47_w21_casefile_')
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s47_w21_casefile_{args.phase}_{run_tag}_precheck')
    readback = readback_unit(G, run_tag, args.phase, scratch, loaded, pin, floor_rows)
    _log(f"(a) unit read-back (built node 7): {readback['built_node7']} checks {readback['built_node7_checks']}; "
         f"expected {readback['expected_closed_forms']}")
    salvage = salvage_probe(G, SED, run_tag, args.phase, scratch, loaded, pin, floor_rows)
    _log(f"(c) salvage 2030 cohort: {salvage['salvage_by_terminal_soh']}; life {salvage['remaining_calendar_life_by_cohort_year']}")
    instances = {'x0': {n: (0.0, 0.0) for n in H.ACTIVE_NODES},
                 'c_star': {n: (N.S_INV, N.E_INV) for n in H.ACTIVE_NODES}}
    builds = {}
    for label, inv_map in instances.items():
        t0 = time.time()
        builds[label] = oracle_path_build(G, srp, label, inv_map, run_tag, args.phase, scratch, loaded, pin,
                                          floor_rows)
        nl_text = {k: (v['sha256'][:12] if v['sha256'] else v['error']) for k, v in builds[label]['esso_nl'].items()}
        _log(f"(b) {label}: key {builds[label]['candidate_key'][:16]} blocks {builds[label]['n_blocks']} "
             f"hook checks {builds[label]['hook_ess_ageing_checks']} readback {builds[label]['hook_readback_all_match']} "
             f"ESSO NL {nl_text} ({time.time() - t0:.1f}s)")
    shutil.rmtree(scratch, ignore_errors=True)

    checks = {
        'a_built_readback_matches_the_file_closed_forms': readback['built_all_match'],
        'a_probe_readback_matches_on_every_node': readback['probe_all_nodes_match'] is True,
        'a_hook_checks_all_pass': all(readback['hook_checks'].values()),
        'c_star_key_is_pinned_c_star': builds['c_star']['candidate_key'] == C_STAR_KEY_PIN,
        'blocks_48_network_plus_3_esso_per_instance': all(b['n_blocks'] == 51 for b in builds.values()),
        'hook_passed_for_both_instances': all(b['hook_ess_ageing_checks'] and all(b['hook_ess_ageing_checks'].values())
                                              and b['hook_readback_all_match'] for b in builds.values()),
        'digest_discriminates_c_star_from_x0_in_every_block': all(
            builds['c_star']['blocks'][k]['sha256'] != builds['x0']['blocks'][k]['sha256']
            for k in builds['c_star']['blocks']),
    }
    comparison = None
    if args.phase == 'after':
        per_instance = {}
        for label in instances:
            a, b = before_record['builds'][label], builds[label]
            differing = {}
            for key in sorted(set(a['blocks']) | set(b['blocks'])):
                ba, bb = a['blocks'].get(key) or {}, b['blocks'].get(key) or {}
                if ba.get('sha256') == bb.get('sha256'):
                    continue
                ca, cb = ba.get('components') or {}, bb.get('components') or {}
                comps = sorted(n for n in set(ca) | set(cb) if ca.get(n) != cb.get(n))
                entry = {'components_differing': comps}
                if key.startswith('ESSO'):
                    ia, ib = (a['esso_items'] or {}).get(key) or {}, b['esso_items'].get(key) or {}
                    items = {}
                    for n in comps:
                        xa, xb = ia.get(n) or {}, ib.get(n) or {}
                        diff_idx = sorted(i for i in set(xa) | set(xb) if xa.get(i) != xb.get(i))
                        items[n] = {'n_items': len(set(xa) | set(xb)), 'n_differing': len(diff_idx),
                                    'examples': [{'index': i, 'before': xa.get(i), 'after': xb.get(i)}
                                                 for i in diff_idx[:4]]}
                    entry['items'] = items
                    entry['nl_before'] = (a.get('esso_nl') or {}).get(key)
                    entry['nl_after'] = b['esso_nl'].get(key)
                differing[key] = entry
            nl = {k: {'before': (a.get('esso_nl') or {}).get(k, {}).get('sha256'), 'after': v.get('sha256'),
                      'identical': (a.get('esso_nl') or {}).get(k, {}).get('sha256') == v.get('sha256')
                      and v.get('sha256') is not None}
                  for k, v in b['esso_nl'].items()}
            per_instance[label] = {'n_blocks_before': len(a['blocks']), 'n_blocks_after': len(b['blocks']),
                                   'n_differing': len(differing), 'differing_blocks': sorted(differing),
                                   'network_blocks_differing': sorted(k for k in differing if not k.startswith('ESSO')),
                                   'esso_blocks_differing': sorted(k for k in differing if k.startswith('ESSO')),
                                   'detail': differing, 'esso_nl_comparison': nl}
        before_leaves = _leaves(json.loads(before_record['ess_params_file']['text']))
        after_leaves = _leaves(json.loads(ess_json_text))
        changed = sorted(k for k in set(before_leaves) | set(after_leaves)
                         if before_leaves.get(k, '<absent>') != after_leaves.get(k, '<absent>'))
        comparison = {'per_instance': per_instance, 'json_leaf_paths_changed': changed,
                      'json_changes': {k: {'before': before_leaves.get(k, '<absent>'),
                                           'after': after_leaves.get(k, '<absent>')} for k in changed},
                      'loaded_before': before_record['ess_ageing_loaded'], 'loaded_after': loaded,
                      'readback_before': before_record['readback_unit']['built_node7'],
                      'readback_after': readback['built_node7'],
                      'salvage_before': before_record['salvage']['salvage_by_terminal_soh'],
                      'salvage_after': salvage['salvage_by_terminal_soh'],
                      'before_record_sha256': args.before_sha256}
        x0, cs = per_instance['x0'], per_instance['c_star']
        checks['b_x0_every_block_digest_identical_before_after'] = x0['n_differing'] == 0
        checks['b_x0_every_network_block_identical'] = not x0['network_blocks_differing']
        checks['b_x0_esso_nl_identical_before_after'] = all(v['identical'] for v in x0['esso_nl_comparison'].values())
        checks['b_c_star_network_blocks_identical'] = not cs['network_blocks_differing']
        checks['b_c_star_esso_blocks_differ'] = len(cs['esso_blocks_differing']) > 0
        checks['file_only_ageing_leaves_changed'] = changed == ALLOWED_CHANGED_PATHS
        checks['before_record_phase_is_before'] = before_record.get('phase') == 'before'
    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    payload = {'stage': STAGE, 'phase': args.phase, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
               'git_head_at_run': head, 'script': os.path.basename(__file__),
               'script_sha256': _sha(os.path.abspath(__file__)), 'harness_sha256': _sha(H.HARNESS_PATH),
               'ess_params_file': {'path': ESS_PARAMS_REL, 'sha256': ess_sha, 'text': ess_json_text,
                                   'git_status': _git(['status', '--porcelain', '--', ESS_PARAMS_REL]).strip()},
               'ess_ageing_loaded': loaded, 'readback_unit': readback, 'salvage': salvage,
               'builds': builds, 'comparison': comparison,
               'digest_definition': ('per block: sha256 over per-component sha256 of {Param values; Var (value, raw '
                                     'lb, raw ub, fixed); Constraint (active, lower, upper, str(body)); Objective '
                                     '(active, str(expr)); Expression str(expr); block active flags}, as handed to '
                                     'optimize() at ADMM initialization (nothing solved; TSO blocks carry the '
                                     'initial consensus values, the same in both phases). ESSO blocks also keep '
                                     'the per-index state and the sha256 of the NL file Pyomo writes for them '
                                     '(symbolic_solver_labels False; written, never solved)'),
               'checks': checks, 'all_ok': all_ok,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
               'wall_clock_s': time.time() - started}
    path = os.path.join(out_root, f'case_file_baseline_{args.phase}.json')
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {os.path.relpath(path, REPO): _sha(path)}
    with open(os.path.join(out_root, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    for k, v in checks.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    if comparison:
        for label, v in comparison['per_instance'].items():
            _log(f"comparison {label}: differing blocks {v['n_differing']} network {v['network_blocks_differing']} "
                 f"ESSO {v['esso_blocks_differing']}; NL {v['esso_nl_comparison']}")
            for key, d in v['detail'].items():
                _log(f"   {key}: components {d['components_differing']}")
                for n, it in (d.get('items') or {}).items():
                    _log(f"      {n}: {it['n_differing']}/{it['n_items']} items; e.g. {it['examples'][:1]}")
        _log(f"json leaves changed {comparison['json_leaf_paths_changed']}")
    _log(f'wrote {os.path.relpath(path, REPO)} sha256={_sha(path)}')
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
