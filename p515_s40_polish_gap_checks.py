"""
P5.15 Addendum 22 item (4) -- zero-solve structural checks for the Step 3.5
polish-gap gate (`p515_s40_polish_gap.py`).

Authority: PLANNER_BRIEF_2026-09-13.md Step 3.5, Addendum 22 item `4_step_3_5`;
frozen spec v11 `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.

Guard armed at 0 permitted solves for the WHOLE script (`SolveProfileGuard(
permitted=(), ...)`, the blocking form -- same convention as
`p515_s32_zero_solve_checks.py`, `p515_s38_zero_solve_checks.py`,
`p515_s39_zero_solve_checks.py`). Builds a REAL, full (TSO+DSO+ESSO)
zero-solve ADMM-ready state via `p515_s32_zero_solve_checks._build_admm_ready_
state` BY IMPORT (the production model-construction sequence,
`create_admm_variables` / `create_distribution_networks_models` /
`create_transmission_network_model` / `_prepare_*_objectives_for_admm` /
`update_*_model_to_admm`, `.optimize` monkeypatched to a stub that never
reaches a solver -- never reimplemented here).

Checks:
  (i)   objective switching (`p56a_oracle.restore_base_objective`) leaves the
        model otherwise unchanged: every Var's value and `.fixed` flag are
        identical before/after, only `admm_objective.active` flips False and
        `objective.active` flips True.
  (ii)  the fixing step (`p56a_oracle.apply_common_values`) fixes EXACTLY the
        intended variables (`vmag_sqr` on both interface nodes, `pg`/`qg` at
        the DSO reference generator, `shared_es_pnet`/`shared_es_qnet` on both
        sides' shared-ESS index) at EXACTLY the intended (common) values, adds
        the `p56a_interface_rows` ConstraintList with exactly
        `2 * n_active_nodes * n_periods` rows, each satisfied at the fixed
        point, and fixes NO other variable.
  (iii) per-block recourse aggregation (`p56a_oracle.per_block_base_objectives`,
        summed) reproduces `planning.get_operational_recourse_components(
        models)['gross_operational_cost_including_settlement']` (the
        settlement-INCLUDED convention `per_block_base_objectives` sums, per
        `_get_operational_recourse_components`'s own comment) at an unpolished
        state, to floating-point precision. The build-default state's base
        objective is exactly 0.0 (per `_build_admm_ready_state`'s own
        docstring -- no generation dispatched yet), so a handful of Var values
        are perturbed directly (`.set_value`, never `.fix`/solve) first, so
        the check exercises a NON-trivial (nonzero) aggregate.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s40_polish_gap_checks.py

Writes data/SRP1/Results/P515S40/polish_gap_checks/polish_gap_checks.json
(new file; `_refuse_overwrite`, same convention as every P515S3x/S40 checks
script).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'polish_gap_checks')
OUT_PATH = os.path.join(OUT_DIR, 'polish_gap_checks.json')

EVAL_ID = 'p515s40_polish_gap_checks_state'


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _snapshot_all_vars(model):
    """{(component_name, index): (value, fixed)} for EVERY Var on `model`,
    descending into every Block (so nested sub-blocks, if any, are covered
    too)."""
    snapshot = {}
    for var in model.component_objects(pe.Var, active=None, descend_into=True):
        for idx in var:
            vdata = var[idx]
            try:
                value = pe.value(vdata, exception=False)
            except Exception:  # noqa: BLE001
                value = None
            snapshot[(var.local_name, idx)] = (value, bool(vdata.fixed))
    return snapshot


def check_i_objective_switching(planning, tso_model, dso_models, results):
    """Pick one TSO block and one DSO block; verify `restore_base_objective`
    changes ONLY the two Objective components' active flags."""
    year = next(iter(planning.transmission_network.years))
    day = next(iter(planning.transmission_network.days))
    t_model = tso_model[year][day]
    node_id = next(iter(planning.distribution_networks))
    d_year = next(iter(planning.distribution_networks[node_id].years))
    d_day = next(iter(planning.distribution_networks[node_id].days))
    d_model = dso_models[node_id][d_year][d_day]

    per_model = {}
    for label, model in (('tso', t_model), ('dso', d_model)):
        has_admm_obj = hasattr(model, 'admm_objective')
        admm_active_before = bool(model.admm_objective.active) if has_admm_obj else None
        base_active_before = bool(model.objective.active)
        before = _snapshot_all_vars(model)

        O.restore_base_objective(model)

        admm_active_after = bool(model.admm_objective.active) if has_admm_obj else None
        base_active_after = bool(model.objective.active)
        after = _snapshot_all_vars(model)

        same_keys = (set(before) == set(after))
        n_changed = sum(1 for k in before if before[k] != after.get(k)) if same_keys else None
        changed_sample = ([k for k in before if before[k] != after.get(k)][:5]
                          if same_keys else None)

        per_model[label] = {
            'has_admm_objective': has_admm_obj,
            'admm_objective_active_before': admm_active_before,
            'admm_objective_active_after': admm_active_after,
            'base_objective_active_before': base_active_before,
            'base_objective_active_after': base_active_after,
            'var_key_sets_identical': same_keys,
            'n_vars_changed': n_changed,
            'changed_sample': [str(k) for k in changed_sample] if changed_sample else [],
            'pass': bool(
                same_keys and n_changed == 0
                and (not has_admm_obj or (admm_active_before and not admm_active_after))
                and (not base_active_before) and base_active_after),
        }

    results['i_objective_switching'] = {
        'per_model': per_model,
        'pass': bool(all(v['pass'] for v in per_model.values())),
    }


def _expected_fixed_targets(planning, models, common):
    """Independently enumerate the (component_name, index) -> expected_value
    pairs `apply_common_values` is documented to fix, from the SAME index
    lookups (`get_node_idx`, `get_reference_node_id`, `get_reference_gen_idx`,
    `get_shared_energy_storage_idx`) `apply_common_values` itself uses --
    cross-checking the RESULT of the production function against an
    independently reconstructed expected set, not re-running its own code."""
    tso = planning.transmission_network
    expected = {}
    n_interface_rows = 0
    for node, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net, d_net = tso.network[year][day], dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node][year][day]
                adn_idx = t_net.get_node_idx(node)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                ref_gen = d_net.get_reference_gen_idx()
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages) if s.bus == node]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages) if s.bus == ref_id]
                for p in t_model.periods:
                    entry = common[(node, year, day, p)]
                    key = (id(t_model), 'vmag_sqr', (adn_idx, 0, 0, p))
                    expected[key] = entry['common_v'] ** 2
                    key = (id(d_model), 'vmag_sqr', (ref_idx, 0, 0, p))
                    expected[key] = entry['common_v'] ** 2
                    key = (id(d_model), 'pg', (ref_gen, 0, 0, p))
                    expected[key] = entry['common_p'] + entry['common_sess_p']
                    key = (id(d_model), 'qg', (ref_gen, 0, 0, p))
                    expected[key] = entry['common_q'] + entry['common_sess_q']
                    for e in t_sess:
                        expected[(id(t_model), 'shared_es_pnet', (e, 0, 0, p))] = entry['common_sess_p']
                        expected[(id(t_model), 'shared_es_qnet', (e, 0, 0, p))] = entry['common_sess_q']
                    for e in d_sess:
                        expected[(id(d_model), 'shared_es_pnet', (e, 0, 0, p))] = entry['common_sess_p']
                        expected[(id(d_model), 'shared_es_qnet', (e, 0, 0, p))] = entry['common_sess_q']
                    n_interface_rows += 2  # p and q, one ConstraintList row each
    return expected, n_interface_rows


def check_ii_fixing_step(planning, tso_model, dso_models, consensus_vars, results):
    models = {'tso': tso_model, 'dso': dso_models}
    # `consensus_vars` is the SAME object `_build_admm_ready_state` built the
    # models with (build-default: ESS z = 0.0 on every entry) -- needed only
    # by `common_coordinated_values` for the ESS family's z lookup.
    common = O.common_coordinated_values(planning, models, consensus_vars)

    before = {}
    for tag, model_by_yd in (('tso', tso_model), *((f'dso{n}', m) for n, m in dso_models.items())):
        for year, by_day in model_by_yd.items():
            for day, model in by_day.items():
                before[id(model)] = _snapshot_all_vars(model)

    n_p56a_rows_before = {}
    for year, by_day in tso_model.items():
        for day, model in by_day.items():
            n_p56a_rows_before[id(model)] = (len(model.p56a_interface_rows)
                                             if hasattr(model, 'p56a_interface_rows') else 0)

    O.apply_common_values(planning, models, common)

    expected, n_interface_rows_expected = _expected_fixed_targets(planning, models, common)

    # NOTE (finding, see Worker report): shared_es_pnet/qnet are ALREADY
    # `fixed=True` at build-default (before any ADMM cycle or
    # `apply_common_values` call -- confirmed by direct inspection), so a
    # "newly fixed" (False->True) transition test would silently miss
    # `apply_common_values` correctly RE-fixing them to a NEW value while
    # `.fixed` stays True throughout. The correct test is per-key, not
    # per-transition: for every EXPECTED key, `.fixed` must be True after and
    # the value must equal the expected target (regardless of its `.fixed`
    # state before); for every OTHER key, NEITHER `.fixed` NOR `.value` may
    # have changed at all.
    covered = set()
    value_mismatches = []
    unexpected_touched = []
    for tag, model_by_yd in (('tso', tso_model), *((f'dso{n}', m) for n, m in dso_models.items())):
        for year, by_day in model_by_yd.items():
            for day, model in by_day.items():
                after = _snapshot_all_vars(model)
                b = before[id(model)]
                for key, (val_after, fixed_after) in after.items():
                    var_name, idx = key
                    full_key = (id(model), var_name, idx)
                    val_before, fixed_before = b[key]
                    if full_key in expected:
                        covered.add(full_key)
                        expected_val = expected[full_key]
                        if not fixed_after:
                            value_mismatches.append({'model': tag, 'var': var_name, 'idx': str(idx),
                                                     'note': 'expected fixed but is NOT fixed'})
                        elif not math.isclose(val_after, expected_val, rel_tol=1e-9, abs_tol=1e-9):
                            value_mismatches.append({'model': tag, 'var': var_name, 'idx': str(idx),
                                                     'value': val_after, 'expected': expected_val})
                    else:
                        if (val_after, fixed_after) != (val_before, fixed_before):
                            unexpected_touched.append({
                                'model': tag, 'var': var_name, 'idx': str(idx),
                                'value_before': val_before, 'fixed_before': fixed_before,
                                'value_after': val_after, 'fixed_after': fixed_after,
                            })

    missing_expected = [
        {'model_id': mid, 'var': var, 'idx': str(idx)}
        for (mid, var, idx) in expected
        if (mid, var, idx) not in covered
    ]

    n_p56a_rows_after = {}
    row_residuals = []
    for year, by_day in tso_model.items():
        for day, model in by_day.items():
            n = len(model.p56a_interface_rows) if hasattr(model, 'p56a_interface_rows') else 0
            n_p56a_rows_after[id(model)] = n
            if hasattr(model, 'p56a_interface_rows'):
                for idx in model.p56a_interface_rows:
                    con = model.p56a_interface_rows[idx]
                    body = pe.value(con.body, exception=False)
                    lower = pe.value(con.lower, exception=False) if con.has_lb() else None
                    residual = (body - lower) if (body is not None and lower is not None) else None
                    row_residuals.append(residual)

    total_rows_after = sum(n_p56a_rows_after.values())
    max_row_residual = max((abs(r) for r in row_residuals if r is not None), default=None)

    results['ii_fixing_step'] = {
        'n_expected_fixed_targets': len(expected),
        'n_expected_covered_and_correct': len(covered) - len(value_mismatches),
        'n_unexpected_touched': len(unexpected_touched),
        'unexpected_touched_sample': unexpected_touched[:10],
        'n_value_mismatches': len(value_mismatches),
        'value_mismatch_sample': value_mismatches[:10],
        'n_missing_expected': len(missing_expected),
        'missing_expected_sample': missing_expected[:10],
        'p56a_interface_rows_expected_count': n_interface_rows_expected,
        'p56a_interface_rows_observed_count': total_rows_after,
        'p56a_interface_rows_max_residual': max_row_residual,
        'pass': bool(
            len(unexpected_touched) == 0 and len(value_mismatches) == 0
            and len(missing_expected) == 0
            and total_rows_after == n_interface_rows_expected
            and (max_row_residual is None or max_row_residual < 1e-9)),
    }
    return models


def check_iii_recourse_aggregation(planning, tso_model, dso_models, esso_model, results):
    """Perturb a handful of dispatch Vars directly (`.set_value`, never
    `.fix`/solve) so the aggregate is non-trivial, then verify
    sum(per_block_base_objectives) == gross_operational_cost_including_settlement."""
    perturbed = []
    year = next(iter(planning.transmission_network.years))
    day = next(iter(planning.transmission_network.days))
    t_model = tso_model[year][day]
    for g in list(t_model.generators)[:2]:
        for p in list(t_model.periods)[:3]:
            t_model.pg[g, 0, 0, p].set_value(0.05)
            perturbed.append(('tso', str(g), str(p)))
    for node_id, by_year in dso_models.items():
        d_model = by_year[year][day]
        for g in list(d_model.generators)[:1]:
            for p in list(d_model.periods)[:3]:
                d_model.pg[g, 0, 0, p].set_value(0.02)
                perturbed.append((f'dso{node_id}', str(g), str(p)))

    models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
    per_block = O.per_block_base_objectives(planning, models)
    sum_per_block = sum(v['weighted_base_objective'] for v in per_block.values())

    components = planning.get_operational_recourse_components(models)
    reference = components['gross_operational_cost_including_settlement']

    close = math.isclose(sum_per_block, reference, rel_tol=1e-9, abs_tol=1e-6)

    results['iii_recourse_aggregation'] = {
        'n_perturbed_vars': len(perturbed),
        'perturbed_sample': perturbed[:10],
        'sum_per_block_weighted_base_objective': sum_per_block,
        'gross_operational_cost_including_settlement': reference,
        'gross_operational_cost_settlement_excluded': components['gross_operational_cost'],
        'absolute_difference': sum_per_block - reference,
        'n_blocks': len(per_block),
        'pass': bool(close and sum_per_block != 0.0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    guard = SolveProfileGuard(permitted=(), label='S40 polish-gap zero-solve check').install()
    results = {}
    try:
        (planning, tso_model, dso_models, esso_model, consensus_vars,
         _dual_vars) = _build_admm_ready_state(EVAL_ID)
        check_i_objective_switching(planning, tso_model, dso_models, results)
        check_ii_fixing_step(planning, tso_model, dso_models, consensus_vars, results)
        check_iii_recourse_aggregation(planning, tso_model, dso_models, esso_model, results)
    finally:
        guard.uninstall()

    verify_failures = guard.verify(expected_solves=0)
    all_pass = all(v.get('pass') for v in results.values())

    payload = {
        'stage': 'P5.15 Addendum 22 item (4) -- Step 3.5 polish-gap zero-solve structural checks',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Step 3.5, Addendum 22 item 4_step_3_5',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts),
                                'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': bool(all_pass),
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S40-POLISH-GAP-CHECKS] wrote {OUT_PATH}')
    print(f'[S40-POLISH-GAP-CHECKS] all_checks_pass={all_pass} '
          f'solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
