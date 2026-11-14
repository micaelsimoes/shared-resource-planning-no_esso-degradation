"""
P5.15 Step 3.2 + 3.3(a) v2 (s32 worker task, residual balancing on s_rho_part)
-- zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a); binding
specification data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json
(supersedes v1 frozen_s32_spec_v1_14a18674.json). The single change under
test: `_update_admm_penalties`'s balancing decision uses
`dual_ratio_balance = s_rho_part / eps_dual` (the rho-dependent part of s
only) instead of the full `dual_ratio = s / eps_dual`. The Boyd stopping
test is UNCHANGED and still uses the full `s` (`dual_pass = s <= eps_dual`).

Verifies, on FRESHLY BUILT C* ADMM models (a NEW eval id) and on synthetic
single-entry perturbations of the ADMM consensus/dual variables, that:

  (a) the balancing unit test covers increase, decrease, dead band, no
      freeze and failure hold, PLUS a case built from the frozen spec v2's
      own `change_from_v1.why` cycle-1 PF/V numbers where the proximal part
      flips the v1 decision (decrease) but not the v2 one (held);
  (b) the stop-rule unit test still uses the FULL s: a real
      `get_admm_boyd_residual_metrics` case with s_rho_part <= eps_dual < s
      does NOT pass the dual test;
  (c) `_update_admm_penalties` still changes ONLY rho; consensus/dual dicts
      are bit-identical before and after;
  (d) legacy metrics and the Boyd r/s/eps/norms are bit-identical to commit
      2d765573's function on a snapshot -- ONLY `dual_ratio_balance` is new;
  (e) the augmented-Lagrangian objective / proximal-centre construction
      functions are byte-identical to commit 2d765573 (pre-v2, and pre-v1
      too -- neither stage touched them);
  (f) other case params (CS1) still load;
  (g) preserved (S31C) fixtures still unpickle;
  (h) the hierarchical and uncoordinated paths have no hunks relative to
      the current committed HEAD (i.e. this v2 change touched nothing
      outside the three permitted production sites).

A BLOCKING `SolveProfileGuard` (permitted call sites = (), i.e. zero solves
anywhere) is armed for the whole check. Reuses the v1 s32 zero-solve check's
helpers (`_build_admm_ready_state`, `_perturb_single_entry`, `_reset_rho`,
`check3_lambda_antisymmetry`, `check4_unit_conversions`,
`check9_other_case_params_load`, `check10_fixtures_unpickle`) by import, per
the task's explicit permission -- they are unmodified by v2 (only the
balancing/stop-rule/comparison-target checks differ).

    python p515_s32v2_zero_solve_checks.py

Writes data/SRP1/Results/P515S32/zero_solve_checks_v2/zero_solve_checks_v2.json
(a NEW directory; nothing under the committed v1
data/SRP1/Results/P515S32/zero_solve_checks/ is read-written except by
import of its Python module).
"""

import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'zero_solve_checks_v2')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks_v2.json')
SNAPSHOT_PATH = os.path.join(OUT_DIR, 'synthetic_admm_snapshot_v2.pkl')

SPEC_V2_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'frozen_s32_spec_v2_516bd749.json')
SPEC_V2_SHA256 = '516bd749c5eff6c5f26c0dac570d10964ca989b2c64394ac1ba5deca32356b85'
PRE_CHANGE_COMMIT = '2d765573'  # P5.15 Step 3.2 + 3.3(a) Boyd stopping rule / residual balancing (production, v1)


def _spec_v2_hash():
    with open(SPEC_V2_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load_module_at_commit(commit, alias):
    """Loads `shared_resources_planning.py` as it existed at `commit` as a
    SEPARATE module named `alias`, via `git show <commit>:...`. Same
    technique as `p515_s32_zero_solve_checks._load_pre_change_module`,
    generalized to an explicit commit rather than HEAD, per this task's
    instruction to compare against commit 2d765573 specifically."""
    proc = subprocess.run(
        ['git', 'show', f'{commit}:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    source = proc.stdout
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix=f'p515_s32v2_{alias}_')
    tmp_path = os.path.join(tmp_dir, f'_shared_resources_planning_{alias}.py')
    with open(tmp_path, 'w') as handle:
        handle.write(source)
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, proc.stdout


def _fake_boyd_metrics_v2(ratios):
    """ratios: {'v': (primal_ratio, dual_ratio, dual_ratio_balance), ...}.
    Unlike v1's `_fake_boyd_metrics`, `dual_ratio` (full s/eps_dual, feeds
    the stopping test) and `dual_ratio_balance` (s_rho_part/eps_dual, feeds
    the v2 balancing decision) are independent, so a proximal-inflated
    `dual_ratio` can be constructed alongside a smaller `dual_ratio_balance`
    -- exactly the frozen-spec v2 scenario."""
    channels = {}
    for group in ('v', 'pf', 'ess'):
        primal_ratio, dual_ratio, dual_ratio_balance = ratios[group]
        channels[group] = {
            'r': primal_ratio, 's': dual_ratio, 'eps_pri': 1.0, 'eps_dual': 1.0,
            'primal_ratio': primal_ratio, 'dual_ratio': dual_ratio,
            'dual_ratio_balance': dual_ratio_balance,
            'primal_pass': primal_ratio <= 1.0, 'dual_pass': dual_ratio <= 1.0,
            'channel_pass': primal_ratio <= 1.0 and dual_ratio <= 1.0,
            'norm_x': 1.0, 'norm_z': 1.0, 'norm_y': 1.0,
            's_rho_part': dual_ratio_balance, 's_proximal_part': 0.0, 'proximal_share': 0.0,
            'n_entries': 1,
        }
    channels['all_boyd_pass'] = all(channels[g]['channel_pass'] for g in ('v', 'pf', 'ess'))
    channels['eps_abs'] = 1e-5
    channels['eps_rel'] = 1e-4
    channels['boyd_eps_source'] = 'case_file'
    return channels


def check_a_balancing_unit_test(planning, tso_model, dso_models, esso_model, results):
    """Check (a): increase / decrease / dead band / no freeze / failure hold
    (same 5 sub-cases as v1's check 7, re-exercised against v2's
    `dual_ratio_balance`-gated code path), PLUS a 6th sub-case built from the
    frozen spec v2's own `change_from_v1.why` numbers (commit 7305a476
    preflight, cycle 1, rho=gamma=1): PF `dual_ratio` (full s) 7734.2 vs
    v1's decrease threshold 3*2541.6=7624.8 -> v1 DECREASED; PF
    `dual_ratio_balance` (rho part only) 5468.9 vs the SAME threshold ->
    v2 HOLDS. V: `dual_ratio` 178.13 vs 5*31.34=156.69 -> v1 DECREASED;
    `dual_ratio_balance` 125.96 vs the same threshold -> v2 HOLDS."""
    admm_params = planning.params.admm
    residual_metrics = V1._dummy_residual_metrics(admm_params)
    sub_results = {}

    # -- increase: primal_ratio > 5 * dual_ratio_balance ---------------------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics_v2({'v': (10.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['increase'] = {
        'action': actions['v'], 'before': before['v'], 'after': after['v'],
        'pass': actions['v'] == 'increased' and abs(after['v'] - 1.5) < 1e-12,
    }

    # -- decrease (PF, threshold 3.0): dual_ratio_balance > 3 * primal_ratio -
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics_v2({'v': (1.0, 1.0, 1.0), 'pf': (1.0, 30.0, 10.0), 'ess': (1.0, 1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['decrease'] = {
        'action': actions['pf'], 'before': before['pf'], 'after': after['pf'],
        'pass': actions['pf'] == 'decreased' and abs(after['pf'] - (1.0 / 1.5)) < 1e-12,
    }

    # -- dead band (held): neither ratio dominates ----------------------------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics_v2({'v': (1.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (2.0, 3.0, 2.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['dead_band_held'] = {
        'action': actions['ess'], 'before': before['ess'], 'after': after['ess'],
        'pass': actions['ess'] == 'held' and after['ess'] == before['ess'],
    }

    # -- no freeze: both ratios <= 1 ("legacy adaptation-converged") yet
    #    imbalanced beyond the band -> still acts (freeze clause removed) ----
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics_v2({'v': (0.5, 0.05, 0.05), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['no_freeze'] = {
        'primal_ratio': 0.5, 'dual_ratio_balance': 0.05,
        'both_ratios_le_1_legacy_converged': True,
        'action': actions['v'], 'before': before['v'], 'after': after['v'],
        'pass': actions['v'] == 'increased' and abs(after['v'] - 1.5) < 1e-12,
    }

    # -- failure hold: allow_update=False -> rho unchanged regardless --------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics_v2({'v': (10.0, 1.0, 1.0), 'pf': (1.0, 30.0, 10.0), 'ess': (1.0, 1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=False)
    sub_results['failure_hold'] = {
        'action_v': actions['v'], 'action_pf': actions['pf'],
        'before': before, 'after': after,
        'pass': (
            actions['v'] == 'held after solver failure'
            and actions['pf'] == 'held after solver failure'
            and after == before
        ),
    }

    # -- proximal-flip: frozen spec v2 change_from_v1.why numbers ------------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    increase_ratio = admm_params.penalty_update['residual_balance_ratio']
    decrease_ratio_pf = admm_params.penalty_update.get(
        'residual_balance_ratio_pf_decrease', increase_ratio)
    boyd = _fake_boyd_metrics_v2({
        'v': (31.34, 178.13, 125.96),
        'pf': (2541.6, 7734.2, 5468.9),
        'ess': (1.0, 1.0, 1.0),
    })
    v1_would_decrease_v = boyd['v']['dual_ratio'] > increase_ratio * boyd['v']['primal_ratio']
    v1_would_decrease_pf = boyd['pf']['dual_ratio'] > decrease_ratio_pf * boyd['pf']['primal_ratio']
    v2_decides_v = boyd['v']['dual_ratio_balance'] > increase_ratio * boyd['v']['primal_ratio']
    v2_decides_pf = boyd['pf']['dual_ratio_balance'] > decrease_ratio_pf * boyd['pf']['primal_ratio']
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['proximal_flip_v2_spec_numbers'] = {
        'source': 'frozen_s32_spec_v2_516bd749.json change_from_v1.why (preflight commit 7305a476, cycle 1, rho=gamma=1)',
        'v_dual_ratio_full': boyd['v']['dual_ratio'], 'v_dual_ratio_balance': boyd['v']['dual_ratio_balance'],
        'v_primal_ratio': boyd['v']['primal_ratio'], 'v_decrease_threshold': increase_ratio * boyd['v']['primal_ratio'],
        'pf_dual_ratio_full': boyd['pf']['dual_ratio'], 'pf_dual_ratio_balance': boyd['pf']['dual_ratio_balance'],
        'pf_primal_ratio': boyd['pf']['primal_ratio'], 'pf_decrease_threshold': decrease_ratio_pf * boyd['pf']['primal_ratio'],
        'v1_rule_would_decrease_v': bool(v1_would_decrease_v),
        'v1_rule_would_decrease_pf': bool(v1_would_decrease_pf),
        'v2_rule_decides_decrease_v': bool(v2_decides_v),
        'v2_rule_decides_decrease_pf': bool(v2_decides_pf),
        'v2_action_v': actions['v'], 'v2_action_pf': actions['pf'],
        'v_before': before['v'], 'v_after': after['v'],
        'pf_before': before['pf'], 'pf_after': after['pf'],
        'pass': (
            v1_would_decrease_v is True and v1_would_decrease_pf is True
            and v2_decides_v is False and v2_decides_pf is False
            and actions['v'] == 'held' and after['v'] == before['v']
            and actions['pf'] == 'held' and after['pf'] == before['pf']
        ),
    }

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['balancing_unit_test'] = {
        'sub_results': sub_results,
        'pass': all(v['pass'] for v in sub_results.values()),
    }


def check_b_stop_rule_uses_full_s(planning, tso_model, dso_models, esso_model,
                                   consensus_vars, dual_vars, results):
    """Check (b): a REAL `get_admm_boyd_residual_metrics` case, on the V
    channel, with s_rho_part <= eps_dual < s (full) must NOT pass the dual
    test -- i.e. `dual_ratio_balance <= 1` while `dual_pass` (which the
    stopping rule reads) is False. Constructed from actual model rho/gamma
    (both 1.0, case-file default) and a single perturbed (node, year, day,
    p) V-channel entry; every other channel/coordinate stays at its
    zero-contribution build default."""
    admm_params = planning.params.admm
    node_id = planning.active_distribution_network_nodes[0]
    year = next(iter(planning.years))
    day = next(iter(planning.days))
    p = 0
    network = planning.transmission_network.network[year][day]
    v_base = network.get_node_base_kv(node_id)

    rho_v = float(pe.value(tso_model[year][day].rho_v))
    gamma_v = admm_params.proximal_regularization['tso']['gamma']['v'] if (
        admm_params.proximal_regularization['enabled']
        and admm_params.proximal_regularization['tso']['enabled']
    ) else 0.0

    # -- pass 0: discover n_entries['v'] (build-default state, all zero) -----
    boyd0 = srp.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
    n_v = boyd0['v']['n_entries']
    eps_abs, eps_rel = boyd0['eps_abs'], boyd0['eps_rel']

    # -- construct dz_v (normalized) = 0.02; r_v = 0 (x=z, isolates dual side)
    dz_norm = 0.02
    s_rho_part_expected = rho_v * dz_norm
    s_full_expected = math.sqrt(rho_v ** 2 + gamma_v ** 2) * dz_norm
    if not (s_full_expected > s_rho_part_expected):
        raise RuntimeError(
            'gamma_v must be > 0 for this test to be meaningful (case-file default is 1.0); '
            f'got rho_v={rho_v}, gamma_v={gamma_v}')
    eps_dual_target = 0.5 * (s_rho_part_expected + s_full_expected)  # strictly between the two
    norm_y_needed = (eps_dual_target - math.sqrt(n_v) * eps_abs) / eps_rel

    consensus_vars['vmag']['tso']['prev'][node_id][year][day][p] = v_base * 1.0
    consensus_vars['vmag']['tso']['current'][node_id][year][day][p] = v_base * (1.0 + dz_norm)
    consensus_vars['vmag']['dso']['current'][node_id][year][day][p] = v_base * (1.0 + dz_norm)  # r_v = 0
    dual_vars['vmag']['dso']['current'][node_id][year][day][p] = norm_y_needed * v_base

    boyd1 = srp.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
    v_entry = boyd1['v']

    ok = (
        abs(v_entry['s_rho_part'] - s_rho_part_expected) < 1e-9
        and abs(v_entry['s'] - s_full_expected) < 1e-9
        and abs(v_entry['r']) < 1e-12
        and v_entry['primal_pass'] is True
        and v_entry['s_rho_part'] <= v_entry['eps_dual']
        and v_entry['dual_ratio_balance'] <= 1.0
        and v_entry['s'] > v_entry['eps_dual']
        and v_entry['dual_pass'] is False
        and v_entry['dual_ratio'] > 1.0
        and v_entry['channel_pass'] is False
    )

    results['stop_rule_uses_full_s'] = {
        'method': 'real get_admm_boyd_residual_metrics call, single perturbed V-channel '
                  'entry, dz_norm=0.02, rho_v/gamma_v from the case-file default (1.0/1.0), '
                  'eps_dual set (via one dual entry) strictly between s_rho_part and s.',
        'rho_v': rho_v, 'gamma_v': gamma_v, 'n_v': n_v, 'dz_norm': dz_norm,
        's_rho_part_expected': s_rho_part_expected, 's_full_expected': s_full_expected,
        'eps_dual_target': eps_dual_target, 'norm_y_needed': norm_y_needed,
        'observed': {
            'r': v_entry['r'], 's': v_entry['s'], 's_rho_part': v_entry['s_rho_part'],
            'eps_pri': v_entry['eps_pri'], 'eps_dual': v_entry['eps_dual'],
            'primal_ratio': v_entry['primal_ratio'], 'dual_ratio': v_entry['dual_ratio'],
            'dual_ratio_balance': v_entry['dual_ratio_balance'],
            'primal_pass': v_entry['primal_pass'], 'dual_pass': v_entry['dual_pass'],
            'channel_pass': v_entry['channel_pass'],
        },
        'pass': ok,
    }


def check_c_penalties_isolated(planning, tso_model, dso_models, esso_model,
                                consensus_vars, dual_vars, results):
    """Check (c): `_update_admm_penalties` changes ONLY rho Params under the
    v2 code path; consensus/dual dicts are bit-identical before/after."""
    admm_params = planning.params.admm
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)

    boyd_metrics = _fake_boyd_metrics_v2({'v': (10.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})
    residual_metrics = V1._dummy_residual_metrics(admm_params)

    consensus_before = deepcopy(consensus_vars)
    dual_before = deepcopy(dual_vars)

    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, admm_params, allow_update=True)

    consensus_unchanged = (consensus_vars == consensus_before)
    dual_unchanged = (dual_vars == dual_before)
    rho_v_changed_as_expected = (actions['v'] == 'increased' and after['v'] == before['v'] * 1.5)

    results['penalties_update_isolated'] = {
        'actions': actions, 'rho_before': before, 'rho_after': after,
        'consensus_vars_unchanged': consensus_unchanged,
        'dual_vars_unchanged': dual_unchanged,
        'rho_v_changed_as_expected': rho_v_changed_as_expected,
        'pass': consensus_unchanged and dual_unchanged and rho_v_changed_as_expected,
    }
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)


def check_d_boyd_and_legacy_bit_identical(planning, tso_model, dso_models, esso_model,
                                           consensus_vars, dual_vars, coord, results):
    """Check (d): legacy metrics AND every pre-existing Boyd field
    (r/s/s_rho_part/s_proximal_part/eps_pri/eps_dual/norm_x/norm_z/norm_y/
    primal_ratio/dual_ratio/*_pass/n_entries, per channel, plus the
    top-level eps_abs/eps_rel/boyd_eps_source/all_boyd_pass) are bit-
    identical between the current module and commit 2d765573's module, on a
    pickled-and-reloaded synthetic snapshot (this stage's OWN, INCLUDING
    dual_vars -- v1's `synthetic_admm_snapshot.pkl` omitted dual_vars, since
    v1 check1 only needed it for the legacy-metrics comparison; the Boyd
    comparison here needs both). `dual_ratio_balance` is the ONLY new field
    (absent from the pre-change dict, present in the new one)."""
    pre_module, _pre_source = _load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_2d765573')

    V1._refuse_overwrite(SNAPSHOT_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot = {
        'tso_model': tso_model, 'dso_models': dso_models, 'esso_model': esso_model,
        'consensus_vars': consensus_vars, 'dual_vars': dual_vars,
        'active_distribution_network_nodes': list(planning.active_distribution_network_nodes),
        'years': list(planning.years), 'days': list(planning.days),
        'num_instants': planning.num_instants,
        'note': 'Synthetic ADMM-ready state built via the same production zero-solve '
                'pipeline as p515_s32_zero_solve_checks.py check1/2 (single-entry '
                'perturbation), INCLUDING dual_vars (v1 omitted it), for the v2 '
                'legacy+Boyd bit-identical-to-2d765573 comparison.',
    }
    with open(SNAPSHOT_PATH, 'wb') as handle:
        pickle.dump(snapshot, handle)
    with open(SNAPSHOT_PATH, 'rb') as handle:
        loaded = pickle.load(handle)
    tso_model, dso_models, esso_model = loaded['tso_model'], loaded['dso_models'], loaded['esso_model']
    consensus_vars, dual_vars = loaded['consensus_vars'], loaded['dual_vars']

    admm_params = planning.params.admm

    new_legacy = srp.get_admm_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars)
    pre_legacy = pre_module.get_admm_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars)

    def _numeric_subset(d):
        return {
            k1: {k2: v2 for k2, v2 in d[k1].items() if isinstance(v2, (int, float))}
            for k1 in ('primal', 'dual')
        }

    new_legacy_numeric = _numeric_subset(new_legacy)
    pre_legacy_numeric = _numeric_subset(pre_legacy)
    legacy_identical = (new_legacy_numeric == pre_legacy_numeric)

    new_boyd = srp.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
    pre_boyd = pre_module.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)

    per_channel_diff = {}
    boyd_fields_identical = True
    for group in ('v', 'pf', 'ess'):
        new_entry = dict(new_boyd[group])
        pre_entry = dict(pre_boyd[group])
        new_only_keys = set(new_entry) - set(pre_entry)
        common_keys = set(new_entry) & set(pre_entry)
        mismatches = {
            k: {'new': new_entry[k], 'pre': pre_entry[k]}
            for k in sorted(common_keys) if new_entry[k] != pre_entry[k]
        }
        channel_ok = (new_only_keys == {'dual_ratio_balance'}) and (len(mismatches) == 0)
        boyd_fields_identical = boyd_fields_identical and channel_ok
        per_channel_diff[group] = {
            'new_only_keys': sorted(new_only_keys),
            'mismatches': mismatches,
            'pass': channel_ok,
        }

    top_level_common = ('eps_abs', 'eps_rel', 'boyd_eps_source', 'all_boyd_pass')
    top_level_mismatches = {
        k: {'new': new_boyd[k], 'pre': pre_boyd[k]}
        for k in top_level_common if new_boyd[k] != pre_boyd[k]
    }
    top_level_ok = (len(top_level_mismatches) == 0)

    results['legacy_and_boyd_bit_identical_to_2d765573'] = {
        'snapshot_coordinate': coord,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'legacy_metrics_identical': legacy_identical,
        'legacy_new': new_legacy_numeric, 'legacy_pre': pre_legacy_numeric,
        'boyd_per_channel': per_channel_diff,
        'boyd_top_level_mismatches': top_level_mismatches,
        'boyd_top_level_pass': top_level_ok,
        'pass': legacy_identical and boyd_fields_identical and top_level_ok,
    }


def check_e_al_objective_unchanged_vs_2d765573(results):
    """Check (e): AL objective / proximal-centre construction functions are
    byte-identical to commit 2d765573 (pre-v1 AND pre-v2 -- neither stage
    touched them)."""
    pre_module, _pre_source = _load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_2d765573_e')
    fn_names = (
        'update_transmission_model_to_admm',
        'update_distribution_models_to_admm',
        'update_shared_energy_storage_model_to_admm',
        '_prepare_transmission_objectives_for_admm',
        '_prepare_distribution_objectives_for_admm',
        '_update_tso_proximal_centres_after_solve',
    )
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        pre_source = inspect.getsource(getattr(pre_module, name))
        identical = (new_source == pre_source)
        per_fn[name] = {
            'identical': identical,
            'new_sha256': hashlib.sha256(new_source.encode()).hexdigest(),
            'pre_sha256': hashlib.sha256(pre_source.encode()).hexdigest(),
        }
        all_identical = all_identical and identical
    results['al_objective_unchanged'] = {
        'pre_change_commit': PRE_CHANGE_COMMIT, 'functions': per_fn, 'pass': all_identical,
    }


def check_f_other_case_params_load(results):
    """Check (f): reuses `p515_s32_zero_solve_checks.check9_other_case_params_load`
    unchanged (not affected by v2 -- eps_abs/eps_rel loading is untouched)."""
    V1.check9_other_case_params_load(results)


def check_g_fixtures_unpickle(results):
    """Check (g): reuses `p515_s32_zero_solve_checks.check10_fixtures_unpickle`
    unchanged (same S31C_FIXTURES set)."""
    V1.check10_fixtures_unpickle(results)


def check_h_hierarchical_uncoordinated_untouched(results):
    """Check (h): hierarchical/uncoordinated paths have no hunks relative to
    the CURRENT committed HEAD (which already contains v1's production
    change; this isolates v2's own uncommitted diff)."""
    proc = subprocess.run(
        ['git', 'show', 'HEAD:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='p515_s32v2_head_')
    tmp_path = os.path.join(tmp_dir, '_shared_resources_planning_head.py')
    with open(tmp_path, 'w') as handle:
        handle.write(proc.stdout)
    import importlib.util
    spec = importlib.util.spec_from_file_location('srp_head', tmp_path)
    head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head_module)

    fn_names = (
        '_run_operational_planning_hierarchical',
        '_run_operational_planning_without_coordination',
    )
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        head_source = inspect.getsource(getattr(head_module, name))
        identical = (new_source == head_source)
        per_fn[name] = {'identical': identical}
        all_identical = all_identical and identical

    diff = subprocess.run(
        ['git', 'diff', 'HEAD', '--', 'shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    hunk_headers = [line for line in diff.splitlines() if line.startswith('@@')]
    hunk_context = [line for line in diff.splitlines() if line.startswith('@@') or line.startswith('+def ') or line.startswith('-def ')]

    results['hierarchical_uncoordinated_untouched'] = {
        'method': 'exact source-text equality (git HEAD, which already contains v1, vs '
                  'the current uncommitted tree, which adds v2) for both path functions, '
                  'PLUS every diff hunk header (isolating v2''s own hunks).',
        'functions': per_fn,
        'diff_hunk_count': len(hunk_headers),
        'diff_hunk_headers': hunk_headers,
        'pass': all_identical,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    V1._refuse_overwrite(OUT_PATH)

    observed_spec_hash = _spec_v2_hash()
    if observed_spec_hash != SPEC_V2_SHA256:
        raise RuntimeError(
            f'frozen s32 spec v2 hash mismatch: file={observed_spec_hash} expected={SPEC_V2_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='S32v2 zero-solve check').install()
    results = {}
    try:
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
            V1._build_admm_ready_state('p515s32v2_zero_solve_check'))

        coord = V1._perturb_single_entry(planning, tso_model, dso_models, consensus_vars, dual_vars)

        check_a_balancing_unit_test(planning, tso_model, dso_models, esso_model, results)
        check_b_stop_rule_uses_full_s(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results)
        check_c_penalties_isolated(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results)
        check_d_boyd_and_legacy_bit_identical(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, coord, results)
        check_e_al_objective_unchanged_vs_2d765573(results)
        check_f_other_case_params_load(results)
        check_g_fixtures_unpickle(results)
        check_h_hierarchical_uncoordinated_untouched(results)

    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.2 + 3.3(a) v2 (s32 worker task, balance on s_rho_part) -- zero-solve verification',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a)',
            'data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json',
        ],
        'spec_file': os.path.relpath(SPEC_V2_PATH, REPO),
        'spec_file_sha256': SPEC_V2_SHA256,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S32v2] wrote {OUT_PATH}')
    print(f'[S32v2] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
