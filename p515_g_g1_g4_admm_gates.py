"""
P5.15 -- G1-G4 gate runner.

Authority: PLANNER_BRIEF_2026-09-13.md Step 1 "Gate" section, as amended by Addendum 1
(G5 reporting), Addendum 3 item 3 (G1 re-specified as a per-node reconciliation gate),
Addendum 4 (per-cycle ESSO detector logging standing requirement, G1 confirmed as
specified), Addendum 5 (log-handling fix; no harness-only workaround) and Addendum 6
(campaign capture: heartbeat, per-period ESSO leak-mechanism capture and
classification, network-failure capture -- authorized "HARNESS change plus a short
smoke test" task, 2026-09-14).

This harness does NOT reimplement any production solve. It reuses, verbatim, the already
-repaired stage harnesses:
  * p514_n_instrumented_cstar.py  (control / perturbation arm: assert_capture_paths_exist,
    capture_esso, and every module-level constant -- S_INV, E_INV, INVEST_YEAR, BUDGET,
    REL, CAP, RHO, REFERENCE_RECOURSE, EFC_BINDING_THRESHOLD, PERMITTED);
  * p514_l_capacity_ladder.py     (ladder initialization-stage harness: main() called
    directly, with its module-level OUT redirected so it cannot collide with the
    committed P514L artifacts).

P5.15-G1PREP (this revision) removes the harness-only ESSO-log-isolation workaround --
production now writes one fresh, logs_dir-resolved ESSO IPOPT log per solve (P5.15-F,
commit 7ca40b93) -- and adds the Addendum-6 campaign capture:
  1. A per-cycle heartbeat file, written atomically after every ESSO coordination solve.
  2. Per-period ESSO leak-mechanism capture (pch, pdch, pnet, s_max, the IPOPT bound
     multipliers of both legs, and every constraint dual whose row references that
     period's pch/pdch or the cohort's degradation Vars), one JSONL file per node per
     round under `esso_capture/<label>/`.
  3. A per-solve leak classification (barrier-set / not-barrier-set / indeterminate),
     one line per solve in `leak_classification_<label>.jsonl`.
  4. Network-failure capture and classification, built from the (no-longer-discarded)
     tee'd stdout of the run plus the FrozenSMOPF snapshot directory, written to
     `network_failures_<label>.jsonl`.
  5. `assert_g_capture_paths` extended per Addendum 6 item 7 (rule eleven): production
     logs_dir absolute on planning/ESSO/TSO, wrapper hooks installed, output root
     fresh, ESSO Suffixes present on a freshly-built (unsolved) probe subproblem.

The new artifacts are LABELED per arm (`heartbeat_<label>.json`, not `heartbeat.json`)
because `run_admm_arm` is called from several CLI gates (g1, g2, g4b, g3_full) that
share ONE output root (`OUT`, `data/SRP1/Results/P515G`) across SEPARATE process
invocations of this script -- an unlabeled, un-guarded filename would collide across
arms and break "keep G2/G3-full/G4 arms working" (see WORKER_REPORT_G1PREP.md). The
`g1` CLI gate is the one exception: it gets its OWN fresh output root,
`data/SRP1/Results/P515G1/`, checked for freshness before anything is written (Addendum
6 item 2 -- "no overwrite, no reuse").

RULE ELEVEN: capture paths are asserted before any solve is attempted.

Writes ONLY new files, under data/SRP1/Results/P515G/ (existing arms) or
data/SRP1/Results/P515G1/ (fresh, the g1 CLI gate) -- neither repaired harness's
default output directory is touched, because both collide with already-committed
artifacts (P514N: esso_models_control.pkl, n1_control.json, ...; P514L: ladder_s1.json,
...).

    python p515_g_g1_g4_admm_gates.py <gate>
    gate in {g1, g2, g4b, g3_init, g3_full}   (g4a == g1; run g1 twice for G4)
"""

import hashlib
import inspect
import json
import os
import pickle
import re
import sys
import time
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p512_a_cold_rescaled_convergence as A  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import p514_l_capacity_ladder as L  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
from definitions import PENALTY_FLEXIBILITY, PENALTY_GENERATION_CURTAILMENT  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G')
os.makedirs(OUT, exist_ok=True)

# Addendum 6 item 2: the g1 CLI gate's OWN fresh output root -- never shared with OUT,
# which holds residue from earlier, killed campaigns (see .p515_g_gate.lock).
OUT_G1 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G1')
# P5.15 Addendum 8: re-runs under the new production baseline (EPS_ESSO_THROUGHPUT = 1e-5,
# ESSO tol 1e-10 / acceptable_tol 1e-9, explicit recovery policy with tier 2, limited-memory
# entries removed). These arms apply NO overrides: every value comes from production.
OUT_G1B = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G1B')
OUT_G3F_B = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G3F_B')
# P5.15 Step 3.0 (Addendum 9): the new-baseline G1 repeated, production defaults, no overrides;
# compared bitwise with P515G1B on per-cycle recourse, residuals, SoH and detector.
OUT_S30 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S30')
# P5.15 Step 3.1 (S31 worker task, Part 3): the signed-table baseline campaign, production
# defaults, no overrides -- own fresh root, never shared with P515S30/P515G1B.
OUT_S31 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31_run')

# G2PREP Fix 2: every remaining arm gets its OWN fresh output root too, for the same
# reason g1 does (OUT/P515G is shared residue from earlier campaigns and multiple
# arms writing `results_dir`-anchored artifacts into the SAME root is exactly the G1
# overwrite mechanism Fix 1 addresses -- one arm's `out_dir` must never collide with
# another's).
OUT_G2 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G2')
# r2: the first G3-full attempt (P515G3F, eval ids p515g3f_node7 / p515g3f_probe) stopped on a
# false pre-check and is preserved as a failed attempt; the re-run uses fresh names.
OUT_G3F = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G3F_r2')
OUT_G4 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G4')

# G2PREP Fix 1: the shared, PRESERVED FrozenSMOPF tree that must never be written to
# by any arm launched from this harness (every arm's `results_dir` is redirected away
# from it -- see `_set_results_dir_for_arm` below). Hashed before/after every arm as a
# post-run integrity check (CLAUDE.md rule: "commit or hash-record the settling
# artifact for any claim you commit").
SHARED_FROZEN_SMOPF_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF')


def _hash_dir_pkls(dir_path):
    """sha256 of every non-recursive `*.pkl` in `dir_path`, keyed by filename. Used to
    prove (not assert) that a shared, preserved directory was not touched by an arm."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith('.pkl'):
            continue
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        digest = hashlib.sha256()
        with open(fpath, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        out[fname] = digest.hexdigest()
    return out


def _set_results_dir_for_arm(planning, results_dir):
    """G2PREP Fix 1.

    Cause of the G1 overwrite: `p56a_oracle.fresh_planning` isolates `logs_dir` per
    eval (planning, every holder `O._holders()` walks -- transmission_network, each
    distribution_network, each holder's per-year/per-day `network[year][day]` object
    -- and `shared_ess_data`) but leaves `results_dir` untouched, still pointing at
    the value frozen into the deep-copied baseline at `SharedResourcesPlanning.__init__`
    time: `data/SRP1/Results` (shared_resources_planning.py:56, absolute since P5.15-F).
    Every save-path callback below reads `<holder>.results_dir` (not `logs_dir`), so
    two concurrent/sequential arms both write into the SAME
    `data/SRP1/Results/FrozenSMOPF/`:

      * `save_failed_tso_block` / `save_selected_tso_comparator`
        (shared_resources_planning.py:4460-4487) -> `transmission_network.results_dir`
        (line 4463, 4478).
      * `save_failed_dso_block` / `save_selected_dso_comparator`
        (shared_resources_planning.py:4651-4691) -> `distribution_network.results_dir`
        (line 4661, 4679).
      * `_save_frozen_network_block` (shared_resources_planning.py:4552-4594) and
        `_save_frozen_smopf_block` (shared_resources_planning.py:4510-4549) -- the
        two functions the callbacks above call -- take `save_dir` as an argument; they
        do not read `results_dir` themselves, but every caller passes
        `os.path.join(<holder>.results_dir, 'FrozenSMOPF')`.

    This function redirects EVERY holder `O._holders()` walks for `logs_dir`, doing the
    exact same traversal, but for `results_dir`, plus `shared_ess_data.results_dir`
    (which `fresh_planning` also redirects for `logs_dir`, outside `O._holders()`).
    `network[year][day].results_dir` is included even though no current callback reads
    it directly (network_data.py:183 copies it at construction time from the OLD,
    shared `results_dir`, before `fresh_planning` ever runs) -- redirected here anyway,
    both to mirror the `logs_dir` traversal exactly (as instructed) and because
    `shared_energy_storage_data.py:220` and `network_data.py:143` do read a
    per-object `.results_dir` for (unrelated, non-failure) Excel writers.
    """
    os.makedirs(results_dir, exist_ok=True)
    planning.results_dir = results_dir
    for holder in O._holders(planning):
        if hasattr(holder, 'results_dir'):
            holder.results_dir = results_dir
        for year in holder.years:
            for day in holder.days:
                holder.network[year][day].results_dir = results_dir
    if hasattr(planning.shared_ess_data, 'results_dir'):
        planning.shared_ess_data.results_dir = results_dir


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _require_fresh_output_root(path):
    """Addendum 6 item 2: 'Refuse to start if the arm's output dir already exists
    (no overwrite, no reuse).' Scoped to the g1 CLI gate's dedicated root and to the
    smoke test's dedicated root -- NOT to `OUT` (P515G), which multiple arms
    (g2, g4b, g3_full) legitimately share across separate process invocations."""
    if os.path.exists(path):
        raise RuntimeError(
            f'refusing to start: output root already exists (no overwrite, no reuse): {path}')


def _atomic_write_json(path, obj):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as handle:
        json.dump(obj, handle, indent=1, default=str)
    os.replace(tmp, path)


class _Tee:
    """Write to every stream given. Used so the run's own console output is preserved
    (this is a foreground run) while ALSO being captured to a file, per Addendum 6
    item 1 ('Stop swallowing stdout ... production prints the [WARNING] network
    -failure context there; tee it to a file')."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout(path):
    with open(path, 'w') as handle:
        tee = _Tee(sys.stdout, handle)
        with redirect_stdout(tee):
            yield path


# ======================================================================================
#  Addendum 6 item 4/5 -- ESSO per-period leak-mechanism capture and classification
# ======================================================================================

# Every ConstraintList whose rows can reference a period's pch/pdch or a cohort's
# degradation Vars (shared_energy_storage_data.py `_build_subproblem`). Identified
# generically below via `identify_variables`, never by hard-coded row indices.
_ESSO_DUAL_FAMILIES = (
    'energy_storage_limits',
    'energy_storage_operation_agg',
    'energy_storage_cohort_pnet_share_h3',
    'energy_storage_capacity_degradation',
)

_COMPLEMENTARITY_LINE_RE = re.compile(r'^Complementarity\.+:\s+(\S+)\s+(\S+)', re.MULTILINE)
_OBJECTIVE_LINE_RE = re.compile(r'^Objective\.+:\s+(\S+)\s+(\S+)', re.MULTILINE)


def _build_esso_var_constraint_map(model):
    """Once per node model (cached by the caller): (var_component_name, var_index) ->
    [(constraint_component_name, constraint_list_index), ...], built by scanning every
    row of `_ESSO_DUAL_FAMILIES` with `identify_variables` -- the generic identification
    Addendum 6 item 4 requires, not hard-coded row indices."""
    mapping = {}
    for cname in _ESSO_DUAL_FAMILIES:
        clist = getattr(model, cname, None)
        if clist is None:
            continue
        for idx in clist:
            con = clist[idx]
            for v in identify_variables(con.body, include_fixed=False):
                key = (v.parent_component().name, v.index())
                mapping.setdefault(key, []).append((cname, idx))
    return mapping


def _duals_for_keys(model, var_map, keys):
    seen = set()
    out = []
    for key in keys:
        for tag in var_map.get(key, ()):
            if tag in seen:
                continue
            seen.add(tag)
            cname, idx = tag
            con = getattr(model, cname)[idx]
            dual = model.dual.get(con)
            out.append({'component': cname, 'index': idx,
                        'dual': (pe.value(dual) if dual is not None else None)})
    return out


def _parse_both_barrier_columns(log_path):
    """Addendum 6 item 4: production's own `_parse_ipopt_barrier_terms`
    (shared_energy_storage_data.py) returns only the SCALED `mu_final` and the
    scaled/unscaled `s_obj` RATIO. This harness-local parser reads the SAME per-solve
    log file (never assumed) and returns BOTH Complementarity columns (scaled,
    unscaled) and both Objective columns, taking the LAST occurrence of each line
    (matching production's own Addendum-5 fix: one fresh log per solve, but robust to
    the rare case it is not)."""
    result = {'mu_scaled': None, 'mu_unscaled': None,
              'obj_scaled': None, 'obj_unscaled': None, 'reason': None,
              # Planner addition (P5.15, from the G1-prep smoke): the TERMINAL BARRIER
              # PARAMETER, i.e. the lg(mu) column of the last iteration row. The summary
              # `Complementarity` line is an optimality-error measure, not mu; on the C*
              # path it varied 45% while lg(mu) sat at -8.60 in every solve and
              # x_small = 10**lg(mu)/(s_obj*eps) matched the measured leak to 0.3-1%.
              'lg_mu_terminal': None, 'mu_barrier_scaled': None, 's_obj': None}
    if not log_path or not os.path.exists(log_path):
        result['reason'] = f'IPOPT log not found: {log_path}'
        return result
    text = open(log_path, 'r', errors='replace').read()
    comp_matches = list(_COMPLEMENTARITY_LINE_RE.finditer(text))
    obj_matches = list(_OBJECTIVE_LINE_RE.finditer(text))
    if not comp_matches:
        result['reason'] = 'Complementarity line not found in IPOPT log'
        return result
    if not obj_matches:
        result['reason'] = 'Objective line not found in IPOPT log'
        return result
    try:
        comp = comp_matches[-1]
        obj = obj_matches[-1]
        result['mu_scaled'] = float(comp.group(1))
        result['mu_unscaled'] = float(comp.group(2))
        result['obj_scaled'] = float(obj.group(1))
        result['obj_unscaled'] = float(obj.group(2))
        if result['obj_unscaled'] not in (None, 0.0):
            result['s_obj'] = result['obj_scaled'] / result['obj_unscaled']
        iteration_rows = _ITERATION_ROW_RE.findall(text)
        if iteration_rows:
            result['lg_mu_terminal'] = float(iteration_rows[-1])
            result['mu_barrier_scaled'] = 10.0 ** result['lg_mu_terminal']
    except ValueError as error:
        result['reason'] = f'could not parse floats: {error}'
    return result


# iteration table row: iter, objective, inf_pr, inf_du, lg(mu), ...  (captures lg(mu))
_ITERATION_ROW_RE = re.compile(r'^\s*\d+r?\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)\s', re.MULTILINE)


def _classify_r(r):
    """Addendum 6 item 5, predeclared thresholds (Planner-supplied, not fit to data)."""
    if r is None:
        return 'indeterminate'
    if 0.5 <= r <= 2:
        return 'barrier-set'
    if r < 0.1:
        return 'not barrier-set'
    return 'indeterminate'


def _esso_capture_stamp(cycle_label):
    return 'init' if cycle_label == 'init' else f'cycle{cycle_label}'


def _capture_esso_solve(sed, models, node_diag, esso_capture_dir, cycle_label, hook_state):
    """One node-round of ESSO solves (either the init call or one ADMM cycle's
    coordination solve). Writes `esso_capture_dir/node{id}_{stamp}.jsonl` per node and
    returns the per-solve leak-classification records (Addendum 6 items 4-5)."""
    os.makedirs(esso_capture_dir, exist_ok=True)
    leak_records = []
    for node_id, model in models.items():
        diag = node_diag.get(node_id)
        log_path = diag.get('log_path') if diag else None
        barrier = _parse_both_barrier_columns(log_path)
        mu_unscaled = barrier['mu_unscaled']
        mu_barrier_unscaled = None
        if barrier['mu_barrier_scaled'] is not None and barrier['s_obj'] not in (None, 0.0):
            mu_barrier_unscaled = barrier['mu_barrier_scaled'] / barrier['s_obj']

        var_map = hook_state['var_maps'].get(node_id)
        if var_map is None:
            var_map = _build_esso_var_constraint_map(model)
            hook_state['var_maps'][node_id] = var_map

        # G3-full fix (Planner, 2026-09-14): a node with NO active cohort-periods (zero
        # investment, e.g. nodes 5 and 9 in G3-full) has its pch/pdch fixed, so IPOPT
        # correctly returns no bound multipliers for it. The pre-check must only judge a
        # node that has active cohort-periods, and must not mark itself done otherwise.
        node_has_active_periods = any(
            (not model._esso_cohort_inactive.get(y_inv, False))
            and SED._esso_cohort_pair_is_within_lifetime(model, y_inv, y)
            for y_inv in model.years for y in model.years)
        if not hook_state['zL_checked'] and node_has_active_periods:
            nonempty = any(
                v.parent_component().name == 'es_pch_per_unit' for v in model.ipopt_zL_out
            )
            if not nonempty:
                raise RuntimeError(
                    'STOP (Addendum 6 item 4 pre-check): model.ipopt_zL_out has no '
                    f'entries for es_pch_per_unit after the first ESSO solve '
                    f'(node={node_id}, log={log_path}). IPOPT did not return bound '
                    'multipliers for this variable; not substituting anything.')
            hook_state['zL_checked'] = True

        records = []
        for y_inv in model.years:
            if model._esso_cohort_inactive.get(y_inv, False):
                continue
            for y in model.years:
                if not SED._esso_cohort_pair_is_within_lifetime(model, y_inv, y):
                    continue
                s_max = pe.value(model.es_s_rated_per_unit[y_inv, y])
                cohort_keys = [('es_D_per_unit', (y_inv, y)),
                               ('es_soh_per_unit_cumul', (y_inv, y))]
                for d in model.days:
                    for p in model.periods:
                        pch_var = model.es_pch_per_unit[y_inv, y, d, p]
                        pdch_var = model.es_pdch_per_unit[y_inv, y, d, p]
                        pch = pe.value(pch_var)
                        pdch = pe.value(pdch_var)
                        pnet = pe.value(model.es_pnet[y, d, p])
                        slack_pnet_up, slack_pnet_down = _esso_slack_values(sed, model, y, d, p)
                        zL_pch = model.ipopt_zL_out.get(pch_var)
                        zU_pch = model.ipopt_zU_out.get(pch_var)
                        zL_pdch = model.ipopt_zL_out.get(pdch_var)
                        zU_pdch = model.ipopt_zU_out.get(pdch_var)
                        pch_key = ('es_pch_per_unit', (y_inv, y, d, p))
                        pdch_key = ('es_pdch_per_unit', (y_inv, y, d, p))
                        duals = _duals_for_keys(
                            model, var_map, [pch_key, pdch_key] + cohort_keys)
                        x_small = min(pch, pdch)
                        z_small = zL_pch if pch <= pdch else zL_pdch
                        r = None
                        if z_small is not None and mu_unscaled not in (None, 0.0):
                            r = z_small * x_small / mu_unscaled
                        r_bar = None
                        if z_small is not None and mu_barrier_unscaled not in (None, 0.0):
                            r_bar = z_small * x_small / mu_barrier_unscaled
                        records.append({
                            'node_id': node_id, 'cycle': cycle_label,
                            'y_inv': y_inv, 'y': y, 'd': d, 'p': p,
                            'pch': pch, 'pdch': pdch, 'pnet': pnet, 's_max': s_max,
                            'slack_pnet_up': slack_pnet_up, 'slack_pnet_down': slack_pnet_down,
                            'zL_pch': zL_pch, 'zU_pch': zU_pch,
                            'zL_pdch': zL_pdch, 'zU_pdch': zU_pdch,
                            'duals': duals, 'r': r, 'class': _classify_r(r),
                            'r_bar': r_bar, 'class_bar': _classify_r(r_bar),
                        })

        stamp = _esso_capture_stamp(cycle_label)
        path = os.path.join(esso_capture_dir, f'node{node_id}_{stamp}.jsonl')
        _refuse_overwrite(path)
        with open(path, 'w') as handle:
            for rec in records:
                handle.write(json.dumps(rec, default=str) + '\n')

        max_ratio, n_periods, argmax, measured = SED._complementarity_ratio_for_model(model)
        counts = {}
        counts_bar = {}
        for rec in records:
            counts[rec['class']] = counts.get(rec['class'], 0) + 1
            counts_bar[rec['class_bar']] = counts_bar.get(rec['class_bar'], 0) + 1
        argmax_record = None
        if argmax is not None:
            match = next((rc for rc in records
                          if rc['y_inv'] == argmax['y_inv'] and rc['y'] == argmax['y']
                          and rc['d'] == argmax['d'] and rc['p'] == argmax['p']), None)
            if match is not None:
                biggest = max(
                    match['duals'],
                    key=lambda item: (abs(item['dual']) if item['dual'] is not None else -1),
                    default=None)
                argmax_record = {
                    'y_inv': argmax['y_inv'], 'y': argmax['y'], 'd': argmax['d'],
                    'p': argmax['p'], 'pch': match['pch'], 'pdch': match['pdch'],
                    'pnet': match['pnet'], 'ratio_min_over_smax': max_ratio,
                    'r': match['r'], 'class': match['class'],
                    'r_bar': match['r_bar'], 'class_bar': match['class_bar'],
                    'largest_dual': biggest,
                }
        leak_records.append({
            'node_id': node_id, 'cycle': cycle_label, 'log_path': log_path,
            'mu_scaled': barrier['mu_scaled'], 'mu_unscaled': mu_unscaled,
            'obj_scaled': barrier['obj_scaled'], 'obj_unscaled': barrier['obj_unscaled'],
            'parse_reason': barrier['reason'],
            'complementarity_ratio_max': max_ratio, 'n_active_cohort_periods': n_periods,
            'spurious_throughput_measured': measured,
            'argmax': argmax_record, 'class_counts': counts, 'class_counts_bar': counts_bar,
            'lg_mu_terminal': barrier['lg_mu_terminal'],
            'mu_barrier_scaled': barrier['mu_barrier_scaled'], 's_obj': barrier['s_obj'],
            # predicted small-leg leak: idle period (equal legs) and one-large-leg period
            'predicted_x_idle': (barrier['mu_barrier_scaled'] / (barrier['s_obj'] * SED.EPS_ESSO_THROUGHPUT)
                                 if barrier['mu_barrier_scaled'] is not None and barrier['s_obj'] else None),
            'predicted_x_large_leg': (barrier['mu_barrier_scaled'] / (2.0 * barrier['s_obj'] * SED.EPS_ESSO_THROUGHPUT)
                                      if barrier['mu_barrier_scaled'] is not None and barrier['s_obj'] else None),
        })
    return leak_records


# ======================================================================================
#  Addendum 6 item 6 -- network-failure capture and classification
# ======================================================================================

_CYCLE_LINE_RE = re.compile(r'^\[INFO\] \t - ADMM Iteration (\d+)')
_NET_FAIL_RE = re.compile(
    r'^\[WARNING\] Network (?P<label>.+?) did not converge for (?P<ctx>.+?): (?P<summary>.+)$')
_NET_RETRY_RE = re.compile(
    r'^\[INFO\] Retrying network solve once for (?P<ctx>.+?), cold start')
_NET_RECOVER_OK_RE = re.compile(
    r'^\[INFO\] Network recovery solve succeeded for (?P<ctx>.+)\.$')
# P5.15 Addendum 7 Part 1 items 2/3: tier-2 retry markers (network.py `_run_smopf`
# after Part 1). Distinct wording from the tier-1 lines above -- never matched by
# them (see `_scan_network_failures` docstring addendum below).
_NET_RETRY_TIER2_RE = re.compile(
    r'^\[INFO\] Retrying network solve \(tier 2: cold, mu_strategy=adaptive\) for (?P<ctx>.+?), with')
_NET_RECOVER_TIER2_OK_RE = re.compile(
    r'^\[INFO\] Network tier-2 recovery solve succeeded for (?P<ctx>.+)\.$')
_NET_LOG_RE = re.compile(
    r'^\[WARNING\] IPOPT (?P<label>.+?) log for (?P<ctx>.+?): (?P<path>.+)$')
_TSO_FINAL_RE = re.compile(
    r'^\[ERROR\] Transmission network (?P<name>\S+), year=(?P<year>\S+), day=(?P<day>\S+) '
    r'did not converge: (?P<summary>.+)$')
_DSO_FINAL_RE = re.compile(
    r'^\[WARNING\] Distribution network node=(?P<node>\d+), network=(?P<name>\S+), '
    r'year=(?P<year>\S+), day=(?P<day>\S+) did not converge: (?P<summary>.+)$')
_CTX_RE = re.compile(r'^(?P<name>.+), year=(?P<year>.+), day=(?P<day>.+)$')
_TERMINATION_RE = re.compile(r'termination=([^,|]+)')


def _parse_ctx(ctx):
    m = _CTX_RE.match(ctx.strip())
    if not m:
        return ctx.strip(), None, None
    return m.group('name'), m.group('year'), m.group('day')


def _extract_termination(summary):
    m = _TERMINATION_RE.search(summary)
    return m.group(1).strip() if m else None


def _build_name_to_agent(planning_problem):
    """network_name -> (agent, node_id), sourced from the SAME planning object the
    run being scanned actually used (never hard-coded)."""
    name_to_agent = {}
    tso_name = getattr(planning_problem.transmission_network, 'name', None)
    if tso_name:
        name_to_agent[tso_name] = ('TSO', None)
    for node_id, dso in planning_problem.distribution_networks.items():
        name_to_agent[dso.name] = ('DSO', node_id)
    return name_to_agent


def _new_network_event(name, year, day, cycle, name_to_agent):
    agent, node_id = name_to_agent.get(name, (None, None))
    return {
        'record_type': 'network_block',
        'network_name': name, 'year': year, 'day': day, 'cycle': cycle,
        'agent': agent, 'node_id': node_id,
        'primary_termination': None, 'primary_summary': None, 'primary_log': None,
        'recovery_attempted': False,
        'recovery_termination': None, 'recovery_summary': None, 'recovery_log': None,
        # P5.15 Addendum 7 Part 1 items 2/3: tier-2 fields.
        'tier2_attempted': False,
        'tier2_termination': None, 'tier2_summary': None, 'tier2_log': None,
        'termination': None, 'class': None,
        'final_summary_crosscheck': None, 'note': None,
    }


def _scan_network_failures(stdout_path, name_to_agent):
    """G2PREP Fix 3.

    Rewritten to classify from the tee'd stdout ONLY, using production's own prints
    (`network.py` `_print_network_failure_context`/`_run_smofp` -- verified at
    `network.py:606-650`, `~590-660` after this fix's line shifts) rather than
    re-parsing each network's IPOPT log file, which is CUMULATIVE across the whole
    run (`file_append='yes'`, unaffected by the P5.15-F ESSO-only per-solve-log fix)
    -- its last `EXIT:` line is therefore whichever cycle last touched that
    (network, year, day) combination, not necessarily the cycle being scanned. That
    was the G1 bug (`primary_exit: Optimal Solution Found` on a failing row).

    One EVENT per (network_name, year, day, cycle) -- NOT per (network_name, year,
    day) alone, which is the second G1 bug: the same DSO network solves once per
    cycle, so the same (name, year, day) triple recurs across many cycles (e.g. node
    5's case33_1 failed at cycles 7, 39, 43, 46, 48, 49); keying by the triple alone
    silently merged all of those into ONE block, overwriting the earlier cycles'
    outcome with the latest and collapsing 6 failed cycles into 4 rows (12
    recovered / 4 unrecovered / 0 not_attempted, against a 6-not_attempted ground
    truth). Including `cycle` in the key/record fixes this.

    `network.py`'s `_run_smofp` prints exactly ONE of two possible sequences per
    (network, year, day) solve, both matched from `_NET_FAIL_RE`'s `label` group
    (previously captured but discarded):

      * recovery NOT eligible (`_is_recoverable_network_failure` False -- e.g. no
        `recovery_options` configured for that network's case file, or a
        termination condition outside {internalSolverError, maxIterations,
        infeasible}): exactly ONE print, `attempt_label='solver'` --
        "[WARNING] Network solver did not converge for <net>, year=Y, day=D:
        status=..., termination=T | warm_start=...". -> class 'not_attempted'.
        (No "Retrying" line is ever printed for this event -- the `if
        recovery_attempted:` guard in `_run_smofp` that prints the retry request
        and the `attempt_label='primary solve'` failure line is False.)
      * recovery eligible: FIRST "[WARNING] Network primary solve did not converge
        for <net>, ...: status=..., termination=T | warm_start=..." (opens the
        event), then "[INFO] Retrying network solve once for <net>, ..., cold
        start, ..." (`recovery_attempted=True`), then EITHER "[INFO] Network
        recovery solve succeeded for <net>, ...." -> class 'recovered' (no further
        did-not-converge print exists for this outcome -- `_run_smofp` only calls
        `_print_network_failure_context` again in the `else` branch, which success
        skips), OR "[WARNING] Network recovery solve did not converge for <net>,
        ...: status=..., termination=T | warm_start=False" (`attempt_label=
        'recovery solve'`) -> class 'unrecovered'.

    `[WARNING] IPOPT {attempt_label} log for <net>, ...: <path>` lines (also
    produced by `_print_network_failure_context`, immediately after each
    did-not-converge print) are attached to whichever event is currently open for
    that (name, year, day) key -- there is at most one open event per key at a time
    (a DSO network solves its (year, day) blocks sequentially, never concurrently,
    in the non-parallel path this harness uses).

    Cycle attribution: `[INFO] \\t - ADMM Iteration N` (shared_resources_planning.py
    line ~2225) prints at the very START of cycle N's body, strictly before that
    cycle's DSO/TSO/ESSO solves run (single-threaded, sequential execution) and
    strictly before the NEXT "ADMM Iteration N+1" line. Every line between one
    "ADMM Iteration N" line and the next is therefore unambiguously cycle N's. Lines
    before the first such marker (the ADMM initialization block) are cycle 'init'.
    This rule was already in force in the pre-fix harness and is validated below
    against `g_control.json`'s own `cycle_trajectory` (`local_solves_ok = False`
    cycles); it is NOT changed by this fix, only the block keying is.

    `[ERROR] Transmission network ... did not converge: ...` /
    `[WARNING] Distribution network node=N, network=... did not converge: ...`
    (shared_resources_planning.py, printed by the CALLER after `.optimize()`
    returns, i.e. after any retry already happened) are kept as a CROSS-CHECK only
    (`final_summary_crosscheck`), never as the classification signal.
    """
    if not os.path.exists(stdout_path):
        return []

    events = []          # finalized (classified) events, in file order
    open_by_key = {}      # (name, year, day) -> the event dict currently in flight
    current_cycle = 'init'
    with open(stdout_path, 'r', errors='replace') as handle:
        lines = handle.readlines()
    for line in lines:
        m = _CYCLE_LINE_RE.match(line)
        if m:
            current_cycle = int(m.group(1))
            continue

        m = _NET_FAIL_RE.match(line)
        if m:
            label = m.group('label')
            name, year, day = _parse_ctx(m.group('ctx'))
            key = (name, year, day)
            termination = _extract_termination(m.group('summary'))
            if label == 'primary solve':
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['primary_termination'] = termination
                ev['primary_summary'] = m.group('summary')
                open_by_key[key] = ev
            elif label == 'solver':
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['primary_termination'] = termination
                ev['primary_summary'] = m.group('summary')
                ev['termination'] = termination
                ev['class'] = 'not_attempted'
                open_by_key[key] = ev
                events.append(ev)
            elif label == 'recovery solve':
                # P5.15 Addendum 7 Part 1 items 2/3: production (network.py
                # `_run_smopf`) prints this line EXACTLY ONCE per (network, year,
                # day) tier-1 failure, whether or not a tier-2 attempt follows --
                # it is the sole failure line when tier 2 is off/ineligible, and
                # the intermediate line immediately before "Retrying ... tier 2"
                # otherwise (never printed twice). The class set here is
                # therefore 'unrecovered' unconditionally; if a tier-2 outcome
                # line follows, it overwrites `ev['class']` on this SAME object
                # (already appended below), never appending a second row for
                # this key.
                ev = open_by_key.get(key)
                if ev is None:
                    ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                    ev['note'] = ('recovery-failure print with no matching open '
                                  'primary-solve event (malformed capture)')
                    open_by_key[key] = ev
                ev['recovery_termination'] = termination
                ev['recovery_summary'] = m.group('summary')
                ev['termination'] = termination
                ev['class'] = 'unrecovered'
                events.append(ev)
            elif label == 'tier-2 recovery solve':
                # Tier-2 failed (both tiers failed). `ev` must already be open and
                # already appended via the 'recovery solve' branch above (which
                # always precedes this print in production); the malformed-
                # capture fallback below is defensive only.
                ev = open_by_key.get(key)
                if ev is None:
                    ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                    ev['note'] = ('tier-2-failure print with no matching open '
                                  'primary-solve event (malformed capture)')
                    open_by_key[key] = ev
                    events.append(ev)
                ev['tier2_termination'] = termination
                ev['tier2_summary'] = m.group('summary')
                ev['termination'] = termination
                ev['class'] = 'unrecovered'
            else:
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['primary_summary'] = m.group('summary')
                ev['note'] = f'unrecognized attempt_label={label!r}'
                ev['class'] = 'indeterminate'
                open_by_key[key] = ev
                events.append(ev)
            continue

        m = _NET_RETRY_RE.match(line)
        if m:
            name, year, day = _parse_ctx(m.group('ctx'))
            ev = open_by_key.get((name, year, day))
            if ev is not None:
                ev['recovery_attempted'] = True
            continue

        m = _NET_RETRY_TIER2_RE.match(line)
        if m:
            name, year, day = _parse_ctx(m.group('ctx'))
            ev = open_by_key.get((name, year, day))
            if ev is not None:
                ev['tier2_attempted'] = True
            continue

        m = _NET_RECOVER_OK_RE.match(line)
        if m:
            # Tier-1 success only -- production never prints this text for a
            # tier-2 success (see `_NET_RECOVER_TIER2_OK_RE` below).
            name, year, day = _parse_ctx(m.group('ctx'))
            key = (name, year, day)
            ev = open_by_key.get(key)
            if ev is None:
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['note'] = ('recovery-success print with no matching open '
                              'primary-solve event (malformed capture)')
                open_by_key[key] = ev
            ev['termination'] = 'recovered'
            ev['class'] = 'recovered_tier1'
            events.append(ev)
            continue

        m = _NET_RECOVER_TIER2_OK_RE.match(line)
        if m:
            # `ev` must already be open and already appended via the 'recovery
            # solve' branch above (always precedes this print in production);
            # the malformed-capture fallback is defensive only.
            name, year, day = _parse_ctx(m.group('ctx'))
            key = (name, year, day)
            ev = open_by_key.get(key)
            if ev is None:
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['note'] = ('tier-2-success print with no matching open '
                              'primary-solve event (malformed capture)')
                open_by_key[key] = ev
                events.append(ev)
            ev['termination'] = 'recovered_tier2'
            ev['class'] = 'recovered_tier2'
            continue

        m = _NET_LOG_RE.match(line)
        if m:
            name, year, day = _parse_ctx(m.group('ctx'))
            ev = open_by_key.get((name, year, day))
            if ev is not None:
                # 'tier-2' check FIRST: the label 'tier-2 recovery solve' also
                # contains the substring 'recovery', so it must not fall into
                # the tier-1 `recovery_log` branch below.
                if 'tier-2' in m.group('label'):
                    ev['tier2_log'] = m.group('path').strip()
                elif 'recovery' in m.group('label'):
                    ev['recovery_log'] = m.group('path').strip()
                else:
                    ev['primary_log'] = m.group('path').strip()
            continue

        # Cross-check only -- never used for classification (see docstring).
        m = _TSO_FINAL_RE.match(line)
        if m:
            ev = open_by_key.get((m.group('name'), m.group('year'), m.group('day')))
            if ev is not None:
                ev['final_summary_crosscheck'] = m.group('summary')
            continue
        m = _DSO_FINAL_RE.match(line)
        if m:
            ev = open_by_key.get((m.group('name'), m.group('year'), m.group('day')))
            if ev is not None:
                ev['final_summary_crosscheck'] = m.group('summary')
                if ev.get('node_id') is None:
                    ev['node_id'] = int(m.group('node'))
                if ev.get('agent') is None:
                    ev['agent'] = 'DSO'
            continue

    # Any event that never received a resolving line (class still None) is a
    # malformed-capture signal -- surfaced loudly, never silently dropped.
    for ev in open_by_key.values():
        if ev.get('class') is None:
            ev['class'] = 'indeterminate'
            note = 'unresolved at end of stdout (no recovery outcome line found)'
            ev['note'] = f"{ev['note']}; {note}" if ev.get('note') else note
            events.append(ev)

    return events


def _scan_frozen_snapshots(results_dir, run_started):
    frozen_dir = os.path.join(results_dir, 'FrozenSMOPF')
    out = []
    if not os.path.isdir(frozen_dir):
        return out
    for fname in sorted(os.listdir(frozen_dir)):
        if not fname.endswith('.pkl'):
            continue
        fpath = os.path.join(frozen_dir, fname)
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            continue
        if mtime < run_started:
            continue  # pre-existing residue from an earlier run, not this one
        entry = {'record_type': 'frozen_snapshot', 'path': fpath, 'filename': fname,
                 'mtime_utc': datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()}
        try:
            with open(fpath, 'rb') as handle:
                payload = pickle.load(handle)
            entry['metadata'] = payload.get('metadata')
        except Exception as error:
            entry['metadata_error'] = f'{type(error).__name__}: {error}'
        out.append(entry)
    return out


def _scan_and_write_network_failures(hook_state, planning_problem):
    name_to_agent = _build_name_to_agent(planning_problem)
    blocks = _scan_network_failures(hook_state['stdout_path'], name_to_agent)
    frozen = _scan_frozen_snapshots(planning_problem.results_dir, hook_state['started'])
    esso_events = hook_state.get('esso_recovery_events', [])
    # P5.15 Addendum 8 parser fix: the failure file holds ONLY classified network events.
    # Earlier runs wrote the frozen-snapshot inventory (and would have written ESSO
    # recovery events) into the same JSONL; those records carry no `class`, which is what
    # surfaced as "empty rows" in G2, G3-full, ablation B and the G2 re-run. No reported
    # count was affected (summaries use the returned lists). They now go to sibling files.
    failures_path = hook_state['network_failures_path']
    directory, basename = os.path.split(failures_path)
    snapshots_path = os.path.join(directory, basename.replace('network_failures_', 'frozen_snapshots_', 1))
    esso_events_path = os.path.join(directory, basename.replace('network_failures_', 'esso_recovery_events_', 1))
    if snapshots_path == failures_path or esso_events_path == failures_path:
        raise RuntimeError(f'cannot derive sibling paths from {failures_path}')

    def _atomic_jsonl(path, records, record_type=None):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, 'w') as handle:
            for record in records:
                row = dict(record)
                if record_type is not None:
                    row['record_type'] = record_type
                handle.write(json.dumps(row, default=str) + '\n')
        os.replace(tmp, path)

    _atomic_jsonl(failures_path, blocks)
    _atomic_jsonl(snapshots_path, frozen)
    _atomic_jsonl(esso_events_path, esso_events, record_type='esso_recovery')
    hook_state['frozen_snapshots_path'] = snapshots_path
    hook_state['esso_recovery_events_path'] = esso_events_path
    hook_state['network_failures_so_far'] = len(blocks)
    return blocks, frozen, esso_events


# ======================================================================================
#  Addendum 6 items 3/4 -- wrapper hooks (heartbeat + ESSO capture), harness-only
# ======================================================================================

@contextmanager
def esso_capture_hooks(planning, hook_state):
    """Wraps `shared_resources_planning.create_shared_energy_storage_model` (the ADMM
    -initialization ESSO solve, `cycle=None` -> stamped 'init') and
    `update_shared_energy_storages_coordination_model_and_solve` (one call per ADMM
    cycle, `cycle=iter`). NOT a production-code change: `shared_resources_planning.py`
    on disk is untouched; this reassigns the module's two names for the lifetime of one
    process-local `with` block, the same convention `p58_rescale.patched_admm_objectives`
    already uses."""
    original_create = srp.create_shared_energy_storage_model
    original_update = srp.update_shared_energy_storages_coordination_model_and_solve
    sed = planning.shared_ess_data

    def _after_round(models, cycle_label):
        n_nodes = len(models)
        tail = list(sed.esso_complementarity_diagnostics)[-n_nodes:] if n_nodes else []
        by_node = {}
        for entry in tail:
            by_node[entry['node_id']] = entry
        leak_records = _capture_esso_solve(
            sed, models, by_node, hook_state['esso_capture_dir'], cycle_label, hook_state)
        with open(hook_state['leak_path'], 'a') as handle:
            for rec in leak_records:
                handle.write(json.dumps(rec, default=str) + '\n')
        hook_state['esso_solves_so_far'] = len(sed.esso_complementarity_diagnostics)

        total_recovery = list(sed.solver_recovery_diagnostics)
        new_events = total_recovery[hook_state['_esso_recovery_seen_count']:]
        for event in new_events:
            tagged = dict(event)
            tagged['cycle'] = cycle_label
            tagged['family'] = 'esso'
            hook_state.setdefault('esso_recovery_events', []).append(tagged)
        hook_state['_esso_recovery_seen_count'] = len(total_recovery)

    def patched_create(shared_ess_data, consensus_vars, candidate_solution):
        esso_model, results = original_create(shared_ess_data, consensus_vars, candidate_solution)
        _after_round(esso_model, 'init')
        return esso_model, results

    def patched_update(planning_problem, models, ess_req, dual_ess, params,
                        from_warm_start=False, cycle=None):
        res = original_update(planning_problem, models, ess_req, dual_ess, params,
                              from_warm_start=from_warm_start, cycle=cycle)
        cycle_label = f'{cycle:03d}' if isinstance(cycle, int) else str(cycle)
        _after_round(models, cycle_label)
        _atomic_write_json(hook_state['heartbeat_path'], {
            'cycle': cycle,
            'utc_timestamp': datetime.now(timezone.utc).isoformat(),
            'wall_s': time.time() - hook_state['started'],
            'esso_solves_so_far': hook_state['esso_solves_so_far'],
            'network_failures_so_far': hook_state.get('network_failures_so_far', 0),
        })
        _scan_and_write_network_failures(hook_state, planning_problem)
        return res

    srp.create_shared_energy_storage_model = patched_create
    srp.update_shared_energy_storages_coordination_model_and_solve = patched_update
    hook_state['installed'] = True
    try:
        yield
    finally:
        srp.create_shared_energy_storage_model = original_create
        srp.update_shared_energy_storages_coordination_model_and_solve = original_update


def assert_g_capture_paths(sed, planning, hook_state):
    """RULE ELEVEN, extended per Addendum 6 item 7: production `logs_dir` set and
    absolute on planning/ESSO/TSO; the wrapper hooks installed; the output root fresh;
    the ESSO IPOPT Suffixes present on a freshly-built (UNSOLVED) probe subproblem
    (`SED._build_subproblem`, a real production function, called standalone -- no
    `.solve()`, so the solve-profile guard is untouched)."""
    missing = []
    if not hasattr(sed, 'esso_complementarity_diagnostics'):
        missing.append('shared_ess_data.esso_complementarity_diagnostics missing')
    if not hasattr(SED, '_get_esso_complementarity_diagnostics'):
        missing.append('shared_energy_storage_data._get_esso_complementarity_diagnostics missing')
    if not hasattr(SED, 'EPS_ESSO_THROUGHPUT'):
        missing.append('shared_energy_storage_data.EPS_ESSO_THROUGHPUT missing')
    if not (planning.logs_dir and os.path.isabs(planning.logs_dir)):
        missing.append(f'planning.logs_dir not absolute/set: {planning.logs_dir!r}')
    if not (sed.logs_dir and os.path.isabs(sed.logs_dir)):
        missing.append(f'shared_ess_data.logs_dir not absolute/set: {sed.logs_dir!r}')
    tso_logs_dir = getattr(planning.transmission_network, 'logs_dir', None)
    if not (tso_logs_dir and os.path.isabs(tso_logs_dir)):
        missing.append(f'transmission_network.logs_dir not absolute/set: {tso_logs_dir!r}')

    # G2PREP Fix 1: every holder's results_dir must be absolute AND under this arm's
    # own out_dir (never the shared data/SRP1/Results/FrozenSMOPF tree) -- checked
    # before any solve is attempted, same as the logs_dir checks above.
    arm_out_dir = hook_state.get('out_dir')

    def _check_results_dir(owner_label, value):
        if not (value and os.path.isabs(value)):
            missing.append(f'{owner_label}.results_dir not absolute/set: {value!r}')
            return
        if not (arm_out_dir and os.path.commonpath([value, arm_out_dir]) == os.path.normpath(arm_out_dir)):
            missing.append(f'{owner_label}.results_dir not under this arm\'s out_dir '
                           f'({arm_out_dir!r}): {value!r}')

    _check_results_dir('planning', getattr(planning, 'results_dir', None))
    _check_results_dir('shared_ess_data', getattr(sed, 'results_dir', None))
    _check_results_dir('transmission_network', getattr(planning.transmission_network, 'results_dir', None))
    for node_id, dso in planning.distribution_networks.items():
        _check_results_dir(f'distribution_networks[{node_id}]', getattr(dso, 'results_dir', None))

    if not hook_state.get('installed'):
        missing.append('wrapper hooks (create_shared_energy_storage_model / '
                       'update_shared_energy_storages_coordination_model_and_solve) not installed')
    if not hook_state.get('out_dir_was_fresh'):
        missing.append(f"output root was not verified fresh before writing: {hook_state.get('out_dir')}")
    active_nodes = list(sed.active_distribution_network_nodes)
    if not active_nodes:
        missing.append('no active_distribution_network_nodes to probe ESSO suffixes')
    else:
        try:
            probe_model = SED._build_subproblem(sed, active_nodes[0])
            for suffix_name in ('ipopt_zL_out', 'ipopt_zU_out', 'dual'):
                if not hasattr(probe_model, suffix_name):
                    missing.append(f'ESSO subproblem model missing suffix {suffix_name}')
            del probe_model
        except Exception as error:
            missing.append(f'could not probe ESSO subproblem suffixes: '
                           f'{type(error).__name__}: {error}')
    if missing:
        raise AssertionError('RULE ELEVEN (gate detector): capture paths missing -> '
                              + '; '.join(missing))
    return {'esso_complementarity_diagnostics_sink': True, 'asserted_before_run': True,
            'logs_dir_absolute': True, 'hooks_installed': True, 'output_root_fresh': True,
            'esso_suffixes_present': True, 'results_dir_absolute_and_under_out_dir': True}


def _group_diagnostics_by_round(diagnostics, n_active_nodes, cycles_run):
    """Group the flat per-solve diagnostics list into [init, cycle 1, cycle 2, ...].

    Verifies the grouping rather than assuming it: reports whether
    len(diagnostics) == n_active_nodes * (cycles_run + 1) exactly (clean case, every
    ESSO solve on every node succeeded on every round) and falls back to reporting the
    flat list plus a per-node summary if it does not (e.g. a local ESSO failure skipped
    an append for one node-round).
    """
    expected_total = n_active_nodes * (cycles_run + 1)
    clean = (n_active_nodes > 0 and len(diagnostics) == expected_total)
    rounds = []
    if clean:
        for r in range(cycles_run + 1):
            block = diagnostics[r * n_active_nodes:(r + 1) * n_active_nodes]
            rounds.append({
                'round': 'init' if r == 0 else r,
                'entries': block,
                'ratio_max': max((e['complementarity_ratio_max'] for e in block
                                   if e['complementarity_ratio_max'] is not None), default=None),
                'bound_max': max((e['spurious_throughput_bound'] for e in block
                                   if e['spurious_throughput_bound'] is not None), default=None),
                'measured_max': max((e['spurious_throughput_measured'] for e in block
                                      if e['spurious_throughput_measured'] is not None), default=None),
            })
    per_node = {}
    for e in diagnostics:
        node = str(e['node_id'])
        acc = per_node.setdefault(node, {'ratio_max': None, 'bound_max': None,
                                          'measured_max': None, 'n_solves': 0})
        acc['n_solves'] += 1
        for key, src in (('ratio_max', 'complementarity_ratio_max'),
                          ('bound_max', 'spurious_throughput_bound'),
                          ('measured_max', 'spurious_throughput_measured')):
            v = e.get(src)
            if v is not None and (acc[key] is None or v > acc[key]):
                acc[key] = v
    return {
        'n_active_nodes': n_active_nodes, 'cycles_run': cycles_run,
        'expected_total_solves': expected_total, 'observed_total_solves': len(diagnostics),
        'grouping_clean': clean,
        'per_round': rounds if clean else None,
        'per_node_summary': per_node,
        'flat_diagnostics': diagnostics,
    }


def _construct_arm_planning(label, out_dir, report, k_override=None,
                             investment_map=None, eval_id=None,
                             num_max_iters_override=None, apply_rho=True):
    """G2PREP: everything `run_admm_arm` does up to (NOT including) the
    `planning.run_operational_planning(...)` call -- eval-dir freshness check,
    `O.fresh_planning`, Fix 1's `results_dir` redirection (away from the shared
    `data/SRP1/Results/FrozenSMOPF`), ADMM/budget/rho parameters, `k_override`, and
    the investment candidate. Returns `(planning, sed, candidate)`.

    Factored out of `run_admm_arm` so the Fix 1 zero-solve verification ("construct
    the planning object for a dummy arm exactly as run_admm_arm does, up to but not
    including the operational-planning call, guard at 0") calls the SAME code
    `run_admm_arm` calls, rather than a re-implementation that could silently drift
    from it.

    `apply_rho` (P5.15 Step 3.2+3.3(a), s32 arm): when False, `N.RHO` (the
    p514_n control/perturbation rho override, v=1.5/pf=300/ess=1) is NOT
    applied -- the case-file rho (`data/SRP1/SRP1_params.json` `admm.rho`,
    1.0 on every network and channel) stays in force, per the frozen s32
    spec ("the harness MUST NOT apply p514_n_instrumented_cstar.RHO").
    Defaults to True so every other arm (g1, g2, s31, s31c, ...) is
    unaffected.
    """
    eval_name = eval_id if eval_id is not None else f'p515g_{label}'
    if eval_id is not None and os.path.exists(os.path.join(O.WORK_DIR, eval_id)):
        raise RuntimeError(
            f'refusing to start: eval dir already exists (network logs append): '
            f'{os.path.join(O.WORK_DIR, eval_id)}')
    planning = O.fresh_planning(eval_name)

    # G2PREP Fix 1: redirect results_dir to this arm's OWN root, before anything else
    # touches the planning object (in particular, before any solve or failure/
    # comparator callback could resolve `<holder>.results_dir` to the shared tree).
    results_dir = os.path.join(out_dir, 'results')
    _set_results_dir_for_arm(planning, results_dir)
    report['results_dir_redirect'] = {
        'target_results_dir': results_dir,
        'planning': planning.results_dir,
        'transmission_network': planning.transmission_network.results_dir,
        'distribution_networks': {str(nid): dso.results_dir
                                  for nid, dso in planning.distribution_networks.items()},
        'shared_ess_data': planning.shared_ess_data.results_dir,
        'network_year_day_sample': {
            holder_name: {
                str(year): {str(day): net.results_dir for day, net in days.items()}
                for year, days in holder.network.items()
            }
            for holder_name, holder in (
                [('transmission_network', planning.transmission_network)] +
                [(f'distribution_networks[{nid}]', dso)
                 for nid, dso in planning.distribution_networks.items()])
        },
    }

    planning.params.admm.num_max_iters = (
        num_max_iters_override if num_max_iters_override is not None else N.CAP)
    planning.params.admm.tol['objective']['rel'] = N.REL
    planning.shared_ess_data.params.budget = N.BUDGET
    report['apply_rho'] = apply_rho
    if apply_rho:
        RH.apply_rho_to_params(planning, N.RHO)
    RH.set_adaptive_penalty(planning, True)
    sed = planning.shared_ess_data

    if k_override is not None:
        for year in sed.years:
            for ess in sed.shared_energy_storages[year]:
                ess.cl_eff = k_override
    report['k_in_force'] = {str(y): getattr(sed.shared_energy_storages[y][0], 'cl_eff', None)
                            for y in sed.years}

    candidate = planning.get_initial_candidate_solution()
    if investment_map is None:
        for node_id in sed.active_distribution_network_nodes:
            candidate['investment'][node_id][N.INVEST_YEAR]['s'] = N.S_INV
            candidate['investment'][node_id][N.INVEST_YEAR]['e'] = N.E_INV
        report['instance'] = {'s_mva': N.S_INV, 'e_mwh': N.E_INV, 'year': N.INVEST_YEAR,
                               'assignment': 'uniform across active nodes (control/perturbation)'}
    else:
        for node_id, (s_val, e_val) in investment_map.items():
            candidate['investment'][node_id][N.INVEST_YEAR]['s'] = s_val
            candidate['investment'][node_id][N.INVEST_YEAR]['e'] = e_val
        report['instance'] = {'year': N.INVEST_YEAR, 'assignment': 'per-node',
                               'investment_map': {str(k): v for k, v in investment_map.items()}}
    srp._rebuild_candidate_total_capacities(planning, candidate)
    return planning, sed, candidate


def run_admm_arm(label, out_dir, k_override=None, investment_map=None,
                  num_max_iters_override=None, eval_id=None, post_run_hook=None,
                  apply_rho=True, full_diagnostics_in_rows=False):
    """One full cold ADMM arm through the production path, reusing p514_n's own
    module-level constants and capture helpers verbatim. `investment_map`, if given,
    overrides the uniform S_INV/E_INV assignment for specific node_ids (others left at
    N.S_INV/N.E_INV for the control/perturbation arms, or at the harness's default 0/0
    if this is a fresh candidate); pass a full dict {node_id: (s, e)} covering every
    active node to avoid ambiguity (this is what G3's full eval does).

    `num_max_iters_override` is a SMOKE-TEST-ONLY parameter (Addendum 6 smoke test):
    the `g1` CLI arm never passes it, so `N.CAP` (90) remains the C* control cap for the
    real campaign.

    `post_run_hook`, if given, is called as
    `post_run_hook(planning=planning, sed=sed, models=models, rows=rows, report=report,
    out_dir=out_dir, label=label)` AFTER the ADMM run and its own captures (esso_capture,
    diagnostics, pickle) are complete, and BEFORE this function returns -- i.e. with the
    SAME final `models` dict (`models['tso']`, `models['dso']`, `models['esso']`) this
    function used to build its own report, zero extra solves. S31 (Part 3, S31 worker
    task) uses this to write `component_levels_terminal.json` without re-implementing
    any part of `run_admm_arm`.

    `apply_rho`: forwarded to `_construct_arm_planning` (see its docstring).
    Defaults to True so every arm other than s32 is unaffected.

    `full_diagnostics_in_rows` (P5.15 Step 3.2+3.3(a), s32 arm): when True,
    each trajectory row is the RAW `admm_diagnostics` entry (every Boyd field
    this stage added, plus every pre-existing field, e.g. `rho_v_before`,
    `rho_v_action`) with `A.cycle_row`'s derived fields (ratios, slack,
    `state_step_norm`, `nonfinite`) layered on top -- i.e. nothing production
    records is dropped. Defaults to False, so every other arm's trajectory
    (built from `A.cycle_row` alone) is unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Addendum 6: these NEW artifacts are labeled per arm (see module docstring) so that
    # g2/g4b/g3_full, which share `OUT` across separate process invocations, cannot
    # collide with one another or with g1.
    heartbeat_path = os.path.join(out_dir, f'heartbeat_{label}.json')
    stdout_path = os.path.join(out_dir, f'stdout_{label}.log')
    leak_path = os.path.join(out_dir, f'leak_classification_{label}.jsonl')
    network_failures_path = os.path.join(out_dir, f'network_failures_{label}.jsonl')
    esso_capture_dir = os.path.join(out_dir, 'esso_capture', label)
    for path in (heartbeat_path, stdout_path, leak_path, network_failures_path):
        _refuse_overwrite(path)
    if os.path.exists(esso_capture_dir):
        raise RuntimeError(f'refusing to reuse a non-fresh ESSO capture dir: {esso_capture_dir}')

    checklist = N.assert_capture_paths_exist()
    started = time.time()
    report = {'stage': 'P5.15 G1-G4', 'arm': label,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'rule_eleven_checklist': checklist}
    hook_state = {
        'started': started, 'out_dir': out_dir, 'out_dir_was_fresh': True,
        'installed': False, 'var_maps': {}, 'zL_checked': False,
        'esso_solves_so_far': 0, 'network_failures_so_far': 0,
        '_esso_recovery_seen_count': 0, 'esso_recovery_events': [],
        'heartbeat_path': heartbeat_path, 'stdout_path': stdout_path,
        'leak_path': leak_path, 'network_failures_path': network_failures_path,
        'esso_capture_dir': esso_capture_dir,
    }
    # G2PREP Fix 1: hash the SHARED, preserved FrozenSMOPF tree before this arm does
    # anything, so a post-run comparison can prove (not assert) it was not touched --
    # bracketing the WHOLE arm, not just the solve window.
    pre_frozen_hashes = _hash_dir_pkls(SHARED_FROZEN_SMOPF_DIR)

    guard = SolveProfileGuard(N.PERMITTED, label=f'P5.15-G {label}').install()
    try:
        with tee_stdout(stdout_path):
            planning, sed, candidate = _construct_arm_planning(
                label, out_dir, report, k_override=k_override,
                investment_map=investment_map, eval_id=eval_id,
                num_max_iters_override=num_max_iters_override,
                apply_rho=apply_rho)

            n_active_nodes = len(sed.active_distribution_network_nodes)
            report['active_distribution_network_nodes'] = list(sed.active_distribution_network_nodes)

            with R.patched_admm_objectives(), esso_capture_hooks(planning, hook_state):
                detector_checklist = assert_g_capture_paths(sed, planning, hook_state)
                report['rule_eleven_checklist']['detector'] = detector_checklist
                _c, _results, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
    finally:
        guard.uninstall()
        report['wall_clock_s'] = time.time() - started

    # G2PREP Fix 1: post-run integrity check on the shared FrozenSMOPF tree.
    post_frozen_hashes = _hash_dir_pkls(SHARED_FROZEN_SMOPF_DIR)
    frozen_modified = [
        {'file': fname, 'pre_sha256': digest, 'post_sha256': post_frozen_hashes.get(fname)}
        for fname, digest in pre_frozen_hashes.items()
        if post_frozen_hashes.get(fname) != digest
    ]
    frozen_new_files = sorted(set(post_frozen_hashes) - set(pre_frozen_hashes))
    if frozen_modified or frozen_new_files:
        print(f'[ERROR] shared FrozenSMOPF directory modified during arm {label}: '
              f'modified={frozen_modified} new={frozen_new_files}')
    report['shared_frozen_smopf_modified'] = frozen_modified
    report['shared_frozen_smopf_new_files'] = frozen_new_files
    report['shared_frozen_smopf_dir'] = os.path.relpath(SHARED_FROZEN_SMOPF_DIR, REPO)

    # Addendum 6 item 6: one more network-failure scan "at the end", covering anything
    # appended after the last ESSO-hook call (the ESSO solve is the last of each cycle,
    # so this normally only catches the run's very last window).
    final_blocks, final_frozen, final_esso_events = _scan_and_write_network_failures(
        hook_state, planning)
    # P5.15 Addendum 7 Part 1 item 3: tier-split classes, plus 'recovered' kept as
    # tier1+tier2 for backward comparison against pre-tier-2 reports (G1's
    # committed stdout, scanned by this same parser, has zero tier-2 lines, so
    # 'recovered_tier2' is 0 there and 'recovered' reproduces the old count).
    _class_counts = {c: sum(1 for b in final_blocks if b['class'] == c)
                      for c in ('recovered_tier1', 'recovered_tier2', 'unrecovered',
                                'not_attempted', 'indeterminate')}
    _class_counts['recovered'] = _class_counts['recovered_tier1'] + _class_counts['recovered_tier2']
    report['network_failures_summary'] = {
        'n_blocks': len(final_blocks),
        'classes': _class_counts,
        'n_frozen_snapshots': len(final_frozen),
        'n_esso_recovery_events': len(final_esso_events),
        'path': os.path.relpath(network_failures_path, REPO),
    }
    report['heartbeat_path'] = os.path.relpath(heartbeat_path, REPO)
    report['stdout_path'] = os.path.relpath(stdout_path, REPO)
    report['leak_classification_path'] = os.path.relpath(leak_path, REPO)
    report['esso_capture_dir'] = os.path.relpath(esso_capture_dir, REPO)

    rows = []
    prev_recourse = None
    for e in (state.get('admm_diagnostics') or []):
        row = A.cycle_row(e, prev_recourse)
        if full_diagnostics_in_rows:
            # s32: every raw admm_diagnostics field (every Boyd field this
            # stage added -- boyd_v_*, boyd_pf_*, boyd_ess_*, rho_*_before/
            # after/action, objective_change_ratio, gap_proxy_* -- plus every
            # pre-existing field), with A.cycle_row's derived fields (ratios,
            # slack, state_step_norm, nonfinite) as an overlay so nothing is
            # silently shadowed.
            merged = dict(e)
            merged.update(row)
            row = merged
        rows.append(row)
        prev_recourse = row.get('recourse')

    last = rows[-1] if rows else {}
    report['cycle_trajectory'] = rows  # full per-cycle table, not just the terminal row
    report.update({
        'cycles_run': len(rows), 'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'converged_at_cycle': next((r['cycle'] for r in rows if r['cycle_convergence']), None),
        'terminal_objective_change_abs': last.get('objective_change_abs'),
        'terminal_objective_tolerance': last.get('objective_tolerance'),
        'rule_ten_terminal_step_over_threshold': (
            last.get('objective_change_abs') / last.get('objective_tolerance')
            if last.get('objective_change_abs') and last.get('objective_tolerance') else None),
        'local_solve_failures': sum(1 for r in rows if r.get('local_solves_ok') is False),
    })
    report['esso_capture'] = N.capture_esso(models['esso'], sed)

    # ---- Addendum-4 standing requirement: per-cycle ESSO complementarity detector ----
    diagnostics = list(sed.esso_complementarity_diagnostics)
    report['esso_complementarity_diagnostics_by_round'] = _group_diagnostics_by_round(
        diagnostics, n_active_nodes, len(rows))

    pickle_path = os.path.join(out_dir, f'esso_models_{label}.pkl')
    _refuse_overwrite(pickle_path)
    try:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(models['esso'], handle)
        report['esso_models_pickle'] = {'path': os.path.relpath(pickle_path, REPO),
                                        'bytes': os.path.getsize(pickle_path)}
    except Exception as error:
        report['esso_models_pickle'] = {'error': f'{type(error).__name__}: {error}'}

    report['solve_profile'] = {'observed': dict(guard.counts),
                               'identity_holds': guard.counts['permitted_solve'] == 51 * len(rows) + 51}

    if post_run_hook is not None:
        # S31 worker task, Part 3: zero extra solves -- `models` is the SAME dict
        # `run_operational_planning` returned above; the guard has already been
        # uninstalled (report timing only), but nothing below calls a solver.
        post_run_hook(planning=planning, sed=sed, models=models, rows=rows,
                      report=report, out_dir=out_dir, label=label)

    path = os.path.join(out_dir, f'g_{label}.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    efc_all = [v['efc_per_day_max'] for v in report['esso_capture'].values() if v['efc_per_day_max']]
    print(f"[P5.15-G {label}] recourse={report['recourse']} "
          f"cycles={report['cycles_run']} solves={guard.counts['permitted_solve']} "
          f"local_failures={report['local_solve_failures']} "
          f"wall={report['wall_clock_s']:.0f}s")
    print(f"   EFC/day max across nodes: {max(efc_all) if efc_all else None} "
          f"(threshold {N.EFC_BINDING_THRESHOLD})")
    print(f"   detector grouping clean: {report['esso_complementarity_diagnostics_by_round']['grouping_clean']} "
          f"(observed {len(diagnostics)} / expected {n_active_nodes * (len(rows) + 1)})")
    print(f"   network failures: {report['network_failures_summary']}")
    return report, path


def run_ladder_init(s_mva, out_dir):
    """Reuse p514_l_capacity_ladder.main() UNMODIFIED, redirecting its module-level OUT
    so the already-committed P514L artifacts (ladder_s1.json etc.) are never touched."""
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f'ladder_s{float(s_mva):g}.json')
    _refuse_overwrite(target)
    original_out = L.OUT
    L.OUT = out_dir
    try:
        L.main(s_mva)
    finally:
        L.OUT = original_out
    return target


def _acquire_exclusive_run_lock():
    """P5.15 Planner guard (harness-only, NOT production).

    This harness MUST NOT run concurrently with another copy of itself. Two concurrent
    campaigns writing into the same `logs_dir` risk interleaving their solver output
    into the same per-solve log file, and the barrier-term parser (this file's
    `_parse_both_barrier_columns`, and production's own `_parse_ipopt_barrier_terms`)
    would then silently attribute the wrong `mu_final`/`s_obj` to a cycle.

    This has now happened three times: once destroying a G1 run, and twice when
    four gates (g1, g2, g4b, g3_full) were launched simultaneously. The results
    would have been contaminated, not merely slow. Fail loudly instead.
    """
    import atexit
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = open(lock_path).read().strip()
        except OSError:
            holder = 'unknown'
        raise SystemExit(
            f'REFUSING TO RUN: another copy of {os.path.basename(__file__)} holds '
            f'{lock_path} (pid/arg: {holder}).\n'
            'Concurrent runs corrupt the shared ESSO IPOPT logs and produce a '
            'FABRICATED per-cycle detector trajectory. Run the gates SEQUENTIALLY. '
            'If no such process exists, remove the lock file by hand.')
    os.write(fd, f'{os.getpid()} {" ".join(sys.argv[1:])}'.encode())
    os.close(fd)
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))


def _esso_slack_values(sed, model, y, d, p):
    """P5.15 Addendum 7 item 1 (G2 re-run): ESSO aggregate-row slack values per period.

    `es_pnet[y,d,p] == sum_cohort(pch - pdch) + slack_es_pnet_up - slack_es_pnet_down` when the
    ESSO slacks are enabled. Captured so the slack-dominated initialization (SoH floor binding)
    is measured rather than inferred from duals. Fails fast (rule eleven) if the ESSO declares
    slacks but the variables are missing; returns (None, None) only when slacks are disabled."""
    slacks_enabled = bool(getattr(sed.params, 'slacks', False))
    has_vars = hasattr(model, 'slack_es_pnet_up') and hasattr(model, 'slack_es_pnet_down')
    if slacks_enabled and not has_vars:
        raise RuntimeError('STOP (rule eleven): ESSO slacks enabled but slack_es_pnet_up/down '
                           'missing on the model; cannot capture slack values.')
    if not has_vars:
        return None, None
    return (pe.value(model.slack_es_pnet_up[y, d, p], exception=False),
            pe.value(model.slack_es_pnet_down[y, d, p], exception=False))


# ---------------------------------------------------------------------------------------
# P5.15 Addendum 8 — ablation C (hypothesis H-epsilon), frozen spec
# data/SRP1/Results/P515A/frozen_ablation_c_spec_v1_410f8262.json: G1 configuration with
# EPS_ESSO_THROUGHPUT = 1e-5 instead of 1e-3, G1-equivalent recovery, so epsilon is the only
# change. `_build_subproblem` and the leak diagnostics read the module global at call time.
# ---------------------------------------------------------------------------------------
OUT_ABL_C = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515A', 'run_c')
ABL_C_EVAL_ID = 'p515a_eps_1e-5'
ABL_C_EPS = 1e-5


def _configure_ablation_c(planning):
    SED.EPS_ESSO_THROUGHPUT = ABL_C_EPS
    RH.set_recovery_policy(planning, enabled=True, tier2_enabled=False,
                           node_overrides={5: {'enabled': False}})
    return planning


# P5.15 Addendum 7 item 1 — G2 re-run: G2's configuration (k = 10,000 on C*) under the NEW
# default recovery policy (every network and the ESSO eligible, case33_1 included; tier 2 on),
# with ESSO slack values captured per period.
OUT_G2R = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G2R')
G2R_EVAL_ID = 'p515g2r_k10000'


# ---------------------------------------------------------------------------------------
# P5.15 Addendum 7 item 2 — ablation A (frozen spec data/SRP1/Results/P515A/
# frozen_ablation_spec_v1_2271c77f.json): G1 configuration with Candidate 4 reverted, under
# G1-equivalent recovery, so exactly one thing differs from G1.
# ---------------------------------------------------------------------------------------
OUT_ABL_A = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515A', 'run_a')
ABL_A_EVAL_ID = 'p515a_candidate4_reverted'


def _configure_ablation_a(planning):
    """Revert Candidate 4 and pin G1's recovery behaviour on a fresh planning object.

    Candidate 4 pinned `SlacksFlexibility.day_balance = True` in code; the pre-Step-1b
    behaviour (and the committed case files) is False, which restores the two-sided band
    `pe.inequality(-SMALL_TOLERANCE, p_up - p_down, SMALL_TOLERANCE)` in
    `flex_energy_balance_p_rule` and removes the `slack_flex_*_balance_*` variables and
    their penalty. Recovery: G1 ran with case33_1 (node 5) ineligible and no tier 2."""
    planning.transmission_network.params.slacks.flexibility.day_balance = False
    for dso in planning.distribution_networks.values():
        dso.params.slacks.flexibility.day_balance = False
    RH.set_recovery_policy(planning, enabled=True, tier2_enabled=False,
                           node_overrides={5: {'enabled': False}})
    return planning


# ---------------------------------------------------------------------------------------
# P5.15 Addendum 7 item 2 — ablation B (authorized because A failed the frozen criterion,
# data/SRP1/Results/P515A/ablation_a_evaluation.json): G1 configuration with Candidate 1
# re-wired, Candidate 4 as in G1, G1-equivalent recovery.
# ---------------------------------------------------------------------------------------
OUT_ABL_B = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515A', 'run_b')
ABL_B_EVAL_ID = 'p515a_candidate1_rewired'
_CANDIDATE1_ROWS = ('sess_phi_limit_lower', 'sess_phi_limit_upper')


def _rewire_candidate1(network_cls):
    """Restore the pre-Step-1b shared-ESS power-factor rows without editing production.

    Pre-1b `network.py` declared, right after `sess_active_sum_limit`,
    `sess_phi_limit_lower/_upper = pe.Constraint(shared_energy_storages, scenarios_market,
    scenarios_operation, periods, rule=partial(sess_phi_limits_lower/_upper, network=network))`,
    and `_SHARED_ESS_OPERATIONAL_CONSTRAINTS` listed both after `sess_active_sum_limit`. The
    callables were retained (unwired) for fixture unpickling. The activation loop in
    `model_construction_helpers` reads that tuple by module-global name at call time, so
    rebinding the module attribute takes effect."""
    from functools import partial
    import model_construction_helpers as MCH
    if not getattr(network_cls, '_p515_candidate1_rewired', False):
        original_build = network_cls.build_model

        def build_model(self, params):
            model = original_build(self, params)
            if hasattr(model, 'sess_converter_capability') and not hasattr(model, 'sess_phi_limit_lower'):
                index = (model.shared_energy_storages, model.scenarios_market,
                         model.scenarios_operation, model.periods)
                model.sess_phi_limit_lower = pe.Constraint(
                    *index, rule=partial(MCH.sess_phi_limits_lower, network=self))
                model.sess_phi_limit_upper = pe.Constraint(
                    *index, rule=partial(MCH.sess_phi_limits_upper, network=self))
            return model

        network_cls.build_model = build_model
        network_cls._p515_candidate1_rewired = True
    names = [n for n in MCH._SHARED_ESS_OPERATIONAL_CONSTRAINTS if n not in _CANDIDATE1_ROWS]
    position = names.index('sess_active_sum_limit') + 1
    MCH._SHARED_ESS_OPERATIONAL_CONSTRAINTS = tuple(
        names[:position] + list(_CANDIDATE1_ROWS) + names[position:])


def _configure_ablation_b(planning):
    tn = planning.transmission_network
    sample_year = next(iter(tn.years)); sample_day = next(iter(tn.days))
    _rewire_candidate1(type(tn.network[sample_year][sample_day]))
    RH.set_recovery_policy(planning, enabled=True, tier2_enabled=False,
                           node_overrides={5: {'enabled': False}})
    return planning


# ===========================================================================
# S31 worker task (2026-09-15) -- Part 3: per-block component levels at the
# terminal cycle, from the returned final models, zero extra solves.
# Authority: PLANNER_BRIEF_2026-09-13.md Addendum 10 and
# P5_15_S31_PENALTY_TABLE_DRAFT.md section 5.
# ===========================================================================

def assert_s31_capture_paths(planning):
    """Rule eleven: verify, BEFORE the run, that a capture path exists for every
    quantity the S31 Part 3 schema requires. Zero solves -- attribute/callable
    checks on the freshly-built (unsolved) TSO model of this SAME planning
    object only (it is discarded after the check; `run_admm_arm` below builds
    its own models through the normal production path)."""
    transmission_network = planning.transmission_network
    year0 = next(iter(transmission_network.years))
    day0 = next(iter(transmission_network.days))
    probe_model = transmission_network.network[year0][day0].build_model(transmission_network.params)
    network = transmission_network.network[year0][day0]
    params = transmission_network.params

    checklist = {
        'tso_total_gen_cost': hasattr(probe_model, 'total_gen_cost'),
        'tso_total_flex_cost': hasattr(probe_model, 'total_flex_cost'),
        'tso_total_load_curt_cost': hasattr(probe_model, 'total_load_curt_cost'),
        'tso_total_gen_curt_penalty': hasattr(probe_model, 'total_gen_curt_penalty'),
        'tso_total_ess_utilization_cost_penalty': hasattr(probe_model, 'total_ess_utilization_cost_penalty'),
        'tso_slack_flex_q_balance_up_present': hasattr(probe_model, 'slack_flex_q_balance_up'),
        'mch_adn_interface_flexibility_cost': callable(getattr(mch, 'adn_interface_flexibility_cost', None)),
        'mch_gen_curtailment_definitional_value': callable(getattr(mch, 'gen_curtailment_definitional_value', None)),
        'mch_ess_complementarity_bilinear_value': callable(getattr(mch, 'ess_complementarity_bilinear_value', None)),
        'mch_load_is_tso_adn_interface': callable(getattr(mch, 'load_is_tso_adn_interface', None)),
        'srp_get_local_detector_components': callable(getattr(srp, '_get_local_detector_components', None)),
        'srp_get_operational_recourse_components': callable(getattr(srp, '_get_operational_recourse_components', None)),
        'srp_get_admm_block_weight': callable(getattr(srp, '_get_admm_block_weight', None)),
        'esso_get_feasibility_violation': callable(getattr(planning.shared_ess_data, 'get_feasibility_violation', None)),
    }
    # Exercise every reporting call site once, zero-solve, on the probe model's
    # initial point, so a signature mismatch fails HERE, not after the campaign.
    checklist['tso_detector_components_evaluate'] = isinstance(
        srp._get_local_detector_components(probe_model, network, params), dict)
    checklist['tso_adn_interface_flex_cost_evaluate'] = isinstance(
        float(pe.value(mch.adn_interface_flexibility_cost(
            probe_model, network, next(iter(probe_model.scenarios_market)),
            next(iter(probe_model.scenarios_operation)), params))), float)

    missing = [name for name, ok in checklist.items() if not ok]
    if missing:
        raise RuntimeError(f'S31 Part 3 capture-path pre-flight FAILED, missing/broken: {missing}')
    return checklist


def _s31_scenario_weighted(model, network, params, func):
    total = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            probability = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
            total += probability * float(pe.value(func(model, network, s_m, s_o, params)))
    return total


def _s31_block_components(model, network, params):
    """Unweighted (per-block, scenario-probability-weighted only -- the SAME
    convention `model.total_*` Expressions and `_get_local_detector_components`
    use) level of every S31 Part 3 component, at the terminal point. Zero
    solves: reads existing model Expressions/Vars only; no re-derivation of
    quantities the objective already computes."""

    out = {
        'generation_cost': float(pe.value(model.total_gen_cost)) if hasattr(model, 'total_gen_cost') else 0.0,
        'flexibility_cost_internal': float(pe.value(model.total_flex_cost)) if hasattr(model, 'total_flex_cost') else 0.0,
        'load_curtailment_cost': float(pe.value(model.total_load_curt_cost)) if hasattr(model, 'total_load_curt_cost') else 0.0,
        'res_curtailment_penalty': float(pe.value(model.total_gen_curt_penalty)) if hasattr(model, 'total_gen_curt_penalty') else 0.0,
        'ess_usage_cost': float(pe.value(model.total_ess_utilization_cost_penalty)) if hasattr(model, 'total_ess_utilization_cost_penalty') else 0.0,
    }

    # Split 2: TSO ADN-interface flexibility cost (row 3 / D1). Removed from the
    # objective; evaluable directly (flex_p_down/flex_q_down of ADN loads are
    # still free variables, only their PRICING was removed).
    out['flexibility_cost_tso_adn_interface_definitional'] = _s31_scenario_weighted(
        model, network, params, mch.adn_interface_flexibility_cost)

    # D-row detector components (rows 10, 11, 12, 13, 15, 16); split 1 keeps ESS
    # complementarity (removed) separate from local/shared day-balance (kept).
    detector = srp._get_local_detector_components(model, network, params)
    out['voltage_slack'] = detector['voltage_slack']                                    # row 12
    out['node_balance_slack'] = detector['node_balance_slack']                          # row 15
    out['branch_flow_slack'] = detector['branch_flow_slack']                            # row 16
    out['flexibility_p_day_balance_slack'] = detector['flexibility_p_day_balance_slack']  # row 13
    out['local_ess_day_balance_slack'] = detector['local_ess_day_balance_slack']        # row 10
    out['shared_ess_day_balance_slack'] = detector['shared_ess_day_balance_slack']      # row 11
    out['detector_penalty_total'] = detector['detector_penalty_total']

    # Split 1 / row 9: bilinear ESS complementarity, definitional (removed from
    # the objective; evaluable directly, pch/pdch remain free variables).
    out['ess_complementarity_bilinear_definitional'] = _s31_scenario_weighted(
        model, network, params, mch.ess_complementarity_bilinear_value)

    # Split 3 / row 14 (D2): orphan slacks. Fixed to 0.0 at model construction
    # (network.py) -- read directly, NOT re-derived, and expected to be exactly
    # 0.0. If a nonzero value appears here, the fix in network.py did not take
    # (a regression, not a measurement).
    orphan_q_raw = 0.0
    orphan_adn_p_raw = 0.0
    if hasattr(model, 'slack_flex_q_balance_up'):
        for c in model.loads:
            is_adn = mch.load_is_tso_adn_interface(network, network.loads[c])
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    orphan_q_raw += float(pe.value(
                        model.slack_flex_q_balance_up[c, s_m, s_o] + model.slack_flex_q_balance_down[c, s_m, s_o]))
                    if is_adn:
                        orphan_adn_p_raw += float(pe.value(
                            model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o]))
    out['orphan_flex_q_day_balance_slack_raw'] = orphan_q_raw
    out['orphan_tso_adn_flex_p_day_balance_slack_raw'] = orphan_adn_p_raw
    # Definitional value at the pre-signature weight PENALTY_FLEXIBILITY: since
    # the raw slacks above are fixed to 0.0, this is 0.0 BY CONSTRUCTION, not an
    # independent measurement -- the variables cannot take any other value, so
    # this quantity cannot be evaluated as potentially nonzero on this model.
    out['orphan_flex_q_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional'] = orphan_q_raw * PENALTY_FLEXIBILITY
    out['orphan_tso_adn_flex_p_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional'] = orphan_adn_p_raw * PENALTY_FLEXIBILITY

    # Definitional / row 5: DSO RES curtailment at the pre-signature weight 1
    # (PENALTY_GENERATION_CURTAILMENT). Removed from the objective (weight set
    # to 0); evaluable directly (pg_avail, pg are still free/parametric as before).
    out['res_curtailment_definitional_at_weight_1'] = _s31_scenario_weighted(
        model, network, params,
        lambda m, n, sm, so, p: mch.gen_curtailment_definitional_value(m, n, sm, so, p, PENALTY_GENERATION_CURTAILMENT))

    return out


def write_component_levels_terminal(planning, sed, models, rows, report, out_dir, label):
    """S31 Part 3. Writes `component_levels_terminal.json` under `out_dir` from
    the FINAL models `run_admm_arm` already built -- zero extra solves."""

    transmission_network = planning.transmission_network
    blocks = {}

    for year in transmission_network.years:
        for day in transmission_network.days:
            model = models['tso'][year][day]
            network = transmission_network.network[year][day]
            params = transmission_network.params
            weight = srp._get_admm_block_weight(transmission_network, year, day)
            unweighted = _s31_block_components(model, network, params)
            weighted = {name: weight * value for name, value in unweighted.items()}
            key = f'TSO|{year}|{day}'
            blocks[key] = {'kind': 'TSO', 'node_id': None, 'year': str(year), 'day': str(day),
                          'admm_block_weight': weight, 'unweighted': unweighted, 'weighted': weighted}

    for node_id, distribution_network in planning.distribution_networks.items():
        for year in distribution_network.years:
            for day in distribution_network.days:
                model = models['dso'][node_id][year][day]
                network = distribution_network.network[year][day]
                params = distribution_network.params
                weight = srp._get_admm_block_weight(distribution_network, year, day)
                unweighted = _s31_block_components(model, network, params)
                weighted = {name: weight * value for name, value in unweighted.items()}
                key = f'DSO|{node_id}|{year}|{day}'
                blocks[key] = {'kind': 'DSO', 'node_id': node_id, 'year': str(year), 'day': str(day),
                              'admm_block_weight': weight, 'unweighted': unweighted, 'weighted': weighted}

    totals_weighted = {}
    for block in blocks.values():
        for name, value in block['weighted'].items():
            totals_weighted[name] = totals_weighted.get(name, 0.0) + value

    recourse_components = srp._get_operational_recourse_components(planning, models)
    esso_feasibility_violation = planning.shared_ess_data.get_feasibility_violation(models['esso'])

    payload = {
        'stage': 'P5.15 Step 3.1 (S31) Part 3 -- per-block component levels at the terminal cycle',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 10', 'P5_15_S31_PENALTY_TABLE_DRAFT.md section 5'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'cycles_run': len(rows),
        'converged_at_cycle': report.get('converged_at_cycle'),
        'blocks': blocks,
        'totals_weighted': totals_weighted,
        'recourse_components': recourse_components,
        'esso_feasibility_violation_D3': esso_feasibility_violation,
        'weight_convention': (
            '"unweighted" = per-block, scenario-probability-weighted only (the '
            'same convention model.total_* Expressions and '
            '_get_local_detector_components use). "weighted" = unweighted * '
            '_get_admm_block_weight(network_data, year, day) (year-count * '
            'day-count * discount annualization), the SAME weight '
            'get_primal_value uses.'
        ),
        'orphan_slack_note': (
            'orphan_flex_q_day_balance_slack_raw and '
            'orphan_tso_adn_flex_p_day_balance_slack_raw are read directly from '
            'variables FIXED to 0.0 by production (row 14 / D2, network.py); '
            'they are expected to be exactly 0.0 in every block. The '
            '_definitional penalty fields derived from them are therefore 0.0 '
            'BY CONSTRUCTION, not an independent measurement -- these two '
            'removed terms cannot be evaluated as potentially nonzero on this '
            'model, because the variables they would have penalized are fixed.'
        ),
    }

    path = os.path.join(out_dir, 'component_levels_terminal.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S31 Part 3] component_levels_terminal.json written: {path}')
    return path


# ===========================================================================
# S31C worker task (2026-09-15) -- Part 3: interface energy settlement / signed
# delta reporting, level-capture arm `s31c`. Authority:
# PLANNER_BRIEF_2026-09-13.md Addendum 12.
# ===========================================================================

OUT_S31C = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C_run')


def assert_s31c_capture_paths(planning):
    """Rule eleven, extended for Addendum 12's signed interface reparametrization
    and interface energy settlement. Reuses `assert_s31_capture_paths` (still
    required -- the S31C hook also calls `write_component_levels_terminal`) and
    adds the new capture paths. Zero solves."""
    checklist = dict(assert_s31_capture_paths(planning))

    transmission_network = planning.transmission_network
    year0 = next(iter(transmission_network.years))
    day0 = next(iter(transmission_network.days))
    tso_probe_model = transmission_network.network[year0][day0].build_model(transmission_network.params)

    checklist['tso_interface_delta_p_present'] = hasattr(tso_probe_model, 'interface_delta_p')
    checklist['tso_interface_delta_q_present'] = hasattr(tso_probe_model, 'interface_delta_q')
    checklist['tso_interface_settlement_present'] = hasattr(tso_probe_model, 'interface_settlement')
    checklist['tso_interface_settlement_weight_present'] = hasattr(tso_probe_model, 'interface_settlement_weight')
    checklist['tso_expected_interface_pf_p_will_exist_after_admm_setup'] = callable(
        getattr(srp, 'create_transmission_network_model', None))

    node0 = next(iter(planning.distribution_networks))
    distribution_network0 = planning.distribution_networks[node0]
    dso_probe_model = distribution_network0.network[year0][day0].build_model(distribution_network0.params)
    checklist['dso_interface_settlement_present'] = hasattr(dso_probe_model, 'interface_settlement')
    checklist['dso_interface_settlement_weight_present'] = hasattr(dso_probe_model, 'interface_settlement_weight')

    checklist['srp_get_interface_reporting_detail'] = callable(getattr(srp, '_get_interface_reporting_detail', None))
    checklist['srp_get_operational_interface_settlement_blocks'] = callable(
        getattr(srp, '_get_operational_interface_settlement_blocks', None))
    checklist['srp_get_local_interface_settlement'] = callable(getattr(srp, '_get_local_interface_settlement', None))

    missing = [name for name, ok in checklist.items() if not ok]
    if missing:
        raise RuntimeError(f'S31C Part 3 capture-path pre-flight FAILED, missing/broken: {missing}')
    return checklist


def _s31c_interface_detail(planning, models):
    """S31C Part 3. Zero solves: reads existing model Vars/Expressions and the
    Part 1 reporting helpers (`_get_interface_reporting_detail`,
    `_get_operational_interface_settlement_blocks`,
    `_get_local_interface_settlement`) only -- no re-derivation."""
    transmission_network = planning.transmission_network

    reporting_detail = srp._get_interface_reporting_detail(planning, models)
    settlement_blocks = srp._get_operational_interface_settlement_blocks(planning, models)

    per_block = {}
    for (kind, node_id, year, day), weighted_value in settlement_blocks.items():
        if kind == 'TSO':
            model = models['tso'][year][day]
            key = f'TSO|{year}|{day}'
        else:
            model = models['dso'][node_id][year][day]
            key = f'DSO|{node_id}|{year}|{day}'
        unweighted_value = srp._get_local_interface_settlement(model)
        per_block[key] = {
            'kind': kind, 'node_id': node_id, 'year': str(year), 'day': str(day),
            'interface_settlement_unweighted': unweighted_value,
            'interface_settlement_weighted': weighted_value,
        }

    t_tso_total = sum(v for (kind, _n, _y, _d), v in settlement_blocks.items() if kind == 'TSO')
    t_dso_by_node = {}
    for (kind, node_id, _y, _d), v in settlement_blocks.items():
        if kind == 'DSO':
            t_dso_by_node[node_id] = t_dso_by_node.get(node_id, 0.0) + v
    t_tso_plus_t_dso_terminal = t_tso_total + sum(t_dso_by_node.values())

    consensus_residual_per_dso = {}
    flexibility_volumes_per_dso = {}
    for node_id, by_year in reporting_detail.items():
        residual_periods = {}
        sum_pi_baseMVA_residual_unweighted = 0.0
        sum_pi_baseMVA_residual_weighted = 0.0
        dso_settlement_sum_pi_p_int_unweighted = 0.0
        dso_settlement_sum_pi_p_int_weighted = 0.0
        abs_delta_p_sum_mw = 0.0
        abs_delta_q_sum_mvar = 0.0
        max_abs_delta_p_mw = 0.0
        max_abs_delta_q_mvar = 0.0
        s_base = None
        for year, by_day in by_year.items():
            for day, day_detail in by_day.items():
                s_base = transmission_network.network[year][day].baseMVA
                # Same ADMM block weight (year-count * day-count * discount
                # annualization) `_get_admm_block_weight` gives, and the SAME one
                # `settlement_blocks` (hence t_tso_plus_t_dso_terminal) already
                # carries -- required so the two are comparable (Addendum 12's
                # gate: "T_TSO + T_DSO ... equals the priced consensus residual").
                block_weight = srp._get_admm_block_weight(transmission_network, year, day)
                dso_settlement_sum_pi_p_int_unweighted += day_detail['dso_settlement_sum_pi_p_int']
                dso_settlement_sum_pi_p_int_weighted += block_weight * day_detail['dso_settlement_sum_pi_p_int']
                for p, period_detail in day_detail['periods'].items():
                    residual_mw = period_detail['p_int_tso_expected_mw'] - period_detail['p_int_dso_expected_mw']
                    priced_residual = period_detail['price_per_mwh'] * residual_mw
                    residual_periods[f'{year}|{day}|{p}'] = {
                        'p_int_tso_expected_mw': period_detail['p_int_tso_expected_mw'],
                        'p_int_dso_expected_mw': period_detail['p_int_dso_expected_mw'],
                        'residual_mw': residual_mw,
                        'priced_residual_pi_baseMVA_residual_unweighted': priced_residual,
                        'priced_residual_pi_baseMVA_residual_weighted': block_weight * priced_residual,
                        'admm_block_weight': block_weight,
                    }
                    sum_pi_baseMVA_residual_unweighted += priced_residual
                    sum_pi_baseMVA_residual_weighted += block_weight * priced_residual
                    for delta_p_mw in period_detail['delta_p_mw'].values():
                        abs_delta_p_sum_mw += abs(delta_p_mw)
                        max_abs_delta_p_mw = max(max_abs_delta_p_mw, abs(delta_p_mw))
                    for delta_q_mvar in period_detail['delta_q_mvar'].values():
                        abs_delta_q_sum_mvar += abs(delta_q_mvar)
                        max_abs_delta_q_mvar = max(max_abs_delta_q_mvar, abs(delta_q_mvar))

        consensus_residual_per_dso[node_id] = {
            'periods': residual_periods,
            'sum_pi_baseMVA_residual_unweighted': sum_pi_baseMVA_residual_unweighted,
            'sum_pi_baseMVA_residual_weighted': sum_pi_baseMVA_residual_weighted,
            'dso_settlement_sum_pi_p_int_unweighted': dso_settlement_sum_pi_p_int_unweighted,
            'dso_settlement_sum_pi_p_int_weighted': dso_settlement_sum_pi_p_int_weighted,
        }
        flexibility_volumes_per_dso[node_id] = {
            'sum_abs_delta_p_mw': abs_delta_p_sum_mw,
            'max_abs_delta_p_mw': max_abs_delta_p_mw,
            'sum_abs_delta_q_mvar': abs_delta_q_sum_mvar,
            'max_abs_delta_q_mvar': max_abs_delta_q_mvar,
            'sum_abs_delta_p_pu': (abs_delta_p_sum_mw / s_base) if s_base else None,
            'max_abs_delta_p_pu': (max_abs_delta_p_mw / s_base) if s_base else None,
            'sum_abs_delta_q_pu': (abs_delta_q_sum_mvar / s_base) if s_base else None,
            'max_abs_delta_q_pu': (max_abs_delta_q_mvar / s_base) if s_base else None,
        }

    return {
        'per_block_interface_settlement': per_block,
        't_tso_total': t_tso_total,
        't_dso_by_node': t_dso_by_node,
        't_tso_plus_t_dso_terminal': t_tso_plus_t_dso_terminal,
        'interface_consensus_residual_per_dso': consensus_residual_per_dso,
        'flexibility_volumes_per_dso': flexibility_volumes_per_dso,
        'interface_reporting_detail': reporting_detail,
    }


def write_interface_settlement_detail_s31c(planning, sed, models, rows, report, out_dir, label):
    """S31C Part 3. REUSES `write_component_levels_terminal` (the s31 post-run
    level writer, unmodified) and EXTENDS its output with the interface energy
    settlement / signed-delta detail Addendum 12 requires, into a companion
    artifact -- zero extra solves, same FINAL models `run_admm_arm` already
    built."""
    base_path = write_component_levels_terminal(planning, sed, models, rows, report, out_dir, label)
    detail = _s31c_interface_detail(planning, models)

    payload = {
        'stage': 'P5.15 Step 3.1-C (S31C) Part 3 -- interface settlement / signed-delta detail',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 12'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'cycles_run': len(rows),
        'converged_at_cycle': report.get('converged_at_cycle'),
        'component_levels_terminal_path': base_path,
        **detail,
        'reading_rule': (
            't_tso_plus_t_dso_terminal (block-weighted, year-count * day-count * '
            'discount annualization) should equal MINUS the priced consensus '
            'residual on the SAME weighting -- t_tso_plus_t_dso_terminal == '
            '-1 * sum(sum_pi_baseMVA_residual_weighted over '
            'consensus_residual_per_dso). Sign: T_TSO = -prob*pi*baseMVA*p_TSO, '
            'T_DSO = +prob*pi*baseMVA*p_DSO, so T_TSO+T_DSO = '
            'prob*pi*baseMVA*(p_DSO - p_TSO); the residual field here is defined '
            '(TSO - DSO) per the worker task text, i.e. the NEGATIVE of that '
            'difference -- verified exactly on the P5.15 S31C Part 2 zero-solve '
            'cancellation-identity check (same-magnitude, opposite-sign) and '
            'reproduced on the Part 4 one-cycle preflight (840,010,674.369291 vs '
            '-840,010,674.369292). The *_unweighted fields are the per-block, '
            'scenario-probability-weighted-only quantities (comparable to '
            '"unweighted" elsewhere in this schema), NOT expected to reconcile '
            'with t_tso_plus_t_dso_terminal on their own (that quantity is '
            'block-weighted).'
        ),
    }

    path = os.path.join(out_dir, 'interface_settlement_detail_s31c.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S31C Part 3] interface_settlement_detail_s31c.json written: {path}')
    return path


# ===========================================================================
# P5.15 Step 3.2 + 3.3(a) -- Boyd stopping rule and residual balancing,
# level-capture arm `s32`. Authority: PLANNER_BRIEF_2026-09-13.md Addendum 9
# sections 3.2, 3.3(a), Addendum 13. Binding specification:
# data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json (supersedes v1
# data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json: the balancing
# dual ratio now uses s_rho_part instead of full s; the stopping rule is
# unchanged).
# ===========================================================================

OUT_S32 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run')
S32_SPEC_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S32', 'frozen_s32_spec_v2_516bd749.json')
S32_SPEC_SHA256 = '516bd749c5eff6c5f26c0dac570d10964ca989b2c64394ac1ba5deca32356b85'
S32_CAP = 150
S32_REL = 1e-4
S32_S31C_G_PATH = os.path.join(OUT_S31C, 'g_baseline.json')

# Frozen spec `report_per_cycle_channel` (per channel v/pf/ess); the
# channel-scoped fields, checked against `get_admm_boyd_residual_metrics`'s
# source and the `admm_diagnostics` dict literal in `shared_resources_planning.py`.
S32_REPORT_PER_CYCLE_CHANNEL_FIELDS = (
    'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
    'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio',
    'dual_ratio_balance',
)
S32_ADMM_DIAGNOSTICS_KEYS = (
    'boyd_v_r', 'boyd_v_s', 'boyd_v_s_rho_part', 'boyd_v_s_proximal_part',
    'boyd_v_proximal_share', 'boyd_v_eps_pri', 'boyd_v_eps_dual',
    'boyd_v_norm_x', 'boyd_v_norm_z', 'boyd_v_norm_y', 'boyd_v_primal_ratio',
    'boyd_v_dual_ratio', 'boyd_v_dual_ratio_balance', 'boyd_v_primal_pass', 'boyd_v_dual_pass', 'boyd_v_channel_pass',
    'boyd_pf_r', 'boyd_pf_s', 'boyd_pf_s_rho_part', 'boyd_pf_s_proximal_part',
    'boyd_pf_eps_pri', 'boyd_pf_eps_dual', 'boyd_pf_norm_x', 'boyd_pf_norm_z',
    'boyd_pf_norm_y', 'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio', 'boyd_pf_dual_ratio_balance', 'boyd_pf_channel_pass',
    'boyd_ess_r', 'boyd_ess_s', 'boyd_ess_s_rho_part', 'boyd_ess_s_proximal_part',
    'boyd_ess_eps_pri', 'boyd_ess_eps_dual', 'boyd_ess_norm_x', 'boyd_ess_norm_z',
    'boyd_ess_norm_y', 'boyd_ess_norm_y_tso', 'boyd_ess_norm_y_dso', 'boyd_ess_norm_y_esso',
    'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_ess_dual_ratio_balance', 'boyd_ess_channel_pass',
    'boyd_all_pass', 'boyd_stop', 'boyd_eps_abs', 'boyd_eps_rel', 'boyd_eps_source',
    'rho_v_before', 'rho_v_after', 'rho_v_action',
    'rho_pf_before', 'rho_pf_after', 'rho_pf_action',
    'rho_ess_before', 'rho_ess_after', 'rho_ess_action',
    'objective_change_ratio', 'objective_change_abs', 'objective_tolerance',
    'gap_proxy_G', 'gap_proxy_G_reason', 'gap_proxy_Q', 'gap_proxy_G_over_Q',
    'recourse', 'gross_operational_cost',
)


def _s32_spec_hash():
    with open(S32_SPEC_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def assert_s32_capture_paths(planning):
    """Rule eleven for the s32 arm: verify, BEFORE the run, that a capture
    path exists for every field the frozen spec's `report_per_cycle_channel`
    and `report_terminal` require (structural: production callables/source
    exist and carry the field), that `boyd_eps_source == 'case_file'`, that
    the spec file's hash matches its own filename, and that every initial
    rho is 1.0 (the case-file value -- N.RHO must NOT be applied). Zero
    solves.

    Reuses `assert_s31c_capture_paths` (still required: the s32 hook also
    calls `write_component_levels_terminal` and
    `write_interface_settlement_detail_s31c`, which cover `report_terminal`'s
    "D rows", "cancellation residual", "per-DSO settlement and flexibility
    volumes" and "network failures by tier" fields) and adds the new Boyd
    checks.
    """
    checklist = dict(assert_s31c_capture_paths(planning))

    # -- spec file identity ---------------------------------------------
    observed_hash = _s32_spec_hash()
    checklist['spec_file_hash_matches'] = (observed_hash == S32_SPEC_SHA256)
    checklist['spec_file_hash_observed'] = observed_hash

    # -- boyd tolerance source and value ----------------------------------
    admm_params = planning.params.admm
    checklist['boyd_eps_source_is_case_file'] = (admm_params.boyd_eps_source == 'case_file')
    checklist['boyd_tol_present'] = (
        'boyd' in admm_params.tol
        and 'eps_abs' in admm_params.tol['boyd']
        and 'eps_rel' in admm_params.tol['boyd']
    )

    # -- initial rho: case-file value (1.0); N.RHO must NOT be applied ----
    rho_all_one = all(
        float(v) == 1.0
        for group in ('v', 'pf', 'ess')
        for v in admm_params.rho[group].values()
    )
    checklist['initial_rho_all_one'] = rho_all_one
    checklist['initial_rho_snapshot'] = {
        group: dict(admm_params.rho[group]) for group in ('v', 'pf', 'ess')
    }

    # -- per-cycle-channel capture: the production function and the balancing
    #    function's new signature exist, and their source literally carries
    #    every field the frozen spec's report_per_cycle_channel requires ----
    checklist['srp_get_admm_boyd_residual_metrics'] = callable(
        getattr(srp, 'get_admm_boyd_residual_metrics', None))
    checklist['srp_update_admm_penalties_accepts_boyd_metrics'] = (
        'boyd_metrics' in inspect.signature(srp._update_admm_penalties).parameters)

    boyd_fn_source = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    for field in S32_REPORT_PER_CYCLE_CHANNEL_FIELDS:
        checklist[f'boyd_field_{field}_in_source'] = (f"'{field}':" in boyd_fn_source)

    module_source = inspect.getsource(srp)
    for key in S32_ADMM_DIAGNOSTICS_KEYS:
        checklist[f'admm_diagnostics_key_{key}_present'] = (f"'{key}':" in module_source)

    # -- report_terminal fields not already covered by assert_s31c_capture_paths
    checklist['s31c_g_baseline_reference_exists'] = os.path.exists(S32_S31C_G_PATH)

    missing = [name for name, ok in checklist.items()
               if isinstance(ok, bool) and not ok]
    if missing:
        raise RuntimeError(f'S32 capture-path pre-flight FAILED, missing/broken: {missing}')
    return checklist


def _s32_rho_trajectory(rows):
    trajectory = {'v': [], 'pf': [], 'ess': []}
    for row in rows:
        for group in ('v', 'pf', 'ess'):
            trajectory[group].append({
                'cycle': row.get('cycle'),
                'rho_before': row.get(f'rho_{group}_before'),
                'rho_after': row.get(f'rho_{group}_after'),
                'action': row.get(f'rho_{group}_action'),
            })
    return trajectory


def _s32_binding_test(last_row):
    binding = {}
    for group in ('v', 'pf', 'ess'):
        binding[group] = {
            'r': last_row.get(f'boyd_{group}_r'),
            's': last_row.get(f'boyd_{group}_s'),
            'eps_pri': last_row.get(f'boyd_{group}_eps_pri'),
            'eps_dual': last_row.get(f'boyd_{group}_eps_dual'),
            'primal_ratio': last_row.get(f'boyd_{group}_primal_ratio'),
            'dual_ratio': last_row.get(f'boyd_{group}_dual_ratio'),
            'dual_ratio_balance': last_row.get(f'boyd_{group}_dual_ratio_balance'),
            'primal_pass': last_row.get(f'boyd_{group}_primal_pass'),
            'dual_pass': last_row.get(f'boyd_{group}_dual_pass'),
            'channel_pass': last_row.get(f'boyd_{group}_channel_pass'),
        }
    return binding


def _s32_system_cost_vs_s31c(rows, report):
    """System cost (recourse) vs the s31c reference trajectory, at matched
    cycles and at the terminal point, WITH each run's own terminal step
    (rule ten: the per-cycle objective change at termination bounds only
    stopping slack, not path divergence -- CLAUDE.md evidence rule). s31c
    (cap 90) did not settle; the bar is reported, not asserted as valid."""
    if not os.path.exists(S32_S31C_G_PATH):
        return {'available': False, 'reason': f's31c reference not found at {S32_S31C_G_PATH}'}

    with open(S32_S31C_G_PATH) as handle:
        s31c_report = json.load(handle)
    s31c_rows = s31c_report.get('cycle_trajectory', [])
    s31c_by_cycle = {r['cycle']: r for r in s31c_rows if r.get('cycle') is not None}
    s32_by_cycle = {r['cycle']: r for r in rows if r.get('cycle') is not None}

    matched_cycles = sorted(set(s31c_by_cycle) & set(s32_by_cycle))
    matched = []
    for cycle in matched_cycles:
        s32_r = s32_by_cycle[cycle].get('recourse')
        s31c_r = s31c_by_cycle[cycle].get('recourse')
        matched.append({
            'cycle': cycle,
            's32_recourse': s32_r,
            's31c_recourse': s31c_r,
            'difference': (s32_r - s31c_r) if (s32_r is not None and s31c_r is not None) else None,
        })

    s32_last = rows[-1] if rows else {}
    s31c_last = s31c_rows[-1] if s31c_rows else {}
    s32_terminal_step = s32_last.get('objective_change_abs')
    s31c_terminal_step = s31c_last.get('objective_change_abs')
    terminal_difference = (
        (s32_last.get('recourse') - s31c_last.get('recourse'))
        if (s32_last.get('recourse') is not None and s31c_last.get('recourse') is not None)
        else None
    )
    error_bar = (
        (abs(s32_terminal_step) + abs(s31c_terminal_step))
        if (s32_terminal_step is not None and s31c_terminal_step is not None) else None
    )

    return {
        'available': True,
        's31c_reference_path': os.path.relpath(S32_S31C_G_PATH, REPO),
        'matched_cycles': matched,
        'terminal': {
            's32_cycle': s32_last.get('cycle'),
            's31c_cycle': s31c_last.get('cycle'),
            's32_recourse': s32_last.get('recourse'),
            's31c_recourse': s31c_last.get('recourse'),
            'difference': terminal_difference,
            's32_terminal_step_objective_change_abs': s32_terminal_step,
            's31c_terminal_step_objective_change_abs': s31c_terminal_step,
            'error_bar_sum_of_terminal_steps': error_bar,
            'determinate_at_gt_error_bar': (
                (abs(terminal_difference) > error_bar)
                if (terminal_difference is not None and error_bar) else None
            ),
        },
        'reading_rule': (
            's31c (cap 90, rule-ten 1.96) was NOT converged (Addendum 13: "still '
            'descending"), so this bar bounds stopping slack only, not path '
            'divergence (CLAUDE.md evidence rule, 2026-09-13 refinement): a '
            'difference smaller than error_bar_sum_of_terminal_steps is '
            'indeterminate, not a result; a difference larger than it is only '
            '"not explained by stopping slack", not evidence of a real limiting '
            'difference, because s31c has not settled. Trajectories differ by '
            'initial rho (pf 1.0 vs 300, v 1.0 vs 1.5) and by the balancing rule '
            '(freeze clause removed), so matched-cycle differences are NOT '
            'attributable to the stopping rule alone (frozen spec `gates.not_a_pass_criterion`).'
        ),
    }


def write_boyd_terminal_s32(planning, sed, models, rows, report, out_dir, label):
    """s32 Part 2: writes `boyd_terminal.json` -- the frozen spec's
    `report_terminal` fields not already covered by
    `write_interface_settlement_detail_s31c` (cancellation residual,
    per-DSO settlement/flexibility volumes) and `write_component_levels_terminal`
    (D rows), which this function calls FIRST, exactly as the s31c hook does.
    Zero extra solves -- reads the SAME final `models` `run_admm_arm` built."""
    settlement_path = write_interface_settlement_detail_s31c(
        planning, sed, models, rows, report, out_dir, label)

    last_row = rows[-1] if rows else {}
    converged_at_cycle = report.get('converged_at_cycle')
    stopped_by = 'boyd' if (converged_at_cycle is not None and converged_at_cycle == last_row.get('cycle')) else 'cap'

    payload = {
        'stage': 'P5.15 Step 3.2 + 3.3(a) (s32) -- Boyd stopping rule / residual balancing terminal report',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a)',
            'PLANNER_BRIEF_2026-09-13.md Addendum 13',
        ],
        'spec_file': os.path.relpath(S32_SPEC_PATH, REPO),
        'spec_file_sha256': S32_SPEC_SHA256,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'cycles': len(rows),
        'stopped_by': stopped_by,
        'converged_at_cycle': converged_at_cycle,
        'binding_test_per_channel': _s32_binding_test(last_row),
        'rho_trajectory_per_channel': _s32_rho_trajectory(rows),
        'system_cost_vs_s31c': _s32_system_cost_vs_s31c(rows, report),
        'network_failures_summary': report.get('network_failures_summary'),
        'component_levels_terminal_and_settlement_detail_path': settlement_path,
        'note_D_rows_and_cancellation_residual': (
            'D rows are in component_levels_terminal.json (written by '
            'write_component_levels_terminal, called first by '
            'write_interface_settlement_detail_s31c above); the cancellation '
            'residual T_TSO + sum(T_DSO) is '
            '"t_tso_plus_t_dso_terminal" and per-DSO settlement/flexibility '
            'volumes are "interface_consensus_residual_per_dso" / '
            '"flexibility_volumes_per_dso" in interface_settlement_detail_s31c.json.'
        ),
    }

    path = os.path.join(out_dir, 'boyd_terminal.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S32] boyd_terminal.json written: {path}')
    return path


# ===========================================================================
# P5.15 Step 3.2 E2 -- gamma tied to rho, cycle-30 freeze, 3 consecutive
# converged cycles. Level-capture arm `s33e2`. Authority:
# PLANNER_BRIEF_2026-09-13.md Addendum 14. Binding specification:
# data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json (supersedes
# v2 data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json). Same
# machinery as `s32` (lock, heartbeat, stdout/stderr, results_dir redirect,
# guard, trajectory, post-run writers) -- other arms (including `s32`) are
# UNCHANGED by this section.
# ===========================================================================

OUT_S33E2 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33_E2_run')
S33E2_SPEC_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S33', 'frozen_s33_e2_spec_v3_825f1f02.json')
S33E2_SPEC_SHA256 = '825f1f02d5319137e0248fc529b9aefdca0251272529ee140e09af6e73a04780'
S33E2_CAP = 150
S33E2_REL = 1e-4
S33E2_S31C_G_PATH = os.path.join(OUT_S31C, 'g_baseline.json')
S33E2_S32_G_PATH = os.path.join(OUT_S32, 'g_baseline.json')
S33E2_MATCHED_CYCLES = (1, 2, 3, 5, 10, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150)

# Per-cycle-channel fields are unchanged from v2 (gamma is not stored as a
# per-channel-suffixed key inside `get_admm_boyd_residual_metrics`'s
# `channel_entry` dict -- it lives in the ADMM-loop-level `admm_diagnostics`
# dict as `gamma_{group}_before/after`, checked via
# S33E2_ADMM_DIAGNOSTICS_KEYS below, together with `freeze_active`'s
# production name `rho_freeze_active`).
S33E2_REPORT_PER_CYCLE_CHANNEL_FIELDS = S32_REPORT_PER_CYCLE_CHANNEL_FIELDS
S33E2_ADMM_DIAGNOSTICS_KEYS = S32_ADMM_DIAGNOSTICS_KEYS + (
    'gamma_v_before', 'gamma_v_after', 'gamma_pf_before', 'gamma_pf_after',
    'gamma_ess_before', 'gamma_ess_after', 'rho_freeze_active',
    'freeze_after_cycle', 'gamma_policy', 'gamma_tau',
)


def _s33e2_spec_hash():
    with open(S33E2_SPEC_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def assert_s33e2_capture_paths(planning):
    """Rule eleven for the s33e2 arm (frozen spec v3). Reuses
    `assert_s31c_capture_paths` (still required: the s33e2 hook also calls
    `write_component_levels_terminal` and
    `write_interface_settlement_detail_s31c`, which cover `report_terminal`'s
    "D rows", "cancellation residual", "per-DSO settlement and flexibility
    volumes" and "network failures by tier" fields) and adds the v3-only
    checks (gamma tied to rho, tau, cycle-30 freeze, 3 consecutive converged
    cycles, the new gamma/freeze diagnostics keys, and the new interface-
    voltage writer). Deliberately does NOT call `assert_s32_capture_paths`
    (that function hashes the v2 spec file against `S32_SPEC_SHA256`, which
    is orthogonal to and would be misleading for a v3 run); the Boyd
    per-cycle-channel field check it performs is reused directly via
    `S32_REPORT_PER_CYCLE_CHANNEL_FIELDS` (unchanged by v3) instead. Zero
    solves.
    """
    checklist = dict(assert_s31c_capture_paths(planning))

    # -- spec file identity (v3) ------------------------------------------
    observed_hash = _s33e2_spec_hash()
    checklist['spec_file_hash_matches'] = (observed_hash == S33E2_SPEC_SHA256)
    checklist['spec_file_hash_observed'] = observed_hash

    # -- boyd tolerance source and value (unchanged from s32) -------------
    admm_params = planning.params.admm
    checklist['boyd_eps_source_is_case_file'] = (admm_params.boyd_eps_source == 'case_file')
    checklist['boyd_eps_abs_is_1e-5'] = (admm_params.tol['boyd']['eps_abs'] == 1e-5)
    checklist['boyd_eps_rel_is_1e-4'] = (admm_params.tol['boyd']['eps_rel'] == 1e-4)

    # -- v3-only case-file settings ----------------------------------------
    checklist['gamma_policy_is_tied_to_rho'] = (
        admm_params.proximal_regularization['tso'].get('gamma_policy') == 'tied_to_rho')
    checklist['gamma_tau_is_1'] = (admm_params.proximal_regularization['tso'].get('tau') == 1.0)
    checklist['freeze_after_cycle_is_30'] = (admm_params.penalty_update.get('freeze_after_cycle') == 30)
    checklist['minimum_consecutive_converged_cycles_is_3'] = (admm_params.minimum_consecutive_converged_cycles == 3)

    # -- initial rho: case-file value (1.0); N.RHO must NOT be applied ----
    rho_all_one = all(
        float(v) == 1.0
        for group in ('v', 'pf', 'ess')
        for v in admm_params.rho[group].values()
    )
    checklist['initial_rho_all_one'] = rho_all_one
    checklist['initial_rho_snapshot'] = {
        group: dict(admm_params.rho[group]) for group in ('v', 'pf', 'ess')
    }

    # -- per-cycle-channel capture (unchanged from v2) ---------------------
    checklist['srp_get_admm_boyd_residual_metrics'] = callable(
        getattr(srp, 'get_admm_boyd_residual_metrics', None))
    checklist['srp_update_admm_penalties_accepts_iter'] = (
        'iter' in inspect.signature(srp._update_admm_penalties).parameters)

    boyd_fn_source = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    for field in S33E2_REPORT_PER_CYCLE_CHANNEL_FIELDS:
        checklist[f'boyd_field_{field}_in_source'] = (f"'{field}':" in boyd_fn_source)

    module_source = inspect.getsource(srp)
    for key in S33E2_ADMM_DIAGNOSTICS_KEYS:
        checklist[f'admm_diagnostics_key_{key}_present'] = (f"'{key}':" in module_source)

    # -- structural: TSO gamma Params are mutable (spec v3 requirement) ---
    checklist['prox_gamma_v_mutable_in_source'] = (
        'model[year][day].prox_gamma_v = pe.Param(mutable=True' in module_source)

    # -- report_terminal fields not already covered by assert_s31c_capture_paths
    checklist['s31c_g_baseline_reference_exists'] = os.path.exists(S33E2_S31C_G_PATH)
    checklist['s32_g_baseline_reference_exists'] = os.path.exists(S33E2_S32_G_PATH)
    checklist['write_interface_voltage_terminal_callable'] = callable(
        globals().get('write_interface_voltage_terminal'))

    missing = [name for name, ok in checklist.items()
               if isinstance(ok, bool) and not ok]
    if missing:
        raise RuntimeError(f'S33E2 capture-path pre-flight FAILED, missing/broken: {missing}')
    return checklist


def _s33e2_gamma_trajectory(rows):
    trajectory = {'v': [], 'pf': [], 'ess': []}
    for row in rows:
        for group in ('v', 'pf', 'ess'):
            trajectory[group].append({
                'cycle': row.get('cycle'),
                'gamma_before': row.get(f'gamma_{group}_before'),
                'gamma_after': row.get(f'gamma_{group}_after'),
                'rho_freeze_active': row.get('rho_freeze_active'),
            })
    return trajectory


def _s33e2_system_cost_vs_references(rows, report):
    """System cost (gross_operational_cost -- see `objective_convention` in
    the payload this feeds) vs BOTH the s31c and s32 reference trajectories,
    at the spec's matched cycles and at the terminal point, WITH each run's
    own terminal step (rule ten: bounds stopping slack only, not path
    divergence -- CLAUDE.md evidence rule)."""
    result = {}
    for ref_name, ref_path in (('s31c', S33E2_S31C_G_PATH), ('s32', S33E2_S32_G_PATH)):
        if not os.path.exists(ref_path):
            result[ref_name] = {'available': False, 'reason': f'{ref_name} reference not found at {ref_path}'}
            continue
        with open(ref_path) as handle:
            ref_report = json.load(handle)
        ref_rows = ref_report.get('cycle_trajectory', [])
        ref_by_cycle = {r['cycle']: r for r in ref_rows if r.get('cycle') is not None}
        e2_by_cycle = {r['cycle']: r for r in rows if r.get('cycle') is not None}

        matched = []
        for cycle in S33E2_MATCHED_CYCLES:
            if cycle not in ref_by_cycle or cycle not in e2_by_cycle:
                continue
            e2_v = e2_by_cycle[cycle].get('gross_operational_cost')
            ref_v = ref_by_cycle[cycle].get('gross_operational_cost')
            matched.append({
                'cycle': cycle,
                's33e2_gross_operational_cost': e2_v,
                f'{ref_name}_gross_operational_cost': ref_v,
                'difference': (e2_v - ref_v) if (e2_v is not None and ref_v is not None) else None,
            })

        e2_last = rows[-1] if rows else {}
        ref_last = ref_rows[-1] if ref_rows else {}
        e2_terminal_step = e2_last.get('objective_change_abs')
        ref_terminal_step = ref_last.get('objective_change_abs')
        terminal_difference = (
            (e2_last.get('gross_operational_cost') - ref_last.get('gross_operational_cost'))
            if (e2_last.get('gross_operational_cost') is not None and ref_last.get('gross_operational_cost') is not None)
            else None
        )
        error_bar = (
            (abs(e2_terminal_step) + abs(ref_terminal_step))
            if (e2_terminal_step is not None and ref_terminal_step is not None) else None
        )
        result[ref_name] = {
            'available': True,
            'reference_path': os.path.relpath(ref_path, REPO),
            'matched_cycles': matched,
            'terminal': {
                's33e2_cycle': e2_last.get('cycle'),
                f'{ref_name}_cycle': ref_last.get('cycle'),
                's33e2_gross_operational_cost': e2_last.get('gross_operational_cost'),
                f'{ref_name}_gross_operational_cost': ref_last.get('gross_operational_cost'),
                'difference': terminal_difference,
                's33e2_terminal_step_objective_change_abs': e2_terminal_step,
                f'{ref_name}_terminal_step_objective_change_abs': ref_terminal_step,
                'error_bar_sum_of_terminal_steps': error_bar,
                'determinate_at_gt_error_bar': (
                    (abs(terminal_difference) > error_bar)
                    if (terminal_difference is not None and error_bar) else None
                ),
            },
        }
    return result


def _interface_voltage_detail(planning, models):
    """Zero solves: per-entry interface-voltage detail at the CURRENT
    (terminal, or preflight-cycle) model state -- TSO copy
    (`expected_interface_vmag[dn, p]`, already in pu -- see
    `update_transmission_model_to_admm`, multiplied by `v_base` elsewhere to
    get kV) and DSO copy (`expected_interface_vmag[p]`, same convention),
    and the distance to the TSO node's own [v_min, v_max] pu bounds
    (`network.get_node_voltage_limits`)."""
    transmission_network = planning.transmission_network
    tso_model = models['tso']
    dso_models_by_node = models['dso']

    entries = []
    for node_id in planning.active_distribution_network_nodes:
        dn = transmission_network.active_distribution_network_nodes.index(node_id)
        dso_model = dso_models_by_node[node_id]
        for year in planning.years:
            for day in planning.days:
                network = transmission_network.network[year][day]
                v_min, v_max = network.get_node_voltage_limits(node_id)
                for p in tso_model[year][day].periods:
                    tso_pu = pe.value(tso_model[year][day].expected_interface_vmag[dn, p])
                    dso_pu = pe.value(dso_model[year][day].expected_interface_vmag[p])
                    dist_to_min = tso_pu - v_min
                    dist_to_max = v_max - tso_pu
                    if dist_to_min <= dist_to_max:
                        nearest_bound, distance = 'v_min', dist_to_min
                    else:
                        nearest_bound, distance = 'v_max', dist_to_max
                    entries.append({
                        'node_id': node_id, 'year': str(year), 'day': str(day), 'period': p,
                        'tso_pu': tso_pu, 'dso_pu': dso_pu,
                        'v_min_pu': v_min, 'v_max_pu': v_max,
                        'distance_to_nearest_bound_pu': distance,
                        'nearest_bound': nearest_bound,
                    })

    per_node_min = {}
    for e in entries:
        nid = e['node_id']
        if nid not in per_node_min or e['distance_to_nearest_bound_pu'] < per_node_min[nid]:
            per_node_min[nid] = e['distance_to_nearest_bound_pu']
    min_distance_overall = min((e['distance_to_nearest_bound_pu'] for e in entries), default=None)
    n_at_bound = sum(1 for e in entries if e['distance_to_nearest_bound_pu'] <= 1e-6)
    n_within_0p005 = sum(1 for e in entries if e['distance_to_nearest_bound_pu'] <= 0.005)

    summary = {
        'n_entries': len(entries),
        'min_distance_to_bound_pu_overall': min_distance_overall,
        'min_distance_to_bound_pu_per_node': {str(k): v for k, v in per_node_min.items()},
        'n_entries_at_bound_within_1e-6_pu': n_at_bound,
        'n_entries_within_0p005_pu': n_within_0p005,
    }
    return entries, summary


def write_interface_voltage_terminal(planning, models, out_dir, label, cycle=None):
    """Spec v3 `report_terminal` -- "per-entry terminal interface V: TSO and
    DSO copies per node/year/day/period (pu), and distance to the TSO node
    bounds". Top-level shape is exactly `{"entries": [...], "summary": {...}}`
    (metadata folded into `summary`, not a separate top-level key)."""
    entries, summary = _interface_voltage_detail(planning, models)
    summary = dict(summary)
    summary.update({
        'stage': 'P5.15 Step 3.2 E2 -- per-entry terminal interface voltage vs TSO node bounds',
        'authority': 'data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json report_terminal',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'cycle': cycle,
    })
    payload = {'entries': entries, 'summary': summary}

    path = os.path.join(out_dir, 'interface_voltage_terminal.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S33E2] interface_voltage_terminal.json written: {path}')
    return path


def write_boyd_terminal_s33e2(planning, sed, models, rows, report, out_dir, label):
    """s33e2 Part 2: writes `boyd_terminal.json` -- the frozen v3 spec's
    `report_terminal` fields not already covered by
    `write_interface_settlement_detail_s31c` (cancellation residual,
    per-DSO settlement/flexibility volumes) and `write_component_levels_terminal`
    (D rows), which this function calls FIRST, exactly as the s32 hook does
    -- and `write_interface_voltage_terminal`. Zero extra solves -- reads
    the SAME final `models` `run_admm_arm` built."""
    settlement_path = write_interface_settlement_detail_s31c(
        planning, sed, models, rows, report, out_dir, label)

    last_row = rows[-1] if rows else {}
    converged_at_cycle = report.get('converged_at_cycle')
    stopped_by = 'boyd' if (converged_at_cycle is not None and converged_at_cycle == last_row.get('cycle')) else 'cap'

    voltage_path = write_interface_voltage_terminal(
        planning, models, out_dir, label, cycle=last_row.get('cycle'))

    with open(S33E2_SPEC_PATH) as handle:
        spec_json = json.load(handle)

    # Consecutive-converged-cycles run at the terminal row, and the cycle it
    # began (scanning backward from the terminal row while cycle_convergence
    # holds; sanity-checked against the row's own recorded count).
    consecutive_converged_at_stop = last_row.get('consecutive_converged_cycles')
    began_at_cycle = None
    if consecutive_converged_at_stop:
        count = 0
        for row in reversed(rows):
            if row.get('cycle_convergence'):
                count += 1
                began_at_cycle = row.get('cycle')
            else:
                break
        if count != consecutive_converged_at_stop:
            began_at_cycle = None

    payload = {
        'stage': 'P5.15 Step 3.2 E2 (s33e2) -- gamma tied to rho, freeze at cycle 30, 3 consecutive converged cycles',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 14',
            'data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json',
        ],
        'spec_file': os.path.relpath(S33E2_SPEC_PATH, REPO),
        'spec_file_sha256': S33E2_SPEC_SHA256,
        'predecessor_spec_file': spec_json.get('predecessor', {}).get('path'),
        'predecessor_spec_sha256': spec_json.get('predecessor', {}).get('sha256'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'objective_convention': (
            'gross_operational_cost (matched-cycle and terminal comparisons '
            'below use gross, NOT net_operational_recourse -- see CLAUDE.md '
            '"Reporting conventions")'
        ),
        'cycles': len(rows),
        'stopped_by': stopped_by,
        'converged_at_cycle': converged_at_cycle,
        'consecutive_converged_at_stop': consecutive_converged_at_stop,
        'consecutive_converged_run_began_at_cycle': began_at_cycle,
        'binding_test_per_channel': _s32_binding_test(last_row),
        'rho_trajectory_per_channel': _s32_rho_trajectory(rows),
        'gamma_trajectory_per_channel': _s33e2_gamma_trajectory(rows),
        'freeze_after_cycle': last_row.get('freeze_after_cycle'),
        'rho_freeze_active_at_terminal': last_row.get('rho_freeze_active'),
        'gamma_policy': last_row.get('gamma_policy'),
        'gamma_tau': last_row.get('gamma_tau'),
        'system_cost_vs_references': _s33e2_system_cost_vs_references(rows, report),
        'e4_noise_floor_delta_c_vs_terminal_steps': {
            'delta_c_from_spec_eps_abs_derivation': spec_json.get('stopping_rule', {}).get('eps_abs_derivation', {}).get('delta_c'),
            'terminal_boyd_r_per_channel': {g: last_row.get(f'boyd_{g}_r') for g in ('v', 'pf', 'ess')},
            'terminal_boyd_s_per_channel': {g: last_row.get(f'boyd_{g}_s') for g in ('v', 'pf', 'ess')},
            'note': (
                'delta_c (E4) is the per-entry local-solver noise floor; '
                'boyd_{g}_r/s are Euclidean-norm residuals over ALL entries '
                'in channel g (not per-entry) -- reported side by side per '
                'report_terminal, not rescaled to per-entry here.'
            ),
        },
        'network_failures_summary': report.get('network_failures_summary'),
        'component_levels_terminal_and_settlement_detail_path': settlement_path,
        'interface_voltage_terminal_path': os.path.relpath(voltage_path, REPO),
        'note_D_rows_and_cancellation_residual': (
            'D rows are in component_levels_terminal.json (written by '
            'write_component_levels_terminal, called first by '
            'write_interface_settlement_detail_s31c above); the cancellation '
            'residual T_TSO + sum(T_DSO) is '
            '"t_tso_plus_t_dso_terminal" and per-DSO settlement/flexibility '
            'volumes are "interface_consensus_residual_per_dso" / '
            '"flexibility_volumes_per_dso" in interface_settlement_detail_s31c.json.'
        ),
    }

    path = os.path.join(out_dir, 'boyd_terminal.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S33E2] boyd_terminal.json written: {path}')
    return path


# ===========================================================================
# P5.15 Step 3.4 (+3.3(b) folded in) -- D5 ESSO AL scaling, fixed sigma, the
# v4 rho/freeze policy, S_ref. Authority: PLANNER_BRIEF_2026-09-13.md
# Addendum 15 item 5. Binding specification:
# data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json (supersedes v3
# data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json). Same
# machinery as `s33e2` (lock, heartbeat, stdout/stderr, results_dir
# redirect, guard, trajectory, post-run writers) -- other arms (including
# `s32`, `s33e2`) are UNCHANGED by this section.
# ===========================================================================

OUT_S34 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34_run')
S34_SPEC_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S34', 'frozen_s34_spec_v4_966940a7.json')
S34_SPEC_SHA256 = '966940a789db3093a57d5fdc23be8ccb172b762bf02d0cd09eeaaf0f52f544e9'
S34_CAP = 150
S34_REL = 1e-4
S34_S31C_G_PATH = os.path.join(OUT_S31C, 'g_baseline.json')
S34_S32_G_PATH = os.path.join(OUT_S32, 'g_baseline.json')
S34_S33E2_G_PATH = os.path.join(OUT_S33E2, 'g_baseline.json')
S34_S33E2_LEAK_PATH = os.path.join(OUT_S33E2, 'leak_classification_baseline.jsonl')
S34_MATCHED_CYCLES = S33E2_MATCHED_CYCLES

# Per-cycle-channel fields are unchanged from v2/v3 (S_ref, sigma and
# al_scale_esso are run-level constants, not per-channel Boyd fields).
S34_REPORT_PER_CYCLE_CHANNEL_FIELDS = S32_REPORT_PER_CYCLE_CHANNEL_FIELDS
S34_ADMM_DIAGNOSTICS_KEYS = S33E2_ADMM_DIAGNOSTICS_KEYS + (
    'sigma_fixed', 'sigma_computed', 'al_scale_esso', 'shared_ess_reference_rating_mva',
    'freeze_after_unchanged_cycles', 'freeze_backstop_cycle',
    'rho_frozen_v', 'rho_frozen_pf', 'rho_frozen_ess',
    'rho_unchanged_streak_v', 'rho_unchanged_streak_pf', 'rho_unchanged_streak_ess',
    'rho_at_clamp_v', 'rho_at_clamp_pf', 'rho_at_clamp_ess',
    'efc_per_day_max',
)


def _s34_spec_hash():
    with open(S34_SPEC_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def assert_s34_capture_paths(planning):
    """Rule eleven for the s34 arm (frozen spec v4). Reuses
    `assert_s31c_capture_paths` (still required: the s34 hook also calls
    `write_component_levels_terminal` and `write_interface_settlement_detail_s31c`,
    which cover `report_terminal`'s "D rows", "cancellation residual",
    "per-DSO settlement and flexibility volumes" and "network failures by
    tier" fields). Deliberately does NOT reuse `assert_s32_capture_paths` or
    `assert_s33e2_capture_paths` -- both hard-assert THEIR OWN spec's rho/
    freeze case-file values (rho==1.0, `freeze_after_cycle==30`), which are
    superseded by v4 and would raise here; the genuinely still-required
    structural checks those functions perform (the Boyd per-cycle-channel
    fields, the interface-voltage writer) are reproduced directly below
    instead. Zero solves.
    """
    checklist = dict(assert_s31c_capture_paths(planning))

    # -- spec file identity (v4) ------------------------------------------
    observed_hash = _s34_spec_hash()
    checklist['s34_spec_file_hash_matches'] = (observed_hash == S34_SPEC_SHA256)
    checklist['s34_spec_file_hash_observed'] = observed_hash

    admm_params = planning.params.admm

    # -- boyd tolerance source and value (unchanged from s32/s33e2) -------
    checklist['boyd_eps_source_is_case_file'] = (admm_params.boyd_eps_source == 'case_file')
    checklist['boyd_eps_abs_is_1e-5'] = (admm_params.tol['boyd']['eps_abs'] == 1e-5)
    checklist['boyd_eps_rel_is_1e-4'] = (admm_params.tol['boyd']['eps_rel'] == 1e-4)

    # -- fixed sigma (b_sigma_fixed) ---------------------------------------
    checklist['objective_scale_is_93635360'] = (admm_params.objective_scale == 93635360.0)
    checklist['objective_scale_source_is_case_file'] = (admm_params.objective_scale_source == 'case_file')
    checklist['objective_scale_assert_factor_at_least_1'] = (admm_params.objective_scale_assert_factor >= 1.0)
    checklist['srp_resolve_common_admm_objective_scale_callable'] = callable(
        getattr(srp, '_resolve_common_admm_objective_scale', None))

    # -- ESSO AL scaling (a_D5_esso_scaling) -------------------------------
    checklist['al_scale_esso_present'] = (admm_params.esso_al_scale.get('source') == 'case_file')
    checklist['al_scale_esso_mode_is_sigma_over_median_block_weight'] = (
        admm_params.esso_al_scale.get('mode') == 'sigma_over_median_block_weight')
    checklist['srp_resolve_esso_al_scale_callable'] = callable(getattr(srp, '_resolve_esso_al_scale', None))
    checklist['srp_update_shared_energy_storage_model_to_admm_accepts_al_scale_esso'] = (
        'al_scale_esso' in inspect.signature(srp.update_shared_energy_storage_model_to_admm).parameters)

    # -- S_ref (d_ess_reference_rating) ------------------------------------
    checklist['shared_ess_reference_rating_mva_is_2p5'] = (admm_params.shared_ess_reference_rating_mva == 2.5)
    checklist['srp_admm_shared_ess_reference_mva_callable'] = callable(
        getattr(srp, '_admm_shared_ess_reference_mva', None))

    # -- initial rho: v 0.0077 / pf 0.198 / ess 0.05, EVERY network + esso -
    initial_rho_ok = (
        all(float(v) == 0.0077 for v in admm_params.rho['v'].values()) and
        all(float(v) == 0.198 for v in admm_params.rho['pf'].values()) and
        all(float(v) == 0.05 for v in admm_params.rho['ess'].values())
    )
    checklist['initial_rho_v_pf_ess_matches_spec_v4'] = initial_rho_ok
    checklist['initial_rho_snapshot'] = {
        group: dict(admm_params.rho[group]) for group in ('v', 'pf', 'ess')
    }

    # -- v4 freeze policy (c_rho_policy) -----------------------------------
    checklist['freeze_after_unchanged_cycles_is_10'] = (admm_params.penalty_update.get('freeze_after_unchanged_cycles') == 10)
    checklist['freeze_backstop_cycle_is_60'] = (admm_params.penalty_update.get('freeze_backstop_cycle') == 60)
    checklist['minimum_consecutive_converged_cycles_is_3'] = (admm_params.minimum_consecutive_converged_cycles == 3)
    checklist['srp_update_admm_penalties_accepts_freeze_state'] = (
        'freeze_state' in inspect.signature(srp._update_admm_penalties).parameters)
    checklist['srp_init_admm_freeze_state_callable'] = callable(getattr(srp, '_init_admm_freeze_state', None))

    # -- per-cycle-channel / admm_diagnostics capture ----------------------
    checklist['srp_get_admm_boyd_residual_metrics'] = callable(
        getattr(srp, 'get_admm_boyd_residual_metrics', None))
    boyd_fn_source = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    for field in S34_REPORT_PER_CYCLE_CHANNEL_FIELDS:
        checklist[f'boyd_field_{field}_in_source'] = (f"'{field}':" in boyd_fn_source)

    module_source = inspect.getsource(srp)
    for key in S34_ADMM_DIAGNOSTICS_KEYS:
        checklist[f'admm_diagnostics_key_{key}_present'] = (f"'{key}':" in module_source)

    checklist['srp_get_admm_efc_per_day_max_callable'] = callable(getattr(srp, '_get_admm_efc_per_day_max', None))

    # -- structural: TSO gamma Params are mutable (spec v3, unchanged by v4)
    checklist['prox_gamma_v_mutable_in_source'] = (
        'model[year][day].prox_gamma_v = pe.Param(mutable=True' in module_source)
    checklist['write_interface_voltage_terminal_callable'] = callable(
        globals().get('write_interface_voltage_terminal'))
    checklist['write_component_levels_terminal_callable'] = callable(
        globals().get('write_component_levels_terminal'))
    checklist['write_interface_settlement_detail_s31c_callable'] = callable(
        globals().get('write_interface_settlement_detail_s31c'))

    # -- capture_additions: per-cycle sidecar hook points must exist -------
    checklist['srp_get_operational_recourse_block_components_callable'] = callable(
        getattr(srp, '_get_operational_recourse_block_components', None))
    checklist['srp_get_operational_objective_component_blocks_callable'] = callable(
        getattr(srp, '_get_operational_objective_component_blocks', None))
    checklist['s34_capture_hooks_callable'] = callable(globals().get('s34_capture_hooks'))

    # -- report_terminal fields not already covered by assert_s33e2_capture_paths
    checklist['s31c_g_baseline_reference_exists'] = os.path.exists(S34_S31C_G_PATH)
    checklist['s32_g_baseline_reference_exists'] = os.path.exists(S34_S32_G_PATH)
    checklist['s33e2_g_baseline_reference_exists'] = os.path.exists(S34_S33E2_G_PATH)
    checklist['s33e2_leak_classification_baseline_exists'] = os.path.exists(S34_S33E2_LEAK_PATH)
    checklist['write_boyd_terminal_s34_callable'] = callable(globals().get('write_boyd_terminal_s34'))

    missing = [name for name, ok in checklist.items()
               if isinstance(ok, bool) and not ok]
    if missing:
        raise RuntimeError(f'S34 capture-path pre-flight FAILED, missing/broken: {missing}')
    return checklist


@contextmanager
def s34_capture_hooks(recourse_jump_path, ess_stride_path, stride=1):
    """P5.15 Step 3.4 `capture_additions` (spec v4): monkeypatches
    `srp.get_admm_boyd_residual_metrics` -- called exactly once per ADMM
    cycle, with the SAME already-solved `tso_model`/`dso_models`/`esso_model`
    and the SAME `consensus_vars` the cycle's own dual/Boyd computation uses
    -- to ALSO, unconditionally (no tolerance gate, unlike production's OWN
    `[RECOURSE JUMP]` print which only fires when recourse stationarity
    FAILS; Z4 found it stops appearing after cycle 62 for that reason),
    write two per-cycle JSONL sidecars:

      1. `recourse_jump_path`: the SAME block decomposition production's own
         `_print_recourse_jump_diagnostics` computes (top-10 |delta| blocks,
         reconciliation), via the SAME unmodified production functions
         (`_get_operational_recourse_block_components`,
         `_get_operational_objective_component_blocks`), every cycle.
      2. `ess_stride_path`: per-entry shared-ESS z (consensus) and x
         (TSO/DSO/ESSO copies) on the given cycle stride (default 1 = every
         cycle), plus EFC/day per node (not just the max the production
         `efc_per_day_max` diagnostic reports).

    Zero extra solves (both sidecars only read already-solved Pyomo Var/Param
    values); the wrapped function's own return value and behaviour are
    unchanged -- a pure side-effecting wrapper, uninstalled (restoring the
    original function) even on error.
    """
    real_fn = srp.get_admm_boyd_residual_metrics
    state = {
        'cycle': 0,
        'previous_recourse_blocks': None,
        'previous_objective_component_blocks': None,
    }

    def wrapper(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):
        state['cycle'] += 1
        cycle = state['cycle']
        operational_models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}

        # ---- (1) unconditional [RECOURSE JUMP] block decomposition ----
        try:
            current_blocks = srp._get_operational_recourse_block_components(planning_problem, operational_models)
            current_obj_blocks = srp._get_operational_objective_component_blocks(planning_problem, operational_models)
            capture_error = None
        except Exception as error:  # pragma: no cover -- defensive, never expected
            current_blocks, current_obj_blocks = None, None
            capture_error = f'{type(error).__name__}: {error}'

        entry = {'cycle': cycle, 'error': capture_error, 'block_deltas': None,
                 'block_total_current': None, 'block_total_previous': None,
                 'objective_component_block_deltas': None}
        if current_blocks is not None:
            entry['block_total_current'] = sum(current_blocks.values())
            if state['previous_recourse_blocks'] is not None:
                previous_blocks = state['previous_recourse_blocks']
                entry['block_total_previous'] = sum(previous_blocks.values())
                deltas = []
                for key in set(current_blocks) | set(previous_blocks):
                    prev_v = previous_blocks.get(key, 0.0)
                    cur_v = current_blocks.get(key, 0.0)
                    agent, node_id, year, day = key
                    deltas.append({
                        'agent': agent, 'node_id': node_id, 'year': str(year), 'day': str(day),
                        'previous': prev_v, 'current': cur_v, 'delta': cur_v - prev_v,
                        'abs_delta': abs(cur_v - prev_v),
                    })
                deltas.sort(key=lambda e: e['abs_delta'], reverse=True)
                entry['block_deltas'] = deltas[:10]
            state['previous_recourse_blocks'] = current_blocks
        if current_obj_blocks is not None:
            # Each block's value is itself a {component_name: value} dict
            # (`_get_local_objective_components`), NOT a scalar -- flatten to
            # (block_key, component_name) before differencing.
            if state['previous_objective_component_blocks'] is not None:
                previous_obj_blocks = state['previous_objective_component_blocks']
                flat_current = {
                    (block_key, name): value
                    for block_key, components in current_obj_blocks.items()
                    for name, value in components.items()
                }
                flat_previous = {
                    (block_key, name): value
                    for block_key, components in previous_obj_blocks.items()
                    for name, value in components.items()
                }
                obj_deltas = []
                for key in set(flat_current) | set(flat_previous):
                    prev_v = flat_previous.get(key, 0.0)
                    cur_v = flat_current.get(key, 0.0)
                    block_key, component_name = key
                    obj_deltas.append({'block_key': str(block_key), 'component': component_name,
                                        'previous': prev_v, 'current': cur_v,
                                        'delta': cur_v - prev_v, 'abs_delta': abs(cur_v - prev_v)})
                obj_deltas.sort(key=lambda e: e['abs_delta'], reverse=True)
                entry['objective_component_block_deltas'] = obj_deltas[:10]
            state['previous_objective_component_blocks'] = current_obj_blocks

        with open(recourse_jump_path, 'a') as handle:
            handle.write(json.dumps(entry, default=str) + '\n')

        # ---- (2) per-entry ESS z/x on the recorded stride, + EFC/day/node ----
        if (cycle - 1) % max(int(stride), 1) == 0:
            ess_entries = []
            for node_id in planning_problem.active_distribution_network_nodes:
                for year in planning_problem.years:
                    for day in planning_problem.days:
                        for power_type in ('p', 'q'):
                            z_series = list(consensus_vars['ess']['z']['current'][node_id][year][day][power_type])
                            x_series = {
                                agent: list(consensus_vars['ess'][agent]['current'][node_id][year][day][power_type])
                                for agent in ('tso', 'dso', 'esso')
                            }
                            ess_entries.append({
                                'node_id': node_id, 'year': str(year), 'day': str(day),
                                'power_type': power_type, 'z': z_series, 'x': x_series,
                            })
            efc_per_node = {}
            for node_id, model in esso_model.items():
                values = []
                for y_inv in model.years:
                    for y in model.years:
                        avg = pe.value(model.es_avg_ch_dch_per_unit[y_inv, y], exception=False)
                        rated = pe.value(model.es_e_rated_per_unit[y_inv, y], exception=False)
                        if avg is None or not rated:
                            continue
                        values.append(avg / (2.0 * rated))
                efc_per_node[str(node_id)] = max(values) if values else None
            with open(ess_stride_path, 'a') as handle:
                handle.write(json.dumps(
                    {'cycle': cycle, 'stride': stride, 'entries': ess_entries,
                     'efc_per_day_per_node': efc_per_node},
                    default=str) + '\n')

        return real_fn(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters)

    srp.get_admm_boyd_residual_metrics = wrapper
    try:
        yield state
    finally:
        srp.get_admm_boyd_residual_metrics = real_fn


def _s34_rho_gamma_freeze_trajectory(rows):
    trajectory = {'v': [], 'pf': [], 'ess': []}
    for row in rows:
        for group in ('v', 'pf', 'ess'):
            trajectory[group].append({
                'cycle': row.get('cycle'),
                'rho_before': row.get(f'rho_{group}_before'),
                'rho_after': row.get(f'rho_{group}_after'),
                'action': row.get(f'rho_{group}_action'),
                'gamma_before': row.get(f'gamma_{group}_before'),
                'gamma_after': row.get(f'gamma_{group}_after'),
                'rho_frozen': row.get(f'rho_frozen_{group}'),
                'rho_unchanged_streak': row.get(f'rho_unchanged_streak_{group}'),
                'rho_at_clamp': row.get(f'rho_at_clamp_{group}'),
            })
    return trajectory


def _s34_system_cost_vs_references(rows, report):
    """System cost (gross_operational_cost) vs s31c, s32 AND s33e2, at the
    spec's matched cycles and at the terminal point, WITH each run's own
    terminal step (rule ten). Extends `_s33e2_system_cost_vs_references` with
    the s33e2 reference itself."""
    result = {}
    for ref_name, ref_path in (('s31c', S34_S31C_G_PATH), ('s32', S34_S32_G_PATH), ('s33e2', S34_S33E2_G_PATH)):
        if not os.path.exists(ref_path):
            result[ref_name] = {'available': False, 'reason': f'{ref_name} reference not found at {ref_path}'}
            continue
        with open(ref_path) as handle:
            ref_report = json.load(handle)
        ref_rows = ref_report.get('cycle_trajectory', [])
        ref_by_cycle = {r['cycle']: r for r in ref_rows if r.get('cycle') is not None}
        s34_by_cycle = {r['cycle']: r for r in rows if r.get('cycle') is not None}

        matched = []
        for cycle in S34_MATCHED_CYCLES:
            if cycle not in ref_by_cycle or cycle not in s34_by_cycle:
                continue
            s34_v = s34_by_cycle[cycle].get('gross_operational_cost')
            ref_v = ref_by_cycle[cycle].get('gross_operational_cost')
            matched.append({
                'cycle': cycle,
                's34_gross_operational_cost': s34_v,
                f'{ref_name}_gross_operational_cost': ref_v,
                'difference': (s34_v - ref_v) if (s34_v is not None and ref_v is not None) else None,
            })

        s34_last = rows[-1] if rows else {}
        ref_last = ref_rows[-1] if ref_rows else {}
        s34_terminal_step = s34_last.get('objective_change_abs')
        ref_terminal_step = ref_last.get('objective_change_abs')
        terminal_difference = (
            (s34_last.get('gross_operational_cost') - ref_last.get('gross_operational_cost'))
            if (s34_last.get('gross_operational_cost') is not None and ref_last.get('gross_operational_cost') is not None)
            else None
        )
        error_bar = (
            (abs(s34_terminal_step) + abs(ref_terminal_step))
            if (s34_terminal_step is not None and ref_terminal_step is not None) else None
        )
        result[ref_name] = {
            'available': True,
            'reference_path': os.path.relpath(ref_path, REPO),
            'matched_cycles': matched,
            'terminal': {
                's34_cycle': s34_last.get('cycle'),
                f'{ref_name}_cycle': ref_last.get('cycle'),
                's34_gross_operational_cost': s34_last.get('gross_operational_cost'),
                f'{ref_name}_gross_operational_cost': ref_last.get('gross_operational_cost'),
                'difference': terminal_difference,
                's34_terminal_step_objective_change_abs': s34_terminal_step,
                f'{ref_name}_terminal_step_objective_change_abs': ref_terminal_step,
                'error_bar_sum_of_terminal_steps': error_bar,
                'determinate_at_gt_error_bar': (
                    (abs(terminal_difference) > error_bar)
                    if (terminal_difference is not None and error_bar) else None
                ),
            },
        }
    return result


def write_boyd_terminal_s34(planning, sed, models, rows, report, out_dir, label):
    """s34 Part 2: writes `boyd_terminal.json` -- the frozen v4 spec's
    `report_terminal` fields not already covered by
    `write_interface_settlement_detail_s31c` (cancellation residual,
    per-DSO settlement/flexibility volumes), `write_component_levels_terminal`
    (D rows) and `write_interface_voltage_terminal`, plus the v4-specific
    additions: sigma_fixed/sigma_computed, al_scale_esso, S_ref, initial rho,
    per-channel freeze cycle and rho_at_clamp, and the system-cost comparison
    against s31c, s32 AND s33e2 at matched cycles. Zero extra solves."""
    settlement_path = write_interface_settlement_detail_s31c(
        planning, sed, models, rows, report, out_dir, label)

    last_row = rows[-1] if rows else {}
    converged_at_cycle = report.get('converged_at_cycle')
    stopped_by = 'boyd' if (converged_at_cycle is not None and converged_at_cycle == last_row.get('cycle')) else 'cap'

    voltage_path = write_interface_voltage_terminal(
        planning, models, out_dir, label, cycle=last_row.get('cycle'))

    with open(S34_SPEC_PATH) as handle:
        spec_json = json.load(handle)

    consecutive_converged_at_stop = last_row.get('consecutive_converged_cycles')
    began_at_cycle = None
    if consecutive_converged_at_stop:
        count = 0
        for row in reversed(rows):
            if row.get('cycle_convergence'):
                count += 1
                began_at_cycle = row.get('cycle')
            else:
                break
        if count != consecutive_converged_at_stop:
            began_at_cycle = None

    # Per-channel freeze cycle: the first row where rho_frozen_<g> is True.
    freeze_cycle_per_channel = {}
    rho_at_clamp_per_channel = {}
    for group in ('v', 'pf', 'ess'):
        freeze_cycle_per_channel[group] = next(
            (row.get('cycle') for row in rows if row.get(f'rho_frozen_{group}')), None)
        rho_at_clamp_per_channel[group] = last_row.get(f'rho_at_clamp_{group}')

    admm_params = planning.params.admm

    payload = {
        'stage': 'P5.15 Step 3.4 (+3.3(b) folded in) (s34) -- D5 ESSO AL scaling, '
                 'fixed sigma, v4 rho/freeze policy, S_ref -- terminal report',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5',
            'data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json',
        ],
        'spec_file': os.path.relpath(S34_SPEC_PATH, REPO),
        'spec_file_sha256': S34_SPEC_SHA256,
        'predecessor_spec_file': spec_json.get('predecessor', {}).get('path'),
        'predecessor_spec_sha256': spec_json.get('predecessor', {}).get('sha256'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'label': label,
        'objective_convention': (
            'gross_operational_cost (matched-cycle and terminal comparisons '
            'below use gross, NOT net_operational_recourse -- see CLAUDE.md '
            '"Reporting conventions")'
        ),
        'cycles': len(rows),
        'stopped_by': stopped_by,
        'converged_at_cycle': converged_at_cycle,
        'consecutive_converged_at_stop': consecutive_converged_at_stop,
        'consecutive_converged_run_began_at_cycle': began_at_cycle,
        'binding_test_per_channel': _s32_binding_test(last_row),
        'rho_gamma_freeze_trajectory_per_channel': _s34_rho_gamma_freeze_trajectory(rows),
        'sigma_fixed': last_row.get('sigma_fixed'),
        'sigma_computed': last_row.get('sigma_computed'),
        'al_scale_esso': last_row.get('al_scale_esso'),
        'shared_ess_reference_rating_mva': last_row.get('shared_ess_reference_rating_mva'),
        'initial_rho': {group: dict(admm_params.rho[group]) for group in ('v', 'pf', 'ess')},
        'freeze_after_unchanged_cycles': last_row.get('freeze_after_unchanged_cycles'),
        'freeze_backstop_cycle': last_row.get('freeze_backstop_cycle'),
        'freeze_cycle_per_channel': freeze_cycle_per_channel,
        'rho_at_clamp_per_channel_at_terminal': rho_at_clamp_per_channel,
        'rho_at_clamp_any_true_gate_failure_flag': any(rho_at_clamp_per_channel.values()),
        'efc_per_day_max_terminal': last_row.get('efc_per_day_max'),
        'system_cost_vs_s31c_s32_s33e2': _s34_system_cost_vs_references(rows, report),
        'network_failures_summary': report.get('network_failures_summary'),
        'component_levels_terminal_and_settlement_detail_path': settlement_path,
        'interface_voltage_terminal_path': os.path.relpath(voltage_path, REPO),
        'recourse_jump_sidecar_path': report.get('s34_recourse_jump_sidecar_path'),
        'ess_entry_stride_sidecar_path': report.get('s34_ess_entry_stride_sidecar_path'),
        'note_D_rows_and_cancellation_residual': (
            'D rows are in component_levels_terminal.json (written by '
            'write_component_levels_terminal, called first by '
            'write_interface_settlement_detail_s31c above); the cancellation '
            'residual T_TSO + sum(T_DSO) is '
            '"t_tso_plus_t_dso_terminal" and per-DSO settlement/flexibility '
            'volumes are "interface_consensus_residual_per_dso" / '
            '"flexibility_volumes_per_dso" in interface_settlement_detail_s31c.json.'
        ),
    }

    path = os.path.join(out_dir, 'boyd_terminal.json')
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S34] boyd_terminal.json written: {path}')
    return path


if __name__ == '__main__':
    _acquire_exclusive_run_lock()
    gate = sys.argv[1] if len(sys.argv) > 1 else None
    if gate == 'ablation_a':
        _require_fresh_output_root(OUT_ABL_A)
        _original_fresh_planning = O.fresh_planning

        def _fresh_planning_ablation_a(eval_id):
            return _configure_ablation_a(_original_fresh_planning(eval_id))

        O.fresh_planning = _fresh_planning_ablation_a
        print('[P5.15 ablation A] Candidate 4 reverted (day_balance=False on TSO and all DSOs); '
              'recovery G1-equivalent (tier2 off, node 5 ineligible)')
        run_admm_arm('ablation_a', OUT_ABL_A, k_override=None, eval_id=ABL_A_EVAL_ID)
    elif gate == 'ablation_b':
        _require_fresh_output_root(OUT_ABL_B)
        _original_fresh_planning_b = O.fresh_planning

        def _fresh_planning_ablation_b(eval_id):
            return _configure_ablation_b(_original_fresh_planning_b(eval_id))

        O.fresh_planning = _fresh_planning_ablation_b
        print('[P5.15 ablation B] Candidate 1 re-wired (sess_phi_limit_lower/upper restored); '
              'Candidate 4 as in G1; recovery G1-equivalent (tier2 off, node 5 ineligible)')
        run_admm_arm('ablation_b', OUT_ABL_B, k_override=None, eval_id=ABL_B_EVAL_ID)
    elif gate == 'ablation_c':
        _require_fresh_output_root(OUT_ABL_C)
        _original_fresh_planning_c = O.fresh_planning

        def _fresh_planning_ablation_c(eval_id):
            return _configure_ablation_c(_original_fresh_planning_c(eval_id))

        O.fresh_planning = _fresh_planning_ablation_c
        print(f'[P5.15 ablation C] EPS_ESSO_THROUGHPUT = {ABL_C_EPS:g} (G1: 1e-3); '
              'recovery G1-equivalent (tier2 off, node 5 ineligible)')
        run_admm_arm('ablation_c', OUT_ABL_C, k_override=None, eval_id=ABL_C_EVAL_ID)
    elif gate == 'g2r':
        _require_fresh_output_root(OUT_G2R)
        _original_fresh_planning_g2r = O.fresh_planning

        def _fresh_planning_g2r(eval_id):
            planning = _original_fresh_planning_g2r(eval_id)
            RH.set_recovery_policy(planning, enabled=True, tier2_enabled=True)
            return planning

        O.fresh_planning = _fresh_planning_g2r
        print('[P5.15 G2 re-run] k=10000; recovery policy: all networks and ESSO enabled '
              '(case33_1 included), tier 2 on; ESSO slack values captured')
        run_admm_arm('k10000_r', OUT_G2R, k_override=10000.0, eval_id=G2R_EVAL_ID)
    elif gate == 'g1b':
        _require_fresh_output_root(OUT_G1B)
        print('[P5.15 G1 re-run, new baseline] production defaults: '
              f'EPS_ESSO_THROUGHPUT={SED.EPS_ESSO_THROUGHPUT:g}, ESSO overrides={SED.ESSO_TOL_OVERRIDES}; '
              'recovery policy production default (all enabled, tier 2 on)')
        run_admm_arm('baseline', OUT_G1B, k_override=None, eval_id='p515g1b_baseline')
    elif gate == 's30':
        _require_fresh_output_root(OUT_S30)
        print('[P5.15 Step 3.0 determinism] repeat of the new-baseline G1, production defaults: '
              f'EPS_ESSO_THROUGHPUT={SED.EPS_ESSO_THROUGHPUT:g}, ESSO overrides={SED.ESSO_TOL_OVERRIDES}; '
              'recovery production default (all enabled, tier 2 on)')
        run_admm_arm('baseline_rep', OUT_S30, k_override=None, eval_id='p515s30_baseline_rep')
    elif gate == 'g3_full_b':
        _require_fresh_output_root(OUT_G3F_B)
        probe_eval_id_b = 'p515g3fb_probe'
        if os.path.exists(os.path.join(O.WORK_DIR, probe_eval_id_b)):
            raise RuntimeError(f'refusing to start: probe eval dir already exists: {probe_eval_id_b}')
        probe_b = O.fresh_planning(probe_eval_id_b)
        active_nodes_b = list(probe_b.shared_ess_data.active_distribution_network_nodes)
        del probe_b
        investment_map_b = {nid: (0.0, 0.0) for nid in active_nodes_b}
        if 7 not in investment_map_b:
            raise RuntimeError(f'node 7 not in active_distribution_network_nodes={active_nodes_b}')
        investment_map_b[7] = (1.62, 3.24)
        print('[P5.15 G3-full re-run, new baseline] production defaults: '
              f'EPS_ESSO_THROUGHPUT={SED.EPS_ESSO_THROUGHPUT:g}, ESSO overrides={SED.ESSO_TOL_OVERRIDES}; '
              'recovery default (tier 2 on); node 7 at 1.62 MVA / 3.24 MWh, others zero')
        run_admm_arm('g3_full_node7_b', OUT_G3F_B, k_override=None, investment_map=investment_map_b,
                     eval_id='p515g3fb_node7')
    elif gate == 'g1':
        _require_fresh_output_root(OUT_G1)
        run_admm_arm('control', OUT_G1, k_override=None, eval_id='p515g1_control')
    elif gate == 'g2':
        # G2PREP Fix 2: own fresh root and fresh eval id, like g1.
        _require_fresh_output_root(OUT_G2)
        run_admm_arm('k10000', OUT_G2, k_override=10000.0, eval_id='p515g2_k10000')
    elif gate == 'g4b':
        # G2PREP Fix 2: own fresh root and fresh eval id, like g1.
        _require_fresh_output_root(OUT_G4)
        run_admm_arm('control_rep2', OUT_G4, k_override=None, eval_id='p515g4_control_rep2')
    elif gate == 'g3_init':
        for s in (1.00, 1.25, 1.62):
            run_ladder_init(s, os.path.join(OUT, 'ladder'))
    elif gate == 'g3_full':
        # G2PREP Fix 2: own fresh root, plus a distinct, fresh probe eval id (the
        # probe below only reads active_distribution_network_nodes -- no solve, no
        # candidate, no investment -- but still MUST NOT reuse an eval id that
        # already has a logs dir under O.WORK_DIR, since network IPOPT logs append).
        _require_fresh_output_root(OUT_G3F)
        probe_eval_id = 'p515g3f_probe_r2'
        probe_eval_dir = os.path.join(O.WORK_DIR, probe_eval_id)
        if os.path.exists(probe_eval_dir):
            raise RuntimeError(
                f'refusing to start: probe eval dir already exists (network logs '
                f'append): {probe_eval_dir}')
        # node 7 only, others zero, per PLANNER task text.
        # discovered lazily: build investment_map after loading a fresh planning to read
        # the active node list, without solving anything.
        probe = O.fresh_planning(probe_eval_id)
        active_nodes = list(probe.shared_ess_data.active_distribution_network_nodes)
        del probe
        investment_map = {nid: (0.0, 0.0) for nid in active_nodes}
        if 7 not in investment_map:
            raise RuntimeError(f'node 7 not in active_distribution_network_nodes={active_nodes}')
        investment_map[7] = (1.62, 3.24)
        run_admm_arm('g3_full_node7', OUT_G3F, k_override=None, investment_map=investment_map,
                     eval_id='p515g3f_node7_r2')
    elif gate == 's31':
        # S31 worker task (PLANNER_BRIEF_2026-09-13.md Addendum 10 / Sequence
        # after signature): the signed-table baseline campaign. Production
        # defaults, no overrides -- own fresh root and eval id.
        _require_fresh_output_root(OUT_S31)
        preflight_eval_id = 'p515s31_preflight_capture_check'
        preflight_eval_dir = os.path.join(O.WORK_DIR, preflight_eval_id)
        if os.path.exists(preflight_eval_dir):
            raise RuntimeError(
                f'refusing to start: preflight eval dir already exists (network '
                f'logs append): {preflight_eval_dir}')
        preflight_planning = O.fresh_planning(preflight_eval_id)
        s31_checklist = assert_s31_capture_paths(preflight_planning)
        del preflight_planning
        print(f'[P5.15 S31] capture-path pre-flight passed: {s31_checklist}')
        print('[P5.15 S31] production defaults: '
              f'EPS_ESSO_THROUGHPUT={SED.EPS_ESSO_THROUGHPUT:g}, ESSO overrides={SED.ESSO_TOL_OVERRIDES}; '
              'recovery policy production default (all enabled, tier 2 on); '
              'signed-table Step 3.1 penalty changes in force')

        def _s31_hook(planning, sed, models, rows, report, out_dir, label):
            write_component_levels_terminal(planning, sed, models, rows, report, out_dir, label)

        run_admm_arm('baseline', OUT_S31, k_override=None, eval_id='p515s31_baseline',
                     post_run_hook=_s31_hook)
    elif gate == 's31c':
        # S31C worker task (PLANNER_BRIEF_2026-09-13.md Addendum 12): the
        # post-signature interface energy settlement / signed-delta baseline
        # campaign. Production defaults, no overrides -- own fresh root and
        # eval id.
        _require_fresh_output_root(OUT_S31C)
        preflight_eval_id = 'p515s31c_preflight_capture_check'
        preflight_eval_dir = os.path.join(O.WORK_DIR, preflight_eval_id)
        if os.path.exists(preflight_eval_dir):
            raise RuntimeError(
                f'refusing to start: preflight eval dir already exists (network '
                f'logs append): {preflight_eval_dir}')
        preflight_planning = O.fresh_planning(preflight_eval_id)
        s31c_checklist = assert_s31c_capture_paths(preflight_planning)
        del preflight_planning
        print(f'[P5.15 S31C] capture-path pre-flight passed: {s31c_checklist}')
        print('[P5.15 S31C] production defaults, no overrides; signed interface '
              'reparametrization + interface energy settlement in force (Addendum 12)')

        def _s31c_hook(planning, sed, models, rows, report, out_dir, label):
            write_interface_settlement_detail_s31c(planning, sed, models, rows, report, out_dir, label)

        run_admm_arm('baseline', OUT_S31C, k_override=None, eval_id='p515s31c_baseline',
                     post_run_hook=_s31c_hook)
    elif gate == 's32':
        # S32 worker task (PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2,
        # 3.3(a); Addendum 13; frozen spec v2
        # data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json,
        # supersedes v1 frozen_s32_spec_v1_14a18674.json -- balancing dual
        # ratio now s_rho_part/eps_dual, stopping rule unchanged): Boyd
        # stopping rule + residual balancing gate, cap 150, case-file rho in
        # force (N.RHO NOT applied) -- own fresh root and eval id.
        if N.REL != S32_REL:
            raise RuntimeError(
                f'p514_n_instrumented_cstar.REL ({N.REL}) no longer matches the '
                f'frozen s32 objective-change (diagnostic) tolerance {S32_REL}; '
                'the spec requires 1e-4.')
        observed_spec_hash = _s32_spec_hash()
        if observed_spec_hash != S32_SPEC_SHA256:
            raise RuntimeError(
                f'frozen s32 spec hash mismatch: file={observed_spec_hash} '
                f'expected={S32_SPEC_SHA256}')
        _require_fresh_output_root(OUT_S32)
        preflight_eval_id = 'p515s32_preflight_capture_check'
        preflight_eval_dir = os.path.join(O.WORK_DIR, preflight_eval_id)
        if os.path.exists(preflight_eval_dir):
            raise RuntimeError(
                f'refusing to start: preflight eval dir already exists (network '
                f'logs append): {preflight_eval_dir}')
        preflight_planning = O.fresh_planning(preflight_eval_id)
        s32_checklist = assert_s32_capture_paths(preflight_planning)
        preflight_admm_params = preflight_planning.params.admm
        del preflight_planning
        print(f'[P5.15 S32] capture-path pre-flight passed: {s32_checklist}')
        print(
            '[P5.15 S32] cap=150, objective rel=1e-4 (diagnostic), adaptive on, '
            f'case-file rho in force (N.RHO NOT applied): '
            f'v={preflight_admm_params.rho["v"]}, pf={preflight_admm_params.rho["pf"]}, '
            f'ess={preflight_admm_params.rho["ess"]}; '
            f'boyd eps_source={preflight_admm_params.boyd_eps_source}, '
            f'eps_abs={preflight_admm_params.tol["boyd"]["eps_abs"]:.1e}, '
            f'eps_rel={preflight_admm_params.tol["boyd"]["eps_rel"]:.1e}'
        )

        def _s32_hook(planning, sed, models, rows, report, out_dir, label):
            write_boyd_terminal_s32(planning, sed, models, rows, report, out_dir, label)

        run_admm_arm('baseline', OUT_S32, k_override=None, eval_id='p515s32_baseline',
                     num_max_iters_override=S32_CAP, apply_rho=False,
                     full_diagnostics_in_rows=True, post_run_hook=_s32_hook)
    elif gate == 's33e2':
        # S33 E2 worker task (PLANNER_BRIEF_2026-09-13.md Addendum 14; frozen
        # spec v3 data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json,
        # supersedes v2 frozen_s32_spec_v2_516bd749.json -- gamma tied to rho
        # (tau=1), rho/gamma adaptation frozen after cycle 30,
        # minimum_consecutive_converged_cycles=3): Boyd stopping rule +
        # residual balancing + tied-gamma stabiliser gate, cap 150,
        # case-file rho in force (N.RHO NOT applied) -- own fresh root and
        # eval id; the s32 run directory is never written.
        if N.REL != S33E2_REL:
            raise RuntimeError(
                f'p514_n_instrumented_cstar.REL ({N.REL}) no longer matches the '
                f'frozen s33e2 objective-change (diagnostic) tolerance {S33E2_REL}; '
                'the spec requires 1e-4.')
        observed_spec_hash = _s33e2_spec_hash()
        if observed_spec_hash != S33E2_SPEC_SHA256:
            raise RuntimeError(
                f'frozen s33e2 spec hash mismatch: file={observed_spec_hash} '
                f'expected={S33E2_SPEC_SHA256}')
        _require_fresh_output_root(OUT_S33E2)
        preflight_eval_id = 'p515s33e2_preflight_capture_check'
        preflight_eval_dir = os.path.join(O.WORK_DIR, preflight_eval_id)
        if os.path.exists(preflight_eval_dir):
            raise RuntimeError(
                f'refusing to start: preflight eval dir already exists (network '
                f'logs append): {preflight_eval_dir}')
        preflight_planning = O.fresh_planning(preflight_eval_id)
        s33e2_checklist = assert_s33e2_capture_paths(preflight_planning)
        preflight_admm_params = preflight_planning.params.admm
        del preflight_planning
        print(f'[P5.15 S33 E2] capture-path pre-flight passed: {s33e2_checklist}')
        print(
            '[P5.15 S33 E2] cap=150, objective rel=1e-4 (diagnostic), adaptive on, '
            f'case-file rho in force (N.RHO NOT applied): '
            f'v={preflight_admm_params.rho["v"]}, pf={preflight_admm_params.rho["pf"]}, '
            f'ess={preflight_admm_params.rho["ess"]}; '
            f'boyd eps_source={preflight_admm_params.boyd_eps_source}, '
            f'eps_abs={preflight_admm_params.tol["boyd"]["eps_abs"]:.1e}, '
            f'eps_rel={preflight_admm_params.tol["boyd"]["eps_rel"]:.1e}; '
            f'gamma_policy={preflight_admm_params.proximal_regularization["tso"]["gamma_policy"]}, '
            f'tau={preflight_admm_params.proximal_regularization["tso"]["tau"]}; '
            f'freeze_after_cycle={preflight_admm_params.penalty_update["freeze_after_cycle"]}; '
            f'minimum_consecutive_converged_cycles={preflight_admm_params.minimum_consecutive_converged_cycles}'
        )

        def _s33e2_hook(planning, sed, models, rows, report, out_dir, label):
            write_boyd_terminal_s33e2(planning, sed, models, rows, report, out_dir, label)

        run_admm_arm('baseline', OUT_S33E2, k_override=None, eval_id='p515s33e2_baseline',
                     num_max_iters_override=S33E2_CAP, apply_rho=False,
                     full_diagnostics_in_rows=True, post_run_hook=_s33e2_hook)
    elif gate == 's34':
        # S34 worker task (PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5;
        # frozen spec v4 data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json,
        # supersedes v3 frozen_s33_e2_spec_v3_825f1f02.json -- D5 ESSO AL
        # scaling, fixed sigma, the v4 rho/freeze policy, S_ref = 2.5 MVA):
        # Boyd stopping rule + residual balancing + tied-gamma stabiliser +
        # D5/sigma/S_ref/v4-freeze gate, cap 150, case-file rho in force
        # (N.RHO NOT applied) -- own fresh root and eval id; the s32/s33e2
        # run directories are never written.
        if N.REL != S34_REL:
            raise RuntimeError(
                f'p514_n_instrumented_cstar.REL ({N.REL}) no longer matches the '
                f'frozen s34 objective-change (diagnostic) tolerance {S34_REL}; '
                'the spec requires 1e-4.')
        observed_spec_hash = _s34_spec_hash()
        if observed_spec_hash != S34_SPEC_SHA256:
            raise RuntimeError(
                f'frozen s34 spec hash mismatch: file={observed_spec_hash} '
                f'expected={S34_SPEC_SHA256}')
        _require_fresh_output_root(OUT_S34)
        preflight_eval_id = 'p515s34_preflight_capture_check'
        preflight_eval_dir = os.path.join(O.WORK_DIR, preflight_eval_id)
        if os.path.exists(preflight_eval_dir):
            raise RuntimeError(
                f'refusing to start: preflight eval dir already exists (network '
                f'logs append): {preflight_eval_dir}')
        preflight_planning = O.fresh_planning(preflight_eval_id)
        s34_checklist = assert_s34_capture_paths(preflight_planning)
        preflight_admm_params = preflight_planning.params.admm
        del preflight_planning
        print(f'[P5.15 S34] capture-path pre-flight passed: {s34_checklist}')
        print(
            '[P5.15 S34] cap=150, objective rel=1e-4 (diagnostic), adaptive on, '
            f'case-file rho in force (N.RHO NOT applied): '
            f'v={preflight_admm_params.rho["v"]}, pf={preflight_admm_params.rho["pf"]}, '
            f'ess={preflight_admm_params.rho["ess"]}; '
            f'boyd eps_source={preflight_admm_params.boyd_eps_source}, '
            f'eps_abs={preflight_admm_params.tol["boyd"]["eps_abs"]:.1e}, '
            f'eps_rel={preflight_admm_params.tol["boyd"]["eps_rel"]:.1e}; '
            f'gamma_policy={preflight_admm_params.proximal_regularization["tso"]["gamma_policy"]}, '
            f'tau={preflight_admm_params.proximal_regularization["tso"]["tau"]}; '
            f'objective_scale={preflight_admm_params.objective_scale} '
            f'(source={preflight_admm_params.objective_scale_source}, '
            f'assert_factor={preflight_admm_params.objective_scale_assert_factor}); '
            f'esso_al_scale={preflight_admm_params.esso_al_scale} '
            f'(numeric value and >1 check deferred to the first cycle -- unresolved '
            f'before any solve); '
            f'shared_ess_reference_rating_mva={preflight_admm_params.shared_ess_reference_rating_mva}; '
            f'freeze_after_unchanged_cycles={preflight_admm_params.penalty_update["freeze_after_unchanged_cycles"]}; '
            f'freeze_backstop_cycle={preflight_admm_params.penalty_update["freeze_backstop_cycle"]}; '
            f'minimum_consecutive_converged_cycles={preflight_admm_params.minimum_consecutive_converged_cycles}'
        )

        s34_recourse_jump_path = os.path.join(OUT_S34, 'recourse_jump_sidecar_baseline.jsonl')
        s34_ess_stride_path = os.path.join(OUT_S34, 'ess_entry_stride_baseline.jsonl')
        _refuse_overwrite(s34_recourse_jump_path)
        _refuse_overwrite(s34_ess_stride_path)

        def _s34_hook(planning, sed, models, rows, report, out_dir, label):
            report['s34_recourse_jump_sidecar_path'] = os.path.relpath(s34_recourse_jump_path, REPO)
            report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(s34_ess_stride_path, REPO)
            write_boyd_terminal_s34(planning, sed, models, rows, report, out_dir, label)

        with s34_capture_hooks(s34_recourse_jump_path, s34_ess_stride_path, stride=1):
            run_admm_arm('baseline', OUT_S34, k_override=None, eval_id='p515s34_baseline',
                         num_max_iters_override=S34_CAP, apply_rho=False,
                         full_diagnostics_in_rows=True, post_run_hook=_s34_hook)
    else:
        print(__doc__)
        sys.exit(1)
