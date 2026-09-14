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
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G')
os.makedirs(OUT, exist_ok=True)

# Addendum 6 item 2: the g1 CLI gate's OWN fresh output root -- never shared with OUT,
# which holds residue from earlier, killed campaigns (see .p515_g_gate.lock).
OUT_G1 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G1')

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

        m = _NET_RECOVER_OK_RE.match(line)
        if m:
            name, year, day = _parse_ctx(m.group('ctx'))
            key = (name, year, day)
            ev = open_by_key.get(key)
            if ev is None:
                ev = _new_network_event(name, year, day, current_cycle, name_to_agent)
                ev['note'] = ('recovery-success print with no matching open '
                              'primary-solve event (malformed capture)')
                open_by_key[key] = ev
            ev['termination'] = 'recovered'
            ev['class'] = 'recovered'
            events.append(ev)
            continue

        m = _NET_LOG_RE.match(line)
        if m:
            name, year, day = _parse_ctx(m.group('ctx'))
            ev = open_by_key.get((name, year, day))
            if ev is not None:
                if 'recovery' in m.group('label'):
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
    tmp = f"{hook_state['network_failures_path']}.tmp{os.getpid()}"
    with open(tmp, 'w') as handle:
        for record in blocks:
            handle.write(json.dumps(record, default=str) + '\n')
        for event in esso_events:
            tagged = dict(event)
            tagged['record_type'] = 'esso_recovery'
            handle.write(json.dumps(tagged, default=str) + '\n')
        for record in frozen:
            handle.write(json.dumps(record, default=str) + '\n')
    os.replace(tmp, hook_state['network_failures_path'])
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
                             num_max_iters_override=None):
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
                  num_max_iters_override=None, eval_id=None):
    """One full cold ADMM arm through the production path, reusing p514_n's own
    module-level constants and capture helpers verbatim. `investment_map`, if given,
    overrides the uniform S_INV/E_INV assignment for specific node_ids (others left at
    N.S_INV/N.E_INV for the control/perturbation arms, or at the harness's default 0/0
    if this is a fresh candidate); pass a full dict {node_id: (s, e)} covering every
    active node to avoid ambiguity (this is what G3's full eval does).

    `num_max_iters_override` is a SMOKE-TEST-ONLY parameter (Addendum 6 smoke test):
    the `g1` CLI arm never passes it, so `N.CAP` (90) remains the C* control cap for the
    real campaign.
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
                num_max_iters_override=num_max_iters_override)

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
    report['network_failures_summary'] = {
        'n_blocks': len(final_blocks),
        'classes': {c: sum(1 for b in final_blocks if b['class'] == c)
                    for c in ('recovered', 'unrecovered', 'not_attempted', 'indeterminate')},
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


if __name__ == '__main__':
    _acquire_exclusive_run_lock()
    gate = sys.argv[1] if len(sys.argv) > 1 else None
    if gate == 'g1':
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
    else:
        print(__doc__)
        sys.exit(1)
