"""
P5.15 Addendum 23 item (2) -- Step 3.5 restated: the interval-hull polish gate.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 23 ("Step 3.5 restated -- interval-hull
polish") and frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`, item `item2_hull_polish`.
Prerequisite reading (Addendum 23 item 1) is `WORKER_REPORT_S41_POLISH_PREREQ.md`.

================================================================================
WHAT THIS SCRIPT DOES, PER THE SPEC
================================================================================
  1. Runs the case-file configuration to Boyd certification IN-PROCESS (same
     `p515_g_g1_g4_admm_gates.run_admm_arm(label='s39_D', apply_rho=False,
     num_max_iters_override=300)` call `p515_s40_polish_gap.py` uses) and
     requires the run's report to reproduce D's own committed report
     BITWISE, cycle 139, cost 650966975.2943751 -- reusing
     `p515_s40_polish_gap._reproduction_check` BY IMPORT, unchanged, with the
     `rule_eleven_checklist` subtree reported but non-gating (per spec).
  2. At the certified cycle, for every coupling entry, computes the closed
     interval [min, max] of the agents' OWN achieved values: V and PF over
     the TSO and DSO copies (2 agents); shared-ESS P/Q over the TSO, DSO and
     ESSO copies (3 agents). A degenerate interval (equal values) becomes a
     fixed value.
  3. Constrains the corresponding Pyomo quantity, IN EACH BLOCK, to that
     interval (never fixes at a midpoint) -- see `apply_hull_bounds` below
     for exactly which Pyomo Var/Expression carries each family, with
     file:line citations, and for the ONE deliberate departure from
     `p56a_oracle.apply_common_values`'s own mechanism for interface P/Q
     (a real defect found while building this harness -- see "THE
     INTERFACE_DELTA_P FINDING" below).
  4. Deactivates the ADMM objective and the p58 rescaled objective, activates
     the unscaled base objective (`p515_s40_polish_gap._switch_to_base_objective`,
     BY IMPORT, unchanged -- the same defect-fix that stage already made).
  5. Re-solves each of the 48 network blocks: primal warm start from the
     certified block solution (automatic -- these are the SAME live Pyomo
     model objects the ADMM run left them in; nothing here resets a Var's
     `.value`), IPOPT's own COMPILED DEFAULT bound push (an explicit
     departure from the case file's tighter production push -- see "BOUND
     PUSH" below), NO multiplier import, production recovery policy
     (tier-1 cold retry, tier-2 cold+adaptive -- inherited unchanged from
     `network._run_smopf`, which `network.run_smopf` calls).
  6. Requires all 48 blocks to solve; otherwise the gate is NOT evaluated
     (same convention as `p515_s40_polish_gap.py`).
  7. Gate: `|Delta| / certified_cost < 0.1%`, `Delta = sum_i [f_i(polished) -
     f_i(certified)]` on the weighted base objective, `gross_operational_cost`
     convention. Reports total Delta, TSO/DSO split, max |Delta_i| and its
     block, the count of hull bounds active at the polished points (within
     1e-9 relative of an interval end) per channel, the per-block table, and
     flagged blocks (`Delta_i > 1e-6 * certified_cost`).

================================================================================
THE interface_delta_p FINDING (a real defect avoided, not fixed, in this harness)
================================================================================
`p56a_oracle.apply_common_values` (the v11 exact-fix harness's own mechanism,
`p515_s40_polish_gap.py`'s only source of "how to fix a coupling entry") pins
the TSO's total interface power via
`_interface_expression(t_model, adn_load, p, 'p') == common_p`
(`p56a_oracle.py:440-441`), where `_interface_expression` (`p56a_oracle.py:390-399`)
computes `t_model.pc[adn_load, 0, 0, p] + flex_p_up[...] - flex_p_down[...]`.

Reading `create_transmission_network_model`
(`shared_resources_planning.py:3580-3607`, "P5.15 Step 3.1-C / Addendum 12
item 2") shows that in the CURRENT ADMM path, `pc` is FIXED once at
construction to the DSO's cycle-0 consensus interface power (the anchor;
confirmed independently in `WORKER_REPORT_S36_CLONE_CAPTURE.md:156-165`), and
`flex_p_up`/`flex_p_down`/`flex_q_up`/`flex_q_down` are ALSO fixed at 0.00 --
interface flexibility is instead carried ENTIRELY by `interface_delta_p`/
`interface_delta_q` (freed, bounded +/- the interface rating). The model's OWN
`pc_adn` Expression (`network.py:434`, `interface_pf_p_transmission_def`,
`model_construction_helpers.py:1167-1185`) correctly adds
`m.interface_delta_p[dn, s_m, s_o, p]`; `p56a_oracle._interface_expression`
does NOT -- it was written before Addendum 12's reparametrization and was
never updated. So in the ADMM-certified model,
`_interface_expression(...) == pc(FIXED) + 0 - 0 == pc`, a CONSTANT, while the
model's real achieved interface power is `pc + interface_delta_p(achieved)` --
a DIFFERENT constant whenever `interface_delta_p` is materially nonzero at
the certified point, which it generically is (it is the sole carrier of
interface deviation). `apply_common_values`'s row therefore equates two
constants that generically differ: a ZERO-GRADIENT, unconditionally
infeasible row, independent of any active rating or voltage bound -- no
free variable in the model appears in it at all.

This is consistent with, and very likely explains, the "frozen from
iteration 0" signature `WORKER_REPORT_S41_POLISH_PREREQ.md` Part 1 found on
ALL SIX TSO blocks read there (constraint violation identical to 16
significant figures from iteration 0 to termination, across warm, cold and
cold+adaptive attempts) -- and plausibly the other six TSO failures the v11
exact-fix run reported (not independently re-read here; the v11 run is not
re-run, per instruction). **This finding is about `p56a_oracle.py`, not
about `p515_s40_polish_gap.py`** (whose own two checked defects -- starting
point, consensus re-fix -- were both absent, `WORKER_REPORT_S41_POLISH_PREREQ.md`
Part 1 SS2-3) and `p56a_oracle.py` is not in this task's forbidden-file list,
but it IS shared code several other committed P5.15 stages depend on, so it
is NOT edited here (out of this task's authorized scope; reported to the
Planner instead, per CLAUDE.md).

**What this harness does instead**: `apply_hull_bounds` below bounds the
model's own `pc_adn`/`qc_adn` (TSO, `network.py:434-435`) and `pg_adn`/
`qg_adn` (DSO, `network.py:439-440`) Expression components DIRECTLY -- the
SAME quantities `common_coordinated_values` already reads via `pe.value(...)`
to obtain `tso_p`/`dso_p`/`tso_q`/`dso_q` in the first place -- via a new
`ConstraintList` of range rows (or an equality row when the interval is
degenerate). This is both the more literal reading of "constrain the
corresponding variable ... to that interval" (spec wording) AND avoids the
interface_delta_p defect entirely, since `pc_adn`/`qc_adn`/`pg_adn`/`qg_adn`
correctly include every term of the model's real interface expression.
`common_coordinated_values` itself (`p56a_oracle.py:233-311`) is reused BY
IMPORT, unchanged -- it only READS `pe.value(...)`, so it is unaffected by
the `_interface_expression` defect.

================================================================================
BOUND PUSH -- "IPOPT default", not production's
================================================================================
`network._create_smopf_solver` applies the case file's `solver_params.options`
(`bound_push`/`bound_frac`/`slack_bound_push`/`slack_bound_frac` = `1e-6` for
case9, `data/SRP1/case9/case9_params.json:42-45`) UNCONDITIONALLY, before the
`from_warm_start` gate (`network.py:555-556`) -- so simply calling
`network.run_smopf(model, holder.params, from_warm_start=False)`, as
`p515_s40_polish_gap.py` does, still uses production's tight push, not
IPOPT's own compiled default (`WORKER_REPORT_S41_POLISH_PREREQ.md` SS2). This
harness explicitly overrides those four options to IPOPT 3.14's documented
compiled default (`0.01` each) on a per-solve-scoped copy of
`holder.params.solver_params.options` (saved and restored around each
polish solve; recovery/tier-2 attempts inherit the override too, since they
build on top of `.options` the same way production's own recovery already
does -- `network.py:557-558, 772-780`).

================================================================================
EXACT FULL-RUN LAUNCH COMMAND (Planner launches; Worker does NOT)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s41_hull_polish.py \\
        > data/SRP1/Results/P515S41/hull_polish_launch.log 2>&1

Preconditions identical in kind to `p515_s40_polish_gap.py` (lock absent, no
forbidden live process, fresh output root, production files clean in git,
D's committed reference report present) -- checked before anything is
written, refuses loudly otherwise. Run attached, alone, both streams
captured via the shell redirection above; never `screen`/`nohup`/`&`.

================================================================================
SMOKE TEST (the Worker runs THIS one, `--smoke-cycles 2`)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s41_hull_polish.py --smoke-cycles 2 \\
        > data/SRP1/Results/P515S41/hull_polish_smoke_launch.log 2>&1
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p56a_oracle as O  # noqa: E402 -- common_coordinated_values, BY IMPORT, unchanged
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator/precondition conventions, BY IMPORT
import network as NET  # noqa: E402 -- _clear_multiplier_suffixes, BY IMPORT
import shared_energy_storage_data as SED  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
# Do NOT modify p515_s40_polish_gap.py (its committed evidence stands) --
# reuse its reproduction-check machinery and constants BY IMPORT, per the
# spec ("reuse its `_reproduction_check`, with the rule_eleven_checklist
# subtree non-gating").
from p515_s40_polish_gap import (  # noqa: E402
    ARM_LABEL, FULL_NUM_MAX_ITERS, D_REFERENCE_PATH, D_CERTIFICATION_CYCLE,
    D_CERTIFIED_COST, _reproduction_check, _build_floor_rows, _refuse_overwrite,
    _switch_to_base_objective,
)

OUT_DIR_FULL = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S41', 'hull_polish')
OUT_DIR_SMOKE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S41', 'hull_polish_smoke')

# Polish solves reach the solver through the SAME call site every other
# network solve in this campaign uses (network.py:_run_smopf_solver_attempt).
POLISH_PERMITTED = [('network.py', '_run_smopf_solver_attempt')]

GATE_THRESHOLD_PCT = 0.1
FLAG_ABS = 1e-6 * D_CERTIFIED_COST  # == 650.966975... ("= 650.97" per spec v12)
HULL_ACTIVE_REL_TOL = 1e-9

# IPOPT 3.14's own compiled defaults (NOT the case file's 1e-6 production
# push) -- see module docstring, "BOUND PUSH".
IPOPT_DEFAULT_BOUND_OPTIONS = {
    'bound_push': 0.01, 'bound_frac': 0.01,
    'slack_bound_push': 0.01, 'slack_bound_frac': 0.01,
}

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + (
    'p515_s41_',)


def _check_preconditions(out_dir):
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True,
                                   check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(CP.PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    if not os.path.isfile(D_REFERENCE_PATH):
        failures.append(f"D's committed reference report missing: {D_REFERENCE_PATH}")

    return failures


# ==============================================================================
#  hull construction and application
# ==============================================================================
def _interval(*values):
    lo, hi = min(values), max(values)
    return lo, hi, (lo == hi)


def _get_or_add_constraint_list(model, name):
    comp = getattr(model, name, None)
    if comp is None:
        comp = pe.ConstraintList()
        model.add_component(name, comp)
    return comp


def _bound_var(var_data, lo, hi, degenerate):
    """Bound (or fix) a genuine Pyomo Var to the hull interval. Explicitly
    unfixes first in the non-degenerate branch: found while running
    `p515_s41_hull_polish_checks.py` -- a shared-ESS index whose installed
    capacity is 0 at model-construction time (`configure_shared_ess_
    operational_state`) is left `.fix()`-ed at 0 with bounds (0, 0); calling
    only `setlb`/`setub` on an already-fixed Var leaves it fixed (IPOPT/the
    NL writer treat `.fixed` Vars as constants regardless of their bounds),
    silently defeating the interval. At the certified C* point every active
    shared-ESS index has nonzero installed capacity and so is genuinely
    unfixed by then, but this guard is defensive and correct regardless of
    that circumstance."""
    if degenerate:
        var_data.fix(lo)
    else:
        if var_data.fixed:
            var_data.unfix()
        var_data.setlb(lo)
        var_data.setub(hi)


def _bound_expr(rows, expr, lo, hi, degenerate):
    if degenerate:
        rows.add(expr == lo)
    else:
        rows.add((lo, expr, hi))


def hull_entries_with_esso(planning, models, consensus_vars):
    """`p56a_oracle.common_coordinated_values` (BY IMPORT, unchanged) gives the
    TSO/DSO achieved values per (node, year, day, period). This adds the THIRD
    ESS endpoint -- the ESSO's OWN achieved dispatch, read from the ALREADY-
    SOLVED `models['esso'][node]` (the same dict `run_operational_planning`
    returned; the ESSO is NOT re-solved here or anywhere in this harness --
    per spec, "the ESSO is not among the 48 network blocks", its copy enters
    ONLY as an interval endpoint). `es_pnet`/`es_qnet`
    (`shared_energy_storage_data.py:439-440`) are in MW; divided by the TSO
    network's `baseMVA` to match `common_sess_p`/`common_sess_q`'s own
    per-unit convention (`p56a_oracle.py:305-306`, which does the same
    conversion for the ADMM consensus `z`)."""
    common = O.common_coordinated_values(planning, models, consensus_vars)
    tso = planning.transmission_network
    esso_models = models['esso']
    # IMPORTANT indexing note (found while building this harness, caught by
    # p515_s41_hull_polish_checks.py): the ESSO subproblem's es_pnet/es_qnet
    # are indexed by POSITIONAL integers into shared_ess_data.years/.days
    # (range(len(years))/range(len(days)), shared_energy_storage_data.py:36-37
    # and the model's own years/days Sets), NOT by the (year, day) LABELS the
    # TSO/DSO network models use -- confirmed against production's own
    # mapping convention at shared_resources_planning.py:4264-4280
    # (`for y in esso_m.years: year = years[y]; ... esso_m.es_pnet[y, d, p]`,
    # `years = list(shared_ess_data.years)`). Mapped here before reading.
    esso_years = list(planning.shared_ess_data.years)
    esso_days = list(planning.shared_ess_data.days)
    out = {}
    for (node, year, day, p), entry in common.items():
        t_net = tso.network[year][day]
        esso_model = esso_models[node]
        y_idx = esso_years.index(year)
        d_idx = esso_days.index(day)
        esso_p_mw = float(pe.value(esso_model.es_pnet[y_idx, d_idx, p]))
        esso_q_mw = float(pe.value(esso_model.es_qnet[y_idx, d_idx, p]))
        merged = dict(entry)
        merged['esso_sess_p'] = esso_p_mw / t_net.baseMVA
        merged['esso_sess_q'] = esso_q_mw / t_net.baseMVA
        merged['esso_sess_p_mw'] = esso_p_mw
        merged['esso_sess_q_mw'] = esso_q_mw
        out[(node, year, day, p)] = merged
    return out


def apply_hull_bounds(planning, models, hull):
    """Constrain every coupling entry, entrywise, to the closed interval of
    the agents' achieved values at the certified cycle. Returns a list of
    descriptors (one per constrained TSO/DSO-side quantity -- the ESSO
    endpoint is NOT itself constrained, it only supplied one interval bound)
    used afterward to count how many hull bounds are active at the polished
    point.

    Voltage (2 agents, TSO+DSO) -- `vmag_sqr` (network.py:285), a genuine
    `pe.Var`: bounded via `.setlb()`/`.setub()` (or `.fix()` when the
    interval is degenerate) directly on the SAME index
    `common_coordinated_values` reads (`t_model.vmag_sqr[adn_idx,0,0,p]`,
    `d_model.vmag_sqr[ref_idx,0,0,p]`). The interval is built on V (the
    achieved `sqrt(vmag_sqr)`, matching `tso_v`/`dso_v`) and squared before
    bounding, since squaring is monotonic on `V >= 0` -- bounding `vmag_sqr`
    to `[min(v)**2, max(v)**2]` is EXACTLY the feasible set `V in
    [min(v), max(v)]`.

    Interface P/Q (2 agents, TSO+DSO) -- `pc_adn`/`qc_adn` (TSO,
    `network.py:434-435`) and `pg_adn`/`qg_adn` (DSO, `network.py:439-440`),
    Pyomo `pe.Expression` components, NOT Vars (module docstring, "THE
    interface_delta_p FINDING" -- this is the one deliberate departure from
    `p56a_oracle.apply_common_values`'s own mechanism). Bounded via a new
    `ConstraintList` range row `(lo, expr, hi)` (or an equality row when
    degenerate), added once per (TSO model / DSO model) and reused across
    periods.

    Shared-ESS P/Q (3 agents, TSO+DSO+ESSO) -- `shared_es_pnet`/
    `shared_es_qnet` (`network.py:398-399`), genuine `pe.Var`s: bounded via
    `.setlb()`/`.setub()` (or `.fix()`) on every TSO-side index in
    `t_sess` and every DSO-side index in `d_sess` (`p56a_oracle._sess_pairs`
    convention, reproduced here identically). The ESSO's own achieved value
    (`hull_entries_with_esso`'s `esso_sess_p`/`esso_sess_q`) contributes only
    to the interval's [min, max] -- there is no ESSO Pyomo Var to bound,
    since the ESSO subproblem is not one of the 48 network blocks and is not
    re-solved here.
    """
    tso = planning.transmission_network
    descriptors = []
    for node, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net, d_net = tso.network[year][day], dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node][year][day]
                dn = list(t_net.active_distribution_network_nodes).index(node)
                adn_idx = t_net.get_node_idx(node)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages)
                          if s.bus == node]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages)
                          if s.bus == ref_id]

                rows_tp = _get_or_add_constraint_list(t_model, 'p515s41_hull_pf_p_rows')
                rows_tq = _get_or_add_constraint_list(t_model, 'p515s41_hull_pf_q_rows')
                rows_dp = _get_or_add_constraint_list(d_model, 'p515s41_hull_pf_p_rows')
                rows_dq = _get_or_add_constraint_list(d_model, 'p515s41_hull_pf_q_rows')

                for p in t_model.periods:
                    entry = hull[(node, year, day, p)]

                    # ---- Voltage (2 agents) ----
                    lo_v, hi_v, deg_v = _interval(entry['tso_v'], entry['dso_v'])
                    lo_vs, hi_vs = lo_v ** 2, hi_v ** 2
                    tv = t_model.vmag_sqr[adn_idx, 0, 0, p]
                    dv = d_model.vmag_sqr[ref_idx, 0, 0, p]
                    _bound_var(tv, lo_vs, hi_vs, deg_v)
                    _bound_var(dv, lo_vs, hi_vs, deg_v)
                    descriptors.append({
                        'channel': 'V', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'tso', 'lo': lo_v, 'hi': hi_v, 'degenerate': deg_v,
                        'get_value': (lambda v=tv: pe.value(v) ** 0.5)})
                    descriptors.append({
                        'channel': 'V', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'dso', 'lo': lo_v, 'hi': hi_v, 'degenerate': deg_v,
                        'get_value': (lambda v=dv: pe.value(v) ** 0.5)})

                    # ---- Interface P (2 agents) ----
                    lo_p, hi_p, deg_p = _interval(entry['tso_p'], entry['dso_p'])
                    tp_expr = t_model.pc_adn[dn, 0, 0, p]
                    dp_expr = d_model.pg_adn[0, 0, p]
                    _bound_expr(rows_tp, tp_expr, lo_p, hi_p, deg_p)
                    _bound_expr(rows_dp, dp_expr, lo_p, hi_p, deg_p)
                    descriptors.append({
                        'channel': 'PF_P', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'tso', 'lo': lo_p, 'hi': hi_p, 'degenerate': deg_p,
                        'get_value': (lambda v=tp_expr: pe.value(v))})
                    descriptors.append({
                        'channel': 'PF_P', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'dso', 'lo': lo_p, 'hi': hi_p, 'degenerate': deg_p,
                        'get_value': (lambda v=dp_expr: pe.value(v))})

                    # ---- Interface Q (2 agents) ----
                    lo_q, hi_q, deg_q = _interval(entry['tso_q'], entry['dso_q'])
                    tq_expr = t_model.qc_adn[dn, 0, 0, p]
                    dq_expr = d_model.qg_adn[0, 0, p]
                    _bound_expr(rows_tq, tq_expr, lo_q, hi_q, deg_q)
                    _bound_expr(rows_dq, dq_expr, lo_q, hi_q, deg_q)
                    descriptors.append({
                        'channel': 'PF_Q', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'tso', 'lo': lo_q, 'hi': hi_q, 'degenerate': deg_q,
                        'get_value': (lambda v=tq_expr: pe.value(v))})
                    descriptors.append({
                        'channel': 'PF_Q', 'node': node, 'year': year, 'day': day, 'period': p,
                        'side': 'dso', 'lo': lo_q, 'hi': hi_q, 'degenerate': deg_q,
                        'get_value': (lambda v=dq_expr: pe.value(v))})

                    # ---- Shared-ESS P (3 agents: TSO, DSO, ESSO) ----
                    lo_ep, hi_ep, deg_ep = _interval(
                        entry['tso_sess_p'], entry['dso_sess_p'], entry['esso_sess_p'])
                    for e in t_sess:
                        var = t_model.shared_es_pnet[e, 0, 0, p]
                        _bound_var(var, lo_ep, hi_ep, deg_ep)
                        descriptors.append({
                            'channel': 'ESS_P', 'node': node, 'year': year, 'day': day,
                            'period': p, 'side': 'tso', 'lo': lo_ep, 'hi': hi_ep,
                            'degenerate': deg_ep, 'get_value': (lambda v=var: pe.value(v))})
                    for e in d_sess:
                        var = d_model.shared_es_pnet[e, 0, 0, p]
                        _bound_var(var, lo_ep, hi_ep, deg_ep)
                        descriptors.append({
                            'channel': 'ESS_P', 'node': node, 'year': year, 'day': day,
                            'period': p, 'side': 'dso', 'lo': lo_ep, 'hi': hi_ep,
                            'degenerate': deg_ep, 'get_value': (lambda v=var: pe.value(v))})

                    # ---- Shared-ESS Q (3 agents: TSO, DSO, ESSO) ----
                    lo_eq, hi_eq, deg_eq = _interval(
                        entry['tso_sess_q'], entry['dso_sess_q'], entry['esso_sess_q'])
                    for e in t_sess:
                        var = t_model.shared_es_qnet[e, 0, 0, p]
                        _bound_var(var, lo_eq, hi_eq, deg_eq)
                        descriptors.append({
                            'channel': 'ESS_Q', 'node': node, 'year': year, 'day': day,
                            'period': p, 'side': 'tso', 'lo': lo_eq, 'hi': hi_eq,
                            'degenerate': deg_eq, 'get_value': (lambda v=var: pe.value(v))})
                    for e in d_sess:
                        var = d_model.shared_es_qnet[e, 0, 0, p]
                        _bound_var(var, lo_eq, hi_eq, deg_eq)
                        descriptors.append({
                            'channel': 'ESS_Q', 'node': node, 'year': year, 'day': day,
                            'period': p, 'side': 'dso', 'lo': lo_eq, 'hi': hi_eq,
                            'degenerate': deg_eq, 'get_value': (lambda v=var: pe.value(v))})
    return descriptors


def _hull_bounds_active(descriptors, rel_tol=HULL_ACTIVE_REL_TOL):
    """Count, per channel, how many constrained (TSO/DSO-side) quantities sit
    within `rel_tol` relative of an interval end at the POLISHED point (read
    AFTER the 48 solves). A degenerate entry counts as active by definition.
    Relative tolerance uses `max(|bound|, 1.0)` as the scale to avoid a
    division blow-up on near-zero reactive-power bounds; documented here,
    not silently chosen."""
    active = Counter()
    total = Counter()
    per_channel_detail = []
    for d in descriptors:
        total[d['channel']] += 1
        if d['degenerate']:
            is_active = True
            val = d['lo']
        else:
            val = d['get_value']()
            scale = max(abs(d['lo']), abs(d['hi']), 1.0)
            is_active = (abs(val - d['lo']) <= rel_tol * scale
                         or abs(val - d['hi']) <= rel_tol * scale)
        if is_active:
            active[d['channel']] += 1
        per_channel_detail.append({
            'channel': d['channel'], 'node': d['node'], 'year': d['year'], 'day': d['day'],
            'period': d['period'], 'side': d['side'], 'lo': d['lo'], 'hi': d['hi'],
            'degenerate': d['degenerate'], 'polished_value': val, 'active': is_active})
    return active, total, per_channel_detail


# ==============================================================================
#  polish
# ==============================================================================
def _polish_solve_one_block(network_obj, model, params):
    """One polish solve. `_switch_to_base_objective` (imported, unchanged)
    deactivates the ADMM and p58-rescaled objectives and activates the base
    one. `NET._clear_multiplier_suffixes` clears `ipopt_zL_in`/`ipopt_zU_in`/
    `dual` explicitly (belt-and-suspenders: `from_warm_start=False` below
    already prevents IPOPT from being TOLD to import them, per
    `network.py:580-599`, but the suffixes may still hold stale values from
    the last real ADMM cycle -- cleared here so "no multiplier import" is
    unambiguous). `bound_push`/`bound_frac`/`slack_bound_push`/
    `slack_bound_frac` are overridden to IPOPT's compiled default on a
    save/restore scoped copy of `params.solver_params.options` (module
    docstring, "BOUND PUSH"); every other option (`tol`, `max_iter`,
    `recovery_options`, ...) is untouched, so production recovery policy
    (tier-1/tier-2) fires exactly as it would on any other network solve.
    Primal warm start needs no action: Pyomo's NL writer emits each Var's
    CURRENT `.value` as x0, and nothing here resets a value."""
    _switch_to_base_objective(model)
    NET._clear_multiplier_suffixes(model)
    solver_params = params.solver_params
    saved_options = dict(solver_params.options or {})
    overridden_options = dict(saved_options)
    overridden_options.update(IPOPT_DEFAULT_BOUND_OPTIONS)
    solver_params.options = overridden_options
    try:
        with redirect_stdout(io.StringIO()):
            result = network_obj.run_smopf(model, params, from_warm_start=False,
                                            print_header=False)
    finally:
        solver_params.options = saved_options
    return bool(srp._solver_result_succeeded(result))


def _polish_all_blocks_hull(planning, models, consensus_vars):
    """The hull-polish step (build items 2-6). Mutates `models` in place
    (no clone -- discarded by the caller regardless, same convention
    `p56a_oracle`/`p515_s40_polish_gap.py` use). Returns the full record;
    never raises on a per-block solver failure (reported, not dropped --
    gate is `None` when `all_solved` is False)."""
    before_recourse = planning.get_operational_recourse_components(models)
    before_per_block = O.per_block_base_objectives(planning, models)

    hull = hull_entries_with_esso(planning, models, consensus_vars)
    descriptors = apply_hull_bounds(planning, models, hull)

    guard = SolveProfileGuard(POLISH_PERMITTED, label='P5.15-S41 hull polish').install()
    try:
        blocks, all_solved = [], True
        for tag, holder in O._tagged_holders(planning):
            node_of = None if tag == 'TSO' else int(tag[3:])
            for year in holder.years:
                for day in holder.days:
                    network_obj = holder.network[year][day]
                    model = (models['tso'][year][day] if tag == 'TSO'
                             else models['dso'][node_of][year][day])
                    ok = _polish_solve_one_block(network_obj, model, holder.params)
                    all_solved &= ok
                    blocks.append({'agent': tag, 'year': year, 'day': day, 'solved': ok})
    finally:
        guard.uninstall()

    after_per_block = O.per_block_base_objectives(planning, models)
    active_by_channel, total_by_channel, hull_bound_detail = _hull_bounds_active(descriptors)

    per_block_records = []
    for b in blocks:
        key = f"{b['agent']}|{b['year']}|{b['day']}"
        before_v = before_per_block[key]['weighted_base_objective']
        after_v = after_per_block[key]['weighted_base_objective']
        per_block_records.append({
            'block': key, 'agent': b['agent'], 'solved': b['solved'],
            'weighted_base_objective_before': before_v,
            'weighted_base_objective_after': after_v,
            'delta': after_v - before_v,
        })

    failed_blocks = [r['block'] for r in per_block_records if not r['solved']]
    flagged_blocks = [r['block'] for r in per_block_records if r['delta'] > FLAG_ABS]

    gate = None
    after_recourse = None
    tso_delta = dso_delta = None
    max_abs_delta_block = None
    if all_solved:
        after_recourse = planning.get_operational_recourse_components(models)
        # P5.15 Addendum 23 / spec v12 item2_hull_polish (Planner correction before the full
        # run): the GATE quantity is Delta = sum_i [f_i(polished) - f_i(certified)] over the
        # 48 blocks' weighted base objectives -- the quantity each block minimises, and the one
        # for which Delta <= 0 holds by construction. It equals the change in
        # gross_operational_cost_INCLUDING_settlement (sum_i f_i reproduces that total exactly).
        # The settlement-EXCLUDED gross_operational_cost additionally moves with the interface
        # settlement transfer, which cancels between TSO and DSO only when both sides hold the
        # same interface values; after independent polishing within the hull it need not (smoke
        # test: -10.6M at 2 cycles). That change is REPORTED, not gated. Denominator: the
        # certified settlement-excluded cost, the convention the oracle's cost is quoted in.
        tso_delta = sum(r['delta'] for r in per_block_records if r['agent'] == 'TSO')
        dso_delta = sum(r['delta'] for r in per_block_records if r['agent'] != 'TSO')
        delta_sum_blocks = tso_delta + dso_delta
        certified_cost = before_recourse['gross_operational_cost']
        relative = abs(delta_sum_blocks) / abs(certified_cost) if certified_cost else None
        incl_before = before_recourse.get('gross_operational_cost_including_settlement')
        incl_after = after_recourse.get('gross_operational_cost_including_settlement')
        sum_f_before = sum(r['weighted_base_objective_before'] for r in per_block_records)
        max_abs_row = max(per_block_records, key=lambda r: abs(r['delta']))
        max_abs_delta_block = {'block': max_abs_row['block'], 'delta': max_abs_row['delta']}
        gate = {
            'gate_quantity': 'Delta = sum over the 48 blocks of [f_i(polished) - f_i(certified)] (weighted base objectives)',
            'denominator': 'certified gross_operational_cost (settlement-excluded; the oracle cost convention)',
            'delta_sum_blocks': delta_sum_blocks, 'tso_delta': tso_delta, 'dso_delta': dso_delta,
            'delta_sign_as_expected_le_0': bool(delta_sum_blocks <= 0.0),
            'certified_cost': certified_cost,
            'max_abs_delta_block': max_abs_delta_block,
            'relative_pct': (relative * 100.0) if relative is not None else None,
            'threshold_pct': GATE_THRESHOLD_PCT,
            'pass': (relative is not None and relative < GATE_THRESHOLD_PCT / 100.0),
            'reconciliation_sum_f_before_vs_gross_including_settlement': {
                'sum_f_before': sum_f_before, 'gross_including_settlement_before': incl_before,
                'abs_diff': (abs(sum_f_before - incl_before) if incl_before is not None else None)},
            'reported_not_gated': {
                'gross_operational_cost_before': before_recourse['gross_operational_cost'],
                'gross_operational_cost_after': after_recourse['gross_operational_cost'],
                'gross_operational_cost_change_settlement_excluded':
                    after_recourse['gross_operational_cost'] - before_recourse['gross_operational_cost'],
                'gross_including_settlement_change': (incl_after - incl_before
                                                      if (incl_after is not None and incl_before is not None) else None),
                'interface_settlement_total_before': before_recourse.get('interface_settlement_total'),
                'interface_settlement_total_after': after_recourse.get('interface_settlement_total'),
            },
        }

    per_block_sorted = sorted(per_block_records, key=lambda r: abs(r['delta']), reverse=True)

    return {
        'before_recourse_components': before_recourse,
        'after_recourse_components': after_recourse,
        'all_solved': all_solved,
        'failed_blocks': failed_blocks,
        'flagged_blocks': flagged_blocks,
        'flag_abs_threshold': FLAG_ABS,
        'n_blocks': len(blocks),
        'per_block': per_block_records,
        'per_block_by_largest_abs_delta': per_block_sorted,
        'hull_bounds_active_by_channel': dict(active_by_channel),
        'hull_bounds_total_by_channel': dict(total_by_channel),
        'hull_active_rel_tol': HULL_ACTIVE_REL_TOL,
        'n_hull_descriptors': len(descriptors),
        'solve_profile': {
            'observed': dict(guard.counts),
            'n_blocks_dispatched': len(blocks),
            'retries_beyond_one_per_block': guard.counts['permitted_solve'] - len(blocks),
            'blocked_calls': guard.counts['blocked_solve'] + guard.counts['blocked_exec'],
        },
        'gate': gate,
    }, hull_bound_detail


# ==============================================================================
#  post-run hook / main
# ==============================================================================
def _make_post_run_hook(out_dir, label, floor_rows_by_node, floor_sidecar_path,
                        recourse_jump_path, ess_stride_path, pf_stride_path,
                        exempt_until_state_path, result_holder):
    def _hook(planning, sed, models, rows, report, out_dir=out_dir, label=label,
              state=None):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(pf_stride_path, REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(
            exempt_until_state_path, REPO)

        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path)

        repro = _reproduction_check(report)
        result_holder['reproduction'] = repro
        if not repro['reproduces']:
            result_holder['polish'] = None
            result_holder['hull_bound_detail'] = None
            result_holder['stopped_before_polish'] = True
            print(f"[S41-HULL-POLISH] REPRODUCTION CHECK FAILED (mode={repro['mode']}): "
                  f"{repro['n_diffs']} diffs. STOPPING before polish. First diffs:")
            for d in repro['first_diffs']:
                print(f'  [FIRST DIFFS] {d}')
            return

        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('S41 hull polish: state/consensus_vars not available to '
                               'post_run_hook -- cannot build the hull.')
        print('[S41-HULL-POLISH] reproduction OK; hull-polishing 48 network blocks '
              'at fixed consensus intervals ...', flush=True)
        polish_started = time.time()
        polish, hull_bound_detail = _polish_all_blocks_hull(
            planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - polish_started
        result_holder['polish'] = polish
        result_holder['hull_bound_detail'] = hull_bound_detail
        result_holder['stopped_before_polish'] = False
        gate = polish['gate']
        if gate is not None:
            print(f"[S41-HULL-POLISH] gate: relative={gate['relative_pct']}% "
                  f"(threshold {gate['threshold_pct']}%) pass={gate['pass']}", flush=True)
        else:
            print(f"[S41-HULL-POLISH] gate NOT evaluated -- polish failed on blocks: "
                  f"{polish['failed_blocks']}", flush=True)
    return _hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-cycles', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    is_smoke = args.smoke_cycles is not None
    sfx = ('_' + args.suffix.strip('_')) if args.suffix.strip('_') else ''
    out_dir = (OUT_DIR_SMOKE if is_smoke else OUT_DIR_FULL) + sfx
    num_max_iters = args.smoke_cycles if is_smoke else FULL_NUM_MAX_ITERS
    run_eval_id = ('p515s41_hull_polish_smoke_run' if is_smoke else 'p515s41_hull_polish_run') + sfx
    precheck_eval_id = run_eval_id + '_precheck'

    failures = _check_preconditions(out_dir)
    if failures:
        for f in failures:
            print(f'[S41-HULL-POLISH PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S41-HULL-POLISH] preconditions passed (no lock, no forbidden process, '
          "fresh output dir, production files clean, D's reference present).")

    G._acquire_exclusive_run_lock()
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    _capture_checklist, floor_rows_by_node, floor_counts_by_node = _build_floor_rows(
        precheck_eval_id)
    print(f'[S41-HULL-POLISH] soh_floor_row_counts_by_node='
          f"{ {n: len(r) for n, r in floor_rows_by_node.items()} }")

    recourse_jump_path = os.path.join(out_dir, 'recourse_jump_sidecar_baseline.jsonl')
    ess_stride_path = os.path.join(out_dir, 'ess_entry_stride_baseline.jsonl')
    floor_sidecar_path = os.path.join(out_dir, 'soh_floor_sidecar_baseline.jsonl')
    pf_stride_path = os.path.join(out_dir, f'pf_entry_stride_{ARM_LABEL}.jsonl')
    exempt_until_state_path = os.path.join(out_dir, f'ess_exempt_until_state_{ARM_LABEL}.jsonl')
    for path in (recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path,
                exempt_until_state_path):
        _refuse_overwrite(path)

    result_holder = {}
    hook = _make_post_run_hook(out_dir, ARM_LABEL, floor_rows_by_node, floor_sidecar_path,
                               recourse_jump_path, ess_stride_path, pf_stride_path,
                               exempt_until_state_path, result_holder)

    print(f'[S41-HULL-POLISH] run CASE-FILE-ALONE (label={ARM_LABEL!r}, apply_rho=False, '
          f'pre_solve_hook=None), num_max_iters_override={num_max_iters} '
          f"({'SMOKE' if is_smoke else 'FULL -- to Boyd certification'}).")
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(exempt_until_state_path):
        report, report_path = G.run_admm_arm(
            ARM_LABEL, out_dir, k_override=None, eval_id=run_eval_id,
            num_max_iters_override=num_max_iters, apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=hook, pre_solve_hook=None)

    print(f"[S41-HULL-POLISH] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f"wall={report['wall_clock_s']:.1f}s")

    hull_bound_detail = result_holder.get('hull_bound_detail')
    hull_bound_detail_path = None
    if hull_bound_detail is not None:
        hull_bound_detail_path = os.path.join(out_dir, 'hull_bound_detail.json')
        _refuse_overwrite(hull_bound_detail_path)
        with open(hull_bound_detail_path, 'w') as handle:
            json.dump(hull_bound_detail, handle, indent=1, default=str)

    payload = {
        'stage': 'P5.15 Addendum 23 item (2) -- Step 3.5 restated: interval-hull polish gate',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 23',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json',
            'WORKER_REPORT_S41_POLISH_PREREQ.md',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm_label': ARM_LABEL,
        'mode': 'smoke' if is_smoke else 'full',
        'smoke_cycles': args.smoke_cycles,
        'num_max_iters_used': num_max_iters,
        'out_dir': os.path.relpath(out_dir, REPO),
        'admm_report_path': os.path.relpath(report_path, REPO),
        'admm_cycles_run': report['cycles_run'],
        'admm_gross_operational_cost': report['gross_operational_cost'],
        'admm_converged_at_cycle': report.get('converged_at_cycle'),
        'reproduction': result_holder.get('reproduction'),
        'stopped_before_polish': result_holder.get('stopped_before_polish'),
        'polish': result_holder.get('polish'),
        'hull_bound_detail_path': (os.path.relpath(hull_bound_detail_path, REPO)
                                    if hull_bound_detail_path else None),
        'ipopt_default_bound_options': IPOPT_DEFAULT_BOUND_OPTIONS,
        'flag_abs_threshold': FLAG_ABS,
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(out_dir, 'hull_polish_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S41-HULL-POLISH] wrote {results_path}')

    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_dir, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S41-HULL-POLISH] wrote {manifest_path}')

    if not result_holder.get('reproduction', {}).get('reproduces'):
        print('[S41-HULL-POLISH] *** REPRODUCTION CHECK FAILED -- stopped before polish. '
              f'See {results_path}. ***')
        sys.exit(1)

    gate = (result_holder.get('polish') or {}).get('gate')
    if gate is None:
        print('[S41-HULL-POLISH] *** GATE NOT EVALUATED -- a polish block failed. '
              f'See {results_path}. ***')
        sys.exit(1)

    print(f"[S41-HULL-POLISH] GATE: relative={gate['relative_pct']:.6f}% "
          f"(threshold {gate['threshold_pct']}%) PASS={gate['pass']}")
    if not gate['pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
