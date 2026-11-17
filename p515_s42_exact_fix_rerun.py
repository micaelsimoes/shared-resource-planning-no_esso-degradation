"""
P5.15 Addendum 24 item 1 -- the exact-fix (midpoint) polish re-run at D's
certified point, with the FIXED `p56a_oracle._interface_expression`.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 24 and frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`,
`item1_helper_fix.exact_fix_rerun`: "the exact-fix (midpoint) polish of
`p515_s40_polish_gap.py` semantics, with the fixed helper, at D's certified
point obtained by the in-process bitwise reproduction; 48 polish solves;
fresh output root and ids; the hull polish is NOT re-run."

================================================================================
WHAT THIS SCRIPT DOES, PER THE SPEC
================================================================================
This is `p515_s40_polish_gap.py`'s own exact-fix (midpoint) mechanism,
UNCHANGED in every respect except one: `p56a_oracle._interface_expression` is
now fixed (P5.15 Addendum 24 item 1, committed separately). No line of THIS
file re-derives the interface expression -- it imports `p56a_oracle.
common_coordinated_values`/`apply_common_values` BY IMPORT, so it
automatically uses whatever `p56a_oracle.py` currently defines, exactly as
`p515_s40_polish_gap.py` and `p515_s41_hull_polish.py` already do.

  1. Runs the case-file configuration to Boyd certification IN-PROCESS (the
     SAME `p515_g_g1_g4_admm_gates.run_admm_arm(label='s39_D', apply_rho=False,
     num_max_iters_override=300)` call `p515_s40_polish_gap.py` and
     `p515_s41_hull_polish.py` use) and requires the run's report to
     reproduce D's own committed report BITWISE, cycle 139, cost
     650966975.2943751 -- reusing `p515_s40_polish_gap._reproduction_check`
     BY IMPORT, unchanged, with the `rule_eleven_checklist` subtree reported
     but NON-GATING (per spec).
  2. Optionally (`--persist-certified-models`) pickles the certified
     TSO/DSO models BEFORE polishing (hash-recorded), so a later polish
     variant needs no further ~80-minute reproduction (spec v13
     `persist_certified_models`, optional).
  3. Fixes coordination EXACTLY at the midpoint: `p56a_oracle.
     common_coordinated_values` (achieved TSO/DSO values, midpoint
     convention) then `p56a_oracle.apply_common_values` (BY IMPORT,
     unchanged -- now using the FIXED `_interface_expression`, i.e. the
     model's own `pc_adn`/`qc_adn`, which INCLUDES `interface_delta_p/q`).
  4. Switches every block to the unscaled base objective
     (`p515_s40_polish_gap._switch_to_base_objective`, BY IMPORT, unchanged
     -- the objective-switch fix that stage already made, per the task).
  5. Re-solves each of the 48 network blocks with `network.run_smopf`,
     production's own SMOPF entry point, production recovery policy
     (tier-1 cold, tier-2 cold+adaptive), THE SAME solver options
     `p515_s40_polish_gap.py` used (case-file bound push, NOT overridden to
     IPOPT's compiled default -- that override is `p515_s41_hull_polish.py`'s
     OWN v12 requirement, not part of "the same exact-fix (midpoint)
     semantics as `p515_s40_polish_gap.py`" this task asks for).
  6. Per block, records: solved/failed, the FINAL attempt's IPOPT exit
     message, final (unscaled) constraint violation, iterations -- parsed
     from that attempt's own IPOPT log (see "LOG PARSING" below) -- and, for
     a failed TSO block, the largest violated interface rows (the
     `interface_delta_p/q` shortfall against its own bound, per (ADN node,
     period), computed directly and exactly, not merely read off a generic
     constraint scan -- see "NODE-7 / INTERFACE DIAGNOSTICS" below).
  7. Scores the run against spec v13's `predictions_recorded_in_advance.
     exact_fix_rerun` (see `PREDICTIONS` below) and reports the
     `rule_eleven_checklist` reproduction as a non-gating cross-check against
     D (cycle 139, cost 650,966,975.2943751).
  8. Requires all 48 blocks to solve before evaluating the (informational,
     NOT the point of this run) `sum_i [f_i(polished) - f_i(certified)]`
     delta -- same convention as `p515_s40_polish_gap.py`/`p515_s41_hull_
     polish.py`; this run's PRIMARY verdict is the prediction score, not that
     delta (12/48 blocks are PREDICTED to fail).

================================================================================
THE ESSO-POSITION-INDEXING FIX -- NOT RELEVANT HERE
================================================================================
`p515_s41_hull_polish.hull_entries_with_esso` needed the ESSO's
position-indexed `es_pnet`/`es_qnet` (`shared_energy_storage_data.py:36-37`,
indexed by POSITION into `shared_ess_data.years`/`.days`, not by the
`(year, day)` LABEL) because it adds the ESSO's OWN achieved dispatch as a
third hull endpoint. `p56a_oracle.common_coordinated_values`/
`apply_common_values` -- what THIS harness uses, unchanged, per the spec's
"the same exact-fix (midpoint) semantics as `p515_s40_polish_gap.py`" --
never read `models['esso']` at all (confirmed by inspection: neither function
references the `esso` key of the `models` dict anywhere in `p56a_oracle.py`).
So the ESSO-position-indexing fix has no code path to apply to here; recorded
as checked, not silently skipped.

================================================================================
LOG PARSING
================================================================================
`network.run_smopf`/`_run_smopf` does not return a log path or parsed
iteration/violation figures -- only a Pyomo `SolverResults` object
(`network.py:746-850`). This harness independently reconstructs the log path
`_create_smopf_solver` would have used (`_expected_log_path`, a read-only
replica of `network.py:560-578`'s `output_file`/day-label/suffix
construction -- NOT a change to that function), for each of the three
possible attempts (primary / tier-1 cold recovery / tier-2 cold+adaptive
recovery, `log_suffix` `None`/`'recovery'`/`'recovery_tier2'`,
`network.py:604, 780, 802`). Per-block stdout is captured (NOT globally
redirected, unlike `p56a_oracle.polish_networks`/`_polish_networks_fixed_
consensus`'s blanket `redirect_stdout`) so `_classify_final_attempt` can
determine, from the DISTINCTIVE, always-printed substrings `_run_smopf`
itself emits (`'Retrying network solve once for'`, `'Retrying network solve
(tier 2:'`, `'... recovery solve succeeded ...'`), which of the three
attempts was FINAL -- deterministically, from production's own control flow
(`network.py:762-807`), not by parsing a log path out of a warning message
(which is only printed on FAILURE, not on a clean primary success) or by
relying on file mtimes. The final attempt's own log file's LAST IPOPT
section (`_parse_ipopt_log_tail`) is then read for `EXIT:` (the message),
`Number of Iterations....: N`, and the `Constraint violation....:` line's
SECOND (unscaled) number from the terminal `(scaled)/(unscaled)` summary
block printed immediately after -- the SAME convention `WORKER_REPORT_S41_
POLISH_PREREQ.md` Part 1 used by hand; confirmed programmatically here
against the same known log
(`p515_s42_exact_fix_rerun_checks.py`).

================================================================================
NODE-7 / INTERFACE DIAGNOSTICS
================================================================================
For every TSO block, for every (ADN node, period), the harness computes the
EXACT required value of `interface_delta_p`/`interface_delta_q` needed to
satisfy `apply_common_values`'s new row (`pc_adn == common_p`, i.e.
`pc[FIXED] + interface_delta_p == common_p` on the ADMM path, since the
`flex_p/q_up/down` legs are fixed at 0 there -- `_interface_expression`'s own
docstring, `p56a_oracle.py`), `required_delta = common_p - pc_fixed_value`,
and compares it against `interface_delta_p`'s own declared bound
(`+/- interface_transf_rating`, `shared_resources_planning.py:3604-3607`).
This is EXACT (not read off a generic `scan_constraints` index), and is
converted to MVA (`* network.baseMVA`, 100 for case9 -- "IPOPT's violation
UNITS are the Pyomo model's own per-unit convention, i.e. exactly this
conversion" -- see `PREDICTIONS`/the harness's own report for where this
conversion is applied to the node-7 rating row specifically). Node 7's own
entries are additionally singled out per block (the node the six
spring/summer predictions name).

================================================================================
PREDICTIONS (spec v13, recorded here VERBATIM before the run -- do not edit
after seeing a result)
================================================================================
  - the six spring/summer TSO blocks (2025/2030/2035 Spring, Summer) --
    where node 7's interface midpoint exceeds 100 MVA -- report LOCAL
    INFEASIBILITY with constraint violation <= 5e-4 MVA;
  - the six autumn/winter TSO blocks (2025/2030/2035 Autumn, Winter) SOLVE;
  - the five DSO-2025 blocks (DSO5 Spring; DSO7 Spring, Autumn; DSO9 Spring,
    Autumn): recorded WITHOUT prediction.
  - if all 12 TSO blocks solve: re-examine what quantity the rating-midpoint
    check (utilization 1.000005) measures.

================================================================================
EXACT FULL-RUN LAUNCH COMMAND (Planner launches; Worker does NOT)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s42_exact_fix_rerun.py --persist-certified-models \\
        > data/SRP1/Results/P515S42/exact_fix_rerun_launch.log 2>&1

(drop `--persist-certified-models` if the ~80-minute reproduction is not
worth trading for the pickle's disk cost -- both variants write to the SAME
fresh, write-once `data/SRP1/Results/P515S42/exact_fix_rerun/`; only run ONE.)

Preconditions identical in kind to `p515_s40_polish_gap.py`/`p515_s41_hull_
polish.py` (lock absent, no forbidden live process, fresh output root,
production files clean in git, D's committed reference report present) --
checked before anything is written, refuses loudly otherwise. Run attached,
alone, both streams captured via the shell redirection above; never
`screen`/`nohup`/`&`.

================================================================================
SMOKE TEST (the Worker runs THIS one, `--smoke-cycles 2`)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s42_exact_fix_rerun.py --smoke-cycles 2 --persist-certified-models \\
        > data/SRP1/Results/P515S42/exact_fix_rerun_smoke_launch.log 2>&1

At 2 cycles the run is NOT D's certified point, so the prediction score is
NOT the prediction test (documented and printed, not silently glossed over)
-- the point of the smoke test is mechanics (reproduction machinery, the
polish/diagnostic capture code paths, the persisted-model pickle), exactly
as `p515_s40_polish_gap.py`/`p515_s41_hull_polish.py`'s own smoke tests are.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p56a_oracle as O  # noqa: E402 -- common_coordinated_values/apply_common_values, BY IMPORT, FIXED
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator/precondition conventions, BY IMPORT
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from helper_functions import solver_result_summary  # noqa: E402
# Do NOT modify p515_s40_polish_gap.py (its committed evidence stands) --
# reuse its reproduction-check machinery and constants BY IMPORT, per the
# spec ("the same exact-fix (midpoint) semantics as p515_s40_polish_gap.py").
from p515_s40_polish_gap import (  # noqa: E402
    ARM_LABEL, FULL_NUM_MAX_ITERS, D_REFERENCE_PATH, D_CERTIFICATION_CYCLE,
    D_CERTIFIED_COST, _reproduction_check, _build_floor_rows, _refuse_overwrite,
    _switch_to_base_objective,
)

OUT_DIR_FULL = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'exact_fix_rerun')
OUT_DIR_SMOKE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'exact_fix_rerun_smoke')

# Polish solves reach the solver through the SAME call site every other
# network solve in this campaign uses (network.py:_run_smopf_solver_attempt).
POLISH_PERMITTED = [('network.py', '_run_smopf_solver_attempt')]

GATE_THRESHOLD_PCT = 0.1
MVA_CONVERSION_NOTE = (
    "IPOPT's 'unscaled' figures are in the Pyomo model's OWN per-unit convention "
    "(the model is built in per-unit of network.baseMVA -- 100 for case9/TSO); "
    "MVA = unscaled_value * network.baseMVA. This is the SAME conversion "
    "WORKER_REPORT_S41_POLISH_PREREQ.md used by hand ('5e-4 MVA over a 100 MVA "
    "rating (~5e-6 in normalized terms)').")

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + (
    'p515_s42_',)

SPRING_SUMMER_TSO_BLOCKS = tuple(
    f'TSO|{year}|{season}' for year in (2025, 2030, 2035) for season in ('Spring', 'Summer'))
AUTUMN_WINTER_TSO_BLOCKS = tuple(
    f'TSO|{year}|{season}' for year in (2025, 2030, 2035) for season in ('Autumn', 'Winter'))
DSO_2025_BLOCKS_NO_PREDICTION = (
    'DSO5|2025|Spring', 'DSO7|2025|Spring', 'DSO7|2025|Autumn',
    'DSO9|2025|Spring', 'DSO9|2025|Autumn')


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
#  IPOPT log location + parsing
# ==============================================================================
def _expected_log_path(network, params, log_suffix=None):
    """Read-only replica of `network.py:560-578`'s `output_file` -> log-path
    construction (NOT a change to that function -- `_run_smopf`/`network.
    run_smopf` never return the path, so it must be reconstructed to find the
    file afterward)."""
    options = dict(params.solver_params.options or {})
    output_file = options.get('output_file')
    if not output_file:
        return None
    configured_path = os.path.join(network.logs_dir, output_file)
    path_stem, path_extension = os.path.splitext(configured_path)
    day_label = ''.join(
        character if character.isalnum() or character in ('-', '_') else '_'
        for character in str(network.day))
    suffix = f'_{network.year}_{day_label}'
    if log_suffix:
        suffix += f'_{log_suffix}'
    return f'{path_stem}{suffix}{path_extension}'


def _classify_final_attempt(captured_text):
    """Which of the (primary / tier-1 cold recovery / tier-2 cold+adaptive
    recovery) attempts was FINAL, from the DISTINCTIVE substrings `_run_smopf`
    itself always prints (`network.py:762-807`) -- deterministic from
    production's own control flow, not a guess. Returns
    (log_suffix_or_None, attempt_label)."""
    tier2_attempted = 'Retrying network solve (tier 2:' in captured_text
    tier1_attempted = 'Retrying network solve once for' in captured_text
    if tier2_attempted:
        return 'recovery_tier2', 'tier-2 cold+adaptive recovery'
    if tier1_attempted:
        return 'recovery', 'tier-1 cold recovery'
    return None, 'primary'


def _parse_ipopt_log_tail(log_path, tail_bytes=2_000_000):
    """Read the LAST IPOPT section of `log_path` (files are appended across
    the whole run + polish attempt, `file_append='yes'`) and extract the
    final iteration count, the UNSCALED constraint violation from the
    terminal summary block (the SECOND number on the 'Constraint
    violation....:' line printed immediately after 'Number of
    Iterations....: N' -- confirmed against the SAME log format
    `WORKER_REPORT_S41_POLISH_PREREQ.md` Part 1 read by hand;
    `p515_s42_exact_fix_rerun_checks.py` re-checks it programmatically), and
    the EXIT message. Never raises -- log parsing is read-only reporting, not
    a solve-path gate; every field is None if not found."""
    out = {'log_path': None, 'found': False, 'iterations': None,
           'constraint_violation_unscaled': None, 'exit_message': None}
    if not log_path or not os.path.isfile(log_path):
        out['log_path'] = os.path.relpath(log_path, REPO) if log_path else None
        return out
    out['log_path'] = os.path.relpath(log_path, REPO)
    with open(log_path, 'rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes))
        raw = handle.read()
    text = raw.decode('utf-8', errors='replace')
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('This is Ipopt version'):
            start_idx = i
    section = lines[start_idx:]
    out['found'] = True

    for i, line in enumerate(section):
        if line.startswith('Number of Iterations'):
            try:
                out['iterations'] = int(line.split(':', 1)[1].strip())
            except (IndexError, ValueError):
                pass
            for follow in section[i + 1:i + 8]:
                if follow.strip().startswith('Constraint violation'):
                    parts = follow.split(':', 1)[1].split()
                    if len(parts) >= 2:
                        try:
                            out['constraint_violation_unscaled'] = float(parts[1])
                        except ValueError:
                            pass
                    break

    for line in section:
        if line.startswith('EXIT:'):
            out['exit_message'] = line[len('EXIT:'):].strip()
    return out


# ==============================================================================
#  interface / node-7 diagnostics
# ==============================================================================
def _tso_interface_diagnostics(planning, models, common, year, day):
    """For EVERY (ADN node, period) on this TSO block: the exact required
    `interface_delta_p/q` value, its declared bound, and the signed excess
    beyond that bound (0 if within bounds) -- see the module docstring,
    "NODE-7 / INTERFACE DIAGNOSTICS". Never solves; only reads/computes from
    already-fixed/already-free model state."""
    tso = planning.transmission_network
    t_net = tso.network[year][day]
    t_model = models['tso'][year][day]
    base_mva = t_net.baseMVA
    rows = []
    for node in sorted(planning.distribution_networks):
        dn = list(t_net.active_distribution_network_nodes).index(node)
        adn_load = t_net.get_adn_load_idx(node)
        for p in t_model.periods:
            entry = common[(node, year, day, p)]
            pc_val = float(pe.value(t_model.pc[adn_load, 0, 0, p]))
            qc_val = float(pe.value(t_model.qc[adn_load, 0, 0, p]))
            dvar_p = t_model.interface_delta_p[dn, 0, 0, p]
            dvar_q = t_model.interface_delta_q[dn, 0, 0, p]
            req_p = entry['common_p'] - pc_val
            req_q = entry['common_q'] - qc_val
            excess_p = max(0.0, req_p - dvar_p.ub, dvar_p.lb - req_p)
            excess_q = max(0.0, req_q - dvar_q.ub, dvar_q.lb - req_q)
            midpoint_s_mva = ((entry['common_p'] ** 2 + entry['common_q'] ** 2) ** 0.5) * base_mva
            rating_mva = dvar_p.ub * base_mva  # rating is symmetric, p and q share it
            rows.append({
                'node': node, 'period': p,
                'required_delta_p_pu': req_p, 'delta_p_bound_pu': [dvar_p.lb, dvar_p.ub],
                'excess_p_pu': excess_p, 'excess_p_mva': excess_p * base_mva,
                'required_delta_q_pu': req_q, 'delta_q_bound_pu': [dvar_q.lb, dvar_q.ub],
                'excess_q_pu': excess_q, 'excess_q_mva': excess_q * base_mva,
                'midpoint_apparent_power_mva': midpoint_s_mva,
                'rating_mva': rating_mva,
                'utilization_ratio': (midpoint_s_mva / rating_mva) if rating_mva else None,
            })
    return rows


# ==============================================================================
#  polish, instrumented
# ==============================================================================
def _polish_one_block_instrumented(network_obj, model, params, agent, year, day):
    """`network.run_smopf`, production's own entry point, UNCHANGED options
    (case-file bound push -- the exact-fix's own semantics, NOT `p515_s41_
    hull_polish.py`'s IPOPT-default-push departure). Captures per-block
    stdout (not globally redirected) to classify the final attempt, then
    parses that attempt's own IPOPT log tail."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = network_obj.run_smopf(model, params, print_header=False)
    captured = buf.getvalue()
    ok = bool(srp._solver_result_succeeded(result))
    log_suffix, attempt_label = _classify_final_attempt(captured)
    log_path = _expected_log_path(network_obj, params, log_suffix=log_suffix)
    ipopt_log = _parse_ipopt_log_tail(log_path)
    diagnostics = {
        'block': f'{agent}|{year}|{day}', 'agent': agent, 'year': year, 'day': day,
        'solved': ok,
        'final_attempt': attempt_label,
        'termination_condition': (str(result.solver.termination_condition)
                                  if result is not None and hasattr(result, 'solver') else None),
        'solver_status': (str(result.solver.status)
                          if result is not None and hasattr(result, 'solver') else None),
        'solver_result_summary': solver_result_summary(result),
        'ipopt_log': ipopt_log,
        'constraint_violation_mva': (
            ipopt_log['constraint_violation_unscaled'] * network_obj.baseMVA
            if ipopt_log.get('constraint_violation_unscaled') is not None else None),
    }
    if not ok:
        families = O.scan_constraints(model, descend=False)
        worst = sorted(families.items(), key=lambda kv: kv[1]['max_violation'], reverse=True)[:5]
        diagnostics['largest_violated_rows'] = [
            {'family': name, 'max_violation': info['max_violation'],
             'worst_index': info['worst_index'], 'n': info['n']}
            for name, info in worst]
    return result, ok, diagnostics


def _polish_all_blocks_instrumented(planning, models, consensus_vars):
    """Same shape as `p515_s40_polish_gap._polish_all_blocks`, instrumented
    per block (see module docstring). Mutates `models` in place (these are
    discarded by the caller regardless, same convention `p55d_d1_polish.py`/
    `p515_s40_polish_gap.py` use)."""
    before_recourse = planning.get_operational_recourse_components(models)
    before_per_block = O.per_block_base_objectives(planning, models)

    common = O.common_coordinated_values(planning, models, consensus_vars)
    O.apply_common_values(planning, models, common)

    interface_diagnostics_by_block = {}
    for year in planning.transmission_network.years:
        for day in planning.transmission_network.days:
            interface_diagnostics_by_block[f'{year}|{day}'] = _tso_interface_diagnostics(
                planning, models, common, year, day)

    guard = SolveProfileGuard(POLISH_PERMITTED, label='P5.15-S42 exact-fix rerun').install()
    blocks = []
    try:
        for tag, holder in O._tagged_holders(planning):
            node_of = None if tag == 'TSO' else int(tag[3:])
            for year in holder.years:
                for day in holder.days:
                    network_obj = holder.network[year][day]
                    model = (models['tso'][year][day] if tag == 'TSO'
                             else models['dso'][node_of][year][day])
                    _switch_to_base_objective(model)
                    _result, ok, diag = _polish_one_block_instrumented(
                        network_obj, model, holder.params, tag, year, day)
                    blocks.append(diag)
    finally:
        guard.uninstall()

    all_solved = all(b['solved'] for b in blocks)
    after_per_block = O.per_block_base_objectives(planning, models)

    per_block_records = []
    for b in blocks:
        key = b['block']
        before_v = before_per_block[key]['weighted_base_objective']
        after_v = after_per_block[key]['weighted_base_objective'] if b['solved'] else None
        record = dict(b)
        record['weighted_base_objective_before'] = before_v
        record['weighted_base_objective_after'] = after_v
        record['delta'] = (after_v - before_v) if after_v is not None else None
        if b['agent'] == 'TSO':
            record['node7_interface_diagnostics'] = [
                r for r in interface_diagnostics_by_block[f"{b['year']}|{b['day']}"]
                if r['node'] == 7]
            record['max_utilization_ratio_all_nodes'] = max(
                (r['utilization_ratio'] for r in interface_diagnostics_by_block[
                    f"{b['year']}|{b['day']}"] if r['utilization_ratio'] is not None),
                default=None)
        per_block_records.append(record)

    failed_blocks = [r['block'] for r in per_block_records if not r['solved']]

    gate = None
    after_recourse = None
    if all_solved:
        after_recourse = planning.get_operational_recourse_components(models)
        recourse_before = before_recourse['gross_operational_cost']
        recourse_after = after_recourse['gross_operational_cost']
        delta = recourse_after - recourse_before
        relative = abs(delta) / abs(recourse_before) if recourse_before else None
        gate = {
            'objective_convention': 'gross_operational_cost -- settlement-excluded system cost',
            'recourse_before': recourse_before, 'recourse_after': recourse_after,
            'delta': delta, 'relative_pct': (relative * 100.0) if relative is not None else None,
            'threshold_pct': GATE_THRESHOLD_PCT,
            'pass': (relative is not None and relative < GATE_THRESHOLD_PCT / 100.0),
            'note': 'informational only for this run -- NOT the point of the exact-fix rerun, '
                    'which is the prediction score below (evaluated only because, contrary to '
                    'prediction, ALL 48 blocks happened to solve).',
        }

    return {
        'before_recourse_components': before_recourse,
        'after_recourse_components': after_recourse,
        'all_solved': all_solved,
        'failed_blocks': failed_blocks,
        'n_blocks': len(blocks),
        'per_block': per_block_records,
        'solve_profile': {
            'observed': dict(guard.counts),
            'n_blocks_dispatched': len(blocks),
            'retries_beyond_one_per_block': guard.counts['permitted_solve'] - len(blocks),
            'blocked_calls': guard.counts['blocked_solve'] + guard.counts['blocked_exec'],
        },
        'gate': gate,
        'mva_conversion_note': MVA_CONVERSION_NOTE,
    }


def _score_predictions(per_block_records):
    by_block = {r['block']: r for r in per_block_records}
    rows = []
    for block in SPRING_SUMMER_TSO_BLOCKS:
        r = by_block.get(block)
        cv_mva = r.get('constraint_violation_mva') if r else None
        predicted = 'infeasible, violation <= 5e-4 MVA'
        observed_infeasible = (r is not None) and (not r['solved'])
        violation_ok = (cv_mva is not None) and (cv_mva <= 5e-4)
        rows.append({'block': block, 'category': 'spring_summer_predicted_infeasible',
                     'predicted': predicted, 'solved': r['solved'] if r else None,
                     'constraint_violation_mva': cv_mva,
                     'matches_prediction': bool(observed_infeasible and violation_ok)})
    for block in AUTUMN_WINTER_TSO_BLOCKS:
        r = by_block.get(block)
        rows.append({'block': block, 'category': 'autumn_winter_predicted_solve',
                     'predicted': 'solves', 'solved': r['solved'] if r else None,
                     'constraint_violation_mva': r.get('constraint_violation_mva') if r else None,
                     'matches_prediction': bool(r is not None and r['solved'])})
    for block in DSO_2025_BLOCKS_NO_PREDICTION:
        r = by_block.get(block)
        rows.append({'block': block, 'category': 'dso_2025_no_prediction',
                     'predicted': None, 'solved': r['solved'] if r else None,
                     'constraint_violation_mva': r.get('constraint_violation_mva') if r else None,
                     'matches_prediction': None})
    n_scored = sum(1 for r in rows if r['matches_prediction'] is not None)
    n_match = sum(1 for r in rows if r['matches_prediction'] is True)
    all_12_tso_solve = all(r['solved'] for r in rows
                           if r['category'] in ('spring_summer_predicted_infeasible',
                                                'autumn_winter_predicted_solve'))
    return {
        'rows': rows, 'n_predictions_scored': n_scored, 'n_predictions_matched': n_match,
        'all_predictions_matched': (n_match == n_scored),
        'all_12_tso_solve': all_12_tso_solve,
        'on_all_12_tso_solve_note': (
            'per spec v13: re-examine what quantity the rating-midpoint check '
            '(utilization 1.000005) measures.' if all_12_tso_solve else None),
    }


def _hash_and_size(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)


def _persist_certified_models(models, out_dir):
    """Optional (spec v13 `persist_certified_models`): pickle the certified
    TSO/DSO models BEFORE polishing mutates them, hash-recorded, so a later
    polish variant needs no further ~80-minute reproduction. The SAME
    dict-of-live-Pyomo-blocks structure `WORKER_REPORT_S36_CLONE_CAPTURE.md`
    already confirmed unpickles (its 7 preserved fixtures)."""
    path = os.path.join(out_dir, 'certified_models.pkl')
    _refuse_overwrite(path)
    payload = {'tso': models['tso'], 'dso': models['dso']}
    with open(path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sha256, size = _hash_and_size(path)
    return {'path': os.path.relpath(path, REPO), 'sha256': sha256, 'size_bytes': size}


# ==============================================================================
#  post-run hook / main
# ==============================================================================
def _make_post_run_hook(out_dir, label, floor_rows_by_node, floor_sidecar_path,
                        recourse_jump_path, ess_stride_path, pf_stride_path,
                        exempt_until_state_path, persist_models, result_holder):
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
            result_holder['prediction_score'] = None
            result_holder['stopped_before_polish'] = True
            print(f"[S42-EXACT-FIX-RERUN] REPRODUCTION CHECK FAILED (mode={repro['mode']}): "
                  f"{repro['n_diffs']} diffs. STOPPING before polish. First diffs:")
            for d in repro['first_diffs']:
                print(f'  [FIRST DIFFS] {d}')
            return

        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('S42 exact-fix rerun: state/consensus_vars not available to '
                               'post_run_hook -- cannot fix consensus.')

        if persist_models:
            print('[S42-EXACT-FIX-RERUN] reproduction OK; persisting certified models '
                  'before polishing ...', flush=True)
            result_holder['persisted_models'] = _persist_certified_models(models, out_dir)
        else:
            result_holder['persisted_models'] = None

        print('[S42-EXACT-FIX-RERUN] exact-fix (midpoint) polishing 48 network blocks, '
              'FIXED helper ...', flush=True)
        polish_started = time.time()
        polish = _polish_all_blocks_instrumented(planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - polish_started
        result_holder['polish'] = polish
        result_holder['stopped_before_polish'] = False
        result_holder['prediction_score'] = _score_predictions(polish['per_block'])
        gate = polish['gate']
        if gate is not None:
            print(f"[S42-EXACT-FIX-RERUN] (informational) delta gate: "
                  f"relative={gate['relative_pct']}% pass={gate['pass']}", flush=True)
        else:
            print(f"[S42-EXACT-FIX-RERUN] delta gate NOT evaluated -- polish failed on blocks: "
                  f"{polish['failed_blocks']}", flush=True)
        ps = result_holder['prediction_score']
        print(f"[S42-EXACT-FIX-RERUN] PREDICTION SCORE: {ps['n_predictions_matched']}/"
              f"{ps['n_predictions_scored']} matched; all_12_tso_solve={ps['all_12_tso_solve']}",
              flush=True)
    return _hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-cycles', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--persist-certified-models', action='store_true', default=False)
    args = parser.parse_args()

    is_smoke = args.smoke_cycles is not None
    sfx = ('_' + args.suffix.strip('_')) if args.suffix.strip('_') else ''
    out_dir = (OUT_DIR_SMOKE if is_smoke else OUT_DIR_FULL) + sfx
    num_max_iters = args.smoke_cycles if is_smoke else FULL_NUM_MAX_ITERS
    run_eval_id = ('p515s42_exact_fix_rerun_smoke_run' if is_smoke
                  else 'p515s42_exact_fix_rerun_run') + sfx
    precheck_eval_id = run_eval_id + '_precheck'

    failures = _check_preconditions(out_dir)
    if failures:
        for f in failures:
            print(f'[S42-EXACT-FIX-RERUN PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S42-EXACT-FIX-RERUN] preconditions passed (no lock, no forbidden process, '
          "fresh output dir, production files clean, D's reference present).")

    G._acquire_exclusive_run_lock()
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    _capture_checklist, floor_rows_by_node, floor_counts_by_node = _build_floor_rows(
        precheck_eval_id)
    print(f'[S42-EXACT-FIX-RERUN] soh_floor_row_counts_by_node='
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
                               exempt_until_state_path, args.persist_certified_models,
                               result_holder)

    print(f'[S42-EXACT-FIX-RERUN] run CASE-FILE-ALONE (label={ARM_LABEL!r}, apply_rho=False, '
          f'pre_solve_hook=None), num_max_iters_override={num_max_iters} '
          f"({'SMOKE' if is_smoke else 'FULL -- to Boyd certification'}).")
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(exempt_until_state_path):
        report, report_path = G.run_admm_arm(
            ARM_LABEL, out_dir, k_override=None, eval_id=run_eval_id,
            num_max_iters_override=num_max_iters, apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=hook, pre_solve_hook=None)

    print(f"[S42-EXACT-FIX-RERUN] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f"wall={report['wall_clock_s']:.1f}s")

    payload = {
        'stage': 'P5.15 Addendum 24 item 1 -- exact-fix (midpoint) rerun with the fixed helper',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 24',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm_label': ARM_LABEL,
        'mode': 'smoke' if is_smoke else 'full',
        'smoke_cycles': args.smoke_cycles,
        'num_max_iters_used': num_max_iters,
        'is_prediction_test': (not is_smoke),
        'out_dir': os.path.relpath(out_dir, REPO),
        'admm_report_path': os.path.relpath(report_path, REPO),
        'admm_cycles_run': report['cycles_run'],
        'admm_gross_operational_cost': report['gross_operational_cost'],
        'reproduction': result_holder.get('reproduction'),
        'stopped_before_polish': result_holder.get('stopped_before_polish'),
        'persisted_models': result_holder.get('persisted_models'),
        'polish': result_holder.get('polish'),
        'prediction_score': result_holder.get('prediction_score'),
        'predictions_recorded_in_advance': {
            'spring_summer_tso_blocks': list(SPRING_SUMMER_TSO_BLOCKS),
            'spring_summer_prediction': 'local infeasibility, constraint violation <= 5e-4 MVA',
            'autumn_winter_tso_blocks': list(AUTUMN_WINTER_TSO_BLOCKS),
            'autumn_winter_prediction': 'solves',
            'dso_2025_blocks_no_prediction': list(DSO_2025_BLOCKS_NO_PREDICTION),
        },
        'mva_conversion_note': MVA_CONVERSION_NOTE,
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(out_dir, 'exact_fix_rerun_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S42-EXACT-FIX-RERUN] wrote {results_path}')

    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_dir, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-EXACT-FIX-RERUN] wrote {manifest_path}')

    if not result_holder.get('reproduction', {}).get('reproduces'):
        print('[S42-EXACT-FIX-RERUN] *** REPRODUCTION CHECK FAILED -- stopped before polish. '
              f'See {results_path}. ***')
        sys.exit(1)

    ps = result_holder.get('prediction_score')
    if ps is None:
        print('[S42-EXACT-FIX-RERUN] *** prediction score not evaluated (polish did not run). '
              f'See {results_path}. ***')
        sys.exit(1)

    if is_smoke:
        print('[S42-EXACT-FIX-RERUN] SMOKE MODE: prediction score is NOT the prediction test '
              f"(2-cycle point, not D's certified point). n_matched={ps['n_predictions_matched']}"
              f"/{ps['n_predictions_scored']} reported for mechanics only.")
    else:
        print(f"[S42-EXACT-FIX-RERUN] PREDICTION TEST: "
              f"{ps['n_predictions_matched']}/{ps['n_predictions_scored']} matched.")


if __name__ == '__main__':
    main()
