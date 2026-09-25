"""
P5.15 Addendum 45 item 1 (Planner task W78) -- THE SCALING-PIN TEST: two-cycle SRP1 arms at x = 0 and at the
smallest node-7 unit, under production (gradient-based) scaling and under `nlp_scaling_method = user-scaling` with
ONE `obj_scaling_factor` in {0.001, 0.002, 0.003, 0.005} for every TSO and DSO network solve.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 45 ("the pin is `nlp_scaling_method = user-scaling` with one
`obj_scaling_factor` for every network solve in every cell ...; the value chosen by a two-cycle SRP1 test between
0.001 and 0.005 (both cells converge, similar iterations); production tolerances unchanged -- matching, not
tightening. Fallback if user-scaling degrades conditioning (constraint scaling reverts to 1): pure (c)");
Planner task W78. Frozen spec v26 (predecessor v25 407a4b33), written by `--freeze-spec` BEFORE any arm runs, holds
every criterion, tolerance, formula and prediction.

WHAT RUNS (per arm, ONE arm per process, so the armed guard is verified EXACTLY per arm)
------------------------------------------------------------------------------------------
The committed SRP1 two-cycle machinery BY IMPORT, exactly as the SRP1 bitwise gate uses it
(`p515_s50_generalization_gate.main` -> `W10.run_arm(ARM, 'on', ...)`):
  * `W10 = p515_s45_snapshot_off_two_cycle_gate` -- its armed `SolveProfileGuard` (bounded, `N.PERMITTED` call
    sites) is installed at import and stays armed for the whole process; `W10.check_preconditions`,
    `W10.capture_path_checklist`, `W10.derive_solves_from_case_file` (51 solves/cycle; base 51 x (cap 2 + 1) = 153),
    `W10.run_arm(arm, 'on', ...)` (snapshots 'on' = no assignment = the case-file default, as the SRP1 gate);
  * `W32 = p515_s49_memory_fix_gate` -- `BASELINE_REFERENCE` (the pinned reference `check_preconditions` verifies)
    and `_load_recert_declaration` (the recert ESS-ageing declaration the SRP1 gate injects into the child hook);
  * `S44.event_level_solve_reconciliation` -- the per-EVENT solve identity; `GUARD.verify(that)` EXACTLY.
DECLARED SUBSTITUTIONS (and no others):
  1. `W10.AA_REFERENCE := W32.BASELINE_REFERENCE` (as W32/W35 do) -- only so `check_preconditions` pins a committed
     file; no comparison against it is made (the candidates are not C*).
  2. `W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS` extended with the p515_s46_ ... p515_s53_ prefixes.
  3. `W10.C_STAR := the cell's candidate` AFTER `check_preconditions` (which checks the C* key pin on the original
     constant) and BEFORE `run_arm` -- `run_arm` builds the candidate from this module global. The two cells are
     asserted equal to the committed canonical forms (x0: s45_a0_c7 spec 9d08ad2f; n7_4h_e1: s47_recert spec
     902f93aa).
  4. `H._config_hook_factory` wrapped (as W35 wraps it): the spec it receives also declares the recert ESS-ageing
     baseline; the hook it returns calls the committed hook FIRST, then (pinned arms only) sets
     `solver_params.options['nlp_scaling_method'] = 'user-scaling'` and `['obj_scaling_factor'] = v` on the TSO
     and on every DSO network holder's params, READ BACK and recorded. Baseline arms set nothing (recorded).
     Recovery tiers inherit the pin: `network._create_smopf_solver` merges `solver_params.options` first and the
     retry `option_overrides` (acceptable_tol / acceptable_iter / warm_start_init_point / mu_strategy) after;
     neither touches a scaling key (asserted before the run).
THE ESSO IS NOT PINNED: it is not a network solve (Addendum 45 names network solves), its objective carries the
oracle's own D5 scaling (sigma fixed, S_ref 2.5 MVA) and no term of the network cost Q; its options are recorded
before and after the arm and must be unchanged.

THE DECISIVE CHECK (C0; per pinned arm, from the IPOPT logs, on EVERY TSO/DSO network solve incl. retries)
----------------------------------------------------------------------------------------------------------
IPOPT (3.14.18, StandardScalingBase) sets df = obj_scaling_factor x df_method, where df_method is the gradient-based
factor ("Scaling parameter for objective function = ..." is printed ONLY by the gradient-based method) or, under
user-scaling through the AMPL interface, the objective's `scaling_factor` suffix (1.0 when absent). It prints the
product as "objective scaling factor = %g" at print level >= 6 (every network case file sets file_print_level 6 --
asserted before the run). C0 holds for an arm iff, for every network solve segment: the options list shows
nlp_scaling_method = user-scaling and obj_scaling_factor = v (each used >= 1 time); the printed factor parses to
EXACTLY v and its text equals format(v, 'g'); no gradient-based line is printed; and the number of parsed
network segments equals the guard's network-solve count (too few fails as loudly as too many). If C0 fails for any
pinned arm the harness exits 2 and the test STOPS (no value is chosen).

MODES (all zero-solve except --arm)
-----------------------------------
  --freeze-spec     ZERO SOLVES. Calibration from committed-run IPOPT logs (read-only, hash-recorded); writes
                    data/SRP1/Results/P515S53/frozen_s53_spec_v26_<sha8>.json (write-once, named by its sha256;
                    predecessor v25 407a4b33 must be tracked, clean and hash as pinned). GUARD.verify(0).
  --arm NAME        ONE arm (153 + attempted retries solves), then the zero-solve log parse and C0.
  --analyse         ZERO SOLVES. Reads the ten arm results and applies the frozen criteria C1-C3 and the
                    selection rule. GUARD.verify(0).

EXACT LAUNCH COMMANDS (repo root; attached, ALONE, both streams captured; never detached; sequential):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w78_scaling_pin_test.py --freeze-spec \\
        > data/SRP1/Results/P515S53/scaling_pin_w78_freeze_launch.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w78_scaling_pin_test.py --arm <ARM> \\
        > data/SRP1/Results/P515S53/scaling_pin_w78/launch_logs/<ARM>.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w78_scaling_pin_test.py --analyse \\
        > data/SRP1/Results/P515S53/scaling_pin_w78/launch_logs/analyse.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/scaling_pin_w78/{arms/<ARM>/{arm/, arm_result.json,
manifest_sha256.json}, analysis/{analysis.json, analysis.md, manifest_sha256.json}, launch_logs/}
"""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s49_memory_fix_gate as W32  # noqa: E402 -- imports W10 first (installs W10.GUARD, armed)

W10 = W32.W10
GUARD = W10.GUARD
H = W10.H
G = W10.G
CP = W10.CP

import p515_s44_scale_measurement as S44  # noqa: E402
import network as NET  # noqa: E402

STAGE = ('P5.15 Addendum 45 item 1, W78 -- the scaling-pin test: two-cycle SRP1 arms, x = 0 and the smallest node-7 '
         'unit, gradient-based vs user-scaling with one obj_scaling_factor in {0.001, 0.002, 0.003, 0.005}')
SCHEMA = 'p515_s53_w78_scaling_pin_test_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 45 (ruling (c) + reference re-run; order item 1: pin test)',
             'Planner task W78']

_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_REL = os.path.join(_P53, 'scaling_pin_w78')
LAUNCH_LOGS_REL = os.path.join(OUT_REL, 'launch_logs')
SPEC_V25 = {'path': os.path.join(_P53, 'frozen_s53_spec_v25_407a4b33.json'),
            'sha256': '407a4b330ba1ac40039f6f94a3616ffde2d7246c5fb6754114f69083eb21dcde'}
SPEC_V26_PREFIX = 'frozen_s53_spec_v26_'

# ---- the instance: two cells, committed canonical forms ----------------------------------------------------------
CELLS = {
    'x0': {'nodes': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)},
           'label': 'x = 0 (no storage at nodes 5, 7, 9)',
           'committed_canonical': {'spec': os.path.join('data', 'SRP1', 'Results', 'P515S45', 'campaign_s45_a0_c7',
                                                        'campaign_spec_s45_a0_c7_9d08ad2f.json'),
                                   'label': 'x0'}},
    'n7u': {'nodes': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)},
            'label': 'the smallest node-7 unit: 0.25 MVA / 1.0 MWh at node 7, investment year 2025 (n7_4h_e1)',
            'committed_canonical': {'spec': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert',
                                                         'campaign_spec_s47_recert_902f93aa.json'),
                                    'label': 'n7_4h_e1'}},
}
PIN_VALUES = (0.001, 0.002, 0.003, 0.005)


def _vtag(v):
    return 'grad' if v is None else 'us' + format(v, 'g').replace('.', 'p')


ARMS = {}
for _cell in ('x0', 'n7u'):
    for _v in (None,) + PIN_VALUES:
        ARMS[f's53w78_{_cell}_{_vtag(_v)}'] = {'cell': _cell, 'value': _v}
DECISIVE_FIRST_ARM = 's53w78_x0_us0p001'

NETWORK_FAMILY = {'case9': 'TSO', 'case33_1': 'DSO5', 'case33_2': 'DSO7', 'case33_3': 'DSO9'}
FAMILIES = ('TSO', 'DSO5', 'DSO7', 'DSO9')
PROTECTED_OPTION_KEYS = ('tol', 'acceptable_tol', 'acceptable_iter', 'compl_inf_tol', 'max_iter', 'linear_solver',
                         'constr_viol_tol', 'dual_inf_tol', 'acceptable_constr_viol_tol', 'mu_strategy',
                         'bound_push', 'bound_frac', 'slack_bound_push', 'slack_bound_frac', 'bound_relax_factor',
                         'honor_original_bounds', 'nlp_scaling_max_gradient', 'nlp_scaling_min_value',
                         'fixed_variable_treatment')
SCALING_KEYS = ('nlp_scaling_method', 'obj_scaling_factor')
EXTRA_FORBIDDEN = ('p515_s46_', 'p515_s47_', 'p515_s48_', 'p515_s49_', 'p515_s50_', 'p515_s51_', 'p515_s52_',
                   'p515_s53_')
EXTRA_CLEAN_FILES = ('network.py', 'solver_parameters.py', 'model_construction_helpers.py',
                     'shared_resources_planning.py', 'shared_energy_storage_data.py', 'p56a_oracle.py',
                     'p515_g_g1_g4_admm_gates.py', 'p515_s44_scale_measurement.py', 'p515_s44_campaign_harness.py',
                     'p515_s45_snapshot_off_two_cycle_gate.py', 'p515_s49_memory_fix_gate.py',
                     'data/SRP1/SRP1.json', 'data/SRP1/SRP1_params.json',
                     'data/SRP1/case9/case9_params.json', 'data/SRP1/case33_1/case33_1_params.json',
                     'data/SRP1/case33_2/case33_2_params.json', 'data/SRP1/case33_3/case33_3_params.json',
                     'data/SRP1/SharedESS/SRP1_ESS_Params.json', os.path.basename(__file__))

# Calibration (zero-solve, --freeze-spec): committed-run IPOPT logs of the same two candidates at production
# scaling, first three solves (init + cycles 1, 2) per block log. DIFFERENT configurations from this test's
# baseline arms (x0: s45_a0_c7; unit: s47_a1a_baseline) -- used only to state predictions.
CALIBRATION_RUNS = {
    'x0': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', 'p515s44_s45_a0_c7_7aa017f09989b56d_run', 'logs'),
    'n7u': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                        'p515s44_s47_a1a_baseline_bd504ecf5a288d44_run', 'logs'),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W78] {msg}', flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _sha(path):
    return CP._sha256_file(_abs(path))


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _git_tracked_clean(rel):
    tracked = bool(_git(['ls-files', '--', rel]))
    clean = not _git(['status', '--porcelain', '--', rel])
    return tracked, clean


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except ImportError:
        pass
    return str(o)


def _write_once(path, payload):
    with open(path, 'x') as handle:
        json.dump(payload, handle, indent=1, default=_json_default)
        handle.write('\n')


# ======================================================================================================================
#  the IPOPT log parser (zero-solve)
# ======================================================================================================================
_ITER_HEADER = re.compile(r'^iter\s+objective\s+inf_pr\s+inf_du\s+lg\(mu\)\s+\|\|d\|\|\s+lg\(rg\)')
_ITER_LINE = re.compile(r'^\s*(\d+)(r?)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)')
_OPTION_LINE = re.compile(r'^\s*(\S+) = (.*?)\s+(\d+)\s*$')


def _num_or_str(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def parse_ipopt_log(path):
    """Split one IPOPT output_file (file_append = yes) into solve segments at each 'List of options:' (printed
    before each solve's banner) and extract, per segment, everything the frozen spec's metrics need. Every
    statistic's formula is in the spec (`formulas`)."""
    with open(_abs(path), errors='replace') as handle:
        text = handle.read()
    chunks = text.split('List of options:')
    if chunks and chunks[0].strip():
        raise RuntimeError(f'{path}: text before the first options list')
    segments = []
    for index, seg in enumerate(chunks[1:]):
        lines = seg.splitlines()
        options = {}
        for line in lines[:80]:
            if 'This program contains Ipopt' in line or line.startswith('*****'):
                break
            m = _OPTION_LINE.match(line)
            if m and m.group(1) != 'Name':
                options[m.group(1)] = {'value': m.group(2).strip(), 'times_used': int(m.group(3))}
        iters = []
        for i, line in enumerate(lines):
            if _ITER_HEADER.match(line) and i + 1 < len(lines):
                m = _ITER_LINE.match(lines[i + 1])
                if m:
                    # groups: 1 iter, 2 'r', 3 objective, 4 inf_pr, 5 inf_du, 6 lg(mu), 7 ||d||, 8 lg(rg),
                    # 9 alpha_du, 10 alpha_pr(+type), 11 ls
                    iters.append({'k': int(m.group(1)), 'restoration': m.group(2) == 'r', 'lg_mu': m.group(6),
                                  'lg_rg': m.group(8)})
        restoration_entries = sum(1 for a, b in zip([{'restoration': False}] + iters, iters)
                                  if b['restoration'] and not a['restoration'])
        perturb = re.findall(r'Perturbation parameters: delta_x=(\S+) delta_s=(\S+)', seg)

        def first(p):
            m = re.search(p, seg)
            return m.group(1) if m else None

        def last(p):
            m = re.findall(p, seg)
            return m[-1] if m else None

        n_iter = last(r'Number of Iterations\.+: (\d+)')
        tail = seg.split('Number of Iterations')[-1] if n_iter is not None else ''
        kkt = re.findall(r'Overall NLP error\.+:\s+(\S+)\s+(\S+)', tail)
        eff_lines = re.findall(r'^objective scaling factor = (\S+)\s*$', seg, flags=re.M)
        grad_lines = re.findall(r'^Scaling parameter for objective function = (\S+)\s*$', seg, flags=re.M)
        mu = last(r'Current barrier parameter mu = (\S+)')
        segments.append({
            'segment_index': index,
            'n_banners': seg.count('This is Ipopt version'),
            'options': options,
            'objective_scaling_factor_lines': eff_lines,
            'gradient_based_objective_scaling_lines': grad_lines,
            'x_scaling': first(r'(?m)^(No x scaling provided|x scaling provided)\s*$'),
            'c_scaling': first(r'(?m)^(No c scaling provided|c scaling provided)\s*$'),
            'd_scaling': first(r'(?m)^(No d scaling provided|d scaling provided)\s*$'),
            'iterations': int(n_iter) if n_iter is not None else None,
            'exit': last(r'EXIT: (.*)'),
            'final_mu': float(mu) if mu is not None else None,
            'n_iter_lines': len(iters),
            'n_iterations_with_inertia_regularisation': sum(1 for it in iters if it['lg_rg'] != '-'),
            'n_factorizations_with_delta_x_positive': sum(1 for dx, _ds in perturb if float(dx) > 0.0),
            'n_restoration_iterations': sum(1 for it in iters if it['restoration']),
            'n_restoration_entries': restoration_entries,
            'final_kkt_scaled': float(kkt[-1][0]) if kkt else None,
            'final_kkt_unscaled': float(kkt[-1][1]) if kkt else None,
        })
    return segments


def _block_logs(logs_dir, net, year, day):
    base = os.path.join(logs_dir, f'optim_log_{net}_{year}_{day}')
    return {'primary': base + '.log', 'recovery': base + '_recovery.log', 'recovery_tier2': base + '_recovery_tier2.log'}


ACCEPTABLE_EXITS = ('Optimal Solution Found.', 'Solved To Acceptable Level.')


def _median(xs):
    return statistics.median(xs) if xs else None


def _p90(xs):
    """nearest-rank p90: sorted(xs)[ceil(0.9 n) - 1]"""
    if not xs:
        return None
    s = sorted(xs)
    return s[math.ceil(0.9 * len(s)) - 1]


def family_stats(records):
    """Per family, over PRIMARY solves (init + cycle 1 + cycle 2). Formulas in the spec."""
    out = {}
    for fam in FAMILIES:
        rs = [r for r in records if r['family'] == fam and r['attempt'] == 'primary']
        its = [r['iterations'] for r in rs if r['iterations'] is not None]
        mus = [r['final_mu'] for r in rs if r['final_mu'] is not None]
        kkts = [r['final_kkt_scaled'] for r in rs if r['final_kkt_scaled'] is not None]
        kktu = [r['final_kkt_unscaled'] for r in rs if r['final_kkt_unscaled'] is not None]
        exits = {}
        for r in rs:
            exits[r['exit']] = exits.get(r['exit'], 0) + 1
        out[fam] = {
            'n_primary_solves': len(rs), 'n_with_iterations': len(its),
            'iterations_median': _median(its), 'iterations_p90': _p90(its),
            'iterations_max': max(its) if its else None, 'iterations_total': sum(its),
            'iterations_by_round_median': {str(k): _median([r['iterations'] for r in rs if r['round'] == k
                                                            and r['iterations'] is not None]) for k in (0, 1, 2)},
            'final_mu_median': _median(mus), 'final_mu_max': max(mus) if mus else None,
            'final_kkt_scaled_median': _median(kkts), 'final_kkt_scaled_max': max(kkts) if kkts else None,
            'final_kkt_unscaled_median': _median(kktu), 'final_kkt_unscaled_max': max(kktu) if kktu else None,
            'iterations_with_inertia_regularisation_total': sum(r['n_iterations_with_inertia_regularisation']
                                                                for r in rs),
            'factorizations_with_delta_x_positive_total': sum(r['n_factorizations_with_delta_x_positive'] for r in rs),
            'restoration_entries_total': sum(r['n_restoration_entries'] for r in rs),
            'restoration_iterations_total': sum(r['n_restoration_iterations'] for r in rs),
            'exits': exits,
            'n_optimal': exits.get('Optimal Solution Found.', 0),
            'n_acceptable_level': exits.get('Solved To Acceptable Level.', 0),
            'n_non_acceptable': sum(v for k, v in exits.items() if k not in ACCEPTABLE_EXITS),
            'n_max_iter': exits.get('Maximum Number of Iterations Exceeded.', 0),
            'objective_scaling_factor_distinct': sorted({x for r in rs for x in r['objective_scaling_factor_lines']}),
            'gradient_based_factor_distinct': sorted({x for r in rs for x in r['gradient_based_objective_scaling_lines']}),
            'c_scaling_distinct': sorted({str(r['c_scaling']) for r in rs}),
            'd_scaling_distinct': sorted({str(r['d_scaling']) for r in rs}),
        }
    return out


def parse_run_logs(logs_dir, years, days, n_rounds=3, primary_only=False):
    """Every network log of one run: records per solve segment, with block identity and attempt kind."""
    records, files, anomalies = [], {}, []
    for net, fam in NETWORK_FAMILY.items():
        for year in years:
            for day in days:
                for attempt, path in _block_logs(logs_dir, net, year, day).items():
                    if attempt != 'primary' and primary_only:
                        continue
                    if not os.path.isfile(path):
                        if attempt == 'primary':
                            anomalies.append(f'missing primary log {path}')
                        continue
                    files[os.path.relpath(path, REPO)] = _sha(path)
                    segs = parse_ipopt_log(path)
                    if attempt == 'primary' and primary_only:
                        segs = segs[:n_rounds]
                    elif attempt == 'primary' and len(segs) != n_rounds:
                        anomalies.append(f'{path}: {len(segs)} segments != {n_rounds}')
                    for s in segs:
                        if s['n_banners'] != 1:
                            anomalies.append(f"{path} segment {s['segment_index']}: {s['n_banners']} banners")
                        s.update({'network': net, 'family': fam, 'year': str(year), 'day': str(day),
                                  'attempt': attempt, 'log': os.path.relpath(path, REPO),
                                  'round': s['segment_index'] if attempt == 'primary' else None})
                        records.append(s)
    return records, files, anomalies


# ======================================================================================================================
#  calibration (zero-solve) and the frozen spec v26
# ======================================================================================================================
def calibrate():
    with open(W10.CASE_JSON) as handle:
        case = json.load(handle)
    years, days = sorted(case['Years']), sorted(case['Days'])
    out = {}
    for cell, rel in CALIBRATION_RUNS.items():
        records, files, anomalies = parse_run_logs(_abs(rel), years, days, primary_only=True)
        stats = family_stats(records)
        out[cell] = {'logs_dir': rel, 'n_logs': len(files), 'logs_sha256': files, 'anomalies': anomalies,
                     'n_segments': len(records),
                     'family_stats': {f: {k: stats[f][k] for k in (
                         'iterations_median', 'iterations_p90', 'iterations_max',
                         'iterations_with_inertia_regularisation_total', 'restoration_entries_total', 'exits',
                         'objective_scaling_factor_distinct', 'gradient_based_factor_distinct',
                         'c_scaling_distinct', 'd_scaling_distinct', 'final_mu_median')} for f in FAMILIES}}
    return out


FORMULAS = {
    'segment': "one IPOPT solve = the text between consecutive 'List of options:' markers of a network output_file "
               "(file_append yes; the options list precedes each solve's banner); exactly one banner per segment",
    'round': 'primary-log segment index: 0 = initialisation solve, 1 = ADMM cycle 1, 2 = ADMM cycle 2',
    'iterations': "'Number of Iterations....: N' (last occurrence in the segment)",
    'termination': "'EXIT: <message>' (last occurrence); acceptable = 'Optimal Solution Found.' or "
                   "'Solved To Acceptable Level.'; max_iter hit = 'Maximum Number of Iterations Exceeded.'",
    'final_mu': "last 'Current barrier parameter mu = X' in the segment (the W74 convention)",
    'inertia_regularisations': "number of iteration-summary lines (the line after each 'iter objective inf_pr ...' "
                               "header) whose lg(rg) column (delta_w) is not '-'; also reported: the number of "
                               "'Perturbation parameters: delta_x=X' lines with X > 0 (per factorization)",
    'restoration': "restoration iterations = iteration-summary lines whose iteration number carries the 'r' suffix; "
                   "restoration ENTRIES = transitions from a non-'r' (or the start) to an 'r' line",
    'final_kkt': "'Overall NLP error' (scaled, unscaled) of the final block after 'Number of Iterations'",
    'effective_objective_scaling': "'objective scaling factor = %g' (IPOPT StandardScalingBase: "
                                   "obj_scaling_factor x method factor); the gradient-based method factor is the "
                                   "separate line 'Scaling parameter for objective function = %e'",
    'median': 'statistics.median over the family\'s primary solves (12 blocks x 3 rounds = 36 per family per arm)',
    'p90': 'nearest rank: sorted(x)[ceil(0.9 n) - 1]',
    'family': 'TSO = case9; DSO5 = case33_1; DSO7 = case33_2; DSO9 = case33_3',
    'network_solves_in_guard': 'GUARD.permitted_solve - 9 ESSO solves (3 nodes x 3 rounds); ESSO recovery events '
                               'make the reconciliation unsupported (fails loudly)',
}

CRITERIA = {
    'C0_decisive_effective_factor': {
        'applies_to': 'pinned arms (value v)',
        'holds_iff': ["every network solve segment (primary + recovery + tier-2 logs) has options "
                      "nlp_scaling_method = 'user-scaling' and obj_scaling_factor == v (each times_used >= 1)",
                      "exactly one 'objective scaling factor = S' line with float(S) == v and S == format(v, 'g')",
                      "no 'Scaling parameter for objective function' line (the gradient-based factor is not computed)",
                      'number of network segments == GUARD network-solve count, and every primary block log holds '
                      'exactly 3 segments'],
        'on_failure': 'exit 2, STOP: no value is chosen; report to the Planner',
        'baseline_arms': "recorded only: options carry neither scaling key; the effective factor equals the "
                         "gradient-based factor on every segment",
    },
    'C0b_tolerances_unchanged': {
        'applies_to': 'every arm',
        'holds_iff': 'for every network segment, the PROTECTED option keys in the IPOPT options list equal the '
                     'case-file options (+ production defaults max_iter 500 and fixed_variable_treatment '
                     'make_parameter), with the production retry overrides on recovery segments (recovery_options '
                     "minus hessian_approximation; tier 2 adds mu_strategy adaptive); keys absent from that "
                     'expectation are absent from the list',
    },
    'C1_convergence': {
        'applies_to': 'each value v, both cells',
        'holds_iff': ['cycles_run == 2 and local_solve_failures == 0',
                      "zero network events of class 'unrecovered' or 'indeterminate'; zero ESSO recovery events",
                      'zero max_iter hits in any network attempt (primary or retry)',
                      "primary non-acceptable network terminations <= the same cell's baseline count",
                      'GUARD.verify(declared per-event count) == [] (exact)'],
        'reported': 'per arm and family: optimal / acceptable-level / other exits; retries; restoration entries',
    },
    'C2_similar_iterations_between_cells': {
        'applies_to': 'each value v (x0 arm vs n7u arm at the same v)',
        'holds_iff': 'for EVERY family F in {TSO, DSO5, DSO7, DSO9}: (|med_n7u - med_x0| <= 3 or '
                     'med_n7u / med_x0 in [0.80, 1.25]) AND (|p90_n7u - p90_x0| <= 4 or p90_n7u / p90_x0 in '
                     '[1/1.35, 1.35]), over the family\'s 36 primary solves per cell',
        'justification': ('the two cells are different instances (one 0.25 MVA unit at node 7 changes the TSO and '
                          'DSO7 problems), so identical counts are not expected even under identical scaling; the '
                          'calibration (spec calibration_zero_solve) shows the untouched DSO5 / DSO9 families within '
                          '3 % in median (46.5 / 46.5, 47 / 48) and 8 % at p90 (55 / 51, 59 / 58) across the cells, '
                          'DSO7 at 1.15 / 1.10 (48 / 55, 62 / 68), while the TSO family differs by 2.25 in median '
                          '(24 / 54) and 2.96 at p90 (26 / 77) under gradient-based scaling (0.002 vs 0.001). A 25 % '
                          'median band (35 % at p90, whose '
                          'nearest-rank value on 36 solves is one of the 4 largest and noisier) with absolute floors '
                          'of 3 / 4 iterations (integer granularity at medians of 20-60) admits genuine instance '
                          'differences and excludes the baseline disparity by a wide margin'),
    },
    'C3_no_conditioning_degradation': {
        'applies_to': 'each value v, each cell c, each family F, against the gradient-based arms',
        'degradation_if_any': [
            "restoration entries (total over the family's 36 primary solves) > the same cell's baseline total",
            'iterations with inertia regularisation (total) > max(1.25 x baseline_c_F, baseline_c_F + 5)',
            'acceptable-level (non-optimal) terminations > baseline_c_F + max(2, 0.25 x baseline_c_F)',
            'median iterations > 1.25 x max(baseline median x0_F, baseline median n7u_F), or p90 > 1.35 x '
            'max(baseline p90 x0_F, baseline p90 n7u_F) -- the matched level may reach the WORSE baseline cell '
            '(that is what matching means for the cell whose factor changes) but not exceed it by more than the '
            'C2 bands',
        ],
        'not_gated_reported': {
            'final_mu': 'reported per solve and summarised; NOT a degradation criterion: the barrier problem is posed '
                        'on the scaled objective, so mu at termination scales with the objective factor and '
                        'differs between arms by construction',
            'final_kkt_scaled': 'reported (scaled and unscaled); NOT a degradation criterion: the scaled error at an '
                                'optimal exit is <= tol by construction in every arm and its unscaled counterpart '
                                'scales with 1/df; the gated, scale-free indicators are restoration entries, '
                                'inertia regularisations, termination class and iteration counts',
        },
    },
    'selection_rule': {
        'eligible': 'values v passing C0, C0b, C1, C2 and C3',
        'choose': 'argmin D(v) = max over F of |ln(med_n7u_F(v) / med_x0_F(v))|; ties (|dD| <= 0.05) broken by the '
                  'smaller total network iterations over both cells, then by the smaller v',
        'outcomes': {
            'STOP': 'C0 fails for any pinned arm -- the pin does not pin; no value chosen',
            'RECOMMEND(v)': 'at least one eligible value',
            'FALLBACK_PURE_C': 'C1 or C3 fails for EVERY value (user-scaling degrades conditioning or convergence) '
                               '-- Addendum 45 fallback: gradient-based retained, sub-resolution caveat, no re-run',
            'NO_VALUE_MEETS_C2': 'some value passes C0, C0b, C1 and C3 but none passes C2 -- conditioning intact, '
                                 'the similarity criterion unmet; reported plainly for the Planner to rule',
        },
    },
}


def spec_v26_content(calibration):
    return {
        'schema': 'p515_frozen_spec_v26',
        'version': 26,
        'stage': STAGE,
        'authority': AUTHORITY,
        'predecessor': {'path': SPEC_V25['path'], 'sha256': SPEC_V25['sha256']},
        'predecessor_not_edited': 'v25 stays as frozen',
        'harness': {'path': os.path.basename(__file__), 'sha256_at_freeze': _sha(__file__),
                    'commit_at_freeze': _git(['log', '-1', '--format=%H', '--', os.path.basename(__file__)])},
        'instance': {
            'case_file': 'data/SRP1/SRP1.json (3 years x 4 days, 1 TSO + 3 DSOs, 1 x 1 scenario)',
            'cells': {c: {'label': CELLS[c]['label'],
                          'canonical': H.canonical_candidate(CELLS[c]['nodes']),
                          'candidate_key': H.candidate_key(H.canonical_candidate(CELLS[c]['nodes'])),
                          'committed_canonical_source': CELLS[c]['committed_canonical']} for c in CELLS},
        },
        'configuration': {
            'machinery': 'W10.run_arm(arm, "on", ...) as the SRP1 bitwise gate (p515_s50_generalization_gate) runs it: '
                         'cap 2, arm label s39_D, apply_rho False, full_diagnostics_in_rows True, snapshots on (no '
                         'assignment = the case-file default), the case-file D oracle with AA keep_memory verified '
                         'by the campaign hook, the recert ESS-ageing declaration injected and verified',
            'baseline_arms': 'production: no scaling key set -> IPOPT default nlp_scaling_method gradient-based, '
                             'obj_scaling_factor 1',
            'pinned_arms': "solver_params.options['nlp_scaling_method'] = 'user-scaling' and "
                           "['obj_scaling_factor'] = v on the TSO holder and on each DSO holder, after the committed "
                           'hook, in the harness only (no case-file edit); retries inherit it',
            'values': list(PIN_VALUES),
            'esso': 'NOT pinned: not a network solve; its own D5 objective scaling; options recorded before/after '
                    'and required unchanged',
            'production_tolerances_unchanged': 'tol, acceptable_tol, acceptable_iter, compl_inf_tol, max_iter and the '
                                               'linear solver are not touched (C0b verifies from every log)',
            'thread_caps': dict(H.THREAD_CAP_ENV),
        },
        'arms': {name: {'cell': a['cell'], 'value': a['value'],
                        'eval_ids': H.eval_ids(f'{W10.CAMPAIGN_ID}_{name}',
                                               H.candidate_key(H.canonical_candidate(CELLS[a['cell']]['nodes'])))}
                 for name, a in ARMS.items()},
        'run_order': [DECISIVE_FIRST_ARM] + [n for n in ARMS if n != DECISIVE_FIRST_ARM] + ['--analyse'],
        'run_order_note': 'the decisive check first: x0 at 0.001; if C0 fails there, STOP before any further arm',
        'solve_profile_per_arm': {
            'declared_base': '51 per cycle x (cap 2 + 1 initialisation) = 153 (W10.derive_solves_from_case_file)',
            'identity': 'observed == 153 + sum over network-failure events of [recovery_attempted] + '
                        '[tier2_attempted] (S44.event_level_solve_reconciliation); GUARD.verify(that) EXACTLY; one '
                        'arm per process so the process guard counts exactly one arm',
        },
        'formulas': FORMULAS,
        'criteria': CRITERIA,
        'calibration_zero_solve': {
            'note': 'committed-run IPOPT logs, production (gradient-based) scaling, first three solves per block. '
                    'x0 = s45_a0_c7 (spec 9d08ad2f), n7u = s47_a1a_baseline (the ageing baseline). DIFFERENT '
                    "configurations from this test's baseline arms: used to state predictions only",
            'runs': calibration,
        },
        'predictions_recorded_before_run': {
            'P1_decisive': 'C0 HOLDS on every pinned arm: the printed factor equals the pin exactly, no '
                           "gradient-based line, and 'No x/c/d scaling provided' on every network solve (no "
                           'scaling_factor suffix is declared on any production model)',
            'P2_baseline_factors': 'gradient-based effective factors at SRP1 as calibrated: TSO x0 0.002, TSO n7u '
                                   '0.001, every DSO 0.001 in both cells; DSOs carry c and d scaling, the TSO d scaling',
            'P3_baseline_disparity': 'baseline TSO median iterations n7u / x0 ~ 2 (calibration 54 / 24) -> C2 FAILS '
                                     'at baseline on TSO; DSO5 / DSO9 within the band; DSO7 inside it (calibration '
                                     '55 / 48 = 1.15)',
            'P4_best_value': 0.001,
            'P4_reason': '0.001 equals the gradient-based factor on 3 of the 4 families in both cells and on the n7u '
                         'TSO; it changes only the x0 TSO factor (0.002 -> 0.001) and removes constraint scaling',
            'P5_iterations_monotone_in_v': 'median iterations non-decreasing in v on every family (a larger factor '
                                           'is an effectively tighter dual-infeasibility tolerance)',
            'P6_failure_modes': [
                'loss of gradient-based constraint scaling (DSO c/d rows, TSO d rows) raises inertia '
                'regularisations and iterations -- the Addendum 45 degradation hazard; most likely on DSO7 / TSO '
                'in the n7u cell, which already carry 3.2x / 6.4x the regularised iterations of x0 at baseline '
                '(calibration DSO7 702 / 219, TSO 750 / 117)',
                'at 0.003 / 0.005: C3 iteration bands exceeded and acceptable-level terminations or max_iter hits '
                '(effectively 3-5x tighter unscaled dual tolerance)',
                'C2 may fail at EVERY value if the TSO disparity is structural (the unit adds complementarity pairs '
                'and regularised iterations) rather than a scaling artefact -- outcome NO_VALUE_MEETS_C2',
            ],
            'P7_outcome': 'RECOMMEND(0.001), probability ~0.5; NO_VALUE_MEETS_C2 ~0.3; FALLBACK_PURE_C ~0.15; '
                          'STOP ~0.05',
            'P8_C2_at_0p001_TSO_ratio': 'med_n7u / med_x0 on TSO at 0.001 in [1.0, 1.6]',
        },
        'not_permitted': ['no campaign harness, alpha row or 3x3', 'no commit to SRP1 case files or production '
                          'modules', 'no committed artifact modified or re-run onto; fresh names throughout'],
    }


def find_spec_v26():
    d = _abs(_P53)
    names = sorted(f for f in os.listdir(d) if f.startswith(SPEC_V26_PREFIX))
    if len(names) != 1:
        raise RuntimeError(f'expected exactly one frozen spec v26, found {names}')
    rel = os.path.join(_P53, names[0])
    sha = _sha(rel)
    if not names[0].endswith(f'{sha[:8]}.json'):
        raise RuntimeError(f'{rel} does not hash to its name ({sha})')
    with open(_abs(rel)) as handle:
        return rel, sha, json.load(handle)


def freeze_spec(started):
    failures = []
    got = _sha(SPEC_V25['path'])
    tracked, clean = _git_tracked_clean(SPEC_V25['path'])
    if got != SPEC_V25['sha256'] or not (tracked and clean):
        failures.append(f'predecessor v25 not as pinned: sha {got} tracked {tracked} clean {clean}')
    tracked, clean = _git_tracked_clean(os.path.basename(__file__))
    if not (tracked and clean):
        failures.append(f'harness must be committed and clean before the freeze (tracked {tracked} clean {clean})')
    existing = [f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V26_PREFIX)]
    if existing:
        failures.append(f'spec v26 already frozen (write-once): {existing}')
    for cell, meta in CELLS.items():
        with open(_abs(meta['committed_canonical']['spec'])) as handle:
            cands = json.load(handle)['candidates']
        ref = next((c['canonical'] for c in cands if c.get('label') == meta['committed_canonical']['label']), None)
        mine = H.canonical_candidate(meta['nodes'])
        if json.loads(json.dumps(mine)) != ref:
            failures.append(f'cell {cell}: canonical {mine} != committed {ref}')
    if failures:
        for f in failures:
            _log(f'[FREEZE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    calibration = calibrate()
    for cell, c in calibration.items():
        if c['anomalies']:
            _log(f'[FREEZE] calibration anomalies in {cell}: {c["anomalies"]}')
            raise SystemExit(1)
        for fam in FAMILIES:
            s = c['family_stats'][fam]
            _log(f"calibration {cell} {fam}: eff {s['objective_scaling_factor_distinct']} med {s['iterations_median']} "
                 f"p90 {s['iterations_p90']} reg {s['iterations_with_inertia_regularisation_total']} "
                 f"res {s['restoration_entries_total']} c {s['c_scaling_distinct']} d {s['d_scaling_distinct']}")
    content = spec_v26_content(calibration)
    content['frozen_utc'] = _utc()
    content['git_head_at_freeze'] = _git(['rev-parse', 'HEAD'])
    text = json.dumps(content, indent=1, default=_json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V26_PREFIX}{sha[:8]}.json')
    with open(_abs(rel), 'x') as handle:
        handle.write(text)
    os.makedirs(_abs(LAUNCH_LOGS_REL), exist_ok=True)
    check_rel, check_sha, _ = find_spec_v26()
    guard_failures = GUARD.verify(0)
    _log(f'frozen spec v26: {rel} sha256={sha} (predecessor v25 {SPEC_V25["sha256"]}); re-read {check_sha == sha}; '
         f'guard {dict(GUARD.counts)} verify(0)={guard_failures}; wall {time.time() - started:.1f}s')
    return 0 if not guard_failures and check_sha == sha else 1


# ======================================================================================================================
#  one arm
# ======================================================================================================================
def _options_snapshot(opts):
    return {k: v for k, v in (opts or {}).items()}


def _expected_options(case_options, retry=None, tier2=False):
    exp = {'max_iter': 500, 'fixed_variable_treatment': NET.IPOPT_FIXED_VARIABLE_TREATMENT}
    exp.update(case_options or {})
    if retry is not None:
        exp.update(retry)
    if tier2:
        exp['mu_strategy'] = 'adaptive'
    return {k: v for k, v in exp.items() if k in PROTECTED_OPTION_KEYS}


def _values_equal(expected, logged):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(logged) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(logged)


def run_one_arm(arm, started):
    if arm not in ARMS:
        raise SystemExit(f'undeclared arm {arm}; declared: {sorted(ARMS)}')
    cell, value = ARMS[arm]['cell'], ARMS[arm]['value']
    spec_rel, spec_sha, spec = find_spec_v26()
    out_root = _abs(os.path.join(OUT_REL, 'arms', arm))
    arm_dir = os.path.join(out_root, 'arm')
    _log(f'{STAGE}')
    _log(f'arm {arm}: cell {cell}, value {value}; spec {spec_rel} ({spec_sha}); HEAD {_git(["rev-parse", "HEAD"])}')
    os.environ.update(H.THREAD_CAP_ENV)

    # ---- preconditions (W10's + this task's) ----
    W10.AA_REFERENCE = dict(W32.BASELINE_REFERENCE)                                   # substitution 1
    W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + EXTRA_FORBIDDEN  # 2
    failures, instance = W10.check_preconditions(out_root)
    status = _git(['status', '--porcelain', '--'] + list(EXTRA_CLEAN_FILES) + [spec_rel])
    if status.strip():
        failures.append(f'files not clean in git:\n{status}')
    if spec['arms'].get(arm) is None or spec['arms'][arm]['value'] != value or spec['arms'][arm]['cell'] != cell:
        failures.append(f'arm {arm} not declared in the frozen spec as ({cell}, {value})')
    if spec['harness']['sha256_at_freeze'] != _sha(__file__):
        failures.append('harness differs from the one the spec was frozen with')
    nodes = CELLS[cell]['nodes']
    canonical = H.canonical_candidate(nodes)
    key = H.candidate_key(canonical)
    if key != spec['instance']['cells'][cell]['candidate_key']:
        failures.append(f'candidate key {key} != spec')
    ids = H.eval_ids(f'{W10.CAMPAIGN_ID}_{arm}', key)
    if ids != spec['arms'][arm]['eval_ids']:
        failures.append(f'eval ids {ids} != spec {spec["arms"][arm]["eval_ids"]}')
    import p56a_oracle as O
    for which in ('run', 'precheck'):
        if os.path.exists(os.path.join(O.WORK_DIR, ids[which])):
            failures.append(f'eval dir exists (collision): {os.path.join(O.WORK_DIR, ids[which])}')
    declaration, recert_spec = W32._load_recert_declaration()
    ess_pin = declaration['ess_params_file']
    if _sha(ess_pin['path']) != ess_pin['sha256']:
        failures.append('ESS parameters file sha256 != recert pin')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        return 1

    # ---- rule eleven: capture paths for every quantity the spec requires ----
    checklist = W10.capture_path_checklist()
    for net in NETWORK_FAMILY:
        pf = os.path.join(REPO, 'data', 'SRP1', net, f'{net}_params.json')
        with open(pf) as handle:
            solver = json.load(handle)['solver']
        opts = solver.get('options') or {}
        checklist[f'{net}_file_print_level_6'] = opts.get('file_print_level') == 6
        checklist[f'{net}_output_file_set'] = bool(opts.get('output_file'))
        checklist[f'{net}_no_scaling_key_in_case_file'] = not any(k in opts for k in SCALING_KEYS)
        checklist[f'{net}_no_scaling_key_in_recovery_options'] = not any(
            k in (solver.get('recovery_options') or {}) for k in SCALING_KEYS)
    src_create = inspect.getsource(NET._create_smopf_solver)
    checklist['network_options_merged_before_overrides'] = _ordered(
        src_create, 'options.update(solver_params.options)', 'options.update(option_overrides)',
        'solver.options[key] = value')
    src_run = inspect.getsource(NET._run_smopf)
    checklist['tier2_overrides_only_mu_strategy'] = "tier2_options['mu_strategy'] = 'adaptive'" in src_run
    checklist['w10_run_arm_resolves_hook_factory_via_H'] = 'H._config_hook_factory(' in inspect.getsource(W10.run_arm)
    checklist['w10_run_arm_reads_module_C_STAR'] = 'H.canonical_candidate(C_STAR)' in inspect.getsource(W10.run_arm)
    checklist['event_level_reconciliation_callable'] = callable(S44.event_level_solve_reconciliation)
    checklist['network_failure_events_carry_attempt_flags'] = (
        "'recovery_attempted'" in inspect.getsource(G._new_network_event)
        and "'tier2_attempted'" in inspect.getsource(G._new_network_event))
    missing = sorted(k for k, v in checklist.items() if not v)
    if missing:
        _log(f'[RULE ELEVEN] capture paths missing: {missing}')
        return 1
    declared = W10.derive_solves_from_case_file()
    base = declared['declared_base_solves_per_arm']
    _log(f"preconditions passed; checklist {len(checklist)} items all true; DECLARED BEFORE THE RUN: "
         f"{declared['solves_per_cycle']} solves/cycle, base {base}; gate on observed == base + every attempted "
         f'retry and GUARD.verify(that) exactly')

    # ---- the run ----
    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')
    pin_record = {}
    orig_factory = H._config_hook_factory
    orig_c_star = W10.C_STAR

    def factory_with_declaration_and_pin(spec_like, holder, overrides=None, **kwargs):   # substitution 4
        spec2 = copy.deepcopy(spec_like)
        spec2['configuration'].update(copy.deepcopy(declaration))
        inner = orig_factory(spec2, holder, overrides=overrides, **kwargs)

        def hook(planning, sed, candidate, report):
            inner(planning=planning, sed=sed, candidate=candidate, report=report)
            holders = [('TSO', planning.transmission_network)] + [
                (f'DSO{n}', dn) for n, dn in sorted(planning.distribution_networks.items())]
            per = {}
            ids_seen = set()
            for tag, h in holders:
                sp = h.params.solver_params
                if not isinstance(sp.options, dict):
                    raise RuntimeError(f'{tag}: solver_params.options is not a dict')
                if id(sp.options) in ids_seen:
                    raise RuntimeError(f'{tag}: options dict shared with another holder')
                ids_seen.add(id(sp.options))
                before = _options_snapshot(sp.options)
                if any(k in before for k in SCALING_KEYS):
                    raise RuntimeError(f'{tag}: a scaling key is already configured: {before}')
                if value is not None:
                    sp.options['nlp_scaling_method'] = 'user-scaling'
                    sp.options['obj_scaling_factor'] = value
                after = _options_snapshot(sp.options)
                expected_after = dict(before)
                if value is not None:
                    expected_after.update({'nlp_scaling_method': 'user-scaling', 'obj_scaling_factor': value})
                logs_dirs = sorted({net.logs_dir for per_y in h.network.values() for net in per_y.values()})
                per[tag] = {'network_name': h.name, 'options_before': before, 'options_after': after,
                            'readback_equals_expected': after == expected_after,
                            'recovery_options': _options_snapshot(sp.recovery_options),
                            'recovery_enabled': getattr(sp, 'recovery_enabled', None),
                            'recovery_tier2_enabled': getattr(sp, 'recovery_tier2_enabled', None),
                            'logs_dirs': logs_dirs}
                if after != expected_after:
                    raise RuntimeError(f'{tag}: pin read-back mismatch {after} != {expected_after}')
            esso_opts = _options_snapshot(sed.params.solver_params.options)
            pin_record.update({'value': value, 'per_network': per, 'esso_options_before_run': esso_opts,
                               'esso_options_object': sed.params.solver_params.options,
                               'applied_after_committed_hook': True})
            report.setdefault('rule_eleven_checklist', {})['w78_scaling_pin'] = {
                'value': value, 'readback_all': all(p['readback_equals_expected'] for p in per.values())}
        return hook

    H._config_hook_factory = factory_with_declaration_and_pin
    W10.C_STAR = dict(nodes)                                                             # substitution 3
    counters = W10.CloneCaptureCounters()
    expected_counts = {}
    try:
        with counters.installed():
            report, summary = W10.run_arm(arm, 'on', arm_dir, counters, expected_counts)
            counters.phase = 'post'
    finally:
        H._config_hook_factory = orig_factory
        W10.C_STAR = orig_c_star
    esso_after = _options_snapshot(pin_record.pop('esso_options_object', None))
    pin_record['esso_options_after_run'] = esso_after
    pin_record['esso_options_unchanged'] = esso_after == pin_record.get('esso_options_before_run')
    _log(f"arm ran: cycles {summary['cycles_run']} gross {summary['gross_operational_cost']} solves "
         f"{(summary['solve_profile'] or {}).get('observed')} failures {summary['network_failures_summary']}")

    # ---- solve reconciliation, guard exact ----
    observed = ((summary['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
    event_level = S44.event_level_solve_reconciliation(report, base)
    reconciled = event_level['expected']
    guard_failures = GUARD.verify(reconciled) if reconciled is not None else ['event-level reconciliation unsupported']
    esso_solves = declared['esso_solves_per_cycle'] * (W10.CAP + 1)
    network_solves_guard = (GUARD.counts['permitted_solve'] - esso_solves) if reconciled is not None else None

    # ---- zero-solve log parse ----
    with open(W10.CASE_JSON) as handle:
        case = json.load(handle)
    years, days = sorted(case['Years']), sorted(case['Days'])
    logs_dir = os.path.join(O.WORK_DIR, ids['run'], 'logs')
    all_dirs = {d for p in pin_record['per_network'].values() for d in p['logs_dirs']}
    records, log_files, anomalies = parse_run_logs(logs_dir, years, days)
    if all_dirs != {logs_dir}:
        anomalies.append(f'network logs_dirs {sorted(all_dirs)} != {logs_dir}')
    net_to_tag = {p['network_name']: tag for tag, p in pin_record['per_network'].items()}

    # C0 (pinned) / baseline record, and C0b
    c0_fail, c0b_fail = [], []
    for r in records:
        opts = r['options']
        where = f"{r['log']}#{r['segment_index']}"
        if value is not None:
            nsm, osf = opts.get('nlp_scaling_method'), opts.get('obj_scaling_factor')
            if not (nsm and nsm['value'] == 'user-scaling' and nsm['times_used'] >= 1):
                c0_fail.append(f'{where}: nlp_scaling_method {nsm}')
            if not (osf and _values_equal(value, osf['value']) and osf['times_used'] >= 1):
                c0_fail.append(f'{where}: obj_scaling_factor {osf}')
            eff = r['objective_scaling_factor_lines']
            if not (len(eff) == 1 and float(eff[0]) == value and eff[0] == format(value, 'g')):
                c0_fail.append(f'{where}: objective scaling factor lines {eff}')
            if r['gradient_based_objective_scaling_lines']:
                c0_fail.append(f"{where}: gradient-based line present {r['gradient_based_objective_scaling_lines']}")
        else:
            if any(k in opts for k in SCALING_KEYS):
                c0_fail.append(f'{where}: baseline arm carries a scaling option')
            eff, grad = r['objective_scaling_factor_lines'], r['gradient_based_objective_scaling_lines']
            if not (len(eff) == 1 and len(grad) == 1 and float(eff[0]) == float(grad[0])):
                c0_fail.append(f'{where}: baseline effective {eff} != gradient-based {grad}')
        pn = pin_record['per_network'][net_to_tag[r['network']]]
        retry = None
        if r['attempt'] != 'primary':
            retry = {k: v for k, v in (pn['recovery_options'] or {}).items() if k != 'hessian_approximation'}
        exp = _expected_options(pn['options_before'], retry=retry, tier2=r['attempt'] == 'recovery_tier2')
        logged = {k: v['value'] for k, v in opts.items() if k in PROTECTED_OPTION_KEYS}
        if set(exp) != set(logged) or not all(_values_equal(exp[k], logged[k]) for k in exp):
            c0b_fail.append(f'{where}: protected options {logged} != expected {exp}')
    n_primary_logs = sum(1 for f in log_files if not f.endswith('_recovery.log') and not f.endswith('_tier2.log'))
    count_ok = (network_solves_guard is not None and len(records) == network_solves_guard
                and n_primary_logs == 48 and not anomalies)
    c0 = {'applies': value is not None, 'n_network_segments_parsed': len(records),
          'n_network_solves_in_guard': network_solves_guard, 'segment_count_equals_guard': count_ok,
          'n_primary_logs': n_primary_logs, 'anomalies': anomalies,
          'n_failures': len(c0_fail), 'failures_first': c0_fail[:50],
          'holds': count_ok and not c0_fail,
          'effective_factor_distinct': sorted({x for r in records for x in r['objective_scaling_factor_lines']}),
          'gradient_line_distinct': sorted({x for r in records for x in r['gradient_based_objective_scaling_lines']}),
          'x_c_d_scaling_distinct': sorted({f"{r['x_scaling']} | {r['c_scaling']} | {r['d_scaling']}"
                                            for r in records})}
    c0b = {'n_failures': len(c0b_fail), 'failures_first': c0b_fail[:50], 'holds': not c0b_fail}
    stats = family_stats(records)
    retries = [r for r in records if r['attempt'] != 'primary']
    classes = (summary['network_failures_summary'] or {}).get('classes') or {}
    run_ok = (summary['cycles_run'] == W10.CAP and not guard_failures
              and GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0
              and pin_record['esso_options_unchanged'])

    result = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
        'git_head_at_run': _git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': _sha(__file__),
        'frozen_spec': {'path': spec_rel, 'sha256': spec_sha},
        'arm': arm, 'cell': cell, 'value': value,
        'instance': {'label': CELLS[cell]['label'], 'canonical': canonical, 'candidate_key': key,
                     'eval_ids': ids, 'case_file_sha256': _sha(W10.CASE_JSON)},
        'objective_convention': 'gross_operational_cost is the settlement-excluded gross cost after 2 cycles '
                                '(informational; the pin test compares solver behaviour, not costs)',
        'declared_substitutions': {
            '1_reference': W32.BASELINE_REFERENCE, '2_extra_forbidden': list(EXTRA_FORBIDDEN),
            '3_candidate': {'W10.C_STAR': {str(k): list(v) for k, v in nodes.items()}},
            '4_hook': {'recert_spec': recert_spec, 'declaration_injected': declaration,
                       'pin': 'after the committed hook; baseline arms set nothing'}},
        'capture_path_checklist_asserted_before_run': checklist,
        'preconditions_instance': instance,
        'pin_record': pin_record,
        'arm_summary': summary,
        'solve_profile': {'declared_before_the_run': declared, 'base': base,
                          'event_level_reconciliation': event_level, 'observed_in_arm': observed,
                          'guard_counts': dict(GUARD.counts), 'guard_verify_exactly': reconciled,
                          'guard_verify_failures': guard_failures, 'esso_solves': esso_solves,
                          'network_solves': network_solves_guard,
                          'retries_by_class': classes},
        'C0_decisive': c0, 'C0b_tolerances_unchanged': c0b,
        'family_stats': stats,
        'retry_segments': [{k: r[k] for k in ('network', 'year', 'day', 'attempt', 'iterations', 'exit')}
                           for r in retries],
        'per_solve_records': [{k: v for k, v in r.items() if k != 'options'} | {
            'options_logged': {k: v['value'] for k, v in r['options'].items() if k != 'output_file'}}
            for r in records],
        'ipopt_logs_sha256': log_files,
        'run_ok': run_ok,
        'wall_clock_s': time.time() - started,
    }
    path = os.path.join(out_root, 'arm_result.json')
    _write_once(path, result)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            manifest[os.path.relpath(os.path.join(root, fname), REPO)] = _sha(os.path.join(root, fname))
    manifest.update(log_files)
    for fname in sorted(os.listdir(logs_dir)):
        rel = os.path.relpath(os.path.join(logs_dir, fname), REPO)
        manifest.setdefault(rel, _sha(rel))
    _write_once(os.path.join(out_root, 'manifest_sha256.json'), manifest)

    _log(f"C0 ({'pinned' if value is not None else 'baseline record'}): holds {c0['holds']}; segments "
         f"{c0['n_network_segments_parsed']} vs guard network solves {c0['n_network_solves_in_guard']}; effective "
         f"factors {c0['effective_factor_distinct']}; gradient lines {c0['gradient_line_distinct']}; x|c|d "
         f"{c0['x_c_d_scaling_distinct']}; failures {c0['n_failures']} {c0['failures_first'][:3]}")
    _log(f"C0b tolerances unchanged: {c0b['holds']} ({c0b['n_failures']} failures) {c0b['failures_first'][:2]}")
    for fam in FAMILIES:
        s = stats[fam]
        _log(f"  {fam}: med {s['iterations_median']} p90 {s['iterations_p90']} max {s['iterations_max']} reg "
             f"{s['iterations_with_inertia_regularisation_total']} res {s['restoration_entries_total']} exits "
             f"{s['exits']} mu_med {s['final_mu_median']} kkt_max {s['final_kkt_scaled_max']}")
    _log(f"guard {dict(GUARD.counts)} verify({reconciled}) = {guard_failures}; ESSO options unchanged "
         f"{pin_record['esso_options_unchanged']}; wrote {os.path.relpath(path, REPO)}; wall "
         f'{time.time() - started:.0f} s')
    if value is not None and not c0['holds']:
        _log('STOP: C0 FAILS -- the effective objective scaling factor is not the pinned value on every network solve')
        return 2
    return 0 if (run_ok and c0['holds'] and c0b['holds']) else 1


def _ordered(text, *needles):
    positions = [text.find(n) for n in needles]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


# ======================================================================================================================
#  analysis (zero-solve)
# ======================================================================================================================
def _within(a, b, abs_floor, ratio_hi):
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_floor:
        return True
    return b > 0 and (1.0 / ratio_hi) <= a / b <= ratio_hi


def analyse(started):
    spec_rel, spec_sha, spec = find_spec_v26()
    results = {}
    for name in ARMS:
        path = _abs(os.path.join(OUT_REL, 'arms', name, 'arm_result.json'))
        if not os.path.isfile(path):
            _log(f'[ANALYSE] missing arm result {name}')
            return 1
        with open(path) as handle:
            results[name] = json.load(handle)
        results[name]['_sha256'] = _sha(path)
        if results[name]['frozen_spec']['sha256'] != spec_sha:
            _log(f'[ANALYSE] {name} ran under a different spec')
            return 1
    arm_of = {(a['cell'], a['value']): n for n, a in ARMS.items()}
    fs = {n: r['family_stats'] for n, r in results.items()}

    def arm_conv(n):
        r = results[n]
        s = r['arm_summary']
        cl = r['solve_profile']['retries_by_class']
        n_maxiter_all = sum(1 for x in r['per_solve_records'] if x['exit'] == 'Maximum Number of Iterations Exceeded.')
        primary_nonacc = sum(fs[n][f]['n_non_acceptable'] for f in FAMILIES)
        return {'cycles_run': s['cycles_run'], 'local_solve_failures': s['local_solve_failures'],
                'unrecovered': cl.get('unrecovered', 0), 'indeterminate': cl.get('indeterminate', 0),
                'esso_recovery_events': (s['network_failures_summary'] or {}).get('n_esso_recovery_events'),
                'max_iter_hits_any_attempt': n_maxiter_all, 'primary_non_acceptable': primary_nonacc,
                'guard_verify_failures': r['solve_profile']['guard_verify_failures'],
                'n_retry_segments': len(r['retry_segments'])}

    conv = {n: arm_conv(n) for n in results}
    per_value = {}
    for v in PIN_VALUES:
        cells = {c: arm_of[(c, v)] for c in CELLS}
        base = {c: arm_of[(c, None)] for c in CELLS}
        c0 = {c: results[cells[c]]['C0_decisive']['holds'] for c in CELLS}
        c0b = {c: results[cells[c]]['C0b_tolerances_unchanged']['holds'] for c in CELLS}
        c1_items = {}
        for c in CELLS:
            a, b = conv[cells[c]], conv[base[c]]
            c1_items[c] = {
                'two_cycles_no_local_failures': a['cycles_run'] == 2 and a['local_solve_failures'] == 0,
                'no_unrecovered_indeterminate_esso_events': (a['unrecovered'] == 0 and a['indeterminate'] == 0
                                                             and not a['esso_recovery_events']),
                'no_max_iter_hits': a['max_iter_hits_any_attempt'] == 0,
                'primary_non_acceptable_le_baseline': a['primary_non_acceptable'] <= b['primary_non_acceptable'],
                'guard_exact': a['guard_verify_failures'] == [],
            }
        c1 = all(all(x.values()) for x in c1_items.values())
        c2_items = {}
        for f in FAMILIES:
            m0, m1 = fs[cells['x0']][f]['iterations_median'], fs[cells['n7u']][f]['iterations_median']
            p0, p1 = fs[cells['x0']][f]['iterations_p90'], fs[cells['n7u']][f]['iterations_p90']
            c2_items[f] = {'med_x0': m0, 'med_n7u': m1, 'p90_x0': p0, 'p90_n7u': p1,
                           'median_ok': _within(m1, m0, 3, 1.25), 'p90_ok': _within(p1, p0, 4, 1.35)}
        c2 = all(x['median_ok'] and x['p90_ok'] for x in c2_items.values())
        c3_items = {}
        for c in CELLS:
            for f in FAMILIES:
                a, b = fs[cells[c]][f], fs[base[c]][f]
                bmax_med = max(fs[base['x0']][f]['iterations_median'], fs[base['n7u']][f]['iterations_median'])
                bmax_p90 = max(fs[base['x0']][f]['iterations_p90'], fs[base['n7u']][f]['iterations_p90'])
                breg = b['iterations_with_inertia_regularisation_total']
                bacc = b['n_acceptable_level']
                c3_items[f'{c}|{f}'] = {
                    'restoration_entries': [a['restoration_entries_total'], b['restoration_entries_total']],
                    'restoration_ok': a['restoration_entries_total'] <= b['restoration_entries_total'],
                    'inertia_reg': [a['iterations_with_inertia_regularisation_total'], breg],
                    'inertia_reg_ok': a['iterations_with_inertia_regularisation_total'] <= max(1.25 * breg, breg + 5),
                    'acceptable_level': [a['n_acceptable_level'], bacc],
                    'acceptable_level_ok': a['n_acceptable_level'] <= bacc + max(2, 0.25 * bacc),
                    'median': [a['iterations_median'], bmax_med],
                    'median_ok': a['iterations_median'] <= 1.25 * bmax_med,
                    'p90': [a['iterations_p90'], bmax_p90],
                    'p90_ok': a['iterations_p90'] <= 1.35 * bmax_p90,
                }
        c3 = all(all(v2 for k2, v2 in x.items() if k2.endswith('_ok')) for x in c3_items.values())
        d = max(abs(math.log(c2_items[f]['med_n7u'] / c2_items[f]['med_x0'])) for f in FAMILIES
                if c2_items[f]['med_x0'] and c2_items[f]['med_n7u'])
        total_iters = sum(fs[cells[c]][f]['iterations_total'] for c in CELLS for f in FAMILIES)
        per_value[format(v, 'g')] = {'arms': cells, 'C0': c0, 'C0b': c0b, 'C1_items': c1_items, 'C1': c1,
                                     'C2_items': c2_items, 'C2': c2, 'C3_items': c3_items, 'C3': c3,
                                     'D': d, 'total_network_iterations_both_cells': total_iters,
                                     'eligible': all(c0.values()) and all(c0b.values()) and c1 and c2 and c3}
    baseline_c2 = {}
    for f in FAMILIES:
        m0, m1 = fs[arm_of[('x0', None)]][f]['iterations_median'], fs[arm_of[('n7u', None)]][f]['iterations_median']
        p0, p1 = fs[arm_of[('x0', None)]][f]['iterations_p90'], fs[arm_of[('n7u', None)]][f]['iterations_p90']
        baseline_c2[f] = {'med_x0': m0, 'med_n7u': m1, 'p90_x0': p0, 'p90_n7u': p1,
                          'median_ok': _within(m1, m0, 3, 1.25), 'p90_ok': _within(p1, p0, 4, 1.35)}
    if not all(all(pv['C0'].values()) for pv in per_value.values()):
        outcome, chosen = 'STOP', None
    else:
        eligible = [(pv['D'], pv['total_network_iterations_both_cells'], float(k), k)
                    for k, pv in per_value.items() if pv['eligible']]
        if eligible:
            best_d = min(e[0] for e in eligible)
            tied = sorted((e for e in eligible if e[0] - best_d <= 0.05), key=lambda e: (e[1], e[2]))
            outcome, chosen = 'RECOMMEND', tied[0][3]
        elif not any(pv['C1'] and pv['C3'] and all(pv['C0b'].values()) for pv in per_value.values()):
            outcome, chosen = 'FALLBACK_PURE_C', None
        else:
            outcome, chosen = 'NO_VALUE_MEETS_C2', None
    arms_table = {n: {'cell': ARMS[n]['cell'], 'value': ARMS[n]['value'],
                      'C0_holds': results[n]['C0_decisive']['holds'],
                      'C0b_holds': results[n]['C0b_tolerances_unchanged']['holds'],
                      'effective_factor_distinct': results[n]['C0_decisive']['effective_factor_distinct'],
                      'x_c_d_scaling_distinct': results[n]['C0_decisive']['x_c_d_scaling_distinct'],
                      'segments_vs_guard': [results[n]['C0_decisive']['n_network_segments_parsed'],
                                            results[n]['C0_decisive']['n_network_solves_in_guard']],
                      'solves_declared_expected': results[n]['solve_profile']['guard_verify_exactly'],
                      'solves_observed': results[n]['solve_profile']['guard_counts']['permitted_solve'],
                      'guard_verify_failures': results[n]['solve_profile']['guard_verify_failures'],
                      'convergence': conv[n],
                      'gross_operational_cost_after_2_cycles': results[n]['arm_summary']['gross_operational_cost'],
                      'family_stats': fs[n], 'arm_result_sha256': results[n]['_sha256']}
                  for n in ARMS}
    payload = {'schema': SCHEMA + '_analysis', 'stage': STAGE, 'timestamp_utc': _utc(),
               'git_head_at_run': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(__file__),
               'frozen_spec': {'path': spec_rel, 'sha256': spec_sha}, 'criteria': CRITERIA,
               'arms': arms_table, 'baseline_C2_reference': baseline_c2, 'per_value': per_value,
               'outcome': outcome, 'recommended_obj_scaling_factor': chosen,
               'predictions': spec['predictions_recorded_before_run']}
    out = _abs(os.path.join(OUT_REL, 'analysis'))
    os.makedirs(out, exist_ok=False)
    _write_once(os.path.join(out, 'analysis.json'), payload)
    manifest = {os.path.relpath(os.path.join(out, 'analysis.json'), REPO): _sha(os.path.join(out, 'analysis.json'))}
    _write_once(os.path.join(out, 'manifest_sha256.json'), manifest)
    guard_failures = GUARD.verify(0)
    _log(f'OUTCOME {outcome}; recommended {chosen}')
    for k, pv in per_value.items():
        _log(f"v={k}: C0 {pv['C0']} C0b {pv['C0b']} C1 {pv['C1']} C2 {pv['C2']} C3 {pv['C3']} D {pv['D']:.3f} "
             f"iters {pv['total_network_iterations_both_cells']} eligible {pv['eligible']}")
    _log(f'guard {dict(GUARD.counts)} verify(0) = {guard_failures}; wall {time.time() - started:.1f}s')
    return 0 if not guard_failures else 1


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-spec', action='store_true')
    group.add_argument('--arm')
    group.add_argument('--analyse', action='store_true')
    args = parser.parse_args()
    started = time.time()
    if args.freeze_spec:
        return freeze_spec(started)
    if args.analyse:
        return analyse(started)
    return run_one_arm(args.arm, started)


if __name__ == '__main__':
    sys.exit(main())
