"""
P5.15 Addendum 30 (task W21, item 1(b)) -- ZERO-SOLVE analysis of the x = 0 digest difference
found by the case-file gate `p515_s47_case_file_baseline_check.py`.

The gate's literal criterion "x = 0 oracle-path model digests identical before/after (every
TSO/DSO/ESSO block)" FAILED: the 48 network blocks are identical, the 3 ESSO blocks are not.
This script reads ONLY the two committed gate records (before / after, pinned by sha256 on the
command line) -- no model is built, nothing is solved -- and establishes, item by item, WHAT
differs in the x = 0 ESSO blocks and whether any of it can reach Q(0):
  X1  the differing components of every x = 0 ESSO block are exactly
      {Constraint energy_storage_capacity_degradation, Expression salvage_value,
       Expression salvage_credit};
  X2  every differing degradation row is DEACTIVATED (active False) in both phases (the cohorts
      are inactive at x = 0: `_configure_esso_cohort_state`);
  X3  the ESSO objective is identical and is `feasibility_penalty` alone (the salvage Expression
      is not part of the ESSO local objective; it enters only the reported recourse);
  X4  the variables of the salvage Expression (es_e_rated_per_unit[., T], es_e_available_per_unit
      [., T]) are pinned to 0 at x = 0 by ACTIVE rows in both phases: rated_e_capacity_unit
      (lower = upper = 0 on es_e_rated_per_unit) and available_e_capacity_unit
      (es_e_available - es_e_rated * SoH == 0), so the salvage is 0 at every feasible point;
  X5  the NL file Pyomo writes for each x = 0 ESSO block (the problem IPOPT receives) is
      byte-identical before / after;
  X6  every other component of the x = 0 ESSO blocks and every network block is identical.

Output (write-once): data/SRP1/Results/P515S47/case_file_baseline/x0_diff_analysis/
    x0_diff_analysis.json, manifest_sha256.json
Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_case_file_x0_diff_analysis.py \\
        --before-sha256 <sha> --after-sha256 <sha> \\
        > data/SRP1/Results/P515S47/case_file_x0_diff_analysis_launch.log 2>&1
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W21 x0 diff analysis (zero solves, no model)').install()

ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'case_file_baseline')
OUT_REL = os.path.join(ROOT, 'x0_diff_analysis')
EXPECTED_DIFFERING = ['Constraint:energy_storage_capacity_degradation', 'Expression:salvage_credit',
                      'Expression:salvage_value']


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W21-x0diff] {msg}', flush=True)


def _t(text):
    return ast.literal_eval(text)


def analyse_block(key, a_items, b_items, a_block, b_block, nl):
    ca, cb = a_block['components'], b_block['components']
    differing = sorted(n for n in set(ca) | set(cb) if ca.get(n) != cb.get(n))
    out = {'components_differing': differing}
    deg_a, deg_b = a_items['Constraint:energy_storage_capacity_degradation'], \
        b_items['Constraint:energy_storage_capacity_degradation']
    diff_rows = sorted(i for i in set(deg_a) | set(deg_b) if deg_a.get(i) != deg_b.get(i))
    out['degradation_rows_differing'] = len(diff_rows)
    out['degradation_rows_total'] = len(set(deg_a) | set(deg_b))
    out['degradation_rows_differing_all_inactive_both_phases'] = all(
        _t(deg_a[i])[0] is False and _t(deg_b[i])[0] is False for i in diff_rows)
    out['objective'] = {'before': a_items['Objective:objective'], 'after': b_items['Objective:objective']}
    out['objective_identical_and_feasibility_penalty_only'] = (
        a_items['Objective:objective'] == b_items['Objective:objective']
        and _t(b_items['Objective:objective']['None'])[1] == 'feasibility_penalty')
    salvage_vars = sorted(set(re.findall(r'es_e_(?:rated|available)_per_unit\[\d+,\d+\]',
                                         b_items['Expression:salvage_value']['None'])))
    pins = {}
    for phase, items in (('before', a_items), ('after', b_items)):
        rated_rows = {_t(v)[3]: _t(v) for v in items['Constraint:rated_e_capacity_unit'].values()}
        avail_rows = [_t(v) for v in items['Constraint:available_e_capacity_unit'].values()]
        per_var = {}
        for var in salvage_vars:
            idx = var[var.index('['):]
            if var.startswith('es_e_rated'):
                row = rated_rows.get(var)
                per_var[var] = {'pinned_by': 'rated_e_capacity_unit', 'row': row,
                                'pinned_to_zero': bool(row) and row[0] is True and row[1] == 0.0 and row[2] == 0.0}
            else:
                body = f'es_e_available_per_unit{idx} - es_e_rated_per_unit{idx}*es_soh_per_unit_cumul{idx}'
                rows = [r for r in avail_rows if r[3] == body]
                rated = rated_rows.get(f'es_e_rated_per_unit{idx}')
                per_var[var] = {'pinned_by': 'available_e_capacity_unit (E_av == E_rated * SoH) with E_rated pinned',
                                'row': rows[0] if rows else None,
                                'pinned_to_zero': (len(rows) == 1 and rows[0][0] is True and rows[0][1] == 0.0
                                                   and rows[0][2] == 0.0 and bool(rated) and rated[0] is True
                                                   and rated[1] == 0.0 and rated[2] == 0.0)}
        pins[phase] = per_var
    out['salvage_variables'] = salvage_vars
    out['salvage_variable_pins'] = pins
    out['salvage_variables_all_pinned_to_zero_both_phases'] = bool(salvage_vars) and all(
        v['pinned_to_zero'] for p in pins.values() for v in p.values())
    out['salvage_expression'] = {'before': a_items['Expression:salvage_value']['None'],
                                 'after': b_items['Expression:salvage_value']['None']}
    out['nl'] = nl
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-sha256', required=True)
    parser.add_argument('--after-sha256', required=True)
    args = parser.parse_args()
    started = time.time()
    out_root = os.path.join(REPO, OUT_REL)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    paths = {'before': os.path.join(REPO, ROOT, 'before', 'case_file_baseline_before.json'),
             'after': os.path.join(REPO, ROOT, 'after', 'case_file_baseline_after.json')}
    got = {k: _sha(p) for k, p in paths.items()}
    if got != {'before': args.before_sha256, 'after': args.after_sha256}:
        raise SystemExit(f'gate records do not hash to the given sha256: {got}')
    with open(paths['before']) as handle:
        before = json.load(handle)
    with open(paths['after']) as handle:
        after = json.load(handle)
    a, b = before['builds']['x0'], after['builds']['x0']
    network_differing = sorted(k for k in b['blocks'] if not k.startswith('ESSO')
                               and a['blocks'][k]['sha256'] != b['blocks'][k]['sha256'])
    per_block = {}
    for key in sorted(k for k in b['blocks'] if k.startswith('ESSO')):
        nl = {'before': a['esso_nl'][key], 'after': b['esso_nl'][key],
              'identical': a['esso_nl'][key]['sha256'] == b['esso_nl'][key]['sha256']
              and b['esso_nl'][key]['sha256'] is not None}
        per_block[key] = analyse_block(key, a['esso_items'][key], b['esso_items'][key], a['blocks'][key],
                                       b['blocks'][key], nl)
    checks = {
        'X1_differing_components_are_degradation_rows_and_salvage_expressions_only': all(
            v['components_differing'] == EXPECTED_DIFFERING for v in per_block.values()),
        'X2_every_differing_degradation_row_inactive_both_phases': all(
            v['degradation_rows_differing_all_inactive_both_phases'] for v in per_block.values()),
        'X3_esso_objective_identical_feasibility_penalty_only': all(
            v['objective_identical_and_feasibility_penalty_only'] for v in per_block.values()),
        'X4_salvage_variables_pinned_to_zero_by_active_rows_both_phases': all(
            v['salvage_variables_all_pinned_to_zero_both_phases'] for v in per_block.values()),
        'X5_esso_nl_identical_before_after': all(v['nl']['identical'] for v in per_block.values()),
        'X6_network_blocks_identical': not network_differing and len(b['blocks']) == 51,
    }
    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    payload = {'stage': 'P5.15 Addendum 30 W21 item 1(b): x = 0 ESSO digest difference analysis (zero solves, no model)',
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'script': os.path.basename(__file__), 'script_sha256': _sha(os.path.abspath(__file__)),
               'inputs': {k: {'path': os.path.relpath(p, REPO), 'sha256': got[k]} for k, p in paths.items()},
               'gate_literal_criterion': ('x = 0 oracle-path model digests identical before/after (every TSO/DSO/ESSO '
                                          'block): FAILED in the gate record (3 ESSO blocks differ)'),
               'network_blocks_differing': network_differing, 'per_esso_block': per_block,
               'checks': checks, 'all_ok': all_ok,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
               'wall_clock_s': time.time() - started}
    os.makedirs(out_root)
    path = os.path.join(out_root, 'x0_diff_analysis.json')
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    with open(os.path.join(out_root, 'manifest_sha256.json'), 'w') as handle:
        json.dump({os.path.relpath(path, REPO): _sha(path)}, handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    for key, v in per_block.items():
        _log(f"{key}: differing {v['components_differing']}; degradation rows differing "
             f"{v['degradation_rows_differing']}/{v['degradation_rows_total']} all inactive "
             f"{v['degradation_rows_differing_all_inactive_both_phases']}; objective {v['objective']['after']}; "
             f"salvage vars {v['salvage_variables']} pinned to 0 {v['salvage_variables_all_pinned_to_zero_both_phases']}; "
             f"NL identical {v['nl']['identical']}")
    for k, v in checks.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    _log(f'wrote {os.path.relpath(path, REPO)} sha256={_sha(path)}')
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
