"""
P5.15 Addendum 39, task W45 -- WILL THE PILOT PASS THE SIGMA CALIBRATION ASSERTION?

ZERO SOLVES. A `SolveProfileGuard(permitted=())` is installed before any production import
and verified at exactly 0 solves / 0 process launches / 0 blocked at the end.

THE QUESTION. `shared_resources_planning._compute_common_admm_objective_scale` computes

    sigma_computed = max over TSO/DSO blocks b of | w_b * f_b(x_init) |,
    w_b = N_years(year) * N_days(day) / (1 + discount) ** (year - first_year)
          (`_get_admm_block_weight`),

where f_b is the block's `objective.expr` (the probability-weighted EXPECTED daily cost over
the block's market x operation scenarios) evaluated at the INITIALIZATION SOLVE of the block.
`_resolve_common_admm_objective_scale` then raises ValueError unless
1/F <= sigma_computed / sigma_fixed <= F (F = `objective_scale_assert_factor`, default 3.0,
sigma_fixed = the case file's `admm.objective_scale`, 93,635,360 for SRP1).

WHAT THIS SCRIPT DOES (no solve anywhere):
  A. parses the committed `[ADMM OF SCALE]` diagnostics (n, min/median/max, the ten largest
     weighted blocks, the fixed-sigma line, the ESSO AL line) of every measured instance, and
     recomputes each printed block's weight with production's `_get_admm_block_weight` so the
     raw per-day expected cost f_b = weighted_total / w_b is visible;
  B. reads (production reader, no model build) the committed 2x2 limit-gate instance and the
     pilot instance (paper years, 2x2), and compares the per-(network, 2025, day) scenario data
     with production's own digest helper `_update_scenario_digest`, the same labels
     `_compute_scenario_metadata` uses; records every block weight, the median block weight
     (`_compute_median_admm_block_weight`) and the pilot case's objective_scale / assert
     factor as production parses them;
  C. PREDICTION (formula preserved here, not in prose):
       floor = max over the limit-gate alpha = 0.50 arm's printed blocks b of
               |weighted_total_b| * w_pilot(b) / w_gate(b)
     This is a hard lower bound on the pilot's sigma_computed IF the pilot's 2025 blocks are
     the same problems as the gate's (same data -- checked in B --, same alpha = 0.50, same
     x = 0 candidate, same production code), because sigma is a MAX and those blocks are in
     the pilot's block set. It is NOT an estimate of the maximum, which lies in the later
     years (not measurable without the initialization solves).
     reference = the paper 5x5 measurement (same years and weights as the pilot, 5x5
     scenarios, pre-Addendum-38 code), reported with the observed W37 -> gate change of the
     TSO blocks under the Addendum-38 pin.

WHY NO BUILD. sigma is read from SOLVED initialization objectives; every committed zero-solve
build recorded `_compute_common_admm_objective_scale` on unsolved blocks as
"ValueError: ... no valid weighted TSO/DSO objective values were found" (e.g.
data/SRP1/Results/P515S44/scale_measurement/paper_cycle_snapoff_memfix_r1/build_record.json,
key `sigma_computed_on_unsolved_blocks`). A zero-solve build therefore cannot give the exact
number; B reads data only.

Invocation (attached, both streams captured, one run):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s52_sigma_check.py \
      --label r1 --scratch <session scratchpad dir> \
      > data/SRP1/Results/P515S52/sigma_check_r1_launch.log 2>&1
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard((), label='P5.15 W45 sigma check (zero solves)').install()

import numpy as np  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

STAGE = 'P5.15 Addendum 39 W45 -- will the pilot pass the sigma calibration assertion (zero solves)'
SCHEMA = 'p515_s52_sigma_check_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 36, 39',
             'data/SRP1/Results/P515S52/frozen_s52_spec_v22_5d8df1e8.json pilot']
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S52', 'sigma_check')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s52_sigma_check.lock')
DAYS = {'Spring': 92, 'Summer': 91, 'Autumn': 91, 'Winter': 91}
DISCOUNT = 0.02  # SRP1.json DiscountFactor; re-read from the case files in part B and checked
SRP1_YEARS = {'2025': 5, '2030': 5, '2035': 5}
GATE_YEARS = {'2025': 5}
PAPER_YEARS = dict(S.PAPER_YEARS)
PILOT_ALPHA = 0.5

# Every measured instance (committed stdout logs carrying the [ADMM OF SCALE] diagnostics).
MEASURED = [
    {'key': 'srp1_1x1_s44', 'instance': 'SRP1 1x1 (3 years x 4 days, 1 market x 1 operation)',
     'years': SRP1_YEARS, 'scenarios': '1x1', 'code': 'pre-Addendum-38 (S44 scale measurement)',
     'candidate': 'C* = ' + S.C_STAR_LABEL,
     'log': 'data/SRP1/Results/P515S44/scale_measurement/srp1_cycle_snapoff_r1/cycle_child_stdout.log'},
    {'key': 'srp1_1x1_bitwise_gate', 'instance': 'SRP1 1x1 (3 years x 4 days, 1 market x 1 operation)',
     'years': SRP1_YEARS, 'scenarios': '1x1', 'code': 'post-Addendum-38 (S51 SRP1 bitwise gate)',
     'candidate': 'C* (gate.json candidate_label; candidate_key 578636da...)',
     'log': 'data/SRP1/Results/P515S51/srp1_bitwise_gate/arm/stdout_s39_D.log'},
    {'key': 'srp1_1x1_s47_n5_4h', 'instance': 'SRP1 1x1 (3 years x 4 days, 1 market x 1 operation)',
     'years': SRP1_YEARS, 'scenarios': '1x1', 'code': 'pre-Addendum-38 (S47 campaign eval)',
     'candidate': 'eval 2a0ba8b2f3d3b99e_n5_4h_e1 (node-5 4 h unit, per the eval directory name)',
     'log': 'data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/2a0ba8b2f3d3b99e_n5_4h_e1/stdout_s39_D.log'},
    {'key': 'w37_2x2_x0', 'instance': 'W37 2x2 smoke (1 year x 4 days, 2 market x 2 operation)',
     'years': GATE_YEARS, 'scenarios': '2x2', 'code': 'pre-Addendum-38 (quadratic scenario-deviation penalty)',
     'candidate': 'x = 0',
     'log': 'data/SRP1/Results/P515S50/multiscenario_smoke/w37_2x2_r2/arm_x0/stdout_w37_x0.log'},
    {'key': 'w37_2x2_unit', 'instance': 'W37 2x2 smoke (1 year x 4 days, 2 market x 2 operation)',
     'years': GATE_YEARS, 'scenarios': '2x2', 'code': 'pre-Addendum-38 (quadratic scenario-deviation penalty)',
     'candidate': 'node-7 0.25 MVA / 1.0 MWh at 2025 (W37_MULTISCENARIO_SMOKE.md, arm unit) = the pilot unit arm',
     'log': 'data/SRP1/Results/P515S50/multiscenario_smoke/w37_2x2_r2/arm_unit/stdout_w37_unit.log'},
    {'key': 'gate_2x2_alpha0p5', 'instance': '2x2 limit gate (1 year x 4 days, 2 market x 2 operation)',
     'years': GATE_YEARS, 'scenarios': '2x2', 'code': 'post-Addendum-38, row 18 alpha = 0.50',
     'candidate': 'x = 0',
     'log': 'data/SRP1/Results/P515S51/2x2_limit_gate/limit_r1/arm_pilot/stdout_s51limit_pilot.log'},
    {'key': 'gate_2x2_alpha1000', 'instance': '2x2 limit gate (1 year x 4 days, 2 market x 2 operation)',
     'years': GATE_YEARS, 'scenarios': '2x2', 'code': 'post-Addendum-38, row 18 alpha = 1000',
     'candidate': 'x = 0',
     'log': 'data/SRP1/Results/P515S51/2x2_limit_gate/limit_r1/arm_large/stdout_s51limit_large.log'},
    {'key': 'paper_5x5', 'instance': 'paper (5 years x 4 days, 5 market x 5 operation)',
     'years': PAPER_YEARS, 'scenarios': '5x5', 'code': 'pre-Addendum-38 (S44 scale measurement)',
     'candidate': 'C* = ' + S.C_STAR_LABEL,
     'log': 'data/SRP1/Results/P515S44/scale_measurement/paper_cycle_snapoff_memfix_r1/cycle_child_stdout.log'},
]
AGENTS_PER_YEAR_DAY = 4  # 1 TSO + 3 DSOs (SRP1.json)

RE_SUMMARY = re.compile(r'\[ADMM OF SCALE\] n=(\d+) \| min=(\S+) \| median=(\S+) \| max=(\S+) \| selected=(\S+)')
RE_ENTRY = re.compile(r'^\s+(TSO|DSO) node=(\S+) year=(\d+) day=(\w+) \| base=(\S+) \| scenario=(\S+) \| total=(\S+)')
RE_FIXED = re.compile(r'Fixed sigma in force sigma_fixed=(\S+) \| sigma_computed=(\S+) \| ratio=(\S+) \| assert_factor=(\S+)')
RE_ESSO = re.compile(r'\[ADMM ESSO AL SCALE\] mode=(\S+) \| sigma=(\S+) \| median_block_weight=(\S+) \| al_scale_esso=(\S+)')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _weight(years, year, day):
    """Production's `_get_admm_block_weight` on a holder carrying the instance's years/days."""
    holder = SimpleNamespace(years=years, days=DAYS, discount_factor=DISCOUNT)
    return srp._get_admm_block_weight(holder, str(year), day)


# ======================================================================================
#  A. committed measurements
# ======================================================================================
def parse_measurement(spec):
    path = os.path.join(REPO, spec['log'])
    tracked = _git(['ls-files', '--', spec['log']])
    with open(path) as handle:
        lines = handle.read().splitlines()
    first = next(i for i, line in enumerate(lines) if RE_SUMMARY.search(line))
    n, vmin, vmed, vmax, sel = RE_SUMMARY.search(lines[first]).groups()
    entries = []
    for line in lines[first + 2:first + 12]:
        m = RE_ENTRY.match(line)
        if not m:
            raise RuntimeError(f'{spec["log"]}: expected a block line, got {line!r}')
        agent, node, year, day, base, scen, total = m.groups()
        w = _weight(spec['years'], year, day)
        entries.append({'agent': agent, 'node': None if node == 'None' else int(node), 'year': year,
                        'day': day, 'weighted_total': float(total), 'weighted_base': float(base),
                        'weighted_scenario_term': float(scen), 'block_weight': w,
                        'raw_expected_daily_cost': float(total) / w})
    fixed = next(RE_FIXED.search(l).groups() for l in lines[first:first + 40] if RE_FIXED.search(l))
    esso = next(RE_ESSO.search(l).groups() for l in lines[first:first + 40] if RE_ESSO.search(l))
    weights = [_weight(spec['years'], y, d) for y in spec['years'] for d in DAYS] * AGENTS_PER_YEAR_DAY
    record = dict(spec)
    record.update({
        'log_sha256': _sha256_file(path), 'log_tracked_in_git': bool(tracked),
        'summary_line_index': first,
        'n_blocks_printed': int(n), 'n_blocks_expected': AGENTS_PER_YEAR_DAY * len(spec['years']) * len(DAYS),
        'weighted_min': float(vmin), 'weighted_median': float(vmed), 'weighted_max': float(vmax),
        'sigma_computed_printed': float(fixed[1]), 'sigma_fixed_printed': float(fixed[0]),
        'ratio_printed': float(fixed[2]), 'assert_factor_printed': float(fixed[3]),
        'ratio_recomputed': float(sel) / float(fixed[0]),
        'argmax_block': {k: entries[0][k] for k in ('agent', 'node', 'year', 'day', 'block_weight',
                                                     'raw_expected_daily_cost')},
        'block_weight_max': max(weights), 'block_weight_min': min(weights),
        'block_weight_median_recomputed': float(np.median(weights)),
        'esso_al_line': {'mode': esso[0], 'sigma': float(esso[1]), 'median_block_weight': float(esso[2]),
                         'al_scale_esso': float(esso[3])},
        'top10': entries,
    })
    if record['n_blocks_printed'] != record['n_blocks_expected']:
        raise RuntimeError(f'{spec["key"]}: n={n} != expected {record["n_blocks_expected"]}')
    if not math.isclose(record['block_weight_median_recomputed'], float(esso[2]), rel_tol=1e-6):
        raise RuntimeError(f'{spec["key"]}: median block weight {record["block_weight_median_recomputed"]} '
                           f'!= printed {esso[2]} -- the weight holder does not match the instance')
    return record


def scan_all_committed_sigma_lines():
    """Scoped search: every tracked file under data/SRP1/Results carrying the fixed-sigma line."""
    out = subprocess.run(['git', 'grep', '-o', '-h', r'sigma_computed=[0-9.e+-]* | ratio=[0-9.e+-]*',
                          '--', 'data/SRP1/Results/*'], cwd=REPO, capture_output=True, text=True).stdout
    files = subprocess.run(['git', 'grep', '-l', 'Fixed sigma in force', '--', 'data/SRP1/Results/*'],
                           cwd=REPO, capture_output=True, text=True).stdout.split()
    counts = {}
    for line in out.splitlines():
        counts[line] = counts.get(line, 0) + 1
    return {'search': "git grep 'Fixed sigma in force' over tracked data/SRP1/Results/* at HEAD",
            'n_files': len(files),
            'distinct_sigma_ratio_strings': dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


# ======================================================================================
#  B. zero-solve data read: gate instance vs pilot instance, 2025 blocks
# ======================================================================================
def read_instance(label, years, out_dir, scratch):
    case, spec, changes = S.derive_case('srp1', {'years': years, 'num_market_scenarios': 2,
                                                 'num_operation_scenarios': 2})
    case_dir = os.path.join(out_dir, 'case')
    os.makedirs(case_dir, exist_ok=True)
    case_path = os.path.join(case_dir, f'SRP1__{label}.json')
    with open(case_path, 'w') as handle:
        json.dump(case, handle, indent='\t')
    rel = os.path.relpath(case_path, S.DATA_DIR)
    planning = srp.SharedResourcesPlanning(S.DATA_DIR, rel)
    planning.name = 'SRP1'
    read_dir = tempfile.mkdtemp(prefix=f'w45_{label}_', dir=scratch)
    planning.results_dir = os.path.join(read_dir, 'Results')
    planning.diagrams_dir = os.path.join(read_dir, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    planning.read_planning_problem()
    return planning, {'case_path': os.path.relpath(case_path, REPO), 'case_sha256': _sha256_file(case_path),
                      'changes_vs_source': changes, 'plot_output_redirected_to': read_dir}


def block_digests(planning, year):
    """Per-(agent, year, day) digests with production's `_update_scenario_digest` and the labels
    of `_compute_scenario_metadata` (market energy + flexibility prices; loads pd/qd/flex up/down;
    generators pg/qg)."""
    out = {}
    for day in sorted(planning.days, key=str):
        d = hashlib.sha256()
        srp._update_scenario_digest(d, ('market', 'energy', year, day), planning.cost_energy_p[year][day])
        srp._update_scenario_digest(d, ('market', 'flexibility', year, day), planning.cost_flex[year][day])
        out[f'market/{year}/{day}'] = d.hexdigest()
    groups = [('tso', None, planning.transmission_network)] + [
        ('dso', n, planning.distribution_networks[n]) for n in sorted(planning.distribution_networks, key=str)]
    for subsystem, node_id, nd in groups:
        for day in sorted(nd.days, key=str):
            net = nd.network[year][day]
            d = hashlib.sha256()
            prefix = (subsystem, node_id, net.name, year, day)
            for load in sorted(net.loads, key=lambda item: str(item.load_id)):
                lp = (*prefix, 'load', load.load_id)
                srp._update_scenario_digest(d, (*lp, 'pd'), load.pd)
                srp._update_scenario_digest(d, (*lp, 'qd'), load.qd)
                srp._update_scenario_digest(d, (*lp, 'flex_p_up'), load.flexibility.active_power.upward)
                srp._update_scenario_digest(d, (*lp, 'flex_p_down'), load.flexibility.active_power.downward)
            for gen in sorted(net.generators, key=lambda item: str(item.gen_id)):
                gp = (*prefix, 'generator', gen.gen_id)
                srp._update_scenario_digest(d, (*gp, 'pg'), gen.pg)
                srp._update_scenario_digest(d, (*gp, 'qg'), gen.qg)
            out[f'{subsystem}/{node_id}/{net.name}/{year}/{day}'] = d.hexdigest()
    return out


def instance_record(planning, meta):
    tn = planning.transmission_network
    weights = {}
    for label, nd in [('TSO', tn)] + [(f'DSO node={n}', planning.distribution_networks[n])
                                      for n in sorted(planning.distribution_networks)]:
        weights[label] = {f'{y}/{d}': srp._get_admm_block_weight(nd, y, d) for y in nd.years for d in nd.days}
    params = planning.params.admm
    return dict(meta, **{
        'years': dict(planning.years), 'days': dict(planning.days),
        'discount_factor': tn.discount_factor,
        'num_market_scenarios': planning.num_market_scenarios,
        'num_operation_scenarios': {'tso': tn.num_oper_scenarios,
                                    **{str(n): planning.distribution_networks[n].num_oper_scenarios
                                       for n in planning.distribution_networks}},
        'n_admm_blocks': sum(len(v) for v in weights.values()),
        'block_weights': weights,
        'median_block_weight_production': srp._compute_median_admm_block_weight(planning),
        'combined_scenario_checksum': planning.scenario_metadata['combined_scenario_checksum'],
        'admm_objective_scale': params.objective_scale,
        'admm_objective_scale_source': params.objective_scale_source,
        'admm_objective_scale_assert_factor': params.objective_scale_assert_factor,
        'esso_al_scale_cfg': params.esso_al_scale,
    })


# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True)
    parser.add_argument('--scratch', required=True, help='directory for the reader\'s plots (not committed)')
    args = parser.parse_args()
    out_dir = os.path.join(OUT_ROOT, args.label)
    if os.path.exists(out_dir):
        print(f'REFUSED: {out_dir} exists (write-once)', file=sys.stderr)
        return 2
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: lock held {OWN_LOCK_PATH}', file=sys.stderr)
        return 2
    os.close(fd)
    try:
        os.makedirs(out_dir)
        record = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'label': args.label,
                  'argv': sys.argv, 'interpreter': sys.executable,
                  'script_sha256': _sha256_file(os.path.abspath(__file__)),
                  'git_head': _git(['rev-parse', 'HEAD']),
                  'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
                  'started_utc': _utc(),
                  'production_sources': {
                      'shared_resources_planning.py': _sha256_file(os.path.join(REPO, 'shared_resources_planning.py')),
                      'admm_parameters.py': _sha256_file(os.path.join(REPO, 'admm_parameters.py'))},
                  'sigma_definition': (
                      'sigma_computed = max_b |w_b * f_b| over every TSO and DSO (year, day) block '
                      '(shared_resources_planning._compute_common_admm_objective_scale), f_b = '
                      'pe.value(model.objective.expr) after the initialization solve, w_b = '
                      '_get_admm_block_weight = N_years * N_days / (1+discount)^(year-first_year); '
                      'raise ValueError unless 1/F <= sigma_computed/sigma_fixed <= F '
                      '(_resolve_common_admm_objective_scale)')}

        # A
        measured = [parse_measurement(spec) for spec in MEASURED]
        record['measured'] = measured
        record['all_committed_fixed_sigma_lines'] = scan_all_committed_sigma_lines()

        # B
        planning_gate, meta_gate = read_instance('w45_gate_2x2', GATE_YEARS, out_dir, args.scratch)
        gate = instance_record(planning_gate, meta_gate)
        gate_digests = block_digests(planning_gate, 2025)
        del planning_gate
        planning_pilot, meta_pilot = read_instance('w45_pilot_2x2', PAPER_YEARS, out_dir, args.scratch)
        pilot = instance_record(planning_pilot, meta_pilot)
        pilot_digests = block_digests(planning_pilot, 2025)
        del planning_pilot
        committed_gate_launch = json.load(open(os.path.join(
            REPO, 'data/SRP1/Results/P515S51/2x2_limit_gate/limit_r1/launch.json')))
        mismatched = sorted(k for k in gate_digests if gate_digests[k] != pilot_digests.get(k))
        record['instances'] = {'gate_2x2': gate, 'pilot_2x2': pilot}
        record['identity_2025'] = {
            'gate_case_sha256_equals_committed_limit_r1': (
                gate['case_sha256'] == committed_gate_launch['derived_case']['sha256']),
            'gate_scenario_checksum_equals_committed_limit_r1': (
                gate['combined_scenario_checksum'] == committed_gate_launch['scenario_checksum']),
            'n_2025_digests_compared': len(gate_digests),
            'mismatched_2025_digests': mismatched,
            'all_2025_block_data_identical': not mismatched and set(gate_digests) == set(pilot_digests),
            'digests_gate': gate_digests, 'digests_pilot': pilot_digests,
        }

        # C
        gate_meas = next(m for m in measured if m['key'] == 'gate_2x2_alpha0p5')
        paper_meas = next(m for m in measured if m['key'] == 'paper_5x5')
        sigma_fixed = pilot['admm_objective_scale']
        factor = pilot['admm_objective_scale_assert_factor']
        floor_rows = []
        for e in gate_meas['top10']:
            w_gate = _weight(GATE_YEARS, e['year'], e['day'])
            w_pilot = _weight(PAPER_YEARS, e['year'], e['day'])
            floor_rows.append({'block': f"{e['agent']} node={e['node']} {e['year']} {e['day']}",
                               'gate_weighted_abs': abs(e['weighted_total']), 'w_gate': w_gate,
                               'w_pilot': w_pilot, 'pilot_weighted_abs': abs(e['weighted_total']) * w_pilot / w_gate})
        floor = max(r['pilot_weighted_abs'] for r in floor_rows)
        # the same bound at full precision: the gate arm's sigma_computed IS its argmax block's
        # |w * f| (gate.json, cycle 0), and every gate block is a 2025 block (w_pilot/w_gate const.)
        gate_json_path = 'data/SRP1/Results/P515S51/2x2_limit_gate/limit_r1/gate.json'
        gate_json = json.load(open(os.path.join(REPO, gate_json_path)))
        gate_sigma_exact = gate_json['arms']['pilot']['cycle_trajectory'][0]['sigma_computed']
        argmax = gate_meas['argmax_block']
        weight_ratio = _weight(PAPER_YEARS, argmax['year'], argmax['day']) / _weight(GATE_YEARS, argmax['year'], argmax['day'])
        floor_exact = gate_sigma_exact * weight_ratio
        w37_x0 = next(m for m in measured if m['key'] == 'w37_2x2_x0')
        gate_tenth = abs(gate_meas['top10'][-1]['weighted_total'])
        tso_w37 = [e for e in w37_x0['top10'] if e['agent'] == 'TSO']
        record['prediction'] = {
            'formula_floor': ('floor = max_b |weighted_total_b(gate alpha=0.50)| * w_pilot(b) / w_gate(b) '
                              'over the gate arm\'s ten printed blocks (all 2025); a hard lower bound on the '
                              'pilot sigma_computed conditional on identity_2025 and the same alpha/candidate/code'),
            'floor_rows': floor_rows,
            'sigma_floor_from_printed_top10': floor,
            'gate_sigma_exact': {'path': gate_json_path, 'sha256': _sha256_file(os.path.join(REPO, gate_json_path)),
                                 'value': gate_sigma_exact},
            'weight_ratio_pilot_over_gate_2025': weight_ratio,
            'sigma_floor': floor_exact,
            'ratio_floor': floor_exact / sigma_fixed,
            'band': [1.0 / factor, factor],
            'ratio_floor_over_lower_band_edge': (floor_exact / sigma_fixed) * factor,
            'paper_5x5_reference': {
                'sigma_computed': paper_meas['sigma_computed_printed'], 'ratio': paper_meas['ratio_recomputed'],
                'argmax': paper_meas['argmax_block'],
                'largest_DSO_block': next({k: e[k] for k in ('node', 'year', 'day', 'weighted_total')}
                                          for e in paper_meas['top10'] if e['agent'] == 'DSO'),
                'ratio_of_largest_DSO_block': next(abs(e['weighted_total']) for e in paper_meas['top10']
                                                   if e['agent'] == 'DSO') / sigma_fixed,
            },
            'tso_change_w37_to_gate_2025': {
                'w37_x0_TSO_blocks_printed': [{'day': e['day'], 'abs_weighted': abs(e['weighted_total'])}
                                              for e in tso_w37],
                'gate_alpha0p5_no_TSO_block_in_top10_so_every_TSO_abs_weighted_below': gate_tenth,
                'implied_min_relative_drop': [1.0 - gate_tenth / abs(e['weighted_total']) for e in tso_w37],
            },
            'upper_edge_sigma': factor * sigma_fixed,
            'upper_edge_over_paper_sigma': factor * sigma_fixed / paper_meas['sigma_computed_printed'],
        }
        record['finished_utc'] = _utc()
    finally:
        GUARD.uninstall()
        os.remove(OWN_LOCK_PATH)
    failures = GUARD.verify(0)
    record['solve_profile'] = {'declared': 'zero solves', 'counts': GUARD.counts, 'verify_failures': failures}
    with open(os.path.join(out_dir, 'sigma_check.json'), 'w') as handle:
        json.dump(record, handle, indent=1, default=str)
    manifest = {}
    for root, _, files in os.walk(out_dir):
        for name in sorted(files):
            p = os.path.join(root, name)
            manifest[os.path.relpath(p, REPO)] = _sha256_file(p)
    manifest[os.path.relpath(os.path.abspath(__file__), REPO)] = record['script_sha256']
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1)
    p = record['prediction']
    print(f"[W45] floor sigma={p['sigma_floor']:.6e} ratio={p['ratio_floor']:.6f} band={p['band']} "
          f"floor/lower_edge={p['ratio_floor_over_lower_band_edge']:.4f}")
    print(f"[W45] identity_2025 all identical: {record['identity_2025']['all_2025_block_data_identical']} "
          f"(n={record['identity_2025']['n_2025_digests_compared']})")
    print(f"[W45] solve profile: {GUARD.counts} failures={failures}")
    if failures:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
