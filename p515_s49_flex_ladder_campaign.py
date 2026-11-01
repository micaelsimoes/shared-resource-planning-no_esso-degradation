"""
P5.15 Addendum 34 (task W33, item 4) -- the FLEXIBILITY-PRICE LADDER of frozen spec v19 (`flex_price_ladder`)
through the campaign harness (`p515_s44_campaign_harness.py`) with the evaluation option `flex_price_multiplier`
(W33 item 1, 2e82a23f; zero-solve checks there; two-cycle bitwise gate at m = 1.0 PASS, 51e74279).

  MODEL VARIANT -- flexibility price x m   (m in {1.5, 2, 3}; the DSO `cost_flex` profile, every year / day / hour,
  growth included, multiplied uniformly in the child before the DSO models are built; no workbook edit)
  under the ageing BASELINE (C2 + phi_cal 0.985 + soh_min 0.70), declared (`configuration.ess_ageing_baseline`).

SIX EVALUATIONS, one campaign root, one frozen spec: m in {1.5, 2, 3} x {x = 0, the smallest node-7 4 h unit
(0.25 MVA / 1.0 MWh at node 7, 2025; nodes 5 and 9 empty)}, in the order
    x0_m1p5, n7_4h_e1_m1p5, x0_m2, n7_4h_e1_m2, x0_m3  (wave 1, 5 at concurrency 5)
    n7_4h_e1_m3                                         (wave 2)
AA-on case file (declared), cap 500, 10 consecutive all-pass cycles, concurrency 5, NO overrides, NO ageing model
variant, NO post-certification. Stop rule as a1a (STOP_RULE): evaluated after wave 1 before wave 2. The m = 1 pair
is the committed baseline and is NOT re-run: x = 0 from s48_x0_capture (Q(0) = 653,859,461.2279255, equal to A0's
x0), the unit from S2 (s47_recert n7_4h_e1); both pinned below by path and sha256 and re-checked at --freeze and
--run.

RECORDED PER POINT (campaign_results.json `points`; objective convention on every table: Q = certified GROSS
operational cost, settlement excluded): status (non-certified with cause), cycles, certification cycle, Q, the bar,
the rule-ten terminal-step-to-threshold ratios (production and gross), Boyd terminal ratios, the multiplier and its
label, the harness read-back (pre-run probes; post-run on the run's own DSO models), the flexibility cost per DSO
(sum over that DSO's blocks of component_levels_terminal.json `weighted.flexibility_cost_internal`, model block
weights; also divided by m = the same flexibility volume at the base price), the solve reconciliation (observed
== 51 x (cycles + 1) + every attempted retry, recovered or not -- see SOLVE_RECONCILIATION), wall time, peak RSS,
AA action counts, per-cycle trajectory path + sha256.
PER m (`per_m`): value(m) = Q_m(0) - Q_m(unit); value - I with I = I(unit) = 317,957.01 (the pinned W2 table);
the resolution bar_unit + bar_x0 (both at the same m) and whether |value - I| exceeds it; the value per full cycle
value / T with T the unit's cell-side full-cycle throughput under W25's definition (p515_s47_tso_marginal_cost,
T4, r2 block weights: T = sum_b w_b sum_p (eff_ch pch + pdch / eff_dch) dt / 2; `unit_throughput_w25` uses W25's
own `case_structure` and `_esso_capture` BY IMPORT and reproduces W25's committed T for the S2 record bitwise,
checked at --freeze and --run); the resolution per cycle; the DSO flexibility saving FC_m(0) - FC_m(unit) per DSO.
The m = 1 row is the pinned reference pair, computed by the same functions from the committed records.

SOLVE_RECONCILIATION (reported per point; a mismatch is listed, not rationalized): the campaign children carry no
exact-count guard of their own; the parent re-derives observed == base + retries from each child's own record
(`solve_profile.observed.permitted_solve`) and `network_failures_s39_D.jsonl`, crediting [recovery_attempted] +
[tier2_attempted] for EVERY network-failure event (so an unrecovered block's retries are counted -- the gap noted at
78a9b230 does not apply), and declaring the reconciliation unsupported when an ESSO recovery or indeterminate event
occurs (same definition as the W33 gate, p515_s49_flex_price_gate.event_level_reconciliation, restated here
because importing the gate module would install its solve-permitting guard in this never-solving parent).

Two modes, attached, both streams captured, never detached:
  --freeze                    ZERO SOLVES: preconditions, pins, references, W25 reproduction, I(x), rule eleven,
                              `freeze_campaign_spec` + validation. Prints the --run command.
  --run --spec-sha256 <sha>   loads THAT spec (the root must hold only it), re-checks everything plus the harness /
                              case-file / ESS-params / script sha256, the memory preflight (refusing), takes the
                              campaign lock, evaluates in two waves, writes campaign_results.json and
                              campaign_manifest_sha256.json.
The parent never solves: SolveProfileGuard(permitted=()) installed before any model import, verified at exactly 0
(W25's module guard, installed when its functions are imported, is verified at 0 as well). Exit codes (--run): 0
every point certified; 3 STOP_FOR_REVIEW (stop rule fired, harness clean); 2 a non-certified point (harness clean);
1 harness / guard / read-back / precondition failure.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_ladder_campaign.py --freeze \\
      > data/SRP1/Results/P515S49/campaign_s49_flex_ladder_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_ladder_campaign.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S49/campaign_s49_flex_ladder_launch.log 2>&1
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S49 flexibility-price ladder parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
FLEX_LABEL = H.FLEX_PRICE_LABEL
_P49 = os.path.join('data', 'SRP1', 'Results', 'P515S49')
_P48 = os.path.join('data', 'SRP1', 'Results', 'P515S48')
_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CAMPAIGN_ID = 's49_flex_ladder'
CAMPAIGN_ROOT = os.path.join(REPO, _P49, 'campaign_s49_flex_ladder')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
ACTIVE_NODES = (5, 7, 9)
YEAR = 2025
MULTIPLIERS = (1.5, 2.0, 3.0)
UNIT_NODE = 7
UNIT = (0.25, 1.0)
CASE_JSON_REL = os.path.join('data', 'SRP1', 'SRP1.json')

ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
SPEC_V19 = {'path': os.path.join(_P49, 'frozen_s49_spec_v19_f8adcc97.json'),
            'sha256': 'f8adcc9787f2b6e6761d037b8971e89fe26a408787f683331f425ffc73b09121'}
ESS_PARAMS_FILE = {'path': H.ESS_PARAMS_FILE_REL,
                   'sha256': '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706',
                   'commit': '2466401d', 'note': 'the ageing BASELINE edit of W21 (Addendum 30)'}
A0_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_spec_s45_a0_c7_9d08ad2f.json'),
           'sha256': '9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34'}
A0_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json'),
              'sha256': '423678b98c0ca9a26dbc76a1b46a5e36a51c8ccccc0459253020bfe1061653b4', 'label': 'x0'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'field': 'candidates.<label>.I_new_eur, matched by candidate_key'}
# The m = 1 references (committed; NOT re-run).
REF_X0 = {'eval_dir': os.path.join(_P48, 'x0_capture', 'evals', 'd2c96b1480402a3b_x0'),
          'evaluation_record_sha256': 'a5ca47231348be7da5cb044c5ee4cdd6b03accfc084125a4b8402ae24df98939',
          'component_levels_terminal_sha256': '06f18a94d6bb5c4d334e09cd9ce16fd692612cfb2bbfcd4e03615947e24b4766',
          'eval_key': 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7',
          'campaign': 's48_x0_capture (spec campaign_spec_s48_x0_capture_4a50c0e2.json), BASELINE declared',
          'expected_Q': 653859461.2279255, 'expected_cycles': 132}
REF_UNIT = {'eval_dir': os.path.join(_P47, 'campaign_s47_recert', 'evals', 'bd504ecf5a288d44_n7_4h_e1'),
            'evaluation_record_sha256': 'd15f5e5ec88d6abacc0d80923ac7ac1ed84cec1791b4262405b6deba5ab976e8',
            'component_levels_terminal_sha256': '9c3c563b1c14e6c469d8a915c7d68f36ce9ba8ca398268a3b50a1992569d13a6',
            'eval_key': 'bd504ecf5a288d447e53ef6f7e8090b017d15549b257aa095ff4f95890a06e42',
            'campaign': 's47_recert (spec campaign_spec_s47_recert_902f93aa.json), S2, BASELINE declared',
            'campaign_manifest': os.path.join(_P47, 'campaign_s47_recert', 'campaign_manifest_sha256.json'),
            'campaign_manifest_sha256': '519f9102a741b4cbc56a41827a322d0049f4c8c6eb70eddf8fab790bebffb94d',
            'expected_Q': 653600033.4601703, 'expected_cycles': 112}
W25_RESULTS = {'path': os.path.join(_P47, 'tso_marginal_cost', 'tso_marginal_cost.json'),
               'sha256': '5acbd2b32d924b47f23650142076180c420bd0ba201b0a63d4d89f4b671b7a31', 'commit': 'b7aca555',
               'field': 'T4.baseline.r2.T_cell_throughput_full_cycle_mwh / captured_eur_per_mwh_cycle'}
PRIOR_W33 = {'checks': {'path': os.path.join(_P49, 'flex_price_checks', 'flex_price_checks.json'), 'commit': '2e82a23f',
                        'expect': 'all_ok true'},
             'gate': {'path': os.path.join(_P49, 'flex_price_gate', 'gate.json'), 'commit': '51e74279',
                      'expect': 'gate_pass true'}}

OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); value(m) = Q_m(0) - '
                        'Q_m(unit); terminal salvage and net_operational_recourse = gross - salvage reported, excluded. '
                        'At m != 1 the DSO flexibility cost inside Q is priced at m x cost_flex.')
VALUE_DEFINITION = ('value(m) = Q_m(0) - Q_m(unit); value - I with I = I(unit) from the pinned W2 table; resolution = '
                    'bar_unit + bar_x0 (record.bar of each, same m): |value - I| <= resolution is INDETERMINATE '
                    '(CLAUDE.md: a difference smaller than the stopping error is not a result; the bar is local and '
                    'bounds stopping slack only); value per full cycle = value / T, T = W25 T4 r2 cell-side '
                    'full-cycle throughput of the unit at the same m; resolution per cycle = resolution / T')
FLEX_COST_DEFINITION = ('flexibility cost per DSO = sum over that DSO\'s 12 blocks of component_levels_terminal.json '
                        'blocks["DSO|<node>|<year>|<day>"].weighted.flexibility_cost_internal (pe.value(total_flex_cost) '
                        'x admm_block_weight = num_years x num_days / 1.02^(y - 2025)); priced at m x cost_flex; '
                        '"at base price" = divided by m (the cost is linear in cost_flex); saving = FC(0) - FC(unit)')
STOP_RULE = ('as a1a (Planner task W15, spec v15 execution.barrier): after every wave, STOP (launch no further wave) if '
             '(a) 2 or more non-certified points fall in ONE region (node+duration ladder) or (b) 3 or more '
             'non-certified points occur overall. A single isolated non-certified point does NOT stop the campaign. '
             'Points already running always finish; unlaunched points are recorded as not_launched_stop_rule and '
             'campaign_results.json carries STOP_FOR_REVIEW true (exit 3).')
REGION_NOTE = ('regions as a1a (_region_key: the node+duration ladder, x0 for the empty candidate), NOT split by m: '
               'x0 at three prices is one region, the unit at three prices another')

GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 11 * GIB // 4  # 2.75 GiB (A0's per-child measure)
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE_TEMPLATE = ('hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {concurrency} x '
                        '{per_child:g} GiB')
MEMORY_RULE = MEMORY_RULE_TEMPLATE.format(concurrency=CONCURRENCY, per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside)')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 34 (flexibility price as a break-even axis)',
    'data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json flex_price_ladder',
    'Planner task W33 item 4 (6 evaluations, m in {1.5, 2, 3} x {x = 0, n7 0.25/1.0 2025}, baseline declared, AA-on '
    'case file, cap 500, 10 cycles, concurrency 5 in two waves 5 + 1, no post-certification, stop rule as a1a)',
    '2e82a23f (harness override + zero-solve checks), 51e74279 (two-cycle bitwise gate at m = 1.0)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', 'p515_s49_flex_price_checks.py', 'p515_s49_flex_price_gate.py',
                     'p515_s47_tso_marginal_cost.py', CASE_JSON_REL)


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _banner(lines):
    _log('!' * 100)
    for line in lines:
        _log(f'!!! {line}')
    _log('!' * 100)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _nodes_full(partial):
    nodes = {n: (0.0, 0.0) for n in ACTIVE_NODES}
    for node, (s_val, e_val) in partial.items():
        nodes[int(node)] = (float(s_val), float(e_val))
    return nodes


def _key_of(nodes):
    return H.candidate_key(H.canonical_candidate(nodes, investment_year=YEAR))


def _eval_key(nodes, m):
    return H.evaluation_key(_key_of(nodes), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            flex_price_multiplier=m)


def _region_key(canonical):
    """a1a's stop-rule region: the node+duration ladder (x0 for the empty candidate)."""
    nz = {int(n): v for n, v in canonical['nodes'].items() if float(v[0]) or float(v[1])}
    if not nz:
        return f"x0_y{canonical['investment_year']}"
    return '+'.join(f'n{n}_{float(nz[n][1]) / float(nz[n][0]):g}h' for n in sorted(nz)) + \
        f"_y{canonical['investment_year']}"


def _m_tag(m):
    return f'{m:g}'.replace('.', 'p')


def points():
    """(label, nodes, m, instance) in launch order: pairs by ascending m."""
    out = []
    for m in MULTIPLIERS:
        out.append((f'x0_m{_m_tag(m)}', _nodes_full({}), m, 'x0'))
        out.append((f'n7_4h_e1_m{_m_tag(m)}', _nodes_full({UNIT_NODE: UNIT}), m, 'unit'))
    return tuple(out)


def waves(labels):
    return [labels[i:i + CONCURRENCY] for i in range(0, len(labels), CONCURRENCY)]


# ======================================================================================================================
#  derived figures (pure functions of an evaluation's own artifacts)
# ======================================================================================================================
def solves_per_cycle_from_case_file():
    """(1 + n_dso) x n_years x n_days + n_esso (W10.derive_solves_from_case_file's formula, restated -- W10's module
    installs a solve-permitting guard at import)."""
    case = _load(CASE_JSON_REL)
    n_yd = len(case['Years']) * len(case['Days'])
    n_dso = len(case['DistributionNetworks'])
    return (1 + n_dso) * n_yd + len({dn['connection_node_id'] for dn in case['DistributionNetworks']})


def solve_reconciliation(rec, per_cycle):
    """observed == per_cycle x (cycles + 1) + every attempted retry (see SOLVE_RECONCILIATION in the docstring)."""
    rec = rec or {}
    observed = ((rec.get('solve_profile') or {}).get('observed') or {}).get('permitted_solve')
    summary = rec.get('network_failures_summary') or {}
    path = summary.get('path')
    events = []
    if path and os.path.isfile(os.path.join(REPO, path)):
        with open(os.path.join(REPO, path)) as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    events = [e for e in events if e.get('record_type', 'network_block') == 'network_block' and e.get('class')]
    retries = sum(int(bool(e.get('recovery_attempted'))) + int(bool(e.get('tier2_attempted'))) for e in events)
    n_esso = int(summary.get('n_esso_recovery_events') or 0)
    n_indet = sum(1 for e in events if e.get('class') == 'indeterminate')
    cycles = rec.get('cycles_run')
    supported = (cycles is not None and n_esso == 0 and n_indet == 0
                 and len(events) == int(summary.get('n_blocks') or 0))
    base = per_cycle * (cycles + 1) if cycles is not None else None
    expected = base + retries if (supported and base is not None) else None
    return {'observed': observed, 'base': base, 'solves_per_cycle': per_cycle, 'retries_credited': retries,
            'events_by_class': dict(Counter(e.get('class') for e in events)), 'n_esso_recovery_events': n_esso,
            'supported': supported, 'expected': expected,
            'holds': (expected is not None and observed == expected),
            'recovered_only_identity': (base + summary.get('classes', {}).get('recovered_tier1', 0)
                                        + 2 * summary.get('classes', {}).get('recovered_tier2', 0))
            if base is not None and summary.get('classes') else None}


def flex_cost_per_dso(eval_dir_rel, m):
    cl = _load(os.path.join(eval_dir_rel, 'component_levels_terminal.json'))
    per, tso = {}, 0.0
    for key, blk in cl['blocks'].items():
        parts = key.split('|')
        val = blk['weighted']['flexibility_cost_internal']
        if parts[0] == 'DSO':
            per.setdefault(parts[1], {'weighted_r2_eur': 0.0, 'n_blocks': 0})
            per[parts[1]]['weighted_r2_eur'] += val
            per[parts[1]]['n_blocks'] += 1
        else:
            tso += val
    for v in per.values():
        v['at_base_price_eur'] = v['weighted_r2_eur'] / m
    return {'per_dso': dict(sorted(per.items())), 'tso_weighted_r2_eur': tso,
            'total_dso_weighted_r2_eur': sum(v['weighted_r2_eur'] for v in per.values()),
            'multiplier': m, 'component_levels_sha256': H.sha256_file(os.path.join(REPO, eval_dir_rel,
                                                                                   'component_levels_terminal.json'))}


_W25 = {}


def _w25():
    """W25's module, imported on first use (it installs its own zero-permitted guard and chdir(REPO) at import)."""
    if 'mod' not in _W25:
        import p515_s47_tso_marginal_cost as W25  # noqa: E402 -- case_structure / _esso_capture, BY IMPORT
        _W25['mod'] = W25
    return _W25['mod']


def unit_throughput_w25(eval_dir_rel, cycles_run):
    """W25's T (p515_s47_tso_marginal_cost.decomposition._conv, r2 weighting), from the evaluation's terminal ESSO
    capture of node 7: T = sum_b w_r2 * sum_p (eff_ch * pch + pdch / eff_dch) * DT_H / 2, same order of operations;
    efficiencies = SharedEnergyStorage defaults (as W25). W25's case_structure / _esso_capture BY IMPORT."""
    W25 = _w25()
    from shared_energy_storage import SharedEnergyStorage
    _case, years, days, blocks = W25.case_structure()
    rows, rel = W25._esso_capture(eval_dir_rel, int(cycles_run), years, days)
    ses = SharedEnergyStorage()
    eff_ch, eff_dch = float(ses.eff_ch), float(ses.eff_dch)
    single = len(rows) == 288 and all(r['y_inv'] == 0 for r in rows.values())
    T = 0.0
    for b in blocks:
        w = b['w_r2']
        yk, dk = b['year'], b['day']
        pch = [rows[(yk, dk, p)]['pch'] for p in range(24)]
        pdch = [rows[(yk, dk, p)]['pdch'] for p in range(24)]
        Tb = sum(eff_ch * pch[p] + pdch[p] / eff_dch for p in range(24)) * W25.DT_H / 2.0
        T += w * Tb
    return {'T_cell_throughput_full_cycle_mwh': T, 'esso_capture_file': rel,
            'esso_capture_sha256': H.sha256_file(os.path.join(REPO, rel)), 'eff_ch': eff_ch, 'eff_dch': eff_dch,
            'single_cohort_288_rows': single, 'weighting': 'w_r2 = num_years * num_days / 1.02^(y - 2025) (W25 T4 r2)'}


def _rule_ten(rec, rows):
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  and prev.get('cycle') == (last.get('cycle') or 0) - 1 else None)
    tol = last.get('objective_tolerance')
    rt = rec.get('rule_ten') or {}
    return {'terminal_step_over_threshold_production': rt.get('terminal_step_over_threshold'),
            'terminal_objective_change_abs_production_net': rt.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': rt.get('terminal_objective_tolerance'),
            'terminal_gross_step_abs': gross_step,
            'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol) else None,
            'boyd_terminal_ratio_max_per_channel': rt.get('boyd_terminal_ratio_max_per_channel')}


def _per_cycle_rows(rec):
    path = rec.get('per_cycle_record_path')
    if not path or not os.path.isfile(os.path.join(REPO, path)):
        return {'path': path, 'present': False}, []
    with open(os.path.join(REPO, path)) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {'path': path, 'present': True, 'sha256': H.sha256_file(os.path.join(REPO, path)), 'n_rows': len(rows),
            'n_rows_equals_cycles_run': len(rows) == rec.get('cycles_run')}, rows


def per_m_row(m, x0, unit, i_x):
    """value / resolution / per-cycle figures from two point results (x0, unit) at the same m."""
    q0, qx = x0.get('Q'), unit.get('Q')
    b0, bx = x0.get('bar'), unit.get('bar')
    value = (q0 - qx) if (q0 is not None and qx is not None) else None
    res = (bx + b0) if (bx is not None and b0 is not None) else None
    vmi = (value - i_x) if value is not None else None
    T = (unit.get('throughput') or {}).get('T_cell_throughput_full_cycle_mwh')
    if vmi is None or res is None:
        reading = 'not available (a point is not certified)'
    elif abs(vmi) <= res:
        reading = 'INDETERMINATE (|value - I| <= resolution)'
    else:
        reading = 'pays (value - I > resolution)' if vmi > 0 else 'does not pay (I - value > resolution)'
    fc0, fcx = (x0.get('flex_cost') or {}).get('per_dso') or {}, (unit.get('flex_cost') or {}).get('per_dso') or {}
    saving = {n: {'weighted_r2_eur': fc0[n]['weighted_r2_eur'] - fcx[n]['weighted_r2_eur'],
                  'at_base_price_eur': fc0[n]['at_base_price_eur'] - fcx[n]['at_base_price_eur']}
              for n in sorted(set(fc0) & set(fcx))}
    return {'m': m, 'Q_x0': q0, 'Q_unit': qx, 'bar_x0': b0, 'bar_unit': bx, 'value_eur': value, 'I_eur': i_x,
            'value_minus_I_eur': vmi, 'resolution_eur': res,
            'value_minus_I_over_resolution': (vmi / res) if (vmi is not None and res) else None,
            'reading': reading, 'T_cell_throughput_full_cycle_mwh': T,
            'value_per_full_cycle_eur_per_mwh': (value / T) if (value is not None and T) else None,
            'I_per_full_cycle_eur_per_mwh': (i_x / T) if T else None,
            'resolution_per_full_cycle_eur_per_mwh': (res / T) if (res is not None and T) else None,
            'status_x0': x0.get('status'), 'status_unit': unit.get('status'),
            'rule_ten_gross_ratio_x0': (x0.get('rule_ten') or {}).get('terminal_gross_step_over_threshold'),
            'rule_ten_gross_ratio_unit': (unit.get('rule_ten') or {}).get('terminal_gross_step_over_threshold'),
            'dso_flexibility_saving_per_dso': saving,
            'dso_flexibility_saving_total_weighted_r2_eur': (sum(v['weighted_r2_eur'] for v in saving.values())
                                                             if saving else None)}


# ======================================================================================================================
#  pins and references
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v19', SPEC_V19), ('ess_params_file', ESS_PARAMS_FILE), ('a0_spec', A0_SPEC),
                      ('a0_results', A0_RESULTS), ('investment_cost_results', INVESTMENT_COST_RESULTS),
                      ('w25_results', W25_RESULTS),
                      ('ref_unit_campaign_manifest', {'path': REF_UNIT['campaign_manifest'],
                                                      'sha256': REF_UNIT['campaign_manifest_sha256']})):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == pin['sha256'] and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for name, pin in PRIOR_W33.items():
        path = os.path.join(REPO, pin['path'])
        present = os.path.isfile(path)
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        payload = _load(pin['path']) if present else {}
        ok_flag = payload.get('all_ok') if name == 'checks' else payload.get('gate_pass')
        entry = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': in_head, 'git_tracked': tracked,
                 'sha256': H.sha256_file(path) if present else None, 'flag': ok_flag, 'expect': pin['expect']}
        out[f'w33_{name}'] = entry
        if not (present and tracked and in_head and ok_flag is True):
            failures.append(f'w33_{name}: {entry}')
    return out, failures


def reference_inputs():
    """The m = 1 references, I(unit), and the W25 reproduction -- all from committed artifacts, zero solves."""
    failures, checks = [], {}
    refs = {}
    for name, ref in (('x0', REF_X0), ('unit', REF_UNIT)):
        rec_path = os.path.join(ref['eval_dir'], 'evaluation_record.json')
        cl_path = os.path.join(ref['eval_dir'], 'component_levels_terminal.json')
        rec = _load(rec_path)
        checks[f'ref_{name}_record_sha256'] = H.sha256_file(os.path.join(REPO, rec_path)) == ref['evaluation_record_sha256']
        checks[f'ref_{name}_component_levels_sha256'] = (H.sha256_file(os.path.join(REPO, cl_path))
                                                         == ref['component_levels_terminal_sha256'])
        checks[f'ref_{name}_tracked'] = bool(H._git(['ls-files', '--', rec_path, cl_path]).strip())
        checks[f'ref_{name}_certified'] = rec.get('status') == 'certified'
        checks[f'ref_{name}_Q'] = rec.get('certified_cost') == ref['expected_Q']
        checks[f'ref_{name}_cycles'] = rec.get('cycles_run') == ref['expected_cycles']
        checks[f'ref_{name}_eval_key_equals_m1_key'] = (rec.get('eval_key') == ref['eval_key'] == _eval_key(
            _nodes_full({}) if name == 'x0' else _nodes_full({UNIT_NODE: UNIT}), 1.0))
        checks[f'ref_{name}_no_flex_multiplier_recorded'] = 'flex_price_multiplier' not in rec
        _traj, rows = _per_cycle_rows(rec)
        refs[name] = {'label': f'm = 1 reference ({name})', 'campaign': ref['campaign'], 'eval_dir': ref['eval_dir'],
                      'evaluation_record_sha256': ref['evaluation_record_sha256'], 'eval_key': rec.get('eval_key'),
                      'candidate_key': rec.get('candidate_key'), 'status': rec.get('status'),
                      'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
                      'Q': rec.get('certified_cost'), 'bar': (rec.get('bar') or {}).get('value'),
                      'rule_ten': _rule_ten(rec, rows), 'flex_cost': flex_cost_per_dso(ref['eval_dir'], 1.0)}
    a0 = _load(A0_RESULTS['path'])['points'][A0_RESULTS['label']]
    checks['a0_x0_Q_equals_s48_x0_Q'] = a0.get('certified_cost_gross_settlement_excluded') == refs['x0']['Q']
    table = _load(INVESTMENT_COST_RESULTS['path'])['candidates']
    unit_key = _key_of(_nodes_full({UNIT_NODE: UNIT}))
    i_vals = sorted({c.get('I_new_eur') for c in table.values() if c.get('candidate_key') == unit_key}, key=repr)
    checks['I_unit_unique_in_w2_table'] = len(i_vals) == 1 and i_vals[0] is not None
    i_unit = i_vals[0] if checks['I_unit_unique_in_w2_table'] else None
    checks['I_unit_is_317957_01'] = i_unit is not None and round(i_unit, 2) == 317957.01
    # W25 reproduction: T of the S2 unit record, bitwise against W25's committed T4 r2 figure
    w25 = _load(W25_RESULTS['path'])['T4']['baseline']
    thr = unit_throughput_w25(REF_UNIT['eval_dir'], refs['unit']['cycles_run'])
    manifest = _load(REF_UNIT['campaign_manifest'])
    checks['w25_record_is_the_s2_unit'] = w25.get('record') == REF_UNIT['eval_dir']
    checks['w25_T_reproduced_bitwise'] = thr['T_cell_throughput_full_cycle_mwh'] == w25['r2']['T_cell_throughput_full_cycle_mwh']
    checks['w25_esso_capture_hash_in_s2_campaign_manifest'] = manifest.get(thr['esso_capture_file']) == thr['esso_capture_sha256']
    checks['w25_single_cohort_288_rows'] = thr['single_cohort_288_rows']
    refs['unit']['throughput'] = thr
    m1 = per_m_row(1.0, refs['x0'], refs['unit'], i_unit)
    checks['m1_value_matches_s2_campaign_results'] = (
        m1['value_eur'] == _load(os.path.join(_P47, 'campaign_s47_recert', 'campaign_results.json'))['points']['n7_4h_e1']['value_eur'])
    failures += [f'reference input check failed: {k}' for k, v in checks.items() if not v]
    return {'references_m1': refs, 'I_unit_eur': i_unit, 'I_source': dict(INVESTMENT_COST_RESULTS),
            'unit_candidate_key': unit_key, 'm1_row': m1,
            'w25_committed': {'T_r2': w25['r2']['T_cell_throughput_full_cycle_mwh'],
                              'captured_r2_eur_per_mwh_cycle': w25['r2']['captured_eur_per_mwh_cycle'],
                              'value_eur_component_sum': w25['r2']['value_eur'],
                              'note': ('W25 divides the T3 component-sum value (Q(0) - Q(x) to <= 0.01 EUR) by T; '
                                       'this ladder divides Q_m(0) - Q_m(unit); the m = 1 figures differ by that sum '
                                       'residual only')},
            'checks': checks}, failures


# ======================================================================================================================
#  memory preflight (A0's measure)
# ======================================================================================================================
_VM_STAT_KEYS = {'pages_free': 'Pages free', 'pages_active': 'Pages active', 'pages_inactive': 'Pages inactive',
                 'pages_speculative': 'Pages speculative', 'pages_wired_down': 'Pages wired down',
                 'pages_purgeable': 'Pages purgeable', 'file_backed_pages': 'File-backed pages',
                 'anonymous_pages': 'Anonymous pages', 'pages_stored_in_compressor': 'Pages stored in compressor',
                 'pages_occupied_by_compressor': 'Pages occupied by compressor'}
_VM_STAT_REQUIRED = ('pages_free', 'pages_inactive', 'pages_wired_down', 'anonymous_pages',
                     'pages_occupied_by_compressor')


def _memory_rule_matches_a0():
    spec = _load(A0_SPEC['path'])
    a0_rule = spec['extra'].get('memory_preflight_rule')
    template_ok = a0_rule == MEMORY_RULE_TEMPLATE.format(concurrency=spec.get('concurrency'),
                                                         per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
    rationale_ok = spec['extra'].get('memory_preflight_rule_rationale') == MEMORY_RULE_RATIONALE
    return {'a0_rule': a0_rule, 'a0_concurrency': spec.get('concurrency'), 'rule': MEMORY_RULE,
            'template_reproduces_a0_rule': template_ok, 'a0_rationale_matches': rationale_ok,
            'match': template_ok and rationale_ok}


def memory_preflight():
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    pages = {name: raw.get(label) for name, label in _VM_STAT_KEYS.items()}
    missing = [name for name in _VM_STAT_REQUIRED if pages[name] is None]
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    out = {'utc': datetime.now(timezone.utc).isoformat(), 'vm_stat_header': first, 'page_size_bytes': page,
           'vm_stat_pages': pages, 'hw_memsize_bytes': total, 'required_bytes': MEMORY_REQUIRED_BYTES,
           'required_gib': MEMORY_REQUIRED_BYTES / GIB, 'per_child_budget_gib': MEMORY_PER_CHILD_BUDGET_BYTES / GIB,
           'concurrency': CONCURRENCY, 'rule': MEMORY_RULE, 'rule_rationale': MEMORY_RULE_RATIONALE,
           'missing_vm_stat_figures': missing}
    if missing:
        out.update({'available_bytes': None, 'available_gib': None, 'pass': False})
        return out
    non_reclaimable = (pages['pages_wired_down'] + pages['anonymous_pages']
                       + pages['pages_occupied_by_compressor']) * page
    avail = total - non_reclaimable
    free_inactive = (pages['pages_free'] + pages['pages_inactive']) * page
    out.update({'wired_bytes': pages['pages_wired_down'] * page, 'anonymous_bytes': pages['anonymous_pages'] * page,
                'compressor_occupied_bytes': pages['pages_occupied_by_compressor'] * page,
                'non_reclaimable_bytes': non_reclaimable, 'non_reclaimable_gib': non_reclaimable / GIB,
                'available_bytes': avail, 'available_gib': avail / GIB,
                'free_plus_inactive_bytes': free_inactive, 'free_plus_inactive_gib': free_inactive / GIB,
                'free_plus_inactive_would_pass_same_threshold': free_inactive >= MEMORY_REQUIRED_BYTES,
                'pass': avail >= MEMORY_REQUIRED_BYTES})
    return out


def _memory_line(memory):
    if memory.get('available_gib') is None:
        return f"vm_stat figures missing {memory['missing_vm_stat_figures']} (rule {memory['rule']})"
    return (f"available (memsize - wired - anonymous - compressor-occupied) = {memory['available_gib']:.2f} GiB "
            f"[wired {memory['wired_bytes'] / GIB:.2f}, anonymous {memory['anonymous_bytes'] / GIB:.2f}, "
            f"compressor-occupied {memory['compressor_occupied_bytes'] / GIB:.2f}]; free+inactive = "
            f"{memory['free_plus_inactive_gib']:.2f} GiB (recorded only); required {memory['required_gib']:.2f} GiB "
            f"({memory['rule']})")


# ======================================================================================================================
#  rule eleven
# ======================================================================================================================
CAMPAIGN_RESULT_FIELDS = (
    'status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'Q', 'bar', 'rule_ten', 'flex_price_multiplier',
    'flex_price_label', 'flex_price_readback', 'flex_cost', 'throughput', 'solve_reconciliation', 'wall_time',
    'peak_rss', 'aa_action_counts', 'per_cycle_trajectory', 'per_m.value_eur', 'per_m.value_minus_I_eur',
    'per_m.resolution_eur', 'per_m.value_per_full_cycle_eur_per_mwh', 'per_m.dso_flexibility_saving_per_dso')


def rule_eleven():
    harness_checks = H.assert_record_capture_paths()
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    hook_src = inspect.getsource(H._config_hook_factory)
    freeze_src = inspect.getsource(H.freeze_campaign_spec)
    import p515_g_g1_g4_admm_gates as G
    cl_src = inspect.getsource(G.write_component_levels_terminal) if hasattr(G, 'write_component_levels_terminal') else ''
    esso_src = inspect.getsource(G._capture_esso_solve) if hasattr(G, '_capture_esso_solve') else ''
    W25 = _w25()
    checks = {
        'status_cycles_Q_bar': all(s in build_src for s in ("'status': status", "'cycles_run': len(rows)",
                                                           "'certified_cost': report.get('gross_operational_cost')",
                                                           "'bar': bar")),
        'rule_ten': "'rule_ten':" in build_src and "'terminal_step_over_threshold':" in build_src,
        'flex_multiplier_entry_and_spec_label': ("cand_entry['flex_price_multiplier'] = flex_m" in freeze_src
                                                 and "spec['flex_price_label'] = FLEX_PRICE_LABEL" in freeze_src),
        'flex_multiplier_applied_in_hook': 'apply_flex_price_multiplier(planning, flex_price_multiplier)' in hook_src,
        'flex_multiplier_passed_by_child': 'flex_price_multiplier=flex_m' in child_src,
        'flex_readback_terminal_in_record': ("holder['flex_price_readback_terminal'] = flex_price_readback_run_models("
                                             in child_src and "'flex_price_readback_terminal':" in child_src),
        'flex_applied_in_record': "'flex_price_applied_in_child': holder.get('flex_price_applied')" in child_src,
        'flex_fields_in_error_records': ('_flex_price_record_fields(entry)' in inspect.getsource(H.main_child)
                                         and '_flex_price_record_fields(entry)' in inspect.getsource(
                                             H._barrier_record_for_missing)),
        'component_levels_flexibility_cost_internal': "'flexibility_cost_internal':" in inspect.getsource(G),
        'component_levels_writer_present': bool(cl_src),
        'esso_capture_per_cycle_node_file': ("f'node{node_id}_{stamp}.jsonl'" in esso_src
                                             and "cycle_label = f'{cycle:03d}'" in inspect.getsource(G)
                                             and G._esso_capture_stamp(f'{87:03d}') == 'cycle087'
                                             and "f'node{NODE}_cycle{cycle_terminal:03d}.jsonl'"
                                             in inspect.getsource(W25._esso_capture)),
        'w25_case_structure_and_esso_capture': (callable(getattr(W25, 'case_structure', None))
                                                and callable(getattr(W25, '_esso_capture', None))
                                                and W25.NODE == UNIT_NODE and W25.DT_H == 1.0),
        'network_failure_event_attempt_flags': ("'recovery_attempted'" in inspect.getsource(G._new_network_event)
                                                and "'tier2_attempted'" in inspect.getsource(G._new_network_event)),
        'solve_profile_in_record': "'solve_profile': report.get('solve_profile')" in build_src,
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'wall_time_peak_rss_aa': ("'wall_time_s': wall" in build_src and "'peak_rss': peak_rss" in build_src
                                  and "'aa_per_cycle': holder.get('aa_sidecar')" in child_src),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s49 flexibility ladder campaign_results): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': harness_checks}


# ======================================================================================================================
#  checks shared by --freeze and --run
# ======================================================================================================================
def _check_case_file_loads_to_declaration():
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded = params.admm.anderson_acceleration
    return {'loaded': loaded, 'equals_declaration': loaded == CASE_FILE_AA}


def _common_checks():
    failures, evidence = [], {}
    case_file = _check_case_file_loads_to_declaration()
    evidence['case_file_aa'] = case_file
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    loaded = H.ess_ageing_canonical_text(H.load_ess_ageing_parameters(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)))
    evidence['ess_file_loads_to_declaration'] = loaded == H.ess_ageing_canonical_text(ESS_AGEING_BASELINE)
    if not evidence['ess_file_loads_to_declaration']:
        failures.append('ESS parameters file does not load to the declaration')
    pins, more = _check_pins()
    evidence['pins'] = pins
    failures += more
    memory_rule = _memory_rule_matches_a0()
    evidence['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f"memory preflight rule differs from A0's measure: {memory_rule}")
    refs, more = reference_inputs()
    evidence['reference_inputs'] = refs
    failures += more
    per_cycle = solves_per_cycle_from_case_file()
    evidence['solves_per_cycle_from_case_file'] = per_cycle
    if per_cycle != 51:
        failures.append(f'solves per cycle from the case file {per_cycle} != 51 (the W10/W32/W33 gates)')
    pts = points()
    labels = [p[0] for p in pts]
    keys = [_eval_key(n, m) for _l, n, m, _i in pts]
    baseline_keys = {REF_X0['eval_key'], REF_UNIT['eval_key']}
    if len(pts) != 6 or len(set(labels)) != 6 or len(set(keys)) != 6 or set(keys) & baseline_keys:
        failures.append('points: expected 6 unique labels / eval keys, all distinct from the m = 1 baseline keys')
    evidence['points'] = [{'label': label, 'instance': inst, 'flex_price_multiplier': m,
                           'nodes': {str(n): list(v) for n, v in nodes.items()}, 'investment_year': YEAR,
                           'candidate_key': _key_of(nodes), 'eval_key': _eval_key(nodes, m)}
                          for label, nodes, m, inst in pts]
    evidence['waves'] = waves(labels)
    spec19 = _load(SPEC_V19['path'])
    evidence['spec_v19_flex_price_ladder'] = spec19['flex_price_ladder']
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, pts


def _validate_spec(spec, pts, refs):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    extra = spec.get('extra') or {}
    cfg = spec['configuration']
    checks = {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'entries_in_order': [e['label'] for e in entries] == [p[0] for p in pts],
        'n_entries_6': len(entries) == 6,
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': ((cfg.get('ess_params_file') or {}).get('path') == ESS_PARAMS_FILE['path']
                                   and (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256']),
        'no_campaign_overrides': cfg.get('overrides') == {},
        'no_ageing_model_variant_anywhere': ('model_variant_label' not in spec
                                             and not any('model_variant' in e for e in entries)),
        'flex_label_at_spec_level': spec.get('flex_price_label') == FLEX_LABEL,
        'cap': spec.get('cap') == CAP,
        'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': cfg.get('arm_label') == 's39_D',
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'post_certification_none': all(e.get('post_certification') is None for e in entries),
        'labels_recorded': extra.get('label') == LABEL and extra.get('flex_label') == FLEX_LABEL,
        'spec_v19_recorded': extra.get('spec_v19') == SPEC_V19,
        'stop_rule_recorded': extra.get('stop_rule') == STOP_RULE,
        'waves_recorded': extra.get('waves') == waves([p[0] for p in pts]),
        'I_frozen': extra.get('I_unit_eur') == refs['I_unit_eur'],
        'm1_row_frozen': (extra.get('m1_row') or {}).get('value_eur') == refs['m1_row']['value_eur'],
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'value_definition_recorded': extra.get('value_definition') == VALUE_DEFINITION,
        'memory_rule_recorded': extra.get('memory_preflight_rule') == MEMORY_RULE,
    }
    for label, nodes, m, _inst in pts:
        entry = by_label.get(label) or {}
        canon = H.canonical_candidate(nodes, investment_year=YEAR)
        checks[f'{label}:canonical_key'] = entry.get('canonical') == canon and entry.get('key') == H.candidate_key(canon)
        checks[f'{label}:no_overrides'] = entry.get('overrides') == {}
        checks[f'{label}:effective_aa_is_declaration'] = entry.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:flex_multiplier'] = entry.get('flex_price_multiplier') == m
        checks[f'{label}:flex_label'] = entry.get('flex_price_label') == FLEX_LABEL
        checks[f'{label}:eval_key_recomputes'] = entry.get('eval_key') == _eval_key(nodes, m)
        checks[f'{label}:eval_key_not_the_baseline_key'] = entry.get('eval_key') != _eval_key(nodes, None)
    return checks


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(started):
    tag = 'S49-FLEX'
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, pts = _common_checks()
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    refs = evidence['reference_inputs']
    extra = {'campaign_script': os.path.basename(__file__),
             'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'label': LABEL, 'flex_label': FLEX_LABEL, 'mode': 'full', 'spec_v19': dict(SPEC_V19),
             'spec_v19_flex_price_ladder': evidence['spec_v19_flex_price_ladder'],
             'predictions_recorded_before_run': evidence['spec_v19_flex_price_ladder'].get('predictions_recorded_before_run'),
             'ess_params_file': dict(ESS_PARAMS_FILE), 'a0_spec': dict(A0_SPEC), 'pins': evidence['pins'],
             'multipliers': list(MULTIPLIERS), 'points': evidence['points'], 'waves': evidence['waves'],
             'stop_rule': STOP_RULE, 'region_note': REGION_NOTE,
             'references_m1': refs['references_m1'], 'm1_row': refs['m1_row'], 'I_unit_eur': refs['I_unit_eur'],
             'I_source': refs['I_source'], 'w25_committed': refs['w25_committed'],
             'reference_checks': refs['checks'],
             'solves_per_cycle_from_case_file': evidence['solves_per_cycle_from_case_file'],
             'post_certification': 'none', 'model_variant': 'none (ageing); flexibility-price multiplier per entry',
             'objective_convention': OBJECTIVE_CONVENTION, 'value_definition': VALUE_DEFINITION,
             'flex_cost_definition': FLEX_COST_DEFINITION,
             'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                'gross_operational_cost[k-1]| (harness W5)'),
             'memory_preflight_rule': MEMORY_RULE, 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
             'memory_preflight_refusing_at': '--run (non-gating at --freeze)', 'memory_at_freeze_non_gating': memory,
             'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID,
        [(label, nodes, {'investment_year': YEAR, 'flex_price_multiplier': m}) for label, nodes, m, _i in pts],
        configuration={'name': (f'{FLEX_LABEL} (m in {list(MULTIPLIERS)}) under {LABEL}: the case file (AA keep_memory '
                                'in data/SRP1/SRP1_params.json) with the ESS ageing parameters declared'),
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('no overrides; no ageing model variant; no post-certification; the flexibility-price '
                                'multiplier is applied per entry in the child configuration hook and read back from '
                                'probe DSO blocks (pre-run) and the run\'s own DSO models (post-run); num_max_iters := '
                                'cap')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=extra)
    checks = _validate_spec(spec, pts, refs)
    guard_failures = PARENT_GUARD.verify(0)
    w25_guard_failures = _w25().GUARD.verify(0)
    _log(f'[{tag}] {FLEX_LABEL} under {LABEL}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for entry in spec['candidates']:
        _log(f"[{tag}]   {entry['label']}: m={entry['flex_price_multiplier']} key={entry['key'][:16]} "
             f"eval_key={entry['eval_key'][:16]} eval_dir={entry['eval_dir']}")
    m1 = refs['m1_row']
    _log(f"[{tag}] m = 1 references: Q(0)={m1['Q_x0']} (bar {m1['bar_x0']}), Q(unit)={m1['Q_unit']} (bar "
         f"{m1['bar_unit']}), value={m1['value_eur']}, I={m1['I_eur']}, value-I={m1['value_minus_I_eur']}, "
         f"resolution={m1['resolution_eur']} -> {m1['reading']}; T={m1['T_cell_throughput_full_cycle_mwh']} MWh "
         f"(W25 committed {refs['w25_committed']['T_r2']}), value/cycle={m1['value_per_full_cycle_eur_per_mwh']} "
         f"(W25 committed {refs['w25_committed']['captured_r2_eur_per_mwh_cycle']})")
    _log(f"[{tag}] m = 1 DSO flexibility saving per DSO: {m1['dso_flexibility_saving_per_dso']}")
    _log(f"[{tag}] reference checks: {refs['checks']}")
    _log(f"[{tag}] waves: {evidence['waves']}; stop rule: {STOP_RULE}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] pins: {evidence['pins']}")
    _log(f"[{tag}] memory rule vs A0: {evidence['memory_rule_vs_a0_spec']}")
    _log(f"[{tag}] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}; W25 module guard '
         f'{_w25().GUARD.counts} verify0_failures={w25_guard_failures}; wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures and not w25_guard_failures
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _point_result(label, rec, spec_point, per_cycle):
    rec = rec or {}
    traj, rows = _per_cycle_rows(rec)
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else None
    canonical = rec.get('candidate_canonical')
    m = spec_point['flex_price_multiplier']
    applied = rec.get('flex_price_applied_in_child') or {}
    pre = applied.get('readback_pre_run') or {}
    post = rec.get('flex_price_readback_terminal') or {}
    eval_dir = rec.get('eval_dir')
    has_levels = bool(eval_dir) and os.path.isfile(os.path.join(REPO, eval_dir, 'component_levels_terminal.json'))
    throughput = None
    if spec_point['instance'] == 'unit' and certified and eval_dir:
        try:
            throughput = unit_throughput_w25(eval_dir, rec.get('cycles_run'))
        except Exception as error:  # noqa: BLE001 -- recorded, never silently dropped
            throughput = {'error': f'{type(error).__name__}: {error}'}
    return {
        'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'label': label, 'instance': spec_point['instance'],
        'flex_price_multiplier': m, 'flex_price_multiplier_in_record': rec.get('flex_price_multiplier'),
        'flex_price_label_in_record': rec.get('flex_price_label'),
        'candidate_key': rec.get('candidate_key'), 'candidate_canonical': canonical,
        'region': _region_key(canonical) if canonical else None,
        'eval_key': rec.get('eval_key'), 'eval_dir': eval_dir,
        'status': rec.get('status'), 'barrier': rec.get('barrier'), 'barrier_cause': rec.get('barrier_cause'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'Q': q, 'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'bar': (rec.get('bar') or {}).get('value'),
        'bar_net_recourse_step_reported': (rec.get('bar_net_recourse_step_reported') or {}).get('value'),
        'rule_ten': _rule_ten(rec, rows),
        'flex_price_readback': {
            'pre_run_probes': {k: pre.get(k) for k in ('n_blocks', 'all_match', 'n_flex_coefficients',
                                                       'n_bitwise_exact', 'max_rel_dev', 'rel_tol')},
            'applied_checks': applied.get('checks'),
            'post_run_run_models': {k: post.get(k) for k in ('n_blocks', 'all_match', 'n_coefficients',
                                                             'n_bitwise_equal_applied_closed_form',
                                                             'n_bitwise_equal_m_times_original',
                                                             'max_rel_dev_vs_m_times_original',
                                                             'tso_arrays_still_the_planning_arrays')},
            'all_match': (pre.get('all_match') is True and post.get('all_match') is True
                          and bool(applied.get('checks')) and all(applied['checks'].values()))},
        'flex_cost': flex_cost_per_dso(eval_dir, m) if has_levels else None,
        'throughput': throughput,
        'solve_reconciliation': solve_reconciliation(rec, per_cycle),
        'first_pass_cycle_per_channel': rec.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': rec.get('terminal_ratios_per_channel'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'network_failures_summary': rec.get('network_failures_summary'),
        'ess_ageing_readback_all_match': {
            'pre_run': ((rec.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run') or {}).get('all_match'),
            'post_run': (rec.get('ess_ageing_readback_terminal') or {}).get('all_match')},
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': {'record': rec.get('peak_rss'),
                     'parent_wait4_ru_maxrss_bytes': (rec.get('parent_view') or {}).get('wait4_ru_maxrss')},
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'anderson_acceleration_effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'case_file_sha256_in_child': rec.get('case_file_sha256_in_child'),
        'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
        'per_cycle_trajectory': traj,
    }


def stop_rule_state(records):
    non_certified = [r for r in records if (r or {}).get('status') != 'certified']
    regions = Counter(_region_key(r['candidate_canonical']) for r in non_certified
                      if (r or {}).get('candidate_canonical'))
    offending = sorted(region for region, count in regions.items() if count >= 2)
    reasons = []
    if offending:
        reasons.append(f'(a) 2 or more non-certified points in one region: {offending} ({dict(regions)})')
    if len(non_certified) >= 3:
        reasons.append(f'(b) 3 or more non-certified points overall: '
                       f'{[r.get("candidate_label") for r in non_certified]}')
    return {'triggered': bool(reasons), 'reasons': reasons,
            'non_certified_labels': [r.get('candidate_label') for r in non_certified],
            'non_certified_by_region': dict(regions)}


def run(started, spec_sha256):
    tag = 'S49-FLEX'
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence, pts = _common_checks()
    failures += more
    refs = evidence['reference_inputs']
    checks = _validate_spec(spec, pts, refs)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['configuration']['ess_params_file']['sha256'] != H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)):
        failures.append('ESS params file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    memory = memory_preflight()
    _log(f"[{tag}] memory preflight: {_memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}; "
         f"vm_stat pages {memory['vm_stat_pages']}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {_memory_line(memory)}')
    if failures:
        for failure in failures:
            _log(f'[{tag} PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    per_cycle = evidence['solves_per_cycle_from_case_file']
    _log(f'[{tag}] {FLEX_LABEL} under {LABEL}')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    labels = [p[0] for p in pts]
    wave_list = waves(labels)
    _log(f'[{tag}] {len(labels)} points in {len(wave_list)} waves {wave_list}; stop rule: {STOP_RULE}')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    records, wave_info, not_launched = [], [], []
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        for number, wave in enumerate(wave_list, start=1):
            state = stop_rule_state(records)
            if state['triggered']:
                not_launched += wave
                _log(f'[{tag}] STOP RULE: not launching wave {number} {wave}; {state["reasons"]}')
                wave_info.append({'wave': number, 'labels': wave, 'launched': False,
                                  'reason': f'stop rule: {state["reasons"]}'})
                continue
            _log(f'[{tag}] launching wave {number}/{len(wave_list)}: {wave} (concurrency {ctx.concurrency})')
            H.evaluate.last_batch_info = {}
            records += H.evaluate(wave, ctx)
            wave_info.append({'wave': number, 'labels': wave, 'launched': True,
                              'stop_rule_after_wave': stop_rule_state(records),
                              **getattr(H.evaluate, 'last_batch_info', {})})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    stop_state = stop_rule_state(records)
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    spec_points = {p['label']: p for p in spec['extra']['points']}
    point_results = {}
    for label in labels:
        if label in not_launched:
            point_results[label] = {'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'label': label,
                                    'instance': spec_points[label]['instance'],
                                    'flex_price_multiplier': spec_points[label]['flex_price_multiplier'],
                                    'status': 'not_launched_stop_rule'}
        else:
            point_results[label] = _point_result(label, by_label.get(label), spec_points[label], per_cycle)
    i_unit = spec['extra']['I_unit_eur']
    per_m = {'1': dict(spec['extra']['m1_row'], source='pinned m = 1 references (s48_x0_capture x0, S2 unit)')}
    for m in MULTIPLIERS:
        x0 = point_results[f'x0_m{_m_tag(m)}']
        unit = point_results[f'n7_4h_e1_m{_m_tag(m)}']
        per_m[f'{m:g}'] = dict(per_m_row(m, x0, unit, i_unit), source='this campaign')
    launched = [label for label in labels if label not in not_launched]
    non_certified = [label for label in launched if point_results[label]['status'] != 'certified']
    harness_errors = [label for label in launched
                      if point_results[label]['status'] not in ('certified', 'not_certified')
                      or point_results[label]['exit_code'] != 0]
    readback_mismatch = [label for label in launched if point_results[label]['status'] in ('certified', 'not_certified')
                         and not (point_results[label]['flex_price_readback']['all_match']
                                  and point_results[label]['ess_ageing_readback_all_match']['pre_run'] is True
                                  and point_results[label]['ess_ageing_readback_all_match']['post_run'] is True)]
    solve_mismatch = [label for label in launched if point_results[label]['status'] in ('certified', 'not_certified')
                      and not point_results[label]['solve_reconciliation']['holds']]
    guard_failures = PARENT_GUARD.verify(0)
    w25_guard_failures = _w25().GUARD.verify(0)
    table = [{'m': row['m'], 'source': row['source'], 'Q_x0': row['Q_x0'], 'Q_unit': row['Q_unit'],
              'value': row['value_eur'], 'I': row['I_eur'], 'value_minus_I': row['value_minus_I_eur'],
              'resolution': row['resolution_eur'], 'reading': row['reading'],
              'T_mwh': row['T_cell_throughput_full_cycle_mwh'],
              'value_per_full_cycle': row['value_per_full_cycle_eur_per_mwh'],
              'resolution_per_full_cycle': row['resolution_per_full_cycle_eur_per_mwh'],
              'rule_ten_gross_x0': row['rule_ten_gross_ratio_x0'], 'rule_ten_gross_unit': row['rule_ten_gross_ratio_unit'],
              'dso_flex_saving_total': row['dso_flexibility_saving_total_weighted_r2_eur']} for row in per_m.values()]
    results = {
        'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL,
        'STOP_FOR_REVIEW': stop_state['triggered'], 'stop_rule_state': stop_state, 'stop_rule': STOP_RULE,
        'region_note': REGION_NOTE,
        'non_certified_points': non_certified, 'not_launched_points': not_launched, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch, 'solve_reconciliation_mismatch_points': solve_mismatch,
        'stage': 'P5.15 Addendum 34 -- flexibility-price ladder (W33 item 4)', 'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'value_definition': VALUE_DEFINITION,
        'flex_cost_definition': FLEX_COST_DEFINITION,
        'predictions_recorded_before_run': spec['extra'].get('predictions_recorded_before_run'),
        'table_objective_convention': ('Q gross (settlement excluded); value = Q_m(0) - Q_m(unit); I = I(unit); '
                                       'value per full cycle = value / T (W25 T4 r2)'),
        'table': table, 'per_m': per_m, 'points': point_results,
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pins', 'memory_rule_vs_a0_spec',
                                                      'solves_per_cycle_from_case_file')},
        'reference_checks_at_run': refs['checks'],
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'wave_info': wave_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures,
                                       'w25_module_guard_counts': dict(_w25().GUARD.counts),
                                       'w25_module_guard_verify_0_failures': w25_guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(CAMPAIGN_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f'[{tag}] {FLEX_LABEL} under {LABEL} -- objective convention: {OBJECTIVE_CONVENTION}')
    for row in table:
        _log(f'[{tag}] {row}')
    for label in labels:
        p = point_results[label]
        _log(f"[{tag}] {label}: status={p['status']} cycles={p.get('cycles_run')} Q={p.get('Q')} bar={p.get('bar')} "
             f"rule_ten_gross={(p.get('rule_ten') or {}).get('terminal_gross_step_over_threshold')} "
             f"solves={(p.get('solve_reconciliation') or {}).get('observed')}/"
             f"{(p.get('solve_reconciliation') or {}).get('expected')}")
    if non_certified:
        _log(f'[{tag}] non-certified points (reported with cause): '
             f"{[(l, point_results[l].get('barrier_cause')) for l in non_certified]}")
    if readback_mismatch:
        _log(f'[{tag}] READ-BACK MISMATCH: {readback_mismatch}')
    if solve_mismatch:
        _log(f'[{tag}] SOLVE RECONCILIATION MISMATCH (reported): {solve_mismatch}')
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}; W25 module guard '
         f'verify0_failures={w25_guard_failures}')
    if stop_state['triggered']:
        _banner([f'STOP_FOR_REVIEW: the stop rule fired: {stop_state["reasons"]}', f'not launched: {not_launched}'])
    if guard_failures or w25_guard_failures or harness_errors or readback_mismatch:
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    if stop_state['triggered']:
        sys.exit(3)
    if non_certified:
        _log(f'[{tag}] harness clean; {len(non_certified)} non-certified point(s)')
        sys.exit(2)
    _log(f'[{tag}] OK: all {len(labels)} points certified')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='evaluate the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    os.chdir(REPO)
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(started, args.spec_sha256)


if __name__ == '__main__':
    main()
