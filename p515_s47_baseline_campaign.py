"""
P5.15 Addendum 30 (task W21, item 3) -- the ageing BASELINE campaigns S2 and S3 of frozen spec
v17, through the campaign harness (`p515_s44_campaign_harness.py`) with the ESS ageing
parameters DECLARED (`configuration.ess_ageing_baseline`, W21 item 2), so every evaluation's
key denotes the baseline it ran under.

  BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)
  -- data/SRP1/SharedESS/SRP1_ESS_Params.json as edited by W21 (2466401d): calibration C2
  (10000 cycles, DoD 0.80, EOL retention 0.80 -> k = 35,851.36), calendar retention 0.985 per
  year, minimum SoH 0.70. Every C3 result (A0, A1a, A1b, A2, A3, the ageing batch) is the
  SENSITIVITY set and never shares a table with these (spec v17 baseline.sensitivity_set);
  the only pre-W21 quantity used here is Q(0), which is ageing-independent (see below).

Launcher pattern of `p515_s45_a1_campaign.py` / `p515_s46_ageing_campaign.py`: preconditions,
locks, clean git, one spec per campaign root, memory preflight (A0's measure at concurrency
5), cap 500, 10 consecutive all-pass cycles, case-file AA declaration, NO overrides, NO model
variant, NO post-certification. A new script rather than an extension of
`p515_s45_a1_campaign.py`: that launcher fixes concurrency 7 in its spec validation and memory
rule, reads its stages' priors from C3 results (which must not enter a baseline table), and
its frozen a1a spec pins its sha256; editing it would make its own committed specs
unreproducible by their script. Nothing is imported from it (its module-level parent guard
would stack on this one); the a1a points are rebuilt from spec v15 and cross-checked against
the committed A1a campaign spec entry by entry.

STAGES (`--stage`), one campaign root and one frozen spec each:
  s2   `s47_recert`       (spec v17 S2) 2 evaluations, one batch: C* (0.96875 MVA / 3.875 MWh
                          at nodes 5, 7, 9; 2025) and n7_4h_e1 (0.25 MVA / 1.0 MWh at node 7;
                          2025). Root data/SRP1/Results/P515S47/campaign_s47_recert/.
  s3   `s47_a1a_baseline` (spec v17 S3) the 30 A1a points exactly as spec v15
                          `A1.ladders_2025`, in waves of 5 with a1a's stop rule. Root
                          data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/.
  n7_4h_e1 is in BOTH stages under the SAME evaluation key (same candidate, same
  configuration); the S3 spec records this (`duplicate_of_s2`), see the worker report.

Q(0). x = 0 is not re-run: Q(0) = 653,859,461.2279255 (bar 9,629.978) from the pinned A0 x0
record. It is ageing-independent: W21's case-file gate (2466401d,
case_file_baseline/after) found all 48 network blocks identical at x = 0 and the NL file of
every x = 0 ESSO block (the problem IPOPT receives) byte-identical before / after the edit;
the gate's literal "every block digest identical" check FAILED because the ESSO structural
digests differ, and x0_diff_analysis shows those differences are only deactivated degradation
rows and the salvage Expression (not in the ESSO objective; its variables pinned to 0 at
x = 0). Both are pinned below and re-checked at --freeze / --run.

RECORDED PER POINT (campaign_results.json `points`; objective convention on every table:
Q = certified GROSS operational cost, settlement excluded; F = I + Q; salvage reported,
excluded): the A1 record set -- status (non-certified with cause), cycles, certification
cycle, Q, the bar, the rule-ten terminal step ratios, terminal salvage, net operational
recourse, I(x) (pinned W2 table, unchanged by ageing), F, budget slack at B = 1e6 (reported,
not applied), investment year, wall time, peak RSS, AA action counts, per-cycle trajectory
path + sha256 -- plus value = Q(0) - Q and value - I; whether any SoH-floor row is active at
the certified point and in which block (the production floor sidecar's terminal line in the
evaluation's boyd_terminal.json: active = |SoH - soh_min| <= 1e-6, with the floor dual), the
SoH trajectory per block (end-of-block SoH, SoH used for available energy) and EFC/day per
block (the harness's ageing_trajectory_terminal, captured for declared specs), and the ESS
ageing read-back (k, phi, floor bound; probe models pre-run and clones post-run).

Two modes per stage, both attached, both streams captured, never detached:
  --stage <s> --freeze                    ZERO SOLVES: preconditions, pins, points, I(x),
                                          Q(0), rule eleven, `freeze_campaign_spec` +
                                          validation. Prints the --run command.
  --stage <s> --run --spec-sha256 <sha>   loads THAT spec from the stage's root (which must
                                          hold only it), re-checks everything plus the
                                          harness / case-file / ESS-params / script sha256,
                                          the memory preflight (refusing), takes the campaign
                                          lock, evaluates, writes campaign_results.json and
                                          campaign_manifest_sha256.json.
The parent never solves: SolveProfileGuard(permitted=()) installed before any model import,
verified at exactly 0 in both modes. Exit codes (--run): 0 every point certified; 3
STOP_FOR_REVIEW (s3 stop rule fired, harness clean); 2 s2 with a non-certified point (harness
clean); 1 harness / guard / precondition failure.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_baseline_campaign.py \\
      --stage s2 --freeze > data/SRP1/Results/P515S47/campaign_s47_recert_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_baseline_campaign.py \\
      --stage s2 --run --spec-sha256 <sha> > data/SRP1/Results/P515S47/campaign_s47_recert_launch.log 2>&1
  and the same with `--stage s3` and `campaign_s47_a1a_baseline_{freeze_launch,launch}.log`.
  One stage at a time, attached, never detached.
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S47 baseline campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
BUDGET_EUR = 1e6
ACTIVE_NODES = (5, 7, 9)
YEAR = 2025
FLOOR_ACTIVE_TOL = 1e-6  # the production floor sidecar's definition (p515_g_g1_g4_admm_gates s38 floor capture)

# The ESS ageing declaration: EXACTLY what the edited file loads to (types as the loader keeps them).
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
SPEC_V17 = {'path': os.path.join(_P47, 'frozen_s47_baseline_spec_v17_ff0056b8.json'),
            'sha256': 'ff0056b850957d47cc3aa2595eab13497e100cc7bcda84514c5ca58e25d1e109'}
SPEC_V15 = {'path': os.path.join(_P45, 'frozen_s45_phaseA_spec_v15_5feefd7b.json'),
            'sha256': '5feefd7b642fc3d480156ad5e52ed6e1cf9d6698cfd40dbb33389bab6e6229fe'}
ESS_PARAMS_FILE = {'path': H.ESS_PARAMS_FILE_REL,
                   'sha256': '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706',
                   'commit': '2466401d', 'note': 'the ageing BASELINE edit of W21 (Addendum 30)'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
A0_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_spec_s45_a0_c7_9d08ad2f.json'),
           'sha256': '9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34'}
A0_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json'),
              'sha256': '423678b98c0ca9a26dbc76a1b46a5e36a51c8ccccc0459253020bfe1061653b4', 'label': 'x0'}
A1A_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_spec_s45_a1a_c71f52ee.json'),
            'sha256': 'c71f52ee535aff607927f83f52783993198b90a8ff44f5426f92e66dc160b851',
            'note': 'the committed C3 A1a spec: used ONLY to cross-check labels / canonical forms / candidate keys'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'commit': '9e623dd304837f2632964f47130d0860c1f016ca',
                           'field': 'candidates.<label>.I_new_eur / slack_new_eur, matched by candidate_key'}
IDENTITY_CHECKS = {'path': os.path.join(_P47, 'identity_checks', 'identity_checks.json'), 'commit': '65525006',
                   'expect': 'all_ok true'}
CASE_FILE_GATE = {'path': os.path.join(_P47, 'case_file_baseline', 'after', 'case_file_baseline_after.json'),
                  'commit': '2466401d',
                  'expect': ('all_ok false with EXACTLY one failing check, b_x0_every_block_digest_identical_before_after '
                             '(the literal digest criterion; see X0_ANALYSIS)')}
X0_ANALYSIS = {'path': os.path.join(_P47, 'case_file_baseline', 'x0_diff_analysis', 'x0_diff_analysis.json'),
               'commit': '2466401d', 'expect': 'all_ok true'}
GATE_EXPECTED_FAILING = ['b_x0_every_block_digest_identical_before_after']

Q0_STATEMENT = (
    'Q(0) is taken from the pinned A0 x0 record (not re-run): x = 0 has no storage. Evidence that it is '
    'ageing-independent: W21 case-file gate (2466401d) at x = 0 -- all 48 TSO/DSO blocks digest-identical before/after '
    'the edit and the NL file of every ESSO block (what IPOPT receives) byte-identical; the literal criterion "every '
    'block digest identical" FAILED because the 3 ESSO structural digests differ, and x0_diff_analysis shows the '
    'differences are only the 18 DEACTIVATED degradation rows per block and the salvage Expression, which is not in '
    'the ESSO objective and whose variables are pinned to 0 by active rows at x = 0.')
OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); F(x) = I(x) + Q(x); '
                        'value = Q(0) - Q(x) with Q(0) from the pinned A0 x0 record; terminal_salvage_value and '
                        'net_operational_recourse = gross - salvage reported, excluded (Addendum 27 item 3).')
BUDGET_CONVENTION = (f'budget_slack_eur = B - I(x) at B = {BUDGET_EUR:g} EUR, REPORTED only (not applied in S2/S3)')
SENSITIVITY_STATEMENT = ('spec v17 baseline.sensitivity_set: all C3 results (A0, A1a, A1b, A2, A3, the ageing batch) '
                         'are the sensitivity set and never share a table with the baseline; no C3 Q(x) appears here '
                         '(Q(0) is ageing-independent, see q0_statement)')
STOP_RULE = ('as a1a (Planner task W15, spec v15 execution.barrier): after every wave, STOP (launch no further wave) if '
             '(a) 2 or more non-certified points fall in ONE region (node+duration ladder) or (b) 3 or more '
             'non-certified points occur overall. A single isolated non-certified point does NOT stop the campaign. '
             'Points already running always finish; unlaunched points are recorded as not_launched_stop_rule and '
             'campaign_results.json carries STOP_FOR_REVIEW true (exit 3).')

GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 11 * GIB // 4  # 2.75 GiB
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE_TEMPLATE = ('hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {concurrency} x '
                        '{per_child:g} GiB')
MEMORY_RULE = MEMORY_RULE_TEMPLATE.format(concurrency=CONCURRENCY, per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside)')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 30',
    'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json baseline, steps S2, S3',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json A1.ladders_2025 (the S3 points)',
    'Planner task W21 items 2-3 (declared ESS ageing identity; launchers, concurrency 5, cap 500, 10 cycles, '
    'case-file AA declaration, no overrides / model variant / post-certification)',
    '65525006 (harness identity + checks), 2466401d (case-file baseline edit + gate + x0 analysis)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', 'p515_s47_identity_checks.py', 'p515_s47_case_file_baseline_check.py',
                     'p515_s47_case_file_x0_diff_analysis.py')


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


def _key_of(nodes, year):
    return H.candidate_key(H.canonical_candidate(nodes, investment_year=year))


def _eval_key(nodes, year):
    return H.evaluation_key(_key_of(nodes, year), {}, case_file_aa=CASE_FILE_AA,
                            ess_ageing_baseline=ESS_AGEING_BASELINE)


def _region_key(canonical):
    """a1a's stop-rule region: the node+duration ladder (x0 for the empty candidate)."""
    nz = {int(n): v for n, v in canonical['nodes'].items() if float(v[0]) or float(v[1])}
    if not nz:
        return f"x0_y{canonical['investment_year']}"
    return '+'.join(f'n{n}_{float(nz[n][1]) / float(nz[n][0]):g}h' for n in sorted(nz)) + \
        f"_y{canonical['investment_year']}"


# ======================================================================================================================
#  the points
# ======================================================================================================================
def _s2_points():
    return (('c_star', _nodes_full({n: (0.96875, 3.875) for n in ACTIVE_NODES}), YEAR),
            ('n7_4h_e1', _nodes_full({7: (0.25, 1.0)}), YEAR))


def _s3_points():
    """spec v15 A1.ladders_2025, in its own order (read from the pinned spec, not re-derived)."""
    ladders = _load(SPEC_V15['path'])['A1']['ladders_2025']
    return tuple((p['label'], _nodes_full({int(n): v for n, v in p['nodes'].items()}), int(p['investment_year']))
                 for p in ladders['points'])


def _s3_points_vs_a1a():
    """The S3 points against spec v15's stated n and the committed A1a spec's entries (labels, canonical forms,
    candidate keys, in order) -- the A1a spec is used for this cross-check only."""
    ladders = _load(SPEC_V15['path'])['A1']['ladders_2025']
    a1a = _load(A1A_SPEC['path'])['candidates']
    pts = _s3_points()
    mine = [(label, H.canonical_candidate(nodes, investment_year=year)) for label, nodes, year in pts]
    theirs = [(e['label'], e['canonical']) for e in a1a]
    return {'n_points': len(pts), 'spec_v15_n': ladders.get('n'), 'n_matches_spec_v15': ladders.get('n') == len(pts),
            'equal_to_committed_a1a_entries_in_order': mine == theirs,
            'candidate_keys_equal_to_committed_a1a': [H.candidate_key(c) for _l, c in mine] == [e['key'] for e in a1a],
            'all_year_2025': all(year == YEAR for _l, _n, year in pts)}


STAGES = {
    's2': {'campaign_id': 's47_recert', 'spec_v17_step': 'S2', 'expected_n': 2, 'points': _s2_points,
           'description': ('P5.15 Addendum 30 S2 -- re-certification under the ageing BASELINE: C* and the smallest '
                           'node-7 4 h unit, one batch of 2, concurrency 5')},
    's3': {'campaign_id': 's47_a1a_baseline', 'spec_v17_step': 'S3', 'expected_n': 30, 'points': _s3_points,
           'description': ('P5.15 Addendum 30 S3 -- the 30 A1a points (spec v15 A1.ladders_2025) under the ageing '
                           'BASELINE, waves of 5, a1a stop rule')},
}


def campaign_root(stage):
    return os.path.join(REPO, _P47, f'campaign_{STAGES[stage]["campaign_id"]}')


# ======================================================================================================================
#  pinned inputs
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v17', SPEC_V17), ('spec_v15', SPEC_V15), ('ess_params_file', ESS_PARAMS_FILE),
                      ('a1a_spec', A1A_SPEC),
                      ('cost_file', COST_FILE), ('a0_spec', A0_SPEC), ('a0_results', A0_RESULTS),
                      ('investment_cost_results', INVESTMENT_COST_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == pin['sha256'] and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for name, pin in (('identity_checks', IDENTITY_CHECKS), ('case_file_gate', CASE_FILE_GATE),
                      ('x0_diff_analysis', X0_ANALYSIS)):
        path = os.path.join(REPO, pin['path'])
        present = os.path.isfile(path)
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip()) if present else True
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        entry = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': in_head, 'git_tracked': tracked,
                 'git_clean': not dirty, 'sha256': H.sha256_file(path) if present else None}
        ok = present and tracked and not dirty and in_head
        if present:
            payload = _load(pin['path'])
            failing = sorted(k for k, v in (payload.get('checks') or {}).items() if not v)
            entry.update({'all_ok': payload.get('all_ok'), 'failing_checks': failing, 'expect': pin['expect']})
            if name == 'case_file_gate':
                ok = ok and payload.get('all_ok') is False and failing == GATE_EXPECTED_FAILING
            else:
                ok = ok and payload.get('all_ok') is True
        out[name] = entry
        if not ok:
            failures.append(f'{name}: {entry}')
    return out, failures


def baseline_inputs():
    failures = []
    a0 = _load(A0_RESULTS['path'])['points'][A0_RESULTS['label']]
    x0_key = _key_of(_nodes_full({}), YEAR)
    spec17 = _load(SPEC_V17['path'])
    v17_ageing = spec17['baseline']['ageing']
    checks = {
        'a0_x0_certified': a0.get('status') == 'certified',
        'a0_x0_candidate_key': a0.get('candidate_key') == x0_key,
        'declaration_matches_spec_v17_baseline': (
            v17_ageing['cycles_n'] == ESS_AGEING_BASELINE['calibration']['cycles_n']
            and v17_ageing['reference_dod_d'] == ESS_AGEING_BASELINE['calibration']['reference_dod_d']
            and v17_ageing['eol_retention_r'] == ESS_AGEING_BASELINE['calibration']['eol_retention_r']
            and v17_ageing['calendar_retention_per_year'] == ESS_AGEING_BASELINE['calendar_retention_per_year']
            and v17_ageing['minimum_soh'] == ESS_AGEING_BASELINE['minimum_soh']),
        'file_loads_to_declaration': (H.ess_ageing_canonical_text(H.load_ess_ageing_parameters(
            os.path.join(REPO, H.ESS_PARAMS_FILE_REL))) == H.ess_ageing_canonical_text(ESS_AGEING_BASELINE)),
    }
    failures += [f'baseline input check failed: {k}' for k, v in checks.items() if not v]
    return {'Q0_eur': a0.get('certified_cost_gross_settlement_excluded'), 'Q0_bar_eur': (a0.get('bar') or {}).get('value'),
            'Q0_source': dict(A0_RESULTS), 'Q0_eval_dir': a0.get('eval_dir'),
            'Q0_certification_cycle': a0.get('certification_cycle'), 'x0_candidate_key': x0_key,
            'q0_statement': Q0_STATEMENT, 'ess_ageing_baseline': ESS_AGEING_BASELINE,
            'k_closed_form': H.ess_ageing_baseline_expected(ESS_AGEING_BASELINE)['k'],
            'spec_v17_predictions_recorded_before_run': spec17.get('predictions_recorded_before_run'),
            'checks': checks}, failures


def investment_costs(points):
    """I(x) and B - I from the pinned W2 table (unchanged by ageing), by candidate key."""
    table = _load(INVESTMENT_COST_RESULTS['path'])['candidates']
    per_point, problems = {}, []
    for label, nodes, year in points:
        key = _key_of(nodes, year)
        hits = {name: c for name, c in table.items() if c.get('candidate_key') == key}
        values = sorted({c.get('I_new_eur') for c in hits.values()}, key=repr)
        if len(values) != 1 or values[0] is None:
            problems.append(f'{label}: key {key[:16]} -> I(x) {values} in the W2 table ({sorted(hits)})')
            per_point[label] = {'candidate_key': key, 'found': False}
            continue
        entry = hits[sorted(hits)[0]]
        i_x = values[0]
        slack = BUDGET_EUR - i_x
        if entry.get('slack_new_eur') is not None and abs(slack - entry['slack_new_eur']) > 1e-6:
            problems.append(f'{label}: B - I {slack} != table slack {entry["slack_new_eur"]}')
        per_point[label] = {'candidate_key': key, 'found': True, 'investment_year': year,
                            'nodes': {str(n): list(v) for n, v in nodes.items()},
                            'I_x_eur': i_x, 'I_power_eur': entry.get('I_new_power_eur'),
                            'I_energy_eur': entry.get('I_new_energy_eur'), 'budget_slack_eur': slack,
                            'budget_feasible_corrected_file': entry.get('budget_feasible_new'),
                            'matched_entries': sorted(hits), 'I_x_source': INVESTMENT_COST_RESULTS['path'],
                            'note': BUDGET_CONVENTION}
    return per_point, problems


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
    'status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'certified_cost_gross_settlement_excluded', 'bar',
    'rule_ten', 'terminal_salvage_value', 'net_operational_recourse', 'I_x_eur', 'F_eur', 'budget_slack_eur',
    'value_eur', 'value_minus_I_eur', 'investment_year', 'soh_floor', 'soh_trajectory_per_block',
    'efc_per_day_per_block', 'ess_ageing_readback', 'wall_time', 'peak_rss', 'aa_action_counts', 'per_cycle_trajectory')


def rule_eleven():
    harness_checks = H.assert_record_capture_paths()
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    hook_src = inspect.getsource(H._config_hook_factory)
    traj_src = inspect.getsource(H.ageing_trajectory_terminal)
    main_child_src = inspect.getsource(H.main_child)
    barrier_src = inspect.getsource(H._barrier_record_for_missing)
    import p515_g_g1_g4_admm_gates as G
    floor_src = inspect.getsource(G.s35ref_capture_hooks)
    s38_src = inspect.getsource(G.s38_pf_capture_hooks)
    import shared_resources_planning as srp
    rc_src = inspect.getsource(srp._get_operational_recourse_components)
    checks = {
        'status': "'status': status" in build_src,
        'barrier_cause': "'barrier_cause': cause" in build_src,
        'cycles_run': "'cycles_run': len(rows)" in build_src,
        'certification_cycle': "'certification_cycle':" in build_src,
        'Q_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'bar_gross_step': "'bar': bar" in build_src and 'gross_step_abs' in inspect.getsource(H._max_step_last_n),
        'rule_ten': "'rule_ten':" in build_src and "'terminal_step_over_threshold':" in build_src,
        'salvage_and_net_recourse': ("'recourse_components': rc" in build_src and "'terminal_salvage_value':" in rc_src
                                     and "'terminal_net_operational_recourse':" in build_src),
        'investment_year_in_canonical': "'candidate_canonical': entry['canonical']" in build_src,
        'soh_floor_terminal_in_boyd_terminal': ("floor_terminal=boyd_terminal.get('soh_floor_multiplier_and_efc_per_"
                                                "cohort_year_terminal')" in child_src
                                                and "'active': active" in floor_src and "'dual': dual_val" in floor_src
                                                and 'abs(soh_val - soh_min) <= 1e-6' in floor_src
                                                and 'with s35ref_capture_hooks(' in s38_src
                                                and 'G.s38_pf_capture_hooks(' in child_src),
        'soh_floor_active_list_in_record': "'soh_floor_rows_active_at_terminal':" in inspect.getsource(
            H._storage_per_node),
        'ageing_trajectory_for_declared_specs': ("if 'ageing_trajectory_terminal' not in holder:" in child_src
                                                 and all(f"'{f}'" in traj_src for f in (
                                                     'efc_per_day', 'soh_end', 'soh_used_for_available_energy',
                                                     'block_year', 'salvage_value'))),
        'ess_ageing_readback_pre_run': ("verified['readback_pre_run'] = ess_ageing_readback_models(" in hook_src
                                        and "'ess_ageing_verified_pre_run':" in child_src),
        'ess_ageing_readback_terminal': ("holder['ess_ageing_readback_terminal'] = ess_ageing_readback_models(" in child_src
                                         and "'ess_ageing_readback_terminal':" in child_src),
        'ess_ageing_declaration_in_error_records': ("'ess_ageing_baseline': spec['configuration']['ess_ageing_baseline']"
                                                    in main_child_src and "'ess_ageing_baseline':" in barrier_src),
        'child_refuses_unpinned_ess_file': "pin.get('sha256') != ess_params_sha256_in_child" in child_src,
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'per_cycle_fields': all(f in H.PER_CYCLE_RECORD_FIELDS for f in (
            'cycle', 'gross_operational_cost', 'terminal_salvage_value', 'objective_change_abs', 'objective_tolerance',
            'consecutive_converged_cycles', 'boyd_all_pass', 'local_solves_ok')),
        'wall_time': "'wall_time_s': wall" in build_src,
        'peak_rss': "'peak_rss': peak_rss" in build_src,
        'aa_action_counts': ("'aa_per_cycle': holder.get('aa_sidecar')" in child_src
                             and "'action_counts'" in inspect.getsource(H.aa_sidecar_summary)),
        'I_x_Q0_frozen_in_spec': True,  # asserted value by value by _validate_spec on the spec itself
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s47 baseline campaign_results): capture paths missing: {missing}')
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


def _common_checks(stage):
    failures, evidence = [], {}
    case_file = _check_case_file_loads_to_declaration()
    evidence['case_file_aa'] = case_file
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    pins, more = _check_pins()
    evidence['pins'] = pins
    failures += more
    memory_rule = _memory_rule_matches_a0()
    evidence['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f"memory preflight rule differs from A0's measure: {memory_rule}")
    base, more = baseline_inputs()
    evidence['baseline_inputs'] = base
    failures += more
    points = STAGES[stage]['points']()
    if len(points) != STAGES[stage]['expected_n']:
        failures.append(f'stage {stage}: {len(points)} points, expected {STAGES[stage]["expected_n"]}')
    labels = [p[0] for p in points]
    keys = [_eval_key(n, y) for _l, n, y in points]
    if len(set(labels)) != len(labels) or len(set(keys)) != len(keys):
        failures.append(f'stage {stage}: duplicate labels or eval keys')
    if stage == 's3':
        vs = _s3_points_vs_a1a()
        evidence['points_vs_spec_v15_and_a1a'] = vs
        if not (vs['n_matches_spec_v15'] and vs['equal_to_committed_a1a_entries_in_order']
                and vs['candidate_keys_equal_to_committed_a1a'] and vs['all_year_2025']):
            failures.append(f's3 points differ from spec v15 A1.ladders_2025 / the committed A1a entries: {vs}')
    evidence['points'] = [{'label': label, 'nodes': {str(n): list(v) for n, v in nodes.items()},
                           'investment_year': year, 'candidate_key': _key_of(nodes, year),
                           'eval_key': _eval_key(nodes, year)} for label, nodes, year in points]
    i_x, problems = investment_costs(points)
    evidence['investment_cost_per_point'] = i_x
    failures += problems
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, points, i_x


def _validate_spec(stage, spec, points, i_x, base):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    extra = spec.get('extra') or {}
    cfg = spec['configuration']
    checks = {
        'campaign_id': spec.get('campaign_id') == STAGES[stage]['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == [p[0] for p in points],
        'n_entries': len(entries) == STAGES[stage]['expected_n'],
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': ((cfg.get('ess_params_file') or {}).get('path') == ESS_PARAMS_FILE['path']
                                   and (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256']),
        'no_campaign_overrides': cfg.get('overrides') == {},
        'no_model_variant_anywhere': 'model_variant_label' not in spec and not any('model_variant' in e for e in entries),
        'cap': spec.get('cap') == CAP,
        'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': cfg.get('arm_label') == 's39_D',
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'label_recorded': extra.get('label') == LABEL,
        'stage_recorded': extra.get('stage') == stage,
        'spec_v17_recorded': extra.get('spec_v17') == SPEC_V17,
        'ess_params_file_recorded': extra.get('ess_params_file') == ESS_PARAMS_FILE,
        'Q0_frozen': (extra.get('baseline_inputs') or {}).get('Q0_eur') == base['Q0_eur'],
        'Q0_bar_frozen': (extra.get('baseline_inputs') or {}).get('Q0_bar_eur') == base['Q0_bar_eur'],
        'q0_statement_recorded': (extra.get('baseline_inputs') or {}).get('q0_statement') == Q0_STATEMENT,
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'budget_convention_recorded': extra.get('budget_convention') == BUDGET_CONVENTION,
        'memory_rule_recorded': extra.get('memory_preflight_rule') == MEMORY_RULE,
        'stop_rule_recorded': (extra.get('stop_rule') == STOP_RULE) if stage == 's3' else True,
        'post_certification_none': all(e.get('post_certification') is None for e in entries),
    }
    for label, nodes, year in points:
        entry = by_label.get(label) or {}
        canon = H.canonical_candidate(nodes, investment_year=year)
        key = H.candidate_key(canon)
        recorded = (extra.get('points') or {}).get(label) or {}
        checks[f'{label}:canonical_key'] = entry.get('canonical') == canon and entry.get('key') == key
        checks[f'{label}:no_overrides'] = entry.get('overrides') == {}
        checks[f'{label}:effective_aa_is_declaration'] = entry.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:eval_key_recomputes_with_declaration'] = entry.get('eval_key') == _eval_key(nodes, year)
        checks[f'{label}:eval_key_is_not_the_undeclared_key'] = entry.get('eval_key') != H.evaluation_key(
            key, {}, case_file_aa=CASE_FILE_AA)
        checks[f'{label}:I_x_frozen'] = (recorded.get('candidate_key') == key and recorded.get('I_x_eur') is not None
                                         and recorded.get('I_x_eur') == (i_x.get(label) or {}).get('I_x_eur'))
        checks[f'{label}:budget_slack_frozen'] = (recorded.get('budget_slack_eur') is not None and recorded.get(
            'budget_slack_eur') == (i_x.get(label) or {}).get('budget_slack_eur'))
    return checks


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(stage, started):
    tag = f'S47-{stage.upper()}'
    root = campaign_root(stage)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, points, i_x = _common_checks(stage)
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    base = evidence['baseline_inputs']
    extra = {'campaign_script': os.path.basename(__file__),
             'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'label': LABEL, 'mode': 'full', 'stage': stage, 'spec_v17_step': STAGES[stage]['spec_v17_step'],
             'stage_description': STAGES[stage]['description'],
             'spec_v17': dict(SPEC_V17), 'spec_v15': dict(SPEC_V15), 'ess_params_file': dict(ESS_PARAMS_FILE),
             'cost_file': dict(COST_FILE), 'a0_spec': dict(A0_SPEC),
             'investment_cost_results': dict(INVESTMENT_COST_RESULTS), 'pins': evidence['pins'],
             'baseline_inputs': base, 'points': {label: dict(i_x[label]) for label, _n, _y in points},
             'sensitivity_statement': SENSITIVITY_STATEMENT, 'post_certification': 'none', 'model_variant': 'none',
             'objective_convention': OBJECTIVE_CONVENTION, 'budget_convention': BUDGET_CONVENTION,
             'soh_floor_active_definition': (f'a floor row es_soh_per_unit_cumul[y_inv, y] >= soh_min is ACTIVE when '
                                             f'|SoH - soh_min| <= {FLOOR_ACTIVE_TOL:g} at the terminal cycle (the '
                                             'production floor sidecar, boyd_terminal.json '
                                             'soh_floor_multiplier_and_efc_per_cohort_year_terminal); the dual is '
                                             'recorded beside it'),
             'value_definition': 'value = Q(0) - Q(x); value - I = Q(0) - Q(x) - I(x) = F(0) - F(x) (I(0) = 0)',
             'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                'gross_operational_cost[k-1]| (harness W5)'),
             'memory_preflight_rule': MEMORY_RULE, 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
             'memory_preflight_refusing_at': '--run (non-gating at --freeze)', 'memory_at_freeze_non_gating': memory,
             'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)}
    if stage == 's3':
        extra.update({'stop_rule': STOP_RULE, 'wave_size': CONCURRENCY,
                      'points_vs_spec_v15_and_a1a': evidence['points_vs_spec_v15_and_a1a'],
                      'duplicate_of_s2': {
                          'label': 'n7_4h_e1', 'eval_key': _eval_key(_nodes_full({7: (0.25, 1.0)}), YEAR),
                          'note': ('n7_4h_e1 is also the S2 unit (campaign s47_recert) under the SAME evaluation key '
                                   '(same candidate, same declared configuration); spec v17 S3 asks for the 30 A1a '
                                   'points exactly, so it is kept here and flagged for the Planner (repeatability '
                                   'pair vs spec v15 execution.cache "a cache hit never re-evaluates")')}})
    else:
        extra.update({'execution': 'one batch of 2 at concurrency 5; non-certified points reported with cause'})
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, STAGES[stage]['campaign_id'], [(label, nodes, {'investment_year': year}) for label, nodes, year in points],
        configuration={'name': (f'{LABEL}: the case file alone (AA keep_memory adopted in data/SRP1/SRP1_params.json) '
                                'with the ESS ageing parameters declared'),
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('no overrides; no model variant; no post-certification; the ESS ageing declaration is '
                                'checked against the loaded parameters and read back from the built ESSO models in '
                                'the child; num_max_iters := cap')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=extra)
    checks = _validate_spec(stage, spec, points, i_x, base)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[{tag}] {LABEL}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for entry in spec['candidates']:
        p = extra['points'][entry['label']]
        _log(f"[{tag}]   {entry['label']}: nodes={p['nodes']} key={entry['key'][:16]} eval_key={entry['eval_key'][:16]} "
             f"eval_dir={entry['eval_dir']} I(x)={p['I_x_eur']} budget_slack={p['budget_slack_eur']}")
    _log(f"[{tag}] Q(0)={base['Q0_eur']} (bar {base['Q0_bar_eur']}); k(closed form)={base['k_closed_form']}")
    _log(f"[{tag}] ess_params_file pin: {spec['configuration']['ess_params_file']}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] baseline input checks: {base['checks']}")
    _log(f"[{tag}] pins: {evidence['pins']}")
    if stage == 's3':
        _log(f"[{tag}] points vs spec v15 / A1a: {evidence['points_vs_spec_v15_and_a1a']}")
        _log(f"[{tag}] duplicate of s2: {extra['duplicate_of_s2']}")
    _log(f"[{tag}] memory rule vs A0: {evidence['memory_rule_vs_a0_spec']}")
    _log(f"[{tag}] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --stage {stage} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _per_cycle_trajectory(rec):
    path = rec.get('per_cycle_record_path')
    if not path or not os.path.isfile(os.path.join(REPO, path)):
        return {'path': path, 'present': False}, []
    with open(os.path.join(REPO, path)) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {'path': path, 'present': True, 'sha256': H.sha256_file(os.path.join(REPO, path)),
            'n_rows': len(rows), 'n_rows_equals_cycles_run': len(rows) == rec.get('cycles_run')}, rows


def _floor_and_ageing(rec):
    """SoH-floor activity per (node, cohort, block) from the evaluation's own boyd_terminal.json (production floor
    sidecar, terminal line), and the SoH / EFC trajectory per block from the record's ageing_trajectory_terminal."""
    eval_dir = rec.get('eval_dir')
    bt_path = os.path.join(REPO, eval_dir, 'boyd_terminal.json') if eval_dir else None
    floor = None
    if bt_path and os.path.isfile(bt_path):
        with open(bt_path) as handle:
            floor = (json.load(handle) or {}).get('soh_floor_multiplier_and_efc_per_cohort_year_terminal')
    ageing = rec.get('ageing_trajectory_terminal') or {}
    year_of = {}
    trajectory, efc = {}, {}
    for node, data in (ageing.get('nodes') or {}).items():
        cells = data.get('cells') or []
        for c in cells:
            year_of[(str(c['y_inv']), str(c['y']))] = (c['investment_year'], c['block_year'])
        if cells:
            trajectory[node] = [{k: c.get(k) for k in ('investment_year', 'block_year', 'n_years', 'D', 'soh_prev_end',
                                                        'soh_end', 'soh_used_for_available_energy', 'e_rated',
                                                        'e_available', 'avg_ch_dch', 'cl_eff', 'phi_cal_in_model')}
                                for c in cells]
            efc[node] = {c['block_year']: c['efc_per_day'] for c in cells}
    storage_nodes = [n for n, v in (rec.get('storage_per_node') or {}).items() if v.get('has_storage')]
    entries = []
    if floor and floor.get('available'):
        for e in floor.get('per_node_per_cohort_year') or []:
            node = str(e.get('node_id'))
            if node not in storage_nodes or e.get('efc_per_day') is None:
                continue  # inactive cohort (SoH fixed at 1, no throughput)
            inv_year, block_year = year_of.get((str(e.get('y_inv')), str(e.get('y'))), (None, None))
            soh, soh_min = e.get('es_soh_per_unit_cumul'), e.get('soh_min')
            entries.append({'node': node, 'y_inv': e.get('y_inv'), 'y': e.get('y'), 'investment_year': inv_year,
                            'block_year': block_year, 'soh': soh, 'soh_min': soh_min,
                            'soh_minus_soh_min': (soh - soh_min) if (soh is not None and soh_min is not None) else None,
                            'active': e.get('active'), 'dual': e.get('dual'), 'efc_per_day': e.get('efc_per_day')})
    active = [e for e in entries if e['active']]
    return {'soh_floor': {'available': bool(floor and floor.get('available')), 'terminal_cycle': (floor or {}).get('cycle'),
                          'definition': f'active <=> |SoH - soh_min| <= {FLOOR_ACTIVE_TOL:g} (production floor sidecar)',
                          'any_floor_row_active': bool(active),
                          'active_rows': [{k: e[k] for k in ('node', 'investment_year', 'block_year', 'soh', 'soh_min',
                                                             'dual')} for e in active],
                          'active_blocks': sorted({f"node {e['node']} cohort {e['investment_year']} block "
                                                   f"{e['block_year']}" for e in active}),
                          'per_active_cohort_block': entries,
                          'min_soh_minus_soh_min': min((e['soh_minus_soh_min'] for e in entries
                                                        if e['soh_minus_soh_min'] is not None), default=None)},
            'soh_trajectory_per_block': trajectory, 'efc_per_day_per_block': efc,
            'salvage_per_node_model': {n: v.get('salvage_value') for n, v in (ageing.get('nodes') or {}).items()}}


def _readback_summary(block):
    if not block:
        return None
    return {'all_match': block.get('all_match'), 'expected': block.get('expected'),
            'per_node': {n: {k: (v.get('readback') or {}).get(k) for k in ('k', 'phi_cal_in_model', 'floor_row_lower')}
                         for n, v in (block.get('per_node') or {}).items()}}


def _point_result(label, rec, spec_point, base):
    rec = rec or {}
    traj, rows = _per_cycle_trajectory(rec)
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else None
    i_x = spec_point.get('I_x_eur')
    value = (base['Q0_eur'] - q) if q is not None else None
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  and prev.get('cycle') == (last.get('cycle') or 0) - 1 else None)
    tol = last.get('objective_tolerance')
    rule_ten = rec.get('rule_ten') or {}
    bar = (rec.get('bar') or {}).get('value')
    canonical = rec.get('candidate_canonical')
    verified = rec.get('ess_ageing_verified_pre_run') or {}
    return {
        'LABEL': LABEL, 'label': label, 'candidate_key': rec.get('candidate_key'), 'candidate_canonical': canonical,
        'investment_year': (canonical or {}).get('investment_year', spec_point.get('investment_year')),
        'region': _region_key(canonical) if canonical else None,
        'eval_key': rec.get('eval_key'), 'eval_dir': rec.get('eval_dir'),
        'status': rec.get('status'), 'barrier': rec.get('barrier'), 'barrier_cause': rec.get('barrier_cause'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'certified_cost_gross_settlement_excluded': q,
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'bar': {'value': bar, 'definition': (rec.get('bar') or {}).get('definition'),
                'n_steps_available': (rec.get('bar') or {}).get('n_steps_available')},
        'x0_bar': base['Q0_bar_eur'], 'bar_sum_with_x0': (bar + base['Q0_bar_eur']) if bar is not None else None,
        'bar_net_recourse_step_reported': (rec.get('bar_net_recourse_step_reported') or {}).get('value'),
        'rule_ten': {
            'terminal_step_over_threshold_production': rule_ten.get('terminal_step_over_threshold'),
            'terminal_objective_change_abs_production_net': rule_ten.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': rule_ten.get('terminal_objective_tolerance'),
            'terminal_gross_step_abs': gross_step,
            'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol) else None,
            'boyd_terminal_ratio_max_per_channel': rule_ten.get('boyd_terminal_ratio_max_per_channel')},
        'Q0_eur': base['Q0_eur'], 'value_eur': value,
        'I_x_eur': i_x, 'value_minus_I_eur': (value - i_x) if (value is not None and i_x is not None) else None,
        'F_eur': (i_x + q) if (q is not None and i_x is not None) else None,
        'budget_slack_eur': spec_point.get('budget_slack_eur'),
        'budget_feasible_corrected_file': spec_point.get('budget_feasible_corrected_file'),
        **_floor_and_ageing(rec),
        'ess_ageing_readback': {'pre_run_probes': _readback_summary(verified.get('readback_pre_run')),
                                'post_run_clones': _readback_summary(rec.get('ess_ageing_readback_terminal')),
                                'child_checks': verified.get('checks'),
                                'ess_params_sha256_in_child': rec.get('ess_params_sha256_in_child')},
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': {'record': rec.get('peak_rss'),
                     'parent_wait4_ru_maxrss_bytes': (rec.get('parent_view') or {}).get('wait4_ru_maxrss')},
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'first_pass_cycle_per_channel': rec.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': rec.get('terminal_ratios_per_channel'),
        'local_solve_failures': rec.get('local_solve_failures'),
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


def run(stage, started, spec_sha256):
    tag = f'S47-{stage.upper()}'
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    root_contents = sorted(os.listdir(root))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence, points, i_x = _common_checks(stage)
    failures += more
    base = evidence['baseline_inputs']
    checks = _validate_spec(stage, spec, points, i_x, base)
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
    _log(f'[{tag}] {LABEL}')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    labels = [p[0] for p in points]
    waves = [labels[i:i + CONCURRENCY] for i in range(0, len(labels), CONCURRENCY)]
    _log(f'[{tag}] {len(labels)} points in {len(waves)} wave(s) of <= {CONCURRENCY}'
         + (f'; stop rule: {STOP_RULE}' if stage == 's3' else ''))
    lock = H.acquire_campaign_lock(STAGES[stage]['campaign_id'], spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    records, wave_info, not_launched = [], [], []
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        for number, wave in enumerate(waves, start=1):
            if stage == 's3':
                state = stop_rule_state(records)
                if state['triggered']:
                    not_launched += wave
                    _log(f'[{tag}] STOP RULE: not launching wave {number} {wave}; {state["reasons"]}')
                    wave_info.append({'wave': number, 'labels': wave, 'launched': False,
                                      'reason': f'stop rule: {state["reasons"]}'})
                    continue
            _log(f'[{tag}] launching wave {number}/{len(waves)}: {wave} (concurrency {ctx.concurrency})')
            H.evaluate.last_batch_info = {}
            records += H.evaluate(wave, ctx)
            wave_info.append({'wave': number, 'labels': wave, 'launched': True,
                              'stop_rule_after_wave': stop_rule_state(records) if stage == 's3' else None,
                              **getattr(H.evaluate, 'last_batch_info', {})})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    stop_state = stop_rule_state(records) if stage == 's3' else {'triggered': False, 'reasons': []}
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    point_results = {}
    for label in labels:
        spec_point = spec['extra']['points'][label]
        if label in not_launched:
            point_results[label] = {'LABEL': LABEL, 'label': label, 'status': 'not_launched_stop_rule',
                                    'I_x_eur': spec_point.get('I_x_eur'),
                                    'budget_slack_eur': spec_point.get('budget_slack_eur'), 'F_eur': None}
        else:
            point_results[label] = _point_result(label, by_label.get(label), spec_point, base)
    launched = [label for label in labels if label not in not_launched]
    non_certified = [label for label in launched if point_results[label]['status'] != 'certified']
    harness_errors = [label for label in launched
                      if point_results[label]['status'] not in ('certified', 'not_certified')
                      or point_results[label]['exit_code'] != 0]
    readback_mismatch = [label for label in launched if point_results[label]['status'] in ('certified', 'not_certified')
                         and not (((point_results[label]['ess_ageing_readback']['pre_run_probes'] or {}).get('all_match'))
                                  and ((point_results[label]['ess_ageing_readback']['post_run_clones'] or {}).get(
                                      'all_match')))]
    guard_failures = PARENT_GUARD.verify(0)
    table = [{'label': l, 'status': point_results[l]['status'], 'cycles': point_results[l].get('cycles_run'),
              'cert_cycle': point_results[l].get('certification_cycle'),
              'Q': point_results[l].get('certified_cost_gross_settlement_excluded'),
              'value': point_results[l].get('value_eur'), 'I_x': point_results[l].get('I_x_eur'),
              'value_minus_I': point_results[l].get('value_minus_I_eur'), 'F': point_results[l].get('F_eur'),
              'budget_slack': point_results[l].get('budget_slack_eur'),
              'bar': (point_results[l].get('bar') or {}).get('value'),
              'floor_active_blocks': (point_results[l].get('soh_floor') or {}).get('active_blocks'),
              'salvage': point_results[l].get('terminal_salvage_value')} for l in labels]
    results = {
        'LABEL': LABEL,
        'STOP_FOR_REVIEW': stop_state['triggered'], 'stop_rule_state': stop_state,
        'non_certified_points': non_certified, 'not_launched_points': not_launched, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch,
        'stage': STAGES[stage]['description'], 'stage_id': stage, 'spec_v17_step': STAGES[stage]['spec_v17_step'],
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'budget_convention': BUDGET_CONVENTION,
        'sensitivity_statement': SENSITIVITY_STATEMENT, 'baseline_inputs': base,
        'table_objective_convention': 'Q gross (settlement excluded); value = Q(0) - Q; F = I + Q; salvage excluded',
        'table': table, 'points': point_results,
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pins', 'memory_rule_vs_a0_spec',
                                                      'investment_cost_per_point')},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'wave_info': wave_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    if stage == 's3':
        results['stop_rule'] = STOP_RULE
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f'[{tag}] {LABEL} -- objective convention: {OBJECTIVE_CONVENTION}')
    for row in table:
        _log(f'[{tag}] {row}')
    if non_certified:
        _log(f'[{tag}] non-certified points (reported with cause): '
             f"{[(l, point_results[l].get('barrier_cause')) for l in non_certified]}")
    if readback_mismatch:
        _log(f'[{tag}] ESS AGEING READ-BACK MISMATCH: {readback_mismatch}')
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    if stop_state['triggered']:
        _banner([f'STOP_FOR_REVIEW: the stop rule fired: {stop_state["reasons"]}', f'not launched: {not_launched}'])
    if guard_failures or harness_errors or readback_mismatch:
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
    parser.add_argument('--stage', required=True, choices=sorted(STAGES))
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
        freeze(args.stage, started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(args.stage, started, args.spec_sha256)


if __name__ == '__main__':
    main()
