"""
P5.15 Addendum 27, Phase A steps A1--A3 through the campaign harness
(`p515_s44_campaign_harness.py`), configuration = the case file alone (AA-on
keep_memory adopted in data/SRP1/SRP1_params.json), NO overrides, no
post-certification. Same launcher pattern as `p515_s45_a0_campaign.py` (task
W6/W7): preconditions, locks, clean-git, one spec per campaign root, memory
preflight, concurrency 7, cap 500, 10 consecutive all-pass cycles.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`
(`A1`, `A2`, `A3`, `execution`, `master_constraints`); Planner task W15 (stage
split a1a/a1b/a2/a3, the operational stop rule below, the selection rules as
code); P5_15_S45_REVERIFY_RULING.md (the bar is the GROSS cost step, fixed in
the harness by W5); year support = W14 (`85c147fa`, per-point
`investment_year`).

FOUR STAGES (`--stage`), one campaign root and one frozen spec each:
  a1a  30 points -- spec v15 `A1.ladders_2025`: each of nodes 5, 7, 9 ALONE
       (the other two zero), E in {1,2,3,4,5} MWh at 2 h (P = E/2) and 4 h
       (P = E/4), investment year 2025. Campaign id `s45_a1a`, root
       `data/SRP1/Results/P515S45/campaign_s45_a1a/`. FROZEN by W15.
  a1b  20 points -- the year ladder at the BEST NODE: both durations, E in
       {1..5}, investment years 2030 and 2035 (spec v15 A1.year_ladder, whose
       own note flags that "20 points" implies both durations).
  a2    5 points -- the 2^|N| presence design of the best per-node settings
       (the 4 subsets with >= 2 nodes; the singletons and x = 0 are already
       evaluated) + C* rounded to the lattice (1.0 MVA / 4.0 MWh at all nodes).
       Spec v15 A2 also names 2 STAGING points: NOT available here, see
       `STAGING_NOT_AVAILABLE` -- the launcher refuses such a point instead of
       mis-specifying it as single-cohort.
  a3    5 points -- the resolution ladder at node 7 (0.25 MVA / 0.5 MWh lattice
       step), FIXED by the Planner (task W17) to the 5 BUDGET-FEASIBLE
       candidates of the 14 priced by W16: (P, E) = (0.5, 1.5), (0.75, 1.5),
       (0.75, 2.5), (1.0, 2.5), (1.25, 2.5) MVA/MWh at node 7 alone, 2025. The
       other 9 were excluded by Planner decision (all above 3.5 MWh, where
       I(x) > B = 1e6). `A3_SELECTION_RULE` / `A3_PURPOSE` / `A3_ANCHOR_RULE`
       carry the rule, the purpose (measure sigma_Q at the lattice step) and
       the x = 0 incumbent ruling; the anchor is still COMPUTED and the freeze
       refuses if it is not the Planner-declared one.
a1b, a2 and a3 are implemented but NOT frozen by W15: their points (a1b, a2) or
their anchor and provenance (a3) are functions of the PRIOR stages' outcomes, so
each `--freeze` reads the prior stages' COMMITTED `campaign_results.json`, pins
them by sha256, recomputes the selection rule and records every input it used.

SELECTION RULES (frozen in spec v15 A1/A3; implemented here so that nothing is
decided by judgement at run time). F(x) = I(x) + Q(x), Q = the certified
`gross_operational_cost`:
  * best node and duration  = argmin F over the a1a records; ties -> lower I(x)
                              (then, if still tied, the lower label: recorded).
  * best setting for node n = argmin F within node n's a1a ladder (same tie rule).
  * incumbent               = argmin F over everything certified so far (A0 +
                              a1a + a1b; same tie rule). It is x = 0.
  * A3 anchor               = the incumbent when it is non-zero; when it is
                              x = 0 (the case here), the best NON-ZERO point of
                              A1's ladders (`A3_ANCHOR_RULE`, Planner ruling
                              W17). The best non-zero over ALL prior stages is
                              recorded beside it with the difference.

BATCHING AND STOP RULE. Waves of `CONCURRENCY` (7) points in spec order (a1a:
5 waves = 7+7+7+7+2); the stop rule is evaluated after every wave, and points
already running always finish. Spec v15 `execution.barrier` says only that "in
A1-A3 a run of barrier points in a region stops for review"; the operational
form frozen by W15 is:
  STOP (launch no further wave) if
    (a) 2 or more non-certified points fall in ONE region (node+duration ladder,
        `_region_key`), or
    (b) 3 or more non-certified points occur overall.
A single isolated non-certified point does NOT stop the campaign: it is recorded
as a barrier point with its cause and the campaign continues.

MEMORY PREFLIGHT (refusing at --run; recorded, non-gating, at --freeze) -- the
measure and the threshold are A0's, byte-identical: available = hw.memsize -
(wired + anonymous + compressor-occupied) pages x page size, required >=
7 x 2.75 GiB. The rule text is checked against the one recorded in A0's frozen
spec (`A0_SPEC`), so the two launchers cannot drift apart silently.

RECORDED PER POINT in campaign_results.json: status, certification cycle, Q
(gross, settlement-excluded), the bar (gross-step definition), the rule-ten
terminal-step-to-threshold ratio, terminal_salvage_value, net operational
recourse, I(x) for that candidate AND year (from the ORDERED list of pinned
I(x) tables `I_X_SOURCES`: W2's committed
`investment_cost/investment_cost_results.json` (9e623dd3) first, then W16's
`investment_cost_a2a3/investment_cost_a2a3_results.json` (1f2ca6f7) -- matched
by candidate key, every point must be found, a key held by more than one source
must agree EXACTLY on I(x), and the source used is recorded per point),
F = I + Q, the budget slack at B = 1e6 EUR
(REPORTED, never applied: spec v15 master_constraints.budget), wall time, peak
RSS, the AA action counts and the per-cycle trajectory path (+ sha256).
OBJECTIVE CONVENTION on every table: F uses GROSS; salvage is reported
separately. Rule eleven (a capture path for every one of those fields) is
asserted BEFORE anything is launched, in both modes.

Two modes, both attached, both streams captured, never detached:
  --stage <s> --freeze          ZERO SOLVES: preconditions, pinned files, the
                                stage's points (a1a: == spec v15; later stages:
                                the selection rule over the pinned prior
                                results), I(x) for every point, rule eleven,
                                then `freeze_campaign_spec` + validation.
  --stage <s> --run --spec-sha256 <sha>
                                loads THAT spec from THIS stage's root (which
                                must hold only it), re-checks everything above
                                plus the harness / case-file / script sha256
                                recorded in the spec, the memory preflight
                                (refusing), takes the campaign lock, evaluates
                                the waves, writes campaign_results.json and
                                campaign_manifest_sha256.json.
Exit codes (--run): 0 every point certified; 3 STOP_FOR_REVIEW (the stop rule
fired, harness clean); 1 harness / guard / precondition failure. The parent
never solves: SolveProfileGuard(permitted=()) is installed before any model
import and verified at exactly 0 in both modes.

EXACT COMMANDS (repo root, canonical interpreter):
  freeze a1a (zero solves):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_a1_campaign.py \\
        --stage a1a --freeze \\
        > data/SRP1/Results/P515S45/campaign_s45_a1a_freeze_launch.log 2>&1
  run a1a (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_a1_campaign.py \\
        --stage a1a --run --spec-sha256 <sha256 printed by --freeze> \\
        > data/SRP1/Results/P515S45/campaign_s45_a1a_launch.log 2>&1
  the same two commands with `--stage a1b`, `--stage a2`, `--stage a3` and that
  stage's log name (`campaign_s45_<stage>_freeze_launch.log` /
  `campaign_s45_<stage>_launch.log`); each --run takes the sha256 its own
  --freeze printed. One stage at a time, attached, never detached.
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 A1 campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 7
REQUIRED_CONSECUTIVE_CYCLES = 10
BUDGET_EUR = 1e6  # spec v15 master_constraints.budget: reported in Phase A, applied only from Phase B
ACTIVE_NODES = (5, 7, 9)
LATTICE_P_STEP = 0.25   # MVA
LATTICE_E_STEP = 0.5    # MWh
MAX_ENERGY_MWH = 5.0    # spec v15 master_constraints.max_capacity
DURATION_MIN_H, DURATION_MAX_H = 2.0, 4.0  # spec v15 master_constraints.duration
TOL = 1e-9

SPEC_V15 = {'path': os.path.join(_P45, 'frozen_s45_phaseA_spec_v15_5feefd7b.json'),
            'sha256': '5feefd7b642fc3d480156ad5e52ed6e1cf9d6698cfd40dbb33389bab6e6229fe'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'commit': '9e623dd304837f2632964f47130d0860c1f016ca',
                           'field': 'candidates.<label>.I_new_eur (corrected cost file), matched by candidate_key'}
INVESTMENT_COST_RESULTS_A2A3 = {
    'path': os.path.join(_P45, 'investment_cost_a2a3', 'investment_cost_a2a3_results.json'),
    'sha256': 'a7e3cca5564d4a77e93ba32d2bc48a029b9001ffd2894aa033fc0d70e51dff19',
    'commit': '1f2ca6f79f181a18f82ec513f38c4c11647d1e7a',
    'field': 'candidates.<label>.I_new_eur (corrected cost file), matched by candidate_key'}
# ORDERED list of pinned I(x) sources (Planner task W17 item 1): searched in this order, the FIRST source that
# holds the candidate key supplies I(x); a key present in more than one source must agree EXACTLY on I(x) or the
# freeze fails loudly. Each point records which source its I(x) came from and what every source holding it says.
# The two tables use different field names for the same quantities, so each source declares its own field map.
I_X_SOURCES = (
    dict(INVESTMENT_COST_RESULTS, name='w2_frozen_table', order=1,
         produced_by='p515_s45_investment_cost_recompute.py (task W2)',
         covers=('x = 0, the reference plans (paper plan, C*, lattice plan, lattice C*, 2 x C*, node7_empty) and '
                 'the SINGLE-NODE ladders at 2025 / 2030 / 2035'),
         fields={'I_x_eur': 'I_new_eur', 'budget_slack_eur': 'slack_new_eur', 'I_power_eur': 'I_new_power_eur',
                 'I_energy_eur': 'I_new_energy_eur', 'budget_feasible': 'budget_feasible_new',
                 'first_stage_feasible': 'first_stage_feasible_new_production_check'}),
    dict(INVESTMENT_COST_RESULTS_A2A3, name='w16_a2a3_table', order=2,
         produced_by='p515_s45_investment_cost_a2a3.py (task W16); SECOND, ADDITIVE table, the first is unchanged',
         covers=('the A2 presence design (4 multi-node points) + the lattice C* (reused from the first table) and '
                 'the 14 new node-7 A3 resolution candidates at 2025'),
         fields={'I_x_eur': 'I_new_eur', 'budget_slack_eur': 'budget_slack_B_minus_I_eur',
                 'I_power_eur': 'I_new_power_eur', 'I_energy_eur': 'I_new_energy_eur',
                 'budget_feasible': 'budget_feasible',
                 'first_stage_feasible': 'first_stage_feasible_production_check'}),
)
I_X_SOURCE_RULE = ('I(x) is read from an ORDERED list of pinned I(x) tables (path + sha256 in this spec): '
                   + ' then '.join(f'{s["order"]}. {s["name"]} ({s["path"]})' for s in I_X_SOURCES)
                   + '. The first source holding the candidate key supplies I(x); a key held by more than one '
                     'source must agree EXACTLY on I(x) (any disagreement fails the freeze); every point records '
                     'the source used and the value each holding source gives.')
# A0's frozen spec: the source of the memory-preflight rule text and of the shared campaign settings.
A0_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_spec_s45_a0_c7_9d08ad2f.json'),
           'sha256': '9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34'}
A0_RESULTS_PATH = os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json')
# specs of OTHER campaigns: never loadable here (each stage looks only in its own root), refused explicitly.
FOREIGN_SPEC_SHA256 = {
    A0_SPEC['sha256']: 'the A0 campaign spec (campaign_s45_a0_c7)',
    '25a0534754bccf1c902a548084d214ed7cdf9227af6b98c05b5485ac866fc061': 'the superseded A0 predecessor spec',
}

GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 11 * GIB // 4  # 2.75 GiB (measured peak RSS per evaluation 2.30-2.42 GiB in A0)
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE = (f'hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {CONCURRENCY} x '
               f'{MEMORY_PER_CHILD_BUDGET_BYTES / GIB:g} GiB')
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside)')

OBJECTIVE_CONVENTION = ('F uses gross; salvage reported separately. Q(x) = certified_cost = gross_operational_cost '
                        '(settlement-excluded); F(x) = I(x) + Q(x); terminal_salvage_value and '
                        'net_operational_recourse = gross - salvage are reported, excluded from F '
                        '(Addendum 27 item 3).')
BUDGET_CONVENTION = (f'budget_slack_eur = B - I(x) at B = {BUDGET_EUR:g} EUR, REPORTED only; the budget is NOT '
                     'applied in Phase A (spec v15 master_constraints.budget: "applied from Phase B only; A1 runs '
                     'without it by construction")')
STOP_RULE = ('spec v15 execution.barrier: "non-certified evaluations recorded with cause; in A1-A3 a run of barrier '
             'points in a region stops for review". Operational form frozen by Planner task W15: after every wave, '
             'STOP (launch no further wave) if (a) 2 or more non-certified points fall in ONE region (node+duration '
             'ladder, see _region_key) or (b) 3 or more non-certified points occur overall. A single isolated '
             'non-certified point does NOT stop the campaign: it is recorded as a barrier point with its cause. '
             'Points already running always finish; unlaunched points are recorded as not_launched_stop_rule and '
             'campaign_results.json carries STOP_FOR_REVIEW true (exit 3).')
LAUNCH_PLAN = (f'waves of {CONCURRENCY} points in spec order; the stop rule is evaluated after each wave (the '
               "harness's evaluate() has no stop hook, so the wave is the unit at which the campaign can stop)")
STAGING_NOT_AVAILABLE = (
    'STAGING IS NOT AVAILABLE in this campaign: the canonical candidate form carries exactly ONE investment year '
    '(p515_s44_campaign_harness.canonical_candidate; multi-cohort candidates were explicitly out of scope of the '
    'year support added by W14, 85c147fa). Spec v15 A2 names 2 staging points ("half the best capacity in y1 plus '
    'half in y2" and "the full capacity in y2 alone"); the first needs two cohorts at one node and cannot be '
    'specified here, and neither is included in stage a2. A staged point is REFUSED rather than mis-specified as a '
    'single-cohort point.')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 (Phase A, A1-A3)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json A1, A2, A3, execution, master_constraints',
    'P5_15_S45_REVERIFY_RULING.md (oracle re-verified; consequence 2: bar on the gross cost step, fixed by W5)',
    'Planner task W15: stages a1a/a1b/a2/a3, the operational stop rule, the selection rules as code',
    'Planner decision W6: Phase A concurrency 7, memory preflight wired + anonymous + compressor-occupied',
    'W14 (85c147fa): per-point investment_year in the campaign harness (2030 / 2035 verified)',
    'W16 (1f2ca6f7): the second, additive I(x) table for the A2 presence design and the A3 resolution '
    'candidates, and the zero-solve sigma_Q estimate from the existing ladders',
    'Planner task W17: the ordered list of pinned I(x) sources; the Planner-fixed a3 point set (the 5 '
    'budget-feasible resolution candidates); the x = 0 incumbent / anchor ruling for A3',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), 'p515_s45_a0_campaign.py', 'p515_s45_harness_phasea_check.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _banner(lines):
    bar = '!' * 100
    _log(bar)
    for line in lines:
        _log(f'!!! {line}')
    _log(bar)


# ======================================================================================================================
#  candidates: helpers shared by the stage builders
# ======================================================================================================================
def _nodes_full(partial):
    """A full active-node map (the harness forbids implicit zeros) from the named nodes alone."""
    nodes = {n: (0.0, 0.0) for n in ACTIVE_NODES}
    for node, (s_val, e_val) in partial.items():
        nodes[int(node)] = (float(s_val), float(e_val))
    return nodes


def _nonzero_nodes(canonical):
    return {int(n): (float(v[0]), float(v[1])) for n, v in canonical['nodes'].items()
            if float(v[0]) != 0.0 or float(v[1]) != 0.0}


def _duration_h(s_val, e_val):
    return (e_val / s_val) if s_val else None


def _region_key(canonical):
    """The 'region' of the stop rule: the node+duration ladder a point belongs to (x0 for the empty
    candidate; multi-node points form their own region, named by every node and duration)."""
    nz = _nonzero_nodes(canonical)
    if not nz:
        return f"x0_y{canonical['investment_year']}"
    parts = [f'n{node}_{_duration_h(*nz[node]):g}h' for node in sorted(nz)]
    return '+'.join(parts) + f"_y{canonical['investment_year']}"


def _on_lattice(value, step):
    return abs(value / step - round(value / step)) < TOL


def _canonical_of(nodes, year):
    return H.canonical_candidate(nodes, investment_year=year)


def _key_of(nodes, year):
    return H.candidate_key(_canonical_of(nodes, year))


def _refuse_staged(label, years_by_node):
    """Refuse a point that places capacity in more than one investment year (spec v15 A2 staging)."""
    years = sorted({int(y) for y in years_by_node.values()})
    if len(years) > 1:
        raise SystemExit(f'[S45-A1 REFUSED] point {label!r} places capacity in years {years}. '
                         f'{STAGING_NOT_AVAILABLE}')
    return years[0] if years else None


# ======================================================================================================================
#  stage a1a: spec v15 A1.ladders_2025 (frozen by W15)
# ======================================================================================================================
def _a1a_points():
    """Spec v15 A1.ladders_2025 in spec order: node-major, then E, then 2 h before 4 h."""
    points = []
    for node in ACTIVE_NODES:
        for energy in (1, 2, 3, 4, 5):
            for hours in (2, 4):
                power = energy / hours
                points.append((f'n{node}_{hours}h_e{energy}', _nodes_full({node: (power, float(energy))}), 2025))
    return tuple(points)


A1A_POINTS = _a1a_points()


def _spec_v15_block(path):
    with open(os.path.join(REPO, path)) as handle:
        return json.load(handle)


def _a1a_points_vs_spec_v15():
    ladders = _spec_v15_block(SPEC_V15['path'])['A1']['ladders_2025']
    spec_points = []
    for point in ladders['points']:
        nodes = {n: (0.0, 0.0) for n in ACTIVE_NODES}
        for node, (s_val, e_val) in point['nodes'].items():
            nodes[int(node)] = (float(s_val), float(e_val))
        spec_points.append((point['label'], nodes, int(point['investment_year'])))
    mine = [(label, {n: (float(v[0]), float(v[1])) for n, v in nodes.items()}, int(year))
            for label, nodes, year in A1A_POINTS]
    return {'equal_to_spec_v15_A1_ladders_2025_in_order': mine == spec_points,
            'spec_n': ladders.get('n'), 'n_points': len(A1A_POINTS),
            'n_matches_spec_n': ladders.get('n') == len(A1A_POINTS),
            'spec_note': ladders.get('note'),
            'all_points_year_2025': all(year == 2025 for _l, _n, year in A1A_POINTS)}


# ======================================================================================================================
#  the selection rules (spec v15 A1 / A3; frozen, so no judgement at run time)
# ======================================================================================================================
def _record_from_results_point(stage, results_path, label, point):
    canonical = point.get('candidate_canonical')
    return {
        'stage': stage, 'source_results': results_path, 'label': label,
        'status': point.get('status'), 'candidate_key': point.get('candidate_key'),
        'canonical': canonical,
        'investment_year': (canonical or {}).get('investment_year'),
        'nodes': _nonzero_nodes(canonical) if canonical else None,
        'Q_gross_eur': point.get('certified_cost_gross_settlement_excluded'),
        'I_x_eur': point.get('I_x_eur'),
        'F_eur_as_recorded': point.get('F_eur'),
    }


def load_prior_records(prior_pins):
    """Read the PRIOR stages' committed campaign_results.json. Returns (records, evidence, problems).

    Every file must exist, be tracked and clean in git (a stage's inputs must be committed evidence),
    and match the sha256 pinned in the frozen spec when one is given. F is RECOMPUTED as I + Q and
    cross-checked against the value the prior campaign recorded."""
    records, evidence, problems = [], [], []
    for pin in prior_pins:
        rel = pin['path']
        abs_path = os.path.join(REPO, rel)
        entry = {'stage': pin['stage'], 'path': rel, 'exists': os.path.isfile(abs_path)}
        if not entry['exists']:
            problems.append(f'prior results missing: {rel} (stage {pin["stage"]})')
            evidence.append(entry)
            continue
        entry['sha256'] = H.sha256_file(abs_path)
        if pin.get('sha256') and pin['sha256'] != entry['sha256']:
            problems.append(f'prior results {rel}: sha256 {entry["sha256"]} != pinned {pin["sha256"]}')
        tracked = bool(H._git(['ls-files', '--', rel]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
        entry.update({'git_tracked': tracked, 'git_clean': not dirty})
        if not tracked or dirty:
            problems.append(f'prior results {rel} must be committed and clean (tracked={tracked}, clean={not dirty})')
        with open(abs_path) as handle:
            data = json.load(handle)
        entry['stop_for_review'] = data.get('STOP_FOR_REVIEW')
        entry['n_points'] = len(data.get('points') or {})
        for label, point in (data.get('points') or {}).items():
            record = _record_from_results_point(pin['stage'], rel, label, point)
            if record['status'] == 'certified':
                if record['Q_gross_eur'] is None or record['I_x_eur'] is None:
                    problems.append(f'{rel}:{label} is certified but lacks Q or I(x)')
                else:
                    record['F_eur'] = record['I_x_eur'] + record['Q_gross_eur']
                    recorded = record['F_eur_as_recorded']
                    if recorded is not None and abs(recorded - record['F_eur']) > 1e-6:
                        problems.append(f'{rel}:{label}: recorded F {recorded} != I + Q {record["F_eur"]}')
            else:
                record['F_eur'] = None
            records.append(record)
        evidence.append(entry)
    return records, evidence, problems


def _argmin_F(records, rule_text):
    """argmin over F; ties -> lower I(x); then -> lower label (deterministic, recorded).

    Returns (best, evidence) with the evidence listing EVERY candidate's F and key (the inputs used)."""
    candidates = [r for r in records if r['status'] == 'certified' and r.get('F_eur') is not None]
    if not candidates:
        raise SystemExit(f'[S45-A1 REFUSED] the selection rule has no certified input: {rule_text}')
    ordered = sorted(candidates, key=lambda r: (r['F_eur'], r['I_x_eur'], r['label']))
    best = ordered[0]
    tied_F = [r['label'] for r in candidates if r['F_eur'] == best['F_eur']]
    tied_F_and_I = [r['label'] for r in candidates
                    if r['F_eur'] == best['F_eur'] and r['I_x_eur'] == best['I_x_eur']]
    evidence = {
        'rule': rule_text,
        'n_certified_inputs': len(candidates),
        'n_inputs_seen': len(records),
        'non_certified_inputs': sorted(r['label'] for r in records if r['status'] != 'certified'),
        'selected': best['label'], 'selected_candidate_key': best['candidate_key'],
        'tie_on_F': tied_F if len(tied_F) > 1 else [],
        'tie_on_F_and_I_broken_by_label': tied_F_and_I if len(tied_F_and_I) > 1 else [],
        'inputs': [{'label': r['label'], 'stage': r['stage'], 'source_results': r['source_results'],
                    'candidate_key': r['candidate_key'], 'investment_year': r['investment_year'],
                    'nodes': {str(n): list(v) for n, v in (r['nodes'] or {}).items()},
                    'I_x_eur': r['I_x_eur'], 'Q_gross_eur': r['Q_gross_eur'], 'F_eur': r['F_eur']}
                   for r in ordered],
    }
    return best, evidence


def _single_node(record, what):
    nz = record['nodes'] or {}
    if len(nz) != 1:
        raise SystemExit(f'[S45-A1 REFUSED] {what}: {record["label"]} has {len(nz)} non-zero nodes '
                         f'({sorted(nz)}); this stage is defined for a single-node candidate only. '
                         f'Planner decision required.')
    node = next(iter(nz))
    return node, nz[node][0], nz[node][1]


def select_best_node_and_duration(records):
    """spec v15 A1.best_node_and_duration, over the a1a ladders only."""
    a1a = [r for r in records if r['stage'] == 'a1a']
    best, evidence = _argmin_F(a1a, 'A1 best node and duration = argmin over the a1a 2025 ladders of '
                                    'F(x) = I(x) + Q(x); ties -> lower I(x)')
    node, power, energy = _single_node(best, 'best node and duration')
    return {'node': node, 'P_mva': power, 'E_mwh': energy, 'duration_h': _duration_h(power, energy),
            'investment_year': best['investment_year'], 'label': best['label'],
            'F_eur': best['F_eur'], 'I_x_eur': best['I_x_eur'], 'Q_gross_eur': best['Q_gross_eur'],
            'evidence': evidence}


def select_best_setting_per_node(records):
    """The best single-node setting for EACH node, within that node's a1a ladder."""
    out = {}
    for node in ACTIVE_NODES:
        ladder = [r for r in records
                  if r['stage'] == 'a1a' and r['nodes'] is not None and list(r['nodes']) == [node]]
        best, evidence = _argmin_F(ladder, f'A2 best setting for node {node} = argmin F within node {node}\'s '
                                           f'a1a ladder; ties -> lower I(x)')
        _node, power, energy = _single_node(best, f'best setting for node {node}')
        out[node] = {'node': node, 'P_mva': power, 'E_mwh': energy, 'duration_h': _duration_h(power, energy),
                     'investment_year': best['investment_year'], 'label': best['label'], 'F_eur': best['F_eur'],
                     'I_x_eur': best['I_x_eur'], 'Q_gross_eur': best['Q_gross_eur'], 'evidence': evidence}
    return out


def select_incumbent(records):
    """The incumbent = argmin F over EVERYTHING certified so far (every prior stage pinned for this stage)."""
    best, evidence = _argmin_F(records, 'A3 incumbent = argmin F over everything certified so far; '
                                        'ties -> lower I(x)')
    return {'label': best['label'], 'stage': best['stage'], 'candidate_key': best['candidate_key'],
            'canonical': best['canonical'], 'nodes': {str(n): list(v) for n, v in (best['nodes'] or {}).items()},
            'investment_year': best['investment_year'], 'F_eur': best['F_eur'], 'I_x_eur': best['I_x_eur'],
            'Q_gross_eur': best['Q_gross_eur'], 'record': best, 'evidence': evidence}


# ======================================================================================================================
#  stage point builders (a1b, a2, a3) -- computed from the prior stages' committed results
# ======================================================================================================================
def _evaluated_keys(records):
    return {r['candidate_key']: r['label'] for r in records if r.get('candidate_key')}


def _refuse_cache_hits(stage, points, records):
    """spec v15 execution.cache: a cache hit never re-evaluates. For a1b / a2 / a3 a repeat is a specification
    error, so the freeze REFUSES (the Planner removes the point or reuses the prior record). a3's points are
    Planner-specified since W17, so it is refused there too rather than silently dropped."""
    seen = _evaluated_keys(records)
    hits = [(label, seen[_key_of(nodes, year)]) for label, nodes, year in points
            if _key_of(nodes, year) in seen]
    if hits:
        raise SystemExit(f'[S45-A1 REFUSED] stage {stage}: these points are already evaluated (spec v15 '
                         f'execution.cache: "a cache hit never re-evaluates"): '
                         f'{[f"{new} == {old}" for new, old in hits]}')


def build_a1b_points(records):
    """The year ladder at the BEST NODE: both durations, E in {1..5}, years 2030 and 2035 (20 points)."""
    best = select_best_node_and_duration(records)
    node = best['node']
    points = []
    for year in (2030, 2035):
        for energy in (1, 2, 3, 4, 5):
            for hours in (2, 4):
                power = energy / hours
                points.append((f'n{node}_{hours}h_e{energy}_y{year}',
                               _nodes_full({node: (power, float(energy))}), year))
    _refuse_cache_hits('a1b', points, records)
    provenance = {
        'rule': ('spec v15 A1.year_ladder: investment years 2030 and 2035 at the best node, E in {1..5}, BOTH '
                 'durations = 20 points (the spec\'s own note flags that the stated count implies both durations)'),
        'best_node_and_duration': best,
        'node_used': node,
        'duration_used': 'both (2 h and 4 h)',
        'years': [2030, 2035],
    }
    return tuple(points), provenance


def build_a2_points(records):
    """The 2^|N| presence design of the best per-node settings (4 new) + the lattice C* (1)."""
    per_node = select_best_setting_per_node(records)
    subsets = ((5, 7), (5, 9), (7, 9), (5, 7, 9))
    points = []
    for subset in subsets:
        label = 'presence_' + '_'.join(f'n{n}' for n in subset)
        years_by_node = {n: per_node[n]['investment_year'] for n in subset}
        year = _refuse_staged(label, years_by_node)  # different years at different nodes would be staging
        nodes = _nodes_full({n: (per_node[n]['P_mva'], per_node[n]['E_mwh']) for n in subset})
        points.append((label, nodes, year))
    points.append(('lattice_c_star_p1.0_e4.0',
                   _nodes_full({n: (1.0, 4.0) for n in ACTIVE_NODES}), 2025))
    points = tuple(points)
    _refuse_cache_hits('a2', points, records)
    provenance = {
        'rule': ('spec v15 A2: "the best single-node setting per node combined in a 2^|N| presence design (4 new '
                 'points) ... C* rounded to the lattice (1.0 MVA / 4.0 MWh at all nodes) - 1 point". The 4 new '
                 'points are the subsets with at least two nodes: the empty set is A0\'s x0 and the three '
                 'singletons are a1a points.'),
        'best_setting_per_node': {str(n): per_node[n] for n in ACTIVE_NODES},
        'subsets': [list(s) for s in subsets],
        'staging': STAGING_NOT_AVAILABLE,
        'staging_points_in_spec_v15_not_included': ['half the best capacity in y1 + half in y2 (two cohorts)',
                                                    'the full best capacity in y2 alone'],
        'lattice_c_star': '1.0 MVA / 4.0 MWh at nodes 5, 7, 9, investment year 2025',
    }
    return points, provenance


# ---- stage a3: the Planner-fixed resolution set (task W17 item 4) ----------------------------------------------
A3_NODE = 7
A3_YEAR = 2025
A3_POINTS_P_E_MVA_MWH = ((0.5, 1.5), (0.75, 1.5), (0.75, 2.5), (1.0, 2.5), (1.25, 2.5))
A3_PLANNER_DECLARED_ANCHOR_LABEL = 'n7_4h_e1'
A3_SELECTION_RULE = (
    'Planner task W17 item 4 (replacing the generated probe of W15): the a3 points are FIXED by the Planner to the '
    'five BUDGET-FEASIBLE candidates of the 14 new node-7 resolution candidates priced by W16 '
    '(data/SRP1/Results/P515S45/investment_cost_a2a3/investment_cost_a2a3_results.json, 1f2ca6f7): '
    '(P, E) = (0.5, 1.5), (0.75, 1.5), (0.75, 2.5), (1.0, 2.5), (1.25, 2.5) MVA/MWh at node 7 ALONE, the other '
    'active nodes zero, investment year 2025. The remaining 9 of W16\'s 14 were EXCLUDED BY PLANNER DECISION '
    'because they all lie above 3.5 MWh, where I(x) exceeds the EUR 1e6 budget (B - I(x) < 0 for every one of '
    'them; see points_provenance.w16_a3_candidates.excluded), so the budget of Phase B could never reach them. '
    'The two single-P-step '
    'neighbours of the anchor are not new points either: P* + 0.25 = 0.5 MVA at E* = 1.0 MWh is a CACHE HIT '
    '(a1a:n7_2h_e1, spec v15 execution.cache: a cache hit never re-evaluates) and P* - 0.25 = 0 MVA does not exist '
    '(P = 0 <=> E = 0).')
A3_ANCHOR_RULE = (
    'A3 anchor (Planner ruling, task W17 item 2). The incumbent = argmin F over everything certified so far is '
    'recorded as always. When the incumbent is x = 0 -- as it is here: no storage point beats x0 by F -- the '
    'resolution probe cannot be centred on it (a zero candidate has no P*, E* to step around), and W15\'s builder '
    'raised. The Planner ruling is that in that case the A3 anchor is the BEST NON-ZERO POINT, argmin F over the '
    'certified non-zero candidates. The scope of that argmin is A1\'s ladders (stages a1a and a1b), which is what '
    'spec v15 A3 names: "resolution probe folded into A1\'s best ladder". The argmin over ALL pinned prior stages '
    '(A0 included) is computed and recorded beside it, with the difference, so the scope is visible rather than '
    'implied. Both the incumbent and the anchor are recorded; the freeze fails if the computed anchor is not the '
    f'one the Planner declared ({A3_PLANNER_DECLARED_ANCHOR_LABEL}).')
A3_PURPOSE = (
    'Purpose (Planner task W17 item 4): measure sigma_Q at the 0.5 MWh / 0.25 MVA lattice resolution IN THE '
    'BUDGET-FEASIBLE REGION, replacing the provisional 1.1e-4 of STEP4_DFO_METHOD.md 4.3. W16\'s zero-solve '
    'estimate from the existing 1 MWh ladders gives sigma_Q max 28,208.38 EUR (4.319e-05 of Q), median '
    '14,374.98 EUR (2.200e-05), min 5,208.63 EUR over 10 ladders, i.e. 0.393 / 0.200 of the provisional figure; '
    'and in 5 of those 10 ladders sigma_Q is NOT separated from the per-point bar (the harness bar, max over the '
    'last 10 cycles of the gross-cost step). A3 measures it on a ladder actually spaced at the lattice step.')


def select_a3_anchor(records):
    """The incumbent and the A3 anchor (A3_ANCHOR_RULE). Returns (anchor_record, evidence); the anchor is the
    incumbent itself when the incumbent is non-zero, and the best non-zero A1-ladder point when it is x = 0."""
    incumbent = select_incumbent(records)
    non_zero = [r for r in records if r['status'] == 'certified' and r.get('F_eur') is not None and r['nodes']]
    best_all, evidence_all = _argmin_F(non_zero, 'best NON-ZERO certified point over every pinned prior stage '
                                                 '(recorded for scope, not used as the anchor); ties -> lower I(x)')
    a1_ladders = [r for r in non_zero if r['stage'] in ('a1a', 'a1b')]
    best_a1, evidence_a1 = _argmin_F(a1_ladders, 'A3 anchor = argmin F over the certified NON-ZERO points of A1\'s '
                                                 'ladders (stages a1a, a1b); ties -> lower I(x)')
    incumbent_is_zero = not incumbent['nodes']
    anchor = best_a1 if incumbent_is_zero else incumbent['record']
    evidence = {
        'rule': A3_ANCHOR_RULE,
        'incumbent': {k: v for k, v in incumbent.items() if k != 'record'},
        'incumbent_is_x0': incumbent_is_zero,
        'anchor_label': anchor['label'], 'anchor_stage': anchor['stage'],
        'anchor_candidate_key': anchor['candidate_key'], 'anchor_nodes': anchor['nodes'],
        'anchor_investment_year': anchor['investment_year'], 'anchor_F_eur': anchor['F_eur'],
        'anchor_I_x_eur': anchor['I_x_eur'], 'anchor_Q_gross_eur': anchor['Q_gross_eur'],
        'anchor_selection_evidence': evidence_a1,
        'best_non_zero_over_all_prior_stages': {
            'label': best_all['label'], 'stage': best_all['stage'], 'nodes': best_all['nodes'],
            'F_eur': best_all['F_eur'], 'I_x_eur': best_all['I_x_eur'], 'Q_gross_eur': best_all['Q_gross_eur'],
            'equals_anchor': best_all['label'] == anchor['label'],
            'F_minus_anchor_F_eur': best_all['F_eur'] - anchor['F_eur'],
            'note': ('recorded for scope: if this differs from the anchor, the A1-ladder scope of spec v15 A3 is '
                     'what decided, and the difference is stated here rather than left implicit'),
            'evidence': evidence_all},
        'planner_declared_anchor_label': A3_PLANNER_DECLARED_ANCHOR_LABEL,
        'computed_anchor_equals_planner_declaration': anchor['label'] == A3_PLANNER_DECLARED_ANCHOR_LABEL,
    }
    if anchor['label'] != A3_PLANNER_DECLARED_ANCHOR_LABEL:
        raise SystemExit(f'[S45-A1 REFUSED] stage a3: the computed A3 anchor is {anchor["label"]!r} '
                         f'(F = {anchor["F_eur"]}), not the Planner-declared anchor '
                         f'{A3_PLANNER_DECLARED_ANCHOR_LABEL!r}. {A3_ANCHOR_RULE}')
    return anchor, evidence


def _a3_w16_candidate_context(selected_labels):
    """The A3 section of W16's pinned table: which of its 14 new candidates this stage keeps and which the
    Planner excluded, each with I(x), the budget slack and the feasibility the table records."""
    with open(os.path.join(REPO, INVESTMENT_COST_RESULTS_A2A3['path'])) as handle:
        table = json.load(handle)
    summary = table['A3_resolution_points']['summary']
    cands = table['candidates']
    kept, excluded = [], []
    for label in summary['new_labels']:
        entry = cands.get(label) or {}
        item = {'label': label, 'nodes_nonzero': entry.get('nodes_nonzero'),
                'investment_year': entry.get('investment_year'),
                'I_x_eur': entry.get('I_new_eur'), 'budget_slack_eur': entry.get('budget_slack_B_minus_I_eur'),
                'budget_feasible': entry.get('budget_feasible'),
                'first_stage_feasible_production_check': entry.get('first_stage_feasible_production_check')}
        (kept if label in selected_labels else excluded).append(item)
    return {'source': dict(INVESTMENT_COST_RESULTS_A2A3),
            'w16_rule': summary.get('rule'),
            'n_proposals_considered': summary.get('n_proposals_considered'),
            'n_new_in_w16': summary.get('n_new'), 'n_kept_here': len(kept), 'n_excluded_here': len(excluded),
            'already_evaluated_labels': summary.get('already_evaluated_labels'),
            'kept': kept, 'excluded': excluded,
            'exclusion_reason': ('excluded by Planner decision (task W17): every excluded candidate lies above '
                                 f'3.5 MWh and has I(x) > B = {BUDGET_EUR:g} EUR (budget slack negative), so the '
                                 'budget cannot reach it'),
            'all_excluded_are_budget_infeasible': all(item['budget_feasible'] is False for item in excluded),
            'all_kept_are_budget_feasible': all(item['budget_feasible'] is True for item in kept),
            'sigma_Q_estimate_from_w16': {k: table['sigma_Q_estimate'].get(k) for k in (
                'n_ladders', 'sigma_Q_max_eur', 'sigma_Q_median_eur', 'sigma_Q_min_eur', 'sigma_Q_max_fraction',
                'sigma_Q_median_fraction', 'provisional_fraction', 'ratio_estimate_max_over_provisional',
                'ratio_estimate_median_over_provisional')}}


def build_a3_points(records):
    """The Planner-fixed resolution set at node 7 (A3_SELECTION_RULE), with the incumbent and the anchor
    recorded (A3_ANCHOR_RULE). Every point is checked against the lattice, the duration band and the maximum
    capacity, and refused (not silently dropped) if it fails one: the set is specified, not generated."""
    anchor, anchor_evidence = select_a3_anchor(records)
    points, admissibility = [], []
    for power, energy in A3_POINTS_P_E_MVA_MWH:
        reasons = []
        duration = energy / power
        if not (DURATION_MIN_H - TOL <= duration <= DURATION_MAX_H + TOL):
            reasons.append(f'duration E/P = {duration:.6g} h outside [{DURATION_MIN_H:g}, {DURATION_MAX_H:g}]')
        if energy > MAX_ENERGY_MWH + TOL:
            reasons.append(f'E = {energy:g} > {MAX_ENERGY_MWH:g} MWh (max_capacity)')
        if not _on_lattice(power, LATTICE_P_STEP):
            reasons.append(f'P = {power:g} not on the {LATTICE_P_STEP:g} MVA lattice')
        if not _on_lattice(energy, LATTICE_E_STEP):
            reasons.append(f'E = {energy:g} not on the {LATTICE_E_STEP:g} MWh lattice')
        label = f'res_n{A3_NODE}_p{power:g}_e{energy:g}_y{A3_YEAR}'
        nodes = _nodes_full({A3_NODE: (power, energy)})
        if reasons:
            raise SystemExit(f'[S45-A1 REFUSED] stage a3: the Planner-specified point {label!r} is inadmissible: '
                             f'{reasons}. The a3 set is specified, not generated: a point that fails the master '
                             f'constraints is a specification error, not a drop.')
        admissibility.append({'label': label, 'node': A3_NODE, 'P_mva': power, 'E_mwh': energy,
                              'duration_h': duration, 'investment_year': A3_YEAR,
                              'candidate_key': _key_of(nodes, A3_YEAR), 'admissible': True})
        points.append((label, nodes, A3_YEAR))
    points = tuple(points)
    _refuse_cache_hits('a3', points, records)
    selected = {label for label, _n, _y in points}
    provenance = {
        'rule': A3_SELECTION_RULE,
        'purpose': A3_PURPOSE,
        'anchor': anchor_evidence,
        'anchor_node': next(iter(anchor['nodes'])) if anchor['nodes'] else None,
        'points_node': A3_NODE,
        'points_node_equals_anchor_node': (bool(anchor['nodes']) and next(iter(anchor['nodes'])) == A3_NODE),
        'investment_year': A3_YEAR,
        'points_admissibility': admissibility,
        'w16_a3_candidates': _a3_w16_candidate_context(selected),
        'n_points': len(points),
    }
    if not provenance['points_node_equals_anchor_node']:
        raise SystemExit(f'[S45-A1 REFUSED] stage a3: the fixed points sit at node {A3_NODE} but the anchor '
                         f'{anchor["label"]!r} is at nodes {sorted(anchor["nodes"] or {})}. Planner decision '
                         f'required.')
    return points, provenance


# ======================================================================================================================
#  the stages
# ======================================================================================================================
STAGES = {
    'a1a': {
        'campaign_id': 's45_a1a',
        'description': ('P5.15 Addendum 27 Phase A, A1a -- the 30 single-node 2025 ladders of spec v15 '
                        'A1.ladders_2025, case-file AA-on keep_memory, campaign harness'),
        'prior': (),
        'builder': None,  # fixed points, checked against spec v15
        'points': A1A_POINTS,
        'expected_n': 30,
    },
    'a1b': {
        'campaign_id': 's45_a1b',
        'description': ('P5.15 Addendum 27 Phase A, A1b -- the year ladder (2030, 2035) at the best node, both '
                        'durations, E in {1..5}: 20 points'),
        'prior': ({'stage': 'a1a', 'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_results.json')},),
        'builder': build_a1b_points,
        'points': None,
        'expected_n': 20,
    },
    'a2': {
        'campaign_id': 's45_a2',
        'description': ('P5.15 Addendum 27 Phase A, A2 -- the presence design of the best per-node settings '
                        '(4 new) and the lattice C* (1): 5 points; staging not available'),
        'prior': ({'stage': 'a1a', 'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_results.json')},
                  {'stage': 'a1b', 'path': os.path.join(_P45, 'campaign_s45_a1b', 'campaign_results.json')}),
        'builder': build_a2_points,
        'points': None,
        'expected_n': 5,
    },
    'a3': {
        'campaign_id': 's45_a3',
        'description': ('P5.15 Addendum 27 Phase A, A3 -- the resolution ladder at node 7: the 5 budget-feasible '
                        'lattice points of W16\'s resolution candidates, at the 0.5 MWh / 0.25 MVA step'),
        # a2 is NOT a prior of a3 (Planner task W17): the a3 points are FIXED by the Planner, so they do not
        # depend on a2's outcome and a3 is frozen before a2 runs. The incumbent and the anchor are computed
        # over the committed a0 + a1a + a1b results, which is the whole certified evidence base at freeze time.
        'prior': ({'stage': 'a0', 'path': A0_RESULTS_PATH},
                  {'stage': 'a1a', 'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_results.json')},
                  {'stage': 'a1b', 'path': os.path.join(_P45, 'campaign_s45_a1b', 'campaign_results.json')}),
        'builder': build_a3_points,
        'points': None,
        'expected_n': 5,
    },
}


def campaign_root(stage):
    return os.path.join(REPO, _P45, f'campaign_{STAGES[stage]["campaign_id"]}')


def stage_points(stage, prior_pins=None):
    """(points, provenance) for a stage. a1a is fixed; every later stage recomputes the frozen selection
    rule over the PRIOR stages' committed results (pinned by sha256 when the spec names them)."""
    spec = STAGES[stage]
    if spec['builder'] is None:
        return spec['points'], {'rule': 'spec v15 A1.ladders_2025 (fixed; checked point by point against the spec)',
                                'points_vs_spec_v15': _a1a_points_vs_spec_v15()}
    pins = list(prior_pins if prior_pins is not None else spec['prior'])
    records, evidence, problems = load_prior_records(pins)
    if problems:
        raise SystemExit('[S45-A1 REFUSED] prior results unusable:\n  ' + '\n  '.join(problems))
    points, provenance = spec['builder'](records)
    provenance['prior_results'] = evidence
    provenance['prior_stages'] = [pin['stage'] for pin in pins]
    return points, provenance


# ======================================================================================================================
#  checks shared by --freeze and --run (zero solves)
# ======================================================================================================================
def _check_case_file_loads_to_declaration():
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded = params.admm.anderson_acceleration
    return {'loaded': loaded, 'equals_declaration': loaded == CASE_FILE_AA}


def _check_pinned_files():
    out = {}
    pins = [('spec_v15', SPEC_V15), ('cost_file', COST_FILE),
            ('investment_cost_results', INVESTMENT_COST_RESULTS)]
    pins += [(f'investment_cost_source_{source["name"]}', source) for source in I_X_SOURCES]
    pins.append(('a0_spec', A0_SPEC))
    for name, pin in pins:
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256']}
    return out


def _memory_rule_matches_a0():
    """The memory measure is A0's; the rule text is compared with the one recorded in A0's frozen spec so
    that the two launchers cannot drift apart silently."""
    path = os.path.join(REPO, A0_SPEC['path'])
    if not os.path.isfile(path):
        return {'a0_spec_present': False, 'match': False}
    with open(path) as handle:
        extra = json.load(handle)['extra']
    return {'a0_spec_present': True, 'a0_rule': extra.get('memory_preflight_rule'), 'rule': MEMORY_RULE,
            'a0_rationale_matches': extra.get('memory_preflight_rule_rationale') == MEMORY_RULE_RATIONALE,
            'match': extra.get('memory_preflight_rule') == MEMORY_RULE}


def _load_i_x_sources():
    """The pinned I(x) tables, in order. Each entry: (source, {label: candidate record})."""
    loaded = []
    for source in I_X_SOURCES:
        with open(os.path.join(REPO, source['path'])) as handle:
            loaded.append((source, json.load(handle)['candidates']))
    return loaded


def investment_costs(points):
    """I(x) per point from the ORDERED list of pinned I(x) tables (I_X_SOURCES), BY CANDIDATE KEY (candidate
    AND investment year). Every point must be found in at least one source; within a source every entry
    carrying that key must agree on I; ACROSS sources a key held by more than one must agree EXACTLY on I(x)
    (otherwise the freeze fails); the first source holding the key supplies the value, and the source used is
    recorded. The budget slack the source records at B = 1e6 must equal B - I. Returns (per_point, problems)."""
    sources = _load_i_x_sources()
    per_point, problems = {}, []
    for label, nodes, year in points:
        key = _key_of(nodes, year)
        found = []
        for source, cands in sources:
            hits = {name: c for name, c in cands.items() if c.get('candidate_key') == key}
            if not hits:
                continue
            fields = source['fields']
            values = sorted({c.get(fields['I_x_eur']) for c in hits.values()}, key=repr)
            if len(values) != 1 or values[0] is None:
                problems.append(f'{label}: entries with key {key[:16]} in {source["name"]} disagree on / lack '
                                f'{fields["I_x_eur"]}: {values}')
                continue
            entry = hits[sorted(hits)[0]]
            found.append({'source': source['name'], 'source_path': source['path'], 'source_order': source['order'],
                          'matched_entries': sorted(hits), 'I_x_eur': values[0],
                          '_entry': entry, '_fields': fields})
        if not found:
            problems.append(f'{label}: candidate key {key[:16]} (year {year}) not found in any pinned I(x) source '
                            f'{[s["path"] for s, _c in sources]}; I(x) for this candidate must be computed and '
                            f'committed first (p515_s45_investment_cost_recompute.py / '
                            f'p515_s45_investment_cost_a2a3.py, zero solves)')
            per_point[label] = {'candidate_key': key, 'investment_year': year, 'found': False,
                                'sources_searched': [s['path'] for s, _c in sources]}
            continue
        per_source = [{k: v for k, v in item.items() if not k.startswith('_')} for item in found]
        distinct = sorted({item['I_x_eur'] for item in found})
        if len(distinct) > 1:
            problems.append(f'{label}: candidate key {key[:16]} is held by {len(found)} pinned I(x) sources which '
                            f'DISAGREE on I(x): {[(i["source"], i["I_x_eur"]) for i in per_source]}')
        chosen = found[0]
        entry, fields = chosen['_entry'], chosen['_fields']
        i_x = chosen['I_x_eur'] if len(distinct) == 1 else None
        slack = (BUDGET_EUR - i_x) if i_x is not None else None
        file_slack = entry.get(fields['budget_slack_eur'])
        if slack is not None and file_slack is not None and abs(slack - file_slack) > 1e-6:
            problems.append(f'{label}: budget slack B - I = {slack} != {chosen["source"]}\'s '
                            f'{fields["budget_slack_eur"]} {file_slack}')
        per_point[label] = {
            'candidate_key': key, 'investment_year': year, 'found': True,
            'I_x_source': chosen['source'], 'I_x_source_path': chosen['source_path'],
            'I_x_sources_holding_this_key': per_source,
            'I_x_sources_agree': len(distinct) == 1,
            'matched_entries': chosen['matched_entries'],
            'I_x_eur': i_x, 'I_power_eur': entry.get(fields['I_power_eur']),
            'I_energy_eur': entry.get(fields['I_energy_eur']),
            'budget_slack_eur': slack, 'budget_slack_in_cost_file_eur': file_slack,
            'budget_feasible_corrected_file': entry.get(fields['budget_feasible']),
            'first_stage_feasible_production_check': entry.get(fields['first_stage_feasible']),
            'nodes': {str(n): list(v) for n, v in nodes.items()},
            'source_rule': I_X_SOURCE_RULE,
            'note': BUDGET_CONVENTION}
    return per_point, problems


_VM_STAT_KEYS = {'pages_free': 'Pages free', 'pages_active': 'Pages active', 'pages_inactive': 'Pages inactive',
                 'pages_speculative': 'Pages speculative', 'pages_wired_down': 'Pages wired down',
                 'pages_purgeable': 'Pages purgeable', 'file_backed_pages': 'File-backed pages',
                 'anonymous_pages': 'Anonymous pages', 'pages_stored_in_compressor': 'Pages stored in compressor',
                 'pages_occupied_by_compressor': 'Pages occupied by compressor'}
_VM_STAT_REQUIRED = ('pages_free', 'pages_inactive', 'pages_wired_down', 'anonymous_pages',
                     'pages_occupied_by_compressor')


def memory_preflight():
    """available = hw.memsize - (wired + anonymous + compressor-occupied) x page size, against
    CONCURRENCY x 2.75 GiB -- A0's measure (Planner decision W6), whose rule text is checked against A0's
    frozen spec. The non-reclaimable load on macOS is wired memory, anonymous (process) memory and the
    pages the compressor occupies; file-backed pages, active or inactive, are cache the kernel reclaims on
    demand, so free + inactive under-counts what new processes can obtain. free + inactive is recorded
    alongside; the page size is read from vm_stat's header; every vm_stat figure used is recorded. A figure missing from vm_stat's output makes the check fail (never a silent zero)."""
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
    out.update({
        'wired_bytes': pages['pages_wired_down'] * page, 'anonymous_bytes': pages['anonymous_pages'] * page,
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
            f"compressor-occupied {memory['compressor_occupied_bytes'] / GIB:.2f}, memsize "
            f"{memory['hw_memsize_bytes'] / GIB:.2f}]; free+inactive = {memory['free_plus_inactive_gib']:.2f} GiB "
            f"(recorded only); required {memory['required_gib']:.2f} GiB ({memory['rule']})")


# ======================================================================================================================
#  rule eleven: a capture path for every per-point field, asserted BEFORE anything runs
# ======================================================================================================================
CAMPAIGN_RESULT_FIELDS = (
    'status', 'certification_cycle', 'certified_cost_gross_settlement_excluded', 'bar', 'rule_ten',
    'terminal_salvage_value', 'net_operational_recourse', 'I_x_eur', 'F_eur', 'budget_slack_eur',
    'investment_year', 'wall_time', 'peak_rss', 'aa_action_counts', 'per_cycle_trajectory')


def rule_eleven():
    """Before any evaluation: a capture path exists for every per-point campaign_results field."""
    harness_checks = H.assert_record_capture_paths()  # the harness's own rule-eleven assertion (no solve)
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    import shared_resources_planning as srp
    rc_src = inspect.getsource(srp._get_operational_recourse_components)
    checks = {
        'status': "'status': status" in build_src,
        'certification_cycle': "'certification_cycle':" in build_src,
        'certified_cost_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'bar_gross_step': "'bar': bar" in build_src and 'gross_step_abs' in inspect.getsource(H._max_step_last_n),
        'bar_net_reported': "'bar_net_recourse_step_reported':" in build_src,
        'rule_ten': "'rule_ten':" in build_src and "'terminal_step_over_threshold':" in build_src,
        'terminal_salvage_value_in_recourse_components': ("'recourse_components': rc" in build_src
                                                          and "'terminal_salvage_value':" in rc_src),
        'net_operational_recourse': ("'terminal_net_operational_recourse': rc.get('net_operational_recourse')"
                                     in build_src and "'net_operational_recourse':" in rc_src),
        'investment_year_in_canonical': ("'investment_year': int(investment_year)"
                                         in inspect.getsource(H.canonical_candidate)
                                         and "'candidate_canonical': entry['canonical']" in build_src
                                         and 'investment_year' in H.EVALUATION_OPTION_KEYS),
        'I_x_and_budget_slack_frozen_in_spec': True,  # asserted point by point by _validate_spec on the spec itself
        'wall_time': "'wall_time_s': wall" in build_src,
        'peak_rss': "'peak_rss': peak_rss" in build_src,
        'aa_action_counts': ("'aa_per_cycle': holder.get('aa_sidecar')" in child_src
                             and "'action_counts'" in inspect.getsource(H.aa_sidecar_summary)),
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'per_cycle_fields': all(f in H.PER_CYCLE_RECORD_FIELDS for f in (
            'cycle', 'gross_operational_cost', 'terminal_salvage_value', 'objective_change_abs',
            'objective_tolerance', 'consecutive_converged_cycles', 'boyd_all_pass', 'local_solves_ok')),
        'error_records_same_schema': ("'anderson_acceleration_effective_in_child':" in inspect.getsource(H.main_child)
                                      and "'anderson_acceleration_effective_in_child':"
                                      in inspect.getsource(H._barrier_record_for_missing)),
        'stop_rule_inputs': ("'status':" in build_src and "'barrier_cause': cause" in build_src),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (A1 campaign_results): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': harness_checks}


# ======================================================================================================================
#  spec validation
# ======================================================================================================================
def _validate_spec(stage, spec, points, i_x):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    stage_spec = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == stage_spec['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == [label for label, _n, _y in points],
        'declaration': spec['configuration'].get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'no_campaign_overrides': spec['configuration'].get('overrides') == {},
        'cap': spec.get('cap') == CAP,
        'concurrency_7': spec.get('concurrency') == CONCURRENCY == 7,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': spec['configuration'].get('arm_label') == 's39_D',
        'not_a_stub_spec': not spec.get('extra', {}).get('test_only_stub'),
        'stage_recorded': spec['extra'].get('stage') == stage,
        'spec_v15_recorded': spec['extra'].get('spec_v15') == SPEC_V15,
        'cost_file_recorded': (spec['extra'].get('cost_file') or {}).get('sha256') == COST_FILE['sha256'],
        'investment_cost_results_recorded': spec['extra'].get('investment_cost_results') == INVESTMENT_COST_RESULTS,
        'investment_cost_sources_recorded': (spec['extra'].get('investment_cost_sources') == [dict(s) for s in
                                                                                             I_X_SOURCES]
                                             and spec['extra'].get('investment_cost_source_rule')
                                             == I_X_SOURCE_RULE),
        'memory_rule_recorded': spec['extra'].get('memory_preflight_rule') == MEMORY_RULE,
        'stop_rule_recorded': spec['extra'].get('stop_rule') == STOP_RULE,
        'launch_plan_recorded': (spec['extra'].get('launch_plan') == LAUNCH_PLAN
                                 and spec['extra'].get('wave_size') == CONCURRENCY),
        'objective_convention_recorded': spec['extra'].get('objective_convention') == OBJECTIVE_CONVENTION,
        'budget_convention_recorded': spec['extra'].get('budget_convention') == BUDGET_CONVENTION,
        'expected_n': (stage_spec['expected_n'] is None or len(entries) == stage_spec['expected_n']),
        'n_entries_equals_n_points': len(entries) == len(points),
    }
    for label, nodes, year in points:
        entry = by_label.get(label) or {}
        canon = _canonical_of(nodes, year)
        key = H.candidate_key(canon)
        recorded = (spec['extra'].get('points') or {}).get(label) or {}
        checks[f'{label}:canonical_key'] = entry.get('canonical') == canon and entry.get('key') == key
        checks[f'{label}:investment_year'] = (entry.get('canonical') or {}).get('investment_year') == year
        checks[f'{label}:no_overrides_no_post_certification'] = (entry.get('overrides') == {}
                                                                 and entry.get('post_certification') is None)
        checks[f'{label}:effective_aa_is_declaration'] = entry.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:eval_key_recomputes'] = entry.get('eval_key') == H.evaluation_key(key, {},
                                                                                           case_file_aa=CASE_FILE_AA)
        point_i = (i_x.get(label) or {})
        checks[f'{label}:I_x_frozen'] = (recorded.get('candidate_key') == key
                                         and recorded.get('I_x_eur') is not None
                                         and recorded.get('I_x_eur') == point_i.get('I_x_eur'))
        checks[f'{label}:budget_slack_frozen'] = (recorded.get('budget_slack_eur') is not None
                                                  and recorded.get('budget_slack_eur')
                                                  == point_i.get('budget_slack_eur'))
        checks[f'{label}:I_x_source_frozen'] = (recorded.get('I_x_source') in
                                                {source['name'] for source in I_X_SOURCES}
                                                and recorded.get('I_x_source') == point_i.get('I_x_source')
                                                and recorded.get('I_x_sources_agree') is True)
    return checks


def _common_checks(stage, prior_pins=None):
    """Everything checked identically at --freeze and --run; returns (failures, evidence, points, i_x)."""
    failures, evidence = [], {}
    case_file = _check_case_file_loads_to_declaration()
    evidence['case_file_aa'] = case_file
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    pinned = _check_pinned_files()
    evidence['pinned_files'] = pinned
    failures += [f'{k}: sha256 on disk {v["sha256_on_disk"]} != pinned {v["sha256_pinned"]}'
                 for k, v in pinned.items() if not v['match']]
    memory_rule = _memory_rule_matches_a0()
    evidence['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f'memory preflight rule differs from A0\'s frozen spec: {memory_rule}')
    points, provenance = stage_points(stage, prior_pins=prior_pins)
    evidence['points_provenance'] = provenance
    evidence['points'] = [{'label': label, 'nodes': {str(n): list(v) for n, v in nodes.items()},
                           'investment_year': year, 'candidate_key': _key_of(nodes, year)}
                          for label, nodes, year in points]
    if stage == 'a1a':
        vs_spec = provenance['points_vs_spec_v15']
        if not (vs_spec['equal_to_spec_v15_A1_ladders_2025_in_order'] and vs_spec['n_matches_spec_n']
                and vs_spec['all_points_year_2025']):
            failures.append(f'a1a points differ from spec v15 A1.ladders_2025: {vs_spec}')
    expected = STAGES[stage]['expected_n']
    if expected is not None and len(points) != expected:
        failures.append(f'stage {stage}: {len(points)} points, expected {expected}')
    labels = [label for label, _n, _y in points]
    if len(set(labels)) != len(labels):
        failures.append(f'duplicate labels in stage {stage}: {labels}')
    keys = [_key_of(nodes, year) for _l, nodes, year in points]
    if len(set(keys)) != len(keys):
        failures.append(f'duplicate candidate keys in stage {stage}')
    i_x, problems = investment_costs(points)
    evidence['investment_cost_per_point'] = i_x
    failures += problems
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, points, i_x


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(stage, started):
    root = campaign_root(stage)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, points, i_x = _common_checks(stage)
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[S45-{stage.upper()} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    recorded_points = {label: dict(i_x[label]) for label, _n, _y in points}
    prior_pins = [{'stage': item['stage'], 'path': item['path'],
                   'sha256': H.sha256_file(os.path.join(REPO, item['path']))}
                  for item in STAGES[stage]['prior']]
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, STAGES[stage]['campaign_id'],
        [(label, nodes, {'investment_year': year}) for label, nodes, year in points],
        configuration={'name': 'case file alone: AA keep_memory adopted in data/SRP1/SRP1_params.json (Addendum 27)',
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'note': ('no overrides; the case file carries AA (declared, checked by the child hook); '
                                'num_max_iters := cap (run_admm_arm always sets it)')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'mode': 'full', 'phase': 'A', 'stage': stage,
               'stage_description': STAGES[stage]['description'],
               'spec_v15': dict(SPEC_V15),
               'cost_file': dict(COST_FILE,
                                 sha256_on_disk_at_freeze=evidence['pinned_files']['cost_file']['sha256_on_disk']),
               'investment_cost_results': dict(INVESTMENT_COST_RESULTS),
               'investment_cost_sources': [dict(source) for source in I_X_SOURCES],
               'investment_cost_source_rule': I_X_SOURCE_RULE,
               'a0_spec': dict(A0_SPEC),
               'points': recorded_points,
               'points_provenance': evidence['points_provenance'],
               'prior_results_pinned': prior_pins,
               'post_certification': 'none (spec v15 execution.hull_polish: on incumbents only; not in A1-A3)',
               'stop_rule': STOP_RULE,
               'objective_convention': OBJECTIVE_CONVENTION,
               'budget_convention': BUDGET_CONVENTION,
               'staging': STAGING_NOT_AVAILABLE,
               'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                  'gross_operational_cost[k-1]| (harness W5); record.bar_net_recourse_step_reported '
                                  '= the net-recourse objective_change_abs max (reported)'),
               'memory_preflight_rule': MEMORY_RULE,
               'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
               'memory_preflight_refusing_at': '--run (non-gating at --freeze)',
               'memory_at_freeze_non_gating': memory,
               'launch_plan': LAUNCH_PLAN,
               'wave_size': CONCURRENCY,
               'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)})
    checks = _validate_spec(stage, spec, points, i_x)
    guard_failures = PARENT_GUARD.verify(0)
    tag = f'S45-{stage.upper()}'
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for entry in spec['candidates']:
        point = recorded_points[entry['label']]
        holders = [(item['source'], item['I_x_eur']) for item in point.get('I_x_sources_holding_this_key', [])]
        _log(f"[{tag}]   {entry['label']}: year={point['investment_year']} nodes={point['nodes']} "
             f"key={entry['key'][:16]} eval_key={entry['eval_key'][:16]} eval_dir={entry['eval_dir']} "
             f"I(x)={point['I_x_eur']} budget_slack={point['budget_slack_eur']} "
             f"I_source={point.get('I_x_source')} I_sources_holding_key={holders}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] case file AA (production loader): {evidence['case_file_aa']}")
    _log(f"[{tag}] pinned files: {evidence['pinned_files']}")
    _log(f"[{tag}] memory rule vs A0 spec: {evidence['memory_rule_vs_a0_spec']}")
    _log(f"[{tag}] points provenance: {json.dumps(evidence['points_provenance'], default=str)}")
    _log(f"[{tag}] prior results pinned: {prior_pins}")
    _log(f"[{tag}] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[{tag}] stop rule: {STOP_RULE}")
    _log(f"[{tag}] objective convention: {OBJECTIVE_CONVENTION}")
    _log(f"[{tag}] budget convention: {BUDGET_CONVENTION}")
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now; vm_stat pages {memory['vm_stat_pages']}")
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


def _point_result(label, rec, spec_point):
    rec = rec or {}
    traj, rows = _per_cycle_trajectory(rec)
    certified = rec.get('status') == 'certified'
    q_gross = rec.get('certified_cost') if certified else None
    i_x = spec_point.get('I_x_eur')
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  and prev.get('cycle') == (last.get('cycle') or 0) - 1 else None)
    tol = last.get('objective_tolerance')
    rule_ten = rec.get('rule_ten') or {}
    canonical = rec.get('candidate_canonical')
    return {
        'label': label, 'candidate_key': rec.get('candidate_key'), 'candidate_canonical': canonical,
        'investment_year': (canonical or {}).get('investment_year', spec_point.get('investment_year')),
        'region': _region_key(canonical) if canonical else None,
        'eval_key': rec.get('eval_key'), 'eval_dir': rec.get('eval_dir'),
        'status': rec.get('status'), 'barrier': rec.get('barrier'), 'barrier_cause': rec.get('barrier_cause'),
        'certification_cycle': rec.get('certification_cycle'), 'cycles_run': rec.get('cycles_run'),
        'certified_cost_gross_settlement_excluded': q_gross,
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'bar': {'value': (rec.get('bar') or {}).get('value'), 'definition': (rec.get('bar') or {}).get('definition'),
                'n_steps_available': (rec.get('bar') or {}).get('n_steps_available')},
        'bar_net_recourse_step_reported': (rec.get('bar_net_recourse_step_reported') or {}).get('value'),
        'rule_ten': {
            'terminal_step_over_threshold_production': rule_ten.get('terminal_step_over_threshold'),
            'terminal_objective_change_abs_production_net': rule_ten.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': rule_ten.get('terminal_objective_tolerance'),
            'terminal_gross_step_abs': gross_step,
            'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol) else None,
            'boyd_terminal_ratio_max_per_channel': rule_ten.get('boyd_terminal_ratio_max_per_channel'),
            'note': ('production ratio = the net-recourse step the stopping test used / its tolerance; the gross '
                     'version is computed here from the per-cycle record (terminal row vs its predecessor)')},
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'I_x_eur': i_x, 'F_eur': (i_x + q_gross) if (q_gross is not None and i_x is not None) else None,
        'budget_slack_eur': spec_point.get('budget_slack_eur'),
        'budget_feasible_corrected_file': spec_point.get('budget_feasible_corrected_file'),
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': {'record': rec.get('peak_rss'),
                     'parent_wait4_ru_maxrss_bytes': (rec.get('parent_view') or {}).get('wait4_ru_maxrss')},
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'aa_per_cycle': rec.get('aa_per_cycle'),
        'first_pass_cycle_per_channel': rec.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': rec.get('terminal_ratios_per_channel'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'anderson_acceleration_effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'case_file_sha256_in_child': rec.get('case_file_sha256_in_child'),
        'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
        'per_cycle_trajectory': traj,
    }


def stop_rule_state(records):
    """The operational stop rule (see STOP_RULE): (a) 2+ non-certified in one region, (b) 3+ overall."""
    non_certified = [r for r in records if (r or {}).get('status') != 'certified']
    regions = Counter(_region_key(r['candidate_canonical']) for r in non_certified
                      if (r or {}).get('candidate_canonical'))
    offending = sorted(region for region, count in regions.items() if count >= 2)
    reasons = []
    if offending:
        reasons.append(f'(a) 2 or more non-certified points in one region: {offending} '
                       f'({dict(regions)})')
    if len(non_certified) >= 3:
        reasons.append(f'(b) 3 or more non-certified points overall: '
                       f'{[r.get("candidate_label") for r in non_certified]}')
    return {'triggered': bool(reasons), 'reasons': reasons,
            'non_certified_labels': [r.get('candidate_label') for r in non_certified],
            'non_certified_by_region': dict(regions)}


def run(stage, started, spec_sha256):
    tag = f'S45-{stage.upper()}'
    if spec_sha256 in FOREIGN_SPEC_SHA256:
        _log(f'[{tag} PRECONDITION FAILED] {spec_sha256} is {FOREIGN_SPEC_SHA256[spec_sha256]}, not a spec of '
             f'this stage')
        raise SystemExit(1)
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    root_contents = sorted(os.listdir(root))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    if spec.get('campaign_id') != STAGES[stage]['campaign_id']:
        failures.append(f'spec campaign_id {spec.get("campaign_id")} != stage {stage} '
                        f'({STAGES[stage]["campaign_id"]})')
    # the later stages' points are RECOMPUTED from the prior results the spec pinned (same sha256), so the
    # frozen selection is reproducible at run time, not merely asserted.
    prior_pins = spec['extra'].get('prior_results_pinned') or []
    if [p['path'] for p in prior_pins] != [p['path'] for p in STAGES[stage]['prior']]:
        failures.append(f'spec prior_results_pinned {[p["path"] for p in prior_pins]} != stage definition '
                        f'{[p["path"] for p in STAGES[stage]["prior"]]}')
        prior_pins = list(STAGES[stage]['prior'])
    more, evidence, points, i_x = _common_checks(stage, prior_pins=prior_pins or None)
    failures += more
    checks = _validate_spec(stage, spec, points, i_x)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
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
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    _log(f'[{tag}] {len(points)} points, waves of {CONCURRENCY}; stop rule: {STOP_RULE}')
    labels = [label for label, _n, _y in points]
    waves = [labels[i:i + CONCURRENCY] for i in range(0, len(labels), CONCURRENCY)]
    lock = H.acquire_campaign_lock(STAGES[stage]['campaign_id'], spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    records, wave_info, not_launched = [], [], []
    stop_state = {'triggered': False, 'reasons': [], 'non_certified_labels': [], 'non_certified_by_region': {}}
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        for number, wave in enumerate(waves, start=1):
            stop_state = stop_rule_state(records)
            if stop_state['triggered']:
                not_launched += wave
                _log(f'[{tag}] STOP RULE: not launching wave {number} {wave}; {stop_state["reasons"]}')
                wave_info.append({'wave': number, 'labels': wave, 'launched': False,
                                  'reason': f'stop rule: {stop_state["reasons"]}'})
                continue
            _log(f'[{tag}] launching wave {number}/{len(waves)}: {wave} (concurrency {ctx.concurrency})')
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
    point_results = {}
    for label in labels:
        spec_point = spec['extra']['points'][label]
        if label in not_launched:
            point_results[label] = {'label': label, 'status': 'not_launched_stop_rule',
                                    'investment_year': spec_point.get('investment_year'),
                                    'I_x_eur': spec_point.get('I_x_eur'),
                                    'budget_slack_eur': spec_point.get('budget_slack_eur'), 'F_eur': None}
        else:
            point_results[label] = _point_result(label, by_label.get(label), spec_point)
    launched = [label for label in labels if label not in not_launched]
    non_certified = [label for label in launched if point_results[label]['status'] != 'certified']
    harness_errors = [label for label in launched
                      if point_results[label]['status'] not in ('certified', 'not_certified')
                      or point_results[label]['exit_code'] != 0]
    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'STOP_FOR_REVIEW': stop_state['triggered'],
        'stop_rule_state': stop_state,
        'non_certified_points': non_certified,
        'not_launched_points': not_launched,
        'stop_rule': STOP_RULE,
        'launch_plan': LAUNCH_PLAN,
        'stage': STAGES[stage]['description'],
        'stage_id': stage,
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION,
        'budget_convention': BUDGET_CONVENTION,
        'points': point_results,
        'points_provenance': evidence['points_provenance'],
        'prior_results_pinned': prior_pins,
        'harness_errors': harness_errors,
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pinned_files', 'memory_rule_vs_a0_spec',
                                                      'investment_cost_per_point')},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'wave_info': wave_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    for label, point in point_results.items():
        if label in not_launched:
            _log(f"[{tag}] {label}: status={point['status']} (stop rule; never launched)")
            continue
        _log(f"[{tag}] {label}: status={point['status']} cycles={point['cycles_run']} "
             f"cert={point['certification_cycle']} Q={point['certified_cost_gross_settlement_excluded']} "
             f"I={point['I_x_eur']} F={point['F_eur']} slack={point['budget_slack_eur']} "
             f"bar={point['bar']['value']} rule10={point['rule_ten']['terminal_step_over_threshold_production']} "
             f"salvage={point['terminal_salvage_value']} net={point['net_operational_recourse']} "
             f"aa={point['aa_action_counts']}")
    _log(f"[{tag}] objective convention: {OBJECTIVE_CONVENTION}")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    if non_certified:
        _log(f'[{tag}] non-certified (barrier) points: {non_certified}')
    if harness_errors:
        _log(f'[{tag}] evaluations with an error status or non-zero exit: {harness_errors}')
    if stop_state['triggered']:
        _banner([f'STOP_FOR_REVIEW: the stop rule fired: {stop_state["reasons"]}',
                 f'not launched: {not_launched}',
                 'Nothing further is launched. See campaign_results.json.'])
    if guard_failures or harness_errors:
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    if stop_state['triggered']:
        sys.exit(3)
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
