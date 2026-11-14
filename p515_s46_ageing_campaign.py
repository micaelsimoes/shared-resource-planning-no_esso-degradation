"""
P5.15 Addenda 28-29 (task W20, item 5) -- THE AGEING BATCH: five MODEL VARIANTS of
the shared-ESS ageing law at the smallest node-7 4 h unit, through the campaign
harness (`p515_s44_campaign_harness.py`, `model_variant`). Launcher pattern of
`p515_s45_a1_campaign.py`: preconditions, locks, clean git, one spec per root,
memory preflight, cap 500, 10 consecutive all-pass cycles, case-file AA
declaration, NO AA overrides, NO post-certification.

  MODEL VARIANT -- not the baseline. Every evaluation of this campaign is a model
  variant; its values are never mixed with the baseline (C3) in a table.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 28-29; frozen spec v16
`data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json` (`ageing_batch`,
`concurrency`); Planner task W20 item 5.

THE INSTANCE. One candidate for every evaluation: node 7 at 0.25 MVA / 1.0 MWh (4 h),
investment year 2025, nodes 5 and 9 zero -- label n7_4h_e1, candidate key
db77e1549af855bc... (recomputed and checked against spec v16 `ageing_batch.candidate`).
THE FIVE EVALUATIONS (soh_min 0.50 in all; `VARIANTS`):
  C2           eol_retention_r 0.80 -> k = 35,851
  C4           eol_retention_r 0.70 -> k = 22,429
  C2_calfade   C2 with calendar_retention_per_year 0.985
  C3_midblock  C3 (eol 0.50, k = 11,541.56) with available energy at the MID-block SoH
               SoH_mid_y = SoH_{y-1} exp(-D_y/2) phi**(n/2) (shared_energy_storage_data.
               _esso_ageing_model_settings); end-of-block SoH still drives the next block,
               the floor and the salvage
  no_ageing    SoH identically 1 (D row D == 0, phi 1)
Each is applied by the child's configuration hook to that evaluation's own shared-ESS
data, read back from the built ESSO models (pre-run probes: refuse on mismatch; post-run
clones: recorded), and enters the eval key.

x = 0 IS NOT RE-RUN: Q(0) is ageing-independent (no storage). It is taken from the A0
record (`A0_RESULTS`, pinned by sha256): x0 certified_cost_gross 653,859,461.2279255,
bar 9,629.978. The baseline value at the unit, value_C3 = Q(0) - Q_C3(x) with Q_C3 from
the A1a record (`A1A_RESULTS`, pinned): 261,807.504426 EUR (spec v16 records 261,807.5).
The zero-solve gate of cb165d4e showed the case-file edit leaves every oracle-path block
at x = 0 and C* unchanged, and the two-cycle gate of 8ed12e30 reproduced the committed
C* trajectory bitwise under the current code, so the pinned A0 / A1a values remain the
baseline this batch is compared with.

RECORDED PER VARIANT (campaign_results.json `points`; objective convention on every
table: Q = certified GROSS operational cost, settlement excluded; salvage reported,
excluded): status (non-certified variants reported with cause), cycles, certification
cycle, Q, value = Q(0) - Q, I(x), value - I, value / value_C3, the bar (max gross step
over the last 10 cycles) and x0's bar and their sum, the terminal gross step and its
ratio to the threshold (rule ten), the resolution verdict (|value - I| against the bar
sum and against spec v16's stated sigma_Q range), k / phi / SoH-point mode AS READ BACK
from the built models (pre-run probes and post-run clones), EFC/day and the SoH
trajectory per block (end-of-block SoH and the SoH used for available energy), the
terminal salvage (recourse components and per node), the predictions recorded in spec
v16 beside the outcome, wall time, peak RSS, the AA action counts, the per-cycle
trajectory path (+ sha256). Rule eleven: a capture path for every one of those fields
is asserted BEFORE anything is launched, in both modes.

MEMORY PREFLIGHT (refusing at --run; recorded, non-gating, at --freeze): A0's measure
(available = hw.memsize - (wired + anonymous + compressor-occupied) pages x page size),
required >= CONCURRENCY (5, spec v16 `concurrency.srp1`) x 2.75 GiB per child. The rule
TEMPLATE is checked against the rule text recorded in A0's frozen spec (with A0's
concurrency 7), and the rationale text must be identical, so the measure cannot drift.

Two modes, both attached, both streams captured, never detached:
  --freeze                      ZERO SOLVES: preconditions, pins, the five points, I(x),
                                Q(0), value_C3, rule eleven, then `freeze_campaign_spec`
                                + validation. Prints the --run command.
  --run --spec-sha256 <sha>     loads THAT spec from the campaign root (which must hold
                                only it), re-checks everything above plus the harness /
                                case-file / ESS-params / script sha256 recorded in the
                                spec, the memory preflight (refusing), takes the campaign
                                lock, evaluates the five in ONE batch of 5, writes
                                campaign_results.json and campaign_manifest_sha256.json.
The parent never solves: SolveProfileGuard(permitted=()) is installed before any model
import and verified at exactly 0 in both modes.
Exit codes (--run): 0 harness clean (the batch always ends in STOP FOR REVIEW, spec v16
`ageing_batch.stop`, recorded as STOP_FOR_REVIEW true); 1 harness / guard / precondition
failure.

EXACT COMMANDS (repo root, canonical interpreter):
  freeze (zero solves):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_ageing_campaign.py --freeze \\
        > data/SRP1/Results/P515S46/campaign_s46_ageing_freeze_launch.log 2>&1
  run (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_ageing_campaign.py \\
        --run --spec-sha256 <sha256 printed by --freeze> \\
        > data/SRP1/Results/P515S46/campaign_s46_ageing_launch.log 2>&1
"""

import argparse
import inspect
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S46 ageing campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

CAMPAIGN_ID = 's46_ageing'
_P46 = os.path.join('data', 'SRP1', 'Results', 'P515S46')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CAMPAIGN_ROOT = os.path.join(REPO, _P46, f'campaign_{CAMPAIGN_ID}')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
TOL = 1e-9

SPEC_V16 = {'path': os.path.join(_P46, 'frozen_s46_ageing_spec_v16_f4295086.json'),
            'sha256': 'f4295086c10b22c7b33f4e7eff0bd38f7c93c09aeefc12117175633e8efc41f0'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
ESS_PARAMS_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
                   'sha256': 'fdce321ffe1bf0f4b81424cdf081d826c3c81c410a4eb9f3b3561675e26b80f0',
                   'note': 'after the W20 case-file edit (cb165d4e): max_energy_to_power_factor 4'}
A0_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_spec_s45_a0_c7_9d08ad2f.json'),
           'sha256': '9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34'}
A0_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json'),
              'sha256': '423678b98c0ca9a26dbc76a1b46a5e36a51c8ccccc0459253020bfe1061653b4', 'label': 'x0'}
A1A_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_results.json'),
               'sha256': 'b6e9adce6fe33878082ce78a92c719c39c674c72b80930eb02b36d2d2c576aec', 'label': 'n7_4h_e1'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'field': 'candidates.<label>.I_new_eur, matched by candidate_key'}
VARIANT_CHECKS = {'path': os.path.join(_P46, 'variant_checks', 'variant_checks.json'), 'commit': 'e5721b6d'}
CASE_FILE_EDIT_GATE = {'path': os.path.join(_P46, 'case_file_edit', 'after', 'case_file_edit_after.json'),
                       'commit': 'cb165d4e'}
TWO_CYCLE_GATE = {'path': os.path.join(_P46, 'default_two_cycle_gate', 'gate.json'), 'commit': '8ed12e30'}
PINNED_EVIDENCE = (('variant_checks', VARIANT_CHECKS), ('case_file_edit_gate', CASE_FILE_EDIT_GATE),
                   ('two_cycle_gate', TWO_CYCLE_GATE))

UNIT_LABEL = 'n7_4h_e1'
UNIT_NODES = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
UNIT_YEAR = 2025
VALUE_C3_PINNED = 261807.504426      # Planner task W20, from A1a (spec v16 records it rounded, 261807.5)
VARIANTS = (
    ('C2', {'eol_retention_r': 0.80, 'calendar_retention_per_year': 1.0,
            'available_energy_soh_point': 'end', 'ageing_enabled': True}),
    ('C4', {'eol_retention_r': 0.70, 'calendar_retention_per_year': 1.0,
            'available_energy_soh_point': 'end', 'ageing_enabled': True}),
    ('C2_calfade', {'eol_retention_r': 0.80, 'calendar_retention_per_year': 0.985,
                    'available_energy_soh_point': 'end', 'ageing_enabled': True}),
    ('C3_midblock', {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,
                     'available_energy_soh_point': 'mid', 'ageing_enabled': True}),
    ('no_ageing', {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,
                   'available_energy_soh_point': 'end', 'ageing_enabled': False}),
)
VARIANT_DESCRIPTIONS = {
    'C2': 'EOL retention 0.80 (cycles 10000, DoD 0.80) -> k = 35,851; soh_min 0.50; phi_cal 1.0',
    'C4': 'EOL retention 0.70 -> k = 22,429; soh_min 0.50; phi_cal 1.0',
    'C2_calfade': 'C2 with calendar fade phi_cal = 0.985 per year',
    'C3_midblock': ('C3 (k = 11,541.56) with available energy in block y at the MID-block SoH '
                    'SoH_{y-1} exp(-D_y/2) phi^(n/2); end-of-block SoH propagates to the next block, the floor '
                    'and the salvage'),
    'no_ageing': 'SoH identically 1 (no cycle or calendar loss): D row D == 0, phi 1',
}
MIDBLOCK_FORMULA = ('SoH_mid_y = SoH_{y-1} * exp(-D_y / 2) * phi**(n / 2): the within-block exponential decay '
                    'SoH(t) = SoH_{y-1} exp(-(D_y/n) t) phi**t at t = n/2, equal to the geometric mean '
                    'sqrt(SoH_{y-1} SoH_y); shared_energy_storage_data._esso_ageing_model_settings (e5721b6d)')
GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 11 * GIB // 4  # 2.75 GiB
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
MEMORY_RULE_TEMPLATE = ('hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {concurrency} x '
                        '{per_child:g} GiB')
MEMORY_RULE = MEMORY_RULE_TEMPLATE.format(concurrency=CONCURRENCY, per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside)')
OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); value = Q(0) - Q(x) '
                        'with Q(0) from the pinned A0 x0 record; F would be I(x) + Q(x); terminal_salvage_value '
                        'and net_operational_recourse are reported, excluded (Addendum 27 item 3).')
RESOLUTION_RULE = ('spec v16 ageing_batch.predictions_recorded_before_run.resolution: "a variant\'s value - I is '
                   'reported with the sum of its bar and x0\'s bar and with sigma_Q; a difference below them is '
                   'indeterminate". Here: indeterminate_vs_bars = |value - I| <= bar_variant + bar_x0; '
                   'indeterminate_vs_sigma_Q = |value - I| <= the upper end of the sigma_Q range spec v16 states '
                   '(10k-18k EUR, Planner prediction text); both reported, neither computed from judgement.')
SIGMA_Q_STATED_RANGE_EUR = (10000.0, 18000.0)
EXECUTION = ('spec v16 ageing_batch.execution: "one batch, 5 evaluations, concurrency 5, cap 500; non-certified '
             'variants reported with cause"; the batch is followed by STOP FOR REVIEW (spec v16 ageing_batch.stop)')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addenda 28-29',
    'data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json ageing_batch, concurrency',
    'Planner task W20 item 5 (one campaign s46_ageing holding 5 evaluations, each with its own model_variant; '
    'x = 0 not re-run, Q(0) from the A0 record; value_C3 261,807.504426 from A1a)',
    'e5721b6d (ageing switches + harness model_variant + zero-solve checks), cb165d4e (case-file edit + gate), '
    '8ed12e30 (two-cycle bitwise gate, defaults unchanged)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), 'p515_s46_variant_checks.py', 'p515_s46_case_file_edit_check.py',
                     'p515_s46_default_two_cycle_gate.py', ESS_PARAMS_FILE['path'],
                     'shared_energy_storage_parameters.py', 'shared_energy_storage.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _points():
    return tuple((f'{UNIT_LABEL}_{name}', dict(UNIT_NODES), UNIT_YEAR, name, dict(mv)) for name, mv in VARIANTS)


def _unit_key():
    return H.candidate_key(H.canonical_candidate(UNIT_NODES, investment_year=UNIT_YEAR))


def _eval_key(mv):
    return H.evaluation_key(_unit_key(), {}, case_file_aa=CASE_FILE_AA, model_variant=mv)


# ======================================================================================================================
#  pinned inputs: spec v16, Q(0), value_C3, I(x)
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v16', SPEC_V16), ('cost_file', COST_FILE), ('ess_params_file', ESS_PARAMS_FILE),
                      ('a0_spec', A0_SPEC), ('a0_results', A0_RESULTS), ('a1a_results', A1A_RESULTS),
                      ('investment_cost_results', INVESTMENT_COST_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == pin['sha256'] and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for name, pin in PINNED_EVIDENCE:
        path = os.path.join(REPO, pin['path'])
        present = os.path.isfile(path)
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], cwd=REPO,
                                 capture_output=True).returncode == 0
        payload = _load(pin['path']) if present else {}
        passed = payload.get('all_ok', payload.get('gate_pass'))
        out[name] = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': in_head, 'git_tracked': tracked,
                     'sha256': H.sha256_file(path) if present else None, 'passed': passed}
        if not (present and tracked and in_head and passed is True):
            failures.append(f'{name}: {out[name]}')
    return out, failures


def baseline_inputs():
    """Q(0) and its bar from the A0 x0 record; Q_C3 at the unit from the A1a record; value_C3; I(x) of the unit
    from the pinned W2 table (by candidate key) cross-checked with the A1a record and spec v16."""
    failures = []
    spec16 = _load(SPEC_V16['path'])
    batch = spec16['ageing_batch']
    a0 = _load(A0_RESULTS['path'])['points'][A0_RESULTS['label']]
    a1 = _load(A1A_RESULTS['path'])['points'][A1A_RESULTS['label']]
    unit_key = _unit_key()
    x0_key = H.candidate_key(H.canonical_candidate({n: (0.0, 0.0) for n in H.ACTIVE_NODES}, investment_year=UNIT_YEAR))
    q0 = a0['certified_cost_gross_settlement_excluded']
    q_c3 = a1['certified_cost_gross_settlement_excluded']
    value_c3 = q0 - q_c3
    table = _load(INVESTMENT_COST_RESULTS['path'])['candidates']
    hits = sorted({c.get('I_new_eur') for c in table.values() if c.get('candidate_key') == unit_key}, key=repr)
    i_x = hits[0] if len(hits) == 1 else None
    checks = {
        'a0_x0_certified': a0.get('status') == 'certified',
        'a0_x0_candidate_key': a0.get('candidate_key') == x0_key,
        'a1a_unit_certified': a1.get('status') == 'certified',
        'a1a_unit_candidate_key': a1.get('candidate_key') == unit_key,
        'spec_v16_candidate_key': batch['candidate']['candidate_key'] == unit_key,
        'spec_v16_candidate_nodes': batch['candidate']['nodes'] == {'7': [0.25, 1.0]},
        'spec_v16_Q0_equals_a0': batch['baseline_value_C3']['Q0'] == q0,
        'spec_v16_Q_C3_equals_a1a': batch['baseline_value_C3']['Q_C3'] == q_c3,
        'value_C3_equals_planner_pin_1e-6': abs(value_c3 - VALUE_C3_PINNED) <= 1e-6,
        'value_C3_equals_spec_v16_rounded': abs(value_c3 - batch['baseline_value_C3']['value']) <= 0.01,
        'I_x_single_value_in_w2_table': i_x is not None,
        'I_x_equals_spec_v16': i_x == batch['candidate']['I_x_eur'],
        'I_x_equals_a1a_record': i_x == a1.get('I_x_eur'),
        'spec_v16_variant_names': sorted(batch['variants']) == sorted(n for n, _mv in VARIANTS),
    }
    failures += [f'baseline input check failed: {k}' for k, v in checks.items() if not v]
    return {
        'Q0_eur': q0, 'Q0_bar_eur': (a0.get('bar') or {}).get('value'), 'Q0_source': dict(A0_RESULTS),
        'Q0_eval_dir': a0.get('eval_dir'), 'Q0_certification_cycle': a0.get('certification_cycle'),
        'Q_C3_eur': q_c3, 'Q_C3_bar_eur': (a1.get('bar') or {}).get('value'), 'Q_C3_source': dict(A1A_RESULTS),
        'Q_C3_eval_dir': a1.get('eval_dir'), 'value_C3_eur': value_c3, 'value_C3_planner_pin_eur': VALUE_C3_PINNED,
        'I_x_eur': i_x, 'I_x_source': dict(INVESTMENT_COST_RESULTS), 'unit_candidate_key': unit_key,
        'x0_candidate_key': x0_key, 'value_C3_minus_I_eur': value_c3 - i_x if i_x is not None else None,
        'predictions_recorded_before_run': batch['predictions_recorded_before_run'],
        'Q0_not_rerun': ('x = 0 has no storage, so Q(0) is ageing-independent (spec v16 ageing_batch.Q0_unchanged); '
                         'taken from the pinned A0 record'),
        'checks': checks}, failures


# ======================================================================================================================
#  memory preflight (A0's measure; see the module docstring)
# ======================================================================================================================
_VM_STAT_KEYS = {'pages_free': 'Pages free', 'pages_active': 'Pages active', 'pages_inactive': 'Pages inactive',
                 'pages_speculative': 'Pages speculative', 'pages_wired_down': 'Pages wired down',
                 'pages_purgeable': 'Pages purgeable', 'file_backed_pages': 'File-backed pages',
                 'anonymous_pages': 'Anonymous pages', 'pages_stored_in_compressor': 'Pages stored in compressor',
                 'pages_occupied_by_compressor': 'Pages occupied by compressor'}
_VM_STAT_REQUIRED = ('pages_free', 'pages_inactive', 'pages_wired_down', 'anonymous_pages',
                     'pages_occupied_by_compressor')


def _memory_rule_matches_a0():
    extra = _load(A0_SPEC['path'])['extra']
    a0_rule = extra.get('memory_preflight_rule')
    a0_concurrency = _load(A0_SPEC['path']).get('concurrency')
    template_reproduces_a0 = a0_rule == MEMORY_RULE_TEMPLATE.format(concurrency=a0_concurrency,
                                                                    per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)
    rationale_ok = extra.get('memory_preflight_rule_rationale') == MEMORY_RULE_RATIONALE
    return {'a0_rule': a0_rule, 'a0_concurrency': a0_concurrency, 'rule': MEMORY_RULE,
            'template_reproduces_a0_rule': template_reproduces_a0, 'a0_rationale_matches': rationale_ok,
            'match': template_reproduces_a0 and rationale_ok}


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
#  rule eleven
# ======================================================================================================================
CAMPAIGN_RESULT_FIELDS = (
    'status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'Q_gross_eur', 'value_eur', 'I_x_eur',
    'value_minus_I_eur', 'value_over_value_C3', 'bar', 'x0_bar', 'bar_sum', 'terminal_gross_step',
    'rule_ten', 'resolution', 'model_variant', 'model_variant_label', 'readback_pre_run', 'readback_terminal',
    'efc_per_day_per_block', 'soh_trajectory_per_block', 'salvage', 'wall_time', 'peak_rss', 'aa_action_counts',
    'per_cycle_trajectory')


def rule_eleven():
    harness_checks = H.assert_record_capture_paths()
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    hook_src = inspect.getsource(H._config_hook_factory)
    traj_src = inspect.getsource(H.ageing_trajectory_terminal)
    main_child_src = inspect.getsource(H.main_child)
    barrier_src = inspect.getsource(H._barrier_record_for_missing)
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
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'per_cycle_gross_and_tolerance': all(f in H.PER_CYCLE_RECORD_FIELDS for f in (
            'cycle', 'gross_operational_cost', 'objective_tolerance', 'objective_change_abs')),
        'salvage_recourse_components': ("'recourse_components': rc" in build_src
                                        and "'terminal_salvage_value':" in rc_src),
        'model_variant_in_record': ("'model_variant': model_variant, 'model_variant_label': MODEL_VARIANT_LABEL"
                                    in child_src and '**variant_extra' in child_src),
        'model_variant_in_error_records': ("'model_variant': entry['model_variant']" in main_child_src
                                           and "'model_variant': entry['model_variant']" in barrier_src),
        'readback_pre_run': ("holder['model_variant_readback_pre_run'] = readback" in hook_src
                             and "'model_variant_readback_pre_run':" in child_src),
        'readback_terminal': ("holder['model_variant_readback_terminal'] = model_variant_readback_models("
                              in child_src and "'model_variant_readback_terminal':" in child_src),
        'ageing_trajectory_terminal': ("holder['ageing_trajectory_terminal'] = ageing_trajectory_terminal(" in child_src
                                       and all(f"'{f}'" in traj_src for f in (
                                           'efc_per_day', 'soh_end', 'soh_used_for_available_energy',
                                           'soh_mid_closed_form', 'salvage_value', 'block_year'))),
        'variant_applied_in_hook_before_build': ('apply_model_variant(sed, model_variant)' in hook_src
                                                 and 'SED._build_subproblem(sed, node_id)' in hook_src),
        'child_refuses_unlabelled_variant': "spec.get('model_variant_label') != MODEL_VARIANT_LABEL" in child_src,
        'wall_time': "'wall_time_s': wall" in build_src,
        'peak_rss': "'peak_rss': peak_rss" in build_src,
        'aa_action_counts': ("'aa_per_cycle': holder.get('aa_sidecar')" in child_src
                             and "'action_counts'" in inspect.getsource(H.aa_sidecar_summary)),
        'I_x_Q0_value_C3_frozen_in_spec': True,  # asserted value by value by _validate_spec on the spec itself
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s46 ageing campaign_results): capture paths missing: {missing}')
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
    pins, more = _check_pins()
    evidence['pins'] = pins
    failures += more
    memory_rule = _memory_rule_matches_a0()
    evidence['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f'memory preflight rule differs from A0\'s measure: {memory_rule}')
    spec16 = _load(SPEC_V16['path'])
    if spec16['concurrency'].get('srp1') != CONCURRENCY:
        failures.append(f"spec v16 concurrency.srp1 {spec16['concurrency'].get('srp1')} != {CONCURRENCY}")
    base, more = baseline_inputs()
    evidence['baseline_inputs'] = base
    failures += more
    points = _points()
    keys = [_eval_key(mv) for _l, _n, _y, _name, mv in points]
    if len(set(keys)) != len(keys) or _eval_key(None) in keys:
        failures.append('variant eval keys are not distinct from each other and from the baseline')
    evidence['points'] = [{'label': label, 'variant': name, 'model_variant': mv,
                           'description': VARIANT_DESCRIPTIONS[name],
                           'candidate_key': _unit_key(), 'eval_key': _eval_key(mv)}
                          for label, _n, _y, name, mv in points]
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, points


def _validate_spec(spec, points, base):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    extra = spec.get('extra') or {}
    checks = {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'entries_in_order': [e['label'] for e in entries] == [p[0] for p in points],
        'n_entries_5': len(entries) == 5,
        'declaration': spec['configuration'].get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'no_campaign_overrides': spec['configuration'].get('overrides') == {},
        'cap': spec.get('cap') == CAP,
        'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': spec['configuration'].get('arm_label') == 's39_D',
        'spec_level_model_variant_label': spec.get('model_variant_label') == H.MODEL_VARIANT_LABEL,
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'spec_v16_recorded': extra.get('spec_v16') == SPEC_V16,
        'ess_params_file_recorded': extra.get('ess_params_file') == ESS_PARAMS_FILE,
        'cost_file_recorded': (extra.get('cost_file') or {}).get('sha256') == COST_FILE['sha256'],
        'memory_rule_recorded': extra.get('memory_preflight_rule') == MEMORY_RULE,
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'resolution_rule_recorded': extra.get('resolution_rule') == RESOLUTION_RULE,
        'Q0_frozen': (extra.get('baseline_inputs') or {}).get('Q0_eur') == base['Q0_eur'],
        'Q0_bar_frozen': (extra.get('baseline_inputs') or {}).get('Q0_bar_eur') == base['Q0_bar_eur'],
        'value_C3_frozen': (extra.get('baseline_inputs') or {}).get('value_C3_eur') == base['value_C3_eur'],
        'I_x_frozen': (extra.get('baseline_inputs') or {}).get('I_x_eur') == base['I_x_eur'],
        'label_recorded': extra.get('label') == H.MODEL_VARIANT_LABEL,
    }
    for label, nodes, year, name, mv in points:
        entry = by_label.get(label) or {}
        canon = H.canonical_candidate(nodes, investment_year=year)
        key = H.candidate_key(canon)
        checks[f'{label}:canonical_key'] = entry.get('canonical') == canon and entry.get('key') == key
        checks[f'{label}:model_variant'] = entry.get('model_variant') == H.validate_model_variant(mv)
        checks[f'{label}:model_variant_label'] = entry.get('model_variant_label') == H.MODEL_VARIANT_LABEL
        checks[f'{label}:no_overrides_no_post_certification'] = (entry.get('overrides') == {}
                                                                 and entry.get('post_certification') is None)
        checks[f'{label}:effective_aa_is_declaration'] = entry.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:eval_key_recomputes'] = entry.get('eval_key') == H.evaluation_key(
            key, {}, case_file_aa=CASE_FILE_AA, model_variant=mv)
        checks[f'{label}:variant_name_recorded'] = (extra.get('variants') or {}).get(label, {}).get('name') == name
    return checks


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(started):
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, points = _common_checks()
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[S46-AGEING FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    base = evidence['baseline_inputs']
    variants_extra = {label: {'name': name, 'model_variant': mv, 'description': VARIANT_DESCRIPTIONS[name],
                              'eval_key': _eval_key(mv)} for label, _n, _y, name, mv in points}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID,
        [(label, nodes, {'investment_year': year, 'model_variant': mv}) for label, nodes, year, _name, mv in points],
        configuration={'name': ('MODEL VARIANT batch (Addenda 28-29): the case file alone (AA keep_memory adopted '
                                'in data/SRP1/SRP1_params.json) + one model_variant per evaluation'),
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'note': ('no AA overrides; no post-certification; each evaluation carries its own '
                                'model_variant (applied and read back in the child); num_max_iters := cap')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'label': H.MODEL_VARIANT_LABEL, 'mode': 'full', 'stage': 'P5.15 Addenda 28-29 ageing batch',
               'spec_v16': dict(SPEC_V16), 'cost_file': dict(COST_FILE), 'ess_params_file': dict(ESS_PARAMS_FILE),
               'a0_spec': dict(A0_SPEC), 'pins': evidence['pins'],
               'baseline_inputs': base, 'variants': variants_extra, 'midblock_formula': MIDBLOCK_FORMULA,
               'post_certification': 'none', 'execution': EXECUTION,
               'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE,
               'sigma_Q_stated_range_eur': list(SIGMA_Q_STATED_RANGE_EUR),
               'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                  'gross_operational_cost[k-1]| (harness W5)'),
               'memory_preflight_rule': MEMORY_RULE, 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
               'memory_preflight_refusing_at': '--run (non-gating at --freeze)',
               'memory_at_freeze_non_gating': memory,
               'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)})
    checks = _validate_spec(spec, points, base)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[S46-AGEING] {H.MODEL_VARIANT_LABEL}')
    _log(f'[S46-AGEING] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for entry in spec['candidates']:
        _log(f"[S46-AGEING]   {entry['label']}: variant={entry['model_variant']} key={entry['key'][:16]} "
             f"eval_key={entry['eval_key'][:16]} eval_dir={entry['eval_dir']}")
    _log(f"[S46-AGEING] Q(0)={base['Q0_eur']} (bar {base['Q0_bar_eur']}) Q_C3={base['Q_C3_eur']} "
         f"value_C3={base['value_C3_eur']} I(x)={base['I_x_eur']} value_C3-I={base['value_C3_minus_I_eur']}")
    _log(f'[S46-AGEING] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[S46-AGEING] baseline input checks: {base['checks']}")
    _log(f"[S46-AGEING] pins: {evidence['pins']}")
    _log(f"[S46-AGEING] memory rule vs A0: {evidence['memory_rule_vs_a0_spec']}")
    _log(f"[S46-AGEING] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[S46-AGEING] memory at freeze (non-gating): {_memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[S46-AGEING] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[S46-AGEING] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[S46-AGEING] run with: --run --spec-sha256 {spec_sha}')
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


def _readback_summary(readback):
    if not readback:
        return None
    node7 = (readback.get('per_node') or {}).get('7') or {}
    rb = node7.get('readback') or {}
    return {'all_match': readback.get('all_match'), 'expected': readback.get('expected'),
            'node7': {'k': rb.get('k'), 'phi_cal_in_model': rb.get('phi_cal_in_model'),
                      'available_energy_soh_point': rb.get('available_energy_soh_point'),
                      'd_row_form': rb.get('d_row_form'), 'checks': node7.get('checks')},
            'per_node_all_checks': {n: all((v.get('checks') or {}).values())
                                    for n, v in (readback.get('per_node') or {}).items()}}


def _point_result(label, rec, name, mv, base):
    rec = rec or {}
    traj, rows = _per_cycle_trajectory(rec)
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else None
    i_x = base['I_x_eur']
    value = (base['Q0_eur'] - q) if q is not None else None
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  and prev.get('cycle') == (last.get('cycle') or 0) - 1 else None)
    tol = last.get('objective_tolerance')
    bar = (rec.get('bar') or {}).get('value')
    bar_sum = (bar + base['Q0_bar_eur']) if bar is not None else None
    vmi = (value - i_x) if value is not None else None
    ageing = rec.get('ageing_trajectory_terminal') or {}
    node7 = (ageing.get('nodes') or {}).get('7') or {}
    cells = node7.get('cells') or []
    rule_ten = rec.get('rule_ten') or {}
    return {
        'label': label, 'variant': name, 'model_variant': rec.get('model_variant', mv),
        'model_variant_label': rec.get('model_variant_label'), 'description': VARIANT_DESCRIPTIONS[name],
        'candidate_key': rec.get('candidate_key'), 'eval_key': rec.get('eval_key'), 'eval_dir': rec.get('eval_dir'),
        'status': rec.get('status'), 'barrier': rec.get('barrier'), 'barrier_cause': rec.get('barrier_cause'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'Q_gross_eur': q, 'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'Q0_eur': base['Q0_eur'], 'value_eur': value, 'I_x_eur': i_x, 'value_minus_I_eur': vmi,
        'value_over_value_C3': (value / base['value_C3_eur']) if value is not None else None,
        'value_C3_eur': base['value_C3_eur'],
        'bar': {'value': bar, 'definition': (rec.get('bar') or {}).get('definition')},
        'x0_bar': base['Q0_bar_eur'], 'bar_sum': bar_sum,
        'terminal_gross_step': gross_step,
        'rule_ten': {'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol)
                     else None, 'terminal_objective_tolerance': tol,
                     'terminal_step_over_threshold_production': rule_ten.get('terminal_step_over_threshold'),
                     'boyd_terminal_ratio_max_per_channel': rule_ten.get('boyd_terminal_ratio_max_per_channel')},
        'resolution': {
            'rule': RESOLUTION_RULE,
            'abs_value_minus_I_eur': abs(vmi) if vmi is not None else None,
            'indeterminate_vs_bars': (abs(vmi) <= bar_sum) if (vmi is not None and bar_sum is not None) else None,
            'sigma_Q_stated_range_eur': list(SIGMA_Q_STATED_RANGE_EUR),
            'indeterminate_vs_sigma_Q_upper': (abs(vmi) <= SIGMA_Q_STATED_RANGE_EUR[1]) if vmi is not None else None,
            'sign_of_value_minus_I': (None if vmi is None else ('positive' if vmi > 0 else 'negative' if vmi < 0
                                                                 else 'zero'))},
        'prediction': {'expert_value_multiplier_vs_C3': base['predictions_recorded_before_run'][
                           'expert_value_multiplier_vs_C3'].get(name),
                       'implied_value_eur': base['predictions_recorded_before_run']['implied_value_eur'].get(name),
                       'implied_value_minus_I_eur': base['predictions_recorded_before_run'][
                           'implied_value_minus_I_eur'].get(name)},
        'readback_pre_run': _readback_summary(rec.get('model_variant_readback_pre_run')),
        'readback_terminal': _readback_summary(rec.get('model_variant_readback_terminal')),
        'model_variant_applied_checks': (rec.get('model_variant_applied_in_child') or {}).get('checks'),
        'efc_per_day_per_block': {c['block_year']: c['efc_per_day'] for c in cells},
        'soh_trajectory_per_block': [{k: c.get(k) for k in ('investment_year', 'block_year', 'n_years', 'D',
                                                            'soh_prev_end', 'soh_end', 'soh_used_for_available_energy',
                                                            'soh_mid_closed_form', 'e_rated', 'e_available',
                                                            'cl_eff', 'phi_cal_in_model')} for c in cells],
        'salvage': {'terminal_salvage_value_recourse_components':
                    (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
                    'per_node_salvage_value': {n: v.get('salvage_value') for n, v in (ageing.get('nodes') or {}).items()},
                    'net_operational_recourse': rec.get('terminal_net_operational_recourse')},
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': {'record': rec.get('peak_rss'),
                     'parent_wait4_ru_maxrss_bytes': (rec.get('parent_view') or {}).get('wait4_ru_maxrss')},
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'anderson_acceleration_effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'case_file_sha256_in_child': rec.get('case_file_sha256_in_child'),
        'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
        'per_cycle_trajectory': traj,
    }


def run(started, spec_sha256):
    tag = 'S46-AGEING'
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence, points = _common_checks()
    failures += more
    base = evidence['baseline_inputs']
    checks = _validate_spec(spec, points, base)
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
    _log(f'[{tag}] {H.MODEL_VARIANT_LABEL}')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    labels = [p[0] for p in points]
    _log(f'[{tag}] {len(labels)} evaluations in ONE batch of {CONCURRENCY}: {labels}')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        batch_info = dict(getattr(H.evaluate, 'last_batch_info', {}))
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    point_results = {label: _point_result(label, by_label.get(label), name, mv, base)
                     for label, _n, _y, name, mv in points}
    non_certified = [label for label in labels if point_results[label]['status'] != 'certified']
    harness_errors = [label for label in labels
                      if point_results[label]['status'] not in ('certified', 'not_certified')
                      or point_results[label]['exit_code'] != 0]
    readback_mismatch = [label for label in labels
                         if point_results[label]['status'] in ('certified', 'not_certified')
                         and not ((point_results[label]['readback_pre_run'] or {}).get('all_match')
                                  and (point_results[label]['readback_terminal'] or {}).get('all_match'))]
    guard_failures = PARENT_GUARD.verify(0)
    table = [{'variant': point_results[l]['variant'], 'status': point_results[l]['status'],
              'cycles': point_results[l]['cycles_run'], 'Q': point_results[l]['Q_gross_eur'],
              'value': point_results[l]['value_eur'], 'I_x': point_results[l]['I_x_eur'],
              'value_minus_I': point_results[l]['value_minus_I_eur'],
              'value_over_value_C3': point_results[l]['value_over_value_C3'],
              'bar': point_results[l]['bar']['value'], 'bar_sum': point_results[l]['bar_sum'],
              'indeterminate_vs_bars': point_results[l]['resolution']['indeterminate_vs_bars'],
              'k_read_back': ((point_results[l]['readback_terminal'] or {}).get('node7') or {}).get('k'),
              'phi_read_back': ((point_results[l]['readback_terminal'] or {}).get('node7') or {}).get('phi_cal_in_model'),
              'soh_point_read_back': ((point_results[l]['readback_terminal'] or {}).get('node7') or {}).get(
                  'available_energy_soh_point'),
              'prediction_value_minus_I': point_results[l]['prediction']['implied_value_minus_I_eur']}
             for l in labels]
    results = {
        'LABEL': H.MODEL_VARIANT_LABEL,
        'STOP_FOR_REVIEW': True,
        'stop_reason': 'spec v16 ageing_batch.stop: STOP FOR REVIEW after the ageing batch',
        'non_certified_points': non_certified, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch,
        'execution': EXECUTION, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE,
        'midblock_formula': MIDBLOCK_FORMULA,
        'baseline_inputs': base,
        'table_objective_convention': 'Q gross (settlement excluded); value = Q(0) - Q; salvage excluded',
        'table': table,
        'points': point_results,
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pins', 'memory_rule_vs_a0_spec')},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
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
    _log(f'[{tag}] {H.MODEL_VARIANT_LABEL} -- objective convention: {OBJECTIVE_CONVENTION}')
    for row in table:
        _log(f'[{tag}] {row}')
    if non_certified:
        _log(f'[{tag}] non-certified variants (reported with cause): '
             f"{[(l, point_results[l]['barrier_cause']) for l in non_certified]}")
    if readback_mismatch:
        _log(f'[{tag}] READ-BACK MISMATCH: {readback_mismatch}')
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    _log(f'[{tag}] STOP FOR REVIEW (spec v16 ageing_batch.stop)')
    if guard_failures or harness_errors or readback_mismatch:
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    _log(f'[{tag}] OK: harness clean')


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
