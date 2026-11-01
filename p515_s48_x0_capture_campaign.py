"""
P5.15 Addendum 32 (task W27, Q4 "terminal TSO capture at x = 0") -- ONE certified evaluation of x = 0 under the
ageing BASELINE declaration, through the campaign harness (`p515_s44_campaign_harness.py`), with the harness's
existing post-certification option `persist_certified_models: true` (NO hull polish, NO reference), so that the
certified terminal TSO / DSO models (Pyomo blocks carrying the terminal IPOPT duals in their `dual` /
`ipopt_zL_out` / `ipopt_zU_out` suffixes) are pickled for the zero-solve bus-7 marginal-cost analysis
(`p515_s48_x0_capture_analysis.py`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 32; frozen spec v18
`data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json` `Q4_terminal_tso_capture`; Planner task W27.

Launcher pattern of `p515_s47_baseline_campaign.py` (not imported: its module-level parent guard would stack on
this one): preconditions, locks, clean git, one spec per campaign root, memory preflight, cap 500, 10
consecutive all-pass cycles, case-file AA declaration, the ESS ageing BASELINE declaration
(`configuration.ess_ageing_baseline`), NO overrides, NO model variant. Differences from s47, all by Planner
instruction (W27): ONE candidate (x = 0, 2025 canonical); concurrency 1; the post-certification request
{persist_certified_models: true, hull_polish: false} (no reference).

The eval key differs from A0's x0 (baseline declaration enters the key) -- expected. x = 0 is ageing-independent
(W21 NL identity, 2466401d; spec v17 / v18 Q(0) statement), so the GATE (spec v18 Q4, gating) is: this evaluation
reproduces the committed A0 x0 evaluation (data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/)
bitwise on every trajectory field and cost (132 cycles, gross 653,859,461.2279255). The gate is evaluated by
`p515_s48_x0_capture_gate.py` (zero solves; the comparator conventions of `p515_s45_reverify_compare.py`, by
import), committed with this launcher BEFORE the run.

Two modes, attached, both streams captured, never detached:
  --freeze                    ZERO SOLVES: preconditions, pins, rule eleven, `freeze_campaign_spec` + validation.
  --run --spec-sha256 <sha>   loads THAT spec (the campaign root must hold only it), re-checks everything plus the
                              harness / case-file / ESS-params / script sha256, the memory preflight (refusing),
                              takes the campaign lock, evaluates, writes campaign_results.json and
                              campaign_manifest_sha256.json (which hash-records certified_models.pkl).
The parent never solves: SolveProfileGuard(permitted=()) installed before any model import, verified at exactly
0 in both modes. Exit codes (--run): 0 certified and persisted; 2 not certified / post-certification not
evaluated (harness clean); 1 harness / guard / precondition failure.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_x0_capture_campaign.py --freeze \\
      > data/SRP1/Results/P515S48/x0_capture_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_x0_capture_campaign.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S48/x0_capture_launch.log 2>&1
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S48 x0 capture campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
_P48 = os.path.join('data', 'SRP1', 'Results', 'P515S48')
_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CAMPAIGN_ID = 's48_x0_capture'
CAMPAIGN_ROOT = os.path.join(REPO, _P48, 'x0_capture')
POINT_LABEL = 'x0'
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 1  # Planner task W27 (one candidate)
REQUIRED_CONSECUTIVE_CYCLES = 10
ACTIVE_NODES = (5, 7, 9)
YEAR = 2025
POST_CERTIFICATION = {'persist_certified_models': True, 'hull_polish': False}

ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
SPEC_V18 = {'path': os.path.join(_P48, 'frozen_s48_spec_v18_8bda2a0a.json'),
            'sha256': '8bda2a0a94d6f6f23ed544bc9b0bb4ce3fcd32c547ed3f472ccb63c53fb36fe6'}
SPEC_V17 = {'path': os.path.join(_P47, 'frozen_s47_baseline_spec_v17_ff0056b8.json'),
            'sha256': 'ff0056b850957d47cc3aa2595eab13497e100cc7bcda84514c5ca58e25d1e109'}
ESS_PARAMS_FILE = {'path': H.ESS_PARAMS_FILE_REL,
                   'sha256': '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706',
                   'commit': '2466401d', 'note': 'the ageing BASELINE edit of W21 (Addendum 30)'}
A0_SPEC = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_spec_s45_a0_c7_9d08ad2f.json'),
           'sha256': '9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34'}
A0_RESULTS = {'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json'),
              'sha256': '423678b98c0ca9a26dbc76a1b46a5e36a51c8ccccc0459253020bfe1061653b4', 'label': 'x0'}
A0_X0_DIR = os.path.join(_P45, 'campaign_s45_a0_c7', 'evals', '7aa017f09989b56d_x0')
A0_X0_PINS = {  # the gate's reference files (committed, 750f96de)
    'evaluation_record.json': '34868965df76a9f1dc62d93d6ce4e2a50ceb4e12f8d3f2aeb4f8c96742bb4835',
    'g_s39_D.json': '9b4d86760fd920b40c5c93400428dc566a5bdab01c27eb9eefcd5cbfd1dbc0d0',
    'per_cycle_record.jsonl': '8663b96213c87dbb69d90790fc4fe771106f2bd838e17d9809f752cdf7a2f01e',
    'component_levels_terminal.json': '32696d9c3afc7f4e3e6624d59f21910509e6e6fe1f5e75a43b33f1413acd7279',
}
EXPECTED_CYCLES = 132
EXPECTED_COST = 653859461.2279255
GATE_TEXT = ('the x = 0 evaluation reproduces the committed A0 x0 evaluation bitwise on every trajectory field and '
             'cost (132 cycles, 653,859,461.2279255) - Q(0) is ageing-independent (W21 NL identity) and the oracle '
             'is deterministic; the persistence step must not alter the trajectory')
OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); terminal salvage and '
                        'net_operational_recourse = gross - salvage reported, excluded')

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
    'PLANNER_BRIEF_2026-09-13.md Addendum 32 (Q4: terminal TSO capture at x = 0)',
    'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json Q4_terminal_tso_capture',
    'Planner task W27 (one evaluation of x = 0, baseline declaration, AA-on case file, cap 500, 10 cycles, '
    'concurrency 1, no overrides, persist_certified_models true, no hull polish)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), 'p515_s48_x0_capture_gate.py', H.ESS_PARAMS_FILE_REL,
                     'shared_energy_storage_parameters.py', 'shared_energy_storage.py', 'network.py',
                     'p515_s42_exact_fix_rerun.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _nodes_x0():
    return {n: (0.0, 0.0) for n in ACTIVE_NODES}


def _key_x0():
    return H.candidate_key(H.canonical_candidate(_nodes_x0(), investment_year=YEAR))


def _eval_key_x0():
    return H.evaluation_key(_key_x0(), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE)


# ======================================================================================================================
#  pins, reference, memory, rule eleven
# ======================================================================================================================
def _check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v18', SPEC_V18), ('spec_v17', SPEC_V17), ('ess_params_file', ESS_PARAMS_FILE),
                      ('a0_spec', A0_SPEC), ('a0_results', A0_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', pin['path']]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', pin['path']]).strip())
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == pin['sha256'] and tracked and not dirty):
            failures.append(f'{name}: {out[name]}')
    for fname, sha in A0_X0_PINS.items():
        rel = os.path.join(A0_X0_DIR, fname)
        path = os.path.join(REPO, rel)
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked = bool(H._git(['ls-files', '--', rel]).strip())
        dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
        out[f'a0_x0:{fname}'] = {'path': rel, 'sha256_pinned': sha, 'sha256_on_disk': got, 'match': got == sha,
                                 'git_tracked': tracked, 'git_clean': not dirty}
        if not (got == sha and tracked and not dirty):
            failures.append(f'a0_x0 {fname}: {out[f"a0_x0:{fname}"]}')
    return out, failures


def reference_inputs():
    failures = []
    a0 = _load(A0_RESULTS['path'])['points'][A0_RESULTS['label']]
    rec = _load(os.path.join(A0_X0_DIR, 'evaluation_record.json'))
    spec18 = _load(SPEC_V18['path'])
    spec17 = _load(SPEC_V17['path'])
    v17_ageing = spec17['baseline']['ageing']
    checks = {
        'a0_x0_certified': a0.get('status') == 'certified' and rec.get('status') == 'certified',
        'a0_x0_candidate_key_is_x0_2025': a0.get('candidate_key') == _key_x0() == rec.get('candidate_key'),
        'a0_x0_eval_dir_pinned': a0.get('eval_dir') == A0_X0_DIR,
        'a0_x0_cycles': rec.get('cycles_run') == EXPECTED_CYCLES == rec.get('certification_cycle'),
        'a0_x0_cost_exact': (type(rec.get('certified_cost')) is float and rec.get('certified_cost') == EXPECTED_COST
                             and a0.get('certified_cost_gross_settlement_excluded') == EXPECTED_COST),
        'a0_x0_no_post_certification': all(rec.get(k) is None for k in (
            'post_certification', 'post_certification_path', 'post_certification_capture_checklist_asserted_before_run')),
        'a0_x0_effective_aa_is_declaration': rec.get('anderson_acceleration_effective_in_child') == CASE_FILE_AA,
        'spec_v18_has_q4': 'Q4_terminal_tso_capture' in spec18,
        'declaration_matches_spec_v17_baseline': (
            v17_ageing['cycles_n'] == ESS_AGEING_BASELINE['calibration']['cycles_n']
            and v17_ageing['reference_dod_d'] == ESS_AGEING_BASELINE['calibration']['reference_dod_d']
            and v17_ageing['eol_retention_r'] == ESS_AGEING_BASELINE['calibration']['eol_retention_r']
            and v17_ageing['calendar_retention_per_year'] == ESS_AGEING_BASELINE['calendar_retention_per_year']
            and v17_ageing['minimum_soh'] == ESS_AGEING_BASELINE['minimum_soh']),
        'file_loads_to_declaration': (H.ess_ageing_canonical_text(H.load_ess_ageing_parameters(
            os.path.join(REPO, H.ESS_PARAMS_FILE_REL))) == H.ess_ageing_canonical_text(ESS_AGEING_BASELINE)),
        'eval_key_differs_from_a0_expected': _eval_key_x0() != rec.get('eval_key'),
    }
    failures += [f'reference input check failed: {k}' for k, v in checks.items() if not v]
    return {'a0_x0_dir': A0_X0_DIR, 'a0_x0_eval_key': rec.get('eval_key'), 'a0_x0_candidate_key': rec.get('candidate_key'),
            'expected_cycles': EXPECTED_CYCLES, 'expected_certified_cost_gross': EXPECTED_COST,
            'x0_candidate_key': _key_x0(), 'x0_eval_key_this_campaign': _eval_key_x0(),
            'q4_spec_v18': spec18.get('Q4_terminal_tso_capture'), 'gate_text': GATE_TEXT, 'checks': checks}, failures


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
    out.update({'non_reclaimable_bytes': non_reclaimable, 'available_bytes': avail, 'available_gib': avail / GIB,
                'free_plus_inactive_gib': (pages['pages_free'] + pages['pages_inactive']) * page / GIB,
                'pass': avail >= MEMORY_REQUIRED_BYTES})
    return out


def rule_eleven():
    """Every quantity the spec (v18 Q4) and the W27 analysis require has a capture path, asserted BEFORE the run."""
    harness_checks = H.assert_record_capture_paths()
    import p515_s42_exact_fix_rerun as EF
    import network as NW
    persist_src = inspect.getsource(EF._persist_certified_models)
    post_src = inspect.getsource(H.run_post_certification)
    child_src = inspect.getsource(H._child_real)
    build_src = inspect.getsource(NW)
    persist_pos = post_src.find("if request.get('persist_certified_models'):")
    polish_pos = post_src.find("if request.get('hull_polish'):")
    checks = {
        # the gate's inputs (trajectory + cost) -- the harness record capture paths
        'harness_record_capture_paths': bool(harness_checks),
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        # the persistence of the certified TSO / DSO models
        'child_runs_post_certification_when_requested': ('if post_request:' in child_src
                                                          and 'run_post_certification(' in child_src),
        'post_certification_persists_when_requested': "out['persisted_models'] = persist_fn(models, eval_dir)" in post_src,
        'persist_before_polish': 0 <= persist_pos < polish_pos,
        'persist_payload_holds_tso_and_dso': "payload = {'tso': models['tso'], 'dso': models['dso']}" in persist_src,
        'persist_hash_recorded': "'sha256': sha256" in persist_src,
        'persist_refuses_overwrite': '_refuse_overwrite(path)' in persist_src,
        # the duals the analysis reads (declared on every network model built by production)
        'network_model_dual_suffix_import': 'model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)' in build_src,
        'network_model_zL_suffix_import': 'model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)' in build_src,
        'network_model_zU_suffix_import': 'model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)' in build_src,
        'post_certification_request_is_persist_only': (
            H.resolve_post_certification(POST_CERTIFICATION, _key_x0())
            == {'persist_certified_models': True, 'hull_polish': False, 'reference': None}),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s48 x0 capture): capture paths missing: {missing}')
    return {'checks': checks, 'harness_record_capture_checklist': harness_checks,
            'note': ('the terminal duals themselves are verified by the analysis (capture-path assertion before '
                     'analysis: non-empty dual suffix on node_balance_p of every 2030 TSO block, and the W25 KKT '
                     'identity against the terminal pf_entry_stride row)')}


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
        failures.append(f"memory preflight rule differs from A0's measure: {memory_rule}")
    ref, more = reference_inputs()
    evidence['reference_inputs'] = ref
    failures += more
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence


def _validate_spec(spec, ref):
    entries = spec['candidates']
    cfg = spec['configuration']
    extra = spec.get('extra') or {}
    entry = entries[0] if len(entries) == 1 else {}
    canon = H.canonical_candidate(_nodes_x0(), investment_year=YEAR)
    return {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'one_entry': len(entries) == 1 and entry.get('label') == POINT_LABEL,
        'canonical_key_x0': entry.get('canonical') == canon and entry.get('key') == _key_x0() == ref['a0_x0_candidate_key'],
        'eval_key_recomputes_with_declaration': entry.get('eval_key') == _eval_key_x0(),
        'eval_key_not_a0s': entry.get('eval_key') != ref['a0_x0_eval_key'],
        'no_overrides': entry.get('overrides') == {} and cfg.get('overrides') == {},
        'effective_aa_is_declaration': entry.get('effective_anderson_acceleration') == CASE_FILE_AA,
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': ((cfg.get('ess_params_file') or {}).get('path') == ESS_PARAMS_FILE['path']
                                   and (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256']),
        'no_model_variant_anywhere': 'model_variant_label' not in spec and 'model_variant' not in entry,
        'post_certification_persist_only_no_polish_no_reference': entry.get('post_certification') == {
            'persist_certified_models': True, 'hull_polish': False, 'reference': None},
        'cap': spec.get('cap') == CAP,
        'concurrency_1': spec.get('concurrency') == CONCURRENCY == 1,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': cfg.get('arm_label') == 's39_D',
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'spec_v18_recorded': extra.get('spec_v18') == SPEC_V18,
        'gate_recorded': extra.get('gate_text') == GATE_TEXT and extra.get('gate_script') == 'p515_s48_x0_capture_gate.py',
        'a0_x0_pins_recorded': extra.get('a0_x0_pins') == A0_X0_PINS and extra.get('a0_x0_dir') == A0_X0_DIR,
        'expected_recorded': (extra.get('expected_cycles') == EXPECTED_CYCLES
                              and extra.get('expected_certified_cost_gross') == EXPECTED_COST),
        'memory_rule_recorded': extra.get('memory_preflight_rule') == MEMORY_RULE,
    }


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(started):
    tag = 'S48-X0CAP'
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence = _common_checks()
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    ref = evidence['reference_inputs']
    extra = {'campaign_script': os.path.basename(__file__),
             'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'gate_script': 'p515_s48_x0_capture_gate.py',
             'gate_script_sha256': H.sha256_file(os.path.join(REPO, 'p515_s48_x0_capture_gate.py')),
             'gate_text': GATE_TEXT, 'label': LABEL, 'mode': 'full',
             'stage': 'P5.15 Addendum 32 W27 -- Q4 terminal TSO capture at x = 0 (one evaluation)',
             'spec_v18': dict(SPEC_V18), 'spec_v17': dict(SPEC_V17), 'ess_params_file': dict(ESS_PARAMS_FILE),
             'a0_spec': dict(A0_SPEC), 'a0_results': dict(A0_RESULTS), 'a0_x0_dir': A0_X0_DIR,
             'a0_x0_pins': dict(A0_X0_PINS), 'expected_cycles': EXPECTED_CYCLES,
             'expected_certified_cost_gross': EXPECTED_COST, 'pins': evidence['pins'], 'reference_inputs': ref,
             'post_certification': dict(POST_CERTIFICATION), 'model_variant': 'none',
             'objective_convention': OBJECTIVE_CONVENTION,
             'memory_preflight_rule': MEMORY_RULE, 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
             'memory_preflight_refusing_at': '--run (non-gating at --freeze)', 'memory_at_freeze_non_gating': memory,
             'rule_eleven': evidence['rule_eleven']['checks'],
             'eval_key_note': ("this evaluation's key includes the ess_ageing_baseline declaration, so it differs from "
                               "A0's x0 key (expected, W27); the candidate key is A0's")}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID,
        [(POINT_LABEL, _nodes_x0(), {'investment_year': YEAR, 'post_certification': dict(POST_CERTIFICATION)})],
        configuration={'name': (f'{LABEL}: the case file alone (AA keep_memory adopted in data/SRP1/SRP1_params.json) '
                                'with the ESS ageing parameters declared; post-certification: persist the certified '
                                'TSO/DSO models only'),
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('no overrides; no model variant; post-certification persist_certified_models only (no '
                                'hull polish, no reference); num_max_iters := cap')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=extra)
    checks = _validate_spec(spec, ref)
    guard_failures = PARENT_GUARD.verify(0)
    entry = spec['candidates'][0]
    _log(f'[{tag}] {LABEL}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    _log(f"[{tag}]   {entry['label']}: key={entry['key'][:16]} eval_key={entry['eval_key'][:16]} "
         f"eval_dir={entry['eval_dir']} post_certification={entry['post_certification']}")
    _log(f"[{tag}] reference: {A0_X0_DIR} eval_key={ref['a0_x0_eval_key'][:16]}; checks {ref['checks']}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] pins: {evidence['pins']}")
    _log(f"[{tag}] memory rule vs A0: {evidence['memory_rule_vs_a0_spec']}")
    _log(f"[{tag}] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[{tag}] memory at freeze (non-gating): available {memory.get('available_gib')} GiB, required "
         f"{memory['required_gib']} GiB -> would {'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _point_result(rec):
    rec = rec or {}
    path = rec.get('per_cycle_record_path')
    traj = {'path': path, 'present': bool(path and os.path.isfile(os.path.join(REPO, path)))}
    if traj['present']:
        with open(os.path.join(REPO, path)) as handle:
            n = sum(1 for line in handle if line.strip())
        traj.update({'sha256': H.sha256_file(os.path.join(REPO, path)), 'n_rows': n,
                     'n_rows_equals_cycles_run': n == rec.get('cycles_run')})
    pc = rec.get('post_certification') or {}
    return {
        'LABEL': LABEL, 'label': POINT_LABEL, 'candidate_key': rec.get('candidate_key'),
        'candidate_canonical': rec.get('candidate_canonical'), 'eval_key': rec.get('eval_key'),
        'eval_dir': rec.get('eval_dir'), 'status': rec.get('status'), 'barrier_cause': rec.get('barrier_cause'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'certified_cost_gross_settlement_excluded': rec.get('certified_cost') if rec.get('status') == 'certified' else None,
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'bar': rec.get('bar'), 'rule_ten': rec.get('rule_ten'),
        'post_certification': pc, 'post_certification_path': rec.get('post_certification_path'),
        'persisted_models': pc.get('persisted_models'),
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': rec.get('peak_rss'),
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'anderson_acceleration_effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'ess_ageing_readback_all_match': {
            'pre_run': (((rec.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run')) or {}).get('all_match'),
            'terminal': (rec.get('ess_ageing_readback_terminal') or {}).get('all_match')},
        'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
        'per_cycle_trajectory': traj,
    }


def run(started, spec_sha256):
    tag = 'S48-X0CAP'
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence = _common_checks()
    failures += more
    ref = evidence['reference_inputs']
    checks = _validate_spec(spec, ref)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['configuration']['ess_params_file']['sha256'] != H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)):
        failures.append('ESS params file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    if spec['extra'].get('gate_script_sha256') != H.sha256_file(os.path.join(REPO, 'p515_s48_x0_capture_gate.py')):
        failures.append('the gate script sha256 differs from the frozen spec')
    memory = memory_preflight()
    _log(f"[{tag}] memory preflight: available {memory.get('available_gib')} GiB, required {memory['required_gib']} "
         f"GiB -> {'PASS' if memory['pass'] else 'REFUSE'}; vm_stat pages {memory['vm_stat_pages']}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {memory}')
    if failures:
        for failure in failures:
            _log(f'[{tag} PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    _log(f'[{tag}] {LABEL}')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([POINT_LABEL], ctx)
        batch_info = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    rec = next((r for r in records if (r or {}).get('candidate_label') == POINT_LABEL), None)
    point = _point_result(rec)
    guard_failures = PARENT_GUARD.verify(0)
    persisted = point.get('persisted_models') or {}
    persisted_ok = bool(persisted.get('path') and os.path.isfile(os.path.join(REPO, persisted['path']))
                        and H.sha256_file(os.path.join(REPO, persisted['path'])) == persisted.get('sha256'))
    results = {
        'LABEL': LABEL, 'stage': spec['extra']['stage'], 'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'instance': {
            'candidate': 'x = 0 (no storage at nodes 5, 7, 9; 2025 canonical)',
            'candidate_key': point.get('candidate_key'), 'eval_key': point.get('eval_key')},
        'reference_inputs': ref, 'point': point, 'persisted_models_file_hash_verified': persisted_ok,
        'gate': ('evaluated separately by p515_s48_x0_capture_gate.py (zero solves) into '
                 'data/SRP1/Results/P515S48/x0_capture_gate/'),
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
    _log(f"[{tag}] {POINT_LABEL}: status={point['status']} cycles={point['cycles_run']} "
         f"cert_cycle={point['certification_cycle']} Q={point['certified_cost_gross_settlement_excluded']!r} "
         f"exit_code={point['exit_code']}")
    _log(f"[{tag}] post-certification: status={(point['post_certification'] or {}).get('status')} "
         f"persisted={persisted} file_hash_verified={persisted_ok}")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    if guard_failures or point['status'] not in ('certified', 'not_certified'):
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    if point['status'] != 'certified' or (point['post_certification'] or {}).get('status') != 'evaluated' \
            or not persisted_ok or point['exit_code'] != 0:
        _log(f'[{tag}] harness clean; not certified or models not persisted')
        sys.exit(2)
    _log(f'[{tag}] OK: certified and persisted; now run the gate (p515_s48_x0_capture_gate.py)')


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
