"""P5.15 S56 step 4 SMOKE -- a serial 3-cycle smoke per arm per instance through the existing campaign harness, under
ONE frozen addendum spec (`frozen_step4_smoke_spec_v1_<sha8>.json`; cites the diagnostics spec ff466289, the step 3 v3
spec and S56_STEP4_DESIGN.md).

Authority: S56_PLAN_v3.md section 7 rows 3-4; S56_STEP4_DESIGN.md decision 9.2 ("a serial 3-cycle smoke per arm, which
measures RSS, s/cycle and the identities"); TASKS.md S56 section (commit 9c88d89d); the Planner's task of 2026-11-23
(item 2). The step 4 MAIN run is NOT part of this order (held for the Expert's X4-1 / X4-2).

RUNS (8, serial, attached, one per invocation, each into a NEW campaign root `campaign_s56s4smoke_<inst>_<arm>`; cap 3;
candidate x0; alpha 0 -- row 18 is not declared, production's default premium alpha is 0.0, so row 18 is not built):
  arm            coupling option (`interface_coupling`)                           AA
  V1             on, per_scenario_voltage_mode 'per_scenario', row18_charge_reactive False   forced off by production
  V2             on, per_scenario_voltage_mode 'pin',          row18_charge_reactive False   forced off by production
  pinned_aa_off  not declared (off)                                                AA off via the harness override
  pinned_aa_on   not declared (off)                                                case default (keep_memory, on)
Order: instance A (2025|Winter) V1, V2, pinned_aa_off, pinned_aa_on; then instance B ({2025, 2035} x Summer), same.
A run refuses unless every earlier run of the order has completed (its initialisation-identity record exists).

INITIALISATION IDENTITY (S-P3). The harness registers each evaluation's initialisation record in its campaign root's
`init_identity/` and compares it BITWISE with every record already there of the same candidate on the same instance in
the same group (`initialisation_identity_group`: off and on-'pin' = 'voltage_pin_constructed'; on-'per_scenario' =
'voltage_pin_not_constructed'), RAISING before ADMM cycle 1 on a mismatch. Each root is new, so before launching run k
this launcher copies (byte for byte) the init records of the earlier runs of the same instance into the new root's
`init_identity/` -- the harness's documented "reference record the launcher placed there". V1's record is then listed,
never compared (other group); V2 is compared with nothing; pinned_aa_off with V2; pinned_aa_on with V2 and pinned_aa_off.

SOLVES. The launcher parent never solves: SolveProfileGuard(permitted=()) for the whole process, verified at 0. Each
harness child runs `run_admm_arm`, whose armed guard counts every solve; the record's `solve_profile` reconciles it:
base = ((1 + n_dso) x n_years x n_days + n_esso_nodes) x (cycles + 1) -> A (1 x 1 blocks): 7 x 4 = 28; B (2 x 1):
11 x 4 = 44; plus every production recovery retry actually attempted (credited per failure event).

MODES
  --freeze --out-dir D                         the frozen smoke spec (zero solves)
  --run --spec S --instance A|B --arm ARM      one smoke run (stdout / stderr captured by the caller; attached)
  --d3 --spec S                                zero solves: D-3 for 2025|Summer (instance B V1's initialisation)
  --evaluate --spec S                          zero solves: the predictions and the recorded quantities
  --manifest --spec S                          sha256 of every file under the smoke directory
Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_step4_smoke.py --run --spec S --instance A
       --arm V1 > <stdout log> 2> <stderr log>
Objective convention (every Q): EUR of total discounted cost; coupled arms: production's primary
`gross_operational_cost` = settlement excluded in full (pin excluded, salvage separate), the contracted-excluded form
beside it; pinned arms: `gross_operational_cost` = contracted settlement and pin excluded (option off). C_central is the
step 3 gross per-day value (settlement weight 0, no salvage); Q_block - w_b x C_central is EUR of total discounted cost.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 smoke launcher (this process never solves)').install()

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (parent side: no model imports)

SOLVER_PATH = '/usr/local/bin/ipopt'
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
SMOKE_DIR = os.path.join(RES, 'step4_smoke_v1')
SPEC_PREFIX = 'frozen_step4_smoke_spec_v1_'
DIAG_SPEC = os.path.join(RES, 'step4_diag_v1', 'frozen_step4_diag_spec_v2_ff466289.json')
DIAG_RESULTS = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'step4_diag_results.json')
DIAG_CENTRAL = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'central_interface_for_children.json')
DIAG_CHILD_B = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'child_B', 'child_results.json')
DIAG_LAUNCHER = 'p515_s56_step4_diag.py'
S3V3_SPEC = os.path.join(RES, 'step3_v3', 'frozen_step3_spec_v3_b8d07450.json')
S3V3_RESULTS = os.path.join(RES, 'step3_v3', 'run_v1', 'step3_v3_results.json')
DESIGN = 'S56_STEP4_DESIGN.md'
INSTANCE_A_RECORD = os.path.join(RES, 'step4_instances_v1', 'step4_instances.json')
INSTANCE_B_RECORD = os.path.join(RES, 'step4_instances_v2', 'step4_instance_b.json')
COUPLING_GATES = 'p515_s56_coupling_gates.py'   # read by AST only (importing it arms a permitted=() guard)
CG_NAMES = ('CASE_FILE_AA', 'CONVERGENCE_DEPTH_TAIL', 'ESS_AGEING_BASELINE', 'ESS_AGEING_LABEL', 'INVESTMENT_YEAR',
            'REQUIRED_CONSECUTIVE_CYCLES', 'COUPLING_V1_NOQ', 'FIXED_SIGMA')
CAP = 3
X0 = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
INSTANCES = ('A', 'B')
ARMS = ('V1', 'V2', 'pinned_aa_off', 'pinned_aa_on')
ORDER = [(i, a) for i in INSTANCES for a in ARMS]
COUPLED = ('V1', 'V2')
GiB = 1024.0 ** 3

PREDICTIONS = {
    'S-P1': 'every run exits 0 with 3 cycles; the solve-count identity holds.',
    'S-P2': ('F1 arms, every cycle: production `priced_residual_pr` price part + `interface_settlement_total` = 0 (rel '
             '1e-9); `gross_operational_cost_contracted_excluded` - `gross_operational_cost` = interface settlement '
             'deviation total (rel 1e-9); Sigma_s omega_s mu_s = 0 per coordinate (abs 1e-9 x max|mu|); V-1 voltage pin '
             'total == 0.'),
    'S-P3': ('initialisation, per instance: pinned AA-off == pinned AA-on == V-2, bitwise (the harness grouping check '
             'passes); V-1 is in its own group and differs.'),
    'S-P4': ('on real cycles, in the coupled arms: `assert_s56_cycle_capture_path` holds; per-block Q from the capture '
             'sums to production\'s per-cycle `gross_operational_cost` within rel 1e-12.'),
    'S-P5': ('peak RSS <= 6 GiB per run. Record s/cycle per run; the Advisor\'s inference was 45-75 s for two blocks. '
             'Recording only.'),
}
PREDICTIONS_OPERATIONAL = {
    'S-P1': ('per run: harness exit code 0 (parent_view.exit_code and exit_code.txt), record cycles_run == 3, '
             'per_cycle_record.jsonl has 3 lines; solve_profile.identity_holds is True, solve_profile.base_solves == '
             'the declared base (A 28, B 44), blocked_solve == blocked_exec == 0; the launcher parent guard verify(0)'),
    'S-P2': ('coupled arms (V1, V2), every cycle c = 1..3, all with rel = |a - b| / max(|a|, |b|, 1):\n'
             '(i) T_c = sum over every agent block (TSO and each DSO) of w x settlement_total_unweighted from the S56 '
             'capture (production `_get_local_interface_settlement(part=\'total\')` x `_get_admm_block_weight` -- the '
             'terms production\'s `_get_operational_interface_settlement_blocks` sums); rel(priced_residual_price_part_c, '
             '-T_c) <= 1e-9. At c = 3 also against the record\'s production `recourse_components.interface_settlement_'
             'total`.\n'
             '(ii) D_c = sum over agent blocks of w x (settlement_total_unweighted - settlement_contracted_unweighted) '
             '(S56 capture); rel(gross_operational_cost_contracted_excluded_c - gross_operational_cost_c, D_c) <= 1e-9 '
             '(per_cycle_record fields). The production deviation part (`part=\'deviation\'`) is not recorded per cycle '
             'for the TSO: the decomposition total - contracted == deviation is checked per cycle on the DSO blocks '
             '(rel(sum_DSO w(total - contracted), per_cycle_response covariance_dso_weighted) <= 1e-9) and at c = 3 on '
             'every block (rel(D_3, record interface_settlement_deviation_total) <= 1e-9 and rel(LHS_3, the same) <= '
             '1e-9).\n'
             '(iii) every deviation-dual coordinate (agent, kind, DN block, period) of interface_duals_per_cycle.jsonl '
             '(`lambda_dev_<kind>_<agent>`, scenario order = header deviation_channel.scenarios): |sum_s omega_s mu_s| '
             '<= 1e-9 x max|mu| (max over that cycle, agent and kind); omega_s = prob_market[s_m] x prob_operation[s_o] '
             'of the block (multiscenario_terminal per_scenario_costs probability of the TSO block).\n'
             '(iv) V1 only: the record\'s terminal recourse_components.voltage_pin_total == 0.0 and voltage_pin_tso == '
             '0.0 and every voltage_pin_dso value == 0.0 (production `_get_local_voltage_pin` returns 0.0 when '
             '`scenario_voltage_pin` is absent, which it is under \'per_scenario\'); production records no per-cycle '
             'pin total, so cycles 1-2 are covered by the structural reading, not by a recorded value'),
    'S-P3': ('per instance: the evaluation record initialisation_identity of pinned_aa_off has n_compared >= 1, every '
             'compared label == V2 (and pinned_aa_on: V2 and pinned_aa_off), all_equal True, no mismatch; V2 '
             'n_compared == 0 with V1 listed under other_candidates_not_compared; V1 group == '
             '\'voltage_pin_not_constructed\', the others \'voltage_pin_constructed\' (pinned records carry no group: '
             'option off -> the harness default group); and V1 gross_operational_cost_hex != V2\'s'),
    'S-P4': ('coupled arms: record s56_cycle_capture_checklist_asserted_before_run all True; s56_cycle_capture_summary '
             'ok with 3 cycle lines; per cycle: rel(sum over (year, day) of q_blocks.per_year_day[*]'
             '.settlement_excluded_in_full, per_cycle_record gross_operational_cost) <= 1e-12'),
    'S-P5': ('per run: record peak_rss.child_python_process_ru_maxrss <= 6 GiB (6 x 2^30 bytes); the IPOPT subprocess '
             'maximum (solver_subprocesses_max_ru_maxrss) is reported beside it. s/cycle = per_cycle_response '
             'cycle_wall_s per cycle, recorded only'),
}
RECORDED_ONLY = ('per cycle and per block, Q_block - w_b x C_central in EUR of total discounted cost, for every block '
                 'with a C_central; w_b = the instance\'s own block weight. Q_block: coupled arms -- the S56 capture '
                 'q_blocks.per_year_day (settlement_excluded_in_full, contracted_excluded beside it) every cycle; pinned '
                 'arms -- instance A (one block): the per-cycle gross_operational_cost (contracted-excluded, option '
                 'off); instance B: per block at the terminal cycle only (multiscenario_terminal '
                 'block_recourse_production_local summed over the block\'s agents) -- option-off runs carry no '
                 'per-block per-cycle capture')
FAILURE_POLICY = ('Failure of any prediction is recorded, never re-run or tuned. A harness fault (exception, guard '
                  'mismatch, child exit != 0) is recorded and the order stops for the Planner; an initialisation-'
                  'identity mismatch raised by the harness is such a stop (it is the S-P3 check). The step 4 main '
                  'run, any cap above 3, concurrent or detached runs and undeclared solves are not permitted.')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 smoke] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def committed_clean(rel):
    return bool(_git('ls-files', '--', rel).strip()) and not _git('status', '--porcelain', '--', rel).strip()


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


def _load(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(x) for x in handle if x.strip()]


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def other_project_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found = os.getpid(), []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(
                re.match(r'p5\S*\.py$', os.path.basename(t)) for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


def coupling_gates_literals():
    with open(COUPLING_GATES, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in CG_NAMES:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = sorted(set(CG_NAMES) - set(out))
    if missing:
        raise RuntimeError(f'{COUPLING_GATES}: literals not found {missing}')
    return out


def root_name(inst, arm):
    return f's56s4smoke_{inst}_{arm}'


def root_path(inst, arm):
    return os.path.join(SMOKE_DIR, f'campaign_{root_name(inst, arm)}')


def arm_options(arm, cg):
    v1 = dict(cg['COUPLING_V1_NOQ'])
    if arm == 'V1':
        return {'interface_coupling': v1}
    if arm == 'V2':
        v2 = dict(v1)
        v2['per_scenario_voltage_mode'] = 'pin'
        return {'interface_coupling': v2}
    if arm == 'pinned_aa_off':
        return {'overrides': {'anderson_acceleration': {'enabled': False}}}
    if arm == 'pinned_aa_on':
        return {}
    raise ValueError(arm)


# ======================================================================================================================
#  capture-path checklist (rule eleven): every quantity the spec requires has a capture path, BEFORE any launch
# ======================================================================================================================
def capture_path_checklist():
    import inspect
    import interface_dual_capture as IDC
    src = inspect.getsource(H)
    checks = {
        'per_cycle_fields_priced_residual_and_both_q': all(f in H.PER_CYCLE_INTERFACE_COUPLING_FIELDS for f in (
            'priced_residual_price_part', 'gross_operational_cost_contracted_excluded', 'interface_coupling_aa_forced_off')),
        'per_cycle_trajectory_gross': 'gross_operational_cost' in H.PER_CYCLE_TRAJECTORY_FIELDS,
        'per_cycle_response_wall_and_dso_deviation': all(f in H.PER_CYCLE_RESPONSE_FIELDS for f in (
            'cycle_wall_s', 'covariance_dso_weighted', 'rss_bytes')),
        's56_capture_fields': all(f in H.S56_CYCLE_CAPTURE_LINE_FIELDS for f in ('q_blocks', 'cycle', 'captured')),
        's56_capture_per_agent_settlement_parts': ("'settlement_total_unweighted': float(total)" in src
                                                    and "'settlement_contracted_unweighted': float(contracted)" in src),
        's56_capture_assertion_exists': callable(getattr(H, 'assert_s56_cycle_capture_path', None)),
        'deviation_duals_recorded': callable(getattr(IDC, 'read_deviation_duals', None))
                                     and "'scenarios'" in inspect.getsource(IDC),
        'record_fields': all(s in src for s in ("'s56_cycle_capture_summary'", "'initialisation_identity'",
                                                "'solve_profile'", 'peak_rss = {', "'recourse_components'")),
        'init_identity_grouping': callable(getattr(H, 'initialisation_identity_group', None))
                                  and getattr(H, 'INIT_IDENTITY_DIR_NAME', None) == 'init_identity',
        'multiscenario_block_recourse_and_probabilities': (
            "'block_recourse_production_local': local_q" in src and "'per_scenario_costs': costs" in src
            and "local_q = block_components[(kind, node_id, year, day)] / weight" in src
            and "['probability'] = omega_m * omega_o" in open('network.py', encoding='utf-8').read()),
        'interface_anchor_in_s31c_detail': True,
    }
    # production source TEXT (the parent imports no model code): the anchor field, the one place pc is fixed in the
    # distributed path, and the pin's zero when absent
    with open('shared_resources_planning.py', encoding='utf-8') as handle:
        srp_text = handle.read()
    checks['interface_anchor_in_s31c_detail'] = ("'anchor_p_mw': anchor_p_by_scenario" in srp_text
                                                  and 'pe.value(local_tso_model.pc[adn_load_idx, s_m, s_o, p]) * s_base'
                                                  in srp_text)
    checks['pc_fixed_once_at_tso_creation'] = (srp_text.count('fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, '
                                                              's_o, p], interface_pf_p)') == 1)
    checks['voltage_pin_zero_when_absent'] = ("    if not hasattr(model, 'scenario_voltage_pin'):\n        return 0.0"
                                              in srp_text)
    checks['s31c_detail_written_with_reporting_detail'] = "'interface_reporting_detail': reporting_detail" in open(
        'p515_g_g1_g4_admm_gates.py', encoding='utf-8').read()
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise RuntimeError(f'capture path missing (before any launch): {failed}')
    return checks


# ======================================================================================================================
#  the frozen spec
# ======================================================================================================================
def _instance_info():
    a, b = _load(INSTANCE_A_RECORD), _load(INSTANCE_B_RECORD)
    return {
        'A': {'record': {'path': INSTANCE_A_RECORD, 'sha256': _sha(INSTANCE_A_RECORD)},
              'derived_instance': a['blocks']['2025|Winter']['derived_instance_declaration'],
              'blocks': ['2025|Winter'],
              'w_b_own': {'2025|Winter': a['blocks']['2025|Winter']['G2_weights']['w_b_derived']},
              'n_years': 1, 'n_days': 1},
        'B': {'record': {'path': INSTANCE_B_RECORD, 'sha256': _sha(INSTANCE_B_RECORD)},
              'derived_instance': b['derived_instance_declaration'],
              'blocks': ['2025|Summer', '2035|Summer'],
              'w_b_own': {bk: b['blocks'][bk]['G2_weights']['w_b_derived'] for bk in ('2025|Summer', '2035|Summer')},
              'n_years': 2, 'n_days': 1},
    }


def build_spec():
    cg = coupling_gates_literals()
    diag = _load(DIAG_RESULTS)
    s3v3 = _load(S3V3_RESULTS)
    inst = _instance_info()
    c_s = s3v3.get('central') or {}
    central = {
        '2025|Winter': {'C_central_eur_per_day': diag['central']['2025|Winter']['C_central'],
                        'source': f'{DIAG_RESULTS} central.2025|Winter (run_v1 singles + S3-1 single 02)'},
        '2035|Summer': {'C_central_eur_per_day': diag['central']['2035|Summer']['C_central'],
                        'source': f'{DIAG_RESULTS} central.2035|Summer (step 3 run_v1 singles)'},
        '2025|Summer': {'C_central_eur_per_day': c_s.get('C_central') if c_s.get('complete') else None,
                        'source': f'{S3V3_RESULTS} central',
                        'status': ('complete' if c_s.get('complete') else
                                   f"UNDEFINED: step 3 v3 single(s) {c_s.get('not_optimal')} not Optimal")},
    }
    for bk, c in central.items():
        c['C_central_hex'] = float.hex(c['C_central_eur_per_day']) if c['C_central_eur_per_day'] is not None else None
    n_dso, n_esso = 3, 3
    solves = {}
    for i in INSTANCES:
        per_round = (1 + n_dso) * inst[i]['n_years'] * inst[i]['n_days'] + n_esso
        solves[i] = {'per_round': per_round, 'rounds': CAP + 1, 'base_per_run': per_round * (CAP + 1)}
    solves['total_base_8_runs'] = 4 * solves['A']['base_per_run'] + 4 * solves['B']['base_per_run']
    configuration = {
        'arm_label': 's39_D',
        'case_file_anderson_acceleration': dict(cg['CASE_FILE_AA']),
        'convergence_depth_tail': dict(cg['CONVERGENCE_DEPTH_TAIL']),
        'ess_ageing_baseline': deepcopy(cg['ESS_AGEING_BASELINE']),
        'ess_ageing_baseline_label': cg['ESS_AGEING_LABEL'],
        'overrides': {},
        'note': ('S56 step 4 smoke: the G-f / step 4 build configuration (p515_s56_coupling_gates.py literals) on the '
                 'derived instance; no row 18 declaration (production default alpha 0.0), no model variant, no flex '
                 'variant, no NRF, no rho0 scale, no post-certification'),
    }
    return {
        'stage': ('P5.15 S56 step 4 smoke: serial 3-cycle smoke per arm (V-1, V-2, pinned AA off, pinned AA on) on '
                  'instance A (2025|Winter) and instance B ({2025, 2035} x Summer); D-3 for 2025|Summer (frozen '
                  'addendum spec v1)'),
        'version': 1, 'kind': 'addendum',
        'predecessor': {'path': DIAG_SPEC, 'sha256': _sha(DIAG_SPEC),
                        'relation': 'the step 3 v2 + step 4 diagnostics spec; this addendum adds the smoke runs and '
                                    'D-3 for 2025|Summer; nothing in it is edited'},
        'cites': {'design': {'path': DESIGN, 'sha256': _sha(DESIGN), 'item': 'decision 9.2 (serial 3-cycle smoke per '
                                                                            'arm)'},
                  'step3_v3_spec': {'path': S3V3_SPEC, 'sha256': _sha(S3V3_SPEC)},
                  'step3_v3_results': {'path': S3V3_RESULTS, 'sha256': _sha(S3V3_RESULTS)},
                  'diag_results': {'path': DIAG_RESULTS, 'sha256': _sha(DIAG_RESULTS)},
                  'diag_central_interface': {'path': DIAG_CENTRAL, 'sha256': _sha(DIAG_CENTRAL)},
                  'diag_child_B_results': {'path': DIAG_CHILD_B, 'sha256': _sha(DIAG_CHILD_B)},
                  'diag_launcher': {'path': DIAG_LAUNCHER, 'sha256': _sha(DIAG_LAUNCHER),
                                    'reused_by_import_in_d3': ['_init_round (trial=True: every optimize stubbed, zero '
                                                               'solves)', 'd3_boxes', '_voltage_rows_box',
                                                               '_box_intersect', '_sqrt_or_none']}},
        'authority': ['S56_PLAN_v3.md section 7 rows 3-4', 'S56_STEP4_DESIGN.md decision 9.2',
                      'TASKS.md S56 section, commit 9c88d89d (step 3 v3 + step 4 smoke active; main run held)',
                      'Planner task 2026-11-23, item 2 (step 4 smoke)'],
        'launcher': SCRIPT,
        'harness': {'path': 'p515_s44_campaign_harness.py', 'sha256': _sha('p515_s44_campaign_harness.py')},
        'coupling_gates_literals_source': {'path': COUPLING_GATES, 'sha256': _sha(COUPLING_GATES)},
        'candidate': {'label': 'x0', 'map': {str(k): list(v) for k, v in X0.items()},
                      'investment_year': cg['INVESTMENT_YEAR']},
        'alpha': {'value': 0.0, 'how': ('interface_deviation_premium not declared: admm_parameters default '
                                        "{'alpha': 0.0}; model_construction_helpers.add_scenario_commitment_terms "
                                        'builds row 18 only when premium_alpha > 0, so row18_charge_reactive is inert')},
        'configuration': configuration,
        'cap': CAP, 'concurrency': 1, 'required_consecutive_cycles': cg['REQUIRED_CONSECUTIVE_CYCLES'],
        'sigma_fixed_expected': cg['FIXED_SIGMA'],
        'arms': {a: {'options': arm_options(a, cg),
                     'aa': {'V1': 'forced off by production under the option', 'V2': 'forced off by production under '
                            'the option', 'pinned_aa_off': 'off via the harness AA override',
                            'pinned_aa_on': 'case default (keep_memory, enabled)'}[a],
                     'init_identity_group': ('voltage_pin_not_constructed' if a == 'V1' else 'voltage_pin_constructed')}
                 for a in ARMS},
        'instances': inst,
        'order': [{'run': k + 1, 'instance': i, 'arm': a, 'campaign_id': root_name(i, a),
                   'campaign_root': root_path(i, a),
                   'init_identity_seeded_from': [root_name(i, x) for x in ARMS[:ARMS.index(a)]]}
                  for k, (i, a) in enumerate(ORDER)],
        'solves': {'per_instance': solves, 'launcher_parent': 0, 'd3': 0, 'evaluate': 0,
                   'derivation': ('p515_g_g1_g4_admm_gates.run_admm_arm solve_profile: base = ((1 + n_dso) x n_years x '
                                  'n_days + n_esso_nodes) x (cycles_run + 1); n_dso 3, n_esso 3 (nodes 5, 7, 9); A: 1 '
                                  'year x 1 day -> 7 per round; B: 2 years x 1 day -> 11 per round; cap 3 -> 4 rounds; '
                                  'plus every recovery retry actually attempted (production recovery unchanged in '
                                  'these runs), credited per failure event by the identity'),
                   'declared_base_per_run': {i: solves[i]['base_per_run'] for i in INSTANCES}},
        'central': central, 'sum_12_w': _load(INSTANCE_A_RECORD)['sum12']['sum_12'],
        'predictions': PREDICTIONS, 'predictions_operational': PREDICTIONS_OPERATIONAL,
        'recorded_only': RECORDED_ONLY,
        'thresholds': {'S-P2_rel': 1e-9, 'S-P2_gauge_abs_factor': 1e-9, 'S-P4_rel': 1e-12, 'S-P5_rss_bytes': 6 * GiB},
        'd3_2025_summer': {
            'method': ('zero solves. (1) The ADMM-prepared models of instance B under V-1 are rebuilt with the '
                       'diagnostics launcher\'s initialisation round in its trial form (every optimize stubbed, sigma '
                       'replaced by sigma_fixed; stopped at _build_pristine_snapshot_bases): the static boxes (TSO '
                       'interface_delta +- rating, TSO ADN-bus V bounds and the fixed voltage slacks, DSO reference-'
                       'generator pg / qg bounds, DSO reference-node V bounds) are build-time and do not depend on the '
                       'solution. (2) The only solution-dependent quantity, the TSO ADN-load pc / qc (fixed once, at TSO '
                       'model creation, at the DSO initial expectation; never changed in the distributed path), is set '
                       'from the V1 instance-B smoke run\'s own value: its interface_settlement_detail_s31c.json '
                       'interface_reporting_detail anchor_p_mw / anchor_q_mvar (production `_get_interface_reporting_'
                       'detail`: pc x s_base, read on the terminal models) divided by s_base. (3) The diagnostics '
                       'launcher\'s d3_boxes (by import) on complete inputs; a copy of it with a scenario filter for '
                       'the 8 available singles'),
            'central_values': ('2025|Summer has no complete central point (step 3 v3 single 12 not Optimal). PRIMARY: '
                               'the 8 Optimal singles (v2 S3-2 00, 01, 02, 10, 11, 20, 21; v3 22), reported as '
                               'INCOMPLETE (8 of 9 scenarios). SUPPLEMENTARY, labelled: the v2 S3-2 merged-default '
                               'per-scenario interface values (all 9 scenarios; the merged solution, not the central '
                               'singles)'),
            'method_gate': ('recorded, not a prediction: the same construction applied to 2035|Summer with the diag '
                            'run_v1 central interface values reproduces run_v1 child B\'s D-3 2035|Summer result '
                            '(n_values, n_outside, n_outside_strict, every family min_margin) -- equality expected '
                            'when the smoke\'s initialisation equals the diag run\'s; and the filtered copy equals '
                            'd3_boxes on complete inputs'),
            'bound_tol_rel': 1e-8,
            'bound_tol_disclosure': ('the same 1e-8 x max(1, |bound|) tolerance as D-3 in ff466289 (IPOPT\'s default '
                                     'bound_relax_factor); it was chosen after a zero-solve draft trial had shown strict '
                                     'margins of -1.9e-10 pu (V at 1.1 pu); the Planner ruled it acceptable '
                                     '(TASKS.md, 9c88d89d)')},
        'failure_policy': FAILURE_POLICY,
        'not_permitted': ['production edits', 'the step 4 main run', 'any cycle cap above 3', 'any undeclared solve',
                          'concurrent runs', 'detached runs', 'overwriting committed artifacts'],
        'objective_convention': ('Q in EUR of total discounted cost. Coupled arms: primary gross_operational_cost = '
                                 'settlement excluded in full (pin excluded, salvage separate), contracted-excluded '
                                 'beside it. Pinned arms: gross_operational_cost = contracted settlement and pin '
                                 'excluded. C_central: step 3 gross EUR/day (settlement weight 0, no salvage); '
                                 'Q_block - w_b C_central in EUR of total discounted cost'),
        'outputs': {'runs': 'campaign_s56s4smoke_<inst>_<arm>/ (harness artifacts) + logs/run_<k>_<inst>_<arm>_'
                            '{stdout,stderr}.log + run_<k>_<inst>_<arm>_launcher.json',
                    'd3': 'd3_2025_Summer_<utc>.json', 'evaluation': 'smoke_evaluation_<utc>.json',
                    'manifest': 'step4_smoke_v1_manifest_sha256.json (sha256 of every file under the smoke dir)'},
    }


def freeze(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith(SPEC_PREFIX)]
    if existing:
        _log(f'REFUSED: a spec already exists: {existing}')
        return 2
    spec = build_spec()
    GRIO.check(spec)
    text = json.dumps(spec, indent=1, sort_keys=True, default=str)
    sha = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(out_dir, f'{SPEC_PREFIX}{sha[:8]}.json')
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)
    _log(f'wrote {os.path.relpath(path, REPO)} (sha256 {sha}); guard {GUARD.counts}')
    return 0 if not GUARD.verify(0) else 1


def _spec(args):
    spec = _load(args.spec)
    sha = _sha(args.spec)
    tag = os.path.basename(args.spec)[len(SPEC_PREFIX):].split('.')[0]
    if not sha.startswith(tag):
        raise RuntimeError(f'spec sha256 {sha[:8]} does not match its name tag {tag}')
    return spec, sha, os.path.relpath(os.path.abspath(args.spec), REPO)


def _init_record_name(inst, arm):
    d = os.path.join(root_path(inst, arm), H.INIT_IDENTITY_DIR_NAME)
    if not os.path.isdir(d):
        return None
    ev = os.path.join(root_path(inst, arm), 'evals')
    own = sorted(os.listdir(ev)) if os.path.isdir(ev) else []
    if len(own) != 1:
        return None
    name = f'{own[0]}.json'
    return name if os.path.isfile(os.path.join(d, name)) else None


# ======================================================================================================================
#  one smoke run
# ======================================================================================================================
def run(args):
    t0 = time.time()
    spec, spec_sha, spec_rel = _spec(args)
    inst, arm = args.instance, args.arm
    k = ORDER.index((inst, arm)) + 1
    tag = f'run_{k}_{inst}_{arm}'
    rec = {'schema': 'p515_s56_step4_smoke_launcher_v1', 'run': k, 'instance': inst, 'arm': arm, 'utc_start': _utc(),
           'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT, 'script_sha256': _sha(SCRIPT),
           'git_head': _git('rev-parse', 'HEAD').strip(), 'interpreter': sys.executable,
           'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'), 'refusals': [], 'error': None}
    rec_path = os.path.join(SMOKE_DIR, f'{tag}_launcher.json')
    if os.path.exists(rec_path):
        _log(f'REFUSED: {rec_path} exists (write-once)')
        return 1
    ref = rec['refusals']
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        ref.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
    for rel in (SCRIPT, spec_rel, 'p515_s44_campaign_harness.py'):
        if not committed_clean(rel):
            ref.append(f'{rel} not committed clean')
    others = other_project_processes()
    if others:
        ref.append(f'other project processes running: {others}')
    root = root_path(inst, arm)
    if os.path.exists(root):
        ref.append(f'campaign root exists: {root}')
    seeds = []
    for earlier in ARMS[:ARMS.index(arm)]:
        name = _init_record_name(inst, earlier)
        if name is None:
            ref.append(f'earlier run {inst}/{earlier} has no initialisation-identity record (order not respected)')
        else:
            seeds.append((earlier, os.path.join(root_path(inst, earlier), H.INIT_IDENTITY_DIR_NAME, name), name))
    try:
        rec['capture_path_checklist'] = capture_path_checklist()
    except Exception as error:  # noqa: BLE001
        ref.append(f'capture-path checklist: {error}')
    for name in ('cites',):
        for key, item in spec[name].items():
            if 'sha256' in item and _sha(item['path']) != item['sha256']:
                ref.append(f'{key}: {item["path"]} sha256 differs from the spec')
    if _sha('p515_s44_campaign_harness.py') != spec['harness']['sha256']:
        ref.append('harness sha256 differs from the spec')
    if ref:
        _log(f'REFUSED before launch (zero solves): {ref}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        os.makedirs(SMOKE_DIR, exist_ok=True)
        write_json(rec_path, rec)
        return 3
    cfg = deepcopy(spec['configuration'])
    cfg['name'] = (f"S56 step 4 smoke run {k}: instance {inst} ({', '.join(spec['instances'][inst]['blocks'])}), arm "
                   f"{arm}; x0, alpha 0, cap {spec['cap']}")
    cfg['derived_instance'] = H.validate_derived_instance(spec['instances'][inst]['derived_instance'])
    label = f'x0_{arm}'
    options = deepcopy(spec['arms'][arm]['options'])
    failures = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT, spec_rel))
    if failures:
        rec['refusals'] = failures
        _log(f'REFUSED by the harness preconditions (zero solves): {failures}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        write_json(rec_path, rec)
        return 3
    spec_path, csha, cspec = H.freeze_campaign_spec(
        root, root_name(inst, arm), [(label, X0, options)], cfg, spec['cap'], 1, spec['authority'],
        required_consecutive_cycles=spec['required_consecutive_cycles'],
        extra={'smoke_spec': {'path': spec_rel, 'sha256': spec_sha}, 'run': k, 'instance': inst, 'arm': arm,
               'campaign_script': SCRIPT, 'campaign_script_sha256': _sha(SCRIPT),
               'objective_convention': spec['objective_convention']})
    entry = cspec['candidates'][0]
    rec['campaign'] = {'root': root, 'spec': {'path': os.path.relpath(spec_path, REPO), 'sha256': csha},
                       'entry': entry}
    # S-P3: the earlier runs' initialisation records, byte for byte (the harness's "reference record the launcher
    # placed there")
    idir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    os.makedirs(idir)
    rec['init_identity_seeded'] = []
    for earlier, src, name in seeds:
        dst = os.path.join(idir, name)
        shutil.copyfile(src, dst)
        rec['init_identity_seeded'].append({'from_arm': earlier, 'file': name, 'sha256_src': _sha(src),
                                            'sha256_dst': _sha(dst)})
    lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
    _log(f'run {k} ({inst}, {arm}): lock {lock}; campaign spec {os.path.relpath(spec_path, REPO)}; seeded '
         f'{[s[0] for s in seeds]}')
    records = None
    try:
        ctx = H.CampaignContext(root, spec_path, csha, cspec, log=_log)
        records = H.evaluate([label], ctx)
    except BaseException as error:  # noqa: BLE001 -- a harness fault, recorded
        rec['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f'HARNESS FAULT: {rec["error"]["error"]}')
    finally:
        rec['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
    r = (records or [None])[0] or {}
    rec['result'] = {'status': r.get('status'), 'cycles_run': r.get('cycles_run'),
                     'exit_code': (r.get('parent_view') or {}).get('exit_code'),
                     'wall_s_parent_view': (r.get('parent_view') or {}).get('wall_s'),
                     'solve_profile': r.get('solve_profile'), 'peak_rss': r.get('peak_rss'),
                     'eval_dir': os.path.join(root, 'evals', entry['eval_dir'])}
    rec['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    rec['wall_s'] = time.time() - t0
    rec['utc_end'] = _utc()
    write_json(rec_path, rec)
    ok = (rec['error'] is None and rec['result']['exit_code'] == 0 and not rec['guard']['verify_failures'])
    _log(f"run {k} ({inst}, {arm}): exit {rec['result']['exit_code']} cycles {rec['result']['cycles_run']} status "
         f"{rec['result']['status']} solves {(rec['result']['solve_profile'] or {}).get('observed')} "
         f"identity {(rec['result']['solve_profile'] or {}).get('identity_holds')} wall {rec['wall_s']:.0f}s -> {rec_path}")
    return 0 if ok else 2


# ======================================================================================================================
#  D-3 for 2025|Summer (zero solves)
# ======================================================================================================================
def _eval_dir(inst, arm):
    ev = os.path.join(root_path(inst, arm), 'evals')
    dirs = sorted(os.listdir(ev)) if os.path.isdir(ev) else []
    if len(dirs) != 1:
        raise RuntimeError(f'expected one eval dir under {ev}: {dirs}')
    return os.path.join(ev, dirs[0])


def d3_boxes_filtered(planning, tso_model, dso_models, central, blocks, bound_tol_rel, scenarios):
    """A copy of p515_s56_step4_diag.d3_boxes (same box construction, margins and tolerance rule) restricted to the
    listed scenario keys; verified equal to d3_boxes on complete inputs (method gate)."""
    import pyomo.environ as pe
    import p515_s56_step4_diag as DG
    tn = planning.transmission_network
    out = {}
    for y, d in blocks:
        bk = f'{y}|{d}'
        c = central.get(bk) or {}
        rec = {'complete': bool(c.get('complete')), 'scenarios_checked': sorted(scenarios), 'per_family': {},
               'n_values': 0, 'n_outside': 0, 'n_outside_strict': 0, 'bound_tol_rel': bound_tol_rel,
               'min_margin': None, 'min_relative_margin': None}
        tb = tso_model[y][d]
        tnet = tn.network[y][d]
        sb_t = tnet.baseMVA
        fam = rec['per_family']

        def add(family, value, box, where, unit):
            lo, hi = box
            m_ = min(value - lo, hi - value)
            half = (hi - lo) / 2.0 if math.isfinite(hi - lo) else math.inf
            e = fam.setdefault(family, {'unit': unit, 'n': 0, 'n_outside': 0, 'n_outside_strict': 0,
                                        'min_margin': math.inf, 'min_margin_at': None,
                                        'min_relative_margin': math.inf, 'box_example': None})
            e['n'] += 1
            rec['n_values'] += 1
            if m_ < 0:
                e['n_outside_strict'] += 1
                rec['n_outside_strict'] += 1
                bound = lo if value - lo < hi - value else hi
                if -m_ > bound_tol_rel * max(1.0, abs(bound)):
                    e['n_outside'] += 1
                    rec['n_outside'] += 1
            if m_ < e['min_margin']:
                e['min_margin'], e['min_margin_at'], e['box_example'] = m_, where, [lo, hi, value]
            rel = m_ / half if half not in (0.0, math.inf) else (math.inf if half == math.inf else -math.inf)
            e['min_relative_margin'] = min(e['min_relative_margin'], rel)
        for dn_i, node_id in enumerate(tn.active_distribution_network_nodes):
            adn_load = tnet.get_adn_load_idx(node_id)
            adn_node = tnet.get_node_idx(node_id)
            dnet = planning.distribution_networks[node_id].network[y][d]
            db = dso_models[node_id][y][d]
            sb_d = dnet.baseMVA
            ref_gen = dnet.get_reference_gen_idx()
            ref_node = dnet.get_node_idx(dnet.get_reference_node_id())
            for (s_m, s_o) in [(a, b) for a in tb.scenarios_market for b in tb.scenarios_operation]:
                if f'{s_m}{s_o}' not in scenarios:
                    continue
                sc = c['interface'][f'{s_m}{s_o}'][str(node_id)]
                for p in tb.periods:
                    where = f'DN{node_id} s{s_m}{s_o} t{p}'
                    P, Q, V = sc['P_mw'][p], sc['Q_mvar'][p], sc['V_pu'][p]
                    pc = pe.value(tb.pc[adn_load, s_m, s_o, p])
                    qc = pe.value(tb.qc[adn_load, s_m, s_o, p])
                    dp, dq = tb.interface_delta_p[dn_i, s_m, s_o, p], tb.interface_delta_q[dn_i, s_m, s_o, p]
                    add('TSO_P_delta_box', P, [(pc + dp.lb) * sb_t, (pc + dp.ub) * sb_t], where, 'MW')
                    add('TSO_Q_delta_box', Q, [(qc + dq.lb) * sb_t, (qc + dq.ub) * sb_t], where, 'MVAr')
                    vv, vs = tb.vmag[adn_node, s_m, s_o, p], tb.vmag_sqr[adn_node, s_m, s_o, p]
                    rows, _note = DG._voltage_rows_box(tb, tnet, adn_node, s_m, s_o, p, tn.params)
                    add('TSO_V_box', V, DG._box_intersect([vv.lb, vv.ub], [DG._sqrt_or_none(vs.lb),
                                                                            DG._sqrt_or_none(vs.ub)], rows),
                        where, 'pu')
                    g_p, g_q = db.pg[ref_gen, s_m, s_o, p], db.qg[ref_gen, s_m, s_o, p]
                    add('DSO_P_refgen_box', P, [None if g_p.lb is None else g_p.lb * sb_d,
                                                None if g_p.ub is None else g_p.ub * sb_d], where, 'MW')
                    add('DSO_Q_refgen_box', Q, [None if g_q.lb is None else g_q.lb * sb_d,
                                                None if g_q.ub is None else g_q.ub * sb_d], where, 'MVAr')
                    wv, ws = db.vmag[ref_node, s_m, s_o, p], db.vmag_sqr[ref_node, s_m, s_o, p]
                    rows_d, _n = DG._voltage_rows_box(db, dnet, ref_node, s_m, s_o, p,
                                                      planning.distribution_networks[node_id].params)
                    add('DSO_V_box', V, DG._box_intersect([wv.lb, wv.ub], [DG._sqrt_or_none(ws.lb),
                                                                            DG._sqrt_or_none(ws.ub)], rows_d),
                        where, 'pu')
        for e in fam.values():
            for kk in ('min_margin', 'min_relative_margin'):
                if e[kk] == math.inf:
                    e[kk] = None
        finite = [e['min_margin'] for e in fam.values() if e['min_margin'] is not None]
        rel = [e['min_relative_margin'] for e in fam.values() if e['min_relative_margin'] is not None]
        rec['min_margin'] = {f: {'value': e['min_margin'], 'unit': e['unit'], 'at': e['min_margin_at']}
                             for f, e in fam.items()}
        rec['min_relative_margin'] = min(rel) if rel else None
        rec['min_margin_any_unit'] = min(finite) if finite else None
        out[bk] = rec
    return out


def _box_summary(b):
    return {'n_values': b.get('n_values'), 'n_outside': b.get('n_outside'), 'n_outside_strict': b.get('n_outside_strict'),
            'min_margin': b.get('min_margin'), 'min_relative_margin': b.get('min_relative_margin')}


def d3(args):
    t0 = time.time()
    spec, spec_sha, spec_rel = _spec(args)
    out = {'schema': 'p515_s56_step4_smoke_d3_v1', 'utc_start': _utc(), 'spec': {'path': spec_rel, 'sha256': spec_sha},
           'script': SCRIPT, 'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'method': spec['d3_2025_summer'], 'error': None}
    path = os.path.join(SMOKE_DIR, f'd3_2025_Summer_{_stamp()}.json')
    try:
        for key, item in spec['cites'].items():
            if 'sha256' in item and _sha(item['path']) != item['sha256']:
                raise RuntimeError(f'{key}: {item["path"]} sha256 differs from the spec')
        import p515_s56_step4_diag as DG   # installs its own guard (its sites) and blocks pickle; ours stays permitted=()
        import p515_g_g1_g4_admm_gates as G
        import p58_rescale as R
        import pyomo.environ as pe
        import shared_resources_planning as srp
        ev = _eval_dir('B', 'V1')
        s31c_path = os.path.join(ev, 'interface_settlement_detail_s31c.json')
        out['anchor_source'] = {'path': os.path.relpath(s31c_path, REPO), 'sha256': _sha(s31c_path),
                                'eval_record_sha256': _sha(os.path.join(ev, 'evaluation_record.json'))}
        detail = _load(s31c_path)['interface_reporting_detail']
        # ---- the ADMM-prepared models of instance B under V-1, zero solves (the diag trial initialisation round)
        ac = spec['configuration']
        cg = coupling_gates_literals()
        derived = H.validate_derived_instance(spec['instances']['B']['derived_instance'])
        work = os.path.join(SMOKE_DIR, f'd3_work_{_stamp()}')
        os.makedirs(work)
        out['derived_installed'] = H.install_derived_instance(derived, os.path.join(work, 'derived'))
        report, holder = {}, {}
        eval_id = f'p515s56smoke_d3_B_V1_{_stamp()}'
        planning, sed, candidate = G._construct_arm_planning(
            ac['arm_label'], os.path.join(work, 'arm'), report, investment_map=dict(X0), eval_id=eval_id,
            num_max_iters_override=spec['cap'], apply_rho=False, investment_year=spec['candidate']['investment_year'])
        spec_like = {'configuration': {'overrides': {}, 'arm_label': ac['arm_label'],
                                       'case_file_anderson_acceleration': dict(ac['case_file_anderson_acceleration']),
                                       'ess_ageing_baseline': deepcopy(ac['ess_ageing_baseline']),
                                       'ess_params_file': {'path': H.ESS_PARAMS_FILE_REL,
                                                           'sha256': _sha(H.ESS_PARAMS_FILE_REL)},
                                       'ess_ageing_baseline_label': ac['ess_ageing_baseline_label'],
                                       'convergence_depth_tail': dict(ac['convergence_depth_tail']),
                                       'derived_instance': derived},
                     'cap': spec['cap'], 'required_consecutive_cycles': spec['required_consecutive_cycles']}
        hook = H._config_hook_factory(spec_like, holder, overrides={}, model_variant=None,
                                      investment_year=spec['candidate']['investment_year'], expected_floor_rows=None,
                                      flex_price_multiplier=None, interface_deviation_premium=None,
                                      release_solution_bookkeeping=None,
                                      interface_coupling=spec['arms']['V1']['options']['interface_coupling'])
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        out['admm_in_force'] = {k: getattr(a, k, 'ABSENT') for k in (
            'per_scenario_interface_coupling', 'per_scenario_voltage_mode', 'row18_charge_reactive', 'objective_scale')}
        out['admm_in_force']['interface_deviation_premium'] = dict(a.interface_deviation_premium)
        out['coupling_literal_matches_spec'] = cg['COUPLING_V1_NOQ'] == spec['arms']['V1']['options']['interface_coupling']
        cap = {}
        g0 = dict(GUARD.counts)
        DG._init_round(planning, candidate, srp, R, cap, True)
        out['init_round'] = {'stopped_by_sentinel': cap.get('stopped_by_sentinel'), 'exception': cap.get('exception'),
                             'wrappers_restored': cap.get('wrappers_restored'),
                             'sigma_computed_replaced_in_trial': cap.get('sigma_computed_replaced_in_trial'),
                             'p58_applied': cap.get('p58_applied')}
        if 'stopped_by_sentinel' not in cap:
            raise RuntimeError(f"stubbed initialisation round did not reach the sentinel: {cap.get('exception')}")
        tso_model, dso_models = cap['tso_model'], cap['dso_models']
        tn = planning.transmission_network
        blocks = [(next(yy for yy in tn.years if str(yy) == bk.split('|')[0]), bk.split('|')[1])
                  for bk in spec['instances']['B']['blocks']]
        # ---- pc / qc from the smoke run's anchors (the only solution-dependent box input)
        anchor = {'n_set': 0, 'max_abs_change_from_stub_pu': 0.0, 'scenario_common': True, 'per_block': {}}
        for y, d in blocks:
            tb = tso_model[y][d]
            tnet = tn.network[y][d]
            sb = tnet.baseMVA
            pb = {}
            for node_id in tn.active_distribution_network_nodes:
                adn_load = tnet.get_adn_load_idx(node_id)
                det = detail[str(node_id)][str(y)][d]['periods']
                rng = []
                for p in tb.periods:
                    ap, aq = det[str(p)]['anchor_p_mw'], det[str(p)]['anchor_q_mvar']
                    if len(set(ap.values())) != 1 or len(set(aq.values())) != 1:
                        anchor['scenario_common'] = False
                    for s_m in tb.scenarios_market:
                        for s_o in tb.scenarios_operation:
                            for var, val in ((tb.pc[adn_load, s_m, s_o, p], ap[f'{s_m}_{s_o}']),
                                             (tb.qc[adn_load, s_m, s_o, p], aq[f'{s_m}_{s_o}'])):
                                if not var.fixed:
                                    raise RuntimeError(f'{var.name} is not fixed in the stubbed build')
                                old = pe.value(var)
                                var.set_value(val / sb)
                                anchor['max_abs_change_from_stub_pu'] = max(anchor['max_abs_change_from_stub_pu'],
                                                                            abs(old - val / sb))
                                anchor['n_set'] += 1
                    rng.append(ap['0_0'])
                pb[str(node_id)] = {'anchor_p_mw_min': min(rng), 'anchor_p_mw_max': max(rng)}
            anchor['per_block'][f'{y}|{d}'] = pb
        out['anchor'] = anchor
        # ---- central values
        diag = _load(DIAG_RESULTS)
        s3v3 = _load(S3V3_RESULTS)
        s32 = diag['solves']['S3_2|2025|Summer']
        itf8 = {}
        for kk, v in s32['single'].items():
            if v.get('optimal') and 'interface' in v:
                itf8[kk] = v['interface']['00']
        for kk in ('12', '22'):
            v = (s3v3.get('solves') or {}).get(f'single_{kk}') or {}
            if v.get('optimal') and 'interface' in v:
                itf8[kk] = v['interface']['00']
        merged_itf = s32['merged_default']['interface']
        diag_central = _load(DIAG_CENTRAL)
        tol = spec['d3_2025_summer']['bound_tol_rel']
        b2025 = [bl for bl in blocks if str(bl[0]) == '2025']
        b2035 = [bl for bl in blocks if str(bl[0]) == '2035']
        all9 = [f'{a_}{b_}' for a_ in range(3) for b_ in range(3)]
        res = {}
        res['primary_singles'] = d3_boxes_filtered(
            planning, tso_model, dso_models, {'2025|Summer': {'complete': False, 'interface': itf8}}, b2025, tol,
            sorted(itf8))['2025|Summer']
        res['primary_singles']['n_scenarios_available'] = len(itf8)
        res['primary_singles']['scenarios_missing'] = sorted(set(all9) - set(itf8))
        sup_c = {'2025|Summer': {'complete': True, 'interface': merged_itf}}
        res['supplementary_merged_default'] = DG.d3_boxes(planning, tso_model, dso_models, sup_c, b2025,
                                                          tol)['2025|Summer']
        sup_f = d3_boxes_filtered(planning, tso_model, dso_models, sup_c, b2025, tol, all9)['2025|Summer']
        # ---- method gate: 2035|Summer reproduces the diag run_v1 child B D-3 (real initialisation)
        g35 = DG.d3_boxes(planning, tso_model, dso_models, diag_central, b2035, tol)['2035|Summer']
        g35_f = d3_boxes_filtered(planning, tso_model, dso_models, diag_central, b2035, tol, all9)['2035|Summer']
        ref = _load(DIAG_CHILD_B)['d3']['boxes']['2035|Summer']
        fams = sorted(set(ref['per_family']) | set(g35['per_family']))
        fam_cmp = {f: {'ref_min_margin': ref['per_family'].get(f, {}).get('min_margin'),
                       'smoke_min_margin': g35['per_family'].get(f, {}).get('min_margin'),
                       'ref_n_outside_strict': ref['per_family'].get(f, {}).get('n_outside_strict'),
                       'smoke_n_outside_strict': g35['per_family'].get(f, {}).get('n_outside_strict')} for f in fams}
        for f, e in fam_cmp.items():
            e['equal'] = (e['ref_min_margin'] == e['smoke_min_margin']
                          and e['ref_n_outside_strict'] == e['smoke_n_outside_strict'])
            if e['ref_min_margin'] is not None and e['smoke_min_margin'] is not None:
                e['abs_diff'] = abs(e['ref_min_margin'] - e['smoke_min_margin'])
        res['method_gate'] = {
            '2035_vs_diag_child_B': {'counts_equal': all(ref[x] == g35[x] for x in ('n_values', 'n_outside',
                                                                                     'n_outside_strict')),
                                     'counts': {x: [ref[x], g35[x]] for x in ('n_values', 'n_outside',
                                                                              'n_outside_strict')},
                                     'per_family': fam_cmp,
                                     'all_family_min_margins_equal': all(e['equal'] for e in fam_cmp.values())},
            'filtered_copy_equals_d3_boxes': {
                '2035': json.dumps(_box_summary(g35), sort_keys=True) == json.dumps(_box_summary(g35_f), sort_keys=True),
                '2025_merged_default': json.dumps(_box_summary(res['supplementary_merged_default']), sort_keys=True)
                == json.dumps(_box_summary(sup_f), sort_keys=True)}}
        out['results'] = res
        out['zero_solve_verified'] = dict(GUARD.counts) == g0 and GUARD.counts['permitted_solve'] == 0
        out['guard_diag_module'] = {'counts': dict(DG.GUARD.counts), 'verify_failures': DG.GUARD.verify(0)}
        out['pickle_diag_module'] = dict(DG.PK)
        out['work_dir'] = os.path.relpath(work, REPO)
    except Exception:  # noqa: BLE001 -- recorded
        out['error'] = traceback.format_exc()
        _log('D-3 EXCEPTION:\n' + out['error'])
    out['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    out['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['wall_s'] = time.time() - t0
    out['utc_end'] = _utc()
    write_json(path, out)
    ps = ((out.get('results') or {}).get('primary_singles') or {})
    _log(f"D-3 2025|Summer -> {os.path.relpath(path, REPO)}: primary (8 singles) n_outside {ps.get('n_outside')} strict "
         f"{ps.get('n_outside_strict')}; method gate {((out.get('results') or {}).get('method_gate') or {})}; "
         f"error {out['error'] is not None}")
    return 0 if out['error'] is None and not out['guard']['verify_failures'] else 2


# ======================================================================================================================
#  evaluation (zero solves)
# ======================================================================================================================
def _probabilities(ev):
    """omega per block 'year|day' and scenario 's_m_s_o' from the run's multiscenario_terminal (TSO block)."""
    ms = _load(os.path.join(ev, H.MULTISCENARIO_TERMINAL_FILE))
    out = {}
    for key, b in ms['blocks'].items():
        if b.get('kind') == 'TSO':
            out[f"{b['year']}|{b['day']}"] = {s: v['probability'] for s, v in b['per_scenario_costs'].items()}
    return out, ms


def evaluate_run(spec, inst, arm):
    ev = _eval_dir(inst, arm)
    k = ORDER.index((inst, arm)) + 1
    rec = _load(os.path.join(ev, 'evaluation_record.json'))
    rows = _read_jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))
    resp = _read_jsonl(os.path.join(ev, H.PER_CYCLE_RESPONSE_FILE))
    with open(os.path.join(ev, 'exit_code.txt')) as handle:
        exit_txt = int(handle.read().strip())
    launcher = _load(os.path.join(SMOKE_DIR, f'run_{k}_{inst}_{arm}_launcher.json'))
    sp = rec.get('solve_profile') or {}
    obs = sp.get('observed') or {}
    declared = spec['solves']['declared_base_per_run'][inst]
    out = {'run': k, 'instance': inst, 'arm': arm, 'eval_dir': os.path.relpath(ev, REPO),
           'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
           'campaign_spec_sha256': rec.get('campaign_spec_sha256'), 'status': rec.get('status'),
           'exit_code_parent': (launcher.get('result') or {}).get('exit_code'), 'exit_code_txt': exit_txt,
           'cycles_run': rec.get('cycles_run'), 'n_per_cycle_rows': len(rows),
           'solves': {'declared_base': declared, 'base_solves': sp.get('base_solves'),
                      'observed_permitted_solve': obs.get('permitted_solve'),
                      'observed_permitted_exec': obs.get('permitted_exec'), 'blocked_solve': obs.get('blocked_solve'),
                      'blocked_exec': obs.get('blocked_exec'), 'retry_solves_credited': sp.get('retry_solves_credited'),
                      'expected_solves': sp.get('expected_solves'), 'identity_holds': sp.get('identity_holds')},
           'launcher_guard': launcher.get('guard'),
           'peak_rss': rec.get('peak_rss'),
           's_per_cycle': [r.get('cycle_wall_s') for r in resp],
           'wall_time_s': rec.get('wall_time_s'),
           'aa': {'effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
                  'forced_off_per_cycle': [r.get('interface_coupling_aa_forced_off') for r in rows]},
           'local_solve_failures': rec.get('local_solve_failures'),
           'network_failures_summary': rec.get('network_failures_summary')}
    out['S-P1'] = {'held': (out['exit_code_parent'] == 0 and exit_txt == 0 and rec.get('cycles_run') == CAP
                            and len(rows) == CAP and sp.get('identity_holds') is True
                            and sp.get('base_solves') == declared and obs.get('blocked_solve') == 0
                            and obs.get('blocked_exec') == 0 and not (launcher.get('guard') or {}).get('verify_failures'))}
    rss_b = (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss')
    out['S-P5'] = {'child_python_peak_rss_gib': rss_b / GiB if rss_b else None,
                   'ipopt_max_rss_gib': ((rec.get('peak_rss') or {}).get('solver_subprocesses_max_ru_maxrss') or 0) / GiB,
                   'held': bool(rss_b) and rss_b <= spec['thresholds']['S-P5_rss_bytes']}
    out['init_identity'] = rec.get('initialisation_identity')
    rc = rec.get('recourse_components') or {}
    out['terminal_recourse'] = {kk: rc.get(kk) for kk in (
        'gross_operational_cost', 'gross_operational_cost_contracted_excluded', 'interface_settlement_total',
        'interface_settlement_contracted_total', 'interface_settlement_deviation_total', 'voltage_pin_total',
        'voltage_pin_tso', 'voltage_pin_dso', 'terminal_salvage_value', 'q_convention')}
    out['per_cycle_gross'] = [r.get('gross_operational_cost') for r in rows]
    probs, ms = _probabilities(ev)
    # ---- Q per block and Q_block - w_b C_central
    w_own = spec['instances'][inst]['w_b_own']
    qb = {}
    if arm in COUPLED:
        cap_lines = _read_jsonl(os.path.join(ev, H.S56_CYCLE_CAPTURE_FILE))
        header, lines = cap_lines[0], cap_lines[1:]
        out['s56_capture'] = {'header_blocks': header.get('blocks'), 'n_lines': len(lines),
                              'deviation_kinds': header.get('deviation_kinds'),
                              'checklist': rec.get('s56_cycle_capture_checklist_asserted_before_run'),
                              'summary': rec.get('s56_cycle_capture_summary')}
        for ln in lines:
            for bk, v in ln['q_blocks']['per_year_day'].items():
                qb.setdefault(bk, []).append({'cycle': ln['cycle'], 'settlement_excluded_in_full': v[
                    'settlement_excluded_in_full'], 'contracted_excluded': v['contracted_excluded']})
        weights_capture = {}
        for ln in lines[:1]:
            for lab, v in ln['q_blocks']['per_agent'].items():
                yd = '|'.join(lab.split('|')[1:])
                weights_capture.setdefault(yd, set()).add(v['weight'])
        out['w_b_capture'] = {bk: sorted(v) for bk, v in weights_capture.items()}
        out['w_b_capture_equals_own'] = all(len(v) == 1 and list(v)[0] == w_own[bk] for bk, v in weights_capture.items())
        # S-P4
        sums = []
        for ln, r in zip(lines, rows):
            s = sum(v['settlement_excluded_in_full'] for v in ln['q_blocks']['per_year_day'].values())
            sums.append({'cycle': ln['cycle'], 'capture_sum': s, 'production_gross': r['gross_operational_cost'],
                         'rel': _rel(s, r['gross_operational_cost'])})
        ck = rec.get('s56_cycle_capture_checklist_asserted_before_run') or {}
        sm = rec.get('s56_cycle_capture_summary') or {}
        out['S-P4'] = {'checklist_all_true': bool(ck) and all(v is True for v in ck.values()),
                       'summary_ok': sm.get('ok') is True and sm.get('cycle_lines_written') == CAP,
                       'per_cycle': sums,
                       'held': (bool(ck) and all(v is True for v in ck.values()) and sm.get('ok') is True
                                and sm.get('cycle_lines_written') == CAP and len(sums) == CAP
                                and all(x['rel'] <= spec['thresholds']['S-P4_rel'] for x in sums))}
        # S-P2
        th = spec['thresholds']['S-P2_rel']
        per = []
        for ln, r, rr in zip(lines, rows, resp):
            pa = ln['q_blocks']['per_agent']
            T = sum(v['weight'] * v['settlement_total_unweighted'] for v in pa.values())
            D = sum(v['weight'] * (v['settlement_total_unweighted'] - v['settlement_contracted_unweighted'])
                    for v in pa.values())
            Ddso = sum(v['weight'] * (v['settlement_total_unweighted'] - v['settlement_contracted_unweighted'])
                       for lab, v in pa.items() if lab.startswith('DSO'))
            lhs2 = r['gross_operational_cost_contracted_excluded'] - r['gross_operational_cost']
            e = {'cycle': ln['cycle'], 'price_part': r['priced_residual_price_part'], 'settlement_total_T': T,
                 'i_rel': _rel(r['priced_residual_price_part'], -T), 'contracted_minus_primary': lhs2,
                 'deviation_total_D': D, 'ii_rel': _rel(lhs2, D), 'dso_deviation_from_capture': Ddso,
                 'dso_deviation_production_covariance_dso_weighted': rr.get('covariance_dso_weighted'),
                 'ii_dso_decomposition_rel': _rel(Ddso, rr.get('covariance_dso_weighted'))
                 if rr.get('covariance_dso_weighted') is not None else None}
            if ln['cycle'] == CAP:
                e['i_rel_terminal_record'] = _rel(r['priced_residual_price_part'], -rc['interface_settlement_total'])
                e['T_vs_record_rel'] = _rel(T, rc['interface_settlement_total'])
                e['ii_D_vs_record_deviation_rel'] = _rel(D, rc['interface_settlement_deviation_total'])
                e['ii_lhs_vs_record_deviation_rel'] = _rel(lhs2, rc['interface_settlement_deviation_total'])
            e['held'] = (e['i_rel'] <= th and e['ii_rel'] <= th and e['ii_dso_decomposition_rel'] is not None
                         and e['ii_dso_decomposition_rel'] <= th
                         and all(e[x] <= th for x in ('i_rel_terminal_record', 'T_vs_record_rel',
                                                      'ii_D_vs_record_deviation_rel', 'ii_lhs_vs_record_deviation_rel')
                                 if x in e))
            per.append(e)
        # gauge
        duals = _read_jsonl(os.path.join(ev, 'interface_duals_per_cycle.jsonl'))
        dh, dl = duals[0], duals[1:]
        scen = [tuple(x) for x in dh['deviation_channel']['scenarios']]
        blocks = [tuple(x) for x in dh['blocks']]
        gauge = []
        for line in dl:
            fam_out = {}
            for field in dh['deviation_channel']['fields']:
                arr = line[field]
                mx = max((abs(x) for blk in arr for sc in blk for x in sc), default=0.0)
                worst = 0.0
                worst_at = None
                for bi, blk in enumerate(arr):
                    node, year, day = blocks[bi]
                    om = probs[f'{year}|{day}']
                    for t in range(len(blk[0])):
                        g = sum(om[f'{s_m}_{s_o}'] * blk[si][t] for si, (s_m, s_o) in enumerate(scen))
                        if abs(g) > worst:
                            worst, worst_at = abs(g), [node, year, day, t]
                bound = spec['thresholds']['S-P2_gauge_abs_factor'] * mx
                fam_out[field] = {'max_abs_mu': mx, 'max_abs_weighted_sum': worst, 'at': worst_at, 'bound': bound,
                                  'held': worst <= bound}
            gauge.append({'cycle': line['cycle'], 'families': fam_out,
                          'held': all(v['held'] for v in fam_out.values())})
        pin = None
        if arm == 'V1':
            vd = rc.get('voltage_pin_dso') or {}
            pin = {'voltage_pin_total': rc.get('voltage_pin_total'), 'voltage_pin_tso': rc.get('voltage_pin_tso'),
                   'voltage_pin_dso': vd,
                   'held': rc.get('voltage_pin_total') == 0.0 and rc.get('voltage_pin_tso') == 0.0
                   and all(v == 0.0 for v in vd.values()),
                   'scope': 'terminal record (cycle 3) + structural reading for cycles 1-2 (no per-cycle record)'}
        out['S-P2'] = {'identities_per_cycle': per, 'gauge_per_cycle': gauge, 'pin_v1': pin,
                       'held': (len(per) == CAP and all(e['held'] for e in per) and len(gauge) == CAP
                                and all(g['held'] for g in gauge) and (pin is None or pin['held']))}
    else:
        if len(spec['instances'][inst]['blocks']) == 1:
            bk = spec['instances'][inst]['blocks'][0]
            qb[bk] = [{'cycle': r['cycle'], 'contracted_excluded': r['gross_operational_cost'],
                       'source': 'per_cycle_record gross_operational_cost (one block)'} for r in rows]
        else:
            for bk in spec['instances'][inst]['blocks']:
                tot = 0.0
                for key, b in ms['blocks'].items():
                    if f"{b['year']}|{b['day']}" == bk:   # per-day local Q (production block value / weight)
                        tot += b['block_recourse_production_local'] * b['admm_block_weight']
                qb[bk] = [{'cycle': CAP, 'contracted_excluded': tot,
                           'source': ('multiscenario_terminal: sum over the block\'s agents of '
                                      'block_recourse_production_local x admm_block_weight (terminal only)')}]
            out['terminal_block_sum_vs_gross_rel'] = _rel(sum(v[0]['contracted_excluded'] for v in qb.values()),
                                                          rows[-1]['gross_operational_cost'])
    deltas = {}
    for bk, series in qb.items():
        c = spec['central'].get(bk) or {}
        cc = c.get('C_central_eur_per_day')
        if cc is None:
            deltas[bk] = {'C_central': None, 'reason': c.get('status'), 'w_b': w_own[bk], 'Q': series}
            continue
        wc = w_own[bk] * cc
        deltas[bk] = {'C_central': cc, 'w_b': w_own[bk], 'w_b_C_central': wc,
                      'allowance_eur': 2500.0 * w_own[bk] / spec['sum_12_w'],
                      'per_cycle': [dict(s, **{f'{kk}_minus_wC': s[kk] - wc
                                               for kk in ('settlement_excluded_in_full', 'contracted_excluded')
                                               if kk in s}) for s in series]}
    out['Q_block_minus_wC'] = deltas
    return out


def evaluate(args):
    spec, spec_sha, spec_rel = _spec(args)
    path = os.path.join(SMOKE_DIR, f'smoke_evaluation_{_stamp()}.json')
    res = {'schema': 'p515_s56_step4_smoke_evaluation_v1', 'utc': _utc(), 'spec': {'path': spec_rel,
                                                                                   'sha256': spec_sha},
           'script': SCRIPT, 'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'objective_convention': spec['objective_convention'], 'runs': {}, 'errors': {}}
    for inst, arm in ORDER:
        try:
            res['runs'][f'{inst}|{arm}'] = evaluate_run(spec, inst, arm)
        except Exception:  # noqa: BLE001
            res['errors'][f'{inst}|{arm}'] = traceback.format_exc()
    runs = res['runs']
    pr = {}
    pr['S-P1'] = {'statement': spec['predictions']['S-P1'],
                  'per_run': {k: v['S-P1']['held'] for k, v in runs.items()},
                  'held': len(runs) == 8 and all(v['S-P1']['held'] for v in runs.values())}
    p2 = {k: v['S-P2']['held'] for k, v in runs.items() if v['arm'] in COUPLED}
    pr['S-P2'] = {'statement': spec['predictions']['S-P2'], 'per_run': p2, 'held': len(p2) == 4 and all(p2.values())}
    p3 = {}
    for inst in INSTANCES:
        g = {a: (runs.get(f'{inst}|{a}') or {}).get('init_identity') or {} for a in ARMS}
        v1, v2 = g['V1'], g['V2']
        off, on = g['pinned_aa_off'], g['pinned_aa_on']

        def labels(x):
            return sorted(c.get('label') for c in (x.get('compared') or []))
        e = {'V1': {'group': v1.get('initialisation_identity_group'), 'n_compared': v1.get('n_compared'),
                    'hex': v1.get('gross_operational_cost_hex')},
             'V2': {'group': v2.get('initialisation_identity_group'), 'n_compared': v2.get('n_compared'),
                    'listed_not_compared': [c.get('label') for c in v2.get('other_candidates_not_compared') or []],
                    'hex': v2.get('gross_operational_cost_hex')},
             'pinned_aa_off': {'compared': labels(off), 'all_equal': off.get('all_equal'),
                               'hex': off.get('gross_operational_cost_hex'), 'group': off.get('initialisation_identity_group')},
             'pinned_aa_on': {'compared': labels(on), 'all_equal': on.get('all_equal'),
                              'hex': on.get('gross_operational_cost_hex'), 'group': on.get('initialisation_identity_group')}}
        e['held'] = (v1.get('initialisation_identity_group') == 'voltage_pin_not_constructed' and v1.get('n_compared') == 0
                     and v2.get('initialisation_identity_group') == 'voltage_pin_constructed'
                     and v2.get('n_compared') == 0 and 'x0_V1' in e['V2']['listed_not_compared']
                     and labels(off) == ['x0_V2'] and off.get('all_equal') is True
                     and labels(on) == ['x0_V2', 'x0_pinned_aa_off'] and on.get('all_equal') is True
                     and off.get('gross_operational_cost_hex') == v2.get('gross_operational_cost_hex')
                     == on.get('gross_operational_cost_hex')
                     and v1.get('gross_operational_cost_hex') != v2.get('gross_operational_cost_hex'))
        p3[inst] = e
    pr['S-P3'] = {'statement': spec['predictions']['S-P3'], 'per_instance': p3,
                  'held': all(v['held'] for v in p3.values())}
    p4 = {k: v['S-P4']['held'] for k, v in runs.items() if v['arm'] in COUPLED}
    pr['S-P4'] = {'statement': spec['predictions']['S-P4'], 'per_run': p4, 'held': len(p4) == 4 and all(p4.values())}
    p5 = {k: {'rss_gib': v['S-P5']['child_python_peak_rss_gib'], 'ipopt_gib': v['S-P5']['ipopt_max_rss_gib'],
              's_per_cycle': v['s_per_cycle'], 'held': v['S-P5']['held']} for k, v in runs.items()}
    pr['S-P5'] = {'statement': spec['predictions']['S-P5'], 'per_run': p5,
                  'held': len(p5) == 8 and all(v['held'] for v in p5.values())}
    res['predictions'] = pr
    res['launcher_guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    write_json(path, res)
    _log(f"evaluation -> {os.path.relpath(path, REPO)}: {({k: v['held'] for k, v in pr.items()})}; "
         f"errors {list(res['errors'])}")
    return 0 if not res['errors'] else 2


def manifest(args):
    spec, spec_sha, spec_rel = _spec(args)
    path = os.path.join(SMOKE_DIR, 'step4_smoke_v1_manifest_sha256.json')
    outputs = {}
    for root, _dirs, files in os.walk(SMOKE_DIR):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if os.path.abspath(p) == os.path.abspath(path):
                continue
            outputs[os.path.relpath(p, REPO)] = {'sha256': _sha(p), 'bytes': os.path.getsize(p),
                                                 'tracked': bool(_git('ls-files', '--', os.path.relpath(p, REPO)).strip())}
    write_json(path, {'utc': _utc(), 'spec': {'path': spec_rel, 'sha256': spec_sha},
                      'script': {'path': SCRIPT, 'sha256': _sha(SCRIPT)}, 'n_files': len(outputs), 'files': outputs})
    _log(f'manifest -> {os.path.relpath(path, REPO)} ({len(outputs)} files)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--d3', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--manifest', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--out-dir')
    ap.add_argument('--instance', choices=INSTANCES)
    ap.add_argument('--arm', choices=ARMS)
    a = ap.parse_args()
    if a.freeze and a.out_dir:
        return freeze(a)
    if a.run and a.spec and a.instance and a.arm:
        return run(a)
    if a.d3 and a.spec:
        return d3(a)
    if a.evaluate and a.spec:
        return evaluate(a)
    if a.manifest and a.spec:
        return manifest(a)
    ap.error('--freeze --out-dir D | --run --spec S --instance I --arm A | --d3 --spec S | --evaluate --spec S | '
             '--manifest --spec S')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
