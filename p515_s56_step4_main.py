"""P5.15 S56 validation step 4 MAIN RUN -- the four arms (V-1, V-2, pinned AA-off, pinned AA-on) on instance A
(2025|Winter) and instance B ({2025, 2035} x Summer), cap 300, x0, alpha 0, through the existing campaign harness, under
ONE frozen addendum spec (`frozen_step4_main_spec_v1_<sha8>.json`; predecessor the step 4 smoke spec v2 e36c29bd; cites
the diagnostics spec ff466289 and S56_STEP4_DESIGN.md, whose ruled status paragraph governs).

Authority: TASKS.md S56 section, "Expert rulings on the handback" items 1, 2 and 7 and the author's authorization of
2026-11-23 (step 4 main run inside the existing 10-20 h); S56_PLAN_v4.md sections 3 and 7 (row 4); the Planner's task of
2026-11-23 (step 4 main run).

PHASES (each ONE harness campaign: one frozen campaign spec, four entries, concurrency 4, a NEW absolute version-tagged
root `campaign_s56s4main_v1_<phase>_<inst>`):
  main B  -> main A  (the harness campaign lock `.p515_s44_campaign.lock` is one per checkout and a derived instance is
                     campaign-level, so the two instances cannot run as concurrent campaigns: A runs after B -- the
                     task's stated fallback)
  cont600 <inst>      (declared continuation, spec item (v): V-1, plus V-2 if still contracting, cap 600, fresh runs
                     from cycle 1 with the same code; launched only under the spec's continuation rule)
Labels: x0_<arm>_main / x0_<arm>_cont600 (distinct from the smoke's x0_<arm>, whose evaluation keys are the same: the cap
and the pinned capture key do not enter the key, so the initialisation-identity files cannot collide).

INITIALISATION IDENTITY (M2). Before launch the launcher copies, byte for byte, the four serial-smoke initialisation
records of the same instance into the new root's `init_identity/` (the harness's documented "reference record the
launcher placed there"). Each child registers its own record and compares it BITWISE with every record of the same
candidate / instance / group already there (the seeded smoke records and any concurrent sibling registered earlier),
raising before ADMM cycle 1 on a mismatch.

SOLVES. The launcher parent never solves: SolveProfileGuard(permitted=()) for the whole process, verified at 0. Each
harness child runs `run_admm_arm`, whose armed guard counts every solve; its record's `solve_profile` reconciles it:
base = ((1 + n_dso) x n_years x n_days + n_esso_nodes) x (cycles_run + 1) (A 7, B 11 per round) + every production
recovery retry actually attempted.

MODES (zero solves except --run, whose solves happen in the harness children):
  --freeze --out-dir D                               the frozen main spec
  --run --spec S --phase main|cont600 --instance A|B [--arms a,b]   one campaign (attached; stdout / stderr captured
                                                     by the caller)
  --look40 --spec S --instance A|B                   the declared cycle-40 look on V-1 (spec item (iv)); reads the
                                                     running campaign's capture (complete lines only)
  --vmstat --tag T [--spec S]                        one vm_stat snapshot (raw text + free / inactive GiB); with a v2
                                                     spec, into the spec's main directory
Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_step4_main.py --run --spec S --phase main
       --instance B > <stdout log> 2> <stderr log>
Objective convention (every Q): EUR of total discounted cost. Coupled arms: production's primary gross_operational_cost
= settlement excluded in full (pin excluded, salvage separate), contracted-excluded beside it. Pinned arms:
gross_operational_cost = contracted settlement and pin excluded. C_central: step 3 gross EUR/day (settlement weight 0, no
row 18, no pin, no salvage); e_b(k) = Q_b(k) - w_b x C_central in EUR of total discounted cost.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 main launcher (this process never solves)').install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (parent side: no model imports)
import p515_s56_step4_smoke as SM  # noqa: E402  (helpers only; installs its own permitted=() guard -- never solves)

SOLVER_PATH = '/usr/local/bin/ipopt'
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
MAIN_DIR = os.path.join(RES, 'step4_main_v1')
SPEC_PREFIX = 'frozen_step4_main_spec_v1_'
ROOT_TAG = 'v1'
# Spec v2 (2026-11-23, relaunch after the v1 attempt was killed by the session exit): the ONLY launcher change is that
# the campaign roots are taken from the spec. A spec carrying a 'roots' block (v2) sets MAIN_DIR, ROOT_TAG and the
# absolute campaign roots (apply_spec_roots); a spec without one (v1) leaves the v1 construction unchanged.
SPEC_ROOTS = None
SMOKE_SPEC = os.path.join(SM.SMOKE_DIR, 'frozen_step4_smoke_spec_v2_e36c29bd.json')
SMOKE_EVALUATION = os.path.join(SM.SMOKE_DIR, 'smoke_evaluation_20261123T135123.json')
SMOKE_LAUNCHER = 'p515_s56_step4_smoke.py'
DESIGN = 'S56_STEP4_DESIGN.md'
PLAN_V4 = 'S56_PLAN_v4.md'
TASKS = 'TASKS.md'
CAP = {'main': 300, 'cont600': 600}
PHASES = ('main', 'cont600')
INSTANCES = ('A', 'B')
ARMS = ('V1', 'V2', 'pinned_aa_off', 'pinned_aa_on')
COUPLED = ('V1', 'V2')
PINNED = ('pinned_aa_off', 'pinned_aa_on')
CALIBRATED = {'A': ['2025|Winter'], 'B': ['2035|Summer']}
MERGED_ONLY = {'2025|Summer': 185674.155753}
ALLOWANCE_EUR = {'2025|Winter': 228.64, '2035|Summer': 187.56}
D1_FLOOR_EUR = {'2025|Winter': {'production_depth': 597.67, 'tail_depth': 45.67},
                '2035|Summer': {'production_depth': 499.03, 'tail_depth': 40.58}}
DELTA_PIN_EUR_PER_DAY = {'2025|Winter': 0.4056, '2035|Summer': 4.8856}
TAIL_TOL = 1e-6
GiB = 1024.0 ** 3
LOOK40 = {'cycle': 40, 'fit_cycles': [21, 34], 'heldout_cycles': [35, 40], 'projection_fit_cycles': [21, 40],
          'stop_if_projected_cycles_exceed': 600}
M3_CYCLES = 3
# M3: fields EXCLUDED IN ADVANCE from the bitwise comparison with the serial smoke (timing / process / path / byte-count
# fields; the concurrency gate's G3 failed on a wall-time field's byte count, so every such field is named here)
M3_EXCLUDED_KEYS = ('cycle_wall_s', 'rss_bytes', 'ru_maxrss_bytes', 'response_capture_s', 'capture_s', 'log_path',
                    'log_bytes', 'utc', 'pid', 'timestamp_utc', 'wall_s')
M3_FILES = {'per_cycle_record.jsonl': 'every cycle line 1..3 (keys present in both files; keys present in one only are '
                                      'listed)',
            'interface_duals_per_cycle.jsonl': 'cycle lines 1..3 (header excluded)',
            's56_cycle_capture.jsonl': ('coupled arms only (the smoke pinned arms had no capture): cycle lines 1..3, '
                                        'header excluded, the field voltage_pin (added by a662c9bf, absent from the '
                                        'smoke file) listed not compared'),
            'network_ipopt_solve_records.jsonl': 'every record of rounds 0..3, in file order'}

V2_DECLARATION = 'V-2 is a measured, non-deciding arm.'
PREDICTIONS_MUST_HOLD = {
    'M1': ('F1 identities every cycle in the coupled arms: PR price part + settlement total = 0, to rel 1e-9; '
           'contracted - primary = deviation total, to rel 1e-9; Sigma_s omega_s mu_s = 0, to 1e-9 x max|mu|; V-1 pin '
           'total == 0.'),
    'M2': 'Initialisation identity: pinned AA-off == pinned AA-on == V-2, bitwise; V-1 forms its own group.',
    'M3': ('Every run\'s first 3 cycles are bitwise equal to its serial smoke counterpart, with timing fields excluded '
           'in advance. This checks concurrency and continuation equivalence.'),
    'M4': ('Capture-path assertions hold, and the per-block Q sums equal production\'s Q to rel 1e-12, every cycle, in '
           'every arm.'),
    'M5': 'Solve-count identity in every run.',
}
PREDICTIONS_RECORDED = {
    'P3': ('On tail-depth cycles, V-1\'s e_b(k) contracts to <= the block allowance. If it does not, report the smallest '
           '|e_b| reached, R_eff = 4 x |e_b| x Sigma_12 w / w_b, and the dominant term, before anything touches R.'),
    'P4': ('D_b = w_b x C_central - Q_pin,b (pinned AA-off, contracted-excluded): D_b >= 0 on 2035|Summer, and |D_b| on '
           '2025|Winter much smaller. Expert ruling 7 applies: D_b is pinned accounting optimism (>= 0) plus DN '
           're-optimisation gain (<= 0). A negative value is not a defect.'),
    'P5': ('lim Q_V2 <= w_b x (C_central + Delta_pin) + bars, where Delta_pin is 0.4056 EUR/day on 2025|Winter and '
           '4.8856 EUR/day on 2035|Summer. This is subject to the local-solution caveat.'),
    'P6': ('At a matched expectation-channel residual: cycles F1 / pinned AA-off between 1.0 and 2.0; wall time per '
           'cycle between 0.9 and 1.2; AA-off / AA-on >= 2.'),
    'P7': ('After the run, PR over all channels (recomputed from the capture) reduces |Q_k - w x C| at least fivefold '
           'after cycle 20.'),
    'P8': 'The Q step at tail onset is recorded per run.',
    'P9': 'Peak RSS <= 6 GiB per child.',
}
ALSO_RECORD = [
    'M1 for V-1 vs V-2: the Delta V_s trajectory (max and omega-RMS)',
    'M4 effort: cycles and wall time to the same per-block error level',
    'local-solve failures per run',
    'per run, the terminal-step-to-threshold ratio (CLAUDE.md rule)',
    'the settling bars S per block, in validated extrapolation form',
    ('the working level: the loosest Boyd tolerance at which the V-1 per-block error stays <= allowance at every later '
     'tail-depth cycle, also expressed as per-block observables (max and omega-RMS of Delta P_s, Delta Q_s, Delta V_s, '
     'and A_b)'),
]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 step4 main] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def _load(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


def root_name(phase, inst):
    return f's56s4main_{ROOT_TAG}_{phase}_{inst}'


def root_path(phase, inst):
    """ABSOLUTE (the child's rule-eleven assertion requires absolute results dirs). Taken from the spec's 'roots' block
    when the spec has one (v2)."""
    if SPEC_ROOTS is not None:
        return SPEC_ROOTS[phase][inst]
    return os.path.join(REPO, MAIN_DIR, f'campaign_{root_name(phase, inst)}')


def apply_spec_roots(spec):
    """Spec v2: take MAIN_DIR, ROOT_TAG and the absolute campaign roots from the spec's 'roots' block, after checking
    that every root is absolute, inside MAIN_DIR and named campaign_<root_name(phase, inst)>. No-op for a v1 spec."""
    global MAIN_DIR, ROOT_TAG, SPEC_ROOTS
    roots = spec.get('roots')
    if roots is None:
        return
    main_dir, tag, campaign_roots = roots['main_dir'], roots['root_tag'], roots['campaign_roots']
    for phase in PHASES:
        for inst in INSTANCES:
            p = campaign_roots[phase][inst]
            want = os.path.join(REPO, main_dir, f'campaign_s56s4main_{tag}_{phase}_{inst}')
            if not os.path.isabs(p) or p != want:
                raise RuntimeError(f'spec root {phase}/{inst} {p!r} is not the absolute version-tagged root {want!r}')
    MAIN_DIR, ROOT_TAG, SPEC_ROOTS = main_dir, tag, campaign_roots


def label_of(phase, arm):
    return f'x0_{arm}_{phase}'


def arm_options(arm, cg):
    """The smoke's arm options (SM.arm_options, unchanged) plus, on the pinned arms, the per-cycle capture key
    `s56_cycle_capture_pinned` True (a662c9bf; capture only, not in the evaluation key)."""
    options = SM.arm_options(arm, cg)
    if arm in PINNED:
        options[H.S56_CYCLE_CAPTURE_PINNED_KEY] = True
    return options


def smoke_eval_dir(inst, arm):
    ev = os.path.join(SM.root_path(inst, arm), 'evals')
    dirs = sorted(os.listdir(ev))
    if len(dirs) != 1:
        raise RuntimeError(f'expected one smoke eval dir under {ev}: {dirs}')
    return os.path.join(ev, dirs[0])


def smoke_init_record(inst, arm):
    """The smoke run's OWN initialisation record (in its own root), path and file name."""
    ev = smoke_eval_dir(inst, arm)
    name = f'{os.path.basename(ev)}.json'
    path = os.path.join(SM.root_path(inst, arm), H.INIT_IDENTITY_DIR_NAME, name)
    if not os.path.isfile(path):
        raise RuntimeError(f'smoke initialisation record missing: {path}')
    return path, name


def other_copies_of_this_launcher():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    mine = H._ancestor_pids()
    out = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 2 or int(tok[0]) in mine:
            continue
        if any(os.path.basename(t) == SCRIPT for t in tok[1:6]) and '--run' in tok:
            out.append(ln.strip()[:300])
    return out


# ======================================================================================================================
#  capture-path checklist (rule eleven): every quantity the spec requires has a capture path, BEFORE any launch
# ======================================================================================================================
def capture_path_checklist():
    import inspect
    checks = dict(SM.capture_path_checklist())   # the smoke's checklist (coupled-arm sources), unchanged
    with open('p515_s44_campaign_harness.py', encoding='utf-8') as handle:
        hsrc = handle.read()
    with open('shared_resources_planning.py', encoding='utf-8') as handle:
        srp_text = handle.read()
    with open('interface_dual_capture.py', encoding='utf-8') as handle:
        idc_text = handle.read()
    checks.update({
        'pinned_capture_key': getattr(H, 'S56_CYCLE_CAPTURE_PINNED_KEY', None) == 's56_cycle_capture_pinned'
                              and H.s56_cycle_capture_arm(False, True) == H.S56_CAPTURE_ARM_PINNED
                              and H.s56_cycle_capture_arm(True, None) == H.S56_CAPTURE_ARM_COUPLED,
        'pinned_capture_key_not_in_eval_key': 's56_cycle_capture_pinned' not in inspect.signature(
            H.evaluation_key).parameters,
        'pinned_capture_fields': set(H.S56_CYCLE_CAPTURE_LINE_FIELDS_BY_ARM[H.S56_CAPTURE_ARM_PINNED]) >= {
            'primal_expectation', 'dual_expectation_v', 'rho', 'tail_compl_inf_tol', 'q_blocks', 'voltage_pin'},
        'coupled_capture_fields': set(H.S56_CYCLE_CAPTURE_LINE_FIELDS_BY_ARM[H.S56_CAPTURE_ARM_COUPLED]) >= {
            'primal_expectation', 'primal_deviation', 'dual_expectation_v', 'rho', 'tail_compl_inf_tol', 'q_blocks',
            'voltage_pin'},
        'q_primary_per_arm': H.S56_CYCLE_CAPTURE_Q_PRIMARY == {H.S56_CAPTURE_ARM_COUPLED: 'settlement_excluded_in_full',
                                                               H.S56_CAPTURE_ARM_PINNED: 'contracted_excluded'},
        'pinned_record_fields': ("'s56_cycle_capture_checklist_asserted_before_run'" in hsrc
                                 and 'def s56_cycle_capture_pinned_record_fields' in hsrc),
        'tail_per_round_records': ("'compl_inf_tol_in_force'" in open('network.py', encoding='utf-8').read()
                                   and "record['round'] = round_index" in srp_text),
        'idc_metadata_scale_rating_base': all(s in idc_text for s in ("'admm_objective_scale_tso'",
                                                                       "'interface_rating_mva'", "'tso_base_mva'")),
        'idc_deviation_dual_fields': 'lambda_dev_' in idc_text,
        'per_cycle_scenario_mismatch_fields': all(f in H.PER_CYCLE_INTERFACE_COUPLING_FIELDS for f in (
            'scenario_mismatch_v_max_pu', 'scenario_mismatch_v_rms_pu', 'priced_residual_dual_part',
            'boyd_pf_r_expectation_part')),
        'per_cycle_boyd_ratios': all(f in H.PER_CYCLE_RECORD_FIELDS for f in (
            'boyd_v_primal_ratio', 'boyd_v_dual_ratio', 'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio',
            'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_all_pass', 'non_optimal_blocks')),
        'record_rule_ten_and_failures': all(s in hsrc for s in ("'rule_ten': {", "'local_solve_failures'",
                                                               "'peak_rss'")),
        'production_deviation_mismatch_conversion': (
            "mu = store_duals[node_id][year][day]['p'][k][p] / s_base_t * effective_scale / rating_mva" in srp_text),
    })
    failed = sorted(k for k, v in checks.items() if v is not True)
    if failed:
        raise RuntimeError(f'capture path missing (before any launch): {failed}')
    return checks


# ======================================================================================================================
#  the frozen spec
# ======================================================================================================================
def build_spec():
    smoke = _load(SMOKE_SPEC)
    cg = SM.coupling_gates_literals()
    inst = deepcopy(smoke['instances'])
    central = deepcopy(smoke['central'])
    central['2025|Summer'] = {'C_central_eur_per_day': MERGED_ONLY['2025|Summer'], 'status': 'MERGED-ONLY',
                              'role': 'reported, not calibrated',
                              'source': ('step 3 v2 S3-2 merged-default solution (185,674.1557530493 EUR/day); Planner '
                                         'ruling at 98b7f597: labelled merged-only (bias +0.035 EUR/day on 26 pairs), '
                                         'not used for calibration')}
    sum12 = smoke['sum_12_w']
    blocks = {}
    for bk in ('2025|Winter', '2035|Summer', '2025|Summer'):
        i = 'A' if bk == '2025|Winter' else 'B'
        w = inst[i]['w_b_own'][bk]
        blocks[bk] = {'instance': i, 'w_b': w, 'C_central_eur_per_day': central[bk]['C_central_eur_per_day'],
                      'w_b_C_central_eur': w * central[bk]['C_central_eur_per_day'],
                      'calibrated': bk in ALLOWANCE_EUR,
                      'allowance_eur': ALLOWANCE_EUR.get(bk),
                      'allowance_formula_value': (2500.0 * w / sum12) if bk in ALLOWANCE_EUR else None,
                      'D1_floor_eur': D1_FLOOR_EUR.get(bk), 'delta_pin_eur_per_day': DELTA_PIN_EUR_PER_DAY.get(bk)}
    per_round = {i: (1 + 3) * inst[i]['n_years'] * inst[i]['n_days'] + 3 for i in INSTANCES}
    arms = {}
    for a in ARMS:
        arms[a] = {'options': arm_options(a, cg),
                   'role': {'V1': 'calibration arm; formulation of record (Expert ruling 2)',
                            'V2': V2_DECLARATION + ' (Expert ruling 2, variant (b))',
                            'pinned_aa_off': 'control (option off, AA off via the harness override)',
                            'pinned_aa_on': 'control (option off, case-default AA)'}[a],
                   'aa': smoke['arms'][a]['aa'], 'init_identity_group': smoke['arms'][a]['init_identity_group'],
                   'q_primary': 'settlement_excluded_in_full' if a in COUPLED else 'contracted_excluded'}
    return {
        'stage': ('P5.15 S56 validation step 4 MAIN RUN: arms V-1, V-2 (measured, non-deciding), pinned AA-off, pinned '
                  'AA-on on instance A (2025|Winter) and instance B ({2025, 2035} x Summer); x0, alpha 0, cap 300; '
                  'cycle-40 look; continuation to 600 declared'),
        'version': 1, 'kind': 'addendum',
        'predecessor': {'path': SMOKE_SPEC, 'sha256': _sha(SMOKE_SPEC),
                        'relation': ('the step 4 smoke spec v2 (e36c29bd). This addendum adds the main run on the same '
                                     'arms, instances, configuration and candidate; the pinned arms add the per-cycle '
                                     'capture key s56_cycle_capture_pinned (a662c9bf). Nothing in the smoke spec is '
                                     'edited')},
        'cites': {
            'smoke_spec_v2': {'path': SMOKE_SPEC, 'sha256': _sha(SMOKE_SPEC)},
            'diag_spec_ff466289': {'path': SM.DIAG_SPEC, 'sha256': _sha(SM.DIAG_SPEC)},
            'design': {'path': DESIGN, 'sha256': _sha(DESIGN),
                       'governs': 'its "Status: ruled (2026-11-23)" paragraph governs where the design text conflicts'},
            'plan_v4': {'path': PLAN_V4, 'sha256': _sha(PLAN_V4), 'sections': ['3', '7 (row 4)']},
            'tasks': {'path': TASKS, 'sha256_at_freeze': _sha(TASKS),
                      'items': 'S56 "Expert rulings on the handback" items 1, 2, 7; Order (2026-11-23) step 4 main run'},
            'smoke_evaluation': {'path': SMOKE_EVALUATION, 'sha256': _sha(SMOKE_EVALUATION)},
            'smoke_launcher': {'path': SMOKE_LAUNCHER, 'sha256': _sha(SMOKE_LAUNCHER),
                               'reused_by_import': ['arm_options', 'coupling_gates_literals', 'capture_path_checklist',
                                                    'root_path', 'SMOKE_DIR', 'DIAG_SPEC']},
            'diag_results': {'path': SM.DIAG_RESULTS, 'sha256': _sha(SM.DIAG_RESULTS)},
        },
        'authority': ['TASKS.md S56: Expert rulings on the handback (2026-11-23) items 1, 2, 7; author authorization of '
                      'the step 4 main run inside the existing 10-20 h',
                      'S56_STEP4_DESIGN.md (ruled status paragraph governs)', 'S56_PLAN_v4.md sections 3 and 7',
                      'Planner task 2026-11-23 (step 4 main run)'],
        'launcher': SCRIPT,
        'harness': {'path': 'p515_s44_campaign_harness.py', 'sha256': _sha('p515_s44_campaign_harness.py'),
                    'commit_of_pinned_capture_key': 'a662c9bf'},
        'coupling_gates_literals_source': {'path': SM.COUPLING_GATES, 'sha256': _sha(SM.COUPLING_GATES)},
        # ---- (i) arms, instances, cap
        'i_arms_instances_cap': {
            'arms': arms, 'v2_declaration': V2_DECLARATION,
            'instances': inst, 'cap': CAP['main'], 'x0': smoke['candidate'], 'alpha': smoke['alpha'],
            'runs': [{'phase': 'main', 'instance': i, 'arm': a, 'label': label_of('main', a),
                      'campaign_root': os.path.relpath(root_path('main', i), REPO)} for i in ('B', 'A') for a in ARMS],
            'execution_order': ('instance B: one campaign, its four arms concurrent (concurrency 4); then instance A: '
                                'one campaign, its four arms concurrent. The harness campaign lock is one per checkout '
                                'and the derived instance is campaign-level, so A cannot run concurrently with B; the '
                                'task\'s memory rule for starting A during B is therefore moot (A runs after B, its '
                                'stated fallback). vm_stat recorded before and during each campaign'),
            'configuration': smoke['configuration'], 'required_consecutive_cycles': smoke['required_consecutive_cycles'],
            'sigma_fixed_expected': smoke['sigma_fixed_expected'],
            'stopping': ('production: a cycle converges when every Boyd channel passes and every local solve succeeded; '
                         'the run stops after required_consecutive_cycles (10) consecutive converged cycles or at the '
                         'cap. The convergence-depth tail (compl_inf_tol 1e-6) is active in cycle k+1 iff cycle k '
                         'converged (production _convergence_depth_tail_next_state; not sticky)')},
        # ---- (ii) central values and allowances
        'ii_central_and_allowance': {'blocks': blocks, 'sum_12_w': sum12, 'R_eur': 10000.0, 'R_over_4_eur': 2500.0,
                                     'allowance_rule': 'e_b <= (R/4) x w_b / Sigma_12 w (S56_PLAN_v4 section 3)',
                                     'central': central},
        # ---- (iii) tail depth
        'iii_tail_depth_only': {
            'rule': ('calibration and acceptance on tail-depth cycles only (Expert X4-2 (a)); the production-depth floor '
                     '(D-1: +597.67 EUR 2025|Winter, +499.03 EUR 2035|Summer, one-signed) is reported as a limitation'),
            'tail_depth_cycle': ('cycle k (k >= 1) is tail-depth iff every record of round k in the run\'s '
                                 'network_ipopt_solve_records.jsonl (agents TSO, DSO5, DSO7, DSO9; every attempt) has '
                                 'compl_inf_tol_in_force == 1e-6, and there is at least one such record'),
            'cross_check': ('the S56 capture line of cycle k (read after the cycle\'s solves) records the options in '
                            'force for the NEXT cycle\'s solves; recorded beside, never used to classify')},
        # ---- (iv) the cycle-40 look
        'iv_cycle40_look': {
            'when': 'once V-1 of the instance has written its cycle-40 capture line (or the run ended before 40)',
            'what': ('per calibrated block b of the instance (A: 2025|Winter; B: 2035|Summer), e_b(k) = Q_b(k) - w_b x '
                     'C_central, Q_b = V-1 capture q_blocks.per_year_day[b].settlement_excluded_in_full'),
            'fit': ('ordinary least squares of ln|e_b(k)| = a + k ln r on cycles 21..34; validated on the held-out '
                    'cycles 35..40: the held-out error is max over those cycles of |ln(|e_hat(k)| / |e_b(k)|)| '
                    '(recorded with the predicted and observed values); the projection refits on cycles 21..40'),
            'projection': ('k_star = 40 + ln(allowance_b / |e_hat(40)|) / ln r when r < 1 and |e_hat(40)| > allowance; '
                           'k_star = 40 when |e_hat(40)| <= allowance; k_star = infinity when r >= 1'),
            'stop_rule': ('if k_star > 600 on a calibrated block, STOP AND REPORT: the running campaign is terminated '
                          'through the harness\'s own interrupt path (SIGINT to the launcher parent; the harness '
                          'SIGTERMs and reaps its children, records written up to the stop are kept) and nothing '
                          'further is launched. Otherwise continue'),
            'recorded_beside_not_deciding': ('the same fit on e_b(k) - F_b with F_b the D-1 production-depth floor '
                                             '(the cycles of the look are production-depth until the first Boyd pass); '
                                             'the tail-depth classification of cycles 1..40'),
            'output': 'look40_<inst>_<utc>.json in the main directory'},
        # ---- (v) continuation
        'v_continuation': {
            'arms': 'V-1; V-2 too if it is still contracting at cycle 300',
            'cap': CAP['cont600'],
            'condition': ('per instance and arm: the cap-300 run has not settled (record certification.certified is '
                          'not True, i.e. it stopped at the cap) AND its fitted contraction rate r < 1, where r is the '
                          'OLS rate of ln|Q_b(k) - Q_b(k-1)| (V-1: of ln|e_b(k)|) on cycles 261..300, per calibrated '
                          'block; V-2 "still contracting" means the same step-rate fit gives r < 1 on its calibrated '
                          'block'),
            'how': ('a NEW campaign root campaign_s56s4main_v1_cont600_<inst> (labels x0_<arm>_cont600), fresh runs '
                    'from cycle 1 with the same code, configuration and seeding (the harness has no generic resume); '
                    'its cycles 1..300 are compared with the cap-300 run (recorded), its cycles 1..3 with the smoke '
                    '(M3)'),
            'budget_guard': ('launched only if the measured wall of the step 4 runs so far plus the projected '
                             'continuation wall (measured s/cycle x 600) stays inside the author\'s 20 h; otherwise '
                             'stop and report (machine allocation is the author\'s)'),
            'when': 'after both main campaigns, from the same launcher and spec'},
        # ---- (vi) predictions
        'vi_predictions': {
            'must_hold': PREDICTIONS_MUST_HOLD,
            'must_hold_failure': ('a harness or wiring fault: stop and report (stated with the local-solution caveat: '
                                  'these are identities and equalities of the implementation, not optimality claims)'),
            'recorded': PREDICTIONS_RECORDED,
            'recorded_failure': 'reported, not a stop',
            'also_record': ALSO_RECORD,
            'operational': operational_definitions(),
        },
        'cap': CAP, 'concurrency_per_campaign': 4,
        'solves': {'launcher_parent': 0, 'look40': 0, 'evaluate': 0, 'per_round': per_round,
                   'per_run': 'per_round x (cycles_run + 1) + every recovery retry actually attempted (credited per '
                              'failure event by the solve_profile identity)'},
        'labels': {p: {a: label_of(p, a) for a in ARMS} for p in PHASES},
        'm3': {'cycles': M3_CYCLES, 'files': M3_FILES, 'excluded_keys_declared_in_advance': list(M3_EXCLUDED_KEYS),
               'equality': ('canonical json.dumps(sort_keys=True) of each line / record with the excluded keys removed '
                            'recursively; floats compare by their JSON repr (exact round trip)'),
               'smoke_counterpart': 'the smoke run of the same instance and arm (spec e36c29bd, root '
                                    'campaign_s56s4smoke_v2_<inst>_<arm>)'},
        'memory_rule': ('record vm_stat (free + inactive GiB) before each campaign and during it (at the cycle-40 look '
                        'and at least hourly while a campaign runs)'),
        'failure_policy': ('M1-M5 failure: stop and report (harness or wiring fault). A harness fault (exception, guard '
                           'mismatch, child exit != 0, an initialisation-identity mismatch raised by the harness) is '
                           'recorded and stops the order. P3-P9: recorded, never re-run or tuned. The cycle-40 stop '
                           'rule as in (iv)'),
        'not_permitted': ['production or harness edits', 'any run not in this spec', 'cap changes',
                          'overwriting committed artifacts', 'detached runs (screen, nohup, shell backgrounding)',
                          'pgrep -f wait loops', 'writes into .claude/worktrees/'],
        'objective_convention': ('Q in EUR of total discounted cost. Coupled arms: primary gross_operational_cost = '
                                 'settlement excluded in full (pin excluded, salvage separate), contracted-excluded '
                                 'beside it. Pinned arms: gross_operational_cost = contracted settlement and pin '
                                 'excluded. C_central: step 3 gross EUR/day (settlement weight 0, no row 18, no pin, no '
                                 'salvage). e_b(k) = Q_b(k) - w_b C_central and D_b = w_b C_central - Q_pin,b in EUR of '
                                 'total discounted cost'),
        'measured_vs_projected': 'every reported value is labelled measured, fitted or projected',
    }


def operational_definitions():
    return {
        'Q_b': ('per cycle, from the run\'s s56_cycle_capture.jsonl q_blocks.per_year_day[b]: coupled arms '
                'settlement_excluded_in_full (contracted_excluded beside), pinned arms contracted_excluded'),
        'M1': ('coupled arms (V1, V2), every cycle: (i) rel(priced_residual_price_part_c, -T_c) <= 1e-9, T_c = sum over '
               'agent blocks of weight x settlement_total_unweighted (capture); (ii) rel(gross_operational_cost_'
               'contracted_excluded_c - gross_operational_cost_c, D_c) <= 1e-9, D_c = sum over agent blocks of weight x '
               '(settlement_total_unweighted - settlement_contracted_unweighted), plus the DSO decomposition against '
               'per_cycle_response covariance_dso_weighted <= 1e-9 every cycle and against the record\'s '
               'interface_settlement_deviation_total at the terminal cycle; (iii) every deviation-dual coordinate of '
               'interface_duals_per_cycle.jsonl: |sum_s omega_s mu_s| <= 1e-9 x max|mu| (per cycle, agent, kind); '
               '(iv) V1: the capture\'s voltage_pin.total == 0.0 and every voltage_pin.per_agent weighted == 0.0 with '
               'pin_constructed False, EVERY cycle (a662c9bf field), and the terminal record voltage_pin_total == 0.0. '
               'rel = |a - b| / max(|a|, |b|, 1)'),
        'M2': ('per instance: the main V2, pinned_aa_off and pinned_aa_on initialisation records have equal '
               'gross_operational_cost_hex and equal components_hex, equal to the smoke V2 / pinned records\' hex; '
               'every harness comparison of these three has all_equal True and n_compared >= 1; main V1 group '
               '"voltage_pin_not_constructed", its hex equal to the smoke V1 hex and different from the V2 hex, '
               'compared only within its group'),
        'M3': 'per run: the files and exclusions of the spec\'s "m3" block; held iff every compared item is equal',
        'M4': ('every arm: the record\'s s56_cycle_capture_checklist_asserted_before_run all True (coupled: the coupled '
               'checklist; pinned: the pinned checklist), s56_cycle_capture_summary ok with cycle_lines_written == '
               'cycles_run, no capture line with captured False; per cycle rel(sum over (year, day) of the arm\'s '
               'primary Q, per_cycle_record gross_operational_cost) <= 1e-12'),
        'M5': ('every run: solve_profile.identity_holds True, base_solves == per_round x (cycles_run + 1), '
               'blocked_solve == blocked_exec == 0, exit code 0 (parent view and exit_code.txt); launcher guard 0'),
        'P3': ('V1, each calibrated block: T = the tail-depth cycles of the run (cap-300 run, and the continuation when '
               'run). Held iff |e_b(K)| <= allowance_b at the last tail-depth cycle K. Recorded: min over T of |e_b|, '
               'its cycle, R_eff = 4 x min|e_b| x Sigma_12 w / w_b, and the dominant term = the largest of {S_b (settling '
               'bar, tail-depth cycles), |PR_b(K)| (full-channel priced residual), the D-1 tail-depth floor}. If T is '
               'empty: not evaluable (recorded with the production-depth values)'),
        'P4': ('D_b = w_b C_central - Q_pin,b(K), K the last cycle of the pinned AA-off run (contracted-excluded), its '
               'bar = S_b of that run. Recorded: D_b >= 0 on 2035|Summer; |D_b(2025|Winter)| <= |D_b(2035|Summer)| / 3 '
               '("much smaller", operationalised so); a negative D_b is not a defect (Expert ruling 7)'),
        'P5': ('V2, each calibrated block: lim Q_V2,b = the validated extrapolation Q_hat_inf on the run\'s tail-depth '
               'cycles (the terminal value, labelled, when no validated extrapolation exists); held iff lim Q_V2,b - '
               'w_b (C_central + Delta_pin_b) <= S_b(V2); the local-solution caveat is stated beside'),
        'P6': ('per instance: r_pf,exp(k) = sqrt(sum over blocks, periods, kinds p and q of ((x_dso - z_tso) / '
               'rating_MVA)^2) from the capture\'s primal_expectation (the same formula for every arm; for the coupled '
               'arms it reproduces production\'s boyd_pf_r_expectation_part, checked). L = max over the four arms of '
               'min_k r_pf,exp(k). cycles(arm) = first k with r_pf,exp(k) <= L; wall(arm) = mean cycle_wall_s over '
               'cycles 1..cycles(arm). Held iff 1.0 <= cycles(V1)/cycles(AA-off) <= 2.0 and 0.9 <= wall(V1)/wall(AA-'
               'off) <= 1.2 and cycles(AA-off)/cycles(AA-on) >= 2; the V2 ratios recorded beside'),
        'P7': ('V1, each calibrated block, cycles k >= 21: PR_b(k) = price_b(k) + sum over DNs and periods of [lambda_P '
               'dPbar + lambda_Q dQbar] x w_b eff_b / (s_T R) + lambda_V dVbar x w_b eff_b / v_b^2 + sum_s omega_s '
               '[(mu_P,s dP_s + mu_Q,s dQ_s) x w_b eff_b / (s_T R) + mu_V,s dV_s x w_b eff_b / v_b^2], with lambda / mu '
               'the TSO-side stored expectation / deviation duals (interface_duals_per_cycle.jsonl lambda_pf_*_tso, '
               'lambda_dev_*_tso; capture dual_expectation_v tso), dXbar = TSO - DSO expectation copies (capture), dX_s '
               '= (xbar_T + d_T,s) - (xbar_D + d_D,s) (capture expectation + deviation copies; MW, MVAr, kV), eff_b, s_T '
               'and R from the interface-dual header metadata (admm_objective_scale_tso, tso_base_mva, '
               'interface_rating_mva), v_b the interface base kV (= capture kV / interface_voltage_terminal pu, checked '
               'constant), price_b(k) = -sum over the block\'s agents of weight x settlement_total_unweighted. The '
               'P-deviation part is validated against production\'s priced_residual_dual_part (sum over blocks, rel '
               '1e-9 every cycle). Held iff sum_{k >= 21} |Q_b + PR_b - w_b C| <= (1/5) sum_{k >= 21} |Q_b - w_b C| on '
               'every calibrated block; the per-cycle ratios recorded'),
        'P8': ('per run: K1 = the first tail-depth cycle; the step Q(K1) - Q(K1 - 1) total and per block (EUR); "no '
               'tail onset" when none'),
        'P9': 'per child: record peak_rss.child_python_process_ru_maxrss <= 6 GiB; IPOPT maximum beside',
        'S_b': ('validated extrapolation on the last W = min(20, n) cycles of the window (V1 calibration: tail-depth '
                'cycles only; other uses: the run\'s last cycles): steps delta_k = Q_b(k) - Q_b(k-1) on consecutive '
                'cycles; OLS ln|delta_k| = a + k ln r on the window minus its last h = 3 cycles; validated by predicting '
                'Q_b on the h held-out cycles (geometric sum from the last fitted cycle), H = max |predicted - '
                'observed|; refit on the whole window, Q_hat_inf = Q_b(K) + delta_K r / (1 - r); S_b = |Q_hat_inf - '
                'Q_b(K)| + H. Valid only with >= 6 steps of one sign and r < 1; otherwise S_b = the range of Q_b over '
                'the last 10 cycles, labelled "range (no validated extrapolation)"'),
        'working_level': ('V1, each calibrated block: m(k) = max over channels v, pf, ess of max(primal_ratio, '
                          'dual_ratio) (per_cycle_record; production tolerance = 1). For a level s, k_s = the first '
                          'tail-depth cycle with m(k) <= s. Working level s* = the largest s (over the observed m '
                          'values at tail-depth cycles) such that |e_b(k)| <= allowance_b at every tail-depth cycle k >= '
                          'k_s. Expressed at k_s*: s*, k_s*, and the block\'s max and omega-RMS of dP_s, dQ_s, dV_s '
                          '(capture), and A_b = the absolute form of PR_b (sum of absolute terms); "not attained" when no '
                          's qualifies'),
        'M4_effort': ('V1 and V2, each calibrated block: the first cycle after which |Q_b(k) - Q_hat_inf_b| <= '
                      'allowance_b at every later cycle (each arm against its own extrapolated limit), the cumulative '
                      'cycle_wall_s to it, and cycles x wall ratios V1 / V2'),
        'dV_trajectory': ('V1 and V2, every cycle: production scenario_mismatch_v_max_pu and _rms_pu (pooled) and, per '
                          'block, the max and omega-RMS of dV_s from the capture'),
        'terminal_step_to_threshold': ('per run: record rule_ten.terminal_step_over_threshold (objective change / '
                                       'objective tolerance, reported-not-gated) and the terminal Boyd ratio max per '
                                       'channel'),
        'local_solve_failures': 'per run: record local_solve_failures and per-cycle non_optimal_blocks',
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


def _spec(path):
    spec = _load(path)
    sha = _sha(path)
    tag = os.path.basename(path).rsplit('_', 1)[-1].split('.')[0]
    if not sha.startswith(tag):
        raise RuntimeError(f'spec sha256 {sha[:8]} does not match its name tag {tag}')
    apply_spec_roots(spec)
    return spec, sha, os.path.relpath(os.path.abspath(path), REPO)


# ======================================================================================================================
#  one campaign (one instance, one phase)
# ======================================================================================================================
def run(args):
    t0 = time.time()
    spec, spec_sha, spec_rel = _spec(args.spec)
    phase, inst = args.phase, args.instance
    arms = list(ARMS) if phase == 'main' else [a for a in (args.arms or '').split(',') if a]
    tag = f'{ROOT_TAG}_{phase}_{inst}'
    rec = {'schema': 'p515_s56_step4_main_launcher_v1', 'phase': phase, 'instance': inst, 'arms': arms,
           'utc_start': _utc(), 'pid': os.getpid(), 'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT,
           'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'), 'refusals': [],
           'error': None}
    rec_path = os.path.join(MAIN_DIR, f'{tag}_launcher.json')
    pid_path = os.path.join(MAIN_DIR, f'{tag}_launcher_pid.txt')
    if os.path.exists(rec_path):
        _log(f'REFUSED: {rec_path} exists (write-once)')
        return 1
    ref = rec['refusals']
    if phase == 'main' and args.arms:
        ref.append('--arms is for the continuation only (the main phase runs the four arms)')
    if phase == 'cont600' and (not arms or any(a not in ('V1', 'V2') for a in arms) or arms[0] != 'V1'):
        ref.append(f'continuation arms must be V1 [, V2]: {arms}')
    if phase == 'main' and inst == 'A':
        b_rec = os.path.join(MAIN_DIR, f'{ROOT_TAG}_main_B_launcher.json')
        if not os.path.exists(b_rec) or (_load(b_rec).get('result_summary') or {}).get('all_exit_0') is not True:
            ref.append('instance A main runs only after instance B main completed with every child exit 0')
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        ref.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
    for rel in (SCRIPT, spec_rel, 'p515_s44_campaign_harness.py', SMOKE_LAUNCHER):
        if not SM.committed_clean(rel):
            ref.append(f'{rel} not committed clean')
    others = other_copies_of_this_launcher()
    if others:
        ref.append(f'another copy of this launcher is running: {others}')
    root = root_path(phase, inst)
    if os.path.exists(root):
        ref.append(f'campaign root exists: {root}')
    try:
        rec['capture_path_checklist'] = capture_path_checklist()
    except Exception as error:  # noqa: BLE001
        ref.append(f'capture-path checklist: {error}')
    for key, item in spec['cites'].items():
        if 'sha256' in item and _sha(item['path']) != item['sha256']:
            ref.append(f'{key}: {item["path"]} sha256 differs from the spec')
    if _sha('p515_s44_campaign_harness.py') != spec['harness']['sha256']:
        ref.append('harness sha256 differs from the spec')
    seeds = []
    for a in ARMS:
        try:
            seeds.append((a,) + smoke_init_record(inst, a))
        except Exception as error:  # noqa: BLE001
            ref.append(f'smoke initialisation record {inst}/{a}: {error}')
    if ref:
        _log(f'REFUSED before launch (zero solves): {ref}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        os.makedirs(MAIN_DIR, exist_ok=True)
        write_json(rec_path, rec)
        return 3
    i_block = spec['i_arms_instances_cap']
    cfg = deepcopy(i_block['configuration'])
    cfg['name'] = (f"S56 step 4 {phase}: instance {inst} ({', '.join(i_block['instances'][inst]['blocks'])}), arms "
                   f"{', '.join(arms)}; x0, alpha 0, cap {spec['cap'][phase]}")
    cfg['derived_instance'] = H.validate_derived_instance(i_block['instances'][inst]['derived_instance'])
    cg = SM.coupling_gates_literals()
    candidates = []
    for a in arms:
        options = deepcopy(i_block['arms'][a]['options'])
        if options != arm_options(a, cg):
            raise RuntimeError(f'arm {a}: spec options differ from the launcher construction')
        candidates.append((label_of(phase, a), dict(SM.X0), options))
    failures = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT, spec_rel, SMOKE_LAUNCHER))
    if failures:
        rec['refusals'] = failures
        _log(f'REFUSED by the harness preconditions (zero solves): {failures}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        write_json(rec_path, rec)
        return 3
    os.makedirs(MAIN_DIR, exist_ok=True)
    with open(pid_path, 'x') as handle:
        handle.write(f'{os.getpid()}\n')
    spec_path, csha, cspec = H.freeze_campaign_spec(
        root, root_name(phase, inst), candidates, cfg, spec['cap'][phase], len(candidates), spec['authority'],
        required_consecutive_cycles=i_block['required_consecutive_cycles'],
        extra={'main_spec': {'path': spec_rel, 'sha256': spec_sha}, 'phase': phase, 'instance': inst, 'arms': arms,
               'campaign_script': SCRIPT, 'campaign_script_sha256': _sha(SCRIPT),
               'objective_convention': spec['objective_convention'], 'v2_declaration': V2_DECLARATION})
    rec['campaign'] = {'root': root, 'spec': {'path': os.path.relpath(spec_path, REPO), 'sha256': csha},
                       'entries': [{'label': e['label'], 'eval_key': e['eval_key'], 'eval_dir': e['eval_dir']}
                                   for e in cspec['candidates']]}
    idir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    os.makedirs(idir)
    rec['init_identity_seeded'] = []
    for a, src, name in seeds:
        dst = os.path.join(idir, name)
        shutil.copyfile(src, dst)
        rec['init_identity_seeded'].append({'from_smoke_arm': a, 'file': name, 'sha256_src': _sha(src),
                                            'sha256_dst': _sha(dst)})
    lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
    _log(f'{phase} {inst}: lock {lock}; campaign spec {os.path.relpath(spec_path, REPO)}; entries '
         f'{[e["label"] for e in cspec["candidates"]]}; seeded {[s[0] for s in seeds]}')
    records = None
    try:
        ctx = H.CampaignContext(root, spec_path, csha, cspec, log=_log)
        records = H.evaluate([label_of(phase, a) for a in arms], ctx)
        rec['batch_info'] = getattr(H.evaluate, 'last_batch_info', None)
    except BaseException as error:  # noqa: BLE001 -- a harness fault or the declared look-40 interrupt, recorded
        rec['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f'HARNESS FAULT / INTERRUPT: {rec["error"]["error"]}')
    finally:
        rec['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
    rec['results'] = []
    for a, r in zip(arms, records or [None] * len(arms)):
        r = r or {}
        rec['results'].append({'arm': a, 'status': r.get('status'), 'cycles_run': r.get('cycles_run'),
                               'exit_code': (r.get('parent_view') or {}).get('exit_code'),
                               'wall_s_parent_view': (r.get('parent_view') or {}).get('wall_s'),
                               'solve_profile': r.get('solve_profile'), 'peak_rss': r.get('peak_rss'),
                               'eval_dir': r.get('eval_dir')})
    rec['result_summary'] = {'all_exit_0': bool(records) and all(x['exit_code'] == 0 for x in rec['results']),
                             'n_records': len([x for x in rec['results'] if x['status'] is not None])}
    rec['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    rec['wall_s'] = time.time() - t0
    rec['utc_end'] = _utc()
    write_json(rec_path, rec)
    ok = rec['error'] is None and rec['result_summary']['all_exit_0'] and not rec['guard']['verify_failures']
    _log(f"{phase} {inst}: " + '; '.join(f"{x['arm']} exit {x['exit_code']} cycles {x['cycles_run']} status "
                                        f"{x['status']}" for x in rec['results']) + f"; wall {rec['wall_s']:.0f}s -> "
         f"{rec_path}")
    return 0 if ok else 2


# ======================================================================================================================
#  the cycle-40 look (zero solves; spec item (iv))
# ======================================================================================================================
def _complete_jsonl(path):
    """Complete lines only (the file may be appended to while it is read)."""
    with open(path, 'rb') as handle:
        data = handle.read()
    lines = data.split(b'\n')
    out = []
    for ln in lines[:-1]:   # the last element is '' (complete file) or a partial line
        if ln.strip():
            out.append(json.loads(ln))
    return out


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def tail_depth_rounds(solve_records):
    by_round = {}
    for r in solve_records:
        by_round.setdefault(r.get('round'), []).append(r.get('compl_inf_tol_in_force'))
    return {k: (len(v) > 0 and all(x == TAIL_TOL for x in v)) for k, v in by_round.items() if k is not None}


def contraction_look(series, allowance, look):
    """series: {cycle: e}. The fit, validation and projection of spec item (iv)."""
    f0, f1 = look['fit_cycles']
    h0, h1 = look['heldout_cycles']
    p0, p1 = look['projection_fit_cycles']
    out = {'fit_cycles': [f0, f1], 'heldout_cycles': [h0, h1], 'projection_fit_cycles': [p0, p1]}
    need = list(range(p0, p1 + 1))
    if any(k not in series or series[k] is None or series[k] == 0.0 for k in need):
        out['status'] = 'insufficient cycles (or a zero / missing value) for the declared windows'
        return out
    xs = list(range(f0, f1 + 1))
    a, b = _ols(xs, [math.log(abs(series[k])) for k in xs])
    held = []
    for k in range(h0, h1 + 1):
        pred = math.exp(a + b * k)
        held.append({'cycle': k, 'abs_e_observed': abs(series[k]), 'abs_e_predicted': pred,
                     'abs_log_error': abs(math.log(pred / abs(series[k])))})
    out['fit'] = {'a': a, 'ln_r': b, 'r': math.exp(b)}
    out['heldout'] = held
    out['heldout_max_abs_log_error'] = max(x['abs_log_error'] for x in held)
    xs2 = list(range(p0, p1 + 1))
    a2, b2 = _ols(xs2, [math.log(abs(series[k])) for k in xs2])
    r2 = math.exp(b2)
    e_hat = math.exp(a2 + b2 * p1)
    if e_hat <= allowance:
        k_star = float(p1)
    elif r2 < 1.0:
        k_star = p1 + math.log(allowance / e_hat) / b2
    else:
        k_star = math.inf
    out['projection_fit'] = {'a': a2, 'ln_r': b2, 'r': r2, 'abs_e_hat_at_last': e_hat}
    out['k_star'] = k_star
    out['status'] = 'evaluated'
    return out


def look40(args):
    spec, spec_sha, spec_rel = _spec(args.spec)
    inst = args.instance
    look = spec['iv_cycle40_look']
    root = root_path('main', inst)
    blocks = spec['ii_central_and_allowance']['blocks']
    out = {'schema': 'p515_s56_step4_look40_v1', 'utc': _utc(), 'instance': inst,
           'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT, 'script_sha256': _sha(SCRIPT),
           'definition': look, 'blocks': {}, 'error': None}
    path = os.path.join(MAIN_DIR, f'look40_{inst}_{_stamp()}.json')
    try:
        ev = os.path.join(root, 'evals', [d for d in sorted(os.listdir(os.path.join(root, 'evals')))
                                          if d.endswith(H._sanitize_id(label_of('main', 'V1')))][0])
        cap_lines = _complete_jsonl(os.path.join(ev, H.S56_CYCLE_CAPTURE_FILE))
        lines = [x for x in cap_lines if not x.get('header')]
        out['n_capture_lines'] = len(lines)
        out['eval_dir'] = os.path.relpath(ev, REPO)
        out['run_ended'] = os.path.exists(os.path.join(ev, 'exit_code.txt'))
        solve_recs = _complete_jsonl(os.path.join(ev, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE))
        tail = tail_depth_rounds(solve_recs)
        out['tail_depth_cycles_1_40'] = sorted(k for k, v in tail.items() if v and 1 <= k <= 40)
        for bk in CALIBRATED[inst]:
            b = blocks[bk]
            wc = b['w_b_C_central_eur']
            series = {x['cycle']: (x['q_blocks']['per_year_day'][bk]['settlement_excluded_in_full'] - wc
                                   if x.get('captured') else None) for x in lines if x['cycle'] <= look['cycle']}
            res = contraction_look(series, b['allowance_eur'], look)
            floor = b['D1_floor_eur']['production_depth']
            adj = {k: (v - floor if v is not None else None) for k, v in series.items()}
            res_adj = contraction_look(adj, b['allowance_eur'], look)
            stop = res.get('status') == 'evaluated' and res['k_star'] > look['stop_if_projected_cycles_exceed']
            out['blocks'][bk] = {'w_b_C_central': wc, 'allowance_eur': b['allowance_eur'],
                                 'e_b_by_cycle_measured': {str(k): v for k, v in sorted(series.items())},
                                 'look': res, 'stop_rule_fires': stop,
                                 'recorded_beside_floor_adjusted': {'F_b_production_depth_eur': floor,
                                                                    'look': res_adj}}
        out['stop_rule_fires'] = any(v['stop_rule_fires'] for v in out['blocks'].values())
        if out['n_capture_lines'] < look['cycle'] and not out['run_ended']:
            out['note'] = 'look taken before cycle 40 was written (not the declared look)'
    except Exception:  # noqa: BLE001
        out['error'] = traceback.format_exc()
    out['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    write_json(path, out)
    _log(f"look40 {inst} -> {os.path.relpath(path, REPO)}: lines {out.get('n_capture_lines')} stop "
         f"{out.get('stop_rule_fires')} " + '; '.join(
             f"{bk}: r {((v['look'].get('projection_fit') or {}).get('r'))} k* {v['look'].get('k_star')}"
             for bk, v in out['blocks'].items()) + f" error {out['error'] is not None}")
    return 0 if out['error'] is None else 2


# ======================================================================================================================
#  vm_stat snapshot
# ======================================================================================================================
def vmstat(args):
    if args.spec:
        _spec(args.spec)   # v2: the snapshot goes to the spec's main directory
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(text.split('page size of ')[1].split(' ')[0])
    vals = {}
    for ln in text.splitlines()[1:]:
        if ':' in ln:
            k, v = ln.split(':', 1)
            try:
                vals[k.strip()] = int(v.strip().rstrip('.'))
            except ValueError:
                pass
    free = vals.get('Pages free', 0) * page / GiB
    inactive = vals.get('Pages inactive', 0) * page / GiB
    d = os.path.join(MAIN_DIR, 'vmstat')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'vmstat_{_stamp()}_{args.tag}.json')
    write_json(path, {'utc': _utc(), 'tag': args.tag, 'page_size': page, 'free_gib': free, 'inactive_gib': inactive,
                      'free_plus_inactive_gib': free + inactive,
                      'active_gib': vals.get('Pages active', 0) * page / GiB,
                      'wired_gib': vals.get('Pages wired down', 0) * page / GiB, 'raw': text})
    _log(f'vm_stat {args.tag}: free {free:.2f} GiB, inactive {inactive:.2f} GiB, free+inactive {free + inactive:.2f} GiB '
         f'-> {os.path.relpath(path, REPO)}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--look40', action='store_true')
    ap.add_argument('--vmstat', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--out-dir')
    ap.add_argument('--phase', choices=PHASES)
    ap.add_argument('--instance', choices=INSTANCES)
    ap.add_argument('--arms')
    ap.add_argument('--tag')
    a = ap.parse_args()
    if a.freeze and a.out_dir:
        return freeze(a)
    if a.run and a.spec and a.phase and a.instance:
        return run(a)
    if a.look40 and a.spec and a.instance:
        return look40(a)
    if a.vmstat and a.tag:
        return vmstat(a)
    ap.error('--freeze --out-dir D | --run --spec S --phase P --instance I [--arms a,b] | --look40 --spec S --instance I'
             ' | --vmstat --tag T')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
