"""P5.15 S56 -- step 3 completion (v2) and the step 4 pre-run diagnostics D-1, D-2, D-3 under ONE frozen spec.

Authority: S56_PLAN_v3.md section 7 rows 3-4; TASKS.md S56 section (Planner rulings recorded at fc5dc766);
S56_STEP4_DESIGN.md (decision 8: D-1 local-solver floor, D-2 hard-pin merged, D-3 ADMM-only boxes); the Planner's
task of 2026-11-23 ("step 3 completion and the step 4 pre-run diagnostics D-1, D-2 and D-3 under ONE frozen
diagnostics spec"). Spec: frozen_step4_diag_spec_v2_<sha8>.json (version 2; predecessor = the step 3 spec v1
4017dc70), committed with this launcher BEFORE launch.

THE ORDER (every solve declared; counts frozen in the spec and verified EXACTLY by armed guards):
  S3-R  reproduction: one solve of each run_v1 merged model (2025|Winter, 2035|Summer) with run_v1's exact build
        (the step 3 harness's own functions, by import), default start, recovery off. Gate: objective bitwise equal to
        run_v1 and every per-scenario interface P, Q, V equal to run_v1's. A failed gate is recorded and the reproduced
        solution is used anyway (the difference stated).                                                 2 solves
  S3-1  the 2025|Winter single-scenario model of scenario (0, 2), warm-started (primal only) from the reproduced
        merged solution's scenario-02 slice; recovery off. Then 2025|Winter C_central = sum_s omega_s C_s (run_v1's 8
        singles + this one) and step 3's P3 / P4 for that block.                                         1 solve
  S3-2  2025|Summer comparator, run_v1's 11-solve design by import of the step 3 harness's `phase_b` (merged default,
        9 singles, merged warm from the singles); its phase-A gates first (zero solves).                 11 solves
  D-3   (child process per instance, A = 2025|Winter alone, B = {2025, 2035} x Summer) the ADMM initialisation
        round, option on, 'per_scenario', alpha 0, p58 rescale as the harness applies it, stopped by a sentinel at
        production's `_build_pristine_snapshot_bases` (called once, right before ADMM cycle 1; nothing of cycle 1
        runs). Solve count from code: TSO one run_smopf per (year, day) block (`NetworkData.optimize`), each DSO one per
        block, the ESSO one `_optimize` per active node (3) -> A 1 + 3 + 3 = 7, B 2 + 6 + 3 = 11. Sigma's calibration
        assert (`_resolve_common_admm_objective_scale`) runs inside it. Then, ZERO solves: the inventory of every
        ADMM-only bound (code reading listed in the spec + a programmatic diff of every ADMM-prepared block against a
        plain build of the same network with the same candidate) and the check of each block's C_central interface
        values (the single-scenario solutions) against the interface boxes.                            7 + 11 solves
  D-1   (same child, after D-3) the target block's 4 ADMM-prepared local models (TSO + 3 DSOs) with every per-scenario
        interface P, Q, V copy FIXED at the block's central solution (TSO: interface_delta_p/q = P_s / s_base - pc,
        vmag at the ADN bus; DSO: reference-generator pg / qg (+ the scenario-free shared-ESS term, 0 at x0) and vmag at
        the reference node), solved at production depth then at tail depth (`compl_inf_tol` of the TSO/DSO holders:
        production = the captured baseline, tail = admm.convergence_depth_tail['compl_inf_tol'], applied with
        production's own `_apply_convergence_depth_tail`), from_warm_start=True (the ADMM-cycle solve form), recovery
        off. offset = sum of the 4 production `_get_operational_recourse_block_components` values (w_b x (local
        primal - settlement total - pin): settlement excluded in full) - w_b x C_central. A: 2025|Winter, B:
        2035|Summer.                                                                                     8 + 8 solves
  D-2   the merged model of 2025|Winter and 2035|Summer with ONE added equality set making the interface voltage
        scenario-common at the interface / DN reference bus (merged node TN_<dn>): vmag_sqr[n, s, t] =
        vmag_sqr[n, (0,0), t] for s != (0,0), warm-started (primal only) from the reproduced merged solution (S3-R's
        saved primal file); recovery off. Delta_pin = C_hardpin - C_central.                             2 solves
  Parent 16, child A 15, child B 19: 50 solves in all.

EVERY SOLVE SAVES ITS PRIMAL VALUES (Planner ruling 1): gzip JSON (mtime 0) {var name: value} in component order,
hashed in the manifest. No pickle anywhere (pickle.load / loads blocked).

GUARDS. SolveProfileGuard(permitted = this file's `_solve_declared` (parent), `_init_round` and `_d1_solve` (child))
is installed before any project import; the parent also imports the step 3 harness, whose own guard (permitted only
its `_solve_one`) is then installed on top and counts the same 16 solves (every parent solve passes through it, via
`_solve_declared`, which replaces the step 3 module's `_solve_one` name and calls the original). Each process verifies
its exact declared count per site. Single-instance lock; the repo-root campaign lock and any other running p5*.py
process refuse the launch; the output directory must not exist. Children run attached and one at a time, both
streams captured to files.

MODES
  --freeze --out-dir D      writes the frozen spec into D (D created; zero solves).
  --trial --spec S --out-dir D   zero solves: phase A, every capture path on unsolved models (primal save/load, the
                            single-02 slice map, the hard-pin construction and warm map), and each child with STUBBED
                            optimize calls (a stand-in point; sigma's computation replaced by the fixed sigma, trial
                            only) through D-3's inventory and D-1's fixing and cost read. Guard verified at 0.
  --run --spec S --out-dir D     the declared run.
  (internal) --child --child-in F --child-out G
Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_step4_diag.py --run --spec S --out-dir D
       > <stdout log> 2> <stderr log>
Objective convention (every value): gross -- the step 3 merged/single model.objective (settlement weight 0, no row
18, no pin; generation + flexibility + load curtailment + slack terms; no salvage), EUR per representative day,
probability-weighted; D-1 local values: production `_get_operational_recourse_block_components` (settlement excluded
in full, pin excluded, salvage separate), EUR of total discounted cost (w_b applied).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PERMITTED = ((SCRIPT, '_solve_declared'), (SCRIPT, '_init_round'), (SCRIPT, '_d1_solve'))
GUARD = SolveProfileGuard(permitted=PERMITTED, label='P5.15 S56 step 3 v2 + step 4 diagnostics D-1..D-3').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (this stage reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import ast  # noqa: E402
import gzip  # noqa: E402
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
from types import SimpleNamespace  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

CANONICAL_PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
SOLVER_PATH = '/usr/local/bin/ipopt'
CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
SPEC_PREFIX = 'frozen_step4_diag_spec_v2_'
STEP3_SPEC = os.path.join(RES, 'step3_central_v1', 'frozen_step3_central_spec_v1_4017dc70.json')
STEP3_RESULTS = os.path.join(RES, 'step3_central_v1', 'run_v1', 'central_ms_x0_results.json')
STEP3_HARNESS = 'p515_s56_central_ms_x0.py'
INSTANCE_A_RECORD = os.path.join(RES, 'step4_instances_v1', 'step4_instances.json')
INSTANCE_B_RECORD = os.path.join(RES, 'step4_instances_v2', 'step4_instance_b.json')
COUPLING_GATES = 'p515_s56_coupling_gates.py'   # read by AST only (importing it would arm a permitted=() guard)
GiB = 1024.0 ** 3


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 diag] {msg}', flush=True)


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


def rss():
    k = 1 if sys.platform == 'darwin' else 1024
    return {'self_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * k,
            'children_peak_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * k}


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


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


def _float_list_equal(a, b):
    return len(a) == len(b) and all(float(x) == float(y) for x, y in zip(a, b))


# ======================================================================================================================
#  primal values (Planner ruling 1: every solve saves its primal values, non-pickle, hashed)
# ======================================================================================================================
def save_primal(model, path, meta):
    import pyomo.environ as pe
    vals, n_none = {}, 0
    for v in model.component_data_objects(pe.Var, active=True, descend_into=True):
        x = v.value
        if x is None:
            n_none += 1
        vals[v.name] = None if x is None else float(x)
    payload = {'schema': 's56_step4_diag_primal_v1', 'meta': meta, 'n_vars': len(vals), 'n_none': n_none,
               'format': 'var name -> value (Python float repr, exact round trip); component_data_objects order',
               'vars': vals}
    raw = json.dumps(payload, separators=(',', ':')).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'xb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0, filename='') as gz:
            gz.write(raw)
    return {'path': os.path.relpath(path, REPO), 'sha256': _sha(path), 'bytes': os.path.getsize(path),
            'n_vars': len(vals), 'n_none': n_none}


def load_primal(path):
    with gzip.open(path, 'rb') as handle:
        return json.loads(handle.read().decode())


def warm_from_primal(model, payload):
    """free Var <- the saved value of the same name (fixed Vars untouched)."""
    import pyomo.environ as pe
    vals = payload['vars']
    r = {'set': 0, 'skipped_fixed': 0, 'missing': 0, 'none_value': 0}
    for v in model.component_data_objects(pe.Var, active=True, descend_into=True):
        if v.fixed:
            r['skipped_fixed'] += 1
            continue
        name = v.name
        if name not in vals:
            r['missing'] += 1
            continue
        x = vals[name]
        if x is None:
            r['none_value'] += 1
            continue
        v.set_value(x, skip_validation=True)
        r['set'] += 1
    return r


# ======================================================================================================================
#  constants read from the step 4 build tooling (AST literals of p515_s56_coupling_gates.py, never imported here)
# ======================================================================================================================
CG_NAMES = ('CASE_FILE_AA', 'CONVERGENCE_DEPTH_TAIL', 'ESS_AGEING_BASELINE', 'ESS_AGEING_LABEL', 'INVESTMENT_YEAR',
            'CAP_BUILD', 'REQUIRED_CONSECUTIVE_CYCLES', 'COUPLING_V1_NOQ', 'FIXED_SIGMA')


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


# ======================================================================================================================
#  the frozen spec
# ======================================================================================================================
PREDICTIONS = {
    'S3R_gate': ('Reproduction (ruling 1): each reproduced merged objective is bitwise equal to run_v1 '
                 '(175,188.7137093924 and 153,017.2487446982), and every per-scenario interface P, Q and V equals the '
                 'run_v1 value. If either gate fails, it is recorded and the reproduced solution is used anyway, with '
                 'the difference stated.'),
    'P2v2': 'S3-1: the 2025|Winter single-02 solve is Optimal.',
    'P3v2': 'S3-1: objective = merged scenario-02 term (171,052.349 EUR) minus an offset in [0.030, 0.040] EUR.',
    'S3_2': ('S3-2 (2025|Summer comparator, 11 solves): all Optimal; P3 and P4 as in run_v1 (relative 1e-6 for the '
             'warm-started merged, 1e-5 for the default-start merged, against sum_s omega_s C_s).'),
    'D3_sigma': 'D-3: the sigma assert passes on both instances (for B this settles G3).',
    'D3_boxes': ('D-3: every value is inside its box (each block\'s C_central interface values against every ADMM-only '
                 'bound). The minimum margin per block is reported.'),
    'D1a': 'D-1: production-depth |offset| >= allowance (H2).',
    'D1b': 'D-1: tail-depth |offset| <= allowance / 3.',
    'D2': 'D-2: Delta_pin >= -0.04 EUR/day (reported weighted against the allowance).',
}
PREDICTIONS_OPERATIONAL = {
    'S3R_gate': ('per block: float(reproduced merged_default objective) == run_v1 merged_default objective (==, not '
                 'a tolerance); for every scenario key, DN and kind in (P_mw, Q_mvar, V_pu) the 24-value list equals '
                 'run_v1 merged_default.interface element-wise (==). Recorded: float.hex of both, max |diff| per kind'),
    'P2v2': ("solution loaded (solver_result_succeeded), termination 'optimal', IPOPT exit 'Optimal Solution Found.'"),
    'P3v2': ('offset = 171052.34899059896 (run_v1 merged_default per_scenario[02].objective_per_scenario) - single 02 '
             'objective; 0.030 <= offset <= 0.040'),
    'S3_2': ('11/11 optimal (P2 operational definition of spec v1); |warm - sum| / |sum| <= 1e-6; |default - sum| / '
             '|sum| <= 1e-5'),
    'D3_sigma': ('the initialisation round reaches the sentinel (no exception) and the recorded ratio sigma_computed / '
                 'sigma_fixed lies in [1/f, f] (f = objective_scale_assert_factor as read), on A and on B'),
    'D3_boxes': ('per block, per (side, kind, DN, scenario, hour): margin = min(value - lower, upper - value); a value '
                 'is outside when -margin > 1e-8 x max(1, |bound|) (d3.bound_tol_rel, IPOPT default '
                 'bound_relax_factor); held iff no value outside on any of the 3 blocks; the strict count (any '
                 'negative margin) is reported beside it'),
    'D1a': '|offset_production| >= allowance_b, per block (2025|Winter 228.64, 2035|Summer 187.56 EUR)',
    'D1b': '|offset_tail| <= allowance_b / 3, per block',
    'D2': 'Delta_pin = C_hardpin - C_central >= -0.04 EUR per representative day, per block',
}
FAILURE_POLICY = ('Failure of any prediction is recorded, never re-run or tuned (D-1 failures are explicitly NOT a stop; '
                  'an infeasible local problem is reported, it points to H4). A reproduction-gate failure is recorded '
                  'and the reproduced solution is used anyway. A guard-count mismatch or a harness exception is a '
                  'harness fault: recorded, and the run stops for the Planner.')


def build_spec():
    import p515_s44_campaign_harness as H
    with open(STEP3_SPEC, encoding='utf-8') as handle:
        s3 = json.load(handle)
    with open(INSTANCE_A_RECORD) as handle:
        a = json.load(handle)
    with open(INSTANCE_B_RECORD) as handle:
        b = json.load(handle)
    with open(STEP3_RESULTS) as handle:
        r1 = json.load(handle)
    cg = coupling_gates_literals()
    w12 = a['sum12']['per_block']
    sum12 = a['sum12']['sum_12']
    blocks = {
        '2025|Winter': {'instance': 'A', 'w_b': w12['2025|Winter'], 'allowance_eur': 2500.0 * w12['2025|Winter'] / sum12,
                        'roles': ['S3-R', 'S3-1', 'D-3', 'D-1', 'D-2']},
        '2035|Summer': {'instance': 'B', 'w_b': w12['2035|Summer'], 'allowance_eur': 2500.0 * w12['2035|Summer'] / sum12,
                        'roles': ['S3-R', 'D-3', 'D-1', 'D-2']},
        '2025|Summer': {'instance': 'B', 'w_b': w12['2025|Summer'], 'allowance_eur': 2500.0 * w12['2025|Summer'] / sum12,
                        'roles': ['S3-2', 'D-3']},
    }
    spec = {
        'stage': ('P5.15 S56 step 3 completion (v2: reproduction, single 02, 2025|Summer comparator) and step 4 '
                  'pre-run diagnostics D-1, D-2, D-3 (frozen spec v2)'),
        'version': 2,
        'predecessor': {'path': STEP3_SPEC, 'sha256': _sha(STEP3_SPEC),
                        'relation': 'step 3 spec v1 (run_v1); this spec adds the reproduction, the single-02 solve, the '
                                    '2025|Summer comparator and the step 4 diagnostics'},
        'authority': [
            'S56_PLAN_v3.md section 7 rows 3-4', 'TASKS.md S56 section (Planner rulings recorded at fc5dc766)',
            'S56_STEP4_DESIGN.md decision 8 (D-1, D-2, D-3)',
            'Planner task 2026-11-23: step 3 completion and the step 4 pre-run diagnostics under ONE frozen spec; '
            'rulings 1 (reproduction solves, every solve saves its primal values), 2 (instance B), 3 (2025|Summer in '
            'the comparator)'],
        'launcher': SCRIPT,
        'step3_harness': {'path': STEP3_HARNESS, 'sha256': _sha(STEP3_HARNESS),
                          'pinned_by_v1_spec': s3['harness_sha256'],
                          'reused_functions': ['build', 'configure', 'slice_network', 'capture_solution', 'costs',
                                               'interface', 'max_violation', 'warm_map', 'component_positions',
                                               'declared_positions', 'coupling_check', 'slice_identity_gate',
                                               'random_point', 'relaxation_limits', 'block_weight', 'optimal',
                                               'phase_b', '_solve_one (wrapped)']},
        'instance_12_block': dict(s3['instance']),
        'step3_v1_settings_reused': {k: s3[k] for k in ('objective_alignment', 'solver', 'pre_solve_gates',
                                                        'reported', 'objective_convention')},
        'run_v1_results': {'path': STEP3_RESULTS, 'sha256': _sha(STEP3_RESULTS),
                           'merged_default_objective': {bk: r1['solves'][bk]['merged_default']['costs']['objective']
                                                        for bk in ('2025|Winter', '2035|Summer')},
                           'merged_default_objective_hex': {
                               bk: float.hex(r1['solves'][bk]['merged_default']['costs']['objective'])
                               for bk in ('2025|Winter', '2035|Summer')},
                           'merged_02_term_2025_Winter': r1['solves']['2025|Winter']['merged_default']['costs'][
                               'per_scenario']['02']['objective_per_scenario']},
        'instances': {
            'A': {'record': {'path': INSTANCE_A_RECORD, 'sha256': _sha(INSTANCE_A_RECORD)},
                  'derived_instance': a['blocks']['2025|Winter']['derived_instance_declaration'],
                  'blocks': ['2025|Winter'], 'd1_block': '2025|Winter'},
            'B': {'record': {'path': INSTANCE_B_RECORD, 'sha256': _sha(INSTANCE_B_RECORD)},
                  'derived_instance': b['derived_instance_declaration'],
                  'blocks': ['2025|Summer', '2035|Summer'], 'd1_block': '2035|Summer',
                  'gates_G1_G5': b['gates_summary']},
            'retired': {'path': os.path.join(RES, 'step4_instances_v1', 'SRP1__s56_step4_2035_Summer.json'),
                        'note': '2035-only instance: failed G1 (growth from years[0]); kept, not used'},
        },
        'blocks': blocks,
        'sum_12_w': sum12, 'R_eur': 10000.0,
        'allowance_definition': '(R/4) w_b / sum_12 w (12-block weights), EUR of total discounted planning cost',
        'admm_configuration': {
            'source': f'{COUPLING_GATES} literals (AST) + p515_s44_campaign_harness._config_hook_factory (the G-f / '
                      'step 4 build configuration)',
            'coupling': cg['COUPLING_V1_NOQ'], 'premium': None, 'alpha': 0.0,
            'case_file_anderson_acceleration': cg['CASE_FILE_AA'],
            'convergence_depth_tail': cg['CONVERGENCE_DEPTH_TAIL'],
            'ess_ageing_baseline': cg['ESS_AGEING_BASELINE'], 'ess_ageing_baseline_label': cg['ESS_AGEING_LABEL'],
            'ess_params_file': {'path': H.ESS_PARAMS_FILE_REL, 'sha256': _sha(H.ESS_PARAMS_FILE_REL)},
            'investment_year': cg['INVESTMENT_YEAR'], 'candidate': 'x0 {5: (0, 0), 7: (0, 0), 9: (0, 0)}',
            'cap': cg['CAP_BUILD'], 'required_consecutive_cycles': cg['REQUIRED_CONSECUTIVE_CYCLES'],
            'sigma_fixed_expected': cg['FIXED_SIGMA'], 'arm_label': 's39_D', 'apply_rho': False,
            'p58_rescale': 'p58_rescale.patched_admm_objectives around the run (as G.run_admm_arm applies it)',
            'thread_caps_in_child_env': dict(H.THREAD_CAP_ENV),
        },
        'solves': {
            'parent': {'S3-R': 2, 'S3-1': 1, 'S3-2': 11, 'D-2': 2, 'total': 16, 'site': '_solve_declared',
                       'also_counted_by': 'the step 3 harness guard (its _solve_one), 16'},
            'child_A': {'D-3 init': 7, 'D-1': 8, 'total': 15, 'sites': {'_init_round': 7, '_d1_solve': 8}},
            'child_B': {'D-3 init': 11, 'D-1': 8, 'total': 19, 'sites': {'_init_round': 11, '_d1_solve': 8}},
            'total': 50,
            'derivation_from_code': (
                'init round: create_distribution_networks_models_sequential -> NetworkData.optimize: one '
                'Network.run_smopf per (year, day) per DSO (3 DSOs); create_transmission_network_model -> one '
                'run_smopf per block; create_shared_energy_storage_model -> SharedEnergyStorageData.optimize: one '
                '_optimize per active node (5, 7, 9); shared_ess_initialization standalone (no price-taker solve); '
                'A: 1 block -> 3 + 1 + 3 = 7; B: 2 blocks -> 6 + 2 + 3 = 11. A production recovery attempt (enabled '
                'in the init round, as in production) would raise the count: the guard then fails loudly. D-1: 4 '
                'models x 2 depths, recovery off. Parent: one run_smopf each through the step 3 harness _solve_one.'),
        },
        'starts': {
            'S3-R': 'freshly built merged model (run_v1 build), production initialisers, from_warm_start=False',
            'S3-1': ('single-scenario model sliced (step 3 slice_network) to (0, 2); every free Var set from the '
                     'reproduced merged model at the same index with scenario (0, 2) (scenario positions read from the '
                     'declarations); no duals; from_warm_start=False'),
            'S3-2': 'run_v1 design via step 3 phase_b: merged default; singles 00..22 default; merged warm from singles',
            'D-1': ('the ADMM-prepared models as the initialisation round left them (init solution + its multipliers), '
                    'interface copies fixed; production depth with from_warm_start=True; tail depth warm-started '
                    '(from_warm_start=True) from the production-depth solve'),
            'D-2': 'freshly built merged model + hard-pin rows; every free Var from the S3-R saved primal file (by '
                   'name); no duals; from_warm_start=False',
        },
        'recovery': {'S3-R, S3-1, S3-2, D-2': 'off (step 3 configure)', 'D-1': 'off (TSO and DSO holders, set after '
                     'the initialisation round)', 'D-3 init round': 'production setting (unchanged)'},
        'd1': {
            'fix': {'TSO': 'interface_delta_p/q[dn, s, t] fixed at P_s / s_base - pc[adn load, s, t] (pc fixed by '
                           'production at the DSO initial expectation); vmag[ADN bus, s, t] fixed at V_s',
                    'DSO': 'pg/qg[reference generator, s, t] fixed at P_s / s_base + shared-ESS term (0 at x0, '
                           'checked); vmag[reference node, s, t] fixed at V_s'},
            'central_solution': 'the block\'s single-scenario solutions (C_central = sum_s omega_s C_s): interface '
                                'P_mw / Q_mvar / V_pu per scenario and DN (step 3 interface capture)',
            'depths': {'production': 'compl_inf_tol of each holder as captured by production '
                                     '_capture_convergence_depth_tail_baseline (TSO case file 5e-4; DSO not set -> '
                                     'IPOPT default)',
                       'tail': 'admm.convergence_depth_tail[compl_inf_tol] (declared 1e-6), via '
                               '_apply_convergence_depth_tail(active=True), restored after'},
            'offset': ('sum over the TSO and 3 DSO blocks of production _get_operational_recourse_block_components '
                       '(w_b x (get_primal_value - settlement total - pin)) - w_b x C_central; EUR of total discounted '
                       'cost'),
            'order': 'DSO5, DSO7, DSO9, TSO at production depth; then the same at tail depth',
            'active_bound_tol': 1e-6,
            'active_bound_definition': ('free Var with finite bound and |value - bound| <= tol (model units); '
                                        'active inequality row with |body - bound| <= tol'),
        },
        'd2': {'pin_rows': 'vmag_sqr[TN_<dn>, s_m, s_o, t] == vmag_sqr[TN_<dn>, 0, 0, t] for every (s_m, s_o) != '
                           '(0, 0), DN 5, 7, 9, every hour (8 x 24 x 3 = 576 rows)',
               'component': 's56_d2_hard_pin (added in this diagnostic only; no production change)',
               'objective': 'model.objective (gross, as step 3)'},
        'd3': {
            'code_read': [
                'shared_resources_planning.create_transmission_network_model (ADN-bus voltage slacks fixed 0; ADN load '
                'pc/qc bounds removed and fixed at the DSO initial expectation; ADN flexibility legs fixed 0; alias '
                'interface_delta freed within +-rating)',
                'model_construction_helpers.interface_pf_p/q_transmission_def (pc/qc_curt fixed at EQUALITY_TOLERANCE '
                'under l_curt)',
                'shared_resources_planning._wire_per_scenario_interface_coupling (every interface_delta copy freed '
                'within +-rating at the alias value; deviation Params and terms, no bound)',
                'shared_resources_planning.create_distribution_networks_models_sequential (expected interface Vars, '
                'expected_interface_vmag NonNegativeReals)',
                'shared_resources_planning.update_transmission_model_to_admm / update_distribution_models_to_admm '
                '(Params and objective terms)',
                'centralized_coordination.combine_networks (the merge drops the DN reference generators -- their '
                'P/Q limits exist only in the DSO models -- and the DN reference node -- its V limits and slack exist '
                'only in the DSO models)',
                'model_construction_helpers.vmag_bounds / vmag_sqr_bounds / voltage_magnitude_*_cons_rule'],
            'programmatic': ('per ADMM-prepared block (after the initialisation round): every Var data object compared '
                             'by name with a plain build of the same network (NetworkData.build_model + '
                             'update_model_with_candidate_solution with the run candidate): (lb, ub, fixed, fixed '
                             'value); Var and Constraint components present only in the ADMM block; constraint rows '
                             'whose active state differs'),
            'bound_tol_rel': 1e-8,
            'bound_tol_note': ('a value is OUTSIDE its box when it exceeds the bound by more than 1e-8 x max(1, '
                               '|bound|): IPOPT\'s default bound_relax_factor (not overridden in the step 3 options), '
                               'under which the central solutions were computed; the strict count is reported '
                               'beside it. Chosen after a zero-solve draft trial (scratch, before freezing) showed strict margins of '
                               '-1.9e-10 pu on run_v1\'s 2035|Summer singles (V at the 1.1 pu limit)'),
            'interface_boxes': ('per block, DN, scenario, hour: TSO P and Q box [pc + lb(delta), pc + ub(delta)] x '
                                's_base; TSO V box = vmag bounds n sqrt(vmag_sqr bounds) n ([v_min, v_max] when the '
                                'voltage slacks at the ADN bus are fixed at 0); DSO P / Q = reference-generator pg / qg '
                                'bounds x s_base; DSO V = vmag / vmag_sqr bounds at the reference node n ([v_min, '
                                'v_max] when its voltage slacks are fixed or absent)'),
        },
        'predictions': PREDICTIONS,
        'predictions_operational': PREDICTIONS_OPERATIONAL,
        'thresholds': {'P3v2_offset_eur': [0.030, 0.040], 'P3_rel': 1e-6, 'P4_rel': 1e-5, 'D1b_fraction': 1.0 / 3.0,
                       'D2_min_eur_per_day': -0.04},
        'failure_policy': FAILURE_POLICY,
        'not_permitted': ['production edits', 'formulation changes', 'any solve not declared',
                          'overwriting committed artifacts', 'concurrent or detached runs', 'any ADMM cycle',
                          'step 4 smoke'],
        'capture': {'every_solve': ['status', 'termination', 'IPOPT exit', 'iterations', 'mu_final', 'floor status',
                                    'compl_inf_tol in force', 'wall', 'RSS', 'objective', 'primal values file + sha256'],
                    'run': ['guard counts per process and site', 'pickle 0/0', 'RSS per process', 'wall',
                            'stdout and stderr of every process']},
        'objective_convention': ('C values gross (step 3: model.objective, settlement weight 0, no row 18, no pin, no '
                                 'salvage, EUR/representative day); D-1 values in EUR of total discounted cost, '
                                 'settlement excluded in full, pin excluded, salvage separate; Delta_pin and D-1 '
                                 'offsets also stated weighted (w_b) against the allowance'),
        'output': os.path.join(RES, 'step4_diag_v1', 'run_v1'),
    }
    return spec


def freeze(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith(SPEC_PREFIX)]
    if existing:
        _log(f'REFUSED: a spec already exists: {existing}')
        return 2
    spec = build_spec()
    text = json.dumps(spec, indent=1, sort_keys=True, default=str)
    GRIO.check(spec)
    sha = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(out_dir, f'{SPEC_PREFIX}{sha[:8]}.json')
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)
    _log(f'wrote {os.path.relpath(path, REPO)} (sha256 {sha}); guard {GUARD.counts}')
    return 0 if not GUARD.verify(0) else 1


# ======================================================================================================================
#  PARENT
# ======================================================================================================================
S3 = None
ORIG_S3_SOLVE_ONE = None
CTX = {'trial': False, 'out_dir': None, 'solves': [], 'logs_root': None}


def _solve_declared(nd, m, params):
    """THE parent solve site: installed as the step 3 harness's `_solve_one` name, so every parent solve (including
    those of the step 3 harness's own `phase_b`) passes through here; calls the original (whose guard permits it),
    then saves the primal values. In --trial it never solves."""
    tag = os.path.relpath(nd.logs_dir, CTX['logs_root']) if CTX['logs_root'] else 'unknown'
    if CTX['trial']:
        st = {'trial_not_solved': True, 'solution_loaded': False, 'termination_condition': None,
              'ipopt_records': [], 'wall_s': 0.0}
    else:
        st = ORIG_S3_SOLVE_ONE(nd, m, params)
    st['tag'] = tag
    t0 = time.perf_counter()
    try:
        st['primal'] = save_primal(m, os.path.join(CTX['out_dir'], 'primal', tag.replace(os.sep, '__') + '.json.gz'),
                                   {'tag': tag, 'solution_loaded': st.get('solution_loaded'),
                                    'termination_condition': st.get('termination_condition'), 'trial': CTX['trial']})
    except Exception:  # noqa: BLE001 -- recorded; never aborts the declared solve set
        st['primal_error'] = traceback.format_exc()
    st['primal_save_s'] = time.perf_counter() - t0
    CTX['solves'].append({'tag': tag, 'loaded': st.get('solution_loaded'), 'tc': st.get('termination_condition'),
                          'exit': (st.get('ipopt_records') or [{}])[-1].get('exit')})
    _log(f"solve {tag}: {st.get('termination_condition')} {(st.get('ipopt_records') or [{}])[-1].get('exit')} "
         f"it {(st.get('ipopt_records') or [{}])[-1].get('iterations')} wall {st.get('wall_s', 0):.1f}s")
    return st


def _ipopt_last(st):
    r = (st.get('ipopt_records') or [{}])[-1]
    return {k: r.get(k) for k in ('exit', 'iterations', 'mu_final', 'mu_floor', 'floor_status', 'obj_scaling_factor',
                                  'compl_inf_tol_in_force', 'tol_in_force')}


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else math.inf)


def parent(args):
    global S3, ORIG_S3_SOLVE_ONE
    trial = bool(args.trial)
    pre = []
    spec_rel = os.path.relpath(os.path.abspath(args.spec), REPO)
    spec_sha = _sha(args.spec)
    tag = os.path.basename(args.spec)[len(SPEC_PREFIX):].split('.')[0]
    if not spec_sha.startswith(tag):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
    if not trial:
        for rel in (SCRIPT, spec_rel, STEP3_HARNESS):
            if not committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
    if os.path.exists(args.out_dir):
        pre.append(f'{args.out_dir} exists')
    if os.path.exists(CAMPAIGN_LOCK):
        pre.append('campaign lock exists')
    others = other_project_processes()
    if others:
        pre.append(f'other project processes running: {others}')
    lock = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), '.step4_diag.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[S56 diag] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        return 1
    try:
        return parent_run(args, spec_rel, spec_sha, trial)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def parent_run(args, spec_rel, spec_sha, trial):
    global S3, ORIG_S3_SOLVE_ONE
    t_start = time.perf_counter()
    with open(args.spec, encoding='utf-8') as handle:
        spec = json.load(handle)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    CTX.update(trial=trial, out_dir=out_dir, logs_root=os.path.join(out_dir, 'logs'))
    res = {'stage': spec['stage'], 'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT,
           'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(), 'utc_start': _utc(),
           'trial': trial, 'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'objective_convention': spec['objective_convention'], 'checks': [], 'error': None}
    checks = res['checks']

    def ck(name, ok, detail=None):
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:500]}' if detail is not None and not ok else ''))
        return bool(ok)
    try:
        # ---- the step 3 harness, by import (its guard is installed on top of ours; its _solve_one is wrapped)
        import p515_s56_central_ms_x0 as S3mod
        S3 = S3mod
        ORIG_S3_SOLVE_ONE = S3.__dict__['_solve_one']
        S3._solve_one = _solve_declared
        import numpy as np
        import pyomo.environ as pe
        from centralized_coordination import combine_networks
        ck('step 3 harness sha256 == spec pin == v1 spec pin', _sha(STEP3_HARNESS) == spec['step3_harness']['sha256']
           == spec['step3_harness']['pinned_by_v1_spec'])
        for key, path, sha in (('step 3 v1 spec', spec['predecessor']['path'], spec['predecessor']['sha256']),
                               ('run_v1 results', spec['run_v1_results']['path'], spec['run_v1_results']['sha256']),
                               ('instance A record', spec['instances']['A']['record']['path'],
                                spec['instances']['A']['record']['sha256']),
                               ('instance B record', spec['instances']['B']['record']['path'],
                                spec['instances']['B']['record']['sha256'])):
            ck(f'{key} sha256 == spec', _sha(path) == sha)
        with open(spec['predecessor']['path'], encoding='utf-8') as handle:
            s3spec = json.load(handle)
        with open(spec['run_v1_results']['path']) as handle:
            r1 = json.load(handle)
        for name in ('instance_case_A', 'instance_case_B'):
            d = spec['instances'][name[-1]]['derived_instance']
            ck(f'{name} case sha256 == declaration', _sha(d['case_path']) == d['case_sha256'])
        # capture checklist (rule eleven), before anything is read
        for name in ('build', 'configure', 'slice_network', 'capture_solution', 'costs', 'interface', 'max_violation',
                     'warm_map', 'component_positions', 'declared_positions', 'coupling_check', 'slice_identity_gate',
                     'random_point', 'relaxation_limits', 'block_weight', 'optimal', 'phase_b', '_with_scen', '_strip',
                     '_tup'):
            assert callable(getattr(S3, name, None)), name
        for name in ('save_primal', 'load_primal', 'warm_from_primal', 'child_main', 'd3_inventory', 'd3_boxes',
                     'd1_fix', 'd1_round', 'active_bounds', 'add_hard_pin'):
            assert callable(globals().get(name)), name
        # ---- read the 12-block instance (production reader, as run_v1)
        inst = spec['instance_12_block']
        ck('12-block case sha256 == declaration', _sha(inst['case_path']) == inst['case_sha256'])
        planning = S3.S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                                        os.path.join(out_dir, 'instance_read'), S3._Stages())
        cks = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
        ck('12-block combined scenario checksum == declaration', cks == inst['scenario_checksum'], cks)
        tn, dns = planning.transmission_network, planning.distribution_networks
        dn_ids = list(dns)
        bad = []
        for y in tn.years:
            for d in tn.days:
                t = tn.network[y][d]
                for dn in dns:
                    n = dns[dn].network[y][d]
                    if not (list(n.prob_market_scenarios) == list(t.prob_market_scenarios)
                            and list(n.prob_operation_scenarios) == list(t.prob_operation_scenarios)
                            and np.array_equal(np.asarray(n.cost_energy_p), np.asarray(t.cost_energy_p))
                            and np.array_equal(np.asarray(n.cost_flex), np.asarray(t.cost_flex))):
                        bad.append(f'{y}|{d}|{dn}')
        ck('probabilities and prices identical across TN and every DN, every block', not bad, bad)
        merged = combine_networks(tn, dns)
        params = S3.configure(merged.params, s3spec)
        res['solver_options_in_force'] = dict(params.solver_params.options)
        res['recovery'] = [params.solver_params.recovery_enabled, params.solver_params.recovery_tier2_enabled]
        ck('solver options == run_v1 (in force)', dict(params.solver_params.options) == r1['solver_options_in_force'],
           [dict(params.solver_params.options), r1['solver_options_in_force']])
        ck('recovery off', res['recovery'] == [False, False])
        ck('solver_params.verbose is False', params.solver_params.verbose is False)
        # ---- phase A for the three blocks (zero solves)
        decl = S3.declared_positions()
        tol = s3spec['pre_solve_gates']['slice_identity_rel_tol']
        res['phase_A'] = {}

        def ykey(bk):
            y_s, d = bk.split('|')
            return next(yy for yy in tn.years if str(yy) == y_s), d
        for bk in ('2025|Winter', '2035|Summer', '2025|Summer'):
            y, d = ykey(bk)
            md = merged.network[y][d]
            mg, brec = S3.build(md, params, s3spec)
            vpos = S3.component_positions(mg, decl, pe.Var)
            cpos = S3.component_positions(mg, decl, pe.Constraint)
            ra = {'build': {k: v for k, v in brec.items() if k != 'formulation'},
                  'formulation_ok': brec['alpha0_nopin_settlement0_ok'],
                  'n_scenarios': [len(md.prob_market_scenarios), len(md.prob_operation_scenarios)],
                  'block_weight': S3.block_weight(tn, y, d), 'relaxation_limits': S3.relaxation_limits(tn, dns, y, d)}
            ck(f'{bk}: alpha 0, no pin, settlement 0, no ADN / shared ESS', ra['formulation_ok'])
            ck(f'{bk}: 3 x 3 scenarios', ra['n_scenarios'] == [3, 3])
            ck(f"{bk}: block weight == spec w_b (rel 1e-12)",
               _rel(ra['block_weight']['weight'], spec['blocks'][bk]['w_b']) <= 1e-12,
               [ra['block_weight']['weight'], spec['blocks'][bk]['w_b']])
            ra['coupling'] = S3.coupling_check(mg, vpos, cpos)
            ck(f'{bk}: no constraint couples scenarios', ra['coupling']['ok'],
               {k: v for k, v in ra['coupling'].items() if k != 'examples_multi'})
            S3.random_point(mg, s3spec['pre_solve_gates']['random_point_seed'],
                            s3spec['pre_solve_gates']['random_point_box'])
            ra['slice_identity'] = S3.slice_identity_gate(md, params, s3spec, mg, vpos, cpos, tol)
            ck(f'{bk}: slice identity at a random point (rel <= {tol})', ra['slice_identity']['ok'],
               {'objective_rel': ra['slice_identity']['objective_rel'],
                'bad': [k for k, v in ra['slice_identity']['scenarios'].items() if not v['ok']]})
            ra['slice_identity'] = {'ok': ra['slice_identity']['ok'],
                                    'objective_rel': ra['slice_identity']['objective_rel'],
                                    'per_scenario_ok': {k: v['ok'] for k, v in ra['slice_identity']['scenarios'].items()}}
            res['phase_A'][bk] = ra
            del mg
        # ---- code identity
        mods = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if f and os.path.abspath(f).startswith(REPO + os.sep) and f.endswith('.py'):
                rel = os.path.relpath(os.path.abspath(f), REPO)
                mods[rel] = {'sha256': _sha(rel), 'committed_clean': committed_clean(rel)}
        res['imported_repo_modules_parent'] = mods
        dirty = sorted(k for k, v in mods.items() if not v['committed_clean'])
        if not trial:
            ck('every imported repo module committed clean', not dirty, dirty)
        else:
            res['trial_dirty_imported_modules'] = dirty
        failed = [c['check'] for c in checks if not c['ok']]
        res['refused_before_any_solve'] = failed
        if failed:
            _log(f'REFUSED before any solve: {failed}')
        else:
            phases(res, spec, s3spec, r1, planning, tn, dns, dn_ids, merged, params, decl, ykey, out_dir, trial)
    except Exception:  # noqa: BLE001 -- recorded; results and guard verification still written
        res['error'] = traceback.format_exc()
        _log('EXCEPTION:\n' + res['error'])
    # ---- guards (parent)
    exp = 0 if (trial or res.get('refused_before_any_solve')) else spec['solves']['parent']['total']
    gv = GUARD.verify(exp)
    site_ok = (dict(GUARD.permitted_sites) == ({f'{SCRIPT}:_solve_declared': exp} if exp else {}))
    res['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                    'expected': exp, 'verify_failures': gv, 'sites_exact': site_ok}
    if S3 is not None:
        gv3 = S3.GUARD.verify(exp)
        res['guard_step3_harness'] = {'label': S3.GUARD.label, 'counts': dict(S3.GUARD.counts),
                                      'permitted_sites': dict(S3.GUARD.permitted_sites), 'expected': exp,
                                      'verify_failures': gv3}
        res['pickle_step3_module'] = dict(S3.PK)
    res['pickle'] = dict(PK)
    children = res.get('children') or {}
    res['guards_ok'] = bool(not gv and site_ok and not (res.get('guard_step3_harness') or {}).get('verify_failures')
                            and PK == {'load': 0, 'loads': 0}
                            and all((c or {}).get('guards_ok') for c in children.values())
                            and (trial or bool(res.get('refused_before_any_solve')) or len(children) == 2))
    res['peak_rss_parent'] = rss()
    res['total_wall_s'] = time.perf_counter() - t_start
    if not trial and not res.get('refused_before_any_solve') and res.get('error') is None:
        try:
            res['predictions'] = evaluate(res, spec)
        except Exception:  # noqa: BLE001
            res['predictions_error'] = traceback.format_exc()
    res['utc_end'] = _utc()
    path = os.path.join(out_dir, 'step4_diag_results.json')
    write_json(path, res)
    outputs = {}
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, out_dir)] = _sha(p)
    write_json(os.path.join(out_dir, 'manifest_sha256.json'),
               {'outputs': outputs, 'spec': {'path': spec_rel, 'sha256': spec_sha},
                'script': {'path': SCRIPT, 'sha256': res['script_sha256']},
                'imported_repo_modules_parent': {k: v['sha256'] for k, v in
                                                 (res.get('imported_repo_modules_parent') or {}).items()}})
    _log(f"wrote {path}; guards_ok {res['guards_ok']}; error {res['error'] is not None}; "
         f"predictions {({k: v.get('held') for k, v in (res.get('predictions') or {}).items()})}")
    if res.get('refused_before_any_solve'):
        return 3
    return 0 if (res['guards_ok'] and res['error'] is None) else 2


def _interface_gate(new_itf, old_itf):
    out = {'equal': True, 'max_abs_diff': {'P_mw': 0.0, 'Q_mvar': 0.0, 'V_pu': 0.0}, 'n_lists': 0, 'n_lists_equal': 0}
    for sc, per in old_itf.items():
        for dn, rec in per.items():
            for kind in ('P_mw', 'Q_mvar', 'V_pu'):
                a, b = new_itf[sc][dn][kind], rec[kind]
                out['n_lists'] += 1
                eq = _float_list_equal(a, b)
                out['n_lists_equal'] += int(eq)
                out['equal'] &= eq
                out['max_abs_diff'][kind] = max(out['max_abs_diff'][kind],
                                                max(abs(float(x) - float(y)) for x, y in zip(a, b)))
    return out


def central_from_singles(singles, probs):
    """C_central and the interface values of the central point (the single-scenario solutions)."""
    complete = all(s is not None and 'costs' in s and 'interface' in s for s in singles.values()) and len(singles) == 9
    if not complete:
        return {'complete': False}
    c = sum(probs[k] * singles[k]['costs']['objective'] for k in singles)
    itf = {k: singles[k]['interface']['00'] for k in singles}
    return {'complete': True, 'C_central': c, 'interface': itf,
            'parts': {'C_gen_plus_flex': sum(probs[k] * singles[k]['costs']['C_gen_plus_flex'] for k in singles),
                      'slack_terms': sum(probs[k] * singles[k]['costs']['total_slack_penalties'] for k in singles)}}


def phases(res, spec, s3spec, r1, planning, tn, dns, dn_ids, merged, params, decl, ykey, out_dir, trial):
    import pyomo.environ as pe
    logs = os.path.join(out_dir, 'logs')
    sol = res.setdefault('solves', {})
    probs = None
    # ================================================================================================ S3-R
    _log('S3-R: reproduction of the two run_v1 merged solves')
    kept = {}
    for bk in ('2025|Winter', '2035|Summer'):
        y, d = ykey(bk)
        md = merged.network[y][d]
        m, brec = S3.build(md, params, s3spec)
        md.logs_dir = os.path.join(logs, 'S3R', bk.replace('|', '_'))
        os.makedirs(md.logs_dir)
        st = S3._solve_one(md, m, params)
        r = {'build': {k: v for k, v in brec.items() if k != 'formulation'}, 'solve': st, 'optimal': S3.optimal(st),
             'ipopt': _ipopt_last(st)}
        if st.get('solution_loaded') or trial:
            r.update(S3.capture_solution(m, md, params, s3spec, dn_ids, res['phase_A'][bk]['relaxation_limits']))
        old = r1['solves'][bk]['merged_default']
        if 'costs' in r and 'interface' in r and not trial:
            obj = r['costs']['objective']
            gi = _interface_gate(r['interface'], old['interface'])
            r['gate'] = {'objective': obj, 'objective_hex': float.hex(obj), 'run_v1_objective': old['costs']['objective'],
                         'run_v1_objective_hex': float.hex(old['costs']['objective']),
                         'objective_bitwise_equal': obj == old['costs']['objective'],
                         'objective_diff_eur': obj - old['costs']['objective'], 'interface': gi,
                         'held': obj == old['costs']['objective'] and gi['equal']}
        sol[f'S3R|{bk}'] = r
        probs = {f'{a}{b}': md.prob_market_scenarios[a] * md.prob_operation_scenarios[b]
                 for a in range(len(md.prob_market_scenarios)) for b in range(len(md.prob_operation_scenarios))}
        if bk == '2025|Winter':
            kept['m'], kept['md'] = m, md
        else:
            del m
    # ================================================================================================ S3-1
    _log('S3-1: 2025|Winter single 02, warm from the reproduced merged slice')
    m_w, md_w = kept['m'], kept['md']
    nd, srec = S3.slice_network(md_w, 0, 2)
    ms, brec = S3.build(nd, params, s3spec)
    vpos = S3.component_positions(m_w, decl, pe.Var)
    wmap = {'set': 0, 'skipped_fixed': 0, 'unmapped': 0, 'none_value': 0, 'outside_bounds': 0}
    for comp in ms.component_objects(pe.Var, active=True):
        pos = vpos.get(comp.name)
        mc = getattr(m_w, comp.name)
        for idx, v in comp.items():
            if v.fixed:
                wmap['skipped_fixed'] += 1
                continue
            if pos in (None, 'unknown', (None, None)):
                wmap['unmapped'] += 1
                continue
            val = mc[S3._with_scen(S3._strip(idx, pos), pos, 0, 2)].value
            if val is None:
                wmap['none_value'] += 1
                continue
            if (v.lb is not None and val < v.lb) or (v.ub is not None and val > v.ub):
                wmap['outside_bounds'] += 1
            v.set_value(val, skip_validation=True)
            wmap['set'] += 1
    try:
        at = {'objective': float(pe.value(ms.objective)), 'max_violation': S3.max_violation(ms)}
    except Exception:  # noqa: BLE001 -- capture only
        at = {'capture_error': traceback.format_exc()}
    nd.logs_dir = os.path.join(logs, 'S3_1', 'single_02')
    os.makedirs(nd.logs_dir)
    st = S3._solve_one(nd, ms, params)
    r = {'slice': srec, 'warm_mapping': wmap, 'at_mapped_point': at, 'solve': st, 'optimal': S3.optimal(st),
         'ipopt': _ipopt_last(st), 'probability': probs['02']}
    if st.get('solution_loaded') or trial:
        r.update(S3.capture_solution(ms, nd, params, s3spec, dn_ids, res['phase_A']['2025|Winter']['relaxation_limits']))
    sol['S3_1|2025|Winter|single_02'] = r
    del ms, nd, m_w, kept
    # ================================================================================================ S3-2
    _log('S3-2: 2025|Summer comparator (step 3 phase_b, 11 solves)')
    r32 = {'phase_A': {'2025|Summer': res['phase_A']['2025|Summer']}, 'solves': {}}
    a_ns = SimpleNamespace(out_dir=os.path.join(out_dir, 'S3_2'), trial=False)
    os.makedirs(a_ns.out_dir)
    CTX['logs_root_s32'] = os.path.join(a_ns.out_dir, 'logs')
    saved_root = CTX['logs_root']
    CTX['logs_root'] = out_dir
    try:
        S3.phase_b(r32, a_ns, s3spec, [], ['2025|Summer'], tn, merged, params, decl, dn_ids)
    finally:
        CTX['logs_root'] = saved_root
    sol['S3_2|2025|Summer'] = r32['solves'].get('2025|Summer')
    for v in (sol['S3_2|2025|Summer'] or {}).values():
        for rr in ([v] if 'solve' in v else list(v.values())):
            rr['ipopt'] = _ipopt_last(rr['solve'])
    # ================================================================================================ central values
    central = {}
    s1 = r1['solves']['2025|Winter']['single']
    singles_w = {k: (sol['S3_1|2025|Winter|single_02'] if k == '02' else s1[k]) for k in s1}
    central['2025|Winter'] = central_from_singles(singles_w, probs)
    central['2035|Summer'] = central_from_singles(r1['solves']['2035|Summer']['single'], probs)
    central['2025|Summer'] = central_from_singles((sol['S3_2|2025|Summer'] or {}).get('single') or {}, probs)
    for bk, c in central.items():
        c['w_b'] = spec['blocks'][bk]['w_b']
        c['allowance_eur'] = spec['blocks'][bk]['allowance_eur']
        c['source'] = {'2025|Winter': 'run_v1 singles 00..22 except 02 + S3-1 single 02',
                       '2035|Summer': 'run_v1 singles', '2025|Summer': 'S3-2 singles'}[bk]
    res['central'] = {bk: {k: v for k, v in c.items() if k != 'interface'} for bk, c in central.items()}
    cin_central = os.path.join(out_dir, 'central_interface_for_children.json')
    write_json(cin_central, central)
    # ================================================================================================ D-3 / D-1 children
    res['children'] = {}
    for inst in ('A', 'B'):
        idef = spec['instances'][inst]
        cdir = os.path.join(out_dir, f'child_{inst}')
        os.makedirs(cdir)
        cin = {'instance': inst, 'derived': idef['derived_instance'], 'blocks': idef['blocks'],
               'd1_block': idef['d1_block'], 'central_path': cin_central, 'central_sha256': _sha(cin_central),
               'spec_path': os.path.abspath(res['spec']['path']), 'spec_sha256': res['spec']['sha256'],
               'child_dir': cdir, 'trial': trial,
               'expected_sites': {} if trial else spec['solves'][f'child_{inst}']['sites']}
        cin_path = os.path.join(cdir, 'child_input.json')
        write_json(cin_path, cin)
        cout = os.path.join(cdir, 'child_results.json')
        env = dict(os.environ)
        env.update(spec['admm_configuration']['thread_caps_in_child_env'])
        t0 = time.perf_counter()
        _log(f'child {inst}: D-3 init round + D-1 ({idef["d1_block"]})')
        with open(os.path.join(cdir, 'child_stdout.log'), 'x') as so, open(os.path.join(cdir, 'child_stderr.log'),
                                                                             'x') as se:
            rc = subprocess.run([sys.executable, '-u', SCRIPT, '--child', '--child-in', cin_path, '--child-out', cout],
                                cwd=REPO, stdout=so, stderr=se, env=env).returncode
        try:
            with open(cout) as handle:
                cres = json.load(handle)
        except Exception:  # noqa: BLE001
            cres = {'error': f'no child result: {traceback.format_exc()}'}
        cres['exit_code'] = rc
        cres['wall_s_parent_view'] = time.perf_counter() - t0
        res['children'][inst] = cres
        _log(f"child {inst}: exit {rc} error {str(cres.get('error'))[:300]} guards_ok {cres.get('guards_ok')}")
    # ================================================================================================ D-2
    _log('D-2: hard-pin merged solves')
    for bk in ('2025|Winter', '2035|Summer'):
        y, d = ykey(bk)
        md = merged.network[y][d]
        m, brec = S3.build(md, params, s3spec)
        pin = add_hard_pin(m, md, dn_ids)
        src = sol[f'S3R|{bk}']['solve'].get('primal') or {}
        rec = {'pin': pin, 'warm_source': src}
        if src.get('path') and _sha(os.path.join(REPO, src['path'])) == src.get('sha256'):
            rec['warm_mapping'] = warm_from_primal(m, load_primal(os.path.join(REPO, src['path'])))
        else:
            rec['warm_mapping'] = {'error': 'S3-R primal file missing or hash mismatch'}
        try:
            rec['at_warm_point'] = {'objective': float(pe.value(m.objective)),
                                    'max_violation': S3.max_violation(m),
                                    'pin_max_abs_residual': max(abs(pe.value(c.body) - pe.value(c.upper))
                                                                for c in m.s56_d2_hard_pin.values())}
        except Exception:  # noqa: BLE001
            rec['at_warm_point'] = {'capture_error': traceback.format_exc()}
        md.logs_dir = os.path.join(logs, 'D2', bk.replace('|', '_'))
        os.makedirs(md.logs_dir)
        st = S3._solve_one(md, m, params)
        rec.update({'solve': st, 'optimal': S3.optimal(st), 'ipopt': _ipopt_last(st)})
        if st.get('solution_loaded') or trial:
            rec.update(S3.capture_solution(m, md, params, s3spec, dn_ids, res['phase_A'][bk]['relaxation_limits']))
            try:
                rec['pin_max_abs_residual_after'] = max(abs(pe.value(c.body) - pe.value(c.upper))
                                                        for c in m.s56_d2_hard_pin.values())
            except Exception:  # noqa: BLE001
                rec['pin_max_abs_residual_after'] = None
        c = central.get(bk) or {}
        if 'costs' in rec and c.get('complete') and not trial:
            dp = rec['costs']['objective'] - c['C_central']
            rec['delta_pin'] = {'C_hardpin': rec['costs']['objective'], 'C_central': c['C_central'],
                                'delta_pin_eur_per_day': dp, 'w_b': c['w_b'], 'weighted_eur': c['w_b'] * dp,
                                'allowance_eur': c['allowance_eur'],
                                'weighted_over_allowance': c['w_b'] * dp / c['allowance_eur']}
        sol[f'D2|{bk}'] = rec
        del m


def add_hard_pin(m, md, dn_ids):
    """D-2: ONE added equality set -- the interface voltage scenario-common at the interface / DN reference bus
    (merged node TN_<dn>): vmag_sqr[n, s, t] == vmag_sqr[n, (0, 0), t] for every (s_m, s_o) != (0, 0)."""
    import pyomo.environ as pe
    nodes = [md.get_node_idx(f'TN_{dn}') for dn in dn_ids]
    idx = [(n, a, b, p) for n in nodes for a in m.scenarios_market for b in m.scenarios_operation for p in m.periods
           if (a, b) != (0, 0)]
    m.s56_d2_hard_pin_index = pe.Set(initialize=idx, dimen=4, ordered=True)
    m.s56_d2_hard_pin = pe.Constraint(m.s56_d2_hard_pin_index,
                                      rule=lambda mm, n, a, b, p: mm.vmag_sqr[n, a, b, p] == mm.vmag_sqr[n, 0, 0, p])
    return {'nodes': {str(dn): n for dn, n in zip(dn_ids, nodes)}, 'n_rows': len(idx),
            'node_ids': [f'TN_{dn}' for dn in dn_ids]}


def evaluate(res, spec):
    th = spec['thresholds']
    sol = res['solves']
    pr = {}
    # reproduction gate
    g = {bk: (sol.get(f'S3R|{bk}') or {}).get('gate') for bk in ('2025|Winter', '2035|Summer')}
    pr['S3R_gate'] = {'statement': spec['predictions']['S3R_gate'], 'per_block': g,
                      'held': all(v and v['held'] for v in g.values())}
    s1 = sol.get('S3_1|2025|Winter|single_02') or {}
    pr['P2v2'] = {'statement': spec['predictions']['P2v2'], 'optimal': s1.get('optimal'), 'ipopt': s1.get('ipopt'),
                  'held': bool(s1.get('optimal'))}
    term = spec['run_v1_results']['merged_02_term_2025_Winter']
    if 'costs' in s1:
        off = term - s1['costs']['objective']
        pr['P3v2'] = {'statement': spec['predictions']['P3v2'], 'merged_02_term_run_v1': term,
                      'merged_02_term_reproduced': ((sol.get('S3R|2025|Winter') or {}).get('costs') or {})
                      .get('per_scenario', {}).get('02', {}).get('objective_per_scenario'),
                      'single_02_objective': s1['costs']['objective'], 'offset_eur': off,
                      'range': th['P3v2_offset_eur'],
                      'held': th['P3v2_offset_eur'][0] <= off <= th['P3v2_offset_eur'][1]}
    else:
        pr['P3v2'] = {'statement': spec['predictions']['P3v2'], 'held': False, 'reason': 'single 02 not loaded'}
    # step 3 P3 / P4 for 2025|Winter (run_v1 merged starts) and 2025|Summer (S3-2)
    with open(spec['run_v1_results']['path']) as handle:
        r1 = json.load(handle)
    p34 = {}
    cw = res['central']['2025|Winter']
    if cw.get('complete'):
        mw = r1['solves']['2025|Winter']['merged_warm']['costs']['objective']
        mdft = r1['solves']['2025|Winter']['merged_default']['costs']['objective']
        rep = ((sol.get('S3R|2025|Winter') or {}).get('costs') or {}).get('objective')
        p34['2025|Winter'] = {'C_central': cw['C_central'], 'merged_warm_run_v1': mw, 'merged_default_run_v1': mdft,
                              'merged_default_reproduced': rep,
                              'P3_rel': _rel(mw, cw['C_central']), 'P3_diff_eur': mw - cw['C_central'],
                              'P3_held': _rel(mw, cw['C_central']) <= th['P3_rel'],
                              'P4_rel': _rel(mdft, cw['C_central']), 'P4_diff_eur': mdft - cw['C_central'],
                              'P4_held': _rel(mdft, cw['C_central']) <= th['P4_rel']}
    s32 = sol.get('S3_2|2025|Summer') or {}
    cs = res['central']['2025|Summer']
    runs = [('merged_default', s32.get('merged_default')), ('merged_warm', s32.get('merged_warm'))] + \
           [(f'single_{k}', v) for k, v in (s32.get('single') or {}).items()]
    not_opt = [n for n, r in runs if not (r and r.get('optimal'))]
    if cs.get('complete') and 'costs' in (s32.get('merged_warm') or {}) and 'costs' in (s32.get('merged_default') or {}):
        mw, mdft = s32['merged_warm']['costs']['objective'], s32['merged_default']['costs']['objective']
        p34['2025|Summer'] = {'C_central': cs['C_central'], 'merged_warm': mw, 'merged_default': mdft,
                              'P3_rel': _rel(mw, cs['C_central']), 'P3_diff_eur': mw - cs['C_central'],
                              'P3_held': _rel(mw, cs['C_central']) <= th['P3_rel'],
                              'P4_rel': _rel(mdft, cs['C_central']), 'P4_diff_eur': mdft - cs['C_central'],
                              'P4_held': _rel(mdft, cs['C_central']) <= th['P4_rel']}
    pr['step3_P3_P4'] = {'per_block': p34}
    pr['S3_2'] = {'statement': spec['predictions']['S3_2'], 'n_solves': len(runs), 'not_optimal': not_opt,
                  'P3_P4': p34.get('2025|Summer'),
                  'held': (len(runs) == 11 and not not_opt and bool(p34.get('2025|Summer'))
                           and p34['2025|Summer']['P3_held'] and p34['2025|Summer']['P4_held'])}
    # D-3
    ch = res.get('children') or {}
    sig = {i: (c.get('d3') or {}).get('sigma') for i, c in ch.items()}
    pr['D3_sigma'] = {'statement': spec['predictions']['D3_sigma'], 'per_instance': sig,
                      'held': len(sig) == 2 and all(v and v.get('assert_passed') for v in sig.values())}
    boxes = {}
    for c in ch.values():
        for bk, b in ((c.get('d3') or {}).get('boxes') or {}).items():
            boxes[bk] = {'min_margin': b.get('min_margin'), 'n_outside': b.get('n_outside'),
                         'n_outside_strict': b.get('n_outside_strict'), 'n_values': b.get('n_values'),
                         'min_relative_margin': b.get('min_relative_margin'), 'complete': b.get('complete')}
    pr['D3_boxes'] = {'statement': spec['predictions']['D3_boxes'], 'per_block': boxes,
                      'held': len(boxes) == 3 and all(b['complete'] and b['n_outside'] == 0 for b in boxes.values())}
    # D-1
    d1 = {}
    for c in ch.values():
        x = c.get('d1') or {}
        if x.get('block'):
            d1[x['block']] = {dep: (x.get(dep) or {}).get('offset_eur') for dep in ('production', 'tail')}
            d1[x['block']]['allowance_eur'] = spec['blocks'][x['block']]['allowance_eur']
            d1[x['block']]['all_loaded'] = {dep: (x.get(dep) or {}).get('all_loaded') for dep in ('production', 'tail')}
    pr['D1a'] = {'statement': spec['predictions']['D1a'],
                 'per_block': {bk: {'offset_eur': v['production'], 'allowance_eur': v['allowance_eur'],
                                    'held': v['production'] is not None and abs(v['production']) >= v['allowance_eur']}
                               for bk, v in d1.items()}}
    pr['D1a']['held'] = len(d1) == 2 and all(v['held'] for v in pr['D1a']['per_block'].values())
    pr['D1b'] = {'statement': spec['predictions']['D1b'],
                 'per_block': {bk: {'offset_eur': v['tail'], 'allowance_eur': v['allowance_eur'],
                                    'bound_eur': v['allowance_eur'] * th['D1b_fraction'],
                                    'held': v['tail'] is not None
                                    and abs(v['tail']) <= v['allowance_eur'] * th['D1b_fraction']}
                               for bk, v in d1.items()}}
    pr['D1b']['held'] = len(d1) == 2 and all(v['held'] for v in pr['D1b']['per_block'].values())
    # D-2
    d2 = {bk: (sol.get(f'D2|{bk}') or {}).get('delta_pin') for bk in ('2025|Winter', '2035|Summer')}
    pr['D2'] = {'statement': spec['predictions']['D2'], 'per_block': d2,
                'held': all(v is not None and v['delta_pin_eur_per_day'] >= th['D2_min_eur_per_day']
                            for v in d2.values())}
    return pr


# ======================================================================================================================
#  CHILD (one instance): D-3 initialisation round + inventory + boxes, then D-1
# ======================================================================================================================
class _StopInit(Exception):
    """Raised at production's `_build_pristine_snapshot_bases` (called once, after the initialisation round and the
    ADMM preparation, right before ADMM cycle 1): nothing of cycle 1 runs."""


def _fake_ok():
    from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
    r = SolverResults()
    r.solver.status = SolverStatus.ok
    r.solver.termination_condition = TerminationCondition.optimal
    return r


def _fill_none(block):
    import pyomo.environ as pe
    n = 0
    for v in block.component_data_objects(pe.Var, descend_into=True):
        if v.value is None:
            x = 0.0
            if v.lb is not None and x < v.lb:
                x = v.lb
            if v.ub is not None and x > v.ub:
                x = v.ub
            v.set_value(x, skip_validation=True)
            n += 1
    return n


def _init_round(planning, candidate, srp, R, cap, trial):
    """THE child initialisation-round solve site: production run_operational_planning (distributed) under the p58
    rescale, stopped by the sentinel at `_build_pristine_snapshot_bases`. Wrappers record (never alter) what
    production computes; in --trial every optimize is stubbed and sigma's computation returns the fixed sigma."""
    originals = {k: getattr(srp, k) for k in ('_build_pristine_snapshot_bases', '_resolve_common_admm_objective_scale',
                                              '_compute_common_admm_objective_scale', '_drain_network_ipopt_solve_records',
                                              'create_shared_energy_storage_model', 'create_transmission_network_model',
                                              'create_distribution_networks_models', 'create_admm_variables')}

    def w_vars(planning_problem):
        consensus_vars, dual_vars = originals['create_admm_variables'](planning_problem)
        cap['consensus_vars'], cap['dual_vars'] = consensus_vars, dual_vars
        return consensus_vars, dual_vars

    def w_bases(admm_parameters, transmission_network, tso_model, distribution_networks, dso_models):
        cap['tso_model'], cap['dso_models'] = tso_model, dso_models
        raise _StopInit('sentinel at _build_pristine_snapshot_bases (before ADMM cycle 1)')

    def w_compute(planning_problem, tso_model, dso_models):
        entries = []
        tnw = planning_problem.transmission_network
        import pyomo.environ as pe
        for year in tnw.years:
            for day in tnw.days:
                w = srp._get_admm_block_weight(tnw, year, day)
                entries.append({'agent': f'TSO|{year}|{day}', 'weight': w,
                                'raw': float(pe.value(tso_model[year][day].objective.expr))})
        for node_id, dnw in planning_problem.distribution_networks.items():
            for year in dnw.years:
                for day in dnw.days:
                    w = srp._get_admm_block_weight(dnw, year, day)
                    entries.append({'agent': f'DSO{node_id}|{year}|{day}', 'weight': w,
                                    'raw': float(pe.value(dso_models[node_id][year][day].objective.expr))})
        for e in entries:
            e['weighted_abs'] = abs(e['weight'] * e['raw'])
        cap['sigma_entries'] = entries
        if trial:
            value = float(planning_problem.params.admm.objective_scale)
            cap['sigma_computed_replaced_in_trial'] = True
            return value
        value = originals['_compute_common_admm_objective_scale'](planning_problem, tso_model, dso_models)
        cap['sigma_computed'] = value
        return value

    def w_resolve(objective_scale_computed, admm_parameters):
        cap['sigma_window'] = {'sigma_fixed': admm_parameters.objective_scale,
                               'assert_factor': admm_parameters.objective_scale_assert_factor,
                               'sigma_computed': objective_scale_computed,
                               'ratio': (objective_scale_computed / admm_parameters.objective_scale
                                         if admm_parameters.objective_scale else None)}
        try:
            out = originals['_resolve_common_admm_objective_scale'](objective_scale_computed, admm_parameters)
        except Exception as error:
            cap['sigma_window']['assert_passed'] = False
            cap['sigma_window']['assert_error'] = f'{type(error).__name__}: {error}'
            raise
        cap['sigma_window']['assert_passed'] = True
        cap['sigma_window']['used'] = out[0]
        return out

    def w_drain(planning_problem, round_index):
        out = originals['_drain_network_ipopt_solve_records'](planning_problem, round_index)
        cap.setdefault('drained', []).append({'round': round_index, 'records': deepcopy(out)})
        return out

    def w_esso(shared_ess_data, consensus_vars, candidate_solution):
        if trial:
            def esso_stub(models, **_k):
                out = {}
                for n in models:
                    _fill_none(models[n])
                    out[n] = _fake_ok()
                return out
            shared_ess_data.optimize = esso_stub
        esso_model, results = originals['create_shared_energy_storage_model'](shared_ess_data, consensus_vars,
                                                                              candidate_solution)
        cap['esso_model'], cap['esso_results'] = esso_model, results
        return esso_model, results

    def w_tso(planning_problem, consensus_vars, candidate_solution):
        if trial:
            tnw = planning_problem.transmission_network
            tnw.optimize = lambda model, **_k: {y: {d: (_fill_none(model[y][d]), _fake_ok())[1] for d in tnw.days}
                                                for y in tnw.years}
        tso_model, results = originals['create_transmission_network_model'](planning_problem, consensus_vars,
                                                                            candidate_solution)
        cap['tso_results'] = results
        return tso_model, results

    def w_dso(distribution_networks, consensus_vars, candidate_solution, **kw):
        if trial:
            for dnw in distribution_networks.values():
                dnw.optimize = (lambda dd: (lambda model, **_k: {y: {d: (_fill_none(model[y][d]), _fake_ok())[1]
                                                                     for d in dd.days} for y in dd.years}))(dnw)
        dso_models, results = originals['create_distribution_networks_models'](distribution_networks, consensus_vars,
                                                                               candidate_solution, **kw)
        cap['dso_results'] = results
        return dso_models, results

    srp._build_pristine_snapshot_bases = w_bases
    srp.create_admm_variables = w_vars
    srp._compute_common_admm_objective_scale = w_compute
    srp._resolve_common_admm_objective_scale = w_resolve
    srp._drain_network_ipopt_solve_records = w_drain
    srp.create_shared_energy_storage_model = w_esso
    srp.create_transmission_network_model = w_tso
    srp.create_distribution_networks_models = w_dso
    p58_applied = {}
    t0 = time.perf_counter()
    try:
        with R.patched_admm_objectives(p58_applied):
            out = planning.run_operational_planning(type='distributed', candidate_solution=deepcopy(candidate),
                                                    print_results=False, debug_flag=False, return_state=True)
        cap['returned_without_sentinel'] = True
        cap['state_initialization_failed'] = (out[-1] or {}).get('initialization_failed')
    except _StopInit as stop:
        cap['stopped_by_sentinel'] = str(stop)
    except Exception:  # noqa: BLE001 -- recorded (a sigma assert failure lands here)
        cap['exception'] = traceback.format_exc()
    finally:
        for k, v in originals.items():
            setattr(srp, k, v)
        if trial:
            for holder in [planning.transmission_network] + list(planning.distribution_networks.values()) + \
                          [planning.shared_ess_data]:
                holder.__dict__.pop('optimize', None)
    cap['wall_s'] = time.perf_counter() - t0
    cap['p58_applied'] = p58_applied
    cap['wrappers_restored'] = all(getattr(srp, k) is v for k, v in originals.items())


def _var_state(block):
    import pyomo.environ as pe
    out = {}
    for comp in block.component_objects(pe.Var, active=True, descend_into=True):
        for _idx, v in comp.items():
            out[v.name] = (comp.name, v.lb, v.ub, bool(v.fixed), v.value if v.fixed else None)
    return out


def _con_state(block):
    import pyomo.environ as pe
    out = {}
    for comp in block.component_objects(pe.Constraint, active=None, descend_into=True):
        for _idx, c in comp.items():
            out[c.name] = (comp.name, bool(c.active))
    return out


def _diff_blocks(admm_block, plain_block):
    va, vp = _var_state(admm_block), _var_state(plain_block)
    changes, new = {}, {}
    for name, (comp, lb, ub, fx, fv) in va.items():
        if name not in vp:
            e = new.setdefault(comp, {'n': 0, 'n_fixed': 0, 'n_finite_lb': 0, 'n_finite_ub': 0, 'lb_range': None,
                                      'ub_range': None, 'examples': []})
            e['n'] += 1
            e['n_fixed'] += int(fx)
            if lb is not None:
                e['n_finite_lb'] += 1
                e['lb_range'] = [min(e['lb_range'][0], lb), max(e['lb_range'][1], lb)] if e['lb_range'] else [lb, lb]
            if ub is not None:
                e['n_finite_ub'] += 1
                e['ub_range'] = [min(e['ub_range'][0], ub), max(e['ub_range'][1], ub)] if e['ub_range'] else [ub, ub]
            if len(e['examples']) < 2:
                e['examples'].append([name, lb, ub, fx, fv])
            continue
        _c, plb, pub, pfx, pfv = vp[name]
        kinds = []
        if lb != plb:
            kinds.append('lb')
        if ub != pub:
            kinds.append('ub')
        if fx != pfx:
            kinds.append('fixed' if fx else 'unfixed')
        elif fx and fv != pfv:
            kinds.append('fixed_value')
        for k in kinds:
            e = changes.setdefault(f'{comp}:{k}', {'n': 0, 'examples': []})
            e['n'] += 1
            if len(e['examples']) < 3:
                e['examples'].append({'var': name, 'admm': [lb, ub, fx, fv], 'plain': [plb, pub, pfx, pfv]})
    ca, cp = _con_state(admm_block), _con_state(plain_block)
    new_cons, act = {}, {}
    for name, (comp, active) in ca.items():
        if name not in cp:
            e = new_cons.setdefault(comp, {'n_rows': 0, 'n_active': 0})
            e['n_rows'] += 1
            e['n_active'] += int(active)
        elif active != cp[name][1]:
            e = act.setdefault(comp, {'n': 0, 'now_active': active})
            e['n'] += 1
    gone = sorted({c for n, (c, _a) in cp.items() if n not in ca})
    return {'n_vars_admm': len(va), 'n_vars_plain': len(vp), 'var_bound_or_fix_changes': changes,
            'var_components_only_in_admm': new, 'constraint_components_only_in_admm': new_cons,
            'constraint_active_state_changes': act, 'constraint_components_only_in_plain': gone}


def d3_inventory(planning, candidate, tso_model, dso_models, blocks):
    """Zero solves. Every ADMM-prepared block of the listed (year, day) blocks against a plain build of the same
    network with the same candidate (production NetworkData.build_model + update_model_with_candidate_solution)."""
    tn = planning.transmission_network
    cap_total = candidate['total_capacity']
    out = {}
    plain_t = tn.build_model()
    tn.update_model_with_candidate_solution(plain_t, cap_total)
    for y, d in blocks:
        out[f'TSO|{y}|{d}'] = _diff_blocks(tso_model[y][d], plain_t[y][d])
    del plain_t
    for node_id, dnw in planning.distribution_networks.items():
        plain_d = dnw.build_model()
        dnw.update_model_with_candidate_solution(plain_d, cap_total)
        for y, d in blocks:
            out[f'DSO{node_id}|{y}|{d}'] = _diff_blocks(dso_models[node_id][y][d], plain_d[y][d])
        del plain_d
    return out


def _sqrt_or_none(x):
    return None if x is None else math.sqrt(max(x, 0.0))


def _box_intersect(*boxes):
    lo, hi = -math.inf, math.inf
    for b in boxes:
        if b is None:
            continue
        if b[0] is not None:
            lo = max(lo, b[0])
        if b[1] is not None:
            hi = min(hi, b[1])
    return [lo, hi]


def _voltage_rows_box(model, network, node_idx, s_m, s_o, p, params):
    """[v_min, v_max] of the voltage-magnitude rows when their slacks are absent or fixed at 0; None otherwise."""
    import pyomo.environ as pe
    node = network.nodes[node_idx]
    if not params.slacks.grid_operation.voltage:
        return [node.v_min, node.v_max], 'no voltage slacks'
    dn_s = model.slack_v_sqr_down[node_idx, s_m, s_o, p]
    up_s = model.slack_v_sqr_up[node_idx, s_m, s_o, p]
    lo = node.v_min if (dn_s.fixed and pe.value(dn_s) == 0.0) or (dn_s.ub is not None and dn_s.ub == 0.0) else None
    hi = node.v_max if (up_s.fixed and pe.value(up_s) == 0.0) or (up_s.ub is not None and up_s.ub == 0.0) else None
    return [lo, hi], f'slack down fixed0={lo is not None}, up fixed0={hi is not None}'


def d3_boxes(planning, tso_model, dso_models, central, blocks, bound_tol_rel):
    """Zero solves. Each block's C_central interface values (the single-scenario solutions) against the effective
    interface boxes of the ADMM-prepared models. A value counts as OUTSIDE when it exceeds a bound by more than
    bound_tol_rel x max(1, |bound|) (IPOPT's default bound_relax_factor, the relaxation the central solves themselves
    were computed under); the strict count (any negative margin) is recorded beside it."""
    import pyomo.environ as pe
    tn = planning.transmission_network
    out = {}
    for y, d in blocks:
        bk = f'{y}|{d}'
        c = central.get(bk) or {}
        rec = {'complete': bool(c.get('complete')), 'per_family': {}, 'n_values': 0, 'n_outside': 0,
               'n_outside_strict': 0, 'bound_tol_rel': bound_tol_rel, 'min_margin': None,
               'min_relative_margin': None}
        if not c.get('complete'):
            out[bk] = rec
            continue
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
                    rows, _note = _voltage_rows_box(tb, tnet, adn_node, s_m, s_o, p, tn.params)
                    add('TSO_V_box', V, _box_intersect([vv.lb, vv.ub], [_sqrt_or_none(vs.lb), _sqrt_or_none(vs.ub)],
                                                       rows), where, 'pu')
                    g_p, g_q = db.pg[ref_gen, s_m, s_o, p], db.qg[ref_gen, s_m, s_o, p]
                    add('DSO_P_refgen_box', P, [None if g_p.lb is None else g_p.lb * sb_d,
                                                None if g_p.ub is None else g_p.ub * sb_d], where, 'MW')
                    add('DSO_Q_refgen_box', Q, [None if g_q.lb is None else g_q.lb * sb_d,
                                                None if g_q.ub is None else g_q.ub * sb_d], where, 'MVAr')
                    wv, ws = db.vmag[ref_node, s_m, s_o, p], db.vmag_sqr[ref_node, s_m, s_o, p]
                    rows_d, _n = _voltage_rows_box(db, dnet, ref_node, s_m, s_o, p,
                                                   planning.distribution_networks[node_id].params)
                    add('DSO_V_box', V, _box_intersect([wv.lb, wv.ub], [_sqrt_or_none(ws.lb), _sqrt_or_none(ws.ub)],
                                                       rows_d), where, 'pu')
        for e in fam.values():
            for k in ('min_margin', 'min_relative_margin'):
                if e[k] == math.inf:
                    e[k] = None
        finite = [e['min_margin'] for e in fam.values() if e['min_margin'] is not None]
        rel = [e['min_relative_margin'] for e in fam.values() if e['min_relative_margin'] is not None]
        rec['min_margin'] = {f: {'value': e['min_margin'], 'unit': e['unit'], 'at': e['min_margin_at']}
                             for f, e in fam.items()}
        rec['min_relative_margin'] = min(rel) if rel else None
        rec['min_margin_any_unit'] = min(finite) if finite else None
        out[bk] = rec
    return out


def d1_fix(planning, tso_model, dso_models, central, y, d):
    """Fix every per-scenario interface P, Q, V copy of block (y, d) at the central solution; verify the resulting
    interface expressions reproduce it."""
    import pyomo.environ as pe
    from model_construction_helpers import sess_na_scenario
    tn = planning.transmission_network
    tb, tnet = tso_model[y][d], tn.network[y][d]
    sb_t = tnet.baseMVA
    rec = {'n_fixed': {'TSO_delta_p': 0, 'TSO_delta_q': 0, 'TSO_vmag': 0, 'DSO_pg_ref': 0, 'DSO_qg_ref': 0,
                       'DSO_vmag_ref': 0}, 'max_abs_reproduction_error': {'P_mw': 0.0, 'Q_mvar': 0.0, 'V_pu': 0.0},
           'checks': {}}
    for dn_i, node_id in enumerate(tn.active_distribution_network_nodes):
        adn_load = tnet.get_adn_load_idx(node_id)
        adn_node = tnet.get_node_idx(node_id)
        dnw = planning.distribution_networks[node_id]
        dnet = dnw.network[y][d]
        db = dso_models[node_id][y][d]
        sb_d = dnet.baseMVA
        ref_gen = dnet.get_reference_gen_idx()
        ref_id = dnet.get_reference_node_id()
        ref_node = dnet.get_node_idx(ref_id)
        rec['checks'][f'DN{node_id}'] = {
            'dn_baseMVA': sb_d, 'tn_baseMVA': sb_t,
            'ref_node_loads': [ld.load_id for ld in dnet.loads if ld.bus == ref_id],
            'ref_node_shunt_gs_bs': [dnet.nodes[ref_node].gs, dnet.nodes[ref_node].bs],
            'tso_flex_legs_fixed_zero': all(
                getattr(tb, leg)[adn_load, a, b, p].fixed and pe.value(getattr(tb, leg)[adn_load, a, b, p]) == 0.0
                for leg in ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down')
                for a in tb.scenarios_market for b in tb.scenarios_operation for p in tb.periods)}
        ess = [e for e in db.shared_energy_storages if dnet.shared_energy_storages[e].bus == ref_id]
        na = sess_na_scenario(db)
        max_ess = 0.0
        for a in tb.scenarios_market:
            for b in tb.scenarios_operation:
                sc = central['interface'][f'{a}{b}'][str(node_id)]
                for p in tb.periods:
                    P, Q, V = sc['P_mw'][p], sc['Q_mvar'][p], sc['V_pu'][p]
                    tb.interface_delta_p[dn_i, a, b, p].fix(P / sb_t - pe.value(tb.pc[adn_load, a, b, p]))
                    tb.interface_delta_q[dn_i, a, b, p].fix(Q / sb_t - pe.value(tb.qc[adn_load, a, b, p]))
                    tb.vmag[adn_node, a, b, p].fix(V)
                    rec['n_fixed']['TSO_delta_p'] += 1
                    rec['n_fixed']['TSO_delta_q'] += 1
                    rec['n_fixed']['TSO_vmag'] += 1
                    e_p = sum(pe.value(db.shared_es_pnet[e, na[0], na[1], p]) for e in ess)
                    e_q = sum(pe.value(db.shared_es_qnet[e, na[0], na[1], p]) for e in ess)
                    max_ess = max(max_ess, abs(e_p), abs(e_q))
                    db.pg[ref_gen, a, b, p].fix(P / sb_d + e_p)
                    db.qg[ref_gen, a, b, p].fix(Q / sb_d + e_q)
                    db.vmag[ref_node, a, b, p].fix(V)
                    rec['n_fixed']['DSO_pg_ref'] += 1
                    rec['n_fixed']['DSO_qg_ref'] += 1
                    rec['n_fixed']['DSO_vmag_ref'] += 1
                    err = rec['max_abs_reproduction_error']
                    err['P_mw'] = max(err['P_mw'], abs(pe.value(tb.pc_adn[dn_i, a, b, p]) * sb_t - P),
                                      abs(pe.value(db.pg_adn[a, b, p]) * sb_d - P))
                    err['Q_mvar'] = max(err['Q_mvar'], abs(pe.value(tb.qc_adn[dn_i, a, b, p]) * sb_t - Q),
                                        abs(pe.value(db.qg_adn[a, b, p]) * sb_d - Q))
                    err['V_pu'] = max(err['V_pu'], abs(pe.value(tb.vmag_adn[dn_i, a, b, p]) - V),
                                      abs(pe.value(db.vmag_adn[a, b, p]) - V))
        rec['checks'][f'DN{node_id}']['max_abs_shared_ess_term_pu'] = max_ess
    rec['reproduces_central_1e-9'] = all(v <= 1e-9 for v in rec['max_abs_reproduction_error'].values())
    return rec


def active_bounds(model, tol):
    import pyomo.environ as pe
    out = {'vars_at_lb': 0, 'vars_at_ub': 0, 'n_free_vars': 0, 'ineq_rows_at_lb': 0, 'ineq_rows_at_ub': 0,
           'n_ineq_rows': 0, 'by_component': {}}
    for v in model.component_data_objects(pe.Var, active=True, descend_into=True):
        if v.fixed or v.value is None:
            continue
        out['n_free_vars'] += 1
        comp = v.parent_component().name
        if v.lb is not None and v.value - v.lb <= tol:
            out['vars_at_lb'] += 1
            out['by_component'][f'{comp}:lb'] = out['by_component'].get(f'{comp}:lb', 0) + 1
        if v.ub is not None and v.ub - v.value <= tol:
            out['vars_at_ub'] += 1
            out['by_component'][f'{comp}:ub'] = out['by_component'].get(f'{comp}:ub', 0) + 1
    for c in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
        if c.equality:
            continue
        body = pe.value(c.body, exception=False)
        if body is None:
            continue
        out['n_ineq_rows'] += 1
        if c.has_lb() and body - pe.value(c.lower) <= tol:
            out['ineq_rows_at_lb'] += 1
        if c.has_ub() and pe.value(c.upper) - body <= tol:
            out['ineq_rows_at_ub'] += 1
    return out


def _d1_solve(network, model, params, trial):
    """THE child D-1 solve site: production Network.run_smopf, from_warm_start=True (the ADMM-cycle form)."""
    from helper_functions import solver_result_succeeded
    n0 = len(getattr(network, 'ipopt_solve_records', None) or [])
    t0 = time.perf_counter()
    if trial:
        return {'trial_not_solved': True, 'solution_loaded': False, 'wall_s': 0.0, 'ipopt_records': []}
    result = network.run_smopf(model, params, from_warm_start=True, print_header=True)
    recs = list(getattr(network, 'ipopt_solve_records', None) or [])[n0:]
    return {'wall_s': time.perf_counter() - t0, 'ipopt_records': recs,
            'termination_condition': str(result.solver.termination_condition) if result is not None else None,
            'solver_status': str(result.solver.status) if result is not None else None,
            'solution_loaded': bool(result is not None and solver_result_succeeded(result)), 'rss': rss()}


def _active_objective(model):
    import pyomo.environ as pe
    act = [o for o in model.component_data_objects(pe.Objective, active=True)]
    if len(act) != 1:
        return {'n_active': len(act)}
    return {'name': act[0].name, 'value': float(pe.value(act[0]))}


def d1_cycle1_params(planning, models, cap, srp):
    """Zero solves. Put the ADMM-prepared models in the Param state production gives them right before the cycle-1
    solves (requests = the other side's initial values, duals = the initial 0, the deviation Params, the shared-ESS
    capacities): production's own `_set_interface_deviation_params` and `update_distribution_coordination_models_
    and_solve` / `update_transmission_coordination_model_and_solve`, with every network `optimize` replaced ON THE
    INSTANCE by a stub that returns 'ok' WITHOUT touching the model (no solve; restored after). No consensus or dual
    update is made between the two (no cycle-1 DSO solve precedes the TSO here): the TSO's requests are the DSOs'
    initial expectations. Without this the expectation-channel Params (e.g. dual_vmag_req) hold no value and the ADMM
    objective cannot be written."""
    tn, dns = planning.transmission_network, planning.distribution_networks
    cv, dv = cap['consensus_vars'], cap['dual_vars']
    a = planning.params.admm
    caps = planning.shared_ess_data.get_updated_capacities(models['esso'])
    holders = [tn] + list(dns.values())
    calls = {'stub_calls': 0}

    def stub_for(h):
        def stub(model, **_k):
            calls['stub_calls'] += 1
            return {y: {d: _fake_ok() for d in h.days} for y in h.years}
        return stub
    for h in holders:
        h.optimize = stub_for(h)
    try:
        srp._set_interface_deviation_params(planning, models['tso'], models['dso'], cv, dv, 'dso')
        srp.update_distribution_coordination_models_and_solve(
            dns, models['dso'], cv['vmag'], dv['vmag']['dso'], cv['pf'], dv['pf']['dso'], cv['ess'], dv['ess']['dso'],
            a, caps, from_warm_start=True, parallel_execution=planning.parallel_execution, cycle=1,
            dso_pristine_base=None)
        srp._set_interface_deviation_params(planning, models['tso'], models['dso'], cv, dv, 'tso')
        srp.update_transmission_coordination_model_and_solve(
            tn, models['tso'], cv['vmag'], dv['vmag']['tso'], cv['pf'], dv['pf']['tso'], cv['ess'], dv['ess']['tso'],
            a, caps, from_warm_start=True, cycle=1, tso_pristine_base=None)
    finally:
        for h in holders:
            h.__dict__.pop('optimize', None)
    calls['stubs_removed'] = all('optimize' not in h.__dict__ for h in holders)
    return calls


def d1_round(planning, models, srp, y, d, depth, central, child_dir, tol, trial):
    import pyomo.environ as pe
    tn = planning.transmission_network
    order = [(f'DSO{n}', planning.distribution_networks[n], models['dso'][n][y][d])
             for n in sorted(planning.distribution_networks)] + [('TSO', tn, models['tso'][y][d])]
    out = {'depth': depth, 'agents': {}}
    for label, holder, model in order:
        network = holder.network[y][d]
        network.logs_dir = os.path.join(child_dir, f'logs_d1_{depth}')
        os.makedirs(network.logs_dir, exist_ok=True)
        options = holder.params.solver_params.options or {}
        st = _d1_solve(network, model, holder.params, trial)
        r = {'compl_inf_tol_option': options.get('compl_inf_tol'), 'solve': st,
             'ipopt': {k: (st.get('ipopt_records') or [{}])[-1].get(k) for k in (
                 'exit', 'iterations', 'mu_final', 'mu_floor', 'mu_over_floor', 'floor_status', 'obj_scaling_factor',
                 'compl_inf_tol_in_force', 'compl_inf_tol_logged', 'tol_in_force')},
             'active_objective': None, 'local_objective_components': None}
        for key, fn in (('active_objective', lambda: _active_objective(model)),
                        ('local_objective_components',
                         lambda: srp._get_local_objective_components(model, holder.params)),
                        ('active_bounds', lambda: active_bounds(model, tol))):
            try:
                r[key] = fn()
            except Exception:  # noqa: BLE001 -- capture only
                r.setdefault('capture_errors', {})[key] = traceback.format_exc()
        try:
            r['primal'] = save_primal(model, os.path.join(child_dir, 'primal', f'D1_{depth}_{label}_{y}_{d}.json.gz'),
                                      {'depth': depth, 'agent': label, 'block': f'{y}|{d}',
                                       'solution_loaded': st.get('solution_loaded'), 'trial': trial})
        except Exception:  # noqa: BLE001
            r['primal_error'] = traceback.format_exc()
        out['agents'][label] = r
        _log(f"D-1 {depth} {label} {y}|{d}: {st.get('termination_condition')} {r['ipopt'].get('exit')} "
             f"it {r['ipopt'].get('iterations')} mu {r['ipopt'].get('mu_final')} wall {st.get('wall_s', 0):.1f}s")
    comp = srp._get_operational_recourse_block_components(planning, models)
    per = {}
    for (kind, node_id, yy, dd), value in comp.items():
        if kind == 'SALVAGE' or (yy, dd) != (y, d):
            continue
        label = 'TSO' if kind == 'TSO' else f'DSO{node_id}'
        holder = tn if kind == 'TSO' else planning.distribution_networks[node_id]
        model = models['tso'][y][d] if kind == 'TSO' else models['dso'][node_id][y][d]
        per[label] = {'q_block_settlement_excluded_in_full': float(value),
                      'weight': srp._get_admm_block_weight(holder, y, d),
                      'primal_value': float(holder.network[y][d].get_primal_value(model, holder.params)),
                      'settlement_total': float(srp._get_local_interface_settlement(model, part='total')),
                      'voltage_pin': float(srp._get_local_voltage_pin(model))}
    total = sum(v['q_block_settlement_excluded_in_full'] for v in per.values())
    w_b = central['w_b']
    out['per_agent_q'] = per
    out['coupling_wired'] = srp._per_scenario_interface_coupling_wired(models)
    out['sum_q_eur'] = total
    out['w_b_C_central_eur'] = w_b * central['C_central']
    out['offset_eur'] = total - w_b * central['C_central']
    out['offset_eur_per_day'] = out['offset_eur'] / w_b
    out['allowance_eur'] = central['allowance_eur']
    out['all_loaded'] = all(a['solve'].get('solution_loaded') for a in out['agents'].values())
    out['all_optimal'] = all(a['ipopt'].get('exit') == 'Optimal Solution Found.' for a in out['agents'].values())
    return out


def child_main(args):
    t_start = time.perf_counter()
    with open(args.child_in) as handle:
        cin = json.load(handle)
    trial = bool(cin['trial'])
    cdir = cin['child_dir']
    out = {'schema': 's56_step4_diag_child_v1', 'instance': cin['instance'], 'utc_start': _utc(), 'trial': trial,
           'error': None, 'pid': os.getpid(), 'env_thread_caps': {k: os.environ.get(k) for k in (
               'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS')}, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH')}
    try:
        if _sha(cin['spec_path']) != cin['spec_sha256']:
            raise RuntimeError('spec sha256 changed')
        if _sha(cin['central_path']) != cin['central_sha256']:
            raise RuntimeError('central input sha256 changed')
        with open(cin['spec_path']) as handle:
            spec = json.load(handle)
        with open(cin['central_path']) as handle:
            central = json.load(handle)
        ac = spec['admm_configuration']
        if out['env_thread_caps'] != ac['thread_caps_in_child_env']:
            raise RuntimeError(f"thread caps not in force: {out['env_thread_caps']}")
        if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
            raise RuntimeError('NLP_SOLVER_PATH')
        import p515_s44_campaign_harness as H
        import p515_g_g1_g4_admm_gates as G
        import p58_rescale as R
        import shared_resources_planning as srp
        # capture checklist (rule eleven), before anything is built
        for name in ('_build_pristine_snapshot_bases', '_resolve_common_admm_objective_scale',
                     '_compute_common_admm_objective_scale', '_drain_network_ipopt_solve_records',
                     'create_shared_energy_storage_model', 'create_transmission_network_model',
                     'create_distribution_networks_models', '_get_operational_recourse_block_components',
                     '_get_local_interface_settlement', '_get_local_voltage_pin', '_get_local_objective_components',
                     '_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
                     '_per_scenario_interface_coupling_wired', '_get_admm_block_weight', 'create_admm_variables',
                     '_set_interface_deviation_params', 'update_distribution_coordination_models_and_solve',
                     'update_transmission_coordination_model_and_solve'):
            assert callable(getattr(srp, name, None)), name
        import inspect
        assert '_build_pristine_snapshot_bases(' in inspect.getsource(srp._run_operational_planning)
        cg = coupling_gates_literals()
        for k, v in (('coupling', cg['COUPLING_V1_NOQ']), ('convergence_depth_tail', cg['CONVERGENCE_DEPTH_TAIL']),
                     ('case_file_anderson_acceleration', cg['CASE_FILE_AA'])):
            if ac[k] != v:
                raise RuntimeError(f'spec admm_configuration.{k} != {COUPLING_GATES} literal')
        derived = H.validate_derived_instance(cin['derived'])
        out['derived_installed'] = H.install_derived_instance(derived, os.path.join(cdir, 'derived'))
        report, holder = {}, {}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        eval_id = f"p515s56diag_{cin['instance']}_{'trial' if trial else 'run'}_{stamp}"
        x0 = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
        planning, sed, candidate = G._construct_arm_planning(
            ac['arm_label'], os.path.join(cdir, 'arm'), report, investment_map=x0, eval_id=eval_id,
            num_max_iters_override=ac['cap'], apply_rho=ac['apply_rho'], investment_year=ac['investment_year'])
        out['eval_id'] = eval_id
        spec_like = {'configuration': {'overrides': {}, 'arm_label': ac['arm_label'],
                                       'case_file_anderson_acceleration': dict(ac['case_file_anderson_acceleration']),
                                       'ess_ageing_baseline': deepcopy(ac['ess_ageing_baseline']),
                                       'ess_params_file': dict(ac['ess_params_file']),
                                       'ess_ageing_baseline_label': ac['ess_ageing_baseline_label'],
                                       'convergence_depth_tail': dict(ac['convergence_depth_tail']),
                                       'derived_instance': derived},
                     'cap': ac['cap'], 'required_consecutive_cycles': ac['required_consecutive_cycles']}
        hook = H._config_hook_factory(spec_like, holder, overrides={}, model_variant=None,
                                      investment_year=ac['investment_year'], expected_floor_rows=None,
                                      flex_price_multiplier=None, interface_deviation_premium=ac['premium'],
                                      release_solution_bookkeeping=None, interface_coupling=ac['coupling'])
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        out['admm_in_force'] = {k: getattr(a, k, 'ABSENT') for k in (
            'per_scenario_interface_coupling', 'per_scenario_voltage_mode', 'row18_charge_reactive', 'objective_scale',
            'objective_scale_assert_factor', 'convergence_depth_tail', 'num_max_iters', 'shared_ess_initialization')}
        out['admm_in_force']['interface_deviation_premium'] = dict(a.interface_deviation_premium)
        out['hook_holder'] = {k: v for k, v in holder.items() if not k.startswith('_')}
        out['recovery_in_force_init'] = {
            lab: [h.params.solver_params.recovery_enabled, h.params.solver_params.recovery_tier2_enabled]
            for lab, h in [('TSO', planning.transmission_network)] +
            [(f'DSO{n}', planning.distribution_networks[n]) for n in sorted(planning.distribution_networks)]}
        # ---------------------------------------------------------------------------------------------- D-3 init
        _log(f"child {cin['instance']}: initialisation round")
        cap = {}
        _init_round(planning, candidate, srp, R, cap, trial)
        init_sites = dict(GUARD.permitted_sites)
        out['init_round'] = {k: v for k, v in cap.items() if k not in ('tso_model', 'dso_models', 'esso_model',
                                                                       'tso_results', 'dso_results', 'esso_results')}
        out['init_round']['guard_sites_after'] = init_sites

        def term(r):
            return {'status': str(r.solver.status), 'termination': str(r.solver.termination_condition)}
        if 'tso_results' in cap:
            out['init_round']['tso_results'] = {f'{y}|{d}': term(v) for y, dd in cap['tso_results'].items()
                                                for d, v in dd.items()}
        if 'dso_results' in cap:
            out['init_round']['dso_results'] = {f'DSO{n}|{y}|{d}': term(v) for n, yy in cap['dso_results'].items()
                                                for y, dd in yy.items() for d, v in dd.items()}
        if 'esso_results' in cap:
            out['init_round']['esso_results'] = {f'ESSO{n}': term(v) for n, v in cap['esso_results'].items()}
        sw = cap.get('sigma_window') or {}
        f = sw.get('assert_factor')
        out_d3 = {'sigma': dict(sw, window=([sw['sigma_fixed'] / f, sw['sigma_fixed'] * f] if f else None),
                                assert_passed=bool(sw.get('assert_passed')) and 'stopped_by_sentinel' in cap,
                                largest_entries=sorted(cap.get('sigma_entries') or [], key=lambda e: -e['weighted_abs'])[:6],
                                trial_replaced=bool(cap.get('sigma_computed_replaced_in_trial')))}
        out['d3'] = out_d3
        if 'stopped_by_sentinel' not in cap:
            raise RuntimeError(f"initialisation round did not reach the sentinel: "
                               f"{cap.get('exception') or cap.get('state_initialization_failed')}")
        tso_model, dso_models, esso_model = cap['tso_model'], cap['dso_models'], cap['esso_model']
        models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
        blocks = []
        for bk in cin['blocks']:
            y_s, d = bk.split('|')
            blocks.append((next(yy for yy in planning.transmission_network.years if str(yy) == y_s), d))
        # copy the initialisation-round IPOPT logs into the child dir (hashed by the parent manifest)
        import p56a_oracle as O
        src_logs = os.path.join(O.WORK_DIR, eval_id, 'logs')
        if os.path.isdir(src_logs):
            shutil.copytree(src_logs, os.path.join(cdir, 'ipopt_logs_init'))
        out['init_round']['ipopt_logs_source'] = os.path.relpath(src_logs, REPO)
        # ---------------------------------------------------------------------------------------------- D-3 zero solves
        g0 = dict(GUARD.counts)
        _log(f"child {cin['instance']}: D-3 inventory and boxes (zero solves)")
        out_d3['inventory'] = d3_inventory(planning, candidate, tso_model, dso_models, blocks)
        out_d3['boxes'] = d3_boxes(planning, tso_model, dso_models, central, blocks, spec['d3']['bound_tol_rel'])
        out_d3['zero_solve_verified'] = dict(GUARD.counts) == g0
        # ---------------------------------------------------------------------------------------------- D-1
        y1_s, d1d = cin['d1_block'].split('|')
        y1 = next(yy for yy in planning.transmission_network.years if str(yy) == y1_s)
        c1 = central[cin['d1_block']]
        out_d1 = {'block': cin['d1_block']}
        out['d1'] = out_d1
        if not c1.get('complete'):
            raise RuntimeError(f"central solution of {cin['d1_block']} incomplete: D-1 not run")
        holders = [('TSO', planning.transmission_network)] + [
            (f'DSO{n}', planning.distribution_networks[n]) for n in sorted(planning.distribution_networks)]
        rec_off = {}
        for lab, h in holders:
            sp = h.params.solver_params
            rec_off[lab] = {'before': [sp.recovery_enabled, sp.recovery_tier2_enabled]}
            sp.recovery_enabled = False
            sp.recovery_tier2_enabled = False
            rec_off[lab]['after'] = [sp.recovery_enabled, sp.recovery_tier2_enabled]
        out_d1['recovery'] = rec_off
        g1 = dict(GUARD.counts)
        out_d1['cycle1_params'] = d1_cycle1_params(planning, models, cap, srp)
        out_d1['cycle1_params']['zero_solve_verified'] = dict(GUARD.counts) == g1
        out_d1['fix'] = d1_fix(planning, tso_model, dso_models, c1, y1, d1d)
        out_d1['objectives_evaluable_before_solve'] = {}
        for lab, mdl in [('TSO', tso_model[y1][d1d])] + [(f'DSO{n}', dso_models[n][y1][d1d])
                                                          for n in sorted(dso_models)]:
            try:
                out_d1['objectives_evaluable_before_solve'][lab] = _active_objective(mdl)
            except Exception as error:  # noqa: BLE001
                out_d1['objectives_evaluable_before_solve'][lab] = {'error': f'{type(error).__name__}: {error}'}
        baseline = srp._capture_convergence_depth_tail_baseline(planning, a)
        out_d1['depth_settings'] = {'production_baseline': baseline,
                                    'tail_value': a.convergence_depth_tail['compl_inf_tol'],
                                    'source': 'production _capture_convergence_depth_tail_baseline / '
                                              'admm.convergence_depth_tail (declared by the configuration hook)'}
        tol = spec['d1']['active_bound_tol']
        out_d1['production'] = d1_round(planning, models, srp, y1, d1d, 'production', c1, cdir, tol, trial)
        out_d1['tail_switch_on'] = srp._apply_convergence_depth_tail(planning, a, True, baseline, 'D1-tail')
        try:
            out_d1['tail'] = d1_round(planning, models, srp, y1, d1d, 'tail', c1, cdir, tol, trial)
        finally:
            out_d1['tail_switch_off'] = srp._apply_convergence_depth_tail(planning, a, False, baseline, 'D1-restore')
        out_d1['sites_after'] = dict(GUARD.permitted_sites)
    except Exception:  # noqa: BLE001 -- recorded
        out['error'] = traceback.format_exc()
        _log('child EXCEPTION:\n' + out['error'])
    exp_sites = {f'{SCRIPT}:{k}': v for k, v in cin['expected_sites'].items()}
    exp = sum(exp_sites.values())
    gv = GUARD.verify(exp)
    out['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                    'expected_sites': exp_sites, 'verify_failures': gv,
                    'sites_exact': dict(GUARD.permitted_sites) == exp_sites}
    out['pickle'] = dict(PK)
    out['guards_ok'] = not gv and out['guard']['sites_exact'] and PK == {'load': 0, 'loads': 0}
    mods = {}
    for name, mod in list(sys.modules.items()):
        fl = getattr(mod, '__file__', None)
        if fl and os.path.abspath(fl).startswith(REPO + os.sep) and fl.endswith('.py'):
            rel = os.path.relpath(os.path.abspath(fl), REPO)
            mods[rel] = {'sha256': _sha(rel), 'committed_clean': committed_clean(rel)}
    out['imported_repo_modules'] = mods
    out['dirty_imported_modules'] = sorted(k for k, v in mods.items() if not v['committed_clean'])
    out['rss'] = rss()
    out['wall_s'] = time.perf_counter() - t_start
    out['utc_end'] = _utc()
    write_json(args.child_out, out)
    return 0 if (out['error'] is None and out['guards_ok']) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--trial', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--out-dir')
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--child-in')
    ap.add_argument('--child-out')
    a = ap.parse_args()
    if a.child:
        return child_main(a)
    if a.freeze:
        return freeze(a)
    if (a.trial or a.run) and a.spec and a.out_dir and not (a.trial and a.run):
        return parent(a)
    ap.error('one of --freeze --out-dir D | --trial/--run --spec S --out-dir D')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
