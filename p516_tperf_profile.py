"""P5.16 T-perf -- where the solve time goes. Part A (this machine) and the Part B (--studio) mode.

Authority: P5_16_TPERF_SPEC_v1.md (author-authorized diagnostic, 2026-10-31); TASKS.md line "T-perf".
Not a production change; nothing here enters a table. No ADMM cycle is run.

PART A INSTANCE (hash-verified BEFORE unpickling, against the declared value AND both committed manifests):
  campaign s53_w86_tail_recert (spec ddd6cd44), cell n7_4h_e1, eval key ca8927e75d628bd1...,
  candidate_key db77e1549af855bc... (2025 cohort; node 7 0.25 MVA / 1.0 MWh; nodes 5, 9 empty),
  certified at cycle 112 (certified_cost 653,599,356.24, gross_operational_cost, settlement excluded --
  quoted from post_certification.json for identification only; no objective is compared here).
  certified_models.pkl  sha256 fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04
                        payload {'tso': {year: {day: ConcreteModel}}, 'dso': {node: {year: {day: ConcreteModel}}}}
  esso_models_s39_D.pkl sha256 95f7b38a6ec21833f7b5521ec8332d8022ef6801b4095d1410225438b2cedfc9
                        payload {node: ConcreteModel}            (ESSO: 0 solves; its size is reported)
  Blocks used: TSO 2025 Spring (case9) and DSO node 7 2025 Spring (case33_2). Each case solves its OWN
  `clone()` of the loaded block (the clone deep-copies the block's build-time `network` / `params`
  objects held by its rule partials); the loaded blocks and the pickle files are never modified.

PRODUCTION SOLVE PATH. Every solve is `network.run_smopf(model, params, from_warm_start=...)`, called on
the `network` / `params` objects the block was BUILT with (recovered from its pickled bound-rule partial,
the p515_s48 precedent). That is the exact call `NetworkData.optimize` makes inside
`update_transmission_coordination_model_and_solve` / `update_distribution_coordination_models_and_solve`
(the ADMM-cycle path); the Param updates those functions perform before the call need the ADMM state
(consensus, duals), which is not in the pickle -- the certified block already carries cycle 112's Params.
Options are production's (`_create_smopf_solver`: fixed_variable_treatment, max_iter 500, then the case
file's options from the block's own params), with these additions on the per-case params copy, recorded
per solve:
  all cases   output_file   -> 'tperf_<block>_<case>.log' under <out>/logs (network.logs_dir is redirected
                               on the clone's network copy; production appends '_<year>_<day>'), asserted
                               absent before the solve -> one fresh file per solve
              print_timing_statistics yes   (output only; implies timing_statistics yes)
  case iii    print_info_string yes, print_level 5, nlp_scaling_method gradient-based (both the defaults)

DECLARED SOLVES: exactly 6 = 2 blocks x 3 cases, armed `SolveProfileGuard` (bounded; permitted call site
network.py:_run_smopf_solver_attempt), installed BEFORE any production import, `verify(6)` exact at the end
(too few fails too). An additional per-case cap (this harness) blocks a second solve inside one case, so a
production recovery attempt after a failed primary is blocked (no 7th solve), recorded, and the run stops.
  (i)   warm   -- re-solve AT the certified point (cycle-112 values, Params, multiplier suffixes),
                  from_warm_start=True (production copies ipopt_z*_out -> ipopt_z*_in; dual is exported).
  (ii)  cold   -- production INITIALISATION start point on the certified block: every free Var reset to its
                  build-time `initialize=` rule (Var._rule_init, None where the Var has none), then
                  `configure_shared_ess_operational_state` per shared ESS with the block's own capacity
                  Params (what the builders do after build: it sets shared_es_soc = 0.5 E), then production's
                  `_clear_multiplier_suffixes` (a freshly built block has empty suffixes); from_warm_start=False.
                  Fixed Vars are untouched; bounds / fixed flags / active flags / Params are asserted
                  unchanged by the reset. NOT reproduced (needs an ADMM initialisation): the initialisation
                  solve's own problem (no AL terms, row 18 inactive, initial-consensus ADN pc/qc); here the
                  certified cycle-112 problem is solved from the initialisation point.
  (iii) warm + diagnostics -- as (i) from an untouched clone, with the case-iii options above.
Order: TSO i, ii, iii; DSO i, ii, iii.

GRADIENT-SCALING FACTORS. IPOPT 3.14.18 prints the per-row scaling vectors only at print level J_VECTOR (8)
(IpNLPScaling.cpp), which also prints every iterate vector; at the production file_print_level 6 it prints
the objective factor ('Scaling parameter for objective function = %e', 'objective scaling factor = %g') and
whether c / d scaling was applied. The per-row factors are therefore COMPUTED, method label
'replica_of_IpGradientScaling_3.14.18': the NL file Pyomo wrote for that very solve is captured (hard link,
time recorded separately) and evaluated at IPOPT's starting point (AmplTNLP::get_starting_point: x0 from the
NL x segment, else 0 projected onto the bounds), with fixed (lb == ub) columns removed
(fixed_variable_treatment make_parameter): df = 100 / max|grad f| if that max > 100 else 1, floored at 1e-8;
per constraint class (c = equality rows g_l == g_u, d = the rest) row factors min(1, 100 / max_j |J_ij|) if
the class's largest row max exceeds 100, else no scaling (all 1), floored at 1e-8. VALIDATED per solve
against IPOPT's own log: the printed objective factor, the c/d 'scaling provided' lines, the variable and
constraint counts and the Jacobian nonzero counts. Span = log10(max / min) over {df} U {dc_i} U {dd_i}.

MEASURED PER SOLVE (capture paths asserted BEFORE the first solve, `capture_checklist`): Pyomo NL write
(NLWriter.__call__ wall), presolve total, IPOPT process wall (_execute_command), IPOPT OverallAlgorithm /
function evaluations / linear solver (timing statistics; linear solver := LinearSystemSymbolicFactorization
+ LinearSystemFactorization + LinearSystemBackSolve, wall), result load := sol read (_postsolve) +
ModelSolutions.load_from; iterations, restoration iterations and phases, mu_final (production's own
`parse_ipopt_attempt_segment` on this solve's log), termination; variables at bounds (|x - b| <= 1e-6 max(1,|b|),
and 1e-8); peak RSS: resource.getrusage (self, children) and a psutil monitor of this process and its
IPOPT child sampled every 25 ms while the solve runs. Full IPOPT output: the output_file log (file level 6)
and the process stdout+stderr (Pyomo's TeeStream merges both) written to <case>_stdout_stderr.log.

PART A OUTPUT (write-once; refuses if the directory exists): data/SRP1/Results/P516TPERF/local/
    tperf_local.json, logs/ (6 IPOPT output_file logs + 6 stdout/stderr logs), nl/ (6 NL files),
    manifest_sha256.json (--manifest, after the run; it also hashes the two launch logs)
LAUNCH (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P516TPERF && set -o noclobber && \\
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_profile.py \\
        > data/SRP1/Results/P516TPERF/local_launch_stdout.log 2> data/SRP1/Results/P516TPERF/local_launch_stderr.log
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_profile.py --manifest
Zero-solve dry run (blocking guard; run_smopf is reached and blocked; NL files are written by model.write for
the replica self-test; writes under the given scratch directory, which must not exist, plus the lock file):
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_tperf_profile.py --dry-run /tmp/<new dir>

PART B (Planner rulings: unit cell; derive_case('paper', {}) of the committed case file; exactly 6 declared solves;
nothing else running). See `run_studio_mode` and `studio_build_blocks`. ONE paper-scale TSO block (2025 Spring) and
ONE DSO block (node 7, 2025 Spring) plus the ESSO are built through the s44 zero-solve build path (all `.optimize`
intercepted; a blocking guard armed over the whole build), not the whole instance.
  --studio      the Studio only (refuses on hw.model Mac17,3, on < 10 GiB available, on any modified tracked file,
                and on any other harness-python / IPOPT process). Output data/SRP1/Results/P516TPERF/studio/.
                Cases per block: (ii) cold from the built point, then (i) warm and (iii) warm+diag, each from its own
                clone of the post-(ii) block. Launch (attached, both streams), then the manifest:
      /path/to/opf_env_py311/bin/python -u p516_tperf_profile.py --studio \\
          > data/SRP1/Results/P516TPERF/studio_launch_stdout.log 2> data/SRP1/Results/P516TPERF/studio_launch_stderr.log
      /path/to/opf_env_py311/bin/python -u p516_tperf_profile.py --manifest --studio-manifest
  --studio-dry  this machine, ZERO solves (blocking guard for the whole run): the same build and case preparation,
                the same capture checklist, then run_smopf per case blocked at OptSolver.solve (option dicts as
                production assembles them, compared with `intended_passed_options`), one NL write per block (not a
                solve). Output data/SRP1/Results/P516TPERF/studio_dry/; manifest: --manifest --manifest-root studio_dry
  --restriction-equivalence  ZERO solves: SRP1-scale full build vs the one-block build, NL/row/col hashes of the
                compared blocks. Output data/SRP1/Results/P516TPERF/studio_dry_equiv/.
Concurrency: an exclusive flock on data/SRP1/Results/P516TPERF/.p516_tperf.lock plus a process scan; the
harness refuses to run while another copy holds it.
"""
import os
import sys

# Thread caps exactly as the production campaign child (launch.json `thread_caps_in_child_env`; MA97 is
# OpenMP-threaded, so this matters for the linear-solver time). Set before numpy / Pyomo load; the IPOPT
# subprocess inherits os.environ.
THREAD_CAP_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                  'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
THREAD_ENV_BEFORE = {k: os.environ.get(k) for k in THREAD_CAP_ENV}
os.environ.update(THREAD_CAP_ENV)

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import argparse  # noqa: E402


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--dry-run', metavar='SCRATCH_DIR', default=None)
    g.add_argument('--manifest', action='store_true')
    g.add_argument('--studio', action='store_true')
    g.add_argument('--studio-dry', action='store_true',
                   help='Part B rehearsal on this machine: one-block paper build + option assembly, ZERO solves')
    g.add_argument('--restriction-equivalence', action='store_true',
                   help='ZERO solves: SRP1-scale full vs one-block build, NL hashes of the compared blocks')
    ap.add_argument('--studio-manifest', action='store_true', help='with --manifest: hash the studio root')
    ap.add_argument('--manifest-root', choices=('local', 'studio', 'studio_dry', 'studio_dry_equiv'), default=None,
                    help='with --manifest: the root to hash (default local; --studio-manifest = studio)')
    return ap.parse_args(argv)


ARGS = _parse_args()

from p513_solve_profile_guard import SolveProfileGuard, SolverInvocationBlocked  # noqa: E402

PERMITTED = (('network.py', '_run_smopf_solver_attempt'),)
DECLARED_SOLVES = 6
if ARGS.manifest:
    _GUARD = None
elif ARGS.dry_run:
    _GUARD = SolveProfileGuard((), label='P5.16 T-perf DRY RUN (zero solves)').install()
elif ARGS.studio_dry or ARGS.restriction_equivalence:
    _GUARD = SolveProfileGuard((), label='P5.16 T-perf Part B ZERO-SOLVE mode (blocking)').install()
else:
    _GUARD = SolveProfileGuard(PERMITTED, label=f'P5.16 T-perf ({DECLARED_SOLVES} declared solves)').install()

import copy  # noqa: E402
import fcntl  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import psutil  # noqa: E402

THIS = os.path.abspath(__file__)
STAGE = 'P5.16 T-perf Part A -- solve-time profile on the certified SRP1 unit-cell blocks'
AUTHORITY = ['P5_16_TPERF_SPEC_v1.md (author-authorized 2026-10-31)', 'TASKS.md T-perf line']
PY = '/opt/anaconda3/envs/opf_env_py311/bin/python'
TPERF_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P516TPERF')
LOCAL_OUT_REL = os.path.join(TPERF_ROOT_REL, 'local')
STUDIO_OUT_REL = os.path.join(TPERF_ROOT_REL, 'studio')
LOCK_REL = os.path.join(TPERF_ROOT_REL, '.p516_tperf.lock')
LAUNCH_LOGS_REL = {'local': [os.path.join(TPERF_ROOT_REL, 'local_launch_stdout.log'),
                             os.path.join(TPERF_ROOT_REL, 'local_launch_stderr.log')],
                   'studio': [os.path.join(TPERF_ROOT_REL, 'studio_launch_stdout.log'),
                              os.path.join(TPERF_ROOT_REL, 'studio_launch_stderr.log')]}
D0_FORBIDDEN = (os.path.join('data', 'SRP1', 'Results', 'P516D0'),)

CAMPAIGN_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
EVAL_REL = os.path.join(CAMPAIGN_REL, 'evals', 'ca8927e75d628bd1_n7_4h_e1')
INPUTS = {
    'certified_models': (os.path.join(EVAL_REL, 'certified_models.pkl'),
                         'fe06ca485003b4fc9c0fd9788279bd287dd813d659179b311885e8ab03efda04'),
    'esso_models': (os.path.join(EVAL_REL, 'esso_models_s39_D.pkl'),
                    '95f7b38a6ec21833f7b5521ec8332d8022ef6801b4095d1410225438b2cedfc9'),
    'production_solve_records': (os.path.join(EVAL_REL, 'network_ipopt_solve_records.jsonl'), None),
    'post_certification': (os.path.join(EVAL_REL, 'post_certification.json'), None),
}
MANIFESTS = (os.path.join(CAMPAIGN_REL, 'campaign_manifest_sha256.json'),
             os.path.join(EVAL_REL, 'child_manifest_sha256.json'))
INSTANCE = {
    'campaign': 's53_w86_tail_recert', 'campaign_spec_sha256_prefix': 'ddd6cd44',
    'cell_label': 'n7_4h_e1',
    'eval_key': 'ca8927e75d628bd188cf74f9d81436cbb3e0be53f64e02d9876ebb271a21951d',
    'candidate_key': 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a',
    'candidate_canonical': {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}},
    'certification_cycle': 112,
}
YEAR, DAY, DSO_NODE = 2025, 'Spring', 7
BLOCKS = ('tso', 'dso')
CASES = ('i_warm', 'ii_cold', 'iii_warm_diag')
CASE_IS_WARM = {'i_warm': True, 'ii_cold': False, 'iii_warm_diag': True}
CASE_III_OPTIONS = {'print_info_string': 'yes', 'print_level': 5, 'nlp_scaling_method': 'gradient-based'}
ALL_CASE_OPTIONS = {'print_timing_statistics': 'yes'}
AT_BOUND_RTOL = (1e-6, 1e-8)
RSS_SAMPLE_S = 0.025
IPOPT_SCALING = {'nlp_scaling_max_gradient': 100.0, 'nlp_scaling_min_value': 1e-8,
                 'nlp_scaling_obj_target_gradient': 0.0, 'nlp_scaling_constr_target_gradient': 0.0,
                 'obj_scaling_factor': 1.0, 'nlp_lower_bound_inf': -1e19, 'nlp_upper_bound_inf': 1e19,
                 'source': 'IPOPT 3.14.18 defaults (IpGradientScaling.cpp, IpNLPScaling.cpp, AmplTNLP.cpp); '
                           'none of these keys is set by the case-file options (asserted per solve)'}
LINEAR_SOLVER_TIMERS = ('LinearSystemSymbolicFactorization', 'LinearSystemFactorization', 'LinearSystemBackSolve')
PRODUCTION_FILES = ('network.py', 'network_data.py', 'shared_resources_planning.py', 'model_construction_helpers.py',
                    'helper_functions.py', 'network_parameters.py', 'p513_solve_profile_guard.py')
T0 = time.time()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [T-perf +{time.time() - T0:8.2f}s] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _rss():
    return psutil.Process().memory_info().rss


def _rusage():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'self_ru_maxrss': s.ru_maxrss, 'children_ru_maxrss': c.ru_maxrss, 'units': 'bytes (macOS ru_maxrss)'}


def _write_json(path, obj):
    import gate_result_io as GRIO
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    try:
        GRIO.check(obj, default=GRIO.json_default_item)
    except Exception:
        # Nothing is lost: the record is pickled beside the refused path, then the refusal is raised.
        with open(path + '.refused_record.pkl', 'xb') as handle:
            pickle.dump(obj, handle)
        raise
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, default=GRIO.json_default_item, indent=1, sort_keys=False)


# ======================================================================================================================
#  concurrency refusal
# ======================================================================================================================
class Lock:
    def __init__(self, path):
        self.path = path
        self.handle = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, 'a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f'refusing to run: another p516_tperf_profile.py holds {self.path}')
        others = []
        me = {os.getpid(), os.getppid()}
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if proc.info['pid'] not in me and 'p516_tperf_profile.py' in cmd and 'python' in cmd:
                others.append({'pid': proc.info['pid'], 'cmdline': cmd[:300]})
        if others:
            raise RuntimeError(f'refusing to run: other p516_tperf_profile.py processes {others}')
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f'{os.getpid()} {_utc()}\n')
        self.handle.flush()
        return {'path': self.path, 'pid': os.getpid(), 'other_copies_found': others}

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _concurrent_process_snapshot():
    """Other python processes and the load average, recorded beside every timing (machine is shared)."""
    rows = []
    me = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] != me and ('python' in cmd.lower() or 'ipopt' in cmd.lower()):
            rows.append({'pid': proc.info['pid'], 'cmdline': cmd[:200]})
    return {'loadavg_1_5_15': list(os.getloadavg()), 'other_python_or_ipopt_processes': rows}


# ======================================================================================================================
#  inputs
# ======================================================================================================================
def verify_inputs():
    manifests = {rel: json.load(open(os.path.join(REPO, rel))) for rel in MANIFESTS}
    out = {}
    for key, (rel, declared) in INPUTS.items():
        now = _sha(os.path.join(REPO, rel))
        pinned = {m: d.get(rel) for m, d in manifests.items()}
        problems = []
        if declared is not None and declared != now:
            problems.append(f'declared {declared}')
        for m, value in pinned.items():
            if value != now:
                problems.append(f'{m}: {value}')
        if problems:
            raise RuntimeError(f'input hash mismatch for {rel}: now {now}; {problems}')
        out[key] = {'path': rel, 'sha256': now, 'declared_in_task': declared, 'manifest_pins': pinned,
                    'size_bytes': os.path.getsize(os.path.join(REPO, rel))}
    return out


def _build_time_objects(block):
    """The `network` / `params` the block was BUILT with (pickled bound-rule partial of `slack_v_sqr_down`,
    network.py: bounds=partial(voltage_slack_down_bounds, network=network, params=params)); p515_s48 precedent."""
    import functools
    import model_construction_helpers as mch
    fcn = block.slack_v_sqr_down._rule_bounds._initializer._fcn
    if not (isinstance(fcn, functools.partial) and fcn.func is mch.voltage_slack_down_bounds):
        raise RuntimeError(f'unexpected bound rule on {block.name}: {fcn!r}')
    return fcn.keywords['network'], fcn.keywords['params']


def pickle_structure(payload):
    struct = {'top_keys': sorted(payload.keys())}
    struct['tso'] = {str(y): sorted(d.keys()) for y, d in payload['tso'].items()}
    struct['dso'] = {str(n): {str(y): sorted(d.keys()) for y, d in yd.items()} for n, yd in payload['dso'].items()}
    struct['model_type'] = type(payload['tso'][YEAR][DAY]).__name__
    struct['n_tso_blocks'] = sum(len(d) for d in payload['tso'].values())
    struct['n_dso_blocks'] = sum(len(d) for yd in payload['dso'].values() for d in yd.values())
    return struct


def block_size(model, pe):
    nv = nf = 0
    for v in model.component_data_objects(pe.Var, descend_into=True):
        nv += 1
        nf += bool(v.fixed)
    n_eq = n_ineq = 0
    for c in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
        if c.equality:
            n_eq += 1
        else:
            n_ineq += 1
    objs = [o.name for o in model.component_data_objects(pe.Objective, active=True)]
    return {'pyomo_vars': nv, 'pyomo_vars_fixed': nf, 'pyomo_vars_free': nv - nf,
            'pyomo_active_constraints': n_eq + n_ineq, 'pyomo_active_equalities': n_eq,
            'pyomo_active_inequalities': n_ineq, 'active_objectives': objs,
            'scenarios_market': len(model.scenarios_market) if hasattr(model, 'scenarios_market') else None,
            'scenarios_operation': len(model.scenarios_operation) if hasattr(model, 'scenarios_operation') else None,
            'periods': len(model.periods) if hasattr(model, 'periods') else None}


# ======================================================================================================================
#  cold start = production initialisation point
# ======================================================================================================================
def _norm(x):
    """A comparable form of a raw bound / Param value (a Pyomo expression object would turn `==` into a relation)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    try:
        import pyomo.environ as pe
        return ('expr', str(x), pe.value(x))
    except Exception:  # noqa: BLE001
        return ('repr', repr(x))


def _structure_signature(state):
    """Everything in a capture_block_mutable_state() EXCEPT Var values and suffixes."""
    return {'params': {n: {i: _norm(v) for i, v in d.items()} for n, d in state['params'].items()},
            'vars': {n: {i: (_norm(e[1]), _norm(e[2]), bool(e[3])) for i, e in d.items()}
                     for n, d in state['vars'].items()},
            'constraints': state['constraints'], 'objectives': state['objectives']}


def reset_to_initialisation_point(model, block, network_module, mch, pe):
    before = network_module.capture_block_mutable_state(model)
    n_reset = n_none = n_skipped_fixed = 0
    n_reference_skipped = 0
    for comp in model.component_objects(pe.Var, active=None, descend_into=True):
        if comp.is_reference():
            n_reference_skipped += 1
            continue
        init = comp._rule_init
        parent = comp.parent_block()
        for index in comp:
            vd = comp[index]
            if vd.fixed:
                n_skipped_fixed += 1
                continue
            value = None if init is None else init(parent, index)
            if value is None:
                n_none += 1
            vd.set_value(value, skip_validation=True)
            n_reset += 1
    shared = list(model.shared_energy_storages)
    if block == 'dso' and len(shared) != 1:
        raise RuntimeError(f'DSO block has {len(shared)} shared ESS (expected 1)')
    soc_set = {}
    for e in shared:
        mch.configure_shared_ess_operational_state(model, e, pe.value(model.shared_es_s_rated_fixed[e]),
                                                   pe.value(model.shared_es_e_rated_fixed[e]))
        soc_set[str(e)] = sorted({round(pe.value(model.shared_es_soc[i]), 15) for i in model.shared_es_soc
                                  if (i[0] if isinstance(i, tuple) else i) == e
                                  and model.shared_es_soc[i].value is not None})[:4]
    network_module._clear_multiplier_suffixes(model)
    after = network_module.capture_block_mutable_state(model)
    unchanged = _structure_signature(before) == _structure_signature(after)
    if not unchanged:
        raise RuntimeError('cold reset changed bounds / fixed flags / active flags / Params -- refusing')
    return {'free_vars_reset': n_reset, 'reset_to_None_no_initializer': n_none,
            'reference_components_skipped': n_reference_skipped,
            'fixed_vars_untouched': n_skipped_fixed, 'shared_ess_configured': [str(e) for e in shared],
            'shared_es_soc_values_after_configure_sample': soc_set,
            'suffixes_after_clear': {s: len(getattr(model, s)) for s in ('dual', 'ipopt_zL_in', 'ipopt_zU_in')},
            'structure_unchanged_bounds_fixed_active_params': unchanged,
            'method': ('Var._rule_init(parent, index) for every free VarData (None where no initializer); then '
                       'model_construction_helpers.configure_shared_ess_operational_state per shared ESS with the '
                       "block's own shared_es_s/e_rated_fixed; then network._clear_multiplier_suffixes")}


# ======================================================================================================================
#  instrumentation (timing / log / NL capture) -- wraps Pyomo internals for THIS process only
# ======================================================================================================================
class Instrument:
    def __init__(self, nl_dir, dry):
        self.nl_dir = nl_dir
        self.dry = dry
        self.case = None
        self.events = None
        self.cap_blocks = 0
        self.installed = {}

    def begin(self, case_key):
        self.case = case_key
        self.events = {'solve_calls': 0, 'nl_write': [], 'presolve': [], 'execute': [], 'postsolve': [],
                       'load_from': [], 'solve_wall': [], 'nl_capture': [], 'solver_options_passed': None,
                       'stdout_stderr_text': None, 'returncode': None, 'cap_blocked': 0}

    def end(self):
        ev = self.events
        self.case = None
        self.events = None
        return ev

    def install(self):
        from pyomo.opt.base.solvers import OptSolver
        from pyomo.opt.solver.shellcmd import SystemCallSolver
        from pyomo.repn.plugins.nl_writer import NLWriter
        from pyomo.core.base.PyomoModel import ModelSolutions
        inst = self
        orig = {'solve': OptSolver.solve, 'presolve': SystemCallSolver._presolve,
                'execute': SystemCallSolver._execute_command, 'postsolve': SystemCallSolver._postsolve,
                'nl_call': NLWriter.__call__, 'load_from': ModelSolutions.load_from}
        self.orig = orig

        def solve(solver_self, *args, **kwargs):
            if inst.events is None:
                raise SolverInvocationBlocked('T-perf: solve outside a declared case')
            if inst.events['solve_calls'] >= 1:
                inst.events['cap_blocked'] += 1
                inst.cap_blocks += 1
                raise SolverInvocationBlocked('T-perf: second solve inside one declared case (production '
                                              'recovery after a failed primary) -- blocked by the per-case cap')
            inst.events['solve_calls'] += 1
            inst.events['solver_options_passed'] = {str(k): v for k, v in dict(solver_self.options).items()}
            t = time.perf_counter()
            try:
                return orig['solve'](solver_self, *args, **kwargs)
            finally:
                inst.events['solve_wall'].append(time.perf_counter() - t)

        def presolve(solver_self, *args, **kwargs):
            t = time.perf_counter()
            try:
                return orig['presolve'](solver_self, *args, **kwargs)
            finally:
                if inst.events is not None:
                    inst.events['presolve'].append(time.perf_counter() - t)

        def execute(solver_self, command):
            t = time.perf_counter()
            out = orig['execute'](solver_self, command)
            if inst.events is not None:
                inst.events['execute'].append(time.perf_counter() - t)
                inst.events['returncode'] = out[0]
                inst.events['stdout_stderr_text'] = out[1]
                inst.events['command'] = list(command.cmd) if hasattr(command, 'cmd') else None
            return out

        def postsolve(solver_self, *args, **kwargs):
            t = time.perf_counter()
            try:
                return orig['postsolve'](solver_self, *args, **kwargs)
            finally:
                if inst.events is not None:
                    inst.events['postsolve'].append(time.perf_counter() - t)

        def nl_call(writer_self, model, filename, solver_capability, io_options):
            t = time.perf_counter()
            out = orig['nl_call'](writer_self, model, filename, solver_capability, io_options)
            dt = time.perf_counter() - t
            if inst.events is not None:
                inst.events['nl_write'].append(dt)
                t2 = time.perf_counter()
                target = os.path.join(inst.nl_dir, f'{inst.case}.nl')
                try:
                    os.link(filename, target)
                    how = 'hard_link'
                except OSError:
                    shutil.copyfile(filename, target)
                    how = 'copy'
                inst.events['nl_capture'].append({'path': target, 'method': how, 'source_tmp': filename,
                                                  'capture_s': time.perf_counter() - t2,
                                                  'io_options': {str(k): str(v) for k, v in dict(io_options).items()}})
            return out

        def load_from(sol_self, *args, **kwargs):
            t = time.perf_counter()
            try:
                return orig['load_from'](sol_self, *args, **kwargs)
            finally:
                if inst.events is not None:
                    inst.events['load_from'].append(time.perf_counter() - t)

        OptSolver.solve = solve
        SystemCallSolver._presolve = presolve
        SystemCallSolver._execute_command = execute
        SystemCallSolver._postsolve = postsolve
        NLWriter.__call__ = nl_call
        ModelSolutions.load_from = load_from
        self.installed = {'OptSolver.solve': solve, 'SystemCallSolver._presolve': presolve,
                          'SystemCallSolver._execute_command': execute, 'SystemCallSolver._postsolve': postsolve,
                          'NLWriter.__call__': nl_call, 'ModelSolutions.load_from': load_from}
        return self

    def is_wired(self):
        from pyomo.opt.base.solvers import OptSolver
        from pyomo.opt.solver.shellcmd import SystemCallSolver
        from pyomo.repn.plugins.nl_writer import NLWriter
        from pyomo.core.base.PyomoModel import ModelSolutions
        from pyomo.solvers.plugins.solvers.IPOPT import IPOPT
        live = {'OptSolver.solve': IPOPT.solve, 'SystemCallSolver._presolve': IPOPT._presolve,
                'SystemCallSolver._execute_command': IPOPT._execute_command,
                'SystemCallSolver._postsolve': IPOPT._postsolve, 'NLWriter.__call__': NLWriter.__call__,
                'ModelSolutions.load_from': ModelSolutions.load_from}
        _ = (OptSolver, SystemCallSolver)
        return {k: live[k] is v for k, v in self.installed.items()}


class RssMonitor(threading.Thread):
    def __init__(self, interval=RSS_SAMPLE_S):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_self = 0
        self.peak_child = 0
        self.peak_sum = 0
        self.n = 0
        self.child_pids = set()

    def run(self):
        me = psutil.Process()
        while not self.stop_event.is_set():
            try:
                s = me.memory_info().rss
                c = 0
                for ch in me.children(recursive=True):
                    try:
                        c += ch.memory_info().rss
                        self.child_pids.add(ch.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_self = max(self.peak_self, s)
                self.peak_child = max(self.peak_child, c)
                self.peak_sum = max(self.peak_sum, s + c)
                self.n += 1
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()
        return {'peak_self_rss': self.peak_self, 'peak_child_rss': self.peak_child, 'peak_self_plus_child_rss':
                self.peak_sum, 'n_samples': self.n, 'interval_s': self.interval, 'child_pids_seen': sorted(self.child_pids)}


# ======================================================================================================================
#  IPOPT log parsing
# ======================================================================================================================
_TIMING_RE = re.compile(r'^(\s*)([A-Za-z0-9 ]+?)\.*:\s+([-0-9.eE+]+) \(sys:\s+([-0-9.eE+]+) wall:\s+([-0-9.eE+]+)\)')
_ITER_RE = re.compile(r'^\s*(\d+)(r?)\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+')


def parse_ipopt_log(text):
    out = {'problem': {}, 'timing_statistics': {}, 'summary': {}}
    for m in re.finditer(r'Number of nonzeros in ([A-Za-z ]+?)\.*:\s+(\d+)', text):
        out['problem'][f'nnz {m.group(1).strip()}'] = int(m.group(2))
    for label in ('Total number of variables', 'variables with only lower bounds',
                  'variables with lower and upper bounds', 'variables with only upper bounds',
                  'Total number of equality constraints', 'Total number of inequality constraints',
                  'inequality constraints with only lower bounds', 'inequality constraints with lower and upper bounds',
                  'inequality constraints with only upper bounds'):
        m = re.search(re.escape(label) + r'\.*:\s+(\d+)', text)
        out['problem'][label] = int(m.group(1)) if m else None
    in_timing = False
    for line in text.splitlines():
        if line.startswith('Timing Statistics:'):
            in_timing = True
            continue
        if in_timing:
            m = _TIMING_RE.match(line)
            if m:
                out['timing_statistics'][m.group(2).strip()] = {
                    'cpu': float(m.group(3)), 'sys': float(m.group(4)), 'wall': float(m.group(5)),
                    'depth': len(m.group(1))}
    iters = []
    for line in text.splitlines():
        m = _ITER_RE.match(line)
        if m:
            iters.append((int(m.group(1)), m.group(2) == 'r'))
    # The iteration table is printed once per iteration at print_level >= 5; the file log at level 6 may
    # repeat the header but each iteration line once. Deduplicate by (iter, r) order of appearance.
    restoration = [k for k, r in iters if r]
    phases = 0
    prev = False
    for _, r in iters:
        if r and not prev:
            phases += 1
        prev = r
    out['summary']['iteration_lines'] = len(iters)
    out['summary']['restoration_iteration_lines'] = len(restoration)
    out['summary']['restoration_phases'] = phases
    m = re.search(r'Number of Iterations\.+:\s+(\d+)', text)
    out['summary']['iterations'] = int(m.group(1)) if m else None
    exits = re.findall(r'EXIT: (.*)', text)
    out['summary']['exit'] = exits[-1].strip() if exits else None
    for key, pat in (('total_s_ipopt_wo_fevals', r'Total seconds in IPOPT \(w/o function evaluations\)\s+=\s+([-0-9.eE+]+)'),
                     ('total_s_nlp_fevals', r'Total seconds in NLP function evaluations\s+=\s+([-0-9.eE+]+)'),
                     ('total_s_ipopt', r'Total seconds in IPOPT\s+=\s+([-0-9.eE+]+)')):
        m = re.search(pat, text)
        out['summary'][key] = float(m.group(1)) if m else None
    m = re.search(r'Scaling parameter for objective function = (\S+)', text)
    out['summary']['scaling_parameter_for_objective_function'] = float(m.group(1)) if m else None
    m = re.search(r'objective scaling factor = (\S+)', text)
    out['summary']['objective_scaling_factor_printed'] = float(m.group(1)) if m else None
    for cls in ('c', 'd', 'x'):
        if re.search(rf'^No {cls} scaling provided', text, re.M):
            out['summary'][f'{cls}_scaling_provided'] = False
        elif re.search(rf'^{cls} scaling provided', text, re.M):
            out['summary'][f'{cls}_scaling_provided'] = True
        else:
            out['summary'][f'{cls}_scaling_provided'] = None
    m = re.search(r'This is Ipopt version (\S+), running with linear solver (\S+)', text)
    out['summary']['ipopt_version'] = m.group(1).rstrip(',') if m else None
    out['summary']['linear_solver'] = m.group(2).rstrip('.') if m else None
    ts = out['timing_statistics']
    if all(k in ts for k in LINEAR_SOLVER_TIMERS):
        out['summary']['linear_solver_wall_s'] = sum(ts[k]['wall'] for k in LINEAR_SOLVER_TIMERS)
        out['summary']['linear_solver_cpu_s'] = sum(ts[k]['cpu'] for k in LINEAR_SOLVER_TIMERS)
    else:
        out['summary']['linear_solver_wall_s'] = None
        out['summary']['linear_solver_cpu_s'] = None
    out['summary']['overall_algorithm_wall_s'] = ts.get('OverallAlgorithm', {}).get('wall')
    out['summary']['function_evaluations_wall_s'] = ts.get('Function Evaluations', {}).get('wall')
    out['summary']['pd_system_solver_total_wall_s'] = ts.get('PDSystemSolverTotal', {}).get('wall')
    return out


# ======================================================================================================================
#  NL file: evaluation at IPOPT's starting point -> IpGradientScaling replica
# ======================================================================================================================
_UNARY = {13: (math.floor, lambda x: 0.0), 14: (math.ceil, lambda x: 0.0), 15: (abs, lambda x: 1.0 if x >= 0 else -1.0),
          16: (lambda x: -x, lambda x: -1.0), 37: (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
          38: (math.tan, lambda x: 1.0 / math.cos(x) ** 2), 39: (math.sqrt, lambda x: 0.5 / math.sqrt(x)),
          40: (math.sinh, math.cosh), 41: (math.sin, math.cos), 42: (math.log10, lambda x: 1.0 / (x * math.log(10.0))),
          43: (math.log, lambda x: 1.0 / x), 44: (math.exp, math.exp), 45: (math.cosh, math.sinh),
          46: (math.cos, lambda x: -math.sin(x)), 49: (math.atan, lambda x: 1.0 / (1.0 + x * x)),
          51: (math.asin, lambda x: 1.0 / math.sqrt(1.0 - x * x)), 53: (math.acos, lambda x: -1.0 / math.sqrt(1.0 - x * x)),
          50: (math.asinh, lambda x: 1.0 / math.sqrt(x * x + 1.0)), 52: (math.acosh, lambda x: 1.0 / math.sqrt(x * x - 1.0)),
          47: (math.atanh, lambda x: 1.0 / (1.0 - x * x))}
_BINARY = {0, 1, 2, 3, 5}


class NLProblem:
    """Minimal reader of a TEXT .nl file as Pyomo's NL writer v2 emits it (g header; segments S, V, C, O, d, x,
    r, b, k, J, G). Expressions: prefix opcodes o0 o1 o2 o3 o5 o54 and the unary functions above, n<const>,
    v<idx> (idx >= n_vars -> defined variable). Anything else raises (the replica is then reported as not
    computed, with the reason)."""

    def __init__(self, path):
        with open(path) as handle:
            raw = handle.read().split('\n')
        self.lines = [ln.split('#', 1)[0].strip() for ln in raw]
        if not self.lines[0].startswith('g'):
            raise ValueError(f'not a text NL file (header {self.lines[0][:10]!r})')
        h = [ln.split() for ln in self.lines[1:10]]
        self.n, self.m, self.n_obj = int(h[0][0]), int(h[0][1]), int(h[0][2])
        self.n_eqns_header = int(h[0][4])
        self.nnz_jac_header, self.nnz_grad_header = int(h[6][0]), int(h[6][1])
        self.pos = 10
        self.C, self.O, self.V, self.J, self.G = {}, {}, {}, {}, {}
        self.x0, self.r, self.b = {}, [None] * self.m, [None] * self.n
        self.obj_sense = {}
        self._parse()

    def _next(self):
        while self.pos < len(self.lines) and self.lines[self.pos] == '':
            self.pos += 1
        ln = self.lines[self.pos]
        self.pos += 1
        return ln

    def _expr(self):
        tok = self._next()
        c = tok[0]
        if c == 'n':
            return ('n', float(tok[1:]))
        if c == 'v':
            return ('v', int(tok[1:]))
        if c == 'o':
            code = int(tok[1:])
            if code == 54:
                k = int(self._next())
                return ('sum', [self._expr() for _ in range(k)])
            if code in _BINARY:
                a = self._expr()
                b = self._expr()
                return ('b', code, a, b)
            if code in _UNARY:
                return ('u', code, self._expr())
            raise ValueError(f'unsupported NL opcode o{code}')
        raise ValueError(f'unsupported NL token {tok!r}')

    def _parse(self):
        while True:
            while self.pos < len(self.lines) and self.lines[self.pos] == '':
                self.pos += 1
            if self.pos >= len(self.lines):
                break
            ln = self.lines[self.pos]
            self.pos += 1
            tag, rest = ln[0], ln[1:].split()
            if tag == 'S':
                for _ in range(int(rest[1])):
                    self._next()
            elif tag == 'V':
                i, nlin = int(rest[0]), int(rest[1])
                lin = {}
                for _ in range(nlin):
                    j, coef = self._next().split()
                    lin[int(j)] = lin.get(int(j), 0.0) + float(coef)
                self.V[i] = (lin, self._expr())
            elif tag == 'C':
                self.C[int(rest[0])] = self._expr()
            elif tag == 'O':
                self.obj_sense[int(rest[0])] = int(rest[1])
                self.O[int(rest[0])] = self._expr()
            elif tag == 'd':
                for _ in range(int(rest[0])):
                    self._next()
            elif tag == 'x':
                for _ in range(int(rest[0])):
                    j, val = self._next().split()
                    self.x0[int(j)] = float(val)
            elif tag == 'r':
                for i in range(self.m):
                    self.r[i] = [float(t) for t in self._next().split()]
            elif tag == 'b':
                for i in range(self.n):
                    self.b[i] = [float(t) for t in self._next().split()]
            elif tag == 'k':
                for _ in range(int(rest[0])):
                    self._next()
            elif tag == 'J':
                i, k = int(rest[0]), int(rest[1])
                self.J[i] = {}
                for _ in range(k):
                    j, coef = self._next().split()
                    self.J[i][int(j)] = float(coef)
            elif tag == 'G':
                i, k = int(rest[0]), int(rest[1])
                self.G[i] = {}
                for _ in range(k):
                    j, coef = self._next().split()
                    self.G[i][int(j)] = float(coef)
            else:
                raise ValueError(f'unsupported NL segment {ln[:20]!r}')

    def bounds(self, j):
        t = self.b[j]
        kind = int(t[0])
        if kind == 0:
            return t[1], t[2]
        if kind == 1:
            return -math.inf, t[1]
        if kind == 2:
            return t[1], math.inf
        if kind == 3:
            return -math.inf, math.inf
        if kind == 4:
            return t[1], t[1]
        raise ValueError(f'variable bound type {kind}')

    def row_bounds(self, i):
        t = self.r[i]
        kind = int(t[0])
        if kind == 0:
            return t[1], t[2]
        if kind == 1:
            return -math.inf, t[1]
        if kind == 2:
            return t[1], math.inf
        if kind == 3:
            return -math.inf, math.inf
        if kind == 4:
            return t[1], t[1]
        raise ValueError(f'constraint type {kind} (complementarity) unsupported')

    def starting_point(self):
        x = []
        for j in range(self.n):
            if j in self.x0:
                x.append(self.x0[j])
            else:
                lo, up = self.bounds(j)
                x.append(max(lo, min(up, 0.0)))
        return x


def _ad(node, x, n, dcache, nl):
    """Forward-mode sparse AD: returns (value, {var_index: derivative})."""
    t = node[0]
    if t == 'n':
        return node[1], {}
    if t == 'v':
        i = node[1]
        if i < n:
            return x[i], {i: 1.0}
        return _defined(i, x, n, dcache, nl)
    if t == 'sum':
        val = 0.0
        g = {}
        for ch in node[1]:
            v, gc_ = _ad(ch, x, n, dcache, nl)
            val += v
            for k, d in gc_.items():
                g[k] = g.get(k, 0.0) + d
        return val, g
    if t == 'u':
        f, df = _UNARY[node[1]]
        v, g = _ad(node[2], x, n, dcache, nl)
        s = df(v)
        return f(v), {k: s * d for k, d in g.items()}
    code, a, b = node[1], node[2], node[3]
    va, ga = _ad(a, x, n, dcache, nl)
    vb, gb = _ad(b, x, n, dcache, nl)
    if code == 0:
        g = dict(ga)
        for k, d in gb.items():
            g[k] = g.get(k, 0.0) + d
        return va + vb, g
    if code == 1:
        g = dict(ga)
        for k, d in gb.items():
            g[k] = g.get(k, 0.0) - d
        return va - vb, g
    if code == 2:
        g = {k: vb * d for k, d in ga.items()}
        for k, d in gb.items():
            g[k] = g.get(k, 0.0) + va * d
        return va * vb, g
    if code == 3:
        g = {k: d / vb for k, d in ga.items()}
        for k, d in gb.items():
            g[k] = g.get(k, 0.0) - va * d / (vb * vb)
        return va / vb, g
    if code == 5:
        val = va ** vb
        g = {}
        if ga:
            s = vb * va ** (vb - 1.0) if vb != 0 else 0.0
            for k, d in ga.items():
                g[k] = s * d
        if gb:
            s = val * math.log(va)
            for k, d in gb.items():
                g[k] = g.get(k, 0.0) + s * d
        return val, g
    raise ValueError(f'opcode {code}')


def _defined(i, x, n, dcache, nl):
    if i in dcache:
        return dcache[i]
    lin, expr = nl.V[i]
    val, g = _ad(expr, x, n, dcache, nl)
    g = dict(g)
    for j, coef in lin.items():
        if j < n:
            val += coef * x[j]
            g[j] = g.get(j, 0.0) + coef
        else:
            vj, gj = _defined(j, x, n, dcache, nl)
            val += coef * vj
            for k, d in gj.items():
                g[k] = g.get(k, 0.0) + coef * d
    dcache[i] = (val, g)
    return dcache[i]


def gradient_scaling_replica(nl_path):
    t0 = time.perf_counter()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    nl = NLProblem(nl_path)
    n = nl.n
    x = nl.starting_point()
    fixed = set()
    for j in range(n):
        lo, up = nl.bounds(j)
        if lo == up:
            fixed.add(j)
    dcache = {}
    sp = IPOPT_SCALING
    mg, mv = sp['nlp_scaling_max_gradient'], sp['nlp_scaling_min_value']
    # objective (single objective expected)
    if nl.n_obj != 1:
        raise ValueError(f'{nl.n_obj} objectives')
    _, gobj = _ad(nl.O[0], x, n, dcache, nl)
    gobj = dict(gobj)
    for j, coef in nl.G.get(0, {}).items():
        gobj[j] = gobj.get(j, 0.0) + coef
    max_grad_f = max((abs(d) for j, d in gobj.items() if j not in fixed), default=0.0)
    df = mg / max_grad_f if max_grad_f > mg else 1.0
    df = max(df, mv) * sp['obj_scaling_factor']
    rows = {'c': [], 'd': []}
    nnz = {'c': 0, 'd': 0}
    for i in range(nl.m):
        _, g = _ad(nl.C[i], x, n, dcache, nl) if i in nl.C else (0.0, {})
        tot = {}
        for j, coef in nl.J.get(i, {}).items():
            tot[j] = coef + g.get(j, 0.0)
        for j, d in g.items():
            if j not in tot:
                raise ValueError(f'row {i}: nonlinear derivative on column {j} absent from its J segment')
        lo, up = nl.row_bounds(i)
        cls = 'c' if lo == up else 'd'
        entries = [abs(d) for j, d in tot.items() if j not in fixed]
        nnz[cls] += len(entries)
        rows[cls].append(max(entries) if entries else 0.0)
    out = {'method': 'replica_of_IpGradientScaling_3.14.18 (NL file evaluated at AmplTNLP starting point)',
           'n_free_vars': n - len(fixed), 'n_nl_vars': n, 'n_fixed_lb_eq_ub_removed': len(fixed),
           'n_eq_rows': len(rows['c']), 'n_ineq_rows': len(rows['d']),
           'nnz_jac_c_free_cols': nnz['c'], 'nnz_jac_d_free_cols': nnz['d'],
           'nl_header': {'n_vars': nl.n, 'n_cons': nl.m, 'n_eqns': nl.n_eqns_header,
                         'nnz_jacobian': nl.nnz_jac_header, 'nnz_obj_gradient': nl.nnz_grad_header},
           'n_defined_variables': len(nl.V), 'x0_from_nl_x_segment': len(nl.x0),
           'max_abs_obj_gradient': max_grad_f, 'df': df}
    factors = [df]
    for cls in ('c', 'd'):
        rm = rows[cls]
        arow_max = max(rm) if rm else 0.0
        provided = bool(rm) and arow_max > mg
        if provided:
            fac = [max(min(1.0, mg / r) if r > 0 else 1.0, mv) for r in rm]
        else:
            fac = [1.0] * len(rm)
        factors.extend(fac)
        nz = [r for r in rm if r > 0]
        out[cls] = {'scaling_provided': provided, 'arow_max': arow_max,
                    'row_max_gradient_min_nonzero': min(nz) if nz else None,
                    'row_max_gradient_span_decades': (math.log10(max(nz) / min(nz)) if nz else None),
                    'rows_scaled_below_1': sum(1 for f in fac if f < 1.0),
                    'factor_min': min(fac) if fac else None, 'factor_max': max(fac) if fac else None,
                    'zero_rows': sum(1 for r in rm if r == 0)}
    out['factor_min_all'] = min(factors)
    out['factor_max_all'] = max(factors)
    out['factor_span_decades_all'] = math.log10(max(factors) / min(factors))
    cf = factors[1:]
    out['constraint_factor_span_decades'] = math.log10(max(cf) / min(cf)) if cf else None
    allrows = [r for r in rows['c'] + rows['d'] if r > 0]
    out['row_max_gradient_span_decades_all'] = math.log10(max(allrows) / min(allrows)) if allrows else None
    out['compute_s'] = time.perf_counter() - t0
    out['span_definition'] = 'log10(max/min) over {df} U {dc_i} U {dd_i}; unscaled rows count as 1'
    return out


def validate_replica(rep, ipopt):
    s, p = ipopt['summary'], ipopt['problem']
    checks = {}
    printed = s.get('scaling_parameter_for_objective_function')
    checks['df_vs_scaling_parameter_line'] = (None if printed is None else
                                              abs(rep['df'] - printed) <= 1e-6 * max(abs(printed), 1e-300))
    printed_g = s.get('objective_scaling_factor_printed')
    checks['df_vs_objective_scaling_factor_line_6sig'] = (None if printed_g is None else
                                                          abs(rep['df'] - printed_g) <= 1e-5 * max(abs(printed_g), 1e-300))
    checks['c_scaling_provided_agrees'] = (None if s.get('c_scaling_provided') is None else
                                           s['c_scaling_provided'] == rep['c']['scaling_provided'])
    checks['d_scaling_provided_agrees'] = (None if s.get('d_scaling_provided') is None else
                                           s['d_scaling_provided'] == rep['d']['scaling_provided'])
    checks['n_vars_agrees'] = p.get('Total number of variables') == rep['n_free_vars']
    checks['n_eq_agrees'] = p.get('Total number of equality constraints') == rep['n_eq_rows']
    checks['n_ineq_agrees'] = p.get('Total number of inequality constraints') == rep['n_ineq_rows']
    checks['nnz_eq_jac_agrees'] = p.get('nnz equality constraint Jacobian') == rep['nnz_jac_c_free_cols']
    checks['nnz_ineq_jac_agrees'] = p.get('nnz inequality constraint Jacobian') == rep['nnz_jac_d_free_cols']
    checks['all_pass'] = all(v is True for v in checks.values())
    return checks


# ======================================================================================================================
#  per-solve measurement
# ======================================================================================================================
def vars_at_bounds(model, pe):
    out = {}
    for rtol in AT_BOUND_RTOL:
        n_lb = n_ub = n_free = 0
        for v in model.component_data_objects(pe.Var, descend_into=True):
            if v.fixed or v.value is None:
                continue
            n_free += 1
            lb, ub = v.lb, v.ub
            at_lb = lb is not None and abs(v.value - lb) <= rtol * max(1.0, abs(lb))
            at_ub = ub is not None and abs(v.value - ub) <= rtol * max(1.0, abs(ub))
            n_lb += at_lb
            n_ub += at_ub and not at_lb
        out[f'rtol_{rtol:g}'] = {'at_lower': n_lb, 'at_upper': n_ub, 'at_any': n_lb + n_ub,
                                 'free_vars_with_value': n_free}
    out['definition'] = '|x - b| <= rtol * max(1, |b|) on the loaded solution, free (unfixed) Vars only'
    return out


def expected_log_path(network, options):
    """Read-only replica of network.py `_create_smopf_solver`'s output_file path construction (p515_s42 precedent;
    run_smopf does not return the path)."""
    configured_path = os.path.join(network.logs_dir, options['output_file'])
    stem, ext = os.path.splitext(configured_path)
    day_label = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(network.day))
    return f'{stem}_{network.year}_{day_label}{ext}'


def run_case(case_key, block, case, model, network, params, inst, network_module, pe, logs_dir, dry):
    rec = {'case_key': case_key, 'block': block, 'case': case, 'from_warm_start': CASE_IS_WARM[case]}
    log_path = expected_log_path(network, params.solver_params.options)
    if os.path.exists(log_path):
        raise RuntimeError(f'refusing: IPOPT log {log_path} exists (append would mix solves)')
    rec['ipopt_output_file'] = os.path.relpath(log_path, REPO)
    rec['concurrent_before'] = _concurrent_process_snapshot()
    rec['rusage_before'] = _rusage()
    rec['rss_before'] = _rss()
    guard_before = dict(_GUARD.counts)
    inst.begin(case_key)
    mon = RssMonitor()
    mon.start()
    t0 = time.perf_counter()
    result = network.run_smopf(model, params, from_warm_start=CASE_IS_WARM[case])
    wall = time.perf_counter() - t0
    rec['rss_monitor'] = mon.stop()
    ev = inst.end()
    rec['rusage_after'] = _rusage()
    rec['rss_after'] = _rss()
    rec['concurrent_after'] = _concurrent_process_snapshot()
    guard_after = dict(_GUARD.counts)
    rec['guard_delta'] = {k: guard_after[k] - guard_before[k] for k in guard_after}
    rec['solve_calls_in_case'] = ev['solve_calls']
    rec['cap_blocked_in_case'] = ev['cap_blocked']
    rec['solver_options_passed'] = ev['solver_options_passed']
    rec['command'] = ev.get('command')
    rec['returncode'] = ev['returncode']
    succeeded = bool(network_module.solver_result_succeeded(result))
    rec['succeeded'] = succeeded
    rec['termination_condition'] = (str(result.solver.termination_condition)
                                    if result is not None and hasattr(result, 'solver') else None)
    rec['solver_status'] = (str(result.solver.status) if result is not None and hasattr(result, 'solver') else None)
    rec['solver_message'] = (str(getattr(result.solver, 'message', None))
                             if result is not None and hasattr(result, 'solver') else None)
    stdout_path = os.path.join(logs_dir, f'{case_key}_stdout_stderr.log')
    if ev['stdout_stderr_text'] is not None:
        with open(stdout_path, 'x') as handle:
            handle.write(ev['stdout_stderr_text'])
        rec['ipopt_stdout_stderr_file'] = os.path.relpath(stdout_path, REPO)
    else:
        rec['ipopt_stdout_stderr_file'] = None
    one = lambda xs: (xs[0] if len(xs) == 1 else (None if not xs else xs))  # noqa: E731
    nl_cap = sum(c['capture_s'] for c in ev['nl_capture'])
    timing = {
        'run_smopf_wall_s': wall,
        'opt_solver_solve_wall_s': one(ev['solve_wall']),
        'presolve_wall_s (NL write + command build + NL capture)': one(ev['presolve']),
        'nl_write_wall_s (NLWriter.__call__)': one(ev['nl_write']),
        'nl_capture_hardlink_s (harness overhead, inside presolve)': nl_cap,
        'ipopt_process_wall_s (_execute_command)': one(ev['execute']),
        'sol_read_wall_s (_postsolve)': one(ev['postsolve']),
        'load_from_wall_s (ModelSolutions.load_from)': one(ev['load_from']),
    }
    try:
        res_load = ev['postsolve'][0] + ev['load_from'][0]
    except IndexError:
        res_load = None
    timing['result_load_wall_s (sol read + load_from)'] = res_load
    if len(ev['solve_wall']) == 1 and len(ev['presolve']) == 1 and len(ev['execute']) == 1 and len(ev['postsolve']) == 1:
        timing['pyomo_solve_epilogue_s (solve - presolve - process - postsolve)'] = (
            ev['solve_wall'][0] - ev['presolve'][0] - ev['execute'][0] - ev['postsolve'][0])
        timing['run_smopf_outside_solve_s (option assembly, suffix copy, record parse, load_from)'] = (
            wall - ev['solve_wall'][0])
    rec['timing'] = timing
    rec['nl_capture'] = ev['nl_capture']
    try:
        _post_solve_parse(rec, ev, log_path, model, network, network_module, pe, wall, res_load, succeeded)
    except Exception as error:  # noqa: BLE001 -- the raw logs / NL are on disk; a parse fault must not cost a solve
        rec['post_solve_parse_error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f'post-solve parse error recorded for {case_key}: {error}')
    if dry:
        rec['dry_run_note'] = 'blocked by the zero-solve guard inside run_smopf (expected); no IPOPT process'
    return rec, succeeded


def _post_solve_parse(rec, ev, log_path, model, network, network_module, pe, wall, res_load, succeeded):
    if os.path.exists(log_path):
        text = open(log_path, errors='replace').read()
        rec['ipopt_log'] = parse_ipopt_log(text)
        rec['production_attempt_parse'] = network_module.parse_ipopt_attempt_segment(
            text, dict(ev['solver_options_passed'] or {}))
    else:
        rec['ipopt_log'] = None
        rec['production_attempt_parse'] = None
    rec['production_solve_records'] = [dict(r) for r in network.ipopt_solve_records]
    if succeeded:
        rec['vars_at_bounds'] = vars_at_bounds(model, pe)
    # derived shares (per solve)
    s = (rec['ipopt_log'] or {}).get('summary', {})
    nlw = ev['nl_write'][0] if len(ev['nl_write']) == 1 else None
    if nlw is not None and res_load is not None:
        rec['shares'] = {
            'nl_write_plus_result_load_over_run_smopf_wall': (nlw + res_load) / wall,
            'nl_write_over_run_smopf_wall': nlw / wall,
            'result_load_over_run_smopf_wall': res_load / wall,
            'ipopt_process_over_run_smopf_wall': ev['execute'][0] / wall if ev['execute'] else None,
            'linear_solver_over_overall_algorithm': (s.get('linear_solver_wall_s') / s['overall_algorithm_wall_s']
                                                     if s.get('linear_solver_wall_s') is not None
                                                     and s.get('overall_algorithm_wall_s') else None),
            'function_evals_over_overall_algorithm': (s.get('function_evaluations_wall_s') / s['overall_algorithm_wall_s']
                                                      if s.get('function_evaluations_wall_s') is not None
                                                      and s.get('overall_algorithm_wall_s') else None),
            'definition': 'denominator run_smopf_wall_s = wall of network.run_smopf (the whole production solve call)',
        }
    if ev['stdout_stderr_text']:
        rec['stdout_parse'] = parse_ipopt_log(ev['stdout_stderr_text'])['summary']


# ======================================================================================================================
#  capture-path checklist (asserted BEFORE the first solve)
# ======================================================================================================================
def capture_checklist(inst, prepared, network_module, out_dir, dry, deferred=()):
    """`deferred` (Part B only; empty for Part A, whose behaviour and record are unchanged): per-case item names
    recorded but not failed here, because they are asserted per case immediately before that case's solve."""
    import pyomo.environ  # noqa: F401
    from pyomo.opt import WriterFactory
    from pyomo.repn.plugins.nl_writer import NLWriter
    items = {}
    wired = inst.is_wired()
    items['instrument_wrappers_live (NL write, presolve, process, sol read, load_from, per-case cap)'] = all(wired.values())
    items['nl_writer_is_the_wrapped_class'] = type(WriterFactory('nl')) is NLWriter
    src = inspect.getsource(network_module._run_smopf)
    items['run_smopf_calls_load_from'] = 'model.solutions.load_from(result)' in src
    items['run_smopf_attempt_uses_solver_solve'] = 'solver.solve(model' in inspect.getsource(
        network_module._run_smopf_solver_attempt)
    items['production_attempt_parser_exists'] = callable(getattr(network_module, 'parse_ipopt_attempt_segment', None))
    items['guard_installed'] = _GUARD is not None
    items['psutil_monitor'] = psutil.Process().memory_info().rss > 0
    items['rusage'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 0
    per_case = {}
    for ck, p in prepared.items():
        opts = p['params'].solver_params.options
        net = p['network']
        lp = expected_log_path(net, opts)
        c = {
            'output_file_set': bool(opts.get('output_file')),
            'log_dir_is_out_logs': os.path.abspath(net.logs_dir) == os.path.abspath(os.path.join(out_dir, 'logs')),
            'log_path_fresh': not os.path.exists(lp),
            'file_print_level_ge_6 (mu lines, objective scaling factor, iteration table)': int(opts.get('file_print_level', 5)) >= 6,
            'print_timing_statistics_yes': opts.get('print_timing_statistics') == 'yes',
            'scaling_defaults_untouched': not any(k in opts for k in IPOPT_SCALING if k.startswith(('nlp_scaling', 'obj_scaling'))
                                                  and not (p['case'] == 'iii_warm_diag' and k == 'nlp_scaling_method')),
            'solver_executable_exists': os.path.isfile(p['params'].solver_params.solver_path),
            'network_records_deque': isinstance(getattr(net, 'ipopt_solve_records', None), deque),
            'warm_suffixes_present': (len(p['model'].ipopt_zL_out) > 0 and len(p['model'].dual) > 0)
            if CASE_IS_WARM[p['case']] else True,
            'cold_suffixes_cleared': (len(p['model'].dual) == 0 and len(p['model'].ipopt_zL_in) == 0)
            if not CASE_IS_WARM[p['case']] else True,
            'iii_diagnostic_options': all(opts.get(k) == v for k, v in CASE_III_OPTIONS.items())
            if p['case'] == 'iii_warm_diag' else True,
        }
        per_case[ck] = c
    items['per_case'] = per_case
    items['nl_dir_exists'] = os.path.isdir(os.path.join(out_dir, 'nl'))
    failures = [k for k, v in items.items() if k != 'per_case' and v is not True]
    for ck, c in per_case.items():
        failures += [f'{ck}:{k}' for k, v in c.items() if v is not True and k not in deferred]
    items['failures'] = failures
    items['asserted_before_first_solve'] = True
    if deferred:
        items['deferred_per_case_items (asserted per case immediately before that case\'s solve)'] = list(deferred)
    if failures:
        raise RuntimeError(f'capture checklist failed before any solve: {failures}')
    return items


# ======================================================================================================================
#  Part A
# ======================================================================================================================
def production_reference_iterations():
    """ZERO-SOLVE read of the committed production per-attempt records of THIS cell (child manifest-verified):
    the iterations each ADMM round's solve of the two blocks actually took."""
    path = os.path.join(REPO, INPUTS['production_solve_records'][0])
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    out = {}
    for block, net in (('tso', 'case9'), ('dso', 'case33_2')):
        sel = [r for r in rows if r['network'] == net and r['year'] == YEAR and r['day'] == DAY]
        warm = [r['iterations'] for r in sel if r['round'] > 0 and r['attempt'] == 'primary']
        init = [r['iterations'] for r in sel if r['round'] == 0 and r['attempt'] == 'primary']
        srt = sorted(warm)
        out[block] = {'network': net, 'n_records': len(sel),
                      'attempts': sorted({r['attempt'] for r in sel}),
                      'initialisation_round0_iterations': init,
                      'cycle_rounds_iterations_min': min(warm), 'cycle_rounds_iterations_median': srt[len(srt) // 2],
                      'cycle_rounds_iterations_max': max(warm),
                      'cycle_rounds_iterations_mean': sum(warm) / len(warm),
                      'cycle_rounds_le_30': sum(1 for w in warm if w <= 30), 'cycle_rounds_n': len(warm),
                      'last_10_rounds': [r['iterations'] for r in sel if r['attempt'] == 'primary'][-10:],
                      'floor_status_counts': {s: sum(1 for r in sel if str(r.get('floor_status')) == s)
                                              for s in sorted({str(r.get('floor_status')) for r in sel})},
                      'restoration_or_recovery_attempts': [r['round'] for r in sel if r['attempt'] != 'primary']}
    out['source'] = INPUTS['production_solve_records'][0]
    out['note'] = ('production cycle k warm-starts from cycle k-1 with UPDATED Params; case (i) here re-solves the '
                   'identical cycle-112 problem from its own solution, a lower bound on that cost')
    return out


def run_local(dry_dir=None):
    dry = dry_dir is not None
    out_dir = os.path.abspath(dry_dir) if dry else os.path.join(REPO, LOCAL_OUT_REL)
    for forbidden in D0_FORBIDDEN:
        if os.path.abspath(out_dir).startswith(os.path.abspath(os.path.join(REPO, forbidden))):
            raise RuntimeError('refusing: output under the D0 root')
    lock = Lock(os.path.join(REPO, LOCK_REL))
    lock_info = lock.acquire()
    try:
        if os.path.exists(out_dir):
            raise RuntimeError(f'refusing: output directory exists: {out_dir}')
        os.makedirs(os.path.join(out_dir, 'logs'))
        os.makedirs(os.path.join(out_dir, 'nl'))
        record = {'schema': 'p516_tperf_local_v1', 'stage': STAGE, 'authority': AUTHORITY, 'utc_start': _utc(),
                  'mode': 'dry_run' if dry else 'part_a', 'instance': INSTANCE,
                  'objective_convention': 'none computed or compared (timing / iterations / memory only)',
                  'lock': lock_info, 'thread_env_before': THREAD_ENV_BEFORE,
                  'thread_env_in_force': {k: os.environ.get(k) for k in THREAD_CAP_ENV}}
        record['provenance'] = {
            'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'hw_model': _git_free_cmd(['sysctl', '-n', 'hw.model']),
            'mem_total_bytes': psutil.virtual_memory().total, 'cpu_count_logical': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'git_head': _git(['rev-parse', 'HEAD']), 'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
            'production_files_git_status': _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES)),
            'production_files_sha256': {f: _sha(os.path.join(REPO, f)) for f in PRODUCTION_FILES},
            'harness_sha256': _sha(THIS), 'harness_git_status': _git(['status', '--porcelain', '--', 'p516_tperf_profile.py']),
            'ipopt_version_probe (not a solve)': _git_free_cmd(['/usr/local/bin/ipopt', '--version']),
        }
        if record['provenance']['production_files_git_status']:
            raise RuntimeError(f"production files not clean: {record['provenance']['production_files_git_status']}")
        _log('verifying input hashes (declared + campaign manifest + child manifest)')
        record['inputs'] = verify_inputs()

        import pyomo
        import pyomo.environ as pe
        import network as network_module
        import model_construction_helpers as mch
        record['provenance']['pyomo_version'] = pyomo.version.version
        record['memory'] = {'rss_after_imports': _rss()}

        _log('unpickling certified_models.pkl')
        t = time.perf_counter()
        with open(os.path.join(REPO, INPUTS['certified_models'][0]), 'rb') as handle:
            payload = pickle.load(handle)
        record['memory']['certified_models_unpickle_s'] = time.perf_counter() - t
        record['memory']['rss_after_unpickle'] = _rss()
        if not (isinstance(payload, dict) and 'tso' in payload and 'dso' in payload
                and DAY in payload['tso'].get(YEAR, {}) and DAY in payload['dso'].get(DSO_NODE, {}).get(YEAR, {})):
            raise RuntimeError('pickle lacks the 2025 Spring TSO / DSO node-7 blocks')
        record['pickle_structure'] = pickle_structure(payload)
        originals = {'tso': payload['tso'][YEAR][DAY], 'dso': payload['dso'][DSO_NODE][YEAR][DAY]}
        prepared = {}
        record['blocks'] = {}
        for block in BLOCKS:
            orig = originals[block]
            net0, prm0 = _build_time_objects(orig)
            record['blocks'][block] = {
                'model_name': orig.name, 'network': net0.name, 'year': net0.year, 'day': net0.day,
                'baseMVA': net0.baseMVA, 'size_pyomo_certified_point': block_size(orig, pe),
                'suffix_sizes': {s: len(getattr(orig, s)) for s in ('dual', 'ipopt_zL_out', 'ipopt_zU_out',
                                                                    'ipopt_zL_in', 'ipopt_zU_in')},
                'production_solver_params': {'solver': prm0.solver_params.solver,
                                             'solver_path': prm0.solver_params.solver_path,
                                             'options': dict(prm0.solver_params.options or {}),
                                             'recovery_enabled': getattr(prm0.solver_params, 'recovery_enabled', None),
                                             'recovery_tier2_enabled': getattr(prm0.solver_params, 'recovery_tier2_enabled', None),
                                             'recovery_options': dict(prm0.solver_params.recovery_options or {}),
                                             'release_solution_bookkeeping': getattr(prm0.solver_params, 'release_solution_bookkeeping', None),
                                             'verbose': prm0.solver_params.verbose},
                'production_logs_dir_in_pickle': net0.logs_dir}
            for case in CASES:
                ck = f'{block}_{case}'
                t = time.perf_counter()
                model = orig.clone()
                clone_s = time.perf_counter() - t
                net, prm = _build_time_objects(model)
                if net is net0 or prm is prm0:
                    raise RuntimeError('clone shares network/params with the loaded block')
                net.logs_dir = os.path.join(out_dir, 'logs')
                net.ipopt_solve_records = deque(maxlen=network_module.IPOPT_SOLVE_RECORDS_MAXLEN)
                options = dict(prm.solver_params.options or {})
                production_options = dict(options)
                options['output_file'] = f'tperf_{ck}.log'
                options.update(ALL_CASE_OPTIONS)
                if case == 'iii_warm_diag':
                    options.update(CASE_III_OPTIONS)
                prm.solver_params.options = options
                entry = {'clone_s': clone_s,
                         'options_added_or_changed': {k: v for k, v in options.items()
                                                      if production_options.get(k) != v},
                         'production_options': production_options}
                if case == 'ii_cold':
                    entry['cold_reset'] = reset_to_initialisation_point(model, block, network_module, mch, pe)
                entry['size_pyomo_at_solve'] = block_size(model, pe)
                prepared[ck] = {'block': block, 'case': case, 'model': model, 'network': net, 'params': prm,
                                'entry': entry}
        del originals
        del payload
        gc.collect()
        record['memory']['rss_after_clone_and_release'] = _rss()

        _log('ESSO pickle: size only (0 solves)')
        with open(os.path.join(REPO, INPUTS['esso_models'][0]), 'rb') as handle:
            esso = pickle.load(handle)
        record['esso'] = {'keys': sorted(esso.keys()), 'node_7_size': block_size_esso(esso[DSO_NODE], pe)}
        del esso
        gc.collect()
        record['production_reference_iterations'] = production_reference_iterations()

        inst = Instrument(os.path.join(out_dir, 'nl'), dry).install()
        record['capture_checklist'] = capture_checklist(inst, prepared, network_module, out_dir, dry)
        record['memory']['rss_build_footprint_before_first_solve'] = _rss()
        record['memory']['rusage_before_first_solve'] = _rusage()
        _log('capture checklist holds; starting the declared solves' + (' (DRY: all blocked)' if dry else ''))

        record['solves'] = {}
        stopped = None
        for ck, p in prepared.items():
            _log(f'case {ck}: run_smopf(from_warm_start={CASE_IS_WARM[p["case"]]})')
            rec, ok = run_case(ck, p['block'], p['case'], p['model'], p['network'], p['params'], inst,
                               network_module, pe, os.path.join(out_dir, 'logs'), dry)
            rec['preparation'] = p['entry']
            record['solves'][ck] = rec
            s = (rec.get('ipopt_log') or {}).get('summary', {})
            _log(f'case {ck}: ok={ok} term={rec["termination_condition"]} iters={s.get("iterations")} '
                 f'restoration={s.get("restoration_iteration_lines")} run_smopf={rec["timing"]["run_smopf_wall_s"]:.3f}s '
                 f'ipopt={rec["timing"]["ipopt_process_wall_s (_execute_command)"]}')
            if dry:
                if dry_nl_selftest(ck, p['model'], out_dir, rec) is False:
                    stopped = f'dry-run NL self-test failed at {ck}'
                    break
                continue
            if not ok or rec['solve_calls_in_case'] != 1 or rec['cap_blocked_in_case']:
                stopped = (f'solve failed or count off at {ck}: ok={ok} calls={rec["solve_calls_in_case"]} '
                           f'cap_blocked={rec["cap_blocked_in_case"]} -- stopping (task: record and stop)')
                _log(stopped)
                break
        record['stopped'] = stopped

        # scaling replica (post-processing, after every solve)
        if not dry:
            for ck, rec in record['solves'].items():
                if not rec['nl_capture']:
                    rec['gradient_scaling'] = {'computed': False, 'reason': 'no NL captured'}
                    continue
                try:
                    rep = gradient_scaling_replica(rec['nl_capture'][0]['path'])
                    rep['validation_against_ipopt_log'] = validate_replica(rep, rec['ipopt_log'])
                    rep['computed'] = True
                    rec['gradient_scaling'] = rep
                except Exception as error:  # noqa: BLE001 -- recorded; the solves are already done
                    rec['gradient_scaling'] = {'computed': False, 'reason': f'{type(error).__name__}: {error}',
                                               'traceback': traceback.format_exc()}

        record['memory']['rusage_end'] = _rusage()
        expected = 0 if dry else DECLARED_SOLVES
        if dry:
            fails = []
            if _GUARD.counts['permitted_solve'] or _GUARD.counts['permitted_exec']:
                fails.append(f'dry run reached a permitted solve {_GUARD.counts}')
            if _GUARD.counts['blocked_solve'] != len(record['solves']):
                fails.append(f"blocked {_GUARD.counts['blocked_solve']} != cases {len(record['solves'])}")
        else:
            fails = _GUARD.verify(expected)
        record['guard'] = {'permitted': [list(p) for p in PERMITTED] if not dry else [],
                           'declared_solves': expected, 'counts': dict(_GUARD.counts),
                           'verify_failures': fails, 'per_case_cap_blocks': inst.cap_blocks}
        record['status'] = ('complete' if not fails and stopped is None else
                            ('stopped' if stopped else 'guard_failed'))
        try:
            record['summary'] = summarise(record) if not dry else None
        except Exception as error:  # noqa: BLE001 -- the per-solve records are complete without it
            record['summary'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        record['utc_end'] = _utc()
        record['wall_s'] = time.time() - T0
        name = 'tperf_local.json' if not dry else 'tperf_dry_run.json'
        _write_json(os.path.join(out_dir, name), record)
        _log(f'status={record["status"]} guard={_GUARD.counts} verify_failures={fails} -> {name}')
        return 0 if record['status'] == 'complete' else 2
    finally:
        lock.release()


def _git_free_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'{type(error).__name__}: {error}'


def block_size_esso(model, pe):
    nv = nf = 0
    for v in model.component_data_objects(pe.Var, descend_into=True):
        nv += 1
        nf += bool(v.fixed)
    nc = sum(1 for _ in model.component_data_objects(pe.Constraint, active=True, descend_into=True))
    return {'pyomo_vars': nv, 'pyomo_vars_fixed': nf, 'pyomo_vars_free': nv - nf, 'pyomo_active_constraints': nc,
            'active_objectives': [o.name for o in model.component_data_objects(pe.Objective, active=True)]}


def dry_nl_selftest(ck, model, out_dir, rec):
    """Zero-solve: write this case's NL with Pyomo's writer and run the replica on it (parser self-test)."""
    path = os.path.join(out_dir, 'nl', f'{ck}_selftest.nl')
    try:
        t = time.perf_counter()
        model.write(path, format='nl', io_options={'symbolic_solver_labels': False})
        rec['dry_nl_write_s'] = time.perf_counter() - t
        rep = gradient_scaling_replica(path)
        rec['dry_gradient_scaling_selftest'] = rep
        _log(f'  dry NL self-test {ck}: n_free={rep["n_free_vars"]} eq={rep["n_eq_rows"]} ineq={rep["n_ineq_rows"]} '
             f'df={rep["df"]:.6g} c_prov={rep["c"]["scaling_provided"]} d_prov={rep["d"]["scaling_provided"]} '
             f'span={rep["factor_span_decades_all"]:.3f} rowgrad_span={rep["row_max_gradient_span_decades_all"]:.3f} '
             f'({rep["compute_s"]:.2f}s)')
        return True
    except Exception as error:  # noqa: BLE001
        rec['dry_gradient_scaling_selftest'] = {'error': f'{type(error).__name__}: {error}',
                                                'traceback': traceback.format_exc()}
        _log(f'  dry NL self-test {ck} FAILED: {error}')
        return False


def summarise(record):
    rows = {}
    for ck, rec in record['solves'].items():
        s = (rec.get('ipopt_log') or {}).get('summary', {})
        p = (rec.get('ipopt_log') or {}).get('problem', {})
        g = rec.get('gradient_scaling') or {}
        t = rec['timing']
        rows[ck] = {
            'termination': rec['termination_condition'], 'iterations': s.get('iterations'),
            'restoration_iterations': s.get('restoration_iteration_lines'), 'restoration_phases': s.get('restoration_phases'),
            'mu_final': (rec.get('production_attempt_parse') or {}).get('mu_final'),
            'floor_status': (rec.get('production_attempt_parse') or {}).get('floor_status'),
            'n_vars_ipopt': p.get('Total number of variables'), 'n_eq': p.get('Total number of equality constraints'),
            'n_ineq': p.get('Total number of inequality constraints'),
            'nnz_jac_eq': p.get('nnz equality constraint Jacobian'), 'nnz_jac_ineq': p.get('nnz inequality constraint Jacobian'),
            'nnz_hessian': p.get('nnz Lagrangian Hessian'),
            'run_smopf_s': t['run_smopf_wall_s'], 'nl_write_s': t['nl_write_wall_s (NLWriter.__call__)'],
            'result_load_s': t['result_load_wall_s (sol read + load_from)'],
            'ipopt_process_s': t['ipopt_process_wall_s (_execute_command)'],
            'ipopt_overall_algorithm_s': s.get('overall_algorithm_wall_s'),
            'ipopt_linear_solver_s': s.get('linear_solver_wall_s'),
            'ipopt_function_evals_s': s.get('function_evaluations_wall_s'),
            'shares': rec.get('shares'),
            'peak_self_rss': rec['rss_monitor']['peak_self_rss'], 'peak_ipopt_rss': rec['rss_monitor']['peak_child_rss'],
            'vars_at_bounds_1e-6': ((rec.get('vars_at_bounds') or {}).get('rtol_1e-06') or {}).get('at_any'),
            'scaling_df': g.get('df'), 'scaling_span_decades': g.get('factor_span_decades_all'),
            'scaling_factor_min': g.get('factor_min_all'), 'scaling_factor_max': g.get('factor_max_all'),
            'row_gradient_span_decades': g.get('row_max_gradient_span_decades_all'),
            'scaling_replica_validated': (g.get('validation_against_ipopt_log') or {}).get('all_pass'),
        }
    return rows


# ======================================================================================================================
#  Part B -- the Studio (--studio), its zero-solve rehearsal on this machine (--studio-dry), and the zero-solve
#  check that the one-block restriction leaves the block unchanged (--restriction-equivalence)
# ======================================================================================================================
STUDIO_DRY_OUT_REL = os.path.join(TPERF_ROOT_REL, 'studio_dry')
STUDIO_EQUIV_OUT_REL = os.path.join(TPERF_ROOT_REL, 'studio_dry_equiv')
LAUNCH_LOGS_REL['studio_dry'] = [os.path.join(TPERF_ROOT_REL, 'studio_dry_launch_stdout.log'),
                                 os.path.join(TPERF_ROOT_REL, 'studio_dry_launch_stderr.log')]
LAUNCH_LOGS_REL['studio_dry_equiv'] = [os.path.join(TPERF_ROOT_REL, 'studio_dry_equiv_launch_stdout.log'),
                                       os.path.join(TPERF_ROOT_REL, 'studio_dry_equiv_launch_stderr.log')]
# Planner ruling (Part B): the unit cell, investment year 2025 (the `_construct_arm_planning` default cohort).
STUDIO_UNIT_CELL = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
STUDIO_CASE_ORDER = ('ii_cold', 'i_warm', 'iii_warm_diag')   # no certified point at paper scale: cold first
GIB = 1 << 30
STUDIO_DRY_RSS_ABORT_BYTES = 16 * GIB    # task stop condition on this 24 GB machine (RSS of process + children)
STUDIO_RSS_ABORT_BYTES = 24 * GIB        # the Studio (32 GB): the s44 precedent's 24 GiB rss_tree limit
STUDIO_MIN_AVAILABLE_BYTES = 10 * GIB    # see run_studio_mode; the one-block build needs far less than the s44 full build
# Tracked inputs of the build: every repository module loaded once the build modules are imported (computed, not
# listed) plus the case data under data/SRP1 (Results excluded). --studio-dry / --restriction-equivalence refuse if
# any is modified (the harness itself excepted); --studio refuses on ANY modified tracked file (the author's
# pre-flight: `git status` clean on tracked files).
STUDIO_DATA_PATHSPEC = ('data/SRP1', ':(exclude)data/SRP1/Results')


def _loaded_repo_modules():
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f).startswith(REPO + os.sep) and os.path.dirname(os.path.abspath(f)) == REPO:
            out.add(os.path.basename(f))
    return sorted(out)


def _build_input_status():
    """Import the build modules (module-level code only; no build), then `git status` of every loaded repository
    module and of the case data. Returns (status_text, loaded_modules, dirty_lines_excluding_the_harness)."""
    _studio_modules()
    mods = _loaded_repo_modules()
    status = _git_status(['--untracked-files=no', '--'] + mods + list(STUDIO_DATA_PATHSPEC))
    dirty = [ln for ln in status.splitlines() if not ln.endswith('p516_tperf_profile.py')]
    return status, mods, dirty
STUDIO_DRY_PREDICTIONS = {
    'D1_dso_block_variables': (
        'a paper-scale DSO block (node 7, 2025 Spring, 25 scenario combinations) has about 269k variables '
        '(Planner, citing REVISION_CONTEXT.md; the figure appears in P5_15_S44_SELECTION_REPORT.md:85 and '
        'STEP4_DFO_METHOD.md:258 and is the s44 per-block len(Var data) 269,050 at Network.build_model)'),
    'D2_build_rss': ('peak RSS of this process over the build of the TSO block + the DSO block + the ESSO '
                     '<= 6 GiB'),
}


class MemWatchdog(threading.Thread):
    """RSS of this process plus its descendants, sampled every `interval` s; above `limit` it writes an abort record
    (best effort) and ends the process with exit code 97 (os._exit: the memory is released at once)."""

    def __init__(self, limit_bytes, abort_path, interval=0.5):
        super().__init__(daemon=True)
        self.limit = limit_bytes
        self.abort_path = abort_path
        self.interval = interval
        self.stop_event = threading.Event()
        self.stage = 'start'
        self.n = 0
        self.peak = {'rss_tree': 0, 'rss_self': 0, 'rss_children': 0, 'stage': None, 't': None}
        self.t0 = time.time()

    def run(self):
        me = psutil.Process()
        while not self.stop_event.is_set():
            try:
                s = me.memory_info().rss
                c = 0
                for ch in me.children(recursive=True):
                    try:
                        c += ch.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.Error:
                self.stop_event.wait(self.interval)
                continue
            self.n += 1
            if s + c > self.peak['rss_tree']:
                self.peak = {'rss_tree': s + c, 'rss_self': s, 'rss_children': c, 'stage': self.stage,
                             't': time.time() - self.t0}
            if s + c > self.limit:
                rec = {'cause': 'rss_tree above limit', 'limit_bytes': self.limit, 'rss_tree': s + c, 'rss_self': s,
                       'rss_children': c, 'stage': self.stage, 't': time.time() - self.t0, 'utc': _utc()}
                try:
                    with open(self.abort_path, 'x') as handle:
                        json.dump(rec, handle, indent=1)
                except Exception:  # noqa: BLE001 -- best effort; the stderr line below is the primary record
                    pass
                print(f'MEMORY WATCHDOG ABORT: {rec}', file=sys.stderr, flush=True)
                os._exit(97)
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()
        return self.summary()

    def summary(self):
        return {'limit_bytes': self.limit, 'measure': 'rss of this process + all descendants (psutil)',
                'interval_s': self.interval, 'n_samples': self.n, 'peak': dict(self.peak)}


def _foreign_compute_processes():
    """Other processes that would compete for the machine: python interpreters running a repository harness
    (`p5<digit>…py`, `w1<digit><digit>…py`, `main.py`) or anything under this repository, and IPOPT processes."""
    me = {os.getpid(), os.getppid()}
    pat = re.compile(r'(^|[/\s])(p5\d\w*\.py|w1\d\d\w*\.py|main\.py)(\s|$)')
    rows = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = (proc.info['name'] or '').lower()
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me:
            continue
        if 'ipopt' in name or ('python' in name and (pat.search(cmd) or REPO in cmd)):
            rows.append({'pid': proc.info['pid'], 'name': name, 'cmdline': cmd[:300]})
    return rows


class _PlanningView:
    """Read-only view of a planning object with `distribution_networks` replaced by a subset. Passed ONLY to
    `update_distribution_models_to_admm`, which reads nothing from the planning object but `distribution_networks`
    (shared_resources_planning.py; checked by reading its body), so that it iterates the node-7 DSO that was built
    and not the two that were not. The planning object itself is never modified through this view."""

    def __init__(self, planning, distribution_networks):
        object.__setattr__(self, '_planning', planning)
        object.__setattr__(self, 'distribution_networks', distribution_networks)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_planning'), name)

    def __setattr__(self, name, value):
        raise AttributeError('read-only planning view')


def _stage_factory(stages, wd):
    from contextlib import contextmanager

    @contextmanager
    def stage(name):
        wd.stage = name
        t = time.perf_counter()
        rss0 = _rss()
        _log(f'stage: {name}')
        ok = False
        try:
            yield
            ok = True
        finally:
            stages.append({'stage': name, 'wall_s': time.perf_counter() - t, 'rss_before': rss0, 'rss_after': _rss(),
                           'self_ru_maxrss_after': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'ok': ok})
    return stage


def _studio_modules():
    import pyomo.environ as pe
    import network as network_module
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p515_s44_scale_measurement as S44
    return pe, network_module, srp, G, S44


def studio_read(out_dir, instance, baseline_label, stage, record):
    """Derive the case (p515_s44_scale_measurement.derive_case(instance, {})), read it with production's reader
    (that stage's `read_planning_from_derived_case`: diagrams / results / logs redirected under `out_dir`), install
    it as the oracle baseline, and redirect the oracle's eval directory (`p56a_oracle.WORK_DIR`, where
    `fresh_planning` creates `<eval_id>/logs`) under `out_dir` so that nothing is written outside the output root."""
    import types
    pe, network_module, srp, G, S44 = _studio_modules()
    O = G.O
    case, spec, changes = S44.derive_case(instance, {})
    case_path = os.path.join(out_dir, f'derived_case_{instance}.json')
    with open(case_path, 'x') as handle:
        json.dump(case, handle, indent=1)
    launch = {'derived_case': {'path': os.path.relpath(case_path, REPO), 'sha256': _sha(case_path)},
              'instance': baseline_label}
    record['derived_case'] = {'instance': instance, 'path': launch['derived_case']['path'],
                              'sha256': launch['derived_case']['sha256'], 'spec': spec, 'changes_vs_source': changes,
                              'source_case': S44.SOURCE_CASE_REL, 'source_case_sha256': _sha(S44.SOURCE_CASE),
                              'derivation': f"p515_s44_scale_measurement.derive_case({instance!r}, {{}})"}
    planning0 = S44.read_planning_from_derived_case(launch, out_dir, types.SimpleNamespace(stage=stage))
    record['scenario_checksum'] = S44.inject_oracle_baseline(O, planning0, launch)
    record['scenario_checksum_equals_srp1_canonical'] = record['scenario_checksum'] == O.CANONICAL_CHECKSUM
    record['baseline_label'] = baseline_label
    record['planning_dimensions'] = S44.planning_dimensions(planning0)
    before = O.WORK_DIR
    O.WORK_DIR = os.path.join(out_dir, 'oracle_evals')
    record['oracle_work_dir_redirect'] = {'from': os.path.relpath(before, REPO), 'to': os.path.relpath(O.WORK_DIR, REPO),
                                          'reason': 'p56a_oracle.fresh_planning creates <WORK_DIR>/<eval_id>/logs; '
                                                    'kept under this output root'}
    return planning0


def studio_build_blocks(out_dir, eval_id, restrict, stage, rec):
    """The s44 zero-solve build path (`_construct_arm_planning` with apply_rho=False; production `create_*_model`
    with `.optimize` intercepted on THIS planning instance by `p515_s44_scale_measurement.Interceptor`; the ADMM
    objective preparation of that stage's child_build), on the unit-cell candidate.

    restrict=True builds ONE TSO block (YEAR, DAY) and ONE DSO block (node DSO_NODE, YEAR, DAY) plus the ESSO,
    not the whole instance. Mechanism (harness-only; no production change): AFTER `create_admm_variables` (which
    keeps the full consensus structure) and AFTER the ESSO AL scale is resolved over the full block set, the
    `years` / `days` dicts of the TSO holder and of the node-7 DSO holder on this private planning copy are replaced
    by one-entry dicts with the same weights, for the duration of the create / prepare / update calls, then
    restored; `create_distribution_networks_models` and `_prepare_distribution_objectives_for_admm` receive
    {DSO_NODE: dn}; `update_distribution_models_to_admm` receives a `_PlanningView` with that subset. The one
    year-list-dependent quantity in the block, the ADMM block weight's annualisation (int(year) - int(years[0])),
    is unchanged because YEAR is the instance's first year (asserted). The ESSO is not restricted (all nodes, all
    years: it has no scenario index and is small); it gets 0 solves. `--restriction-equivalence` checks the
    restriction bitwise (NL files) against the full build at SRP1 scale.

    ZERO solves: every agent's `.optimize` is intercepted (declared counts checked exactly); the caller arms a
    blocking guard over this whole call."""
    pe, network_module, srp, G, S44 = _studio_modules()
    R = G.R
    construct_report = {}
    with stage(f'construct arm planning ({eval_id}; s44 path, apply_rho=False, unit-cell candidate)'):
        planning, sed, candidate = G._construct_arm_planning(
            eval_id, os.path.join(out_dir, f'arm_build_{eval_id}'), construct_report,
            investment_map=STUDIO_UNIT_CELL, eval_id=eval_id, num_max_iters_override=1, apply_rho=False)
    rec['construct_report'] = construct_report
    if planning.parallel_execution:
        raise RuntimeError('ParallelExecution is on; this build follows the sequential path')
    params = planning.params.admm
    scale = params.objective_scale
    if scale is None:
        raise RuntimeError('case file has no fixed objective_scale')
    rec['objective_scale_used'] = scale
    rec['admm_settings_in_force (set by _construct_arm_planning; none enters a block NLP at build)'] = {
        'num_max_iters': params.num_max_iters, 'tol_objective_rel': params.tol['objective']['rel'],
        'adaptive_penalty': params.adaptive_penalty, 'esso_budget': sed.params.budget,
        'rho': {k: (dict(v) if isinstance(v, dict) else v) for k, v in dict(params.rho).items()}
        if isinstance(params.rho, dict) else str(params.rho)}
    tn = planning.transmission_network
    dns = {DSO_NODE: planning.distribution_networks[DSO_NODE]} if restrict else dict(planning.distribution_networks)
    with stage('ESSO AL scale (full block set, before any restriction)'):
        al_scale = srp._resolve_esso_al_scale(planning, params, scale)[0]
    rec['al_scale_esso'] = al_scale
    with stage('create_admm_variables (full instance)'):
        cv, dv = srp.create_admm_variables(planning)
    saved = []
    if restrict:
        restriction = {}
        for label, holder in (('tso', tn), (f'dso_{DSO_NODE}', dns[DSO_NODE])):
            years, days = holder.years, holder.days
            if list(years)[0] != YEAR or YEAR not in years or DAY not in days:
                raise RuntimeError(f'{label}: restriction needs {YEAR} as first year and {DAY} among days '
                                   f'(years {list(years)}, days {list(days)})')
            saved.append((holder, years, days))
            holder.years = {YEAR: years[YEAR]}
            holder.days = {DAY: days[DAY]}
            restriction[label] = {'years_full': {str(k): v for k, v in years.items()},
                                  'days_full': dict(days), 'years_built': {str(YEAR): years[YEAR]},
                                  'days_built': {DAY: days[DAY]}}
        rec['restriction'] = restriction
    else:
        rec['restriction'] = None
    interceptor = S44.Interceptor()
    init_capture = {}

    def capture(inner, key, pick):
        # The block(s) exactly as production hands them to `.optimize` for the INITIALISATION solve (round 0):
        # cloned here, at the intercepted call, before the interceptor returns "no result".
        def _stub(models, *args, **kwargs):
            block = pick(models)
            if block is not None:
                init_capture[key] = {'model': block.clone(), 'optimize_args': [repr(a) for a in args],
                                     'optimize_kwargs': {k: repr(v) for k, v in kwargs.items()}}
            return inner(models, *args, **kwargs)
        return _stub

    tn.optimize = capture(interceptor.network(tn, 'tso'), 'tso', lambda m: m.get(YEAR, {}).get(DAY))
    for node_id, dn in dns.items():
        dn.optimize = capture(interceptor.network(dn, 'dso'), 'dso' if node_id == DSO_NODE else f'dso_{node_id}',
                              (lambda m: m.get(YEAR, {}).get(DAY)) if node_id == DSO_NODE else (lambda m: None))
    sed.optimize = capture(interceptor.esso(sed), 'esso', lambda m: m.get(DSO_NODE))
    try:
        with stage(f'create_distribution_networks_models (nodes {sorted(dns)}; optimize intercepted)'):
            dso_models, _ = srp.create_distribution_networks_models(dns, cv, candidate['total_capacity'],
                                                                    parallel_execution=False)
        with stage('create_transmission_network_model (optimize intercepted)'):
            tso_model, _ = srp.create_transmission_network_model(planning, cv, candidate['total_capacity'])
        with stage('create_shared_energy_storage_model (optimize intercepted)'):
            esso_model, _ = srp.create_shared_energy_storage_model(sed, cv, candidate['investment'])
        with stage('ADMM objective preparation (s44 child_build sequence)'):
            srp._prepare_distribution_objectives_for_admm(dns, dso_models)
            srp._prepare_transmission_objectives_for_admm(tn, tso_model)
            view = _PlanningView(planning, dns) if restrict else planning
            with R.patched_admm_objectives():
                srp.update_distribution_models_to_admm(view, dso_models, params, scale)
                srp.update_transmission_model_to_admm(planning, tso_model, params, scale)
            srp.update_shared_energy_storage_model_to_admm(planning, esso_model, params, al_scale_esso=al_scale)
    finally:
        for holder in [tn] + list(dns.values()) + [sed]:
            if 'optimize' in vars(holder):
                del holder.optimize
        for holder, years, days in saved:
            holder.years = years
            holder.days = days
    declared = {'dso': len(dns), 'tso': 1, 'esso_init': 1}
    observed = interceptor.counts()
    rec['interceptor'] = {'declared': declared, 'observed': observed, 'exact_match': observed == declared,
                          'mechanism': ('instance attribute `.optimize` on this planning copy\'s TSO holder, the built '
                                        'DSO holder(s) and shared_ess_data, replaced by '
                                        'p515_s44_scale_measurement.Interceptor stubs (record the call, return None '
                                        'per block / node; never a solver); deleted after the build')}
    if observed != declared:
        raise RuntimeError(f'interceptor counts {observed} != declared {declared}')
    rec['models_built'] = {'tso': {str(y): sorted(d) for y, d in tso_model.items()},
                           'dso': {str(n): {str(y): sorted(d) for y, d in yd.items()} for n, yd in dso_models.items()},
                           'esso_nodes': sorted(esso_model)}
    if restrict and not (list(tso_model) == [YEAR] and list(tso_model[YEAR]) == [DAY] and list(dso_models) == [DSO_NODE]
                         and list(dso_models[DSO_NODE]) == [YEAR] and list(dso_models[DSO_NODE][YEAR]) == [DAY]):
        raise RuntimeError(f'restricted build produced {rec["models_built"]}')
    rec['candidate'] = {'investment_map': {str(k): list(v) for k, v in STUDIO_UNIT_CELL.items()},
                        'investment_year': construct_report.get('instance', {}).get('year')}
    if sorted(init_capture) != ['dso', 'esso', 'tso']:
        raise RuntimeError(f'initialisation capture incomplete: {sorted(init_capture)}')
    rec['initialisation_capture'] = {k: {'optimize_args': v['optimize_args'], 'optimize_kwargs': v['optimize_kwargs']}
                                     for k, v in init_capture.items()}
    rec['solved_problem'] = (
        'the INITIALISATION problem of each block: the block cloned at the moment production calls '
        '`.optimize` inside create_*_model (NetworkData.optimize would run run_smopf(block, holder.params, '
        'from_warm_start=False) on it -- the round-0 solve). The ADMM-prepared blocks (after update_*_to_admm) '
        'are built and sized but NOT solvable as they stand: their objectives reference AL Params '
        '(dual_vmag_req, vmag_req, ...) that production sets only in the per-cycle update from the ADMM state')
    return {'bases': {'tso': init_capture['tso']['model'], 'dso': init_capture['dso']['model']},
            'esso_init': init_capture['esso']['model'],
            'admm_prepared': {'tso': tso_model[YEAR][DAY], 'dso': dso_models[DSO_NODE][YEAR][DAY]},
            'esso': esso_model, 'tso_model': tso_model, 'dso_models': dso_models}


def pyomo_jacobian_nnz(model, pe):
    """Pyomo-level nonzeros: distinct Vars per active constraint body (identify_variables), summed; free = not fixed.
    Before Pyomo's NL presolve and before IPOPT removes lb == ub columns (the NL-level counts are separate)."""
    from pyomo.core.expr.visitor import identify_variables
    t = time.perf_counter()
    out = {'eq': {'rows': 0, 'nnz_all_vars': 0, 'nnz_free_vars': 0},
           'ineq': {'rows': 0, 'nnz_all_vars': 0, 'nnz_free_vars': 0}}
    for c in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
        n_all = n_free = 0
        for v in identify_variables(c.body, include_fixed=True):
            n_all += 1
            n_free += not v.fixed
        k = 'eq' if c.equality else 'ineq'
        out[k]['rows'] += 1
        out[k]['nnz_all_vars'] += n_all
        out[k]['nnz_free_vars'] += n_free
    objs = list(model.component_data_objects(pe.Objective, active=True))
    out['objective_gradient_nnz_free'] = (sum(1 for v in identify_variables(objs[0].expr, include_fixed=False))
                                          if len(objs) == 1 else None)
    out['total_nnz_free_vars'] = out['eq']['nnz_free_vars'] + out['ineq']['nnz_free_vars']
    out['compute_s'] = time.perf_counter() - t
    out['definition'] = ('per active constraint: number of distinct Vars in its body (pyomo identify_variables); '
                         'free = Var not fixed; Pyomo level, before NL presolve / IPOPT fixed-variable removal')
    return out


def studio_prepare_case(model, ck, case, logs_dir, network_module, source):
    """Per-case network / params copies (the clone's own, asserted distinct from its source block's), the log
    redirect, and the options Part A used for the same case: production options + output_file + timing statistics
    (+ the case-iii diagnostics). Cold: production's `_clear_multiplier_suffixes` (a freshly built block has none)."""
    net, prm = _build_time_objects(model)
    net_src, prm_src = _build_time_objects(source)
    if net is net_src or prm is prm_src:
        raise RuntimeError(f'{ck}: clone shares network/params with its source block')
    net.logs_dir = logs_dir
    net.ipopt_solve_records = deque(maxlen=network_module.IPOPT_SOLVE_RECORDS_MAXLEN)
    production_options = dict(prm.solver_params.options or {})
    options = dict(production_options)
    options['output_file'] = f'tperf_{ck}.log'
    options.update(ALL_CASE_OPTIONS)
    if case == 'iii_warm_diag':
        options.update(CASE_III_OPTIONS)
    prm.solver_params.options = options
    if case == 'ii_cold':
        network_module._clear_multiplier_suffixes(model)
    entry = {'options_added_or_changed': {k: v for k, v in options.items() if production_options.get(k) != v},
             'production_options': production_options,
             'solver': prm.solver_params.solver, 'solver_path': prm.solver_params.solver_path,
             'recovery_enabled': getattr(prm.solver_params, 'recovery_enabled', None),
             'recovery_tier2_enabled': getattr(prm.solver_params, 'recovery_tier2_enabled', None),
             'network': net.name, 'year': net.year, 'day': net.day, 'baseMVA': net.baseMVA}
    return net, prm, entry


def intended_passed_options(production_options, case, log_path):
    """What `_create_smopf_solver` is intended to hand IPOPT for this case, stated independently of it: the pinned
    fixed_variable_treatment and max_iter, then the case file's options, then this harness's additions; output_file
    becomes the absolute per-(year, day) path with file_append yes; warm cases add warm_start_init_point yes."""
    import helper_functions as HF
    o = {'fixed_variable_treatment': HF.IPOPT_FIXED_VARIABLE_TREATMENT, 'max_iter': 500}
    o.update(production_options)
    o.update(ALL_CASE_OPTIONS)
    if case == 'iii_warm_diag':
        o.update(CASE_III_OPTIONS)
    o['output_file'] = log_path
    o['file_append'] = 'yes'
    if CASE_IS_WARM[case]:
        o['warm_start_init_point'] = 'yes'
    return o


def studio_extra_checklist(record, out_dir, wd, build_zero):
    """Capture paths the Part B record needs beyond the per-solve checklist (asserted before the first solve)."""
    items = {
        'memory_watchdog_running': wd.is_alive(),
        'out_logs_dir_exists': os.path.isdir(os.path.join(out_dir, 'logs')),
        'out_nl_dir_exists': os.path.isdir(os.path.join(out_dir, 'nl')),
        'derived_case_written_and_hashed': bool(record.get('derived_case', {}).get('sha256')),
        'build_wall_and_stages_recorded': record.get('build_wall_s') is not None and bool(record.get('build_stages')),
        'build_memory_recorded': record.get('build_memory', {}).get('self_ru_maxrss_at_build_end') is not None,
        'block_sizes_recorded (vars, rows, Pyomo nnz)': all(
            (record.get('blocks') or {}).get(b, {}).get('size_pyomo_built', {}).get('pyomo_nnz') is not None
            for b in BLOCKS),
        'esso_size_recorded': (record.get('esso') or {}).get(f'node_{DSO_NODE}') is not None,
        'build_produced_zero_solves (guard counts all 0 at build end)': build_zero,
        'scaling_replica_callable': callable(gradient_scaling_replica) and callable(validate_replica),
        'rss_monitor_class (per-solve peak RSS, 25 ms)': issubclass(RssMonitor, threading.Thread),
    }
    failures = [k for k, v in items.items() if v is not True]
    items['failures'] = failures
    if failures:
        raise RuntimeError(f'studio capture checklist failed before any solve: {failures}')
    return items


def _git_status(args):
    return _git(['status', '--porcelain'] + list(args))


def run_studio_mode(dry):
    """--studio (dry=False; the Studio only) and --studio-dry (dry=True; zero solves, allowed on this machine).

    Both run the SAME build (`studio_read` + `studio_build_blocks(restrict=True)`) and the same case preparation and
    capture checklist. --studio-dry then reaches `network.run_smopf` once per case under the blocking guard (blocked
    at OptSolver.solve, before any NL write or process launch: IPOPT is never invoked), records the option dict
    production assembled and compares it with `intended_passed_options`, and writes one NL file per block with
    Pyomo's writer (an NL write, not a solve) for the IPOPT-level counts. --studio runs the 6 declared solves.

    Guards. --studio-dry: the module-level BLOCKING guard covers the whole run; the build must leave every count at
    0, and the option-assembly step must give exactly 6 blocked / 0 permitted. --studio: the module-level BOUNDED
    guard (6 declared) plus a second, BLOCKING guard stacked over it for the whole build (uninstalled after the
    build, which restores the bounded guard); both must read 0 at build end, then verify(6) at the end."""
    hw = _git_free_cmd(['sysctl', '-n', 'hw.model'])
    if not dry and hw == 'Mac17,3':
        raise RuntimeError('refusing: --studio must not run on this 24 GB machine (hw.model Mac17,3; CLAUDE.local.md)')
    avail = psutil.virtual_memory().available
    if not dry and avail < STUDIO_MIN_AVAILABLE_BYTES:
        raise RuntimeError(f'refusing: {avail / GIB:.1f} GiB available < {STUDIO_MIN_AVAILABLE_BYTES / GIB:.0f} GiB')
    foreign = _foreign_compute_processes()
    if not dry and foreign:
        raise RuntimeError(f'refusing: other compute processes are running (nothing else may run during the timed '
                           f'solves): {foreign}')
    mode = 'studio_dry' if dry else 'studio'
    out_dir = os.path.join(REPO, STUDIO_DRY_OUT_REL if dry else STUDIO_OUT_REL)
    lock = Lock(os.path.join(REPO, LOCK_REL))
    lock_info = lock.acquire()
    try:
        if os.path.exists(out_dir):
            raise RuntimeError(f'refusing: output directory exists: {out_dir}')
        os.makedirs(os.path.join(out_dir, 'logs'))
        os.makedirs(os.path.join(out_dir, 'nl'))
        record = {'schema': f'p516_tperf_{mode}_v1', 'stage': f'P5.16 T-perf Part B ({mode})', 'authority': AUTHORITY,
                  'mode': mode, 'utc_start': _utc(), 'lock': lock_info,
                  'objective_convention': 'none computed or compared (sizes / timing / iterations / memory only)',
                  'instance': {'case': "p515_s44_scale_measurement.derive_case('paper', {}) of the committed case file",
                               'candidate': 'unit cell: node 7 0.25 MVA / 1.0 MWh, nodes 5 and 9 empty, cohort 2025',
                               'blocks': f'TSO {YEAR} {DAY}; DSO node {DSO_NODE} {YEAR} {DAY}; ESSO (0 solves)'},
                  'declared_solves': 0 if dry else DECLARED_SOLVES,
                  'thread_env_before': THREAD_ENV_BEFORE,
                  'thread_env_in_force': {k: os.environ.get(k) for k in THREAD_CAP_ENV}}
        record['provenance'] = {
            'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'hw_model': hw, 'mem_total_bytes': psutil.virtual_memory().total, 'mem_available_at_start': avail,
            'cpu_count_logical': psutil.cpu_count(), 'cpu_count_physical': psutil.cpu_count(logical=False),
            'git_head': _git(['rev-parse', 'HEAD']), 'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD']),
            'git_status_tracked': _git_status(['--untracked-files=no']),
            'production_files_sha256': {f: _sha(os.path.join(REPO, f)) for f in PRODUCTION_FILES},
            'harness_sha256': _sha(THIS), 'harness_git_status': _git_status(['--', 'p516_tperf_profile.py']),
            'other_compute_processes_at_start': foreign, 'loadavg_1_5_15_at_start': list(os.getloadavg()),
        }
        if not dry and record['provenance']['git_status_tracked']:
            raise RuntimeError(f"refusing: tracked files modified: {record['provenance']['git_status_tracked']}")
        (record['provenance']['git_status_build_inputs'], record['provenance']['repository_modules_loaded'],
         dirty_inputs) = _build_input_status()
        if dirty_inputs:
            raise RuntimeError(f'refusing: build inputs modified: {dirty_inputs}')
        if dry:
            record['predictions_recorded_before_run'] = STUDIO_DRY_PREDICTIONS
            _write_json(os.path.join(out_dir, 'predictions.json'),
                        {'utc': _utc(), 'harness_sha256': record['provenance']['harness_sha256'],
                         'predictions': STUDIO_DRY_PREDICTIONS})
        wd = MemWatchdog(STUDIO_DRY_RSS_ABORT_BYTES if dry else STUDIO_RSS_ABORT_BYTES,
                         os.path.join(out_dir, 'watchdog_abort.json'))
        wd.start()
        try:
            return _studio_body(dry, mode, out_dir, record, wd)
        except BaseException as error:
            record['status'] = 'error'
            record['error'] = f'{type(error).__name__}: {error}'
            record['traceback'] = traceback.format_exc()
            record['guard_counts_at_error'] = dict(_GUARD.counts)
            record['watchdog'] = wd.summary()
            try:
                _write_json(os.path.join(out_dir, f'tperf_{mode}_error.json'), record)
            except Exception:  # noqa: BLE001 -- the traceback reaches stderr through __main__
                pass
            raise
    finally:
        lock.release()


def _studio_body(dry, mode, out_dir, record, wd):
    stages = []
    stage = _stage_factory(stages, wd)
    record['memory'] = {'rss_at_start': _rss()}
    build_guard = None if dry else SolveProfileGuard((), label='P5.16 T-perf studio BUILD phase (zero solves)').install()
    build_t0 = time.perf_counter()
    record['build'] = {}
    try:
        studio_read(out_dir, 'paper', 'paper', stage, record)
        record['memory']['rss_after_read'] = _rss()
        built = studio_build_blocks(out_dir, 'p516_tperf_studio_dry_build' if dry else 'p516_tperf_studio_build', True,
                                    stage, record['build'])
    finally:
        if build_guard is not None:
            build_guard.uninstall()
            record['build_phase_blocking_guard'] = {'counts': dict(build_guard.counts),
                                                    'verify_0_failures': build_guard.verify(0)}
    record['build_wall_s'] = time.perf_counter() - build_t0
    record['build_stages'] = list(stages)
    record['stages_all'] = stages   # the same list, extended to the end of the run
    record['build_memory'] = {'rss_at_build_end': _rss(), 'self_ru_maxrss_at_build_end':
                              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'ru_units': 'bytes (macOS)',
                              'watchdog_peak_at_build_end': dict(wd.peak),
                              'rss_after_read_before_construct': record['memory']['rss_after_read']}
    record['guard_counts_at_build_end'] = dict(_GUARD.counts)
    build_zero = all(v == 0 for v in _GUARD.counts.values()) and (
        build_guard is None or (not record['build_phase_blocking_guard']['verify_0_failures']))
    record['build_produced_zero_solves'] = build_zero
    _log(f'build done in {record["build_wall_s"]:.1f}s; guard {_GUARD.counts}; zero solves: {build_zero}; '
         f'ru_maxrss {record["build_memory"]["self_ru_maxrss_at_build_end"] / GIB:.2f} GiB')
    if not build_zero:
        raise RuntimeError(f'build was not zero-solve: guard {_GUARD.counts} {record.get("build_phase_blocking_guard")}')

    pe, network_module, srp, G, S44 = _studio_modules()
    bases = built['bases']
    record['blocks'] = {}
    for block, model in bases.items():
        with stage(f'size {block} block (Pyomo vars / rows / nnz; the solved initialisation problem)'):
            net0, prm0 = _build_time_objects(model)
            size = block_size(model, pe)
            size['pyomo_nnz'] = pyomo_jacobian_nnz(model, pe)
            prepared_size = block_size(built['admm_prepared'][block], pe)
            record['blocks'][block] = {'network': net0.name, 'year': net0.year, 'day': net0.day,
                                       'baseMVA': net0.baseMVA, 'size_pyomo_built': size,
                                       'size_pyomo_admm_prepared_reference (not solved)': prepared_size}
        _log(f'{block}: vars {size["pyomo_vars"]} (free {size["pyomo_vars_free"]}), rows eq {size["pyomo_active_equalities"]}'
             f' ineq {size["pyomo_active_inequalities"]}, nnz(free) {size["pyomo_nnz"]["total_nnz_free_vars"]}, '
             f'scenarios {size["scenarios_market"]}x{size["scenarios_operation"]}')
    record['esso'] = {f'node_{n}': block_size_esso(m, pe) for n, m in built['esso'].items()}
    record['esso']['node_7_initialisation_problem'] = block_size_esso(built['esso_init'], pe)
    record['solved_problem'] = record['build'].get('solved_problem')
    del built
    gc.collect()
    record['memory']['rss_after_release_of_unused'] = _rss()
    if dry:
        d1 = record['blocks']['dso']['size_pyomo_built']['pyomo_vars']
        peak = record['build_memory']['self_ru_maxrss_at_build_end']
        record['predictions_outcome'] = {
            'D1_dso_block_variables': {'measured_pyomo_vars': d1, 'relative_to_269k': d1 / 269_000.0,
                                       'within_5pct': abs(d1 / 269_000.0 - 1.0) <= 0.05},
            'D2_build_rss': {'measured_self_ru_maxrss_bytes': peak, 'measured_gib': peak / GIB,
                             'watchdog_peak_rss_tree_bytes': record['build_memory']['watchdog_peak_at_build_end'][
                                 'rss_tree'], 'holds': peak <= 6 * GIB}}

    logs_dir = os.path.join(out_dir, 'logs')
    inst = Instrument(os.path.join(out_dir, 'nl'), dry).install()
    # Stand-ins for the pre-solve checklist: every case prepared exactly as it will be, from a clone of the BUILT
    # block (on the Studio the warm cases are re-prepared from the post-(ii) block; their suffix item is deferred
    # to the per-case check immediately before each warm solve).
    prepared = {}
    with stage('prepare 6 case stand-ins (clones of the built blocks)'):
        for block in BLOCKS:
            for case in STUDIO_CASE_ORDER:
                ck = f'{block}_{case}'
                t = time.perf_counter()
                model = bases[block].clone()
                clone_s = time.perf_counter() - t
                net, prm, entry = studio_prepare_case(model, ck, case, logs_dir, network_module, bases[block])
                entry['clone_s'] = clone_s
                prepared[ck] = {'block': block, 'case': case, 'model': model, 'network': net, 'params': prm,
                                'entry': entry}
    record['capture_checklist'] = capture_checklist(inst, prepared, network_module, out_dir, dry,
                                                    deferred=('warm_suffixes_present',))
    record['capture_checklist_studio'] = studio_extra_checklist(record, out_dir, wd, build_zero)
    record['memory']['rss_before_first_solve'] = _rss()
    _log('capture checklists hold (warm-suffix item deferred to the per-case check)')
    record['solves'] = {}
    stopped = None

    if dry:
        for ck, p in prepared.items():
            wd.stage = f'option assembly (blocked) {ck}'
            log_path = expected_log_path(p['network'], p['params'].solver_params.options)
            rec, ok = run_case(ck, p['block'], p['case'], p['model'], p['network'], p['params'], inst,
                               network_module, pe, logs_dir, True)
            rec['preparation'] = p['entry']
            rec['options_intended'] = intended_passed_options(p['entry']['production_options'], p['case'], log_path)
            rec['options_match_intended'] = rec['solver_options_passed'] == rec['options_intended']
            rec['options_diff'] = {k: [rec['options_intended'].get(k), (rec['solver_options_passed'] or {}).get(k)]
                                   for k in set(rec['options_intended']) | set(rec['solver_options_passed'] or {})
                                   if rec['options_intended'].get(k) != (rec['solver_options_passed'] or {}).get(k)}
            record['solves'][ck] = rec
            print(f'OPTIONS {ck} (passed by production, solve blocked; match_intended={rec["options_match_intended"]}):\n'
                  + json.dumps(rec['solver_options_passed'], indent=1, sort_keys=True), flush=True)
        record['nl_selftest'] = {}
        for block in BLOCKS:
            wd.stage = f'NL write (not a solve) {block}'
            nl_rec = {}
            dry_nl_selftest(f'{block}_built', prepared[f'{block}_ii_cold']['model'], out_dir, nl_rec)
            record['nl_selftest'][block] = nl_rec
        fails = []
        if _GUARD.counts['permitted_solve'] or _GUARD.counts['permitted_exec'] or _GUARD.counts['blocked_exec']:
            fails.append(f'dry run reached a permitted solve or a process launch {_GUARD.counts}')
        if _GUARD.counts['blocked_solve'] != len(prepared):
            fails.append(f"blocked {_GUARD.counts['blocked_solve']} != cases {len(prepared)}")
        if not all(r['options_match_intended'] for r in record['solves'].values()):
            fails.append('option assembly differs from intended: '
                         + str({ck: r['options_diff'] for ck, r in record['solves'].items() if r['options_diff']}))
        expected = 0
    else:
        del prepared
        gc.collect()
        for block in BLOCKS:
            solved_cold = None
            for case in STUDIO_CASE_ORDER:
                ck = f'{block}_{case}'
                source = bases[block] if case == 'ii_cold' else solved_cold
                t = time.perf_counter()
                model = source.clone()
                clone_s = time.perf_counter() - t
                net, prm, entry = studio_prepare_case(model, ck, case, logs_dir, network_module, source)
                entry['clone_s'] = clone_s
                cl = capture_checklist(inst, {ck: {'block': block, 'case': case, 'model': model, 'network': net,
                                                   'params': prm}}, network_module, out_dir, False)
                log_path = expected_log_path(net, prm.solver_params.options)
                wd.stage = f'solve {ck}'
                _log(f'case {ck}: run_smopf(from_warm_start={CASE_IS_WARM[case]})')
                rec, ok = run_case(ck, block, case, model, net, prm, inst, network_module, pe, logs_dir, False)
                rec['preparation'] = entry
                rec['capture_checklist_case'] = cl
                rec['options_intended'] = intended_passed_options(entry['production_options'], case, log_path)
                rec['options_match_intended'] = rec['solver_options_passed'] == rec['options_intended']
                record['solves'][ck] = rec
                s = (rec.get('ipopt_log') or {}).get('summary', {})
                _log(f'case {ck}: ok={ok} term={rec["termination_condition"]} iters={s.get("iterations")} '
                     f'run_smopf={rec["timing"]["run_smopf_wall_s"]:.3f}s '
                     f'peak_ipopt_rss={rec["rss_monitor"]["peak_child_rss"] / GIB:.2f} GiB')
                if not ok or rec['solve_calls_in_case'] != 1 or rec['cap_blocked_in_case']:
                    stopped = (f'solve failed or count off at {ck}: ok={ok} calls={rec["solve_calls_in_case"]} '
                               f'cap_blocked={rec["cap_blocked_in_case"]} -- stopping')
                    _log(stopped)
                    break
                if case == 'ii_cold':
                    solved_cold = model
            if stopped:
                break
        for ck, rec in record['solves'].items():
            if not rec['nl_capture']:
                rec['gradient_scaling'] = {'computed': False, 'reason': 'no NL captured'}
                continue
            try:
                rep = gradient_scaling_replica(rec['nl_capture'][0]['path'])
                rep['validation_against_ipopt_log'] = validate_replica(rep, rec['ipopt_log'])
                rep['computed'] = True
                rec['gradient_scaling'] = rep
            except Exception as error:  # noqa: BLE001 -- recorded; the solves are already done
                rec['gradient_scaling'] = {'computed': False, 'reason': f'{type(error).__name__}: {error}',
                                           'traceback': traceback.format_exc()}
        fails = _GUARD.verify(DECLARED_SOLVES)
        expected = DECLARED_SOLVES
    record['guard'] = {'permitted': [] if dry else [list(p) for p in PERMITTED], 'declared_solves': expected,
                       'counts': dict(_GUARD.counts), 'verify_failures': fails, 'per_case_cap_blocks': inst.cap_blocks,
                       'note': ('dry: every case reaches OptSolver.solve and is blocked there (6 expected); no NL '
                                'write by a solver, no IPOPT process' if dry else
                                'bounded guard, permitted site network.py:_run_smopf_solver_attempt, exact count')}
    record['stopped'] = stopped
    record['status'] = 'complete' if not fails and stopped is None else ('stopped' if stopped else 'guard_failed')
    try:
        record['summary'] = summarise(record) if not dry else None
    except Exception as error:  # noqa: BLE001 -- the per-solve records are complete without it
        record['summary'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    record['memory']['rusage_end'] = _rusage()
    record['loadavg_1_5_15_at_end'] = list(os.getloadavg())
    record['other_compute_processes_at_end'] = _foreign_compute_processes()
    record['watchdog'] = wd.stop()
    record['utc_end'] = _utc()
    record['wall_s'] = time.time() - T0
    name = f'tperf_{mode}.json'
    _write_json(os.path.join(out_dir, name), record)
    _log(f'status={record["status"]} guard={_GUARD.counts} verify_failures={fails} -> {name}')
    return 0 if record['status'] == 'complete' else 2


def run_restriction_equivalence():
    """Zero solves (blocking guard over the whole run). At SRP1 scale (derive_case('srp1', {}) = the committed case
    file unchanged; the full build is small), build the instance twice in one process from the same installed
    baseline -- once in full, once restricted exactly as --studio / --studio-dry restrict it -- and compare the
    TSO (YEAR, DAY) block, the DSO node-DSO_NODE (YEAR, DAY) block and the node-DSO_NODE ESSO between the two:
    sha256 of the NL / .row / .col files Pyomo writes with symbolic labels (NL writes, not solves) and the Pyomo
    sizes. The blocks NL-compared are the initialisation-problem captures (what --studio solves); the ADMM-prepared
    blocks are compared on Pyomo sizes (they cannot be NL-written: unset AL Params). Controls: a second write of the
    full DSO capture must hash identically (writer determinism) and differently from the TSO capture."""
    out_dir = os.path.join(REPO, STUDIO_EQUIV_OUT_REL)
    lock = Lock(os.path.join(REPO, LOCK_REL))
    lock_info = lock.acquire()
    try:
        if os.path.exists(out_dir):
            raise RuntimeError(f'refusing: output directory exists: {out_dir}')
        os.makedirs(os.path.join(out_dir, 'nl'))
        record = {'schema': 'p516_tperf_restriction_equivalence_v1', 'utc_start': _utc(), 'lock': lock_info,
                  'objective_convention': 'none computed or compared',
                  'provenance': {'git_head': _git(['rev-parse', 'HEAD']),
                                 'harness_sha256': _sha(THIS), 'hw_model': _git_free_cmd(['sysctl', '-n', 'hw.model'])}}
        (record['provenance']['git_status_build_inputs'], record['provenance']['repository_modules_loaded'],
         dirty) = _build_input_status()
        if dirty:
            raise RuntimeError(f'refusing: build inputs modified: {dirty}')
        wd = MemWatchdog(STUDIO_DRY_RSS_ABORT_BYTES, os.path.join(out_dir, 'watchdog_abort.json'))
        wd.start()
        stages = []
        stage = _stage_factory(stages, wd)
        pe, network_module, srp, G, S44 = _studio_modules()
        studio_read(out_dir, 'srp1', 'srp1_restriction_equivalence', stage, record)
        record['build_full'], record['build_restricted'] = {}, {}
        full = studio_build_blocks(out_dir, 'p516_tperf_equiv_full', False, stage, record['build_full'])
        restricted = studio_build_blocks(out_dir, 'p516_tperf_equiv_restricted', True, stage, record['build_restricted'])
        second_day = [d for d in full['tso_model'][YEAR] if d != DAY][0]
        # NL-compared: the initialisation-problem captures (what --studio solves). The ADMM-prepared blocks cannot
        # be NL-written (unset AL Params; see studio_build_blocks) and are compared on Pyomo sizes only.
        pairs = {'tso': (full['bases']['tso'], restricted['bases']['tso']),
                 'dso': (full['bases']['dso'], restricted['bases']['dso']),
                 'esso_node7': (full['esso_init'], restricted['esso_init'])}
        record['admm_prepared_size_comparison'] = {}
        for key in ('tso', 'dso'):
            sa = block_size(full['admm_prepared'][key], pe)
            sb = block_size(restricted['admm_prepared'][key], pe)
            record['admm_prepared_size_comparison'][key] = {'full': sa, 'restricted': sb, 'identical': sa == sb}
        sa, sb = block_size_esso(full['esso'][DSO_NODE], pe), block_size_esso(restricted['esso'][DSO_NODE], pe)
        record['admm_prepared_size_comparison']['esso_node7'] = {'full': sa, 'restricted': sb, 'identical': sa == sb}

        def write(model, stem):
            path = os.path.join(out_dir, 'nl', f'{stem}.nl')
            model.write(path, format='nl', io_options={'symbolic_solver_labels': True})
            return {ext: _sha(path[:-3] + ext) for ext in ('.nl', '.row', '.col')}

        record['comparison'] = {}
        for key, (a, b) in pairs.items():
            ha, hb = write(a, f'full_{key}'), write(b, f'restricted_{key}')
            sa = block_size(a, pe) if key != 'esso_node7' else block_size_esso(a, pe)
            sb = block_size(b, pe) if key != 'esso_node7' else block_size_esso(b, pe)
            record['comparison'][key] = {'full_sha256': ha, 'restricted_sha256': hb, 'nl_row_col_identical': ha == hb,
                                         'full_size': sa, 'restricted_size': sb, 'sizes_identical': sa == sb}
            _log(f'{key}: NL/row/col identical={ha == hb}; sizes identical={sa == sb}')
        # control: a block of the same build that must differ (the full build's ADMM-prepared TSO (YEAR, second
        # day) cannot be written, so the control is the full DSO init capture vs the full TSO init capture's NL)
        ctrl = write(full['bases']['dso'], 'full_dso_control_copy')
        record['discrimination_control'] = {
            'what': ('the full build\'s DSO init capture written a second time: must hash IDENTICALLY to its first '
                     'write (writer determinism) and DIFFERENTLY from the TSO init capture (the hash discriminates)'),
            'second_write_sha256': ctrl,
            'second_write_identical_to_first': ctrl == record['comparison']['dso']['full_sha256'],
            'differs_from_full_tso_spring_nl': ctrl['.nl'] != record['comparison']['tso']['full_sha256']['.nl'],
            'unused_second_day': second_day}
        record['guard'] = {'counts': dict(_GUARD.counts), 'verify_0_failures': _GUARD.verify(0)}
        ok = (all(c['nl_row_col_identical'] and c['sizes_identical'] for c in record['comparison'].values())
              and all(c['identical'] for c in record['admm_prepared_size_comparison'].values())
              and record['discrimination_control']['differs_from_full_tso_spring_nl']
              and record['discrimination_control']['second_write_identical_to_first']
              and not record['guard']['verify_0_failures'])
        record['status'] = 'equivalent' if ok else 'NOT_EQUIVALENT_or_guard_failed'
        record['stages'] = stages
        record['watchdog'] = wd.stop()
        record['utc_end'] = _utc()
        _write_json(os.path.join(out_dir, 'restriction_equivalence.json'), record)
        _log(f'status={record["status"]} guard={_GUARD.counts}')
        return 0 if ok else 2
    finally:
        lock.release()


# ======================================================================================================================
#  manifest
# ======================================================================================================================
MANIFEST_ROOTS = {'local': LOCAL_OUT_REL, 'studio': STUDIO_OUT_REL, 'studio_dry': STUDIO_DRY_OUT_REL,
                  'studio_dry_equiv': STUDIO_EQUIV_OUT_REL}


def manifest(root_name='local'):
    root = os.path.join(REPO, MANIFEST_ROOTS[root_name])
    target = os.path.join(root, 'manifest_sha256.json')
    if os.path.exists(target):
        raise RuntimeError(f'refusing to overwrite {target}')
    files = {}
    for dirpath, _, names in os.walk(root):
        for nm in sorted(names):
            p = os.path.join(dirpath, nm)
            files[os.path.relpath(p, REPO)] = _sha(p)
    for rel in LAUNCH_LOGS_REL[root_name]:
        if os.path.exists(os.path.join(REPO, rel)):
            files[rel] = _sha(os.path.join(REPO, rel))
    files[os.path.relpath(THIS, REPO)] = _sha(THIS)
    if root_name == 'local':
        for rel in INPUTS.values():
            files[rel[0]] = _sha(os.path.join(REPO, rel[0]))
    else:
        # Part B roots: the Part A pickles are not inputs (and need not exist on the Studio); the source case
        # file and its params file are.
        for rel in (os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_params.json')):
            files[rel] = _sha(os.path.join(REPO, rel))
    _write_json(target, {'schema': 'p516_tperf_manifest_v1', 'utc': _utc(), 'root': root_name, 'files': files}
                if root_name != 'local' else {'schema': 'p516_tperf_manifest_v1', 'utc': _utc(), 'files': files})
    print(f'manifest: {len(files)} files -> {target}')
    return 0


if __name__ == '__main__':
    try:
        if ARGS.manifest:
            sys.exit(manifest(ARGS.manifest_root or ('studio' if ARGS.studio_manifest else 'local')))
        if ARGS.studio:
            sys.exit(run_studio_mode(dry=False))
        if ARGS.studio_dry:
            sys.exit(run_studio_mode(dry=True))
        if ARGS.restriction_equivalence:
            sys.exit(run_restriction_equivalence())
        sys.exit(run_local(ARGS.dry_run))
    except Exception:
        traceback.print_exc()
        if _GUARD is not None:
            print(f'guard counts at error: {_GUARD.counts}', file=sys.stderr, flush=True)
        sys.exit(1)
