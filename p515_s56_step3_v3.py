"""P5.15 S56 -- step 3 v3: complete the 2025|Summer centralized comparator (singles 12 and 22, warm from the saved
merged-default slice) under ONE frozen spec (version 3; predecessor = the step 3 v2 / step 4 diagnostics spec
`frozen_step4_diag_spec_v2_ff466289.json`).

Authority: S56_PLAN_v3.md section 7 rows 3-4; TASKS.md S56 section (commit 9c88d89d: "Step 3 v3 (2025|Summer singles
12, 22 warm from the saved merged slice, 2 solves)"); P5_15_S56_STEP4_DIAG_REPORT.md decision 1 (a); the Planner's task
of 2026-11-23 (item 1).

THE ORDER (2 solves, declared in the spec and verified EXACTLY by armed guards):
  1. Read the 12-block SRP1-grid 3 x 3 instance with the step 3 harness's reader (as the v2 run did), merge
     (`combine_networks`), configure the solver exactly as step 3 v1 / v2 (the step 3 harness's `configure`: tol 1e-5,
     compl_inf_tol 1e-6, MA97, recovery off). Zero solves.
  2. Build the 2025|Summer merged model with the step 3 harness's `build` (run_v1's build: loads fixed, the ADMM-path
     scalar alignment, settlement weight 0, no row 18, no pin) and set every free Var from the SAVED primal file of the
     v2 run's S3-2 merged-default solve (`run_v1/primal/S3_2__logs__2025_Summer__merged_default.json.gz`, by name; its
     sha256 is pinned in the spec and re-verified before reading). Gate (recorded, not a stop): the objective at the
     loaded point equals the saved merged-default objective (185,674.1557530493).
  3. For scenario (1, 2) then (2, 2): slice the block (step 3 `slice_network`), build the single model (step 3 `build`),
     set every free Var from the loaded merged model at the same index with that scenario (scenario positions read
     from the declarations, step 3 `declared_positions` / `component_positions` -- the v2 run's S3-1 mapping, verbatim
     in method); no duals; from_warm_start=False; recovery off; solve once through `_solve_declared` (the step 3
     harness's `_solve_one`, wrapped). Every solve saves its primal values (gzip JSON, hashed).        2 solves
  4. Zero solves: 2025|Summer C_central = sum_s omega_s C_s (the v2 run's 7 Optimal singles + these two), step 3 P3 /
     P4 for the block (S3-2's merged warm / merged default against C_central), and the interface values of the central
     point (written for the step 4 smoke's D-3 of 2025|Summer).

GUARDS. SolveProfileGuard(permitted = this file's `_solve_declared`) is installed before any project import; the step 3
harness's own guard (permitting its `_solve_one`) is installed on top when it is imported and counts the same solves.
Both verified at exactly 2 solves and 2 launches (0 in --trial). pickle.load / loads blocked. Single-instance lock; the
repo-root campaign lock and any other running p5*.py process refuse the launch; the output directory must not exist.

MODES
  --freeze --out-dir D            writes the frozen spec into D (zero solves).
  --trial --spec S --out-dir D    zero solves: everything except the two solves (the mapping, the gate at the loaded
                                  point, the primal save path).
  --run --spec S --out-dir D      the declared run.
Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_step3_v3.py --run --spec S --out-dir D
       > <stdout log> 2> <stderr log>
Objective convention (every value): gross -- the step 3 merged / single-scenario `model.objective` (settlement weight
0, no row 18, no pin, no salvage), EUR per representative day, probability-weighted for C_central.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PERMITTED = ((SCRIPT, '_solve_declared'),)
GUARD = SolveProfileGuard(permitted=PERMITTED, label='P5.15 S56 step 3 v3 (2025|Summer singles 12, 22)').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (this stage reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

SOLVER_PATH = '/usr/local/bin/ipopt'
CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
SPEC_PREFIX = 'frozen_step3_spec_v3_'
PRED_SPEC = os.path.join(RES, 'step4_diag_v1', 'frozen_step4_diag_spec_v2_ff466289.json')
V2_RESULTS = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'step4_diag_results.json')
V2_MERGED_PRIMAL = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'primal',
                                'S3_2__logs__2025_Summer__merged_default.json.gz')
STEP3_HARNESS = 'p515_s56_central_ms_x0.py'
BLOCK = '2025|Summer'
SCENARIOS = ('12', '22')
GiB = 1024.0 ** 3

PREDICTIONS = {
    'S3v3_P1': 'both solves Optimal;',
    'S3v3_P2': 'each objective = the merged per-scenario term minus an offset in [0.030, 0.040] EUR;',
    'S3v3_P3': 'P3/P4 hold (1e-6 / 1e-5).',
}
PREDICTIONS_OPERATIONAL = {
    'S3v3_P1': ("per single (12, 22): solution loaded (production solver_result_succeeded), termination 'optimal', "
                "IPOPT exit 'Optimal Solution Found.' (step 3 harness `optimal`)"),
    'S3v3_P2': ('per single s: offset_s = (v2 run S3-2 merged_default costs.per_scenario[s].objective_per_scenario) - '
                '(single s objective); 0.030 <= offset_s <= 0.040'),
    'S3v3_P3': ('C_central = sum_s omega_s C_s over the 9 singles of 2025|Summer (the v2 run S3-2 singles 00, 01, 02, 10, '
                '11, 20, 21 + this stage\'s 12, 22; omega_s = the singles\' recorded probability); P3: |S3-2 merged_warm '
                '- C_central| / |C_central| <= 1e-6; P4: |S3-2 merged_default - C_central| / |C_central| <= 1e-5; both '
                'must hold'),
}
FAILURE_POLICY = ('Failure of any prediction is recorded, never re-run or tuned. A non-Optimal solve is recorded; then '
                  'C_central is undefined and the step 4 smoke proceeds without it for 2025|Summer (Planner task, item '
                  '1). A guard-count mismatch or a harness exception is a harness fault: recorded, and the stage stops '
                  'for the Planner.')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 s3v3] {msg}', flush=True)


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


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else math.inf)


# ---- primal values: the v2 launcher's helpers (p515_s56_step4_diag.save_primal / load_primal), same format. Copied,
# not imported: importing that launcher would install its guard, which permits only its own solve sites.
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
    """free Var <- the saved value of the same name (fixed Vars untouched). As p515_s56_step4_diag.warm_from_primal."""
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
#  the frozen spec
# ======================================================================================================================
def build_spec():
    with open(PRED_SPEC, encoding='utf-8') as handle:
        pred = json.load(handle)
    with open(V2_RESULTS) as handle:
        v2 = json.load(handle)
    s32 = v2['solves']['S3_2|2025|Summer']
    md = s32['merged_default']
    terms = {k: md['costs']['per_scenario'][k]['objective_per_scenario'] for k in SCENARIOS}
    singles_v2 = {k: {'optimal': v.get('optimal'), 'objective': (v.get('costs') or {}).get('objective'),
                      'probability': v.get('probability'), 'ipopt_exit': (v.get('ipopt') or {}).get('exit')}
                  for k, v in s32['single'].items()}
    return {
        'stage': 'P5.15 S56 step 3 v3: 2025|Summer singles 12 and 22 warm from the saved merged-default slice; '
                 '2025|Summer C_central and step 3 P3 / P4 (frozen spec v3)',
        'version': 3,
        'predecessor': {'path': PRED_SPEC, 'sha256': _sha(PRED_SPEC),
                        'relation': 'step 3 v2 + step 4 diagnostics spec (run_v1: S3-2 singles 12, 22 maxIter); this '
                                    'spec completes S3-2 with two warm single solves'},
        'authority': ['S56_PLAN_v3.md section 7 rows 3-4',
                      'TASKS.md S56 section, commit 9c88d89d (step 3 v3 active)',
                      'P5_15_S56_STEP4_DIAG_REPORT.md decision 1 (a)',
                      'Planner task 2026-11-23, item 1 (step 3 v3)'],
        'launcher': SCRIPT,
        'step3_harness': {'path': STEP3_HARNESS, 'sha256': _sha(STEP3_HARNESS),
                          'pinned_by_predecessor': pred['step3_harness']['sha256'],
                          'reused_functions': ['build', 'configure', 'slice_network', 'capture_solution',
                                               'component_positions', 'declared_positions', 'relaxation_limits',
                                               'block_weight', 'optimal', '_with_scen', '_strip', '_solve_one (wrapped)',
                                               'S.read_planning_from_derived_case', '_Stages']},
        'instance_12_block': dict(pred['instance_12_block']),
        'step3_v1_settings_reused': dict(pred['step3_v1_settings_reused']),
        'block': BLOCK, 'scenarios': {'12': [1, 2], '22': [2, 2]},
        'v2_results': {'path': V2_RESULTS, 'sha256': _sha(V2_RESULTS),
                       'merged_default_objective': md['costs']['objective'],
                       'merged_default_objective_hex': float.hex(md['costs']['objective']),
                       'merged_warm_objective': s32['merged_warm']['costs']['objective'],
                       'merged_per_scenario_terms': terms, 'singles': singles_v2},
        'warm_source': {'path': V2_MERGED_PRIMAL, 'sha256': _sha(V2_MERGED_PRIMAL),
                        'recorded_in_v2_results': md['solve']['primal']},
        'start': ('freshly built merged 2025|Summer model (step 3 build); every free Var set by name from the warm source '
                  '(the v2 S3-2 merged-default primal values); then per single: sliced (step 3 slice_network), built '
                  '(step 3 build), every free Var set from the loaded merged model at the same index with that scenario '
                  '(positions from the declarations, as the v2 S3-1 mapping); no duals; from_warm_start=False'),
        'recovery': 'off (step 3 configure; one attempt per solve)',
        'solves': {'parent': 2, 'site': '_solve_declared', 'also_counted_by': 'the step 3 harness guard (its _solve_one)',
                   'derivation': 'one Network.run_smopf per single (12, 22) through the step 3 harness _solve_one'},
        'gates_recorded': {
            'loaded_point_objective': ('pe.value(merged.objective) at the loaded point vs the saved merged-default '
                                       'objective: recorded (bitwise and relative); a failure is recorded, not a stop'),
            'warm_source_hash': 'sha256 of the warm source == spec, before reading (refuse otherwise, zero solves)'},
        'predictions': PREDICTIONS, 'predictions_operational': PREDICTIONS_OPERATIONAL,
        'thresholds': {'offset_eur': [0.030, 0.040], 'P3_rel': 1e-6, 'P4_rel': 1e-5},
        'failure_policy': FAILURE_POLICY,
        'not_permitted': ['production edits', 'any solve not declared', 'overwriting committed artifacts',
                          'concurrent or detached runs', 'recovery'],
        'outputs': {'results': 'step3_v3_results.json', 'central': 'central_2025_Summer.json (C_central, parts, the '
                                                                   'central point interface P/Q/V per scenario and DN)',
                    'primal': 'primal/<tag>.json.gz', 'manifest': 'manifest_sha256.json'},
        'objective_convention': ('gross: step 3 model.objective (settlement weight 0, no row 18, no pin, no salvage), '
                                 'EUR per representative day; C_central probability-weighted'),
        'output': os.path.join(RES, 'step3_v3', 'run_v1'),
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


# ======================================================================================================================
#  the run
# ======================================================================================================================
S3 = None
ORIG_S3_SOLVE_ONE = None
CTX = {'trial': False, 'out_dir': None}


def _solve_declared(nd, m, params, tag):
    """THE solve site: calls the step 3 harness's `_solve_one` (its guard permits it), then saves the primal values.
    In --trial it never solves (the primal save path is still exercised)."""
    if CTX['trial']:
        st = {'trial_not_solved': True, 'solution_loaded': False, 'termination_condition': None,
              'ipopt_records': [], 'wall_s': 0.0}
    else:
        st = ORIG_S3_SOLVE_ONE(nd, m, params)
    st['tag'] = tag
    try:
        st['primal'] = save_primal(m, os.path.join(CTX['out_dir'], 'primal', f'{tag}.json.gz'),
                                   {'tag': tag, 'solution_loaded': st.get('solution_loaded'),
                                    'termination_condition': st.get('termination_condition'), 'trial': CTX['trial']})
    except Exception:  # noqa: BLE001 -- recorded; never aborts the declared solve set
        st['primal_error'] = traceback.format_exc()
    last = (st.get('ipopt_records') or [{}])[-1]
    _log(f"solve {tag}: {st.get('termination_condition')} {last.get('exit')} it {last.get('iterations')} "
         f"wall {st.get('wall_s', 0):.1f}s")
    return st


def _ipopt_last(st):
    r = (st.get('ipopt_records') or [{}])[-1]
    return {k: r.get(k) for k in ('exit', 'iterations', 'mu_final', 'mu_floor', 'floor_status', 'obj_scaling_factor',
                                  'compl_inf_tol_in_force', 'tol_in_force')}


def run(args):
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
    lock = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), '.step3_v3.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[S56 s3v3] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        return 1
    try:
        return run_locked(args, spec_rel, spec_sha, trial)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def run_locked(args, spec_rel, spec_sha, trial):
    global S3, ORIG_S3_SOLVE_ONE
    t_start = time.perf_counter()
    with open(args.spec, encoding='utf-8') as handle:
        spec = json.load(handle)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir)
    CTX.update(trial=trial, out_dir=out_dir)
    res = {'stage': spec['stage'], 'spec': {'path': spec_rel, 'sha256': spec_sha}, 'script': SCRIPT,
           'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(), 'utc_start': _utc(),
           'trial': trial, 'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'objective_convention': spec['objective_convention'], 'checks': [], 'error': None, 'solves': {}}
    checks = res['checks']

    def ck(name, ok, detail=None):
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:500]}' if detail is not None and not ok else ''))
        return bool(ok)
    try:
        import p515_s56_central_ms_x0 as S3mod
        S3 = S3mod
        ORIG_S3_SOLVE_ONE = S3.__dict__['_solve_one']
        import pyomo.environ as pe
        from centralized_coordination import combine_networks
        ck('step 3 harness sha256 == spec pin == predecessor pin',
           _sha(STEP3_HARNESS) == spec['step3_harness']['sha256'] == spec['step3_harness']['pinned_by_predecessor'])
        ck('predecessor spec sha256 == spec', _sha(spec['predecessor']['path']) == spec['predecessor']['sha256'])
        ck('v2 results sha256 == spec', _sha(spec['v2_results']['path']) == spec['v2_results']['sha256'])
        ck('warm source sha256 == spec == v2 record',
           _sha(spec['warm_source']['path']) == spec['warm_source']['sha256']
           == spec['warm_source']['recorded_in_v2_results']['sha256'])
        for name in ('build', 'configure', 'slice_network', 'capture_solution', 'component_positions',
                     'declared_positions', 'relaxation_limits', 'block_weight', 'optimal', '_with_scen', '_strip'):
            assert callable(getattr(S3, name, None)), name
        with open(spec['v2_results']['path']) as handle:
            v2 = json.load(handle)
        s32 = v2['solves']['S3_2|2025|Summer']
        inst = spec['instance_12_block']
        ck('12-block case sha256 == declaration', _sha(inst['case_path']) == inst['case_sha256'])
        with open(spec['predecessor']['path'], encoding='utf-8') as handle:
            pred = json.load(handle)
        with open(pred['predecessor']['path'], encoding='utf-8') as handle:
            s3spec = json.load(handle)   # the step 3 v1 spec (4017dc70): solver, alignment, reported thresholds
        ck('step 3 v1 spec sha256 == predecessor record', _sha(pred['predecessor']['path'])
           == pred['predecessor']['sha256'])
        ck('step 3 v1 settings == the v3 spec declaration',
           {k: s3spec[k] for k in spec['step3_v1_settings_reused']} == spec['step3_v1_settings_reused'])
        planning = S3.S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                                        os.path.join(out_dir, 'instance_read'), S3._Stages())
        cks = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
        ck('12-block combined scenario checksum == declaration', cks == inst['scenario_checksum'], cks)
        tn, dns = planning.transmission_network, planning.distribution_networks
        dn_ids = list(dns)
        merged = combine_networks(tn, dns)
        params = S3.configure(merged.params, s3spec)
        res['solver_options_in_force'] = dict(params.solver_params.options)
        res['recovery'] = [params.solver_params.recovery_enabled, params.solver_params.recovery_tier2_enabled]
        ck('solver options == the v2 run (in force)',
           dict(params.solver_params.options) == v2['solver_options_in_force'],
           [dict(params.solver_params.options), v2['solver_options_in_force']])
        ck('recovery off', res['recovery'] == [False, False])
        ck('solver_params.verbose is False', params.solver_params.verbose is False)
        y = next(yy for yy in tn.years if str(yy) == BLOCK.split('|')[0])
        d = BLOCK.split('|')[1]
        md = merged.network[y][d]
        res['block_weight'] = S3.block_weight(tn, y, d)
        limits = S3.relaxation_limits(tn, dns, y, d)
        decl = S3.declared_positions()
        # ---- the merged model at the saved point (zero solves)
        m, brec = S3.build(md, params, s3spec)
        ck('2025|Summer merged: alpha 0, no pin, settlement 0, no ADN / shared ESS', brec['alpha0_nopin_settlement0_ok'])
        payload = load_primal(spec['warm_source']['path'])
        wm = warm_from_primal(m, payload)
        n_free = sum(1 for v in m.component_data_objects(pe.Var, active=True, descend_into=True) if not v.fixed)
        obj_loaded = float(pe.value(m.objective))
        saved = spec['v2_results']['merged_default_objective']
        res['merged_loaded'] = {'build': {k: v for k, v in brec.items() if k != 'formulation'}, 'warm_mapping': wm,
                                'n_free_vars': n_free, 'payload_n_vars': payload['n_vars'],
                                'payload_n_none': payload['n_none'], 'objective_at_loaded_point': obj_loaded,
                                'objective_at_loaded_point_hex': float.hex(obj_loaded),
                                'saved_merged_default_objective': saved,
                                'bitwise_equal': obj_loaded == saved, 'rel_diff': _rel(obj_loaded, saved),
                                'max_violation_at_loaded_point': S3.max_violation(m)}
        ck('loaded point: objective == saved merged-default objective (recorded; rel <= 1e-12)',
           _rel(obj_loaded, saved) <= 1e-12, [obj_loaded, saved])
        ck('loaded point: no saved name missing from the fresh build', wm['missing'] == 0, wm)
        vpos = S3.component_positions(m, decl, pe.Var)
        failed = [c['check'] for c in checks if not c['ok'] and not c['check'].startswith('loaded point: objective')]
        res['refused_before_any_solve'] = failed
        if failed:
            _log(f'REFUSED before any solve: {failed}')
        else:
            CTX['solve_phase_started'] = True
            logs = os.path.join(out_dir, 'logs', BLOCK.replace('|', '_'))
            for key in SCENARIOS:
                s_m, s_o = spec['scenarios'][key]
                nd, srec = S3.slice_network(md, s_m, s_o)
                ms, brec_s = S3.build(nd, params, s3spec)
                wmap = {'set': 0, 'skipped_fixed': 0, 'unmapped': 0, 'none_value': 0, 'outside_bounds': 0}
                for comp in ms.component_objects(pe.Var, active=True):
                    pos = vpos.get(comp.name)
                    mc = getattr(m, comp.name)
                    for idx, v in comp.items():
                        if v.fixed:
                            wmap['skipped_fixed'] += 1
                            continue
                        if pos in (None, 'unknown', (None, None)):
                            wmap['unmapped'] += 1
                            continue
                        val = mc[S3._with_scen(S3._strip(idx, pos), pos, s_m, s_o)].value
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
                nd.logs_dir = os.path.join(logs, f'single_{key}')
                os.makedirs(nd.logs_dir)
                st = _solve_declared(nd, ms, params, f'S3v3__2025_Summer__single_{key}')
                r = {'slice': srec, 'build': {k: v for k, v in brec_s.items() if k != 'formulation'},
                     'formulation_ok': brec_s['alpha0_nopin_settlement0_ok'], 'warm_mapping': wmap,
                     'at_mapped_point': at, 'solve': st, 'optimal': S3.optimal(st) if not trial else False,
                     'ipopt': _ipopt_last(st), 'probability': md.prob_market_scenarios[s_m] * md.prob_operation_scenarios[s_o]}
                if st.get('solution_loaded') or trial:
                    r.update(S3.capture_solution(ms, nd, params, s3spec, dn_ids, limits))
                res['solves'][f'single_{key}'] = r
                del ms, nd
            if not trial:
                res['central'] = central(res, spec, s32)
    except Exception:  # noqa: BLE001 -- recorded; results and guard verification still written
        res['error'] = traceback.format_exc()
        _log('EXCEPTION:\n' + res['error'])
    # declared count: 0 in --trial, when refused, or when an exception came before the solve phase; otherwise 2 (an
    # exception inside the solve phase still owes the declared count, so a shortfall fails the guard loudly)
    exp = spec['solves']['parent'] if (CTX.get('solve_phase_started') and not trial) else 0
    gv = GUARD.verify(exp)
    site_ok = dict(GUARD.permitted_sites) == ({f'{SCRIPT}:_solve_declared': exp} if exp else {})
    res['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                    'expected': exp, 'verify_failures': gv, 'sites_exact': site_ok}
    if S3 is not None:
        gv3 = S3.GUARD.verify(exp)
        res['guard_step3_harness'] = {'label': S3.GUARD.label, 'counts': dict(S3.GUARD.counts),
                                      'permitted_sites': dict(S3.GUARD.permitted_sites), 'expected': exp,
                                      'verify_failures': gv3}
        res['pickle_step3_module'] = dict(S3.PK)
    res['pickle'] = dict(PK)
    res['guards_ok'] = bool(not gv and site_ok and not (res.get('guard_step3_harness') or {}).get('verify_failures')
                            and PK == {'load': 0, 'loads': 0})
    mods = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f).startswith(REPO + os.sep) and f.endswith('.py'):
            rel = os.path.relpath(os.path.abspath(f), REPO)
            mods[rel] = {'sha256': _sha(rel), 'committed_clean': committed_clean(rel)}
    res['imported_repo_modules'] = mods
    res['dirty_imported_modules'] = sorted(k for k, v in mods.items() if not v['committed_clean'])
    res['peak_rss'] = rss()
    res['total_wall_s'] = time.perf_counter() - t_start
    if not trial and not res.get('refused_before_any_solve') and res.get('error') is None:
        try:
            res['predictions'] = evaluate(res, spec)
        except Exception:  # noqa: BLE001
            res['predictions_error'] = traceback.format_exc()
    res['utc_end'] = _utc()
    path = os.path.join(out_dir, 'step3_v3_results.json')
    write_json(path, res)
    if res.get('central') and res['central'].get('complete'):
        write_json(os.path.join(out_dir, 'central_2025_Summer.json'), res['central_with_interface'])
    res.pop('central_with_interface', None)
    outputs = {}
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            outputs[os.path.relpath(p, out_dir)] = _sha(p)
    write_json(os.path.join(out_dir, 'manifest_sha256.json'),
               {'outputs': outputs, 'spec': {'path': spec_rel, 'sha256': spec_sha},
                'script': {'path': SCRIPT, 'sha256': res['script_sha256']},
                'imported_repo_modules': {k: v['sha256'] for k, v in mods.items()}})
    _log(f"wrote {path}; guards_ok {res['guards_ok']}; error {res['error'] is not None}; "
         f"predictions {({k: v.get('held') for k, v in (res.get('predictions') or {}).items()})}")
    if res.get('refused_before_any_solve'):
        return 3
    return 0 if (res['guards_ok'] and res['error'] is None) else 2


def central(res, spec, s32):
    singles, probs, src = {}, {}, {}
    for k, v in s32['single'].items():
        if k in SCENARIOS:
            v = res['solves'][f'single_{k}']
            src[k] = 'step 3 v3 (this run)'
        else:
            src[k] = 'v2 run S3-2'
        singles[k] = v
        probs[k] = v['probability']
    complete = (len(singles) == 9 and all(s.get('optimal') and 'costs' in s and 'interface' in s
                                          for s in singles.values()))
    out = {'block': BLOCK, 'complete': complete, 'sources': src, 'probabilities': probs,
           'probability_sum': sum(probs.values()), 'w_b': res['block_weight']['weight'],
           'not_optimal': sorted(k for k, s in singles.items() if not s.get('optimal'))}
    if complete:
        out['C_central'] = sum(probs[k] * singles[k]['costs']['objective'] for k in singles)
        out['C_central_hex'] = float.hex(out['C_central'])
        out['C_s'] = {k: singles[k]['costs']['objective'] for k in sorted(singles)}
        out['parts'] = {'C_gen_plus_flex': sum(probs[k] * singles[k]['costs']['C_gen_plus_flex'] for k in singles),
                        'slack_terms': sum(probs[k] * singles[k]['costs']['total_slack_penalties'] for k in singles)}
        out['weighted_eur'] = out['w_b'] * out['C_central']
        res['central_with_interface'] = dict(out, interface={k: singles[k]['interface']['00'] for k in sorted(singles)},
                                             interface_note=('the central point: each single-scenario solution\'s '
                                                             'interface P_mw / Q_mvar / V_pu per DN (step 3 interface '
                                                             'capture, key 00 of the single model)'))
    return out


def evaluate(res, spec):
    th = spec['thresholds']
    pr = {}
    sol = res['solves']
    opt = {k: bool((sol.get(f'single_{k}') or {}).get('optimal')) for k in SCENARIOS}
    pr['S3v3_P1'] = {'statement': spec['predictions']['S3v3_P1'], 'per_single': opt,
                     'ipopt': {k: (sol.get(f'single_{k}') or {}).get('ipopt') for k in SCENARIOS},
                     'held': all(opt.values())}
    per = {}
    for k in SCENARIOS:
        s = sol.get(f'single_{k}') or {}
        term = spec['v2_results']['merged_per_scenario_terms'][k]
        if 'costs' in s:
            off = term - s['costs']['objective']
            per[k] = {'merged_term': term, 'single_objective': s['costs']['objective'], 'offset_eur': off,
                      'held': th['offset_eur'][0] <= off <= th['offset_eur'][1]}
        else:
            per[k] = {'merged_term': term, 'held': False, 'reason': 'single not loaded'}
    pr['S3v3_P2'] = {'statement': spec['predictions']['S3v3_P2'], 'range': th['offset_eur'], 'per_single': per,
                     'held': all(v['held'] for v in per.values())}
    c = res.get('central') or {}
    if c.get('complete'):
        mw, mdft = spec['v2_results']['merged_warm_objective'], spec['v2_results']['merged_default_objective']
        p3, p4 = _rel(mw, c['C_central']), _rel(mdft, c['C_central'])
        pr['S3v3_P3'] = {'statement': spec['predictions']['S3v3_P3'], 'C_central': c['C_central'],
                         'merged_warm': mw, 'merged_default': mdft,
                         'P3_rel': p3, 'P3_diff_eur': mw - c['C_central'], 'P3_held': p3 <= th['P3_rel'],
                         'P4_rel': p4, 'P4_diff_eur': mdft - c['C_central'], 'P4_held': p4 <= th['P4_rel'],
                         'held': p3 <= th['P3_rel'] and p4 <= th['P4_rel']}
    else:
        pr['S3v3_P3'] = {'statement': spec['predictions']['S3v3_P3'], 'held': False,
                         'reason': f"C_central undefined (not optimal: {c.get('not_optimal')})"}
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--trial', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--spec')
    ap.add_argument('--out-dir')
    a = ap.parse_args()
    if a.freeze and a.out_dir:
        return freeze(a)
    if (a.trial or a.run) and a.spec and a.out_dir and not (a.trial and a.run):
        return run(a)
    ap.error('one of --freeze --out-dir D | --trial/--run --spec S --out-dir D')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
