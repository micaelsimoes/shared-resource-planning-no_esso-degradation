"""
P5.15 Addendum 32 Q3, task W30 -- NOMAD stub check of the OrthoMADS DIRECTION / MESH machinery
(STEP4_DFO_METHOD.md section 6 check (ii)) against the in-house OrthoMADS of p515_s47_phase_b_record.py.

RUNNABLE ONLY FROM THE SEPARATE ENVIRONMENT (a python3.12 venv holding PyNomadBBO 4.6.0, pyomo 6.9.5, numpy),
NEVER from the canonical opf_env_py311: the script refuses to run if `sys.prefix` is the canonical env or if
PyNomad does not import. See README.md in this directory for the environment and the exact command.

ZERO ADMM / OPF solves: the objective is a cheap analytic stub on the SAME 7-variable granular lattice as the
in-house Phase B (z = (zP5, zE5, zP7, zE7, zP9, zE9, zY), P = 0.25 zP, E = 0.5 zE, zY = year index), with the
same closed-form constraints (the in-house `Lattice.reasons`: bounds, P = 0 <=> E = 0, 2P <= E <= 4P, E <= 5,
year in 0..2, budget). The in-house module installs its own SolveProfileGuard(permitted=()) at import.

WHAT IS COMPARED
  M1  Direction / mesh machinery, poll by poll, from NOMAD's own debug display (DISPLAY_DEGREE 4):
      M1a  NOMAD's Householder columns == I - 2 v v^T of NOMAD's printed unit-sphere v (the in-house formula
           of `householder_columns`, applied to NOMAD's v instead of the Halton v).
      M1b  NOMAD's "scaled and mesh projected" directions == in-house `project_direction(h, Delta)` applied to
           NOMAD's printed columns h at NOMAD's frame size Delta (mesh size 1).
      M1c  NOMAD's generated points == clip(center + d, bounds) (NOMAD snaps to bounds; the in-house rejects).
      M1d  NOMAD's (n+1)-th (second-pass) direction == - sum of the post-snap displacements of the n first-pass
           points kept by NOMAD's rank reduction (greedy rank over NOMAD's printed sorted order), NOT re-scaled.
      M1e  Frame / mesh size sequence (NOMAD) vs Delta sequence (in-house).
  M2  Terminal point on two stubs (A: F = I(x), positive slopes; B: F = I(x) + convex quadratic), NOMAD vs
      in-house `run_mads` (the production function, unchanged), both from the same start, vs the lattice
      minimum by enumeration of `Lattice.domain()`.
"""
import os
import sys

sys.dont_write_bytecode = True  # never write cpython-312 .pyc into the repository's __pycache__

CANONICAL_PREFIX = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311'
if os.path.realpath(sys.prefix) == os.path.realpath(CANONICAL_PREFIX):
    raise SystemExit('REFUSED: this stub runs only from the separate NOMAD environment, never the canonical env')
try:
    import PyNomad  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f'REFUSED: PyNomad not importable ({exc}); run from the separate NOMAD environment')

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, REPO)
import p515_s47_phase_b_record as R  # noqa: E402  (installs its zero-solve guard)

OUT = HERE
N = R.N_VARS
YEARS = (2025, 2030, 2035)
# stub unit costs (EUR per MVA / per MWh), positive, cheaper in later years -- NOT the SRP1 costs
STUB_COSTS = {2025: {'power': 400.0, 'energy': 300.0}, 2030: {'power': 320.0, 'energy': 240.0},
              2035: {'power': 260.0, 'energy': 190.0}}
STUB_BUDGET = 1e12  # inactive
LB = [0, 0, 0, 0, 0, 0, 0]
UB = [R.ZE_MAX] * 6 + [len(YEARS) - 1]
Z_START = (4, 8, 4, 8, 4, 8, 0)  # 1 MVA / 4 MWh at every node, 2025 (C* rounded to the lattice)
TARGET_P = (2, 6, 0)
TARGET_E = (6, 9, 0)
NOMAD_SEEDS = (1, 2, 3)
MAX_ITERATIONS_DEBUG = 12
LAT = R.Lattice(YEARS, STUB_COSTS, budget=STUB_BUDGET)


def q_stub(name, z):
    if name == 'A_affine':
        return 0.0
    q = 0.0
    for i in range(3):
        q += 100.0 * ((z[2 * i] - TARGET_P[i]) ** 2 + (z[2 * i + 1] - TARGET_E[i]) ** 2)
    return q + 50.0 * (z[-1] - 1) ** 2


def f_stub(name, z):
    return LAT.investment_cost(z) + q_stub(name, z)


# ---------------------------------------------------------------------------------------------------------------
# in-house: the production run_mads with a stub evaluate_fn
# ---------------------------------------------------------------------------------------------------------------
def run_inhouse(name, completion_cap=None):
    saved = R.COMPLETION_CAP
    if completion_cap is not None:
        R.COMPLETION_CAP = completion_cap  # in-process only, for the stub arm labelled so
    try:
        cache, n_calls = {}, [0]

        def key_of(z):
            return LAT.label(LAT.canonical_z(z))

        def evaluate_fn(zs):
            n_calls[0] += 1
            out = []
            for z in zs:
                zc = LAT.canonical_z(z)
                out.append({'eval_key': key_of(zc), 'status': 'certified', 'Q': q_stub(name, zc), 'bar': 0.0,
                            'canonical': None, 'source': 'stub'})
            return out

        z0 = LAT.canonical_z(Z_START)
        inc = {'eval_key': key_of(z0), 'z': z0, 'label': LAT.label(z0), 'I': LAT.investment_cost(z0),
               'Q': q_stub(name, z0), 'F': f_stub(name, z0), 'bar': 0.0, 'source': 'stub'}
        cache[inc['eval_key']] = {'eval_key': inc['eval_key'], 'status': 'certified', 'Q': inc['Q'], 'bar': 0.0}
        polls = []
        term = R.run_mads(LAT, cache, key_of, inc, evaluate_fn, sigma_q=0.0, delta0=R.DELTA_0,
                          max_new_evaluations=10 ** 6, max_polls=R.MAX_POLLS, batch_size=R.CONCURRENCY,
                          log=lambda m: None, on_poll=polls.append)
        seq = [{'poll': p['poll_index'], 'halton_t': p['halton_t'], 'Delta': p['poll_size_delta'],
                'incumbent': p['incumbent']['label'], 'incumbent_z': list(p['incumbent']['z']),
                'decision': p['decision'], 'directions': p['directions'],
                'n_candidates': len(p['candidates']),
                'n_feasible': sum(1 for c in p['candidates'] if c['feasible']),
                'n_completion': (p['completion'] or {}).get('n_feasible')} for p in polls]
        return {'stub': name, 'completion_cap': R.COMPLETION_CAP, 'termination': term['termination'],
                'certificate': term['termination_certificate'], 'incumbent': term['incumbent'],
                'n_polls': term['n_polls'], 'n_new_evaluations': term['n_new_evaluations'], 'polls': seq}
    finally:
        R.COMPLETION_CAP = saved


# ---------------------------------------------------------------------------------------------------------------
# NOMAD
# ---------------------------------------------------------------------------------------------------------------
def nomad_params(seed, anisotropic, display_degree, max_iterations=None):
    extra = [f'MAX_ITERATIONS {max_iterations}'] if max_iterations else []
    return extra + [f'ANISOTROPIC_MESH {"yes" if anisotropic else "no"}', 'DIMENSION 7', 'BB_INPUT_TYPE (I I I I I I I)', 'BB_OUTPUT_TYPE OBJ EB',
            'LOWER_BOUND ( ' + ' '.join(map(str, LB)) + ' )', 'UPPER_BOUND ( ' + ' '.join(map(str, UB)) + ' )',
            'DIRECTION_TYPE ORTHO N+1 NEG', f'INITIAL_FRAME_SIZE * {R.DELTA_0}', 'EVAL_OPPORTUNISTIC false',
            'QUAD_MODEL_SEARCH false', 'NM_SEARCH false', 'SPECULATIVE_SEARCH false',
            f'DISPLAY_DEGREE {display_degree}', 'DISPLAY_ALL_EVAL true',
            'DISPLAY_STATS BBE ( SOL ) OBJ CONS_H FRAME_SIZE MESH_SIZE', f'SEED {seed}']


def run_nomad(name, seed, log_path, anisotropic, display_degree, max_iterations=None):
    """One NOMAD run in a FRESH interpreter (this script with --one-nomad-run). Reason (observed in this task with
    PyNomadBBO 4.6.0): within one process, a SEED equal to the previous optimize() call's SEED is not re-applied and
    the run starts from a fixed default RNG state instead, so two runs 'with seed 1' in one process differ and runs
    with different seeds can coincide. One process per run makes SEED effective for every run."""
    import subprocess
    args = json.dumps([name, seed, log_path, anisotropic, display_degree, max_iterations])
    cp = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one-nomad-run', args],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f'NOMAD child failed rc={cp.returncode}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}')
    payload = json.loads(cp.stdout.strip().splitlines()[-1])
    return payload['rec'], payload['evals']


def _run_nomad_in_process(name, seed, log_path, anisotropic, display_degree, max_iterations=None):
    evals = []

    def bb(x):
        z = tuple(int(round(x.get_coord(i))) for i in range(N))
        why = LAT.reasons(z)
        f = f_stub(name, z) if not why else 0.0
        evals.append({'z': list(z), 'F': None if why else f, 'reasons': why})
        x.setBBO(f'{f!r} {len(why)}'.encode())
        return 1

    params = nomad_params(seed, anisotropic, display_degree, max_iterations)
    sys.stdout.flush()
    fd = os.dup(1)
    with open(log_path, 'w') as fh:
        os.dup2(fh.fileno(), 1)
        try:
            res = PyNomad.optimize(bb, list(Z_START), [], [], params)
        finally:
            sys.stdout.flush()
            os.dup2(fd, 1)
            os.close(fd)
    res = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in res.items()}
    best = res.get('f_single_best')
    first_best = next((i + 1 for i, e in enumerate(evals) if e['F'] is not None and e['F'] == best), None)
    return {'stub': name, 'seed': seed, 'anisotropic_mesh': anisotropic, 'params': params, 'result': res,
            'n_bb_evals': len(evals), 'bbe_at_which_final_best_first_evaluated': first_best,
            'log': os.path.basename(log_path)}, evals


FLOAT_LINE = re.compile(r'(-?\d+(?:\.\d+)?(?:e-?\d+)?)')


def _nums(s):
    return [float(a) for a in FLOAT_LINE.findall(s)]


def _pt(s):
    m = re.search(r'\(([^)]*)\)', s)
    return [int(round(float(a))) for a in m.group(1).split()]


def parse_nomad_log(path):
    """One record per MegaIteration poll: frame / mesh size, unit v, 2n columns, 2n projected dirs, center,
    snapped points (in generation order), sorted order, second-pass direction and point."""
    polls, cur, block = [], None, None
    frame = mesh = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('delta mesh  size ='):
                mesh = _nums(s.split('=', 1)[1])
            elif s.startswith('Delta frame size ='):
                frame = _nums(s.split('=', 1)[1])
            elif s.startswith('Unit sphere direction:'):
                cur = {'frame_size': frame, 'mesh_size': mesh, 'v': _nums(s.split(':', 1)[1]), 'H_cols': [],
                       'proj_dirs': [], 'before_proj': [], 'after_proj': [], 'sorted': [], 'center': None,
                       'second_dir': None, 'second_point': None, 'ids': []}
                polls.append(cur)
                block = 'first'
            elif cur is None:
                continue
            elif s.startswith('Poll direction before scaling'):
                cur['H_cols'].append(_nums(s.split(':', 1)[1]))
            elif s.startswith('Generate second pass trial point'):
                block = 'second'
            elif s.startswith('Scaled and mesh projected poll direction:'):
                d = [int(round(a)) for a in _nums(s.split(':', 1)[1])]
                if block == 'second':
                    cur['second_dir'] = d
                else:
                    cur['proj_dirs'].append(d)
            elif s.startswith('Frame center:') and cur['center'] is None:
                cur['center'] = _pt(s)
            elif s.startswith('Point before projection:') and block == 'first':
                cur['before_proj'].append(_pt(s))
            elif s.startswith('Point after projection:'):
                if block == 'first':
                    cur['after_proj'].append(_pt(s))
                else:
                    cur['second_point'] = _pt(s)
            elif s.startswith('Generated point: #') and block == 'first':
                cur['ids'].append(int(re.search(r'#(\d+)', s).group(1)))
            elif s.startswith('Generated point not inserted') and block == 'first':
                cur['ids'].append(None)  # e.g. equal to the frame center after the bound snap
                cur.setdefault('not_inserted', []).append(s)
            elif s.startswith('Number of trial points after reduction to form a basis:'):
                cur['n_after_reduction'] = int(s.rsplit(':', 1)[1])
                cur['kept_nomad'] = []
                block = 'kept'
            elif block == 'kept' and len(cur['kept_nomad']) < cur['n_after_reduction'] and '#' in s:
                cur['kept_nomad'].append(int(re.search(r'#(\d+)', s).group(1)))
            elif s.startswith('Evaluation points after sort:'):
                block = 'sorted'
            elif block == 'sorted':
                if s.startswith('#'):
                    cur['sorted'].append(int(re.search(r'#(\d+)', s).group(1)))
                else:
                    block = 'first_done'
            elif s.startswith('Update:') or s.startswith('Last Iteration'):
                cur.setdefault('update', []).append(s)
    return polls


def check_poll(p):
    n = len(p['v'])
    v = np.array(p['v'])
    H = np.eye(n) - 2.0 * np.outer(v, v)  # the in-house formula (householder_columns), applied to NOMAD's v
    cols = np.array(p['H_cols'])  # NOMAD order: H1, -H1, H2, -H2, ...
    expected_cols = np.array([s * H[:, j] for j in range(n) for s in (1.0, -1.0)])
    deltas = [int(round(a)) for a in p['frame_size']]
    delta = deltas[0] if len(set(deltas)) == 1 else None  # isotropic frame -> scalar Delta
    m1a = float(np.max(np.abs(cols - expected_cols))) if cols.shape == expected_cols.shape else None
    def _proj(h):
        # the in-house project_direction when the frame is isotropic; otherwise its rounding rule
        # (R._round_half_away(Delta_i h_i / ||h||_inf)) with NOMAD's per-coordinate frame size
        if delta is not None:
            return list(R.project_direction(list(h), delta))
        m = max(abs(a) for a in h)
        return [R._round_half_away(deltas[i] * h[i] / m) for i in range(n)]

    inhouse_proj_nomad_cols = [_proj(c) for c in p['H_cols']]
    inhouse_proj_from_v = [_proj(c) for c in expected_cols]
    m1b = inhouse_proj_nomad_cols == p['proj_dirs']
    m1b_v = inhouse_proj_from_v == p['proj_dirs']
    c = p['center']
    snapped = [[min(max(c[i] + d[i], LB[i]), UB[i]) for i in range(n)] for d in p['proj_dirs']]
    m1c = snapped == p['after_proj']
    m1c_before = [[c[i] + d[i] for i in range(n)] for d in p['proj_dirs']] == p['before_proj']
    # M1d: greedy rank over the sorted order, on post-snap displacements
    if len(p['ids']) != len(p['after_proj']):
        raise RuntimeError('NOMAD log parse: generated-point ids and projected points misaligned')
    by_id = {pid: pt for pid, pt in zip(p['ids'], p['after_proj']) if pid is not None}
    kept, rank = [], 0
    for pid in p['sorted']:
        if pid not in by_id:
            continue
        trial = kept + [pid]
        r = np.linalg.matrix_rank(np.array([[by_id[k][i] - c[i] for i in range(n)] for k in trial], dtype=float))
        if r > rank:
            kept, rank = trial, r
        if rank == n:
            break
    neg = [-sum(by_id[k][i] - c[i] for k in kept) for i in range(n)] if rank == n else None
    m1d = (neg == p['second_dir']) if p['second_dir'] is not None else None
    kept_nomad = p.get('kept_nomad')
    neg_nomad_kept = ([-sum(by_id[k][i] - c[i] for k in kept_nomad) for i in range(n)]
                      if kept_nomad and all(k in by_id for k in kept_nomad) else None)
    return {'frame_size': deltas, 'frame_isotropic': delta is not None, 'mesh_size': p['mesh_size'][0] if p['mesh_size'] else None, 'center': c,
            'v_nomad_unit_sphere': p['v'], 'M1a_max_abs_H_diff': m1a, 'M1b_proj_equal_inhouse_rule': m1b,
            'M1b_proj_equal_inhouse_rule_from_v': m1b_v, 'M1c_bound_snap_equal_clip': m1c,
            'M1c_before_projection_equal_center_plus_d': m1c_before,
            'nomad_projected_dirs_2n': p['proj_dirs'], 'generated_points_after_snap': p['after_proj'], 'rank_reduction_kept_ids': kept, 'rank': rank,
            'M1d_second_pass_reproduced': m1d, 'nomad_kept_ids': kept_nomad,
            'M1d_greedy_rank_kept_equals_nomad_kept': (sorted(kept) == sorted(kept_nomad)) if kept_nomad else None,
            'M1d_second_dir_equals_neg_sum_of_nomad_kept': ((neg_nomad_kept == p['second_dir'])
                                                            if p['second_dir'] is not None else None),
            'n_generated_not_inserted': len(p.get('not_inserted', [])), 'nomad_second_dir': p['second_dir'],
            'nomad_second_point': p['second_point'], 'n_sorted': len(p['sorted']),
            'reproduced_second_dir': neg, 'nomad_update': p.get('update')}


def enumerate_min(name):
    best = None
    for z in LAT.domain():
        f = f_stub(name, z)
        if best is None or (f, LAT.investment_cost(z), LAT.label(z)) < best[:3]:
            best = (f, LAT.investment_cost(z), LAT.label(z), z)
    return {'F': best[0], 'label': best[2], 'z': list(best[3])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    t0 = time.time()
    out = {'task': 'P5.15 Addendum 32 Q3 W30 NOMAD stub check (STEP4 section 6 (ii))',
           'environment': {'python': sys.version, 'prefix': sys.prefix, 'platform': platform.platform(),
                           'machine': platform.machine(), 'pynomad_version_banner': 'see nomad logs header',
                           'numpy': np.__version__},
           'stub': {'years': YEARS, 'unit_costs_eur': STUB_COSTS, 'budget': STUB_BUDGET, 'lb': LB, 'ub': UB,
                    'z_start': Z_START, 'target_zP': TARGET_P, 'target_zE': TARGET_E,
                    'A_affine': 'F = I(z) (stub costs, positive slopes)',
                    'B_quadratic': 'F = I(z) + 100 sum_n[(zP_n - tP_n)^2 + (zE_n - tE_n)^2] + 50 (zY - 1)^2'},
           'inhouse': {'module': 'p515_s47_phase_b_record.py', 'sha256': sha256(os.path.join(REPO,
                                                                                         'p515_s47_phase_b_record.py')),
                       'HALTON_T0': R.HALTON_T0, 'DELTA_0': R.DELTA_0, 'POLL_DESIGN': R.POLL_DESIGN,
                       'COMPLETION_CAP': R.COMPLETION_CAP, 'first_polls_directions': []},
           'runs': {}}
    for k in range(3):
        for delta in (4, 2, 1):
            t, u, dirs = R.poll_directions(k, delta)
            out['inhouse']['first_polls_directions'].append({'k': k, 'halton_t': t, 'Delta': delta,
                                                              'halton_u': u, 'directions': [list(d) for d in dirs]})
    for name in ('A_affine', 'B_quadratic'):
        run = {'lattice_min_by_enumeration': enumerate_min(name), 'inhouse': run_inhouse(name), 'nomad_M1': [],
               'nomad_M2': []}
        if run['inhouse']['termination']['reason'] == 'STOP_FOR_REVIEW_completion_cap':
            run['inhouse_cap_lifted'] = run_inhouse(name, completion_cap=10 ** 6)
        for aniso in (False, True):
            for seed in NOMAD_SEEDS:
                tag = f'{name}_{"aniso" if aniso else "iso"}_seed{seed}'
                # M1: first MAX_ITERATIONS_DEBUG polls at DISPLAY_DEGREE 4 (directions exposed)
                log_path = os.path.join(OUT, f'nomad_M1_{tag}.log')
                rec, _ = run_nomad(name, seed, log_path, aniso, 4, MAX_ITERATIONS_DEBUG)
                polls = parse_nomad_log(log_path)
                rec['polls'] = [check_poll(p) for p in polls]
                rec['frame_size_sequence'] = [p['frame_size'] for p in rec['polls']]
                rec['all_M1a_max_abs'] = max((p['M1a_max_abs_H_diff'] or 0.0) for p in rec['polls'])
                rec['all_M1b'] = all(p['M1b_proj_equal_inhouse_rule'] for p in rec['polls'])
                rec['all_M1b_from_v'] = all(p['M1b_proj_equal_inhouse_rule_from_v'] for p in rec['polls'])
                rec['all_M1c'] = all(p['M1c_bound_snap_equal_clip'] for p in rec['polls'])
                rec['M1d_negsum_nomad_kept_counts'] = {
                    str(k): sum(1 for p in rec['polls'] if p['M1d_second_dir_equals_neg_sum_of_nomad_kept'] is k)
                    for k in (True, False, None)}
                rec['M1d_counts'] = {str(k): sum(1 for p in rec['polls'] if p['M1d_second_pass_reproduced'] is k)
                                     for k in (True, False, None)}
                rec['log_sha256'] = sha256(log_path)
                run['nomad_M1'].append(rec)
                # M2: full run to NOMAD's own termination at DISPLAY_DEGREE 2
                log_path = os.path.join(OUT, f'nomad_M2_{tag}.log')
                rec, evals = run_nomad(name, seed, log_path, aniso, 2)
                m1_first = run['nomad_M1'][-1]['polls'][0]
                m1_pts = {tuple(pt) for pt in m1_first['generated_points_after_snap']}
                if m1_first['nomad_second_point']:
                    m1_pts.add(tuple(m1_first['nomad_second_point']))
                first_poll_m2 = [tuple(e['z']) for e in evals[1:1 + N]]
                rec['first_poll_points_match_M1_same_seed'] = all(pt in m1_pts for pt in first_poll_m2)
                rec['log_sha256'] = sha256(log_path)
                run['nomad_M2'].append(rec)
        out['runs'][name] = run
    guard_failures = R.PARENT_GUARD.verify(0)
    out['solve_profile_guard'] = {'counts': dict(R.PARENT_GUARD.counts), 'verify_0_failures': guard_failures}
    if guard_failures:
        raise SystemExit(f'guard verify(0) failed: {guard_failures}')
    out['wall_seconds'] = time.time() - t0
    path = os.path.join(OUT, 'nomad_stub_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    # console summary
    for name, run in out['runs'].items():
        print(f'== {name}: lattice min {run["lattice_min_by_enumeration"]}')
        ih = run['inhouse']
        print(f'   in-house: {ih["termination"]["reason"]} inc={ih["incumbent"]["label"]} F={ih["incumbent"]["F"]} '
              f'polls={ih["n_polls"]} evals={ih["n_new_evaluations"]} Delta seq={[p["Delta"] for p in ih["polls"]]}')
        if 'inhouse_cap_lifted' in run:
            ih = run['inhouse_cap_lifted']
            print(f'   in-house (cap lifted): {ih["termination"]["reason"]} inc={ih["incumbent"]["label"]} '
                  f'F={ih["incumbent"]["F"]} polls={ih["n_polls"]} evals={ih["n_new_evaluations"]} '
                  f'Delta seq={[p["Delta"] for p in ih["polls"]]}')
        for rec in run['nomad_M1']:
            print(f'   NOMAD M1 aniso={rec["anisotropic_mesh"]} seed {rec["seed"]}: polls={len(rec["polls"])} '
                  f'frames={rec["frame_size_sequence"]} M1a={rec["all_M1a_max_abs"]:.2e} '
                  f'M1b={rec["all_M1b"]}/{rec["all_M1b_from_v"]} M1c={rec["all_M1c"]} M1d={rec["M1d_counts"]} '
                  f'M1d_nomad_kept={rec["M1d_negsum_nomad_kept_counts"]}')
        for rec in run['nomad_M2']:
            r = rec['result']
            print(f'   NOMAD M2 aniso={rec["anisotropic_mesh"]} seed {rec["seed"]}: x={r.get("x_single_best")} '
                  f'f={r.get("f_single_best")} bbe={rec["n_bb_evals"]} total={r.get("nb_evals")} '
                  f'best_first_at_bbe={rec["bbe_at_which_final_best_first_evaluated"]} stop={r.get("stop_reason")} '
                  f'first_poll_matches_M1={rec["first_poll_points_match_M1_same_seed"]}')
    print(f'results: {path} sha256 {sha256(path)}  wall {out["wall_seconds"]:.1f}s')
    print(f'solve profile guard (permitted=()): counts {out["solve_profile_guard"]["counts"]} verify(0) failures '
          f'{out["solve_profile_guard"]["verify_0_failures"]}')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--one-nomad-run':
        _rec, _evals = _run_nomad_in_process(*json.loads(sys.argv[2]))
        print(json.dumps({'rec': _rec, 'evals': _evals}))
    else:
        main()
