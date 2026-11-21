"""P5.15 S56 -- build and verify the dependence stress-test instance on both candidate grids (ZERO SOLVES).

Planner task "Dependence instance" (TASKS.md, S56 author decision 3): comonotone re-pairing of the operation scenarios
within each network, by rank of the daily net demand, every invariant verified, for
  srp1grid_3x3     data/SRP1/Results/P515S56/instances_v1/SRP1__s56_srp1grid_3x3.json       (checksum 73d0a230...)
  five_block_3x3   data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json             (checksum 0248a5c8...)
The definition (statistic, tie rule, coherence, hook, assertion) is in p515_s56_dependence_instance.py.

Per grid:
  1. BASE READ through the production reader the 3 x 3 pair used (p515_s44_scale_measurement.
     read_planning_from_derived_case), with a capture-only DependenceHook at shared_resources_planning.
     _compute_scenario_metadata: snapshot of every scenario array (audit A's enumeration) and the derivation
     (p515_s56_dependence_instance.derive) at the checksum point; the same derivation on the post-read object.
  2. PROFILES FILE (permutations, statistic, per-array sha256 of the base and permuted arrays, market sha256, post-read
     expectations, probabilities) and a human-readable permutation record.
  3. DEPENDENCE READ: the same case + DependenceHook(profiles) (verifies the base bytes, recomputes the permutation,
     permutes in place); in-child assertion at the checksum point and after the read; a second, independent read
     must reproduce the checksum.
  4. CHECKS (expected vs observed): arrays = base[perm] bitwise at both points; multiset of per-scenario profiles per
     network block unchanged (bitwise); probabilities unchanged (1/3, 1/9 per pair); EV means unchanged (bitwise or
     ulp, with the summation-order explanation verified by recomputation and by an exactly-rounded sum); market arrays
     unchanged; statistic non-decreasing in the new index in every block; object walk vs the base differs only in the
     declared scenario arrays (declared-exclusion walk: 0 diffs; bookkeeping-only-exclusion walk: exactly the arrays
     the permutation changes); new checksum; blocks changed per network; ties; the assertion passes on the
     dependence read and RAISES on the base (index-paired) read and on a wrong expected checksum.
  5. ZERO-SOLVE SUMMARY of the changed dependence: per block, pairwise Pearson / Spearman of the daily net demand
     across the 3 scenarios under index pairing vs comonotone pairing, and the spread (range, population std) of the
     total system net demand across the joint scenarios.

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0, plus the
guard installed by the imported audit A module (verified at 0).

Output: NEW directory data/SRP1/Results/P515S56/dependence_instance_v1/ (created by this script with exist_ok=False:
refuses if it exists; refuses if another copy runs). Launch logs are siblings of the directory.

Launch (attached, both streams captured, noclobber):
    set -o noclobber; /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s56_build_dependence_instance.py \\
        > data/SRP1/Results/P515S56/dependence_instance_v1_launch_stdout.log \\
        2> data/SRP1/Results/P515S56/dependence_instance_v1_launch_stderr.log
Options: --out <absolute dir> (dry runs: outside the repository; export runs: the real output directory);
         --provenance-repo <dir> (the git repository the code was exported from; default: this script's directory).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 dependence instance (never solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager, redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402  (the production read of a derived case)
import p515_s56_expected_profiles as AA  # noqa: E402  (snapshot / compute_means / walk_compare; own permitted=() guard)
import p515_s56_ev_injection as EVI  # noqa: E402
import p515_s56_dependence_instance as DI  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'dependence_instance_v1')
if '--out' in sys.argv:
    OUT_DIR = os.path.abspath(sys.argv[sys.argv.index('--out') + 1])
else:
    OUT_DIR = os.path.join(REPO, OUT_REL)
PROV_REPO = (os.path.abspath(sys.argv[sys.argv.index('--provenance-repo') + 1]) if '--provenance-repo' in sys.argv
             else REPO)
N_SCEN = 3

GRIDS = [
    {'grid': 'srp1grid_3x3', 'label': 's56_srp1grid_3x3_comonotone', 'base_label': 's56_srp1grid_3x3',
     'case': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3.json'),
     'record': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'instance_record.json'),
     'checksum_prefix': '73d0a230',
     'ev_profiles': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1',
                                 'SRP1__s56_srp1grid_3x3_ev_profiles.json'),
     'years': [2025, 2030, 2035]},
    {'grid': 'five_block_3x3', 'label': 's53_3x3_comonotone', 'base_label': 's53_3x3',
     'case': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json'),
     'record': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json'),
     'checksum_prefix': '0248a5c8',
     'ev_profiles': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'input_audit_A', 'instance',
                                 'SRP1__s56_3x3_ev_profiles.json'),
     'years': [2025, 2028, 2031, 2034, 2037]},
]
DATA_INPUTS = [os.path.join('data', 'SRP1', p) for p in (
    'SRP1.json', 'SRP1_params.json', os.path.join('MarketData', 'SRP1_market_data.xlsx'),
    os.path.join('SharedESS', 'SRP1_ESS.xlsx'), os.path.join('SharedESS', 'SRP1_ESS_Params.json'),
    os.path.join('case9', 'case9_operational_data.xlsx'), os.path.join('case9', 'case9_params.json'),
    os.path.join('case33_1', 'case33_1_operational_data.xlsx'), os.path.join('case33_1', 'case33_1_params.json'),
    os.path.join('case33_2', 'case33_2_operational_data.xlsx'), os.path.join('case33_2', 'case33_2_params.json'),
    os.path.join('case33_3', 'case33_3_operational_data.xlsx'), os.path.join('case33_3', 'case33_3_params.json'))]
# the declared exclusions of audit A's walk minus every scenario array / probability field: what may differ between two
# reads of the same case regardless of the scenario arrays (the checksum record and the read's output directories)
BOOKKEEPING_EXCLUSIONS = {
    'SharedResourcesPlanning': {'scenario_metadata', 'results_dir', 'diagrams_dir', 'logs_dir'},
    'NetworkData': {'results_dir', 'diagrams_dir', 'logs_dir'},
    'Network': {'results_dir', 'diagrams_dir', 'logs_dir'},
    'SharedEnergyStorageData': {'results_dir', 'logs_dir'},
}
FIELD_OF_ATTR = {'pd': 'pd', 'qd': 'qd', 'pg': 'pg', 'qg': 'qg',
                 'flexibility.active_power.upward': 'flex_p_up', 'flexibility.active_power.downward': 'flex_p_down',
                 'flexibility.reactive_power.upward': 'flex_q_up', 'flexibility.reactive_power.downward': 'flex_q_down'}
WALK_PATH_RE = re.compile(
    r"^planning\.(transmission_network|distribution_networks\[(\d+)\])\.network\[(\d+)\]\['([^']+)'\]\."
    r"(loads|generators)\[(\d+)\]\.(" + '|'.join(re.escape(a) for a in FIELD_OF_ATTR) + r")$")
EV_EXPLANATION = ('the mean is AA.compute_means: np.sum(w[:, None] * A, axis=0) with w = [1/3] * 3. The products '
                  'w[s] * A[s, t] are the same multiset after the permutation; np.sum along axis 0 adds the rows in '
                  'index order ((r0 + r1) + r2), so a permutation changes only the ORDER of two floating additions, '
                  'which can move the rounded result by a few ulp. Verified two ways: (i) recomputing the BASE mean '
                  'with its rows taken in the permuted order reproduces the dependence mean bit for bit; (ii) an '
                  'order-independent, exactly-rounded mean (math.fsum of the products per hour) is bitwise identical '
                  'for the base and the dependence arrays. Size: two orders of a 3-term recursive sum differ by at '
                  'most ~2 ulp of sum_s |w_s A[s, t]| (the textbook bound, 2 * gamma_2 * sum |x|); the difference '
                  'in ulp of the RESULT is larger only where the terms have mixed signs and cancel (mean near zero)')
# textbook bound: each order's recursive 3-term sum is within gamma_2 * S of the exact sum (S = sum_s |w_s A[s,t]|,
# gamma_2 = 2u / (1 - 2u), u = 2^-53), so two orders differ by <= ~4u S; spacing(S) lies in (u S, 2u S], so 4u S is
# at most 4 ulp of S
EV_BOUND_ULP_OF_ABS_SUM = 4.0

CHECKS = []
PREDICTIONS = []   # expectations stated in the task, recorded expected vs observed; NOT build-integrity checks
STARTED = time.time()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, expected=None, observed=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'expected': expected, 'observed': observed})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + ('' if ok else f' -- expected {str(expected)[:300]}; observed '
                                                               f'{str(observed)[:600]}'))


def _pred(name, held, expected=None, observed=None):
    PREDICTIONS.append({'prediction': name, 'held': bool(held), 'expected': expected, 'observed': observed})
    _log(f'{"PRED HELD  " if held else "PRED FAILED"} {name}' + ('' if held else f' -- expected {str(expected)[:300]}; '
                                                                             f'observed {str(observed)[:600]}'))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, check=True):
    return subprocess.run(['git', '-C', PROV_REPO, *args], capture_output=True, text=True, check=check).stdout


def _head_blob_sha256(rel):
    """sha256 of the file's bytes at the provenance repository's HEAD, or None if not tracked."""
    r = subprocess.run(['git', '-C', PROV_REPO, 'cat-file', '-p', f'HEAD:{rel}'], capture_output=True)
    if r.returncode != 0:
        return None
    return hashlib.sha256(r.stdout).hexdigest()


def _input_record(rel):
    path = os.path.join(REPO, rel)
    sha = _sha(path)
    head = _head_blob_sha256(rel)
    return {'path': rel, 'sha256': sha, 'tracked_at_head': head is not None,
            'equals_head': (sha == head) if head is not None else None}


def _rel(path):
    return os.path.relpath(path, REPO)


def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def _provenance():
    return {'provenance_repo': PROV_REPO, 'code_dir': REPO, 'git_head': _git(['rev-parse', 'HEAD']).strip(),
            'tracked_status_porcelain': _git(['status', '--porcelain', '--untracked-files=no']),
            'utc': datetime.now(timezone.utc).isoformat()}


def _module_path_audit():
    mods = {}
    outside = {}
    for name, m in sorted(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if not f:
            continue
        f = os.path.abspath(f)
        if f.startswith(REPO + os.sep):
            rel = os.path.relpath(f, REPO)
            if rel.endswith('.py'):
                mods[name] = _input_record(rel)
        elif PROV_REPO != REPO and f.startswith(PROV_REPO + os.sep):
            outside[name] = f
    not_head = {n: r['path'] for n, r in mods.items() if r['tracked_at_head'] and not r['equals_head']}
    untracked = {n: r['path'] for n, r in mods.items() if not r['tracked_at_head']}
    return {'modules': mods, 'tracked_modules_not_equal_head': not_head, 'untracked_modules': untracked,
            'modules_loaded_from_provenance_repo_outside_code_dir': outside}


class _Stages:
    @contextmanager
    def stage(self, _name):
        yield


@contextmanager
def _walk_exclusions(excl):
    old = AA.EXCLUDED_ATTRS
    AA.EXCLUDED_ATTRS = excl
    try:
        yield
    finally:
        AA.EXCLUDED_ATTRS = old


def production_read(case_rel, read_dir, hook):
    if os.path.exists(read_dir):
        raise RuntimeError(f'refusing to overwrite {read_dir}')
    os.makedirs(read_dir)
    hook.install()
    stream = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(stream):
            planning = S.read_planning_from_derived_case({'derived_case': {'path': case_rel}}, read_dir, _Stages())
    finally:
        hook.uninstall()
        with open(os.path.join(read_dir, 'production_stdout.log'), 'w') as handle:
            handle.write(stream.getvalue())
    if hook.calls != 1:
        raise RuntimeError(f'{read_dir}: checksum hook called {hook.calls} times (expected 1)')
    return planning, {'case_path': case_rel, 'case_sha256': _sha(os.path.join(REPO, case_rel)),
                      'wall_s': time.time() - t0, 'read_dir': os.path.relpath(read_dir, OUT_DIR),
                      'scenario_metadata': dict(planning.scenario_metadata), 'hook_apply_counts': hook.apply_counts}


def _bytes(a):
    return np.ascontiguousarray(np.asarray(a, dtype='<f8')).tobytes()


def _blocks_of_keys(keys):
    out = {}
    for ks in keys:
        p = ks.split('|')
        out.setdefault('|'.join(p[:3]), []).append(ks)
    return out


def _tuple_key(ks):
    """key string -> (holder, year, day, kind, index, field) without the generator type."""
    p = ks.split('|')
    return (p[0], int(p[1]), p[2], p[3], int(p[4]), p[-1])


def _ulp_diff(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    sp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    with np.errstate(invalid='ignore', divide='ignore'):
        u = np.where(a == b, 0.0, np.abs(a - b) / np.where(sp > 0, sp, 1.0))
    return float(np.max(u)) if u.size else 0.0


def _avg_ranks(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind='stable')
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    da, db = a - a.mean(), b - b.mean()
    den = math.sqrt(float(np.sum(da * da)) * float(np.sum(db * db)))
    return None if den == 0.0 else float(np.sum(da * db)) / den


def _spearman(a, b):
    return _pearson(_avg_ranks(a), _avg_ranks(b))


# ======================================================================================================================
#  one grid
# ======================================================================================================================
def build_grid(g):
    gname = g['grid']
    gdir = os.path.join(OUT_DIR, gname)
    os.makedirs(gdir)
    pre = f'[{gname}] '
    with open(os.path.join(REPO, g['record'])) as handle:
        rec_base = json.load(handle)
    pin_case_sha = rec_base['case_sha256']
    pin_ck = rec_base.get('scenario_checksum') or rec_base.get('facts', {}).get('scenario_checksum')
    case_sha = _sha(os.path.join(REPO, g['case']))
    _ck(pre + 'base case sha256 == its instance record', case_sha == pin_case_sha, pin_case_sha, case_sha)
    _ck(pre + 'pinned base checksum starts with the Planner-given prefix', str(pin_ck).startswith(g['checksum_prefix']),
        g['checksum_prefix'], pin_ck)

    # ---------------- 1. base read ----------------
    cap = {}

    def on_point_base(planning):
        refs = EVI.array_refs(planning)
        cap['snap'] = AA.snapshot(planning)
        cap['derive'] = DI.derive(planning)
        cap['stat_alt'] = DI.daily_net_demand(planning, include_adn_loads=True)
        op_ids = [id(a) for _k, kind, a, _w, _b in refs if kind == 'operation']
        cap['op_ids_unique'] = len(set(op_ids)) == len(op_ids)
        cap['n_op'] = len(op_ids)
    p_base, read_base = production_read(g['case'], os.path.join(gdir, 'reads', 'base'),
                                        DI.DependenceHook(profiles=None, on_point=on_point_base))
    base_ck = read_base['scenario_metadata']['combined_scenario_checksum']
    _log(pre + f'base read: checksum {base_ck} ({read_base["wall_s"]:.1f} s)')
    _ck(pre + 'base read: combined scenario checksum == pinned', base_ck == pin_ck, pin_ck, base_ck)
    snap_b, der = cap['snap'], cap['derive']
    keys_all = list(snap_b)
    op_keys = [AA.key_str(k) for k in keys_all if k[0] != 'market']
    mk_keys = [AA.key_str(k) for k in keys_all if k[0] == 'market']
    key_of = {AA.key_str(k): k for k in keys_all}
    _ck(pre + 'derive() operation keys == audit A enumeration (operation keys, in order)', der['keys'] == op_keys,
        len(op_keys), len(der['keys']))
    _ck(pre + 'no two operation keys share an ndarray (in-place permutation is safe)', cap['op_ids_unique'], True,
        cap['op_ids_unique'])
    post_b = AA.snapshot(p_base)
    der_post = DI.derive(p_base)
    same_perm = der_post['permutations'] == der['permutations']
    same_stat = all(der_post['statistic'][bk]['values'] == der['statistic'][bk]['values'] for bk in der['statistic'])
    _ck(pre + 'derive() on the post-read object == at the checksum point (permutations; statistic bitwise)',
        same_perm and same_stat, True, {'permutations': same_perm, 'statistic': same_stat})
    post_changed = [AA.key_str(k) for k in keys_all if _bytes(snap_b[k]['A']) != _bytes(post_b[k]['A'])]
    post_zero = all(not np.any(post_b[key_of[k]]['A']) for k in post_changed)
    _ck(pre + 'base read: arrays production changes after the checksum point == TSO ADN loads pd / qd (zeroed)',
        len(post_changed) > 0 and post_zero and all(k.startswith('TSO|') and '|load|' in k and k.split('|')[-1] in
                                                    ('pd', 'qd') for k in post_changed),
        'TSO|...|load|<adn>|pd/qd -> 0', post_changed[:8])
    adn_ok = all(set(int(k.split('|')[4]) for k in post_changed if k.startswith(bk + '|'))
                 == set(v['adn_load_indices']) for bk, v in der['statistic'].items() if bk.startswith('TSO|'))
    _ck(pre + 'statistic ADN-load exclusion == the loads production zeroes (every TSO block)', adn_ok, True, adn_ok)

    # coverage: every (n_oper x T) float array reachable from the planning object is one of the enumerated arrays
    ref_ids = {id(a) for _k, _kind, a, _w, _b in EVI.array_refs(p_base)}
    T = int(p_base.num_instants)
    uncovered, other_n = [], Counter()
    seen = set()
    stack = [(p_base, 'planning')]
    while stack:
        x, path = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape == (N_SCEN, T) and id(x) not in ref_ids:
                uncovered.append(path)
            elif x.ndim >= 1 and x.shape[0] == N_SCEN and id(x) not in ref_ids:
                other_n[f'{x.shape}'] += 1
            continue
        if isinstance(x, dict):
            stack.extend((v, f'{path}[{k!r}]') for k, v in x.items())
        elif isinstance(x, (list, tuple)):
            stack.extend((v, f'{path}[{i}]') for i, v in enumerate(x))
        elif hasattr(x, '__dict__') and not isinstance(x, type) and type(x).__module__ not in ('builtins',):
            stack.extend((v, f'{path}.{k}') for k, v in vars(x).items())
    _ck(pre + f'coverage: no ({N_SCEN} x {T}) array reachable from the planning object outside the enumerated scenario '
              f'arrays', not uncovered, [], uncovered[:10])

    # ---------------- 2. profiles file ----------------
    stat = der['statistic']
    perms = der['permutations']
    n_changed_ck = sum(1 for k in op_keys if der['permuted_sha256'][k] != der['base_sha256'][k])
    post_expected = {k: DI.array_sha(post_b[key_of[k]]['A']) for k in post_changed}
    n_changed_post = sum(1 for k in op_keys if der['permuted_sha256'][k] != der['base_sha256'][k]
                         and k not in post_expected)
    block_probs = {}
    for hname, holder in EVI.holders(p_base):
        for y in holder.years:
            for d in holder.days:
                net = holder.network[y][d]
                block_probs[DI.block_key(hname, y, d)] = {
                    'operation': [float(v) for v in net.prob_operation_scenarios],
                    'market': [float(v) for v in net.prob_market_scenarios]}
    ties = {bk: [[s, t] for s in range(N_SCEN) for t in range(s + 1, N_SCEN) if v['values'][s] == v['values'][t]]
            for bk, v in stat.items()}
    n_ties = sum(len(v) for v in ties.values())
    non_identity = {bk: not ident for bk, ident in der['identity'].items()}
    profiles_name = f'SRP1__{g["label"]}_profiles.json'
    prof_path = os.path.join(gdir, profiles_name)
    profiles = {
        'schema': DI.SCHEMA, 'instance_label': g['label'], 'grid': gname,
        'base_instance': {'label': g['base_label'], 'case_path': g['case'], 'case_sha256': case_sha,
                          'scenario_checksum': base_ck, 'instance_record': g['record'],
                          'instance_record_sha256': _sha(os.path.join(REPO, g['record']))},
        'description': ('dependence STRESS TEST (TASKS.md S56 author decision 3): comonotone re-pairing of the '
                        'operation scenarios within each network and (year, day) block by rank of the daily net '
                        'demand; complete within-network daily profiles, marginals, probabilities and EV inputs '
                        'preserved; market arrays not permuted; not a calibrated or worst-case joint model'),
        'statistic': DI.STATISTIC, 'tie_rule': DI.TIE_RULE, 'array_sha_definition': DI.ARRAY_SHA_DEFINITION,
        'coherence': ('each load / generator quantity is sampled with its own seed (network._update_network_with_'
                      'operational_data), so a scenario of a network is the index s: the tuple of row s of all of that '
                      'network block\'s operation arrays; the permutation moves that tuple as a unit'),
        'n_market_scenarios': int(p_base.num_market_scenarios), 'n_operation_scenarios': N_SCEN,
        'market_probabilities': {str(y): [float(v) for v in p_base.prob_market_scenarios[y]] for y in sorted(p_base.years)},
        'block_probabilities': block_probs,
        'permutations': perms,
        'statistic_original_order': {bk: v['values'] for bk, v in stat.items()},
        'statistic_detail_original_order': {bk: {'load_MWh': v['load_MWh'], 'res_MWh': v['res_MWh'],
                                                 'adn_load_indices': v['adn_load_indices'],
                                                 'res_generators': v['res_generators'], 'dt_h': v['dt_h']}
                                            for bk, v in stat.items()},
        'ties': {bk: v for bk, v in ties.items() if v},
        'n_blocks': len(perms), 'n_blocks_non_identity': sum(non_identity.values()),
        'operation_keys': op_keys,
        'base_sha256': der['base_sha256'], 'permuted_sha256': der['permuted_sha256'],
        'market_sha256': der['market_sha256'],
        'post_read_expected_note': ('after read_planning_problem production replaces the TSO ADN loads\' pd / qd by '
                                    'np.zeros (shared_resources_planning._add_adn_node_to_transmission_network); the '
                                    'sha256 they take then (zeros, identical in the base and the re-paired instance)'),
        'post_read_expected_sha256': post_expected,
        'n_arrays_changed': {'checksum': n_changed_ck, 'post_read': n_changed_post},
        'scenario_checksum': None,
        'scenario_checksum_note': 'computed by the production read with the hook; recorded in the instance record',
    }
    with open(prof_path, 'x') as handle:
        GRIO.dump(profiles, handle, indent=0)
    prof = DI.load_profiles(prof_path)
    rt = (prof['permutations'] == perms and prof['statistic_original_order'] == {bk: v['values'] for bk, v in stat.items()}
          and prof['operation_keys'] == op_keys)
    _ck(pre + 'profiles file round trip (permutations, statistic bitwise, keys)', rt, True, rt)

    # human-readable permutation record
    perm_rec = {'schema': 'p515_s56_dependence_permutations_v1', 'grid': gname, 'instance_label': g['label'],
                'statistic': DI.STATISTIC, 'tie_rule': DI.TIE_RULE, 'blocks': {}}
    for bk in sorted(perms, key=lambda b: (b.split('|')[0], int(b.split('|')[1]), b.split('|')[2])):
        v = stat[bk]
        perm_rec['blocks'][bk] = {'permutation_new_index_takes_original': perms[bk], 'identity': der['identity'][bk],
                                  'net_demand_MWh_original_order': v['values'],
                                  'net_demand_MWh_new_order': [v['values'][p] for p in perms[bk]],
                                  'load_MWh_original_order': v['load_MWh'], 'res_MWh_original_order': v['res_MWh'],
                                  'ties': ties[bk]}
    perm_path = os.path.join(gdir, f'permutations_{gname}.json')
    with open(perm_path, 'x') as handle:
        GRIO.dump(perm_rec, handle, indent=1)

    # ---------------- 3. dependence read ----------------
    cap_d = {}

    def on_point_dep(planning):
        cap_d['snap'] = AA.snapshot(planning)
        try:
            cap_d['assert_checksum_point'] = DI.assert_dependence_arrays(planning, prof, point='checksum')
        except DI.DependenceInstanceMismatch as exc:
            cap_d['assert_checksum_point'] = {'ok': False, 'error': str(exc)[:2000], 'summary': exc.summary}
    p_dep, read_dep = production_read(g['case'], os.path.join(gdir, 'reads', 'dependence'),
                                      DI.DependenceHook(profiles=prof, on_point=on_point_dep))
    dep_ck = read_dep['scenario_metadata']['combined_scenario_checksum']
    _log(pre + f'dependence read: checksum {dep_ck} ({read_dep["wall_s"]:.1f} s)')
    snap_d = cap_d['snap']
    post_d = AA.snapshot(p_dep)
    exp_apply = [{'n_operation_arrays_permuted': len(op_keys), 'n_arrays_changed': n_changed_ck,
                  'n_market_arrays_checked_untouched': len(mk_keys),
                  'n_blocks_non_identity': sum(non_identity.values())}]
    _ck(pre + 'hook: applied once, every operation array permuted, market arrays untouched',
        read_dep['hook_apply_counts'] == exp_apply, exp_apply, read_dep['hook_apply_counts'])
    _ck(pre + 'dependence checksum != base checksum', dep_ck != base_ck, f'!= {base_ck}', dep_ck)
    _ck(pre + 'in-child assertion PASSES at the checksum point', cap_d['assert_checksum_point'].get('ok') is True,
        'ok', cap_d['assert_checksum_point'].get('failures', cap_d['assert_checksum_point'].get('error')))

    # (a) arrays == base[perm] bitwise, both points; market arrays unchanged
    def perm_check(sb, sd):
        bad_op, bad_mk = [], []
        for k in keys_all:
            ks = AA.key_str(k)
            if k[0] == 'market':
                if _bytes(sd[k]['A']) != _bytes(sb[k]['A']):
                    bad_mk.append(ks)
            else:
                if _bytes(sd[k]['A']) != _bytes(sb[k]['A'][perms[DI.block_of(k)], :]):
                    bad_op.append(ks)
        return bad_op, bad_mk
    for label, sb, sd in (('checksum point', snap_b, snap_d), ('after the read', post_b, post_d)):
        bad_op, bad_mk = perm_check(sb, sd)
        _ck(pre + f'{label}: every operation array == base[perm, :] bitwise', not bad_op, 0, bad_op[:5])
        _ck(pre + f'{label}: every market array == base bitwise (not permuted)', not bad_mk, 0, bad_mk[:5])
        _ck(pre + f'{label}: key list == base key list', list(sd) == list(sb), len(sb), len(sd))
    n_changed_obs = sum(1 for k in keys_all if k[0] != 'market' and _bytes(snap_d[k]['A']) != _bytes(snap_b[k]['A']))
    _ck(pre + 'number of operation arrays whose bytes change == profiles n_arrays_changed[checksum]',
        n_changed_obs == n_changed_ck, n_changed_ck, n_changed_obs)
    alias_bad = []
    for hname, holder in EVI.holders(p_dep):
        for y in holder.years:
            for d in holder.days:
                nb, nd = p_base_holder_block(p_base, hname, y, d), holder.network[y][d]
                for fld in ('cost_energy_p', 'cost_flex'):
                    if _bytes(getattr(nd, fld)) != _bytes(getattr(nb, fld)):
                        alias_bad.append(f'{hname}|{y}|{d}|{fld}')
    _ck(pre + 'every network block\'s own market arrays (cost_energy_p / cost_flex) == base bitwise', not alias_bad, 0,
        alias_bad[:5])

    # (b) multiset of per-scenario profiles per network block (and per array), bitwise
    by_block = _blocks_of_keys(op_keys)
    ms_bad, row_bad = {}, 0
    for label, sb, sd in (('checksum', snap_b, snap_d), ('post_read', post_b, post_d)):
        bad = []
        for bk, ks_list in by_block.items():
            tb = sorted(b''.join(_bytes(sb[key_of[k]]['A'][s]) for k in ks_list) for s in range(N_SCEN))
            td = sorted(b''.join(_bytes(sd[key_of[k]]['A'][s]) for k in ks_list) for s in range(N_SCEN))
            if tb != td:
                bad.append(bk)
            for k in ks_list:
                if sorted(_bytes(sb[key_of[k]]['A'][s]) for s in range(N_SCEN)) != \
                        sorted(_bytes(sd[key_of[k]]['A'][s]) for s in range(N_SCEN)):
                    row_bad += 1
        ms_bad[label] = bad
    _ck(pre + 'multiset of complete per-scenario block profiles (all of a block\'s operation arrays, row s) unchanged '
              'in every network block, bitwise, both points', not ms_bad['checksum'] and not ms_bad['post_read'],
        {'blocks': len(by_block), 'bad': 0}, {k: v[:5] for k, v in ms_bad.items()})
    _ck(pre + 'multiset of rows of every operation array unchanged, bitwise, both points', row_bad == 0, 0, row_bad)
    # within-block coherence: every array of a block moved by the same permutation (rows of the scenario tuple together)
    coh_bad = [bk for bk, ks_list in by_block.items()
               if any(_bytes(snap_d[key_of[k]]['A'][j]) != _bytes(snap_b[key_of[k]]['A'][perms[bk][j]])
                      for k in ks_list for j in range(N_SCEN))]
    _ck(pre + 'within-block coherence: scenario j of the re-paired block == scenario perm[j] of the base block for '
              'EVERY array of the block (the daily profile moves as a unit)', not coh_bad, 0, coh_bad[:5])

    # (c) probabilities
    third = [1.0 / 3.0] * 3
    pb_bad, pair_bad, n_pairs = [], [], 0
    for hname, holder in EVI.holders(p_dep):
        for y in holder.years:
            for d in holder.days:
                nd, nb = holder.network[y][d], p_base_holder_block(p_base, hname, y, d)
                if (list(nd.prob_operation_scenarios) != list(nb.prob_operation_scenarios)
                        or list(nd.prob_market_scenarios) != list(nb.prob_market_scenarios)
                        or list(nd.prob_operation_scenarios) != third or list(nd.prob_market_scenarios) != third):
                    pb_bad.append(f'{hname}|{y}|{d}')
                for s_m in range(len(nd.prob_market_scenarios)):
                    for s_o in range(len(nd.prob_operation_scenarios)):
                        n_pairs += 1
                        if SRP._scenario_probability(nd, s_m, s_o) != 1.0 / 9.0:
                            pair_bad.append(f'{hname}|{y}|{d}|{s_m}|{s_o}')
    mk_prob_same = all(list(p_dep.prob_market_scenarios[y]) == list(p_base.prob_market_scenarios[y]) == third
                       for y in p_dep.years)
    _ck(pre + 'probabilities unchanged: operation and market [1/3]*3 on every block (== base), planning market [1/3]*3 '
              'every year, _scenario_probability == 1/9 for every pair', not pb_bad and not pair_bad and mk_prob_same,
        {'blocks_bad': 0, 'pairs_bad': 0}, {'blocks_bad': pb_bad[:5], 'pairs_bad': pair_bad[:5], 'n_pairs': n_pairs,
                                             'market_same': mk_prob_same})

    # (d) EV means
    means_b = AA.compute_means(snap_b)
    means_d = AA.compute_means(snap_d)
    n_bit, ulp_max, abs_max, diff_keys = 0, 0.0, 0.0, []
    order_repro_bad, fsum_bad = [], []
    ulp_hist, ulp_abs_max, n_el_diff, n_el_mixed, bound_bad = Counter(), 0.0, 0, 0, []
    for k in keys_all:
        if _bytes(means_b[k]) == _bytes(means_d[k]):
            n_bit += 1
        else:
            diff_keys.append(AA.key_str(k))
            ulp_max = max(ulp_max, _ulp_diff(means_b[k], means_d[k]))
            abs_max = max(abs_max, float(np.max(np.abs(means_b[k] - means_d[k]))))
        if k[0] != 'market':
            A, w = snap_b[k]['A'], snap_b[k]['w']
            p = perms[DI.block_of(k)]
            reordered = np.sum(w[p][:, None] * A[p, :], axis=0).reshape(1, -1)
            if _bytes(reordered) != _bytes(means_d[k]):
                order_repro_bad.append(AA.key_str(k))
            Ad, wd = snap_d[k]['A'], snap_d[k]['w']
            fb = np.array([math.fsum(float(w[s]) * float(A[s, t]) for s in range(N_SCEN)) for t in range(A.shape[1])])
            fd = np.array([math.fsum(float(wd[s]) * float(Ad[s, t]) for s in range(N_SCEN)) for t in range(Ad.shape[1])])
            if _bytes(fb) != _bytes(fd):
                fsum_bad.append(AA.key_str(k))
            # per element: the difference in ulp of the result, and in ulp of the textbook scale sum_s |w_s A[s, t]|
            mb, md = means_b[k].ravel(), means_d[k].ravel()
            dd = np.abs(mb - md)
            el = dd > 0
            if np.any(el):
                n_el_diff += int(np.sum(el))
                u_res = dd[el] / np.spacing(np.maximum(np.abs(mb[el]), np.abs(md[el])))
                scale = np.sum(np.abs(w[:, None] * A), axis=0)[el]
                u_abs = dd[el] / np.spacing(scale)
                ulp_hist.update(float(v) for v in u_res)
                ulp_abs_max = max(ulp_abs_max, float(np.max(u_abs)))
                n_el_mixed += int(np.sum((np.any(A > 0, axis=0) & np.any(A < 0, axis=0))[el]))
                if np.any(u_abs > EV_BOUND_ULP_OF_ABS_SUM):
                    bound_bad.append(AA.key_str(k))
    ev_ok = n_bit == len(keys_all) or (not order_repro_bad and not fsum_bad and not bound_bad)
    _ck(pre + 'EV means (audit A compute_means) unchanged: bitwise, or differing only by summation order (reorder-'
              'reproduced bitwise, fsum-identical, |diff| <= 4 ulp of sum_s |w_s A[s,t]|)', ev_ok,
        {'bitwise': len(keys_all), 'else': 'reorder-reproduced, fsum-identical, <= 4 ulp of sum |terms|'},
        {'n_bitwise': n_bit, 'n_arrays': len(keys_all), 'n_differ': len(diff_keys), 'max_ulp_of_result': ulp_max,
         'max_ulp_of_abs_sum': ulp_abs_max, 'max_abs': abs_max, 'reorder_not_reproducing': len(order_repro_bad),
         'fsum_differ': len(fsum_bad), 'over_bound': len(bound_bad)})
    _pred(pre + 'EV: Planner expectation "bitwise or to 1 ulp": every element of every mean within 1 ulp of the result',
        ulp_max <= 1.0, '<= 1 ulp', {'max_ulp_of_result': ulp_max,
                                     'histogram_ulp_of_result': {str(k): v for k, v in sorted(ulp_hist.items())}})
    _ck(pre + 'EV: base mean recomputed with the rows in the permuted order == dependence mean, bitwise (every '
              'operation array)', not order_repro_bad, 0, order_repro_bad[:5])
    _ck(pre + 'EV: exactly-rounded mean (math.fsum per hour) bitwise identical, base vs dependence (every operation '
              'array)', not fsum_bad, 0, fsum_bad[:5])
    # vs the stored EV profiles file of this grid (keys it injects)
    ev_store = {'path': g['ev_profiles'], 'sha256': _sha(os.path.join(REPO, g['ev_profiles']))}
    with open(os.path.join(REPO, g['ev_profiles'])) as handle:
        evp = json.load(handle)
    ev_src = evp.get('source_instance', {}).get('scenario_checksum')
    ev_store['source_scenario_checksum'] = ev_src
    _ck(pre + 'stored EV profiles file was derived from this base instance (source checksum == base checksum)',
        ev_src == base_ck, base_ck, ev_src)
    nb_b, nb_d, ulp_d, n_cmp = 0, 0, 0.0, 0
    for ks, vals in evp['arrays'].items():
        if ks not in key_of:
            continue
        stored = np.asarray(vals, dtype=np.float64).reshape(1, -1)
        n_cmp += 1
        nb_b += _bytes(stored) == _bytes(means_b[key_of[ks]])
        if _bytes(stored) == _bytes(means_d[key_of[ks]]):
            nb_d += 1
        else:
            ulp_d = max(ulp_d, _ulp_diff(stored, means_d[key_of[ks]]))
    ev_store.update({'n_compared': n_cmp, 'n_stored_eq_base_mean_bitwise': nb_b,
                     'n_stored_eq_dependence_mean_bitwise': nb_d, 'max_ulp_stored_vs_dependence_mean': ulp_d})
    _ck(pre + 'stored EV profiles == base means bitwise (every stored key)', n_cmp > 0 and nb_b == n_cmp, n_cmp, nb_b)
    ev_rec = {'definition': AA.FORMULA, 'n_arrays': len(keys_all), 'n_bitwise_equal': n_bit, 'n_differ': len(diff_keys),
              'differ_first': diff_keys[:20], 'differ_by_family': dict(Counter(AA.family_of(key_of[k]) for k in diff_keys)),
              'max_ulp': ulp_max, 'max_abs_diff_pu_or_eur': abs_max,
              'reorder_reproduces_dependence_mean_bitwise': not order_repro_bad,
              'fsum_mean_bitwise_identical': not fsum_bad, 'explanation': EV_EXPLANATION,
              'n_elements_differing': n_el_diff, 'histogram_ulp_of_result': {str(k): v for k, v in sorted(ulp_hist.items())},
              'max_ulp_of_abs_sum': ulp_abs_max, 'bound_ulp_of_abs_sum': EV_BOUND_ULP_OF_ABS_SUM,
              'n_elements_differing_with_mixed_sign_terms': n_el_mixed,
              'stored_ev_profiles_comparison': ev_store}

    # (e) statistic monotone in the new index
    stat_d = DI.daily_net_demand(p_dep)
    mono_bad = [bk for bk, v in stat_d.items() if any(v['values'][s + 1] < v['values'][s] for s in range(N_SCEN - 1))]
    strict = sum(1 for v in stat_d.values() if all(v['values'][s + 1] > v['values'][s] for s in range(N_SCEN - 1)))
    sorted_eq = all(stat_d[bk]['values'] == [stat[bk]['values'][p] for p in perms[bk]] for bk in stat)
    _ck(pre + 'statistic non-decreasing in the new index in every network block (recomputed after the read)',
        not mono_bad, 0, mono_bad[:5])
    _ck(pre + 'recomputed statistic == base statistic taken in permuted order (bitwise, every block)', sorted_eq, True,
        sorted_eq)
    _ck(pre + 'derive() on the re-paired instance is the identity in every block', all(
        DI.comonotone_permutation(v['values']) == list(range(N_SCEN)) for v in stat_d.values()), True, None)

    # (f) object walks
    w_decl = AA.walk_compare(p_dep, p_base)
    _ck(pre + 'object walk vs base, audit A declared exclusions (scenario fields): 0 differences',
        w_decl['n_diffs'] == 0 and not w_decl['uncompared_types'], 0,
        {k: w_decl[k] for k in ('n_diffs', 'diffs_first', 'uncompared_types')})
    with _walk_exclusions(BOOKKEEPING_EXCLUSIONS):
        w_full = AA.walk_compare(p_dep, p_base, max_report=10 ** 7)
    exp_diff = {_tuple_key(k) for k in op_keys if der['permuted_sha256'][k] != der['base_sha256'][k]
                and k not in post_expected}
    got_diff, unparsed = set(), []
    for dd in w_full['diffs_first']:
        m = WALK_PATH_RE.match(dd['path'])
        if not m or not dd['what'].startswith('ndarray differs'):
            unparsed.append(dd)
            continue
        hname = 'TSO' if m.group(1) == 'transmission_network' else f'DSO{m.group(2)}'
        got_diff.add((hname, int(m.group(3)), m.group(4), 'load' if m.group(5) == 'loads' else 'gen', int(m.group(6)),
                      FIELD_OF_ATTR[m.group(7)]))
    _ck(pre + 'object walk vs base, bookkeeping exclusions only: the differences are EXACTLY the operation arrays the '
              'permutation changes (nothing else)', not unparsed and got_diff == exp_diff
        and w_full['n_diffs'] == len(exp_diff) and not w_full['uncompared_types'],
        {'n': len(exp_diff)}, {'n_diffs': w_full['n_diffs'], 'unparsed_first': unparsed[:5],
                               'missing': sorted(map(str, exp_diff - got_diff))[:5],
                               'extra': sorted(map(str, got_diff - exp_diff))[:5],
                               'uncompared': w_full['uncompared_types']})
    walk_rec = {'declared_exclusions': {k: w_decl[k] for k in ('n_leaves_compared', 'n_diffs', 'uncompared_types',
                                                               'excluded_attribute_hits')},
                'bookkeeping_exclusions': {'exclusions': {c: sorted(v) for c, v in BOOKKEEPING_EXCLUSIONS.items()},
                                           'n_leaves_compared': w_full['n_leaves_compared'],
                                           'n_diffs': w_full['n_diffs'], 'n_expected': len(exp_diff),
                                           'diff_fields': dict(Counter(t[-1] for t in got_diff)),
                                           'excluded_attribute_hits': w_full['excluded_attribute_hits']}}

    # (g) the in-child assertion: pass / raise
    tests = {'checksum_point': cap_d['assert_checksum_point']}

    def run_assert(name, planning, **kw):
        try:
            tests[name] = {'raised': False, 'summary': DI.assert_dependence_arrays(planning, prof, **kw)}
        except DI.DependenceInstanceMismatch as exc:
            tests[name] = {'raised': True, 'error': str(exc)[:2000], 'summary': exc.summary}
        return tests[name]
    t = run_assert('dependence_post_read', p_dep, point='post_read', expected_scenario_checksum=dep_ck)
    _ck(pre + 'in-child assertion PASSES on the dependence read (post_read, with its checksum)', not t['raised'],
        'pass', t.get('error'))
    t = run_assert('base_read_index_paired', p_base, point='post_read')
    s = t.get('summary') or {}
    _ck(pre + 'in-child assertion RAISES on the base (index-paired) read, with 0 arrays differing from the base',
        t['raised'] and s.get('n_differ_from_base') == 0 and s.get('n_mismatches', 0) > 0, 'raise; n_differ 0',
        {'raised': t['raised'], 'n_differ_from_base': s.get('n_differ_from_base'), 'n_mismatches': s.get('n_mismatches')})
    t = run_assert('dependence_read_with_base_checksum', p_dep, point='post_read', expected_scenario_checksum=base_ck)
    _ck(pre + 'in-child assertion RAISES on the dependence read when the expected checksum is the base one',
        t['raised'], 'raise', t['raised'])

    # (h) a second, independent read reproduces the instance
    p_re, read_re = production_read(g['case'], os.path.join(gdir, 'reads', 'dependence_reread'),
                                    DI.DependenceHook(profiles=prof))
    re_ck = read_re['scenario_metadata']['combined_scenario_checksum']
    t = run_assert('dependence_reread_post_read', p_re, point='post_read', expected_scenario_checksum=dep_ck)
    _ck(pre + 'second read with the hook: same checksum, assertion passes', re_ck == dep_ck and not t['raised'],
        dep_ck, {'checksum': re_ck, 'raised': t['raised']})
    del p_re

    # (i) blocks changed per network; ties; alternative TSO statistic
    per_net = Counter()
    per_net_total = Counter()
    for bk, ni in non_identity.items():
        per_net_total[bk.split('|')[0]] += 1
        per_net[bk.split('|')[0]] += int(ni)
    _ck(pre + 'ties in the statistic (exact equality within a block): count recorded; tie rule declared',
        True, 'recorded', n_ties)
    alt_perm = {bk: DI.comonotone_permutation(v['values']) for bk, v in cap['stat_alt'].items()}
    alt_diff = sorted(bk for bk in perms if alt_perm[bk] != perms[bk])
    alt_rec = {'definition': 'as STATISTIC but the TSO ADN loads included at their pre-zeroing sampled pd',
               'n_blocks_whose_permutation_would_differ': len(alt_diff), 'blocks': alt_diff,
               'note': 'reporting only; the declared statistic excludes the ADN loads (the model never sees them)'}
    _ck(pre + 'alternative TSO statistic: only TSO blocks can differ', all(b.startswith('TSO|') for b in alt_diff),
        'TSO only', alt_diff[:5])

    # ---------------- 5. summary of the changed dependence ----------------
    nets = [h for h, _ in EVI.holders(p_base)]
    dsos = [h for h in nets if h != 'TSO']
    pairs = [(a, b) for i, a in enumerate(nets) for b in nets[i + 1:]]
    summ_blocks = {}
    for y in sorted(p_base.years):
        for d in p_base.days:
            nd_idx = {h: np.array(stat[DI.block_key(h, y, d)]['values']) for h in nets}
            nd_com = {h: nd_idx[h][perms[DI.block_key(h, y, d)]] for h in nets}
            row = {}
            for tag, v in (('index_paired', nd_idx), ('comonotone', nd_com)):
                sys_all = sum(v[h] for h in nets)
                sys_dso = sum(v[h] for h in dsos)
                row[tag] = {
                    'net_demand_MWh': {h: v[h].tolist() for h in nets},
                    'pearson': {f'{a}~{b}': _pearson(v[a], v[b]) for a, b in pairs},
                    'spearman': {f'{a}~{b}': _spearman(v[a], v[b]) for a, b in pairs},
                    'system_all_MWh': sys_all.tolist(), 'system_dso_MWh': sys_dso.tolist(),
                    'system_all_range_MWh': float(sys_all.max() - sys_all.min()),
                    'system_all_std_MWh': float(np.sqrt(np.mean((sys_all - sys_all.mean()) ** 2))),
                    'system_dso_range_MWh': float(sys_dso.max() - sys_dso.min()),
                    'system_dso_std_MWh': float(np.sqrt(np.mean((sys_dso - sys_dso.mean()) ** 2))),
                    'system_all_mean_MWh': float(sys_all.mean()), 'system_dso_mean_MWh': float(sys_dso.mean())}
            for k in ('system_all_range_MWh', 'system_all_std_MWh', 'system_dso_range_MWh', 'system_dso_std_MWh'):
                a, b = row['index_paired'][k], row['comonotone'][k]
                row[f'ratio_{k}'] = (b / a) if a > 0 else None
            summ_blocks[f'{y}|{d}'] = row
    agg = {}
    for tag in ('index_paired', 'comonotone'):
        agg[tag] = {}
        for metric in ('pearson', 'spearman'):
            agg[tag][metric] = {}
            for a, b in pairs:
                vals = [r[tag][metric][f'{a}~{b}'] for r in summ_blocks.values() if r[tag][metric][f'{a}~{b}'] is not None]
                agg[tag][metric][f'{a}~{b}'] = {'mean': float(np.mean(vals)) if vals else None,
                                                'min': float(np.min(vals)) if vals else None,
                                                'max': float(np.max(vals)) if vals else None,
                                                'n_negative': sum(1 for x in vals if x < 0), 'n_blocks': len(vals)}
    widen = {}
    for k in ('system_all_range_MWh', 'system_all_std_MWh', 'system_dso_range_MWh', 'system_dso_std_MWh'):
        r = [row[f'ratio_{k}'] for row in summ_blocks.values() if row[f'ratio_{k}'] is not None]
        widen[k] = {'n_blocks_wider': sum(1 for row in summ_blocks.values()
                                          if row['comonotone'][k] > row['index_paired'][k] * (1 + 1e-12)),
                    'n_blocks_equal': sum(1 for row in summ_blocks.values()
                                          if abs(row['comonotone'][k] - row['index_paired'][k])
                                          <= 1e-12 * max(1.0, row['index_paired'][k])),
                    'n_blocks_narrower': sum(1 for row in summ_blocks.values()
                                             if row['comonotone'][k] < row['index_paired'][k] * (1 - 1e-12)),
                    'ratio_mean': float(np.mean(r)) if r else None, 'ratio_min': float(np.min(r)) if r else None,
                    'ratio_max': float(np.max(r)) if r else None}
    _ck(pre + 'system net-demand range (all networks) under comonotone pairing >= index pairing in every block '
              '(sum of maxima / minima)', widen['system_all_range_MWh']['n_blocks_narrower'] == 0, 0,
        widen['system_all_range_MWh'])
    for k in ('system_all_std_MWh', 'system_dso_std_MWh', 'system_all_range_MWh', 'system_dso_range_MWh'):
        wv = widen[k]
        _pred(pre + f'task expectation "the spread of total system net demand widens": {k} comonotone >= index-paired '
                    f'in every block and > in at least one', wv['n_blocks_narrower'] == 0 and wv['n_blocks_wider'] >= 1,
              'narrower 0, wider >= 1', wv)
    spear_one = all(v == 1.0 for r in summ_blocks.values() for v in r['comonotone']['spearman'].values()
                    if v is not None)
    _ck(pre + 'comonotone pairing: Spearman == 1 for every network pair in every block (no ties)', spear_one or n_ties > 0,
        1.0, spear_one)
    summary = {'grid': gname, 'networks': nets, 'statistic': DI.STATISTIC,
               'note': ('TSO net demand = minus TSO RES availability (every TSO load is an ADN load); correlations '
                        'over the 3 scenarios of each (year, day) block, equal weights (1/3); Spearman with average '
                        'ranks; system = sum over the networks of the scenario-s net demand (joint scenario s)'),
               'per_block': summ_blocks, 'aggregate_over_blocks': agg, 'system_spread_change': widen}

    # ---------------- records ----------------
    nb_rec = {'per_network_non_identity': dict(per_net), 'per_network_blocks': dict(per_net_total),
              'total_non_identity': sum(per_net.values()), 'total_blocks': sum(per_net_total.values())}
    record = {
        'schema': 'p515_s56_dependence_instance_record_v1',
        'stage': 'P5.15 S56 dependence stress-test instance (author decision 3; zero solves)', 'grid': gname,
        'utc': datetime.now(timezone.utc).isoformat(), 'instance_label': g['label'],
        'description': profiles['description'],
        'base_instance': profiles['base_instance'],
        'case_path': g['case'], 'case_sha256': case_sha,
        'case_note': ('the re-paired instance has NO case file of its own: it is the base case read through the '
                      'production reader with p515_s56_dependence_instance.DependenceHook(load_profiles(<profiles>)) '
                      'installed at shared_resources_planning._compute_scenario_metadata'),
        'profiles_path': _rel(prof_path) if prof_path.startswith(REPO + os.sep) else prof_path,
        'profiles_sha256': _sha(prof_path),
        'permutations_path': os.path.relpath(perm_path, OUT_DIR), 'permutations_sha256': _sha(perm_path),
        'scenario_checksum': dep_ck, 'base_scenario_checksum': base_ck,
        'warning': ('read WITHOUT the hook the case is the base (index-paired) instance (checksum '
                    f'{base_ck[:8]}...); a child must call p515_s56_dependence_instance.assert_dependence_arrays('
                    'planning, profiles, point="post_read", expected_scenario_checksum=<scenario_checksum>) after its '
                    'read'),
        'definition': {'statistic': DI.STATISTIC, 'tie_rule': DI.TIE_RULE, 'coherence': profiles['coherence'],
                       'array_sha': DI.ARRAY_SHA_DEFINITION},
        'blocks_changed': nb_rec, 'n_ties': n_ties, 'ties': profiles['ties'],
        'n_arrays_changed': profiles['n_arrays_changed'],
        'loading_route': ('p515_s44_scale_measurement.read_planning_from_derived_case on the base case + '
                          'p515_s56_dependence_instance.DependenceHook(profiles) at shared_resources_planning.'
                          '_compute_scenario_metadata'),
        'assertion': {'module': 'p515_s56_dependence_instance.py',
                      'module_sha256': _sha(os.path.join(REPO, 'p515_s56_dependence_instance.py')),
                      'function': 'assert_dependence_arrays(planning, profiles, point="post_read", '
                                  'expected_scenario_checksum=None)'},
    }
    detail = {'reads': {'base': read_base, 'dependence': read_dep, 'dependence_reread': read_re},
              'coverage': {'uncovered_n_by_T_arrays': uncovered, 'other_arrays_with_first_dim_n': dict(other_n)},
              'post_read_changed_keys': post_changed, 'ev_means': ev_rec, 'object_walks': walk_rec,
              'assertion_tests': tests, 'blocks_changed': nb_rec, 'ties': profiles['ties'], 'n_ties': n_ties,
              'alternative_tso_statistic': alt_rec, 'n_operation_arrays': len(op_keys), 'n_market_arrays': len(mk_keys),
              'n_arrays_changed': profiles['n_arrays_changed'],
              'statistic_monotone': {'n_blocks': len(stat_d), 'n_non_decreasing': len(stat_d) - len(mono_bad),
                                     'n_strictly_increasing': strict}}
    return record, detail, summary, gdir


def p_base_holder_block(planning, hname, y, d):
    holder = planning.transmission_network if hname == 'TSO' else planning.distribution_networks[int(hname[3:])]
    return holder.network[y][d]


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    started = datetime.now(timezone.utc).isoformat()
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if os.path.exists(OUT_DIR):
        _log(f'REFUSE: {OUT_DIR} exists')
        return 2
    prov_start = _provenance()
    if prov_start['tracked_status_porcelain'].strip() and PROV_REPO == REPO:
        _log('REFUSE: the working tree carries uncommitted changes to tracked files; run from a clean `git archive '
             'HEAD` export (--provenance-repo <repo> --out <repo>/' + OUT_REL + f'):\n{prov_start["tracked_status_porcelain"]}')
        return 2
    os.makedirs(OUT_DIR)   # exist_ok=False: refuses if it appeared meanwhile
    _log(f'output {OUT_DIR}; code {REPO}; provenance {PROV_REPO} HEAD {prov_start["git_head"]}')
    inputs = {rel: _input_record(rel) for rel in DATA_INPUTS + [g[k] for g in GRIDS
                                                                 for k in ('case', 'record', 'ev_profiles')]}
    bad_inputs = {k: v for k, v in inputs.items() if v['tracked_at_head'] and not v['equals_head']}
    _ck('every tracked data input == its committed (HEAD) bytes', not bad_inputs, {}, bad_inputs)

    records, details, summaries = {}, {}, {}
    for g in GRIDS:
        _log(f'=== grid {g["grid"]} ===')
        rec, det, summ, gdir = build_grid(g)
        records[g['grid']], details[g['grid']], summaries[g['grid']] = rec, det, summ

    aa_fail = AA.GUARD.verify(0)
    _ck('audit A module guard verify(0) == []', aa_fail == [], [], aa_fail)
    guard_fail = GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', guard_fail == [], [], guard_fail)
    guards = {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                      'verify_0': guard_fail},
              'audit_A_module': {'label': AA.GUARD.label, 'counts': dict(AA.GUARD.counts), 'verify_0': aa_fail}}
    mod_audit = _module_path_audit()
    _ck('every loaded tracked repository module == its committed (HEAD) bytes',
        not mod_audit['tracked_modules_not_equal_head'], {}, mod_audit['tracked_modules_not_equal_head'])
    _ck('no repository module loaded from outside the code directory',
        not mod_audit['modules_loaded_from_provenance_repo_outside_code_dir'], {},
        mod_audit['modules_loaded_from_provenance_repo_outside_code_dir'])
    prov_end = _provenance()
    _ck('provenance: HEAD unchanged during the run', prov_end['git_head'] == prov_start['git_head'],
        prov_start['git_head'], prov_end['git_head'])
    failed = [c['check'] for c in CHECKS if not c['ok']]
    for gname, rec in records.items():
        rec.update({'git_head': prov_start['git_head'], 'script': SCRIPT, 'script_sha256': _sha(os.path.abspath(__file__)),
                    'all_checks_pass': not failed, 'failed_checks': failed,
                    'checks': [c for c in CHECKS if c['check'].startswith(f'[{gname}] ')],
                    'predictions': [q for q in PREDICTIONS if q['prediction'].startswith(f'[{gname}] ')],
                    'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) armed for the whole '
                                   'run, verify(0))', 'guards': guards})
        with open(os.path.join(OUT_DIR, gname, 'instance_record.json'), 'x') as handle:
            GRIO.dump(rec, handle, indent=1, default=GRIO.json_default_item)
    payload = {
        'schema': 'p515_s56_dependence_checks_v1', 'script': SCRIPT, 'script_sha256': _sha(os.path.abspath(__file__)),
        'module_sha256': _sha(os.path.join(REPO, 'p515_s56_dependence_instance.py')),
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - STARTED,
        'interpreter': sys.executable, 'provenance_start': prov_start, 'provenance_end': prov_end,
        'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) armed for the whole run, verify(0))',
        'definition': {'statistic': DI.STATISTIC, 'tie_rule': DI.TIE_RULE, 'array_sha': DI.ARRAY_SHA_DEFINITION},
        'instances': {g: {'label': r['instance_label'], 'base_case': r['case_path'], 'base_checksum':
                          r['base_scenario_checksum'], 'scenario_checksum': r['scenario_checksum'],
                          'profiles': r['profiles_path'], 'profiles_sha256': r['profiles_sha256']}
                      for g, r in records.items()},
        'per_grid': details, 'inputs': inputs, 'module_path_audit': mod_audit,
        'checks': CHECKS, 'failed_checks': failed, 'all_checks_ok': not failed, 'guards': guards,
        'predictions': PREDICTIONS, 'predictions_failed': [q['prediction'] for q in PREDICTIONS if not q['held']],
    }
    with open(os.path.join(OUT_DIR, 'dependence_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(OUT_DIR, 'dependence_summary.json'), 'x') as handle:
        GRIO.dump({'schema': 'p515_s56_dependence_summary_v1', 'zero_solves': True, 'grids': summaries}, handle,
                  indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            pth = os.path.join(root, f)
            outputs[os.path.relpath(pth, OUT_DIR)] = _sha(pth)
    manifest = {'outputs': outputs, 'inputs': inputs,
                'code': {'p515_s56_dependence_instance.py': _sha(os.path.join(REPO, 'p515_s56_dependence_instance.py')),
                         SCRIPT: _sha(os.path.abspath(__file__))},
                'note': 'the launch logs (siblings of this directory) are hashed in the Worker report'}
    with open(os.path.join(OUT_DIR, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    for gname, rec in records.items():
        _log(f'{gname}: {rec["instance_label"]} checksum {rec["scenario_checksum"]} (base {rec["base_scenario_checksum"]}); '
             f'profiles sha256 {rec["profiles_sha256"]}; blocks changed {rec["blocks_changed"]}')
    _log(f'all_checks_ok = {not failed}; {len(CHECKS)} checks; failed = {failed}; predictions failed = '
         f'{[q["prediction"] for q in PREDICTIONS if not q["held"]]}')
    return 0 if not failed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
