"""P5.15 S56 -- the S56 search neighbourhood N(x) on the staged (node, year) lattice, and diagnostic 1 (ZERO SOLVES).

Authority: S56_PLAN_v2.md, Expert item X8 (neighbourhood additions (d)-(f) to Addendum 88 R3), adopted in S56_PLAN_v3.md
section 4; TASKS.md S56 "Zero-solve diagnostics before the pilot spec" item (1). The lattice, its admissibility rule, its
canonical form and the R3 neighbourhood are those of `p516_lattice` (imported, NOT edited, NOT re-implemented).
RESUMED 2026-11-23: the 2026-11-21 launch into diag_neighbourhood_residual_v1/ refused at import (the `git archive`
export carried no .env, so production's SolverParameters exited for want of NLP_SOLVER_PATH; no output written). That
directory is preserved untouched; this script refuses to write into it. Output now goes to diag_neighbourhood_residual_v2/.

THE NEIGHBOURHOOD N(x). x is an admissible point of `p516_lattice.GeneralLattice` (z in Z^{2|N||Y|}, (zP, zE) per
(node, year) cohort; P = 0.25 zP MVA, E = 0.5 zE MWh). "Active" cohort = (zP, zE) != (0, 0). Move classes, generated
in this order; within a class the order is cohort order (node-major, then year, the lattice order), then the step
order listed:

  R3  Addendum 88 R3 exactly as built by `p516_lattice.GeneralLattice.neighbourhood(x)` (its admissible, de-duplicated
      `points`, in its own order): (a) coordinate steps zP+-1, zE+-1 on active cohorts; (b) activation of inactive
      cohorts to (1, 1) / (1, 2) granules and deactivation of active smallest cohorts ((1, 1) or (1, 2)) to (0, 0);
      (c) one-block shifts y -> y-1, y+1 of an active cohort at the same (P, E), merged additively into the target.
  d   REMOVAL of ANY active cohort: (zP, zE) -> (0, 0) on one active cohort (R3(b) removes only the smallest cohorts).
  e   DIAGONAL RESIZE of one ACTIVE cohort: (zP, zE) -> (zP + dP, zE + dE), (dP, dE) in DIAGONAL_STEPS =
      ((+1, +1), (-1, -1), (+1, +2), (-1, -2)) -- the SAME-SIGN combinations only. Why: the duration window
      zP <= zE <= 2 zP is a cone whose extreme rays are (1, 1) (2 h) and (1, 2) (4 h); same-sign steps move a cohort
      along those rays, i.e. scale it at (near-)constant duration. (+1, +1) and (+1, +2) are duration-feasible from
      EVERY admissible cohort (zP+1 <= zE+1 <= 2zP+2; zP+1 <= zE+2 <= 2zP+2); (-1, -1) is duration-feasible unless
      zE = 2 zP (4 h edge), (-1, -2) unless zE = zP (2 h edge). These are exactly the size changes the coordinate
      steps cannot make (a single zP or zE step leaves the window from either edge). The opposite-sign combinations
      ((+1, -1), (-1, +1), (+1, -2), (-1, +2)) change the duration by >= 1 granule ratio at about constant size; they
      are NOT part of N(x) and are counted separately as the informational class `e_opp` (OPPOSITE_DIAGONAL_STEPS),
      so the Planner can rule on them with the counts in hand. Inactive cohorts are not resized: (+1, +1) / (+1, +2)
      from (0, 0) are the R3(b) activations, and the negative steps leave the lattice.
  f   REDISTRIBUTION PAIRS: one R3 DECREASE on cohort i combined with one R3 INCREASE on cohort j != i (any node,
      any year), generated ONLY when that increase applied alone to x is BUDGET-INFEASIBLE, i.e. I(x + inc) > B with
      I the lattice's `investment_cost` (the master's investment row; the budget clause of `GeneralLattice.reasons`).
      R3 decreases on cohort i (active): zP-1, zE-1, and deactivation to (0, 0) when i is a smallest cohort.
      R3 increases on cohort j: zP+1, zE+1 when j is active; activation to (1, 1) and (1, 2) when j is inactive.
      (The same primitives as the informational redistribution moves of reuse audit J, `r3_primitive_moves`.)
      An increase that is budget-infeasible AND violates another family alone (duration / cumulative cap) is still
      a trigger; the pair is then decided by the admissibility rule like every other point.

ADMISSIBILITY: `GeneralLattice.reasons(z) == []` (the production rule as mirrored by p516_lattice). The diagnostic
re-decides EVERY generated raw point with production's own first-stage check
(`shared_resources_planning._check_candidate_first_stage_feasibility` on the candidate whose total capacities
`_rebuild_candidate_total_capacities` builds) and fails if the two disagree on any point.
CANONICAL FORM / DE-DUPLICATION: points are identified by z (the canonical v2 form of p516_lattice is a bijection of z;
the diagnostic asserts key-uniqueness == z-uniqueness on the named points). A point is kept once, attributed to the
FIRST class (order R3, d, e, f) that generates it admissibly; x itself is never a neighbour. Counts are reported
both as `new` (after de-duplication against x, R3 and the earlier classes) and as `admissible` (before it).

DIAGNOSTIC 1 (`__main__`): |N(x)| for every admissible point of the staged domain on the 3 x 3 SRP1-grid instance
(2025/2030/2035; 37,041 plans + x0 = 37,042, audit J), enumerated as audit J enumerated it (per-node cohort triples with
cumulative zE <= max_capacity / 0.5 and closed-form cost <= B + 1 EUR, every candidate decided by production's check).
Full enumeration (no sampling). Distribution (min / median / max) overall, by number of active cohorts and by budget-
slack band; named points x0, the reference unit (n7 0.25 MVA / 1 MWh, 2025) and the 2030 plan (n5 0.25/0.5 + n7
1/3.5, 2030); plans with |N| > COMPLETION_CAP (60); per-class contribution relative to |R3|.

GUARDS (`__main__` only): SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly
0; pickle.load / pickle.loads blocked and counted (0). No model is built. Importing this module installs nothing.

Output: a NEW directory (default data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/; refuses if it holds anything
other than this script's own launch logs, or if another copy is running): neighbourhood_sizes.json,
neighbourhood_sizes_table.csv, manifest_neighbourhood_sha256.json.
Launch (attached, both streams captured; may run from a clean `git archive` export with --data-root the repository):
    mkdir data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_neighbourhood.py \\
        > data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/launch_neighbourhood_stdout.log \\
        2> data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/launch_neighbourhood_stderr.log
Exit: 0 written and every check holds; 3 written with failed checks; 2 refused before computing.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

if __name__ == '__main__':
    from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

    GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 diagnostic 1 neighbourhood sizes (never solves)').install()
    import pickle  # noqa: E402

    PICKLE_COUNTS = {'load': 0, 'loads': 0}

    def _blocked_load(*_a, **_k):
        PICKLE_COUNTS['load'] += 1
        raise RuntimeError('diagnostic 1: pickle.load is blocked (no model loads)')

    def _blocked_loads(*_a, **_k):
        PICKLE_COUNTS['loads'] += 1
        raise RuntimeError('diagnostic 1: pickle.loads is blocked (no model loads)')

    pickle.load, pickle.loads = _blocked_load, _blocked_loads

import p516_lattice as GL  # noqa: E402  (stdlib-only at import)

CLASS_ORDER = ('r3', 'd', 'e', 'f')
INFO_CLASSES = ('e_opp',)
DIAGONAL_STEPS = ((+1, +1), (-1, -1), (+1, +2), (-1, -2))
OPPOSITE_DIAGONAL_STEPS = ((+1, -1), (-1, +1), (+1, -2), (-1, +2))  # informational only, NOT in N(x)
COMPLETION_CAP = GL.COMPLETION_CAP


# ======================================================================================================================
#  primitives
# ======================================================================================================================
def r3_decreases(lat, z, n, y):
    """R3 single-cohort decreases on (n, y): zP-1, zE-1 when active; deactivation when it is a smallest cohort."""
    zp, ze = lat.pair(z, n, y)
    if (zp, ze) == (0, 0):
        return []
    out = [('zP-1', (zp - 1, ze)), ('zE-1', (zp, ze - 1))]
    if (zp, ze) in GL.SMALLEST_COHORTS:
        out.append(('deactivate', (0, 0)))
    return out


def r3_increases(lat, z, n, y):
    """R3 single-cohort increases on (n, y): zP+1, zE+1 when active; activations (1, 1), (1, 2) when inactive."""
    zp, ze = lat.pair(z, n, y)
    if (zp, ze) != (0, 0):
        return [('zP+1', (zp + 1, ze)), ('zE+1', (zp, ze + 1))]
    return [(f'activate_{GL.GRANULE_P_MVA * a:g}_{GL.GRANULE_E_MWH * b:g}', (a, b)) for a, b in GL.ACTIVATION_PAIRS]


def raw_added_moves(lat, z_inc):
    """Raw (class, move, z) of the added classes d, e, f and the informational class e_opp, in generation order, plus
    the trigger record of class f. Nothing is filtered here."""
    active = lat.active_cohorts(z_inc)
    budget = lat.inst.budget_eur
    raw = []
    for n, y in active:                                                       # d
        raw.append(('d', f'remove:n{n}:y{y}', lat.with_pair(z_inc, n, y, 0, 0)))
    for n, y in active:                                                       # e
        zp, ze = lat.pair(z_inc, n, y)
        for dp, de in DIAGONAL_STEPS:
            raw.append(('e', f'diag:{dp:+d}{de:+d}:n{n}:y{y}', lat.with_pair(z_inc, n, y, zp + dp, ze + de)))
    triggers = []                                                             # f
    for nj, yj in lat.cohorts:
        for inc_name, (ip, ie) in r3_increases(lat, z_inc, nj, yj):
            z_j = lat.with_pair(z_inc, nj, yj, ip, ie)
            i_alone = lat.investment_cost(z_j)
            if i_alone > budget:
                triggers.append({'cohort': (nj, yj), 'increase': inc_name, 'z_alone': z_j, 'I_alone_eur': i_alone,
                                 'alone_reasons': lat.reasons(z_j)})
    for trig in triggers:
        nj, yj = trig['cohort']
        z_j = trig['z_alone']
        for ni, yi in active:
            if (ni, yi) == (nj, yj):
                continue
            for dec_name, (dp, de) in r3_decreases(lat, z_inc, ni, yi):
                raw.append(('f', f'redist:{dec_name}:n{ni}:y{yi}+{trig["increase"]}:n{nj}:y{yj}',
                            lat.with_pair(z_j, ni, yi, dp, de)))
    for n, y in active:                                                       # e_opp (informational)
        zp, ze = lat.pair(z_inc, n, y)
        for dp, de in OPPOSITE_DIAGONAL_STEPS:
            raw.append(('e_opp', f'diagopp:{dp:+d}{de:+d}:n{n}:y{y}', lat.with_pair(z_inc, n, y, zp + dp, ze + de)))
    return raw, triggers


def s56_neighbourhood(lat, z_inc, keep_points=True, decide=None):
    """N(x) = R3 + d + e + f (module docstring). `decide(z) -> bool`, if given, is called on EVERY raw point (R3 raw
    points included, recovered from the R3 record) and its verdict compared with `GeneralLattice.reasons`; the
    disagreements are returned (empty = the production rule and the lattice agree on every generated point).
    Returns counts by class, the points (if `keep_points`), the f triggers and the disagreements."""
    z_inc = lat.check_z(z_inc)
    r3 = lat.neighbourhood(z_inc)              # raises if x is inadmissible
    seen = {z_inc: 'incumbent'}
    points = []
    disagreements = []
    counts = {c: {'raw': 0, 'admissible': 0, 'new': 0, 'dup_of_earlier': 0, 'dropped_by_reason_class': {}}
              for c in CLASS_ORDER + INFO_CLASSES}
    r3_raw = r3['n_raw']
    counts['r3']['raw'] = r3_raw
    counts['r3']['raw_by_part'] = r3['n_raw_by_part']
    counts['r3']['admissible'] = r3['n_admissible'] + len(r3['duplicates'])
    counts['r3']['dropped_by_reason_class'] = dict(r3['dropped_by_reason_class'])
    counts['r3']['dup_of_earlier'] = len(r3['duplicates'])
    for p in r3['points']:
        z = tuple(p['z'])
        seen[z] = p['move']
        counts['r3']['new'] += 1
        if keep_points:
            points.append({'class': 'r3', 'part': p['part'], 'move': p['move'], 'z': z})
    counts['r3']['admissible_by_part'] = {part: sum(1 for p in r3['points'] if p['part'] == part)
                                          for part in ('a', 'b', 'c')}
    if decide is not None:
        for p in r3['points']:
            if decide(tuple(p['z'])) is not True:
                disagreements.append({'class': 'r3', 'move': p['move'], 'z': list(p['z']), 'lattice': True})
        for p in r3['dropped']:
            if decide(tuple(p['z'])) is not False:
                disagreements.append({'class': 'r3', 'move': p['move'], 'z': list(p['z']), 'lattice': False})
    raw, triggers = raw_added_moves(lat, z_inc)
    seen_info = dict(seen)
    for cls, move, z in raw:
        c = counts[cls]
        c['raw'] += 1
        why = lat.reasons(z)
        ok = not why
        if decide is not None and decide(z) is not ok:
            disagreements.append({'class': cls, 'move': move, 'z': list(z), 'lattice': ok})
        if not ok:
            for rc in sorted(set(lat.reason_classes(why))):
                c['dropped_by_reason_class'][rc] = c['dropped_by_reason_class'].get(rc, 0) + 1
            continue
        c['admissible'] += 1
        book = seen_info if cls in INFO_CLASSES else seen
        if z in book:
            c['dup_of_earlier'] += 1
            continue
        book[z] = move
        if cls not in INFO_CLASSES:
            seen_info[z] = move
        c['new'] += 1
        if keep_points:
            points.append({'class': cls, 'part': None, 'move': move, 'z': z})
    size = sum(counts[c]['new'] for c in CLASS_ORDER)
    f_trig_only_budget = sum(1 for t in triggers if all(r.startswith('budget') for r in t['alone_reasons']))
    return {'z': z_inc, 'A': len(lat.active_cohorts(z_inc)), 'I_x_eur': lat.investment_cost(z_inc),
            'size': size, 'size_r3': counts['r3']['new'], 'size_with_e_opp_info': size + counts['e_opp']['new'],
            'over_cap': size > COMPLETION_CAP, 'counts': counts,
            'f_triggers': len(triggers), 'f_triggers_budget_sole_reason': f_trig_only_budget,
            'f_triggers_detail': [{'cohort': list(t['cohort']), 'increase': t['increase'],
                                   'I_alone_eur': t['I_alone_eur'], 'alone_reasons': t['alone_reasons']}
                                  for t in triggers],
            'points': points, 'disagreements': disagreements}


# ======================================================================================================================
#  diagnostic 1
# ======================================================================================================================
def _main():
    import argparse
    import csv
    import hashlib
    import json
    import platform
    import statistics
    import subprocess
    import time
    import types
    from collections import Counter, defaultdict
    from datetime import datetime, timezone
    from itertools import product

    import numpy as np

    import gate_result_io as GRIO
    import shared_energy_storage_data as ESSD
    import shared_resources_planning as SRP

    script = os.path.basename(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=CODE_ROOT,
                    help='repository root holding data/ (inputs) -- the repository when run from an export')
    ap.add_argument('--out', default=os.path.join('data', 'SRP1', 'Results', 'P515S56', 'diag_neighbourhood_residual_v2'),
                    help='output directory, relative to --data-root unless absolute')
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    os.chdir(data_root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    checks = []

    def log(msg):
        print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)

    def ck(name, ok, detail=None):
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))

    def sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def git(argv, cwd):
        r = subprocess.run(['git', *argv], cwd=cwd, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    def input_record(path):
        tracked = git(['ls-files', '--', path], data_root)
        clean = git(['status', '--porcelain', '--', path], data_root) if tracked else None
        return {'path': path, 'sha256': sha(path), 'tracked': bool(tracked and tracked.strip()),
                'clean': (not clean.strip()) if clean is not None else None}

    # ---------------------------------------------------------------- refusals
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    others = [ln.strip() for ln in ps.splitlines() if len(ln.split()) >= 3 and int(ln.split()[0]) != os.getpid()
              and os.path.basename(ln.split()[1]).startswith('python')
              and any(os.path.basename(t) == script for t in ln.split()[2:4])]
    if others:
        log(f'REFUSE: another copy of {script} is running: {others}')
        return 2
    if os.path.basename(os.path.normpath(out_dir)) == 'diag_neighbourhood_residual_v1':
        log(f'REFUSE: {out_dir} is the preserved partial directory of the interrupted 2026-11-21 attempt')
        return 2
    allowed = {'launch_neighbourhood_stdout.log', 'launch_neighbourhood_stderr.log'}
    if os.path.exists(out_dir):
        present = set(os.listdir(out_dir))
        if not present <= allowed:
            log(f'REFUSE: output directory {out_dir} exists and holds {sorted(present - allowed)}')
            return 2
    else:
        os.makedirs(out_dir)

    # ---------------------------------------------------------------- code provenance
    code_files = {}
    for mod_name, mod in (('p516_lattice', GL), ('shared_resources_planning', SRP),
                          ('shared_energy_storage_data', ESSD), ('gate_result_io', GRIO)):
        code_files[mod_name] = {'file': os.path.abspath(mod.__file__), 'sha256': sha(mod.__file__)}
    code_files['this_script'] = {'file': os.path.abspath(__file__), 'sha256': sha(__file__)}
    code_files['p513_solve_profile_guard'] = {'file': os.path.join(CODE_ROOT, 'p513_solve_profile_guard.py'),
                                              'sha256': sha(os.path.join(CODE_ROOT, 'p513_solve_profile_guard.py'))}
    ck('every imported repository module resolved from the code root (no mixed trees)',
       all(os.path.dirname(v['file']) == CODE_ROOT for v in code_files.values()), code_files)
    export_record = None
    marker = os.path.join(CODE_ROOT, 'EXPORT_RECORD.json')
    if os.path.isfile(marker):
        with open(marker) as handle:
            export_record = json.load(handle)

    # ---------------------------------------------------------------- instance, production check
    case_path = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3.json')
    audit_j_path = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'reuse_audit_J', 'reuse_audit_J.json')
    with open(case_path) as handle:
        case = json.load(handle)
    data_dir = os.path.join('data', 'SRP1')
    inst = GL.load_instance_from_case_dict(case, data_dir, case_source={'path': case_path, 'sha256': sha(case_path)})
    lat = GL.GeneralLattice(inst)
    years = {int(y): int(v) for y, v in case['Years'].items()}
    nodes = [int(dn['connection_node_id']) for dn in case['DistributionNetworks']]
    ck('instance nodes (5, 7, 9) and years (2025, 2030, 2035)', tuple(nodes) == (5, 7, 9)
       and tuple(years) == (2025, 2030, 2035), [nodes, list(years)])
    essd = ESSD.SharedEnergyStorageData()
    essd.data_dir = data_dir
    essd.years = years
    essd.discount_factor = float(case['DiscountFactor'])
    essd.params_file = case['SharedEnergyStorage']['params_file']
    essd.read_parameters_from_file()
    essd.create_shared_energy_storages(types.SimpleNamespace(
        years=years, transmission_network=types.SimpleNamespace(active_distribution_network_nodes=nodes),
        distribution_networks={n: types.SimpleNamespace(name=dn['name'])
                               for n, dn in zip(nodes, case['DistributionNetworks'])}))
    essd.data_file = case['SharedEnergyStorage']['data_file']
    essd.read_shared_energy_storage_data_from_file()
    essd.active_distribution_network_nodes = nodes
    pp = types.SimpleNamespace(shared_ess_data=essd)
    prm = essd.params
    ck('production params == lattice instance (budget, max_capacity, E/P ratios, discount)',
       (prm.budget, prm.max_capacity, prm.min_energy_to_power_ratio, prm.max_energy_to_power_ratio,
        essd.discount_factor) == (inst.budget_eur, inst.max_capacity_mwh, inst.min_ratio, inst.max_ratio,
                                  inst.discount_factor),
       [prm.budget, prm.max_capacity, prm.min_energy_to_power_ratio, prm.max_energy_to_power_ratio])
    n_prod_calls = [0]

    def production_ok(z):
        n_prod_calls[0] += 1
        inv = {n: {y: {'s': GL.GRANULE_P_MVA * lat.pair(z, n, y)[0], 'e': GL.GRANULE_E_MWH * lat.pair(z, n, y)[1]}
                   for y in lat.years} for n in lat.nodes}
        cand = {'investment': inv, 'total_capacity': {n: {} for n in lat.nodes}}
        SRP._rebuild_candidate_total_capacities(pp, cand)
        return bool(SRP._check_candidate_first_stage_feasibility(pp, cand)[0])

    # ---------------------------------------------------------------- the staged domain (audit J's enumeration)
    log('enumerating the staged admissible domain (production-decided)')
    ze_cap = int(round(inst.max_capacity_mwh / GL.GRANULE_E_MWH))
    all_alive_last = all(lat.years[-1] in lat.alive_blocks[y] for y in lat.years)
    ck('all cohorts alive in the last block (per-node cumulative cap = sum over cohorts)', all_alive_last,
       {str(k): list(v) for k, v in lat.alive_blocks.items()})
    opts = [(0, 0)] + [(p, e) for e in range(1, ze_cap + 1) for p in range(1, ze_cap + 1) if p <= e <= 2 * p]
    unit = {y: (sum(a for a, _b in lat._coef[y]) * GL.GRANULE_P_MVA, sum(b for _a, b in lat._coef[y]) * GL.GRANULE_E_MWH)
            for y in lat.years}
    budget = inst.budget_eur
    per_node = []
    for combo in product(opts, repeat=len(lat.years)):
        if sum(e for _p, e in combo) > ze_cap:
            continue
        c = sum(unit[y][0] * p + unit[y][1] * e for (p, e), y in zip(combo, lat.years))
        if c <= budget + 1.0:
            per_node.append((c, combo))
    per_node.sort()
    costs = np.array([c for c, _ in per_node])
    domain, n_cand = [], 0
    for c1, t1 in per_node:
        for c2, t2 in per_node:
            if c1 + c2 > budget + 1.0:
                break
            j_end = int(np.searchsorted(costs, budget + 1.0 - c1 - c2, side='right'))
            for _c3, t3 in per_node[:j_end]:
                n_cand += 1
                z = lat.x0()
                for n, trip in zip(lat.nodes, (t1, t2, t3)):
                    for y, (p, e) in zip(lat.years, trip):
                        if (p, e) != (0, 0):
                            z = lat.with_pair(z, n, y, p, e)
                if production_ok(z):
                    domain.append(z)
    domain.sort()
    domain_set = set(domain)
    with open(audit_j_path) as handle:
        audit_j = json.load(handle)
    sd_j = audit_j['Q3']['staged_domain']
    by_a = Counter(len(lat.active_cohorts(z)) for z in domain)
    ck('staged domain size == audit J n_admissible_incl_x0 (37,042 = 37,041 + x0)',
       len(domain) == sd_j['n_admissible_incl_x0'] == 37042, [len(domain), sd_j['n_admissible_incl_x0']])
    ck('staged domain by number of active cohorts == audit J', {str(k): v for k, v in sorted(by_a.items())}
       == sd_j['by_number_of_active_cohorts'], [dict(by_a), sd_j['by_number_of_active_cohorts']])
    ck('every domain point admissible by the lattice rule too', all(not lat.reasons(z) for z in domain))
    ck('x0 in the domain', lat.x0() in domain_set)

    # ---------------------------------------------------------------- slack bands (operational edges)
    gran_p_min = min(unit[y][0] for y in lat.years)
    gran_e_min = min(unit[y][1] for y in lat.years)
    act_min = min(unit[y][0] * a + unit[y][1] * b for y in lat.years for a, b in GL.ACTIVATION_PAIRS)
    edges = [0.0, gran_p_min, gran_e_min, act_min, 250000.0, 500000.0, budget + 1.0]
    edge_names = ['cheapest 0.25 MVA granule', 'cheapest 0.5 MWh granule', 'cheapest activation (0.25/0.5)',
                  '250 kEUR', '500 kEUR', 'B']
    if sorted(edges) != edges:
        raise RuntimeError(f'slack band edges not ascending: {edges}')
    band_labels = [f'[{edges[i]:,.0f}, {edges[i + 1]:,.0f})' for i in range(len(edges) - 1)]

    def band_of(slack):
        for i in range(len(edges) - 1):
            if edges[i] <= slack < edges[i + 1]:
                return band_labels[i]
        raise RuntimeError(f'slack {slack} outside the bands')

    # ---------------------------------------------------------------- named points
    named = {'x0': lat.x0(),
             'reference_unit_n7_0.25_1_y2025': lat.with_pair(lat.x0(), 7, 2025, 1, 2),
             'plan_2030_n5_0.25_0.5__n7_1_3.5': lat.with_pair(lat.with_pair(lat.x0(), 5, 2030, 1, 1), 7, 2030, 4, 7)}
    ck('the 2030 plan I(x) == 992,268.11 EUR', round(lat.investment_cost(named['plan_2030_n5_0.25_0.5__n7_1_3.5']), 2)
       == 992268.11, lat.investment_cost(named['plan_2030_n5_0.25_0.5__n7_1_3.5']))
    named_out = {}
    for name, z in named.items():
        nb = s56_neighbourhood(lat, z, keep_points=True, decide=production_ok)
        keys = [lat.key(p['z']) for p in nb['points']]
        ck(f'{name}: neighbour keys unique == neighbour z unique (canonical de-duplication)',
           len(set(keys)) == len({p['z'] for p in nb['points']}) == len(nb['points']))
        ck(f'{name}: production check == lattice rule on every generated raw point', not nb['disagreements'],
           nb['disagreements'][:5])
        ck(f'{name}: every neighbour in the production-decided staged domain',
           all(p['z'] in domain_set for p in nb['points']))
        j_r3 = audit_j['Q3']['neighbourhoods'][name]['r3_p516']
        ck(f'{name}: |R3| == audit J r3_p516 n_admissible', nb['size_r3'] == j_r3['n_admissible'],
           [nb['size_r3'], j_r3['n_admissible']])
        j_red = audit_j['Q3']['neighbourhoods'][name]['informational_redistribution_pairs']
        f_adm = nb['counts']['f']['admissible']
        ck(f'{name}: class f admissible <= audit J unrestricted redistribution admissible (f is the budget-'
           f'triggered subset)', f_adm <= j_red['n_admissible'], [f_adm, j_red['n_admissible']])
        named_out[name] = {
            'z': list(z), 'label': lat.label(z), 'key': lat.key(z), 'A': nb['A'], 'I_x_eur': nb['I_x_eur'],
            'budget_slack_eur': budget - nb['I_x_eur'], 'slack_band': band_of(budget - nb['I_x_eur']),
            'size': nb['size'], 'size_r3': nb['size_r3'], 'size_with_e_opp_info': nb['size_with_e_opp_info'],
            'over_cap_60': nb['over_cap'], 'counts': nb['counts'], 'f_triggers': nb['f_triggers'],
            'f_triggers_budget_sole_reason': nb['f_triggers_budget_sole_reason'],
            'f_triggers_detail': nb['f_triggers_detail'],
            'audit_J_unrestricted_redistribution_admissible': j_red['n_admissible'],
            'points': [{'class': p['class'], 'part': p['part'], 'move': p['move'], 'label': lat.label(p['z']),
                        'key': lat.key(p['z']), 'I_x_eur': lat.investment_cost(p['z'])} for p in nb['points']]}
        log(f'{name}: |N| = {nb["size"]} (R3 {nb["size_r3"]}; new d/e/f = '
            f'{[nb["counts"][c]["new"] for c in ("d", "e", "f")]}; e_opp info {nb["counts"]["e_opp"]["new"]})')

    # ---------------------------------------------------------------- every plan
    log(f'enumerating N(x) for all {len(domain)} domain points (every raw point re-decided by production)')
    rows = []
    disagreements_total, disagreement_examples = 0, []
    agg = {c: Counter() for c in CLASS_ORDER + INFO_CLASSES}
    trig_total = trig_budget_sole = 0
    t_enum = time.time()
    for idx, z in enumerate(domain):
        nb = s56_neighbourhood(lat, z, keep_points=False, decide=production_ok)
        disagreements_total += len(nb['disagreements'])
        disagreement_examples += nb['disagreements'][:max(0, 5 - len(disagreement_examples))]
        for c in CLASS_ORDER + INFO_CLASSES:
            for k in ('raw', 'admissible', 'new', 'dup_of_earlier'):
                agg[c][k] += nb['counts'][c][k]
            for rc, v in nb['counts'][c]['dropped_by_reason_class'].items():
                agg[c][f'dropped_{rc}'] += v
        trig_total += nb['f_triggers']
        trig_budget_sole += nb['f_triggers_budget_sole_reason']
        slack = budget - nb['I_x_eur']
        cnt = nb['counts']
        rows.append({'index': idx, 'key': lat.key(z), 'label': lat.label(z), 'A': nb['A'],
                     'n_nodes_with_cohorts': len({n for n, _y in lat.active_cohorts(z)}),
                     'I_x_eur': nb['I_x_eur'], 'budget_slack_eur': slack, 'slack_band': band_of(slack),
                     'N_size': nb['size'], 'over_cap_60': nb['over_cap'],
                     'r3': cnt['r3']['new'], 'r3_a': cnt['r3']['admissible_by_part']['a'],
                     'r3_b': cnt['r3']['admissible_by_part']['b'], 'r3_c': cnt['r3']['admissible_by_part']['c'],
                     'd_new': cnt['d']['new'], 'e_new': cnt['e']['new'], 'f_new': cnt['f']['new'],
                     'd_admissible': cnt['d']['admissible'], 'e_admissible': cnt['e']['admissible'],
                     'f_admissible': cnt['f']['admissible'], 'f_raw': cnt['f']['raw'], 'f_triggers': nb['f_triggers'],
                     'e_opp_new_info': cnt['e_opp']['new'], 'N_size_with_e_opp_info': nb['size_with_e_opp_info'],
                     'z': ' '.join(str(v) for v in z)})
        if (idx + 1) % 5000 == 0:
            log(f'  {idx + 1}/{len(domain)} ({time.time() - t_enum:.0f} s)')
    enum_wall = time.time() - t_enum
    ck('production check == lattice rule on EVERY generated raw point of EVERY plan (0 disagreements)',
       disagreements_total == 0, disagreement_examples)

    # ---------------------------------------------------------------- summaries
    def dist(vals):
        vals = list(vals)
        if not vals:
            return None
        a = np.array(vals, dtype=float)
        return {'n': len(vals), 'min': float(a.min()), 'p10': float(np.percentile(a, 10)),
                'p25': float(np.percentile(a, 25)), 'median': float(statistics.median(vals)),
                'p75': float(np.percentile(a, 75)), 'p90': float(np.percentile(a, 90)), 'max': float(a.max()),
                'mean': float(a.mean())}

    sizes = [r['N_size'] for r in rows]
    by_a_rows, by_band_rows = defaultdict(list), defaultdict(list)
    for r in rows:
        by_a_rows[r['A']].append(r)
        by_band_rows[r['slack_band']].append(r)

    def group_summary(rs):
        return {'n_plans': len(rs), 'N_size': dist(r['N_size'] for r in rs), 'r3': dist(r['r3'] for r in rs),
                'd_new': dist(r['d_new'] for r in rs), 'e_new': dist(r['e_new'] for r in rs),
                'f_new': dist(r['f_new'] for r in rs), 'n_over_cap_60': sum(r['over_cap_60'] for r in rs),
                'N_size_with_e_opp_info': dist(r['N_size_with_e_opp_info'] for r in rs),
                'n_over_cap_60_with_e_opp_info': sum(r['N_size_with_e_opp_info'] > COMPLETION_CAP for r in rs)}

    sum_r3 = sum(r['r3'] for r in rows)
    contribution = {}
    for c in ('d', 'e', 'f', 'e_opp'):
        col = f'{c}_new' if c != 'e_opp' else 'e_opp_new_info'
        ratios = [r[col] / r['r3'] for r in rows if r['r3'] > 0]
        contribution[c] = {'sum_new_over_all_plans': sum(r[col] for r in rows),
                           'sum_r3_over_all_plans': sum_r3,
                           'pooled_fraction_of_r3': sum(r[col] for r in rows) / sum_r3,
                           'per_plan_fraction_of_r3': dist(ratios),
                           'n_plans_class_adds_at_least_one': sum(1 for r in rows if r[col] > 0),
                           'n_plans_with_r3_zero_where_class_adds': sum(1 for r in rows if r['r3'] == 0 and r[col] > 0)}
    hist = Counter(sizes)
    over = [r for r in rows if r['over_cap_60']]
    summary = {
        'n_plans': len(rows), 'sampled': False, 'sampling_seed': None,
        'N_size': dist(sizes), 'R3_size': dist(r['r3'] for r in rows),
        'N_size_histogram': {str(k): v for k, v in sorted(hist.items())},
        'n_plans_over_cap_60': len(over), 'completion_cap': COMPLETION_CAP,
        'over_cap_plans': [{k: r[k] for k in ('key', 'label', 'A', 'I_x_eur', 'N_size', 'r3', 'd_new', 'e_new',
                                              'f_new')} for r in sorted(over, key=lambda r: -r['N_size'])[:50]],
        'n_plans_r3_zero': sum(1 for r in rows if r['r3'] == 0),
        'n_plans_r3_le_4': sum(1 for r in rows if r['r3'] <= 4),
        'n_plans_N_le_4': sum(1 for r in rows if r['N_size'] <= 4),
        'by_number_of_active_cohorts': {str(a): group_summary(rs) for a, rs in sorted(by_a_rows.items())},
        'by_budget_slack_band': {b: group_summary(by_band_rows[b]) for b in band_labels if b in by_band_rows},
        'class_contribution_relative_to_r3': contribution,
        'class_totals_over_all_plans': {c: dict(v) for c, v in agg.items()},
        'f_triggers_total': trig_total, 'f_triggers_budget_sole_reason_total': trig_budget_sole,
        'with_e_opp_info': {'N_size': dist(r['N_size_with_e_opp_info'] for r in rows),
                            'n_plans_over_cap_60': sum(r['N_size_with_e_opp_info'] > COMPLETION_CAP for r in rows)},
    }
    ck('|N| >= |R3| on every plan (the added classes never remove points)', all(r['N_size'] >= r['r3'] for r in rows))
    ck('named-point rows of the bulk table == the named-point enumeration',
       all(next(r for r in rows if r['key'] == v['key'])['N_size'] == v['size'] for v in named_out.values()))

    payload = {
        'schema': 'p515_s56_diag_neighbourhood_v1', 'script': script, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0,
        'enumeration_wall_s': enum_wall, 'interpreter': sys.executable, 'python': platform.python_version(),
        'code_root': CODE_ROOT, 'data_root': data_root, 'export_record': export_record,
        'git_head_data_root': (git(['rev-parse', 'HEAD'], data_root) or '').strip(),
        'solve_claim': ('ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any repository '
                        'import, verify(0)); pickle.load/loads blocked (0)'),
        'definition': {
            'neighbourhood': __doc__.split('THE NEIGHBOURHOOD N(x).')[1].split('ADMISSIBILITY:')[0].strip(),
            'admissibility': ('GeneralLattice.reasons(z) == [] (p516_lattice); every generated raw point (R3 raw incl. '
                              'dropped, d, e, f, e_opp) re-decided by production _check_candidate_first_stage_feasibility'
                              ' after _rebuild_candidate_total_capacities; disagreements counted'),
            'dedup': 'by z (bijective with the v2 canonical key); first generating class in order R3, d, e, f',
            'class_order': list(CLASS_ORDER), 'info_classes': list(INFO_CLASSES),
            'diagonal_steps_in_N': [list(s) for s in DIAGONAL_STEPS],
            'opposite_diagonal_steps_informational': [list(s) for s in OPPOSITE_DIAGONAL_STEPS],
            'f_trigger': 'I(x + inc) > B (lattice investment_cost, master term order) for the increase alone',
            'size': '|N(x)| = |R3 new| + |d new| + |e new| + |f new|',
            'class_contribution': ('pooled_fraction_of_r3 = sum over plans of class-new / sum over plans of |R3|; '
                                   'per_plan_fraction_of_r3 = distribution over plans with |R3| > 0 of class-new/|R3|'),
            'slack_bands': {'edges_eur': edges, 'edge_names': ['0'] + edge_names, 'labels': band_labels,
                            'unit_granule_costs_eur': {str(y): {'p_0.25MVA': unit[y][0], 'e_0.5MWh': unit[y][1]}
                                                       for y in lat.years}},
            'domain_enumeration': ('per-node cohort triples over (0,0) and the duration-admissible pairs with '
                                   'P, E <= max_capacity; cumulative zE <= max_capacity/0.5; closed-form cost <= B + 1; '
                                   'node triples with cost <= B + 1; every candidate decided by production (audit J)'),
            'distribution_stats': 'median = statistics.median; p10..p90 = numpy.percentile (linear)'},
        'instance': {'case': case_path, 'sha256': sha(case_path), 'lattice': inst.describe()},
        'domain': {'n_candidates_checked_by_production': n_cand, 'n_admissible_incl_x0': len(domain),
                   'by_number_of_active_cohorts': {str(k): v for k, v in sorted(by_a.items())}},
        'named_points': named_out, 'summary': summary,
        'production_check_calls': n_prod_calls[0], 'disagreements_total': disagreements_total,
        'inputs': {p: input_record(p) for p in (case_path, audit_j_path,
                                                os.path.join(data_dir, 'SharedESS', case['SharedEnergyStorage']['params_file']),
                                                os.path.join(data_dir, 'SharedESS', case['SharedEnergyStorage']['data_file']))},
        'code_files': code_files,
    }
    guard_fail = GUARD.verify(0)
    ck('own SolveProfileGuard(permitted=()) verify(0) == []', not guard_fail, guard_fail)
    ck('pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    payload['guards'] = {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                         'verify_0': guard_fail, 'pickle_counts': dict(PICKLE_COUNTS)}
    payload['checks'] = checks
    payload['failed_checks'] = [c['check'] for c in checks if not c['ok']]
    payload['all_checks_ok'] = all(c['ok'] for c in checks)
    json_path = os.path.join(out_dir, 'neighbourhood_sizes.json')
    csv_path = os.path.join(out_dir, 'neighbourhood_sizes_table.csv')
    with open(csv_path, 'x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(json_path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1)
    manifest = {'outputs': {os.path.basename(json_path): sha(json_path), os.path.basename(csv_path): sha(csv_path)},
                'inputs': payload['inputs'], 'code_files': code_files, 'export_record': export_record,
                'csv_columns': list(rows[0].keys()),
                'note': 'the launch logs are written while the script runs; their sha256 are recorded in the report'}
    with open(os.path.join(out_dir, 'manifest_neighbourhood_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    log(f'|N|: {summary["N_size"]}; over cap 60: {len(over)}; wall {time.time() - t0:.0f} s')
    log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(_main())
