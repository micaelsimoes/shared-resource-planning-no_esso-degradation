"""P5.16 convention change, gate G-CC-4 (zero solves) -- the general (node, year) lattice `p516_lattice.py` against
(1) the old 7-variable lattice (p515_s47_phase_b_record.Lattice) and (2) the production master rows
(shared_energy_storage_data._build_master_problem, candidates loaded by `_load_candidate_solution_into_master_model`).

Authority: P5_16_CONVENTION_CHANGE_SPEC_v4.md section 1 C5 and section 2 G-CC-4; PLANNER_BRIEF_2026-09-13.md
Addendum 88 R3; STEP4_DFO_METHOD.md 1.1-1.2, 5.4.

ZERO SOLVES: a blocking SolveProfileGuard(permitted=()) is armed before any project import and verified at exactly 0/0
at the end (p515_s47_phase_b_record installs its own blocking PARENT_GUARD at import; both are verified).

PARTS
  B1 back-compatibility on SRP1 (years 2025/2030/2035): old `Lattice.domain()` (W2 unit costs, the established
     construction of p515_s47_phase_b_record l.1176-1184) mapped into the general lattice -- each admissible, one
     common nonzero year; an independent enumeration of every admissible general point with <= 1 nonzero year per node
     (per node: none, or one (year, zP, zE) with 1 <= zP <= zE <= 2 zP <= 22, zE <= 11 -- complete for the admissible
     set since zE >= 12 violates the cumulative rule and the rest violates the duration rule); its common-year subset
     maps back onto the old domain; counts equal; reason classes agree on 2,000 seeded inadmissible raw old points.
  B2 oracle = the production master rows on SRP1: the master built from the SRP1 planning problem read zero-solve
     (Diagrams / Results / Logs side-writes redirected under OUT/side_writes), 200 seeded general points (x = 0, the L4
     plan, and 198 drawn in five categories), totals supplied by p56a_oracle.vector_to_candidate (production
     `_rebuild_candidate_total_capacities`), loaded by `_load_candidate_solution_into_master_model`; every active row's
     body evaluated; verdict (tolerance 0) and per-family violated (node, year) sets compared with the module; I(x)
     against pe.value(model.investment_cost).
  B3 neighbourhoods (Addendum 88 R3) on five seeded incumbents (x = 0, a single cohort, a two-node plan, a staged plan,
     a plan at the budget) on the five-block instance years (p515_s44_scale_measurement.derive_case('paper', 1 x 1),
     the derivation spec v4 C4 names) and on SRP1 (where every listed and dropped point is also put to the master rows).
  B4 canonical form / keys: idempotence, round trip, key = harness candidate_key, x = 0 alias, no collisions.

OUTPUT: data/SRP1/Results/P516CC/g_cc4/ (refuses if it exists). run.log (stdout + stderr tee), g_cc4_results.json,
back_compat.json, oracle_points.json, neighbourhoods.json, planning_read_stdout.log, side_writes/.
LAUNCH (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P516CC
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_lattice_gate.py \
        > data/SRP1/Results/P516CC/g_cc4_launch_stdout.log 2> data/SRP1/Results/P516CC/g_cc4_launch_stderr.log
    /opt/anaconda3/envs/opf_env_py311/bin/python -u p516_lattice_gate.py --manifest
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.16 G-CC-4 lattice gate (zero solves)').install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

STAGE = 'P5.16 convention change C5 -- gate G-CC-4 (general lattice; zero solves)'
SCHEMA = 'p516_lattice_gate_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P516CC', 'g_cc4')
OUT = os.path.join(REPO, OUT_REL)
LAUNCH_LOGS_REL = [os.path.join('data', 'SRP1', 'Results', 'P516CC', 'g_cc4_launch_stdout.log'),
                   os.path.join('data', 'SRP1', 'Results', 'P516CC', 'g_cc4_launch_stderr.log')]
SRP1_DIR = os.path.join(REPO, 'data', 'SRP1')
SRP1_CASE = os.path.join(SRP1_DIR, 'SRP1.json')
SEED = 20261031
N_REASON_SAMPLE = 2000
N_ORACLE = 200
TOL_VERDICT = 0.0
I_REL_TOL_GATE = 1e-9
W2_TABLE = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost', 'investment_cost_results.json'),
            'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b'}
HASHED_SOURCES = ['p516_lattice.py', 'p516_lattice_gate.py', 'p515_s47_phase_b_record.py', 'p515_s44_campaign_harness.py',
                  'p515_s44_scale_measurement.py', 'p56a_oracle.py', 'shared_energy_storage_data.py',
                  'shared_resources_planning.py', 'p513_solve_profile_guard.py', 'gate_result_io.py',
                  os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
                  os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'), W2_TABLE['path']]
PREDICTIONS = {
    'recorded': 'in this script before the run (Planner task, P5.16 C5 + G-CC-4)',
    'L1': 'old-domain round trip exact (counts equal)',
    'L2': 'the master-row oracle agrees with the module on 200/200 points and I(x) to <= 1e-12 relative',
    'L3': '|N(x=0)| = 30 on the five-block instance years (all 15 activations x 2 energies admissible under the budget)',
    'L4': 'a staged two-cohort plan of 3 + 3 MWh at one node is rejected by the cumulative rule and by the master row alike',
}


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # recorded, not fatal
        return f'ERROR {exc}'


# ======================================================================================================================
#  manifest mode (after the run)
# ======================================================================================================================
def manifest():
    import gate_result_io as GRIO
    target = os.path.join(OUT, 'manifest_sha256.json')
    if os.path.exists(target):
        raise SystemExit(f'refusing to overwrite {target}')
    files = {}
    for root, _dirs, names in os.walk(OUT):
        for name in sorted(names):
            path = os.path.join(root, name)
            files[os.path.relpath(path, REPO)] = sha256_file(path)
    for rel in LAUNCH_LOGS_REL:
        path = os.path.join(REPO, rel)
        if os.path.exists(path):
            files[rel] = sha256_file(path)
    with open(target, 'w') as handle:
        handle.write(GRIO.dumps({'schema': 'p516_lattice_gate_manifest_v1', 'utc': _utc(), 'files': files},
                                indent=1, sort_keys=True))
    print(f'manifest: {len(files)} files -> {target}')
    return 0


# ======================================================================================================================
#  the run
# ======================================================================================================================
def run(trial_out=None):
    global OUT, OUT_REL
    if trial_out:   # debugging only: an absolute directory OUTSIDE the repository; never the evidence root
        trial_out = os.path.abspath(trial_out)
        if trial_out.startswith(REPO + os.sep):
            raise SystemExit('--trial-out must be outside the repository')
        OUT, OUT_REL = trial_out, trial_out
    if os.path.exists(OUT):
        raise SystemExit(f'refusing: {OUT_REL} exists')
    os.makedirs(OUT)
    log = open(os.path.join(OUT, 'run.log'), 'w')
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    t_start = time.time()
    print(f'{STAGE}\nstart {_utc()} pid {os.getpid()} python {sys.executable}')
    try:
        return _run_body(t_start)
    except BaseException:
        traceback.print_exc()
        print(f'guard counts at error: {GUARD.counts}', file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def _run_body(t_start):
    import gate_result_io as GRIO
    import p515_s47_phase_b_record as PB          # installs its own blocking PARENT_GUARD
    import p515_s44_campaign_harness as H
    import p515_s44_scale_measurement as SM
    import p516_lattice as L
    import p56a_oracle as O
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    import shared_energy_storage_data as SESD
    import shared_resources_planning as SRP
    from shared_resources_planning import SharedResourcesPlanning

    checks = []          # (id, pass, detail)

    def check(cid, ok, detail):
        ok = bool(ok)
        checks.append({'id': cid, 'pass': ok, 'detail': detail})
        print(f"[check] {'PASS' if ok else 'FAIL'} {cid}: {detail if not isinstance(detail, dict) else json.dumps(detail, default=str)[:400]}")
        return ok

    def write(name, payload):
        with open(os.path.join(OUT, name), 'w') as handle:
            handle.write(GRIO.dumps(payload, indent=1))

    # ---- capture-path assertion (CLAUDE.md rule eleven): every required quantity has a capture path, before work ----
    capture_paths = {
        'old_domain': (PB.Lattice, 'domain'), 'old_reasons': (PB.Lattice, 'reasons'),
        'old_z_of_canonical': (PB.Lattice, 'z_of_canonical'), 'w2_costs': (PB, 'unit_costs_from_w2'),
        'old_canonical_candidate': (H, 'canonical_candidate'), 'harness_candidate_key': (H, 'candidate_key'),
        'five_block_years': (SM, 'derive_case'), 'module_instance_loader': (L, 'load_instance'),
        'module_instance_from_dict': (L, 'load_instance_from_case_dict'), 'module_reasons': (L.GeneralLattice, 'reasons'),
        'module_families': (L.GeneralLattice, 'family_violations'), 'module_I': (L.GeneralLattice, 'investment_cost'),
        'module_neighbourhood': (L.GeneralLattice, 'neighbourhood'), 'module_canonical': (L.GeneralLattice, 'canonical_of_z'),
        'module_z_of_canonical': (L.GeneralLattice, 'z_of_canonical'), 'module_canonicalize': (L.GeneralLattice, 'canonicalize'),
        'module_resolution_by_bound': (L, 'resolution_by_bound'),
        'master_build': (SESD.SharedEnergyStorageData, 'build_master_problem'),
        'master_loader': (SESD, '_load_candidate_solution_into_master_model'),
        'totals_rebuild': (SRP, '_rebuild_candidate_total_capacities'), 'candidate_builder': (O, 'vector_to_candidate'),
        'production_first_stage_check': (SRP, '_check_candidate_first_stage_feasibility'),
        'production_I_replica': (O, 'investment_cost'),
    }
    missing = [k for k, (obj, attr) in capture_paths.items() if not callable(getattr(obj, attr, None))]
    if missing:
        raise RuntimeError(f'capture paths missing (fail fast, nothing run): {missing}')
    print(f'[capture] {len(capture_paths)} capture paths present')
    MASTER_COMPONENTS = ('rated_s_capacity', 'rated_e_capacity', 'energy_storage_maximum_capacity',
                         'energy_storage_power_to_energy_factor', 'energy_storage_investment', 'benders_cuts',
                         'investment_cost', 'es_s_investment', 'es_e_investment', 'es_s_rated', 'es_e_rated')

    provenance = {'stage': STAGE, 'schema': SCHEMA, 'started_utc': _utc(), 'python': sys.executable,
                  'git_HEAD': _git('rev-parse', 'HEAD'),
                  'git_status_of_hashed_sources': _git('status', '--porcelain', '--', *HASHED_SOURCES),
                  'sources_sha256': {p: sha256_file(os.path.join(REPO, p)) for p in HASHED_SOURCES},
                  'seed': SEED, 'predictions': PREDICTIONS,
                  'guards': ['p516_lattice_gate GUARD permitted=() (armed before project imports)',
                             'p515_s47_phase_b_record.PARENT_GUARD permitted=() (installed at its import)']}
    if provenance['sources_sha256'][W2_TABLE['path']] != W2_TABLE['sha256']:
        raise RuntimeError('W2 unit-cost table hash differs from the pin')
    print(f"[prov] HEAD {provenance['git_HEAD']}; dirty hashed sources: {provenance['git_status_of_hashed_sources']!r}")

    # ---- instances ----------------------------------------------------------------------------------------------------
    inst = L.load_instance(SRP1_CASE)
    lat = L.GeneralLattice(inst)
    case5, spec5, changes5 = SM.derive_case('paper', {'num_market_scenarios': 1, 'num_operation_scenarios': 1})
    inst5 = L.load_instance_from_case_dict(case5, SRP1_DIR, case_source={
        'derived_by': "p515_s44_scale_measurement.derive_case('paper', {num_market_scenarios: 1, "
                      "num_operation_scenarios: 1})", 'from': SM.SOURCE_CASE_REL, 'changes': changes5})
    lat5 = L.GeneralLattice(inst5)
    print(f'[inst] SRP1 nodes {inst.nodes} years {inst.years} alive {lat.alive_blocks}')
    print(f'[inst] five-block nodes {inst5.nodes} years {inst5.years} alive {lat5.alive_blocks}')
    results = {'provenance': provenance, 'instances': {'srp1': inst.describe(), 'five_block': inst5.describe()},
               'alive_blocks': {'srp1': {str(k): list(v) for k, v in lat.alive_blocks.items()},
                                'five_block': {str(k): list(v) for k, v in lat5.alive_blocks.items()}}}

    check('C0_constants_equal_old_lattice',
          L.GRANULE_P_MVA == PB.GRANULE_P_MVA and L.GRANULE_E_MWH == PB.GRANULE_E_MWH
          and inst.max_capacity_mwh == PB.E_MAX_MWH and inst.budget_eur == PB.BUDGET_EUR
          and inst.nodes == tuple(PB.ACTIVE_NODES) and inst.min_ratio == 2.0 and inst.max_ratio == 4.0,
          {'granules': [L.GRANULE_P_MVA, L.GRANULE_E_MWH], 'old': [PB.GRANULE_P_MVA, PB.GRANULE_E_MWH],
           'max_capacity': inst.max_capacity_mwh, 'budget': inst.budget_eur, 'nodes': list(inst.nodes),
           'ratios': [inst.min_ratio, inst.max_ratio]})
    check('C0_five_block_tcal_norm_5', lat5.alive_blocks[inst5.years[0]] == inst5.years and len(inst5.years) == 5,
          {'years': list(inst5.years), 'alive_first': list(lat5.alive_blocks[inst5.years[0]])})

    # ==================================================================================================================
    #  B1 back-compatibility
    # ==================================================================================================================
    w2 = json.load(open(os.path.join(REPO, W2_TABLE['path'])))
    unit_costs = PB.unit_costs_from_w2(w2)
    old = PB.Lattice(tuple(sorted(unit_costs)), unit_costs)
    check('B1_old_years_equal_instance', old.years == inst.years, {'old': list(old.years), 'instance': list(inst.years)})
    # module's discounted expected unit cost per year vs the W2 table (the old lattice's costs)
    unit_cost_cmp = {}
    for y in inst.years:
        zp1 = lat.with_pair(lat.x0(), inst.nodes[0], y, 4, 0)   # 1 MVA (inadmissible, but I(x) is linear)
        ze1 = lat.with_pair(lat.x0(), inst.nodes[0], y, 0, 2)   # 1 MWh
        unit_cost_cmp[str(y)] = {'power_module': lat.investment_cost(zp1), 'power_w2': old.c_p[y],
                                 'energy_module': lat.investment_cost(ze1), 'energy_w2': old.c_e[y]}
    max_rel_uc = max(abs(v['power_module'] - v['power_w2']) / v['power_w2'] for v in unit_cost_cmp.values())
    max_rel_uc = max(max_rel_uc, max(abs(v['energy_module'] - v['energy_w2']) / v['energy_w2']
                                     for v in unit_cost_cmp.values()))
    check('B1_unit_costs_module_vs_W2', max_rel_uc <= 1e-12, {'max_rel_diff': max_rel_uc, 'per_year': unit_cost_cmp})

    def old_to_general(z7):
        z7 = old.canonical_z(z7)
        if not old.has_storage(z7):
            return lat.x0()
        y = old.years[z7[-1]]
        z = lat.x0()
        for n, (zp, ze) in zip(old.nodes, old.node_pairs(z7)):
            z = lat.with_pair(z, n, y, zp, ze)
        return z

    def general_to_old(z):
        active = lat.active_cohorts(z)
        years_used = {y for _n, y in active}
        nodes_used = [n for n, _y in active]
        if len(nodes_used) != len(set(nodes_used)) or len(years_used) > 1:
            return None
        y = years_used.pop() if years_used else old.years[0]
        z7 = []
        for n in old.nodes:
            z7 += list(lat.pair(z, n, y))
        return tuple(z7) + (old.years.index(y),)

    t0 = time.time()
    old_domain = old.domain()
    old_images, b1_fail = {}, []
    for z7 in old_domain:
        z = old_to_general(z7)
        active = lat.active_cohorts(z)
        ok = lat.admissible(z) and len({y for _n, y in active}) <= 1 and general_to_old(z) == old.canonical_z(z7)
        if not ok:
            b1_fail.append({'z7': list(z7), 'z': list(z), 'reasons': lat.reasons(z)})
        old_images[lat.key(z)] = z7
    check('B1_old_domain_maps_admissible_one_year', not b1_fail,
          {'n_old_domain': len(old_domain), 'n_fail': len(b1_fail), 'first_fail': b1_fail[:3]})
    # independent enumeration of the general admissible set with <= 1 nonzero year per node
    pairs = [(zp, ze) for zp in range(1, 12) for ze in range(1, 12) if zp <= ze <= 2 * zp]
    per_node = [None] + [(y, p) for y in inst.years for p in pairs]
    gen_le1, gen_common = [], []
    for combo in product(per_node, repeat=len(inst.nodes)):
        z = lat.x0()
        for n, entry in zip(inst.nodes, combo):
            if entry is not None:
                z = lat.with_pair(z, n, entry[0], *entry[1])
        if not lat.admissible(z):
            continue
        gen_le1.append(z)
        if len({entry[0] for entry in combo if entry is not None}) <= 1:
            gen_common.append(z)
    back_fail = []
    gen_common_keys = set()
    for z in gen_common:
        z7 = general_to_old(z)
        gen_common_keys.add(lat.key(z))
        if z7 is None or old.reasons(z7) or old.canonical_z(z7) != z7:
            back_fail.append({'z': list(z), 'z7': z7, 'old_reasons': old.reasons(z7) if z7 else None})
    old_domain_set = {tuple(old.canonical_z(z7)) for z7 in old_domain}
    mapped_back = {general_to_old(z) for z in gen_common}
    counts = {'old_domain': len(old_domain), 'general_le1_year_per_node_admissible': len(gen_le1),
              'general_common_year_admissible': len(gen_common),
              'general_le1_year_per_node_mixed_years (new, not in the old lattice)': len(gen_le1) - len(gen_common),
              'enumerated_per_node_options': len(per_node), 'enumeration_s': round(time.time() - t0, 1)}
    l1_ok = (not b1_fail and not back_fail and len(old_domain) == len(gen_common)
             and set(old_images) == gen_common_keys and mapped_back == old_domain_set)
    check('B1_round_trip_counts_equal (L1)', l1_ok, {**counts, 'n_back_fail': len(back_fail),
                                                     'first_back_fail': back_fail[:3],
                                                     'key_sets_equal': set(old_images) == gen_common_keys,
                                                     'z7_sets_equal': mapped_back == old_domain_set})
    # reasons agreement on 2,000 seeded inadmissible raw old points
    rng = random.Random(SEED)
    sample, n_draws = [], 0
    while len(sample) < N_REASON_SAMPLE:
        n_draws += 1
        if len(sample) % 2 == 0:   # wide box
            z7 = tuple(rng.randint(-1, 12) for _ in range(6)) + (rng.randrange(len(old.years)),)
        else:                      # structurally plausible nodes (enriches duration-, bound- and budget-only failures)
            z7 = []
            for _n in old.nodes:
                if rng.random() < 0.25:
                    z7 += [0, 0]
                else:
                    zp = rng.randint(1, 7)
                    z7 += [zp, rng.randint(max(0, zp - 1), 2 * zp + 1)]
            z7 = tuple(z7) + (rng.randrange(len(old.years)),)
        if not old.reasons(z7):
            continue
        sample.append(z7)
    agree, disagree, class_counts = 0, [], {}
    for z7 in sample:
        r_old = old.reasons(z7)
        z = lat.x0()   # raw map (negative entries kept): node pairs at the year index z7[-1]
        for n, (zp, ze) in zip(old.nodes, old.node_pairs(z7)):
            z = lat.with_pair(z, n, old.years[z7[-1]], zp, ze)
        r_new = lat.reasons(z)
        c_old, c_new = L.GeneralLattice.reason_classes(r_old), L.GeneralLattice.reason_classes(r_new)
        key = '+'.join(c_old)
        class_counts[key] = class_counts.get(key, 0) + 1
        if c_old == c_new:
            agree += 1
        else:
            disagree.append({'z7': list(z7), 'old': r_old, 'new': r_new})
    check('B1_reason_classes_agree_2000', agree == N_REASON_SAMPLE,
          {'agree': agree, 'n': N_REASON_SAMPLE, 'draws': n_draws, 'old_class_multisets': class_counts,
           'first_disagreements': disagree[:5]})
    write('back_compat.json', {'counts': counts, 'old_domain_z7': [list(z) for z in old_domain],
                               'old_domain_v2_keys': [lat.key(old_to_general(z)) for z in old_domain],
                               'back_fail': back_fail, 'b1_fail': b1_fail,
                               'reason_sample': {'seed': SEED, 'n': N_REASON_SAMPLE, 'draws': n_draws,
                                                 'old_class_multisets': class_counts, 'disagreements': disagree,
                                                 'z7': [list(z) for z in sample]},
                               'unit_costs_module_vs_w2': unit_cost_cmp})

    # ==================================================================================================================
    #  B2 oracle: the production master rows on SRP1
    # ==================================================================================================================
    side = os.path.join(OUT, 'side_writes')
    os.makedirs(os.path.join(side, 'Diagrams'))
    os.makedirs(os.path.join(side, 'Results', 'Logs'))
    diag_dir = os.path.join(SRP1_DIR, 'Diagrams')
    diag_before = {f: os.path.getmtime(os.path.join(diag_dir, f)) for f in os.listdir(diag_dir)} \
        if os.path.isdir(diag_dir) else None
    planning = SharedResourcesPlanning(SRP1_DIR, 'SRP1.json')
    planning.diagrams_dir = os.path.join(side, 'Diagrams')
    planning.results_dir = os.path.join(side, 'Results')
    planning.logs_dir = os.path.join(side, 'Results', 'Logs')
    stream = io.StringIO()
    t0 = time.time()
    with redirect_stdout(stream):
        planning.read_planning_problem()
    with open(os.path.join(OUT, 'planning_read_stdout.log'), 'w') as handle:
        handle.write(stream.getvalue())
    diag_after = {f: os.path.getmtime(os.path.join(diag_dir, f)) for f in os.listdir(diag_dir)} \
        if os.path.isdir(diag_dir) else None
    checksum = planning.scenario_metadata.get('combined_scenario_checksum')
    print(f'[oracle] planning read in {time.time() - t0:.1f} s; scenario checksum {checksum}')
    check('B2_planning_read_canonical_srp1', checksum == O.CANONICAL_CHECKSUM,
          {'checksum': checksum, 'canonical': O.CANONICAL_CHECKSUM})
    check('B2_no_side_write_into_data_SRP1_Diagrams', diag_before == diag_after,
          {'n_before': None if diag_before is None else len(diag_before),
           'n_after': None if diag_after is None else len(diag_after),
           'side_writes': sorted(os.listdir(os.path.join(side, 'Diagrams')))})
    esd = planning.shared_ess_data
    model = esd.build_master_problem()
    missing_c = [c for c in MASTER_COMPONENTS if not hasattr(model, c)]
    if missing_c:
        raise RuntimeError(f'master lacks components (capture path): {missing_c}')
    years_m = list(esd.years)
    nodes_m = [esd.shared_energy_storages[years_m[0]][e].bus for e in model.energy_storages]
    same_inst = {
        'years': years_m == list(inst.years), 'nodes': nodes_m == list(inst.nodes),
        'active_dn_nodes': list(esd.active_distribution_network_nodes) == list(inst.nodes),
        'block_widths': {int(y): int(w) for y, w in esd.years.items()} == inst.block_widths,
        't_cal': all(esd.shared_energy_storages[y][e].t_cal == inst.t_cal_years for y in years_m
                     for e in model.energy_storages),
        'max_capacity': esd.params.max_capacity == inst.max_capacity_mwh,
        'ratios': (esd.params.min_energy_to_power_ratio, esd.params.max_energy_to_power_ratio) == (inst.min_ratio, inst.max_ratio),
        'budget': esd.params.budget == inst.budget_eur, 'discount': esd.discount_factor == inst.discount_factor,
        'prob_market': list(esd.prob_market_scenarios) == list(inst.prob_market),
        'cost_power': all(esd.cost_investment['power'][m][y] == inst.cost_power[m][y]
                          for m in range(len(inst.prob_market)) for y in years_m),
        'cost_energy': all(esd.cost_investment['energy'][m][y] == inst.cost_energy[m][y]
                           for m in range(len(inst.prob_market)) for y in years_m)}
    check('B2_module_loader_reads_what_production_reads (bitwise)', all(same_inst.values()), same_inst)

    # row map: constraint -> (family, node, year)
    families = ('rated_s_capacity', 'rated_e_capacity', 'energy_storage_maximum_capacity',
                'energy_storage_power_to_energy_factor', 'energy_storage_investment', 'benders_cuts')
    row_map = []
    for fam in families:
        comp = getattr(model, fam)
        for idx in comp:
            con = comp[idx]
            if not con.active:
                continue
            node_year = None
            if fam != 'energy_storage_investment':
                v = [var for var in identify_variables(con.body)
                     if var.parent_component().name in ('es_e_rated', 'es_s_rated', 'es_e_investment')]
                if fam in ('rated_s_capacity', 'rated_e_capacity'):
                    v = [var for var in v if var.parent_component().name in ('es_e_rated', 'es_s_rated')]
                e_idx, y_idx = v[0].index()
                node_year = (nodes_m[e_idx], years_m[y_idx])
            row_map.append((fam, idx, con, node_year))
    other_active = [c.name for c in model.component_data_objects(pe.Constraint, active=True)
                    if c.parent_component().name not in families]
    if other_active:
        raise RuntimeError(f'unexpected active master rows: {other_active[:5]}')
    row_counts = {fam: sum(1 for r in row_map if r[0] == fam) for fam in families}
    print(f'[oracle] master rows {row_counts}')

    def violation(con):
        body = pe.value(con.body)
        lo = pe.value(con.lower) if con.has_lb() else None
        up = pe.value(con.upper) if con.has_ub() else None
        v = 0.0
        if lo is not None:
            v = max(v, lo - body)
        if up is not None:
            v = max(v, body - up)
        return v, body, lo, up

    def master_eval(z):
        x = {(n, y): {'s': L.GRANULE_P_MVA * lat.pair(z, n, y)[0], 'e': L.GRANULE_E_MWH * lat.pair(z, n, y)[1]}
             for n in inst.nodes for y in inst.years}
        candidate = O.vector_to_candidate(planning, x)
        SESD._load_candidate_solution_into_master_model(esd, model, candidate)
        fam_viol = {f: set() for f in families}
        max_eq_residual, budget_margin, worst = 0.0, None, 0.0
        for fam, _idx, con, node_year in row_map:
            v, body, _lo, up = violation(con)
            if fam in ('rated_s_capacity', 'rated_e_capacity'):
                max_eq_residual = max(max_eq_residual, v)
            if fam == 'energy_storage_investment':
                budget_margin = up - body
            worst = max(worst, v)
            if v > TOL_VERDICT:
                fam_viol[fam].add(node_year if node_year is not None else 'budget')
        prod_ok, prod_reasons = SRP._check_candidate_first_stage_feasibility(planning, candidate)
        return {'feasible': worst <= TOL_VERDICT, 'max_violation': worst,
                'violated': {f: sorted(map(str, s)) for f, s in fam_viol.items() if s},
                'fam_sets': fam_viol, 'max_rated_equality_residual': max_eq_residual,
                'budget_margin_eur': budget_margin, 'I_master': pe.value(model.investment_cost),
                'I_p56a_replica': O.investment_cost(planning, candidate),
                'production_first_stage_check': {'feasible': bool(prod_ok), 'reasons': prod_reasons}}

    # the 200 points
    def rand_pair(rng_, lo=1, hi=6, sloppy=False):
        zp = rng_.randint(lo, hi)
        if sloppy:
            return zp, rng_.randint(max(0, zp - 1), 2 * zp + 1)
        return zp, rng_.randint(zp, 2 * zp)

    rng = random.Random(SEED + 2)
    l4 = lat.with_pair(lat.with_pair(lat.x0(), inst.nodes[0], inst.years[0], 4, 6), inst.nodes[0], inst.years[1], 4, 6)
    points = [('fixed:x0', lat.x0()), ('fixed:L4_staged_3+3MWh_one_node', l4)]
    cats = [('sparse_random', 40), ('staged_two_cohort_one_node', 40), ('over_5MWh_cumulative', 40),
            ('near_budget', 39), ('multi_node_staged', 39)]
    for cat, count in cats:
        made = 0
        while made < count:
            z = lat.x0()
            if cat == 'sparse_random':
                for n, y in lat.cohorts:
                    if rng.random() < 0.25:
                        z = lat.with_pair(z, n, y, *rand_pair(rng, 1, 4, sloppy=True))
            elif cat == 'staged_two_cohort_one_node':
                n = rng.choice(inst.nodes)
                y1, y2 = rng.sample(inst.years, 2)
                z = lat.with_pair(lat.with_pair(z, n, y1, *rand_pair(rng, 1, 5)), n, y2, *rand_pair(rng, 1, 5))
            elif cat == 'over_5MWh_cumulative':
                n = rng.choice(inst.nodes)
                y1, y2 = rng.sample(inst.years, 2)
                p1, p2 = rand_pair(rng, 2, 6), rand_pair(rng, 2, 6)
                z = lat.with_pair(lat.with_pair(z, n, y1, *p1), n, y2, *p2)
                if max(lat.cumulative_energy(z).values()) <= inst.max_capacity_mwh:
                    continue
            elif cat == 'near_budget':
                for n in rng.sample(inst.nodes, rng.randint(1, 3)):
                    z = lat.with_pair(z, n, rng.choice(inst.years), *rand_pair(rng, 1, 4))
                if abs(lat.investment_cost(z) - inst.budget_eur) > 0.15 * inst.budget_eur:
                    continue
            else:
                for n in inst.nodes:
                    for y in rng.sample(inst.years, rng.randint(0, 2)):
                        z = lat.with_pair(z, n, y, *rand_pair(rng, 1, 3))
            points.append((cat, z))
            made += 1
    assert len(points) == N_ORACLE, len(points)

    fam_map = {'energy_storage_power_to_energy_factor': ('p0_iff_e0', 'energy_storage_power_to_energy_factor'),
               'energy_storage_maximum_capacity': ('energy_storage_maximum_capacity',),
               'energy_storage_investment': ('energy_storage_investment',)}
    oracle_rows, n_agree, max_rel_I, max_rel_I_p56a, min_abs_margin = [], 0, 0.0, 0.0, float('inf')
    disagreements = []
    t0 = time.time()
    for cat, z in points:
        mod_reasons = lat.reasons(z)
        mod_fam = lat.family_violations(z)
        mod_I = lat.investment_cost(z)
        m = master_eval(z)
        rel_I = abs(mod_I - m['I_master']) / max(abs(m['I_master']), 1.0)
        rel_I_p = abs(mod_I - m['I_p56a_replica']) / max(abs(m['I_p56a_replica']), 1.0)
        max_rel_I, max_rel_I_p56a = max(max_rel_I, rel_I), max(max_rel_I_p56a, rel_I_p)
        min_abs_margin = min(min_abs_margin, abs(m['budget_margin_eur']))
        fam_agree = {}
        for mfam, mod_keys in fam_map.items():
            mod_set = set()
            for k in mod_keys:
                mod_set |= {('budget' if v == 'budget' else tuple(v)) for v in mod_fam[k]}
            fam_agree[mfam] = mod_set == m['fam_sets'][mfam]
        verdict_agree = (not mod_reasons) == m['feasible']
        agree = verdict_agree and all(fam_agree.values()) and m['max_rated_equality_residual'] == 0.0 \
            and m['production_first_stage_check']['feasible'] == m['feasible']
        n_agree += agree
        row = {'category': cat, 'z': list(z), 'label': lat.label(z), 'key': lat.key(z),
               'module_admissible': not mod_reasons, 'module_reasons': mod_reasons,
               'module_families': {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in mod_fam.items()},
               'master_feasible': m['feasible'], 'master_max_violation': m['max_violation'],
               'master_violated_rows': m['violated'], 'master_rated_equality_residual': m['max_rated_equality_residual'],
               'budget_margin_eur': m['budget_margin_eur'], 'I_module_eur': mod_I, 'I_master_eur': m['I_master'],
               'I_rel_diff': rel_I, 'I_p56a_replica_eur': m['I_p56a_replica'],
               'production_first_stage_check': m['production_first_stage_check'],
               'family_agree': fam_agree, 'verdict_agree': verdict_agree, 'agree': agree}
        oracle_rows.append(row)
        if not agree:
            disagreements.append(row)
    print(f'[oracle] {len(points)} points in {time.time() - t0:.1f} s')
    by_cat = {}
    for row in oracle_rows:
        c = by_cat.setdefault(row['category'], {'n': 0, 'module_admissible': 0, 'master_feasible': 0, 'agree': 0})
        c['n'] += 1
        c['module_admissible'] += row['module_admissible']
        c['master_feasible'] += row['master_feasible']
        c['agree'] += row['agree']
    check('B2_oracle_verdict_and_families_agree_200 (L2 part 1)', n_agree == N_ORACLE,
          {'agree': n_agree, 'n': N_ORACLE, 'by_category': by_cat, 'first_disagreements': disagreements[:3]})
    check('B2_I_equals_master_investment_cost_1e-9 (gate)', max_rel_I <= I_REL_TOL_GATE,
          {'max_rel_diff': max_rel_I, 'max_rel_diff_vs_p56a_replica': max_rel_I_p56a,
           'min_abs_budget_margin_eur': min_abs_margin})
    check('B2_I_equals_master_investment_cost_1e-12 (L2 part 2)', max_rel_I <= 1e-12, {'max_rel_diff': max_rel_I})
    l4_row = oracle_rows[1]
    l4_ok = (not l4_row['module_admissible']
             and any(r.startswith('bound: node') and 'cumulative' in r for r in l4_row['module_reasons'])
             and 'energy_storage_maximum_capacity' in l4_row['master_violated_rows'])
    check('B2_L4_staged_3+3MWh_rejected_by_cumulative_and_master_row (L4)', l4_ok,
          {'label': l4_row['label'], 'module_reasons': l4_row['module_reasons'],
           'module_families': l4_row['module_families'], 'master_violated_rows': l4_row['master_violated_rows'],
           'I_module_eur': l4_row['I_module_eur']})
    write('oracle_points.json', {'seed': SEED + 2, 'n': len(oracle_rows), 'tolerance_verdict': TOL_VERDICT,
                                 'master_row_counts': row_counts, 'by_category': by_cat, 'rows': oracle_rows})

    # ==================================================================================================================
    #  B3 neighbourhoods (Addendum 88 R3)
    # ==================================================================================================================
    def up_moves(lt, z):
        out = []
        for n, y in lt.active_cohorts(z):
            zp, ze = lt.pair(z, n, y)
            out += [lt.with_pair(z, n, y, zp + 1, ze), lt.with_pair(z, n, y, zp, ze + 1)]
        for n, y in lt.cohorts:
            if lt.pair(z, n, y) == (0, 0):
                out += [lt.with_pair(z, n, y, *p) for p in L.ACTIVATION_PAIRS]
        return [w for w in out if lt.admissible(w)]

    def incumbents(lt, rng_):
        inc = {'x0': lt.x0()}
        while True:
            n, y = rng_.choice(lt.cohorts)
            z = lt.with_pair(lt.x0(), n, y, *rand_pair(rng_, 1, 3))
            if lt.admissible(z):
                inc['single_cohort'] = z
                break
        while True:
            n1, n2 = rng_.sample(lt.nodes, 2)
            z = lt.with_pair(lt.with_pair(lt.x0(), n1, rng_.choice(lt.years), *rand_pair(rng_, 1, 2)),
                             n2, rng_.choice(lt.years), *rand_pair(rng_, 1, 2))
            if lt.admissible(z):
                inc['two_node'] = z
                break
        while True:
            n = rng_.choice(lt.nodes)
            y1, y2 = sorted(rng_.sample(lt.years, 2))
            z = lt.with_pair(lt.with_pair(lt.x0(), n, y1, *rand_pair(rng_, 1, 2)), n, y2, *rand_pair(rng_, 1, 2))
            if lt.admissible(z):
                inc['staged'] = z
                break
        z = lt.x0()
        while True:
            moves = up_moves(lt, z)
            if not moves:
                break
            z = rng_.choice(moves)
        inc['at_budget'] = z
        return inc

    def verify_neighbourhood(lt, name, z_inc, nb, with_master):
        out = {'incumbent': name, 'label': nb['incumbent_label'], 'A': nb['A'], 'n_raw': nb['n_raw'],
               'n_raw_by_part': nb['n_raw_by_part'], 'size_bound': nb['size_bound'],
               'size_bound_30_plus_5A': 30 + 5 * nb['A'], 'n_admissible': nb['n_admissible'],
               'n_dropped': nb['n_dropped'], 'dropped_by_reason_class': nb['dropped_by_reason_class'],
               'duplicates': nb['duplicates'], 'I_incumbent_eur': lt.investment_cost(z_inc),
               'budget_slack_eur': lt.inst.budget_eur - lt.investment_cost(z_inc)}
        ok_bound = nb['n_raw'] <= nb['size_bound']
        if len(lt.cohorts) == 15:
            ok_bound = ok_bound and nb['size_bound'] == 30 + 5 * nb['A']
        listed_ok = all(lt.admissible(tuple(p['z'])) and tuple(p['z']) != z_inc for p in nb['points'])
        dropped_ok = all(lt.reasons(tuple(p['z'])) for p in nb['dropped'])
        # shifts merge correctly
        shift_fail = []
        for p in nb['points'] + nb['dropped']:
            if not p['move'].startswith('shift:'):
                continue
            _s, n_s, ys = p['move'].split(':')[:3]
            n = int(n_s[1:])
            y_src, y_tgt = (int(v[1:]) for v in ys.split('->'))
            zn = tuple(p['z'])
            src, tgt = lt.pair(z_inc, n, y_src), lt.pair(z_inc, n, y_tgt)
            exp = lt.with_pair(lt.with_pair(z_inc, n, y_src, 0, 0), n, y_tgt, src[0] + tgt[0], src[1] + tgt[1])
            others_same = all(lt.pair(zn, m, y) == lt.pair(z_inc, m, y) for m, y in lt.cohorts
                              if (m, y) not in ((n, y_src), (n, y_tgt)))
            tot_same = (sum(lt.pair(zn, n, y)[0] for y in lt.years) == sum(lt.pair(z_inc, n, y)[0] for y in lt.years)
                        and sum(lt.pair(zn, n, y)[1] for y in lt.years) == sum(lt.pair(z_inc, n, y)[1] for y in lt.years))
            if not (zn == exp and others_same and tot_same and abs(lt.years.index(y_src) - lt.years.index(y_tgt)) == 1):
                shift_fail.append(p['move'])
        out.update({'size_bound_holds': ok_bound, 'listed_all_admissible': listed_ok,
                    'dropped_all_inadmissible': dropped_ok, 'shift_merge_failures': shift_fail,
                    'n_shifts': sum(1 for p in nb['points'] + nb['dropped'] if p['move'].startswith('shift:')),
                    'n_merged_shifts': sum(1 for p in nb['points'] + nb['dropped'] if p['move'].endswith(':merged'))})
        # determinism
        out['deterministic'] = json.dumps(lt.neighbourhood(z_inc), sort_keys=True) == json.dumps(nb, sort_keys=True)
        if name == 'x0':
            expected = {lt.key(lt.with_pair(lt.x0(), n, y, *pp)) for n, y in lt.cohorts for pp in L.ACTIVATION_PAIRS
                        if lt.investment_cost(lt.with_pair(lt.x0(), n, y, *pp)) <= lt.inst.budget_eur}
            got = {p['key'] for p in nb['points']}
            out['x0_expected_activation_count'] = len(expected)
            out['x0_equals_budget_fitting_activations'] = (got == expected and nb['n_raw_by_part']['a'] == 0
                                                           and nb['n_raw_by_part']['c'] == 0
                                                           and all(p['move'].startswith('activate:') for p in nb['points']))
        if with_master:
            mm = []
            for p in nb['points'] + nb['dropped']:
                m = master_eval(tuple(p['z']))
                if m['feasible'] != p['admissible']:
                    mm.append({'move': p['move'], 'module': p['admissible'], 'master': m['feasible'],
                               'violated': m['violated']})
            out['master_rows_agree_on_all_listed_and_dropped'] = not mm
            out['master_disagreements'] = mm
        out['pass'] = bool(ok_bound and listed_ok and dropped_ok and not shift_fail and out['deterministic']
                           and out.get('x0_equals_budget_fitting_activations', True)
                           and out.get('master_rows_agree_on_all_listed_and_dropped', True))
        return out

    nb_out = {}
    for inst_name, lt, with_master, seed in (('five_block', lat5, False, SEED + 3), ('srp1', lat, True, SEED + 4)):
        rng = random.Random(seed)
        incs = incumbents(lt, rng)
        nb_out[inst_name] = {'seed': seed, 'incumbents': {}}
        for name, z_inc in incs.items():
            nb = lt.neighbourhood(z_inc)
            ver = verify_neighbourhood(lt, name, z_inc, nb, with_master)
            nb_out[inst_name]['incumbents'][name] = {'verification': ver, 'neighbourhood': nb}
            check(f'B3_{inst_name}_{name}', ver['pass'],
                  {k: ver[k] for k in ('label', 'A', 'n_raw', 'size_bound', 'n_admissible', 'dropped_by_reason_class',
                                       'budget_slack_eur') if k in ver}
                  | ({'x0_expected': ver.get('x0_expected_activation_count')} if name == 'x0' else {})
                  | ({'master_agree': ver.get('master_rows_agree_on_all_listed_and_dropped')} if with_master else {}))
    l3 = nb_out['five_block']['incumbents']['x0']['neighbourhood']['n_admissible']
    check('B3_L3_five_block_x0_neighbourhood_is_30', l3 == 30,
          {'n_admissible': l3, 'expected_by_budget': nb_out['five_block']['incumbents']['x0']['verification']
           ['x0_expected_activation_count']})
    # resolution by bound: pure-function behaviour on synthetic numbers (no LB exists yet)
    pts = nb_out['five_block']['incumbents']['staged']['neighbourhood']['points']
    ub = 100.0
    f_lb = {p['key']: ub + (5.0 if i % 3 == 0 else (1.0 if i % 3 == 1 else -1.0)) for i, p in enumerate(pts)}
    rb = L.resolution_by_bound(pts, f_lb, 2.0, ub)
    expected_k = sum(1 for i in range(len(pts)) if i % 3 == 0)
    rb2 = L.resolution_by_bound(pts, lambda k: None, 0.0, ub)
    rb3 = L.resolution_by_bound(pts, f_lb, {p['key']: 1.0 for p in pts}, ub)
    expected_k3 = sum(1 for i in range(len(pts)) if i % 3 in (0, 1))
    check('B3_resolution_by_bound_pure', rb['k'] == expected_k and rb2['k'] == 0 and rb3['k'] == expected_k3
          and rb['statement'] == f'resolved by the lower bound: {expected_k} of {len(pts)}',
          {'k': rb['k'], 'expected': expected_k, 'k_no_lb': rb2['k'], 'k_margin_map': rb3['k'],
           'expected_margin_map': expected_k3, 'statement': rb['statement'],
           'rule': 'resolved iff F_LB - margin >= UB (synthetic F_LB = UB + {5, 1, -1}; margin 2 then 1)'})
    write('neighbourhoods.json', nb_out)

    # ==================================================================================================================
    #  B4 canonical form and keys
    # ==================================================================================================================
    all_z = {}
    for z7 in old_domain:
        all_z[old_to_general(z7)] = None
    for _c, z in points:
        all_z[z] = None
    for inst_name, lt in (('five_block', lat5), ('srp1', lat)):
        for rec in nb_out[inst_name]['incumbents'].values():
            for p in rec['neighbourhood']['points']:
                all_z[(inst_name, tuple(p['z']))] = None
    n_checked, failures, keys_seen = 0, [], {}
    for item in all_z:
        lt, z = ({'five_block': lat5, 'srp1': lat}[item[0]], item[1]) if isinstance(item[0], str) else (lat, item)
        c = lt.canonical_of_z(z)
        text = json.dumps(c)
        ok = (lt.canonicalize(c) == c and lt.canonicalize(lt.canonicalize(c)) == c
              and lt.canonicalize(json.loads(text)) == c and lt.z_of_canonical(c) == z
              and lt.z_of_canonical(json.loads(text)) == z and L.candidate_key(c) == H.candidate_key(c)
              and list(c['nodes']) == sorted(c['nodes'], key=int)
              and all(list(v) == sorted(v, key=int) and len(v) == len(lt.years) for v in c['nodes'].values())
              and all(isinstance(a, float) for v in c['nodes'].values() for pr in v.values() for a in pr))
        k = (id(lt), L.candidate_key(c))
        if k in keys_seen and keys_seen[k] != z:
            ok = False
        keys_seen[k] = z
        n_checked += 1
        if not ok:
            failures.append({'z': list(z)})
    check('B4_canonical_idempotent_keys_round_trip', not failures, {'n': n_checked, 'failures': failures[:3]})
    old_x0_keys = {H.candidate_key(H.canonical_candidate({n: (0.0, 0.0) for n in inst.nodes}, inst.nodes, y))
                   for y in inst.years}
    v2_x0 = {L.candidate_key(lat.canonicalize({str(n): {str(y): [0.0, 0.0]} for n in inst.nodes}))
             for y in inst.years}
    check('B4_x0_alias_single_v2_key', len(old_x0_keys) == len(inst.years) and len(v2_x0) == 1
          and v2_x0 == {lat.key(lat.x0())} and lat.label(lat.x0()) == L.X0_LABEL,
          {'old_schema_x0_keys': len(old_x0_keys), 'v2_x0_keys': sorted(v2_x0)})
    refusals = {}
    for name, bad in (('off_lattice', {'5': {'2025': [0.3, 1.0]}}), ('unknown_node', {'6': {'2025': [0.25, 0.5]}}),
                      ('unknown_year', {'5': {'2026': [0.25, 0.5]}}), ('negative', {'5': {'2025': [-0.25, 0.5]}})):
        try:
            lat.canonicalize(bad)
            refusals[name] = False
        except ValueError:
            refusals[name] = True
    check('B4_canonicalize_refuses_bad_input', all(refusals.values()), refusals)

    # ---- informational: can the cumulative cap bind while the budget holds? (no gate; recorded for the report) ------
    cap_info = {}
    for inst_name, lt in (('srp1', lat), ('five_block', lat5)):
        zmax = int(round(lt.inst.max_capacity_mwh / L.GRANULE_E_MWH)) + 1
        cohort_pairs = [(zp, ze) for zp in range(1, zmax + 1) for ze in range(1, zmax + 1) if zp <= ze <= 2 * zp]
        best, n_over = None, 0
        n0 = lt.nodes[0]
        for k in (1, 2):
            for ys in ([(y,) for y in lt.years] if k == 1 else
                       [(a, b) for i, a in enumerate(lt.years) for b in lt.years[i + 1:]]):
                for prs in product(cohort_pairs, repeat=k):
                    z = lt.x0()
                    for y, pr in zip(ys, prs):
                        z = lt.with_pair(z, n0, y, *pr)
                    if not lt.family_violations(z)['energy_storage_maximum_capacity']:
                        continue
                    n_over += 1
                    i_x = lt.investment_cost(z)
                    if best is None or i_x < best[0]:
                        best = (i_x, lt.label(z))
        cap_info[inst_name] = {'n_over_cap_plans_enumerated (one node, <= 2 cohorts, zE <= %d)' % zmax: n_over,
                               'min_I_over_cap_eur': best[0], 'argmin': best[1], 'budget_eur': lt.inst.budget_eur,
                               'cap_can_bind_with_budget_slack': best[0] <= lt.inst.budget_eur}
        print(f'[info] {inst_name}: cheapest over-cap plan {best[1]} I = {best[0]:.2f} vs B {lt.inst.budget_eur:g}')
    results['informational_cap_vs_budget'] = cap_info

    # ---- guards -------------------------------------------------------------------------------------------------------
    guard_failures = GUARD.verify(0) + PB.PARENT_GUARD.verify(0)
    check('G_zero_solves_guard_0_0', not guard_failures,
          {'gate_guard': GUARD.counts, 'pb_parent_guard': PB.PARENT_GUARD.counts, 'failures': guard_failures})

    results.update({'checks': checks, 'all_pass': all(c['pass'] for c in checks),
                    'predictions_vs_outcomes': {
                        'L1': {'prediction': PREDICTIONS['L1'], 'outcome': counts,
                               'held': next(c['pass'] for c in checks if c['id'].startswith('B1_round_trip'))},
                        'L2': {'prediction': PREDICTIONS['L2'],
                               'outcome': {'agree': n_agree, 'n': N_ORACLE, 'max_rel_I': max_rel_I},
                               'held': n_agree == N_ORACLE and max_rel_I <= 1e-12},
                        'L3': {'prediction': PREDICTIONS['L3'], 'outcome': {'n_x0_neighbourhood': l3}, 'held': l3 == 30},
                        'L4': {'prediction': PREDICTIONS['L4'], 'outcome': {'module_reasons': l4_row['module_reasons'],
                                                                           'master_violated': l4_row['master_violated_rows']},
                               'held': l4_ok}},
                    'guard_counts': {'gate': GUARD.counts, 'pb_parent': PB.PARENT_GUARD.counts},
                    'wall_s': round(time.time() - t_start, 1), 'finished_utc': _utc()})
    write('g_cc4_results.json', results)
    n_fail = sum(1 for c in checks if not c['pass'])
    print(f"[done] {len(checks)} checks, {n_fail} failed; all_pass {results['all_pass']}; wall {results['wall_s']} s")
    return 0 if results['all_pass'] else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', action='store_true')
    ap.add_argument('--trial-out', default=None, help='debugging only: write to this directory outside the repository')
    args = ap.parse_args()
    sys.exit(manifest() if args.manifest else run(args.trial_out))
