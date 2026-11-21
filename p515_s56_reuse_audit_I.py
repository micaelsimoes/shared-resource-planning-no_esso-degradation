"""P5.15 S56 reuse audit I -- staged installations and terminal value on the SRP1 grid (ZERO SOLVES, read-only).

Question (Planner, 2026-11-21): which parts of the P5.16 work (general node-year lattice C5, salvage in the ESSO
objective C1, 'mid' SoH point C3, five-block instance C4) can be reused, decoupled from the 3-year blocks and the
mid-block SoH point, to evaluate staged plans with terminal value in the objective on the 2025/2030/2035 5-year grid.

Parts (every number is computed here from production code or the lattice module, nothing typed):
  A  lattice count -- the full staged (node, year) domain on SRP1 (several cohorts per node allowed), by import of
     `p516_lattice` (GeneralLattice; admissibility = its `reasons`), counted per node and combined under the budget
     with an exact recheck of every triple within 1e-6 EUR of B; cross-checks against C5 L1 (6,694 single-cohort-
     per-node points incl. x0) and the 1,705-point common-year domain; a seeded sample checked point by point with
     `GeneralLattice.reasons`; the cheapest plan over the cumulative cap (C5 F-1, SRP1 value 1,108,893.20 EUR).
  B  ESSO structure for staged plans at node 5 on SRP1 ('end', salvage reported only, as committed), built by
     production (`_build_subproblem`, `_update_model_with_candidate_solution`), no solve:
       B1 cohort-0 rows (throughput, D, SoH chain, floor, available energy) identical as built with and without later
          cohorts (string of every row + active flag);
       B2 every variable in a cohort-registered row belongs to that cohort;
       B3 a later cohort's SoH chain starts at 1 in its installation block (no reference to an earlier block);
       B4 the coupling inventory: every active row that references two or more cohorts, by family;
       B5 H3 pro-rata rows: N_active(y) - 1 active per (y, d, p), shares = E_i / sum E;
       B6 salvage with `in_objective` switched on IN MEMORY (no file edit) on the SRP1 grid with 'end': the linear
          credit's coefficients against the hand formula, the reporting expression's identity at a synthetic point,
          rlf per cohort, disc_T, no converter variable;
       B7 kappa on the SRP1 grid by production's `_resolve_esso_al_scale` and the G-CC-3 (ii) slack-safety number
          (salvage marginal per MW-period at calendar-only terminal SoH vs 1e-3 * PENALTY_ESSO_SLACK) and dS/dTP.
  C  static source facts (file:line located by pattern at run time; sha256 of each file).

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0.
Writes ONLY into data/SRP1/Results/P515S56/reuse_audit_I/ (planning-read side writes redirected under
side_writes/). The launch command creates the directory to hold the two launch logs; the script refuses if it holds
anything else.

Launch (attached, both streams captured):
  mkdir data/SRP1/Results/P515S56/reuse_audit_I && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_reuse_audit_I.py \
    > data/SRP1/Results/P515S56/reuse_audit_I/launch_stdout.log \
    2> data/SRP1/Results/P515S56/reuse_audit_I/launch_stderr.log
then  ... p515_s56_reuse_audit_I.py --manifest
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 reuse audit I (zero solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

SCHEMA = 'p515_s56_reuse_audit_I_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'reuse_audit_I')
OUT = os.path.join(REPO, OUT_REL)
LAUNCH_LOGS = ('launch_stdout.log', 'launch_stderr.log')
SRP1_DIR = os.path.join(REPO, 'data', 'SRP1')
SRP1_CASE = os.path.join(SRP1_DIR, 'SRP1.json')
SEED = 20261121
N_SAMPLE = 20000
BOUNDARY_EPS_EUR = 1e-6
Z_MAX = 11  # one granule beyond every bound (E <= 5 MWh -> zE <= 10; P <= 2.5 MVA -> zP <= 10)
NODE = 5
PLANS = {  # node-5 staged plans (MVA, MWh) by installation year; all admissible (cumulative 2.5 MWh, I < B)
    'A_2025': {2025: (0.5, 1.0), 2030: (0.0, 0.0), 2035: (0.0, 0.0)},
    'B_2025_2030': {2025: (0.5, 1.0), 2030: (0.25, 0.5), 2035: (0.0, 0.0)},
    'C_2025_2030_2035': {2025: (0.5, 1.0), 2030: (0.25, 0.5), 2035: (0.25, 1.0)},
}
PER_COHORT_VARS = ('es_s_rated_per_unit', 'es_e_rated_per_unit', 'es_s_available_per_unit', 'es_e_available_per_unit',
                   'es_pch_per_unit', 'es_pdch_per_unit', 'es_avg_ch_dch_per_unit', 'es_soh_per_unit_cumul',
                   'es_D_per_unit')
COHORT_FAMILIES = ('energy_storage_charging_discharging', 'energy_storage_capacity_degradation')
HASHED_SOURCES = ['p515_s56_reuse_audit_I.py', 'p516_lattice.py', 'p515_s44_campaign_harness.py',
                  'p515_g_g1_g4_admm_gates.py', 'p515_s47_phase_b_record.py', 'shared_energy_storage_data.py',
                  'shared_energy_storage_parameters.py', 'shared_resources_planning.py', 'definitions.py',
                  'p513_solve_profile_guard.py', 'gate_result_io.py',
                  os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_5x3y_1x1.json'),
                  os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
                  os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
                  os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_5x3y_ESS_Params.json')]
# Static facts: (id, file, regex). Line numbers are located at run time, never typed.
SOURCE_FACTS = [
    ('harness_canonical_candidate_single_year', 'p515_s44_campaign_harness.py',
     r"return \{'investment_year': int\(investment_year\),"),
    ('harness_investment_year_default', 'p515_s44_campaign_harness.py', r'^INVESTMENT_YEAR = 2025'),
    ('harness_ess_params_file_pinned', 'p515_s44_campaign_harness.py', r'^ESS_PARAMS_FILE_REL = '),
    ('harness_refuses_one_scenario_derived_instance', 'p515_s44_campaign_harness.py',
     r"'planning_has_more_than_one_scenario_combination'"),
    ('harness_model_variant_keys', 'p515_s44_campaign_harness.py', r'^MODEL_VARIANT_KEYS = '),
    ('harness_record_certified_cost_is_gross', 'p515_s44_campaign_harness.py',
     r"'certified_cost': report.get\('gross_operational_cost'\) if certified else None"),
    ('harness_record_terminal_net_recourse', 'p515_s44_campaign_harness.py', r"'terminal_net_operational_recourse'"),
    ('gates_construct_arm_planning_no_staging', 'p515_g_g1_g4_admm_gates.py',
     r'Multi-cohort \(staging\) candidates are NOT supported here'),
    ('gates_single_year_write', 'p515_g_g1_g4_admm_gates.py',
     r"candidate\['investment'\]\[node_id\]\[invest_year\]\['s'\] = s_val"),
    ('pb_lattice_n_vars_7', 'p515_s47_phase_b_record.py', r'^N_VARS = '),
    ('pb_primes_10', 'p515_s47_phase_b_record.py', r'^PRIMES = '),
    ('pb_run_mads_single_year_canonical', 'p515_s47_phase_b_record.py',
     r"'canonical': \{'investment_year': lattice.year_of\(z\),"),
    ('pb_objective_convention_F_I_plus_gross', 'p515_s47_phase_b_record.py', r'^OBJECTIVE_CONVENTION = '),
    ('lattice_neighbourhood', 'p516_lattice.py', r'def neighbourhood\(self, z_inc\):'),
    ('lattice_resolution_by_bound', 'p516_lattice.py', r'^def resolution_by_bound'),
    ('definitions_penalty_esso_slack', 'definitions.py', r'^PENALTY_ESSO_SLACK = '),
    ('params_penalty_key_absent', 'shared_energy_storage_parameters.py', r'(?i)penalty'),
    ('params_in_objective_default', 'shared_energy_storage_parameters.py', r'self.in_objective = False'),
    ('params_soh_point_key', 'shared_energy_storage_parameters.py', r"params_data.get\('available_energy_soh_point'\)"),
    ('esso_soh_chain_prev', 'shared_energy_storage_data.py', r'prev_soh = model.es_soh_per_unit_cumul\[y_inv, y - 1\]'),
    ('esso_soh_floor', 'shared_energy_storage_data.py',
     r'model.es_soh_per_unit_cumul\[y_inv, y\] >= shared_energy_storage.soh_min\)'),
    ('esso_D_row_investment_block_width', 'shared_energy_storage_data.py',
     r'num_years = shared_ess_data.years\[repr_years\[y_inv\]\]$'),
    ('esso_h3_share_from_investment', 'shared_energy_storage_data.py',
     r'share = pe.value\(model.es_e_investment_fixed\[y_inv\]\) / total_e_rated'),
    ('esso_available_capacity_sum', 'shared_energy_storage_data.py',
     r'e_available \+= pe.value\(model.es_e_available_per_unit\[y_inv, year_idx\]\)'),
    ('esso_objective_switch', 'shared_energy_storage_data.py',
     r"if getattr\(shared_ess_data.params.salvage_value, 'in_objective', False\):"),
    ('esso_terminal_discount', 'shared_energy_storage_data.py', r'^def _get_terminal_discount_factor'),
    ('esso_remaining_life', 'shared_energy_storage_data.py', r'^def _get_remaining_calendar_life'),
    ('esso_expected_energy_cost_undiscounted', 'shared_energy_storage_data.py',
     r'^def _get_expected_energy_investment_cost'),
    ('esso_salvage_sensitivity_power_zero', 'shared_energy_storage_data.py', r"sensitivities\['s'\]\[year_inv\]\[node_id\] = 0.00"),
    ('master_budget_row_investment_only', 'shared_energy_storage_data.py',
     r'model.energy_storage_investment.add\(investment_cost_total <= shared_ess_data.params.budget\)'),
    ('srp_first_stage_budget_check', 'shared_resources_planning.py', r'if expected_investment_cost > params.budget \+ tolerance:'),
    ('srp_net_recourse_gross_minus_salvage', 'shared_resources_planning.py',
     r'net_operational_recourse = gross_operational_cost - terminal_salvage_value'),
    ('srp_gross_from_networks_only', 'shared_resources_planning.py',
     r'gross_operational_cost_including_settlement = planning_problem.transmission_network.get_primal_value'),
    ('srp_kappa_set', 'shared_resources_planning.py', r'models\[node_id\].salvage_objective_kappa.set_value\('),
    ('srp_esso_admm_objective_copies_objective', 'shared_resources_planning.py',
     r'obj = copy\(models\[node_id\].objective.expr\)'),
    ('srp_salvage_sensitivities_added', 'shared_resources_planning.py',
     r'salvage_sensitivities\[capacity_type\]\[year\]\[node_id\]'),
    ('srp_block_weight', 'shared_resources_planning.py', r'^def _get_admm_block_weight'),
    ('srp_rebuild_total_capacities', 'shared_resources_planning.py', r'^def _rebuild_candidate_total_capacities'),
    ('srp_esso_data_gets_investment_not_total', 'shared_resources_planning.py',
     r"planning_problem.shared_ess_data.update_data_with_candidate_solution\($"),
]


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


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
    files['p515_s56_reuse_audit_I.py'] = sha256_file(os.path.join(REPO, 'p515_s56_reuse_audit_I.py'))
    with open(target, 'w') as handle:
        handle.write(GRIO.dumps({'schema': SCHEMA + '_manifest', 'utc': _utc(), 'files': files},
                                indent=1, sort_keys=True))
    print(f'manifest: {len(files)} files -> {target}')
    return 0


def _preflight():
    if not os.path.isdir(OUT):
        raise SystemExit(f'refusing: {OUT_REL} must be created by the launch command (mkdir) to hold the launch logs')
    extra = sorted(set(os.listdir(OUT)) - set(LAUNCH_LOGS))
    if extra:
        raise SystemExit(f'refusing: {OUT_REL} already holds {extra}')
    mine = os.path.basename(__file__)
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout.splitlines()
    others = []
    for line in ps:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] == str(os.getpid()):
            continue
        # a python process whose script argument is this file (shells that merely mention it are not copies)
        if 'python' in os.path.basename(parts[1]) and any(os.path.basename(p) == mine for p in parts[2:]):
            others.append(line)
    if others:
        raise SystemExit(f'refusing: another copy is running: {others}')


# ======================================================================================================================
#  Part A -- staged-domain count on SRP1
# ======================================================================================================================
def part_a(check):
    import numpy as np
    import p516_lattice as L
    inst = L.load_instance(SRP1_CASE)
    lat = L.GeneralLattice(inst)
    years, nodes, budget = inst.years, inst.nodes, inst.budget_eur
    t0 = time.time()

    # per-cohort options: the lattice's own per-cohort verdict (sign, P = 0 <=> E = 0, duration)
    cohort_opts = [(zp, ze) for zp in range(Z_MAX + 1) for ze in range(Z_MAX + 1)
                   if not lat._cohort_reasons(nodes[0], years[0], zp, ze)[0]]

    def node_cost(n, pairs):
        total = 0.0
        for y, (zp, ze) in zip(years, pairs):
            p, e = L.GRANULE_P_MVA * zp, L.GRANULE_E_MWH * ze
            for c_s, c_e in lat._coef[y]:
                total += c_s * p
                total += c_e * e
        return total

    def cumulative_ok(pairs):
        cum = {y: 0.0 for y in years}
        for y_inv, (_zp, ze) in zip(years, pairs):
            for y in lat.alive_blocks[y_inv]:
                cum[y] += L.GRANULE_E_MWH * ze
        return all(v <= inst.max_capacity_mwh for v in cum.values())

    per_node = {}
    over_cap_min = None
    for n in nodes:
        opts = []
        for pairs in product(cohort_opts, repeat=len(years)):
            cost = node_cost(n, pairs)
            if not cumulative_ok(pairs):
                if over_cap_min is None or cost < over_cap_min[0]:
                    over_cap_min = (cost, n, pairs)
                continue
            if cost <= budget + 1.0:
                opts.append((pairs, cost, sum(1 for p in pairs if p != (0, 0))))
        per_node[n] = opts
    sizes = {str(n): len(v) for n, v in per_node.items()}

    # combine three nodes under the budget, grouped by the number of active cohorts per node
    groups = {n: {} for n in nodes}
    for n in nodes:
        for pairs, cost, k in per_node[n]:
            groups[n].setdefault(k, []).append(cost)
        for k in groups[n]:
            groups[n][k] = np.sort(np.array(groups[n][k]))
    ks = sorted({k for n in nodes for k in groups[n]})
    count_by_k = {}
    ambiguous = 0
    n0, n1, n2 = nodes
    for k0, k1, k2 in product(ks, repeat=3):
        if k0 not in groups[n0] or k1 not in groups[n1] or k2 not in groups[n2]:
            continue
        a, b, c = groups[n0][k0], groups[n1][k1], groups[n2][k2]
        ab = (a[:, None] + b[None, :]).ravel()
        lo = np.searchsorted(c, budget - BOUNDARY_EPS_EUR - ab, side='right').sum()
        hi = np.searchsorted(c, budget + BOUNDARY_EPS_EUR - ab, side='right').sum()
        ambiguous += int(hi - lo)
        if lo:
            count_by_k[(k0, k1, k2)] = int(lo)
    # exact recheck of every ambiguous triple with GeneralLattice.reasons
    exact_extra = 0
    if ambiguous:
        import bisect
        third = sorted((c, pairs) for pairs, c, _ in per_node[n2])
        third_costs = [c for c, _ in third]
        for (p0, c0, _), (p1, c1, _) in product(per_node[n0], per_node[n1]):
            lo_i = bisect.bisect_right(third_costs, budget - BOUNDARY_EPS_EUR - c0 - c1)
            hi_i = bisect.bisect_right(third_costs, budget + BOUNDARY_EPS_EUR - c0 - c1)
            for c2, p2 in third[lo_i:hi_i]:
                z = tuple(v for pairs in (p0, p1, p2) for pair in pairs for v in pair)
                if lat.admissible(z):
                    exact_extra += 1
    total = sum(count_by_k.values()) + exact_extra
    by_total_cohorts, by_staged_nodes, by_max_k = {}, {}, {}
    for (k0, k1, k2), v in count_by_k.items():
        t = k0 + k1 + k2
        s = sum(1 for k in (k0, k1, k2) if k >= 2)
        by_total_cohorts[str(t)] = by_total_cohorts.get(str(t), 0) + v
        by_staged_nodes[str(s)] = by_staged_nodes.get(str(s), 0) + v
        by_max_k[str(max(k0, k1, k2))] = by_max_k.get(str(max(k0, k1, k2)), 0) + v
    single_cohort_per_node = sum(v for (k0, k1, k2), v in count_by_k.items() if max(k0, k1, k2) <= 1)

    # common-year single-cohort domain (the old 7-variable domain incl. x0)
    common = 1  # x0
    for yi, y in enumerate(years):
        per = {}
        for n in nodes:
            per[n] = [(pairs, cost) for pairs, cost, k in per_node[n]
                      if k == 0 or (k == 1 and pairs[yi] != (0, 0))]
        for combo in product(per[n0], per[n1], per[n2]):
            if all(c[0] == ((0, 0),) * len(years) for c in combo):
                continue
            z = tuple(v for pairs, _ in combo for pair in pairs for v in pair)
            if sum(c for _, c in combo) <= budget:
                common += 1
    wall_count = time.time() - t0

    # seeded sample, checked point by point against GeneralLattice.reasons
    rng = random.Random(SEED)
    sets = {n: {pairs: cost for pairs, cost, _ in per_node[n]} for n in nodes}
    disagree, n_adm = [], 0
    for i in range(N_SAMPLE):
        z = []
        member = True
        cost = 0.0
        for n in nodes:
            if rng.random() < 0.8:
                pairs, c, _ = per_node[n][rng.randrange(len(per_node[n]))]
            else:
                pairs = tuple((rng.randint(0, Z_MAX), rng.randint(0, Z_MAX)) for _ in years)
            z += [v for pair in pairs for v in pair]
            if pairs in sets[n]:
                cost += sets[n][pairs]
            else:
                member = False
        mine = member and cost <= budget
        theirs = lat.admissible(tuple(z))
        n_adm += theirs
        if mine != theirs:
            disagree.append({'z': z, 'mine': mine, 'lattice': theirs, 'reasons': lat.reasons(tuple(z))})
    nb_x0 = lat.neighbourhood(lat.x0())

    # A6 -- per-node option sets complete: every single-node-active point of the per-cohort grid, put to
    # GeneralLattice.reasons, is admissible iff it is in the per-node set with cost <= B
    a6_mismatch, a6_n = [], 0
    for ni, n in enumerate(nodes):
        kept = {pairs: c for pairs, c, _ in per_node[n]}
        for pairs in product(cohort_opts, repeat=len(years)):
            z = [0] * lat.dim
            for y, (zp, ze) in zip(years, pairs):
                i = lat.index(n, y)
                z[i], z[i + 1] = zp, ze
            a6_n += 1
            mine = pairs in kept and kept[pairs] <= budget
            if mine != lat.admissible(tuple(z)):
                a6_mismatch.append({'node': n, 'pairs': [list(p) for p in pairs]})
    # A7 -- every three-node combination of per-node options with summed cost <= B + A7_MARGIN, put to
    # GeneralLattice.reasons: the admissible ones are exactly the counted ones
    import bisect
    a7_margin = 1.0e5
    third = sorted((c, pairs) for pairs, c, _ in per_node[n2])
    third_costs = [c for c, _ in third]
    a7_checked, a7_adm, a7_mismatch = 0, 0, []
    for (p0, c0, _), (p1, c1, _) in product(per_node[n0], per_node[n1]):
        hi_i = bisect.bisect_right(third_costs, budget + a7_margin - c0 - c1)
        for c2, p2 in third[:hi_i]:
            z = tuple(v for pairs in (p0, p1, p2) for pair in pairs for v in pair)
            adm = lat.admissible(z)
            a7_checked += 1
            a7_adm += adm
            if adm != (c0 + c1 + c2 <= budget):
                a7_mismatch.append({'z': list(z), 'sum_node_costs': c0 + c1 + c2, 'lattice': adm})

    out = {
        'instance': inst.describe(),
        'per_cohort_option_count_incl_zero': len(cohort_opts),
        'per_node_option_count_cumulative_ok_cost_le_B_plus_1': sizes,
        'COUNT_ADMISSIBLE_STAGED_INCL_X0': total,
        'count_excluding_x0': total - 1,
        'boundary_ambiguous_triples_rechecked_exactly': ambiguous, 'boundary_ambiguous_admissible': exact_extra,
        'split_by_total_active_cohorts': dict(sorted(by_total_cohorts.items(), key=lambda kv: int(kv[0]))),
        'split_by_number_of_nodes_with_2plus_cohorts': dict(sorted(by_staged_nodes.items())),
        'split_by_max_cohorts_at_one_node': dict(sorted(by_max_k.items())),
        'single_cohort_per_node_incl_x0': single_cohort_per_node,
        'common_year_single_cohort_incl_x0': common,
        'cheapest_over_cap_plan_one_node': {'I_eur': over_cap_min[0], 'node': over_cap_min[1],
                                            'pairs_by_year': {str(y): list(p) for y, p in zip(years, over_cap_min[2])},
                                            'budget_eur': budget, 'over_budget': over_cap_min[0] > budget},
        'sample': {'seed': SEED, 'n': N_SAMPLE, 'n_admissible_by_lattice': n_adm, 'disagreements': disagree[:20],
                   'n_disagreements': len(disagree)},
        'neighbourhood_x0': {'n_admissible': nb_x0['n_admissible'], 'n_raw': nb_x0['n_raw'],
                             'size_bound': nb_x0['size_bound']},
        'a6_single_node_points_checked': a6_n, 'a6_mismatches': a6_mismatch[:20], 'a6_n_mismatches': len(a6_mismatch),
        'a7_margin_eur': a7_margin, 'a7_points_checked_by_lattice': a7_checked, 'a7_admissible_by_lattice': a7_adm,
        'a7_mismatches': a7_mismatch[:20], 'a7_n_mismatches': len(a7_mismatch),
        'wall_s': wall_count,
    }
    check('A6_per_node_sets_complete_against_GeneralLattice', not a6_mismatch,
          {'checked': a6_n, 'mismatches': len(a6_mismatch)})
    check('A7_count_reproduced_point_by_point_by_GeneralLattice', not a7_mismatch and a7_adm == total,
          {'checked': a7_checked, 'admissible': a7_adm, 'count': total, 'mismatches': len(a7_mismatch)})
    check('A1_single_cohort_per_node_equals_C5_L1_6694', single_cohort_per_node == 6694,
          {'got': single_cohort_per_node, 'C5_L1_incl_x0': 6694, 'S55_informational_excl_x0': 6693})
    check('A2_common_year_equals_old_domain_1705', common == 1705, {'got': common})
    check('A3_sample_agrees_with_GeneralLattice_reasons', not disagree, {'n_disagreements': len(disagree)})
    check('A4_cheapest_over_cap_plan_over_budget_C5_F1', over_cap_min[0] > budget
          and abs(over_cap_min[0] - 1108893.20) < 0.01, {'I_eur': over_cap_min[0]})
    check('A5_neighbourhood_x0_is_18_on_SRP1_C5', nb_x0['n_admissible'] == 18, {'got': nb_x0['n_admissible']})
    print(f'[A] staged domain on SRP1: {total} admissible incl. x0 ({wall_count:.1f} s)')
    return out


# ======================================================================================================================
#  Part B -- ESSO structure (production build, zero solves)
# ======================================================================================================================
def part_b(check):
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.repn import generate_standard_repn
    import definitions as DEFS
    import shared_energy_storage_data as SED
    import shared_resources_planning as SRP
    from shared_resources_planning import SharedResourcesPlanning

    side = os.path.join(OUT, 'side_writes')
    os.makedirs(os.path.join(side, 'Diagrams'))
    os.makedirs(os.path.join(side, 'Results', 'Logs'))
    watched = [os.path.join(SRP1_DIR, 'Diagrams'), os.path.join(SRP1_DIR, 'Results', 'Logs'),
               os.path.join(SRP1_DIR, 'Results')]

    def snap():
        return {d: sorted((f, os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d)) if os.path.isdir(d)
                else None for d in watched}

    before, git_before = snap(), _git('status', '--porcelain')
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
    read_s = time.time() - t0
    sed = planning.shared_ess_data
    years = list(sed.years)
    soh_point, ageing_on = SED._esso_ageing_model_settings(sed)
    sv = sed.params.salvage_value
    config = {'years': {str(y): sed.years[y] for y in years}, 'available_energy_soh_point': soh_point,
              'ageing_enabled': ageing_on, 'salvage_enabled': sv.enabled,
              'salvage_in_objective': getattr(sv, 'in_objective', None),
              'energy_recovery_fraction': sv.energy_recovery_fraction,
              'recycling_floor_fraction': sv.recycling_floor_fraction, 'cost_basis': sv.cost_basis,
              'health_basis': sv.health_basis, 'calendar_life_basis': sv.calendar_life_basis,
              'params_available_energy_soh_point_declared': sed.params.available_energy_soh_point,
              'slacks': sed.params.slacks, 'planning_read_s': read_s,
              'scenario_checksum': (planning.scenario_metadata or {}).get('combined_scenario_checksum')}
    check('B0_srp1_config_end_salvage_reported_only', soh_point == 'end' and sv.enabled and
          not getattr(sv, 'in_objective', False) and [int(y) for y in years] == [2025, 2030, 2035],
          {k: config[k] for k in ('available_energy_soh_point', 'salvage_enabled', 'salvage_in_objective', 'years')})

    def build(plan, in_objective=False):
        saved = sv.in_objective
        sv.in_objective = in_objective
        try:
            model = SED._build_subproblem(sed, NODE)
        finally:
            sv.in_objective = saved
        cand = {NODE: {y: {'s': plan[int(y)][0], 'e': plan[int(y)][1]} for y in years}}
        SED._update_model_with_candidate_solution(sed, {NODE: model}, cand)
        return model

    def cohort_rows(model, y_inv):
        rows = {}
        for name, idx, y in model._esso_cohort_constraints[y_inv]:
            if name in COHORT_FAMILIES:
                con = getattr(model, name)[idx]
                rows[f'{name}[{idx}]|y={y}'] = (str(con.expr), con.active)
        n_y = len(years)
        for y in range(n_y):  # available-energy row of (y_inv, y): built in (y_inv, y) order, one per pair
            con = model.available_e_capacity_unit[y_inv * n_y + y + 1]
            rows[f'available_e_capacity_unit[{y_inv * n_y + y + 1}]|y={y}'] = (str(con.expr), con.active)
        return rows

    models = {name: build(plan) for name, plan in PLANS.items()}
    # B1 -- cohort-0 rows identical across plans
    r0 = {name: cohort_rows(m, 0) for name, m in models.items()}
    same = all(r0[name] == r0['A_2025'] for name in r0)
    check('B1_cohort0_rows_identical_with_and_without_later_cohorts', same,
          {'n_rows': len(r0['A_2025']), 'n_active': sum(1 for v in r0['A_2025'].values() if v[1])})
    r1 = {name: cohort_rows(models[name], 1) for name in ('B_2025_2030', 'C_2025_2030_2035')}
    check('B1b_cohort1_rows_identical_with_and_without_cohort_2035', r1['B_2025_2030'] == r1['C_2025_2030_2035'],
          {'n_rows': len(r1['B_2025_2030'])})

    # B2 / B3 -- variable ownership of cohort-registered rows; SoH chain start
    own_violations, chain = [], {}
    mC = models['C_2025_2030_2035']
    for y_inv in mC.years:
        for name, idx, y in mC._esso_cohort_constraints[y_inv]:
            con = getattr(mC, name)[idx]
            if name == 'energy_storage_cohort_pnet_share_h3':
                continue  # the H3 row couples cohorts by design (against agg_pnet); inventoried in B4/B5
            for v in identify_variables(con.body, include_fixed=True):
                comp = v.parent_component().name
                if comp in PER_COHORT_VARS and v.index()[0] != y_inv:
                    own_violations.append({'row': f'{name}[{idx}]', 'y_inv': y_inv, 'var': v.name})
            if name == 'energy_storage_capacity_degradation' and y == y_inv and con.active:
                vs = sorted(v.name for v in identify_variables(con.body, include_fixed=True))
                if any('es_soh_per_unit_cumul' in s for s in vs):
                    chain.setdefault(str(y_inv), []).append(vs)
    check('B2_every_cohort_row_references_only_its_own_cohort', not own_violations,
          {'n_violations': len(own_violations), 'sample': own_violations[:5]})
    starts_at_one = {}
    for y_inv, lists in chain.items():
        soh_rows = [vs for vs in lists if any(s.startswith('es_D_per_unit') for s in vs)]
        prior = [s for vs in soh_rows for s in vs
                 if s.startswith('es_soh_per_unit_cumul') and s != f'es_soh_per_unit_cumul[{y_inv},{y_inv}]']
        starts_at_one[y_inv] = {'soh_chain_row_vars': soh_rows, 'references_earlier_block_soh': bool(prior)}
    fixed_pre = {f'{yi},{y}': (mC.es_soh_per_unit_cumul[yi, y].fixed, pe.value(mC.es_soh_per_unit_cumul[yi, y]))
                 for yi in mC.years for y in mC.years if y < yi}
    check('B3_later_cohorts_soh_chain_starts_at_one_in_installation_block',
          all(not v['references_earlier_block_soh'] for v in starts_at_one.values())
          and all(f and val == 1.0 for f, val in fixed_pre.values()),
          {'chain_start_rows': starts_at_one, 'pre_installation_soh_fixed_at_1': fixed_pre})

    # B4 -- coupling inventory (active rows referencing >= 2 cohorts, unfixed variables only)
    coupling = {}
    for name, m in models.items():
        fam_count = {}
        for con in m.component_data_objects(pe.Constraint, active=True):
            cohorts = {v.index()[0] for v in identify_variables(con.body, include_fixed=False)
                       if v.parent_component().name in PER_COHORT_VARS}
            if len(cohorts) >= 2:
                fam = con.parent_component().name
                fam_count[fam] = fam_count.get(fam, 0) + 1
        coupling[name] = fam_count
    expected_fams = {'rated_s_capacity', 'rated_e_capacity', 'energy_storage_operation_agg',
                     'energy_storage_cohort_pnet_share_h3'}
    # plan A: only the aggregate rated-capacity definitions may reference two cohorts (the in-window rated_per_unit
    # Vars of an inactive cohort stay unfixed, pinned to its zero Param by rated_*_capacity_unit)
    check('B4_cohort_coupling_only_through_aggregate_rows_and_H3',
          all(set(f) <= expected_fams for f in coupling.values())
          and set(coupling['A_2025']) <= {'rated_s_capacity', 'rated_e_capacity'}, coupling)

    # B5 -- H3 pro-rata rows
    h3 = {}
    n_dp = len(list(mC.days)) * len(list(mC.periods))
    for name, m in models.items():
        per_year = {}
        for y in m.years:
            active = sum(1 for yi in m.years for _n, idx, yy in m._esso_cohort_constraints[yi]
                         if _n == 'energy_storage_cohort_pnet_share_h3' and yy == y
                         and m.energy_storage_cohort_pnet_share_h3[idx].active)
            n_act = sum(1 for yi in m.years if not m._esso_cohort_inactive[yi]
                        and SED._esso_cohort_pair_is_within_lifetime(m, yi, y))
            shares = {str(yi): pe.value(m.es_pnet_cohort_share_h3[yi, y]) for yi in m.years}
            per_year[str(years[y])] = {'n_active_cohorts': n_act, 'active_h3_rows': active,
                                       'expected': max(n_act - 1, 0) * n_dp, 'shares': shares}
        h3[name] = per_year
    check('B5_h3_rows_N_active_minus_1_per_period',
          all(v['active_h3_rows'] == v['expected'] for p in h3.values() for v in p.values()), h3)

    # B6 -- salvage with in_objective on in memory (SRP1 grid, 'end')
    mS = build(PLANS['C_2025_2030_2035'], in_objective=True)
    term_idx = len(years) - 1
    disc_t = SED._get_terminal_discount_factor(sed)
    repn = generate_standard_repn(mS.salvage_credit_obj.expr, compute_values=True)
    coefs = {v.name: c for v, c in zip(repn.linear_vars, repn.linear_coefs)}
    idx = sed.get_shared_energy_storage_idx(NODE)
    hand, cohorts = {}, {}
    const_hand = 0.0
    for y_inv, year in enumerate(years):
        ess = sed.shared_energy_storages[year][idx]
        age, rem, rlf = SED._get_remaining_calendar_life(sed, y_inv, ess)
        c_e = SED._get_expected_energy_investment_cost(sed, year)
        e_inv = PLANS['C_2025_2030_2035'][int(year)][1]
        unit = disc_t * sv.energy_recovery_fraction * c_e * rlf
        if rlf > 0:
            hand[f'es_soh_per_unit_cumul[{y_inv},{term_idx}]'] = -unit * e_inv / (1 - ess.soh_min)
            const_hand += unit * e_inv * ess.soh_min / (1 - ess.soh_min)
        n_blocks_alive = len(range(y_inv, min(y_inv + round(ess.t_cal / sed.years[year]), len(years))))
        soh_cal = ess.phi_cal ** (sed.years[year] * n_blocks_alive)
        coefficients = mS._salvage_throughput_coefficients
        cohorts[str(year)] = {'age_at_terminal_y': age, 'remaining_life_y': rem, 'rlf': rlf,
                              'expected_energy_cost_installation_year_eur_per_mwh_undiscounted': c_e,
                              'e_mwh': e_inv, 'soh_min': ess.soh_min, 'phi_cal': ess.phi_cal, 't_cal': ess.t_cal,
                              'calendar_only_terminal_soh': soh_cal,
                              'salvage_at_calendar_only_soh_eur_pv2025': unit * e_inv * (soh_cal - ess.soh_min)
                              / (1 - ess.soh_min),
                              'dS_dTP_eur_per_mwh_at_calendar_only_soh': (-unit * soh_cal
                                                                          / ((1 - ess.soh_min) * 2 * ess.cl_eff)),
                              'k_cl_eff': ess.cl_eff,
                              'max_throughput_coef_c': max((max(v) for k, v in coefficients.items() if k[0] == y_inv),
                                                           default=0.0)}
    coef_ok = (set(coefs) == set(hand) and all(abs(coefs[k] - hand[k]) <= 1e-12 * abs(hand[k]) for k in hand)
               and abs(repn.constant - const_hand) <= 1e-12 * max(abs(const_hand), 1.0))
    no_converter = not any(n.startswith('es_s_') for n in coefs)
    # reporting expression identity at a synthetic consistent point (calendar-only terminal SoH)
    for y_inv, year in enumerate(years):
        var_r = mS.es_e_rated_per_unit[y_inv, term_idx]
        if not var_r.fixed:
            e_inv = PLANS['C_2025_2030_2035'][int(year)][1]
            soh = cohorts[str(year)]['calendar_only_terminal_soh']
            var_r.set_value(e_inv)
            mS.es_soh_per_unit_cumul[y_inv, term_idx].set_value(soh)
            mS.es_e_available_per_unit[y_inv, term_idx].set_value(e_inv * soh)
    s_rep, s_obj = pe.value(mS.salvage_value), pe.value(mS.salvage_credit_obj)
    check('B6_salvage_objective_coefficients_equal_hand_formula_on_SRP1_end',
          coef_ok and no_converter and abs(s_rep + s_obj) <= 1e-9 * max(abs(s_rep), 1.0),
          {'coefs': coefs, 'hand': hand, 'constant': repn.constant, 'constant_hand': const_hand,
           'no_converter_variable': no_converter, 'salvage_value_reporting': s_rep, 'salvage_credit_obj': s_obj})
    check('B6b_kappa_param_initialised_zero', pe.value(mS.salvage_objective_kappa) == 0.0,
          {'kappa_at_build': pe.value(mS.salvage_objective_kappa)})

    # B7 -- kappa on SRP1 and slack safety
    sigma = planning.params.admm.objective_scale
    with redirect_stdout(io.StringIO()):
        a, applied = SRP._resolve_esso_al_scale(planning, planning.params.admm, sigma)
        w_med = SRP._compute_median_admm_block_weight(planning)
    kappa = (a if applied else 1.0) / sigma
    bar = 1e-3 * DEFS.PENALTY_ESSO_SLACK
    worst = {}
    for y_inv, year in enumerate(years):
        c = cohorts[str(year)]
        worst[str(year)] = kappa * c['max_throughput_coef_c'] * c['calendar_only_terminal_soh']
    peak = max(worst.values())
    b7 = {'sigma_fixed': sigma, 'esso_al_scale_a': a, 'al_scale_applied': applied, 'median_block_weight': w_med,
          'kappa': kappa, 'kappa_times_w_med': kappa * w_med, 'PENALTY_ESSO_SLACK': DEFS.PENALTY_ESSO_SLACK,
          'bar_1e-3_penalty': bar, 'salvage_marginal_per_MW_period_by_cohort_at_calendar_only_soh': worst,
          'peak': peak, 'peak_over_bar': peak / bar, 'penalty_needed_for_bar': peak / 1e-3,
          'gcc3_five_block_peak_reference': 5.84}
    check('B7_kappa_equals_one_over_median_block_weight', abs(kappa * w_med - 1.0) <= 1e-12, b7)
    # slack safety is a MEASUREMENT here (no prediction was recorded for the SRP1 grid); b7['peak_over_bar'] > 1
    # means the G-CC-3 (ii) bar would fail on this grid at today's PENALTY_ESSO_SLACK
    b7['slack_safety_holds_at_PENALTY_ESSO_SLACK'] = peak <= bar

    after, git_after = snap(), _git('status', '--porcelain')
    check('B8_no_side_write_outside_output_dir', before == after and git_before == git_after,
          {'watched': watched, 'git_status_unchanged': git_before == git_after})
    return {'config': config, 'plans_node5_mva_mwh': {k: {str(y): list(v) for y, v in p.items()} for k, p in PLANS.items()},
            'cohort0_row_count': len(r0['A_2025']), 'coupling_inventory': coupling, 'h3': h3,
            'soh_chain_start': starts_at_one, 'pre_installation_soh_fixed': fixed_pre,
            'salvage': {'terminal_discount_factor': disc_t, 'terminal_horizon_years': sum(float(v) for v in sed.years.values()),
                        'terminal_date': int(years[0]) + int(sum(float(v) for v in sed.years.values())),
                        'cohorts': cohorts, 'objective_coefs': coefs, 'objective_constant': repn.constant,
                        'reporting_value_at_synthetic_point': s_rep, 'objective_value_at_synthetic_point': s_obj},
            'kappa_and_slack_safety': b7}


# ======================================================================================================================
#  Part C -- static source facts
# ======================================================================================================================
def part_c(check):
    facts = {}
    for fid, rel, pattern in SOURCE_FACTS:
        path = os.path.join(REPO, rel)
        rx = re.compile(pattern)
        hits = []
        with open(path) as handle:
            for i, line in enumerate(handle, start=1):
                if rx.search(line.rstrip('\n')):
                    hits.append({'line': i, 'text': line.strip()[:200]})
        facts[fid] = {'file': rel, 'pattern': pattern, 'hits': hits[:12], 'n_hits': len(hits)}
    check('C1_every_positive_fact_located',
          all(f['n_hits'] > 0 for k, f in facts.items() if k != 'params_penalty_key_absent'),
          {k: f['n_hits'] for k, f in facts.items()})
    check('C2_no_penalty_key_in_ess_parameters_reader', facts['params_penalty_key_absent']['n_hits'] == 0,
          facts['params_penalty_key_absent'])
    git = {'head': _git('rev-parse', 'HEAD'), 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
           'f6c23977_ancestor_of_head': subprocess.run(['git', 'merge-base', '--is-ancestor', 'f6c23977', 'HEAD'],
                                                       cwd=REPO).returncode == 0,
           '7324aa9c_ancestor_of_head': subprocess.run(['git', 'merge-base', '--is-ancestor', '7324aa9c', 'HEAD'],
                                                       cwd=REPO).returncode == 0,
           'production_commits_since_p516_start': _git('log', '--format=%h %s', 'efc21eb6^..HEAD', '--',
                                                        *[f for f in _git('ls-files', '*.py').split()
                                                          if '/' not in f and not re.match(r'p[0-9]', f)]).splitlines(),
           'production_diff_f6c23977_to_head': _git('diff', '--stat', 'f6c23977', 'HEAD', '--',
                                                    'shared_energy_storage_data.py', 'shared_energy_storage_parameters.py',
                                                    'shared_resources_planning.py')}
    return {'facts': facts, 'git': git,
            'sources_sha256': {rel: sha256_file(os.path.join(REPO, rel)) for rel in HASHED_SOURCES}}


def main():
    global OUT
    if '--manifest' in sys.argv:
        return manifest()
    if '--rehearsal' in sys.argv:  # debugging only: a directory OUTSIDE the repository; never the evidence run
        OUT = os.path.abspath(sys.argv[sys.argv.index('--rehearsal') + 1])
        if OUT.startswith(REPO + os.sep):
            raise SystemExit('refusing: --rehearsal must point outside the repository')
    _preflight()
    import gate_result_io as GRIO
    started = _utc()
    checks = []

    def check(cid, ok, detail):
        ok = bool(ok)
        checks.append({'id': cid, 'pass': ok, 'detail': detail})
        print(f"[check] {'PASS' if ok else 'FAIL'} {cid}", flush=True)
        return ok

    result = {'schema': SCHEMA, 'script': os.path.basename(__file__), 'started_utc': started,
              'interpreter': sys.executable}
    errors = {}
    for name, fn in (('A_lattice_count', part_a), ('B_esso_structure', part_b), ('C_source_facts', part_c)):
        try:
            result[name] = fn(check)
        except Exception as exc:  # recorded, not hidden
            import traceback
            errors[name] = traceback.format_exc()
            print(errors[name], file=sys.stderr, flush=True)
    guard_fail = GUARD.verify(0)
    result.update({'finished_utc': _utc(), 'errors': errors, 'checks': checks,
                   'failed_checks': [c['id'] for c in checks if not c['pass']],
                   'guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': guard_fail}})
    with open(os.path.join(OUT, 'reuse_audit_I.json'), 'w') as handle:
        handle.write(GRIO.dumps(result, indent=1))
    print(f"[done] checks {sum(c['pass'] for c in checks)}/{len(checks)}; errors {sorted(errors)}; "
          f"guard {dict(GUARD.counts)} failures {guard_fail}", flush=True)
    return 0 if (not errors and not guard_fail and all(c['pass'] for c in checks)) else 1


if __name__ == '__main__':
    sys.exit(main())
