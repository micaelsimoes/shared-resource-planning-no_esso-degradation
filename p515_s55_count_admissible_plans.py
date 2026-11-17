"""P5.15 S55 -- count of admissible investment plans under the baseline constraints (ZERO SOLVES).

Question (Expert, for one sentence in the paper): under the baseline constraints, how many admissible investment
plans are there, excluding x = 0?

PRIMARY RULE = the production search lattice, `p515_s47_phase_b_record.Lattice` (the lattice the Phase B searches
s47, s51 and the s53 certificate used; s51 calls `PB.Lattice(years, PB.unit_costs_from_w2(w2))`), built here
exactly as s47/s51 build it. The admissible set is `Lattice.domain()` without x = 0. Nothing is re-implemented
for the primary count: membership is `Lattice.reasons(z) == []`, cost is `Lattice.investment_cost(z)`.

CROSS-CHECKS (all must hold, else the script exits non-zero after writing its evidence):
  (a) completeness of `domain()`: every point of the grid zP, zE in 0..11 per node (one granule beyond every bound:
      E up to 5.5 MWh, P up to 2.75 MVA) x every year index is classified with `Lattice.reasons`; the reason-free
      storage points must equal `domain()` minus x = 0;
  (b) the production master's first-stage check `shared_resources_planning._check_candidate_first_stage_feasibility`
      (ratio, energy cap, budget with tolerance 1e-8), fed from a real `SharedEnergyStorageData` whose parameters and
      cost workbook are read by the production readers, applied to every grid point: its feasible set must equal (a);
  (c) `PB.w2_cross_check` (Lattice I(x) vs the production master's I_new_eur on the 105 W2 candidates, <= 1e-6 EUR);
  (d) the s51 frozen spec's `extra.domain_I_x_eur` (1705 labels incl. x0): same label set, I(x) within 1e-6 EUR;
  (e) the 2030 plan y2030 n5 0.25/0.5 + n7 1/3.5: I = B - p515_s53_f2_certificate.EXPECTED_BUDGET_SLACK_EUR to 1e-6;
  (f) the W182g inventory of 47 manuscript plans: verdict (ADMISSIBLE/INADMISSIBLE) == membership, I within 1e-6.

INFORMATIONAL (not the production rule): the count when each node may take its own installation year (STEP4 1.1
single-cohort form, 9 variables), decided by the production master check (b); the production lattice does not
admit it (p515_s47_phase_b_record.py docstring: per-node years are not evaluable by the frozen oracle, ambiguity A1).

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0; the
imported PB module's own parent guard verified at 0; pickle.load / loads blocked and counted.

Writes ONLY to a NEW directory data/SRP1/Results/P515S55/admissible_plan_count/. The launch command creates the
directory to hold launch.log (`mkdir` fails if it exists); the script refuses if the directory holds anything other
than launch.log, or if another copy of this script is running.
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S55 admissible-plan count (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('S55 count: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('S55 count: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s47_phase_b_record as PB  # noqa: E402  (installs its own parent guard, permitted=())
import shared_energy_storage_data as ESSD  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'admissible_plan_count')
OUT_JSON = 'admissible_plan_count.json'
OUT_CSV = 'admissible_plans.csv'
OUT_MANIFEST = 'manifest_sha256.json'
ALLOWED_PREEXISTING = {'launch.log'}

CASE_FILE = os.path.join('data', 'SRP1', 'SRP1.json')
ESS_DIR = os.path.join('data', 'SRP1')  # SharedEnergyStorageData.data_dir; files resolve under <dir>/SharedESS/
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
S47_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                        'campaign_spec_s47_phase_b_8cfa264e.json')
S51_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                        'campaign_spec_s51_f2_phase_b_5ce295e1.json')
W182G_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182g_cost_horizon_budget',
                          'w182g_cost_horizon_budget.json')
S53_CERT = 'p515_s53_f2_certificate.py'
S51_SCRIPT = 'p515_s51_f2_phase_b.py'

GRID_MAX = 11            # zP, zE in 0..11 per node: one granule beyond E_max (10) and beyond P_max implied (10)
BOUNDARY_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05)
TOL_EUR = 1e-6
PLAN_2030 = (1, 1, 4, 7, 0, 0, 1)   # y2030: n5 0.25/0.5, n7 1/3.5
PLAN_REF_UNIT = (0, 0, 1, 2, 0, 0, 0)  # y2025: n7 0.25/1 (reference unit)

# exact-text anchors (file, text): every constraint's source line is located, never typed
ANCHORS = {
    'nodes': ('p515_s47_phase_b_record.py', 'ACTIVE_NODES = (5, 7, 9)'),
    'granule_P': ('p515_s47_phase_b_record.py', 'GRANULE_P_MVA = 0.25'),
    'granule_E': ('p515_s47_phase_b_record.py', 'GRANULE_E_MWH = 0.5'),
    'E_max': ('p515_s47_phase_b_record.py', 'E_MAX_MWH = 5.0'),
    'zE_max': ('p515_s47_phase_b_record.py', 'ZE_MAX = 10  # E_MAX_MWH / GRANULE_E_MWH'),
    'budget': ('p515_s47_phase_b_record.py', 'BUDGET_EUR = 1e6'),
    'common_year_vector': ('p515_s47_phase_b_record.py',
                           'N_VARS = 2 * len(ACTIVE_NODES) + 1  # 7: (zP, zE) per node + the common year index'),
    'lattice_class_doc': ('p515_s47_phase_b_record.py',
                          '"""The common-year single-cohort lattice. z = [zP5, zE5, zP7, zE7, zP9, zE9, zY] (ints)."""'),
    'rule_nonneg': ('p515_s47_phase_b_record.py', 'if zp < 0 or ze < 0:'),
    'rule_P0_iff_E0': ('p515_s47_phase_b_record.py', 'if (zp == 0) != (ze == 0):'),
    'rule_duration_inclusive': ('p515_s47_phase_b_record.py', 'if zp > 0 and not (zp <= ze <= 2 * zp):'),
    'rule_E_cap': ('p515_s47_phase_b_record.py', 'if ze > ZE_MAX:'),
    'rule_year_bounds': ('p515_s47_phase_b_record.py', 'if self.has_storage(z) and not (0 <= z[-1] < len(self.years)):'),
    'rule_budget_le': ('p515_s47_phase_b_record.py', 'if not out and self.investment_cost(z) > self.budget:'),
    'I_x_lattice': ('p515_s47_phase_b_record.py',
                    'return sum(self.c_p[y] * GRANULE_P_MVA * zp + self.c_e[y] * GRANULE_E_MWH * ze for zp, ze in self.node_pairs(z))'),
    'unit_costs_discounted': ('p515_s47_phase_b_record.py',
                              "return {int(y): {'power': v['power_eur_per_mva_discounted'], 'energy': v['energy_eur_per_mwh_discounted']}"),
    'domain': ('p515_s47_phase_b_record.py', 'def domain(self):'),
    's51_builds_lattice': ('p515_s51_f2_phase_b.py', 'lattice = PB.Lattice(years, costs)'),
    'master_check_def': ('shared_resources_planning.py',
                         'def _check_candidate_first_stage_feasibility(planning_problem, candidate_solution):'),
    'master_check_tol': ('shared_resources_planning.py', 'tolerance = 1e-8'),
    'master_check_ratio_min': ('shared_resources_planning.py',
                               'if e_value + tolerance < params.min_energy_to_power_ratio * s_value:'),
    'master_check_ratio_max': ('shared_resources_planning.py',
                               'if e_value > params.max_energy_to_power_ratio * s_value + tolerance:'),
    'master_check_discount': ('shared_resources_planning.py',
                              '(1.00 + shared_ess_data.discount_factor) ** (int(year) - int(years[0]))'),
    'master_check_cap': ('shared_resources_planning.py',
                         "if total_capacity['e'] > params.max_capacity + tolerance:"),
    'master_check_budget': ('shared_resources_planning.py',
                            'if expected_investment_cost > params.budget + tolerance:'),
    'master_row_budget': ('shared_energy_storage_data.py',
                          'model.energy_storage_investment.add(investment_cost_total <= shared_ess_data.params.budget)'),
    'master_row_cap': ('shared_energy_storage_data.py',
                       'model.energy_storage_maximum_capacity.add(model.es_e_rated[e, y] <= shared_ess_data.params.max_capacity)'),
    'params_budget_read': ('shared_energy_storage_parameters.py',
                           "planning_parameters.budget = float(params_data['budget'])"),
    'params_cap_read': ('shared_energy_storage_parameters.py',
                        "planning_parameters.max_capacity = float(params_data['max_capacity'])"),
    'expected_slack_2030': ('p515_s53_f2_certificate.py', 'EXPECTED_BUDGET_SLACK_EUR = 7731.891715629841'),
    'budget_infeasible_cache': ('p515_s51_f2_phase_b.py',
                                "BUDGET_INFEASIBLE_CACHE_LABELS = ('n7_4h_e4_m2', 'n7_4h_e5_m2')"),
}

CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {detail}' if detail is not None and not ok else ''))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _input_record(path):
    tracked = bool(_git(['ls-files', '--', path]).strip())
    clean = not _git(['status', '--porcelain', '--', path]).strip()
    return {'path': path, 'sha256': _sha(path), 'tracked': tracked, 'clean': clean}


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


def locate_anchors():
    out, missing = {}, []
    cache = {}
    for key, (path, text) in ANCHORS.items():
        if path not in cache:
            with open(path) as handle:
                cache[path] = handle.read().splitlines()
        lines = [i + 1 for i, ln in enumerate(cache[path]) if text in ln]
        if not lines:
            missing.append(key)
        out[key] = {'file': path, 'text': text, 'lines': lines}
    return out, missing


def z_of_plan(lattice, year, nodes):
    """nodes: iterable of (node, P_MVA, E_MWh). Returns the lattice z (common year) or raises."""
    z = [0] * (2 * len(lattice.nodes))
    for n, p, e in nodes:
        i = lattice.nodes.index(int(n))
        zp, ze = p / PB.GRANULE_P_MVA, e / PB.GRANULE_E_MWH
        if zp != int(zp) or ze != int(ze):
            raise ValueError(f'off-lattice plan {year} {nodes}')
        z[2 * i], z[2 * i + 1] = int(zp), int(ze)
    return tuple(z) + (lattice.years.index(int(year)),)


def build_production_ess_data(years, discount_factor):
    """A real SharedEnergyStorageData whose params and cost workbook are read by the production readers
    (`read_parameters_from_file` -> shared_energy_storage_parameters._read_parameters_from_file;
    `read_shared_energy_storage_data_from_file` -> _get_operational_scenarios_info_from_excel_file and
    _get_investment_costs_from_excel_file). No network, no model."""
    d = ESSD.SharedEnergyStorageData()
    d.data_dir = ESS_DIR
    d.params_file = os.path.basename(ESS_PARAMS)
    d.data_file = 'SRP1_ESS.xlsx'
    d.years = list(years)
    d.discount_factor = float(discount_factor)
    d.read_parameters_from_file()
    d.read_shared_energy_storage_data_from_file()
    return d


def production_check_fn(ess_data, nodes, years):
    """Closure calling the production master check on a candidate given per node (zP, zE, year or None)."""
    pp = types.SimpleNamespace(shared_ess_data=ess_data)
    zero = {y: {'s': 0.0, 'e': 0.0} for y in years}
    inv_cache, tc_cache = {}, {}

    def node_dicts(zp, ze, year):
        key = (zp, ze, year)
        if key not in inv_cache:
            if year is None or (zp == 0 and ze == 0):
                inv_cache[key], tc_cache[key] = zero, zero
            else:
                s, e = zp * PB.GRANULE_P_MVA, ze * PB.GRANULE_E_MWH
                inv_cache[key] = {y: ({'s': s, 'e': e} if y == year else {'s': 0.0, 'e': 0.0}) for y in years}
                # one installation; t_cal = 15 y covers the 15-y horizon, so the cohort is in service from its year on
                tc_cache[key] = {y: ({'s': s, 'e': e} if y >= year else {'s': 0.0, 'e': 0.0}) for y in years}
        return inv_cache[key], tc_cache[key]

    def check(per_node):
        inv, tc = {}, {}
        for n, (zp, ze, year) in zip(nodes, per_node):
            inv[n], tc[n] = node_dicts(zp, ze, year)
        ok, why = SRP._check_candidate_first_stage_feasibility(pp, {'investment': inv, 'total_capacity': tc})
        return ok, why
    return check


def closed_form_I(ess_data, per_node):
    """Cross-check only: the srp:2252-2258 sum written out from the production-read cost tables."""
    years = list(ess_data.years)
    total = 0.0
    for zp, ze, year in per_node:
        if year is None or (zp == 0 and ze == 0):
            continue
        disc = 1.0 / ((1.0 + ess_data.discount_factor) ** (int(year) - int(years[0])))
        for sc, prob in enumerate(ess_data.prob_market_scenarios):
            total += disc * prob * (ess_data.cost_investment['power'][sc][year] * zp * PB.GRANULE_P_MVA
                                    + ess_data.cost_investment['energy'][sc][year] * ze * PB.GRANULE_E_MWH)
    return total


def main():
    started = datetime.now(timezone.utc).isoformat()
    # ---------------- refusals before anything is computed ----------------
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if os.path.exists(OUT_DIR):
        present = set(os.listdir(OUT_DIR))
        if not present <= ALLOWED_PREEXISTING:
            _log(f'REFUSE: output directory {OUT_DIR} exists and holds {sorted(present - ALLOWED_PREEXISTING)}')
            return 2
    else:
        os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
        os.mkdir(OUT_DIR)
    anchors, missing = locate_anchors()
    if missing:
        _log(f'REFUSE: anchors not found: {missing}')
        return 2

    inputs = {k: _input_record(p) for k, p in {
        'case_file': CASE_FILE, 'ess_params': ESS_PARAMS, 'cost_file': PB.COST_FILE['path'],
        'w2_investment_cost_results': PB.INVESTMENT_COST_RESULTS['path'], 's47_spec': S47_SPEC, 's51_spec': S51_SPEC,
        'w182g_json': W182G_JSON, 's53_certificate_script': S53_CERT, 's51_script': S51_SCRIPT,
        'phase_b_record_module': 'p515_s47_phase_b_record.py', 'srp_module': 'shared_resources_planning.py',
        'essd_module': 'shared_energy_storage_data.py', 'essp_module': 'shared_energy_storage_parameters.py',
        'guard_module': 'p513_solve_profile_guard.py', 'writer_module': 'gate_result_io.py', 'this_script': SCRIPT,
    }.items()}
    _ck('cost file sha256 == PB.COST_FILE pin', inputs['cost_file']['sha256'] == PB.COST_FILE['sha256'],
        inputs['cost_file']['sha256'])
    _ck('W2 results sha256 == PB.INVESTMENT_COST_RESULTS pin',
        inputs['w2_investment_cost_results']['sha256'] == PB.INVESTMENT_COST_RESULTS['sha256'])

    # ---------------- the production lattice, built as s47 / s51 build it ----------------
    with open(PB.INVESTMENT_COST_RESULTS['path']) as handle:
        w2 = json.load(handle)
    costs = PB.unit_costs_from_w2(w2)
    years = tuple(sorted(costs))
    lattice = PB.Lattice(years, costs)
    with open(CASE_FILE) as handle:
        case = json.load(handle)
    case_years = tuple(sorted(int(y) for y in case['Years']))
    _ck('lattice years == case-file representative years', years == case_years, [years, case_years])
    _ck('W2 master facts: budget 1e6, max_capacity 5 MWh',
        (w2.get('master_facts') or {}).get('budget_eur') == PB.BUDGET_EUR
        and (w2.get('master_facts') or {}).get('max_capacity_mwh') == PB.E_MAX_MWH, w2.get('master_facts'))
    w2x = PB.w2_cross_check(lattice, w2)
    _ck('(c) Lattice I(x) == W2 production-master I_new_eur (<= 1e-6)', w2x['ok'], w2x)

    # production parameters / cost workbook through the production readers
    ess = build_production_ess_data(years, case['DiscountFactor'])
    prm = ess.params
    params_view = {'budget': prm.budget, 'max_capacity': prm.max_capacity,
                   'min_energy_to_power_ratio': prm.min_energy_to_power_ratio,
                   'max_energy_to_power_ratio': prm.max_energy_to_power_ratio,
                   'discount_factor': ess.discount_factor, 'prob_scenarios': list(ess.prob_market_scenarios)}
    _ck('production params == lattice constants (budget, E cap, ratio 2..4)',
        prm.budget == PB.BUDGET_EUR and prm.max_capacity == PB.E_MAX_MWH
        and prm.min_energy_to_power_ratio == 2.0 and prm.max_energy_to_power_ratio == 4.0, params_view)
    unit_cost_diff = {}
    for y in years:
        disc = 1.0 / (1.0 + ess.discount_factor) ** (y - years[0])
        cp = disc * sum(p * ess.cost_investment['power'][s][y] for s, p in enumerate(ess.prob_market_scenarios))
        ce = disc * sum(p * ess.cost_investment['energy'][s][y] for s, p in enumerate(ess.prob_market_scenarios))
        unit_cost_diff[y] = {'power_eur_per_mva_discounted': cp, 'energy_eur_per_mwh_discounted': ce,
                             'abs_diff_power': abs(cp - lattice.c_p[y]), 'abs_diff_energy': abs(ce - lattice.c_e[y])}
    _ck('discounted expected unit costs (production-read workbook) == lattice unit costs (<= 1e-6)',
        all(v['abs_diff_power'] <= TOL_EUR and v['abs_diff_energy'] <= TOL_EUR for v in unit_cost_diff.values()),
        unit_cost_diff)

    # ---------------- PRIMARY: Lattice.domain() ----------------
    domain = lattice.domain()
    admissible = [z for z in domain if lattice.has_storage(z)]
    adm_set = set(admissible)
    _ck('domain()[0] is x = 0 and appears once', domain[0] == lattice.x0() and domain.count(lattice.x0()) == 1)
    _ck('every admissible point has Lattice.reasons == []', all(not lattice.reasons(z) for z in admissible))
    _ck('admissible points distinct', len(adm_set) == len(admissible))

    # ---------------- (a) + (b): full grid, Lattice.reasons and the production master check ----------------
    _log(f'grid: zP, zE in 0..{GRID_MAX} per node x {len(years)} years = {(GRID_MAX + 1) ** 6 * len(years)} points')
    check = production_check_fn(ess, lattice.nodes, years)
    per_node_grid = [(zp, ze) for zp in range(GRID_MAX + 1) for ze in range(GRID_MAX + 1)]
    grid_lattice_ok, grid_master_ok = set(), set()
    reason_classes = Counter()
    rule_feasible_over_budget = []   # bound/duration-feasible, budget-rejected (by the lattice)
    n_grid = 0
    master_vs_lattice_disagree = []
    for yi, year in enumerate(years):
        for combo in product(per_node_grid, repeat=len(lattice.nodes)):
            z = tuple(v for pair in combo for v in pair) + (yi,)
            if not lattice.has_storage(z):
                continue
            n_grid += 1
            why = lattice.reasons(z)
            ok_m, why_m = check([(zp, ze, year) for zp, ze in combo])
            if not why:
                grid_lattice_ok.add(z)
            else:
                for cls in sorted({r.split(':', 1)[0] for r in why}):
                    reason_classes[cls] += 1
                if all(r.startswith('budget') for r in why):
                    rule_feasible_over_budget.append(z)
            if ok_m:
                grid_master_ok.add(z)
            if ok_m != (not why) and len(master_vs_lattice_disagree) < 20:
                master_vs_lattice_disagree.append({'z': list(z), 'lattice': why, 'master': why_m})
    _ck('(a) grid reason-free storage points == domain() minus x = 0', grid_lattice_ok == adm_set,
        {'grid_only': len(grid_lattice_ok - adm_set), 'domain_only': len(adm_set - grid_lattice_ok)})
    _ck('(b) production master check feasible set == lattice admissible set', grid_master_ok == adm_set,
        {'master_only': len(grid_master_ok - adm_set), 'lattice_only': len(adm_set - grid_master_ok),
         'first_disagreements': master_vs_lattice_disagree})
    _ck('no admissible point has zP or zE at the grid edge (grid is wide enough)',
        all(max(z[:-1]) < GRID_MAX for z in grid_lattice_ok | grid_master_ok))

    # ---------------- I(x), slack, cross-check vs the production-read workbook ----------------
    B = lattice.budget
    rows = []
    max_cf_diff = 0.0
    for z in sorted(admissible, key=lambda q: (q[-1], q)):
        year = lattice.years[z[-1]]
        I = lattice.investment_cost(z)
        cf = closed_form_I(ess, [(zp, ze, year) for zp, ze in lattice.node_pairs(z)])
        max_cf_diff = max(max_cf_diff, abs(cf - I))
        nm = lattice.nodes_map(z)
        rows.append({'label': lattice.label(z), 'year': year, 'n_nodes': sum(1 for p, e in nm.values() if p or e),
                     'nodes_with_storage': '+'.join(str(n) for n, (p, e) in nm.items() if p or e),
                     **{f'P{n}_MVA': nm[n][0] for n in lattice.nodes}, **{f'E{n}_MWh': nm[n][1] for n in lattice.nodes},
                     'z': ' '.join(str(v) for v in z), 'I_eur': I, 'slack_eur': B - I, 'slack_pct_of_B': 100.0 * (B - I) / B,
                     'within_1pct_of_B': bool(B - I < 0.01 * B)})
    _ck('lattice I(x) == closed form from production-read workbook, all admissible plans (<= 1e-6)',
        max_cf_diff <= TOL_EUR, max_cf_diff)

    # ---------------- (d) the s51 frozen spec domain ----------------
    with open(S51_SPEC) as handle:
        s51 = json.load(handle)
    spec_dom = s51['extra']['domain_I_x_eur']
    spec_sum = s51['extra']['domain_summary']
    mine = {r['label'] + '_m2': r['I_eur'] for r in rows}
    mine['x0_m2'] = 0.0
    same_labels = set(spec_dom) == set(mine)
    max_spec_diff = max(abs(spec_dom[k] - mine[k]) for k in mine if k in spec_dom)
    _ck('(d) s51 frozen spec domain: same 1705 labels (x0 incl.)', same_labels,
        {'spec_only': sorted(set(spec_dom) - set(mine))[:10], 'mine_only': sorted(set(mine) - set(spec_dom))[:10]})
    _ck('(d) s51 frozen spec domain: I(x) within 1e-6', max_spec_diff <= TOL_EUR, max_spec_diff)
    with open(S47_SPEC) as handle:
        s47_sum = json.load(handle)['extra']['domain_summary']

    # ---------------- splits ----------------
    by_year = Counter(r['year'] for r in rows)
    by_n = Counter(r['n_nodes'] for r in rows)
    by_year_n = Counter((r['year'], r['n_nodes']) for r in rows)
    by_subset = Counter(r['nodes_with_storage'] for r in rows)
    by_year_subset = Counter((r['year'], r['nodes_with_storage']) for r in rows)
    _ck('per-year split == s47 and s51 frozen spec domain_summary',
        {str(k): v for k, v in by_year.items()} == {str(k): v for k, v in spec_sum['per_year'].items()}
        == {str(k): v for k, v in s47_sum['per_year'].items()}
        and len(rows) == spec_sum['n_storage_points'] == s47_sum['n_storage_points'],
        {'mine': dict(by_year), 's51': spec_sum, 's47': s47_sum})

    # ---------------- known plans ----------------
    known = {}
    i_2030 = lattice.investment_cost(PLAN_2030)
    with open(S53_CERT) as handle:
        m = re.search(r'^EXPECTED_BUDGET_SLACK_EUR = ([0-9.eE+-]+)$', handle.read(), re.M)
    expected_slack = float(m.group(1))
    known['2030_plan_n5_0.25_0.5__n7_1_3.5'] = {
        'label': lattice.label(PLAN_2030), 'z': list(PLAN_2030), 'I_eur': i_2030, 'slack_eur': B - i_2030,
        'in_set': PLAN_2030 in adm_set, 'expected_I_from_s53_slack': B - expected_slack,
        'abs_diff_vs_s53': abs(i_2030 - (B - expected_slack)), 'printed_992268.11_abs_diff': abs(i_2030 - 992268.11)}
    _ck('(e) 2030 plan I == B - s53 EXPECTED_BUDGET_SLACK_EUR (<= 1e-6) and in the set',
        known['2030_plan_n5_0.25_0.5__n7_1_3.5']['abs_diff_vs_s53'] <= TOL_EUR and PLAN_2030 in adm_set,
        known['2030_plan_n5_0.25_0.5__n7_1_3.5'])
    _ck('(e) 2030 plan I rounds to 992,268.11', round(i_2030, 2) == 992268.11, i_2030)
    i_ref = lattice.investment_cost(PLAN_REF_UNIT)
    known['reference_unit_n7_0.25_1_2025'] = {'label': lattice.label(PLAN_REF_UNIT), 'z': list(PLAN_REF_UNIT),
                                              'I_eur': i_ref, 'slack_eur': B - i_ref, 'in_set': PLAN_REF_UNIT in adm_set}
    _ck('reference unit 7: 0.25/1 @2025 in the set', PLAN_REF_UNIT in adm_set, known['reference_unit_n7_0.25_1_2025'])
    # s51 cache budget facts (recorded verdicts of the 2025 node-7 4 h ladder, incl. the two over budget)
    s51_rows = {}
    for lab, row in s51['extra']['budget_facts']['cache_rows'].items():
        z = tuple(row['z'])
        s51_rows[lab] = {'z': row['z'], 'recorded_budget_feasible': row['budget_feasible'],
                         'recorded_I_x_eur': row['I_x_eur'], 'I_eur': lattice.investment_cost(z),
                         'in_set': (z in adm_set) if lattice.has_storage(z) else None}
    known['s51_budget_facts_cache_rows'] = s51_rows
    _ck('s51 recorded budget verdicts agree with membership (incl. n7_4h_e4/e5 over budget)',
        all(r['in_set'] is None or r['in_set'] == r['recorded_budget_feasible'] for r in s51_rows.values())
        and all(abs(r['I_eur'] - r['recorded_I_x_eur']) <= TOL_EUR for r in s51_rows.values())
        and any(r['recorded_budget_feasible'] is False for r in s51_rows.values()), s51_rows)
    # W182g inventory (47 plans named in main.tex)
    with open(W182G_JSON) as handle:
        inv = json.load(handle)['C3']['inventory']
    w182g = []
    for p in inv:
        z = z_of_plan(lattice, p['year'], p['nodes'])
        in_set = z in adm_set
        w182g.append({'plan': p['plan'], 'z': list(z), 'recorded_verdict': p['verdict'], 'recorded_I_eur': p['I_eur'],
                      'I_eur': lattice.investment_cost(z), 'in_set': in_set,
                      'agrees': in_set == p['verdict'].startswith('ADMISSIBLE')  # 'ADMISSIBLE' / '... (BOUNDARY)'
                      and abs(lattice.investment_cost(z) - p['I_eur']) <= TOL_EUR})
    over = [w for w in w182g if w['recorded_verdict'] == 'INADMISSIBLE']
    _ck('(f) W182g 47-plan inventory: verdict == membership and I within 1e-6 for every plan',
        len(w182g) == 47 and all(w['agrees'] for w in w182g), [w for w in w182g if not w['agrees']])
    _ck('(f) the W182g over-budget plans (recorded) are outside the set',
        len(over) == 5 and not any(w['in_set'] for w in over), [w['plan'] for w in over])
    known['w182g_inventory'] = w182g

    # ---------------- boundary sensitivity ----------------
    over_I = sorted(lattice.investment_cost(z) for z in rule_feasible_over_budget)
    adm_I = sorted(r['I_eur'] for r in rows)
    sens = {'B_eur': B, 'max_admissible_I_eur': adm_I[-1], 'min_slack_eur': B - adm_I[-1],
            'min_over_budget_I_eur': over_I[0] if over_I else None,
            'n_rule_feasible_over_budget': len(over_I),
            'n_exactly_at_B': sum(1 for v in adm_I if v == B),
            'min_abs_B_minus_I_over_all_rule_feasible_eur': min(abs(B - v) for v in adm_I + over_I),
            'bands': {}}
    for f in BOUNDARY_FRACTIONS:
        below = sum(1 for v in adm_I if B - v < f * B)
        above = sum(1 for v in over_I if v - B <= f * B)
        sens['bands'][f'{f:g}'] = {
            'admissible_with_slack_below_fraction_of_B': below,
            'over_budget_within_fraction_of_B': above,
            'count_if_B_lowered_by_fraction': sum(1 for v in adm_I if v <= (1 - f) * B),
            'count_if_B_raised_by_fraction': sum(1 for v in adm_I + over_I if v <= (1 + f) * B),
            'admissible_within_by_year': dict(Counter(r['year'] for r in rows if B - r['I_eur'] < f * B)),
        }
    _ck('no plan within the production tolerance 1e-8 of B (<=, <, <= +1e-8 give one count)',
        sens['min_abs_B_minus_I_over_all_rule_feasible_eur'] > 1e-8, sens['min_abs_B_minus_I_over_all_rule_feasible_eur'])

    # ---------------- INFORMATIONAL: per-node installation years (not the production lattice) ----------------
    node_opts = [(0, 0, None)] + [(zp, ze, y) for (zp, ze) in per_node_grid if zp > 0 and zp <= ze <= 2 * zp
                                  and ze <= PB.ZE_MAX for y in years]
    n_per_node_year, n_per_node_year_common, n_per_node_year_mixed = 0, 0, 0
    for combo in product(node_opts, repeat=len(lattice.nodes)):
        active_years = {y for zp, ze, y in combo if y is not None}
        if not active_years:
            continue
        ok_m, _why = check(list(combo))
        if ok_m:
            n_per_node_year += 1
            if len(active_years) == 1:
                n_per_node_year_common += 1
            else:
                n_per_node_year_mixed += 1
    _ck('informational per-node-year enumeration: its common-year subset == primary count',
        n_per_node_year_common == len(rows), [n_per_node_year_common, len(rows)])

    # ---------------- guards ----------------
    guard_fail = GUARD.verify(0)
    pb_guard_fail = PB.PARENT_GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', not guard_fail, guard_fail)
    _ck('PB parent guard verify(0) == []', not pb_guard_fail, pb_guard_fail)
    _ck('pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)

    rule = ('x = (P_n, E_n)_{n in {5, 7, 9}} with ONE common installation year y in {2025, 2030, 2035} (the SRP1 '
            'representative years); P_n in 0.25*Z>=0 MVA, E_n in 0.5*Z>=0 MWh; P_n = 0 <=> E_n = 0; for P_n > 0: '
            '2*P_n <= E_n <= 4*P_n (inclusive; zP <= zE <= 2*zP in lattice units); E_n <= 5 MWh (no separate P cap; '
            'P_n <= 2.5 MVA is implied); I(x) = sum_n 1.02^-(y-2025) * sum_c w_c (cS_{c,y} P_n + cE_{c,y} E_n), '
            'w = (0.35, 0.55, 0.10), unit costs from SRP1_ESS.xlsx; admissible iff I(x) <= 1e6 EUR (inclusive, no '
            'tolerance in the lattice; the production master check uses <= 1e6 + 1e-8); x = 0 excluded. '
            'Implemented by p515_s47_phase_b_record.Lattice.reasons / .domain (used by s47, s51, s53).')
    payload = {
        'schema': 'p515_s55_admissible_plan_count_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable,
        'git_head': _git(['rev-parse', 'HEAD']).strip(),
        'question': 'number of admissible investment plans under the baseline constraints, excluding x = 0',
        'rule_operational': rule,
        'COUNT_ADMISSIBLE_EXCLUDING_X0': len(rows),
        'node_set': list(lattice.nodes), 'year_set': list(lattice.years),
        'budget_eur': B, 'budget_comparison': 'I(x) <= B (lattice: reject iff I > B, no tolerance)',
        'E_max_MWh': PB.E_MAX_MWH, 'granules': {'P_MVA': PB.GRANULE_P_MVA, 'E_MWh': PB.GRANULE_E_MWH},
        'duration_h': [2, 4], 'duration_inclusive': True, 'min_unit': '0.25 MVA / 0.5 MWh',
        'unit_costs_discounted_eur': {str(y): {'power_per_MVA': lattice.c_p[y], 'energy_per_MWh': lattice.c_e[y]}
                                      for y in years},
        'unit_cost_check_vs_production_read_workbook': {str(k): v for k, v in unit_cost_diff.items()},
        'production_params_read': params_view,
        'cost_file_sha256': inputs['cost_file']['sha256'],
        'split_by_year': {str(k): by_year[k] for k in sorted(by_year)},
        'split_by_n_nodes': {str(k): by_n[k] for k in sorted(by_n)},
        'split_by_year_and_n_nodes': {f'{y}|{n}': by_year_n[(y, n)] for y, n in sorted(by_year_n)},
        'split_by_node_subset': {k: by_subset[k] for k in sorted(by_subset)},
        'split_by_year_and_node_subset': {f'{y}|{s}': by_year_subset[(y, s)] for y, s in sorted(by_year_subset)},
        'boundary_sensitivity': sens,
        'grid': {'zP_zE_range': [0, GRID_MAX], 'n_storage_points_classified': n_grid,
                 'lattice_reason_classes': dict(reason_classes),
                 'n_rule_feasible_over_budget': len(rule_feasible_over_budget)},
        'known_plans': known,
        'frozen_spec_domain_summary': {'s47': s47_sum, 's51': spec_sum},
        'informational_not_production': {
            'per_node_installation_years_count': n_per_node_year,
            'of_which_common_year': n_per_node_year_common, 'of_which_mixed_years': n_per_node_year_mixed,
            'rule': ('as the primary rule but each node with storage takes its own year in {2025, 2030, 2035} (STEP4 '
                     '1.1 single-cohort form); decided by the production master check; NOT the production search '
                     'lattice, which enforces a common year')},
        'anchors': anchors, 'inputs': inputs, 'checks': CHECKS,
        'failed_checks': [c['check'] for c in CHECKS if not c['ok']],
        'all_checks_ok': all(c['ok'] for c in CHECKS),
        'guards': {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                           'verify_0': guard_fail},
                   'pb_parent': {'label': PB.PARENT_GUARD.label, 'counts': dict(PB.PARENT_GUARD.counts),
                                 'verify_0': pb_guard_fail},
                   'pickle_counts': dict(PICKLE_COUNTS)},
    }
    json_path = os.path.join(OUT_DIR, OUT_JSON)
    with open(json_path, 'w') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=False)
    csv_path = os.path.join(OUT_DIR, OUT_CSV)
    fields = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    manifest = {'outputs': {OUT_JSON: _sha(json_path), OUT_CSV: _sha(csv_path)},
                'inputs': inputs, 'note': 'launch.log is written while the script runs; its sha256 is recorded in '
                                          'the Worker report, not here'}
    with open(os.path.join(OUT_DIR, OUT_MANIFEST), 'w') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'COUNT (excluding x = 0) = {len(rows)}; by year {dict(by_year)}; by n_nodes {dict(by_n)}')
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
