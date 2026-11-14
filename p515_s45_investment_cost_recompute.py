"""P5.15 Addendum 27, task W2 -- I(x) and budget position with the corrected cost file.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 (task W2);
data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json (`master_constraints`,
`A0`, `A1`, `cost_file`).

ZERO SOLVES. A `SolveProfileGuard` with NO permitted call site is installed before any
production module is imported and stays installed for the whole run; `verify(0)` is
checked exactly (zero permitted solves, zero process launches, zero blocked attempts).

Machinery REUSED (by import) from p515_s44_addendum26_confirmations.py item 2:
  * I(x) = `model.investment_cost` of the production Benders master built by
    `shared_ess_data.build_master_problem()`, candidate loaded with production's
    `load_candidate_solution_into_master_model`, evaluated with `pe.value` (no solve);
  * budget slack read from the master's OWN budget row (ub - body) and max-capacity /
    ratio rows checked (`_master_rows_state`);
  * cross-check: `p56a_oracle.investment_cost` (independent transcription);
  * OLD cost file: bytes from git (`git show 2cada62b~1:<path>`) written to a temporary
    file OUTSIDE the repository and read by production's own reader
    `_read_shared_energy_storage_data_from_file` into a deep copy of the loaded
    SharedEnergyStorageData; the on-disk workbook is never written.
  * workbook parsing helpers (`_read_workbook_bytes`, `_compare_workbooks`,
    `_expected_unit_costs`).
Power / energy split of I(x): `pyomo.repn.generate_standard_repn` of the production
`model.investment_cost` expression (linear), terms grouped by the parent Var
(`es_s_investment` = power, `es_e_investment` = energy); the split is checked to sum to
`pe.value(model.investment_cost)`.

CONCURRENCY: the reused confirmations script's `_preflight_no_concurrent_harness` refuses
while any p5* python process is alive. The Planner authorized this zero-solve build to run
concurrently with the AA C* re-verification campaign, so that preflight is NOT called here;
instead the running p5* processes are recorded (read-only) and the campaign lock is only
read (never touched). The script writes only under OUT_DIR; a before/after scan of files
modified anywhere in the repository during the run is recorded.

Usage (attached, both streams captured by the caller):
    python p515_s45_investment_cost_recompute.py > <OUT_DIR>/launch.log 2>&1
    python p515_s45_investment_cost_recompute.py --manifest
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = 'P5.15 Addendum 27 W2 -- I(x) and budget position with the corrected cost file (zero solves)'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 (task W2)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json '
    '(master_constraints, A0, A1, cost_file)',
]
SPEC_REL = 'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json'
SPEC_PATH = os.path.join(REPO, SPEC_REL)
S44_SELECTION_SPEC_REL = ('data/SRP1/Results/P515S44/campaign_s44_selection_aa/'
                          'campaign_spec_s44_selection_aa_4135c8d4.json')
OUT_REL = 'data/SRP1/Results/P515S45/investment_cost'
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'investment_cost_results.json'
MANIFEST_NAME = 'manifest_sha256.json'
LAUNCH_LOG_NAME = 'launch.log'

XLSX_REL = 'data/SRP1/SharedESS/SRP1_ESS.xlsx'
XLSX_PATH = os.path.join(REPO, XLSX_REL)
NEW_SHA256 = 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'
OLD_REV = '2cada62b~1'
OLD_SHA256 = '1458147446e9b70190465f42cef761af9cfd8b209e91035d858e2e63941d1414'
BUDGET_EXPECTED = 1.0e6
MAX_CAPACITY_EXPECTED = 5.0
ACTIVE_NODES = (5, 7, 9)
LADDER_ENERGIES = (1.0, 2.0, 3.0, 4.0, 5.0)
DURATIONS_H = (2, 4)
CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
CONCURRENT_CAMPAIGN_DIR_REL = 'data/SRP1/Results/P515S45/reverify_aa_c_star'


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git(args, binary=False):
    res = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=not binary)
    return res.returncode, res.stdout, res.stderr


# ======================================================================================
#  candidates (instance identity = the campaign harness's canonical form and key)
# ======================================================================================
def _reference_candidates(spec, s44_sel):
    paper = [c for c in s44_sel['candidates'] if c['label'].startswith('paper_plan')]
    if len(paper) != 1:
        raise RuntimeError(f'expected one paper_plan candidate in {S44_SELECTION_SPEC_REL}')
    paper_nodes = {int(n): tuple(v) for n, v in paper[0]['canonical']['nodes'].items()
                   if tuple(v) != (0.0, 0.0)}
    cs = (0.96875, 3.875)
    return [
        ('zero', 2025, {}, 'x = 0 at every node'),
        ('paper_plan', 2025, paper_nodes,
         f'{S44_SELECTION_SPEC_REL} candidate {paper[0]["label"]} canonical (S44 selection)'),
        ('c_star', 2025, {5: cs, 7: cs, 9: cs}, 'C*: 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9'),
        ('node7_empty', 2025, {5: cs, 9: cs}, 'C* at nodes 5 and 9; node 7 zero'),
        ('two_c_star', 2025, {n: (1.9375, 7.75) for n in ACTIVE_NODES},
         '2 x C*: 1.9375 MVA / 7.75 MWh at nodes 5, 7, 9'),
        ('lattice_plan', 2025, {7: (1.5, 3.0)}, 'lattice plan: node 7, 1.5 MVA / 3.0 MWh'),
        ('lattice_c_star', 2025, {n: (1.0, 4.0) for n in ACTIVE_NODES},
         'lattice C*: 1.0 MVA / 4.0 MWh at nodes 5, 7, 9'),
    ]


def _spec_points(block, group, default_year=None):
    out = []
    for p in block:
        year = p.get('investment_year', default_year)
        out.append((p['label'], int(year), {int(n): tuple(v) for n, v in p['nodes'].items()},
                    f'spec v15 {group}'))
    return out


def _year_ladder():
    out = []
    for year in (2030, 2035):
        for n in ACTIVE_NODES:
            for h in DURATIONS_H:
                for e in LADDER_ENERGIES:
                    out.append((f'n{n}_{h}h_e{int(e)}_y{year}', year, {n: (e / h, e)},
                                'W2 year ladder (every node, both durations, E in 1..5)'))
    return out


# ======================================================================================
#  evaluation (production master expression; reused confirmations machinery)
# ======================================================================================
def _full_candidate(O, planning, node_map, year):
    nodes, years = O.nodes_and_years(planning)
    if year not in years:
        raise ValueError(f'investment year {year} not in case years {years}')
    x = {}
    for n in nodes:
        for y in years:
            s, e = node_map.get(n, (0.0, 0.0)) if y == year else (0.0, 0.0)
            x[(n, y)] = {'s': float(s), 'e': float(e)}
    return O.vector_to_candidate(planning, x)


def _split_investment_cost(master, sed, pe, generate_standard_repn):
    """Linear decomposition of the production expression by parent Var, node and year."""
    repn = generate_standard_repn(master.investment_cost.expr, compute_values=True)
    if repn.nonlinear_expr is not None or repn.quadratic_vars:
        raise RuntimeError('model.investment_cost is not linear')
    years = list(sed.years)
    node_of = {e: sed.shared_energy_storages[years[0]][e].bus for e in master.energy_storages}
    parts = {'power': 0.0, 'energy': 0.0, 'constant': float(repn.constant)}
    by_node = {}
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        name = var.parent_component().name
        e_idx, y_idx = var.index()
        kind = {'es_s_investment': 'power', 'es_e_investment': 'energy'}.get(name)
        if kind is None:
            raise RuntimeError(f'unexpected variable {var.name} in model.investment_cost')
        v = pe.value(var) * coef
        parts[kind] += v
        key = f'{node_of[e_idx]}|{years[y_idx]}'
        slot = by_node.setdefault(key, {'power': 0.0, 'energy': 0.0,
                                        'coef_power_eur_per_mva': None,
                                        'coef_energy_eur_per_mwh': None})
        slot[kind] += v
        slot['coef_power_eur_per_mva' if kind == 'power' else 'coef_energy_eur_per_mwh'] = coef
    return parts, by_node


def _evaluate_all(C, O, srp, sed, planning, candidates, pe, generate_standard_repn, H,
                  planning_for_feasibility):
    master = sed.build_master_problem()
    budget_row = master.energy_storage_investment[1]
    out = {}
    for label, year, node_map, source in candidates:
        cand = _full_candidate(O, planning, node_map, year)
        sed.load_candidate_solution_into_master_model(master, cand)
        i_master = pe.value(master.investment_cost)
        rows = C._master_rows_state(master, pe)
        parts, by_node_year = _split_investment_cost(master, sed, pe, generate_standard_repn)
        split_sum = parts['power'] + parts['energy'] + parts['constant']
        i_oracle = O.investment_cost(types.SimpleNamespace(shared_ess_data=sed), cand)
        feasible, reasons = srp._check_candidate_first_stage_feasibility(
            planning_for_feasibility, cand)
        canonical = H.canonical_candidate({n: node_map.get(n, (0.0, 0.0)) for n in ACTIVE_NODES},
                                          investment_year=year)
        energies = [v[1] for v in node_map.values()] or [0.0]
        out[label] = {
            'source': source,
            'candidate_canonical': canonical,
            'candidate_key': H.candidate_key(canonical),
            'I_x_eur_master_expression': i_master,
            'I_x_eur_p56a_transcription': i_oracle,
            'abs_diff_master_vs_transcription_eur': abs(i_oracle - i_master),
            'I_power_part_eur': parts['power'],
            'I_energy_part_eur': parts['energy'],
            'I_constant_part_eur': parts['constant'],
            'abs_diff_split_sum_vs_master_eur': abs(split_sum - i_master),
            'I_by_node_year': {k: v for k, v in by_node_year.items()
                               if v['power'] != 0.0 or v['energy'] != 0.0},
            'budget_eur': sed.params.budget,
            'budget_slack_B_minus_I_eur': sed.params.budget - i_master,
            'budget_slack_from_master_row_eur': rows['budget_slack_from_row_eur'],
            'budget_feasible': rows['budget_slack_from_row_eur'] >= 0.0,
            'max_energy_per_node_mwh': max(energies),
            'energy_le_max_capacity_per_node': max(energies) <= sed.params.max_capacity,
            'master_rows': rows,
            'master_budget_row_body_eur': pe.value(budget_row.body),
            'first_stage_feasible_production_check': feasible,
            'first_stage_reasons': reasons,
        }
    return out


# ======================================================================================
#  workbook diff (cell level, values as production reads them AND stored formulas)
# ======================================================================================
def _cells(data, data_only):
    import openpyxl
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        wb = openpyxl.load_workbook(path, data_only=data_only)
        out = {}
        for ws in wb.worksheets:
            cells = {}
            for row in ws.iter_rows():
                for c in row:
                    if c.value is not None:
                        v = c.value
                        cells[c.coordinate] = v if isinstance(v, (int, float, str, bool)) else str(v)
            out[ws.title] = cells
        return out
    finally:
        os.unlink(path)


def _cell_diff(old, new):
    report = {}
    for sheet in sorted(set(old) | set(new), key=str):
        a, b = old.get(sheet), new.get(sheet)
        if a is None or b is None:
            report[sheet] = {'present_in_old': a is not None, 'present_in_new': b is not None}
            continue
        diffs, ratios = [], []
        for coord in sorted(set(a) | set(b)):
            va, vb = a.get(coord), b.get(coord)
            if va != vb:
                d = {'cell': coord, 'old': va, 'new': vb}
                if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                        and not isinstance(va, bool) and va != 0):
                    d['ratio_new_over_old'] = vb / va
                    ratios.append(vb / va)
                diffs.append(d)
        numeric_same = sum(1 for coord, va in a.items()
                           if isinstance(va, (int, float)) and b.get(coord) == va)
        entry = {'n_cells_old': len(a), 'n_cells_new': len(b), 'n_differing_cells': len(diffs),
                 'n_numeric_cells_unchanged': numeric_same, 'differences': diffs}
        if ratios:
            entry['numeric_ratio_new_over_old'] = {'min': min(ratios), 'max': max(ratios),
                                                   'n': len(ratios)}
        report[sheet] = entry
    return report


def _production_level_ratios(sed_new, sed_old):
    out = {}
    for kind in ('power', 'energy'):
        per = {}
        ratios = []
        for m in sed_new.cost_investment[kind]:
            for year in sed_new.cost_investment[kind][m]:
                vn = sed_new.cost_investment[kind][m][year]
                vo = sed_old.cost_investment[kind][m][year]
                r = vn / vo if vo else None
                per[f'scenario{m}|{year}'] = {'new': vn, 'old': vo, 'ratio_new_over_old': r}
                if r is not None:
                    ratios.append(r)
        out[kind] = {'per_scenario_year': per,
                     'ratio_min': min(ratios) if ratios else None,
                     'ratio_max': max(ratios) if ratios else None}
    out['scenario_weights'] = {'new': list(sed_new.prob_market_scenarios),
                               'old': list(sed_old.prob_market_scenarios),
                               'identical': list(sed_new.prob_market_scenarios)
                               == list(sed_old.prob_market_scenarios)}
    return out


# ======================================================================================
#  frontier
# ======================================================================================
def _frontier(evals_by_file, ladder_labels):
    out = {}
    for n in ACTIVE_NODES:
        for h in DURATIONS_H:
            for year in (2025, 2030, 2035):
                key = f'node{n}|{h}h|{year}'
                entry = {}
                for file_label, evals in evals_by_file.items():
                    points = []
                    for e in LADDER_ENERGIES:
                        lab = ladder_labels[(n, h, year, e)]
                        r = evals[lab]
                        points.append({'label': lab, 'e_mwh': e, 's_mva': e / h,
                                       'I_eur': r['I_x_eur_master_expression'],
                                       'slack_eur': r['budget_slack_B_minus_I_eur'],
                                       'budget_feasible': r['budget_feasible']})
                    feas = [p['e_mwh'] for p in points if p['budget_feasible']]
                    entry[file_label] = {'largest_budget_feasible_e_mwh': max(feas) if feas else None,
                                         'all_ladder_points_feasible': len(feas) == len(points),
                                         'points': points}
                out[key] = entry
    return out


# ======================================================================================
#  salvage (reporting expression) -- located in source, no solve
# ======================================================================================
def _salvage_section(C, SED, sed_new, sed_old):
    sed_py = 'shared_energy_storage_data.py'
    srp_py = 'shared_resources_planning.py'
    h_py = 'p515_s44_campaign_harness.py'
    years = list(sed_new.years)
    idx = 0
    per_year = {}
    for y_inv, year in enumerate(years):
        ses = sed_new.shared_energy_storages[year][idx]
        age, remaining, frac = SED._get_remaining_calendar_life(sed_new, y_inv, ses)
        term = SED._get_terminal_discount_factor(sed_new)
        c_new = SED._get_expected_energy_investment_cost(sed_new, year)
        c_old = SED._get_expected_energy_investment_cost(sed_old, year)
        rec = sed_new.params.salvage_value.energy_recovery_fraction
        per_year[str(year)] = {
            'age_at_terminal_years': age, 'remaining_life_years': remaining,
            'remaining_life_fraction': frac, 't_cal_years': ses.t_cal,
            'terminal_discount_factor': term,
            'expected_energy_unit_cost_new_eur_per_mwh': c_new,
            'expected_energy_unit_cost_old_eur_per_mwh': c_old,
            'ratio_new_over_old': c_new / c_old if c_old else None,
            'salvage_eur_per_mwh_of_residual_energy_new': term * rec * c_new * frac,
            'salvage_eur_per_mwh_of_residual_energy_old': term * rec * c_old * frac,
            'note': ('salvage = terminal_discount * energy_recovery_fraction * expected energy unit '
                     'cost (COST FILE) * remaining_life_fraction * residual_energy, residual_energy = '
                     'floor*e_rated + (1-floor)*(e_available - soh_min*e_rated)/(1-soh_min) at the '
                     'terminal year; zero whenever remaining_life_fraction = 0'),
        }
    return {
        'salvage_params_in_force': vars(sed_new.params.salvage_value),
        'salvage_params_source': 'data/SRP1/SharedESS/SRP1_ESS_Params.json "salvage_value"',
        'expression_builder': C._func_span(SED._build_terminal_salvage_value_expression),
        'expression_attached_to_esso_model': [
            C._loc(sed_py, 'model.salvage_value = pe.Expression(expr=salvage_value)'),
            C._loc(sed_py, 'model.salvage_credit = pe.Expression(expr=-model.salvage_value)')],
        'esso_objective': C._loc(sed_py, 'model.objective = pe.Objective(', 2)
                          + ' (expr=model.feasibility_penalty only; salvage NOT in the ESSO objective)',
        'in_master_investment_cost': False,
        'master_investment_cost_expression': C._loc(
            sed_py, 'model.investment_cost = pe.Expression(expr=investment_cost)'),
        'reads_the_cost_file': C._func_span(SED._get_expected_energy_investment_cost)
                               + ' (expected ENERGY investment unit cost from the workbook)',
        'reporting_reads': {
            'get_salvage_value': C._loc(sed_py, 'def get_salvage_value(self, models):'),
            'recourse_components_salvage': C._loc(
                srp_py, "terminal_salvage_value = planning_problem.shared_ess_data.get_salvage_value(models['esso'])"),
            'net_recourse_definition': C._loc(
                srp_py, 'net_operational_recourse = gross_operational_cost - terminal_salvage_value'),
            'admm_cycle_recourse_is_net': C._loc(
                srp_py, "recourse = recourse_components['net_operational_recourse']", 2),
            'admm_objective_change_uses_net_recourse': C._loc(
                srp_py, 'objective_change_abs = abs(recourse - previous_recourse)'),
            'objective_change_is_diagnostic_only_cycle_convergence': C._loc(
                srp_py, 'cycle_convergence = boyd_all_pass and local_solves_ok'),
            'benders_outer_candidate_total_uses_net_recourse': C._loc(
                srp_py, 'candidate_total = investment_cost + operational_recourse'),
        },
        'campaign_harness_evaluation_record': {
            'Q_x_excluding_salvage': [
                C._loc(h_py, "'certified_cost': report.get('gross_operational_cost') if certified else None,")
                + ' certified_cost (= gross_operational_cost when certified)',
                C._loc(h_py, "'terminal_gross_operational_cost': report.get('gross_operational_cost'),")
                + ' terminal_gross_operational_cost',
                C._loc(h_py, "'recourse_components': rc,")
                + ' recourse_components.gross_operational_cost'],
            'salvage': [
                C._loc(h_py, "'recourse_components': rc,")
                + ' recourse_components.terminal_salvage_value',
                C._loc(h_py, "'terminal_net_operational_recourse': rc.get('net_operational_recourse'),")
                + ' terminal_net_operational_recourse (= gross - salvage)'],
            'objective_convention_string': C._loc(h_py, "'objective_convention': ('gross_operational_cost:"),
            'per_cycle_fields': C._loc(h_py, "'cycle', 'local_solves_ok', 'recourse', 'gross_operational_cost', 'terminal_salvage_value',")
                                + " (per-cycle 'recourse' is the NET recourse; 'terminal_salvage_value' carried per cycle)",
        },
        'per_investment_year': per_year,
    }


# ======================================================================================
#  write-scope evidence
# ======================================================================================
def _modified_since(t0):
    out = []
    for root, dirs, files in os.walk(REPO):
        if root == REPO and '.git' in dirs:
            dirs.remove('.git')
        for name in files:
            p = os.path.join(root, name)
            try:
                if os.path.getmtime(p) >= t0:
                    out.append(os.path.relpath(p, REPO))
            except OSError:
                continue
    return sorted(out)


def _classify_modified(paths):
    mine, campaign, other = [], [], []
    for p in paths:
        if p.startswith(OUT_REL + '/'):
            mine.append(p)
        elif p.startswith(CONCURRENT_CAMPAIGN_DIR_REL + '/'):
            campaign.append(p)
        else:
            other.append(p)
    return {'under_OUT_DIR': mine, 'under_concurrent_campaign_dir': campaign, 'elsewhere': other}


def _running_p5_processes():
    res = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    me = os.getpid()
    out = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != me and 'python' in parts[1] \
                and re.search(r'\bp5\d', parts[1]):
            out.append(line.strip()[:300])
    return out


def _write_manifest():
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    files = {}
    for name in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, name)
        if p == path or not os.path.isfile(p):
            continue
        with open(p, 'rb') as fh:
            data = fh.read()
        files[os.path.relpath(p, REPO)] = {'sha256': _sha256_bytes(data), 'bytes': len(data)}
    rc, head, _ = _git(['rev-parse', 'HEAD'])
    with open(path, 'w') as fh:
        json.dump({'stage': STAGE, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'git_HEAD': head.strip(), 'files': files}, fh, indent=1)
    print(f'wrote {path} ({len(files)} files)')


# ======================================================================================
#  main
# ======================================================================================
def main():
    os.chdir(REPO)
    results_path = os.path.join(OUT_DIR, RESULTS_NAME)
    if os.path.exists(results_path):
        raise RuntimeError(f'refusing to overwrite existing artifact {results_path}')
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    started = datetime.now(timezone.utc)
    porcelain_before = _git(['status', '--porcelain'])[1]
    lock_before = open(CAMPAIGN_LOCK).read() if os.path.exists(CAMPAIGN_LOCK) else None
    concurrency = {
        'reused_preflight_NOT_called': 'p515_s44_addendum26_confirmations._preflight_no_concurrent_harness '
                                       '(bypassed: Planner authorized this zero-solve build to run '
                                       'concurrently with the AA C* re-verification campaign)',
        'running_p5_python_processes_at_start': _running_p5_processes(),
        'campaign_lock_content_at_start_read_only': lock_before,
    }

    # Workbook bytes (never writes the on-disk file).
    new_bytes = open(XLSX_PATH, 'rb').read()
    rc, old_bytes, err = _git(['show', f'{OLD_REV}:{XLSX_REL}'], binary=True)
    if rc != 0:
        raise RuntimeError(f'git show {OLD_REV}:{XLSX_REL} failed: {err}')
    if _sha256_bytes(new_bytes) != NEW_SHA256:
        raise RuntimeError(f'on-disk workbook sha256 {_sha256_bytes(new_bytes)} != {NEW_SHA256}')
    if _sha256_bytes(old_bytes) != OLD_SHA256:
        raise RuntimeError(f'old workbook sha256 {_sha256_bytes(old_bytes)} != {OLD_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 A27 W2 investment cost').install()
    try:
        import pyomo.environ as pe
        from pyomo.repn import generate_standard_repn
        import p515_s44_addendum26_confirmations as C
        import p515_s44_campaign_harness as H
        import p56a_oracle as O
        import shared_resources_planning as srp
        import shared_energy_storage_data as SED

        spec = json.load(open(SPEC_PATH))
        spec_sha = _sha256_bytes(open(SPEC_PATH, 'rb').read())
        if not spec_sha.startswith('5feefd7b'):
            raise RuntimeError(f'spec sha256 {spec_sha} does not match its name')
        if spec['cost_file']['sha256'] != NEW_SHA256:
            raise RuntimeError('spec v15 cost_file.sha256 differs from the on-disk workbook')
        s44_sel = json.load(open(os.path.join(REPO, S44_SELECTION_SPEC_REL)))

        print('[W2] loading baseline (production read of data/SRP1/SRP1.json) ...', flush=True)
        planning = O.load_baseline()['planning']
        sed_new = planning.shared_ess_data
        if sed_new.params.budget != BUDGET_EXPECTED:
            raise RuntimeError(f'case-file budget {sed_new.params.budget} != {BUDGET_EXPECTED}')
        if sed_new.params.max_capacity != MAX_CAPACITY_EXPECTED:
            raise RuntimeError(f'case-file max_capacity {sed_new.params.max_capacity}')

        sed_old = deepcopy(sed_new)
        with tempfile.TemporaryDirectory(prefix='p515s45_w2_') as tmpdir:
            tmp_path = os.path.join(tmpdir, 'SRP1_ESS_old.xlsx')
            with open(tmp_path, 'wb') as fh:
                fh.write(old_bytes)
            try:
                SED._read_shared_energy_storage_data_from_file(sed_old, tmp_path)
            except SystemExit as exc:
                raise RuntimeError(f'production reader exited on the old workbook: {exc}')
        if _sha256_bytes(open(XLSX_PATH, 'rb').read()) != NEW_SHA256:
            raise RuntimeError('on-disk workbook changed during the run')

        candidates = (_reference_candidates(spec, s44_sel)
                      + _spec_points(spec['A0']['points'], 'A0', spec['A0']['investment_year'])
                      + _spec_points(spec['A1']['ladders_2025']['points'], 'A1 ladders_2025')
                      + _year_ladder())
        labels = [c[0] for c in candidates]
        dup = sorted({lab for lab in labels if labels.count(lab) > 1})
        if dup:
            raise RuntimeError(f'duplicate candidate labels: {dup}')
        if len(spec['A0']['points']) != 8 or len(spec['A1']['ladders_2025']['points']) != 30:
            raise RuntimeError('spec v15 A0/A1 point counts differ from 8/30')
        ladder_labels = {}
        for lab, year, nm, _ in candidates:
            m = re.fullmatch(r'n(\d)_(\d)h_e(\d)(?:_y(\d{4}))?', lab)
            if m:
                (n, s_e), = nm.items()
                h, e = int(m.group(2)), float(m.group(3))
                if abs(s_e[1] - e) > 0 or abs(s_e[0] - e / h) > 1e-15:
                    raise RuntimeError(f'ladder label {lab} inconsistent with {nm}')
                ladder_labels[(int(m.group(1)), h, year, e)] = lab
        if len(ladder_labels) != 90:
            raise RuntimeError(f'expected 90 ladder points (30 + 60), got {len(ladder_labels)}')

        print(f'[W2] evaluating {len(candidates)} candidates with the NEW and OLD cost files ...',
              flush=True)
        shim_old = types.SimpleNamespace(shared_ess_data=sed_old)
        ev_new = _evaluate_all(C, O, srp, sed_new, planning, candidates, pe,
                               generate_standard_repn, H, planning)
        ev_old = _evaluate_all(C, O, srp, sed_old, planning, candidates, pe,
                               generate_standard_repn, H, shim_old)

        per_candidate = {}
        for lab, year, nm, source in candidates:
            n_, o_ = ev_new[lab], ev_old[lab]
            per_candidate[lab] = {
                'source': source,
                'candidate_canonical': n_['candidate_canonical'],
                'candidate_key': n_['candidate_key'],
                'I_new_eur': n_['I_x_eur_master_expression'],
                'I_old_eur': o_['I_x_eur_master_expression'],
                'ratio_I_new_over_I_old': (n_['I_x_eur_master_expression'] / o_['I_x_eur_master_expression']
                                           if o_['I_x_eur_master_expression'] else None),
                'I_new_power_eur': n_['I_power_part_eur'], 'I_new_energy_eur': n_['I_energy_part_eur'],
                'I_old_power_eur': o_['I_power_part_eur'], 'I_old_energy_eur': o_['I_energy_part_eur'],
                'slack_new_eur': n_['budget_slack_B_minus_I_eur'],
                'slack_old_eur': o_['budget_slack_B_minus_I_eur'],
                'budget_feasible_new': n_['budget_feasible'],
                'budget_feasible_old': o_['budget_feasible'],
                'energy_le_5_mwh_per_node': n_['energy_le_max_capacity_per_node'],
                'max_capacity_rows_violated': n_['master_rows']['max_capacity_rows_violated'],
                'ratio_rows_violated': n_['master_rows']['ratio_rows_violated'],
                'first_stage_feasible_new_production_check': n_['first_stage_feasible_production_check'],
                'first_stage_reasons_new': n_['first_stage_reasons'],
                'abs_diff_master_vs_p56a_new_eur': n_['abs_diff_master_vs_transcription_eur'],
                'abs_diff_master_vs_p56a_old_eur': o_['abs_diff_master_vs_transcription_eur'],
                'abs_diff_split_sum_vs_master_new_eur': n_['abs_diff_split_sum_vs_master_eur'],
                'abs_diff_split_sum_vs_master_old_eur': o_['abs_diff_split_sum_vs_master_eur'],
            }
        max_xcheck = max(max(v['abs_diff_master_vs_p56a_new_eur'], v['abs_diff_master_vs_p56a_old_eur'])
                         for v in per_candidate.values())
        max_split = max(max(v['abs_diff_split_sum_vs_master_new_eur'],
                            v['abs_diff_split_sum_vs_master_old_eur']) for v in per_candidate.values())

        frontier = _frontier({'new': ev_new, 'old': ev_old}, ladder_labels)

        print('[W2] workbook diff ...', flush=True)
        diff_values = _cell_diff(_cells(old_bytes, True), _cells(new_bytes, True))
        diff_formulas = _cell_diff(_cells(old_bytes, False), _cells(new_bytes, False))
        prod_ratios = _production_level_ratios(sed_new, sed_old)
        claim = {
            'claim': 'energy costs x1.25, power costs unchanged',
            'production_level_power_ratio_range': [prod_ratios['power']['ratio_min'],
                                                   prod_ratios['power']['ratio_max']],
            'production_level_energy_ratio_range': [prod_ratios['energy']['ratio_min'],
                                                    prod_ratios['energy']['ratio_max']],
            'scenario_weights_identical': prod_ratios['scenario_weights']['identical'],
            'sheet_Investment_Cost_Power_cached_values_differing_cells':
                diff_values.get('Investment Cost, Power', {}).get('n_differing_cells'),
            'sheet_Investment_Cost_Energy_cached_values_differing_cells':
                diff_values.get('Investment Cost, Energy', {}).get('n_differing_cells'),
            'sheet_Investment_Cost_Energy_cached_value_ratio':
                diff_values.get('Investment Cost, Energy', {}).get('numeric_ratio_new_over_old'),
        }
        claim['power_unchanged'] = (claim['production_level_power_ratio_range'] == [1.0, 1.0])
        er = claim['production_level_energy_ratio_range']
        claim['energy_x1_25_within_1e-12'] = (er[0] is not None and abs(er[0] - 1.25) <= 1e-12
                                              and abs(er[1] - 1.25) <= 1e-12)

        salvage = _salvage_section(C, SED, sed_new, sed_old)
        unit_costs = {'new': C._expected_unit_costs(sed_new), 'old': C._expected_unit_costs(sed_old)}
        master_facts = {
            'budget_eur': sed_new.params.budget, 'max_capacity_mwh': sed_new.params.max_capacity,
            'min_energy_to_power_ratio': sed_new.params.min_energy_to_power_ratio,
            'max_energy_to_power_ratio': sed_new.params.max_energy_to_power_ratio,
            'discount_factor': sed_new.discount_factor,
            'investment_cost_expression': C._loc('shared_energy_storage_data.py',
                                                 'model.investment_cost = pe.Expression(expr=investment_cost)'),
            'budget_row': C._loc('shared_energy_storage_data.py', 'model.energy_storage_investment.add('),
            'max_capacity_row': C._loc('shared_energy_storage_data.py',
                                       'model.energy_storage_maximum_capacity.add('),
            'ratio_rows': C._loc('shared_energy_storage_data.py',
                                 'model.energy_storage_power_to_energy_factor.add(', 1),
            'loader': C._func_span(SED._load_candidate_solution_into_master_model),
            'old_file_reader': C._func_span(SED._read_shared_energy_storage_data_from_file)
                               + ' (temporary copy outside the repository, deleted after reading)',
            'split_method': 'pyomo.repn.generate_standard_repn(model.investment_cost.expr, '
                            'compute_values=True); terms grouped by parent Var es_s_investment '
                            '(power) / es_e_investment (energy); checked to sum to '
                            'pe.value(model.investment_cost)',
        }
    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    guard_record = {'permitted': [], 'counts': dict(guard.counts), 'verify_0_failures': failures}
    if failures:
        raise RuntimeError(f'solve-profile guard: {failures}')

    modified = _classify_modified(_modified_since(t0))
    porcelain_after = _git(['status', '--porcelain'])[1]
    lock_after = open(CAMPAIGN_LOCK).read() if os.path.exists(CAMPAIGN_LOCK) else None
    payload = {
        'stage': STAGE, 'authority': AUTHORITY,
        'spec_path': SPEC_REL, 'spec_sha256': spec_sha,
        'spec_master_constraints': spec['master_constraints'],
        'started_utc': started.isoformat(),
        'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
        'script_sha256': _sha256_bytes(open(os.path.abspath(__file__), 'rb').read()),
        'solve_profile_guard': guard_record,
        'concurrency': concurrency,
        'cost_files': {
            'new': {'path': XLSX_REL, 'sha256': _sha256_bytes(new_bytes), 'bytes': len(new_bytes),
                    'read_by': 'production case read (p56a_oracle.load_baseline), file as on disk'},
            'old': {'git_rev': f'{OLD_REV}:{XLSX_REL}', 'sha256': _sha256_bytes(old_bytes),
                    'bytes': len(old_bytes),
                    'read_by': master_facts['old_file_reader']},
        },
        'master_facts': master_facts,
        'expected_unit_costs_per_case_year': unit_costs,
        'max_abs_diff_master_vs_p56a_eur': max_xcheck,
        'max_abs_diff_power_plus_energy_vs_master_eur': max_split,
        'candidates': per_candidate,
        'budget_feasible_frontier': frontier,
        'workbook_diff': {
            'claim_check': claim,
            'production_level_unit_costs': prod_ratios,
            'cell_diff_cached_values_openpyxl_data_only': diff_values,
            'cell_diff_stored_formulas_openpyxl': diff_formulas,
        },
        'salvage': salvage,
        'write_scope': {
            'git_status_porcelain_before': porcelain_before.splitlines(),
            'git_status_porcelain_after': porcelain_after.splitlines(),
            'porcelain_identical': porcelain_before == porcelain_after,
            'files_modified_during_run': modified,
            'campaign_lock_unchanged': lock_before == lock_after,
            'note': ('git status collapses untracked directories; files_modified_during_run is a '
                     'walk of the repository (excluding .git) for mtime >= run start. The results '
                     'JSON itself and the launch log are written after/around this scan.'),
        },
    }
    with open(results_path, 'w') as fh:
        json.dump(C._jsonable(payload), fh, indent=1, default=str)

    print('[W2] guard counts', guard.counts, 'verify(0) failures', failures)
    print(f'[W2] max |master - p56a| = {max_xcheck:.3e} EUR; max |power+energy - master| = {max_split:.3e}')
    print('[W2] claim check:', json.dumps(claim, default=str))
    print(f"{'label':34s} {'I_new':>14s} {'I_old':>14s} {'ratio':>8s} {'slack_new':>14s} feas E<=5")
    for lab, v in per_candidate.items():
        ratio = f"{v['ratio_I_new_over_I_old']:.5f}" if v['ratio_I_new_over_I_old'] else '   -   '
        print(f"{lab:34s} {v['I_new_eur']:14.2f} {v['I_old_eur']:14.2f} {ratio:>8s} "
              f"{v['slack_new_eur']:14.2f} {str(v['budget_feasible_new']):5s} "
              f"{v['energy_le_5_mwh_per_node']}")
    for key, v in frontier.items():
        print(f"[W2] frontier {key}: new E_max={v['new']['largest_budget_feasible_e_mwh']} "
              f"old E_max={v['old']['largest_budget_feasible_e_mwh']}")
    print('[W2] files modified during run:', json.dumps(modified))
    print('[W2] porcelain identical before/after:', porcelain_before == porcelain_after)
    print('[W2] done')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        _write_manifest()
        sys.exit(0)
    sys.exit(main())
