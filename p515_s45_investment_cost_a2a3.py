"""P5.15 Addendum 27, task W16 -- I(x) for the A2 presence design and the A3 resolution
points, plus a first sigma_Q estimate from the EXISTING ladders.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27; Planner task W16;
data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json (`A2`, `A3`,
`master_constraints`, `cost_file`); STEP4_DFO_METHOD.md 4.3 (the sigma_Q definition).

ZERO SOLVES. A `SolveProfileGuard` with NO permitted call site is installed before any
production module is imported and stays installed for the whole run; `verify(0)` is
checked exactly.

ADDITIVE, NEVER IN PLACE. The frozen I(x) table
`data/SRP1/Results/P515S45/investment_cost/investment_cost_results.json`
(commit 9e623dd3, sha256 28152120...) covers x = 0, the reference plans and the
SINGLE-NODE ladders; it is NOT modified, NOT overwritten and NOT superseded. This run
writes a SECOND table under a new directory that records the predecessor's path, commit
and sha256, and re-checks that sha256 after the run. Where a candidate is already in the
predecessor (the lattice C*, and every A3 proposal that is a cache hit) the predecessor's
value is REUSED as authoritative; the recomputation is recorded beside it as a
cross-check only.

WHAT IS COVERED (Planner task W16):
  A) the A2 presence design -- the per-node best 2025 settings from a1a combined over the
     non-empty subsets of {5,7,9} of size >= 2, plus the lattice C*. The points are built
     by the PRODUCTION launcher path `p515_s45_a1_campaign.stage_points('a2')`, which
     applies the frozen selection rule (argmin F = I + Q within each node's a1a ladder;
     ties -> lower I(x), then lower label) to the COMMITTED a1a/a1b campaign_results.json.
  B) the lattice C* (1.0 MVA / 4.0 MWh at nodes 5, 7, 9, 2025) -- looked up in the
     predecessor BY CANDIDATE KEY and reused if present.
  C) the A3 resolution points at node 7 alone, 2025, on the 0.25 MVA / 0.5 MWh lattice
     with 2 <= E/P <= 4 and E <= 5 MWh: every half-MWh step E in {0.5, 1.5, 2.5, 3.5, 4.5}
     paired with EVERY lattice P in the enumeration domain (a systematic enumeration; each
     P is kept or dropped with its reason), PLUS the single-P-step neighbours (P +- 0.25)
     of the best node-7 2025 ladder point at fixed E. Every proposal is marked
     already-evaluated or not, by candidate key against the committed A0 / a1a / a1b
     records, so the campaign can cache-hit rather than re-run.

I(x) PATH -- the same production path as `p515_s45_investment_cost_recompute.py` (W2),
reused BY IMPORT (`_evaluate_all`, `_full_candidate`, `_split_investment_cost`):
  * I(x) = `model.investment_cost` of the production Benders master built by
    `shared_ess_data.build_master_problem()`, candidate loaded with production's
    `load_candidate_solution_into_master_model`, evaluated with `pe.value` (no solve);
  * cross-checked against `p56a_oracle.investment_cost` (independent transcription);
  * power / energy split by `pyomo.repn.generate_standard_repn`, checked to sum to
    `pe.value(model.investment_cost)`;
  * budget slack read from the master's OWN budget row; max-capacity and ratio rows read.
Only the CORRECTED (on-disk) cost file is evaluated here; BOTH workbook sha256s are
recorded (the corrected one from disk, the superseded one from git), as W2's table does.

SIGMA_Q (ESTIMATE, from existing data only -- no new evaluation): spec v15 A3 / STEP4 4.3
say "the largest non-monotone jump in Q along the ladder relative to the smooth trend is
the measured sigma_Q". Operative formula frozen here: for each ladder (node, duration,
investment year) already evaluated, Q is fitted against E by ordinary least squares on the
basis [1, E, E^2] (the lowest-order smooth model that admits the diminishing returns the
ladders show; the linear fit [1, E] is reported beside it), and
    sigma_Q(ladder) = max_i |Q_i - Qhat_i|
with the per-ladder fraction sigma_Q / mean(Q). This is an ESTIMATE from the 5-point
ladders, NOT the measurement A3 will make: 5 points and 3 fitted parameters leave 2
degrees of freedom, so the residual is a lower bound on the deviation a denser ladder
would expose, and the fit absorbs part of any real curvature.

CONCURRENCY: refuses to run while any other p5* python process is alive (read-only `ps`; same rule
as `p515_s44_addendum26_confirmations._preflight_no_concurrent_harness`, excluding this process and
its ancestors -- see `_preflight_no_concurrent_harness` below for why the shared helper cannot be
called directly from a shell whose own command line names this script).

Usage (attached, both streams captured by the caller):
    python p515_s45_investment_cost_a2a3.py > <OUT_DIR>/launch.log 2>&1
    python p515_s45_investment_cost_a2a3.py --manifest
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = ('P5.15 Addendum 27 W16 -- I(x) for the A2 presence design and the A3 resolution points, '
         'plus a first sigma_Q estimate from the existing ladders (zero solves)')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 (Phase A, A2-A3); Planner task W16',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json (A2, A3, master_constraints, cost_file)',
    'STEP4_DFO_METHOD.md 4.3 (sigma_Q definition; provisional 1.1e-4)',
    'p515_s45_investment_cost_recompute.py (W2): the I(x) production path reused by import',
    'p515_s45_a1_campaign.py (W15, fbb1296f): the frozen selection rules and the a2 point builder',
]

_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
SPEC_REL = os.path.join(_P45, 'frozen_s45_phaseA_spec_v15_5feefd7b.json')
OUT_REL = os.path.join(_P45, 'investment_cost_a2a3')
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'investment_cost_a2a3_results.json'
MANIFEST_NAME = 'manifest_sha256.json'

# The predecessor table -- read, pinned, re-checked; never written.
PREDECESSOR = {
    'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
    'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
    'commit': '9e623dd304837f2632964f47130d0860c1f016ca',
    'produced_by': 'p515_s45_investment_cost_recompute.py (task W2)',
    'covers': 'x = 0, the reference plans (paper plan, C*, lattice plan, lattice C*, 2 x C*, node7_empty) '
              'and the SINGLE-NODE ladders at 2025 / 2030 / 2035',
}

XLSX_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
XLSX_PATH = os.path.join(REPO, XLSX_REL)
NEW_SHA256 = 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'
OLD_REV = '2cada62b~1'
OLD_SHA256 = '1458147446e9b70190465f42cef761af9cfd8b209e91035d858e2e63941d1414'

BUDGET_EUR = 1.0e6              # spec v15 master_constraints.budget (REPORTED in Phase A)
MAX_ENERGY_MWH = 5.0            # spec v15 master_constraints.max_capacity (per node)
DURATION_MIN_H, DURATION_MAX_H = 2.0, 4.0   # spec v15 master_constraints.duration
LATTICE_P_STEP = 0.25           # MVA
LATTICE_E_STEP = 0.5            # MWh
ACTIVE_NODES = (5, 7, 9)
TOL = 1e-9

# --- A3 enumeration domain (stated, not implied) -------------------------------------
A3_NODE = 7                     # Planner task W16: node 7 alone, 2025
A3_YEAR = 2025
A3_HALF_STEP_ENERGIES = (0.5, 1.5, 2.5, 3.5, 4.5)   # the half-MWh steps between the ladder levels
A3_P_ENUMERATION = tuple(round(0.25 * k, 10) for k in range(1, 11))  # 0.25 .. 2.50 MVA
A3_P_DOMAIN_NOTE = ('P is enumerated over every multiple of 0.25 MVA from 0.25 to 2.50 -- the full lattice P range '
                    'spanned by the A0/A1 ladders (0.25 MVA at E = 0.5/1.0, 2.50 MVA at E = 5.0 / 2 h). Each (P, E) '
                    'is then kept or dropped against 2 <= E/P <= 4, E <= 5 MWh and the lattice, with its reason.')

PRIOR_RESULTS = (
    {'stage': 'a0', 'path': os.path.join(_P45, 'campaign_s45_a0_c7', 'campaign_results.json')},
    {'stage': 'a1a', 'path': os.path.join(_P45, 'campaign_s45_a1a', 'campaign_results.json')},
    {'stage': 'a1b', 'path': os.path.join(_P45, 'campaign_s45_a1b', 'campaign_results.json')},
)

SIGMA_Q_PROVISIONAL_FRACTION = 1.1e-4   # STEP4 1.4 / 4.3: the provisional resolution of the oracle
SIGMA_Q_FORMULA = (
    'For each ladder (stage, node, duration h, investment year) already evaluated, with points (E_i, Q_i), '
    'Q_i = certified_cost_gross_settlement_excluded (GROSS, settlement-excluded): fit Q by ordinary least '
    'squares (numpy.linalg.lstsq, rcond=None) on the design matrix X = [1, E, E^2] (primary; the linear basis '
    '[1, E] is reported beside it). residual_i = Q_i - Qhat_i; '
    'sigma_Q(ladder) = max_i |residual_i|; fraction = sigma_Q(ladder) / mean_i(Q_i). '
    'Campaign-level estimate = max and median of sigma_Q(ladder) over the ladders, with the same fractions. '
    'This is the operative reading of spec v15 A3 / STEP4 4.3 "the largest non-monotone jump in Q along the '
    'ladder relative to the smooth trend": the smooth trend is the quadratic least-squares fit and the jump is '
    'the deviation from it.')
SIGMA_Q_CAVEAT = (
    'ESTIMATE from the EXISTING 5-point ladders, not the A3 measurement. (i) 5 points and 3 fitted parameters '
    'leave 2 degrees of freedom, so the fit absorbs part of any real curvature and the residual is a LOWER bound '
    'on what a denser ladder would expose. (ii) The ladder spacing is 1 MWh; sigma_Q is meant to bound the '
    'oracle noise at the 0.5 MWh / 0.25 MVA lattice step, which no existing ladder resolves. (iii) The residual '
    'mixes oracle noise with model misspecification: a genuinely non-quadratic Q(E) inflates it.')

OBJECTIVE_CONVENTION = ('Q(x) = certified_cost_gross_settlement_excluded (GROSS operational cost, '
                        'settlement-excluded); F(x) = I(x) + Q(x); terminal_salvage_value and '
                        'net_operational_recourse are reported by the campaigns and are EXCLUDED from F.')
BUDGET_CONVENTION = (f'budget_slack_eur = B - I(x) at B = {BUDGET_EUR:g} EUR, REPORTED only; the budget is NOT '
                     'applied in Phase A (spec v15 master_constraints.budget).')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    with open(path, 'rb') as handle:
        return _sha256_bytes(handle.read())


def _git(args, binary=False):
    res = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=not binary)
    return res.returncode, res.stdout, res.stderr


def _ancestor_pids():
    """This process and every ancestor, by PPID chain (ps -o ppid=)."""
    chain, pid = [], os.getpid()
    while pid and pid not in chain:
        chain.append(pid)
        out = subprocess.run(['ps', '-o', 'ppid=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        pid = int(out) if out.isdigit() else 0
    return chain


def _preflight_no_concurrent_harness():
    """Read-only: refuse to run while another p5* harness process is alive.

    Same rule and same `ps` reading as `p515_s44_addendum26_confirmations._preflight_no_concurrent_harness`,
    which is NOT called directly here for one reason: its match is on the command string, and the shell
    that launches this script has THIS script's path in its own command line, so it refuses on its own
    launcher. This variant excludes exactly this process and its ancestors (by PPID chain), which are
    recorded; everything else is refused as before."""
    res = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    mine = set(_ancestor_pids())
    others = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in mine:
            continue
        if 'python' in cmd and re.search(r'\bp5\d', cmd):
            others.append(line.strip()[:300])
    if others:
        raise RuntimeError('refusing to run concurrently with another harness process: ' + ' | '.join(others))
    return {'checked_with': 'ps -axo pid=,command=',
            'rule': 'refuse while any other p5* python process is alive',
            'shared_helper': ('p515_s44_addendum26_confirmations._preflight_no_concurrent_harness (same rule; not '
                              'called directly because it matches the launching shell\'s own command line, which '
                              'contains this script\'s path)'),
            'self_and_ancestor_pids_excluded': sorted(mine),
            'other_p5_harness_processes': []}


def _git_state(rel):
    tracked = bool(_git(['ls-files', '--', rel])[1].strip())
    dirty = bool(_git(['status', '--porcelain', '--', rel])[1].strip())
    return {'git_tracked': tracked, 'git_clean': not dirty}


# ======================================================================================
#  (C) the A3 enumeration
# ======================================================================================
def _on_lattice(value, step):
    return abs(value / step - round(value / step)) < TOL


def _bound_checks(power, energy):
    """Every bound spec v15 master_constraints puts on a single-node point. Returns the reasons it
    fails (empty = admissible)."""
    reasons = []
    if power <= 0.0 or energy <= 0.0:
        reasons.append(f'P = {power:g}, E = {energy:g}: not both positive (production requires s == 0 <=> e == 0)')
    else:
        duration = energy / power
        if not (DURATION_MIN_H - TOL <= duration <= DURATION_MAX_H + TOL):
            reasons.append(f'duration E/P = {duration:.6g} h outside [{DURATION_MIN_H:g}, {DURATION_MAX_H:g}]')
    if energy > MAX_ENERGY_MWH + TOL:
        reasons.append(f'E = {energy:g} > {MAX_ENERGY_MWH:g} MWh (max_capacity)')
    if not _on_lattice(power, LATTICE_P_STEP):
        reasons.append(f'P = {power:g} not on the {LATTICE_P_STEP:g} MVA lattice')
    if not _on_lattice(energy, LATTICE_E_STEP):
        reasons.append(f'E = {energy:g} not on the {LATTICE_E_STEP:g} MWh lattice')
    return reasons


def _a3_enumeration(best_node7, evaluated_by_key, H):
    """Systematic enumeration of the A3 proposals. Returns (rows, summary).

    rows: one per (P, E) proposal considered, with its verdict and reason(s)."""
    rows = []
    p_star, e_star = best_node7['P_mva'], best_node7['E_mwh']

    def _row(family, power, energy, extra=None):
        reasons = _bound_checks(power, energy)
        key = None
        if power > 0.0 and energy > 0.0:
            key = H.candidate_key(H.canonical_candidate({n: ((power, energy) if n == A3_NODE else (0.0, 0.0))
                                                         for n in ACTIVE_NODES},
                                                        investment_year=A3_YEAR))
        already = evaluated_by_key.get(key) if key else None
        item = {
            'family': family,
            'label': f'res_n{A3_NODE}_p{power:g}_e{energy:g}_y{A3_YEAR}',
            'node': A3_NODE, 'P_mva': power, 'E_mwh': energy, 'investment_year': A3_YEAR,
            'duration_h': (energy / power) if power else None,
            'candidate_key': key,
            'admissible': not reasons,
            'dropped_because': reasons,
            'already_evaluated': bool(already),
            'already_evaluated_as': already,
            'verdict': ('dropped' if reasons else ('already_evaluated' if already else 'new')),
        }
        if extra:
            item.update(extra)
        return item

    for energy in A3_HALF_STEP_ENERGIES:
        p_lo, p_hi = energy / DURATION_MAX_H, energy / DURATION_MIN_H
        for power in A3_P_ENUMERATION:
            rows.append(_row(f'half_step_E{energy:g}', power, energy,
                             {'admissible_P_interval_mva': [p_lo, p_hi],
                              'interval_endpoints_on_lattice': [_on_lattice(p_lo, LATTICE_P_STEP),
                                                                _on_lattice(p_hi, LATTICE_P_STEP)]}))
    for sign, name in ((+1, 'P_plus'), (-1, 'P_minus')):
        rows.append(_row('P_step_neighbour_of_best_node7_2025', p_star + sign * LATTICE_P_STEP, e_star,
                         {'direction': name, 'from_label': best_node7['label'],
                          'from_P_mva': p_star, 'from_E_mwh': e_star}))
    summary = {
        'rule': ('Planner task W16: every half-MWh step E in {0.5, 1.5, 2.5, 3.5, 4.5} paired with each lattice P '
                 'satisfying 2 <= E/P <= 4 and E <= 5, PLUS the single-P-step neighbours (P +- 0.25) of the best '
                 'node-7 2025 ladder point at fixed E; node 7 alone, investment year 2025'),
        'P_enumeration_domain_mva': list(A3_P_ENUMERATION),
        'P_enumeration_domain_note': A3_P_DOMAIN_NOTE,
        'best_node7_2025_ladder_point': best_node7,
        'n_proposals_considered': len(rows),
        'n_new': sum(1 for r in rows if r['verdict'] == 'new'),
        'n_already_evaluated': sum(1 for r in rows if r['verdict'] == 'already_evaluated'),
        'n_dropped': sum(1 for r in rows if r['verdict'] == 'dropped'),
        'new_labels': [r['label'] for r in rows if r['verdict'] == 'new'],
        'already_evaluated_labels': [f"{r['label']} == {r['already_evaluated_as']}"
                                     for r in rows if r['verdict'] == 'already_evaluated'],
    }
    return rows, summary


# ======================================================================================
#  sigma_Q from the existing ladders (ESTIMATE; no new evaluation)
# ======================================================================================
def _ladders_from_records(prior_points):
    """Group the committed single-node certified points into (stage, node, duration, year) ladders."""
    groups = {}
    for point in prior_points:
        nodes = point['nodes'] or {}
        if point['status'] != 'certified' or len(nodes) != 1:
            continue
        node, (power, energy) = next(iter(nodes.items()))
        if not power:
            continue
        duration = energy / power
        key = f"{point['stage']}|n{node}|{duration:g}h|y{point['investment_year']}"
        groups.setdefault(key, []).append({
            'label': point['label'], 'candidate_key': point['candidate_key'],
            'E_mwh': energy, 'P_mva': power, 'Q_gross_eur': point['Q_gross_eur'],
            'I_x_eur': point['I_x_eur'], 'F_eur': point['F_eur'], 'bar_eur': point.get('bar_eur'),
        })
    for key in groups:
        groups[key].sort(key=lambda p: p['E_mwh'])
    return {k: v for k, v in groups.items() if len(v) >= 4}


def _ls_fit(energies, values, degree, np):
    design = np.vstack([np.power(np.asarray(energies, dtype=float), d) for d in range(degree + 1)]).T
    coefs, _res, rank, _sv = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)
    fitted = design @ coefs
    residuals = np.asarray(values, dtype=float) - fitted
    return {'basis': [f'E^{d}' for d in range(degree + 1)], 'coefficients': [float(c) for c in coefs],
            'rank': int(rank), 'n_points': len(energies), 'dof': len(energies) - (degree + 1),
            'fitted': [float(v) for v in fitted], 'residuals': [float(v) for v in residuals],
            'max_abs_residual_eur': float(np.max(np.abs(residuals))),
            'rms_residual_eur': float(np.sqrt(np.mean(residuals ** 2)))}


def _sigma_q_section(ladders, np):
    per_ladder = {}
    for key, points in sorted(ladders.items()):
        energies = [p['E_mwh'] for p in points]
        values = [p['Q_gross_eur'] for p in points]
        quad = _ls_fit(energies, values, 2, np)
        lin = _ls_fit(energies, values, 1, np)
        mean_q = float(np.mean(values))
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        second = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        bars = [p['bar_eur'] for p in points if p['bar_eur'] is not None]
        worst = int(np.argmax(np.abs(quad['residuals'])))
        per_ladder[key] = {
            'points': points,
            'E_mwh': energies,
            'Q_gross_eur': values,
            'mean_Q_eur': mean_q,
            'monotone_decreasing_in_E': all(d < 0 for d in diffs),
            'first_differences_eur': diffs,
            'second_differences_eur': second,
            'max_abs_second_difference_eur': float(np.max(np.abs(second))) if second else None,
            'quadratic_fit': quad,
            'linear_fit': lin,
            'sigma_Q_eur': quad['max_abs_residual_eur'],
            'sigma_Q_fraction_of_mean_Q': quad['max_abs_residual_eur'] / mean_q,
            'sigma_Q_at_label': points[worst]['label'],
            'sigma_Q_at_E_mwh': energies[worst],
            'bars_eur': bars,
            'max_bar_eur': max(bars) if bars else None,
            'mean_bar_eur': float(np.mean(bars)) if bars else None,
            'sigma_Q_over_max_bar': (quad['max_abs_residual_eur'] / max(bars)) if bars else None,
        }
    sigmas = [v['sigma_Q_eur'] for v in per_ladder.values()]
    fractions = [v['sigma_Q_fraction_of_mean_Q'] for v in per_ladder.values()]
    all_bars = [b for v in per_ladder.values() for b in v['bars_eur']]
    mean_q_all = float(np.mean([v['mean_Q_eur'] for v in per_ladder.values()]))
    return {
        'status': 'ESTIMATE from existing ladders (zero new evaluations)',
        'formula': SIGMA_Q_FORMULA,
        'caveat': SIGMA_Q_CAVEAT,
        'objective_convention': OBJECTIVE_CONVENTION,
        'n_ladders': len(per_ladder),
        'sigma_Q_max_eur': max(sigmas), 'sigma_Q_median_eur': float(np.median(sigmas)),
        'sigma_Q_min_eur': min(sigmas),
        'sigma_Q_max_fraction': max(fractions), 'sigma_Q_median_fraction': float(np.median(fractions)),
        'sigma_Q_min_fraction': min(fractions),
        'provisional_fraction': SIGMA_Q_PROVISIONAL_FRACTION,
        'provisional_eur_at_mean_Q': SIGMA_Q_PROVISIONAL_FRACTION * mean_q_all,
        'mean_Q_over_all_ladders_eur': mean_q_all,
        'ratio_estimate_max_over_provisional': max(fractions) / SIGMA_Q_PROVISIONAL_FRACTION,
        'ratio_estimate_median_over_provisional': float(np.median(fractions)) / SIGMA_Q_PROVISIONAL_FRACTION,
        'bars': {'n': len(all_bars), 'max_eur': max(all_bars) if all_bars else None,
                 'mean_eur': float(np.mean(all_bars)) if all_bars else None,
                 'min_eur': min(all_bars) if all_bars else None,
                 'definition': 'max over the last 10 cycles of |gross_operational_cost[k] - gross_operational_cost[k-1]| '
                               '(the campaign harness bar, fixed on the GROSS step by W5)',
                 'sigma_Q_max_over_max_bar': (max(sigmas) / max(all_bars)) if all_bars else None},
        'per_ladder': per_ladder,
    }


# ======================================================================================
#  manifest
# ======================================================================================
def _write_manifest():
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    files = {}
    for name in sorted(os.listdir(OUT_DIR)):
        candidate = os.path.join(OUT_DIR, name)
        if candidate == path or not os.path.isfile(candidate):
            continue
        with open(candidate, 'rb') as handle:
            data = handle.read()
        files[os.path.relpath(candidate, REPO)] = {'sha256': _sha256_bytes(data), 'bytes': len(data)}
    inputs = {PREDECESSOR['path']: {'sha256': _sha256_file(os.path.join(REPO, PREDECESSOR['path'])),
                                    'role': 'predecessor I(x) table (not modified)'}}
    for pin in PRIOR_RESULTS:
        inputs[pin['path']] = {'sha256': _sha256_file(os.path.join(REPO, pin['path'])),
                               'role': f'committed {pin["stage"]} campaign results (input)'}
    inputs[XLSX_REL] = {'sha256': _sha256_file(XLSX_PATH), 'role': 'cost file in force'}
    inputs[SPEC_REL] = {'sha256': _sha256_file(os.path.join(REPO, SPEC_REL)), 'role': 'frozen Phase A spec v15'}
    with open(path, 'w') as handle:
        json.dump({'stage': STAGE, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
                   'files': files, 'hash_inventory_of_inputs': inputs}, handle, indent=1)
    print(f'wrote {path} ({len(files)} files, {len(inputs)} pinned inputs)')


# ======================================================================================
#  main
# ======================================================================================
def main():
    os.chdir(REPO)
    results_path = os.path.join(OUT_DIR, RESULTS_NAME)
    if os.path.exists(results_path):
        raise RuntimeError(f'refusing to overwrite existing artifact {results_path}')
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.time()

    # --- pinned inputs, checked before anything is built -----------------------------
    pred_path = os.path.join(REPO, PREDECESSOR['path'])
    pred_sha_before = _sha256_file(pred_path)
    if pred_sha_before != PREDECESSOR['sha256']:
        raise RuntimeError(f'predecessor sha256 {pred_sha_before} != pinned {PREDECESSOR["sha256"]}')
    new_bytes = open(XLSX_PATH, 'rb').read()
    rc, old_bytes, err = _git(['show', f'{OLD_REV}:{XLSX_REL}'], binary=True)
    if rc != 0:
        raise RuntimeError(f'git show {OLD_REV}:{XLSX_REL} failed: {err}')
    if _sha256_bytes(new_bytes) != NEW_SHA256:
        raise RuntimeError(f'on-disk workbook sha256 {_sha256_bytes(new_bytes)} != {NEW_SHA256}')
    if _sha256_bytes(old_bytes) != OLD_SHA256:
        raise RuntimeError(f'superseded workbook sha256 {_sha256_bytes(old_bytes)} != {OLD_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 A27 W16 investment cost a2/a3').install()
    try:
        import numpy as np
        import pyomo.environ as pe
        from pyomo.repn import generate_standard_repn
        import p515_s44_addendum26_confirmations as C
        import p515_s44_campaign_harness as H
        import p515_s45_investment_cost_recompute as W2
        import p515_s45_a1_campaign as A1      # installs its own zero-permit guard on top of ours
        import p56a_oracle as O
        import shared_resources_planning as srp

        concurrency = _preflight_no_concurrent_harness()
        _log('[W16] concurrency preflight passed (no other p5* python process alive)')

        spec_sha = _sha256_file(os.path.join(REPO, SPEC_REL))
        if not spec_sha.startswith('5feefd7b'):
            raise RuntimeError(f'spec v15 sha256 {spec_sha} does not match its name')

        # --- (A) the A2 presence design, built by the production launcher path -------
        _log('[W16] building the A2 points through p515_s45_a1_campaign.stage_points("a2") ...')
        a2_points, a2_provenance = A1.stage_points('a2')

        # --- prior records (for the per-node bests, the cache marks and sigma_Q) -----
        prior_records, prior_evidence, prior_problems = A1.load_prior_records(list(PRIOR_RESULTS))
        if prior_problems:
            raise RuntimeError('prior results unusable: ' + '; '.join(prior_problems))
        # carry each record's bar (not part of A1's record schema) for the sigma_Q section
        bar_by_key = {}
        for pin in PRIOR_RESULTS:
            with open(os.path.join(REPO, pin['path'])) as handle:
                for point in (json.load(handle).get('points') or {}).values():
                    bar = (point.get('bar') or {}).get('value')
                    if point.get('candidate_key') and bar is not None:
                        bar_by_key[point['candidate_key']] = bar
        for record in prior_records:
            record['bar_eur'] = bar_by_key.get(record['candidate_key'])
        evaluated_by_key = {r['candidate_key']: f"{r['stage']}:{r['label']}" for r in prior_records
                            if r.get('candidate_key')}

        per_node_best = A1.select_best_setting_per_node([r for r in prior_records if r['stage'] == 'a1a'])
        best_node7 = per_node_best[A3_NODE]

        # --- (C) the A3 enumeration --------------------------------------------------
        a3_rows, a3_summary = _a3_enumeration(best_node7, evaluated_by_key, H)

        # --- the candidate list for I(x) ---------------------------------------------
        candidates = []
        for label, nodes, year in a2_points:
            nz = {n: v for n, v in nodes.items() if v != (0.0, 0.0)}
            candidates.append((label, year, nz,
                               'W16 (A) A2 presence design / lattice C*, built by '
                               'p515_s45_a1_campaign.stage_points("a2")'))
        for row in a3_rows:
            if row['verdict'] == 'dropped':
                continue
            candidates.append((row['label'], A3_YEAR, {A3_NODE: (row['P_mva'], row['E_mwh'])},
                               'W16 (C) A3 resolution point (node 7, 2025, 0.25 MVA / 0.5 MWh lattice)'))
        labels = [c[0] for c in candidates]
        dup = sorted({lab for lab in labels if labels.count(lab) > 1})
        if dup:
            raise RuntimeError(f'duplicate candidate labels: {dup}')

        _log(f'[W16] evaluating I(x) for {len(candidates)} candidates with the corrected cost file ...')
        planning = O.load_baseline()['planning']
        sed = planning.shared_ess_data
        if sed.params.budget != BUDGET_EUR:
            raise RuntimeError(f'case-file budget {sed.params.budget} != {BUDGET_EUR}')
        if sed.params.max_capacity != MAX_ENERGY_MWH:
            raise RuntimeError(f'case-file max_capacity {sed.params.max_capacity} != {MAX_ENERGY_MWH}')
        evals = W2._evaluate_all(C, O, srp, sed, planning, candidates, pe, generate_standard_repn, H, planning)

        # --- the predecessor, by candidate key: reuse, never recompute over ----------
        with open(pred_path) as handle:
            pred_candidates = json.load(handle)['candidates']
        pred_by_key = {}
        for name, entry in pred_candidates.items():
            pred_by_key.setdefault(entry['candidate_key'], []).append((name, entry))

        per_candidate, disagreements = {}, []
        for label, year, node_map, source in candidates:
            got = evals[label]
            key = got['candidate_key']
            hits = pred_by_key.get(key, [])
            reuse = None
            if hits:
                names = sorted(n for n, _e in hits)
                values = sorted({e['I_new_eur'] for _n, e in hits}, key=repr)
                entry = hits[0][1]
                reuse = {
                    'present_in_predecessor': True,
                    'predecessor_labels': names,
                    'predecessor_path': PREDECESSOR['path'], 'predecessor_sha256': PREDECESSOR['sha256'],
                    'I_new_eur_frozen_AUTHORITATIVE': values[0] if len(values) == 1 else None,
                    'predecessor_entries_agree': len(values) == 1,
                    'slack_new_eur_frozen': entry.get('slack_new_eur'),
                    'budget_feasible_new_frozen': entry.get('budget_feasible_new'),
                    'recomputed_here_crosscheck_eur': got['I_x_eur_master_expression'],
                    'abs_diff_recomputed_vs_frozen_eur': (abs(got['I_x_eur_master_expression'] - values[0])
                                                          if len(values) == 1 else None),
                }
                if reuse['abs_diff_recomputed_vs_frozen_eur'] is None or \
                        reuse['abs_diff_recomputed_vs_frozen_eur'] > 1e-6:
                    disagreements.append(f'{label}: recomputed {got["I_x_eur_master_expression"]} vs frozen {values}')
            per_candidate[label] = {
                'source': source,
                'candidate_canonical': got['candidate_canonical'],
                'candidate_key': key,
                'investment_year': year,
                'nodes_nonzero': {str(n): list(v) for n, v in node_map.items()},
                'I_new_eur': (reuse['I_new_eur_frozen_AUTHORITATIVE'] if reuse
                              else got['I_x_eur_master_expression']),
                'I_new_eur_provenance': ('REUSED from the predecessor table (candidate already covered there); '
                                         'the recomputation here agrees and is recorded as a cross-check'
                                         if reuse else 'computed here (production master model.investment_cost)'),
                'I_new_power_eur': got['I_power_part_eur'],
                'I_new_energy_eur': got['I_energy_part_eur'],
                'I_new_constant_eur': got['I_constant_part_eur'],
                'I_by_node_year_eur': got['I_by_node_year'],
                'abs_diff_split_sum_vs_master_eur': got['abs_diff_split_sum_vs_master_eur'],
                'I_new_eur_p56a_transcription': got['I_x_eur_p56a_transcription'],
                'abs_diff_master_vs_p56a_eur': got['abs_diff_master_vs_transcription_eur'],
                'budget_eur': got['budget_eur'],
                'budget_slack_B_minus_I_eur': got['budget_slack_B_minus_I_eur'],
                'budget_slack_from_master_row_eur': got['budget_slack_from_master_row_eur'],
                'budget_feasible': got['budget_feasible'],
                'max_energy_per_node_mwh': got['max_energy_per_node_mwh'],
                'energy_le_5_mwh_per_node': got['energy_le_max_capacity_per_node'],
                'max_capacity_rows_violated': got['master_rows']['max_capacity_rows_violated'],
                'ratio_rows_violated': got['master_rows']['ratio_rows_violated'],
                'first_stage_feasible_production_check': got['first_stage_feasible_production_check'],
                'first_stage_reasons': got['first_stage_reasons'],
                'already_evaluated_as': evaluated_by_key.get(key),
                'predecessor_reuse': reuse,
                'cost_file_new_sha256': NEW_SHA256,
                'cost_file_superseded_sha256': OLD_SHA256,
                'budget_convention': BUDGET_CONVENTION,
            }
        if disagreements:
            raise RuntimeError('recomputation disagrees with the predecessor table: ' + '; '.join(disagreements))
        max_xcheck = max(v['abs_diff_master_vs_p56a_eur'] for v in per_candidate.values())
        max_split = max(v['abs_diff_split_sum_vs_master_eur'] for v in per_candidate.values())

        # --- lattice step costs (from this table; the method doc's figures predate the fix)
        step_costs = {
            'note': ('lattice step cost of I(x) at 2025, from the unit costs in force (derived from the linear '
                     'power/energy split of the candidates evaluated here)'),
            'per_mva_2025_eur': None, 'per_mwh_2025_eur': None,
        }
        probe = next((v for v in per_candidate.values()
                      if v['investment_year'] == 2025 and len(v['nodes_nonzero']) == 1), None)
        if probe is not None:
            (s_val, e_val), = [tuple(v) for v in probe['nodes_nonzero'].values()]
            step_costs['derived_from'] = probe['candidate_key']
            step_costs['per_mva_2025_eur'] = probe['I_new_power_eur'] / s_val
            step_costs['per_mwh_2025_eur'] = probe['I_new_energy_eur'] / e_val
            step_costs['cost_of_one_lattice_P_step_0.25_mva_eur'] = step_costs['per_mva_2025_eur'] * LATTICE_P_STEP
            step_costs['cost_of_one_lattice_E_step_0.5_mwh_eur'] = step_costs['per_mwh_2025_eur'] * LATTICE_E_STEP

        # --- sigma_Q ------------------------------------------------------------------
        _log('[W16] sigma_Q estimate from the existing ladders ...')
        ladders = _ladders_from_records(prior_records)
        sigma_q = _sigma_q_section(ladders, np)
        sigma_q['comparison_with_lattice_step_cost'] = {
            'cost_of_one_lattice_P_step_0.25_mva_eur': step_costs.get('cost_of_one_lattice_P_step_0.25_mva_eur'),
            'cost_of_one_lattice_E_step_0.5_mwh_eur': step_costs.get('cost_of_one_lattice_E_step_0.5_mwh_eur'),
            'sigma_Q_max_over_P_step_cost': (sigma_q['sigma_Q_max_eur']
                                             / step_costs['cost_of_one_lattice_P_step_0.25_mva_eur']
                                             if step_costs.get('cost_of_one_lattice_P_step_0.25_mva_eur') else None),
            'sigma_Q_max_over_E_step_cost': (sigma_q['sigma_Q_max_eur']
                                             / step_costs['cost_of_one_lattice_E_step_0.5_mwh_eur']
                                             if step_costs.get('cost_of_one_lattice_E_step_0.5_mwh_eur') else None),
            'why': ('STEP4 5.2: if sigma_Q exceeds one lattice step\'s investment cost, the reported optimum is the '
                    'incumbent at the coarsest poll size whose step cost exceeds 2 sigma_Q'),
        }
    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(f'solve-profile guard: {failures}')
    # p515_s45_a1_campaign installs its own zero-permit guard at import, wrapping ours: it would see
    # (and block) a solve before ours does, so it is verified at exactly 0 too.
    nested_failures = A1.PARENT_GUARD.verify(expected_solves=0)
    if nested_failures:
        raise RuntimeError(f'nested (A1 campaign) solve-profile guard: {nested_failures}')

    pred_sha_after = _sha256_file(pred_path)
    if pred_sha_after != PREDECESSOR['sha256']:
        raise RuntimeError(f'predecessor table changed during the run: {pred_sha_after}')
    modified = W2._classify_modified(W2._modified_since(t0))
    modified_elsewhere = [p for p in modified['elsewhere'] if not p.startswith(OUT_REL + '/')]

    payload = {
        'stage': STAGE, 'authority': AUTHORITY,
        'started_utc': started.isoformat(), 'wall_clock_s': time.time() - t0,
        'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
        'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'additive_not_in_place': {
            'predecessor': dict(PREDECESSOR, sha256_before_run=pred_sha_before,
                                sha256_after_run=pred_sha_after,
                                unchanged=pred_sha_before == pred_sha_after,
                                git_state=_git_state(PREDECESSOR['path'])),
            'this_table': {'path': os.path.join(OUT_REL, RESULTS_NAME),
                           'relation': 'SECOND, ADDITIVE I(x) table; the predecessor is neither modified nor '
                                       'superseded. Candidates present in the predecessor keep the predecessor\'s '
                                       'value (reused, authoritative); the recomputation is a cross-check.'},
        },
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_0_failures': failures,
                                'nested_a1_campaign_guard_counts': dict(A1.PARENT_GUARD.counts),
                                'nested_a1_campaign_verify_0_failures': nested_failures,
                                'note': ('p515_s45_a1_campaign installs a second zero-permit guard at import; it '
                                         'wraps this one, so any solve would raise from the outer guard first. '
                                         'Both are verified at exactly 0.')},
        'concurrency': concurrency,
        'spec': {'path': SPEC_REL, 'sha256': spec_sha},
        'cost_files': {
            'in_force_corrected': {'path': XLSX_REL, 'sha256': _sha256_bytes(new_bytes), 'bytes': len(new_bytes),
                                   'read_by': 'production case read (p56a_oracle.load_baseline), file as on disk'},
            'superseded': {'git_rev': f'{OLD_REV}:{XLSX_REL}', 'sha256': _sha256_bytes(old_bytes),
                           'bytes': len(old_bytes),
                           'note': 'recorded for provenance only; NOT evaluated here (W2\'s table holds I_old)'},
        },
        'conventions': {'objective': OBJECTIVE_CONVENTION, 'budget': BUDGET_CONVENTION},
        'master_facts': {
            'budget_eur': BUDGET_EUR, 'max_capacity_mwh': MAX_ENERGY_MWH,
            'duration_bounds_h': [DURATION_MIN_H, DURATION_MAX_H],
            'lattice_P_step_mva': LATTICE_P_STEP, 'lattice_E_step_mwh': LATTICE_E_STEP,
            'I_x_path': ('model.investment_cost of the production Benders master '
                         '(shared_ess_data.build_master_problem), candidate loaded with '
                         'load_candidate_solution_into_master_model, evaluated with pe.value; reused by import '
                         'from p515_s45_investment_cost_recompute._evaluate_all (task W2)'),
            'cross_check': 'p56a_oracle.investment_cost (independent transcription)',
            'split_method': ('pyomo.repn.generate_standard_repn(model.investment_cost.expr, compute_values=True); '
                             'terms grouped by parent Var es_s_investment (power) / es_e_investment (energy)'),
            'max_abs_diff_master_vs_p56a_eur': max_xcheck,
            'max_abs_diff_power_plus_energy_vs_master_eur': max_split,
        },
        'lattice_step_costs': step_costs,
        'prior_results_pinned': prior_evidence,
        'A2_presence_design': {
            'provenance': a2_provenance,
            'best_setting_per_node': {str(n): per_node_best[n] for n in ACTIVE_NODES},
            'points': [{'label': lab, 'investment_year': yr,
                        'nodes': {str(n): list(v) for n, v in nodes.items()}} for lab, nodes, yr in a2_points],
        },
        'A3_resolution_points': {'summary': a3_summary, 'proposals': a3_rows},
        'candidates': per_candidate,
        'sigma_Q_estimate': sigma_q,
        'write_scope': {'files_modified_during_run': modified,
                        'modified_outside_OUT_DIR': modified_elsewhere,
                        'note': 'walk of the repository (excluding .git) for mtime >= run start; the results JSON '
                                'and the launch log are written after/around this scan'},
    }
    with open(results_path, 'w') as handle:
        json.dump(C._jsonable(payload), handle, indent=1, default=str)

    # ---------------------------------------------------------------- console summary
    print()
    print('[W16] guard counts', guard.counts, 'verify(0) failures', failures)
    print(f'[W16] max |master - p56a| = {max_xcheck:.3e} EUR; max |power+energy - master| = {max_split:.3e} EUR')
    print('[W16] best 2025 setting per node (argmin F = I + Q within that node\'s a1a ladder):')
    for node in ACTIVE_NODES:
        best = per_node_best[node]
        print(f"        node {node}: {best['label']:12s} P={best['P_mva']:g} MVA E={best['E_mwh']:g} MWh "
              f"({best['duration_h']:g} h)  I={best['I_x_eur']:.2f}  Q={best['Q_gross_eur']:.2f}  "
              f"F={best['F_eur']:.2f}")
    print(f"{'label':34s} {'I_new_eur':>15s} {'slack@1e6':>15s} feas E<=5 src  key")
    for lab, v in per_candidate.items():
        src = 'reuse' if v['predecessor_reuse'] else 'new  '
        print(f"{lab:34s} {v['I_new_eur']:15.2f} {v['budget_slack_B_minus_I_eur']:15.2f} "
              f"{str(v['budget_feasible']):5s} {str(v['energy_le_5_mwh_per_node']):5s} {src} {v['candidate_key'][:16]}")
    print(f"[W16] A3 enumeration: {a3_summary['n_proposals_considered']} proposals -> "
          f"{a3_summary['n_new']} new, {a3_summary['n_already_evaluated']} already evaluated, "
          f"{a3_summary['n_dropped']} dropped")
    for row in a3_rows:
        if row['verdict'] != 'dropped':
            print(f"        {row['verdict']:18s} {row['label']:28s} dur={row['duration_h']}"
                  f"{'  == ' + row['already_evaluated_as'] if row['already_evaluated_as'] else ''}")
    print(f"[W16] sigma_Q ESTIMATE over {sigma_q['n_ladders']} ladders: max {sigma_q['sigma_Q_max_eur']:.1f} EUR "
          f"({sigma_q['sigma_Q_max_fraction']:.3e} of Q), median {sigma_q['sigma_Q_median_eur']:.1f} EUR "
          f"({sigma_q['sigma_Q_median_fraction']:.3e}); provisional {SIGMA_Q_PROVISIONAL_FRACTION:.1e} "
          f"= {sigma_q['provisional_eur_at_mean_Q']:.1f} EUR; max bar {sigma_q['bars']['max_eur']:.1f} EUR")
    for key, v in sorted(sigma_q['per_ladder'].items()):
        print(f"        {key:22s} sigma_Q={v['sigma_Q_eur']:10.1f} ({v['sigma_Q_fraction_of_mean_Q']:.2e}) "
              f"at {v['sigma_Q_at_label']:16s} max_bar={v['max_bar_eur']:.1f} "
              f"monotone={v['monotone_decreasing_in_E']}")
    print('[W16] predecessor unchanged:', pred_sha_before == pred_sha_after, pred_sha_after)
    print('[W16] files modified outside OUT_DIR:', json.dumps(modified_elsewhere))
    print('[W16] done')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        _write_manifest()
        sys.exit(0)
    sys.exit(main())
