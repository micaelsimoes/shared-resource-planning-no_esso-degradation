"""P5.15 Addendum 40 ruling 1, task W73 -- alpha-row curtailment RECLASSIFICATION on the primal indicator. ZERO SOLVES.

Sibling of `p515_s53_alpha_row_recompute.py` (W72), not an edit of it: the W72 script's sha256 is pinned in the committed
`recompute_w72_manifest_sha256.json` and recorded inside `alpha_row_recompute.json`; editing it in place would make the
committed W72 evidence fail its own re-verification and blur which code produced which committed number. This script
imports nothing from it and re-reads the committed per-eval captures directly.

Planner rulings implemented (W73, revising W72's):
  1. a third class `at_availability`, not `capability_bound`, for curtailment entries that sit at availability within
     the barrier offset;
  2. classification on the PRIMAL c-threshold, c <= k x TOL_MW, identically for TSO and DSO -- the dual signature
     (kappa / phi) is NOT used as a discriminator; k is derived from the barrier offset and the solver tolerances
     (FORMULAS['k_derivation']) and the sensitivity at k = 1, 2, 3, 5 is reported for every cell;
  3. the headline re-reported per cell with the corrected classes;
  4. the barrier finding recorded as a methodological caveat (FORMULAS['barrier_caveat']), withdrawing "dual-established".
Also (W73 "also answer"): why the unit cell has no TSO entries above tolerance and ~0.35 MWh TSO curtailment against
~5 MWh in the x = 0 cells, from the committed captures plus the TSO IPOPT logs of each cell (the logs are NOT in any
campaign manifest; they are hashed here at read time and that limitation is recorded in the output).

Read-only: nothing is built or solved. A `SolveProfileGuard(permitted=())` is installed BEFORE any Pyomo / production /
harness import and verified at exactly 0 at the end; the verification is written into the output. No committed artifact
and no frozen spec is modified; response_terminal.json files are untouched (the corrected classes exist only in this
output). Outputs are write-once.

INSTANCE: campaign s53_alpha_row_v25 (spec 70965374, frozen spec v25 407a4b33) -- the 2 x 2 pilot instance, BASELINE
flexibility price (NOT the m = 2 variant), x = 0 (candidate 8435c718...) at alpha in {0, 0.1, 0.25, 0.5, 1.0} and the
node-7 unit (0.25 MVA / 1.0 MWh, 2025) at alpha = 0.5; candidate_key and eval_key recorded per cell.

OBJECTIVE / ENERGY CONVENTION: energies are MWh per representative day x admm_block_weight x scenario probability (the
Q weighting of W72 FORMULAS['curtailment_totals'] / ['entry_mwh_eur']); no objective value is reported here.

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber; one at a time):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_reclass.py --run \
      > data/SRP1/Results/P515S53/alpha_row/reclass_w73/reclass_w73_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_reclass.py --manifest
  (--run writes OUT_DIR/alpha_row_reclass.json, refusing if it exists; --manifest writes
   OUT_DIR/reclass_w73_manifest_sha256.json over the script, the output, the launch log and every input read, refusing if
   it exists, after re-checking the output's recorded input hashes.)
"""

import ast
import glob
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W73 alpha-row reclassification (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402  -- decoder only: load_response_terminal

SCRIPT_REL = os.path.basename(os.path.abspath(__file__))
ALPHA_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'alpha_row')
ROOT = os.path.join(ALPHA_ROOT, 'campaign_s53_alpha_row_v25')
OUT_DIR = os.path.join(ALPHA_ROOT, 'reclass_w73')
OUT_JSON = os.path.join(OUT_DIR, 'alpha_row_reclass.json')
OUT_LOG = os.path.join(OUT_DIR, 'reclass_w73_launch.log')
OUT_MANIFEST = os.path.join(OUT_DIR, 'reclass_w73_manifest_sha256.json')

CELLS = {'x0_a0p00': ('b123cd978794d690_x0_a0p00', 2), 'x0_a0p10': ('7516903c91153a29_x0_a0p10', 3),
         'x0_a0p25': ('62b46280a65f7744_x0_a0p25', 3), 'x0_a0p50': ('7d53b6f21b686a44_x0_a0p50', 1),
         'x0_a1p00': ('1bb2d63a07273887_x0_a1p00', 2), 'n7_4h_e1_a0p50': ('711fce9aa74d6878_n7_4h_e1_a0p50', 1)}
EVAL_FILES = ('evaluation_record.json', 'response_terminal.json')
UNIT_LABEL = 'n7_4h_e1_a0p50'

ROW_ANALYSIS_MANIFEST = (os.path.join(ALPHA_ROOT, 'row_analysis_manifest_sha256.json'),
                         '673b1ed0a94a1e144305e449088b2d4b70d557981711927d97444b580c300c25')   # commit 9e88d7bd
W72_MANIFEST = (os.path.join(ALPHA_ROOT, 'recompute_w72', 'recompute_w72_manifest_sha256.json'),
                '66947369a18fa1960404b860598c693e89cecef779284193134787ac6eed01ee')           # commit 59fc409f
W72_OUTPUT = os.path.join(ALPHA_ROOT, 'recompute_w72', 'alpha_row_recompute.json')
W72_SCRIPT = 'p515_s53_alpha_row_recompute.py'
CURTAILMENT_AUDIT = 'p515_s53_curtailment_audit.py'          # verified against the hash W72 recorded for it
# Production sources read for constants / the pg bound (hashed and recorded; must be git-clean at HEAD).
PRODUCTION_SOURCES = ('definitions.py', 'model_construction_helpers.py')
# Case files whose IPOPT options enter the k derivation (hashed and recorded; must be git-clean at HEAD).
CASE_PARAMS = {'TSO': os.path.join('data', 'SRP1', 'case9', 'case9_params.json'),
               'DSO5': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_params.json'),
               'DSO7': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_params.json'),
               'DSO9': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_params.json')}
# TSO IPOPT logs of each cell's run directory (NOT in any campaign manifest; hashed at read time).
TSO_LOG_GLOB = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                            'p515s44_s53_alpha_row_v25_{key16}_run', 'logs', 'optim_log_case9_*.log')

K_SET = (1, 2, 3, 5)
K_CHOSEN = 2
IPOPT_BARRIER_TOL_FACTOR_DEFAULT = 10.0     # IPOPT default (not set in any case file; not echoed as user-set in logs)

FORMULAS = {
    'weights': 'w_b = admm_block_weight of the block (response_terminal curtailment_by_block); omega = scenario '
               'probability of the entry / scenario; energies mwh_w = sum w_b omega c_mw (MW x 1 h periods).',
    'TOL_MW': 'TOL_MW = TOL_FACTOR x baseMVA, TOL_FACTOR = EQUALITY_TOLERANCE = 1e-5 (response_terminal constants; '
              'definitions.py). Read per block from curtailment_by_block tol_mw and asserted identical (1e-3 MW) for '
              'every TSO and DSO block. An entry exists iff c_mw > TOL_MW (harness capture rule).',
    'c': 'c_mw = (pg_avail - pg) x baseMVA per generator-hour-scenario (harness). NOTE: the model bound is pg <= '
         'pg_avail + EQUALITY_TOLERANCE (model_construction_helpers.pg_bounds), so the pg-bound slack is c + TOL_MW, '
         'not c; the S-capability row pg^2 + qg^2 <= sg_avail^2 limits pg to pg_avail exactly when qg = 0 or the power '
         'factor is fixed.',
    'at_availability': 'W73 ruling 2: an entry is at_availability iff c_mw <= k x TOL_MW, applied identically to TSO and '
                       'DSO entries and taking precedence over the recorded slack class. The sub-tolerance residue '
                       '(E_plus minus the energy of all entries, i.e. generator-hours with 0 < c <= TOL_MW) is '
                       'at_availability by the same definition and is reported separately.',
    'capability_bound': 'entry with c_mw > k x TOL_MW and recorded sg_capability_slack_pu2 <= CAP_SLACK_TOL = 1e-6 '
                        'p.u.^2 (the recorded harness rule, unchanged).',
    'below_capability': 'entry with c_mw > k x TOL_MW and recorded sg_capability_slack_pu2 > CAP_SLACK_TOL (recorded '
                        'class "interior"): curtailed below the S-capability limit, classified on the primal '
                        'indicator.',
    'identity': 'E_plus = at_availability(entries) + sub-tolerance residue + capability_bound + below_capability, per '
                'agent; the absolute residual of this identity is recorded per cell and k.',
    'row18_condition': 'DSO entry, alpha > 0, d_p_mw <= D_TOL_MW (1e-3, p515_s53_curtailment_audit) and '
                       'price_scenario_eur_mwh < alpha * row18_premium_eur_mwh (the launcher\'s definition; W72 '
                       'FORMULAS row18_condition, unchanged). share_below_capability = row18-condition energy (count) '
                       'within below_capability / below_capability energy (count); share_all_remaining = the same over '
                       'below_capability + capability_bound.',
    'k_derivation': (
        'Barrier offset of an UNCURTAILED unit at availability, from the model and the solver tolerances. (1) Geometry: '
        'two barrier terms act on c -- the pg upper bound at c = -T (pg <= pg_avail + T, T = EQUALITY_TOLERANCE = 1e-5 '
        'p.u., so TOL_MW = T x baseMVA) and the S-capability row at c = 0 (qg = 0 or fixed power factor; with qg_avail '
        '> 0 and free qg the row is slack at c = 0 and only the bound acts, which gives a SMALLER offset, so the '
        'two-term case is the worst case). (2) Barrier complementarity: every pair has z x s = mu (one constant per '
        'solve, scaled). The pg-directional barrier forces are mu / (c + T) from the bound and mu / c from the row '
        '(d/dpg of -mu ln(sg_avail^2 - pg^2 - qg^2) = 2 pg mu / s_cap with s_cap = 2 pg c to first order, also with '
        'qg = tan(phi) pg). (3) Stationarity in pg with zero own cost: lambda = mu/(c + T) + mu/c, lambda the scaled '
        'marginal value of energy at the unit. With r = c / T and rho = lambda T / mu: 1/(r + 1) + 1/r = rho, i.e. '
        'r(rho) = ((2 - rho) + sqrt(rho^2 + 4)) / (2 rho), decreasing in rho. (4) Solver tolerance: IPOPT terminates '
        'only when the scaled complementarity is <= tol (s_c = 1 unless the mean |multiplier| exceeds s_max = 100), so '
        'mu <= tol; tol = 1e-5 = T in all four case files (TSO case9_params.json, DSO case33_*_params.json; also '
        'echoed in the TSO logs and recorded as ipopt_options_in_force_dso). Hence rho >= lambda: the offset of an '
        'uncurtailed unit is r <= r(lambda) at every admissible terminal barrier parameter. (5) Choice: k = '
        'ceil(r(1)) = ceil((1 + sqrt 5) / 2) = ceil(1.618) = 2 -- the band c <= 2 TOL_MW contains the barrier offset '
        'of every uncurtailed unit whose scaled marginal value is >= 1, at the loosest barrier parameter IPOPT\'s '
        'termination admits. Equivalently c <= k TOL_MW holds for every uncurtailed unit with rho >= 1/k + 1/(k + 1): '
        'k = 1 -> 1.5, k = 2 -> 0.833, k = 3 -> 0.583, k = 5 -> 0.367. PREMISE AND LIMIT: the tolerances fix mu, not '
        'lambda; the premise "scaled marginal value >= 1" is a scale convention (IPOPT gradient-based scaling '
        'normalises the largest objective gradient to <= 100), not a guarantee: a unit at availability in an hour '
        'whose scaled marginal value is below 0.833 (at mu = tol) can sit above 2 TOL_MW and be counted curtailed. The '
        'k = 3 and k = 5 rows bound that effect. The derivation does not use the observed c = 1.30 TOL_MW, the duals '
        'of any entry, or any classification count.'),
    'geometry_consistency_check': 'NOT a discriminator (ruling 2). For each TSO entry, r = c / TOL_MW; the barrier '
                                  'geometry of k_derivation predicts the capability-row share of the bus marginal value '
                                  'kappa_pred = (1/r) / (1/r + 1/(r + 1)) = (r + 1) / (2 r + 1); compared with the W72 '
                                  'kappa = 2 (pg_mw / B) |sg_capability_dual_raw| / |lmp_bus_dual_raw| and phi = kappa '
                                  '+ |pg_zU_raw| / |lmp_bus_dual_raw|; also the two barrier products |cap dual| x '
                                  'cap slack and |pg_zU_raw| x (c_mw + TOL_MW) / B, which the geometry says are equal.',
    'capability_margin_sensitivity': 'REPORTED BESIDE, NOT SUBSTITUTED. The recorded capability test (slack of the '
                                     'SQUARED row <= 1e-6 p.u.^2) is size-dependent in MVA: slack ~= 2 sg_avail '
                                     '(sg_avail - sg), so it admits a margin of up to 5e-7 / sg_avail p.u. For entries '
                                     'with c > k TOL_MW, the size-uniform primal alternative margin = sg_avail_mva - '
                                     'sg_mva <= k TOL_MW is evaluated and the entries where the two tests disagree are '
                                     'counted (n, mwh_w).',
    'barrier_caveat': (
        'METHODOLOGICAL CAVEAT (W73 ruling 4). In these terminal IPOPT solutions |dual| x slack is constant across the '
        'near-active curtailment constraints (per W72: ~3.9e-4 raw for the TSO capability rows, ~9.1e-6 raw at the '
        'DSO), which is barrier complementarity: dual = mu_eff / slack. A large dual at a near-active constraint '
        'therefore reports a small slack, not an economically binding constraint, and dual magnitude cannot '
        'discriminate active from inactive near-active constraints here. The primal indicator (c against k x TOL_MW) '
        'is the sounder basis. The earlier statement that the curtailment mechanism is "dual-established" is '
        'withdrawn; the correct phrasing is "classified below-capability on the primal indicator, with the row-18 '
        'condition met". The W72 reclassification of the 60 TSO entries to capability_bound on their duals is '
        'superseded by at_availability (primal, c = 1.30 x TOL_MW).'),
    'headline': 'dso_E_plus_mwh_w = sum over DSO blocks of E_plus (all generator-hours, classification-independent; '
                'equals W72 dso_E_plus_mwh_w). movement = below_capability(k) - W72 recorded DSO interior (n, mwh_w).',
    'tso_log_fields': 'per TSO block log (appended, one solve per cycle): the FINAL solve is the text between the '
                      'second-to-last and the last "EXIT:" line (last-match, never first-match); from it: "objective '
                      'scaling factor", the last "Current barrier parameter mu", the unscaled "Complementarity", '
                      '"Number of Iterations", the echoed tol and compl_inf_tol. mu_floor_pred = min(tol, '
                      'compl_inf_tol x obj_scale) / (barrier_tol_factor + 1), barrier_tol_factor = 10 (IPOPT default); '
                      'at_floor iff |mu_last / mu_floor_pred - 1| <= 1e-3. ASSUMPTION recorded: the final solve in the '
                      'log is the terminal ADMM cycle\'s TSO solve (n_exit - cycles_run reported per cell).',
    'tso_barrier_gap_estimate': 'n_pairs = (variables with only lower bounds) + 2 (variables with lower and upper '
                                'bounds) + (variables with only upper bounds) + the same for inequality constraints, '
                                'from the final solve\'s problem-size lines; gap_est_raw = n_pairs x mu_last / '
                                'obj_scale; gap_est_w = admm_block_weight x gap_est_raw; summed over TSO blocks. An '
                                'order-of-magnitude estimate of the barrier term z\'s in the terminal TSO objectives '
                                '(sum of complementarity products on the central path), NOT a measured cost difference.',
    'tso_curtailment_per_block': 'E_plus_w = w_b sum_s omega_s sum_t V_plus[s,t]; E_net_w likewise over V_net (which '
                                 'may be negative: pg may exceed pg_avail by up to TOL_MW within the pg bound).',
}


def _sha(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with open(os.path.join(REPO, path)) as f:
        return json.load(f)


def _ast_constant(path, name):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return ast.unparse(node.value)
    raise KeyError(f'{name} not a module-level assignment in {path}')


def _git_clean(paths):
    out = subprocess.run(['git', 'status', '--porcelain', '--', *paths], capture_output=True, text=True, cwd=REPO)
    return out.stdout.strip() == '', out.stdout.strip()


def tso_log_paths():
    out = {}
    for label, (ed, _pair) in CELLS.items():
        key16 = ed.split('_')[0]
        paths = sorted(p for p in glob.glob(TSO_LOG_GLOB.format(key16=key16)) if not p.endswith('_recovery.log'))
        out[label] = [os.path.relpath(p, REPO) for p in paths]
    return out


def verify_inputs():
    """Verify every input before it is read. Returns ({path: sha256} of everything read, problems, unpinned)."""
    read, problems, unpinned = {}, [], {}

    def check(path, expect):
        got = _sha(path)
        read[path] = got
        if expect is not None and got != expect:
            problems.append(f'{path}: sha256 {got} != pinned {expect}')

    check(*ROW_ANALYSIS_MANIFEST)
    ram = _load(ROW_ANALYSIS_MANIFEST[0])
    for n in (1, 2, 3):
        p = os.path.join(ROOT, f'pair_{n}_manifest_sha256.json')
        check(p, ram[p])
    pair_manifests = {n: _load(os.path.join(ROOT, f'pair_{n}_manifest_sha256.json')) for n in (1, 2, 3)}
    for label, (ed, pair) in CELLS.items():
        for fn in EVAL_FILES:
            p = os.path.join(ROOT, 'evals', ed, fn)
            if p not in pair_manifests[pair]:
                problems.append(f'{p} not in pair_{pair} manifest')
                continue
            check(p, pair_manifests[pair][p])
    check(*W72_MANIFEST)
    w72m = _load(W72_MANIFEST[0])
    check(W72_OUTPUT, w72m.get(W72_OUTPUT))
    check(W72_SCRIPT, w72m.get(W72_SCRIPT))
    w72_inputs = _load(W72_OUTPUT)['inputs_sha256']
    check(CURTAILMENT_AUDIT, w72_inputs.get(CURTAILMENT_AUDIT))
    clean, detail = _git_clean(list(PRODUCTION_SOURCES) + list(CASE_PARAMS.values()))
    if not clean:
        problems.append(f'production sources / case files not git-clean: {detail}')
    for p in list(PRODUCTION_SOURCES) + list(CASE_PARAMS.values()):
        check(p, None)
        unpinned[p] = 'git-clean at HEAD (checked); not in a campaign manifest'
    for label, paths in tso_log_paths().items():
        if len(paths) != 20:
            problems.append(f'{label}: expected 20 TSO logs, found {len(paths)}')
        for p in paths:
            check(p, None)
            unpinned[p] = 'IPOPT log in the cell run directory; NOT in any campaign manifest; hashed at read time'
    return read, problems, unpinned


def r_of_rho(rho):
    return ((2.0 - rho) + math.sqrt(rho * rho + 4.0)) / (2.0 * rho)


def k_derivation_numbers():
    return {'r_of_rho_1': r_of_rho(1.0), 'k_chosen': K_CHOSEN, 'k_chosen_equals_ceil_r1': math.ceil(r_of_rho(1.0)) == K_CHOSEN,
            'rho_threshold_by_k': {str(k): 1.0 / k + 1.0 / (k + 1) for k in K_SET},
            'r_check_at_rho_threshold': {str(k): r_of_rho(1.0 / k + 1.0 / (k + 1)) for k in K_SET}}


def ipopt_options_in_force(evaluation_record):
    """Case-file IPOPT options per agent (TSO from case9_params.json; DSO from the case files, checked against the
    ipopt_options_in_force_dso the harness recorded)."""
    out = {}
    for agent, path in CASE_PARAMS.items():
        opts = _load(path)['solver']['options']
        out[agent] = {'tol': opts.get('tol'), 'compl_inf_tol': opts.get('compl_inf_tol', 1e-4),
                      'compl_inf_tol_source': 'case_file' if 'compl_inf_tol' in opts else 'ipopt_default',
                      'mu_strategy': opts.get('mu_strategy', 'monotone (ipopt_default)')}
    rec = evaluation_record['response_terminal']['ipopt_options_in_force_dso']
    agree = all(abs(rec[a]['tol']['value'] - out[a]['tol']) == 0.0 for a in ('DSO5', 'DSO7', 'DSO9'))
    return out, agree


def _mwh(entries, wb):
    return sum(wb[e['block']] * e['omega'] * e['c_mw'] for e in entries)


def _eur(entries, wb):
    return sum(wb[e['block']] * e['omega'] * e['c_mw'] * e['price_scenario_eur_mwh'] for e in entries)


def _set(entries, wb):
    return {'n': len(entries), 'mwh_w': _mwh(entries, wb), 'eur_w': _eur(entries, wb)}


def cell_reclass(label, ed, d_tol, cap_slack_tol):
    E = os.path.join(ROOT, 'evals', ed)
    rec = _load(os.path.join(E, 'evaluation_record.json'))
    rt = H.load_response_terminal(os.path.join(E, 'response_terminal.json'))
    alpha = float(rec['interface_deviation_premium']['alpha'])
    cbb = rt['curtailment_by_block']
    wb = {k: v['admm_block_weight'] for k, v in cbb.items()}
    tols = {v['tol_mw'] for v in cbb.values()}
    bases = {v['baseMVA'] for v in cbb.values()}
    assert len(tols) == 1 and len(bases) == 1, (tols, bases)
    tol_mw = tols.pop()
    base = bases.pop()
    assert abs(tol_mw - rt['constants']['TOL_FACTOR'] * base) <= 1e-15, (tol_mw, base)
    # ---- per-agent E_plus and per-TSO-block totals (classification-independent) ----
    e_plus = {'TSO': 0.0, 'DSO': 0.0}
    e_net = {'TSO': 0.0, 'DSO': 0.0}
    priced = {'TSO': 0.0, 'DSO': 0.0}
    tso_blocks = {}
    for key, blk in cbb.items():
        agent = 'TSO' if blk['kind'] == 'TSO' else 'DSO'
        w = blk['admm_block_weight']
        ep = sum(w * sc['omega'] * sum(sc['V_plus_mw']) for sc in blk['scenarios'].values())
        en = sum(w * sc['omega'] * sum(sc['V_net_mw']) for sc in blk['scenarios'].values())
        pr = sum(w * sc['omega'] * sum(p * v for p, v in zip(sc['price_eur_mwh'], sc['V_plus_mw']))
                 for sc in blk['scenarios'].values())
        e_plus[agent] += ep
        e_net[agent] += en
        priced[agent] += pr
        if agent == 'TSO':
            tso_blocks[key] = {'E_plus_mwh_w': ep, 'E_net_mwh_w': en, 'priced_eur_w': pr, 'weight': w}
    entries = rt['curtailment_entries']
    for e in entries:
        e['_agent'] = 'TSO' if e['network'] == 'TSO' else 'DSO'
        e['_row18'] = bool(e['_agent'] == 'DSO' and alpha > 0 and e.get('d_p_mw') is not None
                           and e.get('row18_premium_eur_mwh')
                           and e['d_p_mw'] <= d_tol and e['price_scenario_eur_mwh'] < alpha * e['row18_premium_eur_mwh'])
        slack = e['sg_capability_slack_pu2']
        e['_slack_capb'] = slack is not None and slack <= cap_slack_tol
        assert (e['class'] == 'capability_bound') == e['_slack_capb'], ('recorded class rule mismatch', e['class'])
        e['_margin_mva'] = (e['sg_avail_mva'] - e['sg_mva']) if e.get('sg_mva') is not None else None
    ent_mwh = {a: _mwh([e for e in entries if e['_agent'] == a], wb) for a in ('TSO', 'DSO')}
    residue = {a: e_plus[a] - ent_mwh[a] for a in ('TSO', 'DSO')}
    by_k = {}
    for k in K_SET:
        thr = k * tol_mw
        row = {'threshold_mw': thr}
        for a in ('TSO', 'DSO'):
            sel = [e for e in entries if e['_agent'] == a]
            at = [e for e in sel if e['c_mw'] <= thr]
            rest = [e for e in sel if e['c_mw'] > thr]
            capb = [e for e in rest if e['_slack_capb']]
            below = [e for e in rest if not e['_slack_capb']]
            below_r18 = [e for e in below if e['_row18']]
            rest_r18 = [e for e in rest if e['_row18']]
            at_set = _set(at, wb)
            at_set['by_recorded_class'] = {c: _set([e for e in at if e['class'] == c], wb)
                                           for c in ('capability_bound', 'interior')}
            at_set['mwh_w_incl_sub_tol_residue'] = at_set['mwh_w'] + residue[a]
            b = _set(below, wb)
            cb = _set(capb, wb)
            applicable = a == 'DSO' and alpha > 0     # the row-18 condition is defined for DSO entries at alpha > 0
            # size-uniform MVA margin test, beside only
            disagree_capb = [e for e in capb if e['_margin_mva'] is not None and e['_margin_mva'] > thr]
            disagree_below = [e for e in below if e['_margin_mva'] is not None and e['_margin_mva'] <= thr]
            row[a] = {
                'at_availability': at_set,
                'capability_bound': cb,
                'below_capability': b,
                'row18': {
                    'applicable': applicable,
                    'below_capability_meeting': _set(below_r18, wb),
                    'share_below_capability_energy': (_mwh(below_r18, wb) / b['mwh_w'])
                    if (applicable and b['mwh_w']) else None,
                    'share_below_capability_count': (len(below_r18) / len(below)) if (applicable and below) else None,
                    'all_remaining_meeting': _set(rest_r18, wb),
                    'share_all_remaining_energy': (_mwh(rest_r18, wb) / (b['mwh_w'] + cb['mwh_w']))
                    if (applicable and (b['mwh_w'] + cb['mwh_w'])) else None},
                'identity_abs_residual_mwh': abs(e_plus[a] - (at_set['mwh_w'] + residue[a] + cb['mwh_w'] + b['mwh_w'])),
                'capability_margin_sensitivity': {
                    'recorded_capb_but_margin_gt_k_tol': _set(disagree_capb, wb),
                    'recorded_below_but_margin_le_k_tol': _set(disagree_below, wb)},
            }
        by_k[str(k)] = row
    # ---- geometry consistency check on TSO entries (NOT a discriminator) ----
    geo = []
    for e in entries:
        if e['_agent'] != 'TSO':
            continue
        r = e['c_mw'] / tol_mw
        la = e['lmp_bus_dual_raw']
        kap = 2.0 * (e['pg_mw'] / base) * abs(e['sg_capability_dual_raw']) / abs(la)
        phi = kap + abs(e['pg_zU_raw']) / abs(la)
        geo.append({'r': r, 'kappa_pred': (r + 1.0) / (2.0 * r + 1.0), 'kappa_rec': kap, 'phi_rec': phi,
                    'rho_implied': 1.0 / (r + 1.0) + 1.0 / r,
                    'cap_product': abs(e['sg_capability_dual_raw'] * e['sg_capability_slack_pu2']),
                    'bound_product': abs(e['pg_zU_raw']) * (e['c_mw'] + tol_mw) / base,
                    'qg_mvar': e['qg_mvar'], 'lmp_bus_eur_mwh': e['lmp_bus_eur_mwh'],
                    'price_scenario_eur_mwh': e['price_scenario_eur_mwh'], 'block': e['block'],
                    'scenario': e['scenario'], 'hour': e['hour'], 'gen': e['gen']})
    geo_summary = None
    if geo:
        geo_summary = {
            'n': len(geo), 'r_range': [min(g['r'] for g in geo), max(g['r'] for g in geo)],
            'max_abs_kappa_pred_minus_rec': max(abs(g['kappa_pred'] - g['kappa_rec']) for g in geo),
            'kappa_rec_range': [min(g['kappa_rec'] for g in geo), max(g['kappa_rec'] for g in geo)],
            'phi_rec_range': [min(g['phi_rec'] for g in geo), max(g['phi_rec'] for g in geo)],
            'rho_implied_range': [min(g['rho_implied'] for g in geo), max(g['rho_implied'] for g in geo)],
            'cap_product_range': [min(g['cap_product'] for g in geo), max(g['cap_product'] for g in geo)],
            'bound_product_range': [min(g['bound_product'] for g in geo), max(g['bound_product'] for g in geo)],
            'max_rel_cap_minus_bound_product': max(abs(g['cap_product'] - g['bound_product']) / g['cap_product']
                                                   for g in geo),
            'price_scenario_eur_mwh_range': [min(g['price_scenario_eur_mwh'] for g in geo),
                                             max(g['price_scenario_eur_mwh'] for g in geo)],
            'locations': sorted({(g['block'], g['scenario'], g['gen']) for g in geo}),
            'hours': sorted({g['hour'] for g in geo})}
    return {
        'alpha': alpha, 'candidate_key': rec['candidate_key'], 'eval_key': rec['eval_key'], 'eval_dir': E,
        'cycles_run': rec['cycles_run'], 'status': rec['status'],
        'TOL_MW': tol_mw, 'baseMVA': base,
        'E_plus_mwh_w': e_plus, 'E_net_mwh_w': e_net, 'priced_E_plus_eur_w': priced,
        'entries_mwh_w': ent_mwh, 'sub_tol_residue_mwh_w': residue,
        'n_entries': {a: sum(1 for e in entries if e['_agent'] == a) for a in ('TSO', 'DSO')},
        'dso_E_plus_mwh_w': e_plus['DSO'],
        'by_k': by_k, 'tso_geometry_check': geo_summary, 'tso_blocks': tso_blocks,
        'ipopt_options_recorded_dso': rec['response_terminal']['ipopt_options_in_force_dso'],
        '_evaluation_record': rec,
    }


def parse_tso_log(path):
    txt = open(os.path.join(REPO, path), errors='replace').read()
    parts = txt.split('EXIT:')
    n_exit = len(parts) - 1
    final = parts[-2]
    exit_msg = parts[-1].split('\n')[0].strip()

    def last(pattern, cast=float, group=1):
        m = re.findall(pattern, final)
        if not m:
            return None
        v = m[-1]
        return cast(v[group - 1] if isinstance(v, tuple) else v)
    return {'n_exit': n_exit, 'exit': exit_msg,
            'obj_scale': last(r'objective scaling factor = ([0-9.eE+-]+)'),
            'mu_last': last(r'Current barrier parameter mu = ([0-9.eE+-]+)'),
            'compl_unscaled': last(r'Complementarity\.+:\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)', float, 2),
            'compl_scaled': last(r'Complementarity\.+:\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)', float, 1),
            'iterations': last(r'Number of Iterations\.+: (\d+)', int),
            'tol_echo': last(r'\n\s+tol = ([0-9.eE+-]+)'),
            'compl_inf_tol_echo': last(r'compl_inf_tol = ([0-9.eE+-]+)'),
            'n_var_lb_only': last(r'variables with only lower bounds:\s+(\d+)', int),
            'n_var_lb_ub': last(r'variables with lower and upper bounds:\s+(\d+)', int),
            'n_var_ub_only': last(r'variables with only upper bounds:\s+(\d+)', int),
            'n_ineq_lb_only': last(r'inequality constraints with only lower bounds:\s+(\d+)', int),
            'n_ineq_lb_ub': last(r'inequality constraints with lower and upper bounds:\s+(\d+)', int),
            'n_ineq_ub_only': last(r'inequality constraints with only upper bounds:\s+(\d+)', int),
            'mtime_utc': datetime.fromtimestamp(os.path.getmtime(os.path.join(REPO, path)), timezone.utc).isoformat()}


def tso_unit_cell_investigation(cells, log_paths):
    per_cell = {}
    for label, paths in log_paths.items():
        c = cells[label]
        blocks = {}
        for p in paths:
            m = re.search(r'optim_log_case9_(\d+)_(\w+)\.log$', p)
            key = f'TSO|{m.group(1)}|{m.group(2)}'
            s = parse_tso_log(p)
            tol, cit, sc = s['tol_echo'], s['compl_inf_tol_echo'], s['obj_scale']
            if None not in (tol, cit, sc, s['mu_last']):
                floor = min(tol, cit * sc) / (IPOPT_BARRIER_TOL_FACTOR_DEFAULT + 1.0)
                s['mu_floor_pred'] = floor
                s['at_floor'] = abs(s['mu_last'] / floor - 1.0) <= 1e-3
            s.update(c['tso_blocks'].get(key, {}))
            s['n_pairs'] = (s['n_var_lb_only'] + 2 * s['n_var_lb_ub'] + s['n_var_ub_only'] + s['n_ineq_lb_only']
                            + 2 * s['n_ineq_lb_ub'] + s['n_ineq_ub_only'])
            s['barrier_gap_est_raw'] = s['n_pairs'] * s['mu_last'] / s['obj_scale']
            s['barrier_gap_est_w_eur'] = s['weight'] * s['barrier_gap_est_raw']
            blocks[key] = s
        vals = list(blocks.values())
        per_cell[label] = {
            'alpha': c['alpha'], 'candidate_key': c['candidate_key'], 'eval_key': c['eval_key'],
            'cycles_run': c['cycles_run'], 'n_exit_minus_cycles': sorted({v['n_exit'] - c['cycles_run'] for v in vals}),
            'all_final_exits_optimal': all(v['exit'] == 'Optimal Solution Found.' for v in vals),
            'tol_echo': sorted({v['tol_echo'] for v in vals}), 'compl_inf_tol_echo': sorted({v['compl_inf_tol_echo']
                                                                                            for v in vals}),
            'obj_scale_range': [min(v['obj_scale'] for v in vals), max(v['obj_scale'] for v in vals)],
            'mu_last_range': [min(v['mu_last'] for v in vals), max(v['mu_last'] for v in vals)],
            'mu_last_median': statistics.median(v['mu_last'] for v in vals),
            'n_at_floor': sum(int(v.get('at_floor', False)) for v in vals),
            'compl_unscaled_median': statistics.median(v['compl_unscaled'] for v in vals),
            'iterations_median': statistics.median(v['iterations'] for v in vals),
            'TSO_E_plus_mwh_w': c['E_plus_mwh_w']['TSO'], 'TSO_E_net_mwh_w': c['E_net_mwh_w']['TSO'],
            'TSO_priced_E_plus_eur_w': c['priced_E_plus_eur_w']['TSO'],
            'TSO_n_entries': c['n_entries']['TSO'],
            'TSO_barrier_gap_est_w_eur': sum(v['barrier_gap_est_w_eur'] for v in vals),
            'blocks': blocks}
    x0 = [k for k in per_cell if k.startswith('x0_')]
    u = per_cell[UNIT_LABEL]
    ratios = {}
    for key in u['blocks']:
        num = statistics.median(per_cell[k]['blocks'][key]['E_plus_mwh_w'] for k in x0)
        ratios[key] = {'x0_median_E_plus': num, 'unit_E_plus': u['blocks'][key]['E_plus_mwh_w'],
                       'x0_median_mu_last': statistics.median(per_cell[k]['blocks'][key]['mu_last'] for k in x0),
                       'unit_mu_last': u['blocks'][key]['mu_last'],
                       'x0_obj_scale': sorted({per_cell[k]['blocks'][key]['obj_scale'] for k in x0}),
                       'unit_obj_scale': u['blocks'][key]['obj_scale']}
    w72_unit = _load(W72_OUTPUT)['row']['unit']
    gap_x05 = per_cell['x0_a0p50']['TSO_barrier_gap_est_w_eur']
    gap_u = u['TSO_barrier_gap_est_w_eur']
    gap_cmp = {'x0_a0p50_TSO_gap_est_w_eur': gap_x05, 'unit_TSO_gap_est_w_eur': gap_u,
               'x0_minus_unit_gap_est_w_eur': gap_x05 - gap_u,
               'w72_value_eur': w72_unit['value_eur'], 'w72_resolution_two_cell': w72_unit['resolution_two_cell'],
               'gap_difference_over_resolution': (gap_x05 - gap_u) / w72_unit['resolution_two_cell'],
               'gap_difference_over_value': (gap_x05 - gap_u) / w72_unit['value_eur'],
               'note': 'ORDER-OF-MAGNITUDE ESTIMATE, not a measured objective difference: on the central path the '
                       'complementarity pairs sum to n_pairs x mu (scaled), i.e. n_pairs x mu / obj_scale in raw '
                       'objective units (EUR per representative day, the ADMM-augmented TSO objective, not Q itself). '
                       'DSO blocks not examined.'}
    n_lower = sum(1 for v in ratios.values() if v['unit_E_plus'] < v['x0_median_E_plus'])
    n_mu_lower = sum(1 for v in ratios.values() if v['unit_mu_last'] < v['x0_median_mu_last'])
    return {
        'per_cell': per_cell, 'unit_vs_x0_by_block': ratios, 'barrier_gap_comparison': gap_cmp,
        'n_blocks_unit_E_plus_below_x0_median': n_lower, 'n_blocks_unit_mu_below_x0_median': n_mu_lower,
        'n_blocks': len(ratios),
        'provenance_note': 'The TSO IPOPT logs are NOT in any campaign manifest (the run directories under '
                           'data/SRP1/Results/P56A/evals are untracked); they are hashed at read time into this stage\'s '
                           'manifest and their mtimes recorded. Their correspondence to the terminal cycle rests on the '
                           'last-match rule and n_exit - cycles_run, not on a manifest.',
    }


def cross_check_w72(cells, w72):
    out = {}
    ok = True
    for label, c in cells.items():
        w = w72['cells'][label]
        d_e = abs(c['dso_E_plus_mwh_w'] - w['dso_E_plus_mwh_w'])
        d_t = abs(c['E_plus_mwh_w']['TSO'] - w['curtailment_totals']['TSO']['E_plus_mwh_w'])
        rec_int = w['dso_interior_corrected']
        # recorded DSO interior at k = 0 equivalence: all DSO interior entries (c > TOL by capture)
        out[label] = {'dso_E_plus_abs_diff': d_e, 'tso_E_plus_abs_diff': d_t,
                      'w72_dso_interior_recorded': {'n': rec_int['n'], 'mwh_w': rec_int['mwh_w']},
                      'w72_TOL_MW': w['TOL_MW'], 'TOL_MW_equal': w['TOL_MW'] == c['TOL_MW'],
                      'w72_row18_condition_and_interior': w['row18_condition_and_interior']}
        ok = ok and d_e <= 1e-9 * max(1.0, w['dso_E_plus_mwh_w']) and d_t <= 1e-9 and w['TOL_MW'] == c['TOL_MW']
    return out, ok


def headline(cells, xc):
    rows = {}
    for label, c in cells.items():
        w = xc[label]['w72_dso_interior_recorded']
        k2 = c['by_k'][str(K_CHOSEN)]['DSO']
        per_k = {}
        for k in K_SET:
            b = c['by_k'][str(k)]['DSO']['below_capability']
            per_k[str(k)] = {'n': b['n'], 'mwh_w': b['mwh_w'], 'dn_vs_w72': b['n'] - w['n'],
                             'dmwh_vs_w72': b['mwh_w'] - w['mwh_w'],
                             'dmwh_rel_pct_vs_w72': 100.0 * (b['mwh_w'] - w['mwh_w']) / w['mwh_w'] if w['mwh_w'] else None}
        rows[label] = {
            'alpha': c['alpha'],
            'dso_E_plus_mwh_w_unchanged': c['dso_E_plus_mwh_w'],
            'at_availability_mwh_w_entries': k2['at_availability']['mwh_w'],
            'at_availability_n_entries': k2['at_availability']['n'],
            'at_availability_mwh_w_incl_sub_tol_residue': k2['at_availability']['mwh_w_incl_sub_tol_residue'],
            'below_capability': {'n': k2['below_capability']['n'], 'mwh_w': k2['below_capability']['mwh_w']},
            'capability_bound': {'n': k2['capability_bound']['n'], 'mwh_w': k2['capability_bound']['mwh_w']},
            'row18_share_below_capability_energy': k2['row18']['share_below_capability_energy'],
            'row18_share_below_capability_count': k2['row18']['share_below_capability_count'],
            'w72_dso_interior_recorded': w,
            'below_capability_movement_vs_w72_by_k': per_k,
            'tso_at_k2': {cls: {'n': c['by_k'][str(K_CHOSEN)]['TSO'][cls]['n'],
                                'mwh_w': c['by_k'][str(K_CHOSEN)]['TSO'][cls]['mwh_w']}
                          for cls in ('at_availability', 'capability_bound', 'below_capability')},
        }
    return rows


def run():
    if os.path.exists(OUT_JSON):
        raise SystemExit(f'REFUSED: {OUT_JSON} exists (write-once)')
    started = datetime.now(timezone.utc).isoformat()
    read, problems, unpinned = verify_inputs()
    print(f'[W73] inputs verified: {len(read)} files, problems: {problems}', flush=True)
    if problems:
        fails = GUARD.verify(0)
        raise SystemExit(f'REFUSED: input verification failed {problems}; guard verify(0) {fails}')
    d_tol = _ast_constant(CURTAILMENT_AUDIT, 'D_TOL_MW')
    cap_slack_tol = _ast_constant(CURTAILMENT_AUDIT, 'CAP_SLACK_TOL')
    eq_tol = _ast_constant('definitions.py', 'EQUALITY_TOLERANCE')
    mch_src = open(os.path.join(REPO, 'model_construction_helpers.py')).read()
    pg_bound_text = 'return (0.0, gen.pg[s_o][p] + EQUALITY_TOLERANCE)'
    pg_bound_present = pg_bound_text in mch_src
    if not pg_bound_present:
        raise SystemExit('REFUSED: the pg upper bound used in the k derivation is not in model_construction_helpers.py')
    cells = {}
    for label, (ed, _pair) in CELLS.items():
        cells[label] = cell_reclass(label, ed, d_tol, cap_slack_tol)
        c = cells[label]
        k2 = c['by_k'][str(K_CHOSEN)]
        print(f"[W73] {label}: alpha={c['alpha']} TOL_MW={c['TOL_MW']} E+ TSO={c['E_plus_mwh_w']['TSO']!r} "
              f"DSO={c['dso_E_plus_mwh_w']!r} k=2 DSO at_av n={k2['DSO']['at_availability']['n']} "
              f"below n={k2['DSO']['below_capability']['n']} mwh={k2['DSO']['below_capability']['mwh_w']!r} "
              f"capb n={k2['DSO']['capability_bound']['n']} TSO at_av n={k2['TSO']['at_availability']['n']}", flush=True)
    opts, opts_agree = ipopt_options_in_force(cells['x0_a0p00']['_evaluation_record'])
    for c in cells.values():
        c.pop('_evaluation_record')
    for a, o in opts.items():
        if o['tol'] != eq_tol:
            raise SystemExit(f'REFUSED: k derivation assumes IPOPT tol == EQUALITY_TOLERANCE; {a} tol {o["tol"]}')
    w72 = _load(W72_OUTPUT)
    xc, xc_ok = cross_check_w72(cells, w72)
    print(f'[W73] cross-check vs W72 alpha_row_recompute.json: ok={xc_ok}', flush=True)
    log_paths = tso_log_paths()
    tso_inv = tso_unit_cell_investigation(cells, log_paths)
    print(f"[W73] TSO logs: unit obj_scale {tso_inv['per_cell'][UNIT_LABEL]['obj_scale_range']} mu "
          f"{tso_inv['per_cell'][UNIT_LABEL]['mu_last_range']}; x0_a0p50 obj_scale "
          f"{tso_inv['per_cell']['x0_a0p50']['obj_scale_range']} mu {tso_inv['per_cell']['x0_a0p50']['mu_last_range']}",
          flush=True)
    fails = GUARD.verify(0)
    counts = dict(GUARD.counts)
    GUARD.uninstall()
    out = {
        'stage': 'P5.15 Addendum 40 ruling 1, W73: alpha-row curtailment reclassification on the primal indicator '
                 '(at_availability class; k derivation and sensitivity; barrier caveat; TSO unit-cell explanation)',
        'utc_started': started, 'utc_finished': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
        'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
        'instance': {'campaign': 's53_alpha_row_v25', 'spec': '70965374', 'frozen_spec': 'v25 407a4b33',
                     'flexibility_price': 'BASELINE (NOT the m = 2 variant)',
                     'cells': {k: {'eval_dir': v['eval_dir'], 'candidate_key': v['candidate_key'],
                                   'eval_key': v['eval_key'], 'alpha': v['alpha']} for k, v in cells.items()}},
        'energy_convention': 'MWh per representative day x admm_block_weight x scenario probability (Q weighting); no '
                             'objective values reported.',
        'formulas': FORMULAS,
        'constants': {'K_SET': K_SET, 'K_CHOSEN': K_CHOSEN, 'D_TOL_MW': d_tol, 'CAP_SLACK_TOL': cap_slack_tol,
                      'EQUALITY_TOLERANCE': eq_tol, 'pg_bound_source_text': pg_bound_text,
                      'pg_bound_present': pg_bound_present,
                      'IPOPT_BARRIER_TOL_FACTOR_DEFAULT': IPOPT_BARRIER_TOL_FACTOR_DEFAULT},
        'k_derivation': dict(k_derivation_numbers(), ipopt_options_case_files=opts,
                             dso_case_file_tol_agrees_with_recorded_in_force=opts_agree),
        'inputs_sha256': read, 'inputs_unpinned_note': unpinned,
        'headline': headline(cells, xc),
        'cells': cells,
        'tso_unit_cell_investigation': tso_inv,
        'cross_check_vs_w72': {'per_cell': xc, 'all_ok': xc_ok},
        'guard': {'counts': counts, 'verify_0': fails, 'declared_solves': 0},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'x') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'[W73] wrote {OUT_JSON}; guard {counts} verify(0) {fails}', flush=True)
    if fails or not xc_ok:
        raise SystemExit(1)


def manifest():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
    out = _load(OUT_JSON)
    bad = [p for p, h in out['inputs_sha256'].items() if _sha(p) != h]
    if out['script_sha256'] != _sha(SCRIPT_REL):
        bad.append(SCRIPT_REL)
    if bad:
        raise SystemExit(f'REFUSED: changed since the run: {bad}')
    m = {OUT_JSON: _sha(OUT_JSON), OUT_LOG: _sha(OUT_LOG), SCRIPT_REL: _sha(SCRIPT_REL)}
    m.update(out['inputs_sha256'])
    with open(OUT_MANIFEST, 'x') as f:
        json.dump(m, f, indent=1)
    fails = GUARD.verify(0)
    print(f'[W73] wrote {OUT_MANIFEST}: {len(m)} entries; guard {dict(GUARD.counts)} verify(0) {fails}', flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--run']:
        run()
    elif sys.argv[1:] == ['--manifest']:
        manifest()
    else:
        raise SystemExit('usage: p515_s53_alpha_row_reclass.py --run | --manifest')
