"""P5.15 Addendum 45 item 3, task W81 -- the corrected barrier-gap estimate on SRP1, TERMINAL ADMM solves, x = 0 versus the
smallest node-7 unit (n7 0.25 MVA / 1.0 MWh, 2025), against Phase A's bars. READ-ONLY, ZERO SOLVES: nothing is built, solved
or re-run, no committed artifact is modified; SolveProfileGuard(permitted=()) is armed at import and verify(0) is reached on
every exit path (try/finally).

Addendum 45 item 3 (verbatim): "Zero solves: the corrected gap estimate (terminal ADMM solves) on the SRP1 x = 0 /
smallest-unit pair, against Phase A's bars."

CELLS (Phase A, AA-on C3 configuration, frozen spec v15; problem instances named by candidate_key, recorded in the output):
  x0       a0_c7:x0              data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0
  unit     a0_c7:n7_p0.25_e1.0   data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7eb1ce62c2509f54_n7_p0_25_e1_0
  unit_dup a1a:n7_4h_e1          data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1  (same candidate
           key; a bitwise reproduction; used only for prediction P9)
  IPOPT main logs: data/SRP1/Results/P56A/evals/p515s44_<campaign>_<eval_key16>_run/logs/optim_log_<network>_<y>_<day>.log
  (git-ignored; NOT covered by any committed manifest -- the harness scans every tracked *manifest*.json to scope that claim --
  so each log is sha256-recorded at read time and re-verified by --manifest).

IDENTIFICATION OF THE TERMINAL ADMM SOLVE (and exclusion of the hull polish):
  Each block's log is appended once per IPOPT run. With K = evaluation_record.cycles_run, production solves every TSO/DSO
  block once at initialisation and once per ADMM cycle; a post-certification hull polish, when requested, appends ONE more
  solve (primary attempt; retries go to *_recovery*.log files). Two layouts are admitted and nothing else:
    layout A (no polish): post_certification is None, n_solves == K + 1, cold-multiplier solves == [1];
    layout B (polish):    post_certification present, n_solves == K + 2, cold-multiplier solves == [1, K + 2].
  A solve is cold-multiplier iff the first "||curr_z_L||_inf" of the run is exactly 1.0 (IPOPT's default z_L init: the
  initialisation and the polish; every ADMM-cycle solve is multiplier-warm). In BOTH layouts the terminal ADMM solve is solve
  number K + 1 (1-based) = the initialisation plus K cycles; the polish (layout B only) is solve K + 2 and is never read.
  Further checks: no *_recovery*.log for the block; network IPOPT runs == 48 x (K + 1) and 48 x (K + 1) + (IPOPT runs in
  every ESSO log, optim_log_esso_node<n>_init.txt + _cycle<k>.txt) == solve_profile.observed.permitted_solve;
  and the units check below ties solve K + 1 to the certified terminal state. W73's defect was reading the LAST solve, which
  in layout B is the polish; control C10 re-applies this rule to W74's 2x2 logs (layout B) and must reproduce W74's
  terminal-ADMM figure exactly, and the last-solve reading must reproduce W73's.

FORMULAS (preserved here and in the output's `formulas`):
  n_pairs     = var_lb_only + 2 var_lb_ub + var_ub_only + ineq_lb_only + 2 ineq_lb_ub + ineq_ub_only   (IPOPT problem-size
                lines of the solve; the number of barrier complementarity pairs; fixed variables are removed by
                fixed_variable_treatment = make_parameter and equalities carry no pair) -- identical to W73/W74.
  mu_last     = the last "Current barrier parameter mu" of the solve;  s = the solve's "objective scaling factor".
  g_b         = w_b x n_pairs x mu_last / s                     (PRIMARY; EUR of Q; identical to W73/W74's formula)
  g_b^upper   = w_b x n_pairs x C_unscaled                       (UPPER VARIANT; C_unscaled = the final "Complementarity"
                (unscaled) of the solve, IPOPT's max-norm of the complementarity products, so sum_i s_i z_i <= n_pairs x C)
  G_agent(c)  = sum of g_b over the agent's blocks (TSO: 12; DSO: 36 = 3 DNs x 12);  G_total = G_TSO + G_DSO
  Delta G     = G(x0) - G(unit)  -- the estimated barrier contribution to value = Q(0) - Q(unit) (positive: value overstated)
  R           = |Delta G_total| / (bar(x0) + bar(unit)),  bar = Phase A bar (max |gross step| over the last 10 cycles,
                phase_a_tables.json T1 `bar_eur`) -- Addendum 44 ruling 6: the bar-sum is the operative resolution test;
                sigma_Q is provenance only and is not used.
  w_b         = the block's admm_block_weight (component_levels_terminal.json), cross-checked against production's
                shared_resources_planning._get_admm_block_weight on SRP1.json's years/days/discount and against
                settlement_weighted / settlement_unweighted.
DERIVATION: IPOPT minimises s f(x) with a log barrier mu sum ln(slack). At an exact barrier-subproblem solution every pair
  has slack x multiplier = mu in the scaled problem, so the Lagrangian duality gap is n_pairs mu (scaled objective units) =
  n_pairs mu / s in units of f; for a convex problem this bounds f(x_mu) - f*. The solved f is production's rescaled
  subproblem objective (p515_g_g1_g4_admm_gates.run_admm_arm solves under p58_rescale.patched_admm_objectives, which
  multiplies admm_objective = base / effective_scale + AL by effective_scale = sigma / w_b), so f = base (EUR per
  representative day, scenario-probability weighted, settlement included) + effective_scale x AL; Q = sum_b w_b x (base_b -
  settlement_b), so one unit of f in block b is w_b EUR of Q. UNITS CHECK (asserted per block, not assumed): the terminal
  solve's final unscaled objective / base_unweighted, base_unweighted = generation + internal flexibility + load curtailment +
  RES curtailment penalty + ESS usage + detector (slack) penalty total + interface settlement, all unweighted, from the
  committed terminal artifacts; the alternative units (f = base / effective_scale) would give a ratio ~ w_b / sigma ~ 5e-6.
LIMITATIONS (also in the output): an ORDER-OF-MAGNITUDE estimate of the barrier gap, NOT a measured change in Q. (1) The
  subproblems are nonconvex AC-OPF: n mu is not a bound, only a scale. (2) n_pairs mu is the complementarity sum on the
  central path; to first order the primal offset involves only the strictly active pairs (~ n_active mu); inactive pairs
  load the dual side, so the primary figure over-states the primal offset. (3) The final iterate is not exactly on the
  central path (the ratio C_scaled / mu_last is reported; the upper variant brackets it). (4) The gap is on the whole
  subproblem objective, which carries the settlement transfer and effective_scale x AL terms that are not in Q; it is not
  attributed to Q's components. (5) A block's barrier offset also moves the ADMM consensus point, which this per-block
  estimate does not capture. (6) Only a re-solve at a pinned scaling / tighter mu would MEASURE the change; none is made.
  The ESSO is outside Q (_get_operational_recourse_components sums the TSO and DSO network blocks only) and is not estimated.

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w81_srp1_gap_estimate.py --run \
      > data/SRP1/Results/P515S53/srp1_gap_w81_launch_r1.log 2>&1
  (r0 went to srp1_gap_w81_launch.log and STOPPED at input verification on two harness defects; kept as evidence)
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w81_srp1_gap_estimate.py --manifest \
      > data/SRP1/Results/P515S53/srp1_gap_w81_manifest_launch.log 2>&1
"""

import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W81 SRP1 barrier-gap estimate (read-only)').install()

P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CAMPAIGN_MANIFEST = {'a0_c7': os.path.join(P45, 'campaign_s45_a0_c7', 'campaign_manifest_sha256.json'),
                     'a1a': os.path.join(P45, 'campaign_s45_a1a', 'campaign_manifest_sha256.json')}
CAMPAIGN_ID = {'a0_c7': 's45_a0_c7', 'a1a': 's45_a1a'}
CELLS = {
    'x0': {'identity': 'a0_c7:x0', 'campaign': 'a0_c7',
           'eval_dir': os.path.join(P45, 'campaign_s45_a0_c7', 'evals', '7aa017f09989b56d_x0')},
    'unit': {'identity': 'a0_c7:n7_p0.25_e1.0', 'campaign': 'a0_c7',
             'eval_dir': os.path.join(P45, 'campaign_s45_a0_c7', 'evals', '7eb1ce62c2509f54_n7_p0_25_e1_0')},
    'unit_dup': {'identity': 'a1a:n7_4h_e1', 'campaign': 'a1a',
                 'eval_dir': os.path.join(P45, 'campaign_s45_a1a', 'evals', '7eb1ce62c2509f54_n7_4h_e1')},
}
EVAL_FILES = ('evaluation_record.json', 'component_levels_terminal.json', 'interface_settlement_detail_s31c.json')
RUN_LOGS = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', 'p515s44_{campaign_id}_{key16}_run', 'logs')
CASE_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
PHASE_A_TABLES = os.path.join(P45, 'phase_a_tables', 'phase_a_tables.json')
PHASE_A_MANIFEST = os.path.join(P45, 'phase_a_tables', 'manifest_sha256.json')
BASE_COMPONENTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
                   'ess_usage_cost', 'detector_penalty_total')

# control C10: W74's 2x2 alpha-row logs (layout B)
W74_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'polish_feasibility_w74', 'polish_feasibility_w74.json')
W74_MANIFEST = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'polish_feasibility_w74',
                            'polish_feasibility_w74_manifest_sha256.json')
W74_CONTROL_CELLS = ('x0_a0p50', 'n7_4h_e1_a0p50')
W74_TERMINAL_DIFF = 15078.913057791275
W73_COMMITTED_DIFF = 18790.22018567154

OUT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'srp1_gap_w81')
PREDICTIONS = os.path.join(OUT, 'predictions_w81.json')
PREDICTIONS_SHA256 = '4a569f6ca08971a7d5f98c6e515fd43a91143244af32c23c1814fb0f2d6acc39'
PREDICTIONS_COMMIT = 'eb3efc5f'
OUT_JSON = os.path.join(OUT, 'srp1_gap_w81.json')
OUT_MANIFEST = os.path.join(OUT, 'srp1_gap_w81_manifest_sha256.json')
# r0 (srp1_gap_w81_launch.log) STOPPED at input verification on two harness defects (phase_a_tables manifest shape; ESSO
# init logs not globbed) -- no output written, guard verify(0) == []; it is kept, not overwritten; r1 logs here:
LAUNCH_LOG_R0_FAILED = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'srp1_gap_w81_launch.log')
LAUNCH_LOG = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'srp1_gap_w81_launch_r1.log')

FORMULAS = {
    'n_pairs': 'var_lb_only + 2 var_lb_ub + var_ub_only + ineq_lb_only + 2 ineq_lb_ub + ineq_ub_only (IPOPT problem-size '
               'lines of the solve) -- identical to W73 FORMULAS[tso_barrier_gap_estimate] and W74',
    'terminal_admm_solve': 'solve number K + 1 (1-based) of the block log, K = evaluation_record.cycles_run; admitted '
                           'layouts: A (post_certification None, n_solves == K + 1, cold == [1]) and B (post_certification '
                           'present, n_solves == K + 2, cold == [1, K + 2], solve K + 2 = the hull polish, never read); '
                           'cold iff the run\'s first ||curr_z_L||_inf == 1.0',
    'g_primary_b': 'w_b x n_pairs x mu_last / obj_scale (EUR of Q) -- W73/W74 formula, on the terminal ADMM solve',
    'g_upper_b': 'w_b x n_pairs x C_unscaled (C_unscaled = the solve\'s final unscaled Complementarity, IPOPT max-norm)',
    'G_agent': 'sum of g_b over the agent\'s blocks (TSO 12, DSO 36); G_total = G_TSO + G_DSO',
    'Delta_G': 'G(x0) - G(unit): estimated barrier contribution to value = Q(0) - Q(unit); positive = value overstated',
    'R': '|Delta G_total| / (bar(x0) + bar(unit)), bars = phase_a_tables.json T1 bar_eur (Addendum 44 ruling 6)',
    'units_check': 'ratio_b = f_ipopt_final_unscaled(terminal solve) / base_unweighted_b; base_unweighted_b = sum of '
                   f'{list(BASE_COMPONENTS)} (component_levels_terminal.json unweighted) + interface_settlement_unweighted '
                   '(interface_settlement_detail_s31c.json); residual_b = f - base (EUR/rep. day; = effective_scale x AL '
                   'terms + rounding). Hypothesis EUR/rep.day holds iff all |ratio - 1| <= 0.05 and median <= 0.01',
    'central_path_ratio': 'C_scaled_final / mu_last (1 on the central path)',
    'mu_floor_pred': 'min(tol, compl_inf_tol x obj_scale) / (barrier_tol_factor + 1), barrier_tol_factor = 10 (IPOPT '
                     'default, not set in the options list); at_floor iff |mu_last / mu_floor_pred - 1| <= 1e-3; tol and '
                     'compl_inf_tol from the options list printed immediately before the run (None if absent)',
    'block_weight_crosscheck': 'w_b == shared_resources_planning._get_admm_block_weight(SimpleNamespace(years, days, '
                               'discount_factor from SRP1.json), y, d) to 1e-9 relative, and == settlement_weighted / '
                               'settlement_unweighted to 1e-9 relative',
}
LIMITATIONS = [
    'ORDER-OF-MAGNITUDE estimate of the interior-point barrier gap; NOT a measured change in Q (no re-solve is made).',
    'Subproblems are nonconvex AC-OPF: n_pairs x mu is not a bound on f(x_mu) - f*, only its scale.',
    'n_pairs x mu is the central-path complementarity SUM; the first-order primal offset involves only strictly active pairs '
    '(~ n_active x mu), so the primary figure over-states the primal offset; inactive pairs load the dual side.',
    'The final iterate is not exactly on the central path (central_path_ratio reported); the upper variant brackets it.',
    'The gap is on the whole solved subproblem objective (base incl. settlement transfer + effective_scale x AL terms), '
    'not attributed to Q\'s components.',
    'Per-block estimate: the barrier offset\'s effect on the ADMM consensus point (hence on other blocks) is not captured.',
    'ESSO blocks are outside Q (_get_operational_recourse_components sums TSO + DSO network blocks) and are not estimated.',
    'Cells are Phase A (AA-on C3 configuration, spec v15); the post-Addendum-30 baseline (C2) cells are not examined.',
]


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    h = hashlib.sha256()
    with open(_abs(rel), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)


def _tracked_clean(rel):
    tracked = _git('ls-files', '--error-unmatch', rel).returncode == 0
    clean = tracked and _git('diff', '--quiet', 'HEAD', '--', rel).returncode == 0
    return tracked, clean


def _manifest_entries(man):
    """Committed manifests come in two shapes: {path: sha} (campaign, W74) or {'files': [[path, {sha256}]],
    'hash_inventory_of_inputs': [...]} (phase_a_tables)."""
    out = {}
    if isinstance(man, dict):
        for k, v in man.items():
            if isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v):
                out[k] = v
        for key in ('files', 'hash_inventory_of_inputs'):
            section = man.get(key) or {}
            items = section.items() if isinstance(section, dict) else section
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                    out[item[0]] = item[1].get('sha256')
    return out


class Verifier:
    def __init__(self):
        self.records = {}
        self.failures = []

    def manifest(self, man_rel):
        tracked, clean = _tracked_clean(man_rel)
        if not (tracked and clean):
            self.failures.append(f'manifest not tracked+clean: {man_rel}')
        self.records[man_rel] = {'sha256': _sha(man_rel), 'git_tracked': tracked, 'git_clean': clean,
                                 'role': 'committed manifest'}
        return _manifest_entries(_load(man_rel))

    def against(self, rel, man_rel, entries):
        got = _sha(rel)
        exp = entries.get(rel)
        ok = exp is not None and exp == got
        tracked, clean = _tracked_clean(rel)
        if not ok:
            self.failures.append(f'{rel}: sha {got} vs manifest {man_rel} entry {exp}')
        self.records[rel] = {'sha256': got, 'verified_against_manifest': man_rel, 'manifest_sha256': exp,
                             'matches': ok, 'git_tracked': tracked, 'git_clean': clean}
        return ok

    def tracked_only(self, rel, role):
        tracked, clean = _tracked_clean(rel)
        if not (tracked and clean):
            self.failures.append(f'not tracked+clean: {rel}')
        self.records[rel] = {'sha256': _sha(rel), 'git_tracked': tracked, 'git_clean': clean, 'role': role}


# ----------------------------------------------------------------------------------------------------------------------
_PAIR_KEYS = (('var_lb_only', r'variables with only lower bounds'),
              ('var_lb_ub', r'variables with lower and upper bounds'),
              ('var_ub_only', r'variables with only upper bounds'),
              ('ineq_lb_only', r'inequality constraints with only lower bounds'),
              ('ineq_lb_ub', r'inequality constraints with lower and upper bounds'),
              ('ineq_ub_only', r'inequality constraints with only upper bounds'))


def parse_log(rel):
    """Every IPOPT run in an appended log, in order. Options blocks ('List of options:') are attributed to the run that
    FOLLOWS them (IPOPT prints the list before the banner); a run with no preceding list in its own gap gets None."""
    with open(_abs(rel), errors='replace') as handle:
        txt = handle.read()
    starts = [m.start() for m in re.finditer(r'This is Ipopt version', txt)]
    opt_starts = [m.start() for m in re.finditer(r'List of options:', txt)]
    runs = []
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(txt)
        seg = txt[st:end]
        prev = starts[i - 1] if i else -1
        opts_at = [o for o in opt_starts if prev < o < st]
        opts = {'tol': None, 'compl_inf_tol': None}
        if opts_at:
            block = txt[opts_at[-1]:st]
            for k in opts:
                m = re.search(rf'^\s+{k} = (\S+)', block, re.M)
                opts[k] = float(m.group(1)) if m else None

        def first(p, cast=float):
            m = re.search(p, seg)
            return cast(m.group(1)) if m else None

        def last(p, cast=float):
            m = re.findall(p, seg)
            return cast(m[-1]) if m else None
        n = {k: first(rf'{v}:\s+(\d+)', int) for k, v in _PAIR_KEYS}
        pairs = (None if any(v is None for v in n.values()) else
                 n['var_lb_only'] + 2 * n['var_lb_ub'] + n['var_ub_only'] + n['ineq_lb_only'] + 2 * n['ineq_lb_ub']
                 + n['ineq_ub_only'])
        k_iter = seg.rfind('Number of Iterations')
        final = seg[k_iter:] if k_iter >= 0 else ''
        mo = re.search(r'Objective\.+:\s+(\S+)\s+(\S+)', final)
        mc = re.search(r'Complementarity\.+:\s+(\S+)\s+(\S+)', final)
        runs.append({'index_1based': i + 1,
                     'obj_scale': first(r'objective scaling factor = (\S+)'),
                     'z_L_inf_start': first(r'\|\|curr_z_L\|\|_inf = (\S+)'),
                     'mu_last': last(r'Current barrier parameter mu = (\S+)'),
                     'iterations': first(r'Number of Iterations\.+: (\d+)', int),
                     'objective_scaled_final': float(mo.group(1)) if mo else None,
                     'objective_unscaled_final': float(mo.group(2)) if mo else None,
                     'compl_scaled_final': float(mc.group(1)) if mc else None,
                     'compl_unscaled_final': float(mc.group(2)) if mc else None,
                     'exit': last(r'EXIT: (.*)', str), 'n_var': first(r'Total number of variables\.+:\s+(\d+)', int),
                     'options_tol': opts['tol'], 'options_compl_inf_tol': opts['compl_inf_tol'],
                     'n_pairs': pairs, **n})
    return runs


def identify(runs, cycles_run, post_certification_present):
    n = len(runs)
    cold = [r['index_1based'] for r in runs if r['z_L_inf_start'] == 1.0]
    if not post_certification_present:
        layout, ok = 'A', (n == cycles_run + 1 and cold == [1])
    else:
        layout, ok = 'B', (n == cycles_run + 2 and cold == [1, cycles_run + 2])
    return {'layout': layout, 'n_solves': n, 'cold_multiplier_solves': cold, 'identification_holds': ok,
            'terminal_index_1based': cycles_run + 1, 'polish_index_1based': cycles_run + 2 if layout == 'B' else None}


def _floor(run):
    tol, cit, s = run['options_tol'], run['options_compl_inf_tol'], run['obj_scale']
    if None in (tol, cit, s, run['mu_last']):
        return None, None
    f = min(tol, cit * s) / 11.0
    return f, abs(run['mu_last'] / f - 1.0) <= 1e-3


# ----------------------------------------------------------------------------------------------------------------------
def block_list(case):
    years = sorted(case['Years'], key=int)
    days = list(case['Days'])
    blocks = [('TSO', None, case['TransmissionNetwork']['name'], y, d) for y in years for d in days]
    for dn in case['DistributionNetworks']:
        blocks += [('DSO', dn['connection_node_id'], dn['name'], y, d) for y in years for d in days]
    return blocks


def cell_analysis(label, spec, case, ver, srp):
    ed = spec['eval_dir']
    man_rel = CAMPAIGN_MANIFEST[spec['campaign']]
    entries = ver.manifest(man_rel)
    for f in EVAL_FILES:
        ver.against(os.path.join(ed, f), man_rel, entries)
    rec = _load(os.path.join(ed, 'evaluation_record.json'))
    comp = _load(os.path.join(ed, 'component_levels_terminal.json'))['blocks']
    settl = _load(os.path.join(ed, 'interface_settlement_detail_s31c.json'))['per_block_interface_settlement']
    K = rec['cycles_run']
    key16 = rec['eval_key'][:16]
    if not os.path.basename(ed).startswith(key16):
        raise SystemExit(f'{label}: eval dir {ed} does not start with eval_key[:16] {key16}')
    pc_present = rec.get('post_certification') is not None
    logs_dir = RUN_LOGS.format(campaign_id=CAMPAIGN_ID[spec['campaign']], key16=key16)
    # every ESSO log of the run (optim_log_esso_node<n>_init.txt + _cycle<k>.txt); runs counted by IPOPT banner
    esso_logs = sorted(glob.glob(os.path.join(_abs(logs_dir), 'optim_log_esso_node*.txt')))
    n_esso_runs = 0
    for p in esso_logs:
        with open(p, errors='replace') as handle:
            n_esso_runs += handle.read().count('This is Ipopt version')
    recovery_logs = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(_abs(logs_dir), '*recovery*')))
    sim = SimpleNamespace(years={y: v for y, v in case['Years'].items()}, days=dict(case['Days']),
                          discount_factor=case['DiscountFactor'])
    blocks = {}
    for kind, node, net, y, d in block_list(case):
        key = f'TSO|{y}|{d}' if kind == 'TSO' else f'DSO|{node}|{y}|{d}'
        rel = os.path.join(logs_dir, f'optim_log_{net}_{y}_{d}.log')
        runs = parse_log(rel)
        ident = identify(runs, K, pc_present)
        t = runs[K] if len(runs) > K else None          # solve K + 1 (1-based)
        last = runs[-1] if runs else None
        if t is None or None in (t['n_pairs'], t['mu_last'], t['obj_scale'], t['compl_unscaled_final'],
                                 t['objective_unscaled_final']):
            raise SystemExit(f'{label} {key}: terminal solve K + 1 = {K + 1} absent or incomplete in {rel} '
                             f'({len(runs)} runs)')
        c = comp[key]
        s = settl[key]
        w = c['admm_block_weight']
        w_prod = srp._get_admm_block_weight(sim, y, d)
        w_settl = s['interface_settlement_weighted'] / s['interface_settlement_unweighted']
        base = sum(c['unweighted'][n] for n in BASE_COMPONENTS) + s['interface_settlement_unweighted']
        floor, at_floor = _floor(t) if t else (None, None)
        g = w * t['n_pairs'] * t['mu_last'] / t['obj_scale']
        g_up = w * t['n_pairs'] * t['compl_unscaled_final']
        blocks[key] = {
            'agent': kind, 'node': node, 'network': net, 'year': y, 'day': d, 'log': rel, 'sha256_at_read': _sha(rel),
            'mtime_utc': datetime.fromtimestamp(os.path.getmtime(_abs(rel)), timezone.utc).isoformat(),
            **ident, 'terminal_admm_solve': t, 'last_solve_is_terminal': last is t,
            'weight': w, 'weight_production_fn': w_prod, 'weight_from_settlement': w_settl,
            'weight_crosscheck_ok': abs(w_prod / w - 1) <= 1e-9 and abs(w_settl / w - 1) <= 1e-9,
            'base_unweighted_eur_per_rep_day': base,
            'units_ratio_f_over_base': t['objective_unscaled_final'] / base,
            'units_residual_f_minus_base': t['objective_unscaled_final'] - base,
            'mu_floor_pred': floor, 'at_floor': at_floor,
            'central_path_ratio': t['compl_scaled_final'] / t['mu_last'],
            'g_primary_eur': g, 'g_upper_eur': g_up}
    n_net = len(blocks)
    identity = {'n_network_blocks': n_net, 'n_esso_logs': len(esso_logs), 'n_esso_ipopt_runs': n_esso_runs,
                'n_network_ipopt_runs': sum(b['n_solves'] for b in blocks.values()),
                'solve_profile_permitted_solve': rec['solve_profile']['observed']['permitted_solve'],
                'expected': n_net * (K + 1 + (1 if pc_present else 0)) + n_esso_runs,
                'definition': 'permitted_solve == 48 x (K + 1) [network: init + K cycles, no polish] + IPOPT runs in '
                              'every ESSO log (init + cycles); network runs also counted directly'}
    identity['network_runs_equal_48_x_K_plus_1'] = identity['n_network_ipopt_runs'] == n_net * (K + 1)
    identity['holds'] = ((not pc_present) and identity['expected'] == identity['solve_profile_permitted_solve']
                         and identity['network_runs_equal_48_x_K_plus_1'])
    out = {'identity': spec['identity'], 'eval_dir': ed, 'candidate_key': rec['candidate_key'], 'eval_key': rec['eval_key'],
           'campaign_spec_sha256': rec['campaign_spec_sha256'], 'status': rec['status'], 'cycles_run': K,
           'certification_cycle': rec.get('certification_cycle'), 'post_certification_present': pc_present,
           'local_solve_failures': rec.get('local_solve_failures'), 'logs_dir': logs_dir,
           'recovery_logs': recovery_logs, 'solve_count_identity': identity,
           'identification_holds_all_blocks': all(b['identification_holds'] for b in blocks.values()),
           'weights_crosscheck_all': all(b['weight_crosscheck_ok'] for b in blocks.values()),
           'certified_gross_operational_cost': rec['certified_cost'], 'blocks': blocks}
    out['aggregates'] = aggregates(blocks)
    return out


def aggregates(blocks):
    agg = {}
    groups = {'TSO': lambda b: b['agent'] == 'TSO', 'DSO': lambda b: b['agent'] == 'DSO',
              **{f'DSO{n}': (lambda n: lambda b: b['node'] == n)(n) for n in (5, 7, 9)}, 'total': lambda b: True}
    for g, sel in groups.items():
        bl = [b for b in blocks.values() if sel(b)]
        ts = [b['terminal_admm_solve'] for b in bl]
        agg[g] = {'n_blocks': len(bl), 'G_primary_eur': sum(b['g_primary_eur'] for b in bl),
                  'G_upper_eur': sum(b['g_upper_eur'] for b in bl),
                  'obj_scale_min': min(t['obj_scale'] for t in ts), 'obj_scale_max': max(t['obj_scale'] for t in ts),
                  'obj_scale_median': statistics.median(t['obj_scale'] for t in ts),
                  'mu_last_min': min(t['mu_last'] for t in ts), 'mu_last_max': max(t['mu_last'] for t in ts),
                  'n_pairs_min': min(t['n_pairs'] for t in ts), 'n_pairs_max': max(t['n_pairs'] for t in ts),
                  'iterations_median': statistics.median(t['iterations'] for t in ts),
                  'n_at_floor': sum(1 for b in bl if b['at_floor']),
                  'central_path_ratio_median': statistics.median(b['central_path_ratio'] for b in bl),
                  'units_ratio_min': min(b['units_ratio_f_over_base'] for b in bl),
                  'units_ratio_max': max(b['units_ratio_f_over_base'] for b in bl),
                  'units_abs_dev_median': statistics.median(abs(b['units_ratio_f_over_base'] - 1) for b in bl),
                  'exits': sorted({t['exit'] for t in ts})}
    return agg


# ----------------------------------------------------------------------------------------------------------------------
def control_c10(ver):
    """Parser + identification rule on W74's 2x2 TSO logs (layout B)."""
    entries = ver.manifest(W74_MANIFEST)
    ver.against(W74_JSON, W74_MANIFEST, entries)
    w74 = _load(W74_JSON)
    res = {}
    for label in W74_CONTROL_CELLS:
        cell = w74['cells'][label]
        K = cell['cycles_run']
        pc_present = bool(cell['inventory']['post_certification']['present'])
        rows = {r['block']: r for r in w74['w73_solve_attribution'][label]['blocks']}
        s_term, s_last, ok_all = 0.0, 0.0, True
        for block, b in cell['log_blocks'].items():
            if not block.startswith('TSO'):
                continue
            ok_all &= ver.against(b['log'], W74_MANIFEST, entries)
            runs = parse_log(b['log'])
            ident = identify(runs, K, pc_present)
            ok_all &= ident['identification_holds']
            w = rows[block]['weight_multiscenario_terminal']
            t, last = runs[K], runs[-1]
            s_term += w * t['n_pairs'] * t['mu_last'] / t['obj_scale']
            s_last += w * last['n_pairs'] * last['mu_last'] / last['obj_scale']
        res[label] = {'cycles_run': K, 'post_certification_present': pc_present, 'layout_B_identification_all': ok_all,
                      'G_TSO_terminal_rule': s_term, 'G_TSO_last_solve': s_last,
                      'w74_terminal_committed': w74['w73_solve_attribution'][label]['recomputed_on_terminal_admm_solve'],
                      'w73_committed': w74['w73_solve_attribution'][label]['w73_committed_TSO_gap_w_eur']}
    d_term = res['x0_a0p50']['G_TSO_terminal_rule'] - res['n7_4h_e1_a0p50']['G_TSO_terminal_rule']
    d_last = res['x0_a0p50']['G_TSO_last_solve'] - res['n7_4h_e1_a0p50']['G_TSO_last_solve']
    res['difference_terminal_rule'] = d_term
    res['difference_last_solve'] = d_last
    res['reproduces_w74_terminal'] = abs(d_term - W74_TERMINAL_DIFF) <= 1e-9 * abs(W74_TERMINAL_DIFF)
    res['last_solve_reproduces_w73'] = abs(d_last - W73_COMMITTED_DIFF) <= 1e-9 * abs(W73_COMMITTED_DIFF)
    return res


def scan_manifests_for_logs(log_dirs):
    """Scoped negative claim: which git-tracked *manifest*.json files mention any SRP1 run-log directory used here."""
    files = [f for f in _git('ls-files').stdout.splitlines()
             if 'manifest' in os.path.basename(f) and f.endswith('.json') and '.claude/worktrees' not in f]
    hits = {d: [] for d in log_dirs}
    for f in files:
        try:
            with open(_abs(f), errors='replace') as handle:
                txt = handle.read()
        except OSError:
            continue
        for d in log_dirs:
            if d in txt:
                hits[d].append(f)
    return {'scope': 'every git-tracked file whose basename contains "manifest" and ends in .json, at HEAD of the working '
                     'tree (excluding .claude/worktrees); substring search for the run-log directory path',
            'n_manifest_files_scanned': len(files), 'hits': hits}


# ----------------------------------------------------------------------------------------------------------------------
def score_predictions(cells, pa, control):
    x, u, d = cells['x0'], cells['unit'], cells['unit_dup']
    ax, au = x['aggregates'], u['aggregates']
    bar_sum = pa['bar_sum']
    tso_keys = [k for k in x['blocks'] if k.startswith('TSO')]
    lower = sum(1 for k in tso_keys
                if u['blocks'][k]['terminal_admm_solve']['obj_scale'] < x['blocks'][k]['terminal_admm_solve']['obj_scale'])
    all_blocks = [b for c in (x, u, d) for b in c['blocks'].values()]
    fields = ('obj_scale', 'mu_last', 'n_pairs', 'iterations', 'objective_unscaled_final', 'compl_unscaled_final')
    dup_same = all(all(u['blocks'][k]['terminal_admm_solve'][f] == d['blocks'][k]['terminal_admm_solve'][f] for f in fields)
                   for k in u['blocks'])
    dG = ax['total']['G_primary_eur'] - au['total']['G_primary_eur']
    dG_up = ax['total']['G_upper_eur'] - au['total']['G_upper_eur']
    outcomes = {
        'P1_identification': all(c['identification_holds_all_blocks'] and c['solve_count_identity']['holds']
                                 and not c['recovery_logs'] and not c['post_certification_present'] for c in (x, u, d)),
        'P2_units': (all(abs(b['units_ratio_f_over_base'] - 1) <= 0.05 for b in all_blocks)
                     and statistics.median(abs(b['units_ratio_f_over_base'] - 1) for b in all_blocks) <= 0.01),
        'P3_x0_TSO_level': 300.0 <= ax['TSO']['G_primary_eur'] <= 5000.0,
        'P4_unit_TSO_scale': lower >= 9,
        'P5_TSO_sign': au['TSO']['G_primary_eur'] < ax['TSO']['G_primary_eur'],
        'P6_DSO_similar': abs(ax['DSO']['G_primary_eur'] - au['DSO']['G_primary_eur']) <= 0.10 * ax['DSO']['G_primary_eur'],
        'P7_headline': abs(dG) / bar_sum < 0.25,
        'P8_upper_variant': abs(dG_up) / bar_sum < 1.0,
        'P9_duplicate': dup_same,
        'P10_control': bool(control['reproduces_w74_terminal'] and control['last_solve_reproduces_w73']),
    }
    return outcomes, {'n_TSO_blocks_unit_obj_scale_lower': lower, 'n_all_blocks_units_checked': len(all_blocks)}


def phase_a_bars(ver):
    entries = ver.manifest(PHASE_A_MANIFEST)
    ver.against(PHASE_A_TABLES, PHASE_A_MANIFEST, entries)
    t1 = _load(PHASE_A_TABLES)['T1']
    rows = {r['identity']: r for r in t1['rows']}
    keep = ('identity', 'candidate_key', 'eval_dir', 'cycles_run', 'Q_gross_eur', 'bar_eur', 'terminal_gross_step_abs',
            'terminal_gross_step_over_threshold', 'terminal_objective_tolerance', 'window_direction', 'value_eur',
            'F_minus_F0_resolution_bar_eur')
    sel = {lab: {k: rows[CELLS[lab]['identity']].get(k) for k in keep} for lab in CELLS}
    bar_sum = sel['x0']['bar_eur'] + sel['unit']['bar_eur']
    return {'source': PHASE_A_TABLES, 'bar_definition': t1['formulas']['bar'], 'rows': sel, 'bar_sum': bar_sum,
            'bar_sum_equals_table_res_bar': abs(bar_sum - sel['unit']['F_minus_F0_resolution_bar_eur']) <= 1e-6,
            'value_eur_unit': sel['unit']['value_eur']}


def run():
    started = time.time()
    if os.path.exists(_abs(OUT_JSON)):
        raise SystemExit(f'REFUSED: output exists (write-once): {OUT_JSON}')
    ver = Verifier()
    # predictions: committed before this harness ran
    if _sha(PREDICTIONS) != PREDICTIONS_SHA256:
        raise SystemExit('predictions file hash differs from the recorded one')
    ver.tracked_only(PREDICTIONS, f'predictions (committed at {PREDICTIONS_COMMIT}, before the harness)')
    ver.tracked_only(CASE_JSON, 'SRP1 case definition (years, days, discount, DN node mapping)')
    case = _load(CASE_JSON)
    # capture-path assertion BEFORE any analysis: every required input exists
    missing = []
    for lab, spec in CELLS.items():
        for f in EVAL_FILES:
            if not os.path.isfile(_abs(os.path.join(spec['eval_dir'], f))):
                missing.append(os.path.join(spec['eval_dir'], f))
        key16 = os.path.basename(spec['eval_dir'])[:16]
        logs_dir = RUN_LOGS.format(campaign_id=CAMPAIGN_ID[spec['campaign']], key16=key16)
        for kind, node, net, y, d in block_list(case):
            p = os.path.join(logs_dir, f'optim_log_{net}_{y}_{d}.log')
            if not os.path.isfile(_abs(p)):
                missing.append(p)
    for p in (PHASE_A_TABLES, PHASE_A_MANIFEST, W74_JSON, W74_MANIFEST, *CAMPAIGN_MANIFEST.values()):
        if not os.path.isfile(_abs(p)):
            missing.append(p)
    if missing:
        raise SystemExit(f'CAPTURE PATH MISSING ({len(missing)}): {missing[:10]}')
    print(f'[W81] capture paths present for all {len(CELLS)} cells x {len(block_list(case))} blocks', flush=True)

    import shared_resources_planning as srp   # production _get_admm_block_weight; import only, nothing built or solved
    pa = phase_a_bars(ver)
    cells = {lab: cell_analysis(lab, spec, case, ver, srp) for lab, spec in CELLS.items()}
    for lab, c in cells.items():
        a = c['aggregates']
        print(f"[W81] {lab} {c['identity']} K={c['cycles_run']} ident={c['identification_holds_all_blocks']} "
              f"solve-identity={c['solve_count_identity']} recovery={len(c['recovery_logs'])} "
              f"w-xcheck={c['weights_crosscheck_all']} | G_TSO={a['TSO']['G_primary_eur']:.2f} "
              f"G_DSO={a['DSO']['G_primary_eur']:.2f} G_total={a['total']['G_primary_eur']:.2f} "
              f"(upper {a['total']['G_upper_eur']:.2f}) units ratio [{a['total']['units_ratio_min']:.6f}, "
              f"{a['total']['units_ratio_max']:.6f}]", flush=True)
    if ver.failures:
        raise SystemExit(f'INPUT VERIFICATION FAILED: {ver.failures}')
    control = control_c10(ver)
    print(f"[W81] C10 control: terminal-rule diff {control['difference_terminal_rule']:.9f} (W74 {W74_TERMINAL_DIFF}) "
          f"-> {control['reproduces_w74_terminal']}; last-solve diff {control['difference_last_solve']:.9f} "
          f"(W73 {W73_COMMITTED_DIFF}) -> {control['last_solve_reproduces_w73']}", flush=True)
    if ver.failures:
        raise SystemExit(f'INPUT VERIFICATION FAILED (control): {ver.failures}')

    ax, au = cells['x0']['aggregates'], cells['unit']['aggregates']
    diff = {}
    for g in ('TSO', 'DSO', 'DSO5', 'DSO7', 'DSO9', 'total'):
        dp = ax[g]['G_primary_eur'] - au[g]['G_primary_eur']
        du = ax[g]['G_upper_eur'] - au[g]['G_upper_eur']
        diff[g] = {'x0_G_primary': ax[g]['G_primary_eur'], 'unit_G_primary': au[g]['G_primary_eur'],
                   'delta_G_primary': dp, 'delta_G_primary_over_bar_sum': dp / pa['bar_sum'],
                   'x0_G_upper': ax[g]['G_upper_eur'], 'unit_G_upper': au[g]['G_upper_eur'],
                   'delta_G_upper': du, 'delta_G_upper_over_bar_sum': du / pa['bar_sum']}
    R = abs(diff['total']['delta_G_primary']) / pa['bar_sum']
    verdict = {
        'R': R, 'R_upper_variant': abs(diff['total']['delta_G_upper']) / pa['bar_sum'], 'bar_sum': pa['bar_sum'],
        'delta_G_total_primary': diff['total']['delta_G_primary'],
        'delta_G_total_over_value': diff['total']['delta_G_primary'] / pa['value_eur_unit'],
        'value_eur_unit': pa['value_eur_unit'],
        'verdict': ('SUB-RESOLUTION at SRP1' if R < 1 else 'AT/ABOVE RESOLUTION at SRP1'),
        'upper_variant_crosses_1': abs(diff['total']['delta_G_upper']) / pa['bar_sum'] >= 1,
        'rule': _load(PREDICTIONS)['frozen_verdict_rule'],
        'sigma_Q_note': 'sigma_Q (Phase A, 10-18k) is provenance only (Addendum 44 ruling 6) and is not used'}
    outcomes, aux = score_predictions(cells, pa, control)
    preds = _load(PREDICTIONS)['predictions']
    scored = {k: {'statement': v['statement'], 'expected': v['expected'], 'observed': outcomes[k],
                  'result': 'CONFIRMED' if outcomes[k] == v['expected'] else 'REFUTED'} for k, v in preds.items()}
    manifest_scan = scan_manifests_for_logs(sorted({c['logs_dir'] for c in cells.values()}))
    payload = {
        'task': 'P5.15 Addendum 45 item 3 / W81: SRP1 barrier-gap estimate, terminal ADMM solves, x0 vs n7 0.25/1.0, '
                'against Phase A bars (read-only, zero solves)',
        'utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git('rev-parse', 'HEAD').stdout.strip(),
        'script_sha256': _sha(os.path.basename(__file__)),
        'objective_convention': 'Q = certified gross_operational_cost (settlement-excluded, gross of salvage); gap '
                                'figures are EUR of Q (block weight applied); value = Q(0) - Q(unit)',
        'formulas': FORMULAS, 'limitations': LIMITATIONS,
        'claim_scope': {
            'licenses': 'an order-of-magnitude comparison of the estimated interior-point barrier offsets of the two Phase '
                        'A SRP1 cells, TSO and DSO, on their terminal ADMM solves, against the bar-sum',
            'does_not_license': ['a measured change in Q or in value', 'a statement about the post-Addendum-30 (C2 '
                                 'baseline) cells', 'a statement about the 2x2 DSO blocks (not examined here)',
                                 'a statement about the ESSO', 'a statement about any other SRP1 cell'],
            'searched': ['Phase A tables T1 (bars, value)', 'the three evaluation directories\' committed artifacts',
                         'the 144 TSO/DSO IPOPT main logs of the three run directories and their recovery-log globs',
                         'every tracked *manifest*.json for coverage of those logs (manifest_scan)',
                         'W74 JSON + manifest and the 40 2x2 TSO logs for control C10']},
        'phase_a': pa, 'cells': cells, 'differences_x0_minus_unit': diff, 'verdict': verdict,
        'control_C10': control, 'predictions': {'file': PREDICTIONS, 'sha256': PREDICTIONS_SHA256,
                                                'committed_at': PREDICTIONS_COMMIT, 'scored': scored, 'aux': aux},
        'manifest_scan_for_srp1_logs': manifest_scan, 'inputs': ver.records}
    return payload, started


def main_run():
    payload, started, exit_code = None, time.time(), 1
    try:
        payload, started = run()
        exit_code = 0
    except SystemExit as exc:
        print(f'[W81] STOP: {exc}', flush=True)
    except Exception:                                   # noqa: BLE001 -- reported, then verify still runs
        traceback.print_exc()
    finally:
        failures = GUARD.verify(0)
        print(f'[W81] guard counts {dict(GUARD.counts)} verify(0) {failures}', flush=True)
        if payload is not None:
            payload['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': failures}
            payload['wall_s'] = time.time() - started
            os.makedirs(_abs(OUT), exist_ok=True)
            with open(_abs(OUT_JSON), 'x') as handle:
                json.dump(payload, handle, indent=1)
            v = payload['verdict']
            print(f"[W81] VERDICT {v['verdict']}: Delta G_total {v['delta_G_total_primary']:.2f} EUR, bar-sum "
                  f"{v['bar_sum']:.2f}, R = {v['R']:.4f} (upper variant {v['R_upper_variant']:.4f}); "
                  f"Delta G / value = {v['delta_G_total_over_value']:.4f}", flush=True)
            for g, dd in payload['differences_x0_minus_unit'].items():
                print(f"[W81] {g}: x0 {dd['x0_G_primary']:.2f} unit {dd['unit_G_primary']:.2f} delta "
                      f"{dd['delta_G_primary']:.2f} ({dd['delta_G_primary_over_bar_sum']:.4f} x bar-sum) | upper x0 "
                      f"{dd['x0_G_upper']:.2f} unit {dd['unit_G_upper']:.2f} delta {dd['delta_G_upper']:.2f}", flush=True)
            for k, s in payload['predictions']['scored'].items():
                print(f"[W81] {k}: expected {s['expected']} observed {s['observed']} -> {s['result']}", flush=True)
            print(f'[W81] wrote {OUT_JSON}; wall {payload["wall_s"]:.1f}s', flush=True)
        GUARD.uninstall()
        sys.exit(exit_code if not failures else 1)


def main_manifest():
    exit_code = 1
    try:
        if os.path.exists(_abs(OUT_MANIFEST)):
            raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
        out = _load(OUT_JSON)
        m = {OUT_JSON: _sha(OUT_JSON), LAUNCH_LOG: _sha(LAUNCH_LOG), LAUNCH_LOG_R0_FAILED: _sha(LAUNCH_LOG_R0_FAILED),
             os.path.basename(__file__): _sha(os.path.basename(__file__))}
        bad = [] if out['script_sha256'] == m[os.path.basename(__file__)] else [os.path.basename(__file__)]
        for rel, r in out['inputs'].items():
            if _sha(rel) != r['sha256']:
                bad.append(rel)
            m[rel] = r['sha256']
        for c in out['cells'].values():
            for b in c['blocks'].values():      # the SRP1 IPOPT logs: in NO committed manifest; hashed at read time
                if _sha(b['log']) != b['sha256_at_read']:
                    bad.append(b['log'])
                m[b['log']] = b['sha256_at_read']
        if bad:
            raise SystemExit(f'REFUSED: changed since the run: {bad[:5]}')
        with open(_abs(OUT_MANIFEST), 'x') as handle:
            json.dump(m, handle, indent=1)
        print(f'[W81] wrote {OUT_MANIFEST}: {len(m)} entries', flush=True)
        exit_code = 0
    except SystemExit as exc:
        print(f'[W81] STOP: {exc}', flush=True)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
    finally:
        failures = GUARD.verify(0)
        print(f'[W81] guard counts {dict(GUARD.counts)} verify(0) {failures}', flush=True)
        GUARD.uninstall()
        sys.exit(exit_code if not failures else 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--run']:
        main_run()
    elif sys.argv[1:] == ['--manifest']:
        main_manifest()
    else:
        GUARD.verify(0)
        GUARD.uninstall()
        raise SystemExit('usage: --run | --manifest')
