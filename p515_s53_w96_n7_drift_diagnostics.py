"""
P5.15 Addendum 50, Planner task W96 item 2 -- the node-7 drift diagnostics, FROM RECORDS ONLY; the W95 analysis
(p515_s53_w95_x0_drift_diagnostics.py, commit d757394e -- NOT edited) applied to the second cell of the pair, plus the
two-cell drift comparison.

Cell: data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/c82522f470b35b58_n7_4h_e1 (candidate n7_4h_e1,
node 7 = [0.25, 1.0], eval key c82522f4..., candidate_key db77e154...; certified at cycle 69, Q = 842,595,839.5131028
gross, settlement excluded; tail active on cycles 61-69; certification window 60-69).

The pair has finished (no campaign running, no campaign lock). This script:
  * arms p513_solve_profile_guard.SolveProfileGuard(permitted=()) for its whole run and checks verify(0) == [] on
    every exit path (the guard is the only non-stdlib import; it imports pyomo.opt -- no production module, no
    harness, no model, no pickle is loaded; asserted at exit);
  * reads JSON / JSONL records and IPOPT log byte ranges, JSONL streamed line by line;
  * writes only to a NEW directory (refuses to overwrite an existing output file);
  * hashes every input at read time (IPOPT logs are gitignored, so their sha256 is recorded here).

Items (W96 item 2; Addendum 50):
  A  rule ten, as the harness defines it (identical to W95 A).
  B  G6 verification: mu/floor per terminal-round solve; the four terminal error metrics against the tolerances in
     force; the ACCEPTABLE exit (IPOPT acceptable criteria, acceptable_iter consecutive 'A' iterates) and the solve at
     the largest mu/floor explained from their logs; and WHY solves reach the floor -- per solve, the last iterate
     whose barrier parameter is above the floor, with the four optimality tests evaluated on it (the reason IPOPT
     did not stop there), grouped by the objective scaling factor S IPOPT printed.
  C  (i)-(v) per cycle over the window 60-69, as W95 C; supplementary 50-69 (does the drift predate the tail?).
  D  scoring inputs for Addendum 50's predictions (H_row18, H_B, H_C; the node-7 G6 prediction).
  E  the two-cell drift comparison (x = 0 from its committed records): steps at certification, their difference, and
     a PROJECTION -- not a measurement -- of the value if the difference persists.

Run: /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s53_w96_n7_drift_diagnostics.py
"""
import hashlib
import json
import math
import os
import re
import resource
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W96 node-7 records-only diagnostics (never solves)').install()

PAIR_REL = 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair'
EVAL_REL = f'{PAIR_REL}/evals/c82522f470b35b58_n7_4h_e1'
EVAL = os.path.join(REPO, EVAL_REL)
X0_EVAL = os.path.join(REPO, f'{PAIR_REL}/evals/f6e9cd53fdbb8ee8_x0')
W95_JSON = os.path.join(REPO, 'data/SRP1/Results/P515S53/w95_x0_drift/w95_x0_drift_diagnostics.json')
CAMPAIGN_RESULTS = os.path.join(REPO, f'{PAIR_REL}/campaign_results.json')
OUT_REL = 'data/SRP1/Results/P515S53/w96_n7_drift'
OUT = os.path.join(REPO, OUT_REL)
OUT_JSON = os.path.join(OUT, 'w96_n7_drift_diagnostics.json')
OUT_MANIFEST = os.path.join(OUT, 'manifest_sha256.json')

FORBIDDEN_MODULES = ('shared_resources_planning', 'network', 'network_data', 'model_construction_helpers',
                     'p515_s44_campaign_harness', 'uncoordinated_benchmark', 'p515_g_g1_g4_admm_gates',
                     'p515_s53_w90_3x3_campaign', 'energy_storage', 'helper_functions')

CERT_CYCLE = 69
WINDOW = list(range(60, 70))            # the certification window (evaluation_record.bar.window)
TAIL_CYCLES_EXPECTED = list(range(61, 70))
CUM_WINDOWS = ((60, 69), (61, 69), (66, 69))
CHANNELS = ('v', 'pf', 'ess')
EXPECTED_Q = 842595839.5131028
EXPECTED_BAR = 4010.5701702833176
N_BLOCKS = 80
CONTEXT_CYCLES = list(range(50, 60))    # supplementary: the ten cycles before the window
X0_CERT_CYCLE = 72
X0_EXPECTED_Q = 842832534.7623764
X0_TAIL_FIRST = 64
N7_TAIL_FIRST = 61
EXPECTED_VALUE = 236695.2492736578
EXPECTED_RESOLUTION = 17964.945860743523
EXPECTED_I = 317957.0085035586
PROJECTION_HORIZONS = (1, 3, 5, 10, 20, 30)
IPOPT_DEFAULTS = {'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4, 'tol': 1e-8,
                  'acceptable_tol': 1e-6, 'acceptable_iter': 15, 'acceptable_dual_inf_tol': 1e10,
                  'acceptable_constr_viol_tol': 1e-2, 'acceptable_compl_inf_tol': 1e-2,
                  'acceptable_obj_change_tol': 1e20,
                  'mu_linear_decrease_factor': 0.2, 'mu_superlinear_decrease_power': 1.5}
IPOPT_DEFAULTS_SOURCE = ('IPOPT 3.14 options reference (coin-or.github.io/Ipopt/OPTIONS.html): tol 1e-8, '
                         'dual_inf_tol 1, constr_viol_tol 1e-4, compl_inf_tol 1e-4, acceptable_tol 1e-6, '
                         'acceptable_iter 15, acceptable_dual_inf_tol 1e10, acceptable_constr_viol_tol 1e-2, '
                         'acceptable_compl_inf_tol 1e-2, acceptable_obj_change_tol 1e20; tol / acceptable_tol are '
                         'compared with the SCALED NLP error, the other tolerances with UNSCALED quantities')

INPUTS = {}   # rel path -> hash record


# ======================================================================================================================
#  tracked reads (W95's, without the running-cell read guard: the pair has finished)
# ======================================================================================================================
def _hash_whole(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _register(path, sha, size_before, note, extra=None):
    size_after = os.path.getsize(path)
    rec = {'sha256': sha, 'size_bytes': size_before, 'size_unchanged_during_read': size_after == size_before,
           'mtime_utc': datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(),
           'hashed_at_utc': datetime.now(timezone.utc).isoformat(), 'role': note}
    if extra:
        rec.update(extra)
    INPUTS[os.path.relpath(path, REPO)] = rec
    if size_after != size_before:
        raise RuntimeError(f'input changed size while being read: {path}')


def iter_jsonl(path, note):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            if raw.strip():
                yield json.loads(raw)
    _register(path, h.hexdigest(), size, note)


def load_json(path, note, limit=2 * 1024 * 1024):
    size = os.path.getsize(path)
    if size > limit:
        raise RuntimeError(f'{path} is {size} bytes; this script loads only small JSON documents whole')
    with open(path, 'rb') as handle:
        data = handle.read()
    _register(path, hashlib.sha256(data).hexdigest(), size, note)
    return json.loads(data)


def iter_text_lines(path, note):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            yield raw.decode('utf-8', errors='replace')
    _register(path, h.hexdigest(), size, note)


def read_log_segment(path, start, end, note):
    size = os.path.getsize(path)
    with open(path, 'rb') as handle:
        handle.seek(start)
        seg = handle.read(end - start)
    rel = os.path.relpath(path, REPO)
    if rel not in INPUTS:
        _register(path, _hash_whole(path), size, note, {'gitignored': True, 'segments_read': []})
    INPUTS[rel]['segments_read'].append({'bytes': [start, end], 'sha256': hashlib.sha256(seg).hexdigest()})
    INPUTS[rel]['file_size_equals_last_segment_end'] = (size == end)
    return seg.decode('utf-8', errors='replace')


# ======================================================================================================================
#  IPOPT log parsing (W95's, plus per-iterate metrics and the acceptable flags)
# ======================================================================================================================
_OPT_RE = re.compile(r'(?:^|(?<=\s))([A-Za-z_][A-Za-z0-9_]*) = (\S+)', re.M)
_METRIC_RE = re.compile(r'^(Objective|Dual infeasibility|Constraint violation|Variable bound violation|'
                        r'Complementarity|Overall NLP error)\.*:\s+(\S+)\s+(\S+)\s*$', re.M)
_ITER_RE = re.compile(r'Number of Iterations\.+: (\d+)')
_MU_RE = re.compile(r'Current barrier parameter mu = (\S+)')
_EXIT_RE = re.compile(r'EXIT: (.*)')
_NLP_VALUES_RE = re.compile(r'\*\*\*Current NLP Values for Iteration (\d+):')
_ITER_LINE_RE = re.compile(r'^\s*(\d+)[rR]?\s+[-+]?\d\.\d+e[-+]\d+\s+\S+\s+\S+\s+[-+]?\d+\.\d\s', re.M)
_BANNER = 'This is Ipopt version'


def parse_options(text):
    at = text.find('List of options:')
    ban = text.find(_BANNER)
    if at < 0 or (0 <= ban < at):
        return None, None
    block = text[at:ban if ban >= 0 else len(text)]
    opts, merged = {}, []
    for line in block.splitlines():
        found = _OPT_RE.findall(line)
        if len(found) > 1:
            merged.append(line[:120] + ('...' if len(line) > 120 else ''))
        for name, value in found:
            opts[name] = value
    return opts, merged


def _metrics_of(block):
    metrics = {}
    for name, scaled, unscaled in _METRIC_RE.findall(block):
        key = name.lower().replace(' ', '_')
        metrics[key] = {'scaled': float(scaled), 'unscaled': float(unscaled)}
    return metrics


def parse_terminal_metrics(text):
    it = list(_ITER_RE.finditer(text))
    if not it:
        return None, None
    return int(it[-1].group(1)), _metrics_of(text[it[-1].start():])


def parse_iterates(text):
    """Per iterate k: the barrier parameter in force when iterate k was evaluated (the last 'Current barrier
    parameter mu' line before its '***Current NLP Values for Iteration k' block) and that block's (scaled, unscaled)
    metrics. IPOPT at print level >= 6 prints the block for every iterate."""
    marks = list(_NLP_VALUES_RE.finditer(text))
    mu_marks = [(m.start(), float(m.group(1))) for m in _MU_RE.finditer(text)]
    out, j, current_mu = [], 0, None
    for i, m in enumerate(marks):
        while j < len(mu_marks) and mu_marks[j][0] < m.start():
            current_mu = mu_marks[j][1]
            j += 1
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        block_end = text.find('Overall NLP error', m.end(), end)
        block_end = text.find('\n', block_end) if block_end >= 0 else end
        out.append({'iter': int(m.group(1)), 'mu': current_mu, 'metrics': _metrics_of(text[m.end():block_end])})
    return out


def acceptable_flags(text):
    """The iteration-summary table: iteration number -> True when the line carries IPOPT's trailing 'A' flag (the
    iterate met the acceptable criteria)."""
    flags = {}
    for line in text.splitlines():
        m = _ITER_LINE_RE.match(line)
        if m:
            flags[int(m.group(1))] = line.rstrip().endswith(' A')
    return flags


def _num(opts, name):
    if opts is not None and name in opts:
        return float(opts[name]), 'options_list'
    return IPOPT_DEFAULTS.get(name), 'ipopt_default'


def optimality_tests(metrics, tols):
    if not metrics or not all(m in metrics for m in ('dual_infeasibility', 'constraint_violation', 'complementarity',
                                                     'overall_nlp_error')):
        return None
    t = {
        'overall_nlp_error_scaled_le_tol': metrics['overall_nlp_error']['scaled'] <= tols['tol'],
        'dual_infeasibility_unscaled_le_dual_inf_tol': metrics['dual_infeasibility']['unscaled'] <= tols['dual_inf_tol'],
        'constraint_violation_unscaled_le_constr_viol_tol':
            metrics['constraint_violation']['unscaled'] <= tols['constr_viol_tol'],
        'complementarity_unscaled_le_compl_inf_tol': metrics['complementarity']['unscaled'] <= tols['compl_inf_tol'],
    }
    t['all_four_pass'] = all(t.values())
    return t


def acceptable_tests(metrics, tols):
    if not metrics:
        return None
    t = {
        'overall_nlp_error_scaled_le_acceptable_tol': metrics['overall_nlp_error']['scaled'] <= tols['acceptable_tol'],
        'dual_infeasibility_unscaled_le_acceptable_dual_inf_tol':
            metrics['dual_infeasibility']['unscaled'] <= tols['acceptable_dual_inf_tol'],
        'constraint_violation_unscaled_le_acceptable_constr_viol_tol':
            metrics['constraint_violation']['unscaled'] <= tols['acceptable_constr_viol_tol'],
        'complementarity_unscaled_le_acceptable_compl_inf_tol':
            metrics['complementarity']['unscaled'] <= tols['acceptable_compl_inf_tol'],
    }
    t['all_pass'] = all(t.values())
    return t


def metric_ratios(metrics, tols):
    if not metrics or 'overall_nlp_error' not in metrics:
        return None
    return {
        'overall_nlp_error_scaled_over_tol': metrics['overall_nlp_error']['scaled'] / tols['tol'],
        'dual_infeasibility_unscaled_over_dual_inf_tol': metrics['dual_infeasibility']['unscaled'] / tols['dual_inf_tol'],
        'constraint_violation_unscaled_over_constr_viol_tol':
            metrics['constraint_violation']['unscaled'] / tols['constr_viol_tol'],
        'complementarity_unscaled_over_compl_inf_tol':
            metrics['complementarity']['unscaled'] / tols['compl_inf_tol'],
    }


# ======================================================================================================================
#  helpers (W95's)
# ======================================================================================================================
def _q(values, p):
    s = sorted(values)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {'n': len(v), 'min': min(v), 'p25': _q(v, 0.25), 'median': statistics.median(v), 'p75': _q(v, 0.75),
            'max': max(v), 'mean': statistics.fmean(v), 'sum': sum(v)}


def monotone_updates_to_floor(mu_final, floor, kappa, theta, limit=50):
    if mu_final is None or floor is None:
        return None
    mu, n = mu_final, 0
    while mu > floor * (1.0 + 1e-3) and n < limit:
        mu = max(floor, min(kappa * mu, mu ** theta))
        n += 1
    return n


def block_key_from_record(r):
    agent = r['agent']
    if agent == 'TSO':
        return f"TSO|{r['year']}|{r['day']}"
    return f"DSO{agent[3:]}|{r['year']}|{r['day']}"


def block_key_from_sidecar(e):
    if e['agent'] == 'TSO':
        return f"TSO|{e['year']}|{e['day']}"
    return f"DSO{e['node_id']}|{e['year']}|{e['day']}"


def descending_run_start(dQ, last):
    k = last
    while k - 1 in dQ and dQ[k - 1] < 0:
        k -= 1
    return k


# ======================================================================================================================
#  main
# ======================================================================================================================
def analyse(out):
    # ---------------------------------------------------------------- records
    ev = load_json(os.path.join(EVAL, 'evaluation_record.json'), 'evaluation record (certification, rule ten as recorded)')
    pcr = {r['cycle']: r for r in iter_jsonl(os.path.join(EVAL, 'per_cycle_record.jsonl'),
                                             'per-cycle trajectory + response fields (Q, residual ratios, row-18 aggregates)')}
    g = load_json(os.path.join(EVAL, 'g_s39_D.json'), 'full stage report: cycle_trajectory with Boyd r/s/norms per channel')
    traj = {r['cycle']: r for r in g['cycle_trajectory']}
    aa = {r['cycle']: r for r in iter_jsonl(os.path.join(EVAL, 'aa_per_cycle.jsonl'), 'Anderson acceleration per cycle')}
    sidecar = {r['cycle']: r for r in iter_jsonl(os.path.join(EVAL, 'recourse_jump_sidecar_baseline.jsonl'),
                                                 'per-cycle block recourse deltas (top 10 by |delta|) + block totals')}
    tail_state = load_json(os.path.join(EVAL, 'convergence_depth_tail_state.json'), 'tail state per cycle')
    failures = list(iter_jsonl(os.path.join(EVAL, 'network_failures_s39_D.jsonl'),
                               'network-block failure events (primary maxIterations -> recovery)'))
    records = defaultdict(list)
    for r in iter_jsonl(os.path.join(EVAL, 'network_ipopt_solve_records.jsonl'),
                        'production per-solve IPOPT records (round = ADMM cycle; round 0 = initialisation)'):
        records[r['round']].append(r)

    assert ev['certification_cycle'] == CERT_CYCLE and ev['status'] == 'certified'
    assert ev['certified_cost'] == EXPECTED_Q and ev['bar']['value'] == EXPECTED_BAR
    assert [w['cycle'] for w in ev['bar']['window']] == WINDOW
    assert sorted(pcr) == list(range(1, CERT_CYCLE + 1)) == sorted(traj)
    cycles_tail = [r['cycle'] for r in tail_state['per_cycle'] if r['active']]
    assert cycles_tail == TAIL_CYCLES_EXPECTED, cycles_tail
    for k in range(1, CERT_CYCLE + 1):
        assert pcr[k]['gross_operational_cost'] == traj[k]['gross_operational_cost']
    Q = {k: pcr[k]['gross_operational_cost'] for k in pcr}
    dQ = {k: Q[k] - Q[k - 1] for k in range(2, CERT_CYCLE + 1)}
    out['identity'] = {'candidate_label': ev['candidate_label'], 'candidate_canonical': ev['candidate_canonical'],
                       'candidate_key': ev['candidate_key'], 'eval_key': ev['eval_key'],
                       'campaign_spec_sha256': ev['campaign_spec_sha256'], 'certified_cost': ev['certified_cost'],
                       'bar': ev['bar']['value'], 'tail_active_cycles': cycles_tail}

    # ---------------------------------------------------------------- A. rule ten
    last = traj[CERT_CYCLE]
    step = abs(Q[CERT_CYCLE] - Q[CERT_CYCLE - 1])
    tol_re = max(last['objective_absolute_tolerance'],
                 last['objective_relative_tolerance'] * max(abs(Q[CERT_CYCLE]), abs(Q[CERT_CYCLE - 1]), 1.0))
    ratio = last['objective_change_abs'] / last['objective_tolerance']
    out['A_rule_ten'] = {
        'definition': ('rule_ten_terminal_step_over_threshold = objective_change_abs[last] / objective_tolerance[last] '
                       '(p515_g_g1_g4_admm_gates.py:1351-1353; recorded as evaluation_record.rule_ten by '
                       'p515_s44_campaign_harness.py:1848-1854); objective_change_abs = |Q_k - Q_(k-1)|, '
                       'objective_tolerance = max(tol.abs, tol.rel * max(|Q_k|, |Q_(k-1)|, 1)) '
                       '(shared_resources_planning.py:3340-3343)'),
        'terminal_cycle': CERT_CYCLE, 'Q_terminal': Q[CERT_CYCLE], 'Q_previous': Q[CERT_CYCLE - 1],
        'terminal_step_signed': dQ[CERT_CYCLE],
        'terminal_step_abs_recomputed_from_Q': step, 'terminal_step_abs_recorded': last['objective_change_abs'],
        'tol_abs': last['objective_absolute_tolerance'], 'tol_rel': last['objective_relative_tolerance'],
        'threshold_recomputed': tol_re, 'threshold_recorded': last['objective_tolerance'],
        'ratio': ratio,
        'ratio_recorded_in_evaluation_record': ev['rule_ten']['terminal_step_over_threshold'],
        'ratio_equals_record_bitwise': ratio == ev['rule_ten']['terminal_step_over_threshold'],
        'threshold_recompute_rel_diff': abs(tol_re - last['objective_tolerance']) / last['objective_tolerance'],
        'window_series': {k: {'dQ_signed': dQ[k], 'objective_change_abs': traj[k]['objective_change_abs'],
                              'objective_tolerance': traj[k]['objective_tolerance'],
                              'ratio': traj[k]['objective_change_abs'] / traj[k]['objective_tolerance'],
                              'step_over_bar': abs(dQ[k]) / EXPECTED_BAR} for k in WINDOW},
        'note': f'step_over_bar is supplementary (|dQ_k| / the certified bar {EXPECTED_BAR}), not rule ten',
    }

    # ---------------------------------------------------------------- B. G6 verification
    term = records[CERT_CYCLE]
    assert len(term) == N_BLOCKS, len(term)
    assert all(r['attempt'] == 'primary' for r in term)
    keys = [block_key_from_record(r) for r in term]
    assert len(set(keys)) == N_BLOCKS
    band = Counter()
    for r in term:
        x = r['mu_over_floor']
        if x is None:
            band['none'] += 1
        elif r['floor_status'] == 'at':
            band['at_floor (|ratio-1|<=1e-3)'] += 1
        elif r['floor_status'] == 'below':
            band['below_floor'] += 1
        elif x <= 5.0:
            band['(1, 5]'] += 1
        else:
            band['> 5'] += 1
    per_block, fails, tol_seen, update_counts = [], [], Counter(), Counter()
    merged_lines_total = 0
    for r in sorted(term, key=block_key_from_record):
        text = read_log_segment(r['log_path'], r['log_bytes'][0], r['log_bytes'][1],
                                'IPOPT log (gitignored) -- terminal-round attempt byte range read')
        opts, merged = parse_options(text)
        merged_lines_total += len(merged or [])
        n_it, metrics = parse_terminal_metrics(text)
        mus = _MU_RE.findall(text)
        exits = _EXIT_RE.findall(text)
        tols = {}
        for name in ('tol', 'dual_inf_tol', 'constr_viol_tol', 'compl_inf_tol', 'acceptable_tol', 'acceptable_iter',
                     'acceptable_dual_inf_tol', 'acceptable_constr_viol_tol', 'acceptable_compl_inf_tol',
                     'acceptable_obj_change_tol'):
            tols[name], src = _num(opts, name)
            tols[f'{name}_source'] = src
        tol_seen[json.dumps(tols, sort_keys=True)] += 1
        kappa, kappa_src = _num(opts, 'mu_linear_decrease_factor')
        theta, theta_src = _num(opts, 'mu_superlinear_decrease_power')
        n_upd = monotone_updates_to_floor(r['mu_final'], r['mu_floor'], kappa, theta)
        update_counts[n_upd] += 1
        checks = optimality_tests(metrics, tols)
        iterates = parse_iterates(text)
        flags = acceptable_flags(text)
        trailing_a = 0
        for k in sorted(flags, reverse=True):
            if flags[k]:
                trailing_a += 1
            else:
                break
        # the last iterate evaluated with the barrier parameter ABOVE the floor, and the four optimality tests on it
        above = [it for it in iterates if it['mu'] is not None and it['mu'] > r['mu_floor'] * (1.0 + 1e-3)]
        last_above = above[-1] if above else None
        at_floor_its = [it for it in iterates if it['mu'] is not None and abs(it['mu'] / r['mu_floor'] - 1.0) <= 1e-3]
        mu_levels = []
        for it in iterates:
            if it['mu'] is not None and (not mu_levels or mu_levels[-1]['mu'] != it['mu']):
                mu_levels.append({'mu': it['mu'], 'first_iter': it['iter'], 'n_iterates': 0})
            if mu_levels:
                mu_levels[-1]['n_iterates'] += 1
        s = r['obj_scaling_factor']
        compl_ratio_check = (metrics['complementarity']['scaled'] / metrics['complementarity']['unscaled']
                             if metrics and metrics.get('complementarity', {}).get('unscaled') else None)
        consistency = {
            'one_banner': text.count(_BANNER) == 1,
            'iterations_match_record': n_it == r['iterations'],
            'mu_final_matches_record': (float(mus[-1]) == r['mu_final']) if mus else False,
            'exit_matches_record': (exits[-1].strip() == r['exit']) if exits else False,
            'compl_inf_tol_logged_equals_in_force': (opts is not None and 'compl_inf_tol' in opts
                                                     and float(opts['compl_inf_tol']) == r['compl_inf_tol_in_force']),
            'tol_logged_equals_in_force': (opts is not None and 'tol' in opts
                                           and float(opts['tol']) == r['tol_in_force']),
            'last_iterate_parsed_equals_terminal_count': bool(iterates) and iterates[-1]['iter'] == n_it,
            'terminal_metrics_equal_last_iterate_block': bool(iterates) and bool(metrics) and all(
                iterates[-1]['metrics'].get(m) == metrics.get(m) for m in
                ('objective', 'dual_infeasibility', 'constraint_violation', 'complementarity', 'overall_nlp_error')),
        }
        entry = {'block': block_key_from_record(r), 'agent': r['agent'], 'log': os.path.relpath(r['log_path'], REPO),
                 'log_bytes': r['log_bytes'], 'exit': r['exit'], 'iterations': r['iterations'],
                 'mu_final': r['mu_final'], 'mu_floor': r['mu_floor'], 'mu_over_floor': r['mu_over_floor'],
                 'floor_status': r['floor_status'], 'obj_scaling_factor': s,
                 'complementarity_scaled_over_unscaled_at_termination': compl_ratio_check,
                 'mu_update': {'mu_linear_decrease_factor': kappa, 'mu_linear_decrease_factor_source': kappa_src,
                               'mu_superlinear_decrease_power': theta, 'mu_superlinear_decrease_power_source': theta_src,
                               'next_monotone_mu': max(r['mu_floor'], min(kappa * r['mu_final'], r['mu_final'] ** theta)),
                               'monotone_updates_from_mu_final_to_floor': n_upd},
                 'mu_levels_traversed': mu_levels,
                 'tolerances_in_force': tols, 'options_list_parsed': opts, 'options_list_merged_lines': merged,
                 'terminal_metrics': metrics, 'checks': checks,
                 'metric_over_tolerance': metric_ratios(metrics, tols),
                 'acceptable_checks_at_termination': acceptable_tests(metrics, tols),
                 'trailing_acceptable_flagged_iterations': trailing_a,
                 'last_iterate_above_floor': (None if last_above is None else {
                     'iter': last_above['iter'], 'mu': last_above['mu'], 'metrics': last_above['metrics'],
                     'optimality_tests': optimality_tests(last_above['metrics'], tols),
                     'metric_over_tolerance': metric_ratios(last_above['metrics'], tols)}),
                 'n_iterates_at_floor': len(at_floor_its),
                 'consistency_with_production_record': consistency}
        per_block.append(entry)
        if checks is None or not checks['all_four_pass'] or not all(consistency.values()):
            fails.append(entry['block'])
    agg_margin = {}
    for name in ('overall_nlp_error_scaled_over_tol', 'dual_infeasibility_unscaled_over_dual_inf_tol',
                 'constraint_violation_unscaled_over_constr_viol_tol', 'complementarity_unscaled_over_compl_inf_tol'):
        agg_margin[name] = _dist([e['metric_over_tolerance'][name] for e in per_block if e['metric_over_tolerance']])
    # why the floor: grouped by the objective scaling factor
    by_s = defaultdict(list)
    for e in per_block:
        by_s[repr(e['obj_scaling_factor'])].append(e)
    floor_mechanism = {}
    for s, es in sorted(by_s.items()):
        la = [e['last_iterate_above_floor'] for e in es if e['last_iterate_above_floor']]
        floor_mechanism[s] = {
            'n': len(es), 'agents': dict(Counter(e['agent'] for e in es)),
            'mu_floor_values': sorted({e['mu_floor'] for e in es}),
            'floor_status': dict(Counter(e['floor_status'] for e in es)),
            'exits': dict(Counter(e['exit'] for e in es)),
            'last_above_floor_mu_values': dict(Counter(repr(x['mu']) for x in la)),
            'last_above_floor_all_four_pass_count': sum(1 for x in la if x['optimality_tests']
                                                         and x['optimality_tests']['all_four_pass']),
            'last_above_floor_failing_tests': dict(Counter(
                name for x in la if x['optimality_tests']
                for name, ok in x['optimality_tests'].items() if name != 'all_four_pass' and not ok)),
            'last_above_floor_complementarity_unscaled_over_compl_inf_tol': _dist(
                [x['metric_over_tolerance']['complementarity_unscaled_over_compl_inf_tol'] for x in la
                 if x['metric_over_tolerance']]),
            'last_above_floor_complementarity_scaled_over_mu': _dist(
                [x['metrics']['complementarity']['scaled'] / x['mu'] for x in la
                 if x['metrics'].get('complementarity') and x['mu']]),
            'terminal_complementarity_scaled_over_unscaled': _dist(
                [e['complementarity_scaled_over_unscaled_at_termination'] for e in es]),
        }
    special = {}
    acc = [e for e in per_block if e['exit'] != 'Optimal Solution Found.']
    mx = max(per_block, key=lambda e: e['mu_over_floor'] or 0)
    special['non_optimal_exits'] = [{k: e[k] for k in (
        'block', 'exit', 'iterations', 'mu_final', 'mu_floor', 'mu_over_floor', 'obj_scaling_factor',
        'terminal_metrics', 'checks', 'metric_over_tolerance', 'acceptable_checks_at_termination',
        'trailing_acceptable_flagged_iterations', 'mu_levels_traversed', 'log', 'log_bytes')}
        | {'acceptable_iter_in_force': e['tolerances_in_force']['acceptable_iter'],
           'acceptable_tol_in_force': e['tolerances_in_force']['acceptable_tol']} for e in acc]
    special['max_mu_over_floor_block'] = mx['block']
    special['max_mu_over_floor_is_the_acceptable_exit'] = [e['block'] for e in acc] == [mx['block']]
    by_agent_ratio = defaultdict(list)
    for r in term:
        by_agent_ratio[r['agent']].append(r['mu_over_floor'])
    window_mu = {}
    for k in CONTEXT_CYCLES + WINDOW:
        rs = records[k]
        window_mu[k] = {'n': len(rs), 'tail_active': k in cycles_tail,
                        'floor_status': dict(Counter(r['floor_status'] for r in rs)),
                        'floor_status_by_S': {repr(s): dict(Counter(r['floor_status'] for r in rs
                                                                    if r['obj_scaling_factor'] == s))
                                              for s in sorted({r['obj_scaling_factor'] for r in rs})},
                        'mu_over_floor': _dist([r['mu_over_floor'] for r in rs]),
                        'exits': dict(Counter(r['exit'] for r in rs)),
                        'non_optimal': [block_key_from_record(r) for r in rs if r['exit'] != 'Optimal Solution Found.']}
    s_by_round = {}
    for k in sorted(records):
        for r in records[k]:
            s_by_round.setdefault(r['agent'], defaultdict(set))[k // 10 * 10].add(r['obj_scaling_factor'])
    out['B_g6_verification'] = {
        'terminal_round': CERT_CYCLE, 'n_solves': len(term),
        'attempts': dict(Counter(r['attempt'] for r in term)),
        'final_accepted_attempt_rule': ('every block has exactly one attempt (primary) in the terminal round, so the '
                                        'final accepted attempt is that primary'),
        'exits': dict(Counter(r['exit'] for r in term)),
        'mu_over_floor_distribution': _dist([r['mu_over_floor'] for r in term]),
        'mu_over_floor_by_agent': {a: _dist(v) for a, v in sorted(by_agent_ratio.items())},
        'mu_over_floor_band_counts': dict(band),
        'at_floor_blocks': sorted(block_key_from_record(r) for r in term if r['floor_status'] == 'at'),
        'at_floor_by_agent': dict(Counter(r['agent'] for r in term if r['floor_status'] == 'at')),
        'monotone_updates_from_mu_final_to_floor_counts': {str(k): v for k, v in sorted(update_counts.items(),
                                                                                         key=lambda kv: str(kv[0]))},
        'mu_final_values_seen': dict(Counter(repr(r['mu_final']) for r in term)),
        'mu_floor_definition': 'network.py:627 -- min(tol, compl_inf_tol * obj_scaling_factor) / 11',
        'termination_test': ('IPOPT: scaled overall NLP error <= tol AND unscaled dual inf <= dual_inf_tol AND '
                             'unscaled constraint violation <= constr_viol_tol AND unscaled complementarity <= '
                             'compl_inf_tol; ACCEPTABLE termination: the acceptable_* analogues hold for '
                             'acceptable_iter consecutive iterates (the trailing A flags of the iteration table)'),
        'ipopt_defaults_used_when_not_listed': IPOPT_DEFAULTS, 'ipopt_defaults_source': IPOPT_DEFAULTS_SOURCE,
        'tolerance_sets_seen': dict(tol_seen),
        'options_list_merged_physical_lines_total': merged_lines_total,
        'metric_over_tolerance_distribution': agg_margin,
        'blocks_failing_any_check_or_consistency': fails,
        'floor_mechanism_by_obj_scaling_factor': floor_mechanism,
        'special_solves': special,
        'obj_scaling_factor_by_agent_and_decade_of_rounds': {a: {str(d): sorted(v) if len(v) <= 3 else
                                                                 [min(v), max(v), f'{len(v)} values']
                                                                 for d, v in sorted(dd.items())}
                                                             for a, dd in sorted(s_by_round.items())},
        'per_block': per_block,
        'mu_per_round_cycles_50_69_supplementary': window_mu,
        'network_failure_events': [{k: f.get(k) for k in ('cycle', 'agent', 'node_id', 'network_name', 'year', 'day',
                                                           'primary_termination', 'recovery_attempted')}
                                   for f in failures],
    }

    # ---------------------------------------------------------------- C(i). per-block dQ decomposition (W95 C(i))
    ci = {}
    for k in WINDOW:
        sc = sidecar[k]
        assert sc['error'] is None
        total = sc['block_total_current'] - sc['block_total_previous']
        top = sc['block_deltas']
        top_sum = sum(e['delta'] for e in top)
        top_abs = sum(e['abs_delta'] for e in top)
        bound = min(e['abs_delta'] for e in top)
        rem = total - top_sum
        n_unlisted = N_BLOCKS - len(top)
        lower_abs_total = top_abs + abs(rem)
        upper_abs_total = top_abs + n_unlisted * bound
        assert len(records[k]) == N_BLOCKS and all(r['attempt'] == 'primary' for r in records[k])
        rounds = {block_key_from_record(r): r for r in records[k]}
        entries = []
        for e in top:
            bk = block_key_from_sidecar(e)
            rr = rounds.get(bk)
            entries.append({'block': bk, 'delta': e['delta'], 'previous': e['previous'], 'current': e['current'],
                            'iterations_this_round': rr['iterations'] if rr else None,
                            'mu_over_floor_this_round': rr['mu_over_floor'] if rr else None,
                            'exit_this_round': rr['exit'] if rr else None})
        comp = Counter(x['block'].split('|')[0] for x in entries)
        top3_abs = sum(sorted((e['abs_delta'] for e in top), reverse=True)[:3])
        oc = sc['objective_component_block_deltas'] or []
        classified = {e['block_key']: e['delta'] for e in oc if e['component'] == 'classified_total'}
        for x, e in zip(entries, top):
            tup = str((e['agent'], e['node_id'], int(e['year']), e['day']))
            x['classified_total_delta_if_listed'] = classified.get(tup)
            x['q_delta_minus_classified_delta_if_listed'] = (e['delta'] - classified[tup]) if tup in classified else None
        dso_it_all = [r['iterations'] for r in records[k] if r['agent'] != 'TSO']
        dso_it_top = [x['iterations_this_round'] for x in entries if not x['block'].startswith('TSO')]
        ci[k] = {
            'dQ_from_block_totals': total, 'dQ_from_per_cycle_record': dQ[k],
            'block_total_current_minus_Q': sc['block_total_current'] - Q[k],
            'top10': entries, 'top10_signed_sum': top_sum, 'top10_abs_sum': top_abs,
            'top10_signed_share_of_dQ': top_sum / total if total else None,
            'remainder_70_unlisted_blocks_signed': rem,
            'remainder_share_of_dQ': rem / total if total else None,
            'unlisted_block_abs_delta_upper_bound': bound,
            'min_unlisted_blocks_contributing': math.ceil(abs(rem) / bound) if bound > 0 else None,
            'sum_abs_block_delta_bounds': [lower_abs_total, upper_abs_total],
            'cancellation_ratio_bounds_abs_dQ_over_sum_abs': [abs(total) / upper_abs_total, abs(total) / lower_abs_total],
            'top3_share_of_sum_abs_bounds': [top3_abs / upper_abs_total, top3_abs / lower_abs_total],
            'top10_composition_by_agent': dict(comp),
            'top10_sign_counts': {'negative': sum(1 for e in top if e['delta'] < 0),
                                  'positive': sum(1 for e in top if e['delta'] > 0)},
            'median_iterations_all_80_this_round': statistics.median([r['iterations'] for r in records[k]]),
            'median_iterations_top10_this_round': statistics.median(
                [x['iterations_this_round'] for x in entries if x['iterations_this_round'] is not None]),
            'median_iterations_all_60_dso_this_round': statistics.median(dso_it_all),
            'median_iterations_dso_in_top10_this_round': statistics.median(dso_it_top) if dso_it_top else None,
            'dso_in_top10_iteration_percentile_ranks': [sum(1 for v in dso_it_all if v < it) / len(dso_it_all)
                                                        for it in dso_it_top],
            'objective_component_top10': [{'block': e['block_key'], 'component': e['component'], 'delta': e['delta']}
                                          for e in oc],
        }
    recur = defaultdict(lambda: {'n_cycles_in_top10': 0, 'sum_listed_delta': 0.0, 'cycles': []})
    for k in WINDOW:
        for e in ci[k]['top10']:
            recur[e['block']]['n_cycles_in_top10'] += 1
            recur[e['block']]['sum_listed_delta'] += e['delta']
            recur[e['block']]['cycles'].append(k)
    recur_sorted = sorted(recur.items(), key=lambda kv: (-kv[1]['n_cycles_in_top10'], kv[1]['sum_listed_delta']))
    out['C_i_per_block_dQ'] = {
        'recorded': ('recourse_jump_sidecar_baseline.jsonl (p515_g_g1_g4_admm_gates.py:2932-3016): per cycle, the '
                     'block totals (sum over all 80 blocks = Q) and the TOP 10 blocks by |delta| only; objective '
                     'components top 10 likewise ("unclassified" is NOT a Q component)'),
        'not_recoverable_from_records': ('the individual deltas of the 70 blocks outside each cycle\'s top 10; '
                                         'bounded here (each <= the 10th |delta|, net = remainder)'),
        'per_cycle': ci,
        'blocks_recurring_in_window_top10': dict(recur_sorted),
    }

    # ---------------------------------------------------------------- C(ii). consensus / dual movement (W95 C(ii))
    prox, cyc = {}, None
    prox_re = re.compile(r'\[TSO PROX\] cycle=(\d+) \| updated_blocks=(\d+) \| held_failed_blocks=(\d+) \| '
                         r'V max=(\S+) \| PF max=(\S+) \| ESS max=(\S+)')
    loc_re = re.compile(r'\[TSO PROX\]\[(V|PF|ESS) MAX\] (.*)$')
    for line in iter_text_lines(os.path.join(EVAL, 'stdout_s39_D.log'),
                                'production stdout: [TSO PROX] per-cycle max normalized TSO interface movement'):
        m = prox_re.search(line)
        if m:
            cyc = int(m.group(1))
            prox[cyc] = {'updated_blocks': int(m.group(2)), 'held_failed_blocks': int(m.group(3)),
                         'v_max_normalized': float(m.group(4)), 'pf_max_normalized': float(m.group(5)),
                         'ess_max_normalized': float(m.group(6))}
            continue
        m = loc_re.search(line)
        if m and cyc in prox:
            prox[cyc][f'{m.group(1).lower()}_max_at'] = m.group(2).strip()
    cii = {}
    for k in WINDOW:
        t, tp = traj[k], traj[k - 1]
        row = {'aa_action': aa[k]['aa_action'], 'aa_accepted': aa[k]['aa_accepted'],
               'tso_prox_inf_norm_movement': prox.get(k)}
        for c in CHANNELS:
            rho = t[f'rho_{c}_before']
            gamma = t[f'gamma_{c}_before']
            s = t[f'boyd_{c}_s']
            row[c] = {
                'rho': rho, 'gamma': gamma, 'r_l2': t[f'boyd_{c}_r'], 's_l2': s,
                'dz_tso_l2_normalized_eq_s_over_rho': (s / rho) if (rho and gamma == 0.0) else None,
                'dy_dso_l2_normalized_eq_rho_times_r': (rho * t[f'boyd_{c}_r']) if c in ('v', 'pf') else None,
                'norm_x': t[f'boyd_{c}_norm_x'], 'norm_z': t[f'boyd_{c}_norm_z'], 'norm_y': t[f'boyd_{c}_norm_y'],
                'norm_y_change_vs_previous_cycle': t[f'boyd_{c}_norm_y'] - tp[f'boyd_{c}_norm_y'],
                'legacy_primal_max': t.get(f'primal_{c}'), 'legacy_dual_max': t.get(f'dual_{c}'),
            }
        row['worst_v_primal'] = {x: t.get(f'worst_v_primal_{x}') for x in ('node', 'year', 'day', 'period', 'difference')}
        row['worst_pf_primal'] = {x: t.get(f'worst_pf_primal_{x}') for x in ('node', 'year', 'day', 'period', 'type',
                                                                               'difference')}
        row['worst_pf_dual'] = {x: t.get(f'worst_pf_dual_{x}') for x in ('agent', 'node', 'year', 'day', 'period',
                                                                           'type', 'change')}
        cii[k] = row
    out['C_ii_consensus_and_dual_movement'] = {
        'recorded': ('g_s39_D.json cycle_trajectory: Boyd r, s, ||x||, ||z||, ||y|| per channel; worst-entry locations; '
                     'stdout [TSO PROX] max normalized TSO-side movement per channel'),
        'derived': ('||dz_TSO||_2 = s / rho when gamma = 0; ||dy_DSO||_2 = rho * r for V and PF -- exact only when '
                    'Anderson acceleration did not act that cycle (aa_action recorded per row)'),
        'not_recoverable_from_records': 'per-entry consensus / dual vectors per cycle; per-block dual movement',
        'per_cycle': cii,
    }

    # ---------------------------------------------------------------- C(iii). residual margins
    ciii = {}
    for k in WINDOW:
        t = traj[k]
        row = {}
        for c in CHANNELS:
            row[c] = {'primal_ratio': t[f'boyd_{c}_primal_ratio'], 'dual_ratio': t[f'boyd_{c}_dual_ratio'],
                      'eps_pri': t[f'boyd_{c}_eps_pri'], 'eps_dual': t[f'boyd_{c}_eps_dual'],
                      'channel_pass': t[f'boyd_{c}_channel_pass']}
        binding = max(((c, kind, row[c][f'{kind}_ratio']) for c in CHANNELS for kind in ('primal', 'dual')),
                      key=lambda x: x[2])
        row['binding'] = {'channel': binding[0], 'kind': binding[1], 'ratio': binding[2]}
        row['objective_step_ratio'] = traj[k]['objective_change_abs'] / traj[k]['objective_tolerance']
        row['boyd_all_pass'] = t['boyd_all_pass']
        ciii[k] = row
    out['C_iii_residual_margins'] = {
        'recorded': 'per_cycle_record / cycle_trajectory boyd_{c}_{primal,dual}_ratio = r/eps_pri, s/eps_dual',
        'per_cycle': ciii,
    }

    # ---------------------------------------------------------------- C(iv). row-18 terms
    fields = ('row18_charge_weighted', 'E_abs_d_p_mwh_weighted', 'E_abs_d_p_mwh_unweighted',
              'sum_omega_d2_p_mw2h_weighted', 'market_part_mw2h_weighted', 'operation_part_mw2h_weighted',
              'covariance_dso_weighted', 'curtailed_res_dso_mwh_weighted', 'curtailed_res_tso_mwh_weighted',
              'max_abs_d_p_mw')
    civ = {}
    for k in WINDOW:
        r, rp = pcr[k], pcr[k - 1]
        assert r['response_captured'] and rp['response_captured']
        row = {'dQ': dQ[k]}
        for f in fields:
            row[f] = r[f]
            row[f'delta_{f}'] = r[f] - rp[f]
        row['share_of_dQ_row18_charge'] = row['delta_row18_charge_weighted'] / dQ[k] if dQ[k] else None
        row['share_of_dQ_covariance_dso'] = row['delta_covariance_dso_weighted'] / dQ[k] if dQ[k] else None
        row['dQ_minus_row18_minus_cov_dso'] = dQ[k] - row['delta_row18_charge_weighted'] - row['delta_covariance_dso_weighted']
        civ[k] = row
    cum = {}
    for a, b in CUM_WINDOWS:
        c = {'dQ': Q[b] - Q[a - 1],
             'd_row18_charge_weighted': pcr[b]['row18_charge_weighted'] - pcr[a - 1]['row18_charge_weighted'],
             'd_covariance_dso_weighted': pcr[b]['covariance_dso_weighted'] - pcr[a - 1]['covariance_dso_weighted'],
             'd_E_abs_d_p_mwh_weighted': pcr[b]['E_abs_d_p_mwh_weighted'] - pcr[a - 1]['E_abs_d_p_mwh_weighted']}
        c['row18_share_of_dQ'] = c['d_row18_charge_weighted'] / c['dQ'] if c['dQ'] else None
        cum[f'{a}-{b}'] = c
    out['C_iv_row18_terms'] = {
        'recorded': 'per_cycle_record response fields (as W95 C(iv))',
        'not_recoverable_from_records': ('per-block row-18 charge / deviation per cycle; the Q leg |d_q| per cycle; the '
                                         'TSO settlement deviation part per cycle'),
        'per_cycle': civ, 'cumulative': cum,
    }

    # ---------------------------------------------------------------- C(v). iterations and mu per cycle
    cv = {}
    for k in WINDOW:
        rs = records[k]
        by = defaultdict(list)
        for r in rs:
            by['TSO' if r['agent'] == 'TSO' else 'DSO'].append(r)
        cv[k] = {
            'compl_inf_tol_in_force': sorted({(r['agent'], r['compl_inf_tol_in_force']) for r in rs}),
            'tail_active': k in cycles_tail,
            'iterations_all': _dist([r['iterations'] for r in rs]),
            'iterations_tso': _dist([r['iterations'] for r in by['TSO']]),
            'iterations_dso': _dist([r['iterations'] for r in by['DSO']]),
            'iterations_dso7': _dist([r['iterations'] for r in rs if r['agent'] == 'DSO7']),
            'mu_final_dso': _dist([r['mu_final'] for r in by['DSO']]),
            'mu_final_tso': _dist([r['mu_final'] for r in by['TSO']]),
            'mu_over_floor_all': _dist([r['mu_over_floor'] for r in rs]),
            'floor_status': dict(Counter(r['floor_status'] for r in rs)),
            'exits': dict(Counter(r['exit'] for r in rs)),
            'warm_start': dict(Counter(str(r['warm_start']) for r in rs)),
            'cycle_wall_s': pcr[k]['cycle_wall_s'],
            'top5_iterations': sorted(({'block': block_key_from_record(r), 'iterations': r['iterations']} for r in rs),
                                      key=lambda x: (-x['iterations'], x['block']))[:5],
        }
    out['C_v_iterations_and_mu'] = {
        'recorded': ('network_ipopt_solve_records.jsonl, one per block per round; round k = ADMM cycle k; '
                     f'compl_inf_tol_in_force switches to 1e-6 at round {N7_TAIL_FIRST} = the first tail cycle'),
        'per_cycle': cv,
    }

    # ---------------------------------------------------------------- supplementary: 50-69
    lead_blocks = [b for b, v in recur_sorted if v['n_cycles_in_top10'] >= 8]
    ctx = {}
    for k in CONTEXT_CYCLES + WINDOW:
        t, sc = traj[k], sidecar[k]
        listed = {block_key_from_sidecar(e): e['delta'] for e in (sc['block_deltas'] or [])}
        ctx[k] = {'dQ': dQ.get(k), 'aa_action': aa[k]['aa_action'], 'tail_active': k in cycles_tail,
                  'compl_inf_tol_dso': sorted({r['compl_inf_tol_in_force'] for r in records[k] if r['agent'] != 'TSO'}),
                  'dz_tso_l2_v': (t['boyd_v_s'] / t['rho_v_before']) if t['gamma_v_before'] == 0.0 else None,
                  'dz_tso_l2_pf': (t['boyd_pf_s'] / t['rho_pf_before']) if t['gamma_pf_before'] == 0.0 else None,
                  'dz_tso_l2_ess': (t['boyd_ess_s'] / t['rho_ess_before']) if (t['gamma_ess_before'] == 0.0
                                                                               and t['rho_ess_before']) else None,
                  'rho_v': t['rho_v_before'], 'rho_pf': t['rho_pf_before'], 'rho_ess': t['rho_ess_before'],
                  'primal_ratio_v': t['boyd_v_primal_ratio'], 'dual_ratio_v': t['boyd_v_dual_ratio'],
                  'primal_ratio_pf': t['boyd_pf_primal_ratio'], 'dual_ratio_pf': t['boyd_pf_dual_ratio'],
                  'primal_ratio_ess': t['boyd_ess_primal_ratio'], 'dual_ratio_ess': t['boyd_ess_dual_ratio'],
                  'boyd_all_pass': t['boyd_all_pass'],
                  'row18_charge_weighted': pcr[k]['row18_charge_weighted'],
                  'lead_block_deltas_if_listed': {b: listed.get(b) for b in lead_blocks},
                  'top10_composition_by_agent': dict(Counter(b.split('|')[0] for b in listed)),
                  'top10_signed_sum': sum(listed.values()),
                  'dso_iterations_median': statistics.median(
                      [r['iterations'] for r in records[k] if r['agent'] != 'TSO'])}
    out['supplementary_context_cycles_50_69'] = {
        'note': ('SUPPLEMENTARY, outside the certification window: the same quantities for cycles 50-59 beside 60-69, '
                 f'to show whether the movement predates the tail switch at cycle {N7_TAIL_FIRST}. lead blocks = '
                 'blocks in the window top-10 on >= 8 of the 10 window cycles; None = not in that cycle\'s top 10 '
                 '(not zero)'),
        'lead_blocks': lead_blocks, 'per_cycle': ctx,
    }

    # ---------------------------------------------------------------- D. scoring evidence
    run_start = descending_run_start(dQ, CERT_CYCLE)
    out['D_scoring_evidence'] = {
        'operationalisation_note': ('Addendum 50 states the predictions qualitatively; no threshold is frozen. The '
                                    'verdicts are the Worker\'s reading, for the Planner to rule on.'),
        'dQ_window': {k: dQ[k] for k in WINDOW},
        'terminal_descending_run': {'first_cycle': run_start, 'last_cycle': CERT_CYCLE,
                                    'steps': {k: dQ[k] for k in range(run_start, CERT_CYCLE + 1)},
                                    'sum': Q[CERT_CYCLE] - Q[run_start - 1]},
        'dQ_step_increments_in_run': {k: dQ[k] - dQ[k - 1] for k in range(run_start + 1, CERT_CYCLE + 1)},
        'H_row18_part1_row18_share_of_dQ_per_cycle': {k: civ[k]['share_of_dQ_row18_charge'] for k in WINDOW},
        'H_row18_part1_cumulative': cum,
        'H_row18_part2': 'NOT RECOVERABLE from records (per-block deviation changes per cycle not recorded)',
        'H_B_H_C_top10_signed_share_of_dQ': {k: ci[k]['top10_signed_share_of_dQ'] for k in WINDOW},
        'H_B_H_C_min_unlisted_blocks_contributing': {k: ci[k]['min_unlisted_blocks_contributing'] for k in WINDOW},
        'H_B_H_C_top3_share_of_sum_abs_bounds': {k: ci[k]['top3_share_of_sum_abs_bounds'] for k in WINDOW},
        'H_B_H_C_cancellation_ratio_bounds': {k: ci[k]['cancellation_ratio_bounds_abs_dQ_over_sum_abs'] for k in WINDOW},
        'H_C_dso_entries_in_top10_over_window': {
            'dso': sum(sum(v for a, v in ci[k]['top10_composition_by_agent'].items() if a.startswith('DSO'))
                       for k in WINDOW),
            'by_agent': dict(sum((Counter(ci[k]['top10_composition_by_agent']) for k in WINDOW), Counter())),
            'total': 10 * len(WINDOW)},
        'H_C_top10_vs_all_median_iterations': {k: [ci[k]['median_iterations_top10_this_round'],
                                                   ci[k]['median_iterations_all_80_this_round']] for k in WINDOW},
        'H_C_dso_top10_vs_all_dso_median_iterations': {k: [ci[k]['median_iterations_dso_in_top10_this_round'],
                                                           ci[k]['median_iterations_all_60_dso_this_round']]
                                                       for k in WINDOW},
        'G6_prediction_inputs': {'at_floor_count': band.get('at_floor (|ratio-1|<=1e-3)', 0),
                                 'median_mu_over_floor': statistics.median([r['mu_over_floor'] for r in term]),
                                 'prediction': 'G6 as frozen fails likewise (<= 3/80 at floor, median mu/floor ~ 3)'},
    }
    return Q, dQ, lead_blocks


def compare(out, Q7, dQ7, lead7):
    # ---------------------------------------------------------------- E. two-cell drift comparison
    x0_pcr = {r['cycle']: r for r in iter_jsonl(os.path.join(X0_EVAL, 'per_cycle_record.jsonl'),
                                                'x = 0 per-cycle trajectory (committed 47a54c89) -- drift comparison')}
    x0_ev = load_json(os.path.join(X0_EVAL, 'evaluation_record.json'), 'x = 0 evaluation record')
    cr = load_json(CAMPAIGN_RESULTS, 'campaign results (value, resolution, I, R as the launcher computed them)')
    w95 = load_json(W95_JSON, 'W95 x = 0 diagnostics output (committed d757394e) -- lead blocks', limit=1024 * 1024)
    assert x0_ev['certification_cycle'] == X0_CERT_CYCLE and x0_ev['certified_cost'] == X0_EXPECTED_Q
    Q0 = {k: x0_pcr[k]['gross_operational_cost'] for k in x0_pcr}
    dQ0 = {k: Q0[k] - Q0[k - 1] for k in range(2, X0_CERT_CYCLE + 1)}
    vr = cr['value_and_R']
    value = Q0[X0_CERT_CYCLE] - Q7[CERT_CYCLE]
    assert value == vr['value_eur'] == EXPECTED_VALUE, (value, vr['value_eur'])
    assert vr['resolution'] == EXPECTED_RESOLUTION and vr['I_eur'] == EXPECTED_I
    s0, s7 = dQ0[X0_CERT_CYCLE], dQ7[CERT_CYCLE]
    d_value_per_cycle = s0 - s7      # value = Q0 - Q7: its change per cycle when both cells keep their terminal steps
    run0, run7 = descending_run_start(dQ0, X0_CERT_CYCLE), descending_run_start(dQ7, CERT_CYCLE)

    def aligned(dq, anchor, rng):
        return {str(j): dq.get(anchor + j) for j in rng}
    proj = {}
    for n in PROJECTION_HORIZONS:
        v = value + n * d_value_per_cycle
        proj[str(n)] = {'value': v, 'value_change': n * d_value_per_cycle,
                        'value_change_over_resolution': abs(n * d_value_per_cycle) / EXPECTED_RESOLUTION,
                        'value_minus_I': v - EXPECTED_I}
    mean_incr0 = (dQ0[X0_CERT_CYCLE] - dQ0[run0]) / (X0_CERT_CYCLE - run0) if X0_CERT_CYCLE > run0 else None
    mean_incr7 = (dQ7[CERT_CYCLE] - dQ7[run7]) / (CERT_CYCLE - run7) if CERT_CYCLE > run7 else None
    out['E_drift_comparison'] = {
        'objective_convention': 'Q = gross_operational_cost, settlement excluded; value = Q(x0) - Q(n7_4h_e1)',
        'cells': {'x0': {'eval_key': x0_ev['eval_key'], 'certification_cycle': X0_CERT_CYCLE, 'tail_first': X0_TAIL_FIRST,
                         'Q_cert': Q0[X0_CERT_CYCLE], 'terminal_step': s0, 'terminal_descending_run_first_cycle': run0,
                         'terminal_descending_run_sum': Q0[X0_CERT_CYCLE] - Q0[run0 - 1],
                         'mean_step_increment_in_run': mean_incr0,
                         'step_increments_in_run': {k: dQ0[k] - dQ0[k - 1] for k in range(run0 + 1, X0_CERT_CYCLE + 1)},
                         'lead_blocks_W95': w95['supplementary_context_cycles_53_72']['lead_blocks']},
                  'n7_4h_e1': {'eval_key': out['identity']['eval_key'], 'certification_cycle': CERT_CYCLE,
                               'tail_first': N7_TAIL_FIRST, 'Q_cert': Q7[CERT_CYCLE], 'terminal_step': s7,
                               'terminal_descending_run_first_cycle': run7,
                               'terminal_descending_run_sum': Q7[CERT_CYCLE] - Q7[run7 - 1],
                               'mean_step_increment_in_run': mean_incr7,
                               'step_increments_in_run': {k: dQ7[k] - dQ7[k - 1] for k in range(run7 + 1, CERT_CYCLE + 1)},
                               'lead_blocks': lead7}},
        'value_at_certification': value, 'resolution': EXPECTED_RESOLUTION, 'I': EXPECTED_I,
        'terminal_step_difference_x0_minus_n7': d_value_per_cycle,
        'aligned_to_certification_cycle_offset_-12_to_0': {'x0': aligned(dQ0, X0_CERT_CYCLE, range(-12, 1)),
                                                           'n7': aligned(dQ7, CERT_CYCLE, range(-12, 1))},
        'aligned_to_tail_engagement_offset_-3_to_8': {'x0': aligned(dQ0, X0_TAIL_FIRST, range(-3, 9)),
                                                      'n7': aligned(dQ7, N7_TAIL_FIRST, range(-3, 9))},
        'aligned_to_descending_run_start_offset_0_to_6': {'x0': aligned(dQ0, run0, range(0, 7)),
                                                          'n7': aligned(dQ7, run7, range(0, 7))},
        'PROJECTION_NOT_A_MEASUREMENT': {
            'assumption': ('both cells continue with their TERMINAL steps unchanged (x0 %.6f, n7 %.6f EUR/cycle); '
                           'both runs were still accelerating at certification, so this is neither a bound nor an '
                           'estimate of the limit' % (s0, s7)),
            'value_change_per_cycle': d_value_per_cycle,
            'cycles_until_value_moves_by_its_resolution': EXPECTED_RESOLUTION / abs(d_value_per_cycle),
            'horizons': proj},
    }


def main():
    started = datetime.now(timezone.utc).isoformat()
    os.makedirs(OUT, exist_ok=True)
    for path in (OUT_JSON, OUT_MANIFEST):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    out = {'schema': 'p515_s53_w96_n7_drift_v1', 'task': 'W96 item 2 (PLANNER_BRIEF_2026-09-13.md Addendum 50)',
           'cell': EVAL_REL, 'window_cycles': WINDOW, 'certification_cycle': CERT_CYCLE,
           'sibling_of': 'p515_s53_w95_x0_drift_diagnostics.py (commit d757394e, not edited)',
           'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (the contracted interface '
                                    'settlement and the voltage pin removed; the settlement DEVIATION part and the '
                                    'row-18 charge are inside Q); terminal salvage reported separately, not in Q'),
           'constraint': 'records only; no production module, no pickle, no model; SolveProfileGuard(permitted=()) armed',
           'started_utc': started}
    Q7, dQ7, lead7 = analyse(out)
    compare(out, Q7, dQ7, lead7)
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)
    if loaded:
        raise RuntimeError(f'forbidden modules imported: {loaded}')
    out['forbidden_modules_imported'] = loaded
    out['pickle_loads_performed'] = 0
    out['guard_at_write'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    if out['guard_at_write']['verify_0_failures']:
        raise RuntimeError(f"guard verify(0) failed before write: {out['guard_at_write']}")
    out['peak_rss_bytes_ru_maxrss_self'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['peak_rss_units'] = 'bytes (macOS ru_maxrss)'
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'x') as handle:
        json.dump(out, handle, indent=1, sort_keys=False)
        handle.write('\n')
    script = os.path.abspath(__file__)
    manifest = {
        'schema': 'p515_s53_w96_manifest_v1',
        'script': {'path': os.path.relpath(script, REPO), 'sha256': _hash_whole(script)},
        'guard_module': {'path': 'p513_solve_profile_guard.py',
                         'sha256': _hash_whole(os.path.join(REPO, 'p513_solve_profile_guard.py'))},
        'interpreter': sys.executable, 'python_version': sys.version,
        'inputs': INPUTS, 'n_inputs': len(INPUTS),
        'outputs': {os.path.relpath(OUT_JSON, REPO): {'sha256': _hash_whole(OUT_JSON),
                                                       'size_bytes': os.path.getsize(OUT_JSON)}},
        'note': ('IPOPT logs under data/SRP1/Results/P56A/evals are gitignored; their whole-file sha256 and the '
                 'sha256 of each byte range read are recorded here at read time'),
    }
    with open(OUT_MANIFEST, 'x') as handle:
        json.dump(manifest, handle, indent=1)
        handle.write('\n')
    print(f'wrote {os.path.relpath(OUT_JSON, REPO)} and {os.path.relpath(OUT_MANIFEST, REPO)}; '
          f'{len(INPUTS)} inputs hashed; peak RSS {out["peak_rss_bytes_ru_maxrss_self"] / 2**20:.1f} MiB')


if __name__ == '__main__':
    try:
        main()
    finally:
        GUARD.uninstall()
        failures = GUARD.verify(0)
        print(f'[W96] guard counts {GUARD.counts} verify(0) -> {failures}', flush=True)
        if failures:
            raise SystemExit(f'guard verify(0) failed: {failures}')
