"""
P5.15 Addendum 50, Planner task W95 -- the x = 0 drift diagnostics, FROM RECORDS ONLY.

Cell: data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0 (candidate x0, eval key
f6e9cd53fdbb8ee8..., candidate_key 8435c718...; certified at cycle 72, Q = 842,832,534.7623764 gross, settlement
excluded; tail active on cycles 64-72).

CONSTRAINT (Addendum 50: "records only, no model loading while the cell runs"). The node-7 cell of the same pair is
running while this script runs. This script therefore:
  * imports the Python STANDARD LIBRARY ONLY (asserted at exit: no pyomo, no production module, no harness);
  * loads no .pkl, builds no model, runs no solve;
  * reads JSON / JSONL records and IPOPT log byte ranges, JSONL streamed line by line;
  * refuses to read anything under the running node-7 eval dir or its IPOPT working dir;
  * writes only to a NEW directory (refuses to overwrite an existing output file);
  * hashes every input at read time (IPOPT logs are gitignored, so their sha256 is recorded here).

Items (PLANNER_BRIEF_2026-09-13.md Addendum 50; task W95):
  A  rule ten -- terminal-step-to-threshold, as the harness defines it (p515_g_g1_g4_admm_gates.py:1351, carried into
     the evaluation record by p515_s44_campaign_harness.py:1848):
         objective_change_abs[last] / objective_tolerance[last]
     objective_change_abs = |Q_k - Q_{k-1}| and objective_tolerance = max(tol.abs, tol.rel * max(|Q_k|, |Q_{k-1}|, 1))
     (production shared_resources_planning.py:3340-3343; this arm sets tol.rel = 1e-4, recorded per trajectory row).
  B  G6 verification (Ruling 1), zero solves: mu_final / mu_floor per terminal-round solve from the production solve
     records (network.py parse_ipopt_attempt_segment: mu_floor = min(tol, compl_inf_tol * S) / 11); and, for the
     final accepted attempt of each block in the terminal round, the four terminal error metrics parsed from that
     attempt's own IPOPT byte range, against the tolerances in force (IPOPT termination test: SCALED overall NLP
     error <= tol; UNSCALED dual infeasibility <= dual_inf_tol, constraint violation <= constr_viol_tol,
     complementarity <= compl_inf_tol; options not in the printed options list take IPOPT's documented defaults
     dual_inf_tol = 1, constr_viol_tol = 1e-4). The options list is parsed merge-aware (W79: an over-long
     output_file entry drops its newline and the next option shares the physical line).
  C  (i)-(v) per cycle over the window 63-72, each stated with what the records contain and what they cannot give.
  D  the numbers the Addendum 50 predictions (H_row18, H_B, H_C) are scored on. The scoring itself is in the Worker
     report; the operationalisations used are recorded in the output (`scoring_evidence`), none of them frozen.

Run: /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s53_w95_x0_drift_diagnostics.py
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
PAIR_REL = 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair'
EVAL_REL = f'{PAIR_REL}/evals/f6e9cd53fdbb8ee8_x0'
EVAL = os.path.join(REPO, EVAL_REL)
IPOPT_LOG_DIR = os.path.join(REPO, 'data/SRP1/Results/P56A/evals/p515s44_s53_w91_3x3_pair_f6e9cd53fdbb8ee8_run/logs')
OUT_REL = 'data/SRP1/Results/P515S53/w95_x0_drift'
OUT = os.path.join(REPO, OUT_REL)
OUT_JSON = os.path.join(OUT, 'w95_x0_drift_diagnostics.json')
OUT_MANIFEST = os.path.join(OUT, 'manifest_sha256.json')

# The running cell -- never read.
FORBIDDEN_READ_PREFIXES = (
    os.path.join(REPO, f'{PAIR_REL}/evals/c82522f470b35b58_n7_4h_e1'),
    os.path.join(REPO, 'data/SRP1/Results/P56A/evals/p515s44_s53_w91_3x3_pair_c82522f470b35b58'),
    os.path.join(REPO, '.p515_s44_campaign.lock'),
)
FORBIDDEN_MODULES = ('pyomo', 'shared_resources_planning', 'network', 'network_data', 'model_construction_helpers',
                     'p515_s44_campaign_harness', 'uncoordinated_benchmark', 'p515_g_g1_g4_admm_gates', 'numpy')

CERT_CYCLE = 72
WINDOW = list(range(63, 73))            # Addendum 50: cycles 63-72
TAIL_CYCLES_EXPECTED = list(range(64, 73))
CHANNELS = ('v', 'pf', 'ess')
EXPECTED_Q = 842832534.7623764
EXPECTED_BAR = 13954.375690460205
N_BLOCKS = 80
IPOPT_DEFAULTS = {'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4, 'tol': 1e-8,
                  'acceptable_tol': 1e-6, 'mu_linear_decrease_factor': 0.2, 'mu_superlinear_decrease_power': 1.5}
CONTEXT_CYCLES = list(range(53, 63))    # supplementary: the ten cycles before the window
IPOPT_DEFAULTS_SOURCE = ('IPOPT 3.14 options reference (coin-or.github.io/Ipopt/OPTIONS.html): tol 1e-8, '
                         'dual_inf_tol 1, constr_viol_tol 1e-4, compl_inf_tol 1e-4, acceptable_tol 1e-6; tol is '
                         'compared with the SCALED NLP error, the other three with UNSCALED (absolute) quantities')

INPUTS = {}   # rel path -> hash record


# ======================================================================================================================
#  tracked reads
# ======================================================================================================================
def _guard(path):
    ap = os.path.abspath(path)
    for prefix in FORBIDDEN_READ_PREFIXES:
        if ap.startswith(prefix):
            raise RuntimeError(f'refusing to read from the running cell: {ap}')
    return ap


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
    """Stream a JSONL file line by line; the sha256 is computed over exactly the bytes parsed."""
    path = _guard(path)
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            if raw.strip():
                yield json.loads(raw)
    _register(path, h.hexdigest(), size, note)


def load_json(path, note):
    """Small JSON documents only (the largest read here is g_s39_D.json, < 1 MB)."""
    path = _guard(path)
    size = os.path.getsize(path)
    if size > 2 * 1024 * 1024:
        raise RuntimeError(f'{path} is {size} bytes; this script loads only small JSON documents whole')
    with open(path, 'rb') as handle:
        data = handle.read()
    _register(path, hashlib.sha256(data).hexdigest(), size, note)
    return json.loads(data)


def iter_text_lines(path, note):
    path = _guard(path)
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for raw in handle:
            h.update(raw)
            yield raw.decode('utf-8', errors='replace')
    _register(path, h.hexdigest(), size, note)


def read_log_segment(path, start, end, note):
    """One attempt's own byte range of an appended IPOPT log. The whole file is hashed (streamed) at read time."""
    path = _guard(path)
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
#  IPOPT log parsing (terminal round)
# ======================================================================================================================
_OPT_RE = re.compile(r'(?:^|(?<=\s))([A-Za-z_][A-Za-z0-9_]*) = (\S+)', re.M)
_METRIC_RE = re.compile(r'^(Objective|Dual infeasibility|Constraint violation|Variable bound violation|'
                        r'Complementarity|Overall NLP error)\.*:\s+(\S+)\s+(\S+)\s*$', re.M)
_ITER_RE = re.compile(r'Number of Iterations\.+: (\d+)')
_MU_RE = re.compile(r'Current barrier parameter mu = (\S+)')
_EXIT_RE = re.compile(r'EXIT: (.*)')
_BANNER = 'This is Ipopt version'


def parse_options(text):
    """Merge-aware (W79): a name is matched at a line start or after whitespace, so an entry that shares its
    physical line with a truncated output_file entry is still found. Returns name -> string value, and the list of
    physical lines that held more than one entry."""
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


def parse_terminal_metrics(text):
    """The final 'Number of Iterations' block of ONE attempt: (scaled, unscaled) per metric."""
    it = list(_ITER_RE.finditer(text))
    if not it:
        return None, None
    tail = text[it[-1].start():]
    metrics = {}
    for name, scaled, unscaled in _METRIC_RE.findall(tail):
        key = name.lower().replace(' ', '_')
        metrics[key] = {'scaled': float(scaled), 'unscaled': float(unscaled)}
    return int(it[-1].group(1)), metrics


def _num(opts, name):
    if opts is not None and name in opts:
        return float(opts[name]), 'options_list'
    return IPOPT_DEFAULTS.get(name), 'ipopt_default'


# ======================================================================================================================
#  helpers
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
    """Number of IPOPT monotone mu updates, new_mu = max(floor, min(kappa * mu, mu ** theta)), from mu_final until
    mu reaches the floor. 0 = already at the floor; 1 = one update above the clamp (the Ruling 1 hypothesis)."""
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


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    for path in (OUT_JSON, OUT_MANIFEST):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')

    out = {'schema': 'p515_s53_w95_x0_drift_v1', 'task': 'W95 (PLANNER_BRIEF_2026-09-13.md Addendum 50)',
           'cell': EVAL_REL, 'window_cycles': WINDOW, 'certification_cycle': CERT_CYCLE,
           'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (the contracted interface '
                                    'settlement and the voltage pin removed; the settlement DEVIATION part and the '
                                    'row-18 charge are inside Q) -- shared_resources_planning.py '
                                    '_get_operational_recourse_components; terminal salvage = 0 here, so gross = net'),
           'constraint': 'records only; stdlib only; no pickle, no model, no solve; node-7 cell not read',
           'started_utc': datetime.now(timezone.utc).isoformat()}

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
    records = defaultdict(list)
    for r in iter_jsonl(os.path.join(EVAL, 'network_ipopt_solve_records.jsonl'),
                        'production per-solve IPOPT records (round = ADMM cycle; round 0 = initialisation)'):
        records[r['round']].append(r)

    assert ev['certification_cycle'] == CERT_CYCLE and ev['status'] == 'certified'
    assert ev['certified_cost'] == EXPECTED_Q and ev['bar']['value'] == EXPECTED_BAR
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
    last, prev = traj[CERT_CYCLE], traj[CERT_CYCLE - 1]
    step = abs(Q[CERT_CYCLE] - Q[CERT_CYCLE - 1])
    tol_re = max(last['objective_absolute_tolerance'],
                 last['objective_relative_tolerance'] * max(abs(Q[CERT_CYCLE]), abs(Q[CERT_CYCLE - 1]), 1.0))
    ratio = last['objective_change_abs'] / last['objective_tolerance']
    per_cycle_ratio = {k: {'dQ_signed': dQ[k], 'objective_change_abs': traj[k]['objective_change_abs'],
                           'objective_tolerance': traj[k]['objective_tolerance'],
                           'ratio': traj[k]['objective_change_abs'] / traj[k]['objective_tolerance'],
                           'step_over_bar': abs(dQ[k]) / EXPECTED_BAR} for k in WINDOW}
    out['A_rule_ten'] = {
        'definition': ('rule_ten_terminal_step_over_threshold = objective_change_abs[last] / objective_tolerance[last] '
                       '(p515_g_g1_g4_admm_gates.py:1351-1353; recorded as evaluation_record.rule_ten by '
                       'p515_s44_campaign_harness.py:1848-1854); objective_change_abs = |Q_k - Q_(k-1)|, '
                       'objective_tolerance = max(tol.abs, tol.rel * max(|Q_k|, |Q_(k-1)|, 1)) '
                       '(shared_resources_planning.py:3340-3343)'),
        'terminal_cycle': CERT_CYCLE,
        'Q_terminal': Q[CERT_CYCLE], 'Q_previous': Q[CERT_CYCLE - 1],
        'terminal_step_abs_recomputed_from_Q': step,
        'terminal_step_abs_recorded': last['objective_change_abs'],
        'tol_abs': last['objective_absolute_tolerance'], 'tol_rel': last['objective_relative_tolerance'],
        'threshold_recomputed': tol_re, 'threshold_recorded': last['objective_tolerance'],
        'ratio': ratio,
        'ratio_recorded_in_evaluation_record': ev['rule_ten']['terminal_step_over_threshold'],
        'ratio_equals_record_bitwise': ratio == ev['rule_ten']['terminal_step_over_threshold'],
        'threshold_recompute_rel_diff': abs(tol_re - last['objective_tolerance']) / last['objective_tolerance'],
        'window_series': per_cycle_ratio,
        'note': ('step_over_bar is supplementary (|dQ_k| / the certified bar 13,954.38), not rule ten; the bar is '
                 'the largest window step, attained at cycle 63'),
    }

    # ---------------------------------------------------------------- B. G6 verification
    term = records[CERT_CYCLE]
    assert len(term) == N_BLOCKS, len(term)
    assert all(r['attempt'] == 'primary' for r in term)
    keys = [block_key_from_record(r) for r in term]
    assert len(set(keys)) == N_BLOCKS
    ratios = [r['mu_over_floor'] for r in term]
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
    over5 = [{'block': block_key_from_record(r), 'mu_over_floor': r['mu_over_floor'], 'mu_final': r['mu_final'],
              'mu_floor': r['mu_floor'], 'iterations': r['iterations']} for r in term
             if r['mu_over_floor'] is not None and r['mu_over_floor'] > 5.0]
    per_block = []
    fails = []
    tol_seen = Counter()
    update_counts = Counter()
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
        for name in ('tol', 'dual_inf_tol', 'constr_viol_tol', 'compl_inf_tol', 'acceptable_tol'):
            tols[name], src = _num(opts, name)
            tols[f'{name}_source'] = src
        tol_seen[json.dumps({k: v for k, v in tols.items()}, sort_keys=True)] += 1
        kappa, kappa_src = _num(opts, 'mu_linear_decrease_factor')
        theta, theta_src = _num(opts, 'mu_superlinear_decrease_power')
        n_upd = monotone_updates_to_floor(r['mu_final'], r['mu_floor'], kappa, theta)
        mu_update = {'mu_linear_decrease_factor': kappa, 'mu_linear_decrease_factor_source': kappa_src,
                     'mu_superlinear_decrease_power': theta, 'mu_superlinear_decrease_power_source': theta_src,
                     'next_monotone_mu': max(r['mu_floor'], min(kappa * r['mu_final'], r['mu_final'] ** theta)),
                     'monotone_updates_from_mu_final_to_floor': n_upd}
        update_counts[n_upd] += 1
        checks = None
        if metrics and all(m in metrics for m in ('dual_infeasibility', 'constraint_violation', 'complementarity',
                                                  'overall_nlp_error')):
            checks = {
                'overall_nlp_error_scaled_le_tol': metrics['overall_nlp_error']['scaled'] <= tols['tol'],
                'dual_infeasibility_unscaled_le_dual_inf_tol':
                    metrics['dual_infeasibility']['unscaled'] <= tols['dual_inf_tol'],
                'constraint_violation_unscaled_le_constr_viol_tol':
                    metrics['constraint_violation']['unscaled'] <= tols['constr_viol_tol'],
                'complementarity_unscaled_le_compl_inf_tol':
                    metrics['complementarity']['unscaled'] <= tols['compl_inf_tol'],
            }
            checks['all_four_pass'] = all(checks.values())
            margins = {
                'overall_nlp_error_scaled_over_tol': metrics['overall_nlp_error']['scaled'] / tols['tol'],
                'dual_infeasibility_unscaled_over_dual_inf_tol':
                    metrics['dual_infeasibility']['unscaled'] / tols['dual_inf_tol'],
                'constraint_violation_unscaled_over_constr_viol_tol':
                    metrics['constraint_violation']['unscaled'] / tols['constr_viol_tol'],
                'complementarity_unscaled_over_compl_inf_tol':
                    metrics['complementarity']['unscaled'] / tols['compl_inf_tol'],
            }
        else:
            margins = None
        consistency = {
            'one_banner': text.count(_BANNER) == 1,
            'iterations_match_record': n_it == r['iterations'],
            'mu_final_matches_record': (float(mus[-1]) == r['mu_final']) if mus else False,
            'exit_matches_record': (exits[-1].strip() == r['exit']) if exits else False,
            'compl_inf_tol_logged_equals_in_force': (opts is not None and 'compl_inf_tol' in opts
                                                     and float(opts['compl_inf_tol']) == r['compl_inf_tol_in_force']),
            'tol_logged_equals_in_force': (opts is not None and 'tol' in opts
                                           and float(opts['tol']) == r['tol_in_force']),
        }
        entry = {'block': block_key_from_record(r), 'log': os.path.relpath(r['log_path'], REPO),
                 'log_bytes': r['log_bytes'], 'exit': r['exit'], 'iterations': r['iterations'],
                 'mu_final': r['mu_final'], 'mu_floor': r['mu_floor'], 'mu_over_floor': r['mu_over_floor'],
                 'floor_status': r['floor_status'], 'obj_scaling_factor': r['obj_scaling_factor'],
                 'mu_update': mu_update, 'tolerances_in_force': tols, 'options_list_parsed': opts, 'options_list_merged_lines': merged,
                 'terminal_metrics': metrics, 'checks': checks, 'metric_over_tolerance': margins,
                 'consistency_with_production_record': consistency}
        per_block.append(entry)
        if checks is None or not checks['all_four_pass'] or not all(consistency.values()):
            fails.append(entry['block'])
    agg_margin = {}
    for name in ('overall_nlp_error_scaled_over_tol', 'dual_infeasibility_unscaled_over_dual_inf_tol',
                 'constraint_violation_unscaled_over_constr_viol_tol', 'complementarity_unscaled_over_compl_inf_tol'):
        agg_margin[name] = _dist([e['metric_over_tolerance'][name] for e in per_block if e['metric_over_tolerance']])
    by_agent_ratio = defaultdict(list)
    for r in term:
        by_agent_ratio[r['agent']].append(r['mu_over_floor'])
    tail_window_mu = {}
    for k in TAIL_CYCLES_EXPECTED:
        rs = records[k]
        tail_window_mu[k] = {'n': len(rs), 'floor_status': dict(Counter(r['floor_status'] for r in rs)),
                             'mu_over_floor': _dist([r['mu_over_floor'] for r in rs]),
                             'n_over_5': sum(1 for r in rs if (r['mu_over_floor'] or 0) > 5.0),
                             'exits': dict(Counter(r['exit'] for r in rs))}
    out['B_g6_verification'] = {
        'terminal_round': CERT_CYCLE,
        'n_solves': len(term),
        'attempts': dict(Counter(r['attempt'] for r in term)),
        'final_accepted_attempt_rule': ('every block has exactly one attempt (primary) in the terminal round, so the '
                                        'final accepted attempt is that primary; all 80 exits listed in exits'),
        'exits': dict(Counter(r['exit'] for r in term)),
        'mu_over_floor_distribution': _dist(ratios),
        'mu_over_floor_by_agent': {a: _dist(v) for a, v in sorted(by_agent_ratio.items())},
        'mu_over_floor_band_counts': dict(band),
        'solves_mu_over_floor_gt_5': over5,
        'monotone_updates_from_mu_final_to_floor_counts': {str(k): v for k, v in sorted(update_counts.items(),
                                                                                         key=lambda kv: str(kv[0]))},
        'mu_final_values_seen': dict(Counter(repr(r['mu_final']) for r in term)),
        'band_arithmetic_note': (
            'IPOPT monotone update (IpMonotoneMuUpdate): new_mu = max(mu_floor, min(kappa * mu, mu ** theta)), '
            'kappa = mu_linear_decrease_factor, theta = mu_superlinear_decrease_power (defaults 0.2, 1.5; not set '
            'in the options list). min(kappa*mu, mu**theta) = mu**theta whenever mu < kappa ** (1/(theta-1)) = '
            f'{0.2 ** (1 / 0.5):.4g}; at mu_final = 2.5059e-9 the next value is mu**1.5 = {2.5059035596800618e-09 ** 1.5:.4g}, '
            'far below every floor, so it clamps to the floor in ONE update. The (1, 5] band follows from the linear '
            'factor alone (1/0.2 = 5); monotone_updates_from_mu_final_to_floor applies the full rule.'),
        'mu_floor_definition': 'network.py:627 -- min(tol, compl_inf_tol * obj_scaling_factor) / 11',
        'termination_test': ('IPOPT: scaled overall NLP error <= tol AND unscaled dual inf <= dual_inf_tol AND '
                             'unscaled constraint violation <= constr_viol_tol AND unscaled complementarity <= '
                             'compl_inf_tol'),
        'ipopt_defaults_used_when_not_listed': IPOPT_DEFAULTS, 'ipopt_defaults_source': IPOPT_DEFAULTS_SOURCE,
        'tolerance_sets_seen': {k: v for k, v in tol_seen.items()},
        'options_list_merged_physical_lines_total': merged_lines_total,
        'metric_over_tolerance_distribution': agg_margin,
        'blocks_failing_any_check_or_consistency': fails,
        'per_block': per_block,
        'tail_window_mu_per_round_supplementary': tail_window_mu,
    }

    # ---------------------------------------------------------------- C(i). per-block dQ decomposition
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
        rounds = {block_key_from_record(r): r for r in records[k]}
        it_all = [r['iterations'] for r in records[k]]
        med_it = statistics.median(it_all)
        entries = []
        for e in top:
            bk = block_key_from_sidecar(e)
            rr = rounds.get(bk)
            entries.append({'block': bk, 'delta': e['delta'], 'previous': e['previous'], 'current': e['current'],
                            'iterations_this_round': rr['iterations'] if rr else None,
                            'mu_over_floor_this_round': rr['mu_over_floor'] if rr else None})
        comp = Counter(x['block'].split('|')[0] for x in entries)
        top3_abs = sum(sorted((e['abs_delta'] for e in top), reverse=True)[:3])
        oc = sc['objective_component_block_deltas'] or []
        # where a listed block's classified_total delta is also in the (top-10) component list:
        #   Q delta - classified delta = change of the Q terms outside the classified set (row-18 charge, settlement
        #   deviation part, ...). The component list's block_key is str(tuple) with an int year.
        classified = {}
        for e in oc:
            if e['component'] == 'classified_total':
                classified[e['block_key']] = e['delta']
        for x, e in zip(entries, top):
            tup = str((e['agent'], e['node_id'], int(e['year']), e['day']))
            x['classified_total_delta_if_listed'] = classified.get(tup)
            x['q_delta_minus_classified_delta_if_listed'] = (e['delta'] - classified[tup]) if tup in classified else None
        dso_it_all = [r['iterations'] for r in records[k] if r['agent'] != 'TSO']
        dso_it_top = [x['iterations_this_round'] for x in entries if not x['block'].startswith('TSO')]
        ci[k] = {
            'dQ_from_block_totals': total, 'dQ_from_per_cycle_record': dQ[k],
            'block_total_current_minus_Q': sc['block_total_current'] - Q[k],
            'top10': entries,
            'top10_signed_sum': top_sum, 'top10_abs_sum': top_abs,
            'top10_signed_share_of_dQ': top_sum / total if total else None,
            'remainder_70_unlisted_blocks_signed': rem,
            'remainder_share_of_dQ': rem / total if total else None,
            'unlisted_block_abs_delta_upper_bound': bound,
            'min_unlisted_blocks_contributing': math.ceil(abs(rem) / bound) if bound > 0 else None,
            'sum_abs_block_delta_bounds': [lower_abs_total, upper_abs_total],
            'cancellation_ratio_bounds_abs_dQ_over_sum_abs': [abs(total) / upper_abs_total,
                                                               abs(total) / lower_abs_total],
            'top3_share_of_sum_abs_bounds': [top3_abs / upper_abs_total, top3_abs / lower_abs_total],
            'top10_composition_by_agent': dict(comp),
            'top10_sign_counts': {'negative': sum(1 for e in top if e['delta'] < 0),
                                  'positive': sum(1 for e in top if e['delta'] > 0)},
            'median_iterations_all_80_this_round': med_it,
            'median_iterations_top10_this_round': statistics.median(
                [x['iterations_this_round'] for x in entries if x['iterations_this_round'] is not None]),
            'median_iterations_all_60_dso_this_round': statistics.median(dso_it_all),
            'median_iterations_dso_in_top10_this_round': statistics.median(dso_it_top) if dso_it_top else None,
            'dso_in_top10_iteration_percentile_ranks': [
                sum(1 for v in dso_it_all if v < it) / len(dso_it_all) for it in dso_it_top],
            'objective_component_top10': [{'block': e['block_key'], 'component': e['component'], 'delta': e['delta']}
                                          for e in oc],
        }
    # blocks recurring in the window's top-10 lists, with their summed listed deltas
    recur = defaultdict(lambda: {'n_cycles_in_top10': 0, 'sum_listed_delta': 0.0, 'cycles': []})
    for k in WINDOW:
        for e in ci[k]['top10']:
            recur[e['block']]['n_cycles_in_top10'] += 1
            recur[e['block']]['sum_listed_delta'] += e['delta']
            recur[e['block']]['cycles'].append(k)
    recur_sorted = sorted(recur.items(), key=lambda kv: (-kv[1]['n_cycles_in_top10'], kv[1]['sum_listed_delta']))
    out['C_i_per_block_dQ'] = {
        'recorded': ('recourse_jump_sidecar_baseline.jsonl (p515_g_g1_g4_admm_gates.py:2932-3016): per cycle, the '
                     'block totals (sum over all 80 blocks of production _get_operational_recourse_block_components '
                     '= Q) and the TOP 10 blocks by |delta| only (deltas[:10]); objective components top 10 likewise '
                     '(_get_operational_objective_component_blocks, where "unclassified" = objective.expr minus '
                     'classified parts and includes the ADMM augmented-Lagrangian terms, settlement, row-18 charge '
                     'and voltage pin -- NOT a Q component)'),
        'not_recoverable_from_records': ('the individual deltas of the 70 blocks outside each cycle\'s top 10; '
                                         'bounded here (each <= the 10th |delta|, net = remainder)'),
        'per_cycle': ci,
        'blocks_recurring_in_window_top10': dict(recur_sorted),
    }

    # ---------------------------------------------------------------- C(ii). consensus / dual movement
    prox = {}
    cyc = None
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
                'rho': rho, 'gamma': gamma,
                'r_l2': t[f'boyd_{c}_r'], 's_l2': s,
                'dz_tso_l2_normalized_eq_s_over_rho': (s / rho) if (rho and gamma == 0.0) else None,
                'dy_dso_l2_normalized_eq_rho_times_r': (rho * t[f'boyd_{c}_r']) if c in ('v', 'pf') else None,
                'norm_x': t[f'boyd_{c}_norm_x'], 'norm_z': t[f'boyd_{c}_norm_z'], 'norm_y': t[f'boyd_{c}_norm_y'],
                'norm_y_change_vs_previous_cycle': t[f'boyd_{c}_norm_y'] - tp[f'boyd_{c}_norm_y'],
                'legacy_primal_max': t.get(f'primal_{c}'), 'legacy_dual_max': t.get(f'dual_{c}'),
            }
        row['worst_v_primal'] = {x: t.get(f'worst_v_primal_{x}') for x in ('node', 'year', 'day', 'period',
                                                                             'difference')}
        row['worst_pf_primal'] = {x: t.get(f'worst_pf_primal_{x}') for x in ('node', 'year', 'day', 'period', 'type',
                                                                               'difference')}
        row['worst_pf_dual'] = {x: t.get(f'worst_pf_dual_{x}') for x in ('agent', 'node', 'year', 'day', 'period',
                                                                           'type', 'change')}
        cii[k] = row
    out['C_ii_consensus_and_dual_movement'] = {
        'recorded': ('g_s39_D.json cycle_trajectory: Boyd r, s, ||x||, ||z||, ||y|| per channel (production '
                     'get_admm_boyd_residual_metrics, shared_resources_planning.py:7044); worst-entry locations; '
                     'stdout [TSO PROX] max normalized TSO-side movement per channel (shared_resources_planning.py:5729)'),
        'derived': ('||dz_TSO||_2 = s / rho when gamma = 0 (s = sqrt(rho^2 + gamma^2) ||dz||); ||dy_DSO||_2 = rho * r '
                    'for V and PF from the production dual update (shared_resources_planning.py:8494-8500: '
                    'lambda += rho (x - z) [/rating * s_base], y = lambda / base) -- exact only when Anderson '
                    'acceleration did not act that cycle (aa_action recorded per row; "off" on all of 63-72)'),
        'not_recoverable_from_records': ('per-entry consensus / dual vectors per cycle (only norms, maxima and '
                                         'argmax locations are recorded); per-block dual movement'),
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
        row['dQ_minus_row18_minus_cov_dso'] = (dQ[k] - row['delta_row18_charge_weighted']
                                               - row['delta_covariance_dso_weighted'])
        civ[k] = row
    cum = {}
    for a, b in ((63, 72), (64, 72), (66, 72)):
        cum[f'{a}-{b}'] = {
            'dQ': Q[b] - Q[a - 1],
            'd_row18_charge_weighted': pcr[b]['row18_charge_weighted'] - pcr[a - 1]['row18_charge_weighted'],
            'd_covariance_dso_weighted': pcr[b]['covariance_dso_weighted'] - pcr[a - 1]['covariance_dso_weighted'],
            'd_E_abs_d_p_mwh_weighted': pcr[b]['E_abs_d_p_mwh_weighted'] - pcr[a - 1]['E_abs_d_p_mwh_weighted'],
        }
        cum[f'{a}-{b}']['row18_share_of_dQ'] = cum[f'{a}-{b}']['d_row18_charge_weighted'] / cum[f'{a}-{b}']['dQ']
    out['C_iv_row18_terms'] = {
        'recorded': ('per_cycle_record response fields (p515_s44_campaign_harness.py per_cycle_response_record, '
                     'lines 3597-3650): row18_charge_weighted = sum over DSO blocks of w_b x row18_deviation_charge '
                     '(alpha x premium_t x (d+ + d-) over P and Q -- the row-18 term inside Q); E_abs_d_p = '
                     'sum_s omega_s sum_t |d_p| (P leg only), weighted by w_b and unweighted; sum omega d^2 and its '
                     'market / operation split; covariance_dso_weighted = DSO settlement DEVIATION part (inside Q); '
                     'max |d_p|'),
        'premium_note': ('premium_t = pibar_t (alpha 0.5, no floor), fixed by construction and read back equal at '
                         'activation (activation_readback); the charge moves only through the deviations'),
        'not_recoverable_from_records': ('per-block row-18 charge / deviation per cycle (only the DSO-summed '
                                         'aggregates are recorded per cycle; per-block values exist at the terminal '
                                         'point only, multiscenario_terminal.json); the Q leg |d_q| per cycle; the '
                                         'TSO settlement deviation part per cycle'),
        'per_cycle': civ,
        'cumulative': cum,
    }

    # ---------------------------------------------------------------- C(v). iterations and mu per cycle
    cv = {}
    for k in WINDOW:
        rs = records[k]
        assert len(rs) == N_BLOCKS
        by = defaultdict(list)
        for r in rs:
            by['TSO' if r['agent'] == 'TSO' else 'DSO'].append(r)
        cv[k] = {
            'compl_inf_tol_in_force': sorted({(r['agent'], r['compl_inf_tol_in_force']) for r in rs}),
            'tail_active': k in cycles_tail,
            'iterations_all': _dist([r['iterations'] for r in rs]),
            'iterations_tso': _dist([r['iterations'] for r in by['TSO']]),
            'iterations_dso': _dist([r['iterations'] for r in by['DSO']]),
            'mu_final_dso': _dist([r['mu_final'] for r in by['DSO']]),
            'mu_final_tso': _dist([r['mu_final'] for r in by['TSO']]),
            'mu_over_floor_all': _dist([r['mu_over_floor'] for r in rs]),
            'floor_status': dict(Counter(r['floor_status'] for r in rs)),
            'exits': dict(Counter(r['exit'] for r in rs)),
            'warm_start': dict(Counter(str(r['warm_start']) for r in rs)),
            'cycle_wall_s': pcr[k]['cycle_wall_s'],
            'top5_iterations': sorted(({'block': block_key_from_record(r), 'iterations': r['iterations']}
                                       for r in rs), key=lambda x: (-x['iterations'], x['block']))[:5],
        }
    out['C_v_iterations_and_mu'] = {
        'recorded': 'network_ipopt_solve_records.jsonl (production network.py _append_ipopt_solve_record), one per '
                    'block per round; round k = ADMM cycle k (round 0 = initialisation; compl_inf_tol_in_force '
                    'switches to 1e-6 at round 64 = the first tail cycle)',
        'per_cycle': cv,
    }

    # ---------------------------------------------------------------- supplementary: the ten cycles before the window
    lead_blocks = [b for b, v in recur_sorted if v['n_cycles_in_top10'] >= 8]
    ctx = {}
    for k in CONTEXT_CYCLES + WINDOW:
        t, sc = traj[k], sidecar[k]
        listed = {block_key_from_sidecar(e): e['delta'] for e in (sc['block_deltas'] or [])}
        ctx[k] = {'dQ': dQ.get(k), 'aa_action': aa[k]['aa_action'], 'tail_active': k in cycles_tail,
                  'compl_inf_tol_dso': sorted({r['compl_inf_tol_in_force'] for r in records[k] if r['agent'] != 'TSO'}),
                  'dz_tso_l2_v': (t['boyd_v_s'] / t['rho_v_before']) if t['gamma_v_before'] == 0.0 else None,
                  'dz_tso_l2_pf': (t['boyd_pf_s'] / t['rho_pf_before']) if t['gamma_pf_before'] == 0.0 else None,
                  'rho_v': t['rho_v_before'], 'rho_pf': t['rho_pf_before'],
                  'primal_ratio_v': t['boyd_v_primal_ratio'], 'dual_ratio_v': t['boyd_v_dual_ratio'],
                  'primal_ratio_pf': t['boyd_pf_primal_ratio'], 'dual_ratio_pf': t['boyd_pf_dual_ratio'],
                  'row18_charge_weighted': pcr[k]['row18_charge_weighted'],
                  'lead_block_deltas_if_listed': {b: listed.get(b) for b in lead_blocks},
                  'dso_iterations_median': statistics.median(
                      [r['iterations'] for r in records[k] if r['agent'] != 'TSO'])}
    out['supplementary_context_cycles_53_72'] = {
        'note': ('SUPPLEMENTARY, outside the Addendum 50 window: the same quantities for cycles 53-62 beside 63-72, '
                 'to show whether the movement predates the tail switch at cycle 64. lead blocks = blocks in the '
                 'window top-10 on >= 8 of the 10 window cycles; None = not in that cycle\'s top 10 (not zero)'),
        'lead_blocks': lead_blocks,
        'per_cycle': ctx,
    }

    # ---------------------------------------------------------------- D. scoring evidence
    steps = [dQ[k] for k in WINDOW]
    tail_steps = {k: dQ[k] for k in range(66, 73)}
    incr = {k: dQ[k] - dQ[k - 1] for k in range(67, 73)}
    dso_top = sum(ci[k]['top10_composition_by_agent'].get('DSO5', 0) + ci[k]['top10_composition_by_agent'].get('DSO7', 0)
                  + ci[k]['top10_composition_by_agent'].get('DSO9', 0) for k in WINDOW)
    out['D_scoring_evidence'] = {
        'operationalisation_note': ('Addendum 50 states the three predictions qualitatively; no threshold is '
                                    'frozen. The quantities below are the ones the Worker report scores on; the '
                                    'verdicts are the Worker\'s reading, for the Planner to rule on.'),
        'dQ_window': {k: dQ[k] for k in WINDOW},
        'dQ_66_72': tail_steps,
        'dQ_step_increments_67_72': incr,
        'H_row18_part1_row18_share_of_dQ_per_cycle': {k: civ[k]['share_of_dQ_row18_charge'] for k in WINDOW},
        'H_row18_part1_cumulative': cum,
        'H_row18_part2': 'NOT RECOVERABLE from records (per-block deviation changes per cycle not recorded)',
        'H_B_H_C_top10_signed_share_of_dQ': {k: ci[k]['top10_signed_share_of_dQ'] for k in WINDOW},
        'H_B_H_C_min_unlisted_blocks_contributing': {k: ci[k]['min_unlisted_blocks_contributing'] for k in WINDOW},
        'H_B_H_C_top3_share_of_sum_abs_bounds': {k: ci[k]['top3_share_of_sum_abs_bounds'] for k in WINDOW},
        'H_B_H_C_cancellation_ratio_bounds': {k: ci[k]['cancellation_ratio_bounds_abs_dQ_over_sum_abs']
                                               for k in WINDOW},
        'H_C_dso_entries_in_top10_over_window': {'dso': dso_top, 'total': 10 * len(WINDOW)},
        'H_C_top10_vs_all_median_iterations': {k: [ci[k]['median_iterations_top10_this_round'],
                                                   ci[k]['median_iterations_all_80_this_round']] for k in WINDOW},
        'H_C_dso_top10_vs_all_dso_median_iterations': {k: [ci[k]['median_iterations_dso_in_top10_this_round'],
                                                           ci[k]['median_iterations_all_60_dso_this_round']]
                                                       for k in WINDOW},
        'steps_sum_63_72': sum(steps),
    }

    # ---------------------------------------------------------------- self-checks and write
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULES)
    if loaded:
        raise RuntimeError(f'forbidden modules imported: {loaded}')
    out['forbidden_modules_imported'] = loaded
    out['peak_rss_bytes_ru_maxrss_self'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['peak_rss_units'] = 'bytes (macOS ru_maxrss)'
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'x') as handle:
        json.dump(out, handle, indent=1, sort_keys=False)
        handle.write('\n')
    script = os.path.abspath(__file__)
    manifest = {
        'schema': 'p515_s53_w95_manifest_v1',
        'script': {'path': os.path.relpath(script, REPO), 'sha256': _hash_whole(script)},
        'interpreter': sys.executable, 'python_version': sys.version,
        'inputs': INPUTS,
        'n_inputs': len(INPUTS),
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
    main()
