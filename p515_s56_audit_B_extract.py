"""
P5.15 S56 input audit, part B (Planner task, 2026-11-21): ZERO SOLVES, READ-ONLY extraction of the runtime / memory /
concurrency facts needed to design and cost the parallel pilot of the S56 study (expected-profile and stochastic
searches on the five-block 3 x 3 instance at 1x and 2x flexibility price, plus a common stochastic evaluation).

WHAT IT READS (nothing is written outside OUT_DIR):
  * for every evaluation directory of the campaign roots in CAMPAIGNS: `evaluation_record.json` (status, cycles,
    certification cycle, certified cost, wall-time and peak-RSS fields, candidate / eval key, derived instance),
    `wait4_rusage.json` (parent-reaped rusage, parent-view wall), `launch.json` (start time, thread caps) and
    `per_cycle_record.jsonl` (per-cycle wall / RSS where production recorded them);
  * the campaign spec (`concurrency`) and `campaign_results.json` (`batch_info.max_concurrent_observed`, footprint
    peaks when the launcher captured them);
  * for the evaluations in IPOPT_BREAKDOWN: every IPOPT `output_file` log in the evaluation's P56A working directory
    (`data/SRP1/Results/P56A/evals/<run id>/logs`), parsed for `Total seconds in IPOPT` (IPOPT 3.14 wall time of the
    solve inside the IPOPT process: excludes Pyomo NL write, process start-up and solution load), aligned to ADMM
    rounds through the evaluation's own `network_ipopt_solve_records.jsonl` when it exists.
Every file read is sha256-hashed into the manifest.

GUARDS: SolveProfileGuard(permitted=()) installed before anything else and verified at exactly 0; pickle.load /
pickle.loads blocked and counted (0 expected). No production module is imported.

OUTPUT (refuses if OUT_DIR exists): OUT_DIR/audit_B_extract.json and OUT_DIR/manifest_sha256.json.
COMMAND (repo root, canonical interpreter, attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_audit_B_extract.py \\
      > <scratch>/audit_B_launch.log 2>&1
(the launch log cannot live in OUT_DIR because OUT_DIR must not exist at launch; it is copied in afterwards by
nobody -- the script itself writes `run_log.txt` with everything it printed).
"""
import glob
import hashlib
import json
import os
import pickle
import re
import statistics
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 audit B extraction (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('audit B: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('audit B: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'input_audit_B')
if len(sys.argv) > 2 and sys.argv[1] == '--scratch-out':  # rehearsal only: write to a scratch directory instead
    OUT_DIR = sys.argv[2]
OUT_JSON = os.path.join(OUT_DIR, 'audit_B_extract.json')
OUT_MAN = os.path.join(OUT_DIR, 'manifest_sha256.json')
OUT_LOG = os.path.join(OUT_DIR, 'run_log.txt')
P56A_WORK = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals')
GIB = 1024 ** 3

# (group, campaign root glob). Groups name the instance class; the per-eval derived_instance field is the check.
CAMPAIGNS = [
    ('srp1_single_scenario_W187_screen_conc1', 'data/SRP1/Results/P515S55/e1_screen_1x/campaign_s55_step7_*'),
    ('srp1_single_scenario_s47_phase_b', 'data/SRP1/Results/P515S47/campaign_s47_phase_b'),
    ('srp1_single_scenario_s51_f2_phase_b', 'data/SRP1/Results/P515S51/campaign_s51_f2_phase_b'),
    ('srp1_single_scenario_s53_f2_certificate_r1', 'data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1'),
    ('srp1_single_scenario_w86_tail_recert', 'data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_recert'),
    ('multiscenario_2x2_pilot_s52', 'data/SRP1/Results/P515S52/campaign_s52_pilot*'),
    ('multiscenario_3x3_5block_smoke', 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w9*_3x3_smoke_*'),
    ('multiscenario_3x3_5block_pair', 'data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair'),
    ('multiscenario_3x3_5block_continuation', 'data/SRP1/Results/P515S53/w98_continuation/campaign_s53_w98_*'),
]

# evaluations whose IPOPT logs are parsed (by campaign root and eval-dir name)
IPOPT_BREAKDOWN = [
    ('data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair', 'f6e9cd53fdbb8ee8_x0'),
    ('data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair', 'c82522f470b35b58_n7_4h_e1'),
    ('data/SRP1/Results/P515S55/e1_screen_1x/campaign_s55_step7_x0_a_1x', '5cfe69a615ae3708_x0_a_1x'),
    ('data/SRP1/Results/P515S55/e1_screen_1x/campaign_s55_step7_y2025__n7_p0_25_e0_5_n_1x', None),
    ('data/SRP1/Results/P515S47/campaign_s47_phase_b', None),  # every eval: concurrency-5 comparison
]

# network name -> agent (case file connection ids: case33_1 @ 5, case33_2 @ 7, case33_3 @ 9)
AGENT_OF_NETWORK = {'case9': 'TSO', 'case33_1': 'DSO_node5', 'case33_2': 'DSO_node7', 'case33_3': 'DSO_node9'}

READ = {}
LOG_LINES = []


def log(msg):
    line = f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56-B] {msg}'
    print(line, flush=True)
    LOG_LINES.append(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    READ[path] = sha256_file(path)
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    READ[path] = sha256_file(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_text(path):
    READ[path] = sha256_file(path)
    with open(path, errors='replace') as f:
        return f.read()


def _ts(s):
    return datetime.fromisoformat(s).timestamp() if s else None


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {'n': len(xs), 'min': min(xs), 'median': statistics.median(xs), 'mean': statistics.fmean(xs),
            'max': max(xs), 'sum': sum(xs)}


def eval_facts(eval_dir):
    rec_p = os.path.join(eval_dir, 'evaluation_record.json')
    out = {'eval_dir': eval_dir, 'has_evaluation_record': os.path.exists(rec_p)}
    if os.path.exists(rec_p):
        rec = read_json(rec_p)
        di = rec.get('derived_instance') or {}
        out.update({
            'label': rec.get('candidate_label'), 'eval_key': rec.get('eval_key'),
            'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
            'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
            'certification_cycle': rec.get('certification_cycle'),
            'certified_cost_gross_eur': rec.get('certified_cost'),
            'wall_time_s': rec.get('wall_time_s'), 'peak_rss': rec.get('peak_rss'),
            'release_solution_bookkeeping': rec.get('release_solution_bookkeeping'),
            'derived_instance_case_path': di.get('case_path'), 'derived_instance_case_sha256': di.get('case_sha256'),
            'solves_observed': ((rec.get('solve_profile') or {}).get('observed') or {}).get('permitted_solve'),
            'solves_per_cycle': (rec.get('solve_profile') or {}).get('solves_per_cycle'),
            'n_dso': (rec.get('solve_profile') or {}).get('n_dso'),
            'n_years': (rec.get('solve_profile') or {}).get('n_years'),
            'n_days': (rec.get('solve_profile') or {}).get('n_days'),
            'thread_caps_seen_by_child': rec.get('thread_caps_seen_by_child'),
            'working_dir_run': (rec.get('working_dir_ids') or {}).get('run'),
            'local_solve_failures': rec.get('local_solve_failures'),
        })
    w4 = os.path.join(eval_dir, 'wait4_rusage.json')
    if os.path.exists(w4):
        d = read_json(w4)
        out['wait4'] = {'wall_s_parent_view': d.get('wall_s_parent_view'),
                        'ru_maxrss_bytes': (d.get('rusage') or {}).get('ru_maxrss'),
                        'ru_utime_s': (d.get('rusage') or {}).get('ru_utime'),
                        'ru_stime_s': (d.get('rusage') or {}).get('ru_stime'),
                        'exit_code': d.get('exit_code')}
    la = os.path.join(eval_dir, 'launch.json')
    if os.path.exists(la):
        d = read_json(la)
        out['started_utc'] = d.get('started_utc')
        out['thread_caps_in_child_env'] = d.get('thread_caps_in_child_env')
    pc = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    if os.path.exists(pc):
        rows = read_jsonl(pc)
        walls = [r.get('cycle_wall_s') for r in rows]
        rss = [r.get('rss_bytes') for r in rows]
        out['per_cycle'] = {'n_rows': len(rows), 'cycle_wall_s': _stats(walls),
                            'cycle_wall_s_first3': walls[:3], 'cycle_wall_s_last3': walls[-3:],
                            'rss_bytes': _stats(rss), 'ru_maxrss_bytes_max': max(
                                [r.get('ru_maxrss_bytes') for r in rows if r.get('ru_maxrss_bytes')] or [0]) or None}
    # derived per-cycle average over the whole evaluation
    wall = (out.get('wait4') or {}).get('wall_s_parent_view')
    if wall and out.get('cycles_run'):
        out['wall_per_cycle_s_including_init_and_terminal'] = wall / out['cycles_run']
    return out


def campaign_facts(group, root):
    specs = sorted(glob.glob(os.path.join(root, 'campaign_spec_*.json')))
    spec = read_json(specs[0]) if specs else {}
    res_p = os.path.join(root, 'campaign_results.json')
    res = read_json(res_p) if os.path.exists(res_p) else {}
    batch = res.get('batch_info') or {}
    if isinstance(batch, list):
        mco = max([b.get('max_concurrent_observed') or 0 for b in batch if isinstance(b, dict)] or [0])
    else:
        mco = batch.get('max_concurrent_observed')
    footprints = {}
    for k, v in (res.get('per_cell') or {}).items():
        fp = (v or {}).get('footprint_peak_reported_not_gated') or {}
        if fp.get('lifetime_max_phys_footprint_bytes'):
            footprints[k] = fp['lifetime_max_phys_footprint_bytes']
    fp = res.get('footprint_peak_reported_not_gated') or {}
    if fp.get('lifetime_max_phys_footprint_bytes'):
        footprints['_campaign'] = fp['lifetime_max_phys_footprint_bytes']
    evals = [eval_facts(d.rstrip('/')) for d in sorted(glob.glob(os.path.join(root, 'evals', '*/')))]
    # overlap: number of children alive at each child's midpoint, from launch start + parent-view wall
    spans = [(e.get('label'), _ts(e.get('started_utc')), (e.get('wait4') or {}).get('wall_s_parent_view'))
             for e in evals]
    spans = [(lab, s, s + w) for lab, s, w in spans if s is not None and w is not None]
    for e in evals:
        s, w = _ts(e.get('started_utc')), (e.get('wait4') or {}).get('wall_s_parent_view')
        if s is None or w is None:
            continue
        # time-weighted mean number of OTHER children alive during this child's life (sampled every 60 s)
        n, acc = 0, 0
        t = s
        while t < s + w:
            acc += sum(1 for _l, a, b in spans if a <= t < b) - 1
            n += 1
            t += 60.0
        e['mean_other_children_alive'] = acc / n if n else None
    return {'group': group, 'root': root, 'spec_path': specs[0] if specs else None,
            'spec_concurrency': spec.get('concurrency'), 'max_concurrent_observed': mco,
            'campaign_wall_clock_s': res.get('wall_clock_s'), 'footprint_peaks_bytes': footprints,
            'n_evals': len(evals), 'evals': evals}


TOTAL_RE = re.compile(r'Total seconds in IPOPT\s*=\s*([0-9.eE+-]+)')
EXIT_RE = re.compile(r'^EXIT: (.*)$', re.M)
NET_RE = re.compile(r'^optim_log_(case\d+(?:_\d+)?)_(\d{4})_([A-Za-z]+)(?:_[A-Za-z0-9]+)?\.log$')  # incl. *_recovery
ESSO_RE = re.compile(r'^optim_log_esso_node(\d+)_(cycle(\d+)|init)\.txt$')


def ipopt_breakdown(eval_dir, run_id):
    logs = os.path.join(P56A_WORK, run_id, 'logs')
    if not os.path.isdir(logs):
        return {'logs_dir': logs, 'present': False}
    recs_p = os.path.join(eval_dir, 'network_ipopt_solve_records.jsonl')
    recs = read_jsonl(recs_p) if os.path.exists(recs_p) else None
    per_round = {}
    totals = {'TSO': 0.0, 'DSO_node5': 0.0, 'DSO_node7': 0.0, 'DSO_node9': 0.0, 'ESSO': 0.0}
    by_block = {}
    counts = {'network_solves_parsed': 0, 'esso_solves_parsed': 0, 'files': 0, 'alignment_mismatch_records': [],
              'network_records': None, 'network_records_aligned': 0, 'recovery_attempt_records': 0}
    exits = {}
    texts = {}
    for name in sorted(os.listdir(logs)):
        path = os.path.join(logs, name)
        m, me = NET_RE.match(name), ESSO_RE.match(name)
        if not (m or me):
            continue
        text = read_text(path)
        counts['files'] += 1
        secs = [float(x) for x in TOTAL_RE.findall(text)]
        for ex in EXIT_RE.findall(text):
            exits[ex.strip()] = exits.get(ex.strip(), 0) + 1
        if m:
            texts[os.path.abspath(path)] = path
            net, year, day = m.group(1), m.group(2), m.group(3)
            agent = AGENT_OF_NETWORK.get(net, net)
            counts['network_solves_parsed'] += len(secs)
            totals[agent] = totals.get(agent, 0.0) + sum(secs)
            b = by_block.setdefault(f'{agent}|{year}|{day}', {'n_solves': 0, 'sum_s': 0.0, 'max_s': 0.0})
            b['n_solves'] += len(secs)
            b['sum_s'] += sum(secs)
            b['max_s'] = max([b['max_s']] + secs)
        else:
            node, rd = me.group(1), (int(me.group(3)) if me.group(3) else 0)
            counts['esso_solves_parsed'] += len(secs)
            totals['ESSO'] += sum(secs)
            by_block.setdefault(f'ESSO|node{node}', {'n_solves': 0, 'sum_s': 0.0, 'max_s': 0.0})
            b = by_block[f'ESSO|node{node}']
            b['n_solves'] += len(secs)
            b['sum_s'] += sum(secs)
            b['max_s'] = max([b['max_s']] + secs)
            per_round.setdefault(rd, {}).setdefault('ESSO', 0.0)
            per_round[rd]['ESSO'] += sum(secs)
    for b in by_block.values():
        if b.get('n_solves'):
            b['mean_s'] = b['sum_s'] / b['n_solves']
    # network solves -> ADMM round through production's own records: each record names its log file and the byte
    # range [start, end) its solve appended (retry attempts included, in their own *_recovery logs)
    if recs is not None:
        counts['network_records'] = len(recs)
        cache = {}
        for r in recs:
            lp = os.path.abspath(r.get('log_path') or '')
            rel = texts.get(lp)
            a, z = (r.get('log_bytes') or [None, None])
            if rel is None or a is None:
                counts['alignment_mismatch_records'].append({'log_path': r.get('log_path'), 'reason': 'log not parsed'})
                continue
            if rel not in cache:
                with open(rel, 'rb') as f:
                    cache[rel] = f.read()
            found = TOTAL_RE.findall(cache[rel][a:z].decode('utf-8', 'replace'))
            if len(found) != 1:
                counts['alignment_mismatch_records'].append({'log_path': rel, 'bytes': [a, z], 'n_found': len(found)})
                continue
            agent = AGENT_OF_NETWORK.get(r['network'], r['network'])
            per_round.setdefault(r.get('round'), {}).setdefault(agent, 0.0)
            per_round[r.get('round')][agent] += float(found[0])
            counts['network_records_aligned'] += 1
            if r.get('attempt') != 'primary':
                counts['recovery_attempt_records'] += 1
    # per-round shares against the recorded cycle wall
    pc_p = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    cyc_wall = {}
    if os.path.exists(pc_p):
        for r in read_jsonl(pc_p):
            if r.get('cycle_wall_s') is not None:
                cyc_wall[r['cycle']] = r['cycle_wall_s']
    rows = []
    for rd in sorted(k for k in per_round if k is not None):
        tot = sum(per_round[rd].values())
        row = {'round': rd, 'ipopt_s_by_agent': per_round[rd], 'ipopt_s_total': tot}
        if rd in cyc_wall:
            row['cycle_wall_s'] = cyc_wall[rd]
            row['ipopt_share_of_cycle_wall'] = tot / cyc_wall[rd]
        rows.append(row)
    cyc_rows = [r for r in rows if r['round'] >= 1]
    esso_rounds = sorted(k for k in per_round if k is not None)
    summary = {
        'round_alignment_available': recs is not None and not counts['alignment_mismatch_records'],
        'n_rounds_esso_logged': len(esso_rounds),
        'mean_ipopt_s_per_round_from_totals': (sum(totals.values()) / len(esso_rounds)) if esso_rounds else None,
        'mean_ipopt_s_per_round_from_totals_by_agent': ({a: v / len(esso_rounds) for a, v in totals.items()}
                                                        if esso_rounds else None),
        'mean_ipopt_s_per_round_note': ('totals over every logged solve (retries included) divided by the number of '
                                        'ESSO-logged rounds (init + cycles); valid without round alignment'),
        'ipopt_s_total_all_rounds_by_agent': totals,
        'ipopt_s_total_all_rounds': sum(totals.values()),
        'cycle_rows_with_wall': sum(1 for r in cyc_rows if 'cycle_wall_s' in r),
        'mean_ipopt_s_per_cycle': (statistics.fmean([r['ipopt_s_total'] for r in cyc_rows])
                                   if cyc_rows and recs is not None else None),
        'mean_ipopt_s_per_cycle_by_agent': ({a: statistics.fmean([r['ipopt_s_by_agent'].get(a, 0.0) for r in cyc_rows])
                                            for a in totals} if cyc_rows and recs is not None else None),
        'mean_cycle_wall_s': (statistics.fmean([r['cycle_wall_s'] for r in cyc_rows if 'cycle_wall_s' in r])
                              if any('cycle_wall_s' in r for r in cyc_rows) else None),
        'ipopt_share_of_cycle_wall': _stats([r.get('ipopt_share_of_cycle_wall') for r in cyc_rows]),
        'largest_single_network_block_mean_s': max(
            [(v['mean_s'], k) for k, v in by_block.items() if not k.startswith('ESSO') and v.get('mean_s')] or [(None, None)]),
    }
    return {'logs_dir': logs, 'present': True, 'counts': counts, 'exit_messages': exits,
            'network_solve_records_path': recs_p if recs is not None else None,
            'summary': summary, 'by_block': by_block, 'per_round': rows,
            'semantics': ('IPOPT 3.14 "Total seconds in IPOPT": wall time inside the IPOPT process for one solve '
                          '(excludes Pyomo NL write, process start-up, .sol read and load_from); rounds aligned by '
                          'the evaluation\'s own network_ipopt_solve_records.jsonl (round 0 = initialisation); ESSO '
                          'per-cycle log files carry the round in their name')}


def main():
    if os.path.exists(OUT_DIR):
        log(f'REFUSING: output directory exists: {OUT_DIR}')
        return 1
    started = datetime.now(timezone.utc).isoformat()
    campaigns = []
    for group, pattern in CAMPAIGNS:
        roots = sorted(r for r in glob.glob(pattern) if os.path.isdir(r))
        if not roots:
            log(f'no campaign root matches {pattern}')
        for root in roots:
            cf = campaign_facts(group, root)
            log(f'{group}: {root} conc={cf["spec_concurrency"]} max_obs={cf["max_concurrent_observed"]} '
                f'evals={cf["n_evals"]}')
            campaigns.append(cf)
    breakdowns = []
    for root, name in IPOPT_BREAKDOWN:
        dirs = ([os.path.join(root, 'evals', name)] if name else
                sorted(d.rstrip('/') for d in glob.glob(os.path.join(root, 'evals', '*/'))))
        for d in dirs:
            rec_p = os.path.join(d, 'evaluation_record.json')
            if not os.path.exists(rec_p):
                breakdowns.append({'eval_dir': d, 'error': 'no evaluation_record.json'})
                continue
            rec = read_json(rec_p)
            run_id = (rec.get('working_dir_ids') or {}).get('run')
            b = ipopt_breakdown(d, run_id)
            b.update({'eval_dir': d, 'label': rec.get('candidate_label'), 'eval_key': rec.get('eval_key'),
                      'candidate_key': rec.get('candidate_key'), 'run_id': run_id})
            s = b.get('summary') or {}
            log(f'IPOPT {d}: total {s.get("ipopt_s_total_all_rounds")} s; per cycle {s.get("mean_ipopt_s_per_cycle")}; '
                f'per round (totals) {s.get("mean_ipopt_s_per_round_from_totals")}; '
                f'cycle wall {s.get("mean_cycle_wall_s")}')
            breakdowns.append(b)
    # three instance files and the case-file name the task cited
    instance_files = {}
    for p in ('data/SRP1/SRP1.json', 'data/SRP1/SRP1_5x3y_1x1.json',
              'data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json', 'data/SRP1/SRP1__s53_3x3.json'):
        instance_files[p] = read_json(p) if os.path.exists(p) else 'ABSENT'
        if instance_files[p] != 'ABSENT':
            d = instance_files[p]
            instance_files[p] = {'sha256': READ[p], 'Years': d.get('Years'), 'NumMarketScenarios':
                                 d.get('NumMarketScenarios'), 'ParallelExecution': d.get('ParallelExecution'),
                                 'tn_operation_scenarios': (d.get('TransmissionNetwork') or {}).get(
                                     'num_operation_scenarios'),
                                 'dn_operation_scenarios': [x.get('num_operation_scenarios')
                                                            for x in d.get('DistributionNetworks', [])]}
    guard_fail = GUARD.verify(0)
    payload = {
        'stage': 'P5.15 S56 input audit part B (zero solves, read-only extraction)',
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'script': os.path.basename(__file__), 'script_sha256': sha256_file(os.path.abspath(__file__)),
        'objective_convention': 'certified_cost_gross_eur = gross_operational_cost, settlement excluded (quoted only to '
                                'identify cells; no objective is compared here)',
        'units': {'wall': 's', 'rss/footprint': 'bytes (macOS ru_maxrss is bytes)'},
        'instance_files': instance_files,
        'campaigns': campaigns, 'ipopt_breakdowns': breakdowns,
        'guards': {'solve_guard_counts': GUARD.counts, 'solve_guard_verify_0_failures': guard_fail,
                   'pickle_counts': PICKLE_COUNTS},
    }
    os.makedirs(OUT_DIR)
    with open(OUT_JSON, 'w') as f:
        json.dump(payload, f, indent=1, sort_keys=False)
    log(f'guards: solve {GUARD.counts} verify(0) failures {guard_fail}; pickle {PICKLE_COUNTS}')
    log(f'wrote {OUT_JSON}')
    with open(OUT_LOG, 'w') as f:
        f.write('\n'.join(LOG_LINES) + '\n')
    manifest = {'inputs_read_sha256': dict(sorted(READ.items())),
                'outputs_sha256': {OUT_JSON: sha256_file(OUT_JSON), OUT_LOG: sha256_file(OUT_LOG),
                                   os.path.basename(__file__): sha256_file(os.path.abspath(__file__))},
                'n_inputs': len(READ)}
    with open(OUT_MAN, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'manifest {OUT_MAN} sha256 {sha256_file(OUT_MAN)}; n_inputs {len(READ)}', flush=True)
    return 0 if (not guard_fail and PICKLE_COUNTS == {'load': 0, 'loads': 0}) else 1


if __name__ == '__main__':
    sys.exit(main())
