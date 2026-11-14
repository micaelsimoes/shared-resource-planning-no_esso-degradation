"""P5.15 W131 -- three records-only diagnostics before the v3 campaign spec is frozen. ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (the imported W112 module installs its own zero-permit guard; it is verified at 0 as well). Nothing is built,
unpickled or solved; only JSON / JSONL records and IPOPT text logs are read.

TASK 1 -- non-Optimal accepted solves and criterion v3 replays.
  Cells: the 10 W118 r2 re-settle cells (8 certified, 2 F2 uncertified) and the two settled SRP1 references x0 d110bd1a
  (W102) and unit 3f084f2f (W103).
  SOURCES (per cell):
    network blocks (12 TSO + 36 DSO per cycle): network_ipopt_solve_records.jsonl (committed; one record per IPOPT
      attempt, field `round` = the ADMM cycle, 0 = initialisation; `exit` = the IPOPT EXIT message parsed from the
      block's log). FINAL ACCEPTED ATTEMPT of a block in a cycle = its last record in file order for that (round, agent,
      network, year, day). Cross-checked: (i) network_failures_s39_D.jsonl lists exactly the (cycle, block) pairs that
      have a non-primary attempt; (ii) the EXIT line inside each record's log byte range (log file on disk, sha256
      recorded here) equals the record's `exit`.
    ESSO solves (3 per cycle, nodes 5/7/9): NOT in the network records. Source = the per-solve ESSO IPOPT logs
      optim_log_esso_node{N}_{init|cycleNNN}[_recovery[_tier2]].txt in the run's logs directory (untracked; every file
      read is sha256-recorded in w131_log_inventory.json). The final attempt is tier2 > recovery > primary (the
      production naming of shared_energy_storage_data._create_solver / _optimize). Cross-checked against
      esso_recovery_events_s39_D.jsonl (committed) and leak_classification_s39_D.jsonl's log_path (committed).
  all_optimal_k := every final accepted attempt of the 48 network blocks and 3 ESSO solves of cycle k exited
    "Optimal Solution Found." (IPOPT's text).
  v3-alpha: the committed rule (settling_criterion_v2.SettlingRuleV2 for the W118 cells, settling_criterion.SettlingRule
    for the W101 references), fed boyd_k AND all_optimal_k; everything else as the certifying run.
  gamma: the same committed rule fed the ORIGINAL boyd_k (no reset), with the branch verdict vetoed at any cycle whose
    certifying window (the oscillatory window [k-W+1, k], or the monotone window [k-L+1, k]) contains a non-Optimal
    cycle (subclass override of `evaluate`; the module's clauses are called, not re-implemented; the monotone
    conjunction is re-evaluated from the module's own b_parts only when the oscillatory verdict is vetoed).
  The replays can only run over the recorded cycles; a certified run stops at k*, so "not certified by k*" means "not
  certified within the recorded run", not "not certified by the cap".

TASK 2 -- H_regime on cell 2ab0ce2d (campaign s53_f2_certificate_r1, N_old 337, original regime).
  t_sum(k) = sum over pf p-entries w * pi * (x_dso - z_tso_current), from W112's own functions (_price_weight,
  _stream_stride) on the cell's pf_entry_stride_s39_D.jsonl and interface_settlement_detail_s31c.json.
  VERDICT RULE (recorded here before anything is computed):
    k_close := the first cycle c such that |t_sum(m)| <= GAP_BOUND (= TAU / 2 = 2,269.53 EUR, the gap clause) for
      every m in [c, 337]; none if |t_sum(337)| > GAP_BOUND.
    V := [k_close - 10, k_close] (the cycle and the preceding 10).
    rho_pf rises in V := some k in V has rho_pf_after(k) > rho_pf_after(k - 1) (per_cycle_record rho actions) or the
      stride's rho_pf in force at k above that at k - 1.
    AA step accepted in V := some k in V has aa_accepted True (aa_per_cycle.jsonl).
    k_close exists and (rho_pf rises in V or AA accepted in V)       -> "H_regime supported"
    k_close exists and rho_pf constant over V and no AA accepted in V -> "W127 slow-dual reading"
    no k_close                                                        -> "not closed"
  Dominant residual entries (as W126): entries ranked by their mean share of ||r||^2 (stride entry r) over
  BEFORE = [k_close - 10, k_close - 1] and AFTER = [k_close, min(k_close + 10, 337)]; the prefix reaching 99 %.

TASK 3 -- SoH floor of the S46 ageing cells and the S45 a1a C3 unit at their old certificates.
  Sources: evaluation_record.json ageing_trajectory_terminal (per node, per block year: soh_prev_end, soh_end) and the
  soh_floor_sidecar_baseline.jsonl row of the certification cycle (per node, cohort y_inv, year y:
  es_soh_per_unit_cumul, soh_min, dual, active), cross-checked with boyd_terminal.json's terminal copy.
  min SoH per node and year = min over cohorts of es_soh_per_unit_cumul[y_inv, y]; flag < 0.70 (the current case file).

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w131_prefreeze/):
  w131_prefreeze_diagnostics.json, w131_log_inventory.json, launch.log, manifest_sha256.json
Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w131_prefreeze && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w131_prefreeze_diagnostics.py \\
        > data/SRP1/Results/P515S53/w131_prefreeze/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w131_prefreeze_diagnostics.py --manifest
"""
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W131 prefreeze diagnostics zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import p515_s53_w112_consensus_gap as W112  # noqa: E402 -- t_sum-from-stride functions (installs its own guard)

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results')
S53 = os.path.join(RES, 'P515S53')
OUT_DIR = os.path.join(REPO, S53, 'w131_prefreeze')
OUT_JSON = os.path.join(OUT_DIR, 'w131_prefreeze_diagnostics.json')
OUT_INV = os.path.join(OUT_DIR, 'w131_log_inventory.json')

OPTIMAL = 'Optimal Solution Found.'
N_NETWORK_BLOCKS = 48
N_ESSO = 3

W118 = os.path.join(S53, 'w118_resettle')
W101 = os.path.join(S53, 'w101_srp1_continuation')
CELLS_V2 = {
    'pb_y2030_n9': 'campaign_s53_w118_resettle_r2_pb_y2030_n9/evals/344c936fc23919be_pb_y2030_n9',
    'pb_y2030_n7': 'campaign_s53_w118_resettle_r2_pb_y2030_n7/evals/828f03fa9ac2ca6e_pb_y2030_n7',
    'pb_y2025_n5': 'campaign_s53_w118_resettle_r2_pb_y2025_n5/evals/ca29c5e818366a6e_pb_y2025_n5',
    'pb_y2030_n5': 'campaign_s53_w118_resettle_r2_pb_y2030_n5/evals/3846675acea98ea7_pb_y2030_n5',
    'pb_y2025_n9': 'campaign_s53_w118_resettle_r2_pb_y2025_n9/evals/b7bce5a8bf2365b9_pb_y2025_n9',
    'pb_y2025_n7': 'campaign_s53_w118_resettle_r2_pb_y2025_n7/evals/b9b8e4be4a6c82b6_pb_y2025_n7',
    'yl_y2030': 'campaign_s53_w118_resettle_r2_yl_y2030/evals/6e4f11a95dc3d585_yl_y2030',
    'yl_y2035': 'campaign_s53_w118_resettle_r2_yl_y2035/evals/3a2387f46448f9db_yl_y2035',
    'f2_challenger': 'campaign_s53_w118_resettle_r2_f2_challenger/evals/1fe91e86f11e76af_f2_challenger',
    'f2_incumbent': 'campaign_s53_w118_resettle_r2_f2_incumbent/evals/24c5ccb6f285219f_f2_incumbent',
}
CELLS_V1 = {
    'x0': 'campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0',
    'unit_n7_4h_e1': 'campaign_s53_w101_srp1_cont_n7_4h_e1/evals/3f084f2ffaeef2b7_n7_4h_e1',
}

T2_EVAL = os.path.join(S53, 'campaign_s53_f2_certificate_r1', 'evals',
                       '2ab0ce2dbd07d42c_y2030__n5_p0_25_e0_5__n7_p1_25_e3_m2')
T2_N_OLD = 337
T2_W117_T_SUM = -1806.5195560455322          # w117_triage_recompute.json cells.2ab0ce2d.t_sum (cross-check only)
T2_PRE = 10
T2_POST = 10
T2_DOMINANT_CUM = 0.99
T2_LIST_MAX = 20
T2_VERDICTS = ('H_regime supported', 'W127 slow-dual reading', 'not closed')

T3_CELLS = {
    'C2': os.path.join(RES, 'P515S46', 'campaign_s46_ageing', 'evals', 'c6b53015fcf65e24_n7_4h_e1_c2'),
    'C4': os.path.join(RES, 'P515S46', 'campaign_s46_ageing', 'evals', '65a5da775d1ff5b2_n7_4h_e1_c4'),
    'C2_calfade': os.path.join(RES, 'P515S46', 'campaign_s46_ageing', 'evals', '98e2857016a16d1c_n7_4h_e1_c2_calfade'),
    'C3_midblock': os.path.join(RES, 'P515S46', 'campaign_s46_ageing', 'evals', 'ed4a1acc7059784d_n7_4h_e1_c3_midblock'),
    'no_ageing': os.path.join(RES, 'P515S46', 'campaign_s46_ageing', 'evals', '06f092d164f13819_n7_4h_e1_no_ageing'),
    'C3_unit_S45_a1a': os.path.join(RES, 'P515S45', 'campaign_s45_a1a', 'evals', '7eb1ce62c2509f54_n7_4h_e1'),
}
T3_FLOOR_CURRENT = 0.70
CASE_ESS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')

INVENTORY = {}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_state(rels):
    tracked = set(subprocess.run(['git', 'ls-files', '--'] + rels, cwd=REPO, capture_output=True, text=True,
                                 check=True).stdout.split('\n'))
    dirty = set(line[3:] for line in subprocess.run(['git', 'status', '--porcelain', '--'] + rels, cwd=REPO,
                                                    capture_output=True, text=True, check=True).stdout.split('\n')
                if line.strip())
    return {r: {'tracked': r in tracked, 'clean_vs_HEAD': (r in tracked) and r not in dirty} for r in rels}


def _campaign_manifest(eval_rel):
    root = eval_rel.split(os.sep + 'evals' + os.sep)[0]
    path = os.path.join(root, 'campaign_manifest_sha256.json')
    return path, json.load(open(os.path.join(REPO, path)))


def _verify(eval_rel, names):
    """sha256 of each named input against the campaign manifest that hash-records it; git state recorded; a tracked
    input must be clean against HEAD."""
    man_rel, man = _campaign_manifest(eval_rel)
    rels = [os.path.join(eval_rel, n) for n in names]
    git = _git_state(rels)
    out = {}
    for rel in rels:
        now = _sha(os.path.join(REPO, rel))
        pinned = man.get(rel)
        if pinned is None:
            raise RuntimeError(f'{rel}: not hash-recorded in {man_rel}')
        if now != pinned:
            raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned}')
        if git[rel]['tracked'] and not git[rel]['clean_vs_HEAD']:
            raise RuntimeError(f'{rel}: tracked but modified against HEAD')
        out[rel] = {'sha256': now, 'manifest': man_rel, **git[rel]}
    return out


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _campaign_spec(eval_rel):
    root = eval_rel.split(os.sep + 'evals' + os.sep)[0]
    specs = [f for f in os.listdir(os.path.join(REPO, root)) if f.startswith('campaign_spec_') and f.endswith('.json')]
    if len(specs) != 1:
        raise RuntimeError(f'{root}: campaign specs {specs}')
    return os.path.join(root, specs[0]), json.load(open(os.path.join(REPO, root, specs[0])))


# ======================================================================================================================
#  TASK 1 -- sources
# ======================================================================================================================

_EXIT_RE = re.compile(rb'EXIT: ([^\n]*)')


def _exit_in_bytes(path, lo, hi):
    with open(path, 'rb') as handle:
        start = max(lo, hi - 16384)
        handle.seek(start)
        blob = handle.read(hi - start)
    found = _EXIT_RE.findall(blob)
    return found[-1].decode().strip() if found else None


def _esso_exit(path):
    with open(path, 'rb') as handle:
        blob = handle.read()
    found = _EXIT_RE.findall(blob)
    return (found[-1].decode().strip() if found else None), len(found)


def network_final_attempts(eval_rel, n_cycles):
    recs = _jsonl(os.path.join(eval_rel, 'network_ipopt_solve_records.jsonl'))
    chains = collections.OrderedDict()
    for i, r in enumerate(recs):
        key = (r['round'], r['agent'], r['network'], str(r['year']), r['day'])
        chains.setdefault(key, []).append(r)
    rounds = collections.Counter(k[0] for k in chains)
    bad_rounds = {k: v for k, v in rounds.items() if v != N_NETWORK_BLOCKS}
    if bad_rounds or sorted(rounds) != list(range(0, n_cycles + 1)):
        raise RuntimeError(f'{eval_rel}: network rounds {sorted(rounds)[:3]}..{max(rounds)} bad {bad_rounds}')
    agents_per_round = collections.Counter(k[1][:3] for k in chains if k[0] == 1)
    if agents_per_round != {'TSO': 12, 'DSO': 36}:
        raise RuntimeError(f'{eval_rel}: agents per round {agents_per_round}')
    order_ok = all(c[0]['attempt'] == 'primary' and len({a['attempt'] for a in c}) == len(c) for c in chains.values())
    # log-byte cross-check of every record's exit
    log_check = {'n_records': len(recs), 'agree': 0, 'disagree': [], 'log_missing': 0}
    files = set()
    for r in recs:
        p = r['log_path']
        if not os.path.exists(p):
            log_check['log_missing'] += 1
            continue
        files.add(p)
        e = _exit_in_bytes(p, r['log_bytes'][0], r['log_bytes'][1])
        if e == r['exit'] or (e is not None and r['exit'] is not None and e == r['exit'].strip()):
            log_check['agree'] += 1
        else:
            log_check['disagree'].append({'round': r['round'], 'agent': r['agent'], 'network': r['network'],
                                          'year': r['year'], 'day': r['day'], 'attempt': r['attempt'],
                                          'record_exit': r['exit'], 'log_exit': e})
    for p in sorted(files):
        INVENTORY[os.path.relpath(p, REPO)] = {'sha256': _sha(p), 'bytes': os.path.getsize(p), 'kind': 'network_log'}
    finals = {}
    for key, chain in chains.items():
        f = chain[-1]
        finals[key] = {'final_attempt': f['attempt'], 'final_exit': f['exit'], 'n_attempts': len(chain),
                       'primary_exit': chain[0]['exit'], 'attempts': [a['attempt'] for a in chain]}
    return finals, {'n_records': len(recs), 'n_blocks_x_rounds': len(chains), 'attempt_order_primary_first': order_ok,
                    'log_byte_crosscheck': {**log_check, 'n_disagree': len(log_check['disagree']),
                                            'n_log_files_hashed': len(files)},
                    'exit_counts_all_attempts': dict(collections.Counter(r['exit'] for r in recs)),
                    'attempt_counts': dict(collections.Counter(r['attempt'] for r in recs))}


def esso_final_attempts(eval_rel, logs_dir, n_cycles):
    leak = _jsonl(os.path.join(eval_rel, 'leak_classification_s39_D.jsonl'))
    events = _jsonl(os.path.join(eval_rel, 'esso_recovery_events_s39_D.jsonl'))
    nodes = sorted({int(x['node_id']) for x in leak})
    if len(nodes) != N_ESSO:
        raise RuntimeError(f'{eval_rel}: ESSO nodes {nodes}')
    leak_paths = {(int(x['node_id']), x['cycle']): x['log_path'] for x in leak}
    listing = set(f for f in os.listdir(logs_dir) if f.startswith('optim_log_esso_'))
    expected = set()
    finals = {}
    unexpected_variants = []
    missing = []
    leak_path_mismatch = []
    for k in range(0, n_cycles + 1):
        stamp = 'init' if k == 0 else f'cycle{k:03d}'
        for n in nodes:
            base = f'optim_log_esso_node{n}_{stamp}'
            chain = [(a, f'{base}{suf}.txt') for a, suf in (('primary', ''), ('recovery', '_recovery'),
                                                               ('recovery_tier2', '_recovery_tier2'))]
            present = [(a, f) for a, f in chain if f in listing]
            for _, f in chain:
                expected.add(f)
            dups = sorted(f for f in listing if f.startswith(base + '_dup') or re.match(re.escape(base) + r'_recovery.*_dup', f))
            if dups:
                unexpected_variants += dups
            if not present or present[0][0] != 'primary':
                missing.append((k, n))
                continue
            lp = leak_paths.get((n, 'init' if k == 0 else f'{k:03d}'))
            if lp is not None and os.path.basename(lp) != present[0][1]:
                leak_path_mismatch.append({'cycle': k, 'node': n, 'leak_log_path': lp})
            exits = []
            for a, f in present:
                p = os.path.join(logs_dir, f)
                e, n_exit = _esso_exit(p)
                INVENTORY[os.path.relpath(p, REPO)] = {'sha256': _sha(p), 'bytes': os.path.getsize(p),
                                                       'kind': 'esso_log', 'n_exit_lines': n_exit}
                exits.append((a, e))
            finals[(k, n)] = {'final_attempt': exits[-1][0], 'final_exit': exits[-1][1], 'n_attempts': len(exits),
                              'primary_exit': exits[0][1]}
    stray = sorted(listing - expected)
    n_retry = sum(1 for v in finals.values() if v['n_attempts'] > 1)
    return finals, {'nodes': nodes, 'n_logs_listed': len(listing), 'n_solves': len(finals), 'missing': missing,
                    'stray_files_not_in_naming_scheme': stray, 'dup_variants': unexpected_variants,
                    'n_with_retry_logs': n_retry, 'n_esso_recovery_events_committed': len(events),
                    'retry_logs_match_recovery_events': n_retry == len(events),
                    'leak_classification_log_path_mismatches': leak_path_mismatch,
                    'exit_counts_final': dict(collections.Counter(v['final_exit'] for v in finals.values()))}


def cycle_optimality(eval_rel, n_cycles):
    net, net_meta = network_final_attempts(eval_rel, n_cycles)
    recs = _jsonl(os.path.join(eval_rel, 'network_ipopt_solve_records.jsonl'))
    logs_dir = os.path.dirname(recs[0]['log_path'])
    esso, esso_meta = esso_final_attempts(eval_rel, logs_dir, n_cycles)
    # network_failures cross-check: (cycle, block) pairs with a non-primary attempt
    fails = _jsonl(os.path.join(eval_rel, 'network_failures_s39_D.jsonl'))
    retried = sorted((k[0], k[2], str(k[3]), k[4]) for k, v in net.items() if v['n_attempts'] > 1)
    listed = sorted((int(f['cycle']), f['network_name'], str(f['year']), f['day']) for f in fails)
    per_cycle = {}
    nonopt = []
    for (rnd, agent, network, year, day), v in net.items():
        pc = per_cycle.setdefault(rnd, {'n_network': 0, 'n_esso': 0, 'non_optimal': []})
        pc['n_network'] += 1
        if v['final_exit'] != OPTIMAL:
            item = {'cycle': rnd, 'family': 'network', 'agent': agent, 'network': network, 'year': year, 'day': day,
                    'final_attempt': v['final_attempt'], 'final_exit': v['final_exit'],
                    'primary_exit': v['primary_exit'], 'attempts': v['attempts']}
            pc['non_optimal'].append(item)
            nonopt.append(item)
    for (k, n), v in esso.items():
        pc = per_cycle.setdefault(k, {'n_network': 0, 'n_esso': 0, 'non_optimal': []})
        pc['n_esso'] += 1
        if v['final_exit'] != OPTIMAL:
            item = {'cycle': k, 'family': 'esso', 'node': n, 'final_attempt': v['final_attempt'],
                    'final_exit': v['final_exit'], 'primary_exit': v['primary_exit']}
            pc['non_optimal'].append(item)
            nonopt.append(item)
    coverage = all(per_cycle[k]['n_network'] == N_NETWORK_BLOCKS and per_cycle[k]['n_esso'] == N_ESSO
                   for k in range(0, n_cycles + 1))
    all_opt = {k: not per_cycle[k]['non_optimal'] for k in range(1, n_cycles + 1)}
    meta = {'network': net_meta, 'esso': esso_meta, 'logs_dir': os.path.relpath(logs_dir, REPO),
            'coverage_every_cycle_48_network_3_esso': coverage, 'cycles_covered': [0, n_cycles],
            'network_failures_crosscheck': {'retried_blocks_in_records': len(retried), 'listed_in_failures': len(listed),
                                            'identical': retried == listed}}
    return all_opt, sorted(nonopt, key=lambda x: (x['cycle'], x['family'])), meta


# ======================================================================================================================
#  TASK 1 -- rules (committed modules; gamma = a veto on the branch verdict)
# ======================================================================================================================

def _gamma_class(base, monotone_formula):
    class Gamma(base):
        nonopt_cycles = frozenset()

        def evaluate(self, k):
            cert_a, cert_b, a_parts, b_parts, reasons = super().evaluate(k)
            vetoed = []
            if cert_a and any(lo <= c <= hi for c in self.nonopt_cycles for lo, hi in [a_parts['window']]):
                cert_a = False
                vetoed.append('oscillatory')
                cert_b = monotone_formula(b_parts)
            if cert_b and any(b_parts['window'][0] <= c <= b_parts['window'][1] for c in self.nonopt_cycles):
                cert_b = False
                vetoed.append('monotone')
            if vetoed:
                reasons = list(reasons) + ['gamma_non_optimal_cycle_in_window']
                self.gamma_vetoes.append({'cycle': k, 'branches': vetoed,
                                          'window_a': a_parts['window'], 'window_b': b_parts['window'],
                                          'v1_replay_phase_no_decision': isinstance(self, SC1.SettlingRule)
                                          and k <= self.n})
            return cert_a, cert_b, a_parts, b_parts, reasons
    return Gamma


def _mono_v1(b):   # settling_criterion.SettlingRule.evaluate's monotone conjunction, from its own b_parts
    return bool(b['lo_ge_k0_plus_K_EXCL'] and b['no_sign_change_in_window'] and b['range_le_tau']
                and (b['stationary'] or b['strictly_decreasing']))


def _mono_v2(b):   # settling_criterion_v2.SettlingRuleV2.evaluate's monotone conjunction, from its own b_parts
    return bool(b['lo_ge_k0_plus_K_EXCL'] and b['no_sign_change_in_window'] and b['range_le_tau']
                and b['steps_decreasing'] and b['last_step_times_L_le_tau'])


GammaV1 = _gamma_class(SC1.SettlingRule, _mono_v1)
GammaV2 = _gamma_class(SC2.SettlingRuleV2, _mono_v2)


def _summ(dec, last_k):
    if dec is None:
        return {'status': f'not certified within the recorded cycles 1..{last_k}', 'k_star': None}
    keys = ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W', 'window', 'range', 't_sum_k_star', 'Q_k_star')
    out = {k: dec.get(k) for k in keys}
    out['lapse_events'] = dec.get('lapse_events')
    out['n_gap_refusals'] = len(dec.get('gap_refusals') or [])
    return out


def _run_v2(rule, q, boyd, t, last):
    for k in range(1, last + 1):
        if k > rule.effective_cap():
            break
        rule.observe(k, q.get(k), bool(boyd.get(k, False)), t.get(k))
        if rule.decision is not None:
            break
    return rule.decision


def _run_v1(rule, q, boyd, last):
    for k in range(1, min(last, rule.cap) + 1):
        rule.observe(k, q[k], boyd[k])
        if rule.decision is not None and rule.decision.get('status') == 'certified':
            break
    return rule.decision


def _counts(nonopt, n_cycles, first_k0, cert_k0, window):
    cyc = sorted({x['cycle'] for x in nonopt if x['cycle'] >= 1})
    by_cycle = collections.Counter(x['cycle'] for x in nonopt if x['cycle'] >= 1)
    out = {'non_optimal_cycles': cyc, 'n_non_optimal_cycles': len(cyc),
           'n_non_optimal_solves_cycles_1_to_end': sum(by_cycle.values()),
           'n_non_optimal_solves_by_cycle': {str(k): v for k, v in sorted(by_cycle.items())},
           'init_round_0_non_optimal': [x for x in nonopt if x['cycle'] == 0]}
    for name, k0 in (('after_first_k0', first_k0), ('after_certifying_k0', cert_k0)):
        if k0 is None:
            out[name] = None
            continue
        sel = [c for c in cyc if c >= k0]
        out[name] = {'k0': k0, 'range': [k0, n_cycles], 'cycles': sel, 'n_cycles': len(sel),
                     'n_solves': sum(by_cycle[c] for c in sel)}
    if window is not None:
        inside = [c for c in cyc if window[0] <= c <= window[1]]
        out['certifying_window'] = {'window': window, 'non_optimal_cycles_inside': inside, 'any_inside': bool(inside)}
    else:
        out['certifying_window'] = None
    return out


def task1():
    cells = {}
    for cell, sub in list(CELLS_V2.items()) + list(CELLS_V1.items()):
        v1 = cell in CELLS_V1
        eval_rel = os.path.join(W101 if v1 else W118, sub)
        dec_name = 'settling_decision.json' if v1 else 'resettle_decision.json'
        line_name = 'settling_continuation_cycle_record.jsonl' if v1 else 'resettle_cycle_record.jsonl'
        inputs = _verify(eval_rel, ['per_cycle_record.jsonl', 'network_ipopt_solve_records.jsonl',
                                    'network_failures_s39_D.jsonl', 'esso_recovery_events_s39_D.jsonl',
                                    'leak_classification_s39_D.jsonl', dec_name, line_name])
        spec_rel, spec = _campaign_spec(eval_rel)
        inputs[spec_rel] = {'sha256': _sha(os.path.join(REPO, spec_rel)), **_git_state([spec_rel])[spec_rel]}
        rows = _jsonl(os.path.join(eval_rel, 'per_cycle_record.jsonl'))
        if [r['cycle'] for r in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f'{cell}: per_cycle_record cycles not 1..n')
        n_cycles = len(rows)
        lines = {x['cycle']: x for x in _jsonl(os.path.join(eval_rel, line_name))}
        dec = json.load(open(os.path.join(REPO, eval_rel, dec_name)))
        all_opt, nonopt, meta = cycle_optimality(eval_rel, n_cycles)
        q = {r['cycle']: r['gross_operational_cost'] for r in rows}
        boyd = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
        boyd3 = {k: boyd[k] and all_opt[k] for k in boyd}
        nonopt_cycles = frozenset(x['cycle'] for x in nonopt if x['cycle'] >= 1)
        # local_solves_ok consistency (report-only): a non-Optimal-but-accepted cycle keeps local_solves_ok True
        lso = {'cycles_local_solves_ok_false': [r['cycle'] for r in rows if not r['local_solves_ok']],
               'non_optimal_cycles_with_local_solves_ok_true': sorted(c for c in nonopt_cycles if rows[c - 1]['local_solves_ok'])}
        if v1:
            cand = spec['candidates'][0]['settling_continuation']
            n, cap, p_max = cand['hold_after_cycle'], spec['cap'], cand['settling_rule']['p_max']
            rule_desc = {'module': 'settling_criterion (v1)', 'N': n, 'cap': cap, 'p_max': p_max}
            base = _run_v1(SC1.SettlingRule(n, cap, p_max), q, boyd, n_cycles)
            a = _run_v1(SC1.SettlingRule(n, cap, p_max), q, boyd3, n_cycles)
            g_rule = GammaV1(n, cap, p_max)
            g_rule.nonopt_cycles, g_rule.gamma_vetoes = nonopt_cycles, []
            g = _run_v1(g_rule, q, boyd, n_cycles)
            first_k0 = None
            for k in range(1, n_cycles + 1):
                if q[k] is not None and boyd[k]:
                    first_k0 = k
                    break
        else:
            cr = spec['candidates'][0]['settling_resettle']['cap_rule']
            p_max = spec['candidates'][0]['settling_resettle']['settling_rule']['p_max']
            kw = ({'cap': cr['cap']} if cr['kind'] == 'fixed' else
                  {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
            rule_desc = {'module': 'settling_criterion_v2', 'p_max': p_max, 'cap_rule': cr}
            t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
            _, base, _ = SC2.replay(q, boyd, t, p_max, last=n_cycles, **kw)
            _, a, _ = SC2.replay(q, boyd3, t, p_max, last=n_cycles, **kw)
            g_rule = GammaV2(p_max, **kw)
            g_rule.nonopt_cycles, g_rule.gamma_vetoes = nonopt_cycles, []
            g = _run_v2(g_rule, q, boyd, t, n_cycles)
            first_k0 = dec.get('N')
        # the base replay must reproduce the committed decision
        repro = {k: (base or {}).get(k) == dec.get(k) for k in ('status', 'k_star', 'branch', 'k0', 'window')}
        if dec.get('status') == 'uncertified' and base is None:
            repro = {'status': False, 'note': 'base replay reached the record end without the cap decision'}
        certified = dec.get('status') == 'certified'
        window = dec.get('window') if certified else None
        cnt = _counts(nonopt, n_cycles, first_k0, dec.get('k0'), window)
        a_s, g_s = _summ(a, n_cycles), _summ(g, n_cycles)
        cells[cell] = {
            'eval_dir': eval_rel, 'campaign_spec': spec_rel, 'rule': rule_desc, 'cycles_recorded': n_cycles,
            'committed_decision': {k: dec.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W',
                                                             'window', 't_sum_k_star', 'cap', 'lapse_events')},
            'base_replay_reproduces_committed': repro, 'base_replay': _summ(base, n_cycles),
            'sources': meta, 'local_solves_ok_vs_optimal': lso,
            'counts': cnt, 'non_optimal_solves': nonopt,
            'v3_alpha': {**a_s, 'k_star_minus_committed': (a_s['k_star'] - dec['k_star'])
                         if (a_s.get('k_star') is not None and dec.get('k_star') is not None) else None,
                         'certificate_holds': (a_s.get('status') == 'certified' and a_s.get('k_star') == dec.get('k_star'))
                         if certified else None},
            'gamma': {**g_s, 'vetoes': g_rule.gamma_vetoes,
                      'certificate_holds': (g_s.get('status') == 'certified' and g_s.get('k_star') == dec.get('k_star'))
                      if certified else None},
            'inputs_sha256': inputs,
        }
        _log(f'T1 {cell}: cycles {n_cycles}; committed {dec.get("status")} k* {dec.get("k_star")} window {window}; '
             f'non-Optimal cycles {cnt["non_optimal_cycles"]}; in window {(cnt["certifying_window"] or {}).get("non_optimal_cycles_inside")}; '
             f'v3a {a_s.get("status")} {a_s.get("k_star")}; gamma {g_s.get("status")} {g_s.get("k_star")}; '
             f'repro {repro}; coverage {meta["coverage_every_cycle_48_network_3_esso"]}; '
             f'log x-check disagree {meta["network"]["log_byte_crosscheck"]["n_disagree"]}')
    return cells


# ======================================================================================================================
#  TASK 2
# ======================================================================================================================

def task2():
    names = ['per_cycle_record.jsonl', 'pf_entry_stride_s39_D.jsonl', 'aa_per_cycle.jsonl',
             'interface_settlement_detail_s31c.json']
    inputs = _verify(T2_EVAL, names)
    detail = json.load(open(os.path.join(REPO, T2_EVAL, 'interface_settlement_detail_s31c.json')))
    pi, w = W112._price_weight(detail)
    stride_rel = os.path.join(T2_EVAL, 'pf_entry_stride_s39_D.jsonl')
    per = W112._stream_stride(stride_rel, pi, w)
    pcr = {r['cycle']: r for r in _jsonl(os.path.join(T2_EVAL, 'per_cycle_record.jsonl'))}
    aa = {r['cycle']: r for r in _jsonl(os.path.join(T2_EVAL, 'aa_per_cycle.jsonl'))}
    n = max(per)
    if sorted(per) != list(range(1, n + 1)) or n != T2_N_OLD or sorted(pcr) != list(range(1, n + 1)):
        raise RuntimeError(f'T2: cycles stride {min(per)}..{n}, pcr {len(pcr)}')
    ident = W112._detail_identity(detail)
    t = {k: per[k]['t_sum'] for k in per}
    validation = {'t_sum_337': t[n], 'w117_t_sum': T2_W117_T_SUM, 'minus_w117': t[n] - T2_W117_T_SUM,
                  'terminal_identity_t_tso_plus_t_dso': ident['t_tso_plus_t_dso_terminal'],
                  'minus_terminal_identity': t[n] - ident['t_tso_plus_t_dso_terminal'], 'detail_identity': ident}
    bound = SC2.GAP_BOUND
    k_close = None
    if abs(t[n]) <= bound:
        k_close = n
        while k_close > 1 and abs(t[k_close - 1]) <= bound:
            k_close -= 1
    k_close_literal_2270 = None
    if abs(t[n]) < 2270.0:
        k_close_literal_2270 = n
        while k_close_literal_2270 > 1 and abs(t[k_close_literal_2270 - 1]) < 2270.0:
            k_close_literal_2270 -= 1
    first_ever_below = next((k for k in range(1, n + 1) if abs(t[k]) <= bound), None)
    out = {'eval_dir': T2_EVAL, 'n_cycles': n, 'gap_bound': bound, 'validation': validation,
           'k_close': k_close, 'k_close_literal_strict_2270': k_close_literal_2270,
           'first_cycle_ever_at_or_below_bound': first_ever_below,
           'cycles_above_bound_after_first_ever': [k for k in range(first_ever_below or n + 1, n + 1) if abs(t[k]) > bound],
           't_sum_by_cycle': {str(k): t[k] for k in range(1, n + 1)},
           't_by_node_by_cycle': {str(k): per[k]['t_by_node'] for k in range(1, n + 1)},
           'aa_accepted_counts_whole_run': dict(collections.Counter(bool(aa[k]['aa_accepted']) for k in aa)),
           'aa_action_counts_whole_run': dict(collections.Counter(aa[k]['aa_action'] for k in aa)),
           'rho_pf_after_distinct_whole_run': sorted({pcr[k]['rho_pf_after'] for k in pcr}),
           'inputs_sha256': inputs}
    rho_change_cycles = [k for k in range(2, n + 1) if pcr[k]['rho_pf_after'] != pcr[k - 1]['rho_pf_after']]
    out['rho_pf_after_change_cycles_whole_run'] = [{'cycle': k, 'from': pcr[k - 1]['rho_pf_after'],
                                                     'to': pcr[k]['rho_pf_after'], 'action': pcr[k]['rho_pf_action']}
                                                    for k in rho_change_cycles]
    if k_close is None:
        out['verdict'] = 'not closed'
        _log(f'T2: not closed; t_sum(337) {t[n]}')
        return out
    lo = max(1, k_close - T2_PRE)
    rows = []
    for k in range(lo, k_close + 1):
        rows.append({'cycle': k, 't_sum': t[k], 'abs_t_sum_le_bound': abs(t[k]) <= bound,
                     't_by_node': per[k]['t_by_node'], 'sum_abs_gap_mw_p': per[k]['sum_abs_gap_mw_p'],
                     'rho_pf_after': pcr[k]['rho_pf_after'], 'rho_pf_action': pcr[k]['rho_pf_action'],
                     'rho_pf_in_force_stride': per[k]['rho_pf_set'], 'rho_freeze_active': pcr[k].get('rho_freeze_active'),
                     'aa_enabled': aa[k].get('aa_enabled'), 'aa_accepted': aa[k].get('aa_accepted'),
                     'aa_action': aa[k].get('aa_action'), 'aa_reset': aa[k].get('aa_reset'),
                     'boyd_pf_primal_ratio': pcr[k]['boyd_pf_primal_ratio'], 'boyd_all_pass': pcr[k]['boyd_all_pass'],
                     'local_solves_ok': pcr[k]['local_solves_ok'], 'gross_operational_cost': pcr[k]['gross_operational_cost']})
    rises = [k for k in range(lo, k_close + 1) if k > 1 and (pcr[k]['rho_pf_after'] > pcr[k - 1]['rho_pf_after']
                                                          or max(per[k]['rho_pf_set']) > max(per[k - 1]['rho_pf_set']))]
    accepted = [k for k in range(lo, k_close + 1) if aa[k].get('aa_accepted')]
    if rises or accepted:
        verdict = 'H_regime supported'
    else:
        verdict = 'W127 slow-dual reading'
    out.update({'window_V': [lo, k_close], 'rows_V': rows, 'rho_pf_rise_cycles_in_V': rises,
                'aa_accepted_cycles_in_V': accepted, 'verdict': verdict,
                'aa_accepted_fraction_before_V': (sum(1 for k in range(1, lo) if aa[k].get('aa_accepted')) / (lo - 1))
                if lo > 1 else None})
    # dominant residual entries before / after (W126 ranking: mean share of ||r||^2)
    before = list(range(max(1, k_close - T2_PRE), k_close))
    after = list(range(k_close, min(k_close + T2_POST, n) + 1))
    keep = set(before) | set(after)
    ent = {}
    keys = None
    with open(os.path.join(REPO, stride_rel)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['cycle'] not in keep:
                continue
            kk = [(int(e['node_id']), str(e['year']), str(e['day']), e['power_type'], int(e['period']))
                  for e in row['entries']]
            if keys is None:
                keys = kk
            elif kk != keys:
                raise RuntimeError('T2: stride entry order changed')
            ent[row['cycle']] = [(e['r'], e['x_dso'] - e['z_tso_current'], e['interface_rating']) for e in row['entries']]

    def _rank(cycles):
        m = len(keys)
        mean_share = [0.0] * m
        mean_gap = [0.0] * m
        mean_tsum = [0.0] * m
        for k in cycles:
            tot = sum(r * r for r, _, _ in ent[k])
            for j, (r, g, _) in enumerate(ent[k]):
                mean_share[j] += (r * r / tot) / len(cycles)
                mean_gap[j] += g / len(cycles)
                n_, y_, d_, pt, p_ = keys[j]
                if pt == 'p':
                    mean_tsum[j] += w[(str(n_), y_, d_, p_)] * pi[(str(n_), y_, d_, p_)] * g / len(cycles)
        order = sorted(range(m), key=lambda j: -mean_share[j])
        cum, dom = 0.0, []
        for j in order:
            dom.append(j)
            cum += mean_share[j]
            if cum >= T2_DOMINANT_CUM:
                break

        def _row(j):
            n_, y_, d_, pt, p_ = keys[j]
            return {'node_id': n_, 'year': y_, 'day': d_, 'block': f'{y_} {d_}', 'power_type': pt, 'period': p_,
                    'hour': p_ + 1, 'mean_share_r2': mean_share[j], 'mean_gap_x_minus_z_mw': mean_gap[j],
                    'mean_t_sum_contribution_eur': mean_tsum[j] if pt == 'p' else None}
        agg = collections.defaultdict(float)
        for j in range(m):
            n_, y_, d_, pt, _ = keys[j]
            agg[f'{n_}|{y_}|{d_}|{pt}'] += mean_share[j]
        tsum_top = sorted((j for j in range(m) if keys[j][3] == 'p'), key=lambda j: -abs(mean_tsum[j]))[:10]
        return {'cycles': [cycles[0], cycles[-1]], 'n_dominant_99pct': len(dom),
                'dominant_listed': [_row(j) for j in dom[:T2_LIST_MAX]],
                'share_by_node_block_type_top10': dict(sorted(agg.items(), key=lambda kv: -kv[1])[:10]),
                'top10_t_sum_contributors': [_row(j) for j in tsum_top]}
    out['dominant_residual_before'] = _rank(before) if before else None
    out['dominant_residual_after'] = _rank(after)
    _log(f'T2: k_close {k_close} (literal <2270: {k_close_literal_2270}); V {lo}..{k_close}; rho rises {rises}; '
         f'AA accepted {accepted}; verdict {verdict}; t_sum(337) {t[n]} (W117 diff {validation["minus_w117"]}, '
         f'identity diff {validation["minus_terminal_identity"]})')
    return out


# ======================================================================================================================
#  TASK 3
# ======================================================================================================================

def task3():
    out = {}
    case = json.load(open(os.path.join(REPO, CASE_ESS)))
    for cell, rel in T3_CELLS.items():
        inputs = _verify(rel, ['evaluation_record.json', 'soh_floor_sidecar_baseline.jsonl', 'boyd_terminal.json'])
        rec = json.load(open(os.path.join(REPO, rel, 'evaluation_record.json')))
        bt = json.load(open(os.path.join(REPO, rel, 'boyd_terminal.json')))
        side = _jsonl(os.path.join(rel, 'soh_floor_sidecar_baseline.jsonl'))
        kc = rec.get('certification_cycle')
        last = side[-1]
        if [r['cycle'] for r in side] != list(range(1, len(side) + 1)) or last['cycle'] != kc:
            raise RuntimeError(f'T3 {cell}: sidecar cycles 1..{last["cycle"]} vs certification {kc}')
        term = bt.get('soh_floor_multiplier_and_efc_per_cohort_year_terminal') or {}
        bt_rows = term.get('per_node_per_cohort_year')
        bt_match = None
        if bt_rows is not None:
            strip = lambda rows: [{k: r[k] for k in ('node_id', 'y_inv', 'y', 'dual', 'es_soh_per_unit_cumul',
                                                     'soh_min', 'active')} for r in rows]
            bt_match = strip(bt_rows) == strip(last['entries']) and term.get('cycle') == kc
        per_node_year = {}
        for e in last['entries']:
            key = (str(e['node_id']), str(e['y']))
            cur = per_node_year.get(key)
            if cur is None or e['es_soh_per_unit_cumul'] < cur['min_soh']:
                per_node_year[key] = {'node_id': e['node_id'], 'y': e['y'], 'min_soh': e['es_soh_per_unit_cumul'],
                                      'argmin_cohort_y_inv': e['y_inv'], 'dual': e['dual'], 'active': e['active'],
                                      'soh_min_in_run': e['soh_min'], 'efc_per_day': e.get('efc_per_day')}
        rows = sorted(per_node_year.values(), key=lambda r: (int(r['node_id']), int(r['y'])))
        for r in rows:
            r['below_0_70'] = r['min_soh'] < T3_FLOOR_CURRENT
            r['margin_to_0_70'] = r['min_soh'] - T3_FLOOR_CURRENT
        traj = rec.get('ageing_trajectory_terminal')
        traj_rows = None
        if traj:
            traj_rows = []
            for nid, v in traj['nodes'].items():
                for c in v['cells']:
                    traj_rows.append({'node_id': nid, 'investment_year': c['investment_year'], 'block_year': c['block_year'],
                                      'y': c['y'], 'soh_prev_end': c['soh_prev_end'], 'soh_end': c['soh_end'],
                                      'soh_mid_closed_form': c.get('soh_mid_closed_form'),
                                      'soh_used_for_available_energy': c.get('soh_used_for_available_energy'),
                                      'efc_per_day': c.get('efc_per_day'), 'below_0_70': c['soh_end'] < T3_FLOOR_CURRENT})
        all_min = min(e['es_soh_per_unit_cumul'] for r in side for e in r['entries'])
        duals = [e['dual'] for e in last['entries'] if e['dual'] is not None]
        out[cell] = {
            'eval_dir': rel, 'status': rec.get('status'), 'certification_cycle': kc, 'cycles_run': rec.get('cycles_run'),
            'model_variant': rec.get('model_variant'), 'model_variant_label': rec.get('model_variant_label'),
            'soh_min_in_run': sorted({e['soh_min'] for e in last['entries']}),
            'dual_sign_convention': last.get('dual_sign_convention'),
            'terminal_min_soh_per_node_year': rows,
            'terminal_trajectory_evaluation_record': traj_rows,
            'terminal_trajectory_available': traj_rows is not None,
            'min_soh_terminal': min(r['min_soh'] for r in rows),
            'any_below_0_70_terminal': any(r['below_0_70'] for r in rows),
            'min_soh_over_all_cycles': all_min,
            'floor_rows_active_at_terminal': sum(1 for e in last['entries'] if e['active']),
            'floor_row_duals_terminal_max_abs': max((abs(d) for d in duals), default=None),
            'boyd_terminal_copy_matches_sidecar': bt_match,
            'inputs_sha256': inputs,
        }
        _log(f'T3 {cell}: k_cert {kc}; soh_min in run {out[cell]["soh_min_in_run"]}; min terminal '
             f'{out[cell]["min_soh_terminal"]:.6f}; below 0.70 {out[cell]["any_below_0_70_terminal"]}; '
             f'max |dual| {out[cell]["floor_row_duals_terminal_max_abs"]}; bt match {bt_match}')
    return {'current_case_file_minimum_soh': case['ageing']['minimum_soh'],
            'current_case_file_sha256': _sha(os.path.join(REPO, CASE_ESS)), 'cells': out}


def run():
    t0 = time.time()
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_INV):
        raise RuntimeError(f'{OUT_DIR}: outputs exist; write-once')
    os.makedirs(OUT_DIR, exist_ok=True)
    _log(f'W131 start; git HEAD {subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()}')
    t1 = task1()
    t2 = task2()
    t3 = task3()
    guards = {'w131': {'counts': dict(_GUARD.counts), 'verify_0_failures': _GUARD.verify(0)},
              'w112_imported': {'counts': dict(W112._GUARD.counts), 'verify_0_failures': W112._GUARD.verify(0)}}
    res = {
        'stage': 'P5.15 W131 -- prefreeze diagnostics (records only)', 'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        'script': os.path.relpath(THIS, REPO), 'script_sha256': _sha(THIS),
        'definitions': __doc__, 'optimal_text': OPTIMAL,
        'modules': {'settling_criterion': _sha(os.path.join(REPO, 'settling_criterion.py')),
                    'settling_criterion_v2': _sha(os.path.join(REPO, 'settling_criterion_v2.py')),
                    'p515_s53_w112_consensus_gap': _sha(os.path.join(REPO, 'p515_s53_w112_consensus_gap.py'))},
        'task1': t1, 'task2': t2, 'task3': t3, 'guards': guards,
        'log_inventory_file': os.path.relpath(OUT_INV, REPO), 'n_logs_inventoried': len(INVENTORY),
        'wall_s': time.time() - t0,
    }
    inputs = {}
    for c in t1.values():
        inputs.update(c['inputs_sha256'])
    inputs.update(t2['inputs_sha256'])
    for c in t3['cells'].values():
        inputs.update(c['inputs_sha256'])
    res['inputs_sha256'] = inputs
    with open(OUT_INV, 'x') as handle:
        GRIO.dump(INVENTORY, handle, indent=1, sort_keys=True)
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(res, handle, indent=1)
    _log(f'guards {guards}')
    fails = guards['w131']['verify_0_failures'] + guards['w112_imported']['verify_0_failures']
    _log(f'wrote {OUT_JSON} and {OUT_INV} ({len(INVENTORY)} logs); wall {res["wall_s"]:.1f} s')
    return 1 if fails else 0


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    res = json.load(open(OUT_JSON))
    for rel, v in res['inputs_sha256'].items():
        entries[rel + ' (input)'] = v['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
