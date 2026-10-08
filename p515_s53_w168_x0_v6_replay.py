"""P5.15 Addendum 68 decision 4, Planner task W168 -- THE SETTLING DECISION OF THE x = 0 REFERENCE `ref:7aa017f0`
REPLAYED UNDER SETTLING RULE v6 ON ITS COMMITTED W101 PER-CYCLE RECORD. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: nothing
earlier is edited (settling_criterion*.py, the hooks, the W101 records, the frozen tables are read only).

THE INSTANCE. ref:7aa017f0 = the SRP1 x = 0 candidate (candidate_key 8435c718..., eval_key d110bd1a5977df1e...), the W101
continuation eval dir data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/
d110bd1a5977df1e_x0, certified there under settling rule v1 (`settling_criterion.py`) at k* = end = 181,
Q(181) = 653,873,702.1876609 (gross_operational_cost, settlement excluded) -- the value the frozen tables
(frozen_step6_tables_v1_590088fe.json, tables.cells['ref:7aa017f0'].Q) carry.

THE RULE, AS IMPLEMENTED (nothing re-implemented):
  * the v6 rule class the v6 campaign consults -- `p515_s53_w142_resettle_v6_hooks.HookedRuleV6` (a subclass of
    `settling_criterion_v6.SettlingRuleV6` whose only addition is that observe(k, q, boyd, t_sum) reads all_clean_k from
    the bound state) -- driven exactly as the campaign's recourse wrapper drives it (`p515_s53_w118_resettle_hooks.
    make_wrappers._observe`: rule.observe(c, Q_c, boyd_c, t_sum_c) once per cycle from cycle 1, the decision taken at the
    first certification or at the cap). The state bound here carries only `all_clean` and `errors` (the two attributes
    HookedRuleV6 reads). PRIMARY ANSWER.
  * cross-check: `settling_criterion_v6.replay` (the from-records entry point W142 used) on the same inputs must return
    the identical decision.
  * what the campaign wrapper adds around the rule and what it does NOT: the cap (gated cells N_old + 100; here W101's
    own cap 232 = N + 100, N = 132, ceiling 300 -- never reached, the record ends at 181) and the held regime (AA off,
    tight tail on, rho frozen) applied by the RUN for every cycle c > the run's first residual pass k0_run
    (`ResettleState.held(c)`: c > first_pass). The regime is not a rule input: k0 is the rule's own (the first cycle
    with Q not None and boyd_k; a lapse resets it), N := the first k0; N_old enters only the cap and the replay gate;
    the gap clause and the monotone branch are inside the rule. So the held-regime start is a PRECONDITION OF THE RUN
    (held from k0_run + 1), checked here against the record, not an input the rule can be given.
  * labelled VARIANT (not the code's return): the window restricted to cycles >= the first held cycle of the record
    (133): the same rule with boyd_k read as False for every cycle < 133 (so the code's own k0 becomes 133 and nothing
    before it is read). Reported because the W101 record held the regime only from 133, not from k0 + 1 = 124.

INPUTS PER CYCLE (asserted BEFORE the rule runs; a missing input stops the run with exit 1 and names it):
  Q_k        per_cycle_record.jsonl gross_operational_cost (every cycle 1..181, a float)
  boyd_k     per_cycle_record.jsonl boyd_all_pass AND local_solves_ok (W139 / W142 from-records reading); cross-checked
             against the in-run settling_continuation_cycle_record.jsonl boyd_k (what the wrapper passes) -- equal
  t_sum_k    NOT in per_cycle_record.jsonl nor in settling_continuation_cycle_record.jsonl (W101 predates the t_sum
             capture). What the record has instead: the run's pf consensus copies, per cycle, in pf_entry_stride_s39_D
             .jsonl (untracked, sha256-recorded in the W101 campaign manifest); `p515_s53_w118_resettle_checks.
             record_series` (W112's committed reader, used by W118 / W131 / W137 / W139 / W141 / W142 on this record)
             derives t_sum_k from it with production's price and weight, validated at the terminal cycle against
             production's own interface_settlement_detail_s31c.json t_tso_plus_t_dso_terminal. Recorded as DERIVED, and
             the decision's dependence on it is reported: the cycles at which the rule actually READ t_sum (a branch
             would certify) are listed, and at each the gap clause is re-evaluated with production's terminal value
             where one exists.
  all_clean_k  NOT captured in-run by W101 (the exit capture is W132+). What the record has instead: the network
             IPOPT solve records (network_ipopt_solve_records.jsonl, committed) and the ESSO IPOPT logs (untracked,
             sha256-inventoried by W131); `p515_s53_w139_v5_from_records.block_finals` + `classify_cycle` (=
             `settling_criterion_v5.classify_block_exit`, the classifier v6 uses) recompute it for all 51 blocks of
             every cycle; every log read is checked against W131's committed inventory; the result is cross-checked
             against the committed W142 floor replay's non_clean_cycles for this record.
  holds      settling_continuation_cycle_record.jsonl holds / aa / tail_apply / tail_next / rho (not rule inputs;
             reported per cycle, with the natural values, to locate the held and the natural regime starts).

OUTPUTS (write-once; data/SRP1/Results/P515S53/w168_x0_v6_replay/, created by the launch `mkdir`, which fails if it
exists; the script refuses unless the directory holds only launch.log): manifest_inputs_sha256.json (written BEFORE the
rule runs), w168_x0_v6_replay_decision.json, w168_x0_v6_replay_summary.md, manifest_sha256.json; after the W100 typing
test: manifest_post_run_sha256.json (--post-run).

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 solves and
0 executions at the end, together with every guard the imports arm; pickle.load / pickle.loads blocked for the whole run
(re-blocked after the imports) and their counters verified at 0.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w168_x0_v6_replay && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w168_x0_v6_replay.py \\
        > data/SRP1/Results/P515S53/w168_x0_v6_replay/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w168_x0_v6_replay/w168_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w168_x0_v6_replay/w168_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w168_x0_v6_replay.py --post-run
  Trial runs: --out-dir <scratch dir> (recorded as a trial).
Exit: 0 = written, every integrity check holds, guards 0 (the v6 OUTCOME, whatever it is, never changes the exit code);
3 = written, an integrity check failed (listed); 1 = precondition / capture-path / guard fault (nothing written beyond
the input manifest).
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W168 x0 v6 replay (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W168: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W168: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402 -- HookedRuleV6, P_MAX (stdlib at import)
import p515_s53_w139_v5_from_records as W139  # noqa: E402 -- block_finals / classify_cycle (arms its guards)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

K118 = W139.K137.K118          # p515_s53_w118_resettle_checks (record_series; its guards are in W139.GUARDS)
W131 = W139.W131


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w168_x0_v6_replay', GUARD),) + tuple(W139.GUARDS))

SCRIPT_REL = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w168_x0_v6_replay')
OUT_LOG = 'launch.log'
OUT_MAN_IN = 'manifest_inputs_sha256.json'
OUT_JSON = 'w168_x0_v6_replay_decision.json'
OUT_MD = 'w168_x0_v6_replay_summary.md'
OUT_MAN = 'manifest_sha256.json'
OUT_POST = 'manifest_post_run_sha256.json'
OUT_TYPING_JSON = 'w168_bool_typing_test.json'
OUT_TYPING_LOG = 'w168_bool_typing_test.log'

REF_ID = 'ref:7aa017f0'
CAMPAIGN_ROOT = os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0')
EVAL_DIR = os.path.join(CAMPAIGN_ROOT, 'evals', 'd110bd1a5977df1e_x0')
CAMPAIGN_MAN = os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json')
PCR = os.path.join(EVAL_DIR, 'per_cycle_record.jsonl')
CYC = os.path.join(EVAL_DIR, 'settling_continuation_cycle_record.jsonl')
DEC = os.path.join(EVAL_DIR, 'settling_decision.json')
EVREC = os.path.join(EVAL_DIR, 'evaluation_record.json')
DETAIL = os.path.join(EVAL_DIR, 'interface_settlement_detail_s31c.json')
STRIDE = os.path.join(EVAL_DIR, 'pf_entry_stride_s39_D.jsonl')
NETREC = os.path.join(EVAL_DIR, 'network_ipopt_solve_records.jsonl')
LEAK = os.path.join(EVAL_DIR, 'leak_classification_s39_D.jsonl')
ESSO_EVENTS = os.path.join(EVAL_DIR, 'esso_recovery_events_s39_D.jsonl')
DECLARED_SHA = {PCR: 'bf38d5a54938e597c6508c6a64c9f8ee6a84aac14f897a556b939fc409d840dc',
                DEC: '6c559f2fc39a79dcd54e9339541fb09b599f87689c26a489a8e7e77b17e80be6'}
FZ_REL = os.path.join(S53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA8 = '590088fe'
W131_INV = os.path.join(S53, 'w131_prefreeze', 'w131_log_inventory.json')
W131_MAN = os.path.join(S53, 'w131_prefreeze', 'manifest_sha256.json')
W142_REPLAY = os.path.join(S53, 'w142_resettle_v6', 'floor_replay', 'w142_floor_replay.json')
W142_REPLAY_MAN = os.path.join(S53, 'w142_resettle_v6', 'floor_replay', 'manifest_sha256.json')
W142_V6REC = os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json')
W142_V6REC_MAN = os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'manifest_sha256.json')
EXPECTED_N_CYCLES = 181
HELD_KEYS = ('aa', 'tail_apply', 'tail_next', 'rho')
TAU = SC6.TAU
BAND_FRACTION = 0.93                       # Addendum 68: "differs by less than the band (0.93 tau)"
PREDICTION = {'source': 'PLANNER_BRIEF_2026-09-13.md Addendum 68 decision 4 (expert); TASKS.md Addendum 68 order',
              'text': ('certifies under v6 at a cycle in [174, 195], window range <= tau, value within 0.93 tau of the '
                       'tabulated one'),
              'k_range': [174, 195], 'range_le_tau': True, 'abs_dQ_le': BAND_FRACTION}
CODE_FILES = (SCRIPT_REL, 'settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v5.py',
              'settling_criterion_v6.py', 'p515_s53_w142_resettle_v6_hooks.py', 'p515_s53_w139_resettle_v5_hooks.py',
              'p515_s53_w118_resettle_hooks.py', 'p515_s53_w118_resettle_checks.py', 'p515_s53_w112_consensus_gap.py',
              'p515_s53_w139_v5_from_records.py', 'p515_s53_w131_prefreeze_diagnostics.py', 'gate_result_io.py',
              'p513_solve_profile_guard.py')


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _jl(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        return json.load(handle)


def _jsonl(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _tracked(rel):
    return subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True).returncode == 0


def _committed_clean(rel):
    return _tracked(rel) and _git('status', '--porcelain', '--', rel) == ''


def _last_commit(rel):
    return _git('log', '-1', '--format=%h', '--', rel) or None


# ======================================================================================================================
#  the inputs
# ======================================================================================================================
def read_series():
    """Q, boyd (from-records reading), t_sum (stride-derived) and the terminal validation, through the committed
    reader K118.record_series (W112's sha-verified stride reader). Also the in-run cycle lines."""
    with contextlib.redirect_stdout(io.StringIO()):
        q, b, t, val, rows = K118.record_series(K118.X0_CELL)
    if os.path.normpath(K118.W112.CELLS[K118.X0_CELL]) != os.path.normpath(EVAL_DIR):
        raise RuntimeError(f'K118.X0_CELL resolves to {K118.W112.CELLS[K118.X0_CELL]}, not {EVAL_DIR}')
    lines = {x['cycle']: x for x in _jsonl(CYC)}
    return q, b, t, val, rows, lines


def clean_series(n):
    """{cycle: all_clean_k} for cycles 1..n from W139's readers (51 blocks per cycle; v5's classifier), the meta, the
    non-clean cycles with their blocks, and the per-cycle class counts."""
    finals, meta = W139.block_finals(EVAL_DIR, n)
    per_cycle = {k: W139.classify_cycle(finals[k]) for k in range(1, n + 1)}
    clean = {k: all(e['clean'] for e in per_cycle[k].values()) for k in per_cycle}
    non_clean = {k: {b: {f: e.get(f) for f in ('class', 'attempt', 'exit', 'reason', 'max_ratio')}
                     for b, e in per_cycle[k].items() if not e['clean']} for k in per_cycle if not clean[k]}
    classes = {}
    for k in per_cycle:
        for e in per_cycle[k].values():
            key = f"{e['family']}|{e['class']}|{e['attempt']}"
            classes[key] = classes.get(key, 0) + 1
    n_blocks = {k: len(per_cycle[k]) for k in per_cycle}
    return clean, non_clean, meta, classes, n_blocks


def capture_path_checklist(q, b, t, val, rows, lines, clean, clean_meta, n_blocks):
    """Rule eleven: every input v6 needs per cycle, and where the record carries it. Returns (checks, inventory)."""
    n = EXPECTED_N_CYCLES
    cyc = list(range(1, n + 1))
    pcr_keys = set().union(*(set(r) for r in rows.values()))
    cyc_keys = set().union(*(set(x) for x in lines.values()))
    checks = {
        'Q_k:per_cycle_record_holds_exactly_cycles_1_181': sorted(rows) == cyc,
        'Q_k:gross_operational_cost_a_float_every_cycle': all(isinstance(rows[k].get('gross_operational_cost'), float)
                                                              for k in cyc),
        'Q_k:reader_equals_record': all(q[k] == rows[k]['gross_operational_cost'] for k in cyc),
        'boyd_k:boyd_all_pass_and_local_solves_ok_bool_every_cycle': all(
            isinstance(rows[k].get('boyd_all_pass'), bool) and isinstance(rows[k].get('local_solves_ok'), bool)
            for k in cyc),
        'boyd_k:reader_equals_record': all(b[k] == (rows[k]['boyd_all_pass'] and rows[k]['local_solves_ok'])
                                           for k in cyc),
        'boyd_k:in_run_cycle_record_holds_cycles_1_181': sorted(lines) == cyc,
        'boyd_k:equals_in_run_boyd_k_every_cycle': all(lines[k].get('boyd_k') is b[k] for k in cyc),
        'boyd_k:in_run_gross_equals_record_every_cycle': all(lines[k].get('gross') == q[k] for k in cyc),
        't_sum_k:in_per_cycle_record': 't_sum' in pcr_keys,
        't_sum_k:in_in_run_cycle_record': 't_sum' in cyc_keys,
        't_sum_k:derived_from_pf_stride_every_cycle': all(isinstance(t.get(k), float) for k in cyc),
        't_sum_k:terminal_validation_vs_production_detail_pass': bool(val.get('pass')),
        'all_clean_k:in_per_cycle_record': any(k_ in pcr_keys for k_ in ('all_clean_k', 'all_optimal_k')),
        'all_clean_k:in_in_run_cycle_record': any(k_ in cyc_keys for k_ in ('all_clean_k', 'all_optimal_k')),
        'all_clean_k:recomputed_51_blocks_every_cycle': (bool(clean_meta.get('coverage_51_every_cycle'))
                                                         and all(n_blocks.get(k) == 51 for k in cyc)),
        'all_clean_k:esso_logs_none_missing': not clean_meta.get('esso_missing'),
        'all_clean_k:network_log_byte_crosscheck_0_disagreements': clean_meta.get(
            'network_log_byte_crosscheck_disagreements') == 0,
        'all_clean_k:bool_every_cycle': all(isinstance(clean.get(k), bool) for k in cyc),
        'holds:in_run_holds_every_cycle': all(isinstance(lines[k].get('holds'), dict)
                                              and set(HELD_KEYS) <= set(lines[k]['holds']) for k in cyc),
    }
    # what is REQUIRED for the rule to run (an input with no capture path at all stops the run); the two "in record"
    # flags for t_sum and all_clean are FACTS (False = not recorded in-run), satisfied by the derived paths
    required = [c for c in checks if not c.endswith((':in_per_cycle_record', ':in_in_run_cycle_record'))]
    inventory = {
        'Q_k': {'source': f'{PCR} gross_operational_cost', 'status': 'recorded'},
        'boyd_k': {'source': (f'{PCR} boyd_all_pass AND local_solves_ok (W139 / W142 from-records reading); equals the '
                              f'in-run {os.path.basename(CYC)} boyd_k on every cycle (what the wrapper passes)'),
                   'status': 'recorded'},
        't_sum_k': {'source': (f'DERIVED: {STRIDE} (untracked; sha256 in {CAMPAIGN_MAN}) through K118.record_series / '
                               'W112._stream_stride with the terminal detail prices and weights; validated at the '
                               'terminal cycle against production\'s t_tso_plus_t_dso_terminal'),
                    'status': 'not recorded in-run (W101 predates the t_sum capture); derived from the run\'s own '
                              'consensus copies', 'terminal_validation': val},
        'all_clean_k': {'source': (f'DERIVED: {NETREC} (committed) + the ESSO IPOPT logs ({clean_meta.get("logs_dir")}, '
                                   'untracked, inventoried by W131) through W139.block_finals + classify_cycle '
                                   '(settling_criterion_v5.classify_block_exit = v6\'s classifier)'),
                        'status': 'not captured in-run (W101 predates the exit capture); recomputed from the solve '
                                  'records and logs', 'meta': clean_meta},
        'holds': {'source': f'{CYC} holds / aa / tail_apply / tail_next / rho', 'status': 'recorded; not a rule input'},
    }
    return checks, required, inventory


def regime_table(lines, rows):
    """Per cycle: the holds as applied by W101 and the natural values (AA action, tight tail passed, rho frozen and
    unchanged). Returns (table, summary)."""
    table = {}
    for k in sorted(lines):
        x = lines[k]
        h = x.get('holds') or {}
        aa = x.get('aa') or {}
        ta = x.get('tail_apply') or {}
        tn = x.get('tail_next') or {}
        rho = x.get('rho') or {}
        r = rows[k]
        table[k] = {'phase': x.get('phase'), 'held': {g: h.get(g) for g in HELD_KEYS},
                    'all_held': all(h.get(g) is True for g in HELD_KEYS),
                    'aa_action': aa.get('action'), 'aa_off_natural': str(aa.get('action') or '').startswith('off'),
                    'tail_active_passed': ta.get('active_passed'),
                    'tail_next_value': tn.get('value', tn.get('returned')),
                    'rho_frozen_all': all((rho.get('frozen') or {}).get(g) is True for g in ('v', 'pf', 'ess')),
                    'rho_after': {g: r.get(f'rho_{g}_after') for g in ('v', 'pf', 'ess')},
                    'rho_freeze_active': r.get('rho_freeze_active'), 'replay_equal': x.get('replay_equal')}
    ks = sorted(table)

    def first_from(pred):
        """the first cycle c such that pred holds on every cycle >= c (None if not at the last cycle)."""
        c = None
        for k in reversed(ks):
            if pred(k):
                c = k
            else:
                break
        return c

    last = ks[-1]
    rho_last = table[last]['rho_after']
    summ = {
        'first_cycle_all_four_held_through_end': first_from(lambda k: table[k]['all_held']),
        'held_cycles_count': sum(1 for k in ks if table[k]['all_held']),
        'phase_by_cycle_ranges': _ranges([(k, table[k]['phase']) for k in ks]),
        'natural_aa_off_from': first_from(lambda k: table[k]['aa_off_natural']),
        'natural_tight_tail_passed_from': first_from(lambda k: table[k]['tail_active_passed'] is True),
        'natural_rho_frozen_all_channels_and_equal_to_end_value_from': first_from(
            lambda k: table[k]['rho_frozen_all'] and table[k]['rho_after'] == rho_last),
        'replay_equal_true_cycles': _ranges([(k, table[k]['replay_equal']) for k in ks]),
    }
    nat = [summ['natural_aa_off_from'], summ['natural_tight_tail_passed_from'],
           summ['natural_rho_frozen_all_channels_and_equal_to_end_value_from']]
    summ['natural_regime_from'] = max(nat) if all(v is not None for v in nat) else None
    return table, summ


def _ranges(pairs):
    out = []
    for k, v in pairs:
        if out and out[-1]['value'] == v and out[-1]['to'] == k - 1:
            out[-1]['to'] = k
        else:
            out.append({'from': k, 'to': k, 'value': v})
    return out


# ======================================================================================================================
#  the rule
# ======================================================================================================================
def run_hooked(q, b, t, clean, cap, cap_ceiling, last):
    """The v6 campaign's rule class driven as the recourse wrapper drives it: one observe(c, Q, boyd, t_sum) per cycle
    from 1, all_clean_k read from the bound state; stops at the decision, the cap, or the end of the record."""
    state = SimpleNamespace(all_clean=dict(clean), errors=[])
    rule = V6.HookedRuleV6(V6.P_MAX, cap=cap, cap_ceiling=cap_ceiling).bind(state)
    recs = []
    k = 0
    while True:
        k += 1
        if k > last or k > rule.effective_cap():
            break
        recs.append(rule.observe(k, q.get(k), bool(b.get(k, False)), t.get(k)))
        if rule.decision is not None:
            break
    return recs, rule.decision, rule, state


def compact(rec):
    """The per-cycle v6 state, compact (the full record is kept for the eligible cycles)."""
    a = rec.get('certA_parts') or {}
    bp = rec.get('certB_parts') or {}
    return {'k': rec['k'], 'Q': rec['Q'], 'boyd_k': rec['boyd_k'], 't_sum': rec['t_sum'], 'all_clean_k':
            rec.get('all_clean_k'), 'lapse': rec['lapse'], 'k0': rec['k0'], 'N': rec['N'], 'eligible': rec['eligible'],
            'dQ': rec['dQ'], 's_k': rec['s_k'], 'sign_change': rec['sign_change'],
            'turning_point': rec['turning_point'], 'n_T': rec['len_T'], 'T': rec['T'], 'A': rec['A'],
            'P_hat': rec['P_hat'], 'n_w': rec['W'], 'window': rec['window'], 'range': rec['range'],
            'range_over_tau': rec['range_over_tau'],
            'osc_at_least_3_tp': a.get('at_least_3_turning_points'),
            'osc_swings_non_increasing_floored': a.get('swings_non_increasing_floored'),
            'osc_window_inside_run': a.get('window_inside_run'), 'osc_range_le_tau': a.get('range_le_tau'),
            'certA': rec['certA'], 'mono_window': bp.get('window'), 'mono_no_sign_change': bp.get(
                'no_sign_change_in_window'), 'mono_range_le_tau': bp.get('range_le_tau'),
            'mono_last_step_times_L_le_tau': bp.get('last_step_times_L_le_tau'), 'certB': rec['certB'],
            'branch_would_certify': rec['branch_would_certify'], 'gap_abs': rec['gap_abs'], 'gap_ok': rec['gap_ok'],
            'vetoed': rec.get('vetoed'), 'decision': rec['decision'], 'reasons': rec['reasons'],
            'turning_point_floor_rejection': rec.get('turning_point_floor_rejection')}


def _dec_core(d):
    keys = ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W', 'P_hat', 'window', 'band', 'band_width', 'range',
            'range_over_tau', 'T', 'A', 'Q_k_star', 't_sum_k_star', 'n_vetoes', 'non_clean_cycles', 'gap_refusals',
            'lapse_events', 'turning_point_floor_rejections', 'reasons')
    return {k: (d or {}).get(k) for k in keys}


# ======================================================================================================================
def post_run(out_dir):
    rels = [os.path.join(out_dir, f) for f in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W168 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    if os.path.exists(os.path.join(REPO, out_dir, OUT_POST)):
        _log(f'[W168 post-run] refusing to overwrite {os.path.join(out_dir, OUT_POST)}')
        sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_POST), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[W168 post-run] wrote {os.path.join(out_dir, OUT_POST)}: ' + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': dict(PICKLE_COUNTS), 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and PICKLE_COUNTS == {'load': 0, 'loads': 0}}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None, help='trial runs only')
    args = ap.parse_args()
    trial = args.out_dir is not None
    out_dir = args.out_dir if trial else OUT_DIR
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W168'
    # ---- preconditions ---------------------------------------------------------------------------------------------
    pre = []
    od = os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch creates it with mkdir)')
    else:
        present = sorted(os.listdir(od))
        if [p for p in present if p != OUT_LOG]:
            pre.append(f'{out_dir} already holds {present} (only {OUT_LOG} may exist)')
    script_clean = _committed_clean(SCRIPT_REL)
    if not trial and not script_clean:
        pre.append(f'{SCRIPT_REL} not committed clean (production run)')
    tracked_inputs = {'per_cycle_record': PCR, 'cycle_record': CYC, 'settling_decision': DEC, 'evaluation_record': EVREC,
                      'campaign_manifest': CAMPAIGN_MAN, 'settlement_detail': DETAIL, 'network_solve_records': NETREC,
                      'leak_classification': LEAK, 'esso_recovery_events': ESSO_EVENTS, 'frozen_tables': FZ_REL,
                      'w131_log_inventory': W131_INV, 'w131_manifest': W131_MAN, 'w142_floor_replay': W142_REPLAY,
                      'w142_floor_replay_manifest': W142_REPLAY_MAN, 'w142_v6_from_records': W142_V6REC,
                      'w142_v6_from_records_manifest': W142_V6REC_MAN}
    for f in CODE_FILES:
        tracked_inputs[f'code:{f}'] = f
    inputs = {}
    for key, rel in tracked_inputs.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        cc = _committed_clean(rel)
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': cc, 'last_commit': _last_commit(rel)}
        if not cc and not (key == f'code:{SCRIPT_REL}' and trial):
            pre.append(f'{rel} not committed clean')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    cman = _jl(CAMPAIGN_MAN)
    for rel, want in DECLARED_SHA.items():
        if inputs_sha(inputs, rel) != want:
            pre.append(f'{rel} sha256 {inputs_sha(inputs, rel)} != declared {want}')
    for rel in (PCR, CYC, DEC, EVREC, DETAIL, NETREC, LEAK, ESSO_EVENTS):
        if cman.get(rel) != inputs_sha(inputs, rel):
            pre.append(f'{rel} != its W101 campaign manifest entry')
    stride_sha = _sha(STRIDE) if os.path.exists(os.path.join(REPO, STRIDE)) else None
    inputs['pf_stride'] = {'path': STRIDE, 'sha256': stride_sha, 'committed_clean': False, 'tracked': _tracked(STRIDE),
                           'in_w101_campaign_manifest': cman.get(STRIDE) == stride_sha}
    if stride_sha is None or cman.get(STRIDE) != stride_sha:
        pre.append(f'{STRIDE} missing or != its W101 campaign manifest entry')
    if not os.path.basename(FZ_REL).split('_')[-1].startswith(FZ_SHA8) or not inputs['frozen_tables']['sha256'].startswith(
            FZ_SHA8):
        pre.append(f'frozen tables sha256 {inputs["frozen_tables"]["sha256"][:8]} != {FZ_SHA8}')
    for rel, man_rel in ((W131_INV, W131_MAN), (W142_REPLAY, W142_REPLAY_MAN), (W142_V6REC, W142_V6REC_MAN)):
        if _jl(man_rel).get(rel) != inputs_sha(inputs, rel):
            pre.append(f'{rel} != its manifest {man_rel}')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    fz = _jl(FZ_REL)
    cell = fz['tables']['cells'][REF_ID]
    evrec = _jl(EVREC)
    v1 = _jl(DEC)
    instance = {'reference_id': REF_ID, 'eval_key': evrec.get('eval_key'), 'candidate_key': evrec.get('candidate_key'),
                'candidate_label': evrec.get('candidate_label'), 'candidate_canonical': evrec.get('candidate_canonical'),
                'eval_dir': EVAL_DIR, 'frozen_tables_cell': {k: cell.get(k) for k in (
                    'eval_key', 'candidate_key', 'Q', 'Q_cc', 't', 'k_star', 'end_cycle', 'band_width',
                    'range_over_tau', 'status', 'name')}}
    if not (cell['eval_key'] == instance['eval_key'] and cell['candidate_key'] == instance['candidate_key']):
        _log(f'[{tag} PRECONDITION FAILED] the frozen tables cell {REF_ID} is not the W101 eval dir\'s instance')
        sys.exit(1)
    if fz['constants']['TAU'] != TAU:
        _log(f'[{tag} PRECONDITION FAILED] frozen TAU {fz["constants"]["TAU"]} != settling_criterion_v6.TAU {TAU}')
        sys.exit(1)
    # ---- the inputs, then the capture-path assertion (BEFORE the rule runs) ----------------------------------------
    q, b, t, val, rows, lines = read_series()
    clean, non_clean, clean_meta, classes, n_blocks = clean_series(EXPECTED_N_CYCLES)
    checks, required, cap_inventory = capture_path_checklist(q, b, t, val, rows, lines, clean, clean_meta, n_blocks)
    # every IPOPT log read for all_clean_k against W131's committed inventory
    w131_inv = _jl(W131_INV)
    logs_read = dict(W131.INVENTORY)
    log_x = {'n_logs_read': len(logs_read), 'not_in_w131_inventory': sorted(p for p in logs_read if p not in w131_inv),
             'sha_differs_from_w131_inventory': sorted(p for p, v in logs_read.items()
                                                       if p in w131_inv and w131_inv[p].get('sha256') != v['sha256'])}
    checks['all_clean_k:every_log_read_in_w131_inventory_same_sha256'] = (not log_x['not_in_w131_inventory']
                                                                          and not log_x['sha_differs_from_w131_inventory'])
    required.append('all_clean_k:every_log_read_in_w131_inventory_same_sha256')
    failing = [c for c in required if checks[c] is not True]
    for p, v in sorted(logs_read.items()):
        inputs[f'log:{p}'] = {'path': p, 'sha256': v['sha256'], 'committed_clean': False, 'kind': v.get('kind')}
    man_in = {v['path']: v['sha256'] for v in inputs.values()}
    with open(os.path.join(od, OUT_MAN_IN), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man_in, indent=1, sort_keys=True) + '\n')
    _log(f'[{tag}] input manifest written ({len(man_in)} files: {len(logs_read)} IPOPT logs); script committed clean '
         f'{script_clean}; trial {trial}')
    _log(f'[{tag}] capture path: ' + '; '.join(f'{c}={checks[c]}' for c in checks))
    if failing:
        _log(f'[{tag} CAPTURE PATH FAILED -- the rule is NOT run] missing / failing: {failing}')
        sys.exit(1)
    _log(f'[{tag}] capture path holds (t_sum_k and all_clean_k DERIVED, not recorded in-run: see the inventory)')
    # ---- the cap the campaign wrapper would apply, and the regime -----------------------------------------------
    cap, cap_ceiling = int(v1['cap']), max(int(v1['cap']), SC6.SC2.CAP_CEILING)
    n_v1 = v1.get('N')
    cap_basis = {'cap': cap, 'cap_ceiling': cap_ceiling, 'source': f'{DEC} cap (W101: N + 100, N = {n_v1})',
                 'equals_gated_v6_rule_N_old_plus_100': cap == n_v1 + V6.CAP_AFTER_N_OLD,
                 'binds': cap <= EXPECTED_N_CYCLES}
    reg_table, reg = regime_table(lines, rows)
    # ---- the rule: primary (the campaign's class), cross-check (replay), labelled variant -------------------------
    recs, dec, rule, state = run_hooked(q, b, t, clean, cap, cap_ceiling, EXPECTED_N_CYCLES)
    _o2, dec2, rule2 = SC6.replay(q, b, t, clean, V6.P_MAX, cap=cap, cap_ceiling=cap_ceiling, last=EXPECTED_N_CYCLES)
    same = (json.dumps(_dec_core(dec), sort_keys=True, default=str)
            == json.dumps(_dec_core(dec2), sort_keys=True, default=str))
    k0_rule = (dec or {}).get('k0') if dec else rule.k0
    held_from = reg['first_cycle_all_four_held_through_end']
    precondition = {'campaign_rule': ('p515_s53_w118_resettle_hooks.ResettleState.held(c) = first_pass is not None and '
                                      'c > first_pass: the regime is held from k0_run + 1'),
                    'k0_rule': k0_rule, 'required_held_from': (k0_rule + 1) if k0_rule else None,
                    'held_from_in_record': held_from,
                    'met': (held_from is not None and k0_rule is not None and held_from <= k0_rule + 1),
                    'natural_regime_from': reg['natural_regime_from'],
                    'natural_regime_from_le_k0_plus_1': (reg['natural_regime_from'] is not None and k0_rule is not None
                                                         and reg['natural_regime_from'] <= k0_rule + 1)}
    b_var = {k: (b[k] if k >= held_from else False) for k in b} if held_from else None
    var = None
    if b_var is not None:
        _ov, dv, rv = SC6.replay(q, b_var, t, clean, V6.P_MAX, cap=cap, cap_ceiling=cap_ceiling, last=EXPECTED_N_CYCLES)
        var = {'label': (f'VARIANT (not the code\'s return): window restricted to cycles >= {held_from} (the first held '
                         f'cycle) -- boyd_k read as False for every cycle < {held_from}, so the rule\'s k0 = '
                         f'{held_from}; everything else identical'),
               'decision_core': _dec_core(dv), 'decision': dv, 'k0_at_end': rv.k0,
               'per_cycle_eligible': [compact(r) for r in _ov if r['eligible']]}
    # ---- the report quantities -------------------------------------------------------------------------------------
    q181 = cell['Q']
    elig = [r for r in recs if r['eligible']]
    first_eval = elig[0]['k'] if elig else None
    read_t = [r['k'] for r in recs if r['branch_would_certify'] is not None]
    detail = _jl(DETAIL)
    t_prod_terminal = detail['t_tso_plus_t_dso_terminal']
    rep = {'certified': bool(dec and dec.get('status') == 'certified'), 'decided_within_record': dec is not None}
    if dec and dec.get('status') == 'certified':
        ks = dec['k_star']
        step_ks = q[ks] - q[ks - 1]
        step_181 = q[EXPECTED_N_CYCLES] - q[EXPECTED_N_CYCLES - 1]
        d_q = dec['Q_k_star'] - q181
        bar = abs(step_ks) + abs(step_181)
        a_parts = dec['certA_parts'] if dec['branch'] == 'oscillatory' else None
        gt = SC6.growth_test(dec['A'], SC6.GROWTH_TEST_FLOOR)
        rep.update({
            'k_star_v6': ks, 'branch': dec['branch'], 'window': dec['window'], 'n_w': dec['W'], 'P_hat': dec['P_hat'],
            'window_start': dec['window'][0], 'window_end': dec['window'][1], 'range': dec['range'],
            'range_over_tau': dec['range_over_tau'], 'band': dec['band'], 'band_width': dec['band_width'],
            'turning_points_all': dec['T'], 'turning_points_used_for_P_hat': dec['T'][-3:], 'swings_A': dec['A'],
            'swing_check': {'rule': 'v6 (A): swings >= TAU/10, in order, non-increasing (chain)',
                            'ok': gt[0], **gt[1], 'certA_parts_value': (a_parts or {}).get('swings_non_increasing_floored')},
            'turning_point_floor_rejections': dec.get('turning_point_floor_rejections'),
            'gap_clause': {'t_sum_k_star_stride': dec['t_sum_k_star'], 'abs_t_sum': abs(dec['t_sum_k_star']),
                           'gap_bound_tau_over_2': SC6.GAP_BOUND,
                           'abs_t_sum_over_gap_bound': abs(dec['t_sum_k_star']) / SC6.GAP_BOUND,
                           'production_terminal_t_tso_plus_t_dso': t_prod_terminal if ks == EXPECTED_N_CYCLES else None,
                           'passes_with_production_terminal_value': (abs(t_prod_terminal) <= SC6.GAP_BOUND)
                           if ks == EXPECTED_N_CYCLES else None,
                           'cycles_at_which_the_rule_read_t_sum': read_t,
                           'gap_refusals': dec.get('gap_refusals')},
            'clean_veto': {'vetoes': dec.get('vetoes'), 'n_vetoes': dec.get('n_vetoes'),
                           'non_clean_cycles': dec.get('non_clean_cycles'),
                           'window_all_clean': dec.get('window_all_clean')},
            'out_of_window_reads': dec.get('out_of_window_reads'),
            'Q_k_star_v6': dec['Q_k_star'], 'Q_181_tabulated': q181, 'dQ_v6_minus_tabulated': d_q,
            'resolution': {'terminal_step_at_k_star_v6': step_ks, 'terminal_step_at_181': step_181,
                           'error_bar_sum_abs_terminal_steps': bar,
                           'determinate': abs(d_q) > bar if d_q != 0 else False,
                           'note': ('the two cycles coincide: the difference is exactly 0, no stopping-slack question '
                                    'arises') if ks == EXPECTED_N_CYCLES else
                           'a difference smaller than the bar is indeterminate (CLAUDE.md)'},
            'abs_dQ_over_tau': abs(d_q) / TAU, 'band_0_93_tau': BAND_FRACTION * TAU,
            'abs_dQ_le_0_93_tau': abs(d_q) <= BAND_FRACTION * TAU,
            'terminal_step_to_threshold': {'terminal_step_abs_over_EPS0': abs(step_ks) / SC6.EPS0,
                                           'range_over_tau': dec['range_over_tau']},
            'same_as_v1_certificate': {f: dec.get(f) == v1.get(f) for f in ('k_star', 'window', 'W', 'P_hat', 'T',
                                                                             'band', 'branch', 'Q_k_star', 'k0')},
        })
    else:
        last_rec = recs[-1]
        rep.update({'state_at_181': compact(last_rec), 'rule_k0_at_181': rule.k0,
                    'what_is_missing': last_rec['reasons']})
    pred = None
    if rep.get('k_star_v6') is not None:
        pk_in = PREDICTION['k_range'][0] <= rep['k_star_v6'] <= PREDICTION['k_range'][1]
        pred = {**PREDICTION, 'outcome': {
            'k_star_in_174_195': pk_in, 'k_star_v6': rep['k_star_v6'],
            'range_le_tau': rep['range_over_tau'] <= 1.0, 'range_over_tau': rep['range_over_tau'],
            'abs_dQ_le_0_93_tau': rep['abs_dQ_le_0_93_tau'], 'dQ': rep['dQ_v6_minus_tabulated']},
            'held': bool(pk_in and rep['range_over_tau'] <= 1.0 and rep['abs_dQ_le_0_93_tau'])}
    w142 = _jl(W142_V6REC)['item1_v6_on_every_record'].get('x0', {}).get('v6', {})
    w142_x = {'source': W142_V6REC, 'w142_x0_v6': {k: w142.get(k) for k in ('status', 'k_star', 'window', 'W',
                                                                             'range_over_tau', 'T')},
              'equal': (w142.get('status') == (dec or {}).get('status') and w142.get('k_star') == (dec or {}).get(
                  'k_star') and w142.get('window') == (dec or {}).get('window'))}
    w142_nc = _jl(W142_REPLAY)['reports']['x0']['non_clean_cycles']
    clean_x = {'recomputed_non_clean_cycles': sorted(non_clean), 'w142_floor_replay_non_clean_cycles': w142_nc,
               'equal': sorted(non_clean) == w142_nc}
    integrity = {'hooked_equals_replay': same, 'no_state_errors': not state.errors,
                 'all_clean_equals_w142_floor_replay': clean_x['equal'], 'w142_v6_from_records_equal': w142_x['equal'],
                 'frozen_cell_k_star_181': cell['k_star'] == 181 and cell['end_cycle'] == 181,
                 'tabulated_Q_equals_record_Q181': q181 == q[EXPECTED_N_CYCLES]}
    # ---- outputs -------------------------------------------------------------------------------------------------
    guards, pk, gok = guards_state()
    doc = {'schema': 'p515_s53_w168_x0_v6_replay_v1',
           'stage': 'P5.15 Addendum 68 decision 4, W168 -- v6 replay of the settling decision of ref:7aa017f0',
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'last_commit': _last_commit(SCRIPT_REL)},
           'trial': trial, 'git_head': _git('rev-parse', 'HEAD'), 'utc': _utc(), 'interpreter': sys.executable,
           'definition': __doc__,
           'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED (frozen tables convention)',
           'instance': instance,
           'record_hashes': {PCR: inputs_sha(inputs, PCR), DEC: inputs_sha(inputs, DEC), CYC: inputs_sha(inputs, CYC),
                             STRIDE: stride_sha, NETREC: inputs_sha(inputs, NETREC)},
           'constants': {'TAU': TAU, 'EPS0': SC6.EPS0, 'GAP_BOUND': SC6.GAP_BOUND, 'SWING_FLOOR': SC6.SWING_FLOOR,
                         'K_EXCL': SC6.K_EXCL, 'W_MIN': SC6.W_MIN, 'W_FACTOR': SC6.W_FACTOR, 'P_MAX': V6.P_MAX,
                         'L_MONO': 2 * V6.P_MAX, 'version': SC6.VERSION, 'reading': SC6.READING},
           'rule_path': {'primary': 'p515_s53_w142_resettle_v6_hooks.HookedRuleV6 (bound state: all_clean, errors)',
                         'cross_check': 'settling_criterion_v6.replay', 'cap': cap_basis,
                         'first_cycle_observed': 1, 'first_cycle_evaluated': first_eval},
           'capture_path': {'checks': checks, 'required': required, 'inventory': cap_inventory,
                            'log_crosscheck_w131': log_x},
           'all_clean': {'non_clean_cycles': non_clean, 'class_counts': classes, 'crosscheck': clean_x},
           'regime': {'summary': reg, 'precondition_holds_from_k0_plus_1': precondition, 'per_cycle': reg_table},
           'decision_primary': dec, 'decision_core_primary': _dec_core(dec), 'decision_core_replay': _dec_core(dec2),
           'report': rep, 'variant_restricted_to_held_cycles': var, 'prediction': pred,
           'v1_certificate': v1, 'w142_crosscheck': w142_x, 'integrity': integrity,
           'per_cycle_v6_state': [compact(r) for r in recs],
           'per_cycle_v6_full_records_eligible_cycles': [r for r in recs if r['eligible']],
           'guards': guards, 'pickle_guard': pk, 'guards_ok': gok, 'solves': 0 if gok else None,
           'wall_s': time.time() - t0}
    written = []
    jp = os.path.join(out_dir, OUT_JSON)
    with open(os.path.join(REPO, jp), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(doc, indent=1, sort_keys=True) + '\n')
    written.append(jp)
    mp = os.path.join(out_dir, OUT_MD)
    with open(os.path.join(REPO, mp), 'x', encoding='utf-8') as h:
        h.write(summary_md(doc))
    written.append(mp)
    man = {rel: _sha(rel) for rel in sorted(written + [os.path.join(out_dir, OUT_MAN_IN)])}
    with open(os.path.join(od, OUT_MAN), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[{tag}] v6 (HookedRuleV6, the campaign class): {_dec_core(dec)["status"]} k* {_dec_core(dec)["k_star"]} '
         f'window {_dec_core(dec)["window"]} n_w {_dec_core(dec)["W"]} range/tau {_dec_core(dec)["range_over_tau"]} '
         f'T {_dec_core(dec)["T"]} t_sum {_dec_core(dec)["t_sum_k_star"]} vetoes {_dec_core(dec)["n_vetoes"]}')
    if rep.get('k_star_v6') is not None:
        _log(f'[{tag}] Q(k*) {rep["Q_k_star_v6"]!r} - Q(181) tabulated {q181!r} = {rep["dQ_v6_minus_tabulated"]!r} '
             f'(bar {rep["resolution"]["error_bar_sum_abs_terminal_steps"]:.4f}; 0.93 tau {BAND_FRACTION * TAU:.2f}); '
             f'first evaluated cycle {first_eval}; t_sum read at {read_t}')
    _log(f'[{tag}] regime: held from {held_from}; natural regime from {reg["natural_regime_from"]}; precondition '
         f'(held from k0 + 1 = {precondition["required_held_from"]}) met {precondition["met"]}')
    if var:
        _log(f'[{tag}] variant (>= {held_from}): {var["decision_core"]["status"]} k* {var["decision_core"]["k_star"]} '
             f'window {var["decision_core"]["window"]} k0 {var["decision_core"]["k0"]} T {var["decision_core"]["T"]}')
    if pred:
        _log(f'[{tag}] prediction {pred["text"]}: held {pred["held"]} ({pred["outcome"]})')
    _log(f'[{tag}] integrity {integrity}; guards ok {gok} ({ {k: v["counts"] for k, v in guards.items()} }); pickle '
         f'{pk["counts"]}; manifest {len(man)} entries; wall {time.time() - t0:.1f} s')
    if not gok:
        sys.exit(1)
    sys.exit(0 if all(integrity.values()) else 3)


def inputs_sha(inputs, rel):
    for v in inputs.values():
        if v['path'] == rel:
            return v['sha256']
    return None


def _f(x, nd=2):
    return 'None' if x is None else f'{x:,.{nd}f}'


def summary_md(doc):
    rep = doc['report']
    dc = doc['decision_core_primary']
    reg = doc['regime']['summary']
    pre = doc['regime']['precondition_holds_from_k0_plus_1']
    var = doc['variant_restricted_to_held_cycles']
    pred = doc['prediction']
    inv = doc['capture_path']['inventory']
    L = ['# W168 -- settling rule v6 replayed on the x = 0 reference `ref:7aa017f0` (zero-solve)', '',
         f"Generated by `{doc['script']['path']}` (sha256 {doc['script']['sha256'][:8]}, committed clean "
         f"{doc['script']['committed_clean']}), git HEAD {doc['git_head'][:8]}, {doc['utc']}. Objective convention: "
         f"{doc['objective_convention']}.", '',
         f"Instance: eval_key `{doc['instance']['eval_key']}`, candidate_key `{doc['instance']['candidate_key']}` "
         f"(x = 0), eval dir `{doc['instance']['eval_dir']}`; per_cycle_record.jsonl sha256 "
         f"{doc['record_hashes'][PCR][:8]}, settling_decision.json {doc['record_hashes'][DEC][:8]}.", '',
         '## Verdict', '',
         f"- v6 as implemented (`HookedRuleV6`, the v6 campaign's rule class; cross-checked by "
         f"`settling_criterion_v6.replay`, equal: {doc['integrity']['hooked_equals_replay']}): **{dc['status']}** at "
         f"k\\*' = **{dc['k_star']}**, branch {dc.get('branch')}.",
         ]
    if rep.get('k_star_v6') is not None:
        gc = rep['gap_clause']
        L += [f"- Window [{rep['window_start']}, {rep['window_end']}], n_w = {rep['n_w']} (P_hat {rep['P_hat']}); "
              f"range {_f(rep['range'])} EUR, range/tau {rep['range_over_tau']:.4f}; band {rep['band']}.",
              f"- Turning points {rep['turning_points_all']}; used for P_hat {rep['turning_points_used_for_P_hat']}; "
              f"swings {[round(a, 2) for a in rep['swings_A']]}; swing check (v6 (A), floor tau/10 = "
              f"{_f(doc['constants']['SWING_FLOOR'])}): ok {rep['swing_check']['ok']}, excluded "
              f"{rep['swing_check']['excluded_swing_indices']}, pairs {rep['swing_check']['pairs_compared']}; (B) "
              f"rejections {rep['turning_point_floor_rejections']}.",
              f"- Gap clause: |t_sum(k\\*')| = {_f(gc['abs_t_sum'])} vs tau/2 = {_f(gc['gap_bound_tau_over_2'])} "
              f"(ratio {gc['abs_t_sum_over_gap_bound']:.4f}); production terminal t = "
              f"{gc['production_terminal_t_tso_plus_t_dso']!r} passes {gc['passes_with_production_terminal_value']}. "
              f"The rule read t_sum only at cycles {gc['cycles_at_which_the_rule_read_t_sum']}; gap refusals "
              f"{gc['gap_refusals']}.",
              f"- Clean-cycle veto: {rep['clean_veto']['n_vetoes']} vetoes; non-clean cycles "
              f"{rep['clean_veto']['non_clean_cycles']}; window all clean {rep['clean_veto']['window_all_clean']}.",
              f"- Q(k\\*') = {rep['Q_k_star_v6']!r}; Q(181) tabulated = {rep['Q_181_tabulated']!r}; "
              f"**Q(k\\*') - Q(181) = {rep['dQ_v6_minus_tabulated']!r}** EUR. Resolution: terminal step at k\\*' "
              f"{_f(rep['resolution']['terminal_step_at_k_star_v6'], 4)}, at 181 "
              f"{_f(rep['resolution']['terminal_step_at_181'], 4)}, bar {_f(rep['resolution']['error_bar_sum_abs_terminal_steps'], 4)}"
              f" -- {rep['resolution']['note']}. Against 0.93 tau = {_f(rep['band_0_93_tau'])}: "
              f"|dQ| <= 0.93 tau {rep['abs_dQ_le_0_93_tau']}.",
              f"- Same as the v1 certificate on {', '.join(k for k, v in rep['same_as_v1_certificate'].items() if v)}"
              f"; differs on {[k for k, v in rep['same_as_v1_certificate'].items() if not v] or 'nothing'}.",
              f"- Terminal step / EPS0 at k\\*' = {rep['terminal_step_to_threshold']['terminal_step_abs_over_EPS0']:.3f}"
              f"; range/tau {rep['range_over_tau']:.4f}.", '']
    else:
        L += [f"- NOT decided within the record (cycles 1..181). State at 181: {rep.get('state_at_181')}", '']
    L += ['## Held-regime start', '',
          f"- The v6 campaign holds the regime from k0_run + 1 ({pre['campaign_rule']}). Rule k0 = {pre['k0_rule']} "
          f"-> required held from {pre['required_held_from']}; the W101 record holds all four (AA, tail apply, tail "
          f"next, rho) from **{reg['first_cycle_all_four_held_through_end']}** -> precondition met: **{pre['met']}**.",
          f"- Natural values (unheld): AA action 'off' from {reg['natural_aa_off_from']}; tight tail passed from "
          f"{reg['natural_tight_tail_passed_from']}; rho frozen on all channels at its end value from "
          f"{reg['natural_rho_frozen_all_channels_and_equal_to_end_value_from']} -> natural regime from "
          f"{reg['natural_regime_from']} (<= k0 + 1: {pre['natural_regime_from_le_k0_plus_1']}).",
          f"- Phases: {reg['phase_by_cycle_ranges']}.", '']
    if var:
        vc = var['decision_core']
        L += [f"- {var['label']}: **{vc['status']}** at k\\* = {vc['k_star']}, window {vc['window']}, k0 {vc['k0']}, T "
              f"{vc['T']}, A {vc['A']}, range/tau {vc['range_over_tau']}, t_sum {vc['t_sum_k_star']}.", '']
    if pred:
        o = pred['outcome']
        L += ['## Recorded prediction', '', f"{pred['source']}: \"{pred['text']}\".", '',
              f"- k\\*' in [174, 195]: {o['k_star_in_174_195']} (k\\*' = {o['k_star_v6']})",
              f"- window range <= tau: {o['range_le_tau']} (range/tau {o['range_over_tau']:.4f})",
              f"- |Q(k\\*') - Q(181)| <= 0.93 tau: {o['abs_dQ_le_0_93_tau']} (dQ = {o['dQ']!r})",
              f"- **Prediction held: {pred['held']}**", '']
    L += ['## Inputs (capture path, asserted before the rule ran)', '']
    for k, v in inv.items():
        L += [f"- {k}: {v['status']} -- {v['source']}"]
    L += ['', f"Checks: {doc['capture_path']['checks']}", '',
          f"all_clean_k cross-check vs the W142 floor replay: {doc['all_clean']['crosscheck']}; class counts "
          f"{doc['all_clean']['class_counts']}; logs read vs W131 inventory: "
          f"{ {k: v for k, v in doc['capture_path']['log_crosscheck_w131'].items()} }.", '',
          f"Integrity: {doc['integrity']}. Guards ok {doc['guards_ok']} (solves {doc['solves']}); pickle "
          f"{doc['pickle_guard']['counts']}.", '']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
