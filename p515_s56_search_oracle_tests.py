"""P5.15 S56 EV search driver -- INTEGRATION TESTS (engineering build: a script with pass/fail checks, no prediction
spec). Authority: S56_SEARCH_DRIVER_DESIGN.md "Oracle mapping and key rules" (Planner decisions, acc80a2e);
P5_15_S56_SEARCH_DRIVER_REPORT.md (driver build dcea9628 / 1d3b6cdb); the Planner's integration task of 2026-11-23.

MODES (each writes a NEW directory under data/SRP1/Results/P515S56/search_oracle_v2/ -- v1 holds the integration
build's evidence (e1b87c27) and is never written; attached, both streams captured by the caller:
`> <dir>_stdout.log 2> <dir>_stderr.log`):
  --zero-solve   ZERO SOLVES (SolveProfileGuard(permitted=()) installed before any repository import, verify(0);
                 pickle.load / loads blocked and counted 0):
     T1 the mapping on REAL records (the committed ev_harness_v1 smoke: x0 and the reference unit, 2x, 2 cycles):
        F = I(x) + recourse = the cost breakdown total; bar_F = the record's net-recourse step (I constant) and <= the
        triangle bound; refusals (exit code, status error, missing files) -> harness_error, never a value.
     T2 the settled mapping's arithmetic: the smoke unit's files + a SYNTHETIC pricing fixture with hand-computed PR,
        A, G and gates (independent recomputation), the settled conditions one by one, the gates' boundaries.
     T3 cap in the eval key: composition == freeze_campaign_spec (S56 entries, caps 2 / 3 / 300 / 500, both depths);
        settled runs at different K distinct; the alias with ev_harness_v1/smoke gone (recomposed under the new rule
        and through the driver's freeze-time collision scan on the 2x configuration); the pricing key's refusals and
        keying.
     T4 option-off identity: harness diff vs the pre-integration commit is insertions only; every entry of every
        committed campaign spec WITHOUT an S56 EV-plan key: full eval key post == pre (== recorded where pre ==
        recorded), candidate keys unchanged; entries WITH S56 keys listed (they change by design); freeze pre == post
        on four entry sets without S56 keys; build_evaluation_record pre == post on committed records.
     T5 preflight refusals: memory rule (injected vm_stat), undeclared per-slot, provisional mapping, harness
        preconditions, solver path; real freeze / pilot refusals; the mapping capture-path assertion. Rulings 1-7
        (601775c0): the mapping parameters FINAL; a real pilot spec refuses without complete predictions or with a
        per-slot figure other than 4.0, carries s_share 500 and the ruled staged plan, and passes the pilot
        capture-path checklist.
     T6 cache roots and the start pool (fixture records in a scratch root, labelled SYNTHETIC): index, ignore reasons,
        re-verification (a tampered record refuses), best start with its runner-up margin.
     T7 the pilot against stubs: freeze (both grids), two attached launches (blocks 5 + 2) through the driver CLI,
        a third launch = done, evaluation (synthetic traces, s_share read from the frozen spec) -> K_EV, e_EV and every
        reported output; refusals; --pilot-run --preflight-only (refused on a stub pilot; on a test-only harness
        pilot: zero solves, record written, nothing launched, the pilot root unchanged, the lock released).
     T8 the wall-time projection and the count formula.
     T9 a stub-children search at 2x through the driver CLI: frozen (the alias refusal gone), run by attached
        launches to termination, final state == the in-process synthetic replay.
  --real-check   AFTER the memory probe: ONE real 3-cycle evaluation of the reference unit, `s56_settled_stop_disabled`
                 True and `s56_pricing_capture` True, cap 3, SRP1-grid EV twin, 2x; the parent guard at 0, the child's
                 solves == the declared ((1 + 3) x 12 + 3) x (3 + 1) = 204; the mapping must give a COMPLETE summary,
                 status not_settled.
  --manifest     sha256 of every file under the output root + the code files.
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
os.chdir(CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 search oracle tests parent (never solves)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('oracle tests: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('oracle tests: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s44_campaign_harness as H  # noqa: E402  (parent side: no model import)
import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'search_oracle_v2')   # v1: e1b87c27 (never written)
OUT = os.path.join(CODE_ROOT, OUT_REL)
SMOKE = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'ev_harness_v1', 'smoke', 'campaign_s56evh_v1_smoke')
SMOKE_SPEC = os.path.join(SMOKE, 'campaign_spec_s56evh_v1_smoke_39882ca6.json')
SMOKE_EVALS = {'unit': os.path.join(SMOKE, 'evals', '45368596f9cb1a2e_unit'),
               'x0': os.path.join(SMOKE, 'evals', '1933e753d4bae5ab_x0')}
PRE_REV = 'acc80a2e'   # the harness before this integration (the driver build's harness + the design commit)
REAL_CAP = 3
REAL_DECLARED_SOLVES = ((1 + 3) * 12 + 3) * (REAL_CAP + 1)   # 204: (1 + n_dso) n_years n_days + n_esso, x (cycles + 1)
SOLVER_PATH = '/usr/local/bin/ipopt'
CHECKS = []


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(item, name, ok, detail=None):
    CHECKS.append({'item': item, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{item}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def git(*argv):
    r = subprocess.run(['git', *argv], cwd=CODE_ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def new_dir(name):
    d = os.path.join(OUT, name)
    if os.path.exists(d) and [n for n in os.listdir(d) if not n.endswith('.log')]:
        raise SystemExit(f'REFUSE: {d} exists and holds non-log files (write-once)')
    os.makedirs(d, exist_ok=True)
    return d


def _evt():
    import p515_s56_ev_harness_tests as EVT
    return EVT


def harness_config(flex=2.0, exit_cap=500, k=400, conc=5, per_slot=3.0):
    """The EV harness configuration (the driver gate's `harness_config` with the pricing key at the settled depth)."""
    EVT = _evt()
    derived, evp = EVT.srp1grid_ev_declaration()
    cfg, required = EVT._spec_like(derived, cap=exit_cap)
    opts = {H.S56_COHORT_FORMULATION_KEY: 'independent_units_all_node_years', H.S56_TV_IN_OBJECTIVE_KEY: True,
            H.S56_ESSO_SLACK_PENALTY_KEY: 1e4, 'flex_price_multiplier': flex, H.S56_EV_PROFILES_KEY: dict(evp)}
    settled = dict(opts, **{H.S56_SETTLED_STOP_DISABLED_KEY: True, H.S56_PRICING_CAPTURE_KEY: True})
    return {'by_depth': {'exit': {'configuration': cfg, 'cap': exit_cap, 'entry_options': opts},
                         'settled': {'configuration': cfg, 'cap': k, 'entry_options': settled}},
            'concurrency': conc, 'required_consecutive_cycles': required, 'memory_per_slot_gib': per_slot}


def oracle_for(lat, hc, kind='stub_children'):
    spec = {'oracle': {'kind': kind, 'seed': 'oracle-tests', 'target': list(lat.x0())},
            'start': {'z': list(lat.x0())}, 'harness': hc, 'arm': 'k', 'test_only': True, 'search_id': 'k'}
    return D.HarnessOracle(spec, lat, lock_path=None)


# ======================================================================================================================
#  T1 the mapping on real records
# ======================================================================================================================
def t1():
    out = {}
    for lab, d in SMOKE_EVALS.items():
        files = D.read_eval_files(os.path.join(CODE_ROOT, d))
        rec, bd = files['record'], files['breakdown']
        ex = D.map_record(files, P.DEPTH_EXIT)
        out[lab] = ex
        inv = rec['s56_investment']['investment_cost_eur']
        last = files['rows'][-1]
        ck(1, f'{lab}: exit summary from the real record: status not_certified (cap 2 reached), read at the terminal '
              f'cycle 2', ex['status'] == 'not_certified' and ex['read_cycle'] == 2
           and ex['read_at'].startswith('terminal cycle'), [ex.get('status'), ex.get('read_at')])
        if lab == 'unit':
            rc_dir = os.path.join(OUT, 'real_check', 'campaign_s56orc_v1_real_check', 'evals')
            if os.path.isdir(rc_dir):   # the real 3-cycle check's record (run before this suite), mapped again here
                ed = os.path.join(rc_dir, os.listdir(rc_dir)[0])
                rf = D.read_eval_files(ed)
                rx = D.map_record(rf, P.DEPTH_EXIT)
                rs = D.map_record(rf, P.DEPTH_SETTLED)
                out['real_check_remapped'] = {'exit': rx, 'settled': {k: rs.get(k) for k in (
                    'status', 'F', 'V', 'gates', 'gates_pass', 'settled_checks', 'leak_flag')}}
                ck(1, 'the real 3-cycle settled-key record re-mapped: exit not_certified read at the terminal cycle 3 '
                      '(no would-be certification cycle); settled complete, not_settled',
                   rx['status'] == 'not_certified' and rx['read_cycle'] == 3 and rx['read_at'].startswith(
                       'terminal cycle') and rs['status'] == D.NOT_SETTLED_STATUS
                   and all(D._fin(rs['V'][k]) for k in 'S PR G A'.split()), [rx.get('read_at'), rs.get('status')])
        ck(1, f'{lab}: F = I(x) + recourse(2) exactly, = the cost breakdown total_planning_cost_eur',
           ex['F'] == inv + last['recourse'] and abs(ex['F'] - bd['total_planning_cost_eur']) <= 1e-9 * ex['F'],
           [ex['F'], inv + last['recourse'], bd['total_planning_cost_eur']])
        ck(1, f'{lab}: recourse = gross - TV on every cycle (exact)', ex['recourse_identity_bad_cycles'] == [])
        ref = rec['bar_net_recourse_step_reported']['value']
        ck(1, f'{lab}: bar_F = the record\'s net-recourse step (I constant), rel 1e-12; <= the triangle bound '
              f'bar_gross + range(TV)', abs(ex['bar'] - ref) <= 1e-12 * ref and ex['bar_within_triangle_bound'],
           [ex['bar'], ref, ex['bar_triangle_bound']])
        st = D.map_record(files, P.DEPTH_SETTLED)
        ck(1, f'{lab}: settled mapping of a record without the pricing capture -> harness_error naming the file',
           st['status'] == 'harness_error' and 'pricing' in st['reason'] and st['F'] is None, st.get('reason'))
        for name, mut in (('exit code 1', lambda f: f.update(exit_code=1)),
                          ('record status error', lambda f: f['record'].update(status='error')),
                          ('no per-cycle file', lambda f: f.update(rows=None)),
                          ('no exit_code.txt', lambda f: f.update(exit_code=None))):
            f2 = copy.deepcopy(files)
            mut(f2)
            r = D.map_record(f2, P.DEPTH_EXIT)
            ck(1, f'{lab}: {name} -> harness_error, no value', r['status'] == 'harness_error' and r['F'] is None,
               r.get('reason'))
        f3 = copy.deepcopy(files)
        f3['record']['s56_investment'] = {}
        r = D.map_record(f3, P.DEPTH_EXIT)
        ck(1, f'{lab}: no I(x) in the record -> harness_error', r['status'] == 'harness_error', r.get('reason'))
        f4 = copy.deepcopy(files)
        f4['record']['solve_profile']['identity_holds'] = False
        r = D.map_record(f4, P.DEPTH_EXIT)
        ck(1, f'{lab}: guards not exact -> not certified (listed, never a value accepted)',
           r['status'] == 'not_certified' and r['guards']['solve_profile_identity_holds'] is False)
    dF = out['unit']['F'] - out['x0']['F']
    out['dF_unit_minus_x0_cycle2'] = dF
    out['note'] = ('2-cycle unconverged smoke: a mapping test, not a result; the difference is indeterminate '
                   '(bar 891.5 MEUR)')
    return out


# ======================================================================================================================
#  T2 settled arithmetic (synthetic pricing fixture on the real smoke files)
# ======================================================================================================================
def fixture_pricing(sigma=1000.0):
    """A small SYNTHETIC pricing fixture (2 interface periods, 1 ESS block of 2 periods, 1 slack block, 2 closure rows,
    2 settlement blocks), every value chosen so that PR, A, G are hand-computable."""
    return {
        'schema': H.S56_PRICING_SCHEMA, 'sigma': sigma, 'synthetic': True,
        'interface': [{'node_id': 7, 'year': 2025, 'day': 'Spring', 'w': 2.0, 'tso_base_mva': 100.0,
                       'rating_mva': 200.0, 'v_base_kv': 10.0,
                       'p_tso': [1.0, 2.0], 'p_dso': [0.99, 2.0], 'q_tso': [0.5, 0.0], 'q_dso': [0.5, 0.004],
                       'v_tso': [10.0, 10.1], 'v_dso': [10.0, 10.0],
                       'lambda_p_tso': [4.0, 0.0], 'lambda_q_tso': [0.0, 2.0], 'lambda_v_tso': [0.0, 3.0],
                       'lambda_p_dso': [0.0, 0.0], 'lambda_q_dso': [0.0, 0.0], 'lambda_v_dso': [0.0, 0.0]}],
        'ess': [{'node_id': 7, 'year': 2025, 'day': 'Spring', 'rating_mva': {'tso': 2.5, 'dso': 2.5, 'esso': 1.25},
                 'z_p': [0.10, 0.0], 'z_q': [0.0, 0.0],
                 'x_p_tso': [0.11, 0.0], 'x_p_dso': [0.10, 0.0], 'x_p_esso': [0.10, 0.02],
                 'x_q_tso': [0.0, 0.0], 'x_q_dso': [0.0, 0.0], 'x_q_esso': [0.0, 0.0],
                 'lambda_p_tso': [5.0, 0.0], 'lambda_p_dso': [0.0, 0.0], 'lambda_p_esso': [0.0, -1.0],
                 'lambda_q_tso': [0.0, 0.0], 'lambda_q_dso': [0.0, 0.0], 'lambda_q_esso': [7.0, 0.0]}],
        'esso_slacks': [{'node_id': 7, 'year': 2025, 'day': 'Spring', 'p_up': [0.0, 1e-6], 'p_down': [0.0, 0.0],
                         'q_up': [2e-6, 0.0], 'q_down': [0.0, 0.0]}],
        'closure': [{'kind': 'TSO', 'node_id': None, 'year': 2025, 'day': 'Spring', 'index': [0, 0, 0],
                     'slack_up_pu': 3e-6, 'slack_down_pu': 0.0, 'base_mva': 100.0, 'row_dual': -0.5, 'active': True},
                    {'kind': 'DSO', 'node_id': 7, 'year': 2025, 'day': 'Spring', 'index': [0, 0, 0],
                     'slack_up_pu': 0.0, 'slack_down_pu': 0.0, 'base_mva': 100.0, 'row_dual': 9.0, 'active': True}],
        'settlement_blocks': [{'kind': 'TSO', 'node_id': None, 'year': 2025, 'day': 'Spring', 'total': -100.0,
                               'contracted': -100.0, 'deviation': 0.0},
                              {'kind': 'DSO', 'node_id': 7, 'year': 2025, 'day': 'Spring', 'total': 70.0,
                               'contracted': 70.0, 'deviation': 0.0}],
        'efficiency': {'7': {'2025': [0.95, 0.95], '2030': [0.95, 0.95], '2035': [0.95, 0.95]}}, 'units': {}}


def fixture_expected(sigma=1000.0, leak=5.0):
    """The fixture's PR / A / G by hand (the formula of p515_s56_search_driver.pricing_components, re-derived)."""
    i_p = 4.0 * 0.01 * sigma / (100.0 * 200.0)              # lam_p dP / (S_b R)
    i_q = 2.0 * (-0.004) * sigma / (100.0 * 200.0)          # lam_q dQ / (S_b R)
    i_v = 3.0 * 0.1 * sigma / (10.0 ** 2)                    # lam_v dV / V_b^2
    e_p = sigma * 5.0 * 0.01 / (2 * 2.5) + sigma * (-1.0) * 0.02 / (2 * 1.25)   # tso period 0 + esso period 1
    e_q = sigma * 7.0 * 0.0 / (2 * 1.25)
    dual = i_p + i_q + i_v + e_p + e_q
    price = -(-100.0 + 70.0)                                  # -sum settlement total
    pr = -price + dual
    a = abs(price) + abs(i_p) + abs(i_q) + abs(i_v) + abs(sigma * 5.0 * 0.01 / 5.0) + abs(sigma * -1.0 * 0.02 / 2.5)
    g_slack = sigma * abs(-1.0) * 1e-6 / 2.5 + sigma * abs(7.0) * 2e-6 / 2.5
    g_closure = sigma * 0.5 * 3e-6
    return {'PR': pr, 'A': a, 'G': g_slack + leak + g_closure, 'dual': dual, 'price': price}


def _settled_files(lab='unit', K=2, conv=(True, True), tail=(True, True), leak=5.0, sigma=1000.0):
    files = D.read_eval_files(os.path.join(CODE_ROOT, SMOKE_EVALS[lab]))
    files['record']['s56_settled_stop_disabled'] = True
    files['record']['s56_certification_stop_summary'] = {'ok': True}
    files['record']['cap'] = K
    for r, c in zip(files['rows'], conv):
        r['cycle_convergence'] = c
    for p, t in zip(files['tail']['per_cycle'], tail):
        p['active'] = t
    files['breakdown']['leak_detector']['priced_loss_eur'] = leak
    pr = fixture_pricing(sigma)
    tot = sum(s['total'] for s in pr['settlement_blocks'])
    files['record']['recourse_components']['interface_settlement_total'] = tot
    files['pricing'] = pr
    return files


# The S56 pilot's seven real eval dirs (committed with P5_15_S56_PILOT_REPORT.md): the settlement identity on real
# cancelling block totals (TSO ~ -1.3e9 EUR, DSO ~ +1.3e9 EUR, net 6-72 EUR).
PILOT_EVALS = [os.path.join('data', 'SRP1', 'Results', 'P515S56', 'pilot_ev_v1', 'launches', f'launch_00{n}', 'evals',
                            d) for n, d in ((1, '142a02890a1d630f_x0_i1'), (1, 'fe40736832cfbdba_unit_i1'),
                                            (1, 'dddc5b8382a39c0a_staged_i1'), (1, '830a6306e0c27969_x0_i2'),
                                            (1, '6b05f4c326441274_unit_i2'), (2, '0f20f46f0a99c162_x0_i3'),
                                            (2, 'ecac63e76319052f_unit_i3'))]


def _production_order_total(blocks, part='total'):
    """The order of shared_resources_planning._get_operational_recourse_components (:1254-1260): TSO blocks summed,
    DSO blocks accumulated per node, total = tso + sum(per-node). Re-stated here to show the order is the cause."""
    tso = sum(b[part] for b in blocks if b['kind'] == 'TSO')
    dso = {}
    for b in blocks:
        if b['kind'] == 'DSO':
            dso[b['node_id']] = dso.get(b['node_id'], 0.0) + b[part]
    return tso + sum(dso.values())


def settlement_identity_checks(params):
    """The settlement identity checks under the identity tolerance (Expert handback 3 ruling 2; the Planner's ruling
    of the pilot evaluation): max(64 u sum|per-block totals|, 1e-6 EUR)."""
    def old_rule(a, b):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))

    ck(2, 'identity_tolerance: the 1e-6 EUR floor on small totals; 64 u sum|s| on large ones',
       D.identity_tolerance([-100.0, 70.0]) == 1e-6 and D.identity_tolerance([-2.0e9, 2.0e9]) == 64 * 2.0 ** -53
       * 4.0e9, [D.identity_tolerance([-100.0, 70.0]), D.identity_tolerance([-2.0e9, 2.0e9])])
    # a cancelling fixture (SYNTHETIC values at the pilot's magnitudes): production order != list order
    vals = [('TSO', None, -647616499.7659234), ('TSO', None, -647616499.7651111), ('DSO', 5, 431744001.11370003),
            ('DSO', 7, 431744000.9), ('DSO', 9, 431745003.6273345)]
    days = ('Spring', 'Summer', 'Autumn', 'Winter', 'Spring')

    def big(record_offset=0.0, contracted_offset=0.0):
        f = _settled_files()
        f['pricing']['settlement_blocks'] = [
            {'kind': k, 'node_id': n, 'year': 2025 if i < 4 else 2030, 'day': days[i], 'total': v,
             'contracted': v + (contracted_offset if i == 0 else 0.0), 'deviation': 0.0}
            for i, (k, n, v) in enumerate(vals)]
        f['record']['recourse_components']['interface_settlement_total'] = \
            _production_order_total(f['pricing']['settlement_blocks']) + record_offset
        return f

    fb = big()
    blocks = fb['pricing']['settlement_blocks']
    list_sum, prod_sum = sum(b['total'] for b in blocks), _production_order_total(blocks)
    tol = D.identity_tolerance(b['total'] for b in blocks)
    ck(2, 'settlement fixture sanity: production-order and list-order sums differ, beyond the old 1e-9 x max(1, |T|) '
          'form and within the identity tolerance', list_sum != prod_sum and not old_rule(list_sum, prod_sum)
       and abs(list_sum - prod_sum) <= tol, [list_sum, prod_sum, list_sum - prod_sum, tol])
    sb = D.map_record(fb, P.DEPTH_SETTLED, params)
    sc = sb.get('settlement_check') or (sb.get('pricing') or {}).get('settlement_check') or {}
    ck(2, 'cancelling blocks summed in another order than production reconcile with the record (not harness_error)',
       sb['status'] != 'harness_error' and sc.get('blocks_reconcile_with_record') is True
       and sc.get('contracted_equals_total') is True and sc.get('identity_tolerance_eur') == tol,
       [sb.get('status'), sb.get('reason'), sc])
    for name, kw, key in (('record offset 2 x tolerance', {'record_offset': 2 * tol}, 'blocks_reconcile_with_record'),
                          ('contracted offset 2 x tolerance', {'contracted_offset': 2 * tol},
                           'contracted_equals_total')):
        s_ = D.map_record(big(**kw), P.DEPTH_SETTLED, params)
        sc_ = (s_.get('pricing') or {}).get('settlement_check') or {}
        ck(2, f'settlement identity: {name} -> {key} False -> harness_error', s_['status'] == 'harness_error'
           and sc_.get(key) is False, [s_.get('status'), sc_.get(key)])
    for off, want in ((0.9e-6, True), (1.1e-6, False)):
        f_ = _settled_files()
        f_['record']['recourse_components']['interface_settlement_total'] += off
        s_ = D.map_record(f_, P.DEPTH_SETTLED, params)
        sc_ = s_.get('settlement_check') or (s_.get('pricing') or {}).get('settlement_check') or {}
        ck(2, f'settlement identity floor 1e-6 EUR on the small fixture: record offset {off:g} -> reconcile {want}',
           sc_.get('blocks_reconcile_with_record') is want and (s_['status'] == 'harness_error') is (not want),
           [s_.get('status'), sc_])
    # the seven real pilot records: the record's total is the production-order sum of the captured blocks BITWISE
    # (both float64), and the mapping's list-order sum reconciles under the identity tolerance
    rows = []
    for rel in PILOT_EVALS:
        files = D.read_eval_files(os.path.join(CODE_ROOT, rel))
        if not files.get('record') or not files.get('pricing'):
            rows.append({'eval_dir': rel, 'missing': True})
            continue
        bl, rec_t = files['pricing']['settlement_blocks'], files['record']['recourse_components'][
            'interface_settlement_total']
        lt, pt, tl = sum(b['total'] for b in bl), _production_order_total(bl), D.identity_tolerance(
            b['total'] for b in bl)
        rows.append({'eval_dir': rel, 'record': rec_t, 'production_order_sum': pt, 'list_order_sum': lt,
                     'abs_diff': abs(lt - rec_t), 'tolerance': tl, 'ratio': abs(lt - rec_t) / tl,
                     'record_is_production_order_bitwise': pt == rec_t,
                     'contracted_record_is_production_order_bitwise': _production_order_total(bl, 'contracted') ==
                     files['record']['recourse_components']['interface_settlement_contracted_total'],
                     'old_rule_holds': old_rule(lt, rec_t), 'new_rule_holds': abs(lt - rec_t) <= tl})
    ck(2, 'pilot records (7, real): record interface_settlement_total == production-order sum of the captured blocks '
          'bitwise (and the contracted total likewise) -- summation order, not reduced precision',
       len(rows) == 7 and all(r.get('record_is_production_order_bitwise') is True and
                              r.get('contracted_record_is_production_order_bitwise') is True for r in rows), rows)
    ck(2, 'pilot records (7, real): list-order sum reconciles under the identity tolerance on all 7; the old 1e-9 form '
          'failed on all 7', all(r.get('new_rule_holds') is True and r.get('old_rule_holds') is False for r in rows),
       rows)
    return rows


def t2():
    params = dict(D.MAPPING_PARAMETERS, W=2, bar_window=2)
    exp = fixture_expected()
    f = _settled_files()
    s = D.map_record(f, P.DEPTH_SETTLED, params)
    ck(2, 'settled summary complete on the fixture (real smoke files + synthetic pricing, W = 2 for a 2-cycle record)',
       s['status'] in (D.SETTLED_STATUS, D.NOT_SETTLED_STATUS) and all(D._fin(s['V'][k]) for k in 'S PR G A'.split())
       and isinstance(s['gates_pass'], bool), s.get('reason') or s.get('status'))
    for k in ('PR', 'A', 'G'):
        ck(2, f'{k} = the independent hand computation (rel 1e-12)', abs(s['V'][k] - exp[k]) <= 1e-12 * max(1.0,
                                                                                                          abs(exp[k])),
           [s['V'][k], exp[k]])
    ck(2, 'PR sign: Q - price + dual (Expert ruling 3): PR = -price + dual', abs(
        s['V_parts']['PR']['minus_price'] + exp['price']) <= 1e-12 and abs(s['V_parts']['PR']['dual'] - exp['dual'])
       <= 1e-9 * abs(exp['dual']), s['V_parts']['PR'])
    rows = f['rows']
    Fs = [rows[0]['recourse'], rows[1]['recourse']]
    ck(2, 'S = range of F over the last W cycles (rel 1e-12)', abs(s['V']['S'] - abs(Fs[1] - Fs[0])) <= 1e-12 * abs(
        Fs[1] - Fs[0]), [s['V']['S'], abs(Fs[1] - Fs[0])])
    ck(2, 'all settled conditions hold on the fixture -> status settled', s['status'] == D.SETTLED_STATUS,
       s['settled_checks'])
    for name, kw, key in (('a cycle without the production certification test', {'conv': (True, False)},
                           'would_have_certified_each_of_last_W'),
                          ('a cycle not at tail depth', {'tail': (False, True)}, 'last_W_at_tail_depth')):
        s2 = D.map_record(_settled_files(**kw), P.DEPTH_SETTLED, params)
        ck(2, f'settled condition: {name} -> not_settled, the failed condition named',
           s2['status'] == D.NOT_SETTLED_STATUS and key in s2['not_settled_reasons'], s2.get('not_settled_reasons'))
    s3 = D.map_record(_settled_files(K=3), P.DEPTH_SETTLED, params)
    ck(2, 'a record that stopped before its cap K (2 cycles of K = 3) -> harness_error: F(K) and S do not exist '
          '(reached_K False recorded)', s3['status'] == 'harness_error' and s3['settled_checks']['reached_K'] is False,
       [s3.get('status'), s3.get('reason')])
    f5 = _settled_files()
    f5['record']['solve_profile']['observed']['blocked_solve'] = 1
    s5 = D.map_record(f5, P.DEPTH_SETTLED, params)
    ck(2, 'a blocked solve -> not_settled (guards) and gates_pass False', s5['status'] == D.NOT_SETTLED_STATUS
       and s5['gates_pass'] is False)
    ck(2, 'gates on the fixture: dP 0.01 MW <= 12.5 kW; ESSO slack 2e-6 <= 1e-5; closure 3e-6 p.u. <= 1e-5; D1',
       s['gates'] == {'status_ok_and_guards_exact': True, 'interface_p_ceiling_12p5kw': True, 'esso_slack_gate': True,
                      'network_closure_slack_le_tau_cl': True, 'd1_efficiency_equality': True}, s['gates'])
    # boundaries
    for name, mut, gate, want in (
            ('dP = 12.5 kW exactly passes', lambda p: p['interface'][0].update(p_dso=[1.0 - 0.0125, 2.0]),
             'interface_p_ceiling_12p5kw', True),
            ('dP = 12.6 kW fails', lambda p: p['interface'][0].update(p_dso=[1.0 - 0.0126, 2.0]),
             'interface_p_ceiling_12p5kw', False),
            ('ESSO slack 2e-5 fails', lambda p: p['esso_slacks'][0].update(q_up=[2e-5, 0.0]), 'esso_slack_gate', False),
            ('closure 2e-5 p.u. fails', lambda p: p['closure'][0].update(slack_up_pu=2e-5),
             'network_closure_slack_le_tau_cl', False),
            ('unequal efficiencies across active cohorts fail D1',
             lambda p: p['efficiency']['7'].update({'2025': [0.95, 0.94]}), 'd1_efficiency_equality', None)):
        f6 = _settled_files()
        mut(f6['pricing'])
        if gate == 'd1_efficiency_equality':
            f6['record']['s56_investment']['active_cohorts'] = [[7, 2025], [7, 2030]]
            want = False
        s6 = D.map_record(f6, P.DEPTH_SETTLED, params)
        ck(2, f'gate boundary: {name}', s6['gates'][gate] is want and s6['gates_pass'] is (want and all(
            v for k, v in s6['gates'].items() if k != gate)), s6['gates'])
    f7 = _settled_files(leak=25.01)
    s7 = D.map_record(f7, P.DEPTH_SETTLED, params)
    ck(2, 'leak 25.01 EUR > 25: flagged, NOT in gates_pass (reported)', s7['leak_flag'] is True and s7['gates_pass']
       is True and 'leak' not in ' '.join(s7['gates']), s7['leak'])
    f8 = _settled_files()
    f8['pricing']['closure'][0]['row_dual'] = None
    s8 = D.map_record(f8, P.DEPTH_SETTLED, params)
    ck(2, 'a closure row without its dual -> harness_error (incomplete G), never a silent 0',
       s8['status'] == 'harness_error', s8.get('reason'))
    f9 = _settled_files()
    f9['pricing']['settlement_blocks'][0]['deviation'] = 1.0
    s9 = D.map_record(f9, P.DEPTH_SETTLED, params)
    ck(2, 'a non-zero settlement deviation part (not one scenario) -> harness_error (the price derivation fails)',
       s9['status'] == 'harness_error', s9.get('reason'))
    pilot_settlement_rows = settlement_identity_checks(params)
    # the policy consumes the summary
    try:
        c = P.confirm(s, s, dict(D.map_record(f, P.DEPTH_EXIT, params), status='certified'),
                      dict(D.map_record(f, P.DEPTH_EXIT, params), status='certified'), 15000.0)
        ok = c['result'] in ('accepted', 'unresolved_after_confirmation')
    except Exception as error:  # noqa: BLE001
        ok, c = False, str(error)
    ck(2, 'the policy\'s confirm() consumes two mapped settled summaries (shape check)', ok, c)
    return {'fixture_expected': exp, 'summary': {k: s[k] for k in ('status', 'F', 'V', 'gates_pass', 'gates',
                                                                  'settled_checks')},
            'pilot_settlement_identity_rows': pilot_settlement_rows,
            'note': 'SYNTHETIC pricing fixture on the real smoke files: an arithmetic check, not a measurement'}


# ======================================================================================================================
#  T3 cap in the eval key; the pricing key
# ======================================================================================================================
def t3(lat, scratch):
    out = {}
    names = {n: D.named_start(lat, n) for n in ('x0', 'reference_unit', 'plan_2030')}
    mism, keys = [], {}
    for cap in (2, 3, 300, 500):
        hc = harness_config(exit_cap=cap, k=cap)
        orc = oracle_for(lat, hc)
        for depth in P.DEPTHS:
            bd = hc['by_depth'][depth]
            cands = [(f'{n}_{depth}', None, orc.entry_options(z, depth)) for n, z in names.items()]
            _p, _s, hs = H.freeze_campaign_spec(os.path.join(scratch, f'cap{cap}_{depth}'), f'cap{cap}{depth}', cands,
                                                dict(bd['configuration'], name='t3'), cap, 5, ['oracle tests T3'],
                                                required_consecutive_cycles=hc['required_consecutive_cycles'],
                                                extra={'test_only_stub': True})
            for e in hs['candidates']:
                z = names[e['label'].rsplit('_', 1)[0]]
                keys[(cap, depth, e['label'])] = e['eval_key']
                if e['eval_key'] != orc.eval_key(z, depth) or e.get('s56_cycle_cap_keyed') != cap:
                    mism.append([cap, e['label']])
    ck(3, 'driver composition == freeze_campaign_spec on S56 staged entries (3 points x 2 depths x caps 2/3/300/500); '
          'each entry records the keyed cap', not mism, mism[:5])
    ck(3, 'every (cap, depth, point) has a distinct key (24 keys)', len(set(keys.values())) == len(keys) == 24)
    ck(3, 'settled runs at different K have different keys', len({keys[(c, 'settled', 'x0_settled')]
                                                                   for c in (2, 3, 300, 500)}) == 4)
    # the alias with the committed smoke
    smoke = D.load_json(os.path.join(CODE_ROOT, SMOKE_SPEC))
    smoke_keys = {e['eval_key'] for e in smoke['candidates']}
    sm_base = {}
    for e in smoke['candidates']:
        z = names['x0'] if e['label'] == 'x0' else names['reference_unit']
        opts = {k: e[k] for k in H.S56_EV_PLAN_SETTING_KEYS}
        opts['flex_price_multiplier'] = e['flex_price_multiplier']
        opts[H.S56_STAGED_INVESTMENT_KEY] = D.staged_plan_of(lat, z)
        pre = D.compose_eval_key(H, smoke['configuration'], opts, 2, H.candidate_key(lat.canonical_of_z(z)))
        sm_base[e['label']] = pre
    ck(3, 'the smoke entries recomposed under the new rule at their own cap 2 differ from their recorded keys (the '
          'recorded keys predate the cap rule)', all(v not in smoke_keys for v in sm_base.values()), sm_base)
    ex_keys = {orc.eval_key(names[n], d) for cap in (3, 300, 500) for orc in [oracle_for(lat, harness_config(
        exit_cap=cap, k=cap))] for n in ('x0', 'reference_unit') for d in P.DEPTHS}
    ck(3, 'no 2x EV exit / settled key of x0 or the unit at caps 3 / 300 / 500 is a key of the committed smoke spec '
          '(the alias is gone)', not (ex_keys & smoke_keys), len(ex_keys & smoke_keys))
    # the freeze-time collision scan on the 2x configuration (the driver gate's item 4 control, which was REFUSED)
    t0 = time.time()
    spec = {'oracle': {'kind': 'stub_children', 'seed': 'x', 'target': list(lat.x0())}, 'start': {'z': list(lat.x0())},
            'harness': harness_config(), 'arm': 'ev_2x', 'test_only': True, 'search_id': 'alias', 'cache_sources': []}
    try:
        res = D.freeze_collision_scan(spec, lat, os.path.join(scratch, 'alias_root'), CODE_ROOT)
        ok, detail = res['collisions'] == [], {k: v for k, v in res.items() if k != 'collisions'}
    except D.Refused as error:
        ok, detail = False, error.reasons
    ck(3, 'the freeze-time collision scan of the 2x EV configuration (74,084 keys) now finds no committed artefact '
          '(it named the smoke spec before the cap rule)', ok, detail)
    out['collision_scan'] = detail
    out['collision_scan_s'] = time.time() - t0
    # the pricing key
    EVT = _evt()
    derived, evp = EVT.srp1grid_ev_declaration()
    cfg, required = EVT._spec_like(derived, cap=3)
    cfg0, _r = EVT._spec_like(None, cap=3)
    unit = D.staged_plan_of(lat, names['reference_unit'])
    ev_opts = dict(harness_config()['by_depth']['exit']['entry_options'], **{H.S56_STAGED_INVESTMENT_KEY: unit})
    k = [0]

    def freeze(cands, c=cfg):
        k[0] += 1
        return H.freeze_campaign_spec(os.path.join(scratch, f'pk_{k[0]:02d}'), f'pk{k[0]}', cands,
                                      dict(c, name='t3'), 3, 5, ['oracle tests T3'],
                                      required_consecutive_cycles=required, extra={'test_only_stub': True})[2]

    def refused(cands, c=cfg):
        try:
            freeze(cands, c)
            return 'not refused'
        except ValueError as error:
            return str(error)
    r = {'no S56 EV-plan key': refused([('u', {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)},
                                         {'flex_price_multiplier': 2.0, H.S56_PRICING_CAPTURE_KEY: True})], cfg0),
         'per-scenario coupling': refused([('u', None, dict(ev_opts, **{H.S56_PRICING_CAPTURE_KEY: True,
                                                                         'interface_coupling':
                                                                             dict(EVT.COUPLING_V1_NOQ)}))]),
         'non-bool': refused([('u', None, dict(ev_opts, **{H.S56_PRICING_CAPTURE_KEY: 1}))])}
    for name, msg in r.items():
        ck(3, f'pricing key True refused: {name}', msg != 'not refused' and H.S56_PRICING_CAPTURE_KEY in msg, msg[:200])
    sp = freeze([('a', None, ev_opts), ('t', None, dict(ev_opts, **{H.S56_PRICING_CAPTURE_KEY: True}))])
    spf = freeze([('f', None, dict(ev_opts, **{H.S56_PRICING_CAPTURE_KEY: False}))])
    by = {e['label']: e for e in sp['candidates']}
    ck(3, 'pricing key True: keyed (a distinct eval key), recorded in the entry; False: recorded, not keyed',
       by['t']['eval_key'] != by['a']['eval_key'] and by['t'][H.S56_PRICING_CAPTURE_KEY] is True
       and H.S56_PRICING_CAPTURE_KEY not in by['a'] and spf['candidates'][0]['eval_key'] == by['a']['eval_key']
       and spf['candidates'][0][H.S56_PRICING_CAPTURE_KEY] is False)
    ck(3, 'record fields of the pricing key empty when absent', H.s56_pricing_record_fields({'label': 'x'}, {}) == {})
    return out


# ======================================================================================================================
#  T4 option-off identity
# ======================================================================================================================
def full_key(mod, spec, entry):
    """The full eval key `freeze_campaign_spec` of module `mod` gives a committed entry (every rule)."""
    EVT = _evt()
    ek = EVT._eval_key_from_spec(mod, spec, entry)
    if any(k in entry for k in mod.S56_EV_PLAN_KEYS):
        ek = mod.s56_evaluation_key(ek, mod.s56_ev_plan_in_key(mod.validate_s56_ev_plan(entry)))
        ek = mod.s56_settled_stop_evaluation_key(ek, entry.get(mod.S56_SETTLED_STOP_DISABLED_KEY)) \
            if hasattr(mod, 's56_settled_stop_evaluation_key') else ek
        if hasattr(mod, 's56_pricing_capture_evaluation_key'):
            ek = mod.s56_pricing_capture_evaluation_key(ek, entry.get(mod.S56_PRICING_CAPTURE_KEY))
            ek = mod.s56_cycle_cap_evaluation_key(ek, int(spec['cap']))
    elif hasattr(mod, 's56_settled_stop_evaluation_key'):
        ek = mod.s56_settled_stop_evaluation_key(ek, entry.get(mod.S56_SETTLED_STOP_DISABLED_KEY))
    return ek


def t4(scratch):
    EVT = _evt()
    diff = git('diff', '-U0', PRE_REV, '--', 'p515_s44_campaign_harness.py') or ''
    deleted = [ln for ln in diff.splitlines() if ln.startswith('-') and not ln.startswith('---')]
    added = [ln for ln in diff.splitlines() if ln.startswith('+') and not ln.startswith('+++')]
    ck(4, f'harness vs {PRE_REV}: insertions only ({len(added)} added, 0 deleted)', not deleted and added, deleted[:5])
    pre, _pp = EVT._load_module_at(PRE_REV, 'p515_s44_campaign_harness.py', scratch, 'harness_pre_integration')
    files = sorted(f for f in (git('ls-files') or '').splitlines() if os.path.basename(f).startswith('campaign_spec_')
                   and f.endswith('.json'))
    n_entries = n_s56 = n_rec = 0
    cand_bad, pre_post_bad, rec_bad_post, rec_bad_pre, s56_entries, errors = [], [], [], [], [], []
    for f in files:
        spec = D.load_json(os.path.join(CODE_ROOT, f))
        for e in spec.get('candidates') or []:
            n_entries += 1
            if e.get('canonical') is not None and 'key' in e and H.candidate_key(e['canonical']) != e['key']:
                cand_bad.append([f, e.get('label')])
            s56 = any(k in e for k in H.S56_EV_PLAN_KEYS)
            try:
                kp, kq = full_key(pre, spec, e), full_key(H, spec, e)
            except Exception as error:  # noqa: BLE001 -- recorded (an old spec shape)
                errors.append([f, e.get('label'), f'{type(error).__name__}: {error}'[:200]])
                continue
            if s56:
                n_s56 += 1
                s56_entries.append({'spec': f, 'label': e.get('label'), 'recorded_eq_pre': kp == e.get('eval_key'),
                                    'post_ne_pre': kq != kp})
                continue
            if kp != kq:
                pre_post_bad.append([f, e.get('label')])
            if 'eval_key' in e:
                n_rec += 1
                if kq != e['eval_key']:
                    rec_bad_post.append((f, e.get('label')))
                if kp != e['eval_key']:
                    rec_bad_pre.append((f, e.get('label')))
    res = {'n_spec_files': len(files), 'n_entries': n_entries, 'n_entries_with_s56_keys': n_s56,
           'n_eval_keys_recorded_non_s56': n_rec, 'n_post_ne_recorded': len(rec_bad_post),
           'post_ne_recorded_first': rec_bad_post[:20], 'n_recompute_errors': len(errors), 'errors': errors[:20],
           's56_entries': s56_entries}
    ck(4, f'committed specs: candidate keys of every entry reproduce ({len(files)} spec files, {n_entries} entries)',
       not cand_bad and n_entries > 0, cand_bad[:5])
    ck(4, 'entries WITHOUT an S56 EV-plan key: full eval key post == pre on every entry', not pre_post_bad,
       pre_post_bad[:5])
    ck(4, 'entries WITHOUT an S56 EV-plan key: post == recorded wherever pre == recorded (same set of historic '
          'non-reproducing shapes)', sorted(rec_bad_post) == sorted(rec_bad_pre), len(rec_bad_post))
    ck(4, 'entries WITH S56 EV-plan keys: every one changes under the cap rule (by design; listed with whether the '
          'pre harness reproduces its recorded key)', s56_entries and all(x['post_ne_pre'] for x in s56_entries),
       [x for x in s56_entries if not x['post_ne_pre']][:5])
    res['n_s56_entries_recorded_eq_pre'] = sum(1 for x in s56_entries if x['recorded_eq_pre'])
    # freeze pre vs post on entry sets without S56 keys
    x0, unit = {5: (0, 0), 7: (0, 0), 9: (0, 0)}, {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}
    derived_sp = EVT.srp1grid_sp_declaration()
    cfg_ev, required = EVT._spec_like(derived_sp, cap=3)
    sets = {
        'srp1_plain': ([('x0', x0, {}), ('unit', unit, {}), ('unit_2030', unit, {'investment_year': 2030}),
                        ('unit_flex2', unit, {'flex_price_multiplier': 2.0}), ('unit_rho', unit, {'rho0_scale': 2.0})],
                       {'overrides': {}}),
        'srp1_pinned_aa': ([('unit_pinned', unit, {'overrides': {'anderson_acceleration': {'enabled': False}},
                                                   H.S56_CYCLE_CAPTURE_PINNED_KEY: True})], {'overrides': {}}),
        'sp_derived_coupled_calib': ([('unit_v1', unit, {'interface_coupling': dict(EVT.COUPLING_V1_NOQ),
                                                         H.S56_CERTIFICATION_STOP_DISABLED_KEY: True})], cfg_ev),
        'sp_derived_plain': ([('x0', x0, {}), ('unit', unit, {})], cfg_ev),
    }
    for name, (cands, cfg) in sets.items():
        try:
            specs = {}
            for tag, mod in (('pre', pre), ('post', H)):
                p, _s, _sp = mod.freeze_campaign_spec(os.path.join(scratch, f'id_{name}_{tag}'), f'id_{name}',
                                                      copy.deepcopy(cands), dict(cfg, name=f'id {name}'), 3, 1,
                                                      ['oracle tests T4'], required_consecutive_cycles=10)
                s = D.load_json(p)
                specs[tag] = P.canonical_json({k: v for k, v in s.items() if k not in ('frozen_utc', 'harness',
                                                                                       'git_head')})
            same, detail = specs['pre'] == specs['post'], None
        except Exception as error:  # noqa: BLE001
            same, detail = False, f'{type(error).__name__}: {error}'
        ck(4, f'freeze_campaign_spec pre == post byte-identical (volatile fields excluded): {name}', same, detail)
    # build_evaluation_record pre == post on committed artefacts
    recs = sorted(f for f in (git('ls-files') or '').splitlines() if f.endswith('/evaluation_record.json'))
    c5 = []
    for f in recs:
        record = D.load_json(os.path.join(CODE_ROOT, f))
        rp, sp = record.get('report_path'), record.get('campaign_spec_path')
        if not rp or not sp or not os.path.isfile(rp) or not os.path.isfile(sp) or record.get('status') == 'error':
            continue
        spec = D.load_json(sp)
        entry = next((e for e in spec['candidates'] if e['label'] == record['candidate_label']), None)
        if entry is None:
            continue
        kw = dict(spec=spec, spec_path=os.path.join(CODE_ROOT, sp), spec_sha256=record['campaign_spec_sha256'],
                  entry=entry, report=D.load_json(rp), component_levels=None, floor_terminal=None,
                  published_caps=None, peak_rss=record.get('peak_rss'), wall=record.get('wall_time_s'),
                  eval_dir=os.path.join(CODE_ROOT, os.path.dirname(f)))
        a = json.dumps(pre.build_evaluation_record(**kw), sort_keys=True, default=str)
        b = json.dumps(H.build_evaluation_record(**kw), sort_keys=True, default=str)
        c5.append({'record': f, 'identical': a == b})
        if len(c5) >= 12:
            break
    ck(4, 'build_evaluation_record pre == post on committed records', len(c5) >= 4 and all(x['identical'] for x in c5),
       {'n': len(c5)})
    ck(4, 'the child record helpers add nothing when the keys are absent', H.s56_pricing_record_fields({}, {}) == {}
       and H.s56_settled_stop_record_fields({}, {}) == {})
    res['c5'] = c5
    res['diff_lines_added'] = len(added)
    return res


# ======================================================================================================================
#  T5 preflight refusals
# ======================================================================================================================
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages wired down:                         {wired}.
Anonymous pages:                          {anon}.
Pages occupied by compressor:             {comp}.
"""


def t5(lat, scratch):
    out = {}
    gib = 1 << 30
    pages = gib // 16384
    m_ok = D.memory_preflight(5, 3.0, VM_STAT.format(wired=4 * pages, anon=10 * pages, comp=2 * pages), 64 * gib)
    m_bad = D.memory_preflight(5, 3.0, VM_STAT.format(wired=4 * pages, anon=40 * pages, comp=6 * pages), 64 * gib)
    m_edge = D.memory_preflight(5, 3.0, VM_STAT.format(wired=0, anon=49 * pages, comp=0), 64 * gib)
    ck(5, 'memory rule: 64 - 16 = 48 GiB available >= 5 x 3 GiB passes; 64 - 50 = 14 < 15 refuses; 15 == 15 passes',
       m_ok['pass'] and not m_bad['pass'] and m_edge['pass'], [m_ok['available_gib'], m_bad['available_gib'],
                                                               m_edge['available_gib']])
    ck(5, 'memory rule: an undeclared per-slot figure refuses', not D.memory_preflight(
        1, None, VM_STAT.format(wired=1, anon=1, comp=1), 64 * gib)['pass'])
    ck(5, 'memory rule: a vm_stat without the compressor line refuses', not D.memory_preflight(
        1, 1.0, 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages wired down: 1.\n', 64 * gib)['pass'])
    mp = D.MAPPING_PARAMETERS
    ck(5, 'ruling 1: the mapping parameters are FINAL (provisional empty; tau_cl 1e-5 p.u., ESSO slack gate 1e-5)',
       mp['provisional'] == [] and mp['tau_cl_pu'] == 1e-5 and mp['esso_slack_tol'] == 1e-5, mp)
    try:
        D.real_launch_preflight(H, os.path.join(scratch, 'never_created'), 1, 3.0, CODE_ROOT,
                                mapping=dict(D.MAPPING_PARAMETERS, provisional=['tau_cl_pu']))
        msg = 'not refused'
    except D.Refused as error:
        msg = error.reasons
    out['real_launch_preflight_refusals'] = msg
    ck(5, 'real-launch preflight refuses a provisional mapping (one declared provisional), and names it',
       msg != 'not refused' and any('provisional' in m for m in msg), msg)
    env_saved = os.environ.pop('NLP_SOLVER_PATH', None)
    try:
        try:
            D.real_launch_preflight(H, os.path.join(scratch, 'never_created'), 1, 3.0, CODE_ROOT,
                                    mapping=dict(D.MAPPING_PARAMETERS, provisional=[]))
            msg2 = 'not refused'
        except D.Refused as error:
            msg2 = error.reasons
    finally:
        if env_saved is not None:
            os.environ['NLP_SOLVER_PATH'] = env_saved
    ck(5, 'real-launch preflight refuses without NLP_SOLVER_PATH = /usr/local/bin/ipopt (final mapping)',
       msg2 != 'not refused' and any('NLP_SOLVER_PATH' in m for m in msg2), msg2)
    out['real_launch_preflight_refusals_final_mapping'] = msg2
    probe_alive = 'p515_s44_campaign_harness.py --child' in (subprocess.run(
        ['ps', '-axo', 'command'], capture_output=True, text=True).stdout)
    out['live_campaign_child_at_test_time'] = probe_alive
    if probe_alive:
        ck(5, 'real-launch preflight names a live campaign child (the memory probe) as a harness precondition failure',
           any('forbidden process' in m for m in msg2), msg2)
    ck(5, 'real-launch preflight never created the launch dir', not os.path.exists(os.path.join(scratch,
                                                                                                'never_created')))
    hc = harness_config()
    inst = D.build_lattice(CODE_ROOT)[1]
    common = dict(search_id='t5', arm='ev_2x', lattice=lat, instance_rec=inst, start_name='x0', n_max=10, k_ev=400,
                  oracle='harness', class_rank=P.CLASS_RANK, test_only=False, data_root=CODE_ROOT)
    for name, kw, needle in (('provisional mapping', {'harness_config': hc,
                                                      'mapping': dict(D.MAPPING_PARAMETERS, provisional=['tau_cl_pu'])},
                              'provisional'),
                             ('no memory_per_slot_gib', {'harness_config': {k: v for k, v in hc.items()
                                                                           if k != 'memory_per_slot_gib'},
                                                         'mapping': dict(D.MAPPING_PARAMETERS, provisional=[])},
                              'memory_per_slot_gib')):
        try:
            D.build_spec(**common, **kw)
            m = 'not refused'
        except D.Refused as error:
            m = '; '.join(error.reasons)
        ck(5, f'real search freeze refused: {name}', needle in m, m)
    bad = copy.deepcopy(hc)
    bad['by_depth']['settled']['entry_options'].pop(H.S56_PRICING_CAPTURE_KEY)
    try:
        oracle_for(lat, bad, kind='harness')
        m = 'not refused'
    except D.Refused as error:
        m = '; '.join(error.reasons)
    ck(5, 'real oracle refused when the settled depth does not declare the pricing capture (mapping capture-path '
          'assertion, before any run)', 'settled_declares_pricing_capture' in m, m)
    orc = oracle_for(lat, hc, kind='harness')
    out['mapping_capture_checklist'] = orc.mapping_checklist
    ck(5, 'mapping capture-path assertion passes on the integration harness with the EV configuration',
       all(orc.mapping_checklist.values()), orc.mapping_checklist)
    good = {'predictions': {k: {'statement': f'test {k}', 'evaluated_as': 'test', 'on_failure': 'test'}
                            for k in D.PILOT_PREDICTION_IDS}}
    incomplete = {'predictions': {k: v for k, v in good['predictions'].items() if k != 'PP8'}}
    for name, kw, needle in (('no predictions', {}, 'predictions'),
                             ('predictions without PP8', {'predictions': incomplete}, 'PP8'),
                             ('memory_per_slot_gib 3.0 (ruling 7: 4.0)', {'predictions': good,
                                                                         'memory_per_slot_gib': 3.0}, 'ruling 7'),
                             ('initialisations other than the ruled', {'predictions': good,
                                                                      'init_variants': (('i1', 1.0), ('i2', 0.5),
                                                                                        ('i3', 3.0))}, 'ruled')):
        try:
            D.pilot_spec(pilot_id='t5', grid='srp1grid', oracle='harness', test_only=False, data_root=CODE_ROOT, **kw)
            m = 'not refused'
        except D.Refused as error:
            m = '; '.join(error.reasons)
        ck(5, f'real pilot freeze refused: {name}', needle in m, m)
    sp = D.pilot_spec(pilot_id='t5', grid='srp1grid', oracle='harness', test_only=False, data_root=CODE_ROOT,
                      predictions=good)
    out['real_pilot_spec_in_memory'] = {k: sp[k] for k in ('memory_per_slot_gib', 'rules', 'init_variants',
                                                           'staged_plan_label', 'capture_path_checklist')}
    ck(5, 'real pilot spec (in memory, not frozen): rulings 2, 3, 7 -- s_share 500, rho0_scale 1.0 / 0.5 / 2.0 FINAL, '
          'the ruled staged plan, 4.0 GiB per slot, mapping final, pilot capture-path checklist all true',
       sp['rules']['s_share_eur'] == 500.0 and sp['memory_per_slot_gib'] == 4.0
       and sp['init_variants']['values'] == [['i1', 1.0], ['i2', 0.5], ['i3', 2.0]]
       and sp['init_variants']['status'].startswith('FINAL')
       and sp['staged_plan_label'] == 'n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5'
       and sp['mapping']['provisional'] == [] and all(sp['capture_path_checklist'].values())
       and sp['predictions_recorded_before_run'] == good
       and [e['rho0_scale'] for e in sp['entries']] == [1.0, 1.0, 1.0, 0.5, 0.5, 2.0, 2.0]
       and all(('rho0_scale' in e['options']) == (e['rho0_scale'] != 1.0) for e in sp['entries']),
       out['real_pilot_spec_in_memory'])
    for name, kw in (('an unknown grid', {'grid': 'other', 'oracle': 'stub_children', 'test_only': True}),
                     ('a stub pilot not test-only', {'grid': 'srp1grid', 'oracle': 'stub_children',
                                                     'test_only': False})):
        try:
            D.pilot_spec(pilot_id='t5', data_root=CODE_ROOT, **kw)
            m = 'not refused'
        except D.Refused as error:
            m = '; '.join(error.reasons)
        ck(5, f'pilot freeze refused: {name}', m != 'not refused', m)
    return out


# ======================================================================================================================
#  T6 cache roots and the start pool
# ======================================================================================================================
def _fixture_eval(root, key, z, lat, F, status='certified', bar=4000.0):
    """A SYNTHETIC real-format eval dir: the smoke unit's files with the record's keys and F rewritten."""
    d = os.path.join(root, 'evals', f'{key[:16]}_fixture')
    shutil.copytree(os.path.join(CODE_ROOT, SMOKE_EVALS['unit']), d,
                    ignore=shutil.ignore_patterns('*.pkl', '*.xlsx', 'planning_read', 'esso_capture', 'results'))
    rec = D.load_json(os.path.join(d, 'evaluation_record.json'))
    rec.update(eval_key=key, candidate_key=lat.key(tuple(z)), status=status, synthetic_fixture=True)
    rec['certification'] = dict(rec['certification'], certification_cycle=2 if status == 'certified' else None)
    rec['s56_investment'] = dict(rec['s56_investment'], investment_cost_eur=0.0)
    rows = D._jsonl(os.path.join(d, 'per_cycle_record.jsonl'))
    rows[0].update(recourse=F - bar, gross_operational_cost=F - bar, terminal_salvage_value=0.0)
    rows[1].update(recourse=F, gross_operational_cost=F, terminal_salvage_value=0.0)
    os.remove(os.path.join(d, 'evaluation_record.json'))
    os.remove(os.path.join(d, 'per_cycle_record.jsonl'))
    with open(os.path.join(d, 'evaluation_record.json'), 'w') as h:
        json.dump(rec, h)
    with open(os.path.join(d, 'per_cycle_record.jsonl'), 'w') as h:
        h.write(''.join(json.dumps(r) + '\n' for r in rows))
    return d


def t6(lat, scratch):
    hc = harness_config()
    orc = oracle_for(lat, hc, kind='harness')
    domain = D.enumerate_domain(lat)
    cache = os.path.join(scratch, 'cache_root')
    pts = {'x0': D.named_start(lat, 'x0'), 'reference_unit': D.named_start(lat, 'reference_unit'),
           'plan_2030': D.named_start(lat, 'plan_2030')}
    Fs = {'x0': 873.90e6, 'reference_unit': 873.89e6, 'plan_2030': 873.85e6}
    for n, z in pts.items():
        _fixture_eval(cache, orc.eval_key(z, P.DEPTH_EXIT), z, lat, Fs[n])
    _fixture_eval(cache, 'f' * 64, pts['x0'], lat, 1.0)          # another configuration: ignored
    shutil.copytree(os.path.join(CODE_ROOT, SMOKE_EVALS['x0']), os.path.join(cache, 'evals', 'smoke_copy'),
                    ignore=shutil.ignore_patterns('*.pkl', '*.xlsx', 'planning_read', 'esso_capture', 'results'))
    idx = D.build_cache_index(orc, lat, CODE_ROOT, [cache], domain=domain)
    ents, ign = idx[0]['entries'], idx[0]['ignored']
    ck(6, 'cache index: the 3 fixture exit records of this configuration indexed at depth exit; the foreign-key '
          'fixture and the smoke copy (pre-cap-rule key) ignored with their reason',
       sorted(tuple(e['z']) for e in ents) == sorted(pts.values()) and all(e['depth'] == 'exit' for e in ents)
       and len(ign) == 2 and all('another configuration' in i['reason'] or 'dir name' in i['reason'] for i in ign),
       {'entries': len(ents), 'ignored': ign})
    spec = {'cache_sources': idx}
    recs = D.load_cache(spec, orc, CODE_ROOT)
    ck(6, 'load_cache maps every indexed record through the oracle mapping (exit F = the fixture F)',
       all(abs(recs[(tuple(z), 'exit')]['F'] - Fs[n]) <= 1e-6 for n, z in pts.items()),
       {n: recs[(tuple(z), 'exit')].get('F') for n, z in pts.items()})
    z, rep = D.best_start(lat, [{'name': n, 'z': list(z)} for n, z in pts.items()], recs)
    ck(6, 'best start = the certified pool member with the lowest exit F (plan_2030 here), runner-up margin and e '
          'reported', tuple(z) == tuple(pts['plan_2030']) and rep['runner_up']['name'] == 'reference_unit'
       and abs(rep['runner_up']['margin'] - 40000.0) < 1e-3, rep)
    try:
        D.best_start(lat, [{'name': 'x0', 'z': list(pts['x0'])},
                           {'name': 'other', 'z': list(lat.with_pair(lat.x0(), 9, 2035, 1, 1))}], recs)
        m = 'not refused'
    except D.Refused as error:
        m = '; '.join(error.reasons)
    ck(6, 'best start refuses a pool member without a cached exit record (names it)', 'other' in m, m)
    target = os.path.join(CODE_ROOT, ents[0]['eval_dir'], 'evaluation_record.json')
    rec = D.load_json(target)
    rec['tampered'] = True
    with open(target, 'w') as h:
        json.dump(rec, h)
    try:
        D.load_cache(spec, orc, CODE_ROOT)
        m = 'not refused'
    except D.Refused as error:
        m = '; '.join(error.reasons)
    ck(6, 'a record changed after the freeze refuses the launch', 'changed since the freeze' in m, m)
    return {'n_indexed': len(ents), 'ignored': ign, 'best_start': rep,
            'note': 'SYNTHETIC fixture records (smoke files with rewritten keys and F)'}


# ======================================================================================================================
#  T7 the pilot against stubs
# ======================================================================================================================
def drv(argv, logp, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    with open(logp + '_stdout.log', 'w') as so, open(logp + '_stderr.log', 'w') as se:
        r = subprocess.run([sys.executable, '-u', os.path.join(CODE_ROOT, 'p515_s56_search_driver.py'), *argv],
                           cwd=CODE_ROOT, stdout=so, stderr=se, env=env)
    return r.returncode


def t7(scratch):
    out = {}
    root = os.path.join(scratch, 'pilot_stub')
    lock = os.path.join(scratch, 'pilot_scratch.lock')
    logs = os.path.join(scratch, 'pilot_logs')
    os.makedirs(logs)
    rc = drv(['--pilot-freeze', '--root', root, '--pilot-id', 'pilotstub', '--grid', 'srp1grid', '--oracle',
              'stub_children', '--test-only'], os.path.join(logs, 'freeze'))
    _p, sha, spec = D.load_pilot_spec(root) if rc == 0 else (None, None, None)
    ck(7, 'pilot freeze (stub, test-only, SRP1 grid): 7 entries, blocks of 5 + 2, distinct eval keys, every entry '
          'stop-disabled with the pricing capture at K_pilot 300',
       rc == 0 and len(spec['entries']) == 7 and [len(b) for b in spec['blocks']] == [5, 2]
       and len({e['eval_key'] for e in spec['entries']}) == 7 and spec['k_pilot'] == 300
       and all(e['options'][H.S56_SETTLED_STOP_DISABLED_KEY] is True and e['options'][H.S56_PRICING_CAPTURE_KEY]
               is True for e in spec['entries']), rc)
    roles = sorted((e['role'], e['init']) for e in spec['entries'])
    ck(7, 'x0 and the unit from three initialisations (rho0_scale 1.0 / 0.5 / 2.0, FINAL by ruling 3), one staged plan',
       roles == sorted([('x0', 'i1'), ('x0', 'i2'), ('x0', 'i3'), ('unit', 'i1'), ('unit', 'i2'), ('unit', 'i3'),
                        ('staged', 'i1')]) and spec['init_variants']['status'].startswith('FINAL'), roles)
    codes = []
    for i in range(3):
        codes.append(drv(['--pilot-run', '--root', root, '--test-only', '--lock-path', lock],
                         os.path.join(logs, f'run{i + 1}'), env_extra={D.STUB_DELAY_ENV: f'0.2:0.6:pilot{i}'}))
    ck(7, 'three attached pilot launches: block 1 (5) -> exit 4, block 2 (2) -> exit 0, then done -> exit 0',
       codes == [4, 0, 0], codes)
    lr = [D.load_json(D.launch_paths(root, n)[1]) for n in (1, 2)]
    ck(7, 'launch records: the frozen blocks with their frozen eval keys, hash-chained, every child exit 0',
       [r['block'] for r in lr] == spec['blocks'] and lr[1]['previous_launch_manifest_sha256'] == D.sha256_file(
           D.launch_paths(root, 1)[2]) and all(x['exit_code'] == '0' for r in lr for x in r['records']))
    res_dir = os.path.join(scratch, 'pilot_eval')
    rc_other = drv(['--pilot-evaluate', '--root', root, '--out', res_dir + '_othershare', '--s-share-eur', '400'],
                   os.path.join(logs, 'eval_othershare'))
    ck(7, 'pilot evaluation with an s_share other than the frozen 500 refuses (ruling 2), writing nothing',
       rc_other == 2 and not os.path.exists(res_dir + '_othershare'), rc_other)
    rc_ev = drv(['--pilot-evaluate', '--root', root, '--out', res_dir], os.path.join(logs, 'eval'))
    res = D.load_json(os.path.join(res_dir, 'pilot_result.json')) if rc_ev in (0, 5) else {}
    out['result'] = res
    need = ('K_EV', 'e_EV', 'e_EV_terms', 'pairs', 'dF_exit_spread_3x3', 'dF_set_spread_3x3', 'max_pair_offset',
            'max_bar_sum', 't_exit_h', 't_set_h', 'psi', 'psi_per_block', 'runs')
    ck(7, 'pilot evaluation (SYNTHETIC traces; s_share 500 read from the frozen spec) writes every output: K_EV, e_EV '
          'and its terms, 15 pairs, spreads, bars, t_exit, t_set, psi, per-run single offsets and the leak at the '
          'settled point',
       rc_ev == 0 and res.get('s_share_eur') == 500.0 and all(res.get(k) is not None for k in need)
       and len(res['pairs']) == 15 and res['synthetic']
       and all('single_offset' in r and 'leak_settled_point' in r for r in res['runs'].values()),
       {k: res.get(k) for k in ('K_EV', 'e_EV', 'failures')})
    if res.get('K_EV') is not None:
        e_rule = max(2 * max(p['offset'] for p in res['pairs']),
                     max(p['dF_exit'] for p in res['pairs'] if p['kind'] == 'unit-x0')
                     - min(p['dF_exit'] for p in res['pairs'] if p['kind'] == 'unit-x0'),
                     max(p['bar_sum'] for p in res['pairs']))
        ck(7, 'e_EV re-derived from the reported pairs by the frozen rule', abs(e_rule - res['e_EV']) <= 1e-6,
           [e_rule, res['e_EV']])
        ok_k = all(r['S_at_K_EV'] <= 500 for r in res['runs'].values()) and res['K_EV'] >= max(
            r['c_star'] for r in res['runs'].values())
        ck(7, 'K_EV satisfies its rule on the reported values (S <= s_share at K_EV, K_EV >= every c*)', ok_k)
    # direct: pilot_outputs on traces with no exit, and with an unreachable share
    tr = [D.synthetic_trace(f'r{i}', role, 'i1', 'x', 300) for i, role in enumerate(('x0', 'unit', 'staged'))]
    r_small = D.pilot_outputs(tr, 1e-9)
    ck(7, 'an unreachable s_share -> K_EV None with the failure named (a stop, not a value)',
       r_small['K_EV'] is None and r_small['failures'], r_small['failures'])
    tr2 = copy.deepcopy(tr)
    tr2[0]['consec'] = {k: 0 for k in tr2[0]['consec']}
    r_none = D.pilot_outputs(tr2, 500.0)
    ck(7, 'a run without a would-be certification cycle -> failure named, no e_EV', 'e_EV' not in r_none and
       r_none['failures'], r_none['failures'])
    rc5 = drv(['--pilot-freeze', '--root', os.path.join(scratch, 'pilot_five'), '--pilot-id', 'p5', '--grid',
               'five_block', '--oracle', 'stub_children', '--test-only'], os.path.join(logs, 'freeze_five'))
    sp5 = D.load_pilot_spec(os.path.join(scratch, 'pilot_five'))[2] if rc5 == 0 else {}
    ck(7, 'pilot freeze on the five-block grid (the grid is a parameter): 7 entries, keys distinct from the SRP1 grid',
       rc5 == 0 and len(sp5['entries']) == 7 and not ({e['eval_key'] for e in sp5['entries']}
                                                     & {e['eval_key'] for e in spec['entries']}), rc5)
    out['launch_codes'] = codes
    out['preflight_only'] = t7_preflight_only(scratch, root, logs)
    return out


def t7_preflight_only(scratch, stub_root, logs):
    """--pilot-run --preflight-only: refused on a stub pilot; on a test-only HARNESS-oracle pilot (scratch lock): zero
    solves (the driver's armed guard), the record written, nothing launched, the pilot root unchanged, the lock
    released. Whether the preflight itself passes depends on the machine's state (memory, live processes), so both
    outcomes are accepted here provided the record names every refusal."""
    out = {}
    lock = os.path.join(scratch, 'pilot_pf.lock')
    rc_stub = drv(['--pilot-run', '--root', stub_root, '--preflight-only', '--out', os.path.join(scratch, 'pf_stub'),
                   '--lock-path', lock], os.path.join(logs, 'pf_stub'))
    ck(7, '--preflight-only on a stub pilot is refused (exit 2), nothing written',
       rc_stub == 2 and not os.path.exists(os.path.join(scratch, 'pf_stub')), rc_stub)
    root = os.path.join(scratch, 'pilot_harness_testonly')
    rc_f = drv(['--pilot-freeze', '--root', root, '--pilot-id', 'pilotpf', '--grid', 'srp1grid', '--oracle', 'harness',
                '--test-only'], os.path.join(logs, 'pf_freeze'))
    before = sorted(D.files_under(root)) if rc_f == 0 else None
    pf_out = os.path.join(scratch, 'pf_harness')
    rc = drv(['--pilot-run', '--root', root, '--preflight-only', '--out', pf_out, '--lock-path', lock],
             os.path.join(logs, 'pf_harness'))
    rec = D.load_json(os.path.join(pf_out, 'pilot_preflight.json')) if os.path.isfile(
        os.path.join(pf_out, 'pilot_preflight.json')) else {}
    after = sorted(D.files_under(root)) if rc_f == 0 else None
    out.update({'freeze_exit': rc_f, 'exit': rc, 'record': rec})
    ok = (rc_f == 0 and rc in (0, 2) and rec.get('launched') is False and rec.get('solves') == 0
          and rec.get('launch_dir_created') is False and before == after and not os.path.exists(lock)
          and rec.get('next_launch_index') == 1 and rec.get('pass') is (rc == 0)
          and (rc == 0 or rec['real_launch_preflight'].get('refusals')))
    ck(7, '--preflight-only on a test-only harness pilot: record written (pass or every refusal named), nothing '
          'launched, no launch dir, the pilot root unchanged, the lock released',
       ok, {'freeze': rc_f, 'exit': rc, 'pass': rec.get('pass'),
            'refusals': (rec.get('real_launch_preflight') or {}).get('refusals')})
    rc_again = drv(['--pilot-run', '--root', root, '--preflight-only', '--out', pf_out, '--lock-path', lock],
                   os.path.join(logs, 'pf_harness_again'))
    ck(7, '--preflight-only refuses an existing --out (write-once)', rc_again == 2, rc_again)
    return out


# ======================================================================================================================
#  T8 projection
# ======================================================================================================================
def t8(lat):
    cp = D.count_projection([{'n': 18, 'p': 7, 'c': 2}, {'n': 20, 'p': 1, 'c': 1}], {'n': 15, 'c': 3})
    # E_1 = min(18, 5 ceil(7/5) = 10) = 10; E_2 = min(20, 5) = 5; N = 1 + (10 + 2) + (5 + 1) + 15 + 3 = 37;
    # blocks = 1 + 2 + 1 + 3 = 7; confirmation batches = 6
    ck(8, 'count formula on a worked example: N_search 37, blocks 7, confirmation batches 6',
       (cp['N_search'], cp['blocks'], cp['confirmation_batches']) == (37, 7, 6), cp)
    pw = D.project_wall_time(1.5, 3.0, 1.2, 7, 6)
    ck(8, 'T = 1.18 (1.2 x 1.5 x 7 + 3.0 x 6) = 36.108 h', abs(pw['T_h'] - 1.18 * (1.2 * 1.5 * 7 + 18.0)) < 1e-12, pw)
    try:
        D.project_wall_time(None, 3.0, 1.2, 7, 6)
        m = 'not refused'
    except D.Refused as error:
        m = error.reasons
    ck(8, 'the projection refuses an unmeasured t_exit (the pilot measures it)', m != 'not refused', m)
    env = D.LatticeEnv(lat)
    land = D.SyntheticLandscape('t8', D.named_start(lat, 'plan_2030'), D.named_start(lat, 'x0'))
    st, _r = P.run_to_end(P.initial_state({'start': list(D.named_start(lat, 'x0')), 'n_max': 200, 'arm': 't8',
                                           'domain_size': D.DOMAIN_SIZE}), {}, env,
                          lambda a: {(tuple(z), a['depth']): land.summary(z, a['depth']) for z in a['zs']})
    pj = D.projection_from_state(st)
    n_actions = {}
    for e in st['evaluated']:
        n_actions[(e['action_index'], e['depth'])] = 1
    ck(8, 'projection_from_state counts blocks and confirmation batches as the synthetic search issued them',
       pj['blocks'] + pj['confirmation_batches'] == len(n_actions) and pj['evaluations'] == st['budget_used'], pj)
    return {'count_example': cp, 'projection_example': pw, 'synthetic_state_projection': pj}


# ======================================================================================================================
#  T9 a stub-children search at 2x through the driver CLI (the configuration the alias used to refuse)
# ======================================================================================================================
def t9(lat, scratch):
    root, lock, logs = os.path.join(scratch, 'search_2x'), os.path.join(scratch, 's9.lock'), os.path.join(scratch, 'logs')
    os.makedirs(logs)
    hcp = os.path.join(scratch, 'harness_config_2x.json')
    D.write_once_json(hcp, harness_config())
    rc0 = drv(['--freeze', '--root', root, '--search-id', 't9stub', '--arm', 'ev_2x', '--start', 'x0', '--n-max', '12',
               '--k-ev', '400', '--oracle', 'stub_children', '--synthetic-seed', 't9', '--harness-config', hcp,
               '--test-only'], os.path.join(logs, 'freeze'))
    ck(9, 'the 2x EV stub search freezes (its collision scan, refused before the cap rule, passes)', rc0 == 0, rc0)
    codes = []
    while rc0 == 0 and len(codes) < 40:
        codes.append(drv(['--run', '--root', root, '--lock-path', lock], os.path.join(logs, f'run{len(codes) + 1:02d}'),
                         env_extra={D.STUB_DELAY_ENV: '0.2:0.5:t9'}))
        if codes[-1] != 4:
            break
    spec = D.load_spec(root)[2] if rc0 == 0 else None
    ok = rc0 == 0 and codes and codes[-1] in (0, 5)
    detail = codes
    if ok:
        land = D.synthetic_landscape_of(spec, lat)
        st, _r = P.run_to_end(P.initial_state(D.policy_config(spec)), {}, D.LatticeEnv(lat),
                              lambda a: {(tuple(z), a['depth']): land.summary(z, a['depth']) for z in a['zs']})
        res = D.load_json(os.path.join(root, D.RESULTS_FILE))
        ok = P.canonical_json(res['state']) == P.canonical_json(st)
        detail = {'codes': codes, 'budget': st['budget_used'], 'termination': st['termination']}
    ck(9, 'the stub search terminates through attached launches and its final state == the in-process synthetic '
          'replay (the launch path after the integration edits)', ok, detail)
    return {'codes': codes}


# ======================================================================================================================
#  --real-check
# ======================================================================================================================
def real_check(out_dir):
    res = {'schema': 'p515_s56_search_oracle_real_check_v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
           'declared': {'entry': 'reference unit (n7 2025 0.25 MVA / 1 MWh)', 'grid': 'SRP1-grid EV twin', 'flex': 2.0,
                        'cap': REAL_CAP, 'keys': [H.S56_SETTLED_STOP_DISABLED_KEY, H.S56_PRICING_CAPTURE_KEY],
                        'solves': REAL_DECLARED_SOLVES,
                        'solves_formula': '((1 + n_dso) n_years n_days + n_esso_nodes) x (cycles + 1) = '
                                          '((1 + 3) x 3 x 4 + 3) x 4',
                        'expectation': 'complete summary, status not_settled (3 cycles cannot reach tail depth / '
                                       'certification)'},
           'error': None}
    lat, _inst = D.build_lattice(CODE_ROOT)
    root = os.path.join(out_dir, 'campaign_s56orc_v1_real_check')
    try:
        refusals = []
        if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
            refusals.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
        for rel in (SCRIPT, 'p515_s44_campaign_harness.py', 'p515_s56_search_driver.py', 'p515_s56_ev_harness_tests.py',
                    'p515_s56_ev_injection.py', 'p516_lattice.py', 'p515_s56_cohort_diagnostics.py'):
            if not (git('ls-files', '--', rel) or '').strip() or (git('status', '--porcelain', '--', rel) or '').strip():
                refusals.append(f'{rel} not committed clean')
        mem = D.memory_preflight(1, 3.0)
        res['memory_preflight'] = mem
        if not mem.get('pass'):
            refusals.append(f'memory: {mem}')
        pre = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT, 'p515_s56_search_driver.py'))
        res['harness_preconditions'] = pre
        refusals += pre
        if refusals:
            raise RuntimeError(f'refused before launch (zero solves): {refusals}')
        hc = harness_config(exit_cap=REAL_CAP, k=REAL_CAP, conc=1)
        orc = oracle_for(lat, hc, kind='harness')
        z = D.named_start(lat, 'reference_unit')
        bd = hc['by_depth']['settled']
        opts = orc.entry_options(z, P.DEPTH_SETTLED)
        spec_path, csha, cspec = H.freeze_campaign_spec(
            root, 's56orc_v1_real_check', [('unit_settled_k3', None, opts)],
            dict(bd['configuration'], name='S56 search oracle integration: real 3-cycle settled-key check, reference '
                                           'unit, SRP1-grid EV twin, 2x, cap 3'), REAL_CAP, 1,
            ['Planner integration task 2026-11-23 (S56 search driver), item 5'],
            required_consecutive_cycles=hc['required_consecutive_cycles'],
            extra={'campaign_script': SCRIPT, 'declared_solves_per_evaluation': REAL_DECLARED_SOLVES})
        e = cspec['candidates'][0]
        res['campaign'] = {'root': os.path.relpath(root, CODE_ROOT), 'spec': os.path.relpath(spec_path, CODE_ROOT),
                           'spec_sha256': csha, 'eval_key': e['eval_key'], 'candidate_key': e['key']}
        ck('R', 'the frozen eval key == the driver composition (settled depth, cap 3)',
           e['eval_key'] == orc.eval_key(z, P.DEPTH_SETTLED) and e['key'] == lat.key(z))
        lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
        log(f'real check: lock {lock}; spec {os.path.relpath(spec_path, CODE_ROOT)}')
        try:
            ctx = H.CampaignContext(root, spec_path, csha, cspec, log=log)
            records = H.evaluate(['unit_settled_k3'], ctx)
        finally:
            res['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
        rec = records[0] if records else {}
        ed = os.path.join(root, 'evals', e['eval_dir'])
        sp_ = rec.get('solve_profile') or {}
        obs = sp_.get('observed') or {}
        res['record_excerpt'] = {k: rec.get(k) for k in ('status', 'cycles_run', 'solve_profile', 'wall_time_s',
                                                         'peak_rss', 's56_certification_stop_summary',
                                                         's56_pricing_capture_summary', 'local_solve_failures')}
        ck('R', 'child exit 0, 3 cycles run (the stop disabled), record status not_certified',
           (rec.get('parent_view') or {}).get('exit_code') == 0 and rec.get('cycles_run') == REAL_CAP,
           {'exit': (rec.get('parent_view') or {}).get('exit_code'), 'cycles': rec.get('cycles_run'),
            'status': rec.get('status')})
        ck('R', f'solves == declared {REAL_DECLARED_SOLVES} exactly (identity holds, nothing blocked, no retry)',
           obs.get('permitted_solve') == REAL_DECLARED_SOLVES and obs.get('permitted_exec') == REAL_DECLARED_SOLVES
           and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0
           and sp_.get('base_solves') == REAL_DECLARED_SOLVES and sp_.get('retry_solves_credited') == 0
           and sp_.get('identity_holds') is True, sp_)
        ck('R', 'the settled key\'s child path ran: stop-disabling summary ok (length raised and restored), its '
                'checklist asserted before the run', (rec.get('s56_certification_stop_summary') or {}).get('ok') is True
           and rec.get('s56_certification_stop_checklist_asserted_before_run') is not None,
           rec.get('s56_certification_stop_summary'))
        ck('R', 'the pricing capture written, its checklist asserted before the run',
           (rec.get('s56_pricing_capture_summary') or {}).get('status') == 'written'
           and rec.get('s56_pricing_capture_checklist_asserted_before_run') is not None,
           rec.get('s56_pricing_capture_summary'))
        summ, _src = orc.summary_from_dir(ed, z, P.DEPTH_SETTLED)
        res['settled_summary'] = summ
        complete = (summ.get('status') in (D.SETTLED_STATUS, D.NOT_SETTLED_STATUS) and D._fin(summ.get('F'))
                    and isinstance(summ.get('V'), dict) and all(D._fin(summ['V'].get(k)) for k in 'S PR G A'.split())
                    and isinstance(summ.get('gates_pass'), bool))
        ck('R', 'the mapping gives a COMPLETE settled summary (F, S, PR, G, A finite; gates_pass a bool)', complete,
           summ.get('reason') or summ.get('status'))
        ck('R', 'flagged unsettled (3 cycles: not at tail depth / not certified on the last W)',
           summ.get('status') == D.NOT_SETTLED_STATUS, summ.get('not_settled_reasons'))
        ck('R', 'F at K = the cost breakdown total (same convention)', summ.get('F_equals_breakdown_total') is True,
           summ.get('F'))
        ck('R', 'settlement derivation holds on the real record: deviation part 0, contracted == total, blocks '
                'reconcile with the record', all((summ.get('settlement_check') or {}).get(k) for k in (
                    'deviation_is_zero', 'contracted_equals_total', 'blocks_reconcile_with_record')),
           summ.get('settlement_check'))
        ex = D.map_record(D.read_eval_files(ed), P.DEPTH_EXIT)
        res['exit_view'] = ex
        ck('R', 'exit view of the stop-disabled run: no would-be certification cycle in 3 cycles -> not_certified, '
                'F and bar read at the terminal cycle', ex['status'] == 'not_certified' and D._fin(ex['F'])
           and ex['read_cycle'] == REAL_CAP and ex['read_at'].startswith('terminal cycle'), ex['status'])
    except Exception as error:  # noqa: BLE001 -- recorded (a harness fault)
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
        ck('R', 'real check ran without a harness fault', False, res['error'])
    return res


# ======================================================================================================================
def finish(out_dir, name, payload):
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == [] in this parent', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s44_campaign_harness.py',
        'p515_s56_neighbourhood.py', 'p516_lattice.py', 'p513_solve_profile_guard.py')}
    payload.update({'finished_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable,
                    'git_head': (git('rev-parse', 'HEAD') or '').strip(), 'code_sha256': code,
                    'guards': {'counts': dict(GUARD.counts), 'verify_0': gf, 'pickle': dict(PICKLE_COUNTS)},
                    'checks': CHECKS, 'failed_checks': [f"[{c['item']}] {c['check']}" for c in CHECKS if not c['ok']],
                    'all_checks_ok': all(c['ok'] for c in CHECKS)})
    D.write_once_json(os.path.join(out_dir, f'{name}.json'), payload)
    log(f'{sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} checks ok; failed: {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--zero-solve', action='store_true')
    mode.add_argument('--real-check', action='store_true')
    mode.add_argument('--manifest', action='store_true')
    ap.add_argument('--name', default=None, help='output subdirectory (new)')
    ap.add_argument('--items', default='1,2,3,4,5,6,7,8,9')
    ap.add_argument('--out-root', default=None, help='rehearsals only: an output root outside the repository')
    args = ap.parse_args()
    global OUT
    if args.out_root:
        OUT = os.path.abspath(args.out_root)
    if args.manifest:
        files = [f for f in D.files_under(OUT) if os.path.basename(f) != 'manifest_sha256.json']
        man = {'files': {os.path.relpath(f, OUT): D.sha256_file(f) for f in files},
               'code_sha256': {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
                   SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s44_campaign_harness.py')},
               'git_head': (git('rev-parse', 'HEAD') or '').strip(), 'utc': datetime.now(timezone.utc).isoformat()}
        D.write_once_json(os.path.join(OUT, 'manifest_sha256.json'), man)
        log(f'manifest: {len(files)} files')
        return 0
    if args.real_check:
        out_dir = new_dir(args.name or 'real_check')
        return finish(out_dir, 'real_check', real_check(out_dir))
    out_dir = new_dir(args.name or 'zero_solve')
    scratch = os.path.join(out_dir, 'test_trees')
    os.makedirs(scratch)
    items = [s.strip() for s in args.items.split(',') if s.strip()]
    t0 = time.time()
    lat, inst = D.build_lattice(CODE_ROOT)
    results = {}
    for it, fn in (('1', lambda: t1()), ('2', lambda: t2()), ('3', lambda: t3(lat, os.path.join(scratch, 't3'))),
                   ('4', lambda: t4(os.path.join(scratch, 't4'))), ('5', lambda: t5(lat, os.path.join(scratch, 't5'))),
                   ('6', lambda: t6(lat, os.path.join(scratch, 't6'))), ('7', lambda: t7(os.path.join(scratch, 't7'))),
                   ('8', lambda: t8(lat)), ('9', lambda: t9(lat, os.path.join(scratch, 't9')))):
        if it not in items:
            continue
        os.makedirs(os.path.join(scratch, f't{it}'), exist_ok=True)
        t1_ = time.time()
        try:
            results[it] = fn()
        except Exception as error:  # noqa: BLE001 -- a test-script fault is a failed check, recorded
            results[it] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            traceback.print_exc()
            ck(it, 'item ran without an exception', False, results[it]['error'])
        results[it] = {'result': results[it], 'wall_s': time.time() - t1_}
    payload = {'schema': 'p515_s56_search_oracle_tests_v1', 'script': SCRIPT, 'items': items, 'instance': inst,
               'solve_claim': 'ZERO SOLVES, guard-verified; driver subprocesses arm their own guard; stub children '
                              'import no model', 'results': results, 'wall_s': time.time() - t0,
               'scratch_note': 'test_trees/ holds frozen campaign / pilot specs carrying EV-configuration eval keys: '
                               'hash-recorded in the manifest, NOT committed (a later collision scan would hit them)'}
    return finish(out_dir, 'zero_solve', payload)


if __name__ == '__main__':
    sys.exit(main())
