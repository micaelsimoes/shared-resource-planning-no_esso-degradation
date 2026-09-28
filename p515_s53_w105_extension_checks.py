"""
P5.15 Addendum 54 Ruling 1, Planner task W105 -- ZERO-SOLVE checks for the C* settling extension (spec v40).

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W101 / W98
checks modules, imported for their helpers, arm their own; every guard is verified at 0). No model is built and nothing
is solved: the write-only proof reads W104's persisted C* models (certified_models.pkl = TSO + DSO, esso_models_s39_D.pkl
= ESSO; both sha256-verified against W104's committed campaign manifest) and a fresh SRP1 planning object (data only).

WHAT IS CHECKED
  R   the hooks driven through the REAL wrappers with W104's RECORDED values (per_cycle_record, cycle record, decision,
      recourse_blocks_all and the g_s39_D Boyd rows, all hash-verified): R1 bitwise through 187 on every gated field
      (per-cycle, cycle-line, the mirror's settling record, the mirror's decision, and the W101 all-block lines except
      their timing), holds inert through 87 and held 88..287, the certificate length 10**9 throughout, the run reaching
      287, a synthetic creep 188..287, a Boyd lapse at 200 recorded without stopping the run, the report-only rule
      never stopping the run, Q decomposition reconciling on every recorded cycle, the Boyd capture equal to the
      recorded rows, the ESS movement equal to a brute-force recomputation; R2 a one-ulp gross perturbation at 150
      ABORTS at 150 with its magnitude; R3 a one-ulp pf_primal ratio at 120 aborts at 120; R4 a cycle-line-only
      difference (the AA action at 60) aborts at 60; R5 a settling synthetic continuation: the report-only rule
      certifies, the run continues to 287, first_would_certify recorded.
  L   WRITE-ONLY on real objects: the real `get_admm_boyd_residual_metrics` and `_get_operational_recourse_components`
      (and the two block functions) through the wrappers on W104's cycle-187 C* models: every Var value / fixed flag /
      bound, every Param value, every Constraint / Objective active flag of the TSO, DSO and ESSO models, the solver
      options (network holders and ESSO), the ADMM parameters, consensus_vars and dual_vars: fingerprints before ==
      after; the return values are the same objects; Q_187 recomputed from the persisted models vs W104's record; the
      Q decomposition reconciles; capture shapes and bytes per cycle; the capture overhead (seconds per cycle).
  H   the holds with REAL production functions (AA, tail, rho), and the real install layering (extension hooks first,
      the interface-dual capture and the harness appender on top; every production function restored; `_child_real`
      enters them in that order and asserts the checklist before the run).
  K   keys: for EVERY entry of every committed campaign spec the W105 harness's key equals the pre-W105 harness's
      (ab0bcbc9 = HEAD at the start of W105, sha256 pinned, loaded from git), continuations passed through; the
      extension key follows the declared formula, its base key equals the recert c_star eval key, and it appears in no
      committed campaign spec OUTSIDE ITS OWN ROOT (W108: the W105 stage root, excluded exactly as the launcher's
      pre_launch_assertion excludes it; the campaign roots live under it); negative controls: a planted committed-
      looking spec outside the root, and one in a sibling directory sharing the root's name prefix, are both refused;
      positive control: the campaign's own committed spec (a planted one in the r2 campaign root, and the committed
      v40 spec, superseded before any run) is accepted.
  P   `assert_extension_preconditions` holds for the declaration and refuses negative controls; the validator refuses
      an `early_stop` key and malformed declarations.

W108 (r2, before any run). The r1 module (committed 49342e8c; output zero_solve_checks/w105_zero_solve_checks.json
9688a018, pinned by v40 9bc1779d) required the extension key to be absent from EVERY committed campaign spec; W105's
own committed campaign spec (1a9f1f24, commit 4461e077) carries it, so the checks the --run mode re-runs could never
hold (W107's launch was refused with zero solves). r2 excludes the W105 stage root, as the pre-launch assertion does,
and writes to a NEW directory (zero_solve_checks_r2); the r1 output is kept, never overwritten.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w105_extension_checks.py \\
      > data/SRP1/Results/P515S53/w105_c_star_extension/zero_solve_checks_r2_launch.log 2>&1
"""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W105 settling-extension zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w105_settling_extension_hooks as E  # noqa: E402
import p515_s53_w101_settling_continuation_hooks as C101  # noqa: E402
import p515_s53_w98_continuation_checks as K98  # noqa: E402 -- its stand-in helpers (arms its own permitted=() guard)

GUARDS = (('w105_checks', GUARD), ('w98_checks_imported', K98.GUARD))
W105_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w105_c_star_extension')
# r2 (W108): a NEW output directory and file names; r1 (zero_solve_checks/w105_zero_solve_checks.json, 9688a018,
# pinned by the superseded v40) is kept and never written to.
OUT_DIR_REL = os.path.join(W105_ROOT_REL, 'zero_solve_checks_r2')
OUT_FILE = 'w105_zero_solve_checks_r2.json'
OUT_MANIFEST = 'w105_zero_solve_checks_r2_manifest_sha256.json'
# W108: the extension campaign (r2 id and root; the v40 root campaign_s53_w105_c_star_ext is write-once and superseded)
# and the roots excluded from the key-absence rule -- the W105 stage root, EXACTLY the launcher's pre_launch_assertion
# rule (L.committed_eval_keys(exclude_roots=(ROOT_REL,)), ROOT_REL = W105_ROOT_REL): a spec is inside a root iff its
# path starts with root + os.sep.
EXT_CAMPAIGN_ID = 's53_w105_c_star_ext_r2'
EXT_CAMPAIGN_ROOT_REL = os.path.join(W105_ROOT_REL, f'campaign_{EXT_CAMPAIGN_ID}')
V40_CAMPAIGN_SPEC_REL = os.path.join(W105_ROOT_REL, 'campaign_s53_w105_c_star_ext',
                                     'campaign_spec_s53_w105_c_star_ext_1a9f1f24.json')
KEY_EXCLUDED_ROOTS = (W105_ROOT_REL,)
PRE_W105_HARNESS = {'commit': 'ab0bcbc9', 'sha256': '3ad624cd74f9a832037d20d2480adb9b03d559aee1c5d9675dc8aeb6c3b9d8a2'}
W104_MANIFEST_REL = os.path.join(E.W104_ROOT, 'campaign_manifest_sha256.json')
W104_TSO_DSO_PKL_REL = os.path.join(E.W104_EVAL_DIR, 'certified_models.pkl')
W104_ESSO_PKL_REL = os.path.join(E.W104_EVAL_DIR, 'esso_models_s39_D.pkl')
RECERT_SPEC_REL = os.path.join(C101.RECERT_ROOT, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json')
W104_SPEC_REL = os.path.join(E.W104_ROOT, 'campaign_spec_s53_w101_srp1_cont_c_star_1a7483ef.json')
SYNTH_CREEP_RATE = -265.0     # synthetic continuation in the drive (a test fixture, not a prediction)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jt(x):
    return GRIO.dumps(x, default=GRIO.json_default, sort_keys=True)


# ======================================================================================================================
#  W104 fixtures
# ======================================================================================================================
def w104_fixtures():
    """W104's recorded C* evidence, each file hash-verified against the W105 declaration (per_cycle_record, cycle record,
    decision) or the W104 constants (recourse_blocks_all, g_s39_D)."""
    decl = E.declaration()
    rows, lines, dec = E.load_replay_reference(decl)
    out = {'rows': rows, 'lines': lines, 'decision': dec}
    for key in ('recourse_blocks_all', 'g_s39_D'):
        pin = E.W104[key]
        got = H.sha256_file(_abs(pin['path']))
        if got != pin['sha256']:
            raise RuntimeError(f'{pin["path"]} sha256 {got} != {pin["sha256"]}')
    out['blocks'] = {r['cycle']: r for r in _read_jsonl(_abs(E.W104['recourse_blocks_all']['path']))}
    g = json.load(open(_abs(E.W104['g_s39_D']['path'])))
    out['g'] = {r['cycle']: r for r in g['cycle_trajectory']}
    return out


def boyd_from_g(row):
    """Production's Boyd return value for one cycle, rebuilt from the recorded g_s39_D row (every field it holds)."""
    fields = ('r', 's', 's_rho_part', 's_proximal_part', 'proximal_share', 'eps_pri', 'eps_dual', 'norm_x', 'norm_z',
              'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance', 'primal_pass', 'dual_pass', 'channel_pass')
    out = {}
    for g in E.CHANNELS:
        out[g] = {f: row[f'boyd_{g}_{f}'] for f in fields}
    for f in ('norm_y_tso', 'norm_y_dso', 'norm_y_esso'):
        out['ess'][f] = row[f'boyd_ess_{f}']
    out['all_boyd_pass'] = row['boyd_all_pass']
    out['eps_abs'] = row['boyd_eps_abs']
    out['eps_rel'] = row['boyd_eps_rel']
    out['boyd_eps_source'] = row['boyd_eps_source']
    return out


# Production's iteration order for the block dicts (TSO years x days, then DSO node x years x days, then SALVAGE), from
# W104's committed interface-dual header (`interface_duals_per_cycle.jsonl` line 1 'blocks', production's own order).
# The recorded all-block rows are sorted alphabetically; rebuilding the dicts in production's order makes the hooks'
# float sums (block_sum) reproduce W104's bit for bit.
PRODUCTION_YEARS = ('2025', '2030', '2035')
PRODUCTION_DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
PRODUCTION_NODES = (5, 7, 9)


def _production_rank(key):
    agent, node_id, year, day = key
    if agent == 'SALVAGE':
        return (2, 0, 0, 0)
    y, d = PRODUCTION_YEARS.index(str(year)), PRODUCTION_DAYS.index(str(day))
    return (0, 0, y, d) if agent == 'TSO' else (1, PRODUCTION_NODES.index(node_id), y, d)


def blocks_from_line(bline):
    """(blocks, obj) as production's two block functions return them (in production's dict order), rebuilt from a
    recorded W101 all-block line."""
    rows = sorted(bline['blocks'], key=lambda r: _production_rank((r['agent'], r['node_id'], r['year'], r['day'])))
    blocks = {(r['agent'], r['node_id'], r['year'], r['day']): r['value'] for r in rows}
    obj = {}
    for r in sorted(bline['objective_component_blocks'],
                    key=lambda r: _production_rank((r['agent'], r['node_id'], r['year'], r['day']))):
        obj[(r['agent'], r['node_id'], r['year'], r['day'])] = {k: v for k, v in r.items()
                                                               if k not in ('agent', 'node_id', 'year', 'day')}
    return blocks, obj


# ======================================================================================================================
#  a fake SRP1-shaped ESS world (for the drive; the real-object check is L)
# ======================================================================================================================
NODES = (5, 7, 9)
YEARS = {2025: 5, 2030: 5, 2035: 5}
DAYS = {'Spring': 92, 'Summer': 91, 'Autumn': 91, 'Winter': 91}
PERIODS = 24


def _sched_value(cycle, family, side, n, yi, di, p):
    """A deterministic schedule value (MW); moves every cycle so sum |dx| is non-trivial."""
    base = {'p': 1.0, 'q': 0.3, 'charge': 0.5, 'discharge': 0.4}[family]
    s = {'esso': 0.0, 'tso': 0.01, 'dso': 0.02, 'z': 0.005}[side]
    return base * math.sin(0.37 * p + 0.11 * n + 0.5 * yi + 0.3 * di + s) + 0.001 * cycle * ((p % 5) - 2)


class _VarIdx(dict):
    pass


def fake_ess_world():
    """(pp, tso_model, dso_models, esso_model, consensus_vars, set_cycle): SimpleNamespace stand-ins with the SRP1
    shape (3 nodes x 3 years x 4 days x 24 periods, 3 cohorts, 1 x 1 scenarios, baseMVA 100); `set_cycle(c)` writes
    the cycle's deterministic schedules into them."""
    years, days = list(YEARS), list(DAYS)
    idx = {5: 0, 7: 1, 9: 2}
    tnet = {y: {d: SimpleNamespace(baseMVA=100.0, prob_market_scenarios=[1.0], prob_operation_scenarios=[1.0],
                                   get_shared_energy_storage_idx=lambda n: idx[n]) for d in days} for y in years}
    dnets = {n: SimpleNamespace(network={y: {d: SimpleNamespace(
        baseMVA=100.0, prob_market_scenarios=[1.0], prob_operation_scenarios=[1.0], get_reference_node_id=lambda: 1,
        get_shared_energy_storage_idx=lambda r: 0) for d in days} for y in years}) for n in NODES}
    pp = SimpleNamespace(active_distribution_network_nodes=list(NODES), years=dict(YEARS), days=dict(DAYS),
                         transmission_network=SimpleNamespace(network=tnet), distribution_networks=dnets)

    def net_block(n_ess):
        return SimpleNamespace(periods=list(range(PERIODS)), scenarios_market=[0], scenarios_operation=[0],
                               shared_es_pch=_VarIdx({(e, 0, 0, p): SimpleNamespace(value=0.0) for e in range(n_ess)
                                                      for p in range(PERIODS)}),
                               shared_es_pdch=_VarIdx({(e, 0, 0, p): SimpleNamespace(value=0.0) for e in range(n_ess)
                                                       for p in range(PERIODS)}))
    tso = {y: {d: net_block(3) for d in days} for y in years}
    dso = {n: {y: {d: net_block(1) for d in days} for y in years} for n in NODES}
    esso = {n: SimpleNamespace(years=[0, 1, 2], days=[0, 1, 2, 3], periods=list(range(PERIODS)),
                               es_pch_per_unit=_VarIdx({(yi, y, d, p): SimpleNamespace(value=0.0) for yi in range(3)
                                                        for y in range(3) for d in range(4) for p in range(PERIODS)}),
                               es_pdch_per_unit=_VarIdx({(yi, y, d, p): SimpleNamespace(value=0.0) for yi in range(3)
                                                         for y in range(3) for d in range(4) for p in range(PERIODS)}))
            for n in NODES}
    cv = {'ess': {side: {'current': {n: {y: {d: {'p': [0.0] * PERIODS, 'q': [0.0] * PERIODS} for d in days}
                                         for y in years} for n in NODES}} for side in E.ESS_SIDES}}

    def set_cycle(c):
        for n in NODES:
            for yi, y in enumerate(years):
                for di, d in enumerate(days):
                    for side in E.ESS_SIDES:
                        for fam in ('p', 'q'):
                            cv['ess'][side]['current'][n][y][d][fam] = [_sched_value(c, fam, side, n, yi, di, p)
                                                                        for p in range(PERIODS)]
                    for p in range(PERIODS):
                        # ESSO: split the charge over cohort 0 (2/3) and 1 (1/3); cohort 2 inactive (0)
                        ch = max(_sched_value(c, 'charge', 'esso', n, yi, di, p), 0.0)
                        dch = max(_sched_value(c, 'discharge', 'esso', n, yi, di, p), 0.0)
                        esso[n].es_pch_per_unit[(0, yi, di, p)].value = ch * 2.0 / 3.0
                        esso[n].es_pch_per_unit[(1, yi, di, p)].value = ch / 3.0
                        esso[n].es_pdch_per_unit[(0, yi, di, p)].value = dch
                        for side, blk, e in (('tso', tso[y][d], idx[n]), ('dso', dso[n][y][d], 0)):
                            blk.shared_es_pch[(e, 0, 0, p)].value = max(_sched_value(c, 'charge', side, n, yi, di, p),
                                                                        0.0) / 100.0
                            blk.shared_es_pdch[(e, 0, 0, p)].value = max(
                                _sched_value(c, 'discharge', side, n, yi, di, p), 0.0) / 100.0
    return pp, tso, dso, esso, cv, set_cycle


def brute_force_movement(c):
    """sum |x(c) - x(c-1)| per node for p_esso, p_tso, charge_esso and discharge_tso, straight from _sched_value."""
    years, days = list(YEARS), list(DAYS)
    out = {}
    for n in NODES:
        acc = {'p_esso': 0.0, 'p_tso': 0.0, 'charge_esso': 0.0, 'discharge_tso': 0.0}
        for yi in range(len(years)):
            for di in range(len(days)):
                for p in range(PERIODS):
                    acc['p_esso'] += abs(_sched_value(c, 'p', 'esso', n, yi, di, p)
                                         - _sched_value(c - 1, 'p', 'esso', n, yi, di, p))
                    acc['p_tso'] += abs(_sched_value(c, 'p', 'tso', n, yi, di, p)
                                        - _sched_value(c - 1, 'p', 'tso', n, yi, di, p))
                    ch = lambda k: (max(_sched_value(k, 'charge', 'esso', n, yi, di, p), 0.0) * 2.0 / 3.0  # noqa: E731
                                    + max(_sched_value(k, 'charge', 'esso', n, yi, di, p), 0.0) / 3.0)
                    acc['charge_esso'] += abs(ch(c) - ch(c - 1))
                    dt = lambda k: max(_sched_value(k, 'discharge', 'tso', n, yi, di, p), 0.0) / 100.0 * 100.0  # noqa
                    acc['discharge_tso'] += abs(dt(c) - dt(c - 1))
        out[str(n)] = acc
    return out


# ======================================================================================================================
#  R -- the recorded drive
# ======================================================================================================================
def _standins(fx, scripts):
    """Stand-in originals returning W104's RECORDED values (cycles <= 187) or the scripted synthetic ones."""
    calls = {}

    def log(name, *a):
        calls.setdefault(name, []).append(a)

    def baseline(pp, admm):
        log('baseline', pp, admm)
        return 'baseline-return'

    def apply(pp, admm, active, base, cycle):
        log('apply', active, cycle)
        return {'active': bool(active), 'cycle': cycle}

    def boyd(pp, tso, dso, esso, cv, dv, ap):
        log('boyd')
        return scripts['boyd'].pop(0)

    def aa(aa_state, layout, cv, dv, wb, rho, bm, it):
        log('aa', it, bm['all_boyd_pass'])
        act = scripts['aa'].pop(0)
        return {'cycle': it, 'action': act if act is not None else
                (E.AA_OFF_ACTION if bm['all_boyd_pass'] else 'accepted')}

    def nxt(conv, aa_enabled, aa_record):
        log('next', conv)
        v = scripts['next'].pop(0)
        return bool(conv) if v is None else v

    def pen(tso, dso, esso, rm, bm, params, iter=None, allow_update=True, freeze_state=None):
        log('pen', iter, allow_update)
        return scripts['pen'].pop(0)

    def recourse(pp, models):
        log('recourse')
        return scripts['rc'].pop(0)

    def efc(esso):
        log('efc')
        return scripts['efc'].pop(0)

    return {'_capture_convergence_depth_tail_baseline': baseline, '_apply_convergence_depth_tail': apply,
            'get_admm_boyd_residual_metrics': boyd, '_anderson_acceleration_cycle_step': aa,
            '_convergence_depth_tail_next_state': nxt, '_update_admm_penalties': pen,
            '_get_operational_recourse_components': recourse, '_get_admm_efc_per_day_max': efc}, calls


def _pen_from_line(line):
    r = line['rho']
    return (dict(r['actions']), dict(r['rho_before']), dict(r['rho_after']), dict(r['gamma_before']),
            dict(r['gamma_after']), r['rho_freeze_active'], {g: {'frozen': r['frozen'][g]} for g in E.CHANNELS})


def _synthetic_blocks(fx, c, q_shift):
    """Cycle-187 recorded blocks with q_shift added to the TSO 2025 Spring block's generation cost (so the synthetic
    change sits 100 % in TSO generation, a test fixture)."""
    blocks, obj = blocks_from_line(fx['blocks'][187])
    key = next(k for k in blocks if k[0] == 'TSO' and k[2] == '2025' and k[3] == 'Spring')
    blocks = dict(blocks)
    obj = {k: dict(v) for k, v in obj.items()}
    blocks[key] = blocks[key] + q_shift
    for f in ('generation_cost', 'economic_market_cost', 'classified_total', 'objective_value'):
        obj[key][f] = obj[key][f] + q_shift
    return blocks, obj


def drive(fx, variant='bitwise', sink=None):
    """The real wrappers (E.make_wrappers) driven in production's call order over cycles 1..287: recorded values
    through 187, a synthetic continuation after. Returns a dict of what happened (and the sink)."""
    decl = E.declaration()
    reference = (fx['rows'], fx['lines'], fx['decision'])
    sink = [] if sink is None else sink
    st = E.ExtensionState(decl, None, E.CAP, reference=reference, sink=sink)
    scripts = {'boyd': [], 'aa': [], 'next': [], 'pen': [], 'rc': [], 'efc': []}
    orig, calls = _standins(fx, scripts)
    blocks_by_cycle = {}

    def blocks_fn(pp, models):
        return blocks_by_cycle[st.cycle][0]

    def obj_fn(pp, models):
        return blocks_by_cycle[st.cycle][1]
    fake_srp = SimpleNamespace(_get_operational_recourse_block_components=blocks_fn,
                               _get_operational_objective_component_blocks=obj_fn)
    w = E.make_wrappers(st, orig, srp_module=fake_srp)
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    pp, tso, dso, esso, cv, set_cycle = fake_ess_world()
    models = {'tso': tso, 'dso': dso, 'esso': esso}
    # esso_side_terms needs pyomo values: the fake ESSO carries none -> recorded as a capture error unless patched
    esso_terms_orig = E.esso_side_terms
    E.esso_side_terms = lambda pp_, em: {str(n): {'salvage_value': 0.0, 'feasibility_penalty': 0.0} for n in NODES}
    raised = None
    q187 = fx['rows'][187]['gross_operational_cost']
    g187 = fx['g'][187]
    line187 = fx['lines'][187]
    certs_seen = []
    try:
        w['_capture_convergence_depth_tail_baseline'](pp, admm)
        active = False
        for c in range(1, E.CAP + 1):
            if c <= E.REPLAY_THROUGH:
                row, line, g = fx['rows'][c], fx['lines'][c], fx['g'][c]
                bm = boyd_from_g(g)
                if variant == 'ulp_pf_primal_at_120' and c == 120:
                    bm = copy.deepcopy(bm)
                    bm['pf']['primal_ratio'] = math.nextafter(bm['pf']['primal_ratio'], math.inf)
                rc = {'gross_operational_cost': row['gross_operational_cost'], 'net_operational_recourse': row['recourse'],
                      'terminal_salvage_value': row['terminal_salvage_value']}
                if variant == 'ulp_gross_at_150' and c == 150:
                    rc['gross_operational_cost'] = math.nextafter(rc['gross_operational_cost'], math.inf)
                aa_act = (line.get('aa') or {}).get('action') if c <= E.N_HOLD else None
                if variant == 'aa_action_at_60' and c == 60:
                    aa_act = 'accepted' if aa_act != 'accepted' else 'rejected (safeguard; memory retained)'
                nxt = (line.get('tail_next') or {}).get('value') if c <= E.N_HOLD else None
                pen = _pen_from_line(line)
                efc = row['efc_per_day_max']
                local_ok = row['local_solves_ok']
                active = (line['tail_apply']['active_passed'] if c <= E.N_HOLD else line['tail_apply']['natural_active'])
                blocks_by_cycle[c] = blocks_from_line(fx['blocks'][c])
            else:
                j = c - E.REPLAY_THROUGH
                if variant == 'settles':
                    qs = 6000.0 * (0.6 ** (j / 15.0)) * math.cos(2 * math.pi * j / 30.0) - 6000.0
                else:
                    qs = SYNTH_CREEP_RATE * j + 40.0 * math.sin(j)
                bm = copy.deepcopy(boyd_from_g(g187))
                # R1: a lapse at 200 is recorded and does not stop the run; R5: a lapse at 190 resets k0 so the damped
                # synthetic oscillation is judged on its own swings (W104's 29.9 k EUR swing is otherwise in A)
                lapse = (variant == 'bitwise' and c == 200) or (variant == 'settles' and c == 190)
                if lapse:
                    bm['all_boyd_pass'] = False
                    bm['pf']['channel_pass'] = False
                    bm['pf']['primal_ratio'] = 1.5
                rc = {'gross_operational_cost': q187 + qs, 'net_operational_recourse': q187 + qs,
                      'terminal_salvage_value': 0.0}
                aa_act = None
                nxt = None
                pen = _pen_from_line(line187)
                efc = 1.0
                local_ok = True
                active = True
                blocks_by_cycle[c] = _synthetic_blocks(fx, c, qs + (fx['blocks'][187]['gross_operational_cost'] - q187))
            set_cycle(c)
            scripts['boyd'].append(bm)
            scripts['aa'].append(aa_act)
            scripts['next'].append(nxt)
            scripts['pen'].append(pen)
            if local_ok:
                scripts['rc'].append(rc)
            scripts['efc'].append(efc)
            with contextlib.redirect_stdout(io.StringIO()):
                w['_apply_convergence_depth_tail'](pp, admm, active, object(), c)
                w['get_admm_boyd_residual_metrics'](pp, tso, dso, esso, cv, {}, admm)
                if local_ok:
                    aa_rec = w['_anderson_acceleration_cycle_step'](object(), None, cv, {}, None, {}, bm, c)
                    w['_get_operational_recourse_components'](pp, models)
                else:
                    aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
                w['_convergence_depth_tail_next_state'](bool(bm['all_boyd_pass'] and local_ok), True, aa_rec)
                w['_update_admm_penalties']({}, {}, {}, {}, bm, object(), iter=c, allow_update=local_ok,
                                            freeze_state={})
                w['_get_admm_efc_per_day_max'](esso)
            if st.cur.get('report_only_would_certify'):
                certs_seen.append(c)
        w['_apply_convergence_depth_tail'](pp, admm, False, object(), None)
    except RuntimeError as error:
        raised = str(error)
    finally:
        E.esso_side_terms = esso_terms_orig
    return {'state': st, 'sink': sink, 'raised': raised, 'calls': calls, 'admm': admm, 'certs_seen': certs_seen}


def _sink_files(sink):
    files = {}
    for fname, obj in sink:
        files.setdefault(fname, []).append(obj)
    return files


def tests_R():
    fx = w104_fixtures()
    res = {}
    # ---- R1: bitwise through 187, then the synthetic creep with a lapse at 200 --------------------------------------
    t0 = time.time()
    d = drive(fx, 'bitwise')
    wall_drive = time.time() - t0
    st, files = d['state'], _sink_files(d['sink'])
    lines = files.get(E.CYCLE_FILE, [])
    creep = files.get(E.CREEP_FILE, [])
    blines = files.get(E.BLOCKS_FILE, [])
    ess = files.get(E.ESS_SCHEDULE_FILE, [])
    decs = files.get(E.DECISION_FILE, [])
    summ = st.summary()
    # independent post-hoc comparisons (the in-cycle gate already aborted on any difference)
    line_eq = all(not E.line_compare(x, fx['lines'][x['cycle']]) for x in lines if x['cycle'] <= 187)
    blocks_eq = all(_jt({k: v for k, v in b.items() if k != 'capture_s'})
                    == _jt({k: v for k, v in fx['blocks'][b['cycle']].items() if k != 'capture_s'})
                    for b in blines if b['cycle'] <= 187)
    blocks_bad = [b['cycle'] for b in blines if b['cycle'] <= 187 and
                  _jt({k: v for k, v in b.items() if k != 'capture_s'})
                  != _jt({k: v for k, v in fx['blocks'][b['cycle']].items() if k != 'capture_s'})][:5]
    boyd_eq = all(_jt(E.boyd_capture(boyd_from_g(fx['g'][x['cycle']]))) == _jt(x['boyd']) for x in creep
                  if x['cycle'] <= 187)
    recon = [x['cycle'] for x in creep if not ((x.get('q_decomposition') or {}).get('reconciliation') or {})
             .get('reconciles')]
    recon_max = max(abs(x['q_decomposition']['reconciliation']['gross_minus_sum_components']) for x in creep
                    if x['cycle'] <= 187)
    bf = {c: brute_force_movement(c) for c in (2, 150, 250)}
    by_creep = {x['cycle']: x for x in creep}
    ess_ok = all(abs(by_creep[c]['ess_movement']['per_node'][n][f] - bf[c][n][f]) <= 1e-9 * max(1.0, bf[c][n][f])
                 for c in bf for n in bf[c] for f in bf[c][n])
    holds_through = sorted({_jt(x['holds']) for x in lines if x['cycle'] <= E.N_HOLD})
    holds_after = sorted({_jt(x['holds']) for x in lines if x['cycle'] > E.N_HOLD})
    aa_forced = [a for a in d['calls'].get('aa', []) if a[0] > E.N_HOLD]
    pen_after = sorted({a[1] for a in d['calls'].get('pen', []) if a[0] > E.N_HOLD})
    cert_len = sorted({x['certificate_length_in_force_at_cycle_end'] for x in lines})
    ext_vs_mirror = all(_jt(x['settling']) == _jt(x['settling_w104_mirror']) for x in lines if x['cycle'] <= 186)
    lapse = summ['lapse_events']
    r1 = {
        'raised': d['raised'], 'lines': len(lines), 'creep_lines': len(creep), 'blocks_lines': len(blines),
        'ess_schedule_lines': len(ess), 'decision_lines': len(decs),
        'replay_bitwise_through': summ['replay_bitwise_through_cycle'], 'first_divergence': summ['replay_first_divergence'],
        'mirror_decision_equals_w104': summ['mirror_decision_equals_w104'],
        'cycle_lines_1_187_equal_w104_posthoc': line_eq, 'blocks_lines_1_187_equal_w104_except_capture_s': blocks_eq,
        'blocks_lines_differing_first5': blocks_bad,
        'boyd_capture_equals_recorded_g_rows_1_187': boyd_eq,
        'q_decomposition_reconciles_every_cycle': not recon, 'q_non_reconciling_cycles': recon[:10],
        'q_max_abs_gross_minus_sum_components_1_187': recon_max,
        'ess_movement_equals_brute_force_at_2_150_250': ess_ok,
        'ess_movement_first_cycle_unavailable': by_creep[1]['ess_movement']['available'] is False,
        'holds_through_87': holds_through, 'holds_after_87': holds_after,
        'aa_calls_after_87_all_forced_true': bool(aa_forced) and all(a[1] is True for a in aa_forced),
        'pen_allow_update_after_87': pen_after,
        'certificate_length_values_seen': cert_len, 'certificate_length_after_exit': d['admm'].minimum_consecutive_converged_cycles,
        'report_only_rule_equals_mirror_1_186': ext_vs_mirror,
        'lapse_events': lapse, 'stopped_by': summ['stopped_by'], 'last_cycle': summ['last_cycle'],
        'summary_ok': summ['ok'], 'capture_errors': summ['capture_errors'][:5], 'errors': summ['errors'],
        'report_only_first_would_certify_cycle': summ['report_only_first_would_certify_cycle'],
        'wall_s': wall_drive}
    r1['ok'] = bool(
        d['raised'] is None and len(lines) == E.CAP and len(creep) == E.CAP and len(blines) == E.CAP
        and len(ess) == E.CAP + 1 and len(decs) == 1 and summ['replay_bitwise_through_cycle'] == 187
        and summ['mirror_decision_equals_w104'] is True and line_eq and blocks_eq and boyd_eq and not recon and ess_ok
        and r1['ess_movement_first_cycle_unavailable']
        and holds_through == [_jt({'aa': False, 'rho': False, 'tail_apply': False, 'tail_next': False})]
        and holds_after == [_jt({'aa': True, 'rho': True, 'tail_apply': True, 'tail_next': True})]
        and r1['aa_calls_after_87_all_forced_true'] and pen_after == [False]
        and cert_len == [E.CERTIFICATION_DISABLED_THRESHOLD] and d['admm'].minimum_consecutive_converged_cycles == 10
        and ext_vs_mirror and len(lapse) == 1 and lapse[0]['cycle'] == 200 and summ['stopped_by'] == 'cap'
        and summ['last_cycle'] == E.CAP and summ['ok'] is True)
    res['R1_bitwise_1_187_then_creep_to_287_lapse_at_200'] = r1
    # ---- R2-R4: divergences abort at the cycle ----------------------------------------------------------------------
    for name, variant, cyc, field_kind in (('R2_one_ulp_gross_at_150', 'ulp_gross_at_150', 150, 'gross_operational_cost'),
                                           ('R3_one_ulp_pf_primal_ratio_at_120', 'ulp_pf_primal_at_120', 120,
                                            'boyd_pf_primal_ratio'),
                                           ('R4_cycle_line_only_aa_action_at_60', 'aa_action_at_60', 60, 'aa')):
        dd = drive(fx, variant)
        s2 = dd['state'].summary()
        f2 = _sink_files(dd['sink'])
        div = s2['replay_first_divergence'] or {}
        ok = (dd['raised'] is not None and f'REPLAY DIVERGED at cycle {cyc}' in dd['raised']
              and len(f2.get(E.CYCLE_FILE, [])) == cyc and s2['replay_bitwise_through_cycle'] == cyc - 1
              and s2['stopped_by'] == 'replay_divergence_abort' and not s2['ok']
              and f2[E.CYCLE_FILE][-1].get('replay_equal') is False)
        if field_kind == 'aa':
            ok = ok and div.get('cycle_line_fields_differing') == ['aa'] and div.get('fields_differing') == []
        else:
            ok = ok and field_kind in (div.get('fields_differing') or [])
        res[name] = {'ok': bool(ok), 'raised': dd['raised'], 'lines': len(f2.get(E.CYCLE_FILE, [])),
                     'divergence': {k: div.get(k) for k in ('cycle', 'fields_differing', 'cycle_line_fields_differing',
                                                           'gross_difference_run_minus_recorded',
                                                           'max_relative_difference')}}
    # ---- R5: a settling synthetic continuation: the report-only rule certifies, the run continues ---------------------
    d5 = drive(fx, 'settles')
    s5 = d5['state'].summary()
    f5 = _sink_files(d5['sink'])
    dec5 = (f5.get(E.DECISION_FILE) or [{}])[0]
    ok5 = (d5['raised'] is None and s5['report_only_first_would_certify_cycle'] is not None
           and s5['report_only_first_would_certify_cycle'] > 187 and s5['last_cycle'] == E.CAP
           and len(f5.get(E.CYCLE_FILE, [])) == E.CAP and s5['ok'] is True
           and sorted({x['certificate_length_in_force_at_cycle_end'] for x in f5[E.CYCLE_FILE]})
           == [E.CERTIFICATION_DISABLED_THRESHOLD]
           and dec5.get('first_would_certify', {}).get('k_star') == s5['report_only_first_would_certify_cycle']
           and len(s5['report_only_would_certify_cycles']) >= 1)
    res['R5_report_only_certification_does_not_stop'] = {
        'ok': bool(ok5), 'first_would_certify': s5['report_only_first_would_certify_cycle'],
        'branch': s5['report_only_first_would_certify_branch'],
        'n_would_certify_cycles': len(s5['report_only_would_certify_cycles']),
        'would_certify_first10': s5['report_only_would_certify_cycles'][:10], 'last_cycle': s5['last_cycle'],
        'status_at_cap': s5['report_only_status_at_cap']}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  L -- write-only on real objects, shapes, sizes, overhead
# ======================================================================================================================
def _fingerprint_block(b, h, counts):
    import pyomo.environ as pe
    counts['blocks'] += 1
    for v in b.component_data_objects(pe.Var, descend_into=True, sort=True):
        val = v.value
        h.update(f'{v.name}|{float.hex(float(val)) if val is not None else None}|{v.fixed}|{v.lb}|{v.ub}\n'.encode())
        counts['vars'] += 1
    for pc in b.component_objects(pe.Param, descend_into=True, sort=True):
        for idx in sorted(pc.keys(), key=repr):
            try:
                val = pe.value(pc[idx])
                txt = float.hex(float(val)) if isinstance(val, (int, float)) else repr(val)
            except Exception as error:  # noqa: BLE001
                txt = f'<{type(error).__name__}>'
            h.update(f'{pc.name}[{idx!r}]|{pc.mutable}|{txt}\n'.encode())
            counts['params'] += 1
    for c in b.component_data_objects(pe.Constraint, descend_into=True, sort=True):
        h.update(f'{c.name}|{c.active}\n'.encode())
        counts['constraints'] += 1
    for o in b.component_data_objects(pe.Objective, descend_into=True, sort=True):
        h.update(f'{o.name}|{o.active}\n'.encode())
        counts['objectives'] += 1


def fingerprint_models(models):
    h = hashlib.sha256()
    counts = {'vars': 0, 'params': 0, 'constraints': 0, 'objectives': 0, 'blocks': 0}
    for y in sorted(models['tso'], key=str):
        for d in sorted(models['tso'][y], key=str):
            _fingerprint_block(models['tso'][y][d], h, counts)
    for nd in sorted(models['dso'], key=str):
        for y in sorted(models['dso'][nd], key=str):
            for d in sorted(models['dso'][nd][y], key=str):
                _fingerprint_block(models['dso'][nd][y][d], h, counts)
    for nd in sorted(models['esso'], key=str):
        _fingerprint_block(models['esso'][nd], h, counts)
    return h.hexdigest(), counts


def fingerprint_planning(planning, srp):
    h = hashlib.sha256()
    for label, nd in srp._convergence_depth_tail_holders(planning):
        K._deep_hex({'label': label, 'options': dict(nd.params.solver_params.options or {}),
                     'recovery': dict(nd.params.solver_params.recovery_options or {})}, h)
        for y in nd.years:
            for d in nd.days:
                net = nd.network[y][d]
                K._deep_hex({'baseMVA': float(net.baseMVA),
                             'branches': [(br.fbus, br.tbus, float(br.rate), bool(br.status)) for br in net.branches]}, h)
    sp = planning.shared_ess_data.params.solver_params
    K._deep_hex({'esso_solver_options': dict(getattr(sp, 'options', None) or {}),
                 'esso_recovery': dict(getattr(sp, 'recovery_options', None) or {})}, h)
    K._deep_hex(json.loads(json.dumps(vars(planning.params.admm), default=str, sort_keys=True)), h)
    return h.hexdigest()


def _tree_hash(obj):
    h = hashlib.sha256()
    K._deep_hex(obj, h)
    return h.hexdigest()


def tests_L():
    import p56a_oracle as O
    import shared_resources_planning as srp
    res = {}
    manifest = json.load(open(_abs(W104_MANIFEST_REL)))
    pins = {}
    for rel in (W104_TSO_DSO_PKL_REL, W104_ESSO_PKL_REL):
        pins[rel] = {'manifest_sha256': manifest.get(rel), 'sha256': H.sha256_file(_abs(rel))}
    eval_id = f'p515s53_w105_writeonly_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)
    try:
        bad = [r for r, p in pins.items() if p['manifest_sha256'] is None or p['sha256'] != p['manifest_sha256']]
        if bad:
            raise RuntimeError(f'persisted model sha256 differs from W104 campaign manifest: {bad}')
        t0 = time.time()
        with open(_abs(W104_TSO_DSO_PKL_REL), 'rb') as handle:
            models = pickle.load(handle)
        with open(_abs(W104_ESSO_PKL_REL), 'rb') as handle:
            models['esso'] = pickle.load(handle)
        load_s = time.time() - t0
        planning = O.fresh_planning(eval_id)
        cv, dv = srp.create_admm_variables(planning)
        rnd = random.Random(54)

        def fill(node):
            if isinstance(node, dict):
                for k in node:
                    if isinstance(node[k], list):
                        node[k] = [rnd.uniform(-3.0, 3.0) for _ in node[k]]
                    elif isinstance(node[k], float):
                        node[k] = rnd.uniform(-3.0, 3.0)
                    else:
                        fill(node[k])
        fill(cv)
        fill(dv)
        before_m, counts = fingerprint_models(models)
        before_p = fingerprint_planning(planning, srp)
        before_cv, before_dv = _tree_hash(cv), _tree_hash(dv)
        # the real production functions, through the wrappers, in the in-cycle state
        sink = []
        st = E.ExtensionState(E.declaration(), None, E.CAP, reference=({}, {}, None), sink=sink)
        originals = {name: getattr(srp, name) for name in E.WRAPPED}
        w = E.make_wrappers(st, originals, srp_module=srp)
        rets, walls = [], []
        for c in (1, 2):
            st.cycle, st.phase = c, 'in_cycle'
            st.cur = {'cycle': c, 'phase': 'replay', 'regime': 'pre_hold', 'boyd_k': True}
            if c == 2:   # move one consensus copy between the two calls so sum |dx| is non-trivial
                n0, y0, d0 = E.ess_block_order(planning)[0]
                moved_orig = cv['ess']['esso']['current'][n0][y0][d0]['p'][3]
                cv['ess']['esso']['current'][n0][y0][d0]['p'][3] = moved_orig + 0.25
            ta = time.time()
            b = w['get_admm_boyd_residual_metrics'](planning, models['tso'], models['dso'], models['esso'], cv, dv,
                                                    planning.params.admm)
            tb = time.time()
            rc = w['_get_operational_recourse_components'](planning, models)
            tc = time.time()
            rets.append((b, rc))
            cap = st.cur.get('_boyd_capture') or {}
            walls.append({'boyd_production_plus_capture_s': tb - ta, 'boyd_and_ess_capture_only_s': cap.get('capture_s'),
                          'recourse_production_plus_blocks_line_plus_q_s': tc - tb,
                          'q_decomposition_plus_esso_terms_s': st.cur.get('_q_capture_s')})
            if c == 1:
                cap1 = cap
                qd1 = st.cur.get('_q_decomposition')
            else:
                cap2 = cap
            st.cur['finalized'] = True
        # production values WITHOUT the wrappers, on the same state as the second wrapped call (consensus moved)
        b_plain = srp.get_admm_boyd_residual_metrics(planning, models['tso'], models['dso'], models['esso'], cv, dv,
                                                     planning.params.admm)
        rc_plain = srp._get_operational_recourse_components(planning, models)
        after_m, counts2 = fingerprint_models(models)
        cv['ess']['esso']['current'][n0][y0][d0]['p'][3] = moved_orig    # restored exactly (not by subtraction)
        after_p = fingerprint_planning(planning, srp)
        after_cv, after_dv = _tree_hash(cv), _tree_hash(dv)
        q187_rec = json.loads(open(_abs(E.W104['per_cycle_record']['path'])).read().splitlines()[186])[
            'gross_operational_cost']
        q187_now = rc_plain['gross_operational_cost']
        # shapes and bytes
        sched = cap1['ess_schedules']
        n_blocks = len(E.ess_block_order(planning))
        shape_ok = (all(len(sched[k][s]) == n_blocks and all(len(x) == 24 for x in sched[k][s])
                        for k in ('p', 'q') for s in E.ESS_SIDES)
                    and all(len(sched[k][s]) == n_blocks for k in ('charge', 'discharge') for s in E.ESS_CHARGE_SIDES))
        mv2 = cap2['ess_movement']
        creep_line = {'cycle': 150, 'phase': 'extension', 'regime': 'hold', 'gross': rc['gross_operational_cost'],
                      'step': -265.0, 'local_solves_ok': True, 'boyd_k': True, 'q_decomposition': None,
                      'ess_movement': mv2, 'boyd': cap1['boyd'],
                      'capture_s': {'boyd_and_ess': 0.01, 'q_decomposition': 0.01}}
        # a q_decomposition with deltas (two identical captures -> deltas 0)
        blocks = srp._get_operational_recourse_block_components(planning, models)
        obj = srp._get_operational_objective_component_blocks(planning, models)
        qd2 = E.q_decomposition(blocks, obj, rc_plain['gross_operational_cost'], qd1)
        qd2['esso_side_terms_not_in_gross_Q'] = qd1.get('esso_side_terms_not_in_gross_Q')
        creep_line['q_decomposition'] = qd2
        bytes_creep = len(GRIO.dumps(creep_line, default=GRIO.json_default).encode()) + 1
        bytes_ess = len(GRIO.dumps({'cycle': 150, **sched}, default=GRIO.json_default).encode()) + 1
        blines = [o for f, o in sink if f == E.BLOCKS_FILE]
        bytes_blocks = len(GRIO.dumps(blines[-1], default=GRIO.json_default).encode()) + 1
        # the write cost of the two new lines per cycle (append + flush + fsync, as ExtensionState._write)
        wtmp = tempfile.mkdtemp(prefix='w105_write_cost_')
        try:
            wst = E.ExtensionState(E.declaration(), wtmp, E.CAP, reference=({}, {}, None), sink=None)
            tw = time.time()
            for _i in range(5):
                wst._write(E.CREEP_FILE, creep_line)
                wst._write(E.ESS_SCHEDULE_FILE, {'cycle': 150, **sched})
            write_s_per_cycle = (time.time() - tw) / 5.0
        finally:
            shutil.rmtree(wtmp)
        agents = qd1['agents']
        res['L1_write_only_real_c_star_models'] = {
            'holds': bool(before_m == after_m and counts == counts2 and before_p == after_p and before_cv == after_cv
                          and before_dv == after_dv and all(r[0] is not None for r in rets)
                          and _jt(E.boyd_capture(rets[1][0])) == _jt(E.boyd_capture(b_plain))
                          and rets[0][1]['gross_operational_cost'] == rc_plain['gross_operational_cost']
                          and not st.capture_errors and not st.errors),
            'persisted_models': pins, 'load_s': load_s, 'model_fingerprint_before': before_m,
            'model_fingerprint_after': after_m, 'components': counts,
            'planning_fingerprint_equal': before_p == after_p, 'consensus_vars_equal': before_cv == after_cv,
            'dual_vars_equal': before_dv == after_dv, 'capture_errors': st.capture_errors[:5], 'errors': st.errors,
            'note': ('the wrappers return production\'s own values (compared with an unwrapped call on the same '
                     'objects); the ESSO fingerprint covers esso_models_s39_D.pkl; solver options include the ESSO\'s')}
        res['L2_q187_from_persisted_models'] = {
            'holds': True, 'Q187_recorded_w104': q187_rec, 'Q187_recomputed_from_persisted_models': q187_now,
            'bitwise_equal': q187_rec == q187_now, 'difference': q187_now - q187_rec,
            'note': 'report-only: the persisted TSO / DSO / ESSO models are W104\'s terminal (cycle-187) state'}
        res['L3_q_decomposition_real'] = {
            'holds': bool(qd1['reconciliation']['reconciles'] is True and len(qd1['blocks']) == 48
                          and sorted(agents) == ['DSO_5', 'DSO_7', 'DSO_9', 'ESSO', 'TSO']
                          and all(agents['ESSO'][c] == 0.0 for c in E.Q_COMPONENTS_ALL)
                          and all(v == 0.0 for r in qd2['blocks_delta'] for k_, v in r.items()
                                  if k_ in E.Q_COMPONENTS_ALL)),
            'reconciliation': qd1['reconciliation'],
            'agents_at_187': {a: {c: v.get(c) for c in E.Q_COMPONENTS_ALL + ('value', 'ess_terms')}
                              for a, v in agents.items()},
            'esso_side_terms_not_in_gross_Q': qd1.get('esso_side_terms_not_in_gross_Q'),
            'blocks': len(qd1['blocks'])}
        res['L4_shapes_sizes_overhead'] = {
            'holds': bool(shape_ok and mv2['available'] is True and cap1['ess_movement']['available'] is False
                          and abs(mv2['per_node'][str(n0)]['p_esso'] - 0.25) <= 1e-12
                          and all(v == 0.0 for k_, v in mv2['all_nodes'].items() if k_ != 'p_esso')),
            'ess_blocks': n_blocks, 'periods': 24, 'nodes': list(planning.active_distribution_network_nodes),
            'ess_families': {'p': list(E.ESS_SIDES), 'q': list(E.ESS_SIDES), 'charge': list(E.ESS_CHARGE_SIDES),
                             'discharge': list(E.ESS_CHARGE_SIDES)},
            'floats_per_cycle_ess_schedule': n_blocks * 24 * (2 * len(E.ESS_SIDES) + 2 * len(E.ESS_CHARGE_SIDES)),
            'moved_one_value_by_0.25_sum_abs_dp_esso_node': mv2['per_node'][str(n0)]['p_esso'],
            'ess_movement_second_call_all_nodes': mv2['all_nodes'],
            'bytes_per_cycle': {'creep_line': bytes_creep, 'ess_schedule_line_real_values': bytes_ess,
                                'w101_blocks_line_unchanged': bytes_blocks},
            'bytes_287_cycles_MiB': {'creep': 287 * bytes_creep / 2 ** 20, 'ess_schedule': 287 * bytes_ess / 2 ** 20},
            'overhead_s_per_cycle_measured': walls,
            'write_s_per_cycle_creep_plus_ess_lines_fsync': write_s_per_cycle,
            'new_overhead_s_per_cycle_estimate': (max(x['boyd_and_ess_capture_only_s'] for x in walls)
                                                  + max(x['q_decomposition_plus_esso_terms_s'] for x in walls)
                                                  + write_s_per_cycle),
            'overhead_note': ('new work per cycle = Boyd + ESS capture, Q decomposition + ESSO terms, and the two '
                              'new line writes; the recourse + two block functions + W101 block line were already '
                              'paid in W104 (its per-cycle walls include them)'),
            'boyd_capture_keys': sorted(cap1['boyd']), 'boyd_channel_fields': sorted(cap1['boyd']['pf']),
            'work_dir_created_for_fresh_planning': os.path.relpath(work, REPO)}
    except Exception as error:  # noqa: BLE001
        res['L_error'] = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    finally:
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)
    return res


# ======================================================================================================================
#  H -- holds with REAL production functions, layering
# ======================================================================================================================
def _state():
    return E.ExtensionState(E.declaration(), None, E.CAP, reference=({}, {}, None), sink=[])


def _h_aa_real():
    import numpy as np
    import admm_anderson_acceleration as aam
    import shared_resources_planning as srp
    dim = 6
    rng = np.random.default_rng(20260928)
    A = 0.9 * np.eye(dim) + 0.02 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    real_collect, real_write = aam.collect_w, aam.write_back_w
    n = E.N_HOLD

    def collect(layout, cv, dv, rho, check_antisymmetry=True):
        return np.array(cv['w'], dtype=float)

    def write_back(layout, w, cv, dv, rho):
        cv['w'] = np.array(w, dtype=float)

    def run(wrapped):
        st = _state()
        w = E.make_wrappers(st, {'_anderson_acceleration_cycle_step': srp._anderson_acceleration_cycle_step,
                                 **{k: None for k in E.WRAPPED if k != '_anderson_acceleration_cycle_step'}})
        step = w['_anderson_acceleration_cycle_step'] if wrapped else srp._anderson_acceleration_cycle_step
        state = aam.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
        cv = {'w': np.zeros(dim)}
        trace = []
        for c in range(1, n + 5):
            st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
            w_before = np.array(cv['w'])
            cv['w'] = A @ cv['w'] + b * (1.0 + 0.1 * np.sin(c))
            r = 1.0 / c
            boyd = {'all_boyd_pass': False, **{g: {'r': r, 's': r, 'primal_ratio': r, 'dual_ratio': r}
                                               for g in E.CHANNELS}}
            rec = step(state, None, cv, {}, w_before, {'v': 1.0, 'pf': 1.0, 'ess': 1.0}, boyd, c)
            trace.append({'cycle': c, 'action': rec['action'], 'w_hex': [float.hex(float(x)) for x in cv['w']],
                          'boyd_seen_by_caller': boyd['all_boyd_pass']})
        return trace

    aam.collect_w, aam.write_back_w = collect, write_back
    try:
        ref = run(False)
        wrp = run(True)
    finally:
        aam.collect_w, aam.write_back_w = real_collect, real_write
    restored = aam.collect_w is real_collect and aam.write_back_w is real_write
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = [(ref[i]['action'], wrp[i]['action'], ref[i]['w_hex'] != wrp[i]['w_hex']) for i in range(n, n + 4)]
    acts = (all(a == 'accepted' and bw == E.AA_OFF_ACTION for a, bw, _d in after) and all(d for _a, _b, d in after)
            and all(not t['boyd_seen_by_caller'] for t in wrp))
    n_acc = sum(1 for t in ref[:n] if t['action'] == 'accepted')
    return {'holds': inert and acts and restored and n_acc > 0, 'inert_through_87_bitwise': inert, 'acts_after_87': acts,
            'n_accepted_extrapolations_through_87': n_acc, 'aam_restored': restored}


def _h_tail_real(srp):
    names = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
             '_convergence_depth_tail_next_state')
    real = {k: getattr(srp, k) for k in names}
    n = E.N_HOLD

    def run(wrapped):
        st = _state()
        w = E.make_wrappers(st, dict(real, **{k: None for k in E.WRAPPED if k not in real}))
        fn = w if wrapped else real
        pp = K98._fake_holders()
        admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                               minimum_consecutive_converged_cycles=10)
        base = fn['_capture_convergence_depth_tail_baseline'](pp, admm)
        active_next = False
        trace = []
        for c in range(1, n + 5):
            if wrapped and st.cur is not None:
                st.cur['finalized'] = True
            rec = fn['_apply_convergence_depth_tail'](pp, admm, active_next, base, c)
            opts = K98._holder_options(pp)
            conv = 78 <= c <= n
            aa_rec = {'cycle': c, 'action': E.AA_OFF_ACTION if conv else 'accepted'}
            active_next = fn['_convergence_depth_tail_next_state'](conv, True, aa_rec)
            trace.append({'cycle': c, 'apply_record': rec, 'options_after_apply': opts, 'next': bool(active_next)})
        return trace, admm

    ref, _a = run(False)
    wrp, admm_w = run(True)
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = wrp[n:]
    acts = (all(t['apply_record']['active'] is True for t in after)
            and all(set(v['compl_inf_tol'] for v in t['options_after_apply'].values()) == {1e-6} for t in after)
            and all(t['next'] is True for t in after) and any(r['apply_record']['active'] is False for r in ref[n + 1:]))
    return {'holds': inert and acts and admm_w.minimum_consecutive_converged_cycles == E.CERTIFICATION_DISABLED_THRESHOLD,
            'inert_through_87': inert, 'acts_after_87': acts,
            'certificate_length_after_baseline': admm_w.minimum_consecutive_converged_cycles}


def _h_rho_real(srp):
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    admm = params.admm
    real = srp._update_admm_penalties
    n = E.N_HOLD

    def metrics(c):
        hi = (c % 2 == 1)
        rm = {'primal': {g: 1.0 for g in E.CHANNELS} | {f'{g}_mean': 1.0 for g in E.CHANNELS},
              'dual': {f'{g}_mean': 1.0 for g in E.CHANNELS}}
        bm = {g: {'primal_ratio': 50.0 if hi else 0.5, 'dual_ratio_balance': 0.5 if hi else 50.0, 'dual_ratio': 0.5,
                  'r': 1.0, 's': 1.0, 'eps_pri': 1.0, 'eps_dual': 1.0} for g in E.CHANNELS}
        return rm, bm

    def run(wrapped):
        st = _state()
        w = E.make_wrappers(st, dict({'_update_admm_penalties': real},
                                     **{k: None for k in E.WRAPPED if k != '_update_admm_penalties'}))
        tso, dso, esso = K98._rho_models()
        fs = srp._init_admm_freeze_state()
        trace = []
        with contextlib.redirect_stdout(io.StringIO()):
            for c in range(1, n + 5):
                st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
                rm, bm = metrics(c)
                fn = w['_update_admm_penalties'] if wrapped else real
                actions, before, after, bg, ag, rfa, fs = fn(tso, dso, esso, rm, bm, admm, iter=c, allow_update=True,
                                                             freeze_state=fs)
                trace.append({'cycle': c, 'actions': dict(actions), 'rho_hex': K98._rho_read(tso, dso, esso),
                              'freeze_state': copy.deepcopy(fs), 'changed': before != after})
        return trace

    ref = run(False)
    wrp = run(True)
    inert = all(ref[i] == wrp[i] for i in range(n))
    moved = sum(1 for t in ref[:n] if t['changed'])
    after_ref = [t['changed'] for t in ref[n:]]
    after_wrp = [t['changed'] for t in wrp[n:]]
    frozen = all(wrp[i]['rho_hex'] == wrp[n - 1]['rho_hex'] for i in range(n, n + 4))
    acts = all(after_ref) and not any(after_wrp) and frozen
    return {'holds': inert and acts and moved > 0, 'inert_through_87_bitwise': inert, 'acts_after_87': acts,
            'cycles_with_rho_change_through_87': moved, 'wrapped_rho_after_87_equals_rho_at_87_bitwise': frozen}


def _h_layering(srp):
    before = {name: getattr(srp, name) for name in E.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w105_layering_')
    n = E.N_HOLD
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with E.settling_extension_hooks(scratch, E.declaration(), holder, cap=E.CAP) as st:
            installed_inner = srp.get_admm_boyd_residual_metrics
            with IDC.interface_dual_capture_hooks(scratch, holder):
                idc_outer = srp.get_admm_boyd_residual_metrics is not installed_inner
                with H.convergence_depth_append_hooks(stub):
                    base = srp._capture_convergence_depth_tail_baseline(pp, admm)
                    active = False
                    for c in range(1, n + 3):
                        if st.cur is not None:
                            st.cur['finalized'] = True
                        srp._apply_convergence_depth_tail(pp, admm, active, base, c)
                        conv = c <= n
                        active = srp._convergence_depth_tail_next_state(
                            conv if c <= n else False, True,
                            {'cycle': c, 'action': E.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
                    st.cur['finalized'] = True
                    srp._apply_convergence_depth_tail(pp, admm, active, base, None)
        files_written = sorted(os.listdir(scratch))
    finally:
        shutil.rmtree(scratch)
    after = {name: getattr(srp, name) for name in before}
    restored = all(after[k] is before[k] for k in before)
    next_events = [e[1]['value'] for e in stub.events if e[0] == 'next_state']
    apply_events = [e[1] for e in stub.events if e[0] == 'apply']
    appender_saw_held = (next_events[n] is True and apply_events[n + 1]['record']['active'] is True)
    child_src = inspect.getsource(H._child_real)
    i_cont = child_src.find('with continuation_cm, \\')
    i_settle = child_src.find('settling_cm, \\', i_cont)
    i_ext = child_src.find('extension_cm, \\', i_cont)
    i_idc = child_src.find('IDC.interface_dual_capture_hooks(eval_dir, holder) as dual_capture', i_cont)
    i_s38 = child_src.find('G.s38_pf_capture_hooks(', i_cont)
    i_app = child_src.find('convergence_depth_append_hooks(appender)', i_cont)
    order_ok = 0 < i_cont < i_settle < i_ext < i_idc < i_s38 < i_app
    pre_ok = (0 < child_src.find('W105C.assert_extension_preconditions(extension, spec, tail_checklist, aa_on)')
              < child_src.find('G.run_admm_arm('))
    summ = holder.get(E.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and order_ok and pre_ok and idc_outer
                          and summ.get('phase') == 'ended' and summ.get('certificate_length_restored_at_exit') == 10
                          and not summ.get('errors') and admm.minimum_consecutive_converged_cycles == 10),
            'appender_recorded_held_tail_value_after_87': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_extension_boyd_wrapper': idc_outer,
            'child_real_order_continuation_settling_extension_idc_s38_appender': order_ok,
            'child_real_checklist_before_run_admm_arm': pre_ok, 'files_written_by_install_without_cycles': files_written,
            'summary_phase': summ.get('phase'), 'summary_errors': summ.get('errors')}


def tests_H():
    import shared_resources_planning as srp
    res = {'H1_aa_hold_real_production': _h_aa_real(), 'H2_tail_hold_real_production': _h_tail_real(srp),
           'H3_rho_hold_real_production': _h_rho_real(srp), 'H4_layering_real_install': _h_layering(srp)}
    got = E._certificate_length_writes_in_source()
    tampered = sorted(got + ['st.params.minimum_consecutive_converged_cycles = 0'])
    res['H5_certificate_length_writes_only_disable_restore'] = {
        'holds': (got == sorted(E.CERTIFICATE_LENGTH_WRITES) and tampered != sorted(E.CERTIFICATE_LENGTH_WRITES)
                  and 'early_stop' not in inspect.getsource(E.make_wrappers)
                  and 'SETTLING_CERTIFIED_THRESHOLD' not in inspect.getsource(E)),
        'writes_in_source': got, 'negative_control_extra_write_detected': tampered != sorted(E.CERTIFICATE_LENGTH_WRITES)}
    return {'holds': all(v['holds'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w105():
    import importlib.util
    import subprocess
    src = subprocess.run(['git', 'show', f"{PRE_W105_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W105_HARNESS['sha256']:
        raise RuntimeError(f'pre-W105 harness sha256 {sha} != pinned {PRE_W105_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w105_pre_harness_')
    path = os.path.join(tmp, '_w105_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w105_pre_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod, sha


def _key_args(spec, e):
    cfg = spec.get('configuration') or {}
    overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
    return (e['key'], overrides), dict(
        case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=e.get('model_variant'),
        ess_ageing_baseline=cfg.get('ess_ageing_baseline'), flex_price_multiplier=e.get('flex_price_multiplier'),
        derived_instance=cfg.get('derived_instance'), interface_deviation_premium=e.get('interface_deviation_premium'),
        convergence_depth_tail=cfg.get('convergence_depth_tail'),
        certification_continuation=e.get('certification_continuation'),
        settling_continuation=e.get('settling_continuation'))


def extension_key():
    recert = json.load(open(_abs(RECERT_SPEC_REL)))
    cfg = recert['configuration']
    e = next(x for x in recert['candidates'] if x['label'] == E.BASE_CELL)
    kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
              convergence_depth_tail=cfg['convergence_depth_tail'])
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    decl = E.declaration()
    ext = H.evaluation_key(e['key'], e['overrides'], settling_extension=decl, **kw)
    formula = hashlib.sha256(json.dumps({'base_evaluation_key': e['eval_key'], 'settling_extension': decl},
                                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'recert_eval_key': e['eval_key'], 'base_key_now': base, 'extension_key': ext,
            'base_equals_recert': base == e['eval_key'], 'formula_holds': ext == formula,
            'differs_from_recert': ext != e['eval_key'], 'differs_from_w104': ext != E.W104['eval_key']}


def _key_holders(scanned, key):
    """The spec paths among `scanned` = [(rel, spec)] with an entry whose eval key (H._entry_eval_key) is `key`."""
    return sorted({rel for rel, spec in scanned for e in spec.get('candidates') or [] if H._entry_eval_key(e) == key})


def _outside_roots(rels, exclude_roots=KEY_EXCLUDED_ROOTS):
    """The paths of `rels` outside every root of `exclude_roots` (the pre-launch assertion's rule: a path is inside a
    root iff it starts with root + os.sep)."""
    return [r for r in rels if not any(r.startswith(root + os.sep) for root in exclude_roots)]


def tests_K():
    pre, pre_sha = _harness_pre_w105()
    n_specs = n_entries = n_equal = n_equal_committed = n_with_eval_key = 0
    mismatch, errors, committed_diff = [], [], []
    scanned = []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            try:
                args, kw = _key_args(spec, e)
                new, old = H.evaluation_key(*args, **kw), pre.evaluation_key(*args, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
            if 'eval_key' in e:
                n_with_eval_key += 1
                if e['eval_key'] == new:
                    n_equal_committed += 1
                else:
                    committed_diff.append({'spec': rel, 'label': e.get('label')})
    ext = extension_key()
    key = ext['extension_key']
    holding = _key_holders(scanned, key)
    outside = _outside_roots(holding)
    # ---- controls (W108): planted committed-looking specs run through the same holder scan and the same root rule ---
    def planted(rel):
        return (rel, {'campaign_id': 'w108_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': key}]})
    planted_outside_rel = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w108_planted_negative_control',
                                       'campaign_planted', 'campaign_spec_planted_00000000.json')
    planted_sibling_rel = os.path.join(W105_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_own_rel = os.path.join(EXT_CAMPAIGN_ROOT_REL, f'campaign_spec_{EXT_CAMPAIGN_ID}_00000000.json')
    ctl_outside = _outside_roots(_key_holders(scanned + [planted(planted_outside_rel)], key))
    ctl_sibling = _outside_roots(_key_holders(scanned + [planted(planted_sibling_rel)], key))
    ctl_own_holders = _key_holders(scanned + [planted(planted_own_rel)], key)
    ctl_own = _outside_roots(ctl_own_holders)
    controls = {
        'i_planted_outside_root_refused': {'planted': planted_outside_rel, 'outside_found': ctl_outside,
                                           'refused': planted_outside_rel in ctl_outside},
        'i_planted_sibling_prefix_refused': {'planted': planted_sibling_rel, 'outside_found': ctl_sibling,
                                             'refused': planted_sibling_rel in ctl_sibling},
        'ii_own_campaign_spec_accepted': {'planted': planted_own_rel, 'holders': ctl_own_holders,
                                          'outside_found': ctl_own,
                                          'accepted': planted_own_rel in ctl_own_holders and planted_own_rel not in ctl_own},
        'ii_v40_campaign_spec_accepted': {'path': V40_CAMPAIGN_SPEC_REL,
                                          'committed_and_holds_the_key': V40_CAMPAIGN_SPEC_REL in holding,
                                          'accepted': V40_CAMPAIGN_SPEC_REL not in outside},
    }
    parts = {
        'key_regression_no_mismatch': not mismatch,
        'key_regression_no_errors': not errors,
        'key_regression_every_entry_equal': n_equal == n_entries,
        'extension_base_equals_recert': ext['base_equals_recert'],
        'extension_formula_holds': ext['formula_holds'],
        'extension_differs_from_recert': ext['differs_from_recert'],
        'extension_differs_from_w104': ext['differs_from_w104'],
        'extension_key_absent_from_committed_specs_outside_own_root': not outside,
        'own_campaign_root_inside_excluded_root': not _outside_roots([planted_own_rel]),
        'control_i_planted_outside_root_refused': controls['i_planted_outside_root_refused']['refused'],
        'control_i_planted_sibling_prefix_refused': controls['i_planted_sibling_prefix_refused']['refused'],
        'control_ii_own_campaign_spec_accepted': controls['ii_own_campaign_spec_accepted']['accepted'],
        'control_ii_v40_campaign_spec_accepted': controls['ii_v40_campaign_spec_accepted']['accepted'],
    }
    holds = all(v is True for v in parts.values())
    return {'holds': holds, 'parts': parts, 'pre_w105_harness': {**PRE_W105_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries,
            'entries_new_equals_pre_w105': n_equal, 'mismatches': mismatch, 'errors': errors,
            'entries_carrying_eval_key': n_with_eval_key,
            'entries_recomputed_equal_committed_eval_key_REPORTED': n_equal_committed,
            'entries_recomputed_differing_from_committed_eval_key_first20_REPORTED': committed_diff[:20],
            'n_entries_recomputed_differing_from_committed_eval_key_REPORTED': len(committed_diff),
            'extension_key': ext, 'key_excluded_roots': list(KEY_EXCLUDED_ROOTS),
            'own_campaign_root': EXT_CAMPAIGN_ROOT_REL,
            'extension_key_in_committed_specs_all_REPORTED': holding,
            'extension_key_in_committed_specs_outside_own_root': outside, 'controls': controls}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    decl = E.declaration()
    try:
        good = E.assert_extension_preconditions(decl, {'cap': E.CAP}, {'tail_enabled_for_this_run': True}, True)
        good_ok = all(good.values())
    except Exception as error:  # noqa: BLE001
        good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
    negatives = {}
    for name, (spec, tail, aa) in {'cap_187': ({'cap': 187}, {'tail_enabled_for_this_run': True}, True),
                                   'tail_off': ({'cap': E.CAP}, {'tail_enabled_for_this_run': False}, True),
                                   'aa_off': ({'cap': E.CAP}, {'tail_enabled_for_this_run': True}, False)}.items():
        try:
            E.assert_extension_preconditions(decl, spec, tail, aa)
            negatives[name] = 'NOT refused'
        except RuntimeError as error:
            negatives[name] = f'refused: {error}'
    bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
           'wrong_N': dict(decl, hold_after_cycle=88), 'wrong_cap': dict(decl, cap=387),
           'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], per_cycle_record_sha256='0' * 64)),
           'extra_key': dict(decl, extra=1), 'rule_mode_latching': dict(decl, settling_rule_mode='latching'),
           'p_max_23': dict(decl, settling_rule=dict(decl['settling_rule'], p_max=23)),
           'abort_false': dict(decl, abort_on_replay_divergence=False),
           'capture_off': dict(decl, captures=dict(decl['captures'], ess_schedule_movement=False)),
           'w101_declaration': C101.declaration_for('c_star', 22)}
    refused = {}
    for name, d in bad.items():
        try:
            E.validate_settling_extension(d)
            refused[name] = False
        except ValueError as error:
            refused[name] = str(error)[:160]
    # the harness refuses mixing the extension with a continuation
    try:
        H.evaluation_key('0' * 64, {}, settling_extension=decl, settling_continuation=C101.declaration_for('c_star', 22))
        mixed = False
    except ValueError:
        mixed = True
    ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values()) and mixed
    return {'holds': bool(ok), 'checklist': good, 'negative_controls': negatives, 'validator_refuses': refused,
            'harness_refuses_extension_plus_continuation': mixed}


# ======================================================================================================================
import p515_s53_w101_continuation_checks as K  # noqa: E402 -- _deep_hex (arms its own permitted=() guard)
GUARDS = GUARDS + (('w101_checks_imported', K.GUARD),)


def run_all_checks():
    out = {}
    sections = (('R', tests_R), ('L', tests_L), ('H', tests_H), ('K', tests_K), ('P', tests_P))
    ok = True
    for sid, fn in sections:
        t0 = time.time()
        try:
            r = fn()
        except Exception as error:  # noqa: BLE001 -- recorded as a failing section
            r = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        r_ok = _section_ok(sid, r)
        out[sid] = {'holds': r_ok, 'wall_s': time.time() - t0, 'result': r}
        ok = ok and r_ok
    return {'all_hold': ok, 'sections': out, 'declaration': E.declaration()}


def _section_ok(sid, r):
    if 'error' in r and len(r) <= 2:
        return False
    if sid in ('R', 'H', 'K', 'P'):
        return r.get('holds') is True
    return bool(r) and all(v.get('holds') is True for v in r.values())


CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'p515_s53_w105_settling_extension_hooks.py',
                         'p515_s53_w101_settling_continuation_hooks.py', 'settling_criterion.py',
                         'interface_dual_capture.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
                         'shared_resources_planning.py', 'shared_energy_storage_data.py', 'network.py',
                         'admm_anderson_acceleration.py', 'p515_s53_w98_continuation_checks.py',
                         'p515_s53_w101_continuation_checks.py')


def main():
    started = _utc()
    out_dir = _abs(OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_FILE, OUT_MANIFEST):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    res = run_all_checks()
    code_pins = {rel: H.sha256_file(_abs(rel)) for rel in CODE_PINNED_BY_CHECKS}
    guards = {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}
    doc = {'schema': 'p515_s53_w105_zero_solve_checks_v1', 'task': 'W105 (PLANNER_BRIEF_2026-09-13.md Addendum 54)',
           'revision': 'r2 (Planner task W108: K excludes the W105 stage root, as pre_launch_assertion does)',
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'code_sha256': code_pins, 'guards': guards, **res}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    for sid, r in (res.get('sections') or {}).items():
        print(f"[W105-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W105-CHECKS] all_hold={res.get('all_hold')} guards={guards}")
    print(f"[W105-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (res.get('all_hold') and guards_ok) else 1)


if __name__ == '__main__':
    main()
