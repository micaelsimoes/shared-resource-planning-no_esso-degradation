"""
P5.15 W187 -- the report of the 1x unit-neighbourhood screen (Addendum 91 and its amendments 1-3; Addendum 85 ruling 4
(vii); STEP7_SIMULATION_PLAN section 5.3; P3 of Addendum 89). ZERO SOLVES, NO MODEL BUILD, NO PICKLE LOAD.

It reads COMMITTED cell outputs only (each input must be git-tracked and clean; its sha256 is recorded) and writes ONE
new JSON file in the screen root (opened 'x': never over an existing file) through the shared gate-result writer
`gate_result_io`. It never recomputes certification: every status, Q, bar and rule-ten value is the evaluator's own
recorded field. Recomputed here, as checks only: the bar from the per-cycle record (equality with the recorded bar), the
lattice label / investment cost of each candidate (the pinned W2 table through the real `p515_s47_phase_b_record`
lattice, as the launcher's `build_lattice`), and the comparisons.

INSTANCE. The 15 cells of the stage spec frozen_s55_step7_screen_spec_v1_5cbf4a55.json (spec v1.1 25071d2f): x = 0 at
1x under A (its N alias by G0) and the 14 admissible unit neighbours of x = 0 under N at 1x (0.25 MVA / 0.5 MWh at
every non-empty subset of nodes {5, 7, 9}, investment year 2025 or 2030). Every cell is a PRODUCTION-EXIT evaluation
(ten consecutive passing cycles, cap 500, no settling_resettle declaration): there is no rule-v6 band, k*, k0 or
branch. Each value is reported with its eval key and candidate key.

OBJECTIVE CONVENTIONS
  Q  = certified_cost = gross_operational_cost, contracted interface settlement EXCLUDED (Addendum 83 ruling 2);
       salvage reported separately; net = gross - terminal salvage.
  F  = I(x) + Q(x), I(x) = the pinned W2 investment cost (p515_s47_phase_b_record.OBJECTIVE_CONVENTION: salvage and
       net recourse excluded from F, Addendum 27 item 3). F(0) = Q(0).

COMPARISONS (each neighbour n against x0_A_1x; margin m = value(n) - value(x0), so m < 0 means n is cheaper)
  quantities  dQ = Q(n) - Q(x0) (as ordered by the Planner); dF = F(n) - F(x0) = dQ + I(n) (the frozen comparator's
              quantity); dF_net = dF - (salvage(n) - salvage(x0)) (informational only).
  thresholds  T_v6bar  = settling_criterion_v6.determinacy_threshold(bar_n, bar_x0) = max(3 x max(bar_n, bar_x0), 2 TAU)
                         with b = the screen's recorded bar (max |dQ_k| over the last 10 cycles), because a production-
                         exit cell has no band (P3's max{3b, 2 tau} with b read as the screen bar);
              T_frozen = p515_s47_phase_b_record.resolution(bar_n, bar_x0, sigma_Q) = max(bar_n + bar_x0, sigma_Q),
                         sigma_Q = 18,449.66 (the comparator frozen in every screen cell spec: plan 5.3, Addendum 85
                         ruling 4 (vii));
              T_band   = max(3 x larger band width, 2 TAU): NOT COMPUTABLE (no band at production exit), recorded None.
  verdicts    rule-v6 sign form: |m| >= T -> 'neighbour better' (m < 0) or 'x0 better' (m > 0); else 'within
              resolution' (settling_criterion_v6.determinate_certified). The frozen comparator's own one-sided
              classification p515_s47_phase_b_record.classify(F(x0), F(n), T_frozen) is recorded beside it.
  P3          "no no-ageing unit neighbour beats x = 0 by more than max{3b, 2 tau}; at least one is within resolution":
              clause 1 holds iff no neighbour has verdict 'neighbour better'; clause 2 holds iff at least one has
              'within resolution'. Evaluated for every (quantity, threshold) reading on the 14 neighbours (the set the
              stage spec and Addendum 85 ruling 4 (vii) define) and, informationally, on the 6 single-node cells.

HOLDERS (informational): Q(x0, 1x) against the committed x = 0 records -- W86 tail recert (spec ddd6cd44, the same eval
key 5cfe69a6...) and E3's C_OPEN (E.21 row 1, W101 d110bd1a..., rule-v6 certified at 181, band from the frozen step-6
tables 590088fe).

GUARDS. SolveProfileGuard(permitted=()) installed before any project import, verified at exactly 0 (and the
p515_s47_phase_b_record parent guard it arms at import, also at 0). pickle.load / loads / Unpickler blocked for the
whole run (0 calls required). pyomo Model.__init__ blocked and counted (0 required). Refuses: an existing output file,
an input that is untracked or modified, a pinned input whose sha256 differs, a missing capture path (asserted before
any reading).

OUTPUT data/SRP1/Results/P515S55/e1_screen_1x/w187_screen_report.json; `--post-run` writes
w187_screen_report_manifest_sha256.json (this script, the output, the launch log, every input) -- both write-once.
`--trial-out DIR` (outside the repository) writes w187_screen_report_TRIAL.json there instead, marked trial, and is the
only mode that runs from an uncommitted script.

Command (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w187_screen_report.py \\
      > data/SRP1/Results/P515S55/e1_screen_1x/w187_screen_report_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w187_screen_report.py --post-run
"""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W187 screen report (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0, 'Unpickler': 0}


def _pickle_refused(kind):
    def _f(*_a, **_k):
        PICKLE_COUNTS[kind] += 1
        raise RuntimeError(f'W187 screen report: pickle.{kind} refused -- no pickle load is permitted')
    return _f


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):  # noqa: D401
        PICKLE_COUNTS['Unpickler'] += 1
        raise RuntimeError('W187 screen report: pickle.Unpickler refused -- no pickle load is permitted')


pickle.load, pickle.loads, pickle.Unpickler = _pickle_refused('load'), _pickle_refused('loads'), _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('W187 screen report: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402  (pure)
import p515_s47_phase_b_record as PB  # noqa: E402  (pure lattice primitives; arms its own parent guard)

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
SCREEN_ROOT = os.path.join(P55, 'e1_screen_1x')
STAGE_SPEC = {'path': os.path.join(SCREEN_ROOT, 'frozen_s55_step7_screen_spec_v1_5cbf4a55.json'),
              'sha256': '5cbf4a5555466d9e561cfaaa48c76a0fa2eebbbabde74b606e3e52190d9e4f1f'}
SPEC_V1_1 = {'path': os.path.join(P55, 'frozen_step7_campaign_spec_v1_1_25071d2f.json'),
             'sha256': '25071d2f7545ee1c27a352539687ddeebcc46fbcab3a864b2ecb71889228e5f3'}
X0_CELL = 'x0_A_1x'
OUT_JSON = os.path.join(SCREEN_ROOT, 'w187_screen_report.json')
OUT_LOG = os.path.join(SCREEN_ROOT, 'w187_screen_report_launch.log')
OUT_MANIFEST = os.path.join(SCREEN_ROOT, 'w187_screen_report_manifest_sha256.json')

W86_ROOT = os.path.join(P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
W86 = {'spec': os.path.join(W86_ROOT, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json'),
       'results': os.path.join(W86_ROOT, 'campaign_results.json'),
       'record': os.path.join(W86_ROOT, 'evals', '5cfe69a615ae3708_x0', 'evaluation_record.json'),
       'per_cycle': os.path.join(W86_ROOT, 'evals', '5cfe69a615ae3708_x0', 'per_cycle_record.jsonl')}
C_OPEN_ROOT = os.path.join(P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0')
C_OPEN = {'results': os.path.join(C_OPEN_ROOT, 'campaign_results.json'),
          'record': os.path.join(C_OPEN_ROOT, 'evals', 'd110bd1a5977df1e_x0', 'evaluation_record.json'),
          'record_sha256': 'c08fa4335c7e266d6d61439fe36d28d56cc7ead31b4956d0f2c6d1fe2b0e8956',
          'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546'}
STEP6_TABLES = {'path': os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json'),
                'sha256': '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'}
SIGMA_Q_EXPECTED_EUR = 18449.66  # the launcher's constant (p515_s55_step7_launcher.SIGMA_Q_EXPECTED_EUR)

OBJECTIVE_CONVENTION = ('Q = certified_cost = gross_operational_cost, contracted interface settlement EXCLUDED (Addendum '
                        '83 ruling 2); salvage reported separately; net = gross - terminal salvage. F = I(x) + Q(x) '
                        'with I(x) from the pinned W2 table (p515_s47_phase_b_record.OBJECTIVE_CONVENTION; salvage '
                        'excluded from F, Addendum 27 item 3); F(0) = Q(0).')

INPUTS = {}  # rel path -> sha256 of every file read


def _log(msg):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} [W187-REPORT] {msg}", flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _committed_clean(rel):
    return bool(_git(['ls-files', '--', rel]).strip()) and not _git(['status', '--porcelain', '--', rel]).strip()


def _read(rel, pin=None, jsonl=False):
    """Read a committed, clean input; record its sha256; refuse on a pin mismatch."""
    if not os.path.isfile(rel):
        raise SystemExit(f'REFUSED: input missing: {rel}')
    if not _committed_clean(rel):
        raise SystemExit(f'REFUSED: input not committed and clean: {rel}')
    digest = _sha(rel)
    if pin is not None and digest != pin:
        raise SystemExit(f'REFUSED: {rel} sha256 {digest} != pinned {pin}')
    INPUTS[rel] = digest
    with open(rel) as f:
        if jsonl:
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _cell_root(stage, cell):
    return os.path.join(SCREEN_ROOT, os.path.dirname(stage['cells'][cell]['campaign_spec']))


def _eval_dir(root, spec):
    return os.path.join(root, 'evals', spec['candidates'][0]['eval_dir'])


def capture_checklist(stage):
    """Before any reading: every path the report needs exists (rule eleven)."""
    missing = []
    for cell in stage['cell_order']:
        root = _cell_root(stage, cell)
        spec_rel = os.path.join(SCREEN_ROOT, stage['cells'][cell]['campaign_spec'])
        for p in (spec_rel, os.path.join(root, 'campaign_results.json'),
                  os.path.join(root, 'campaign_manifest_sha256.json')):
            if not os.path.isfile(p):
                missing.append(p)
        if os.path.isfile(spec_rel):
            with open(spec_rel) as f:
                spec = json.load(f)
            ed = _eval_dir(root, spec)
            for name in ('evaluation_record.json', 'per_cycle_record.jsonl'):
                if not os.path.isfile(os.path.join(ed, name)):
                    missing.append(os.path.join(ed, name))
    for p in list(W86.values()) + [C_OPEN['results'], C_OPEN['record'], STEP6_TABLES['path'],
                                   PB.INVESTMENT_COST_RESULTS['path'], PB.PHASE_A_TABLES['path'], SPEC_V1_1['path']]:
        if not os.path.isfile(p):
            missing.append(p)
    return missing


def bar_from_trajectory(rows, n=10):
    """max |gross[k] - gross[k-1]| over the last n cycles (the harness's documented bar definition) -- a check only."""
    g = {r['cycle']: r.get('gross_operational_cost') for r in rows}
    last = max(g)
    steps = [abs(g[k] - g[k - 1]) for k in range(last - n + 1, last + 1)
             if g.get(k) is not None and g.get(k - 1) is not None]
    return max(steps) if len(steps) == n else None


def build_lattice():
    """As p515_s55_step7_launcher.build_lattice: the PB lattice on the pinned W2 / Phase A tables."""
    w2 = _read(PB.INVESTMENT_COST_RESULTS['path'], PB.INVESTMENT_COST_RESULTS['sha256'])
    tables = _read(PB.PHASE_A_TABLES['path'], PB.PHASE_A_TABLES['sha256'])
    costs = PB.unit_costs_from_w2(w2)
    lattice = PB.Lattice(tuple(sorted(costs)), costs)
    cross = PB.w2_cross_check(lattice, w2)
    sigma = PB.sigma_q_from_tables(tables)
    if not cross['ok']:
        raise SystemExit(f'REFUSED: I(x) closed form != W2 table: {cross}')
    if round(sigma['sigma_Q_eur'], 2) != SIGMA_Q_EXPECTED_EUR:
        raise SystemExit(f"REFUSED: sigma_Q {sigma['sigma_Q_eur']} != {SIGMA_Q_EXPECTED_EUR}")
    return lattice, {'I_x_cross_check_vs_W2': cross, 'sigma_Q': sigma, 'unit_costs': costs}


def cell_view(stage, cell, lattice):
    entry = stage['cells'][cell]
    root = _cell_root(stage, cell)
    spec = _read(os.path.join(SCREEN_ROOT, entry['campaign_spec']), entry['sha256'])
    res = _read(os.path.join(root, 'campaign_results.json'))
    _read(os.path.join(root, 'campaign_manifest_sha256.json'))
    ed = _eval_dir(root, spec)
    rec = _read(os.path.join(ed, 'evaluation_record.json'))
    rows = _read(os.path.join(ed, 'per_cycle_record.jsonl'), jsonl=True)
    scr = (res.get('outcome') or {}).get('screen') or {}
    rc = rec.get('recourse_components') or {}
    rt = rec.get('rule_ten') or {}
    z = lattice.z_of_canonical(rec['candidate_canonical'])
    last = max(rows, key=lambda r: r['cycle'])
    bar_rec = (rec.get('bar') or {}).get('value')
    bar_traj = bar_from_trajectory(rows)
    checks = {
        'record_eval_key_equals_stage_spec': rec.get('eval_key') == entry['eval_key'],
        'results_eval_key_equals_stage_spec': res.get('eval_key') == entry['eval_key'],
        'results_exit_code_0': res.get('exit_code') == 0,
        'results_guards_all_zero': (res.get('guards') or {}).get('all_zero') is True,
        'outcome_faults_empty': not (res.get('outcome') or {}).get('faults'),
        'outcome_stops_empty': not (res.get('outcome') or {}).get('stops'),
        'screen_certified_cost_equals_record': scr.get('certified_cost') == rec.get('certified_cost'),
        'record_certified_cost_equals_gross': rec.get('certified_cost') == rc.get('gross_operational_cost'),
        'screen_bar_equals_record_bar': scr.get('bar_eur') == bar_rec,
        'record_bar_equals_trajectory_bar': bar_traj == bar_rec,
        'trajectory_last_gross_equals_certified_cost': last.get('gross_operational_cost') == rec.get('certified_cost'),
        'trajectory_last_cycle_equals_cycles_run': last['cycle'] == rec.get('cycles_run'),
        'on_lattice': z is not None,
        'lattice_label_matches_cell': (z is not None and (lattice.label(z) + ('_A_1x' if cell == X0_CELL else '_N_1x'))
                                       == cell),
        'solve_identity_holds': (rec.get('solve_profile') or {}).get('identity_holds') is True,
    }
    sp = rec.get('solve_profile') or {}
    return {
        'cell': cell, 'campaign_id': spec.get('campaign_id'), 'eval_key': rec.get('eval_key'),
        'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
        'z': list(z) if z is not None else None, 'lattice_label': lattice.label(z) if z is not None else None,
        'n_storage_nodes': sum(1 for v in rec['candidate_canonical']['nodes'].values() if v[0] or v[1]),
        'I_eur': lattice.investment_cost(z) if z is not None else None,
        'status': rec.get('status'), 'protocol': 'production exit (10 consecutive passing cycles, cap 500)',
        'cap': rec.get('cap'), 'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'stopped_by': (rec.get('stopped_by_trajectory') or {}).get('stopped_by'),
        'boyd_converged_at_cycle': (rec.get('stopped_by_trajectory') or {}).get('converged_at_cycle'),
        'first_pass_cycle_per_channel': rec.get('first_pass_cycle_per_channel'),
        'Q_gross_eur': rec.get('certified_cost'),
        'gross_including_settlement_eur': rc.get('gross_operational_cost_including_settlement'),
        'interface_settlement_total_eur': rc.get('interface_settlement_total'),
        'terminal_salvage_eur': rc.get('terminal_salvage_value'),
        'net_operational_recourse_eur': rc.get('net_operational_recourse'),
        'detector_penalty_total_eur': rc.get('detector_penalty_total'),
        'band': None, 'band_width_eur': None, 'k_star': None, 'k0': None, 'branch': None,
        'band_note': 'not applicable: production-exit cell (no settling_resettle declaration; stage spec phase_text)',
        'bar_eur': bar_rec, 'bar_definition': (rec.get('bar') or {}).get('definition'),
        'bar_recomputed_from_trajectory_eur': bar_traj,
        'bar_net_recourse_step_reported_eur': (rec.get('bar_net_recourse_step_reported') or {}).get('value'),
        'terminal_step_eur': rt.get('terminal_objective_change_abs'),
        'terminal_threshold_eur': rt.get('terminal_objective_tolerance'),
        'stopping_ratio': rt.get('terminal_step_over_threshold'),
        'boyd_terminal_ratio_max_per_channel': rt.get('boyd_terminal_ratio_max_per_channel'),
        'solves_observed': (sp.get('observed') or {}).get('permitted_solve'),
        'solves_expected': sp.get('expected_solves'), 'solve_identity_holds': sp.get('identity_holds'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'network_failure_classes': (rec.get('network_failures_summary') or {}).get('classes'),
        'n_esso_recovery_events': (rec.get('network_failures_summary') or {}).get('n_esso_recovery_events'),
        'wall_child_process_s': (rec.get('wall_time_s') or {}).get('child_process_s'),
        'wall_launcher_s': res.get('wall_clock_s'),
        'peak_rss_child_bytes': (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss'),
        'rho_read_back': 'not present (the screen declares no rho0_scale; the read-back is an E2 check)',
        'rho_terminal_after': {g: last.get(f'rho_{g}_after') for g in ('v', 'pf', 'ess')},
        'rho_terminal_action': {g: last.get(f'rho_{g}_action') for g in ('v', 'pf', 'ess')},
        'git_head_at_run': res.get('git_head_at_run'), 'results_utc': res.get('utc'),
        'comparator_frozen_in_cell_spec': (spec.get('extra') or {}).get('comparator'),
        'checks': checks, 'checks_all_true': all(checks.values()),
    }


def verdict_v6(margin, bar_a, bar_b):
    det, thr, binds = SC6.determinate_certified(margin, bar_a, bar_b)
    v = ('neighbour better' if margin < 0 else 'x0 better') if det else 'within resolution'
    return {'threshold_eur': thr, 'binds': binds, 'ratio_abs_margin_over_threshold': abs(margin) / thr,
            'determinate': det, 'verdict': v}


def verdict_frozen(margin, bar_a, bar_b, sigma_q, f_x0, f_n):
    res = PB.resolution(bar_a, bar_b, sigma_q)
    det = abs(margin) > res
    v = ('neighbour better' if margin < 0 else 'x0 better') if det else 'within resolution'
    return {'threshold_eur': res, 'binds': 'bar sum' if bar_a + bar_b >= sigma_q else 'sigma_Q',
            'ratio_abs_margin_over_threshold': abs(margin) / res, 'determinate': det, 'verdict': v,
            'pb_classify_of_neighbour_vs_x0': PB.classify(f_x0, f_n, res)}


def compare(x0, n, sigma_q):
    dq = n['Q_gross_eur'] - x0['Q_gross_eur']
    df = dq + n['I_eur'] - x0['I_eur']
    df_net = df - (n['terminal_salvage_eur'] - x0['terminal_salvage_eur'])
    f_x0, f_n = x0['Q_gross_eur'] + x0['I_eur'], n['Q_gross_eur'] + n['I_eur']
    out = {'cell': n['cell'], 'eval_key': n['eval_key'], 'candidate_key': n['candidate_key'],
           'lattice_label': n['lattice_label'], 'n_storage_nodes': n['n_storage_nodes'],
           'I_eur': n['I_eur'], 'salvage_eur': n['terminal_salvage_eur'],
           'bar_n_eur': n['bar_eur'], 'bar_x0_eur': x0['bar_eur'],
           'dQ_eur': dq, 'dF_eur': df, 'dF_net_eur_informational': df_net,
           'T_band_eur': None, 'T_band_note': 'not computable: no rule-v6 band at production exit'}
    for qname, m in (('dQ', dq), ('dF', df)):
        out[f'{qname}__T_v6bar'] = verdict_v6(m, n['bar_eur'], x0['bar_eur'])
        out[f'{qname}__T_frozen'] = verdict_frozen(m, n['bar_eur'], x0['bar_eur'], sigma_q,
                                                   x0['Q_gross_eur'] if qname == 'dQ' else f_x0,
                                                   n['Q_gross_eur'] if qname == 'dQ' else f_n)
    out['dF_net__T_v6bar_informational'] = verdict_v6(df_net, n['bar_eur'], x0['bar_eur'])
    return out


def p3(comparisons, reading):
    vs = [c[reading]['verdict'] for c in comparisons]
    c1 = 'neighbour better' not in vs
    c2 = 'within resolution' in vs
    return {'n': len(vs), 'n_neighbour_better': vs.count('neighbour better'), 'n_x0_better': vs.count('x0 better'),
            'n_within_resolution': vs.count('within resolution'),
            'clause_1_no_neighbour_beats_x0_held': c1, 'clause_2_at_least_one_within_resolution_held': c2,
            'P3_held': c1 and c2,
            'neighbours_better': [c['cell'] for c in comparisons if c[reading]['verdict'] == 'neighbour better'],
            'within_resolution': [c['cell'] for c in comparisons if c[reading]['verdict'] == 'within resolution']}


def holders(x0, sigma_q):
    out = {}
    w86_spec = _read(W86['spec'])
    w86_res = _read(W86['results'])
    w86 = _read(W86['record'])
    w86_rows = _read(W86['per_cycle'], jsonl=True)
    x0_rows = _read(os.path.join(os.path.dirname(x0['_record_path']), 'per_cycle_record.jsonl'), jsonl=True)
    fields = sorted(set().union(*[r.keys() for r in x0_rows]) & set().union(*[r.keys() for r in w86_rows]))
    rows_equal = len(x0_rows) == len(w86_rows) and all(
        all(a.get(k) == b.get(k) for k in fields) for a, b in zip(x0_rows, w86_rows))
    first_diff = next(({'cycle': a.get('cycle'), 'fields': [k for k in fields if a.get(k) != b.get(k)]}
                       for a, b in zip(x0_rows, w86_rows) if any(a.get(k) != b.get(k) for k in fields)), None)
    w86_rt = w86.get('rule_ten') or {}
    d = x0['Q_gross_eur'] - w86['certified_cost']
    out['W86_tail_recert'] = {
        'spec': W86['spec'], 'spec_sha256': INPUTS[W86['spec']], 'spec_campaign_id': w86_spec.get('campaign_id'),
        'record': W86['record'], 'eval_key': w86.get('eval_key'), 'candidate_key': w86.get('candidate_key'),
        'eval_key_equals_screen_x0': w86.get('eval_key') == x0['eval_key'],
        'git_head_at_run': w86_res.get('git_head_at_run'), 'status': w86.get('status'),
        'protocol': 'production exit (ten consecutive passing cycles, cap 500), tight tail enabled',
        'Q_gross_eur': w86['certified_cost'], 'cycles_run': w86.get('cycles_run'),
        'bar_eur': (w86.get('bar') or {}).get('value'), 'terminal_step_eur': w86_rt.get('terminal_objective_change_abs'),
        'stopping_ratio': w86_rt.get('terminal_step_over_threshold'),
        'salvage_eur': (w86.get('recourse_components') or {}).get('terminal_salvage_value'),
        'Q_screen_x0_minus_Q_W86_eur': d, 'bitwise_equal_Q': d == 0.0,
        'per_cycle_rows_equal_on_common_fields': rows_equal, 'per_cycle_n_rows': [len(x0_rows), len(w86_rows)],
        'per_cycle_common_fields': fields, 'per_cycle_first_difference': first_diff,
        'resolution_terminal_step_sum_eur': x0['terminal_step_eur'] + w86_rt.get('terminal_objective_change_abs'),
        'resolution_frozen_eur': PB.resolution(x0['bar_eur'], (w86.get('bar') or {}).get('value'), sigma_q),
    }
    c_res = _read(C_OPEN['results'])
    c_rec = _read(C_OPEN['record'], C_OPEN['record_sha256'])
    tables = _read(STEP6_TABLES['path'], STEP6_TABLES['sha256'])
    row = [c for c in tables['tables']['cells'].values() if c.get('eval_key') == C_OPEN['eval_key']]
    if len(row) != 1:
        raise SystemExit(f'REFUSED: {len(row)} step-6 table rows carry the C_OPEN eval key (1 required)')
    row = row[0]
    c_rt = c_rec.get('rule_ten') or {}
    d = x0['Q_gross_eur'] - row['Q']
    out['C_OPEN_E21_row1'] = {
        'record': C_OPEN['record'], 'eval_key': c_rec.get('eval_key'), 'candidate_key': c_rec.get('candidate_key'),
        'candidate_key_equals_screen_x0': c_rec.get('candidate_key') == x0['candidate_key'],
        'git_head_at_run': c_res.get('git_head_at_run'), 'status': row.get('status'),
        'protocol': 'rule v6 certified (settled)', 'step6_table_row': row.get('cell'), 'step6_name': row.get('name'),
        'Q_gross_eur': row['Q'], 'Q_equals_record_certified_cost': row['Q'] == c_rec.get('certified_cost'),
        'k_star': row.get('k_star'), 'end_cycle': row.get('end_cycle'), 'band_width_eur': row.get('band_width'),
        'salvage_eur': row.get('salvage'), 'record_bar_eur': (c_rec.get('bar') or {}).get('value'),
        'terminal_step_eur': c_rt.get('terminal_objective_change_abs'),
        'stopping_ratio': c_rt.get('terminal_step_over_threshold'),
        'Q_screen_x0_minus_Q_C_OPEN_eur': d,
        'threshold_v6_screen_bar_vs_C_OPEN_band_eur': SC6.determinacy_threshold(x0['bar_eur'], row['band_width']),
        'ratio_v6': abs(d) / SC6.determinacy_threshold(x0['bar_eur'], row['band_width']),
        'threshold_frozen_bar_sum_eur': PB.resolution(x0['bar_eur'], (c_rec.get('bar') or {}).get('value'), sigma_q),
        'ratio_frozen': abs(d) / PB.resolution(x0['bar_eur'], (c_rec.get('bar') or {}).get('value'), sigma_q),
        'resolution_terminal_step_sum_eur': x0['terminal_step_eur'] + c_rt.get('terminal_objective_change_abs'),
        'note': ('mixed comparison: a production-exit value (stopped at ten passing cycles) against a rule-v6 settled '
                 'value on a different code head; informational, never a comparator (plan 5.3)'),
    }
    return out


def main_run(trial_out=None):
    started = datetime.now(timezone.utc)
    out_json = os.path.join(trial_out, 'w187_screen_report_TRIAL.json') if trial_out else OUT_JSON
    if os.path.exists(out_json):
        raise SystemExit(f'REFUSED: {out_json} exists (write-once)')
    _log(f'HEAD {_git(["rev-parse", "HEAD"]).strip()}; guard installed; pickle and model builds blocked'
         + (f'; TRIAL output {out_json}' if trial_out else ''))
    if not trial_out and not _committed_clean(os.path.relpath(os.path.abspath(__file__), REPO)):
        raise SystemExit('REFUSED: this script is not committed and clean')
    stage = _read(STAGE_SPEC['path'], STAGE_SPEC['sha256'])
    _read(SPEC_V1_1['path'], SPEC_V1_1['sha256'])
    missing = capture_checklist(stage)
    if missing:
        raise SystemExit(f'REFUSED: capture paths missing: {missing}')
    _log(f"capture checklist: {len(stage['cell_order'])} cells, every path present")
    lattice, lat_ev = build_lattice()
    sigma_q = lat_ev['sigma_Q']['sigma_Q_eur']
    cells = {}
    for cell in stage['cell_order']:
        v = cell_view(stage, cell, lattice)
        root = _cell_root(stage, cell)
        with open(os.path.join(SCREEN_ROOT, stage['cells'][cell]['campaign_spec'])) as f:
            v['_record_path'] = os.path.join(_eval_dir(root, json.load(f)), 'evaluation_record.json')
        cells[cell] = v
        _log(f"{cell}: {v['status']} Q {v['Q_gross_eur']:.2f} bar {v['bar_eur']:.2f} cycles {v['cycles_run']} "
             f"ratio {v['stopping_ratio']:.3e} checks {v['checks_all_true']}")
    x0 = cells[X0_CELL]
    nbrs = [c for c in stage['cell_order'] if c != X0_CELL]
    z_nbrs = sorted(tuple(cells[c]['z']) for c in nbrs)
    neighbourhood = sorted(lattice.neighbourhood(lattice.x0()))
    comps = [compare(x0, cells[c], sigma_q) for c in nbrs]
    readings = ['dQ__T_v6bar', 'dQ__T_frozen', 'dF__T_v6bar', 'dF__T_frozen']
    p3_out = {r: {'all_14': p3(comps, r), 'single_node_6_informational':
                  p3([c for c in comps if c['n_storage_nodes'] == 1], r)} for r in readings}
    hold = holders(x0, sigma_q)
    for v in cells.values():
        v.pop('_record_path', None)
    all_checks = all(v['checks_all_true'] for v in cells.values())
    guard_fail = GUARD.verify(0)
    pb_fail = PB.PARENT_GUARD.verify(0)
    payload = {
        'schema': 'p515_s55_w187_screen_report_v1', 'stage': 'P5.15 W187 report (Addendum 91, amendments 1-3)',
        'script': os.path.basename(__file__), 'script_sha256': _sha(os.path.abspath(__file__)),
        'git_head': _git(['rev-parse', 'HEAD']).strip(), 'utc': started.isoformat(),
        'objective_convention': OBJECTIVE_CONVENTION,
        'definitions': {
            'margin': 'm = value(neighbour) - value(x0_A_1x); m < 0: the neighbour is cheaper',
            'dQ': 'Q(n) - Q(x0), Q = gross_operational_cost settlement-excluded (as ordered)',
            'dF': 'F(n) - F(x0) = dQ + I(n) - I(x0), I from the pinned W2 table via p515_s47_phase_b_record.Lattice',
            'dF_net_informational': 'dF - (salvage(n) - salvage(x0))',
            'T_v6bar': 'settling_criterion_v6.determinacy_threshold(bar_n, bar_x0) = max(3 * max(bar_n, bar_x0), '
                       f'2 * TAU), TAU = {SC6.TAU}; b read as the screen bar (no band at production exit)',
            'T_frozen': 'p515_s47_phase_b_record.resolution(bar_n, bar_x0, sigma_Q) = max(bar_n + bar_x0, sigma_Q), '
                        f'sigma_Q = {sigma_q} (the comparator frozen in every screen cell spec)',
            'T_band': 'max(3 * larger band width, 2 TAU): not computable at production exit',
            'bar': 'the record bar: max over the last 10 cycles of |gross[k] - gross[k-1]|',
            'verdict_v6': 'settling_criterion_v6.determinate_certified: |m| >= T -> sign verdict, else within resolution',
            'verdict_frozen': '|m| > T -> sign verdict, else within resolution; PB.classify(F(x0), F(n), T) beside it',
            'stopping_ratio': 'rule_ten.terminal_step_over_threshold = |dQ| at the last cycle / production objective '
                              'tolerance (1e-4 of Q)',
            'P3': 'clause 1: no neighbour with verdict "neighbour better"; clause 2: at least one "within resolution"',
            'unit_neighbour': ('the stage spec phase_text and Addendum 85 ruling 4 (vii): "the 14 admissible unit '
                               'neighbours of x = 0 under N"; plan 5.3: "the smallest unit at any non-empty subset of '
                               'the three nodes, 2025 or 2030"'),
        },
        'TAU_eur': SC6.TAU, 'sigma_Q_eur': sigma_q, 'lattice': {
            'I_x_cross_check_vs_W2': lat_ev['I_x_cross_check_vs_W2'],
            'unit_costs_discounted': {str(k): v for k, v in lat_ev['unit_costs'].items()},
            'x0_lattice_neighbourhood_size': len(neighbourhood),
            'screen_neighbours_equal_lattice_neighbourhood_of_x0': z_nbrs == [tuple(z) for z in neighbourhood]},
        'stage_spec': STAGE_SPEC, 'spec_v1_1': SPEC_V1_1, 'cell_order': stage['cell_order'],
        'cells': cells, 'comparisons_vs_x0': comps, 'P3': p3_out, 'holders_x0_1x': hold,
        'cell_checks_all_true': all_checks,
        'launcher_summary_line_quirk': ('the launcher prints rep.get(k_star / band_width / Q_end / harness_bar_eur / '
                                        'stopping_ratio_...) from outcome["cell_report"], which exists only for the '
                                        'certified phases (e3, e2); a screen outcome carries outcome["screen"] '
                                        '(status, certified_cost, cycles_run, bar_eur, rule_ten), so every value prints '
                                        'None (p515_s55_step7_launcher.py cell_outcome and the [RUN] _log line)'),
        'inputs_sha256': dict(sorted(INPUTS.items())),
        'guards': {'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_fail},
                   'pb_parent_guard': {'counts': dict(PB.PARENT_GUARD.counts), 'verify_0_failures': pb_fail},
                   'pickle_calls': dict(PICKLE_COUNTS), 'pyomo_model_init_calls': dict(BUILD_COUNTS),
                   'all_zero': (not guard_fail and not pb_fail and not any(PICKLE_COUNTS.values())
                                and not any(BUILD_COUNTS.values()))},
    }
    payload['trial'] = bool(trial_out)
    text = GRIO.dumps(payload, indent=1, sort_keys=False)
    with open(out_json, 'x') as f:
        f.write(text + '\n')
    _log(f'wrote {out_json} sha256 {_sha(out_json)}')
    for r in readings:
        a = p3_out[r]['all_14']
        _log(f"P3 [{r}] 14: better {a['n_neighbour_better']} x0 {a['n_x0_better']} within {a['n_within_resolution']} "
             f"-> clause1 {a['clause_1_no_neighbour_beats_x0_held']} clause2 "
             f"{a['clause_2_at_least_one_within_resolution_held']}")
    _log(f"cell checks all true {all_checks}; guards all_zero {payload['guards']['all_zero']}")
    return 0 if (all_checks and payload['guards']['all_zero']) else 1


def main_post_run():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
    with open(OUT_JSON) as f:
        payload = json.load(f)
    files = {os.path.relpath(os.path.abspath(__file__), REPO): None, OUT_JSON: None, OUT_LOG: None}
    files.update({k: None for k in payload['inputs_sha256']})
    manifest = {p: _sha(p) for p in sorted(files)}
    bad = [p for p, s in payload['inputs_sha256'].items() if manifest[p] != s]
    if bad:
        raise SystemExit(f'REFUSED: inputs changed since the run: {bad}')
    with open(OUT_MANIFEST, 'x') as f:
        f.write(GRIO.dumps(manifest, indent=1) + '\n')
    _log(f'wrote {OUT_MANIFEST} ({len(manifest)} files); guard counts {GUARD.counts}; verify {GUARD.verify(0)}')
    return 0 if not GUARD.verify(0) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out', default=None, help='a directory OUTSIDE the repository for a trial (no record)')
    a = ap.parse_args()
    if a.trial_out and os.path.realpath(a.trial_out).startswith(REPO + os.sep):
        raise SystemExit('REFUSED: --trial-out must be outside the repository')
    sys.exit(main_post_run() if a.post_run else main_run(a.trial_out))
