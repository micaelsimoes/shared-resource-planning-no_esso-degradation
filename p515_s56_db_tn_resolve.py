"""P5.15 S56 -- D_b causal split: TN-only re-solve at the pinned AA-off arm's realized DN exchanges (step 4, both
calibration blocks + the merged-only 2025|Summer block, reported, not deciding). Exactly 6 IPOPT solves.

Authority: TASKS.md "Expert rulings on handback 3 (2026-11-23)" item 5 (D_b mapping not confirmed; causal split needs a
TN-only re-solve at the pinned AA-off arm's realized DN exchanges, both blocks, guarded, frozen spec).

Definitions (EUR of total discounted cost; every quantity per block b; w_b = production `_get_admm_block_weight`):
  w_b C_central          step 3 gross comparator (frozen step 4 spec `ii_central_and_allowance`)
  Q_pin,b                pinned AA-off arm, terminal cycle K, contracted-settlement-excluded convention
                         (s56_cycle_capture.jsonl q_blocks.per_year_day[b].contracted_excluded -- the value step 4 used)
                         = w_b [TSO base + sum_DN (DSO base + DSO covariance settlement)]  (recomputed and checked)
  C_TN-resolve,b         w_b [C_TN,B,b + sum_DN DSO base_b]: C_TN,B,b = the TN-only model objective (settlement weight 0,
                         no pin, no row 18, no ADMM terms) with every ADN pc/qc fixed PER SCENARIO at the DN's realized
                         exchange of that scenario (Arm B of D0); DSO base_b = the pinned arm's own DN operating cost
                         (multiscenario_terminal per_scenario_cost_checks.expected_base_objective: sum_s omega_s obj_s, no
                         settlement, no pin) -- the C_central convention (model.objective with settlement weight 0).
  pinned-accounting term T1_b = C_TN-resolve,b - Q_pin,b          (sign empirical)
  DN re-optimization gain T2_b = w_b C_central - C_TN-resolve,b   (<= 0 only up to local optimality)
  T1_b + T2_b = D_b = w_b C_central - Q_pin,b                      (identity, checked at the adopted M1 tolerance)
  Beside (a reading, not a decomposition): T1_b = w_b (C_TN,A,b - TSO base_b) + w_b [(C_TN,B,b - C_TN,A,b) - cov_b],
  C_TN,A = the same TN model with the ADN pc/qc fixed at the TSO's scenario-common exchange (D0 Arm A, the control).

Machinery re-used, not re-implemented (D0 v2, spec precedent 72125c3b): `p515_s56_tn_realized_resolve.py` (v1) block_data,
build_arm, model_gates, extract, max_violation, configure; `p515_s56_tn_realized_resolve_v2.py` build_arm_v2 /
isolate_shared_ess (x0: inert). Production reader `p515_s44_scale_measurement.read_planning_from_derived_case`, production
Network.run_smopf, production success test `helper_functions.solver_result_succeeded` (nothing extracted from a rejected
solve; every status asserted).

ISOLATION. Run from a clean `git archive` export of 4c586e5e with this script and its spec overlaid from the commit that
froze them (EXPORT_PROVENANCE.json at the export root, written by the export step: export commit, overlay commit, sha256
of every exported root-level file and of the overlay). The working directory is the export, so relative solver logs land
there. Outputs are copied into the main checkout afterwards.

GUARDS. SolveProfileGuard(permitted=[(this file, '_solve_one')]) installed before any project import; the guards the
imported D0 modules install at import are uninstalled right after (asserted by identity); verified at exactly 6 solves and
6 executions (0 under --trial or a pre-solve refusal), imported guard counts all 0; pickle.load/loads blocked, verified 0.
Refuses if another copy of this script runs; the output directory must not exist; stdout/stderr captured by the caller.

Usage (from the export root):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_db_tn_resolve.py --spec SPEC --out-dir DIR [--trial]
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from pyomo.opt.base.solvers import OptSolver  # noqa: E402
from pyomo.opt.solver.shellcmd import SystemCallSolver  # noqa: E402

GUARD = SolveProfileGuard(permitted=((SCRIPT_REL, '_solve_one'),),
                          label='P5.15 S56 D_b causal split: TN-only re-solve at realized DN exchanges').install()
_MY_SOLVE, _MY_EXEC = OptSolver.solve, SystemCallSolver._execute_command

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (this stage reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import p515_s56_tn_realized_resolve_v2 as V2  # noqa: E402  (installs V2.GUARD; V2 uninstalls the v1 / post-check guards)

V2.GUARD.uninstall()
assert OptSolver.solve is _MY_SOLVE and SystemCallSolver._execute_command is _MY_EXEC, \
    'this script guard not restored after uninstalling the imported D0 v2 guard'
pickle.load, pickle.loads = _blk('load'), _blk('loads')   # V2 / v1 re-armed their own blockers: re-arm this script's
H = V2.H

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import helper_functions as HF  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

CHECKS = H.CHECKS            # one check list (v1 helpers append to it)
U = 2.0 ** -53               # unit roundoff (binary64)
PROVENANCE_FILE = 'EXPORT_PROVENANCE.json'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [Db] {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))
    return bool(ok)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def own_copies_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found, others = os.getpid(), [], []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python'):
            names = [os.path.basename(t) for t in tok[2:4]]
            if SCRIPT_REL in names:
                found.append(ln.strip()[:300])
            elif any(n.startswith('p5') and n.endswith('.py') for n in names):
                others.append(ln.strip()[:300])
    return found, others


def provenance(spec_path, trial):
    """The export record: export commit, overlay commit, sha256 of the exported root files and the overlay; this script,
    its spec and every imported root-level module are checked against it."""
    p = os.path.join(REPO, PROVENANCE_FILE)
    if not os.path.exists(p):
        return None, ([] if trial else [f'{PROVENANCE_FILE} missing (not a provenance-recorded export)'])
    prov = json.load(open(p))
    bad = []
    files = {**prov.get('exported_root_files_sha256', {}), **{k: v['sha256'] for k, v in prov.get('overlay', {}).items()}}
    rel_spec = os.path.relpath(os.path.abspath(spec_path), REPO)
    for rel in (SCRIPT_REL, rel_spec):
        if files.get(rel) != _sha(os.path.join(REPO, rel)):
            bad.append(f'{rel}: sha256 differs from the export record')
    mods = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.dirname(os.path.abspath(f)) == REPO and f.endswith('.py'):
            rel = os.path.basename(f)
            sha = _sha(f)
            mods[rel] = sha
            if files.get(rel) != sha:
                bad.append(f'imported module {rel}: sha256 differs from the export record')
    return {'export_commit': prov.get('export_commit'), 'overlay_commit': prov.get('overlay_commit'),
            'provenance_file_sha256': _sha(p), 'imported_root_modules_sha256': mods}, ([] if trial else bad)


# --------------------------------------------------------------------------------------------------- solve (guarded)
def _solve_one(net, m, params):
    """The ONE permitted solve site. Production Network.run_smopf from the model's own values; loaded = production
    `solver_result_succeeded` on the returned result (as D0 v2 `_solve_one`)."""
    n_rec = len(getattr(net, 'ipopt_solve_records', []) or [])
    t0 = time.perf_counter()
    result = net.run_smopf(m, params, from_warm_start=False, print_header=True)
    wall = time.perf_counter() - t0
    recs = list(getattr(net, 'ipopt_solve_records', []) or [])[n_rec:]
    loaded = bool(HF.solver_result_succeeded(result))
    ex = (recs or [{}])[-1]
    tc = str(result.solver.termination_condition) if result is not None and hasattr(result, 'solver') else None
    return {'wall_s': wall, 'ipopt_records': recs, 'termination_condition': tc,
            'solver_status': str(result.solver.status) if result is not None and hasattr(result, 'solver') else None,
            'result_returned': result is not None, 'solver_result_summary': HF.solver_result_summary(result),
            'loaded_by_production': loaded,
            'strict_optimal': loaded and tc == 'optimal' and ex.get('exit') == 'Optimal Solution Found.'}


# ------------------------------------------------------------------------------------------------ inputs (no solve)
def load_arm(inst):
    """The pinned AA-off arm's terminal records, each sha256-checked against the child manifest pin."""
    edir = inst['eval_dir']
    man = json.load(open(os.path.join(edir, 'child_manifest_sha256.json')))
    rec = {}
    for fname in ('multiscenario_terminal.json', 'evaluation_record.json', 's56_cycle_capture.jsonl'):
        p = os.path.join(edir, fname)
        r = {'path': p, 'sha256': _sha(p), 'pinned_sha256_child_manifest': man.get(p)}
        r['pinned_ok'] = r['sha256'] == r['pinned_sha256_child_manifest']
        _ck(f"{inst['label']}: {fname} sha256 == child manifest pin", r['pinned_ok'], r)
        rec[fname] = r
    rec['child_manifest_sha256.json'] = {'path': os.path.join(edir, 'child_manifest_sha256.json'),
                                         'sha256': _sha(os.path.join(edir, 'child_manifest_sha256.json'))}
    ms = json.load(open(os.path.join(edir, 'multiscenario_terminal.json')))
    er = json.load(open(os.path.join(edir, 'evaluation_record.json')))
    cap = _jsonl(os.path.join(edir, 's56_cycle_capture.jsonl'))
    cap_lines = [x for x in cap if not x.get('header')]
    _ck(f"{inst['label']}: candidate_canonical == spec (x0)", er['candidate_canonical'] == inst['candidate_canonical'],
        er['candidate_canonical'])
    _ck(f"{inst['label']}: certified, terminal == certification cycle == spec K",
        er['status'] == 'certified' and er.get('certification_cycle') == er['cycles_run'] == inst['K'],
        {'status': er['status'], 'cycles_run': er['cycles_run'], 'K_spec': inst['K']})
    _ck(f"{inst['label']}: eval key == spec", er.get('eval_key') == inst['eval_key'], er.get('eval_key'))
    _ck(f"{inst['label']}: capture header arm pinned, primary convention contracted_excluded",
        cap[0].get('capture_arm') == 'pinned' and cap[0].get('q_primary_convention') == 'contracted_excluded',
        {k: cap[0].get(k) for k in ('capture_arm', 'q_primary_convention')})
    last = cap_lines[-1]
    _ck(f"{inst['label']}: last capture line = cycle K, captured", last['cycle'] == inst['K'] and last['captured'],
        {'cycle': last['cycle'], 'captured': last['captured']})
    return {'edir': edir, 'inputs': rec, 'ms': ms, 'er': er, 'cap_lines': cap_lines}


def block_accounting(lab, bk, ms, cap_lines, w):
    """Q_pin from the capture (the step 4 value) and its recomputation from the terminal multiscenario record; the DSO
    base and covariance parts; the last-10-cycle steps of each part (resolution)."""
    y, d = bk.split('|')
    tb = ms['blocks'][f'TSO|{y}|{d}']
    dso = {k.split('|')[1]: v for k, v in ms['blocks'].items() if k.startswith('DSO|') and k.endswith(f'|{y}|{d}')}
    last = cap_lines[-1]
    q_pin = last['q_blocks']['per_year_day'][bk]['contracted_excluded']
    pa = last['q_blocks']['per_agent']
    tso_base = tb['block_recourse_production_local']
    dn_base = {dn: b['per_scenario_cost_checks']['expected_base_objective'] for dn, b in dso.items()}
    dn_cov = {dn: b['settlement']['deviation'] for dn, b in dso.items()}
    dn_local = {dn: b['block_recourse_production_local'] for dn, b in dso.items()}
    for dn in dso:
        _ck(f'{lab} {bk} DSO{dn}: base + covariance == block_recourse_production_local (rel <= 1e-12)',
            H._rel(dn_base[dn] + dn_cov[dn], dn_local[dn]) <= 1e-12, [dn_base[dn], dn_cov[dn], dn_local[dn]])
        cap_dso = pa[f'DSO{dn}|{y}|{d}']
        _ck(f'{lab} {bk} DSO{dn}: capture contracted_excluded == w x block local (rel <= 1e-12)',
            H._rel(cap_dso['contracted_excluded'], w * dn_local[dn]) <= 1e-12, [cap_dso['contracted_excluded'], w * dn_local[dn]])
        _ck(f'{lab} {bk} DSO{dn}: capture settlement_excluded_in_full == w x base (rel <= 1e-12)',
            H._rel(cap_dso['settlement_excluded_in_full'], w * dn_base[dn]) <= 1e-12,
            [cap_dso['settlement_excluded_in_full'], w * dn_base[dn]])
    _ck(f'{lab} {bk}: capture TSO contracted_excluded == w x TSO block local (rel <= 1e-12)',
        H._rel(pa[f'TSO|{y}|{d}']['contracted_excluded'], w * tso_base) <= 1e-12,
        [pa[f'TSO|{y}|{d}']['contracted_excluded'], w * tso_base])
    q_re = w * (tso_base + sum(dn_base.values()) + sum(dn_cov.values()))
    _ck(f'{lab} {bk}: Q_pin(capture) == w (TSO base + sum DSO base + sum cov) (rel <= 1e-12)',
        H._rel(q_pin, q_re) <= 1e-12, [q_pin, q_re])
    # resolution of the DN base part and of Q_pin: max |step| over the last 10 cycles (the step 4 bar convention)
    def steps(series):
        cyc = sorted(series)[-11:]
        return max(abs(series[cyc[i]] - series[cyc[i - 1]]) for i in range(1, len(cyc)))
    s_q = {x['cycle']: x['q_blocks']['per_year_day'][bk]['contracted_excluded'] for x in cap_lines if x.get('captured')}
    s_dn = {x['cycle']: sum(v['settlement_excluded_in_full'] for k, v in x['q_blocks']['per_agent'].items()
                            if k.startswith('DSO') and k.endswith(f'|{y}|{d}')) for x in cap_lines if x.get('captured')}
    s_tso = {x['cycle']: x['q_blocks']['per_agent'][f'TSO|{y}|{d}']['contracted_excluded']
             for x in cap_lines if x.get('captured')}
    return {'Q_pin': q_pin, 'Q_pin_recomputed': q_re, 'tso_base_eur_per_day': tso_base,
            'dn_base_eur_per_day': dn_base, 'dn_cov_eur_per_day': dn_cov, 'dn_local_eur_per_day': dn_local,
            'dn_base_sum_eur_per_day': sum(dn_base.values()), 'dn_cov_sum_eur_per_day': sum(dn_cov.values()),
            'last10_max_step_Q_pin': steps(s_q), 'last10_max_step_w_dn_base': steps(s_dn),
            'last10_max_step_w_tso_base': steps(s_tso)}


def interface_voltage_mismatch(r_arm, ms, bk, dn_ids, side):
    """max |V_TN-resolve(ADN bus) - V_saved(side)| over scenarios and hours, per DN (p.u.); side 'DSO' = the DN's own
    root voltage in the pinned arm's terminal state, 'TSO' = the TSO block's interface voltage."""
    y, d = bk.split('|')
    out = {}
    for dn in dn_ids:
        if side == 'DSO':
            prof = ms['blocks'][f'DSO|{dn}|{y}|{d}']['interface_profiles']['per_scenario']
        else:
            prof = ms['blocks'][f'TSO|{y}|{d}']['interface_profiles']['per_scenario'][str(dn)]
        diffs = [abs(r_arm['hours'][k][t]['adn'][str(dn)]['vmag_pu'] - prof[k]['v'][t])
                 for k in r_arm['hours'] for t in range(len(r_arm['hours'][k]))]
        out[str(dn)] = {'max_abs_pu': max(diffs), 'mean_abs_pu': sum(diffs) / len(diffs)}
    return out


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    a = ap.parse_args()
    t_start = time.perf_counter()
    pre = []
    spec_sha = _sha(a.spec)
    tag = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not spec_sha.startswith(tag):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
    prov, bad = provenance(a.spec, a.trial)
    pre += bad
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    mine, others = own_copies_running()
    if mine:
        pre.append(f'another copy of {SCRIPT_REL} runs: {mine}')
    if pre:
        print('[Db] PRECONDITION FAILED:', pre, flush=True)
        return 1
    return run(a, spec_sha, prov, others, t_start)


def run(a, spec_sha, prov, others, t_start):
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    # capture checklist, asserted before anything is read or solved (CLAUDE.md rule eleven)
    capture = {'Q_pin': 'block_accounting', 'w_C_central': 'spec', 'D_b': 'statistics',
               'C_TN_resolve': 'statistics', 'C_TN_B_objective': 'H.extract', 'C_TN_A_objective': 'H.extract',
               'T1_pinned_accounting': 'statistics', 'T2_DN_reoptimization_gain': 'statistics',
               'B3_identity': 'statistics', 'status_iterations_time': '_solve_one',
               'production_success_test': '_solve_one', 'max_constraint_violation': 'H.max_violation',
               'slack_detector_terms': 'H.extract', 'interface_voltage_mismatch': 'interface_voltage_mismatch',
               'dso_covariance_per_block': 'H.block_data', 'resolution_last10_steps': 'block_accounting',
               'control_reading_beside': 'statistics', 'predictions': 'predictions'}
    missing = [k for k in spec['reported'] if k not in capture]
    assert not missing, f'spec quantities without a capture path: {missing}'
    for k, fn in capture.items():
        mod, _, name = fn.rpartition('.')
        assert (hasattr(H, name) if mod == 'H' else (name in globals() or name == 'spec')), (k, fn)
    os.makedirs(a.out_dir)
    expected = 0 if a.trial else spec['guards']['expected_solves']
    res = {'stage': spec['stage'], 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'provenance': prov, 'run_directory': REPO,
           'utc_start': datetime.now(timezone.utc).isoformat(), 'trial': a.trial,
           'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'lp_solver_path': os.environ.get('LP_SOLVER_PATH'), 'dotenv_present_in_export': os.path.exists('.env'),
           'other_p5_python_processes_at_launch': others,
           'objective_convention': spec['objective_convention'], 'instances': {}}
    res['modules'] = {p: {'path': p, 'sha256': _sha(p)} for p in (
        'network.py', 'network_data.py', 'model_construction_helpers.py', 'shared_resources_planning.py',
        'helper_functions.py', 'p515_s44_scale_measurement.py', 'p515_s56_tn_realized_resolve.py',
        'p515_s56_tn_realized_resolve_v2.py', 'p515_s56_tn_realized_postcheck.py', spec['step4']['spec_path'],
        spec['step4']['evaluation_path'])}
    s4 = json.load(open(spec['step4']['spec_path']))
    ev4 = json.load(open(spec['step4']['evaluation_path']))
    _ck('step 4 spec sha256 == spec', _sha(spec['step4']['spec_path']) == spec['step4']['spec_sha256'])
    _ck('step 4 evaluation sha256 == spec', _sha(spec['step4']['evaluation_path']) == spec['step4']['evaluation_sha256'])
    central = s4['ii_central_and_allowance']['blocks']
    ctx = {}
    for inst in spec['instances']:
        lab = inst['label']
        _ck(f'{lab}: case sha256 == spec', _sha(inst['case_path']) == inst['case_sha256'])
        cs = json.load(open(inst['campaign_spec_path']))
        di = cs['configuration']['derived_instance']
        _ck(f'{lab}: case == the campaign spec declaration (path, sha256, checksum)',
            di['case_path'] == inst['case_path'] and di['case_sha256'] == inst['case_sha256']
            and di['scenario_checksum'] == inst['scenario_checksum'])
        cand_spec = [c for c in cs['candidates'] if c['eval_dir'] == os.path.basename(inst['eval_dir'])]
        _ck(f'{lab}: eval dir is the campaign\'s pinned AA-off arm (AA off, no per-scenario coupling)',
            len(cand_spec) == 1 and cand_spec[0]['label'] == 'x0_pinned_aa_off_main'
            and cand_spec[0]['effective_anderson_acceleration']['enabled'] is False
            and 'interface_coupling' not in cand_spec[0], cand_spec[0] if cand_spec else None)
        cd = load_arm(inst)
        planning = H.S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                                       os.path.join(a.out_dir, f'read_{lab}'), H._Stages())
        checksum = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
        _ck(f'{lab}: combined scenario checksum == declaration', checksum == inst['scenario_checksum'], checksum)
        dn_ids = list(planning.transmission_network.active_distribution_network_nodes)
        _ck(f'{lab}: TN ADN nodes == [5, 7, 9]', dn_ids == [5, 7, 9], dn_ids)
        params_solve = H.configure(planning.transmission_network.params, spec)
        tn = planning.transmission_network
        blocks_inst = sorted(f'{int(y)}|{d}' for y in tn.years for d in tn.days)
        _ck(f'{lab}: instance blocks == spec', blocks_inst == sorted(inst['blocks']), blocks_inst)
        cand = planning.get_initial_candidate_solution()
        iy = H._year_key(planning.shared_ess_data.years, inst['candidate_canonical']['investment_year'])
        for n, (s_val, e_val) in inst['candidate_canonical']['nodes'].items():
            cand['investment'][int(n)][iy]['s'] = float(s_val)
            cand['investment'][int(n)][iy]['e'] = float(e_val)
        srp._rebuild_candidate_total_capacities(planning, cand)
        tn.update_data_with_candidate_solution(cand['total_capacity'])
        avail = {}
        for n in dn_ids:
            pub = cd['er']['storage_per_node'][str(n)]['published_available_capacity_terminal']
            avail[n] = {y: {'s': pub[str(int(y))]['s_available'], 'e': pub[str(int(y))]['e_available']} for y in tn.years}
        bdata, acc = {}, {}
        for year in tn.years:
            for day in tn.days:
                bk = f'{int(year)}|{day}'
                bdata[(str(int(year)), day)] = H.block_data(lab, cd['ms'], tn, year, day, dn_ids)
                w = bdata[(str(int(year)), day)]['w']
                _ck(f'{lab} {bk}: w_b production == step 4 spec w_b', w == central[bk]['w_b'], [w, central[bk]['w_b']])
                acc[bk] = block_accounting(lab, bk, cd['ms'], cd['cap_lines'], w)
                d4 = ev4['instances'][lab]['P4_D_b'][bk]
                acc[bk]['step4_bar_S_b'] = d4['bar_S_b']
                acc[bk]['step4_D_b'] = d4['D_b']
                _ck(f'{lab} {bk}: Q_pin == step 4 evaluation Q_pin_K (bitwise)', acc[bk]['Q_pin'] == d4['Q_pin_K'],
                    [acc[bk]['Q_pin'], d4['Q_pin_K']])
                _ck(f'{lab} {bk}: w C_central == step 4 evaluation (bitwise)',
                    central[bk]['w_b_C_central_eur'] == d4['w_b_C_central'])
        arms_gate, arms_iso = {}, {}
        for arm in ('A', 'B'):
            models, recs, iso = V2.build_arm_v2(tn, cand, avail, lab, arm, bdata, dn_ids)
            arms_gate[arm], arms_iso[arm] = recs, iso
            del models
        ctx[lab] = {'tn': tn, 'cand': cand, 'avail': avail, 'bdata': bdata, 'dn_ids': dn_ids, 'cd': cd,
                    'params': params_solve, 'acc': acc}
        res['instances'][lab] = {
            'eval_dir': cd['edir'], 'eval_key': inst['eval_key'], 'K': inst['K'], 'inputs': cd['inputs'],
            'case_path': inst['case_path'], 'case_sha256': inst['case_sha256'], 'scenario_checksum': checksum,
            'candidate_canonical': inst['candidate_canonical'],
            'available_capacity_applied': {str(n): {str(y): v for y, v in dd.items()} for n, dd in avail.items()},
            'solver_options_in_force': dict(params_solve.solver_params.options),
            'recovery': [params_solve.solver_params.recovery_enabled, params_solve.solver_params.recovery_tier2_enabled],
            'pre_solve_model_gates': {arm: {f'{k[0]}|{k[1]}': v for k, v in r.items()} for arm, r in arms_gate.items()},
            'pre_solve_shared_ess_isolation': {arm: {f'{k[0]}|{k[1]}': {kk: vv for kk, vv in v.items()
                                                                       if kk != 'fixed_schedule'}
                                                     for k, v in r.items()} for arm, r in arms_iso.items()},
            'accounting': acc, 'blocks': {}}
        _ck(f'{lab}: bound_relax_factor not overridden', 'bound_relax_factor' not in params_solve.solver_params.options)
    failed = [c['check'] for c in CHECKS if not c['ok']]
    if failed:
        res['refused_before_any_solve'] = failed
        _log(f'REFUSED before any solve: {len(failed)} failed checks')
    # ---- phase B: the solves (per instance, Arm A then Arm B, every block)
    for inst in ([] if (failed or a.trial) else spec['instances']):
        lab = inst['label']
        c = ctx[lab]
        tn, bdata, dn_ids = c['tn'], c['bdata'], c['dn_ids']
        for arm in ('A', 'B'):
            models, recs, iso = V2.build_arm_v2(tn, c['cand'], c['avail'], lab, arm, bdata, dn_ids)
            for year in tn.years:
                for day in tn.days:
                    key = (str(int(year)), day)
                    bk = f'{key[0]}|{day}'
                    net, m = tn.network[year][day], models[year][day]
                    blk = res['instances'][lab]['blocks'].setdefault(bk, {})
                    r = {'model_gates_ok': recs[key]['ok'], 'shared_ess_isolation_ok': iso[key]['ok']}
                    net.logs_dir = os.path.join(a.out_dir, 'logs', lab, arm)
                    os.makedirs(net.logs_dir, exist_ok=True)
                    st = _solve_one(net, m, c['params'])
                    r['solve'] = st
                    _ck(f'{lab} {arm} {bk}: solve status asserted (production success test)', st['loaded_by_production'],
                        st['solver_result_summary'])
                    if st['loaded_by_production']:
                        try:
                            r.update(H.extract(m, net, tn.params, bdata[key], dn_ids))
                            r['interface_voltage_mismatch_vs_dso_root'] = interface_voltage_mismatch(
                                r, c['cd']['ms'], bk, dn_ids, 'DSO')
                            r['interface_voltage_mismatch_vs_tso_saved'] = interface_voltage_mismatch(
                                r, c['cd']['ms'], bk, dn_ids, 'TSO')
                        except Exception:  # noqa: BLE001 -- a capture failure is recorded, never hides the solve
                            r['extract_error'] = traceback.format_exc()
                            _ck(f'{lab} {arm} {bk}: extraction', False, r['extract_error'][-400:])
                    blk[arm] = r
                    ex = (st['ipopt_records'] or [{}])[-1]
                    _log(f"{lab} {arm} {bk}: {st['termination_condition']} {ex.get('exit')} loaded "
                         f"{st['loaded_by_production']} it {ex.get('iterations')} wall {st['wall_s']:.1f}s "
                         f"obj {r.get('costs', {}).get('objective')}")
                    models[year][day] = None
            del models
        for bk, blk in res['instances'][lab]['blocks'].items():
            y, d = bk.split('|')
            bd = bdata[(y, d)]
            blk['inputs_used'] = {'w': bd['w'], 'omega': bd['omega'], 'cov_block': bd['cov_block'],
                                  'cov_saved_by_dn': {str(k): v for k, v in bd['cov_saved'].items()},
                                  'cov_recomputed_by_dn': {str(k): v for k, v in bd['cov_recomputed'].items()},
                                  'tso_cost_saved': bd['tso_cost_saved'],
                                  'tso_p_common_mw': {str(k): v for k, v in bd['tso_p'].items()},
                                  'dso_p_realized_mw': {str(k): v for k, v in bd['dso_p'].items()},
                                  'prices_eur_per_mwh_by_hour_by_market': bd['prices']}
    if not (failed or a.trial):
        try:
            res['statistics'] = statistics(res, ctx, spec, central)
            res['predictions'] = predictions(res, spec)
        except Exception:  # noqa: BLE001 -- recorded; the per-block records are still written
            res['statistics_error'] = traceback.format_exc()
            _ck('statistics computed', False, res['statistics_error'][-400:])
    res['provenance_recheck_at_end'] = provenance(a.spec, a.trial)
    res['total_wall_s'] = time.perf_counter() - t_start
    gv = GUARD.verify(expected, expected)
    imported = {'v2': dict(V2.GUARD.counts), 'v1': dict(H.GUARD.counts), 'postcheck': dict(V2.PCK.GUARD.counts)}
    imported_zero = all(v == 0 for dd in imported.values() for v in dd.values())
    res['guard'] = {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                    'permitted_sites': dict(GUARD.permitted_sites), 'expected_solves': expected,
                    'verify_failures': gv, 'imported_guards_uninstalled_after_import': imported,
                    'imported_guards_all_zero': imported_zero}
    res['pickle'] = {'this': dict(PK), 'v2_module': dict(V2.PK), 'v1_module': dict(H.PK), 'postcheck_module': dict(V2.PCK.PK)}
    res['guards_ok'] = (not gv) and imported_zero and all(dd == {'load': 0, 'loads': 0} for dd in res['pickle'].values())
    res['checks'] = CHECKS
    res['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    out = os.path.join(a.out_dir, 'db_tn_resolve_results.json')
    with open(out, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    manifest = {'outputs': {os.path.basename(out): _sha(out)},
                'inputs': {**{f'{lab}/{k}': v for lab, cc in res['instances'].items() for k, v in cc['inputs'].items()},
                           **{f'module/{k}': v for k, v in res['modules'].items()}},
                'spec': {'path': a.spec, 'sha256': spec_sha}, 'provenance': prov,
                'other_files': {os.path.relpath(os.path.join(dp, f), a.out_dir): _sha(os.path.join(dp, f))
                                for dp, _d, fs in os.walk(a.out_dir) for f in fs
                                if os.path.join(dp, f) != out},
                'note': 'stdout/stderr launch logs are written while the script runs; hashed in the stage manifest'}
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump(manifest, h, indent=1)
    _log(f"wrote {out}; guards_ok {res['guards_ok']}; failed checks {len(res['failed_checks'])}")
    if failed:
        return 3
    return 0 if (res['guards_ok'] and not res['failed_checks']) else 2


# ------------------------------------------------------------------------------------------------------- statistics
def statistics(res, ctx, spec, central):
    out = {'objective_convention': spec['objective_convention'], 'per_block': {}}
    for lab, ic in res['instances'].items():
        acc = ctx[lab]['acc']
        for bk, blk in ic['blocks'].items():
            A, B = blk.get('A', {}), blk.get('B', {})
            w = blk['inputs_used']['w']
            ac = acc[bk]
            wc = central[bk]['w_b_C_central_eur']
            row = {'instance': lab, 'w_b': w, 'calibrated': central[bk]['calibrated'],
                   'w_C_central': wc, 'C_central_eur_per_day': central[bk]['C_central_eur_per_day'],
                   'Q_pin': ac['Q_pin'], 'D_b': wc - ac['Q_pin'], 'D_b_equals_step4_bitwise': (wc - ac['Q_pin']) == ac['step4_D_b'],
                   'accounting': {k: v for k, v in ac.items()},
                   'A_status': V2._arm_status(A), 'B_status': V2._arm_status(B),
                   'valid_B': bool(V2._arm_status(B)['loaded_by_production'] and 'costs' in B),
                   'valid_A': bool(V2._arm_status(A)['loaded_by_production'] and 'costs' in A)}
            if row['valid_B']:
                cb = B['costs']['objective']
                c_res = w * (cb + ac['dn_base_sum_eur_per_day'])
                t1 = c_res - ac['Q_pin']
                t2 = wc - c_res
                summands = [abs(c_res), abs(ac['Q_pin']), abs(wc), abs(c_res), abs(wc), abs(ac['Q_pin'])]
                tol = max(64.0 * U * sum(summands), 1e-6)
                row.update({'C_TN_B_objective_eur_per_day': cb, 'C_TN_B_gen_plus_flex_eur_per_day':
                            B['costs']['C_gen_plus_flex'], 'C_TN_B_costs': B['costs'],
                            'C_TN_resolve': c_res, 'T1_pinned_accounting': t1, 'T2_DN_reoptimization_gain': t2,
                            'B3': {'lhs_T1_plus_T2': t1 + t2, 'rhs_D_b': wc - ac['Q_pin'],
                                   'abs_diff': abs((t1 + t2) - (wc - ac['Q_pin'])), 'sum_abs_summands': sum(summands),
                                   'tolerance': tol, 'held': abs((t1 + t2) - (wc - ac['Q_pin'])) <= tol},
                            'resolution': {'T1_last10_max_step_Q_pin': ac['last10_max_step_Q_pin'],
                                           'T2_last10_max_step_w_dn_base': ac['last10_max_step_w_dn_base'],
                                           'step4_bar_S_b_Q_pin': ac['step4_bar_S_b']},
                            'B_max_violation': B['max_violation']['max'],
                            'B_detector': B['detector_components']['detector_penalty_total'],
                            'B_slack_penalties': B['costs']['total_slack_penalties'],
                            'B_interface_voltage_mismatch_vs_dso_root': B['interface_voltage_mismatch_vs_dso_root']})
            if row['valid_A'] and row['valid_B']:
                ca = A['costs']['objective']
                ctrl = w * (ca - ac['tso_base_eur_per_day'])
                d1 = w * ((row['C_TN_B_objective_eur_per_day'] - ca) - ac['dn_cov_sum_eur_per_day'])
                row['control_reading_beside'] = {
                    'note': 'a reading beside T1, not a decomposition: T1 = w (C_A - TSO base) + w [(C_B - C_A) - cov]',
                    'C_TN_A_objective_eur_per_day': ca, 'w_C_A_minus_TSO_base': ctrl,
                    'w_D1_realized_vs_common_net_of_cov': d1, 'sum': ctrl + d1,
                    'sum_minus_T1': (ctrl + d1) - row['T1_pinned_accounting'],
                    'A_interface_voltage_mismatch_vs_tso_saved': A['interface_voltage_mismatch_vs_tso_saved'],
                    'A_max_violation': A['max_violation']['max']}
            out['per_block'][bk] = row
    return out


def predictions(res, spec):
    st = res['statistics']['per_block']
    P = spec['predictions']
    out = {}
    for name in ('B1', 'B2', 'B3'):
        out[name] = {'statement': P[name]['statement'], 'per_block': {}}
    for bk, row in st.items():
        tau = P['B2']['tolerance_eur_per_day'] * row['w_b']
        deciding = row['calibrated']
        if not row['valid_B']:
            for name in ('B1', 'B2', 'B3'):
                out[name]['per_block'][bk] = {'deciding': deciding, 'held': None, 'note': 'indeterminate: Arm B not loaded'}
            continue
        t1, t2 = row['T1_pinned_accounting'], row['T2_DN_reoptimization_gain']
        out['B1']['per_block'][bk] = {'deciding': deciding, 'T1': t1, 'held': t1 > 0.0,
                                      'numeric_addition_T1_ge_D_b_minus_tau': t1 >= row['D_b'] - tau,
                                      'D_b': row['D_b'], 'tau': tau,
                                      'determinate_vs_last10_Q_pin_step': abs(t1) > row['resolution']['T1_last10_max_step_Q_pin']}
        out['B2']['per_block'][bk] = {'deciding': deciding, 'T2': t2, 'tau': tau, 'held': t2 <= tau,
                                      'determinate_vs_last10_dn_base_step': abs(t2) > row['resolution']['T2_last10_max_step_w_dn_base']}
        out['B3']['per_block'][bk] = {'deciding': deciding, **row['B3']}
    for name in ('B1', 'B2', 'B3'):
        dec = [v for v in out[name]['per_block'].values() if v['deciding']]
        out[name]['held_on_both_calibration_blocks'] = (all(v['held'] is True for v in dec) if len(dec) == 2
                                                       and all(v['held'] is not None for v in dec) else None)
    n_strict = sum(1 for row in st.values() for s in ('A_status', 'B_status') if row[s]['strict_optimal'])
    out['P_solves'] = {'statement': P['P_solves']['statement'], 'n_strict_optimal': n_strict,
                       'n_loaded': sum(1 for row in st.values() for s in ('A_status', 'B_status')
                                       if row[s]['loaded_by_production']),
                       'held': n_strict == spec['guards']['expected_solves']}
    ctrl = {}
    for bk, row in st.items():
        cr = row.get('control_reading_beside')
        if cr:
            tau = P['B2']['tolerance_eur_per_day'] * row['w_b']
            ctrl[bk] = {'w_C_A_minus_TSO_base': cr['w_C_A_minus_TSO_base'], 'tau': tau,
                        'held': cr['w_C_A_minus_TSO_base'] <= tau}
    out['CTRL'] = {'statement': P['CTRL']['statement'], 'per_block': ctrl, 'deciding': False}
    return out


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
