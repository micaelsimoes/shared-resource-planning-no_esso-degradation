"""P5.15 S56 -- T8 binding check (2 ESSO solves) + T9 three-change attribution (4 ESSO solves). Exactly 6 solves.

Authority: TASKS.md "Expert rulings on handback 3 (2026-11-23)" item 4: T8 (a) accepted (designed slack); two
millisecond solves confirming the per-cohort complementarity row binds (row dual + hourly profile); a within-cohort leak
detector with ceiling lambda.L priced <= 1 % of R/4 (= 25 EUR); explain the three undiagnosed T9 changes (t_ii_b
stored-energy rise, t_iii_indep window margin, t_iv_peak leak / transfer) before the harness wiring spec is frozen.

WHAT RUNS. The solve tests are NOT re-implemented: each child imports `p515_s56_cohort_followup_tests` (the cohort
follow-up harness, unedited) and calls its own `child_solves(args)` with a tag of THIS stage's frozen spec, whose
`solves` block carries the v1 test definitions verbatim (node, t_ii, t_iii, t_iv, t_v) and this stage's tags:
  binding      order [t_v_tv_on (= T8), t8b_two_cohort (= T8b)]   2 solves, no model change
  t9_rows_off  order [t_ii_b, t_iii_indep, t_iv_peak]              3 solves, the two new row families
                                                                     (energy_storage_cohort_complementarity, _sum_limit)
                                                                     deactivated immediately before each solve
  t9_sum_off   order [t_ii_b]                                      1 solve, the sum rows only deactivated
The only addition is a class-level wrapper around production `SharedEnergyStorageData.optimize` (installed in the child
before any test runs): it applies the tag's declared deactivation to the node model, calls the production method
unchanged, and then READS the solved model (no write): per active cohort and (block, day, period) p_ch, p_dch, SoC, the
complementarity row value and its dual (`model.dual`, production's IMPORT_EXPORT suffix), the sum row value and dual, and
the IPOPT bound multipliers of p_ch and p_dch (`ipopt_zL_out` / `ipopt_zU_out`, production's IMPORT suffixes).

PROCESSES AND GUARDS. Parent (zero solves): SolveProfileGuard(permitted=()) installed before any import, verified at 0;
launches the three children one after the other (attached, stdout and stderr captured per child), then evaluates.
Child: SolveProfileGuard(permitted=(('shared_energy_storage_data.py', '_run_solver_attempt'),)) installed before any
project import, verified at EXACTLY the tag's declared count (the harness module's own guard, installed on top at import
with the same permitted site, is verified at the same count); a recovery retry fails it. pickle.load/loads blocked in
every process, verified 0. The wrapper's call count is checked against the declared count too.

ISOLATION. Run from a clean `git archive` export of 4c586e5e with this script and its spec overlaid from the commit
that froze them (EXPORT_PROVENANCE.json, written by p515_s56_db_t8_export.py); the script, its spec and every imported
root module are checked against it; the working directory is the export.

Usage (from the export root):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=... <canonical python> -u p515_s56_t8_binding_t9.py \\
      --spec SPEC --out-dir DIR [--trial]
  (--trial: zero solves: provenance not required; every child runs with --no-solve: the production optimize is not
   called and the capture reads each test model at its built values -- exercises the build, the row map and the
   capture path; every guard verified at 0)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = '--child' in sys.argv
if CHILD:  # the export this file sits in is the ONLY project source on sys.path (as the harness child)
    sys.path[:] = [HERE] + [p for p in sys.path[1:] if os.path.abspath(p or '.') != HERE]
elif HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

SOLVE_SITE = ('shared_energy_storage_data.py', '_run_solver_attempt')
GUARD = SolveProfileGuard(permitted=(SOLVE_SITE,) if CHILD else (),
                          label=f"P5.15 S56 T8 binding / T9 attribution ({'child' if CHILD else 'parent'})").install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

PROVENANCE_FILE = 'EXPORT_PROVENANCE.json'
PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
COMP = 'energy_storage_cohort_complementarity'
SUM = 'energy_storage_cohort_sum_limit'
ROW_ORDER = ('energy_storage_cohort_soc', 'energy_storage_cohort_soc_window', 'energy_storage_cohort_soc_window',
             'energy_storage_cohort_converter', SUM, COMP)


def _log(msg):
    print(f"[t8t9 {'child' if CHILD else 'parent'} {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _dump(obj, path):
    import gate_result_io as GRIO
    with open(path, 'x') as handle:
        handle.write(GRIO.dumps(obj, indent=1, sort_keys=True))


def provenance(spec_path, trial):
    p = os.path.join(HERE, PROVENANCE_FILE)
    if not os.path.exists(p):
        return None, ([] if trial else [f'{PROVENANCE_FILE} missing (not a provenance-recorded export)'])
    prov = json.load(open(p))
    files = {**prov.get('exported_root_files_sha256', {}), **{k: v['sha256'] for k, v in prov.get('overlay', {}).items()}}
    bad = []
    rel_spec = os.path.relpath(os.path.abspath(spec_path), HERE)
    for rel in (SCRIPT, rel_spec):
        if files.get(rel) != _sha(os.path.join(HERE, rel)):
            bad.append(f'{rel}: sha256 differs from the export record')
    mods = {}
    for _name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.dirname(os.path.abspath(f)) == HERE and f.endswith('.py'):
            sha = _sha(f)
            mods[os.path.basename(f)] = sha
            if files.get(os.path.basename(f)) != sha:
                bad.append(f'imported module {os.path.basename(f)}: sha256 differs from the export record')
    return {'export_commit': prov.get('export_commit'), 'overlay_commit': prov.get('overlay_commit'),
            'provenance_file_sha256': _sha(p), 'imported_root_modules_sha256': mods}, ([] if trial else bad)


# ======================================================================================================================
#  CHILD
# ======================================================================================================================
def _row_map(m):
    """(y_inv, y, d, p) -> {'sum': row, 'comp': row} from the production registry, whose per-(cohort, block) entry list
    is built in the fixed order [per (d, p): soc, window lo, window hi, converter, sum, comp] + [per d: closure]
    (`_build_esso_cohort_units_rows`); the order is checked, and each comp row's variables are checked to be that
    period's p_ch / p_dch."""
    from pyomo.core.expr.visitor import identify_variables
    reg = m._esso_cohort_units_rows
    out, bad = {}, []
    for (y_inv, y), entries in reg['cohort'].items():
        k = 0
        for d in m.days:
            for p in m.periods:
                names = tuple(e[0] for e in entries[k:k + 6])
                if names != ROW_ORDER:
                    bad.append(((y_inv, y, d, p), names))
                rows = {'sum': getattr(m, SUM)[entries[k + 4][1]], 'comp': getattr(m, COMP)[entries[k + 5][1]]}
                vs = {id(v) for v in identify_variables(rows['comp'].body)}
                if vs != {id(m.es_pch_per_unit[y_inv, y, d, p]), id(m.es_pdch_per_unit[y_inv, y, d, p])}:
                    bad.append(((y_inv, y, d, p), 'comp row variables'))
                out[(y_inv, y, d, p)] = rows
                k += 6
            if entries[k][0] != 'energy_storage_cohort_soc_final':
                bad.append(((y_inv, y, d), entries[k][0]))
            k += 1
        if k != len(entries):
            bad.append(((y_inv, y), f'{k} != {len(entries)} entries'))
    return out, bad


def capture_model(m, sed, node, DG):
    import pyomo.environ as pe
    years = list(sed.years)
    days = list(sed.days)
    rmap, bad = _row_map(m)
    dual, zl, zu = m.dual, m.ipopt_zL_out, m.ipopt_zU_out
    recs = []
    for y in m.years:
        for c in DG.active_cohorts(m, y):
            for d in m.days:
                for p in m.periods:
                    pch, pdch = m.es_pch_per_unit[c, y, d, p], m.es_pdch_per_unit[c, y, d, p]
                    rows = rmap[(c, y, d, p)]
                    r = {'cohort': years[c], 'block': years[y], 'day': days[d], 'period': p,
                         'pch_mw': pe.value(pch), 'pdch_mw': pe.value(pdch),
                         'min_pch_pdch_mw': min(pe.value(pch), pe.value(pdch)),
                         'soc_mwh': pe.value(m.es_soc_per_unit[c, y, d, p]),
                         'zL_pch': zl.get(pch), 'zU_pch': zu.get(pch), 'zL_pdch': zl.get(pdch), 'zU_pdch': zu.get(pdch)}
                    for key in ('comp', 'sum'):
                        row = rows[key]
                        r[f'{key}_active'] = bool(row.active)
                        r[f'{key}_value'] = pe.value(row.body)
                        r[f'{key}_upper'] = pe.value(row.upper)
                        r[f'{key}_dual'] = dual.get(row) if row.active else None
                    recs.append(r)
    diag = list(getattr(sed, 'esso_complementarity_diagnostics', []) or [])
    return {'hourly': recs, 'row_map_problems': bad, 'n_dual_entries': len(dual), 'n_zL': len(zl), 'n_zU': len(zu),
            'mu_final': (diag[-1] if diag else {}).get('mu_final'),
            'objective': pe.value(m.objective) if m.objective.active else None,
            'admm_objective': pe.value(m.admm_objective) if hasattr(m, 'admm_objective') else None,
            'feasibility_penalty': pe.value(m.feasibility_penalty)}


def child_main(a):
    spec = json.load(open(a.spec))
    tag = a.tag
    tdef = spec['stage_tags'][tag]
    declared = spec['solves'][tag]['declared_count']
    node = spec['solves']['node']
    import p515_s56_cohort_followup_tests as T   # installs its own guard on top (permitted: the same solve site)
    import shared_energy_storage_data as SED
    import p515_s56_cohort_diagnostics as DG
    pickle.load, pickle.loads = _blk('load'), _blk('loads')
    prov, bad = provenance(a.spec, a.no_solve)
    if bad:
        raise SystemExit(f'refusing: provenance {bad}')
    orig = SED.SharedEnergyStorageData.optimize
    calls, captures = [], {}

    def wrapped(self, models, from_warm_start=False, cycle=None):
        m = models[node]
        deact = {}
        for fam in tdef['deactivate']:
            n = 0
            for c in getattr(m, fam).values():
                if c.active:
                    c.deactivate()
                    n += 1
            deact[fam] = n
        if a.no_solve:   # trial: zero solves; the capture reads the model at its built values
            res = {n: None for n in models}
        else:
            res = orig(self, models, from_warm_start=from_warm_start, cycle=cycle)
        calls.append({'cycle': cycle, 'deactivated_active_rows': deact, 'solved': not a.no_solve})
        try:
            captures[str(cycle)] = {'test': spec['solves'][tag]['order'][cycle - 1], 'deactivated_active_rows': deact,
                                    **capture_model(m, self, node, DG)}
        except Exception:  # noqa: BLE001 -- recorded, never hides the solve
            captures[str(cycle)] = {'capture_error': traceback.format_exc()}
        return res

    SED.SharedEnergyStorageData.optimize = wrapped
    ns = argparse.Namespace(spec=a.spec, work=a.work, tag=tag)
    t0 = time.time()
    rc = T.child_solves(ns)
    SED.SharedEnergyStorageData.optimize = orig
    # prices and block weights for the leak detector (zero solves: production reader on the same SRP1 case)
    prices = None
    if tdef.get('read_prices'):
        planning, _info = T._read_planning('SRP1.json', os.path.join(a.work, 'side_prices'))
        tn = planning.transmission_network
        sedp = planning.shared_ess_data
        prices = {'discount_factor': tn.discount_factor, 'years': {str(k): v for k, v in tn.years.items()},
                  'days': {str(k): v for k, v in tn.days.items()}, 'sed_years': {str(k): v for k, v in sedp.years.items()},
                  'sed_days': {str(k): v for k, v in sedp.days.items()}, 'per_block_day': {}}
        for y in tn.years:
            for d in tn.days:
                net = tn.network[y][d]
                pm = list(net.prob_market_scenarios)
                prices['per_block_day'][f'{y}|{d}'] = {
                    'prob_market': pm, 'cost_energy_p': [list(map(float, row)) for row in net.cost_energy_p],
                    'expected_price_eur_per_mwh': [sum(pm[s] * net.cost_energy_p[s][t] for s in range(len(pm)))
                                                   for t in range(len(net.cost_energy_p[0]))]}
    expected = 0 if a.no_solve else declared
    gv = GUARD.verify(expected)
    out = {'tag': tag, 'declared': declared, 'expected_solves_this_process': expected, 'no_solve_trial': a.no_solve,
           'child_solves_rc': rc, 'wall_s': time.time() - t0,
           'wrapper_calls': calls, 'wrapper_calls_ok': len(calls) == declared, 'captures': captures,
           'prices': prices, 'provenance': prov,
           'guard': {'counts': dict(GUARD.counts), 'permitted_sites': dict(GUARD.permitted_sites),
                     'verify_failures': gv},
           'harness_module_guard': {'counts': dict(T.GUARD.counts), 'verify_failures': T.GUARD.verify(expected)},
           'pickle': dict(PK), 'harness_module_file': os.path.abspath(T.__file__),
           'production_module_file': os.path.abspath(SED.__file__)}
    _dump(out, os.path.join(a.work, f'stage_capture_{tag}.json'))
    ok = (rc == 0 and not gv and not out['harness_module_guard']['verify_failures'] and out['wrapper_calls_ok']
          and PK == {'load': 0, 'loads': 0})
    return 0 if ok else 2


# ======================================================================================================================
#  PARENT
# ======================================================================================================================
def _get(d, path):
    for k in path.split('.'):
        d = d[k]
    return d


def own_copies_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found, others = os.getpid(), [], []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python'):
            names = [os.path.basename(t) for t in tok[2:4]]
            if SCRIPT in names:
                found.append(ln.strip()[:300])
            elif any(n.startswith('p5') and n.endswith('.py') for n in names):
                others.append(ln.strip()[:300])
    return found, others


def parent_main(a):
    t_start = time.time()
    pre = []
    spec_sha = _sha(a.spec)
    tag8 = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not spec_sha.startswith(tag8):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag8}')
    prov, bad = provenance(a.spec, a.trial)
    pre += bad
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    mine, others = own_copies_running()
    if mine:
        pre.append(f'another copy runs: {mine}')
    if pre:
        print('[t8t9] PRECONDITION FAILED:', pre, flush=True)
        return 1
    spec = json.load(open(a.spec))
    # capture checklist, asserted before any child runs (CLAUDE.md rule eleven)
    capture = {'T8_hourly_pch_pdch_row_value_row_dual': 'capture_model', 'bound_multipliers_pch_pdch': 'capture_model',
               'T8_objective_and_L_reproduction': 'evaluate', 'leak_detector_lambda_L': 'leak_detector',
               'T9_rows_off_values': 'evaluate', 'T9_sum_off_values': 'evaluate', 'mu_final': 'capture_model',
               'guards': 'child_main', 'predictions': 'evaluate'}
    missing = [k for k in spec['reported'] if k not in capture]
    assert not missing, f'spec quantities without a capture path: {missing}'
    assert all(fn in globals() for fn in capture.values()), capture
    for tag in spec['stage_tags']:
        assert spec['solves'][tag]['declared_count'] == len(spec['solves'][tag]['order']), tag
    assert sum(spec['solves'][t]['declared_count'] for t in spec['stage_tags']) == spec['guards']['expected_solves_total']
    os.makedirs(a.out_dir)
    res = {'stage': spec['stage'], 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script': SCRIPT,
           'script_sha256': _sha(SCRIPT), 'provenance': prov, 'run_directory': HERE, 'trial': a.trial,
           'utc_start': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable,
           'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'), 'lp_solver_path': os.environ.get('LP_SOLVER_PATH'),
           'other_p5_python_processes_at_launch': others, 'children': {}}
    for rel, sha in spec['inputs'].items():
        res.setdefault('input_checks', {})[rel] = {'sha256': _sha(rel), 'ok': _sha(rel) == sha}
    inputs_ok = all(v['ok'] for v in res['input_checks'].values())
    if not inputs_ok:
        res['refused'] = 'input sha256 mismatch'
    for tag in ([] if not inputs_ok else spec['stage_tags']):
        work = os.path.join(a.out_dir, f'work_{tag}')
        os.makedirs(work)
        so, se = os.path.join(a.out_dir, f'child_{tag}_stdout.log'), os.path.join(a.out_dir, f'child_{tag}_stderr.log')
        cmd = [PY, '-u', SCRIPT, '--child', 'solves', '--spec', a.spec, '--work', work, '--tag', tag] + \
            (['--no-solve'] if a.trial else [])
        _log(f'child {tag}: {" ".join(cmd)}')
        t0 = time.time()
        with open(so, 'x') as fo, open(se, 'x') as fe:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fo, stderr=fe, env=dict(os.environ)).returncode
        res['children'][tag] = {'rc': rc, 'wall_s': time.time() - t0, 'cmd': cmd}
        _log(f'child {tag}: rc {rc} ({time.time() - t0:.1f} s)')
    if inputs_ok:   # under --trial this exercises the evaluation on the zero-solve captures (values not meaningful)
        try:
            res['evaluation'] = evaluate(spec, a.out_dir)
        except Exception:  # noqa: BLE001
            res['evaluation_error'] = traceback.format_exc()
            _log(res['evaluation_error'])
    res['provenance_recheck_at_end'] = provenance(a.spec, a.trial)
    gv = GUARD.verify(0)
    res['guard_parent'] = {'counts': dict(GUARD.counts), 'verify_failures': gv}
    res['pickle_parent'] = dict(PK)
    res['wall_s'] = time.time() - t_start
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    ok_children = all(c['rc'] == 0 for c in res['children'].values())
    res['all_ok'] = (not gv and PK == {'load': 0, 'loads': 0} and ok_children and inputs_ok
                     and (a.trial or 'evaluation' in res))
    out = os.path.join(a.out_dir, 't8_binding_t9_results.json')
    _dump(res, out)
    man = {'files': {os.path.relpath(os.path.join(dp, f), a.out_dir): _sha(os.path.join(dp, f))
                     for dp, _d, fs in os.walk(a.out_dir) for f in fs},
           'spec': {'path': a.spec, 'sha256': spec_sha}, 'provenance': prov,
           'note': 'the parent stdout/stderr launch logs are written while the script runs; hashed in the stage manifest'}
    _dump(man, os.path.join(a.out_dir, 'manifest_sha256.json'))
    _log(f'wrote {out}; all_ok {res["all_ok"]}')
    return 0 if res['all_ok'] else 2


# ---------------------------------------------------------------------------------------------------------- evaluate
def _load_child(out_dir, tag):
    work = os.path.join(out_dir, f'work_{tag}')
    rec = json.load(open(os.path.join(work, f'child_solves_{tag}.json')))
    cap = json.load(open(os.path.join(work, f'stage_capture_{tag}.json')))
    return rec, cap


def _same(a, b):
    return {'new': a, 'frozen': b, 'bitwise': a == b, 'new_hex': float(a).hex(), 'frozen_hex': float(b).hex(),
            'rel': abs(a - b) / max(abs(a), abs(b), 1e-300)}


def leak_detector(cap, prices, spec):
    """lambda.L over the horizon. L = sum over cohorts, block-days, hours of max(min(p_ch, p_dch), 0) dt (MWh per
    representative day), weighted by days[d] x years[y] (the real days a representative block-day stands for) and by
    the production discount 1/(1+r)^(y - y0); priced at the TN expected day-ahead energy price of that block-day hour
    (primary), and beside: the conversion-loss form ((1/eta_dch - eta_ch) x the same) and the row-dual implied value
    (ESSO objective units, not EUR)."""
    dt = 1.0
    years = list(prices['sed_years'])
    y0 = int(years[0])
    r = prices['discount_factor']
    eta_ch, eta_dch = spec['leak_detector']['eta_ch'], spec['leak_detector']['eta_dch']
    tot = {'L_mwh_per_representative_days_sum': 0.0, 'L_horizon_mwh_undiscounted': 0.0, 'L_horizon_mwh_discounted': 0.0,
           'lambdaL_energy_price_eur_discounted': 0.0, 'lambdaL_energy_price_eur_undiscounted': 0.0,
           'lambdaL_loss_form_eur_discounted': 0.0, 'dual_implied_objective_units': 0.0}
    per = {}
    for h in cap['hourly']:
        mval = max(h['min_pch_pdch_mw'], 0.0) * dt
        if mval <= 0.0:
            continue
        yk, dk = str(h['block']), h['day']
        wdays = prices['sed_days'][dk] * prices['sed_years'][yk]
        disc = 1.0 / (1.0 + r) ** (int(yk) - y0)
        pi = prices['per_block_day'][f'{yk}|{dk}']['expected_price_eur_per_mwh'][h['period']]
        tot['L_mwh_per_representative_days_sum'] += mval
        tot['L_horizon_mwh_undiscounted'] += mval * wdays
        tot['L_horizon_mwh_discounted'] += mval * wdays * disc
        tot['lambdaL_energy_price_eur_discounted'] += mval * wdays * disc * pi
        tot['lambdaL_energy_price_eur_undiscounted'] += mval * wdays * pi
        tot['lambdaL_loss_form_eur_discounted'] += (1.0 / eta_dch - eta_ch) * mval * wdays * disc * pi
        if h['comp_dual'] is not None and h['comp_active']:
            s2 = spec['leak_detector']['S_c_mva'][str(h['cohort'])] ** 2
            tot['dual_implied_objective_units'] += abs(h['comp_dual']) * max(h['pch_mw'], h['pdch_mw']) / s2 * mval
        k = f"{h['cohort']}|{yk}|{dk}"
        per[k] = per.get(k, 0.0) + mval
    ceiling = spec['leak_detector']['ceiling_eur']
    tot['ceiling_eur'] = ceiling
    tot['primary'] = 'lambdaL_energy_price_eur_discounted'
    tot['primary_within_ceiling'] = tot['lambdaL_energy_price_eur_discounted'] <= ceiling
    tot['loss_form_within_ceiling'] = tot['lambdaL_loss_form_eur_discounted'] <= ceiling
    tot['L_per_cohort_block_day_mwh'] = per
    return tot


def evaluate(spec, out_dir):
    ev = {'children': {}}
    frozen = json.load(open(spec['frozen_records']['after']))['tests']
    v2 = json.load(open(spec['frozen_records']['v2']))['tests']
    recs = {}
    for tag in spec['stage_tags']:
        rec, cap = _load_child(out_dir, tag)
        recs[tag] = (rec, cap)
        ev['children'][tag] = {'errors': rec['errors'], 'guard': rec['guard'], 'stage_guard': cap['guard'],
                               'harness_module_guard': cap['harness_module_guard'], 'wrapper_calls': cap['wrapper_calls'],
                               'pickle': cap['pickle'], 'solve_log': [{k: v for k, v in e.items() if k != 'stdout_tail'}
                                                                      for e in rec['solve_log']],
                               'row_map_problems': {c: x.get('row_map_problems') for c, x in cap['captures'].items()},
                               'capture_errors': {c: x['capture_error'] for c, x in cap['captures'].items()
                                                  if 'capture_error' in x}}
    P = {}
    # ---------------- T8 / T8b
    rec, cap = recs['binding']
    tol_bind = spec['predictions']['T8-B1']['binding_threshold']
    sep = spec['predictions']['T8-B1']['separation_factor']
    nb_max = spec['predictions']['T8-B1']['nonbinding_value_max']
    for name, cyc in (('T8', '1'), ('T8b', '2')):
        test = 't_v_tv_on' if name == 'T8' else 't8b_two_cohort'
        new, old = rec['tests'][test], frozen[test]
        c = cap['captures'][cyc]
        hourly = c['hourly']
        binding = [h for h in hourly if h['comp_active'] and h['comp_value'] >= tol_bind]
        nonbinding = [h for h in hourly if h['comp_active'] and h['comp_value'] <= nb_max]
        nb_dual_max = max((abs(h['comp_dual']) for h in nonbinding if h['comp_dual'] is not None), default=None)
        b_duals = [abs(h['comp_dual']) if h['comp_dual'] is not None else None for h in binding]
        b1 = (len(binding) > 0 and all(d is not None and d > 0.0 for d in b_duals)
              and nb_dual_max is not None and min(b_duals) >= sep * nb_dual_max)
        sets = {}
        for h in binding:
            sets.setdefault(f"{h['cohort']}|{h['block']}|{h['day']}", []).append(h['period'])
        per_bd = {}
        for h in hourly:
            k = f"{h['cohort']}|{h['block']}|{h['day']}"
            dd = per_bd.setdefault(k, {'L': 0.0, 'L_hours_1_4': 0.0})
            mval = h['min_pch_pdch_mw']
            dd['L'] += mval
            if h['period'] in (1, 2, 3, 4):
                dd['L_hours_1_4'] += mval
        multi = sorted(k for k in per_bd if k.startswith('2025|') and k.split('|')[1] in ('2030', '2035'))
        b3 = (set(sets) == set(multi) and all(sorted(v) == [1, 2, 3, 4] for v in sets.values()))
        L_new = max(v['max_mwh_per_day'] for v in new['within'].values())
        L_old = max(v['max_mwh_per_day'] for v in old['within'].values())
        P[name] = {
            'B1_row_dual_nonzero_where_binding': {
                'held': b1, 'n_binding': len(binding), 'binding_dual_abs_min': min((d for d in b_duals if d is not None), default=None),
                'binding_dual_abs_max': max((d for d in b_duals if d is not None), default=None),
                'binding_dual_signs': sorted({(h['comp_dual'] > 0) - (h['comp_dual'] < 0) for h in binding
                                              if h['comp_dual'] is not None}),
                'nonbinding_n': len(nonbinding), 'nonbinding_dual_abs_max': nb_dual_max, 'separation_factor': sep},
            'B2_reproduction': {'admm_objective': _same(new['admm_objective'], old['admm_objective']),
                                'feasibility_penalty': _same(new['feasibility_penalty'], old['feasibility_penalty']),
                                'L_max_mwh_per_day': _same(L_new, L_old),
                                'held_bitwise': (new['admm_objective'] == old['admm_objective']
                                                 and new['feasibility_penalty'] == old['feasibility_penalty'] and L_new == L_old),
                                'held_rel_1e-9': all(_same(x, y)['rel'] <= 1e-9 for x, y in (
                                    (new['admm_objective'], old['admm_objective']), (L_new, L_old)))},
            'B3_leak_hours_are_charge_hours_1_4': {
                'held': b3, 'binding_periods_by_cohort_block_day': {k: sorted(v) for k, v in sets.items()},
                'expected_cohort_block_days': multi,
                'share_of_L_in_hours_1_4': {k: (v['L_hours_1_4'] / v['L'] if v['L'] > 0 else None)
                                            for k, v in per_bd.items() if k in multi}},
            'mu_final': c['mu_final'],
            'bound_multipliers_at_binding': [{k: h[k] for k in ('cohort', 'block', 'day', 'period', 'pch_mw', 'pdch_mw',
                                                                'comp_value', 'comp_dual', 'zL_pch', 'zU_pch', 'zL_pdch',
                                                                'zU_pdch', 'sum_value', 'sum_dual')} for h in binding],
            'leak_detector': leak_detector(c, cap['prices'], spec)}
    # ---------------- T9
    rec_off, cap_off = recs['t9_rows_off']
    rec_sum, cap_sum = recs['t9_sum_off']
    t9 = {}
    for item in spec['predictions']['T9-R']['quantities']:
        test, path = item['test'], item['path']
        x_v2, x_after = _get(v2[test], path), _get(frozen[test], path)
        x_off = _get(rec_off['tests'][test], path)
        dchg = x_after - x_v2
        row = {'test': test, 'path': path, 'v2': x_v2, 'after': x_after, 'rows_off': x_off, 'change_v2_to_after': dchg,
               'rows_off_minus_v2': x_off - x_v2, 'fraction_of_change_left_with_rows_off':
                   (x_off - x_v2) / dchg if dchg else None,
               'held': abs(x_off - x_v2) <= spec['predictions']['T9-R']['fraction'] * abs(dchg)}
        if test in rec_sum['tests']:
            x_sum = _get(rec_sum['tests'][test], path)
            den = x_after - x_off
            row['sum_rows_off_only'] = x_sum
            row['share_of_change_from_sum_rows'] = (x_after - x_sum) / den if den else None
        t9[f'{test}:{path}'] = row
    pen = {}
    for test in ('t_ii_b', 't_iii_indep', 't_iv_peak'):
        a_, o_ = frozen[test]['feasibility_penalty'], rec_off['tests'][test]['feasibility_penalty']
        pen[test] = {'after': a_, 'rows_off': o_, 'v2': v2[test]['feasibility_penalty'], 'abs_diff_off_vs_after': abs(o_ - a_),
                     'held': abs(o_ - a_) <= spec['predictions']['T9-F']['abs_tol']}
    soc = {}
    for test in ('t_ii_b', 't_iii_indep', 't_iv_peak'):
        out = {}
        for bk, bd in rec_off['tests'][test]['detectors']['block_day'].items():
            for ck, pc in bd['per_cohort'].items():
                s_off = pc['soc_mwh']
                s_v2 = v2[test]['detectors']['block_day'][bk]['per_cohort'][ck]['soc_mwh']
                s_af = frozen[test]['detectors']['block_day'][bk]['per_cohort'][ck]['soc_mwh']
                out[f'{bk}|{ck}'] = {'max_abs_off_minus_v2': max(abs(x - y) for x, y in zip(s_off, s_v2)),
                                     'max_abs_after_minus_v2': max(abs(x - y) for x, y in zip(s_af, s_v2))}
        soc[test] = out
    P['T9'] = {'R_rows_off': t9, 'F_penalty': pen, 'soc_trajectories': soc,
               'R_held_all': all(v['held'] for v in t9.values()), 'F_held_all': all(v['held'] for v in pen.values())}
    sh = [v.get('share_of_change_from_sum_rows') for v in t9.values() if 'share_of_change_from_sum_rows' in v]
    P['T9']['S_sum_rows_share'] = {'values': sh, 'threshold': spec['predictions']['T9-S']['min_share'],
                                   'held': bool(sh) and all(s is not None and s >= spec['predictions']['T9-S']['min_share']
                                                            for s in sh)}
    P['T9']['captures_rows_off'] = {c: {k: x.get(k) for k in ('test', 'deactivated_active_rows', 'mu_final',
                                                              'feasibility_penalty')}
                                    for c, x in cap_off['captures'].items()}
    P['T9']['captures_sum_off'] = {c: {k: x.get(k) for k in ('test', 'deactivated_active_rows', 'mu_final',
                                                             'feasibility_penalty')}
                                   for c, x in cap_sum['captures'].items()}
    ev['predictions'] = P
    ev['all_children_clean'] = all(not v['errors'] and not v['stage_guard']['verify_failures']
                                   and not v['harness_module_guard']['verify_failures'] and not v['capture_errors']
                                   and all(not p for p in v['row_map_problems'].values())
                                   for v in ev['children'].values())
    ev['total_solves'] = sum(v['stage_guard']['counts']['permitted_solve'] for v in ev['children'].values())
    ev['total_solves_ok'] = ev['total_solves'] == spec['guards']['expected_solves_total']
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir')
    ap.add_argument('--trial', action='store_true')
    ap.add_argument('--child')
    ap.add_argument('--work')
    ap.add_argument('--tag')
    ap.add_argument('--no-solve', action='store_true')
    a = ap.parse_args()
    if CHILD:
        return child_main(a)
    return parent_main(a)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
