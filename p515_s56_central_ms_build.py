"""P5.15 S56 -- V0: merged multi-scenario centralised model, BUILD ONLY (ZERO SOLVES; no unpickling).

Question (Planner task after audit D): can the production centralised merge (`centralized_coordination.combine_networks`
+ `Network.build_model`) build ONE 3 x 3 block (2025 Spring) of the multi-scenario instance, and what does it carry?

What it does (production functions only; nothing edited):
  - SolveProfileGuard(permitted=()) installed BEFORE any project import; every solve and solver exec raises; verified at
    exactly 0. pickle.load / pickle.loads blocked (counted, verified 0).
  - reads SRP1 (single scenario, data/SRP1/SRP1.json; combined scenario checksum == the SRP1 canonical constant) and the
    3 x 3 instance (data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json; checksum == the campaign
    declaration), both with the reader the 3 x 3 harness used (`p515_s44_scale_measurement.read_planning_from_derived_case`,
    production SharedResourcesPlanning; plots and logs redirected into the output directory). x = 0: no candidate.
  - combine_networks(TN, DNs) for each; Network.build_model(merged.params) of the 2025 Spring merged day network
    (the Step-1 probe's call, `p515_s55_central_feasibility_probe.py`); counts variables (all, free), constraints, by
    component; wall time and peak RSS.
  - asserts and records: scenario sets and probabilities identical across the TN and every DN (market and operation),
    for every (year, day) of the 3 x 3 instance; per-scenario DN data preserved in the merged network (every DN load's
    pd/qd and every kept DN generator's scenario arrays equal the DN's own, element by element); market price arrays
    TN vs DN; SRP1 2025 Spring merged model == 38,184 variables (the Step-1 probe); 3 x 3 2025 Spring size against the
    Advisor prediction 343,656; what the merge drops: shared ESS (combine_networks emptying
    `shared_energy_storages`, the code line anchored), the ADN node list, the DN reference buses and generators, and
    the multi-scenario commitment structure (row 18, interface-voltage pin, expected-value Vars, interface_delta, the
    settlement) -- each checked absent on the built model.
Output: NEW directory (refuses if it exists or another copy / a campaign runs); one JSON written through gate_result_io
and a manifest.
Usage: <canonical python> -u p515_s56_central_ms_build.py --spec SPEC --out-dir DIR [--trial]
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 V0 merged multi-scenario build (never solves)').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (V0 reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
from centralized_coordination import combine_networks  # noqa: E402

CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
CONCURRENT_SCRIPTS = ('p515_s44_campaign_harness.py', 'p515_s56_concurrency_gate.py', SCRIPT_REL,
                      'p515_s56_tn_realized_resolve.py')
SRP1_CANONICAL_CHECKSUM = '5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358'
SRP1_CASE = os.path.join('data', 'SRP1', 'SRP1.json')
MERGE_ANCHOR = ('centralized_coordination.py', 75,
                'combined_network.network[year][day].shared_energy_storages = list()')
CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [V0] {msg}', flush=True)


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


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def committed_clean(rel):
    return bool(_git('ls-files', '--', rel).strip()) and not _git('status', '--porcelain', '--', rel).strip()


def _input_record(path):
    tracked = bool(_git('ls-files', '--', path).strip())
    clean = (not _git('status', '--porcelain', '--', path).strip()) if tracked else None
    return {'path': path, 'sha256': _sha(path), 'tracked': tracked, 'clean': clean}


def concurrent_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(
                os.path.basename(t) in CONCURRENT_SCRIPTS for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == 'darwin' else r / 1024


class _Stages:
    @contextmanager
    def stage(self, name):
        t0 = time.time()
        _log(f'stage: {name}')
        yield
        _log(f'stage done ({time.time() - t0:.1f} s): {name}')


def counts(model):
    nv = sum(1 for _ in model.component_data_objects(pe.Var, active=True))
    nv_free = sum(1 for v in model.component_data_objects(pe.Var, active=True) if not v.fixed)
    nc = sum(1 for _ in model.component_data_objects(pe.Constraint, active=True))
    rows = {c.name: sum(1 for _ in c.values()) for c in model.component_objects(pe.Constraint, active=True)}
    vrows = {v.name: sum(1 for _ in v.values()) for v in model.component_objects(pe.Var, active=True)}
    return {'vars': nv, 'vars_free': nv_free, 'constraints': nc, 'rows_by_component': rows, 'vars_by_component': vrows}


def _eq(a, b):
    """Element-wise equality of nested scenario data (lists / dicts / numpy arrays / scalars)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        try:
            return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
        except Exception:  # noqa: BLE001
            return False
    return a == b


SCEN_ATTRS_SKIP = ('load_id', 'old_load_id', 'bus', 'gen_id', 'old_gen_id')


def scenario_structure(planning):
    """Per (year, day): TN and DN market / operation probabilities; all must be identical."""
    tn = planning.transmission_network
    out, ok = {}, True
    for y in tn.years:
        for d in tn.days:
            t = tn.network[y][d]
            rec = {'TN': {'market': list(t.prob_market_scenarios), 'operation': list(t.prob_operation_scenarios)}}
            for dn, dnet in planning.distribution_networks.items():
                n = dnet.network[y][d]
                rec[str(dn)] = {'market': list(n.prob_market_scenarios), 'operation': list(n.prob_operation_scenarios)}
                ok &= rec[str(dn)] == rec['TN']
                ok &= _eq(n.cost_energy_p, t.cost_energy_p)
            out[f'{y}|{d}'] = rec
    return out, ok


def dn_data_preserved(merged_day, planning, y, d):
    """Every DN load / kept DN generator in the merged day network carries its source element's data unchanged."""
    res = {'loads_checked': 0, 'generators_checked': 0, 'mismatches': []}
    for ld in merged_day.loads:
        lid = str(ld.load_id)
        if not lid.startswith('ADN_'):
            continue
        dn = int(lid.split('_')[1])
        src = next(x for x in planning.distribution_networks[dn].network[y][d].loads if x.load_id == ld.old_load_id)
        for k, v in vars(src).items():
            if k in SCEN_ATTRS_SKIP:
                continue
            if not _eq(v, getattr(ld, k, None)):
                res['mismatches'].append(['load', lid, k])
        res['loads_checked'] += 1
    for g in merged_day.generators:
        gid = str(g.gen_id)
        if not gid.startswith('ADN_'):
            continue
        dn = int(gid.split('_')[1])
        src = next(x for x in planning.distribution_networks[dn].network[y][d].generators if x.gen_id == g.old_gen_id)
        for k, v in vars(src).items():
            if k in SCEN_ATTRS_SKIP:
                continue
            if not _eq(v, getattr(g, k, None)):
                res['mismatches'].append(['gen', gid, k])
        res['generators_checked'] += 1
    return res


def build_one(label, planning, y, d):
    tn, dns = planning.transmission_network, planning.distribution_networks
    t0 = time.perf_counter()
    merged = combine_networks(tn, dns)
    merge_s = time.perf_counter() - t0
    md = merged.network[y][d]
    t1 = time.perf_counter()
    m = md.build_model(merged.params)
    build_s = time.perf_counter() - t1
    c = counts(m)
    # (present on the model, number of entries): the merged network keeps is_transmission, so the TN-side interface
    # components are DECLARED but indexed over the emptied ADN / shared-ESS sets
    absent = {n: [hasattr(m, n), (len(getattr(m, n)) if hasattr(m, n) else 0)]
              for n in ('row18_deviation_charge', 'scenario_voltage_pin', 'expected_interface_pf_p',
                        'expected_interface_vmag', 'interface_delta_p', 'interface_delta_q', 'pc_adn', 'qc_adn',
                        'pg_adn', 'shared_es_pnet', 'shared_es_soc')}
    rec = {'label': label, 'year': str(y), 'day': d, 'merge_s_all_blocks': merge_s, 'build_s': build_s,
           'peak_rss_mb_after': rss_mb(), **{k: c[k] for k in ('vars', 'vars_free', 'constraints')},
           'rows_by_component': c['rows_by_component'], 'vars_by_component': c['vars_by_component'],
           'n_scenarios_market': len(list(m.scenarios_market)), 'n_scenarios_operation': len(list(m.scenarios_operation)),
           'merged_counts': {'nodes': len(md.nodes), 'branches': len(md.branches), 'generators': len(md.generators),
                             'loads': len(md.loads), 'shared_energy_storages': len(md.shared_energy_storages),
                             'active_distribution_network_nodes': len(md.active_distribution_network_nodes),
                             'energy_storages': len(md.energy_storages)},
           'components_present_on_built_model': absent,
           'interface_settlement_weight': float(pe.value(m.interface_settlement_weight)),
           'interface_settlement_value_at_build': float(pe.value(m.interface_settlement)),
           'merged_prob_market': list(md.prob_market_scenarios),
           'merged_prob_operation': list(md.prob_operation_scenarios)}
    # what the merge drops relative to the sources
    rec['dropped'] = {
        'shared_energy_storages_in_TN_source': len(tn.network[y][d].shared_energy_storages),
        'shared_energy_storages_in_DN_sources': {str(k): len(v.network[y][d].shared_energy_storages)
                                                 for k, v in dns.items()},
        'dn_reference_buses_dropped': {str(k): v.network[y][d].get_reference_node_id() for k, v in dns.items()},
        'dn_reference_generators_dropped': {str(k): str(v.network[y][d].generators[
            v.network[y][d].get_reference_gen_idx()].gen_id) for k, v in dns.items()},
        'tn_adn_loads_dropped': [str(ld.load_id) for ld in tn.network[y][d].loads
                                 if ld.bus in tn.network[y][d].active_distribution_network_nodes]}
    return rec, md, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    a = ap.parse_args()
    pre = []
    spec_sha = _sha(a.spec)
    tag = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not spec_sha.startswith(tag):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
    if not a.trial:
        for rel in (SCRIPT_REL, os.path.relpath(os.path.abspath(a.spec), REPO)):
            if not committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    if os.path.exists(CAMPAIGN_LOCK):
        pre.append(f'campaign lock {CAMPAIGN_LOCK} exists')
    others = concurrent_processes()
    if others:
        pre.append(f'concurrent processes: {others}')
    if pre:
        print('[V0] PRECONDITION FAILED:', pre, flush=True)
        return 1
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    v0 = spec['V0']
    t_start = time.perf_counter()
    os.makedirs(a.out_dir)
    res = {'stage': spec['stage'] + ' -- V0', 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'utc_start': datetime.now(timezone.utc).isoformat(), 'trial': a.trial, 'interpreter': sys.executable,
           'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()), verify(0)); no unpickling'}
    line = open(MERGE_ANCHOR[0]).read().splitlines()[MERGE_ANCHOR[1] - 1]
    _ck(f'anchor {MERGE_ANCHOR[0]}:{MERGE_ANCHOR[1]} holds the shared-ESS emptying', MERGE_ANCHOR[2] in line, line)
    res['modules'] = {p: _input_record(p) for p in ('centralized_coordination.py', 'network.py', 'network_data.py',
                                                    'model_construction_helpers.py', 'p515_s44_scale_measurement.py',
                                                    SRP1_CASE, v0['instance_3x3']['case_path'])}
    out = {}
    # ---- SRP1, single scenario: the Step-1 probe's 38,184
    p1 = S.read_planning_from_derived_case({'derived_case': {'path': SRP1_CASE}}, os.path.join(a.out_dir, 'srp1'),
                                           _Stages())
    ck1 = (p1.scenario_metadata or {}).get('combined_scenario_checksum')
    _ck('SRP1 combined scenario checksum == canonical', ck1 == SRP1_CANONICAL_CHECKSUM, ck1)
    y1 = next(y for y in p1.transmission_network.years if int(y) == int(v0['year']))
    r1, _md1, m1 = build_one('SRP1 single scenario', p1, y1, v0['day'])
    _ck(f"SRP1 {v0['year']} {v0['day']} merged model vars == {v0['srp1_expected_vars']} (Step-1 probe)",
        r1['vars'] == v0['srp1_expected_vars'], r1['vars'])
    out['srp1'] = r1
    del m1, _md1, p1
    # ---- 3 x 3
    inst = v0['instance_3x3']
    _ck('3 x 3 case sha256 == declaration', _sha(inst['case_path']) == inst['case_sha256'])
    p3 = S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                           os.path.join(a.out_dir, 's53_3x3'), _Stages())
    ck3 = (p3.scenario_metadata or {}).get('combined_scenario_checksum')
    _ck('3 x 3 combined scenario checksum == declaration', ck3 == inst['scenario_checksum'], ck3)
    struct, sok = scenario_structure(p3)
    _ck('3 x 3: scenario sets and probabilities identical across TN and every DN, and TN price arrays == DN, '
        'every (year, day)', sok)
    out['scenario_structure_3x3'] = struct
    y3 = next(y for y in p3.transmission_network.years if int(y) == int(v0['year']))
    r3, md3, m3 = build_one('3 x 3', p3, y3, v0['day'])
    pres = dn_data_preserved(md3, p3, y3, v0['day'])
    _ck('3 x 3: per-scenario DN data preserved in the merged network (every DN load and kept generator)',
        not pres['mismatches'] and pres['loads_checked'] > 0, pres)
    r3['dn_data_preserved'] = pres
    _ck('3 x 3: merged model indexed over 3 market x 3 operation scenarios',
        r3['n_scenarios_market'] == 3 and r3['n_scenarios_operation'] == 3)
    _ck('3 x 3: merged probabilities == TN probabilities',
        r3['merged_prob_market'] == struct[f"{y3}|{v0['day']}"]['TN']['market']
        and r3['merged_prob_operation'] == struct[f"{y3}|{v0['day']}"]['TN']['operation'])
    pres_c = r3['components_present_on_built_model']
    _ck('3 x 3: no row 18, no voltage pin, no expected-value Vars (absent); interface_delta, pc_adn and shared-ESS '
        'components have 0 entries',
        not any(pres_c[k][0] for k in ('row18_deviation_charge', 'scenario_voltage_pin', 'expected_interface_pf_p',
                                       'expected_interface_vmag'))
        and all(pres_c[k][1] == 0 for k in pres_c), pres_c)
    _ck('3 x 3: merged shared_energy_storages and ADN node list empty',
        r3['merged_counts']['shared_energy_storages'] == 0
        and r3['merged_counts']['active_distribution_network_nodes'] == 0, r3['merged_counts'])
    ratio = r3['vars'] / r1['vars']
    r3['vars_ratio_to_srp1'] = ratio
    r3['vars_minus_prediction'] = r3['vars'] - v0['predicted_vars_3x3']
    r3['vars_rel_dev_from_prediction'] = (r3['vars'] - v0['predicted_vars_3x3']) / v0['predicted_vars_3x3']
    by_comp_ratio = {k: (r3['vars_by_component'].get(k, 0), r1['vars_by_component'].get(k, 0))
                     for k in set(r3['vars_by_component']) | set(r1['vars_by_component'])}
    r3['components_not_scaling_x9'] = {k: v for k, v in by_comp_ratio.items() if v[0] != 9 * v[1]}
    out['3x3'] = r3
    res['results'] = out
    res['prediction_P5'] = {
        'builds': True, 'vars_3x3': r3['vars'], 'predicted': v0['predicted_vars_3x3'],
        'rel_dev': r3['vars_rel_dev_from_prediction'], 'tolerance_rel': v0['prediction_tolerance_rel'],
        'srp1_vars': r1['vars'], 'srp1_expected': v0['srp1_expected_vars'],
        'held': (abs(r3['vars_rel_dev_from_prediction']) <= v0['prediction_tolerance_rel']
                 and r1['vars'] == v0['srp1_expected_vars'])}
    res['total_wall_s'] = time.perf_counter() - t_start
    res['peak_rss_mb'] = rss_mb()
    gv = GUARD.verify(0, 0)
    res['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': gv}
    res['pickle'] = dict(PK)
    res['guards_ok'] = (not gv) and PK == {'load': 0, 'loads': 0}
    res['checks'] = CHECKS
    res['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(a.out_dir, 'central_ms_build_results.json')
    with open(path, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump({'outputs': {os.path.basename(path): _sha(path)}, 'inputs': res['modules'],
                   'spec': {'path': a.spec, 'sha256': spec_sha}}, h, indent=1)
    _log(f"wrote {path}; guards_ok {res['guards_ok']}; failed checks {res['failed_checks']}")
    return 0 if (res['guards_ok'] and not res['failed_checks']) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
