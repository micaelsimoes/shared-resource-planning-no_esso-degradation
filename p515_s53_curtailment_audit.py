"""
P5.15 Addendum 41 (task W53) -- renewable-curtailment audit: ZERO SOLVES, NO MODEL CONSTRUCTION, read-only.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 41 ("Zero-solve read first"; "Decision rule, recorded"); Planner
task W53. Nothing is built or solved: an armed `SolveProfileGuard(permitted=())` is installed before any Pyomo /
production import and verified at exactly 0 at the end; `Network.build_model`, `NetworkData.build_model`,
`SharedEnergyStorageData.build_subproblem` / `.build_master_problem` are replaced by RAISING blockers for the whole
run (declared count 0). The campaign lock is not taken. No existing file is modified; output is write-once.

INSTANCES (certified records only; every input verified against its campaign's COMMITTED manifest before use):
  srp1_x0    SRP1, x = 0 (candidate 8435c718...), campaign s48_x0_capture, eval d2c96b14... (132 cycles, ruled PASS by
             P5_15_S48_X0_CAPTURE_GATE_RULING.md 8f9d67f2); persisted terminal models `certified_models.pkl`
             (hash-recorded, not committed) -> PER GENERATOR x HOUR resolution, with duals.
  srp1_unit  SRP1, node 7 0.25 MVA / 1.0 MWh, 2025 (candidate db77e154...), campaign s47_recert, eval bd504ecf...
             (112 cycles); the identical eval key in s47_a1a_baseline is checked equal. No persisted models, no terminal
             workbook -> BLOCK resolution only (network x year x day, day-summed), from component_levels_terminal.json.
  pilot_x0 / pilot_unit
             the 2 x 2 pilot, campaign s52_pilot_nopersist (f239ee02), evals 7d53b6f2... (x = 0, 74 cycles) and
             711fce9a... (node 7 0.25 / 1.0, 72 cycles). Run WITHOUT persist_certified_models, BUT each eval wrote
             production's operational workbook of the terminal models BEFORE post-certification
             (`results/SRP1_distributed_terminal.xlsx`, hash-recorded) -> PER GENERATOR x HOUR x SCENARIO resolution
             of pg_avail (sheet `Generation`, row `Pg, [MW]` of a curtaillable generator = pg_avail * baseMVA,
             network.py `_process_results`) and pg (row `Pg_net, [MW]`); NO duals (no models).

FORMULAS (units: MW per 1 h period = MWh; B = baseMVA; omega_s = prob_market * prob_operation of scenario s)
  c[g,s,t]        = (pg_avail[g,s,t] - pg[g,s,t]) * B       the production definitional curtailment
                    (model_construction_helpers.gen_curtailment_penalty / gen_curtailment_definitional_value); may be
                    negative by up to EQUALITY_TOLERANCE * B per generator-hour (pg upper bound = pg_avail + tol).
  TOL_MW          = EQUALITY_TOLERANCE * B = 1e-3 MW: the bound tolerance band; c <= TOL_MW is "not curtailed".
  per network-hour: V_net = sum_g c, V_plus = sum_g max(c, 0), V_tol = sum_g c [c > TOL_MW], A = sum_g pg_avail * B.
  per block (network, year, day): E_net = sum_s omega_s sum_t V_net  -- reconciled against the certified
                    component_levels_terminal `res_curtailment_definitional_at_weight_1` (unweighted, per block).
  horizon (Q weighting): H_x = sum_blocks w_b * (per-block omega-weighted day sum), w_b = admm_block_weight (the same
                    weight the certified Q uses).
  priced curtailment at the hourly market price (what the "priced" re-baseline branch would charge at the
                    CURRENT dispatch): C = sum_b w_b sum_s omega_s sum_t pi[s,t] * V_plus[s,t]  [EUR, Q units].
                    Since pi >= 0 (checked) and the current point is feasible for the priced problem,
                    0 <= Q_priced(x) - Q_current(x) <= C(x) at a global optimum (a first-order bound, not a solve).
  resolution      Q level: C(x) against bar(x) (record bar: max |gross step| over the last 10 cycles);
                    value level: first-order dC = C(0) - C(unit) against bar(0) + bar(unit), and the rigorous bound
                    |d value| <= max(C(0), C(unit)) against the same.
  reachability    (Addendum 41) the shared ESS sits at the TN/DN interface busbar: in the DSO model it is in the
                    node balance of the DN reference bus (bus 1, 345 kV, upstream of the DN transformer 1-2), and the
                    DSO's settled / committed exchange is pg_adn = pg[ref] - shared_es_pnet (net of the storage); in
                    the TSO model it is in the node balance of TN bus 5/7/9 with the DN load. TSO-side RES
                    curtailment (case9 buses 4/6/8) is REACHABLE; DN-side RES curtailment (behind the DSO exchange
                    point) is NOT (Addendum 41's classification, applied mechanically).
CAUSE (where identifiable)
  srp1_x0 (duals): per curtailed generator-hour (c > TOL_MW): 'capability_bound' when the apparent-power row
                    sg_capability (pg^2 + qg^2 <= sg_avail^2) is active (slack <= CAP_SLACK_TOL p.u.^2; its dual
                    recorded) -- P is below pg_avail because the inverter also supplies Q at its S limit; else
                    'interior' (zU of pg ~ 0): the bus's marginal value of RES is ~0 (LMP_b = dual(node_balance_p)/B
                    reported with the active voltage / branch-flow rows of that network-hour).
  pilot (no duals): primal indicators only -- apparent-power curtailment Sg_curt against c (real curtailment below
                    the S capability vs P-for-Q at capability); the row-18 threshold condition
                    d[s,t] <= +D_TOL and pi[m(s),t] < alpha * pibar_t (d = p_int[s,t] - pbar_t from
                    multiscenario_terminal.json; pibar_t = sum_m omega_m pi[m,t]); max LV voltage of the DN and the
                    DN transformer loading (sheets Voltage, Branch Loading); for TSO entries whether every
                    conventional unit is at Pmin. These are INDICATORS, not a dual-established cause.
                    Row-18 condition (W53, stated before the evidence run): d[s,t] <= +D_TOL (import at or below the
                    committed schedule, so one more MWh of RES opens or widens a deviation, costing alpha*pibar_t,
                    while saving only pi[m(s),t]) and pi[m(s),t] < alpha * pibar_t.

Output (write-once, new directory): data/SRP1/Results/P515S53/curtailment_audit/{curtailment_audit.json,
curtailment_audit.md, manifest_sha256.json (written by `--manifest`), launch.log (shell)}.
EXACT COMMAND (repo root; attached; both streams; noclobber):
    mkdir data/SRP1/Results/P515S53/curtailment_audit && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_curtailment_audit.py \\
        > data/SRP1/Results/P515S53/curtailment_audit/launch.log 2>&1 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s53_curtailment_audit.py --manifest \\
        >> data/SRP1/Results/P515S53/curtailment_audit/launch.log 2>&1
"""
import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W53 curtailment audit (zero solves)').install()

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.calculus.derivatives import Modes, differentiate  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402

from definitions import EQUALITY_TOLERANCE, PENALTY_GENERATION_CURTAILMENT  # noqa: E402

STAGE = 'P5.15 Addendum 41 W53 -- renewable-curtailment audit (zero solves, no model construction)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 41', 'Planner task W53']
_R = os.path.join('data', 'SRP1', 'Results')
OUT_REL = os.path.join(_R, 'P515S53', 'curtailment_audit')
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
SRP1_UNIT_KEY = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'
UNIT_CANONICAL = {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}}
RULING = {'path': 'P5_15_S48_X0_CAPTURE_GATE_RULING.md', 'commit': '8f9d67f2'}
PILOT_COMMIT = 'f239ee02'
BASELINE_LABEL_PREFIX = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'   # the current baseline, not C3-era
W25_JSON = os.path.join(_R, 'P515S47', 'tso_marginal_cost', 'tso_marginal_cost.json')
W43_JSON = os.path.join(_R, 'P515S51', 'mechanism_analysis', 'w43_r1', 'analysis.json')
BITWISE_GATE_JSON = os.path.join(_R, 'P515S52', 'srp1_bitwise_gate', 'gate.json')
CASE_FMT = {'TSO': os.path.join('data', 'SRP1', 'case9', 'case9_{y}.json'),
            5: os.path.join('data', 'SRP1', 'case33_1', 'case33_1_{y}.json'),
            7: os.path.join('data', 'SRP1', 'case33_2', 'case33_2_{y}.json'),
            9: os.path.join('data', 'SRP1', 'case33_3', 'case33_3_{y}.json')}
DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
N_T = 24

INSTANCES = {
    'srp1_x0': {'label': 'SRP1, x = 0', 'campaign_root': os.path.join(_R, 'P515S48', 'x0_capture'),
                'point': ('point', None), 'eval_dir_name': 'd2c96b1480402a3b_x0', 'candidate_key': X0_KEY,
                'source': 'models', 'models_rel': 'certified_models.pkl',
                'models_sha256': '03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce'},
    'srp1_unit': {'label': 'SRP1, node 7 0.25 MVA / 1.0 MWh (2025)',
                  'campaign_root': os.path.join(_R, 'P515S47', 'campaign_s47_recert'),
                  'point': ('points', 'n7_4h_e1'), 'eval_dir_name': 'bd504ecf5a288d44_n7_4h_e1',
                  'candidate_key': SRP1_UNIT_KEY, 'source': 'component_levels',
                  'duplicate': {'campaign_root': os.path.join(_R, 'P515S47', 'campaign_s47_a1a_baseline'),
                                'point': ('points', 'n7_4h_e1'), 'eval_dir_name': 'bd504ecf5a288d44_n7_4h_e1'}},
    'pilot_x0': {'label': '2x2 pilot, x = 0', 'campaign_root': os.path.join(_R, 'P515S52', 'campaign_s52_pilot_nopersist'),
                 'point': ('points', 'x0'), 'eval_dir_name': '7d53b6f21b686a44_x0', 'candidate_key': X0_KEY,
                 'source': 'workbook', 'workbook_rel': os.path.join('results', 'SRP1_distributed_terminal.xlsx'),
                 'workbook_sha256': 'ecbb11f43ac62272a40dc1973b0bb3bc3ab914f89d91ff180d51df2b6b2148ee'},
    'pilot_unit': {'label': '2x2 pilot, node 7 0.25 MVA / 1.0 MWh (2025)',
                   'campaign_root': os.path.join(_R, 'P515S52', 'campaign_s52_pilot_nopersist'),
                   'point': ('points', 'n7_4h_e1'), 'eval_dir_name': '711fce9aa74d6878_n7_4h_e1',
                   'candidate_key': SRP1_UNIT_KEY,
                   'source': 'workbook', 'workbook_rel': os.path.join('results', 'SRP1_distributed_terminal.xlsx'),
                   'workbook_sha256': '1a28ceb214dc44c274d2d046ed87d85db8bbf6117dd464c0a9b5492a3afe499d'},
}
PAIRS = {'srp1': ('srp1_x0', 'srp1_unit'), 'pilot': ('pilot_x0', 'pilot_unit')}

TOL_FACTOR = EQUALITY_TOLERANCE          # TOL_MW = EQUALITY_TOLERANCE * baseMVA
CAP_SLACK_TOL = 1e-6                     # sg_capability active when (sg_avail^2 - sg_sqr) <= this (p.u.^2)
DUAL_TOL = 1e-3                          # a voltage / branch row is listed as active when |dual| >= this (raw)
ROW_SLACK_TOL = 1e-6                     # ... and its slack <= this (raw row units)
D_TOL_MW = 1e-3                          # row-18 deviation d counted as negative when d < -D_TOL_MW
V_AT_BOUND_TOL = 1e-4                    # p.u.; a DN voltage is 'at 1.1' when >= Vmax - this
TRAFO_AT_RATING_TOL = 1e-3               # workbook 'Flow_ij' of DN branch 1-2 (1.0 = at rating) counted at rating above 1 - this
CONV_PMIN_TOL_MW = 1e-2                  # TSO conventional unit 'at Pmin' when Pg <= Pmin + this
RECON_REL_TOL = 1e-6                     # reconciliation against component_levels (relative, floor 1 MWh)


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout


def _tracked_clean(rel):
    return {'tracked': bool(_git(['ls-files', '--', rel]).strip()),
            'clean': not _git(['status', '--porcelain', '--', rel]).strip()}


def _load_json(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


# ======================================================================================================================
#  model-construction blockers (declared count 0)
# ======================================================================================================================
class ConstructionBlockers:
    def __init__(self):
        import network as network_module
        import network_data as network_data_module
        import shared_energy_storage_data as sed
        self.targets = [(network_module.Network, 'build_model'), (network_data_module.NetworkData, 'build_model'),
                        (sed.SharedEnergyStorageData, 'build_subproblem'),
                        (sed.SharedEnergyStorageData, 'build_master_problem')]
        self.blocked_calls = []
        self.originals = {}

    def install(self):
        def blocker(name):
            def _raise(*a, **k):
                self.blocked_calls.append(name)
                raise RuntimeError(f'W53: model construction blocked ({name})')
            return _raise
        for cls, attr in self.targets:
            self.originals[(cls, attr)] = getattr(cls, attr)
            setattr(cls, attr, blocker(f'{cls.__name__}.{attr}'))
        return self

    def uninstall(self):
        for (cls, attr), fn in self.originals.items():
            setattr(cls, attr, fn)


# ======================================================================================================================
#  inputs: certified records, verified against the committed manifests (before any analysis)
# ======================================================================================================================
def _point(results, spec):
    key, sub = spec
    return results.get(key) if sub is None else (results.get(key) or {}).get(sub)


def resolve_instance(name, spec):
    failures = []
    root = spec['campaign_root']
    results_rel = os.path.join(root, 'campaign_results.json')
    manifest_rel = os.path.join(root, 'campaign_manifest_sha256.json')
    results = _load_json(results_rel)
    manifest = _load_json(manifest_rel)
    point = _point(results, spec['point']) or {}
    eval_rel = os.path.join(root, 'evals', spec['eval_dir_name'])
    files = {'campaign_results': results_rel, 'campaign_manifest': manifest_rel,
             'evaluation_record': os.path.join(eval_rel, 'evaluation_record.json'),
             'component_levels': os.path.join(eval_rel, 'component_levels_terminal.json')}
    if spec['source'] == 'workbook':
        files['multiscenario_terminal'] = os.path.join(eval_rel, 'multiscenario_terminal.json')
        files['workbook'] = os.path.join(eval_rel, spec['workbook_rel'])
    if spec['source'] == 'models':
        files['models'] = os.path.join(eval_rel, spec['models_rel'])
    verified = {}
    for role, rel in files.items():
        tc = _tracked_clean(rel)
        present = os.path.isfile(os.path.join(REPO, rel))
        sha = _sha(os.path.join(REPO, rel)) if present else None
        entry = {'path': rel, 'present': present, 'sha256': sha, **tc}
        if role in ('campaign_results', 'campaign_manifest'):
            entry['ok'] = present and tc['tracked'] and tc['clean']
        else:
            entry['manifest_sha256'] = manifest.get(rel)
            entry['ok'] = present and sha is not None and sha == manifest.get(rel) and (
                (tc['tracked'] and tc['clean']) or role in ('workbook', 'models'))
        verified[role] = entry
        if not entry['ok']:
            failures.append(f'{name}: input {role} not verified ({rel})')
    if spec['source'] == 'models' and verified['models']['sha256'] != spec['models_sha256']:
        failures.append(f'{name}: persisted models sha256 differs from the recorded {spec["models_sha256"]}')
    if spec['source'] == 'workbook' and verified['workbook']['sha256'] != spec['workbook_sha256']:
        failures.append(f'{name}: workbook sha256 differs from the recorded {spec["workbook_sha256"]}')
    record = _load_json(files['evaluation_record'])
    status_ok = point.get('status') == 'certified' and record.get('status') == 'certified'
    eval_key = point.get('eval_key') or ''
    key_ok = (eval_key.startswith(spec['eval_dir_name'].split('_')[0]) and record.get('eval_key') == eval_key
              and point.get('candidate_key') == record.get('candidate_key'))
    if spec['candidate_key'] is not None:
        key_ok = key_ok and point.get('candidate_key') == spec['candidate_key']
    canon = point.get('candidate_canonical') or record.get('candidate_canonical') or {}
    if name.endswith('_unit'):
        canon_ok = (int(canon.get('investment_year', -1)) == UNIT_CANONICAL['investment_year'] and
                    {str(k): [float(a) for a in v] for k, v in (canon.get('nodes') or {}).items()}
                    == UNIT_CANONICAL['nodes'])
    else:
        canon_ok = point.get('candidate_key') == X0_KEY
    if not status_ok:
        failures.append(f'{name}: point not certified')
    if not str(point.get('LABEL', '')).startswith(BASELINE_LABEL_PREFIX):
        failures.append(f'{name}: not a current-baseline record (LABEL {point.get("LABEL")!r})')
    if not key_ok:
        failures.append(f'{name}: eval / candidate key mismatch')
    if not canon_ok:
        failures.append(f'{name}: candidate canonical is not the declared instance')
    q = (point.get('certified_cost_gross_settlement_excluded') if 'certified_cost_gross_settlement_excluded' in point
         else point.get('certified_cost_gross'))
    info = {'label': spec['label'], 'source': spec['source'], 'campaign_root': root,
            'ageing_baseline_label': point.get('LABEL'),
            'campaign_id': record.get('campaign_id'), 'campaign_spec_sha256': results.get('campaign_spec_sha256'),
            'git_head_at_run': results.get('git_head_at_run'), 'eval_dir': eval_rel, 'eval_key': eval_key,
            'candidate_key': point.get('candidate_key'), 'candidate_canonical': canon,
            'status': point.get('status'), 'cycles_run': point.get('cycles_run'),
            'certification_cycle': point.get('certification_cycle'), 'Q_gross_settlement_excluded': q,
            'record_certified_cost': record.get('certified_cost'),
            'bar': (record.get('bar') or {}).get('value'), 'bar_definition': (record.get('bar') or {}).get('definition'),
            'rule_ten_terminal_step_over_threshold': (record.get('rule_ten') or {}).get('terminal_step_over_threshold'),
            'objective_convention': record.get('objective_convention'),
            'wall_time_s': (record.get('wall_time_s') or {}).get('child_process_s'),
            'wall_time_record': record.get('wall_time_s'),
            'inputs_verified': verified,
            'checks': {'certified': status_ok, 'keys': key_ok, 'canonical': canon_ok}}
    if spec.get('duplicate'):
        d = spec['duplicate']
        dres = _load_json(os.path.join(d['campaign_root'], 'campaign_results.json'))
        dman = _load_json(os.path.join(d['campaign_root'], 'campaign_manifest_sha256.json'))
        dpt = _point(dres, d['point']) or {}
        dcl_rel = os.path.join(d['campaign_root'], 'evals', d['eval_dir_name'], 'component_levels_terminal.json')
        dsha = _sha(os.path.join(REPO, dcl_rel))
        info['duplicate'] = {'campaign_root': d['campaign_root'], 'eval_key': dpt.get('eval_key'),
                             'status': dpt.get('status'), 'component_levels': dcl_rel, 'sha256': dsha,
                             'manifest_sha256': dman.get(dcl_rel), **_tracked_clean(dcl_rel),
                             'Q_gross_settlement_excluded': dpt.get('certified_cost_gross_settlement_excluded'),
                             'same_eval_key': dpt.get('eval_key') == eval_key}
        if not (dsha == dman.get(dcl_rel) and dpt.get('eval_key') == eval_key and dpt.get('status') == 'certified'):
            failures.append(f'{name}: duplicate record not verified')
    if name == 'srp1_x0':
        rp = os.path.join(REPO, RULING['path'])
        blob = _git(['rev-parse', f'{RULING["commit"]}:{RULING["path"]}']).strip()
        ok = (os.path.isfile(rp) and blob == _git(['hash-object', RULING['path']]).strip()
              and not _git(['status', '--porcelain', '--', RULING['path']]).strip()
              and '**PASS.**' in open(rp).read())
        info['gate_ruling'] = {**RULING, 'verified_committed_unmodified_and_pass': ok}
        if not ok:
            failures.append('srp1_x0: gate ruling not verified')
    return info, failures


# ======================================================================================================================
#  penalty in force, per path, by file:line at each run's commit and at HEAD
# ======================================================================================================================
LINE_PATTERNS = [
    ('definitions.py', 'PENALTY_GENERATION_CURTAILMENT = ', 'the constant (initialisation / standalone weight)'),
    ('definitions.py', 'EQUALITY_TOLERANCE = ', 'bound tolerance (pg upper bound = pg_avail + tol)'),
    ('model_construction_helpers.py', 'model.penalty_gen_curtailment = pe.Param(initialize=PENALTY_GENERATION_CURTAILMENT',
     'Param created at the constant on every build (first hit: OBJ_MIN_COST, which every SRP1 network uses)'),
    ('model_construction_helpers.py', 'gen_curt_penalty += penalty * network.baseMVA * (model.pg_avail',
     'the objective term: penalty * B * (pg_avail - pg)'),
    ('model_construction_helpers.py', 'model.interface_settlement_weight = pe.Param(initialize=0.00',
     'settlement weight 0 on every build (the initialisation import is unpriced)'),
    ('model_construction_helpers.py', 'return m.pg[ref_gen_idx, s_m, s_o, p] - shared_ess_p',
     'pg_adn = pg[ref] - shared_es_pnet: the DSO exchange is net of the storage at the reference bus'),
    ('model_construction_helpers.py', 'Pd += model.shared_es_pnet[e, s_m0, s_o0, p]',
     'the shared ESS enters the node balance of its own bus'),
    ('shared_resources_planning.py', 'results = transmission_network.optimize(tso_model)',
     'TSO initialisation / benchmark solve at the build weights'),
    ('shared_resources_planning.py', 'results[node_id] = distribution_network.optimize(dso_model)',
     'DSO initialisation / benchmark solve (sequential) at the build weights'),
    ('shared_resources_planning.py', 'res = distribution_network.optimize(dso_model)',
     'DSO initialisation solve (parallel worker) at the build weights'),
    ('shared_resources_planning.py', '_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)',
     'ADMM path: DSO weights reset after the initialisation solve'),
    ('shared_resources_planning.py', '_prepare_transmission_objectives_for_admm(transmission_network, tso_model)',
     'ADMM path: TSO weights reset after the initialisation solve'),
    ('shared_resources_planning.py', ' model[year][day].penalty_gen_curtailment.set_value(0.00)',
     'ADMM TSO subproblem: curtailment weight set to 0'),
    ('shared_resources_planning.py', 'dso_model[year][day].penalty_gen_curtailment.set_value(0.00)',
     'ADMM DSO subproblem: curtailment weight set to 0'),
    ('shared_resources_planning.py', 'interface_settlement_weight.set_value(1.00)',
     'ADMM path: settlement weight set to 1'),
    ('shared_resources_planning.py', "results['tso'] = transmission_network.optimize(tso_model)",
     'TSO solve in the hierarchical / uncoordinated benchmarks (enclosing function shown)'),
    ('shared_resources_planning.py', "results['dso'][node_id] = distribution_network.optimize(dso_model)",
     'DSO solve in the hierarchical / uncoordinated benchmarks (enclosing function shown)'),
]


def _enclosing_def(lines, idx):
    for j in range(idx, -1, -1):
        m = re.match(r'^\s*def\s+([A-Za-z_0-9]+)', lines[j])
        if m:
            return m.group(1)
    return None


def penalty_provenance(commits):
    out = {}
    for label, commit in commits.items():
        full = _git(['rev-parse', commit]).strip()
        per = {'commit': full, 'patterns': [], 'all_references_outside_diagnostics': []}
        cache = {}
        for fname, pattern, role in LINE_PATTERNS:
            if fname not in cache:
                cache[fname] = _git(['show', f'{full}:{fname}']).splitlines()
            lines = cache[fname]
            hits = [{'line': i + 1, 'text': ln.strip(), 'def': _enclosing_def(lines, i)}
                    for i, ln in enumerate(lines) if pattern in ln]
            per['patterns'].append({'file': fname, 'pattern': pattern, 'role': role, 'hits': hits})
        grep = _git(['grep', '-n', '-e', 'penalty_gen_curtailment', '-e', 'PENALTY_GENERATION_CURTAILMENT', full,
                     '--', '*.py'])
        for row in grep.splitlines():
            parts = row.split(':', 3)
            if len(parts) < 4:
                continue
            fname, lineno, text = parts[1], int(parts[2]), parts[3]
            if re.match(r'^p\d', os.path.basename(fname)):
                continue
            if fname not in cache:
                cache[fname] = _git(['show', f'{full}:{fname}']).splitlines()
            per['all_references_outside_diagnostics'].append(
                {'file': fname, 'line': lineno, 'text': text.strip(), 'def': _enclosing_def(cache[fname], lineno - 1)})
        per['scope_of_reference_search'] = ('git grep of every tracked *.py at the commit for penalty_gen_curtailment / '
                                            'PENALTY_GENERATION_CURTAILMENT, excluding diagnostic harnesses '
                                            '(basename matching ^p[0-9])')
        out[label] = per
    return out


# ======================================================================================================================
#  srp1_x0: persisted terminal models (per generator x hour, with duals)
# ======================================================================================================================
def _base_mva(net, year):
    return float(_load_json(CASE_FMT[net].format(y=year))['baseMVA'])


def _curtailable_gens(m):
    gens = set()
    for k in m.gen_curt_penalty_scenario:
        for v in identify_variables(m.gen_curt_penalty_scenario[k].expr, include_fixed=True):
            if v.parent_component().local_name == 'pg':
                gens.add(v.index()[0])
    return sorted(gens)


def _gen_bus(m):
    out = {}
    t0 = next(iter(m.periods))
    for idx, row in m.node_balance_p.items():
        if idx[-1] != t0:
            continue
        for v in identify_variables(row.body, include_fixed=True):
            if v.parent_component().local_name == 'pg':
                out[v.index()[0]] = idx[0]
    return out


def _storage_buses(m):
    out = set()
    if not hasattr(m, 'shared_es_pnet'):
        return []
    t0 = next(iter(m.periods))
    for idx, row in m.node_balance_p.items():
        if idx[-1] != t0:
            continue
        if any(v.parent_component().local_name == 'shared_es_pnet' for v in identify_variables(row.body, include_fixed=True)):
            out.add(idx[0])
    return sorted(out)


def _dso_price_series(m, B, ref_gen):
    """pi_t = d(interface_settlement)/d(pg[ref]) / (omega * B): exact read of the market price the DSO settles at."""
    s_m, s_o = next(iter(m.scenarios_market)), next(iter(m.scenarios_operation))
    omega = 1.0
    if len(m.scenarios_market) * len(m.scenarios_operation) != 1:
        raise RuntimeError('SRP1 models are expected to hold one scenario')
    pis = []
    for t in m.periods:
        g = differentiate(m.interface_settlement.expr, wrt=m.pg[ref_gen, s_m, s_o, t], mode=Modes.reverse_numeric)
        pis.append(float(g) / (omega * B))
    return pis


def _active_rows(m, fam, s_m, s_o, t, transformer_idx=()):
    """Rows of `fam` at (s_m, s_o, t) with |dual| >= DUAL_TOL and slack <= ROW_SLACK_TOL. Branch rows carry
    `is_transformer` (the branch index has a ratio Var `r`, i.e. it is a transformer branch)."""
    rows = []
    comp = getattr(m, fam, None)
    if comp is None:
        return rows
    for idx, c in comp.items():
        if idx[-1] != t or idx[-3:-1] != (s_m, s_o):
            continue
        lam = m.dual.get(c)
        if lam is None or abs(lam) < DUAL_TOL:
            continue
        body = pe.value(c.body)
        slack = min((pe.value(c.upper) - body) if c.has_ub() else float('inf'),
                    (body - pe.value(c.lower)) if c.has_lb() else float('inf'))
        if slack <= ROW_SLACK_TOL:
            row = {'row': f'{fam}[{idx[0]}]', 'dual_raw': float(lam), 'slack': float(slack)}
            if fam.startswith('branch'):
                row['is_transformer'] = idx[0] in transformer_idx
            rows.append(row)
    return rows


def _network_hour_rows(m, s_m, s_o, t, transformer_idx):
    return {'voltage': (_active_rows(m, 'voltage_magnitude_upper_cons', s_m, s_o, t)
                        + _active_rows(m, 'voltage_magnitude_lower_cons', s_m, s_o, t)),
            'branch': (_active_rows(m, 'branch_flow_limit', s_m, s_o, t, transformer_idx)
                       + _active_rows(m, 'branch_flow_limit_ji', s_m, s_o, t, transformer_idx))}


def analyse_srp1_models(pickle_path, cl_blocks, w25):
    with open(pickle_path, 'rb') as handle:
        payload = pickle.load(handle)
    nets = [('TSO', payload['tso'])] + [(n, payload['dso'][n]) for n in sorted(payload['dso'])]
    blocks, entries, readback, topology = {}, [], [], {}
    price_by_block = {}
    for net, years in nets:
        for y in sorted(years):
            for d in DAYS:
                m = years[y][d]
                B = _base_mva(net, y)
                tol = TOL_FACTOR * B
                bkey = f'TSO|{y}|{d}' if net == 'TSO' else f'DSO|{net}|{y}|{d}'
                gens = _curtailable_gens(m)
                gbus = _gen_bus(m)
                (s_m, s_o), = [(a, b) for a in m.scenarios_market for b in m.scenarios_operation]
                readback.append({'block': bkey, 'penalty_gen_curtailment': float(pe.value(m.penalty_gen_curtailment)),
                                 'interface_settlement_weight': float(pe.value(m.interface_settlement_weight)),
                                 'n_active_objectives': len(list(m.component_data_objects(pe.Objective, active=True)))})
                if net != 'TSO':
                    ref_gen = [v.index()[0] for v in identify_variables(m.pg_adn[s_m, s_o, next(iter(m.periods))].expr)
                               if v.parent_component().local_name == 'pg']
                    ref_gen = ref_gen[0]
                    pi = _dso_price_series(m, B, ref_gen)
                    price_by_block[(str(y), d, net)] = pi
                    if (y, d) == (sorted(years)[0], DAYS[0]):
                        topology[f'DSO{net}'] = {'reference_gen': ref_gen, 'reference_bus_index': gbus.get(ref_gen),
                                                 'shared_ess_in_node_balance_of_bus_index': _storage_buses(m),
                                                 'pg_adn_expr': str(m.pg_adn[s_m, s_o, next(iter(m.periods))].expr),
                                                 'curtailable_gen_bus_index': {g: gbus.get(g) for g in gens}}
                elif (y, d) == (sorted(years)[0], DAYS[0]):
                    topology['TSO'] = {'shared_ess_in_node_balance_of_bus_index': _storage_buses(m),
                                       'curtailable_gen_bus_index': {g: gbus.get(g) for g in gens}}
                hourly = {k: [0.0] * N_T for k in ('V_net', 'V_plus', 'V_tol', 'A', 'Sg_curt')}
                transformer_idx = {i[0] for i in m.r} if hasattr(m, 'r') else set()
                row_cache = {}
                for g in gens:
                    for ti, t in enumerate(m.periods):
                        v = m.pg[g, s_m, s_o, t]
                        av = pe.value(m.pg_avail[g, s_o, t])
                        c = (av - pe.value(v)) * B
                        sg_av = pe.value(m.sg_avail[g, s_o, t])
                        sg = math.sqrt(max(pe.value(m.sg_sqr[g, s_m, s_o, t]), 0.0))
                        hourly['V_net'][ti] += c
                        hourly['V_plus'][ti] += max(c, 0.0)
                        hourly['A'][ti] += av * B
                        hourly['Sg_curt'][ti] += max(sg_av - sg, 0.0) * B
                        if c > tol:
                            hourly['V_tol'][ti] += c
                            cap = m.sg_capability[g, s_m, s_o, t] if (g, s_m, s_o, t) in m.sg_capability else None
                            cap_slack = (pe.value(cap.upper) - pe.value(cap.body)) if cap is not None else None
                            cap_dual = m.dual.get(cap) if cap is not None else None
                            zu = m.ipopt_zU_out.get(v, 0.0)
                            bus = gbus.get(g)
                            lmp_b = m.dual[m.node_balance_p[bus, s_m, s_o, t]] / B
                            cls = ('capability_bound' if (cap is not None and cap_slack <= CAP_SLACK_TOL)
                                   else 'interior')
                            e = {'instance': 'srp1_x0', 'network': 'TSO' if net == 'TSO' else f'DSO{net}',
                                 'year': y, 'day': d, 'scenario': f'{s_m}_{s_o}', 'hour': ti, 'gen': g, 'bus_index': bus,
                                 'c_mw': c, 'pg_avail_mw': av * B, 'pg_mw': pe.value(v) * B,
                                 'qg_mvar': pe.value(m.qg[g, s_m, s_o, t]) * B, 'sg_avail_mva': sg_av * B,
                                 'sg_curt_mva': max(sg_av - sg, 0.0) * B, 'class': cls,
                                 'sg_capability_slack_pu2': cap_slack, 'sg_capability_dual_raw': cap_dual,
                                 'pg_zU_raw': zu, 'lmp_bus_eur_mwh': lmp_b}
                            if t not in row_cache:
                                row_cache[t] = _network_hour_rows(m, s_m, s_o, t, transformer_idx)
                            e['active_rows_network_hour'] = row_cache[t]
                            e['transformer_limit_active'] = any(r['is_transformer'] for r in row_cache[t]['branch'])
                            e['voltage_bound_active'] = bool(row_cache[t]['voltage'])
                            entries.append(e)
                blocks[bkey] = {'network': 'TSO' if net == 'TSO' else f'DSO{net}', 'year': y, 'day': d,
                                'baseMVA': B, 'tol_mw': tol, 'curtailable_gens': gens,
                                'scenarios': {f'{s_m}_{s_o}': {'omega': 1.0, 'hourly': hourly}}}
            gc.collect()
    del payload, nets
    gc.collect()
    # prices: identical across the three DSOs, cross-checked against W25's committed x0 price series
    pr, price_checks = {}, {}
    w25_by = {(str(r['year']), r['day']): r['price_series'] for r in w25['per_day']}
    for (y, d, n), pi in price_by_block.items():
        pr.setdefault((y, d), []).append(pi)
    for (y, d), lst in pr.items():
        spread = max(max(abs(a - b) for a, b in zip(lst[0], other)) for other in lst)
        dev = max(abs(a - b) for a, b in zip(lst[0], w25_by[(y, d)]))
        price_checks[f'{y}_{d}'] = {'max_abs_diff_across_dsos': spread, 'max_abs_diff_vs_w25_price_series': dev}
    prices = {(y, d): lst[0] for (y, d), lst in pr.items()}
    # reconciliation with the certified component levels, and block weights
    recon = {}
    for bkey, b in blocks.items():
        e_net = sum(b['scenarios']['0_0']['hourly']['V_net'])
        cl = cl_blocks[bkey]
        ref = cl['unweighted']['res_curtailment_definitional_at_weight_1']
        b['admm_block_weight'] = cl['admm_block_weight']
        b['certified_res_curtailment_penalty_unweighted'] = cl['unweighted']['res_curtailment_penalty']
        recon[bkey] = {'models_E_net_mwh': e_net, 'component_levels_mwh': ref,
                       'abs_diff': abs(e_net - ref), 'rel_diff': abs(e_net - ref) / max(abs(ref), 1.0)}
        b['scenarios']['0_0']['price'] = prices[(str(b['year']), b['day'])]
    return blocks, entries, readback, topology, recon, price_checks


# ======================================================================================================================
#  pilot: production's terminal operational workbook (per generator x hour x scenario; no duals)
# ======================================================================================================================
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def read_workbook(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    need = ('Generation', 'MarketData', 'Voltage', 'Branch Loading')
    missing = [s for s in need if s not in wb.sheetnames]
    if missing:
        raise RuntimeError(f'workbook sheets missing: {missing}')
    gen = {}
    conv = defaultdict(dict)
    rows = wb['Generation'].iter_rows(values_only=True)
    hdr = next(rows)
    if list(hdr[:10]) != ['Operator', 'ADN Node ID', 'Generator ID', 'Node ID', 'Type', 'Year', 'Day', 'Quantity',
                          'Market Scenario', 'Operation Scenario'] or list(hdr[10:34]) != list(range(N_T)):
        raise RuntimeError(f'Generation header unexpected: {hdr}')
    for r in rows:
        op, adn, gid, node, typ, y, d, q, sm, so = r[:10]
        if not (_num(sm) and _num(so)):
            continue                      # the 'Expected' rows
        net = 'TSO' if op == 'TSO' else f'DSO{int(adn)}'
        key = (net, int(y), d, int(sm), int(so))
        vals = [float(v) for v in r[10:34]]
        if typ == 'Conventional' and q == 'Pg, [MW]':
            conv[key][int(gid)] = vals
            continue
        gen.setdefault(key, {}).setdefault(int(gid), {'type': typ, 'node': int(node)})[q] = vals
    # curtaillable generators = those with a Pg_net row (network.py writes it only for them)
    curt = {k: {g: v for g, v in gs.items() if 'Pg_net, [MW]' in v} for k, gs in gen.items()}
    prices = {}
    rows = wb['MarketData'].iter_rows(values_only=True)
    hdr = next(rows)
    for r in rows:
        y, d, q, sm = r[:4]
        if q == 'Energy' and _num(sm):
            prices[(int(y), d, int(sm))] = [float(v) for v in r[4:28]]
    volt = {}
    rows = wb['Voltage'].iter_rows(values_only=True)
    hdr = next(rows)
    for r in rows:
        op, adn, node, y, d, q, sm, so = r[:8]
        if op != 'DSO' or q != 'Vmag, [p.u.]' or not (_num(sm) and _num(so)):
            continue
        volt.setdefault((f'DSO{int(adn)}', int(y), d, int(sm), int(so)), {})[int(node)] = [float(v) for v in r[8:32]]
    trafo = {}
    rows = wb['Branch Loading'].iter_rows(values_only=True)
    hdr = next(rows)
    for r in rows:
        op, adn, bid, fb, tb, y, d, q, sm, so = r[:10]
        if op != 'DSO' or not (_num(sm) and _num(so)) or not str(q).startswith('Flow_ij'):
            continue
        if int(fb) == 1 and int(tb) == 2:
            trafo[(f'DSO{int(adn)}', int(y), d, int(sm), int(so))] = [float(v) for v in r[10:34]]
    wb.close()
    return curt, conv, prices, volt, trafo


def analyse_pilot(name, info, spec):
    ev = info['eval_dir']
    ms = _load_json(os.path.join(ev, 'multiscenario_terminal.json'))
    cl = _load_json(os.path.join(ev, 'component_levels_terminal.json'))['blocks']
    curt, conv, prices, volt, trafo = read_workbook(os.path.join(REPO, ev, spec['workbook_rel']))
    alpha = ms['summary']['alpha_in_force']
    # market probabilities from a TSO block (omega_{m,o} = pm * po)
    any_tso = next(v for k, v in ms['blocks'].items() if k.startswith('TSO|'))
    om = defaultdict(float)
    for key, sc in any_tso['per_scenario_costs'].items():
        om[int(key.split('_')[0])] += sc['probability']
    blocks, entries, recon = {}, [], {}
    case_vmax = {}
    for bkey, mb in ms['blocks'].items():
        parts = bkey.split('|')
        net = 'TSO' if parts[0] == 'TSO' else f'DSO{parts[1]}'
        y, d = int(parts[-2]), parts[-1]
        B = _base_mva('TSO' if net == 'TSO' else int(net[3:]), 2025)
        tol = TOL_FACTOR * B
        if net != 'TSO' and net not in case_vmax:
            nodes = _load_json(CASE_FMT[int(net[3:])].format(y=2025))['nodes']
            case_vmax[net] = {int(n['bus_i']): float(n['Vmax']) for n in nodes}
        pibar = [sum(om[m_] * prices[(y, d, m_)][t] for m_ in om) for t in range(N_T)]
        pibar_dev = None
        if net != 'TSO':
            prem = mb['dispersion']['row18_premium_by_period']
            pibar_dev = max(abs(pibar[t] - float(prem[str(t)])) for t in range(N_T))
        scen = {}
        e_net_w = 0.0
        for skey, sc in mb['per_scenario_costs'].items():
            s_m, s_o = (int(a) for a in skey.split('_'))
            gens = curt[(net, y, d, s_m, s_o)]
            hourly = {k: [0.0] * N_T for k in ('V_net', 'V_plus', 'V_tol', 'A', 'Sg_curt')}
            pi = prices[(y, d, s_m)]
            node = parts[1] if net != 'TSO' else None
            if node is not None:
                p_s = mb['interface_profiles']['per_scenario'][skey]['p']
                pbar = mb['interface_profiles']['committed']['p_mw']
            for g, rows in sorted(gens.items()):
                for t in range(N_T):
                    av, pg = rows['Pg, [MW]'][t], rows['Pg_net, [MW]'][t]
                    c = av - pg
                    hourly['V_net'][t] += c
                    hourly['V_plus'][t] += max(c, 0.0)
                    hourly['A'][t] += av
                    hourly['Sg_curt'][t] += rows['Sg_curt, [MVA]'][t]
                    if c > tol:
                        hourly['V_tol'][t] += c
                        e = {'instance': name, 'network': net, 'year': y, 'day': d, 'scenario': skey, 'hour': t,
                             'gen': g, 'bus': rows['node'], 'type': rows['type'], 'c_mw': c, 'pg_avail_mw': av,
                             'pg_mw': pg, 'qg_mvar': rows['Qg_net, [MVAr]'][t], 'sg_curt_mva': rows['Sg_curt, [MVA]'][t],
                             'pi_eur_mwh': pi[t], 'pibar_eur_mwh': pibar[t], 'alpha_pibar': alpha * pibar[t],
                             'class_primal': ('below_apparent_capability' if rows['Sg_curt, [MVA]'][t] >= 0.5 * c
                                              else 'P_for_Q_at_capability')}
                        if node is not None:
                            dd = p_s[t] - pbar[t]
                            vv = volt[(net, y, d, s_m, s_o)]
                            lv = {n_: v[t] for n_, v in vv.items() if n_ != 1}
                            nmax = max(lv, key=lv.get)
                            e.update({'p_int_mw': p_s[t], 'pbar_mw': pbar[t], 'd_mw': dd,
                                      'row18_condition': bool(dd <= D_TOL_MW and pi[t] < alpha * pibar[t]),
                                      'v_hv_bus1_pu': vv[1][t], 'v_lv_max_pu': lv[nmax], 'v_lv_max_node': nmax,
                                      'v_lv_at_vmax': bool(lv[nmax] >= case_vmax[net][nmax] - V_AT_BOUND_TOL),
                                      'v_gen_bus_pu': vv.get(rows['node'], [None] * N_T)[t],
                                      'transformer_1_2_loading_workbook': trafo[(net, y, d, s_m, s_o)][t],
                                      'transformer_1_2_at_rating': bool(trafo[(net, y, d, s_m, s_o)][t]
                                                                        >= 1.0 - TRAFO_AT_RATING_TOL)})
                        else:
                            cg = conv[(net, y, d, s_m, s_o)]
                            e['all_conventional_at_pmin'] = all(v[t] <= CONV_PMIN_TOL_MW for v in cg.values())
                        entries.append(e)
            hourly['price'] = pi
            scen[skey] = {'omega': sc['probability'], 'hourly': hourly,
                          'multiscenario_generation_renewable_curtailed_s': sc['generation_renewable_curtailed']['s'],
                          'multiscenario_curtailment_penalty': sc['generation_renewable_curtailment_penalty']}
            e_net_w += sc['probability'] * sum(hourly['V_net'])
        ref = cl[bkey]['unweighted']['res_curtailment_definitional_at_weight_1']
        recon[bkey] = {'workbook_E_net_mwh': e_net_w, 'component_levels_mwh': ref, 'abs_diff': abs(e_net_w - ref),
                       'rel_diff': abs(e_net_w - ref) / max(abs(ref), 1.0),
                       'pibar_vs_production_row18_premium_max_abs_diff': pibar_dev,
                       'sg_curt_vs_multiscenario_max_abs_diff': max(
                           abs(sum(s['hourly']['Sg_curt']) - s['multiscenario_generation_renewable_curtailed_s'])
                           for s in scen.values())}
        blocks[bkey] = {'network': net, 'year': y, 'day': d, 'baseMVA': B, 'tol_mw': tol,
                        'admm_block_weight': cl[bkey]['admm_block_weight'],
                        'certified_res_curtailment_penalty_unweighted': cl[bkey]['unweighted']['res_curtailment_penalty'],
                        'scenarios': scen, 'pibar': pibar}
    min_price = min(min(v) for v in prices.values())
    return blocks, entries, recon, {'alpha_in_force': alpha, 'market_probabilities': dict(om), 'min_price': min_price}


# ======================================================================================================================
#  aggregation, resolution and the recorded decision rule
# ======================================================================================================================
def aggregate(blocks):
    per_net = defaultdict(lambda: defaultdict(float))
    per_net_scen = defaultdict(lambda: defaultdict(float))
    rows = []
    for bkey, b in blocks.items():
        w = b['admm_block_weight']
        for skey, s in b['scenarios'].items():
            h, om, pi = s['hourly'], s['omega'], s['price'] if 'price' in s else s['hourly']['price']
            day = {k: sum(h[k]) for k in ('V_net', 'V_plus', 'V_tol', 'A', 'Sg_curt')}
            priced = sum(p * v for p, v in zip(pi, h['V_plus']))
            priced_net = sum(p * v for p, v in zip(pi, h['V_net']))
            priced_tol = sum(p * v for p, v in zip(pi, h['V_tol']))
            n_hours = sum(1 for v in h['V_tol'] if v > 0.0)
            rows.append({'block': bkey, 'network': b['network'], 'year': b['year'], 'day': b['day'], 'scenario': skey,
                         'omega': om, 'w': w, 'day_mwh': day, 'hours_above_tol': n_hours,
                         'priced_day_eur': priced, 'hourly_V_net': h['V_net'], 'hourly_V_tol': h['V_tol']})
            for k, v in day.items():
                per_net[b['network']][f'H_{k}_mwh'] += w * om * v
                per_net_scen[(b['network'], skey)][f'H_{k}_mwh'] += w * om * v
            per_net[b['network']]['C_eur'] += w * om * priced
            per_net_scen[(b['network'], skey)]['C_eur'] += w * om * priced
            per_net[b['network']]['C_net_eur'] += w * om * priced_net
            per_net[b['network']]['C_tol_eur'] += w * om * priced_tol
            per_net_scen[(b['network'], skey)]['C_net_eur'] += w * om * priced_net
            per_net_scen[(b['network'], skey)]['C_tol_eur'] += w * om * priced_tol
            per_net[b['network']]['hours_above_tol'] += n_hours
    for v in per_net.values():
        v['share_of_available_net_pct'] = 100.0 * v['H_V_net_mwh'] / v['H_A_mwh'] if v['H_A_mwh'] else None
    total = {k: sum(v[k] for v in per_net.values()) for k in ('H_V_net_mwh', 'H_V_plus_mwh', 'H_V_tol_mwh',
                                                               'H_A_mwh', 'C_eur', 'C_net_eur', 'C_tol_eur')}
    return {'per_network': {k: dict(v) for k, v in per_net.items()},
            'per_network_scenario': {f'{k[0]}|{k[1]}': dict(v) for k, v in per_net_scen.items()},
            'total': total, 'rows': rows}


def _block_key(e):
    return (f'TSO|{e["year"]}|{e["day"]}' if e['network'] == 'TSO'
            else f'DSO|{e["network"][3:]}|{e["year"]}|{e["day"]}')


def priced_by_class(entries, blocks, keyfn):
    """Horizon split of the above-tol curtailment: H = sum w_b omega_s c, C = sum w_b omega_s pi[s,t] c over the
    entries (c > TOL_MW), grouped by keyfn(entry). Same weighting as `aggregate`."""
    out = defaultdict(lambda: {'H_above_tol_mwh': 0.0, 'C_above_tol_eur': 0.0, 'generator_hours': 0})
    for e in entries:
        b = blocks[_block_key(e)]
        sc = b['scenarios'][e['scenario']]
        pi = sc['price'] if 'price' in sc else sc['hourly']['price']
        k = keyfn(e)
        out[k]['H_above_tol_mwh'] += b['admm_block_weight'] * sc['omega'] * e['c_mw']
        out[k]['C_above_tol_eur'] += b['admm_block_weight'] * sc['omega'] * pi[e['hour']] * e['c_mw']
        out[k]['generator_hours'] += 1
    return {str(k): dict(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def srp1_unit_blocks(cl_blocks, x0_blocks):
    rows, per_net = [], defaultdict(lambda: defaultdict(float))
    for bkey, cl in cl_blocks.items():
        net = 'TSO' if bkey.startswith('TSO|') else f'DSO{bkey.split("|")[1]}'
        e_net = cl['unweighted']['res_curtailment_definitional_at_weight_1']
        w = cl['admm_block_weight']
        xb = x0_blocks[bkey]
        pi = xb['scenarios']['0_0']['price']
        x0_net = sum(xb['scenarios']['0_0']['hourly']['V_net'])
        lo, hi = (min(pi) * e_net, max(pi) * e_net) if e_net >= 0 else (max(pi) * e_net, min(pi) * e_net)
        rows.append({'block': bkey, 'network': net, 'w': w, 'E_net_mwh_day': e_net, 'x0_E_net_mwh_day': x0_net,
                     'delta_unit_minus_x0_mwh_day': e_net - x0_net,
                     'res_curtailment_penalty_unweighted': cl['unweighted']['res_curtailment_penalty'],
                     'priced_day_eur_bounds': [lo, hi]})
        per_net[net]['H_V_net_mwh'] += w * e_net
        per_net[net]['C_eur_upper_bound'] += w * max(hi, 0.0)
        per_net[net]['H_delta_vs_x0_mwh'] += w * (e_net - x0_net)
        per_net[net]['dC_vs_x0_eur_abs_upper_bound'] += w * max(pi) * abs(e_net - x0_net)
    total = {k: sum(v[k] for v in per_net.values()) for k in ('H_V_net_mwh', 'C_eur_upper_bound',
                                                               'H_delta_vs_x0_mwh', 'dC_vs_x0_eur_abs_upper_bound')}
    return {'rows': rows, 'per_network': {k: dict(v) for k, v in per_net.items()}, 'total': total}


def decide(inst, agg, unit_blk):
    """Apply the recorded rule mechanically. Reachable = TSO-side (see module docstring)."""
    tests = []
    for name in ('srp1_x0', 'pilot_x0', 'pilot_unit'):
        bar = inst[name]['bar']
        a = agg[name]
        c_all = a['total']['C_eur']
        c_reach = a['per_network'].get('TSO', {}).get('C_eur', 0.0)
        per_s = {k: v['C_eur'] for k, v in a['per_network_scenario'].items() if k.startswith('TSO|')}
        tests.append({'instance': name, 'resolution': 'per hour', 'bar_eur': bar, 'C_all_networks_eur': c_all,
                      'C_net_all_networks_eur': a['total']['C_net_eur'], 'C_tol_all_networks_eur': a['total']['C_tol_eur'],
                      'C_reachable_TSO_eur': c_reach, 'C_reachable_per_scenario_eur': per_s,
                      'C_all_over_bar': c_all / bar, 'C_reachable_over_bar': c_reach / bar,
                      'material_all': c_all > bar, 'material_reachable': c_reach > bar or any(v > bar for v in per_s.values()),
                      'C_per_network_eur': {k: v['C_eur'] for k, v in a['per_network'].items()}})
    bar = inst['srp1_unit']['bar']
    c_all = unit_blk['total']['C_eur_upper_bound']
    c_reach = unit_blk['per_network'].get('TSO', {}).get('C_eur_upper_bound', 0.0)
    tests.append({'instance': 'srp1_unit', 'resolution': 'block (upper bound: day net volume x the day max price)',
                  'bar_eur': bar, 'C_all_networks_eur_upper_bound': c_all, 'C_reachable_TSO_eur_upper_bound': c_reach,
                  'C_all_over_bar': c_all / bar, 'C_reachable_over_bar': c_reach / bar,
                  'material_all': c_all > bar, 'material_reachable': c_reach > bar,
                  'C_per_network_eur_upper_bound': {k: v['C_eur_upper_bound'] for k, v in unit_blk['per_network'].items()}})
    value = {}
    for pair, (x0, un) in PAIRS.items():
        res = inst[x0]['bar'] + inst[un]['bar']
        c0 = agg[x0]['total']['C_eur']
        if pair == 'srp1':
            dc_abs = unit_blk['total']['dC_vs_x0_eur_abs_upper_bound']
            cu = unit_blk['total']['C_eur_upper_bound']
            value[pair] = {'value_resolution_eur': res, 'C_x0_eur': c0, 'C_unit_eur_upper_bound': cu,
                           'first_order_dC_abs_upper_bound_eur': dc_abs, 'dC_over_resolution': dc_abs / res,
                           'rigorous_bound_max_C_eur': max(c0, cu), 'rigorous_bound_over_resolution': max(c0, cu) / res}
        else:
            cu = agg[un]['total']['C_eur']
            value[pair] = {'value_resolution_eur': res, 'C_x0_eur': c0, 'C_unit_eur': cu,
                           'first_order_dC_eur': c0 - cu, 'dC_over_resolution': abs(c0 - cu) / res,
                           'dC_reachable_TSO_eur': (agg[x0]['per_network'].get('TSO', {}).get('C_eur', 0.0)
                                                    - agg[un]['per_network'].get('TSO', {}).get('C_eur', 0.0)),
                           'rigorous_bound_max_C_eur': max(c0, cu), 'rigorous_bound_over_resolution': max(c0, cu) / res}
    material_reachable = any(t['material_reachable'] for t in tests)
    below_everywhere = not any(t['material_all'] for t in tests)
    if material_reachable:
        branch = 'REBASELINE (material in a scenario the storage bus can reach)'
    elif below_everywhere:
        branch = 'TIE_BREAKER_STAYS (below resolution everywhere)'
    else:
        branch = ('NEITHER BRANCH AS WORDED: reachable (TSO-side) curtailment is below resolution everywhere, but '
                  'DN-side (unreachable) curtailment exceeds the Q bar in at least one instance -- Planner ruling')
    return {'tests': tests, 'value_level': value, 'material_reachable_any': material_reachable,
            'below_resolution_everywhere_all_networks': below_everywhere, 'branch_selected_mechanically': branch}


# ======================================================================================================================
#  Markdown
# ======================================================================================================================
def _f(x, nd=2):
    if x is None:
        return '-'
    if isinstance(x, str):
        return x
    return f'{x:,.{nd}f}'


def write_markdown(path, res):
    L = []
    a = L.append
    a(f'# {res["stage"]}\n')
    a('Zero solves (armed SolveProfileGuard(permitted=()), counts '
      f'{res["solve_profile_guard"]["counts"]}, verify(0) failures {res["solve_profile_guard"]["verify_0_failures"]}); '
      f'model-construction blockers {res["model_construction"]["blockers"]}, blocked calls '
      f'{res["model_construction"]["blocked_calls"]}. all_checks_pass = {res["all_checks_pass"]}; failing: '
      f'{res["failing_checks"]}.\n')
    a('**Conventions.** Volumes in MWh (1 h periods). `c = (pg_avail - pg) * baseMVA`, the production definitional '
      'curtailment (active power). Per-day values are per representative day and scenario; horizon values are '
      'weighted by `admm_block_weight` and omega_s (the Q weighting). Priced amounts `C` are EUR at the scenario\'s '
      'hourly market price on the positive part of c, in the same units as the certified Q = `gross_operational_cost` '
      '(settlement-excluded); they are what pricing curtailment at the market price would add at the CURRENT '
      'dispatch (an upper bound on the change in Q). `C on net c` is the penalty term itself evaluated at the '
      'current point (the term is linear in c, negatives included); `C above tol only` restricts to generator-hours '
      'with c > TOL_MW. TOL_MW = EQUALITY_TOLERANCE x baseMVA = '
      f'{res["method"]["TOL_MW"]} MW per generator-hour.\n')
    a('## 1. Instances and the resolution recoverable from committed records\n')
    a('| instance | campaign / eval key | candidate | cycles | Q (gross, settlement-excl.) | bar | rule-ten ratio | '
      'source | resolution recoverable | duals |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    for k, v in res['instances'].items():
        a(f'| {k} ({v["label"]}) | {v["campaign_id"]} / {v["eval_key"][:16]} | {v["candidate_key"][:8]} | '
          f'{v["certification_cycle"]} | {_f(v["Q_gross_settlement_excluded"])} | {_f(v["bar"])} | '
          f'{_f(v["rule_ten_terminal_step_over_threshold"], 4)} | {res["resolution"][k]["source"]} | '
          f'{res["resolution"][k]["recoverable"]} | {res["resolution"][k]["duals"]} |')
    a('')
    for k, v in res['resolution'].items():
        a(f'- **{k}**: {v["statement"]} Missing: {v["missing"]} Cost to obtain: {v["cost_to_obtain"]}')
    a('\n## 2. Curtailment penalty in force, per path\n')
    a('Read-back from the certified artifacts: ' + res['penalty']['readback_summary'] + '\n')
    a('| path | weight on c | settlement weight | where (file:line at the run commits / HEAD) |')
    a('|---|---|---|---|')
    for row in res['penalty']['table']:
        a(f'| {row["path"]} | {row["weight"]} | {row["settlement"]} | {row["where"]} |')
    a('')
    a('Per-commit line numbers (pattern hits with the enclosing function):\n')
    a('| pattern role | ' + ' | '.join(res['penalty']['provenance'].keys()) + ' |')
    a('|---|' + '---|' * len(res['penalty']['provenance']))
    labels = list(res['penalty']['provenance'].keys())
    for i, (fname, pattern, role) in enumerate(LINE_PATTERNS):
        cells = []
        for lab in labels:
            hits = res['penalty']['provenance'][lab]['patterns'][i]['hits']
            cells.append(', '.join(f'{fname}:{h["line"]} ({h["def"] or "module level"})' for h in hits) or 'absent')
        a(f'| {role} | ' + ' | '.join(cells) + ' |')
    a('')
    a('## 3. Volumes per network (horizon, Q weighting)\n')
    for name in ('srp1_x0', 'pilot_x0', 'pilot_unit'):
        ag = res['aggregates'][name]
        a(f'**{name}** ({res["instances"][name]["label"]}); bar {_f(res["instances"][name]["bar"])}.\n')
        a('| network | net c (MWh) | positive part (MWh) | above tol (MWh) | available RES (MWh) | net share % | '
          'network-hours above tol | C at hourly price (EUR) | C on net c | C above tol only |')
        a('|---|---|---|---|---|---|---|---|---|---|')
        for net, v in sorted(ag['per_network'].items()):
            a(f'| {net} | {_f(v["H_V_net_mwh"])} | {_f(v["H_V_plus_mwh"])} | {_f(v["H_V_tol_mwh"])} | '
              f'{_f(v["H_A_mwh"], 0)} | {_f(v["share_of_available_net_pct"], 5)} | {int(v["hours_above_tol"])} | '
              f'{_f(v["C_eur"])} | {_f(v["C_net_eur"])} | {_f(v["C_tol_eur"])} |')
        t = ag['total']
        a(f'| **all** | {_f(t["H_V_net_mwh"])} | {_f(t["H_V_plus_mwh"])} | {_f(t["H_V_tol_mwh"])} | '
          f'{_f(t["H_A_mwh"], 0)} | | | **{_f(t["C_eur"])}** | {_f(t["C_net_eur"])} | {_f(t["C_tol_eur"])} |\n')
        a('Per network and scenario (omega-weighted contribution to Q): ' + '; '.join(
            f'{k} net {_f(v["H_V_net_mwh"])} MWh, C {_f(v["C_eur"])}' for k, v in sorted(ag['per_network_scenario'].items()))
          + '\n')
    u = res['srp1_unit_block_level']
    a(f'**srp1_unit** (block resolution only); bar {_f(res["instances"]["srp1_unit"]["bar"])}.\n')
    a('| network | net c (MWh) | unit - x0 (MWh) | C upper bound (EUR) | abs dC vs x0 upper bound (EUR) |')
    a('|---|---|---|---|---|')
    for net, v in sorted(u['per_network'].items()):
        a(f'| {net} | {_f(v["H_V_net_mwh"])} | {_f(v["H_delta_vs_x0_mwh"], 4)} | {_f(v["C_eur_upper_bound"])} | '
          f'{_f(v["dC_vs_x0_eur_abs_upper_bound"])} |')
    a('')
    a('## 4. Per network, year, day and scenario (MWh per representative day; rows with a positive part >= 0.01)\n')
    for name in ('srp1_x0', 'pilot_x0', 'pilot_unit'):
        a(f'**{name}**\n')
        a('| network | year | day | scen | omega | net | pos. part | above tol | Sg_curt (MVAh) | hours > tol | '
          'priced (EUR/day) |')
        a('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in res['aggregates'][name]['rows']:
            if r['day_mwh']['V_plus'] >= 0.01:
                a(f'| {r["network"]} | {r["year"]} | {r["day"]} | {r["scenario"]} | {r["omega"]} | '
                  f'{_f(r["day_mwh"]["V_net"], 4)} | {_f(r["day_mwh"]["V_plus"], 4)} | {_f(r["day_mwh"]["V_tol"], 4)} | '
                  f'{_f(r["day_mwh"]["Sg_curt"], 4)} | {r["hours_above_tol"]} | {_f(r["priced_day_eur"])} |')
        a('')
    a('**srp1_unit** (block resolution: day net only)\n')
    a('| block | net (MWh/day) | x0 net (MWh/day) | unit - x0 | penalty term in Q |')
    a('|---|---|---|---|---|')
    for r in u['rows']:
        if abs(r['E_net_mwh_day']) >= 0.01:
            a(f'| {r["block"]} | {_f(r["E_net_mwh_day"], 4)} | {_f(r["x0_E_net_mwh_day"], 4)} | '
              f'{_f(r["delta_unit_minus_x0_mwh_day"], 5)} | {r["res_curtailment_penalty_unweighted"]} |')
    a('\n## 5. Hourly profiles (MW, sum over the network\'s curtaillable generators, above tol) -- '
      'the network-day-scenarios with at least 0.1 MWh above tol\n')
    for name in ('srp1_x0', 'pilot_x0', 'pilot_unit'):
        sel = [r for r in res['aggregates'][name]['rows'] if r['day_mwh']['V_tol'] >= 0.1]
        a(f'**{name}**: {len(sel)} rows\n')
        if sel:
            a('| network | year | day | scen | ' + ' | '.join(f'h{t}' for t in range(N_T)) + ' |')
            a('|---|---|---|---|' + '---|' * N_T)
            for r in sel:
                a(f'| {r["network"]} | {r["year"]} | {r["day"]} | {r["scenario"]} | '
                  + ' | '.join(f'{v:.3f}' if v else '0' for v in r['hourly_V_tol']) + ' |')
        a('')
    a('## 6. Causes\n')
    a('Topology read from the certified srp1_x0 models (bus indices are 0-based model indices; DN index 0 = case33 '
      'bus 1, the 345 kV busbar; the DN transformer is branch 1-2): ' + json.dumps(res['topology']) + '\n')
    for k, v in res['causes'].items():
        a(f'**{k}**: {v["statement"]}\n')
        if v.get('table'):
            a('| ' + ' | '.join(v['table'][0].keys()) + ' |')
            a('|' + '---|' * len(v['table'][0]))
            for row in v['table']:
                a('| ' + ' | '.join(_f(x, 4) if isinstance(x, float) else str(x) for x in row.values()) + ' |')
            a('')
        for split in ('horizon_split_by_class', 'horizon_split_by_class_row18_trafo',
                      'horizon_split_by_class_transformer_voltage'):
            if v.get(split):
                a(f'Horizon split ({split}; above-tol entries; Q weighting; C at the hourly price):\n')
                a('| group | generator-hours | H above tol (MWh) | C above tol (EUR) |')
                a('|---|---|---|---|')
                for g, x in v[split].items():
                    a(f'| {g} | {x["generator_hours"]} | {_f(x["H_above_tol_mwh"])} | {_f(x["C_above_tol_eur"])} |')
                a('')
        if v.get('real_curtailment_hours'):
            a('Hours with curtailment below the apparent capability (network, year, day, scenario, hour): '
              + '; '.join(str(tuple(h)) for h in v['real_curtailment_hours']) + '\n')
    a('## 7. Resolution and the recorded decision rule\n')
    a(res['decision']['resolution_choice'] + '\n')
    a('| instance | resolution | bar | C all networks | C reachable (TSO) | C/bar | C reachable/bar | material (all) | '
      'material (reachable) |')
    a('|---|---|---|---|---|---|---|---|---|')
    for t in res['decision']['tests']:
        call = t.get('C_all_networks_eur', t.get('C_all_networks_eur_upper_bound'))
        cre = t.get('C_reachable_TSO_eur', t.get('C_reachable_TSO_eur_upper_bound'))
        a(f'| {t["instance"]} | {t["resolution"]} | {_f(t["bar_eur"])} | {_f(call)} | {_f(cre)} | '
          f'{_f(t["C_all_over_bar"], 4)} | {_f(t["C_reachable_over_bar"], 6)} | {t["material_all"]} | '
          f'{t["material_reachable"]} |')
    a('')
    a('| pair | value resolution | C(0) | C(unit) | first-order dC | dC/resolution | rigorous bound max C | bound/resolution |')
    a('|---|---|---|---|---|---|---|---|')
    for p, v in res['decision']['value_level'].items():
        cu = v.get('C_unit_eur', v.get('C_unit_eur_upper_bound'))
        dc = v.get('first_order_dC_eur', v.get('first_order_dC_abs_upper_bound_eur'))
        a(f'| {p} | {_f(v["value_resolution_eur"])} | {_f(v["C_x0_eur"])} | {_f(cu)} | {_f(dc)} | '
          f'{_f(v["dC_over_resolution"], 5)} | {_f(v["rigorous_bound_max_C_eur"])} | '
          f'{_f(v["rigorous_bound_over_resolution"], 4)} |')
    a(f'\n**Branch selected (mechanical application of the recorded rule):** '
      f'{res["decision"]["branch_selected_mechanically"]}\n')
    a('## 8. What the re-baseline branch would require (not implemented)\n')
    for item in res['rebaseline_cost']:
        a(f'- {item}')
    a('\n## 9. Checks\n')
    a('| check | value |')
    a('|---|---|')
    for k, v in res['checks'].items():
        a(f'| {k} | {v} |')
    with open(path, 'x') as handle:
        handle.write('\n'.join(L) + '\n')


# ======================================================================================================================
#  main
# ======================================================================================================================
def main(test_out_dir=None):
    started = time.time()
    out_dir = os.path.abspath(test_out_dir) if test_out_dir else os.path.join(REPO, OUT_REL)
    if test_out_dir and os.path.commonpath([out_dir, REPO]) == REPO:
        raise SystemExit('--test-out-dir must be outside the repository')
    if not os.path.isdir(out_dir):
        raise SystemExit(f'output directory must exist (created by the launch command): {out_dir}')
    for fname in ('curtailment_audit.json', 'curtailment_audit.md', 'manifest_sha256.json'):
        if os.path.exists(os.path.join(out_dir, fname)):
            raise SystemExit(f'refusing to overwrite {fname}')
    blockers = ConstructionBlockers().install()
    # ---- inputs, verified before anything else
    inst, failures = {}, []
    for name, spec in INSTANCES.items():
        info, f = resolve_instance(name, spec)
        inst[name] = info
        failures += f
    for rel in (W25_JSON, W43_JSON, BITWISE_GATE_JSON):
        tc = _tracked_clean(rel)
        if not (tc['tracked'] and tc['clean']):
            failures.append(f'supporting input not committed/clean: {rel}')
    # ---- capture-path checklist (rule eleven), asserted before the analysis
    checklist = {}
    for name, info in inst.items():
        cl = _load_json(os.path.join(info['eval_dir'], 'component_levels_terminal.json'))['blocks']
        checklist[f'{name}_component_levels_blocks_nonempty'] = len(cl) > 0
        checklist[f'{name}_component_levels_curtailment_fields'] = all(
            'res_curtailment_definitional_at_weight_1' in b['unweighted'] and 'res_curtailment_penalty' in b['unweighted']
            and 'admm_block_weight' in b for b in cl.values())
        checklist[f'{name}_bar_recorded'] = isinstance(info['bar'], float)
        if INSTANCES[name]['source'] == 'workbook':
            ms = _load_json(os.path.join(info['eval_dir'], 'multiscenario_terminal.json'))
            checklist[f'{name}_multiscenario_alpha'] = 'alpha_in_force' in ms['summary']
            checklist[f'{name}_multiscenario_dso_profiles'] = all(
                'interface_profiles' in b and 'p_mw' in b['interface_profiles'].get('committed', {})
                and all('p' in v for v in b['interface_profiles'].get('per_scenario', {}).values())
                and len(b['interface_profiles'].get('per_scenario', {})) == len(b['per_scenario_costs'])
                for k, b in ms['blocks'].items() if k.startswith('DSO|'))
            checklist[f'{name}_multiscenario_row18_premium_by_period'] = all(
                len((b.get('dispersion') or {}).get('row18_premium_by_period') or {}) == N_T
                for k, b in ms['blocks'].items() if k.startswith('DSO|'))
            checklist[f'{name}_multiscenario_scenario_probabilities'] = all(
                all('probability' in s and 'generation_renewable_curtailed' in s for s in b['per_scenario_costs'].values())
                for b in ms['blocks'].values())
            checklist[f'{name}_multiscenario_blocks_match_component_levels'] = set(ms['blocks']) == set(cl)
            wb = openpyxl.load_workbook(os.path.join(REPO, info['inputs_verified']['workbook']['path']), read_only=True)
            checklist[f'{name}_workbook_sheets'] = all(s in wb.sheetnames for s in
                                                       ('Generation', 'MarketData', 'Voltage', 'Branch Loading'))
            wb.close()
        if INSTANCES[name]['source'] == 'models':
            checklist[f'{name}_models_file_present'] = info['inputs_verified']['models']['present']
    w25 = _load_json(W25_JSON)
    w25 = w25['T2'] if 'T2' in w25 else w25
    checklist['w25_price_series_12_days'] = len([r for r in w25['per_day'] if len(r.get('price_series') or []) == N_T]) == 12
    missing = sorted(k for k, v in checklist.items() if not v)
    _log(f'[W53] inputs: {len(failures)} failures {failures}; capture-path checklist {len(checklist)} items, '
         f'{len(missing)} False: {missing}')
    if failures or missing:
        raise SystemExit('INPUT / CAPTURE-PATH CHECK FAILED (fail fast, before analysis)')
    # ---- penalty in force
    commits = {'s48_x0_capture run': inst['srp1_x0']['git_head_at_run'],
               's47_recert run': inst['srp1_unit']['git_head_at_run'],
               's52 pilot run': inst['pilot_x0']['git_head_at_run'], 'HEAD': 'HEAD'}
    prov = penalty_provenance(commits)
    # ---- srp1_x0 (models)
    _log('[W53] srp1_x0: loading the persisted terminal models')
    cl_x0 = _load_json(os.path.join(inst['srp1_x0']['eval_dir'], 'component_levels_terminal.json'))['blocks']
    x0_blocks, x0_entries, x0_readback, topology, x0_recon, x0_price_checks = analyse_srp1_models(
        os.path.join(REPO, inst['srp1_x0']['inputs_verified']['models']['path']), cl_x0, w25)
    gc.collect()
    _log(f'[W53] srp1_x0: {len(x0_blocks)} blocks, {len(x0_entries)} generator-hours above tol')
    # ---- srp1_unit (block level) and its duplicate
    cl_unit = _load_json(os.path.join(inst['srp1_unit']['eval_dir'], 'component_levels_terminal.json'))['blocks']
    cl_dup = _load_json(inst['srp1_unit']['duplicate']['component_levels'])['blocks']
    dup_equal = all(cl_unit[k]['unweighted']['res_curtailment_definitional_at_weight_1']
                    == cl_dup[k]['unweighted']['res_curtailment_definitional_at_weight_1'] for k in cl_unit)
    unit_blk = srp1_unit_blocks(cl_unit, x0_blocks)
    # ---- pilot (workbooks)
    pilot = {}
    for name in ('pilot_x0', 'pilot_unit'):
        _log(f'[W53] {name}: reading the terminal workbook')
        pilot[name] = analyse_pilot(name, inst[name], INSTANCES[name])
        gc.collect()
    # ---- aggregates and decision
    agg = {'srp1_x0': aggregate(x0_blocks)}
    for name in ('pilot_x0', 'pilot_unit'):
        agg[name] = aggregate(pilot[name][0])
    decision = decide(inst, agg, unit_blk)
    decision['resolution_choice'] = (
        'Resolution used: the bar (record.bar, the stopping-slack error of Q). Justification: pricing curtailment at '
        'the hourly market price changes Q by at most C(x) at a global optimum (the current point stays feasible; '
        'the price is non-negative), so C(x) <= bar(x) means the re-baseline cannot move Q by more than its own '
        'stopping error; at the value level, |d value| <= max(C(0), C(unit)) rigorously and ~ C(0) - C(unit) to first '
        'order, both compared with bar(0) + bar(unit), the committed value resolution. sigma_Q is not used (it is a '
        'dispersion calibration, not an error of Q). A volume threshold is not used as the criterion (volumes are '
        'reported for the manuscript) because the decision concerns Q and the storage value. Reading: C <= bar '
        'establishes "below resolution"; C > bar only means the bound does not exclude a Q change above the bar '
        '(the re-optimised change can be far smaller than C), i.e. "not shown to be below resolution".')
    # ---- causes
    causes = {}
    by_cls = defaultdict(lambda: {'n': 0, 'mwh_day_sum': 0.0, 'dual_min': None, 'dual_max': None})
    for e in x0_entries:
        k = (e['network'], e['class'], e['transformer_limit_active'], e['voltage_bound_active'])
        by_cls[k]['n'] += 1
        by_cls[k]['mwh_day_sum'] += e['c_mw']
        dl = e['sg_capability_dual_raw']
        if dl is not None:
            by_cls[k]['dual_min'] = dl if by_cls[k]['dual_min'] is None else min(by_cls[k]['dual_min'], dl)
            by_cls[k]['dual_max'] = dl if by_cls[k]['dual_max'] is None else max(by_cls[k]['dual_max'], dl)
    interior = [e for e in x0_entries if e['class'] == 'interior']
    cap = [e for e in x0_entries if e['class'] == 'capability_bound']
    causes['srp1_x0'] = {
        'statement': (f'{len(x0_entries)} generator-hours above tol: {len(cap)} capability_bound, {len(interior)} '
                      'interior. capability_bound = the apparent-power row sg_capability (pg^2 + qg^2 <= sg_avail^2) '
                      'is active with a non-zero dual: pg is below pg_avail because the inverter supplies Q at its S '
                      'limit (Sg_curt ~ 0) -- a binding constraint, not surplus. Every entry carries the active '
                      'voltage / branch rows of its network-hour with their duals (JSON); the table splits by whether '
                      'the DN transformer limit (a branch with a ratio Var) and a voltage bound are active then.'),
        'table': [{'network': k[0], 'class': k[1], 'transformer_limit_active': k[2], 'voltage_bound_active': k[3],
                   'generator_hours': v['n'], 'sum_c_mw_over_entries_unweighted': v['mwh_day_sum'],
                   'sg_capability_dual_raw_min': v['dual_min'], 'sg_capability_dual_raw_max': v['dual_max']}
                  for k, v in sorted(by_cls.items(), key=lambda kv: str(kv[0]))],
        'share_of_above_tol_energy_with_transformer_limit_active': (
            sum(e['c_mw'] for e in x0_entries if e['transformer_limit_active'])
            / max(sum(e['c_mw'] for e in x0_entries), 1e-300)),
        'share_of_above_tol_energy_with_a_voltage_bound_active': (
            sum(e['c_mw'] for e in x0_entries if e['voltage_bound_active'])
            / max(sum(e['c_mw'] for e in x0_entries), 1e-300)),
        'transformer_row_duals_raw_at_curtailed_hours': sorted({round(r['dual_raw'], 6) for e in x0_entries
                                                               for r in e['active_rows_network_hour']['branch']
                                                               if r['is_transformer']})[:5] + ['...'],
        'capability_bound_examples_top10': sorted(cap, key=lambda e: -e['c_mw'])[:10],
        'interior_entries': interior,
        'horizon_split_by_class_transformer_voltage': priced_by_class(
            x0_entries, x0_blocks, lambda e: (e['network'], e['class'], 'transformer_limit_active=%s'
                                              % e['transformer_limit_active'], 'voltage_bound_active=%s'
                                              % e['voltage_bound_active'])),
        'horizon_split_by_class': priced_by_class(x0_entries, x0_blocks, lambda e: e['class']),
        'capability_bound_share_of_above_tol_energy': (sum(e['c_mw'] for e in cap)
                                                        / max(sum(e['c_mw'] for e in x0_entries), 1e-300)),
        'sg_curt_over_c_max_capability_bound': max((e['sg_curt_mva'] / e['c_mw'] for e in cap), default=None),
        'lmp_bus_min_capability_bound_eur_mwh': min((e['lmp_bus_eur_mwh'] for e in cap), default=None)}
    for name in ('pilot_x0', 'pilot_unit'):
        ents = pilot[name][1]
        tot = sum(e['c_mw'] for e in ents) or 1e-300
        dso = [e for e in ents if e['network'] != 'TSO']
        tso = [e for e in ents if e['network'] == 'TSO']
        tab = defaultdict(lambda: defaultdict(float))
        for e in ents:
            k = (e['network'], e['class_primal'], e.get('row18_condition', 'n/a'), e.get('v_lv_at_vmax', 'n/a'),
                 e.get('transformer_1_2_at_rating', 'n/a'))
            tab[k]['generator_hours'] += 1
            tab[k]['sum_c_mw'] += e['c_mw']
        causes[name] = {
            'statement': (f'{len(ents)} generator-hours above tol ({len(dso)} DN-side, {len(tso)} TSO-side). NO DUALS '
                          '(models not persisted): the columns are primal indicators only. Shares below are of the '
                          'above-tol energy summed over entries (unweighted). Share '
                          f'below the apparent capability: '
                          f'{sum(e["c_mw"] for e in ents if e["class_primal"] == "below_apparent_capability") / tot:.4f}; '
                          'share in DN entries meeting the row-18 condition (d <= +tol and pi_s < alpha*pibar): '
                          f'{sum(e["c_mw"] for e in dso if e["row18_condition"]) / tot:.4f}; share with an LV '
                          f'voltage at Vmax: {sum(e["c_mw"] for e in dso if e["v_lv_at_vmax"]) / tot:.4f}; max '
                          f'transformer 1-2 loading (workbook Flow_ij, 1.0 = at rating) at a curtailed hour: '
                          f'{max((e["transformer_1_2_loading_workbook"] for e in dso), default=None)}; share with the '
                          f'transformer at rating: {sum(e["c_mw"] for e in dso if e["transformer_1_2_at_rating"]) / tot:.4f}; '
                          f'share below the apparent capability AND meeting the row-18 condition: '
                          f'{sum(e["c_mw"] for e in dso if e["row18_condition"] and e["class_primal"] == "below_apparent_capability") / tot:.4f}; '
                          f'TSO entries with all '
                          f'conventional units at Pmin: {sum(1 for e in tso if e["all_conventional_at_pmin"])} of '
                          f'{len(tso)}.'),
            'table': [{'network': k[0], 'class_primal': k[1], 'row18_condition': k[2], 'lv_at_vmax': k[3],
                       'transformer_at_rating': k[4], 'generator_hours': int(v['generator_hours']),
                       'sum_c_mw_over_entries_unweighted': v['sum_c_mw']}
                      for k, v in sorted(tab.items(), key=lambda kv: str(kv[0]))],
            'top_entries': sorted(ents, key=lambda e: -e['c_mw'])[:25],
            'horizon_split_by_class': priced_by_class(ents, pilot[name][0], lambda e: (e['network'] == 'TSO' and 'TSO')
                                                      or ('DN', e['class_primal'])),
            'horizon_split_by_class_row18_trafo': priced_by_class(
                ents, pilot[name][0], lambda e: (e['network'], e['class_primal'],
                                                 'row18_condition=%s' % e.get('row18_condition', 'n/a'),
                                                 'transformer_at_rating=%s' % e.get('transformer_1_2_at_rating', 'n/a'))),
            'real_curtailment_hours': sorted({(e['network'], e['year'], e['day'], e['scenario'], e['hour'])
                                              for e in dso if e['class_primal'] == 'below_apparent_capability'})}
    # ---- resolution statements
    wall = {k: v['wall_time_s'] for k, v in inst.items()}
    gate = _load_json(BITWISE_GATE_JSON)
    resolution = {
        'srp1_x0': {'source': 'persisted terminal models', 'recoverable': 'generator x hour (1 scenario)',
                    'duals': 'yes', 'statement': 'Full per-generator, per-hour resolution with every multiplier.',
                    'missing': 'nothing for this audit.', 'cost_to_obtain': '-'},
        'srp1_unit': {'source': 'component_levels_terminal.json', 'recoverable': 'block (network x year x day), day sum',
                      'duals': 'no',
                      'statement': ('No persisted models and no terminal workbook exist for this eval (s47 ran '
                                    'without post-certification persistence); the only committed curtailment '
                                    'quantity is the per-block net day sum `res_curtailment_definitional_at_weight_1`.'),
                      'missing': 'the per-hour profile, the positive part (net only), the capability/interior split.',
                      'cost_to_obtain': ('one re-run of s47_recert point n7_4h_e1 (eval key bd504ecf) with '
                                         'persist_certified_models, as s48_x0_capture did for x = 0: '
                                         f'~{wall["srp1_unit"] / 3600:.2f} h wall (recorded run {wall["srp1_unit"]:.0f} s, '
                                         '112 cycles), ~2.6 GB RSS, ~164 MB pickle, plus a bitwise reproduction gate '
                                         'against s47_recert.')},
        'pilot_x0': {'source': 'terminal operational workbook (production writer, pre post-certification)',
                     'recoverable': 'generator x hour x scenario', 'duals': 'no',
                     'statement': ('Contrary to the pre-task caveat, per-hour and per-scenario curtailment IS '
                                   'recoverable: the hash-recorded workbook holds pg_avail and pg per curtaillable '
                                   'generator, hour and scenario; it reconciles with the certified component levels '
                                   '(check below).'),
                     'missing': 'every multiplier (no models), so the cause is shown by primal indicators only.',
                     'cost_to_obtain': ('one re-run of the pilot pair with persist_certified_models '
                                        f'(recorded walls {wall["pilot_x0"] / 3600:.2f} h and {wall["pilot_unit"] / 3600:.2f} h '
                                        'at concurrency 2, ~4 h for the pair) and the +5 GiB certified-model pickle '
                                        'transient per child that the no-persistence ruling avoided; the frozen '
                                        'campaign_s52_pilot (with persistence) is the named run.')},
    }
    resolution['pilot_unit'] = dict(resolution['pilot_x0'])
    # ---- penalty table
    ro = {'srp1_x0_models_penalty_values': sorted({r['penalty_gen_curtailment'] for r in x0_readback}),
          'srp1_x0_models_settlement_weights': sorted({r['interface_settlement_weight'] for r in x0_readback})}
    for name in INSTANCES:
        cl = _load_json(os.path.join(inst[name]['eval_dir'], 'component_levels_terminal.json'))['blocks']
        ro[f'{name}_component_levels_res_curtailment_penalty_values'] = sorted(
            {b['unweighted']['res_curtailment_penalty'] for b in cl.values()})
        ro[f'{name}_component_levels_definitional_nonzero_blocks'] = sum(
            1 for b in cl.values() if b['unweighted']['res_curtailment_definitional_at_weight_1'] != 0.0)
    for name in ('pilot_x0', 'pilot_unit'):
        ro[f'{name}_multiscenario_per_scenario_penalty_values'] = sorted(
            {s['multiscenario_curtailment_penalty'] for b in pilot[name][0].values() for s in b['scenarios'].values()})
    w43 = _load_json(W43_JSON)
    head = prov['HEAD']['patterns']

    def _where(i):
        return ', '.join(f'{p["file"]}:{h["line"]}' for p in [head[i]] for h in p['hits'])
    penalty = {
        'provenance': prov,
        'readback': ro,
        'readback_summary': (f'certified models (srp1_x0, 48 blocks): penalty_gen_curtailment = '
                             f'{ro["srp1_x0_models_penalty_values"]}, interface_settlement_weight = '
                             f'{ro["srp1_x0_models_settlement_weights"]}; component levels res_curtailment_penalty = '
                             + '; '.join(f'{n} {ro[f"{n}_component_levels_res_curtailment_penalty_values"]}'
                                         for n in INSTANCES)
                             + '; pilot per-scenario penalty = '
                             + '; '.join(f'{n} {ro[f"{n}_multiscenario_per_scenario_penalty_values"]}'
                                         for n in ('pilot_x0', 'pilot_unit'))
                             + f'. PENALTY_GENERATION_CURTAILMENT imported now = {PENALTY_GENERATION_CURTAILMENT}.'),
        'table': [
            {'path': 'ADMM TSO subproblem (every cycle; the certified Q)', 'weight': '0 EUR/MWh',
             'settlement': '1', 'where': f'{_where(12)} (set) after the build at {_where(2)}; call {_where(11)}'},
            {'path': 'ADMM DSO subproblem (every cycle; the certified Q)', 'weight': '0 EUR/MWh',
             'settlement': '1', 'where': f'{_where(13)} (set) after the build at {_where(2)}; call {_where(10)}'},
            {'path': 'initialisation solve (TSO and DSO, before _prepare_*_for_admm)',
             'weight': f'{PENALTY_GENERATION_CURTAILMENT} EUR/MWh (the constant)', 'settlement': '0',
             'where': f'{_where(0)}; {_where(2)}; term {_where(3)}; solves {_where(7)}, {_where(8)}, {_where(9)}'},
            {'path': 'standalone / uncoordinated benchmark (_run_operational_planning_without_coordination)',
             'weight': f'{PENALTY_GENERATION_CURTAILMENT} EUR/MWh (never reset there; see the reference list)',
             'settlement': '0', 'where': f'{_where(0)}; {_where(2)}; solves {_where(15)}, {_where(16)} (enclosing functions in the per-commit table)'},
        ],
        'w43_initialisation_evidence': {'path': W43_JSON, 'commit': 'bf28ee55', 'not_a_result': w43.get('not_a_result'),
                                        'kappa_formula': w43['formulas'].get('kappa_eff')},
    }
    # ---- re-baseline cost (the other branch; not implemented)
    rebaseline = [
        ('Data parameter: a curtailment-price mode read from the case file (default = today: the constant '
         f'{PENALTY_GENERATION_CURTAILMENT} at build, 0 in the ADMM subproblems), and a mode pricing c at the '
         'scenario\'s hourly market price network.cost_energy_p[s_m][p] in BOTH networks, i.e. the weight inside '
         f'the term at {_where(3)} becomes per (s_m, p), and the ADMM resets at {_where(12)} / {_where(13)} keep it '
         'instead of zeroing it.'),
        (f'SRP1 bitwise gate at the old value: the two-cycle SRP1 gate pattern (p515_s52_srp1_bitwise_gate.py): '
         f'{gate["solve_profile"]["base_declared"]} solves declared, recorded wall {gate["wall_clock_s"]:.0f} s.'),
        (f'Re-baseline on SRP1 before anything else runs: x = 0 (recorded {wall["srp1_x0"]:.0f} s, 132 cycles) and '
         f'the smallest unit (recorded {wall["srp1_unit"]:.0f} s, 112 cycles): ~{(wall["srp1_x0"] + wall["srp1_unit"]) / 3600:.1f} h '
         'serial, ~0.9 h at concurrency 2; plus the pilot pair if the pilot is to be restated '
         f'(recorded {wall["pilot_x0"] / 3600:.2f} h + {wall["pilot_unit"] / 3600:.2f} h at concurrency 2).'),
        'The 3 x 3 selection is frozen only after the decision (Addendum 41).']
    # ---- checks
    checks = {
        'inputs_verified_against_committed_manifests': not failures,
        'capture_path_checklist_all_true': not missing,
        'srp1_x0_reconciles_with_component_levels': all(r['rel_diff'] <= RECON_REL_TOL for r in x0_recon.values()),
        'srp1_x0_prices_identical_across_dsos': all(v['max_abs_diff_across_dsos'] <= 1e-9 for v in x0_price_checks.values()),
        'srp1_x0_prices_match_w25': all(v['max_abs_diff_vs_w25_price_series'] <= 1e-9 for v in x0_price_checks.values()),
        'srp1_unit_duplicate_record_equal': dup_equal,
        'pilot_x0_reconciles_with_component_levels': all(r['rel_diff'] <= RECON_REL_TOL for r in pilot['pilot_x0'][2].values()),
        'pilot_unit_reconciles_with_component_levels': all(r['rel_diff'] <= RECON_REL_TOL for r in pilot['pilot_unit'][2].values()),
        'pilot_x0_sg_curt_matches_multiscenario': all(r['sg_curt_vs_multiscenario_max_abs_diff'] <= 1e-6
                                                      for r in pilot['pilot_x0'][2].values()),
        'pilot_unit_sg_curt_matches_multiscenario': all(r['sg_curt_vs_multiscenario_max_abs_diff'] <= 1e-6
                                                        for r in pilot['pilot_unit'][2].values()),
        'pilot_prices_non_negative': min(pilot[n][3]['min_price'] for n in pilot) >= 0.0,
        'pilot_pibar_matches_production_row18_premium': all(
            r['pibar_vs_production_row18_premium_max_abs_diff'] <= 1e-9 for n in pilot for r in pilot[n][2].values()
            if r['pibar_vs_production_row18_premium_max_abs_diff'] is not None),
        'srp1_prices_non_negative': min(min(b['scenarios']['0_0']['price']) for b in x0_blocks.values()) >= 0.0,
        'penalty_zero_in_every_certified_block': all(
            v == [0.0] for k, v in ro.items() if k.endswith('_penalty_values') and 'settlement' not in k),
        'no_model_construction_call': len(blockers.blocked_calls) == 0,
    }
    guard_failures = GUARD.verify(0)
    checks['solve_profile_guard_verified_0'] = not guard_failures
    failing = sorted(k for k, v in checks.items() if not v)
    blockers.uninstall()
    res = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'git_head': _git(['rev-parse', 'HEAD']).strip(), 'script': os.path.basename(__file__),
           'script_sha256': _sha(os.path.abspath(__file__)),
           'script_git_status': _git(['status', '--porcelain', '--', os.path.basename(__file__)]).strip() or 'clean',
           'interpreter': sys.executable,
           'mode': 'TEST (not evidence)' if test_out_dir else 'real',
           'objective_convention': ('volumes in MWh; priced amounts in EUR at the scenario hourly market price, '
                                    'block-weighted by admm_block_weight and omega_s: the units of the certified '
                                    'Q = gross_operational_cost (settlement-excluded)'),
           'method': {'formulas': __doc__, 'TOL_MW': TOL_FACTOR * 100.0, 'CAP_SLACK_TOL': CAP_SLACK_TOL,
                      'DUAL_TOL': DUAL_TOL, 'ROW_SLACK_TOL': ROW_SLACK_TOL, 'D_TOL_MW': D_TOL_MW,
                      'V_AT_BOUND_TOL': V_AT_BOUND_TOL, 'CONV_PMIN_TOL_MW': CONV_PMIN_TOL_MW,
                      'RECON_REL_TOL': RECON_REL_TOL, 'EQUALITY_TOLERANCE': EQUALITY_TOLERANCE},
           'instances': inst, 'resolution': resolution, 'capture_path_checklist_asserted_before_analysis': checklist,
           'penalty': penalty, 'topology': topology,
           'aggregates': agg, 'srp1_unit_block_level': unit_blk,
           'srp1_x0': {'blocks': x0_blocks, 'entries_above_tol': x0_entries, 'readback': x0_readback,
                       'reconciliation': x0_recon, 'price_checks': x0_price_checks},
           'pilot': {n: {'blocks': pilot[n][0], 'entries_above_tol': pilot[n][1], 'reconciliation': pilot[n][2],
                         'meta': pilot[n][3]} for n in pilot},
           'causes': causes, 'decision': decision, 'rebaseline_cost': rebaseline,
           'checks': checks, 'all_checks_pass': not failing, 'failing_checks': failing,
           'model_construction': {'blockers': [f'{c.__name__}.{a}' for c, a in blockers.targets],
                                  'blocked_calls': blockers.blocked_calls},
           'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
           'wall_clock_s': time.time() - started}
    res = _jsonable(res)
    jpath = os.path.join(out_dir, 'curtailment_audit.json')
    with open(jpath, 'x') as handle:
        json.dump(res, handle, indent=1)
    write_markdown(os.path.join(out_dir, 'curtailment_audit.md'), res)
    GUARD.uninstall()
    for name, ag in agg.items():
        _log(f'[W53] {name}: total {({k: round(v, 3) for k, v in ag["total"].items()})}')
    _log(f'[W53] srp1_unit: {({k: round(v, 3) for k, v in unit_blk["total"].items()})}')
    _log(f'[W53] decision: {decision["branch_selected_mechanically"]}')
    _log(f'[W53] checks: {checks}')
    _log(f'[W53] guard {res["solve_profile_guard"]["counts"]} verify0_failures={guard_failures}; wrote {jpath}')
    _log(f'[W53] {"ALL CHECKS PASS" if not failing else "FAILING CHECKS: " + str(failing)}')
    if failing:
        sys.exit(1)


def write_manifest(out_dir):
    man = {}
    for fname in sorted(os.listdir(out_dir)):
        if fname == 'manifest_sha256.json':
            continue
        man[os.path.relpath(os.path.join(out_dir, fname), REPO)] = _sha(os.path.join(out_dir, fname))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        json.dump(man, handle, indent=2)
    return man


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--test-out-dir', default=None, help='development only: write outside the repository')
    args = parser.parse_args()
    if args.manifest:
        print(json.dumps(write_manifest(os.path.join(REPO, OUT_REL)), indent=2))
        GUARD.uninstall()
    else:
        main(args.test_out_dir)
