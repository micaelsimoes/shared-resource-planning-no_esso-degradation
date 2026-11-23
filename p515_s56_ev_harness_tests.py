"""P5.15 S56 critical path step 1 -- engineering tests of the EV-plan harness keys (TASKS.md "Working mode and EV
priority", author + Expert 2026-11-23: a script with pass/fail checks; no frozen prediction spec).

The keys (`p515_s44_campaign_harness`, section "S56 EV-PLAN KEYS"; each default absent): s56_cohort_formulation,
s56_terminal_value_in_objective, s56_esso_slack_penalty, s56_ev_profiles, s56_staged_investment.

MODES (each run alone, attached, both streams captured; every mode except --smoke is ZERO-SOLVE: this process's
SolveProfileGuard(permitted=()) is installed before any repository import and verified at 0, with the guards of the
modules it imports):
  --build-ev-five-block   The five-block EV twin (no five-block EV instance in the injection-hook schema exists; Audit
                          A's five-block EV variant `input_audit_A/instance/SRP1__s56_3x3_ev_profiles.json` is the
                          predecessor schema, without the A5 rule and the post-read expectations). Built from the W89
                          five-block 3 x 3 SP case with Audit E's builder functions (`p515_s56_build_instances`), as a
                          derived instance, with Audit E's gates (route, read-back bitwise, assertion passes on EV and
                          raises on un-injected / SP / SRP1 reads, object walk) plus a cross-check against Audit A's
                          means.
  --nokey --pre-rev R     OPTION-OFF IDENTITY: (c1) module diff vs R, deleted lines exactly the declared ones;
                          (c2) entries without the keys select nothing and add no record field; (c3) freeze_campaign_
                          spec pre vs post identical on entries without the keys (SRP1 and a derived instance);
                          (c4) every committed campaign spec: candidate and eval keys recomputed pre and post, equal to
                          each other and to the recorded ones; (c5) build_evaluation_record pre vs post on committed
                          evaluation artifacts, identical.
  --nl-stub --harness pre|post --pre-rev R   NL identity (r4/r5 precedent): an option-off evaluation (SRP1, reference
                          unit, flex x2) through `_construct_arm_planning` + the harness configuration hook of the PRE or
                          POST harness, two stubbed ADMM cycles; every TSO / DSO / ESSO NL file of cycles 1-2 hashed.
  --nl-compare            pre vs post NL hashes and per-cycle diagnostics, identical.
  --keys                  key uniqueness / determinism (in process and in a second process), staged canonicalisation
                          (permutation invariance, single-year keys unchanged, C5 refusals), freeze-time refusals.
  --stub-run --grid srp1grid|five_block   ALL KEYS ON with the EV twin, a staged mixed-duration multi-cohort plan, 2x
                          flexibility price and the V-1 coupling: freeze, child declaration and capture path, EV install
                          with injection, configuration hook (order, read-backs), EV assertion raising after the
                          multiplier and on un-injected / SP / SRP1 reads, multiplier read-back per family, TV (kappa,
                          'mid' refused, converter excluded by perturbation), cohort row inventory on every node-year,
                          network-side gross-flow switch, two stubbed ADMM cycles, the cost breakdown on the run's
                          models (every field captured).
  --smoke                 THE ONE SOLVED RUN: a 2-cycle EV evaluation of x0 and of the reference unit at 2x on the
                          SRP1-grid EV twin, all keys on, through the campaign harness (freeze, lock, evaluate);
                          declared solves per evaluation ((1 + 3) x 12 + 3) x (2 + 1) = 153, checked exactly.
  --smoke-recheck         ZERO SOLVES: the smoke's checks on its recorded artifacts with the corrected solve-count
                          comparison (the --smoke run compared a dict with an integer; its FAIL lines stand in
                          smoke.json), plus the table of measured numbers.
  --manifest              sha256 of every file under the output directory.

Usage (repo root, canonical interpreter, attached, both streams captured; one mode per invocation):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --build-ev-five-block
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --nokey --pre-rev aaaa71e2
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --nl-stub --harness pre --pre-rev aaaa71e2
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --nl-stub --harness post --pre-rev aaaa71e2
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --nl-compare
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --keys
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --stub-run --grid srp1grid
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --stub-run --grid five_block
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_ev_harness_tests.py --smoke
  <python> -u p515_s56_ev_harness_tests.py --manifest
Output: data/SRP1/Results/P515S56/ev_harness_v1/<mode>/ (each mode refuses an existing output).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 EV harness tests (this process never solves)').install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (parent side: no model imports at module import)

SCRIPT = os.path.basename(os.path.abspath(__file__))
SCHEMA = 'p515_s56_ev_harness_tests_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'ev_harness_v1')
OUT = os.path.join(REPO, OUT_REL)
SOLVER_PATH = '/usr/local/bin/ipopt'
DEFAULT_PRE_REV = 'aaaa71e2'

# the module lines this change deleted or rewrote (declared before the --nokey run; every other change an insertion)
DECLARED_DELETED_LINES = (
    "        canon = canonical_candidate(cand, investment_year=options.get('investment_year',",
    "                                                                      INVESTMENT_YEAR))",
    "                         interface_coupling=None):",
    "    and nothing is recorded.\"\"\"",
    "def install_derived_instance(derived, eval_dir):",
    "    Zero solves. Returns the evidence; raises on any mismatch.\"\"\"",
    "    if derived is not None:",
    "                                                interface_coupling=interface_coupling),",
    "            's56_calibration_error': s56_calibration_error}",
)

# instances
SRP1GRID_SP_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'instance_record.json')
SRP1GRID_EV_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'ev_instance_record.json')
W89_CASE = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
W89_RECORD = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json')
AUDIT_A_FIVE_BLOCK_EV_PROFILES = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'input_audit_A', 'instance',
                                              'SRP1__s56_3x3_ev_profiles.json')
FIVE_DIR_REL = os.path.join(OUT_REL, 'five_block_ev')
FIVE_EV_LABEL = 's56_five_block_3x3_ev'
FIVE_EV_CASE = 'SRP1__s56_five_block_3x3_ev.json'
FIVE_EV_PROFILES = 'SRP1__s56_five_block_3x3_ev_profiles.json'
FIVE_EV_RECORD = 'ev_instance_record.json'

COUPLING_V1_NOQ = {'per_scenario_interface_coupling': True, 'per_scenario_voltage_mode': 'per_scenario',
                   'row18_charge_reactive': False}   # = p515_s56_coupling_gates.COUPLING_V1_NOQ (checked in --stub-run)
FLEX_M = 2.0
SLACK_PENALTY = 1e4
COHORT_MODE = 'independent_units_all_node_years'
NODES = (5, 7, 9)
# staged mixed-duration multi-cohort plans (C5-admissible; node 7 holds two cohorts from its second year)
STAGED = {
    'srp1grid': {'5': {'2030': [0.5, 1.0]}, '7': {'2025': [0.25, 1.0], '2030': [0.25, 0.5]}, '9': {'2035': [0.25, 0.5]}},
    'five_block': {'5': {'2028': [0.5, 1.0]}, '7': {'2025': [0.25, 1.0], '2031': [0.25, 0.5]},
                   '9': {'2037': [0.25, 0.5]}},
}
REF_UNIT = {'7': {'2025': [0.25, 1.0]}}   # the reference unit, as a staged plan
X0_STAGED = {'5': {'2025': [0.0, 0.0]}}   # x = 0, as a staged plan (every other node-year zero)
SMOKE_CAP = 2
SMOKE_DECLARED_SOLVES_PER_EVAL = ((1 + 3) * 12 + 3) * (SMOKE_CAP + 1)   # 153


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _dump(path, obj):
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)


TAG = ''   # --tag: a suffix for a re-run's output directory (an earlier attempt is never overwritten)


def _new_out(name):
    path = os.path.join(OUT, name + TAG)
    if os.path.exists(path):
        raise SystemExit(f'REFUSED: {os.path.relpath(path, REPO)} exists (write-once)')
    os.makedirs(path)
    return path


def _provenance():
    return {'git_head': _git('rev-parse', 'HEAD').strip(), 'interpreter': sys.executable,
            'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'), 'script_sha256': _sha(os.path.abspath(__file__)),
            'harness_sha256': _sha(os.path.join(REPO, 'p515_s44_campaign_harness.py')),
            'dirty_tracked': [ln for ln in _git('status', '--porcelain', '--untracked-files=no').splitlines()]}


class Checks:
    def __init__(self):
        self.items = {}

    def __call__(self, name, ok, detail=None):
        ok = bool(ok)
        self.items[name] = {'ok': ok, 'detail': detail}
        _log(f'{"PASS" if ok else "FAIL"} {name}' + ('' if ok else f' -- {str(detail)[:600]}'))
        return ok

    def summary(self):
        failed = sorted(k for k, v in self.items.items() if not v['ok'])
        return {'n_checks': len(self.items), 'n_pass': len(self.items) - len(failed), 'failed': failed,
                'all_pass': bool(self.items) and not failed}


def _finish(res, ck, out_dir, name, extra_guards=()):
    res['checks'] = ck.items
    res['summary'] = ck.summary()
    guards = {'own': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0': GUARD.verify(0)}}
    for label, g in extra_guards:
        guards[label] = {'label': g.label, 'counts': dict(g.counts), 'verify_0': g.verify(0)}
    res['guards'] = guards
    res['zero_solve_verified'] = all(not g['verify_0'] for g in guards.values())
    res['finished_utc'] = _utc()
    _dump(os.path.join(out_dir, f'{name}.json'), res)
    ok = res.get('error') is None and res['summary']['all_pass'] and res['zero_solve_verified']
    _log(f"[{name}] {'PASS' if ok else 'FAIL'}: {res['summary']['n_pass']}/{res['summary']['n_checks']} checks; "
         f"failed {res['summary']['failed'][:12]}; error {res.get('error')}; zero solves {res['zero_solve_verified']}")
    return 0 if ok else 2


def _load_module_at(rev, path_in_repo, scratch, name):
    text = _git('show', f'{rev}:{path_in_repo}')
    path = os.path.join(scratch, f'{name}.py')
    with open(path, 'x') as handle:
        handle.write(text)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = H.REPO
    mod.CASE_FILE = H.CASE_FILE
    return mod, path


def srp1grid_ev_declaration():
    with open(SRP1GRID_EV_RECORD) as handle:
        rec = json.load(handle)
    derived = H.validate_derived_instance(rec['derived_instance_declaration_draft'])
    ev = {'profiles_path': rec['profiles_path'], 'profiles_sha256': rec['profiles_sha256'],
          'ev_scenario_checksum': rec['scenario_checksum']}
    return derived, ev


def srp1grid_sp_declaration():
    with open(SRP1GRID_SP_RECORD) as handle:
        rec = json.load(handle)
    for key in ('derived_instance_declaration', 'derived_instance_declaration_draft'):
        if key in rec:
            return H.validate_derived_instance(rec[key])
    raise RuntimeError(f'{SRP1GRID_SP_RECORD}: no derived declaration')


def five_block_sp_declaration():
    with open(W89_RECORD) as handle:
        return H.validate_derived_instance(json.load(handle)['derived_instance_declaration'])


def five_block_ev_declaration():
    path = os.path.join(FIVE_DIR_REL, FIVE_EV_RECORD)
    with open(path) as handle:
        rec = json.load(handle)
    if not rec.get('all_checks_pass'):
        raise RuntimeError(f'{path}: the five-block EV build did not pass its gates')
    derived = H.validate_derived_instance(rec['derived_instance_declaration'])
    ev = {'profiles_path': rec['profiles_path'], 'profiles_sha256': rec['profiles_sha256'],
          'ev_scenario_checksum': rec['scenario_checksum']}
    return derived, ev


def ev_declaration(grid):
    return srp1grid_ev_declaration() if grid == 'srp1grid' else five_block_ev_declaration()


def sp_declaration(grid):
    return srp1grid_sp_declaration() if grid == 'srp1grid' else five_block_sp_declaration()


def s56_options(ev, staged, extra=None, coupling=True):
    out = {H.S56_COHORT_FORMULATION_KEY: COHORT_MODE, H.S56_TV_IN_OBJECTIVE_KEY: True,
           H.S56_ESSO_SLACK_PENALTY_KEY: SLACK_PENALTY, H.S56_EV_PROFILES_KEY: dict(ev),
           H.S56_STAGED_INVESTMENT_KEY: deepcopy(staged), 'flex_price_multiplier': FLEX_M}
    if coupling:   # the V-1 option: inactive in production at one scenario (the EV twin); declared in the stub runs only
        out['interface_coupling'] = dict(COUPLING_V1_NOQ)
    out.update(extra or {})
    return out


# ======================================================================================================================
#  --build-ev-five-block
# ======================================================================================================================
def build_ev_five_block(_args):
    import numpy as np
    out_dir = os.path.join(REPO, FIVE_DIR_REL)
    if os.path.exists(out_dir):
        raise SystemExit(f'REFUSED: {FIVE_DIR_REL} exists (write-once)')
    os.makedirs(out_dir)
    import p515_s56_build_instances as BI   # Audit E's builder (installs its own permitted=() guard)
    import p515_s56_expected_profiles as AA
    import p515_s56_ev_injection as EVI
    BI.OUT_DIR = out_dir   # its production_read writes reads/<label>/ under OUT_DIR
    ck = Checks()
    res = {'schema': SCHEMA + '_build_ev_five_block', 'started_utc': _utc(), 'provenance': _provenance(),
           'error': None, 'builder_functions': 'p515_s56_build_instances.production_read; p515_s56_expected_profiles.'
                                               'snapshot / compute_means / joint_mean_check / walk_compare'}
    try:
        sp = five_block_sp_declaration()
        w89_abs = os.path.join(REPO, W89_CASE)
        ck('W89 case sha256 == its declaration', _sha(w89_abs) == sp['case_sha256'], sp['case_sha256'])
        p_sp, snap_sp, rec_sp = BI.production_read(w89_abs, 'sp')
        post_sp = AA.snapshot(p_sp)
        sp_checksum = rec_sp['scenario_metadata']['combined_scenario_checksum']
        ck('SP read checksum == W89 declaration', sp_checksum == sp['scenario_checksum'],
           {'read': sp_checksum, 'declared': sp['scenario_checksum']})
        ck('SP grid: years 2025/2028/2031/2034/2037 x 3, 4 days, 3 x 3 scenarios',
           sorted(p_sp.years) == [2025, 2028, 2031, 2034, 2037] and all(v == 3 for v in p_sp.years.values())
           and len(p_sp.days) == 4 and p_sp.num_market_scenarios == 3
           and p_sp.transmission_network.num_oper_scenarios == 3, {'years': dict(p_sp.years)})
        ck('EVI.array_refs == audit A _array_refs (keys, order) on the SP read',
           [k for k, *_ in EVI.array_refs(p_sp)] == [k for k, *_ in AA._array_refs(p_sp)])
        keys = list(snap_sp)
        means = AA.compute_means(snap_sp)
        joint = AA.joint_mean_check(p_sp, snap_sp, means)
        scale = max(1.0, max(float(np.max(np.abs(m))) for m in means.values()))
        ck('joint 9-pair mean == factorized mean (<= 1e-12 x largest |mean|)', joint <= 1e-12 * scale, joint)
        means_post = AA.compute_means(post_sp)
        changed_post = [AA.key_str(k) for k in keys if not np.array_equal(snap_sp[k]['A'], post_sp[k]['A'])]
        ck('SP: the only arrays production changes after the checksum point are TSO ADN pd / qd',
           bool(changed_post) and all(k.startswith('TSO|') and '|load|' in k and k.split('|')[-1] in ('pd', 'qd')
                                      for k in changed_post), changed_post[:10])
        a5_keys = [AA.key_str(k) for k in keys if EVI.is_a5(k)]
        post_expected = {AA.key_str(k): means_post[k] for k in keys
                         if not EVI.is_a5(k) and not BI._bitwise(means_post[k], means[k])}
        ck('post-read expectation differs from the mean only at TSO ADN pd / qd (production zeroing)',
           set(post_expected) == set(changed_post) and all(not np.any(v) for v in post_expected.values()))
        # the EV case: W89 with the five scenario counts 3 -> 1 (nothing else)
        with open(w89_abs) as handle:
            w89_case = json.load(handle)
        ev_case = json.loads(json.dumps(w89_case))
        changes = [{'key': 'NumMarketScenarios', 'from': ev_case['NumMarketScenarios'], 'to': 1}]
        ev_case['NumMarketScenarios'] = 1
        changes.append({'key': f'TransmissionNetwork[{ev_case["TransmissionNetwork"]["name"]}].num_operation_scenarios',
                        'from': ev_case['TransmissionNetwork']['num_operation_scenarios'], 'to': 1})
        ev_case['TransmissionNetwork']['num_operation_scenarios'] = 1
        for dn in ev_case['DistributionNetworks']:
            changes.append({'key': f'DistributionNetworks[{dn["name"]}].num_operation_scenarios',
                            'from': dn['num_operation_scenarios'], 'to': 1})
            dn['num_operation_scenarios'] = 1
        ck('EV case == W89 case except the five scenario counts', BI.strip_counts(ev_case) == BI.strip_counts(w89_case))
        ck('EV case keeps the first year 2025 and the five-block Years (no derived instance drops the first year)',
           ev_case['Years'] == w89_case['Years'] and min(int(y) for y in ev_case['Years']) == 2025, ev_case['Years'])
        ess_params = ev_case['SharedEnergyStorage']['params_file']
        ck("EV case ESS params file is not the 'mid' file", ess_params not in H.S56_MID_SOH_PARAMS_FILES, ess_params)
        ev_case_path = os.path.join(out_dir, FIVE_EV_CASE)
        with open(ev_case_path, 'x') as handle:
            handle.write(json.dumps(ev_case, indent='\t'))
        ev_case_sha = _sha(ev_case_path)
        prof_path = os.path.join(out_dir, FIVE_EV_PROFILES)
        profiles = {
            'schema': EVI.SCHEMA, 'instance_label': FIVE_EV_LABEL,
            'source_instance': {'label': sp['instance_label'], 'case_path': sp['case_path'],
                                'case_sha256': sp['case_sha256'], 'scenario_checksum': sp_checksum},
            'case_file': os.path.relpath(ev_case_path, REPO), 'formula': AA.FORMULA,
            'a5': ('reactive-flexibility arrays (flex_q_up, flex_q_down) NOT injected: production\'s constant '
                   'np.ones((1, 24)) * EQUALITY_TOLERANCE is kept (S56_DESIGN_NOTE_v1 D2; as the SRP1-grid twin)'),
            'a2_a3': 'plain probability-weighted means of the arrays as stored (audit A, "as built")',
            'n_arrays': sum(1 for k in keys if not EVI.is_a5(k)), 'n_a5_keys': len(a5_keys),
            'arrays': {AA.key_str(k): means[k].ravel().tolist() for k in keys if not EVI.is_a5(k)},
            'post_read_expected': {k: v.ravel().tolist() for k, v in post_expected.items()},
            'a5_keys': a5_keys, 'ev_scenario_checksum': None,
            'ev_scenario_checksum_note': 'computed by the production read with injection; in ev_instance_record.json',
            'builder': SCRIPT,
        }
        with open(prof_path, 'x') as handle:
            GRIO.dump(profiles, handle, indent=0)
        prof = EVI.load_profiles(prof_path)
        rt = set(prof['arrays']) == {AA.key_str(k) for k in keys if not EVI.is_a5(k)} and all(
            BI._bitwise(prof['arrays'][AA.key_str(k)], means[k]) for k in keys if not EVI.is_a5(k))
        ck('profiles file round trip bitwise (every injected array)', rt)
        # cross-check against Audit A's five-block EV means (predecessor schema, d0d3a600)
        with open(AUDIT_A_FIVE_BLOCK_EV_PROFILES) as handle:
            audit_a = json.load(handle)
        common = sorted(set(audit_a['arrays']) & set(prof['arrays']))
        same = [k for k in common if BI._bitwise(np.asarray(audit_a['arrays'][k], dtype=np.float64).reshape(1, -1),
                                                 prof['arrays'][k])]
        res['audit_A_cross_check'] = {'audit_A_file': AUDIT_A_FIVE_BLOCK_EV_PROFILES,
                                      'audit_A_sha256': _sha(AUDIT_A_FIVE_BLOCK_EV_PROFILES),
                                      'n_audit_A_arrays': len(audit_a['arrays']), 'n_new_arrays': len(prof['arrays']),
                                      'n_common': len(common), 'n_bitwise_equal': len(same),
                                      'only_in_audit_A': sorted(set(audit_a['arrays']) - set(prof['arrays']))[:20],
                                      'n_only_in_audit_A': len(set(audit_a['arrays']) - set(prof['arrays']))}
        ck('Audit A five-block EV means == the new means on every common key (bitwise)',
           len(common) == len(prof['arrays']) and len(same) == len(common), res['audit_A_cross_check'])
        # the EV read with injection
        p_ev, snap_ev, rec_ev = BI.production_read(ev_case_path, 'ev', profiles=prof)
        ev_checksum = rec_ev['scenario_metadata']['combined_scenario_checksum']
        n_inj = len(keys) - len(a5_keys)
        ck('EV read: injected every non-A5 array once, kept every A5 array',
           rec_ev['injection'] == [{'n_injected': n_inj, 'n_a5_kept': len(a5_keys)}], rec_ev['injection'])
        ck('EV read: checksum-point arrays == means (bitwise)', all(
            BI._bitwise(snap_ev[k]['A'], means[k]) for k in keys if not EVI.is_a5(k)))
        post_ev = AA.snapshot(p_ev)
        ck('EV read: post-read arrays == mean of the SP post-read arrays (bitwise)', all(
            BI._bitwise(post_ev[k]['A'], means_post[k]) for k in keys if not EVI.is_a5(k)))
        ck('EV checksum differs from the SP checksum and the SRP1 canonical',
           ev_checksum not in (sp_checksum, BI.SRP1_CANONICAL_CHECKSUM), ev_checksum)
        pt = rec_ev['assertion_at_checksum_point']
        ck('in-child assertion PASSES at the checksum point', len(pt) == 1 and pt[0].get('ok') is True)
        tests = {}
        try:
            tests['ev_post_read'] = EVI.assert_ev_arrays(p_ev, prof, point='post_read',
                                                         expected_scenario_checksum=ev_checksum)
            ok = tests['ev_post_read']['ok']
        except EVI.EvInjectionMismatch as exc:
            tests['ev_post_read'] = {'ok': False, 'error': str(exc)[:2000]}
            ok = False
        ck('in-child assertion PASSES on the EV read (post_read, EV checksum)', ok)
        walk = AA.walk_compare(p_ev, p_sp)
        ck('EV vs SP: every non-scenario item identical (object walk, declared exclusions only)',
           walk['n_diffs'] == 0 and not walk['uncompared_types'], {k: walk[k] for k in ('n_diffs', 'diffs_first')})

        def expect_raise(name, planning, **kw):
            try:
                s = EVI.assert_ev_arrays(planning, prof, point='post_read', **kw)
                tests[name] = {'raised': False, 'failures': s.get('failures')}
                return False
            except EVI.EvInjectionMismatch as exc:
                tests[name] = {'raised': True, 'error': str(exc)[:1500]}
                return True
        ck('assertion RAISES on the EV read with the SP checksum expected',
           expect_raise('ev_read_wrong_checksum', p_ev, expected_scenario_checksum=sp_checksum))
        ck('assertion RAISES on the SP read', expect_raise('sp_read', p_sp))
        del p_sp
        p_un, _s, rec_un = BI.production_read(ev_case_path, 'ev_without_injection')
        ck('EV case read without injection: checksum differs from the EV checksum',
           rec_un['scenario_metadata']['combined_scenario_checksum'] != ev_checksum,
           rec_un['scenario_metadata']['combined_scenario_checksum'])
        ck('assertion RAISES on the EV case read without injection', expect_raise('ev_without_injection', p_un))
        del p_un
        p_1, _s1, _r1 = BI.production_read(os.path.join(REPO, BI.CASE_SRP1_REL), 'srp1')
        ck('assertion RAISES on a plain SRP1.json read', expect_raise('srp1_plain', p_1))
        del p_1
        res['assertion_tests'] = tests
        declaration = H.validate_derived_instance({
            'instance_label': FIVE_EV_LABEL, 'case_path': os.path.relpath(ev_case_path, REPO),
            'case_sha256': ev_case_sha, 'scenario_checksum': ev_checksum, 'source_case_path': sp['case_path'],
            'source_case_sha256': sp['case_sha256'],
            'changes_vs_source': changes + [{'key': 'scenario arrays (all but A5)',
                                             'from': 'production synthesis at one scenario (= draw 0)',
                                             'to': (f'probability-weighted means of {sp["instance_label"]}, injected at '
                                                    f'shared_resources_planning._compute_scenario_metadata from '
                                                    f'{os.path.relpath(prof_path, REPO)} (sha256 {_sha(prof_path)})')}]})
        facts = {'years': {str(y): v for y, v in p_ev.years.items()}, 'days': dict(p_ev.days),
                 'expected_block_counts': BI.S.expected_block_counts(p_ev),
                 'declared_solves_per_cycle': BI.S.declared_solve_profile(p_ev, 1)['solves_per_cycle']}
        ck('EV twin: 20 (year, day) blocks per network, 83 solves per cycle',
           facts['declared_solves_per_cycle'] == (1 + 3) * 20 + 3, facts)
        for label, g in (('builder_module', BI.GUARD), ('audit_A_module', AA.GUARD)):
            ck(f'guard {label} verify(0) == []', g.verify(0) == [], g.verify(0))
        record = {'schema': SCHEMA + '_five_block_ev_instance', 'utc': _utc(), 'instance_label': FIVE_EV_LABEL,
                  'case_path': os.path.relpath(ev_case_path, REPO), 'case_sha256': ev_case_sha,
                  'profiles_path': os.path.relpath(prof_path, REPO), 'profiles_sha256': _sha(prof_path),
                  'scenario_checksum': ev_checksum, 'source_instance': sp, 'derived_instance_declaration': declaration,
                  'facts': facts, 'audit_A_cross_check': res['audit_A_cross_check'],
                  'loading_route': ('p515_s44_campaign_harness.install_derived_instance(derived, eval_dir, '
                                    'ev_profiles=p515_s56_ev_injection.load_profiles(profiles_path, profiles_sha256)) '
                                    '-- entry key s56_ev_profiles'),
                  'all_checks_pass': ck.summary()['all_pass'], 'failed_checks': ck.summary()['failed']}
        _dump(os.path.join(out_dir, FIVE_EV_RECORD), record)
        res['record'] = {k: record[k] for k in ('case_path', 'case_sha256', 'profiles_path', 'profiles_sha256',
                                                'scenario_checksum', 'all_checks_pass')}
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    import p515_s56_build_instances as BI2
    import p515_s56_expected_profiles as AA2
    return _finish(res, ck, out_dir, 'build_ev_five_block', extra_guards=(('builder_module', BI2.GUARD),
                                                                          ('audit_A_module', AA2.GUARD)))


# ======================================================================================================================
#  --nokey
# ======================================================================================================================
def _module_diff(pre_path, post_path):
    import difflib
    with open(pre_path) as handle:
        a = handle.read().splitlines()
    with open(post_path) as handle:
        b = handle.read().splitlines()
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            deleted.extend(a[i1:i2])
        if tag in ('replace', 'insert'):
            inserted.extend(b[j1:j2])
    return {'n_lines_pre': len(a), 'n_lines_post': len(b), 'deleted': deleted, 'n_inserted': len(inserted)}


def _spec_norm(spec):
    s = json.loads(json.dumps(spec, sort_keys=True, default=str))
    s.pop('frozen_utc', None)
    s.pop('harness', None)
    return s


def _freeze(mod, root, cid, candidates, cfg_extra=None):
    cfg = {'name': 'S56 EV harness zero-solve spec check', 'overrides': {}}
    cfg.update(cfg_extra or {})
    _p, _d, spec = mod.freeze_campaign_spec(root, cid, candidates, cfg, 3, 1, ['zero-solve check'],
                                            required_consecutive_cycles=10)
    return spec


def _eval_key_from_spec(mod, spec, entry):
    conf = spec.get('configuration') or {}
    return mod.evaluation_key(
        entry['key'], entry.get('overrides') or {}, case_file_aa=conf.get('case_file_anderson_acceleration'),
        model_variant=entry.get('model_variant'), ess_ageing_baseline=conf.get('ess_ageing_baseline'),
        flex_price_multiplier=entry.get('flex_price_multiplier'), derived_instance=conf.get('derived_instance'),
        interface_deviation_premium=entry.get('interface_deviation_premium'),
        convergence_depth_tail=conf.get('convergence_depth_tail'),
        certification_continuation=entry.get('certification_continuation'),
        settling_continuation=entry.get('settling_continuation'), settling_extension=entry.get('settling_extension'),
        settling_resettle=entry.get('settling_resettle'), rho0_scale=entry.get('rho0_scale'),
        nrf_option=entry.get('nrf_option'), interface_coupling=entry.get('interface_coupling'))


def nokey(args):
    out_dir = _new_out('nokey')
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    ck = Checks()
    res = {'schema': SCHEMA + '_nokey', 'started_utc': _utc(), 'pre_rev': args.pre_rev, 'provenance': _provenance(),
           'declared_deleted_lines': list(DECLARED_DELETED_LINES), 'error': None}
    try:
        pre, pre_path = _load_module_at(args.pre_rev, 'p515_s44_campaign_harness.py', scratch, 'harness_pre')
        post_path = os.path.join(REPO, 'p515_s44_campaign_harness.py')
        res['pre_sha256'], res['post_sha256'] = _sha(pre_path), _sha(post_path)
        # (c1)
        c1 = _module_diff(pre_path, post_path)
        res['c1_module'] = c1
        ck('c1: deleted / rewritten module lines are exactly the declared ones', c1['deleted']
           == list(DECLARED_DELETED_LINES), c1['deleted'])
        # (c2) entries without the keys
        entries = {'off': {}, 'flex2': {'flex_price_multiplier': 2.0}, 'v1': {'interface_coupling': dict(COUPLING_V1_NOQ)},
                   'pinned': {H.S56_CYCLE_CAPTURE_PINNED_KEY: True}, 'year2030': {'investment_year': 2030},
                   'rho_nrf': {'rho0_scale': 2.0, 'nrf_option': True}}
        c2 = {}
        for name, opts in entries.items():
            entry = dict(opts, label=name, canonical=H.canonical_candidate({5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)},
                                                                            investment_year=opts.get('investment_year',
                                                                                                     2025)))
            sel = {'freeze': H.s56_freeze_entry(opts, {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}, None),
                   'child': H.s56_child_declaration({}, entry), 'record_fields': H.s56_ev_plan_record_fields(entry, {}),
                   'storage_view_is_same_object': H._s56_storage_view(entry['canonical']) is entry['canonical'],
                   'investment_map_pre_eq_post': (pre.investment_map_from_canonical(entry['canonical'])
                                                  == H.investment_map_from_canonical(entry['canonical'])),
                   'investment_year_pre_eq_post': (pre.investment_year_from_canonical(entry['canonical'])
                                                   == H.investment_year_from_canonical(entry['canonical']))}
            c2[name] = {k: (v if not isinstance(v, dict) or not v else 'non-empty') for k, v in sel.items()}
            ck(f'c2:{name}: nothing selected, no record field, canonical helpers unchanged',
               sel['freeze'] is None and sel['child'] is None and sel['record_fields'] == {}
               and sel['storage_view_is_same_object'] and sel['investment_map_pre_eq_post']
               and sel['investment_year_pre_eq_post'], c2[name])
        res['c2'] = c2
        # (c3) freeze pre vs post, entries without the keys, SRP1 and a derived instance
        x0 = {5: (0, 0), 7: (0, 0), 9: (0, 0)}
        unit = {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}
        cands = [('x0', x0, {}), ('unit', unit, {}), ('unit_2030', unit, {'investment_year': 2030}),
                 ('unit_flex2', unit, {'flex_price_multiplier': 2.0}),
                 ('unit_v1', unit, {'interface_coupling': dict(COUPLING_V1_NOQ)}),
                 ('unit_pinned', unit, {'overrides': {'anderson_acceleration': {'enabled': False}},
                                        H.S56_CYCLE_CAPTURE_PINNED_KEY: True}),
                 ('unit_aa', unit, {'overrides': {'anderson_acceleration': {'enabled': True}}})]
        derived_sp = srp1grid_sp_declaration()
        c3 = {}
        for tag, cfg in (('srp1', None), ('srp1grid_sp_derived', {'derived_instance': derived_sp})):
            specs = {lab: _spec_norm(_freeze(mod, os.path.join(scratch, f'spec_{tag}_{lab}'), f's56evh_c3_{tag}',
                                             deepcopy(cands), cfg)) for lab, mod in (('pre', pre), ('post', H))}
            same = specs['pre'] == specs['post']
            c3[tag] = {'identical': same, 'eval_keys': {e['label']: e['eval_key'] for e in specs['post']['candidates']}}
            ck(f'c3:{tag}: freeze_campaign_spec pre == post on entries without the keys', same)
        res['c3'] = c3
        # (c4) every committed campaign spec
        files = sorted(f for f in _git('ls-files').splitlines() if os.path.basename(f).startswith('campaign_spec_')
                       and f.endswith('.json'))
        n_entries = n_cand_rec = n_eval_rec = 0
        cand_mismatch_post, cand_pre_post, eval_pre_post, eval_mismatch_post, eval_mismatch_pre = [], [], [], [], []
        errors = []
        for f in files:
            with open(f) as handle:
                spec = json.load(handle)
            for entry in spec.get('candidates') or []:
                n_entries += 1
                canon = entry.get('canonical')
                if canon is None:
                    continue
                kp, kq = pre.candidate_key(canon), H.candidate_key(canon)
                if kp != kq:
                    cand_pre_post.append([f, entry.get('label')])
                if 'key' in entry:
                    n_cand_rec += 1
                    if kq != entry['key']:
                        cand_mismatch_post.append([f, entry.get('label')])
                try:
                    ep, eq = _eval_key_from_spec(pre, spec, entry), _eval_key_from_spec(H, spec, entry)
                except Exception as error:  # noqa: BLE001 -- recorded (an old spec shape)
                    errors.append([f, entry.get('label'), f'{type(error).__name__}: {error}'[:300]])
                    continue
                if ep != eq:
                    eval_pre_post.append([f, entry.get('label')])
                if 'eval_key' in entry:
                    n_eval_rec += 1
                    if eq != entry['eval_key']:
                        eval_mismatch_post.append([f, entry.get('label')])
                    if ep != entry['eval_key']:
                        eval_mismatch_pre.append([f, entry.get('label')])
        res['c4'] = {'n_spec_files': len(files), 'n_entries': n_entries, 'n_candidate_keys_recorded': n_cand_rec,
                     'n_eval_keys_recorded': n_eval_rec, 'candidate_key_pre_ne_post': cand_pre_post,
                     'candidate_key_post_ne_recorded': cand_mismatch_post, 'eval_key_pre_ne_post': eval_pre_post,
                     'eval_key_post_ne_recorded': eval_mismatch_post[:50],
                     'n_eval_key_post_ne_recorded': len(eval_mismatch_post),
                     'eval_key_pre_ne_recorded_same_set': sorted(map(tuple, eval_mismatch_pre))
                     == sorted(map(tuple, eval_mismatch_post)),
                     'recompute_errors': errors[:50], 'n_recompute_errors': len(errors)}
        ck('c4: candidate keys of every committed spec entry: pre == post', not cand_pre_post and n_entries > 0)
        ck('c4: candidate keys of every committed spec entry == the recorded key', not cand_mismatch_post,
           cand_mismatch_post[:5])
        ck('c4: eval keys of every committed spec entry: pre == post', not eval_pre_post, eval_pre_post[:5])
        ck('c4: eval keys: post == recorded wherever pre == recorded (the same set of historic mismatches)',
           res['c4']['eval_key_pre_ne_recorded_same_set'], {'n_post_ne_recorded': len(eval_mismatch_post)})
        # (c5) build_evaluation_record pre vs post on committed artifacts
        recs = sorted(f for f in _git('ls-files').splitlines() if f.endswith('/evaluation_record.json'))
        c5 = []
        for f in recs:
            d = os.path.dirname(f)
            with open(f) as handle:
                record = json.load(handle)
            rp = record.get('report_path')
            sp = record.get('campaign_spec_path')
            if not rp or not sp or not os.path.isfile(rp) or not os.path.isfile(sp) or record.get('status') == 'error':
                continue
            with open(sp) as handle:
                spec = json.load(handle)
            entry = next((e for e in spec['candidates'] if e['label'] == record['candidate_label']), None)
            if entry is None:
                continue
            with open(rp) as handle:
                report = json.load(handle)
            kw = dict(spec=spec, spec_path=os.path.join(REPO, sp), spec_sha256=record['campaign_spec_sha256'],
                      entry=entry, report=report, component_levels=None, floor_terminal=None, published_caps=None,
                      peak_rss=record.get('peak_rss'), wall=record.get('wall_time_s'), eval_dir=os.path.join(REPO, d))
            a = json.dumps(pre.build_evaluation_record(**kw), sort_keys=True, default=str)
            b = json.dumps(H.build_evaluation_record(**kw), sort_keys=True, default=str)
            c5.append({'record': f, 'identical': a == b, 'sha256_pre': hashlib.sha256(a.encode()).hexdigest()})
            if len(c5) >= 12:
                break
        res['c5'] = c5
        ck('c5: build_evaluation_record pre == post on committed evaluation artifacts',
           len(c5) >= 4 and all(x['identical'] for x in c5), {'n': len(c5)})
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    return _finish(res, ck, out_dir, 'nokey')


# ======================================================================================================================
#  stubs (zero solves)
# ======================================================================================================================
def _stand_in_point(block, salt):
    """A deterministic non-degenerate point for every free Var (the step 1 probe's rule,
    p515_s56_step1_completion_checks._stand_in_point, reproduced so that this module does not import the gate module
    -- whose pickle guard would refuse the floor-row precheck's oracle read)."""
    import pyomo.environ as pe
    n = 0
    for i, v in enumerate(block.component_data_objects(pe.Var, descend_into=True)):
        if v.fixed:
            continue
        lb = v.lb if (v.lb is not None and abs(v.lb) <= 10.0) else -10.0
        ub = v.ub if (v.ub is not None and abs(v.ub) <= 10.0) else 10.0
        if v.lb is not None and lb < v.lb:
            lb = v.lb
        if v.ub is not None and ub > v.ub:
            ub = v.ub
        if ub < lb:
            ub = lb
        f = ((i * 2654435761 + salt) % 1000003) / 1000003.0
        v.set_value(lb + f * (ub - lb), skip_validation=True)
        n += 1
    return n


def _fake_ok():
    from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
    r = SolverResults()
    r.solver.status = SolverStatus.ok
    r.solver.termination_condition = TerminationCondition.optimal
    return r


class NLStubs:
    """Every network / ESSO `optimize` replaced: at ADMM cycles 1-2 (call n = 2, 3) each block's NL file is written
    (labels off, the production form) and hashed, then a deterministic stand-in point is set; every call returns an
    ok / optimal SolverResults."""

    def __init__(self, nl_dir, hash_nl=True):
        self.nl_dir = nl_dir
        self.hash_nl = hash_nl
        self.calls = {}
        self.nl_hashes = {}

    @staticmethod
    def _salt(tag, n, year=None, day=None):
        return sum(ord(c) for c in f'{tag}|{year}|{day}') * 7919 + n * 104729

    def _nl(self, block, key):
        if not self.hash_nl:
            return
        path = os.path.join(self.nl_dir, 'tmp.nl')
        block.write(path, format='nl', io_options={'symbolic_solver_labels': False})
        self.nl_hashes[key] = _sha(path)
        os.remove(path)

    def network(self, tag):
        def stub(model, *args, **kwargs):
            n = self.calls[tag] = self.calls.get(tag, 0) + 1
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    if n in (2, 3):
                        self._nl(model[year][day], f'{tag}|{year}|{day}|cycle{n - 1}')
                    _stand_in_point(model[year][day], self._salt(tag, n, year, day))
                    results[year][day] = _fake_ok()
            return results
        return stub

    def esso(self, models, *args, **kwargs):
        n = self.calls['ESSO'] = self.calls.get('ESSO', 0) + 1
        for node_id, model in models.items():
            if n in (2, 3):
                self._nl(model, f'ESSO|{node_id}|cycle{n - 1}')
            _stand_in_point(model, self._salt('ESSO', n, node_id))
        return {node_id: _fake_ok() for node_id in models}


DIAG_FIELDS = ('cycle', 'recourse', 'gross_operational_cost', 'terminal_salvage_value', 'objective_change_abs',
               'local_solves_ok', 'priced_residual_pr')


def _diag(entry):
    out = {}
    for k in sorted(entry):
        v = entry[k]
        if k in DIAG_FIELDS or k.startswith('boyd_'):
            if isinstance(v, bool) or v is None:
                out[k] = v
            elif isinstance(v, (int, float)):
                out[k] = float.hex(float(v))
            else:
                out[k] = repr(v)
    return out


def _stub_admm(planning, sed, candidate, stubs, n_cycles):
    """Two (n_cycles) stubbed ADMM cycles through production's run_operational_planning inside the p58 rescaling the
    campaign child installs; the fixed sigma (as the step 1 probe: a stand-in point would trip the computed scale)."""
    import shared_resources_planning as srp
    import p58_rescale as R
    planning.transmission_network.optimize = stubs.network('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = stubs.network(f'DSO{node_id}')
    sed.optimize = stubs.esso
    original_scale = srp._compute_common_admm_objective_scale
    srp._compute_common_admm_objective_scale = (
        lambda planning_problem, tso_model, dso_models: float(planning_problem.params.admm.objective_scale))
    try:
        with R.patched_admm_objectives():
            _c, _r, models, _s, _p, state = planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate), print_results=False, debug_flag=False,
                return_state=True)
    finally:
        srp._compute_common_admm_objective_scale = original_scale
    return models, state


def _spec_like(derived=None, cap=SMOKE_CAP):
    """The configuration of the step 4 runs (p515_s56_coupling_gates.spec_like, by AST: importing that module arms a
    pickle guard): case-file AA declaration, ESS ageing baseline declaration with its file pin, the convergence-depth
    tail declaration; plus the derived instance."""
    import ast
    with open(os.path.join(REPO, 'p515_s56_coupling_gates.py')) as handle:
        tree = ast.parse(handle.read())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('CASE_FILE_AA', 'ESS_AGEING_BASELINE', 'ESS_AGEING_LABEL', 'CONVERGENCE_DEPTH_TAIL',
                        'REQUIRED_CONSECUTIVE_CYCLES', 'COUPLING_V1_NOQ'):
                consts[name] = ast.literal_eval(node.value)
    if consts['COUPLING_V1_NOQ'] != COUPLING_V1_NOQ:
        raise RuntimeError('COUPLING_V1_NOQ differs from p515_s56_coupling_gates')
    cfg = {'overrides': {}, 'arm_label': 's39_D', 'case_file_anderson_acceleration': dict(consts['CASE_FILE_AA']),
           'ess_ageing_baseline': deepcopy(consts['ESS_AGEING_BASELINE']),
           'ess_ageing_baseline_label': consts['ESS_AGEING_LABEL'],
           'convergence_depth_tail': dict(consts['CONVERGENCE_DEPTH_TAIL'])}
    if derived is not None:
        cfg['derived_instance'] = derived
    return cfg, int(consts['REQUIRED_CONSECUTIVE_CYCLES'])


# ======================================================================================================================
#  --nl-stub / --nl-compare (option-off NL identity, pre vs post harness)
# ======================================================================================================================
def nl_stub(args):
    out_dir = os.path.join(OUT, 'nl_identity' + TAG)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'nl_stub_{args.harness}.json')
    if os.path.exists(out_path):
        raise SystemExit(f'REFUSED: {out_path} exists')
    scratch = os.path.join(out_dir, f'scratch_{args.harness}')
    os.makedirs(scratch)
    ck = Checks()
    res = {'schema': SCHEMA + '_nl_stub', 'harness': args.harness, 'started_utc': _utc(), 'provenance': _provenance(),
           'pre_rev': args.pre_rev, 'configuration': 'SRP1 canonical, reference unit 2025, flex x2, option off, 2 '
                                                     'stubbed ADMM cycles', 'error': None}
    try:
        import p515_g_g1_g4_admm_gates as G
        from p515_s40_polish_gap import _build_floor_rows
        if args.harness == 'pre':
            mod, path = _load_module_at(args.pre_rev, 'p515_s44_campaign_harness.py', scratch, 'harness_pre')
        else:
            mod, path = H, os.path.join(REPO, 'p515_s44_campaign_harness.py')
        res['harness_sha256'] = _sha(path)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        _cc, floor_rows, _fc = _build_floor_rows(f'p515s56evh_nl_{args.harness}_{stamp}_pre')
        report, holder = {}, {}
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'arm'), report, investment_map={5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)},
            eval_id=f'p515s56evh_nl_{args.harness}_{stamp}', num_max_iters_override=2, apply_rho=False,
            investment_year=2025)
        cfg_like, required = _spec_like(None)
        spec = {'configuration': {'overrides': {}, 'case_file_anderson_acceleration':
                                  cfg_like['case_file_anderson_acceleration']},
                'cap': 2, 'required_consecutive_cycles': required}
        mod._config_hook_factory(spec, holder, overrides={}, investment_year=2025, expected_floor_rows=floor_rows,
                                 flex_price_multiplier=FLEX_M)(planning=planning, sed=sed, candidate=candidate,
                                                               report=report)
        stubs = NLStubs(scratch)
        models, state = _stub_admm(planning, sed, candidate, stubs, 2)
        diags = (state or {}).get('admm_diagnostics') or []
        res['n_cycles'] = len(diags)
        res['nl_hashes'] = stubs.nl_hashes
        res['diagnostics'] = [_diag(e) for e in diags]
        res['stub_calls'] = stubs.calls
        res['rule_eleven_checklist_keys'] = sorted(report.get('rule_eleven_checklist', {}))
        ck('two stubbed ADMM cycles ran', len(diags) == 2, len(diags))
        ck('NL files hashed for every TSO / DSO / ESSO block at cycles 1-2', len(stubs.nl_hashes) == 2 * (12 + 36 + 3),
           len(stubs.nl_hashes))
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    return _finish(res, ck, out_dir, f'nl_stub_{args.harness}')


def nl_compare(_args):
    out_dir = os.path.join(OUT, 'nl_identity' + TAG)
    ck = Checks()
    with open(os.path.join(out_dir, 'nl_stub_pre.json')) as handle:
        pre = json.load(handle)
    with open(os.path.join(out_dir, 'nl_stub_post.json')) as handle:
        post = json.load(handle)
    res = {'schema': SCHEMA + '_nl_compare', 'started_utc': _utc(), 'provenance': _provenance(), 'error': None,
           'inputs': {n: _sha(os.path.join(out_dir, f'nl_stub_{n}.json')) for n in ('pre', 'post')}}
    ck('both stub runs passed their own checks', pre['summary']['all_pass'] and post['summary']['all_pass'])
    ck('the two runs used different harness files (pre != post sha256)', pre['harness_sha256'] != post['harness_sha256'])
    differ = sorted(k for k in set(pre['nl_hashes']) | set(post['nl_hashes'])
                    if pre['nl_hashes'].get(k) != post['nl_hashes'].get(k))
    res['nl_differ'] = differ
    ck('NL identity: every TSO / DSO / ESSO block, cycles 1-2, pre == post',
       bool(pre['nl_hashes']) and not differ, differ[:10])
    ck('per-cycle diagnostics (float.hex) pre == post', pre['diagnostics'] == post['diagnostics'])
    ck('rule-eleven checklist sections pre == post', pre['rule_eleven_checklist_keys'] == post['rule_eleven_checklist_keys'],
       {'pre': pre['rule_eleven_checklist_keys'], 'post': post['rule_eleven_checklist_keys']})
    return _finish(res, ck, out_dir, 'nl_compare')


# ======================================================================================================================
#  --keys
# ======================================================================================================================
def _key_table():
    """Eval keys of a grid of settings (parent side; lattice and file reads only)."""
    derived, ev = srp1grid_ev_declaration()
    case_rel = H.s56_case_rel(derived)
    lattice = H.s56_lattice(case_rel)
    plans = {'x0': X0_STAGED, 'unit': REF_UNIT, 'staged': STAGED['srp1grid']}
    settings = {
        'none': {},
        'cohort_all': {H.S56_COHORT_FORMULATION_KEY: COHORT_MODE},
        'cohort_indep': {H.S56_COHORT_FORMULATION_KEY: 'independent_units'},
        'cohort_legacy': {H.S56_COHORT_FORMULATION_KEY: 'legacy_h3'},
        'tv_on': {H.S56_TV_IN_OBJECTIVE_KEY: True},
        'tv_off': {H.S56_TV_IN_OBJECTIVE_KEY: False},
        'slack_1e4': {H.S56_ESSO_SLACK_PENALTY_KEY: 1e4},
        'slack_10000_int': {H.S56_ESSO_SLACK_PENALTY_KEY: 10000},
        'slack_1e3': {H.S56_ESSO_SLACK_PENALTY_KEY: 1e3},
        'ev': {H.S56_EV_PROFILES_KEY: dict(ev)},
        'all': {H.S56_COHORT_FORMULATION_KEY: COHORT_MODE, H.S56_TV_IN_OBJECTIVE_KEY: True,
                H.S56_ESSO_SLACK_PENALTY_KEY: 1e4, H.S56_EV_PROFILES_KEY: dict(ev)},
    }
    out = {}
    for pname, plan in plans.items():
        canon, facts = H.s56_staged_canonical(plan, case_rel, lattice=lattice)
        ck_ = H.candidate_key(canon)
        for sname, sopts in settings.items():
            declared = H.validate_s56_ev_plan(sopts)
            keyed = H.s56_ev_plan_in_key(declared)
            for flex in (None, 2.0):
                base = H.evaluation_key(ck_, {}, flex_price_multiplier=flex, derived_instance=derived,
                                        interface_coupling=dict(COUPLING_V1_NOQ))
                out[f'{pname}|{sname}|flex{flex}'] = {'candidate_key': ck_, 'eval_key': H.s56_evaluation_key(base, keyed),
                                                      'keyed': keyed, 'investment_cost_eur': facts['investment_cost_eur']}
    return out


def keys(args):
    if args.keys_child:
        print('KEYTABLE ' + json.dumps(_key_table(), sort_keys=True), flush=True)
        return 0
    out_dir = _new_out('keys')
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    ck = Checks()
    res = {'schema': SCHEMA + '_keys', 'started_utc': _utc(), 'provenance': _provenance(), 'error': None}
    try:
        table = _key_table()
        res['key_table'] = table
        # uniqueness: distinct computations give distinct keys; declarations equal to the default share the base key
        groups = {}
        for name, v in table.items():
            groups.setdefault(v['eval_key'], []).append(name)
        expected_same = []
        for plan in ('x0', 'unit', 'staged'):
            for flex in ('None', '2.0'):
                expected_same.append(sorted(f'{plan}|{s}|flex{flex}' for s in ('none', 'cohort_legacy', 'tv_off',
                                                                              'slack_1e3')))
                expected_same.append(sorted(f'{plan}|{s}|flex{flex}' for s in ('slack_1e4', 'slack_10000_int')))
        shared = sorted(sorted(v) for v in groups.values() if len(v) > 1)
        res['shared_key_groups'] = shared
        ck('keys: the only shared eval keys are the declared-default groups (none / legacy / tv off / 1e3) and 1e4 == '
           '10000', shared == sorted(expected_same), {'shared': shared[:8]})
        n_distinct = len(groups)
        ck('keys: number of distinct eval keys == settings x plans x flex minus the declared-default merges',
           n_distinct == len(table) - sum(len(g) - 1 for g in expected_same), n_distinct)
        # every keyed eval key differs from its base
        derived, ev = srp1grid_ev_declaration()
        ck('keys: a keyed S56 eval key never equals the bare candidate key or the base key',
           all(v['eval_key'] != v['candidate_key'] for v in table.values() if v['keyed']))
        # determinism across processes
        proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--keys', '--keys-child'], cwd=REPO,
                              capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith('KEYTABLE ')), None)
        other = json.loads(line[len('KEYTABLE '):]) if line else None
        res['subprocess'] = {'returncode': proc.returncode, 'stderr_tail': proc.stderr[-2000:]}
        ck('determinism: the key table recomputed in a second process is identical',
           other is not None and json.dumps(other, sort_keys=True) == json.dumps(table, sort_keys=True),
           res['subprocess'])
        # staged canonicalisation
        case_rel = H.s56_case_rel(derived)
        lat = H.s56_lattice(case_rel)
        plan = STAGED['srp1grid']
        permuted = {n: dict(reversed(list(plan[n].items()))) for n in reversed(list(plan))}
        as_ints = {int(n): {int(y): tuple(v) for y, v in per.items()} for n, per in plan.items()}
        with_zeros = deepcopy(plan)
        with_zeros['9']['2025'] = [0.0, 0.0]
        canons = [H.s56_staged_canonical(p, case_rel, lattice=lat)[0] for p in (plan, permuted, as_ints, with_zeros)]
        keys_ = [H.candidate_key(c) for c in canons]
        ck('staged: permutation / int-or-str keys / explicit zeros give one canonical form and one key',
           len(set(keys_)) == 1 and all(c == canons[0] for c in canons), keys_)
        ck('staged: canonical is the lattice schema v2 with every node-year',
           canons[0]['schema'] == 'v2' and all(len(v) == 3 for v in canons[0]['nodes'].values()), canons[0])
        unit_v1 = H.canonical_candidate({5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)})
        unit_v2 = H.s56_staged_canonical(REF_UNIT, case_rel, lattice=lat)[0]
        ck('staged: a single-cohort plan in schema v2 never shares the single-year key',
           H.candidate_key(unit_v1) != H.candidate_key(unit_v2))
        # single-year keys unchanged (pre harness)
        pre, _p = _load_module_at(args.pre_rev, 'p515_s44_campaign_harness.py', scratch, 'harness_pre')
        grid = []
        for year in (2025, 2030, 2035):
            for s7, e7 in ((0.0, 0.0), (0.25, 0.5), (0.25, 1.0), (0.5, 1.5), (1.0, 3.75)):
                for s5 in (0.0, 0.75):
                    cand = {5: (s5, 2 * s5), 7: (s7, e7), 9: (0.0, 0.0)}
                    grid.append((pre.candidate_key(pre.canonical_candidate(cand, investment_year=year)),
                                 H.candidate_key(H.canonical_candidate(cand, investment_year=year))))
        ck('single-year keys: canonical_candidate + candidate_key pre == post on 30 candidates x 3 years',
           all(a == b for a, b in grid) and len(grid) == 30, sum(a != b for a, b in grid))
        # C5 refusals
        refusals = {
            'duration_above_4h': {'7': {'2025': [0.25, 1.5]}},
            'duration_below_2h': {'7': {'2030': [0.5, 0.5]}},
            'p0_iff_e0': {'7': {'2025': [0.25, 0.0]}},
            'negative': {'7': {'2025': [-0.25, 0.5]}},
            'off_lattice': {'7': {'2025': [0.3, 1.0]}},
            'unknown_year': {'7': {'2028': [0.25, 1.0]}},
            'unknown_node': {'6': {'2025': [0.25, 1.0]}},
            'cumulative_max_capacity': {'7': {'2025': [1.25, 5.0], '2030': [0.25, 0.5]}},
            'budget': {str(n): {str(y): [1.25, 5.0] for y in (2025,)} for n in NODES},
        }
        ref = {}
        for name, bad in refusals.items():
            try:
                H.s56_staged_canonical(bad, case_rel, lattice=lat)
                ref[name] = None
            except ValueError as error:
                ref[name] = str(error)[:400]
        res['c5_refusals'] = ref
        for name, msg in ref.items():
            ck(f'C5 refusal: {name}', msg is not None, msg)
        ck('C5 refusal reasons name their class (duration / bound / budget)',
           'duration' in (ref['duration_above_4h'] or '') and 'bound' in (ref['cumulative_max_capacity'] or '')
           and 'budget' in (ref['budget'] or ''), {k: ref[k] for k in ('duration_above_4h', 'cumulative_max_capacity',
                                                                       'budget')})
        # freeze-time refusals and acceptance
        x0 = None
        cfg = {'derived_instance': derived}

        def freeze_refused(tag, opts, cand=x0, conf=cfg):
            try:
                _freeze(H, os.path.join(scratch, f'spec_{tag}'), f's56evh_{tag}', [(tag, cand, opts)], conf)
                return None
            except ValueError as error:
                return str(error)[:400]
        bad_ev_ck = dict(ev, ev_scenario_checksum='0' * 64)
        bad_ev_sha = dict(ev, profiles_sha256='0' * 64)
        fr = {
            'ev_wrong_checksum': freeze_refused('ev_ck', {H.S56_EV_PROFILES_KEY: bad_ev_ck,
                                                          H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}),
            'ev_wrong_sha': freeze_refused('ev_sha', {H.S56_EV_PROFILES_KEY: bad_ev_sha,
                                                      H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}),
            'ev_without_derived': freeze_refused('ev_nod', {H.S56_EV_PROFILES_KEY: dict(ev),
                                                            H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}, conf=None),
            'staged_with_candidate': freeze_refused('st_cand', {H.S56_STAGED_INVESTMENT_KEY: X0_STAGED},
                                                    cand={5: (0, 0), 7: (0, 0), 9: (0, 0)}),
            'staged_with_year': freeze_refused('st_year', {H.S56_STAGED_INVESTMENT_KEY: X0_STAGED,
                                                           'investment_year': 2030}),
            'staged_inadmissible': freeze_refused('st_bad', {H.S56_STAGED_INVESTMENT_KEY: {'7': {'2025': [0.25, 1.5]}}}),
            'cohort_unknown_value': freeze_refused('co_bad', {H.S56_COHORT_FORMULATION_KEY: 'independent',
                                                              H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}),
            'tv_non_bool': freeze_refused('tv_bad', {H.S56_TV_IN_OBJECTIVE_KEY: 1,
                                                     H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}),
            'slack_non_positive': freeze_refused('sl_bad', {H.S56_ESSO_SLACK_PENALTY_KEY: 0.0,
                                                            H.S56_STAGED_INVESTMENT_KEY: X0_STAGED}),
        }
        # the 'mid' ESS parameters file: a scratch copy of the EV case naming SRP1_5x3y_ESS_Params.json
        with open(derived['case_path']) as handle:
            mid_case = json.load(handle)
        mid_case['SharedEnergyStorage']['params_file'] = H.S56_MID_SOH_PARAMS_FILES[0]
        mid_rel = os.path.relpath(os.path.join(scratch, 'case_mid_soh.json'), REPO)
        with open(mid_rel, 'x') as handle:
            json.dump(mid_case, handle, indent='\t')
        try:
            H.s56_ess_params_check(mid_rel)
            fr['mid_soh_params_file'] = None
        except ValueError as error:
            fr['mid_soh_params_file'] = str(error)[:400]
        res['freeze_refusals'] = fr
        for name, msg in fr.items():
            ck(f'freeze refusal: {name}', msg is not None, msg)
        accepted = _freeze(H, os.path.join(scratch, 'spec_accept'), 's56evh_accept',
                           [('x0', None, s56_options(ev, X0_STAGED)), ('unit', None, s56_options(ev, REF_UNIT)),
                            ('unit_v1_only', {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)},
                             {'interface_coupling': dict(COUPLING_V1_NOQ)})], cfg)
        e0, e1, e2 = accepted['candidates']
        res['accepted_spec_entries'] = {e['label']: {k: e[k] for k in e if k not in ('canonical',)} for e in
                                        accepted['candidates']}
        ck('accepted: spec and keyed entries carry the S56 label; the plain entry does not',
           accepted.get('s56_ev_plan_label') == H.S56_EV_PLAN_LABEL and e0.get('s56_ev_plan_label')
           == H.S56_EV_PLAN_LABEL and 's56_ev_plan_label' not in e2)
        ck('accepted: the entry records every key, the ESS params check and I(x)', all(
            k in e1 for k in H.S56_EV_PLAN_KEYS) and e1['s56_ess_params_check']['available_energy_soh_point_in_file']
           is None and e1['s56_investment']['investment_cost_eur'] > 0 and e0['s56_investment']['investment_cost_eur']
           == 0.0, {'unit_I': e1['s56_investment']['investment_cost_eur']})
        ck('accepted: eval key == the key table (same settings, flex 2, V-1, EV derived)',
           e1['eval_key'] == table['unit|all|flex2.0']['eval_key'], {'entry': e1['eval_key'],
                                                                     'table': table['unit|all|flex2.0']['eval_key']})
        ck('accepted: the plain entry has no S56 field', not any(k.startswith('s56_') for k in e2))
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    return _finish(res, ck, out_dir, 'keys')


# ======================================================================================================================
#  --stub-run (all keys on, zero solves)
# ======================================================================================================================
def _per_family_readback(planning, original, m):
    """Multiplier read-back per variable family on PROBE DSO blocks built by production after the multiplier: every
    coefficient of flex_p_down / flex_q_down in flex_cost_scenario == m x the original cost_flex x baseMVA."""
    from pyomo.repn import generate_standard_repn
    out = {f: {'n': 0, 'n_ok': 0, 'max_rel_dev': 0.0} for f in H.FLEX_PRICE_VARS}
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        year, day = next(iter(dn.years)), next(iter(dn.days))
        net = dn.network[year][day]
        probe = net.build_model(dn.params)
        for s_m in probe.scenarios_market:
            for s_o in probe.scenarios_operation:
                repn = generate_standard_repn(probe.flex_cost_scenario[s_m, s_o].expr, compute_values=True)
                for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                    fam = var.parent_component().local_name
                    t = var.index()[-1]
                    target = m * float(original[(node_id, year, day)][s_m][t] * net.baseMVA)
                    dev = abs(float(coef) - target) / abs(target)
                    out[fam]['n'] += 1
                    out[fam]['n_ok'] += int(dev <= H.FLEX_PRICE_READBACK_REL_TOL)
                    out[fam]['max_rel_dev'] = max(out[fam]['max_rel_dev'], dev)
    return out


def _gross_flow_check(planning, sed, models, investment):
    """Network side of D1: on every TSO / DSO block, the per-unit gross-flow flags set exactly on the node-years with
    >= 2 active cohorts, and every flagged unit's `sess_comp` entries inactive."""
    import model_construction_helpers as MCH
    import shared_energy_storage_data as SED
    counts = SED.shared_ess_active_cohort_counts(sed, investment)
    bad, n_flag = [], 0
    holders = [('TSO', planning.transmission_network, models['tso'])] + [
        (f'DSO{n}', planning.distribution_networks[n], models['dso'][n]) for n in sorted(planning.distribution_networks)]
    for tag, holder, blocks in holders:
        for year in holder.years:
            for day in holder.days:
                model = blocks[year][day]
                flags = getattr(model, MCH.SHARED_ESS_COHORT_GROSS_FLOW_ATTRIBUTE, None) or {}
                n_true = sum(1 for v in flags.values() if v is True)
                if tag == 'TSO':
                    expected = sum(1 for n in counts if counts[n][year] >= 2)
                else:
                    expected = int(counts[int(tag[3:])][year] >= 2)
                n_flag += n_true
                if n_true != expected:
                    bad.append({'block': f'{tag}|{year}|{day}', 'flags_true': n_true, 'expected': expected})
                for idx, v in flags.items():
                    if v is True and any(e.active for e in MCH._component_entries_for_shared_ess(model.sess_comp, idx)):
                        bad.append({'block': f'{tag}|{year}|{day}', 'unit': idx, 'sess_comp_active': True})
    return {'n_flags_true': n_flag, 'violations': bad[:20], 'n_violations': len(bad),
            'active_cohort_counts': {str(n): {str(y): c for y, c in v.items()} for n, v in counts.items()}}


def stub_run(args):
    grid = args.grid
    out_dir = _new_out(f'stub_run_{grid}')
    scratch = os.path.join(out_dir, 'scratch')
    os.makedirs(scratch)
    ck = Checks()
    res = {'schema': SCHEMA + '_stub_run', 'grid': grid, 'started_utc': _utc(), 'provenance': _provenance(),
           'error': None}
    t0 = time.time()
    try:
        import pyomo.environ as pe
        import p515_g_g1_g4_admm_gates as G
        import p515_s56_ev_injection as EVI
        import p515_s44_scale_measurement as S
        import shared_energy_storage_data as SED
        from p515_s40_polish_gap import _build_floor_rows
        derived, ev = ev_declaration(grid)
        cfg, required = _spec_like(derived, cap=2)
        cfg['name'] = f'S56 EV harness stub run ({grid})'
        staged = STAGED[grid]
        options = s56_options(ev, staged)
        res['options'] = options
        _p, _d, spec = H.freeze_campaign_spec(os.path.join(scratch, 'campaign'), f's56evh_stub_{grid}',
                                              [('staged', None, options)], cfg, 2, 1, ['zero-solve stub run'],
                                              required_consecutive_cycles=required)
        entry = spec['candidates'][0]
        res['entry'] = {k: v for k, v in entry.items() if k != 'canonical'}
        ck('freeze: staged plan accepted (C5), key schema v2, label at spec and entry',
           entry['canonical']['schema'] == 'v2' and spec.get('s56_ev_plan_label') == H.S56_EV_PLAN_LABEL
           and entry.get('s56_ev_plan_label') == H.S56_EV_PLAN_LABEL)
        plan = H.s56_child_declaration(spec, entry)
        checklist = H.assert_s56_ev_plan_capture_path(plan)
        res['capture_path_checklist'] = checklist
        ck('capture path asserted before any solve: every cost-breakdown field has a source',
           all(v for k, v in checklist.items() if isinstance(v, bool)) and checklist['cost_breakdown_fields']
           == list(H.S56_COST_BREAKDOWN_FIELDS))
        ck('uncoordinated_benchmark is not on the harness path (gap logged)',
           checklist['uncoordinated_benchmark_not_on_the_harness_path'], checklist['uncoordinated_benchmark_scan'])
        holder = {'s56_ev_injection_at_read': {}}
        installed = H.install_derived_instance(derived, os.path.join(scratch, 'derived'), ev_profiles=plan['ev_profiles'],
                                               ev_record=holder['s56_ev_injection_at_read'])
        res['derived_installed'] = installed
        res['ev_injection_at_read'] = holder['s56_ev_injection_at_read']
        ck('EV install: injected once, assertion holds on the read, checksum == the EV declaration',
           installed['scenario_checksum_in_child'] == ev['ev_scenario_checksum']
           and holder['s56_ev_injection_at_read'].get('hook_calls') == 1
           and holder['s56_ev_injection_at_read']['assertion_post_read']['ok'] is True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        _cc, floor_rows, _fc = _build_floor_rows(f'p515s56evh_stub_{grid}_{stamp}_pre')
        report = {}
        inv_map = H.investment_map_from_canonical(entry['canonical'])
        inv_year = H.investment_year_from_canonical(entry['canonical'])
        ck('staged: the child writes zeros at the first instance year before the hook',
           inv_year == min(int(y) for y in staged_years(entry)) and all(v == (0.0, 0.0) for v in inv_map.values()),
           {'year': inv_year, 'map': inv_map})
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'arm'), report, investment_map=inv_map, eval_id=f'p515s56evh_stub_{grid}_'
                                                                                         f'{stamp}',
            num_max_iters_override=2, apply_rho=False, investment_year=inv_year)
        hook = H._config_hook_factory(spec, holder, overrides=entry['overrides'], investment_year=inv_year,
                                      expected_floor_rows=floor_rows, flex_price_multiplier=entry['flex_price_multiplier'],
                                      interface_coupling=entry['interface_coupling'], s56_ev_plan=plan)
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        applied = holder['s56_ev_plan_applied']
        res['hook_order'] = holder.get('s56_hook_order')
        res['applied'] = applied
        ck('hook order: EV assertion -> ESSO settings + staged write -> flexibility multiplier (Audit J)',
           holder.get('s56_hook_order') == ['ev_assertion', 'esso_settings_and_staged_write', 'flex_price_multiplier'],
           holder.get('s56_hook_order'))
        ck('settings in force: cohort mode, TV in objective, slack 1e4; SoH point end',
           applied['in_force']['cohort_formulation'] == COHORT_MODE and applied['in_force']['salvage_value_in_objective']
           is True and applied['in_force']['esso_slack_penalty'] == SLACK_PENALTY
           and applied['available_energy_soh_point'] == 'end', applied['in_force'])
        ck('staged write: candidate == canonical plan on every node-year', applied['staged']['written_equals_canonical'])
        ck('probe read-back ok on every node (row inventory, TV credit, slack coefficients); floor rows == precheck',
           applied['all_ok'] and applied['floor_rows_identical_to_precheck'])
        tv = {n: v['tv'] for n, v in applied['probe_readback'].items()}
        res['tv_readback'] = tv
        ck('TV: kappa in the ESSO objective; credit linear in the terminal-SoH Vars only (converter excluded); closed '
           'form <= 1e-12', all(v.get('ok') for v in tv.values()), tv)
        # converter exclusion by perturbation (power cost x2: credit identical; energy cost x2: credit x2)
        from pyomo.repn import generate_standard_repn
        node = 7

        def credit(scale_power=1.0, scale_energy=1.0):
            saved = deepcopy(sed.cost_investment)
            try:
                for kind, sc in (('power', scale_power), ('energy', scale_energy)):
                    for scen in sed.cost_investment[kind]:
                        for y in sed.cost_investment[kind][scen]:
                            sed.cost_investment[kind][scen][y] = sed.cost_investment[kind][scen][y] * sc
                m = SED._build_subproblem(sed, node)
                SED._update_model_with_candidate_solution(sed, {node: m}, {node: candidate['investment'][node]})
                r = generate_standard_repn(m.salvage_credit_obj.expr, compute_values=True)
                return {v.name: float(c) for v, c in zip(r.linear_vars, r.linear_coefs)}, float(r.constant or 0.0)
            finally:
                sed.cost_investment = saved
        c0, k0 = credit()
        cp, kp = credit(scale_power=2.0)
        ce, ke = credit(scale_energy=2.0)
        res['converter_perturbation'] = {'base': [c0, k0], 'power_x2': [cp, kp], 'energy_x2': [ce, ke]}
        ck('converter excluded: power investment cost x2 leaves the TV credit bitwise unchanged',
           c0 == cp and k0 == kp and any(v != 0.0 for v in c0.values()), {'base': c0})
        ck('positive control: energy investment cost x2 doubles every TV credit coefficient',
           all(abs(ce[k] - 2.0 * v) <= 1e-12 * abs(v) for k, v in c0.items() if v != 0.0) and ce != c0)
        # 'mid' refused in the child
        saved_point = getattr(sed, 'available_energy_soh_point', None)
        sed.available_energy_soh_point = 'mid'
        try:
            H.apply_s56_ev_plan_settings(planning, sed, deepcopy(candidate), {}, dict(plan, staged=False), {})
            mid = None
        except RuntimeError as error:
            mid = str(error)[:300]
        finally:
            if saved_point is None:
                del sed.available_energy_soh_point
            else:
                sed.available_energy_soh_point = saved_point
        ck("'mid' SoH point refused in the child", mid is not None and "'end' required" in mid, mid)
        # the EV assertion after the multiplier RAISES (why the order matters)
        try:
            EVI.assert_ev_arrays(planning, plan['ev_profiles'], point='post_read',
                                 expected_scenario_checksum=ev['ev_scenario_checksum'])
            after = None
        except EVI.EvInjectionMismatch as exc:
            after = str(exc)[:400]
        ck('EV assertion RAISES when run after the multiplier (DSO cost_flex != stored mean)',
           after is not None and 'market arrays' in after, after)
        # multiplier read-back per family
        fam = _per_family_readback(planning, holder['_flex_price_original_arrays'], FLEX_M)
        res['multiplier_per_family'] = fam
        ck('multiplier read-back: flex_p_down AND flex_q_down coefficients == 2 x original (each family present)',
           all(v['n'] > 0 and v['n_ok'] == v['n'] for v in fam.values()), fam)
        ck('multiplier: TSO arrays and planning.cost_flex untouched', holder['flex_price_applied']['checks'][
            'tso_arrays_unchanged'] and holder['flex_price_applied']['checks']['planning_cost_flex_unchanged'])
        # the assertion raising on un-injected / SP / SRP1 reads (the in-child guard against draw 0)
        neg = {}
        sp = sp_declaration(grid)
        for name, case in (('ev_case_without_injection', derived['case_path']), ('sp', sp['case_path']),
                           ('srp1_plain', H.S56_INSTANCE_CASE_REL)):
            p_other = S.read_planning_from_derived_case({'derived_case': {'path': case}},
                                                        os.path.join(scratch, f'neg_{name}'), H._NoStageLog())
            try:
                EVI.assert_ev_arrays(p_other, plan['ev_profiles'], point='post_read',
                                     expected_scenario_checksum=ev['ev_scenario_checksum'])
                neg[name] = None
            except EVI.EvInjectionMismatch as exc:
                neg[name] = str(exc)[:300]
            del p_other
        res['assertion_negative'] = neg
        for name, msg in neg.items():
            ck(f'EV assertion RAISES on the {name} read', msg is not None, msg)
        # two stubbed ADMM cycles with everything in force
        stubs = NLStubs(scratch, hash_nl=False)
        models, state = _stub_admm(planning, sed, candidate, stubs, 2)
        diags = (state or {}).get('admm_diagnostics') or []
        ck('two stubbed ADMM cycles ran with every key in force', len(diags) == 2, len(diags))
        rows_inv = {}
        for n in sed.active_distribution_network_nodes:
            ok, det = H.s56_esso_row_inventory_check(models['esso'][n], candidate['investment'], sed, n, COHORT_MODE)
            rows_inv[str(n)] = {'ok': ok, **det}
        res['row_inventory_run_models'] = rows_inv
        multi = {n: [y for y, c in det['active_cohort_counts'].items() if c >= 2] for n, det in rows_inv.items()}
        single = {n: [y for y, c in det['active_cohort_counts'].items() if c == 1] for n, det in rows_inv.items()}
        res['node_years'] = {'multi_cohort': multi, 'single_cohort': single}
        ck('cohort mode: arrangement-A rows active on EVERY node-year with >= 1 active cohort (single and multi), none '
           'elsewhere; aggregate circle off there (run models)', all(v['ok'] for v in rows_inv.values())
           and any(multi.values()) and any(single.values()), {n: v['n_violations'] for n, v in rows_inv.items()})
        gf = _gross_flow_check(planning, sed, models, candidate['investment'])
        res['network_gross_flow'] = gf
        ck('network side (D1): gross-flow flags exactly on multi-cohort node-years, their sess_comp inactive',
           gf['n_violations'] == 0 and gf['n_flags_true'] > 0, gf['violations'][:5])
        breakdown = H.s56_cost_breakdown(planning, sed, models, diags, plan, holder)
        res['cost_breakdown'] = breakdown
        ck('cost breakdown: every declared field present', all(f in breakdown for f in H.S56_COST_BREAKDOWN_FIELDS))
        ck('cost breakdown: every field captured (non-None) on this configuration',
           all(breakdown[f] is not None for f in H.S56_COST_BREAKDOWN_FIELDS),
           [f for f in H.S56_COST_BREAKDOWN_FIELDS if breakdown[f] is None])
        ck('cost breakdown: harness checks (flex split reconciles, split read-back, kappa = a / sigma, kappa iff TV)',
           all(v for k, v in breakdown['checks'].items() if k != 'leak_within_ceiling'), breakdown['checks'])
        # production writes the coupling's per-cycle fields only while the coupling is ACTIVE, which needs > 1 scenario
        # (shared_resources_planning._per_scenario_interface_coupling_active): on the one-scenario EV twin the field is
        # absent and the record carries the label with value None
        ck('cost breakdown: priced_residual_pr label carried; value None on the one-scenario EV twin (coupling '
           'inactive in production)', breakdown['priced_residual_pr']['label'] == H.S56_PRICED_RESIDUAL_PR_LABEL
           and breakdown['priced_residual_pr']['value_terminal_cycle'] is None
           and not any('priced_residual_pr' in e for e in diags), breakdown['priced_residual_pr'])
        ck('cost breakdown: per-cohort throughput for every active cohort-block of the staged plan',
           sum(len(v) for v in breakdown['per_cohort_throughput'].values()) > 0)
        ck('cost breakdown: Q slack read (the corrected mode builds the pair)',
           breakdown['esso_slack_totals']['totals']['q']['n_nodes_with_pair'] == len(sed.active_distribution_network_nodes))
        k = breakdown['salvage_objective_kappa']
        ck('kappa in force on every run ESSO model == a / sigma (> 0; a = the ESSO AL scale param, sigma = the common '
           'objective scale param)', all(v is not None and v['equals_a_over_sigma'] and v['kappa'] > 0.0
                                         and v['sigma'] == planning.params.admm.objective_scale for v in k.values()), k)
        ck('stub calls: init + 2 cycles on every network and the ESSO', all(v == 3 for v in stubs.calls.values()),
           stubs.calls)
        res['wall_s'] = time.time() - t0
    except Exception as error:  # noqa: BLE001 -- recorded
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    return _finish(res, ck, out_dir, f'stub_run_{grid}')


def staged_years(entry):
    return {y for per in entry['canonical']['nodes'].values() for y in per}


# ======================================================================================================================
#  --smoke (the one solved run)
# ======================================================================================================================
def _committed_clean(rel):
    tracked = bool(_git('ls-files', '--', rel).strip())
    return tracked and not _git('status', '--porcelain', '--', rel).strip()


def smoke(_args):
    out_dir = _new_out('smoke')
    t0 = time.time()
    root = os.path.join(out_dir, 'campaign_s56evh_v1_smoke')   # absolute (run_admm_arm refuses a relative root)
    res = {'schema': SCHEMA + '_smoke', 'started_utc': _utc(), 'provenance': _provenance(), 'error': None,
           'declared': {'cap': SMOKE_CAP, 'concurrency': 1, 'entries': ['x0', 'unit'],
                        'solves_per_evaluation': SMOKE_DECLARED_SOLVES_PER_EVAL,
                        'solves_formula': '((1 + n_dso) x n_years x n_days + n_esso_nodes) x (cycles + 1) = '
                                          '((1 + 3) x 3 x 4 + 3) x 3',
                        'solves_total': 2 * SMOKE_DECLARED_SOLVES_PER_EVAL,
                        'keys': {H.S56_COHORT_FORMULATION_KEY: COHORT_MODE, H.S56_TV_IN_OBJECTIVE_KEY: True,
                                 H.S56_ESSO_SLACK_PENALTY_KEY: SLACK_PENALTY, 'flex_price_multiplier': FLEX_M,
                                 H.S56_EV_PROFILES_KEY: 'SRP1-grid EV twin', H.S56_STAGED_INVESTMENT_KEY: 'x0 / unit'},
                        'not_declared': ('interface_coupling: production activates the per-scenario coupling only with '
                                         '> 1 scenario (_per_scenario_interface_coupling_active); on the one-scenario EV '
                                         'twin it is inert, so the smoke does not declare it'),
                        'configuration': 'step 4 spec_like: case-file AA declared (keep_memory, on; not forced off '
                                         'without the coupling), ESS ageing baseline declared, convergence-depth tail '
                                         'declared'}}
    ck = Checks()
    try:
        ref = []
        if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
            ref.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
        for rel in (SCRIPT, 'p515_s44_campaign_harness.py', 'p515_s56_ev_injection.py', 'p516_lattice.py',
                    'p515_s56_cohort_diagnostics.py'):
            if not _committed_clean(rel):
                ref.append(f'{rel} not committed clean')
        if ref:
            raise RuntimeError(f'refused before launch (zero solves): {ref}')
        derived, ev = srp1grid_ev_declaration()
        cfg, required = _spec_like(derived, cap=SMOKE_CAP)
        cfg['name'] = 'S56 EV harness smoke: SRP1-grid EV twin, x0 and the reference unit at 2x, all keys, cap 2'
        cands = [('x0', None, s56_options(ev, X0_STAGED, coupling=False)),
                 ('unit', None, s56_options(ev, REF_UNIT, coupling=False))]
        failures = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT,))
        if failures:
            raise RuntimeError(f'refused by the harness preconditions (zero solves): {failures}')
        spec_path, csha, cspec = H.freeze_campaign_spec(
            root, 's56evh_v1_smoke', cands, cfg, SMOKE_CAP, 1,
            ['TASKS.md Working mode and EV priority, critical path step 1 (author + Expert 2026-11-23)'],
            required_consecutive_cycles=required,
            extra={'campaign_script': SCRIPT, 'campaign_script_sha256': _sha(os.path.abspath(__file__)),
                   'declared_solves_per_evaluation': SMOKE_DECLARED_SOLVES_PER_EVAL})
        res['campaign'] = {'root': os.path.relpath(root, REPO), 'spec': os.path.relpath(spec_path, REPO),
                           'spec_sha256': csha,
                           'entries': [{k: e[k] for k in ('label', 'key', 'eval_key', 'eval_dir')} for e in
                                       cspec['candidates']]}
        lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
        _log(f'smoke: lock {lock}; spec {os.path.relpath(spec_path, REPO)}')
        records = None
        try:
            ctx = H.CampaignContext(root, spec_path, csha, cspec, log=_log)
            records = H.evaluate(['x0', 'unit'], ctx)
        finally:
            res['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
        res['evaluations'] = {}
        for r, e in zip(records or [], cspec['candidates']):
            ed = os.path.join(root, 'evals', e['eval_dir'])
            per_cycle = []
            pc_path = os.path.join(ed, 'per_cycle_record.jsonl')
            if os.path.exists(pc_path):
                with open(pc_path) as handle:
                    per_cycle = [json.loads(t) for t in handle if t.strip()]
            cb = r.get('s56_cost_breakdown') or {}
            sp_ = r.get('solve_profile') or {}
            res['evaluations'][e['label']] = {
                'status': r.get('status'), 'exit_code': (r.get('parent_view') or {}).get('exit_code'),
                'cycles_run': r.get('cycles_run'), 'solve_profile': sp_, 'peak_rss': r.get('peak_rss'),
                'wall_time_s': r.get('wall_time_s'),
                'cycle_wall_s': [c.get('cycle_wall_s') for c in per_cycle],
                'rss_bytes_per_cycle': [c.get('rss_bytes') for c in per_cycle],
                'cost_breakdown': cb, 'hook_order': r.get('s56_hook_order'),
                'ev_injection_at_read': r.get('s56_ev_injection_at_read'),
                'derived_instance_checks_in_child': r.get('derived_instance_checks_in_child'),
                'multiscenario_terminal_status': (r.get('multiscenario_terminal') or {}).get('status'),
                'eval_dir': os.path.relpath(ed, REPO)}
            lab = e['label']
            ck(f'{lab}: child exit 0, 2 cycles', (r.get('parent_view') or {}).get('exit_code') == 0
               and r.get('cycles_run') == SMOKE_CAP, {'exit': (r.get('parent_view') or {}).get('exit_code'),
                                                      'cycles': r.get('cycles_run'), 'status': r.get('status')})
            ck(f'{lab}: solves == declared {SMOKE_DECLARED_SOLVES_PER_EVAL} (identity holds, nothing blocked)',
               _solves_as_declared(sp_), sp_)
            ck(f'{lab}: cost breakdown written, harness checks ok', cb.get('status') == 'written'
               and cb.get('harness_checks_ok') is True, cb.get('checks'))
            ck(f'{lab}: hook order EV assertion -> settings -> multiplier', r.get('s56_hook_order')
               == ['ev_assertion', 'esso_settings_and_staged_write', 'flex_price_multiplier'], r.get('s56_hook_order'))
    except Exception as error:  # noqa: BLE001 -- recorded (a harness fault)
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    res['wall_s'] = time.time() - t0
    return _finish(res, ck, out_dir, 'smoke')


def _solves_as_declared(sp_):
    """The solve profile of one smoke evaluation against the declared count: permitted solves == declared, nothing
    blocked, the base == declared, no retry credited, production's identity holds. (The --smoke run of 2026-11-23
    compared the `observed` DICT with the integer -- a defect of this check, recorded as FAIL in smoke.json; the
    corrected comparison is applied to the recorded profiles by --smoke-recheck, zero solves.)"""
    obs = sp_.get('observed') or {}
    return (obs.get('permitted_solve') == SMOKE_DECLARED_SOLVES_PER_EVAL
            and obs.get('permitted_exec') == SMOKE_DECLARED_SOLVES_PER_EVAL
            and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0
            and sp_.get('base_solves') == SMOKE_DECLARED_SOLVES_PER_EVAL and sp_.get('retry_solves_credited') == 0
            and sp_.get('identity_holds') is True)


def smoke_recheck(_args):
    """ZERO SOLVES: the smoke's checks re-evaluated on its RECORDED artifacts (smoke.json, each evaluation_record.json,
    s56_cost_breakdown.json and per_cycle_record.jsonl), with the corrected solve-count comparison; the measured smoke
    numbers (RSS, s/cycle, cost breakdown, ESSO slacks at 1e4, leak detector) collected in one table."""
    src = os.path.join(OUT, 'smoke')
    out_dir = _new_out('smoke_recheck')
    ck = Checks()
    with open(os.path.join(src, 'smoke.json')) as handle:
        smk = json.load(handle)
    res = {'schema': SCHEMA + '_smoke_recheck', 'started_utc': _utc(), 'provenance': _provenance(), 'error': None,
           'source': {'smoke_json': os.path.relpath(os.path.join(src, 'smoke.json'), REPO),
                      'sha256': _sha(os.path.join(src, 'smoke.json'))}, 'evaluations': {}}
    root = os.path.join(REPO, smk['campaign']['root'])
    for e in smk['campaign']['entries']:
        ed = os.path.join(root, 'evals', e['eval_dir'])
        with open(os.path.join(ed, 'evaluation_record.json')) as handle:
            r = json.load(handle)
        with open(os.path.join(ed, H.S56_COST_BREAKDOWN_FILE)) as handle:
            cb = json.load(handle)
        with open(os.path.join(ed, 'per_cycle_record.jsonl')) as handle:
            pc = [json.loads(t) for t in handle if t.strip()]
        with open(os.path.join(ed, 'exit_code.txt')) as handle:
            exit_code = int(handle.read().strip())
        lab = e['label']
        sp_ = r.get('solve_profile') or {}
        rss = r.get('peak_rss') or {}
        res['evaluations'][lab] = {
            'eval_dir': os.path.relpath(ed, REPO), 'exit_code': exit_code, 'cycles_run': r.get('cycles_run'),
            'status': r.get('status'), 'solve_profile_observed': sp_.get('observed'),
            'peak_rss_bytes': {'child_python_process': rss.get('child_python_process_ru_maxrss'),
                               'production_state': rss.get('production_state_peak_rss_ru_maxrss'),
                               'solver_subprocesses_max': rss.get('solver_subprocesses_max_ru_maxrss')},
            'wall_time_s': r.get('wall_time_s'), 'cycle_wall_s': [c.get('cycle_wall_s') for c in pc],
            'rss_bytes_per_cycle': [c.get('rss_bytes') for c in pc],
            'gross_operational_cost_per_cycle': [c.get('gross_operational_cost') for c in pc],
            'objective_change_abs_per_cycle': [c.get('objective_change_abs') for c in pc],
            'cost_breakdown': {k: cb[k] for k in ('objective_convention', 'investment_cost_eur',
                                                  'gross_operational_cost_eur', 'terminal_value_credit_eur',
                                                  'net_operational_recourse_eur', 'total_planning_cost_eur',
                                                  'flexibility_cost_reactive_down_eur', 'salvage_objective_kappa',
                                                  'priced_residual_pr')},
            'flexibility_cost_eur': {k: cb['flexibility_cost_eur'][k] for k in (
                'active_down_eur', 'reactive_down_eur', 'per_dso', 'reconciliation_with_total_flex_cost',
                'max_rel_diff_vs_flex_cost_scenario_expression')},
            'flex_price_split_readback': cb['flex_price_split_readback'],
            'esso_slack_totals': cb['esso_slack_totals'],
            'leak_detector': {k: v for k, v in cb['leak_detector'].items()},
            'per_cohort_throughput': cb['per_cohort_throughput'],
            'exit_certificate_e6_summary': cb['exit_certificate_e6']['summary'],
            'network_certificate': {k: v for k, v in cb['exit_certificate_e6']['network'].items() if k != 'per_block'},
            'breakdown_checks': cb['checks'], 'hook_order': r.get('s56_hook_order'),
            'ev_injection_at_read': r.get('s56_ev_injection_at_read'),
            'ev_assertion_in_hook_ok': (r.get('s56_ev_assertion_in_hook') or {}).get('ok'),
            'derived_instance_checks_in_child': r.get('derived_instance_checks_in_child'),
            'applied_in_force': (r.get('s56_ev_plan_applied_in_child') or {}).get('in_force'),
            'multiscenario_terminal_status': (r.get('multiscenario_terminal') or {}).get('status'),
            'operational_workbook_status': (r.get('operational_workbook') or {}).get('status')}
        ck(f'{lab}: child exit 0, 2 cycles', exit_code == 0 and r.get('cycles_run') == SMOKE_CAP)
        ck(f'{lab}: solves == declared {SMOKE_DECLARED_SOLVES_PER_EVAL} (permitted solves and execs, 0 blocked, no '
           f'retry, identity holds)', _solves_as_declared(sp_), sp_.get('observed'))
        ck(f'{lab}: cost breakdown written, harness checks ok', (r.get('s56_cost_breakdown') or {}).get('status')
           == 'written' and (r.get('s56_cost_breakdown') or {}).get('harness_checks_ok') is True
           and _sha(os.path.join(ed, H.S56_COST_BREAKDOWN_FILE)) == r['s56_cost_breakdown']['sha256'])
        ck(f'{lab}: hook order EV assertion -> settings -> multiplier', r.get('s56_hook_order')
           == ['ev_assertion', 'esso_settings_and_staged_write', 'flex_price_multiplier'])
        ck(f'{lab}: EV injected once at the read, assertion ok at the read and in the hook',
           (r.get('s56_ev_injection_at_read') or {}).get('hook_calls') == 1
           and ((r.get('s56_ev_injection_at_read') or {}).get('assertion_post_read') or {}).get('ok') is True
           and (r.get('s56_ev_assertion_in_hook') or {}).get('ok') is True)
        ck(f'{lab}: ESSO slack penalty 1e4, cohort mode and TV in force in the child',
           (r.get('s56_ev_plan_applied_in_child') or {}).get('in_force') == {
               'cohort_formulation': COHORT_MODE, 'salvage_value_enabled': True, 'salvage_value_in_objective': True,
               'esso_slack_penalty': SLACK_PENALTY})
        ck(f'{lab}: kappa = a / sigma on every ESSO model', all(v and v['equals_a_over_sigma'] for v in
                                                              cb['salvage_objective_kappa'].values()))
    res['smoke_parent_guard'] = smk.get('guards')
    ck('smoke parent process: zero solves (guard verified)', smk.get('zero_solve_verified') is True)
    return _finish(res, ck, out_dir, 'smoke_recheck')


# ======================================================================================================================
#  --manifest
# ======================================================================================================================
COMMIT_EXCLUDED_PARTS = ('/scratch', '/planning_read/', '/results/', '/esso_capture/')
COMMIT_EXCLUDED_SUFFIXES = ('.pkl', '.xlsx', '.pdf', '.png', '.nl')
COMMIT_MAX_BYTES = 1_300_000
COMMIT_SIZE_EXEMPT = (FIVE_EV_PROFILES,)   # the five-block EV instance's profiles: the instance itself


def committed_by_rule(rel, size):
    """The declared selection of the files committed with the evidence (everything else is recorded by sha256 only):
    no scratch / production-read / results / ESSO-capture directory, no pickle / workbook / plot / NL file, and at
    most COMMIT_MAX_BYTES unless it is the five-block EV profiles file."""
    if any(part in '/' + rel for part in COMMIT_EXCLUDED_PARTS) or rel.endswith(COMMIT_EXCLUDED_SUFFIXES):
        return False
    return size <= COMMIT_MAX_BYTES or os.path.basename(rel) in COMMIT_SIZE_EXEMPT


def manifest(_args):
    out = {}
    target = os.path.join(OUT_REL, 'manifest_sha256.json')
    for root, _dirs, files in os.walk(OUT):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, REPO)
            if rel == target:
                continue
            size = os.path.getsize(path)
            out[rel] = {'sha256': _sha(path), 'bytes': size, 'committed': committed_by_rule(rel, size)}
    path = os.path.join(REPO, target)
    if os.path.exists(path):
        raise SystemExit(f'REFUSED: {target} exists (write-once)')
    _dump(path, {'schema': SCHEMA + '_manifest', 'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').strip(),
                 'selection_rule': {'excluded_parts': list(COMMIT_EXCLUDED_PARTS),
                                    'excluded_suffixes': list(COMMIT_EXCLUDED_SUFFIXES),
                                    'max_bytes': COMMIT_MAX_BYTES, 'size_exempt': list(COMMIT_SIZE_EXEMPT)},
                 'n_files': len(out), 'n_committed': sum(1 for v in out.values() if v['committed']),
                 'files': dict(sorted(out.items()))})
    _log(f'manifest: {len(out)} files ({sum(1 for v in out.values() if v["committed"])} committed) -> {target} '
         f'({_sha(path)})')
    return 0


def main():
    ap = argparse.ArgumentParser()
    for flag in ('--build-ev-five-block', '--nokey', '--nl-stub', '--nl-compare', '--keys', '--keys-child',
                 '--stub-run', '--smoke', '--smoke-recheck', '--manifest'):
        ap.add_argument(flag, action='store_true')
    ap.add_argument('--pre-rev', default=DEFAULT_PRE_REV)
    ap.add_argument('--harness', choices=('pre', 'post'))
    ap.add_argument('--grid', choices=('srp1grid', 'five_block'))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    global TAG
    TAG = f'_{a.tag}' if a.tag else ''
    if a.build_ev_five_block:
        return build_ev_five_block(a)
    if a.nokey:
        return nokey(a)
    if a.nl_stub and a.harness:
        return nl_stub(a)
    if a.nl_compare:
        return nl_compare(a)
    if a.keys:
        return keys(a)
    if a.stub_run and a.grid:
        return stub_run(a)
    if a.smoke:
        return smoke(a)
    if a.smoke_recheck:
        return smoke_recheck(a)
    if a.manifest:
        return manifest(a)
    ap.error('one mode')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
