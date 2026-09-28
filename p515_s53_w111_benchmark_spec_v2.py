"""P5.15 Addendum 55, Planner task W111 -- BENCHMARK SPEC v2 and its ZERO-SOLVE CHECKS. NO BENCHMARK STAGE RUNS HERE.

A `SolveProfileGuard(permitted=())` is armed BEFORE any production import in both modes, and `verify(0)` must return []
at the end (W106's module, imported for its checks, arms its own zero-solve guard at import as well; both are verified).

WHAT v2 CHANGES (Addendum 55; everything else is v1 46b6ba96 verbatim -- models, the Q181 reference, stages, solve counts,
W93 settings, claim definition, predictions):
  1. C4 scope, option (a). The consensus item `lambda_dso_vs_lambda_tso` leaves the S48 units check and is gated on the
     SETTLED cycle-181 models (inside `units_check_identities_only`), tolerance 0.05 EUR/MWh UNCHANGED. The S48 units
     check keeps every true units item. prediction_usable = S48 units check (units items only) AND the settled
     identities (consensus item included, 0.05) AND the lambda_t sidecar cross-check
     (`p515_s53_w93_uncoordinated_benchmark.prediction_usable`). Settling datum recorded: the item was 0.0583 on the
     S48 / cycle-132 point and 0.026 on the settled cycle-181 models (W106 checks, 1ba8ae36). Node 7 / 2030 Summer /
     hour 6 is recorded as a candidate lambda != pi hour and flagged in the lambda look's table.
  2. Curtailment reporting (W109 f6e3533f: model, not helper; convention pending the author). `curtailment_table`
     reports for every arm, start and the coordinated cell the NET (production's definitional quantity, frozen, the
     primary), the POSITIVE PART (c > 0) and the NEGATIVE PART (c < 0), EUR at 1 EUR/MWh block-weighted with raw MWh
     alongside, TSO and DSO separately. The helper's definition, the model and Q are unchanged.

v2 is built as a deep copy of v1 with the changed keys overwritten (`build_spec_v2`); before writing, the keys that
must be identical are also rebuilt from the current code by W106's own `build_spec` and compared with v1 (refuses on
any difference). The v1 -> v2 key diff is printed at freeze and recorded by check C10.

MODE --freeze-spec: writes data/SRP1/Results/P515S53/w106_uncoordinated_settled/frozen_s53_benchmark_spec_v2_<hash8>.json
  once (refuses unless the highest existing version is exactly 1 and v1 still has its committed sha256; refuses if the
  bound files or this script are not clean in git).
MODE --checks (default): write-once under data/SRP1/Results/P515S53/w106_uncoordinated_settled/w111_zero_solve_checks/.
  C0 C1 C3 C6 C7 C8 C2 C5 C4b: W106's checks, unchanged, as they apply to v2 (C7 binds the highest spec = v2)
  C4v2  the lambda units-check code path under the v2 scope, on loaded models, no solve: S48 units check (units items
        only) PASS; settled identities incl. the consensus item at 0.05 PASS; sidecar PASS; prediction_usable True;
        every S48 units item and the settled consensus value equal to W106's committed evidence; the candidate hour
        addresses exactly one row and is flagged. NEGATIVE CONTROL: the consensus item planted back into the S48 check
        (`include_consensus=True`, the v1 scope) FAILS exactly as before -- 0.058284988273939575 > 0.05 on one row, at
        the candidate address -- and prediction_usable is then False
  C9    the signed curtailment parts on the settled models: net 589.43 (= the helper and the record), positive part
        651.34 (DSO 651.34, TSO 0.00), negative part -61.91 (TSO -19.24, DSO -42.67), within the 2-decimal rounding of
        the stated figures; positive + negative = net; per-agent parts equal to W109's committed figures; the report's
        `curtailment_table` carries net / positive / negative for the coordinated row and an arm row built from the
        same parts; negative controls: a stage output without the parts gives None and the report capture check lists it
  C10   v1 unchanged (committed sha256); the v1 -> v2 key diff; the keys ruled identical are identical; the differing
        top-level keys are within the declared set

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w111_benchmark_spec_v2.py --freeze-spec > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/w111_freeze_spec_v2.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w111_benchmark_spec_v2.py --checks > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/w111_zero_solve_checks.log 2>&1
Exit 0 = done / all checks pass; 1 = a check failed; 2 = refused.
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W111 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib at import
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import
import p515_s53_w106_benchmark_repoint as W106  # noqa: E402 -- arms its own zero-solve guard at import (verified too)

PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
THIS = os.path.basename(__file__)
OUT_ROOT_REL = BENCH.OUT_ROOT_REL
CHECKS_DIR_REL = os.path.join(OUT_ROOT_REL, 'w111_zero_solve_checks')
LAUNCH_LOGS_REL = os.path.join(OUT_ROOT_REL, 'launch_logs')
V1 = {'path': os.path.join(OUT_ROOT_REL, 'frozen_s53_benchmark_spec_v1_46b6ba96.json'),
      'sha256': '46b6ba967a3ac11f1fc2ee26c2e8a656d00a09b43ebd66446c22bfc67831799c',
      'committed_in': '1ba8ae369c50e5a71dc00baa0e817b021e61920a', 'code_commit': '64a1a18247be68567d60c600773fb74d05992c3d'}
W106_CHECKS = {'path': os.path.join(OUT_ROOT_REL, 'w106_zero_solve_checks', 'w106_zero_solve_checks.json'),
               'sha256': '175fe5d3ab45751ccffacb786df89708a90d32550d8f19dd296971ba1bbf1d86',
               'committed_in': '1ba8ae369c50e5a71dc00baa0e817b021e61920a'}
W109 = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w109_tso_curtailment',
                             'w109_tso_curtailment_look.json'),
        'sha256': '538361eda3fd29df11d9ae6cdbb5583f91ac7631cb0db650efa8db2285f0509f',
        'committed_in': 'f6e3533fa8318090556253f26380b50d63dc7f2c'}
# Addendum 55 / Planner task W111: the figures at the settled coordinated cell, stated to 2 decimals (EUR at 1 EUR/MWh,
# block-weighted). Asserted by C9 within half a unit of the last stated decimal (the rounding of the statement).
EXPECTED_SETTLED_CURTAILMENT_EUR = {
    ('net', 'total'): 589.43,
    ('positive_part', 'total'): 651.34, ('positive_part', 'DSO'): 651.34, ('positive_part', 'TSO'): 0.00,
    ('negative_part', 'total'): -61.91, ('negative_part', 'TSO'): -19.24, ('negative_part', 'DSO'): -42.67,
}
STATED_FIGURE_HALF_UNIT = 0.005
RECONCILIATION_ABS_EUR = 1e-9          # positive + negative - net, and against W109's committed per-agent figures
RECORD_MATCH_ABS_EUR = 1e-6            # W106 C4b's rule: recomputed net within 1e-6 EUR of the record figure
# C10: the top-level keys v2 may differ from v1 in (every other key is identical by construction and asserted).
V2_CHANGED_TOP_KEYS = ('version', 'predecessor', 'predecessor_note', 'stage', 'authority', 'frozen_utc', 'git_head',
                       'code_sha256_binding', 'code_sha256_informational', 'stages_in_addendum_49_order', 'lambda_t',
                       'curtailment_reporting', 'report_capture_paths', 'zero_solve_checks',
                       'c4_scope_addendum_55', 'curtailment_signed_parts_addendum_55')
# keys W106's builder (current code) must reproduce exactly as in v1 before v2 is written
V1_REBUILT_IDENTICAL_KEYS = ('schema', 'instance', 'coordinated', 'repointed_inputs', 'stages_in_addendum_49_order',
                             'solve_counts', 'wall_estimate', 'launch_rules', 'preparation_command', 'claim',
                             'tie_breaker', 'perturbation', 'arm_network_compl_inf_tol', 'declared_choices_status',
                             'tolerances', 'consistency_convention', 'lambda_t', 'curtailment_reporting',
                             'predictions_recorded_before_any_run', 'objective_convention', 'code_sha256_binding_rule',
                             'not_permitted')
INFORMATIONAL_FILES = W106.INFORMATIONAL_FILES + (THIS,)
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W111 +{time.time() - _T0:8.1f}s] {msg}', flush=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _load_verified_json(entry):
    got = _sha(entry['path'])
    if got != entry['sha256']:
        raise RuntimeError(f"{entry['path']}: sha256 {got} != declared {entry['sha256']}")
    with open(_abs(entry['path'])) as handle:
        return json.load(handle)


def _w106_c4():
    rec = _load_verified_json(W106_CHECKS)
    return next(r for r in rec['results'] if r['id'] == 'C4_lambda_units_check_code_path')


# ======================================================================================================================
#  the v1 -> v2 key diff
# ======================================================================================================================
def key_diff(a, b, path=''):
    """[{'path', 'kind'}] for every key path where `a` and `b` differ (dicts recursed; equal-length lists by index)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f'{path}.{k}' if path else str(k)
            if k not in a:
                out.append({'path': p, 'kind': 'added'})
            elif k not in b:
                out.append({'path': p, 'kind': 'removed'})
            else:
                out.extend(key_diff(a[k], b[k], p))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(key_diff(x, y, f'{path}[{i}]'))
    elif a != b or type(a) is not type(b):
        out.append({'path': path, 'kind': 'changed'})
    return out


# ======================================================================================================================
#  the frozen spec v2
# ======================================================================================================================
LAMBDA_LOOK_WHAT_V2 = (
    'units check on the S48 models, UNITS items only (W28/W31 reproduced to 1e-9; Param-vs-dual identities to 1e-3; '
    'the DSO-vs-TSO consensus item computed and reported, not gating: Addendum 55 option (a)), then the settled '
    'cycle-181 models: identities INCLUDING the consensus item lambda_dso_vs_lambda_tso at 0.05 EUR/MWh, the sidecar '
    'third-source cross-check, the pf-stride capture (informational) and the lambda_t vs pi_t table with the candidate '
    'lambda != pi hour (node 7 / 2030 Summer / hour 6) flagged; prediction_usable = units (units items only) AND '
    'settled identities (consensus item included) AND sidecar')


def c4_scope_record():
    w106 = _w106_c4()
    s48 = w106['units_check_s48']['items']['lambda_dso_vs_lambda_tso']
    settled = w106['identities_coordinated']['items']['lambda_dso_vs_lambda_tso']
    return {
        'ruling': ('Addendum 55, option (a): lambda_dso_vs_lambda_tso is a consensus-agreement quantity, not a units '
                   'quantity; it moves OUT of the S48 units check and is gated on the settled cycle-181 models (the '
                   'scope Addendum 54 gave the look). Not a tolerance widened after the fact'),
        'moved_item': 'lambda_dso_vs_lambda_tso',
        'tolerance_eur_per_mwh': BENCH.UNITS_CHECK_TOL['consensus_eur'],
        'tolerance_unchanged_from_v1': True,
        's48_units_items_kept': sorted(k for k in w106['units_check_s48']['items'] if k != 'lambda_dso_vs_lambda_tso'),
        'gated_on': 'identities_coordinated (units_check_identities_only on the settled cycle-181 models)',
        'still_computed_on_s48': 'units_check_s48.consensus_s48_informational (reported, not gating)',
        'prediction_usable_rule': BENCH.PREDICTION_USABLE_RULE,
        'settling_datum': {
            'quantity': 'max over 864 rows of |lambda_dso_full - lambda_tso_full| [EUR/MWh]',
            's48_cycle_132_point': s48['max_abs_eur_per_mwh'],
            'settled_cycle_181': settled['max_abs_eur_per_mwh'],
            'limit': s48['limit'],
            'reading': ('Addendum 55: the S48 point is the old cycle-132 certificate (residual-certified, unsettled); '
                        'the dual agreement tightened 0.0583 -> 0.026 on settling'),
            'source': {**W106_CHECKS, 'keys': ['C4_lambda_units_check_code_path.units_check_s48.items.'
                                               'lambda_dso_vs_lambda_tso',
                                               'C4_lambda_units_check_code_path.identities_coordinated.items.'
                                               'lambda_dso_vs_lambda_tso']},
        },
        'candidate_lambda_neq_pi_hours': [dict(c) for c in BENCH.CANDIDATE_LAMBDA_NEQ_PI_HOURS],
        'candidate_note': ('the one S48 row above 0.05 under v1 (W106: 1 of 864 rows); `hour` is the lambda table\'s '
                           '1-based hour (= period + 1); the lambda look flags it per row '
                           '(candidate_lambda_neq_pi_hour_addendum_55) and lists it under '
                           'candidate_lambda_neq_pi_hours_addendum_55, for the mechanism table'),
        'negative_control': ('W111 check C4v2: the item planted back into the S48 check (include_consensus=True, the v1 '
                             'scope) fails exactly as before (0.058284988273939575 > 0.05, one row, at the candidate '
                             'address) and prediction_usable is then False'),
    }


def curtailment_signed_parts_record():
    return {
        'definition': BENCH.CURTAILMENT_SIGNED_PARTS_DEFINITION,
        'primary': 'net (production\'s definitional quantity; frozen)',
        'reported': ['net', 'positive_part', 'negative_part'],
        'fields': list(BENCH.CURTAILMENT_FIELDS),
        'by_agent': ['TSO', 'DSO'],
        'rows': 'every arm and start (phase A) and both passive tie-breaker variants, and the coordinated cell',
        'captured_by': {'coordinated': 'common-q-gate: curtailment_signed_parts (settled models)',
                        'arms_and_variants': 'arm / passive-tie-breaker: phase_A.curtailment_signed_parts'},
        'expected_at_settled_coordinated_cell_eur_at_1_block_weighted': {
            f'{part}|{agent}': v for (part, agent), v in EXPECTED_SETTLED_CURTAILMENT_EUR.items()},
        'expected_asserted_by': (f'W111 check C9 (within {STATED_FIGURE_HALF_UNIT} EUR, the rounding of the stated '
                                 f'2-decimal figures; parts reconciled to the net within {RECONCILIATION_ABS_EUR} EUR)'),
        'w109': {**W109, 'verdict': ('model, not helper: the curtaillable pg upper bound is pg_avail + '
                                     'EQUALITY_TOLERANCE (1e-5 pu) and every negative entry is inside that band; '
                                     'helper = production\'s own term per block')},
        'not_changed': ['uncoordinated_benchmark.curtailment_report (the helper definition)', 'the model', 'Q'],
        'convention_status': 'pending the author (TASKS.md Addendum 55); net stays the primary until ruled',
    }


def build_spec_v2(v1):
    fresh = W106.build_spec(2)                  # W106's builder on the current code: the identity check
    mismatched = [k for k in V1_REBUILT_IDENTICAL_KEYS if fresh.get(k) != v1.get(k)]
    if mismatched:
        raise RuntimeError(f'current code does not rebuild v1 for keys {mismatched}; refusing to freeze')
    v2 = copy.deepcopy(v1)
    v2['version'] = 2
    v2['predecessor'] = {'path': V1['path'], 'sha256': V1['sha256'], 'version': 1,
                         'committed_in': V1['committed_in'], 'code_commit': V1['code_commit']}
    v2['predecessor_note'] = ('v2 = v1 with the Addendum 55 changes only (C4 scope, option (a); signed curtailment '
                              'reporting after W109); models, the Q181 reference, stages, solve counts, W93 settings, '
                              'the claim definition and the predictions are v1\'s (C10 asserts it). v1 is not edited')
    v2['stage'] = ('P5.15 Addendum 55 (W111): SRP1 uncoordinated benchmark on the settled cycle-181 x = 0 models -- '
                   'benchmark spec v2 (C4 consensus item on the settled models; net / positive / negative curtailment)')
    v2['authority'] = BENCH.AUTHORITY + ['TASKS.md Addendum 54 order (W106)', 'TASKS.md Addendum 55 (W111)']
    v2['frozen_utc'] = _utc()
    v2['git_head'] = _git(['rev-parse', 'HEAD'])
    v2['code_sha256_binding'] = fresh['code_sha256_binding']
    v2['code_sha256_informational'] = {name: W106._informational_pin(name) for name in INFORMATIONAL_FILES}
    lam = v2['stages_in_addendum_49_order'][0]
    if lam.get('stage') != 'lambda-look':
        raise RuntimeError(f'stage 1 of v1 is not the lambda look: {lam.get("stage")}')
    lam['what'] = LAMBDA_LOOK_WHAT_V2
    v2['lambda_t']['prediction_usable_rule'] = BENCH.PREDICTION_USABLE_RULE
    v2['curtailment_reporting'] = (
        v1['curtailment_reporting'] + '. v2 (Addendum 55; W109 ' + W109['committed_in'][:8] + ', model not helper): '
        'every row also carries the NET (production\'s definitional quantity, frozen, the primary), the POSITIVE PART '
        '(entries with c > 0) and the NEGATIVE PART (entries with c < 0), same EUR-at-1 block weighting, raw MWh '
        'alongside, TSO and DSO separately (curtailment_signed_parts_addendum_55); convention pending the author')
    v2['report_capture_paths'] = fresh['report_capture_paths']
    v2['zero_solve_checks'] = {
        'command': (f'set -o noclobber && {PY} -u {THIS} --checks > {LAUNCH_LOGS_REL}/w111_zero_solve_checks.log '
                    '2>&1'),
        'output': CHECKS_DIR_REL, 'note': 'run AFTER this freeze; C7 verifies this spec binds; C10 diffs it against v1',
        'v1_checks_superseded_for_v2': {'command': v1['zero_solve_checks']['command'],
                                        'output': v1['zero_solve_checks']['output']}}
    v2['c4_scope_addendum_55'] = c4_scope_record()
    v2['curtailment_signed_parts_addendum_55'] = curtailment_signed_parts_record()
    return v2


def freeze_spec():
    dirty = _git(['status', '--porcelain', '--'] + list(BENCH.FROZEN_SPEC_BOUND_FILES) + [THIS])
    if dirty:
        print(f'REFUSING to freeze: bound files not clean in git:\n{dirty}', flush=True)
        return 2
    highest = W106._highest_existing_version()
    if highest != 1:
        print(f'REFUSING to freeze: highest existing benchmark spec version is {highest}, expected exactly 1', flush=True)
        return 2
    v1 = _load_verified_json(V1)
    v2 = build_spec_v2(v1)
    text = GRIO.dumps(v2, indent=1, sort_keys=True) + '\n'
    data = text.encode()
    sha = hashlib.sha256(data).hexdigest()
    path = _abs(os.path.join(OUT_ROOT_REL, f'frozen_s53_benchmark_spec_v2_{sha[:8]}.json'))
    with open(path, 'xb') as handle:
        handle.write(data)
    if H.sha256_file(path) != sha:
        raise RuntimeError('frozen spec sha mismatch after write')
    diff = key_diff(v1, json.loads(text))
    failures = _GUARD.verify(0) + W106._GUARD.verify(0)
    _log(f'frozen spec v2: {os.path.relpath(path, REPO)} sha256 {sha}; predecessor v1 {V1["sha256"]}; git_head '
         f'{v2["git_head"]}; declared solves {v2["solve_counts"]}; guard verify(0) {failures}')
    _log(f'v1 -> v2 key diff ({len(diff)} paths):')
    for d in diff:
        _log(f"  {d['kind']:8s} {d['path']}")
    return 0 if not failures else 1


# ======================================================================================================================
#  checks
# ======================================================================================================================
def c4_v2(UB, planning, timings):
    w106 = _w106_c4()
    x0a = BENCH._load_json(BENCH.S48['x0_capture_analysis'])
    dnp = BENCH._load_json(BENCH.S48['dn_plateau'])
    t = time.time()
    s48 = BENCH._load_pickle_verified(BENCH.S48['certified_models'])
    timings['unpickle_s48_certified_models_s'] = time.time() - t
    rows_s48 = UB.interface_price_terms(planning, s48)
    del s48
    gc.collect()
    units = BENCH.units_check(rows_s48, x0a, dnp, include_consensus=BENCH.UNITS_CHECK_S48_CONSENSUS_GATING)
    planted = BENCH.units_check(rows_s48, x0a, dnp, include_consensus=True)       # NEGATIVE: the v1 scope
    t = time.time()
    settled_payload = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    timings['unpickle_settled_certified_models_s'] = time.time() - t
    rows = UB.interface_price_terms(planning, settled_payload)
    candidate_counts = BENCH.candidate_hour_counts(rows)
    identities = BENCH.units_check_identities_only(rows)
    sidecar = BENCH.lambda_sidecar_cross_check(rows)
    stride = BENCH.pf_capture_cross_check(rows)
    BENCH.flag_rows(rows)
    candidates = BENCH.flag_candidate_hours(rows)
    usable = BENCH.prediction_usable(units, identities, sidecar)
    usable_planted = BENCH.prediction_usable(planted, identities, sidecar)

    w106_units = w106['units_check_s48']['items']
    w106_consensus_s48 = w106_units['lambda_dso_vs_lambda_tso']
    w106_consensus_settled = w106['identities_coordinated']['items']['lambda_dso_vs_lambda_tso']
    declared = BENCH.CANDIDATE_LAMBDA_NEQ_PI_HOURS[0]
    address = {k: declared[k] for k in ('node_id', 'year', 'day', 'hour')}
    planted_item = planted['items']['lambda_dso_vs_lambda_tso']
    planted_failing = sorted(k for k, v in planted['items'].items() if not v['passed'])
    above = planted['consensus_s48_informational']['rows_above_limit']
    flagged_rows = [r for r in rows if r.get('candidate_lambda_neq_pi_hour_addendum_55') is True]
    assertions = {
        'units_s48_passed_units_items_only': units['passed'] is True,
        'units_s48_consensus_item_not_gating': ('lambda_dso_vs_lambda_tso' not in units['items']
                                               and units['consensus_item_gating_here'] is False),
        'units_s48_kept_items_are_w106_items_minus_consensus': (
            sorted(units['items']) == sorted(k for k in w106_units if k != 'lambda_dso_vs_lambda_tso')),
        'units_s48_every_item_equals_w106_evidence': all(
            units['items'][k]['max_abs_eur_per_mwh'] == w106_units[k]['max_abs_eur_per_mwh']
            and units['items'][k]['n_compared'] == w106_units[k]['n_compared'] for k in units['items']),
        'units_s48_consensus_still_reported_equals_w106': (
            units['consensus_s48_informational']['max_abs_eur_per_mwh'] == w106_consensus_s48['max_abs_eur_per_mwh']),
        'identities_settled_passed': identities['passed'] is True,
        'identities_settled_include_consensus_item_at_0p05': (
            'lambda_dso_vs_lambda_tso' in identities['items']
            and identities['items']['lambda_dso_vs_lambda_tso']['limit'] == 0.05
            and identities['items']['lambda_dso_vs_lambda_tso']['n_compared'] == 864
            and identities['items']['lambda_dso_vs_lambda_tso']['passed'] is True),
        'identities_settled_consensus_equals_w106': (
            identities['items']['lambda_dso_vs_lambda_tso']['max_abs_eur_per_mwh']
            == w106_consensus_settled['max_abs_eur_per_mwh']),
        'sidecar_passed': sidecar['passed'] is True,
        'prediction_usable_true': usable is True,
        'tolerances_unchanged': (BENCH.UNITS_CHECK_TOL == {'repro_eur': 1e-9, 'kkt_eur': 1e-3, 'consensus_eur': 0.05}
                                 and BENCH.LAMBDA_NEQ_PI_TOL_EUR == 0.05),
        'candidate_hour_matches_exactly_one_settled_row': candidate_counts == [1] and len(candidates['rows']) == 1,
        'candidate_hour_flagged_in_table_exactly_once': len(flagged_rows) == 1,
        'NEGATIVE_planted_consensus_fails': planted['passed'] is False and planted_failing == ['lambda_dso_vs_lambda_tso'],
        'NEGATIVE_planted_value_exactly_as_before': (
            planted_item['max_abs_eur_per_mwh'] == w106_consensus_s48['max_abs_eur_per_mwh']
            and planted_item['limit'] == 0.05 and planted_item['max_abs_eur_per_mwh'] > 0.05),
        'NEGATIVE_planted_one_row_at_candidate_address': (
            len(above) == 1 and {k: above[0][k] for k in address} == address
            and {k: planted['consensus_s48_informational']['argmax_row'][k] for k in address} == address),
        'NEGATIVE_prediction_usable_false_under_v1_scope': usable_planted is False,
    }
    result = {
        'id': 'C4v2_lambda_units_check_code_path_v2_scope', 'passed': all(v is True for v in assertions.values()),
        'assertions': assertions,
        'units_check_s48': units, 'identities_coordinated': identities,
        'lambda_sidecar_cross_check': {k: v for k, v in sidecar.items() if k != 'per_row_lambda_sidecar_k'},
        'pf_capture_cross_check_coordinated': stride, 'prediction_usable': usable,
        'prediction_usable_rule': BENCH.PREDICTION_USABLE_RULE,
        'candidate_lambda_neq_pi_hours_addendum_55': candidates,
        'settling_datum': {'s48_cycle_132_point': units['consensus_s48_informational']['max_abs_eur_per_mwh'],
                           'settled_cycle_181': identities['items']['lambda_dso_vs_lambda_tso']['max_abs_eur_per_mwh'],
                           'limit': 0.05},
        'NEGATIVE_planted_back_into_s48_v1_scope': {
            'passed': planted['passed'], 'failing_items': planted_failing,
            'consensus_item': planted_item, 'rows_above_limit': above,
            'prediction_usable': usable_planted,
            'w106_recorded_value': w106_consensus_s48['max_abs_eur_per_mwh']},
        'w106_evidence': W106_CHECKS, 'n_rows_coordinated': len(rows),
        'not_computed': 'the lambda_t vs pi_t table summary (the lambda-look stage); flag_rows ran only to test the flag',
    }
    return result, settled_payload


def _signed_value(signed, part, agent):
    totals = signed['totals']
    if agent == 'total':
        return totals['TSO'][part]['eur_at_1_block_weighted'] + totals['DSO'][part]['eur_at_1_block_weighted']
    return totals[agent][part]['eur_at_1_block_weighted']


def c9_curtailment(UB, planning, settled_payload):
    models = {'tso': settled_payload['tso'], 'dso': settled_payload['dso']}
    helper = UB.curtailment_report(planning, models)
    signed = BENCH.curtailment_signed_parts(planning, models, helper)
    w109 = _load_verified_json(W109)
    record = BENCH.COORDINATED['res_curtailment_definitional_at_weight_1']
    observed = {f'{part}|{agent}': _signed_value(signed, part, agent) for (part, agent) in EXPECTED_SETTLED_CURTAILMENT_EUR}
    expected = {f'{part}|{agent}': v for (part, agent), v in EXPECTED_SETTLED_CURTAILMENT_EUR.items()}
    w109_parts = {('positive_part', 'TSO'): w109['tso']['pos_part_eur_at_1_block_weighted'],
                  ('negative_part', 'TSO'): w109['tso']['neg_part_eur_at_1_block_weighted'],
                  ('positive_part', 'DSO'): w109['dso']['pos_part_eur_at_1_block_weighted'],
                  ('negative_part', 'DSO'): w109['dso']['neg_part_eur_at_1_block_weighted'],
                  ('net', 'TSO'): w109['tso']['eur_at_1_block_weighted'],
                  ('net', 'DSO'): w109['dso']['eur_at_1_block_weighted']}
    w109_diff = {f'{p}|{a}': _signed_value(signed, p, a) - v for (p, a), v in w109_parts.items()}
    residual = signed['reconciliation']['positive_plus_negative_minus_net']

    # the report stage's own table, on stage outputs as the report reads them (JSON round trip)
    signed_json = json.loads(GRIO.dumps(signed, sort_keys=True, default=GRIO.json_default_item))
    helper_totals = json.loads(GRIO.dumps(helper['totals'], sort_keys=True))
    gate_stub = {'evaluation_at_0': {'curtailment': {'totals': helper_totals}}, 'curtailment_signed_parts': signed_json}
    arm_stub = {'phase_A': {'evaluation': {'curtailment': {'totals': helper_totals}},
                            'curtailment_signed_parts': signed_json}}
    arms = {run_id: None for run_id in BENCH.ARM_RUN_IDS}
    arms['arm_passive_cold'] = arm_stub
    variants = {run_id: None for run_id in BENCH.VARIANT_RUN_IDS}
    table = BENCH.curtailment_table(gate_stub, arms, variants)
    coord_row = table['coordinated']['settled_cycle_181_signed_parts_from_models']
    arm_row = table['arms_and_variants_phase_A']['arm_passive_cold']['signed_parts']

    def row_value(row, part, agent):
        return row[part]['eur_at_1_block_weighted'] if agent == 'total' else row[part]['by_agent'][agent][
            'eur_at_1_block_weighted']
    table_ok = all(abs(row_value(row, p, a) - v) <= STATED_FIGURE_HALF_UNIT
                   for row in (coord_row, arm_row) for (p, a), v in EXPECTED_SETTLED_CURTAILMENT_EUR.items())
    # negative controls: stage outputs without the parts
    gate_bare = {'evaluation_at_0': gate_stub['evaluation_at_0']}
    arm_bare = {'phase_A': {'evaluation': arm_stub['phase_A']['evaluation']}}
    arms_bare = dict(arms, arm_passive_cold=arm_bare)
    table_bare = BENCH.curtailment_table(gate_bare, arms_bare, variants)
    capture_bare = BENCH.report_capture_check({'common_q_gate': gate_bare, 'arm_passive_cold': arm_bare})
    gate_src = inspect.getsource(BENCH.stage_common_q_gate)
    arm_src = inspect.getsource(BENCH.stage_arm)
    assertions = {
        'net_equals_helper_totals_bitwise': all(
            signed['totals'][a]['net'][f] == helper['totals'][a][f] for a in ('TSO', 'DSO')
            for f in BENCH.CURTAILMENT_FIELDS),
        'net_total_equals_record_within_1e-6': abs(_signed_value(signed, 'net', 'total') - record) <= RECORD_MATCH_ABS_EUR,
        'expected_figures_within_stated_rounding': all(abs(observed[k] - expected[k]) <= STATED_FIGURE_HALF_UNIT
                                                        for k in expected),
        'positive_plus_negative_equals_net': all(abs(residual[a][f]) <= RECONCILIATION_ABS_EUR
                                                 for a in residual for f in residual[a]),
        'per_block_entries_equal_helper': (signed['reconciliation'][
            'max_abs_block_mwh_rep_day_entries_or_parts_minus_helper_net'] <= RECONCILIATION_ABS_EUR),
        'parts_equal_w109_committed': all(abs(v) <= RECONCILIATION_ABS_EUR for v in w109_diff.values()),
        'tso_positive_part_zero': signed['totals']['TSO']['positive_part']['eur_at_1_block_weighted'] == 0.0,
        'report_table_coordinated_and_arm_rows_carry_net_positive_negative': table_ok,
        'stage_common_q_gate_captures_signed_parts': (
            "curtailment_signed_parts(planning, models, ev0['curtailment'])" in gate_src
            and "'curtailment_signed_parts': signed" in gate_src),
        'stage_arm_captures_signed_parts': (
            "'curtailment_signed_parts': curtailment_signed_parts(planning, models, evaluation['curtailment'])"
            in arm_src),
        'NEGATIVE_missing_parts_give_none': (
            table_bare['coordinated']['settled_cycle_181_signed_parts_from_models'] is None
            and table_bare['arms_and_variants_phase_A']['arm_passive_cold']['signed_parts'] is None),
        'NEGATIVE_missing_parts_listed_absent_by_report_capture_check': (
            'coordinated_curtailment_signed_parts@common_q_gate' in capture_bare['absent']
            and 'arm_curtailment_signed_parts@arm_passive_cold' in capture_bare['absent']),
    }
    compact = {k: v for k, v in signed.items() if k != 'per_block'}
    return {'id': 'C9_curtailment_signed_parts', 'passed': all(v is True for v in assertions.values()),
            'assertions': assertions, 'instance': {'cell': 'd110bd1a5977df1e_x0', 'certification_cycle': 181,
                                                   'certified_models_sha256': BENCH.COORDINATED['certified_models']['sha256']},
            'objective_convention': ('curtailment in EUR at 1 EUR/MWh, block-weighted (years x days x discount); raw '
                                     'MWh day-weighted (years x days, undiscounted) and per representative day; not '
                                     'part of Q (evaluation weight 0)'),
            'observed_eur_at_1_block_weighted': observed, 'expected_stated': expected,
            'observed_minus_expected': {k: observed[k] - expected[k] for k in expected},
            'tolerances': {'stated_figure_half_unit_eur': STATED_FIGURE_HALF_UNIT,
                           'reconciliation_abs_eur': RECONCILIATION_ABS_EUR, 'record_match_abs_eur': RECORD_MATCH_ABS_EUR},
            'record_eur_at_1_block_weighted': record,
            'signed_parts': compact, 'helper_totals': helper['totals'], 'w109_minus_this': w109_diff,
            'w109_evidence': W109, 'report_table_rows': {'coordinated': coord_row, 'arm_passive_cold_stub': arm_row},
            'NEGATIVE_capture_check_without_parts': capture_bare}


def c10_spec_diff():
    v1 = _load_verified_json(V1)
    latest = BENCH._latest_frozen_spec()
    if latest is None or latest[1] != 2:
        return {'id': 'C10_v1_v2_key_diff', 'passed': False, 'reason': f'latest spec is {latest}'}
    with open(latest[0]) as handle:
        v2 = json.load(handle)
    diff = key_diff(v1, v2)
    changed_top = sorted({d['path'].split('.')[0].split('[')[0] for d in diff})
    identical = {k: v1.get(k) == v2.get(k) for k in sorted(set(v1) | set(v2)) if k not in V2_CHANGED_TOP_KEYS}
    stages_other_than_lambda = [d for d in diff if d['path'].startswith('stages_in_addendum_49_order')
                                and d['path'] != 'stages_in_addendum_49_order[0].what']
    lambda_t_other = [d for d in diff if d['path'].startswith('lambda_t')
                      and d['path'] != 'lambda_t.prediction_usable_rule']
    checks = {
        'v1_sha256_unchanged': _sha(V1['path']) == V1['sha256'],
        'v1_file_clean_in_git': _git(['status', '--porcelain', '--', V1['path']]) == '',
        'v2_predecessor_is_v1': (v2.get('predecessor') or {}).get('sha256') == V1['sha256'],
        'changed_top_keys_within_declared_set': set(changed_top) <= set(V2_CHANGED_TOP_KEYS),
        'every_other_key_identical': all(identical.values()),
        'stages_differ_only_in_lambda_look_what': not stages_other_than_lambda,
        'lambda_t_differs_only_in_prediction_usable_rule': not lambda_t_other,
        'coordinated_models_and_q181_identical': v1['coordinated'] == v2['coordinated'],
        'solve_counts_identical': v1['solve_counts'] == v2['solve_counts'],
        'predictions_identical': v1['predictions_recorded_before_any_run'] == v2['predictions_recorded_before_any_run'],
        'tolerances_identical': v1['tolerances'] == v2['tolerances'],
        'claim_identical': v1['claim'] == v2['claim'],
    }
    return {'id': 'C10_v1_v2_key_diff', 'passed': all(checks.values()), 'checks': checks,
            'v1': {**V1}, 'v2': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'n_paths': len(diff), 'diff': diff, 'changed_top_keys': changed_top,
            'identical_top_keys': sorted(k for k, v in identical.items() if v),
            'not_identical_top_keys_outside_declared_set': sorted(k for k, v in identical.items() if not v)}


def run_checks():
    out_dir = _abs(CHECKS_DIR_REL)
    if os.path.exists(out_dir):
        print(f'REFUSING: output exists (write-once): {out_dir}', flush=True)
        return 2
    os.makedirs(out_dir)
    results, timings = [], {}

    def record(fn, *args):
        try:
            res = fn(*args)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res = {'id': getattr(fn, '__name__', 'check'), 'passed': False,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f"{res.get('id')}: {'PASS' if res.get('passed') is True else 'FAIL'}")
        results.append(res)
        return res

    record(W106.c0_preconditions)
    record(W106.c1_model_hash)
    record(W106.c3_unwired)
    c7 = record(W106.c7_spec_binding)
    if (c7.get('spec') or {}).get('version') != 2:
        c7['passed'] = False
        c7['w111_note'] = 'FAIL: the binding spec is not v2'
        _log('C7: the binding spec is not v2 -> FAIL')
    record(W106.c8_production_unchanged)
    record(W106.c6_eval_keys)
    record(c10_spec_diff)
    import shared_resources_planning as srp          # production imports only after the guards (armed at import)
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    import p58_rescale as R
    t = time.time()
    planning = O.load_baseline()['planning']
    timings['load_baseline_planning_s'] = time.time() - t
    try:
        lam, settled_payload = c4_v2(UB, planning, timings)
        results.append(lam)
        _log(f"{lam['id']}: {'PASS' if lam['passed'] else 'FAIL'} "
             f"{ {k: v for k, v in lam['assertions'].items() if v is not True} or 'all assertions True'}; settling "
             f"datum {lam['settling_datum']}")
        record(W106.c2_capture, srp, UB, R, planning, settled_payload)
        c9 = record(c9_curtailment, UB, planning, settled_payload)
        if 'observed_eur_at_1_block_weighted' in c9:
            _log(f"C9 observed {c9['observed_eur_at_1_block_weighted']}")
        del settled_payload
        gc.collect()
        t = time.time()
        w86_curtailment = W106.c4_w86_curtailment(UB, planning, timings)
        timings['c4b_w86_s'] = time.time() - t
        settled_net = c9.get('signed_parts', {}).get('totals')
        curt = {'id': 'C4b_coordinated_curtailment_recomputed',
                'settled_cycle_181': {'record': BENCH.COORDINATED['res_curtailment_definitional_at_weight_1'],
                                      'recomputed_from_models_net_eur_at_1_block_weighted': (
                                          None if settled_net is None else
                                          settled_net['TSO']['net']['eur_at_1_block_weighted']
                                          + settled_net['DSO']['net']['eur_at_1_block_weighted'])},
                'w86_cycle_132': {'record': BENCH.COORDINATED_PREVIOUS_REFERENCE[
                    'res_curtailment_definitional_at_weight_1'], 'recomputed_from_models': w86_curtailment}}
        curt['w86_cycle_132']['recomputed_minus_record_eur'] = (
            w86_curtailment['eur_at_1_block_weighted'] - curt['w86_cycle_132']['record'])
        s = curt['settled_cycle_181']
        s['recomputed_minus_record_eur'] = (None if s['recomputed_from_models_net_eur_at_1_block_weighted'] is None
                                            else s['recomputed_from_models_net_eur_at_1_block_weighted'] - s['record'])
        curt['passed'] = (s['recomputed_minus_record_eur'] is not None
                          and abs(s['recomputed_minus_record_eur']) <= RECORD_MATCH_ABS_EUR
                          and abs(curt['w86_cycle_132']['recomputed_minus_record_eur']) <= RECORD_MATCH_ABS_EUR)
        curt['pass_rule'] = 'recomputed net eur_at_1_block_weighted within 1e-6 EUR of the record figure (both cycles)'
        results.append(curt)
        _log(f"C4b_coordinated_curtailment_recomputed: {'PASS' if curt['passed'] else 'FAIL'} "
             f"181 {s['recomputed_from_models_net_eur_at_1_block_weighted']!r} (record {s['record']!r}); 132 "
             f"{w86_curtailment['eur_at_1_block_weighted']!r}")
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        results.append({'id': 'C4v2_C2_C9_C4b_model_checks', 'passed': False,
                        'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
    record(W106.c5_typing_test, out_dir)
    guard_failures = _GUARD.verify(0) + W106._GUARD.verify(0)
    passed = all(r.get('passed') is True for r in results) and not guard_failures
    payload = {'stage': 'P5.15 W111 zero-solve checks (Addendum 55; benchmark spec v2)', 'utc': _utc(),
               'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS),
               'solve_profile_guard': {'permitted': [], 'verify_0': guard_failures,
                                       'counts_w111': dict(_GUARD.counts), 'counts_w106_import': dict(W106._GUARD.counts)},
               'passed': bool(passed), 'n_checks': len(results),
               'failed': [r.get('id') for r in results if r.get('passed') is not True],
               'timings_s': timings, 'results': results}
    with open(os.path.join(out_dir, 'w111_zero_solve_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            manifest[os.path.relpath(os.path.join(root, name), REPO)] = H.sha256_file(os.path.join(root, name))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    _log(f"W111 checks: {'ALL PASS' if passed else 'FAIL ' + str(payload['failed'])}; guard verify(0) "
         f'{guard_failures}; counts W111 {dict(_GUARD.counts)} W106-import {dict(W106._GUARD.counts)}')
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--freeze-spec', action='store_true')
    group.add_argument('--checks', action='store_true')
    args = parser.parse_args(argv)
    try:
        return freeze_spec() if args.freeze_spec else run_checks()
    finally:
        W106._GUARD.uninstall()
        _GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
