"""P5.15 Addendum 54 Ruling 3, Planner task W106 -- the SRP1 uncoordinated benchmark (Addendum 49) RE-POINTED to the
settled cycle-181 x = 0 models: the FROZEN BENCHMARK SPEC and the ZERO-SOLVE CHECKS. NO BENCHMARK STAGE RUNS HERE
(not the lambda_t look, not the common-Q gate: they run after the C* extension, one at a time, by the commands the
spec lists). A `SolveProfileGuard(permitted=())` is armed BEFORE any production import in both modes and
`verify(0)` must return [] at the end.

The re-pointing itself is in `p515_s53_w93_uncoordinated_benchmark.py` (`COORDINATED` = `SETTLED`; `W86` and
`_load_w86_models` unwired, retained) and `p515_s53_w93_uncoordinated_benchmark_checks.py`. No production file is
changed (`uncoordinated_benchmark.py` takes the models as arguments and holds no cell constant).

MODE --freeze-spec: writes data/SRP1/Results/P515S53/w106_uncoordinated_settled/frozen_s53_benchmark_spec_v<N>_<hash8>.json
  once (<hash8> = the first 8 hex of the file's own sha256; N = 1 + the highest version found anywhere under data/;
  refuses if the bound files are not clean in git). Content: the coordinated models, their sha, the reference Q and its
  source; the code pins at current HEAD (`code_sha256_binding`, enforced by every stage through
  `frozen_spec_binding_failures`); the stage list in Addendum 49's order with the exact command, log path, declared
  solve count and wall estimate per stage; the claim definition and decomposition; the predictions recorded before any
  run; the inventory of every re-pointed input.

MODE --checks (default): results written once under
  data/SRP1/Results/P515S53/w106_uncoordinated_settled/w106_zero_solve_checks/.
  C0 preconditions snapshot (no campaign lock; no campaign child or G1-G4 gate process; other p515 processes listed)
  C1 model hash: the settled certified_models.pkl (full sha256) against COORDINATED, the campaign manifest committed in
     e4b5c992 (read from git), the child manifest and post_certification; the other coordinated artifacts likewise;
     negative control: a wrong declared sha is refused by the harness's own verifier
  C2 capture checklist: the harness's `common_capture_checklist` (production functions + the settled cell's records)
     and `_model_capture_checks` on the loaded settled models (duals required); every REPORT_CAPTURE_PATHS quantity has
     a producing stage whose source writes its top-level key; `report_capture_check` negative control
  C3 W86 path unwired: AST scan of the harness and the W93 checks -- `W86`, `_W86_EVAL`, `_load_w86_models`,
     `_W93_OUT_ROOT_REL`, `_W93_EVAL_ID_PREFIX` referenced only at their definitions (and W86 inside the retained
     loader); retained (hasattr); negative control: a planted reference in a stage function is detected
  C4 lambda_t units-check code path on loaded models, no solve: the S48 units check (reproduces W28 / W31), the settled
     identities, the sidecar third-source cross-check, the pf-stride capture; the coordinated curtailment recomputed
     from the settled (cycle 181) and W86 (cycle 132) models with `uncoordinated_benchmark.curtailment_report`. The
     lambda_t vs pi_t TABLE and its flags are NOT computed (that is the look, recorded as the prediction when it runs)
  C5 W100's repository-wide boolean-typing test (`p515_gate_result_bool_typing_test.main`) passes
  C6 committed eval keys unchanged: `evaluation_key` recomputed for every entry of every committed campaign spec equals
     the frozen eval key, under the harness at HEAD (W106 changes no harness) AND under the working-tree harness (which
     may carry a parallel task's uncommitted edit -- W105 -- reported, not attributed)
  C7 the frozen spec binds (`frozen_spec_binding_failures() == []`); negative controls (altered pin; name-hash mismatch)
  C8 no production file changed: git status clean for the frozen spec's bound files and the campaign harness's
     production list (the harness itself reported separately: a parallel task may be editing it)

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  mkdir -p data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w106_benchmark_repoint.py --freeze-spec > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/w106_freeze_spec.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w106_benchmark_repoint.py --checks > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/w106_zero_solve_checks.log 2>&1
Exit 0 = done / all checks pass; 1 = a check failed; 2 = refused.
"""

import argparse
import ast
import gc
import glob
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W106 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib at import
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import

PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
OUT_ROOT_REL = BENCH.OUT_ROOT_REL
CHECKS_DIR_REL = os.path.join(OUT_ROOT_REL, 'w106_zero_solve_checks')
LAUNCH_LOGS_REL = os.path.join(OUT_ROOT_REL, 'launch_logs')
HARNESS = 'p515_s53_w93_uncoordinated_benchmark.py'
CHECKS = 'p515_s53_w93_uncoordinated_benchmark_checks.py'
THIS = os.path.basename(__file__)
INFORMATIONAL_FILES = ('p515_s44_campaign_harness.py', 'settling_criterion.py', THIS,
                       'p515_gate_result_bool_typing_test.py', 'gate_result_bool_typing_historical.json')
UNWIRED_NAMES = ('W86', '_W86_EVAL', '_load_w86_models', '_W93_OUT_ROOT_REL', '_W93_EVAL_ID_PREFIX')
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W106 +{time.time() - _T0:8.1f}s] {msg}', flush=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _git_blob_sha(commit, rel):
    blob = subprocess.run(['git', 'show', f'{commit}:{rel}'], cwd=REPO, capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest(), blob


# ======================================================================================================================
#  the frozen benchmark spec
# ======================================================================================================================
def _stage_command(args, log_name):
    return (f'set -o noclobber && {PY} -u {HARNESS} {args} > '
            f'{LAUNCH_LOGS_REL}/{log_name}.log 2>&1')


def stage_list():
    """Addendum 49's order: lambda_t look -> common-Q gate -> 24-solve check -> three arms x three starts (+ passive at
    tie-breakers 0.1 and 10) -> consistency re-evaluation (inside each arm run, phase B) -> report."""
    d = BENCH.SRP1_DECLARED
    arm_solves = d['arm_solves'] + d['reevaluation_solves']
    stages = [
        {'order': 1, 'stage': 'lambda-look', 'run_id': 'lambda_look', 'command': _stage_command('--stage lambda-look',
                                                                                            'lambda_look'),
         'declared_solves': 0, 'guard': 'SolveProfileGuard(permitted=()); verify(0) at the end',
         'wall_estimate_min': [2, 6],
         'what': ('units check on the S48 models (W28/W31 reproduced to 1e-9; Param-vs-dual identities to 1e-3), then '
                  'the settled cycle-181 models: identities, the sidecar third-source cross-check, the pf-stride '
                  'capture (informational) and the lambda_t vs pi_t table; prediction_usable = units AND identities '
                  'AND sidecar')},
        {'order': 2, 'stage': 'common-q-gate', 'run_id': 'common_q_gate',
         'command': _stage_command('--stage common-q-gate', 'common_q_gate'),
         'declared_solves': 0, 'guard': 'SolveProfileGuard(permitted=()); verify(0) at the end',
         'wall_estimate_min': [2, 5],
         'what': ('evaluate_common_q (evaluation tie-breaker 0, require_unchanged) on the settled models reproduces '
                  'Q181 = 653,873,702.1876609 BITWISE (and the record\'s other recourse fields); idempotent repeat; '
                  'negative controls (tie-breaker 1 refused; repriced differs by exactly the priced curtailment)')},
        {'order': 3, 'stage': 'tso-coupling-check', 'run_id': 'tso_coupling_check',
         'command': _stage_command('--stage tso-coupling-check', 'tso_coupling_check'),
         'declared_solves': d['coupling_check_solves'],
         'declared_solves_detail': {'fixed_side': d['coupling_check_fixed_side_solves'],
                                    'penalty_side': d['coupling_check_penalty_side_solves']},
         'guard': ("SolveProfileGuard(permitted=(('uncoordinated_benchmark.py', '_solve_block'),)); verify(0) before "
                   'solves, verify(24) at the end'),
         'wall_estimate_min': [3, 8], 'reported_not_gating': True,
         'what': ('penalty (9e10 tracking) vs fixed interface at the SAME targets = the settled cycle-181 DSO schedule '
                  '(P, Q, and the fixed side\'s V pin): TN cost, interface residual, iterations, dual infeasibility')},
    ]
    order = 4
    for arm in ('passive', 'price_taker'):
        for start in ('cold', 'warm_from_certified', 'perturbed'):
            stages.append({
                'order': order, 'stage': 'arm', 'arm': arm, 'start': start, 'run_id': f'arm_{arm}_{start}',
                'command': _stage_command(f'--stage arm --arm {arm} --start {start}', f'arm_{arm}_{start}'),
                'declared_solves': arm_solves,
                'declared_solves_detail': {'phase_A_arm': d['arm_solves'],
                                           'phase_B_consistency_reevaluation': d['reevaluation_solves'],
                                           'phase_C_sequential_pass_only_if_triggered': d['sequential_pass_solves'],
                                           'total_if_triggered': arm_solves + d['sequential_pass_solves']},
                'guard': ("SolveProfileGuard(permitted=(('uncoordinated_benchmark.py', '_solve_block'),)); exact "
                          'cumulative verify at every phase boundary: 0, 48, 84 (132 if the declared trigger fires)'),
                'dso_decision_tie_breaker': BENCH.TIE_BREAKER['decision'][f'{arm}_dso'],
                'wall_estimate_min': [3, 10]})
            order += 1
    for value, tag in ((0.1, '0p1'), (10.0, '10')):
        stages.append({
            'order': order, 'stage': 'passive-tie-breaker', 'value': value, 'run_id': f'passive_tie_breaker_{tag}',
            'command': _stage_command(f'--stage passive-tie-breaker --value {value:g}', f'passive_tie_breaker_{tag}'),
            'declared_solves': d['arm_solves'],
            'guard': ("SolveProfileGuard(permitted=(('uncoordinated_benchmark.py', '_solve_block'),)); verify(0) "
                      'before, verify(48) at the end'),
            'wall_estimate_min': [2, 6],
            'what': 'passive, cold, DSO decision tie-breaker at the variant value; no consistency step (declared)'})
        order += 1
    stages.append({'order': order, 'stage': 'report', 'run_id': 'report',
                   'command': _stage_command('--stage report', 'report'), 'declared_solves': 0,
                   'guard': 'SolveProfileGuard(permitted=()); verify(0) at the end', 'wall_estimate_min': [0, 1]})
    return stages


def repointed_inputs():
    """The inventory (W106 task: 'find every place W86 / cycle-132 inputs enter'), as found at HEAD before W106."""
    return [
        {'where': f'{HARNESS} module docstring (INSTANCE / COORDINATED ARM; stages 1, 2; commands; OUTPUT)',
         'was': 'W86 cell 5cfe69a615ae3708_x0, cycle 132, 653,858,731.5686293, d816bb4b / 15f38409; w93_uncoordinated',
         'now': 'settled d110bd1a5977df1e_x0, cycle 181, Q181, 99ab1070 / 118487e1; w106_uncoordinated_settled'},
        {'where': f'{HARNESS} OUT_ROOT_REL', 'was': 'w93_uncoordinated',
         'now': 'w106_uncoordinated_settled (old kept as _W93_OUT_ROOT_REL, unwired)'},
        {'where': f'{HARNESS} EVAL_ID_PREFIX (IPOPT log dirs)', 'was': 'p515s53w93_',
         'now': 'p515s53w106_ (old kept as _W93_EVAL_ID_PREFIX, unwired)'},
        {'where': f'{HARNESS} W86 / _W86_EVAL constants (eval key 5cfe69a6, spec ddd6cd44, certified_models d816bb4b, '
                  'esso 15f38409, pf_entry_stride 38631ea3, certified_gross 653858731.5686293, cycle 132)',
         'was': 'the coordinated cell', 'now': 'UNWIRED, RETAINED; COORDINATED = SETTLED carries every input'},
        {'where': f'{HARNESS} provenance() coordinated_cell', 'was': 'W86', 'now': 'COORDINATED (+ previous, reported)'},
        {'where': f'{HARNESS} common_capture_checklist (w86_* entries: record, manifest, post_certification, shas)',
         'was': 'W86 record / child manifest', 'now': ('coordinated_* entries on the settled record, child AND campaign '
                                                       'manifest, settling_decision Q_k*, sidecar sha, evidence-commit blob')},
        {'where': f'{HARNESS} _model_capture_checks prefix', 'was': "'w86'", 'now': "'coordinated'"},
        {'where': f'{HARNESS} pf_capture_cross_check', 'was': "W86['pf_entry_stride'] (38631ea3), cycle 132",
         'now': "COORDINATED['pf_entry_stride'] (f7114758), cycle 181"},
        {'where': f'{HARNESS} stage_lambda_look (models + output keys *_w86)', 'was': 'W86 models; identities_w86, '
                  'pf_capture_cross_check_w86, summary_w86, coordination_proper_hours_w86, table_w86',
         'now': 'settled models; *_coordinated keys; + lambda_sidecar_cross_check (third source, new)'},
        {'where': f'{HARNESS} _load_w86_models', 'was': 'called by the common-Q gate and the W93 checks',
         'now': 'UNWIRED, RETAINED; _load_coordinated_models'},
        {'where': f'{HARNESS} stage_common_q_gate (record + models; reference Q)',
         'was': 'W86 record recourse_components (Q132)',
         'now': 'settled record recourse_components; reference_q = Q181 from settling_decision, bitwise-compared'},
        {'where': f'{HARNESS} stage_tso_coupling_check (reference structure; targets incl. the V pin)',
         'was': 'W86 models', 'now': 'settled models'},
        {'where': f'{HARNESS} stage_arm (reference structure; warm-from-certified values; perturbed start base; '
                  'coordinated schedule; ESSO salvage)', 'was': 'W86 models + W86 ESSO', 'now': 'settled models + ESSO'},
        {'where': f'{HARNESS} stage_report (q_coord; coordinated source; lambda summary key)',
         'was': "W86['certified_gross'] (Q132)", 'now': 'Q181; + curtailment table and report capture check'},
        {'where': f'{HARNESS} Q_MIN_OVER_STARTS_NOTE', 'was': 'names W86 5cfe69a615ae3708_x0',
         'now': 'names the settled cell'},
        {'where': f'{CHECKS} docstring (srp1-zero; commands; output path)', 'was': 'W86; w93_uncoordinated',
         'now': 'settled; w106_uncoordinated_settled'},
        {'where': f'{CHECKS} unit_common_q_gate_bitwise literal', 'was': '653858731.5686293 (Q132)',
         'now': "BENCH.COORDINATED['certified_gross'] (Q181)"},
        {'where': f'{CHECKS} srp1_zero_checks / srp1_solve_checks', 'was': "BENCH.W86['evaluation_record'], "
                  'BENCH._load_w86_models()', 'now': "BENCH.COORDINATED['evaluation_record'], "
                                                     'BENCH._load_coordinated_models()'},
        {'where': 'uncoordinated_benchmark.py (production module)', 'was': 'no cell constant (models are arguments)',
         'now': 'unchanged'},
        {'where': 'coordinated curtailment figure', 'was': '589.1778397145646 (W86 record, cycle 132)',
         'now': '589.425936658749 (settled record, cycle 181); the 132 figure reported beside it'},
    ]


def predictions():
    return {
        'recorded_before_any_run': True,
        'lambda_t_vs_pi_t_addendum_49': {
            'framing': ('Addendum 49 Ruling 1: coordination proper is the value of pricing DN flexibility at lambda_t '
                        'rather than at the wholesale price; it lives where lambda_t != pi_t -- congested or voltage-'
                        'constrained interfaces, and RES-covered hours (Addendum 32 mechanism: the TN marginal cost '
                        'is conventional at zero there). From the settled cycle-181 x = 0 cell, lambda_t (interface-P '
                        'consensus dual, converted to EUR/MWh with sigma and the interface rating only) is tabulated '
                        'against pi_t per node and hour; the hours with |lambda_t - pi_t| > 0.05 EUR/MWh AND c_flex < '
                        'pi_t bound coordination proper from above. That table, produced by the lambda-look stage, IS '
                        'the prediction and is recorded before any arm runs; it is usable only if prediction_usable'),
            'sign_note': ('the sign of passive - price-taker is not guaranteed: where lambda_t < pi_t (RES-covered '
                          'hours) a price-taker activates downward flexibility the system does not need and its cost '
                          'can exceed the passive arm\'s -- a finding (a misaligned tariff is worse than passivity), '
                          'not an error (Addendum 49 Ruling 1)'),
            'competing_advisor': ('the Advisor\'s "inside the band" expectation (Addendum 49 Ruling 1; TASKS.md '
                                  'Addendum 49 order): at SRP1, coordination proper -- min(passive, price-taker) - '
                                  'coordinated -- is inside the larger band, i.e. not determinate. Recovered as '
                                  'quoted in the brief and TASKS.md; the Advisor\'s original note is not in the '
                                  'repository (searched: *.md at the root, the W93 commit a8c58da0 message and '
                                  'docstrings)'),
        },
        'units_check': ('the S48 units check reproduces W28\'s LMP7 and W31\'s per-hour lmp / y0 / pi to 1e-9 '
                        'EUR/MWh and lambda_E to 1e-3 (Addendum 32: bus-7 marginal cost = DSO flexibility shadow '
                        'price) -- unchanged from W93 (the S48 models are not re-pointed); the Param-vs-dual '
                        'identities hold on the settled models to 1e-3 (KKT) and DSO vs TSO lambda within 0.05'),
        'lambda_t_sidecar_third_source': (
            'row 180 / s_base equals the persisted Params BITWISE on both sides (interface_dual_capture: the sidecar '
            'value is the Param of the NEXT cycle\'s solve; AA not accepted at 180/181). Row 181 lambda_dso in EUR/MWh '
            'agrees with the Param-derived lambda_dso_full within 0.05 EUR/MWh; it differs from lambda_dso_full only '
            'through the TSO interface step of cycle 181 (the DSO solves before the TSO within a cycle, so the DSO '
            'AL term used the cycle-180 TSO value while the dual update uses the cycle-181 one), hence expected well '
            'inside the tolerance at a settled point (pf primal residual 6.4e-6 at 181)'),
        'common_q_gate': ('PASS_BITWISE: evaluate_common_q at tie-breaker 0 on the settled models returns '
                          'gross_operational_cost == 653,873,702.1876609 (hex ' + float(BENCH.COORDINATED[
                              'certified_gross']).hex() + ') -- the W93 design reproduces production\'s own '
                          '_get_operational_recourse_components on the models it was computed from; the gate can '
                          'discriminate tie-breaker 0 from 1 because the settled cell curtails (589.43 EUR at weight '
                          '1 > 0), and the repriced difference equals the priced curtailment to 1e-5 EUR'),
        'coordinated_curtailment': ('the settled cycle-181 figure is 589.425936658749 (record); recomputed from the '
                                    'persisted models by the gate it matches the record; 589.1778397145646 at cycle '
                                    '132 (W86) -- both reported'),
        'arm_band': ('no quantitative arm-band expectation is recorded in Addendum 49, TASKS.md or W93/W94 (searched: '
                     'PLANNER_BRIEF_2026-09-13.md Addendum 49, TASKS.md Addendum 49 order, the W93/W94 commit '
                     'messages and harness docstrings). Restated against the settled models: each arm\'s band is the '
                     'spread of its three starts; a difference is claimed only above max(arm band, the coordinated '
                     'reproducibility band 0.011 % x Q181 = ' + f"{BENCH.COORDINATED_REPRODUCIBILITY_BAND_REL * BENCH.COORDINATED['certified_gross']:,.2f}" +
                     ' EUR); the C3-era DSO multimodality ~0.3 % is context only ("not used as a number", Addendum 49 '
                     'Resolution). The settled cell\'s settling band width (4,209.17 EUR) is far inside the 0.011 % '
                     'band and does not change the larger band'),
        'restated_from_w93_w94': {
            'passive_tie_breaker_value_independence': ('re-solving passive at 0.1 and 10 EUR/MWh leaves the interface '
                                                       'schedule within solver tolerance of the 1 EUR/MWh run; any '
                                                       'residual is the non-unique distribution of curtailment among '
                                                       'units (Addendum 49 clarification)'),
            'coupling_check_pin_feasible_by_construction': ('the fixed side\'s V targets come from the settled '
                                                            'coordinated point, where the TN held those voltages at '
                                                            'buses 5/7/9 (W94, restated for cycle 181)'),
            'coupling_check_role': 'reported, not gating: it decides whether the 9e10 scaling hazard was live',
            'q_min_over_starts': BENCH.Q_MIN_OVER_STARTS_NOTE,
            'w93_checks_srp1_solves': ('re-evaluation of one settled DSO block at its own voltage triggers nothing; one '
                                       'fixed-interface TSO block at the settled DSO schedule solves with interface '
                                       'residual <= 1e-6 MW (W93 checks, group srp1-solves, not part of the order)'),
        },
    }


def build_spec(version):
    c = BENCH.COORDINATED
    q = float(c['certified_gross'])
    decision_sha = _sha(c['settling_decision'])
    stages = stage_list()
    total = sum(s['declared_solves'] for s in stages)
    total_if_all_triggered = total + 6 * BENCH.SRP1_DECLARED['sequential_pass_solves']
    head = _git(['rev-parse', 'HEAD'])
    return {
        'schema': 'frozen_s53_benchmark_spec/v1', 'version': version,
        'predecessor': None,
        'predecessor_note': ('first frozen benchmark spec: W93/W94 carried the configuration in the harness only '
                             '(a8c58da0, b9ba413d; writers migrated in ed5ba42a); nothing ran under them'),
        'stage': 'P5.15 Addendum 54 Ruling 3 (W106): SRP1 uncoordinated benchmark on the settled cycle-181 x = 0 models',
        'authority': BENCH.AUTHORITY + ['TASKS.md Addendum 54 order (W106)'],
        'frozen_utc': _utc(), 'git_head': head,
        'code_sha256_binding': {name: _sha(name) for name in BENCH.FROZEN_SPEC_BOUND_FILES},
        'code_sha256_binding_rule': ('every benchmark stage refuses unless each listed file on disk has exactly this '
                                     'sha256 (p515_s53_w93_uncoordinated_benchmark.frozen_spec_binding_failures)'),
        'code_sha256_informational': {name: _informational_pin(name) for name in INFORMATIONAL_FILES},
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (production '
                                 '_get_operational_recourse_components); evaluation tie-breaker 0 in every arm; '
                                 'salvage reported separately'),
        'instance': {'problem': 'SRP1', 'label': 'x0', 'candidate': {str(k): v for k, v in BENCH.X0['nodes'].items()},
                     'investment_year': BENCH.X0['investment_year'], 'candidate_key': BENCH.X0['candidate_key']},
        'coordinated': {
            'cell': 'd110bd1a5977df1e_x0', 'eval_key': c['eval_key'], 'campaign_spec_sha256': c['campaign_spec_sha256'],
            'stage_spec': c['stage_spec'], 'evidence_commit': c['evidence_commit'],
            'certified_models': c['certified_models'], 'esso_models': c['esso_models'],
            'pf_entry_stride': c['pf_entry_stride'], 'interface_duals': c['interface_duals'],
            'evaluation_record': {'path': c['evaluation_record'], 'sha256': _sha(c['evaluation_record'])},
            'settling_decision': {'path': c['settling_decision'], 'sha256': decision_sha},
            'campaign_manifest': {'path': c['campaign_manifest'], 'sha256': _sha(c['campaign_manifest'])},
            'reference_q': {'value': q, 'hex': q.hex(), 'source': c['certified_gross_source'],
                            'certification_cycle': c['certification_cycle'],
                            'certification_rule': c['certification_rule']},
            'res_curtailment_definitional_at_weight_1': c['res_curtailment_definitional_at_weight_1'],
            'not_re_run': 'the coordinated arm is the settled cell, reported as its settled value (Addendum 53 Ruling 4)',
            'reproducibility_band_rel': BENCH.COORDINATED_REPRODUCIBILITY_BAND_REL,
            'reproducibility_band_eur': BENCH.COORDINATED_REPRODUCIBILITY_BAND_REL * q,
            'previous_reference_reported_only': BENCH.COORDINATED_PREVIOUS_REFERENCE,
        },
        'repointed_inputs': repointed_inputs(),
        'stages_in_addendum_49_order': stages,
        'solve_counts': {'total_declared': total, 'total_if_every_arm_triggers_the_sequential_pass':
                         total_if_all_triggered,
                         'per_stage_exact': 'each stage verifies its own declared count EXACTLY (too few fails)'},
        'wall_estimate': {
            'total_min': [sum(s['wall_estimate_min'][0] for s in stages),
                          sum(s['wall_estimate_min'][1] for s in stages)],
            'basis': ('ESTIMATES, not measurements: the settled continuation ran 181 cycles of 48 network solves + '
                      'the ESSO in 4,288 s (~0.5 s per network solve incl. overhead); IPOPT time per SRP1 network solve '
                      '0.02-0.43 s (logs of that run); planning load ~10.8 s (p56a_oracle docstring); model build and '
                      'the 164 MB unpickling not measured before the freeze (the W106 checks time the unpickling). '
                      'Addendum 54 Ruling 3 sizes the whole benchmark at ~1 h'),
        },
        'launch_rules': ('one stage at a time, attached, alone, both streams to a NEW log (noclobber); never detached; '
                         'the order above; run after the C* extension (TASKS.md Addendum 54 order); each stage refuses '
                         'while a campaign / G1-G4 gate / other p515_s53_* process or lock is live'),
        'preparation_command': f'mkdir -p {LAUNCH_LOGS_REL}',
        'claim': {
            'definition': 'benefit = min(Q_passive, Q_price_taker) - Q_coordinated (Addendum 49 Ruling 1)',
            'q_arm': BENCH.Q_MIN_OVER_STARTS_NOTE,
            'q_coordinated': 'Q181 = 653,873,702.1876609 (settled; not re-run)',
            'decomposition': ['passive - price_taker', 'price_taker - coordinated', 'passive - coordinated'],
            'resolution': ('claimed only above the larger of the best arm\'s multimodality band (spread over its three '
                           'starts) and the coordinated reproducibility band (0.011 %); the Step-5 DSO band is re-read '
                           'when available (Addendum 49 Resolution)'),
            'verdicts': ['coordination beats the best uncoordinated arrangement', 'inside the band',
                         'the best uncoordinated arrangement beats coordination'],
            'gate_first': 'no arm is reported unless the common-Q gate PASSED (Addendum 49 Ruling 3)',
        },
        'tie_breaker': BENCH.TIE_BREAKER,
        'perturbation': BENCH.PERTURBATION,
        'arm_network_compl_inf_tol': BENCH.ARM_NETWORK_COMPL_INF_TOL,
        'declared_choices_status': ('W93 declared for the Planner to confirm: arm network compl_inf_tol 1e-6, '
                                    'perturbation seed 20260926 / delta 0.05, lambda != pi tolerance 0.05 EUR/MWh. No '
                                    'confirmation found in PLANNER_BRIEF_2026-09-13.md, TASKS.md or REVISION_CONTEXT.md '
                                    '(searched for W93, 20260926, compl_inf_tol); frozen here as declared'),
        'tolerances': {'lambda_neq_pi_eur': BENCH.LAMBDA_NEQ_PI_TOL_EUR, 'units_check': BENCH.UNITS_CHECK_TOL,
                       'consistency': BENCH.CONSISTENCY_TOL,
                       'negative_control_explained_abs_eur': BENCH.NEGATIVE_CONTROL_EXPLAINED_ABS_TOL_EUR},
        'consistency_convention': ('Addendum 49: TN voltages bounded; each DN re-evaluated at the TN\'s actual '
                                   'interface voltage with the DSO decisions fixed (phase B, 36 solves per arm run); '
                                   'max |dV| at nodes 5/7/9 per hour and any DN violation reported; one sequential '
                                   'pass (48 solves) defines the arm cost if a DN limit is violated'),
        'lambda_t': {'channel_scaling': BENCH.LAMBDA_CHANNEL_SCALING,
                     'sources': ['consensus-dual Param dual_pf_p_req in the settled persisted models (DSO and TSO)',
                                 'TN interface-bus power-balance dual (TSO dual suffix)',
                                 'lambda_t sidecar interface_duals_per_cycle.jsonl rows 180 and 181 (committed in '
                                 'e4b5c992) -- third source, new in W106',
                                 'DN reference-bus dual (y0) and the pf-stride capture (informational, as in W93)'],
                     'sidecar_rule': BENCH.SIDECAR_ALIGNMENT_NOTE,
                     'prediction_usable_rule': ('units check (S48: W28/W31 reproduced) AND identities on the settled '
                                                'models AND the sidecar cross-check; the units check runs first')},
        'curtailment_reporting': ('author requirement (TASKS.md Addendum 49): curtailed energy per arm and start '
                                  'beside the coordinated figure on the SAME weighting -- EUR at 1 EUR/MWh, '
                                  'block-weighted (years x days x discount) -- with raw MWh (day-weighted and per '
                                  'representative day) alongside; coordinated: 589.425936658749 at cycle 181 (record) '
                                  'and recomputed from the models by the gate; 589.1778397145646 at cycle 132 (W86) '
                                  'reported beside it'),
        'report_capture_paths': {k: {'stage_output': v[0], 'key_path': list(v[1])}
                                 for k, v in BENCH.REPORT_CAPTURE_PATHS.items()},
        'predictions_recorded_before_any_run': predictions(),
        'zero_solve_checks': {'command': (f'set -o noclobber && {PY} -u {THIS} --checks > '
                                          f'{LAUNCH_LOGS_REL}/w106_zero_solve_checks.log 2>&1'),
                              'output': CHECKS_DIR_REL,
                              'note': 'run AFTER this freeze; C7 verifies this spec binds'},
        'not_permitted': ['running any benchmark stage before the C* extension (TASKS.md Addendum 54 order)',
                          'any production-file change', 'any change to the formulation or to the Addendum 49 '
                          'definitions'],
    }


def _informational_pin(name):
    disk = _sha(name)
    try:
        head, _blob = _git_blob_sha('HEAD', name)
    except subprocess.CalledProcessError:
        head = None
    return {'disk': disk, 'head': head, 'working_tree_differs_from_head': disk != head}


def _highest_existing_version():
    best = 0
    for path in glob.glob(_abs(os.path.join('data', '**', 'frozen_s53_benchmark_spec_v*_*.json')), recursive=True):
        match = re.fullmatch(r'frozen_s53_benchmark_spec_v(\d+)_([0-9a-f]{8})\.json', os.path.basename(path))
        if match:
            best = max(best, int(match.group(1)))
    return best


def freeze_spec():
    dirty = _git(['status', '--porcelain', '--'] + list(BENCH.FROZEN_SPEC_BOUND_FILES) + [THIS])
    if dirty:
        print(f'REFUSING to freeze: bound files not clean in git:\n{dirty}', flush=True)
        return 2
    version = _highest_existing_version() + 1
    spec = build_spec(version)
    text = GRIO.dumps(spec, indent=1, sort_keys=True) + '\n'
    data = text.encode()
    sha = hashlib.sha256(data).hexdigest()
    os.makedirs(_abs(OUT_ROOT_REL), exist_ok=True)
    path = _abs(os.path.join(OUT_ROOT_REL, f'frozen_s53_benchmark_spec_v{version}_{sha[:8]}.json'))
    with open(path, 'xb') as handle:
        handle.write(data)
    if H.sha256_file(path) != sha:
        raise RuntimeError('frozen spec sha mismatch after write')
    failures = _GUARD.verify(0)
    _log(f'frozen spec v{version}: {os.path.relpath(path, REPO)} sha256 {sha}; git_head {spec["git_head"]}; '
         f'declared solves {spec["solve_counts"]}; guard verify(0) {failures}')
    return 0 if not failures else 1


# ======================================================================================================================
#  checks
# ======================================================================================================================
def c0_preconditions():
    out = {'campaign_lock_absent': not os.path.exists(H.CAMPAIGN_LOCK_PATH),
           'legacy_lock_absent': not os.path.exists(H.LEGACY_RUN_LOCK_PATH),
           'benchmark_lock_absent': not os.path.exists(BENCH.LOCK_PATH)}
    ps = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    mine = {str(p) for p in H._ancestor_pids()}
    forbidden, observed = [], []
    for line in ps.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in mine:
            continue
        if any(s in line for s in ('p515_s4', 'p515_g_g1_g4_admm_gates')):
            forbidden.append(line.strip())
        elif 'p515_' in line or 'ipopt' in line:
            observed.append(line.strip())
    out['no_campaign_or_gate_process'] = not forbidden
    out['forbidden_processes'] = forbidden
    out['other_p515_or_ipopt_processes_observed'] = observed
    out['git_head'] = _git(['rev-parse', 'HEAD'])
    passed = all(out[k] for k in ('campaign_lock_absent', 'legacy_lock_absent', 'benchmark_lock_absent',
                                  'no_campaign_or_gate_process'))
    return {'id': 'C0_preconditions', 'passed': passed, **out}


def c1_model_hash():
    c = BENCH.COORDINATED
    t = time.time()
    disk = H.sha256_file(_abs(c['certified_models']['path']))
    wall = time.time() - t
    blob_sha, blob = _git_blob_sha(c['evidence_commit'], c['campaign_manifest'])
    committed_manifest = json.loads(blob)
    child = json.load(open(_abs(c['child_manifest'])))
    post = json.load(open(_abs(c['post_certification'])))
    rows = {}
    for name in ('certified_models', 'esso_models', 'pf_entry_stride', 'interface_duals'):
        rel, declared = c[name]['path'], c[name]['sha256']
        got = disk if name == 'certified_models' else H.sha256_file(_abs(rel))
        rows[name] = {'path': rel, 'declared': declared, 'disk': got,
                      'campaign_manifest_at_evidence_commit': committed_manifest.get(rel),
                      'child_manifest': child.get(rel)}
        rows[name]['all_equal'] = len({declared, got, committed_manifest.get(rel), child.get(rel)}) == 1
    rows['certified_models']['post_certification'] = (post.get('persisted_models') or {}).get('sha256')
    rows['certified_models']['all_equal'] = (rows['certified_models']['all_equal']
                                             and rows['certified_models']['post_certification'] == disk)
    for name in ('evaluation_record', 'settling_decision', 'post_certification'):
        rel = c[name]
        got = H.sha256_file(_abs(rel))
        rows[name] = {'path': rel, 'disk': got, 'campaign_manifest_at_evidence_commit': committed_manifest.get(rel),
                      'child_manifest': child.get(rel),
                      'all_equal': got == committed_manifest.get(rel) == child.get(rel)}
    wrong = dict(c['certified_models'], sha256='0' * 64)
    try:
        BENCH._verified_path(wrong)
        negative = False
    except RuntimeError:
        negative = True
    passed = (all(r['all_equal'] for r in rows.values()) and blob_sha == H.sha256_file(_abs(c['campaign_manifest']))
              and disk == '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a' and negative)
    return {'id': 'C1_model_hash', 'passed': passed, 'certified_models_sha256_full': disk,
            'certified_models_hash_wall_s': wall, 'evidence_commit': c['evidence_commit'],
            'campaign_manifest_blob_sha256_at_evidence_commit': blob_sha,
            'campaign_manifest_working_file_equals_blob': blob_sha == H.sha256_file(_abs(c['campaign_manifest'])),
            'artifacts': rows, 'NEGATIVE_wrong_declared_sha_refused': negative}


def _report_paths_static():
    """Every REPORT_CAPTURE_PATHS top-level key is written as a string literal by its producing stage function."""
    producers = {'lambda_look': BENCH.stage_lambda_look, 'common_q_gate': BENCH.stage_common_q_gate,
                 'tso_coupling_check': BENCH.stage_tso_coupling_check, 'arm_*': BENCH.stage_arm,
                 'passive_tie_breaker_*': BENCH.stage_arm}
    out = {}
    for quantity, (source, path) in BENCH.REPORT_CAPTURE_PATHS.items():
        src = inspect.getsource(producers[source])
        out[quantity] = f"'{path[0]}'" in src
    return out


def c2_capture(srp, UB, R, planning, settled_models):
    common = BENCH.common_capture_checklist(srp, UB, R, planning)
    model_checks = BENCH._model_capture_checks(settled_models, 'coordinated', need_duals=True)
    static = _report_paths_static()
    negative = BENCH.report_capture_check({'lambda_look': None, 'common_q_gate': {}})
    passed = (all(v is True for v in common.values()) and all(v is True for v in model_checks.values())
              and all(static.values()) and negative['all_present'] is False and len(negative['absent']) > 0)
    return {'id': 'C2_capture_checklist', 'passed': passed,
            'common_capture_checklist': common, 'n_common': len(common),
            'common_false': sorted(k for k, v in common.items() if v is not True),
            'model_capture_checks': model_checks,
            'model_false': sorted(k for k, v in model_checks.items() if v is not True),
            'report_capture_paths_written_by_producer': static,
            'NEGATIVE_report_capture_check_missing_outputs': {'all_present': negative['all_present'],
                                                              'n_absent': len(negative['absent'])}}


def _unwired_references(source, names=UNWIRED_NAMES):
    """[(name, enclosing top-level def or '<module>', lineno)] for every Name load of the given names that is not a
    permitted occurrence (definitions; W86 inside the retained `_load_w86_models`)."""
    tree = ast.parse(source)
    found = []
    for node in tree.body:
        scope = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else '<module>'
        for sub in ast.walk(node):
            name = None
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                name = sub.id
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                name = sub.attr
            if name not in names:
                continue
            if scope == '_load_w86_models' and name == 'W86':
                continue
            if scope == '<module>' and isinstance(node, ast.Assign) and name == '_W86_EVAL':
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if targets == ['W86']:
                    continue                       # the retained W86 dict is built from _W86_EVAL
            found.append((name, scope, getattr(sub, 'lineno', None)))
    return found


def c3_unwired():
    harness_src = open(_abs(HARNESS)).read()
    checks_src = open(_abs(CHECKS)).read()
    refs = {'harness': _unwired_references(harness_src), 'w93_checks': _unwired_references(checks_src),
            'this_script_stage_calls': []}
    planted = harness_src.replace("    record = _load_json(COORDINATED['evaluation_record'])\n",
                                  "    record = _load_json(W86['evaluation_record'])\n", 1)
    planted_refs = _unwired_references(planted)
    retained = {name: hasattr(BENCH, name) for name in UNWIRED_NAMES}
    retained['_load_w86_models_callable'] = callable(getattr(BENCH, '_load_w86_models', None))
    wired = {'COORDINATED_is_SETTLED': BENCH.COORDINATED is BENCH.SETTLED,
             'OUT_ROOT_is_w106': BENCH.OUT_ROOT_REL.endswith('w106_uncoordinated_settled'),
             'coordinated_models_sha_is_settled': BENCH.COORDINATED['certified_models']['sha256'].startswith(
                 '99ab1070'),
             'coordinated_q_is_q181': float(BENCH.COORDINATED['certified_gross']).hex() == float(
                 653873702.1876609).hex()}
    passed = (not refs['harness'] and not refs['w93_checks'] and planted != harness_src
              and any(r[1] == 'stage_common_q_gate' and r[0] == 'W86' for r in planted_refs)
              and all(retained.values()) and all(wired.values()))
    return {'id': 'C3_w86_unwired', 'passed': passed, 'unwired_names': list(UNWIRED_NAMES),
            'references_outside_definitions': refs, 'retained': retained, 'wired': wired,
            'NEGATIVE_planted_reference_detected': planted_refs}


def c4_lambda_and_curtailment(srp, UB, planning, timings):
    x0a = BENCH._load_json(BENCH.S48['x0_capture_analysis'])
    dnp = BENCH._load_json(BENCH.S48['dn_plateau'])
    t = time.time()
    s48 = BENCH._load_pickle_verified(BENCH.S48['certified_models'])
    timings['unpickle_s48_certified_models_s'] = time.time() - t
    rows_s48 = UB.interface_price_terms(planning, s48)
    del s48
    gc.collect()
    units = BENCH.units_check(rows_s48, x0a, dnp)
    t = time.time()
    settled_payload = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    timings['unpickle_settled_certified_models_s'] = time.time() - t
    rows = UB.interface_price_terms(planning, settled_payload)
    identities = BENCH.units_check_identities_only(rows)
    sidecar = BENCH.lambda_sidecar_cross_check(rows)
    stride = BENCH.pf_capture_cross_check(rows)
    settled_curtailment = UB.curtailment_report(planning, {'tso': settled_payload['tso'],
                                                           'dso': settled_payload['dso']})
    settled_row = BENCH._curtailment_row(settled_curtailment['totals'])
    sidecar_compact = {k: v for k, v in sidecar.items() if k != 'per_row_lambda_sidecar_k'}
    result = {'units_check_s48': units, 'identities_coordinated': identities,
              'lambda_sidecar_cross_check': sidecar_compact, 'pf_capture_cross_check_coordinated': stride,
              'n_rows_coordinated': len(rows),
              'not_computed': 'the lambda_t vs pi_t table, its flags and summary (the lambda-look stage)'}
    return result, settled_payload, settled_row


def c4_w86_curtailment(UB, planning, timings):
    t = time.time()
    w86 = BENCH._load_pickle_verified(BENCH.W86['certified_models'])      # the reported cycle-132 figure only
    timings['unpickle_w86_certified_models_s'] = time.time() - t
    report = UB.curtailment_report(planning, {'tso': w86['tso'], 'dso': w86['dso']})
    del w86
    gc.collect()
    return BENCH._curtailment_row(report['totals'])


def c5_typing_test(out_dir):
    import p515_gate_result_bool_typing_test as T
    out = os.path.join(out_dir, 'bool_typing_test.json')
    code = T.main(['--out', out, '--quiet'])
    with open(out) as handle:
        rec = json.load(handle)
    return {'id': 'C5_w100_bool_typing_test', 'passed': code == 0, 'exit_code': code,
            'output': os.path.relpath(out, REPO),
            'summary': {k: rec.get(k) for k in rec if not isinstance(rec.get(k), (list, dict))}}


def _harness_from_git(commit):
    """The campaign harness as committed at `commit`, imported from a temporary copy (the W101 pattern)."""
    import importlib.util
    sha, blob = _git_blob_sha(commit, 'p515_s44_campaign_harness.py')
    tmp = tempfile.mkdtemp(prefix='w106_harness_')
    path = os.path.join(tmp, '_w106_harness_at_head.py')
    with open(path, 'wb') as handle:
        handle.write(blob)
    spec = importlib.util.spec_from_file_location('_w106_harness_at_head', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shutil.rmtree(tmp)
    return module, sha


def _recompute_keys(module, specs):
    n = n_equal = 0
    mismatch, errors = [], []
    for rel in specs:
        spec = json.load(open(_abs(rel)))
        cfg = spec.get('configuration') or {}
        for e in spec.get('candidates') or []:
            n += 1
            overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
            kw = dict(case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=e.get('model_variant'),
                      ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
                      flex_price_multiplier=e.get('flex_price_multiplier'),
                      derived_instance=cfg.get('derived_instance'),
                      interface_deviation_premium=e.get('interface_deviation_premium'),
                      convergence_depth_tail=cfg.get('convergence_depth_tail'),
                      certification_continuation=e.get('certification_continuation'),
                      settling_continuation=e.get('settling_continuation'))
            try:
                key = module.evaluation_key(e['key'], overrides, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if key == H._entry_eval_key(e):
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'recomputed': key[:16],
                                 'frozen': H._entry_eval_key(e)[:16]})
    return {'entries': n, 'recomputed_equals_frozen': n_equal, 'mismatches': mismatch[:20], 'errors': errors[:20],
            'holds': n > 0 and n_equal == n and not mismatch and not errors}


def c6_eval_keys():
    """Every entry of every committed campaign spec: the frozen eval key is reproduced by `evaluation_key` of the
    harness at HEAD (W106 changes no harness) AND of the working-tree harness (which may carry a parallel task's
    uncommitted edit -- W105 -- recorded, not attributed to W106)."""
    specs = [p for p in _git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()]
    head_module, head_sha = _harness_from_git('HEAD')
    at_head = _recompute_keys(head_module, specs)
    in_tree = _recompute_keys(H, specs)
    disk = _sha('p515_s44_campaign_harness.py')
    return {'id': 'C6_committed_eval_keys_unchanged', 'passed': bool(at_head['holds'] and in_tree['holds']),
            'committed_specs': len(specs), 'harness_at_head': {'sha256': head_sha, **at_head},
            'harness_working_tree': {'sha256': disk, 'differs_from_head': disk != head_sha,
                                     'git_status': _git(['status', '--porcelain', '--',
                                                         'p515_s44_campaign_harness.py']), **in_tree},
            'w106_modifies_harness': False,
            'settling_criterion_sha256_disk': _sha('settling_criterion.py')}


def c7_spec_binding():
    failures = BENCH.frozen_spec_binding_failures()
    latest = BENCH._latest_frozen_spec()
    negatives = {}
    if latest is not None:
        path, version, hash8 = latest
        tmp = tempfile.mkdtemp(prefix='w106_spec_neg_')
        saved = BENCH.OUT_ROOT_REL
        try:
            spec = json.load(open(path))
            spec['code_sha256_binding'][HARNESS] = '0' * 64
            data = (GRIO.dumps(spec, indent=1, sort_keys=True) + '\n').encode()
            sha = hashlib.sha256(data).hexdigest()
            with open(os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_{sha[:8]}.json'), 'wb') as handle:
                handle.write(data)
            BENCH.OUT_ROOT_REL = os.path.relpath(tmp, REPO)
            altered = BENCH.frozen_spec_binding_failures()
            negatives['altered_pin_refused'] = any(HARNESS in f for f in altered)
            os.remove(os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_{sha[:8]}.json'))
            shutil.copyfile(path, os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_00000000.json'))
            renamed = BENCH.frozen_spec_binding_failures()
            negatives['name_hash_mismatch_refused'] = any('does not start with its name hash' in f for f in renamed)
        finally:
            BENCH.OUT_ROOT_REL = saved
            shutil.rmtree(tmp)
    passed = latest is not None and not failures and negatives and all(negatives.values())
    return {'id': 'C7_frozen_spec_binds', 'passed': bool(passed),
            'spec': None if latest is None else {'path': os.path.relpath(latest[0], REPO), 'version': latest[1],
                                                 'sha256': H.sha256_file(latest[0])},
            'binding_failures': failures, 'NEGATIVE': negatives}


def c8_production_unchanged():
    """No production file changed: every file the benchmark depends on (the frozen spec's bound list, which includes
    the production modules) is clean in git, and so is every other production file of the campaign harness's list
    except the harness itself, whose working-tree state (a parallel task's edit, if any) is reported."""
    harness = 'p515_s44_campaign_harness.py'
    files = sorted(set(BENCH.FROZEN_SPEC_BOUND_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) - {harness})
    status = _git(['status', '--porcelain', '--'] + files)
    last = _git(['log', '-1', '--format=%H', '--', 'uncoordinated_benchmark.py'])
    return {'id': 'C8_no_production_change', 'passed': status == '', 'git_status': status, 'files_checked': files,
            'harness_git_status_reported': _git(['status', '--porcelain', '--', harness]),
            'uncoordinated_benchmark_last_commit': last}


def run_checks():
    out_dir = _abs(CHECKS_DIR_REL)
    if os.path.exists(out_dir):
        print(f'REFUSING: output exists (write-once): {out_dir}', flush=True)
        return 2
    os.makedirs(out_dir)
    results, timings, extra = [], {}, {}

    def record(fn, *args):
        try:
            res = fn(*args)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res = {'id': getattr(fn, '__name__', 'check'), 'passed': False,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f"{res.get('id')}: {'PASS' if res.get('passed') else 'FAIL'}")
        results.append(res)
        return res

    record(c0_preconditions)
    record(c1_model_hash)
    record(c3_unwired)
    record(c7_spec_binding)
    record(c8_production_unchanged)
    record(c6_eval_keys)
    import shared_resources_planning as srp          # production imports only after the guard (armed at import)
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    import p58_rescale as R
    t = time.time()
    planning = O.load_baseline()['planning']
    timings['load_baseline_planning_s'] = time.time() - t
    try:
        lam, settled_payload, settled_curtailment = c4_lambda_and_curtailment(srp, UB, planning, timings)
        lam_ok = (lam['units_check_s48']['passed'] and lam['identities_coordinated']['passed']
                  and lam['lambda_sidecar_cross_check']['passed'])
        results.append({'id': 'C4_lambda_units_check_code_path', 'passed': bool(lam_ok), **lam})
        _log(f"C4_lambda_units_check_code_path: {'PASS' if lam_ok else 'FAIL'} "
             f"(units {lam['units_check_s48']['passed']}, identities {lam['identities_coordinated']['passed']}, "
             f"sidecar {lam['lambda_sidecar_cross_check']['passed']})")
        record(c2_capture, srp, UB, R, planning, settled_payload)
        del settled_payload
        gc.collect()
        w86_curtailment = c4_w86_curtailment(UB, planning, timings)
        curt = {'id': 'C4b_coordinated_curtailment_recomputed',
                'settled_cycle_181': {'record': BENCH.COORDINATED['res_curtailment_definitional_at_weight_1'],
                                      'recomputed_from_models': settled_curtailment},
                'w86_cycle_132': {'record': BENCH.COORDINATED_PREVIOUS_REFERENCE[
                    'res_curtailment_definitional_at_weight_1'], 'recomputed_from_models': w86_curtailment}}
        for k in ('settled_cycle_181', 'w86_cycle_132'):
            curt[k]['recomputed_minus_record_eur'] = (curt[k]['recomputed_from_models']['eur_at_1_block_weighted']
                                                      - curt[k]['record'])
        curt['passed'] = all(abs(curt[k]['recomputed_minus_record_eur']) <= 1e-6 for k in ('settled_cycle_181',
                                                                                         'w86_cycle_132'))
        curt['pass_rule'] = 'recomputed eur_at_1_block_weighted within 1e-6 EUR of the record figure (both cycles)'
        results.append(curt)
        _log(f"C4b curtailment: 181 {settled_curtailment['eur_at_1_block_weighted']!r} (record "
             f"{curt['settled_cycle_181']['record']!r}); 132 {w86_curtailment['eur_at_1_block_weighted']!r}")
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        results.append({'id': 'C4_C2_model_checks', 'passed': False, 'error': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc()})
    record(c5_typing_test, out_dir)
    guard_failures = _GUARD.verify(0)
    passed = all(r.get('passed') is True for r in results) and not guard_failures
    payload = {'stage': 'P5.15 W106 zero-solve checks (Addendum 54 Ruling 3)', 'utc': _utc(),
               'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS),
               'solve_profile_guard': {'permitted': [], 'verify_0': guard_failures, 'counts': dict(_GUARD.counts)},
               'passed': bool(passed), 'n_checks': len(results),
               'failed': [r.get('id') for r in results if r.get('passed') is not True],
               'timings_s': timings, 'results': results, **extra}
    with open(os.path.join(out_dir, 'w106_zero_solve_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            manifest[os.path.relpath(os.path.join(root, name), REPO)] = H.sha256_file(os.path.join(root, name))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    _log(f"W106 checks: {'ALL PASS' if passed else 'FAIL ' + str(payload['failed'])}; guard verify(0) "
         f'{guard_failures}; counts {dict(_GUARD.counts)}')
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
        _GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
