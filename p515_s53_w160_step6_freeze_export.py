"""P5.15 Addendum 65, Planner task W160 -- FREEZE THE STEP 6 TABLES WITH A RECORDED HASH AND EXPORT THEM
MANUSCRIPT-READY. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W157 builder (p515_s53_w157_step6_tables.py, 4c89daf8)
is IMPORTED and its functions are CALLED; it is not edited. No harness, scorer or production module is edited.

WHAT IT DOES
  1. Takes the W157 tables AS THEY ARE: data/SRP1/Results/P515S53/w157_step6_tables_a64/w157_step6_tables.json
     (commit b913ea94; sha256 checked against its committed manifest). As an integrity check it also re-reads every
     W157 input through `W157.load_inputs()` (committed clean, manifests matched) and rebuilds the ten tables with the
     W157 functions; the rebuild must equal the committed tables after a JSON round trip.
  2. Adds, per the Addendum 65 order (Planner task W160):
     T6  both NRF arms in full from report_v3.json (8d42dfb8; spec v5 bca69f97): Q by start and the best start; the
         recorded decomposition; the passive arm's benefit relative to Q181 and its multiple of the larger band,
         DERIVED from the recorded values and labelled "derived"; the no-reverse-flow definition; the consistency
         re-evaluation NRF violations per arm and start; the failing sweep blocks per arm; the curtailment table.
     T2  the column `at_or_above_0.95_tau_counted`, True on EXACTLY TEN certificates (asserted). Scope: every
         certificate the tables use -- the 49 T2 cells plus the certificates of T4 (W118 year ladder), T5 (W118 Phase
         B) and T10 (A64 cells), appended to T2 as `cells_appended_w160`; superseded certificates excluded
         (pb_y2025_n5); bitwise twins counted once (the unit reference 3f084f2f = e_c2_calfade = g070_neutrality,
         counted on ref:bd504ecf). `i_5a6a88b4` flagged beside (monotone, 0.939). `d_36686489`'s uncertified cause reads
         "growth test after five TSO recoveries (not the dual dead zone; Addendum 65)" (T2 and T9).
     T8  the eps_AE 0.50 column keeps the label "superseded (0.50 era: C3, soh_min 0.50, E/P <= 10, no tight tail)".
     T11 the 3 x 3 instance figures (x = 0 optimal; R in [0.909, 0.934]) of Addenda 52 / 54, from committed records,
         with sources; R derived by the stated formula.
  3. FREEZES: writes frozen_step6_tables_v1_<sha8>.json (sha8 = the first 8 hex of the JSON's own sha256) and its
     .md into data/SRP1/Results/P515S53/w160_step6_frozen/. The frozen JSON carries no wall-clock field, so its hash
     is a function of its content; the build record (time, git HEAD, guards, wall) is a separate file. It records the
     predecessor (the W157 a64 tables JSON sha), every input sha and the package commit e3437284.
  4. EXPORTS into .../w160_step6_frozen/export/: T1-T11 as CSV and LaTeX (booktabs, plain tabular, \\caption,
     \\label{tab:Tn}), rounded per Addendum 65 (k EUR 1 dp, EUR/MWh nearest 10, multiples 2 dp) and the README rules;
     README.md (rounding table, column dictionary, main-text / supplementary split "author decides", the 3 x 3
     note); paragraphs.md (verbatim slices of P5_15_STEP6_PACKAGE.md at e3437284, read from the git object store).
  5. CHECKS every figure of paragraphs.md that has a table counterpart against the frozen JSON at the precision the
     paragraph writes it. Mismatches are FINDINGS (reported; they never change the exit code); paragraphs.md is not
     edited.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, together with every guard the W157 import arms; `pickle.load` / `pickle.loads` are blocked for the whole run (this
script's own block is re-installed after the W157 import) and every counter is verified at 0.

MODE (repo root, canonical interpreter; attached, both streams captured; outputs opened 'x', never overwritten):
    mkdir -p data/SRP1/Results/P515S53/w160_step6_frozen && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w160_step6_freeze_export.py \\
        > data/SRP1/Results/P515S53/w160_step6_frozen/launch.log 2>&1
  --out-dir REL   write elsewhere (trial runs; the default is the directory above). The output directory must hold
                  nothing but launch.log.
Exit: 0 = written, every integrity check holds, the ten-count holds and the guards are at 0; 3 = written, a check
failed (listed); 1 = harness fault or precondition (nothing written).
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W160 Step 6 freeze and export (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W160: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W160: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w157_step6_tables as W157  # noqa: E402 -- arms its own guards and blocks pickle (W145 chain)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the W157 import

GUARDS = W157._dedupe((('w160_step6_freeze_export', GUARD),) + tuple(W157.GUARDS))

TAU = W157.TAU
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_DEFAULT = os.path.join(S53, 'w160_step6_frozen')
SCRIPT_REL = os.path.basename(__file__)

# ---- the predecessor and the inputs ---------------------------------------------------------------------------------
PRED_DIR = os.path.join(S53, 'w157_step6_tables_a64')
PRED_JSON = os.path.join(PRED_DIR, 'w157_step6_tables.json')
PRED_MD = os.path.join(PRED_DIR, 'w157_step6_tables.md')
PRED_MAN = os.path.join(PRED_DIR, 'manifest_sha256.json')
PRED_SHA = 'e08a01614f6af19b6ca0e590223be270ce84b87157a473f884b244d7e5a2a636'
PRED_COMMIT = 'b913ea94'
PACKAGE_REL = 'P5_15_STEP6_PACKAGE.md'
PACKAGE_COMMIT = 'e3437284'
BRIEF_REL = 'PLANNER_BRIEF_2026-09-13.md'
BENCH_REL = os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
BENCH_MAN = os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'manifest_sha256.json')
BENCH_COMMIT = '8d42dfb8'
BENCH_SPEC_REL = os.path.join(S53, 'w116_benchmark_nrf', 'frozen_s53_benchmark_spec_v5_bca69f97.json')
BENCH_SPEC_SHA = 'bca69f97a6e60e92e3a7dc51d6bdc9f71fe463c50a840952d9d6fc78bac7d2c6'
W159_REL = os.path.join(S53, 'w159_closing_reads', 'w159_closing_reads.json')
W159_MAN = os.path.join(S53, 'w159_closing_reads', 'manifest_sha256.json')
W159_COMMIT = '62749030'
A64_SUMMARY_REL = W157.A64_SUMMARY_DEFAULT
A64_SUMMARY_MAN = os.path.join(S53, 'w155_a64_cells', 'w155_summary_after_04_e_soh050_manifest_sha256.json')
G070_RESULTS_REL = os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_g070_neutrality', 'campaign_results.json')
G070_COMMIT = '0e1d7c0e'
# the 3 x 3 instance (T11)
W91_RESULTS_REL = os.path.join(S53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'campaign_results.json')
W99_POSTHOC_REL = os.path.join(S53, 'w99_stage1_posthoc', 'w99_stage1_posthoc_analysis.json')
A51_REPORT_REL = 'P5_15_ADDENDUM51_CONTINUATION_REPORT.md'
A53_REPORT_REL = 'P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md'
T11_SOURCE_COMMITS = {'Addendum 52 (brief)': '4a80c3e2', 'Addendum 54 (brief)': '271e9325',
                      'A51 continuation report': '58ff8d88', 'A53 SRP1 continuation report': 'ebe34021',
                      '3 x 3 stage-1 evidence (replay 72/72)': '4689475e', 'W99 post-hoc analysis': '3fd1c15b'}

# ---- the Addendum 65 rulings ------------------------------------------------------------------------------------------
FLAG = 0.95
COL_COUNTED = 'at_or_above_0.95_tau_counted'
COL_TWIN = 'at_or_above_0.95_tau_twin_of'
COL_TWINS = 'at_or_above_0.95_tau_twins_counted_once'
COL_BESIDE = 'flag_beside_0.95_tau'
UNIT_REF = 'ref:bd504ecf'
TWINS_OF_UNIT = ('e_c2_calfade', 'g070_neutrality')
EXPECTED_TEN = ('d_3632b0ae', 'd_9246ed01', 'c_6597a79d', 'd_c7fee8be', 'd_c52e1670', UNIT_REF, 'b_2a0ba8b2',
                'b_4649234b', 'pb_y2030_n7', 'pb_y2025_n7')
SUPERSEDED = ('pb_y2025_n5',)
BESIDE_CELL = 'i_5a6a88b4'
BESIDE_TEXT = 'monotone, 0.939 (below 0.95 tau; flagged beside the ten, Addendum 65)'
D366 = 'd_36686489'
D366_CAUSE = 'growth test after five TSO recoveries (not the dual dead zone; Addendum 65)'
T8_EPS050_LABEL = 'superseded (0.50 era: C3, soh_min 0.50, E/P ≤ 10, no tight tail)'

OBJECTIVE_CAPTION = ('Objective convention: gross = gross_operational_cost, settlement excluded (primary); net = gross '
                     '− terminal salvage credit, beside; Q_cc = Q + t_sum, report-only.')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as handle:
        return _sha_bytes(handle.read())


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _git_blob(commit, rel):
    r = subprocess.run(['git', 'show', f'{commit}:{rel}'], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _last_commit(rel):
    return _git('log', '-1', '--format=%H', '--', rel) or None


# ======================================================================================================================
#  inputs
# ======================================================================================================================
def read_inputs():
    """The inputs W160 adds to W157's (W157's own are re-read by W157.load_inputs). Each committed clean, sha recorded,
    checked against its manifest where one exists; the package is read from the git object store at e3437284."""
    problems, rec, docs = [], {}, {}
    plan = (('PRED', PRED_JSON, PRED_MAN), ('PRED_MD', PRED_MD, PRED_MAN), ('BENCH', BENCH_REL, BENCH_MAN),
            ('BENCH_SPEC', BENCH_SPEC_REL, None), ('W159', W159_REL, W159_MAN),
            ('A64SUM', A64_SUMMARY_REL, A64_SUMMARY_MAN), ('G070', G070_RESULTS_REL, None),
            ('W91', W91_RESULTS_REL, None), ('W99', W99_POSTHOC_REL, None), ('A51', A51_REPORT_REL, None),
            ('A53', A53_REPORT_REL, None))
    for key, rel, man in plan:
        if not os.path.exists(os.path.join(REPO, rel)):
            problems.append(f'{rel} missing')
            continue
        clean = W157.L132._committed_clean(rel)
        sha = _sha(rel)
        ent = {'path': rel, 'sha256': sha, 'committed_clean': clean, 'last_commit': _last_commit(rel),
               'manifest': man, 'manifest_sha256_matches': None}
        if not clean:
            problems.append(f'{rel} is not committed clean')
        if man is not None:
            if not W157.L132._committed_clean(man):
                problems.append(f'{man} is not committed clean')
            m = json.load(open(os.path.join(REPO, man)))
            ent['manifest_sha256_matches'] = (m.get(rel) == sha)
            ent['manifest_file_sha256'] = _sha(man)
            if m.get(rel) != sha:
                problems.append(f'{rel} sha {sha[:8]} != its manifest entry {str(m.get(rel))[:8]}')
        rec[key] = ent
        if rel.endswith('.json'):
            docs[key] = json.load(open(os.path.join(REPO, rel)))
        else:
            docs[key] = open(os.path.join(REPO, rel), encoding='utf-8').read()
    if rec.get('PRED', {}).get('sha256') != PRED_SHA:
        problems.append(f'predecessor sha {rec.get("PRED", {}).get("sha256")} != {PRED_SHA}')
    if rec.get('BENCH_SPEC', {}).get('sha256') != BENCH_SPEC_SHA:
        problems.append('benchmark spec v5 sha != bca69f97')
    # the package and the brief from the git object store
    for key, rel, commit in (('PACKAGE', PACKAGE_REL, PACKAGE_COMMIT), ('BRIEF', BRIEF_REL, 'HEAD')):
        blob = _git_blob(commit, rel)
        if blob is None:
            problems.append(f'git show {commit}:{rel} failed')
            continue
        wc = open(os.path.join(REPO, rel), 'rb').read()
        rec[key] = {'path': rel, 'read_from': f'git object store (git show {commit}:{rel})', 'commit': commit,
                    'commit_full': _git('rev-parse', commit), 'sha256': _sha_bytes(blob),
                    'working_copy_equals_blob': wc == blob}
        docs[key] = blob.decode('utf-8')
    for name, c in T11_SOURCE_COMMITS.items():
        if subprocess.run(['git', 'cat-file', '-e', f'{c}^{{commit}}'], cwd=REPO).returncode != 0:
            problems.append(f'{name} commit {c} not found')
    return docs, rec, problems


# ======================================================================================================================
#  the W157 rebuild (integrity)
# ======================================================================================================================
def rebuild_w157():
    docs, inputs, problems = W157.load_inputs()
    if problems:
        return None, docs, inputs, problems
    cells = W157.cells_table(docs)
    claims, p1 = W157.claims_table(docs, cells)
    be, p2 = W157.break_even(docs, cells)
    yl, p3 = W157.year_ladder(docs)
    a64, p4, _ = W157.a64_rows(docs, True, W157.A64_SUMMARY_DEFAULT)
    tables = {'claims': claims, 'cells': cells, 'break_even': be, 'year_ladder': yl,
              'phase_b': W157.phase_b(docs, claims), 'benchmark': W157.benchmark(docs), 'discount': W157.discount(docs),
              'ageing': W157.ageing(docs), 'dead_zone': W157.dead_zone(docs, cells), 'a64': a64}
    return json.loads(GRIO.dumps(tables, sort_keys=True)), docs, inputs, p1 + p2 + p3 + p4


# ======================================================================================================================
#  T6 additions
# ======================================================================================================================
def t6_additions(bench):
    probs = []
    coord_q = bench['coordinated']['q']
    coord_band = bench['claim']['bands_eur']['coordinated_reproducibility_0.011pct']
    arms = {}
    for arm, v in bench['per_arm_nrf'].items():
        qbs = v['q_by_start']
        argmin = min(qbs, key=lambda s: qbs[s])
        if argmin != v['best_start'] or qbs[argmin] != v['q_best']:
            probs.append(f'T6 {arm}: best start {v["best_start"]} / q_best is not the argmin of q_by_start')
        runs = {s: f'nrf_arm_{arm}_{s}_r2' for s in qbs}
        cons = {s: bench['consistency_nrf'][r] for s, r in runs.items()}
        curt = {s: bench['curtailment_table']['arms_and_variants_phase_A'][r] for s, r in runs.items()}
        sweep_key = f'sweep_{arm}_cold'
        sw = bench['sweep'][sweep_key]
        failing = list(sw['failing_blocks'])
        hours = {b: h for b, h in sw['per_block_hours_not_accepted'].items() if h}
        if len(failing) != sw['n_blocks_tn_cannot_accept'] or sorted(failing) != sorted(hours) \
                or sum(len(h) for h in hours.values()) != sw['n_hours_tn_cannot_accept']:
            probs.append(f'T6 {sweep_key}: failing blocks / hours inconsistent with the recorded counts')
        arms[arm] = {
            'Q_by_start': dict(qbs), 'best_start': v['best_start'], 'Q_best': v['q_best'],
            'multimodality_band': v['multimodality_band_eur'], 'n_starts': v['n_starts'],
            'arm_cost_source_by_start': dict(v['arm_cost_source_by_start']),
            'run_by_start': runs,
            'consistency_nrf_by_start': {s: {'nrf_violations_n': c['nrf_violations']['n'],
                                             'nrf_violations_max_excess_pu': c['nrf_violations']['max_excess_pu'],
                                             'max_abs_dv_dn_pu': c['max_abs_dv_dn_pu'],
                                             'pass_effect_eur': c['pass_effect_eur'], 'trigger': c['trigger']}
                                         for s, c in cons.items()},
            'curtailment_phase_A_by_start': {s: {'eur_at_1_block_weighted': c['eur_at_1_block_weighted'],
                                                 'mwh_day_weighted': c['mwh_day_weighted'],
                                                 'by_agent': c['by_agent'], 'arm_cost_source': c['arm_cost_source']}
                                             for s, c in curt.items()},
            'sweep_unconstrained_cold': {'sweep': sweep_key, 'n_blocks_tn_cannot_accept': sw['n_blocks_tn_cannot_accept'],
                                         'n_hours_tn_cannot_accept': sw['n_hours_tn_cannot_accept'],
                                         'failing_blocks': failing, 'hours_not_accepted_by_block': hours,
                                         'statement': sw['statement']},
            'source': f'{BENCH_REL} per_arm_nrf.{arm}, consistency_nrf, curtailment_table.arms_and_variants_phase_A, '
                      f'sweep.{sweep_key} (commit {BENCH_COMMIT})'}
    dec = bench['claim']['decomposition']
    qp = bench['per_arm_nrf']['passive']['q_best']
    qt = bench['per_arm_nrf']['price_taker']['q_best']
    pband = bench['per_arm_nrf']['passive']['multimodality_band_eur']
    benefit_p = qp - coord_q
    larger = max(pband, coord_band)
    derived = {
        'label': 'derived (W160) from the recorded values of report_v3.json; NOT recorded in report_v3.json',
        'formula': 'benefit_passive = Q_passive_NRF(best of 3 starts) - Q181; relative = benefit_passive / Q181; '
                   'larger band = max(passive multimodality band, coordinated reproducibility band 0.011 %); '
                   'multiple = benefit_passive / larger band (the claim definition of report_v3 applied to the '
                   'passive arm)',
        'benefit_passive_eur': benefit_p, 'benefit_passive_relative': benefit_p / coord_q,
        'larger_band_eur': larger, 'multiple_of_larger_band': benefit_p / larger,
        'exceeds_larger_band': benefit_p > larger,
        'equals_recorded_decomposition_passive_NRF_minus_coordinated': benefit_p == dec['passive_NRF_minus_coordinated_eur'],
        'abs_difference_to_recorded_decomposition': abs(benefit_p - dec['passive_NRF_minus_coordinated_eur'])}
    if derived['abs_difference_to_recorded_decomposition'] > 1e-6:
        probs.append('T6 derived passive benefit differs from the recorded decomposition by > 1e-6 EUR')
    if (qt - coord_q) != bench['claim']['benefit_eur'] or (qp - qt) != dec['passive_NRF_minus_price_taker_NRF_eur']:
        if abs((qt - coord_q) - bench['claim']['benefit_eur']) > 1e-6 \
                or abs((qp - qt) - dec['passive_NRF_minus_price_taker_NRF_eur']) > 1e-6:
            probs.append('T6 recorded claim / decomposition not reproduced from the arm Qs')
    pt_curt = arms['price_taker']['curtailment_phase_A_by_start'][arms['price_taker']['best_start']]
    add = {
        'arms_in_full': arms,
        'decomposition_recorded': dict(dec, label='recorded (report_v3.json claim.decomposition)'),
        'price_taker_arm_claim_recorded': {'benefit_eur': bench['claim']['benefit_eur'],
                                           'benefit_relative': bench['claim']['benefit_relative'],
                                           'larger_band_eur': bench['claim']['larger_band_eur'],
                                           'determinate': bench['claim']['determinate'],
                                           'multiple_label': 'multiple = benefit / larger band, computed by W157 '
                                                             '(report_v3 records benefit, band and determinate)'},
        'passive_arm_vs_coordinated_derived': derived,
        'no_reverse_flow_definition': bench['no_reverse_flow_definition'],
        'coordinated_reverse_flow': {'statement': bench['coordinated_reverse_flow_count']['statement'],
                                     'totals_material': bench['coordinated_reverse_flow_count']['totals']['material'],
                                     'caveat': bench['claim']['reverse_flow_caveat']},
        'curtailment_table_conventions': {k: bench['curtailment_table'][k]
                                          for k in ('convention_status', 'weighting', 'signed_parts_definition')},
        'curtailment_coordinated_settled_181': bench['curtailment_table']['coordinated']['settled_cycle_181_recomputed_from_models'],
        'curtailment_passive_tie_breaker_variants': {
            k: {kk: v[kk] for kk in ('eur_at_1_block_weighted', 'mwh_day_weighted', 'by_agent')}
            for k, v in bench['curtailment_table']['arms_and_variants_phase_A'].items() if 'tie_breaker' in k},
        'price_taker_best_start_TN_curtailment_TWh_day_weighted':
            pt_curt['by_agent']['TSO']['mwh_day_weighted'] / 1e6,
        'q_min_over_starts_note': bench['claim']['q_min_over_starts_note'],
        'source': {'path': BENCH_REL, 'commit': BENCH_COMMIT, 'spec': BENCH_SPEC_REL, 'spec_sha256': BENCH_SPEC_SHA}}
    return add, probs


# ======================================================================================================================
#  T2 -- the >= 0.95 tau count (Addendum 65 ruling 2)
# ======================================================================================================================
def _spec_text(spec):
    if not isinstance(spec, dict):
        return None
    if spec.get('series') == 'frozen_s53_resettle_ext_spec':
        return f"ext v{spec.get('version')} (rule v{spec.get('criterion_version')})"
    if spec.get('series') in ('W118 r2', 'A64 v1'):
        return f"{spec['series']} (rule v{spec.get('criterion_version')})"
    return f"v{spec.get('version')}" + (f" ({spec.get('mode')})" if spec.get('mode') else '')


def appended_cells(tables, w157_docs, a64sum):
    """Certificates the tables use outside T2: T4 (W118 year ladder), T5 (W118 Phase B), T10 (A64 cells)."""
    w118 = w157_docs['W118']
    out = {}
    for cell in ('yl_y2030', 'yl_y2035', 'pb_y2025_n5', 'pb_y2025_n7', 'pb_y2025_n9', 'pb_y2030_n5', 'pb_y2030_n7',
                 'pb_y2030_n9'):
        r = w118['reports'][cell]
        out[cell] = {'cell': cell, 'row_origin': 'T4' if cell.startswith('yl_') else 'T5',
                     'item': 'year ladder' if cell.startswith('yl_') else 'Phase B', 'status': r['status'],
                     'branch': r.get('branch'), 'k0_run': r.get('k0_run'), 'k_star': r.get('k_star'),
                     'end_cycle': r.get('k_star') or r.get('k_cap'), 'band_width': r.get('band_width'),
                     'range_over_tau': r.get('range_over_tau'), 's_signed': r.get('s_signed'),
                     't_sum_k_star': r.get('t_sum_k_star'), 'replay_bitwise_through': r.get('replay_bitwise_through'),
                     'certifying_spec': {'series': 'W118 r2', 'criterion_version': 2, 'stage_spec': w118['stage_spec']},
                     'eval_key': r.get('eval_key'), 'candidate_key': r.get('candidate_key'),
                     'candidate_canonical': r.get('candidate_canonical'), 'superseded': cell in SUPERSEDED,
                     'source': f'{W157.INPUTS["W118"][0]} reports.{cell}'}
    for cell in ('h_x0_m175', 'h_unit_m175', 'e_soh050', 'g070_neutrality'):
        r = a64sum['reports'][cell]
        out[cell] = {'cell': cell, 'row_origin': 'T10', 'item': r.get('item'), 'status': r['status'],
                     'branch': r.get('branch'), 'k0_run': r.get('k0_run'), 'k_star': r.get('k_star'),
                     'end_cycle': r.get('end_cycle'), 'band_width': r.get('band_width'),
                     'range_over_tau': r.get('range_over_tau'), 's_signed': None, 't_sum_k_star': r.get('t_sum_k_star'),
                     'replay_bitwise_through': None,
                     'certifying_spec': {'series': 'A64 v1', 'criterion_version': r.get('criterion_version'),
                                         'stage_spec': a64sum['stage_spec']},
                     'eval_key': r.get('eval_key'), 'candidate_key': r.get('candidate_key'),
                     'candidate_canonical': r.get('candidate_canonical'), 'superseded': False,
                     'source': f'{A64_SUMMARY_REL} reports.{cell}'}
    # cross-check against the T10 / T5 rows the tables carry
    probs = []
    t10 = {}
    for row in tables['a64']['rows']:
        t10.update(row.get('cells') or {})
    for cell, r in t10.items():
        if (r['k_star'], r['range_over_tau'], r['eval_key']) != (out[cell]['k_star'], out[cell]['range_over_tau'],
                                                                   out[cell]['eval_key']):
            probs.append(f'appended {cell} differs from its T10 record')
    for r in tables['phase_b']:
        if r['cell'] in out and (r['k_star'], r['range_over_tau'], r['eval_key']) != (
                out[r['cell']]['k_star'], out[r['cell']]['range_over_tau'], out[r['cell']]['eval_key']):
            probs.append(f'appended {r["cell"]} differs from its T5 record')
    for y in ('2030', '2035'):
        if tables['year_ladder']['per_year'][y]['eval_key'] != out[f'yl_y{y}']['eval_key']:
            probs.append(f'appended yl_y{y} differs from its T4 record')
    return out, probs


def apply_count(tables, appended, sx, g070):
    probs = []
    cells = tables['cells']
    allc = dict(cells)
    allc.update(appended)
    # twin evidence (bitwise through 172 against the unit 3f084f2f)
    c2 = sx['c2_calfade_consistency']
    g28 = g070['gate_detail']['G28']
    twin_evidence = {
        'e_c2_calfade': {'reproduced': c2['reproduced'], 'field': c2['field'], 'through': c2['through'],
                         'first_difference': c2['first_difference'],
                         'source': f'{W157.INPUTS["SX"][0]} c2_calfade_consistency'},
        'g070_neutrality': {'reproduced': g28['reproduced'], 'field': g28['field'], 'through': g28['through'],
                            'first_difference': g28['first_difference'],
                            'gate': g070['gates']['G28_g070_bitwise_vs_3f084f2f_through_172'],
                            'source': f'{G070_RESULTS_REL} gate_detail.G28 (commit {G070_COMMIT})'}}
    twins_ok = (c2['reproduced'] is True and g28['reproduced'] is True and c2['through'] == 172
                and g28['through'] == 172 and g070['gates']['G28_g070_bitwise_vs_3f084f2f_through_172'] is True
                and all(allc[t]['range_over_tau'] == allc[UNIT_REF]['range_over_tau'] for t in TWINS_OF_UNIT))
    if not twins_ok:
        probs.append('twin evidence for the unit 3f084f2f = e_c2_calfade = g070_neutrality does not hold')
    registry = []
    for name in sorted(allc):
        r = allc[name]
        cert = r['status'] == 'certified'
        rot = r.get('range_over_tau')
        ge = bool(cert and rot is not None and rot >= FLAG)
        sup = bool(r.get('superseded')) or name in SUPERSEDED
        twin_of = UNIT_REF if name in TWINS_OF_UNIT else None
        counted = bool(ge and not sup and twin_of is None)
        r[COL_COUNTED] = counted
        r[COL_TWIN] = (f'{UNIT_REF} (the unit 3f084f2f; bitwise twin, counted once)' if twin_of else None)
        r[COL_TWINS] = list(TWINS_OF_UNIT) if name == UNIT_REF else None
        r['at_or_above_0.95_tau_note'] = (
            'counted' if counted else
            'superseded (excluded)' if (ge and sup) else
            'bitwise twin of the unit (counted once, on ref:bd504ecf)' if (ge and twin_of) else
            'not a certificate' if not cert else 'below 0.95 tau')
        if name == BESIDE_CELL:
            if not (r.get('branch') == 'monotone' and round(rot, 3) == 0.939):
                probs.append(f'{BESIDE_CELL} is not monotone at 0.939')
            r[COL_BESIDE] = BESIDE_TEXT
        else:
            r.setdefault(COL_BESIDE, None)
        if cert:
            registry.append({'cell': name, 'table': 'T2' if name in cells else r['row_origin'],
                             'range_over_tau': rot, 'ge_0_95': ge, 'superseded': sup, 'twin_of': twin_of,
                             'counted': counted, 'eval_key': r.get('eval_key')})
    counted = [x['cell'] for x in registry if x['counted']]
    return {'scope': ('every certificate the tables use: the 49 T2 cells (certified ones) plus the certificates of T4 '
                      '(W118 year ladder), T5 (W118 Phase B) and T10 (A64 cells), appended to T2; superseded '
                      'certificates excluded; bitwise twins counted once (Addendum 65 ruling 2)'),
            'threshold': FLAG, 'counted_cells': counted, 'n_counted': len(counted),
            'expected_ten_addendum65': list(EXPECTED_TEN),
            'counted_equals_expected': sorted(counted) == sorted(EXPECTED_TEN),
            'twins': {UNIT_REF: list(TWINS_OF_UNIT), 'evidence': twin_evidence, 'holds': twins_ok},
            'beside': {BESIDE_CELL: BESIDE_TEXT}, 'registry': registry}, probs


# ======================================================================================================================
#  T11 -- the 3 x 3 instance (Addenda 52 / 54)
# ======================================================================================================================
def t11(docs, tables):
    vr = docs['W91']['value_and_R']
    post = docs['W99']['item3_POSTHOC_damped_oscillation']['R_under_posthoc_limit']
    v_srp1 = next(r['V'] for r in tables['discount'] if r['rate'] == 0.02)
    chk_v = next(r for r in tables['claims'] if r['claim_id'] == 'CHECK:headline_V_minus_I_settled')
    V, D = vr['value_eur'], post['posthoc_primary_L']['D']
    r_hi, r_lo = V / v_srp1, (V - D) / v_srp1
    w91c = _last_commit(W91_RESULTS_REL)
    w99c = _last_commit(W99_POSTHOC_REL)
    rows = [
        {'quantity': '3 x 3 storage value V = Q(0) - Q(unit), at certification (cycle 72)', 'value': V, 'unit': 'EUR',
         'kind': 'keur', 'label': 'recorded', 'source': f'{W91_RESULTS_REL} value_and_R.value_eur ({w91c[:8]})'},
        {'quantity': '3 x 3 investment I (unit)', 'value': vr['I_eur'], 'unit': 'EUR', 'kind': 'keur',
         'label': 'recorded', 'source': f'{W91_RESULTS_REL} value_and_R.I_eur ({w91c[:8]})'},
        {'quantity': '3 x 3 value - I (x = 0 optimal under the baseline)', 'value': vr['value_minus_I'], 'unit': 'EUR',
         'kind': 'keur', 'label': f'recorded; determinate {vr["value_minus_I_determinate"]}',
         'source': f'{W91_RESULTS_REL} value_and_R.value_minus_I ({w91c[:8]}); Addendum 52 (brief '
                   f'{T11_SOURCE_COMMITS["Addendum 52 (brief)"]}); {A51_REPORT_REL} '
                   f'({T11_SOURCE_COMMITS["A51 continuation report"]}) line 60'},
        {'quantity': '3 x 3 resolution of the value', 'value': vr['resolution'], 'unit': 'EUR', 'kind': 'keur',
         'label': 'recorded', 'source': f'{W91_RESULTS_REL} value_and_R.resolution ({w91c[:8]})'},
        {'quantity': '3 x 3 |value - I| / resolution', 'value': abs(vr['value_minus_I']) / vr['resolution'],
         'unit': 'x', 'kind': 'x', 'label': 'derived (W160): |value_minus_I| / resolution ("4.5x" in Addendum 52)',
         'source': f'{W91_RESULTS_REL} value_and_R'},
        {'quantity': 'post-hoc settled descent D of the 3 x 3 x = 0 cell after certification', 'value': D,
         'unit': 'EUR', 'kind': 'keur', 'label': 'recorded (post hoc, damped-cosine fit; W99)',
         'source': f'{W99_POSTHOC_REL} item3_POSTHOC_damped_oscillation.R_under_posthoc_limit.posthoc_primary_L.D '
                   f'({w99c[:8]})'},
        {'quantity': 'V_SRP1 settled (the reference of R)', 'value': v_srp1, 'unit': 'EUR', 'kind': 'keur',
         'label': 'recorded (T7 row 2 %; = T1 CHECK headline V)', 'source': 'T7 (W153 discount row)'},
        {'quantity': 'R at 3 x 3 certification against the settled V_SRP1 (upper end)', 'value': r_hi, 'unit': '-',
         'kind': 'ratio3', 'label': 'derived (W160): V / V_SRP1_settled',
         'source': f'Addendum 54 (brief {T11_SOURCE_COMMITS["Addendum 54 (brief)"]}); {A53_REPORT_REL} '
                   f'({T11_SOURCE_COMMITS["A53 SRP1 continuation report"]}) lines 118-119 ("0.9336")'},
        {'quantity': 'R with the 3 x 3 x = 0 post-hoc settled descent (lower end)', 'value': r_lo, 'unit': '-',
         'kind': 'ratio3', 'label': 'derived (W160): (V - D) / V_SRP1_settled (W99 formula with R_ref = V_SRP1 settled)',
         'source': f'Addendum 54; {A53_REPORT_REL} lines 118-119 ("0.9090")'},
        {'quantity': 'R predicted from the mean-profile spread', 'value': vr['R_prefix_recorded'], 'unit': '-',
         'kind': 'ratio4', 'label': 'recorded', 'source': f'{W91_RESULTS_REL} value_and_R.R_prefix_recorded'},
        {'quantity': '3 x 3 x = 0 replay bitwise through certification', 'value': None, 'text': '72 / 72 cycles',
         'unit': '-', 'kind': 'txt', 'label': 'recorded (stated in Addendum 52; not re-read here)',
         'source': f'Addendum 52; stage-1 evidence {T11_SOURCE_COMMITS["3 x 3 stage-1 evidence (replay 72/72)"]}; '
                   f'{A51_REPORT_REL} ({T11_SOURCE_COMMITS["A51 continuation report"]})'}]
    probs = []
    if (round(r_lo, 3), round(r_hi, 3)) != (0.909, 0.934):
        probs.append(f'T11 R range [{r_lo}, {r_hi}] does not round to [0.909, 0.934]')
    if v_srp1 != chk_v['I_other'] + chk_v['d_gross']:
        if abs(v_srp1 - (chk_v['I_other'] + chk_v['d_gross'])) > 1e-6:
            probs.append('T11 V_SRP1 from T7 differs from the T1 headline V')
    if vr['value_minus_I_determinate'] is not True:
        probs.append('T11 3 x 3 value - I not recorded determinate')
    if 'R ∈ [0.909, 0.934]' not in docs['A53'] or 'value − I = −81,262 €' not in docs['A51'] \
            or 'R ∈ [0.909, 0.934]' not in docs['BRIEF']:
        probs.append('T11 source sentences not found in the A51 / A53 reports or the brief')
    return {'rows': rows, 'R_range_derived': [r_lo, r_hi], 'R_range_rounded': [round(r_lo, 3), round(r_hi, 3)],
            'formula_R': '[(V - D) / V_SRP1_settled, V / V_SRP1_settled]; V = 3 x 3 value at certification, D = the '
                         'post-hoc settled descent of the 3 x 3 x = 0 cell (W99), V_SRP1_settled = 253,539.62 EUR',
            'note': ('the 3 x 3 figures are not in T1-T10; they come from Addenda 52 / 54 and the committed records '
                     'named in each row'),
            'source_commits': T11_SOURCE_COMMITS}, probs


# ======================================================================================================================
#  rounding and formatting (the JSON keeps full precision)
# ======================================================================================================================
ROUNDING = [
    ('EUR amounts (Q, value, I, differences, bands, thresholds, bars, gaps, slacks, t_sum, salvage)',
     'k EUR, 1 dp', 'Addendum 65'),
    ('EUR/MWh (break-even, margin, slope, energy cost)', 'nearest 10 EUR/MWh', 'Addendum 65'),
    ('multiples (x): abs(margin) / threshold or bar', '2 dp', 'Addendum 65'),
    ('EUR/MVA (power cost p_cost; fit coefficient c)', 'nearest 10 EUR/MVA', 'W160 choice (unit price, as EUR/MWh)'),
    ('percentages (benefit relative, node shares)', '1 dp', 'W160 choice'),
    ('percentages of slope agreement (T3)', '2 dp', 'W160 choice (scored against a 5 % criterion; the package quotes 2 dp)'),
    ('EFC/day (incl. the PV-weighted EFC of T8)', '2 dp', 'W160 choice'),
    ('SoH, available-energy fraction AE', '3 dp', 'W160 choice'),
    ('range / tau', '2 dp', 'W160 choice'),
    ('elasticity eps_AE', '2 dp', 'W160 choice'),
    ('dimensionless ratios (R, pf_primal ratio)', '3 dp (R predicted 0.9331 at 4 dp, as recorded)', 'W160 choice'),
    ('cycles (k0, k*, end, replay, turning points), counts, hours, blocks', 'integer', 'W160 choice'),
    ('energy (reverse-flow MWh)', '1 dp MWh', 'W160 choice'),
    ('curtailment energy (day-weighted)', 'GWh, 1 dp', 'W160 choice'),
    ('p.u. quantities (NRF excess)', '3 significant figures', 'W160 choice'),
    ('discount rate', '%, 0 dp', 'W160 choice'),
    ('ageing calibration k', 'integer', 'W160 choice'),
]


def _num(v, dp, style, sep=True):
    s = f'{v:,.{dp}f}' if (sep and style != 'csv') else f'{v:.{dp}f}'
    if s.startswith('-'):
        # a value that rounds to zero is written without a sign
        if float(s.replace(',', '')) == 0:
            s = s[1:]
        elif style != 'csv':
            s = '−' + s[1:]
    return s


def fmt(v, kind, style='csv'):
    """style: 'csv' (plain numbers, ASCII minus) or 'tex' / 'md' (thousands separator, unicode minus mapped later)."""
    if kind == 'txt':
        return '' if v is None else str(v)
    if v is None:
        return '' if style == 'csv' else '—'
    if kind == 'bool':
        return 'yes' if v else 'no'
    if kind == 'keur':
        return _num(v / 1000.0, 1, style)
    if kind in ('eur_mwh', 'eur_mva'):
        return _num(round(v / 10.0) * 10, 0, style)
    if kind == 'x':
        return _num(v, 2, style)
    if kind == 'pct':
        return _num(100.0 * v, 1, style)
    if kind == 'pct2':
        return _num(100.0 * v, 2, style)
    if kind in ('efc', 'rot', 'eps'):
        return _num(v, 2, style)
    if kind in ('soh', 'ratio3'):
        return _num(v, 3, style)
    if kind == 'ratio4':
        return _num(v, 4, style)
    if kind == 'int':
        return _num(v, 0, style, sep=False) if isinstance(v, (int, float)) else str(v)
    if kind == 'mwh':
        return _num(v, 1, style)
    if kind == 'gwh':
        return _num(v / 1000.0, 1, style)
    if kind == 'pu':
        return f'{v:.3g}'
    if kind == 'rate':
        return _num(100.0 * v, 0, style)
    raise ValueError(kind)


TEX_MAP = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
           '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '<': r'$<$', '>': r'$>$', '|': r'$|$',
           'τ': r'$\tau$', '×': r'$\times$', '≥': r'$\geq$', '≤': r'$\leq$', '−': r'$-$', '€': r'\texteuro{}',
           'ε': r'$\varepsilon$', 'λ': r'$\lambda$', 'π': r'$\pi$', '–': '--', '—': '---', '·': r'$\cdot$',
           '…': r'\ldots{}', '≈': r'$\approx$', '∈': r'$\in$', '→': r'$\rightarrow$', '₀': r'$_0$', 'Δ': r'$\Delta$',
           '±': r'$\pm$', '’': "'", '“': '``', '”': "''", '≠': r'$\neq$', '\u00a0': '~'}


def tex(s):
    return ''.join(TEX_MAP.get(ch, ch) for ch in str(s))


# ======================================================================================================================
#  export tables: (name, caption, columns, rows); a column is (header, getter, kind, align, in_latex)
# ======================================================================================================================
def _status_text(r):
    if r is None:
        return ''
    if r.get('status') == 'certified':
        s = f"cert. {r.get('branch') or ''} k*{r.get('k_star')}".replace('  ', ' ')
        if r.get('range_over_tau') is not None:
            s += f" r/τ {r['range_over_tau']:.2f}"
        return s
    return f"uncert. ({r.get('cause_uncertified') or 'uncertified'}) end {r.get('end_cycle')}"


def build_exports(fz):
    t = fz['tables']
    cells = t['cells']
    allc = dict(cells)
    allc.update(t['cells_appended_w160'])
    claims = {r['claim_id']: r for r in t['claims']}
    ex = []

    # ---- T1 ----
    def unc(r, k):
        u = r.get('uncertified_form_beside_A64_ruling4')
        return None if u is None else u.get(k)
    cols = [('claim', lambda r: r['claim_id'], 'txt', 'l', True),
            ('statement', lambda r: r['statement'], 'txt', 'l', False),
            ('ref cell', lambda r: r['ref_cell'], 'txt', 'l', True),
            ('ref status', lambda r: _status_text(allc.get(r['ref_cell'])), 'txt', 'l', False),
            ('other cell', lambda r: r['other_cell'], 'txt', 'l', True),
            ('other status', lambda r: _status_text(allc.get(r['other_cell'])), 'txt', 'l', False),
            ('d gross (k€)', lambda r: r['d_gross'], 'keur', 'r', True),
            ('rule', lambda r: r['gross_rule'], 'txt', 'l', True),
            ('threshold / bar (k€)', lambda r: r['gross_threshold_or_bar'], 'keur', 'r', True),
            ('× gross', lambda r: r['gross_multiple'], 'x', 'r', True),
            ('verdict (gross)', lambda r: r['gross_verdict'], 'txt', 'l', True),
            ('d net (k€)', lambda r: r['d_net'], 'keur', 'r', True),
            ('× net', lambda r: r['net_multiple'], 'x', 'r', True),
            ('verdict (net)', lambda r: r['net_verdict_W153'], 'txt', 'l', True),
            ('net label', lambda r: 'recorded' if r['net_label'] == W157.NET_LABEL_RECORDED else
             'validated by form + salvage identity', 'txt', 'l', False),
            ('d Q_cc (k€, report-only)', lambda r: r['d_Qcc_report_only'], 'keur', 'r', True),
            ('× Q_cc (report-only)', lambda r: r['Qcc_multiple_report_only'], 'x', 'r', True),
            ('uncertified form beside: bar (k€)', lambda r: unc(r, 'bar'), 'keur', 'r', True),
            ('uncertified form beside: ×', lambda r: unc(r, 'multiple_Q'), 'x', 'r', True),
            ('uncertified form beside: verdict', lambda r: unc(r, 'verdict'), 'txt', 'l', True),
            ('notes', lambda r: ' / '.join(r['notes']), 'txt', 'l', False)]
    ex.append(('T1', 'Pairwise claims (60): difference, determinacy threshold or uncertified bar, multiple and verdict; '
               'net of salvage beside (labelled); Q_cc report-only. Determinacy: max(3 × larger band, 2τ) between '
               'certified cells; 3·max(|gap|, |slack|) with an uncertified cell; τ = 4,539.07 €.', cols, t['claims']))

    # ---- T2 ----
    t2rows = [dict(r, _origin='T2') for _, r in sorted(cells.items())] + \
             [dict(r, _origin=r['row_origin']) for _, r in sorted(t['cells_appended_w160'].items())]
    cols = [('cell', lambda r: r['cell'], 'txt', 'l', True),
            ('table', lambda r: r['_origin'], 'txt', 'l', True),
            ('item', lambda r: r.get('item'), 'txt', 'l', False),
            ('status', lambda r: r['status'], 'txt', 'l', True),
            ('branch', lambda r: r.get('branch'), 'txt', 'l', True),
            ('k0', lambda r: r.get('k0_run'), 'int', 'r', True),
            ('k* / end', lambda r: r.get('k_star') or r.get('end_cycle'), 'int', 'r', True),
            ('range/τ', lambda r: r.get('range_over_tau') if r['status'] == 'certified' else None, 'rot', 'r', True),
            ('≥ 0.95 τ (range)', lambda r: (r.get('range_over_tau') is not None and r['status'] == 'certified'
                                             and r['range_over_tau'] >= FLAG), 'bool', 'l', True),
            (COL_COUNTED, lambda r: r[COL_COUNTED], 'bool', 'l', True),
            ('≥ 0.95 τ note', lambda r: r['at_or_above_0.95_tau_note'], 'txt', 'l', True),
            ('flag beside', lambda r: r.get(COL_BESIDE), 'txt', 'l', True),
            ('NCTP cycles', lambda r: None if r.get('turning_points_at_non_clean_cycle') is None else
             ','.join(str(x) for x in r['turning_points_at_non_clean_cycle']), 'txt', 'l', True),
            ('cause (uncertified)', lambda r: r.get('cause_uncertified'), 'txt', 'l', True),
            ('gap (k€)', lambda r: r.get('gap'), 'keur', 'r', True),
            ('slack (k€)', lambda r: r.get('slack'), 'keur', 'r', True),
            ('band (k€)', lambda r: r.get('band_width'), 'keur', 'r', True),
            ('non-clean cycles after N', lambda r: None if r.get('non_clean_after_N') is None else
             len(r['non_clean_after_N']), 'int', 'r', True),
            ('replay bitwise through', lambda r: r.get('replay_bitwise_through'), 'int', 'r', True),
            ('certifying spec', lambda r: _spec_text(r.get('certifying_spec')), 'txt', 'l', True),
            ('eval key', lambda r: (r.get('eval_key') or '')[:16] or None, 'txt', 'l', True),
            ('candidate key', lambda r: (r.get('candidate_key') or '')[:8] or None, 'txt', 'l', False)]
    ex.append(('T2', 'Certification status of every cell the tables use: the 49 claim cells and the references, then '
               'the certificates of T4, T5 and T10 appended (column "table"). at_or_above_0.95_tau_counted is true '
               'on exactly ten certificates: superseded certificates excluded, bitwise twins (the unit 3f084f2f = '
               'e_c2_calfade = g070_neutrality) counted once (Addendum 65).', cols, t2rows))

    # ---- T3 ----
    be = t['break_even']
    t3rows = [dict(be['committed_W145'], _fit='committed W145 (3 intervals)'),
              dict(be['conservative_A64'], _fit='conservative A64 (4 intervals, + d_4a82a64a) -- manuscript figure')]
    cols = [('fit', lambda r: r['_fit'], 'txt', 'l', True),
            ('n certified', lambda r: r['certified_only']['n'], 'int', 'r', True),
            ('intervals', lambda r: ', '.join(r['uncertified_as_intervals']), 'txt', 'l', True),
            ('certified-only e* (€/MWh)', lambda r: r['certified_only']['breakeven_marginal_4h_energy_cost'],
             'eur_mwh', 'r', True),
            ('banded midpoint e* (€/MWh)', lambda r: r['banded_mid']['breakeven_marginal_4h_energy_cost'],
             'eur_mwh', 'r', True),
            ('e* min (€/MWh)', lambda r: r['breakeven_range'][0], 'eur_mwh', 'r', True),
            ('e* max (€/MWh)', lambda r: r['breakeven_range'][1], 'eur_mwh', 'r', True),
            ('margin min (€/MWh)', lambda r: r['margin_to_cost_range'][0], 'eur_mwh', 'r', True),
            ('margin max (€/MWh)', lambda r: r['margin_to_cost_range'][1], 'eur_mwh', 'r', True),
            ('slope b + c/4 min (€/MWh)', lambda r: r['slope_b_plus_c_over_4_range'][0], 'eur_mwh', 'r', True),
            ('slope b + c/4 max (€/MWh)', lambda r: r['slope_b_plus_c_over_4_range'][1], 'eur_mwh', 'r', True),
            ('b + c/4 agreement at midpoint (%)', lambda r: r['slope_b_plus_c_over_4_rel_mid'], 'pct2', 'r', True),
            ('b + c/4 agreement at corners (%)', lambda r: '{} / {}'.format(
                fmt(r['slope_b_plus_c_over_4_rel_corners'][0], 'pct2'), fmt(r['slope_b_plus_c_over_4_rel_corners'][1],
                                                                            'pct2')), 'txt', 'l', True),
            ('b agreement at midpoint (%)', lambda r: r['slope_b_rel_mid'], 'pct2', 'r', True),
            ('b agreement at corners (%)', lambda r: '{} / {}'.format(
                fmt(r['slope_b_rel_corners'][0], 'pct2'), fmt(r['slope_b_rel_corners'][1], 'pct2')), 'txt', 'l', True)]
    ex.append(('T3', f"Break-even fit, node 7 (slope b + c/4; Addendum 64 rulings 3-4). Energy cost "
               f"{fmt(be['e_cost_eur_per_MWh'], 'eur_mwh', 'tex')} €/MWh; power cost "
               f"{fmt(be['p_cost_eur_per_MVA'], 'eur_mva', 'tex')} €/MVA. Manuscript figure: break-even ≤ "
               f"{fmt(be['manuscript_figure']['breakeven_max_eur_per_mwh'], 'eur_mwh', 'tex')} €/MWh; margin ≥ "
               f"{fmt(be['manuscript_figure']['margin_min_eur_per_mwh'], 'eur_mwh', 'tex')} €/MWh.", cols, t3rows))

    # ---- T4 ----
    yl = t['year_ladder']
    t4rows = []
    for yr, v in yl['per_year'].items():
        t4rows.append({'row': yr, 'Mg': v['M_gross_w153_form'], 'sal': v['salvage_last'], 'Mn': v['M_net_w153_form'],
                       'k': v['k_star'], 'ek': v['eval_key'][:16]})
    t4rows.append({'row': '2035 − 2030', 'Mg': yl['D_gross'], 'Mn': yl['D_net'],
                   'gt': yl['gross_v6']['threshold'], 'gx': yl['gross_v6']['multiple'], 'gv': yl['gross_v6']['verdict'],
                   'nt': yl['net_v6']['threshold'], 'nx': yl['net_v6']['multiple'], 'nv': yl['net_v6']['verdict'],
                   'lab': 'validated by form + salvage identity (W154b)'})
    cols = [('row', lambda r: r['row'], 'txt', 'l', True),
            ('M gross (k€)', lambda r: r.get('Mg'), 'keur', 'r', True),
            ('salvage (k€)', lambda r: r.get('sal'), 'keur', 'r', True),
            ('M net (k€)', lambda r: r.get('Mn'), 'keur', 'r', True),
            ('k*', lambda r: r.get('k'), 'int', 'r', True),
            ('threshold gross (k€)', lambda r: r.get('gt'), 'keur', 'r', True),
            ('× gross', lambda r: r.get('gx'), 'x', 'r', True),
            ('verdict gross', lambda r: r.get('gv'), 'txt', 'l', True),
            ('threshold net (k€)', lambda r: r.get('nt'), 'keur', 'r', True),
            ('× net', lambda r: r.get('nx'), 'x', 'r', True),
            ('verdict net', lambda r: r.get('nv'), 'txt', 'l', True),
            ('net label', lambda r: r.get('lab'), 'txt', 'l', False),
            ('eval key', lambda r: r.get('ek'), 'txt', 'l', False)]
    ex.append(('T4', 'Year ladder, 2035 − 2030 (M = I + Q − Q181, W118 form); gross primary, net beside. The '
               'investment-year comparison in this instance is decided by the salvage convention, not by operation.',
               cols, t4rows))

    # ---- T5 ----
    cols = [('cell', lambda r: r['cell'], 'txt', 'l', True),
            ('status', lambda r: r['status'], 'txt', 'l', True),
            ('k0', lambda r: allc.get(r['cell'], {}).get('k0_run'), 'int', 'r', True),
            ('k*', lambda r: r.get('k_star') or allc.get(r['cell'], {}).get('k_star'), 'int', 'r', True),
            ('range/τ', lambda r: allc.get(r['cell'], {}).get('range_over_tau'), 'rot', 'r', True),
            (COL_COUNTED, lambda r: allc.get(r['cell'], {}).get(COL_COUNTED), 'bool', 'l', True),
            ('M gross (k€)', lambda r: r['M_gross'], 'keur', 'r', True),
            ('M Q_cc (k€, report-only)', lambda r: r['M_cc_report_only'], 'keur', 'r', True),
            ('v6 threshold (k€)', lambda r: r['v6_threshold'], 'keur', 'r', True),
            ('×', lambda r: r['v6_multiple'], 'x', 'r', True),
            ('v6 verdict', lambda r: r['v6_verdict'], 'txt', 'l', True),
            ('recorded (W118 rule)', lambda r: (r.get('recorded_W118_rule') or {}).get('verdict'), 'txt', 'l', True),
            ('superseded', lambda r: bool(r.get('superseded')), 'bool', 'l', True),
            ('note', lambda r: r['note'], 'txt', 'l', False)]
    ex.append(('T5', 'Phase B certificates against x = 0 (gross; v6 determinacy rule applied by DET.resolve_v6). '
               'pb_y2025_n5 is superseded by its v6 re-run pb_y2025_n5_v6.', cols, t['phase_b']))

    # ---- T6 ----
    b = t['benchmark']
    a = b['w160_additions']
    q181 = b['coordinated_Q']
    t6rows = [{'arr': 'coordinated (settled x = 0, Q181)', 'start': None, 'Q': q181,
               'band': b['coordinated_reproducibility_band'], 'lab': 'recorded'}]
    for arm, name in (('passive', 'passive NRF'), ('price_taker', 'price-taker NRF')):
        ar = a['arms_in_full'][arm]
        sw = ar['sweep_unconstrained_cold']
        if arm == 'passive':
            d = a['passive_arm_vs_coordinated_derived']
            ben, rel, mul, lab = d['benefit_passive_eur'], d['benefit_passive_relative'], d['multiple_of_larger_band'], \
                'derived (W160)'
        else:
            ben, rel, mul, lab = b['benefit'], b['benefit_relative'], b['multiple'], 'recorded (× computed by W157)'
        t6rows.append({'arr': name, 'start': f"best of 3: {ar['best_start']}", 'Q': ar['Q_best'],
                       'band': ar['multimodality_band'], 'ben': ben, 'rel': rel, 'mul': mul, 'lab': lab,
                       'swb': sw['n_blocks_tn_cannot_accept'], 'swh': sw['n_hours_tn_cannot_accept'],
                       'swf': '; '.join(x.replace('TSO|-|', '').replace('|', ' ') for x in sw['failing_blocks'])})
        for s in ('cold', 'perturbed', 'warm_from_certified'):
            c = ar['consistency_nrf_by_start'][s]
            cu = ar['curtailment_phase_A_by_start'][s]
            t6rows.append({'arr': name, 'start': s, 'Q': ar['Q_by_start'][s], 'lab': 'recorded',
                           'nv': c['nrf_violations_n'], 'nx': c['nrf_violations_max_excess_pu'],
                           'pe': c['pass_effect_eur'], 'ct': cu['by_agent']['TSO']['mwh_day_weighted'],
                           'cd': cu['by_agent']['DSO']['mwh_day_weighted']})
    t6rows.append({'arr': 'passive NRF − price-taker NRF', 'start': 'best − best',
                   'ben': a['decomposition_recorded']['passive_NRF_minus_price_taker_NRF_eur'], 'lab': 'recorded'})
    cols = [('arrangement', lambda r: r['arr'], 'txt', 'l', True),
            ('start', lambda r: r['start'], 'txt', 'l', True),
            ('Q gross (k€)', lambda r: r.get('Q'), 'keur', 'r', True),
            ('band (k€)', lambda r: r.get('band'), 'keur', 'r', True),
            ('Q − Q181 (k€)', lambda r: r.get('ben'), 'keur', 'r', True),
            ('% of Q181', lambda r: r.get('rel'), 'pct', 'r', True),
            ('× larger band', lambda r: r.get('mul'), 'x', 'r', True),
            ('figure label', lambda r: r.get('lab'), 'txt', 'l', True),
            ('NRF violations (n)', lambda r: r.get('nv'), 'int', 'r', True),
            ('NRF max excess (p.u.)', lambda r: r.get('nx'), 'pu', 'r', True),
            ('consistency pass effect (k€)', lambda r: r.get('pe'), 'keur', 'r', True),
            ('TN curtailment (GWh, day-weighted)', lambda r: r.get('ct'), 'gwh', 'r', True),
            ('DN curtailment (GWh, day-weighted)', lambda r: r.get('cd'), 'gwh', 'r', True),
            ('sweep: blocks TN cannot accept (of 12)', lambda r: r.get('swb'), 'int', 'r', True),
            ('sweep: hours', lambda r: r.get('swh'), 'int', 'r', True),
            ('sweep: failing blocks', lambda r: r.get('swf'), 'txt', 'l', True)]
    rf = a['coordinated_reverse_flow']['totals_material']
    ex.append(('T6', 'Uncoordinated benchmark (Addendum 57; spec v5 bca69f97; report_v3 8d42dfb8), both '
               'no-reverse-flow (NRF) arms in full. NRF: pg_adn[s_m, s_o, p] ≥ 0 at every DSO interface, scenario and '
               'period (no export from a DN to the TN). Benefit = min(Q_passive, Q_price-taker) − Q181; each arm is '
               'the minimum over three starts. The coordinated solution has '
               f"{rf['count']} reverse-flow interface-hours ({fmt(rf['energy_mwh_block_weighted'], 'mwh', 'tex')} MWh, "
               'block-weighted). Sweep: the unconstrained arm (no interface rule), cold start.', cols, t6rows))

    # ---- T7 ----
    cols = [('rate (%)', lambda r: r['rate'], 'rate', 'r', True),
            ('V (k€)', lambda r: r['V'], 'keur', 'r', True),
            ('I (k€)', lambda r: r['I'], 'keur', 'r', True),
            ('value − I (k€)', lambda r: r['value_minus_I'], 'keur', 'r', True),
            ('threshold (k€)', lambda r: r['threshold_a61_conservative'], 'keur', 'r', True),
            ('×', lambda r: r['multiple'], 'x', 'r', True),
            ('verdict', lambda r: r['verdict'], 'txt', 'l', True)]
    ex.append(('T7', 'Discount rate (fixed plan: unit at node 7, 0.25 MVA / 1 MWh); one discount factor per '
               'representative year applied to the five years of its block; I paid in 2025.', cols, t['discount']))

    # ---- T8 ----
    e_claim = {'C2': 'E:n7_4h_e1_C2:value_minus_I', 'C2_calfade': 'E:n7_4h_e1_C2_calfade:value_minus_I',
               'C3_unit': 'E:C3_unit_value_minus_I', 'C4': 'E:n7_4h_e1_C4:value_minus_I',
               'C3_midblock': 'E:n7_4h_e1_C3_midblock:value_minus_I', 'no_ageing': 'E:n7_4h_e1_no_ageing:value_minus_I'}
    lab050 = t['ageing']['column_labels']['eps_AE_050_superseded']
    cols = [('arm', lambda r: r['arm'], 'txt', 'l', True),
            ('cell', lambda r: r['cell'], 'txt', 'l', True),
            ('value (k€)', lambda r: r['value'], 'keur', 'r', True),
            ('I (k€)', lambda r: r['I'], 'keur', 'r', True),
            ('value − I (k€)', lambda r: r['value_minus_I'], 'keur', 'r', True),
            ('threshold (k€, from T1)', lambda r: claims[e_claim[r['arm']]]['gross_threshold_or_bar'], 'keur', 'r', True),
            ('× (from T1)', lambda r: claims[e_claim[r['arm']]]['gross_multiple'], 'x', 'r', True),
            ('verdict', lambda r: r['verdict'], 'txt', 'l', True),
            ('floor binds (0.70)', lambda r: r['floor_year_070'] or 'never', 'txt', 'l', True),
            ('AE (PV-weighted)', lambda r: r['AE'], 'soh', 'r', True),
            ('EFC/day (PV-weighted)', lambda r: r['EFC'], 'efc', 'r', True),
            ('k', lambda r: r['k'], 'int', 'r', True),
            ('ε_AE (0.70)', lambda r: r['eps_AE_070'], 'eps', 'r', True),
            ('ε_AE resolvable (0.70)', lambda r: r['eps_AE_resolvable_070'], 'bool', 'l', True),
            (f'ε_AE (0.50) -- {lab050}', lambda r: r['eps_AE_050_superseded'], 'eps', 'r', True)]
    ex.append(('T8', 'Ageing arms at minimum SoH 0.70, fixed plan (unit at node 7, 0.25 MVA / 1 MWh), gross. The '
               'ε_AE (0.50) column is the superseded 0.50-era comparator (Addendum 65 read (a)).', cols,
               t['ageing']['rows']))

    # ---- T9 ----
    def share(r, n):
        sh = r.get('share_by_node')
        return None if not sh else sh.get(n)
    cols = [('cell', lambda r: r['cell'], 'txt', 'l', True),
            ('status', lambda r: r['status'], 'txt', 'l', True),
            ('cause', lambda r: r.get('cause'), 'txt', 'l', True),
            ('t_sum at cap (k€)', lambda r: r.get('t_sum_at_cap'), 'keur', 'r', True),
            ('share n5 (%)', lambda r: share(r, '5'), 'pct', 'r', True),
            ('share n7 (%)', lambda r: share(r, '7'), 'pct', 'r', True),
            ('share n9 (%)', lambda r: share(r, '9'), 'pct', 'r', True),
            ('pf_primal last', lambda r: r.get('pf_primal_last'), 'ratio3', 'r', True),
            ('lapse resets at', lambda r: ','.join(str(e['cycle']) for e in (r.get('lapse_events') or [])) or None,
             'txt', 'l', True),
            ('entry', lambda r: r.get('entry'), 'txt', 'l', False)]
    ex.append(('T9', 'Dual dead zone (Addendum 58 Ruling 1; Addendum 62; Addendum 64 ruling 7): the gap-refused and '
               'related uncertified cells, with the priced consensus gap at the cap and its node split.', cols,
               t['dead_zone']['cells']))

    # ---- T10 ----
    a6 = t['a64']
    sb = a6['scored']['B']

    def c10(r, which, k):
        # a cell outside the W155 summary (the settled unit reference bd504ecf) is read from T2
        c = (r.get('cells') or {}).get(r[which]) or allc.get(f'ref:{r[which]}') or {}
        return c.get(k)

    def k0k(r, which):
        k0, ks = c10(r, which, 'k0_run'), c10(r, which, 'k_star')
        return None if ks is None else f"{'—' if k0 is None else k0} / {ks}"
    cols = [('claim', lambda r: r['claim_id'], 'txt', 'l', True),
            ('statement', lambda r: r['statement'], 'txt', 'l', False),
            ('ref', lambda r: r['ref_cell'], 'txt', 'l', True),
            ('ref k0 / k*', lambda r: k0k(r, 'ref_cell'), 'txt', 'l', True),
            ('ref range/τ', lambda r: c10(r, 'ref_cell', 'range_over_tau'), 'rot', 'r', True),
            ('other', lambda r: r['other_cell'], 'txt', 'l', True),
            ('other k0 / k*', lambda r: k0k(r, 'other_cell'), 'txt', 'l', True),
            ('other range/τ', lambda r: c10(r, 'other_cell', 'range_over_tau'), 'rot', 'r', True),
            ('d gross (k€)', lambda r: r['d_gross'], 'keur', 'r', True),
            ('rule', lambda r: r['gross_rule'], 'txt', 'l', True),
            ('threshold (k€)', lambda r: r['gross_threshold_or_bar'], 'keur', 'r', True),
            ('×', lambda r: r['gross_multiple'], 'x', 'r', True),
            ('verdict', lambda r: r['gross_verdict'], 'txt', 'l', True),
            ('d Q_cc (k€, report-only)', lambda r: r['d_Qcc_report_only'], 'keur', 'r', True),
            ('× Q_cc (report-only)', lambda r: r['Qcc_multiple_report_only'], 'x', 'r', True)]
    ex.append(('T10', 'Addendum 64 rows: the m = 1.75 pair and the minimum-SoH 0.50 row (both cells certified in each '
               f"claim). At soh_min 0.50: EFC/day {fmt(sb['efc_per_day']['2025'], 'efc', 'tex')} / "
               f"{fmt(sb['efc_per_day']['2030'], 'efc', 'tex')} / {fmt(sb['efc_per_day']['2035'], 'efc', 'tex')} "
               f"(2025 / 2030 / 2035); 2035 SoH_end {fmt(sb['soh_end_2035'], 'soh', 'tex')}; the floor never binds "
               f"(|dual| ≤ {sb['floor_duals_abs_max']:.1e}).", cols, a6['rows']))

    # ---- T11 ----
    def t11val(r, style):
        return r.get('text') if r['kind'] == 'txt' else fmt(r['value'], r['kind'], style)
    cols = [('quantity', lambda r: r['quantity'], 'txt', 'l', True),
            ('value', lambda r: r, 'row', 'r', True),
            ('unit', lambda r: {'EUR': 'k€', 'x': '×'}.get(r['unit'], r['unit']), 'txt', 'l', True),
            ('label', lambda r: r['label'], 'txt', 'l', True),
            ('source', lambda r: r['source'], 'txt', 'l', False)]
    ex.append(('T11', 'The 3 × 3 multi-scenario instance (Addenda 52 and 54): x = 0 optimal under the baseline; '
               f"R ∈ [{t['three_by_three']['R_range_rounded'][0]:.3f}, {t['three_by_three']['R_range_rounded'][1]:.3f}] "
               'against 0.933 predicted. Not part of T1-T10; sources per row.', cols, t['three_by_three']['rows']))
    return ex, t11val


def render(ex, t11val):
    """-> {filename: text} for CSV and LaTeX, plus per-table structural facts for the checks."""
    files, facts = {}, {}
    for name, caption, cols, rows in ex:
        def cell(c, r, style):
            hdr, get, kind, _al, _lt = c
            v = get(r)
            if kind == 'row':
                return t11val(v, style)
            return fmt(v, kind, style)
        # CSV: every column
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        w.writerow([c[0] for c in cols])
        for r in rows:
            w.writerow([cell(c, r, 'csv') for c in cols])
        files[f'{name}.csv'] = buf.getvalue()
        # LaTeX
        lc = [c for c in cols if c[4]]
        L = [f'% {name}: generated by {SCRIPT_REL} (W160) from the frozen Step 6 tables; requires \\usepackage{{booktabs}}.',
             '% Plain tabular, no siunitx. The CSV beside this file carries every column; columns omitted here for '
             'width are listed in README.md.',
             '\\begin{table}[htbp]', '\\centering', '\\scriptsize',
             f'\\caption{{{tex(caption + " " + OBJECTIVE_CAPTION)}}}', f'\\label{{tab:{name}}}',
             '\\begin{tabular}{' + ''.join(c[3] for c in lc) + '}', '\\toprule',
             ' & '.join(tex(c[0]) for c in lc) + ' \\\\', '\\midrule']
        for r in rows:
            L.append(' & '.join(tex(cell(c, r, 'tex')) for c in lc) + ' \\\\')
        L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
        txt = '\n'.join(L)
        files[f'{name}.tex'] = txt
        body = [ln for ln in L if ln.endswith('\\\\')]
        facts[name] = {'n_rows': len(rows), 'n_csv_columns': len(cols), 'n_latex_columns': len(lc),
                       'latex_columns_omitted': [c[0] for c in cols if not c[4]],
                       'latex_ascii_only': all(ord(ch) < 128 for ch in txt),
                       'latex_non_ascii': sorted({ch for ch in txt if ord(ch) >= 128}),
                       'latex_row_cell_counts_ok': all(_count_amp(ln) == len(lc) - 1 for ln in body),
                       'latex_braces_balanced': _braces_balanced(txt),
                       'csv_rows_parse': len(list(csv.reader(io.StringIO(files[f'{name}.csv'])))) == len(rows) + 1}
    return files, facts


def _count_amp(line):
    n, i = 0, 0
    while i < len(line):
        if line[i] == '\\':
            i += 2
            continue
        if line[i] == '&':
            n += 1
        i += 1
    return n


def _braces_balanced(txt):
    d, i = 0, 0
    while i < len(txt):
        if txt[i] == '\\':
            i += 2
            continue
        if txt[i] == '{':
            d += 1
        elif txt[i] == '}':
            d -= 1
            if d < 0:
                return False
        i += 1
    return d == 0


def md_table(name, caption, cols, rows, t11val):
    L = [f'### {name}', '', caption + ' ' + OBJECTIVE_CAPTION, '']
    L.append('| ' + ' | '.join(c[0] for c in cols) + ' |')
    L.append('|' + '|'.join('---:' if c[3] == 'r' else '---' for c in cols) + '|')
    for r in rows:
        vals = []
        for c in cols:
            v = c[1](r)
            s = t11val(v, 'md') if c[2] == 'row' else fmt(v, c[2], 'md')
            vals.append(str(s).replace('|', '\\|').replace('\n', ' '))
        L.append('| ' + ' | '.join(vals) + ' |')
    L.append('')
    return L


# ======================================================================================================================
#  paragraphs.md (verbatim slices of the package at e3437284)
# ======================================================================================================================
SLICES = (
    ('paragraphs', '## (ii) Certification paragraph — draft text', '## (iii) Limitations paragraph — draft text'),
    ('paragraphs', '## (iii) Limitations paragraph — draft text', '## (iv) Reproducibility note — draft text and sources'),
    ('paragraphs', '## (iv) Reproducibility note — draft text and sources', '## (v) Prediction scorecard (supplementary material)'),
    ('map', '## (vi) Response-to-reviewers map', '## Not confirmed'),
    ('sentences', '### Two sentences for the new rows (accepted, Addendum 65)', '## (ii) Certification paragraph — draft text'),
    ('scorecard', '## (v) Prediction scorecard (supplementary material)', '## (vi) Response-to-reviewers map'),
)
SLICE_TITLES = {'paragraphs': 'Draft paragraphs: certification (ii), limitations (iii), reproducibility (iv)',
                'map': 'Response-to-reviewers map (vi)',
                'sentences': 'The verbatim and accepted sentences (from section (i))',
                'scorecard': 'Prediction scorecard (v) -- supplementary material'}


def paragraphs_md(pkg, pkg_sha):
    lines = pkg.split('\n')
    probs, out, ranges = [], [], []
    out += ['# Step 6 paragraphs, reviewer map, sentences and scorecard -- VERBATIM',
            '',
            f'<!-- W160: every block below is copied verbatim from `{PACKAGE_REL}` at commit `{PACKAGE_COMMIT}` '
            f'(read from the git object store; blob sha256 `{pkg_sha}`). The lines marked "W160" (this comment and the '
            'block headers) are the only text not from the package. Nothing is rewritten; inconsistencies with the '
            'frozen tables are reported by the W160 figure check, not edited here. -->', '']
    last_group = None
    for group, start, end in SLICES:
        si = [i for i, ln in enumerate(lines) if ln == start]
        ei = [i for i, ln in enumerate(lines) if ln == end]
        if len(si) != 1 or len(ei) != 1 or ei[0] <= si[0]:
            probs.append(f'slice {start!r} -> {end!r} not found exactly once')
            continue
        a, b = si[0], ei[0]
        while b > a and lines[b - 1].strip() in ('', '---'):
            b -= 1
        if group != last_group:
            out += [f'<!-- W160 block: {SLICE_TITLES[group]} -->', '']
            last_group = group
        out += [f'<!-- W160: package lines {a + 1}-{b} -->'] + lines[a:b] + ['']
        ranges.append({'group': group, 'start_heading': start, 'lines': [a + 1, b]})
    return '\n'.join(out) + '\n', ranges, probs


# ======================================================================================================================
#  the figure check (paragraphs.md against the frozen JSON)
# ======================================================================================================================
def _dp(written):
    s = written.replace(',', '').lstrip('+−-')
    return len(s.split('.')[1]) if '.' in s else 0


def _wval(written):
    s = written.replace(',', '').replace('−', '-').replace('+', '')
    return float(s)


def figure_checks(fz, ptext, w153c):
    t = fz['tables']
    cells = t['cells']
    allc = dict(cells)
    allc.update(t['cells_appended_w160'])
    cl = {r['claim_id']: r for r in t['claims']}
    t10 = {r['claim_id']: r for r in t['a64']['rows']}
    t10c = {}
    for r in t['a64']['rows']:
        t10c.update(r.get('cells') or {})
    ag = {r['arm']: r for r in t['ageing']['rows']}
    dz = {r['cell']: r for r in t['dead_zone']['cells']}
    be = t['break_even']
    b = t['benchmark']
    sb = t['a64']['scored']['B']
    st = [r for r in cells.values() if r.get('certification_stats_included')]
    cert = [r for r in st if r['status'] == 'certified']
    # the m = 2 dead-zone cells T9 carries with a per-node record (j_5f3cccb4 and the L cells of T9)
    m2cells = [r['cell'] for r in t['dead_zone']['cells']
               if r.get('share_by_node') and (r['cell'].startswith('j_') or r['cell'].startswith('l_'))]
    m2dz = [dz[c] for c in m2cells if c in dz] + [dz['ref:5ca4f86c'], dz['ref:e28de4ac']]
    mono_l = [r for r in cert if r['branch'] == 'monotone' and r['cell'].startswith('l_')]

    def q(xs, p):
        xs = sorted(xs)
        pos = (len(xs) - 1) * p
        i = int(pos)
        f = pos - i
        return xs[i] if f == 0 else xs[i] + (xs[i + 1] - xs[i]) * f

    def med(xs):
        return q(xs, 0.5)
    C = []

    def add(cid, fragment, written, kind, value, ref, absval=False, note=None, status=None):
        C.append({'id': cid, 'fragment': fragment, 'written': written, 'kind': kind, 'table_value': value,
                  'table_ref': ref, 'absval': absval, 'note': note, 'status_override': status})
    # -- sentences
    add('S1a', '(+14.2 k€ and +46.3 k€,', '+14.2', 'keur', t10['H:m1.75:value_minus_I']['d_gross'], 'T10 H:m1.75 d gross')
    add('S1b', '(+14.2 k€ and +46.3 k€,', '+46.3', 'keur', cl['H:m2:value_minus_I']['d_gross'], 'T1 H:m2 d gross')
    d15, d175 = cl['H:m1.5:value_minus_I']['d_gross'], t10['H:m1.75:value_minus_I']['d_gross']
    add('S1c', '≈ 1.63 by linear interpolation', '1.63', 'raw', 1.5 + 0.25 * (-d15) / (d175 - d15),
        'derived: 1.5 + 0.25·(−d(1.5)) / (d(1.75) − d(1.5)) from T1 H:m1.5 and T10 H:m1.75')
    add('S2a', 'EFC/day +0.18,', '+0.18', 'raw', None, 'T10 carries EFC/day at 0.50 only; the 0.70 per-year EFC/day '
        '(W153) is not in T1-T10', status='no table counterpart')
    add('S2b', 'adds only +4.9 k€, within resolution', '+4.9', 'keur', t10['E:soh050:delta_value_vs_070']['d_gross'],
        'T10 E:soh050:delta_value_vs_070')
    add('S2c', 'adds only +4.9 k€, within resolution', 'within resolution', 'verdict',
        t10['E:soh050:delta_value_vs_070']['gross_verdict'], 'T10 verdict')
    add('S2d', 'the unit still loses 59.5 k€ determinately', '59.5', 'keur', t10['E:soh050:value_minus_I']['d_gross'],
        'T10 E:soh050:value_minus_I', absval=True)
    add('S2e', 'the unit still loses 59.5 k€ determinately', 'determinate', 'verdict',
        t10['E:soh050:value_minus_I']['gross_verdict'], 'T10 verdict')
    add('V1a', 'without ageing the unit is at break-even (−4.1 k€, within resolution)', '−4.1', 'keur',
        cl['E:n7_4h_e1_no_ageing:value_minus_I']['d_gross'], 'T1 E no_ageing value − I')
    add('V1b', 'without ageing the unit is at break-even (−4.1 k€, within resolution)', 'within resolution', 'verdict',
        cl['E:n7_4h_e1_no_ageing:value_minus_I']['gross_verdict'], 'T1 verdict')
    aged = [ag[a]['value_minus_I'] for a in ('C2', 'C2_calfade', 'C3_unit', 'C4', 'C3_midblock')]
    add('V1c', 'loses 31.6–73.7 k€ across the aged arms', '31.6', 'keur', max(aged), 'T8 aged arms: smallest loss',
        absval=True)
    add('V1d', 'loses 31.6–73.7 k€ across the aged arms', '73.7', 'keur', min(aged), 'T8 aged arms: largest loss',
        absval=True)
    add('V1e', '−4,140.54 (0.32×)', '−4,140.54', 'eur', cl['E:n7_4h_e1_no_ageing:value_minus_I']['d_gross'], 'T1')
    add('V1f', '−4,140.54 (0.32×)', '0.32', 'raw', cl['E:n7_4h_e1_no_ageing:value_minus_I']['gross_multiple'], 'T1 ×')
    add('V1g', 'Aged arms: C2 −31,607.24 … C3_unit −73,746.66', '−31,607.24', 'eur', ag['C2']['value_minus_I'], 'T8 C2')
    add('V1h', 'Aged arms: C2 −31,607.24 … C3_unit −73,746.66', '−73,746.66', 'eur', ag['C3_unit']['value_minus_I'],
        'T8 C3_unit')
    # -- certification paragraph
    add('C1', 'τ = 4,539.07 €', '4,539.07', 'eur', fz['constants']['TAU'], 'constants.TAU')
    add('C2', 'τ/10 = 453.91 €', '453.91', 'eur', fz['constants']['TAU'] / 10, 'constants.TAU / 10')
    add('C3', 'τ/2 = 2,269.53 €', '2,269.53', 'eur', fz['constants']['TAU'] / 2, 'constants.TAU / 2')
    add('C4', 'Ten of the certificates the tables use stop within 5 % of τ', '10', 'raw',
        t['at_or_above_0_95_tau']['n_counted'], 'T2 at_or_above_0.95_tau_counted (count of true)')
    add('C5', 'Across the 42 re-settled SRP1 cells, 32 certified (24 oscillatory, 8 monotone)', '42', 'raw', len(st),
        'T2 cells with certification_stats_included')
    add('C6', 'Across the 42 re-settled SRP1 cells, 32 certified (24 oscillatory, 8 monotone)', '32', 'raw', len(cert), 'T2')
    add('C7', 'Across the 42 re-settled SRP1 cells, 32 certified (24 oscillatory, 8 monotone)', '24', 'raw',
        sum(r['branch'] == 'oscillatory' for r in cert), 'T2 branch')
    add('C8', 'Across the 42 re-settled SRP1 cells, 32 certified (24 oscillatory, 8 monotone)', '8', 'raw',
        sum(r['branch'] == 'monotone' for r in cert), 'T2 branch')
    unc = [r for r in st if r['status'] != 'certified']
    for cid, word, n in (('C9', 'gap clause', '5'), ('C10', 'lapse reset', '2'), ('C11', 'growth test', '3')):
        add(cid, 'follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test', n, 'raw',
            sum((r.get('cause_uncertified_W157') or r.get('cause_uncertified')) == word for r in unc),
            f'T2 cause (W157 wording) = {word}')
    add('C12', 'certification cycle was 174', '174', 'raw', med([r['k_star'] for r in cert]), 'T2 median k* (32 certified)')
    dk = [r['k_star'] - r['k0_run'] for r in cert]
    w153_dk = w153c['result']['totals']['cycles_after_k0_k_star_minus_k0_at_decision']['median']
    add('C13', 'certification came a median 63.5 cycles after the first residual pass (range 21–89)', '63.5', 'raw',
        med(dk), 'T2: median of k* − k0, where T2 k0 = k0_run = the first residual pass N',
        note=(f'W153 (input, not a table) gives 63.5 for k* − k0 AT DECISION (k0 reset by lapses on e_c3_midblock and '
              f'e_no_ageing) = {w153_dk}; measured from the first residual pass N, as the sentence says, the median is '
              f'{med(dk)} (W153 k_star_minus_N median '
              f"{w153c['result']['totals']['cycles_after_k0_k_star_minus_N']['median']})"))
    add('C14', 'certification came a median 63.5 cycles after the first residual pass (range 21–89)', '21', 'raw',
        min(dk), 'T2 min k* − k0')
    add('C15', 'certification came a median 63.5 cycles after the first residual pass (range 21–89)', '89', 'raw',
        max(dk), 'T2 max k* − k0')
    add('C16', 'The median range/τ at certification was 0.86', '0.86', 'raw', med([r['range_over_tau'] for r in cert]),
        'T2 median range/τ (32 certified)')
    add('C17', 'The four cells added under Addendum 64 all certified', '4', 'raw',
        sum(1 for c in t10c.values() if c['status'] == 'certified'), 'T10 cells certified')
    ks = [r['k_star'] for r in cert]
    for cid, w, p in (('C18', '138', 0), ('C19', '156.75', .25), ('C20', '174', .5), ('C21', '198.25', .75), ('C22', '400', 1)):
        add(cid, 'min 138, q1 156.75, median 174, q3 198.25, max 400', w, 'raw', q(ks, p), f'T2 k* quantile {p} (linear)')
    rot = [r['range_over_tau'] for r in cert]
    for cid, w, v in (('C23', '0.012', min(rot)), ('C24', '0.860', med(rot)), ('C25', '0.997', max(rot))):
        add(cid, 'min 0.012, median 0.860, max 0.997', w, 'raw', v, 'T2 range/τ (32 certified)')
    add('C26', '| Cells at range/τ ≥ 0.95 (W153 set) | 6:', '6', 'raw', sum(x >= FLAG for x in rot), 'T2 (42-cell set)')
    tse = [r['terminal_step_over_EPS0'] for r in cert]
    tsa = [r['terminal_step_over_EPS0'] for r in st]
    add('C27', 'median 1.72, max 10.42; 25 of 42 cells above 1', '1.72', 'raw', med(tse), 'T2 terminal step / EPS0, certified')
    add('C28', 'median 1.72, max 10.42; 25 of 42 cells above 1', '10.42', 'raw', max(tse), 'T2, certified')
    add('C29', 'median 1.72, max 10.42; 25 of 42 cells above 1', '25', 'raw', sum(x > 1 for x in tsa),
        'T2, all 42 cells', note=f'over all 42 cells; over the 32 certified cells the count is {sum(x > 1 for x in tse)}')
    add('C30', '46, in 2 cells (`d_f759dd48` 17, `pb_y2025_n5_v6` 29)', '46', 'raw',
        sum(r.get('n_vetoes') or 0 for r in st), 'T2 n_vetoes')
    add('C31', '46, in 2 cells (`d_f759dd48` 17, `pb_y2025_n5_v6` 29)', '17', 'raw', cells['d_f759dd48']['n_vetoes'], 'T2')
    add('C32', '46, in 2 cells (`d_f759dd48` 17, `pb_y2025_n5_v6` 29)', '29', 'raw', cells['pb_y2025_n5_v6']['n_vetoes'], 'T2')
    nca = [len(r.get('non_clean_after_N') or []) for r in st]
    add('C33', '21 cycles in 10 cells, **all TSO**', '21', 'raw', sum(nca), 'T2 non-clean cycles after N')
    add('C34', '21 cycles in 10 cells, **all TSO**', '10', 'raw', sum(1 for n in nca if n), 'T2')
    for cid, frag, w, c in (('C35', '`b_2a0ba8b2` 0.970', '0.970', 'b_2a0ba8b2'),
                            ('C36', '`b_4649234b` 0.989', '0.989', 'b_4649234b'),
                            ('C37', '`bd504ecf` (3f084f2f) 0.959', '0.959', UNIT_REF),
                            ('C38', '`pb_y2030_n7` 0.994', '0.994', 'pb_y2030_n7'),
                            ('C39', '`pb_y2025_n5` 0.993 (superseded)', '0.993', 'pb_y2025_n5'),
                            ('C40', '`pb_y2025_n7` 0.974', '0.974', 'pb_y2025_n7')):
        add(cid, frag, w, 'raw', allc[c]['range_over_tau'], f'T2 {c} range/τ')
    for cid, frag, c in (('C41', '| `j_a11d7966` | 138 | bar 15,879.58 |', 'j_a11d7966'),
                         ('C43', '| `d_4a82a64a` | 108 | bar 70,743.93 |', 'd_4a82a64a')):
        add(cid, frag, frag.split('|')[2].strip(), 'raw', ','.join(str(x) for x in cells[c]['turning_points_at_non_clean_cycle']),
            f'T2 {c} NCTP', status=None)
        add(cid + 'b', frag, frag.split('bar ')[1].split(' ')[0], 'eur', cells[c]['uncertified_form_beside']['bar'],
            f'T2 {c} uncertified form bar')
    # -- limitations paragraph
    ts = [r['t_sum_at_cap'] for r in m2dz]
    add('L1', 'a priced gap of −8.8 to −9.4 k€', '−8.8', 'keur', max(ts), 'T9 m = 2 cells and F2 references: max t_sum')
    add('L2', 'a priced gap of −8.8 to −9.4 k€', '−9.4', 'keur', min(ts), 'T9: min t_sum')
    for n, w, cid in (('5', '55', 'L3'), ('9', '31', 'L4'), ('7', '14', 'L5')):
        vals = sorted({f"{100 * dz[c]['share_by_node'][n]:.0f}" for c in m2cells})
        add(cid, 'split about 55 % at node 5, 31 % at node 9 and 14 % at node 7', w, 'set', vals,
            f'T9 node {n} share of the m = 2 cells with a per-node record ({", ".join(m2cells)}), integer %')
    add('L5b', 'in each of the six m = 2 cells of T9 that carry a per-node record', '6', 'raw', len(m2cells),
        'T9: m = 2 cells with a per-node record', note=f'T9 carries {", ".join(m2cells)}')
    pf = [dz[c]['pf_primal_last'] for c in m2cells]
    add('L6', 'the power-flow primal residual frozen near 0.69', '0.69', 'near', [min(pf), max(pf)],
        'T9 pf_primal last, six m = 2 cells (within ±0.01 of the written value)')
    add('L7', 'its consensus gap is small (+1.1 k€)', '+1.1', 'keur', dz[D366]['t_sum_at_cap'], 'T9 d_36686489 t_sum')
    add('L8', 'Uncertified by the growth test after five non-clean TSO recoveries.', '5', 'raw',
        len(cells[D366]['non_clean_after_N']), 'T2 d_36686489 non-clean cycles after N')
    band = t['ageing']['eps_AE_band_over_resolvable_070']
    add('L9', '(ε_AE 1.04–1.85', '1.04', 'raw', band[0], 'T8 ε_AE band over resolvable arms (0.70)')
    add('L10', '(ε_AE 1.04–1.85', '1.85', 'raw', band[1], 'T8')
    e050 = {a: r['eps_AE_050_superseded'] for a, r in ag.items() if r['eps_AE_050_superseded'] is not None}
    add('L11', '> against ≈ 0.6 at 0.50).', '0.6', 'approx', e050,
        'T8 ε_AE (0.50, superseded) column, per arm', status='approximate')
    tsum = {'j_5f3cccb4': '−8,847.90', 'l_45aa25a6': '−9,197.61', 'l_7c455554': '−9,247.76', 'l_b2251bc5': '−9,401.86',
            'l_0ee93aca': '−9,300.09', 'ref:5ca4f86c': '−9,234.42', 'ref:e28de4ac': '−9,300.13'}
    for i, (c, w) in enumerate(tsum.items()):
        frag = f'| {"F2 incumbent" if c == "ref:5ca4f86c" else "F2 challenger" if c == "ref:e28de4ac" else "`" + c + "`"} | {w} |'
        add(f'L12.{i}', frag, w, 'eur', dz[c]['t_sum_at_cap'], f'T9 {c} t_sum at cap')
    add('L13', '- pf_primal last 0.6819–0.6999.', '0.6819', 'raw', min(pf), 'T9 min pf_primal last (m = 2 cells)')
    add('L14', '- pf_primal last 0.6819–0.6999.', '0.6999', 'raw', max(pf), 'T9 max')
    add('L15', 'The m = 1.5 cell `h_f9eae48f` has the same split at −2,481.75.', '−2,481.75', 'eur',
        dz['h_f9eae48f']['t_sum_at_cap'], 'T9 h_f9eae48f')
    hs = dz['h_f9eae48f']['share_by_node']
    add('L16', 'The m = 1.5 cell `h_f9eae48f` has the same split at −2,481.75.', '55/14/31', 'split',
        [hs['5'], hs['7'], hs['9']], 'T9 h_f9eae48f shares n5/n7/n9, integer %',
        note=f"the m = 2 split is 55/14/31; h_f9eae48f's node-7 share is {100 * hs['7']:.3f} %, which rounds to 15")
    add('L17', 'lapse resets at 218 and 222', '218,222', 'raw',
        ','.join(str(e['cycle']) for e in dz['l_0ee93aca']['lapse_events']), 'T9 l_0ee93aca lapse events')
    add('L18', 'T9 records its t_sum at the cap as **+1,082.23** (node split 58/11/30 %)', '+1,082.23', 'eur',
        dz[D366]['t_sum_at_cap'], 'T9')
    s3 = dz[D366]['share_by_node']
    add('L19', 'T9 records its t_sum at the cap as **+1,082.23** (node split 58/11/30 %)', '58/11/30', 'split',
        [s3['5'], s3['7'], s3['9']], 'T9 d_36686489 shares')
    add('L20', 'Seven are L cells with range/τ 0.012–0.171.', '7', 'raw', len(mono_l), 'T2 monotone L certificates')
    add('L21', 'Seven are L cells with range/τ 0.012–0.171.', '0.012', 'raw', min(r['range_over_tau'] for r in mono_l), 'T2')
    add('L22', 'Seven are L cells with range/τ 0.012–0.171.', '0.171', 'raw', max(r['range_over_tau'] for r in mono_l), 'T2')
    add('L23', '**`i_5a6a88b4` has 0.939**', '0.939', 'raw', cells[BESIDE_CELL]['range_over_tau'], 'T2')
    for cid, w, a in (('L24', '1.04', 'no_ageing'), ('L25', '1.30', 'C2'), ('L26', '1.85', 'C4')):
        add(cid, '1.04 (no ageing), 1.30 (C2), 1.85 (C4)', w, 'raw', ag[a]['eps_AE_070'], f'T8 ε_AE (0.70) {a}')
    # -- scorecard
    add('P14', '**held** (1 and 8)', '1', 'raw', b['sweep']['sweep_passive_cold']['n_blocks'], 'T6 sweep passive')
    add('P14b', '**held** (1 and 8)', '8', 'raw', b['sweep']['sweep_price_taker_cold']['n_blocks'], 'T6 sweep price-taker')
    add('P20', 'pb_y2030_n9 **missed** (+27,488)', '+27,488', 'eur', allc['pb_y2030_n9']['s_signed'],
        'T2 appended pb_y2030_n9 s (W118)')
    add('P22', 'certifies at 173 (Planner, A59)', '173', 'raw', cells['b_2a0ba8b2']['k_star'], 'T2 b_2a0ba8b2 k*')
    add('P24', '`pb_y2025_n5_v6` later **held** (+20,913)', '+20,913', 'eur', cells['pb_y2025_n5_v6']['s_signed'], 'T2 s')
    cw = be['committed_W145']
    add('P32a', '**held** at the midpoints: 0.59 % on b, 0.31 % on b + c/4', '0.59', 'pct', cw['slope_b_rel_mid'],
        'T3 committed W145 b agreement at midpoint')
    add('P32b', '**held** at the midpoints: 0.59 % on b, 0.31 % on b + c/4', '0.31', 'pct',
        cw['slope_b_plus_c_over_4_rel_mid'], 'T3 committed W145 b + c/4 agreement at midpoint')
    # the worst corners on b and on b + c/4 (Addendum 64 ruling 3: "the worst corners (7.1 %, 3.7 %)")
    add('P32c', 'Corners 7.1 % / 3.7 % are interval sensitivity', '7.1', 'pct', max(cw['slope_b_rel_corners']),
        'T3 committed W145: worst corner on b')
    add('P32d', 'Corners 7.1 % / 3.7 % are interval sensitivity', '3.7', 'pct',
        max(cw['slope_b_plus_c_over_4_rel_corners']), 'T3 committed W145: worst corner on b + c/4')
    add('P32e', 'Conservative 4-interval fit (report-only, W157): 0.47 % at the midpoint', '0.47', 'pct',
        be['conservative_A64']['slope_b_plus_c_over_4_rel_mid'], 'T3 conservative b + c/4 midpoint')
    add('P33a', '**held** (182,702; ≥ 63.5 k; conservative ≥ 61.3 k)', '182,702', 'eur',
        cw['certified_only']['breakeven_marginal_4h_energy_cost'], 'T3 committed certified-only e*')
    add('P33b', '**held** (182,702; ≥ 63.5 k; conservative ≥ 61.3 k)', '63.5', 'keur', cw['margin_to_cost_range'][0],
        'T3 committed margin min (k€/MWh)')
    add('P33c', '**held** (182,702; ≥ 63.5 k; conservative ≥ 61.3 k)', '61.3', 'keur',
        be['manuscript_figure']['margin_min_eur_per_mwh'], 'T3 conservative margin min (k€/MWh)')
    add('P35', 'size **missed low** (−14.45 k)', '−14.45', 'keur', d15, 'T1 H:m1.5')
    add('P36a', '**held** (+46.3 k); **held** (+46.5 k)', '+46.3', 'keur', cl['H:m2:value_minus_I']['d_gross'], 'T1 H:m2')
    add('P36b', '**held** (+46.3 k); **held** (+46.5 k)', '+46.5', 'keur', cl['I:m2:second_MWh']['d_gross'], 'T1 I second MWh')
    add('P37', '**missed** (determinate, 1.95×)', '1.95', 'raw', cl['J:e4_to_e5']['gross_multiple'], 'T1 J:e4_to_e5 ×')
    add('P38', '**held** (−8,848)', '−8,848', 'eur', dz['j_5f3cccb4']['t_sum_at_cap'], 'T9 j_5f3cccb4')
    h175 = t10['H:m1.75:value_minus_I']
    add('P47a', '**held**: +14,249.16 inside [+8, +22] k€', '+14,249.16', 'eur', h175['d_gross'], 'T10')
    add('P47b', 'determinate 1.32×. Threshold 10,775 fell below', '1.32', 'raw', h175['gross_multiple'], 'T10 ×')
    add('P47c', 'determinate 1.32×. Threshold 10,775 fell below', '10,775', 'eur', h175['gross_threshold_or_bar'], 'T10')
    add('P47d', 'Unit k\\* 121 is below [130, 185]; k₀ 108 / 101 inside', '121', 'raw', t10c['h_unit_m175']['k_star'], 'T10')
    add('P47e', 'Unit k\\* 121 is below [130, 185]; k₀ 108 / 101 inside', '108', 'raw', t10c['h_x0_m175']['k0_run'], 'T10')
    add('P47f', 'Unit k\\* 121 is below [130, 185]; k₀ 108 / 101 inside', '101', 'raw', t10c['h_unit_m175']['k0_run'], 'T10')
    s05 = t10['E:soh050:delta_value_vs_070']
    add('P48a', '**held**: +4,916.67 inside [+4, +22] k€', '+4,916.67', 'eur', s05['d_gross'], 'T10')
    add('P48b', 'within resolution (0.38×, threshold 13,054.22 as predicted)', '0.38', 'raw', s05['gross_multiple'], 'T10')
    add('P48c', 'within resolution (0.38×, threshold 13,054.22 as predicted)', '13,054.22', 'eur',
        s05['gross_threshold_or_bar'], 'T10')
    add('P48d', 'Floor never binds (duals ≤ 5.1 × 10⁻¹⁰; 2035 SoH_end 0.677)', '5.1e-10', 'sci', sb['floor_duals_abs_max'],
        'T10 scored B floor duals')
    add('P48e', 'Floor never binds (duals ≤ 5.1 × 10⁻¹⁰; 2035 SoH_end 0.677)', '0.677', 'raw', sb['soh_end_2035'], 'T10')
    for cid, y, w in (('P48f', '2025', '1.192'), ('P48g', '2030', '1.100'), ('P48h', '2035', '0.909')):
        add(cid, 'EFC/day inside every range (1.192 / 1.100 / 0.909)', w, 'raw', sb['efc_per_day'][y], f'T10 EFC/day {y}')
    # -- reviewer map
    add('M1', 'T4: year ladder (gross +42,458.65 determinate; net −2,286.25 within resolution)', '+42,458.65', 'eur',
        t['year_ladder']['D_gross'], 'T4')
    add('M2', 'T4: year ladder (gross +42,458.65 determinate; net −2,286.25 within resolution)', '−2,286.25', 'eur',
        t['year_ladder']['D_net'], 'T4')
    add('M3', 'T10 (0.50: Δvalue +4,916.67 within resolution; value − I −59,500.72 determinate)', '−59,500.72', 'eur',
        t10['E:soh050:value_minus_I']['d_gross'], 'T10')
    add('M4', '−31,607.24 vs −64,417.39, both determinate', '−64,417.39', 'eur', ag['C2_calfade']['value_minus_I'], 'T8')
    # (arm, value - I as written, multiple as written, floor-binds column as written in the package R3.6 table)
    r36 = (('no_ageing', '−4,140.54', '0.32', '—'), ('C2', '−31,607.24', '2.50', 'never'),
           ('C2_calfade', '−64,417.39', '4.93', '2035'), ('C3_unit', '−73,746.66', '5.74', '2035'),
           ('C4', '−45,196.70', '3.58', 'never'), ('C3_midblock', '−65,891.88', '5.12', '2035'))
    claim_of = {'C2': 'E:n7_4h_e1_C2:value_minus_I', 'C2_calfade': 'E:n7_4h_e1_C2_calfade:value_minus_I',
                'C3_unit': 'E:C3_unit_value_minus_I', 'C4': 'E:n7_4h_e1_C4:value_minus_I',
                'C3_midblock': 'E:n7_4h_e1_C3_midblock:value_minus_I', 'no_ageing': 'E:n7_4h_e1_no_ageing:value_minus_I'}
    for i, (a, w, x, fbw) in enumerate(r36):
        add(f'M5.{i}a', f'| {w} |', w, 'eur', ag[a]['value_minus_I'], f'T8 {a}')
        add(f'M5.{i}b', f'| {w} | {x}×', x, 'raw', cl[claim_of[a]]['gross_multiple'], f'T1 {claim_of[a]} ×')
        fb = ag[a]['floor_year_070']
        frag_fb = f"| {x}×, {'**within resolution**' if a == 'no_ageing' else 'determinate'} | {fbw} |"
        tv = '—' if (a == 'no_ageing' and fb is None) else (str(fb) if fb else 'never')
        add(f'M5.{i}c', frag_fb, fbw, 'raw', tv, f'T8 {a} floor binds (no_ageing: no floor year recorded, shown —)')
    add('M6', '| +69,606.12 | 5.41×, determinate |', '+69,606.12', 'eur', cl['E:n7_4h_e1_no_ageing:vs_C3']['d_gross'],
        'T1 E no_ageing vs C3')
    add('M6b', '| +69,606.12 | 5.41×, determinate |', '5.41', 'raw', cl['E:n7_4h_e1_no_ageing:vs_C3']['gross_multiple'], 'T1')
    add('M7', '| −59,500.72 (Δ vs 0.70: +4,916.67) | 4.71×, determinate (Δ: 0.38×, within resolution) |', '4.71', 'raw',
        t10['E:soh050:value_minus_I']['gross_multiple'], 'T10')
    disc = {round(r['rate'], 2): r for r in t['discount']}
    add('M8', 'T7: value − I negative and determinate at 0 / 2 / 5 / 8 %', 'determinate ×4', 'raw',
        'determinate ×4' if all(r['verdict'] == 'determinate' and r['value_minus_I'] < 0 for r in disc.values())
        else 'not all', 'T7 verdicts and signs')
    add('M9', 'R ∈ [0.909, 0.934]) | headline percentages regenerated', '[0.909, 0.934]', 'raw',
        f"[{t['three_by_three']['R_range_rounded'][0]:.3f}, {t['three_by_three']['R_range_rounded'][1]:.3f}]", 'T11')
    add('M10', '- **Result:** **+90,896,608.40 € = 13.9 %** of the coordinated Q181, determinate at 1,263.75× the larger band',
        '+90,896,608.40', 'eur', b['benefit'], 'T6')
    add('M11', '- **Result:** **+90,896,608.40 € = 13.9 %** of the coordinated Q181, determinate at 1,263.75× the larger band',
        '13.9', 'pct', b['benefit_relative'], 'T6')
    add('M12', '- **Result:** **+90,896,608.40 € = 13.9 %** of the coordinated Q181, determinate at 1,263.75× the larger band',
        '1,263.75', 'raw', b['multiple'], 'T6')
    add('M13', '  (71,926.11). Source: T6;', '71,926.11', 'eur', b['larger_band'], 'T6')
    add('M14', '- **Caveat stated with it:** 4 reverse-flow interface-hours in the coordinated solution.', '4', 'raw',
        b['reverse_flow_interface_hours'], 'T6')
    add('M15', 'At SRP1 the unit does not pay: value − I = −64,417.39.', '−64,417.39', 'eur',
        cl['CHECK:headline_V_minus_I_settled']['d_gross'], 'T1 CHECK headline')

    # evaluate
    out = []
    for c in C:
        found = c['fragment'] in ptext
        w, k, v = c['written'], c['kind'], c['table_value']
        if c['status_override'] == 'no table counterpart':
            status, shown = 'no table counterpart', None
        elif k == 'verdict':
            shown = v
            status = 'match' if v == w else 'MISMATCH'
        elif k in ('eur', 'keur', 'pct'):
            dp = _dp(w)
            x = v / 1000.0 if k == 'keur' else 100.0 * v if k == 'pct' else v
            if c['absval']:
                x = abs(x)
            shown = f'{x:.{dp}f}'
            status = 'match' if float(shown) == _wval(w) else 'MISMATCH'
        elif k == 'raw':
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                dp = _dp(w)
                shown = f'{v:.{dp}f}'
                status = 'match' if float(shown) == _wval(w) else 'MISMATCH'
            else:
                shown = str(v)
                status = 'match' if shown == w else 'MISMATCH'
        elif k == 'set':
            shown = '/'.join(v)
            status = 'match' if v == [w] else 'MISMATCH'
        elif k == 'split':
            shown = '/'.join(f'{100 * x:.0f}' for x in v)
            status = 'match' if shown == w else 'MISMATCH'
        elif k == 'near':
            shown = f'{v[0]:.4f}–{v[1]:.4f}'
            status = 'match' if all(abs(x - _wval(w)) <= 0.01 + 1e-12 for x in v) else 'MISMATCH'
        elif k == 'sci':
            shown = f'{v:.1e}'
            status = 'match' if shown == w else 'MISMATCH'
        elif k == 'approx':
            shown = ', '.join(f'{a} {x:.2f}' for a, x in v.items())
            status = 'approximate: ' + ('every arm rounds to 0.6' if all(round(x, 1) == 0.6 for x in v.values())
                                        else 'not every arm rounds to 0.6')
        else:
            raise ValueError(k)
        out.append({'id': c['id'], 'fragment_verbatim': c['fragment'], 'fragment_found_in_paragraphs_md': found,
                    'written': w, 'table_ref': c['table_ref'], 'table_value_full_precision': v if not isinstance(v, list)
                    else list(v), 'table_value_at_written_precision': shown, 'status': status, 'note': c['note']})
    return out


# ======================================================================================================================
#  README
# ======================================================================================================================
COLUMN_DICT = {
    'T1': [('claim', 'claim id; the prefix before the first colon is the claim family (B, C, CHECK, E, G, H, I, J, L), '
                     'as in the W157 tables'),
           ('ref cell / other cell', 'the two cells of the difference (T2 rows)'),
           ('d gross', 'difference on Q = gross_operational_cost, settlement excluded; value form = Q(0) − Q(x) − I '
                       'where the claim is "value − I"'),
           ('rule', 'max(3 × larger bar, 2τ) between certified cells (Addendum 61); uncertified form 3·max(|gap|, '
                    '|slack|) with an uncertified cell (Addendum 58)'),
           ('threshold / bar', 'the determinacy threshold or the uncertified bar of the rule'),
           ('× gross / × net / × Q_cc', '|difference| / threshold-or-bar'),
           ('verdict', 'determinate if × > 1 (the uncertified form tests gross and Q_cc), else within resolution'),
           ('d net', 'difference on Q_net = Q − terminal salvage credit; net label: "recorded" (G rows) or "validated '
                     'by form + salvage identity (W154b)"'),
           ('d Q_cc', 'difference on Q + t_sum, report-only'),
           ('uncertified form beside', 'Addendum 64 ruling 4: the flagged certificates j_a11d7966 / d_4a82a64a '
                                       'treated as uncertified; bar, × and verdict')],
    'T2': [('table', 'T2 = the 49 W157 cells; T4 / T5 / T10 = certificates appended by W160 for the ≥ 0.95 τ scope'),
           ('k0', 'first residual pass N of the run (k0_run)'), ('k* / end', 'certification cycle, or the end / cap'),
           ('range/τ', 'range of Q over the certifying window / τ (certified cells)'),
           ('≥ 0.95 τ (range)', 'range/τ ≥ 0.95 (W157 flag)'),
           (COL_COUNTED, 'true on exactly ten certificates (Addendum 65 ruling 2): ≥ 0.95 τ, not superseded, '
                         'bitwise twins counted once'),
           ('≥ 0.95 τ note', 'counted / superseded (excluded) / bitwise twin of the unit / below 0.95 tau / not a '
                             'certificate'),
           ('flag beside', 'i_5a6a88b4 (monotone, 0.939), flagged beside the ten'),
           ('NCTP cycles', 'turning points on non-clean cycles (Addendum 64 ruling 4)'),
           ('cause (uncertified)', 'the cause the settling rule records; d_36686489 per Addendum 65'),
           ('gap / slack / band', 'uncertified view |t_sum| and |s|; certified band width'),
           ('non-clean cycles after N', 'count of non-clean cycles after the first residual pass'),
           ('replay bitwise through', 'cycle through which the run replayed its original bitwise'),
           ('certifying spec', 'stage spec under which the cell certified (v4 / v5 / v6 / ext v3 / W118 r2 / A64 v1)'),
           ('eval key / candidate key', 'instance identifiers (prefixes; full keys in the JSON)')],
    'T3': [('e*', 'break-even energy cost = b + c/4 − p_cost/4 (€/MWh)'),
           ('certified-only / banded midpoint', 'OLS on certified points only / on all points with uncertified ones '
                                                'at their interval midpoints'),
           ('e* min / max, margin min / max', 'range over the box of interval corners; margin = energy cost − e*'),
           ('agreement', 'relative difference of the slope between the two fits (Addendum 64 ruling 3)')],
    'T4': [('M', 'I + Q − Q181 (W118 form), gross and net'), ('threshold / × / verdict', 'v6 rule between the two '
                                                                                         'certified cells')],
    'T5': [('M gross', 'I + Q − Q181 of each Phase B cell'), ('v6 threshold / × / verdict', 'DET.resolve_v6 against x = 0'),
           ('recorded (W118 rule)', 'the verdict under the superseded rule')],
    'T6': [('Q gross', 'gross operational cost of the arrangement, settlement excluded'),
           ('band', 'coordinated: reproducibility band 0.011 % of Q181; arm: multimodality band over three starts'),
           ('Q − Q181', 'benefit of coordination over the arm; passive arm DERIVED by W160 from recorded values'),
           ('× larger band', 'benefit / max(arm band, coordinated band)'),
           ('NRF violations / max excess', 'consistency re-evaluation, hard DN no-reverse-flow limit (report_v3 '
                                           'consistency_nrf)'),
           ('consistency pass effect', 'Q change of the sequential consistency pass'),
           ('TN / DN curtailment', 'RES curtailment, phase A, day-weighted (report_v3 curtailment_table)'),
           ('sweep', 'unconstrained arm (no interface rule), cold start: blocks and hours the TN cannot accept the DN '
                     'exchange, failing blocks')],
    'T7': [('V', 'value Q(0) − Q(unit) at the rate'), ('threshold', 'Addendum 61 conservative threshold')],
    'T8': [('value / value − I', 'fixed-plan value of the unit under the arm'),
           ('threshold / ×', 'from the matching T1 E claim'),
           ('floor binds (0.70)', 'first year the 0.70 SoH floor binds, or never'),
           ('AE / EFC/day', 'PV-weighted available-energy fraction and EFC/day (ext spec v3 definitions)'),
           ('ε_AE', 'elasticity of value to available energy; the 0.50 column is superseded (label in the header)')],
    'T9': [('t_sum at cap', 'priced interface-consensus gap at the cap'), ('share', 'node split of t_sum'),
           ('pf_primal last', 'power-flow primal residual ratio, last cycle'),
           ('lapse resets at', 'cycles of Boyd-lapse resets')],
    'T10': [('ref / other', 'cells of the claim, with k0 / k* and range/τ'), ('d gross / threshold / × / verdict',
                                                                              'as in T1')],
    'T11': [('value', 'as recorded or derived (label)'), ('source', 'file and commit, addendum')],
}

SPLIT_MAIN = [('headline x = 0 at SRP1', 'T1 CHECK:headline_V_minus_I_settled; T7 (2 % row)'),
              ('break-even fit', 'T3'),
              ('flexibility ladder 1.5 / 1.75 / 2', 'T1 H:m1.5, H:m2; T10 H:m1.75'),
              ('R3.6 ageing rows', 'T8; T10 E:soh050 rows'),
              ('year ladder, gross and net', 'T4'),
              ('discount', 'T7'),
              ('benchmark with both arms (and the mechanism sentence)', 'T6; paragraphs.md sentence 2'),
              ('the 3 × 3 instance (x = 0 optimal, R ∈ [0.909, 0.934])', 'T11')]
SPLIT_SUPP = [('Phase B', 'T5'), ('C / G', 'T1 C and G rows'), ('L (F2)', 'T1 L rows'),
              ('certification statistics', 'T2; paragraphs.md (ii) sources'), ('dead-zone table', 'T9'),
              ('scorecard', 'paragraphs.md (v)')]


def readme(fz_name, fz_sha, facts, t11doc):
    L = ['# Step 6 tables — manuscript-ready export (W160)', '',
         f'Frozen tables: `../{fz_name}` (sha256 `{fz_sha}`). Every file here is generated from that JSON by '
         f'`{SCRIPT_REL}`; the JSON keeps full precision, the CSV and LaTeX files are rounded as below.', '',
         '**Objective convention (every table).** Q = `gross_operational_cost`, settlement excluded — the primary. '
         'Net = Q − terminal salvage credit, shown beside. Q_cc = Q + t_sum, report-only. value = Q(0) − Q(x); '
         'F = Q + I; τ = 4,539.07 €. Each LaTeX caption states it.', '',
         '## Files', '',
         '- `T1.csv` … `T11.csv`: every column; `T1.tex` … `T11.tex`: booktabs, plain `tabular`, `\\caption`, '
         '`\\label{tab:Tn}`, `\\scriptsize`. Requires `\\usepackage{booktabs}`; `\\texteuro{}` is in the LaTeX kernel '
         '(replace by `\\euro{}` if `eurosym` is preferred). The LaTeX files were not compiled here (no TeX '
         'installation on this machine); they were checked for ASCII-only text, balanced braces and the cell count '
         'of every row.',
         '- `paragraphs.md`: the draft paragraphs, the reviewer map, the sentences and the scorecard, verbatim from '
         '`P5_15_STEP6_PACKAGE.md` at `e3437284`.',
         '- T11 is not part of T1–T10: it carries the 3 × 3 instance figures with their sources (below).', '',
         '## Rounding', '', '| quantity | rounding | rule from |', '|---|---|---|']
    for q, r, s in ROUNDING:
        L.append(f"| {q.replace('|', chr(92) + '|')} | {r} | {s} |")
    L += ['', 'Values are rounded to nearest from the full-precision JSON. A value that rounds to zero is written '
              'without a sign. An empty CSV cell (— in LaTeX) means not applicable or not recorded.', '',
          '## Columns omitted from the LaTeX tables (present in the CSV)', '']
    for n, f in facts.items():
        L.append(f"- {n}: {', '.join(f['latex_columns_omitted']) or 'none'}")
    L += ['', '## Column dictionary', '']
    for n, items in COLUMN_DICT.items():
        L.append(f'**{n}**')
        L.append('')
        for k, v in items:
            L.append(f'- `{k}`: {v}')
        L.append('')
    L += ['## Main text and supplementary material — the expert\'s suggestion (**author decides**)', '',
          'Addendum 65. This split is the expert\'s suggestion; the author decides.', '',
          '| main text | tables |', '|---|---|']
    L += [f'| {a} | {b} |' for a, b in SPLIT_MAIN]
    L += ['', '| supplementary | tables |', '|---|---|']
    L += [f'| {a} | {b} |' for a, b in SPLIT_SUPP]
    L += ['', 'Not named in the suggestion: the T1 B rows, the I rows (second MWh at m = 2), the J rows (energy ladder '
              'at m = 2) and the E "vs C3" rows — the author places them.', '',
          '## The 3 × 3 instance (T11)', '',
          'The figures "x = 0 optimal, R ∈ [0.909, 0.934]" are not in T1–T10. They come from Addendum 52 (brief '
          f"`{T11_SOURCE_COMMITS['Addendum 52 (brief)']}`; report `{A51_REPORT_REL}` "
          f"`{T11_SOURCE_COMMITS['A51 continuation report']}`) and Addendum 54 (brief "
          f"`{T11_SOURCE_COMMITS['Addendum 54 (brief)']}`; report `{A53_REPORT_REL}` "
          f"`{T11_SOURCE_COMMITS['A53 SRP1 continuation report']}`). T11 gives them from the committed records: "
          f'`{W91_RESULTS_REL}` (V, I, value − I, resolution) and `{W99_POSTHOC_REL}` (the post-hoc settled descent '
          'D). R is derived (labelled) by ' + t11doc['formula_R'] + '.', '']
    return '\n'.join(L) + '\n'


# ======================================================================================================================
def pickle_state():
    base = W157.pickle_state()
    counts = dict(base['counts'], w160=dict(PICKLE_COUNTS))
    blocked = pickle.load is not _PICKLE_ORIG[0] and pickle.loads is not _PICKLE_ORIG[1]
    ok = blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())
    return {'counts': counts, 'pickle_load_and_loads_blocked': blocked, 'ok': ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()
    tag = 'W160'
    out_dir = args.out_dir
    if os.path.isdir(os.path.join(REPO, out_dir)):
        extra = [f for f in os.listdir(os.path.join(REPO, out_dir)) if f != 'launch.log']
        if extra:
            _log(f'[{tag} PRECONDITION FAILED] {out_dir} already holds {extra} (write-once)')
            sys.exit(1)
    script_clean = W157.L132._committed_clean(SCRIPT_REL)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; out {out_dir}')
    docs, inputs, problems = read_inputs()
    if problems:
        _log(f'[{tag} PRECONDITION FAILED] inputs: {problems}')
        sys.exit(1)
    pred = docs['PRED']
    rebuilt, w157_docs, w157_inputs, p_rb = rebuild_w157()
    if rebuilt is None:
        _log(f'[{tag} PRECONDITION FAILED] W157.load_inputs: {p_rb}')
        sys.exit(1)
    failed = list(p_rb)
    rebuild_equal = {k: rebuilt.get(k) == pred['tables'].get(k) for k in pred['tables']}
    _log(f'[{tag}] inputs: {len(inputs)} W160 + {len(w157_inputs)} W157, all committed clean; W157 rebuild equals the '
         f'predecessor: {all(rebuild_equal.values())}')
    if w157_inputs['BENCH']['sha256'] != inputs['BENCH']['sha256']:
        failed.append('report_v3 sha differs between the W157 and W160 reads')

    tables = copy.deepcopy(pred['tables'])
    # T6
    add6, p = t6_additions(docs['BENCH'])
    failed += p
    tables['benchmark']['w160_additions'] = add6
    # T2 appended + the count
    appended, p = appended_cells(tables, w157_docs, docs['A64SUM'])
    failed += p
    tables['cells_appended_w160'] = appended
    cnt, p = apply_count(tables, appended, w157_docs['SX'], docs['G070'])
    failed += p
    tables['at_or_above_0_95_tau'] = cnt
    for r in tables['phase_b']:
        r[COL_COUNTED] = appended.get(r['cell'], {}).get(COL_COUNTED) if r['cell'] in appended else \
            tables['cells'].get(r['cell'], {}).get(COL_COUNTED)
    for row in tables['a64']['rows']:
        for c, v in (row.get('cells') or {}).items():
            v[COL_COUNTED] = appended[c][COL_COUNTED]
            v[COL_TWIN] = appended[c][COL_TWIN]
    # d_36686489
    d = tables['cells'][D366]
    d366_ok = d['cause_uncertified'] == 'growth test' and len(d['non_clean_after_N'] or []) == 5
    d['cause_uncertified_W157'] = d['cause_uncertified']
    d['cause_uncertified'] = D366_CAUSE
    d['cause_source'] = 'Addendum 65 ruling 3 (W157 recorded "growth test"; five non-clean TSO cycles after N: ' + \
        ', '.join(str(x) for x in d['non_clean_after_N']) + ')'
    for r in tables['dead_zone']['cells']:
        if r['cell'] == D366:
            r['cause_W157'] = r['cause']
            r['cause'] = D366_CAUSE
    # T8 label
    tables['ageing']['column_labels'] = {'eps_AE_050_superseded': T8_EPS050_LABEL}
    # T11
    t11doc, p = t11(docs, tables)
    failed += p
    tables['three_by_three'] = t11doc

    # inputs record
    all_inputs = {'w160': inputs, 'w157_via_load_inputs': w157_inputs}
    frozen = {
        'schema': 'p515_s53_w160_frozen_step6_tables_v1', 'version': 1,
        'stage': 'P5.15 W160 -- Step 6 tables frozen (Addendum 65)',
        'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 65 (order: closing reads -> tables frozen (hash recorded) '
                     '-> manuscript-ready export); Planner task W160',
        'predecessor': {'path': PRED_JSON, 'sha256': PRED_SHA, 'commit': PRED_COMMIT, 'manifest': PRED_MAN,
                        'manifest_sha256': inputs['PRED']['manifest_file_sha256'],
                        'builder': 'p515_s53_w157_step6_tables.py', 'builder_sha256': pred['script']['sha256'],
                        'builder_commit': pred['script']['last_commit']},
        'package': {'path': PACKAGE_REL, 'commit': PACKAGE_COMMIT, 'commit_full': inputs['PACKAGE']['commit_full'],
                    'blob_sha256': inputs['PACKAGE']['sha256']},
        'closing_reads': {'path': W159_REL, 'commit': W159_COMMIT, 'sha256': inputs['W159']['sha256']},
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean},
        'objective_convention': pred['objective_convention'], 'constants': pred['constants'],
        'net_labels': pred['net_labels'],
        'w160_changes': [
            'T6: benchmark.w160_additions (both NRF arms in full; recorded decomposition; passive arm derived; NRF '
            'definition; consistency violations; failing sweep blocks; curtailment table)',
            'T2: cells_appended_w160 (certificates of T4, T5, T10); columns ' + COL_COUNTED + ', ' + COL_TWIN + ', ' +
            COL_TWINS + ', at_or_above_0.95_tau_note, ' + COL_BESIDE + ' on every T2 row and appended row; ' +
            COL_COUNTED + ' also on T5 rows and T10 cell records; at_or_above_0_95_tau (the count, scope, twins)',
            f'T2 / T9: {D366} cause = "{D366_CAUSE}" (W157 value kept as cause_uncertified_W157 / cause_W157)',
            'T8: ageing.column_labels.eps_AE_050_superseded = "' + T8_EPS050_LABEL + '"',
            'T11: three_by_three (Addenda 52 / 54, from committed records)'],
        'w157_rebuild_equals_predecessor': rebuild_equal,
        'inputs': all_inputs, 'tables': tables}
    # figure checks (need paragraphs text first)
    ptext, ranges, p = paragraphs_md(docs['PACKAGE'], inputs['PACKAGE']['sha256'])
    failed += p
    fchk = figure_checks(frozen, ptext, w157_docs['W153C'])
    frozen['paragraphs_md_source_ranges'] = ranges
    frozen['paragraph_figure_checks'] = fchk
    mism = [c for c in fchk if c['status'] == 'MISMATCH']
    frozen['paragraph_figure_check_summary'] = {
        'n_checks': len(fchk), 'n_match': sum(c['status'] == 'match' for c in fchk),
        'n_mismatch': len(mism), 'mismatch_ids': [c['id'] for c in mism],
        'n_no_table_counterpart': sum(c['status'] == 'no table counterpart' for c in fchk),
        'n_approximate': sum(c['status'].startswith('approximate') for c in fchk),
        'all_fragments_found': all(c['fragment_found_in_paragraphs_md'] for c in fchk)}
    checks = {
        'predecessor_sha_matches_expected_and_manifest': (inputs['PRED']['sha256'] == PRED_SHA
                                                          and inputs['PRED']['manifest_sha256_matches'] is True),
        'every_input_committed_clean': all(v.get('committed_clean', True) for v in inputs.values()
                                           if 'committed_clean' in v)
        and all(v['committed_clean'] for v in w157_inputs.values() if 'committed_clean' in v),
        'every_input_manifest_matches': all(v['manifest_sha256_matches'] is not False for v in inputs.values()
                                            if 'manifest_sha256_matches' in v),
        'w157_rebuild_equals_predecessor_all_tables': all(rebuild_equal.values()),
        'package_blob_equals_working_copy': inputs['PACKAGE']['working_copy_equals_blob'],
        'count_at_or_above_0_95_tau_equals_10': cnt['n_counted'] == 10,
        'counted_set_equals_addendum65_ten': cnt['counted_equals_expected'],
        'column_true_on_exactly_ten_rows': sum(1 for r in list(tables['cells'].values()) + list(appended.values())
                                               if r[COL_COUNTED] is True) == 10,
        'twins_bitwise_evidence_holds': cnt['twins']['holds'],
        'd_36686489_growth_test_with_five_non_clean': d366_ok,
        't6_passive_derived_equals_recorded_decomposition':
            add6['passive_arm_vs_coordinated_derived']['abs_difference_to_recorded_decomposition'] <= 1e-6,
        't11_R_rounds_to_0909_0934': t11doc['R_range_rounded'] == [0.909, 0.934],
        'paragraph_fragments_all_found_verbatim': frozen['paragraph_figure_check_summary']['all_fragments_found'],
    }
    failed += [k for k, v in checks.items() if v is not True]
    frozen['checks'] = checks
    frozen['failed_checks'] = sorted(set(failed))
    # the ten-count assertion
    assert cnt['n_counted'] == 10, f'at_or_above_0.95_tau_counted is true on {cnt["n_counted"]} certificates, not 10'

    # write the frozen JSON (content-hashed name)
    body = GRIO.dumps(frozen, indent=1, sort_keys=True).encode('utf-8') + b'\n'
    fz_sha = _sha_bytes(body)
    fz_name = f'frozen_step6_tables_v1_{fz_sha[:8]}.json'
    os.makedirs(os.path.join(REPO, out_dir, 'export'), exist_ok=False)
    written = {}

    def wr(rel, data, mode='xb'):
        p_ = os.path.join(REPO, out_dir, rel)
        with open(p_, mode) as handle:
            handle.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[os.path.join(out_dir, rel)] = _sha(os.path.join(out_dir, rel))
    wr(fz_name, body)
    frozen_rt = json.loads(body)
    # exports
    ex, t11val = build_exports(frozen_rt)
    files, facts = render(ex, t11val)
    for fn, txt in files.items():
        wr(os.path.join('export', fn), txt)
    export_ok = {n: (f['latex_ascii_only'] and f['latex_row_cell_counts_ok'] and f['latex_braces_balanced']
                     and f['csv_rows_parse']) for n, f in facts.items()}
    wr(os.path.join('export', 'README.md'), readme(fz_name, fz_sha, facts, t11doc))
    wr(os.path.join('export', 'paragraphs.md'), ptext)
    # the frozen Markdown
    md = [f'# Step 6 tables — FROZEN v1 (`{fz_name}`)', '',
          f'sha256 of the frozen JSON: `{fz_sha}`. Predecessor: `{PRED_JSON}` (sha256 `{PRED_SHA}`, commit '
          f'`{PRED_COMMIT}`). Package: `{PACKAGE_REL}` at `{PACKAGE_COMMIT}`. Closing reads: W159 `{W159_COMMIT}`. '
          f'Built by `{SCRIPT_REL}`; zero solves, pickle blocked. The JSON keeps full precision; the tables below are '
          'rounded as the export README states.', '',
          f"**Objective convention (every table):** {pred['objective_convention']}.", '',
          '## What W160 changed against the predecessor', '']
    md += [f'- {c}' for c in frozen['w160_changes']]
    md += ['', '## The ≥ 0.95 τ count (Addendum 65 ruling 2)', '',
           f"Counted: {cnt['n_counted']} — {', '.join(cnt['counted_cells'])}. Scope: {cnt['scope']}. Twins: "
           f"{UNIT_REF} = {' = '.join(TWINS_OF_UNIT)} (evidence holds: {cnt['twins']['holds']}). Beside: "
           f'{BESIDE_CELL} — {BESIDE_TEXT}.', '', '## Tables', '']
    for name, caption, cols, rows in ex:
        md += md_table(name, caption, cols, rows, t11val)
    md += ['## Paragraph figure check (paragraphs.md against this JSON, at the written precision)', '',
           '| id | written | table value | status | table reference | note |', '|---|---|---|---|---|---|']
    for c in fchk:
        md.append(f"| {c['id']} | {c['written']} | {c['table_value_at_written_precision']} | {c['status']} | "
                  f"{c['table_ref']} | {c['note'] or ''} |".replace('\n', ' '))
    md += ['', '## Checks', ''] + [f'- {k}: {v}' for k, v in checks.items()] + \
          [f'- export {n} (LaTeX ASCII / cell counts / braces; CSV parse): {v}' for n, v in export_ok.items()]
    if frozen['failed_checks'] or not all(export_ok.values()):
        md += ['', '**FAILED:** ' + '; '.join(frozen['failed_checks'] + [f'export {n}' for n, v in export_ok.items()
                                                                          if not v])]
    md_name = fz_name[:-5] + '.md'
    wr(md_name, '\n'.join(md) + '\n')
    if not all(export_ok.values()):
        failed.append('export structural checks')
    # guards, build record, manifest
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    pk = pickle_state()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    build = {'schema': 'p515_s53_w160_build_record_v1', 'utc': datetime.now(timezone.utc).isoformat(),
             'git_head': _git('rev-parse', 'HEAD'), 'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
             'script_committed_clean': script_clean, 'frozen_json': fz_name, 'frozen_json_sha256': fz_sha,
             'frozen_md': md_name, 'export_facts': facts, 'export_structural_ok': export_ok,
             'guards': guards, 'pickle_guard': pk, 'failed': sorted(set(failed)), 'exit_code': code,
             'wall_s': time.time() - t0}
    wr('w160_build_record.json', GRIO.dumps(build, indent=1, sort_keys=True) + '\n')
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for grp in all_inputs.values():
        for v in grp.values():
            man[v['path']] = v['sha256']
            if v.get('manifest'):
                man[v['manifest']] = v.get('manifest_file_sha256') or _sha(v['manifest'])
    wr('manifest_sha256.json', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[{tag}] frozen {out_dir}/{fz_name} sha256 {fz_sha}; md {md_name}; export {len(files) + 2} files')
    _log(f"[{tag}] >= 0.95 tau counted: {cnt['n_counted']} {cnt['counted_cells']}; equals the Addendum 65 ten: "
         f"{cnt['counted_equals_expected']}; twins evidence {cnt['twins']['holds']}")
    s = frozen['paragraph_figure_check_summary']
    _log(f"[{tag}] paragraph figure check: {s['n_checks']} checks, {s['n_match']} match, {s['n_mismatch']} MISMATCH "
         f"{s['mismatch_ids']}, {s['n_no_table_counterpart']} no table counterpart, {s['n_approximate']} approximate; "
         f"fragments all found {s['all_fragments_found']}")
    for c in fchk:
        if c['status'] != 'match':
            _log(f"[{tag}]   {c['id']} {c['status']}: written {c['written']!r}, table {c['table_value_at_written_precision']!r} "
                 f"({c['table_ref']}){' -- ' + c['note'] if c['note'] else ''}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    for n, v in export_ok.items():
        _log(f"[{tag}] export {n}: structural ok {v}; rows {facts[n]['n_rows']}; LaTeX cols {facts[n]['n_latex_columns']}"
             f"{'' if facts[n]['latex_ascii_only'] else ' NON-ASCII ' + repr(facts[n]['latex_non_ascii'])}")
    if failed:
        _log(f'[{tag}] FAILED: {sorted(set(failed))}')
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle {pk}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
