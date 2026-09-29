"""P5.15 W130 (Addendum 58, Benchmark and Ruling 1 bullets) -- benchmark addendum. ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Model LOADS are permitted (Planner W130): the settled coordinated x = 0 models are unpickled once (sha256 verified
first) and evaluated; nothing is built for solving and nothing is solved. Every other input is a committed,
hash-recorded JSON(L) record (pinned against the manifest that records it, and required git-tracked and clean).

OBJECTIVE CONVENTION (every value): Q = gross_operational_cost, settlement EXCLUDED, salvage excluded (x = 0 salvage 0;
the arms' salvage block -0.0). t_sum = the priced interface-consensus gap (t_tso_plus_t_dso_terminal);
Q_cc = Q + t_sum, a first-order diagnostic.

INSTANCES
  coordinated  SRP1 x = 0 (candidate 8435c718...), settled cycle-181 cell of campaign s53_w101_srp1_cont_x0, eval
               d110bd1a5977df1e..., certified_models.pkl sha256 99ab1070..., esso_models_s39_D.pkl sha256 118487e1...
               (the ESSO dict only provides the salvage block).  Q181 = 653,873,702.1876609.
  NRF arms     W122 (spec v5 bca69f97) phase-C (after the sequential pass) evaluations of the six *_r2 runs under
               w116_benchmark_nrf/, all on candidate 8435c718 (asserted).
  F2 pair      incumbent  5ca4f86c (y2030 n5 0.25/0.5, n7 1/3.5, m = 2; candidate 59757776), re-run as f2_incumbent
                          (campaign s53_w118_resettle_r2_f2_incumbent, eval 24c5ccb6, spec fc791891, evidence 075823cf),
                          UNCERTIFIED at the cap 281 by the gap clause; values at the cap.
               corner     f3aa335e = y2025 n7 0.75/3 (n7_4h_e3_m2, candidate fb21d822), the old certificate (S51 F2
                          ladder, cycle 160) W117 used in claim L:y2025__n7_p0.75_e3 ("F2 plan vs corner").

TASK 1 -- both NRF arms per block (12 blocks = year x representative day), by agent (TSO, DSO_5, DSO_7, DSO_9).
  arm value[a, s, agent, b] = W122 run (a, s) phase_C_sequential_pass.evaluation.block_components[label]
                              (label TSO|-|y|d, DSO|n|y|d; production _get_operational_recourse_block_components under
                              the common-Q pricing, weighted EUR);  best start of arm a = argmin_s arm_cost Q
                              (asserted: price_taker -> warm_from_certified, passive -> perturbed, as W130 names them).
  coord value[agent, b]     = uncoordinated_benchmark.evaluate_common_q(planning, Q181 models,
                              evaluation_curtailment_penalty=0, require_unchanged=True)['block_components'][label]
                              -- the SAME evaluation function the arms were evaluated with (it calls production's
                              _get_operational_recourse_block_components after verifying no pricing Param changes);
                              production's _get_operational_recourse_block_components is ALSO called directly and
                              the two asserted bitwise equal.
  VALIDATION  (a) coord TSO[b] vs w106 tso_coupling_check.json per_block[TSO|-|y|d].certified_coordinated
                  .block_gross_cost_weighted, 12 blocks: max |diff| and bitwise count;
              (b) sum of the 48 coord blocks vs Q181 (and evaluate_common_q gross_operational_cost vs Q181 bitwise);
              (c) cross-check, not required: coord blocks vs the in-run all-block capture of production's function at
                  cycle 181 (recourse_blocks_all.jsonl, last row, cycle 181), 48 blocks: max |diff|.
  benefit[a, agent, b]      = arm value[a, best, agent, b] - coord value[agent, b];  benefit total[b] = sum over agents.
  arm totals                = sum over the 48 blocks (= the arm's phase-C Q to rounding; the difference is recorded).

TASK 2 -- the two blocks where the W124 mechanism test did not hold (read from w124_block_energy_check.json
  blocks[*].verdict_R_total != 'H1'; asserted = {2025|Winter: outside_rule, 2025|Autumn: undetermined}).
  DSO multimodality band, per arm a and block b:
      D[a, s, b]       = sum_{n in 5,7,9} arm value[a, s, DSO_n, b]     (all three starts s of the arm's _r2 runs)
      band[a, b]       = max_s D[a, s, b] - min_s D[a, s, b]
      band_node[a,n,b] = the same per node (reported beside it)
      in total         = (i) sum_b band[a, b]  and  (ii) max_s sum_b D[a, s, b] - min_s sum_b D[a, s, b] (both stated)
  RULE (expert, Addendum 58; reading fixed here BEFORE any value is computed):
      B[b] = benefit total[price_taker, b] (price-taker NRF - coordinated, all four agents).
      PRIMARY band = band[price_taker, b] (the arm B is measured on); beside it the passive arm's band and the max of
      the two -- a verdict that differs between readings is listed.
      within_band                 |B| <= band
      beyond_band_negative        |B| >  band and B < 0: the coordinated point in that block is a worse local optimum
                                  of the joint problem than the arm's; the certified Q(x = 0) is an UPPER BOUND there;
                                  flagged as a limitation
      beyond_band_positive        |B| >  band and B > 0: not named by the expert's rule; reported as such
  Negative total benefit: every block (both arms) with benefit total < 0 is listed with its value.
  Context (not a verdict): the coordinated per-block terminal step (cycle 180 -> 181) of the in-run capture, and the
  coordinated per-block range (max - min, block total and per agent) over the certifying window [150, 181] of
  settling_decision.json, from the same in-run capture (recourse_blocks_all.jsonl) -- the coordinated side's own
  per-block resolution, which the expert's rule does not use.

TASK 3 -- F2 plan vs corner (Addendum 58 Ruling 1: "the F2 conclusion stands if the +102 kEUR margin exceeds
  3 x max(gap, slack) in both terms").  F = I + Q.
      incumbent i: Q_i = Q at cap (evaluation_record terminal_gross_operational_cost, cycle 281; = w118 summary Q_at_cap),
                   t_i = interface_settlement_detail t_tso_plus_t_dso_terminal (= w118 summary t_sum_at_cap, the
                         in-cycle value, to 1e-6 relative; both recorded),
                   s_i = Q_i - Q_N_old(181) (w118 summary s_signed; recomputed), I_i = S53 F2 certificate r1
                   final_incumbent.I.
      corner c:    Q_c, t_c from its own records (evaluation_record certified_cost / terminal_gross_operational_cost,
                   interface_settlement_detail t_tso_plus_t_dso_terminal), I_c = S53 F2 certificate r1
                   cached_box_neighbours row I_x_eur (cross-checked against W117 claim L:y2025__n7_p0.75_e3).
      margin_gross = (I_c + Q_c) - (I_i + Q_i)
      margin_cc    = (I_c + Q_c + t_c) - (I_i + Q_i + t_i)
      READING (stated):  gap = max(|t_c|, |t_i|)  (the larger |t_sum| of the two cells);  slack = |s_i|;
                         bar = 3 x max(gap, slack), the same bar for both terms.
      verdict per term: 'stands' iff margin > bar (margin positive = the plan cheaper than the corner), else 'pending'.
  Beside it (context only): W117's recorded d_Q / d_Qcc (both on the OLD incumbent certificate).

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w130_benchmark_addendum/):
  w130_benchmark_addendum.json, launch.log, manifest_sha256.json
Smoke (no file written): add --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w130_benchmark_addendum && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w130_benchmark_addendum.py \\
        > data/SRP1/Results/P515S53/w130_benchmark_addendum/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w130_benchmark_addendum.py --manifest
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W130 benchmark addendum zero-solve').install()

import gate_result_io as GRIO  # noqa: E402 -- stdlib only

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results')
S53 = os.path.join(RES, 'P515S53')
OUT_DIR = os.path.join(REPO, S53, 'w130_benchmark_addendum')
OUT_JSON = os.path.join(OUT_DIR, 'w130_benchmark_addendum.json')

# ---- coordinated (settled x = 0, cycle 181) -------------------------------------------------------------------------
X0_CAMPAIGN = os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0')
X0_EVAL = os.path.join(X0_CAMPAIGN, 'evals', 'd110bd1a5977df1e_x0')
X0_MANIFEST = os.path.join(X0_CAMPAIGN, 'campaign_manifest_sha256.json')
MODELS = {'path': os.path.join(X0_EVAL, 'certified_models.pkl'),
          'sha256': '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'}
ESSO = {'path': os.path.join(X0_EVAL, 'esso_models_s39_D.pkl'),
        'sha256': '118487e1853bbad176f324b1173df8c4198cb6c7709c25d7dce9f94ce4c199fa'}
X0_RECORD = os.path.join(X0_EVAL, 'evaluation_record.json')
X0_SETTLING = os.path.join(X0_EVAL, 'settling_decision.json')
X0_BLOCKS_ALL = os.path.join(X0_EVAL, 'recourse_blocks_all.jsonl')
Q181 = 653873702.1876609
X0_INSTANCE = {'case': 'SRP1', 'candidate': 'x = 0',
               'candidate_key': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57',
               'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',
               'campaign': 's53_w101_srp1_cont_x0', 'certification_cycle': 181,
               'certified_models_sha256': MODELS['sha256'], 'esso_models_sha256': ESSO['sha256']}

# ---- NRF arms (W122) ------------------------------------------------------------------------------------------------
NRF = os.path.join(S53, 'w116_benchmark_nrf')
ARMS = ('price_taker', 'passive')
STARTS = ('cold', 'warm_from_certified', 'perturbed')
BEST_DECLARED = {'price_taker': 'warm_from_certified', 'passive': 'perturbed'}   # as named by Planner task W130
TCC = os.path.join(S53, 'w106_uncoordinated_settled', 'tso_coupling_check', 'tso_coupling_check.json')
W124 = os.path.join(S53, 'w124_block_energy_check', 'w124_block_energy_check.json')
W124_EXPECTED = {'2025|Winter': 'outside_rule', '2025|Autumn': 'undetermined'}

# ---- F2 pair --------------------------------------------------------------------------------------------------------
INC_CAMPAIGN = os.path.join(S53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_f2_incumbent')
INC_EVAL = os.path.join(INC_CAMPAIGN, 'evals', '24c5ccb6f285219f_f2_incumbent')
INC_MANIFEST = os.path.join(INC_CAMPAIGN, 'campaign_manifest_sha256.json')
W118_SUMMARY = os.path.join(S53, 'w118_resettle', 'w118_resettle_summary.json')
W118_SUMMARY_MANIFEST = os.path.join(S53, 'w118_resettle', 'w118_resettle_summary_manifest_sha256.json')
CORNER_CAMPAIGN = os.path.join(RES, 'P515S51', 'campaign_s51_f2_ladder')
CORNER_EVAL = os.path.join(CORNER_CAMPAIGN, 'evals', 'f3aa335e6c1eda69_n7_4h_e3_m2')
CORNER_MANIFEST = os.path.join(CORNER_CAMPAIGN, 'campaign_manifest_sha256.json')
CERT = os.path.join(S53, 'campaign_s53_f2_certificate_r1', 'campaign_results.json')
CERT_MANIFEST = os.path.join(S53, 'campaign_s53_f2_certificate_r1', 'campaign_manifest_sha256.json')
W117 = os.path.join(S53, 'w117_triage_recompute', 'w117_triage_recompute.json')
W117_CLAIM = 'L:y2025__n7_p0.75_e3'
CORNER_LABEL = 'y2025__n7_p0.75_e3'
INC_ORIGINAL_KEY = '5ca4f86c3406c0424d3b63b8f39df4568f1be77f7f2d72642491539c0eb42663'
CORNER_KEY = 'f3aa335e6c1eda6950712aec00aff122d4578760f0dd02ff64996b95998eaace'
BAR_FACTOR = 3.0

NODES = ('5', '7', '9')
YEARS = ('2025', '2030', '2035')
DAYS = ('Winter', 'Spring', 'Summer', 'Autumn')
AGENTS = ('TSO', 'DSO_5', 'DSO_7', 'DSO_9')

READINGS = {
    'task2_band': 'band[a, b] = max_s - min_s over the arm three _r2 starts of D[a, s, b] = sum over nodes 5, 7, 9 of '
                  'the phase-C DSO block components; in total: (i) sum_b band[a, b], (ii) spread over starts of '
                  'sum_b D[a, s, b]',
    'task2_rule_primary_band': 'band[price_taker, b] (the arm the benefit is measured on); beside it band[passive, b] '
                               'and max of the two',
    'task2_compared_quantity': 'B[b] = total benefit (price-taker NRF best start - coordinated, all four agents); the '
                               'DSO-only benefit against the same band is reported beside it',
    'task3_gap': 'gap = max(|t_sum corner (old certificate)|, |t_sum incumbent at cap|)',
    'task3_slack': 'slack = |s| of the incumbent = |Q at cap - Q_N_old(181)|',
    'task3_bar': 'bar = 3 x max(gap, slack), the same bar in the gross and the Q_cc term',
    'task3_verdict': "per term: 'stands' iff margin > bar, else 'pending'",
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=REPO)


INPUTS = {}


def _pin(rel, manifest_rel=None, *, tracked=True, declared=None):
    """sha256 of an input against the manifest that hash-records it (default: manifest_sha256.json in its directory),
    and against a declared sha when given; if `tracked`, it must also be git-tracked and clean against HEAD."""
    manifest_rel = manifest_rel or os.path.join(os.path.dirname(rel), 'manifest_sha256.json')
    now = _sha(os.path.join(REPO, rel))
    pinned = _load(manifest_rel).get(rel)
    if pinned is None:
        raise RuntimeError(f'{rel}: not hash-recorded in {manifest_rel}')
    if pinned != now:
        raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned} ({manifest_rel})')
    if declared is not None and declared != now:
        raise RuntimeError(f'{rel}: sha256 {now} != declared {declared}')
    entry = {'sha256': now, 'manifest': manifest_rel}
    if tracked:
        is_tracked = _git('ls-files', '--error-unmatch', rel).returncode == 0
        clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=REPO).returncode == 0
        if not (is_tracked and clean):
            raise RuntimeError(f'{rel}: tracked {is_tracked}, clean {clean}')
        entry['git_last_commit'] = _git('log', '-1', '--format=%H', '--', rel).stdout.strip()
    else:
        entry['git'] = 'not committed (hash-recorded in its campaign manifest only)'
    INPUTS[rel] = entry


def _label(agent, y, d):
    return f'TSO|-|{y}|{d}' if agent == 'TSO' else f'DSO|{agent.split("_")[1]}|{y}|{d}'


def _run_rel(arm, start):
    run = f'nrf_arm_{arm}_{start}_r2'
    return run, os.path.join(NRF, run, f'{run}.json')


# ======================================================================================================================
def task3_f2():
    """F2 plan (incumbent 5ca4f86c at the cap of its re-run) vs the corner f3aa335e (old certificate). Records only."""
    for name in ('evaluation_record.json', 'interface_settlement_detail_s31c.json', 'per_cycle_record.jsonl'):
        _pin(os.path.join(INC_EVAL, name), INC_MANIFEST)
        _pin(os.path.join(CORNER_EVAL, name), CORNER_MANIFEST)
    _pin(W118_SUMMARY, W118_SUMMARY_MANIFEST)
    _pin(CERT, CERT_MANIFEST)
    _pin(W117)

    # incumbent at the cap
    inc_rec = _load(os.path.join(INC_EVAL, 'evaluation_record.json'))
    inc_det = _load(os.path.join(INC_EVAL, 'interface_settlement_detail_s31c.json'))
    with open(os.path.join(REPO, INC_EVAL, 'per_cycle_record.jsonl')) as handle:
        inc_last = [json.loads(line) for line in handle if line.strip()][-1]
    summ = _load(W118_SUMMARY)['reports']['f2_incumbent']
    assert summ['original_eval_key'] == INC_ORIGINAL_KEY, summ['original_eval_key']
    assert summ['eval_key'] == inc_rec['eval_key'] and summ['candidate_key'] == inc_rec['candidate_key']
    assert summ['status'] == 'uncertified' and summ['reasons'] == ['gap_clause'], (summ['status'], summ['reasons'])
    q_i = inc_rec['terminal_gross_operational_cost']
    assert inc_rec['cycles_run'] == summ['k_cap'] == inc_last['cycle'] == inc_det['cycles_run'], 'cap cycle'
    assert q_i == summ['Q_at_cap'] == inc_last['gross_operational_cost'], (q_i, summ['Q_at_cap'])
    t_i = inc_det['t_tso_plus_t_dso_terminal']
    # the summary's in-cycle t_sum and the terminal detail agree to rounding (not bitwise); the detail value is used
    # (the same source as the corner and as W117), the difference recorded
    assert abs(t_i - summ['t_sum_at_cap']) <= 1e-6 * max(1.0, abs(t_i)), (t_i, summ['t_sum_at_cap'])
    q_i_old = summ['Q_N_old']
    s_i = q_i - q_i_old
    assert s_i == summ['s_signed'], (s_i, summ['s_signed'])

    # corner (old certificate)
    c_rec = _load(os.path.join(CORNER_EVAL, 'evaluation_record.json'))
    c_det = _load(os.path.join(CORNER_EVAL, 'interface_settlement_detail_s31c.json'))
    with open(os.path.join(REPO, CORNER_EVAL, 'per_cycle_record.jsonl')) as handle:
        c_last = [json.loads(line) for line in handle if line.strip()][-1]
    assert c_rec['eval_key'] == CORNER_KEY and c_rec['status'] == 'certified'
    q_c = c_rec['terminal_gross_operational_cost']
    assert q_c == c_rec['certified_cost'] == c_last['gross_operational_cost'], 'corner Q'
    assert c_rec['certification_cycle'] == c_rec['cycles_run'] == c_last['cycle'] == c_det['cycles_run'], 'corner k'
    t_c = c_det['t_tso_plus_t_dso_terminal']

    # investment costs (the certificate campaign's own records)
    cert = _load(CERT)
    fi = cert['final_incumbent']
    assert fi['eval_key'] == INC_ORIGINAL_KEY and fi['Q'] == q_i_old, (fi['eval_key'], fi['Q'], q_i_old)
    rows = [r for r in cert['termination_certificate']['cached_box_neighbours']['rows'] if r['label'] == CORNER_LABEL]
    assert len(rows) == 1 and rows[0]['eval_key'] == CORNER_KEY and rows[0]['Q_eur'] == q_c, rows
    i_i, i_c = fi['I'], rows[0]['I_x_eur']

    # W117 cross-check (old incumbent certificate)
    w117 = [c for c in _load(W117)['claims'] if c['claim_id'] == W117_CLAIM]
    assert len(w117) == 1, W117_CLAIM
    w117 = w117[0]
    assert w117['ref']['eval_key'] == INC_ORIGINAL_KEY and w117['other']['eval_key'] == CORNER_KEY
    assert w117['other']['Q'] == q_c and w117['other']['t_sum'] == t_c, 'W117 corner values'
    assert abs(w117['I_other'] - i_c) <= 1e-6 and abs(w117['I_ref'] - i_i) <= 1e-6, 'W117 I values'

    margin_gross = (i_c + q_c) - (i_i + q_i)
    margin_cc = (i_c + q_c + t_c) - (i_i + q_i + t_i)
    gap = max(abs(t_c), abs(t_i))
    slack = abs(s_i)
    bar = BAR_FACTOR * max(gap, slack)
    old_margin_gross = (i_c + q_c) - (i_i + q_i_old)
    return {
        'reading': {k: READINGS[k] for k in ('task3_gap', 'task3_slack', 'task3_bar', 'task3_verdict')},
        'objective_convention': 'F = I + Q, Q = gross_operational_cost (settlement excluded, salvage excluded); '
                                'Q_cc = Q + t_sum',
        'incumbent': {'role': 'F2 plan', 'original_eval_key': INC_ORIGINAL_KEY, 'rerun_eval_key': inc_rec['eval_key'],
                      'candidate_key': inc_rec['candidate_key'], 'candidate_canonical': inc_rec['candidate_canonical'],
                      'campaign': inc_rec['campaign_id'], 'status_settling_rule_v2': summ['status'],
                      'reasons': summ['reasons'], 'cap_cycle': inc_rec['cycles_run'],
                      'I': i_i, 'I_source': f'{CERT} final_incumbent.I',
                      'Q_at_cap': q_i, 'Q_source': f'{INC_EVAL}/evaluation_record.json terminal_gross_operational_cost '
                                                  '(= w118 summary Q_at_cap = per_cycle_record last row)',
                      't_sum_at_cap': t_i, 't_source': f'{INC_EVAL}/interface_settlement_detail_s31c.json '
                                                       't_tso_plus_t_dso_terminal (= w118 summary t_sum_at_cap)',
                      't_sum_at_cap_w118_summary_in_cycle': summ['t_sum_at_cap'],
                      't_detail_minus_summary': t_i - summ['t_sum_at_cap'],
                      'Q_cc_at_cap': q_i + t_i, 'Q_N_old_181': q_i_old, 's_signed': s_i,
                      'band_width_last_60': summ['band_width'], 'drift_mean_dQ_last_25': summ['drift_rate_mean_dQ_last_25'],
                      'terminal_step_abs': summ['terminal_step_abs'], 'terminal_step_over_EPS0': summ['terminal_step_over_EPS0'],
                      'production_record_status_field': inc_rec['status']},
        'corner': {'role': 'corner', 'label': CORNER_LABEL, 'eval_key': CORNER_KEY, 'eval_label': c_rec.get('candidate_label'),
                   'candidate_key': c_rec['candidate_key'], 'candidate_canonical': c_rec['candidate_canonical'],
                   'campaign': c_rec.get('campaign_id'), 'certification_cycle': c_rec['certification_cycle'],
                   'I': i_c, 'I_source': f'{CERT} termination_certificate.cached_box_neighbours.rows[{CORNER_LABEL}].I_x_eur',
                   'Q': q_c, 'Q_source': f'{CORNER_EVAL}/evaluation_record.json certified_cost',
                   't_sum': t_c, 't_source': f'{CORNER_EVAL}/interface_settlement_detail_s31c.json t_tso_plus_t_dso_terminal',
                   'Q_cc': q_c + t_c, 'settling_slack': 'not measured (old certificate, not re-settled)'},
        'margin_gross': margin_gross, 'margin_Q_cc': margin_cc,
        'gap': gap, 'gap_from': 'incumbent' if abs(t_i) >= abs(t_c) else 'corner', 'slack': slack, 'bar': bar,
        'margin_gross_over_bar': margin_gross / bar, 'margin_cc_over_bar': margin_cc / bar,
        'verdict_gross': 'stands' if margin_gross > bar else 'pending',
        'verdict_Q_cc': 'stands' if margin_cc > bar else 'pending',
        'context_w117_old_incumbent_certificate': {
            'claim_id': W117_CLAIM, 'd_Q': w117['d_Q'], 'd_Qcc': w117['d_Qcc'], 'bar_R1': w117['bar_R1'],
            'verdict_R1': w117['verdict_R1'], 'recomputed_old_margin_gross': old_margin_gross,
            'recomputed_minus_w117_d_Q': old_margin_gross - w117['d_Q']},
    }


# ======================================================================================================================
def coordinated_blocks():
    """Unpickle the Q181 models (sha verified first) and evaluate them with the arms' evaluation function."""
    _pin(MODELS['path'], X0_MANIFEST, tracked=False, declared=MODELS['sha256'])
    _pin(ESSO['path'], X0_MANIFEST, tracked=False, declared=ESSO['sha256'])
    _log(f"certified_models.pkl sha256 verified {MODELS['sha256']}; esso_models_s39_D.pkl {ESSO['sha256']}")
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    import p56a_oracle as O

    planning = O.load_baseline()['planning']
    _log('baseline planning loaded')
    with open(os.path.join(REPO, MODELS['path']), 'rb') as handle:
        payload = pickle.load(handle)
    with open(os.path.join(REPO, ESSO['path']), 'rb') as handle:
        esso = pickle.load(handle)
    models = {'tso': payload['tso'], 'dso': payload['dso'], 'esso': esso}
    _log('certified models unpickled')
    ev = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=0.0, require_unchanged=True)
    direct = srp._get_operational_recourse_block_components(planning, models)
    direct = {UB.block_label(k, n, y, d): v for (k, n, y, d), v in direct.items() if k != 'SALVAGE'}
    mism = {k: (ev['block_components'][k], direct.get(k)) for k in ev['block_components']
            if ev['block_components'][k] != direct.get(k)}
    if mism or set(direct) != set(ev['block_components']):
        raise RuntimeError(f'evaluate_common_q vs direct production blocks differ: {list(mism)[:5]}')
    return {'block_components': ev['block_components'], 'gross_operational_cost': ev['gross_operational_cost'],
            'gross_operational_cost_hex': ev['gross_operational_cost_hex'], 'salvage_block': ev['salvage_block'],
            'pricing_changed_by_evaluation': ev['pricing_changed_by_evaluation'],
            'terminal_salvage_value': ev['recourse_components'].get('terminal_salvage_value'),
            'direct_production_call_bitwise_equal': True,
            'function': 'uncoordinated_benchmark.evaluate_common_q(evaluation_curtailment_penalty=0, '
                        'require_unchanged=True) -> srp._get_operational_recourse_block_components; direct call '
                        'asserted bitwise equal'}


def by_agent(bc, y, d):
    return {a: bc[_label(a, y, d)] for a in AGENTS}


def main():
    t0 = time.time()
    dry = '--dry' in sys.argv
    if not dry and os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')

    # ---- records --------------------------------------------------------------------------------------------------
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for start in STARTS:
            run, rel = _run_rel(arm, start)
            _pin(rel)
            res = _load(rel)
            assert res['instance']['candidate_key'] == X0_INSTANCE['candidate_key'], (run, res['instance'])
            assert res['arm'] == arm and res['start'] == start, (run, res['arm'], res['start'])
            ev = res['phase_C_sequential_pass']['evaluation']
            assert res['arm_cost']['source'] == 'phase_C_sequential_pass'
            assert res['arm_cost']['gross_operational_cost'] == ev['gross_operational_cost']
            bc = ev['block_components']
            assert len(bc) == 48, (run, len(bc))
            arms[arm][start] = {'run': run, 'q': ev['gross_operational_cost'], 'bc': bc,
                                'salvage_block': ev.get('salvage_block'),
                                'block_sum_minus_q': sum(bc.values()) - ev['gross_operational_cost']}
    best = {arm: min(STARTS, key=lambda s: arms[arm][s]['q']) for arm in ARMS}
    assert best == BEST_DECLARED, best
    _log(f'arms read; best starts {best}')
    _pin(TCC)
    tcc = _load(TCC)
    assert tcc['instance']['candidate_key'] == X0_INSTANCE['candidate_key']
    _pin(W124)
    w124 = _load(W124)
    not_h1 = {b: v['verdict_R_total'] for b, v in w124['blocks'].items() if v['verdict_R_total'] != 'H1'}
    assert not_h1 == W124_EXPECTED, not_h1
    for rel in (X0_RECORD, X0_SETTLING, X0_BLOCKS_ALL):
        _pin(rel, X0_MANIFEST)
    x0_rec = _load(X0_RECORD)
    assert x0_rec['candidate_key'] == X0_INSTANCE['candidate_key'] and x0_rec['eval_key'] == X0_INSTANCE['eval_key']
    assert _load(X0_SETTLING)['Q_k_star'] == Q181
    settling = _load(X0_SETTLING)
    window = settling['window']
    assert settling['k_star'] == 181 and window[1] == 181, (settling['k_star'], window)
    with open(os.path.join(REPO, X0_BLOCKS_ALL)) as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    cap_row = all_rows[-1]
    win_rows = [r for r in all_rows if window[0] <= r['cycle'] <= window[1]]
    assert [r['cycle'] for r in win_rows] == list(range(window[0], window[1] + 1)), 'window cycles'
    win_vals = {}
    for r in win_rows:
        for x in r['blocks']:
            if x['agent'] == 'SALVAGE':
                continue
            key = f"{x['year']}|{x['day']}"
            ag = 'TSO' if x['agent'] == 'TSO' else f"DSO_{x['node_id']}"
            win_vals.setdefault(key, {}).setdefault(r['cycle'], {})[ag] = x['value']
    coord_window_range = {}
    for key, per_cycle in win_vals.items():
        tot = [sum(v.values()) for v in per_cycle.values()]
        coord_window_range[key] = {'block_total': max(tot) - min(tot),
                                   'per_agent': {a: max(v[a] for v in per_cycle.values())
                                                 - min(v[a] for v in per_cycle.values()) for a in AGENTS}}
    assert cap_row['cycle'] == 181 and cap_row['gross_operational_cost'] == Q181, cap_row['cycle']
    inrun = {_label('TSO' if r['agent'] == 'TSO' else f"DSO_{r['node_id']}", r['year'], r['day']): r['value']
             for r in cap_row['blocks'] if r['agent'] != 'SALVAGE'}
    inrun_step = {_label('TSO' if r['agent'] == 'TSO' else f"DSO_{r['node_id']}", r['year'], r['day']): r['delta']
                  for r in cap_row['deltas_vs_previous_cycle'] if r['agent'] != 'SALVAGE'}

    f2 = task3_f2()
    _log(f"F2: margin gross {f2['margin_gross']:.2f}, Q_cc {f2['margin_Q_cc']:.2f}, bar {f2['bar']:.2f} -> "
         f"{f2['verdict_gross']} / {f2['verdict_Q_cc']}")

    # ---- the coordinated models (model load, zero solves) ---------------------------------------------------------
    co = coordinated_blocks()
    cbc = co['block_components']
    assert len(cbc) == 48 and set(cbc) == set(arms['price_taker']['cold']['bc']), 'label sets'

    # ---- validation -----------------------------------------------------------------------------------------------
    tso_rows = {}
    for y in YEARS:
        for d in DAYS:
            lab = _label('TSO', y, d)
            ref = tcc['per_block'][lab]['certified_coordinated']['block_gross_cost_weighted']
            tso_rows[lab] = {'models': cbc[lab], 'tso_coupling_check': ref, 'diff': cbc[lab] - ref,
                             'bitwise': cbc[lab] == ref}
    block_sum = sum(cbc.values())
    validation = {
        'tso_vs_tso_coupling_check': {
            'source': f'{TCC} per_block[TSO|-|y|d].certified_coordinated.block_gross_cost_weighted',
            'n_blocks': len(tso_rows), 'n_bitwise_equal': sum(r['bitwise'] for r in tso_rows.values()),
            'max_abs_diff_eur': max(abs(r['diff']) for r in tso_rows.values()), 'per_block': tso_rows,
            'total_models': sum(r['models'] for r in tso_rows.values()),
            'total_tso_coupling_check': tcc['tn_cost_weighted_totals']['certified_coordinated']},
        'total_vs_q181': {'sum_48_blocks': block_sum, 'Q181': Q181, 'diff_eur': block_sum - Q181,
                          'rel_diff': (block_sum - Q181) / Q181,
                          'evaluate_common_q_gross': co['gross_operational_cost'],
                          'evaluate_common_q_gross_bitwise_Q181': co['gross_operational_cost'] == Q181,
                          'salvage_block': co['salvage_block']},
        'cross_check_in_run_capture_cycle_181': {
            'source': f'{X0_BLOCKS_ALL} last row (cycle 181) blocks[*].value',
            'n_blocks': len(inrun), 'n_bitwise_equal': sum(cbc[k] == inrun[k] for k in inrun),
            'max_abs_diff_eur': max(abs(cbc[k] - inrun[k]) for k in inrun)},
        'pricing_changed_by_evaluation': co['pricing_changed_by_evaluation'],
    }
    validation['passed'] = (validation['tso_vs_tso_coupling_check']['max_abs_diff_eur'] <= 1e-6
                            and abs(validation['total_vs_q181']['diff_eur']) <= 1e-6
                            and validation['total_vs_q181']['evaluate_common_q_gross_bitwise_Q181']
                            and not co['pricing_changed_by_evaluation'])
    _log(f"validation: TSO {validation['tso_vs_tso_coupling_check']['n_bitwise_equal']}/12 bitwise, max "
         f"{validation['tso_vs_tso_coupling_check']['max_abs_diff_eur']!r}; total - Q181 "
         f"{validation['total_vs_q181']['diff_eur']!r}; in-run capture "
         f"{validation['cross_check_in_run_capture_cycle_181']['n_bitwise_equal']}/48 bitwise, max "
         f"{validation['cross_check_in_run_capture_cycle_181']['max_abs_diff_eur']!r}; passed {validation['passed']}")

    # ---- task 1 ---------------------------------------------------------------------------------------------------
    blocks = {}
    totals = {'coordinated': {a: 0.0 for a in AGENTS}}
    for arm in ARMS:
        totals[f'{arm}_best'] = {a: 0.0 for a in AGENTS}
        totals[f'benefit_{arm}'] = {a: 0.0 for a in AGENTS}
    for y in YEARS:
        for d in DAYS:
            b = f'{y}|{d}'
            coord = by_agent(cbc, y, d)
            row = {'coordinated': {**coord, 'total': sum(coord.values())},
                   'coordinated_range_over_certifying_window': coord_window_range[b],
                   'coordinated_terminal_step_cycle_180_181': {a: inrun_step[_label(a, y, d)] for a in AGENTS}}
            for a in AGENTS:
                totals['coordinated'][a] += coord[a]
            for arm in ARMS:
                vals = by_agent(arms[arm][best[arm]]['bc'], y, d)
                ben = {a: vals[a] - coord[a] for a in AGENTS}
                row[f'{arm}_best'] = {**vals, 'total': sum(vals.values())}
                row[f'benefit_{arm}'] = {**ben, 'total': sum(ben.values()), 'DSO': ben['DSO_5'] + ben['DSO_7'] + ben['DSO_9']}
                for a in AGENTS:
                    totals[f'{arm}_best'][a] += vals[a]
                    totals[f'benefit_{arm}'][a] += ben[a]
                # task 2 band
                per_start = {s: by_agent(arms[arm][s]['bc'], y, d) for s in STARTS}
                dso = {s: per_start[s]['DSO_5'] + per_start[s]['DSO_7'] + per_start[s]['DSO_9'] for s in STARTS}
                row[f'band_{arm}'] = {
                    'DSO': max(dso.values()) - min(dso.values()),
                    'DSO_per_start': dso,
                    'per_node': {f'DSO_{n}': max(per_start[s][f'DSO_{n}'] for s in STARTS)
                                 - min(per_start[s][f'DSO_{n}'] for s in STARTS) for n in NODES},
                    'TSO_context': max(per_start[s]['TSO'] for s in STARTS) - min(per_start[s]['TSO'] for s in STARTS),
                    'block_total_context': max(sum(per_start[s].values()) for s in STARTS)
                    - min(sum(per_start[s].values()) for s in STARTS)}
            blocks[b] = row
    for key in list(totals):
        totals[key]['total'] = sum(totals[key][a] for a in AGENTS)
    for arm in ARMS:
        totals[f'benefit_{arm}']['DSO'] = sum(totals[f'benefit_{arm}'][f'DSO_{n}'] for n in NODES)
    arm_q = {arm: {s: arms[arm][s]['q'] for s in STARTS} for arm in ARMS}
    totals_check = {arm: {'sum_48_blocks_best': totals[f'{arm}_best']['total'], 'phase_C_Q_best': arm_q[arm][best[arm]],
                          'diff': totals[f'{arm}_best']['total'] - arm_q[arm][best[arm]],
                          'claim_arm_minus_Q181': arm_q[arm][best[arm]] - Q181}
                    for arm in ARMS}

    # ---- task 2 ---------------------------------------------------------------------------------------------------
    band_totals = {}
    for arm in ARMS:
        sums = {s: sum(sum(by_agent(arms[arm][s]['bc'], y, d)[f'DSO_{n}'] for n in NODES) for y in YEARS for d in DAYS)
                for s in STARTS}
        band_totals[arm] = {'sum_of_per_block_bands': sum(blocks[b][f'band_{arm}']['DSO'] for b in blocks),
                            'spread_of_total_DSO_over_starts': max(sums.values()) - min(sums.values()),
                            'total_DSO_per_start': sums,
                            'Q_per_start': arm_q[arm], 'Q_band': max(arm_q[arm].values()) - min(arm_q[arm].values())}

    def verdict(benefit, band):
        if abs(benefit) <= band:
            return 'within_band'
        return 'beyond_band_negative' if benefit < 0 else 'beyond_band_positive'

    two = {}
    for b, w124_verdict in W124_EXPECTED.items():
        row = blocks[b]
        B = row['benefit_price_taker']['total']
        B_dso = row['benefit_price_taker']['DSO']
        bands = {'price_taker (primary)': row['band_price_taker']['DSO'], 'passive': row['band_passive']['DSO'],
                 'max_of_both': max(row['band_price_taker']['DSO'], row['band_passive']['DSO'])}
        verdicts = {k: verdict(B, v) for k, v in bands.items()}
        two[b] = {'w124_verdict': w124_verdict, 'benefit_total': B, 'benefit_by_agent': row['benefit_price_taker'],
                  'bands': bands, 'band_per_node_price_taker': row['band_price_taker']['per_node'],
                  'band_per_node_passive': row['band_passive']['per_node'],
                  'abs_benefit_over_primary_band': (abs(B) / bands['price_taker (primary)']
                                                    if bands['price_taker (primary)'] > 0 else None),
                  'verdicts': verdicts, 'readings_differ': len(set(verdicts.values())) > 1,
                  'dso_only_benefit': B_dso, 'dso_only_verdicts': {k: verdict(B_dso, v) for k, v in bands.items()},
                  'limitation_flag_upper_bound': verdicts['price_taker (primary)'] == 'beyond_band_negative',
                  'coordinated_terminal_step_block_total': sum(row['coordinated_terminal_step_cycle_180_181'].values()),
                  'coordinated_block_range_over_certifying_window': coord_window_range[b],
                  'abs_benefit_over_coordinated_window_range': (
                      abs(B) / coord_window_range[b]['block_total'] if coord_window_range[b]['block_total'] > 0
                      else None)}
    negatives = {arm: {b: {'benefit_total': blocks[b][f'benefit_{arm}']['total'],
                           'band_DSO_own_arm': blocks[b][f'band_{arm}']['DSO']}
                       for b in blocks if blocks[b][f'benefit_{arm}']['total'] < 0} for arm in ARMS}
    min_benefit = {arm: min(((blocks[b][f'benefit_{arm}']['total'], b) for b in blocks)) for arm in ARMS}

    guard_failures = _GUARD.verify(0)
    out = {
        'stage': 'P5.15 W130 (Addendum 58) -- benchmark addendum: both NRF arms per block by agent, the two '
                 'non-H1 blocks against the DSO multimodality band, the F2 plan-vs-corner check (zero solves; one '
                 'model load)',
        'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').stdout.strip(), 'script_sha256': _sha(THIS),
        'interpreter': sys.executable, 'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
        'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED, salvage excluded; block values are '
                                'weighted EUR (admm block weight); benefit = arm - coordinated (positive = coordination '
                                'cheaper)',
        'instance': {'coordinated': X0_INSTANCE, 'arms_candidate_key': X0_INSTANCE['candidate_key'],
                     'arm_runs': {arm: {s: arms[arm][s]['run'] for s in STARTS} for arm in ARMS},
                     'f2_incumbent': {k: f2['incumbent'][k] for k in ('original_eval_key', 'rerun_eval_key',
                                                                      'candidate_key')},
                     'f2_corner': {k: f2['corner'][k] for k in ('eval_key', 'candidate_key')}},
        'readings': READINGS,
        'inputs_sha256': INPUTS,
        'coordinated_evaluation': {k: co[k] for k in ('gross_operational_cost', 'gross_operational_cost_hex',
                                                      'salvage_block', 'terminal_salvage_value',
                                                      'direct_production_call_bitwise_equal', 'function')},
        'validation': validation,
        'task1': {'best_start': best, 'arm_Q_per_start': arm_q, 'per_block': blocks, 'totals': totals,
                  'totals_check': totals_check,
                  'arm_block_sum_minus_q_per_run': {arm: {s: arms[arm][s]['block_sum_minus_q'] for s in STARTS}
                                                    for arm in ARMS}},
        'task2': {'band_totals': band_totals, 'two_blocks': two, 'negative_total_benefit_blocks': negatives,
                  'min_block_benefit': {arm: {'block': v[1], 'benefit_total': v[0]} for arm, v in min_benefit.items()}},
        'task3': f2,
        'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures, 'counts': dict(_GUARD.counts)},
        'wall_s': time.time() - t0,
    }
    if dry:
        GRIO.dumps(out, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log('--dry: nothing written')
    else:
        with open(OUT_JSON, 'x') as handle:
            GRIO.dump(out, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
        _log(f'wrote {OUT_JSON} sha256 {_sha(OUT_JSON)}')

    # ---- log tables -------------------------------------------------------------------------------------------------
    for arm in ARMS:
        _log(f'--- {arm} (best start {best[arm]}) vs coordinated, EUR weighted; benefit = arm - coordinated ---')
        _log('block        ' + ' '.join(f'{a:>16s}' for a in AGENTS) + f"{'total':>16s} | "
             + ' '.join(f'{"b_" + a:>14s}' for a in AGENTS) + f"{'b_total':>16s} {'band_DSO':>12s}")
        for b, row in blocks.items():
            v, bn = row[f'{arm}_best'], row[f'benefit_{arm}']
            _log(f'{b:12s} ' + ' '.join(f'{v[a]:16.2f}' for a in AGENTS) + f"{v['total']:16.2f} | "
                 + ' '.join(f'{bn[a]:14.2f}' for a in AGENTS) + f"{bn['total']:16.2f} {row[f'band_{arm}']['DSO']:12.4f}")
        t, bt = totals[f'{arm}_best'], totals[f'benefit_{arm}']
        _log(f"{'TOTAL':12s} " + ' '.join(f'{t[a]:16.2f}' for a in AGENTS) + f"{t['total']:16.2f} | "
             + ' '.join(f'{bt[a]:14.2f}' for a in AGENTS) + f"{bt['total']:16.2f}")
    _log('--- coordinated (Q181 models) ---')
    for b, row in blocks.items():
        _log(f'{b:12s} ' + ' '.join(f"{row['coordinated'][a]:16.2f}" for a in AGENTS) + f"{row['coordinated']['total']:16.2f}")
    _log(f"{'TOTAL':12s} " + ' '.join(f"{totals['coordinated'][a]:16.2f}" for a in AGENTS)
         + f"{totals['coordinated']['total']:16.2f}")
    _log(f'band totals {json.dumps(band_totals)}')
    for b, v in two.items():
        _log(f"two blocks {b}: B {v['benefit_total']:.2f} bands {v['bands']} verdicts {v['verdicts']} "
             f"dso-only {v['dso_only_benefit']:.2f} {v['dso_only_verdicts']}; coordinated window range "
             f"{json.dumps(v['coordinated_block_range_over_certifying_window'])}; terminal step "
             f"{v['coordinated_terminal_step_block_total']:.4f}")
    _log(f'negative total-benefit blocks {negatives}; min {min_benefit}')
    _log(f"F2 {json.dumps({k: f2[k] for k in ('margin_gross', 'margin_Q_cc', 'gap', 'slack', 'bar', 'verdict_gross', 'verdict_Q_cc')})}")
    _log(f'guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {time.time() - t0:.1f} s')
    return 0 if not guard_failures and validation['passed'] else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    with open(OUT_JSON) as handle:
        res = json.load(handle)
    for rel, v in res['inputs_sha256'].items():
        suffix = ' (input)' if 'git_last_commit' in v else ' (hash-recorded input, not committed)'
        entries[rel + suffix] = v['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else main())
