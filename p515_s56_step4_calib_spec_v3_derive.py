"""P5.15 S56 step 4 CALIBRATION RE-RUN: derive the frozen spec v3 from spec v2 7ca210fb (zero solves, reads only).

Why: Expert handback 3, decision 1 (a), APPROVED 2026-11-23 (TASKS.md "Expert rulings on handback 3" item 1): re-run
the calibration arm V-1 with production's certification stop disabled, to a cycle cap declared before launch (A 140,
B 320), the trajectory to the old exit (46 / 107 cycles) reproducing step 4 main bitwise; frozen decision rule; the
interface V upper-bound multipliers captured. The precedent is `p515_s56_step4_main_spec_v2_derive.py` (v1 -> v2):
v3 = v2 loaded verbatim, then ONLY the key paths listed in `CHANGED_TOP` / `CHANGED_I` are set, added or removed; a
key-path diff checks that nothing else changed (`UNCHANGED_MUST_HOLD` below are compared equal explicitly).
Serialisation as v1 / v2 (json.dumps indent 1, sort_keys, default=str); the file name carries the first 8 hex digits
of the text's sha256. Predecessor: v2 (path, sha256).
Usage: <canonical python> p515_s56_step4_calib_spec_v3_derive.py
"""
import hashlib
import json
import os
import sys
from copy import deepcopy

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
V2 = os.path.join(RES, 'step4_main_v2', 'frozen_step4_main_spec_v2_7ca210fb.json')
V2_MANIFEST = os.path.join(RES, 'step4_main_v2', 'step4_main_v2_manifest_sha256.json')
V3_DIR = os.path.join(RES, 'step4_calib_v1')
V3_PREFIX = 'frozen_step4_calib_spec_v3_'
TAG = 'v3'
PHASE = 'calib'
CAP = {'A': 140, 'B': 320}
K_OLD = {'A': 46, 'B': 107}
HARNESS_COMMIT = '091b7745'
CHECKS_DIR = os.path.join(V3_DIR, 'harness_checks')
CHECKS_COMPARE = os.path.join(CHECKS_DIR, 'stub', 'compare_20261123T185032.json')
CHECKS_NOKEY = os.path.join(CHECKS_DIR, 'nokey', 'nokey_check.json')
CHECKS_MANIFEST = os.path.join(V3_DIR, 'harness_checks_manifest_sha256.json')
REF_EVAL = {'A': os.path.join(RES, 'step4_main_v2', 'campaign_s56s4main_v2_main_A', 'evals',
                              'a918ba210411acec_x0_v1_main'),
            'B': os.path.join(RES, 'step4_main_v2', 'campaign_s56s4main_v2_main_B', 'evals',
                              '917aa06e2a2e1b55_x0_v1_main')}
PREFIX_FILES = ('per_cycle_record.jsonl', 's56_cycle_capture.jsonl', 'interface_duals_per_cycle.jsonl',
                'network_ipopt_solve_records_append.jsonl')
CITED = {
    'handback_3': 'P5_15_S56_STEP4_EXPERT_HANDBACK_3.md',
    'step4_result_report': 'P5_15_S56_STEP4_RESULT_REPORT.md',
    'step4_postdiag_report': 'P5_15_S56_STEP4_POSTDIAG_REPORT.md',
    'step4_postdiag_spec': os.path.join(RES, 'step4_postdiag_v1', 'frozen_step4_postdiag_spec_v1_72f4d80e.json'),
    'step4_postdiag_results': os.path.join(RES, 'step4_postdiag_v1', 'postdiag_results_20261123T170607.json'),
    'step4_main_v2_spec': V2,
    'step4_main_v2_manifest': V2_MANIFEST,
    'step4_main_evaluation': os.path.join(RES, 'step4_main_v2', 'step4_main_evaluation_20261123T163553.json'),
    'harness_checks_manifest': CHECKS_MANIFEST,
    'harness_checks_nokey': CHECKS_NOKEY,
    'harness_checks_compare': CHECKS_COMPARE,
    'evaluator_step4': 'p515_s56_step4_main_evaluate.py',
    'postdiag_script': 'p515_s56_step4_postdiag.py',
}
UNCHANGED_MUST_HOLD = ('ii_central_and_allowance', 'iii_tail_depth_only', 'objective_convention', 'memory_rule',
                       'measured_vs_projected', 'kind', 'coupling_gates_literals_source',
                       'i_arms_instances_cap.alpha', 'i_arms_instances_cap.configuration',
                       'i_arms_instances_cap.instances', 'i_arms_instances_cap.required_consecutive_cycles',
                       'i_arms_instances_cap.sigma_fixed_expected', 'i_arms_instances_cap.x0')
CHANGED_TOP = ('version', 'predecessor', 'why_v3', 'stage', 'authority', 'cites', 'roots', 'cap', 'labels', 'launcher',
               'harness', 'solves', 'concurrency_per_campaign', 'failure_policy', 'not_permitted', 'iv_cycle40_look',
               'v_continuation', 'vi_predictions', 'm3', 'prefix_gate', 'stops', 'captures', 'decision_rule',
               'nl_identity_check', 'stop_key', 'v_multiplier_capture', 'evaluation')
CHANGED_I = ('i_arms_instances_cap.arms', 'i_arms_instances_cap.cap', 'i_arms_instances_cap.runs',
             'i_arms_instances_cap.execution_order', 'i_arms_instances_cap.stopping',
             'i_arms_instances_cap.v2_declaration')

DECISION_RULE_VERBATIM = (
    'Calibration blocks: 2025|Winter (A) and 2035|Summer (B). 2025|Summer (B) is reported, merged-only, not deciding.\n'
    'For each calibration block:\n'
    '- Entry cycle: k_in is the smallest tail-depth cycle such that |e_b(k)| <= allowance (228.64 EUR or 187.56 EUR) for '
    'every cycle k from k_in to the cap.\n'
    '- "Enters and stays" requires that k_in exists and that cap - k_in >= 20.\n'
    '- Measured level: max over k >= k_in of the interface Delta P omega-RMS. Report Delta Q and Delta V beside it.\n'
    'Outcomes:\n'
    '- If both calibration blocks enter and stay: the report proposes an added stopping criterion, interface Delta P '
    'omega-RMS <= the larger of the two measured levels. It is a proposal only; no production change.\n'
    '- Otherwise: the report states "R not demonstrated at this stopping depth", with R_eff = 4 x min_{k >= K_old} '
    '|e_b(k)| x Sigma_12 w / w_b per block, and the terminal e_b with its settling bar. Nothing touches R.\n'
    'o2 is reported beside, as a diagnostic only (Expert ruling 3: P7 is re-defined as Q - price + dual, diagnostic '
    'only). Report the terminal-step-to-threshold ratio as usual.')

PREDICTIONS_VERBATIM = {
    'C1': 'the prefix is bitwise on both instances.',
    'C2': ('A enters and stays, with k_in <= 100 and terminal |e_b(140)| <= 150 EUR. Basis: o1 decays ~0.93 per cycle; '
           'price -> 0; the o2 limit was fitted at 95-109 EUR.'),
    'C3': ('on B 2035|Summer, the outcome is recorded as competing, with no directional prediction: enters and stays, '
           'or not.'),
    'C4': 'price_b -> |price_b| <= 50 EUR by the cap on both calibration blocks.',
    'C5': 'peak child RSS <= 4 GiB; mean s/cycle A 18-26, B 38-52.',
    'C6': ('the V upper-bound multipliers are nonzero (the bound is active) on at least 90 % of interface entries at '
           'the terminal cycle.'),
    # the Worker's own numeric predictions, from the post-diagnostic fits (P5_15_S56_STEP4_POSTDIAG_REPORT.md D1, D3)
    'C7': ('A 2025|Winter: o1(140) in [40, 110] EUR. Basis: D1 o1 limit estimates 62.5 (held-out fit), 66.4 '
           '(settling_bar), 84.1 (LSQ3) EUR, r ~0.91.'),
    'C8': ('A 2025|Winter: o2(140) in [80, 130] EUR. Basis: D1 o2 limit estimates 95.1, 97.2, 108.6 EUR, r ~0.85-0.86.'),
    'C9': ('at the terminal cycle, the fraction of interface entries whose voltage sits at its 1.1 pu limit (margin '
           '<= 1.1e-6 pu) lies in [0.10, 0.40] on each side of each instance. Basis: D3 at the old exit: A TSO 186/648 '
           '= 0.287, DSO 116/648 = 0.179; B TSO 253/1296 = 0.195, DSO 179/1296 = 0.138. (This contradicts C6 unless the '
           'multipliers of entries below the limit are nonzero.)'),
    'C10': ('if A enters and stays: its measured level (max over k >= k_in of the 2025|Winter Delta P omega-RMS) lies in '
            '[3e-5, 2.5e-4] MW. Basis: handback 3, ~1.2-1.3e-4 MW at the allowance\'s resolution; 9.4e-4 MW at the old '
            'exit.'),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def paths(obj, prefix=()):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(paths(v, prefix + (str(k),)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(paths(v, prefix + (str(i),)))
        return out
    return {'.'.join(prefix): obj}


def root(inst):
    return os.path.join(REPO, V3_DIR, f'campaign_s56s4calib_{TAG}_{PHASE}_{inst}')


def get(obj, dotted):
    for part in dotted.split('.'):
        obj = obj[part]
    return obj


def operational():
    return {
        'Q_b_and_e_b': ('Q_b(k) = the run\'s s56_cycle_capture.jsonl q_blocks.per_year_day[b].settlement_excluded_in_full '
                        '(EUR of total discounted cost); e_b(k) = Q_b(k) - w_b x C_central (ii_central_and_allowance)'),
        'tail_depth_cycle': 'iii_tail_depth_only.tail_depth_cycle, on the run\'s network_ipopt_solve_records.jsonl',
        'k_in': ('the smallest tail-depth cycle k0 such that |e_b(k)| <= allowance_b for EVERY cycle k in [k0, cap] '
                 '(every cycle, tail-depth or not); evaluable only when the run reached the cap (cycles_run == cap); '
                 'otherwise "not evaluable" with the cycle the run stopped at'),
        'enters_and_stays': 'k_in exists and cap - k_in >= 20',
        'interface_mismatch': ('per block b and cycle k, p515_s56_step4_main_evaluate.capture_scenario_mismatch '
                               '(imported, unchanged): dX_s = (xbar_T + d_T,s) - (xbar_D + d_D,s) from the capture\'s '
                               'expectation + deviation copies (P MW, Q MVAr, V kV); omega-RMS = sqrt(sum_s omega_s x mean '
                               'over the block\'s nodes and periods of dX_s^2); max = max |dX_s|'),
        'measured_level': ('L_b = max over k in [k_in, cap] of the block\'s Delta P omega-RMS; beside: the same max of '
                           'Delta Q and Delta V omega-RMS, and the Delta P / Q / V max-abs over those cycles'),
        'proposal': ('both calibration blocks enter and stay -> proposed criterion: interface Delta P omega-RMS <= '
                     'max(L_2025|Winter, L_2035|Summer) MW (a proposal; no production change)'),
        'R_eff': ('R_eff_b = 4 x min over k in [K_old_b, cap] of |e_b(k)| x Sigma_12 w / w_b, K_old = 46 (A), 107 (B) '
                  '(step 4 main V-1 certification cycles); Sigma_12 w = ii_central_and_allowance.sum_12_w'),
        'terminal_settling_bar': ('p515_s56_step4_main_evaluate.settling_bar (imported, unchanged; spec v2 operational '
                                  'S_b) on the series e_b over the run\'s tail-depth cycles (last W = 20, h = 3)'),
        'price_o1_o2': ('price_b(k) = -sum over the block\'s agents of weight x settlement_total_unweighted (capture); '
                        'dual_b(k) = the TSO-side dual terms (expectation P / Q / V + omega-weighted deviation P / Q / V) '
                        'of p515_s56_step4_postdiag.channel_parts (imported, unchanged), recorded split; o1 = e_b - '
                        'price_b; o2 = e_b - price_b + dual_b (diagnostic only, Expert ruling 3)'),
        'tables': 'e_b, o1, o2 (with its dual split), price_b, Delta P / Q / V omega-RMS at k_in, at k = 20, 40, ... and '
                  'at the cap, per block (2025|Summer beside, merged-only)',
        'terminal_step_to_threshold': ('the evaluation record rule_ten.terminal_step_over_threshold and the terminal Boyd '
                                       'ratio max per channel'),
        'M1': ('F1 identities every cycle, adopted tolerance (Expert ruling 2): (i) |priced_residual_price_part + T| <= '
               '64 u (sum_agents |w s_total| + |priced_residual_price_part|), (ii) |(gross_contracted_excluded - gross) - '
               'D| <= 64 u (sum_agents (|Q_full| + |Q_contracted| + |w s_total| + |w s_contracted|) + |gross_contracted_'
               'excluded| + |gross|), u = 2^-53, floor 1e-6 EUR on each bound; (iii) gauge |sum_s omega_s mu_s| <= 1e-9 '
               'x max|mu| per cycle / agent / kind; (iv) V-1 pin total == 0.0 every cycle. Recorded, not a stop'),
        'C1': 'held iff the prefix gate (prefix_gate) held on A and on B (live and post-run)',
        'C2': ('held iff 2025|Winter enters and stays with k_in <= 100 and |e_b(140)| <= 150 EUR'),
        'C3': 'recorded: the 2035|Summer decision-rule verdict (enters and stays, or not), no direction predicted',
        'C4': 'held iff |price_b(cap)| <= 50 EUR on 2025|Winter (k = 140) and on 2035|Summer (k = 320)',
        'C5': ('held iff, per run, evaluation record peak_rss.child_python_process_ru_maxrss <= 4 GiB and the mean of '
               'per_cycle_record cycle_wall_s over every cycle lies in [18, 26] s (A) / [38, 52] s (B)'),
        'C6': ('per instance and side (tso, dso), at the terminal cycle (the cap) of the multiplier sidecar, entries = '
               'every (block, scenario, period): m = |row_dual| + |zU_vmag_sqr| + |zU_vmag| / (2 sqrt(vmag_sqr)) + '
               '|zU_e| / (2 |e|) (the upper-limit multipliers expressed on vmag_sqr; None -> 0); "nonzero" iff m >= '
               '1e-3 x max m over that side\'s entries at that cycle. Held iff the nonzero fraction >= 0.90 on every '
               '(instance, side). Recorded beside (not deciding): the fractions at absolute thresholds 1e-8, 1e-6, 1e-4, '
               '1e-2; the fraction at the limit by primal margin (v_max - sqrt(vmag_sqr) <= 1e-6 x max(1, v_max)); the '
               'joint table; the component (row / vmag_sqr / vmag / e) carrying the multiplier'),
        'C7': 'held iff o1_2025|Winter(140) in [40, 110] EUR',
        'C8': 'held iff o2_2025|Winter(140) in [80, 130] EUR',
        'C9': 'held iff the primal-margin at-limit fraction (C6 beside) lies in [0.10, 0.40] on every (instance, side)',
        'C10': 'evaluated only if A enters and stays: held iff L_2025|Winter in [3e-5, 2.5e-4] MW',
        'solve_identity': ('per run: solve_profile.identity_holds True, base_solves == per_round x (cycles_run + 1), '
                           'blocked 0, exit 0'),
        'record_certification_fields': ('the evaluation record\'s certification / status fields are computed by the '
                                        'harness from the trajectory under required_consecutive_cycles 10 '
                                        '(structural); with the stop disabled they are recorded, never read as a stop '
                                        'or a certification of this run'),
    }


def main():
    v2_sha = sha(V2)
    if not v2_sha.startswith('7ca210fb'):
        raise SystemExit(f'v2 sha256 {v2_sha} is not 7ca210fb')
    with open(V2, encoding='utf-8') as handle:
        v2 = json.load(handle)
    with open(V2_MANIFEST, encoding='utf-8') as handle:
        man = json.load(handle)['files']
    v3 = deepcopy(v2)
    v3['version'] = 3
    v3['predecessor'] = {'path': V2, 'sha256': v2_sha,
                         'relation': ('spec v2 7ca210fb (the step 4 main run, committed 5f1db156). v3 is v2 with the '
                                      'calibration re-run of Expert decision 1 (a): arm V-1 only, certification stop '
                                      'disabled, caps A 140 / B 320, prefix 46 / 107 bitwise against v2\'s V-1 runs, '
                                      'frozen decision rule, interface V upper-bound multipliers captured. Every v2 key '
                                      'not listed in why_v3.changed is verbatim')}
    v3['why_v3'] = {
        'reason': ('Expert handback 3 decision 1 (a) APPROVED (TASKS.md "Expert rulings on handback 3 (2026-11-23)" item '
                   '1): the per-block allowance cannot be reached at production certification; re-run V-1 with the '
                   'certification stop disabled to a declared cap'),
        'changed_top_level': list(CHANGED_TOP),
        'changed_in_i_arms_instances_cap': list(CHANGED_I),
        'unchanged_checked_equal': list(UNCHANGED_MUST_HOLD),
        'derivation_script': 'p515_s56_step4_calib_spec_v3_derive.py',
    }
    v3['stage'] = ('P5.15 S56 validation step 4 CALIBRATION RE-RUN (Expert decision 1 (a)): arm V-1 only on instance A '
                   '(2025|Winter) and instance B ({2025, 2035} x Summer); x0, alpha 0; certification stop disabled; '
                   'caps A 140, B 320; prefix 1..46 / 1..107 bitwise against step 4 main V-1; frozen decision rule; '
                   'interface V upper-bound multipliers captured')
    v3['authority'] = ['TASKS.md "Expert rulings on handback 3 (2026-11-23)" item 1 (decision 1 (a) approved, (b) '
                       'beside as diagnostic only; caps 140 / 320; prefix 46 / 107 bitwise; decision rule; capture '
                       'interface V upper-bound multipliers); item 2 (M1 tolerance adopted); item 3 (P7 re-defined, '
                       'diagnostic only)',
                       'P5_15_S56_STEP4_EXPERT_HANDBACK_3.md decision 1', 'Planner task 2026-11-23 (calibration re-run)']
    cites = {k: {'path': p, 'sha256': sha(p)} for k, p in CITED.items()}
    cites['tasks'] = {'path': 'TASKS.md', 'sha256_at_freeze': sha('TASKS.md'),
                      'items': '"Expert rulings on handback 3 (2026-11-23)" items 1-3; Order (2026-11-23, after handback '
                               '3 rulings) step (a)'}
    for k in ('design', 'plan_v4', 'smoke_spec_v2', 'smoke_launcher', 'smoke_evaluation', 'diag_results',
              'diag_spec_ff466289'):
        cites[k] = v2['cites'][k]
    v3['cites'] = cites
    v3['roots'] = {'main_dir': V3_DIR, 'root_tag': TAG, 'campaign_roots': {PHASE: {i: root(i) for i in ('A', 'B')}}}
    v3['cap'] = {PHASE: dict(CAP)}
    v3['labels'] = {PHASE: {'V1': f'x0_V1_{PHASE}'}}
    v3['launcher'] = 'p515_s56_step4_calib.py'
    v3['harness'] = {'path': 'p515_s44_campaign_harness.py', 'sha256': sha('p515_s44_campaign_harness.py'),
                     'commit_of_calibration_keys': HARNESS_COMMIT,
                     'commit_of_pinned_capture_key': v2['harness']['commit_of_pinned_capture_key'],
                     'predecessor_sha256_in_v2': v2['harness']['sha256']}
    i_block = v3['i_arms_instances_cap']
    v1_arm = deepcopy(v2['i_arms_instances_cap']['arms']['V1'])
    v1_arm['options']['s56_certification_stop_disabled'] = True
    v1_arm['options']['s56_interface_v_multiplier_capture'] = True
    v1_arm['role'] = ('calibration arm; formulation of record (Expert ruling 2); certification stop disabled to the '
                      'declared cap (Expert decision 1 (a))')
    i_block['arms'] = {'V1': v1_arm}
    i_block['cap'] = dict(CAP)
    i_block['runs'] = [{'phase': PHASE, 'instance': i, 'arm': 'V1', 'label': f'x0_V1_{PHASE}',
                        'campaign_root': root(i), 'cap': CAP[i]} for i in ('A', 'B')]
    i_block['execution_order'] = ('instance A (one campaign, one entry, concurrency 1), then instance B (likewise), '
                                  'serially: one campaign lock per checkout. B is refused unless A exited 0 at its cap, '
                                  'its prefix gate held and no stop fired')
    i_block['stopping'] = ('production\'s per-cycle convergence (every Boyd channel passes and every local solve '
                           'succeeded), its consecutive count, the convergence-depth tail (active in cycle k+1 iff cycle '
                           'k converged; not sticky) and every other production rule are UNCHANGED; only the '
                           'certificate length is raised to 10**9 by the harness key s56_certification_stop_disabled '
                           '(stop_key), so each run ends at its cap (A 140, B 320) or at a declared stop (stops). '
                           'Production\'s rho freeze backstop (freeze_backstop_cycle 200) applies unchanged and falls '
                           'inside B\'s cap')
    i_block['v2_declaration'] = 'V-2 is not run in v3 (V-1 only).'
    v3['concurrency_per_campaign'] = 1
    v3['solves'] = {'launcher_parent': 0, 'evaluate': 0, 'per_round': dict(v2['solves']['per_round']),
                    'per_run': v2['solves']['per_run'],
                    'expected_base_at_cap': {i: v2['solves']['per_round'][i] * (CAP[i] + 1) for i in ('A', 'B')}}
    v3['failure_policy'] = ('stops (stops): a non-bitwise prefix, a harness fault, the B 4.5 h projection. Everything else '
                            '(C1-C10, M1, the decision-rule verdict) is recorded, never re-run or tuned')
    v3['not_permitted'] = ['production edits', 'other arms', 'any continuation beyond the caps', 'any run not in this spec',
                           'cap changes', 'overwriting committed artifacts (the step 4 evaluator / post-diagnostics are '
                           'not re-run onto their outputs)', 'detached runs (screen, nohup, shell backgrounding)',
                           'pgrep -f wait loops', 'writes into .claude/worktrees/', 'touching TASKS.md',
                           'touching step4_db_resolve_v1/ or cohort_t8_binding_v1/']
    del v3['iv_cycle40_look']
    del v3['m3']
    v3['v_continuation'] = {'status': 'none: no continuation beyond the caps (not permitted in v3)'}
    v3['prefix_gate'] = {
        'prefix_cycles': dict(K_OLD),
        'what': ('cycles 1..46 (A) / 1..107 (B) of the v3 run must reproduce step 4 main\'s V-1 per-cycle records '
                 'bitwise, with the declared keys excluded; this also checks that the merged cohort code (7636fa7c) '
                 'leaves the x0 coupled path unchanged'),
        'files': {'per_cycle_record.jsonl': 'every line with cycle <= P (after the run)',
                  's56_cycle_capture.jsonl': 'every cycle line with cycle <= P, header excluded (live and after)',
                  'interface_duals_per_cycle.jsonl': 'every cycle line with cycle <= P, header excluded (live and after)',
                  'network_ipopt_solve_records_append.jsonl': 'every record of rounds 0..P in file order (live and '
                                                              'after)'},
        'excluded_keys_declared_in_advance': list(v2['m3']['excluded_keys_declared_in_advance']),
        'excluded_keys_source': 'spec v2 m3 (unchanged): timing / process / path / byte-count fields only',
        'equality': v2['m3']['equality'],
        'note_required_consecutive_cycles': ('production\'s in-memory diagnostics row field required_consecutive_cycles '
                                             'records the certificate length in force (10**9 in v3); it is not a '
                                             'per_cycle_record field and is not compared'),
        'reference': {i: {'eval_dir': REF_EVAL[i], 'eval_key': None, 'files_sha256': {}} for i in ('A', 'B')},
        'on_difference': 'the run is stopped (stops) and reported',
    }
    for i in ('A', 'B'):
        with open(os.path.join(REF_EVAL[i], 'launch.json'), encoding='utf-8') as handle:
            v3['prefix_gate']['reference'][i]['eval_key'] = json.load(handle)['eval_key']
        for name in PREFIX_FILES:
            p = os.path.join(REF_EVAL[i], name)
            got = sha(p)
            if got != man[p]['sha256']:
                raise SystemExit(f'reference {p} sha256 {got} != the v2 manifest {man[p]["sha256"]}')
            v3['prefix_gate']['reference'][i]['files_sha256'][name] = got
        v3['prefix_gate']['reference'][i]['manifest_class'] = {n: man[os.path.join(REF_EVAL[i], n)]['class']
                                                                for n in PREFIX_FILES}
    v3['stops'] = {
        'prefix': 'a prefix item differs (live monitor, SIGINT to the launcher: the harness interrupt path)',
        'harness_fault': 'an exception, a guard mismatch, a child exit != 0',
        'projection': {'instances': ['B'], 'at_cycle': 20, 'limit_h': 4.5,
                       'definition': ('t(k) = the time the launcher monitor observed cycle k\'s capture line; mean '
                                      's/cycle = (t(20) - t(1)) / 19; projection = (t(1) - t_launch) + mean x (cap - 1), '
                                      't_launch = the mtime of the child\'s launch.json; projection > 4.5 h stops B')},
    }
    v3['stop_key'] = {
        'key': 's56_certification_stop_disabled', 'harness_commit': HARNESS_COMMIT, 'threshold': 10 ** 9,
        'how': ('harness wrapper over production _capture_convergence_depth_tail_baseline (called once before cycle 1 '
                'when the tail is enabled): minimum_consecutive_converged_cycles 10 -> 10**9; restored after the run; no '
                'production edit'),
        'reads_of_the_parameter': {
            'shared_resources_planning.py': {'3474': 'convergence = (consecutive_converged_cycles >= '
                                                     'admm_parameters.minimum_consecutive_converged_cycles)  [the stop '
                                                     'test]',
                                             '3600': "'required_consecutive_cycles': admm_parameters."
                                                     "minimum_consecutive_converged_cycles,  [diagnostics row, "
                                                     "reporting]",
                                             '3774': "f'{admm_parameters.minimum_consecutive_converged_cycles} | '  "
                                                     "[INFO print, reporting]",
                                             '8774': 'docstring text only (REMOVED: ...)'},
            'admm_parameters.py': {'23': 'default 2', '396': 'loaded from the case file', '515-516': 'validated >= 1'},
            'other_production_modules': 'none (asserted before any solve by the harness, module-wide scan)'},
        'eval_key': 'not keyed (W98 rule: the cap and the certificate length are spec-level); V-1 eval key = step 4 '
                    'main\'s',
    }
    v3['v_multiplier_capture'] = {
        'key': 's56_interface_v_multiplier_capture', 'harness_commit': HARNESS_COMMIT,
        'file': 's56_interface_v_multipliers.jsonl',
        'finding_row_or_bound': ('the interface V upper limit is a constraint ROW voltage_magnitude_upper_cons (vmag_sqr - '
                                 'slack_up <= v_max^2) on both sides; on the TSO interface bus (PQ) the slack is fixed '
                                 'at 0 in the ADMM model and the vmag / vmag_sqr bounds are relaxed (1.15 / 1.3225), so '
                                 'the row carries the limit alone; on the DSO reference bus the slack is bounded [0, 0] '
                                 'AND vmag <= 1.1, vmag_sqr <= 1.21 are hard variable bounds (and e <= 1.1501), so the '
                                 'limit is held by the row and two coincident bounds (degenerate). All are captured: '
                                 'row dual (model.dual) and ipopt_zU_out of vmag_sqr, vmag, e, with vmag_sqr and e'),
        'how': ('read from the import suffixes production already declares and loads after every solve; pure reads; '
                'no NL / solve change (nl_identity_check)'),
    }
    v3['nl_identity_check'] = {
        'compare': {'path': CHECKS_COMPARE, 'sha256': sha(CHECKS_COMPARE)},
        'nokey': {'path': CHECKS_NOKEY, 'sha256': sha(CHECKS_NOKEY)},
        'manifest': {'path': CHECKS_MANIFEST, 'sha256': sha(CHECKS_MANIFEST)},
        'what': ('zero-solve, instance A, stubbed solves with synthetic suffix values, keys off vs on: NL of every block '
                 'at ADMM cycles 1-3 identical (12/12); diagnostics cycles 1-10 identical except '
                 'required_consecutive_cycles; off certifies at 10, on runs to the cap; multiplier capture = stub values '
                 'bitwise; option-off identity (nokey) 21/21'),
    }
    v3['captures'] = {
        'every_cycle_post_hoc': ('e_b, price_b, o1, o2 with the dual split (P / Q / V, expectation and deviation), and '
                                 'Delta P / Q / V omega-RMS and max per block: computed POST HOC from the per-cycle '
                                 'capture (s56_cycle_capture.jsonl, interface_duals_per_cycle.jsonl) with the imported '
                                 'post-diagnostic / evaluator functions (decision_rule.operational)'),
        'every_cycle_live': 'the interface V upper-limit multipliers, both sides (v_multiplier_capture)',
        'capture_path_assertion': ('before launch the launcher asserts a capture path for every quantity above (the '
                                   'main launcher checklist, the two harness key paths, the post hoc sources and '
                                   'functions, the prefix references and their hashes, the NL identity check)'),
    }
    v3['decision_rule'] = {'verbatim': DECISION_RULE_VERBATIM, 'K_old': dict(K_OLD), 'operational': operational()}
    v3['vi_predictions'] = {
        'recorded': PREDICTIONS_VERBATIM,
        'recorded_failure': 'failed predictions are recorded; the stops are only those of stops',
        'M1_adopted_tolerance': 'decision_rule.operational.M1 (recorded, not a stop)',
        'also_record': ['terminal-step-to-threshold ratio and terminal Boyd ratios', 'local-solve failures',
                        'solve-count identity', 'peak RSS (child, IPOPT)', 'mean and per-cycle wall', 'vm_stat before, '
                        'hourly, after', 'the evaluation record certification fields (structural; not a stop)'],
    }
    v3['evaluation'] = {'script': 'p515_s56_step4_calib_evaluate.py (zero-solve; written per decision_rule.operational, '
                                  'committed with the evidence)', 'output': V3_DIR}
    # ---- the key-path check
    for k in UNCHANGED_MUST_HOLD:
        if get(v2, k) != get(v3, k):
            raise SystemExit(f'{k} changed')
    p2, p3 = paths(v2), paths(v3)
    changed = sorted(k for k in set(p2) | set(p3) if p2.get(k, '<absent>') != p3.get(k, '<absent>'))
    bad = [k for k in changed if not (k.split('.')[0] in CHANGED_TOP
                                      or any(k == c or k.startswith(c + '.') for c in CHANGED_I))]
    if bad:
        raise SystemExit(f'unexpected changes from v2: {bad}')
    os.makedirs(V3_DIR, exist_ok=True)
    if any(f.startswith(V3_PREFIX) for f in os.listdir(V3_DIR)):
        raise SystemExit('a v3 spec already exists')
    text = json.dumps(v3, indent=1, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(V3_DIR, f'{V3_PREFIX}{digest[:8]}.json')
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)
    print(json.dumps({'path': path, 'sha256': digest, 'n_changed_paths_vs_v2': len(changed),
                      'changed_top_level': sorted({k.split('.')[0] for k in changed})}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
