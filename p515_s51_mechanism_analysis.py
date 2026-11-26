"""
P5.15 Addendum 39 follow-up (task W43) -- the W42 "Task B" mechanism analysis as a COMMITTED
script (CLAUDE.md: preserve the formula, not only the inputs). ZERO SOLVES: it reads only the
committed artifacts of `p515_s51_single_block_ab.py` and never builds or solves a model; a
SolveProfileGuard with NO permitted call site is installed for the whole run and verified
at exactly 0 solves.

WHAT IT RE-DERIVES, per run, per alpha arm, per (year, day) block of the selected DSO, from
the `mechanism` record (arms run at or after 2b7d8647) -- block-local, probability-weighted,
MWh/day (one period = one hour, so a sum over periods of MW is MWh):

    E|d|        = sum_s omega_s sum_t |p_int[s,t] - pbar_t|              (d_p_mw)
    E curt      = sum_s omega_s sum_t (pg_avail - pg) over curtaillable gens
                  = gen_curtaillable_avail_mw - gen_nonref_p_mw, CHECKED against
                  total_gen_curt_penalty / kappa (the non-reference DSO generators must all be
                  curtaillable for this to hold; the residual is recorded)
    E up/down   = the same over the fl_reg loads' flex_p_up / flex_p_down
    E residual  = the same over residual_losses_slacks_mw (signed)
    kappa_eff   = total_gen_curt_penalty / E curt (where E curt > KAPPA_MIN_MWH) -- the RES
                  curtailment price IN FORCE on the solved model, recovered from the artifact
                  and compared with definitions.PENALTY_GENERATION_CURTAILMENT
    objective identity: objective_function_rule_value against the sum of every recorded
                  component (total_gen_cost + total_flex_cost + total_load_curt_cost +
                  total_gen_curt_penalty + total_ess_utilization_cost_penalty +
                  total_slack_penalties + total_ess_complementarity_penalties +
                  interface_settlement_weight * interface_settlement + row18_deviation_charge +
                  voltage_pin_weighted), and the W42 four-term form
                  gen_curt + flex + slack + row18.

THE PREDICTED THRESHOLD (per hour, per block). With two operation scenarios whose curtaillable
availability differs at hour t, holding the schedule means curtailing the higher-availability
operation scenario down to the lower one. Curtailing c MWh there (operation-scenario
probability omega_hi) costs omega_hi * kappa * c, raises pbar_t by omega_hi * c, and reduces
E|d_t| = 2 omega_hi omega_lo (Delta_t - c) by 2 omega_hi omega_lo * c, i.e. saves
alpha * pibar_t * 2 omega_hi omega_lo * c of row 18 charge. Curtail iff
alpha * pibar_t * 2 omega_lo > kappa, so

    alpha_t = kappa / (2 * omega_lo,t * pibar_t)          (omega_lo = 1/2  ->  kappa / pibar_t)

and the block threshold is alpha*_block = max_t alpha_t over the hours that deviate at
alpha = 0 (E|d_t| > HOUR_DEV_TOL_MWH in the alpha = 0 arm). Losses, reactive deviation (row 18
also prices Q) and flexibility are ignored by this prediction; the per-hour test below is what
says whether ignoring them matters.

PER-HOUR TEST, for every positive-alpha arm with a mechanism record: predicted "deviates" iff
alpha < alpha_t; observed "deviates" iff E|d_t| > HOUR_DEV_TOL_MWH. Every disagreement is
listed with its margin (alpha - alpha_t) / alpha_t, never suppressed.

MEASURED PER-BLOCK BRACKET, from EVERY input run (mechanism or not): the per-block RMS
interface-P dispersion production reports (`dispersion.p.rms_mw`) against the gate's declared
DISPERSION_ZERO_TOL_MW -- (largest alpha above the tolerance, smallest alpha at or below it).
Duplicate alphas across runs are checked for bitwise agreement and the result recorded.

Every input gate.json is verified against its run's committed manifest_sha256.json before use,
and every input run must be the same derived instance (derived-case sha256) and selected node.

EXACT COMMAND (repo root; attached; both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_mechanism_analysis.py --label <write-once-label> \\
        > data/SRP1/Results/P515S51/mechanism_analysis_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/mechanism_analysis/<label>/
Exit 0 on success, 1 on error, 2 on a precondition refusal.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard((), label='P5.15 W43 mechanism analysis -- zero solves').install()

from definitions import PENALTY_GENERATION_CURTAILMENT  # noqa: E402

STAGE = ('P5.15 Addendum 39 follow-up W43 -- committed re-derivation of the W42 mechanism '
         'analysis (E|d|, E curt, kappa, predicted per-block threshold, objective identity)')
SCHEMA = 'p515_s51_mechanism_analysis_v1'
AB_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'single_block_ab')
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'mechanism_analysis')
DEFAULT_RUNS = ('ab_r2', 'alpha_star_r1', 'alpha_star_r2_fine', 'alpha_legs_r1')

# Declared constants (operational definitions, not fixture literals):
DISPERSION_ZERO_TOL_MW = 1.0e-2   # = p515_s51_single_block_ab.DISPERSION_ZERO_TOL_MW (reused, not tuned)
HOUR_DEV_TOL_MWH = 1.0e-2         # an hour "deviates" iff E|d_t| exceeds this (same number, per hour)
KAPPA_MIN_MWH = 1.0               # kappa_eff is reported only where E curt exceeds this
AVAIL_EQUAL_TOL_MW = 1.0e-9       # operation scenarios with equal availability define no omega_lo


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'<git failed: {error}>'


def load_run(label):
    """Load a committed A/B run's gate.json after verifying it against the run's manifest."""
    run_dir = os.path.join(AB_ROOT, label)
    gate_path = os.path.join(run_dir, 'gate.json')
    manifest_path = os.path.join(run_dir, 'manifest_sha256.json')
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    rel = os.path.relpath(gate_path, REPO)
    actual = _sha256_file(gate_path)
    if manifest.get(rel) != actual:
        raise RuntimeError(f'{rel}: sha256 {actual} does not match the manifest ({manifest.get(rel)})')
    with open(gate_path) as handle:
        gate = json.load(handle)
    return gate, {'gate_json': rel, 'sha256': actual,
                  'manifest': os.path.relpath(manifest_path, REPO),
                  'manifest_sha256': _sha256_file(manifest_path),
                  'git_head_of_run': gate.get('git_head'),
                  'derived_case_sha256': gate['derived_case']['sha256'],
                  'selected_node': gate.get('selected_node'),
                  'alphas': gate.get('alphas')}


def _operation_scenario_probabilities(per_scenario):
    """Marginal operation-scenario probabilities from the recorded (s_m, s_o) probabilities."""
    omega = {}
    for key, entry in per_scenario.items():
        s_o = key.split('_')[1]
        omega[s_o] = omega.get(s_o, 0.0) + entry['probability']
    return omega


def block_mechanism(mechanism):
    """Block-level and per-hour probability-weighted quantities from one mechanism record."""
    per_scenario = mechanism['per_scenario']
    n_periods = len(next(iter(per_scenario.values()))['series']['d_p_mw'])

    def weighted(fn):
        per_hour = [0.0] * n_periods
        for entry in per_scenario.values():
            series = entry['series']
            for t in range(n_periods):
                per_hour[t] += entry['probability'] * fn(series, t)
        return per_hour

    hours = {
        'E_abs_d_mwh': weighted(lambda s, t: abs(s['d_p_mw'][t])),
        'E_curt_mwh': weighted(lambda s, t: s['gen_curtaillable_avail_mw'][t] - s['gen_nonref_p_mw'][t]),
        'E_flex_up_mwh': weighted(lambda s, t: s['flex_p_up_mw'][t]),
        'E_flex_down_mwh': weighted(lambda s, t: s['flex_p_down_mw'][t]),
        'E_flex_down_cost': weighted(lambda s, t: s['flex_p_down_cost'][t]),
        'E_residual_losses_slacks_mwh': weighted(lambda s, t: s['residual_losses_slacks_mw'][t]),
        'E_interface_p_mwh': weighted(lambda s, t: s['interface_p_mw'][t]),
    }
    totals = {k: sum(v) for k, v in hours.items()}
    obj = mechanism['objective_totals']
    kappa_eff = (obj['total_gen_curt_penalty'] / totals['E_curt_mwh']
                 if totals['E_curt_mwh'] > KAPPA_MIN_MWH else None)
    curt_check_residual = totals['E_curt_mwh'] * PENALTY_GENERATION_CURTAILMENT - obj['total_gen_curt_penalty']

    # per-hour omega_lo: the operation scenario with the LOWER curtaillable availability
    omega = _operation_scenario_probabilities(per_scenario)
    omega_lo = [None] * n_periods
    if len(omega) == 2:
        avail_by_so = {}
        for key, entry in per_scenario.items():
            s_o = key.split('_')[1]
            avail_by_so.setdefault(s_o, entry['series']['gen_curtaillable_avail_mw'])
        (so_a, av_a), (so_b, av_b) = sorted(avail_by_so.items())
        for t in range(n_periods):
            if abs(av_a[t] - av_b[t]) > AVAIL_EQUAL_TOL_MW:
                omega_lo[t] = omega[so_a] if av_a[t] < av_b[t] else omega[so_b]
    return {'n_periods': n_periods, 'hours': hours, 'totals': totals,
            'omega_operation': omega, 'omega_lo_by_hour': omega_lo,
            'kappa_eff': kappa_eff, 'curt_vs_penalty_residual_at_kappa': curt_check_residual,
            'objective_totals': obj}


def objective_identity(block):
    obj = block['mechanism']['objective_totals']
    full = (obj.get('total_gen_cost', 0.0) + obj.get('total_flex_cost', 0.0)
            + obj.get('total_load_curt_cost', 0.0) + obj.get('total_gen_curt_penalty', 0.0)
            + obj.get('total_ess_utilization_cost_penalty', 0.0) + obj.get('total_slack_penalties', 0.0)
            + obj.get('total_ess_complementarity_penalties', 0.0)
            + obj['interface_settlement_weight'] * obj.get('interface_settlement', 0.0)
            + obj.get('row18_deviation_charge', 0.0) + obj.get('voltage_pin_weighted', 0.0))
    four = (obj.get('total_gen_curt_penalty', 0.0) + obj.get('total_flex_cost', 0.0)
            + obj.get('total_slack_penalties', 0.0) + obj.get('row18_deviation_charge', 0.0))
    value = block['objective_function_rule_value']
    return {'objective_function_rule_value': value,
            'sum_of_all_recorded_components': full, 'residual_all_components': value - full,
            'w42_four_term_sum': four, 'residual_four_term': value - four}


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--runs', nargs='+', default=list(DEFAULT_RUNS),
                        help='committed single_block_ab run labels to read')
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.label)
    if os.path.exists(out_dir):
        print(f'REFUSED: output directory already exists (write-once): {out_dir}', file=sys.stderr)
        return 2

    try:
        runs, inputs = {}, {}
        for label in args.runs:
            runs[label], inputs[label] = load_run(label)
        instances = {v['derived_case_sha256'] for v in inputs.values()}
        nodes = {v['selected_node'] for v in inputs.values()}
        if len(instances) != 1 or len(nodes) != 1:
            raise RuntimeError(f'input runs are not one instance/node: {instances} {nodes}')

        # ---- the zero-solve price rows (pibar_t) of the selected DSO, from the launch record
        node = nodes.pop()
        price_rows = {}
        for key, block in next(iter(runs.values()))['price_table'].items():
            parts = key.split(':')   # DSO:<node>:<name>:<year>:<day>
            if parts[1] == str(node):
                price_rows[f'{parts[3]}:{parts[4]}'] = [r['pibar'] for r in block['rows']]
        for label, gate in runs.items():   # the price table must be identical across runs
            for key, block in gate['price_table'].items():
                parts = key.split(':')
                if parts[1] == str(node):
                    if [r['pibar'] for r in block['rows']] != price_rows[f'{parts[3]}:{parts[4]}']:
                        raise RuntimeError(f'{label}: pibar differs for {key}')

        # ---- per run / arm / block mechanism quantities
        mechanism = {}
        for label, gate in runs.items():
            for arm, record in gate['arms'].items():
                for bkey, block in record['blocks'].items():
                    if 'mechanism' not in block or 'capture_error' in block['mechanism']:
                        continue
                    m = block_mechanism(block['mechanism'])
                    mechanism.setdefault(label, {}).setdefault(arm, {})[bkey] = {
                        'alpha': record['alpha'],
                        'rms_mw': block['dispersion']['p']['rms_mw'],
                        'totals': m['totals'],
                        'kappa_eff': m['kappa_eff'],
                        'curt_vs_penalty_residual_at_kappa': m['curt_vs_penalty_residual_at_kappa'],
                        'omega_operation': m['omega_operation'],
                        'objective_identity': objective_identity(block),
                        'per_hour': m['hours'],
                        'omega_lo_by_hour': m['omega_lo_by_hour'],
                    }

        # ---- the alpha = 0 reference with a mechanism record (for the deviating hours)
        reference = None
        for label, arms in mechanism.items():
            if 'alpha_0.0' in arms:
                reference = (label, arms['alpha_0.0'])
        if reference is None:
            raise RuntimeError('no alpha = 0 arm with a mechanism record among the input runs')
        ref_label, ref_blocks = reference

        prediction = {}
        for bkey, ref in ref_blocks.items():
            pibar = price_rows[bkey]
            dev0 = ref['per_hour']['E_abs_d_mwh']
            alpha_t = []
            for t, p in enumerate(pibar):
                w_lo = ref['omega_lo_by_hour'][t]
                alpha_t.append(PENALTY_GENERATION_CURTAILMENT / (2.0 * w_lo * p)
                               if (w_lo is not None and p > 0.0) else None)
            deviating = [t for t in range(len(pibar)) if dev0[t] > HOUR_DEV_TOL_MWH]
            candidates = [alpha_t[t] for t in deviating if alpha_t[t] is not None]
            prediction[bkey] = {
                'kappa': PENALTY_GENERATION_CURTAILMENT,
                'pibar_by_hour': pibar,
                'alpha_t_by_hour': alpha_t,
                'alpha0_E_abs_d_by_hour': dev0,
                'hours_deviating_at_alpha0': deviating,
                'predicted_alpha_star_block': max(candidates) if candidates else None,
                'argmax_hour': (deviating[[alpha_t[t] for t in deviating].index(max(candidates))]
                                if candidates else None),
                'min_pibar_over_deviating_hours': min(pibar[t] for t in deviating) if deviating else None,
            }

        # ---- per-hour prediction test for every positive-alpha arm with a mechanism record
        hour_test = {}
        for label, arms in mechanism.items():
            for arm, blocks in arms.items():
                for bkey, blk in blocks.items():
                    alpha = blk['alpha']
                    if alpha <= 0.0:
                        continue
                    pred = prediction[bkey]
                    agree, disagree = 0, []
                    for t in pred['hours_deviating_at_alpha0']:
                        a_t = pred['alpha_t_by_hour'][t]
                        if a_t is None:
                            continue
                        predicted_dev = alpha < a_t
                        observed = blk['per_hour']['E_abs_d_mwh'][t]
                        observed_dev = observed > HOUR_DEV_TOL_MWH
                        if predicted_dev == observed_dev:
                            agree += 1
                        else:
                            disagree.append({'hour': t, 'pibar': pred['pibar_by_hour'][t], 'alpha_t': a_t,
                                             'margin_alpha_minus_alpha_t_over_alpha_t': (alpha - a_t) / a_t,
                                             'predicted_deviates': predicted_dev,
                                             'observed_E_abs_d_mwh': observed})
                    hour_test.setdefault(label, {}).setdefault(arm, {})[bkey] = {
                        'alpha': alpha, 'n_hours_tested': agree + len(disagree),
                        'n_agree': agree, 'disagreements': disagree}

        # ---- measured per-block bracket from EVERY run (production's per-block RMS)
        rms_by_alpha = {}
        duplicates = []
        for label, gate in runs.items():
            for arm, record in gate['arms'].items():
                alpha = record['alpha']
                for bkey, block in record['blocks'].items():
                    value = block['dispersion']['p']['rms_mw']
                    slot = rms_by_alpha.setdefault(bkey, {})
                    if alpha in slot:
                        duplicates.append({'block': bkey, 'alpha': alpha, 'run': label,
                                           'rms_mw': value, 'first_run': slot[alpha]['run'],
                                           'first_rms_mw': slot[alpha]['rms_mw'],
                                           'bitwise_equal': value == slot[alpha]['rms_mw']})
                    else:
                        slot[alpha] = {'rms_mw': value, 'run': label}
        bracket = {}
        for bkey, slot in rms_by_alpha.items():
            alphas = sorted(slot)
            above = [a for a in alphas if slot[a]['rms_mw'] > DISPERSION_ZERO_TOL_MW]
            at_or_below = [a for a in alphas if slot[a]['rms_mw'] <= DISPERSION_ZERO_TOL_MW]
            star = min((a for a in at_or_below if a > 0.0), default=None)
            below = [a for a in above if star is None or a < star]
            pred = prediction.get(bkey, {}).get('predicted_alpha_star_block')
            bracket[bkey] = {
                'rms_mw_by_alpha': {str(a): slot[a]['rms_mw'] for a in alphas},
                'measured_bracket': [below[-1] if below else None, star],
                'predicted_alpha_star_block': pred,
                'prediction_inside_measured_bracket': (
                    None if (pred is None or star is None) else
                    ((below[-1] if below else 0.0) < pred <= star)),
                'non_monotone_alphas_above_tol_after_star': [a for a in above if star is not None and a > star],
            }

        # ---- holding cost: E curt at the held arm against E|d| at alpha = 0
        holding = {}
        for label, arms in mechanism.items():
            for arm, blocks in arms.items():
                for bkey, blk in blocks.items():
                    ref = ref_blocks[bkey]['totals']['E_abs_d_mwh']
                    holding.setdefault(label, {}).setdefault(arm, {})[bkey] = {
                        'alpha': blk['alpha'],
                        'E_abs_d_mwh': blk['totals']['E_abs_d_mwh'],
                        'E_curt_mwh': blk['totals']['E_curt_mwh'],
                        'E_abs_d_removed_mwh': ref - blk['totals']['E_abs_d_mwh'],
                        'E_curt_over_E_abs_d_removed': (blk['totals']['E_curt_mwh'] / (ref - blk['totals']['E_abs_d_mwh'])
                                                        if (ref - blk['totals']['E_abs_d_mwh']) > 1.0 else None),
                    }

        guard_failures = GUARD.verify(0)
        payload = {
            'schema': SCHEMA, 'stage': STAGE, 'label': args.label, 'created_utc': _utc(),
            'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'instance_derived_case_sha256': next(iter(instances)), 'selected_node': node,
            'inputs': inputs,
            'declared_constants': {'DISPERSION_ZERO_TOL_MW': DISPERSION_ZERO_TOL_MW,
                                   'HOUR_DEV_TOL_MWH': HOUR_DEV_TOL_MWH, 'KAPPA_MIN_MWH': KAPPA_MIN_MWH,
                                   'AVAIL_EQUAL_TOL_MW': AVAIL_EQUAL_TOL_MW,
                                   'PENALTY_GENERATION_CURTAILMENT': PENALTY_GENERATION_CURTAILMENT},
            'formulas': {
                'E_abs_d_mwh': 'sum_s omega_s sum_t |d_p_mw[s][t]|',
                'E_curt_mwh': 'sum_s omega_s sum_t (gen_curtaillable_avail_mw - gen_nonref_p_mw)[s][t]',
                'kappa_eff': 'total_gen_curt_penalty / E_curt_mwh (E_curt_mwh > KAPPA_MIN_MWH)',
                'alpha_t': 'kappa / (2 * omega_lo_t * pibar_t); omega_lo_t = probability of the operation '
                           'scenario with the lower curtaillable availability at hour t',
                'predicted_alpha_star_block': 'max_t alpha_t over hours with alpha=0 E|d_t| > HOUR_DEV_TOL_MWH',
                'hour_test': 'predicted deviates iff alpha < alpha_t; observed deviates iff E|d_t| > HOUR_DEV_TOL_MWH',
                'measured_bracket': '(largest alpha with block rms > DISPERSION_ZERO_TOL_MW, smallest positive '
                                    'alpha with block rms <= it), over every input run',
            },
            'objective_convention': ('block-local, UNWEIGHTED by year/day/discount; the standalone DSO '
                                     'objective (`objective_function_rule`) as solved by the A/B, with '
                                     'interface_settlement_weight as recorded on the model (0 in the '
                                     'standalone build)'),
            'not_a_result': ('uncoordinated standalone DSO solves (the ADMM initialization solve); '
                             'nothing here is a coordinated (ADMM) quantity'),
            'alpha0_reference_run': ref_label,
            'mechanism': mechanism,
            'prediction': prediction,
            'hour_test': hour_test,
            'block_bracket': bracket,
            'duplicate_alpha_checks': duplicates,
            'holding': holding,
            'solve_profile': {'declared': 0, 'counts': dict(GUARD.counts), 'verify_failures': guard_failures},
        }
        if guard_failures:
            raise RuntimeError(f'solve guard: {guard_failures}')

        os.makedirs(out_dir)
        out_path = os.path.join(out_dir, 'analysis.json')
        with open(out_path, 'w') as handle:
            json.dump(payload, handle, indent=1, default=str)
        manifest = {os.path.relpath(out_path, REPO): _sha256_file(out_path)}
        for info in inputs.values():
            manifest[info['gate_json']] = info['sha256']
            manifest[info['manifest']] = info['manifest_sha256']
        with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        # ---- console summary
        print(f'[W43] inputs verified against manifests: {list(inputs)}; instance {next(iter(instances))[:8]}; node {node}')
        print(f'[W43] solves: {GUARD.counts} (declared 0)')
        for label, arms in mechanism.items():
            for arm, blocks in sorted(arms.items(), key=lambda kv: kv[1][next(iter(kv[1]))]['alpha']):
                for bkey, blk in blocks.items():
                    tot, oi = blk['totals'], blk['objective_identity']
                    print(f'[W43] {label:18s} {arm:12s} {bkey:12s} E|d|={tot["E_abs_d_mwh"]:9.4f} '
                          f'Ecurt={tot["E_curt_mwh"]:9.4f} Eup={tot["E_flex_up_mwh"]:.4f} '
                          f'Edown={tot["E_flex_down_mwh"]:.4f} Eres={tot["E_residual_losses_slacks_mwh"]:9.4f} '
                          f'kappa_eff={blk["kappa_eff"]} curt_chk={blk["curt_vs_penalty_residual_at_kappa"]:.2e} '
                          f'obj_res_all={oi["residual_all_components"]:.2e} obj_res_4={oi["residual_four_term"]:.2e}')
        for bkey, pred in prediction.items():
            b = bracket[bkey]
            print(f'[W43] {bkey:12s} predicted alpha*={pred["predicted_alpha_star_block"]} '
                  f'(hour {pred["argmax_hour"]}, min pibar over deviating hours {pred["min_pibar_over_deviating_hours"]}); '
                  f'measured bracket {b["measured_bracket"]}; inside={b["prediction_inside_measured_bracket"]}')
        for label, arms in hour_test.items():
            for arm, blocks in arms.items():
                for bkey, h in blocks.items():
                    print(f'[W43] hour test {label} {arm} {bkey}: {h["n_agree"]}/{h["n_hours_tested"]} agree; '
                          f'disagreements {[(d["hour"], round(d["margin_alpha_minus_alpha_t_over_alpha_t"], 4), round(d["observed_E_abs_d_mwh"], 5)) for d in h["disagreements"]]}')
        print(f'[W43] duplicate-alpha checks: {len(duplicates)}, all bitwise equal = '
              f'{all(d["bitwise_equal"] for d in duplicates)}')
        print(f'[W43] wrote {os.path.relpath(out_path, REPO)}')
        return 0
    except Exception as error:  # noqa: BLE001
        print(f'[W43] ERROR: {error!r}', file=sys.stderr)
        raise
    finally:
        GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
