"""
P5.15 Step 3.4 (+3.3(b) folded in) worker task -- two-cycle preflight, WITH
the D5 detector gate (frozen spec v4, `changes_from_v3.a_D5_esso_scaling.preflight_gate`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5; binding
specification data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json
(supersedes v3 data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json).
Runs `run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s
own `__main__` -- the Planner launches the 150-cycle gate, not this script)
for EXACTLY TWO ADMM cycles through the (real) `s34` arm machinery
(case-file rho in force, `apply_rho=False`; full per-cycle diagnostics
captured in the trajectory, including the v4 sigma/S_ref/al_scale_esso/
freeze fields; the `s34_capture_hooks` recourse-jump and ESS-entry-stride
sidecars wired exactly as the real gate wires them), into a fresh smoke
root `data/SRP1/Results/P515S34/preflight/`, then runs the s34 terminal
writer (`write_boyd_terminal_s34`, zero extra solves) on the final models.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s34_preflight.py

DECISIVE, per the spec's `preflight_gate`: from THIS run's ESSO solves
(`leak_classification_preflight.jsonl`, cycles 1 and 2), compares
`mu_unscaled` (mu_final), `s_obj`, `complementarity_ratio_max` and
`spurious_throughput_measured`, per node, against the SAME cycle/node
entries in the COMMITTED s33e2 baseline
(`data/SRP1/Results/P515S33_E2_run/leak_classification_baseline.jsonl`,
READ ONLY). A quantity is flagged MATERIALLY WORSE if it increases by more
than 10x (an explicit, Worker-chosen order-of-magnitude convention -- the
spec states "worsens materially" without a numeric bound; this script
reports the ratio for every quantity regardless, so the Planner can apply a
different threshold). If ANY quantity is flagged, this script prints STOP,
does not report the run as ready to commit, and exits nonzero -- per the
task's explicit instruction, it does NOT proceed to redesign epsilon or the
ESSO tolerance itself.

Also verifies every v4 capture field is populated (including sigma_fixed/
sigma_computed/al_scale_esso/shared_ess_reference_rating_mva and the
per-channel freeze diagnostics), that cycle-1 rho actions follow the v4
rule (case-file rho in force, not N.RHO), and reports cycle-1/cycle-2 Boyd
values per channel, EFC/day, and the ESS per-entry step (from the
ess_entry_stride sidecar) for comparison against s33e2's cycle 1-2 -- which
this script does NOT expect to match, since rho, S_ref and the ESSO AL
scaling all changed (see the spec's `predictions`).
"""

import json
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34', 'preflight')
S33E2_LEAK_PATH = G.S34_S33E2_LEAK_PATH  # data/SRP1/Results/P515S33_E2_run/leak_classification_baseline.jsonl

NUM_CYCLES = 2
PERMITTED_SOLVES = 51 * NUM_CYCLES + 51  # 153 = 51 * 3, per the arm's own identity (see run_admm_arm)
MATERIAL_WORSENING_RATIO = 10.0  # explicit Worker convention, see module docstring

REQUIRED_FIELDS_PER_CHANNEL = (
    'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
    'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance',
)
ALLOWED_NONE_FIELDS = {'gap_proxy_G', 'gap_proxy_G_over_Q'}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _field_completeness(row):
    missing_or_nonfinite = []
    field_report = {}
    for group in ('v', 'pf', 'ess'):
        for field in REQUIRED_FIELDS_PER_CHANNEL:
            key = f'boyd_{group}_{field}'
            value = row.get(key)
            ok = _finite(value)
            field_report[key] = {'value': value, 'finite': ok}
            if not ok:
                missing_or_nonfinite.append(key)
        for key in (f'rho_{group}_before', f'rho_{group}_after', f'rho_{group}_action'):
            value = row.get(key)
            present = value is not None
            field_report[key] = {'value': value, 'present': present}
            if not present:
                missing_or_nonfinite.append(key)
        for key in (f'gamma_{group}_before', f'gamma_{group}_after',
                    f'rho_frozen_{group}', f'rho_unchanged_streak_{group}', f'rho_at_clamp_{group}'):
            value = row.get(key)
            present = value is not None
            field_report[key] = {'value': value, 'present': present}
            if not present:
                missing_or_nonfinite.append(key)

    # `objective_change_abs`/`objective_tolerance`/`objective_change_ratio`
    # are legitimately None at cycle 1 (no previous recourse yet -- see
    # `_run_operational_planning`'s "Recourse stationarity requires one
    # previous successful cycle."); recorded, not flagged as missing, same
    # convention as p515_s33e2_preflight.py's `_field_completeness`.
    for key in ('objective_change_abs', 'objective_tolerance', 'objective_change_ratio',
                'rho_freeze_active', 'gamma_policy', 'gamma_tau',
                'freeze_after_unchanged_cycles', 'freeze_backstop_cycle'):
        value = row.get(key)
        field_report[key] = {'value': value}

    for key in ('recourse', 'gross_operational_cost', 'sigma_fixed', 'sigma_computed',
                'al_scale_esso', 'shared_ess_reference_rating_mva'):
        value = row.get(key)
        ok = _finite(value)
        field_report[key] = {'value': value, 'finite': ok}
        if not ok:
            missing_or_nonfinite.append(key)

    # efc_per_day_max: spec-declared "None if unavailable that cycle" -- a
    # present KEY is required; the VALUE itself may legitimately be None.
    field_report['efc_per_day_max'] = {'value': row.get('efc_per_day_max'), 'key_present': ('efc_per_day_max' in row)}
    if 'efc_per_day_max' not in row:
        missing_or_nonfinite.append('efc_per_day_max')

    for key in ('gap_proxy_G', 'gap_proxy_Q', 'gap_proxy_G_over_Q', 'gap_proxy_G_reason'):
        value = row.get(key)
        field_report[key] = {'value': value}
        if key in ALLOWED_NONE_FIELDS:
            if value is not None:
                missing_or_nonfinite.append(f'{key}_expected_None_got_{value}')
        elif key == 'gap_proxy_Q':
            if not _finite(value):
                missing_or_nonfinite.append(key)
        elif key == 'gap_proxy_G_reason':
            if not value:
                missing_or_nonfinite.append(key)

    return (len(missing_or_nonfinite) == 0), missing_or_nonfinite, field_report


def _load_leak_jsonl(path):
    """{(node_id, cycle_label): entry} -- 'cycle_label' as the raw string the
    file stores (e.g. 'init', '001', '002')."""
    by_key = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_key[(entry['node_id'], str(entry['cycle']))] = entry
    return by_key


def _detector_gate(preflight_leak_path, baseline_leak_path):
    """Decisive comparison: for cycles '001' and '002', per node, compare
    mu_unscaled (mu_final), s_obj, complementarity_ratio_max and
    spurious_throughput_measured against the s33e2 baseline at the SAME
    cycle/node. A quantity is 'worse' when it INCREASES (mu/complementarity/
    spurious throughput are all quantities that should stay small; a
    material increase indicates the barrier or the leak got worse, not
    better)."""
    preflight = _load_leak_jsonl(preflight_leak_path)
    baseline = _load_leak_jsonl(baseline_leak_path)

    metrics = ('mu_unscaled', 's_obj', 'complementarity_ratio_max', 'spurious_throughput_measured')
    rows = []
    any_worse = False
    for cycle_label in ('001', '002'):
        for node_id in (5, 7, 9):
            key = (node_id, cycle_label)
            pre_entry = preflight.get(key)
            base_entry = baseline.get(key)
            row = {'cycle': cycle_label, 'node_id': node_id, 'preflight_entry_found': pre_entry is not None,
                   'baseline_entry_found': base_entry is not None}
            if pre_entry is None or base_entry is None:
                row['comparable'] = False
                rows.append(row)
                continue
            row['comparable'] = True
            for metric in metrics:
                pv = pre_entry.get(metric)
                bv = base_entry.get(metric)
                ratio = None
                worse = False
                if isinstance(pv, (int, float)) and isinstance(bv, (int, float)) and bv != 0:
                    ratio = pv / bv
                    worse = (ratio > MATERIAL_WORSENING_RATIO)
                row[metric] = {'preflight': pv, 's33e2_baseline': bv, 'ratio_preflight_over_baseline': ratio,
                                'material_worsening_flag': worse}
                any_worse = any_worse or worse
            rows.append(row)

    return {
        'material_worsening_ratio_threshold': MATERIAL_WORSENING_RATIO,
        'metrics_compared': metrics,
        'rows': rows,
        'any_material_worsening': any_worse,
        'pass': not any_worse,
    }


def _ess_entry_stride_step(sidecar_path):
    """Reads the s34_capture_hooks ESS-entry-stride sidecar (this preflight
    run's own, label 'preflight') and reports the per-entry z step between
    cycle 1 and cycle 2 -- the observable one-step motion within a 2-cycle
    preflight (no 'cycle 0' exists to diff cycle 1 against)."""
    if not os.path.exists(sidecar_path):
        return {'available': False, 'path': sidecar_path}

    by_cycle = {}
    with open(sidecar_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_cycle[entry['cycle']] = entry

    if 1 not in by_cycle or 2 not in by_cycle:
        return {'available': False, 'reason': 'cycle 1 or 2 not found in sidecar', 'cycles_present': sorted(by_cycle)}

    c1_by_key = {(e['node_id'], e['year'], e['day'], e['power_type']): e for e in by_cycle[1]['entries']}
    c2_by_key = {(e['node_id'], e['year'], e['day'], e['power_type']): e for e in by_cycle[2]['entries']}

    max_abs_step = 0.0
    max_step_entry = None
    steps = []
    for key, c1_entry in c1_by_key.items():
        c2_entry = c2_by_key.get(key)
        if c2_entry is None:
            continue
        for period, (z1, z2) in enumerate(zip(c1_entry['z'], c2_entry['z'])):
            step = abs(z2 - z1)
            steps.append(step)
            if step > max_abs_step:
                max_abs_step = step
                max_step_entry = {'node_id': key[0], 'year': key[1], 'day': key[2],
                                   'power_type': key[3], 'period': period, 'z1': z1, 'z2': z2, 'abs_step': step}

    mean_step = (sum(steps) / len(steps)) if steps else None
    return {
        'available': True, 'n_entries_compared': len(steps),
        'mean_abs_z_step_cycle1_to_cycle2': mean_step,
        'max_abs_z_step_cycle1_to_cycle2': max_abs_step,
        'max_abs_z_step_entry': max_step_entry,
        'efc_per_day_per_node_cycle1': by_cycle[1].get('efc_per_day_per_node'),
        'efc_per_day_per_node_cycle2': by_cycle[2].get('efc_per_day_per_node'),
    }


def main():
    G._require_fresh_output_root(OUT_DIR)

    if not os.path.exists(S33E2_LEAK_PATH):
        raise RuntimeError(
            f's33e2 baseline leak_classification file not found (needed READ ONLY, NOT re-run): {S33E2_LEAK_PATH}')

    precheck_eval_id = 'p515s34_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s34_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S34 preflight] capture-path pre-flight passed (spec v4): {checklist}')
    print(
        f'[S34 preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
        f'pf={precheck_admm_params.rho["pf"]}, ess={precheck_admm_params.rho["ess"]}; '
        f'boyd_eps_source={precheck_admm_params.boyd_eps_source}; '
        f'objective_scale={precheck_admm_params.objective_scale} '
        f'(source={precheck_admm_params.objective_scale_source}); '
        f'esso_al_scale={precheck_admm_params.esso_al_scale}; '
        f'shared_ess_reference_rating_mva={precheck_admm_params.shared_ess_reference_rating_mva}; '
        f'freeze_after_unchanged_cycles={precheck_admm_params.penalty_update["freeze_after_unchanged_cycles"]}; '
        f'freeze_backstop_cycle={precheck_admm_params.penalty_update["freeze_backstop_cycle"]}; '
        f'minimum_consecutive_converged_cycles={precheck_admm_params.minimum_consecutive_converged_cycles}'
    )
    print(f'[S34 preflight] spec_file={G.S34_SPEC_PATH} spec_sha256={G.S34_SPEC_SHA256}')

    recourse_jump_path = os.path.join(OUT_DIR, 'recourse_jump_sidecar_preflight.jsonl')
    ess_stride_path = os.path.join(OUT_DIR, 'ess_entry_stride_preflight.jsonl')

    def hook(planning, sed, models, rows, report, out_dir, label):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, REPO)
        G.write_boyd_terminal_s34(planning, sed, models, rows, report, out_dir, label)

    with G.s34_capture_hooks(recourse_jump_path, ess_stride_path, stride=1):
        report, report_path = G.run_admm_arm(
            'preflight', OUT_DIR, k_override=None, eval_id='p515s34_preflight',
            num_max_iters_override=NUM_CYCLES, apply_rho=False, full_diagnostics_in_rows=True,
            post_run_hook=hook)

    print(f'[S34 preflight] arm report: {report_path}')
    print(f"[S34 preflight] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")

    inner_observed = report['solve_profile'].get('observed', {})
    inner_counts_exact = (
        inner_observed.get('permitted_solve') == PERMITTED_SOLVES
        and inner_observed.get('permitted_exec') == PERMITTED_SOLVES
        and inner_observed.get('blocked_solve') == 0
        and inner_observed.get('blocked_exec') == 0
    )
    print(
        f'[S34 preflight] solve counts checked EXACTLY against {PERMITTED_SOLVES} '
        f"(run_admm_arm's own internal SolveProfileGuard, N.PERMITTED call sites): "
        f'observed={inner_observed} exact={inner_counts_exact} '
        f'identity_holds={report["solve_profile"].get("identity_holds")}'
    )

    rows = report['cycle_trajectory']
    if len(rows) != NUM_CYCLES:
        raise RuntimeError(f'expected exactly {NUM_CYCLES} cycles, got {len(rows)}')

    # ---- capture-field completeness / finiteness (both cycles) -----------
    all_fields_ok = True
    field_completeness_by_cycle = {}
    for row in rows:
        ok, missing, field_report = _field_completeness(row)
        all_fields_ok = all_fields_ok and ok
        field_completeness_by_cycle[row['cycle']] = {
            'all_ok': ok, 'missing_or_nonfinite': missing, 'per_field': field_report,
        }
        print(f"[S34 preflight] cycle {row['cycle']}: all required fields populated: {ok}"
              + ('' if ok else f' MISSING/NONFINITE: {missing}'))

    # ---- print cycle values per channel, plus v4 run-level constants -----
    for row in rows:
        print(
            f"[S34 preflight] cycle {row['cycle']} | sigma_fixed={row.get('sigma_fixed'):.6e} | "
            f"sigma_computed={row.get('sigma_computed'):.6e} | al_scale_esso={row.get('al_scale_esso'):.6e} | "
            f"S_ref={row.get('shared_ess_reference_rating_mva')} | efc_per_day_max={row.get('efc_per_day_max')}"
        )
        for group in ('v', 'pf', 'ess'):
            print(
                f"[S34 preflight] cycle {row['cycle']} {group.upper()} | "
                f"r={row.get(f'boyd_{group}_r'):.6e} | "
                f"s={row.get(f'boyd_{group}_s'):.6e} | "
                f"action={row.get(f'rho_{group}_action')} | "
                f"rho_before={row.get(f'rho_{group}_before'):.6e} | "
                f"rho_after={row.get(f'rho_{group}_after'):.6e} | "
                f"gamma_before={row.get(f'gamma_{group}_before'):.6e} | "
                f"gamma_after={row.get(f'gamma_{group}_after'):.6e} | "
                f"rho_frozen={row.get(f'rho_frozen_{group}')} | "
                f"rho_unchanged_streak={row.get(f'rho_unchanged_streak_{group}')} | "
                f"rho_at_clamp={row.get(f'rho_at_clamp_{group}')}"
            )

    # ---- cycle-1 rho actions follow the v4 rule: adaptive, case-file rho,
    #      no legacy freeze at cycle 1 (10 << cycle 1; backstop is cycle 60) -
    cycle1 = rows[0]
    cycle1_rho_rule_ok = all(
        cycle1.get(f'rho_{g}_action') in ('increased', 'decreased', 'held')
        and cycle1.get(f'rho_frozen_{g}') is False
        for g in ('v', 'pf', 'ess')
    )
    print(f'[S34 preflight] cycle-1 rho actions follow the v4 rule (adaptive, unfrozen): {cycle1_rho_rule_ok} '
          f"(actions: v={cycle1.get('rho_v_action')}, pf={cycle1.get('rho_pf_action')}, ess={cycle1.get('rho_ess_action')})")

    # ---- detector gate (DECISIVE) ------------------------------------------
    preflight_leak_path = os.path.join(OUT_DIR, 'leak_classification_preflight.jsonl')
    detector_gate = _detector_gate(preflight_leak_path, S33E2_LEAK_PATH)
    print(f"[S34 preflight] DETECTOR GATE (vs s33e2 baseline, threshold {MATERIAL_WORSENING_RATIO}x): "
          f"any_material_worsening={detector_gate['any_material_worsening']}")
    for row in detector_gate['rows']:
        if not row.get('comparable'):
            print(f"[S34 preflight] detector gate: NOT COMPARABLE at {row}")
            continue
        flags = [m for m in detector_gate['metrics_compared'] if row[m]['material_worsening_flag']]
        print(f"[S34 preflight] detector gate cycle={row['cycle']} node={row['node_id']} "
              f"worsened_metrics={flags or 'none'}")

    # ---- ESS per-entry step (cycle 1 -> cycle 2) ---------------------------
    ess_step = _ess_entry_stride_step(ess_stride_path)
    print(f'[S34 preflight] ESS per-entry z step (cycle 1 -> cycle 2): {ess_step}')

    payload = {
        'stage': 'P5.15 Step 3.4 (+3.3(b) folded in) worker task -- two-cycle preflight, D5 detector gate',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5',
            'data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json',
        ],
        'spec_file': os.path.relpath(G.S34_SPEC_PATH, REPO),
        'spec_file_sha256': G.S34_SPEC_SHA256,
        'checklist': checklist,
        'num_cycles': NUM_CYCLES,
        'cycles_rows': rows,
        'field_completeness_by_cycle': field_completeness_by_cycle,
        'field_completeness_all_ok': all_fields_ok,
        'cycle1_rho_rule_ok': cycle1_rho_rule_ok,
        'detector_gate': detector_gate,
        'ess_per_entry_step_cycle1_to_cycle2': ess_step,
        's33e2_baseline_leak_path_read_only': os.path.relpath(S33E2_LEAK_PATH, REPO),
        'solve_profile': report['solve_profile'],
        'solve_profile_permitted_exact': PERMITTED_SOLVES,
        'solve_profile_counts_exact': inner_counts_exact,
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S34 preflight] wrote {payload_path}')

    if not detector_gate['pass']:
        print('[S34 preflight] STOP: the D5 detector gate found a material worsening vs the s33e2 baseline. '
              'Per the task instruction, this script does NOT proceed to redesign epsilon or the ESSO tolerance -- '
              'report to the Planner.')
        sys.exit(2)

    if not (all_fields_ok and inner_counts_exact and cycle1_rho_rule_ok):
        sys.exit(1)


if __name__ == '__main__':
    main()
