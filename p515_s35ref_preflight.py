"""
P5.15 Addendum 16 item 1 worker task -- ONE-cycle preflight for the s35ref
arm ("run 1", reference equilibrium), NOT the 500-cycle gate itself.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 16 item 1; binding
specification data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json
(supersedes v4 data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json).
Runs `run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s
own `__main__` -- the Planner launches the 500-cycle gate, THIS script never
does, and the Worker task that produced it explicitly forbids launching
anything longer than this one-cycle preflight) for EXACTLY ONE ADMM cycle
through the (real) `s35ref` arm machinery (case-file rho in force --
rho_ess = 0.1125 on every network and the ESSO, data/SRP1/SRP1_params.json;
`apply_rho=False`; full per-cycle diagnostics captured in the trajectory,
including the v4/v5 sigma/S_ref/al_scale_esso/freeze fields; the
`s35ref_capture_hooks` recourse-jump, ESS-entry-stride and NEW SoH-floor-
multiplier + per-cohort-year-EFC sidecars wired exactly as the real gate
wires them), into a fresh smoke root
`data/SRP1/Results/P515S35/preflight_ref/`, then runs the s35ref terminal
writer (`write_boyd_terminal_s35ref`, zero extra solves) on the final
models.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s35ref_preflight.py

DECISIVE for this preflight (per the Worker task, NOT the D5 detector gate --
that compares against s33e2 at cycle labels '001'/'002' and was already
passed by s34's own two-cycle preflight; it is deliberately NOT re-run here,
since spec v5 changes nothing D5-related and this preflight is one cycle,
not two -- see WORKER_REPORT_S35REF_PREP.md "Unexpected findings" for this
scope decision):
  1. the rule-eleven checklist (`assert_s35ref_capture_paths`) passes;
  2. rho_ess IN FORCE is 0.1125 on every network and the ESSO;
  3. the SoH-floor sidecar is written with the EXPECTED number of rows per
     node (matching the PRE-SOLVE `_identify_soh_floor_rows` count), finite
     SoH values, and a dual key present per row;
  4. the equality rows are excluded BY CONSTRUCTION (`_identify_soh_floor_rows`
     raises if a candidate floor row is itself an equality row, or if either
     preceding row of its triple is NOT an equality -- checked for every
     triple, not sampled; this preflight would not have reached the run at
     all if that had failed);
  5. EFC/day per cohort-year is populated on every floor row.

Reports the floor rows' SoH and duals at cycle 1 -- EXPECTED to show every
row inactive (SoH far from soh_min) this early; that is the documented
prediction, not a failure (the frozen spec's own `predictions.stop_cycle`
puts the floor binding near cycle ~440).
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
import p515_s34_preflight as S34PF  # noqa: E402 -- reused BY CALLING, not copied

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'preflight_ref')

NUM_CYCLES = 1
# 51 solves per cycle + 51 for initialization, per the arm's own identity
# (see run_admm_arm / p515_s34_preflight.py's own PERMITTED_SOLVES, whose
# two-cycle preflight was 51*2 + 51 = 153); one cycle is therefore
# 51*1 + 51 = 102.
PERMITTED_SOLVES = 51 * NUM_CYCLES + 51


def _floor_sidecar_check(sidecar_path, expected_counts_by_node):
    """Reads the FIRST (cycle 1, and only, in this one-cycle preflight) line
    of the SoH-floor sidecar this run itself wrote and checks: row counts
    per node match the PRE-SOLVE expected counts, every SoH value is finite,
    every entry carries a 'dual' key (value may legitimately be None only if
    the ESSO capture layer could not resolve model.dual for that specific
    solve -- reported, not silently assumed), and (informationally) whether
    any row is already active (expected False this early)."""
    if not os.path.exists(sidecar_path):
        return {'available': False, 'path': sidecar_path}
    lines = []
    with open(sidecar_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    if not lines:
        return {'available': False, 'reason': 'sidecar empty', 'path': sidecar_path}

    cycle1 = lines[0]
    entries = cycle1.get('entries', [])
    by_node = defaultdict(list)
    for entry in entries:
        by_node[entry['node_id']].append(entry)

    row_counts_by_node = {node_id: len(rows) for node_id, rows in by_node.items()}
    row_counts_match_expected = {
        node_id: (row_counts_by_node.get(node_id) == expected_counts_by_node.get(node_id))
        for node_id in set(row_counts_by_node) | set(expected_counts_by_node)
    }
    all_soh_finite = all(
        isinstance(e.get('es_soh_per_unit_cumul'), (int, float)) and math.isfinite(e['es_soh_per_unit_cumul'])
        for e in entries
    ) if entries else False
    all_rows_have_dual_key = all(('dual' in e) for e in entries) if entries else False
    finite_duals = [e['dual'] for e in entries if isinstance(e.get('dual'), (int, float)) and math.isfinite(e['dual'])]
    any_active = any(e.get('active') for e in entries)
    efc_populated = all(('efc_per_day' in e) for e in entries) if entries else False

    return {
        'available': True,
        'cycle': cycle1.get('cycle'),
        'dual_sign_convention': cycle1.get('dual_sign_convention'),
        'n_entries': len(entries),
        'n_nodes': len(by_node),
        'row_counts_by_node': row_counts_by_node,
        'expected_row_counts_by_node': dict(expected_counts_by_node),
        'row_counts_match_expected': row_counts_match_expected,
        'all_row_counts_match_expected': all(row_counts_match_expected.values()) if row_counts_match_expected else False,
        'all_soh_finite': all_soh_finite,
        'all_rows_have_dual_key': all_rows_have_dual_key,
        'n_duals_finite': len(finite_duals),
        'n_duals_total': len(entries),
        'any_row_active_at_cycle1': any_active,
        'efc_per_day_populated_on_every_row': efc_populated,
        'entries': entries,
    }


def main():
    G._require_fresh_output_root(OUT_DIR)

    precheck_eval_id = 'p515s35ref_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist, floor_rows_by_node = G.assert_s35ref_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S35REF preflight] capture-path pre-flight passed (spec v5): {checklist}')
    print(
        f'[S35REF preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
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
    print(f'[S35REF preflight] spec_file={G.S35REF_SPEC_PATH} spec_sha256={G.S35REF_SPEC_SHA256}')

    rho_ess_ok = all(float(v) == G.S35REF_INITIAL_RHO_ESS for v in precheck_admm_params.rho['ess'].values())
    print(f'[S35REF preflight] rho_ess == {G.S35REF_INITIAL_RHO_ESS} on every network and the ESSO: {rho_ess_ok} '
          f'({dict(precheck_admm_params.rho["ess"])})')

    expected_floor_counts = {node_id: len(rows_) for node_id, rows_ in floor_rows_by_node.items()}
    print(f'[S35REF preflight] pre-solve floor row counts by node (Addendum 16, decisive): {expected_floor_counts}')

    recourse_jump_path = os.path.join(OUT_DIR, 'recourse_jump_sidecar_preflight.jsonl')
    ess_stride_path = os.path.join(OUT_DIR, 'ess_entry_stride_preflight.jsonl')
    floor_sidecar_path = os.path.join(OUT_DIR, 'soh_floor_sidecar_preflight.jsonl')

    def hook(planning, sed, models, rows, report, out_dir, label):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, G.REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, G.REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, G.REPO)
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                      floor_rows_by_node=floor_rows_by_node,
                                      floor_sidecar_path=floor_sidecar_path)

    with G.s35ref_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                 floor_rows_by_node, stride=1):
        report, report_path = G.run_admm_arm(
            'preflight', OUT_DIR, k_override=None, eval_id='p515s35ref_preflight',
            num_max_iters_override=NUM_CYCLES, apply_rho=False, full_diagnostics_in_rows=True,
            post_run_hook=hook)

    print(f'[S35REF preflight] arm report: {report_path}')
    print(f"[S35REF preflight] cycles_run={report['cycles_run']} "
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
        f'[S35REF preflight] solve counts checked EXACTLY against {PERMITTED_SOLVES} '
        f"(run_admm_arm's own internal SolveProfileGuard, N.PERMITTED call sites; "
        f"1 cycle plus initialization -- s34's own two-cycle preflight was 153 = 51*3, "
        f"so 1 cycle is 51*2 = 102): "
        f'observed={inner_observed} exact={inner_counts_exact} '
        f'identity_holds={report["solve_profile"].get("identity_holds")}'
    )

    rows = report['cycle_trajectory']
    if len(rows) != NUM_CYCLES:
        raise RuntimeError(f'expected exactly {NUM_CYCLES} cycle, got {len(rows)}')

    # ---- capture-field completeness / finiteness (reuses s34 preflight's
    #      OWN `_field_completeness`, BY CALLING IT -- not copied) ---------
    all_fields_ok = True
    field_completeness_by_cycle = {}
    for row in rows:
        ok, missing, field_report = S34PF._field_completeness(row)
        all_fields_ok = all_fields_ok and ok
        field_completeness_by_cycle[row['cycle']] = {
            'all_ok': ok, 'missing_or_nonfinite': missing, 'per_field': field_report,
        }
        print(f"[S35REF preflight] cycle {row['cycle']}: all required fields populated: {ok}"
              + ('' if ok else f' MISSING/NONFINITE: {missing}'))

    for row in rows:
        print(
            f"[S35REF preflight] cycle {row['cycle']} | sigma_fixed={row.get('sigma_fixed'):.6e} | "
            f"sigma_computed={row.get('sigma_computed'):.6e} | al_scale_esso={row.get('al_scale_esso'):.6e} | "
            f"S_ref={row.get('shared_ess_reference_rating_mva')} | efc_per_day_max={row.get('efc_per_day_max')}"
        )
        for group in ('v', 'pf', 'ess'):
            print(
                f"[S35REF preflight] cycle {row['cycle']} {group.upper()} | "
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

    cycle1 = rows[0]
    cycle1_rho_rule_ok = all(
        cycle1.get(f'rho_{g}_action') in ('increased', 'decreased', 'held')
        and cycle1.get(f'rho_frozen_{g}') is False
        for g in ('v', 'pf', 'ess')
    )
    print(f'[S35REF preflight] cycle-1 rho actions follow the v5 rule (adaptive, unfrozen): {cycle1_rho_rule_ok} '
          f"(actions: v={cycle1.get('rho_v_action')}, pf={cycle1.get('rho_pf_action')}, ess={cycle1.get('rho_ess_action')})")

    # al_scale_esso > 1 numeric check (deferred to the first cycle, same as
    # s34's own dispatch print notes -- unresolved before any solve) --------
    al_scale_val = cycle1.get('al_scale_esso')
    al_scale_gt_1 = isinstance(al_scale_val, (int, float)) and al_scale_val > 1.0
    print(f'[S35REF preflight] al_scale_esso > 1 at cycle 1: {al_scale_gt_1} (value={al_scale_val})')

    # ---- SoH floor sidecar + EFC per cohort-year (Addendum 16, decisive) --
    floor_check = _floor_sidecar_check(floor_sidecar_path, expected_floor_counts)
    print(f"[S35REF preflight] floor sidecar check: available={floor_check.get('available')} "
          f"n_entries={floor_check.get('n_entries')} "
          f"row_counts_match_expected={floor_check.get('all_row_counts_match_expected')} "
          f"all_soh_finite={floor_check.get('all_soh_finite')} "
          f"all_rows_have_dual_key={floor_check.get('all_rows_have_dual_key')} "
          f"n_duals_finite={floor_check.get('n_duals_finite')}/{floor_check.get('n_duals_total')} "
          f"efc_per_day_populated_on_every_row={floor_check.get('efc_per_day_populated_on_every_row')} "
          f"any_row_active_at_cycle1={floor_check.get('any_row_active_at_cycle1')} "
          f"(expected False -- too early, not a failure)")
    for e in floor_check.get('entries', []):
        print(f"[S35REF preflight] floor row node={e['node_id']} y_inv={e['y_inv']} y={e['y']} "
              f"constraint_idx={e['constraint_idx']} soh={e.get('es_soh_per_unit_cumul')} "
              f"soh_min={e.get('soh_min')} dual={e.get('dual')} active={e.get('active')} "
              f"efc_per_day={e.get('efc_per_day')}")

    payload = {
        'stage': 'P5.15 Addendum 16 item 1 worker task -- one-cycle preflight (spec v5, run 1 prep)',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 16 item 1',
            'data/SRP1/Results/P515S35/frozen_s35_reference_spec_v5_995548ab.json',
        ],
        'spec_file': os.path.relpath(G.S35REF_SPEC_PATH, G.REPO),
        'spec_file_sha256': G.S35REF_SPEC_SHA256,
        'checklist': checklist,
        'num_cycles': NUM_CYCLES,
        'permitted_solves': PERMITTED_SOLVES,
        'cycles_rows': rows,
        'field_completeness_by_cycle': field_completeness_by_cycle,
        'field_completeness_all_ok': all_fields_ok,
        'cycle1_rho_rule_ok': cycle1_rho_rule_ok,
        'rho_ess_matches_0p1125': rho_ess_ok,
        'al_scale_esso_gt_1_at_cycle1': al_scale_gt_1,
        'expected_floor_row_counts_by_node_pre_solve': expected_floor_counts,
        'floor_sidecar_check': floor_check,
        'equality_rows_excluded_by_construction': (
            '_identify_soh_floor_rows raises if a candidate floor row is itself an '
            'equality, or if either of the two preceding rows in its triple is NOT '
            'an equality -- checked for every (y_inv, y) triple at the pre-solve '
            'probe, not sampled; floor_rows_by_node therefore cannot contain an '
            'equality-row constraint_idx if this preflight reached this point '
            '(it did not raise).'
        ),
        'd5_detector_gate_not_rerun_here': (
            'the D5 (a_D5_esso_scaling) detector gate compares THIS run vs the '
            's33e2 baseline at cycle labels 001/002 and was already run and PASSED '
            'by s34\'s own two-cycle preflight (unchanged mechanism in v5); this '
            'one-cycle preflight does not repeat it -- see WORKER_REPORT_S35REF_PREP.md'
        ),
        'solve_profile': report['solve_profile'],
        'solve_profile_permitted_exact': PERMITTED_SOLVES,
        'solve_profile_counts_exact': inner_counts_exact,
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S35REF preflight] wrote {payload_path}')

    overall_ok = (
        all_fields_ok and inner_counts_exact and cycle1_rho_rule_ok and rho_ess_ok
        and floor_check.get('available') and floor_check.get('all_row_counts_match_expected')
        and floor_check.get('all_soh_finite') and floor_check.get('all_rows_have_dual_key')
        and floor_check.get('efc_per_day_populated_on_every_row')
    )
    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
