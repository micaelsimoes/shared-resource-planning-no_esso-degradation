"""
P5.15 Addendum 19 (s37 preparation worker task) -- TWO-cycle preflight for
the s37_rho0p01 arm, NOT the 150-cycle gate itself.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 19; binding specification
data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json.

Runs `p515_g_g1_g4_admm_gates.run_s37_arm('s37_rho0p01', num_max_iters_
override=2, output_root_override=...)` BY IMPORT (never through the
harness's own `__main__` -- the Planner launches the 150-cycle gate; THIS
script never does) for EXACTLY TWO ADMM cycles through the REAL s37_rho0p01
arm machinery (rho_ess=0.01 fixed on every network + the esso, balancing
exemption on the ess channel, standalone shared-ESS initialization, v/pf
unchanged from run 1 -- the SAME configuration `_s37_configure_hook`
applies to the real 150-cycle run), into a fresh smoke root
`data/SRP1/Results/P515S37/preflight_rho0p01/` (refuses to overwrite).

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s37_preflight.py

Verifies:
  1. Bounded solve-profile identity: `run_admm_arm`'s OWN internal
     SolveProfileGuard (N.PERMITTED call sites, armed for the whole run) is
     checked EXACTLY against `51*3` (2 cycles + 1 initialization) PLUS the
     recovery-retry overhead THIS preflight's own network_failures file
     records (tier-1 recovery = +1 solve, tier-2 recovery = +2 solves per
     event) -- both the base count and the retry overhead are reported; a
     recovered failure must not break the identity (raises if it does).
  2. rho_ess = 0.01 everywhere (every network + the esso) and CONSTANT
     across both cycles, with the 'exempt (fixed)' action label, frozen
     from cycle 1, never at a rho clamp.
  3. V/PF balancing acts normally (adaptive, non-exempt, ordinary action
     labels).
  4. Every capture path populated (recourse-jump, ess-entry-stride, SoH-
     floor sidecars; boyd_terminal.json; g_<label>.json; interface
     voltage/settlement terminal writers; component levels; the ESSO
     models pickle).
  5. The evaluator (`p515_s37_evaluate.py`) runs on THIS preflight's own
     output in DRY mode (zero solves, no write) -- exercising the SAME
     production evaluation logic (certification, predictions, monitoring)
     the real gate's evaluation will use, on a 2-cycle run (an edge case
     for the cycles-2-20 RMS window, which this preflight cannot fill --
     reported, not treated as a failure).

Reports cycle-1/2 ESS step size, sign-change fraction, EFC, and channel
ratios, via the evaluator's OWN helper functions (`p515_s37_evaluate`),
never reimplemented.
"""
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s37_evaluate as E37  # noqa: E402

ARM_KEY = 's37_rho0p01'
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S37', 'preflight_rho0p01')
NUM_CYCLES = 2
# 51 solves per cycle + 51 for initialization (the SAME identity every
# other arm's own `run_admm_arm` reports as `51 * len(rows) + 51`, e.g.
# s34/s35ref's two-cycle preflights: 51*2 + 51 = 153 = 51*3); a recovered
# network failure adds EXTRA solve attempts on top of this base (tier-1
# retry = +1 solve, tier-2 retry = +2 solves per recovered event) -- the
# spec's own instruction for this preflight.
BASE_PERMITTED_SOLVES = 51 * (NUM_CYCLES + 1)


def main():
    report, report_path = G.run_s37_arm(
        ARM_KEY, num_max_iters_override=NUM_CYCLES, output_root_override=OUT_DIR)

    print(f'[S37 preflight] arm report: {report_path}')
    print(f"[S37 preflight] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")

    rows = report['cycle_trajectory']
    if len(rows) != NUM_CYCLES:
        raise RuntimeError(f'expected exactly {NUM_CYCLES} cycles, got {len(rows)}')

    # ---- (1) bounded solve-profile identity, WITH recovery-retry overhead -
    classes = report['network_failures_summary']['classes']
    n_tier1 = classes.get('recovered_tier1', 0)
    n_tier2 = classes.get('recovered_tier2', 0)
    retry_overhead = n_tier1 * 1 + n_tier2 * 2
    expected_permitted_solves = BASE_PERMITTED_SOLVES + retry_overhead
    observed_permitted_solves = report['solve_profile']['observed'].get('permitted_solve')
    solve_identity_exact = (observed_permitted_solves == expected_permitted_solves)
    print(
        f'[S37 preflight] solve-profile identity: base={BASE_PERMITTED_SOLVES} '
        f'(51*({NUM_CYCLES}+1)) + retry_overhead={retry_overhead} '
        f'(tier1={n_tier1}*1 + tier2={n_tier2}*2) = expected {expected_permitted_solves}; '
        f'observed permitted_solve={observed_permitted_solves}; exact={solve_identity_exact}'
    )
    if not solve_identity_exact:
        raise RuntimeError(
            f'S37 preflight solve-profile identity FAILED: expected {expected_permitted_solves} '
            f'(base {BASE_PERMITTED_SOLVES} + retry overhead {retry_overhead}), '
            f'observed {observed_permitted_solves}')

    # ---- (2) rho_ess = 0.01 everywhere, constant across both cycles, with
    #      the 'exempt (fixed)' action label, frozen, never at clamp -------
    rho_ess_values = [r.get('rho_ess_before') for r in rows] + [rows[-1].get('rho_ess_after')]
    # `rho_ess_before`/`_after` are `_get_admm_penalty_summary`'s AVERAGE
    # over every TSO/DSO/ESSO model (its own documented "averaging
    # convention") -- exact float equality to 0.01 is too strict (0.01 is
    # not exactly representable, and averaging several models' Params
    # introduces float64 rounding at the ~1e-17 relative level even when
    # every underlying Param is bit-identical); `math.isclose` at a tight
    # tolerance is the correct check.
    rho_ess_constant_at_0p01 = all(math.isclose(v, 0.01, rel_tol=1e-9) for v in rho_ess_values)
    ess_action_always_exempt = all(r.get('rho_ess_action') == 'exempt (fixed)' for r in rows)
    ess_frozen_every_cycle = all(r.get('rho_frozen_ess') is True for r in rows)
    ess_never_at_clamp = all(r.get('rho_at_clamp_ess') is False for r in rows)
    ess_exempt_diagnostic_true = all(r.get('balancing_exempt_ess') is True for r in rows)
    print(
        f'[S37 preflight] rho_ess constant at 0.01: {rho_ess_constant_at_0p01} ({rho_ess_values}); '
        f'action always "exempt (fixed)": {ess_action_always_exempt}; '
        f'frozen every cycle: {ess_frozen_every_cycle}; never at clamp: {ess_never_at_clamp}; '
        f'balancing_exempt_ess diagnostic true every cycle: {ess_exempt_diagnostic_true}'
    )

    # ---- (3) V/PF balancing acts normally (adaptive, unfrozen, ordinary
    #      action labels -- explicitly NOT exempt) --------------------------
    v_pf_normal = all(
        r.get(f'rho_{c}_action') in ('increased', 'decreased', 'held')
        and r.get(f'balancing_exempt_{c}') is False
        for r in rows for c in ('v', 'pf')
    )
    print(f'[S37 preflight] V/PF balancing acts normally (adaptive, non-exempt): {v_pf_normal} '
          f"(actions: {[{'cycle': r['cycle'], 'v': r.get('rho_v_action'), 'pf': r.get('rho_pf_action')} for r in rows]})")

    # ---- (4) every capture path populated ----------------------------------
    settlement_hits = glob.glob(os.path.join(OUT_DIR, 'interface_settlement_detail_*.json'))
    capture_paths = {
        'recourse_jump_sidecar': report.get('s34_recourse_jump_sidecar_path'),
        'ess_entry_stride_sidecar': report.get('s34_ess_entry_stride_sidecar_path'),
        'soh_floor_sidecar': report.get('s35ref_soh_floor_sidecar_path'),
        'boyd_terminal': os.path.join(OUT_DIR, 'boyd_terminal.json'),
        'g_baseline': report_path,
        'interface_voltage_terminal': os.path.join(OUT_DIR, 'interface_voltage_terminal.json'),
        'interface_settlement_detail': settlement_hits[0] if settlement_hits else None,
        'component_levels_terminal': os.path.join(OUT_DIR, 'component_levels_terminal.json'),
        'esso_models_pickle': report.get('esso_models_pickle', {}).get('path'),
    }
    capture_paths_populated = {}
    for name, rel_or_abs in capture_paths.items():
        if rel_or_abs is None:
            capture_paths_populated[name] = False
            continue
        abs_path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(REPO, rel_or_abs)
        capture_paths_populated[name] = os.path.exists(abs_path) and os.path.getsize(abs_path) > 0
    all_capture_paths_populated = all(capture_paths_populated.values())
    print(f'[S37 preflight] capture paths populated: {capture_paths_populated}')

    # ---- (5) the evaluator runs on the preflight output in DRY mode -------
    #      (p515_s37_evaluate.ARM_ROOTS is keyed on the REAL arm output
    #      roots; temporarily pointed at THIS preflight's own directory for
    #      the duration of one in-process call, then restored -- the SAME
    #      production evaluation function, not reimplemented, exercised on
    #      an edge-case-sized (2-cycle) run.)
    original_root = E37.ARM_ROOTS[ARM_KEY]
    E37.ARM_ROOTS[ARM_KEY] = OUT_DIR
    try:
        eval_rc = E37.main(['--dry-run'])
    finally:
        E37.ARM_ROOTS[ARM_KEY] = original_root
    print(f'[S37 preflight] evaluator dry-run return code: {eval_rc}')

    # ---- report cycle-1/2 ESS step size, sign-change fraction, EFC, and
    #      channel ratios (evaluator's OWN helper functions, not
    #      reimplemented) ---------------------------------------------------
    vectors, stride_path = E37._load_ess_stride_p_vectors(OUT_DIR)
    step_series = E37._step_series(vectors)
    osc_series = E37._oscillation_series(step_series)
    cycle_monitoring = {}
    for r in rows:
        c = r['cycle']
        step_entry = step_series.get(c)
        cycle_monitoring[c] = {
            'efc_per_day_max': r.get('efc_per_day_max'),
            'channel_ratios': {
                g: {'primal_ratio': r.get(f'boyd_{g}_primal_ratio'), 'dual_ratio': r.get(f'boyd_{g}_dual_ratio')}
                for g in ('v', 'pf', 'ess')
            },
            'step_size': {'rms': step_entry['rms'], 'max_abs': step_entry['max_abs']} if step_entry else None,
            'oscillation': osc_series.get(c),
        }
        print(f"[S37 preflight] cycle {c}: efc_per_day_max={cycle_monitoring[c]['efc_per_day_max']} "
              f"step_size={cycle_monitoring[c]['step_size']} "
              f"oscillation={cycle_monitoring[c]['oscillation']} "
              f"channel_ratios={cycle_monitoring[c]['channel_ratios']}")
    note_step_size_cycle_1 = (
        'cycle 1 has no preceding cycle captured in THIS preflight (z_0 is the construction-time '
        'consensus, not part of the stride sidecar), so its own step size is None here -- expected, '
        'not a failure; the real 150-cycle run captures cycle 1 the same way.'
    )

    payload = {
        'stage': 'P5.15 Addendum 19 (s37 preparation worker task) -- s37_rho0p01 two-cycle preflight',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 19',
                      'data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json'],
        'arm': ARM_KEY, 'num_cycles': NUM_CYCLES, 'out_dir': os.path.relpath(OUT_DIR, REPO),
        'solve_profile_identity': {
            'base_permitted_solves': BASE_PERMITTED_SOLVES, 'n_tier1_recovered': n_tier1,
            'n_tier2_recovered': n_tier2, 'retry_overhead': retry_overhead,
            'expected_permitted_solves': expected_permitted_solves,
            'observed_permitted_solves': observed_permitted_solves,
            'exact': solve_identity_exact,
        },
        'rho_ess_check': {
            'values_before_then_terminal_after': rho_ess_values, 'constant_at_0p01': rho_ess_constant_at_0p01,
            'action_always_exempt_fixed': ess_action_always_exempt,
            'frozen_every_cycle': ess_frozen_every_cycle, 'never_at_clamp': ess_never_at_clamp,
            'balancing_exempt_diagnostic_true_every_cycle': ess_exempt_diagnostic_true,
        },
        'v_pf_normal_balancing': v_pf_normal,
        'capture_paths_populated': capture_paths_populated,
        'all_capture_paths_populated': all_capture_paths_populated,
        'evaluator_dry_run_return_code': eval_rc,
        'ess_entry_stride_sidecar_path': os.path.relpath(stride_path, REPO) if stride_path else None,
        'cycle_monitoring': cycle_monitoring,
        'note_step_size_cycle_1': note_step_size_cycle_1,
        'network_failures_summary': report['network_failures_summary'],
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S37 preflight] wrote {payload_path}')

    overall_ok = (
        solve_identity_exact and rho_ess_constant_at_0p01 and ess_action_always_exempt
        and ess_frozen_every_cycle and ess_never_at_clamp and v_pf_normal
        and all_capture_paths_populated and eval_rc == 0
    )
    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
