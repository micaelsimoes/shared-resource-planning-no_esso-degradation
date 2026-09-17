"""
P5.15 Addendum 20 (s38 preparation worker task) -- TWO-cycle preflights for
arms `s38_A_tau0` and `s38_B_pfbal`, NOT the 300-cycle gates themselves.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 20; binding specification
data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json.

Runs `p515_g_g1_g4_admm_gates.run_s38_arm(<arm_key>, num_max_iters_override=2,
output_root_override=...)` BY IMPORT (never through the harness's own
`__main__` -- the Planner launches the 300-cycle gates; this script never
does) for EXACTLY TWO ADMM cycles through the REAL arm machinery (rho v/pf/ess
fixed at the spec v9 base, the arm's own tau and balancing_exempt_channels,
standalone shared-ESS initialization, the Addendum 20 freeze policy -- the
SAME configuration `_s38_configure_hook` applies to the real 300-cycle run),
into a fresh smoke root `data/SRP1/Results/P515S38/preflight_<A|B>/` (refuses
to overwrite).

Acquires the harness's OWN exclusive run lock
(`p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`) FIRST, unconditionally
-- refusing if `.p515_g_gate.lock` already exists (held by any other copy of
`p515_g_g1_g4_admm_gates.py`, gate OR preflight) or if the lock file cannot be
created; this is the harness's own mechanism, reused here (not
reimplemented), so a preflight and a real gate launch (or two preflights) can
never collide.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s38_preflight.py A \
        > data/SRP1/Results/P515S38/preflight_A_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s38_preflight.py B \
        > data/SRP1/Results/P515S38/preflight_B_launch.log 2>&1

Run ONE AT A TIME, attached (no `screen`/`nohup`/`&`), stderr captured via the
shell redirection above (this script does not redirect its own stderr).

Verifies:
  1. Bounded solve-profile identity (`51*(2+1)` + recovery-retry overhead),
     exactly the s37 preflight's own method.
  2. tau/gamma IN FORCE per cycle, read from each cycle's OWN
     `gamma_{v,pf,ess}_after` diagnostic field: arm A -- 0 on every channel,
     every cycle (tau=0); arm B -- equals that cycle's OWN
     `rho_{v,pf,ess}_after` on every channel, every cycle (tau=1,
     `gamma_policy=tied_to_rho`).
  3. PF action label: 'exempt (fixed)' every cycle in A; live labels
     ('increased'/'decreased'/'held') every cycle in B. ESS action label:
     'exempt (fixed)' every cycle in BOTH arms. V: live labels in BOTH arms.
  4. The PF capture identity (`pf_entry_stride_<arm>.jsonl`) holds every
     cycle -- RE-VERIFIED here by reading the sidecar back (the run itself
     would already have raised via `s38_pf_capture_hooks` had it failed;
     this is independent confirmation, not a repeat of the same check).
  5. Every capture path populated (recourse-jump, ess-entry-stride, SoH-
     floor, PF-entry-stride sidecars; boyd_terminal.json; g_<arm>.json;
     interface voltage/settlement terminal; component levels; ESSO models
     pickle).
  6. The evaluator (`p515_s38_evaluate.py`) runs on THIS preflight's own
     output in DRY mode (zero solves, no write).

Reports cycle-1/2 rho/gamma/action per channel, wall time, via the
evaluator's own helper functions where applicable (never reimplemented).
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s38_evaluate as E38  # noqa: E402

ARM_KEY_BY_LABEL = {'A': 's38_A_tau0', 'B': 's38_B_pfbal'}
OUT_DIR_BY_LABEL = {
    'A': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38', 'preflight_A'),
    'B': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38', 'preflight_B'),
}
NUM_CYCLES = 2
# 51 solves per cycle + 51 for initialization -- the SAME identity every
# other arm's own `run_admm_arm` reports (`51 * len(rows) + 51`); a
# recovered network failure adds EXTRA solve attempts on top (tier-1 retry
# = +1 solve, tier-2 retry = +2 solves per recovered event).
BASE_PERMITTED_SOLVES = 51 * (NUM_CYCLES + 1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ARM_KEY_BY_LABEL:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]
    arm_key = ARM_KEY_BY_LABEL[label]
    out_dir = OUT_DIR_BY_LABEL[label]

    # Refuse to run concurrently with any other copy of this harness (gate
    # OR preflight) -- reuses the harness's OWN lock mechanism, never
    # reimplemented.
    G._acquire_exclusive_run_lock()

    report, report_path = G.run_s38_arm(
        arm_key, num_max_iters_override=NUM_CYCLES, output_root_override=out_dir)

    print(f'[S38 preflight {label}] arm report: {report_path}')
    print(f"[S38 preflight {label}] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']} "
          f"wall_clock_s={report['wall_clock_s']}")

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
        f'[S38 preflight {label}] solve-profile identity: base={BASE_PERMITTED_SOLVES} '
        f'(51*({NUM_CYCLES}+1)) + retry_overhead={retry_overhead} '
        f'(tier1={n_tier1}*1 + tier2={n_tier2}*2) = expected {expected_permitted_solves}; '
        f'observed permitted_solve={observed_permitted_solves}; exact={solve_identity_exact}'
    )
    if not solve_identity_exact:
        raise RuntimeError(
            f'S38 preflight {label} solve-profile identity FAILED: expected '
            f'{expected_permitted_solves} (base {BASE_PERMITTED_SOLVES} + retry overhead '
            f'{retry_overhead}), observed {observed_permitted_solves}')

    # ---- (2)/(3) tau/gamma in force + action labels, per cycle -----------
    tau_expected = G.S38_ARMS[arm_key]['tau']
    exempt_channels = set(G.S38_ARMS[arm_key]['exempt_channels'])
    per_cycle_channel_check = []
    all_gamma_ok, all_labels_ok = True, True
    for r in rows:
        entry = {'cycle': r['cycle']}
        for ch in ('v', 'pf', 'ess'):
            rho_after = r.get(f'rho_{ch}_after')
            gamma_after = r.get(f'gamma_{ch}_after')
            action = r.get(f'rho_{ch}_action')
            expected_gamma = (0.0 if tau_expected == 0.0 else (tau_expected * rho_after
                                                                if rho_after is not None else None))
            gamma_ok = (gamma_after == expected_gamma) if expected_gamma is not None else False
            if ch in exempt_channels:
                label_ok = (action == 'exempt (fixed)')
            else:
                label_ok = action in ('increased', 'decreased', 'held')
            all_gamma_ok = all_gamma_ok and gamma_ok
            all_labels_ok = all_labels_ok and label_ok
            entry[ch] = {'rho_after': rho_after, 'gamma_after': gamma_after,
                        'gamma_expected': expected_gamma, 'gamma_ok': gamma_ok,
                        'action': action, 'exempt': ch in exempt_channels, 'label_ok': label_ok}
        per_cycle_channel_check.append(entry)
    print(f'[S38 preflight {label}] tau={tau_expected}, exempt_channels={sorted(exempt_channels)}, '
          f'per-cycle channel check: {json.dumps(per_cycle_channel_check, default=str)}')

    # ---- (4) PF capture identity holds every cycle, re-verified from the
    #      sidecar (never re-trusted only from the run's own in-flight raise) -
    pf_hits = glob.glob(os.path.join(out_dir, 'pf_entry_stride_*.jsonl'))
    if not pf_hits:
        raise RuntimeError(f'S38 preflight {label}: no pf_entry_stride sidecar found under {out_dir}')
    pf_stride_path = pf_hits[0]
    pf_identity_per_cycle = []
    with open(pf_stride_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pf_identity_per_cycle.append({
                'cycle': rec['cycle'], 'identity_holds': rec['identity_holds'],
                'rel_err_r': rec['rel_err_r'], 'rel_err_s': rec['rel_err_s'],
                'n_entries': len(rec.get('entries', [])),
            })
    pf_identity_all_hold = (len(pf_identity_per_cycle) == NUM_CYCLES
                            and all(e['identity_holds'] for e in pf_identity_per_cycle))
    print(f'[S38 preflight {label}] PF capture identity per cycle: {pf_identity_per_cycle}')

    # ---- (5) every capture path populated ----------------------------------
    settlement_hits = glob.glob(os.path.join(out_dir, 'interface_settlement_detail_*.json'))
    capture_paths = {
        'recourse_jump_sidecar': report.get('s34_recourse_jump_sidecar_path'),
        'ess_entry_stride_sidecar': report.get('s34_ess_entry_stride_sidecar_path'),
        'soh_floor_sidecar': report.get('s35ref_soh_floor_sidecar_path'),
        'pf_entry_stride_sidecar': report.get('s38_pf_entry_stride_sidecar_path'),
        'boyd_terminal': os.path.join(out_dir, 'boyd_terminal.json'),
        'g_baseline': report_path,
        'interface_voltage_terminal': os.path.join(out_dir, 'interface_voltage_terminal.json'),
        'interface_settlement_detail': settlement_hits[0] if settlement_hits else None,
        'component_levels_terminal': os.path.join(out_dir, 'component_levels_terminal.json'),
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
    print(f'[S38 preflight {label}] capture paths populated: {capture_paths_populated}')

    # ---- (6) the evaluator runs on the preflight output in DRY mode -------
    original_root = E38.ARM_ROOTS[arm_key]
    E38.ARM_ROOTS[arm_key] = out_dir
    try:
        eval_rc = E38.main([out_dir, '--dry-run'])
    finally:
        E38.ARM_ROOTS[arm_key] = original_root
    print(f'[S38 preflight {label}] evaluator dry-run return code: {eval_rc}')

    payload = {
        'stage': f'P5.15 Addendum 20 (s38 preparation worker task) -- {arm_key} two-cycle preflight',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 20',
                      'data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json'],
        'label': label, 'arm': arm_key, 'num_cycles': NUM_CYCLES,
        'out_dir': os.path.relpath(out_dir, REPO),
        'solve_profile_identity': {
            'base_permitted_solves': BASE_PERMITTED_SOLVES, 'n_tier1_recovered': n_tier1,
            'n_tier2_recovered': n_tier2, 'retry_overhead': retry_overhead,
            'expected_permitted_solves': expected_permitted_solves,
            'observed_permitted_solves': observed_permitted_solves,
            'exact': solve_identity_exact,
        },
        'tau_expected': tau_expected, 'exempt_channels': sorted(exempt_channels),
        'per_cycle_channel_check': per_cycle_channel_check,
        'all_gamma_in_force_correct': all_gamma_ok, 'all_action_labels_correct': all_labels_ok,
        'pf_capture_identity_per_cycle': pf_identity_per_cycle,
        'pf_capture_identity_all_hold': pf_identity_all_hold,
        'pf_entry_stride_path': os.path.relpath(pf_stride_path, REPO),
        'capture_paths_populated': capture_paths_populated,
        'all_capture_paths_populated': all_capture_paths_populated,
        'evaluator_dry_run_return_code': eval_rc,
        'network_failures_summary': report['network_failures_summary'],
        'local_solve_failures': report['local_solve_failures'],
        'wall_clock_s': report['wall_clock_s'],
    }
    payload_path = os.path.join(out_dir, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S38 preflight {label}] wrote {payload_path}')

    overall_ok = (
        solve_identity_exact and all_gamma_ok and all_labels_ok and pf_identity_all_hold
        and all_capture_paths_populated and eval_rc == 0
    )
    print(f'[S38 preflight {label}] overall_ok={overall_ok}')
    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
