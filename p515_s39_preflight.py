"""
P5.15 Addendum 21 (s39 preparation worker task, W1 item 6) -- TWO-cycle
preflights for arms `s39_C` and `s39_D`, NOT the 300-cycle gates themselves.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 21; binding specification
data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json.

Runs `p515_g_g1_g4_admm_gates.run_s39_arm(<arm_key>, num_max_iters_override=2,
output_root_override=...)` BY IMPORT (never through the harness's own
`__main__` -- the Planner launches the 300-cycle gates; this script never
does) for EXACTLY TWO ADMM cycles through the REAL arm machinery (rho v/pf/
ess fixed at the spec v10 base, tau=0.0, the arm's own `balancing_exempt_
channels`/`balancing_exempt_until`, standalone shared-ESS initialization,
`minimum_consecutive_converged_cycles=10`, the Addendum 20 freeze policy --
the SAME configuration `_s39_configure_hook` applies to the real 300-cycle
run), into a fresh smoke root `data/SRP1/Results/P515S39/preflight_<C|D>/`
(refuses to overwrite). Because `run_s39_arm` derives its working-dir ids
from `mode` ('preflight' here, since `output_root_override` is given -- the
W1 structural fix), this preflight run can NEVER collide with the later
real 300-cycle launch's own ids, regardless of invocation order.

Acquires the harness's OWN exclusive run lock
(`p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`) FIRST, unconditionally
-- refusing if `.p515_g_gate.lock` already exists (held by any other copy of
`p515_g_g1_g4_admm_gates.py`, gate OR preflight) or if the lock file cannot be
created. ADDITIONALLY (this task's own precondition, CLAUDE.md "attached,
alone" rule): scans the live process table (`ps aux`) for any OTHER process
whose command line names this repository's campaign harness or any s39
script, refusing if one is found (excluding this process's own pid) --
belt-and-suspenders with the lock file, which only detects a HELD lock, not
a harness process that crashed without releasing it or one launched through
a different working directory.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_preflight.py C \\
        > data/SRP1/Results/P515S39/preflight_C_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_preflight.py D \\
        > data/SRP1/Results/P515S39/preflight_D_launch.log 2>&1

Run ONE AT A TIME, attached (no `screen`/`nohup`/`&`), stderr captured via the
shell redirection above (this script does not redirect its own stderr).

Verifies:
  1. Bounded solve-profile identity (`51*(2+1)` + recovery-retry overhead),
     exactly the s37/s38 preflights' own method.
  2. tau/gamma IN FORCE per cycle, read from each cycle's OWN `gamma_
     {v,pf,ess}_after` diagnostic field: 0 on EVERY channel, EVERY cycle,
     in BOTH arms (tau=0.0 for both C and D, unlike s38's arm-dependent tau).
  3. Action labels per cycle: V and PF -- LIVE labels ('increased'/
     'decreased'/'held') in BOTH arms (unlike s38 arm A, which also exempts
     PF); ESS -- 'exempt (fixed)' in BOTH arms (C: unconditional; D:
     conditional, `balancing_exempt_until`) -- D's own `exempt_until_state`
     sidecar is read back and asserted NOT lifted (streak < 5, `lifted`
     False) at cycle 2, since 2 cycles cannot complete a 5-consecutive-cycle
     streak by construction.
  4. `minimum_consecutive_converged_cycles == 10` in force every cycle
     (`required_consecutive_cycles` trajectory field).
  5. The PF capture identity (`pf_entry_stride_<arm>.jsonl`) holds every
     cycle -- RE-VERIFIED here by reading the sidecar back.
  6. Every capture path populated (recourse-jump, ess-entry-stride, SoH-
     floor, PF-entry-stride, ess-exempt-until-state sidecars; boyd_terminal.
     json; g_<arm>.json; interface voltage/settlement terminal; component
     levels; ESSO models pickle).
  7. The evaluator (`p515_s39_evaluate.py`) runs on THIS preflight's own
     output in DRY mode (zero solves, no write).
  8. BITWISE comparison of cycles 1-2 against v9 arm A's OWN committed
     cycles 1-2 (`data/SRP1/Results/P515S38_A_TAU0_run/g_s39_A_tau0.json`
     -- read-only, never re-run, never staged): every field present in
     BOTH rows, EXCLUDING only the fields that differ BY TEXTUAL
     CONSTRUCTION alone (the PF action label and the `balancing_exempt_pf`/
     `balancing_exempt_ess` flags -- A exempts PF and uses the static
     mechanism for ESS; C/D do not exempt PF and, for D, use the NEW
     conditional mechanism for ESS, which reports the SAME 'exempt (fixed)'
     action text but a DIFFERENT underlying `balancing_exempt_ess` flag --
     and `required_consecutive_cycles`, 3 for A vs 10 for C/D), must be
     EQUAL (`==`, not `isclose` -- this is a bitwise claim). A mismatch
     outside that exclusion list is reported by field name and FAILS this
     preflight -- never rationalized away.

Reports cycle-1/2 rho/gamma/action per channel, wall time.
"""
import glob
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s39_evaluate as E39  # noqa: E402

ARM_KEY_BY_LABEL = {'C': 's39_C', 'D': 's39_D'}
OUT_DIR_BY_LABEL = {
    'C': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39', 'preflight_C'),
    'D': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39', 'preflight_D'),
}
NUM_CYCLES = 2
# 51 solves per cycle + 51 for initialization -- the SAME identity every
# other arm's own `run_admm_arm` reports (`51 * len(rows) + 51`); a
# recovered network failure adds EXTRA solve attempts on top (tier-1 retry
# = +1 solve, tier-2 retry = +2 solves per recovered event).
BASE_PERMITTED_SOLVES = 51 * (NUM_CYCLES + 1)

# v9 arm A's own committed 300-cycle run -- READ-ONLY reference for the
# BITWISE comparison (item 8). Never re-run.
V9_ARM_A_TRAJECTORY_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S38_A_TAU0_run', 'g_s38_A_tau0.json')

# Fields excluded from the BITWISE comparison because they differ BY
# TEXTUAL/MECHANISM CONSTRUCTION alone (see module docstring item 8), never
# because of a numerical/physics divergence this preflight is meant to
# catch.
BITWISE_EXCLUDE_FIELDS = {
    'rho_pf_action',              # A: 'exempt (fixed)' (PF also exempt in A); C/D: live label
    'balancing_exempt_pf',        # A: True; C/D: False
    'balancing_exempt_ess',       # A/C: True (static mechanism); D: False (conditional mechanism)
    'required_consecutive_cycles',  # A: 3; C/D: 10 (spec v10 override)
}

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = (
    'p515_g_g1_g4_admm_gates.py',
    'p515_s39_preflight.py',
    'p515_s39_evaluate.py',
    'p515_s39_zero_solve_checks.py',
    'p515_s39_d_policy_replay.py',
)


def _ancestor_pids(max_depth=15):
    """This process's own pid, plus every ancestor up the parent chain
    (its shell, that shell's parent, ...), bounded to `max_depth` hops.
    Needed because the launch shell that starts THIS script's own process
    necessarily has the full command line -- including this script's own
    filename -- in ITS argv too (e.g. `zsh -c 'eval "python p515_s39_
    preflight.py C > ... 2>&1"'`); excluding only `os.getpid()` would flag
    that shell as a 'forbidden process' on every legitimate launch. A
    GENUINELY separate, concurrent process is never an ancestor of this
    one."""
    pids = {os.getpid()}
    current = os.getpid()
    for _ in range(max_depth):
        try:
            ppid_text = subprocess.run(
                ['ps', '-o', 'ppid=', '-p', str(current)], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            break
        if not ppid_text:
            break
        ppid = int(ppid_text)
        if ppid <= 1 or ppid in pids:
            break
        pids.add(ppid)
        current = ppid
    return pids


def _check_preconditions(label, out_dir_override=None):
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in _ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue  # this process itself, or one of its own launching shells
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    out_dir = out_dir_override or OUT_DIR_BY_LABEL[label]
    if os.path.exists(out_dir):
        failures.append(f'preflight output directory already exists (write-once): {out_dir}')

    return failures


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ARM_KEY_BY_LABEL:
        print(__doc__)
        sys.exit(1)
    label = sys.argv[1]
    arm_key = ARM_KEY_BY_LABEL[label]
    # Optional second argument: a suffix (e.g. 'v2') that re-runs this
    # preflight into its OWN output dir under its OWN working-dir ids,
    # leaving an earlier preflight's committed evidence untouched. P5.15
    # Addendum 21: the first C/D preflights ran BEFORE the code was
    # committed, so the Planner re-ran them at the committed HEAD with
    # suffix 'v2' (never re-running a harness onto a cited artifact).
    suffix = sys.argv[2].strip('_') if len(sys.argv) > 2 and sys.argv[2].strip('_') else None
    out_dir = OUT_DIR_BY_LABEL[label] + (f'_{suffix}' if suffix else '')
    mode_label = f'preflight_{suffix}' if suffix else None

    precondition_failures = _check_preconditions(label, out_dir)
    if precondition_failures:
        print(f'[S39 preflight {label}] PRECONDITION CHECK FAILED: {precondition_failures}')
        sys.exit(1)
    print(f'[S39 preflight {label}] preconditions passed (no lock, no forbidden process, fresh output dir)')

    # Acquire the harness's OWN exclusive run lock -- reuses the harness's
    # OWN mechanism, never reimplemented.
    G._acquire_exclusive_run_lock()

    report, report_path = G.run_s39_arm(
        arm_key, num_max_iters_override=NUM_CYCLES, output_root_override=out_dir,
        mode_label_override=mode_label)

    print(f'[S39 preflight {label}] arm report: {report_path}')
    print(f"[S39 preflight {label}] cycles_run={report['cycles_run']} "
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
        f'[S39 preflight {label}] solve-profile identity: base={BASE_PERMITTED_SOLVES} '
        f'(51*({NUM_CYCLES}+1)) + retry_overhead={retry_overhead} '
        f'(tier1={n_tier1}*1 + tier2={n_tier2}*2) = expected {expected_permitted_solves}; '
        f'observed permitted_solve={observed_permitted_solves}; exact={solve_identity_exact}'
    )
    if not solve_identity_exact:
        raise RuntimeError(
            f'S39 preflight {label} solve-profile identity FAILED: expected '
            f'{expected_permitted_solves} (base {BASE_PERMITTED_SOLVES} + retry overhead '
            f'{retry_overhead}), observed {observed_permitted_solves}')

    # ---- (2)/(3)/(4) tau/gamma=0, live V/PF labels, exempt ESS, min
    #      consecutive == 10, per cycle ---------------------------------------
    exempt_channels = set(G.S39_ARMS[arm_key]['exempt_channels'])
    exempt_until = G.S39_ARMS[arm_key]['exempt_until']
    per_cycle_channel_check = []
    all_gamma_zero, all_labels_ok, all_min_consecutive_ok = True, True, True
    for r in rows:
        entry = {'cycle': r['cycle']}
        min_consecutive_ok = (r.get('required_consecutive_cycles') == G.S39_REQUIRED_CONSECUTIVE_CYCLES)
        all_min_consecutive_ok = all_min_consecutive_ok and min_consecutive_ok
        for ch in ('v', 'pf', 'ess'):
            rho_after = r.get(f'rho_{ch}_after')
            gamma_after = r.get(f'gamma_{ch}_after')
            action = r.get(f'rho_{ch}_action')
            gamma_ok = (gamma_after == 0.0)
            if ch in ('v', 'pf'):
                label_ok = action in ('increased', 'decreased', 'held')
            else:  # ess -- exempt in BOTH arms (static for C, conditional for D)
                label_ok = (action == 'exempt (fixed)')
            all_gamma_zero = all_gamma_zero and gamma_ok
            all_labels_ok = all_labels_ok and label_ok
            entry[ch] = {'rho_after': rho_after, 'gamma_after': gamma_after, 'gamma_ok': gamma_ok,
                        'action': action, 'label_ok': label_ok}
        entry['required_consecutive_cycles'] = r.get('required_consecutive_cycles')
        entry['min_consecutive_ok'] = min_consecutive_ok
        per_cycle_channel_check.append(entry)
    print(f'[S39 preflight {label}] exempt_channels={sorted(exempt_channels)}, '
          f'exempt_until={exempt_until}, per-cycle channel check: '
          f'{json.dumps(per_cycle_channel_check, default=str)}')

    # -- D only: exempt_until_state sidecar asserts NOT lifted at cycle 2 --
    exempt_until_not_lifted_ok = True
    exempt_until_sidecar_check = None
    if arm_key == 's39_D':
        hits = glob.glob(os.path.join(out_dir, 'ess_exempt_until_state_*.jsonl'))
        if not hits:
            raise RuntimeError(f'S39 preflight D: no ess_exempt_until_state sidecar found under {out_dir}')
        lines = [json.loads(line) for line in open(hits[0]) if line.strip()]
        last_line = lines[-1] if lines else {}
        ess_state = (last_line.get('channels') or {}).get('ess', {})
        exempt_until_not_lifted_ok = bool(
            len(lines) == NUM_CYCLES and ess_state.get('lifted') is False
            and (ess_state.get('streak') or 0) < 5)
        exempt_until_sidecar_check = {
            'path': os.path.relpath(hits[0], REPO), 'n_lines': len(lines),
            'cycle_2_ess_state': ess_state, 'not_lifted_ok': exempt_until_not_lifted_ok,
        }
        print(f'[S39 preflight D] exempt_until sidecar check: {exempt_until_sidecar_check}')

    # ---- (5) PF capture identity holds every cycle -------------------------
    pf_hits = glob.glob(os.path.join(out_dir, 'pf_entry_stride_*.jsonl'))
    if not pf_hits:
        raise RuntimeError(f'S39 preflight {label}: no pf_entry_stride sidecar found under {out_dir}')
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
    print(f'[S39 preflight {label}] PF capture identity per cycle: {pf_identity_per_cycle}')

    # ---- (6) every capture path populated ----------------------------------
    settlement_hits = glob.glob(os.path.join(out_dir, 'interface_settlement_detail_*.json'))
    exempt_until_state_hits = glob.glob(os.path.join(out_dir, 'ess_exempt_until_state_*.jsonl'))
    capture_paths = {
        'recourse_jump_sidecar': report.get('s34_recourse_jump_sidecar_path'),
        'ess_entry_stride_sidecar': report.get('s34_ess_entry_stride_sidecar_path'),
        'soh_floor_sidecar': report.get('s35ref_soh_floor_sidecar_path'),
        'pf_entry_stride_sidecar': report.get('s38_pf_entry_stride_sidecar_path'),
        'ess_exempt_until_state_sidecar': report.get('s39_ess_exempt_until_state_sidecar_path'),
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
    print(f'[S39 preflight {label}] capture paths populated: {capture_paths_populated}')

    # ---- (7) the evaluator runs on the preflight output in DRY mode -------
    original_root = E39.ARM_ROOTS[arm_key]
    E39.ARM_ROOTS[arm_key] = out_dir
    try:
        eval_rc = E39.main([out_dir, '--dry-run'])
    finally:
        E39.ARM_ROOTS[arm_key] = original_root
    print(f'[S39 preflight {label}] evaluator dry-run return code: {eval_rc}')

    # ---- (8) BITWISE comparison of cycles 1-2 against v9 arm A ------------
    if not os.path.exists(V9_ARM_A_TRAJECTORY_PATH):
        raise RuntimeError(f'v9 arm A reference trajectory not found: {V9_ARM_A_TRAJECTORY_PATH}')
    with open(V9_ARM_A_TRAJECTORY_PATH) as handle:
        arm_a_rows = json.load(handle)['cycle_trajectory']
    arm_a_by_cycle = {r['cycle']: r for r in arm_a_rows[:2]}

    bitwise_report = {}
    bitwise_all_match = True
    for r in rows:
        cycle = r['cycle']
        a_row = arm_a_by_cycle.get(cycle)
        if a_row is None:
            bitwise_report[cycle] = {'error': f'v9 arm A trajectory has no cycle {cycle}'}
            bitwise_all_match = False
            continue
        common_keys = (set(r) & set(a_row)) - BITWISE_EXCLUDE_FIELDS
        mismatches = {}
        for key in sorted(common_keys):
            v_this, v_a = r[key], a_row[key]
            if v_this != v_a:
                mismatches[key] = {'this_arm': v_this, 'v9_arm_A': v_a}
        cycle_match = (len(mismatches) == 0)
        bitwise_all_match = bitwise_all_match and cycle_match
        bitwise_report[cycle] = {
            'n_fields_compared': len(common_keys), 'n_mismatches': len(mismatches),
            'match': cycle_match, 'mismatches': mismatches,
        }
    print(f'[S39 preflight {label}] BITWISE vs v9 arm A (cycles 1-2): '
          f'{json.dumps(bitwise_report, default=str)}')
    if not bitwise_all_match:
        print(f'[S39 preflight {label}] *** BITWISE COMPARISON FAILED *** -- see mismatches above. '
              f'Reported as-is; NOT rationalized.')

    payload = {
        'stage': f'P5.15 Addendum 21 (s39 preparation worker task) -- {arm_key} two-cycle preflight',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 21',
                      'data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json'],
        'label': label, 'arm': arm_key, 'num_cycles': NUM_CYCLES,
        'out_dir': os.path.relpath(out_dir, REPO),
        'solve_profile_identity': {
            'base_permitted_solves': BASE_PERMITTED_SOLVES, 'n_tier1_recovered': n_tier1,
            'n_tier2_recovered': n_tier2, 'retry_overhead': retry_overhead,
            'expected_permitted_solves': expected_permitted_solves,
            'observed_permitted_solves': observed_permitted_solves,
            'exact': solve_identity_exact,
        },
        'exempt_channels': sorted(exempt_channels), 'exempt_until': exempt_until,
        'per_cycle_channel_check': per_cycle_channel_check,
        'all_gamma_zero': all_gamma_zero, 'all_action_labels_correct': all_labels_ok,
        'all_min_consecutive_converged_cycles_10': all_min_consecutive_ok,
        'exempt_until_sidecar_check': exempt_until_sidecar_check,
        'exempt_until_not_lifted_ok': exempt_until_not_lifted_ok,
        'pf_capture_identity_per_cycle': pf_identity_per_cycle,
        'pf_capture_identity_all_hold': pf_identity_all_hold,
        'pf_entry_stride_path': os.path.relpath(pf_stride_path, REPO),
        'capture_paths_populated': capture_paths_populated,
        'all_capture_paths_populated': all_capture_paths_populated,
        'evaluator_dry_run_return_code': eval_rc,
        'bitwise_vs_v9_arm_a': {
            'reference_path': os.path.relpath(V9_ARM_A_TRAJECTORY_PATH, REPO),
            'excluded_fields': sorted(BITWISE_EXCLUDE_FIELDS),
            'per_cycle': bitwise_report, 'all_match': bitwise_all_match,
        },
        'network_failures_summary': report['network_failures_summary'],
        'local_solve_failures': report['local_solve_failures'],
        'wall_clock_s': report['wall_clock_s'],
    }
    payload_path = os.path.join(out_dir, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S39 preflight {label}] wrote {payload_path}')

    overall_ok = (
        solve_identity_exact and all_gamma_zero and all_labels_ok and all_min_consecutive_ok
        and exempt_until_not_lifted_ok and pf_identity_all_hold and all_capture_paths_populated
        and eval_rc == 0 and bitwise_all_match
    )
    print(f'[S39 preflight {label}] overall_ok={overall_ok}')
    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
