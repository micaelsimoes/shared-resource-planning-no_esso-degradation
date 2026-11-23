# P5.15 S56: step 3 v3 and the step 4 smoke (2026-11-23)

**Authority:** `S56_PLAN_v3.md` §7 rows 3–4; `S56_STEP4_DESIGN.md` decision 9.2 (a serial 3-cycle smoke per arm); TASKS.md
S56 section at `9c88d89d`; the Planner's task of 2026-11-23 (item 1: step 3 v3; item 2: the step 4 smoke). The step 4
**main run was not launched**; it stays held for the Expert's X4-1 and X4-2.

**Objective convention (every table).** Q is € of total discounted cost. Coupled arms (V-1, V-2): production's primary
`gross_operational_cost` = settlement excluded in full (pin excluded, salvage separate); the contracted-excluded form
is beside it where stated. Pinned arms (option off): `gross_operational_cost` = contracted settlement and pin excluded.
C_central is step 3's gross €/day (settlement weight 0, no row 18, no pin, no salvage). Q_block − w_b·C_central is €
of total discounted cost, with w_b the instance's own block weight (equal to the 12-block weight, G2).

## Decisions needed

1. **2025|Summer still has no C_central.** Step 3 v3's warm start fixed single 22 but **not single 12**: warm from the
   saved merged-default slice it again stopped at the iteration limit (500), at **the same μ as the cold failure,
   1.8449144625279508e-06** (bitwise-equal `mu_final`), with unscaled dual infeasibility 9.8e5 and constraint violation
   2.6e-4. The warm point itself was right: the single's objective at the mapped point equals the merged scenario-12
   term exactly (187,383.79270576665).
   - **(a)** Use the merged-default per-scenario term for 12 (measured bias of the converged pairs: +0.0345 to +0.0354
     €/day each; single 22 now +0.035297). Zero solves. C_central would then carry one merged term, flagged.
   - **(b)** A spec v4 with a different start or option set for single 12 alone (for example warm from the S3-2
     merged-warm solution, or a solver-option change). One solve, but it changes the comparator's solver settings for
     one scenario, which I would not do without your ruling.
   - **(c) Recommended.** Leave 2025|Summer without a C_central. It is the passenger block of instance B; the two
     calibration blocks (2025|Winter, 2035|Summer) are complete. D-3 for 2025|Summer was run on the 8 available
     singles and, supplementary, on the merged-default solution (below): no value lies outside a box.
2. **Pinned AA-on and AA-off are bitwise identical for all 3 cycles, on both instances.** AA reported "insufficient
   memory (m_k=0)" in every cycle of both AA-on runs, so the smoke measured nothing about AA. This is not a failure
   (no prediction covered it). It matters for the main run only if the AA-on arm is kept "if memory allows": its
   difference from AA-off can appear only after cycle 3.

## Blocked on the author

None.

## Changed

| commit | content |
|---|---|
| `a1113aac` | `p515_s56_step3_v3.py`, spec v3 `frozen_step3_spec_v3_b8d07450.json` (predecessor `ff466289`), zero-solve `trial_v1` — before launch |
| `c7720654` | step 3 v3 `run_v1` evidence (results, manifest, IPOPT logs, both primal files, stdout/stderr) |
| `4856bd46` | `p515_s56_step4_smoke.py`, smoke spec v1 `frozen_step4_smoke_spec_v1_2e60b2ca.json` (predecessor `ff466289`, cites `S56_STEP4_DESIGN.md` 9.2 and spec v3) — before launch |
| `bb8639cd` | smoke spec **v2** `frozen_step4_smoke_spec_v2_e36c29bd.json` (predecessor v1 `2e60b2ca`), the launcher fix, and the preserved v1 attempt (below) |
| this commit | 8 smoke runs' evidence (205 files), D-3, evaluation, `step4_smoke_v1_manifest_sha256.json` (605 files hashed, 220 tracked), this report |

- **No production file was edited.** No harness file was edited.
- **Smoke spec v1 → v2.** v1's run 1 (A, V-1) was refused **before any solve** by `run_admm_arm`'s rule-eleven
  assertion `assert_g_capture_paths` ("results_dir not absolute/set"): my launcher passed a relative campaign root.
  The v1 root `campaign_s56s4smoke_A_V1/` is kept and committed; its record has `status: error` and no solve profile
  (the run stopped before `run_operational_planning`). v2 changes only the root naming (absolute,
  `campaign_s56s4smoke_v2_<inst>_<arm>`); arms, configuration, cap, predictions and thresholds are v1's.
- **Configuration (every run):** the G-f / step 4 build configuration (`p515_s56_coupling_gates.py` literals): case-file
  AA declared (`keep_memory`, enabled), convergence-depth tail declared (1e-6), ESS ageing baseline declared; derived
  instance A `s56_step4_2025_Winter` (`92e6283b…`) or B `s56_step4_2025_2035_Summer` (`b1cd5440…`); candidate x0
  (investment year 2025); α = 0 (no premium declared; production default `alpha 0.0`, so row 18 is not built);
  cap 3; required consecutive cycles 10; concurrency 1; thread caps 1.
- **Arms:**
  - V-1: `interface_coupling` = {on, `'per_scenario'`, `row18_charge_reactive` False} (the G-f declaration);
  - V-2: {on, `'pin'`, `row18_charge_reactive` False}. The Q-charge flag is inert at α = 0 (row 18 is built only when
    α > 0); I set it equal to V-1's so the two arms differ only in the voltage mode;
  - pinned AA-off: option not declared; `overrides` {`anderson_acceleration`: {`enabled`: False}};
  - pinned AA-on: option not declared; case default.
- **Initialisation identity (S-P3).** Each run has a new root, so before run k the launcher copied, byte for byte, the
  earlier runs' `init_identity/*.json` of the same instance into the new root (the harness's documented "reference
  record the launcher placed there"). The harness then made the comparison itself, and would have raised before
  cycle 1 on a mismatch.

## Found

### Item 1: step 3 v3 (spec `b8d07450`, 2 solves, guards exact 2/2 on both guards)

| prediction (verbatim) | outcome |
|---|---|
| both solves Optimal | **failed**: single 22 Optimal (241 iterations, μ at floor 9.09e-11); **single 12 maxIter** (500 iterations, μ 1.8449e-6, above floor) |
| each objective = the merged per-scenario term minus an offset in [0.030, 0.040] € | **failed** (12 not loaded). Single 22: 194,669.14632169058 − 194,669.11102497295 = **0.035297 €** (inside the range) |
| P3/P4 hold (1e-6 / 1e-5) | **not evaluable**: C_central undefined |

- **Warm start (both singles):** 30,936 Vars set, 4,800 fixed skipped, 2,448 None, 0 unmapped (the S3-1 pattern).
  The merged model rebuilt from the saved primal file reproduced the saved merged-default objective **bitwise**
  (185,674.1557530493); 278,424 Vars set, 0 names missing.
- **2025|Summer C_central: undefined.** Singles available: 00, 01, 02, 10, 11, 20, 21 (v2 run) and 22 (v3).
- Wall 43 s; peak RSS 0.88 GB (parent), IPOPT 0.15 GB.

### Item 2: the smoke (spec `e36c29bd`, 8 runs, serial)

| prediction (verbatim) | outcome |
|---|---|
| **S-P1:** every run exits 0 with 3 cycles; the solve-count identity holds. | **held**: 8/8 exit 0 (parent view and `exit_code.txt`), 3 cycles, 3 per-cycle lines; observed solves = declared base on every run (A 28, B 44; 288 in all), 0 retries, 0 blocked; launcher parent guard 0 |
| **S-P2:** F1 arms, every cycle: price part + `interface_settlement_total` = 0 (rel 1e-9); contracted − primary = deviation total (rel 1e-9); Σ_s ω_s μ_s = 0 per coordinate (abs 1e-9 × max\|μ\|); V-1 voltage pin total ≡ 0. | **held** on all 4 coupled runs, every cycle. Worst: (i) 1.6e-13; (ii) 9.3e-11 (A V-1, cycle 1: 477.98551 € vs 477.98551 €, a difference of two ~2e8 sums); gauge 1.5e-12 × max\|μ\|; pin total, TSO and every DSO 0.0 (terminal record; see Not confirmed) |
| **S-P3:** initialisation, per instance: pinned AA-off ≡ pinned AA-on ≡ V-2, bitwise (the harness grouping check passes); V-1 is in its own group and differs. | **held** on A and B. A: V-2 = AA-off = AA-on = `0x1.bbb9febecf6ffp+13`, V-1 `-0x1.fa7e3723c4b36p+11`. B: `0x1.d2ad25c1d62ffp+15` ×3, V-1 `0x1.7c7dbcc7a0caep+13`. AA-off compared with V-2, AA-on with V-2 and AA-off, all equal; V-1 listed and not compared |
| **S-P4:** on real cycles, coupled arms: `assert_s56_cycle_capture_path` holds; per-block Q from the capture sums to production's per-cycle `gross_operational_cost` within rel 1e-12. | **held**: checklist all True on 4/4 runs; capture summary ok (3 lines); worst rel 2.5e-16 |
| **S-P5:** peak RSS ≤ 6 GiB per run; s/cycle recorded (Advisor: 45–75 s for two blocks). | **held**: A 2.08–2.29 GiB, B 3.23–3.39 GiB (harness child); IPOPT ≤ 0.347 GiB. Two-block means 46.3–52.4 s/cycle |

**S-P2 operational detail** (frozen in the spec before launch):
- (i) T_c is the sum of w × `settlement_total_unweighted` over every agent block of the S56 capture — these are the
  terms production's `_get_operational_interface_settlement_blocks` adds. At cycle 3 it was also checked against the
  record's production `interface_settlement_total` (worst 4.4e-14).
- (ii) D_c is the sum of w × (total − contracted) from the capture. Production records no per-cycle deviation total, so
  `total − contracted == deviation` was checked:
  - per cycle on the DSO blocks, against `covariance_dso_weighted`: equal to rel 0 on every cycle;
  - at cycle 3 on every block, against the record's `interface_settlement_deviation_total`: worst 1.7e-12.
- (iii) ω_s are read from the run's own `multiscenario_terminal` (1/9 each). Every coordinate passed. The worst ratio,
  1.5e-12, was on the Q deviation duals.

### Per-run table

s/cycle is `cycle_wall_s` per cycle. Q_b − w_b·C_central is in € of total discounted cost, settlement excluded in full
for V-1 / V-2 and contracted-excluded for the pinned arms, per cycle 1 / 2 / 3.

| run | instance, arm | exit | solves declared / observed | s/cycle | peak RSS (GiB) | block | Q_b − w_b·C_central (€) |
|---|---|---|---|---|---|---|---|
| 1 | A, V-1 | 0 | 28 / 28 | 27.7, 22.2, 18.1 | 2.08 | 2025\|Winter | +131,628,644.95 / +24,595,340.41 / **+7,853,707.99** |
| 2 | A, V-2 | 0 | 28 / 28 | 34.9, 37.2, 18.6 | 2.29 | 2025\|Winter | +129,994,973.17 / +37,973,781.92 / **+7,405,913.21** |
| 3 | A, pinned AA-off | 0 | 28 / 28 | 40.1, 37.4, 18.5 | 2.18 | 2025\|Winter | +129,363,884.93 / +37,741,032.20 / **+6,827,041.41** |
| 4 | A, pinned AA-on | 0 | 28 / 28 | 41.6, 40.0, 20.5 | 2.13 | 2025\|Winter | identical to run 3 (bitwise) |
| 5 | B, V-1 | 0 | 44 / 44 | 63.2, 39.2, 36.5 | 3.23 | 2035\|Summer | +168,936,051.48 / +38,498,420.12 / **+4,380,569.75** |
| 6 | B, V-2 | 0 | 44 / 44 | 53.9, 56.0, 37.1 | 3.35 | 2035\|Summer | +176,104,048.83 / +51,100,103.10 / **+12,709,367.07** |
| 7 | B, pinned AA-off | 0 | 44 / 44 | 70.4, 50.4, 35.2 | 3.39 | 2035\|Summer | not captured / not captured / **+11,913,018.40** |
| 8 | B, pinned AA-on | 0 | 44 / 44 | 70.3, 50.3, 36.7 | 3.39 | 2035\|Summer | identical to run 7 (bitwise) |

- **w_b·C_central:**
  - 2025|Winter: 455.0 × 175,188.678750 = 79,710,848.83 €;
  - 2035|Summer: 373.2585 × 153,017.214164 = 57,114,972.23 €.

  Capture weights equal the instance's own weights on every coupled run. Per-block allowances: 228.64 € and 187.56 €.
- **Contracted-excluded form, coupled arms, cycle 3:**
  - A V-1 +7,845,806.42 €; A V-2 +7,392,929.33 €;
  - B V-1 (2035|Summer) +4,287,628.10 €; B V-2 +12,728,985.16 €.
- **2025|Summer (no C_central), Q_b, settlement excluded in full:**
  - V-1: 245,187,065.29 / 121,485,013.16 / 90,075,991.08;
  - V-2: 245,752,325.32 / 133,722,174.16 / 98,308,554.20;
  - pinned (contracted-excluded), terminal only: 97,857,907.71.
- **Pinned instance B:** per-block Q exists only at the terminal cycle, from `multiscenario_terminal`. Option-off runs
  carry no per-block per-cycle capture. The two terminal blocks sum to production's cycle-3 Q exactly (rel 0).
- **Whole-run wall** (launcher view): A 152 / 172 / 182 / 193 s; B 287 / 297 / 302 / 302 s.
- **Local solves:** 0 local solve failures and 0 network-failure events on every run. V-1 and V-2 had AA forced off by
  production in every cycle.
- **What the cycle-3 numbers are.** They are 8 % to 22 % of w·C_central: 3-cycle iterates, far from settled. They are
  recorded only, as the task asked. They carry no settling bar.

### D-3 for 2025|Summer (zero solves; `d3_2025_Summer_20261123T135053.json`)

**Method** (frozen in spec v2 before launch):
- The ADMM-prepared models of instance B under V-1 were rebuilt with the diagnostics launcher's own initialisation
  round in its trial form: every optimize stubbed, σ fixed. The run stopped at `_build_pristine_snapshot_bases`.
- The only solution-dependent box input, the TSO ADN-load pc / qc, was set from the V-1 instance-B smoke run's own
  value. Its source is `interface_settlement_detail_s31c.json` `anchor_p_mw` / `anchor_q_mvar` (production
  `_get_interface_reporting_detail`). In the distributed path production fixes pc once, at TSO model creation, at the
  DSO initial expectation, and never changes it. The anchors were scenario-common, as expected.
- The boxes were computed with the diagnostics launcher's `d3_boxes`, by import.
- The tolerance is the same 1e-8·max(1,|b|) as in `ff466289`. **Disclosure:** that tolerance was chosen after a
  zero-solve draft trial had shown strict margins of −1.9e-10 pu; the Planner ruled it acceptable.

**Method gate (recorded).** The same construction applied to 2035|Summer, with run_v1's central values, reproduces
run_v1 child B's D-3 result **exactly**:
- n_values 3,888;
- outside 0 and strict 38 on both;
- every family's minimum margin is identical (difference 0.0).

The smoke's V-1 initialisation therefore equals the diagnostics run's. The scenario-filtered copy of `d3_boxes` used
for the 8-scenario case equals `d3_boxes` on both complete inputs.

| 2025\|Summer central values | values | outside (tol) | strict | TSO P δ-box | TSO Q δ-box | DSO ref-gen P | DSO ref-gen Q | V (TSO = DSO) |
|---|---|---|---|---|---|---|---|---|
| **primary: 8 Optimal singles (12 missing; incomplete)** | 3,456 | **0** | 32 (16 TSO V + 16 DSO V) | 42.33 MW (DN7 s02 t14) | 98.65 MVAr | **0.490 MW** (DN7 s02 t13: 99.511 vs limit 100.001) | 86.51 MVAr | −2.03e-10 pu (strict; inside tol) |
| supplementary: S3-2 merged-default solution (9 scenarios) | 3,888 | 0 | 0 | 42.33 MW | 98.65 MVAr | 0.490 MW | 86.51 MVAr | +1.98e-8 pu |

The table gives minimum margins, min(value − lower, upper − value). The guard was 0 on both the launcher and the
imported module, pickle 0/0. The step took 17.6 s and 0.95 GiB.

## Unexpected findings

- **Single 12's stall is start-independent at the recorded precision.** Cold (v2) and warm (v3) both end at μ =
  1.8449144625279508e-06. The same μ also appeared in run_v1's 2025|Winter single 02, which a warm start did fix. Not
  investigated.
- **AA did nothing in 3 cycles** ("insufficient memory (m_k=0)" every cycle), so pinned AA-on ≡ AA-off bitwise. See
  decision 2.
- **The init-identity gross values are small:** −4,051.94 € (A, V-1), 14,199.25 € (A, pin group), 12,175.72 € (B, V-1)
  and 59,734.57 € (B, pin group), against per-cycle Q of about 1e8. G-f r3's was −4,169.41 €. This is the record's own definition (production
  `_get_operational_recourse_components` on the initialisation models before activation). It does not affect the
  bitwise check. Not investigated.
- **The DSO7 reference-generator P limit** is within 0.49 MW of the 2025|Summer central exchange (s02 t13). The same
  feature appeared in 2035|Summer (0.56 MW).
- **The launcher fault behind spec v2** (relative campaign root). It was caught at zero solves by production's
  rule-eleven assertion.
- **The tool moved long commands into background tracking.** The Bash tool moved commands running longer than 180 s
  into tracked background mode. Runs 3–8 were therefore waited on through its completion notifications and exit-line
  polling.
  - No run was detached (no nohup or screen).
  - Each run was launched only after the previous one had exited.
  - Both streams went to files.

## Not confirmed

- **The V-1 pin at cycles 1–2.** Production records no per-cycle pin total. The terminal record gives 0.0, and the
  structural reading is that `_get_local_voltage_pin` returns 0.0 because `scenario_voltage_pin` is absent under
  `'per_scenario'`. I searched `per_cycle_record.jsonl`, `g_s39_D.json` (the trajectory) and `s56_cycle_capture.jsonl`;
  none holds a per-cycle pin.
- **The TSO deviation part per cycle.** Production's `part='deviation'` value for the TSO is not recorded per cycle. It
  was checked only at cycle 3, through the record's total. The DSO part was checked every cycle.
- **2025|Summer C_central, its P3 / P4, and a complete D-3 for it.** I searched `step3_v3/run_v1/step3_v3_results.json`:
  single 12 was not loaded.
- **Pinned instance-B per-block Q at cycles 1–2.** I searched `per_cycle_record.jsonl`, `per_cycle_response.jsonl`,
  `multiscenario_terminal.json` and `component_levels_terminal.json`. Only the terminal values exist.
- **That the smoke's first 3 cycles will match a concurrent main run bitwise.** That is the main run's own gate, and
  the main run was not launched.
- **Anything about convergence, X4-1 or X4-2 from these 3 cycles.** The values are far from settled.
