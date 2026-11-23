# P5.15 S56 step 4 preparation: step 3 v2, step 4 build, pre-run diagnostics (2026-11-23)

**Authority:** `S56_PLAN_v3.md` §7, rows 3–4; TASKS.md, S56 section; `S56_STEP4_DESIGN.md`; the Planner's task of
2026-11-23 (items 1–4).

**Outcome in one line:** items 2a and 2b are done and committed. Items 1 and 3 were **not run**, because two premises
of the task do not hold. No solve was run in this task.

## Decisions needed

1. **The step 3 merged solution was never preserved, so item 1 and D-2 cannot be warm-started from it.**
   - **Where I looked:**
     - `run_v1/central_ms_x0_results.json` holds the costs, the curtailment and the per-scenario interface P, Q and V.
       It holds no Var values.
     - `run_v1/logs/` holds the IPOPT logs only.
     - `p515_s56_central_ms_x0.py` writes no Var values, no `.sol` file and no pickle.
   - **What this blocks:**
     - Item 1's single solve (warm start from the merged scenario-02 slice).
     - D-2's warm start, on both blocks.
     - D-1 and D-3 are **not** blocked by this: they need only the interface P, Q and V, which are preserved.
   - **Options:**
     - **(a) Recommended.** Spec v2 adds one reproduction solve of the merged model per block. It uses run_v1's exact
       build, default start, recovery off, and takes about 28 s per block. It is gated bitwise against run_v1: the
       objective (175,188.7137093924 and 153,017.2487446982) and every per-scenario interface value. Item 1 and D-2
       then warm-start from it. Cost: +2 solves.
     - **(b)** Run D-2 from the default start, without a warm start. Item 1 stays undone.
     - **(c)** Accept the merged value for 2025|Winter (step 3 report, option c). Its bias is about 0.035 €/day, which
       is 16 € weighted.
2. **The single-block derived instance for 2035|Summer is not the 12-block block (gate G1 failed).**
   - **Cause:** production grows prices and loads from `years[0]`:
     - `_read_market_data_from_file`: (1 + g)^(year − initial_year);
     - `network.py` l. 1420–1425: the same for loads.

     A 2035-only instance therefore has no growth.
   - **Measured:** 252 of 644 arrays are equal. Ratio of 12-block to derived:
     - energy price 1.2801 (1.025^10);
     - flexibility price 1.2190;
     - TN flexibility P 1.2801;
     - DN flexibility P 1.3439;
     - DN qd 1.1323;
     - DN pd 1.13–1.29.
   - **2025|Winter passes every gate,** because its `years[0]` is unchanged.
   - **Options:**
     - **(a) Recommended.** A two-year instance: Years {2025: 5, 2035: 5}, Days {Summer: 91}.
       - Data and weights equal the 12-block instance by construction (`years[0]` = 2025, weight ratio exactly 1).
         This needs the same zero-solve gates, run again.
       - Costs: one passenger block (2025|Summer) in that run, and ρ adaptation and stopping are shared across two
         blocks.
       - The per-(year, day) Q capture (item 2b) isolates the target block.
     - **(b)** One four-block instance, {2025, 2035} × {Winter, Summer}.
     - **(c)** A read-time injection of the 12-block 2035 arrays into a 2035-only read (the EV-injection pattern). This
       is new tooling. G4 also shows that the harness would then need investment year 2035.
     - **(d)** A production option for the growth base year. This is a production change and is not permitted here.
3. **Item 3 (D-1, D-2, D-3) was not run.**
   - **Why:**
     - The task's fallback ("2035|Summer as the only complete block") has no valid instance (decision 2).
     - 2025|Winter has no C_central (decision 1).
     - The item asks for one spec with every solve declared.
   - **Recommended:** after rulings 1 and 2, freeze one D-spec that declares:
     - the reproduction solves (2);
     - item 1's single solve (1);
     - the two initialisation rounds (D-3);
     - D-1: 4 local solves × 2 depths × 2 blocks (16);
     - D-2: 2 hard-pin merged solves.

     D-3 on 2025|Winter alone could run now, on the instance that passes, but it would split that spec.

## Blocked on the author

None.

## Changed

| commit | content |
|---|---|
| `8b7d2c26` | item 2a: `p515_s56_step4_instances.py`, `data/SRP1/Results/P515S56/step4_instances_v1/` (derived cases, gates, 3 zero-solve builds), launch logs, `step4_instances_v1_manifest_sha256.json` |
| `430dfba8` | item 2b: `p515_s44_campaign_harness.py` (two changes, below), `p515_s56_step4_capture_check.py`, `coupling_option_v1/step1_completion_probe_r4/`, `step4_harness_checks_v1/`, `step4_harness_checks_v1_manifest_sha256.json` |
| this commit | this report |

- **Not committed (hash-recorded in the manifests):**
  - the read diagram PDFs;
  - the probe scratch, including the `s56_cycle_capture.jsonl` and interface-dual sidecars;
  - the scratch copy of the pre-change harness.
- **No production file was edited. No spec was frozen:** item 1's spec v2 and the D-spec both wait on decisions 1–3.
  TASKS.md is left to the Planner.
- **Configuration of every build:**
  - option on, `per_scenario_voltage_mode` 'per_scenario', `row18_charge_reactive` False (the G-f declaration);
  - α = 0;
  - candidate x0;
  - p58 rescale as `run_admm_arm` applies it;
  - σ_fixed 93,635,360, assert factor 3.0.

### Item 2b, the harness changes (each conditional; option off unchanged)

1. **Initialisation-identity grouping by pin construction.**
   - `initialisation_identity_group()` puts option off and option on 'pin' in `voltage_pin_constructed`, and option
     on 'per_scenario' in `voltage_pin_not_constructed`.
   - `register_initialisation_identity` compares records only within a group.
   - The group, and the other record's coupling, are written only when the option is set.
2. **Per-cycle S56 capture,** `s56_cycle_capture.jsonl`.
   - **When it runs:** only when the coupling is declared, after `assert_s56_cycle_capture_path()` (fail-fast, before
     any solve). It reads at the `get_admm_boyd_residual_metrics` call, the same point as the interface-dual sidecar.
   - **Each cycle line holds:**
     - the expectation P, Q and V primal copies and the deviation P, Q and V copies, on both sides;
     - the expectation V duals on both sides. Every other interface dual, including the deviation V dual, is already in
       `interface_duals_per_cycle.jsonl`, so the two files together hold every interface dual;
     - ρ_pf and ρ_v per block;
     - `compl_inf_tol` in force per holder (the tail state);
     - Q per (year, day) in both conventions, with the per-agent parts and the salvage block.
   - PR is not computed in the run. It is recomputed afterwards from these files.
   - The summary and checklist go into the evaluation record for coupling entries only.

## Found

### Predictions against outcomes

| prediction | outcome |
|---|---|
| P2v2: single 02 Optimal | **not evaluated.** No solve: the warm-start source does not exist (decision 1) |
| P3v2: single 02 = 171,052.349 € − [0.030, 0.040] € | **not evaluated** (same) |
| D-3: every step 3 interface value inside its ADMM-only box | **not evaluated.** Item 3 not run (decision 3) |
| D1a: production-depth \|offset\| ≥ allowance (H2) | **not evaluated** |
| D1b: tail-depth \|offset\| ≤ allowance / 3 | **not evaluated** |
| D-2: Δ_pin ≥ −(merged stopping slack) | **not evaluated** |

### Per-block table

Values are € of total discounted planning cost unless marked €/day. C_central is gross: the step 3 convention, with the
settlement excluded and no salvage.

| block | w_b (12-block) | w_b (derived) | ratio | allowance (R/4)·w_b/Σ₁₂w | C_central (€/day) | D-1 offset, production / tail | Δ_pin | D-3 min margin |
|---|---|---|---|---|---|---|---|---|
| 2025\|Winter | 455.0 | 455.0 | 1.0 | **228.64** | undefined (single 02 maxIter; item 1 not run) | not run | not run | not run |
| 2035\|Summer | 373.2585 | 455.0 (invalid instance) | 1.2190 | **187.56** | 153,017.214164 | not run | not run | not run |

- Σ₁₂ w = **4,975.0944** (12-block instance, `_get_admm_block_weight`); R = 10,000 €.

### Item 2a gates (zero solves; `step4_instances_v1/step4_instances.json`)

| gate | 2025\|Winter (checksum `58801929…`) | 2035\|Summer (checksum `675ac2d8…`) |
|---|---|---|
| G1: per-block data bitwise vs the 12-block block (644 scenario arrays + probabilities; object walks of TN, 3 DN, every params object, the ESS year; network price arrays) | **holds**: 644/644; walks 0 diffs | **fails**: 252/644 arrays (decision 2); walks 0 diffs; network price arrays differ |
| G2: weight ratio w_b(derived)/w_b(12) | 1.0 | 1.21899 (`years[0]` 2025 → 2035) |
| G3: σ inside the window, read from code: [σ_fixed/f, σ_fixed·f] = [3.1212e7, 2.8091e8] | **inside** (zero-solve proxies 7.38e7–7.97e7) | **indeterminate zero-solve** (proxies 2.80e7–6.96e7 straddle 3.12e7). Settles only at the initialisation round |
| G4: harness and ESSO accept the instance (stubbed run to the ESSO cycle-1 call) | accepted (investment year 2025) | the harness default investment year 2025 is **refused** ("not one of the instance investment years [2035]"); **accepted** with 2035 |
| G5: NL identity of the ADMM-prepared blocks (TSO + 3 DSOs; initialisation and ADMM cycle 1; plain, labelled, .row, .col), up to the recorded scale | **8/8 identical** at scale 1.0 | **0/8** (data). The effective scale was made bitwise equal (σ′ = 114,140,981.354, σ′/σ = 1.21899) |

- **The σ proxies** use step 3's merged default start. The lower proxy is w_b × Σω (TN generator cost); the upper
  proxy is w_b × the merged objective.
- **The real σ** is the largest block-agent objective at the initialisation solution, which was not run.
- **For option (a) of decision 2,** 2035|Summer keeps w_b = 373.26, and its own proxies (2.30e7–5.71e7) also straddle
  the window. σ is then the maximum over both blocks' agents.

### Item 2b zero-solve evidence

- **Option-off identity:** `p515_s56_step1_completion_checks.py` into `step1_completion_probe_r4/`.
  - Pre = harness at `8b7d2c26`; post = harness at `430dfba8`.
  - **4/4 configurations identical,** with 0 differences: `srp1_x0`, `srp1grid_x0_a05`, `srp1_x0_on`,
    `srp1grid_x0_a05_on_v1_noq`. The probe script's sha256 was equal pre and post, and both guards stayed at 0.
- **Grouping** (`step4_harness_checks_v1/grouping/grouping_check.json`): the pre-change harness, loaded from git, and
  the post-change harness were run on synthetic records. **8/8 checks hold:**
  - off-only roots are byte-identical pre and post (equal, unequal, other candidate);
  - in a mixed root, on-'pin' is compared with off and equal;
  - on-'per_scenario' is listed and not compared;
  - an on-'pin' record with a different hex is reported as a mismatch against off and on-'pin';
  - before the change, no cross-declaration comparison was made.
- **Capture** (`step4_harness_checks_v1/capture/capture_check.json`): the probe flow on `srp1grid_x0_a05_on_v1_noq`,
  2 stubbed cycles, with the new hook beside the interface-dual hook.
  - **41/41 checks hold:**
    - every declared field is present with its shape;
    - deviation kinds are p, q, v;
    - the interface-dual sidecar carries the PF and every deviation dual.
  - **Per-block Q sums reconcile with production's per-cycle values:**
    - `gross_operational_cost`, rel ≤ 1.8e-16;
    - `gross_operational_cost_contracted_excluded`, rel 0.0.
  - **Size:** 1.09 MB per cycle line on 12 blocks, so about 0.09 MB per cycle on a one-block instance. This is
    separate from the interface-dual sidecar.

### Solve counts, wall time and RSS

| process | solves (guard) | wall | peak RSS |
|---|---|---|---|
| item 2a parent (3 reads + gates) | 0 (verified) | 276 s total | not recorded |
| item 2a build children: 12-block, 2025\|Winter, 2035\|Summer | 0 / 0 / 0 (verified) | 146 / 39 / 40 s | not recorded |
| r4 probes, 8 runs (4 pre, 4 post) + compare | 0 (each verified) | about 28 s (SRP1) and 175 s (3 × 3) each | not recorded |
| grouping check | 0 (verified) | 0.2 s | 0.06 GB (`/usr/bin/time -l`) |
| capture check | 0 (verified) | 199 s | 6.87 GB |

- **No IPOPT solve** was run in this task. Every stderr file is 0 bytes.

## Unexpected findings

- **Growth in a 2035-only read is not uniform by element.**
  - TN load pd and qd are equal to the 12-block arrays.
  - TN flexibility P scales by 1.2801, DN flexibility P by 1.3439 and DN qd by 1.1323. DN pd varies, at 1.13–1.29.
  - Not investigated: it does not change decision 2.
- **The build children wrote evaluation work directories** under `data/SRP1/Results/P56A/evals/p515s56s4_*`
  (untracked), as every oracle construction does.

## Not confirmed

- **2025|Winter C_central and both item-1 predictions.** No solve was run (decision 1). Searched: `run_v1/` and the
  step 3 harness source.
- **σ in the window for 2035|Summer.** Indeterminate zero-solve; it settles at an initialisation round, which was not
  run.
- **The D-3 boxes.** pc_init is known only after the initialisation round, which was not run. The list of ADMM-only
  bounds was not compiled for the same reason. The one already known from the code is the TSO `interface_delta` box:
  [−rating, +rating] about the fixed pc, set in `_wire_per_scenario_interface_coupling`; the G-c checks
  `admm_every_delta_free_within_rating` hold on every declared TSO block of the three item-2a builds. No other
  ADMM-only bound was searched for (the search belongs to D-3).
- **The capture on a solved run.** It has been exercised only on stubbed cycles. Its first real run will be the step 4
  smoke.
