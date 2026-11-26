# P5.15 S56 step 4: post-run zero-solve diagnostics D1–D5 (2026-11-23)

**Spec:** `data/SRP1/Results/P515S56/step4_postdiag_v1/frozen_step4_postdiag_spec_v1_72f4d80e.json` (v1, no predecessor),
committed before the run in `9185df1b` together with `p515_s56_step4_postdiag.py`.
**Evidence:** `data/SRP1/Results/P515S56/step4_postdiag_v1/` — results `postdiag_results_20261123T170607.json`
(sha256 `bbd1aae1…`), stdout / stderr logs (stderr empty), exit code 0, manifest
`step4_postdiag_v1_manifest_sha256.json`.
**Inputs:** the committed step 4 main-run evidence (`5f1db156`, spec v2 `7ca210fb`), read only. Every input file was
hash-checked against the inventory frozen in the spec before computing (unchanged). Runs at `cb062834`; production
unchanged since. **Instance identity:** x = 0 on A (`s56_step4_2025_Winter`, case sha256 `92e6283b…`) and on B
(`s56_step4_2025_2035_Summer`, case sha256 `b1cd5440…`); per-run candidate keys are recorded in the spec.

**Objective convention (every table):** EUR of total discounted cost. Coupled arms: Q_b = settlement excluded in full
(production primary; pin excluded, salvage separate). e_b = Q_b − w_b·C_central. price_b = −Σ_agents(b) w·settlement_total
= +Σ w·ω·π·(P_T − P_D). dual = the TSO-side dual terms of the evaluator's `priced_residual` (expectation P/Q/V plus the
ω-weighted deviation P/Q/V). o1 = e − price, o2 = e − price + dual, o3 = e − price − dual (= e − PR).

**Status:** the run completed with 0 solves (guard verified at 0, together with the evaluator's and the launcher's
guards). No production file, evaluator, launcher or existing artifact was touched. Q2–Q6 held. **Q1 failed** on its
limit clause.

## Blocked on Planner

Nothing blocks. Three readings need the Planner's judgment:

1. **Q1 failed on clause (b), the o2 limit, on both A runs.**
   - Clause (a) held: o1 > 0 and strictly decreasing on both runs.
   - On A V-1 the failure depends on the estimator, and the limit is indeterminate against the 100 € edge:
     - settling_bar Q̂∞ = 108.6 € (H 2.26, S_b 49.8), which is the frozen decisive form;
     - the held-out fit's own limit is 95.1 €;
     - LSQ3 gives 97.2 € (H 1.66).
   - On A V-2 the failure is clear: the limit is 152.7–161.8 € on every estimator.
2. **Q3's competing outcomes: neither occurred as defined.** On B V-1, o1 falls 23 % over cycles 99–107, a mean of
   −3.22 %/cycle. That sits just outside "≤ 3 %/cycle" and is not a plateau, so it is borderline. The o1 limit on B V-1
   cannot be determined: the three estimators give −3,674, −432 and +540 €.
3. **The Q5 background ("D-3 margins ≥ 37 MW and ≥ 85 MVAr") is not exact.**
   - It holds for the TSO δ box, which is Q5's decisive scope.
   - It does not hold for the DSO7 reference-generator P box on B: 0.563 MW at the central solution and 0.489 MW at
     the terminal ADMM state (2025|Summer), which is near the bound but not active.
   - Separately, the interface V rows sit at their bound on every run (1.1 pu for every expectation entry), and those
     rows are hard only in the ADMM model (Not confirmed, item 1).

## Changed

| commit | content |
|---|---|
| `9185df1b` | `p515_s56_step4_postdiag.py` (new, zero-solve) and the frozen spec v1 `72f4d80e`, committed **before** the run |
| (this commit) | results JSON, run logs and exit code, sha256 manifest, this report |

The script imports the step 4 evaluator's `Run` / `eval_dir`, `priced_residual`, `settling_bar` and `v_base`, and the
launcher's spec, root and tail-depth helpers. It copies none of them.

- **Per-channel split.** `priced_residual` does not split P/Q/V and fails on V-2, which has no V deviation kind. The
  P/Q/V split is therefore recomputed with the same formulas and cross-validated:
  - against the imported `priced_residual`: worst |diff| / A = 2.0e-16 (A V-1) and 3.5e-16 (B V-1);
  - against production `priced_residual_dual_part`: worst relative difference 4.9e-13, 2.8e-12, 2.9e-12 and 1.3e-11
    (A V-1, A V-2, B V-1, B V-2).
- **Before computing,** the script asserted:
  - the capture path: every file and field, on all 8 runs;
  - 36 production lines, verbatim (the sign conventions below);
  - the input hashes.
- **Dry run.** Before the freeze, the code paths were exercised on the step 4 **smoke** evidence only, with output to
  the scratchpad, not the repository. `--check` (capture path only) then passed on the main evidence.

**Sign conventions (verified; no contradiction with the brief):**
- `model_construction_helpers.py`:
  - 1771–1772: the TSO block's settlement is −Σωπ·baseMVA·P_T;
  - 1779–1780: the DSO block's settlement is +Σωπ·baseMVA·P_D;
  - 1847: the settlement enters each local objective.
- `shared_resources_planning.py`:
  - 4876: Q in full = gross − settlement total − pin;
  - 4802: production pr_price = +Σ w ω π (P_T − P_D);
  - 5866–5868: the TSO term is +μ_T (P̄_T − P̄_D)/R;
  - 6146–6148: the DSO term is +μ_D (P̄_D − P̄_T)/R;
  - 9248–9251: μ_T += ρ_T (T − D)·s_base/R;
  - 9258–9261: μ_D += ρ_D (D − T)·s_base/R, with the DSO's own ρ;
  - 4505 / 4537 / 4690–4694: the deviation analogues;
  - 4478–4479: the per-scenario δ box ±R;
  - 4933: pc fixed at pc_init.

## Found: predictions against outcomes

| # | prediction (verbatim in the spec) | outcome |
|---|---|---|
| Q1 | A V-1 and V-2: o1 > 0 at every tail-depth cycle and decreasing; fitted o2 limit ∈ [0, 100] €, held-out error ≤ 50 € | **FAILED.** (a) **held** on both: o1 positive and strictly decreasing (V-1 349.3 → 197.4; V-2 307.7 → 206.8). (b) **failed** on both: o2 limit (settling_bar Q̂∞) **108.6** € on V-1 (H 2.26 ≤ 50, but outside [0, 100]) and **152.7** € on V-2 (H 1.31). Beside, non-deciding: V-1 held-out-fit limit 95.1, LSQ3 97.2 (H 1.66); V-2 157.3 and 161.8. V-1's S_b = 49.8 straddles 100 |
| Q2 | A: terminal o2 ∈ [100, 230] € (hand +156) | **held.** A V-1 o2(46) = **+156.14** €. Beside: A V-2 o2(40) = +170.5 €, also in range |
| Q3 | B 2035\|Summer: terminal o1 ∈ [700, 850] €; competing outcomes recorded | **held.** B V-1 o1(107) = **+767.75** €. Beside: B V-2 o1(84) = +2,766.0 €, out of range. Competing outcome (recorded, not a verdict): B V-1 o1 999 → 768 over cycles 99–107, total −23.0 %, mean −3.22 %/cycle → "decays faster than 3 %/cycle". So it is neither "≤ 3 %/cycle" nor a plateau, and it is borderline. B V-2: −0.99 %/cycle ("≤ 3 %/cycle") |
| Q4 | A: D2 € mismatch at the terminal cycle ≤ 50 € | **held.** A V-1: **0.0 €** (absolute and signed). Exactly 0 on every coupled run at each of the last 10 cycles |
| Q5 | no ADMM-only box active at the terminal cycle on A or B | **held** (TSO δ box, the D3 scope). 0 active on 8/8 runs; minimum margins A 47.34 MW / 98.66 MVAr, B 27.34 MW / 97.95 MVAr. Extended scope, beside: the interface V rows are **active** on every run (below) |
| Q6 | D5 holds every cycle of every coupled run; sign of (pr_price + T) mixed | **held.** 277/277 cycles hold; worst \|pr_price + T\| / (64·u·S_T) = 0.119. Signs (+ / −): 26/20, 17/23, 59/48, 39/45 |

The hand values in the brief were reproduced to within 0.46 € (they are integer-rounded):
- A V-1 o1, cycles 38–46: diffs −0.46 to +0.45 €;
- A V-2 o1 at cycles 32 and 40: −0.30 and −0.25 €;
- B V-1 o3 at cycles 101 and 107: −0.05 and −0.31 €;
- A V-1 o2 at cycle 46: +0.14 €.

### D1: offset series (tail-depth window; the terminal cycle is its last row)

All values are in EUR of total discounted cost, with Q = settlement excluded in full. "exp" and "dev" are the TSO-side
expectation and deviation dual parts per channel. V-2 has no V deviation channel ("—"); its voltage-pin column is the
block's weighted pin, which Q excludes.

**A V-1, 2025|Winter** (calibrated; w_b·C_central = 79,710,848.83 €; K = 46)

| k | e_b | price_b | exp P | exp Q | exp V | dev P | dev Q | dev V | dual | o1 | o2 | o3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 38 | −2,138.6 | −2,487.9 | −61.5 | 4.3 | −0.1 | −3.1 | 0.1 | 0.2 | −60.1 | 349.3 | 289.1 | 409.4 |
| 39 | −1,526.5 | −1,849.5 | −55.2 | 2.5 | −0.4 | −8.3 | 0.0 | 0.4 | −61.0 | 323.0 | 262.0 | 384.0 |
| 40 | −1,452.1 | −1,751.4 | −52.0 | 0.4 | −0.1 | −8.7 | 0.0 | 0.1 | −60.3 | 299.3 | 239.0 | 359.6 |
| 41 | −1,423.0 | −1,702.4 | −50.4 | 0.1 | −0.0 | −9.9 | 0.0 | 0.0 | −60.2 | 279.4 | 219.2 | 339.6 |
| 42 | −1,362.9 | −1,623.4 | −48.0 | 0.1 | −0.0 | −10.5 | 0.0 | 0.0 | −58.4 | 260.5 | 202.1 | 319.0 |
| 43 | −1,277.0 | −1,519.4 | −44.9 | 0.1 | 0.0 | −10.3 | 0.0 | 0.0 | −55.1 | 242.3 | 187.2 | 297.5 |
| 44 | −1,245.7 | −1,471.0 | −41.1 | 0.0 | −0.1 | −9.3 | −0.0 | −0.1 | −50.6 | 225.3 | 174.7 | 275.9 |
| 45 | −1,199.7 | −1,409.9 | −37.3 | −0.0 | 0.0 | −8.2 | 0.0 | −0.1 | −45.6 | 210.2 | 164.6 | 255.8 |
| 46 | −1,129.9 | −1,327.3 | −34.4 | 0.0 | 0.0 | −6.9 | 0.0 | 0.0 | −41.3 | 197.4 | 156.1 | 238.8 |

**A V-2, 2025|Winter** (calibrated; K = 40)

| k | e_b | price_b | exp P | exp Q | exp V | dev P | dev Q | dual | o1 | o2 | o3 | pin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | −862.0 | −1,169.7 | −16.0 | 2.4 | −1.0 | −14.1 | 1.0 | −27.7 | 307.7 | 280.0 | 335.4 | 31.7 |
| 33 | −462.6 | −748.4 | −17.9 | 2.8 | −1.9 | −19.8 | 0.4 | −36.4 | 285.8 | 249.4 | 322.1 | 31.5 |
| 34 | −762.1 | −1,035.2 | −24.0 | 0.1 | −1.1 | −19.4 | 0.2 | −44.1 | 273.1 | 229.0 | 317.3 | 31.3 |
| 35 | −968.4 | −1,233.0 | −29.5 | −0.2 | −0.9 | −21.3 | 0.1 | −51.8 | 264.6 | 212.8 | 316.4 | 31.3 |
| 36 | −1,082.4 | −1,337.3 | −31.8 | −0.2 | −0.8 | −22.2 | 0.0 | −54.9 | 254.9 | 200.0 | 309.8 | 31.3 |
| 37 | −1,103.9 | −1,347.5 | −31.5 | −0.1 | −0.8 | −21.4 | 0.0 | −53.7 | 243.6 | 189.9 | 297.4 | 31.3 |
| 38 | −1,059.7 | −1,292.2 | −29.6 | −0.1 | −0.8 | −20.1 | 0.0 | −50.5 | 232.5 | 181.9 | 283.0 | 31.2 |
| 39 | −942.1 | −1,161.6 | −26.1 | −0.1 | −0.7 | −17.1 | 0.0 | −44.0 | 219.5 | 175.6 | 263.5 | 31.2 |
| 40 | −784.7 | −991.4 | −21.4 | −0.1 | −0.7 | −14.0 | 0.0 | −36.2 | 206.8 | 170.5 | 243.0 | 31.2 |

**B V-1, 2035|Summer** (calibrated; w_b·C_central = 57,114,972.23 €; K = 107)

| k | e_b | price_b | exp P | exp Q | exp V | dev P | dev Q | dev V | dual | o1 | o2 | o3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 99 | 250.9 | −746.5 | −32.2 | 0.2 | 0.2 | −7.3 | 0.0 | 0.5 | −38.5 | 997.4 | 958.9 | 1,035.9 |
| 100 | 514.2 | −468.0 | −43.3 | 2.0 | −0.1 | −18.5 | 0.0 | −0.1 | −60.0 | 982.3 | 922.3 | 1,042.2 |
| 101 | 549.4 | −374.3 | −27.1 | 1.6 | −0.1 | −10.6 | 0.0 | −0.1 | −36.2 | 923.7 | 887.5 | 960.0 |
| 102 | 494.8 | −394.6 | −28.8 | 2.3 | −0.2 | −8.7 | −0.0 | −0.1 | −35.6 | 889.4 | 853.8 | 925.0 |
| 103 | 400.1 | −463.1 | −35.2 | 3.0 | −0.4 | −8.6 | −0.0 | −0.1 | −41.3 | 863.3 | 821.9 | 904.6 |
| 104 | 298.3 | −542.9 | −42.4 | 2.7 | −0.3 | −9.0 | −0.0 | −0.0 | −49.0 | 841.1 | 792.1 | 890.2 |
| 105 | 210.9 | −608.0 | −48.0 | 2.3 | −0.2 | −9.4 | −0.0 | −0.0 | −55.4 | 818.9 | 763.5 | 874.3 |
| 106 | 145.2 | −649.4 | −51.0 | 1.7 | −0.2 | −9.5 | 0.0 | −0.0 | −58.9 | 794.6 | 735.7 | 853.5 |
| 107 | 110.1 | −657.6 | −51.3 | 1.8 | −0.2 | −9.2 | 0.0 | 0.1 | −58.9 | 767.8 | 708.8 | 826.7 |

**B V-2, 2035|Summer** (calibrated; K = 84)

| k | e_b | price_b | exp P | exp Q | exp V | dev P | dev Q | dual | o1 | o2 | o3 | pin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 76 | 2,092.3 | −902.0 | −35.2 | 2.4 | −0.3 | −6.2 | 0.1 | −39.1 | 2,994.3 | 2,955.2 | 3,033.4 | 43.3 |
| 77 | 2,468.5 | −504.8 | −36.2 | 2.8 | −0.5 | −17.5 | 0.0 | −51.4 | 2,973.3 | 2,921.9 | 3,024.7 | 43.3 |
| 78 | 2,539.6 | −374.1 | −18.1 | 2.6 | −0.4 | −8.4 | 0.0 | −24.4 | 2,913.7 | 2,889.4 | 2,938.1 | 43.3 |
| 79 | 2,479.6 | −400.7 | −20.0 | 2.6 | −0.4 | −5.3 | 0.0 | −23.1 | 2,880.4 | 2,857.3 | 2,903.5 | 43.3 |
| 80 | 2,355.4 | −501.0 | −28.5 | 2.6 | −0.4 | −4.4 | 0.0 | −30.7 | 2,856.3 | 2,825.7 | 2,887.0 | 43.3 |
| 81 | 2,214.2 | −621.4 | −38.7 | 2.6 | −0.4 | −4.5 | 0.0 | −41.0 | 2,835.6 | 2,794.5 | 2,876.6 | 43.4 |
| 82 | 2,085.7 | −728.9 | −47.8 | 2.6 | −0.4 | −5.2 | 0.0 | −50.7 | 2,814.6 | 2,763.9 | 2,865.4 | 43.4 |
| 83 | 1,984.9 | −806.9 | −54.2 | 2.6 | −0.3 | −6.1 | 0.0 | −58.1 | 2,791.7 | 2,733.7 | 2,849.8 | 43.4 |
| 84 | 1,917.7 | −848.3 | −57.6 | 2.6 | −0.3 | −6.8 | 0.0 | −62.2 | 2,766.0 | 2,703.9 | 2,828.2 | 43.5 |

**B, 2025|Summer** (C_central MERGED-ONLY, not calibrated; w_b·C_central = 84,481,740.87 €). First and last tail cycles
only.

| run | k | e_b | price_b | dual | o1 | o2 | o3 |
|---|---|---|---|---|---|---|---|
| V-1 | 99 | 594.6 | 76.5 | 2.7 | 518.2 | 520.8 | 515.5 |
| V-1 | 107 | 341.0 | −58.8 | −0.2 | 399.8 | 399.5 | 400.0 |
| V-2 | 76 | 5,506.6 | 104.1 | 16.9 | 5,402.4 | 5,419.4 | 5,385.5 |
| V-2 | 84 | 5,492.8 | 235.8 | 6.6 | 5,257.0 | 5,263.6 | 5,250.4 |

**Observations (not conclusions):**
- **Shape of the series.** e_b and price_b move by hundreds of € per cycle and are not monotone on B (e_b rises 251 →
  549 → 110). o1 = e − price is smooth and strictly decreasing in every tail window.
- **Size of the dual.** The dual is small: −23 to −62 € on the calibrated blocks, and almost all of it is the
  expectation-P part (dev P ≈ −3 to −22 €; Q and V ≤ 4.3 € in absolute value). So o2 and o3 differ from o1 by
  roughly ±40–60 €.
- **What o2 leaves on A.** o2 ends at +156 € (V-1) and +171 € (V-2). Its fitted limits are about +95 to +109 € (V-1)
  and +153 to +162 € (V-2).

**D1 geometric fits** (tail window, last 3 cycles held out). The decisive form is settling_bar: r over all steps, with
r_fit from the fit cycles alone; Q̂∞ = X(K) + d_K·r/(1−r); H is the held-out error. Beside it: the limit of the held-out
fit alone, and LSQ3 (L + A r^k, least squares on the fit cycles). All values are EUR of total discounted cost.

| run / block | series | settling_bar: r (r_fit), Q̂∞, H, S_b | held-out-fit limit | LSQ3: r, L, H |
|---|---|---|---|---|
| A V-1 2025\|Winter | o1 | 0.911 (0.908), 66.4, 0.60, 131.6 | 62.5 | 0.903, 84.1, 3.36 |
| A V-1 2025\|Winter | o2 | 0.849 (0.861), **108.6**, 2.26, 49.8 | 95.1 | 0.860, 97.2, 1.66 |
| A V-1 2025\|Winter | o3 | 0.959 (0.951), −162.8, 1.74, 403.3 | −120.2 | 0.944, −35.7, 6.73 |
| A V-2 2025\|Winter | o1 | 0.968 (0.852), −183.0, 12.17, 401.9 | 178.8 | 0.802, 215.4, 24.27 |
| A V-2 2025\|Winter | o2 | 0.780 (0.764), **152.7**, 1.31, 19.2 | 157.3 | 0.754, 161.8, 3.49 |
| A V-2 2025\|Winter | o3 | not valid (r 1.256 ≥ 1); range bar 92.4 | — | r at grid edge (no limit) |
| B V-1 2035\|Summer | o1 | 0.978 (0.995), −432.2, 7.64, 1,207.5 | −3,674.2 | 0.914, 540.3, 7.08 |
| B V-1 2035\|Summer | o2 | 0.955 (0.952), 133.5, 2.24, 577.6 | 206.7 | 0.953, 183.9, 0.71 |
| B V-1 2035\|Summer | o3 | not valid (steps of both signs); range bar 215.5 | — | 0.885, 681.2, 15.27 |
| B V-2 2035\|Summer | o1 | 0.943 (0.911), 2,338.7, 17.67, 445.0 | 2,622.4 | 0.872, 2,660.3, 8.18 |
| B V-2 2035\|Summer | o2 | 0.985 (0.984), 793.6, 0.33, 1,910.6 | 917.9 | 0.984, 909.2, 0.46 |
| B V-2 2035\|Summer | o3 | 0.931 (0.878), 2,538.8, 24.21, 313.6 | 2,801.7 | 0.798, 2,782.6, 14.73 |
| B V-1 2025\|Summer | o1 / o2 / o3 | Q̂∞ 270.9 / 120.5 / 325.4 (H 3.13 / 0.60 / 4.14) | 321.5 / 149.1 / 363.9 | L 332.8 / 150.8 / 377.7 |
| B V-2 2025\|Summer | o1 / o2 / o3 | Q̂∞ 5,137.1 / 5,107.6 / 5,158.2 (H 2.58 / 0.24 / 4.34) | 5,171.1 / 5,113.3 / 5,199.7 | L 5,190.2 / 5,116.5 / 5,226.6 |

- **Held-out errors are small but do not pin the limit.** Every H is small (0.2–24 €), yet the estimators disagree
  whenever r is close to 1. Small H validates the short-horizon trajectory; it does not validate the limit. Example:
  B V-1 o1, where r_fit is 0.995 and the limits range from −3,674 to +540 €.
- **Terminal step against threshold** (the CLAUDE.md rule) is already reported in the step 4 report. This stage does
  not recompute it.

### D2: dual consistency

- **Exact opposition.** At each of the last 10 cycles of all four coupled runs, every stored interface dual satisfies
  λ_T = −λ_D **bitwise**. This covers 72/72 expectation and 648/648 deviation coordinates per block and channel, for
  P, Q and V (V deviation on V-1 only).
- **Mismatch value.** μ̃_T + μ̃_D = 0.0 at every coordinate: the max, the ω-RMS, and the € value both signed and
  absolute are all 0.0 €.
- **ρ.** ρ_T = ρ_D (pf and v) holds per block at every captured cycle of every coupled run: 0 mismatches.
- **Context:** the largest |μ̃_T| (EUR of total discounted cost per MW in one hour) at the terminal cycle is:
  - exp P: 3,874 (A), 11,505 (B 2035|Summer);
  - dev P: 2,532 (A), 8,053 (B 2035|Summer);
  - exp Q: 440 to 500;
  - V: 13 to 46 €/kV.
- **Meaning of "equal and opposite".** μ̃_D = −μ̃_T, i.e. the DSO's € coefficient on its own copy is minus the TSO's
  coefficient on its own copy. It follows from:
  - the updates: SRP 9251/9261 and 9250/9260, and 4690–4694 for the deviation channel. μ_T += ρ_T·r·s_b/R and
    μ_D += ρ_D·(−r)·s_b/R from 0, so the sum is exactly 0 when ρ_T = ρ_D and s_base_T = s_base_D;
  - the objective terms: 5868 and 6148, 4505 and 4537;
  - the Param conversions: 6824 and 7153, 4607.
- **Scales.** w_T·scale_T = w_D·scale_D on every block. Bases: 100 MVA on both sides; 345 kV on both sides.

### D3: ADMM-only boxes at the terminal cycle

| run | TSO δ box P: min margin (MW) / where | TSO δ box Q: min margin (MVAr) | active | V rows at bound (TSO / DSO; per-scenario on V-1, expectation otherwise) | DSO ref-gen P min margin (MW) |
|---|---|---|---|---|---|
| A V-1 | 47.43 / DN7 s1_0 t4 | 98.67 | 0 | 186 / 116 of 648 (expectation 17 / 12 of 72) | 37.12 |
| A V-2 | 47.34 / DN7 s1_0 t4 | 98.66 | 0 | 11 / 13 of 72 | 37.03 |
| A pinned off | 51.08 / DN7 t4 | 98.80 | 0 | 19 / 9 of 72 | — |
| A pinned on | 51.02 / DN7 t4 | 98.79 | 0 | 20 / 8 of 72 | — |
| B V-1 | 27.34 / DN7 2035\|S s0_2 t14 | 97.95 | 0 | 253 / 179 of 1,296 (expectation 19 / 15 of 144) | **0.489** (DN7 2025\|S) |
| B V-2 | 27.34 / DN7 2035\|S s0_2 t14 | 97.95 | 0 | 11 / 13 of 144 | **0.489** |
| B pinned off | 27.71 | 98.74 | 0 | 23 / 8 of 144 | — |
| B pinned on | 27.71 | 98.74 | 0 | 23 / 11 of 144 | — |

- **Definitions.** "Active" means margin ≤ 1e-6·max(1, bound). Every expectation entry at its bound is at v_max =
  1.1 pu (checked on all 8 runs). The side of the per-scenario entries was not recorded. The minimum margins are 3e-9
  to 3.5e-7 pu. The DSO ref-gen box uses the step 3 `relaxation_limits` (±100 MW for DN7; the D-3 model box
  was ±100.001).
- **Cross-check.** anchor (pc_init) + δ_s, taken from `interface_settlement_detail_s31c.json`, equals the captured
  P̄_T + d_T,s to ≤ 4.1e-14 MW on every run.
- **Every cycle ≥ 21** (coupled runs, beside): the TSO δ box was never active. Its minimum was 27.34 MW (B), at
  cycle 21.

### D4: P7 recomputed post hoc (corrected sign)

**POST HOC.** The frozen P7 outcome (failed) stands. Factor = Σ|e| / Σ|form| over cycles ≥ 21, with "held" meaning
≥ 5. Values are EUR of total discounted cost. The recomputed frozen sums equal the evaluation record to rel ≤ 1e-9
(V-1).

| run / block | n | frozen Σ\|e + PR\| (factor) | (a) Σ\|e − price + dual\| (factor) | (b) Σ\|e − price\| (factor) | tail window only: frozen / (a) / (b) |
|---|---|---|---|---|---|
| A V-1 2025\|Winter | 26 | 220,806 (0.39) | 35,934 (**2.42**) | 41,602 (2.09) | 0.45 / **6.73** / 5.34 |
| A V-2 2025\|Winter | 20 | 229,230 (0.52) | 20,895 (**5.76**) | 19,100 (6.30) | 0.43 / 4.25 / 3.51 |
| B V-1 2035\|Summer | 87 | 1,874,115 (0.66) | 735,278 (1.69) | 718,510 (1.73) | 1.07 / 0.40 / 0.38 |
| B V-2 2035\|Summer | 64 | 1,130,016 (0.47) | 321,255 (1.65) | 370,824 (1.43) | 1.43 / 0.79 / 0.78 |
| B V-1 2025\|Summer (merged-only) | 87 | 953,195 (0.79) | 752,363 (1.00) | 762,276 (0.98) | 0.91 / 1.09 / 1.10 |
| B V-2 2025\|Summer (merged-only) | 64 | 1,064,485 (0.71) | 800,342 (0.95) | 816,117 (0.93) | 0.94 / 1.06 / 1.06 |

Over cycles ≥ 21, the corrected forms reach the frozen fivefold only on A V-2.

### D5: M1 rounding check (c = 64, u = 2^-53)

| run | cycles | max \|pr_price + T\| (€, exact T) | worst \|·\| / (c·u·S_T) | S_T range (€) | signs + / − | Spearman of (pr+T) with \|T\| / residual norm | Spearman of \|pr+T\| with \|T\| / residual norm | (ii) beside: worst ratio |
|---|---|---|---|---|---|---|---|---|
| A V-1 | 46 | 1.83e-7 | 0.104 | 2.47e8–3.98e8 | 26 / 20 | −0.28 / −0.29 | 0.01 / −0.04 | 0.0070 |
| A V-2 | 40 | 1.55e-7 | 0.080 | 2.48e8–3.87e8 | 17 / 23 | −0.16 / −0.21 | 0.03 / 0.05 | 0.0081 |
| B V-1 | 107 | 5.31e-7 | 0.114 | 3.64e8–8.02e8 | 59 / 48 | 0.04 / 0.06 | 0.03 / −0.02 | 0.0076 |
| B V-2 | 84 | 3.09e-7 | 0.119 | 3.64e8–8.04e8 | 39 / 45 | −0.01 / −0.01 | 0.15 / 0.18 | 0.0105 |

- **Summation error of T itself.** The evaluator's float sum differs from the exact T by ≤ 4.9e-8 €, and math.fsum by
  ≤ 3.5e-8 €. With the evaluator's own float sum, the worst |pr_price + T| per run reproduces its recorded M1 "beside"
  value bitwise.
- **Recomputed settlements.** Rebuilt from the captured P and the per-scenario prices, the settlements match the
  stored s_a to ≤ 2.1e-15 relative.
- **S_fine / S_T** is 1.0002–1.22.
- **Residual norm** = sqrt(boyd_pf_r_exp² + boyd_pf_r_dev²).

## Not confirmed

1. **The V-row bound multipliers.**
   - The interface V rows sit at their bound on every run: TSO per-scenario 186/648 on A V-1 and 253/1,296 on B V-1.
     Every expectation entry at its bound is at 1.1 pu. The TSO interface-bus voltage slacks are fixed at 0 in the
     ADMM model (SRP 4922–4923).
   - Their multipliers are not captured, so their contribution to any first-order identity is not measured.
   - Searched: the capture and sidecar headers, `per_cycle_record`, `interface_voltage_terminal.json` (distances
     only), `boyd_terminal.json` and `component_levels_terminal.json`.
2. **DSO reference-generator boxes for DN5 and DN9.**
   - Their limits are taken from step 3's `relaxation_limits` (±250). The ADMM model's own box was read by D-3 only
     for DN7 (±100.001; box_example in `step4_diag_v1/run_v1/step4_diag_results.json`).
   - The DN5 and DN9 margins exceed 100 MW, so the 0.001 difference does not matter there.
3. **The o2 limits.**
   - None is determinate on B: S_b runs to 578 (V-1) and 1,911 € (V-2).
   - On A V-1 the limit is indeterminate against the 100 € edge, because the estimators span 95–109 € with S_b 49.8.
   - The terminal-to-threshold ratios were not recomputed here.
4. **What the D1 numbers mean mechanistically.** This stage measures o1, o2 and o3 and their decay. It does not test
   why o2 tends to about +100 € on A V-1 and about +155 € on A V-2 rather than to 0.
5. **The Q3 classification thresholds** (3 % total for a plateau; 3 %/cycle for the decay) are the Worker's
   operational reading, frozen in the spec before the run. B V-1's −3.22 %/cycle sits 0.22 points from the boundary.
