# P5.15 — centralised single-day check, Step 2 result (2026-11-16)

**Authority:** author/Expert GO of 2026-11-16, with three rulings:
1. loads fixed at demand;
2. IPOPT tol 1e-5, compl_inf_tol 1e-6, MA97, recovery off, one attempt per run;
3. a warm-started second solve per day.

**Spec:** `frozen_central_check_spec_v1_0e00f6a8.json`.

**Harness:** `p515_s55_central_check.py`. Both were committed in `3bb5cdf6` before the launch.

**Run:** attached, with stdout and stderr captured (stderr is 0 bytes). Total wall time 50.2 s.

**Guards:** exactly 8 solves and 8 executions from the declared call site, 0 blocked; 1 pickle load, of the pinned
`certified_models.pkl` `99ab1070`.

**Objective convention:** C = TN generation cost + DN flexibility cost, unweighted € per representative day.
- **Slack and detector terms** are reported beside C, not in it.
- **Comparator:** `ref:7aa017f0`, `component_levels_terminal.json` (`033c1e7e`). It is the sum of the TSO and three DSO
  blocks' `generation_cost + flexibility_cost_internal`.

## Decisions needed

None. Every prediction held, and the spec fixed how each outcome reads.
- **Wording, for the Expert:** the check supports a sentence such as: "For the plan without storage, a centralised AC
  OPF of each 2025 representative day (TN and the three DNs in one model) reached the coordinated operating cost to
  within 0.13 € per day (≤ 9 × 10⁻⁷), from both a default and a coordinated start."
- **One caveat.** The hourly interface schedules and voltages are not unique at that cost (see Found). The paper should
  claim the same cost, not the same dispatch.

## Blocked on the author

- Push from PyCharm.

## Changed

- `3bb5cdf6`: harness and frozen spec.
- This commit:
  - `central_check/run_v1/` (`central_check_results.json`, eight IPOPT logs);
  - `run_v1_stdout.log` and `run_v1_stderr.log`;
  - `manifest_sha256.json`;
  - this report and the TASKS.md line.

## Found

### Per day (primary = default start, the centralised result; warm = started from the coordinated solution)

| day | start | status | iterations | wall (s) | C centralised (€) | C coordinated (€) | difference (€) | difference (%) | bars |
|---|---|---|---|---|---|---|---|---|---|
| Spring | primary | Optimal Solution Found | 88 | 2.8 | 139,122.530 | 139,122.406 | **+0.125** | +8.95e-5 | +0.14 |
| Spring | warm | Optimal Solution Found | 84 | 2.9 | 139,122.530 | 139,122.406 | +0.125 | +8.95e-5 | +0.14 |
| Summer | primary | Optimal Solution Found | 72 | 2.4 | 180,444.151 | 180,444.168 | **−0.016** | −8.95e-6 | −0.01 |
| Summer | warm | Optimal Solution Found | 72 | 4.3 | 180,444.151 | 180,444.168 | −0.016 | −8.95e-6 | −0.01 |
| Autumn | primary | Optimal Solution Found | 65 | 2.1 | 169,308.536 | 169,308.537 | **−0.001** | −6.2e-7 | −0.00 |
| Autumn | warm | Optimal Solution Found | 103 | 3.1 | 169,308.536 | 169,308.537 | −0.001 | −6.2e-7 | −0.00 |
| Winter | primary | Optimal Solution Found | 84 | 2.6 | 182,644.269 | 182,644.280 | **−0.011** | −5.9e-6 | −0.01 |
| Winter | warm | Optimal Solution Found | 67 | 2.3 | 182,644.269 | 182,644.280 | −0.011 | −5.9e-6 | −0.01 |

- **Aggregate over the four days (primary):** 671,519.487 € against 671,519.390 €, a difference of **+0.097 €
  (1.4 × 10⁻⁷)**.
- **Bars:** one bar is the coordinated certification band scaled to the day (6.44 × 10⁻⁶ × C): 0.90, 1.16, 1.09 and
  1.18 €. Every difference is ≤ 0.14 bar.
- **Primary and warm** end at the same cost on every day, within 1.2 × 10⁻⁸ €.
- **Full objective against Q-day** (slack terms included, € per day):

  | day | merged objective | coordinated Q-day |
  |---|---|---|
  | Spring | 139,119.843 | 139,119.790 |
  | Summer | 180,441.464 | 180,441.552 |
  | Autumn | 169,305.848 | 169,305.921 |
  | Winter | 182,641.582 | 182,641.664 |

  The slack terms are −2.69 € (merged) against −2.62 € (coordinated): relaxed-bound artefacts of the same kind on both
  sides.
- **Feasibility after each solve:** the largest constraint violation is ≤ 1.2 × 10⁻⁸.
- **IPOPT objective scaling factor:** 0.001.
- **Dropped reference-generator limits:** the centralised interface never exceeds them on any day or hour. That
  includes DN 7, whose coordinated import came within 0.25 MW of its 100 MW limit in Spring.

### Interface, hour by hour (warm and primary are identical to the reported precision)

| day | max \|ΔP\| vs coordinated DSO side, DN 5 / 7 / 9 (MW) | max \|ΔQ\| (MVAr) | max \|ΔV\| at the TN buses (pu) | coordinated TSO–DSO P mismatch (MW) |
|---|---|---|---|---|
| Spring | 0.078 / 0.030 / 0.034 | ≤ 0.0023 | 8.1e-3 | ≤ 1e-4 |
| Summer | 0.021 / 0.002 / 0.023 | ≤ 0.0002 | 2.0e-3 | ≤ 1e-4 |
| Autumn | 0.000 / 0.078 / 0.001 | ≤ 0.0013 | 1.4e-3 | ≤ 1e-4 |
| Winter | 0.017 / 0.044 / 0.071 | ≤ 0.0004 | 3.0e-3 | ≤ 1e-4 |

The two solutions are **essentially the same operating point.**
- **Interface power:** they differ by at most 0.08 MW per hour on interfaces carrying up to about 100 MW.
- **Cost:** within 0.13 € per day.

The differences that remain are larger than the coordinated TSO–DSO mismatch (≤ 10⁻⁴ MW) yet change cost by ≤ 0.13 €:
- **P:** flexibility can be shifted between equal-price hours at no cost.
- **Voltage:** the TN voltage level is unpriced. ΔV is identical at the three TN buses on each day, which reads as a
  uniform level shift.

The hourly series are in `central_check_results.json`, under `days.<day>.<start>.interface_merged` and
`interface_coordinated`, with `hourly_dP_vs_dso_mw`.

### Predictions (recorded in the spec before launch) against outcomes

| prediction | outcome |
|---|---|
| P1: primary "Optimal Solution Found" on 4/4 days | **held**, 4/4 |
| P2: warm Optimal and \|C_warm − C_coord\| ≤ 1e-4 × C_coord on 4/4 | **held**, largest 8.95e-7 (Spring, +0.125 €) |
| P3: \|C_primary − C_warm\| ≤ 1e-4 × C_coord on 4/4 | **held**, ≤ 1.2e-8 € (the same point) |
| P4: C_warm ≤ C_coord + 3 bars on 4/4 | **held**, largest +0.14 bar |
| P5: warm max\|ΔP_int\| ≤ 0.1 MW per DN | **held**, largest 0.078 MW (Spring DN 5, Autumn DN 7) |

**Reading** (as declared in the spec): |C_primary − C_coord| ≤ 3 bars on every day, so the centralised solve and
coordinated operation reach the same operating point within resolution.
- **Sign:** the small differences carry no sign.
- **Spring:** the +0.125 € there is of the size of the coordinated solution's price-weighted interface mismatch. The
  design review estimated it at +0.14 € from the settlement records, which I did not re-derive.

### Pre-solve gates (all four days, before any solve; 20/20)

- **Model alignment:** the pickled models match the networks read here.
- **Objective Params:** they agree among the coordinated models. `penalty_gen_curtailment` (1.0 → 0) and
  `penalty_shared_ess_usage` (0.1 → 0) were aligned to the ADMM values.
- **Warm-start coverage:** all 28,488 used free Vars are set.
- **At the mapped coordinated point:**
  - C equals the comparator to within 9 × 10⁻¹¹ €;
  - the largest constraint violation is 1.0–3.2 × 10⁻⁴, at the interface rows (the consensus residual).

## Not confirmed

- **Global optimality.** It is not established for either solution. Two starts reaching the same point is evidence of
  a common basin, not a proof.
- **The per-day bar** is the overall certification band scaled to the day, not a per-day measurement.
- **Only the four 2025 days were run, as specified.** 2030 and 2035 and the multi-scenario configuration were not run.
- **The +0.14 € consensus-mismatch estimate** comes from the Advisor's reading of the settlement records. It was not
  recomputed here.
