# Figure fig:res_q_cycle as three subfigure files, v6 (zero solves)

The manuscript uses these three files as subfigures (Expert, decision 1 option (a), 2026-11-13). Panel (a) is
adopted as is from `fig_q_cycle_v4/`; (b) and (c) are re-exported. v6 supersedes v5 (`fig_q_cycle_v5/`, kept): at the
author's request (2026-11-13), (b) and (c) now have identical x axes, and the x label sits on (c) only. All three are 95 × 70 mm
with two-decimal y ticks and no panel letters (LaTeX labels them).

## For W190: the three subfigure files

| file | sha256 | panel |
|---|---|---|
| `fig_q_cycle_a.pdf` | `6bfb74a1abff991db4064935fd8b198d88b3ba27c13e9ca67ffe323e760a86a3` | (a) objective of the plan without storage (x = 0), Q − Q(k*) |
| `fig_q_cycle_b.pdf` | `65ec70139332477f4c0469a2e45e2c63e220f63dbc3f99435dd937356e96c549` | (b) objective of the reference unit at 1.5×, Q − Q(cap) |
| `fig_q_cycle_c.pdf` | `cbad023bbf0e001373a9debdbc67d74cdb82757668fde8b2acc91ed38f5cc8ff` | (c) priced interface-consensus gap of the same evaluation, with ±τ/2 |

- **(a)** is a byte copy of `Claude outputs/fig_q_cycle_v4/fig_q_cycle_a.pdf`. A fresh render from the records reproduced it bitwise.
- **(b) and (c)** share one axes rectangle and one set of x ticks, so stacked at the same width their x axes coincide:
  - rectangle (figure fractions): left 0.1596, bottom 0.1308, width 0.8292,
    height 0.8540;
  - x range 84–195;
  - x ticks 100, 120, 140, 160, 180.
- **(b)** keeps its x tick labels but drops the x label; "ADMM cycle" is on (c) only.
- **(c)** has no gap-refusal ticks and no legend. v2–v4 drew 59 vermillion ticks labelled "gap-clause refusals".

**Legend text.** The only legend left is (a)'s "turning points", in plain words; (b) and (c) have none.

## The record each panel reads

- **(a)** cell `ref:7aa017f0` (T2), eval dir `data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0`:
  - Q per cycle = `gross_operational_cost` from `data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0/per_cycle_record.jsonl` (sha256 `bf38d5a5`);
  - k0 = 123, N = 132, k* = 181, the window 150–181, its band and the
    turning points from `data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0/settling_decision.json` (sha256 `6c559f2f`);
  - range/τ from the frozen tables (`tables.cells.ref:7aa017f0.range_over_tau`).

  Cross-checked against `data/SRP1/Results/P515S53/w101_srp1_continuation/w101_three_reference_summary.json` (sha256 `87234aed`) and `data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0/settling_continuation_cycle_record.jsonl` (sha256 `a0c8551a`).
- **(b)** cell `h_f9eae48f` (T2, chosen by the W165 panel rule), eval dir `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f`:
  - Q per cycle = `gross` from `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f/resettle_cycle_record.jsonl` (sha256 `ddfe92db`);
  - k0 = 85, cap = 194, the shaded last 60 cycles
    (135–194) and the turning points from `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f/resettle_decision.json` (sha256 `462b2f3f`).

  Cross-checked against `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f/per_cycle_record.jsonl` (sha256 `b5ab3d9d`).
- **(c)** the same evaluation as (b):
  - the gap per cycle (`t_sum`) from `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f/resettle_cycle_record.jsonl` (sha256 `ddfe92db`);
  - the gap bound τ/2 = 2,269.534 € from `data/SRP1/Results/P515S53/w142_resettle_v6/campaign_s53_w142_resettle_v6_h_f9eae48f/evals/aed2d618a19ca483_h_f9eae48f/resettle_decision.json` (sha256 `462b2f3f`).

**Source of the data.** All panels come from the W165 figure script's data function, `data_fig4`; all
41 of its checks held, on the frozen tables `frozen_step6_tables_v1_590088fe.json`. Before any panel was
drawn, the composite re-rendered from the same data was byte-identical to the committed `fig4_q_versus_cycle.pdf`
(sha256 `45755c82`).

**Objective convention.** Q = `gross_operational_cost`, contracted interface settlement excluded.
- (a) is relative to Q(k*) = 653,873,702.1877 €.
- (b) is relative to Q(cap) = 743,972,938.5179 €.

## What differs from the composite `fig4_q_versus_cycle.pdf`

- **All three:** one size; y tick labels at two decimals; no panel letters (author, 2026-11-13).
- **(b):** its own x tick labels. In the composite these were hidden because (b) shared its x axis with (c). No x
  label; (c) carries it, as in the composite.
- **(b), (c):** one fixed axes rectangle and the same x ticks (v6), in place of each file's own constrained layout. The
  x limits are (85 − 1, 194 + 1), as before.
- **(c):** no gap-refusal ticks and no legend (Expert, 2026-11-13).
- **Unchanged:**
  - axes, limits and labels;
  - the markers: $k_0$, N and k* in (a); $k_0$ and the cap in (b) and (c);
  - turning points, shading, the band lines and the ±τ/2 band.

**Earlier versions, kept:** `fig_q_cycle_v2/`, `v3/`, `v4/` and `v5/`.

**Zero solves:** the W165 and W157 guards were verified at 0, and the pickle load counters at 0.
