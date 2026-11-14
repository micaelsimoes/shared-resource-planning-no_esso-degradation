# Figure fig:res_q_cycle as three subfigure files, v7 (zero solves)

The manuscript uses these three files as subfigures (Expert, decision 1 option (a), 2026-11-13). v7 supersedes v6:
the y-axis labels are now words, not symbols or record field names (Expert, 2026-11-13), and nothing else changed.

| panel | y label, v7 | y label before |
|---|---|---|
| (a) | Objective change (k€) | Q − Q(k*) (k€) (v4) |
| (b) | Objective change (k€) | Q − Q(cap) (k€) (v6) |
| (c) | Interface-consensus gap (k€) | t_sum (k€) (v6) |

- **x label:** "ADMM cycle", already in words, on (a) and (c) only.
- **Only the labels changed.** Before relabelling, the script rendered the panels with their old labels and
  reproduced the predecessors bitwise:
  - (a): `Claude outputs/fig_q_cycle_v4/fig_q_cycle_a.pdf` (`6bfb74a1`);
  - (b): `Claude outputs/fig_q_cycle_v6/fig_q_cycle_b.pdf` (`65ec7013`);
  - (c): `Claude outputs/fig_q_cycle_v6/fig_q_cycle_c.pdf` (`cbad023b`).
- **Other in-plot text, kept as before:**
  - markers $k_0$, N and k* in (a), and $k_0$ and "cap" in (b) and (c);
  - the shading notes;
  - ±τ/2;
  - the (a) legend "turning points".

All three are 95 × 70 mm with two-decimal y ticks and no panel letters (LaTeX labels
them).

## For W190: the three subfigure files

| file | sha256 | panel |
|---|---|---|
| `fig_q_cycle_a.pdf` | `1c8f2239234c09cdd91c082f8ba34ccdaf96f00ca54274c17c8b491c9fd070b2` | (a) objective of the plan without storage (x = 0), relative to certification |
| `fig_q_cycle_b.pdf` | `84f90c7e1ac740914a67a2f8c61205a9dbb198f01d461df988c16d4f5b6ab130` | (b) objective of the reference unit at 1.5×, relative to the cap |
| `fig_q_cycle_c.pdf` | `c2c6ee783584da3713ea7b7b44b01b758fb4c9ea44eeb6cba2b4d8552002da57` | (c) priced interface-consensus gap of the same evaluation, with ±τ/2 |

- **(a)** is v4's panel (a) (the version adopted in v5) with the new y label.
- **(b) and (c)** share one axes rectangle and one set of x ticks, so stacked at the same width their x axes coincide:
  - rectangle (figure fractions): left 0.1465, bottom 0.1308, width 0.8423,
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
  - axes and limits (the y labels changed in v7, above);
  - the markers: $k_0$, N and k* in (a); $k_0$ and the cap in (b) and (c);
  - turning points, shading, the band lines and the ±τ/2 band.

**Earlier versions, kept:** `fig_q_cycle_v2/` to `v6/`.

**Zero solves:** the W165 and W157 guards were verified at 0, and the pickle load counters at 0.
