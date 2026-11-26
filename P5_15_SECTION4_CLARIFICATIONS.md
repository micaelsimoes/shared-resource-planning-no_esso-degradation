# P5.15 — two clarifications for the Expert (Overleaf `70228dd`, 2026-11-16; zero solves, records only)

## Decisions needed (Expert; wording/notation only)

1. **α notation clash.**
   - **The two uses.** α is the deviation-premium coefficient (nomenclature l. 274; l. 1313, 1851, 1859, 1865). It is
     also 1/(2 S^ref) in the storage updates of Appendix A (l. 1916, 1922–1923, 1930, 1970).
   - **β is free.** It appears nowhere in main.tex, and the response letter contains no α.
   - **Recommendation:** rename the second use to β. The change touches five lines of one appendix subsection: the
     definition at l. 1916, the z-update (l. 1922–1923), the dual update (l. 1930) and the residuals (l. 1970). It is
     notation only and changes neither the formulation nor any number.
   - **Alternative:** keep α and add "(not the premium coefficient of Section 3)" at l. 1916.
   - Either way it goes on the W190 list.
2. **Cycle-limit sentence (l. 2075):** the proposed wording is under Found 1.

## Found

### 1. "min{k₀ + 109, 300} cycles when numerical acceptance is assessed"

The 61 cells of `frozen_step6_tables_v1_590088fe.json` group as follows.

**Dynamic cap min{k₀ + 109, 300}.**
- 42 cells record their cap: all equal min{k₀ + 109, 300}.
- 6 cells record no cap (the A64 cells, the W118 year-ladder cells): their stage specs give the same dynamic rule
  (`frozen_s53_a64_cells_spec_v1_44a2dce8.json`, `frozen_s53_resettle_spec_v2_fc791891.json`).
- The Step-7 cells (W184, W185) use the same rule under a run cap of 300 (e.g. W184 C_NRF: k₀ 134, rule cap 243).

**Continued or re-run evaluations, capped at N_old + 100.** Here k₀ is reproduced by the replay and the earlier run
stopped at N_old = k₀ + 9, so the cap equals k₀ + 109.
- The three reference continuations (W101, `w101_three_reference_summary.json`):

  | evaluation | N | k₀ | cap |
  |---|---|---|---|
  | x = 0 | 132 | 123 | 232 |
  | reference unit | 112 | 103 | 212 |
  | C\* | 87 | 78 | 187 (reached) |

  **These three follow the formula.**
- The six phase-B cells: N_old + 100 equals k₀ + 109 in each.
- The 2× incumbent and challenger (`resettle_decision.json`): 281 = 172 + 109 and 261 = 152 + 109.

**Exceptions: three re-runs on an earlier record, replayed to k₀ and capped at N_old + 100.** Here the cap departs
from the formula:

| cell | plan | k₀ | N_old + 100 | min{k₀ + 109, 300} | ended |
|---|---|---|---|---|---|
| `j_5f3cccb4` | 2×, node 7, 4 h, 5 MWh (claims J:e4_to_e5 and J:e5_value_minus_I; the "41.0–51.8 k€ per MWh" range) | 159 | 270 | 268 | 270, uncertified |
| `l_b2251bc5` | 2030, n5 0.25/0.5 + n7 0.75/3 + n9 0.25/0.5 | 211 | 320 | 300 | 320, uncertified |
| `l_2ab0ce2d` | 2030, n5 0.25/0.5 + n7 1.25/3 | 328 | 437 | 300 (k₀ > 300) | **400, certified** |

Sources: frozen tables, `cells.*.cap` and `k0_run`; W142 spec `caps.above_300`
(`p515_s53_w142_resettle_v6_campaign.py:1162–1165`). Two evaluations ran past 300 cycles, and one certified there.

**Proposed wording:** "… and to min{k₀ + 109, 300} cycles when numerical acceptance is assessed; evaluations re-run
from an earlier record were capped at 100 cycles beyond that record's stopping cycle, which allowed up to 437 cycles
for three of them."

### 2. The two searches (l. 2139–2142): every count and stop reason holds

| search | record (`campaign_results.json`, sha256) | new evaluations | stop |
|---|---|---|---|
| baseline, variant A | s47 `campaign_s47_phase_b` (`906f9da3`) | 14, all at the final unit poll (the 14 completion points, 0 cache hits); the earlier polls' 8 directions each were inadmissible | `mesh_local_optimum_unit_poll_failed`, after an unsuccessful unit poll |
| 2×, variant A | s51 `campaign_s51_f2_phase_b` (`2cdd5d60`) | 20, all in poll 2 (completion; success → 5ca4f86c), so N = N^max = 20 | poll 4: `STOP_FOR_REVIEW_completion_cap`, completion 61 feasible points > cap 30, nothing evaluated |
| 2×, variant B | s53 `campaign_s53_f2_certificate_r1` (`61b9c501`) | 7 (and 3 cache hits), from the incumbent 5ca4f86c | `poll_failure_at_unit_mesh` |

- **"evaluated these fourteen plans":** the 14 new evaluations are exactly the 14-point box (7 node subsets ×
  2 years; W173 E2.1).
- **14 + 20 + 7 = 41**, the search evaluations that use the residual-based stopping rule (W177B).
- **Two optional precisions:**
  - **"reached the completion cap":** the record says the completion set (61) *exceeded* the cap (30), so the poll was
    refused. "Reached" is loose; "stopped when the completion set exceeded the cap" is exact.
  - **"Performed 20 evaluations"** omits "new". The searches also used cached plans; the variant-B search had 3 cache
    hits.
