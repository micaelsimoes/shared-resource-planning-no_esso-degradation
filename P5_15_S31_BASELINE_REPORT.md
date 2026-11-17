# P5.15 Step 3.1 — Post-signature baseline campaign and reading rule

**Planner report, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 10; signed table
`P5_15_S31_PENALTY_TABLE_DRAFT.md` (signed, `6525c61f`) §5 reading rule. **Stops for review before 3.2.**

## 0. What ran

- Signed rows implemented in one commit (`fc200780`): row 3 (TSO interface flexibility charge removed), row 5 (RES
  curtailment penalty 0 on both sides), row 8 (ESS-usage flag split), row 9 (network complementarity penalty removed),
  row 14 / D2 (orphan slacks removed and fixed at 0), D-category reporting fields for rows 10–13, 15, 16, D3 (ESSO slacks
  reported directly), D6 (shared-ESS day-balance slack bounded). Zero-solve verification passed for every change;
  preserved fixtures still unpickle.
- Level-capture arm committed before the gate (`b2c24371`). Campaign `s31` at C\*, production defaults, cold, one
  background run through the tool, stderr captured, exclusive lock, heartbeat. **Python exit code 0.**

## 1. Headline

**The campaign did not converge**: 90-cycle cap, rule-ten ratio **30.26**. Under the frozen reading rule, whether Q(x)
moved is therefore **INDETERMINATE**. Recourse was still rising by 1.3–2.3 M per cycle at the cap, against a tolerance of
~51,800.

| | Step 3.0 baseline | **s31 (post-signature)** |
|---|---|---|
| converged | 71 cycles | **no — 90-cycle cap** |
| rule ten | 0.847 | **30.26** |
| recourse (`gross_operational_cost`) | 817,520,272.93 | **517,777,950.83** at cycle 90 (not a fixed point) |
| terminal objective step | 69,273 | **1,566,735** |
| local-solve failures | 0 | 0 |
| network failures | 22, all tier 1 | **69 — 68 tier 1, 1 tier 2**, 0 unrecovered |
| solves | 3,694 | 4,711 |
| ESSO terminal `lg(mu)` | −11.0 | −11.0 (all 273 solves) |

Recourse trajectory: −52,160 at cycle 1, climbing monotonically; 510.3 M at cycle 85, 517.8 M at cycle 90.

## 2. Reading rule — as specified, at a non-converged point

| quantity | value |
|---|---|
| Δ = s31 gross − Step 3.0 recourse | **−299,742,322** (−36.7 %) |
| rule-nine bar (sum of terminal steps) | 1,636,008 — **not valid** (s31 did not settle) |
| verdict | **INDETERMINATE** |

**Attribution at the s31 terminal point** (exact identity: Step 3.0 − s31 = definitional + path, closes to −3e-7):

| row | value the removed/zeroed term would take at this point under pre-signature weights |
|---|---|
| **3 — TSO ADN-interface flexibility charge** | **2,648,518,218** |
| 5 — DSO RES curtailment at weight 1 | 530 |
| 9 — bilinear ESS complementarity | 19.5 |
| 14 — orphan flexibility slacks | 0 (variables fixed at 0) |
| **definitional total** | **2,648,518,768** |
| old definition of Q at the s31 point | 3,166,296,719 |
| path effect (Step 3.0 recourse − old-definition Q at s31 point) | −2,348,776,446 |

**Reading, with the limit stated.** Row 3 dominates every other signed change by six orders of magnitude: at this point
the removed interface charge would be 2.65 e9, five times the entire new recourse. Rows 5, 9 and 14 are negligible. The
split between "definitional" and "path" is exact but not interpretable as an equilibrium comparison, because the s31
point is not an ADMM fixed point; the path term absorbs non-convergence.

At Step 3.0's own first cycle (old code), recourse was 2,808,928,724, and the pre-flight's row-3 definitional value at
the new code's first cycle was 2,803,570,588 — **99.8 %**. The recourse the programme has tracked was, at least early in a
run, almost entirely the anchor-dependent transfer the signed table removed.

## 3. D rows — terminal values (weighted, recourse units)

| row | term | terminal value |
|---|---|---|
| 10 | local ESS day-balance slack | 0 (no local ESS in SRP1) |
| 11 | shared-ESS day-balance slack | −58.06 |
| 12 | squared-voltage slacks | −11,796.94 |
| 13 | flexibility P day-balance slack | −946.50 |
| 15 | node-balance slacks | 0 (inactive) |
| 16 | branch-flow slacks | 0 (inactive) |
| | **detector total** | **−12,801.50** |
| 19 | ESSO aggregate slacks (D3, raw) | −1.7e-5 |

All nonzero D rows are **negative and near-constant**: the slacks sit at IPOPT's bound-relaxation floor (≈ −1e-8 per
variable, multiplied by weights and period counts). They are values of ~zero, not violations.

**Reported Q(x) under the signed D semantics:** 517,790,752.33 with all D rows excluded; 517,789,747.77 with only row 12
excluded. Both differ from `gross` by **+0.0025 %** — **D reclassification is presentation-only; it does not move Q(x).**

Economic components at terminal (weighted): generation cost 308,410,740; internal flexibility cost 209,380,012; load
curtailment, RES curtailment and ESS usage 0.

## 4. Tier 2 fired inside a campaign — for the first time

DSO `case33_3` 2035 Spring, cycle 88: tier-1 cold retry failed; tier-2 (cold + adaptive μ) recovered. This closes the
Step 1 open item "tier 2 not yet exercised in a campaign".

## 5. Why s31 may not converge — hypothesis, not established

**H-free-interface.** Row 3 removed the only objective term on the TSO's ADN-interface flexibility variables. Those
variables remain **free** (bounded by the interface transformer rating, `shared_resources_planning.py` ~2973–2980) and
are now **unpriced** in the TSO, and the interface day-balance rule is skipped for them. The TSO therefore sees a
zero-cost direction at each interface constrained only by the augmented-Lagrangian terms. Consistent with it, but not
proof of it:

- recourse drifts monotonically for 90 cycles instead of settling;
- network failures triple (69 vs 22), concentrated in DSOs (65 of 69);
- DSO internal flexibility cost at terminal is 209 M, 40 % of recourse;
- the definitional row-3 value (a proxy for the magnitude of `flex_p_down + flex_q_down` at the interfaces) is 2.65 e9.

**Alternative:** the change simply exposes the old stopping rule. Its tolerance is relative to recourse, and with the
row-3 mass gone the run starts near zero and must travel far; 3.2's stopping rule might be sufficient on its own. The two
are not exclusive.

**Cheapest discriminator (zero extra ADMM runs):** capture the terminal TSO ADN-interface flexibility variables
(`flex_p_up/down`, `flex_q_up/down` per interface and period) and the interface consensus residuals. Large, persistent
opposing up/down values would support H-free-interface.

## 6. Decisions for review before 3.2

1. **H-free-interface.** Whether to treat the free, unpriced TSO interface flexibility as a defect to resolve before 3.2.
   Options for the author: fix the interface flexibility variables at 0 in the TSO (the interface is already governed by
   consensus); keep them with a symmetric regularization (category R); or leave them and rely on 3.2.
2. **The Step 3.0 baseline's role.** Its recourse (817.5 M) contained a charge now removed. It remains valid as a
   determinism reference but not as an economic Q(x) reference. A converged post-signature run is needed before 3.2's
   gate has something meaningful to compare with.
3. **Whether to run the discriminator in §5** before deciding.

## 7. What is NOT established

- Whether Q(x) moved at convergence, and by how much: s31 did not converge.
- The cause of non-convergence: H-free-interface is a hypothesis.
- The terminal economic recourse under the signed table.

## 8. Evidence

`data/SRP1/Results/P515S31_run/` with `evidence_manifest_sha256.json` (per-period capture hash-recorded, not committed).
Key hashes (first 16): `component_levels_terminal.json` `7663b362a7bff367`, `g_baseline.json` `eda2cdbcc8d4cc46`,
`stdout_baseline.log` `7033172b38226e76`, `network_failures_baseline.jsonl` `2d6b1802301956fd`. Reading rule:
`p515_s31_reading_rule.py` → `P515S31_run/s31_reading_rule.json`. Implementation: `WORKER_REPORT_S31_IMPL.md`;
zero-solve checks `data/SRP1/Results/P515S31/zero_solve_checks.json`.
