# P5.15 Addendum 25 item 1 — campaign-harness gate: Planner ruling (PASS)

Planner note, 2026-09-19. Spec v14 `data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`, `item1_harness.gate`:
"a C* evaluation through the harness, concurrent with two other evaluations, reproduces D bitwise on every numeric
trajectory field and cost". Evidence: `WORKER_REPORT_S44_HARNESS.md`, `campaign_s44_gate/gate_results.json`,
`gate_tie_order_analysis/tie_order_analysis.json` (commits `41758012`…`f825b8b7`).

**Ruling: the gate PASSES.**
- **The gate's own terms hold exactly.** C* through the harness, concurrent with two evaluations: certification at cycle
  139, cost 650,966,975.2943751, 7,179 solves; 0 unclassified differences across all 139 trajectory rows of the full
  report, the four terminal artifacts and the four numeric sidecars (ESS stride, SoH floor, PF stride, ESS exempt state).
- **The strict comparator's 43 unclassified differences are tie ORDER in one diagnostic list** — the recourse-jump
  sidecar's top-10 `objective_component_block_deltas` — whose sort key I changed deliberately at `8682cfdd` (to a total
  order by name). The tie analysis, committed before it ran, explains all 139 of 139 rows:
  - 50 are identical;
  - 78 are reproduced exactly by re-sorting D's committed list with the new key;
  - 11 are exact-tie groups straddling the 10-entry cut.

  No numeric value differs.
- **The Worker was right not to override the strict result.** Ruling on it is the Planner's call, and it rests on
  the committed analysis, not on a comparator change.

**Consequences.**
- **The two companions count for the selection run (spec v14 item 3).** D certifies at the paper's plan (cycle 139,
  cost 653,029,766.99, bar 661.6) and at node-7-empty (cycle 136, cost 651,900,014.16, bar 9,104.7). The stop rule is
  not triggered.
- **Equal cycle counts are a coincidence.** The paper's plan and C* both certify at 139 but their costs differ by
  2,062,792.
- **Peak RSS:** 2.18–2.22 GiB per evaluation with three concurrent; each IPOPT subprocess peaks at 53–57 MB.
- **Per-node zero storage is evaluable.** The paper's plan and node-7-empty certified with 0 local failures, and each
  zero node's published capacity is exactly 0.0.

**Deviation from spec v14, recorded.** Spec v14's per-candidate report lists "hull polish per arm". The D-arm companions
ran without an in-process hull polish and without persisted certified models. Addendum 25's adoption rule requires
(b)–(d) only for the AA arm, so D-arm hull polish is reported only where available. The harness gains an optional
post-certification step (decomposition against a reference, hull polish, persisted models) before the AA-arm
evaluations.

**Follow-ups folded into the next bounded task:**
- `block_deltas`, the sibling list in the same capture hook, gets the same sort-key fix;
- the legacy one-run lock function refuses to start while a campaign lock exists;
- the tie classifier for gates against pre-`8682cfdd` sidecars adopts the analysis's re-sort-and-straddle test rather
  than a longer alias list.
