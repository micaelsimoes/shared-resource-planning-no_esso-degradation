# P5.15 Section 4: three facts, one figure, two audit items (2026-11-17; zero solves)

**Request:** the Expert's handback against Overleaf `5c974a4`. **No manuscript edits.** Every quantity comes from
committed records or from two new zero-solve extraction scripts. Both scripts ran under
`SolveProfileGuard(permitted=())`, verified at 0 solves and 0 executions.

## Decisions needed (Expert; wording only)

1. **Item 5: two conventions are mixed in one sentence.**
   - **The problem.** 808.3 GWh is the positive-part total. The "807.6 GWh in the TN" beside it is the net figure:
     on the positive-part convention the TN share is 807.667 GWh, which rounds to **807.7**.
   - **Options:**
     - (a) positive part throughout: 808.3 / **807.7** / 0.7. This is a one-number edit. **Recommended**, because the
       text has already moved to positive part.
     - (b) net throughout: 808.2 / 807.6 / 0.7.
   - **The coordinated 0.7 GWh** reads the same under both conventions (positive part 0.728, net 0.661).
2. **Item 2: what the paper can say about the reference unit.**
   - **The recorded reason is a screening rule.** It was not chosen for a physical property (details in Found 2).
   - **"Highest operating value among single units" no longer holds** under the baseline calibration.
     - The node-5 unit of the same size is closer to break-even: +53.6 k€ against node 7's +58.5 k€.
     - All three nodes are within the 14.8 k€ resolution of one another.
   - **Recommended wording:** "The reference unit is the smallest 4 h unit at node 7. It was the 2025 single-node plan
     closest to break-even in the initial screening and is kept as the common reference for the sensitivity analyses."
3. **Number format in the figures (optional).** The figures print "8 000" (thin space) and "10 %" / "50 %" (space
   before %). US style is "8,000" and "10%". Change these only if the text uses the US form; each is a one-string
   change.

## Blocked on the author

- Push from PyCharm.
- Upload of the regenerated `fig_soh_reference_unit.pdf` to Overleaf.

## Changed

This commit adds:
- **`p515_s55_curtailment_mechanism.py`** and its output `data/SRP1/Results/P515S55/curtailment_mechanism/`:
  `curtailment_mechanism.json` `c13cd680`, `launch.log` and the manifest.
- **`p515_s55_count_admissible_plans.py`** and its output `data/SRP1/Results/P515S55/admissible_plan_count/`:
  `admissible_plan_count.json` `e6c7a442`, `admissible_plans.csv` `393c1d1e`, `launch.log` and the manifest.
- **`manuscript_review/fig_storage_profiles.py`**, with a new `--us` option. It implies `--v2` and changes printed
  legend text only. `--v2` still rebuilds every `figures_s4_v1` storage file byte for byte.
- **`Claude outputs/figures_s4_v2/`**: `fig_soh_reference_unit.pdf` `6548969e`, the four SoC files (byte-identical to
  v1), previews, verification, NOTE.md and the manifest.
- This report and the TASKS.md line.

## Found

### 1. What causes transmission curtailment in the price-taker arrangement

**Proposed sentence:** "In the price-taker arrangement the transmission network has no export path, its only
demand is the three distribution networks' withdrawals, and its conventional units have no minimum output. Its
renewable output above the distribution networks' fixed interface schedules is therefore curtailed. This occurs in
76 of the 288 representative hours, mostly at night and in the evening, when high wholesale prices lead the
price-taking DSOs to move their demand elsewhere. 92 % of the curtailed energy falls in 2030 and 2035."

**The mechanism is a surplus over fixed schedules with no other sink.**
- **No other sink.** In the network data every TN load sits at a DN interface node. There is no interconnector,
  reference or export generator, and no TN storage at x = 0.
- **The surplus explains the curtailment.** In each curtailing block, available TN renewable output minus the fixed
  DN imports accounts for 93–99.5 % of the recorded curtailment. The rest is loss-sized: 0.2–1.2 MW per hour, against
  coordinated TN losses of 0.26–0.56 MW at the same hours.
- **Coordination absorbs it.** In the same 76 hours the coordinated DNs import the TN renewable output, to within
  0.69 MW, and coordinated TN curtailment is 0 in every block (positive part).
- **Timing.** 63 of the 76 surplus hours, holding 91 % of the surplus energy, are priced above the block's median
  wholesale price. This is the mis-timing Addendum 58 recorded ("pricing flexibility at π_t rather than λ_t").

**The Expert's other candidates do not cause it:**
- **Minimum output of conventional generators:** contradicted. P_min = 0 for all three units in all years.
  Conventional output is at zero in the curtailing hours; there is no must-run floor.
- **The no-reverse-flow rule:** contradicted; it acts the other way. It binds at 48 of the 76 hours (a DN held at
  P_int ≈ 0). There it stops a DSO from exporting, which would have added to the TN surplus. The 25 hours that the arm
  without any interface rule could not accept are all among these.
- **Network limits:** not supported, but not excluded directly.
  - At the surplus hours the arm's total DN import is at most 82.9 MW, against a smallest branch rating of 150 MVA.
  - A cost model built without using curtailment (conventional output = max(0, demand + losses − renewables))
    reproduces the arm's TSO cost to within −0.30 % to +0.001 % in all 12 blocks.
  - The arm's hourly TN flows and voltages were not serialized, so this check is indirect.

**Where it concentrates:**
- **The total** is 807,649.87 MWh net, day-weighted, equal to `report_v3` exactly.

  | grouping | GWh |
  |---|---|
  | 2025 / 2030 / 2035 | 64.7 / 319.5 / 423.5 |
  | Spring / Summer / Autumn / Winter | 357.6 / 195.5 / 116.5 / 138.0 |

- **2035 Spring and 2030 Spring** carry 40 %.
- **By hour:** 01–08 and 18–24, with hour 22 the largest. 14:00–17:00 carry nothing.
- **2025 Autumn** has no surplus hour and no curtailment.
- **The "10 of 12 blocks" rule** comes from W124 (`0c3451cf`). 2025 Winter was "outside the rule" on a cost-ratio test,
  but here it behaves like the other blocks (curtailment / surplus 0.98).

### 2. Why 0.25 MVA / 1 MWh at node 7 in 2025

The unit was chosen by a rule fixed before the run:
- **The rule:** frozen spec v15 (`0a188005`, 2026-09-19) selects "argmin over the 2025 ladders of F(x) = I(x) + Q(x);
  ties → lower I(x)".
- **The selection:** the A1a result (`d0dbd186`, 2026-09-20) reads "Best storage point n7_4h_e1 (0.25 MVA / 1.0 MWh at
  node 7)", F +56,150 € above x = 0.
- **The parts of that choice:**
  - **Size.** It is the smallest 4 h unit on the lattice. 4 h dominated 2 h at every node, and F rose with energy at
    every node.
  - **Node.** At equal investment, F favours the larger operating value. Node 7 beat node 9 by 4.8 k€ and node 5 by
    17.1 k€, with bars of 11.3 and 11.8 k€. The Phase A report itself calls the node differences "within resolution".
  - **Year.** 2025 was the ladder's year (y₁), not a separate decision. The year ladder that followed kept 2025 best
    under gross F.
- **Reasons that do not hold:**
  - A siting reason, "the binding interface / congestion relief at node 7", was proposed in Addendum 22 and struck in
    Addendum 23.
  - The spec-v15 prediction "the best node is the node whose interface binds" met its outcome nominally, but the Phase A
    report refuted its mechanism.
  - The manuscript itself gives no reason.
- **Under the baseline calibration** (Addendum 30), the same ladder ranks node 5's 0.25/1 closest: +53.6 k€, against
  +58.5 k€ for node 7 and +55.8 k€ for node 9, with a resolution of 14.8 k€.
  - Node 7 was kept as the reference.
  - I found no recorded ruling to keep it (searched: the brief and REVISION_CONTEXT for "53,607" and "n5 0.25/1").

### 3. Size of the search space: 1,704 plans

**Proposed sentence:** "With candidate nodes 5, 7 and 9 and installation years 2025, 2030 and 2035, these constraints
admit 1,704 investment plans besides the plan without storage (149, 522 and 1,033 by installation year)."

- **Rule** (the production lattice, `p515_s47_phase_b_record.Lattice`):
  - nodes {5, 7, 9}, with one common year y ∈ {2025, 2030, 2035};
  - per node, P ∈ 0.25ℤ≥0 MVA and E ∈ 0.5ℤ≥0 MWh, with P = 0 ⇔ E = 0, 2P ≤ E ≤ 4P (inclusive) and E ≤ 5 MWh;
  - admissible iff I(x) ≤ 1,000,000 € (present value at 2 %, cost-scenario weights 0.35 / 0.55 / 0.10).
- **Cross-checks:**
  - **The count is already on record.** It equals `domain_summary.n_storage_points = 1704` in the committed s47, s51
    and s53 campaign specs, label by label, with I(x) agreeing to within 10⁻⁶ €.
  - **The production check agrees.** The master feasibility check (`_check_candidate_first_stage_feasibility`), run on
    every grid point, accepts the same 1,704.
  - **Known plans fall on the correct side.** The 2030 returned plan (I = 992,268.11 €) and the reference unit
    (317,957.01 €) are inside. All seven over-budget plans on record are outside.
- **Splits:**

  | grouping | plans |
  |---|---|
  | one / two / three nodes | 183 / 873 / 648 |
  | each single node | 61 |

- **Boundary:**
  - **From above, the count is robust.** The closest plan to the budget is 6.8 k€ below it, so ≤, < and a 10⁻⁸
    tolerance all give 1,704. Raising the budget by 1 % changes nothing.
  - **From below, it is not.** 222 plans lie within 1 % of the budget, so lowering it by 1 % would give 1,482.
- **The Expert's description matches the code.** The sentence should also name the node and year sets, as above.
- **For information only (not the production rule):** allowing a different year per node would give 6,693.

### 4. Figure: "aging"

`Claude outputs/figures_s4_v2/fig_soh_reference_unit.pdf` (`6548969e`).
- **The change:** the legend reads "No aging". Compared with v1 operation by operation, only that string differs, at
  the same position and size.
- **All six figures were decoded from the PDFs:**
  - none of them has any other British spelling;
  - the SoC files and the flexibility figure need no change;
  - all six print in DejaVu Serif at 8.0 / 7.5 / 7.0 pt, as in `figures_s4_v1`.
- **Number format:** see decision 3.

### 5. 808.3 GWh and 0.7 GWh: confirmed on the positive-part convention

Source: `report_v3.json` (`bf6dbf72`), `curtailment_table`, day-weighted and undiscounted.

| | TN | DN | total |
|---|---|---|---|
| price-taker, best start, positive part | 807.667 | 0.623 | **808.290 → 808.3** |
| coordinated (x = 0, `d110bd1a`, k\* 181), positive part | 0.000 | 0.728 | **0.728 → 0.7** |

- **The two figures share a convention** (positive part), so 808.3 and 0.7 are comparable.
- **The TN figure beside them does not** (decision 1).

### 6. Appendix B.3, "approximately 0.02 k€": confirmed

The source is the W188 (a) slack inventory (`8ce38f88`), which gives the signed slack content in Q per cell.

| price | x = 0 | reference unit | effect on the operating-value difference |
|---|---|---|---|
| baseline | −13,012.57 € (`ref:7aa017f0`) | −13,032.47 € (`ref:bd504ecf`) | **19.90 €** |
| 2× | −13,012.57 € (`h_50dea31c`) | −13,032.47 € (`h_74eda68d`) | **19.90 €** |

- **Where it comes from:** all of the 19.90 € is the storage day-balance closure slack, which exists only with storage
  installed.
- **How it acts:** the slacks sit within the solver's relaxed bound, so the penalty paid is 0 €. Their signed content
  raises the unit's operating value by 19.90 € at both prices.
- **"Approximately 0.02 k€" therefore holds.**

## Not confirmed

- **The price-taker arm's hourly network state** (flows, voltages, dispatch) was not saved.
  - The hourly curtailment profile is inferred from the hourly surplus, using coordinated losses as a proxy. Its block
    totals equal the recorded curtailment exactly.
  - The network-limit hypothesis is therefore excluded only indirectly.
  - The other two price-taker starts were not profiled by hour.
- **Why node 7 was kept** after the baseline change: no recorded ruling was found.
- **The 3 × 3 configuration's plan count** was not computed. Its years differ (2025–2037), so its count would differ.
- **Not compiled:** there is no TeX installation here.
