# P5.15 Addenda 40–42: the α row is a two-mechanism cost-of-commitment curve; the premium destroys 97 GWh at α = 1.0. Stop for review

**Planner report, 2026-09-25.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 40, 41, 42.
- **Frozen specs:** v24 `3ac8c185` (predecessor v23 `39a07fd8`), v25 `407a4b33` (predecessor v24). Row campaign spec `70965374`. Every prediction was recorded before its run.
- **Objective convention on every table:** Q = certified `gross_operational_cost`, settlement excluded, **unpolished**. C = Q − row-18 charge. Baseline flexibility price throughout (never the m = 2 variant).
- **Status:** nothing is running. Stopped for review, as Addendum 40 ordered.

## 1. Verdict

1. **The α row is complete — five certified cells at x = 0 plus the unit at α = 0.5.** All four adjacent envelope inequalities hold in both forms; C is monotone increasing, P monotone decreasing; every cell settled at ≤ 6.6 % of its threshold.
2. **Two mechanisms, not one threshold.** 82 % of all deviation volume is removed by α = 0.1. The remainder is physical: at α = 1.0 the premium destroys **97,148 MWh** of renewable energy, worth **7.72 M€** at scenario prices.
3. **The coordinated α\* that Addendum 40 sought does not exist in range.** This was established before the run and the purpose was restated in the frozen spec rather than after the fact.
4. **The storage still does not pay at 2×2**: value − I = **−50,408**, determinate at 2.73×.
5. **The F2 certificate holds** — the 2030 two-node plan is mesh-locally optimal at the unit mesh, with one unresolved indeterminate at *identical* investment.
6. **Addendum 41 selects branch 1: no re-baseline** — and the rule's own premise was refuted.
7. **The initialisation fix is bitwise-clean on SRP1 and did NOT determinately change the storage's value.**

## 2. The α row (`e46c0bda`, `26a02bd3`, `e3095aa2`, `9e88d7bd`, `59fc409f`)

| α | Q | charge | **C = Q − charge** | **P(α)** | bar | rule-10 | cycles |
|---|---|---|---|---|---|---|---|
| 0 | 811,373,572.69 | 0 | 811,373,572.69 | 324,068,720.72 † | 5,683.03 | 0.0071 | 55 |
| 0.1 | 824,438,966.48 | 5,780,306.27 | 818,658,660.21 | 57,803,062.71 | 1,885.91 | 0.0126 | 68 |
| 0.25 | 831,588,203.00 | 10,941,967.08 | 820,646,235.92 | 43,767,868.33 | 2,743.18 | 0.0208 | 70 |
| 0.5 | 841,964,212.56 | 20,065,751.21 | 821,898,461.35 | 40,131,502.42 | 7,898.83 | 0.0292 | 74 |
| 1.0 | 860,741,268.55 | 27,767,759.67 | 832,973,508.88 | 27,767,759.67 | 6,954.88 | 0.0661 | 73 |

† P(0) is the committed **post-hoc reconstruction** — the one cell where P cannot be charge/α. Formula and inputs are preserved in `p515_s53_alpha_row_recompute.py`; recomputed bitwise equal.

**Envelope checks — the self-consistency test.** For α₁ < α₂: (α₂−α₁)P(α₂) ≤ ΔQ ≤ (α₂−α₁)P(α₁).

| interval | lower | ΔQ | upper | verdict |
|---|---|---|---|---|
| (0, 0.1) | 5,780,306 | 13,065,394 | 32,406,872 | within |
| (0.1, 0.25) | 6,565,180 | 7,149,237 | 8,670,459 | within |
| (0.25, 0.5) | 10,032,876 | 10,376,010 | 10,941,967 | within |
| (0.5, 1.0) | 13,883,880 | 18,777,056 | 20,065,751 | within |

All four hold in the **V form** (V = Q + voltage pin, the solver's actual objective — the verdict on record) and in the Q form. C's steps are determinate at 962×, 429×, 118× and 746× their summed bars.

**Why this matters.** Q is *not* a system-cost curve: it mixes resource cost with a transfer to an unmodelled counterparty. Q rises 49.4 M from α = 0 to α = 1.0, but C rises only **21.6 M (2.66 %)**. Reporting Q as a cost curve would have overstated the price of commitment by more than a factor of two. The envelope check then confirms the cells are comparable optima of a common cost function — the concern is *tested*, not argued.

**Not independent evidence.** With C = Q − αP the Q-form envelope is algebraically equivalent to α₁ ≤ −dC/dP ≤ α₂, so a "marginal cost of commitment" table restates the envelope rather than confirming it. What is informative is each ratio's position within its interval: 27.4 %, 27.7 %, 37.7 %, **79.2 %**.

## 3. The two mechanisms

| α | E\|d\| (weighted) | market share of Σωd² | DSO curtailed RES | priced at π_s |
|---|---|---|---|---|
| 0 | 2,381,195 | **0.965** | 630 MWh | 46,647 |
| 0.1 | 526,910 | 0.527 | 694 MWh | 56,844 |
| 0.25 | 441,946 | 0.183 | 1,033 MWh | 57,078 |
| 0.5 | 436,268 | 0.153 | 1,720 MWh | 64,668 |
| 1.0 | 321,769 | 0.186 | **97,148 MWh** | **7,718,524** |

1. **Market-arbitrage suppression is cheap and essentially complete by α = 0.1.** The market share of Σωd² falls 96.5 % → 52.7 % → 18.3 %. This confirms Addendum 39's 2-cycle figure of 91 % on certified data. The instance's own α_arb distribution (median 0.042, p90 0.189) predicts exactly this.
2. **Physical waste dominates the top of the range.** DSO curtailment rises **56×** between α = 0.5 and 1.0. At α = 1.0 the curtailed energy is worth 7.72 M€ — **about a third of the entire 21.6 M cost of commitment is wasted renewables.** 96,413 MWh (99.24 %) is classified below-capability on the primal indicator at k = 2 (see §8b), with the row-18 condition (d ≤ 0 and π_s < α·π̄) met on 92,677 MWh.
3. **Σωd² is not monotone** — it rises from α = 0.5 to 1.0 while E|d| falls, and peak |d| rises 34.51 → 36.05 MW. A high linear premium makes deviation rarer but larger. Readers expecting monotone dispersion will be surprised; it should be stated.

## 4. Corrections to earlier claims — all of them mine

| claim | correction |
|---|---|
| "R = 0.937 confirmed at 0.942" licenses the SRP1-with-caveat fallback | R restates to **1.0313**. The recorded ±0.01 band is FALSIFIED — but the ratio's own resolution is **0.152**, so R − 0.937 = 0.094 is **0.62× resolution: INDETERMINATE**. The band was 15× tighter than the measurement can resolve. The mean-profile argument is **not** refuted. |
| The initialisation fix changed the storage's value | Value moved +23,228, which is 0.72× the **four-cell** bar-sum (32,062) — **INDETERMINATE**. The σ_Q figure of 1.26× is not the governing error: comparing two value estimates is a difference of two differences. |
| The reactive leg carries 51.3 % of the charge | Certified Q-leg share is **zero to solver tolerance** (\|share\| ≤ 3e-6). The 2-cycle smoke figure does not survive certification. W-R9 falsified. |
| The curtailment mechanism is "dual-established" | **Withdrawn.** \|dual\| × slack is constant (barrier complementarity), so dual magnitude reports slack size, not economic bindingness. The **primal** indicator is the sound basis. |
| Pyomo's presolve would silently substitute columns | `NLWriter.__call__` sets `linear_presolve = False` on production's path. The hazard is visible as added rows, not silent. Advisor and Planner both read the config default without checking the call-site override. |

**Recorded predictions falsified:** W-R10 (R band), W-R9 (Q-leg share), W-R3 (P(0) at 1.5–4× P(0.5); actual 8.1×), unit value within 10,000 (missed by 23,228), G9 and G13 of the capture smoke (both limits mine, both wrong), and the `.nl` negative control P3.

## 5. Addendum 41 — curtailment: branch 1, and the rule's premise refuted

- **The 1 €/MWh tie-breaker is not in force on any certified result.** The penalty is zeroed for TSO and DSO in the ADMM subproblems; the constant applies only to the initialisation build and the uncoordinated benchmark. Branch 1's wording describes a mechanism acting on nothing we report, and **branch 2 would have double-counted** — lost injection already forces a compensating import settled at the scenario price.
- **Reachable (TSO-side) curtailment is below resolution everywhere**, C/bar ≤ 0.037; first-order effect on storage value 331.5 against 15,511.
- **On SRP1 there is essentially no surplus curtailment**: all 364 curtailed generator-hours are inverter-capability-bound, with a voltage bound active in the same network-hour for 99.6 % of the energy.
- **The DSO7 transformer binds on *import*** (100 MVA, correct per case file; the 200 MVA premise was DSO5's). So the transformer is a **consequence** of lost RES, not a cause: reactive support → active power surrendered → import rises → transformer saturates. An import-saturated transformer also cannot be relieved by HV-side storage, so unreachability is established by mechanism, not only by convention.

## 6. The initialisation fix (merged `3343c4da`; gate `d5a00bd9`)

Row 18 was inconsistently active at the ADMM initialisation solve, where the interface import is unpriced. The first fix (zeroing the α Param) was **rejected on review**: it leaves a zero-cost ray (d⁺+t, d⁻+t) whose perturbed KKT system has **no solution** for any μ > 0, so the central path does not exist. IPOPT would still most likely *succeed*, at arbitrary d — the objection is determinism and confounding, not failure.

**Adopted:** deactivate the defining rows and fix the pair at 0 during initialisation; at activation set the minimal split from the initialisation solution, unfix, then activate.

**Evidence:** `sha256(A) == sha256(B)` on **12/12** DSO blocks — the α = 0 build and the fixed-and-deactivated α = 0.5 build write byte-identical `.nl` files, so initialisation is bitwise identical across the whole row. Confirmed at runtime: `init identity x0 bitwise_equal=True` on all three pairs. **SRP1 two-cycle bitwise gate PASSED**: 153/153 solves with `GUARD.verify(153) == []`, counter 36/36 with 0 acting, 0 diffs vs the committed C\* reference, tripwire 0.

## 7. Addendum 42 — measurements

- **(2) NL-variable audit (`0634fd3b`).** The model-vs-`.nl` accounting closes **exactly** on every block of SRP1 and 2×2 in both phases. **No hazard of the row-18 class found** (H0 = H1 = H2 = 0), with a built-in positive control that fires. The Addendum's premise needs correcting: Pyomo emits a variable only if it appears in an active constraint or objective, so a free *unreferenced* variable never reaches IPOPT. The operative hazard is a column that **is** written, is costless, and lies in an unconstrained direction. **Gap:** the audit ran only at x = 0; at positive capacity the zero-capacity gate unfixes every shared-ESS copy. Queued.
- **(i) Static audit (`w67`).** **The memory refactor is not feasible as-is.** The row set itself differs between blocks — four `Constraint.Skip` sites depend on RES time series, proven from the committed r2 probe (identical columns, different row counts by season, deltas exactly 3 × 2 × 2 × n_PV × 4 h). Three further families are baked in as Python floats: RES availability, energy/flexibility prices, and the `effective_scale` divisor. **Every blocker varies by day, so the per-network-year layout removes none of them.** `symbolic_solver_labels` is **False** on every production path, traced through all six steps.
  - **Correction to the brief:** `objective_scale` is *not* a mutable Param. The three `admm_*scale` Params are immutable and the objective never references them.
- **(1), (3), positive-capacity extension, D1** — queued for the idle window, well inside the "before the 3×3 pair" deadline.

## 8. Open anomalies

1. **Hull polish.** Polished gross exceeds certified on **all six** cells, and the block-sum sign flag is **False** on x0 at α = 0.5. Addendum 38 recorded "polish Δ ≤ 0 by construction" as restored at 1×1; it does not hold at 2×2 across 80 blocks. Post-certification and excluded from Q, so no headline number is affected — but under the polished convention value = 259,195 and R = 0.9991. On 3 of 6 cells the polish moves gross by more than the cell's own bar.
2. **Shared-file overwrite.** Pair 3's launch overwrote `campaign_heartbeat.json`, destroying pair 2's end state. Timeline survives in `pair_2_results.wave_info`. The heartbeat should be per-pair.
3. **`hull_polish_full.gate.pass` is the string `"True"`** on five of six cells, boolean on one. A strict `is True` reader would silently misread them as failures.
4. **`row18_gate_addendum.json`** records six checks as `false` because the committed S51G gate writes it while the counter is installed. Non-gating; both gating evaluations returned 21/21. A fix was attempted and abandoned (the arm-named eval directory cannot be redirected); caveat recorded beside the artifact.

## 8a. A new hazard: the compared cells are not scaling-matched (`5fc92a8d`)

**Reported first at 1.14× resolution; corrected by W74 to 0.91× — now INDETERMINATE. The mismatch is real; its estimated effect on value is not established.**

The unit cell's 20 terminal TSO solves used an IPOPT objective scaling factor of **0.001**, against **0.0047–0.0071** on every x = 0 cell; final μ was **4.545e-8** against a median of **1.845e-6**, i.e. barrier offsets about **40× smaller**. The probable cause is structural: the unit's TSO problem carries **388 more complementarity pairs** (the storage variables), and IPOPT's gradient-based scaling responds to the changed problem.

Interior-point methods approach the optimum from inside, so each Q carries an upward barrier-gap bias of order (pairs × μ / objective scale). Those biases differ between the two cells being subtracted:

| cell | barrier-gap estimate (TSO blocks) |
|---|---|
| x0 at α = 0.5 | 22,450 |
| unit at α = 0.5 | 3,660 |
| **difference (W73, first estimate)** | 18,790 — 1.14× resolution |
| **difference (W74, CORRECTED)** | **15,079 — 0.91× resolution: INDETERMINATE** |

**Correction (W74, `88d82d9d`).** W73's parser took the **last** solve in each IPOPT log — the hull polish, not the terminal ADMM solve it assumed. The polish adds 360 complementarity pairs per TSO block and 120 per DSO block, and μ differs on 5–9 of 20 TSO blocks. On the terminal ADMM solves the difference is **15,079**, i.e. **0.91× the two-cell resolution** (16,551.50) — so by this programme's own rule the estimated effect is **below its own error bar and indeterminate**.

**What survives the correction.** The scaling mismatch itself is confirmed: the objective scaling factor is **identical on both solves in all 480 blocks**, so 0.001 for the unit against 0.0047–0.0071 for x = 0 stands, as does the μ difference. What is now indeterminate is the **estimated effect on value**, not the mismatch.

**What it would mean if it holds.** `value = Q(0) − Q(unit)` inherits the *difference* of the two biases, so on the corrected estimate the measured value would overstate the true value by about 15,079 — putting it near 252,500 and R near 0.973 — but at 0.91× resolution that is **indeterminate**, making it a candidate explanation rather than an established one. **It is therefore a coherent alternative explanation for the value increase attributed in §4 to the initialisation fix**, and it is consistent with that change already being indeterminate at 0.72× the four-cell bar.

**What survives regardless.** The sign and magnitude of `value` (267,549 ± 15,079 remains overwhelmingly positive), and **`value − I` stays negative** — the storage does not pay — at roughly 1.7–3.7× resolution rather than 2.73×.

**What it casts doubt on.** Every `value = Q(0) − Q(x)` in this programme subtracts a no-storage cell from a storage cell, which differ in variable count and so potentially in scaling. This may be **systematic across the programme**, not specific to this pair.

**Scope of the claim, stated honestly.** It is an order-of-magnitude estimate of the **augmented TSO objective**, not a measured change in Q; DSO blocks were not examined; and the link from the IPOPT logs to the terminal cycle rests on a last-match rule over logs that sit outside any campaign manifest (hashed at read time into the W73 manifest).

**Feasibility of the ordered re-measurement (W74).** Addendum 44 ruling 5 specifies a fixed-configuration polish on the existing certified cells with **no new ADMM runs**. That **cannot be executed as worded**: `certified_models.pkl` is absent in all six cells (`persist_certified_models: False`, per the W48 memory ruling). What survives is two **cycle-7** FrozenSMOPF snapshots per cell — not terminal — and the ESSO terminal models, which the polish never re-solves. Two options, both for the author:

- **(a)** re-run the cells with persistence, gated by a bitwise-reproduction check — but **no full-run bitwise reproduction has ever been demonstrated on the 2×2 instance** (searched: commit log, the `s52_pilot_repro_nopersist` artifacts, the init-identity check — all 2-cycle or initialisation-only), so that gate may not pass;
- **(b)** polish rebuilt models under the committed hull from a start that is **not** the certified point — cheap, but a different experiment from the one ruled.

With the corrected 0.91×, option (a)'s cost may no longer be proportionate to the question.

**Pinned-configuration hazard, if the polish does run.** A user `obj_scaling_factor` is **multiplied** with IPOPT's gradient-based factor, so pinning it alone does not pin the effective scale — `nlp_scaling_method` must also be none or user-scaling. `acceptable_tol` 1e-4 with `acceptable_iter` 5 can terminate a `tol` 1e-8 solve at "acceptable level", and some DSO polish solves already hit the hard-coded `max_iter` 500 and succeed only on retry, which a tighter tolerance will worsen. Solve count: 80 blocks × 6 cells = **480 primary**, ceiling 1,440 with retries; the α row itself used 562.


## 8b. Curtailment classification, corrected (`5fc92a8d`)

A third class `at_availability` (c ≤ k · TOL_MW) separates entries that merely sit at availability within the barrier offset from genuine curtailment, applied identically to TSO and DSO. **k = 2** is derived from the barrier geometry, not from the observations: the model's bound is `pg ≤ pg_avail + TOL`, two barrier terms act on c, stationarity gives 1/(r+1) + 1/r = ρ, and IPOPT's termination forces μ ≤ T, yielding r = (1+√5)/2 = 1.618, so k = ⌈2⌉. Sensitivity is reported at k = 1, 2, 3, 5.

- **The headline total is classification-independent**: DSO curtailment at α = 1.0 remains **97,148.48 MWh**.
- The below-capability subset moves by **−39 entries / −3.51 MWh (−0.0036 %)** at α = 1.0; at k = 5 the change is −54 entries / −7.22 MWh.
- The 60 TSO entries previously relabelled `capability_bound` are **at_availability**: c = 1.30 × TOL_MW. Their κ = 0.64, φ = 1 "dual signature" is exactly what the barrier predicts for a unit at availability (κ matches the geometric prediction within 1.8e-3).
- The unit cell's near-absence of TSO curtailment (0.35 MWh vs ~5 MWh) is **solver numerics, not storage displacing curtailment** — the same scaling difference documented in §8a.
- **Open:** the recorded `capability_bound` test uses the squared row's slack ≤ 1e-6 p.u.², which is size-dependent in MVA; 30 entries (25.16 MWh) at α = 1.0 sit at the S-limit under a size-uniform test. Reported alongside, not substituted.

## 9. Decisions requested

1. **α = 2.0.** The grid brackets the arbitrage mechanism well but catches the physical-waste mechanism only as it takes off — 1,720 → 97,148 MWh between α = 0.5 and 1.0, measured at **one point, at the grid's edge, with nothing beyond it**. One cell (~4 h) would establish whether curtailment saturates or keeps growing. Recommended.
2. **DN-side curtailment**, material (C/bar 4.7–9.1) but unreachable from the HV busbar: the recorded rule names no branch for it.
3. **Art. 13 framing.** Does active power surrendered for reactive support at the inverter's S limit count as curtailment? It is **100 % of SRP1's volume**. Causally the loss is network-driven (voltage bound active in 99.6 % of the network-hours), but the mechanism is reactive support, not a curtailment instruction. Legal question; citation to be verified by the author.
4. **3×3 route confirmation**, and whether the 5×5 confirmation pair runs on this machine — the latter gates the memory refactor, which §7 shows is infeasible without converting four families and replacing the data-dependent `Skip`.
5. **The scaling-matched re-measurement of §8a.** Until it is done, every `value` figure in this programme carries an unquantified solver-scaling component of order its own error bar. I recommend it takes priority over the 3×3 pair.
6. **σ_Q = 18,449.66** is SRP1/C3-era and was never re-derived for the 2×2 instance. It is the stricter test here (bar-sum 16,552), so conclusions are unaffected, but the provenance should be stated in the manuscript.

## 10. The 3×3 prediction, recorded

For the 3×3 pair at α = 0.5, recorded before any 3×3 run:

- **R(3×3) ∈ [0.93, 1.09]**, centred on the 2×2 result of 1.031, on the mean-profile argument that transferred at 2×2. Stated with the resolution caveat: the ratio's resolution at 2×2 was 0.152, so a 3×3 point inside that band confirms nothing sharper than "not distinguishable".
- **value − I remains negative** at 3×3 under the baseline flexibility price, determinate at ≥ 2× the four-cell bar.
- **Cycle time ≈ 9 min**, ~16 h per evaluation, from the measured scaling (SRP1 32 s at 48×1; 2×2 ~190 s at 80×4; paper 42.3 min at 80×25).
- **Production's `sample(n, random_state=seed)` yields the prefix** [1,2,3]×[1,2,3] for a 3-scenario case, verified across 40 market and 10,620 operation seeds. Recommendation: **accept the prefix and record R = 0.9331** rather than change production's scenario draw for a 0.6 % difference — the author's call.

## 11. Evidence

| item | commit |
|---|---|
| specs v24, v25 | `fc5a7a52`, `5ad77b27` |
| capture smoke r1 (FAIL, kept), r2 (PASS 17/17) | `776c2344`, `ca22be17` |
| α row pairs 1, 2, 3 | `e46c0bda`, `26a02bd3`, `e3095aa2` |
| frozen row analysis; recomputation + formulas | `9e88d7bd`; `59fc409f` |
| initialisation fix merged; SRP1 bitwise gate PASS | `3343c4da`; `d5a00bd9` |
| F2 certificate | `97377638` |
| curtailment audit; DN7 transformer rating | `51bd342d`; `270278b9` |
| Addendum 42(2) NL-variable audit | `0634fd3b` |
| forward corrections to two "ZERO SOLVES" claims | `ada960c2`, `425adc5d` |

Every zero-solve claim is backed by an armed `SolveProfileGuard` verified at exactly 0, except two early gate-refusal commits where the claim is **inferred** — both carry committed forward corrections saying so.
