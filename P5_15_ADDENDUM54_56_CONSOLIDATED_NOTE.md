# P5.15 Addenda 54–56: the uncoordinated arms are infeasible as defined; C\*'s creep is half a consensus gap; the re-run campaign needs three rulings

**Planner interim note, 2026-09-28.** For the External Expert and the Author; self-contained. Consolidated at the
author's instruction: one note after the benchmark and the re-run design review.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 49, 53–56.
- **Specs:** C\* extension v41 `fcea4b38` (predecessor v40 `9bc1779d`); benchmark v2 `bb659122` (predecessor v1 `46b6ba96`).
- **Objective convention on every value:** Q = `gross_operational_cost`, settlement excluded. t_sum =
  `t_tso_plus_t_dso_terminal` = Σ w·π·baseMVA·(p_DSO − p_TSO) over interface entries (the priced gap between the DSO
  and TSO copies of interface power); **Q_cc = Q + t_sum** is a first-order, consensus-consistent diagnostic, not a
  costed quantity.
- **Status:** nothing is running. The nine-cell re-run campaign is **held** until Decisions 1–3 are ruled.

## Decisions needed

**1. The uncoordinated arms are infeasible as Addendum 49 defines them (expert; option (c) is the author's).**
All six arm runs failed at a TSO block — passive (three starts) at **2035 Summer**, price-taker (three starts) at
**2025 Spring**, its first TSO block — every attempt "Converged to a point of local infeasibility". The same blocks solve
optimally at the coordinated schedule. Elastic diagnostics (W114, W115; production's builders, exact declared solves,
bitwise reproduction of the failures) give **one mechanism for both arms**:

| arm, block | hours | uncoordinated DN net interface P | coordinated | TN can absorb | binding family |
|---|---|---|---|---|---|
| passive, 2035 Summer | 11–16 | **export 8.2–22.8 MW** | import 166–209 MW | ≈ 3.8–4.0 MW (losses) | bus-7 P balance |
| price-taker, 2025 Spring | 4–7 | **export 13.8–45.3 MW** | import 18.9–20.8 MW | ≈ 4.0–4.3 MW (losses) | bus-4 P balance |

The TN has no load of its own (its only loads are the ADN interfaces) and its generators floor at 0 MW. When the
uncoordinated DNs net-export, the TN can dissipate only its losses, with voltages driven to their limits; everything
else — Q balance, voltage, thermal limits — carries zero slack, and relaxing Q alone does not restore feasibility.
Supply is never the problem (836 MW of TN headroom unused at the price-taker block).

| option | what | cost |
|---|---|---|
| **(a) report infeasibility as the result, with a sweep** | continue each arm past failing blocks and record, per block and hour, whether the TN can accept the DN schedule and by how much it cannot; the claim becomes "without coordination the TN cannot accept the DNs' exchange in *n* of 12 blocks (*h* hours)" | code (continue-on-infeasible, report-only) + ≈ 10 min run; benchmark spec v3 |
| (b) give the arm a recourse | the uncoordinated DN curtails the export the TN refuses (to the TN's absorption limit), valued at the curtailment penalty — a cost comparison becomes possible | a change to the Addendum 49 arm definition (expert) + the same run |
| (c) give the TN a sink | an external-grid export at a price | an instance change (author); re-certification of everything |

**Recommendation: (a) now, (b) if a cost figure is wanted.** (a) needs no formulation change and is itself a strong,
honest coordination result; (b) is the minimal definition under which "uncoordinated costs more by X €" exists.

Beside it, from the λ look (usable; three sources agree): λ_t differs from π_t by more than 0.05 €/MWh on **848 of 864**
rows, and coordination is proper (λ ≠ π and flexibility cheaper than π) in **807 of 864** hours. The common-Q gate
**passed bitwise** at the settled Q181.

**2. C\*: the flat direction is real, but half the gross creep is a consensus gap (expert).**
The 100-cycle diagnostic extension (W110) scored **mixed** under the frozen predictions: P_a (creep in TSO generation)
held, P_b (ESS schedules still moving) held, **P_c (rising residual = ESS channel) failed** — the rising residual is
**pf**, 0.13 → 0.30 — and P_d (rate within ×2 of −265 €/cycle) held at −153.5 €/cycle. No refutation trigger fired.
The pre-registered gap diagnostic (W112) then showed:

- **57 % of the gross descent over cycles 188–287 is the priced interface gap opening** (t_sum 6,909 → 15,616 €);
  pre-registered verdict: *gross creep is the wrong quantity*.
- **Consensus-consistent creep dQ_cc is flat at −66 €/cycle** in all four quarters (held-out fit error 1.1 %); the
  apparent decay of gross Q was the gap's growth slowing.
- The dual update is **exactly standard** (‖Δy‖ = ρ‖r‖ within 1e-12 every cycle): a genuinely slow mode with
  substantial storage, not a defect. On x = 0 and the unit cell the pf residual decays with their oscillation; on C\*
  it grows.
- **Settling closes the gap on the references** — x = 0 23,802 → 143 €, unit 34,057 → 663 € — but **old certificates
  carry priced gaps of the order of the settling slack**: over 174 certified SRP1 cells, median |t_sum| 22.6 k€
  (IQR 7.4–36.7 k), 97 above 20.8 k€.

| option | what |
|---|---|
| **(a) report-only** (recommended for the campaign) | record t_sum per cycle; every restated difference reported in gross and in Q_cc terms; verdicts on gross only |
| (b) a gap clause in certification | certify only when \|t_sum\| ≤ a fraction of τ, in addition to the settling rule — the settled references satisfy it, old certificates do not |
| (c) a creep branch on gross Q | **not recommended**: it would certify a bias |

Also for the expert: the monotone branch certifies a drifting cell with ≈ 3 τ left whenever the decay half-life
exceeds its 44-cycle window (C\*'s is ≈ 102); and P_MAX = 22 / L_MONO = 44 were computed before certification, while
the post-certification periods measured on SRP1 are 29–30 (2× = 60). Recommendation: keep both frozen; report the
monotone branch at L = 60 beside.

**Manuscript statement for C\*, proposed:** "uncertified at the 287-cycle cap; gross objective 650,891,817 € at the cap,
still descending ≈ 111 €/cycle; consensus-consistent descent ≈ 66 €/cycle; interface-consensus gap at the cap 15.6 k€;
value − I = −749 k€, unaffected in sign."

**3. The nine-cell re-run campaign (Addendum 54 Ruling 2): three rulings before its spec (expert).** The campaign is
**nine** cells: the six Phase B neighbours, the 2030 cell, and the F2 certificate's challenger *and* incumbent (the
difference needs both). ≈ 14–17 h on the Mac. The six Phase B cells are identical 0.25 MVA / 0.5 MWh units (they differ
by node and year), with pre-tail gates expected to hold.

- **(a) The 2030 cell cannot be gated.** At its run the ESS parameters differed from today's (Planner-verified by
  `git diff d0dbd186 HEAD -- data/SRP1/SharedESS/SRP1_ESS_Params.json`): calibration **C3** (end-of-life retention
  0.50), minimum SoH 0.50, no calendar retention, and **maximum energy-to-power 10 (now 4)**. A re-run differs from
  cycle 1. Options: a first C2 evaluation without a bitwise gate, together with its 2035 partner (`dab6a8a2`) so the
  ladder is consistent (+1 cell); pinning the old parameters (a second configuration, against Ruling 2's purpose);
  or dropping the year ladder from the baseline tables. **Recommendation: first C2 evaluation of both years.**
- **(b) The regime after the first residual pass.** The references settled under holds (AA off, tail on, ρ frozen,
  even on a residual lapse); production without holds is non-latching. **Recommendation: the same holds**, for parity
  — it matters on the F2 cells, which sit at pf_primal 0.69–0.74, near tolerance.
- **(c) The F2 pair.** τ = 4,539 € would apply unchanged at the doubled flexibility price; the pre-tail gate has never
  been demonstrated on the multiplier path; and the challenger was still descending ≈ 800 €/cycle when certified, with
  C\*-like creep rated ≥ 0.5 likely. Its "no determinate improvement" margin (6,338 €) is the tightest in the triage.
  **Recommendation: run the F2 challenger first**, with the uncertified reporting form frozen in advance.

Design otherwise (within the adopted criterion): gate bitwise through each cell's first residual pass against its
original record; the stop rule's decision point from that pass; cap at the old certification cycle + 100; the C\*
extension's captures plus per-cycle t_sum; restated differences against the **settled** x = 0 (Q181) in gross and Q_cc.

## Blocked on the author

- **Machine time** for the nine-cell campaign (≈ 14–17 h), once the rulings are in.
- Option 1(c) (a TN sink) is an instance change and would be the author's.

## Changed

| what | commits |
|---|---|
| gate-result writer; CLAUDE.md rules (fix at the writer; validate fits out of sample; exclude the run's own artefacts) | `ed5ba42a`, `88a8cd1d`, `2181566c`, `42481c30` |
| SRP1 reference continuations; three-reference summary | `e4b5c992`, `3ee62863`, `82dcebb7`, `62bdeafe` |
| C\* extension: v40 refused by a self-collision in its own pre-run check (zero solves), fixed as v41 | `304bb120`, `39236209`, `477dba3e`, `3090d346` |
| C\* extension run and transfer shares | `c2afe100` |
| priced-gap diagnostics; triage list (33 + 9 cells) joined to terminal gaps | `cbdd78ec`, `e00c7d4f`, `34753387` |
| TSO curtailment look (the RES bound slack; Addendum 56) | `f6e3533f` |
| benchmark re-pointed to the settled models; spec v2 | `64a1a182`, `1ba8ae36`, `f5045cca`, `59c3b995` |
| benchmark stages; arm infeasibility diagnostics | `2ee7a124`, `209f4829`, `b0ed4d14`, `568284d1` |

The benchmark's `report` stage ran with the arms missing and consumed its write-once name under v2, so any further
benchmark run needs spec v3 regardless of Decision 1.

## Found

### Predictions against outcomes

| prediction | outcome |
|---|---|
| H_ess-flat P_a: creep in TSO generation cost | **held** (signed share 1.42) |
| P_b: ESS schedules not decaying | **held** (0.85) |
| P_c: the rising residual is the ESS channel | **failed** — pf rises, ess falls (the ess channel measures agreement among ESS copies; interface lag shows in pf) |
| P_d: rate within ×2 of −265 €/cycle | **held** (−153.5) |
| W112 pre-registered: gap share < 20 % → minor; ≥ 40 % → wrong quantity | **wrong quantity** (57 %) |
| Common-Q gate reproduces Q181 | **held, bitwise** |
| λ look: recorded prediction vs Advisor's "inside the band" | **not scoreable** — no arm completed |
| Settling datum (Addendum 55): the consensus item tightens with settling | 0.058 (cycle 132) → 0.026 (cycle 181) |

### The C\* extension (W110, `c2afe100`)

Replay bitwise through 187; cycles 188–287 run under the same holds; all gates pass; 2.25 h. Q_287 − Q_187 = −15,351;
quarter means −193.6 / −166.2 / −137.9 / −116.3 €/cycle; the settling rule never certifies (two turning points).
Node-7 storage movement and TSO-generation changes co-move at event level (re-timing events coincide with generation
bumps); their 0.93 correlation is mostly a shared trend.

### The benchmark (W113–W115)

- λ look usable; common-Q gate PASS_BITWISE; the 24-solve coupling check: fixed − penalty = −9.88 € of TN cost.
- All six arms infeasible, as in Decision 1; tie-breakers not run; no claim computed.

## Not confirmed

- **Global** infeasibility of the arms: the elastic results are local optima; that no TN operating point exists is
  strongly indicated (the TN would need a sink), not proven.
- Why the price-taker DNs export in hours 4–7 (ESS discharge or the price profile); DSO-internal dispatch not inspected.
- How many blocks each arm fails in: the arms stop at the first failing block (Decision 1(a) measures it).
- That the TN has no load of its own is the Worker's reading of the arm build, consistent with every elastic solution;
  not checked against the case file by the Planner.
- The F2 pre-tail gate and the Phase B common-mode prediction — to be measured by the campaign.
