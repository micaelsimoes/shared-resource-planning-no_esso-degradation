# P5.15 Addendum 52: settling-criterion review — the 10-cycle candidate fails, and the SRP1 reference was not settled at certification

**Planner interim note, 2026-09-27.** For the External Expert and the Author; self-contained. This note is
not the ordered stopping point, which is the benchmark report. It is sent early because it changes a
statement Addendum 52 carries into the manuscript.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 52 ("Settling criterion → Advisor review … no compute;
  runs in parallel with the benchmark").
- **Objective convention on every value:** Q = certified `gross_operational_cost`, settlement excluded.
  V = Q(x = 0) − Q(unit); R = V_3×3 / V_SRP1; V_SRP1 = R_ref = 259,375.33.
- **Evidence:** records only, all committed — 3×3 pair `47a54c89`; 3×3 x = 0 continuation `4689475e`; SRP1
  tight-tail re-certification `0a4bf784` (`data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_recert/evals/{5cfe69a615ae3708_x0, ca8927e75d628bd1_n7_4h_e1, 96c5aa50cc229cc1_c_star}/per_cycle_record.jsonl`).
  Two read-only Advisor reviews; every figure below was re-read from the records by the Planner. No compute.
- **Status:** the order continues — W100 (gate-result writer), then the SRP1 benchmark, which does not depend on
  this note (Decision 4).

**Definitions used throughout.** k0 = first cycle of the unbroken Boyd-pass run that ends at certification.
E_W(k) = max Q − min Q over cycles k−W+1 … k. "Bar" = max |ΔQ| over the 10-cycle certification window (as in
all reports to date).

## Decisions needed

**1. The certification rule for multi-scenario cells (expert).** Addendum 52's candidate — 10-cycle range of Q
≤ τ — **fails on the other cell of the pair:** 3×3 node-7 at its certification cycle 69 has **E10 = 6,639 €**
while its last steps were **−362, −862, −1,310, −1,688 (growing)**. It would have certified node-7 mid-descent,
exactly as the residual rule did. E20(69) = 37,531 refuses. A 10-cycle window is 0.55 of the observed 18.3-cycle
period, so its range is phase-dependent.

Options:
- **(a) Advisor's refined criterion (recommended):** in addition to the Boyd residuals over the window, require
  **≥ 2 sign changes of ΔQ after k0 + 3** (a measured period P̂; no certification on a monotone stretch);
  window W = max(20, ⌈1.1·P̂⌉); **successive half-swing amplitudes non-increasing**; E_W(k) ≤ τ. Certified
  value Q_k with band = the last swing's [min, max]; remaining excursion reported as c·A_last/(1 − c), c the
  measured ratio of the last two half-swings, checked out of sample on the next turning point. **No fit enters
  the decision.** Changes the convergence test only; the optimisation problem is untouched.
- (b) Addendum 52's candidate with W = 20 and E_20 ≤ τ. Simpler; still admits a slow mode (period ≫ 20) passing
  near an extremum.
- **Rejected by the review:** a geometric-decay clause on monotone windows. On 3×3 x = 0 at cycle 77 the last five
  steps decay (r̂ = 0.70) and predict ≤ 1,624 € further fall; **the objective then rose 7,824 €.** Decaying
  steps do not distinguish convergence from the approach to a turning point.

**Threshold.** To first order δR = [(e_x0^3×3 − e_u^3×3) − R·(e_x0^SRP1 − e_u^SRP1)] / V_SRP1, with e the
per-cell settling error. With |e| ≤ τ on all four cells, linear sum (settling errors are one-signed, not
independent): **τ = δ_R · V_SRP1 / (2(1 + R))**; at R = 1, **τ = δ_R · V_SRP1 / 4** →
**9,078 € for δ_R = 0.14; 4,539 € for δ_R = 0.07.** *The expert must state the δ_R the manuscript claim needs:*
0.14 cannot test the recorded prediction (|R − 0.9331| = 0.02) or the band [0.93, 1.09].

Cost: replayed on the recorded cycles 63–88, 3×3 x = 0 **does not certify under (a) by cycle 88** (E22(88) =
28,130). *Projection beyond the records*, assuming the observed contraction continues: ≈ cycle 99 at τ = 9.1 k€,
≈ 108 at 4.5 k€ — **+27 to +36 cycles, ≈ +3–4 h per 3×3 cell** at 376 s/cycle; under (b) ≈ cycle 92. 5×5 not
estimated (no per-cycle wall measured).

**2. The SRP1 reference was not settled at certification — R_ref and every R carry an unmeasured band
(expert; the continuation is an author call on machine time).**

| SRP1 tight-tail cell | k0 | certified | last three steps | E10 at cert. | earlier troughs, vs Q_cert |
|---|---|---|---|---|---|
| x = 0 | 123 | 132 | −1,746, −758, **+154** | 31,150 | +37,773 (86), +39,015 (100), +42,287 (118) |
| unit n7_4h_e1 | 103 | 112 | **+884, +2,385, +3,640** | 29,156 | +36,966 (70), +46,382 (86), +42,982 (100) |
| c_star | 78 | 87 | −5,389, −3,509, −1,822 | 74,193 | — (still descending) |

At k0 on every cell — the cycle at which Anderson acceleration switches off — a transient begins that takes Q
**38–53 k€ below every earlier trough**. x = 0 was certified at the trough; the unit cell 6,909 € above its trough
**on a rising flank with growing steps**. This is the same shape the 3×3 continuation measured (fall 63→77, then
rebound), and there it settled 18.6 % of its half-swing back from the trough.

Addendum 51's statement "SRP1 … settled at certification (the tail moved it 1e-6 in total)" does not follow:
the tail and reference runs share one trajectory (Q_tail − Q_ref = −730 € x = 0, −677 € unit; identical bars,
because the bar is the step at k0, before the tail engages) — which says the tail moved Q little, not that Q had
settled. **Addendum 52's "SRP1 statements unchanged" inherits it.**

**No record bounds the SRP1 settling errors**; the runs end at certification. Under the 3×3 analogy (limit between
the trough and the window start), V_SRP1 ∈ [237 k, 298 k] and **R ∈ [0.75, 1.02]**; these are scales under a
model, not bounds. A competing reading (the post-k0 regime has a lower limit, so the certified Q are upper
bounds) reverses the sign of the x = 0 error; only a continuation distinguishes them.

Options:
- **(a) Continue SRP1 x = 0 and unit, W98-style (recommended):** deterministic replay to certification gated
  bitwise cycle by cycle, then 30 cycles; one at a time, attached. **≈ 1.1–1.3 h per cell, ≈ 2.5 h both**
  (SRP1 cells ran 49–59 min to certification at concurrency 3). Replaces the model-dependent ranges with
  measurements and calibrates the half-swing ratio c that option 1(a) needs. Needs a spec (the W98 hooks exist;
  the SRP1 keys and records differ).
- (b) Report R with the model ranges above and state the SRP1 certification as "certified on residuals" only.

**3. Retire "ratio resolution 0.1405" as the comparator for R.** It is `hypot(res/R_ref, V·res_SRP1/R_ref²)`
(`p515_s53_w89_3x3_campaign.py:1659`) over the bars — the step at k0 on four of five cells — i.e. first-step
stopping slack. The in-window swings on the SRP1 cells are 1.2–3.2× those bars. Every "indistinguishable at the
resolution" statement about R in Addenda 48–52 compares against a quantity that does not measure settling.
**Recommendation:** report R against the Decision-2 ranges until continuations replace them.

**4. The SRP1 benchmark proceeds (no decision needed unless you object).** Neither the arms nor the common-Q gate
touch the criterion. But the coordinated arm's Q (SRP1 x = 0, 653,858,731.57) was certified at the trough of the
k0 transient, so the benchmark report will state the claim against a band **[Q132, Q132 + 31 k€]**, and will not
claim a benefit smaller than that band plus the arms' multi-start spread.

## Blocked on the author

Nothing is blocked now. Decision 2(a) is a run no order names: it will not launch without a ruling.

## Found

- **Mechanism (Addendum 52's candidate: a marginal-resource switch at the interface in RES-covered Spring
  hours).** The λ_t clause **cannot be tested from records** — the only per-period price recorded is the exogenous
  E[π_t], not the interface dual. What the records show weighs **against** a switch: in the Spring blocks carrying
  the oscillation, TSO generation cost and DSO flexibility cost move **in phase** (a substitution would move them
  in anti-phase), with smooth second differences (a switch would leave a kink); the mode sits in the later,
  higher-RES years. Advisor's reading: the under-damped mode of the fixed-ρ ADMM map, suppressed by Anderson
  acceleration until it switches off — plausible, not demonstrated. The criterion in 1(a) is robust to either.
  A decisive test would need per-period dual and flexibility capture on a replay (≈ 2 h); it is not needed for the
  criterion.
- **The objective converges faster than the residuals, not slower** (3×3 x = 0): the binding Boyd ratio contracts
  ≈ 0.975/cycle over 63–88, the objective's mode 0.892/cycle. The residual tolerance simply admits a state still
  ≈ 20 k€ from the limit. Correction to the pair report's wording ("the objective converges more slowly than the
  residuals").

## Not confirmed

- The SRP1 settling errors and their sign (Decision 2); node-7's (3×3 unit) settled value — never continued.
- That the transient at k0 on SRP1 is caused by the AA switch-off; inferred from its timing, not from code
  inspection of those runs.
- 5×5 cost of any criterion.
