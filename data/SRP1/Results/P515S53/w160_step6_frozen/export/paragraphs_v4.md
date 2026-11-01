# Step 6 paragraphs, v4: v3 corrected by its figure check (W162)

<!-- v4, Planner, 2026-10-07: paragraphs_v3.md (sha256 62d26e27..., cb0a0bc6) corrected for the W162 figure check (d1f3e186): RES slack 7.8 k EUR; the tail tolerance change stated per subproblem; the 15-21 k EUR anchor; clause 5 scoped to the window; successful (not clean) local solves for the residual test; the 3 tau figure an estimate; the 0.9 tau bound scoped; 32 GiB; an unsourced fraction dropped. -->
<!-- v3 header follows. Planner, 2026-10-07. Supersedes the prose of paragraphs_v2.md (sha256 73b42c49..., commit 2a1d7f92) for sections
(ii)-(iv) and the accepted sentences. Edits per PLANNER_BRIEF_2026-09-13.md Addendum 66: internal references stripped
from the prose; the "converges more slowly" sentence replaced; P-hat, P_max, lambda and the settling slack defined as
the code computes them; tau as a formula; N filled. The sources tables, the reviewer map and the scorecard are
unchanged from paragraphs_v2.md (accepted, Addendum 66) and keep their identifiers there. The frozen tables are
unchanged (frozen_step6_tables_v1_590088fe.json). This comment block is not manuscript text. -->

## (ii) Certification paragraph

> **Stopping and certification.** Each recourse evaluation runs consensus ADMM until the primal and dual residuals of
> every consensus channel pass Boyd's absolute-plus-relative test [Boyd et al. 2011, §3.3], with ε_abs = 10⁻⁵ and
> ε_rel = 10⁻⁴. Every local NLP must also solve successfully (optimal or locally optimal) in the same cycle.
>
> Passing the residual test does not imply that the objective has settled: on the reference evaluations the objective
> moved by a further 15–21 k€ after the residual-based stopping rule had certified them. From the first residual pass k₀ onwards the regime is
> therefore held fixed: Anderson acceleration off, the tight interior-point tail on (complementarity tolerance 10⁻⁶),
> and the ADMM penalty parameters ρ frozen. An evaluation is certified at the first cycle k\* at which all of the
> following hold:
>
> 1. the objective has shown at least three turning points (extrema) since k₀, so that its period is measured rather
>    than assumed;
> 2. the successive half-swings are not growing. Swings smaller than τ/10 = 453.91 € are treated as noise: they are
>    neither compared nor registered as turning points;
> 3. the range of the objective over the last W = max(20, ⌈1.1 P̂⌉) cycles is at most τ. Here P̂ is the period
>    measured over the three most recent turning points: the number of cycles from the third-last to the last;
> 4. the priced interface-consensus gap satisfies |t_sum| ≤ τ/2 = 2,269.53 €;
> 5. every cycle in the window the test reads was solved cleanly. A local solve is clean when it ends `Optimal`, or
>    `Acceptable` on a primary attempt with all four IPOPT error metrics within 10× the tight-tail tolerances. Cycles
>    before the window are read only for the turning points and the swing history, and are not vetoed.
>
> A monotone branch certifies an evaluation that shows no turning point within 2 P_max = 60 cycles, where P_max = 30
> cycles is the longest period measured on the instance. It requires:
> - steps decreasing over the window;
> - a window range of at most τ;
> - |last step| × 60 ≤ τ, a linear bound on the remaining descent with no fit.
>
> **The threshold.** τ = δR·V/4, with δR = 0.07 and V = 259,375.33 €, the SRP1 value of the reference unit at the
> certificate in force when the rule was frozen. Four evaluations make up a ratio of two values, each in error by at
> most τ, so the ratio is resolved to δR. This gives τ = 4,539.07 €.
>
> **What τ bounds.** Ten of the 46 certificates the tables use stop within 5 % of τ. The rule therefore bounds each
> evaluation's stopping error at about τ, not well inside it. Evaluations continued past a certificate under this rule
> moved by at most 0.9 τ.
>
> **Replays.** Every evaluation repeated from an earlier record replayed that record bitwise up to k₀.
>
> **Uncertified evaluations.** An evaluation that does not certify by its cap is reported in an uncertified form. A
> difference involving it is called determinate only if its margin exceeds three times the larger of two quantities, in
> both gross and gap-corrected terms:
> - its consensus gap |t_sum| at the cap;
> - its settling slack. The slack is the objective's movement from its earlier certificate: the objective at the cap
>   minus the objective at the cycle where the earlier run of the same evaluation had certified. Every uncertified
>   evaluation in the tables has such an earlier certificate. An evaluation without one has no defined slack, and a
>   difference involving it would be reported indeterminate.
>
> **Determinacy between certified evaluations.** A difference between two certified evaluations is determinate only if
> it exceeds max(3 × the larger band, 2τ).
>
> **Outcomes.** Of the 42 SRP1 evaluations run under this rule, 32 certified (24 oscillatory, 8 monotone). The 10
> uncertified evaluations divide as follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the
> growth test. The median certification cycle was 174, and certification came a median 65 cycles after the first
> residual pass (range 21–89). The median range/τ at certification was 0.86. The four further evaluations all
> certified.

## (iii) Limitations paragraph

> **Degenerate interface duals.** In some evaluations, storage discharge drives the transmission network's
> conventional generation to its lower bound while distribution flexibility is at zero activation, where its cost is
> non-smooth. The interface price λ (the consensus dual of the interface power) is then set-valued, λ ∈ [0, c_flex], and
> ADMM's consensus converges at the pace of a degenerate dual. This is a known property of the method, not a defect.
>
> Certification reports these cases as "objective settled; interface-consensus gap unresolved". They appear with large
> storage and a high flexibility price, and they share one fingerprint:
> - a priced gap of −8.8 to −9.4 k€;
> - split about 55 % at node 5, 31 % at node 9 and 14 % at node 7;
> - the power-flow primal residual, as a fraction of its tolerance, stuck near 0.69.
>
> At the baseline price, the 5 MWh evaluation fails to certify for a different reason. Repeated transmission solver
> recoveries break the growth test; its consensus gap is small (+1.1 k€) and of the opposite sign.
>
> **Solver recoveries.** In a few transmission blocks, some solves reach the iteration limit and recover only to
> IPOPT's acceptable level. These cycles are excluded from certification evidence. After the first residual pass they
> number 21 cycles in 10 evaluations, all in transmission blocks; the recurring block is 2035 Spring.
>
> **RES bound slack.** RES output is bounded by availability plus a 10⁻⁵ pu numerical slack. The resulting
> over-production totals ≈ 62 MWh-equivalent over the horizon and is reported separately.
>
> **Monotone branch.** The monotone branch bounds the remaining descent linearly, without a fit. An evaluation drifting
> with a half-life longer than its window can certify with drift left beyond τ: ≈ 3 τ is estimated for the reference
> corner plan's measured half-life at the earlier window L = 44.
>
> **[Hypothesis, not a result]** The 0.70 SoH floor raises the elasticity of value to available energy (ε_AE 1.04–1.85,
> against 0.41–0.62 for the same three resolvable arms at 0.50). This is read as a late-life tail effect: the floor
> removes the low-SoH years whose throughput is least valuable. The decomposition that would show it has not been
> measured.

*Draft note, not manuscript text:* in the manuscript the bracket label "[Hypothesis, not a result]" is dropped. The
sentence "This is read as … has not been measured" already marks it as a hypothesis (Addendum 66).

## (iv) Reproducibility paragraph

> **Determinism.** All results were produced on one machine with one solver build, one evaluation at a time, and
> single-threaded. Every evaluation's environment sets OMP, MKL, OpenBLAS, vecLib and NumExpr to one thread, and this is
> recorded per evaluation. The installed HSL library is built without OpenMP, so MA97 ran serially. Replaying a run
> reproduces its trajectory bitwise:
> - the 3 × 3 multi-scenario instance's x = 0 evaluation replayed its 72 recorded cycles bitwise (72/72);
> - the three SRP1 reference continuations replayed bitwise to their certification cycles (3/3);
> - every evaluation repeated from an earlier record replayed that record bitwise up to its first residual pass before
>   continuing.
>
> **Software and machine.** The computations used IPOPT 3.14.18 with the HSL linear solvers MA97 (network subproblems)
> and MA57 (storage-operator subproblem), Python 3.11.11 and Pyomo 6.9.5 on macOS 27.0. They ran on an Apple M2 Max
> with 32 GiB of memory.
>
> **Solver exits.** Two distribution blocks (node 7 in 2025 Winter, node 5 in 2035 Winter) end some primary solves at
> IPOPT's acceptable level, within 10× the tight-tail tolerances. They are counted as clean under the rule stated
> above.
>
> **Offsets common to every evaluation.** Two effects cancel in every reported difference:
> - **the tight tail:** tightening the interior-point tail lowered the certified
>   objective of each of the three SRP1 references by ≈ 1.1 × 10⁻⁶ relative, with the same sign. The complementarity
>   tolerance went to 10⁻⁶ from 10⁻⁴ in the distribution subproblems and from 5 × 10⁻⁴ in the transmission subproblem;
> - **the RES slack:** the 10⁻⁵ pu RES slack lowers the objective by ≈ 7.8 k€ first-order (≈ 1.2 × 10⁻⁵ of it).

**Supplementary note (not main text).**
- The versions above were read on 2026-10-06 from the environment, which was unchanged since the campaign.
- The linear solvers are taken from the campaign's own IPOPT log banners.

## Accepted sentences (unchanged)

1. Flexibility ladder: *the unit pays at a flexibility price 1.75 and 2 times the base price (+14.2 k€ and +46.3 k€,
   determinate) and breaks even between 1.5 and 1.75 (≈ 1.63 by linear interpolation)*.
2. Minimum SoH: *lowering the end-of-life floor from 0.70 to 0.50 releases cycling in every year (EFC/day +0.18,
   +0.20, +0.27) but adds only +4.9 k€, within resolution; the unit still loses 59.5 k€ determinately, so the floor
   is not what makes storage uneconomic at SRP1*.
3. Ageing: *without ageing the unit is at break-even (−4.1 k€, within resolution); with the baseline calibration and
   the 0.70 floor it loses 31.6–73.7 k€ across the aged arms, determinately*.
4. Coordination: *the benefit is the value of dispatching DN flexibility against the TN's marginal value λ_t rather than
   the wholesale price π_t; coordination — or a locational real-time signal computed by the TSO, which is coordination
   by another name — delivers it, a static rule with wholesale exposure does not.*

## Unchanged from paragraphs_v2.md

The sources tables for (ii)–(iv), the response-to-reviewers map and the prediction scorecard are unchanged and keep
their identifiers. The map's "editorial" rows are the author's to write.
