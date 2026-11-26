# Step 6, round 4 — after `P5_15_ADDENDUM70_ROUND3_REPORT.md` (W174a/b, W175) (expert, 2026-10-08)

Rulings: Addendum 71. Lines refer to the Overleaf clone at `260bd83`. Fourteen edits in `main.tex`, in file order;
all one-sentence except 10 (the benchmark paragraph). After them §2, §3.1, §3.5–3.6 and Appendix A are closed for this
revision; the items listed under "Final pass" are notation and index points to take once, at the whole-paper read.

1. **§2.1, l. 449 (two changes).** "in which every rounded poll direction was inadmissible at every poll, so that" →
   "in which every rounded poll direction was inadmissible at every evaluated poll, so that". And "the runs reported
   here stopped at their evaluation budgets and are described by their recorded certificates." → "of the three runs
   reported here, two stopped when a unit poll failed and one was stopped for review when its completion set exceeded
   the cap; each is described by its recorded certificate."

2. **§2.2.7, l. 647.** "and the master search of Algorithm~\ref{alg:shared_ess_planning_mads} accepts a poll point
   as an improvement only by a determinate margin; smaller differences are ties, reported as ``within resolution''."
   → "; smaller differences are ties, reported as ``within resolution''. The master search of
   Algorithm~\ref{alg:shared_ess_planning_mads} accepted its poll points by the coarser resolution rule stated there;
   every cell it proposed was re-evaluated under this rule before being reported."

3. **§2.3.2, l. 679.** "at the end of the representative year" → "at the end of the block".

4. **§2.3.3, l. 756 and §2.3.5, l. 866.** The "Section~\ref{sec:case_settings}" that closes the storage-parameter list
   ($\eta$, $SoC$, $\varepsilon^{\text{Cl}}$, $c^{\text{Cl}}$, $\varepsilon^{\text{C}}$ at l. 756; $c^{\sigma}$,
   $\varepsilon^{\text{E}}$ at l. 866) → "Section~\ref{sec:case_ess_params}" (the α reference at l. 907 stays).

5. **§2.3.4, l. 841.** "\eqref{eq:soh_chain} the available-energy product" → "\eqref{eq:soh_chain}, the
   available-energy product".

6. **§3.5, l. 1070 — the chemistry (author's choice, Addendum 71 item 7).** Either keep "utility-scale lithium iron
   phosphate battery" and add the citations for the two cycle-life statements and the calendar retention (the EVE
   MB31 datasheet for the 8,000-cycle row; a source for the 10,000-cycle row; an LFP calendar-ageing study), or write
   "utility-scale lithium-ion battery (the NREL ATB utility-scale battery category~\cite{nrel_ess_costs})" and change
   the letter's R1.2(iv) to "the reference technology is utility-scale lithium-ion battery storage; the cycling
   calibrations are datasheet readings (Section~3.5)". Delete the `% [CONFIRM — W175]` under the table either way.

7. **§3.5, table row (l. 1091).** "Datasheet 8\,000 cycles & (8\,000, 1.00, 0.70)" → "Datasheet 8\,000 cycles$^{a}$ &
   (8\,000, 1.00, 0.70)" and add to the caption: "$^{a}$~entered as 10\,000 cycles at 0.80 depth of discharge, the
   same product $N^{\text{DS}} \delta^{\text{DS}}$, which is all \eqref{eq:cycle_life_calibration} uses."

8. **§3.6, l. 1108.** "Every recourse evaluation ran the coordination procedure of \ref{app:admm_updated} under one
   frozen configuration." → "Every certified evaluation reported in this paper ran the coordination procedure of
   \ref{app:admm_updated} under one frozen configuration; the three planning-search campaigns that proposed the
   incumbents ran on earlier states of the code, without the tight tail, and their incumbents were re-evaluated under
   this configuration before being reported."

9. **§3.6, benchmark paragraph (l. 1140–1143).** Replace from "In both, the TSO then dispatches the TN with the
   interface exchanges fixed at the DNs' schedules." to the end of the paragraph by the Worker's wording:

   ```latex
   In both, the TSO then dispatches the TN with the interface exchanges fixed at the DNs' schedules; each DN is then
   re-evaluated at the TN's interface voltage and, where a DN limit is violated, re-solved once at that voltage before
   the TSO dispatches again. The passive DNs select their curtailment by a minimum-curtailment term of 1~\euro/MWh;
   every arrangement is costed with the same function $Q$ as the coordinated value (settlement excluded, that term at
   zero) and reported as the best of three solver starts. The benchmark is evaluated at the plan without shared storage,
   at the single-scenario instance.
   ```

   Delete the `% [CONFIRM — W175]` under it.

10. **Appendix A, preamble, l. 1497.** "that is Gauss--Seidel on the interface channels and, on the storage channel, a
    global-variable consensus in which all three agents solve against the same $z$ before it is updated with residual
    balancing of the penalty parameters, Anderson acceleration and a tightened interior-point tail." → "that is
    Gauss--Seidel on the interface channels and, on the storage channel, a global-variable consensus in which all three
    agents solve against the same $z$ before it is updated; the penalty parameters follow residual balancing, and the
    iteration uses Anderson acceleration and a tightened interior-point tail."

11. **Appendix A.1, l. 1515.** "which puts the agent's local objective on the same footing as a median block's scaled
    objective; the consensus terms themselves are unscaled in every agent." → "which is equivalent to dividing the
    agent's local objective by $\kappa^{\text{E}}$ and puts it on the same footing as a median block's scaled
    objective."

12. **Appendix A.4, l. 1636.** "every local solve of the cycle ended at an optimal status" → "every local solve of the
    cycle ended at a status the solver reports as solved (optimal or acceptable)".

13. **Appendix A.4, l. 1640 (restore the deleted clause).** "The memory is cleared on any change of a penalty parameter
    and on any failed local solve, from the cycle after $k_0$ on (or after the earlier run's stopping cycle in a
    continued evaluation)." → "The memory is cleared on any change of a penalty parameter and on any failed local
    solve, and the acceleration is switched off on any cycle in which every channel passes and, in the certifying
    regime, from the cycle after $k_0$ on (or after the earlier run's stopping cycle in a continued evaluation)."
    Also l. 1644: "(the tail is a declared option, enabled in every campaign)" → "(the tail is a declared option,
    enabled in every campaign behind the reported tables)".

14. **Algorithm 2 (l. 1678–1681 and the loop end, l. 1700–1702).** In the initialisation line, "(in the multi-scenario
    instance the commitment charge and the settlement are activated here)" → "(the interface settlement's weight is set
    to one here and, in the multi-scenario instance, the commitment charge is activated)". At the end of the `\While`
    body, after the `\If{…}{\textbf{exit}\;}` block and before the loop's closing `}`, restore the line
    `        $k \gets k + 1$\;`.

**Comments to delete** (all answered): the `% [CONFIRM — W173]` at l. 444 and l. 452; `% [CONFIRM — W172]` at l. 1517,
1579, 1628, 1647, 1706 and 1717; the two `% [CONFIRM — W175]` of items 6 and 9. The `% [AUTHOR]` note under the
ageing table stays until the citations are in.

---

## Final pass (whole-paper read; from `P5_15_W174B_REAUDIT.md`, all L)

- Hold origin: the holds start one cycle after the first pass (or after the earlier run's stopping cycle), where the
  text still says "at" (T2-1, A3-3, A4-3).
- Window bounds in §2.2.7: the oscillatory window may start at $k_0$ (T2-4); the monotone window starts at or after
  $k_0 + 3$ (T2-5); the gap clause is read at the certifying cycle only (T2-5).
- Indexing: $T^{\text{Cal}}$ and $Y$ carry the cohort's investment year, $T^{\text{Cal}}_{e,y^{\text{Inv}}}$ and
  $Y_{y^{\text{Inv}}}$ (T2-11, T3-7, T3-8); for the storage agent the "ten times" of the clean rule is on its own
  tolerances (T2-9); the rounding of the poll directions ($d = \operatorname{round}(\Delta\,h/\lVert h \rVert_\infty)$)
  and the snap tie-break of variant B are implementation details to state in one clause or leave out (T1-3).
- Literal "Section~4.x" references in §2 and §3 point at the submitted §4 until it is rewritten.
- Nomenclature: the final list (round 1 §A.9 and the Appendix A note), including $\boldsymbol{u}^{0}$, $s = (m,o)$,
  $\sigma^{\pm}$, $c^{\sigma}$, $s^{\pm}$, $\varepsilon^{\text{Cl}}$, $c^{\text{Cl}}$, $\varepsilon^{\text{C}}$,
  $n_{\mathrm{w}}$, $\mathcal{V}$, $\Delta_0$, $\sigma_Q$, $R^{\text{I}}_i$, $\kappa^{\text{E}}$, $w_{y,d}$, $\sigma$.
