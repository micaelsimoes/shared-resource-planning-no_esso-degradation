# Step 6, round 2 — corrections after `P5_15_ADDENDUM68_ROUND1_REPORT.md` and the W171b audit (expert, 2026-10-08)

Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 69. Manuscript lines refer to the Overleaf clone at `42794d4` (the
state W171b audited); the Appendix A paste made since does not move §2. Replacement text is blue, as before.

---

## A. Section 2.1 — the recourse within a block, and Algorithm 1 as run

### A.1 — equation (2) and its list (find: the `subequations` block labelled `eq:operational_recourse_function`
### and the blue "Here:" itemize after it)

Replace the `subequations` block by:

```latex
\begin{subequations}
    \begin{align}
        \textcolor{blue}{Q(\boldsymbol{x})} &= \textcolor{blue}{\min_{\boldsymbol{u}^{0},\,\{\boldsymbol{u}_o\}_{o\in\Omega_O}} \sum_{o\in\Omega_O} \omega_o\, C_o^{\mathrm{Op}} \left(\boldsymbol{x}, \boldsymbol{u}^{0}, \boldsymbol{u}_o \right)}\\
        \textcolor{blue}{\text{s.t.}} & \textcolor{blue}{\quad (\boldsymbol{u}^{0}, \boldsymbol{u}_o) \in \mathcal{U}_o(\boldsymbol{x}), \quad \forall o\in\Omega_O}
    \end{align}
    \label{eq:operational_recourse_function}
\end{subequations}
```

In the "Here:" list, replace the last two items by:

```latex
        \item $\boldsymbol{x}$ contains the scenario-independent investment decisions;
        \item $\boldsymbol{u}^{0}$ contains the operating decisions taken before the scenario is known and common to all
        scenarios of a representative day --- the shared-storage schedule, the TSO's interface exchange and each DSO's
        committed interface import --- and $\boldsymbol{u}_o$ the network dispatch of scenario $o$.
```

and in the paragraph "The absence of a scenario index in $\boldsymbol{x}$…" (l. 381) replace the last sentence by:
"By contrast, the network dispatch $\boldsymbol{u}_o$ adapts to each realization of load, RES generation and market
prices, while the day-ahead decisions $\boldsymbol{u}^{0}$ do not: the recourse of each representative day is itself a
two-stage problem (Subsection~\ref{subsubsec:commitment})."

### A.2 — Algorithm 1 (replace the whole `algorithm` environment labelled `alg:shared_ess_planning_mads`)

```latex
\begin{algorithm}[htbp!]
    \caption{\textcolor{blue}{Derivative-free shared-ESS planning search (mesh adaptive direct search on the
    investment lattice) with ADMM recourse evaluations, as run}}
    \label{alg:shared_ess_planning_mads}
    \DontPrintSemicolon

    \KwIn{Candidate nodes $E^S$; representative years $Y$, days $D$, scenarios and probabilities; lattice units
    $\Delta^S$, $\Delta^E$; duration bounds $\phi^{\text{Min}}, \phi^{\text{Max}}$; capacity limit $E^{\text{Max}}$;
    budget $c^{\text{Inv}}$; budget of new evaluations $N^{\max}$; measured resolution $\sigma_Q$ of the recourse;
    direction seed; poll variant (A or B).}
    \KwOut{Incumbent plan $\boldsymbol{x}^{\star}$, the poll history, and the set of neighbours evaluated at the final
    unit poll (the certificate's scope).}

    \tcp{Decision vector: one cohort per node and one investment year common to all nodes}
    $\boldsymbol{z} = \big( S_e/\Delta^S,\, E_e/\Delta^E \big)_{e \in E^S} \times y^{\text{Inv}} \in \mathbb{Z}^{2|E^S|+1}$,
    with $y^{\text{Inv}} \in Y$ an ordinal; $\boldsymbol{x}(\boldsymbol{z})$ places $(S_e, E_e)$ at node $e$ in year
    $y^{\text{Inv}}$. Admissible: \eqref{eq:master_rated_s}--\eqref{eq:master_budget}, checked in closed form\;
    Cache $\mathcal{C} \gets$ the records of the designed initial search (single-node ladders, run with and without
    the budget as the sensitivity study of Section~4, and their combinations); a cache hit is read, never re-evaluated\;
    $\boldsymbol{x}^{\text{inc}} \gets \arg\min F$ over the certified, budget-feasible points of $\mathcal{C}$
    ($\boldsymbol{x} = 0$ included; ties to the lower investment); $\Delta \gets \Delta_0$; $N \gets 0$\;

    \While{$N < N^{\max}$}{
        \tcp{Poll set at the incumbent}
        \uIf{variant A}{
            $\mathcal{P} \gets$ the $n+1$ OrthoMADS points $\boldsymbol{z}^{\text{inc}} + \Delta \boldsymbol{v}$, with
            $\boldsymbol{v}$ the $n$ Householder directions of the next Halton point and their negative sum; drop the
            inadmissible ones\;
            \If{$\Delta = 1$}{
                $\mathcal{P} \gets \mathcal{P} \cup$ every admissible lattice point within one unit step of
                $\boldsymbol{z}^{\text{inc}}$ in the $\infty$-norm; \lIf{$|\mathcal{P}| > 30$}{\textbf{stop for review}}
            }
        }
        \Else(variant B, $\Delta = 1$ throughout){
            $\mathcal{P} \gets$ the $2n$ OrthoMADS points $\boldsymbol{z}^{\text{inc}} \pm \boldsymbol{v}$, each
            inadmissible point snapped to the nearest admissible point of the incumbent's unit frame; if fewer than
            $n+1$ distinct admissible points result, add admissible unit neighbours until $n+1$\;
        }
        \lIf{the new points of $\mathcal{P}$ exceed $N^{\max} - N$}{\textbf{stop}: budget exhausted}
        Evaluate every new point of $\mathcal{P}$ by Algorithm~\ref{alg:operational_planning_degradation} under the
        production exit (ten consecutive passing cycles, cap 500); record $F$, its bar (the largest objective step over
        the last ten cycles) and the exit; an evaluation that does not reach the exit, or whose storage agent is
        infeasible, is a barrier point, $F = +\infty$ (two in one poll, or three in all, stop the search for review);
        add the records to $\mathcal{C}$; $N \gets N + $ new evaluations\;
        \eIf{some $\boldsymbol{x} \in \mathcal{P}$ has $F(\boldsymbol{x}^{\text{inc}}) - F(\boldsymbol{x}) >
        \max\{\text{bar}_{\boldsymbol{x}} + \text{bar}_{\text{inc}},\, \sigma_Q\}$}{
            $\boldsymbol{x}^{\text{inc}} \gets$ the best such point; $\Delta \gets 2\Delta$ (variant A) or $1$ (variant B)\;
        }{
            \lIf{$\Delta = 1$}{\textbf{terminate}: no evaluated unit neighbour improves $\boldsymbol{x}^{\text{inc}}$}
            $\Delta \gets \Delta/2$\;
        }
    }
    \Return{$\boldsymbol{x}^{\text{inc}}$, the poll history, and the scope of the final unit poll}\;
\end{algorithm}
```

% [CONFIRM — W173] Delta_0 and the double/halve rule of variants A (s47, s51) — the audit records the completion at
% Delta = 1 and the cap 30, not Delta_0; N^max = 20 (s47, s51) and 60 (s53); the Householder construction of the n + 1
% directions (orthomads_n_plus_1_neg); the six-level tie-break of the snap in s53 (not stated, implementation detail).

### A.3 — the paragraph after Algorithm 1 (find: "Algorithm~\ref{alg:shared_ess_planning_mads} is a mesh adaptive direct
### search (MADS) on granular variables"; replace the whole paragraph)

```latex
\textcolor{blue}{Algorithm~\ref{alg:shared_ess_planning_mads} is a mesh adaptive direct search (MADS) on granular
variables~\cite{audet_dennis_2006,abramson_orthomads_2009,audet_granular_2019}, run as a heuristic: it proposes the
incumbents, and every planning statement in this paper is then a pairwise comparison of certified evaluations. Its
decision vector is one converter rating and one energy capacity per interface node, in multiples of $\Delta^S$ and
$\Delta^E$, and one investment year common to all nodes; per-node timing and staged capacity, which the master problem
of Subsection~\ref{subsec:master} admits, were not searched in this study. Two poll variants were used: variant A
(the $n+1$ OrthoMADS directions, completed at unit poll size by the full set of admissible unit neighbours) for the
baseline and for the first search under the doubled flexibility price, and variant B (the $2n$ directions snapped to
the unit frame) for the certificate of the plan found under that price. During the search each evaluation was stopped
by the production exit of the coordination procedure and a candidate was accepted only when it beat the incumbent by
more than the two evaluations' bars and the measured resolution $\sigma_Q$ of the recourse. Every evaluation that enters
a reported comparison was afterwards re-evaluated on the final code under the certification rule of
Subsection~\ref{subsubsec:certification}, and the comparisons use its determinacy threshold; the resolution the search
used is reported in Section~4.7. What the search certifies is the scope of its final unit poll, which the paper states
with each plan: at the baseline incumbent, $\boldsymbol{x} = 0$, every admissible unit neighbour was evaluated (fourteen,
the full box; at a boundary point no positive spanning set is admissible); at the plan found under the doubled
flexibility price the final poll evaluated twelve neighbours, which do not positively span the space, and the plan is
reported as better than each of them, seven determinately and five within resolution, not as a mesh-local optimum. No
global optimality and no optimality gap are claimed. The convergence theory of MADS is the framework's property; the
runs reported here stopped at their evaluation budgets and are described by their recorded certificates.}
```

% [CONFIRM — W173] "fourteen" (x = 0 full box, campaign_s47 termination_certificate) and "twelve" (F2 poll set, T1 L rows)
% against the certificate records — W171b T1-5 counts 61 feasible box neighbours of F2 with 44 unevaluated, i.e. 17
% evaluated; reconcile 12 vs 17 (poll set vs. all evaluated box points) and give the count the sentence should carry.

### A.4 — 2.1, coupling paragraph (find: "it uses the value $Q(\boldsymbol{x})$ returned by the recourse and nothing
### else")

→ "it uses the value $Q(\boldsymbol{x})$ returned by the recourse and the record of its evaluation (its bar and exit),
nothing else".

---

## B. Section 2.2.7 — restore the accepted definitions and state the rule in full

### B.1 — the "Stopping and certification" paragraph: replace "The residual pass at cycle $k_0$ therefore only opens the
certifying regime, which is held fixed from the next cycle on: acceleration off, the tight interior-point tail on, and
the penalty parameters $\rho$ frozen (\ref{app:admm_updated})." by:

```latex
The residual pass at cycle $k_0$ therefore only opens the certifying regime, which is then held fixed: acceleration off,
the tight interior-point tail on, and the penalty parameters $\rho$ frozen (\ref{app:admm_updated}) --- from the next
cycle in a fresh run, or from the stopping cycle of the earlier run in an evaluation continued from one. A later lapse
of the residual test, or a cycle with a failed solve, resets $k_0$ and the turning-point record while the holds stay.
```

### B.2 — the enumerated list: replace items 1 and 2 by

```latex
    \item the objective has shown at least three turning points since $k_0$, so that its period $\hat{P}$ --- the number
    of cycles spanned by the three most recent turning points --- is measured rather than assumed; a step smaller than
    $\tau/100$ carries no sign, and the first three cycles after $k_0$ do not enter the count;
    \item the successive half-swings of the objective are not growing; swings smaller than $\tau/10$ are treated as
    noise and are neither compared nor counted as turning points;
```

and item 3's "$n_{\mathrm{w}} = \max\{20, \lceil 1.1\,\hat{P} \rceil\}$ cycles" → "$n_{\mathrm{w}} = \max\{20, \lceil
1.1\,\hat{P} \rceil\}$ cycles, all after $k_0$,".

### B.3 — the monotone-branch paragraph: replace the whole paragraph by

```latex
\textcolor{blue}{An objective that descends without a sign change is certified by a monotone branch when, over a window
of $2P_{\max}$ cycles starting after $k_0$ with $P_{\max}$ the longest period measured on the instance, its steps are
decreasing (the mean step over the second half of the window below that over the first half, or every step below
$\tau/100$), its range is at most $\tau$, $|\text{last step}| \times 2P_{\max} \le \tau$ --- a linear bound on the
remaining descent that uses no fit --- and conditions 4 and 5 hold on that window. An evaluation that satisfies neither
branch within its cycle cap is reported in an uncertified form, with its consensus gap at the cap and its settling
slack, the objective's movement between the certification cycle of its earlier, residual-based run and the cap. Every
certificate reported in this paper was decided, or re-decided from its committed record, under this rule, and certifies
at the cycle reported; the single-scenario tables carry these certificates. The multi-scenario instance was evaluated
under the production exit (ten consecutive passing cycles with every local solve successful); its reference evaluation
was continued past that exit to measure the remaining movement of the objective, and the ratio of Section~4.5 is
reported as a band that allows for it.}
```

### B.4 — "Determinacy" paragraph: replace "A difference involving an uncertified evaluation is determinate only if it
exceeds three times the larger of that evaluation's consensus gap and settling slack." by "A difference involving
uncertified evaluations is determinate only if it exceeds three times the largest of their consensus gaps and settling
slacks, in both gross and gap-corrected terms (the objective plus the priced gap)." and "when it exceeds $\max\{3\,b,
2\tau\}$" → "when it is at least $\max\{3\,b, 2\tau\}$".

### B.5 — the $Q$ components paragraph (find "generation cost in the TN, activated flexibility and curtailment in the
DNs, the closure-slack penalty"): replace that clause by "generation cost in the TN at the scenario's market price,
activated flexibility in the DNs, load curtailment at its price, the closure-slack penalty of the storage model
(Subsection~\ref{subsubsec:network_storage}) and the feasibility-slack penalties of the network models, all at their
solver bound in every reported evaluation". After "…and is discounted at the representative year.}" add the sentence:
"Renewable curtailment carries no cost in $Q(\boldsymbol{x})$; it is reported as energy."

---

## C. Section 2.3 — the two H rows and the L rows

### C.1 — 2.3 intro, the blue paragraph "The storage schedule itself --- charging, discharging and reactive power in
every hour --- is a consensus variable…": replace by

```latex
\textcolor{blue}{The net storage schedule --- active and reactive power at the interface node in every hour --- is a
consensus variable between the operators' network models and the shared-ESS agent: the network models carry the
state-of-charge dynamics and the converter limits, and the agent carries the ageing chain that links the representative
years. Each model keeps its own split of the net power into charging and discharging, tied by its own complementarity
condition; the split that ages the cells is the agent's. The two views are reconciled by the ADMM consensus channel
on the net schedule (\ref{app:admm_updated}).}
```

### C.2 — 2.3.3, sign convention (find "The net active-power injection at the node is $P^{\text{E}}_{e,y,d,t} =
P^{\text{Dch}}_{e,y,d,t} - P^{\text{Ch}}_{e,y,d,t}$ (positive when discharging)"): → "The net active power of the
storage, in the consumption convention shared with the agent, is $P^{\text{E}}_{e,y,d,t} = P^{\text{Ch}}_{e,y,d,t} -
P^{\text{Dch}}_{e,y,d,t}$ (positive when charging)". In the same subsection's "where" paragraph, "In the multi-scenario
instance the storage schedule of a block is a single scenario-free variable referenced by every market and operation
scenario's balance equations --- a day-ahead commitment --- so that the schedule the agent ages is the one every
scenario runs" → "…so that the net schedule the agent ages is the one every scenario runs".

### C.3 — 2.3.3, initial state (find "$E^{\text{SoC}}_{e,y,d,t_0} = SoC^{0}_e \, E^{\text{Av}}_{e,y}$" in
`eq:soc_closure`): replace the first member by "$E^{\text{SoC}}_{e,y,d,0} = SoC^{0}_e \, E^{\text{Av}}_{e,y}$" and in
the recursion's description add after "and the stored energy follows": "from the constant pre-day state
$E^{\text{SoC}}_{e,y,d,0}$". In the "where" paragraph, "the slack is penalised at $c^{\text{Cl}}$ per MWh in the block
objective" → "the slack is penalised at $c^{\text{Cl}}$ per MWh in the objective of each network model that carries the
storage". Then replace the `% [CONFIRM — W169]` comment (l. 747–748) by the printed sentence:
"The closure slack was at its lower bound, to solver tolerance, at every certificate used in this paper."

### C.4 — 2.3.4, first paragraph (find "up to a slack pair that is penalised in its objective and is never active at a
consensus point within the agent's ratings"): → "up to a slack pair that is penalised in its objective and was at its
lower bound at every certified point". Linear/nonlinear sentence (find "are the agent's nonlinear rows"): → "and the
converter circle are the agent's nonlinear rows".

### C.5 — 2.3.5: replace the `% [CONFIRM — W169]` comment (l. 857–858) by the printed sentences, placed after "…is
checked after every solve.": "It is recorded, not enforced; at the certified points it was below $4 \times 10^{-5}$ of
the rating. The slack pair $\sigma$ was at its lower bound at every certified point." (and delete "is checked after
every solve" → "is checked after every solve and recorded").

### C.6 — 2.3.6: the scenario index. Replace the first sentence of the subsection ("In the multi-scenario instance each
operation scenario $o$ is instantiated inside every network block with its own load and RES realization, while…") by:
"In the multi-scenario instance every scenario $s = (m, o)$ --- a market scenario $m \in \Omega_M$ and an operation
scenario $o \in \Omega_O$ of load and RES generation, with probability $\omega_s = \omega_m \omega_o$ --- is held
inside every network block, while…". In the two equations and the text of the subsection replace the index $o$ by $s$
and $\omega_o$ by $\omega_s$, the sums $\sum_{o \in \Omega_O}$ by $\sum_{s \in \Omega_M \times \Omega_O}$ (four
occurrences), and "($\forall i$, $o$, $t$)" by "($\forall i$, $s$, $t$)". Replace "At a single operation scenario
\eqref{eq:commitment_deviation}--\eqref{eq:commitment_charge} vanish identically, and the single-scenario instance is
unaffected by them." by "With a single scenario \eqref{eq:commitment_deviation}--\eqref{eq:commitment_charge} vanish
identically, so the single-scenario instance is unaffected by them; the commitment is to the expectation over market
and operation scenarios alike, so deviations driven by the price realization are charged like those driven by load and
generation."

### C.7 — 2.3.1 wording (find "a dedicated shared ESS subproblem" in the first paragraph of 2.3): → "a dedicated
shared-ESS subproblem per interface node, spanning all representative years and days".

### C.8 — calibration value (draft and §3.4 when written): C4 $k = 22{,}429$, not 22,430.

---

## D. Response letter

D.1 **R3.6** (B.5 of round 1, now wrong): "(within the horizon it never does)" → "(in 2035 under the baseline, the
mid-block and the unit-retention calibrations; not within the horizon under the other two)".

D.2 **R3.5**: delete the clause "the sentence of the submitted version announcing 0.5 % and 2 % cases …" and keep only
that the sensitivity was not run and is not reported.

D.3 **R1.2**, the basis sentence: "the horizon (three representative years standing for five-year blocks)" → "the
horizon (three representative years standing for five-year blocks at the single-scenario instance; five three-year
blocks at the multi-scenario instance)".

D.4 **Further changes**: add the item "The investment-cost file was corrected: the energy cost per MWh was derived from
the 4-h system cost by dividing by 5 instead of 4 in the submitted version; the corrected values (×1.25) are in
Table~[investment costs], and every result uses them."

D.5 Stray `"` at l. 77: delete.

D.6 **R1.5 / "Figure 15"**: pending the PDF the reviewers read; put that PDF in `manuscript_submitted/` so the Planner
can map the reviewers' figure numbers.

---

## E. Planner — next order (zero-solve)

> Addendum 69 and `STEP6_ROUND2_CORRECTIONS.md` are in the repo root. (W172) audit Appendix A as pasted at the current
> Overleaf HEAD against the code, in the W166/W171b form, including its `% [CONFIRM — W172]` comments, and read the
> interface ratings R^I_i of the three DNs for Section 3.5; (W173) the two `% [CONFIRM — W173]` points of §2.1: Δ_0 and
> the frame rule of s47/s51, and the neighbour counts to print for x = 0 and for the F2 plan (poll set versus all
> evaluated box points — one number each, with its definition in a clause); (W174) once the author confirms the round-2
> corrections are in Overleaf, re-run the number check at that HEAD and re-audit §2.1–2.3 for the rows of W171b only.
> Commit and push; nothing else runs.
