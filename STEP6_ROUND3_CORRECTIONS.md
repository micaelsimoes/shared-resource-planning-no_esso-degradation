# Step 6, round 3 — Appendix A and §2.1 after W172/W173; §3.5–3.6 drafted (expert, 2026-10-08)

Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 70. Manuscript lines refer to the Overleaf clone at `407f8df` (the
state W172/W173 audited). Replacement text is blue. The `% [CONFIRM — W175]` comments name the Planner task that
closes them.

---

## A. Appendix A

### A.1 — A.6, the H row (l. 1674–1675). Replace "At a single scenario all of these terms vanish identically." by:

"At a single scenario the commitment charge and the voltage regularisation are not constructed; the interface
settlement remains in every local objective at full weight and is excluded from the reported cost as a transfer."

### A.2 — the three M rows

(a) **A.1 (l. 1427–1428).** "in the multi-scenario instance these are the \emph{expected} interface quantities,
$\sum_{o} \omega_o P^{\text{I}}_{i,o,t}$, of Subsection~\ref{subsubsec:commitment}" → "in the multi-scenario instance
these are the \emph{expected} interface quantities, $\sum_{s \in \Omega_M \times \Omega_O} \omega_s
P^{\text{I}}_{i,s,t}$, of Subsection~\ref{subsubsec:commitment}".

(b) **A.3 (l. 1528).** "which is the minimiser of the sum of the three agents' storage terms over $z$." → "which is the
minimiser over $z$ of the three agents' consensus terms taken with unit weight; the factor $\kappa^{\text{E}}$ scales the
storage agent's local problem only and does not enter the update."

(c) **A.5 algorithm, end of cycle (l. 1650–1655).** Replace the `\If{the stopping rule…}{\textbf{exit}\;}` block and the
line "Balance $\rho_g$ (unless frozen); apply the Anderson step (unless off); set the tight tail for the next cycle\;" by:

```latex
        Apply the Anderson step (unless off); balance $\rho_g$ (unless frozen) and clear the acceleration memory if
        any $\rho_g$ changed; set the tight tail for the next cycle\;
        \If{the stopping rule in force is met (production: ten consecutive passing cycles; single-scenario tables:
        the certification rule, with its holds)}{
            \textbf{exit}\;
        }
```

### A.3 — the L rows (all cheap; take them)

(d) **Preamble (l. 1410).** "run as a Gauss--Seidel sweep (DSOs, then TSO, then the storage agent)" → "run as a sweep
(DSOs, then TSO, then the storage agent) that is Gauss--Seidel on the interface channels and, on the storage channel,
a global-variable consensus in which all three agents solve against the same $z$ before it is updated".

(e) **A.1 (l. 1421–1423).** "so that every block is solved in the same units and the sum over blocks is the discounted
operating cost of \eqref{eq:recourse_value}." → "so that every block is solved in the same units and the sum over
blocks, after the exclusions of Subsection~\ref{subsubsec:certification}, is the discounted operating cost of
\eqref{eq:recourse_value}."

(f) **A.1 (l. 1438).** "which puts them in the same units as the network blocks' scaled objectives." → "which puts the
agent's local objective on the same footing as a median block's scaled objective; the consensus terms themselves are
unscaled in every agent."

(g) **A.2 (eq. admm_esso_local, l. 1485–1489) and §2.3.5 (eq. esso_objective, l. 851).** In §2.3.5 replace
"$\left( \mathcal{L}^{E,P}_{e,y,d,t} + \mathcal{L}^{E,Q}_{e,y,d,t} \right)$" by "$\kappa^{\text{E}} \left(
\mathcal{L}^{E,P}_{e,y,d,t} + \mathcal{L}^{E,Q}_{e,y,d,t} \right)$" and add after the equation's "subject to" paragraph's
first sentence: "Here $\mathcal{L}^{E,P}$ and $\mathcal{L}^{E,Q}$ are the agent's consensus terms on its net active and
reactive power, written out in \eqref{eq:admm_esso_local}, and $\kappa^{\text{E}}$ the scale of
\ref{app:admm_updated_agents}."

(h) **A.3 (l. 1542–1543).** "No consensus or dual update is made for a block or node in which any of the agents' solves
failed in the cycle; that block keeps its previous values." → "An agent whose solve failed keeps its previous copy; the
interface duals of a block are updated only when both its TSO and its DSO solves succeeded, and $z$ and the three
storage duals of a node only when all three solves succeeded."

(i) **A.3 (l. 1554–1555).** "all $\rho_g$ are frozen from the cycle after the first residual pass." → "all $\rho_g$ are
frozen from the cycle after the first residual pass, or from the stopping cycle of the earlier run in an evaluation
continued from one."

(j) **A.4 (l. 1565–1567).** "$s_{\text{E}} = \lVert \rho^{\text{E}} \alpha\, \Delta z \rVert$" → "$s_{\text{E}} = \lVert
\rho^{\text{E}} \alpha\, \Delta z \rVert$ taken over the three agents (so $\sqrt{3}\,\rho^{\text{E}} \alpha \lVert \Delta
z \rVert$)"; and "$\varepsilon^{\text{dual}} = \sqrt{n}\,\varepsilon_{\text{abs}} + \varepsilon_{\text{rel}} \lVert
\lambda \rVert$" → "$\varepsilon^{\text{dual}} = \sqrt{n}\,\varepsilon_{\text{abs}} + \varepsilon_{\text{rel}} \lVert
\lambda \rVert$, with $\lambda$ the DSO-side dual on the interface channels and all three duals on the storage channel".

(k) **A.4 (l. 1574–1576).** "The single-scenario evaluations of this paper were stopped by the certification rule of
Subsection~\ref{subsubsec:certification}, which uses the first passing cycle $k_0$ as its starting point." → "The
single-scenario evaluations of this paper were stopped by the certification rule of
Subsection~\ref{subsubsec:certification} or by its cycle cap, whichever came first; the holds start at the first
passing cycle, while the rule's own $k_0$ resets on a lapse."

(l) **A.4 (l. 1578–1579).** "is applied to the pair $(z, \lambda/\rho)$ of every channel" → "is applied, on the storage
channel, to $z$ and the three agents' $\lambda_a/\rho$ and, on each interface channel, to the TSO's copy and the
DSO-side $\lambda/\rho$".

(m) **A.4 (l. 1583, 1587).** "from the cycle after $k_0$ on" → "from the cycle after $k_0$ on (or after the earlier run's
stopping cycle in a continued evaluation)"; "the certifying regime holds the tail on from $k_0$" → "the certifying
regime holds the tail on from the cycle after $k_0$".

(n) **A.4 (l. 1586–1587).** "the production setting restores the default tolerance otherwise" → "the production setting
restores each operator's own tolerance otherwise (the tail is a declared option, enabled in every campaign)".

(o) **A.4 (l. 1588–1589).** "A local solve that does not reach an optimal status is retried under a documented sequence
of solver settings; an exit at the solver's acceptable level is counted as a successful solve for the stopping test,
and the certification rule decides separately whether the cycle is clean." → "A local solve that ends at the iteration
limit, infeasible or in a solver error is retried in two tiers, a cold restart with the agent's recovery settings and
the same with the adaptive barrier strategy (Section~\ref{sec:case_settings}); an exit at the solver's acceptable level
is counted as a successful solve for the stopping test, and the certification rule decides separately whether the
cycle is clean."

(p) **A.5 (l. 1605–1606).** "the plan $\boldsymbol{x}$ itself never enters a network model except through these
capacities." → "after the initialisation solves, the plan $\boldsymbol{x}$ enters a network model only through these
capacities."

(q) **A.5 algorithm, initialisation (l. 1628–1630).** Replace the line "$z \gets$ the average of the three agents'
storage copies; initialise the interface duals from the initial solutions; $\rho_g \gets \rho_{g,0}$; convert every
model to its ADMM form \eqref{eq:admm_network_local}, \eqref{eq:admm_esso_local}; $k \gets 1$\;" by:

```latex
    Convert every model to its ADMM form \eqref{eq:admm_network_local}, \eqref{eq:admm_esso_local} with
    $\rho_g \gets \rho_{g,0}$ (in the multi-scenario instance the commitment charge and the settlement are activated
    here); $z \gets$ the average of the three agents' storage copies; storage duals at zero; interface duals set by one
    dual-ascent step from zero on the initial solutions; $k \gets 1$\;
```

(r) **A.5 algorithm (l. 1648–1649).** "record the objective, residuals, gap and solve statuses of the cycle" → "record
the objective, residuals and solve statuses of the cycle (the priced interface gap is recorded by the campaign
harness)".

(s) **A.6 (l. 1668–1669).** "the TSO represents each DN by that expected exchange with a scenario-free adjustment as its
only interface freedom" → "the TSO represents each DN by the DN's expected exchange at initialisation, held fixed,
plus a scenario-free adjustment bounded by the interface rating as its only interface freedom (the same construction
at one scenario)".

---

## B. Section 2.1 — Algorithm 1 and the F2 sentence

(a) **l. 410.** "$\Delta \gets \Delta_0$" → "$\Delta \gets \Delta_0$ ($\Delta_0 = 4$ lattice units in variant A;
$\Delta = 1$ throughout in variant B)". Add $\Delta_0$ to the `\KwIn` list: "initial poll size $\Delta_0$;".

(b) **l. 412.** "\While{$N < N^{\max}$}{" → "\While{polls remain (at most 60) and a poll can be launched}{"; and the line
"\lIf{the new points of $\mathcal{P}$ exceed $N^{\max} - N$}{\textbf{stop}: budget exhausted}" stays — it is the
budget rule as run (a poll is refused only when its new points would exceed $N^{\max} - N$).

(c) **l. 423, variant B completion.** "if fewer than $n+1$ distinct admissible points result, add admissible unit
neighbours until $n+1$\;" → "if fewer than $n+1$ distinct admissible points result, add every admissible unit neighbour
(the poll is refused for review beyond 30 points)\;".

(d) **Paragraph after the algorithm (l. 443).** Replace "at the plan found under the doubled flexibility price the final
poll evaluated twelve neighbours, which do not positively span the space, and the plan is reported as better than each
of them, seven determinately and five within resolution, not as a mesh-local optimum." by: "at the plan found under
the doubled flexibility price, seventeen of its 61 admissible unit neighbours were evaluated (the ten points of the
final poll and seven earlier evaluations) and none is determinately better; of the thirteen re-evaluated under the
certification rule the plan is better than twelve, seven determinately and five within resolution, and within
resolution of the thirteenth --- it is not reported as a mesh-local optimum." Then add, after "…for the baseline and
for the first search under the doubled flexibility price," the clause: "in which every rounded poll direction was
inadmissible at every poll, so that every evaluated point came from the unit-neighbour completion,".

---

## C. Section 3 — the two new subsections, and the sentences the rulings require

Numbering: 3.1 Investment Costs, 3.2 Market Data, 3.3 Transmission Network, 3.4 Active Distribution Networks, **3.5
Shared Energy Storage Parameters** (new; takes the storage paragraphs out of 3.4), **3.6 Evaluation and Certification
Settings** (new). In §2, replace "Section~3.4" (the calibrations) by "Section~\ref{sec:case_ess_params}" and every
"Section~3.5" by "Section~\ref{sec:case_settings}" (three occurrences); "Section~4.7" stays literal until §4 exists.

### C.1 — 3.1 Investment Costs

Replace the table by W167's fragment (`data/SRP1/Results/P515S53/w167_nomenclature_years/year_tables/tab_investment_cost.tex`)
and add after "The corresponding investment costs are reported in Table~\ref{tab:investment_cost}.":

```latex
\textcolor{blue}{The three trajectories enter the plan's investment cost $I(\boldsymbol{x})$ as their
probability-weighted sum, which, $I$ being linear in the unit costs, equals the cost at the expected trajectory
(256.32~k\euro/MVA and 253.88~k\euro/MWh in 2025). The energy-cost row corrects the submitted version, where the
cost per MWh had been derived from the 4-h system cost by dividing by 5 instead of 4.}
```

### C.2 — 3.3 Transmission Network: add one sentence where the TN generators are described:

```latex
\textcolor{blue}{Conventional generation in the TN is priced at the wholesale energy price of the scenario; no
separate generator cost curves are used.}
```

### C.3 — 3.4 Active Distribution Networks: delete the two red paragraphs at its end (the 70 %/60 %/80 % floor sentence
and the 1 %/0.5 %/2 % calendar sentence); their content, as run, is in 3.5. Fix the branch table
`tab:cs1_ieee33_branches`: branch 1 is the interface transformer, rated 200, 100 and 150 MVA for the ADNs at TN
nodes 5, 7 and 9 (the table prints 200 for all three); give it per ADN in the table or state it in the caption.

### C.4 — new 3.5 (paste after 3.4)

```latex
\subsection{\textcolor{blue}{Shared Energy Storage Parameters}}
\label{sec:case_ess_params}

\textcolor{blue}{The shared ESS is a utility-scale lithium iron phosphate battery with charging and discharging
efficiencies $\eta^{\text{Ch}} = 0.97$ and $\eta^{\text{Dch}} = 0.96$, a usable state-of-charge window of 10--90\,\% of
the available energy ($SoC^{\text{Min}} = 0.10$, $SoC^{\text{Max}} = 0.90$) and a daily initial state of 50\,\%
($SoC^{0} = 0.50$); the daily closure slack is bounded by $\varepsilon^{\text{Cl}} = 0.05$ of the available energy (plus
a numerical allowance of $10^{-5}$~p.u.) and priced at $c^{\text{Cl}} = 10^{3}$~\euro/MWh, the network complementarity
tolerance is $\varepsilon^{\text{C}} = 10^{-4}$, and the calendar lifetime is $T^{\text{Cal}} = 15$~years. The agent's
regularisation weight is $\varepsilon^{\text{E}} = 10^{-5}$ and its slack penalty $c^{\sigma} = 10^{3}$. Capacities are
sized on the lattice $\Delta^S = 0.25$~MVA, $\Delta^E = 0.5$~MWh with durations between 2 and 4~h, at most 5~MWh per
node and a budget of 1~M\euro.}

\textcolor{blue}{Ageing is calibrated from datasheet cycle-life statements through
\eqref{eq:cycle_life_calibration}; Table~\ref{tab:ageing_calibrations} lists the calibrations used. The baseline reads
the datasheet count of 10{,}000 cycles at 80\,\% depth of discharge as cycles to 80\,\% retention ($k = 35{,}851$) and
adds a calendar retention of $\phi^{\text{Cal}} = 0.985$ per year (80\,\% retention over the 15-year calendar life
without cycling); the end-of-life floor is $SoH^{\text{Min}} = 0.70$. The other rows are the sensitivities of
Section~4.3: the same count without calendar fade; the same count read as cycles to 50\,\% retention, evaluated at the
end of the block or at its midpoint; a second datasheet of 8{,}000 full cycles to 70\,\% retention; no ageing; and the
baseline with a 0.50 floor.}

\begin{table}[htbp!]
    \centering
    \caption{\textcolor{blue}{Ageing calibrations. $(N^{\text{DS}}, \delta^{\text{DS}}, R^{\text{DS}})$: datasheet cycles,
    depth of discharge and retention; $k_e$ from \eqref{eq:cycle_life_calibration}; SoH point: the state of health
    that sets a block's available energy (end of block, or its midpoint).}}
    \footnotesize\setlength{\tabcolsep}{4pt}
    \begin{tabular}{l c r c c c}
        \toprule
        Calibration & $(N^{\text{DS}}, \delta^{\text{DS}}, R^{\text{DS}})$ & $k_e$ & $\phi^{\text{Cal}}$ & SoH point & $SoH^{\text{Min}}$ \\
        \midrule
        Baseline (cycling + calendar) & (10\,000, 0.80, 0.80) & 35\,851 & 0.985 & end & 0.70 \\
        Cycling only                  & (10\,000, 0.80, 0.80) & 35\,851 & 1.000 & end & 0.70 \\
        Retention 50\,\%              & (10\,000, 0.80, 0.50) & 11\,542 & 1.000 & end & 0.70 \\
        Retention 50\,\%, mid-block   & (10\,000, 0.80, 0.50) & 11\,542 & 1.000 & mid & 0.70 \\
        Datasheet 8\,000 cycles       & (8\,000, 1.00, 0.70)  & 22\,429 & 1.000 & end & 0.70 \\
        No ageing                     & ---                   & $\infty$ & 1.000 & --- & --- \\
        Baseline, 0.50 floor          & (10\,000, 0.80, 0.80) & 35\,851 & 0.985 & end & 0.50 \\
        \bottomrule
    \end{tabular}
    \label{tab:ageing_calibrations}
\end{table}
% [AUTHOR] datasheet citations for the two cycle-life statements and for the calendar retention (candidates recorded in
% the brief: LFP calendar-ageing studies; the EVE MB31 datasheet for the 8,000-cycle row).
% [CONFIRM — W175] the mid-block SoH point: SoH_{y-1} e^{-D/2} phi^{Y/2} (shared_energy_storage_data.py:625-626); the
% "no ageing" arm's implementation (k -> infinity and phi = 1, floor inactive); LFP named as the chemistry in the
% ESS file.
```

### C.5 — new 3.6 (paste after 3.5)

```latex
\subsection{\textcolor{blue}{Evaluation and Certification Settings}}
\label{sec:case_settings}

\textcolor{blue}{Every recourse evaluation ran the coordination procedure of \ref{app:admm_updated} under one frozen
configuration. Local problems were solved with IPOPT~3.14.18 through Pyomo~6.9.5 on Python~3.11.11, the network models
with the MA97 linear solver and the storage agent with MA57, single-threaded throughout. Network solves use a
convergence tolerance of $10^{-5}$, an acceptable tolerance of $10^{-4}$, a complementarity tolerance of
$5 \times 10^{-4}$ (TSO) and $10^{-4}$ (DSOs), and at most 500 iterations; the tight tail sets the complementarity
tolerance to $10^{-6}$. The storage agent solves to a tolerance of $10^{-10}$ with an acceptable tolerance of
$10^{-9}$. A solve that ends at the iteration limit, infeasible or in a solver error is retried as a cold restart with
the agent's recovery settings (TSO: acceptable tolerance $10^{-4}$ after one acceptable iteration; DSOs: their primary
settings; storage agent: acceptable tolerance $10^{-9}$ after one iteration) and then once more with the adaptive
barrier strategy.}

\textcolor{blue}{The ADMM settings are: common objective scale $\sigma = 93{,}635{,}360$; reference converter rating
$S^{\text{ref}} = 2.5$~MVA; interface-transformer ratings $R^{\text{I}} = 2.0$, $1.0$ and $1.5$~p.u. (200, 100 and
150~MVA on a 100~MVA base) at TN nodes 5, 7 and 9; storage-agent scale $\kappa^{\text{E}} = \sigma/\operatorname{median}_b
w_b$ ($227{,}211$ at the single-scenario instance, $386{,}259$ at the multi-scenario instance); initial penalties
$\rho^{V}_0 = 0.0077$, $\rho^{\text{PF}}_0 = 0.198$, $\rho^{\text{E}}_0 = 0.01$; residual balancing with ratio 5
(3 for the decrease of $\rho^{\text{PF}}$), factor 1.5, bounds $[10^{-4}, 10^{4}]$, a per-channel freeze after ten
unchanged cycles and a backstop at cycle 200, the storage channel exempt until its dual ratio has been below one on
five consecutive cycles; residual tolerances $\varepsilon_{\text{abs}} = 10^{-5}$, $\varepsilon_{\text{rel}} =
10^{-4}$; Anderson acceleration with memory 5 and regularisation $10^{-10}$. The production exit is ten consecutive
passing cycles, with a cap of 500 cycles at the multi-scenario instance. In the multi-scenario instance the
commitment premium is $\alpha = 0.5$ and the solver-side voltage regularisation weight is $9 \times 10^{4}$.}

\textcolor{blue}{Certification (Subsection~\ref{subsubsec:certification}) uses $\delta_R = 0.07$ on the reference
value $\mathcal{V} = 259{,}375.33$~\euro{}, hence $\tau = 4{,}539.07$~\euro{}, and $P_{\max} = 30$ cycles; an
evaluation continued from an earlier run is capped 100 cycles after that run's stopping cycle, a fresh one at
$\min\{k_0 + 109, 300\}$. The planning search (Algorithm~\ref{alg:shared_ess_planning_mads}) used $\Delta_0 = 4$
lattice units, budgets of 20 new evaluations for the two variant-A searches and 60 for the variant-B certificate, a
completion cap of 30 points, and the measured resolution $\sigma_Q = 18{,}449.66$~\euro{} of the recourse.}

\textcolor{blue}{The value of coordination (Section~4.4) is measured against two static arrangements of the same
system, each evaluated at the same plan and under a no-reverse-flow rule at every interface: a passive arrangement, in
which the DNs activate no flexibility and curtail renewable generation only as the rule requires, and a price-taker
arrangement, in which each DSO dispatches its network against the wholesale price alone. In both, the TSO then
dispatches the TN with the interface exchanges fixed at the DNs' schedules. Each arrangement is reported as the best
of three solver starts.}
% [CONFIRM — W175] the benchmark paragraph against uncoordinated_benchmark.py and Addenda 49/57/58: the passive arm's
% tie-breaker (1 EUR/MWh minimum-curtailment term), the price-taker arm's definition (lambda = 0, rho = 0 — dispatch at
% the market price), the TSO arm with interface P/Q as fixed parameters and bounded voltages, the three-start rule, and
% that the comparison uses the common evaluation Q (tie-breaker 0).
```

---

## D. Planner — next order (zero-solve)

> Addendum 70 and `STEP6_ROUND3_CORRECTIONS.md` are in the repo root. (W174) once the author confirms rounds 2 and 3 are
> in Overleaf: pull the clone, re-run the number check at that HEAD with a new declarations version, re-audit §2.1–2.3
> for the W171b rows only and Appendix A for the W172 rows only, and verify the values of the new §3.5 and §3.6 against
> code, specs and case files one by one. (W175) the `% [CONFIRM — W175]` comments in §3.5 and §3.6 (mid-block SoH point,
> the no-ageing arm, the chemistry, the benchmark arm definitions). Commit and push; nothing else runs.
