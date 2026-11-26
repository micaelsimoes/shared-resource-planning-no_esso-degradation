# Step 6, round 1 — corrections after the Planner's support report (expert, 2026-10-07)

Source: `P5_15_STEP6_SUPPORT_REPORT.md`, `P5_15_W166_EQUATION_CODE_AUDIT.md`, `P5_15_W167_NOMENCLATURE_YEARS_COSTS_REPORT.md`
(repo HEAD `0af00cf0`). Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 68. Manuscript: Overleaf clone as pulled
2026-10-07 after Part D (§2 = lines 315–899 of `main.tex`). Each item is a find → replace in Overleaf; "find" quotes the
start of the current text. All replacement text is blue, as before. The `% [CONFIRM]` comments the audit has answered are
deleted in the replacements; the ones that remain are new and name the Planner task that closes them (W168–W171).

---

## A. Section 2 — edits that bring the pasted text to what the code does

### A.1 — 2.3 intro, the blue "notational compactness" paragraph (find: "For notational compactness, the operational-scenario index")

Replace the whole paragraph by:

```latex
\textcolor{blue}{For notational compactness, the scenario indices are omitted from the detailed TSO, DSO and shared-ESS
formulations. Each network block model holds every scenario of its block jointly, with the objective weighted by the
scenario probabilities; within the block the storage schedule and the TSO's interface exchange are scenario-free, and
the storage agent carries no scenario index at all (Subsection~\ref{subsubsec:commitment}).}
```

### A.2 — 2.3.2 Installed and Available Capacity

(a) Find "The available power of a cohort equals its rating, while its available energy is reduced by its state of
health at the start of the representative year" → replace the sentence by:

```latex
\textcolor{blue}{The available power of a cohort equals its rating, while its available energy is reduced by its state
of health at the end of the representative year --- a conservative convention; the mid-block evaluation point is a
sensitivity in Section~4 ($\forall e$, $y^\text{Inv}$, $y$ as above):}
```

(b) In the equation that follows, change `SoH_{e,y^\text{Inv},y-1}` → `SoH_{e,y^\text{Inv},y}`; delete the
`% [CONFIRM] which SoH enters year y's available energy` comment (answered: baseline 'end').

(c) After the equation `\label{eq:available_capacity}` block, add:

```latex
\textcolor{blue}{The storage agent publishes $S^{\text{Av}}_{e,y}$ and $E^{\text{Av}}_{e,y}$ to every network block at each
coordination cycle; this is the path by which degradation reaches the operating decisions.}
```

### A.3 — 2.3.3 Storage Operation in the Network Models

(a) Replace the equation with label `eq:soc_closure` (and keep its label on the first of the two) by:

```latex
\textcolor{blue}{
    \begin{equation}
        E^{\text{SoC}}_{e,y,d,t_0} = SoC^{0}_e \, E^{\text{Av}}_{e,y}, \qquad
        E^{\text{SoC}}_{e,y,d,|T|} = SoC^{0}_e \, E^{\text{Av}}_{e,y} + s^{+}_{e,y,d} - s^{-}_{e,y,d}, \qquad
        0 \le s^{\pm}_{e,y,d} \le \varepsilon^{\text{Cl}} E^{\text{Av}}_{e,y}
        \label{eq:soc_closure}
    \end{equation}
}
```

(b) Replace the equation with label `eq:converter_capability` by:

```latex
\textcolor{blue}{
    \begin{equation}
        P^{\text{Ch}}_{e,y,d,t} + P^{\text{Dch}}_{e,y,d,t} \le S^{\text{Av}}_{e,y}, \qquad
        \left(P^{\text{E}}_{e,y,d,t}\right)^2 + \left(Q^{\text{E}}_{e,y,d,t}\right)^2 \le \left(S^{\text{Av}}_{e,y}\right)^2
        \label{eq:converter_capability}
    \end{equation}
}
```

(c) Replace the equation with label `eq:network_complementarity` by:

```latex
\textcolor{blue}{
    \begin{equation}
        \frac{P^{\text{Ch}}_{e,y,d,t}}{S^{\text{Av}}_{e,y}} \; \frac{P^{\text{Dch}}_{e,y,d,t}}{S^{\text{Av}}_{e,y}} \le \varepsilon^{\text{C}}
        \label{eq:network_complementarity}
    \end{equation}
}
```

(d) Replace the "where $\eta^{\text{Ch}}_e$ and $\eta^{\text{Dch}}_e$ are the charging and discharging efficiencies…"
paragraph by:

```latex
\noindent
\textcolor{blue}{where $\eta^{\text{Ch}}_e$ and $\eta^{\text{Dch}}_e$ are the charging and discharging efficiencies,
$SoC^{\text{Min}}_e$, $SoC^{\text{Max}}_e$ and $SoC^{0}_e$ the minimum, maximum and initial states of charge as fractions
of the available energy, and $\Delta t$ the hour length. Equation~\eqref{eq:soc_closure} starts every representative day
at the same state and closes it there up to a slack bounded by a small fraction $\varepsilon^{\text{Cl}}$ of the
available energy; the slack is penalised at $c^{\text{Cl}}$ per MWh in the block objective, so that any closure
shortfall is paid for inside $Q(\boldsymbol{x})$. No energy is carried between representative days, and cumulative
discharge beyond the stored energy is excluded in every day and scenario by
\eqref{eq:soc_recursion}--\eqref{eq:soc_closure}. Reactive power is a converter quantity: it is limited with the active
power by the capability circle in \eqref{eq:converter_capability} and has no effect on the stored energy or on ageing.
Charging and discharging are kept apart by the relaxed, normalised complementarity condition
\eqref{eq:network_complementarity} with a small tolerance $\varepsilon^{\text{C}}$. In the multi-scenario instance the
storage schedule of a block is a single scenario-free variable referenced by every market and operation scenario's
balance equations --- a day-ahead commitment --- so that the schedule the agent ages is the one every scenario runs
(Subsection~\ref{subsubsec:commitment}). The values of $\eta$, $SoC^{\text{Min}}$, $SoC^{\text{Max}}$, $SoC^{0}$,
$\varepsilon^{\text{Cl}}$, $c^{\text{Cl}}$ and $\varepsilon^{\text{C}}$ are given in Section~3.5.}
% [CONFIRM — W169] whether the closure slack was active in any reported evaluation; if it never was, add: "The closure
% slack was inactive in every evaluation reported in this paper."
```

(e) Delete the old `% [CONFIRM] against sess_soc_rule and the network ESS block` comment (answered). Values for 3.5:
$\eta^{\text{Ch}} = 0.97$, $\eta^{\text{Dch}} = 0.96$, $SoC^{\text{Min}} = 0.10$, $SoC^{\text{Max}} = 0.90$, $SoC^{0} = 0.50$,
$\varepsilon^{\text{Cl}} = 0.05$ (plus $10^{-5}$ p.u. numerical slack), $c^{\text{Cl}} = 10^{3}$~€/MWh, $\varepsilon^{\text{C}} = 10^{-4}$.

### A.4 — 2.3.4 Degradation Chain of the Shared-ESS Agent

(a) Replace the first paragraph ("The shared-ESS agent holds the consensus copy of the storage schedule and ages the
cohorts with it. Its schedule is allocated…") and the equation `eq:cohort_allocation` by:

```latex
\textcolor{blue}{The shared-ESS agent holds its own copy of the storage schedule, in consensus with the network models'
copies through the ADMM channel of \ref{app:admm_updated}, and ages the cohorts with it. Its net active power at node
$e$ is the sum of the cohorts' charging and discharging powers, up to a slack pair that is penalised in its objective and
is never active at a consensus point within the agent's ratings ($\forall e$, $y$, $d$, $t$):}

\textcolor{blue}{
    \begin{equation}
        P^{\text{Net}}_{e,y,d,t} = \sum_{y^\text{Inv}} \left( P^{\text{Ch}}_{e,y^\text{Inv},y,d,t} - P^{\text{Dch}}_{e,y^\text{Inv},y,d,t} \right)
        + \sigma^{+}_{e,y,d,t} - \sigma^{-}_{e,y,d,t}, \qquad \sigma^{\pm}_{e,y,d,t} \ge 0
        \label{eq:esso_net_power}
    \end{equation}
}

\textcolor{blue}{When several cohorts are alive at a node, the net schedule is allocated to them in proportion to their
rated energy --- a homogeneous-fleet approximation in which all cohorts at a node cycle at the same depth; every plan
reported in this paper has a single cohort per node, for which the allocation is vacuous ($\forall e$, $y^\text{Inv}$,
$y$, $d$, $t$):}

\textcolor{blue}{
    \begin{equation}
        P^{\text{Ch}}_{e,y^\text{Inv},y,d,t} - P^{\text{Dch}}_{e,y^\text{Inv},y,d,t} =
        \frac{E^{\text{Rated,Unit}}_{e,y^\text{Inv},y}}{\sum_{y'} E^{\text{Rated,Unit}}_{e,y',y}}
        \sum_{y''} \left( P^{\text{Ch}}_{e,y'',y,d,t} - P^{\text{Dch}}_{e,y'',y,d,t} \right)
        \label{eq:cohort_allocation}
    \end{equation}
}
```

Delete the `% [CONFIRM] H3 rule` comment (answered: pro-rata on the net power; inactive for single-cohort plans).

(b) The throughput, loss, chain, floor and calibration equations stand as pasted (the audit matches them: cell-side
energy with $\eta = 0.97/0.96$, $k = 35{,}851$, $\exp(-D)\,\phi^{Y}$ with $\phi = 0.985$, floor 0.70). Delete the
`% [CONFIRM] (a) the row D*(2*cl_eff*E_rated)…` comment. One wording fix in the paragraph after the floor equation:
"Equations \eqref{eq:cycling_loss}--\eqref{eq:soh_floor} are linear in $D$ and $SoH$ for a fixed schedule, and
\eqref{eq:soh_chain} is the only nonlinear row of the agent." → "Equations \eqref{eq:cycling_loss} and
\eqref{eq:soh_floor} are linear; \eqref{eq:soh_chain} and the available-energy product in \eqref{eq:cohort_available}
are the agent's nonlinear rows."

### A.5 — 2.3.5 Objective of the Shared-ESS Agent

Replace the subsection body (the sentence "The agent has no economic objective…", the equation and the "subject to…"
paragraph) by:

```latex
\textcolor{blue}{The agent has no economic objective, and its objective value is not part of $Q(\boldsymbol{x})$. Within
the coordination procedure it minimizes the augmented-Lagrangian terms of the storage consensus channel
(\ref{app:admm_updated_implementation}) plus a feasibility penalty on the slack pair of \eqref{eq:esso_net_power} and a
small linear regularization of throughput,}

\textcolor{blue}{
    \begin{equation}
        \begin{aligned}
            \min \;
            & c^{\sigma} \sum_{e \in E^S} \sum_{y \in Y} \sum_{d \in D} \sum_{t \in T}
            \left( \sigma^{+}_{e,y,d,t} + \sigma^{-}_{e,y,d,t} \right)
            + \varepsilon^{\text{E}} \sum_{e \in E^S} \sum_{y^\text{Inv} \in Y} \sum_{y \in Y} \sum_{d \in D} \sum_{t \in T}
            \left( P^{\text{Ch}}_{e,y^\text{Inv},y,d,t} + P^{\text{Dch}}_{e,y^\text{Inv},y,d,t} \right) \\
            & + \sum_{e \in E^S} \sum_{y \in Y} \sum_{d \in D} \sum_{t \in T}
            \left( \mathcal{L}^{E,P}_{e,y,d,t} + \mathcal{L}^{E,Q}_{e,y,d,t} \right)
        \end{aligned}
        \label{eq:esso_objective}
    \end{equation}
}

\noindent
\textcolor{blue}{subject to \eqref{eq:cohort_rated}--\eqref{eq:available_capacity},
\eqref{eq:esso_net_power}--\eqref{eq:soh_floor}, $0 \le P^{\text{Ch}}_{e,y^\text{Inv},y,d,t},\,
P^{\text{Dch}}_{e,y^\text{Inv},y,d,t} \le S^{\text{Rated,Unit}}_{e,y^\text{Inv},y}$ and the converter capability
inequality $(P^{\text{Net}}_{e,y,d,t})^2 + (Q^{\text{Net}}_{e,y,d,t})^2 \le (S^{\text{Av}}_{e,y})^2$ on the agent's
aggregate schedule. The regularization weight $\varepsilon^{\text{E}}$ is far below the resolution of
Subsection~\ref{subsubsec:certification} and serves one purpose: with a linear, strictly positive price on
$P^{\text{Ch}} + P^{\text{Dch}}$, simultaneous charging and discharging is never optimal, so complementarity in the agent
follows from the problem structure and needs no nonconvex constraint; the residual $\min\{P^{\text{Ch}}, P^{\text{Dch}}\}$
is checked after every solve. The earlier investment-fixing slacks and the apparent-power aggregation are retired:
investments are parameters, and reactive power enters only through the capability circle. The values of $c^{\sigma}$
and $\varepsilon^{\text{E}}$ are given in Section~3.5.}
% [CONFIRM — W169] the slack pair sigma inactive at every certified point; the post-solve min(pch, pdch) detector in the
% production path (Addendum 3 remedy (h)).
```

Values for 3.5: $c^{\sigma} = 10^{3}$, $\varepsilon^{\text{E}} = 10^{-5}$. Delete the old `% [CONFIRM] (a) the value of
EPS_ESSO_THROUGHPUT…` comment.

### A.6 — 2.3.6 Day-Ahead Commitment under Operation Scenarios

(a) In the first paragraph, "while the interface schedule committed to the TSO is scenario-independent by construction
in the TSO model and may be deviated from in the DSO model." → "while the interface schedule committed to the TSO is
scenario-free by construction in the TSO model, which re-dispatches its own network per scenario against it, and may be
deviated from on the DSO side."

(b) In the closing "where $\bar{\pi}_t$ is…" paragraph, replace "The TSO operates on the committed schedule and carries no
deviation term; the interface voltage carries none either, being a physical state rather than a commitment." by:

```latex
The deviation energy is also settled at the scenario price: the contracted part of the interface settlement, at the
committed schedule, is a transfer between operators and is excluded from $Q(\boldsymbol{x})$, while the deviation part is
inside it (Subsection~\ref{subsubsec:certification}). The TSO carries no deviation term. The interface voltage carries
no priced term either, being a physical state rather than a commitment; a solver-side regularisation of its dispersion
is used and excluded from $Q(\boldsymbol{x})$ (\ref{app:admm_updated}).
```

(c) Delete the `% [CONFIRM] (a) alpha = 0.50…` comment (answered: α = 0.5, DSO side, P and Q, expectation is the block's
own variable).

### A.7 — 2.2.7 Recourse Evaluation and Certification

(a) Replace the "where $C^{\text{Op}}_{y,d,m,o}$ is the daily operating cost…" paragraph by:

```latex
\noindent
\textcolor{blue}{where $C^{\text{Op}}_{y,d,m,o}$ is the daily operating cost of representative day $d$ of representative
year $y$ under market scenario $m$ and operation scenario $o$ at the consensus point reached by the coordination
procedure of \ref{app:admm_updated}: generation cost in the TN, activated flexibility and curtailment in the DNs, the
closure-slack penalty of the storage model (Subsection~\ref{subsubsec:network_storage}) and, in the multi-scenario
instance, the commitment-deviation charge and the deviation part of the interface settlement
(Subsection~\ref{subsubsec:commitment}). The contracted settlement at the committed interface schedule is a transfer
between operators and is excluded, as are the storage agent's auxiliary objective and the solver-side regularisation
terms; what is excluded is reported separately. Each representative year stands for a block of $Y_y$ calendar years, and
its cost is weighted by $Y_y$ and discounted at the representative year.}
```

Delete the `% [CONFIRM] discount convention…` comment (answered: block weight $Y_y D_d\,1.02^{-(y-y_0)}$).

(b) In "Stopping and certification": "The residual pass at cycle $k_0$ therefore only opens the certifying regime, which
is then held fixed:" → "The residual pass at cycle $k_0$ therefore only opens the certifying regime, which is held
fixed from the next cycle on:". After "An evaluation that satisfies neither branch within its cycle cap is reported as
uncertified, with the objective's movement since the residual pass (its settling slack) and its consensus gap." add:

```latex
The rule above, with its holds, produced every single-scenario evaluation reported in this paper. The multi-scenario
instance was evaluated under the production exit --- ten consecutive cycles passing the residual test with every local
solve successful --- and then continued past it; its values are reported as a band rather than a certificate
(Section~4.5).
```

(c) Symbol clashes with the rest of the manuscript ($W$, $V$ are taken): in item 3 of the list, `$W = \max\{20, \lceil
1.1\,\hat{P} \rceil\}$` → `$n_{\mathrm{w}} = \max\{20, \lceil 1.1\,\hat{P} \rceil\}$`; in "Resolution", `$V$ the storage
value of the reference unit` → `$\mathcal{V}$ the storage value of the reference unit` and `($V = 259{,}375.33$~\euro{}`
→ `($\mathcal{V} = 259{,}375.33$~\euro{}`; `$\tau = \delta_R V / 4$` → `$\tau = \delta_R \mathcal{V} / 4$`.

### A.8 — 2.3.2, cohort window (notation only)

In the sentence before `eq:cohort_rated`, "($\forall e \in E^S$, $\forall y^\text{Inv} \in Y$, $\forall y \in
[y^\text{Inv};~y^\text{Inv}+T^\text{Cal}_{e,y^\text{Inv}}]$)" → "($\forall e \in E^S$, $\forall y^\text{Inv} \in Y$,
$\forall y \in [\,y^\text{Inv},\; y^\text{Inv}+T^\text{Cal}_{e,y^\text{Inv}})$)" (half-open: a unit installed in
$y^\text{Inv}$ lives for $T^\text{Cal}$ years), and in the sums of `eq:available_capacity` the lower limit
`\max(y-T^\text{Cal}_{e,y},\,y_0)` → `\max(y-T^\text{Cal}_{e,y}+1,\,y_0)`. Same two changes in 2.2.2 (both sums).

### A.9 — Nomenclature (at the final pass, unchanged ruling): add $n_{\mathrm{w}}$, $\mathcal{V}$, $\sigma^{\pm}$,
$c^{\sigma}$, $s^{\pm}$, $\varepsilon^{\text{Cl}}$, $c^{\text{Cl}}$, $\varepsilon^{\text{C}}$, $P^{\text{Net}}$,
$Q^{\text{Net}}$ to the list at the end of `section2_expert_draft.tex`; remove $W$, $V$ from it.

---

## B. Response letter — edits (Decision 1 and Decision 5)

B.1 **R3.3** — replace the response by:

```latex
\rresponse{The equations the reviewer refers to belong to the shared-ESS agent, whose role is the degradation chain:
capacity fade by cycling is driven by the cell-side charging and discharging throughput, so no state of charge is
needed there and none appears. The state of charge is a variable of the TSO and DSO network models, where the storage is
dispatched, and the complete formulation is now printed (Section~2.3.3; see also R2.11--R2.12). On continuity: within
each block (one representative year and one representative day) the shared storage has a single charging, discharging
and state-of-charge schedule, common to all market and operation scenarios --- in the network models it is one
scenario-free variable per hour referenced by every scenario's power balance, and the storage agent has no scenario
index --- so the schedule is non-anticipative by construction rather than by a penalty, and the state-of-health chain is
driven by this one schedule. Scenario deviations are absorbed on the network side: the TSO dispatches its own network
per scenario against the scenario-free interface schedule, and each DSO's substation import varies by scenario, its
deviation from the DSO's own expected import being charged at a premium on the hour's expected market price
(Section~2.3.6). The state of charge evolves over the hours of each representative day with charging and discharging
efficiencies within stated limits of the available energy, starts every day at the same level and closes there up to a
penalised slack bounded by a small fraction of the available energy, so no energy is carried between representative
days; continuity across representative years is carried by the state-of-health chain, which maps the realized
cell-side throughput of each year into the available energy of that year and the next. Cumulative discharge beyond the
stored energy is therefore excluded by the energy limits and the closure condition in every day and scenario.}
```

and delete the `% [CONFIRM against code before submission…]` comment under it.

B.2 **R2.11–R2.12** — "The paper now also says where each constraint is enforced: the SOC dynamics, energy limits and
capability inequality sit in the TSO and DSO network models for every representative day and scenario;" → "The paper
now also says where each constraint is enforced: the SOC dynamics, energy limits and capability inequality sit in the
TSO and DSO network models, once per representative day and common to the scenarios of that day, and the capability
inequality is enforced in the shared-ESS agent as well;".

B.3 **R2.2** — "within an evaluation, every ADMM cycle solves one local problem per network block (three representative
years $\times$ four representative days $\times$ four agents at the single-scenario instance, 48 blocks, with the
scenarios inside each block)" → "within an evaluation, every ADMM cycle solves one local problem per network block
(three representative years $\times$ four representative days $\times$ four network operators at the single-scenario
instance, 48 blocks, with the scenarios inside each block) and one per shared-ESS agent (three, one per interface node,
each spanning all years and days)".

B.4 **R3.1** — "It changes no sign in the 60 reported comparisons and one verdict (a Phase~B neighbour becomes
determinate in the certificate's favour). The investment-year comparison is the one result that depends on the
convention" → "It changes no sign in the 60 reported comparisons and two verdicts: a two-node plan evaluated under the
doubled flexibility price becomes determinate in the certificate's favour, and the investment-year comparison is the
other comparison that depends on the convention".

B.5 **R3.6** — "harsher calibrations $-73.7$ and $-45.2$~k\euro{}; all determinate except the first (Table~[T8])" →
"the datasheet-exact calibration $-45.2$~k\euro{}, the mid-block evaluation point $-65.9$~k\euro{} and the
unit-retention reading $-73.7$~k\euro{}; all determinate except the no-ageing case (Table~[T8])". And "On terminal
available capacity: Table~[T8] gives, per calibration, the year in which the end-of-life floor binds and the terminal
available-energy fraction, together with the equivalent full cycles per day." → "On installed capacity over time:
Table~[T8] gives, per calibration, the year in which the end-of-life floor binds (within the horizon it never does) and
the present-value-weighted available energy, together with the equivalent full cycles per day."

B.6 **R1.2** — "on the latter the planning conclusion is unchanged and the storage value is 0.909--0.934 of its
single-scenario value, against 0.933 predicted from the mean-profile price spread (Section~4.5)" → "on the latter, run
on a finer five-block horizon, the planning conclusion is unchanged and the storage value is 0.909--0.934 of its
single-scenario value, against 0.933 predicted from the mean-profile price spread (Section~4.5)". And in R3.3/R2.2
wherever "three representative years standing for five-year blocks" is stated as the instance, add "(single-scenario
instance; the multi-scenario instance uses five three-year blocks)".

B.7 **R1.5** — "The framework figure was redrawn…" → "The framework figure (Figure~1 of the revised manuscript) was
redrawn…".

B.8 **R2.8** — the status note stays until the two references are added; then the sentence reads "The three references
were added…" with the three clauses.

B.9 Title "revision 1" and "our first reply" are consistent; no change.

---

## C. Planner — next order (zero-solve unless stated), to forward after the §2 corrections are in Overleaf

> Addendum 68 is in the brief and `STEP6_ROUND1_CORRECTIONS.md` is in the repo root. Zero-solve tasks, in this order:
> (W168) replay the settling decision of the x = 0 reference `ref:7aa017f0` under rule v6 on its committed per-cycle
> record (W101); report whether and at which cycle it certifies, the window range, and the certified value's difference
> to the tabulated one; if the record ends before v6 can decide, stop and report the state at its end — no continuation
> without a ruling. (W169) from the committed records of every evaluation in T1–T11: the maximum closure slack
> (s⁺ + s⁻, MWh and fraction of E^Av) over all blocks and the maximum ESSO P-net slack (σ⁺ + σ⁻) over all (e, y, d, t)
> at the certified point; say which records carry them and which do not. (W170) state from the prediction record how
> the 0.933 mean-profile prediction for the 3 × 3 ratio was computed — which instance's profiles and horizon — in three
> lines. (W171) once the author confirms the §2 corrections are in Overleaf: pull the clone, re-run W164 with a new
> declarations version at that HEAD, and audit the revised §2 (lines 315–899 and the `% [CONFIRM — W16x]` comments)
> against the code in the W166 form, rows only where the revised text differs from the code; this replaces the W166
> table for §2.3. Appendix A is not audited again until the expert's draft of it is pasted. Commit and push; nothing
> else runs.

Predictions recorded (expert): W168 — the reference certifies under v6 at a cycle in [174, 195] with a window range
≤ τ and a value within 0.93 τ of the tabulated one; W169 — both slacks are zero to solver tolerance (≤ 10⁻⁶ relative)
at every certified point; W171 — no H-impact row remains in §2.3 after the corrections.
