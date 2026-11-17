# Step 4 — Derivative-free master search: method definition

**External expert, 2026-09-19, v1.** Companion to `PLANNER_BRIEF_2026-09-13.md` (Addendum 26 records
the author's decisions this document implements). This is the specification the Planner writes the
Step 4 spec and the Worker's tasks from; the manuscript's method section is derived from §1, §3 and
§8. Nothing here authorizes a run: authorization stays with the brief's addenda.

Author's decisions taken (2026-09-19): capacities are granular at **0.25 MVA** (power) and
**0.5 MWh** (energy); duration bounded to **2 h ≤ E/P ≤ 4 h**; the **investment year is a
decision variable**.

---

## 1. The master problem

### 1.1 Decision variables

Let $N = \{5, 7, 9\}$ be the interface nodes and $Y = \{y_1 < \dots < y_{|Y|}\}$ the representative
years of the instance (SRP1: 2025, 2030, 2035). The model already carries one investment per node and
year (`es_s_investment[e, y]`, `es_e_investment[e, y]`) and builds a cohort per investment year (H3
pro-rata split). The decision vector is therefore, in its general form,

$$x = \big(P_{n,y},\, E_{n,y}\big)_{n \in N,\ y \in Y}, \qquad P_{n,y} \in 0.25\,\mathbb{Z}_{\ge 0}\ \text{MVA},\quad E_{n,y} \in 0.5\,\mathbb{Z}_{\ge 0}\ \text{MWh},$$

$2|N||Y|$ granular variables (18 on SRP1). The **single-cohort restriction** — at most one investment
year per node, $x = (P_n, E_n, y_n)_{n\in N}$ with $y_n \in Y$ ordinal — has $3|N|$ variables (9) and is
used for Phase A ladders and as the Phase B default when the budget table (§7) requires it. Both live on
the same lattice; a single-cohort point is a general point with one nonzero year per node.

### 1.2 Constraints (all enforced before evaluation; no oracle call for an infeasible $x$)

- Duration: $2\,P_{n,y} \le E_{n,y} \le 4\,P_{n,y}$ whenever $P_{n,y} > 0$; $P_{n,y} = 0 \Leftrightarrow E_{n,y} = 0$.
  On the lattice this admits, e.g., $P = 0.25 \Rightarrow E \in \{0.5, 1.0\}$; $P = 0.5 \Rightarrow E \in
  \{1.0, 1.5, 2.0\}$. The case file's `max_energy_to_power_factor` goes from 10 to 4;
  `min_energy_to_power_factor` stays 2.
- Bounds: `max_capacity` caps **energy** at $E^{\max}$ = 5 MWh per node (production semantics,
  `shared_energy_storage_data.py`; kept by the author); with the duration bound this implies
  $P_{n,y} \le 2.5$ MVA. Per node the lattice therefore has $E \in \{0.5, \dots, 5.0\}$ and the
  admissible $P$ for each $E$.
- Budget: $I(x) \le B$ with $B$ = `budget` (€1e6). **Both constraints live only in the retired Benders
  master; they have never been on the oracle's path**, so they are the master search's to enforce, in
  closed form, before any evaluation. With the committed cost file (2025 expected unit costs
  ≈ €256k/MVA, ≈ €203k/MWh) the paper's own plan costs €1,073k and is **over** the budget; with the
  corrected file on `paper_revisions` (energy costs × 1.25) every reference candidate is over it. Under
  a binding budget the master problem is an **allocation** problem (node, duration, year, split) and
  the claim is "optimal under the budget"; without it the search finds the size at which the marginal
  value of storage meets its marginal cost, bounded by `max_capacity`. **Author's decision pending
  (Addendum 27):** the recommended design is a primary Phase B without the budget and a second,
  cheaper Phase B under €1M sharing the cache, reported as the budget-constrained plan. The Phase A
  single-node ladders (§4.1) run without the budget in either case.

### 1.3 Objective

$$F(x) = I(x) + Q(x),\qquad
I(x) = \sum_{n,y} \frac{1}{(1+d)^{\,y - y_1}} \sum_{m} \omega_m \big( c^S_{m,y} P_{n,y} + c^E_{m,y} E_{n,y} \big),$$

with $d$ the discount factor (0.02), $\omega_m$ the investment-cost scenario weights (0.35/0.55/0.10)
and $c^S_{m,y}, c^E_{m,y}$ the per-year unit costs from `SRP1_ESS.xlsx` — exactly the expression in
`shared_energy_storage_data.py` (`model.investment_cost`). Salvage is excluded until the parked
decision is taken; if reinstated it enters $I(x)$ in closed form, not the oracle. $Q(x)$ is the
**certified** expected operational cost (`gross_operational_cost`) returned by the frozen ADMM
configuration (§2). $I$ is closed-form and cheap; every evaluation cost is in $Q$.

### 1.4 What $Q$ is, mathematically

$Q(x)$ is the value of a certified consensus point of a nonconvex problem, computed by a deterministic
procedure. It is a well-defined function of $x$ (§2.3) but it is not continuous: the ADMM can select a
different local operating point at neighbouring $x$, and Step 3 measured the between-configuration
spread of such points at one candidate as **1.1e-4 of $Q$** (≈ €72k on SRP1). That figure is the
provisional **resolution** $\sigma_Q$ of the oracle; Phase A ladders re-measure it along the lattice
(§4.3). Investment steps on the lattice are ≈ €64k (0.25 MVA) and ≈ €127k (0.5 MWh) at corrected 2025
costs. **Measured (Phase A, Addendum 28):** $\sigma_Q \approx$ 10–18k, well below one lattice step, so
the lattice is resolvable and §5.2's degradation clause is not triggered. Phase A also found $Q$
**affine in $x$ within $\sigma_Q$** on SRP1 (value 227.7k €/MWh, 51.7k €/MVA, additive across nodes):
the storage is a price-taker at these capacities, so the optimum under any budget is a corner and
Phase B is short by nature. The method is unchanged by this; its value is the certificate and the
record, and the nontrivial case is the ageing-sensitivity baseline under which storage pays.

---

## 2. Oracle contract

1. **Frozen configuration.** One ADMM configuration for the whole campaign (Addendum 22 rule): arm D,
   or AA-on if the selection run adopts it (Addendum 25). It never changes during a campaign; costs from
   different configurations never share a table.
2. **Cold start, always.** Every candidate is evaluated from the standard cold initialization. Warm
   continuation between candidates (Addendum 17 §7 design) is **rejected for the campaign**: it makes
   $Q(x)$ depend on the evaluation history, which breaks determinism, cache validity and the
   reproducibility statement. It may be used only in a clearly labelled verification experiment.
3. **Determinism.** $Q(x)$ is bitwise reproducible for a given $x$ (τ = 0; three reproductions of D). The
   campaign harness's own bitwise gate (Addendum 25) extends this to concurrent evaluation.
4. **Certification and the barrier.** An evaluation returns $Q(x)$ only if it certifies under the
   10-consecutive-cycle bar within the cap (500 cycles). A non-certified evaluation, an unrecoverable local
   solve failure, or an infeasible ESSO (SoH floor) returns $F(x) = +\infty$ — the **extreme barrier** of
   MADS (Audet & Dennis 2006) — and is recorded with its cause and its last residuals. A run of barrier
   points in a region stops the campaign for review; it is a robustness finding, not noise.
5. **Per-evaluation record:** certified cost with its bar (max objective step over the last 10 cycles),
   certification cycle, per-channel terminal ratios and rule ten (reported, not gated), component
   decomposition (generation, internal flexibility, curtailment, …), settlement remainder, storage EFC
   and terminal SoH per node, wall time, peak RSS.
6. **Cache.** Keyed by the canonical $x$ (§5.4). A cache hit never re-evaluates. The Phase A results
   seed the Phase B cache.
7. **Evaluation interface.** `evaluate(batch: list[x]) -> list[record]`, batch size = number of concurrent
   slots; each $x$ in its own process (Addendum 25 harness).

---

## 3. The algorithm: MADS with granular variables

The master search is the **Mesh Adaptive Direct Search** framework (Audet & Dennis 2006) with the
**OrthoMADS** poll (Abramson, Audet, Dennis & Le Digabel 2009) and the **granular-variable mesh**
(Audet, Le Digabel & Tribes 2019). Each iteration has two steps:

- **Search** — any finite set of mesh points. Phase A (§4) is the initial search: a designed screen
  that also produces the paper's structural results. In Phase B the search is optional and cheap: a
  quadratic model (NOMAD's default) or the Benders-type local model (§5.6).
- **Poll** — the $2n$ OrthoMADS directions (or the $n+1$ minimal positive basis when the budget table
  says so) around the incumbent at the current poll size. A success moves the incumbent and coarsens
  the mesh; a failure refines it.

Working in the scaled variables $z = D^{-1}(x - x^{\mathrm{ref}})$ with $D = \mathrm{diag}(0.25, 0.5, \dots)$ turns
the lattice into $\mathbb{Z}^n$; mesh and poll sizes are integers in $z$ and the mesh cannot refine below 1.
Termination is **mesh-local optimality on the lattice**: the poll at unit poll size fails, i.e. every
one of the $2n$ (or $n+1$) unit-step neighbours of the incumbent is evaluated and none improves $F$ —
or the evaluation budget is exhausted, in which case the incumbent is reported with the poll size
reached.

Convergence theory (Clarke-stationarity in the continuous case; mesh-local optimality for granular
variables) applies to $F$ as a function on the lattice with the extreme barrier; it does not require
smoothness or continuity of $Q$, which is why the method was chosen over gradient-based or cut-based
masters for this oracle.

---

## 4. Phase A — designed initial search

Phase A is the first search step of MADS: every point is a lattice point and enters the cache. It has
three purposes — a good incumbent for Phase B, the paper's structural results, and the empirical
resolution of the oracle.

### 4.1 A1: single-node ladders (single-cohort)

**A0 — robustness batch (first, one batch):** $x = 0$; the smallest lattice unit at each node
(0.25 MVA with 0.5 and 1.0 MWh); the lattice plan (1.5 MVA / 3.0 MWh at node 7) — 8 evaluations
under the adopted configuration. It extends the selection run's certification evidence (which covered
C\*-sized and larger designs) to the sub-MVA, single-node region the campaign actually searches, and
$x = 0$ is the reference every value-of-storage figure needs. A non-certified point here stops for
review.

**A1 — ladders (reduced for cost, author's decision).** For each node $n$ alone (other nodes zero),
investing in $y_1$ = 2025: energy ladder $E \in \{1, 2, 3, 4, 5\}$ MWh at 2 h ($P = 0.5 \dots 2.5$) and
at 4 h ($P = 0.25 \dots 1.25$) — the whole `max_capacity` range — **30** points. Then the year ladder
(2030, 2035) at the best node and duration only: **20** points. **The budget is not applied in A1**:
the ladders are the sensitivity study behind the budget, and the levels above it show what it
forgoes. Output: value-of-storage curves $Q(0) - Q(x)$ against $I(x)$ per node and duration, and the
timing curve at the best node — the paper's Figure "value of shared storage by size, duration, timing
and node", with the budget marked. $Q(x)$ does not depend on the cost file, so A0 and A1 can run
before the corrected file is on the branch; only $I(x)$ and everything from A2 on depend on it.

### 4.2 A2: combinations and staging

- The best single-node setting per node, combined in a $2^{|N|}$ presence design (8 points, 4 new).
- Staging at the best node: half the best capacity in $y_1$ plus half in $y_2$ (two cohorts), and the
  full capacity in $y_2$ alone — 2 points; this is the first evidence on whether the cost decline
  (2025 → 2030 ≈ −16 % nominal, ≈ −24 % in present value) outweighs the operating value foregone.
- C\* rounded to the lattice (1.0 MVA / 4.0 MWh at all nodes) — 1 point (the lattice plan is in A0).
  Total Phase A ≈ 110 evaluations with A0 and §4.3.

### 4.3 A3: resolution probe

Folded into A1's best ladder: the unit-step points between its levels ($E$ at 0.5 MWh steps around
the budget-feasible levels, and one $P$ step at fixed $E$) — about 4 new points. The largest
non-monotone jump in $Q$ along the ladder, relative to the smooth trend, is the **measured**
$\sigma_Q$ and replaces the provisional 1.1e-4. It sets the Phase B initial mesh (§5.2) and the wording
of the final claim (§8). Phase A total ≈ 70 evaluations, ≈ 7 batches of 10.

---

## 5. Phase B — OrthoMADS poll with optional surrogate search

### 5.1 Variables

Default: the general lattice $(P_{n,y}, E_{n,y})$ — the paper's own decision space (18 variables on
SRP1). Fallback if the budget table (§7) requires it: the single-cohort restriction (9 variables), with
staging checked afterwards by one A2-style probe at the incumbent.

### 5.2 Mesh and poll sizes

Initial poll size in scaled units: $\Delta^p_0 = 4$ (≈ 2 MWh in energy, ≈ 1 MVA in power); in the
in-house implementation the poll size doubles on a successful poll and halves on a failed one, with
the lattice step as its floor (neither size ever falls below 1 scaled unit). **Correction (Addendum
35):** this is *not* NOMAD 4's rule — NOMAD 4.6 uses a 1-2-5 frame ladder and random rather than
Halton directions; the direction construction, lattice rounding, bound handling and the $(n{+}1)$-th
direction were verified equal to NOMAD on 144 logged polls, the frame rule and the completion step
are ours. If
$\sigma_Q$ from §4.3 exceeds one lattice step's investment cost, the campaign's *reported* optimum is the
incumbent at the coarsest poll size whose step cost exceeds $2\sigma_Q$, and the finer polls are
reported as an inconclusive neighbourhood (§8).

### 5.3 Directions and parallelism

OrthoMADS $2n$ directions from the Halton sequence (deterministic; seed recorded), or the $n+1$ basis.
**Full poll, in batches:** all directions are evaluated (no opportunistic stop), in batches of the
concurrent-slot count; the incumbent is updated after the whole poll. With 10 slots: $n+1 = 19$ is two
batches, $2n = 36$ is four. Ordering within the poll by the surrogate model's prediction (§5.6) when
available.

### 5.4 Canonicalization and cache

A lattice point is stored in canonical form: entries with $P_{n,y} = 0$ have $E_{n,y} = 0$; in the
single-cohort form, $y_n := y_1$ whenever $P_n = 0$. Two points with the same canonical form are the
same evaluation. Poll directions that only change inactive entries (a year of a zero-capacity node) are
dropped before evaluation.

### 5.5 Infeasible poll points

Bound, duration, budget: rejected before evaluation (extreme barrier at zero cost). Oracle barrier
(§2.4): recorded; the direction counts as evaluated.

### 5.6 Search step: Benders-type local model (optional)

At each evaluated $x_j$ the oracle can return the capacity sensitivities $g_j = \partial Q/\partial x$
(the ESSO's multipliers on the capacity hand-off, if the Planner confirms they are available without
extra solves; otherwise this step is omitted). Because $Q$ is nonconvex these cuts are not valid lower
bounds; they are used only as a **local model**
$\hat Q(x) = \max_{j \in \mathcal{J}}\{Q(x_j) + g_j^{\top}(x - x_j)\}$ over the $\mathcal{J}$ points within
the trust region $\|z - z^{\mathrm{inc}}\|_\infty \le 2\Delta^p$. The search point is the lattice minimizer of
$I(x) + \hat Q(x)$ in that region (a tiny MILP or enumeration), one evaluation per iteration. It keeps
the manuscript's link to the original Benders-like idea in the honest form: cuts as a search heuristic
inside a globally convergent direct-search frame, with no gap claim. Alternative: NOMAD's built-in
quadratic-model search on the cache.

---

## 6. Implementation

**Recommended: NOMAD 4** through its Python interface (PyNomad; Audet, Le Digabel, Rochon Montplaisir &
Tribes 2022, ACM TOMS "Algorithm 1027"). It implements OrthoMADS, granular variables
(`GRANULARITY`), integer variables, bound constraints, the extreme barrier (`EB` outputs), block
evaluation (`BB_MAX_BLOCK_SIZE` = slot count, which maps onto the campaign harness's batch interface), a
cache file (Phase A seeds it) and the quadratic-model search. Gate before adoption: (i) it installs in
`opf_env_py311` on Apple silicon; (ii) a two-iteration run on a stub objective reproduces the documented
OrthoMADS directions; (iii) the batch interface returns the harness records unchanged. **Fallback:** an
in-house OrthoMADS (Halton directions, granular mesh, full poll) — small, but it must be tested against
NOMAD on a stub before use, and it forfeits the citation.

Whichever is used, the master loop must be **resumable** (cache + incumbent + mesh state on disk, frozen
spec, one campaign-level lock), and every evaluation's record and the poll history are committed with
hashes, as in Step 3.

---

## 7. Budget and instance choice

**Measured (Addendum 25 handoff), replacing the earlier block-count scaling model.** Scenarios live
*inside* each network block (5 market × 5 operation combinations per block), not across blocks: the
paper instance is 80 blocks (5 years × 4 days × TSO + 3 DSOs), each ≈ 25× larger than its SRP1
counterpart (a DSO block ≈ 269k variables). Build-only memory is 18.3 GiB for the models; the pristine
snapshot clones (diagnostics, not method) push it past 24 GiB. Per-cycle time at that size is
**unmeasured**; SRP1's 35 s/cycle does not transfer, because a 25× larger NLP costs more than 25× in
IPOPT.

| instance | blocks | mem / eval | slots on 32 GB | $t_{\mathrm{cycle}}$ | $T_{\mathrm{eval}}$ (AA, ≈ 107 cycles) |
|---|---|---|---|---|---|
| SRP1: 3 × 4 × 1 | 48 | 2.2 GiB (3.4 GB with hull polish) | 10 (7 with polish) | 35 s | ≈ 1.05 h |
| 5 × 4 × 25 (paper), snapshots off | 80 | ≈ 18.3 GiB | **1** | to be timed | 107 × $t_{\mathrm{cycle}}$ |

Campaign size on SRP1: Phase A ≈ 110 evaluations (≈ 12 h with 10 slots); Phase B ≈ $K$ polls ×
($n{+}1$ or $2n$) + searches — with $K \approx 20$ and $n+1 = 19$: ≈ 400 evaluations (≈ 2 days). The
hull polish is run only on incumbents and the final neighbourhood, not on every evaluation, so the
campaign uses 10 slots. **Provisional campaign design:** the search on SRP1; paper-scale evaluation of
the final incumbent (and, if the timed cycle allows, its best neighbours) serially with snapshots off,
which is also where the multi-scenario elements the reviewers asked for (row 18, α sensitivity, R3.6
matrix) are evaluated in Step 5. Confirmed or revised on the timed paper-scale cycle.

---

## 8. What the paper can claim

- The plan is a **mesh-local optimum on the product lattice** (0.25 MVA, 0.5 MWh, 5-year timing):
  every lattice neighbour in the final poll was evaluated and none improves $F$ — or, if the poll
  stopped above unit size because of $\sigma_Q$, a local optimum at the reported poll size with the
  finer neighbourhood reported as unresolved. The claim comes with the number of evaluations, the poll
  history and the measured $\sigma_Q$.
- Each $Q(x)$ is a certified ADMM point (10 plain cycles inside the Boyd tolerances), with the hull
  polish gap and the reproducibility band from Step 3; the campaign used one frozen configuration and
  cold starts, so $Q$ is a deterministic function of $x$ and the campaign is reproducible.
- No global optimality and no gap. The Benders-type cuts, if used, are described as a search heuristic.
- Figures: Phase A value curves (§4.1); the poll history (incumbent $F$ vs evaluation count); the final
  neighbourhood table.

---

## 9. Author decisions taken and still open

Taken (2026-09-19): `budget` (€1e6) and `max_capacity` (energy ≤ 5 MWh per node) kept as master
constraints; salvage excluded from $F(x)$ — it is a reporting expression only, never in the ESSO
objective, so this is a reporting choice with no effect on the oracle; the frozen configuration is
AA-on (`keep_memory`) once the case file carries it (Addendum 27).

Taken (2026-09-19, later): the corrected cost file is `SRP1_ESS.xlsx` at `7ce1d1ab`
(`paper_revisions`, energy costs × 1.25), to be brought onto the working branch and hashed into the
spec; **the campaign runs under the €1M budget** as its only Phase B (the submitted paper's framing,
re-optimized on corrected costs), with the unbudgeted question answered by the A1 ladders; Phase B
in the single-cohort form first, the general lattice only if a staging probe at the incumbent shows
staging helps.

Open (Addendum 28):
1. **Baseline ageing calibration and SoH floor**, decided together with a datasheet citation (C2
   datasheet-80 % vs C4; soh_min 0.50 vs the manuscript's 0.70; calendar fade on or off). The
   investment sign on SRP1 turns on this choice.
2. **Paper-scale route:** a ≥ 64 GiB machine for three evaluations (x = 0, smallest unit, one larger
   design) exploiting the affine objective, after the single-scenario code paths and row 18 (α) are
   in place; otherwise SRP1 results with the caveat plus the reduced-scenario SRP1 variant.
3. The ESSO capacity multipliers stay deferred unless §5.6 is used; if it is, sign and units are
   validated first by a finite-difference check at C\* (two certified runs).

---

## 10. References

- C. Audet, J. E. Dennis Jr., "Mesh adaptive direct search algorithms for constrained optimization",
  SIAM J. Optim. 17(1), 2006.
- M. A. Abramson, C. Audet, J. E. Dennis Jr., S. Le Digabel, "OrthoMADS: a deterministic MADS instance
  with orthogonal directions", SIAM J. Optim. 20(2), 2009.
- C. Audet, S. Le Digabel, C. Tribes, "The mesh adaptive direct search algorithm for granular and
  discrete variables", SIAM J. Optim. 29(2), 2019.
- C. Audet, S. Le Digabel, V. Rochon Montplaisir, C. Tribes, "Algorithm 1027: NOMAD version 4:
  nonlinear optimization with the MADS algorithm", ACM Trans. Math. Softw. 48(3), 2022.
- C. Audet, A. Ianni, S. Le Digabel, C. Tribes, "Reducing the number of function evaluations in mesh
  adaptive direct search algorithms", SIAM J. Optim. 24(2), 2014 ($n+1$ directions).
