# P5.10 — stabilized rescaled ADMM oracle

Branch `feature/derivative-free-planning`. Canonical runtime
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, checksum
`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`, IPOPT
`3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`, HSL `ma97` — all asserted
by the R0 provenance gate in every harness.

**No production change was made.** `data/SRP1/SRP1_params.json` was never
written. The rescaled objective is P5.8's process-local wrapper, removed before
every polish; rho, the adaptive flag and the history channels are set on a
per-evaluation deep copy and on a private copy of the template. IPOPT options,
ADMM tolerances, stopping criteria, penalty rules, the anchor policy, ESS
equations, Benders and the convex models are untouched, and nothing was merged.
No investment optimization, derivative-free search, GPS/MADS, surrogate
optimization, Benders run or convex planning model was executed.

New files: `p510_oracle.py`, `p510_a_state.py`, `p510_b_fixedrho.py`,
`p510_c_endpoint.py`, `p510_e_criteria.py`, `p510_f_replay.py`,
`p510_g_anchor.py`. Evidence: `data/SRP1/Results/P510/`.

> **Repository note, outside the stage but affecting it.** HEAD was merge
> `611059b9`, which brought in origin commit `518aa1d8` ("`.md files
> updated.`"). That commit branched from the P5.8 commit and replaced
> `REVISION_CONTEXT.md` and `LOCAL_NLP_STABILITY_PLAN.md` with a P5.6-C-era
> snapshot; the merge resolved both in its favour, discarding P5.6-D, P5.7,
> P5.8, P5.9 and the migration/provenance corrections. Every line it adds
> relative to its own parent is the old P5.6-C stage specification, which the
> current plan document already retains as historical, so nothing was lost by
> restoring from `df77a9b9` (commit `0bfeb1ae`). This is the second rollback
> from the same source and it will recur until the stale copy on the other
> machine is dealt with. All other artefacts survived the merge intact.

---

## 0 — reproduction gates

The explicit-configuration oracle, instructed to reproduce production's implicit
configuration, returns the accepted values:

| gate | configuration | field | expected | observed | delta |
|---|---|---|---|---|---|
| CURRENT | `CURRENT`, rho `1.5/2.25/1.0`, adaptive on, history inherited | polished total | 828 021 090.360850 | **828 021 090.3608505** | `+4.8e-07` |
| RESCALED | `RESCALED`, same rho, adaptive on, history inherited | pre-polish net recourse | 825 814 074.49 | **825 814 074.4930633** | `+0.003` |

Both residuals are float printing. Everything below rests on these.

---

## A — the oracle state, made explicit

### A1 — what was implicit, measured on the frozen template

Production's warm-start path clones `initial_state['models']`
(`shared_resources_planning.py:2173-2179`) without rebuilding the augmented
objectives, and restores several scalars from `initial_state` when the evaluated
candidate equals the template's (`continuing_same_candidate`, 2068-2095). Read
directly off frozen T0:

| channel | template carries | nominal source says |
|---|---|---|
| **1. rho** | `rho_v = 1.5`, `rho_pf = 2.25`, `rho_ess = 1.0` | parameter file: `1.0 / 1.0 / 1.0` |
| **2. objective scaling** | carried inside the cloned models | not a flag at all |
| **3. `consecutive_converged_cycles`** | `1` | `0` on a cold run |
| **4. `last_recourse`** | `838 496 830.8134136` | — |

Channels 1 and 3 were already known (P5.9-A, P5.8-A0). Channel 2 means CURRENT
versus RESCALED is a property of the template rather than of a declared setting.
Channel 4 is new.

### A2 — channel 4 is asymmetric across candidates

`objective_convergence` is initialised `False` and computed only when
`previous_recourse is not None` (2325-2328, 2375-2380), and `previous_recourse`
is inherited only when `continuing_same_candidate`. **T0's candidate is the base
candidate.** So, evaluating from T0, the base gets a live objective-convergence
test on cycle 1 and every other candidate does not — inside the quantity the
planning problem consumes, `Delta(x) = Q(x) - Q(base)`.

Measured under CURRENT, where the asymmetry lives (it is a property of the
warm-start logic, not of the scaling):

| history | candidate | cycles | cycle-1 objective change | objective converged |
|---|---|---|---|---|
| inherited | `base` | 2 | **1 290 877.21** | False |
| inherited | `se\|node5\|2025\|-10%` | 2 | **None** | False |
| neutralised | `base` | 2 | None | False |
| neutralised | `se\|node5\|2025\|-10%` | 2 | None | False |

**The mechanism is confirmed; its consequence did not bite at this point.** Both
runs took two cycles, because the base's inherited comparison
(`1 290 877.21`) exceeded its tolerance (~`838 497`) and failed the test anyway.
This is a latent defect whose effect depends on how close the inherited
`last_recourse` happens to lie, **not** a demonstrated distortion of the P5.9
numbers. It is reported at that strength deliberately.

Neutralising costs nothing: the base returns `828 021 090.3608505` either way,
identical to the last digit.

### A3 — purity

Same candidate, same declared configuration, different call history:
**bit-identical**.

### The contract now in force

`OracleConfig` names the scaling mode, all three rho values, the adaptive flag,
the history policy, the template id, the initialisation policy and the anchor,
and carries a hash. Every evaluation records rho at start and at end, every
adaptive action, the convergence counters in and out, and the template id.
Neutralisation clears each inherited key **and** the template's
`candidate_solution`, so it survives either mechanism changing.

`Q(x)` is now a function of the candidate and an explicit configuration.

---

## B — fixed-rho rescaled ADMM validation

RESCALED only; adaptive rho disabled, because P5.9-B established the production
rule is self-cancelling after rescaling (§D). `rho_v = 1.5` and `rho_ess = 1.0`
held fixed and not swept: P5.9-A measured `rho_v` inert over a 667× range and
ESS consensus never within 175× of its tolerance on any of 96 cycles. History
neutralised, so base and perturbed candidates share one stopping rule.

| rho_pf | candidate | cycles | pre-polish net recourse | polished total | failed blocks | `primal_pf` | runtime |
|---|---|---|---|---|---|---|---|
| 300 | base | 6 | 827 885 239.54 | 828 008 246.66 | **0** | 9.05e-05 | 324 s |
| 300 | `se\|node5` | 6 | 827 885 356.72 | 828 007 656.38 | **0** | 9.06e-05 | 327 s |
| 300 | `se\|node9` | 6 | 827 885 291.37 | 828 007 602.40 | **0** | 9.05e-05 | 333 s |
| 500 | base | 9 | 827 922 880.79 | 827 977 138.21 | **0** | 7.36e-05 | 452 s |
| 500 | `se\|node5` | 9 | 827 922 956.91 | 827 976 524.84 | **0** | 7.36e-05 | 464 s |
| 500 | `se\|node9` | 9 | 827 922 928.88 | 827 976 483.92 | **0** | 7.36e-05 | 462 s |
| 1000 | base | 16 | 827 960 806.34 | 828 011 656.16 | **0** | 1.56e-05 | 769 s |
| 1000 | `se\|node5` | 16 | 827 960 901.78 | 828 011 042.57 | **0** | 1.56e-05 | 761 s |
| 1000 | `se\|node9` | 16 | 827 960 873.87 | 828 011 012.40 | **0** | 1.56e-05 | 713 s |

**Every candidate at every penalty polished cleanly. No continuation was
required anywhere** — including `se|node9|2025|-10%`, the P5.7-D target that
production can only reach by continuation, and `se|node5|2025|-10%`, which is
`POLISH_FAILURE` direct from T0 under CURRENT (confirmed again in A2).

All three beat production's generation-1 result of `828 021 090.36`: by
`12 843.70` (300), `43 952.15` (500) and `9 434.20` (1000).

**The landscape is penalty-independent**, which P5.9 could not establish because
adaptive rho decayed every configuration toward one attractor:

| rho_pf | `Delta(se\|node5)` | `Delta(se\|node9)` |
|---|---|---|
| 300 | −590.28 | −644.26 |
| 500 | −613.37 | −654.29 |
| 1000 | −613.60 | −643.77 |

Δ varies by 23.32 and 10.52 across a 3.3× range of penalty.

> **A defect in this stage's own declared selection rule, disclosed rather than
> applied.** Criterion 2 was declared before results as "prefer the smallest
> spread of `Delta(x)` across candidates relative to its own mean, i.e. the
> configuration whose candidate differences are best resolved". Those clauses
> are contradictory — smallest spread means the candidates are *closest*, which
> is worse resolution — and the metric is not well-posed on B's single-generation
> data, which yields no noise estimate. Read literally it selects `rho_pf = 1000`
> (ratio 0.048); read by its stated intent it selects `300` (0.088). Criteria 1
> (all clean) and 3 (runtime) are unambiguous and unaffected. Because Δ is
> invariant across all three, nothing rests on the choice, and **stage C was run
> at both `rho_pf = 1000` and `300`** to confirm that. They agree on every
> qualitative conclusion. `rho_pf = 1000` is carried into F for its interface
> agreement, which is the property C's question turns on; `rho_pf = 300` reaches
> a comparable landscape at 2.4× lower cost and is the better practical choice
> if runtime matters.

---

## C — the oracle endpoint. This is the central experiment.

Oracle A is the rescaled ADMM output taken directly; Oracle B is that output
followed by the exact-consensus polish. Both measured on the same run.

**The comparison is not symmetric, and the answer turns on that.** Oracle B
returns one operating point: the polish pins every agent to a common interface
value and the coordinated residual is ~1e-16. Oracle A returns per-agent
solutions that *disagree* at the interface by the ADMM's terminal primal
residual. Each agent's own SMOPF is feasible — IPOPT solved it — but there is no
single physical operating point until they are reconciled.

| | `rho_pf = 300` | `rho_pf = 1000` |
|---|---|---|
| Oracle A interface disagreement | 9.05e-05 p.u. = **18.1 kW** | 1.56e-05 p.u. = **1.6 kW** |
| Oracle A unpolished max network violation | **1.1538e-05** | **1.1538e-05** |
| Oracle B coordinated residual | 1.39e-16 | 5.55e-17 |
| polish effect on recourse | **+73 007** | **+850** |
| polish cost | 32 s of 284 s | 33 s of 708 s (4.7 %) |
| repeatability, both oracles | bit-identical | bit-identical |

The unpolished models are individually **as feasible as the polished ones**:
`1.1538e-05` is IPOPT's own feasibility tolerance, the same figure P5.6-A
through P5.7 recorded on every certificate. Oracle A is not infeasible.

### But the two endpoints rank the candidates in opposite orders

| | order | `Delta(se\|node5)` | `Delta(se\|node9)` |
|---|---|---|---|
| **Oracle A**, `rho_pf = 1000` | base < node9 < node5 | **+95.44** | **+67.53** |
| **Oracle B**, `rho_pf = 1000` | node9 < node5 < base | **−613.60** | **−643.77** |
| **Oracle A**, `rho_pf = 300` | base < node9 < node5 | +117.18 | +51.82 |
| **Oracle B**, `rho_pf = 300` | node9 < node5 < base | −590.28 | −644.26 |

The unpolished oracle says both perturbed candidates are **worse** than base;
the polished oracle says both are **better**. Consistently, at both penalties.

The reason is arithmetic and decisive. Reconciling Oracle A's 1.6 kW
disagreement costs **850** in objective, and the investment signal being ranked
is about **600**. *The cost of making the answer a physical operating point is
larger than the difference being measured.* An unreconciled point therefore
cannot rank candidates, however small its residual looks in p.u.

```
C CONCLUSION — the polish is NOT an unwanted projection.  It is what makes the
result a single physical operating point, it costs under 5% of runtime, and
removing it inverts the investment ranking.  Oracle B is the oracle endpoint.
```

---

## D — adaptive penalty redesign audit (audit only, nothing implemented)

### The current rule, and why it fights rescaling

Production measures the dual residual as

```
dual = rho * |z_current - z_prev| / base       (shared_resources_planning.py:4897, 4938)
```

— **linear in rho** — and then compares `primal_ratio = primal/primal_tol`
against `dual_ratio = dual_mean/dual_tol`, increasing rho when
`primal_ratio > 5 * dual_ratio` and decreasing it when
`dual_ratio > 3 * primal_ratio` for the pf family
(`_update_admm_penalties`, 5440-5520).

Raising rho raises its own measured dual residual proportionally and trips its
own decrease branch. P5.9-B measured the consequence exactly: requested `rho_pf`
of 300, 500 and 1000 terminate at `88.89`, `98.77` and `131.69` — precisely
`300/1.5³`, `500/1.5⁴`, `1000/1.5⁵`. The larger the request, the more decrease
steps, so every configuration is compressed into a narrow attractor near
90-130. **The rule cannot hold a calibrated penalty after rescaling.**

P5.10-B is the counterfactual: with the rule disabled, `rho_pf = 1000` stays at
1000 (§B records `rho_observed_final` unchanged), the polish succeeds for every
candidate at every penalty, and the landscape becomes penalty-independent — none
of which is reachable while the rule is decaying the penalty underneath the
experiment.

### Alternatives, proposed only — no implementation, no parameters selected

1. **Normalize the dual residual by rho before comparison**, or equivalently
   compare `|z_current - z_prev| / base` against a rho-independent threshold.
   The present form conflates "the agents are still moving" with "the penalty is
   large"; only the first is a convergence statement. This is the minimal change
   that removes the feedback.
2. **Rho-independent residual comparison** — compare primal and dual progress in
   commensurate units derived from the base objective rather than from the
   augmented one, so the balance test measures coordination rather than
   penalty magnitude.
3. **Penalty floors** — a per-family floor so a value established by a
   calibration stage cannot be undone inside the first run. A floor is a
   containment measure, not a fix: it leaves the feedback in place above the
   floor.
4. **Recalibrated thresholds** — the `5.0` increase and `3.0` pf-decrease ratios
   were chosen against a residual definition that this stage shows to be
   rho-coupled. They cannot be re-derived meaningfully until (1) or (2) settles
   what the residual means.

The order matters: (1) or (2) first, then (4). (3) is only worth adding if a
redesigned rule still proves unstable.

---

## E — convergence criteria audit (audit only, nothing changed)

96 cycles across A, B, C and F; 9 of them are the cycle that ended an ADMM run
under the stabilized configuration. "Binding" is the criterion with the least
relative slack, slack being `threshold / observed`, so 1.0 is exactly at the
boundary.

### Which criterion stops the stabilized ADMM

| binding criterion | terminating cycles |
|---|---|
| **`stationarity_pf`** | **9 of 9** |
| everything else | 0 |

Slack on those 9 terminating cycles:

| criterion | min | median | max |
|---|---|---|---|
| **`stationarity_pf`** | **1.00** | **1.06** | 1.10 |
| `objective` | 13.9 | **20.9** | 34.8 |
| `consensus_pf` | 110 | 134 | 643 |
| `stationarity_v` | 124 | 126 | 128 |
| `consensus_v` | 491 | 722 | 1.46e+03 |
| `consensus_pf_mean` | 657 | 2.09e+03 | 1.14e+04 |
| `stationarity_ess` | 836 | 880 | 902 |
| `consensus_ess` | 279 | 830 | 1.81e+04 |
| `consensus_ess_mean` | 7.70e+03 | 1.73e+04 | 3.31e+04 |

**The stabilized ADMM stops on interface stationarity alone, on every cycle, and
sits exactly on that threshold (median slack 1.06).** This completes a
progression worth recording: under CURRENT the objective test bound (P5.8-D);
under P5.9's partially stabilized configuration it was 17 of 24 stationarity_pf
and 7 objective; here it is 9 of 9. The objective test now has median slack
20.9 — it never binds and is no longer doing any work.

Every consensus criterion is at least 110× inside tolerance; every voltage and
ESS criterion at least 124×. They are inert.

### On the objective tolerance

`max(1e3, 1e-3 × recourse)` measures `827 945 .. 829 122` here, still **25.1×**
the `33 031` figure P5.6-D derived, and 75 of 96 cycles were declared
objective-converged while the recourse still moved more than it. That criticism
stands but has become moot for this configuration: the criterion is 20.9× slack
and never binds.

### Recommendations — no values selected, nothing implemented

1. **Absolute base-objective improvement.** The natural replacement now that the
   objective test is inert. It must be scaled to the decision resolution, and
   P5.9-D3 leaves open what that resolution is.
2. **Planning-signal-based threshold.** Attractive, but `tau_planning = 33 031`
   was derived from the CURRENT oracle. The stabilized oracle puts the
   best-to-second gap at **32.87** (§F). Any signal-based threshold must be
   re-derived from the stabilized oracle before it is used.
3. **Local KKT residual in base-objective units.** Already in the IPOPT logs and
   already parsed by `p58_rescale.read_log_since`; P5.8-B measured it 390-1488×
   tighter under RESCALED. Now the most defensible optimality measure available,
   and it costs nothing to read.
4. **Consecutive economic stabilization.** Only meaningful once the inherited
   `consecutive_converged_cycles` is made explicit — which §A does, and which
   production still does not.

Given §E's finding that `stationarity_pf` is the sole binding criterion, the
most useful immediate step is not a new objective test but an explicit decision
about what interface stationarity tolerance the planning problem requires.

---

## F — full oracle replay

Eight warm-start generations per candidate under the stabilized configuration
(`RESCALED`, `rho_v = 1.5`, `rho_pf = 1000`, `rho_ess = 1.0`, adaptive off,
history neutralised **at every generation** so the declared stopping rule holds
down the whole chain). The CURRENT arm is read from P5.9-D rather than re-run:
that is production as it actually is, and it reproduced the accepted P5.6-D base
chain to the digit at every generation.

Every generation of every candidate: **VALID, zero failed polish blocks**.
Cycle counts `16, 2, 2, 2, 2, 2, 2, 2`; base chain runtime 1 965 s for eight
generations.

### Δ from base, per generation

| gen | `Delta(se\|node5)` | `Delta(se\|node9)` | best |
|---|---|---|---|
| 1 | −613.60 | −643.77 | node9 |
| 2 | −611.92 | −642.53 | node9 |
| 3 | −608.67 | −640.86 | node9 |
| 4 | −606.48 | −638.39 | node9 |
| 5 | −610.34 | −643.72 | node9 |
| 6 | −600.08 | −633.18 | node9 |
| 7 | −595.00 | −630.45 | node9 |
| 8 | −591.51 | −627.66 | node9 |

**`se|node9|2025|-10%` is best at every one of the eight depths. Zero ranking
flips.** P5.6-C had 6 sign reversals in 9; P5.6-D changed the best candidate at
every depth transition tested; P5.9 had two flips in eight.

### Landscape stability

| | CURRENT (P5.9-D) | stabilized (P5.10-F) |
|---|---|---|
| spread of `Delta(se\|node5)` across depth | 420 636.12 | **22.09** |
| spread of `Delta(se\|node9)` across depth | 459 695.59 | **16.10** |
| cross-depth landscape uncertainty | **459 695.59** | **22.09** |
| mean best-to-second gap (the signal) | 18 085.03 | 32.87 |
| **uncertainty / signal** | **25.42×** | **0.67×** |

A **20 813×** reduction in cross-depth uncertainty, and the ratio crosses below
1 for the first time in this investigation. P5.6-D reported
`tau_planning_refined = 811 438` against a `4.25e5` benchmark; P5.9 reached
`6.0×`; the stabilized oracle reaches `0.67×`. **The landscape is resolvable:
the depth at which the oracle is stopped no longer determines which candidate
wins.**

### What still moves

The absolute chain still drifts: base `828 011 656.16 → 827 810 556.86`,
`−201 099.30` over eight generations, with steps
`−41 495, −15 262, −53 776, −24 504, −26 582, −22 337, −17 144`. That is 3.4×
smaller than P5.9's rescaled chain and 16× smaller than CURRENT's, but it has
not stopped and it does not decay monotonically. The drift is **common-mode** —
Δ moves by 22 while the absolute level moves by 201 099 — which is precisely why
the relative landscape is stable, and it is why absolute objective values from
this oracle must not be compared across depths.

---

## G — interface anchor (secondary, audit only)

No anchor policy was changed. Neither disabling adaptive rho nor neutralising
the history channels perturbs the fixed anchor:

```
stabilized (adaptive off, history neutralised)   max|pc - pc(T0)| = 0.000000e+00 ; qc = 0.000000e+00
adaptive on, history inherited (P5.9 config)     max|pc - pc(T0)| = 0.000000e+00 ; qc = 0.000000e+00
```

Carried forward and not re-measured: the anchor is common across candidates
(P5.8-E2, `0.000000e+00`) and across cold versus warm starts (P5.8-E3, within
`1.003037e-07 p.u.`), and the P5.7 flexibility-against-a-frozen-reference
mechanism persists under RESCALED at about one third the magnitude
(P5.9-E2: `+298 653` against `+865 891` over four generations).

**The anchor cannot influence ranking**: it is bit-identical across candidates
under the locked T0 policy, so it enters every candidate objective as the same
constant reference. It remains a modelling question about what the flexibility
reference should be, not a source of instability.

---

## Success criteria

| requirement | verdict | evidence |
|---|---|---|
| deterministic evaluations | **met** | A3 and C: bit-identical repeats after different call histories, both oracles, both penalties |
| explicit state | **met** | `OracleConfig` + provenance record; both reproduction gates pass |
| no hidden rho / counter inheritance | **met** | all four channels identified and neutralised; neutralisation changes the base result by 0.0 |
| stable candidate ranking | **met** | zero flips in eight depths; uncertainty/signal `25.42× → 0.67×` |
| polish either validated or removed | **validated** | removing it inverts the ranking; reconciliation costs 850 against a signal of ~600 |

---

## What was not done, and what remains open

- No production equation, solver setting, ADMM tolerance, stopping criterion,
  penalty rule, anchor policy or master/Benders code was modified, and nothing
  was merged.
- **Nothing from §D or §E was implemented.** No adaptive rule was redesigned and
  no convergence criterion was changed or parameterised.
- **The population is three candidates.** Ranking stability is established on one
  pair over eight depths. It is strong evidence and it is not a landscape.
- **The signal is small in absolute terms.** The best-to-second gap is `32.87`
  against `tau_numerical = 10.0` — resolvable, at 3.3×, but thin. Whether it is
  physically meaningful is P5.9-D3's open question, unchanged by this stage:
  the rescaled oracle responds proportionally to a 19× capacity change
  (`130 201`, `3.94× tau_planning`) while CURRENT's separations barely track
  perturbation size, which suggests the older, larger signals were numerical
  residue. That reading still rests on one large-perturbation pair.
- **Absolute drift persists** (`−201 099` over eight generations) and does not
  decay monotonically. Only relative comparisons at a common depth are safe.
- `rho_v`, `rho_ess`, the proximal gammas, the penalty clamps and the
  consensus/stationarity tolerances were not swept; `rho_pf` was swept over
  `{300, 500, 1000}` only.
- A self-consistent rescaled T0 build has still not been performed, carried over
  from P5.8 and P5.9.
- The CURRENT arm in §F was reused from P5.9-D rather than re-run.
- `rho_pf = 300` reaches a comparable landscape at 2.4× lower cost than the
  `rho_pf = 1000` configuration carried into F. That trade was not optimised.

---

## Verdict

The oracle's result is now a function of the candidate and an explicit
configuration. Four inheritance channels were identified — rho, objective
scaling, the convergence counter and `last_recourse`, the fourth of them new and
asymmetric between the base candidate and everything compared against it — and
all four are neutralised, at no cost to the reproduced values. Repeat
evaluations after different call histories are bit-identical, and both
reproduction gates return the accepted numbers.

With adaptive rho disabled and the penalty held fixed, the exact-consensus polish
succeeds for every candidate at every penalty tested. **Continuation is not
required anywhere**, including the candidate P5.7-D could reach only through it.
The landscape is penalty-independent, and across eight refinement depths the
preferred candidate does not change once: cross-depth uncertainty falls from
`459 696` to `22.09`, taking uncertainty over signal from `25.42×` to `0.67×`
and crossing below unity for the first time. The depth at which the oracle stops
no longer decides the answer.

The polish is validated rather than removed, and the measurement that settles it
is not the objective magnitude. The unpolished ADMM output is individually
feasible to IPOPT's own tolerance and disagrees at the interface by only 1.6 kW,
but reconciling that disagreement costs `850` against an investment signal of
about `600` — so the unpolished and polished oracles rank the candidates in
**opposite orders**, consistently at both penalties. An unreconciled point cannot
rank investments, and the polish costs under 5 % of runtime.

What remains is bounded and named. The absolute chain still drifts by `201 099`
over eight generations, common-mode, so only same-depth comparisons are safe.
The adaptive penalty rule remains self-cancelling and is untouched; the
stabilized oracle simply does not use it, and §D says what a redesign would have
to change first. Interface stationarity is now the sole binding convergence
criterion, sitting exactly on its threshold, which makes the interface tolerance
— not the objective tolerance — the next parameter that needs a deliberate
decision. And the population is three candidates: the ranking stability
demonstrated here is strong, and it is not yet a landscape.

```
P5.10-A — stabilized rescaled ADMM oracle established
```

```
P5.10 COMPLETE — ready for planner review
```
