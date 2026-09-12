# P5.13-A — expected directional throughput versus expected net power

Definitional decision with code trace. **No solver work.** No quantification is
attempted, and none is possible on SRP1 — see "The structural zero" below.

## Code trace

1. **Network models carry per-scenario dispatch.** Shared-ESS variables are indexed
   `[e, s_m, s_o, p]`, so each market/operation scenario has its own `shared_es_pch`,
   `shared_es_pdch` and `shared_es_pnet = pch - pdch`.
2. **Coordination averages NET power.** `dn_interface_expected_sess_p_def`
   (`model_construction_helpers.py:1798`) sets
   `expected_shared_ess_p[p] = sum_{s_m,s_o} pi_m * pi_o * shared_es_pnet[e,s_m,s_o,p]`.
   The expectation is taken over the **signed net** quantity, so opposite-sign dispatch
   across scenarios cancels here.
3. **Consensus carries that expected schedule only.** The ADMM consensus parameters are
   indexed by period alone (`p_ess_req[p]`, `q_ess_req[p]`, ...), not by scenario.
4. **ESSO re-decomposes the expected net schedule.** It receives the expected net
   quantity and forms its own directional `es_pch_per_unit` / `es_pdch_per_unit`,
   which are indexed by cohort, year, day and period — **with no scenario index**.
5. **Degradation is driven by those directional variables.**
   `shared_energy_storage_data.py:~480`:
   `avg_ch_dch += (num_days/365) * (eff_ch * pch * dt + pdch * dt / eff_dch)`.

So the composition is: per-scenario dispatch -> expectation of **net** -> directional
decomposition -> throughput. The quantity that drives degradation is therefore the
directional throughput **of the expected schedule**, not the expectation of the
realized directional throughput.

These differ whenever scenarios disagree in sign. The Expert's construction: two equally
likely scenarios charging and discharging 1 MW give zero expected net power and 1 MWh
expected throughput; the zero-charge/zero-discharge decomposition is compatible with the
expected net power while omitting the throughput entirely. Relaxed complementarity does
not recover the missing information. With efficiencies the discrepancy persists, since
the degradation-driving quantity is `E[eff_ch*pch + pdch/eff_dch]`, not a directional
decomposition of `E[pnet]`.

## The structural zero — why SRP1 cannot measure this

`SRP1.json` sets `NumMarketScenarios = 1`, and `num_operation_scenarios = 1` for the
transmission network and for all three distribution networks. The expectation is
therefore over a single realization at every level, and **the discrepancy is identically
zero in this configuration — by construction, not by measurement.**

Recorded explicitly so no later reader mistakes it for a measured null. Computing this
quantity on SRP1 would return zero and would constitute a false negative that reads as
validation. By the fifth artifact rule's own logic, a number whose meaning depends on an
unstated configuration assumption is as unpreserved as one whose formula is missing.

## One finding, not two — the single-scenario restriction is structural

Coordination is on expectations by construction (point 3 above), and the evaluation
oracle additionally hard-wires scenario `(0,0)` when reading coordinated quantities back:
`p56a_oracle.py:275-287` (`pc_adn[dn,0,0,p]`, `pg_adn[0,0,p]`, `vmag_sqr[.,0,0,p]`,
`shared_es_pnet[e,0,0,p]`) and `:393-397`. Production's model construction does sum
properly over scenarios; the restriction lives in the coordination/evaluation path.

Consequently, moving to the paper's 25 market/operational combinations per day is **not
a data change the existing oracle would absorb** — it requires changing that path. The
Expert's separate point, that the reduced-case certificate cannot be carried unchanged
into the full stochastic study, is a **direct consequence of this indexing** rather than
an independent concern, and is recorded here as one finding.

## The definitional decision (requires the author's choice)

What is the degradation-driving quantity intended to represent?

**Option A — the committed common schedule.** Degradation reflects the day-ahead
schedule the operators commit to, not realized balancing. The current implementation is
then *correct as written*, and the obligation is editorial: the paper must say that
degradation, throughput and SoH reflect scheduled operation, and must not claim they
represent realized cycling. The `(0,0)` oracle restriction remains a limitation for any
multi-scenario claim.

**Option B — realized operation.** Degradation reflects what the asset actually does
across realizations. This does **not** require a per-scenario ESSO: with fixed
efficiencies, expected directional powers suffice, so the minimal change is to pass
`E[pch]` and `E[pdch]` to the ESSO instead of the single `E[pnet]` — two expectations
rather than one.

Trade-off to state explicitly under Option B: `E[pch]` and `E[pdch]` may both be
strictly positive at the same period, because the scenarios that charge and those that
discharge are mutually exclusive *within a realization* but not *in expectation*. That
is correct for throughput accounting, but it must not be read as simultaneous physical
charge and discharge, and it weakens the ESSO-level complementarity interpretation.

**Planner recommendation.** Option B is the physically faithful choice, and its minimal
form is small and well defined. But it is a modelling decision about what the tool
represents, so it is the author's to make. Whichever is chosen must be recorded before
the ranking baseline is re-derived, because it changes the degradation term and hence
the objective — the same argument that made penalty classification blocking.

## Deliverable 2 — the diagnostic specification

`data/SRP1/Results/P513A/frozen_throughput_diagnostic_spec_v1_bf56149e.json` specifies the
multi-scenario diagnostic that would quantify the discrepancy once a case exists that can
exhibit it. It is frozen now, in the P5.12-Z manner, so it is ready rather than
improvised later. It is a specification only; no case currently satisfies its
precondition.

---

# Correction (2026-09-12) — Option B is blocked, not merely reinterpreted

The Option B paragraph above says the pair of strictly positive expectations "must
not be read as simultaneous physical charge and discharge" and that it "weakens the
ESSO-level complementarity interpretation". That is too weak, and it made B look
like the cheaper option. Complementarity is **actively enforced**, and Option B's
correct values violate it by orders of magnitude. The corrections below supersede
the corresponding statements above; the code trace, the structural zero and the
one-finding framing are unchanged.

## C1 — the magnitude of the violation

`definitions.py:80-81`: `SMALL_TOLERANCE = 1e-4`,
`ESS_COMPLEMENTARITY_TOLERANCE = SMALL_TOLERANCE`. The rows to which it applies are
written in **normalized** variables in `[0, 1]` (`*_hat`, power divided by the
rating). Equal directional values therefore satisfy `x^2 <= 1e-4`, i.e.
`x <= 1%` of rating.

A two-scenario ±50/50 split needs `0.5` in each component: **50x over per
component, 2500x over in the product**. The constraint is not marginally tight
under B; it is binding by three orders of magnitude. No tolerance reading,
efficiency factor or relaxation interpretation closes a gap of that size.

## C2 — the three sites, with two different failure modes

| # | Row | Location | Behaviour under Option B |
|---|---|---|---|
| 1 | `pch_hat * pdch_hat <= slack_es_ch_comp_per_unit[y_inv,y,d,p] + 1e-4` (per cohort) | `shared_energy_storage_data.py:569` | **Slack-absorbed.** The slack is penalized at `:641` via `PENALTY_ESSO_SLACK`, so the violation does not fail — it becomes cost inside `gross_operational_cost`. |
| 2 | `es_pch_hat_agg[y,d,p] * es_pdch_hat_agg[y,d,p] <= 1e-4` (aggregate, `agg_pch`/`agg_pdch` normalized by `es_s_rated[y]` at `:613-616`) | `shared_energy_storage_data.py:618-620` | **No slack — straightforwardly infeasible.** |
| 3 | `shared_es_pch_hat * shared_es_pdch_hat <= 1e-4` (per scenario) plus a penalty term | `model_construction_helpers.py:836-838`, penalty at `:1726` | The rows are **per scenario** and are not themselves violated by B. What B breaks is the *rationale*: see below. |

Site 3 requires care. Because the network rows carry `[e, s_m, s_o, p]`, directions
remain exclusive *within* a realization and the rows hold under B. But the P5.4-H1.6
comment (`shared_energy_storage_data.py:599-605`) justifies the site-2 aggregate row
precisely by compatibility with the network side: "the network agent represents ONE
aggregate shared ESS and imposes complementarity on its aggregate charge/discharge,
so the ESSO aggregate feasible set must be compatible or ADMM would be reconciling
two different feasible sets." Under B the ESSO's directional variables would be
*expectations* while the network's remain *per-realization*. The two sides then no
longer describe the same object, so the compatibility argument that produced site 2
does not survive the switch. Site 2 is downstream of a choice made under Option A
semantics.

## C3 — the corrected implementation cost of Option B

Not "two expectations rather than one". At minimum:

1. the two ESSO complementarity sites (`:569`, `:618-620`) must be redefined or
   removed for the expectation-valued variables;
2. the ADMM channel changes. `update_shared_energy_storage_model_to_admm`
   (`shared_resources_planning.py:3783-3827`) reconciles exactly one expected net
   pair `(es_pnet, es_qnet)` against `(p_req, q_req)`, with one dual pair and one
   `rho`, normalized by `2 * S_rated`. Carrying a directional pair adds a consensus
   quantity, a dual, a residual and a stopping test at every node and period;
3. the P5.4-H1.6 rationale must be re-derived, not merely re-read;
4. `p56a_oracle.py:275-287, :393-397` still hard-wires scenario `(0,0)`, so no
   configuration that can exhibit the discrepancy runs through the existing
   coordination/evaluation path.

The author should choose knowing this. The earlier framing understated B's cost.

## C4 — site 1 belongs to the penalty-classification item

Site 1 is a concrete instance of **a penalty absorbing a modelling inconsistency
rather than signalling one**. Adopt Option B's expectations without the model change
and nothing fails: the correct expected throughput shows up as `PENALTY_ESSO_SLACK`
cost inside `gross_operational_cost` — the exact quantity whose penalty status is
already the blocking item. A penalty classified as operating cost cannot signal a
modelling inconsistency; it silently prices one.

This is the strongest available argument that penalty status must be **settled**,
not documented. It is folded into the penalty-classification item explicitly.

## C5 — Option C: examined, NOT established

*Proposal (the author's, recorded for completeness).* The network agents already
carry scenario indices and already sum over them. They could pass two expectations,
`E[pch]` and `E[pdch]`, used **only** as inputs to the degradation law, while the
ESSO's dispatch variables keep their present semantics and their complementarity
rows untouched — decoupling throughput accounting from the feasible set.

The obstacle was checked before recording the option. Three were found; the first
is structural.

**O1 — the consumed quantity is not per-period, and carries an index the networks
do not have.** The degradation law consumes `es_avg_ch_dch_per_unit[y_inv, y]`
(`shared_energy_storage_data.py:479-490`), a **per-cohort, per-year** scalar formed
by `(num_days/365) * sum_d sum_p` over ESSO variables carrying the investment-vintage
index `y_inv`. It feeds `es_degradation_per_unit[y_inv, y]` and the per-cohort SoH
chain (`:509-532`), each cohort having its own `es_e_rated_per_unit`. The network
models have **no cohort decomposition at all** (`shared_es_pch[e, s_m, s_o, p]`).
An exogenous `E[pch]`, `E[pdch]` can therefore constrain only the cohort *sum*,
leaving the per-cohort split — which is what the law actually consumes —
undetermined. Option C needs a cohort-allocation rule that does not exist in the
code. This is the sharper form of the per-period concern: the quantity is not
per-period, it is per `(cohort, year)`, aggregated over days and periods.

**O2 — the transfer form severs the only channel that prices cycling.** In the
operational subproblem the ESSO objective is `feasibility_penalty` plus the
augmented-Lagrangian terms (`shared_energy_storage_data.py:653-656`;
`shared_resources_planning.py:3808-3822`). Degradation is not a cost term there: it
enters through constraints — throughput -> degradation -> SoH ->
`es_e_available_per_unit` (`:458`, `:509-532`) — and reaches the economics through
available capacity and the salvage term. If the throughput accumulator becomes an
exogenous parameter, degradation stops depending on any ESSO **dispatch** variable.
The ESSO's dispatch preference, expressed through `dual_p_req` and pulling the
networks via the consensus term, is then the only place cycling cost could have
reached the networks — and it no longer carries it. The network objectives contain
no degradation term (their shared-ESS cost terms are `penalty_ess_usage`,
`model_construction_helpers.py:1652`, and the complementarity penalty, `:1726`).
Under the transfer form, **no agent prices cycling against dispatch**, and the ADMM
fixed point stops corresponding to a stationary point of the intended problem.

**O3 — the reconciled form re-imports the conflict it was meant to avoid.** Make
throughput a reconciled consensus quantity with its own dual, and the ESSO must
reproduce an expected *gross* throughput with directional variables that
simultaneously satisfy the `es_pnet` consensus and the complementarity rows —
exactly the pair of demands that is infeasible at site 2 and slack-absorbed at
site 1 whenever scenarios disagree in sign. The violation moves onto a consensus
residual that cannot close without slack. The rows are not avoided.

**Status.** Option C is **not** recorded as feasible and must not be cited as a
cheaper alternative. O1 must be answered with an explicit cohort-allocation rule;
O2 and O3 together show that the hoped-for decoupling exists only in the form that
removes the degradation price. It is recorded as an option **with its obstacles**.

A fourth consideration, worth stating but not decisive: any new consensus channel
changes the ADMM operator whose stability is the subject of the open cycle-21
investigation.

## C6 — the `A >= B` sign expectation now has a preserved derivation

The frozen spec asserted `A >= B` without recording why, which the fifth artifact
rule forbids. Derivation, assumptions and the one case in which a violation is
**not** a diagnostic defect are preserved in
`data/SRP1/Results/P513A/frozen_throughput_diagnostic_spec_v2_8b2d8f77.json`
(SHA-256 `8b2d8f77ac0194521bc2f7b3cdc7769285aa879782de4a80f01c45fcf9f3a5ed`), with
`data/SRP1/Results/P513A/frozen_spec_lineage.json` recording the v1 -> v2 relation.
v1 is preserved unmodified.

v2 also corrects a rule that would have concealed exactly the failure mode C4
describes: v1 said any `A < B` is a diagnostic defect. With the site-1 slack
active, `x_hat` and `y_hat` can both be strictly positive and `B` can exceed `A`,
so `A < B` is then a **finding** — slack absorption — and the diagnostic must
report the active slack alongside every `A`/`B` pair.
