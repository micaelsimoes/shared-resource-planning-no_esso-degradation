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
