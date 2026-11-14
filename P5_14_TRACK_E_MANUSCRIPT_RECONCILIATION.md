# Track E — manuscript reconciliation list

**Claim / current code status / correction required. This is a list, not manuscript
edits — the author owns those. Zero solves.**

## A. The Expert's items

| # | Claim in the manuscript | Current code status | Correction required |
|---|---|---|---|
| E1 | Eq. (1) adds a **positive** salvage expression | The implementation **subtracts** it: `net_operational_recourse = gross_operational_cost − terminal_salvage_value` (`shared_resources_planning.py:722-733`) | Fix the sign in Eq. (1). A residual asset value is a **credit** in a cost minimisation. |
| E2 | Table 6, p.26 — scenario-specific capacities | Contradicts the common physical plan of Sections 2.1–2.2; the response to R2.1 says all numerical results were recomputed, which the table does not support | Either recompute the table under the common-plan formulation or withdraw the claim that results were recomputed |
| E3 | Algorithm 1 retains a recovery cut; Eqs. (10)–(11) retain unsupported cuts | The local-cut master estimate is **not** a rigorous global lower bound; no such guarantee has been established | Remove the global-cut language. Never describe the master estimate as a global lower bound or the procedure as globally convergent Benders |
| E4 | Eqs. (21)–(28) describe **apparent-energy** degradation and the retired squared apparent-power equality | The implementation uses **active-energy** throughput: `eff_ch*pch*dt + pdch*dt/eff_dch` (`shared_energy_storage_data.py:479-490`), with the capability inequality in place of the retired equality | Rewrite the degradation equations in active-energy terms and delete the retired equality |
| E5 | Explicit SOC equations and daily closure are missing | Present in code (`sess_soc_rule`, `model_construction_helpers.py:845+`) | Add the SOC recursion and the daily-closure condition to the manuscript |

## B. Items established since the Expert's review

| # | Claim | Status | Correction required |
|---|---|---|---|
| E6 | **"0.50% optimality gap"** | It is `benders.tol_rel = 0.005` in `data/SRP1/SRP1_params.json` — the **configured stopping tolerance**, reported as an achieved gap | State it as the stopping tolerance. An achieved gap would require a valid lower bound, which E3 says does not exist. **Include the source** so the provenance is unambiguous |
| E7 | **70% minimum SoH** (p.25) | A **live branch divergence**, not an error: `soh_min = 0.70` exists on `paper_revisions` (`a3c76922`, 2026-07-29) and is **not** an ancestor of HEAD; the branch producing every result went `0.10 → 0.50` (`edcfbd95`) and stands at **0.50** | Either reconcile the branches or state which branch's constants the numbers come from. Table 7's 49.62% is consistent with the 0.50 floor binding, i.e. with the results branch |
| E8 | **60% / 80% SoH sensitivity cases** | **Not runnable** while the constants were hard-coded in `SharedEnergyStorage.__init__`; no record of such runs exists on either branch | Withdraw, or run them now that P5.13-C made the constants case-file parameters |
| E9 | **Ranking stability** | Rests on **path-identity cancellation**: identical code paths stop at identical points, so the stopping slack cancels in a difference. The oracle's own resolution is `objective_tolerance = 827,945` against a signal of **32.87** | Do not present ranking stability as oracle precision. State that it holds for reruns of the same candidates on the same code, and not across perturbations |
| E10 | **Branch governance** | `paper_revisions` carries different physics (`soh_min`) from the branch that produced every result | The paper must either reconcile the branches or state explicitly which branch's constants its numbers come from |

## C. Items the paper should add rather than correct

- **The degradation calibration is now explicit.** `(N, D, R) = (10000, 0.80, 0.50)` with
  `k = N·D/(−ln R) = 11541.56` (P5.13-D). The earlier implementation consumed `cl_nom`
  alone, so the reference depth was vestigial and the count and depth had drifted apart
  (P5.13-B). If any published number predates C3, say so.
- **`cl_nom` is a decay constant, not cycles-to-EOL.** The SoH chain is multiplicative per
  day, so at exactly `k` equivalent full cycles the model retains `exp(−1) = 36.8%`.
- **The single-scenario restriction is structural**, not a data choice: coordination is on
  expectations and `p56a_oracle.py:275-287, :393-397` hard-wires scenario `(0,0)`. The
  paper's 25 market/operational combinations are **not** a data change the current
  evaluation path would absorb.
- **Degradation reflects the committed schedule, not realized cycling** (throughput Option
  A, the interim reading). The paper must not claim the latter.

## C2. The governing claim — no existing result was produced by code that was doing anything with storage

**This is the sharpest item on the list and it belongs in the manuscript, not only in the
stage record.**

Every result currently in the repository was produced in a regime where the
storage-specific code paths are **inert**. The candidate behind the whole evidence base is
~0.0106 MVA / 0.0213 MWh per node — about 10.6 kW. **All four of the components that make
this a shared-energy-storage paper rather than a coordination paper are inert in every
existing result:**

| component | evidence of inertness |
|---|---|
| **the ageing model** | C3, a **15.4% change in the degradation constant**, moved the recourse by **nothing to sixteen digits** (`827885239.5417057` before and after) |
| **the ESS consensus channel** | the primal residual **never reached its 0.1 guard**, which is how a `KeyError` survived a week inside the diagnostic behind it (`58f4911b`, 2026-09-06) |
| **the ESSO constraint set** | never approached its feasibility boundary; at 1.00 MVA it fails outright at node 7, the boundary sitting between 0.96875 and 1.00 |
| **the terminal value** | salvage is **3,452 against a recourse of 828M — about 4 ppm** of the objective |

Stated together, that is considerably harder to requalify than any one of them separately.

So the claim to record is stronger and more specific than "the case study is too small to
exhibit the effect":

> **The paper's storage-benefit claims cannot be supported by any result currently in the
> repository — not because the numbers are too small to resolve, but because the mechanism
> that generates them was not executing.**

The distinction matters for what the paper must do. A resolution problem is fixed by a
bigger effect or a tighter tolerance. This is not a resolution problem: at ~10 kW per node
the degradation, complementarity and capacity machinery is numerically inert, so a reported
"incremental storage benefit" from such a run is not a small measurement of a real effect —
it is a measurement of something else.

**Required correction:** any storage-benefit figure in the manuscript must be traced to the
capacity at which it was produced. If it came from a run at the bootstrap capacity, it must
be withdrawn rather than requalified. The ladder gives the largest capacity the tool can
currently evaluate — `C* = 0.96875 MVA / 3.875 MWh` per node — and any restated claim must
sit at or below it, labelled as reduced scale.

## D. Blocking dependency to state in the paper

**Penalty classification is unresolved.** `gross_operational_cost` — the quantity a
zero-salvage objective promotes — "may include artificial penalty terms and is therefore
not necessarily a pure economic operating cost", by its own docstring
(`shared_resources_planning.py:723-724`). Until each penalty is classified as an economic
cost or as a detector that must vanish, no ranking baseline derived from it is an economic
baseline. This blocks any re-derivation, and it should be visible in the paper rather than
implicit.
