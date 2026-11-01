# P5.15 — Planner report: Step 1b (candidates), and two interventions

**Prepared for the author and the external expert. Continues
`P5_15_PLANNER_STEP1_PROGRESS.md`.**

Authority: `PLANNER_BRIEF_2026-09-13.md` with Addendum 1 **as amended** (Candidate 2
unblocked; Candidate 3 withdrawn, replaced by 3′).

| stage | state | commit |
|---|---|---|
| pre-Step-1 `.nl` check | complete — constraint multipliers are **not** exported; A4 stands | — |
| Step 1a — solver policy | complete, gates passed | `6d2349af` |
| Step 1 — ESSO reformulation | complete, smoke-tested | `b03c9b14` |
| **Step 1b Set 1 — candidates 1, 2, 4, 5** | **complete, smoke-tested** | `feb7b5fe` |
| **Candidate 3′** | **implemented, measured, FAILED its gate, REVERTED and HELD** | — |
| Gates G1–G5 | **not started** | — |

---

## 1. Candidate 3′ failed its own gate — the headline of this step

3′ was specified as *conditioning only, feasible set unchanged*, with the gate "identical
dispatch to 1e-8 on a converged fixture; report cold-start iteration count before/after".

Measured on `case33_2` node 7, 2025 Autumn, shared ESS at 0.97 MVA / 1.94 MWh, both arms
solved cold through the production path:

| | before (un-normalized) | after (3′) |
|---|---|---|
| termination | optimal | optimal |
| objective | −0.9473847716012889 | −0.9445205132701792 |
| max dispatch difference | — | **1.95e-05** (gate: 1e-8) |
| objective difference | — | **2.86e-03 ≈ 0.3 %** |
| **cold-start iterations** | **39** | **65** |

**Not a tolerance artifact.** Re-run at `tol = acceptable_tol = 1e-9` the differences did not
shrink (1.9553e-05, 2.890e-03).

**Two independent reasons to hold it.**

1. It misses its stated gate by more than three orders of magnitude. A 0.3 % objective shift
   is far above the ranking signal this project has spent the whole programme bounding.
2. **It made cold-start conditioning worse** — 39 → 65 iterations — which is the opposite of
   the effect it was proposed to deliver. For a change whose only justification is
   conditioning, that is disqualifying on its own.

**The mathematics is not in dispute.** `(pg/pg_avail)² + (qg/pg_avail)² ≤ 1` and
`pg² + qg² ≤ pg_avail²` are the same set for `pg_avail > 0`, so the feasible set is genuinely
unchanged. What differs is **which local optimum the solver reaches**. The per-generator
breakdown localises it to generators 3 and 4 in periods where `pg_avail` is small but well
above ε (0.41–2.84e-03 p.u.) — precisely the low-availability regime the audit measured as
carrying gradients down to 5.44e-05 with zero cold-start margin. Generator 0, the reference
unit, then shows the largest absolute difference because it absorbs the balance consequence.

**Disposition: reverted and held**, with the rationale recorded in place at `sg_avail_rule`
so the next reader cannot re-propose it without seeing the measurement. Evidence:
`data/SRP1/Results/P5151B/candidate3prime_ab.json`,
`candidate3prime_tight_tol_diagnostic.json`, `candidate3prime_per_generator_diff.txt`.

**One check 3′ did pass**, worth keeping if it is ever revisited: part (b) is already
satisfied by the codebase. `renewable_generation_is_unavailable` is one term of an `or`-chained
skip condition shared by `sg_capability`, `gen_pf_upper` and `gen_pf_lower`, so at zero
availability **no inequality row of any of the three is built** — only the box bounds
`pg, qg ∈ [0,0]`, which is not an LICQ-degenerate intersection. The degeneracy 3′ was partly
meant to remove is therefore already absent, and `pg_avail == sg_avail` exactly in this
dataset (`qg_avail ≡ 0`), so the apparent-power threshold and the active-power threshold
coincide.

---

## 2. A deletion that broke preserved evidence — found and repaired

The Worker implemented Candidates 1 and 2 by **deleting** the four retired rule functions
(`sess_phi_limits_lower`, `sess_phi_limits_upper`, `sess_s_sensitivities`,
`sess_e_sensitivities`).

That broke unpickling of **every preserved network fixture**, because the pickles hold
`functools.partial` objects that resolve those names at load time:

| fixture | before repair | after repair |
|---|---|---|
| `P512R/cycle21_pre_setup/snapshot.pkl` — the cycle-21 anchor | **FAILS** | LOADS |
| `P512R/cycle21_prepared/snapshot.pkl` | **FAILS** | LOADS |
| `FrozenSMOPF/matched_success_DSO_node7_…pkl` — the converged comparator | **FAILS** | LOADS |

This is not cosmetic: `cycle21_pre_setup/snapshot.pkl` is the fixture Step 0's DSO arm used
and the anchor of the entire P5.12 line.

**Repair.** Candidates 1 and 2 require the **rows** gone, not the **callables**. All four
functions are restored, unwired and unused, with comments stating they exist only so historical
artifacts remain loadable and must never be called. The rows remain absent from newly built
models — verified by `hasattr` checks on both the DSO and the TSO. This follows the same
"deactivate, do not delete" precedent already applied to `_add_benders_cut`.

**Worth recording for the brief's own wording**: Candidate 1's decision text says "delete both
rows", and that is what invited the error. A future candidate of this shape should say
*unwire* rather than *delete* whenever a preserved artifact could reference the symbol.

---

## 3. What landed in Set 1

**Candidate 1** — power-factor rows unwired, removed from
`_SHARED_ESS_OPERATIONAL_CONSTRAINTS` (which would otherwise raise on every capacity-activation
cycle); `sess_converter_capability` carries the reactive limit alone.

**Candidate 2** — capacity `Var`s and both pinning equalities gone; every reader rewired to the
`_fixed` `Param`s. `network_data._get_sensitivities` is **deactivated behind a guard**,
returning a well-formed all-`None` structure; its three consumers already tolerate `None`,
verified by reading each.

**Candidate 4** — the slacked-equality branch is pinned on in `network_parameters.py`, because
the flag lives in case params under `data/`, which was out of bounds. This made a **latent bug
reachable**: `slack_penalties` called `sum()` on a scalar expression and would have raised at
build time. The audit had flagged it as unreachable dead code; it is now reachable and fixed.

**Candidate 5** — both scenario-deviation penalties skipped at one scenario; verified absent on
a built model, and all seven call sites already guard with `hasattr`.

**Smoke test after the 3′ revert**: DSO and TSO both optimal, DSO objective back to
**−0.9473847716012889**, matching the A/B's "before" value exactly — confirming the revert is
clean and that 3′ alone moved it.

---

## 4. Still open, and needing a decision

1. **Candidate 3′** — drop, or re-gate on a different fixture/capacity, or diagnose the
   low-availability mechanism first? The Planner would not commit it on one fixture where it
   worsened cold-start iterations.
2. **`convex_oracle.py`** calls the retired power-factor rules directly and will raise when its
   shared-ESS block builds. A separate P5.5-C diagnostic module, not one of the two network
   models the brief names. Unfixed, flagged.
3. **`EPS_ESSO_THROUGHPUT`** — carried forward from the previous report. Measured gradient ratio
   59–171× the AL gradient; the Planner's reading is that displacement, not gradient, is the
   comparator (≈4e-7 p.u.), with a two-solve sensitivity check scheduled before the gates.
4. **Four dead `hessian_approximation: limited-memory` entries** in `recovery_options` across
   the ESSO params and three network case files. Not edited.
5. **Historical harnesses broken** by Step 1 and Step 1b, none production:
   `p513_e_gated_capture.py`, `p54h1_gate.py`, `p54f_admm_net_pq.py`, `p55c_c0_traces.py`,
   `audit_p3_snapshots.py`, `p54d_lifecycle_sensitivity_audit.py`,
   `p54d2_sensitivity_root_cause.py`, `p54d2p_validation.py`, `validate_vmag_refactor.py`;
   `p57_fingerprint.py` degrades gracefully. **Two more must be repaired because G1–G3 depend
   on them**: `p514_n_instrumented_cstar.py` and `p514_l_capacity_ladder.py`.

---

## 5. What is NOT established

- **Nothing in Steps 1, 1a or 1b is gated yet.** G1–G5 are what would establish that the SoH
  trajectory, EFC and recourse survive the reformulation, that the perturbation arm converges,
  and that A1/A3/A4 agree to 1e-6 on the reformulated model.
- The ε value is unsettled.
- 3′'s failure mechanism is a plausible hypothesis, not a proof, and was measured on one
  fixture at one capacity.
- No claim is made that the network-side cycle-21 mechanism is resolved beyond Step 0's single
  fixture.
