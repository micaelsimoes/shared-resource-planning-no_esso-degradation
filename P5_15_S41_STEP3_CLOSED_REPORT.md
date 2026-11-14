# P5.15 Addendum 23 — Step 3 closed: the hull polish passes at 0.000309 %

**Planner report, 2026-09-18; stop for review.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 23. Frozen spec:
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` (`a099c7c5`).

## 1. Verdict

**Step 3 is closed.**
- **The restated Step 3.5 gate passes.** At the oracle's certified point the interval-hull polish gap is
  **|Δ|/cost = 0.000309 %** against 0.1 %. All 48 blocks solve without retries, and Δ ≤ 0 as predicted.
- **The baseline is recorded.** The head of `REVISION_CONTEXT.md` now records arm D as the baseline ADMM configuration,
  with the R2.5 package, the campaign rules and the updated manuscript list.
- **One finding reshapes an Addendum 23 instruction:** the v11 exact-fix failures were confounded by a stale harness
  helper (§3).

| order item (Addendum 23) | status | commit |
|---|---|---|
| 3.5 log read, harness checks | done — no defect on the two named axes; a **different** defect found (§3) | `f5dfcb3f` |
| hull-polish harness, checks, smoke | done — 48/48 solved in the smoke test | `5c460c2c`, `e48d51ef` |
| gate-quantity correction (Planner) | sum of per-block changes, as spec v12 defines (§2.3) | `8170150c` |
| hull polish, full run | **PASS**, 0.000309 % | `2e6c5570` |
| `REVISION_CONTEXT.md` head; Step 3 closed | done | this commit |
| persistent-worker bounded task | next, after review | — |
| Step 3.7 | implemented in a detached worktree, default off, not integrated | worktree `c5d18add`, `eacbdfcc`, `040ca67e` |

## 2. Step 3.5 — the interval-hull polish

### 2.1 Setup and reproduction

The case-file configuration ran to certification in-process and reproduced arm D **bitwise**: cycle 139, cost
650,966,975.2943751, 0 gating differences, 1 reported provenance difference. This is the third full reproduction.
Each coupling entry was then bounded to the closed interval of the agents' achieved values at cycle 139 (two agents for
V and PF, three for ESS). The polish used the unscaled base objective, a primal warm start from the certified block
solution, the IPOPT default bound push (overriding the case file's 1e-6 pushes for the polish window), no multiplier
import, and the production recovery policy.

### 2.2 Result

| quantity | value |
|---|---|
| blocks solved | **48 / 48**, 48 solves, 0 recovery retries |
| Δ = Σ_i [f_i(polished) − f_i(certified)] | **−2,012.21** (TSO −33.82, DSO −1,978.39) |
| \|Δ\| / certified cost | **3.09e-6 = 0.000309 %** — gate < 0.1 %: **PASS** |
| Δ ≤ 0 (predicted) | yes |
| max \|Δ_i\| | 775.40, DSO7 2035 Spring |
| blocks with Δ_i > 0 | 6, the largest +0.39 (TSO 2030 Spring) — none near the 1e-6·cost flag (650.97) |
| Σ_i f_i(certified) vs cost including settlement | reconciles to 2.4e-7 |
| hull bounds active at the polished point | V 564, PF_P 458, PF_Q 197, ESS_P 27, ESS_Q 1,267 (counts include degenerate intervals) |

**Reading.** Relaxing each coupling entry to its hull lets every block improve its own base objective by at most 775,
and by 2,012 in total, 3e-6 of cost. The augmented-Lagrangian distortion at the certified tolerance is therefore
negligible: the oracle's point is block-stationary to well within the gate. This is not global-optimality evidence for
the nonconvex problem; Addendum 23 frames it as one element of the R2.5 package.

### 2.3 A gate-quantity correction made before the run

The harness as delivered gated on the change in the **settlement-excluded** cost. Spec v12 defines Δ as the sum of
per-block changes: the quantity each block minimises, and the one for which Δ ≤ 0 holds by construction. The two differ
by the interface settlement transfer, which cancels between TSO and DSO only when both hold the same interface values; after
independent polishing within the hull they need not. I corrected the gate before launch (`8170150c`). Reported, not
gated, in the full run:
- the settlement-excluded cost change is **+11,967**, 0.0018 %;
- the interface settlement total moves from 36,679 to 22,700.

## 3. The exact-fix failures were confounded by a stale harness helper

The zero-solve log read (`WORKER_REPORT_S41_POLISH_PREREQ.md`) found that the six autumn/winter TSO failures of the v11
exact-fix run had a constraint violation of **0.20–0.84 pu, frozen to 16 significant figures from iteration 0**, across
warm, cold and adaptive restarts. That is two to five orders of magnitude beyond any consensus residual: a constraint
row with no free variable. The DSO failures are numerical; given enough iterations they reach 1e-6–1e-7.

Building the hull harness found the cause (`WORKER_REPORT_S41_HULL_POLISH_PREP.md` §2, confirmed in the code):
- **What the helper does.** `p56a_oracle._interface_expression` builds the TSO interface flow as
  `pc + flex_up − flex_down`.
- **Why that is wrong since Addendum 12.** The flexibility legs are fixed at 0 and **all** interface deviation is carried
  by `interface_delta_p/q`, which the helper omits.
- **The consequence.** `apply_common_values` equated two constants that generically differ.

Consequences:
- **Addendum 23's diagnosis is not established.** Its reading — the midpoint breaching active coupling constraints —
  and my earlier attribution of six TSO failures to node 7's rating are both confounded. The rating-midpoint fact (node 7
  midpoints exceed 100 MVA by up to 5e-4 MVA) still holds as a fact about the data, but its causal role in the failures
  is not shown.
- **The manuscript sentence Addendum 23 asked for** ("exact fixing … 17/48 infeasible" as a methodological finding)
  **needs restating.** It is listed as to be restated in `REVISION_CONTEXT.md`.
- **Scope is limited.** A search of every consumer shows only the v11 exact-fix run is affected. P5.5–P5.8 predate
  Addendum 12, `p515_s31c_zero_solve_checks.py` only matches by name, and the hull harness bounds the model's own
  `pc_adn` expressions and bypasses the helper.
- **The helper itself is not fixed:** it is shared code, and fixing it was outside the Worker's authority. Fixing it,
  and optionally re-reading or re-running the exact fix, awaits your authorization.

The Worker also fixed two narrower defects inside the new harness:
- **storage-model indexing:** the ESSO models are indexed by position, not label; without the fix the full run would
  have crashed;
- **a defensive bound-unfix:** it prevents a zero-capacity entry being fixed at 0 rather than bounded.

## 4. Other items in this round

- **Node 7 multiplier:** the rating constraint's multiplier at the certified TSO solves was never captured in any
  committed artifact (search recorded). The polish-derived multiplier is not used, per Addendum 23.
- **Step 3.7 (code only, worktree):**
  - **Implementation:** type-II AA, memory 5, Tikhonov 1e-10, on stacked (z, u) in the residual-test scaling; active
    from cycle 1; memory cleared at every ρ change (state-based); ratchet safeguard without extra solves; off while all
    channels are inside tolerance. Peak RSS is recorded.
  - **Checks:** all zero-solve checks pass. The dual round trip is exact to 2.7e-16, not bitwise, because of two
    chained non-power-of-two factors.
  - **Choices I accepted:** a cycle with a local-solve failure skips AA and clears the memory; the safeguard mark is
    left unchanged on rejection.
  - **Still needed:** the harness arm for the 3.7 run, and the flag-off two-cycle bitwise gate at integration.

## 5. Questions for review

1. **The stale `p56a_oracle` helper:** authorize the fix (include `interface_delta_p/q`)? And should the v11 exact-fix
   be re-read, or re-run as a methodological record, before the manuscript sentence is restated?
2. **Proceed with the persistent-worker bounded task, then Step 3.7, as ordered?** Step 3.7 would integrate the worktree
   code, add the harness arm, and run the flag-off gate and then the flag-on run.

## Evidence

- Spec v12; `WORKER_REPORT_S41_POLISH_PREREQ.md`, `WORKER_REPORT_S41_HULL_POLISH_PREP.md`.
- `p515_s41_hull_polish{,_checks}.py`; `data/SRP1/Results/P515S41/{hull_polish,hull_polish_smoke,hull_polish_checks}/`
  with manifests; `data/SRP1/Results/P515S41/hull_polish_launch.log`.
- Worktree `wt_step37_aa` with `WORKER_REPORT_S41_AA.md` and `data/SRP1/Results/P515S41/aa_checks/`.
- Zero-solve claims are backed by armed `SolveProfileGuard`s.
