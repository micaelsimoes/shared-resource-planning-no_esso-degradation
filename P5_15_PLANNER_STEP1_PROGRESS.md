# P5.15 — Planner progress report: pre-Step-1 check, Step 1a, ESSO reformulation

**Prepared for the author and the external expert. Supersedes nothing; continues
`P5_15_PLANNER_STEP0_STEP2_REVIEW.md`.**

Authority: `PLANNER_BRIEF_2026-09-13.md` **including Addendum 1 as amended** (Candidate 2
unblocked, Candidate 3 withdrawn and replaced by 3′).

| stage | state | commit |
|---|---|---|
| pre-Step-1 `.nl` check | **complete — question answered negatively** | (in this report) |
| Step 1a — solver policy | **complete, gates passed** | `6d2349af` |
| Step 1 — ESSO reformulation | **complete, smoke-tested; G1–G5 NOT yet run** | `b03c9b14` |
| Step 1b — candidates 1, 2, 3′, 4, 5 | **in flight** | — |
| Gates G1–G5 | not started | — |

---

## 1. Pre-Step-1 check — answered, and the conditional does not fire

The audit asked whether the `dual` suffix is exported on nominally cold solves, because if it
were, Step 0's arm A4 would not have been cold on the constraint-multiplier side.

Both preserved P5.12-R `.nl` files contain **exactly two** suffix segments:

```
S4 7492 ipopt_zL_in
S4 7320 ipopt_zU_in
```

Both are `S4` — variable kind, real values. A constraint suffix would be `S1`/`S5`; there is
none, and the string `dual` appears **zero times** in either file.

**Constraint multipliers are not exported**, despite `model.dual` being declared
`IMPORT_EXPORT` at `network.py:480`. **Arm A4 was genuinely cold and stands as written; no
re-run was needed.** The addendum's recovery policy still clears `model.dual`, which is
correct belt-and-braces but not load-bearing.

---

## 2. Step 1a — adopted policy, gates passed

Two production files, 39/11 and 46/10 lines. No case file touched.

- the five hardcoded `warm_start_* = 1e-9` assignments deleted from the ESSO — which both
  fixes the `option_overrides` clobbering and leaves the pushes at IPOPT's default unless
  explicitly configured;
- the five **derived** assignments deleted from `network.py`, including
  `warm_start_mult_bound_push`'s derivation from `bound_push`;
- the TSO-only `acceptable_iter = 0` / `acceptable_tol = tol` override removed;
- `max_iter = 500` inserted **before** the option merge, so configuration still wins;
- both recovery predicates extended to `maxIterations` / `infeasible` /
  `internalSolverError`; recovery reduced to exactly one change — cold start with suffixes
  (including `dual`) cleared, same primary options, exact Hessian.

| gate | result |
|---|---|
| `clobber_probe.clobbered` | **False** (was `True`) |
| ESSO node 7, old policy via override | `maxIterations` — reproduction path intact |
| ESSO node 7, 1e-3 policy via override | **optimal, 30 iterations** (matches Step 0) |
| ESSO node 9, old → 1e-3 | `maxIterations` → **optimal, 23 iterations** |
| DSO fixture, default policy | **optimal, 85 iterations** |
| DSO fixture, cold | optimal, 77 iterations; agrees with default to **3.5e-9** |
| DSO, override `bound_push` only | no `warm_start_*` key derived — derivation confirmed gone |

### Erratum the DSO validation surfaced

The brief's Step 0 text, and the Planner's task derived from it, describe the network policy
as "1e-6 pushes". **That is the TSO's value and the code's fallback.** `case33_2` and
`case33_3` set the generic `bound_push`/`bound_frac`/`slack_bound_*` keys to **1e-5**, and the
pre-1a derivation `options.get('warm_start_bound_push', options.get('bound_push', 1e-6))`
therefore produced **1e-5** on the DSO. Re-running at 1e-6 converges in 96 iterations; at
**1e-5** it reproduces `maxIterations`, with an unconverged objective (1302.28) matching Step
0's A0 in character (1302.21 at the old 3000 cap).

**Step 0's results are unaffected** — A0 ran the production path with no override, hence the
real 1e-5. Only the description was wrong. The Advisor's audit had it right at 1e-5
throughout.

---

## 3. Step 1 — the ESSO reformulation

197 insertions / 375 deletions in `shared_energy_storage_data.py`, plus small additions to
`definitions.py`, `shared_energy_storage_parameters.py` and `shared_resources_planning.py`.

- **Investments as parameters.** The investment `Var`s, their four slacks and
  `energy_storage_capacity_fixing` are deleted; the pre-existing `_fixed` mutable `Param`s are
  consumed directly. The rated-capacity `Var`s were **kept as `Var`s** so the
  cohort-deactivation bookkeeping needed no change — which the brief permits.
- **Log-domain SoH chain.** `es_D_per_unit` replaces `es_soh_per_unit`,
  `es_degradation_per_unit` and `es_degradation_per_unit_cumul`, with the linear row
  `D·(2·cl_eff·E_fixed) == 365·num_years·avg_ch_dch` and
  `soh_cumul == prev·exp(−D)·phi_cal**num_years`, floor `soh_cumul ≥ soh_min`. `phi_cal` is a
  new `ageing.calendar_retention_per_year` defaulting to **1.0**; no case file touched. The
  equal-width-block caveat on `num_years` is recorded in a comment, not changed.
- **Complementarity by regularization.** The complementarity rows, the `*_hat_*` variables,
  the normalization rows, `slack_es_ch_comp_per_unit` and the aggregate row are deleted;
  `feasibility_penalty` gains `EPS_ESSO_THROUGHPUT·Σ(pch + pdch)` with
  `EPS_ESSO_THROUGHPUT = 1e-3`. A post-solve **detector** (not a cost) reports
  `max min(pch, pdch)` against `1e-6·s_max`.
- **Master side.** `_add_benders_cut` returns `False` unconditionally, body retained with a
  comment. **This goes further than "the slack-based branch"**, because the Worker searched and
  found no separate recovery-cut path. It is consistent with Addendum 1's confirmation that the
  sensitivity channel is retired, and is flagged here for the author rather than assumed.

**Nine slack families removed in total.** One bilinear row is deliberately retained and
documented: `available_e_capacity_unit` (`es_e_available == es_e_rated · soh_cumul`), because
substituting the `Param` is valid only inside the calendar-life window.

**Smoke test, through the production entry point** (`create_shared_energy_storage_model`,
model rebuilt from a planning problem — the P5.14-N pickles hold the old structure):

| node | termination | notes |
|---|---|---|
| 5 (zero investment) | optimal | — |
| **7** (previously failing) | **optimal** | SoH `1.0 → 0.9082 → 0.8248 → 0.7491`, floor not binding |
| **9** (previously failing) | **optimal** | same |

Complementarity detector max violation **4.62e-4** against `s_max = 1.0` — small, nonzero,
consistent with complementarity being regularized rather than enforced.

### The required ε/AL measurement, and the Planner's reading of it

At the P5.14-N control state, with `EPS_ESSO_THROUGHPUT = 1e-3`:

| node | AL grad max | AL grad mean | ratio ε/AL (max) | ratio ε/AL (mean) |
|---|---|---|---|---|
| 7 | 1.692e-05 | 5.832e-06 | **59.1** | **171.5** |
| 9 | 1.693e-05 | 5.872e-06 | 59.1 | 170.3 |

The Worker reported this as a conditioning concern and correctly declined to change the value.

**The Planner's reading is that gradient ratio is the wrong comparator, and that the right one
is displacement.** The AL gradient is small at a *converged* point precisely because the
residual is near zero. What bounds the distortion is ε divided by the AL **curvature**: with
`ρ_ess = 1` and the normalization `2·max(S, 0.10)/100`, the curvature is ≈ **2660 per p.u.** at
`S = 0.97 MVA`, so `ε = 1e-3` implies a displacement of ≈ **3.8e-7 p.u. ≈ 0.04 kW** on a
0.97 MVA device — negligible, and far below the `ess` consensus tolerance.

**This is arithmetic, not measurement.** A two-solve ε sensitivity check (1e-3 versus 1e-7 on
the same request, comparing dispatch and objective) is scheduled **before** G1–G5, so the
question is settled empirically rather than by my algebra. If the dispatches differ materially,
ε is distorting and the gates would have been measuring ε rather than the reformulation.

---

## 4. A prerequisite the brief did not anticipate

**Gates G1–G3 depend on harnesses the reformulation breaks.** G1 and G2 re-run the P5.14-N
arms; G3 runs the P5.14-L ladder. But:

- `p514_n_instrumented_cstar.py` hard-asserts on `es_degradation_per_unit` in its
  rule-eleven capture checklist — **broken**;
- `p514_l_capacity_ladder.py` looks up the constraint row `energy_storage_complementarity`,
  which no longer exists — **its gate logic is affected**.

Repairing these two is therefore a **prerequisite for G1–G3**, not optional cleanup, and is
scheduled with the gate run. Eight further historical harnesses break and stay broken, as the
brief permits: `p513_e_gated_capture.py`, `p54h1_gate.py`, `p54f_admm_net_pq.py`,
`p55c_c0_traces.py`, and `p57_fingerprint.py` degrades gracefully.

---

## 5. Two observations recorded, not acted on

**A possible flat direction in the reformulated LP.** Under a deliberately aggressive ±50 %
duty cycle, the model satisfies the charging half with real `pch` but falls back to the heavily
penalized `slack_es_pnet_down` for the discharging half once the SoH floor approaches binding —
because both `pch` and `pdch` add to throughput and hence to `D`. This is consistent with the
governing decision that the SoH floor is the only intertemporal trade-off in operation, but it
may indicate multiple equally optimal `pch`/`pdch` splits rather than a unique optimum.
Recorded in `data/SRP1/Results/P5151/p5151_floor_stress_probe.json`; not investigated.

**Four dead case-file entries.** `hessian_approximation: limited-memory` in `recovery_options`
— the ESSO params file plus all three network case files — is now dead configuration, since
recovery uses the exact Hessian so that a recovery success identifies its cause. **Not
edited**; raised for the author as one item.

---

## 6. What is NOT established

- **The reformulation is smoke-tested, not gated.** G1–G5 are what would establish that the
  SoH trajectory, EFC and recourse are preserved, that the perturbation arm now converges, and
  that A1/A3/A4 agree to 1e-6 on the reformulated model.
- **The ε value is unsettled** pending the sensitivity check above.
- Step 1b is in flight; its candidates are unverified at the time of writing.
- No claim is made that the cycle-21 mechanism is resolved for the networks beyond what Step 0
  measured on one fixture.
