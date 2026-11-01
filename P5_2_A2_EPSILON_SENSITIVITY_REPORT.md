# Stage P5.2-A2 — Narrow-band epsilon sensitivity on the outstanding primary-path case

Diagnostic only. **No production code was changed.** The full 51-solve
initialization was **not** run.

## Provenance

- Script: `p52a2_epsilon_sensitivity.py`. Raw output:
  `data/SRP1/Results/P52A2/p52a2_report.json`.
- Git `f77d829359ff…` (P4.6-B2); tracked working tree clean throughout.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  identical to P5, P5.1, P5.1-B and P5.2-A.
- Target: `case33_2 / node 7 / 2030 / Summer`, exact P5 positive-bootstrap cold
  initialization state, built by replaying the production
  `create_distribution_networks_models_sequential` sequence and stopping before
  IPOPT. Band applied in place on the existing `sess_snet_def` `ConstraintData`
  objects. Machinery reused from `p51_…` and `p52a_…`.
- Held fixed in every branch: `kappa = 1/S_rated = 4701.5` (untouched
  production), the same ranged formulation, primal initial state, bounds,
  objective, `sess_comp`, SOC, IPOPT options, MA97, exact-Hessian primary,
  recovery policy, `from_warm_start=False`. Discipline checks passed in all
  branches (`all OK`).

### Parsing correction (disclosed)

These logs are written with `file_print_level = 6`, so the metric names also
appear on ~1000 per-iteration diagnostic lines. A first pass parsed those and
produced incoherent per-attempt KKT numbers. The parser was corrected to anchor
on the final-summary block (`Number of Iterations....:`, the only place that
token appears with dots + colon) and the whole stage was **re-run**. The
success/failure outcomes were identical across both passes — only the KKT
metrics changed, and the numbers below come from the corrected pass.

---

## 1. Target case — epsilon ladder (§ required metrics)

`S_rated = 0.02127 MVA = 2.1270e-04 p.u.`, `kappa = 4701.5`. All values below
are from the **primary** attempt unless labelled recovery; scaled values quoted.

| | **ε = 1e-5** | **ε = 3e-5** | **ε = 1e-4** |
|---|---|---|---|
| `ε_abs = ε·S_rated²` [p.u.²] | 4.5240e-13 | 1.3572e-12 | 4.5240e-12 |
| Scaled band `±κ·ε_abs` | ±2.1270e-09 | ±6.3809e-09 | ±2.1270e-08 |
| **Primary status** | **`Restoration Failed!`** | **`Optimal Solution Found`** | **`Optimal Solution Found`** |
| **Recovery needed** | **Yes** | **No** | **No** |
| **Primary iterations** | 664 | **1175** | **883** |
| Recovery iterations | 116 | — | — |
| Objective (primary) | 1.85174e+02 † | **−8.2259355555e-04** | **−8.2259338217e-04** |
| Primal infeasibility | 2.7863e-04 | **4.6105e-14** | **4.3450e-14** |
| Dual infeasibility | 1.0000e+03 | **7.5407e-08** | **3.6861e-06** |
| Complementarity | 1.2621e+00 | **9.0909e-09** | **9.0918e-09** |
| Overall NLP error | 6.8136e+02 | **7.5407e-08** | **3.6861e-06** |
| Runtime | 6.9 s | 11.8 s | 9.5 s |
| Final result | `ok`/`optimal` (via recovery) | `ok`/`optimal` | `ok`/`optimal` |
| **`max｜g｜/S_rated²`** | 2.7915e-10 | 2.4782e-06 | 3.5584e-06 |
| **Band utilization** `(max｜g｜/S²)/ε` | 2.7915e-05 | **8.2605e-02** | **3.5584e-02** |
| **max ΔS** [p.u.] | 1.1901e-09 | 8.6665e-08 | 1.2271e-07 |
| **max ΔS** [MVA] | 1.19e-07 | 8.67e-06 | **1.23e-05** |
| **max ΔS / S_rated** | 5.5955e-06 | 4.0746e-04 | **5.7690e-04** |
| Periods within 10 % of a boundary | **0 / 24** | **0 / 24** | **0 / 24** |
| Periods active at a boundary | **0 / 24** | **0 / 24** | **0 / 24** |

† ε = 1e-5's primary attempt ended in restoration failure, so its objective and
KKT values describe a failed point, not a solution. Its recovery attempt
converged cleanly (`Optimal Solution Found`, dual infeasibility 9.1745e-06).

**Both ε = 3e-5 and ε = 1e-4 restore a clean primary exact-Hessian solve** on
the target, with excellent KKT quality (primal infeasibility ~4e-14,
complementarity ~9.1e-09). Their objectives agree with each other to 1.7e-10 and
with the P5.2-A hard-equality branch-A objective (−8.2259349e-04) to ~1e-9 — so
widening the band does not move the solution economically.

## 2. Controls (§ matched-success verification)

The three P5.2-A matched-success controls were re-run at each epsilon that gave
a clean primary success on the target.

### ε = 3e-5 — **a control regresses**

| Control | Final | Clean primary | Primary iters | Primary exit | Dual infeas | ΔS/S_rated |
|---|---|---|---|---|---|---|
| **n5 2025 Winter** | **`error`/`internalSolverError`** | **No — FAILED** | 137 | **`Restoration Failed!`** | 1.0000e+03 | — (no solution) |
| n7 2025 Winter | `ok`/`optimal` | Yes | 249 | Optimal Solution Found | 4.2299e-06 | 3.331e-04 |
| n9 2030 Summer | `ok`/`optimal` | Yes | 940 | Optimal Solution Found | 1.3025e-07 | 4.069e-04 |

`n5 2025 Winter` does not merely lose its primary path — it **fails outright**.
No recovery was attempted because `case33_1` (node 5) defines no
`recovery_options` (a configuration asymmetry first noted in P5.1-B). Under
ε = 1e-5 in P5.2-A this same control was `optimal` in 298 iterations. This is a
genuine regression.

### ε = 1e-4 — **all controls clean**

| Control | Final | Clean primary | Primary iters | Objective | Primal infeas | Dual infeas | Complementarity | ΔS/S_rated | ΔS [MVA] | band util | near 10 % | active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n5 2025 Winter | `ok`/`optimal` | **Yes** | 139 | −7.9644347e-01 | 6.1663e-14 | 3.3450e-08 | 9.0909e-09 | 5.226e-04 | 5.56e-06 | 7.635e-02 | 0/24 | 0/24 |
| n7 2025 Winter | `ok`/`optimal` | **Yes** | 952 | −7.9499630e-01 | 5.0878e-14 | 1.9071e-06 | 9.0922e-09 | 5.235e-04 | 5.57e-06 | 7.642e-02 | 0/24 | 0/24 |
| n9 2030 Summer | `ok`/`optimal` | **Yes** | 1125 | −8.2277925e-01 | 4.8358e-14 | 2.9789e-08 | 9.0909e-09 | 5.810e-04 | 1.24e-05 | 3.578e-02 | 0/24 | 0/24 |

All three converge on the primary exact-Hessian path with no recovery, with
objectives matching their P5.2-A values to ~1e-9 and unchanged qualitative
dispatch.

## 3. Recommendation

| ε | Target clean primary | Controls all clean primary | Acceptable |
|---|---|---|---|
| 1e-5 | **No** (recovery required) | (not tested here; all succeeded in P5.2-A) | No |
| 3e-5 | Yes | **No** — `n5 2025 Winter` fails outright | **No** |
| **1e-4** | **Yes** | **Yes (3/3)** | **Yes** |

**Recommended: `epsilon_rel = 1e-4`.**

The instruction is to recommend the smallest successful epsilon *provided the
controls remain satisfactory*. ε = 3e-5 is smaller and does fix the target, but
it **breaks a matched-success control**, so it is not a valid candidate. The
smallest tested epsilon that satisfies both requirements is **1e-4**.

> Note: the raw JSON field `recommended_epsilon` reads `3e-05`, because the
> script computes that field from target-case success alone, before the control
> results exist. The correct recommendation, taking the controls into account as
> the stage requires, is **1e-4**. Reported here rather than silently deferring
> to the script's field.

### Cost of the wider band

Moving ε from 1e-5 to 1e-4 raises the worst-case physical mismatch from
`ΔS/S_rated = 5.6e-06` to **5.8e-04** — i.e. from ~0.0006 % to ~0.06 % of rated
power, or in absolute terms from 0.12 VA to **≈ 12 VA** on a 21 kVA unit.
`max|g|/S_rated²` stays at **3.6e-06**, still well inside
`EQUALITY_TOLERANCE = 1e-5`. Band utilization never exceeds **0.115** across all
tested cases and epsilons, and **no period in any case is within 10 % of a
boundary, let alone active** — so the wider band is still not being economically
exploited; it is purchasing interior-point conditioning, not feasibility slack.

## 4. Interpretation — kept narrow

1. **Answered.** A modestly wider band does restore a clean primary
   exact-Hessian solve on the outstanding case: the ε = 1e-5 primary attempt
   ends in `Restoration Failed!` with dual infeasibility 1.0e+03, while ε = 3e-5
   and ε = 1e-4 both terminate `Optimal Solution Found` with dual infeasibility
   7.5e-08 and 3.7e-06 respectively.
2. **Established, and it constrains the choice.** Epsilon selection is not
   monotone in "bigger is safer for the target": ε = 3e-5 fixes the target but
   destroys a previously-passing control. Only ε = 1e-4 satisfies target and
   controls simultaneously among the three tested values.
3. **Not established.** That ε = 1e-4 clears the full 51-solve initialization
   (not run, per instruction); that it is safe through ADMM or the outer
   planning loop; that any untested epsilon behaves well; or that the node-5
   missing `recovery_options` asymmetry is unrelated to its sensitivity. Only
   the three instructed epsilons were tested — no others.
4. **No production change is authorized by this stage.** Productionizing any ε
   remains a separate planner decision.

## 5. Scope compliance

`kappa`, the ranged formulation, primal initial state, bounds, objective,
`sess_comp`, SOC, IPOPT options, MA97, exact-Hessian primary, recovery policy
and `warm_start=False` were all held fixed. Only `ε_rel` varied, and only across
the three instructed values. Production `sess_snet_def`,
`shared_ess_snet_def_scale`, ordinary ESS, ADMM and planning logic are
untouched. The full 51-solve initialization was not run.

---

```
P5.2-A2 COMPLETE — waiting for planner review
```
