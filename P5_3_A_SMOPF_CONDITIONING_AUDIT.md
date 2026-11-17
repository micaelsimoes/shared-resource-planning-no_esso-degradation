# P5.3-A — Quantitative structural SMOPF conditioning audit

Diagnostic only. **No production code was changed.**

## Provenance

- Script: `p53a_conditioning_audit.py`. Raw output:
  `data/SRP1/Results/P53A/p53a_audit.json`.
- Git `3c9b79fc` (docs) on production checkpoint `f77d829359ff…`; tracked
  working tree clean apart from the new diagnostic script/report.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  the authorized current reduced SRP1 identity.
- **A1 population:** the real P5 iteration-2 positive-bootstrap candidate from
  production `_build_positive_bootstrap_candidate`, replayed through the real
  production initialization path. `Network.run_smopf` was wrapped so each model
  is audited at its exact cold start immediately before IPOPT, after which the
  real solve proceeds — so DSO results populate the consensus variables and the
  TSO models are audited at their own true cold start.
  **48 network models audited: 36 DSO + 12 TSO.** No manual reconstruction; the
  old cycle-10 pickle was not used.

## A2 — Derivative environment (reported before choosing a method)

`pyomo.contrib.pynumero` imports, but **`AmplInterface.available()` is
`False`** — the ASL shared library is not installed, so `PyomoNLP` and its
sparse analytic Jacobian callback are unavailable. Per plan A2 no large
dependency was installed.

**Method used instead:** Pyomo's own reverse-mode automatic differentiation
(`differentiate(..., Modes.reverse_numeric)` for first derivatives,
`reverse_symbolic` then `reverse_numeric` for second derivatives). This is
analytic and exact at the evaluation point.

**Limitation:** derivatives are taken row-by-row in Python rather than through a
single sparse ASL callback. This is fast enough here (0.2–0.4 s per model) and
exact, but it is not the same instrumentation an ASL build would give, and
extremal singular values rely on sparse iterative methods (below).

---

## A3/A4 — Constraint families and cold-start Jacobian diagnostics

Every model shows **exactly 72 zero-gradient active rows** — identical in all
36 DSO and all 12 TSO models. Aggregated over all 48 models:

| Component | Active rows | **Zero-gradient rows** | grad‖·‖₂ min | grad‖·‖₂ max | min margin / tol |
|---|---|---|---|---|---|
| **`sess_snet_def`** | 1728 | **1728 (100 %)** | **0.00** | **0.00** | n/a (equality) |
| **`branch_flow_limit`** | 30240 | **864** | 0.00 | 1.32e+05 | 1.60e+04 |
| **`branch_flow_limit_ji`** | 864 | **864 (100 %)** | 0.00 | 0.00 | 1.00e+05 |
| **`sess_comp`** | 1728 | 0 | **2.13e-08** | 6.38e-08 | **1.13e-07** |
| **`sg_capability`** | 3732 | 0 | **5.44e-05** | 9.49e-01 | **0.00** |
| `sess_pch_link` | 1728 | 0 | 1.41 | 1.41 | **0.00** |
| `sess_pdch_link` | 1728 | 0 | 1.41 | 1.41 | **0.00** |
| `sess_phi_limit_lower` | 1728 | 0 | 1.21 | 1.21 | **0.00** |
| `sess_phi_limit_upper` | 1728 | 0 | 1.21 | 1.21 | **0.00** |

Gradient-norm bins per model (identical across models): `<1e-12`: 72,
`<1e-10`: 72, `<1e-8`: 72, `<1e-6`: 96, `<1e-4`: 96.

Per-model composition of the 72 zero-gradient rows:

- **DSO:** 24 `sess_snet_def` (equality) + 24 `branch_flow_limit` + 24
  `branch_flow_limit_ji` (both inequalities);
- **TSO:** 72 `sess_snet_def` (3 shared ESS × 24 periods), all equalities.

**The critical distinction the plan asks for.** All three families are squared
magnitudes evaluated at a zero point, so all have exactly vanishing gradients —
but their consequences differ sharply:

- `sess_snet_def` is an **equality**, so it is **always in the active set**. A
  zero gradient there is a genuine LICQ failure at the cold start.
- `branch_flow_limit` / `branch_flow_limit_ji` are **inequalities that are
  strictly inactive** (minimum margin `1.6e+04·tol` and `1.0e+05·tol`
  respectively). A zero gradient on a strictly interior inequality is
  numerically harmless.

## A5 — Column and rank diagnostics

| | DSO models | TSO models |
|---|---|---|
| Columns with derivatives | 9244 | ~2748 |
| Columns with norm `<1e-10` | **48** | 0 |
| Smallest-norm columns | `pij[31,…]`, `qij[31,…]` | **`f[0,…]`** (reference-bus imaginary voltage) |

Equality-Jacobian extremal singular values (sparse `svds`, lobpcg/propack):

| Model | Eq. rows | Cols | Exactly-zero rows | σ_min | σ_max | **Condition** | Near-collinear pairs |
|---|---|---|---|---|---|---|---|
| case33_1/2025/Winter | 4131 | 7612 | 24 | 4.67e-02 | 4.75e+02 | 1.02e+04 | 0 |
| case33_1/2030/Spring | 4131 | 7612 | 24 | 4.97e-02 | 4.75e+02 | 9.55e+03 | 0 |
| case33_1/2030/Winter | 4131 | 7612 | 24 | 5.23e-02 | 4.75e+02 | 9.08e+03 | 0 |
| case33_1/2035/Autumn | 4131 | 7612 | 24 | 5.39e-02 | 4.75e+02 | 8.80e+03 | 0 |
| case33_1/2035/Winter | 4131 | 7612 | 24 | 4.48e-02 | 4.75e+02 | 1.06e+04 | 0 |
| case33_2/2025/Winter | 4131 | 7612 | 24 | 4.63e-02 | 4.75e+02 | 1.02e+04 | 0 |
| case33_2/2030/Summer | 4131 | 7612 | 24 | 5.03e-02 | 4.75e+02 | 9.44e+03 | 0 |
| case33_2/2035/Spring | 4131 | 7612 | 24 | 4.60e-02 | 4.75e+02 | 1.03e+04 | 0 |
| case33_3/2025/Summer | 4131 | 7612 | 24 | 4.29e-02 | 4.75e+02 | 1.11e+04 | 0 |
| case33_3/2025/Autumn | 4131 | 7612 | 24 | 5.45e-02 | 4.75e+02 | 8.71e+03 | 0 |
| case33_3/2030/Summer | 4131 | 7612 | 24 | 4.89e-02 | 4.75e+02 | 9.71e+03 | 0 |
| **case9/2025/Winter** | 1737 | 2748 | **72** | **2.48e-05** | 4.68e+01 | **1.88e+06** | 0 |
| **case9/2030/Spring** | 1737 | 2748 | **72** | **1.91e-05** | 4.68e+01 | **2.45e+06** | 0 |

**Major finding: the TSO equality Jacobian is ~200× worse conditioned than any
DSO's**, and its smallest singular value (`1.9e-05`–`2.5e-05`) sits at the same
order as IPOPT's `tol = 1e-5`. This is an independent, quantitative explanation
for why P5.2-A3's worst physical band-utilization rows were **all TSO rows**.

**No near-collinear equality row pairs** were found in any model, so there is no
exact duplicate-row redundancy; the rank deficiency that exists is the exact
zero-row kind counted above.

## A6 — Curvature

Maximum absolute second derivative by family, over all 48 models:

| Family | max ‖∂²‖ |
|---|---|
| **`sess_snet_def`** | **18806** |
| `voltage_mag_def`, `voltage_mag_sqr_def`, `r_sqr_def`, `sg_capability`, `branch_flow_limit_ji` | 2 |
| `voltage_product_real_def`, `voltage_product_imag_def`, `sess_comp` | 1 |
| `branch_flow_limit` | 0 |

`18806 = 2·κ = 2/S_rated` at `S_rated = 1.0635e-04 p.u.` For the three bootstrap
ratings the curvature is `2κ` = **18806 / 9403 / 6269** respectively — i.e. the
accepted P4 normalization makes this row's curvature scale as `O(1/S_rated)`.

**This is the precise pathology, and it is not generic "bad conditioning":**
`sess_snet_def` simultaneously has an **exactly zero first derivative** and a
**second derivative four orders of magnitude larger than every other family**.

## A7 — Objective-gradient scale

| | Range over 48 models |
|---|---|
| ‖∇f‖₂ | 2.00e+06 – 1.25e+08 |
| ‖∇f‖_∞ | 1.00e+05 – 1.27e+07 |
| smallest nonzero partial | 1.00e+01 |

The objective gradient is `1e6`–`1e8` while the always-active `sess_snet_def`
rows contribute **exactly zero** first-order information and `sess_comp`
contributes `~1e-8`. That is a 14–16 order-of-magnitude disparity between the
objective and the critical constraint rows at the cold start. (Objective scaling
was not changed, per A7.)

---

## A-extra — Other structural checks

### Reference-angle / gauge

`f_bounds` (`model_construction_helpers.py:83`): for `BUS_REF`,
`f ∈ [−EQUALITY_TOLERANCE, +EQUALITY_TOLERANCE] = [−1e-5, +1e-5]`.

The gauge is therefore **not exactly fixed**: a residual rotational freedom of
±1e-5 remains, **exactly the size of IPOPT's `tol`**. Corroborating evidence:
the TSO models' smallest-norm Jacobian columns are precisely `f[0,…]`.

`e_bounds` (`:72`): for `BUS_REF` **in a DSO**,
`e ∈ [vg − SMALL_TOLERANCE, vg + SMALL_TOLERANCE] = vg ± 1e-4`.

**So the DSO reference-bus real voltage is effectively pinned to the generator
setpoint `vg` regardless of `enforce_vg`.** Together with `|f_ref| ≤ 1e-5`, the
coordinated DSO interface voltage magnitude is pinned to ≈ `vg` within ~1e-4.
This answers the plan's question directly: **yes, the bounds effectively pin the
coordinated interface voltage even when `enforce_vg = false`.** For the TSO
reference bus `e ∈ [0, component_max]` and is not pinned.

### Transformer auxiliaries — resolved as code cleanliness only

`case33_x` has 32 branches, of which **branch 31 is the only transformer** and
it is in service. Writing the `.nl` with symbolic labels gives **9268 variables,
of which exactly 24 are `r[…]` and 24 are `r_sqr[…]`** — i.e. only the single
transformer's 24 periods. The Var is declared over all branches, but the NL
writer **eliminates the unused non-transformer entries**; they never reach
IPOPT.

**Classification: code cleanliness only, not a conditioning issue.**

One related observation: `r` and `r_sqr` are created without an `initialize`
value, so they enter the solve with no Pyomo starting point (the audit surfaced
`No value for uninitialized VarData object r_sqr[31,…]`). Low risk, but it is a
cold-start initialization gap.

### Branch-current / apparent-flow rows

`branch_flow_limit` reaches coefficients up to `1.32e+05` (largest row gradient
anywhere in the audit) while its nonlinear subset has exactly zero gradient at
zero flow — a 5+ order-of-magnitude intra-family spread. However, every one of
these rows is **strictly inactive** at the cold start (margins `1.6e+04·tol` and
`1.0e+05·tol`). On this evidence the family does **not** rank materially high,
and per the plan it should **not** be reformulated in P5.3.

---

## Ranked risk table

| Rank | Family | Failure mode (specific) | Evidence |
|---|---|---|---|
| **HIGH** | **`sess_snet_def`** | **Exact zero gradient on an always-active equality, combined with `O(1/S_rated)` curvature** | 1728/1728 rows ‖∇g‖ = 0; ∂² = 18806; equality ⇒ permanently in the active set |
| **HIGH** | **TSO equality-Jacobian conditioning** | **Near rank deficiency at solver scale** | σ_min ≈ 1.9e-05–2.5e-05 ≈ `tol`; condition 1.9e+06–2.4e+06 vs ~1e+04 for DSOs |
| **HIGH** | **`sess_comp`** | **Tiny Jacobian on a numerically active bilinear inequality** | ‖∇g‖ ≈ 2.1e-08–6.4e-08; min margin ≈ 1.1e-07·tol (RHS scales as `S_rated²`) |
| **HIGH** | **`sg_capability`** (RES) | **Inequality exactly at its bound with a small gradient** | min margin **exactly 0.00**; ‖∇g‖ down to 5.4e-05; 3732 rows |
| **MEDIUM** | **Reference gauge `f_ref`** | **Residual gauge freedom equal to `tol`** | `f_ref ∈ ±1e-5`; TSO smallest Jacobian columns are `f[0,…]` |
| **MEDIUM** | Shared-ESS link/PF rows (`sess_pch_link`, `sess_pdch_link`, `sess_phi_limit_*`) | Active at bound at cold start (margin 0.00), but gradients healthy (1.2–1.4) | 4× 1728 rows |
| **MEDIUM** | DSO reference `e` pinning | Interface voltage effectively fixed to `vg ± 1e-4` irrespective of `enforce_vg` — a possibly unintended coupling | `e_bounds` source |
| **LOW** | `branch_flow_limit`, `branch_flow_limit_ji` | Zero gradient at zero flow, but strictly inactive | margins 1.6e+04·tol / 1.0e+05·tol |
| **LOW** | `r` / `r_sqr` transformer auxiliaries | Eliminated by the NL writer; uninitialized start values only | 24 of each in the `.nl` |

### HIGH-risk detail

**1. `sess_snet_def`** — form `κ·((sch−sdch)² − pnet² − qnet²) = 0`. Physical
role: ties shared-ESS apparent charge/discharge to its P/Q injection. Failure
mode: **exact zero gradient (rank degeneracy) on an always-active equality plus
large curvature** — not merely poor scale. Exact reformulations cannot remove
the zero gradient (any finite κ multiplies zero by κ). Deliberate physical
reformulations: the P5.2 two-sided band (makes the point interior — validated
structurally, but its declared band is below solver feasibility resolution), or
the **P5.3-B3 active-power ESS prototype**, which removes the row entirely.
Expected benefit: large. Feasible-set risk: none for the band (superset), and a
deliberate physical change for B3.

**2. TSO equality-Jacobian conditioning** — σ_min at solver tolerance. Physical
role: the TSO carries three shared ESS plus all interface coupling. Failure
mode: **near dependence**, not zero rows alone. Candidate contributors: the
`f_ref` gauge freedom (smallest columns are `f[0,…]`) and the 72 zero
`sess_snet_def` rows. B1 and B3 both target contributors; B1 is the cheap test.

**3. `sess_comp`** — form `sch·sdch ≤ ESS_COMPLEMENTARITY_TOLERANCE·S_rated²`.
Failure mode: **tiny inequality margin combined with a tiny Jacobian**; the RHS
is `1e-4·S_rated² ≈ 1.1e-12` at bootstrap capacity, i.e. seven orders below
`tol`. Reformulation must not silently retune the complementarity tolerance
(explicitly prohibited); B3 moves complementarity onto `pch·pdch` and the plan
requires auditing the new row's scale rather than rescaling it away.

**4. `sg_capability`** — form `pg² + qg² ≤ sg_available²`. Failure mode: **zero
margin** (active at the bound) with small gradients at tiny availability. This
is the RES low-output family that P5.3-A-RES and B2 target; where
`q_available = 0` and `qg` is fixed to zero it is exactly equivalent to the
linear `pg ≤ S_available`.

---

## Preconditions for the B experiments

- **B1 (`f_ref = 0`)** — Phase A confirms the reference imaginary voltage is
  currently only *approximately* fixed (`±1e-5`), and no intentional
  coordinated-interface convention depends on a nonzero `f_ref`; the interface
  coupling runs through `e_ref` pinning and the explicit interface constraints.
  **Precondition satisfied.**
- **B2 (exact RES algebra)** — requires the A-RES generator-class audit to
  identify which rows have `q_available = 0`; `sg_capability`'s zero-margin
  evidence is established, but the per-generator classification is **not yet
  done**. **Precondition not yet satisfied.**
- **B3 (active-power ESS prototype)** — the consumer trace required by B3.1 is
  **not yet done**. **Precondition not yet satisfied.**

---

## Status

Phase A's **network SMOPF conditioning audit (A1–A7) and the A-extra structural
checks are complete**. Still outstanding within P5.3: the A-RES stochastic
audit (RES1–RES5), and the B1/B2/B3 experiments and consolidated review. No
production code was changed, and no B experiment has been started.
