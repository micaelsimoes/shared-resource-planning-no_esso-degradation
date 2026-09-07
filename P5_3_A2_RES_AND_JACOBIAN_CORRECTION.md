# P5.3-A2 — Jacobian diagnostic correction, RES stochastic audit, DSO interface trace

Diagnostic only. **No production code was changed. B1/B2/B3 were not started.**

## Provenance

- Scripts: `p53a2_jacobian_correction.py` (Parts A/B), `p53a2_res_audit.py` (Part C).
  Raw output: `data/SRP1/Results/P53A2/p53a2_jacobian.json`, `p53a2_res.json`.
- Production checkpoint `f77d829359ff…`; scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.
- Parts D/E are source/log traces of the live production code.

**Both issues the planner raised were real defects in my P5.3-A report.** One of
them (Part A) reverses a headline conclusion.

---

## Part A — Corrected equality-Jacobian singular-value interpretation

### What P5.3-A got wrong

P5.3-A reported exactly-zero equality rows *and* nonzero `sigma_min`. That is
self-contradictory, and the planner is right. The cause: `sigma_min` came from
`scipy.sparse.linalg.svds(..., which='SM')`, an iterative method that **failed
to converge to the true smallest singular values** and returned interior values
instead. Those numbers were wrong.

**Corrected method:** exact singular values from the eigenvalues of `J Jᵀ`
(dense symmetric eigensolve), computed on the *repaired* models so no rows are
missing (see Part B).

### Full equality Jacobian

| Model | Rows | Cols | Exactly-zero rows | Zero-row owner | **σ_min** | σ_max |
|---|---|---|---|---|---|---|
| case33_1/2025/Winter | 4251 | 7684 | **24** | `sess_snet_def` (24) | **0.000e+00** | 532.2 |
| case33_2/2030/Summer | 4251 | 7684 | **24** | `sess_snet_def` (24) | **0.000e+00** | 532.2 |
| case33_3/2025/Summer | 4251 | 7684 | **24** | `sess_snet_def` (24) | **0.000e+00** | 532.2 |
| case9/2025/Winter | 1737 | 2748 | **72** | `sess_snet_def` (72) | **0.000e+00** | 46.77 |
| case9/2030/Spring | 1737 | 2748 | **72** | `sess_snet_def` (72) | **0.000e+00** | 46.77 |

- **σ_min of the full equality Jacobian is exactly zero** in every model; the
  five smallest singular values are all `0.00e+00`.
- **Structurally guaranteed minimum nullity from zero rows: 24 (DSO), 72 (TSO).**
- **The full condition number is formally infinite.** The full Jacobian is
  **exactly** row-rank deficient — not "near rank deficient", which is how
  P5.3-A wrongly described it.
- Every exactly-zero row is a `sess_snet_def` row, confirming that P5.3-A
  finding.

*Reported limitation:* the `J Jᵀ` route squares the condition number, so the
`numerical_rank` figure printed for the **full** matrix (4240 / 1701) is not
trustworthy near zero — it exceeds the structurally implied bound (≤ 4227 /
≤ 1665) because roundoff lifts exact zeros above the relative tolerance. The
**structural** zero-row count and the **reduced** spectrum below are the
reliable statements.

### Reduced equality Jacobian (`J_eq_reduced`, exactly-zero rows removed)

| Model | Rows | Removed | **σ_min (nonzero subspace)** | σ_max | **Condition** | Rank | Extra nullity |
|---|---|---|---|---|---|---|---|
| case33_1/2025/Winter | 4227 | 24 | 5.9246e-03 | 532.2 | **8.98e+04** | 4227 | **0** |
| case33_2/2030/Summer | 4227 | 24 | 5.9246e-03 | 532.2 | **8.98e+04** | 4227 | **0** |
| case33_3/2025/Summer | 4227 | 24 | 5.9246e-03 | 532.2 | **8.98e+04** | 4227 | **0** |
| case9/2025/Winter | 1665 | 72 | 3.2880e-02 | 46.77 | **1.42e+03** | 1665 | **0** |
| case9/2030/Spring | 1665 | 72 | 3.2880e-02 | 46.77 | **1.42e+03** | 1665 | **0** |

Once the exactly-zero rows are removed, **every model has full row rank** — the
rank deficiency is entirely and only the `sess_snet_def` zero rows. There is no
*additional* hidden dependence.

### The headline reversal

> **P5.3-A claimed the TSO equality Jacobian is ~200× worse conditioned than the
> DSO's. That is refuted.** On the corrected computation the **TSO is ~63×
> BETTER conditioned than the DSO** (1.42e+03 vs 8.98e+04), and the TSO's
> smallest nonzero singular value (3.29e-02) is *larger* than the DSO's
> (5.92e-03).

The earlier "TSO σ_min ≈ 1.9e-05 ≈ IPOPT tol" statement was an artifact of the
non-converged iterative solver and must be withdrawn. Consequently the P5.3-A
suggestion that TSO conditioning explains P5.2-A3's TSO-concentrated band
utilization **is not supported**; that observation now has no established
Jacobian-conditioning explanation and remains open.

### Near-collinearity claim, corrected in interpretation

P5.3-A's "no near-collinear pairs" scan compared only rows sharing an identical
variable support. **Pairwise non-collinearity does not prove the absence of
multi-row linear dependence**, and the scan could not have detected dependence
across differing supports. The correct statement is the rank result above: after
removing zero rows the reduced Jacobian has **full row rank**, so for these five
models there is no multi-row dependence — established by rank, not by the
pairwise scan.

---

## Part B — Corrected column-norm diagnostics

The planner's challenge was correct: `pij_def` contributes exactly `+1` w.r.t.
`pij`, so a `pij[31]` column norm of 0 was impossible.

**Direct differentiation of the production rows** (repaired models):

| Row | ∂row/∂(own variable) | = 1 ? |
|---|---|---|
| `pij_def` | **1.0** | ✓ |
| `qij_def` | **1.0** | ✓ |
| `pji_def` | **1.0** | ✓ |
| `qji_def` | **1.0** | ✓ |

(These components do not exist in the TSO models — `case9` has no
apparent-power-limited branches — correctly reported as absent rather than zero.)

### Root cause: **a diagnostic assembly bug**, not a model issue

`model.r_sqr` is declared at `network.py:332` **without `initialize=`**, so
every one of its VarData objects has value `None` at the cold start
(**768 per DSO model, 216 per TSO model**). Numeric reverse-AD therefore raised
on every row referencing `r_sqr`, and the P5.3-A audit swallowed that exception
(`except Exception: grads = []`) and **silently skipped the row**.

Per DSO model, **120 rows were dropped**:

| Component | Rows skipped |
|---|---|
| `r_sqr_def` | 24 |
| `pij_def` | 24 |
| `qij_def` | 24 |
| `node_balance_p` | 24 |
| `node_balance_q` | 24 |

`pij[31]` / `qij[31]` appear *only* in those dropped rows, so they accumulated
no column mass and appeared as zero columns.

**Repair confirms it.** Giving the uninitialized variables a nominal value:

| | Without repair | With repair |
|---|---|---|
| Derivative failures (DSO) | **120** | **0** |
| Columns with norm < 1e-10 (DSO) | **48** | **0** |
| `pij[31,0,0,0]` column norm | **0.0** | **1.0** |
| `qij[31,0,0,0]` column norm | **0.0** | **1.0** |

TSO models had **0** failures and **0** small columns both ways (their
`r_sqr` rows are not referenced by any active row here).

**Consequential corrections to P5.3-A:**

1. The "~48 DSO columns with norm < 1e-10" finding is **withdrawn** — artifact.
2. The claim that the TSO's smallest columns are `f[0,…]` is **withdrawn as a
   red flag**: `f[0,…]` has column norm **1.0**, which is healthy. It was merely
   the smallest value in a well-scaled set, and I presented that as suspicious.
3. P5.3-A's DSO equality-row count (4131) was short by exactly the 120 dropped
   rows; the correct count is **4251**.
4. The row-wise statistics in P5.3-A omitted those 120 rows per DSO model.

The only genuine production observation that survives here is minor:
**`r_sqr` enters the NLP with no Pyomo starting value.** IPOPT supplies its own
default, so this is a cold-start initialization gap, not a defect.

---

## Part C — RES stochastic and low-output audit

### C1. Raw copula support and the `abs()` question

The production pipeline was instrumented non-invasively (recording proxies
around `network_data.np` and `network_data.MinMaxScaler`); the algorithm and
seed 2026 are untouched. 32 RES sampling calls were recorded, 2400 values each.

| Quantity | Result across the 32 calls |
|---|---|
| Negative samples before `abs` | **0–4 per 2400 (0 %–0.17 %)** |
| Most negative value seen | −3.58e-02 |
| **Mass created purely by reflection** | **≤ 0.0001 (≤ 0.01 %) of total post-`abs` mass** |
| Values **above** the historical maximum | **0 %–33.5 %** |

> **The `abs()` reflection is quantitatively negligible.** At most 4 values in
> 2400 are negative, and the positive generation created solely by reflecting
> them is ≤ 0.01 % of the total. The planner's hypothesis is measurable and
> turns out **not** to be the significant defect.
>
> **The material support problem is upward overshoot**: in several
> (season, RES-type) combinations **a third of all sampled values exceed the
> historical maximum** (33.5 %, 33.5 %, 33.5 %, and 9.8–11.0 % in others). The
> Gaussian-copula + KDE fit is extrapolating well beyond observed support.

### C2. Realized low-output population (3456 values: 144 generator-instances × 24 h)

| Bin | Count | Share |
|---|---|---|
| **exactly zero** | **1056** | **30.6 %** |
| (0, 1e-6] | **0** | 0 % |
| (1e-6, 1e-5] | **0** | 0 % |
| (1e-5, 1e-4] | **17** | 0.5 % |
| (1e-4, 1e-3] | 65 | 1.9 % |
| > 1e-3 | 2318 | 67.1 % |

`renewable_generation_is_unavailable` fires **1056 times**, and **every one of
those is an exact zero** — there are **no values in (0, 1e-5]** at all.

> **The `EQUALITY_TOLERANCE = 1e-5` structural switch is not currently being
> exercised marginally.** It separates exact zeros from values ≥ 1e-5; no sample
> sits just below it. The nearest live cases are the **17 values in
> (1e-5, 1e-4]**, which sit just *above* the threshold and therefore *do*
> instantiate the nonlinear RES rows at very small availability.

### C3. RES constraint-instantiation map — and a B2 precondition failure

144 curtailable generator instances across the SRP1 population:

| Property | Count |
|---|---|
| `power_factor_control = True` | **144 / 144 (all)** |
| `qg_available` identically zero | **144 / 144 (all)** |
| Generator types present | 2 (`GEN_RES_SOLAR`), 3 (`GEN_RES_WIND`) |

`qg` is zero by construction: `generate_res_generation_profiles` builds it as
`pd.DataFrame(np.zeros(samples.shape))` (`network_data.py:349`).

**`power_factor_profile_rule` is skipped whenever `power_factor_control` is
True** (`model_construction_helpers.py:540-547`). Since all 144 instances have
PF control:

> **`gen_pf_profile` is never instantiated anywhere in the current SRP1
> population.** There is therefore **no row of the form
> `q_available·pg == p_available·qg`** to simplify, and consequently **no row
> that degenerates to `p_available·qg == 0`.**
>
> **P5.3-B2's primary premise does not hold for SRP1.** The exact-cleanup
> experiment as specified would be a no-op on this case study. (It may still
> apply to other case studies — e.g. HR1/OP1 — if any curtailable generator
> there has `power_factor_control = False`; that was not audited here.)

What *is* instantiated instead: `gen_pf_upper` / `gen_pf_lower` (the PF cone
`tan_lo·pg ≤ qg ≤ tan_up·pg`) and `sg_capability`. Because `qg` is free within
the cone rather than fixed at zero, the second B2 simplification
(`pg² + qg² ≤ S²` → `pg ≤ S`) **also does not apply** — it would change the
feasible set.

### C4. `sg_capability` conditioning

From the P5.3-A row audit (unaffected by the Part B bug — `sg_capability` was
never among the skipped components): **3732 active rows, minimum margin exactly
`0.00`, gradient norms from `5.44e-05` to `9.49e-01`, curvature 2.**

With `qg_available ≡ 0`, `sg_available = sqrt(pg_av² + qg_av²) = pg_av`, so the
capability radius **collapses onto the available active power**. The rows with
the smallest gradients are exactly those with the smallest `sg_available`, i.e.
the 17 values in (1e-5, 1e-4].

**Correlation only, no causal claim:** these rows coincide with small
availability, and the family sits at zero margin. Nothing in this audit
establishes that they cause any particular solve failure.

**RES4 answer (availability vs converter rating):** the model currently
**conflates** them. `sg_available` is derived purely from stochastic
availability, so the reactive capability shrinks to zero exactly when active
availability does — whereas a physical inverter retains its MVA rating. The data
contains `pmax`/`qmin`/`qmax` per generator, which could support the cleaner
separation (`0 ≤ pg ≤ P_available`, `pg² + qg² ≤ S_converter²`), but that is a
**feasible-set change** and is not proposed for implementation here.

### C5. Spatial / scenario correlation

Traced through `generate_res_generation_profiles` (`network_data.py:306-352`)
and `_update_network_with_operational_data` (`network.py:1047-1072`):

- The copula is fitted per **(season, RES type)** on `gen_hours`, whose columns
  are the **24 hours**. **It jointly models temporal (intra-day) dependence
  only.**
- **All same-type generators are pooled** into one training set; generator
  identity is discarded at fit time.
- Each physical generator then draws **independently** from that shared pool:
  `...['pg'].sample(n=num_oper_scenarios, random_state=derive_random_seed(seed, 'generator', str(gen_id), 'pg'))`,
  i.e. a **per-generator seed**.

> **No cross-generator spatial correlation is preserved.** Different PV
> generators receive independent draws, as do different wind generators. Any
> historical spatial coherence between sites is lost at the pooling step and is
> not reintroduced at assignment.

*Limitation:* the historical operational-data workbook pools rows by
`(Season, GenType)` without a per-site identifier that this audit could use to
reconstruct site-to-site historical correlations, so a quantitative
historical-vs-synthetic correlation comparison could not be produced. Stated
explicitly rather than estimated.

---

## Part D — DSO reference / interface-voltage semantic trace

This **corrects** a P5.3-A statement.

1. **Are the DSO reference bounds changed after `build_model`?** **Yes.**
   `update_distribution_models_to_admm` (`shared_resources_planning.py:3693-3704`)
   explicitly rewrites them.
2. **Does ADMM setup release the DSO reference voltage?** **Yes, deliberately.**
   The code carries the comment *"Free the interface magnitude while retaining
   the reference angle"* and performs:
   `e[ref].fixed = False`, `e[ref].setub(voltage_upper)`, `e[ref].setlb(0.00)`,
   `f[ref].setub(+EQUALITY_TOLERANCE)`, `f[ref].setlb(−EQUALITY_TOLERANCE)`,
   plus fixing the reference-node voltage slacks to 0.
3. **Coupling to consensus:** `interface_vmag_distribution_def` returns
   `vmag[ref_node_idx]`; the DSO objective gains `rho_v`, `vmag_req` and
   `dual_vmag_req` ADMM terms in the same function.
4. **Attainable range:** during ADMM the DSO interface magnitude may move over
   the full `[0, voltage_upper]` band. **At cold-start initialization — the
   population P5.3 audits — it is still pinned at `vg ± 1e-4`** by `e_bounds`.
5. **Intentional?** **Yes, and documented in code.** The angle reference is
   deliberately retained; the magnitude is deliberately freed.
6/7. Since the magnitude *is* released in ADMM, the premise that the TSO must
   absorb all voltage-consensus movement does not hold.

**Classification: intentional boundary condition.** My P5.3-A framing — that the
DSO interface voltage is "effectively pinned regardless of `enforce_vg`", implying
a possible coordination inconsistency — was correct **only for the
initialization models** and is withdrawn as a general statement.

This *strengthens* B1's precondition: fixing `f_ref = 0` exactly would make the
already-intended "retain the reference angle" behaviour exact, rather than
approximate to ±1e-5.

---

## Part E — Objective-scaling interpretation

SRP1's network params do **not** set `nlp_scaling_method`, so IPOPT's default
`gradient-based` scaling applies (`nlp_scaling_max_gradient = 100`). From the
production log `optim_log_case9_2035_Winter.log`:

```
Scaling parameter for objective function = 8.662691e-06
objective scaling factor = 8.66269e-06
No x scaling provided
No c scaling provided
d scaling provided
HSL_MA97: Enabling scaling 0 due to excess delays
```

> **IPOPT *is* already scaling the large objective gradient internally**
> (factor `8.66e-06`, which maps the observed `~1.2e+07` maximum component onto
> ~100). P5.3-A's raw `1e6–1e8` objective-gradient observation therefore does
> **not** reach the KKT system unscaled, and should not be read as a live
> objective/constraint imbalance.
>
> **However, `No c scaling provided`: the constraints are not scaled.** The raw
> constraint-row disparity — exact zero rows, `~1e-8` rows, `~1e+05` rows —
> reaches the KKT system essentially unchanged.

The log line `HSL_MA97: Enabling scaling 0 due to excess delays` is independent
evidence that the factorization is encountering pivoting difficulty.

---

## Revised risk ranking

| Rank | Family | Failure mode | Status vs P5.3-A |
|---|---|---|---|
| **HIGH** | **`sess_snet_def`** | Exact zero gradient on an always-active equality (**the sole source of exact rank deficiency: σ_min(full) = 0, nullity 24 DSO / 72 TSO**) plus `O(1/S_rated)` curvature (18806) | **Confirmed and strengthened** |
| **HIGH** | **`sess_comp`** | Tiny Jacobian (~2e-08) on a numerically active bilinear inequality; RHS scales as `S_rated²` | Unchanged |
| **HIGH** | **`sg_capability`** | Zero margin on 3732 rows; smallest gradients coincide with the 17 availability values in (1e-5,1e-4]; conflates stochastic availability with converter MVA rating | Unchanged, plus RES4 finding |
| **MEDIUM** | **RES scenario support** | Up to **33.5 %** of samples exceed the historical maximum; `abs()` reflection is negligible (≤0.01 %) | **New; supersedes the `abs()` hypothesis** |
| **MEDIUM** | Unscaled constraints | IPOPT scales the objective but **not** the constraints; MA97 reports excess delays | **New (Part E)** |
| **MEDIUM** | Reference gauge `f_ref` | Residual ±1e-5 gauge freedom; intentional in code, but approximate rather than exact | Reclassified as intentional |
| **LOW** | DSO reference `e` pinning | Applies only to initialization; ADMM deliberately frees it | **Downgraded** |
| **LOW** | `branch_flow_limit`, `branch_flow_limit_ji` | Zero gradient at zero flow but strictly inactive | Unchanged |
| **LOW** | `r`/`r_sqr` | NL writer eliminates unused entries; `r_sqr` merely lacks a start value | Unchanged |
| **WITHDRAWN** | ~~TSO Jacobian ill-conditioning~~ | Refuted: TSO cond 1.42e+03 vs DSO 8.98e+04 | **Withdrawn** |
| **WITHDRAWN** | ~~48 near-zero DSO columns / `f[0]` columns~~ | Diagnostic artifact from uninitialized `r_sqr` | **Withdrawn** |

### Effect on B-experiment preconditions

- **B1 (`f_ref = 0`)** — precondition **satisfied and strengthened** (Part D
  shows retaining the angle reference is explicitly intended).
- **B2 (exact RES algebra)** — precondition **fails for SRP1**: `gen_pf_profile`
  is never instantiated because all 144 RES instances have `power_factor_control
  = True`, and `qg` is not fixed to zero, so neither B2 simplification applies
  without changing the feasible set. B2 should be re-scoped or re-targeted by
  the planner.
- **B3 (active-power ESS prototype)** — the B3.1 consumer trace remains
  outstanding; unchanged by this stage.

---

```
P5.3-A2 COMPLETE — corrected structural and RES audit ready for planner review
```
