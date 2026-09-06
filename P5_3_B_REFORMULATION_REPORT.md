# P5.3-B — Isolated reformulation experiments

Diagnostic only. **No production code was changed and nothing was
productionized.** Each experiment starts from the same accepted production
checkpoint `f77d829359ff…`; no reformulation is stacked on another.

Execution order per the planner's update: **B1 → B2-R → B3**.

| Section | Status |
|---|---|
| **B1 — exact reference-angle gauge** | **Complete** |
| B2-R — RES capability semantics | Pending |
| B3 — active-power ESS prototype | Pending |

---

# B1 — Exact reference-angle gauge

## Provenance

- Script: `p53b1_reference_gauge.py`; raw output
  `data/SRP1/Results/P53B1/p53b1_report.json`.
- Scenario checksum identity unchanged; both branches use the same candidate,
  scenario realization, IPOPT options, MA97, exact Hessian and cold start.
- **A** = production `f_ref ∈ [−1e-5, +1e-5]`; **B** = `f_ref` fixed to exactly
  `0`. Nothing else differs — the hard production `sess_snet_def` equality is
  retained in both, and no P5.2 narrow-band formulation is used.
- `f_ref` is fixed by hooking `Network.run_smopf` and fixing the reference-node
  imaginary voltage immediately before the real solve is delegated to
  production.

**Branch A reproduces the accepted P5 bootstrap baseline exactly** — the same
three persistent failures at `dso/5/2030/Winter`, `dso/5/2035/Winter`,
`dso/9/2025/Summer` — which validates the harness.

## Solver outcomes (51 local solves each)

| | **A (production gauge)** | **B (`f_ref = 0`)** |
|---|---|---|
| DSO succeeded | 33 / 36 | 33 / 36 |
| TSO succeeded | 12 / 12 | 12 / 12 |
| ESSO succeeded | 3 / 3 | 3 / 3 |
| **Persistent failures** | **3** | **3** |
| Primary failures | 3 | **4** |
| Recovery attempts | **0** | **1** |
| Would enter ADMM | No | No |
| Runtime | 276 s | 248 s |
| Iterations — total | 33 073 | **28 922** |
| Iterations — mean | 689.0 | **602.5** |
| Iterations — median | **468.0** | 536.0 |
| Iterations — max | 3000 | 3000 |

## The decisive result: the failures move

The persistent-failure **sets are completely disjoint**:

| Model | **A** | **B** |
|---|---|---|
| `dso/5/2030/Winter` | **FAIL** — maxIterations, 3000 it | **OK** — Optimal, 399 it |
| `dso/5/2035/Winter` | **FAIL** — maxIterations, 3000 it | **OK** — Solved To Acceptable Level, 528 it |
| `dso/9/2025/Summer` | **FAIL** — maxIterations, 3000 it | **OK** — Solved To Acceptable Level, 337 it |
| `dso/5/2025/Spring` | OK — Solved To Acceptable Level, 303 it | **FAIL** — **Error in step computation!**, 544 it |
| `dso/5/2025/Autumn` | OK — Optimal, 409 it | **FAIL** — maxIterations, 3000 it |
| `dso/5/2035/Spring` | OK — Optimal, 742 it | **FAIL** — **Restoration Failed!**, 106 it |

> **`f_ref = 0` fixes all three original P5 bootstrap failures — and breaks three
> previously-successful solves.** The count is unchanged at 3; the identities are
> entirely different.

This is the same *relocation* signature seen with the P5.1/P5.1-B scalar `kappa`
caps, and it is expected: B1 does not touch `sess_snet_def`, which P5.3-A2
established as the sole source of exact equality-rank deficiency.

**A new failure mode also appears.** All three of A's failures are
`maxIterations`. B introduces two modes A never exhibited: **`Error in step
computation!`** and **`Restoration Failed!`**, and requires one recovery where A
required none.

## Derivative diagnostics (corrected P5.3-A2 method)

| Model | Branch | Zero equality rows | σ_min(full) | Reduced σ_min | Reduced condition | `f_ref` column |
|---|---|---|---|---|---|---|
| case33_1/2030/Winter | A | 24 | 0.00 | 5.9246e-03 | 8.983e+04 | norm **1.0** |
| case33_1/2030/Winter | B | 24 | 0.00 | 5.9238e-03 | 8.985e+04 | **absent** |
| case9/2025/Winter | A | 72 | 0.00 | 3.2880e-02 | 1.422e+03 | norm **1.0** |
| case9/2025/Winter | B | 72 | 0.00 | 3.2880e-02 | 1.422e+03 | **absent** |

- **Gauge freedom is removed cleanly.** In B the `f_ref` variable is eliminated
  from the NLP entirely (no column), whereas in A it carries a healthy column
  norm of 1.0.
- **Rank deficiency is unchanged**: 24/72 exactly-zero rows and σ_min(full) = 0
  in both branches, because those rows are all `sess_snet_def`. B1 cannot and
  does not address them.
- **Conditioning is essentially unchanged** — DSO reduced condition rises
  marginally (8.983e+04 → 8.985e+04, +0.02 %); TSO is identical to four
  significant figures.

## Physical / economic comparison — and why it is contaminated

Of 42 comparable objectives, **25 agree to better than 1e-6 relative** (largest
agreeing delta 8.0e-07 relative). The other 17 differ, some very large:
`tso/2030/Winter` +25 970 (5.89× relative), `tso/2025/Autumn` −54 721,
interface `pf_p` up to 66.7 MW and `pf_q` up to 23.4 MVAr, while interface
`vmag` differs by at most 2.4e-04.

**These differences are not attributable to the gauge change.** All 12 TSO
models succeed in both branches, but the TSO cold start is built from the DSO
consensus values, and the DSO *failure set differs between branches*. A failed
DSO solve contributes no interface values, so the TSO models are literally
different problems in A and B. Since initialization completes in neither branch
(`enter_admm = False` both), the downstream economic comparison is contaminated
and **cannot be used to certify "no physical change beyond numerical
tolerance."**

The clean statement available is the one for the 25 models that succeeded in
both branches: they agree to ≤ 8.0e-07 relative, consistent with an exact gauge
restriction of a rotationally-degenerate coordinate.

## Acceptance assessment

| Criterion | Verdict |
|---|---|
| No physical/economic change beyond numerical tolerance | **Not demonstrable** — 25/42 agree to <1e-6, but the remainder is contaminated by the differing failure sets |
| No new failure family appears | **FAILS** — three previously-successful node-5 solves now fail, introducing `Error in step computation!` and `Restoration Failed!`, modes absent from A |
| Gauge freedom removed cleanly | **PASSES** — `f_ref` eliminated from the NLP |
| Initialization robustness / conditioning not worse | **Marginal** — same persistent count (3) but 4 primary failures vs 3 and 1 recovery vs 0; conditioning unchanged (DSO +0.02 %) |

## B1 verdict

**CONTINUE TESTING — do not productionize.**

The gauge fix does exactly what it is supposed to do mathematically (it removes
a genuine ±1e-5 rotational freedom, cleanly, at no conditioning cost) and it
*does* repair all three original P5 failures. But on its own it is not a net
robustness improvement: it relocates the failures onto three different node-5
models and introduces two harsher failure modes. Total iterations fall 12.5 %
while the median rises, so even the effort metric is mixed.

The reason is now well established: B1 leaves the exact zero-gradient
`sess_snet_def` equality untouched, and that is the sole source of exact rank
deficiency. **B1 should be re-evaluated *after* B3**, on top of a formulation
that no longer contains the degenerate rows — where its effect can be isolated
from the dominant defect rather than competing with it.

---

# B2-R — RES capability semantics and conditioning

Started from a fresh accepted production baseline. **B1 was not included**, no
P5.2 narrow-band formulation was used, and no B3 change was present.

## Provenance

- Script: `p53b2r_res_capability_audit.py`; raw output
  `data/SRP1/Results/P53B2R/p53b2r_report.json`.
- Pure analysis of the production data and formulation — **no solve, no model
  change**. The mathematical A/B is gated on B2-R.2, which did not open.

## B2-R.1 — Rating semantics

36 distinct curtailable RES generator-year instances in reduced SRP1
(`case33_1/2/3`, types PV and Wind). Representative raw JSON entry:

```json
{"gen_id": 2, "bus": 6, "Pmax": 40.0, "Pmin": 0.0, "Qmax": 40.0, "Qmin": -40.0,
 "Vg": 1.0, "status": 1, "type": "WIND", "pf_control": 1,
 "pf_max": 0.9, "pf_min": -0.9}
```

| Property | Result |
|---|---|
| `power_factor_control` | **True for all 36** |
| `Qmax == Pmax` | **True for all 36** (and `Qmin == −Pmax`) |
| Explicit apparent-power / MVA / inverter / converter / nameplate field | **None** |
| `Generator` class rating attribute | **None** |

**Traced through the whole chain.** The `Generator` class (`generator.py`)
defines only `gen_id, bus, pmax, pmin, qmax, qmin, vg, status, gen_type,
power_factor_control, max_pf, min_pf`. Across **all 6223 generator entries in
every network JSON in the repository**, the complete field set is exactly
`gen_id, bus, Pmax, Pmin, Qmax, Qmin, Vg, status, type, pf_control, pf_max,
pf_min`. Nothing else exists to read.

Distinguishing the three quantities the plan asks about:

1. **Installed/nameplate capability** — only `Pmax` (active). No apparent-power
   nameplate.
2. **Stochastic resource availability** — `generator.pg[s_o][p]`, produced by
   the copula pipeline as `sample × pmax`; `generator.qg[s_o][p]` is
   **identically zero by construction**.
3. **Optimization upper bounds** — `pg ∈ [0, P_available + tol]` (curtailable),
   `qg ∈ [qmin − tol, qmax + tol]`.

## B2-R.2 — Is there a defensible `S_converter`? **Category C — No**

| Rejected derivation | Why |
|---|---|
| `S = sqrt(Pmax² + Qmax²)` | Would yield `√2·Pmax` **only because `Qmax` was set equal to `Pmax`**. No documented semantics justify it, and 1.414× is not a standard inverter sizing. Explicitly prohibited by the plan. |
| Oversizing factor | Arbitrary; explicitly prohibited. |
| Historical maximum | A data maximum is not an equipment rating; explicitly prohibited. |

`Qmax = Pmax` is a permissive modelling convention — it is what you write when
you do not want the static reactive bound to bind — not a converter rating.

> **The repository does not contain enough information to distinguish converter
> apparent-power rating from stochastic active-power availability.**
> **Per the plan, the mathematical B2-R A/B stops here. No rating was invented.**

## B2-R.3 — Current production feasible set

For a live curtailable RES generator the production constraints are:

```
0 <= pg <= P_available                        (pg_bounds)
pg^2 + qg^2 <= sg_avail^2                     (sg_capability)
tangent_lower*pg <= qg <= tangent_upper*pg    (gen_pf_lower / gen_pf_upper)
qmin <= qg <= qmax                            (qg_bounds)
```

with `sg_avail = sqrt(P_available² + Q_available²)` and `Q_available ≡ 0`, hence
**`sg_avail = P_available`**: the capability circle radius *is* the stochastic
active availability. With `pf_max = 0.9`, `tangent_upper = 0.4843`.

Resulting reactive capability, as a multiple of `P_available`:

| Operating point | Circle allows | PF cone allows | **Effective ｜q｜max** | Binding |
|---|---|---|---|---|
| `pg = 0` | 1.0000 | 0.0000 | **0.0000** | PF cone |
| `pg = 0.5·P_av` | 0.8660 | 0.2422 | **0.2422** | PF cone |
| `pg = 0.90·P_av` | 0.4361 | 0.4358 | **0.4358** | crossover |
| `pg = P_available` | 0.0000 | 0.4843 | **0.0000** | **`sg_capability`** |
| `P_available = 0` | — | — | **0** (all RES rows skipped) | structural switch |

The PF cone binds for `pg ≤ 0.900·P_av`; above that the circle binds. Peak
reactive capability is **0.4359·P_available at pg = 0.900·P_available**, and it
is **exactly zero at both `pg = 0` and `pg = P_available`**.

> **Yes — stochastic irradiance/wind availability is directly controlling
> inverter reactive-power capability.** The capability radius shrinks with
> availability and collapses to zero when availability is zero, so a physical
> inverter that could still supply reactive power at night or under full active
> loading has no reactive capability in this model at all.

## B2-R.4 — Validation against the static network data

| Metric (2400 live cold-start points) | Value |
|---|---|
| Binding restriction at the cold start | **`sg_capability` in 2400 / 2400 (100 %)** |
| Structurally switched off (`P_av ≤ 1e-5`) | 1056 |
| Capability radius ÷ static `qmax` — minimum | **1.21e-04** |
| … median | **0.326** |
| … maximum | 1.02 |

**Contradictions identified:**

- The static bound permits `|qg| ≤ qmax = Pmax`, but the dynamic circle permits
  only `|qg| ≤ P_available`. At the median point the circle is **~3× more
  restrictive** than the static bound, and in the worst case **~8300× more
  restrictive**.
- `qmin/qmax` are therefore **effectively never the binding reactive limit** for
  curtailable RES — they are unreachable in practice.
- At the cold start the circle is the binding restriction in **every** live case.

## B2-R.5 — Initialization interiority

`pg_init = max(0, P_available)` (i.e. `= P_available` for live generators) and
`qg_init = 0` (documented as a "neutral starting point"). Therefore

```
pg_init^2 + qg_init^2 = P_available^2 = sg_avail^2
```

> **Every live RES generator starts exactly ON the nonlinear capability
> boundary — cold-start margin identically 0.0.** This fully explains the
> P5.3-A finding that `sg_capability` has an exactly-zero minimum margin across
> 3732 rows; it is a direct consequence of initializing at full available
> active power with a circle whose radius is that same value.

Under a separated converter rating the same initial point would sit strictly
inside the circle with normalized interior distance
`1 − (P_available / S_converter)²`. **This cannot be quantified because
`S_converter` does not exist in the repository.** (The initialization itself was
not changed, per the plan.)

## B2-R.6 — Low-output points

All **17** realized availability values in `(1e-5, 1e-4]` were examined. Every
one has `Q_available = 0`, so `S_available = P_available ∈ [3.79e-05, 1e-04]`,
a capability-radius-to-`qmax` ratio of order `1e-04`, and — at the initial point
`pg = P_available` — an effective reactive capability of **exactly 0**. Examples:

| Network / year / day | Gen | Hour | `P_available` | radius ÷ `qmax` | ｜q｜max at init |
|---|---|---|---|---|---|
| case33_1 / 2030 / Winter | 5 | 7 | 3.787e-05 | 2.53e-04 | 0 |
| case33_2 / 2025 / Winter | 4 | 7 | 3.844e-05 | 3.84e-04 | 0 |
| case33_1 / 2035 / Summer | 5 | 5 | 4.233e-05 | 1.21e-04 | 0 |
| case33_3 / 2030 / Winter | 5 | 7 | 4.742e-05 | 1.58e-04 | 0 |
| case33_2 / 2025 / Winter | 5 | 7 | 5.171e-05 | 5.17e-04 | 0 |

These are tiny active-capability circles that are *active at the initial point*,
with gradient norms measured in P5.3-A down to `5.44e-05`.

**B2-R.7 / B2-R.8 / B2-R.9 were not run**, because no physical B formulation was
admissible without inventing a rating.

## Proposed data-model extension (required, not implemented)

| Item | Proposal |
|---|---|
| **New field** | `Smax` in the network JSON `generators` entries |
| **Units** | MVA, consistent with `Pmax`/`Qmax` (converted by `/ baseMVA` at parse time, as `Pmax` already is) |
| **Physical meaning** | Continuous apparent-power rating of the grid-side converter/inverter — an **equipment** limit, independent of instantaneous resource availability |
| **Where read** | `_read_network_from_json_file` in `network.py`, alongside the existing `Pmax`/`Qmax` parsing |
| **Storage** | New attribute `Generator.s_rated` (p.u.), default `None` |
| **Legacy handling** | If absent, `s_rated = None` and the model **keeps today's behaviour exactly** (`pg² + qg² ≤ sg_avail²`), so every existing case study is bit-for-bit unchanged. The separated formulation activates only where `Smax` is supplied. |
| **Validation rules** | `Smax > 0`; `Smax ≥ Pmax` (a converter must pass its own active nameplate); warn if `Smax > 2·Pmax` as implausible; require `Smax` for any generator with `pf_control = 1` in new case studies |
| **Formulation once available** | `0 ≤ pg ≤ P_available` (resource) **and** `pg² + qg² ≤ S_converter²` (equipment), retaining the PF cone unchanged. This is a deliberate feasible-set change and must be validated as such. |

Expected numerical benefit, stated as a hypothesis rather than a result: the
capability circle would become a **fixed** equipment circle instead of a
shrinking availability circle, the cold start would move strictly inside it, and
the 17 low-output rows would stop being tiny active circles. None of this is
demonstrated here.

## Stochastic-support recommendations (required regardless of branch)

**Upper support — the substantive issue.** P5.3-A2 measured up to **33.5 %** of
samples above the historical maximum in some (season, RES-type) combinations.
Recommendation, in priority order:

1. **Clip to the installed active nameplate `Pmax`**, not to the historical
   maximum. `Pmax` is a physical equipment limit and is already present for
   every generator; the historical maximum is merely the largest value observed
   in a finite record and may legitimately be exceeded.
2. Better still, **bound the marginal model itself** — the overshoot originates
   in the unbounded Gaussian-KDE marginals, so fitting a support-bounded
   marginal (e.g. beta/truncated KDE on `[0, Pmax]`) fixes the cause rather than
   the symptom, and preserves the copula's dependence structure.
3. Do **not** clip to the historical maximum where the installed rating is
   known — that would discard physically attainable operating points.

**Lower support.** Replacing `abs(sample)` with `max(sample, 0)` is
**recommended on physical grounds** even though P5.3-A2 showed the quantitative
effect is negligible (≤ 0.01 % of mass, 0–4 samples per 2400). Reflection
manufactures generation from a negative excursion, which has no physical
meaning; clipping represents "no output", which does. It is a correctness fix,
not a performance fix, and should be labelled as such.

**Spatial correlation.** Currently the copula is fitted per
`(season, RES type)` on the 24 hourly columns with **all same-type generators
pooled**, and each generator then draws **independently** with a per-`gen_id`
seed — so no site-to-site correlation survives. A future scenario model should
either (a) fit a joint copula over `(site × hour)` so PV sites and wind sites
retain their spatial dependence, or (b) keep the per-type temporal copula but
add a site-correlation layer (e.g. a spatial Gaussian copula on the marginal
ranks, parameterized by inter-site distance). Load–RES cross-correlation could
be included on the same principle if the historical records are time-aligned.
This requires a per-site identifier in the operational-data workbook, which the
current format lacks. **Future stochastic-model work — explicitly separate from
SMOPF conditioning.**

## B2-R verdict

```
B2-R DEFER — insufficient physical rating data for safe reformulation
```

The audit *did* establish the physical concern conclusively: the model conflates
stochastic availability with converter capability, reactive capability collapses
to zero at both `pg = 0` and `pg = P_available`, `sg_capability` is the binding
reactive restriction in 100 % of live cold-start points, and every live RES
generator starts exactly on that boundary. But the repository contains no
defensible converter rating, and the plan rightly forbids inventing one — so the
reformulation cannot be tested safely until the data model is extended.

# B3 — Active-power ESS structural prototype

*Pending planner review of B2-R.*
