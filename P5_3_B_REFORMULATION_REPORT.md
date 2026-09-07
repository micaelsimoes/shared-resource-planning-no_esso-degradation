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

# B3 — Active-power shared-ESS structural prototype

Started from a fresh accepted production baseline. **B1 `f_ref = 0` not
included; no B2-R change; no P5.2 narrow band; no IPOPT/MA97 tuning.** The only
change is the shared-ESS network formulation.

## Provenance

- Scripts: `p53b3_active_power_ess.py` (A/B + rank audit),
  `p53b3_physics_tests.py` (B3.8/B3.9). Raw output:
  `data/SRP1/Results/P53B3/p53b3_report.json`, `p53b3_physics.json`.
- Branch A reproduces the accepted production baseline exactly — the same three
  persistent failures — validating the harness.
- Production source is never modified; the conversion is applied in memory to
  already-built, already-configured production models.

## B3.1 — Dependency / consumer trace (completed before any change)

| Consumer | Location | Classification |
|---|---|---|
| `sess_pch_link`, `sess_pdch_link` | mch:759, 763 | **requires `sch/sdch`** — internal, removed |
| `sess_s_limit` | mch:767 | **requires `sch/sdch`** — internal, replaced |
| `sess_snet_def` | mch:777-779 | **requires `sch/sdch`** — internal, removed |
| `sess_comp` | mch:794-795 | **requires `sch/sdch`** — moved to `pch·pdch` |
| `sess_soc_rule` | mch:815-819 | **requires `sch/sdch`** — moved to `pch/pdch` |
| `ess_utilization_cost_penalty` | mch:1605 | **requires `sch/sdch`** — objective term, mapped to `pch+pdch` |
| `ess_complementarity_penalties` | mch:1678 | **requires `sch/sdch`** — objective term, mapped to `pch·pdch` |
| `_SHARED_ESS_OPERATIONAL_VARIABLES` | mch:836-842 | zero-capacity gating list |
| result `s_ess = sch − sdch` | network.py:1359 | **export only** |
| ADMM residual diagnostic | srp:4741-4752 | **diagnostic only** |
| `sess_phi_limits_lower/upper` | mch:745-755 | **active `pch/pdch`** ✓ survives |
| `sess_pnet_def` | mch:832 | **active `pch/pdch`** ✓ retained unchanged |
| result `p_ess = pch − pdch` | network.py:1360 | **active `pch/pdch`** ✓ |
| `compute_node_load` | mch:1171-1172 | **net `pnet/qnet` only** ✓ |
| expected shared-ESS P/Q rules | mch:1064, 1075, 1751, 1755, 1794, 1798 | **net only** ✓ |
| DSO/TSO scenario-deviation penalties | srp:2800, 2826 | **net only** ✓ |
| Benders sensitivity extraction | srp:6479, 6485, 6526, 6534 | **net only** ✓ |
| ESSO `es_sch_per_unit` / `es_sdch_per_unit` | shared_energy_storage_data.py | **unrelated** — separate model |

**Decisive result of the trace:** nodal balance, ADMM consensus, expected
schedules and Benders sensitivities depend on **`pnet`/`qnet` only**. The only
cross-boundary `sch/sdch` uses are (i) a result *export* field and (ii) an ADMM
*diagnostic* report — `_get_expected_network_shared_ess_charge_discharge_mva`,
called solely from `get_admm_residual_metrics` to populate a `worst_ess_primal`
dictionary. Neither feeds a constraint, objective, consensus update or cut.

## B3.2 — Energy/time convention

- No explicit time step, period duration or `Delta_t` exists **anywhere** in
  production.
- `num_instants = 24` over one representative day, so `dt = 24 h / 24 = 1 h`,
  **derived in the harness from `len(model.periods)`, not hardcoded**.
- Power variables are p.u. on `baseMVA`; `shared_es_soc` and
  `shared_es_e_rated` are p.u. energy on the same base (results scale SOC by
  `baseMVA` to MWh at `network.py:1362`).
- The production recursion `soc_t = soc_prev + eta_ch·sch − sdch/eta_dch`
  carries **no** time factor, so it implicitly assumes `dt = 1 h`; with MVA × 1 h
  = MVAh on a common base the numeric factor is exactly 1, i.e. the production
  equation is dimensionally consistent.

**Prototype SOC (dimensional derivation stated explicitly):**
`SOC_t [p.u.·MVAh] = SOC_{t-1} + eta_ch·pch [p.u.·MVA]·dt [h] − pdch·dt/eta_dch`,
with `dt = 1 h`, so the numeric coefficient is 1 — no factor silently inserted
or omitted.

## B3.3 / B3.4 — The prototype and the derived envelope

**Derivation of the active envelope from the production feasible set:**

```
pch <= sch,   pdch <= sdch,   sch + sdch <= S_rated
=>  pch + pdch <= sch + sdch <= S_rated
```

So **`pch + pdch <= S_rated` is implied by production** and is used as the
baseline active envelope. The box bounds `0 <= pch, pdch <= S_rated` and
`|pnet|, |qnet| <= S_rated` are likewise implied (via `pch ≤ sch ≤ S_rated`, and
via `sess_snet_def` + `sess_s_limit` for `qnet`), so they are **redundant but
explicit**, not new restrictions.

Applied per live shared-ESS index (verified on every model): deactivated
`sess_snet_def` (24), `sess_pch_link` (24), `sess_pdch_link` (24),
`sess_s_limit` (24), `sess_soc_def` (24), `sess_comp` (24); fixed 48
`sch`/`sdch` variables out of the problem; added 96 new rows (SOC, capability,
active envelope, complementarity); retained `sess_pnet_def` unchanged.

**Objective handling, disclosed as part of the reformulation:** production's ESS
utilization penalty (`penalty_ess_usage·baseMVA·(sch+sdch)`) and complementarity
penalty (`PENALTY_ESS_COMPLEMENTARITY·baseMVA·sch·sdch`) are written on
`sch/sdch`. Fixing those to zero would silently delete both terms, so the
prototype adds their exact active-power analogues (`pch+pdch`, `pch·pdch`) back
and replaces the objective. Without this the comparison would be confounded.

## B3.8 — Physics unit tests (all exact, no solve required)

Device: `S_rated = 2.1270e-04 p.u.`, `eta_ch = 0.97`, `eta_dch = 0.96`,
`dt = 1.0 h`, `E_rated = 4.2540e-04 p.u.`

| Test | Result |
|---|---|
| **1 Pure reactive** — `pch = pdch = pnet = 0`, `qnet = 0.5·S_rated` | **ΔSOC = 0.000000e+00 exactly** ✓ |
| **2 Pure charging** — `pch = 0.4·S_rated` | ΔSOC = **+8.252677e-05** = `eta_ch·pch·dt` exactly ✓ |
| **3 Pure discharging** — `pdch = 0.4·S_rated` | ΔSOC = **−8.862411e-05** = `−pdch·dt/eta_dch` exactly ✓ |
| **4 Converter capability** | origin, pure-P-at-rating, pure-Q-at-rating and the on-circle point `(0.8, 0.6)·S_rated` all satisfied; the outside point `(0.9, 0.9)·S_rated` correctly rejected ✓ |
| **5 Complementarity** | row variables are exactly `{shared_es_pch, shared_es_pdch}`; RHS = `1e-4·S_rated²`, tolerance preserved exactly ✓ |
| **6 Zero capacity** | conversion skipped, no new rows, `pch`/`pnet` fixed at 0, `sess_*` rows inactive, no division by zero ✓ |

> **Test 1 is the physical correction this whole stage exists for: reactive
> power no longer changes stored battery energy.**

## B3.9 — Derivative / rank audit

| Model | Branch | Zero equality rows | Owner | Eq. rows | **σ_min(full)** | Reduced σ_min | Reduced cond |
|---|---|---|---|---|---|---|---|
| case33_1/2030/Winter | **A** | **24** | `sess_snet_def` | 4251 | **0.000e+00** | 5.9246e-03 | 8.9835e+04 |
| case33_1/2030/Winter | **B** | **0** | — | 4227 | **5.925e-03** | 5.9246e-03 | 8.9835e+04 |
| case9/2025/Winter | **A** | **72** | `sess_snet_def` | 1737 | **0.000e+00** | 3.2880e-02 | 1.4223e+03 |
| case9/2025/Winter | **B** | **0** | — | 1665 | **3.287e-02** | 3.2871e-02 | 1.4227e+03 |

Answering the plan's headline questions directly:

- **Are all former `sess_snet_def` zero equality rows gone?** **Yes** — 24 (DSO)
  and 72 (TSO) → **0**.
- **Is the full equality Jacobian now full row rank?** **Yes**, empirically:
  σ_min(full) is now strictly positive and equals the reduced σ_min, and the
  full row count drops by exactly the removed rows (4251→4227, 1737→1665).
- **New σ_min / condition:** `5.925e-03` / `8.98e+04` (DSO) and `3.287e-02` /
  `1.42e+03` (TSO) — i.e. **the full Jacobian now attains the conditioning that
  was previously only available on the reduced subspace.**

**New worst families after the reformulation** (gradient/curvature are
structural and reliable):

| Category | Family | Value |
|---|---|---|
| Smallest gradient norm | `branch_flow_limit_ji`, `b3_sess_converter_capability`, `b3_sess_comp_active` | **0.0 — but all three are INEQUALITIES at the zero-dispatch point, not equalities, so they create no rank deficiency** |
| Largest curvature | `pji_def`, `qji_def` | **1.380e+02** (was `sess_snet_def` at **18806** — a 136× reduction) |
| Worst equality family | none degenerate | no zero-gradient equality remains |

*Limitation:* the "tightest margin by family" column of that audit was computed
after the physics tests had mutated variable values, so it is contaminated and
is not quoted; the margins below come from the clean solved A/B run instead.

## B3.5 / B3.6 — Remaining numerical weaknesses, stated plainly

**Complementarity (`pch·pdch ≤ 1e-4·S_rated²`)** — tolerance preserved exactly,
not rescaled:

| `S_rated` | RHS | **RHS ÷ IPOPT `tol`** |
|---|---|---|
| 1.0635e-04 | 1.131e-12 | **1.13e-07** |
| 2.1270e-04 | 4.524e-12 | **4.52e-07** |
| 3.1905e-04 | 1.018e-11 | **1.02e-06** |

Minimum margin over the solved run: **−2.0885e-12** (i.e. *violated*) at
`dso/9/2025/Winter` period 20. The violation is ~7 orders below IPOPT's `tol`,
so the solver accepts it. **The active-power complementarity row remains
numerically weak — moving it from `sch·sdch` to `pch·pdch` did not fix that**,
and no scaling factor was introduced to hide it.

**Converter capability (`pnet² + qnet² ≤ S_rated²`)** — physically cleaner, but:

| `S_rated` | `S_rated²` | **`S_rated²` ÷ IPOPT `tol`** |
|---|---|---|
| 1.0635e-04 | 1.131e-08 | **1.13e-03** |
| 2.1270e-04 | 4.524e-08 | **4.52e-03** |
| 3.1905e-04 | 1.018e-07 | **1.02e-02** |

Minimum margin over the solved run **1.1308e-08**, i.e. `margin/tol = 1.13e-03`.

> **The converter circle is itself physically under-resolved by IPOPT at
> bootstrap capacities**: the entire constraint RHS is 100–1000× below the
> solver's absolute feasibility tolerance. Removing the degenerate equality did
> **not** automatically make the capability circle numerically well-resolved.
> This is recorded as a remaining issue for the next planner decision, exactly
> as B3.6 requires.

## B3.7 — Zero-capacity lifecycle

Preserved. For zero/near-zero capacity the conversion is skipped entirely
(`skipped_zero_capacity`, 0 new rows), the operational variables remain fixed at
zero, the `sess_*` rows remain inactive, and no division by zero occurs. The
obsolete `sess_snet_def_kappa` machinery remains untouched in production and has
no influence on the B branch. **No production code was deleted.**

## B3.10 — Isolated A/B bootstrap solves

Identical candidate, scenario realization, solver, MA97, exact Hessian, IPOPT
options, cold start and non-ESS equations.

| | **A (production)** | **B (active-power prototype)** |
|---|---|---|
| DSO succeeded | 33 / 36 | **36 / 36** |
| TSO succeeded | 12 / 12 | 12 / 12 |
| ESSO succeeded | 3 / 3 | 3 / 3 |
| Primary failures | 3 | **0** |
| Recovery attempts / successes | 0 / 0 | **0 / 0** |
| **Persistent failures** | **3** | **0** |
| Failure identities | `dso/5/2030/Winter`, `dso/5/2035/Winter`, `dso/9/2025/Summer` | **none** |
| Iterations — total | 33 073 | **1 545 (−95.3 %)** |
| Iterations — mean | 689.0 | **32.2** |
| Iterations — median | 468.0 | **27.5** |
| Iterations — max | 3000 | **109** |
| Runtime | 274 s | **37 s (−86 %)** |

> **B eliminates the failure set — it does not relocate it.** All three original
> bootstrap failures are gone and no new failure appears anywhere, in contrast to
> B1 and to every `kappa`-cap experiment in P5.1/P5.1-B. Iteration effort falls
> by a factor of 21 and the worst-case model needs 109 iterations instead of
> hitting the 3000 limit.

## B3.11 — Solution / physics comparison

33 models succeeded in both branches.

| Quantity | A | B |
|---|---|---|
| max ｜`pnet`｜ | 5.554e-05 | 2.100e-06 |
| max ｜`qnet`｜ | 5.367e-09 | **9.776e-08** |
| SOC range | [1.064e-04, 3.946e-04] | [1.061e-04, 3.220e-04] |
| Objective difference — median ｜rel｜ | — | **1.12e-03** |
| Objective difference — max ｜rel｜ | — | 1.157 (two models) |

Categorization of the differences:

- **Expected, from the SOC physics correction.** B uses ~18× more reactive power
  and far less active power. Under A, `qnet` was tied to the apparent
  charge/discharge geometry through `sess_snet_def` and thereby coupled to
  stored energy; under B, reactive power is limited only by the converter circle
  and no longer consumes battery energy. Reactive use rising and active use
  falling is precisely the intended consequence.
- **Expected, from the different feasible geometry.** The median objective
  difference is 0.11 %.
- **Explained outliers, not suspicious.** The two large relative differences
  (`dso/7/2025/Summer`, `dso/7/2035/Winter`) are cases where A converged to
  objective **+5.21 / +5.23** while every other model lands near **−0.8**; B
  reaches **−0.817 / −0.808**. On a minimisation these are A converging to a
  markedly worse point, consistent with A's much larger iteration counts — B
  improves them rather than distorting them.
- **Nothing unexplained was found.**

## B3.12 — ADMM readiness (not run, per the plan)

ADMM was **not** entered. From B3.1, the ADMM consensus variables, expected
shared-ESS P/Q schedules, ESSO aggregate coupling and Benders sensitivities all
use `pnet`/`qnet` only and would remain semantically consistent. **Two consumers
must be updated before ADMM is attempted:**

1. `network.py:1359` — the exported `s_ess = sch − sdch` would report 0. Should
   become the physically meaningful apparent power `sqrt(pnet² + qnet²)`.
2. `shared_resources_planning.py:4741-4752` — the ADMM residual *diagnostic*
   would report zero charge/discharge. Should read `pch`/`pdch`.

Neither is load-bearing, but both must be corrected so diagnostics and exports
do not silently mislead.

## B3.13 — Follow-on end-to-end plan (if approved)

1. **Network shared ESS** — promote the prototype rows into production behind
   the existing zero-capacity gating; retire `sess_snet_def`, `sess_pch_link`,
   `sess_pdch_link`, `sess_s_limit` and the `sch/sdch` variables; keep
   `sess_pnet_def`.
2. **Ordinary network ESS** — apply the identical treatment (it shares the same
   `ess_snet_def` geometry and the P4.6-B2 `kappa_es` normalization would then
   also become unnecessary).
3. **ESSO per-cohort** — convert `es_sch_per_unit`/`es_sdch_per_unit` to active
   `pch/pdch` per cohort, preserving the aggregate P/Q coupling.
4. **Aggregate P/Q coordination** — unchanged in form; verify the ESSO-side
   `es_snet` aggregation still matches the network-side `pnet`.
5. **Cell-side throughput** — redefine as
   `E_throughput = Σ_d Σ_t (D_d/365)·dt·(eta_ch·P_ch + P_dch/eta_dch)`.
6. **Cycling degradation and SoH** — drive from the corrected active throughput.
7. **Benders sensitivities** — already `pnet/qnet`-based; re-validate.
8. **Result processing/export** — fix the two consumers in B3.12.

Calendar degradation remains deferred.

## Decision criteria

| # | Criterion | Verdict |
|---|---|---|
| 1 | `sess_snet_def` zero-gradient equality family eliminated | **Yes** — 24/72 → 0 |
| 2 | Equality-Jacobian rank materially improves | **Yes** — full row rank; σ_min 0 → 5.93e-03 / 3.29e-02 |
| 3 | Three original failures eliminated without relocation | **Yes** — 36/36, zero new failures |
| 4 | Physical SOC behaviour correct | **Yes** — exact charge/discharge increments |
| 5 | Pure reactive no longer changes stored energy | **Yes** — ΔSOC = 0 exactly |
| 6 | No material implementation inconsistency | **None found**; the objective re-creation is disclosed and necessary |
| 7 | Remaining weaknesses explicitly bounded | **Yes** — complementarity RHS 1.1e-07–1.0e-06 × `tol` with a −2.09e-12 violation; capability RHS 1.1e-03–1.0e-02 × `tol` |

Per the plan, B3 is **not** rejected for the residual `sess_comp` weakness —
that is a separately identified inequality-conditioning problem, now measured
rather than hidden.

```
B3 PRODUCTIONIZE CANDIDATE — active-power ESS formulation removes the dominant structural defect
```

---

# Final ranked recommendation

| Rank | Candidate | Recommendation | Rationale |
|---|---|---|---|
| **1** | **B3 — active-power shared ESS** | **PRODUCTIONIZE CANDIDATE** | Removes the sole source of exact equality-rank deficiency, restores full row rank, eliminates all three bootstrap failures with no relocation, cuts iterations 95 %, and corrects the SOC physics |
| **2** | B1 — exact `f_ref = 0` | **CONTINUE TESTING** | Mathematically clean and free, but relocates failures while the dominant defect remains; re-test on top of B3 |
| **3** | B2-R — RES capability separation | **DEFER** | Physically justified, but the repository has no defensible converter rating; needs the `Smax` data-model extension first |

## Required answers

- **Should `f_ref = 0` be adopted?** Not yet. It removes the gauge cleanly at no
  conditioning cost, but on the current formulation it relocates failures. Re-test
  it on top of B3, where its effect can be isolated.
- **Does the active-power ESS formulation remove the exact equality-rank
  deficiency?** **Yes** — 24 (DSO) and 72 (TSO) zero rows → 0, and σ_min(full)
  becomes strictly positive.
- **Does it materially improve bootstrap NLP robustness?** **Yes** — 3 persistent
  failures → 0, 33 073 → 1 545 iterations, 274 s → 37 s, worst model 3000 → 109
  iterations.
- **Does the new active-power complementarity remain numerically problematic?**
  **Yes.** RHS is 1.1e-07–1.0e-06 × IPOPT `tol` and the solved run shows a
  −2.09e-12 violation. Moving to `pch·pdch` did not fix it; no scaling was
  applied to hide it.
- **Can `sch/sdch` be removed from the network SMOPF?** **Yes**, subject to
  updating one export field and one ADMM diagnostic. Every load-bearing
  consumer — nodal balance, consensus, expected schedules, sensitivities — uses
  `pnet/qnet` only.
- **Is the P5.2 narrow-band workaround still necessary after B3?** **No.** It
  existed to make the zero-gradient `sess_snet_def` point interior; B3 deletes
  that row entirely and achieves a strictly better result (0 failures with 1 545
  iterations, versus the narrow band's 0 failures whose declared band was below
  solver resolution). The narrow band should be dropped rather than
  productionized.
- **Is the current RES capability formulation conflating stochastic availability
  with inverter rating?** **Yes** — `sg_avail = P_available`, so the capability
  circle radius *is* the stochastic availability.
- **Is there sufficient data to reformulate the RES converter capability
  safely?** **No** — no rating field exists anywhere in the repository.
- **What should be done about the copula upper-support overshoot?** Bound the
  marginal model (preferred) or clip to the installed `Pmax`; do **not** clip to
  the historical maximum where the installed rating is known, and quantify
  generation above `Pmax` / capacity factor above 1 rather than treating
  historical exceedance as a physical violation.
- **Which nonlinear family is the highest remaining conditioning risk after the
  successful reformulations?** **`sess_comp` / `b3_sess_comp_active`** — a
  bilinear inequality whose RHS scales as `S_rated²` and sits ~7 orders below
  IPOPT's tolerance, with a measured small violation. Second is the converter
  capability circle, under-resolved by 2–3 orders at bootstrap capacities. Both
  are inequality-conditioning problems, not rank deficiencies.

```
P5.3-B COMPLETE — reformulation experiments ready for planner decision
```
