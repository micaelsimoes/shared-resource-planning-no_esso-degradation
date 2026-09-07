# P5.4 — End-to-end active-energy ESS productionization

**Status: checkpoint after A, B, C, D, D2, D2-P, D3, E, E2, H1 and F. G remains
blocked, so no final P5.4 verdict is issued in this revision.**

| Section | Status | Commit |
|---|---|---|
| **A — productionize shared network ESS** | **Complete** | `a4a0bae8` |
| **B — ordinary network ESS parity** | **Complete** | `1e86d40e` |
| **C — ESSO active-energy conversion** | **Complete** | `58f4911b` |
| **D — sensitivity / lifecycle / warm-start audit** | **Complete, with one negative result** | `c3526ec8` |
| **E — production validation before ADMM** | **Complete, revalidated after B/C and again after H1** | `c3526ec8`, `93974d83` |
| **E2 — complementarity significance** | **Complete** | `b0e53bc4` |
| **H1 — dimensionless complementarity** | **Complete — gate PASSED** | `93974d83` |
| **F — live distributed ADMM (net P/Q only)** | **Complete — converged** | `2917b9c9` |
| **D2 — S/E sensitivity root-cause audit** | **Complete — PARTIAL** | `65b261ba` |
| **D2-P — sensitivity-clean productionization** | **Complete — PASS** | `06e921e5` |
| **D3 — distributed cut-consistency audit** | **Complete — FAIL** | `e1afa8e9` |
| G — reduced planning gate | **Still blocked** — D3 found the current cuts demonstrably unsafe | — |
| H — physical-tolerance decision | Deferred to a separate isolated A/B (H1.10) | — |

Every agent in the coordination is active-power based, and all three now share
one relative complementarity semantics. Nothing in P5.4 changed
`ESS_COMPLEMENTARITY_TOLERANCE`, IPOPT tolerances, MA97/exact-Hessian policy,
ADMM rho or tolerances, proximal regularization, objective scaling, Benders
logic, or any objective coefficient. No binary charge/discharge variables and
no additional penalties were introduced. The dimensionless variables added in
H1 are confined to the complementarity rows.

---

# A — Productionize shared network ESS

Commit `a4a0bae8`. Files changed: `definitions.py`,
`model_construction_helpers.py`, `network.py`, `shared_resources_planning.py`
(+101 / −110 lines). No diagnostic wrapper is involved — this is production.

## Retired

`shared_es_sch`, `shared_es_sdch`, `sess_snet_def`, `sess_pch_link`,
`sess_pdch_link`, the apparent-power `sess_s_limit`, the apparent-power
`sess_soc_def` body, and the apparent-power `sess_comp` body.

Also retired, being machinery whose only purpose was normalizing
`sess_snet_def`: `sess_snet_def_kappa`, `shared_ess_snet_def_scale`,
`_sync_sess_snet_def_scale`, `SHARED_ESS_SNET_DEF_SAFE_KAPPA`. **Unrelated
warm-start handling was not touched**, and no production code was deleted
beyond that scope.

## Implemented

| Item | Implementation |
|---|---|
| Time step | New `period_duration_hours(m)` = `HOURS_PER_REPRESENTATIVE_DAY / len(m.periods)`. **Derived, not hardcoded**; verified to be **exactly `1.0 h`** for the 24-instant representative day, reproducing the previous numerical coefficient. |
| SOC | `SOC_t = SOC_{t-1} + eta_ch·pch·dt − pdch·dt/eta_dch` |
| Converter capability | `sess_converter_capability`: `pnet² + qnet² ≤ S_rated²` |
| Active envelope | `sess_active_sum_limit`: `pch + pdch ≤ S_rated` — **derived** from the retired set (`pch ≤ sch`, `pdch ≤ sdch`, `sch + sdch ≤ S_rated`) |
| Explicit bounds | `0 ≤ pch, pdch ≤ S_rated`, `|pnet|, |qnet| ≤ S_rated`, kept in step with installed capacity inside `configure_shared_ess_operational_state` |
| Complementarity | `sess_comp` on `pch·pdch`; `ESS_COMPLEMENTARITY_TOLERANCE` **preserved exactly** |
| Retained unchanged | `sess_pnet_def` (`pnet = pch − pdch`), PF cone rows (they already depended on `pch/pdch`, not `sch/sdch`) |

All new rows use the rated-capacity **variable** `shared_es_s_rated`, exactly as
the retired `sess_s_limit` did, so Benders sensitivity extraction continues to
flow through that variable.

## Objective

The shared-ESS usage penalty moves from `sch + sdch` to `pch + pdch`, and the
complementarity penalty from `sch·sdch` to `pch·pdch`. **All coefficients are
unchanged.** This is part of the active-energy physical correction, **not** an
objective-tuning experiment.

## Output and diagnostic contracts

- **`s_ess`** (`network.py` shared-ESS results). Previously the *signed*
  apparent charge/discharge difference `sch − sdch`. Since the retired
  variables no longer exist, it is now the **converter loading magnitude**
  `sqrt(pnet² + qnet²)` in **MVA**, documented in place. Direction is not lost:
  the `p` field remains the signed load-positive active power. The semantic
  change is explicit rather than silent.
- **ADMM charge/discharge diagnostic.** Renamed
  `_get_expected_network_shared_ess_charge_discharge_mva` →
  `…_charge_discharge_mw` and now reads active `pch/pdch`. **No `MVA` label
  remains on an active-power quantity.**

## Construction and physics checks on the production code

| Check | Result |
|---|---|
| `period_duration_hours` | **1.0 h exactly** |
| Retired components absent | all of `sess_snet_def`, `sess_pch_link`, `sess_pdch_link`, `sess_s_limit` ✓ |
| Retired variables absent | `shared_es_sch`, `shared_es_sdch` ✓ |
| `sess_snet_def_kappa` absent | ✓ |
| New components present | `sess_converter_capability`, `sess_active_sum_limit` ✓ |
| Bounds synced to rating | `pch ∈ [0, 2.127e-04]`, `qnet ∈ [−2.127e-04, 2.127e-04]` ✓ |
| **Pure reactive** | **ΔSOC = +0.000000e+00 exactly** ✓ |
| Pure charging | ΔSOC = +8.252677e-05 = `eta_ch·pch·dt` exactly ✓ |
| Pure discharging | ΔSOC = −8.862411e-05 = `−pdch·dt/eta_dch` exactly ✓ |
| Zero capacity | variables fixed at 0 with bounds collapsed to 0, rows deactivated ✓ |
| Zero → positive restore | variables unfixed, bounds restored, rows reactivated ✓ |

---

# E — Production validation before ADMM (shared-ESS scope)

Complete positive-bootstrap initialization run with the **actual production
implementation**.

| Metric | Production (P5.4-A) | B3 prototype | Old production |
|---|---|---|---|
| DSO | **36 / 36** | 36 / 36 | 33 / 36 |
| TSO | **12 / 12** | 12 / 12 | 12 / 12 |
| ESSO | **3 / 3** | 3 / 3 | 3 / 3 |
| Primary failures | **0** | 0 | 3 |
| Recovery attempts | **0** | 0 | 0 |
| **Persistent failures** | **0** | 0 | 3 |
| Iterations — total | **1 546** | 1 545 | 33 073 |
| Iterations — mean / median / max | **32.2 / 28 / 120** | 32.2 / 27.5 / 109 | 689 / 468 / 3000 |
| Runtime | **35 s** | 37 s | 274 s |

The production implementation reproduces the B3 prototype to within one
iteration in total — confirming the production code implements the validated
formulation.

## Rank / derivative confirmation

| Model | `sess_snet_def` exists? | Zero-gradient equality rows | σ_min(full) | Reduced condition |
|---|---|---|---|---|
| case33_1/2030/Winter | **No** | **0** | **5.9246e-03** | 8.9835e+04 |
| case9/2025/Winter | **No** | **0** | **3.2871e-02** | 1.4227e+03 |

The component no longer exists, **the new ESS formulation introduces no
zero-gradient equality rows**, and both representative full equality Jacobians
are **full row rank**.

## Revalidation after B, C and D (commit `c3526ec8`)

The same production validation was re-run at the final HEAD, so the numbers
below reflect the ordinary-ESS and ESSO conversions as well:

| Metric | Post-A only | **Post-B/C/D (final)** |
|---|---|---|
| DSO | 36 / 36 | **36 / 36** |
| TSO | 12 / 12 | **12 / 12** |
| ESSO | 3 / 3 | **3 / 3** |
| Primary failures | 0 | **0** |
| Recovery attempts | 0 | **0** |
| **Persistent failures** | 0 | **0** |
| Iterations — total | 1 546 | **1 556** |
| Iterations — mean / median / max | 32.2 / 28 / 120 | **32.4 / 28.5 / 119** |
| Runtime | 35 s | **32 s** |
| σ_min(full), case33_1/2030/Winter | 5.9246e-03 | **5.9246e-03** |
| σ_min(full), case9/2025/Winter | 3.2871e-02 | **3.2871e-02** |
| Zero-gradient equality rows | 0 | **0** |
| Converter capability violations | 0 / 864 | **0 / 1 728** |

**Full equality row rank is confirmed in both representative models.** The
residual audit now covers all **1 728** rows (DSO + TSO) rather than the 864 DSO
rows of the original E run.

## Physical residual audit — original E run, 864 DSO rows

Normalized as the plan specifies, and **not** judged from IPOPT's reported
feasibility metric.

| Residual | max | mean | p95 | > 1e-6 | > 1e-4 |
|---|---|---|---|---|---|
| Converter capability `max((pnet²+qnet²−S²)/S², 0)` | **0.000e+00** | 0.000e+00 | 0.000e+00 | **0** | **0** |
| Complementarity `max((pch·pdch − eps·S²)/S², 0)` | **1.847e-04** | 5.415e-05 | 1.718e-04 | **288** | **288** |

> **The converter capability circle is never violated.** This is a better
> outcome than B3's diagnostic run suggested and removes one of the two
> anticipated concerns.
>
> **The active complementarity is materially violated on 288 of 864 rows
> (33 %)**, worst case `1.847e-04` normalized at `dso/9/2025/Winter` period 20.
> Moving complementarity from `sch·sdch` to `pch·pdch` did **not** repair this,
> exactly as B3 predicted, and no scaling was introduced to mask it.

> **Correction (planner, accepted).** An earlier revision of this section read
> the worst product as "roughly 1.7 % of rating flowing simultaneously in both
> directions". **That interpretation was too strong and is withdrawn.** From
> `pch·pdch/S² ≈ 2.847e-04` only the *geometric mean*
> `sqrt(pch·pdch)/S ≈ 1.69 %` is established; a given product is equally
> consistent with one large and one small directional power. The physically
> relevant simultaneous-circulation quantity is `p_circ = min(pch, pdch)`, and
> it is measured — not inferred — in **P5.4-E2** below.

> **Scope correction.** The 864 rows audited above are the **DSO rows only**.
> The TSO models also carry active shared-ESS rows and were not included. P5.4-E2
> audits both, and the TSO rows turn out to be substantially worse.

---

# E2 — Physical significance of the complementarity residual

Script: `p54e2_complementarity_significance.py`. Evidence:
`data/SRP1/Results/P54E2/p54e2_report.json` (all 1 728 rows are persisted).
**No formulation change was made in E2** — this is measurement only, on the
production positive-bootstrap solutions (36/36 DSO, 12/12 TSO, 3/3 ESSO,
0 persistent failures).

All physical parameters (`eta_ch`, `eta_dch`, `dt`, `baseMVA`, `E_rated`) are read
from the live production model and network objects; none is defaulted.

## Scope

`1 728` active shared-ESS rows = **864 DSO + 864 TSO**. The E-stage audit covered
only the DSO half. The DSO half reproduces the E numbers to three significant
figures (`max r_violation` 1.8466e-04 here vs. 1.847e-04 there; residual
differences are IPOPT run-to-run variation, not a formulation difference).

## Circulating power `p_circ = min(pch, pdch)`

| Population | max | mean | median | p95 | p99 |
|---|---|---|---|---|---|
| **All 1 728 rows**, `p_circ/S` | **1.1562e-01** | 1.4283e-02 | 7.9608e-03 | 5.2317e-02 | 9.1039e-02 |
| **All**, `p_circ` [MW] | 2.9219e-03 | 2.8719e-04 | 1.6228e-04 | 1.2828e-03 | 2.4655e-03 |
| **DSO** (864), `p_circ/S` | 1.6690e-02 | 8.9112e-03 | 7.7017e-03 | 1.5696e-02 | 1.6329e-02 |
| **TSO** (864), `p_circ/S` | **1.1562e-01** | 1.9654e-02 | 9.2336e-03 | 7.6454e-02 | 1.0398e-01 |

**Nonzero on 1 728 / 1 728 rows.** Counts above threshold, all rows:

| `p_circ >` | rows |
|---|---|
| `1e-4·S` | **1 728 (100 %)** |
| `1e-3·S` | **1 713 (99.1 %)** |
| `1e-2·S` | **694 (40.2 %)** |

## Product residual, for comparison

| Population | `pch·pdch/S²` max | `r_violation` max | rows with `r_violation > 0` |
|---|---|---|---|
| All | 2.8123e-02 | 2.8023e-02 | **1 062 / 1 728 (61 %)** |
| DSO | 2.8466e-04 | 1.8466e-04 | 288 / 864 (33 %) |
| **TSO** | **2.8123e-02** | **2.8023e-02** | **774 / 864 (90 %)** |

The TSO worst violation is `2.80e-02`, i.e. **≈ 280 × `eps`**, two orders of
magnitude beyond the DSO worst case.

## Worst rows (all TSO; top 6 of the persisted 20)

| Model | p | `pch/S` | `pdch/S` | `p_circ/S` | `p_circ` [MW] | `p_net/S` | `r_prod` | `r_violation` | SOC | `E_circ_loss` [MWh] |
|---|---|---|---|---|---|---|---|---|---|---|
| tso/1/2030/Summer | 13 | 2.3781e-01 | 1.1562e-01 | **1.1562e-01** | 2.459e-03 | 1.2219e-01 | 2.7496e-02 | 2.7396e-02 | 2.6845e-04 | 1.762e-04 |
| tso/2/2030/Summer | 13 | 2.3781e-01 | 1.1562e-01 | 1.1562e-01 | 2.459e-03 | 1.2219e-01 | 2.7496e-02 | 2.7396e-02 | 2.6845e-04 | 1.762e-04 |
| tso/0/2030/Summer | 13 | 2.3781e-01 | 1.1562e-01 | 1.1562e-01 | 2.459e-03 | 1.2219e-01 | 2.7496e-02 | 2.7396e-02 | 2.6845e-04 | 1.762e-04 |
| tso/2/2030/Summer | 14 | 2.4190e-01 | 1.1344e-01 | 1.1344e-01 | 2.413e-03 | 1.2847e-01 | 2.7441e-02 | 2.7341e-02 | 2.9322e-04 | 1.729e-04 |
| tso/1/2030/Summer | 12 | 2.5015e-01 | 1.0925e-01 | 1.0925e-01 | 2.324e-03 | 1.4090e-01 | 2.7329e-02 | 2.7229e-02 | 2.4500e-04 | 1.665e-04 |
| tso/1/2035/Summer | 13 | 1.9726e-01 | 9.1581e-02 | **9.1581e-02** | **2.922e-03** | 1.0568e-01 | 1.8066e-02 | 1.7966e-02 | 4.1444e-04 | **2.094e-04** |

The worst row charges at **23.8 % of rating while simultaneously discharging at
11.6 % of rating**, for a net of only 12.2 %. Note that here the geometric mean
`sqrt(r_prod) = 16.6 %` **overstates** `p_circ/S = 11.6 %` — confirming the
planner's point that the product alone does not pin down simultaneity in either
direction.

## Artificial cycling loss

`E_circ_loss = min(pch, pdch) · dt · (1/eta_dch − eta_ch)` — the stored energy
destroyed by the circulating component, which produces no net injection.

| Quantity | Value |
|---|---|
| Max per period | 2.0940e-06 p.u. = **2.0940e-04 MWh** |
| Worst representative day, total | 1.2655e-03 MWh (`tso/1/2030/Summer`) |
| Worst day, normalized by `E_rated` | **2.9749e-02 (3.0 %)** |
| Worst day, share of legitimate throughput | **3.4210e-02 (3.4 %)**, at `dso/9/2030/Spring` |
| Worst DSO day, normalized by `E_rated` | 1.3051e-02 (1.3 %) |

Legitimate throughput is `eta_ch·pch·dt + pdch·dt/eta_dch`, i.e. the same
active-energy quantity the SOC recursion uses.

## Associated objective penalty

`PENALTY_ESS_COMPLEMENTARITY = 1e2`. The worst row's penalty contribution is
`1e2 · pch · pdch = 1.8389e-07` in per-unit-base objective terms. **At these
capacities the penalty exerts essentially no pressure**, which is a consistent
explanation for why the residual persists: nothing in the objective meaningfully
opposes it, and the inequality itself is scaled by `S²`, which is `O(1e-8)`.

## What E2 establishes

1. The circulating power is **real and not negligible** — 40 % of rows exceed
   1 % of rating, and the worst exceeds 11 %.
2. It is **concentrated in the TSO models**, which the E audit did not cover.
3. Its **energy consequence is small but not zero**: up to 3.4 % of a day's
   legitimate throughput is destroyed as artificial cycling loss.
4. The **objective penalty is numerically irrelevant** at these capacities.

E2 makes no recommendation and changes no formulation; per the plan, the
H decision remains deferred to post-ADMM evidence.

---

# B — Ordinary network ESS parity

Commit `1e86d40e`. Script: `p54b_ordinary_ess_validation.py`. Evidence:
`data/OP1/Results/P54B/p54b_report.json`.

SRP1 instantiates no ordinary ESS, so the live case is **OP1 / case33_3 / 2025 /
Summer**, two ordinary ESS units (`S = 0.005 p.u. = 0.5 MVA`, `E = 0.01 p.u.`) —
the same case the accepted P4.6-B1/B2 stages used, so the comparison is
like-for-like.

## Retired

`es_sch`, `es_sdch`, `ess_snet_def` and the whole kappa_es machinery
(`ordinary_ess_snet_def_scale`, `ess_snet_def_scale_init`, the
`ess_snet_def_scale` Param, and `ORDINARY_ESS_MIN_RATED_POWER`, whose only
purpose was to reject a rating that would have made that scale a division by
zero), `ess_pch_link`, `ess_pdch_link`, the apparent `ess_s_limit`, and the
then-unused `s_bounds`.

## Implemented

| Item | Implementation |
|---|---|
| SOC | `SOC_t = SOC_{t-1} + eta_ch·pch·dt − pdch·dt/eta_dch`, `dt` from `period_duration_hours` |
| Converter capability | `ess_converter_capability`: `pnet² + qnet² ≤ S²` |
| Active envelope | `ess_active_sum_limit`: `pch + pdch ≤ S + EQUALITY_TOLERANCE` — **derived** from the retired set, and the pre-existing tolerance is **preserved, not tightened** |
| Complementarity | `ess_comp` on `es_pch · es_pdch`; **all three `ess_model` branches and every tolerance unchanged** |
| Objective | usage penalty `sch + sdch` → `pch + pdch`, complementarity penalty `sch·sdch` → `pch·pdch`; **coefficients unchanged** |

One deliberate difference from the shared ESS: the ordinary-ESS rating is
**fixed network data, not a decision variable**, so the new rows take `ess.s` as
a constant.

## Results

| Check | Result |
|---|---|
| `period_duration_hours` | **1.0 h exactly** |
| Retired variables absent | `es_sch`, `es_sdch` ✓ |
| Retired components absent | `ess_snet_def`, `ess_snet_def_scale`, `ess_pch_link`, `ess_pdch_link`, `ess_s_limit` ✓ |
| New components present | `ess_converter_capability` (48 rows), `ess_active_sum_limit` (48 rows) ✓ |
| Complementarity acts on active powers | ✓ |
| SOC acts on active powers | ✓ |
| No division by a variable in the new rows | ✓ |
| Bounds synced to rating | `pch ∈ [0, 5.01e-03]`, `qnet ∈ [−5.01e-03, 5.01e-03]` ✓ |
| **Pure reactive** | **ΔSOC = +0.000000e+00 exactly** ✓ |
| Pure charging | ΔSOC = +1.9000e-03 vs `eta_ch·pch·dt` = +1.9000e-03; abs error **2.17e-19** (one ulp) |
| Pure discharging | ΔSOC = −2.105263157894737e-03 = `−pdch·dt/eta_dch` **exactly** ✓ |
| No ordinary ESS | model constructs, all ESS row counts zero ✓ |
| Zero-rated / below-tolerance ESS | **now constructs** rather than being rejected — with no kappa_es there is no division to guard, and the rows simply collapse. The P4.6-B2 construction-time rejection is therefore obsolete and was removed with it. |

## OP1 solve

| Metric | P4.6-B2 phase A (unscaled) | P4.6-B2 phase B (kappa) | **P5.4-B (active)** |
|---|---|---|---|
| Status | ok / optimal | ok / optimal | **ok / optimal** |
| Iterations | 707 | 331 | **144** |
| Recovery used | no | no | **no** |
| Objective | 242.6438 | 242.5403 | 242.7423 |

IPOPT exit `Optimal Solution Found`; constraint violation 2.02e-08, overall NLP
error 3.93e-04 (unscaled).

## Residuals and circulating-power instrumentation (48 rows)

| Residual | max | violating rows |
|---|---|---|
| Converter capability, normalized | **0.000e+00** | **0 / 48** |
| Complementarity, normalized | **0.000e+00** | **0 / 48** |

| `min(pch, pdch)/S` | max | mean | median | p95 | p99 |
|---|---|---|---|---|---|
| | 8.3843e-04 | 1.2197e-04 | 7.5637e-05 | 5.0244e-04 | 7.1894e-04 |

Above threshold: `>1e-4·S` **12 / 48**; `>1e-3·S` **0**; `>1e-2·S` **0**. Peak
circulating power 4.1921e-04 MW.

Artificial cycling loss per representative day: ESS 1 — 1.9865e-04 MWh
(= 1.9865e-04 of `E_rated`, 1.2355e-04 of legitimate throughput); ESS 2 —
1.0177e-04 MWh (1.0177e-04 of `E_rated`, 6.3372e-05 of throughput).

> **The ordinary-ESS case satisfies both new inequalities exactly.** Taken with
> E2, this says the shared-ESS complementarity residual is **not intrinsic to
> the active-energy formulation** — the same formulation on a 0.5 MVA ordinary
> ESS shows no violation at all. The difference is the operating regime: the
> bootstrap shared-ESS capacity is ~24× smaller.

## Sign and units re-confirmation

Processed results equal model values for P and Q; all 7 charging periods report
positive `P` and all 17 discharging periods report negative `P`, so the P4.6-B1
load-positive convention is intact. The `s` field is now the converter loading
magnitude `sqrt(pnet² + qnet²)` in MVA — verified equal to that expression to
`<1e-9` and non-negative throughout — documented in place, with `p` still
carrying the signed active power.

---

# C — ESSO active-energy conversion

Commit `58f4911b`. Script: `p54c_esso_active_energy_validation.py`. Evidence:
`data/SRP1/Results/P54C/p54c_report.json`.

## Trace, before changing anything

The ESSO model is per-investment-cohort (`[y_inv, y]`), over 3 years × 4
representative days × 24 periods, with **no scenario index**. Its apparent
per-cohort variables `es_sch_per_unit` / `es_sdch_per_unit` fed exactly four
places:

1. `energy_storage_charging_discharging` — the degradation throughput
   `avg_ch_dch = Σ_d Σ_p (num_days/365)·(sch + sdch)`;
2. `energy_storage_limits` — `sch ≤ s_max`, `sdch ≤ s_max`;
3. `energy_storage_complementarity` — `sch·sdch ≤ eps` (**absolute**, not
   `eps·S²` as in the network models);
4. `energy_storage_operation_agg` — `es_snet = Σ_cohorts (sch − sdch)` and then
   `es_snet² == es_pnet² + es_qnet²`.

Plus cohort gating (`_set_esso_variable_state`), result processing, and the ADMM
diagnostic helper.

> **The ESSO has no state of charge.** There is no `es_soc` variable and no SOC
> recursion anywhere in `shared_energy_storage_data.py`; the only "SoC"
> occurrences are a stale comment on line 524 and `e_init` on the *data* object,
> which is consumed by the **network** models, not by the ESSO. Confirmed on the
> built model: `has_no_soc_variable = True`, with only SoH variables present.
> The instruction to drive "SOC from active energy" therefore **has no target in
> the ESSO**, and no SOC was invented for it. The ESSO tracks throughput for
> degradation, not stored energy.

## Degradation — what changed and what did not

**Only the throughput input changed:**

```
avg_ch_dch  =  Σ_d Σ_p (num_days/365) · ( eta_ch·pch·dt  +  pdch·dt/eta_dch )
```

**Preserved verbatim**, and re-read off the built model rather than restated:

| Preserved | Value in this case |
|---|---|
| Representative-day weighting `num_days/365` | Spring 0.252055, Summer 0.249315, Autumn 0.249315, Winter 0.249315 |
| Year multiplicities | 2025: 5, 2030: 5, 2035: 5 |
| Cohort `[y_inv, y]` indexing and gating | 288 fixed / 576 free cohort entries |
| Equivalent-cycle normalization `2·cl_nom·E_rated` | `cl_nom = 10000` |
| Calendar-life gating `t_cal` | 15 |
| `soh_min` | 0.5 |
| Cumulative SoH exponent `365·num_years` | present |
| Row counts | 9 throughput rows / 30 degradation rows |

**Units, confirmed before making the change.** The law divides `avg_ch_dch` by
`2·cl_nom·E_rated` with `E_rated` in p.u. **energy**, so the numerator must be
energy. The former expression summed **powers** over a day's periods and was
therefore only dimensionally correct under an *implicit* `dt = 1 h`. That
assumption is now explicit via `period_duration_hours()`, which is exactly
`1.0 h` here, so the change is **unit-correcting rather than a re-tuning**.

Measured on the production row at a 0.001 p.u. test dispatch: measured body
`4.828000e-02`, new active-energy expression `4.828000e-02` (**exact match**),
old apparent expression `4.800000e-02`. The ratio is
**1.005833 = (eta_ch + 1/eta_dch)/2 = (0.97 + 1/0.96)/2** — i.e. the entire
numerical difference is the efficiency correction, with `dt` contributing a
factor of exactly 1.

## Aggregate operation

With active cohort powers the cohort sum **is** the aggregate active power, so
`es_pnet` is defined directly. This retires `es_snet` and the equality
`es_snet² == es_pnet² + es_qnet²` — **the ESSO instance of the same
exact-zero-gradient row P5.4-A retired in the network models** — and replaces it
with the converter capability **inequality** `es_pnet² + es_qnet² ≤ es_s_rated²`,
in parity with `sess_converter_capability`. Aggregate P and Q both remain
represented: P by the cohort sum, Q by the capability circle.

Per-cohort limits and the ESSO complementarity bound keep their existing forms
and tolerances exactly. **The ESSO's absolute complementarity tolerance was not
changed to the network's `eps·S²` form** — see the finding below.

## Follow-on renames

The surviving slack pair now slacks active power, so
`slack_es_snet_up/down` → `slack_es_pnet_up/down`, result keys
`snet_up`/`snet_down` → `pnet_up`/`pnet_down`, Excel labels `Snet, up/down` →
`Pnet, up/down`. The `Snet definition` slack pair and its Excel block are gone
with the equality they slacked. `_get_esso_shared_ess_charge_discharge_mva` →
`_get_esso_shared_ess_charge_discharge_pu`: it returned p.u. while its name and
docstring claimed MVA. The ADMM charge/discharge diagnostic keys `sch`/`sdch`
become `pch`/`pdch` for all three agents. The full results and Excel path was
exercised end-to-end and produces the expected `Pnet, up` / `Pnet, down` rows.

## Results

| Check | Result |
|---|---|
| Retired absent | `es_snet`, `es_sch_per_unit`, `es_sdch_per_unit`, `slack_es_snet_up/down`, `slack_es_snet_def_up/down` ✓ |
| New present | `es_pch_per_unit`, `es_pdch_per_unit` ✓ |
| `dt` | **1.0 h** (24 periods, 3 years, 4 days) |
| Throughput uses active variables | ✓ |
| ESSO solves | **3 / 3** (nodes 5, 7, 9) |
| Aggregate active power preserved | **max \|es_pnet − Σ(pch − pdch)\| = 1.724e-14** ✓ |
| Converter capability violation | **0.000e+00** |
| Per-cohort complementarity violation | **0.000e+00** |
| Cohort gating | 288 fixed / 576 free, unchanged behaviour |

> **Finding — the ESSO complementarity bound is vacuous at these capacities.**
> The per-cohort violation is exactly zero, yet the **aggregate** circulating
> power reaches `min(pch, pdch)/S = 1.9120e-01` over 864 active rows. There is
> no contradiction: the ESSO bound is the **absolute** `pch·pdch ≤ 1e-4`, while
> `S ≈ 2.13e-04 p.u.`, so the bound permits directional powers roughly 47× the
> rating and never binds. The network models use `eps·S²` instead. This
> asymmetry is **pre-existing and was deliberately not changed** — the planner
> excluded introducing a new complementarity tolerance from P5.4's scope — but
> it is a live issue for ADMM, because the ESSO's ESS consensus variable can
> circulate essentially without penalty while the network agents' cannot.

---

# D — Lifecycle and sensitivity audit

Commit `c3526ec8`. Script: `p54d_lifecycle_sensitivity_audit.py`. Evidence:
`data/SRP1/Results/P54D/p54d_report.json`. Case: SRP1 / DSO node 9 / 2030 /
Winter.

## Passing

| Item | Result |
|---|---|
| Obsolete kappa transfer removed | `shared_ess_snet_def_scale`, `_sync_sess_snet_def_scale`, `ordinary_ess_snet_def_scale`, `ess_snet_def_scale_init` all absent ✓ |
| No replacement transformation introduced | ✓ |
| S / E sensitivity rows present | `shared_energy_storage_s_sensitivities`, `shared_energy_storage_e_sensitivities` ✓ |
| New rows reference the rated **variable** | ✓ — so Benders sensitivity still flows through `shared_es_s_rated` |
| `shared_es_s_rated_fixed` still a mutable Param, `shared_es_s_rated` still a Var | ✓ |
| Suffixes present | `dual`, `ipopt_zL_out/in`, `ipopt_zU_out/in`; `dual` is `IMPORT_EXPORT` ✓ |
| Snapshot / clear / restore intact | covers `ipopt_zL_in`, `ipopt_zU_in`, `dual` ✓ |
| Duals actually available after a solve | `dual_s = −2.7674`, `dual_e = −1.6776` ✓ |

## Capacity transitions on one reused model

| Transition | S (p.u.) | inactive | `pch` fixed | `pch` bounds | rows active |
|---|---|---|---|---|---|
| zero | 0.0 | yes | yes | [0, 0] | no |
| → positive | 2.1270e-04 | no | no | [0, 2.1270e-04] | yes |
| → different positive | 5.3174e-04 | no | no | [0, 5.3174e-04] | yes |
| → back to zero | 0.0 | yes | yes | [0, 0] | no |
| → positive again | 2.1270e-04 | no | no | [0, 2.1270e-04] | yes |

**Reused-model identity: the `id()` of every tracked shared-ESS component —
all ten row families, all five operational variables, both rated variables, both
fixed Params, and both sensitivity rows — is constant across all six
snapshots.** The model is reconfigured in place; nothing is rebuilt or
duplicated.

## Negative result — the analytic capacity sensitivity is not confirmed

Swept over four decades of relative step at the bootstrap capacity:

| Relative step | Δobjective | central difference | analytic dual | relative error |
|---|---|---|---|---|
| 0.5 | −6.1863e-05 | −2.9085e-01 | −2.7674e+00 | 8.949e-01 |
| 0.1 | −8.3733e-07 | −1.9684e-02 | −2.7674e+00 | 9.929e-01 |
| 0.01 | +1.4210e-06 | +3.3405e-01 | −2.7674e+00 | 1.121e+00 |
| 0.001 | +1.3167e-06 | +3.0972e-01 | −2.7674e+00 | 1.112e+00 |

**The central difference does not settle toward the dual, and changes sign.**
The objective moves by only `6.2e-05` across a **±50 %** capacity sweep while the
dual predicts `5.9e-04` — an order of magnitude more — so the objective's
dependence on shared-ESS capacity in this reduced local-solve configuration is
close to the solver's own accuracy. At 100× capacity the objective *jumps* by
`±6.06` with opposite signs at different step sizes, i.e. the perturbed solves
land in different local optima and a smooth derivative is not well defined
there. Across three decades of capacity the dual scales roughly as `1/S`
(`dual·S ≈ −1e-3` throughout), which is not the signature of a smooth objective
derivative.

**Attribution.** An identical probe (`fd_probe.py`, run in a clean git worktree
at `a4a0bae8^`) shows the **pre-P5.4-A code fails the same validation, and far
worse**:

| Relative step | pre-A relative error | post-C relative error |
|---|---|---|
| 0.5 | 1.279e+02 | **8.949e-01** |
| 0.1 | *perturbed solve failed* | **9.929e-01** |
| 0.01 | 6.416e+03 | **1.121e+00** |

At 100× capacity the pre-A tree gives relative error ≈ 1.0005 at every step,
i.e. the finite difference is ≈ 0 while the dual is `−0.0275`.

> **This is pre-existing, not introduced by P5.4, and P5.4 strictly improves
> it** (errors of order 1 rather than 10²–10³, and no failed perturbed solves).
> It is nonetheless a real risk to Benders cut quality and is reported, not
> fixed: Benders and local-cut logic are outside P5.4's authorized scope.

---

# H1 — Dimensionless charge/discharge complementarity

Commit `93974d83`. Scripts: `p54h1_normalized_complementarity_audit.py`
(H1.2/H1.3), `p54h1_gate.py` (H1.8/H1.9), `p54b_ordinary_ess_validation.py`
(H1.4). Evidence: `data/SRP1/Results/P54H1/`, `data/OP1/Results/P54B/`,
`data/SRP1/Results/P54E/p54e_report.json`.

H1 answers **one** question: can IPOPT enforce the *existing* `1e-4` relative
complementarity when the row is written at O(1) scale? It does **not** tighten
the physical tolerance — `ESS_COMPLEMENTARITY_TOLERANCE` remains `1e-4`.

## What changed

Only the complementarity inequality, in all three ESS agents:

```
pch * pdch <= eps * S_rated^2        ->        pch_hat * pdch_hat <= eps
```

linked by

```
pch  - S_rated * pch_hat  == 0
pdch - S_rated * pdch_hat == 0
```

The link is written as a product, never as `pch_hat = pch/S`, so **no
expression anywhere divides by the rated capacity** — verified on the built
rows. It keeps a unit coefficient on the physical variable, which is precisely
what prevents the zero-gradient equality defect that `sess_snet_def` had.

Unchanged: `pnet = pch − pdch`, SOC, `pch + pdch <= S`, physical variable
bounds, converter capability (not normalized — its audited violation is zero),
PF/reactive rows, and every objective term and coefficient. The dimensionless
variables are internal: SOC, `pnet`, objectives, results, ADMM diagnostics and
the sensitivity rows all still use the physical `pch/pdch`. Their `[0, 1]`
bounds are implied by the existing envelopes, so they add no restriction.

## H1.1 — exact reformulation, verified

Probed at five states straddling the old boundary `pch·pdch = eps·S²`:

| Probe | old margin `eps·S² − pch·pdch` | new margin `eps − hat product` | signs agree |
|---|---|---|---|
| strictly interior | +3.3930e-12 | +7.5000e-05 | ✓ |
| **on the old boundary** | **+0.0000e+00** | **+0.0000e+00** | ✓ |
| strictly violating | −1.3572e-11 | −3.0000e-04 | ✓ |
| asymmetric interior | +4.1169e-12 | +9.1000e-05 | ✓ |
| asymmetric violating | −4.0264e-10 | −8.9000e-03 | ✓ |

The two margins differ by exactly the factor `S_rated²`, so the feasible set is
identical and only its numerical representation changes. Link residuals were
`0` at every probe.

## H1.2 — zero-capacity and reused-model lifecycle

| Transition | S (p.u.) | `pch_hat` / `pdch_hat` fixed | value | bounds | link / comp rows active | link residual |
|---|---|---|---|---|---|---|
| zero | 0.0 | yes / yes | 0, 0 | [0, 1] | no / no | 0 |
| → positive | 2.1270e-04 | no / no | 0, 0 | [0, 1] | yes / yes | 0 |
| → different positive | 5.3174e-04 | no / no | 0, 0 | [0, 1] | yes / yes | 0 |
| → back to zero | 0.0 | yes / yes | 0, 0 | [0, 1] | no / no | 0 |
| → positive again | 2.1270e-04 | no / no | 0, 0 | [0, 1] | yes / yes | 0 |

The `id()` of every tracked component — both link rows, both hat variables,
`sess_comp`, `sess_pnet_def`, capability, active-sum, SOC, both physical
powers, and both rated quantities — is **constant across all six snapshots**.
The dimensionless pair is gated by exactly the same production lifecycle as the
physical pair.

## H1.3 — derivative and rank audit

| Quantity | DSO `case33_1/2030/Winter` | TSO `case9/2025/Winter` |
|---|---|---|
| Link-row gradient ‖·‖₂ (charge / discharge) | 1.0000e+00 / 1.0000e+00 | 1.0000e+00 / 1.0000e+00 |
| **Unit coefficient on the physical variable** | ✓ / ✓ | ✓ / ✓ |
| Complementarity RHS | **1.0000e-04** | **1.0000e-04** |
| …what the old row's RHS would have been | 4.5240e-12 | 1.1310e-12 |
| **RHS amplification** | **× 2.2104e+07** | **× 8.8417e+07** |
| Comp-row gradient ‖·‖₂ at cold start | 0.0000e+00 | 0.0000e+00 |
| Comp-row gradient ‖·‖₂ at half rating | **7.0711e-01** | **7.0711e-01** |
| Equality rows | 4 155 | 1 449 |
| **Zero-gradient equality rows** | **0** | **0** |
| Link rows among zero-gradient rows | **no** | **no** |
| `σ_min(full)` | 5.9246e-03 | 3.5951e-02 |
| Full row rank | ✓ | ✓ |

All four required structural properties hold. The comp-row gradient is zero at
`pch_hat = pdch_hat = 0`, which the plan anticipates and accepts: this is an
**inequality** that is strictly interior by `1e-4` there, categorically
different from the old zero-gradient *equality* defect. Away from the origin
its gradient is `O(1)` (`0.707` at half rating), i.e. curvature is `O(1)`
rather than scaling as `1/S²`.

*(The TSO `σ_min` here is `3.5951e-02` against `3.2871e-02` in section E; the E
figure is measured on a model captured mid-run with interface values applied,
this one on a freshly built model. Both are full rank.)*

## H1.4 — ordinary network ESS parity (OP1)

| Metric | P4.6-B2 phase A | P4.6-B2 phase B | P5.4-B (active) | **P5.4-H1** |
|---|---|---|---|---|
| Status | ok / optimal | ok / optimal | ok / optimal | **ok / optimal** |
| Iterations | 707 | 331 | 144 | **89** |
| Recovery used | no | no | no | **no** |

**Iterations improved, not regressed.** Complementarity RHS is `1.0000e-04` and
acts on the dimensionless pair; link rows carry the unit coefficient; hat
bounds are `[0, 1]`. Capability violations **0 / 48**; complementarity
violations **0 / 48**. Physical `min(pch,pdch)/S`: max `1.3522e-03`, mean
`2.1803e-04`, `>1e-3` on 2 rows, **`>1e-2` on 0 rows**. SOC physics unchanged
(pure reactive ΔSOC exactly zero; discharge exact; charge to one ulp).

## H1.5 / H1.6 — ESSO

**Trace first.** `S_cohort` is `es_s_rated_per_unit[y_inv, y]` — the same
`s_max` the existing per-cohort limit rows already use, defined by
`rated_s_capacity_unit` as `es_s_investment[y_inv]` within the cohort's
lifetime. `S_total` is `es_s_rated[y]`, which `rated_s_capacity` already
defines as `Σ_{y_inv} es_s_rated_per_unit[y_inv, y]`. Both are p.u. on the same
base as the cohort powers. **No rating was invented or oversized.**

The previous **absolute** `pch·pdch <= 1e-4` is superseded by the relative
condition. At bootstrap capacity (`S ≈ 2.13e-04 p.u.`) the absolute bound
permitted directional powers roughly 47× the rating and never bound — which is
exactly why section C measured zero per-cohort violation alongside 19 %
aggregate circulation.

**H1.6 aggregate row added.** Per-cohort complementarity alone permits one
cohort to charge while another discharges at the same node and time, leaving
the ESSO feasible set incompatible with the single aggregate device the network
agents represent. The aggregate link uses unit coefficients on the physical
cohort sum. **Cohort-level complementarity is preserved** — the aggregate row
does not imply it.

## H1.7 — objective policy

No objective term or coefficient was changed. The usage and complementarity
penalties remain expressed with the physical `pch`/`pdch`, and enforcement is
done by the normalized **hard constraint**.

## H1.8 — production positive-bootstrap gate

| Metric | Pre-H1 baseline (`c3526ec8`) | **Post-H1** |
|---|---|---|
| DSO | 36 / 36 | **36 / 36** |
| TSO | 12 / 12 | **12 / 12** |
| ESSO | 3 / 3 | **3 / 3** |
| Primary failures | 0 | **0** |
| Recovery attempts | 0 | **0** |
| **Persistent failures** | 0 | **0** |
| Iterations — total | 1 556 | **3 499** (+1 943, ×2.25) |
| Iterations — mean / median / max | 32.4 / 28.5 / 119 | **72.9 / 64.0 / 185** |
| Runtime | ~32 s | **44 s** (×1.4) |
| `σ_min(full)` DSO / TSO | 5.9246e-03 / 3.2871e-02 | **5.9246e-03 / 3.2871e-02** |
| Zero-gradient equality rows | 0 | **0** |
| Converter-capability violations | 0 / 1 728 | **0 / 1 728** |

> **The numerical cost is real and should be seen plainly: iterations roughly
> double and runtime rises ~40 %.** No new failure family appeared, no solve
> needed recovery, and the rank diagnostics are bit-identical. The cost buys an
> actually-enforced constraint — this is the solver now doing work it
> previously skipped because the row was below its resolution.

**The decisive comparison** is the *physical* residual, measured in the
original units `max((pch·pdch − eps·S²)/S², 0)` — the same quantity section E
reported:

| | Pre-H1 | **Post-H1** |
|---|---|---|
| max | 2.802e-02 | **0.000e+00** |
| mean | 2.449e-03 | **0.000e+00** |
| rows `>1e-6` | 1 062 / 1 728 | **0 / 1 728** |

The condition that was violated on 1 062 of 1 728 rows is now satisfied on all
of them. The physical requirement did not change; only its numerical
representation did.

## H1.9 — complementarity acceptance metrics

`eps = 1e-4`, so exact enforcement bounds equal-direction simultaneous
charge/discharge at `sqrt(eps) = 1e-2` of rating. Verified empirically rather
than assumed:

| Population | rows | violating | max violation | max link residual |
|---|---|---|---|---|
| Shared network ESS (DSO+TSO) | 1 728 | **0** | **0.0000e+00** | 5.42e-20 |
| Shared network ESS (DSO) | 864 | **0** | 0.0000e+00 | 1.36e-20 |
| Shared network ESS (TSO) | 864 | **0** | 0.0000e+00 | 5.42e-20 |
| ESSO per cohort | 1 728 | **0** | 0.0000e+00 | 1.74e-18 |
| ESSO aggregate | 864 | **0** | 0.0000e+00 | 6.94e-18 |

Physical circulation `min(pch, pdch)/S`:

| Population | max | mean | median | p95 | p99 | `>1e-3` | `>1e-2` | max / `sqrt(eps)` |
|---|---|---|---|---|---|---|---|---|
| Network (all) | **8.3408e-03** | 3.1066e-03 | 1.7909e-03 | 7.2638e-03 | 7.8078e-03 | 1 000 | **0** | **0.834** |
| Network (DSO) | 8.0223e-03 | 4.0397e-03 | 5.2702e-03 | 7.2436e-03 | 7.3987e-03 | 606 | **0** | 0.802 |
| Network (TSO) | 8.3408e-03 | 2.1736e-03 | 8.0705e-04 | 7.3323e-03 | 7.9831e-03 | 394 | **0** | 0.834 |
| ESSO per cohort | 7.9367e-03 | 2.1359e-03 | 5.9152e-04 | 7.5764e-03 | 7.9006e-03 | 746 | **0** | 0.794 |
| **ESSO aggregate** | **7.9349e-03** | 2.1516e-03 | 8.1174e-04 | 7.4427e-03 | 7.8864e-03 | 391 | **0** | 0.794 |

**Every population sits below the `sqrt(eps)` allowance**, and no row anywhere
exceeds `1e-2·S`. Against the E2 baseline:

| | E2 (pre-H1) | **Post-H1** | reduction |
|---|---|---|---|
| Network max `p_circ/S` | 0.11562 | **0.0083408** | **13.9×** |
| ESSO aggregate max `p_circ/S` | ~0.1912 | **0.0079349** | **24.1×** |

Artificial circulating loss:

| Quantity | E2 (pre-H1) | **Post-H1** |
|---|---|---|
| Max per period | 2.0940e-04 MWh | **1.7680e-05 MWh** |
| Worst representative day | 1.2655e-03 MWh | **3.4069e-04 MWh** |
| Worst day / `E_rated` | 2.9749e-02 | **6.0305e-03** |
| Worst day / legitimate throughput | 3.4210e-02 | **3.4303e-02** |

> The absolute loss falls ~5× and the loss-to-`E_rated` ratio ~4.9×, but the
> worst-day **share of throughput is essentially unchanged** (3.42 % → 3.43 %).
> These are different model-days: the ratio is dominated by a day whose
> legitimate throughput is itself small, so it is not a good measure of the
> improvement. The `E_rated`-normalized figure is the meaningful one.

## H1.10 — physical-tolerance decision

**Not acted on, as instructed.** `eps` was not reduced. The normalized row is
now enforced with exactly zero violation, so question 1 is answered
affirmatively; question 2 — whether `1e-4`, permitting up to 1 % circulation,
is *physically* tight enough — is separable and remains open. Observed
circulation reaches 0.83 of that allowance, so the allowance is close to
binding and a later isolated A/B at `1e-5` / `1e-6` is worth running. **That
A/B was not performed here**, to keep formulation scaling isolated from
tolerance tightening.

## H1 pass gate

| # | Criterion | Result |
|---|---|---|
| 1 | All positive-bootstrap local solves succeed | ✓ 36/36, 12/12, 3/3, 0 persistent |
| 2 | No equality-rank defect introduced | ✓ 0 zero-gradient rows, full row rank, `σ_min` unchanged |
| 3 | Network normalized complementarity solver-resolved | ✓ 0 / 1 728 violations |
| 4 | ESSO per-cohort relative semantics, solver-resolved | ✓ 0 / 1 728 violations |
| 5 | ESSO aggregate complementarity solver-resolved | ✓ 0 / 864 violations |
| 6 | `min(pch,pdch)/S` within the `sqrt(eps)` allowance | ✓ max 0.834 of allowance; 0 rows `>1e-2` |
| 7 | Converter capability clean | ✓ 0 / 1 728 |
| 8 | No end-to-end active-energy semantic inconsistency | ✓ |

**All eight criteria pass.**

---

# F — Live distributed ADMM, net-P/Q coordination only

Commit `2917b9c9`. Script: `p54f_admm_net_pq.py`. Evidence:
`data/SRP1/Results/P54F/`.

The exact positive-bootstrap candidate — built by the production
`_build_positive_bootstrap_candidate` — run through the real
`run_operational_planning(type='distributed', ...)`. Rho values and adaptation,
tolerances, proximal regularization, objective scaling, IPOPT options,
MA97/exact-Hessian policy, the recovery policy, seed and scenarios are all
unchanged. **The outer planning loop was not run.** Coordination remains net
`P`/`Q` plus the existing interface-voltage quantities — no `pch`, `pdch`,
circulation or SOC consensus was added.

## Convergence

**Converged in 9 cycles**, 312 s, every cycle with all local solves
successful and **zero recovery diagnostics recorded**.

| Cycle | solves ok | primal V | primal PF | primal ESS | dual V | dual PF | dual ESS | recourse | Δobj rel | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ✓ | 5.570e-02 | 3.919e-01 | 1.928e-02 | 9.997e-02 | 7.274e-01 | 1.471e-02 | 8.204e+08 | — | no |
| 2 | ✓ | 3.393e-02 | 1.823e-01 | 1.060e-02 | 5.011e-02 | 5.321e-01 | 6.565e-03 | 9.605e+08 | 1.459e-01 | no |
| 3 | ✓ | 1.405e-03 | 5.891e-02 | 1.047e-02 | 7.627e-02 | 3.738e-01 | 1.253e-03 | 9.647e+08 | 4.378e-03 | no |
| 4 | ✓ | 1.287e-03 | 3.163e-02 | 4.665e-03 | 2.591e-03 | 1.128e-01 | 1.919e-03 | 9.085e+08 | 5.827e-02 | no |
| 5 | ✓ | 7.181e-04 | 2.489e-02 | 2.063e-03 | 2.664e-03 | 8.687e-02 | 1.821e-03 | 8.737e+08 | 3.828e-02 | no |
| 6 | ✓ | 4.389e-04 | 2.605e-02 | 1.884e-03 | 9.902e-04 | 8.391e-02 | 1.135e-03 | 8.562e+08 | 2.005e-02 | no |
| 7 | ✓ | 2.337e-04 | 1.406e-02 | 1.072e-03 | 7.037e-04 | 5.936e-02 | 9.152e-04 | 8.503e+08 | 6.944e-03 | no |
| 8 | ✓ | 1.686e-04 | 1.102e-02 | 2.395e-03 | 6.816e-04 | 4.881e-02 | 9.902e-04 | 8.486e+08 | 1.917e-03 | no |
| **9** | ✓ | **1.365e-04** | **9.776e-03** | **1.269e-03** | **7.187e-04** | **4.834e-02** | **1.053e-03** | **8.479e+08** | **8.132e-04** | **yes** |

Tolerances (unchanged): primal V `1e-2`, primal PF `1e-2`, primal ESS `1e-1`;
dual V / PF / ESS `1e-2`; objective relative `1e-3`. At cycle 9
`residual_convergence`, `objective_convergence` and `cycle_convergence` are all
true, with `consecutive_converged_cycles = 1` meeting the required 1.

Final `gross_operational_cost = 8.4794e+08`, `terminal_salvage_value =
3.5415e+03`, `recourse = 8.4794e+08`.

Rho evolution (standard adaptation, no intervention): V `1 → 1.5 → 2.25 → 1.5`
then held; PF `1 → 1.5 → 2.25` then held; ESS held at `1` throughout.

## Complementarity sanity diagnostics — not consensus variables

Read off the converged models; never fed into a consensus update.

| Population | rows | violating | max violation | max `p_circ/S` | mean | median | p95 | p99 | `>1e-3` | `>1e-2` | max / `sqrt(eps)` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Network (all) | 1 728 | **0** | 0.0000e+00 | **8.4125e-03** | 1.0126e-03 | 6.8286e-04 | 3.3631e-03 | 5.2060e-03 | 526 | **0** | **0.841** |
| Network (DSO) | 864 | **0** | 0.0000e+00 | 8.4091e-03 | 1.0068e-03 | 6.8180e-04 | 3.2897e-03 | 5.1662e-03 | 263 | **0** | 0.841 |
| Network (TSO) | 864 | **0** | 0.0000e+00 | 8.4125e-03 | 1.0184e-03 | 6.8325e-04 | 3.4264e-03 | 5.1847e-03 | 263 | **0** | 0.841 |
| **ESSO aggregate** | 864 | **0** | 0.0000e+00 | **7.6320e-03** | 9.2889e-04 | 6.1588e-04 | 3.0431e-03 | 4.5683e-03 | 239 | **0** | 0.763 |

Complementarity survives coordination intact: still **zero violations**, still
**no row above `1e-2·S`**, and the mean circulation is in fact ~3× lower than
in the uncoordinated bootstrap.

## Energy-consistency sanity diagnostic

At final electrical consensus, `p_cell = eta_ch·pch − pdch/eta_dch` compared
between DSO and TSO for the same shared ESS, over 288 matched
year/day/period points:

| Quantity | Value |
|---|---|
| max abs `p_cell` difference | **3.0293e-06 p.u.** |
| mean abs `p_cell` difference | 5.6992e-07 p.u. |
| max abs `pnet` difference | 2.9567e-06 p.u. |

The `p_cell` disagreement tracks the `pnet` disagreement almost exactly and
both are at solver-tolerance scale. **This is the direct confirmation of the
coordination decision**: with complementarity locally enforced, agreeing on net
active power is sufficient to make the agents agree on charge/discharge
direction and on cell-side energy rate. No `pch`/`pdch` consensus is needed,
and none was added.

---

# D2 — Shared-S/E sensitivity root-cause audit

Commit `65b261ba`. Scripts: `p54d2_sign_calibration.py`,
`p54d2_sensitivity_root_cause.py`, `p54d2_branch_controlled_fd.py`,
`p54d2_continuation_sweep.py`, `p54d2_sensitivity_clean_ab.py`,
`p54d2_binding_regime_scan.py`. Evidence: `data/SRP1/Results/P54D2/`.

**Diagnostic only.** No production formulation, Benders equation, solver option
or tolerance was changed. The outer planning loop was not run.

**Two independent root causes were found.** One has a concrete, provable
production fix; the other does not, and it is what makes the original
finite-difference test unable to settle the question either way.

## D2.1 — the sensitivity contract, traced end to end

| Step | Code | Quantity |
|---|---|---|
| 1 | `sess_s_sensitivities` / `sess_e_sensitivities` (`model_construction_helpers.py`) | rows `shared_es_s_rated_fixed[e] == shared_es_s_rated[e]`, likewise for E |
| 2 | `network.py` | `model.dual` is an `IMPORT_EXPORT` Suffix, so IPOPT returns the row multiplier |
| 3 | `_get_sensitivities` (`network_data.py:2806`) | `sensitivity = objective_scale * dual / baseMVA`, then `+= annualization * num_years * num_days * sensitivity`, accumulated over days and years |
| 4 | `_get_operational_sensitivities` (`shared_resources_planning.py:1004`) | TSO and every DSO summed per node/year into one **available-capacity** sensitivity |
| 5 | `map_available_capacity_sensitivities_to_investments` (ESSO) | available-capacity → **investment-variable** sensitivity |
| 6 | `get_salvage_value_sensitivities` | salvage term added |
| 7 | `_add_benders_cut` (`shared_resources_planning.py:2026`) | `alpha >= Q + Σ sens_s·(S_inv − S_inv0) + Σ sens_e·(E_inv − E_inv0)` |

**What the outer algorithm assumes each returned dual is:** the complete
first-order derivative `∂Q/∂S` (resp. `∂Q/∂E`) of the local recourse value with
respect to installed capacity — used as the linear coefficient of a local cut.
The cut is only valid to the extent that coefficient is the true derivative.

### Intended signs, derived rather than observed

Every shared-ESS constraint that involves `S` is a relaxation as `S` grows:
`pch + pdch ≤ S`; `pnet² + qnet² ≤ S²`; the box `pch, pdch ∈ [0, S]`,
`pnet, qnet ∈ [−S, S]`; and complementarity, since `pch_hat·pdch_hat ≤ eps` is
`pch·pdch ≤ eps·S²`. The SOC recursion and limits do not involve `S` at all.
**The feasible set is therefore monotonically non-decreasing in `S`, so the
minimum is non-increasing:**

```
∂Q/∂S ≤ 0     (required, by set monotonicity — not inferred from data)
```

For `E` no such theorem holds: the SOC band `[0.1·E, 0.9·E]` widens with `E`,
but the initial and target SOC anchor `0.5·E` **moves** with it, so an increase
in `E` is not a pure relaxation. `∂Q/∂E ≤ 0` is expected but not guaranteed.

## D2.4a — multiplier sign convention, measured

Asserted nowhere in the repo, so it was measured on three trivial parametric
NLPs with known analytic derivatives, solved with the **production IPOPT**
(`/usr/local/bin/ipopt`, resolved from the production params object):

| Problem | Analytic | Suffix | Ratio |
|---|---|---|---|
| `min (x−5)²  s.t. x == c`, orientation `param == var` | `2(c−5)` | `dual` | **+1.000000** (4/4 points) |
| `min (x−5)²  s.t. 0 ≤ x ≤ u` | `2(u−5)` | `ipopt_zU_out` | **+1.000000** (3/3) |
| `min (x+5)²  s.t. l ≤ x ≤ 10` | `2(l+5)` | `ipopt_zL_out` | **+1.000000** (3/3) |

```
dual[param == var] = +dQ/dparam ,    dQ/du = +zU ,    dQ/dl = +zL
```

so the complete parametric derivative is

```
dQ/dθ = dual[fixing row]  +  Σ_bounds ( zU·du/dθ + zL·dl/dθ )  +  direct-expression terms
```

while the Benders extraction reads **only the first term**.

## D2.2 / D2.3 — dependence inventory and bound audit

### S

| Dependence | Class | Detail |
|---|---|---|
| `sess_converter_capability`: `pnet² + qnet² ≤ shared_es_s_rated²` | **A** symbolic | via the rated **Var** |
| `sess_active_sum_limit`: `pch + pdch ≤ shared_es_s_rated` | **A** symbolic | via the rated Var |
| `sess_pch_hat_link` / `sess_pdch_hat_link`: `pch − S_rated·pch_hat == 0` | **A** symbolic | H1 links, via the rated Var |
| `sess_comp`: `pch_hat·pdch_hat ≤ eps` | **A** symbolic | S enters only through the links |
| `shared_es_pch`, `shared_es_pdch` bounds `[0·S, 1·S]` | **C** numeric bound | `_SHARED_ESS_RATED_BOUNDED_VARIABLES` |
| `shared_es_pnet`, `shared_es_qnet` bounds `[−1·S, 1·S]` | **C** numeric bound | same |
| fix/unfix + row (de)activation at the zero-capacity threshold | **D** gating | `configure_shared_ess_operational_state` |
| any other active row referencing `shared_es_s_rated_fixed` | **B** — **none** | verified numerically: **0 active rows** outside the fixing row |

### E

| Dependence | Class | Detail |
|---|---|---|
| `sess_soc_limit_lower`: `soc ≥ shared_es_e_rated · 0.1` | **A** symbolic | |
| `sess_soc_limit_upper`: `soc ≤ shared_es_e_rated · 0.9` | **A** symbolic | verified body: `shared_es_soc[0,0,0,0] <= 0.9*shared_es_e_rated[0]` |
| `sess_soc_def` at `p = 0`, `sess_soc_final` | **A** symbolic | anchored at `shared_es_e_rated · 0.5` |
| SOC initial **value** `0.5·E` | **D** start point | affects which local optimum is reached, not the feasible set |
| any E-scaled variable bound | **C — none** | `shared_es_soc` bounds are `[0, None]`; both SOC slacks `[0, None]` |
| any active row referencing `shared_es_e_rated_fixed` | **B** — **none** | verified numerically |

> **The E path is already sensitivity-clean; the S path is not.** That asymmetry
> is the audit's central structural finding, and it predicts exactly what is
> observed below.

### Measured bound multipliers

`shared_es_pch`, `shared_es_pdch`, `shared_es_pnet` and `shared_es_qnet` all
carry live `zL`/`zU` at the solved point. The resulting bound contribution to
`dQ/dS` is **negative in 40 / 40** scanned points and **8 / 8** audited cases —
i.e. always the sign set monotonicity requires — and it obeys a clean structural
law:

| Case | S ×1 | ×5 | ×20 | ×100 |
|---|---|---|---|---|
| `dso/9/2030/Winter`, `bound × S` | −1.3123e-03 | −1.3102e-03 | −1.3094e-03 | −1.3091e-03 |
| `tso/0/2025/Winter`, `bound × S` | −1.4739e-02 | −6.6690e-03 | −6.5726e-03 | −6.5506e-03 |

`bound_contribution · S` is invariant to four significant figures across a 100×
capacity range (the TSO's first point differs because S is *binding* there).
This is a systematic term, not an artefact.

## D2.4b — envelope decomposition (root cause 1)

Per case, at the bootstrap capacity:

| Case | fixing-row dual | bound contribution | corrected total | fixing share of total |
|---|---|---|---|---|
| dso/5/2025/Winter | **+1.5009e+01** | −1.2607e+01 | +2.4016e+00 | **6.250** |
| dso/5/2030/Summer | **+1.0559e+01** | −6.2427e+00 | +4.3160e+00 | **2.446** |
| dso/7/2025/Spring | **+1.5833e+01** | −1.2625e+01 | +3.2083e+00 | **4.935** |
| dso/7/2030/Winter | −3.2408e+00 | −6.1698e+00 | −9.4106e+00 | 0.344 |
| dso/9/2030/Winter | −3.2400e+00 | −6.1698e+00 | −9.4098e+00 | 0.344 |
| dso/9/2035/Summer | −1.5701e+00 | −4.1119e+00 | −5.6820e+00 | 0.276 |
| tso/0/2025/Winter | −1.1871e+02 | −1.3859e+02 | −2.5730e+02 | 0.461 |
| tso/1/2030/Summer | −5.3083e+01 | −3.1660e+01 | −8.4742e+01 | 0.626 |

> **Root cause 1 — the Benders coefficient is structurally incomplete for S.**
> The quantity Benders extracts is between **28 % and 625 %** of the corrected
> envelope derivative. In three of eight cases it is **positive**, which by the
> D2.1 monotonicity argument is provably not `∂Q/∂S`.

For **E**, the bound contribution is **exactly 0.0 in every one of the 40
scanned points**, and the fixing-row dual is **negative in 40 / 40** — the
structurally complete, sign-consistent behaviour the inventory predicts.

## D2.5 / D2.9 — finite differences, and why they cannot settle this (root cause 2)

### Cold-start FD reproduces the D finding and explains it

Objectives from the cold-start FD cluster into a few discrete values whose
separation is **independent of the step size**. At `rel = 0.001`
(`h = 2.1e-07`) `ΔQ` is still `5.3e-03`, while the corrected derivative predicts
a capacity effect of `~2e-06` — three orders of magnitude smaller.

### It is not solver noise

Solving the **identical unperturbed problem** five times from the production
cold start gives **spread = 0.0000e+00** — bit-identical objectives and one
active set. The solver is deterministic; the variation is a genuine,
reproducible dependence of *which local optimum is selected* on the capacity
value.

### Continuation sweep — the decisive evidence

A ±5 % sweep in 0.5 % steps, each solve seeded from the previous one:

| `S/S₀` | `Q` | fixing-row dual |
|---|---|---|
| 0.950 | −8.017376754515e-01 | −3.5536e+00 |
| 0.955 | −7.991076137578e-01 | **+1.1040e+01** |
| 0.960 | −8.017253267607e-01 | −3.4621e+00 |
| 0.965 | −8.017234421550e-01 | −3.4385e+00 |
| 0.970 | −7.979574629569e-01 | **+1.6369e+01** |
| 0.975 | −7.990532618936e-01 | +1.0579e+01 |
| 0.980 | −7.990470262602e-01 | +1.0551e+01 |
| 0.985 | −7.990408541574e-01 | +1.0522e+01 |
| … | … | … |
| 1.050 | −7.991486325795e-01 | +9.4612e+00 |

Across the sweep `Q` takes **~16 distinct values on at least three discrete
branches**, spanning **4.0e-03**, while the entire capacity effect across the
whole ±5 % window is `≈ |dQ/dS|·0.1·S ≈ 2e-04` — the branch gap is **~20×
larger than the effect being measured**.

> **Root cause 2 — the realized local value function is multi-valued in S and E
> at bootstrap capacities.** The fixing-row dual behaves as a **branch label**:
> it is tightly correlated with which Q level the solve landed on
> (`Q ≈ −0.80172 → dual ≈ −3.4`; `Q ≈ −0.79905 → dual ≈ +10.5`;
> `Q ≈ −0.79786 → dual ≈ +16.1`) and varies smoothly *within* a branch
> (`+10.579 → +10.551 → +10.522` at 0.975/0.980/0.985).

Consequences, stated plainly:

- **Finite differences cannot validate either prediction in this regime.** No
  perturbation window with a stable branch and a resolvable capacity effect was
  found. This is why the P5.4-D result could neither confirm nor refute the
  dual.
- Seeding perturbed solves from the base solution does **not** rescue it: the
  seeded solves converge to a *different* branch (`Q ≈ −0.799075`) than the
  cold-started base (`Q ≈ −0.801702`), so the reference dual and the differences
  then belong to different branches.
- The non-monotone `Q(S)` in the table above does **not** contradict D2.1: set
  monotonicity constrains the global minimum, and the solver returns a local
  one.

## D2.6 / D2.8 — sensitivity-clean reformulation A/B

### Redundancy proof

| Capacity-dependent bound | Implied by |
|---|---|
| `pch ≤ S` | `pch + pdch ≤ S` with `pdch ≥ 0`; independently by `pch = S·pch_hat`, `pch_hat ≤ 1` |
| `pdch ≤ S` | `pch + pdch ≤ S` with `pch ≥ 0`; independently by the H1 link |
| `−S ≤ pnet ≤ S` | `pnet² + qnet² ≤ S²` |
| `−S ≤ qnet ≤ S` | `pnet² + qnet² ≤ S²` |

All four are redundant, so removing them **does not change the feasible set**.

- **A** — production: bounds retained.
- **B** — sensitivity-clean: only those four relaxed. Retained unchanged:
  `pch, pdch ≥ 0`; the H1 hat bounds `[0, 1]`; `sess_active_sum_limit`;
  `sess_converter_capability`; every SOC/energy row; the zero-capacity gating.
  No capacity-independent safety bound was needed — the symbolic rows and the
  H1 links keep every variable bounded.

### Result

| | A | B |
|---|---|---|
| Bound contribution eliminated | — | **8 / 8 cases, exactly 0.0** |
| Objective relative difference | — | up to 6.8e-03 |
| Same ESS active set as A | — | **0 / 8** |
| max \|Δpch\| | — | up to 4.6e-05 (≈ 21 % of rating) |

> **B does exactly what it was designed to do — it makes the fixing-row dual the
> structurally complete envelope derivative.** But **D2.8's equivalence
> requirement is not met numerically in 6 of 8 cases**: A and B land on
> different local optima. Two cases (`dso/9/2035/Summer`, `tso/0/2025/Winter`)
> do agree closely (relative difference 4.5e-06 and 2.9e-05, `max|Δpch|` ~1e-08
> and ~1e-06). The equivalence is provable analytically — identical feasible
> sets — but relaxing the bounds changes the interior-point path, so root cause
> 2 reappears. **This is not evidence against B**; it is the same multiplicity
> defeating the comparison.

## D2.7 — E analogue

No change is proposed or needed. Verified on the built model: `shared_es_soc`
bounds are `[0, None]`, both SOC slacks are `[0, None]`, the SOC limits are
symbolic rows against `shared_es_e_rated`, and the E bound contribution is
exactly `0.0` at every one of the 40 scanned points. SOC fractions, day balance,
efficiencies and energy semantics were not touched.

## D2.10 — broader population, and what could not be covered

Covered: DSO nodes **5, 7 and 9**, a **TSO** case, years **2025 / 2030 / 2035**
and days **Winter / Spring / Summer** — 8 cases, plus a 40-point capacity scan
across two cases at `S ×{1, 5, 20, 100}` and `E ×{1, 0.5, 0.2, 0.05, 0.02}`.

| Requirement | Status |
|---|---|
| DSO node 5 / 7 / 9, TSO, multiple years and days | **met** |
| ≥ 1 case where **S** is operationally binding | **met** — `tso/0/2025/Winter` (active-sum slack 2.2e-06) |
| ≥ 1 case where **E** is operationally binding | **NOT met** |

**No E-binding case exists anywhere in the scan.** The SOC upper slack shrinks
with `E` but never reaches zero, down to `E ×0.02`: in this reduced
operational-only configuration the shared ESS is barely cycled, so the energy
limit is never reached. Rather than manufacture one, this is reported as a gap.
The E-side sensitivity conclusion therefore rests on the structural argument
(no E-dependent bound exists, so the fixing-row dual is complete by
construction) plus 40/40 sign-consistent duals — **not** on an E-binding test.

## D2 production decision

**No Benders equation was modified**, as instructed.

**Recommended, for planner approval — productionize the D2.6 variant B local
formulation correction.** Remove the four redundant capacity-dependent
numerical bounds on `shared_es_pch`, `shared_es_pdch`, `shared_es_pnet` and
`shared_es_qnet` from `_SHARED_ESS_RATED_BOUNDED_VARIABLES`, keeping
nonnegativity, the H1 hat bounds and every symbolic row. This is an exact
reformulation and it makes the quantity Benders already reads the structurally
complete derivative, with no change to Benders itself.

**If instead the bounds are kept**, Benders would need the additional term

```
Σ over pch, pdch entries:  zU
Σ over pnet, qnet entries: (zU − zL)
```

scaled by the same `objective_scale / baseMVA · annualization · num_years ·
num_days` weighting as the fixing-row dual. Variant B is preferable: it removes
the term rather than requiring every consumer to remember it.

**Neither fix resolves root cause 2.** Remaining cause, per the plan's list:
**local-optimum switching**, and specifically local-solution multiplicity — not
insufficient solve accuracy (repeated identical solves agree bit-for-bit), not
complementarity nonsmoothness (H1 leaves zero violation and the branch gap
appears in the objective, not the ESS active set), and not another hidden
parameter dependence (category B is empty for both S and E). **No IPOPT option
was tuned to force agreement.**

## D2 verdict

```
P5.4-D2 PARTIAL — sensitivity root cause identified but production correction still required
```

---

# D2-P — Sensitivity-clean shared-S productionization

Commit `06e921e5`. Script: `p54d2p_validation.py`. Evidence:
`data/SRP1/Results/P54D2P/p54d2p_report.json`.

Productionizes D2.6 variant B. `_SHARED_ESS_RATED_BOUNDED_VARIABLES` — which
rewrote four variable bounds from the fixed capacity parameter — is replaced by
`_SHARED_ESS_ZERO_GATED_BOUND_VARIABLES`, which holds the **capacity-independent**
bounds retained at positive capacity:

| Variable | Before (S-dependent) | After |
|---|---|---|
| `shared_es_pch` | `[0·S, 1·S]` | `[0, None]` — nonnegativity only |
| `shared_es_pdch` | `[0·S, 1·S]` | `[0, None]` |
| `shared_es_pnet` | `[−1·S, 1·S]` | `[None, None]` |
| `shared_es_qnet` | `[−1·S, 1·S]` | `[None, None]` |

The zero-capacity branch is unchanged: the box still collapses to `[0, 0]` and
the variables are still explicitly fixed at `0`. **Benders equations were not
touched and no bound-multiplier term was added to them.**

Retained exactly: `pch, pdch ≥ 0`; `pch_hat, pdch_hat ∈ [0, 1]`;
`pch = S·pch_hat`; `pdch = S·pdch_hat`; `pch + pdch ≤ S`; `pnet = pch − pdch`;
`pnet² + qnet² ≤ S²`; the active-energy SOC rows; H1 normalized
complementarity; PF/reactive rows; every objective term and coefficient; and
the zero-capacity fixing/gating. Nothing divides by S.

## D2-P.1 — exact redundancy

Analytically:

| Removed bound | Implied by |
|---|---|
| `pch ≤ S` | `pch, pdch ≥ 0` with `pch + pdch ≤ S`; independently `pch = S·pch_hat`, `pch_hat ≤ 1` |
| `pdch ≤ S` | symmetric |
| `\|pnet\| ≤ S` | `pnet² + qnet² ≤ S²` for `S > 0` |
| `\|qnet\| ≤ S` | symmetric |

Numerically, at the solved point of all eight cases — **every implied bound
holds, 8/8**, with no negative `pch`/`pdch` anywhere. The worst observed ratios
show the symbolic rows doing the work rather than being slack by accident:

| Case | `max pch/S` | `max pdch/S` | `max \|pnet\|/S` | `max \|qnet\|/S` |
|---|---|---|---|---|
| dso/9/2030/Winter | 0.2199 | 0.2092 | 0.2197 | 0.0000 |
| dso/9/2035/Summer | 0.0071 | 0.0065 | 0.0013 | 0.0000 |
| **tso/0/2025/Winter** | **0.9895** | 0.4669 | **0.9894** | 0.2401 |
| **tso/1/2030/Summer** | 0.2546 | **0.9950** | **0.9949** | 0.0209 |

The two TSO cases run to within 0.5–1 % of the removed bound and still satisfy
it — the capability circle and the active-sum row hold it, exactly as the
redundancy argument requires.

## D2-P.2 — lifecycle on one reused model

| Transition | S | `pch` bounds | `pnet` bounds | fixed | unbounded & unfixed |
|---|---|---|---|---|---|
| zero | 0.0 | `[0.0, 0.0]` | `[0.0, 0.0]` | **yes** | 0 |
| → positive | 2.1270e-04 | `[0.0, None]` | `[None, None]` | no | 48 |
| → different positive | 5.3174e-04 | `[0.0, None]` | `[None, None]` | no | 48 |
| → back to zero | 0.0 | `[0.0, 0.0]` | `[0.0, 0.0]` | **yes** | 0 |
| → positive again | 2.1270e-04 | `[0.0, None]` | `[None, None]` | no | 48 |

All checks pass: component ids constant, zero-capacity variables fixed at 0 with
the box collapsed and rows deactivated, positive-capacity variables free with no
S-dependent bound, hat bounds unchanged at `[0, 1]`, warm-start suffixes intact.

The 48 unbounded-and-unfixed entries at positive capacity are `pnet` and `qnet`
(24 each); they are bounded by `sess_converter_capability`, which D2-P.1 confirms
holds at the solution.

## D2-P.3 — sensitivity structure

Across the eight original D2 cases, with bound S-dependence now **measured** from
the model (configure at S and at `S·(1+1e-3)` and difference the bounds) rather
than read from a factor table:

| Check | Result |
|---|---|
| Any positive-capacity bound depends on S | **False, 8/8** |
| Bound contribution to `dQ/dS` | **exactly 0.0, 8/8** |
| Active rows referencing `shared_es_s_rated_fixed` outside the fixing row | **0, 8/8** |
| Fixing-row dual is the complete local envelope derivative | **True, 8/8** |
| E bound contribution | **exactly 0.0, 8/8** — unchanged, still clean |

The pre-D2-P bound term, which ranged from 28 % to 625 % of the corrected
derivative, is gone. **This is a statement about the local branch derivative
only** — it does not by itself establish global Benders validity, which is what
D3 examines.

## D2-P.4 — operational regression

| Metric | H1 baseline | **Post-D2-P** |
|---|---|---|
| DSO | 36 / 36 | **36 / 36** |
| TSO | 12 / 12 | **12 / 12** |
| ESSO | 3 / 3 | **3 / 3** |
| Primary failures / recoveries / persistent | 0 / 0 / 0 | **0 / 0 / 0** |
| Iterations — total | 3 499 | **3 442** |
| Iterations — mean / median / max | 72.9 / 64.0 / 185 | **71.7 / 64.5 / 137** |
| Runtime | 44 s | **43 s** |
| σ_min(full) DSO / TSO | 5.9246e-03 / 3.2871e-02 | **5.9246e-03 / 3.2871e-02** |
| Zero-gradient equality rows | 0 | **0** |
| Converter-capability violations | 0 / 1 728 | **0 / 1 728** |
| Complementarity violations (physical form) | 0 / 1 728 | **0 / 1 728** |
| `max min(pch,pdch)/S` | 8.3408e-03 | **8.2482e-03** |

Distributed ADMM, same fixed candidate as P5.4-F:

| Metric | P5.4-F | **Post-D2-P** |
|---|---|---|
| Converged | yes | **yes** |
| Cycles | 9 | **9** |
| All local solves ok, every cycle | yes | **yes** |
| Recovery diagnostics | 0 | **0** |
| Final rho (V / PF / ESS) | 1.5 / 2.25 / 1.0 | **1.5 / 2.25 / 1.0** |
| Complementarity violations | 0 | **0** |
| Converter capability | clean | **clean** |
| Energy consistency `max\|ΔP_cell\|` | 3.03e-06 | **2.51e-06** |
| `pch/pdch` consensus added | no | **no** |

**No regression anywhere; iterations are marginally better.**

## D2-P verdict

```
P5.4-D2-P PASS — structurally complete S sensitivity productionized
```

---

# D3 — Distributed cut-consistency / local-branch audit

Commits `926bd5df`, `e1afa8e9`. Scripts: `p54d3_cut_consistency.py`,
`p54d3_analysis.py`. Evidence: `data/SRP1/Results/P54D3/`.

The cut was **constructed but never added to a master problem**, and
`run_planning_problem()` was never invoked.

## D3.1 / D3.2 — base recourse and the production coefficient

`Q0 = 848 258 809.814117` (gross operational cost `848 262 234.99`, terminal
salvage `3 425.17`), reached in **9 cycles**, all local solves successful, zero
recovery diagnostics.

`g0` is the `sensitivities` object returned by
`run_operational_planning(type='distributed', …)` — the exact vector
`_add_benders_cut` consumes, after local rescaling by
`objective_scale / baseMVA`, weighting by `annualization · num_years ·
num_days`, TSO+DSO aggregation, available-capacity → investment-capacity
mapping and the salvage correction. **Not** an isolated raw local dual.

18 coefficients, none `None`. **All 18 are negative** — 9/9 for S and 9/9 for E:

| | node 5 | node 7 | node 9 |
|---|---|---|---|
| **S** 2025 | −1.3034e+07 | −1.3080e+07 | −1.3234e+07 |
| **S** 2030 | −6.3921e+06 | −6.4213e+06 | −6.5959e+06 |
| **S** 2035 | −2.5570e+06 | −2.5792e+06 | −2.7612e+06 |
| **E** 2025 | −6.9121e+06 | −6.8496e+06 | −7.1198e+06 |
| **E** 2030 | −3.9631e+06 | −3.8846e+06 | −4.0446e+06 |
| **E** 2035 | −2.1239e+06 | −1.9958e+06 | −2.0128e+06 |

Sign-consistent with the D2.1 monotonicity requirement throughout — a visible
improvement on the pre-D2-P local duals, which were positive in 3 of 8 cases.
`L(x) = Q0 + g0ᵀ(x − x0)`.

## `tol_cut`, derived rather than chosen

| Component | Value |
|---|---|
| Identical-candidate repeatability of the distributed recourse | **0.0 exactly** — three independent processes reproduced `Q0` bit-for-bit |
| Residual relative objective drift at the cycle where ADMM declares convergence | 3.591e-04 |
| `tol_cut = drift × \|Q0\|` | **3.046e+05** (3.591e-04 relative) |

The recourse is perfectly reproducible, so the binding term is not randomness
but how precisely the converged recourse is *determined* — the ADMM stops while
the objective is still moving by ~3.6e-04 relative.

## D3.3 / D3.4 — candidate population actually covered

**8 candidates completed**, each a full production distributed solve:

| Group | Perturbations run |
|---|---|
| S at node 9, 2025 | −10 %, −5 % (both also from start B) |
| S at node 5, 2025 | −10 %, −5 % |
| E at node 9, 2025 | −10 %, −5 %, −2 %, −1 % |

**Not covered:** positive perturbations, node 7, the ±0.5 %/±1 %/±2 % refinement
for S, and the multi-variable set. The sweeps were stopped once the violation
below had been established from two independent starts — each candidate costs
10–20 minutes of ADMM — so this is a **partial population**, and it is reported
as such. Every completed candidate converged with all local solves successful
and **zero recovery diagnostics**.

## D3.6 — cut safety: decisive violation

| Candidate | `Q_best_observed` | `L(x)` | observed ΔQ | predicted `g0ᵀΔx` | **cut_gap** |
|---|---|---|---|---|---|
| **s\|node9\|2025 −5 %** | 836 789 546.53 | 848 265 846.83 | −11 469 263 | **+7 037** | **−1.1476e+07** |
| **s\|node9\|2025 −10 %** | 836 959 457.84 | 848 272 883.85 | −11 299 352 | +14 074 | **−1.1313e+07** |
| **s\|node5\|2025 −5 %** | 837 398 111.50 | 848 265 740.63 | −10 860 698 | +6 931 | **−1.0868e+07** |
| **e\|node9\|2025 −1 %** | 839 303 246.22 | 848 260 324.18 | −8 955 564 | +1 514 | **−8.9571e+06** |
| **e\|node9\|2025 −10 %** | 847 841 612.43 | 848 273 953.48 | −417 197 | +15 144 | **−4.3234e+05** |
| **e\|node9\|2025 −2 %** | 847 842 640.65 | 848 261 838.55 | −416 169 | +3 029 | **−4.1920e+05** |
| s\|node5\|2025 −10 % | 848 241 371.88 | 848 272 671.45 | −17 438 | +13 862 | −3.1300e+04 |
| e\|node9\|2025 −5 % | 848 069 916.76 | 848 266 381.65 | −188 893 | +7 572 | −1.9646e+05 |

**8 / 8 negative; 6 decisive** (`cut_gap < −tol_cut`). Worst: **−1.1476e+07**,
i.e. **−1.35 % of `Q0`**, 38× `tol_cut`.

> ### Why this is decisive rather than a numerical artefact
>
> Reducing shared-ESS capacity **shrinks** the operational feasible set — the
> only rows that change are `pch + pdch ≤ S`, `pnet² + qnet² ≤ S²` and the H1
> links, all of which tighten. So any operating point feasible at `S − 5 %` is
> also feasible at `S₀`.
>
> At `s|node9 −5 %` the distributed solve reaches a recourse of
> **836 789 546.53**, which is **11 469 263 below `Q0`**. That operating point
> was therefore available at the base candidate and was not found.
>
> **`Q0` overstates the achievable recourse at its own candidate by at least
> 1.35 %**, so any affine cut anchored at `Q0` lies above the true recourse
> function and would cut off feasible master solutions.
>
> The violating runs are sound: converged, 19 and 12 cycles, **all local solves
> ok, zero recoveries**, final residuals inside tolerance
> (`primal_v` 3.1e-05, `primal_pf` 6.3e-03 < 1e-02, `primal_ess` 2.4e-04). The
> `s|node9 −5 %` result is reproduced from **two independent starts** —
> production cold start gives 844 565 756.42, continuation gives
> 836 789 546.53 — both far below `Q0`.

## D3.7 — local linearity and branch classification

| Group | n | max abs error | median abs error | max rel error | median rel error |
|---|---|---|---|---|---|
| Same branch as base | **0** | — | — | — | — |
| Different branch | **8** | 1.1476e+07 | 4.6947e+06 | 1.795 | 1.004 |

**No candidate stayed on the base branch** — 8 distinct recourse levels across 8
candidates. The median relative error of 1.004 means the linear model is, on
average, wrong by about the entire observed change. This is branch selection
dominating, not derivative quality: D2-P already established the coefficient is
the structurally complete derivative *of its own branch*.

### The coefficient is below the method's resolution

| Quantity | Value |
|---|---|
| `max \|g0ᵀΔx\|` over the 8 candidates | 1.5144e+04 |
| `tol_cut` | 3.046e+05 |
| ratio | **0.0497** |
| predictions below `tol_cut` | **8 / 8** |

Even with no violations, the cut's predicted effect is **20× smaller than the
precision to which the recourse is determined**. The coefficient could not be
confirmed against observed recourse at these perturbation sizes.

## D3.8 — S monotonicity of the observed lower envelope

Observed best recourse at node 9 (including the base point):

| `S/S₀` | `Q_best_observed` |
|---|---|
| 0.90 | 836 959 457.84 |
| 0.95 | **836 789 546.53** |
| 1.00 (base) | **848 258 809.81** |

Increasing S from 0.95 to 1.00 **increases** the observed best recourse by
11 469 263, though the feasible set grew. Node 5 shows the same pattern
(observed range 1.0843e+07 across the sweep).

Per the plan's instruction, this is **not** reinterpreted as physical
non-monotonicity: it states that **the lower branch has not been reliably
recovered at the base candidate**. Note also that the completed sweep is
one-sided (negative perturbations only), so the two-sided envelope test is
incomplete.

## D3.9 — classification

Classification **C — unsafe**. At least one generated cut lies above a lower
observed recourse value by far more than numerical tolerance, reproduced from
independent starts, at multiple nodes and for both S and E.

**Benders was not modified**, as instructed. Smallest next planning-level
remedies, offered for planner review and **not implemented**:

1. **Multi-start recourse selection** — the most direct fix for the observed
   defect. The cut's anchor `Q0` is the problem; evaluating each candidate from
   several deterministic starts and cutting from the best observed recourse
   would have removed every violation here. Cost: a multiple of the operational
   solve time per candidate.
2. **Deterministic branch-continuation policy** — always warm-start a candidate
   from the incumbent's converged state, so consecutive candidates stay on one
   branch. Cheaper than (1), and it makes the local derivative meaningful, but
   it fixes the branch rather than finding the lower one.
3. **Trust-region / locally-activated cuts** — keep cuts active only near their
   generating candidate. This limits the damage but does not make the anchor
   correct, and the violations here occur within ±10 %.
4. **Planning-capacity movement regularization** — smaller master steps, same
   caveat as (3).
5. **A decomposition framework appropriate for nonconvex recourse** — the
   honest structural answer if (1) proves too expensive.

**Classical convex Benders guarantees do not hold here**, and nothing in D2-P
changes that: a structurally complete derivative of a local branch is still a
derivative of a local branch. D2-P was necessary — the coefficient is now
sign-consistent and complete — but it is not sufficient.

## D3 verdict

```
P5.4-D3 FAIL — current cuts are demonstrably unsafe under observed recourse branches
```

---

# H — Status after H1

The H question has now split cleanly in two, and H1 answered the first half.

- **Converter capability: not a problem, confirmed again.** Zero violation on
  all 1 728 shared-ESS rows, all 48 ordinary-ESS rows and the ESSO, before and
  after H1. It was correctly left un-normalized.
- **Complementarity, numerical enforceability: RESOLVED.** The `eps·S²` row was
  below IPOPT's resolution at bootstrap capacity; written at O(1) it is
  enforced with exactly zero violation across every population, and the
  *physical* residual in the original units went from violated on 1 062 / 1 728
  rows to **0 / 1 728**.
- **Complementarity, physical sufficiency: still open (H1.10).** With
  `eps = 1e-4` the formulation permits up to `sqrt(eps) = 1 %` of rating
  circulating, and observed circulation reaches **0.83 of that allowance**. So
  the tolerance is close to binding, and whether 1 % is physically acceptable
  is a separate judgement. **No epsilon change was made**, deliberately, to
  keep formulation scaling isolated from tolerance tightening.

The recommendation is a later isolated A/B at `eps = 1e-5` and `1e-6`, run on
its own, with the H1 formulation held fixed. That was not performed here.

---

# Open items carried forward

1. **The current Benders cuts are demonstrably unsafe** (D3). Six of eight
   probed candidates produce a cut lying above an observed feasible recourse,
   the worst by 1.35 % of `Q0`. The cause is the anchor `Q0`, not the
   coefficient: the base candidate's own solve misses a recourse ~11.5 million
   lower that neighbouring candidates find. **This blocks P5.4-G**, and the
   smallest candidate remedies are listed in D3.9 for planner decision.
2. **The cut coefficient is below the method's own resolution** (D3.7): the
   largest predicted effect over the probed range is 0.05 x `tol_cut`, so the
   coefficient cannot be validated against observed recourse at these
   perturbation sizes even where no violation occurs.
3. **D3's candidate population is partial** — positive perturbations, node 7,
   the finer S steps and the multi-variable set were not run, since the
   violation was already decisive. Stated so the coverage is not overread.
4. **No E-binding operating point exists** in the scanned configuration
   (D2.10), so the E-side structural conclusion rests on an argument plus 40/40
   sign-consistent duals rather than on a binding test.
5. **Physical sufficiency of `eps = 1e-4`** (H1.10): enforceable now, but
   permitting circulation at 0.83 of the `sqrt(eps)` allowance. A later isolated
   `1e-5` / `1e-6` A/B is recommended.
6. **The H1 numerical cost**: total network iterations 1 556 -> 3 499 at H1, now
   3 442 after D2-P. No new failure family, no recoveries, rank diagnostics
   identical.

Resolved and no longer open: the ESSO's absolute complementarity semantics
(H1.5), the ESSO aggregate feasible-set incompatibility (H1.6), the
under-resolved network complementarity (H1.1-H1.3), the unexplained P5.4-D
sensitivity mismatch (attributed by D2 to two named causes), and the
structural incompleteness of the shared-S coefficient (fixed by D2-P, now
exactly zero bound contribution and sign-consistent across all 18 aggregated
coefficients).

---

# Not run

**P5.4-G was not run.** `run_planning_problem()` was not invoked during D2-P or
D3. No cut was ever added to a master problem. No final P5.4 verdict is issued
in this revision; issuing one would require asserting a G result that does not
exist -- and D3 indicates G should not run on the current cut machinery.

---

# Verdicts

```
P5.4-H1 PASS — complementarity is numerically resolved consistently across agents
```

```
P5.4-F ADMM PASS — net-P/Q coordination converged with locally consistent charge/discharge
```

```
P5.4-D2 PARTIAL — sensitivity root cause identified but production correction still required
```

```
P5.4-D2-P PASS — structurally complete S sensitivity productionized
```

```
P5.4-D3 FAIL — current cuts are demonstrably unsafe under observed recourse branches
```
