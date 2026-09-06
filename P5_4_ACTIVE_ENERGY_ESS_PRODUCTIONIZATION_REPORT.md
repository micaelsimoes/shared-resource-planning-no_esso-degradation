# P5.4 — End-to-end active-energy ESS productionization

**Status: checkpoint after A, B, C, D, E and E2. F and G are not started, so
no final verdict is issued in this revision.**

| Section | Status | Commit |
|---|---|---|
| **A — productionize shared network ESS** | **Complete** | `a4a0bae8` |
| **B — ordinary network ESS parity** | **Complete** | `1e86d40e` |
| **C — ESSO active-energy conversion** | **Complete** | `58f4911b` |
| **D — sensitivity / lifecycle / warm-start audit** | **Complete, with one negative result** | `c3526ec8` |
| **E — production validation before ADMM** | **Complete, revalidated after B/C** | `c3526ec8` |
| **E2 — complementarity significance** | **Complete** | `b0e53bc4` |
| F — live distributed ADMM | Not started — stopped here per planner instruction | — |
| G — reduced planning gate | Blocked on F | — |
| H — remaining inequality-conditioning decision | Evidence gathered, decision pending | — |

Every agent in the coordination is now active-power based. Nothing in P5.4
introduced dimensionless ESS variables, row scaling, a new complementarity
tolerance, binary charge/discharge variables, or additional penalties.

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

# H — Evidence gathered, decision pending

The E, E2 and B evidence together sharpen what P5.4-H is actually deciding
about:

- **converter capability: not a problem.** Zero violation on all 1 728
  shared-ESS rows and all 48 ordinary-ESS rows, and zero in the ESSO.
- **complementarity: materially violated on the shared ESS**, up to
  `min(pch, pdch) = 11.6 %` of rating, concentrated in the TSO models.
- **but the same formulation is clean on the ordinary ESS** (0 / 48 violations,
  peak circulating power below `1e-3·S`), where the rating is ~24× larger.

That last point is new and it matters: the residual is **a property of the
bootstrap operating regime, not of the active-energy formulation**. A
dimensionless-variable proposal may still be warranted for the complementarity
row, but the evidence no longer points at the formulation itself. **The decision
remains deferred** to post-ADMM evidence as the plan requires, and no row
multiplier or normalization has been introduced.

---

# Open items carried into F

1. **ESSO complementarity is vacuous at bootstrap capacities** (section C). The
   ESSO uses an absolute `pch·pdch ≤ 1e-4` where the network models use
   `eps·S²`; at `S ≈ 2.13e-04 p.u.` the ESSO bound never binds, and its
   aggregate circulating power reaches 19 % of rating. This is directly
   relevant to ADMM ESS consensus. Not changed — a new complementarity
   tolerance was explicitly out of scope.
2. **The analytic capacity sensitivity is unconfirmed by finite differences**
   (section D) — pre-existing, improved but not resolved by P5.4, and a risk to
   Benders cut quality.
3. **The complementarity residual itself** (sections E, E2), pending the H
   decision.

Neither 1 nor 2 was introduced by P5.4, and no end-to-end semantic
inconsistency was found. Every agent is now active-power based, which was the
hard prerequisite for P5.4-F.

**Execution stopped before P5.4-F as instructed. No final P5.4 verdict is
issued in this revision** — issuing one would require asserting results for F
and G, which have not been executed.
