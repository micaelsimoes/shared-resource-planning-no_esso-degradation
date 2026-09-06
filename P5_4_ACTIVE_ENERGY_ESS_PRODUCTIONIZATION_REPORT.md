# P5.4 — End-to-end active-energy ESS productionization

**Status: checkpoint after A and E. Sections B, C, D, F, G, H are not yet
done, so no final verdict is issued in this revision.**

| Section | Status |
|---|---|
| **A — productionize shared network ESS** | **Complete** (`a4a0bae8`) |
| B — ordinary network ESS parity | Not started |
| C — ESSO active-energy conversion | Not started |
| D — sensitivity / lifecycle / warm-start audit | Partially covered by A's smoke tests |
| **E — production validation before ADMM** | **Complete (shared-ESS scope)** |
| **E2 — complementarity significance** | **Complete** |
| F — live distributed ADMM | Blocked on C |
| G — reduced planning gate | Blocked on F |
| H — remaining inequality-conditioning decision | Evidence gathered, decision pending |

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

## Physical residual audit — 864 active shared-ESS rows

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

# D — Partial coverage so far

The zero-capacity lifecycle (fix/unfix, bound collapse and restore, row
activation) was verified directly on the production code, and the removal of
the `kappa` multiplier-transfer logic is complete with **no replacement
transformation introduced** — the new rows depend directly on
`shared_es_s_rated`, so none is needed. The remaining D items (shared-S/E
sensitivities, Benders cut extraction, positive→different-positive transitions,
warm-start suffix behaviour, reused-model identity) are **not yet audited**.

---

# H — Evidence gathered, decision pending

The P5.4-E residual audit provides the evidence P5.4-H asks for:

- converter capability: **not** a problem in production (zero violation);
- complementarity: **materially violated**, 33 % of rows above `1e-4`
  normalized.

On that evidence a dimensionless-variable proposal is likely to be warranted for
the complementarity row specifically, but **the decision is deliberately
deferred** to after the live ADMM/planning evidence, as the plan requires, and
no row multiplier or normalization has been introduced.

---

# Remaining work before a verdict

**B** (ordinary-ESS parity — SRP1 instantiates no ordinary ESS, so this needs
the OP1 test case), **C** (ESSO active-energy conversion including the
throughput/degradation trace — the largest remaining piece and a hard
prerequisite for ADMM), the rest of **D**, then **F** and **G**.

No end-to-end semantic inconsistency has been found. Section C must be completed
before P5.4-F can legitimately run, because the ESSO model still uses apparent
per-cohort charge/discharge while the network agents are now active-power based.

**No final verdict is issued in this revision** — issuing one now would require
asserting results for sections that have not been executed.
