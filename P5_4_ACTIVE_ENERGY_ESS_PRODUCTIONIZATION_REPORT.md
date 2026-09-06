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
> In physical terms `pch·pdch ≈ 2.85·eps·S²`, i.e. roughly 1.7 % of rating
> flowing simultaneously in both directions. Moving complementarity from
> `sch·sdch` to `pch·pdch` did **not** repair this, exactly as B3 predicted, and
> no scaling was introduced to mask it.

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
