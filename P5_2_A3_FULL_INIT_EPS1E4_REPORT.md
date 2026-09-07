# Stage P5.2-A3 — Full positive-bootstrap initialization at `epsilon_rel = 1e-4`

Diagnostic only. **No production code was changed.** ADMM and the outer
planning loop were **not** entered.

## Provenance

- Script: `p52a3_full_initialization_eps1e4.py`. Raw output:
  `data/SRP1/Results/P52A3/p52a3_report.json`.
- Git `f77d829359ff…` (P4.6-B2); tracked working tree clean throughout.
- Scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` —
  identical to P5, P5.1, P5.1-B, P5.2-A and P5.2-A2.
- Candidate from the real production `_build_positive_bootstrap_candidate`;
  complete production initialization path executed
  (`create_distribution_networks_models` → `create_transmission_network_model`
  → `create_shared_energy_storage_model` → `_admm_local_solves_succeeded`).
- Only diagnostic difference from production: the shared-ESS `sess_snet_def`
  rows are converted **in memory** to the two-sided band with
  `epsilon_rel = 1e-4`, applied through a temporary wrapper on
  `configure_shared_ess_operational_state` that is restored afterwards.
  **`kappa = 1/S_rated` is untouched — no cap.**
- Wall clock 242.1 s.

---

## 1. Solver results (§2, §3) — flawless

| | |
|---|---|
| **Total local solves** | **51** |
| DSO | 36 — **36 succeeded, 0 failed** |
| TSO | 12 — **12 succeeded, 0 failed** |
| ESSO | 3 — **3 succeeded, 0 failed** |
| Network solves instrumented | 48 |
| **Primary successes** | **48 / 48** |
| **Primary failures** | **0** |
| **Recovery attempts** | **0** |
| Recovery successes | 0 (none needed) |
| **Clean primary successes** | **48 / 48** |
| **Persistent failures** | **0** |
| `_admm_local_solves_succeeded` | **True** |
| **Would initialization enter ADMM?** | **Yes** |

ESSO console markers confirm the same: 0 primary failures, 0 recovery attempts,
0 solver execution failures. No new failure family of any kind appeared.

*Instrumentation note:* 48 of the 51 solves are network (DSO/TSO) solves with
IPOPT logs, so their primary/recovery attempts and iteration counts are
directly attributable. The 3 ESSO solves use MA57 and define no `output_file`,
so they write no IPOPT log; they are instrumented from console markers only and
their iteration counts are unavailable.

## 2. Comparison against `epsilon_rel = 1e-5` (§4)

| | **P5.2-A, ε = 1e-5** | **P5.2-A3, ε = 1e-4** |
|---|---|---|
| Total local solves | 51 | 51 |
| Persistent failures | 0 | **0** |
| Would enter ADMM | Yes | **Yes** |
| Primary/recovery split | **not instrumented** | **fully instrumented** |
| Primary failures | ≥ 1 known (node 7 / 2030 Summer) | **0** |
| Recovery attempts | ≥ 1 known | **0** |
| Clean primary successes | unknown | **48 / 48** |

> **Primary question — does 1e-4 preserve zero persistent failures while
> reducing primary-path fragility?**
>
> **Yes, on both counts.** Persistent failures remain zero, and the
> recovery dependence known at ε = 1e-5 is gone: every one of the 48 network
> solves now converges on the primary exact-Hessian path with no recovery
> attempted anywhere.

## 3. Physical-relaxation audit (§5) — where the concern is

Evaluated over **every active shared-ESS row in every successful local solve**:
**1728 rows**.

| Metric | Value |
|---|---|
| max `｜g｜/S_rated²` | **2.6331e-04** |
| mean `｜g｜/S_rated²` | 1.2871e-05 |
| 95th percentile `｜g｜/S_rated²` | 5.9805e-05 |
| **max band utilization** `(｜g｜/S²)/1e-4` | **2.6331** |
| mean band utilization | 0.1287 |
| 95th percentile band utilization | 0.5980 |
| **rows above 0.5 utilization** | **126 of 1728 (7.29 %)** |
| **rows above 0.9 utilization** | **22 of 1728 (1.27 %)** |
| **rows active at a boundary** | **20** |
| max `ΔS` [p.u.] | 4.7911e-07 |
| max `ΔS` [MVA] | **4.79e-05** (≈ 48 VA) |
| max `ΔS / S_rated` | **2.2525e-03** |

### The band is reached and exceeded — and it is concentrated in the TSO

**All ten worst rows are TSO rows.**

| Tag | ESS idx | Period | S_rated [p.u.] | Band utilization | `｜g｜/S²` | ΔS/S_rated |
|---|---|---|---|---|---|---|
| `tso/2030/Spring` | 2 | 11 | 2.1270e-04 | **2.6331** | 2.6331e-04 | 3.977e-04 |
| `tso/2025/Autumn` | 2 | 3 | 1.0635e-04 | 1.4543 | 1.4543e-04 | 1.482e-03 |
| `tso/2025/Autumn` | 0 | 3 | 1.0635e-04 | 1.4543 | 1.4543e-04 | 1.482e-03 |
| `tso/2025/Autumn` | 1 | 3 | 1.0635e-04 | 1.4478 | 1.4478e-04 | 1.459e-03 |
| `tso/2030/Spring` | 2 | 8 | 2.1270e-04 | 1.2758 | 1.2758e-04 | 1.832e-03 |
| `tso/2030/Spring` | 1 | 8 | 2.1270e-04 | 1.2758 | 1.2758e-04 | 1.832e-03 |
| `tso/2030/Spring` | 0 | 8 | 2.1270e-04 | 1.2752 | 1.2752e-04 | 1.825e-03 |
| `tso/2035/Autumn` | 1 | 4 | 3.1905e-04 | 1.1755 | 1.1755e-04 | 1.308e-03 |
| `tso/2025/Summer` | 2 | 23 | 1.0635e-04 | 1.1643 | 1.1643e-04 | 6.294e-04 |
| `tso/2030/Summer` | 0 | 23 | 2.1270e-04 | 1.1396 | 1.1396e-04 | 6.786e-04 |

### Why utilization can exceed 1.0 — mechanism, not a measurement error

The scaled band half-width is `κ·ε·S_rated² = ε·S_rated`, e.g.
`1e-4 × 1.0635e-04 ≈ 1.06e-08`. IPOPT's constraint-violation tolerance for
these solves is `tol = 1e-5` (scaled), **three orders of magnitude larger than
the band itself**. The band is therefore *below the solver's own feasibility
resolution*: IPOPT legitimately reports `Optimal Solution Found` at points that
lie outside the declared band, because the violation is far inside its
tolerance.

Two consequences follow, and they pull in opposite directions:

1. The band's benefit is **structural, not feasibility-enforcing** — it changes
   the row from an equality (active with `grad(g) = 0` at the degenerate
   zero-dispatch point) into an inequality that is inactive there. That is what
   fixes the cold-start KKT fragility, and the solver evidence for it is
   unambiguous.
2. The band does **not** bound the physical mismatch in practice at this scale.
   The declared limit is not what actually constrains the solution; IPOPT's
   feasibility tolerance is. Quoting `ε` as a physical error budget would be
   misleading.

### Scale of the physical degradation

Under the hard production equality, earlier stages measured
`|g|/S_rated² ≈ 1e-15 … 1e-11`. Here the mean is `1.29e-05` and the maximum
`2.63e-04` — roughly **seven to nine orders of magnitude larger**. In absolute
terms the mismatch remains small (worst case `ΔS ≈ 48 VA`, i.e. 0.23 % of a
21 kVA unit), but it is no longer negligible relative to the relation being
enforced, and 7.3 % of rows use more than half the nominal band.

Note also that the single-case tests in P5.2-A / P5.2-A2 showed band
utilization ≤ 0.115 with **zero** rows near a boundary. Only the full
population — in particular the TSO models, which carry three shared ESS each
and were never part of the targeted case set — reveals utilization above 1.0.
The earlier targeted evidence was not representative.

## 4. Decision rule (§7) — evaluated criterion by criterion

| # | Criterion | Verdict |
|---|---|---|
| 1 | All 51 local initialization solves ultimately succeed | **Met** — 51/51 |
| 2 | Zero persistent failures | **Met** — 0 |
| 3 | No new systematic failure family | **Met** — none |
| 4 | Primary-path behaviour at least as good as ε = 1e-5, preferably better | **Met, and clearly better** — 48/48 clean primary, 0 recovery |
| 5 | Physical band usage remains **comfortably below** the permitted limit | **NOT met** — max utilization 2.63; 20 rows active at a boundary; 22 rows above 0.9; 126 rows (7.29 %) above 0.5 |
| 6 | No evidence of economic exploitation of the relaxation | **Not demonstrable** — the band is reached and exceeded on TSO rows, so the relation is not merely slack; absolute mismatch stays small (≤ 48 VA) but the evidence does not support a clean "no" |

Criteria 1–4 are satisfied decisively. Criteria 5 and 6 are not. The decision
rule permits recommending productionization **only if all six** hold, so
**`epsilon_rel = 1e-4` is not recommended for productionization on this
evidence.**

Per §7 no new persistent failure occurred, and no epsilon larger than 1e-4 was
tested — nor any other value.

## 5. Interpretation — kept narrow

1. **Established.** At ε = 1e-4 the complete bootstrap initialization succeeds
   entirely on the primary exact-Hessian path: 51/51 solves, 48/48 clean
   primary, zero recovery, zero persistent failures, and the system would enter
   ADMM. This is the strongest solver-side result of the whole P5 sequence.
2. **Established, and it is the blocking finding.** The physical band is not
   respected in practice: 20 rows sit at a boundary and the worst row exceeds
   the declared band by 2.6×, because the band is narrower than IPOPT's own
   constraint tolerance. The mismatch is concentrated in the TSO models, which
   the earlier targeted experiments never sampled.
3. **Not established.** Whether the observed mismatch is economically
   consequential downstream (ADMM and the outer loop were not entered); whether
   a different ε, a differently-scaled band, or a reformulation that places the
   tolerance above the solver's feasibility resolution would behave better. None
   of these was tested.
4. **No production change is authorized by this report.**

## 6. Scope compliance (§6)

Production code, `kappa`, `sess_comp`, SOC, degradation, ordinary ESS, IPOPT
options, MA97, Hessian settings, recovery policy, ADMM and planning/Benders
logic are all untouched. The only diagnostic difference from current production
is the in-memory ranged `sess_snet_def` at `epsilon_rel = 1e-4`. ADMM and the
outer planning loop were not entered.

---

```
P5.2-A3 PARTIAL — initialization succeeds but primary-path or physical concerns remain
```
