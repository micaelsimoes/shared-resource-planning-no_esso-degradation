# Disclosure for the archived multi-scenario (3 × 3) results: draft text (2026-11-21; author decision 5)

**Scope.** The archived five-block 3 × 3 evaluations of the plan without storage and the reference unit (W91 pair,
spec `231558f0`; uncertified in the manuscript, l. 1635 and 1664). Row 18 premium α = 0.5. The text is in US English.
The corrected per-scenario evaluations, once available, will form the main narrative. These diagnostics stay separate
from them.

## Main text (one or two sentences, for the section that reports the archived 3 × 3 results)

> In these multi-scenario evaluations, the transmission network follows a scenario-common interface schedule and
> distribution-network deviations from it are settled financially; scenario-wise equality of the TN–DN exchanges is
> therefore not enforced. The supporting material quantifies this approximation.

## Supporting material (one paragraph)

> **Scenario-wise interface consistency of the archived multi-scenario evaluations.**
>
> **The approximation.** In the archived 3 × 3 evaluations, the TSO operates on the committed, scenario-common
> interface schedule in every scenario, while each DSO's interface exchange varies by scenario. Coordination enforces
> agreement of the expected exchanges only, and deviations are settled at the scenario price plus a deviation premium.
>
> **Its size.** At the evaluation of the plan without storage, the active-power exchanges of the two sides differ by up
> to 48.3 MW in a scenario-hour (probability-weighted RMS 3.8 MW). Reactive power and voltage agree closely
> (≤ 2.4 × 10⁻³ MVAr; ≤ 8.5 × 10⁻⁴ p.u.).
>
> **The cost check.** To size the effect on cost, the TN was re-evaluated with each scenario's realized DN exchange
> imposed, keeping the DN dispatch fixed, and the result was compared with the accounted cost.
> - **The change:** the accounted cost increases by about 0.93 M€ (+927.6 k€ without storage; +930.2 k€ with the
>   reference unit), about 0.11 % of the expected operating cost.
> - **What it contains:** this figure combines two parts and is not the TN redispatch-cost increment alone:
>   - the TN re-evaluation at fixed DN dispatch, +178.9 k€;
>   - the reconciliation of the financial settlement, +748.7 k€.
> - **Where it arises:** almost entirely in hours in which the TN must curtail renewable output while the settlement
>   credits the scenario price.
>
> **What it means for the storage value.** The corresponding change in the reference unit's value, −2.6 k€, is
> unresolved against the comparison threshold of about 18 k€. It does not establish that the value would remain
> unchanged under fully coordinated re-optimization with scenario-wise coupled exchanges.

## Numbers and sources

| quantity | value | source |
|---|---|---|
| max \|ΔP\|, x = 0 | 48.32 MW (DN 7, 2028 Summer, scenario 2_2, h 15) | audit D, `interface_consistency.json` (`f80ec8a9`) |
| ω-RMS ΔP, x = 0 | 3.77 MW (per DN 3.58 / 4.43 / 3.18) | same |
| max \|ΔQ\|, \|ΔV\| | 2.4e-3 MVAr; 8.5e-4 p.u. | same |
| TN re-evaluation at fixed DN dispatch, ΔC = C_B − C_A (x = 0) | +178,866.30 € | D0 v2, `tn_realized_resolve_v2_results.json` (`7bcca20d`) |
| settlement reconciliation (−DSO covariance, x = 0) | +748,745.77 € | same |
| D1 = ΔC − covariance | +927,612.08 € (x = 0); +930,240.58 € (unit) | same |
| reference-unit value change (D1(x0) − D1(unit)) | −2,628.51 € | same |
| comparison threshold (sum of the two certification bars) | 17,964.95 € (13,954.38 + 4,010.57) | same |
| relative size | 927.6 k€ / 842.8 M€ (Q of the archived x = 0 cell) ≈ 0.11 % | Q from `P5_15_ADDENDUM48_50_3X3_REPORT.md` |

**Conventions.**
- C is TN generation + TN flexibility, € per representative day, probability-weighted and block-weighted. It is gross,
  with the settlement excluded and no salvage.
- **"Fixed DN dispatch":** the DN realized exchanges are the archived DSO solution. The DNs did not respond.

## Wording to avoid

- **"Physically balanced" or "consistent exchanges"** for the archived 3 × 3 results.
- **Calling the 0.93 M€ "the redispatch cost".** It includes the settlement reconciliation.
- **"The storage value is unchanged" or "robust to the coupling".** The −2.6 k€ is unresolved; it is not evidence of
  invariance.
- **"Certified"** for the archived pair.

## To check before pasting

- **Placement:** the manuscript lines that describe the archived 3 × 3 formulation (round 9c, l. 1098 and 1158) need to
  sit consistently with the new sentence.
- **Manuscript version:** confirm against the current Overleaf HEAD, which was not read here.
