# P5.15 — Gate G5: PASSED under the re-specified criterion (Addendum 4). No nonconvexity.

**Planner report. Superseded in its verdict by Addendum 4; the evidence below is unchanged.**

> **OUTCOME (Addendum 4, 2026-09-13): G5 PASSES.** The original 1e-6 absolute criterion was
> mis-set — same class as the ε detector threshold — because it compared an objective composed
> entirely of barrier artifacts. Re-specified and **verified independently by the Planner**:
>
> | criterion | threshold | measured | |
> |---|---|---|---|
> | (i) net power `pch − pdch` across A1/A3/A4 | 1e-10 | **4.163e-16** | PASS |
> | (ii) `es_D_per_unit` within the summed analytic leak estimates | 2.5758e-04 | **7.633e-05** | PASS (3.4× headroom) |
> | (ii) `es_soh_per_unit_cumul`, same budget | 2.5758e-04 | **2.182e-05** | PASS |
> | (ii) `es_soh_per_unit_cumul`, absolute | 1e-4 | **1.6387e-05** | PASS |
>
> Leak budget = leak(A1) 9.0694e-05 + leak(A3) 1.6689e-04. The decomposition table below is the
> evidence that no nonconvexity remains. The FAIL framing in the body is retained as the record
> of how the criterion was found to be mis-set.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 1 (G5) and Addendum 3.

## Verdict

G5 requires A1, A3 and A4 to agree to **1e-6 relative** on the reformulated ESSO, and states
that disagreement "means the reformulation left a nonconvexity and Step 1 does not close until
it is identified."

**The gate fails. The disagreement is identified, and it is not a nonconvexity.**

## The decomposition that settles it — node 7 (node 9 identical)

| variable | A1 vs **A3** (`mu_strategy` differs) | A1 vs **A4** (warm vs cold) |
|---|---|---|
| `es_pnet` (control; hard-fixed) | 0.000e+00 | 0.000e+00 |
| **net power `pch − pdch`** per cohort-period | **4.163e-16** | 2.776e-16 |
| `es_avg_ch_dch_per_unit` | **7.633e-05** | 5.655e-12 |
| `es_D_per_unit` | **7.633e-05** | 5.655e-12 |
| `es_soh_per_unit_cumul` | **2.182e-05** | 1.993e-12 |

**The three arms reach the same dispatch decision to machine precision.** What differs is the
*derived degradation chain*, and the reason is structural: net power is **subtractive**
(`pch − pdch`), so the leak cancels exactly; throughput is **additive** (`pch + pdch`), so the
leak accumulates at `2·min(pch,pdch)` per period. The degradation law is fed the additive
quantity.

**A1 vs A4 — warm against cold start, the most different initial points available — agree to
1e-12 on every quantity.** That is where multiple local optima would appear first, and they do
not. The disagreement appears only when `mu_strategy` changes, i.e. when the barrier path
changes, which is precisely the mechanism already established.

## So what G5 actually shows

1. **No nonconvexity is demonstrated.** Same decision, from maximally different starts.
2. **The degradation chain remains solver-dependent**, now at **7.6e-05** relative across
   `mu_strategy` — roughly 100× better than before remedy (h), but not zero. Remedy (h)
   shrinks the artifact; it does not remove it, exactly as its own rationale said.
3. Under the **production** `mu_strategy` (A1 and A4), the reported SoH is reproducible to
   1e-12. A3 is not a production setting.

## The slack test — my predicted formula was wrong

I predeclared `min(slack_up, slack_down) = mu_final/(2·s_obj·PENALTY_ESSO_SLACK)`, predicting
4.548e-12 (A1) and 9.352e-12 (A3). **Falsified.** Measured slacks are **negative, ≈ −1e-8, and
essentially constant across all 288 periods** — an IPOPT bound-relaxation floor
(`bound_relax_factor = 1e-8`), not a μ-scaled residual. Measured/predicted ≈ −2197 (A1),
−1067 (A3).

The arithmetic I offered in support of that hypothesis (predicted 2.767e-6 against an observed
residual of 4.4035e-6, "right order, factor 1.59") was built on a formula that does not apply to
this pair. It was labelled supported-not-established; it is now **withdrawn**.

What *is* established: substituting the **measured** slacks closes the A1-vs-A3 objective gap to
**8.4e-17**, confirming `model.objective ≡ PENALTY_ESSO_SLACK·Σ(slack_up+slack_down) +
eps·Σ(pch+pdch)` exactly, and nothing else. Both terms are solver artifacts on this fixture,
because `es_pnet` is hard-fixed and the objective therefore carries no physical content.

## Why the gate cannot pass as written

G5 compares `model.objective`. On this fixture that objective is **entirely** penalty plus
regularization — both barrier artifacts — and A3 differs precisely in the parameter that sets
the barrier path. **G5 as specified requires two artifacts to agree to 1e-6.** This is the same
defect diagnosed for G1, on a different quantity.

## Recommended re-specification (author decision)

Gate on quantities the model determines:

1. **Dispatch invariance** — `pch − pdch` and `es_pnet` agree across A1/A3/A4 to 1e-9.
   *Currently passes at 4.2e-16.*
2. **Reproducibility under the production barrier strategy** — A1 vs A4 agree to 1e-6 on the
   degradation chain. *Currently passes at 5.7e-12.*
3. **Artifact magnitude declared, not required to vanish** — report the cross-`mu_strategy`
   spread on the degradation chain (**7.6e-05**) as a stated uncertainty on SoH.

Retain the original 1e-6-on-objective form only if the objective is first purged of artifact
terms, which no remedy on the table does.

## Status of the Addendum-3 fallback

Addendum 3 keeps (d)/(e2) as fallback "if the detector at `tol = 1e-8` exceeds 1e-4 relative
throughput at C\*". Measured at `tol = 1e-8` on the ε fixture: detector ratio **4.5351e-06**,
spurious throughput **0.0091 %** (9.1e-05 relative). That is **below** the 1e-4 trigger, but it
is the *fixture*, not C\*. **The trigger has not been evaluated at C\***, and cannot be until
G1 runs.

## Held

- **G1–G4 not run.** They depend on G5 closing, and on G1's own re-specification.
- **`REVISION_CONTEXT.md` rewrite not started.** Its content depends on the gate outcomes.

## Process note — a capture gap that was mine

The original G5 task required the objective, detector, `mu_final` and throughput, but **not** the
solution vector or the slacks — the quantities that discriminate "artifact" from "nonconvexity".
The gate therefore produced an uninterpretable FAIL and needed a second six-solve run. This is
the rule-eleven failure mode: a gate specified on a quantity without requiring capture of what
makes its result interpretable. Rule eleven should be extended from "capture what the spec
requires" to "capture what would distinguish the competing readings of a failure".
