# P5.15 Step 3.1 — Row 3′ as authorized is ill-posed; author decision needed before implementation

**Planner note, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 11. Addendum 11's order was
discriminator → row 3′ implementation → converged campaign. **The discriminator is done. Implementation is stopped**,
because the row-3′ DSO term as written is mathematically ill-posed, and the well-posed alternatives force a choice that
only the author can make. No production code has been changed and no campaign run since `8bb1b8ce`.

Independent review: Advisor (read-only). The Planner re-read every load-bearing code claim (§5).

---

## 1. Discriminator result (zero solves, committed artifacts)

| evidence | capped post-signature campaign (`P515S31_run`) | Step 3.0 |
|---|---|---|
| interface primal PF residual / tolerance, cycles 1 → 90 | 44.5 → 36.0 → 30.9 → 25.2 → 23.5 → 12.8 → 12.4 | 0.18 at terminal |
| worst interface primal difference | 88.9 → 24.9 | 0.35 |
| dual PF residual | 161 → 8.9 | 0.024 |
| implied terminal TSO downward interface flexibility | **1.50–1.57 pu per interface-period in all 12 blocks** | — |
| sum of interface ratings (nodes 5, 7, 9) | **2.0 + 1.0 + 1.5 = 4.5 pu** | — |

The implied downward flexibility totals ≈ 4.5 pu per period — **the sum of the interface ratings**. The downward legs sit
at their upper bounds in every block while consensus holds the net exchange near the DSO value. Indicative (P and Q
legs combined, at average block price), but unambiguous in pattern: **interface consensus never formed, and the TSO's
interface legs are saturated.** The terminal interface variables themselves were not serialized.

## 2. Verified anchor

In the ADMM path the TSO's interface `pc`/`qc` are written in one place, at model creation
(`shared_resources_planning.py:3046–3060`), fixed to `consensus_vars['pf']['dso']['current']`. That value is written
immediately before (`create_distribution_networks_models`, ~3166–3173, 3213–3220) from each DSO's **standalone
initialization solve**, and the DSOs are built and solved before the TSO (`_run_operational_planning` ~2217, ~2223).
**The current anchor is the DSO's uncoordinated interface exchange, fixed for the run** — the quantity Addendum 11
names. Other writes exist only in the hierarchical (~3365) and uncoordinated (~6077) paths.

## 3. Why row 3′ as written is ill-posed

Conventions (verified): TSO interface `pc_adn = pc + flex_p_up − flex_p_down` (`model_construction_helpers.py:1173–1175`),
legs bounded `[0, R]`. DSO interface `pg_adn = pg_ref − shared_es_pnet` — **one signed expression, no split**
(`mch:1197–1205`). The augmented Lagrangian couples only expected values (`mch:2005–2011, 2059–2066`). SRP1 has
**one market scenario and one operation scenario everywhere** (`SRP1.json`), so per-scenario and expected values
coincide.

- **TSO `+c·|p_int − a|`** is well-posed: `+c·(flex_p_up + flex_p_down)` is exact at the optimum.
- **DSO `−c·|p_int − a|` is concave in a minimization.** With split legs `u − d = x − a`, `0 ≤ u, d ≤ R`, minimizing
  `−c(u + d)` gives `u + d = 2R − |x − a|`, so the DSO term becomes **`−2cR + c·|x − a|`: a cost on deviation, not a
  revenue**. The two blocks sum to `2c|x − a| − 2cR`, not zero. Without bounds it is unbounded. A downward-only revenue
  `−c·max(0, a − x)` is concave in the same way. An exact model needs complementarity or integers.
- **"Q as today"** (downward-only `c·flex_q_down`) is not consistent with a symmetric P form. The pre-signature row 3
  itself was downward-only on P as well (`c·(flex_p_down + flex_q_down)`), not `|·|`.

## 4. The structural point the author must weigh

**A transfer that cancels exactly cannot price the unpriced direction.** If `T_TSO + T_DSO = 0` on the consensus set,
the coordinated problem is unchanged; the transfer only shifts the consensus dual by the contract price. It can fix the
accounting and act as a dual warm start, but it cannot remove what the discriminator shows: the **up/down null space**
`(flex_up + k, flex_down + k)` of the TSO interface legs, which moves neither the interface exchange nor any priced term.
The pre-signature charge removed that null space by pricing the downward leg — at the cost of putting a transfer
into Q(x).

So the two aims in Addendum 11 separate:

- **(i) Q(x) = system cost with the transfer reported separately** — an accounting requirement.
- **(ii) no unpriced direction in the TSO block** — a formulation requirement.

## 5. Well-posed options for the author

| option | TSO | DSO | cancels at consensus | convex | removes null space | contract / meaning |
|---|---|---|---|---|---|---|
| **A. signed linear transfer** | `+c̄·(a − x̄)` | `−c̄·(a − x̄)` | **exactly** | linear | **no** | DSO paid per MWh of reduced import relative to its baseline, both directions (symmetric) |
| **A + signed interface variable** | as A; interface as one signed variable `pc ∈ [a − R, a + R]` with the legs removed | as A | exactly | linear | **yes, exactly** | same contract; same feasible set; the interface flexibility is not fixed at zero, it is reparametrized |
| **A + ε-regularizer (R)** | as A, plus `ε·(flex_up + flex_down)` | as A | exactly (ε term excluded from Q, reported) | linear | yes, approximately | same contract; small category-R bias, measured |
| **D2. downward-only, exact** | `+c̄·d`, `x̄ = a − d`, `d ≥ 0` | `−c̄·d`, same | exactly | linear | yes | DSO may only reduce import — **changes the feasible set** |
| B. pre-signature (TSO charge only) | `+c·(down_p + down_q)` | none | no | yes | yes | rejected by the signed table (transfer inside Q; anchor-dependent) |
| C. DSO revenue linearized at the ADMM iterate | `+c·|·|` | `−c·sgn(x̄^k − a)(x̄ − a)` | only at a fixed point | per subproblem | yes | no convergence guarantee; the sign chatters around `x̄ = a`, where the run starts |

Notes: `c̄` is the expected flexibility price per period (equal to `c_flex` in SRP1). The transfer must enter each
block's **physical** objective before the objective-scale division, or cancellation fails in currency units. With one
scenario, the per-scenario vs expected distinction (Jensen) does not arise in SRP1 but will in multi-scenario cases.

**Planner recommendation, for the author's decision:** **option A with the signed interface variable**. It is the only
form that satisfies both (i) and (ii) exactly, keeps the feasible set, adds no bias, and keeps every block linear in the
transfer. It needs the author's explicit agreement that reparametrizing the TSO interface legs as one signed variable
is consistent with "interface flexibility variables are not fixed at zero".

## 6. Decisions requested

1. **Contract:** symmetric (option A family) or downward-only (D2, changes feasibility).
2. **Null space:** signed interface variable (exact) or ε-regularizer (category R).
3. **Q convention:** Q at the P price, as in row 2, or its own price.
4. **Gate wording:** with option A, `T_TSO + T_DSO` equals the consensus residual times `c̄` by construction; the gate
   remains meaningful as a check on implementation, not as evidence of convergence.

## 7. What is NOT established

- That removing the null space alone makes the campaign converge. The discriminator shows saturated legs and absent
  consensus; it does not isolate H3 (null space) from H1 (the mean interface deviation priced only by the dual). Option A
  plus the signed variable addresses both: A as a dual offset, the signed variable for the null space.
- The up legs' terminal values (not serialized); the saturation reading rests on the downward legs and the ratings.

## Evidence

`data/SRP1/Results/P515S31B/discriminator_from_artifacts.json`; `data/SRP1/Results/P515S31B/interface_ratings_vs_implied_downflex.json`
(both zero solves under a blocking guard).
