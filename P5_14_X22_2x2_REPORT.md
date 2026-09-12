# P5.14-X22 — the 2x2 under one formulation: {cold, warm} x {fixed rho 300, adaptive}

**All four cells complete, one candidate, one formulation, guards armed throughout
(0 blocked solves). Two predeclared questions are answered, and one caution of mine is
withdrawn.**

Frozen spec `data/SRP1/Results/P514X22/frozen_2x2_spec_v1_55b10d16.json`, frozen before
the two new cells ran. Instance: the recovered AB1 candidate
(`s = 0.010634892806127162`, `e = 0.021269785612254323` per node and year), C3 active in
every cell.

## 1. The four cells

| cell | cycles | solves | wall | final `rho_pf` | terminal `gross_operational_cost` |
|---|---|---|---|---|---|
| cold_fixed | 50 **(capped)** | 2601 | 2119 s | 300 | 1,310,473,449 |
| cold_adaptive | 32 | 1683 | 1293 s | 3.4683 | 826,833,558 |
| warm_fixed | 6 | 357 | 302 s | 300 | 827,888,692 |
| warm_adaptive | **4** | **255** | 224 s | 88.8889 | 827,738,664 |

Matched-cycle `gross_operational_cost`:

| cycle | cold_fixed | cold_adaptive | warm_fixed | warm_adaptive |
|---|---|---|---|---|
| 1 | 2,355,185,552 | 2,355,185,552 | 828,763,937 | 828,763,937 |
| 2 | 1,637,991,846 | 1,625,523,853 | 828,403,866 | 828,292,555 |
| 4 | 1,519,679,510 | 1,495,672,912 | 828,034,350 | **827,738,664** |
| 6 | 1,504,102,652 | 1,427,350,045 | **827,888,692** | — |
| 20 | 1,424,803,131 | 855,184,212 | — | — |
| 32 | 1,372,691,723 | **826,833,558** | — | — |
| 50 | 1,310,473,449 | — | — | — |

Cycle 1 is identical within each initialization (2,355,185,552 for both cold cells;
828,763,937 for both warm), which independently confirms one shared instance and two
shared starting points.

## 2. C3 is numerically inert on this instance — my confound caution is WITHDRAWN

`warm_fixed` under C3 reproduces P5.10-B's **pre-C3** run exactly:

| | X22 warm_fixed (C3) | P5.10-B pf300 (pre-C3) |
|---|---|---|
| pre-polish recourse | 827885239.5417057 | 827885239.5417057 |
| `dual_pf_mean_ratio` | 0.906183793111952 | 0.906183793111952 |
| `primal_pf` | 9.051837140987118e-05 | 9.051837140987118e-05 |
| cycles | 6 | 6 |

All sixteen digits. I invoked the C3 boundary to block the cold-versus-warm comparison
and **asserted the confound's existence without measuring its size** — the same error
pattern as the salvage channel. The definitional incomparability stands; the numerical
effect on this candidate is **nil**, so the comparison is available.

Likely mechanism, recorded as a hypothesis: at `e = 0.0213` p.u. the degradation term's
coupling into available energy is far below anything the network dispatch can resolve.
The *fact* needs no further run — P5.10-B is itself the C3-off arm of that comparison.

Two by-products: byte-level determinism of the warm path across the P5.13-C/D production
edits, and a retroactive demonstration that those edits were neutral on a *solving*
trajectory, not merely on a model build.

## 3. Primary question — adaptation's advantage is COLD-SPECIFIC

Predeclared: `advantage_is_general` if the warm pair differ by a margin of the order of
the cold pair's 40%; `advantage_is_cold_specific` if they differ by materially less,
"say under 5%".

| pair | difference |
|---|---|
| cold_adaptive vs cold_fixed, at matched cycle 32 | **−39.8%** (826.8 M vs 1372.7 M) |
| warm_adaptive vs warm_fixed, at terminal | **−0.018%** (827,735,215 vs 827,885,240) |

**`advantage_is_cold_specific` is the outcome.** Adaptation mostly compensates for a bad
cold start rather than improving the method.

It is not useless warm, though: 4 cycles and 255 solves against 6 and 357 — **29% fewer
solves** — with a marginally better objective. The gain warm is a cost gain, not an
objective gain.

## 4. Secondary question — the two oracles AGREE on Q

Predeclared: under 1% is evidence the structurally different oracles agree; over 5% is
evidence `Q` is initialization-dependent.

| comparison | difference |
|---|---|
| cold_adaptive vs warm_fixed | **−0.1275%** |
| cold_adaptive vs warm_adaptive | **−0.1094%** |
| warm_adaptive vs warm_fixed | −0.0181% |
| our warm_adaptive vs P5.9-B warm adaptive (pre-C3) | +0.0133% |

**All well under 1%.** A cold independent start and a warm templated start — different
oracles by construction — agree on `Q` to about **one part in a thousand**. That is the
strongest result of the stabilization effort so far, and it is evidence that `Q(x)` is
**well defined** rather than an artifact of where the iteration began.

One qualification that should not be lost: the sign is **systematic**, not noise. The
cold cell is lower in all comparisons, so the templated oracle appears biased **upward by
about 0.11%**. For a ranking whose best-to-second gap is narrower than that, the bias
matters even though the agreement is tight.

`cold_fixed` is capped and therefore contributes only an upper bound, not a value.

## 5. The cost of independence, measured at last

| ratio | value |
|---|---|
| cold_adaptive / warm_adaptive | **6.60x** |
| cold_adaptive / warm_fixed | **4.71x** |

This replaces the withdrawn 15x and my own ">= 3.5x" lower bound with a measured figure:
**an independent oracle costs between five and seven templated ones** — 1683 solves
against 255. That is the number the outer-layer decision rests on, and the tension is
now quantified rather than described: 0.11% of objective bias against a factor of 6.6 in
cost.

## 6. The attractor prediction, in the regime where it was derived

`warm_adaptive` settled at `rho_pf = 88.889` after three decreases (300 -> 200 -> 133.33
-> 88.889, then held) — **inside the predicted [80, 200] band**.

So the prediction holds warm and fails cold, exactly as the scoped reading anticipated:
the rule's operating point is a property of the iterate regime. Even within the warm
regime it is start-dependent along the multiplicative grid — P5.9-B settled at 131.687
from a start of 1000, this cell at 88.889 from 300 — so what the band captures is a
*range*, not a unique fixed point.

## 7. Solve-profile identity, and a label to correct

| cell | solves | `51 x cycles` | residual |
|---|---|---|---|
| cold_adaptive | 1683 | 1632 (32) | +51 |
| cold_fixed | 2601 | 2550 (50) | +51 |
| warm_fixed | 357 | 306 (6) | +51 |
| warm_adaptive | 255 | 204 (4) | +51 |

The residual is **exactly one initialization block of 51 in all four cells**, matching the
P5.12-R ledger structure. The harness labelled it "UNEXPLAINED" because its explanation
list allowed only a 48-block polish pass or zero; the label is wrong and the quantity is
fully accounted for. No polish pass ran in the warm cells (`polished_net_operational_recourse`
is null). Guards: **0 blocked solves in every cell.**

## 8. What this establishes

- `Q(x)` is well defined to ~0.1% across structurally different oracles, with a
  systematic +0.11% bias in the templated one.
- Independence costs **6.6x**.
- Adaptation's large objective effect is **specific to the cold regime**; warm it is a
  29% cost saving.
- Same rho, same tolerances, same candidate: warm converges in 6 cycles, cold does not
  converge in 50. Initialization dominates, as S2 concluded.
