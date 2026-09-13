# P5.14-M — the C* evaluation COMPLETED

**The success branch. 67 cycles, converged inside the cap, zero local-solve failures, no
non-vanishing slacks, solve identity exact (`3468 = 51x67 + 51`).** This is the first
storage-benefit number this tool has produced at material capacity.

Frozen spec `data/SRP1/Results/P514D/frozen_cstar_v1_5a6210aa.json`, frozen before the run.
The predeclared mid-trajectory failure did **not** occur.

## 1. The three-point capacity series

All cold, `rel` 1e-4, C3 active, `rho_pf` 300 adaptive, same networks and horizon.

| point | `s` MVA/node | recourse | terminal step | **rule ten** | solves | cycles |
|---|---|---|---|---|---|---|
| cell 2 | 0 | 820,746,762.5 | 69,063 | **0.8414** | 3,621 | 70 |
| C1 cold | 0.0106 | 819,145,341.2 | 47,545 | **0.5800** | 3,519 | 68 |
| **C\*** | **0.96875** | **816,121,464.2** | 76,144 | **0.9329** | 3,468 | 67 |

**Monotone decreasing in capacity** — the direction the paper claims.

## 2. The pairs, with both bars (rule nine)

| pair | difference | estimated bar | vs estimated | vs bounded (164,000) | % of base |
|---|---|---|---|---|---|
| 0 -> 0.0106 | 1,601,421 | 116,608 | 13.7x | 9.8x | 0.1951% |
| 0.0106 -> C\* | 3,023,877 | 123,689 | 24.4x | 18.4x | 0.3692% |
| **0 -> C\*** | **4,625,298** | 145,207 | 31.9x | **28.2x** | **0.5635%** |

The total effect is **0.56% of the objective**, against the paper's claimed 2.01% — about
a quarter of it, at 2.91 MVA / 11.6 MWh system-wide. Comfortably resolvable: 28x the
bounded bar.

## 3. The D1 anomaly is resolved — it was path divergence

The third point settles it, exactly as anticipated. Per unit of capacity added:

| step | capacity added | difference | per MVA |
|---|---|---|---|
| 0 -> 0.0106 | 0.0319 MVA | 1,601,421 | **50,193,933** |
| 0.0106 -> C\* | 2.8743 MVA | 3,023,877 | **1,052,023** |

**The first step is 47.7x more "valuable" per MVA than the second.** That is economically
incoherent for storage value: 10.6 kW per node cannot deliver half the saving that 958 kW
per node delivers. **The 1.6M first step is therefore not storage value**, and the
path-divergence explanation — reinforced by the rule-ten contrast (0.8414 against
0.5800) — is confirmed.

Consequence: **the 0 -> C\* figure of 4,625,298 is an upper bound on the storage effect,
not a measurement of it**, because it inherits the contaminated first step. The cleaner
quantity is the second step alone, 3,023,877 (0.37%), between two cells that both contain
a shared ESS.

## 4. The limits stated before the run, now with their values

**The 0 -> C\* comparison is the weakest of the three, and both its endpoints stopped near
their bounds** — rule ten 0.8414 and 0.9329. Under the refined ninth rule, the bar bounds
*stopping slack*, not *path divergence*, and is valid only where both runs have settled.
Neither endpoint settled comfortably. So `31.9x the estimated bar` licenses **"not
explained by stopping slack"** and **does not license "a real difference in the limit"**.

**The normalization confound is live in that same pair**: `max(|rating|, 0.10)` gives cell 2
a normalization of 0.10 and C\* one of 0.96875 — a **9.7x difference** on the ESS consensus
channel, layered on the capacity difference, affecting the convergence test and therefore
the stopping point.

## 5. Two corrections to earlier framing

**Reliability did transfer; the 1-in-3 was a boundary phenomenon.** At C\* there were
**zero local-solve failures in 3,468 solves**. The failure rate observed at 1.00 MVA was
not general degradation at material capacity — it was the feasibility boundary the ladder
then bracketed. My earlier "1-in-1,095 became 1-in-3" is correct about 1.00 MVA and
**wrong as a statement about material capacity generally**.

**Cost transferred well, and slightly downward.** 3,468 solves at C\* against 3,519 at
bootstrap capacity — **1.5% fewer**, not more.

## 6. What this run still does NOT exercise

`gross_operational_cost` equals `recourse` exactly, so **salvage is zero at C\***.

The plan invests in 2025 only, with `t_cal = 15` and a 15-year horizon, so the single
cohort is **fully depreciated at the terminal** and the salvage credit vanishes
structurally. So the fourth component of the Track E inertness table remains inert here —
**for a reason of plan shape rather than magnitude.**

Updating that table against this run:

| component | status at C\* |
|---|---|
| ESS consensus channel | **live** — the primal residual passed its 0.1 guard, which is how the diagnostic `KeyError` was reached |
| ESSO constraint set | **live** — its feasibility boundary is 3% above this capacity |
| ageing model | **UNKNOWN** — untested at this capacity; see below |
| terminal value | **still inert** — salvage is exactly zero by plan shape |

## 7. The follow-up this result makes necessary

**A C3-style perturbation at C\* capacity** — change the degradation constant and see
whether the recourse moves. At bootstrap capacity the same perturbation moved nothing to
sixteen digits.

This run does **not** establish that the ageing mechanism is live; it establishes that the
consensus and constraint machinery is. Until the perturbation is run, the 0.37% storage
effect measured here **cannot be attributed to a model in which storage degrades**.

**Not authorized. Reporting and stopping.**
