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

The third point settles it. **The per-MVA ratio I first reported (47.7x) was computed with
mixed conventions and is withdrawn**: step 1 used C1's per-node *cumulative* capacity
(0.0319047, three annual tranches of 0.0106349) while step 2 used a system-wide denominator
with C1 taken at its *annual* investment. Computed consistently — the system-wide factor of
three cancels in any ratio, so only C1's convention matters:

| convention | C1 capacity | step 1 per MVA | step 2 per MVA | ratio |
|---|---|---|---|---|
| per-node cumulative | 0.0319047 | 50,193,933 | 3,227,830 | **15.6x** |
| horizon-average | 0.0212698 | 25,096,967 | 1,063,819 | **23.6x** |

**The ratio is not load-bearing and should not be relied on.** C1's capacity is a *rising
trajectory* while C\*'s is constant, so per-MVA normalization is genuinely ambiguous and the
choice swings the answer by 50%.

**The absolute statement needs no convention and is what carries the conclusion:**
**1.6M of operating saving from 32 kW per node cumulative is not credible on its face.**
The 1.6M first step is therefore not storage value, and the path-divergence explanation —
reinforced by the rule-ten contrast (0.8414 against 0.5800) — stands.

Consequence: **the 0 -> C\* figure of 4,625,298 is an upper bound on the storage effect,
not a measurement of it**, because it inherits the contaminated first step. The cleaner
quantity is the second step alone, 3,023,877 (0.37%), between two cells that both contain
a shared ESS.

**But C\* is the limiting endpoint of every comparison, not only the weakest one.** Its
rule-ten ratio is **0.9329** — it stopped at 93% of its threshold. All three pairs contain
C\*, so all three inherit its near-bound stop. In particular **the 0.37% second step, whose
endpoints are 0.5800 and 0.9329, is limited by the latter** — the same qualification that
applies to the cell-2 pair applies to the cleaner quantity too, and must be stated with it.

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

**This was a deliberate trade, not an oversight.** The 2025-only single-cohort shape was
chosen to sidestep cohort-SOC realizability, and its cost is a structurally zero terminal
value: **the design traded one inert component for another.** A later investment cohort
would carry remaining life and a nonzero salvage credit, at the price of reintroducing
multi-cohort allocation — which is the question the shape was chosen to avoid.

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

---

# Addendum — the EFC check could not be run, and why

**The realized EFC per day at C\* is NOT recoverable from the artifact. This is a reporting
miss.** The frozen campaign spec required "cell 3's realized EFC per day per cohort
alongside terminal SoH, with its margin to the 1.4612 binding threshold", and the harness
did not capture it.

What `d1_cell3.json` contains: per-cycle ADMM diagnostics, residuals, slack **maxima**, and
the objective series. What it does not contain: `es_avg_ch_dch_per_unit`,
`es_degradation_per_unit`, `es_soh_per_unit_cumul` — any of which would give the answer
directly. The ESSO models were not serialized, so **the number cannot be obtained with zero
solves.**

The requested check therefore has a cost, and the options differ sharply in what they buy:

| option | cost | what it gives |
|---|---|---|
| **(a) instrumented re-run of the C\* baseline** | ~3,468 solves, ~35 min | the exact converged EFC/day and SoH, **plus a determinism check** on the 816,121,464 result |
| (b) initialization-only probe at C\* | 51 solves, ~40 s | the EFC of the *initialization* dispatch only — indicative, since the converged dispatch differs |
| (c) fold instrumentation into the perturbation arm | ~3,468 solves | the perturbed arm's EFC, not the baseline's |

**Recommendation: (a).** It answers the question rigorously and, because the baseline
objective is already known to sixteen digits, it doubles as a reproducibility check on the
one number this campaign has produced at material capacity. Option (b) is cheap but cannot
settle the question it is being asked to settle, because cycling at initialization is not
cycling at convergence.

## Why the answer decides whether the perturbation is worth running

- **If cycling is negligible** — and zero non-vanishing slacks plus an inactive SoH floor
  already hint that it is — then **an inert ageing model is physically correct, not a
  defect.** A battery that barely cycles should show almost no cycling degradation, and the
  perturbation's outcome is predictable rather than informative.
- **If cycling is substantial**, approaching the 1.4612/day threshold, then an inert ageing
  model at C\* **is** a genuine defect, and the perturbation is exactly the test.

## The perturbation, predeclared, if warranted

Mechanism already documented and requiring no code change: set
`ageing.calibration.status` back to `DECLARED_NOT_CONSUMED`, which the params file records
as restoring `k = cycle_life_nominal = 10000` exactly. That is **the same 15.4% change
already characterised at bootstrap capacity**, making it a direct like-for-like test rather
than a new one.

Predeclared against the **164,149** bounded bar:

- **moves by more than the bar** -> the ageing model is live at material capacity, and the
  0.37% is attributable to a model in which storage degrades;
- **moves by less, or not at all** -> the ageing model is inert even where the storage is
  doing real work. Combined with salvage at exactly zero, **this run would then be valuing
  effectively ideal, non-degrading storage with no terminal value — and the paper's
  degradation modelling would contribute nothing to any number it reports.** That is the
  deeper finding, and the one to be ready for.
