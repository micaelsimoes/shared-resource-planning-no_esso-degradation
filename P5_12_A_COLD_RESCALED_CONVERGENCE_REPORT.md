# P5.12-A — cold RESCALED fixed-ρ convergence diagnostic

**Outcome: the diagnostic encountered local NLP failures. The trajectory did not
converge, and from cycle 83 onward it broke down permanently.**

---

## 1 — Repository and runtime provenance

| | |
|---|---|
| hostname | `Micaels-Mac-Studio.local` (macOS 26.6.2, arm64) |
| repository | `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation` |
| runtime | `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python` |
| Python / NumPy / pandas / SciPy / Pyomo | 3.11.11 / 2.4.2 / 3.0.0 / 1.17.0 / 6.9.5 |
| IPOPT | 3.14.18, ASL 20241111, `/usr/local/bin/ipopt` |
| linear solver | MA97 |
| scenario checksum | `5a02b77c…10358`, canonical |

`/Users/micaelsimoes/PycharmProjects/shared-resources-planning` was neither
inspected nor modified.

## 2 — Git state

| | initial | final |
|---|---|---|
| branch | `feature/derivative-free-planning` | same |
| HEAD | `808590d23ad250627be9ff9d4ae51d059cbdbcdb` | **unchanged** |
| tracked modifications | 0 | **0** |
| upstream divergence | ahead 8, behind 0 | unchanged |

**HEAD moved between P5.11 and P5.12-A**, from `fbdb2f6a` to `808590d2`. The
intervening commit is a user merge dated 2026-09-10 07:44 bringing in origin
`58140e03` ("`.md files updated.`"). It touched **only** `REVISION_CONTEXT.md`
and `LOCAL_NLP_STABILITY_PLAN.md`; no production file, no P5.10 or P5.11
artifact. `fbdb2f6a` remains an ancestor. This was treated as an explained
accepted commit and the preflight continued.

The merged documents carry a `CURRENT SOURCE OF TRUTH — 2026-09-10` that names
this checkout as authorized, records P5.11's end state correctly, and covers
P5.10 and P5.11. **Observation, not a finding:** the merge resolved both files
wholly in origin's favour, and the result contains no references to P5.6-D,
P5.7 or P5.8 (P5.9 appears once). If that pruning was deliberate, nothing is
wrong; if not, that material is recoverable from `fbdb2f6a`. Nothing was changed
either way.

## 3 — P5.10 reproduction gates

```
python p512_a_preflight.py
```

| gate | expected | observed | Δ | bit-identical |
|---|---|---|---|---|
| CURRENT polished total | `828021090.3608505` | `828021090.3608505` | `0.0` | **yes** |
| RESCALED pre-polish recourse | `825814074.4930633` | `825814074.4930633` | `0.0` | **yes** |

## 4 — P5.11 cycle-25 state verification

| item | expected | result |
|---|---|---|
| template id | `P511-SELFCONSISTENT-T0` | match |
| consensus/dual state hash | `14a7ea85…a625cbf0` | match |
| artifact file hash | `3629784b…909a67bd` | match |
| configuration hash | `a080612dedf2727e` | match |

The saved state verified, so no rebuild-for-verification was required. In
addition, all four fresh trajectories reproduced the accepted P5.11 recourse
markers exactly: cycle 1 `2352009862.13208`, cycle 13 `1461174062.836328`,
cycle 25 `1402384338.113119` (relative tolerance `1e-6`).

## 5 — Code diff

**No tracked file was modified.** Verified individually:
`shared_resources_planning.py`, `network.py`, `admm_parameters.py`,
`solver_parameters.py`, `data/SRP1/SRP1_params.json`, `REVISION_CONTEXT.md`,
`LOCAL_NLP_STABILITY_PLAN.md` — all unchanged. `data/SRP1/Results/P510/`
untouched. All P5.11 artifacts intact (report SHA-256 still `a9f9e342…`).

New untracked files only:

```
p512_a_preflight.py                    A0 provenance and reproduction gates
p512_a_cold_rescaled_convergence.py    A1 trajectory diagnostic
data/SRP1/Results/P512A/               evidence
p512_a_preflight.log, p512a_cap*.log   console logs
```

### A defect in the harness, and its cost

The first launch of all four runs aborted within seconds:
`ValueError: not enough values to unpack (expected 7, got 6)`. The public
`run_operational_planning` returns five values plus `state`
(`shared_resources_planning.py:87-89`); the seven-value form belongs to the
internal `_run_operational_planning`. I had used the internal signature. The
failure was immediate and deterministic, before any model was built, so no solve
time was lost and no partial evidence was written — the four JSON files were
overwritten by the relaunch. Fixed with a one-line correction and a comment
recording the distinction.

## 6 — Commands executed

```
python p512_a_preflight.py
python p512_a_cold_rescaled_convergence.py run 25
python p512_a_cold_rescaled_convergence.py run 50
python p512_a_cold_rescaled_convergence.py run 75
python p512_a_cold_rescaled_convergence.py run 100
```

The four `run` invocations executed concurrently — four workers, the documented
P5.6-B7 ceiling.

### Why four runs rather than one with in-flight snapshots

The stage requires detached polish at cycles 25/50/75/100 that cannot perturb the
live trajectory. Production owns its coordination loop internally, so
intercepting mid-loop would mean hooking a production function and deep-copying
48 models in flight. Instead each checkpoint came from its own cold run with the
cap set to that cycle count. The cold trajectory is deterministic, so the longer
runs reproduce the shorter ones — **verified, not assumed**: all four agree on
the P5.11 markers and all four show local failures at exactly cycles 21-24.
Nothing was spliced.

`num_max_iters` was raised only on each run's own per-evaluation deep copy.
`data/SRP1/SRP1_params.json` was never written and the production default is
untouched.

## 7 — Trajectory

Configuration held exactly as frozen: base candidate, `RESCALED`,
ρ = `1.5 / 300 / 1.0`, adaptive ρ disabled, original cold initialization,
midpoint anchor, production convergence definitions, tolerances, IPOPT options
and proximal regularization unchanged. ρ was observed at
`{v: 1.5, pf: 300.0, ess: 1.0}` on all 48 blocks at the end of every run — it
never moved.

| cycle | recourse | Δ recourse | `primal_pf` | `primal_v` | `dual_pf_mean` | converged |
|---|---|---|---|---|---|---|
| 1 | 2 352 009 862.13 | — | 4.567e-01 | 6.779e-02 | 6.601e+01 | False |
| 5 | 1 511 570 375.36 | −8 486 387.27 | 3.200e-04 | 4.798e-04 | 1.453e-01 | False |
| 10 | 1 478 281 551.10 | −6 044 588.13 | 2.780e-04 | 1.476e-04 | 1.238e-01 | False |
| 20 | 1 424 778 916.27 | — | 2.345e-04 | — | — | False |
| **21-24** | **—** | **—** | — | — | — | **local solve FAILURE** |
| 25 | 1 402 384 338.11 | −22 394 578.15 | 3.589e-04 | 1.229e-04 | 1.059e-01 | False |
| 50 | 1 310 811 413.64 | −3 132 624.25 | 2.227e-04 | 9.021e-06 | 8.736e-02 | False |
| 75 | 1 241 199 734.74 | −2 491 225.90 | 3.765e-04 | 1.697e-05 | 7.702e-02 | False |
| 82 | **1 224 357 544.76** | −2 354 118.80 | — | — | — | False |
| **83-100** | **—** | **—** | — | — | — | **local solve FAILURE, every cycle** |

`cycle_convergence` was **False on all 100 cycles**. Production's own convergence
logic never terminated. No value became nonfinite.

**Local NLP failures occurred on 22 of 100 cycles: 21, 22, 23, 24, and every
cycle from 83 through 100.** The last cycle with a successful local solve was
82. Recourse is therefore undefined from cycle 83 onward.

At cycle 82 the recourse was still falling by ~2.35e6 per cycle, with successive
changes `−2 427 289, −2 398 129, −2 383 128, −2 365 260, −2 354 119` — a decay
ratio near 0.99 per cycle. It was nowhere near a fixed point.

For scale: the trajectory is heading toward ~1.2e9, while the accepted
inherited-template oracle returns ~8.28e8. This is not a slow approach to the
accepted solution.

## 8 — Convergence-criterion slacks

Recorded per cycle as `threshold / observed` (1.0 sits exactly on the bound) for
all six consensus and three stationarity criteria plus the objective test; full
series in `p512a_trajectory_cap100.json`. `primal_pf` ranged
`1.089e-04 … 4.567e-01`, ending at `2.692e-04`. The binding behaviour is not
reported as a headline here because `cycle_convergence` was never true and 22
cycles have no valid local solution behind their residuals.

## 9 — Checkpoint polish (detached, on deep copies only)

The live trajectories were never perturbed.

| checkpoint | pre-polish `primal_pf` | polish | failed blocks | polished total |
|---|---|---|---|---|
| cycle 25 | 3.589e-04 | **FAILED** | `DSO7\|2035\|Spring` | — |
| cycle 50 | 2.227e-04 | success | none | 1 310 859 830.28 |
| cycle 75 | 3.765e-04 | success | none | 1 241 246 061.75 |
| cycle 100 | 2.692e-04 | success | none | 1 186 683 489.67 |

The cycle-50 polished total `1 310 859 830.281126` reproduces P5.11.1's
self-consistent base value `1 310 859 830.2811` to the cent, which independently
confirms the trajectory is the same one P5.11 built.

**None of these is a valid T0.** All three successful polishes land between
`1.19e9` and `1.31e9`, against the accepted `8.28e8`. As the stage states, a
successful polish of a non-converged checkpoint is diagnostic evidence only.

## 10 — Last-window trend ratios

`trend_ratio = median(cycles 91-100) / median(cycles 81-90)`:

| quantity | median 81-90 | median 91-100 | ratio | class |
|---|---|---|---|---|
| `primal_v` | 2.4637e-04 | 5.0273e-04 | 2.041 | increasing |
| `primal_pf` | 2.5467e-04 | 2.5698e-04 | 1.009 | flat |
| `primal_ess` | 7.6937e-05 | 8.4881e-05 | 1.103 | flat |
| `dual_v_mean` | 1.5463e-04 | 1.4432e-04 | 0.933 | flat |
| `dual_pf_mean` | 7.1544e-02 | 6.8944e-02 | 0.964 | flat |
| `dual_ess_mean` | 8.1537e-06 | 8.0196e-06 | 0.984 | flat |
| `state_step_norm_pf` | 2.3848e-04 | 2.2981e-04 | 0.964 | flat |

**These ratios must not be relied on.** Both windows lie entirely inside the
region where every cycle failed its local solves (83-100), so the residuals are
computed over consensus variables that no valid local solution supports. The
table is reported because the stage asks for it; it is descriptive of a
degenerate region and is not evidence about convergence. The production
convergence decision — never true, on any cycle — is the operative one.

## 11 — Exact stopping reason

The mandatory rule that fired:

> *Stop immediately if … Any local NLP block fails.*

First occurrence: **cycle 21**.

**Honest qualification about enforcement.** The rule could not be enforced live.
Production owns the ADMM loop internally and continues past a failed local solve,
recording `local_solves_ok = False` and proceeding; interrupting at cycle 21
would have required hooking a production function, which this stage forbids. The
failures were therefore detected from the recorded per-cycle diagnostics after
each run completed. **All evidence from cycle 21 onward is reported as
diagnostic material, not as a valid continuation of the trajectory.** No second
configuration, penalty, tolerance, anchor or initialization was attempted, and
nothing was restarted from a checkpoint.

### A correction to the P5.11 record

P5.11 reported that its cold RESCALED build "exhausted `num_max_iters = 25` with
`cycle_convergence = False` on every cycle". That is true, but incomplete.
**Cycles 21-24 of that same build contained local NLP failures**, which P5.11 did
not detect because its harness recorded `cycle_convergence` but not
`local_solves_ok`. The P5.11 diagnostic template was therefore produced by a
trajectory whose last five cycles included four failed local solves. This does
not change P5.11's verdict — it strengthens it, and it explains why that template
behaved as an unconverged, unreliable starting point.

## 12 — Evidence that no production file or setting changed

`shared_resources_planning.py`, `network.py`, `admm_parameters.py`,
`solver_parameters.py` and `data/SRP1/SRP1_params.json`: all unchanged.
Production `num_max_iters` remains 25; the horizon was raised only on
per-evaluation deep copies. No IPOPT option, ADMM setting, adaptive-ρ rule,
convergence tolerance, objective scaling, proximal regularization, ESSO
degradation, Benders or convex model was touched. ρ never changed and adaptive ρ
was never enabled. Nothing was committed, merged, pulled, fetched, pushed,
rebased, cherry-picked, reset or cleaned. Accepted P5.10 and P5.11 evidence is
byte-identical.

## 13 — Recommendation for exactly one subsequent stage

**Diagnose the local NLP failures at cycles 21-24 and 83-100 of the cold
RESCALED trajectory.** That is the single next stage, and it is a narrower
question than the one P5.12-A asked.

The reasoning: the cold RESCALED construction does not merely converge slowly —
it loses the ability to solve its own subproblems. The first failure at cycle 21
is transient (cycles 25-82 recover), but the breakdown from cycle 83 is
permanent. Neither pattern is a horizon problem, so extending the horizon
further, trying another penalty, or re-tuning would all be treating the symptom.
The question to answer is which blocks fail, with what IPOPT exit status, and
what distinguishes cycle 21 and cycle 83 from their predecessors. That is
answerable from the per-block solver logs with a read-only harness.

Two things that follow, and that I would not do without a separate decision:

- **The inherited-template oracle is unaffected.** Both P5.10 gates reproduced
  bit-identically here for the third consecutive stage. The accepted P5.10
  result stands with an explicit limitation: the RESCALED formulation is
  validated as a warm start from a CURRENT-built template, and is not validated
  on the cold construction path.
- **Template hardening should stop** until the failure is understood. The
  roadmap's "fresh converged RESCALED T0 A/B" is not reachable, because no
  converged cold RESCALED T0 exists to A/B against.

Repository preservation remains outstanding and outside my authorization: this
checkout is 8 commits ahead of origin, and the P5.9, P5.10, P5.11 and P5.12-A
reports, harnesses and evidence exist in a single unpushed local copy.

## 14 — Final repository state

```
branch : feature/derivative-free-planning
HEAD   : 808590d23ad250627be9ff9d4ae51d059cbdbcdb
tracked modifications : 0
upstream : origin/feature/derivative-free-planning (ahead 8, behind 0)
```

```
P5.12-A FAIL — cold RESCALED diagnostic encountered a numerical or provenance failure
```

Then stopping for planner review.
