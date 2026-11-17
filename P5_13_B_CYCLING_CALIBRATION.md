# P5.13-B — cycling calibration: `cl_nom`, `dod_nom`, `soh_min`, `t_cal`

**No solver work. No production change. Code trace, git history and arithmetic only.**

Date: 2026-09-12. Scope: the degradation law
`es_degradation_per_unit[y_inv,y] * (2 * cl_nom * es_e_rated_per_unit[y_inv,y]) == es_avg_ch_dch_per_unit[y_inv,y]`
(`shared_energy_storage_data.py:509`) and the SoH chain at `:511-532`.

Provenance was recovered from git history, which the first draft of this report did
not attempt. The primary finding below **replaces** that draft's framing.

## F0 — PRIMARY FINDING: the (count, depth) pair was never maintained as a pair

Verified on this branch (`feature/derivative-free-planning`) with
`git log --full-history` over `shared_energy_storage.py`; the default history
simplification omits commits and must not be used for this question.

| Date | Commit | `t_cal` | `cl_nom` | `dod_nom` | `soh_min` | Changed | Message |
|---|---|---|---|---|---|---|---|
| 2024-04-08 | `4c6fef21` | 20 | **10000** | **0.95** | — | initial | Initial commit. |
| 2024-04-09 | `ceb76b9a` | 20 | 10000 | **0.90** | — | `dod_nom` | energy storage. Parameters update |
| 2024-04-11 | `1a318339` | 20 | 10000 | 0.90 | **0.10** | `soh_min` | Model update. Minimum SoH constraint added. |
| 2024-06-21 | `9dc782e6` | **10** | 10000 | 0.90 | 0.10 | `t_cal` | Update |
| 2024-06-21 | `0a7bad94` | **20** | 10000 | 0.90 | 0.10 | `t_cal` | Update |
| 2024-11-08 | `7b92c1fe` | **15** | 10000 | 0.90 | 0.10 | `t_cal` | Calendar life reduced to 15 years. |
| 2025-07-16 | `39f83c47` | **10** | 10000 | 0.90 | 0.10 | `t_cal` | Planning. Test. |
| 2025-07-29 | `7e3b3246` | **15** | 10000 | 0.90 | 0.10 | `t_cal` | vmag_sqr redefined to vmag. |
| 2025-10-13 | `3e5716cd` | **20** | 10000 | 0.90 | 0.10 | `t_cal` | Update. Energy-to-power constraints … |
| 2025-10-13 | `d5e667d2` | **15** | 10000 | 0.90 | 0.10 | `t_cal` | Calendar life. Update. |
| 2025-12-05 | `f952c969` | 15 | 10000 | **0.80** | 0.10 | `dod_nom` | ESS parameters. Update |
| 2026-08-03 | `edcfbd95` | 15 | 10000 | 0.80 | **0.50** | `soh_min` | Minimum SoH constraint added. |

**`cl_nom = 10000` dates from the initial commit of 8 April 2024 and has never been
changed.** It is an original modelling choice, not a debug leftover.

**If it was ever paired with a depth, that depth was `0.95`, not `0.80`.** `dod_nom`
moved `0.95 -> 0.90` the next day and `0.90 -> 0.80` twenty months later; the count was
not revisited at either change. `soh_min` did not exist when `cl_nom` was introduced,
and when it appeared it was `0.10` — it became `0.50` only on 2026-08-03.

So the defect is **not** that one of two readings of `cl_nom` is correct. It is that a
paired `(count, depth)` specification was never maintained as a pair, because the law
consumes only the count. This supersedes the earlier framing of two competing readings
(retained below as F2 for its arithmetic, not for its framing).

### F0.1 — the apparent calibration is an accident of two later edits

With today's constants, `10000` cycles of depth `0.80` implies a retention of
`exp(-8000/10000) = 0.449`, which sits close to today's `soh_min = 0.50` and makes the
implementation *look* calibrated. **It is not.** When `cl_nom` was introduced the depth
was `0.95` and `soh_min` did not exist; when it did, it was `0.10`. The `0.80`/`0.50`
pairing now in the file is the residue of two independent later edits (`f952c969`,
`edcfbd95`) made twenty months and a further eight months apart, neither of which
mentions the other quantity.

The near-agreement must be treated exactly as the two cancelling defects in F3: a
coincidence that must never be cited as justification.

### F0.2 — `t_cal` is a governance finding on its own

Seven value changes after the initial commit, cycling `20 -> 10 -> 20 -> 15 -> 10 -> 15
-> 20 -> 15`. Two of them are buried inside commits whose messages describe unrelated
work: `39f83c47` "Planning. Test." (`15 -> 10`) and `7e3b3246` "vmag_sqr redefined to
vmag." (`10 -> 15`). Calendar life was therefore altered as a side effect of unrelated
changes, twice.

Consequence: **any result depending on `t_cal` is interpretable only with its commit
pinned.** `t_cal` gates cohort activation (`shared_energy_storage_data.py:269`, `:432`,
`:500`) and the salvage remaining-life fraction (`:712-717`), so this is not a cosmetic
parameter. This is an independent argument for moving the constants into a data file
under review.

### F0.3 — the manuscript's 70% is DIVERGENT, not stale

`EXPERT_REVIEW.md:51` records that PDF p.25 states a 70% minimum SoH while the code
defaults to 50%, and that Table 7 reports 49.62%.

The value `soh_min = 0.70` exists — introduced by `a3c76922`, 2026-07-29, "Default
minimum SoH updated." — but **it is not on this branch.** `a3c76922` is on
`paper_revisions` and is *not* an ancestor of HEAD (`git merge-base --is-ancestor`
returns false). The branches diverged at `285cbb42`, 2026-07-28, "Debug."

| Branch | `soh_min` history after the divergence | value today |
|---|---|---|
| `paper_revisions` | `0.10 -> 0.70` on 2026-07-29 (`a3c76922`) | **0.70** |
| `feature/derivative-free-planning` (all P5.x work) | `0.10 -> 0.50` on 2026-08-03 (`edcfbd95`) | **0.50** |

So the two values are **concurrent, not sequential**: the manuscript branch says 0.70
today, and the branch that produced every P5.x result says 0.50. The paper's 70% claim
is supported by the code on its own branch and contradicted by the code that generated
the numbers. Table 7's 49.62% is consistent with the 0.50 floor binding, i.e. with this
branch.

This is a stronger finding than "stale": a live branch divergence in a parameter that
enters both the feasibility floor (`:529`) and the salvage valuation (`:680`, `:748`,
`:787`). It also means the 0.50 figure has its own undocumented origin — `edcfbd95`'s
message, "Minimum SoH constraint added.", does not mention changing a default that had
stood at 0.10 for over two years.

The 60%/80% sensitivity cases remain unsupported on either branch: they would have
required source edits, and no record of such runs exists.

## F1 — the ageing constants are not case data

`shared_energy_storage.py:18-21` holds all four; a repository-wide search finds no other
assignment, and no case file supplies them. `dod_nom` is read nowhere in the repository.
The ageing model is therefore identical for every case, and no case definition can
change it. (Superseded in practice by the authorized parameter move; recorded because it
is the state in which every existing result was produced.)

## F2 — the law assumes a reference DoD of 1.0 (arithmetic retained; framing superseded by F0)

The throughput accumulator (`:479-490`) is **cell-side energy**
(`eff_ch * pch * dt + pdch * dt / eff_dch`), so one **full-depth** cycle of a battery of
rated energy `E` contributes exactly `2E`, efficiencies already accounted for. The
normalization `2 * cl_nom * E` in `:509` is therefore the throughput of `cl_nom` cycles
**at DoD = 1.0**, while the data object declares the nominal depth as `0.80`.

Treating `10000` as a count at `dod_nom` would make the correct normalization
`2 * cl_nom * dod_nom * E`, a factor of `1/0.8 = 1.25`. Per F0, this is one arm of a
pair that was never maintained, not a choice between two intended readings.

## F3 — `cl_nom` functions as a decay constant, not as cycles-to-EOL

The SoH chain is multiplicative per day: `soh = 1 - delta` (`:513`/`:515`),
`soh_cumul[y] = soh_cumul[y-1] * soh ** (365 * num_years)` (`:524`/`:526`), exponent
`1825` for SRP1's `num_years = 5`. With `d = EFC_per_day / cl_nom`, retention after `N`
days is `(1-d)^N ~ exp(-EFC_total / cl_nom)`.

- At exactly `cl_nom` equivalent full cycles the model retains **`exp(-1) = 36.8%`**.
- SoH `0.50` at `-ln(0.5) * cl_nom` = **6931 EFC**; SoH `0.80` at **2231 EFC**;
  both independent of cycling rate.

| EFC/day | `d` | 5-yr retention | 15-yr retention | reaches 0.80 | reaches 0.50 |
|---|---|---|---|---|---|
| 0.5 | 5e-5 | 0.9128 | 0.7605 | 12.2 yr | 38.0 yr |
| 1.0 | 1e-4 | 0.8332 | 0.5784 | 6.1 yr | 19.0 yr |
| 2.0 | 2e-4 | 0.6942 | 0.3345 | 3.1 yr | 9.5 yr |

**Correction to the earlier recommendation.** The first draft advised defaulting to the
manufacturer-count reading because it shortens claimed life and is therefore
conservative. That conflated two questions which point in **opposite** directions:

- *Depth* — count at `dod_nom` versus equivalent full cycles: a factor `1.25`
  **towards shorter** life;
- *End-of-life retention* — the SoH to which the count was quoted. Manufacturer counts
  are conventionally quoted to 70-80% SoH, not 50%. Under that convention the
  implemented law is far **more pessimistic** than a datasheet, reaching 0.80 at only
  2231 EFC against 8000.

On the conventional reading the model **understates** life by roughly `3.6x`. The
conservative-default instruction was sound only for the depth question and is withdrawn
as general advice.

## F4 — a naming defect in the same rows

The comment at `:511` labels `es_soh_per_unit` the "Annual SoH". It is the **daily**
retention factor, raised to `365 * num_years` at `:524` — a factor-365 trap. Bounds
`(0,1)` and `initialize=1.00` are consistent with the daily reading; only the comment is
wrong.

## F5 — a scaling observation, not a finding

At realistic cycling `es_degradation_per_unit ~ 1e-4` against a bound of `1.0`,
multiplied by `2 * cl_nom * E_rated = 20000 * E_rated` in a bilinear equality at `:509`:
an O(1e4) coefficient against an O(1e-4) variable. The P5.4-H series rescaled the
complementarity rows for this kind of reason. Recorded as a **candidate** row-scaling
item for the NLP-stability work, with no claim of contribution to the cycle-21 failure.

## The calibration decision — reframed as a triple

Do not attempt to reconstruct a single intended reading: F0 shows there was never a
maintained one. Make the calibration explicit instead, as a triple

- `N` — cycle count,
- `D` — reference depth of discharge,
- `R` — end-of-life retention,

with the law deriving its characteristic constant

```
k = N * D / (-ln R)
```

and using `2 * k * E_rated` in place of `2 * cl_nom * E_rated`.

Two properties make this structurally unbreakable in the way the present code is not.
First, the law would consume all three quantities, so neither `dod_nom` nor `soh_min`
could drift again without effect — which is exactly how the present pair broke. Second,
the identity `EFC_to_R = -ln(R) * k = N * D` holds for every triple, so each candidate
below reaches **its own** `R` at exactly `N * D` cycles: the candidates differ only in
which retention is reached at that point, and therefore in the steepness of the whole
curve.

This also resolves the "delete `dod_nom` or use it" question by using it.

### Candidate calibrations

`k = N * D / (-ln R)`; `k = 10000` is the present behaviour.

| # | `N` | `D` | `R` | implied `k` | `k` / 10000 | Basis |
|---|---|---|---|---|---|---|
| C1 — as implemented | 10000 | 1.00 | 0.368 | **10000** | 1.000 | status quo; the implicit triple the current law encodes |
| C2 — conventional datasheet | 10000 | 0.80 | 0.80 | **35851** | 3.585 | manufacturer counts quoted to 80% SoH |
| C3 — consistent with this branch's floor | 10000 | 0.80 | 0.50 | **11542** | 1.154 | `soh_min = 0.50` read as the EOL the count refers to |
| C4 — the manuscript branch's floor | 10000 | 0.80 | 0.70 | **22429** | 2.243 | forced onto the table by F0.3; `paper_revisions` says 0.70 |

C4 is not a fourth opinion: the branch divergence means the manuscript currently asserts
a floor that implies this `k`, so choosing C2 or C3 also decides what the paper must say.

### Effect on existing SoH trajectories

Per cohort at 1.0 EFC/day (`delta_day = EFC/k`, exponent `365*5 = 1825` per row):

| Candidate | 5 yr | 10 yr | 15 yr | EFC to 0.50 | EFC to 0.80 |
|---|---|---|---|---|---|
| C1 (`k` = 10000) | 0.8332 | 0.6942 | 0.5784 | 6931 | 2231 |
| C2 (`k` = 35851) | 0.9504 | 0.9032 | 0.8584 | 24850 | 8000 |
| C3 (`k` = 11542) | 0.8537 | 0.7289 | 0.6223 | 8000 | 2575 |
| C4 (`k` = 22429) | 0.9219 | 0.8498 | 0.7834 | 15547 | 5005 |

At 0.5 EFC/day the 15-year retentions are C1 `0.7605`, C2 `0.9265`, C3 `0.7888`,
C4 `0.8851`.

Reading: under C1 a cohort cycled once daily loses **42%** of capacity over the 15-year
horizon and approaches the `0.50` floor; under C2 it loses **14%** and the floor is
never approached, so the floor constraint (`:529`) would stop binding and the salvage
term (`:680`) would change materially. C3 is within 8% of present behaviour at 15 years;
C2 and C4 are not.

**Anything other than `k = 10000` changes the degradation term and therefore the
objective**, so this decision is **blocking** against the ranking-baseline re-derivation,
on the same argument as the throughput and penalty items.

The choice of `(N, D, R)` is the author's, and is a documented modelling decision rather
than a reconstruction.
