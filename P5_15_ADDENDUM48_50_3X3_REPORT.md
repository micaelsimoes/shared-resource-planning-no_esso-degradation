# P5.15 Addenda 48–50: the 3 × 3 pair certifies; the storage does not pay; the objective was still descending at certification

**Planner report, 2026-09-27.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 48, 49, 50.
- **Frozen spec:** v36 `14bbddc7` (predecessor v35 `8aa98dbf`); pair campaign spec `231558f0`.
- **Objective convention on every table:** Q = certified `gross_operational_cost`, settlement excluded. Value = Q(x = 0) − Q(unit).
- **Status:** nothing is running. Stopped for review, as Addendum 48 ordered.

## Decisions needed

**1. The continuation (Addendum 50 Ruling 3) cannot run as ordered. Choose a route — author decision, every route exceeds 4 h.**

The ruling requires restoring the terminal state bitwise and continuing. That is not possible: persistence was refused for the pair, and **even persisted models would not have sufficed** — production's continuation path rebuilds the Anderson-acceleration memory empty, starts the tight tail *off* (the first continued cycle would solve at 1e-4, not 1e-6), and restarts the iteration counter.

| route | what it does | cost (Mac, sequential) |
|---|---|---|
| **A — replay and continue** | re-run each cell deterministically to its certification cycle in one process, gated **bitwise cycle by cycle** against the committed per-cycle record, then continue past certification | **≈ 22.8 h** for 30 further cycles; ≈ 18.5 h for 10 |
| B — restore-based | persist full state at N−1 and N, plus production changes for AA memory, tail flag and iteration offset | larger than A, and **refused by the v36 margin rule** (≈ 27 GiB against 20.75) |
| C — no continuation | report the value with the drift as an explicit uncertainty | none |

Route A needs a spec/harness change — allowing the run to continue past certification (raise `minimum_consecutive_converged_cycles`, or cap at N + 30) — so it becomes a **separately labelled run**, as Ruling 3 foresaw. Bitwise replay has been demonstrated only to cycle 3.

**Recommendation: Route A, 30 cycles, stopping at |ΔQ| < 500 €/cycle for three cycles.** The replay is most of the cost, so 30 cycles costs only ~4 h more than 10. It is the only way to measure the settling the certification rule does not guarantee; an instrumented replay also recovers the per-block ΔQ that the records truncated; and the drift could move R by more than its resolution (§3). It must run on the Mac: the bitwise gate compares against Mac-produced records, and cross-machine bitwise reproduction is not expected.

**2. G6 — the μ-floor criterion measures objective scaling, not depth (expert).**

Ruling 1's verification band (1, 5] is mis-derived, and node-7 shows why the whole floor criterion is uninformative:

- IPOPT's monotone update is `max(floor, min(0.2·μ, μ^1.5))`. Below μ ≈ 0.04 the **superlinear μ^1.5 term** dominates, so a solve at the last μ = 2.5059 × 10⁻⁹ is one update from the floor whatever its μ/floor ratio.
- **Whether a solve reaches the floor is set by IPOPT's objective scaling S.** Unscaled complementarity = scaled / S. On node-7's TSO and DSO7 blocks S = 0.001, so unscaled complementarity is 2.5–5.6 × `compl_inf_tol`; those solves cannot stop and the next update clamps them to the floor. On blocks with S = 0.009 the test passes and IPOPT stops one update above it.

**Recommendation:** future G6 = `Optimal Solution Found` plus the four terminal error metrics within the tail tolerances; μ/floor reported, with the note that it reflects S and says nothing about depth.

## Blocked on the author

Nothing. Nothing will run until decision 1 is made.

## Changed

- **3 × 3 pair certified**, 2026-09-26 10:20 → 2026-09-27 02:38 UTC (16 h 18 m), concurrency 1, (b) on, tight tail on, persistence off.
- Evidence committed: `47a54c89` (the 615-entry campaign manifest and 597-entry launch manifest, re-verified against disk; the commit carries a `TASKS.md` message because a Planner commit swept up the Worker's staged files — content verified byte for byte; the record is in `c2d4b4ce`'s message), `d757394e` (x = 0 diagnostics), `c2d4b4ce` (node-7 diagnostics).
- Benchmark code and tests for Addendum 49 written and committed, **nothing executed**: `a8c58da0`, `b9ba413d`.

## Found

### The pair

| | x = 0 | node-7 unit |
|---|---|---|
| certified at cycle | 72 | 69 |
| Q | 842,832,534.76 | 842,595,839.51 |
| bar | 13,954.38 | 4,010.57 |
| rule ten | 0.0475 | 0.0200 |
| terminal step | −4,005.67 | −1,688.16 |

| quantity | value | resolution / ratio |
|---|---|---|
| **value** | **236,695** | bar-sum 17,965 → 13.2× |
| **value − I** | **−81,262** | **−4.5×: the storage does not pay at 3 × 3** |
| **R** (vs 259,375.33) | **0.9126** | ratio resolution 0.1405 |

### Predictions against outcomes

| prediction | outcome |
|---|---|
| R = 0.9331 (prefix draw, recorded) | 0.9126 — difference 0.0205 = **0.15× resolution: consistent** |
| R ∈ [0.93, 1.09] | 0.9126 — below the floor by 0.12× resolution: a miss inside resolution |
| value − I negative and determinate | **held** (−81,262, 4.5×) |
| both cells certify in 60–95 cycles | **held** (72, 69) |
| 80/80 terminal solves at the μ floor (v35/v36) | **refuted** — x = 0 1/80, node-7 39/80 |
| node-7: G6 fails likewise (Add. 50) | **held** |
| node-7: median μ/floor ≈ 3 (Add. 50) | **held** (3.0628) |
| node-7: ≤ 3/80 at the floor (Add. 50) | **refuted** — 39/80, explained by S |
| H_row18: row-18 terms account for most of ΔQ | **refuted on both cells** — the charge *rises* while Q falls (−7.5 % and −13 % of ΔQ) |
| H_B: uniform, decreasing ΔQ across blocks | **refuted on both cells** |
| H_C: a few DSO blocks, longer solves | **mixed** — concentrated, but DSO7 *and* TSO; longer solves confounded by S |

### The drift

1. **It predates the tail, on both cells.** The same lead blocks move with the same signs from cycle 53 (x = 0) and cycle 50 (node-7); the tail engaged at 64 and 61. The consensus movement shows no break at tail engagement. Addendum 50's reading — "the tail's transient sits inside the certification window" — is **not supported**.
2. **Both cells follow one structure:** Anderson acceleration switches off, then a swing of several cycles, then a monotone descent. Certification fires when the PF dual residual ratio first falls below 1, while the objective is still converging. **The objective converges more slowly than the residuals.**
3. **The net descent is a small residual of large opposing block movements** — |ΔQ| is 0.1–39 % of the summed block movement — concentrated in a persistent set of DSO7 and TSO blocks, almost all in flexibility cost.
4. **Asymmetric:** aligned by the first cycle of descent, x = 0 falls 2–2.8× faster (−1,010, −2,057, −2,797, −3,398 against −362, −862, −1,310, −1,688). The steps are still growing, but their increments are shrinking — consistent with the hump Addendum 50 predicts.
5. **Rule ten cannot see it.** Its threshold (1e-4 relative, ≈ 84 k€) is ~20× the drift step; both cells read "well inside" (0.0475, 0.0200).

### What the drift does to the conclusions

- **The sign conclusion is robust.** x = 0 descends faster, so continued descent *shrinks* the value and makes value − I more negative. It would reverse only if node-7 later descended faster than x = 0.
- **The size of R is not.** *Projection, not a measurement*, holding both terminal steps constant: the value moves by its own resolution in 7.75 cycles; after 10 cycles value 213,520 (R ≈ 0.82), after 30 cycles 167,170 (R ≈ 0.64). The shrinking increments make this pessimistic; Addendum 50's hump predicts 20–60 k€ per cell and an inter-cell difference < 10 k€. **Only a continuation distinguishes them.**

### Convergence quality at the terminal round

- **159 of 160 terminal solves are converged by IPOPT's own definition** — `Optimal`, all four error metrics within the tail tolerances (worst: NLP error 0.905 of `tol`, complementarity 0.897 of `compl_inf_tol`).
- **The one exception:** node-7 DSO7|2034|Autumn exited `Solved To Acceptable Level` after five acceptable iterates, with complementarity at **2.93 × `compl_inf_tol`**. Six such exits occur across node-7's window, all on DSO7 blocks with S = 0.001.

## Not confirmed

- **Why S = 0.001 on the TSO and DSO7 blocks** — gradient-based scaling implies a maximum objective gradient near 1e5 at the start point; the term was not identified, since no model was loaded. It is the **same structural scaling asymmetry** the pin test found; whether it contributes to the drift asymmetry is not established.
- **Per-block ΔQ for 70 of 80 blocks** — the harness truncates to the top ten per cycle; unrecoverable from records, recoverable from an instrumented replay.
- **Bitwise replay beyond cycle 3.**
- **Whether `minimum_consecutive_converged_cycles` enters `evaluation_key`** — determines whether Route A's cells collide with the certified ones.

## Defects recorded, not fixed

- `boyd_terminal.json` reports `stopped_by: 'cap'` for both certified cells — a reporting mislabel (the label assumes certification occurs at the last cycle).
- A Planner commit swept a Worker's staged files into `47a54c89`. Content verified; the record is carried by `c2d4b4ce`. Planner commits now use `--only`.
