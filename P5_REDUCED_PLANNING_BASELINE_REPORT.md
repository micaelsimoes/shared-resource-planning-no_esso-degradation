# Stage P5 — Reduced planning baseline with the accepted P4 formulation (executed)

Integration/stability baseline. **No formulation, solver, ADMM, Benders or ESS
setting was changed**; the only code added is the diagnostic runner
`p5_reduced_planning_baseline.py`, which tees production stdout to a log and
post-hoc parses the INFO/WARNING lines production already prints.

## 1. Checkpoint (§1)

| | |
|---|---|
| **P4.6-B2 commit** | `f77d829359ffd873367f556882546bc2dcc8ec99` |
| **Production files in B2** | `definitions.py`, `model_construction_helpers.py`, `network.py` |
| **Tracked working tree at run time** | **clean** (`git_tracked_dirty: []`, recorded inside the report JSON) |

Commit separation is intact — each accepted change is its own commit:

| Commit | Content |
|---|---|
| `0171f451` | shared-ESS `sess_snet_def` normalization (P4.2) |
| `441244e3` | OP1 shared-ESS data layout |
| `b6342cac` | OP1 scenario seed |
| `231511cb` | **B1** ordinary-ESS load-positive sign correction |
| `5639f397` | OP1 parameter alignment |
| **`f77d8293`** | **B2** ordinary-ESS `ess_snet_def` normalization |

No result directories, logs, pickles, `.env`, `.DS_Store`, `__pycache__` or
unrelated files were committed.

## 2. Exact command and configuration (§2, §11)

```
SharedResourcesPlanning('data/SRP1', 'SRP1.json').run_planning_problem()
```

Run with `/opt/anaconda3/envs/opf_env_py311/bin/python -u p5_reduced_planning_baseline.py`
at git `f77d829359ff`.

Effective configuration **confirmed from the live repository at run time**:

| | |
|---|---|
| Years | 2025, 2030, 2035 |
| Representative days | Spring, Summer, Autumn, Winter |
| Instants / market scenarios | 24 / 1 |
| Random seed | **2026** |
| **Scenario checksum** | **`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`** |
| TSO | `case9` |
| DSOs | `case33_1`@5, `case33_2`@7, `case33_3`@9 |
| Benders | `tol_abs` 10000, `tol_rel` 0.005, `num_max_iters` 6 |
| ADMM | `num_max_iters` 25 |

The scenario checksum **exactly matches the reproducibility identity recorded
in `REVISION_CONTEXT.md`**, confirming this is the established research
baseline and not a drifted configuration.

Raw evidence: `data/SRP1/Results/P5/p5_report.json`,
`data/SRP1/Results/P5/p5_console.log`.

## 3. Outcome

**The planning workflow ran to a clean, production-defined stop, but it did not
complete a full Benders sequence.** It terminated at outer iteration 2 with
`termination_reason = operational_initialization_failure`.

| | |
|---|---|
| Outer iterations executed | **2** of a permitted 6 |
| Termination classification | **`operational_initialization_failure`** |
| Benders converged | **No** (`Convergence not obtained!`) |
| Production execution time | **562.26 s** |
| Wall clock (incl. data load) | 637.5 s |
| Process exit | clean (no crash, no `SystemExit`, no exception) |

## 4. Outer-iteration table (§5)

| Iter | Candidate source | Master estimate | Alpha | Investment | Gross recourse | Salvage | Net recourse | Candidate total | Incumbent (UB) | Gap | ESSO violation | Incumbent updated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `master_solution` | −1 000 000 000.00 | −1 000 000 000.00 | 0.00 | 848 451 891.26 | −0.00 | 848 451 891.26 | **848 451 891.26** | 848 451 891.26 | 217.86 % | −0.000031 | **Yes** (first incumbent) |
| 2 | `positive_bootstrap` | N/A | N/A | 50 000.00 | N/A | N/A | N/A | N/A | 848 451 891.26 | N/A | N/A | No — operational initialization failed |

**Candidate investment decisions**

- **Iteration 1** — the zero-investment candidate: `S = 0.000 MVA` and
  `E = 0.000 MVAh` at **all** shared-ESS nodes (5, 7, 9) in **all** years
  (2025, 2030, 2035).
- **Iteration 2** — `positive_bootstrap`, identical at all three nodes:

  | Node | 2025 S / E | 2030 S / E | 2035 S / E |
  |---|---|---|---|
  | 5, 7, 9 | 0.011 MVA / 0.021 MVAh | 0.021 MVA / 0.043 MVAh | 0.032 MVA / 0.064 MVAh |

  Investment cost 50 000.00. This candidate was never evaluated — the
  operational initialization failed before ADMM started, so no recourse,
  salvage, candidate total or cut exists for it.

**Cut / sensitivity information generated.** One cut-bearing evaluation
(iteration 1). Iteration 2 produced **no** cut: production explicitly reports
*"No ADMM cycle or formal Benders feasibility cut is available; stopping the
outer loop."*

**Bound qualification (preserved deliberately).** `Master = −1 000 000 000.00`
is the initial `alpha` placeholder, **not** a valid global lower bound. The
implementation itself states the local cuts are not global lower bounds and
stops "without claiming optimality". Nothing in this report should be read as a
rigorous Benders lower bound or an optimality gap in the classical sense; the
217.86 % "Gap" is the implementation's own progress indicator against that
placeholder.

## 5. Operational subproblem reporting (§6)

| | |
|---|---|
| Operational evaluations attempted | **2** |
| Evaluations that reached ADMM | **1** (iteration 1) |
| Initialization failures | **1** (iteration 2) |
| ADMM runs converged | **1 / 1 started** |
| ADMM cycles to convergence | **9** |
| ESSO solver failures | **0** |
| Local primary NLP failures | **3** |
| Recovery attempts | **0** |
| Recoveries succeeded | **0** |
| Persistent-for-cycle failures | **3** |

**Iteration 1 final state (cycle 9):** `Primal residuals ok!`,
`Dual residuals ok!`, `Recourse stationarity ok!` — converged cleanly. The
19 primal / 8 dual / 7 stationarity "failed" messages in the log are
intermediate-cycle progress checks from cycles 1–8, not terminal failures; the
final cycle-9 block contains none of them, nor any `[RECOURSE JUMP]` or
`[SLACK COMPONENTS]` line.

**Voltage-slack usage.** `node_P`, `node_Q`, `branch_ij`, `branch_ji`, `flex_P`
and `flex_Q` slack components are **identically zero throughout the entire
run**. Voltage slack is nonzero only in intermediate ADMM cycles
(e.g. `−7.23e-03`, `+1.32e-01`); **no slack line appears in the converged
cycle-9 block**, i.e. slacks are inactive at the accepted iteration-1 solution.

**The three local failures** (all in the iteration-2 initialization, all
`maxIterations`, all `warm_start=False`):

| Network | Node | Year | Day | Termination | Recovery |
|---|---|---|---|---|---|
| `case33_1` | 5 | 2030 | Winter | `maxIterations` | not attempted |
| `case33_1` | 5 | 2035 | Winter | `maxIterations` | not attempted |
| `case33_3` | 9 | 2025 | Summer | `maxIterations` | not attempted |

Recovery was **not** attempted because the production recovery path
(`_is_recoverable_network_failure`) triggers only on `internalSolverError`;
`maxIterations` is outside that class. This is existing, unmodified behaviour —
not a defect introduced here — but it is the mechanical reason a single
`maxIterations` local solve halts the whole outer loop.

**Previously problematic families — did they reappear?**

| Family | Occurrences in P5 |
|---|---|
| DSO `case33_2` / node 7 | **0** |
| TSO `case9` shared-ESS interface | **0** |

**Neither previously problematic family reappeared.** The three failures form a
*different* pattern: `case33_1` (node 5) in **Winter** of the later investment
years, and `case33_3` (node 9) in **Summer 2025**.

## 6. Comparison against the accepted P4.5 smoke (§7)

**Not directly comparable — stated plainly rather than forced.** P4.5 evaluated
the candidate `node 7, investment_year 2025, s_inv = 1.00 MVA,
e_inv = 3.00 MVAh`. **P5 never evaluated that candidate**: iteration 1 was the
all-zero candidate and iteration 2 was the tiny `positive_bootstrap` candidate
that failed initialization. The requested direct comparison therefore does not
apply to this run.

For context only (different candidates, so not a like-for-like result):

| | P4.5 (node 7, 1.00 MVA / 3.00 MVAh) | P5 iteration 1 (zero investment) |
|---|---|---|
| ADMM cycles | 12 | **9** |
| Primary local failures | 2 | **0** |
| Persistent-for-cycle | 2 | **0** |
| ESSO failures | 0 | 0 |
| Voltage slacks active at convergence | 0 | **0** |

The one ADMM evaluation P5 did complete was cleaner than the P4.5 reference —
but on a strictly easier (zero-investment) candidate.

## 7. Acceptance questions (§9)

1. **Does the reduced planning workflow complete?** **No.** It executes and
   terminates cleanly under production control, but stops after 2 of 6 outer
   iterations without converging.
2. **Termination classification?** `operational_initialization_failure`.
3. **Investment solution selected?** Only the **zero-investment** incumbent from
   iteration 1 (0.000 MVA / 0.000 MVAh at nodes 5, 7, 9 in 2025/2030/2035). No
   positive investment was ever successfully evaluated or accepted.
4. **Planning objective and decomposition?** Incumbent
   **848 451 891.26** = investment **0.00** + gross recourse
   **848 451 891.26** + salvage **−0.00** (net recourse 848 451 891.26). ESSO
   feasibility violation −0.000031, within `BENDERS_FEASIBILITY_TOLERANCE`.
5. **Operational evaluations performed?** 2 attempted; 1 reached and completed
   ADMM; 1 failed at initialization.
6. **Local primary failures?** **3**.
7. **Recoveries?** **0** attempted, 0 succeeded (all three were `maxIterations`,
   outside the recoverable class).
8. **Persistent-for-cycle failures?** **3**.
9. **Repeated node/network/year/day pattern?** Yes, a coherent one:
   `case33_1`/node 5 in **Winter** (2030 and 2035) plus `case33_3`/node 9 in
   Summer 2025 — all during the *same* iteration-2 initialization. It is **not**
   the previously diagnosed `case33_2`/node-7 or TSO `case9` family (0 each).
10. **Does every required ADMM operational evaluation converge?** Every ADMM run
    that started converged (1/1, in 9 cycles). The second evaluation never
    started, so this is "yes, for those that ran" — not a clean yes overall.
11. **Voltage slacks inactive at accepted solutions?** **Yes.** No slack line in
    the converged cycle-9 block; node/branch/flex slacks identically zero
    throughout.
12. **Does recourse stationarity pass?** **Yes** at the accepted solution
    (`Recourse stationarity ok!` at cycle 9). Intermediate-cycle stationarity
    checks failed while converging, which is expected.
13. **Consistent with the local-cut/nonconvex methodology?** **Yes.** The
    implementation refuses to add a cut without a converged operational
    evaluation, does not claim optimality, and preserves the "local cuts are not
    global lower bounds" qualification. The observed behaviour matches that
    contract exactly.
14. **Any evidence the new ESS normalizations introduced a regression?** **No
    direct evidence**, and two points argue against it: both previously
    problematic failure families are entirely absent, and the one completed ADMM
    evaluation converged cleanly in 9 cycles with zero local failures and zero
    active slacks. The three failures occurred in a regime the P4 validation
    never exercised (see §8) — that is an open question, **not** a demonstrated
    regression.

## 8. Unresolved issue requiring planner review

**A single unrecovered `maxIterations` local solve halts the outer loop.** The
`positive_bootstrap` candidate could not be initialized, so P5 cannot produce a
completed reduced planning baseline. Per §8/§10 of the P5 specification I did
**not** retune anything, did **not** patch the formulation, and did **not**
begin a conditioning investigation.

One factual observation is recorded for the planner, deliberately without
causal claim or follow-up analysis:

> The iteration-2 bootstrap candidate installs **very small** shared-ESS
> capacities — `S = 0.011 MVA` in 2025, i.e. `1.1e-04 p.u.` on `baseMVA = 100`.
> Under the accepted P4.2 shared-ESS normalization
> `kappa_e = 1/S_scale_e`, that corresponds to `kappa ≈ 9091`. The P4.3–P4.5
> validation range was `s = 0.01`–`0.02 p.u.` (`kappa = 100`–`50`), so this
> candidate sits roughly two orders of magnitude outside the validated scale
> regime. Whether that is related to the three `maxIterations` failures is
> **not established** — the failures are in DSO local solves at nodes 5 and 9,
> and no diagnostic was run. This is flagged only so the planner can decide
> whether the next stage should examine the bootstrap-capacity regime, the
> `maxIterations` recovery policy, or neither.

Frozen local state: the existing framework wrote IPOPT logs for each failing
solve (`data/SRP1/Results/Logs/optim_log_case33_1_2030_Winter.log`,
`…_case33_1_2035_Winter.log`, `…_case33_3_2025_Summer.log`). No frozen-pickle
capture was triggered, since that path is driven by the operational-planning
snapshot callbacks rather than the initialization stage.

## 9. Deferred items (§10)

Not investigated in this stage: active-energy SOC/cycling correction; calendar
degradation; salvage redesign; further ordinary-ESS sign-convention work; OP1
permanent parameterization; MA57/MA97 comparisons; ADMM retuning; the full
experiment matrix.

Recorded verbatim:

> `Deferred configuration review: decide later whether the SRP1-aligned OP1 operational parameters remain permanent or should be replaced by a stable OP1-specific configuration.`

---

```
P5 PARTIAL — planning completed with issues requiring planner review
```
