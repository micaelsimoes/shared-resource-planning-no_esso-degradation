# P5.16 T-perf A2 / A2b: warm-start barrier A/B (`mu_init`). The six declared solves ran over two launches (a2: 1, a2b: 5) under ruling (a). Three of the four restated objective gates show a real difference (ratio 2.45 / 41.3 / 23.0)

Worker report for the Planner, 2026-10-31. Authority: the author's "Decision 4" (≤ 6 declared solves),
`P5_16_TPERF_SPEC_v1.md`, and the Planner's ruling (a) of 2026-10-31. Baseline: `P5_16_TPERF_REPORT.md` (Part A). No
production code was changed. Nothing here enters a table. The first launch (a2) is reported unchanged in the section
"A2 (first launch)" below.

**The six declared solves span two launches (ruling (a)).**
- **a2** ran 1 solve: `tso_c1_ident_mu`, done. Manifest `c22e94d3ac566783afcbeafa68a4d3a25e1f4a517149f027c390a58183be8d72`,
  `a2.json` `e87d67b2aefe053ba1deec04f237f0c673530fdec6adc120b7585e6d8bdb178b`.
- **a2b** ran the 5 that a2 did not: TSO c2, c3; DSO c1, c2, c3. Its bounded guard permitted exactly 5, and
  `verify(5)` returned no failures.
- **TSO c1 was not re-run.** a2b read its record by reference. Before unpickling anything, a2b checked a2's manifest
  against the ruling's hash, and checked `a2.json` and c1's IPOPT log against that manifest. The split is recorded in
  `a2b.json` → `continuation.split`, in `a2b/predictions.json`, and in the a2b manifest.

**Objective convention.** "f" is the block's active ADMM objective, `p58_rescaled_admm_objective`. It is a local block
objective, not `gross_operational_cost` and not `net_operational_recourse`, and it is compared only between solves of
the same block.
- Gates use the Pyomo value at the loaded point, as a2 did. IPOPT's final unscaled objective is reported beside it;
  the gate ratios agree to 4 significant figures under either.

**Instance.** The same as Part A and a2:
- campaign `s53_w86_tail_recert`, cell `n7_4h_e1`;
- eval key `ca8927e75d628bd1…`, candidate `db77e1549af855bc…`, certified at cycle 112;
- `certified_models.pkl` sha256 `fe06ca48…efda04`, verified before unpickling;
- blocks: TSO 2025 Spring (`case9`) and DSO node 7 2025 Spring (`case33_2`);
- options: the pickled pre-tail production options, plus `mu_init` 1e-6 on c1 and c3;
- perturbation on c2 and c3: +10 × the terminal mismatch, unchanged from a2.

## Blocked on Planner

1. **Three of the four restated objective gates returned "real difference". Ruling (a) calls that a stop, so the A/B
   now needs a ruling.**

   | gate | \|Δf\| | bar = max terminal unscaled compl. | ratio | verdict | rel. diff (beside, not gated) |
   |---|---|---|---|---|---|
   | TSO c1 (a2) vs Part A TSO warm limit | 6.776e-5 | 7.189e-5 (c1) | **0.943** | indeterminate (inside stopping slack) — pass | 1.30e-9 |
   | TSO c3 vs c2 | 1.458e-4 | 5.952e-5 (c2) | **2.45** | real difference | 2.79e-9 |
   | DSO c1 vs Part A DSO warm limit | 7.124e-4 | 1.724e-5 (c1) | **41.3** | real difference | 1.31e-8 |
   | DSO c3 vs c2 | 4.615e-4 | 2.006e-5 (c3) | **23.0** | real difference | 8.46e-9 |

   - **Signs.** In the DSO pairs, the `mu_init` solve ends higher, which is worse for a minimisation: c1 +7.1e-4,
     c3 +4.6e-4. On the TSO, c3 ends lower than c2 by 1.46e-4.
   - **What the operational bar omits.** The `mu_init` DSO solves stop with **35–38× larger terminal unscaled dual
     infeasibility** than the μ = 0.1 solves: 1.35e-3 (c1) and 1.50e-3 (c3), against 3.56e-5 (Part A) and 4.29e-5
     (c2). Constraint violation follows the same pattern: 2.6e-8 and 2.7e-8, against 9.6e-12 and 1.4e-11.
     - Ruling (a)'s full bar contains a "dual infeasibility × typical step" term. The operational form, which the
       ruling itself specified, uses complementarity only.
     - For the dual term to cover |Δf|, the "typical step" would have to be 0.53 (DSO c1), 0.31 (DSO c3/c2) and
       0.048 (TSO c3/c2), in variable units.
     - The measured ∞-norm distance between DSO c3 and c2 is only 5.6e-4 p.u. The ∞ × ∞ product therefore does not
       cover |Δf|. I could not compute an ‖dx‖₁ reading, because primal values were not serialized.
   - **How I applied "a stop".** I did not stop the run on a gate. Ruling (a) lists the stops (failed solve,
     restoration, hash mismatch), and the harness stopped on those only. I read the gate "stop" as a stop for review,
     which is this item.
     - Consequence: the DSO solves ran after the TSO c3/c2 gate had returned 2.45, and they are still within the
       declared 5.
     - If a run-level stop was meant, then those three DSO solves ran against the ruling.
   - **The variable rule.** TSO c3/c2: max |dx| 7.93e-5 p.u. ≤ 1e-4, so the two **agree**. DSO c3/c2: 5.65e-4 p.u.
     > 1e-4, so they **do not agree**. Both are reported, and neither is a stop.

   **Options.**
   - **(a) Recommended: accept as recorded.** A4 is recorded FAILED, and R7 is "not yet" (below). Refer the bar
     question (whether the dual-infeasibility term should enter, and in which norm) to the Advisor. Zero solves.
   - **(b)** Settle "stopping slack versus a different limit" with a tighter-tolerance pair on the DSO, for example
     c2 and c3 at a smaller `tol` or `dual_inf_tol`. This would take solves beyond Decision 4's six, so it needs the
     author.
   - **(c)** Take no further action on R7 until Part B's Studio timing exists.
2. **Commit.** I staged `p516_tperf_a2.py` (now `cb3ca3a9…`) and this report by filename. I committed nothing.
   - **a2's harness bytes.** a2 ran harness bytes `019d89a9…`. Re-staging replaced the previously staged blob
     `47834395a966…`. Those bytes are preserved in the evidence root as `a2b/a2_harness_as_run_019d89a9.py.txt`:
     recovered from that blob, checked against a2's manifest entry, and hashed in the a2b manifest.
   - **a2b evidence.** `data/SRP1/Results/P516TPERF/a2b/` and `a2b_launch_*.log` are untracked, as `data/` is.
     `manifest_sha256.json` has sha256 `8dc01e26217f339ea3a67f86847ba8071dfefef19f3ac3df71068c17430e2a3d` and
     covers 36 files. `a2b.json` has sha256 `efab2e519752e5fc33f140b9189f4523a0ecb5d58f8386fc7f2309b23e159ddf`.
3. **Provenance of the Part A harness: a recorded mismatch, not a fault (ruling (a)).**
   - a2b imported `p516_tperf_profile.py` from the working tree, sha256
     `2c85fc936ccf034de7cfe324da7342918dee9508d2839c3cbca91c733fac046c`. The Part B Worker had edited it again since
     a2, so this matches neither the committed `d013452f…` nor a2's `b8dbb6aa…`.
   - The harness hashed the file before and after the import; the two hashes were equal.
   - Against HEAD, every function and class a2b uses is source-identical except `capture_checklist`. That function
     gained the keyword `deferred=()`, and its default path is unchanged.
   - Recorded in `a2b.json` → `provenance.part_a_harness_import`.
4. **"Both reports".** I took this to mean the a2 and a2b sections of this file, and recorded the split in both. I did
   not edit `P5_16_TPERF_REPORT.md` (Part A).

## Changed (a2b)

- **`p516_tperf_a2.py` gained a `--continue-from a2` mode.** Without the flag, a2 mode behaves as before, apart from
  added refusals and records.
  - **Setup.**
    - Declares 5 solves (`verify(5)`).
    - Writes to `P516TPERF/a2b/`, write-once.
    - Refuses to write under `a2/`, `studio_dry/`, `studio_dry_equiv/`, `local/`, `studio/`, `azero/` and `P516D0/`.
  - **TSO c1 by reference.**
    - Prepares the TSO c1 clone but does not solve it; its options still enter the comparator check, which was "ok".
    - Verifies a2 by reference, as described above.
    - Copies a2's harness bytes out of the git blob.
    - Evaluates the restated gate on a2's TSO c1, with zero solves.
  - **New capture: `final_statistics`.** It reads IPOPT's terminal block, the text after the unique
    "Number of Iterations" line: objective, dual infeasibility, constraint violation, variable-bound violation,
    complementarity and overall NLP error, each scaled and unscaled.
  - **`slack_gate` (ruling (a)).**
    - Test: |Δf| ≤ the maximum of the two solves' terminal unscaled complementarity. Ratio < 1 passes.
    - Recorded but not gated: the 1e-9 relative figure, dual infeasibility and constraint violation.
    - Reported beside: the same gate computed on IPOPT's log objectives.
  - **Variable agreement for c3 vs c2.** Agreement means max |dx| ≤ 1e-4 p.u. The test is reported, not a stop.
  - **Stops and predictions.**
    - The c1-gate stop is removed in this mode. The stops are a failed solve or per-case cap block, restoration, and
      an input or reference hash mismatch.
    - Predictions are evaluated over all six solves, with TSO c1 from a2. A4 is evaluated in ruling (a)'s form, and
      its original 1e-9 / 1e-6 form is reported beside it.
  - **`--manifest`** in this mode also hashes a2's manifest, `a2.json` and c1's log.
- **Dry runs (zero solves).**
  - **`/tmp/a2b_dry1`: exit 1, with zero solves (guard 0/0/0/0).** My first `final_statistics` failed its own
    uniqueness check. At `file_print_level` 6, IPOPT prints the same-named lines at every iteration, giving 24
    "Objective" lines in Part A's TSO log.
    - The error was raised before unpickling. I fixed the parser to read only the block after "Number of
      Iterations".
  - **`/tmp/a2b_dry2`: exit 0.**
    - Guard: 0 permitted, **5 blocked**, 0 executions; `verify` clean.
    - Comparator check "ok"; capture checklist clean on the five cases.
    - Parser values: Part A TSO compl 5.9151e-5, DSO 1.1266e-5; a2's c1 7.1888e-5. These equal the log lines
      quoted in the a2 section.
- **Launch.**
  - **Waited for the lock.** The Part B Worker's `p516_tperf_profile.py --restriction-equivalence` and then
    `--studio-dry` held the shared T-perf lock, and the lock refuses rather than waits. I waited until neither was
    running.
  - **Command** (from the repository root), attached and alone, both streams captured, `set -o noclobber`:
    `python -u p516_tperf_a2.py --continue-from a2 > data/SRP1/Results/P516TPERF/a2b_launch_stdout.log 2> …/a2b_launch_stderr.log`
  - **Result:** exit 0, status `complete`, 11:15:36–11:15:51. The stderr log is 0 bytes.
  - **Guard:** `{'permitted_solve': 5, 'permitted_exec': 5, 'blocked_solve': 0, 'blocked_exec': 0}`;
    `verify(5)` returned []; per-case cap blocks 0.
  - **State:** git HEAD `7324aa9c`; production files clean.
  - **Concurrent load:** `p516_d0_t01c.py --child` was running (the D0 Worker). Load average 3.71 / 4.07 / 3.26.
    Thread caps were as in production.
  - **Then** `--continue-from a2 --manifest`: 36 files.
- **Evidence root** `data/SRP1/Results/P516TPERF/a2b/`:
  - `a2b.json`;
  - `predictions.json` (`cc681edc…`, written before the first solve);
  - `run.log`;
  - `logs/` (5 IPOPT logs at `file_print_level` 6, plus 5 stdout/stderr logs);
  - `nl/` (5 NL files);
  - `a2_harness_as_run_019d89a9.py.txt`;
  - `manifest_sha256.json`.

## Found (all six solves; TSO c1 from a2)

### Per solve

f is IPOPT's final unscaled objective. "NLP err / tol" is the scaled overall NLP error at termination divided by
`tol` 1e-5.

| solve | iters | restor. | μ₀ | μ_final (= floor) | f | compl. (unsc.) | dual inf (unsc.) | constr. viol. | NLP err / tol |
|---|---|---|---|---|---|---|---|---|---|
| TSO c1 ident + mu (a2) | 9 | 0 | 1e-6 | 4.545e-8 | −52244.700062139 | 7.19e-5 | 6.40e-3 | 2.1e-9 | 0.64 |
| TSO c2 pert | 48 | 0 | 0.1 | 4.545e-8 | −52248.405358495 | 5.95e-5 | 3.01e-3 | 1.1e-9 | 0.30 |
| TSO c3 pert + mu | 10 | 0 | 1e-6 | 4.545e-8 | −52248.405504329 | 4.79e-5 | 5.79e-4 | 1.4e-10 | 0.058 |
| DSO c1 ident + mu | **11** | 0 | 1e-6 | 9.091e-9 | 54561.135802079 | 1.72e-5 | 1.35e-3 | 2.6e-8 | 0.135 |
| DSO c2 pert | 51 | 0 | 0.1 | 9.091e-9 | 54561.309422845 | 1.04e-5 | 4.29e-5 | 1.4e-11 | 0.0043 |
| DSO c3 pert + mu | 11 | 0 | 1e-6 | 9.091e-9 | 54561.309884367 | 2.01e-5 | 1.50e-3 | 2.7e-8 | 0.150 |
| *Part A TSO warm (ref.)* | 22 | 0 | 0.1 | 4.5e-8 | −52244.700129903 | 5.92e-5 | 2.94e-3 | 7.0e-10 | 0.29 |
| *Part A DSO warm (ref.)* | 51 | 0 | 0.1 | 9.1e-9 | 54561.135089702 | 1.13e-5 | 3.56e-5 | 9.6e-12 | 0.0036 |

- All six ended `optimal` ("Optimal Solution Found."), with no restoration and μ_final at the floor.

### μ trace per solve

lg(μ), from the iteration table; the counts are iterations at each level.

- **TSO c1 (a2):** −6.0 (it 0–6), −7.3 (it 7–9).
- **TSO c2:** −1.0 (it 0), −1.7 (it 1–22, **22 iterations**), −2.5 (it 23–27), −3.8 (it 28–36), −5.7 (it 37–44),
  −7.3 (it 45–48).
- **TSO c3:** −6.0 (it 0–6), −7.3 (it 7–10).
- **DSO c1:** −6.0 (it 0–6), −8.0 (it 7–11).
- **DSO c2:** −1.0 (it 0), −2.5 (it 1–19), −3.8 (it 20–31), −5.7 (it 32–44), −8.0 (it 45–51).
- **DSO c3:** −6.0 (it 0–6), −8.0 (it 7–11).
- Every `mu_init` solve spends 7 iterations at μ = 1e-6, then drops straight to the floor (monotone, with the floor
  reached in one update).
- The full per-iteration lists are in `a2b.json` → `solves.<case>.mu_trace.lg_mu_per_iteration`.

### c3 against c2: the A/B on the perturbed problem

Same problem, same options except `mu_init`.

| | TSO c2 → c3 | DSO c2 → c3 |
|---|---|---|
| iterations | 48 → 10 (**−79 %**) | 51 → 11 (**−78 %**) |
| `run_smopf` wall | 0.176 → 0.114 s (**−35 %**) | 0.546 → 0.311 s (**−43 %**) |
| IPOPT process | 0.108 → 0.040 s (−63 %) | 0.319 → 0.109 s (−66 %) |
| OverallAlgorithm / linear solver | 0.072 → 0.015 / 0.043 → 0.008 s | 0.267 → 0.061 / 0.177 → 0.040 s |
| NL write / result load | 0.053 → 0.057 / 0.010 → 0.011 s | 0.184 → 0.158 / 0.038 → 0.039 s |
| NL write + load share of `run_smopf` | 36 % → **59 %** | 41 % → **63 %** |
| objective gate, ruling (a) | ratio 2.45: real difference | ratio 23.0: real difference |
| max \|dx\| (p.u.) / Vars > 1e-6 | 7.93e-5 / 91 of 2666: **agree** (≤ 1e-4) | 5.65e-4 / 578 of 10074: **do not agree** |
| largest components | `shared_es_pdch_hat` 7.9e-5, `shared_es_pch_hat` 5.0e-5 | `shared_es_pch_hat` 5.6e-4, `shared_es_pdch_hat` 5.5e-4, `flex_p_down` 9.9e-5 |

- Both columns have 0 Vars missing on either side. Vars with no value on either side: TSO 216, DSO 744. These are
  Vars without an initializer, as in a2.
- The wall times are single solves under background load (the D0 child process).
- The worst DSO Var is `shared_es_pch_hat[0,0,0,10]`.

### Perturbed baseline (c2, no `mu_init`) against Part A's warm counts

- **TSO: 48 against 22, 2.18×.** The perturbation, of order 1e-4 to 1e-5 p.u. on the consensus targets, adds a
  22-iteration phase at lg μ −1.7 that the identical problem does not have. Wall 0.176 against 0.117 s.
- **DSO: 51 against 51, 1.00×.** The same μ ladder apart from the −1.7 step. Wall 0.546 against 0.429 s.

### DSO c1 against the certified values (reported, not gated)

- **Objective:** +0.04225 (7.74e-7 relative). Part A's move between limits was +0.04153; c1 adds +7.1e-4 on top.
- **Variables:** max |dx| 6.67e-3 p.u. (`shared_es_pch_hat`); 1932 of 10074 above 1e-6.
- **Options:** c1's options, less `output_file` and `mu_init`, equal Part A's warm options, with an empty difference
  set.

### Predictions against outcomes (all six solves)

| Prediction | Outcome |
|---|---|
| A1: identical problem with `mu_init` 1e-6, TSO ≤ 8 and DSO ≤ 15 iterations | **FAILED on the TSO half.** TSO 9 > 8 (a2). DSO **11 ≤ 15 held** (51 → 11, −78 %). |
| A2: (2) within ±30 % of Part A's 22 / 51 | **FAILED on the TSO half.** TSO 48 (2.18×). DSO 51 (1.00×) held. |
| A3: (3) at least 40 % fewer iterations than (2) | **HELD.** TSO −79 % (48 → 10); DSO −78 % (51 → 11). |
| A4: agreement gates hold, no restoration | **FAILED.** Ruling (a) form: 1 of 4 objective gates pass (TSO c1, 0.943). TSO c3/c2 2.45, DSO c1 41.3 and DSO c3/c2 23.0 are real differences. DSO c3/c2 Vars 5.65e-4 > 1e-4. No restoration in any of the 6 solves. The original 1e-9 / 1e-6 form, reported: 0 of 4. |
| A5: DSO wall time of (3) at least 30 % below (2) | **HELD.** `run_smopf` 0.546 → 0.311 s, −43.0 %. Single solves, under load. |

### R7 recommendation (a recommendation, not a decision)

**Do not adopt `mu_init` on warm-started network solves yet.**
- **The gain.** On the perturbed warm re-solve, `mu_init` 1e-6 cut iterations by 78–79 % (48 → 10 TSO, 51 → 11 DSO)
  and the `run_smopf` wall by 35–43 % (0.176 → 0.114 s, 0.546 → 0.311 s), with no restoration.
- **The difference in the result.** On the DSO, the `mu_init` solves end at a different point from the μ = 0.1
  solves.
  - The objective difference is 23–41× the terminal-complementarity bar, and in every DSO pair the `mu_init` solve's
    objective is the higher one.
  - Variables differ by up to 5.65e-4 p.u.
  - The `mu_init` solves stop with 35–38× larger dual infeasibility, all within `tol`.
  - Whether this is a looser stop or a different limit is not resolved.
- **Caveat on the wall gain.** At SRP1 block size the gain is masked by NL write and result load. These are untouched
  by `mu_init`, and they become 59–63 % of the `mu_init` solve's wall. So a block-level 35–43 % is the ceiling from
  IPOPT-internal savings at this size. The production-relevant figure, per Part B, is the whole `run_smopf` call.

## Unexpected findings (a2b)

- **Part of the iteration saving is plausibly a shallower stop.** This is a hypothesis, not a result.
  - The μ = 0.1 path's last μ update jumps to the floor, −5.7 → −8.0 on the DSO, so the final iterate overshoots
    `tol` by about 250×: NLP error 0.36–0.43 % of `tol`.
  - The `mu_init` solves stop at 13.5–15 % of `tol` on the DSO, with dual infeasibility and constraint violation
    larger by the factors in the per-solve table.
  - Both forms satisfy the same tolerance, so they can differ by more than the complementarity-only bar measures.
  - The TSO pattern is weaker. There the μ = 0.1 solves themselves stop at 29–30 % of `tol`, and TSO c3 stops deeper
    than c2 (0.058 against 0.30).
- **The perturbation lengthens the TSO's μ = 0.1 path.** On the TSO it adds a 22-iteration plateau at lg μ −1.7.
  The DSO was unchanged.
- **My first dry run failed.** My terminal-statistics parser met per-iteration blocks at print level 6. Zero solves
  had run, and I fixed the parser before the launch (see Changed).

## Not confirmed (a2b)

- **Whether the DSO differences are stopping slack or a different limit.** Settling this needs tighter-tolerance
  solves, which are not authorized. The primal values were not serialized, so the ‖dx‖₁-based reading of the
  ruling's dual-infeasibility term cannot be computed after the fact.
- **That the wall reductions repeat.** One solve per case, under load average about 3.7 with the D0 child process
  running. The IPOPT-internal times (OverallAlgorithm, linear solver) move with the iteration counts. NL write and
  result load do not.
- **Whether Part B's Studio path takes its options from pickled blocks** (a2's finding (ii)). Not checked.
- **Where I looked:** `a2b.json`; `run.log`; the five IPOPT logs; a2's `a2.json`, manifest and c1 log; Part A's
  `tperf_local.json` and warm logs; and `git show HEAD:p516_tperf_profile.py`, compared definition by definition.

---

# A2 (first launch, 1 solve), as reported before ruling (a)

*Ruling (a), 2026-10-31: this launch is the first of the two launches that together make up the six declared solves.
a2 ran 1 (`tso_c1_ident_mu`); a2b ran the other 5. Under ruling (a)'s restated gate, TSO c1 is "indeterminate
(inside stopping slack)", ratio 0.943, so the c1 stop below would not have fired. The text below is unchanged except
that its headings are one level lower.*

## P5.16 T-perf A2: warm-start barrier A/B (`mu_init`). Run under ruling (b); stopped after solve 1 of 6 at the c1 objective gate (1.30 × the 1e-9 bar)

Worker report for the Planner, 2026-10-31. Authority: the author's "Decision 4" (≤ 6 declared solves), under
`P5_16_TPERF_SPEC_v1.md`, and the Planner's ruling (b) of 2026-10-31 on the stopped first attempt. Baseline:
`P5_16_TPERF_REPORT.md` (Part A). **One solve was run** (TSO c1), then the harness stopped on its own stop rule. No
production code was changed. Nothing here enters a table.

**Objective convention.** "f" below is the block's active ADMM objective, `p58_rescaled_admm_objective`, evaluated
at a point (Pyomo value at the loaded point, or IPOPT's final unscaled objective where stated). It is a local block
objective, not `gross_operational_cost` and not `net_operational_recourse`. It is compared only between points of the
same block.

**Instance.** The same instance as Part A: campaign `s53_w86_tail_recert`, cell `n7_4h_e1`, eval key
`ca8927e75d628bd1…`, candidate `db77e1549af855bc…`, certified at cycle 112; `certified_models.pkl` sha256
`fe06ca48…efda04`, verified against the declared value and both campaign manifests before unpickling; blocks TSO 2025
Spring (`case9`) and DSO node 7 2025 Spring (`case33_2`).

### Blocked on Planner

1. **The run stopped after one of the six declared solves, because the c1 objective gate failed by a factor of
   1.30. I need a ruling on how to continue.**
   - TSO c1 (identical problem, `mu_init` 1e-6, pickled pre-tail options) converged: `optimal`, 9 iterations, no
     restoration. Its objective is −52244.70006213897. Part A's warm limit is −52244.700129902631. The difference is
     +6.776e-5 absolute, **1.297e-9 relative, against the 1e-9 gate**. IPOPT's own final objective gives the same
     figure, 1.2970436e-9.
   - **This stop was my choice, and the task did not require it.** The task named the stop rules as failed solve,
     restoration and hash mismatch. I kept the stop on a failed c1 gate, which the first build had carried over from
     the original task. Without that stop, the other five solves would have run. As a result the guard verified 1
     solve against 6 declared (`verify` failures: "solve count 1 != declared 6", "process-launch count 1 !=
     declared 6"). That is the guard working correctly for a stopped run, not a fault in the harness.
   - **How the difference compares with where the two solves stopped.** I used a proxy here; it is not a defined
     bar. Both solves ended at the same μ floor, 4.545e-8. At termination the unscaled complementarity was 7.19e-5
     for c1 and 5.92e-5 for Part A's warm solve. Dual infeasibility (unscaled) was 6.40e-3 and 2.94e-3. The 6.78e-5
     objective difference is smaller than either run's terminal complementarity. By this proxy, stopping slack can
     explain it. The 1e-9 gate is 5.2e-5 absolute on this block, which is 0.73 × (c1) and 0.88 × (Part A) the
     terminal complementarity of a single run. For comparison, Part A's
     warm and cold solves agreed to 3.7e-11, and both of them started at μ = 0.1.
   - **The c1 variables against the certified values** (reported, not gated, under ruling (b)): max |dx| 8.63e-4
     p.u. (`shared_es_pch_hat[1,0,0,12]`). 903 of 2666 compared Vars exceed 1e-6. A further 216 Vars have no value on
     one side; Part A's report gives 216 TSO Vars without an initializer. The objective sits 4.38e-7 relative from
     the certified value, which is the move between limits already recorded as finding (i).

   **Options.** The Planner rules on each; (c) and (d) may need the author or the expert.
   - **(a) Recommended: continue with the five solves not yet run, into a new root.** These are TSO c2 and c3, and
     DSO c1, c2 and c3.
     - Write to a new root, e.g. `P516TPERF/a2b/`. `a2/` is write-once and holds the TSO c1 evidence.
     - Record the c1 gate result; do not stop on it. Stop only on the rules the task named.
     - Small harness change: a continuation flag that selects the five cases, declares 5 solves (`verify(5)`) and
       records c1 TSO by reference to `a2/a2.json`.
     - Total over a2 and a2b: 6 solves, within Decision 4. Wall time is about 1 min.
     - Whether this counts as "the six declared solves" in two launches, rather than one, is the Planner's call.
   - **(b)** As (a), but keep the stop on the c1 gate. DSO c1 could stop the run again for the same reason.
   - **(c)** Relaunch all six into a new root. That makes 7 solves in total, because TSO c1 would be solved again.
     This exceeds Decision 4, so it needs the author.
   - **(d)** Restate the c1 objective gate to a bar based on stopping slack, for example the terminal
     complementarity, before continuing. This changes the scope of a gate, so the expert or Planner rules.
2. **Commit.** I staged `p516_tperf_a2.py` and this report by filename, and committed nothing.
   - The evidence root `data/SRP1/Results/P516TPERF/a2/` and the launch logs are untracked, as `data/` is by design.
   - `manifest_sha256.json` has sha256 `c22e94d3ac566783afcbeafa68a4d3a25e1f4a517149f027c390a58183be8d72` and
     covers 20 files.
   - `a2.json` has sha256 `e87d67b2…bb178b`.
   - The Planner decides how to commit or hash-record the evidence.
3. **Provenance: A2 ran on an uncommitted copy of Part A's harness** (see Unexpected findings).
   - A2 imported `p516_tperf_profile.py` as it was in the working tree: sha256 `b8dbb6aa…`, ` M` in git, being
     edited by the Part B readiness Worker. The committed version is `d013452f…`.
   - The diff touches none of the functions A2 uses (`run_case`, `Instrument`, `parse_ipopt_log`,
     `_post_solve_parse`, `Lock`, `verify_inputs`, `_write_json`), with one exception: `capture_checklist` gained a
     keyword `deferred=()`, and its default leaves behaviour unchanged.
   - The A2 manifest pins `b8dbb6aa…`. If the Part B Worker changes that file again before committing, no committed
     version will match the manifest. Committing the Part B harness with this hash recorded, or noting the mismatch,
     closes the gap.

### Changed

- **`p516_tperf_a2.py` edited for ruling (b).** The bytes that ran have sha256 `019d89a9db85d686…981df51`; record
  schema `p516_tperf_a2_v2` with a `ruling` field.
  - **c1 gate.** The objective, as the Pyomo value at the loaded point, is compared with Part A's warm limit at ≤ 1e-9
    relative. The limit is the constant `PART_A_WARM_LIMIT_DECLARED`: TSO −52244.700129902631, DSO 54561.135089701755.
    `compare_c1` reports c1's free Vars against the certified values (worst 20, maximum per component, count above
    1e-6 per component), and the per-variable gate flags are renamed "informational only".
  - **Also reported for c1:** IPOPT's final objective from c1's log, and whether the options IPOPT was passed (less
    `output_file` and `mu_init`) equal Part A's warm (i) options. They are equal, with an empty difference set.
  - **c3 against c2:** unchanged, objective ≤ 1e-9 and every free Var ≤ 1e-6.
  - **`comparator_check` restated** (zero solves; a real run refuses on failure, a dry run records and continues):
    - (i) the comparator parsed from Part A's committed warm log (hash checked against Part A's manifest
      `fbfd6310…`) must equal the declared 17-digit value exactly;
    - (ii) c1's options, less `output_file` and `mu_init`, must equal Part A's warm (i) options less `output_file`.
    - The certifying options and the offset between the certified point and Part A's limit are recorded but not gated.
  - **`mu_trace`** adds `lg_mu_per_iteration`. Self-test on Part A's warm logs: 23 entries for TSO (22 iterations) and
    52 for DSO (51 iterations).
  - **Kept as built:**
    - the predictions file;
    - the perturbation: +10 × the terminal mismatch per channel, from T0.1b `coupling.json`, hash-verified against the
      T0.1b manifest `4e5bbd45…`;
    - the bounded guard (`verify(6)`, per-case cap);
    - the stop rules (failed solve or cap block, restoration, hash mismatch), plus the c1-gate stop discussed under
      Blocked on Planner, item 1.
- **Dry run (zero solves)**, `--dry-run /tmp/a2_dry2`, exit 0. The guard reported 0 permitted, 6 blocked and 0
  executions. The comparator check was "ok" on both blocks: comparator equal to declared True, option regime match
  True. The capture checklist and the A2 items held. `predictions.json` was written before the first blocked solve.
- **Launch**, attached, alone, both streams captured, `set -o noclobber`:
  `python -u p516_tperf_a2.py > …/a2_launch_stdout.log 2> …/a2_launch_stderr.log`. Exit 2 (status `stopped`). The
  stderr log is 0 bytes.
  - git HEAD `7324aa9c`; production files clean (`production_files_git_status` empty).
  - The comparator check was "ok", and `predictions.json` (sha256 `ca4b850d…`) was written before solve 1.
  - Then `--manifest` (20 files).
- **Evidence root** `data/SRP1/Results/P516TPERF/a2/`, new and write-once: `a2.json`, `predictions.json`, `run.log`,
  `logs/a2_tso_c1_ident_mu_2025_Spring.log` (IPOPT, file level 6), `logs/tso_c1_ident_mu_stdout_stderr.log`,
  `nl/tso_c1_ident_mu.nl`, `manifest_sha256.json`. Launch logs: `data/SRP1/Results/P516TPERF/a2_launch_*.log`.
- **Concurrent load during the run.** One other compute process was running: `p516_d0_t01b.py --child` (pid 99165,
  about 100 % of one core; it belongs to the D0 Worker). The load average was 2.35 / 2.13 / 2.03; during Part A's TSO
  warm solve it was 1.82. Thread caps were as in production (`OMP_NUM_THREADS=1` and so on).

### Found

#### Solve 1: TSO c1 (identical problem, `mu_init` 1e-6, pickled pre-tail options)

| quantity | A2 TSO c1 (`mu_init` 1e-6) | Part A TSO warm (i) (no `mu_init`) |
|---|---|---|
| termination / restoration | optimal / 0 lines | optimal / 0 |
| iterations | **9** | 22 |
| lg(mu) per iteration | −6.0 (it 0–6), −7.3 (it 7–9) | −1.0 (it 0), then −3.8, −5.7, −7.3 |
| μ at iteration 0 (first "Current barrier parameter") | 1e-6 | 0.1 |
| `mu_final` / μ floor | 4.545e-8 / 4.545e-8 (at the floor) | 4.5e-8 (lg −7.3) |
| final f (IPOPT, unscaled) | −52244.700062138974 | −52244.700129902631 |
| complementarity / dual inf (unscaled, final) | 7.19e-5 / 6.40e-3 | 5.92e-5 / 2.94e-3 |
| constraint violation / variable-bound violation | 2.09e-9 / 9.60e-9 | 7.01e-10 / 9.60e-9 |
| `run_smopf` wall | 0.142 s | 0.117 s |
| IPOPT process / OverallAlgorithm / linear solver | 0.070 / 0.019 / 0.010 s | 0.058 / 0.032 / 0.018 s |
| NL write / result load | 0.051 / 0.015 s | 0.045 / 0.009 s |

- **Iterations and algorithm time.** Iterations fell from 22 to 9 (−59 %) and IPOPT's OverallAlgorithm time from
  0.032 s to 0.019 s.
- **Wall time.** The `run_smopf` wall did not fall: 0.142 s against 0.117 s. NL write and result load are the bulk of
  it on this block, and they rose here.
  - One solve on each side, under different background load, so this is not a timing result.
- **Starting point.** Iteration 0 starts at f = −52243.141 in both runs, not at the certified −52244.723. The warm
  start's bound pushes (1e-6) move the start off the certified point. I did not investigate this further.

#### Gate on solve (1) (ruling (b))

- **Objective: FAILED**, 1.297e-9 relative against 1e-9. The figures are under Blocked on Planner, item 1.
- **Variables against the certified values: reported, not gated.** Max |dx| 8.627e-4 p.u., 903 / 2666 above 1e-6.
  - Components with the largest maximum deviation: `shared_es_pch_hat` 8.6e-4, `shared_es_pdch_hat` 7.4e-4,
    `vmag_sqr` 8.1e-5, `voltage_product_real` 7.9e-5, `qg` 5.9e-5, `e` 4.5e-5, `vmag` 3.9e-5.
  - Components with the most Vars over 1e-6: `vmag_sqr` 202, `voltage_product_real` 202, `e` 182, `qg` 62, `vmag` 61.
  - Full lists are in `a2.json` → `agreement.tso_c1_vs_part_a_limit.vars_vs_certified_reported_not_gated`.

#### Findings carried from the stopped attempt (as the Planner directed)

- **(i) Part A's "from the optimum" is partly a move between limits.** The certified point sits under
  `compl_inf_tol` 1e-6 (μ floor 9.09e-11), but the pickled blocks carry the pre-tail options. TSO: 5e-4, floor
  4.545e-8. DSO: not passed, floor 9.09e-9.
  - Part A's 22 / 51 "from the optimum" iterations therefore include a move between limits: f +4.37e-7 relative on
    the TSO (+0.02284), +7.61e-7 relative on the DSO (+0.04153).
  - So attributing the warm count to the barrier restart is partly confounded. This run records it again: the
    comparator check's `recorded_not_gated` section gives TSO limit − certified = 0.0228413 (4.372e-7) and DSO
    0.0415347 (7.613e-7).
- **(ii) Pickled blocks of a tail campaign do not carry the tail option.** The tail writes `compl_inf_tol` onto the
  NetworkData params object (`shared_resources_planning._apply_convergence_depth_tail`), not onto the one the block's
  rule partial holds. Any stage that takes "production options" from pickled blocks of a tail campaign gets the
  pre-tail options. Flagged for every such stage. I did not check Part B's Studio path.

#### Perturbation (applied to the prepared c2 and c3 clones; not solved)

Unchanged from the first attempt: +10 × the maximum |terminal mismatch| per channel, uniform over each target Param.

| Param | TSO shift (Param units) | DSO shift (Param units) |
|---|---|---|
| `vmag_req` | +6.246e-5 pu | +3.067e-5 pu |
| `p_pf_req` | +1.905e-4 pu | +2.715e-5 pu |
| `q_pf_req` | +1.698e-5 pu | +8.829e-7 pu |
| `p_ess_req` | +1.119e-5 pu | +6.461e-6 pu |
| `q_ess_req` | +1.777e-7 pu | +1.777e-7 pu |

#### Predictions against outcomes

| Prediction | Outcome |
|---|---|
| A1: identical problem with `mu_init` 1e-6, TSO ≤ 8 and DSO ≤ 15 iterations | **FAILED on the TSO half: 9 > 8** (22 → 9, −59 %). DSO not run. |
| A2: (2) within ±30 % of Part A's 22 / 51 | **not tested** (c2 not run) |
| A3: (3) at least 40 % fewer iterations than (2) | **not tested** |
| A4: agreement gates hold, no restoration | **FAILED on its measured part.** TSO c1 objective gate 1.297e-9 > 1e-9; no restoration (0 lines). The c3-vs-c2 gates were not run. |
| A5: DSO wall time of (3) at least 30 % below (2) | **not tested** |

- **μ trace for all six solves:** measured for TSO c1 only (table above). The other five solves were not run, so
  they have no trace.
- **Part A's warm traces, for reference** (measured by the same parser, from Part A's logs):
  - TSO: −1.0, −3.8, −5.7, −7.3 over 22 iterations;
  - DSO: −1.0, −2.5, …, −8.0 over 51 iterations.

### Unexpected findings

- **The 1e-9 objective gate is tighter than the terminal complementarity of the solves it compares.**
  - The gate is about 5.2e-5 absolute on the TSO. The terminal unscaled complementarity was 5.9e-5 to 7.2e-5.
  - Part A's warm and cold solves agreed to 3.7e-11, far inside the gate. Both started at μ = 0.1 and so took similar
    barrier paths.
  - A solve entering the barrier at μ = 1e-6 stops at a different point within the same tolerance. That point is not
    necessarily closer.
  - The same consideration applies to the c3-vs-c2 gate. There the two solves start at different μ by construction.
- **Part A's harness changed under A2** (Blocked on Planner, item 3). It is unrelated to A2's logic, but it affects
  the provenance of the hash.

### Not confirmed

- **A2–A5, the DSO half of A1, and the c3-vs-c2 agreement.** These solves were not run: the run stopped after TSO c1.
- **That the 6.78e-5 difference is stopping slack, not a different limit.** This is inferred only from the
  terminal-complementarity proxy; I measured no such bar. Settling it would need a third solve at a tighter tolerance,
  which is not authorized.
- **That `mu_init` reduces wall time.** On one TSO solve the `run_smopf` wall rose by 0.025 s, under different
  background load (the D0 process). Not a timing result.
- **Whether the Part B Studio path takes its options from pickled blocks.** Not checked; it is outside this task.
- **Where I looked:** `a2.json`, `run.log`, c1's IPOPT log, Part A's `tperf_local.json` and warm logs, and
  `git diff -- p516_tperf_profile.py` (hunk headers and the shared-function hunks only).

**R7 recommendation (a recommendation, not a decision): not yet; adoption of `mu_init` cannot be recommended on the
evidence so far.**
- On the TSO identical problem, `mu_init` 1e-6 cut IPOPT iterations from 22 to 9 (−59 %) with no restoration.
- It landed 1.3e-9 relative from Part A's limit, outside the 1e-9 gate but inside the terminal-complementarity proxy.
- It did not reduce the `run_smopf` wall: 0.142 s against 0.117 s.
- The perturbed A/B that R7 needs, (3) against (2) on both blocks, was not run. Continuing under option (a) would
  supply it.
