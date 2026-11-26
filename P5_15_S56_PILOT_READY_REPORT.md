# P5.15 S56 critical path: integration rulings applied, gates re-run, EV pilot spec frozen (Worker report, 2026-11-24)

**Authority:** `S56_SEARCH_DRIVER_DESIGN.md`, "Oracle mapping and key rules" and "Integration rulings" 1–7 (`601775c0`);
`P5_15_S56_SEARCH_DRIVER_REPORT.md` (driver build and Integration section); the Planner's pilot-ready task.

**Solve claim.** The driver gate, the oracle suite, every driver process, the pilot freeze and the preflight ran under
an armed `SolveProfileGuard(permitted=())`, verified at 0; pickle load / loads blocked, 0. Gate F was the only
solving run: 10 evaluations at cap 3, each child's armed bounded guard reconciled to 204 = 51 × (3 + 1) (G4 holds on
all 10), the gate parent at 0. **The pilot was not launched.**

**Objective convention (every EUR figure):** F = I(x) + gross_operational_cost − TV credit, EUR of total discounted
planning cost (cost-breakdown schema v2); gross settlement-excluded. Gate F compares records bitwise and quotes no
objective value.

## Blocked on Planner

1. **Gate F (ev_v1) FAILS as frozen, on G3 only. CLAUDE.md stops a campaign on a gate failure, so the pilot launch
   waits for your ruling.**
   - **G0, G1, G2 and G4 hold on all five candidates.** Every per-cycle record (G1) and every IPOPT solve (G2: 204 final
     blocks and 192 solve records per candidate) is bitwise equal, serial against 5-way.
   - **G3 fails on 3 of 5 candidates**, on two fields only: `interface_dual_capture.bytes_written` and
     `bytes_per_cycle_line_max`. They differ by 1–2 bytes.
   - **Mechanism (zero-solve diagnosis, committed):** they are byte counts of `interface_duals_per_cycle.jsonl`, and
     each line of that file carries the timing field `capture_s`.
     - With `capture_s` removed, every line is bitwise equal, A against B.
     - On every line, the byte difference equals the difference in the text length of `capture_s`.
     - I did not declare these two timing-derived counts in the G3 exclusion list. I did not re-run or tune the gate
       (failure policy).
   - **Options:**
     - **(a)** Rule that the failure is a declaration omission, and accept gate F on the evidence and the diagnosis.
       Cost: none.
     - **(b)** Freeze a v2 gate spec that adds the two fields to G3's exclusions, and re-run it. Cost: about 26 min of
       solves, plus a Worker task.
   - **Recommendation: (a).** No computed value differs. The diagnosis shows the whole mechanism.
2. **Driver gate v2: 162/167. Five item 6 checks fail, and all five are explained by the cap-in-key rule of
   `8948db26`.**
   - Two checks fail on the settled-key relation (ev, draw0).
   - Three checks fail on option-off identity (s56_ev_staged, s56_draw0_staged, calibration_key_coupled).
   - Why they fail: the checks were written at `dcea9628`. They compare against the pre-cap-rule harness `b0391b80`
     and expect settled = sk(exit).
   - The cap rule now wraps every S56 EV-plan key last.
   - Zero-solve recomputation from the gate's own frozen specs:
     - exit = cap(base) and settled = cap(sk(base)) on all 3 plans × 2 instances;
     - post = cap(pre) on every entry of the three identity sets;
     - the only differing entry fields are `eval_key`, `eval_dir`, `working_dir_ids` and `s56_cycle_cap_keyed`.
     - The SRP1 single-year set (no S56 key) is still identical.
   - The integration report predicted only item 4 would fail, and missed these five.
   - Ruling 6 re-scoped item 4 only, so I did not re-scope item 6.
   - **Options:** (a) re-scope the five checks to the cap-wrapped relations (zero solves, about 1 h with a re-run); or
     (b) accept them as stated with the diagnosis.
3. **Readings I took in the recorded predictions.** Each is stated in the frozen spec. Change any you disagree with
   before the launch: that needs a v2 spec, because the spec is frozen.
   - **PP4 and PP5** are evaluated at dF_set at K_EV (the frozen e_EV rule in `pilot_outputs`). The K = 300 values are
     reported beside them.
   - **PP8 "at 1-way":** the pilot never runs at 1-way.
     - Block 1 (5-way) per-cycle walls are divided by gate F's κ = 1.220.
     - Block 2 (2-way) is not deflated, so it is an upper bound.
   - **PP6** is evaluated on all three unit runs and the staged run. x0 is reported beside.
4. **Gate F was adapted beyond configuration alone.**
   - Comparing all five children against serial needed a per-label comparison loop, a G1 exclusion hook (G1 had none
     in v1), candidate triples for staged plans, and a lattice-membership check in place of the s47 domain check.
   - The default profile (`srp1_v1`) keeps v1's meaning. Its committed evidence cannot be re-run anyway: its directory
     exists.
   - The comparison criteria are unchanged.
5. **Author action (not a decision for you):** the grid is `srp1grid`, the outcome of the memory-probe rule. The
   author's one-line confirmation is still pending, and the spec records that.

## Changed

| commit | content |
|---|---|
| `5d84ccf5` | Rulings 1–7: `p515_s56_search_policy.py`, `p515_s56_search_driver.py`, `p515_s56_search_gate.py`, `p515_s56_search_oracle_tests.py` |
| `859598b0` | Evidence `search_driver_v2/`, `search_driver_v2_item6_diagnosis/`, `search_oracle_v2/` |
| `83891647` | `p515_s56_concurrency_gate.py`: profile `ev_v1` |
| `9e0e8250` | Gate F frozen spec `frozen_concurrency_gate_ev_spec_v1_8a2dfc66.json` + freeze log (committed before the run) |
| `0e3f7fda` | Gate F run evidence + `concurrency_gate_ev_v1_g3_diagnosis/` |
| `33dc00fa` | `pilot_ev_v1/pilot_spec_92547279.json` + freeze logs; `pilot_ev_v1_predictions.json` |
| (this commit) | `pilot_ev_v1_preflight/` + this report |

**Rulings, as implemented:**
- **Ruling 1:** `MAPPING_PARAMETERS` final: τ_cl = 1e-5 p.u. and the ESSO slack gate = 1e-5. `provisional` is now `[]`.
- **Ruling 2:** `S_SHARE_EUR = 500` is frozen into the pilot spec's `rules`. `--pilot-evaluate` reads it there and
  refuses any other value.
- **Ruling 3:** `PILOT_INIT_VARIANTS` = (1.0, 0.5, 2.0), status FINAL.
  - `rho0_scale` is written only when the value is not 1.0.
  - The staged plan is asserted against its ruled lattice label and key.
- **Ruling 5 (policy):** `confirm` returns `unresolved_after_confirmation` with `not_settled: True` and the side(s)
  named.
  - No acceptance test and no held-out check is applied to a not-settled value.
  - A harness_error or unknown status on either side still stops the search.
  - The review trigger `not_settled_review(state)` is derived from the state, so no state field was added. It fires on
    3 consecutive not-settled confirmations, or on more than 25 % of at least 4 confirmations, and then terminates with
    `stop_for_review_not_settled`.
  - The driver maps that termination to exit 5 (`REVIEW_TERMINATIONS`).
  - The terminal report lists the flag and the review record.
- **Ruling 5 (tests):**
  - Gate item 7 (18 cases): 7 confirm-level cases plus 11 trigger boundaries, including 1 of 4 = 25 % (no stop), 2 of
    4 (stop), 2 of 3 (no stop: fewer than 4), a budget-unrun confirmation that does not count, and the trigger across
    polls.
  - Gate item 8 (7 cases): in real synthetic searches, each pattern reached on the path it produces:
    - one not-settled confirmation: the search continues;
    - 3 consecutive: stop;
    - 2 of 4: stop by the share rule;
    - 1 of many: never a stop.
- **Ruling 6:** the item 4 control now asserts the smoke alias is absent.
  - The 2× EV configuration freezes.
  - The collision scan finds nothing.
  - No smoke-spec key is among the configuration's 74,084 domain keys.
- **Ruling 7:** a real pilot defaults to 4.0 GiB per slot and refuses any other value.
- **Pilot additions:**
  - A real freeze requires `--predictions` (PP1–PP8, each with statement / evaluated_as / on_failure, embedded
    verbatim with the file's sha256).
  - The freeze runs `pilot_capture_checklist` (rule eleven: a capture path exists for every PP quantity).
  - New mode `--pilot-run --preflight-only --out <new dir>`, at zero solves:
    - it takes the lock, checks the chain of completed blocks and runs `real_launch_preflight` for the next block;
    - it writes the record and releases the lock;
    - it writes nothing under the pilot root and launches nothing.
- **Oracle tests:** output to `search_oracle_v2/`. T5 and T7 were updated for the rulings: refusals without
  predictions, at 3.0 GiB and with other initialisations; s_share read from the spec; preflight-only cases.

**Gate F `ev_v1` configuration** (frozen in spec `8a2dfc66` before the run):
- SRP1-grid 3 × 3 EV twin (2025/2030/2035), 2×, harness pin `05a9838c`, cap 3.
- The pilot's i1 entry options: every S56 EV-plan key, including the settled stop-disabled key and the pricing capture.
- Five candidates: the reference unit (`ae319309…`), x0, the staged plan, n5 2025 0.25/0.5 and n9 2025 0.25/1.
- Root A = the five at concurrency 1; root B = the five at concurrency 5.
- **G1 exclusions** (declared): `per_cycle_record.jsonl` `[*].cycle_wall_s`, `rss_bytes`, `ru_maxrss_bytes`,
  `response_capture_s`.
- **G3 additions** (declared): the EV record's paths; its wall and runtime fields; `operational_workbook` sha256 and
  size (the xlsx is timestamped); `response_terminal` sha256 and capture_cost;
  `initialisation_identity.other_candidates_not_compared` (it depends on sibling order).
- Freeze checks 8/8: configuration = the real check's block, lattice membership, exclusion coverage, capture path,
  comparator self-test.

## Found

**Driver gate (`search_driver_v2/`, from `5d84ccf5`):** 162/167, 1,247 s, attached, stderr empty, guards 0.

| item | checks | result |
|---|---|---|
| 1 N(x) | 11 | pass |
| 2 order determinism | 4 | pass |
| 3 order vs values | 2 | pass |
| 4 stub replay + smoke alias absent (ruling 6) | 8 | pass |
| 5 kill and resume | 36 | pass |
| 6 keys | 28 | **23 pass, 5 FAIL** (Blocked 2; diagnosis `3a42a2fc…`) |
| 7 boundaries + not_settled + review trigger | 45 | pass |
| 8 arithmetic + not_settled in-search | 25 | pass |
| 9 published histories | 6 | pass |

- `search_gate.json` `1aaf8f34…`; `manifest_sha256.json` `4775b047…` (7,100 files; the test trees are hash-recorded,
  not committed).
- `gate_launch_stdout.log` `09ad97ca…`; `gate_launch_stderr.log` is empty.

**Oracle suite (`search_oracle_v2/`, from `5d84ccf5`):** 110/110, 57 s, stderr empty, guards 0.
- `zero_solve.json` `e7d771bd…`; manifest `5c7b8844…` (385 files).
- T7's preflight-only case on a test-only harness pilot passed (23.13 GiB available against 20.0 GiB required).

**Gate F verdict: FAIL as frozen (G3).** `gate_result.json` `f4d6905c…`, manifest `2d7b90c2…`, `run_launch.log`
`5e938499…`. Diagnosis `f582fe31…`, all_hold true. Wall time 1,540 s (A 1,027 s, B 501 s).

| prediction (recorded in spec `8a2dfc66`) | outcome |
|---|---|
| P1: every candidate bitwise identical, A vs B (G1 ∧ G2 ∧ G3) | **failed**: G1 true, G2 true, G3 false (Blocked 1) |
| P2 (Worker expectation, not gated): contention 1.10–1.30× | **held**: 1.220 on the shared candidate; 1.198–1.230 per candidate against its own A |
| P3 (Worker expectation, not gated): peak child RSS 2.3–4.0 GiB | **held**: 2.374–2.409 GiB on all 10 children |

- G0: A max concurrency 1, B max concurrency 5. Every child exited 0, status not_certified, 3 cycles.
- Memory available during B: at least 14.46 GiB.
- Mean s/cycle at cycles 1–3: 21.7–26.1 at 1-way, 26.7–31.2 at 5-way.

**Pilot spec (frozen, committed, not launched):** `data/SRP1/Results/P515S56/pilot_ev_v1/pilot_spec_92547279.json`
(sha256 `925472791ffd…`).
- Configuration: SRP1 grid; harness oracle; not test-only; K 300; 2×; required consecutive cycles 10; 4.0 GiB per
  slot; s_share 500; mapping provisional `[]`; staged label as ruled; capture checklist all true.
- Blocks: `[x0_i1, unit_i1, staged_i1, x0_i2, unit_i2]`, `[x0_i3, unit_i3]`.
- Predictions PP1–PP8: `pilot_ev_v1_predictions.json` `9cb89ef3…`, embedded verbatim.

**Numbers derived from existing data** (they inform the predictions; they are not predictions):
- **Real check (1-way, 3 cycles):** 24.55 s/cycle; child overhead 125.9 s; peak RSS 2.43 GiB.
- **Smoke (1-way):** x0 21.5 / 22.8 s; unit 25.2 / 23.8 s.
- **Projection, one run to K = 300:**
  - about 2.05 h at 1-way;
  - about 2.50 h per 5-way block (gate F's measured 5-way s/cycle; overhead × 1.23);
  - pilot total about 5.0 h over two launches, each under 4 h.
  - The cycle time at tail depth is **unmeasured** on EV and may be longer.
- **RSS growth:** the real check's per-cycle RSS still grows (+110 MB, then +54 MB). PP7 at 300 cycles rests on
  ruling 7's full-run step 4 figure, not on this trajectory.

**Preflight-only, zero solves:** `data/SRP1/Results/P515S56/pilot_ev_v1_preflight/pilot_preflight.json` `5c8a0957…`,
from `33dc00fa`.
- Result: **pass**. Harness preconditions `[]`; refusals `[]`.
- Memory: 24.49 GiB available against 20.0 GiB required.
- Next block: 1. Launched false; launch dir not created; lock released; pilot root unchanged.
- The logs `pf_stdout.log` (`5d6d4e58…`) and `pf_stderr.log` (empty) were captured in the scratchpad and copied in,
  because `--out` must not exist before the run.

**Exact pilot launch commands** (after Blocked 1 is ruled; one attached launch per block; never detached):

```
cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation
# block 1 (5 runs, ~2.5 h projected) -> exit 4
NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_search_driver.py \
  --pilot-run --root data/SRP1/Results/P515S56/pilot_ev_v1 \
  > data/SRP1/Results/P515S56/pilot_ev_v1/launch_001_stdout.log 2> data/SRP1/Results/P515S56/pilot_ev_v1/launch_001_stderr.log
# block 2 (2 runs) -> exit 0: the same command with launch_002_*.log
# then (zero solves): --pilot-evaluate --root data/SRP1/Results/P515S56/pilot_ev_v1 --out data/SRP1/Results/P515S56/pilot_ev_v1_result
```

**Unexpected findings:**
- **Memory margin at launch is thin.** 23.1–24.5 GiB was available against the 20.0 GiB required. Another resident
  workload of about 4 GiB would make the preflight refuse.
- **A process slip.** While waiting for the driver gate I briefly ran a waiter that re-evaluated `pgrep -f` in a
  loop, against the task's waiting rule.
  - I killed it within about 2 min, by its explicit PID, and replaced it with a `kill -0` poll of the gate's PID.
  - Its command line did not contain `p515_s4`, and nothing launched while it was alive.
- **Scratchpad leftovers.** This session's scratchpad holds files from the earlier integration task (latest
  `commit_list.txt`, 22:05, listing the `e1b87c27` evidence). Nothing in it belongs to an earlier attempt at this
  task, and nothing was adopted.

## Not confirmed

- **No pilot run.** PP1–PP8 are unevaluated. The tail-depth cycle time, c*, the full-run RSS and ψ are unmeasured.
- **Gate F's G3 diagnosis covers only the fields that differed.** Equality of every other G3 field is the gate's own
  result.
- **Concurrency at 2-way (pilot block 2) was not measured.**
- **Item 6's re-scoped relations hold on the gate's own specs** (diagnosis), but item 6 as a gate was not re-run
  re-scoped.
- **`pilot_evaluate` on real pilot records** has run only on synthetic traces (T7). Searched: no committed EV record at
  settled depth exists.
