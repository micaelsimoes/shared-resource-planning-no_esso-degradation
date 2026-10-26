# P5.15 W183i — Step-7 hooks module, router patch, launchers, spec v1 (Addendum 86; zero solves except gate (a)(iv))

**Authority:** Addendum 86 (W183i order; W184/W185 re-authorized conditionally).

**State after W183i.**

| item | value |
|---|---|
| production | `6f43479e` (unchanged) |
| harness | `p515_s44_campaign_harness.py` **`d9a944f2…`** (from `c34ebdc5…`; +3-line router branch, authorized by ruling 1) |
| hooks | `p515_s55_step7_resettle_hooks.py` `9318ade3…` |
| Step-7 launcher | `p515_s55_step7_launcher.py` `3b990d19…` |
| E1 launcher | v3 `532cf6e7…` |
| spec v1 | **`data/SRP1/Results/P515S55/frozen_step7_campaign_spec_v1_96345241.json`** (`96345241…`; first version, no predecessor) |
| E3 stage spec | `bd2cfa4d…` |
| E2 stage spec | `7631a6d7…` |

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Runs.**
- **Solves: 153, all in gate (a)(iv).** The gate's armed guard verified them exactly.
- **Everything else** ran under `SolveProfileGuard(permitted=())` verified at 0, with pickle 0 and builds 0.
- **Execution.** Every part ran attached, with both streams captured. Scripts were committed before their runs.
- **Order.** (a) ran alone; (b) and (c) ran in parallel on separate files; (d) ran after both.

| part | report | commits |
|---|---|---|
| (a) hooks + router + gates (i)–(iv) | `P5_15_W183I_A_ROUTER.md` | `299f6a4a`, `637ce3c5` |
| (b) Step-7 fixed-plan launcher | `P5_15_W183I_B_STEP7_LAUNCHER.md` | `02e8d467`, `396c0d62` |
| (c) E1 launcher v3 (resumable) | `P5_15_W183I_C_E1_V3.md` | `4aca667e`, `7ddd6ce7` |
| (d) spec v1 freeze; real E3/E2 freezes; E3 preconditions-only | `P5_15_W183I_D_SPEC_V1.md` | `2a62ee1d`, `d8dcc139`, `36af2aa9` |

**W184's condition holds** ((a), (b) and (d)). E3 is launched next under its Addendum 85 line, as a separate commit with its
launch log.

## Decisions needed

None before W184/W185. For Addendum 87 (before W186), from W183i (c):
1. **A single evaluation can exceed 4 h.** One evaluation running to cap 500 takes about 4.7 h. 2× production-exit
   records show a median of 1.41 h and a maximum of 2.96 h. Recommend accepting under D1 (Addendum 85 ruling 9).
2. **A commit is needed between E1 launches.** The resume reads only git-tracked records. Recommend keeping git.
3. **A harness fault now stops the arm** at that launch.
4. **One launch per evaluating poll;** polls without evaluations are processed by the next launch.
5. **A crashed launch leaves unlisted eval dirs,** which the next resume refuses. Recovery needs a ruling at the time.

## Planner rulings made within the addenda (recorded)

- **(a) cell identity.** Arm `A` (baseline by the absence of `model_variant`), gate `none_first_evaluation` (W155
  convention), item `E3`/`E2`, claim group `step7_E3`/`step7_E2`. These describe the cells truthfully.
- **(b) choices.**
  - The screen's x = 0 key `5cfe69a6…` is held by the committed W86 spec `ddd6cd44`; it is reported and evaluated anew
    (Addendum 85 W183g (c), "reported and never reused").
  - 1× is declared by absence.
  - An uncertified cell reads in W132's uncertified form ("indeterminate (slack undefined)"). W184 stops for review on
    an uncertified E3 in any case.
  - The curtailment columns: W184's report must first reproduce Table 12's coordinated (C_OPEN) curtailment from row 1
    with the same formula before reporting C_NRF's.
- **(d) diagnostics.** The converter-circle residual, max |P^E|/S^Av and SOC-within-limits are not recorded values;
  they are recoverable from the run's own persisted `certified_models.pkl` (persist on for E3/E2). That file is accepted
  as their capture path (the C\* failure mode was the absence of serialized models). W184/W185 read it under a guarded,
  logged read (Addendum 85 ruling 6). The closure slacks, interface residual ratios and settlement are recorded in
  committed outputs. "ESSO σ slack" names nothing in the code and is reported as not applicable.

## Blocked on the author

- **Push from PyCharm.**

## Changed

- See the part table.
- **The harness change** is the router branch only (the Planner read the diff): 3 lines after the W155 branch in
  `resettle_hooks_module`.
- **No production or committed-artifact change.** No earlier hooks module, gate or wrapper was edited.

## Found — predictions against outcomes

| prediction (Addendum 86) | outcome |
|---|---|
| (a)(i) every committed eval key recomputes | **held** — 7,213/7,213 entries of 252 v2 specs, equal to the stored keys and to the pre-patch harness's; 11 v1 specs carry no key |
| (a)(ii) routing identity | **held** — 9 committed schemas routed to the same module, with identical validation outcomes |
| (a)(iii) neutrality | **held** — no committed declaration takes the new branch; planted entries of all 9 cells route and validate; 25/25 planted defects refused |
| (a)(iv) 0 genuine, `GATE_PASS_RULED` True, provenance ≤ 4 | **held** — 153/153 solves, 0 retries; GATE_PASS (frozen) False, RULED True; 0 genuine vs W48 (provenance 3) and vs W83 (provenance 4); Q = 740,922,817.2425401 bitwise = W48; `per_cycle_record.jsonl` byte-identical to the W183f and W83 arms |
| (b) key counts 1/8/15; 0 committed holders of the E3/E2 keys | **held** — 263/263 dry-run assertions. E3's base = W183c's x0+NRF key `759802ec…`. The replicated-mean ρ check reproduces the committed W183f row bitwise (pf 0.1980000000000002/48, ess 0.010000000000000004/51, v 0.011549999999999998/48), plus 1,074 cycle checks on two other committed records |
| (c) split run ≡ in-process loop | **held** — 447/447: byte-equal at every batch and poll boundary, after B+ advances and A7 stops; 7/7 foreign records and 24/24 tampered states refused; W183g (b) checks 198/198 on v3 |
| (d) freeze | **done** — every pin matched the clean working tree. Keys equal the committed dry runs (E3 1, E2 8, screen 15 + alias, E1 7 + alias, batch 4). Capture checklists present per phase. E3 preconditions-only: 22/22 held, guards 0 |

## Not confirmed

- **Not yet executed:** the real `--run` path (lock, child, results write), the hooks module inside a solve, and the NRF
  terminal capture on a solved model. W184 is the first exercise of all of them.
- **The E1 freeze and resumable `--run`** have not been exercised on real roots.
- **The diagnostics read from the persisted pickle** have not been tested.
- **The holder scan** covered git-tracked `data/*campaign_spec_*.json` only.
