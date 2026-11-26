# P5.15 S56 critical path step 3: memory probe of the corrected SP model (Worker report, 2026-11-23)

Authority: TASKS.md, "Working mode and EV priority", critical path step 3. This is an engineering measurement: one script
with pass/fail checks and no prediction spec. The grid is the author's decision; this report supplies the measurement.

**Objective convention (the EUR figures below; carried for identification only, since 2 cycles is not a converged
point):** EUR of total discounted cost.
- `gross` is production's `gross_operational_cost`. On this coupled entry that means the settlement is excluded in full.
- `net` = gross − terminal-value credit.
- `total` = I(x) + net.

**Memory units:** GiB = 2^30 bytes.

## Blocked on Planner

1. **The two memory measures disagree on the five-block grid, and both exceed 20.7 GiB.**
   - **Peak tree RSS** (the decision quantity the task names) is **20.97 GiB**.
   - **Peak tree phys_footprint** is **24.99 GiB**. The harness child's `ri_lifetime_max_phys_footprint` is 24.79 GiB.
   - The 20.7 GiB criterion was defined as a **footprint**: S56_PLAN_v2.md says "peak footprint … ≤ 0.85 × 24.4 = 20.7
     GiB".
   - On this grid, RSS reads low because macOS compressed part of the child's memory. The compressor grew from 1.3 to
     6.0 GiB during the run. Compressed pages leave RSS but stay in the footprint.
   - The two measures do not change the answer, but they differ by 4 GiB at the margin. You need to decide which one
     the author is given. I recommend giving both.
2. **The five-block footprint peak falls after the ADMM cycles, in the harness's terminal phase.** That phase follows
   the last IPOPT process and runs 16.5 min of the 48 min evaluation: the multiscenario terminal, the response terminal,
   the cost breakdown and the workbook.
   - Build plus 2 cycles alone peaks at 20.97 GiB RSS and 22.48 GiB footprint.
   - The terminal phase peaks at 19.60 GiB RSS and 24.99 GiB footprint.
   - Every full evaluation runs this phase once, so it belongs in the footprint. Whether it can be thinned is a harness
     question that this task did not cover.
3. **The other Worker's command lines tripped the harness's forbidden-process scan twice (zero solves each time).**
   - Their `/bin/zsh -c …` shell named `p515_s44_campaign_harness.py` in a `git status` argument, and
     `check_campaign_preconditions` matches the substring `p515_s4` anywhere in `ps aux`.
   - Both refusals are preserved as `srp1grid_r1_refused*` and `srp1grid_r2_refused*`.
   - I waited for that shell (pid 30498) to exit before the third attempt.
   - This is not a fault. But any concurrent agent that merely mentions the harness filename blocks every campaign
     launch on the machine.

## Changed

| commit | content |
|---|---|
| `bdb604e8` | `p515_s56_memory_probe.py` (new; no tracked file edited). It has four modes: clone export, zero-solve trial, solved run, manifest. |
| (this commit) | Evidence `data/SRP1/Results/P515S56/memory_probe_v1/` + `manifest_sha256.json` + this report |

**Isolation.** Both grids ran from a detached clone of `b0391b80` at
`…/scratchpad/memprobe/export_b0391b80`. The harness code at `b0391b80` is the latest committed: HEAD `28ba36b4`
changed only TASKS.md, and the other Worker had not committed.
- **Departure from the precedent export script:** this is a `git clone --shared` with a detached checkout, not a
  `git archive`. The campaign harness calls git in `freeze_campaign_spec` (HEAD, the case file's and ESS parameters'
  last commits) and in `check_campaign_preconditions` (`git status` of the production files), and an archive export
  has no repository.
- The clone was verified as follows:
  - HEAD = `b0391b80c388…`;
  - `git status --porcelain` empty;
  - all 990 root-level tracked files byte-identical to `git show b0391b80:<file>`;
  - the two instance files identical to the commit.
- The script was overlaid, untracked, from `bdb604e8`, sha256 `de635f4f…`. The record is `EXPORT_PROVENANCE.json`,
  copied into the evidence directory.
- The solver environment was `NLP_SOLVER_PATH=/usr/local/bin/ipopt` and `LP_SOLVER_PATH` set to the `.env` clp path.
  The clone has no `.env`.
- Each run was attached, alone and serial, with stdout and stderr captured. Both stderr logs are 0 bytes.

**Configuration, both grids** (existing harness keys only):
- The step 4 run configuration from `p515_s56_ev_harness_tests._spec_like`: case-file AA declared, ESS ageing
  baseline, convergence-depth tail.
- The derived SP instance.
- `interface_coupling` = V-1 (`per_scenario`, `row18_charge_reactive` False). AA is forced off by production; this was
  read back per cycle.
- No `interface_deviation_premium`, so α = 0 (production default).
- `s56_cohort_formulation` = `independent_units_all_node_years`.
- `s56_terminal_value_in_objective` = True.
- `s56_esso_slack_penalty` = 1e4.
- `flex_price_multiplier` = 2.0.
- No `s56_ev_profiles` (SP).
- Cap 2, concurrency 1.

**Staged plans:** `p515_s56_ev_harness_tests.STAGED[grid]`, already mapped onto each grid's years. Both are valid, so no
substitute plan was needed. Each has 4 cohorts, mixed durations (4 h, 2 h, 2 h, 2 h), and two cohorts at node 7.

| grid | plan (`GeneralLattice.label`) | `GeneralLattice.key` = entry key | eval key | I(x) |
|---|---|---|---|---|
| five-block | `n5_y2028_p0.5_e1__n7_y2025_p0.25_e1__n7_y2031_p0.25_e0.5__n9_y2037_p0.25_e0.5` | `89698816…` | `cff0df6a…` | 886,775.78 |
| SRP1-grid | `n5_y2030_p0.5_e1__n7_y2025_p0.25_e1__n7_y2030_p0.25_e0.5__n9_y2035_p0.25_e0.5` | `87170f8d…` | `2bddebff…` | 862,548.70 |

**Instances:**
- five-block: W89 `SRP1__s53_3x3.json` (case `2a64e3c5`, scenario checksum `0248a5c8`, the step 0 SP instance, not the
  comonotone variant).
- SRP1-grid: `instances_v1/SRP1__s56_srp1grid_3x3.json` (case `71d34849`, checksum `73d0a230`).

**Campaign specs:**
- `campaign_spec_s56memprobe_v1_five_block_9920cfa0.json`
- `campaign_spec_s56memprobe_v1_srp1grid_c827a307.json`

**Memory capture:**
- The launcher samples every 1 s; the largest observed gap was 1.07 s.
- Each sample reads RSS (psutil) and phys_footprint (`proc_pid_rusage`) of every process in the tree (launcher, harness
  child, IPOPT), plus `vm_stat`. Every sample is kept in `<grid>/samples.jsonl`.
- Watchdog, declared in the script: tree RSS ≥ 28 GiB, or swapouts ≥ 1 GiB since launch → SIGTERM the tree. It did not
  trigger on either grid.

## Found

| grid | peak tree RSS | peak single process (RSS) | peak tree footprint | s/cycle (1, 2) | build | pass/fail |
|---|---|---|---|---|---|---|
| five-block (20 blocks) | **20.97 GiB** | 20.52 GiB (harness child) | 24.99 GiB | 802.3, 555.1 | 216.9 s | PASS 18/18 |
| SRP1-grid 3 × 3 (12 blocks) | **15.03 GiB** | 14.79 GiB (harness child) | 12.47 GiB | 452.0, 307.6 | 131.7 s | PASS 18/18 |

**five-block ≤ 20.7 GiB: no** (20.97 GiB tree RSS; 24.99 GiB tree footprint).

- **Checks, each grid, all passed:**
  - exit 0 and 2 cycles;
  - **solve count exact**: five-block 249 = ((1+3)·5·4+3)·3 and SRP1-grid 153 = ((1+3)·3·4+3)·3, both declared and
    re-derived from the case before launch. Observed permitted solve/exec counts equal them, nothing was blocked, the
    base equals the declared count, 0 retries, identity holds;
  - no local-solve failure: 0 failures, every network-failure class 0, 0 ESSO recovery events;
  - the launcher guards, at 0 solves;
  - `GeneralLattice.key` = entry key, with no C5 reasons;
  - the capture path was asserted before any solve.
- **Configuration read back in the child** (all passed):
  - cohort mode `independent_units_all_node_years`, in force and on every probe ESSO model, with the row inventory ok;
  - TV in the objective, κ present per node, κ = a/σ on the run's models;
  - slack penalty 1e4, in force and on all four slack coefficient families per node;
  - flex ×2: 8/8 apply checks; the pre-run and terminal read-backs match;
  - staged write equals the canonical plan;
  - per-scenario coupling in force = the declaration; `interface_coupling_active` True and AA forced off on cycles 1
    and 2;
  - hook order: settings + staged write, then the multiplier.
- **Where the peaks fall:**
  - five-block: the RSS peak is at 20:42:38, in cycle 2. The footprint peak is at 21:01:58, in the terminal phase (see
    Blocked 2).
  - SRP1-grid: build + ADMM peaks at 14.82 GiB RSS / 12.47 GiB footprint, and the terminal phase at 15.03 / 12.35.
    Footprint stays below RSS here: the compressor held at about 2.0 GiB throughout.
- **IPOPT** never ran more than one process at a time; it peaked at 0.34–0.35 GiB.
- **The harness's own ru_maxrss of the child** agrees with the sampler:
  - five-block: 20.55 GiB (sampler: 20.52);
  - SRP1-grid: 14.84 GiB (sampler: 14.79).
- **Swap:** swapouts stayed at 16 pages, unchanged from before, on both grids. `vm.swapusage` was 0.25 M used before and
  after.

| vm_stat free + inactive (GiB) | before | at the RSS peak | min | after (+20 s) |
|---|---|---|---|---|
| five-block | 13.85 | 13.45 | 11.55 | 25.49 |
| SRP1-grid | 18.45 | 13.72 | 13.36 | 22.27 |

- **Outside the tree:** no other process reached 1 GiB in any read (every 30 samples). Before each run the largest
  were the Claude app processes (≤ 0.9 GiB). The other Worker's zero-solve search gate and driver ran at about 80 MB.
- **Wall time:**
  - five-block: child 2,851 s (47.5 min), of which production ADMM wall clock was 1,900 s.
  - SRP1-grid: child 1,673 s (27.9 min), of which production ADMM wall clock was 1,074 s.
  - For comparison, G-f r3 (SRP1-grid SP, x0, no cohort/TV keys) took 421 / 303 s per cycle and 15.9 GiB child
    ru_maxrss.
- **Cost breakdown** (2 cycles, identification only; convention above):

| grid | gross (cycle 2) | TV credit | total |
|---|---|---|---|
| five-block | 1,395,123,637.17 | 74,410.67 | 1,395,936,002.27 |
| SRP1-grid | 1,099,742,608.60 | 70,251.28 | 1,100,534,906.02 |

  - This is the first solved run with a nonzero TV credit.
  - Leak detector λ·L (conversion loss) is 88.22 € on five-block and 133.47 € on SRP1-grid, both above the 25 €
    ceiling. This is reported, not gated, and the point is unconverged.
  - ESSO P/Q slack sums are at the bound-relaxation level (≈ −1e-5).

## Not confirmed

- **Two cycles is not the full-run peak.** The terminal phase was included, but its size at a converged point, and the
  growth of the per-cycle state over 100+ cycles, were not measured. Audit B's 3 × 3 continuation ran for many cycles
  at 15.2 GiB ru_maxrss and 19.3 GiB campaign footprint; that is a pre-cohort, pre-TV formulation, so it is not
  comparable here.
- **Which measure the author wants** (RSS or footprint): not decided here (see Blocked 1).
- **What in the terminal phase holds the extra 2.5 GiB of footprint on five-block:** not attributed. No per-object
  memory profile was taken.
- **Reproducibility of the peaks:** one run per grid, with no repeat. RSS readings on five-block depend on how much
  macOS chose to compress, which depends on the other memory in use. Free + inactive before the run differed: 13.85 GiB
  for five-block, 18.45 GiB for SRP1-grid.
- The zero-solve trial runs were executed on a separate trial clone (script from the working tree) and are not part of
  the evidence. They exercised the freeze, the capture assertion, the sampler and the record reading on the committed
  EV smoke record.
