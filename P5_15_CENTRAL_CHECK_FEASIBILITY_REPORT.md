# P5.15 — centralised single-day check, Step 1: feasibility (2026-11-16; zero solves)

**Request:** the Expert's handback "centralised single-day check (feasibility first)", with the author's freeze
reopening for these values only. This report answers Step 1. **Nothing was solved:** a build-only probe ran under
`SolveProfileGuard(permitted=())`, verified at 0 solves and 0 executions, with pickle loads at 0.

## Decisions needed (before Step 2 can be frozen)

1. **The merged model's demand band (Expert).**
   - **Cause.** The merge reuses the transmission model, which holds loads as variables within ±10⁻⁵ pu of demand
     (`pc_bounds`, `qc_bounds`). That is ±1 kW and ±1 kVAr per load-hour. The DSO models hold the same loads as fixed
     parameters.
   - **Effect.** Over 96 loads and 24 h, the band can lower the centralised cost by at most **199, 209, 240 and 238 €**
     (Spring, Summer, Autumn, Winter). That is 1.2–1.4 × 10⁻³ of the day's coordinated cost. The coordinated
     evaluation resolves Q to about 6 × 10⁻⁶ (band 4.2 k€ on 653.9 M€), so the band would dominate any coordination gap
     near that resolution. The sign is known: it can only lower the centralised cost.
   - **Options:**
     - (a) **Fix pc and qc at demand in the harness (recommended).** One statement per day model. It reproduces the DSO
       models' treatment exactly and adds no constraint, variable or term.
     - (b) Run the merge as is and report the band bound beside every difference.

   Under the handback's rule ("model needs changes beyond merging → stop and report") this is your call, which is why
   I stopped here.
2. **Solver settings for the frozen spec (Expert).** The merged network inherits the TSO's IPOPT options:
   tol 10⁻⁵, compl_inf_tol 5 × 10⁻⁴, MA97, and a recovery tier with acceptable_tol 10⁻⁴ and acceptable_iter 1.
   - **Proposal:**
     - tol 10⁻⁵;
     - **compl_inf_tol 10⁻⁶**, the tightened value used once the residual test passes;
     - MA97;
     - **recovery tier disabled**, so each day has one primary attempt and a failure is recorded, not retried with
       relaxed options.
3. **Step 2 go (author).** Confirmation to run the four 2025 days under a frozen spec with recorded predictions, by
   17 Nov.

## Blocked on the author

- The Step 2 confirmation (decision 3).
- Push from PyCharm.

## Changed

- `21eb51b2`: `p515_s55_central_feasibility_probe.py`. It uses production functions only and edits no production file.
- Output in `data/SRP1/Results/P515S55/central_feasibility_probe/`: `central_feasibility_probe_v1.json`, `launch.log`,
  `launch_stderr.log` and `manifest_sha256.json`.

## Found

### 1a. Can the existing code build the merged single-day model without new model development? **Yes.**

**The production path.**
- `centralized_coordination.combine_networks(TN, DNs)` is the merge behind `run_operational_planning(type='centralized')`.
- Each DN reference bus is replaced by its TN interface bus; DN branches attach directly to that bus.
- The DN reference generator and the TN's interface loads are dropped, so the interface quantities are the physical
  flows rather than consensus copies.
- `Network.build_model(params)` on one day's merged network builds that day alone. This is the call
  `validate_vmag_refactor.py` already made, as a structure check only. The merged model had never been solved.

**All four 2025 days build without error.** Every structural difference from TSO + 3 DSOs is accounted for, row by row:

| item | TSO + 3 DSOs | merged | why |
|---|---|---|---|
| nodes (node-balance, voltage rows) | 2,592 rows | 2,520 | 3 DN reference buses merged into the TN buses (3 × 24) |
| `voltage_mag_def`, `vmag` | 144 | 0 | the explicit interface voltage exists only for consensus |
| shared-ESS rows (`sess_*`) | 144 each | 0 | x = 0: fixed at 0 and inactive in the coordinated models; dropped by the merge |
| `pg`, `qg` | 576 | 504 | 3 DN reference generators (the interface) dropped |
| flexibility vars and day-balance slacks | 2,376 / 99 | 2,304 / 96 | the TSO's 3 interface loads dropped (their flexibility is outside Q) |
| `interface_delta_p/q` | 72 | 0 | consensus-only, fixed at 0 |
| `pc`, `qc` | 72 Vars + DN Params | **2,304 Vars** | **the demand band (decision 1)** |

- No constraint family is missing, and the merge adds none.
- **Parameters.** TN and DN parameters are identical except the solver options (decision 2).
- **Data.** Energy prices, flexibility prices and base MVA (100) are equal in all 12 day × DN pairs. The interface
  voltage limits are equal ([0.90, 1.10] on both sides).
- **Kept DN generators** are all RES (types 2 and 3), not controllable. They carry no generation cost, as in the DSO
  models.
- **Objective.** The merged objective has the same terms as Q (generation + flexibility + slack penalties + zero
  terms). The interface settlement is identically 0, since there are no ADN nodes.

### 1b. Build and solve time, memory

- **Build:** 0.5–0.65 s per day. The whole probe (read the planning problem, merge, build 4 merged + 16 network models)
  took 16.8 s.
- **Memory:** peak RSS 447 MB for the whole probe.
- **Size:** 38,184 variables (37,992 free) and 24,075–24,207 constraints per day, against 35,832 and 25,881–26,013 for
  TSO + 3 DSOs.
- **Solve:** not measured, because measuring it needs a solve.
  - **Estimate:** about 10 s – 2 min per day with MA97, under 1 GB. From a default start, the TSO alone took 19–25
    IPOPT iterations in the coordinated run (`network_ipopt_solve_records.jsonl`, round 0). The merged problem is about
    10× larger.
  - The whole Step 2 run should take well under an hour.
  - **Risk:** a default-start AC OPF on 105 buses may stop at a local point or fail; the spec records that, it does not
    retry.

### Comparator (the handback's, for Step 2)

Source: `ref:7aa017f0`, `component_levels_terminal.json` (`033c1e7e`), 2025, unweighted € per day, TSO + 3 DSOs.

| day | TN generation + DN flexibility | detector terms | Q day |
|---|---|---|---|
| Spring | 139,122.4 | −2.6 | 139,119.8 |
| Summer | 180,444.2 | −2.6 | 180,441.6 |
| Autumn | 169,308.5 | −2.6 | 169,305.9 |
| Winter | 182,644.3 | −2.6 | 182,641.7 |

The interface exchanges for the requested comparison are available from both sides without a solve:
- **merged:** the flows on the DN branches attached to the TN buses;
- **coordinated:** the hourly interface P and Q in the certified models of `d110bd1a`.

### Proposed Step 2 predictions (to be frozen before the run, with the spec)

- **P1.** All four days end "Optimal Solution Found" from the default start.
- **P2.** With decision 1 (a), centralised ≤ coordinated + 1 € per day, because the coordinated point is feasible in the
  merged model up to the consensus tolerance. This holds only if IPOPT finds the same basin; a higher centralised cost
  would mean a different local optimum, not a coordination gap.
- **P3.** |difference| < 10⁻⁴ of the day's cost on every day, with interface exchanges within 0.1 MW per hour.

## Not confirmed

- **Solve time and convergence:** not measured.
- **Whether the default start reaches the coordinated basin:** unknown until Step 2.
