# P5.12-X — comparator reconstruction and sensitivity extension

**Authorship note.** Planner-executed; the Worker remained unavailable (external API
spend limit). Reconstruction plan, baseline acceptance rule and execution order were
frozen to `data/SRP1/Results/P512X/frozen_reconstruction_plan.json`
(SHA-256 prefix `313d40f00f71006083f154e7`) **before** the first solve. Materiality
thresholds were reused verbatim from P5.12-W; none was redefined afterwards.

## Instruction conflict — TSO comparator NOT RUN

The stage mandates a baseline verification at `warm_start_bound_push = 1e-5` *and*
that all other settings match the historical baseline. For **TSO case9, 2025 Summer,
cycle 7** the historical value is **1e-6** (inherited from case9's configured
`bound_push = 1e-6`; its cycle-7 option echo also shows `acceptable_iter = 0`,
`acceptable_tol = 1e-5`, `compl_inf_tol = 5e-4`). Running 1e-5 would be a 10x
perturbation, not a baseline, and the gate would fail by construction; running the
mandated LOW = 1e-6 would merely duplicate its baseline and HIGH = 1e-4 would be a
100x perturbation. No substitution was made. Proposed resolution, for user decision:
apply the same +/-10x ladder around this fixture's own baseline — BASELINE 1e-6,
LOW 1e-7, HIGH 1e-5.

## DSO comparator — reconstruction gate PASSED (byte-identical)

Fixture: DSO `case33_2`, node 7, 2025 Autumn, cycle 7 (preserved pre-solve model from
the P5.12-R cold trajectory). Missing context (network/`NetworkParameters`) was
reconstructed from `fresh_planning`, which enforces the canonical scenario checksum;
the preserved model remained authoritative for every numerical value.

The baseline replay at `1e-5` reproduced the historical cycle-7 trace
(`optim_log_case33_2_2025_Autumn.log`, 8th solve, lines 69446-77136)
**byte-identically: 0 differing lines in 7691**, after normalizing only the log path,
elapsed time and the fresh-file leading blank line. Option echo, iteration-0 row,
full iteration table, 91 iterations, termination, final residuals and evaluation
counts (104 / 91) all match exactly.

**This establishes that preserved comparator fixtures are faithfully replayable from
(preserved pre-solve model + deterministically reconstructed context).**

## Sensitivity result

NL hash identical across all three runs (`0a40d1e6…`); only `warm_start_bound_push`
and `output_file` differed.

| | LOW `1e-6` | BASELINE `1e-5` | HIGH `1e-4` |
|---|---|---|---|
| termination | Optimal | Optimal | Optimal |
| iterations | 102 | 91 | 125 |
| objective (unscaled) | 354.16017301652028 | 354.15536755956111 | 354.15407497225556 |
| constraint violation | 3.87e-07 | 8.22e-08 | 8.13e-09 |
| safeguard / restoration | 0 / 0 | 0 / 0 | 0 / 0 |
| relative objective difference | 1.357e-05 (indeterminate band) | — | 3.650e-06 (indeterminate band) |
| max scaled primal distance | 6.582e-02 (material) | — | 3.350e-02 (material) |
| columns > 1e-3 | 279 | — | 53 |
| iteration change | +12.1% | — | +37.4% (below 50% path threshold) |
| `expected_interface_vmag` / `_pf_p` / `_pf_q` max scaled difference | 1.89e-09 / 4.94e-09 / 1.60e-08 | — | 1.37e-09 / 3.09e-09 / 7.37e-09 |

### Classification (multi-axis, frozen P5.12-W thresholds, no post-hoc change)

The frozen objective-equivalence criterion is relative difference <= 1e-6. LOW is
1.36e-05 and HIGH is 3.65e-06, so **neither variant lies inside the equivalence
region**; neither reaches the 1e-3 material threshold either. The frozen rules
therefore place both in the **indeterminate band**, and the fixture must NOT be
recorded as fully ROBUST.

Recorded classification:

- **solve validity: ROBUST** — BASELINE, LOW and HIGH all Optimal.
- **objective equivalence: INDETERMINATE** — 1.36e-05 (LOW), 3.65e-06 (HIGH); above
  the 1e-6 equivalence bound, below the 1e-3 material bound.
- **interface/propagating-output equivalence: EQUIVALENT** — interface vmag / pf_p /
  pf_q agree to <= 1.6e-08, far inside 1e-6.
- **branch/regime: EQUIVALENT** — the predeclared branch rule requires primal AND
  objective both > 1e-3; the objective difference is 1.4e-05. The material primal
  motion is compensating null-space redistribution (`qg[1,0,0,19]` +0.0658173 against
  `qg[2,0,0,19]` -0.0658124, cancelling to ~5e-06), i.e. internal dispatch
  non-uniqueness, not a different branch.
- **path sensitivity: NOT MATERIAL** — +12.1% / +37.4% iterations, below the frozen
  50% threshold; zero safeguard and zero restoration rows.

Objective reproducibility here (1.4e-05 relative) is five orders of magnitude softer
than `TARGET_cycle20` (2.2e-10); the two fixtures are not equally robust.

### Planning-unit significance of the objective difference

Block weight = `N_2025 x D_Autumn x annualization` = `5 x 91 x 1.0` = **455**
(single market/operation scenario). The LOW objective gap of 4.805e-03 block units
maps to **≈2.19 planning units**; HIGH maps to ≈0.59. Against the accepted
cross-depth uncertainty of 22.09 and a best-to-second gap of 32.87, one block's
option-sensitivity is ~10% of the uncertainty budget — non-negligible, not decisive,
and recorded as a contributing noise source rather than a classification change.

## Breadth counts (previously-unknown fixtures; positive control excluded)

Reported per axis rather than forced into one bucket.

| fixture | validity | objective | interface/output | branch | path |
|---|---|---|---|---|---|
| `TARGET_cycle20` | ROBUST | EQUIVALENT (2.2e-10) | EQUIVALENT | EQUIVALENT | not material |
| DSO `case33_2` comparator | ROBUST | **INDETERMINATE** (1.4e-05) | EQUIVALENT (1.6e-08) | EQUIVALENT | not material |

- NEW fixtures tested: **2**
- OUTCOME FLIP (validity): **0**
- OPERATIONAL-OUTPUT / BRANCH FLIP: **0**
- PATH-ONLY: **0**
- Fully ROBUST on all axes: **1** (`TARGET_cycle20`)
- Robust in validity/interface/branch but objective-equivalence INDETERMINATE: **1**
  (DSO comparator)
- Not run: TSO `case9` comparator (instruction conflict, referred to user)

Solves executed this stage: **3**.

## Verdict

BREADTH PROBE INCONCLUSIVE
