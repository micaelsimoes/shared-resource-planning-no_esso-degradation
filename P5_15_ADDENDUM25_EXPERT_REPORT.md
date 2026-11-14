# P5.15 Addendum 25 — handoff to the External Expert and the Author

**Planner report, 2026-09-19.** Self-contained; continues `P5_15_S43_STEP37_REPORT.md`. Item-by-item detail is in
`P5_15_S44_SELECTION_REPORT.md` (`34ac8d99`); the harness gate ruling is in `P5_15_S44_GATE_RULING.md` (`8ce0b872`).

- **Configuration:** oracle D as in the case file (`fb3de341`) unless stated.
- **Cost convention:** `gross_operational_cost`, settlement excluded.
- **Order carried out:** alias fix + campaign harness → AA variant arm → selection run → paper-scale build, with the
  Addendum 26 confirmations alongside. Everything is committed; nothing is running.

## 1. Verdict

1. **D certifies across the design space tested.** It certified at all four candidates within cap 500, with zero
   local-solve failures.
2. **The AA `keep_memory` arm meets Addendum 25's adoption rule.** It certifies faster than D at all four candidates,
   with (b)–(d) passing at each. The case file is **not yet updated**; Addendum 25 places that "before Phase A", so it
   awaits your confirmation.
3. **Paper scale does not fit 32 GB as currently built,** but for a narrower reason than the ~100 GB estimate. The
   model state is about 18 GiB. What crosses the 24 GiB line is the snapshot machinery.
4. **Three Addendum 26 confirmations bear on the Step 4 spec and need author decisions:** the cost file, the budget,
   and what `max_capacity` limits.

## 2. The selection run

| candidate | D certifies at | AA certifies at | saved | (b) \|Q_AA − Q_D\| (band) | (c) residual | (d) hull, 48/48 |
|---|---|---|---|---|---|---|
| C\* (0.96875 / 3.875 at 5, 7, 9) | 139 | **107** | 23 % | 15,965 (97,645) | −8.8e-8 | 0.00025 % |
| paper's plan (node 7 only, 1.62 / 3.24) | 139 | **116** | 17 % | 312 (97,954) | 3.1e-7 | 0.00027 % |
| node-7-empty (C\* at 5, 9) | 136 | **107** | 21 % | 18,526 (97,785) | 1.0e-7 | 0.00032 % |
| 2×C\* (1.9375 / 7.75 at 5, 7, 9) | 187 | **180** | **4 %** | 12,893 (97,221) | −3.6e-7 | 0.000025 % |

- **The AA arm.** Keeping the memory on rejection improved C\* from 109 (Step 3.7) to 107. Both predictions missed:
  the expert's 80–95 and my 78–100. So memory loss was not the main limit on AA.
- **The saving shrinks with storage size**, from 17–23% at the three moderate candidates to 4% at 2×C\*. There,
  storage is the slowest channel and recovered network failures are frequent.
- **Every AA-vs-D cost difference reconciles exactly** to generation plus internal flexibility cost, with every other
  component zero. Each is within its band. Under the one-configuration rule these AA costs replace, and are never mixed
  with, D's.
- **Harness gate.** C\* through the new harness, concurrent with two other evaluations, reproduced D exactly on every
  numeric field and on cost. The strict comparator's 43 differences were tie **order** in one diagnostic list, from my
  deliberate sort fix; the committed analysis explains all 139 rows. The Worker correctly reported the failure rather
  than overriding it. I ruled on it.
- **Per-node zero storage evaluates cleanly.**
- **Memory per evaluation** is 2.2 GiB, or 3.4–3.5 GB with the post-certification hull polish. That allows about 10
  concurrent evaluations on 32 GB, or about 7 with polish.

## 3. Paper scale: what was measured

Build only, zero solves, run alone, under a 24 GiB watchdog.

- **The instance:** 5 representative years (2025–2037, three calendar years each) × 4 days × 25 scenario combinations
  (5 market × 5 operation). It was built from data already in the repository and the one historical SRP1
  configuration of that shape.
- **Scenarios live inside each block, not across blocks.** The instance is **80 network blocks (48 at SRP1), each
  about 25× larger**: a DSO block has about 269k variables and 191k constraints. It is not about 2,000 small blocks.

| stage | memory footprint |
|---|---|
| 60 DSO blocks | 16.2 GiB |
| + 20 TSO blocks, ESSO models, ADMM preparation | 18.3 GiB |
| + pristine TSO/DSO clones kept for writing failure snapshots | **> 24 GiB → watchdog abort** (407 s) |

**Reading:**
- **One serial evaluation is plausibly feasible on 32 GB** if the snapshot clones are disabled or rebuilt on demand.
  They are diagnostics, not part of the method.
- **A parallel campaign at paper scale is not feasible** on this machine.
- **Cycle time is unmeasured.** With each network NLP about 25× larger, SRP1's 35 s/cycle does not transfer.
- **Four code paths do not yet support a multi-scenario instance:**
  - `p56a_oracle.load_baseline` is hard-wired to `SRP1.json`;
  - `run_admm_arm` assumes 51 solves per cycle;
  - some code reads only scenario (0, 0);
  - the ESS workbook's investment-cost scenario probabilities overwrite `prob_market_scenarios`. This one is
    uninvestigated and could be a genuine defect.

## 4. Addendum 26 confirmations — three need author decisions

| item | finding |
|---|---|
| **Cost file** | Committed `SRP1_ESS.xlsx` (sha256 `14581474…1414`) is **not confirmed** as the corrected file. A later version, `7ce1d1ab` "Costs updated." (2026-07-29, every energy cost ×1.25), exists **only on `paper_revisions`**, outside this branch's history. STEP4's quoted unit costs match the committed, older file. |
| **Budget** (committed file, €, 2025) | Paper's plan I = 1,073,285 (**−73k** vs B = 1e6); C\* I = 3,105,985 (**−2.1M**); lattice plan 1.5/3.0 at node 7: I = 993,782 (+6k); lattice C\* 1.0/4.0 all nodes: −2.2M. With the newer file **every** one of these is over budget. `budget` and `max_capacity` exist only in the Benders master, not on the oracle's path. |
| **`max_capacity`** | Caps **energy** at 5 MWh (`shared_energy_storage_data.py:360`), not power as STEP4 §1.2 reads it. 2×C\* (7.75 MWh) lies outside it. |
| **x = 0** | Builds cleanly: variables, normalization and captures all work. Solving at all-zero under D is untested. |
| **ESSO capacity multipliers** | Imported on every ESSO solve and already preserved in the committed models, at no extra solves. They are in scaled ADMM units with unestablished sign, energy duals are ≈ 0 in the ESSO, and they are **not yet shown to be ∂Q/∂x**. |

**What this means for Step 4.** Under the budget as it stands, the feasible design space is small: roughly one
~1 MVA unit in total at 2025 costs.
- **The selection candidates were larger than the campaign will search.** C\* and 2×C\* sit far outside the feasible
  space. That is fine for a configuration-robustness test, but the robustness evidence is for larger storage than the
  campaign's region.
- **The campaign's own region is untested for certification.** That region is single-node, sub-MVA designs, down to
  x = 0.

## 5. Recommendations

1. **Adopt AA-on (`keep_memory`).** The rule is met at every candidate, costs reconcile exactly, and the hull polish
   passes.
   - **Before Phase A:** write it into the case file, then re-verify that the case-file-alone run reproduces the AA C\*
     evaluation bitwise (107 cycles, 650,982,939.94).
   - **Record with the adoption** that the saving falls to 4% at the largest candidate. At sizes above the budget the
     campaign should not expect AA to save much.
2. **Decide the cost file first.** It changes I(x) at every point, and with it which lattice points are budget-feasible.
   Then decide `max_capacity`'s meaning, which sets the lattice's upper bound.
3. **Begin Phase A with a robustness batch in the budget-feasible region:** x = 0, the smallest lattice units at each
   node (0.25 MVA with 0.5 / 1.0 MWh), and the lattice plan. Evaluate them under the adopted configuration before the
   ladders proper. This extends "does the oracle certify across the design space" to the region the campaign will
   actually search.
4. **Paper scale:** a bounded task to
   - (a) make the pristine snapshot clones switchable, off for campaign and verification runs;
   - (b) generalize the four single-scenario code paths, including a check of the probability overwrite;
   - (c) re-run the build measurement and time one cycle.

   Take the campaign-instance decision on that measurement. Candidates are SRP1 for the search with paper-scale
   verification of the final incumbent, or a reduced scenario count.
5. **ESSO multipliers:** defer. They matter only if §5.6's optional search step is used. If it is, validate sign and
   units first with a small finite-difference check at C\* (two extra certified runs).

## 6. Questions for the Expert and the Author

1. Confirm AA-on adoption, given the 4% margin at 2×C\*?
2. Which cost file is the corrected one: the committed file, or `7ce1d1ab` from `paper_revisions`?
3. `max_capacity`: energy ≤ 5 MWh (production) or power ≤ 5 MVA (STEP4)?
4. Should the Phase A robustness batch in the budget-feasible region (§5.3) come before the ladders?
5. Paper scale: authorize §5.4's bounded task? And provisionally, which campaign instance?
6. Should the node 7 result be worded as recorded: "the DN's interface branch at node 7 is the only active interface
   constraint"?

## 7. Evidence index

| item | commit |
|---|---|
| spec v14 | `cdeaa940` |
| alias fix; harness, checks, gate, tie analysis; ruling | `8682cfdd`; `a60ea791`…`f825b8b7`; `8ce0b872` |
| follow-ups, AA variant, flag-off, post-certification step, smoke | `631d4183`, `42f48b92`, `1fdf3142`, `8f2b2736`, `92ddc722`, `1371dd73` |
| variant at C\*; D at 2×C\* | `d178c5c6` |
| selection AA campaign (script, dry run, run) | `68ec56ac`, `a72dd1f4`, `7b7a6078` |
| Addendum 26 confirmations | `acb74b5b`, `c3960b68`, `96fa8cc8`, `6aab57d8` |
| paper-scale script, calibration, build | `5f6625db`, `5fad6334`, `37e5d86f` |
| stage report; `REVISION_CONTEXT.md` | `34ac8d99` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted certified models, per-entry strides,
`esso_capture/` and `results/` are hash-recorded in the campaign manifests, not committed.
