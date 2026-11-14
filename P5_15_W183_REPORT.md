# P5.15 W183 — code changes and gates for Step 7 (Addendum 83)

**Authority:** Addendum 83 (W183 order: W183a–f; zero solves except the named builds and the two-cycle bitwise gate).

**Code head after W183.**
- **Production `6f43479e`.**
  - `shared_resources_planning.py` `ade9c843…`.
  - `network_parameters.py` `30ea1b23…`.
  - `model_construction_helpers.py` unchanged, `a20a224f…`.
- **Campaign harness:** `p515_s44_campaign_harness.py` `c34ebdc5…`.
- **E1 launcher:** `p515_s55_e1_launcher.py` `320d503c…`.
- **Repo HEAD** after the parts: `c5b292dd`.
- **Overleaf** is not part of this order. Round 8 was committed meanwhile by the author/expert (`22e19b56`, `bd3a9f35`); W183 did not read or check it.

**Objective convention** (Addendum 83 ruling 2): Q = `gross_operational_cost`, contracted interface settlement
excluded.

**Runs.**
- **Solves: 153, all in the two-cycle bitwise gate** (51 × (cap 2 + 1), 0 retries). The gate's armed bounded guard
  verified the count exactly.
- **Builds:** every other part either built models with every `optimize` stubbed, or built nothing. All ran under
  `SolveProfileGuard(permitted=())`, verified at 0.
- **Pickle loads, every one declared and counted:**
  - 115 fixture loads in W183c gate 1;
  - 3 loads of row 2's `certified_models.pkl` in W183e (two trial runs and the real run; see decision 5).
- **Order of work.** Each part ran attached and alone, after the Advisor's design review. The parts that edit the
  evaluation key or production ran one at a time.
- **Interruption.** W183a was cut off by a usage limit mid-edit. A fresh Worker reviewed and kept its predecessor's
  uncommitted key-builder edit, then completed the task.

| part | content | report | commits |
|---|---|---|---|
| Advisor | engineering review of W183a/b/c/d/f before any change | (this report, "Design") | — |
| W183-0 | pre-change NL + component-digest reference (8 configurations × both DSO builders); G0's first run | `P5_15_W183_0_NL_REFERENCE.md` | `62c62d08`, `33dbf690` |
| W183e | E6 export | `P5_15_W183E_E6_EXPORT.md` | `5c1e6f23`, `d794f11b` |
| W183a | `rho0_scale` and `nrf_option` keyed harness options | `P5_15_W183A_OPTIONS.md` | `8d324c10`, `f61d8e5b` |
| W183c | NRF production option + terminal capture; gates 1–5 | `P5_15_W183C_NRF.md` | `6f43479e`, `e717850f` |
| W183b | E1 launcher + seed-key dry run | `P5_15_W183B_E1_LAUNCHER.md` | `cb36b22e`, `6a99e088` |
| W183d | G0 on the final production head | `P5_15_W183D_G0.md` | `dbf7a25c`, `d4ab31ca` |
| W183f | two-cycle bitwise gate (gate part; spec v1 freeze held) | `P5_15_W183F_BITWISE_GATE.md` | `17b4c1cf`, `c5b292dd` |

**Stop.** The two-cycle gate returned **GATE_PASS=False as frozen**, and Addendum 83 names no fallback. Under CLAUDE.md,
a gate failure other than G6/G27 stops the chain from W184 onwards. W183b also found that, as frozen, the E1 search
stops at its first poll (decision 2). **Spec v1 is not frozen; nothing is queued.**

## Decisions needed

1. **The two-cycle gate's single failed item (expert; a gate-scope ruling).**
   - **The finding.** One item failed: `vs_committed_w48_arm_zero_genuine_diffs`. All 44 "genuine" diffs are in
     `per_cycle_record.jsonl`: 22 keys × 2 rows, each absent in the W48 reference arm and `null` in the new arm.
   - The 22 keys are the harness's `PER_CYCLE_RESPONSE_FIELDS`, added to the record field list by W64 (`70950e8c`,
     2026-09-24). The W48 arm predates W64.
   - **Planner check.**
     - The 29 shared keys are bitwise equal on both rows, and none of the 22 extra keys is non-null.
     - The new arm's `per_cycle_record.jsonl` is byte-identical (`f8506905…`) to the committed W83 gate arm's
       (`P515S53/tight_tail_w83/srp1_bitwise_gate/arm/`).
   - **Precedent.** The W83 gate (`eb8cf81f`) failed on the same item for the same reason. No ruling on that failure
     exists in the brief, REVISION_CONTEXT.md, TASKS.md or the P5_15 reports. The tight tail was adopted afterwards
     (Addendum 48) without mentioning it.
   - **Everything else held:**
     - 153/153 solves;
     - the C\* reference rows[:2] with 0 diffs over the 29 trajectory fields;
     - presence 57/57;
     - Q = 740,922,817.2425401, equal to W48's;
     - 11 of the 12 compared files with 0 genuine diffs.
   - **Options:**
     - (a) Rule the absent-vs-null keys as field-list drift and accept the gate as passed. Cost: none.
     - (b) Substitute the whole-arm reference with a committed post-W64 arm (W83's, tail enabled but inert at 2 cycles)
       and compare.
       - (b′) Zero-solve: re-run only the gate's comparator on the committed W183f arm against the W83 arm.
       - (b″) Re-run the gate with the substitution. Cost: 153 solves, about 2 min.
     - (c) Declare a comparator rule for keys absent in the reference and null in the new arm.
   - **Recommendation: (b′), then (c) as a standing rule.** (b′) uses the gate's own convention, costs no solves, and
     also settles the unruled W83 failure.

2. **The E1 search stops at its first poll for any storage incumbent (expert; a method ruling, before W186).**
   - **The finding.** This uses the committed `p515_s53_f2_certificate.build_poll`, unchanged, at Halton k = 0 (t = 17).
     With any storage seed as incumbent, only 6–7 distinct feasible poll points remain after snapping, fewer than
     n + 1 = 8. The unit-box completion then exceeds the cap of 30, and the poll is refused.
   - Completion sizes: unit 31, node 7 0.5/2 34, historical incumbent 61; x = 0 gives 14 and runs.
   - A and N behave the same.
   - At 2× the unit's margin is positive, so the incumbent is likely a storage seed in both arms.
   - The refusal depends on k. The historical incumbent refuses only at k = 0 and 8; the unit runs only at k = 2; node 7
     0.5/2 runs only at k = 2 and 3.
   - **Options:**
     - (a) Accept a stop at poll 0: E1 then reports its 7 seeds and the cross-evaluations only, a seed comparison rather
       than a search.
     - (b) Change the completion rule or the cap for E1. This is a method change, and both arms would be matched.
     - (c) Choose the Halton start. Choosing it after this table is outcome-driven and needs an explicit, recorded
       ruling.
   - **Planner view.** If E1 is to remain a search, (b) with the cap raised to cover the largest seed's completion
     (61), declared before any E1 solve and applied to both arms, is the least arbitrary. The budget is ≤ 30 new
     evaluations per arm, so the cap and the evaluation budget would have to be reconciled. That is the expert's call.

3. **Spec v1 open points (expert).**
   - **The 4-slot batch root.** Proposed `P515S55/e1_2x_batch4`, with the cache root-scoped so batch records never seed
     an arm. The launcher already enforces root scoping.
   - **E2/E3 launchers are not built.** The order named none. A thin fixed-plan launcher over `freeze_campaign_spec` +
     `evaluate` is needed (precedent: ext v6). Proposed as W183g.
   - **"7 distinct seed keys"** read as 7 evaluations plus 1 G0-justified alias (x = 0 under N). It held under this
     reading.
   - **"Both DSO builders" in the bitwise gate** read as: the solve-bearing gate covers the sequential builder, which
     is the only one the configuration runs (`ParallelExecution: false`). The parallel builder is covered by W183c's
     zero-solve NL and component identity.
   - **The NRF binding tolerance** is `1e-3` MW, set by W183c. The binding quantity is the expected interface P per
     side. Per block, the minimum row body and the count of rows within tolerance are also recorded (at one scenario
     they coincide).
   - **The E1 barrier stop rule.** The launcher keeps the inherited Phase B rule A7 (stop after 2 new barrier
     evaluations in a poll, or 3 overall). Plan §9 says "stop for review on a barrier point"; the stricter reading
     needs a loop owned by the launcher.
   - **Launcher choices to confirm:**
     - Halton start k₀ = 0 per arm;
     - arm A declares no `model_variant` (this keeps A's keys equal to E.21 rows 35 and 40);
     - no post-certification on seed or search entries;
     - which four node-7 seeds the batch gate re-runs (6 of the 7 evaluated seeds hold node-7 storage).
   - **Not yet built:** the rule-v6 certified cross-evaluations and the 1× screen, which need declarations.
   - **The launcher's G0 pointer** for `--freeze` / `--run`:
     `P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz` `ca195bd3…`. `g0.json` records it.

4. **Two harness points to confirm (Planner kept both; the expert may overrule).**
   - **The `previous_iter.ess` guard.** W183a added a guard: with `rho0_scale` declared, `previous_iter.ess` must be off.
     Otherwise `rho_ess_prev` would start unscaled. The case file has it off. Kept.
   - **Undeclared runs now carry a `w183a_nrf_option` checklist entry.** Once production has the NRF marker, every
     undeclared evaluation records this entry and requires the flag off on every network. This changes the record
     format of undeclared runs; it does not change any key.

5. **E6 (expert).**
   - Panel (b) reports the distribution-network flexibility as a cost in €, not a volume in MWh. Row 1's MWh volume
     exists only in row 1's pickle. Recommend accepting the € measure.
   - Row 2's pickle was read **three** times: two trial runs and the real run, each under the armed one-file
     allow-list. The ruling said "a zero-solve read". Nothing was modified, but this is recorded.
   - The export also holds row 2's node-7 schedule for all 12 blocks, labelled "supporting".

## Blocked on the author

- **Push from PyCharm.** The terminal cannot authenticate; the branch is ahead of origin.

## Changed

| commit | content |
|---|---|
| `62c62d08`, `33dbf690` | `p515_s53_w183_nl_digest.py`; `P515S55/w183_nl_reference_pre/` (gz JSON 10.2 MB; ESSO NL committed; network NL hash-recorded) |
| `5c1e6f23`, `d794f11b` | `p515_s53_w183e_e6_export.py`; `P515S55/e6_export/` |
| `8d324c10`, `f61d8e5b` | harness options `rho0_scale` / `nrf_option` (+284 lines in `p515_s44_campaign_harness.py`); `p515_s53_w183a_option_tests.py`; `P515S55/w183a_option_tests/`, `w183_nl_post_w183a/` |
| `6f43479e`, `e717850f` | **production:** `shared_resources_planning.py` (+39), `network_parameters.py` (+10); harness NRF capture; `--nrf` flag in the digest harness; `p515_s53_w183c_nrf_gates.py`; `P515S55/w183c_nrf_gates/`, `w183_nl_post_w183c_{off,on_x0,on_unit}/` |
| `cb36b22e`, `6a99e088` | `p515_s55_e1_launcher.py`; `P515S55/w183b_e1_dryrun/` |
| `dbf7a25c`, `d4ab31ca` | `p515_s53_w183d_g0.py`; `P515S55/w183d_g0/g0.json` (`8cef45d6…`); `P515S55/w183d_e1_dryrun_final_g0/` |
| `17b4c1cf`, `c5b292dd` | `p515_s55_w183f_bitwise_gate.py`; `P515S55/w183f_srp1_bitwise_gate/` + launch log |
| this commit | `P5_15_W183_REPORT.md`, TASKS.md, REVISION_CONTEXT.md |

**Production diff (the Planner read it in full).**
- `NetworkParameters.no_reverse_flow = False`. The params reader sets it only from the key `no_reverse_flow`, which
  must be a JSON bool; no params file declares it.
- Module-level named rules: TSO `pc_adn[dn,s_m,s_o,p] >= 0` and DSO `pg_adn[s_m,s_o,p] >= 0`.
- Constructed only when the option is on, beside the expected-interface-P constraint in the TSO builder and in both
  DSO builders.

No committed artifact was overwritten, and no earlier gate or harness was edited or re-run onto its output.

## Found — predictions against outcomes (expert, Addendum 83)

| prediction | outcome |
|---|---|
| G0 identical at x = 0 | **held**, before the change (W183-0) and on the final head (W183d, G0_PASS). ESSO NL identical per node; network NL and component digests identical on all 48 sequential + 36 parallel blocks; ESSO digests differ only in `energy_storage_capacity_degradation`, every differing index inactive. Unit at 2×, N vs A: only the 6 degradation rows ([1],[2],[4],[5],[7],[8]) differ; `.col` identical. Combined key ≠ single-option keys ≠ base. A second process rebuilt two configurations byte for byte |
| the NRF-off NL identity holds | **held** — 696/696 blocks byte-identical (NL, labelled NL, `.row`, `.col`, block and component digests) to the pre-change reference; 8/8 eval keys equal. With the option on: TSO 12 × 72 = 864 and DSO 36 × 24 = 864 rows in each builder, all active, nothing else differs; absent from every off build. 115/115 preserved fixtures unpickle |
| the two-cycle gate passes | **failed as frozen** on one item: absent-vs-null drift of the 22 W64 keys, identical to the unruled W83 failure (decision 1). Every other item held |
| the E1 key dry run yields 7 distinct seed keys | **held** — 7 keys to evaluate + 1 alias (x = 0 under N → x = 0/A/2×), accepted with production equal to current |

**W183a (no prediction).**
- **T1:** all 7,213 entries of the 252 v2 campaign specs recompute to their stored keys, and equal the pre-change
  harness's. 11 v1 specs carry no key and were skipped.
- **T2:** the scale 1.0 and `False` give the same key as an undeclared entry; 0.5, 2.0 and True change it. Wrapper
  threading is proven on resettle, continuation and extension entries.
- **T3:** all 13 ρ entries equal S39 × s exactly at 0.5 and 2.0. Every S39 refusal is kept, including under a declared
  scale.
- **T5:** 174/174 blocks are identical.
- **Not measured:** that ρ₀ reaches the built ADMM Params (established by code reading:
  `p515_g_g1_g4_admm_gates.py:1275` before `:1283`; `shared_resources_planning.py:5112–5122, 5378–5388, 5483`).

**W183b (no prediction).**
- The A-arm keys equal E.21 rows 35 and 40 (W118 r2) exactly.
- Row 40's F2 Phase B namesake differs only in `convergence_depth_tail`.
- The unit's N key at 1× reproduces row 54's base key `054976fb…`, so `eol_retention_r` 0.50 keeps the key space
  consistent.
- The cache loader is root-scoped. A planted sibling root with the same keys at different costs was never read.
- Memory preflight at 4 slots: 11.0 GiB required, 14.63 GiB available (single measurement).

**W183e (E6).**
- The declared day is **2025 Spring**: 40,181.22 €, 15.8 % of the 253,539.62 € value. It leads the next day by
  12,170.61 € against a bar of 103.37 €.
- Value on that day: TN −39,756 €, DN flexibility −423 €. Horizon: TN −158,938.59 €, DN −94,581.13 €.
- SoC reaches both the 10 % and 90 % limits of E^Av(2025) = 0.880458 MWh.
- E^Av by year: 2030 0.779817, 2035 0.69999999 MWh (at the floor).
- Cross-checks:
  - JSONL schedule = pickle at k\*, bitwise;
  - SoC vs net-power recursion: 5.6e-8 (TSO copy), 2.7e-5 MWh (DSO copy, small simultaneous charge/discharge);
  - per-block costs vs the record: ≤ 1.5e-8 €.

**Design (Advisor, before implementation).** Adopted:
- key fields enter only when declared and non-default, threaded through the four wrapper branches;
- NRF rows constructed only when on, in `shared_resources_planning.py` + `network_parameters.py`. An MCH edit would
  have made the gate refuse on its file pin;
- both NL and component digests gated, since NL identity alone passes a construct-and-deactivate implementation;
- the pre-change reference captured first;
- the 7-key alias reading;
- the gate as sequential-only.

The Advisor also found that row 54 is a resettle key. Base-key equality needs the ext-v6 configuration block; the
launcher uses it.

## Not confirmed

- **ADMM-cycle models.** Only the models handed to IPOPT at the initialisation solve were built. Persistence of the
  NRF rows and of the scaled ρ through `update_*_to_admm` rests on code reading.
- **The parallel DSO builder in a worker process** (ProcessPoolExecutor pickling of models with the new rules) was not
  exercised. It was called in-process only.
- **Not executed:** the launcher's `--freeze` / `--run`; the harness child's NRF and ρ branches through a launched,
  solving child; NRF on in any solving run.
- **Process-level absence of shared writes at 4 slots:** read, not traced. The batch gate inside E1 measures it.
- **W116 spec v3's source search** (`p515_s53_w116_benchmark_spec_v3.py` l. 692) would now find `no_reverse_flow` in
  production. Its committed record describes pre-W183c code. It was not re-run.
- **W183a's test script T4** asserts "no production NRF marker", which is now false by design. Do not re-run it as is;
  W183c gate 5 is its inversion.
