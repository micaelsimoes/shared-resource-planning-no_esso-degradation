# P5.15 W184 — E3: coordination under a matched no-reverse-flow restriction (Addendum 85 W184 line; Addendum 86)

**Authority:** Addendum 85 (W184 line), re-authorized by Addendum 86. Spec v1 `frozen_step7_campaign_spec_v1_96345241.json`.
**Cell:** `e3_cnrf_x0_1x`, x = 0, 1×, `nrf_option` True, certified Step-7 resettle protocol (rule v6), cap 300, persist on.

**Code:**
- production `6f43479e`;
- harness `d9a944f2…`;
- hooks `9318ade3…`;
- launcher `3b990d19…`;
- cell spec `1363bcef…`.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded; salvage 0 at x = 0.

**Run.** One certified evaluation.
- Exact frozen command, attached, alone, both streams captured.
- Exit 0, 4,892.7 s (1.36 h), peak memory 3.70 GB.
- No other project process ran.
- Worker report: `P5_15_W184_E3_RUN.md`.
- Commits: `d7d4875e` (run outputs and launch log; 664/664 manifest entries verified), `bb4afafc` (diagnostics
  script), `41cc3433` (report and outputs).

## Decisions needed

1. **H3a failed; plan §3's fallback applies (expert, reporting only).** The reverse-flow term is determinate, not within
   resolution. Plan §3 and §8 name the fallback: "the paper reports both conditional terms and the headline becomes the
   NRF-conditional saving; the reverse-flow term is a separate, stated contribution of the interface arrangement."
   - The fallback changes reporting, not configuration.
   - E2 does not depend on E3's outcome (Addendum 85 ruling 8).
   - W185's two conditions hold (a declared status; 864 active rows per copy).
   - **W185 therefore proceeds** (CLAUDE.md: a reporting stop does not idle independent cells, Addendum 63). The §4.4
     wording is round 9.
2. **Coordinated curtailment columns (author/expert).** Table 12 and E.23 print `---` for the coordinated rows, so there
   was nothing to reproduce. Substitute checks passed:
   - the formula reproduces Table 12's static columns at one decimal (807.6 / 0.6; 781.7 / 201.2);
   - it reproduces report_v3's committed coordinated record for row 1 bitwise.

   Whether the coordinated rows should now print values depends on the net-versus-parts convention still pending with the
   author (W109).

## Blocked on the author

- Push from PyCharm.
- The W109 curtailment convention (decision 2).

## Found — hypotheses against outcomes

| hypothesis | outcome |
|---|---|
| H3a: Q(C_NRF) − Q(C_OPEN) positive and within resolution (< 2τ = 9.1 k€) | **failed on the margin; the sign held.** +20,463.88 €; both cells certified (bands 4,155.22 and 4,209.17 €); rule-v6 threshold max{3 × 4,209.17, 2τ} = 12,627.51 €; **determinate, positive, 1.62×**; 3.13 × 10⁻⁵ of Q(C_NRF). Under the bar refinement this rules out stopping slack only; it is not a statement about the limit. The between-configuration spread measured at C\* (71,926 €) is 3.5× the term; plan §3 applies that spread only to the mixed comparison |
| H3b: Q(P_NRF) − Q(C_NRF) = 90.9 ± 0.1 M€, 12.2 % at one decimal | **held** — 90,876,144.52 €, 12.2019 % of Q(P_NRF) = 744,770,310.59 € (report_v3 `per_arm_nrf.price_taker.q_best`). Resolved under the declared procedure: the ADMM measure is max(band 4,155, spread at C\* 71,926) = 71,926 €, against a static three-start spread of 0.0042 € |
| H3c: the four C_OPEN reverse-flow hours are among C_NRF's binding hours; 864 rows per copy active at terminal | **held** — the binding set is **exactly** those four (node 7, 2025: Autumn h19, Spring h22, Spring h2, Summer h22), by both measures (expected-value rule and per-block row bodies, tolerance 1e-3 MW). There, interface P is −5e-7 to −1e-6 MW on both copies; C_OPEN had −3.157 to −0.067 MW. Active rows: TSO 864/864; DSO 5/7/9 288 each (864); `nrf_capture_error` False |

**Run facts.**
- **Q(C_NRF) = 653,894,166.07 €**, certified (rule v6, oscillatory branch).
- k\* = 206 (k₀ 134; rule cap 243; run cap 300).
- Band [653,891,810.49, 653,895,965.71] €, width 4,155.22 €.
- Stopping ratio 1.475 × 10⁻³ (C_OPEN's is 1.536 × 10⁻³).
- 0 vetoes, 0 gap refusals, 0 non-clean cycles, 0 local solve failures.

**Decompositions** (the split terms carry no error bar of their own; only the totals are resolved):

| difference | TN conventional | DN flexibility |
|---|---|---|
| Q(C_NRF) − Q(C_OPEN) | +127,299.52 € | −106,835.64 € |
| Q(P_NRF) − Q(C_NRF) | 70,039,288.54 € (consistent with W124's TN term) | 20,836,855.98 € |

**Diagnostics** (one guarded load of the run's own `certified_models.pkl`, sha256 `b4fcc6a8…`; three negative controls
refused):
- At x = 0 every shared-storage entry has S = E = 0 and fixed variables. The circle residual is 0 (no active row). SOC
  is 0 against limits of 0. |P^E|/S^Av is undefined (S = 0).
- The closure slack is 0 in all 48 blocks.
- Interface residual ratio maxima: v 0.033, pf 0.031, ess 5e-21.
- Settlement total 406.99 € (excluded from Q); the identity closes to 2e-7 €.
- "ESSO σ slack": not applicable.

## Not confirmed

- **No error bar for the TN/DN split.** It is computable zero-solve from `recourse_blocks_all.jsonl`.
- **The between-configuration spread** at this instance was not measured.
- **C_OPEN is E.21 row 1** (W101 run, `d110bd1a`, certified at 181 and v6-decided from records), read by eval key as
  ordered. Its run predates W183c. W183c's off-build identity (696/696 blocks) and Z3 (rows 1–2 differ only in a module
  off their import path) support model equality, but the two cells did not run on the same code head.
- **The ESSO-side state** of the pickle was not read.
