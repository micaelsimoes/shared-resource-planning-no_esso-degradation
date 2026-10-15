# P5.15 W178c — the letter v4 at Overleaf `9fd19e7` (Addendum 77, W178 item (d); zero-solve)

**Worker, 2026-10-15. ZERO SOLVES, no model, no pickle, no production import.** Read-only on the clone,
`manuscript_review/`, the frozen JSON (`590088fe`), every committed artifact and every earlier script.

**What was audited:**
- **Letter:** `response_to_reviewers_draft.tex` at clone HEAD `9fd19e7` ("Update on Overleaf."), sha256 `18992d8f…`,
  291 lines. It is byte-identical to `manuscript_review/response_to_reviewers_v4.tex`.
- **Manuscript:** `main.tex` at `9fd19e7`, sha256 `d18384d6…`, 2,511 lines. It is byte-identical to
  `manuscript_review/main_820ee1b_fixed.tex`.
- **Clone state:** tracked files clean, both files equal to their HEAD blobs. Nothing was pulled or committed there.

**Script:** `p515_s53_w178c_letter_checks.py`, commit `45e04e6c`, sha256 `48e5d8b9…`.
- It imports W177b-3′ (`p515_s53_w177b3r_checks.py`, `e8b23002…`) without editing it, and reuses its `numbering()`,
  `letter_pointers()`, `_resolve()`, `_region_text()`, `cite_order()` and its SEC / TAB / FIG targets.

**Outputs** (write-once) in `data/SRP1/Results/P515S53/w178c_letter/`:

| file | sha256 |
|---|---|
| `w178c_letter_checks.json` | `102b16c7…` |
| `w178c_letter_checks.md` (generated tables) | `c62a7ff4…` |
| `manifest_sha256.json` (36 inputs, the git-show blobs) | `ecdcf0d0…` |
| `launch.log` (stdout + stderr) | `5d84ba96…` |
| W100 typing test `w178c_bool_typing_test.json` | `a7c17590…` |
| W100 typing test `.log` | `bd25e25c…` |
| `manifest_post_run_sha256.json` | `807b4536…` |

- **W100 typing test:** PASS over 129,364 files.
- **Guards:**
  - Own `SolveProfileGuard(permitted=())` verified at exactly 0. So were the guards of the imported W177b-3′, W177b-3,
    W167 and Phase B record modules (all counts 0).
  - `pickle.load` / `loads` blocked: 0 calls.
  - Production modules imported: none.
  - The run checked its capture paths before running, and refuses to run concurrently with another copy of itself.
- **Run result:** exit 0; check failures 0; identity checks 13/13. Repository HEAD at run: `45e04e6c`.

**Objective convention (every value):** Q = certified `gross_operational_cost`, settlement excluded; net = Q −
terminal salvage credit; value = Q(0) − Q(x); F = Q + I.

**Rating (W174b):**
- **H:** a referee re-deriving from the text gets a different model, number or verdict.
- **M:** imprecise, but not a different result.
- **L:** wording.

---

## Blocked on Planner

Nothing blocks the audit. Rulings or forwards needed, by weight:

1. **Two new M rows, both from L4 (R3.7/R3.8, letter l. 274). The expert or author must choose the wording.**
   - **(a) "The revised paper reports no annual figures: every result is a present value over the horizon".** Appendix E
     as it stands contradicts this.
     - It still holds the submitted table "Operational planning results by representative year and representative day"
       (`main.tex` l. 2391). The table gives daily costs, generation and curtailment for 2025–2037, the submitted
       instance.
     - The table is introduced (l. 2386) as enabling "a more granular assessment of seasonal and inter-annual
       performance differences".
     - The claim holds for costs once the expert composes Appendix E from the export (ruling 3).
     - Even then, non-monetary per-year figures remain: Table 11 EFC/day, and §4.3 l. 1554 "+0.18, 0.20 and 0.27
       equivalent full cycles per day in 2025, 2030 and 2035". That residue is L.
   - **(b) "the representative-day weights and the block length of each representative year are stated (Section~3)".**
     - The block lengths are stated: §3 l. 1108, "each standing for a five-year block" / "three-year block".
     - The day weights are not stated anywhere. §3 says only "four seasonal representative days". §2.2.7 l. 728 defines
       D_d symbolically: "the numbers of calendar years and days represented".
     - The values in `SRP1.json` "Days" (Spring 92, Summer 91, Autumn 91, Winter 91) occur nowhere in main.tex. A
       regex scan for a standalone 91 or 92 in comment-stripped main.tex finds 0 hits.
   - **Options:**
     - **Manuscript:** add one clause to §3 giving the four day counts, and recompose Appendix E (already ordered).
     - **Letter:** trim both sentences back to "aggregated as in Eq. (13)".

   The first option costs one clause plus the pending Appendix E work. **Recommendation:** add the day counts; the
   letter then holds as written once Appendix E is rebuilt.
2. **Three M rows carried unchanged. Each is resolved by work already ordered or ruled.**
   - **LM-9: Appendix E is cited for content it does not hold yet.**
     - R2.5 l. 159 cites it for bitwise replays.
     - R3.1 l. 220 cites it for "the full claims table" with salvage.
     - Appendix E contains none of "settling slack", "bitwise", "salvage" or "claims".
     - This is the pending composition (ruling 3).
   - **LM-3: opening l. 58, "which we state as an assumption (Section~4.8)".** §4.8 l. 1733 states an inference ("is
     the search's answer under each of them, but this was not run"), and the word "assumption" does not occur in §4.8.
     The sentence is unchanged in both the letter and the manuscript.
3. **Three new L rows (letter wording):**
   - **R1.4 l. 97 (claim R14-6).** L3 adds "and the rate enters the coordination weights, and the paper says so
     (Section~4.6)".
     - This is true of the code: `shared_resources_planning.py:3872`,
       `annualization = 1.0 / ((1.0 + discount_factor) ** (year − y0))`, inside the block weight.
     - It is printed in A.1 l. 1780: w_{y,d} = Y_y D_d (1+r)^{−(y−y_0)}.
     - §4.6 does not say it. §4.6 says "exact at the plan without storage and first-order at the reference unit, whose
       ageing couples the years".
   - **R2.10 l. 192–193 (claim R210-2).** "with the converter rating entering only through the P–Q–S capability
     inequality".
     - This is true of the degradation chain (29)–(35).
     - But the rating also enters the network storage block through eq. (26) P^Ch + P^Dch ≤ S^Av, new since `e10c104`,
       and through the normalisation in (28).
     - It enters the agent through 0 ≤ P^Ch, P^Dch ≤ S^Rated,Unit (A.2 l. 1845).
   - **R1.4 l. 97 (claim R14-2).** "fuel and market prices … as operational scenarios": the Introduction (l. 380) names
     "market prices"; fuel is not mentioned.
4. **One pointer's cited sentence moved (pointer number MATCH; L).**
   - R2.3 l. 136 says "the investment being a fixed parameter of the recourse (Section~2.3.2)". The sentence "The
     investment plan is a fixed parameter of the recourse" was in §2.3.2 at `e10c104` (old l. 728).
   - The B1 rewrite removed it. §2.3.2 now says the cohort ratings "are the fixed additions … of the candidate plan"
     (l. 770).
   - The explicit statement is in §2.2.6 l. 696: "… are fixed parameters of the operational subproblem".
   - This is in substance in §2.3.2. R2.3 is not a changed reply, and this was found by the pointer needles.
5. **L1–L4 are not in `manuscript_review/w177b_fixes.tex`.** The task says they are, but the file holds A1–A20 and B1–B4
   only; no "L1" to "L4" text occurs in it. L1–L4 were checked against the letter-edit content of Addendum 77 rulings
   1, 2, 4 and 3. All four are applied (see below).

## Prediction vs outcome

| prediction (expert, Addendum 77) | outcome |
|---|---|
| (d) 0 pointer MISMATCH | **HELD.** 156 numbered pointers: **151 MATCH, 0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED**, 0 undeclared. The 16 table MISMATCHes at `e10c104` all MATCH now. Mechanism: the `\addtocounter{table}{-1}` after each of the three nomenclature longtables (l. 224, 287, 345). Without that adjustment, the same 16 pointers would read +3 (Table 1 → 4, Tables 7–15 → 10–18). |

## Changed replies

**Changed replies:** the letter diff `e10c104 → 9fd19e7` (clone git) has 10 hunks. They fall in exactly these 9 blocks:
- the opening "What we did not do", l. 58;
- R1.2, l. 78;
- R1.4, l. 97;
- R2.2, l. 125;
- R2.5–2.6, l. 152;
- R2.10, l. 194;
- R2.11–2.12, l. 210;
- R3.1, l. 220–222;
- R3.7–3.8, l. 272–275.

No hunk lies outside them. Each whole reply was audited, not only the edited sentences.

### Part 1 — numbers

The run found **67 numeric tokens** in the changed replies, with pointers and reviewer item numbers masked. Every token
carries a declaration; an undeclared token would have failed the run.

| status | count |
|---|---|
| match | 55 |
| derived, holds | 5 |
| not a quantity ("one recourse evaluation", "couples the two", the "two-phase" label, …) | 7 |
| **MISMATCH** | **0** |

Sources by reply:

- **Opening, l. 58:**
  - "Two" requests: two "We did not" sentences.
  - "five" years and "twenty-five" scenarios: submitted `main.tex` l. 664 and l. 673.
  - "3 × 3": the 3 × 3 instance file.
- **R1.2, l. 77–81:**
  - 0.909, 0.934 and 0.933: frozen `three_by_three` rows 8, 7 and 9. R predicted is 0.9331; the letter's 3 dp matches.
    W177b-1′ called this approximate under the export's 4 dp rule.
  - "five" blocks: the 3 × 3 years.
  - 9-bus / three DNs / 33-bus: node counts in the case files; 108 = 9 + 3 × 33.
  - 18.25 and 92.16: submitted l. 140.
  - 13.9: frozen `benchmark.benefit_relative`.
  - 15 = 3 × 5 (SRP1 Years).
  - Three five-year blocks (SRP1.json) and five three-year blocks (the 3 × 3 instance).
- **R1.4, l. 97:**
  - Rates 0 / 2 / 5 / 8 and −40.8 / −64.4 / −92.7 / −114.6: frozen `discount` (`value_minus_I`, k€ at 1 dp).
  - The production rate 2 % (three tokens): `SRP1.json` DiscountFactor 0.02.
- **R2.2, l. 125:**
  - 0.25 / 0.5: `p515_s45_a1_campaign.py` LATTICE_P_STEP / LATTICE_E_STEP, read as text.
  - 2 / 4 h: `SRP1_ESS_Params.json` E/P bounds.
  - Three years × four days × four operators = 48 (SRP1.json).
  - Three agents: distinct connection nodes 5, 7, 9.
- **R2.5–2.6, l. 153–171:**
  - Three turning points, one period, 0.9 τ: W163 N7, N9/N10 and N25.
  - 42 / 32 / 10: W153 certification totals.
  - Three starts: export T6, "best of 3".
- **R3.1, l. 220–221:**
  - The "two-node plan of §4.1": claim `C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5`, 93.9 → 84.0 k€.
  - 60 / one verdict: W153 `n_claims` 60, `n_verdict_differs` 1.
  - The "two-node plan": the F2 incumbent has storage at nodes 5 and 7.
  - "doubled": `flex_price_multiplier` 2.0 in the W118 r2 F2-incumbent spec, which carries the incumbent's eval key.
  - "one of its unit neighbours": W153 `claims_verdict_differs` has one entry.
  - 2035 / 2030, +42.5 / −2.3: frozen `year_ladder`, D_gross 42,458.65, D_net −2,286.25.
  - 10–90 %: `definitions.py` 0.10 / 0.90.

### Part 2 — claims

The run rated **58 claims** in the changed replies. For each, the letter fragment occurs exactly once inside its reply.
The evidence was located mechanically and is quoted in the JSON with its main.tex line numbers.

| rating | count |
|---|---|
| consistent | 43 |
| H | 0 |
| M | 5 |
| L | 10 |

Of the 58, **17 sit on lines edited since `e10c104`**: 14 consistent, M 2 (R37-3, R37-4) and L 1 (R14-6).

**The edited claims, with the main.tex text they describe:**

- **Opening l. 58, "(Section~4.8)" for 5 × 5: consistent.** LM-4 is resolved by A15. §4.8 l. 1734: "The multi-scenario
  evidence is the $3 \times 3$ instance: the $5 \times 5$ scenario set of the submitted version could not be evaluated
  within the memory available and is left to future work." Record: `REVISION_CONTEXT.md` l. 450, "The 5×5 pair does not
  run on this machine".
- **R1.2 l. 78 (L2), "the reference unit again loses determinately to the plan without storage": consistent.** LL-2 is
  resolved.
  - Record: value − I = −81,261.76 €, 4.52× its recorded resolution of 17,964.95 €.
  - §4.5 l. 1602: "The planning search was not repeated at this instance; the reference unit was evaluated against the
    plan without storage."
  - Table 13 l. 1622: "$-$81.3~k\euro & determinate".
  - l. 1603 is explicit that these are production-exit evaluations, which is what "determinately" rests on here.
- **R1.4 l. 97 (L3), "re-weight … rather than re-solving them; exact at the plan without storage and first-order at the
  unit plan, where the ageing chain couples the years": consistent.** LM-7 is resolved.
  - W153 discount row: the per-year weighted costs of the two settled 2 % evaluations reproduce the recorded Q to
    1.2 × 10⁻⁷ €.
  - §4.6 l. 1664: "The rates other than 2\,\% re-weight the per-year costs of the 2\,\% evaluations rather than
    re-solving them; this is exact at the plan without storage and first-order at the reference unit, whose ageing
    couples the years."
  - SoH chain: `shared_energy_storage_data.py:724`.
  - The size of the first-order error was not measured.
- **R1.4 l. 97, "and the rate enters the coordination weights, and the paper says so (Section~4.6)": L.** See Blocked 3.
- **R2.2 l. 125, "Eq.~(13) gives how each recourse evaluation aggregates…": consistent.** eq:recourse_value l. 709–711
  has `\frac{Y_y}{(1+r)^{y-y_0}}`, D_d and `\omega_m \, \omega_o`. Code: `shared_resources_planning.py:3873`.
- **R2.5 l. 152, "we now say so (Section~2.2.7)": consistent.** LM-8 is resolved by A4. §2.2.7 l. 733: "No convergence
  guarantee is available for consensus ADMM on a nonconvex recourse of this kind; each evaluation is therefore stopped
  by an explicit rule."
- **R2.10 l. 194, "Equations~(29)--(35) were rewritten accordingly and checked against the implementation":
  consistent.**
  - (29)–(35) are exactly the agent chain: net power, allocation, throughput, cycling loss, SoH chain, floor,
    calibration.
  - Code anchors located: throughput `shared_energy_storage_data.py:665`; cycling loss `:702–703`; floor `:732`;
    calibration note `SRP1_ESS_Params.json` l. 27.
  - The full code audit of (29)–(30) is W178a's (`4504ebf4`, prediction (a) HELD); it was not re-done here.
- **R2.11–2.12 l. 210, "Section~2.3.3, Eqs.~(20)--(28)": consistent.** Those are exactly the nine equations of §2.3.3.
- **R3.1 l. 220, "Eqs.~(38)--(39)": consistent.** LL-9 is resolved by A8.
  - eq:salvage_credit l. 1341–1343 prints the coded form: `\bar{c}^{\text{Inv,E}}_{y^\text{Inv}}\, E^{\text{Inv}}…`
    and `…SoH^{\text{Min}}_e}{1 - SoH^{\text{Min}}_e}`.
  - `SRP1_ESS_Params.json` l. 12–13: EXPECTED_INSTALLATION_ENERGY_COST, NORMALIZED_ABOVE_MINIMUM_SOH.
  - Recorded credits 11.8 / 56.5 k€ = Table 14.
- **R3.1 l. 221 (L1), "Among the 60 comparisons … no sign and one verdict": consistent.** LH-1 is resolved.
  - W153: 60 claims, 0 sign changes, 1 verdict change (`L:y2030__n5_p0.25_e0.5__n7_p1_e3`).
  - §4.6 l. 1686: "applying the salvage credit changes no sign and one verdict".
- **R3.1 l. 221, "a two-node plan evaluated under the doubled flexibility price becomes determinate in its favour
  against one of its unit neighbours": consistent.**
  - Incumbent: 2030, node 5 0.25/0.5 + node 7 1/3.5. Neighbour: node 7 1/3.
  - d = F(neighbour) − F(incumbent): +26,427 € gross (0.947×, within the uncertified bar) and +34,120 € net (1.223×,
    determinate). The positive sign means the neighbour is worse, so the change is in the incumbent's favour.
- **R3.1 l. 221, "Separately, the investment-year comparison depends on the convention": consistent.** §4.6 l. 1686:
  "The investment-year comparison above is not among the 60". T4: +42,458.65 gross (determinate, 3.46×) and −2,286.25
  net (within, 0.19×).
- **R3.1 l. 221, "The SOC limits are now stated once, in Section~3.5 … (Eq.~(22))": consistent.**
  - §3.5 l. 1266: "a usable state-of-charge window of 10--90\,\% of the available energy ($SoC^{\text{Min}} = 0.10$,
    $SoC^{\text{Max}} = 0.90$)".
  - Neither "10--90" nor "SoC^{\text{Min}} = 0.10" occurs elsewhere (l. 1–1262 and l. 1306–2511 scanned).
  - eq:soc_limits l. 870–872.
- **R3.7 l. 272, "Eq.~(13): one discount factor per representative year …": consistent** (eq. (13); code `:3872`).
- **R3.7 l. 274 (L4), "no annual figures …" and "day weights … stated (Section~3)": M and M.** See Blocked 1.

**The unedited claims in the changed replies (41):**

- **consistent (29):** among them:
  - the abstract names lithium-ion (A1; LM-5 resolved);
  - the Introduction states what is and is not in scope (A3, l. 381; LM-6 resolved);
  - the abstract says both uncertainties;
  - §2.3.3 / §2.3.4 / A.2 place each storage constraint as the letter says (R2.11–2.12, 7 claims, all consistent);
  - the throughput formula = eq. (31) = code `:665`.
- **M (3):** O-2 (LM-3), R25-4 and R31-3 (LM-9).
- **L (9):**
  - O-3 and R12-9 (LL-1: neither "marginal" nor "Shapley" is in §5);
  - R14-2 (fuel);
  - R210-2 (rating "only");
  - R22-6 (LL-4);
  - R22-7 (LL-5);
  - R25-2 (LL-6, monotone branch not mentioned);
  - R25-6 (LL-7);
  - R25-7 (LL-8, "we state this limitation explicitly" — no such statement; nearest is §2.1 l. 477).

**W177b-2′ letter rows in the changed replies, re-scored:**

| status | rows |
|---|---|
| resolved (8) | LH-1, LM-4, LM-5, LM-6, LM-7, LM-8, LL-2, LL-9 |
| unchanged (8) | LM-3, LM-9, LL-1, LL-4, LL-5, LL-6, LL-7, LL-8 |
| replaced (1) | LM-10: the rename claim is gone; the new L4 sentence is M for a different reason (Blocked 1) |

Not re-scored (their replies are unchanged): LM-1, LM-2, LL-3, LL-10.

**L1–L4 against Addendum 77:**
- **L1** (ruling 1, R3.1): applied.
  - The year pair is placed outside the 60 by "Separately".
  - Its sign change is visible in the numbers but not stated in words.
  - The multiples 0.95× / 1.22× are not quoted. That is fine: the ruling does not require them.
- **L2** (ruling 2, R1.2): applied.
- **L3** (ruling 4, R1.4): applied, plus the added clause R14-6 (L).
- **L4** (ruling 3, R3.7/3.8): applied as a replacement sentence rather than a trim, which introduces the two M rows.

## Pointers — every numbered pointer of the letter against main.tex at `9fd19e7`

**156 pointers** (W177b-3′'s extractor; at `e10c104` it found 151). The 5 extra are new or changed pointers:
- l. 58: "Section~4.8" for 5 × 5;
- l. 97: "Section~4.6";
- l. 152: "Section~2.2.7";
- R3.1's "Section~4.6" and "Eq.~(22)";
- equation ranges on edited lines.

| kind | MATCH | SUBMITTED |
|---|---|---|
| section | 88 | 1 |
| appendix | 21 | |
| table | 16 | |
| figure | 10 | 3 |
| equation | 9 | |
| reference [7]–[9] | 4 | |
| algorithm | 2 | |
| "new 2.2.7" | 1 | |
| "former 2.2.7" | | 1 |

**MISMATCH: none. AMBIGUOUS: none.**

All 9 equation pointers MATCH:

| letter l. | pointer | `9fd19e7` | was at `e10c104` |
|---|---|---|---|
| 125, 272, 274, 275 | Eq. (13) | 13 | 11 |
| 194 | (29)–(35) | 29–35 | 20–26 |
| 210 | (20)–(28) | 20–28 | labels absent at `e10c104` |
| 220, 222 | (38)–(39) | 38–39 | 29–30 |
| 221 | (22) | 22 | 16 |

Range members are consecutive.

**The 16 `e10c104` table MISMATCHes all MATCH now.** Each corresponds to a unique v4 pointer, three of them on edited
lines (l. 97, 220, 222). They are:
- Tables 7–15 (×2);
- Table 15 (×2);
- Table 12;
- Table 14 (×2);
- Table 6 (×2);
- Table 11 (×6);
- Table 1 (l. 286).

**SUBMITTED (5, unchanged; they refer to the submitted version):**
- l. 100 "Figure 2" (R1.5 heading);
- l. 102 "Figure~2" ("of the submitted version");
- l. 141 "former 2.2.7";
- l. 232 "Section~4.4" and "Figure~15" (the reviewer's).

**Content needles.** 47 pointers carry needles. Two are absent:
- **l. 136 "Section~2.3.2":** the verbatim sentence is gone, the substance remains (Blocked 4).
- **l. 160 "Appendix~E" "settling slack":** LM-9.

The l. 220 / 222 Appendix E pointers sit on edited lines and carry no pointer needle. Their content, "salvage" and
"claims", is checked as claim R31-3: both are absent.

## Static numbering against the expert's compile (ruling 5)

The static numbering is W177b-3′'s `numbering()` with each `\addtocounter{table}{n}` applied, in document order, to the
table objects after it. LaTeX counters are global, so a group does not undo the step. No other counter manipulation
exists in main.tex (`\setcounter`, `\numberwithin`, other `\addtocounter`); the script refuses if one appears.

| compile (Addendum 77 ruling 5) | static at `9fd19e7` | agree |
|---|---|---|
| Tables 1–15 restored | body tables 1–15, contiguous, 15 tables (unadjusted 4–18) | yes |
| Q(x) (13) | eq:recourse_value 13 | yes |
| cohort ratings (14)–(15) | eq:cohort_rated 14, eq:cohort_rated_energy 15 | yes |
| SOC block (20)–(28) | eq:storage_net_active_power … eq:network_complementarity 20–28 | yes |
| SOC limits (22) | eq:soc_limits 22 | yes |
| agent and degradation (29)–(35) | eq:esso_net_power … eq:cycle_life_calibration 29–35 | yes |
| salvage (38)–(39) | eq:salvage_credit 38, eq:salvage_remaining_life 39 | yes |

**7/7 agree; no disagreement to adjudicate.**

The compile also settles the one assumption the static method could not verify at `e10c104`. The three
`\addtocounter{table}{-1}` restore Tables 1–15 only if each caption-less longtable stepped the table counter; otherwise
the first body table would be numbered −2. So the compile confirms W177b-3′ assumption 3, which drove the 16 `e10c104`
MISMATCHes.

The expert's "0 undefined, 0 duplicate labels" also holds statically: 121 labels, 90 `\ref`/`\eqref` keys (181 uses),
0 undefined, 0 duplicate.

**Which to trust:** on any disagreement I would trust the compile. It resolves the package behaviour (longtable,
elsarticle appendix, amsmath) that the static method only assumes. Here the two agree, and the static numbers are
reproducible from committed inputs while the compile log is not in the repository.

Appendix tables (static): D.16–D.19 and E.20. The compile statement does not cover them, and no letter pointer uses
them.

## Unexpected findings (not acted on)

1. **§4 opening l. 1359–1361 still carries the commented `% [CONFIRM — W177] eq:salvage_credit …` block.** A8 says to
   delete it. It is invisible in the PDF; this is W178b's scope (A1–A20 as worded).
2. **R3.6 (unchanged reply) says Table 11 gives "the present-value-weighted available energy, together with the
   equivalent full cycles per day" per calibration.** The 0.50-floor row still prints "---" for both (l. 1540). This is
   W178b item (c).
3. **The task text says L1–L4 are in `w177b_fixes.tex`; they are not** (Blocked 5).

## Not confirmed

- **No compile here.** `pdflatex`, `latexmk` and `tex` are not on PATH. The static numbers rest on W177b-3′'s recorded
  assumptions, now confirmed for tables and equations by the expert's compile as quoted in Addendum 77.
- **The compile output is not in the repository.** I searched the clone and `manuscript_review/` for `.aux`, `.log`,
  `.bbl` and `main*.pdf`: none. The compile numbers are taken from the brief's text.
- **Citation numbers [7]–[9]** follow static first-citation order (elsarticle-num, unsorted), as at `e10c104`. They were
  not confirmed by BibTeX.
- **The size of the first-order error in the re-weighted discount rows (R1.4)** was not measured. Measuring it needs
  re-solves at 0 / 5 / 8 %, which are not permitted.
- **The code audit of eqs. (29)–(30) and of the §2.3 statements** is W178a's (`4504ebf4`). Here only the anchors for
  (31)–(35) were located.
- **"checked against the implementation" (R2.10)** is rated on the anchors and on W178a's result. It was not
  re-derived.
- **Unchanged replies were not audited** for numbers or claims. That covers the opening "Why", R1.5, R2.1/2.9,
  R2.3/2.4/2.7, R2.8, R3.2–R3.6 and further changes; only their pointers were checked.
- **Rendering.** Whether elsarticle prints the letter's "Appendix~A.5" as "Appendix A.5" or "A.5" in the PDF does not
  affect the number MATCH. The W178a report notes a related `Appendix~\ref` doubling in main.tex at l. 755. Not
  examined here.
