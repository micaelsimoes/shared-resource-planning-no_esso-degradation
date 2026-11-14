# W189 part 2 — round 9b structure audit at Overleaf 46c9d77 (zero solves)

HEAD `46c9d779743b40a42da73c0050d8923e6af6fb55` (parent `a87e1950fbdbb3667f37e64a43c298ba675e6cff`), 2026-11-10T18:13:45+00:00. Static numbering (no TeX).
No objective value is computed in this stage.

## 1 Identity

| file | sha256 | md5 | declared md5 | = review copy | hunks clone/expert | +/- equal | ranges | context | a87e195 = round-8 copy |
|---|---|---|---|---|---|---|---|---|---|
| main.tex | `2739a78e62a6` | `49139cc2` | 49139cc2 | True | [34, 34] | False | False | False | True |
| response_to_reviewers_draft.tex | `dbb76f81a83b` | `ed8f5b13` | ed8f5b13 | True | [10, 10] | True | True | True | True |
| highlights.tex | `13e37f6d452d` | `4cedd45c` | 4cedd45c | True | [0, 0] | True | True | True | True |
| cover_letter.tex | `2d6caf04fcd2` | `0b9e4c21` | 0b9e4c21 | True | [1, 1] | True | True | True | True |

- Figure 6 PDF: {'clone_sha256': '287bccf7b3dabc31d2984b8b430d0351ee8f9f89b9c546459e65ac3197b222b5', 'review_copy_sha256': '287bccf7b3dabc31d2984b8b430d0351ee8f9f89b9c546459e65ac3197b222b5', 'byte_identical': True, 'working_copy_equals_HEAD_blob': True, 'absent_at_a87e195': True}
- files changed a87e195 -> HEAD: ['cover_letter.tex', 'fig_soc_soh_reference_unit.pdf', 'main.tex', 'response_to_reviewers_draft.tex']
- held: **True**

## 2 Label map

- labels 136 -> 102; {'unchanged': 83, 'removed': 35, 'renumbered': 18, 'added': 1}; by class {'algorithm': {'unchanged': 2}, 'appendix': {'unchanged': 7, 'removed': 10, 'renumbered': 6}, 'equation': {'unchanged': 43, 'renumbered': 2}, 'figure': {'removed': 11, 'renumbered': 3, 'unchanged': 5, 'added': 1}, 'section': {'unchanged': 22, 'removed': 1}, 'table': {'unchanged': 4, 'removed': 13, 'renumbered': 7}}
- disagree with the stated numbering: **1**; titles {'subsec:res_certification': {'stated': 'Numerical Performance, Scope and Limitations', 'at_46c9d77': 'Numerical Performance, Scope and Limitations', 'agrees': True}}
- main-text equation max number {'a87e195': 41, '46c9d77': 41} (stated: (1)-(40) unchanged (STEP6 "Equations (1)-(40) of the main text: unchanged"; "40 equations"))
- citations: order identical True; renumbered []

| label | class | a87e195 | 46c9d77 | status | stated | agrees | caption / title at HEAD |
|---|---|---|---|---|---|---|---|
| `app:admm_updated_scenarios` | appendix | A.7 | None | removed | A.7 -> None | True | Scenario Mechanics of the Multi-Scenario Instance |
| `app:admm_updated_settings` | appendix | A.8 | A.7 | renumbered | A.8 -> A.7 | True | Solver and Coordination Settings |
| `app:adn_node5_description` | appendix | D.1 | None | removed | D.1 -> None | True | ADN Connected to TN Node~5 |
| `app:adn_node7_description` | appendix | D.2 | None | removed | D.2 -> None | True | ADN Connected to TN Node~7 |
| `app:adn_node9_description` | appendix | D.3 | None | removed | D.3 -> None | True | ADN Connected to TN Node~9 |
| `app:dns_description` | appendix | D | None | removed | D -> None | True | Distribution Networks |
| `app:market_data` | appendix | B | None | removed | B -> None | True | Market Data |
| `app:results` | appendix | E | B | renumbered | E -> B | True | Evaluation Record |
| `app:results_benchmark` | appendix | E.5 | None | removed | E.5 -> None | True | Static Arrangements: Solver Starts |
| `app:results_claims` | appendix | E.2 | B.2 | renumbered | E.2 -> B.2 | True | Comparisons |
| `app:results_evaluations` | appendix | E.3 | None | removed | E.3 -> None | True | Evaluations |
| `app:results_gap` | appendix | E.6 | None | removed | E.6 -> None | True | Evaluations Refused by the Gap Clause |
| `app:results_neighbours` | appendix | E.4 | B.3 | renumbered | E.4 -> B.3 | True | Neighbours of the Plan without Storage |
| `app:results_statistics` | appendix | E.7 | B.4 | renumbered | E.7 -> B.4 | True | Certification Statistics and Reproducibility |
| `app:salvage_credit` | appendix | E.1 | B.1 | renumbered | E.1 -> B.1 | True | Terminal Salvage Credit |
| `app:tn_description` | appendix | C | None | removed | C -> None | True | Transmission Network |
| `eq:salvage_credit` | equation | E.1 | B.1 | renumbered | E.1 -> B.1 | True |  |
| `eq:salvage_remaining_life` | equation | E.2 | B.2 | renumbered | E.2 -> B.2 | True |  |
| `fig:cs3_adn_node_5_flexibility` | figure | D.12 | None | removed | D.12 -> None | True | Flexible load profiles for the ADN at TN node~5. |
| `fig:cs3_adn_node_5_load_scenarios_2025` | figure | D.11 | None | removed | D.11 -> None | True | ADN at TN node 5: load profiles for 2025. |
| `fig:cs3_adn_node_5_res_scenarios_2025` | figure | D.10 | None | removed | D.10 -> None | True | ADN at TN node 5: RES profiles for 2025. |
| `fig:cs3_adn_node_7_flexibility` | figure | D.15 | None | removed | D.15 -> None | True | Flexible load profiles for the ADN at TN node~7. |
| `fig:cs3_adn_node_7_load_scenarios_2025` | figure | D.14 | None | removed | D.14 -> None | True | ADN at TN node 7: load profiles for 2025. |
| `fig:cs3_adn_node_7_res_scenarios_2025` | figure | D.13 | None | removed | D.13 -> None | True | ADN at TN node 7: RES profiles for 2025. |
| `fig:cs3_adn_node_9_flexibility` | figure | D.18 | None | removed | D.18 -> None | True | Flexible load profiles for the ADN at TN node~9. |
| `fig:cs3_adn_node_9_load_scenarios_2025` | figure | D.17 | None | removed | D.17 -> None | True | ADN at TN node 9: load profiles for 2025. |
| `fig:cs3_adn_node_9_res_scenarios_2025` | figure | D.16 | None | removed | D.16 -> None | True | ADN at TN node 9: RES profiles for 2025. |
| `fig:cs3_market_prices` | figure | B.8 | None | removed | B.8 -> None | True | Wholesale energy and flexibility market prices in 2025. |
| `fig:cs3_tn_res_scenarios_2025` | figure | C.9 | None | removed | C.9 -> None | True | TN RES generation profiles for the seasonal scenarios in 2025. |
| `fig:investment_lattice` | figure | E.19 | B.9 | renumbered | E.19 -> B.8 | False | Admissible single-node investment plans for installation in 2025 under |
| `fig:res_benchmark` | figure | 6 | 7 | renumbered | 6 -> 7 | True | Comparison of operating costs without shared storage: (a) costs under  |
| `fig:res_q_cycle` | figure | 7 | 8 | renumbered | 7 -> 8 | True | Numerical acceptance of two single-scenario evaluations: (a) objective |
| `subsec:res_limitations` | section | 4.8 | None | removed | 4.8 -> None | True | Limitations |
| `tab:appE_benchmark` | table | E.23 | None | removed | E.23 -> None | True | Static no-reverse-flow (NRF) arrangements at the plan without storage, |
| `tab:appE_claims` | table | E.20 | B.10 | renumbered | E.20 -> B.10 | True |  |
| `tab:appE_evaluations` | table | E.21 | None | removed | E.21 -> None | True |  |
| `tab:appE_gap` | table | E.24 | None | removed | E.24 -> None | True | Evaluations refused by the gap clause, and two others with the same or |
| `tab:appE_neighbours` | table | E.22 | B.11 | renumbered | E.22 -> B.11 | True | Neighbours of the plan without storage re-evaluated under the certific |
| `tab:cs1_ieee33_branches` | table | D.14 | None | removed | D.14 -> None | True | IEEE 33-bus test system: branch data. Branch 1 is the interface transf |
| `tab:cs1_ieee33_buses` | table | D.13 | None | removed | D.13 -> None | True | IEEE 33-bus test system: bus data. |
| `tab:cs1_ieee33_generators` | table | D.15 | None | removed | D.15 -> None | True | IEEE 33-bus test system: TN connection data. |
| `tab:cs1_ieee33_loads` | table | D.16 | None | removed | D.16 -> None | True | IEEE 33-bus test system: load data. |
| `tab:cs3_adn_node_5_res_generators` | table | D.17 | None | removed | D.17 -> None | True | ADN at TN node~5: installed RES capacity in each representative year. |
| `tab:cs3_adn_node_7_res_generators` | table | D.18 | None | removed | D.18 -> None | True | ADN at TN node~7: installed RES capacity in each representative year. |
| `tab:cs3_adn_node_9_res_generators` | table | D.19 | None | removed | D.19 -> None | True | ADN at TN node~9: installed RES capacity in each representative year. |
| `tab:cs3_tn_res_generators` | table | C.12 | None | removed | C.12 -> None | True | TN RES generator locations and installed capacities for the multi-scen |
| `tab:res_ageing` | table | 7 | 6 | renumbered | 7 -> 6 | True | Investment margins and operating indicators of the reference unit unde |
| `tab:res_benchmark` | table | 8 | 7 | renumbered | 8 -> 7 | True | Operating costs and RES curtailment without shared storage. |
| `tab:res_breakeven` | table | 5 | None | removed | 5 -> None | True | Estimated marginal break-even energy cost for 4~h storage at node~7, w |
| `tab:res_discount` | table | 11 | None | removed | 11 -> None | True | Effect of discount rate on the reference investment (node~7, 0.25~MVA  |
| `tab:res_energy_ladder_m2` | table | 6 | 5 | renumbered | 6 -> 5 | True | Investment margins and incremental gains for 4~h storage at node~7 in  |
| `tab:res_multiscenario` | table | 9 | 8 | renumbered | 9 -> 8 | True | Multi-scenario evaluation of the reference unit (node~7, 0.25~MVA / 1~ |
| `tab:res_year_ladder` | table | 10 | 9 | renumbered | 10 -> 9 | True | Installation-year and discount-rate sensitivities of the reference uni |
| `fig:res_soc_soh` | figure | None | 6 | added | None -> 6 | True | Storage operation and degradation of the reference unit (node~7, 0.25~ |

Headings (status not unchanged):

| kind | a87e195 | title | 46c9d77 | title | method | status |
|---|---|---|---|---|---|---|
| subsection | 4.7 | Certification Statistics and Computational Pe | 4.7 | Numerical Performance, Scope and Limitations | label subsec:res_certification | retitled |
| subsection | 4.8 | Limitations | None |  | not located | removed / not located |
| subsection | A.7 | Scenario Mechanics of the Multi-Scenario Inst | None |  | not located | removed / not located |
| subsection | A.8 | Solver and Coordination Settings | A.7 | Solver and Coordination Settings | label app:admm_updated_settings | renumbered |
| section | B | Market Data | None |  | not located | removed / not located |
| section | C | Transmission Network | None |  | not located | removed / not located |
| section | D | Distribution Networks | None |  | not located | removed / not located |
| subsection | D.1 | ADN Connected to TN Node~5 | None |  | not located | removed / not located |
| subsection | D.2 | ADN Connected to TN Node~7 | None |  | not located | removed / not located |
| subsection | D.3 | ADN Connected to TN Node~9 | None |  | not located | removed / not located |
| section | E | Evaluation Record | B | Evaluation Record | label app:results | renumbered |
| subsection | E.1 | Terminal Salvage Credit | B.1 | Terminal Salvage Credit | label app:salvage_credit | renumbered |
| subsection | E.2 | Comparisons | B.2 | Comparisons | label app:results_claims | renumbered |
| subsection | E.3 | Evaluations | None |  | not located | removed / not located |
| subsection | E.4 | Neighbours of the Plan without Storage | B.3 | Neighbours of the Plan without Storage | label app:results_neighbours | renumbered |
| subsection | E.5 | Static Arrangements: Solver Starts | None |  | not located | removed / not located |
| subsection | E.6 | Evaluations Refused by the Gap Clause | None |  | not located | removed / not located |
| subsection | E.7 | Certification Statistics and Reproducibility | B.4 | Certification Statistics and Reproducibility | label app:results_statistics | renumbered |

## 3 Letter v12 pointers

- v10 at a87e195 re-derived: {'counts': {'MATCH': 167, 'SUBMITTED': 5}, 'n': 172, 'equals_committed_w183h': True, 'n_committed': 172}
- v10 at HEAD: {'MATCH': 130, 'MISMATCH': 37, 'SUBMITTED': 5}
- **v12: {'MATCH': 171, 'SUBMITTED': 8, 'MISMATCH': 1}** (by extractor {'W177b-3prime letter_pointers': 177, 'W183h lettered': 3})
- stale declarations: []
- Table 7: {"table7_pointers": [{"line": 170, "verdict": "MATCH", "expected_at_head": "7", "context": "tings, about 0.12 of its determinacy threshold, with every sign and verdict unchanged (Section~4.7). The spread across independent starts is reported for the static benchmark arms (three-start bands, Table~7). The check", "in_R2_6": "R2.5 and R2.6 --- ADMM convergence and the adaptive penalty", "denotes": ["table 7 tab:res_benchmark \"Operating costs and RES curtailment without shared storage.\""]}], "benchmark_bullet_lines": [279], "benchmark_bullet_pointers": [{"line": 279, "text": "Sections~3.5 and~4.4)", "verdict": "MATCH"}, {"line": 279, "text": "Sections~3.5 and~4.4)", "verdict": "MATCH"}, {"line": 279, "text": "Section~4.4)", "verdict": "MATCH"}], "reading": "the \"benchmark pointer\" Addendum 90 names (v11 \"three-start bands, Table 6\") is the R2.6 sentence \"three-start bands, Table~7\";

| line | pointer | verdict | expected at HEAD | v10 line | note / why |
|---|---|---|---|---|---|
| 81 | Tables~4--9) | MATCH | 4--9 | 81 | range member(s) removed since a87e195 (W183h wording): ['tab:res_breakeven', 'tab:res_discount'] |
| 86 | Tables~4--9 | MATCH | 4--9 | 86 | range member(s) removed since a87e195 (W183h wording): ['tab:res_breakeven', 'tab:res_discount'] |
| 100 | Figure 2 | SUBMITTED | 2 | 100 | carried from W183h v10 (W180c SUBMITTED) |
| 102 | Figure~2 | SUBMITTED | 2 | 102 | carried from W183h v10 (W180c SUBMITTED) |
| 141 | former 2.2.7 | SUBMITTED | 2.2.7 | 141 | carried from W183h v10 (W180c SUBMITTED) |
| 152 | Appendix~A.4) | MATCH | A.4 | 152 | R2.5: "What can be said about a point the procedure accepts is now stated (Appendix~A.4)" -- STEP6 hunk 3: the |
| 170 | Section~4.7) | MATCH | 4.7 | None | R2.6: the penalty re-evaluations "(Section~4.7)" -- STEP6 hunk 15 (W185 paragraph in 4.7); and "the limitation |
| 170 | Section~4.7) | MATCH | 4.7 | None | R2.6: the penalty re-evaluations "(Section~4.7)" -- STEP6 hunk 15 (W185 paragraph in 4.7); and "the limitation |
| 170 | Table~7) | MATCH | 7 | None | R2.6: "three-start bands, Table~7" -- the static benchmark arms (Addendum 90: v11's pointer was wrong, v12 pri |
| 171 | Appendix~A.3--A.5 and A.7 | MATCH | A.3--A.5 | None | R2.6 \rchanges; v10 l. 173 "Appendix~A.3--A.5 and A.8" |
| 171 | Appendix~A.3--A.5 and A.7 | MATCH | A.7 | None | R2.6 \rchanges; v10 l. 173 "and A.8" (settings; STEP6: A.8 -> A.7) |
| 171 | Appendix~B | MATCH | B | None | R2.6 \rchanges; v10 l. 173 "Appendix~E" (the evaluation record; STEP6: E -> B) |
| 171 | Section~2.2.6 | MATCH | 2.2.6 | None | R2.6 \rchanges; v10 l. 173 "Section~2.2.6" |
| 171 | Section~4.7 | MATCH | 4.7 | None | R2.6 \rchanges; v10 l. 173 "Section~4.7; Section~4.8" (STEP6: 4.8 merged into 4.7) |
| 227 | Appendix~A.4) | MATCH | A.4 | 229 | R3.2: "an approximate first-order stationary point ... (Section~2.2.6 and Appendix~A.4)" -- STEP6 hunk 3 |
| 227 | Section~2.2.6 | MATCH | 2.2.6 | 229 | R3.2: "(Section~2.2.6 and Appendix~A.4)" -- STEP6 hunk 3: "One sentence in 2.2.7 [2.2.6 at a87e195] points to  |
| 230 | Figure~15 | SUBMITTED | 15 | 232 | carried from W183h v10 (W180c SUBMITTED) |
| 230 | Section~4.4 | SUBMITTED | 4.4 | 232 | carried from W183h v10 (W180c SUBMITTED) |
| 231 | Figure~6) | MATCH | 6 | None | R3.2: "in one figure (Figure~6): the stored energy of the reference unit ... and the state-of-health trajector |
| 232 | Appendices~A.4--A.5 | MATCH | A.4--A.5 | None | R3.2 \rchanges: the characterization (A.4) and the acceptance rule (A.5); v10 l. 235 "Appendix~A.5" |
| 232 | Section~2.2.6 | MATCH | 2.2.6 | None | R3.2 \rchanges; v10 l. 235 "Section~2.2.6" |
| 232 | Section~4.7 | MATCH | 4.7 | None | R3.2 \rchanges; v10 l. 235 "Section~4.7 (replaces the former Computational performance)" |
| 232 | Section~4 | MATCH | 4 | None | R3.2 \rchanges; v10 l. 235 "Section~4" |
| 279 | Section~4.4) | MATCH | 4.4 | 282 | benchmark bullet: the coordinated arrangement re-evaluated with the no-reverse-flow rule "(Section~4.4)" -- ST |
| 282 | Appendix~E | SUBMITTED | None | 285 | further changes: "The submitted Appendix~E held per-block operating results of the submitted instance" |
| 282 | Appendices~B--D | SUBMITTED | None | 285 | further changes: "The market, network and profile data of the submitted Appendices~B--D" |
| 283 | Appendix~D | SUBMITTED | None | 286 | further changes: "The table of the distribution networks' transmission connection in the submitted Appendix~D  |
| 285 | Fig.~B.8) | MISMATCH | B.9 | 288 |  |

- v10 pointers dropped in v12: [(171, 'Table~8)'), (172, 'Section~4.8)'), (173, 'Appendix~A.3--A.5 and A.8'), (173, 'Appendix~A.3--A.5 and A.8'), (173, 'Appendix~E'), (173, 'Section~2.2.6'), (173, 'Section~4.7'), (173, 'Section~4.8'), (235, 'Appendix~A.5'), (235, 'Section~2.2.6'), (235, 'Section~4'), (235, 'Section~4.7')]

## 4 Dangling

- undefined 0 []; duplicate labels 0; uses of removed labels typeset 0, in comments 0
- kind-word mismatches 0; literal pointers 0
- supplementary pointers {'main_distinct': ['S1', 'S1--S3', 'S2', 'S3', 'S4', 'S5', 'S6'], 'main_numbers': [1, 2, 3, 4, 5, 6], 'letter_distinct': ['S1--S6', 'S3'], 'all_within_S1_S6': True, 'note': "inventory only; the supplement content is W189a's task"}

- typed Appendix letter B-E (main: should be \ref or gone; letter: only C-E, removed at HEAD): main 0, letter 3
  - letter l. 282: "Appendix~E" — \item The appendices were reorganised. The submitted Appendix~E held per-block operating results of the submitted instance,
  - letter l. 282: "Appendices~B--" — , and the certification statistics. The market, network and profile data of the submitted Appendices~B--D, the table of every single-scenario recourse
  - letter l. 283: "Appendix~D" — \item The table of the distribution networks' transmission connection in the submitted Appendix~D printed $\pm$250~MW/MVAr for every network; the netw
- typed lettered number of a removed appendix (C.n, D.n, E.n): main 0, letter 0
- "evaluation record": main 5, letter 0
  - main l. 748: "evaluation record" — Parameter values are given in~\ref{app:results_statistics}, and the single-scenario evaluation records are provided in~\ref{app:results}.
  - main l. 1571: "evaluation record" — life floor and the 0.50 floor of the last calibration. Values are read from the certified evaluation records.}}
  - main l. 2030: "evaluation record" — hat do not meet the applicable criterion are reported as within numerical resolution. The evaluation records and comparison thresholds are provided in
  - main l. 2116: "Evaluation Record" — \section{\textcolor{blue}{Evaluation Record}}
  - main l. 2120: "evaluation record" — eported in Table~\ref{tab:res_multiscenario}. All entries are generated from the archived evaluation records. Throughout, $Q$ is the gross operational
- market / network / profile data: main 3, letter 1
  - main l. 1248: "Market Data" — \subsection{Market Data}
  - main l. 2045: "market data" — \KwIn{Plan $\boldsymbol{x}$; network and market data of every block; ESS parameters; $\rho^{V}_0$, $\rho^{\text{
  - main l. 2120: "market, network and profile data" — ts (Table~S5) and the evaluations refused by the gap clause (Table~S6), together with the market, network and profile data of the case study (Sections
  - letter l. 282: "market, network and profile data" — storage re-evaluated under the certification rule, and the certification statistics. The market, network and profile data of the submitted Appendices~
- "scenario mechanics": main 0, letter 0
- "limitations" (Section 4.8 merged): main 3, letter 4
  - main l. 364: "limitations" — s or single-level formulations; later multi-stage and bi-level formulations leave several limitations.
  - main l. 1720: "Limitations" — \subsection{\textcolor{blue}{Numerical Performance, Scope and Limitations}}
  - main l. 1728: "limitation" — : a different initialisation could in principle reach a different local solution, and the limitation below stands.
  - letter l. 161: "limitations" — the limitations section explains the mechanism behind most of them (a set-v
  - letter l. 170: "limitation" — check covers the penalty schedule and not the initialisation of the coordination, and the limitation is stated (Section~4.7).}
  - letter l. 278: "limitations" — . The five-year block discretization is stated as a modelling choice and listed among the limitations.
  - letter l. 280: "limitations" — \item The limitations section is new: set-valued interface duals with large stora
- "break-even" table / fit (Table 5 removed): main 1, letter 0
  - main l. 1348: "break-even fit" — hey are valuation probes outside the admissible set, used in the capacity ladders and the break-even fit of Subsection~\ref{subsec:res_breakeven} and 
- discount-rate table / sensitivity (Table 11 merged): main 1, letter 3
  - main l. 1681: "discount-rate sensitivities" — \caption{\textcolor{blue}{Installation-year and discount-rate sensitivities of the reference unit (node~7, 0.25~MVA / 1~MWh).}}
  - letter l. 75: "discount-rate sensitivity" — --- not about the cost of capital; the cost of capital is treated separately through the discount-rate sensitivity requested in R1.4. Operational unce
  - letter l. 88: "discount-rate sensitivity" — \subsection*{R1.4 --- Uncertainty and discount-rate sensitivity}
  - letter l. 97: "discount-rate sensitivity" — 3.3), so fuel-cost uncertainty enters only through the market-price scenarios. On (ii): a discount-rate sensitivity was added at a fixed plan: at 0, 2
- supplementary pointer S<n>: main 18, letter 2
  - main l. 1250: "S1" — test_project_repo}. Seasonal profiles are provided in the supplementary material (Section~S1). Wholesale energy prices are assumed to increase at an a
  - main l. 1257: "S2" — locations and individual capacities are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1257: "S3" — ons and individual capacities are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1292: "S3" — among loads according to the flexible-load ratios of the supplementary material (Section~S3).
  - main l. 1293: "S2" — nd flexibility profiles for the ADNs are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1293: "S3" — ibility profiles for the ADNs are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1353: "S5" — reported. The evaluation procedure and voltage-consistency checks are detailed in~Section~S5 of the supplementary material.
  - main l. 1627: "S5" — ve year--season blocks for passive operation and eight for price-taker operation (Section~S5 of the supplementary material). To separate the two effec
  - main l. 1754: "S6" — ede consensus. The affected evaluations and their diagnostic measures are listed in Table~S6; their cost comparisons are assessed using the uncertifie
  - main l. 1758: "S4" — the prescribed local-solve quality checks are excluded from certification evidence (Table~S4). Renewable-output bounds include a numerical slack of $1
  - main l. 2120: "S4" — urse evaluation the comparisons use, with its stopping status, band, gap and slack (Table~S4), the solver starts of the static arrangements (Table~S5)
  - main l. 2120: "S5" — atus, band, gap and slack (Table~S4), the solver starts of the static arrangements (Table~S5) and the evaluations refused by the gap clause (Table~S6)
  - letter l. 282: "S1--S6" — he evaluations refused by the gap clause are provided as supplementary material (Sections~S1--S6), generated from the same archived records.
  - letter l. 283: "S3" — 0~MW/MVAr in the case file that every reported run used. The supplementary table (Section~S3) is generated from the case files and gives one row per n
- "supplementary": main 7, letter 2
  - main l. 1250: "supplementary" — TEST project repository~\cite{attest_project_repo}. Seasonal profiles are provided in the supplementary material (Section~S1). Wholesale energy prices
  - main l. 1257: "supplementary" — Generator locations and individual capacities are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1292: "supplementary" — \%. Flexible demand is allocated among loads according to the flexible-load ratios of the supplementary material (Section~S3).
  - main l. 1293: "supplementary" — TN and the RES generation, load and flexibility profiles for the ADNs are provided in the supplementary material (Sections~S2 and~S3).
  - main l. 1353: "supplementary" — The evaluation procedure and voltage-consistency checks are detailed in~Section~S5 of the supplementary material.
  - main l. 1627: "supplementary" — eason blocks for passive operation and eight for price-taker operation (Section~S5 of the supplementary material). To separate the two effects, coordi
  - main l. 2120: "supplementary" — eighbours}), followed by the certification statistics and the reproducibility record. The supplementary material lists every recourse evaluation the c
  - letter l. 282: "supplementary" — of the static arrangements and the evaluations refused by the gap clause are provided as supplementary material (Sections~S1--S6), generated from the 
  - letter l. 283: "supplementary" — ode~7 is connected at $\pm$100~MW/MVAr in the case file that every reported run used. The supplementary table (Section~S3) is generated from the case 

## 5 Hygiene

- [CONFIRM: 0; tightened pattern raw 0 / typeset 0; other files {'response_to_reviewers_draft.tex': 0, 'highlights.tex': 0, 'cover_letter.tex': 0}
- general `[A-Za-z]+~\ref{app:` by word: {'in': 9, 'of': 2}
- blue: token statuses {'kept': 16925, 'new_blue': 1298, 'new': 77, 'rewritten': 2}; changed lines lacking blue 5 (with newly written text 5); by class {'prose': 1, 'table cell': 2, 'table note': 2}
- captions changed lacking blue: []

| line | + line | class | where | non-blue changed | moved | new | first words |
|---|---|---|---|---|---|---|---|
| 1250 | True | prose | prose / Market Data | 5 | 0 | 5 | in the supplementary material (Section~S1). |
| 1685 | True | table cell | tabular / Investment Year and Discount Rate | 6 | 0 | 6 | \multicolumn{4}{l}{(a) Installation year, 2\,\% discount rate} |
| 1695 | True | table cell | tabular / Investment Year and Discount Rate | 4 | 0 | 4 | \multicolumn{4}{l}{(b) Discount rate, 2025} |
| 1709 | True | table note | prose / Investment Year and Discount Rate | 44 | 26 | 18 | (a) $M I Q(\boldsymbol{x}) Q(\boldsymbol{0})$ is the cost increase relative to the plan without stor |
| 1710 | True | table note | prose / Investment Year and Discount Rate | 5 | 0 | 5 | (b) 2025; all Subsection~\ref{subsubsec:certification}; rates |

## 6 Counts

Rule: comment-stripped source (W167 strip_comments); regions: main text = \section{Introduction} .. \appendix; Appendix A = its \section .. the next appendix \section; Appendix B = its \section .. the unnumbered declaration section (or \bibliography); abstract counted apart; float environments (table, figure, longtable, algorithm, starred forms) removed and their \caption words counted apart; display math (equation, align, gather, multline, eqnarray, displaymath, subequations content) removed; inline math ($..$, $$..$$, \(..\), \[..\]) removed; heading titles removed and counted apart; arguments of \ref, \eqref, \cite*, \label, \url, \includegraphics, \begin, \end, \hspace, \vspace and the colour argument of \textcolor removed; every other control word removed with the text of its arguments kept; braces, ~ and escaped characters become spaces; a word is a whitespace-separated token holding a letter or digit (hyphenated compounds count once)

Pages: NOT estimable statically: no TeX on this machine, no .aux/.log/.pdf of the round-9b compile in the repository or the clone; the expert reports 63 pages (STEP6), not confirmed; source lines and typeset characters per region are given as proxies only, not as pages

| revision | region | prose words | heading words | caption words | floats | equations | lines |
|---|---|---|---|---|---|---|---|
| head_46c9d77 | abstract | 246 | 0 | 0 | {} | {'numbered_rows': 0, 'distinct_numbers': 0, 'first': None, 'last': None} | 11 |
| head_46c9d77 | main | 6692 | 95 | 491 | {'figure': 8, 'algorithm': 1, 'table': 9} | {'numbered_rows': 42, 'distinct_numbers': 41, 'first': '1', 'last': '41'} | 1446 |
| head_46c9d77 | appendix A | 2605 | 31 | 11 | {'algorithm': 1} | {'numbered_rows': 8, 'distinct_numbers': 8, 'first': 'A.1', 'last': 'A.8'} | 315 |
| head_46c9d77 | appendix B | 1550 | 16 | 132 | {'table': 2, 'figure': 1} | {'numbered_rows': 2, 'distinct_numbers': 2, 'first': 'B.1', 'last': 'B.2'} | 206 |
- head_46c9d77: main-text tables + figures 17 (algorithms 1); appendices ['A', 'B']; target <= 10 met: False
| a87e195 | abstract | 246 | 0 | 0 | {} | {'numbered_rows': 0, 'distinct_numbers': 0, 'first': None, 'last': None} | 11 |
| a87e195 | main | 6130 | 96 | 471 | {'figure': 7, 'algorithm': 1, 'table': 11} | {'numbered_rows': 42, 'distinct_numbers': 41, 'first': '1', 'last': '41'} | 1464 |
| a87e195 | appendix A | 2497 | 37 | 11 | {'algorithm': 1} | {'numbered_rows': 8, 'distinct_numbers': 8, 'first': 'A.1', 'last': 'A.8'} | 301 |
| a87e195 | appendix B | 49 | 2 | 8 | {'figure': 1} | {'numbered_rows': 0, 'distinct_numbers': 0, 'first': None, 'last': None} | 28 |
| a87e195 | appendix C | 37 | 2 | 22 | {'table': 1, 'figure': 1} | {'numbered_rows': 0, 'distinct_numbers': 0, 'first': None, 'last': None} | 47 |
| a87e195 | appendix D | 194 | 20 | 173 | {'table': 7, 'figure': 9} | {'numbered_rows': 0, 'distinct_numbers': 0, 'first': None, 'last': None} | 373 |
| a87e195 | appendix E | 1898 | 27 | 369 | {'table': 5, 'figure': 1} | {'numbered_rows': 2, 'distinct_numbers': 2, 'first': 'E.1', 'last': 'E.2'} | 379 |
- a87e195: main-text tables + figures 18 (algorithms 1); appendices ['A', 'B', 'C', 'D', 'E']; target <= 10 met: False

Negative controls: [('1 main.tex: expert diff + line altered', True), ('1 response_to_reviewers_draft.tex: expert diff + line altere', True), ('1 cover_letter.tex: expert diff + line altered', True), ('l1 an injected table shifts tab:res_energy_ladder_m2 5 -> 6 ', True), ('p1 R2.6 "Table~7" -> "Table~8" (declared target kept)', True), ('p2 a paired MATCH "Section~4.7" -> "Section~4.8"', True), ('p3 a declared MATCH "Appendix~A.4)" -> number A.5 (declared ', True), ('d1 an injected \\ref{tab:res_discount} is undefined', True), ('d2 a duplicated \\label{tab:res_ageing} is reported', True), ('h1 a planted "[CONFIRM" is counted', True), ('h2 a planted "Appendix~\\ref{app:" is matched', True), ('h3 the caption of fig:res_soc_soh unwrapped in a copy is det', True), ('n1 a 7-word sentence with inline math adds exactly 7 prose w', True), ('n2 an added numbered equation adds one numbered row to the m', True)]

Guards ok True (58 guards); check failures []
