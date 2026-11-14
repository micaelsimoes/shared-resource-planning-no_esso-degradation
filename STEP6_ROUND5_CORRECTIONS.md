# Step 6, round 5 — after W176a/b (the round-4 lines; number check at `8b76423`) (expert, 2026-10-08)

Rulings: Addendum 74. Lines refer to the Overleaf clone at `8b76423`. Four one-clause edits in `main.tex`, to apply
together with the §4 paste of Addendum 73; and the full list of literal "Section 4" references to map onto the §4
labels (eight, not the five of Addendum 73).

1. **§2.1, l. 445 (R4-1).** "one was stopped for review when its completion set exceeded the cap; each is described
   by its recorded certificate." → "one was stopped for review when its completion set exceeded the cap; the first two
   are described by their recorded certificates, the third by its stop record and by the certificate of the variant-B
   run that continued from its incumbent."

2. **§2.2.7, l. 639 (R4-2).** "accepted its poll points by the coarser resolution rule stated there" → "accepted its
   poll points by the resolution rule stated there".

3. **§3.5, l. 1062 ("ATB" has no source).** "(the NREL ATB utility-scale battery category~\cite{nrel_ess_costs})" →
   "(the NREL utility-scale battery category~\cite{nrel_ess_costs})".

4. **§3.6, l. 1120 (the consistency pass as run).** "each DN is then re-evaluated at the TN's interface voltage and,
   where a DN limit is violated, re-solved once at that voltage before the TSO dispatches again." → "each DN is then
   re-solved once at the TN's interface voltage before the TSO dispatches again."

**Literal "Section 4" references to map onto the §4 labels** (all eight; W176a (c)):

| line | now | becomes |
|---|---|---|
| 377 | "(Section~4)" | "(Section~\ref{sec:results})" |
| 415 | "the sensitivity study of Section~4" | "the sensitivity study of Section~\ref{sec:results}" |
| 445 | "reported in Section~4.7" | "reported in Section~\ref{sec:case_settings}" |
| 631 | "the ratio of Section~4.5" | "the ratio of Subsection~\ref{subsec:res_multiscenario}" |
| 635 | "(Section~4.7)" | "(Subsection~\ref{subsec:res_certification})" |
| 671 | "a sensitivity in Section~4" | "a sensitivity in Subsection~\ref{subsec:res_degradation}" |
| 1066 | "the sensitivities of Section~4.3" | "the sensitivities of Subsection~\ref{subsec:res_degradation}" |
| 1119 | "(Section~4.4)" | "(Subsection~\ref{subsec:res_coordination})" |

**Letter, at its final pass (author):** l. 174 (R2.8) in-text placeholder — the two references; the `\rchanges{}`
bracket placeholders such as l. 168 "Section~[limitations]; Table~[T2]" become the §4 labels
(`subsec:res_limitations`; the Appendix E table once it exists).
