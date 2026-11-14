## Amendment 2 to Addendum 91 — supplement v5 wording; W187 execution note; round-10 confirmation (expert, 2026-11-12, on the handback at `688ab470`)

**Held.** Supplement v4 `d175ea0e` (0 mismatches; Table 2 rebuilt 48/48; no old names; every README name present in
`main.tex` at `3c379fa`); W190a clean (byte-identical files, Appendix B generated, 40/40, 0/0 mismatches, letter v13
172/8/0, structure clean). The interpretations of the mapping are accepted.

**Ruling 1 — v5, one-minute regeneration, no data file changed.** Replacement wording for the strings outside the mapping:
- (a) "(Appendix E) printed '---'" → "(evaluation-record files, Tables B.10, B.11 and S4–S6) printed '---'";
  "Appendix E amounts" → "Evaluation-record amounts"; "Appendix B-D values" → "Case-data values (Sections S1–S3)".
- (b) Every parenthesised LaTeX label in a description (e.g. "(fig:cs3_market_prices)") is removed, including the
  three whose labels still exist: a reader of the archive has no use for them, and the README table already maps each
  file to its manuscript pointer.
- (c) "not printed in Sections S2–S3" → "the single-scenario years of the manuscript's Table 2; the remaining columns
  are the multi-scenario years"; "printed mark b of Table S5" → "the over-budget mark b of Table S5"; "Table S6 itself
  prints no mark" → "Table S6 carries the mark in its over_budget column".
- (d) The `table` column definitions keep "E.20 or E.21" and "E.20, E.22 or E.23": they describe the values stored in
  the CSVs, as the Planner says.
Predecessor `d175ea0e`; verifier 0 mismatches; v5 is the submitted file. If anything in (a)–(c) does not match a
generated string exactly, apply the nearest wording and record it; do not hold the upload for it.

**Ruling 2 — W187 execution.** Cell 1 was started in the tool's background mode rather than a foreground call. It is
inside the session's process tree, both streams captured, the campaign lock held: this is not a detached run in the
sense of the `CLAUDE.md` rule, and the cell is not restarted (a restart would cost the cell and gain nothing). The
deviation is recorded in the W187 report with the reason, and cells 2–15 run in the foreground if the tool allows a
1.4 h call; if it does not, the same background mode is used and the report says so per cell.

**Round 10 (confirmed).** Expert delivery 11-14: §4.3 sentence on W187 if `P5_15_W187_REPORT.md` is in by 11-15 12:00,
else cover letter v5 with the W187 bracket removed; letter v14; the marked copy. Then W190.
