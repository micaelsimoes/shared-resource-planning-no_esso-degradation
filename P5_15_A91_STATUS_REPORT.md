# P5.15 Addendum 91 status: W189a-v3 and W190a done at Overleaf `3c379fa`; W187 running (2026-11-12)

**Authority:** Addendum 91 (Planner order: W189a-v3, W187 on the author's go, W190a after the round-9c paste).

**Overleaf HEAD `3c379fa`** (`3c379fa44f37…`, 2026-11-12 13:23 UTC), round 9c pasted. The Planner checked byte
identity at 13:26 UTC, and W190a re-checked it:
- `main.tex` md5 `cd1433d7…` = `main_46c9d77_round9c.tex`;
- letter `74e88c78…` = v13;
- cover letter `63b7ddee…` = v4;
- highlights `4cedd45c…` = v3.

`52a607d` is an empty commit (same tree as `46c9d77`).

**Code.** Production head `f6c23977`, unchanged. W187 runs under spec v1.1 `25071d2f`.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Solves:**
- W187 only (running).
- W189a-v3 and W190a: zero solves under armed guards, verified at 0, with 0 pickle loads.
  - `verify_appE.py` starts `gen_appE.py` as child processes, outside the in-process guard.
  - That script imports only csv, os, re and sys, so no solve is reachable.

| part | report | commits |
|---|---|---|
| W189a-v3, supplement v3 | `P5_15_W188B_SUPPLEMENT_v3.md` | `5c4074d2`, `13076571`, `4041dad3` |
| W190a, round-9c audit | `P5_15_W190A_REPORT.md` | `ca71db16`, `9a5a78b9` |
| W187, the 15-cell screen | `P5_15_W187_REPORT.md` (when it lands) | per cell, as each lands |

## Decisions needed

1. **The supplement README's generated half still uses the `a87e195` names (expert; zero-solve; before the 11-15
   freeze).**
   - **The problem.** The prose half (v2) is correct. The per-file column descriptions come from the generator's
     `COLUMN_SPECS` and still say:
     - "Table E.20 (tab:appE_claims)" … "Table E.24", 16 occurrences;
     - "Appendix C (…)" and "Appendix D (…)" for the case-data files;
     - "not printed in Appendices C–D".

     Round 9c has an Appendix B evaluation record and no Appendix C or D. The prose above calls the same objects Table
     B.10, B.11, S4–S6 and Sections S1–S3. The Planner confirmed this in the v3 zip.
   - **Options:**
     - (a) Submit v3 as is. Cost: the supplement points to tables and appendices the paper no longer has.
     - (b) The expert supplies the old→new name mapping. The Worker rewords those strings only and regenerates as v4:
       predecessor `e6f5ce6a`, README bytes only, no CSV change, about 10 min.
   - **Recommendation: (b).**
   - **Also (wording, expert's choice).** The prose says `E20_E21_groups.csv` holds the group headings "as printed".
     The file's column is `title_latex`, LaTeX source with `$…$` and `\ref{}`, and the E.21 groups belong to Table S4,
     which the paper no longer prints.
2. **Record only, no action proposed.**
   - **Appendix B line count.** The expert's "215/215 lines" for the Appendix B gate reproduces as **218/218**. The
     `--reduced` output (sha `8e61636b`) occurs exactly once, bitwise, in `main.tex` (l. 2104–2321). The Planner
     regenerated it independently with the same result. The gate holds; only the count differs.
   - **Delay paragraph.** Cover letter v4's delay paragraph has two sentences (an apology and a reason), not the "one
     sentence" of ruling 4.

## Blocked on the author

- **Push from PyCharm.** The terminal cannot authenticate.
- **Cover letter v5.** If W187 is not reported by 11-15 12:00, the expert removes the remaining W187 bracket in
  cover letter v5.

## Changed

- **Generator and verifier.**
  - `gen_supplement.py` selects the README prose by version: prose v1 for v ≤ 2, prose v2 from v3. v1 mode and the v2
    bytes still reproduce: v2 regenerated to `864a5779` before and after the change.
  - `verify_supplement.py`: V7 and V9 are version-keyed, and V4 now expects `verify_appE.py` 40 PASS (it was 33; since
    `441722c9` the script has 40 checks). This was needed for any version to pass. The Planner read the diff and
    accepts it.
- **Supplement v3:** `manuscript_review/supplement_v3/SRP1_supplement_v3_e6f5ce6a.zip`
  (`e6f5ce6a5ef9e11c25fb4ef0ee9c273f36fff160bf735e8a56a77c91d903155d`, predecessor `864a5779`). The Planner
  re-hashed it.
- **W190a:** two harnesses (`p515_s55_w190a_numbers.py`, `p515_s55_w190a_structure.py`) that import the W189 P1/P2
  harnesses unedited. Outputs are in `w190a_audit/` with a sha256 inventory; no W189 output was touched.
- **No change to:** production, the campaign harness, hooks, launchers, specs, the clone, `gen_appE.py` or
  `verify_appE.py`.

## Found — predictions against outcomes

| prediction | outcome |
|---|---|
| W189a-v3: verifier 0 mismatches; TN-connection CSV unchanged | **held** — 1,623 entries, 0 MISMATCH; 70,851 case-file cells, 0 differences; Table 2 rebuilt 48/48; `D_ieee33_tn_connection.csv` byte-identical to v2 (node 7 ±100); v2→v3 = `README.md` + `SHA256SUMS.txt` only; deterministic (4 runs `e6f5ce6a`); README has no "version 1"; 70,851; "Table S5" |
| W190a (a): identity, clone diff = expert diffs | **held** — main 8/8, letter 2/2 and cover letter 2/2 hunks; highlights 0; no other file changed |
| W190a (b): `gen_appE.py --reduced` = Appendix B bitwise; full mode = `appendixE_evaluation_record.tex` | **held** — 218/218 lines, once in `main.tex`; full mode bitwise; 657 printed entries 0 MISMATCH |
| W190a (c): `verify_appE.py` 40 PASS | **held** — 40 PASS, 0 FAIL |
| W190a (d): numbers 0 MISMATCH; ruling 4 (i)–(iv); cover letter v4 | **held** — 0 MISMATCH, 0 UNSOURCED, 18 new tokens sourced. (i) B.4 "penalised in the agent's own objective", no "in Q". (ii) 90 % reached 8/8 days and 10 % 6/8 (Winter 13.9 %, 15.4 %). (iii) 171 € ≥ 170.69 € = 169.27 voltage + 1.42 closure. (iv) "Table S5" at both places, 0 "Section S5". Cover letter: 20.5 k€, 0.02 %, 90.9 M€, 12.2 % match the records; one bracket left (W187); EST-D-26-06323 once |
| W190a (e): letter v13 pointers 0 MISMATCH; W189's MISMATCH named | **held** — 172 MATCH, 8 SUBMITTED, 0 MISMATCH. W189's one MISMATCH was v12 l. 285 `Fig.~B.8` (target `fig:investment_lattice`, static B.9); v13 l. 285 `Fig.~B.9` MATCH. Addendum 91 ruling 5 closed |
| W190a (f): 0 `[CONFIRM`, 0 prefixes, 0 undefined, blue, numbering 9b = 9c | **held** — 102 labels; static numbering identical (126 objects); every 9b→9c line wrapped. The W189 blue rule still flags l. 1685/1695 because it counts the untypeset `\multicolumn` arguments; the wrappers are present. 17 floats, equations (1)–(41) and 6,712 main-text words (+20) unchanged in structure |
| W187 P3 (blind) | **pending** — cell 1 `x0_A_1x` launched 13:29 UTC; about 21 h serial |

**W187 execution note.** The Worker launched cell 1 through its tool's background mode, not the foreground call the
task specified.
- **What the Planner checked.** The process (PID 31351, child 31554) belongs to this session's process tree, both
  streams go to `run_x0_A_1x_launch.log`, and it holds the campaign lock. It is not `screen`, `nohup` or shell
  backgrounding.
- **Decision.** Killing it would waste a valid run, so it continues.
- **Reporting.** The deviation will be recorded in `P5_15_W187_REPORT.md`.

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX here. The expert's 0/0/0, 63 pages and printed B.9 are not
  reproduced. W190a searched `manuscript/` and `manuscript_review/` to depth 3 for round-9c `.aux`, `.log` and PDF
  files and found none.
- **Table numbers B.10/B.11.** These names in the supplement prose are inferred from a static count.
- **The remote state of Overleaf** comes from local refs, with the clone's last fetch at 13:26 UTC.
- **Table S5's content** against the l. 1353 wording is not checked.
- **The repository bool-typing test** was not run during W187; it scans `data/` while W187 is writing.
- **Supplement v1's bytes** (`a94d8193`) are reproducible only by the generator at `5d7955da`. This has been true
  since `80778c12`; v1 is superseded.
