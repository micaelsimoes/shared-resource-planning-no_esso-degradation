# P5.15 W181a: identity of Overleaf cd57efc with round 7, and Appendix E regenerated (Addendum 80, W181 items (a), (b))

**Worker, 2026-10-16.** Zero solves, zero model builds, zero pickle loads. This is the short check that Addendum 80
ruling 5 asks for, not a re-audit, so W180a's prose ratings were not re-run (the five round-7 sentences are W181b's
item (c)). The harness only read the manuscript clone, `manuscript_review/`, the frozen JSON and the specs. It ran no
fetch, pull or checkout, and it changed nothing in the manuscript.

## Blocked on Planner

None. All six predictions of (a) and (b) held. One expectation in the task text is a matter of wording:

1. **The Appendix E block differs in three hunks, not two.** The task said that only the E.4 paragraph and the E.21
   note differ. Those are the only differences outside comments, but the block also differs in a third hunk, the
   generator's header comment. The comment block above `\section{Evaluation Record}` gained "Round 7 (Addendum 80): E.4
   second phase and the E.21 note on evaluations 40 and 41." (new lines 2383–2384). This is the expert's own hunk 4 of
   `main_e5322f8_round7.diff`, which `STEP6_ROUND7_CORRECTIONS.md` and Addendum 80 list as "the generator's header
   comment". It is a LaTeX comment, so it does not appear in the compiled document. The harness records the outcome as
   "HELD for every non-comment line; FAILED as worded".

## Changed

- **`ed6490c6`** — new harness `p515_s53_w181a_identity_appendixE.py` (sha256 `28da7546…`).
  - Committed before the real run. The run records `git_head` `ed6490c613729ace8aa1846c501da8df490619de` and the same
    script sha256.
  - Sections 0–5 and 7–11 are copied verbatim from `p515_s53_w180a_appendixE_check.py` (lines 191–863 and 892–1216).
    That code is W179a's entry check plus W180a's row-16 convention (Addendum 79 ruling 1). The W180a script was not
    imported, edited or run, and its output directory was not touched.
  - New code: a unified-diff parser that reads each hunk by its header counts, a hunk-by-hunk comparison of the +/−
    lines in order (ranges and context lines reported alongside), a negative control for that comparison, and a
    classifier for the block hunks.
- **This commit** — this report and `data/SRP1/Results/P515S53/w181a_identity_appendixE/`. The launch command created
  the directory fresh with `mkdir`, and every file in it was opened `'x'` and written once.

| file | sha256 |
|---|---|
| `p515_s53_w181a_identity_appendixE.py` | `28da75460d70f20c8f49f1a2b2dd86f00950878776a7b00012ed1c94b977baec` |
| `w181a_identity_appendixE.json` | `93c5c09c3ce1c25d1809e0d29d2a37619c2e4935b2393a55312a52a7bc584c94` |
| `w181a_identity_appendixE.md` | `f3caf0c934b24bb5700b6be2125ff5ba0b0d6977a89e85b0545bdfe961e55b0e` |
| `appE_generated_w181a.tex` | `623ddbbdcaa684b4f7d70d525c5519d91466f3fd7ebff735dc6349c483d94986` |
| `verify_appE_output.txt` | `7a0c47e9eee3b365075f0b46bdc6d20e7ec0ff40eab7f68c8213695861d213a7` |
| `launch.log` (stdout + stderr) | `23b32be420966bd369516de5f5a19fcce8a1a9e652ebd43acc2d34e752333619` |
| `manifest_sha256.json` (inputs + outputs) | `20be968d77294fa813121400f405986f7dd17592b1eb6fc018e5f7537e29f0b2` |
| `w181a_bool_typing_test.json` / `.log` | `81853cb3…` / `1363c547…` |
| `manifest_post_run_sha256.json` | `3261da2729821304640da1bfceb97188813178af0ca12c39f08d81f951c6c2f8` |

Commands (repo root; canonical interpreter; run attached; both streams captured):

```
mkdir data/SRP1/Results/P515S53/w181a_identity_appendixE && \
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w181a_identity_appendixE.py \
    > data/SRP1/Results/P515S53/w181a_identity_appendixE/launch.log 2>&1          # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \
    --out data/SRP1/Results/P515S53/w181a_identity_appendixE/w181a_bool_typing_test.json \
    > data/SRP1/Results/P515S53/w181a_identity_appendixE/w181a_bool_typing_test.log 2>&1   # exit 0, VERDICT: PASS
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w181a_identity_appendixE.py --post-run   # exit 0
```

Guards:
- `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0. Counts:
  permitted_solve 0, permitted_exec 0, blocked_solve 0, blocked_exec 0.
- `pickle.load` / `pickle.loads` were blocked for the whole run; counters 0 / 0.
- Every JSON output went through `gate_result_io`. The repository-wide bool-typing test passed.
- The harness exited 0; all 15 integrity checks were True.

Two scratch trials ran before the commit, into `<scratchpad>/w181a_trial_1` and `w181a_trial_2`. Both exited 0 with
the same outcomes.

## Found

Objective convention: not applicable; this stage compares text and bytes. The entry check uses W179a's frozen
convention: Q = gross_operational_cost, settlement excluded. Instance: Overleaf clone
`manuscript/6a67305f25e8348fb71380c3` at HEAD `cd57efcb54399cb652a0ac2dcd64e5d3a78d89fc`, whose parent is
`e5322f85dced8034312c9afb97c19001bce6bfa0`. The frozen JSON is `frozen_step6_tables_v1_590088fe.json` (sha256
`590088fe…`).

### (a) Identity — expert prediction "identical and the diffs equal": **HELD**

| revision | file | sha256 | lines | byte-identical to |
|---|---|---|---|---|
| cd57efc | `main.tex` | `ff7b045edd17136fce7a37d9ef7ec16c8515ab4f8df69f3c33cc021127bbed35` | 2,713 | `manuscript_review/main_e5322f8_round7.tex`: **yes** |
| cd57efc | `response_to_reviewers_draft.tex` | `06cc24604c54955964c6000edf48de2aba34408d6a6e499e55122ac705d5cb2c` | 292 | `manuscript_review/response_to_reviewers_v8.tex`: **yes** |
| e5322f8 | `main.tex` | `b3282295…` | 2,710 | `main_48099ab_round6.tex`: yes (= W180a) |
| e5322f8 | `response_to_reviewers_draft.tex` | `34a5754b…` | 292 | `response_to_reviewers_v7.tex`: yes |

- Working-tree reads equal `git show HEAD:` for both files, and both files are clean in the clone. Its only untracked
  entry is `.DS_Store`.
- `git diff e5322f8 cd57efc` touches exactly `main.tex` and `response_to_reviewers_draft.tex`.
- **main.tex: 6 hunks in the clone diff, 6 in `main_e5322f8_round7.diff`.** In every hunk the +/− lines are equal and
  in order, and so are the hunk ranges and the context lines. Hunks: §2.3.3 at 966 (−1 +1), the `[AUTHOR]` comment at
  1304 (+2), Table 13 at 1616→1618 (−1 +1), the generator header comment at 2378→2380 (−1 +2), the E.21 note at
  2577→2580 (−1 +1), E.4 at 2618→2621 (−1 +1).
- **Letter: 2 hunks in the clone diff, 2 in `letter_v7_v8.diff`.** The +/− lines, ranges and context are equal: the
  version line at 1 (−1 +1) and R2.2 at 122 (−1 +1).
- Negative control, run in memory: one character was appended to one `+` line of each expert diff. The comparison
  flagged exactly that hunk, for both files.

### (b) Appendix E — expert prediction "bitwise, 33 PASS, 0 MISMATCH, 0 cells changed": **HELD**

- **Bitwise: HELD.**
  - The patched `gen_appE.py` (committed `fc2fad77`) ran in-process with the explicit target
    `data/SRP1/Results/P515S53/w181a_identity_appendixE/appE_generated_w181a.tex`. Its stdout was `ok 58 61`.
  - The output is 321 lines, sha256 `623ddbbd…`. It occurs exactly **once** in `main.tex` at cd57efc, at **lines
    2377–2697** (bytes 188,871–223,017), starting and ending on line boundaries.
  - Lines 2377–2697 run from the `% ====` rule above `% APPENDIX E -- EVALUATION RECORD` to the end of the appendix
    body. The block as located by content spans lines 2377–2698. The only block line outside the generated range is
    line 2698, which is empty and comes right before `\section*{Declaration…`.
  - The output equals `manuscript_review/appendixE_evaluation_record.tex` byte for byte (same sha256 `623ddbbd…`). It
    also equals the verifier's own regeneration.
  - Neither `appE.tex` nor any `manuscript_review/` file was written.
- **`verify_appE.py`: 33 PASS, 0 FAIL, no exception: HELD.** The full output is in `verify_appE_output.txt`. The
  verifier writes its own copy into a system `tempfile.mkdtemp()` directory under `/var/folders/…`, outside the
  repository.
- **Entry check at cd57efc against the frozen JSON: 1,623 entries, 0 MISMATCH: HELD.**
  - 1,443 entries are frozen fields and MATCH. 180 entries are not frozen fields; all agree with their named source and
    0 disagree. Mapping problems: 0.
  - Per table (entries / MATCH / not-frozen agrees): E.20 639 / 581 / 58; E.21 854 / 732 / 122; E.22 18 / 18 / 0;
    E.23 49 / 49 / 0; E.24 63 / 63 / 0.
  - E.21 has 61 rows numbered 1–61, mapped one-to-one onto the 61 frozen cells, and the map is the same at e5322f8.
    E.20 has 58 rows numbered 1–58.
  - Cross-check: the same checker run on e5322f8 finds 0 entries that disagree.
- **Table cells changed e5322f8 → cd57efc: 0 of 1,637 compared: HELD.**
  - No row-set changes, captions, group headers or other table lines changed.
  - One table **note** changed: the E.21 note, line 2580 → 2583.
- **Every difference in the Appendix E block, classified.** The block is lines 2375–2695 at e5322f8 and 2377–2698 at
  cd57efc. The generator's output changed from W180a's `appE_generated_w180a.tex` in the same three hunks.

| # | class | where | e5322f8 → cd57efc | change |
|---|---|---|---|---|
| 1 | **comment** | block header (generator header comment) | l. 2381 → 2383–2384 | "…the four W179 check comments closed." + " Round 7 / % (Addendum 80): E.4 second phase and the E.21 note on evaluations 40 and 41." |
| 2 | **note** (E.21) | §E.2 Evaluations, Table E.21 note | l. 2580 → 2583 | "Evaluations~1, 2, 40 and~41 continue earlier runs" → "Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are re-decided from the records of earlier runs" |
| 3 | **prose** (E.4) | §E.4 Static Arrangements: Solver Starts | l. 2621 → 2624 | "a re-solve of each distribution network at the transmission network's interface voltage with its no-reverse-flow rows switched off" → "a re-evaluation of each distribution network at the transmission network's interface voltage, with its decisions held at the first phase and its voltage, thermal and no-reverse-flow rows switched off" |

Planner's context expectation, "only the E.4 paragraph and the E.21 note differ": **held for every non-comment line.**
As worded it fails, because of hunk 1, which is a comment (see Blocked on Planner).

**Mutation test (in memory only): 13 of 13 caught.**
- Nine single-cell edits were each caught at exactly the edited table / row / column by the frozen check, and as
  exactly one changed cell by the diff:
  - E.20 row 3 Δ gross 208.3 → 208.4
  - E.21 row 1 Band 4.2 → 4.3
  - E.21 row 16 Status → "cert., osc.^s"
  - E.21 row 16 Rule → "record"
  - E.22 row 19 × 3.40 → 3.41
  - E.23 passive/cold NRF n 29 → 30
  - E.24 row 42 Resid. 0.692 → 0.693
  - E.21 row 40 Rule → "run"
  - E.20 row 34 net verdict → "w.r."
- One note edit with a frozen value (E.22 note 51.3 → 51.4) was caught by both comparisons.
- Three text edits outside the frozen entries were caught by the diff and left the frozen check unchanged: the E.23
  caption, the E.21 caption, and the E.21 note reverted to its round-6 wording.

## Not confirmed

- **Compilation.** Nothing was compiled. The expert's statement (0 errors, label numbers identical to e5322f8) was not
  re-verified here.
- **The prose of the five round-7 sentences.** This is W181b's item (c). This task did not rate §2.3.3, Table 13, E.4,
  the E.21 note or R2.2 against their sources; it only established that they are the expert's text, byte for byte.
- **Ageing citations under §3.5.** This is W181b's item (d). Not checked here.
- **The remote.** The clone's HEAD is cd57efc as present on disk. Nothing was fetched, so whether Overleaf has moved
  since was not checked.
- **The verifier's temporary file.** It was hashed (`623ddbbd…`, equal to the generator output) but not preserved; it
  lives in a system temp directory outside the repository.
