# P5.15 W180b: Table E.21, the rule version behind every row (Addendum 79, item (b))

**Worker, 2026-10-16.** No solves, no model builds, no pickle loads. Production code, records, specs, the frozen
JSON, `manuscript_review/` and the manuscript clone were only read.

**Manuscript.** `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `e5322f85dced8034312c9afb97c19001bce6bfa0`
(`e5322f8`).
- The harness read it with `git rev-parse HEAD` and `git show e5322f8:main.tex`. There was no fetch, pull, checkout or
  `git status` in the clone.
- Blob sha256 `b3282295ece4cc3b…` matches the declared hash. The working copy equals the blob.
- The file has 2,711 lines. Table E.21 runs from l. 2495 to l. 2581.

**Repository.** HEAD `c76f6312`, branch `feature/derivative-free-planning`.

---

## Blocked on Planner

None.

---

## Changed

- **Commit:** see the hash in the final message. This report is committed in that same commit.
- **New harness:** `p515_s53_w180b_row16_class.py` (sha256 `47d173b4…`).
- **Outputs** in `data/SRP1/Results/P515S53/w180b_row16_class/`. The directory was new and empty except for
  `launch.log`, and each file was written once.
  - `w180b_row16_class.json`: sha256 `bd6343ed7970b343311afe025309c9729d298dfb770db148895574469252b639`. It holds the
    per-row table, with the evidence path for every verdict.
  - `manifest_sha256.json`: sha256 `eb8135d4…`. It lists 81 inputs with their sha256 and git status. **All 81 are
    committed.** It also hashes the output JSON and the script.
  - `manifest_post_run_sha256.json`: sha256 `b262cc1f…`. It hashes the JSON, the manifest and `launch.log`.
  - `launch.log`: sha256 `87061bfe…`. The run was attached, with both streams captured, and exited 0. The `--post-run`
    step also exited 0.
- **Guards:**
  - `SolveProfileGuard(permitted=())` was installed before any project import and verified at exactly 0 (0 solves,
    0 execs).
  - `pickle.load` and `pickle.loads` were blocked and called 0 times.
  - No production module and no `settling_criterion*` module was imported (asserted).
  - The harness refuses to run when another copy is running or the output directory is not fresh.
  - The JSON was written through `gate_result_io` (the shared writer).
- **Nothing else changed.** `manuscript_review/`, the clone, production code, frozen files, specs, W179 files,
  TASKS.md, REVISION_CONTEXT.md and the brief are untouched.

---

## Found

### Prediction (b) against its outcome

| Prediction (expert, Addendum 79 (b)) | Outcome |
|---|---|
| 0 further E.21 rows, besides row 16, print a certificate issued under an earlier rule version without a v6 decision on its record. | **HELD.** Exactly one row is flagged, row 16 (`pb_y2025_n5`). There are 0 further rows, and all 61 rows are consistent. |

**How a row is flagged.** A row is flagged when all three of these hold:
- the printed decision was issued under a rule version below 6;
- its Status is a certificate or a superseded certificate;
- no v6 certificate exists at the printed k\*.

**Negative controls.** Nine checks in the JSON (`mutation_self_test`) all pass.
- A v2 row (17) with a planted "not decided", or with a v6 k\* moved by one cycle, is flagged.
- Row 16 restored to its `48099ab` print ("cert., osc." / "record") is flagged and inconsistent.
- A wrong Rule label on a v6-run row or a v4 row, a changed v6 cause, or a wrong band is detected.
- The unmutated rows flag exactly {16}.

### Row 16

Row 16 now prints **"superseded^s"**, Rule **"---"** and **k\* 167**, with range 0.99 and band 4.5.
- **Issuing certificate:** the W118 r2 v2 certificate (spec `fc791891`, `settling_criterion_v2.SettlingRuleV2`). Its
  `campaign_results` G15 records it as certified at 167, stopped by the settling rule.
- **v6 on its record:** W142 item 1 gives "not decided within the recorded cycles 1..167", with 1 veto.
  - The W142 floor-replay AB arm records the veto at cycle 167.
  - The window is [147, 167], and its non-clean cycle is 167.
- **Caption:** the `^s` legend ("…certified under an earlier version of the rule, whose cycle, range and band are
  shown, and not decided under the rule of …") is present verbatim.

### Per-row classification (61 rows)

| Class | Rows | n | Printed Rule | Consistent |
|---|---|---:|---|---|
| v6 run, certified (spec v6 96c23404 / ext v3 84775dc4 / A64 v1 44a2dce8) | 4, 5, 7, 10, 11, 15, 22, 23, 26–30, 32–36, 38, 43, 44, 46–49, 52–61 | 35 | run | 35/35 |
| v6 run, uncertified | 6, 8, 9, 31, 37, 39, 42, 45, 50, 51 | 10 | run | 10/10 |
| v1 certificate (W101); v6 on record certified at the same k\* (181, 172); **reference continuation** (W163 N49) | 1, 2 | 2 | record | 2/2 |
| certificate issued under v6 from records (W142 item 2, `d_c52e1670`, k\* 150) | 3 | 1 | record | 1/1 |
| v4 certificate; v6 on record certified at the same k\* (173, 173) | 12, 13 | 2 | record | 2/2 |
| v5 certificate; v6 on record certified at the same k\* (148) | 14 | 1 | record | 1/1 |
| v2 certificate (W118 r2); v6 on record certified at the same k\* | 17, 18, 19, 20, 21, 24, 25 | 7 | record | 7/7 |
| v2 uncertified (F2 references); v6 on record uncertified at the same end (281, 261) | 40, 41 | 2 | record | 2/2 |
| **v2 certificate, NOT decided by v6 on its record** | **16** | 1 | --- | 1/1 (superseded) |

- **By issuing rule version:** v6 on 46 rows, v2 on 10, v4 on 2, v5 on 1 and v1 on 2.
- **By printed Rule:** "run" on 45 rows, "record" on 15 and "---" on 1.

**What "consistent" means.** For every row the harness checked these entries:
- the identity of the row: the plan and year printed against the frozen candidate, and k\*/end against T2;
- the issuing run's `campaign_results`: its eval key against the frozen cell's, and its stage spec against the
  issuing spec;
- Status, branch, k\*/end, Rule and the `^s` marker against what the v6 decision implies;
- for certified rows, range and band against the v6 decision, at print precision.

All 61 rows pass every check.

**How "record" rows were matched to a v6 decision.**
- Every W142 item-1 record was matched to its row by instance identity: equal `candidate_key`, and the eval-dir key
  equal to the frozen `eval_key[:16]`.
- For the certified record rows the v6 Q(k\*) is also bitwise equal to a second source:
  - to the frozen Q on rows 1, 2, 12, 13 and 14;
  - to the W118 summary's Q(k\*) on rows 17–21, 24 and 25 (the frozen appended rows carry no Q);
  - to the W101 summary on rows 1 and 2.
- The v6 band width is bitwise equal to the frozen one on every decided record row.
- For x = 0 (row 1), W168 decides v6 at 181 as well, equal to W142.

### How uncertified rows and "run" rows are handled

- **Uncertified rows (6, 8, 9, 31, 37, 39, 40, 41, 42, 45, 50, 51).**
  - An uncertified row prints no certificate, so it cannot count toward the prediction.
  - Each is still classified. The harness records the rule that issued its uncertified decision, and checks that v6
    left it uncertified at the printed end. That holds on all 12: v6-run G15 `k_cap` on 10 rows, and the W142 item-1
    `k_cap` on rows 40 and 41.
  - Each printed cause is compared with the W153 cause formula, applied to the v6 decision's own fields. The formula
    is at `p515_s53_w153_step5_zero_solve_rows.py` l. 645: "gap clause" if there are gap refusals, else "lapse reset"
    if there are lapse events, else "growth test" if `swings_growing` is among the reasons. All 12 agree.
- **"run" rows (45).**
  - The issuing run is itself a run under the v6 rule. Its stage spec's `stop_rule.class` is
    `settling_criterion_v6.SettlingRuleV6`, in spec v6 `96c23404`, ext v3 `84775dc4` or A64 v1 `44a2dce8`.
  - Its `campaign_results` eval key equals the frozen cell's, and its G15 decision is the v6 decision.
  - No re-decision from records is needed for these rows, and none was looked for.

### Scope of the negative claim ("no other row")

- **What was searched:**
  - All 61 rows of E.21 in the `e5322f8` blob.
  - **v6 decisions:**
    - all 18 records of W142 item 1 (`w142_v6_from_records.json`), with the floor-replay AB arm for causes and vetoes;
    - W142 item 2 (`d_c52e1670_v6_from_records_certificate.json`);
    - W168 (`w168_x0_v6_replay_decision.json`);
    - the `campaign_results.json` G15 of every cell of spec v6 (34), ext v3 (7) and A64 v1 (4).
  - **Issuing runs:** the `campaign_results.json` of the issuing run of every row (61 files, with paths per row in the
    JSON).
  - **Reference continuations:** W163 N49, mapped to eval keys through the W101 summary.
- **What was not searched:**
  - the W118 r1 campaigns;
  - the v3 (W132) runs, and the v4/v5 runs of cells not printed in E.21;
  - the 3 × 3 evaluations, which are not in E.21;
  - pickled models, which were blocked.

---

## Not confirmed

- **The literal wording of the prediction does not fit rows 40 and 41.** The prediction says "every 'record' row has
  a v6-from-records certificate or is a reference continuation". Rows 40 and 41 are "record" rows without a
  certificate: they are uncertified under v2, and v6 on their record leaves them uncertified.
  - The prediction's main statement is about printed certificates, and it holds.
  - Rows 40 and 41 print no certificate and carry a v6 decision.
  - The E.21 note calls them continuations of earlier runs, but W163 N49 does not list them among the reference
    continuations. That set is x0, unit and C\*.
- **Row 41's cause.** On row 41 (`ref:e28de4ac`), "gap clause" rests on the W153 first-match formula only.
  - The W142 floor-replay AB arm has 25 gap refusals, but its reasons at the cap are `insufficient_turning_points` and
    `monotone_window_not_reached`, not `gap_clause`.
  - This is the same situation as row 39 (`j_5f3cccb4`, a v6 run), which T2 also labels "gap clause".
  - I did not re-derive which reading the manuscript intends.
- **The v6 decisions were taken as recorded.** They were read from W142, W168 and the G15 gates. The rule was not
  re-run on any record, and no per-cycle record was opened.
- **Rows 1 and 2 print "cert." without a branch.** v6 puts both on the oscillatory branch. This is a printing
  convention (T2's branch field is empty for the references), not an inconsistency, and I did not check why.
- **Columns other than Status, Rule and k\*/end were not rated.** Range and band were compared for certified rows only.
  k0, Gap, Slack, N.c. and Replay are W180a's scope.
