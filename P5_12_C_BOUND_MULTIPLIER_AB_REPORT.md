# P5.12-C — bound-multiplier A/B preflight report

**STOPPED BEFORE SOLVING: the exact preserved cycle-21 failing-block input is unavailable. Neither Arm A nor Arm B was attempted.** The required closing phrase below marks delivery of this report, not successful execution of the A/B.

## Repository and scope

- Host: `Micaels-Mac-Studio.local`.
- Repository: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`.
- Branch: `feature/derivative-free-planning`.
- Initial and final HEAD: `dd0001675e4e38cbf4ec0282df312469bcb435e5`.
- Required ancestor `72468913499f5f9eca77ddbef6ed71fd61be7a97` verified by `git merge-base --is-ancestor` (exit 0).
- Recorded upstream: `origin/feature/derivative-free-planning`; ahead 0, behind 0. No network refresh was performed.
- Checkpoint commit `dd000167` preserves P5.11/P5.12 artifacts. The difference from the required ancestor comprises documentation, diagnostic harnesses/evidence and `.gitignore`, with no production modification.
- No tracked modifications at preflight. Existing untracked files were left untouched; their status is recorded below.
- The supplied P5.12-C authorization supersedes the governing documents' P5.12-B-only stop language. No newer contradictory authorization was found in the inspected governing context. No governing document was edited.
- No model/reasoning-setting change was performed or claimed verified.

## Artifact validation and binding failure

The committed P5.12-B report, harness, aggregate forensic JSON, provenance and filtered log are present. Their SHA-256 values are recorded below. The ignored P5.11 diagnostic pickle is also present and matches its recorded hash; it represents the later cycle-25 diagnostic state and cannot substitute for the exact cycle-21 pre-solve state.

| Artifact | SHA-256 |
|---|---|
| `P5_12_B_CYCLE21_FORENSIC_REPORT.md` | `71fc4e6b69eaee27ae41d3bb2b5fcd31ae0ed64d81836f4107c6046a89b4b340` |
| `p512_b_cycle21_forensic.py` | `41602c12b55cbd8751fc927f0e015b7193eaffd7e300f504de71446784a0429d` |
| `data/SRP1/Results/P512B/p512b_cycle21_forensic.json` | `d0177f8236abe3bdd5ffdb32b48131100dc63ff00936eba3a1767e379905cf0d` |
| `data/SRP1/Results/P512B/provenance.json` | `5c21a54593d49578e5a9d185d9fda906fbf834cac216c511a8c3b2fc779cc632` |
| `data/SRP1/Results/P512B/logs/p512_b.log` | `10eecb92cabc3d8eaf66206a1fb8b96e8699c7c2e4a88c5a930377b9113f1390` |
| `data/SRP1/Results/P511/p511_selfconsistent_t0.pkl` | `3629784b277724e1f2c406f4a6dbd4599ea92b9d7ee126ac67bb2148909a67bd` |

No exact cycle-21 failing-block snapshot or corresponding snapshot-hash manifest was found in the checkout, including ignored files. The search found only the P5.7 template, the P5.11 template, and the two historical cycle-7 matched-success pickles. Therefore an exact failing-state SHA-256 cannot be reported. The aggregate JSON hash is **not** an exact input-state hash.

Inspection of `p512_b_cycle21_forensic.py` establishes the limitation:

- `_vector_summary` records aggregate statistics, a truncated fingerprint, and only four leading values for selected variable families. It does not save the full indexed primal state.
- `_param_summary` records aggregates and a truncated fingerprint, not full coordination parameters.
- `_suffix_summary` records counts, nonfinite counts, extrema and L1 norms. It does not save indexed constraint-dual or bound-multiplier values.
- `capture_block` returns this summary dictionary. `persist` writes JSON; no complete pre-solve model is serialized by this harness.
- The chained production DSO failure callback is installed only when `node_id == 7` (`shared_resources_planning.py:4589`). Chaining it does not establish preservation of every DSO block. No matching cycle-21 artifact is present.

For the recorded `DSO:case33_3|2025|Spring` failure, the JSON contains bound-suffix counts 7492 and 7320 and constraint-dual count 7826, but not the indexed values needed to initialize either arm. For example, the 792-entry `e` family retains only four leading values plus summaries. Those summaries cannot uniquely reconstruct the input.

The P5.12-B report's assertion of full preservation is thus insufficient for this experiment. This finding concerns replayable-state preservation; it does not replace the recorded failure outcome with a new numerical conclusion.

## Immediate-stop decision

Triggered rules: the preserved exact state/manifest is missing, and the required local artifact cannot be reconstructed or verified from available summaries. Replaying an ADMM trajectory is expressly prohibited by P5.12-C. No reconstruction, alternative initialization, repair, retry, or substitute template was attempted.

The stop occurred before runtime execution gates or numerical experimentation. R0 was not rerun; this report makes no fresh runtime-provenance claim. Historical P5.12-B provenance remains in its unchanged file.

## Arm equivalence and results

| Measurement | Arm A | Arm B |
|---|---|---|
| Solve attempted | No | No |
| Exact input-state hash | Unavailable | Unavailable |
| Primal/dual equivalence | Cannot establish | Cannot establish |
| Bound-input-only isolation | Not tested | Not tested |
| Solver status / iterations / objective / KKT | Not measured | Not measured |
| Runtime / process exit status | No arm process launched | No arm process launched |
| Accepted 3000-iteration failure reproduced | Not tested | Not applicable |

No new causal inference about bound-multiplier suppression is supported. The historical failure remains evidence from P5.12-B; it was not reproduced in P5.12-C.

## Commands and read-only inspection

All repository commands used the stated repository root. No numerical harness was invoked. Executed inspection commands (repeated reads are retained by description where applicable):

```sh
cat /Users/micaelsimoes/.codex/attachments/7f21ee12-c43c-4324-b50a-85cd42d2dce6/pasted-text.txt
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count HEAD...@{upstream}
git merge-base --is-ancestor 72468913499f5f9eca77ddbef6ed71fd61be7a97 HEAD
cat REVISION_CONTEXT.md
cat LOCAL_NLP_STABILITY_PLAN.md
cat P5_12_B_WORKER_PLAN.md
cat P5_12_B_CYCLE21_FORENSIC_REPORT.md
sed -n '1,440p' LOCAL_NLP_STABILITY_PLAN.md
cat p512_b_cycle21_forensic.py
rg --files --hidden -g '!\.git/**' data/SRP1/Results/P512B data/SRP1/Results/FrozenSMOPF data/SRP1/Results/P511
hostname
git log -3 --oneline
rg -n 'snapshot|pickle|FrozenSMOPF' shared_resources_planning.py network_data.py p56a_oracle.py
rg --files --hidden --no-ignore data/SRP1/Results | rg '(pkl$|pickle$|manifest|snapshot|P512B)'
cat data/SRP1/Results/P511/p511_selfconsistent_t0_meta.json
sed -n '4550,4620p' shared_resources_planning.py
rg --files --hidden --no-ignore -g '!.git/**' -g '*.pkl' -g '*.pickle' -g '*manifest*' -g '*cycle21*'
shasum -a 256 P5_12_B_CYCLE21_FORENSIC_REPORT.md p512_b_cycle21_forensic.py data/SRP1/Results/P512B/p512b_cycle21_forensic.json data/SRP1/Results/P512B/provenance.json data/SRP1/Results/P512B/logs/p512_b.log data/SRP1/Results/P511/p511_selfconsistent_t0.pkl
/usr/bin/python3 -c 'import json; p="data/SRP1/Results/P512B/p512b_cycle21_forensic.json"; d=json.load(open(p)); r=d["cycle21_failure"]; print("top-level keys:",list(d)); print("failure:",r["key"]); print("record keys:",list(r["record"])); print("multipliers:",r["record"]["multipliers"]); print("primal e:",r["record"]["primal"]["e"])'
git diff --name-only 72468913499f5f9eca77ddbef6ed71fd61be7a97 HEAD
git diff --exit-code
git diff --cached --exit-code
```

The system Python inspection printed sandbox cache-write warnings but successfully read the JSON. It did not launch IPOPT. A separate standard-library-only Python reporting command writes this new report exclusively, computes the hashes above, and records status; it imports no repository or numerical modules. Final verification reads this report's hash and Git status/diff.

## Files and final state

Only new file: `P5_12_C_BOUND_MULTIPLIER_AB_REPORT.md` (this report). No diagnostic harness or arm evidence was created. The code/documentation diff for tracked files is empty; the sole addition is this report. Existing accepted evidence and the ignored diagnostic pickle are unchanged. No production formulation, settings, source, governing document or scenario parameter was changed. No commit, push, fetch, merge, planning execution, trajectory replay or local solve was performed.

Pre-existing status immediately before report creation:

```text
## feature/derivative-free-planning...origin/feature/derivative-free-planning
?? .DS_Store
?? .env
?? __pycache__/
?? data/.DS_Store
?? data/CS1/.DS_Store
?? data/CS1/Diagrams/
?? data/CS1/Results/
?? data/CS2/
?? data/CS3/
?? data/CS4/
?? data/CS5/
?? data/CS6/
?? data/CS7/.DS_Store
?? data/CS7/Diagrams/
?? data/CS7/Results/
?? data/CS7_2/
?? data/CS7_HighFlexibility/
?? data/CS7_HighInvestment/
?? data/HR1/.DS_Store
?? data/HR1/Diagrams/
?? data/HR1/Results/
?? data/IEEE9/
?? data/OP1/.DS_Store
?? data/OP1/Results/.DS_Store
?? data/OP1/Results/OP1_operational_planning_results_centralized.xlsx
?? data/OP1/Results/OP1_operational_planning_results_hierarchical_N=1.xlsx
?? data/OP1/Results/OP1_operational_planning_results_hierarchical_N=2.xlsx
?? data/OP1/Results/OP1_operational_planning_results_no_coordination.xlsx
?? data/OP2/.DS_Store
?? data/OP2/Diagrams/
?? data/OP2/Results/
?? data/PT/
?? data/SRP1/.DS_Store
?? data/SRP1/Diagrams/
?? data/SRP1/Results/.DS_Store
?? data/SRP1/Results/20251220_1/
?? "data/SRP1/Results/20251221_3 years.zip"
?? "data/SRP1/Results/20251221_3 years/"
?? "data/SRP1/Results/20251221_5 years/"
?? "data/SRP1/Results/20251222_1 year.zip"
?? "data/SRP1/Results/20251222_1 year/"
?? "data/SRP1/Results/5 year gap, 1 market, 1 TN, 1 DNs, 3 M/"
?? "data/SRP1/Results/5 year gap, 5 market, 1 TN, 1 DNs, 3 M/"
?? data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl
?? data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl
?? data/SRP1/Results/Logs/
?? data/SRP1/Results/ProvenanceCheck/
?? "data/SRP1/Results/SRP1_operational_planning_results_distributed_without ESS.xlsx"
?? data/SRP1/Results/SRP1_operational_planning_results_no_coordination.xlsx
?? data/SRP1/Results/SRP1_planning_results.xlsx
?? data/SRP1/Results/SRP1_planning_results_5years.xlsx
?? data/SRP1/Results/case9.xlsx
?? ipopt.log
?? optim_log.txt
?? optim_log_node_5.txt
?? optim_log_node_7.txt
?? optim_log_node_9.txt
?? p57_d1_chain_studio.log

```

Final status is the same with this report added as one untracked file.

## Recommendation for planner review

Resolve the missing exact-state preservation prerequisite before authorizing execution of either arm. Provide the original complete cycle-21 snapshot and its verifiable manifest if an external copy exists. If none exists, a separate planner decision is required; no recovery experiment is designed or authorized by this report.

P5.12-C A/B COMPLETE — waiting for planner review
