# Caveat: four W75 line citations of `p515_g_g1_g4_admm_gates.py` are stale after W76

This note concerns the W75 stringified-boolean audit, committed in `9f00377a`. Its evidence is
`stringified_bool_audit_w75.json` in this directory and the script
`p515_s53_w75_stringified_bool_audit.py`. The note is recorded going forward (Planner task W77).
The W75 commit is not amended and none of its files are modified.

## What changed

W76 edited `p515_g_g1_g4_admm_gates.py` (sha256
`172260450ba167c13c8776881a01c1b0ee096baa79cb98fa2650f83385ad458b` ->
`e147a34f1be39119196845059eaa77b581148f6f191cc3197cd9275c4b0ad5aa`). It replaced all 21
`default=str` hooks with `default=_json_default` and added a 14-line `_json_default` helper at
line 212. Every line after that helper moved down by 14.

W75 cites four `(file, line, needle)` producer sites in that module. They are listed in
`PRODUCERS` in `p515_s53_w75_stringified_bool_audit.py` (lines 69-79) and in
`C_producers_verified` in `stringified_bool_audit_w75.json`. None of the four verifies against the
module after W76:

| W75 citation | needle | after W76 |
|---|---|---|
| line 1431 | `default=str` | **gone.** The cited `default=str` text was replaced. The same `json.dump(report, ...)` call is now at line 1445 with `default=_json_default`. |
| line 3864 | `default=str` | **gone.** The cited `default=str` text was replaced. The same `json.dump(payload, ...)` call is now at line 3878 with `default=_json_default`. |
| line 3120 | `'determinate_at_gt_error_bar': (` | **moved** to line 3134. The text is unchanged. |
| line 3680 | `'determinate_at_gt_error_bar': (` | **moved** to line 3694. The text is unchanged. |

W75 also has occurrence lists that record field references in the same module: lines 2286,
2544, 3120, 3680 and 4284 for `determinate_at_gt_error_bar`, and lines 4203, 4212, 4219 and 4221.
Each of these has also moved down by 14. The text at each line is unchanged.

## The committed W75 result remains valid

The citations are historical. They describe the module as it was when W75 ran, and they all verify
against that module. The blob at `88d82d9d` and the blob at `9f00377a` are the same file, with
sha256 `172260450ba167c13c8776881a01c1b0ee096baa79cb98fa2650f83385ad458b`. All four needles were
re-checked against it in W77 and all four are present at their cited lines. No committed W75
claim changes.

Retrieve that blob with:

    git show 9f00377a:p515_g_g1_g4_admm_gates.py

(`git show 88d82d9d:p515_g_g1_g4_admm_gates.py` gives the same bytes.)

**Consequence:** running `p515_s53_w75_stringified_bool_audit.py` again against the working tree
after W76 would report these four citations as unverified. That is expected. It is not a
regression in the W75 finding. Anyone re-verifying W75 must check the citations against the blob
above, not against the current module. Per repository rule, do not re-run W75 onto its committed
output path.
