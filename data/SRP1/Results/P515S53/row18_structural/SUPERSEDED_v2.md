# SUPERSEDED -- frozen spec v2

`frozen_s53_row18_structural_spec_v2_5123e67b.json`
(sha256 `5123e67bd9643ff2037977def18ce3e12aaa1cd7de6d7ead4a2bae4f9ba7675f`, committed at 6b28aadb)
is **superseded** by
`frozen_s53_row18_structural_spec_v3_1064db50.json`
(sha256 `1064db50f8d339ac8bbc9ffdcdb2485e9e06b20707dbb73748dd840fe3878ff7`, Planner task W57),
which records v2's sha256 as its `predecessor`. (`SUPERSEDED.md` in this directory is v1's note.)

v2 **was run**: it governed the zero-solve checks r5 (`../init_fix_zero_solve_checks/r5/`, committed at
7ccc2cf6). Against v2's predictions recorded before that run, A, A2, C, D, E and F passed and **two
failed**, and both outcomes stand:

* **B FAILED (vacuous).** Every per-index activation assertion held, but e = value(flow) -
  value(expectation) was 0.0 at all 2304 indices (stubbed solves load no solution), so the declared
  non-vacuity requirement failed: the minimal-split assertion tested nothing. Planner ruling (W57,
  option (a)): make it non-vacuous on an EMULATED loaded solution -- not a solve -- declared in v3.
* **C2 FAILED (spec error).** n_expression_data was +1 in both phases, v2 predicted +0.
  `row18_deviation_charge` is a scalar Expression, so +1 is correct; v3 corrects the prediction
  (Planner ruling W57, Q1). W51's r4 C2 compared only variable and active-row counts and so never
  showed the same +1.

v2 is left unmodified.
