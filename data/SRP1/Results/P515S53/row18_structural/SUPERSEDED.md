# SUPERSEDED -- frozen spec v1

`frozen_s53_row18_structural_spec_v1_05934ab0.json`
(sha256 `05934ab0803a6744a5f22106865baafa27eb46f76d4240adbbc53244aac7be04`, committed at d573e145)
is **superseded** by
`frozen_s53_row18_structural_spec_v2_5123e67b.json`
(sha256 `5123e67bd9643ff2037977def18ce3e12aaa1cd7de6d7ead4a2bae4f9ba7675f`, Planner task W56),
which records v1's sha256 as its `predecessor`.

v1 **was run** (unlike an unrun superseded spec): it governed the .nl probe r1
(`nl_probe/r1/`, committed at bf36d414). Against v1's predictions recorded before that run:
P0, P1 (A == B byte-identical) and P2 **held**; **P3 FAILED** -- arm D has the same 36232
columns as A and +192 rows, not fewer columns. That outcome stands; it is not erased by v2.

Why P3 was wrong: v1 assumed the NL writer's linear presolve (CONFIG default True) would
substitute away one Var per active 2-variable row. On production's path
(`SolverFactory('ipopt').solve` -> converter -> `Block.write` -> `NLWriter.__call__`),
`pyomo/repn/plugins/nl_writer.py` `NLWriter.__call__` forces `config.linear_presolve = False`
(Pyomo 6.9.5), so the hazard shows as added rows, not substituted columns. v2 restates P3
accordingly (as an explanation of r1, not a pre-run prediction), corrects v1's hazard and
writer-flag statements, and records the source citation.

v1 is left unmodified. The r1 probe script (`p515_s53_row18_structural_nl_probe.py`) still pins
v1 and is not re-run; any re-run must use a new label (r2).
