# SUPERSEDED -- never run

`campaign_spec_s52_pilot_b62dc2b5.json`
(sha256 `b62dc2b5cd57eacdfd267d583f09e2d61b646006df09836c26d57f71e5499523`,
campaign id `s52_pilot`, frozen at commit e52c8135 by the launcher at 298e58f0)
was **superseded before any run**. No evaluation, lock or result exists for it.

Reason (Planner ruling, P5.15 Addendum 36/39 task W48): `persist_certified_models` is dropped from
the pilot. It was the W47 Worker's addition, not in spec v22. The model pickle costs a +5.1 GiB
transient and +2.3 GiB retained per child, and it is why the memory preflight refused (19.76 GiB
available against 20 GiB required). `multiscenario_terminal.json` and the workbook already record
what the manuscript needs. If the models are ever wanted, a single evaluation can be re-run
deterministically. Without the pickle the terminal-phase transient budget is 4 GiB (was 6 GiB), so
the pilot requires 2 x 7 + 4 = 18 GiB (was 20 GiB).

Successor:
`data/SRP1/Results/P515S52/campaign_s52_pilot_nopersist/campaign_spec_s52_pilot_nopersist_08790b4d.json`
(sha256 `08790b4db71a8161ac968479f3547f7908942a910dc387e33b93018161e45f54`,
campaign id `s52_pilot_nopersist`, launcher `p515_s52_pilot_campaign.py` at bc937cde).
The successor records this spec as `extra.predecessor_spec` (path + sha256). The eval keys are
unchanged: x0 7d53b6f2..., n7_4h_e1 711fce9a....

The old spec is left unmodified. It cannot be run by accident, for three reasons:
- The current launcher only looks in the `campaign_s52_pilot_nopersist` root, and its `--run`
  refuses this sha256 explicitly (exit 1, verified).
- The current launcher's sha256 differs from the one this spec pins, so its script check fails.
- `--run` requires that the root hold only the spec. This file breaks that requirement.
