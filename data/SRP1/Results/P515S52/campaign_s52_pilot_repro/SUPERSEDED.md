# SUPERSEDED -- never run

`campaign_spec_s52_pilot_repro_444ee0d0.json`
(sha256 `444ee0d014dbeb63b73e268931e7597ec9fb18a43d6301e30e12b660f85391a7`,
campaign id `s52_pilot_repro`, frozen at commit e52c8135 by the launcher at 298e58f0)
was **superseded before any run**. No evaluation, lock or result exists for it.

Reason (P5.15 Addendum 36/39 task W48): this spec was re-frozen together with the pilot. The Planner
dropped `persist_certified_models` from the pilot, which changed the launcher, and this spec pins the
launcher's sha256. The repro's own configuration is unchanged: x0, cap 2, concurrency 1, no
post-certification, same eval key 7d53b6f2.... Its memory requirement drops from 1 x 7 + 6 = 13 GiB to
1 x 7 + 4 = 11 GiB because the terminal-phase transient budget falls from 6 to 4 GiB.

Successor:
`data/SRP1/Results/P515S52/campaign_s52_pilot_repro_nopersist/campaign_spec_s52_pilot_repro_nopersist_2d0fccd5.json`
(sha256 `2d0fccd5414e1a33a9ee9e3e66012784e13255d492fabc9021a7cbcad8f30568`,
campaign id `s52_pilot_repro_nopersist`, launcher `p515_s52_pilot_campaign.py` at bc937cde).
The successor records this spec as `extra.predecessor_spec` (path + sha256).

The old spec is left unmodified. It cannot be run by accident, for three reasons:
- The current launcher only looks in the `campaign_s52_pilot_repro_nopersist` root, and its `--run`
  refuses this sha256 explicitly (exit 1).
- The current launcher's sha256 differs from the one this spec pins, so its script check fails.
- `--run` requires that the root hold only the spec. This file breaks that requirement.
