# SUPERSEDED -- never run

`campaign_spec_s45_a0_25a05347.json`
(sha256 `25a0534754bccf1c902a548084d214ed7cdf9227af6b98c05b5485ac866fc061`,
campaign id `s45_a0`, frozen at commit 6d8f4720 by the launcher at c1469fab)
was **superseded before any run**. No evaluation, lock or result exists for it.

Reason: concurrency 8 -> 7 and the memory preflight measure (Planner decision,
P5.15 Addendum 27 task W6). The old preflight, free + inactive >= 8 x 3 GiB,
under-counted reclaimable memory: it ignored file-backed pages, which are
reclaimable cache.

Successor:
`data/SRP1/Results/P515S45/campaign_s45_a0_c7/campaign_spec_s45_a0_c7_9d08ad2f.json`
(sha256 `9d08ad2f144b67ea97dae8dc25d91288fc86a77dd52c7f53276a52c5f00f8b34`,
campaign id `s45_a0_c7`, launcher `p515_s45_a0_campaign.py` at 8bd0102c).
The successor records this spec as `extra.predecessor_spec` (path + sha256).

The old spec is left unmodified. It cannot be run by accident, for three reasons:
- The current launcher only looks in the `campaign_s45_a0_c7` root, and its
  `--run` refuses this sha256 explicitly (exit 1).
- The old launcher text (c1469fab) passes its clean-git precondition only if the
  committed launcher is reverted.
- The old launcher's `--run` requires that this root hold only the spec. This
  file breaks that requirement.
