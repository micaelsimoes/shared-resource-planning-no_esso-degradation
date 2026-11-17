# SUPERSEDED -- never run

`campaign_spec_s53_f2_certificate_3b9d6df9.json`
(sha256 `3b9d6df9b14013744d89702fee1917cdb8febf24901be3bd7a533d64de142725`,
campaign id `s53_f2_certificate`. Frozen at git HEAD d00fda67 and committed in a9277128 by
`p515_s53_f2_certificate.py`, launcher sha256 `bcdbad66...`.)
was **superseded before any run**. No evaluation, lock or result exists for it.

Reason: the Planner's three rulings on the W49 report (P5.15 Addendum 40 ruling 4, task W50).
1. **Snap tie-break.** "Prefer the incumbent's investment year" is now a tie-break level placed before
   "lower I(x)". The order is: l1 to the rounded point, then l_inf, then l1 to the incumbent, then same
   year as the incumbent, then lower I(x), then label. Under the frozen order, directions 0, 5 and 12
   all snapped to the same 2035 point, and 6 of the 11 poll points were 2035 plans. That drift came
   from the tie-break, not from the geometry.
2. **Cached box neighbours.** Every feasible box neighbour of the incumbent that is in the cache but
   outside the poll set is recorded with its F, its difference from the incumbent and the resolution.
   Each is classified better, worse or INDETERMINATE (symmetric: |dF| <= resolution means
   indeterminate), at zero evaluation cost. A cached box neighbour that is determinately better
   refuses the freeze and the run.
3. **Scope.** The certificate records its own scope:
   - the incumbent is not interior;
   - the rank of the poll displacement set and of the full box, and whether each positively spans;
   - the claim is "poll failure over the recorded poll set at unit mesh", not a positive-spanning-set
     certificate;
   - the count and identity of the box neighbours outside the poll set.

Successor:
`data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1/campaign_spec_s53_f2_certificate_r1_803571c0.json`
(sha256 `803571c0efcf828eb9716d6c6255909da91d882a57a7bd4e053805f96f88027f`,
campaign id `s53_f2_certificate_r1`, launcher `p515_s53_f2_certificate.py` at a5ccc7fb, sha256
`59f75ce46b42f41053478f8ea50e495555259ef6b92b91766aeda1e013d21df6`; freeze log
`data/SRP1/Results/P515S53/campaign_s53_f2_certificate_r1_freeze_launch.log`).
The successor records this spec as `extra.predecessor_spec` (path + sha256 + reason).

The old spec and its freeze log (`campaign_s53_f2_certificate_freeze_launch.log`) are left unmodified.
The old spec cannot be run by accident, for three reasons:
- The current launcher's `--run` refuses this sha256 and this campaign id explicitly (exit 1,
  verified), and `--freeze` refuses the id.
- The current launcher's sha256 differs from the one this spec pins, so its script check fails.
- `--run` requires that the root hold only the spec. This file breaks that requirement.
