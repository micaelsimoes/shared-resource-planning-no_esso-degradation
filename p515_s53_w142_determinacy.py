"""P5.15 Addendum 61 ruling 2, Planner task W142 item 3 -- THE REPORT-STAGE DETERMINACY FLOOR (the v6 scorer).

A difference between two CERTIFIED cells is DETERMINATE iff |margin| >= max(3 x the larger of the two cells' bars,
2 TAU), a cell's bar being its band width (`settling_criterion_v6.determinate_certified`, the formula's single home).
A difference involving an UNCERTIFIED cell keeps the uncertified-form rule unchanged: BAR = 3 x max(|gap|, |slack|) over
the uncertified cell(s), in both gross and Q_cc (`p515_s53_w132_resettle_v3_campaign.resolve`, called, not copied).

`resolve_v6` replaces W132's settled-vs-settled branch (resolution = the sum of the two band widths, strict >) with the
floor; the superseded verdict is recorded beside it, report-only, so every re-scored verdict carries its predecessor.
`score_claim_v6` is W132's `score_claim` (called) with the gross and net resolutions recomputed by `resolve_v6` and the
primary verdict re-derived exactly as W132 derives it. `difference_certified_pair` scores the W118-form differences
(Phase B and the year ladder against x = 0 and between the two years) on the same rule.

Pure below the imports; zero solves (W132's launcher module arms its own zero-permit guards at import).
"""
import settling_criterion_v6 as SC6
import p515_s53_w132_resettle_v3_campaign as L132

TAU = SC6.TAU
RULE_NAME = 'certified_pair_determinacy_floor_v6'
RULE_TEXT = ('Addendum 61 ruling 2: a difference between CERTIFIED cells is determinate iff |margin| >= max(3 x the larger '
             'of the two band widths, 2 TAU); Q_cc beside, report-only, on the same threshold; a difference involving an '
             'uncertified cell keeps the uncertified form (3 x max(|gap|, |slack|), both terms) unchanged')


def resolve_v6(d_q, d_cc, views):
    """The v6 resolution of a difference over its two scorer views (`L132.view_from_report` shape)."""
    unc = [v for v in views if v.get('status') != 'certified']
    if unc:
        r = dict(L132.resolve(d_q, d_cc, views))
        r['rule_v6'] = 'uncertified form (unchanged by Addendum 61)'
        return r
    bars = [v['band'] for v in views]
    det, thr, binds = SC6.determinate_certified(d_q, *bars)
    m_cc = abs(d_cc) if (d_q > 0) == (d_cc > 0) else -abs(d_cc)
    old = L132.resolve(d_q, d_cc, views)
    return {'rule': RULE_NAME, 'rule_text': RULE_TEXT, 'bars': bars, 'larger_bar': max(bars),
            'three_x_larger_bar': SC6.DETERMINACY_BAR_FACTOR * max(bars),
            'two_tau': SC6.DETERMINACY_TAU_MULTIPLE * TAU, 'threshold': thr, 'binding_term': binds,
            'margin_Q': abs(d_q), 'margin_Qcc': m_cc,
            'verdict': 'determinate' if det else 'within resolution',
            'verdict_Qcc_report_only': 'determinate' if m_cc >= thr else 'within resolution',
            'margin_over_threshold': abs(d_q) / thr,
            'superseded_rule_report_only': {'rule': old.get('rule'), 'resolution': old.get('resolution'),
                                            'verdict': old.get('verdict'),
                                            'margin_over_resolution': old.get('margin_over_resolution')}}


def _primary(out, cl):
    if cl['net_of_salvage']:
        primary = (out.get('net_of_salvage') or {}).get('resolution')
        primary_d = (out.get('net_of_salvage') or {}).get('d_net')
    else:
        primary, primary_d = out['gross'], out.get('d_Q')
    return primary, primary_d


def score_claim_v6(cl, rv, ov):
    """One claim (W117 definition) on two views, W132's scorer with the v6 resolution. Pure."""
    out = L132.score_claim(cl, rv, ov)
    if 'd_Q' not in out:
        out['scorer'] = 'v6 (not scored: a cell has no result)'
        return out
    old_verdict = out.get('verdict')
    out['gross_superseded_report_only'] = out['gross']
    out['gross'] = resolve_v6(out['d_Q'], out['d_Qcc'], (rv, ov))
    if out.get('net_of_salvage'):
        n = out['net_of_salvage']
        n['resolution_superseded_report_only'] = n['resolution']
        n['resolution'] = resolve_v6(n['d_net'], n['d_net_cc'], (rv, ov))
    primary, primary_d = _primary(out, cl)
    out['verdict'] = (primary or {}).get('verdict') if primary else 'not scored (no net figure)'
    out['verdict_superseded_rule_report_only'] = old_verdict
    out['verdict_changed_by_v6'] = out['verdict'] != old_verdict
    out['d_primary'] = primary_d
    out['scorer'] = 'v6 (p515_s53_w142_determinacy.score_claim_v6)'
    return out


def difference_certified_pair(d_q, d_cc, view_a, view_b):
    """A W118-form difference (Phase B vs x = 0, the year ladder) on the v6 rule."""
    return resolve_v6(d_q, d_cc, (view_a, view_b))
