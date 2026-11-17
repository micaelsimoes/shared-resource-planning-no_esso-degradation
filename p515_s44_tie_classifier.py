"""
P5.15 Addendum 25 item 2 follow-up (c) -- the re-sort-and-straddle TIE-ORDER
classifier, factored out of `p515_s44_gate_tie_analysis.py` (commit 7ac75e9f)
into reusable functions, for any bitwise gate that compares a recourse-jump
sidecar written AFTER a sort-key fix against one written BEFORE it.

Authority: `P5_15_S44_GATE_RULING.md` (8ce0b872), "Follow-ups": "the tie
classifier for gates against pre-8682cfdd sidecars adopts the analysis's
re-sort-and-straddle test rather than a longer alias list."

`p515_s44_gate_tie_analysis.py` itself is NOT modified (it is the settling
artifact of a committed ruling); `p515_s44_followups_check.py` verifies that
`classify_top_k_list(..., 'objective_component_block_deltas')` reproduces
that script's committed per-row classification on the same committed inputs,
row for row.

Two top-k diagnostic lists in the recourse-jump sidecar
(`p515_g_g1_g4_admm_gates.s34_capture_hooks`) carry a sort key that was
changed to a TOTAL order by name:
  * `objective_component_block_deltas` -- key (-abs_delta, block_key,
    component) since 8682cfdd;
  * `block_deltas` -- key (-abs_delta, str(agent), str(node_id), year, day)
    since the Addendum 25 item 2 follow-up (a).
A sidecar written before the relevant change broke EXACT abs_delta ties by
per-process set-iteration (hash-seed) order. The test, per list (verbatim the
analysis script's `_classify_row`, parametrized by the list's key):
  (1) IDENTICAL: the two lists are equal; else
  (2) RESORT: sorting the reference list by the NEW key reproduces the new
      list exactly (every field of every entry); else
  (3) STRADDLE: with k0 = the first index whose abs_delta equals the last
      entry's abs_delta (the last tie group, which the top-k cut can split):
      resorted reference[:k0] == new[:k0] exactly; at every position >= k0
      the `tail_match_fields` are bitwise identical (objective list:
      abs_delta and block_key -- its ties are aliases within one block;
      block list: abs_delta only -- its ties are across blocks); and every
      entry present on both sides with the same identity is identical in
      every field;
  otherwise UNEXPLAINED_<reason>.
A row is explained iff every field of the row outside the classified lists is
identical and every classified list is identical / resort / straddle.

Pure functions; no model import, no solve.
"""

TIE_ORDER_FIELDS = ('objective_component_block_deltas', 'block_deltas')


def _obj_key(e):
    """The 8682cfdd key (p515_g_g1_g4_admm_gates.s34_capture_hooks), verbatim."""
    return (-e['abs_delta'], e['block_key'], e['component'])


def _block_key(e):
    """The follow-up (a) key (p515_g_g1_g4_admm_gates.s34_capture_hooks), verbatim."""
    return (-e['abs_delta'], str(e['agent']), str(e['node_id']), e['year'], e['day'])


FIELD_SPECS = {
    'objective_component_block_deltas': {
        'sort_key': _obj_key,
        'tail_match_fields': ('abs_delta', 'block_key'),
        'identity': lambda e: (e['block_key'], e['component']),
    },
    'block_deltas': {
        'sort_key': _block_key,
        'tail_match_fields': ('abs_delta',),
        'identity': lambda e: (str(e['agent']), str(e['node_id']), e['year'], e['day']),
    },
}


def classify_top_k_list(ref_list, new_list, field):
    """Classify one top-k list pair. Returns (class, detail); class is one of
    'identical', 'resort', 'straddle' or 'UNEXPLAINED_<reason>'."""
    spec = FIELD_SPECS[field]
    key, tail_fields, identity = spec['sort_key'], spec['tail_match_fields'], spec['identity']
    if ref_list == new_list:
        return 'identical', None
    if ref_list is None or new_list is None or len(ref_list) != len(new_list):
        return 'UNEXPLAINED_shape', None
    ra = sorted(ref_list, key=key)
    if ra == new_list:
        return 'resort', None
    last_ad = new_list[-1]['abs_delta']
    if ra[-1]['abs_delta'] != last_ad:
        return 'UNEXPLAINED_last_abs_delta', None
    k0 = next(i for i, e in enumerate(new_list) if e['abs_delta'] == last_ad)
    if ra[:k0] != new_list[:k0]:
        return 'UNEXPLAINED_prefix', None
    for p, q in zip(ra[k0:], new_list[k0:]):
        if any(p[f] != q[f] for f in tail_fields):
            return 'UNEXPLAINED_tail', None
    index_new = {identity(e): e for e in new_list[k0:]}
    for e in ra[k0:]:
        other = index_new.get(identity(e))
        if other is not None and other != e:
            return 'UNEXPLAINED_shared_entry_differs', None
    detail = {'k0': k0, 'straddling_tie_abs_delta': last_ad,
              'reference_members_at_cut': [list(identity(e)) for e in ra[k0:]],
              'new_members_at_cut': [list(identity(e)) for e in new_list[k0:]]}
    return 'straddle', detail


def classify_row(ref_row, new_row, fields=TIE_ORDER_FIELDS):
    """Classify one sidecar row pair over the top-k lists in `fields`.
    Returns {'class': 'identical'|'explained'|'UNEXPLAINED_...', 'per_field': {...}}.
    Every row field NOT in `fields` must be identical for the row to be explained."""
    rest_a = {k: v for k, v in ref_row.items() if k not in fields}
    rest_b = {k: v for k, v in new_row.items() if k not in fields}
    if rest_a != rest_b:
        return {'class': 'UNEXPLAINED_other_fields_differ', 'per_field': {}}
    per_field = {}
    for field in fields:
        cls, detail = classify_top_k_list(ref_row.get(field), new_row.get(field), field)
        per_field[field] = {'class': cls, 'detail': detail}
    classes = [v['class'] for v in per_field.values()]
    bad = [c for c in classes if c.startswith('UNEXPLAINED')]
    if bad:
        overall = bad[0]
    elif all(c == 'identical' for c in classes):
        overall = 'identical'
    else:
        overall = 'explained'
    return {'class': overall, 'per_field': per_field}


def _row_index_of_diff(field_path, sidecar_prefix):
    """'<prefix>[12].objective_component_block_deltas[4].component' -> (12, 'objective_component_block_deltas')."""
    if not field_path.startswith(sidecar_prefix + '['):
        return None, None
    rest = field_path[len(sidecar_prefix) + 1:]
    idx_text, _sep, tail = rest.partition(']')
    try:
        idx = int(idx_text)
    except ValueError:
        return None, None
    tail = tail.lstrip('.')
    top = tail.split('[', 1)[0].split('.', 1)[0]
    return idx, top


def reclassify_sidecar_diffs(diffs, ref_rows, new_rows, sidecar_prefix, fields=TIE_ORDER_FIELDS):
    """For gates using `p515_s40_clone_capture_preflight._diff` on a recourse-jump
    sidecar: split `diffs` (normally the gate's GENUINE bucket for that sidecar)
    into (tie_order_diffs, still_genuine, row_classes). A diff moves to
    `tie_order_diffs` ONLY IF it lies inside one of `fields` of row i AND row i
    classifies as explained by `classify_row`; everything else stays genuine."""
    row_cache = {}
    tie_order, genuine = [], []
    n = min(len(ref_rows or []), len(new_rows or []))
    for d in diffs:
        idx, top = _row_index_of_diff(str(d.get('field', '')), sidecar_prefix)
        if idx is None or top not in fields or idx >= n:
            genuine.append(d)
            continue
        if idx not in row_cache:
            row_cache[idx] = classify_row(ref_rows[idx], new_rows[idx], fields)
        if row_cache[idx]['class'] in ('identical', 'explained'):
            tie_order.append(d)
        else:
            genuine.append(d)
    row_classes = {i: {'class': c['class'],
                       'per_field': {f: v['class'] for f, v in c['per_field'].items()}}
                   for i, c in sorted(row_cache.items())}
    return tie_order, genuine, row_classes
