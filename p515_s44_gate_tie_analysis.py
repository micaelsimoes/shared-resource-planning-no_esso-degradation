"""
P5.15 Addendum 25 item 1 -- post-gate, ZERO-SOLVE analysis of the 43 recourse-jump
sidecar differences that the committed gate classifier left unclassified
(campaign s44_gate, `gate_results.json`: GATE FAIL, genuine = 43, all in
`recourse_jump_sidecar_baseline.jsonl[*].objective_component_block_deltas`).

This script does NOT change the gate's result or its classifier: `gate_results.json`
stays as written (FAIL). It tests ONE stated hypothesis and reports it, for the
Planner to rule on:

  H: every difference between D's committed recourse-jump sidecar and the c_star
     evaluation's sidecar is explained by the sort-key change of 8682cfdd --
     D's run broke EXACT abs_delta ties by per-process set-iteration order
     (hash seed), this run breaks them by (-abs_delta, block_key, component) --
     including tie groups the committed classifier does not list (it knows only
     {economic_market_cost, generation_cost}), and including rows where a tie
     group straddles the top-10 cut, so D's top-10 holds a different member of
     the same tie group.

Test, per row (139 rows, both sidecars read-only):
  (0) every row field other than `objective_component_block_deltas` identical;
  (1) IDENTICAL: the two top-10 lists are equal; else
  (2) RESORT: sorting D's top-10 by the 8682cfdd key reproduces this run's list
      exactly (every field of every entry); else
  (3) STRADDLE: with k0 = the first index whose abs_delta equals the 10th
      entry's abs_delta (the last tie group, which the cut can split):
      resorted D[:k0] == this run[:k0] exactly; for every position >= k0 the
      abs_delta values are identical (bitwise) and the block_key is identical;
      and every entry present on both sides with the same (block_key,
      component) is identical in every field;
  otherwise the row is UNEXPLAINED (H fails for it).
Also reported: the distinct tie groups (component-name sets with bitwise-equal
abs_delta within one block) and whether their members' previous/current are
identical (aliases) or only their delta is.

Armed `SolveProfileGuard(permitted=())`, verified 0. Write-once output:
data/SRP1/Results/P515S44/gate_tie_order_analysis/tie_order_analysis.json
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 gate tie-order analysis (zero solves)').install()

D_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run', 'recourse_jump_sidecar_baseline.jsonl')
C_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'evals',
                      '578636daa6d6360d_c_star', 'recourse_jump_sidecar_baseline.jsonl')
GATE_RESULTS = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'gate_results.json')
OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'gate_tie_order_analysis')
FIELD = 'objective_component_block_deltas'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        h.update(handle.read())
    return h.hexdigest()


def _key(e):
    """The 8682cfdd sort key (p515_g_g1_g4_admm_gates.s34_capture_hooks), verbatim."""
    return (-e['abs_delta'], e['block_key'], e['component'])


def _load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tie_groups(entries):
    groups = {}
    for e in entries:
        groups.setdefault((e['block_key'], e['abs_delta']), []).append(e)
    out = []
    for (block, _ad), members in groups.items():
        if len(members) > 1:
            prevs = {m['previous'] for m in members}
            curs = {m['current'] for m in members}
            out.append({'block_key': block, 'components': sorted(m['component'] for m in members),
                        'values_identical': len(prevs) == 1 and len(curs) == 1})
    return out


def _classify_row(a_row, b_row):
    rest_a = {k: v for k, v in a_row.items() if k != FIELD}
    rest_b = {k: v for k, v in b_row.items() if k != FIELD}
    if rest_a != rest_b:
        return 'UNEXPLAINED_other_fields_differ', None
    la, lb = a_row.get(FIELD), b_row.get(FIELD)
    if la == lb:
        return 'identical', None
    if la is None or lb is None or len(la) != len(lb):
        return 'UNEXPLAINED_shape', None
    ra = sorted(la, key=_key)
    if ra == lb:
        return 'resort', None
    last_ad = lb[-1]['abs_delta']
    if ra[-1]['abs_delta'] != last_ad:
        return 'UNEXPLAINED_last_abs_delta', None
    k0 = next(i for i, e in enumerate(lb) if e['abs_delta'] == last_ad)
    if ra[:k0] != lb[:k0]:
        return 'UNEXPLAINED_prefix', None
    for p, q in zip(ra[k0:], lb[k0:]):
        if p['abs_delta'] != q['abs_delta'] or p['block_key'] != q['block_key']:
            return 'UNEXPLAINED_tail', None
    index_b = {(e['block_key'], e['component']): e for e in lb[k0:]}
    for e in ra[k0:]:
        other = index_b.get((e['block_key'], e['component']))
        if other is not None and other != e:
            return 'UNEXPLAINED_shared_entry_differs', None
    detail = {'k0': k0, 'straddling_tie_abs_delta': last_ad,
              'D_members_at_cut': [(e['block_key'], e['component']) for e in ra[k0:]],
              'this_run_members_at_cut': [(e['block_key'], e['component']) for e in lb[k0:]]}
    return 'straddle', detail


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'output root exists (write-once): {OUT}')
    a, b = _load(D_PATH), _load(C_PATH)
    with open(GATE_RESULTS) as handle:
        gate = json.load(handle)
    rows = []
    counts = {}
    tie_group_names = {}
    for i, (x, y) in enumerate(zip(a, b)):
        cls, detail = _classify_row(x, y)
        counts[cls] = counts.get(cls, 0) + 1
        for side, row in (('D', x), ('this_run', y)):
            for g in _tie_groups(row.get(FIELD) or []):
                name = ' | '.join(g['components'])
                t = tie_group_names.setdefault(name, {'rows_D': 0, 'rows_this_run': 0, 'values_identical': set()})
                t['rows_D' if side == 'D' else 'rows_this_run'] += 1
                t['values_identical'].add(g['values_identical'])
        if cls != 'identical':
            rows.append({'row': i, 'cycle': y.get('cycle'), 'class': cls, 'detail': detail})
    unexplained = [r for r in rows if r['class'].startswith('UNEXPLAINED')]
    guard_failures = GUARD.verify(0)
    payload = {
        'stage': 'P5.15 Addendum 25 item 1 -- s44_gate post-gate tie-order analysis (zero solves)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': ('every recourse-jump sidecar difference is the 8682cfdd sort-key change acting on '
                       'exact abs_delta ties (incl. tie groups straddling the top-10 cut)'),
        'gate_result_unchanged': {'gate_pass': gate.get('gate_pass'),
                                  'totals_by_class': gate['c_star_vs_D']['totals_by_class']},
        'inputs': {'D_sidecar': os.path.relpath(D_PATH, REPO), 'D_sidecar_sha256': _sha(D_PATH),
                   'this_run_sidecar': os.path.relpath(C_PATH, REPO), 'this_run_sidecar_sha256': _sha(C_PATH),
                   'gate_results_sha256': _sha(GATE_RESULTS)},
        'n_rows': {'D': len(a), 'this_run': len(b)},
        'row_class_counts': counts,
        'tie_groups_observed': {k: {'rows_D': v['rows_D'], 'rows_this_run': v['rows_this_run'],
                                    'values_identical': sorted(v['values_identical'])}
                                for k, v in sorted(tie_group_names.items())},
        'non_identical_rows': rows,
        'n_unexplained_rows': len(unexplained),
        'hypothesis_holds_for_every_row': (len(a) == len(b) and not unexplained),
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
    }
    os.makedirs(OUT)
    out_path = os.path.join(OUT, 'tie_order_analysis.json')
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    with open(os.path.join(OUT, 'manifest_sha256.json'), 'w') as handle:
        json.dump({os.path.relpath(out_path, REPO): _sha(out_path)}, handle, indent=2)
    GUARD.uninstall()
    print(f"[S44-TIE] rows={payload['n_rows']} classes={counts} unexplained={len(unexplained)} "
          f"hypothesis_holds={payload['hypothesis_holds_for_every_row']} guard={GUARD.counts}")
    print(f"[S44-TIE] tie groups: {json.dumps(payload['tie_groups_observed'])}")
    if not payload['hypothesis_holds_for_every_row'] or guard_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
