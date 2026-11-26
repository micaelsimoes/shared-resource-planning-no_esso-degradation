"""P5.15 Addendum 53/54 triage list (Planner) joined to W112's terminal priced-gap scan. ZERO SOLVES, stdlib only.

The 33 SRP1 cells flagged at threshold 3 x 20,806.32 = 62,418.96 EUR (Addendum 53 Ruling 2), plus the 9 extra cells at
75 kEUR (item D), as inventoried by the read-only Advisor triage (2026-09-28) and summarised in
P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md. The eval-dir prefixes were never committed before this file (W112 could
not flag them); they are recorded here with the item each supports.

Join: every row of data/SRP1/Results/P515S53/w112_consensus_gap/w112_settlement_scan.json whose eval-dir name starts
with a listed prefix (path-qualified; a prefix matching several eval dirs is reported with all matches).
t_sum_terminal = t_tso_plus_t_dso_terminal = sum w * pi * baseMVA * (p_DSO - p_TSO) (W112 identity). Q_cc = Q + t_sum.
Output: data/SRP1/Results/P515S53/triage_list_gap_join/triage_list_gap_join.json (write-once) + manifest.
"""
import hashlib
import json
import os

REPO = os.path.dirname(os.path.abspath(__file__))
SCAN = 'data/SRP1/Results/P515S53/w112_consensus_gap/w112_settlement_scan.json'
OUT_DIR = 'data/SRP1/Results/P515S53/triage_list_gap_join'
OUT = os.path.join(OUT_DIR, 'triage_list_gap_join.json')

TRIAGE_62K = {
    'B_phase_a_ladder': ['7aa017f0', 'bd504ecf', '2a0ba8b2', '0dd237f0'],
    'C_phase_b': ['a30a9faf', 'd7030f59', '4a852725', 'd0c1f160', '1bff3ed2', '10c73abd'],
    'E_ageing': ['c6b53015', '65a5da77', '98e28570', 'ed4a1acc', '06f092d1', '7eb1ce62'],
    'G_year_ladder': ['549476cd', 'dab6a8a2'],
    'H_flex_ladder': ['aa8a76d7', 'f9eae48f', '50dea31c', '74eda68d'],
    'I_marginal_mwh_x2': ['5a6a88b4'],
    'J_f2_ladder': ['f3aa335e', 'a11d7966', '5f3cccb4'],
    'L_f2_certificate': ['5ca4f86c', 'e28de4ac', '7b199ef9', 'e1da0984', '0ee93aca', 'df1a5525', '2ab0ce2d'],
}
EXTRA_75K = {'D_break_even_fit': ['c52e1670', '4a82a64a', '3632b0ae', '36686489', 'd3709599', 'a12d95a2',
                                  'f759dd48', 'c7fee8be', '9246ed01']}
PHASE_MISMATCHED_8 = ['a30a9faf', 'd7030f59', '4a852725', 'd0c1f160', '1bff3ed2', '10c73abd', '549476cd', 'e28de4ac']


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    scan = json.load(open(os.path.join(REPO, SCAN)))
    rows = scan['rows']
    out = {'scan': {'path': SCAN, 'sha256': _sha(SCAN)}, 'threshold_eur': 3 * 20806.323693990707,
           'objective_convention': 'Q = gross_operational_cost, settlement excluded; Q_cc = Q + t_sum',
           'phase_mismatched_8': PHASE_MISMATCHED_8, 'items': {}}
    n = 0
    for group, table in (('flagged_62k', TRIAGE_62K), ('extra_75k', EXTRA_75K)):
        for item, prefixes in table.items():
            entries = []
            for pre in prefixes:
                n += group == 'flagged_62k'
                hits = [r for r in rows if os.path.basename(r['eval_dir'].rstrip('/')).startswith(pre)]
                entries.append({'prefix': pre, 'matches': [
                    {k: r.get(k) for k in ('path', 'eval_dir', 'eval_key', 'status', 't_sum_terminal',
                                           'converged_at_cycle_in_detail', 'cycles_run', 'label')} for r in hits]})
            out['items'][item] = {'group': group, 'cells': entries}
    out['n_flagged_62k'] = n
    os.makedirs(os.path.join(REPO, OUT_DIR), exist_ok=False)
    with open(os.path.join(REPO, OUT), 'x') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(REPO, OUT_DIR, 'manifest_sha256.json'), 'x') as f:
        json.dump({OUT: _sha(OUT), 'p515_s53_triage_list_gap_join.py': _sha('p515_s53_triage_list_gap_join.py'),
                   SCAN: _sha(SCAN)}, f, indent=1)
    for item, v in out['items'].items():
        for c in v['cells']:
            ts = [m['t_sum_terminal'] for m in c['matches']]
            print(f"{item:20s} {c['prefix']} n={len(c['matches'])} t_sum={[round(t) if t is not None else None for t in ts]}")
    print('flagged 62k:', n)


if __name__ == '__main__':
    main()
