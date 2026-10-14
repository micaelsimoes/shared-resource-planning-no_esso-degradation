"""
P5.15 Addendum 76 order, Planner task W177b-2 (Part 2 of W177(b)) -- record checks behind the claim audit of
Section 4.1-4.6 and the Conclusions of main.tex at Overleaf 51a6fb5 (P5_15_W177B_PART2.md). ZERO SOLVES, NO MODEL,
NO PICKLE. Static reading of committed (and on-disk) JSON records and of the read-only manuscript clone.

What it does:
  A. Identity: clone HEAD (declared 51a6fb5), main.tex sha256 (declared, full) and line count (2,312), the letter's
     sha256, clone *.tex unmodified; the frozen tables sha256 = 590088fe...; W153 salvage add-back sha256 = 215fd21d...;
     W173 counts sha256 = 0aa337fc....
  B. Line map: the headings of Section 4.1-4.6 and the Conclusions, and the line of every manuscript fragment the report
     quotes (exact substring search; every fragment must be found exactly once in its stated range).
  C. Item (c) salvage: over the 60 T1 claims (frozen JSON) -- sign changes (d_gross vs d_net), verdict changes
     (gross_verdict vs net_verdict_W153 and vs net_verdict_recomputed_here); the same over the W153 record; the claim-id
     sets of T1 and W153 equal; any T1 claim touching a year-ladder cell (yl_*) or the reference-unit candidate at
     2030/2035; duplicates among the 60 (same cell pair and same d_gross / d_net); T4 (year ladder) 2035 - 2030 gross /
     net signs and verdicts; the A64 rows (not among the 60).
  D. Item (b) F2: the 14 L rows -- the non-neighbour row, and for the 13 neighbours the counts determinate-positive,
     within-positive, within-negative; the neighbour named for the salvage verdict change.
  E. Item (d): (i) every T1 B/C/G claim, T5 Phase B row and T4 year-ladder plan against x = 0 at the baseline with its
     verdict (year-ladder plans vs x = 0 thresholded here by the frozen rule max(3 x larger band, 2 tau) with the
     frozen bands); (ii) the 14 x = 0 neighbours from W173 (search-time margins; which have a v6 re-evaluation);
     (iii) a scan of every evaluation_record.json on disk under data/ -- distinct canonical candidates evaluated on
     the single-scenario SRP1 instance at flexibility multiplier 1 (None) under the BASELINE ageing label and the
     baseline ageing signature (k 35,851.36, phi_cal 0.985, floor 0.70), with node count and energy, and whether each
     is covered by (i)/(ii); records without an ageing label (earlier configurations) are listed separately.
  F. Item (e): every campaign_spec_*.json under data/ -- instance (derived case path), candidates, whether it declares
     a poll design; every campaign_results.json under data/ with a poll_history / termination_certificate, and its
     instance; the p5*.py files that call run_mads or write poll_history; the evaluation records on the 3 x 3 instance.
  G. Other record checks used by single rows: H-cell flexibility multipliers read from the evaluation records; the
     flagged 0.75 MVA / 3 MWh certificate; J bars; E cells' candidate; W153 discount method (reweighting) and per-year
     value signs; T9 dead-zone causes; benchmark sweep and instance.

Guards: SolveProfileGuard(permitted=()) installed before anything else and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit). Refuses to overwrite an existing output.

Output (new directory): data/SRP1/Results/P515S53/w177b_part2/w177b_part2_checks.json and manifest_sha256.json.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S53/w177b_part2 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b_part2_claim_checks.py \\
      > data/SRP1/Results/P515S53/w177b_part2/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W177b-2 claim checks (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W177b-2: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W177b-2: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(P53, 'w177b_part2')
OUT_JSON = os.path.join(OUT_DIR, 'w177b_part2_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
DECLARED_CLONE_HEAD = '51a6fb5'
DECLARED_MAIN_SHA256 = '61fe630cc427d622a855e127f33f96d4be1a7362fe23552b87d2d54318a39346'
DECLARED_MAIN_LINES = 2312

FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
W153_SALVAGE = os.path.join(P53, 'w153_step5_rows', 'w153_salvage_addback.json')
W153_SALVAGE_SHA = '215fd21dd467a173307dd7ae724ea190a25bfa329bcfed2d830768daf3bcd1cb'
W153_DISCOUNT = os.path.join(P53, 'w153_step5_rows', 'w153_discount_row.json')
W173_COUNTS = os.path.join(P53, 'w173_search_counts', 'w173_search_counts.json')
W173_SHA = '0aa337fca5cfaf56da7489a4cdd872da86bc6eb2cd108fa8b7d5ca6ff2c1a5d9'
W3X3_RESULTS = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'campaign_results.json')
S47_A1A = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_a1a_baseline', 'campaign_results.json')
SIGMA_Q = 18449.663947025518  # the measured resolution of the recourse used by the search (main.tex l. 1131; W173)
INSTANCE_3X3_TOKEN = 'SRP1__s53_3x3.json'
INSTANCE_3X3_SHA_PREFIX = '2a64e3c5bd06c30c'

BASELINE_LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
BASELINE_K = 35851.3609417964
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

# section headings (exact text in main.tex) in order; the Conclusions end at \appendix
HEADINGS = [
    ('4.1', '\\subsection{\\textcolor{blue}{Planning Result at the Baseline}}'),
    ('4.2', '\\subsection{\\textcolor{blue}{Break-even Conditions}}'),
    ('4.3', '\\subsection{\\textcolor{blue}{Degradation Decides the Investment}}'),
    ('4.4', '\\subsection{\\textcolor{blue}{Value of Coordination}}'),
    ('4.5', '\\subsection{\\textcolor{blue}{Multi-Scenario Instance}}'),
    ('4.6', '\\subsection{\\textcolor{blue}{Investment Year and Discount Rate}}'),
    ('4.7', '\\subsection{\\textcolor{blue}{Certification Statistics and Computational Performance}}'),
    ('Concl', '\\section{Conclusions}'),
    ('appendix', '\\appendix'),
]

# fragments the report quotes: (id, section, exact substring)
FRAGMENTS = [
    ('a_fourteen', '4.1', 'All fourteen admissible unit neighbours of this plan were evaluated, and none reduced the '
                          'total discounted cost beyond the search resolution.'),
    ('a_returned', '4.1', 'the planning search of Algorithm~\\ref{alg:shared_ess_planning_mads} returned the plan without '
                          'shared storage'),
    ('b_seventeen', '4.2', 'only seventeen of its sixty-one admissible unit neighbours were evaluated'),
    ('b_thirteen', '4.2', 'Of these, thirteen were re-evaluated under the certification procedure, and none was '
                          'determinately better than the returned plan.'),
    ('b_seven_six', '4.2', 'The returned plan is determinately better than seven of them and within resolution of the '
                           'other six.'),
    ('b_every_4h', '4.2', 'the value of every evaluated 4~h configuration at node~7 exceeds its investment cost'),
    ('c_sixty', '4.6', 'Across the 60 pairwise comparisons reported in this section, applying the salvage credit changes '
                       'no sign. It changes two verdicts: the investment-year comparison above'),
    ('c_year_net', '4.6', 'net of salvage it is $-2.3$~k\\euro{}, within resolution (0.19 times)'),
    ('d_every_other', 'Concl', 'every other evaluated plan, from 0.5 to 5~MWh and at one or two nodes, also loses '
                               'determinately'),
    ('d_each_certified', 'Concl', 'each evaluation is certified by an explicit stopping rule'),
    ('d_coordination_impl', 'Concl', 'the value of shared storage cannot be assessed apart from its ageing or from the '
                                     'coordination of the operators who share it'),
    ('e_search', '4.5', 'The planning search again returned the plan without shared storage'),
    ('e_three_searches', '4.7', 'The three planning searches of Algorithm~\\ref{alg:shared_ess_planning_mads} made 41 new '
                                'recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60)'),
    ('disc_any_rate', '4.6', 'The unit does not pay at any rate between 0 and 8\\,\\%'),
    ('disc_mech', '4.6', 'The investment is paid in 2025 and the operational savings accrue over the horizon, so a higher '
                         'rate lowers the value of the unit'),
    ('ageing_both_floors', '4.3', 'The negative investment margin persists across the tested ageing calibrations and both '
                                  'floor settings.'),
    ('lattice_fourteen', None, 'The plan without storage therefore has fourteen admissible neighbours: the smallest unit at '
                               'any non-empty subset of the three nodes, installed in 2025 or 2030.'),
    ('preamble_every_number', None, 'Every reported number is a certified recourse evaluation or a difference between two '
                                    'of them'),
]
LETTER_FRAGMENTS = [
    ('letter_R31', 'It changes no sign in the 60 reported comparisons and two verdicts'),
]


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd=None):
    return subprocess.run(['git'] + args, cwd=cwd or REPO, capture_output=True, text=True, check=True).stdout


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def _canon_key(c):
    if not c:
        return None
    nodes = c.get('nodes') or {}
    return (int(c.get('investment_year')),
            tuple((str(n), float(v[0]), float(v[1])) for n, v in sorted(nodes.items())))


def _nodes_active(c):
    return [n for n, v in sorted((c.get('nodes') or {}).items()) if float(v[0]) > 0 or float(v[1]) > 0]


def _energies(c):
    return [float(v[1]) for n, v in sorted((c.get('nodes') or {}).items()) if float(v[0]) > 0 or float(v[1]) > 0]


def _label(c):
    if c is None:
        return None
    act = [(n, v) for n, v in sorted(c['nodes'].items()) if float(v[0]) > 0 or float(v[1]) > 0]
    if not act:
        return 'x0 (y%d)' % c['investment_year']
    return 'y%d__' % c['investment_year'] + '__'.join('n%s_p%g_e%g' % (n, v[0], v[1]) for n, v in act)


# ---------------------------------------------------------------------------------------------------------------------
def section_identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', '*.tex'], cwd=CLONE)
    with open(MAIN_TEX) as f:
        lines = f.read().split('\n')
    n_lines = len(lines) - (1 if lines and lines[-1] == '' else 0)
    out = {
        'clone_head': head,
        'clone_head_matches_declared': head.startswith(DECLARED_CLONE_HEAD),
        'main_tex_sha256': _sha(MAIN_TEX),
        'main_tex_sha256_matches_declared': _sha(MAIN_TEX) == DECLARED_MAIN_SHA256,
        'main_tex_lines': n_lines,
        'main_tex_lines_match_declared': n_lines == DECLARED_MAIN_LINES,
        'letter_sha256': _sha(LETTER),
        'clone_tex_status_porcelain': status,
        'clone_tex_unmodified': status.strip() == '',
        'frozen_sha256': _sha(FROZEN),
        'frozen_sha256_matches': _sha(FROZEN) == FROZEN_SHA,
        'w153_salvage_sha256': _sha(W153_SALVAGE),
        'w153_salvage_sha256_matches': _sha(W153_SALVAGE) == W153_SALVAGE_SHA,
        'w173_counts_sha256': _sha(W173_COUNTS),
        'w173_counts_sha256_matches': _sha(W173_COUNTS) == W173_SHA,
        'w153_discount_sha256': _sha(W153_DISCOUNT),
        'w3x3_results_sha256': _sha(W3X3_RESULTS),
        'repo_head': _git(['rev-parse', 'HEAD']).strip(),
    }
    return out, lines


def section_line_map(lines):
    heads = {}
    for sid, text in HEADINGS:
        hits = [i + 1 for i, l in enumerate(lines) if l.strip() == text]
        heads[sid] = hits
    ranges = {}
    order = [h[0] for h in HEADINGS]
    for a, b in zip(order[:-1], order[1:]):
        if len(heads[a]) == 1 and len(heads[b]) == 1:
            ranges[a] = [heads[a][0], heads[b][0] - 1]
    frag = []
    for fid, sec, text in FRAGMENTS:
        hits = [i + 1 for i, l in enumerate(lines) if text in l]
        in_range = None
        if sec is not None and sec in ranges and len(hits) == 1:
            in_range = ranges[sec][0] <= hits[0] <= ranges[sec][1]
        frag.append({'id': fid, 'section': sec, 'lines': hits, 'found_once': len(hits) == 1,
                     'in_stated_section': in_range})
    with open(LETTER) as f:
        llines = f.read().split('\n')
    lfrag = []
    for fid, text in LETTER_FRAGMENTS:
        hits = [i + 1 for i, l in enumerate(llines) if text in l]
        lfrag.append({'id': fid, 'lines': hits, 'found_once': len(hits) == 1})
    return {'headings': heads, 'ranges': ranges, 'fragments': frag, 'letter_fragments': lfrag}


# ---------------------------------------------------------------------------------------------------------------------
def section_c_salvage(fz, w153):
    claims = fz['tables']['claims']
    rows = []
    for c in claims:
        rows.append({
            'claim_id': c['claim_id'], 'family': c['family'], 'ref_cell': c['ref_cell'], 'other_cell': c['other_cell'],
            'd_gross': c['d_gross'], 'd_net': c['d_net'], 'gross_multiple': c['gross_multiple'],
            'net_multiple': c['net_multiple'], 'gross_verdict': c['gross_verdict'],
            'net_verdict_W153': c['net_verdict_W153'], 'net_verdict_recomputed_here': c['net_verdict_recomputed_here'],
            'sign_changes': _sign(c['d_gross']) != _sign(c['d_net']),
            'verdict_changes_W153': c['gross_verdict'] != c['net_verdict_W153'],
            'verdict_changes_recomputed': c['gross_verdict'] != c['net_verdict_recomputed_here'],
        })
    w_claims = w153['result']['claims']
    w_rows = [{'claim_id': c['claim_id'], 'sign_differs': c['sign_differs_between_conventions'],
               'verdict_differs': c['verdict_differs_between_conventions'], 'gross_verdict': c['gross_verdict'],
               'net_verdict': c['net_verdict']} for c in w_claims]
    # cells touching the year ladder or the reference unit at 2030 / 2035
    yl_cells = set(fz['tables']['year_ladder']['per_year'].keys())
    ref_unit_late = []
    for c in claims:
        for cell, inst in (c.get('instance') or {}).items():
            cc = inst.get('candidate_canonical') or {}
            if cc and cc.get('investment_year') in (2030, 2035) and _nodes_active(cc) == ['7'] and \
                    cc['nodes']['7'] == [0.25, 1.0]:
                ref_unit_late.append({'claim_id': c['claim_id'], 'cell': cell})
    yl_keys = {v['eval_key'] for v in fz['tables']['year_ladder']['per_year'].values()}
    claims_with_yl_eval_key = [c['claim_id'] for c in claims
                               if any((inst.get('eval_key') in yl_keys) for inst in (c.get('instance') or {}).values())]
    claims_with_yl_cell = [c['claim_id'] for c in claims
                           if str(c['ref_cell']).startswith('yl_') or str(c['other_cell']).startswith('yl_')]
    # duplicates: same (ref, other, d_gross, d_net)
    seen = {}
    dups = []
    for c in claims:
        # same cell pair and the same |d| (a claim may state the difference with the opposite sign, e.g. V - I)
        key = (tuple(sorted((c['ref_cell'], c['other_cell']))), round(abs(c['d_gross']), 3), round(abs(c['d_net']), 3))
        if key in seen:
            dups.append({'claim_id': c['claim_id'], 'duplicate_of': seen[key]})
        else:
            seen[key] = c['claim_id']
    yl = fz['tables']['year_ladder']
    t4 = {'D_gross': yl['D_gross'], 'D_net': yl['D_net'], 'gross_verdict': yl['gross_v6']['verdict'],
          'gross_multiple': yl['gross_v6']['multiple'], 'net_verdict': yl['net_v6']['verdict'],
          'net_multiple': yl['net_v6']['multiple'], 'threshold': yl['gross_v6']['threshold'],
          'sign_changes': _sign(yl['D_gross']) != _sign(yl['D_net']),
          'verdict_changes': yl['gross_v6']['verdict'] != yl['net_v6']['verdict'],
          'year_ladder_eval_keys': sorted(yl_keys)}
    a64_ids = [r['claim_id'] for r in fz['tables']['a64']['rows']]
    return {
        'n_claims_T1': len(claims),
        'n_claims_W153': len(w_claims),
        'claim_id_sets_equal': sorted(r['claim_id'] for r in rows) == sorted(r['claim_id'] for r in w_rows),
        'n_sign_changes_T1': sum(r['sign_changes'] for r in rows),
        'n_verdict_changes_T1_vs_W153_net': sum(r['verdict_changes_W153'] for r in rows),
        'n_verdict_changes_T1_vs_recomputed_net': sum(r['verdict_changes_recomputed'] for r in rows),
        'verdict_changes_T1': [r for r in rows if r['verdict_changes_W153'] or r['verdict_changes_recomputed']],
        'n_sign_differs_W153': sum(bool(r['sign_differs']) for r in w_rows),
        'n_verdict_differs_W153': sum(bool(r['verdict_differs']) for r in w_rows),
        'verdict_differs_W153': [r for r in w_rows if r['verdict_differs']],
        'claims_touching_reference_unit_at_2030_or_2035': ref_unit_late,
        'claims_with_year_ladder_eval_key': claims_with_yl_eval_key,
        'claims_with_yl_cell': claims_with_yl_cell,
        'year_ladder_years': sorted(yl_cells),
        'duplicates_among_60': dups,
        'n_distinct_comparisons_among_60': len(claims) - len(dups),
        'T4_2035_minus_2030': t4,
        'a64_claim_ids_not_in_60': [i for i in a64_ids if i not in {r['claim_id'] for r in rows}],
        'rows': rows,
    }


def section_b_f2(fz):
    L = [c for c in fz['tables']['claims'] if c['family'] == 'L']
    nb = [c for c in L if 'not_a_neighbour' not in c['claim_id']]
    out = {'n_L_rows': len(L), 'n_neighbour_rows': len(nb),
           'non_neighbour_rows': [c['claim_id'] for c in L if 'not_a_neighbour' in c['claim_id']],
           'incumbent_cells': sorted({c['ref_cell'] for c in L})}
    det_pos = [c for c in nb if c['gross_verdict'] == 'determinate' and c['d_gross'] > 0]
    det_neg = [c for c in nb if c['gross_verdict'] == 'determinate' and c['d_gross'] < 0]
    win_pos = [c for c in nb if c['gross_verdict'] != 'determinate' and c['d_gross'] > 0]
    win_neg = [c for c in nb if c['gross_verdict'] != 'determinate' and c['d_gross'] < 0]
    out.update({'n_determinate_positive': len(det_pos), 'n_determinate_negative': len(det_neg),
                'n_within_positive': len(win_pos), 'n_within_negative': len(win_neg),
                'within_negative': [{'claim_id': c['claim_id'], 'other_cell': c['other_cell'], 'd_gross': c['d_gross'],
                                     'bar': c['gross_threshold_or_bar'], 'multiple': c['gross_multiple'],
                                     'verdict': c['gross_verdict']} for c in win_neg],
                'rows': [{'claim_id': c['claim_id'], 'other_cell': c['other_cell'], 'd_gross': c['d_gross'],
                          'bar': c['gross_threshold_or_bar'], 'multiple': c['gross_multiple'],
                          'gross_verdict': c['gross_verdict'], 'd_net': c['d_net'], 'net_multiple': c['net_multiple'],
                          'net_verdict': c['net_verdict_W153'], 'ref_status': c['ref_status']} for c in nb]})
    dz = [c for c in fz['tables']['dead_zone']['cells'] if c['cell'] == 'ref:5ca4f86c']
    out['incumbent_dead_zone_row'] = dz[0] if dz else None
    return out


# ---------------------------------------------------------------------------------------------------------------------
def _eval_records():
    recs = []
    for f in sorted(glob.glob(os.path.join('data', '**', 'evaluation_record.json'), recursive=True)):
        try:
            d = _load(f)
        except Exception as exc:  # pragma: no cover - reported
            recs.append({'path': f, 'error': repr(exc)})
            continue
        cfg = d.get('configuration') or {}
        di = (cfg.get('derived_instance') or {})
        exp = ((d.get('ess_ageing_readback_terminal') or {}).get('expected') or {})
        recs.append({
            'path': f, 'campaign_id': d.get('campaign_id'), 'eval_key': d.get('eval_key'),
            'candidate_canonical': d.get('candidate_canonical'), 'candidate_label': d.get('candidate_label'),
            'status': d.get('status'), 'flex_price_multiplier': d.get('flex_price_multiplier'),
            'ageing_label': d.get('ess_ageing_baseline_label'),
            'ageing_k': exp.get('k'), 'ageing_phi_cal': exp.get('phi_cal_in_model'),
            'ageing_floor': exp.get('floor_row_lower'), 'ageing_enabled': exp.get('ageing_enabled'),
            'derived_case_path': di.get('case_path'), 'derived_case_sha256': di.get('case_sha256'),
            'overrides': sorted((d.get('evaluation_overrides_effective') or {}).keys()),
        })
    return recs


def _is_baseline_record(r):
    return (r.get('derived_case_path') is None and r.get('flex_price_multiplier') in (None, 1.0)
            and r.get('ageing_label') == BASELINE_LABEL and not r.get('overrides')
            and r.get('ageing_k') is not None and abs(float(r['ageing_k']) - BASELINE_K) < 1e-6
            and r.get('ageing_phi_cal') == 0.985 and r.get('ageing_floor') == 0.7 and r.get('ageing_enabled') is True)


def section_d_inventory(fz, w173, recs):
    t = fz['tables']
    tau = fz['constants']['TAU']
    ref_band = t['cells']['ref:7aa017f0']['band_width']
    covered = {}
    rows_i = []
    for c in t['claims']:
        if c['family'] in ('B', 'C', 'G') and c['ref_cell'] == 'ref:7aa017f0':
            cc = c['instance'][c['other_cell']]['candidate_canonical'] if c['other_cell'] in c['instance'] else None
            rows_i.append({'source': 'T1 ' + c['claim_id'], 'candidate': _label(cc), 'n_nodes': len(_nodes_active(cc)),
                           'E_per_node': _energies(cc), 'd_gross': c['d_gross'], 'multiple': c['gross_multiple'],
                           'verdict': c['gross_verdict'], 'net_verdict': c['net_verdict_W153']})
            covered.setdefault(_canon_key(cc), []).append('T1 ' + c['claim_id'])
    phase_b_rows_without_candidate = []
    for r in t['phase_b']:
        if 'candidate_canonical' not in r:
            # pb_y2025_n5_v6: "the extension v6 re-run (claim C:y2025__n5_p0.25_e0.5); T1 carries its full row"
            phase_b_rows_without_candidate.append({'cell': r['cell'], 'note': r.get('note')})
            continue
        cc = r['candidate_canonical']
        rows_i.append({'source': 'T5 ' + r['cell'] + (' (superseded)' if r['superseded'] else ''), 'candidate': _label(cc),
                       'n_nodes': len(_nodes_active(cc)), 'E_per_node': _energies(cc), 'd_gross': r['M_gross'],
                       'multiple': r['v6_multiple'], 'verdict': r['v6_verdict']})
        covered.setdefault(_canon_key(cc), []).append('T5 ' + r['cell'])
    for y, v in t['year_ladder']['per_year'].items():
        app = t['cells_appended_w160']['yl_y%s' % y]
        thr = max(3.0 * max(ref_band, app['band_width']), 2.0 * tau)
        cc = app['candidate_canonical']
        rows_i.append({'source': 'T4 yl_y%s vs x0 (threshold computed here: max(3 x larger band, 2 tau))' % y,
                       'candidate': _label(cc), 'n_nodes': 1, 'E_per_node': _energies(cc),
                       'd_gross': v['M_gross_w153_form'], 'd_net': v['M_net_w153_form'], 'threshold': thr,
                       'multiple': v['M_gross_w153_form'] / thr, 'multiple_net': v['M_net_w153_form'] / thr,
                       'verdict': 'determinate' if abs(v['M_gross_w153_form']) >= thr else 'within resolution',
                       'net_verdict': 'determinate' if abs(v['M_net_w153_form']) >= thr else 'within resolution',
                       'bands': {'ref': ref_band, 'yl': app['band_width']}})
        covered.setdefault(_canon_key(cc), []).append('T4 yl_y%s' % y)
    within = [r for r in rows_i if r['verdict'] != 'determinate' or r.get('net_verdict') not in (None, 'determinate')]
    # (ii) the 14 x0 neighbours (W173)
    nbr = []
    for r in w173['item2']['x0']['rows']:
        n_nodes = r['label'].count('__n')
        nbr.append({'label': r['label'], 'n_nodes': n_nodes, 'search_F_minus_F0': r['search_F_minus_F_inc_eur'],
                    'search_resolution': r['search_resolution_eur'],
                    'search_multiple': r['search_F_minus_F_inc_eur'] / r['search_resolution_eur'],
                    'search_class': r['search_symmetric_class'],
                    'v6_reevaluated': bool(r['frozen_table_rows'])})
    # (iii) record scan
    base = [r for r in recs if 'error' not in r and _is_baseline_record(r)]
    by_cand = {}
    for r in base:
        k = _canon_key(r['candidate_canonical'])
        by_cand.setdefault(k, []).append(r)
    inv = []
    for k, rs in sorted(by_cand.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
        cc = rs[0]['candidate_canonical']
        inv.append({'candidate': _label(cc), 'n_nodes': len(_nodes_active(cc)), 'E_per_node': _energies(cc),
                    'E_total': sum(_energies(cc)), 'n_records': len(rs),
                    'statuses': sorted({str(r['status']) for r in rs}),
                    'campaigns': sorted({str(r['campaign_id']) for r in rs}),
                    'covered_by_frozen_tables': covered.get(k, []),
                    'is_x0_neighbour_in_W173': _label(cc) in {n['label'] for n in nbr}})
    # the s47 A1a baseline ladder (30 plans, production exit, earlier code state): margins against the search-era
    # resolution max(bar_x + bar_0, sigma_Q), bar_0 = the pinned x0 record's bar (campaign baseline_inputs)
    a1a = _load(S47_A1A)
    q0_bar = a1a['baseline_inputs']['Q0_bar_eur']
    a1a_rows = []
    for r in a1a['table']:
        res = max(r['bar'] + q0_bar, SIGMA_Q)
        a1a_rows.append({'label': r['label'], 'status': r['status'], 'value_minus_I': r['value_minus_I'],
                         'bar': r['bar'], 'bar_x0': q0_bar, 'search_resolution': res,
                         'multiple': abs(r['value_minus_I']) / res,
                         'loses_beyond_search_resolution': (r['value_minus_I'] < 0 and abs(r['value_minus_I']) > res)})
    a1a_summary = {'n': len(a1a_rows), 'n_loses_beyond_search_resolution':
                   sum(r['loses_beyond_search_resolution'] for r in a1a_rows),
                   'min_multiple': min(r['multiple'] for r in a1a_rows),
                   'min_abs_value_minus_I': min(abs(r['value_minus_I']) for r in a1a_rows),
                   'max_abs_value_minus_I': max(abs(r['value_minus_I']) for r in a1a_rows),
                   'git_head_at_run': a1a.get('git_head_at_run'), 'label': a1a.get('LABEL'), 'rows': a1a_rows}
    no_label = [r for r in recs if 'error' not in r and r.get('ageing_label') is None]
    no_label_cands = {}
    for r in no_label:
        no_label_cands.setdefault(_canon_key(r['candidate_canonical']), set()).add(str(r['campaign_id']))
    no_label_rows = []
    for k, camps in sorted(no_label_cands.items(), key=lambda kv: str(kv[0])):
        cc = {'investment_year': k[0], 'nodes': {n: [p, e] for n, p, e in k[1]}} if k else None
        no_label_rows.append({'candidate': _label(cc), 'n_nodes': len(_nodes_active(cc)) if cc else None,
                              'campaigns': sorted(camps)})
    return {
        'tau': tau, 'ref_band': ref_band, 'phase_b_rows_without_candidate': phase_b_rows_without_candidate,
        'frozen_rows_vs_x0': rows_i, 'frozen_rows_not_determinate': within,
        'x0_neighbours_W173': nbr,
        's47_a1a_baseline_ladder': a1a_summary,
        'x0_neighbours_three_node': [n for n in nbr if n['n_nodes'] == 3],
        'x0_neighbours_not_reevaluated': [n for n in nbr if not n['v6_reevaluated']],
        'record_scan': {
            'n_records_total': len(recs), 'n_records_error': sum('error' in r for r in recs),
            'n_baseline_records': len(base), 'n_baseline_distinct_candidates': len(inv),
            'baseline_candidates': inv,
            'baseline_candidates_three_node': [r for r in inv if r['n_nodes'] == 3],
            'baseline_candidates_not_covered_and_not_x0_neighbour':
                [r for r in inv if not r['covered_by_frozen_tables'] and not r['is_x0_neighbour_in_W173']
                 and r['n_nodes'] > 0],
            'baseline_E_range_over_candidates_with_storage': (
                [min(min(r['E_per_node']) for r in inv if r['n_nodes'] > 0),
                 max(max(r['E_per_node']) for r in inv if r['n_nodes'] > 0)] if inv else None),
            'n_records_without_ageing_label': len(no_label),
            'candidates_without_ageing_label': no_label_rows,
            'criteria': 'derived_case_path absent (single-scenario SRP1); flex_price_multiplier None or 1.0; '
                        'ess_ageing_baseline_label == %r; no evaluation overrides; terminal readback expected k == %r, '
                        'phi_cal_in_model 0.985, floor_row_lower 0.7, ageing_enabled True' % (BASELINE_LABEL, BASELINE_K),
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
def section_e_3x3(recs):
    specs = sorted(glob.glob(os.path.join('data', '**', 'campaign_spec_*.json'), recursive=True))
    spec_rows = []
    for f in specs:
        try:
            s = _load(f)
        except Exception as exc:
            spec_rows.append({'path': f, 'error': repr(exc)})
            continue
        cfg = s.get('configuration') or {}
        di = cfg.get('derived_instance') or {}
        txt = json.dumps(s)
        is3 = INSTANCE_3X3_TOKEN in str(di.get('case_path')) or str(di.get('case_sha256', '')).startswith(
            INSTANCE_3X3_SHA_PREFIX)
        mentions3 = INSTANCE_3X3_TOKEN in txt or INSTANCE_3X3_SHA_PREFIX in txt
        extra = s.get('extra') or {}
        spec_rows.append({
            'path': f, 'campaign_id': s.get('campaign_id'), 'derived_case_path': di.get('case_path'),
            'is_3x3_instance': is3, 'mentions_3x3_anywhere': mentions3,
            'declares_poll_design': ('poll_design' in extra) or ('poll_design' in s),
            'candidates': [c.get('label') for c in (s.get('candidates') or []) if isinstance(c, dict)],
            'candidates_canonical': [_label(c.get('canonical')) for c in (s.get('candidates') or [])
                                     if isinstance(c, dict)],
        })
    results = sorted(glob.glob(os.path.join('data', '**', 'campaign_results.json'), recursive=True))
    res_rows = []
    for f in results:
        try:
            d = _load(f)
        except Exception as exc:
            res_rows.append({'path': f, 'error': repr(exc)})
            continue
        if not isinstance(d, dict):
            continue
        has_poll = 'poll_history' in d or 'termination_certificate' in d
        if has_poll:
            sp = d.get('campaign_spec_path')
            inst = None
            if sp and os.path.exists(sp):
                s = _load(sp)
                inst = ((s.get('configuration') or {}).get('derived_instance') or {}).get('case_path')
            txt = json.dumps(d)
            res_rows.append({'path': f, 'campaign_spec_path': sp, 'derived_case_path_of_spec': inst,
                             'mentions_3x3_anywhere': INSTANCE_3X3_TOKEN in txt or INSTANCE_3X3_SHA_PREFIX in txt,
                             'n_polls': d.get('n_polls'), 'termination': d.get('termination'),
                             'final_incumbent': d.get('final_incumbent') if not isinstance(d.get('final_incumbent'),
                                                                                           dict) else
                             d['final_incumbent'].get('label', None)})
    # scripts that run the MADS loop or write a poll history
    scripts = sorted(f for f in glob.glob('p5*.py') if f != os.path.basename(__file__))
    callers = []
    for f in scripts:
        with open(f, errors='replace') as fh:
            txt = fh.read()
        if ('run_mads(' in txt) or ("'poll_history'" in txt) or ('"poll_history"' in txt):
            callers.append({'script': f, 'calls_run_mads': 'run_mads(' in txt,
                            'mentions_3x3_instance': INSTANCE_3X3_TOKEN in txt or 's53_3x3' in txt})
    recs3 = [r for r in recs if 'error' not in r and INSTANCE_3X3_TOKEN in str(r.get('derived_case_path'))]
    git_log = _git(['log', '--all', '--format=%h %s'])
    git_hits = [l for l in git_log.splitlines()
                if ('3 x 3' in l or '3x3' in l or '3 × 3' in l)
                and any(w in l.lower() for w in ('search', 'mads', 'poll', 'neighbour'))]
    return {
        'n_campaign_specs_scanned': len(spec_rows),
        'specs_on_3x3_instance': [r for r in spec_rows if r.get('is_3x3_instance')],
        'specs_mentioning_3x3_not_on_instance': [r for r in spec_rows if r.get('mentions_3x3_anywhere')
                                                 and not r.get('is_3x3_instance')],
        'specs_declaring_poll_design': [r for r in spec_rows if r.get('declares_poll_design')],
        'specs_on_3x3_declaring_poll_design': [r for r in spec_rows if r.get('declares_poll_design')
                                               and r.get('is_3x3_instance')],
        'n_campaign_results_scanned': len(results),
        'search_campaign_results': res_rows,
        'search_campaign_results_on_3x3': [r for r in res_rows if INSTANCE_3X3_TOKEN in
                                           str(r.get('derived_case_path_of_spec'))],
        'scripts_running_search': callers,
        'evaluation_records_on_3x3': [{'path': r['path'], 'campaign_id': r['campaign_id'],
                                       'candidate': _label(r['candidate_canonical']), 'status': r['status']}
                                      for r in recs3],
        'distinct_candidates_evaluated_on_3x3': sorted({_label(r['candidate_canonical']) for r in recs3}),
        'git_log_all_branches_hits_3x3_and_search_words': git_hits,
    }


# ---------------------------------------------------------------------------------------------------------------------
def section_g_misc(fz, recs):
    t = fz['tables']
    out = {}
    # H cells: flexibility multiplier from the evaluation records (eval_key prefix = cell suffix)
    h = {}
    for cell in ('h_50dea31c', 'h_74eda68d', 'h_aa8a76d7', 'h_f9eae48f'):
        ek = t['cells'][cell]['eval_key']
        m = [r for r in recs if r.get('eval_key') == ek]
        h[cell] = {'eval_key': ek, 'records': [r['path'] for r in m],
                   'flex_price_multiplier': sorted({str(r['flex_price_multiplier']) for r in m}),
                   'candidate': _label(t['cells'][cell]['candidate_canonical']),
                   'status': t['cells'][cell]['status'], 'cause_uncertified': t['cells'][cell]['cause_uncertified']}
    for cell in ('h_x0_m175', 'h_unit_m175'):
        ek = t['cells_appended_w160'][cell]['eval_key']
        m = [r for r in recs if r.get('eval_key') == ek]
        h[cell] = {'eval_key': ek, 'records': [r['path'] for r in m],
                   'flex_price_multiplier': sorted({str(r['flex_price_multiplier']) for r in m}),
                   'candidate': _label(t['cells_appended_w160'][cell]['candidate_canonical'])}
    out['H_cells'] = h
    d4 = t['cells']['d_4a82a64a']
    iv = t['break_even']['conservative_A64']['intervals']['n7_4h_e3']
    b43 = [c for c in t['claims'] if c['claim_id'] == 'B:n7_4h_e3'][0]
    out['flagged_n7_4h_e3'] = {'flag_non_clean_turning_point': d4.get('flag_non_clean_turning_point'),
                               'uncertified_bar': iv['bar'], 'ratio_with_uncertified_bar': b43['d_gross'] / iv['bar'],
                               'd_gross': b43['d_gross']}
    out['J_bars'] = {c['claim_id']: {'d_gross': c['d_gross'], 'bar': c['gross_threshold_or_bar'],
                                     'multiple': c['gross_multiple'], 'verdict': c['gross_verdict']}
                     for c in t['claims'] if c['family'] in ('I', 'J', 'H')}
    e_c = {}
    for cell in ('e_c2', 'e_c2_calfade', 'e_c3_unit', 'e_c3_midblock', 'e_c4', 'e_no_ageing'):
        e_c[cell] = _label(t['cells'][cell]['candidate_canonical'])
    e_c['e_soh050'] = _label(t['cells_appended_w160']['e_soh050']['candidate_canonical'])
    out['E_cells_candidate'] = e_c
    out['E_cells_all_reference_unit'] = len(set(e_c.values())) == 1
    out['ageing_rows'] = [{k: r[k] for k in ('arm', 'value_minus_I', 'verdict', 'floor_year_070', 'AE', 'EFC')}
                          for r in t['ageing']['rows']]
    out['a64_rows'] = [{k: r.get(k) for k in ('claim_id', 'd_gross', 'gross_threshold_or_bar', 'gross_multiple',
                                              'gross_verdict')} for r in t['a64']['rows']]
    disc = _load(W153_DISCOUNT)['result']
    out['discount'] = {'formula': disc['formula'], 'V_per_year': disc['V_per_year'],
                       'all_V_per_year_undiscounted_positive': all(v['V_undiscounted'] > 0
                                                                   for v in disc['V_per_year'].values()),
                       'table': [{k: r[k] for k in ('rate', 'V', 'value_minus_I', 'a61_threshold_conservative',
                                                    'value_minus_I_verdict_a61')} for r in disc['table']],
                       'is_reweighting_of_one_evaluation_pair': ('V(r) = sum_y V_y_undiscounted' in disc['formula'])}
    out['dead_zone_causes'] = [{'cell': c['cell'], 'cause': c['cause']} for c in t['dead_zone']['cells']]
    b = t['benchmark']
    out['benchmark'] = {'instance': b['instance']['label'], 'candidate_key': b['instance']['candidate_key'],
                        'sweep': b['sweep'], 'reverse_flow_interface_hours': b['reverse_flow_interface_hours'],
                        'multiple': b['multiple'], 'benefit_relative': b['benefit_relative'],
                        'arms': b['arms']}
    out['three_by_three_rows'] = [{k: r.get(k) for k in ('quantity', 'value', 'label', 'source')}
                                  for r in t['three_by_three']['rows']]
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MANIFEST):
        raise SystemExit('refusing to overwrite an existing output in %s' % OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    ident, lines = section_identity()
    fz = _load(FROZEN)
    w153 = _load(W153_SALVAGE)
    w173 = _load(W173_COUNTS)
    recs = _eval_records()
    result = {
        'stage': 'P5.15 W177b-2 (Addendum 76 Part 2) -- record checks behind the claim audit of Section 4.1-4.6 and the '
                 'Conclusions',
        'objective_convention': fz['objective_convention'],
        'started_utc': started,
        'identity': ident,
        'line_map': section_line_map(lines),
        'item_c_salvage': section_c_salvage(fz, w153),
        'item_b_f2': section_b_f2(fz),
        'item_d_inventory': section_d_inventory(fz, w173, recs),
        'item_e_3x3': section_e_3x3(recs),
        'misc': section_g_misc(fz, recs),
    }
    integrity = {
        'clone_head_matches_declared': ident['clone_head_matches_declared'],
        'main_tex_sha256_matches_declared': ident['main_tex_sha256_matches_declared'],
        'main_tex_lines_match_declared': ident['main_tex_lines_match_declared'],
        'clone_tex_unmodified': ident['clone_tex_unmodified'],
        'frozen_sha256_matches': ident['frozen_sha256_matches'],
        'w153_salvage_sha256_matches': ident['w153_salvage_sha256_matches'],
        'w173_counts_sha256_matches': ident['w173_counts_sha256_matches'],
        'every_fragment_found_once': all(f['found_once'] for f in result['line_map']['fragments']),
        'every_sectioned_fragment_in_its_section': all(f['in_stated_section'] is not False
                                                       for f in result['line_map']['fragments']),
        'every_letter_fragment_found_once': all(f['found_once'] for f in result['line_map']['letter_fragments']),
        'no_evaluation_record_read_error': result['item_d_inventory']['record_scan']['n_records_error'] == 0,
        'frozen_unchanged_at_end': _sha(FROZEN) == FROZEN_SHA,
        'main_tex_unchanged_at_end': _sha(MAIN_TEX) == DECLARED_MAIN_SHA256,
    }
    result['integrity'] = integrity
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts,
                                                'verify_0_failures': gfail},
                        'pickle': PICKLE_COUNTS, 'forbidden_modules_imported': forbidden}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)
    script = os.path.basename(__file__)
    inputs = [MAIN_TEX, LETTER, FROZEN, W153_SALVAGE, W153_DISCOUNT, W173_COUNTS, W3X3_RESULTS, S47_A1A]
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'script': {script: _sha(script)},
                'inputs': {p: _sha(p) for p in inputs},
                'evaluation_records_scanned': {r['path']: _sha(r['path']) for r in recs}}
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    ok = all(integrity.values()) and not gfail and PICKLE_COUNTS == {'load': 0, 'loads': 0} and not forbidden
    print(json.dumps({'integrity': integrity, 'guards': result['guards'], 'ok': ok}, indent=1))
    c = result['item_c_salvage']
    print('item (c): claims %d / W153 %d, ids equal %s; sign changes %d; verdict changes T1 %d / W153 %d; T4 sign %s '
          'verdict %s; dups %d; distinct %d' % (
              c['n_claims_T1'], c['n_claims_W153'], c['claim_id_sets_equal'], c['n_sign_changes_T1'],
              c['n_verdict_changes_T1_vs_W153_net'], c['n_verdict_differs_W153'],
              c['T4_2035_minus_2030']['sign_changes'], c['T4_2035_minus_2030']['verdict_changes'],
              len(c['duplicates_among_60']), c['n_distinct_comparisons_among_60']))
    b = result['item_b_f2']
    print('item (b): neighbours %d: det+ %d, det- %d, within+ %d, within- %d' % (
        b['n_neighbour_rows'], b['n_determinate_positive'], b['n_determinate_negative'], b['n_within_positive'],
        b['n_within_negative']))
    d = result['item_d_inventory']
    print('item (d): frozen rows vs x0 %d, not determinate %d; x0 neighbours three-node %d, not re-evaluated %d; '
          'baseline records %d, distinct candidates %d, three-node %d, uncovered %d' % (
              len(d['frozen_rows_vs_x0']), len(d['frozen_rows_not_determinate']), len(d['x0_neighbours_three_node']),
              len(d['x0_neighbours_not_reevaluated']), d['record_scan']['n_baseline_records'],
              d['record_scan']['n_baseline_distinct_candidates'],
              len(d['record_scan']['baseline_candidates_three_node']),
              len(d['record_scan']['baseline_candidates_not_covered_and_not_x0_neighbour'])))
    e = result['item_e_3x3']
    print('item (e): specs %d, on 3x3 %d (poll design %d); search results %d, on 3x3 %d; distinct 3x3 candidates %s' % (
        e['n_campaign_specs_scanned'], len(e['specs_on_3x3_instance']), len(e['specs_on_3x3_declaring_poll_design']),
        len(e['search_campaign_results']), len(e['search_campaign_results_on_3x3']),
        e['distinct_candidates_evaluated_on_3x3']))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
