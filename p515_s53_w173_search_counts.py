"""
P5.15 Addendum 69, Planner task W173 -- the two `% [CONFIRM -- W173]` points of manuscript section 2.1
(Overleaf clone manuscript/6a67305f25e8348fb71380c3/ at 407f8df, main.tex l. 438 and l. 446). ZERO SOLVES, READ-ONLY.

Item 1 (l. 438): Delta_0 and the frame-size rule of variant A (s47 Phase B record, s51 F2 Phase B) as the code ran
  them: the frozen spec field `extra.poll_design` (spec path + sha256), the code constants and the update lines (found
  by exact text), and the REALIZED poll-size sequence from each run's `poll_history` (poll index, Halton t, Delta,
  dispositions, decision, next Delta). The same for the s53 F2 certificate continuation (variant B). Also: N^max
  (= MAX_NEW_EVALUATIONS, a cap on NEW evaluations, cache hits not counted), the completion rule / cap 30, the
  Householder n + 1 construction (recomputed with PB.poll_directions and compared with the recorded directions), the
  s53 snap tie-break (levels and which level decided each snap, from the recorded snap table).

Item 2 (l. 446): neighbour counts for the two certificates.
  Box = every bound-, rule- (P = 0 <=> E = 0, 2 h <= E/P <= 4 h, E <= 5 MWh, year index in range) and budget-feasible
  canonical lattice point z != inc with ||z - inc||_inf <= 1 in (zP5, zE5, zP7, zE7, zP9, zE9, zY): the production
  search's own `PB.Lattice.neighbourhood`, recomputed here from the W2 unit-cost table exactly as the launchers build
  the lattice, and compared with the box each run recorded (`lattice_neighbourhood_of_incumbent`).
  (a) the poll set of the final iteration: size, evaluated (new + cache hits);
  (b) the box: size, evaluated at any stage of the search (s53 `in_cache` = the cache at the start of the
      continuation plus its new evaluations), not evaluated; and, separately, a scoped scan of every
      `evaluation_record.json` on disk under data/ (one per evaluation directory, tracked or not) and of the `points`
      of every git-tracked `campaign_results.json` under data/SRP1/Results for an evaluation at the same canonical
      candidate and the same flexibility-price multiplier (to state "not evaluated" with its scope);
  (c) each evaluated box neighbour vs the certificate: search-time class (symmetric: worse / better if the
      difference exceeds the search resolution max(bar_x + bar_inc, sigma_Q), else within), and the re-settled
      frozen Step-6 table row (frozen_step6_tables_v1_590088fe.json: `phase_b` rows and `claims` for x = 0; `claims`
      family L for F2) with its verdict.
  The current sentence of main.tex l. 443 is quoted from the clone by exact text.

GUARDS: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0; every other
SolveProfileGuard instance installed by the imported launchers (their parent guards) is found in sys.modules and
verified at 0 too. pickle.load / pickle.loads blocked for the whole run, counters verified at 0. No model is built;
the only project code used is the pure lattice / direction functions of p515_s47_phase_b_record.py (PB) and
p515_s53_f2_certificate.py (S53.directions_2n), which the searches themselves ran.

Output (refuses to overwrite): data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json,
manifest_sha256.json (sha256 of every input read and of the output). Command (repo root, canonical interpreter,
attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w173_search_counts.py \\
      > data/SRP1/Results/P515S53/w173_search_counts/launch.log 2>&1
Exit: 0 written and every integrity check holds; 3 written with integrity failures listed; 1 guard fault.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W173 search counts (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W173: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W173: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import p515_s47_phase_b_record as PB  # noqa: E402  (pure lattice / direction functions; installs its parent guard)
import p515_s53_f2_certificate as S53  # noqa: E402  (directions_2n; installs its parent guard)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w173_search_counts')
OUT_JSON = os.path.join(OUT_DIR, 'w173_search_counts.json')
OUT_MAN = os.path.join(OUT_DIR, 'manifest_sha256.json')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
MAIN_TEX_SHA = '9891057c03437cda16da749885001dbdedddde2d917d4a11deeeb2d68ade3062'
CLONE_HEAD = '407f8df'

_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b')
_P51 = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'campaign_s53_f2_certificate_r1')
RUNS = {
    's47': {'results': os.path.join(_P47, 'campaign_results.json'),
            'spec': os.path.join(_P47, 'campaign_spec_s47_phase_b_8cfa264e.json'),
            'spec_sha256': '8cfa264e7ab0d92ac0832657c9b328d70b45851ba92f4623384d11fddb334185',
            'results_sha256': '906f9da353ebe2158c13176c22ea49602d9cdbfab142f7334b7e54e9db1fdb58',
            'variant': 'A', 'script': 'p515_s47_phase_b_record.py'},
    's51': {'results': os.path.join(_P51, 'campaign_results.json'),
            'spec': os.path.join(_P51, 'campaign_spec_s51_f2_phase_b_5ce295e1.json'),
            'spec_sha256': '5ce295e17c5165ea2148d1240c918ebca05b8aafbeeb22e2ebfc03f9bf120094',
            'results_sha256': '2cdd5d6048919980b6d3d7109709b3441a99b5eb353be3285a09210a9a996eb7',
            'variant': 'A', 'script': 'p515_s51_f2_phase_b.py'},
    's53': {'results': os.path.join(_P53, 'campaign_results.json'),
            'spec': os.path.join(_P53, 'campaign_spec_s53_f2_certificate_r1_803571c0.json'),
            'spec_sha256': '803571c0efcf828eb9716d6c6255909da91d882a57a7bd4e053805f96f88027f',
            'results_sha256': '61b9c501a2dc3a957fe06246c0048bfc4c61798626dc3a91783bd463602f4a94',
            'variant': 'B', 'script': 'p515_s53_f2_certificate.py'},
}
FROZEN_TABLES = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen',
                             'frozen_step6_tables_v1_590088fe.json')
FROZEN_TABLES_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
W2_COSTS = PB.INVESTMENT_COST_RESULTS['path']

# code locations, found by exact text (file, needle); every needle must be found at least once
CODE_NEEDLES = [
    ('p515_s47_phase_b_record.py', 'DELTA_0 = 4'),
    ('p515_s47_phase_b_record.py', 'DELTA_MIN = 1'),
    ('p515_s47_phase_b_record.py', "POLL_DESIGN = 'orthomads_n_plus_1_neg'"),
    ('p515_s47_phase_b_record.py', 'MAX_NEW_EVALUATIONS = 20'),
    ('p515_s47_phase_b_record.py', 'MAX_POLLS = 60'),
    ('p515_s47_phase_b_record.py', 'COMPLETION_CAP = 30'),
    ('p515_s47_phase_b_record.py', 'def householder_columns(t, n=N_VARS):'),
    ('p515_s47_phase_b_record.py', 'def poll_directions(k, delta, n=N_VARS, t0=HALTON_T0, design=POLL_DESIGN):'),
    ('p515_s47_phase_b_record.py', 'neg = [-sum(c[i] for c in cols) for i in range(n)]'),
    ('p515_s47_phase_b_record.py', 'return tuple(_round_half_away(delta * a / m) for a in h)'),
    ('p515_s47_phase_b_record.py', 'delta = int(delta0)'),
    ('p515_s47_phase_b_record.py',
     "completion = lattice.completion(tuple(inc['z'])) if (UNIT_POLL_COMPLETION and delta == DELTA_MIN) else None"),
    ('p515_s47_phase_b_record.py', "if completion is not None and completion['over_cap']:"),
    ('p515_s47_phase_b_record.py', 'if n_new + len(new) > max_new_evaluations:'),
    ('p515_s47_phase_b_record.py', 'for k in range(max_polls):'),
    ('p515_s47_phase_b_record.py', 'delta *= 2'),
    ('p515_s47_phase_b_record.py', 'elif delta == DELTA_MIN:'),
    ('p515_s47_phase_b_record.py', 'delta = max(DELTA_MIN, delta // 2)'),
    ('p515_s47_phase_b_record.py', 'def neighbourhood(self, z_inc):'),
    ('p515_s51_f2_phase_b.py', 'import p515_s47_phase_b_record as PB'),
    ('p515_s51_f2_phase_b.py', 'result = PB.run_mads(lattice, cache, key_of, inc, evaluate_fn, sigma_q, on_poll=on_poll,'),
    ('p515_s51_f2_phase_b.py', "'pb_delta0_4': PB.DELTA_0 == 4"),
    ('p515_s53_f2_certificate.py', 'N_DIRECTIONS = 2 * N_VARS'),
    ('p515_s53_f2_certificate.py', 'MIN_FEASIBLE_POLL_POINTS = N_VARS + 1'),
    ('p515_s53_f2_certificate.py', "POLL_DESIGN = 'orthomads_2n'"),
    ('p515_s53_f2_certificate.py', 'DELTA_UNIT = PB.DELTA_MIN'),
    ('p515_s53_f2_certificate.py', 'MAX_NEW_EVALUATIONS = 60'),
    ('p515_s53_f2_certificate.py', 'FIRST_HALTON_K = 4'),
    ('p515_s53_f2_certificate.py', 'def directions_2n(k, delta=DELTA_UNIT):'),
    ('p515_s53_f2_certificate.py', 'def snap_key(lattice, z, r, z_inc):'),
    ('p515_s53_f2_certificate.py', "SNAP_KEY_LEVELS = ('l1_to_rounded', 'linf_to_rounded', 'l1_to_incumbent', 'same_year_as_incumbent', 'lower_I',"),
    ('p515_s53_f2_certificate.py', 'triggered = n_distinct < MIN_FEASIBLE_POLL_POINTS'),
    ('p515_s53_f2_certificate.py', 'def certificate_scope(lattice, key_of, cache, inc, record, sigma_q):'),
]

# manuscript text to locate (exact substrings of main.tex)
TEX_NEEDLES = {
    'alg_init_delta0': r'$\Delta \gets \Delta_0$; $N \gets 0$',
    'alg_while': r'\While{$N < N^{\max}$}',
    'alg_variantA_poll': r'the $n+1$ OrthoMADS points $\boldsymbol{z}^{\text{inc}} + \Delta \boldsymbol{v}$',
    'alg_completion': r'\If{$\Delta = 1$}',
    'alg_cap': r'\lIf{$|\mathcal{P}| > 30$}',
    'alg_variantB': r'variant B, $\Delta = 1$ throughout',
    'alg_variantB_completion': r'add admissible unit neighbours until $n+1$',
    'alg_budget': r'\lIf{the new points of $\mathcal{P}$ exceed $N^{\max} - N$}',
    'alg_success': r'$\Delta \gets 2\Delta$ (variant A) or $1$ (variant B)',
    'alg_failure_terminate': r'\lIf{$\Delta = 1$}{\textbf{terminate}',
    'alg_halve': r'$\Delta \gets \Delta/2$',
    'confirm_438': '% [CONFIRM — W173] Delta_0',
    'confirm_446': '% [CONFIRM — W173] "fourteen"',
    'sentence_scope_start': 'What the search certifies is the scope of its final unit poll',
    'sentence_fourteen': 'every admissible unit neighbour was evaluated (fourteen, the full box;',
    'sentence_twelve': 'the final poll evaluated twelve neighbours, which do not positively span the space, and the plan is '
                       'reported as better than each of them, seven determinately and five within resolution, not as a '
                       'mesh-local optimum.',
}

FAILED = []


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _check(name, ok, detail=None):
    if not ok:
        FAILED.append({'check': name, 'detail': detail})
    return bool(ok)


def _tracked_clean(path):
    tracked = bool(_git('ls-files', '--', path))
    clean = _git('status', '--porcelain', '--', path) == ''
    return {'git_tracked': tracked, 'git_clean': clean}


def code_locations():
    out = []
    cache = {}
    for rel, needle in CODE_NEEDLES:
        if rel not in cache:
            with open(rel, encoding='utf-8') as fh:
                cache[rel] = fh.read().split('\n')
        lines = [i + 1 for i, ln in enumerate(cache[rel]) if needle in ln]
        _check(f'code needle found: {rel}: {needle}', bool(lines))
        out.append({'file': rel, 'text': needle, 'lines': lines})
    return out


def tex_locations(tex_lines):
    out = {}
    joined = '\n'.join(tex_lines)
    for key, needle in TEX_NEEDLES.items():
        lines = [i + 1 for i, ln in enumerate(tex_lines) if needle in ln]
        _check(f'tex needle found: {key}', bool(lines) or needle in joined, needle)
        out[key] = {'text': needle, 'lines': lines}
    return out


def norm_canonical(can):
    """{'investment_year', 'nodes'} -> the canonical z of the search lattice (missing nodes = 0)."""
    nodes = {str(n): list((can.get('nodes') or {}).get(str(n), [0.0, 0.0])) for n in PB.ACTIVE_NODES}
    return {'investment_year': int(can['investment_year']), 'nodes': nodes}


def z_of(lattice, can):
    return lattice.z_of_canonical(norm_canonical(can))


def linf(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def sym_class(diff_f_minus_finc, res):
    """positive diff = the neighbour is worse."""
    if diff_f_minus_finc is None or res is None:
        return 'barrier_or_unresolved'
    if diff_f_minus_finc > res:
        return 'worse'
    if -diff_f_minus_finc > res:
        return 'better'
    return 'within_resolution'


# ======================================================================================================================
#  item 1
# ======================================================================================================================
def item1(res, specs, lattice):
    out = {}
    for name, cfg in RUNS.items():
        r, s = res[name], specs[name]
        pd = (s.get('extra') or {}).get('poll_design')
        polls = []
        for p in r['poll_history']:
            disp = Counter(c['disposition'] for c in p['candidates'])
            rec = {'poll_index': p['poll_index'], 'halton_t': p['halton_t'], 'delta': p['poll_size_delta'],
                   'incumbent': p['incumbent']['label'], 'dispositions': dict(disp),
                   'n_new_evaluations': p['n_new_evaluations'], 'n_cache_hits': p['n_cache_hits'],
                   'completion_n_feasible': (p.get('completion') or {}).get('n_feasible'),
                   'decision': p['decision'], 'next_incumbent': p.get('next_incumbent'),
                   'next_poll_size': p.get('next_poll_size'),
                   'n_direction_points_feasible': sum(1 for c in p['candidates']
                                                      if c['poll_part'].startswith('direction')
                                                      and c['disposition'] in ('cache_hit', 'new_evaluation'))}
            # recompute the directions with the production functions and compare with the record
            if cfg['variant'] == 'A':
                t, _u, dirs = PB.poll_directions(p['poll_index'], p['poll_size_delta'])
            else:
                t, _u, dirs = S53.directions_2n(p['poll_index'])
            rec['directions_recomputed_equal_record'] = (t == p['halton_t']
                                                        and [list(d) for d in dirs] == p['directions'])
            _check(f'{name} poll {p["poll_index"]}: directions recomputed = recorded',
                   rec['directions_recomputed_equal_record'])
            rec['n_directions'] = len(p['directions'])
            if cfg['variant'] == 'B':
                rec['snap_decided_by'] = dict(Counter(f"{sn.get('result')}:{sn.get('decided_by')}"
                                                      for sn in p['snap_table']))
                rec['n_distinct_feasible_poll_points'] = p.get('n_distinct_feasible_poll_points')
                rec['completion_triggered'] = p.get('completion_triggered')
            polls.append(rec)
        out[name] = {
            'variant': cfg['variant'], 'script': cfg['script'],
            'spec': {'path': cfg['spec'], 'sha256': specs[name + '_sha'], 'field': 'extra.poll_design',
                     'poll_design': pd},
            'spec_iteration_rule': (s.get('extra') or {}).get('iteration_rule'),
            'spec_poll_rule': (s.get('extra') or {}).get('poll_rule'),
            'spec_snap_rule': (s.get('extra') or {}).get('snap_rule'),
            'results': {'path': cfg['results'], 'sha256': specs[name + '_res_sha']},
            'delta_sequence': [p['delta'] for p in polls],
            'polls': polls, 'termination': r['termination'],
            'n_new_evaluations_total': r['n_new_evaluations'],
            'max_new_evaluations': (pd or {}).get('max_new_evaluations'),
        }
    return out


# ======================================================================================================================
#  item 2
# ======================================================================================================================
def table_rows(tables, lattice):
    """Every frozen-table row with a canonical candidate: cells, phase_b rows, claims (with both cells)."""
    cells = tables['cells']
    by_z = {}
    for cid, c in cells.items():
        if c.get('candidate_canonical'):
            by_z.setdefault(z_of(lattice, c['candidate_canonical']), []).append(('cells', cid))
    for row in tables['phase_b']:
        if row.get('candidate_canonical'):
            by_z.setdefault(z_of(lattice, row['candidate_canonical']), []).append(('phase_b', row['cell']))
    return by_z


def claims_for(tables, ref_cells, other_z, lattice):
    out = []
    for cl in tables['claims']:
        if cl.get('ref_cell') not in ref_cells:
            continue
        inst = (cl.get('instance') or {}).get(cl.get('other_cell')) or {}
        can = inst.get('candidate_canonical')
        if can is None:
            can = (tables['cells'].get(cl.get('other_cell')) or {}).get('candidate_canonical')
        if can is None:
            continue
        if z_of(lattice, can) == other_z:
            out.append({k: cl.get(k) for k in ('claim_id', 'family', 'ref_cell', 'other_cell', 'ref_status',
                                               'other_status', 'd_gross', 'gross_rule', 'gross_threshold_or_bar',
                                               'gross_multiple', 'gross_verdict', 'd_Qcc_report_only',
                                               'Qcc_verdict_report_only')})
    return out


def scan_all_campaigns(targets, lattice, flex_m):
    """Every git-tracked campaign_results.json under data/SRP1/Results: points at a target canonical candidate whose
    flexibility-price multiplier equals flex_m (point field, else top-level field, else 1.0). Returns hits per z."""
    tracked = _git('ls-files', 'data').split('\n')
    files = [f for f in tracked if f.startswith('data/SRP1/Results/') and f.endswith('campaign_results.json')]
    # every evaluation_record.json on disk under data/ (one per evaluation directory), tracked or not
    rec_files = sorted(os.path.relpath(os.path.join(dp, 'evaluation_record.json'), REPO)
                       for dp, _d, fs in os.walk('data') if 'evaluation_record.json' in fs)
    rec_tracked = set(f for f in tracked if f.endswith('evaluation_record.json'))
    hits = {z: [] for z in targets}
    n_points = 0
    for f in rec_files:
        try:
            d = _load(f)
        except Exception as exc:  # noqa: BLE001
            _check(f'scan: {f} loads', False, repr(exc))
            continue
        can = d.get('candidate_canonical')
        if not can or 'investment_year' not in can:
            continue
        n_points += 1
        m = d.get('flex_price_multiplier')
        m = 1.0 if m is None else float(m)
        if m != flex_m:
            continue
        try:
            z = z_of(lattice, can)
        except Exception:  # noqa: BLE001
            continue
        if z in hits:
            hits[z].append({'file': f, 'label': d.get('candidate_label'), 'status': d.get('status'),
                            'eval_key': None, 'kind': 'evaluation_record'})
    for f in files:
        try:
            d = _load(f)
        except Exception as exc:  # noqa: BLE001
            _check(f'scan: {f} loads', False, repr(exc))
            continue
        top_m = d.get('flex_price_multiplier')
        pts = d.get('points')
        items = pts.items() if isinstance(pts, dict) else enumerate(pts or [])
        for _k, p in items:
            if not isinstance(p, dict):
                continue
            can = p.get('candidate_canonical') or p.get('canonical')
            if not can or 'investment_year' not in can:
                continue
            n_points += 1
            m = p.get('flex_price_multiplier', top_m)
            m = 1.0 if m is None else float(m)
            if m != flex_m:
                continue
            try:
                z = z_of(lattice, can)
            except Exception:  # noqa: BLE001  (off-lattice candidate)
                continue
            if z in hits:
                hits[z].append({'file': f, 'label': p.get('label'), 'status': p.get('status'),
                                'eval_key': p.get('eval_key'), 'kind': 'campaign_results_point'})
    return {'n_files': len(files), 'n_evaluation_records_on_disk': len(rec_files),
            'n_evaluation_records_tracked': len(rec_tracked),
            'n_evaluation_records_on_disk_untracked': sum(1 for f in rec_files if f not in rec_tracked),
            'n_points_with_canonical': n_points, 'hits': hits}


def x0_certificate(res47, lattice, tables, by_z):
    r = res47
    cert = r['termination_certificate']
    last = r['poll_history'][-1]
    inc_z = tuple(r['final_incumbent']['z'])
    box = lattice.neighbourhood(inc_z)
    recorded = [e['label'] for e in r['lattice_neighbourhood_of_incumbent']]
    _check('x0: recomputed box labels = recorded lattice_neighbourhood_of_incumbent',
           sorted(lattice.label(z) for z in box) == sorted(recorded))
    polled = [c for c in last['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]
    pol_by_label = {c['label']: c for c in polled}
    rows = []
    for z in box:
        lab = lattice.label(z)
        c = pol_by_label.get(lab)
        diff = None if c is None or c['F_inc_minus_F_eur'] is None else -c['F_inc_minus_F_eur']
        ref = by_z.get(z, [])
        claims = claims_for(tables, {'ref:7aa017f0'}, z, lattice)
        pb_rows = [row for row in tables['phase_b'] if row.get('candidate_canonical')
                   and z_of(lattice, row['candidate_canonical']) == z]
        rows.append({'label': lab, 'z': list(z), 'in_final_poll': c is not None,
                     'disposition': None if c is None else c['disposition'],
                     'search_F_minus_F_inc_eur': diff,
                     'search_resolution_eur': None if c is None else c['resolution_eur'],
                     'search_outcome_A4': None if c is None else c['outcome'],
                     'search_symmetric_class': sym_class(diff, None if c is None else c['resolution_eur']),
                     'frozen_table_rows': ref,
                     'frozen_phase_b': [{k: row.get(k) for k in ('cell', 'status', 'superseded', 'M_gross',
                                                                 'v6_threshold', 'v6_multiple', 'v6_verdict')}
                                        for row in pb_rows],
                     'frozen_claims_vs_x0': claims})
    n_ev = sum(1 for x in rows if x['in_final_poll'])
    resettled = [x for x in rows if x['frozen_claims_vs_x0'] or
                 any(not p['superseded'] for p in x['frozen_phase_b'])]
    return {
        'incumbent': r['final_incumbent']['label'], 'incumbent_z': list(inc_z),
        'certificate_record': {'path': RUNS['s47']['results'], 'field': 'termination_certificate', 'value': cert},
        'a_final_poll': {'poll_index': last['poll_index'], 'delta': last['poll_size_delta'],
                         'n_points_listed': len(last['candidates']),
                         'n_directions_rejected_infeasible': sum(1 for c in last['candidates']
                                                                 if c['disposition'] == 'rejected_infeasible'),
                         'n_poll_set_feasible_distinct': len(polled),
                         'n_evaluated_new': sum(1 for c in polled if c['disposition'] == 'new_evaluation'),
                         'n_cache_hits': sum(1 for c in polled if c['disposition'] == 'cache_hit')},
        'b_box': {'n_feasible': len(box), 'n_evaluated': n_ev, 'n_not_evaluated': len(box) - n_ev,
                  'box_equals_final_poll_set': sorted(pol_by_label) == sorted(lattice.label(z) for z in box)},
        'c_search_time': dict(Counter(x['search_symmetric_class'] for x in rows)),
        'c_resettled_in_frozen_tables': {
            'n': len(resettled),
            'labels': [x['label'] for x in resettled],
            'verdicts': [{'label': x['label'],
                          'phase_b': [p for p in x['frozen_phase_b'] if not p['superseded']],
                          'claims': [{k: cl[k] for k in ('claim_id', 'd_gross', 'gross_threshold_or_bar',
                                                         'gross_verdict')} for cl in x['frozen_claims_vs_x0']]}
                         for x in resettled]},
        'rows': rows,
    }


def f2_certificate(res53, lattice, tables, by_z, scan):
    r = res53
    cert = r['termination_certificate']
    inc_z = tuple(r['final_incumbent']['z'])
    box = lattice.neighbourhood(inc_z)
    recorded = {e['label']: e for e in r['lattice_neighbourhood_of_incumbent']}
    _check('F2: recomputed box labels = recorded lattice_neighbourhood_of_incumbent',
           sorted(lattice.label(z) for z in box) == sorted(recorded))
    _check('F2: box size = certificate scope n_feasible',
           len(box) == cert['box_neighbourhood']['n_feasible'])
    poll = {e['label']: e for e in cert['poll_set']}
    cached = {e['label']: e for e in cert['cached_box_neighbours']['rows']}
    rows = []
    for z in box:
        lab = lattice.label(z)
        if lab in poll:
            e = poll[lab]
            where, diff, rs = 'final_poll_set', -e['F_inc_minus_F_eur'], e['resolution_eur']
        elif lab in cached:
            e = cached[lab]
            where, diff, rs = 'cached_outside_poll_set', e['F_minus_F_inc_eur'], e['resolution_eur']
        else:
            where, diff, rs = 'not_evaluated_in_search', None, None
        claims = claims_for(tables, {'ref:5ca4f86c'}, z, lattice)
        rows.append({'label': lab, 'z': list(z), 'where': where, 'in_cache_recorded': recorded[lab]['in_cache'],
                     'polled_in_final_poll_recorded': recorded[lab]['polled_in_final_poll'],
                     'search_F_minus_F_inc_eur': diff, 'search_resolution_eur': rs,
                     'search_symmetric_class': sym_class(diff, rs) if diff is not None else None,
                     'frozen_table_rows': by_z.get(z, []), 'frozen_L_claims': claims,
                     'other_campaign_evaluations_at_m2': scan['hits'].get(z, [])})
        _check(f'F2 {lab}: evaluated-in-search agrees with recorded in_cache',
               (where != 'not_evaluated_in_search') == bool(recorded[lab]['in_cache']))
    evaluated = [x for x in rows if x['where'] != 'not_evaluated_in_search']
    not_ev = [x for x in rows if x['where'] == 'not_evaluated_in_search']
    not_ev_but_elsewhere = [x['label'] for x in not_ev if x['other_campaign_evaluations_at_m2']]
    # L claims against the incumbent: neighbours and non-neighbours
    l_claims = [cl for cl in tables['claims'] if cl.get('family') == 'L']
    l_detail = []
    for cl in l_claims:
        inst = (cl.get('instance') or {}).get(cl.get('other_cell')) or {}
        can = inst.get('candidate_canonical') or (tables['cells'].get(cl.get('other_cell')) or {}).get(
            'candidate_canonical')
        z = z_of(lattice, can)
        l_detail.append({'claim_id': cl['claim_id'], 'other_cell': cl['other_cell'], 'label': lattice.label(z),
                         'linf_to_incumbent': linf(z, inc_z), 'is_box_neighbour': z in set(box),
                         'd_gross': cl['d_gross'], 'gross_threshold_or_bar': cl['gross_threshold_or_bar'],
                         'gross_multiple': cl['gross_multiple'], 'gross_verdict': cl['gross_verdict'],
                         'd_Qcc_report_only': cl['d_Qcc_report_only'],
                         'Qcc_verdict_report_only': cl['Qcc_verdict_report_only']})
    l_nb = [x for x in l_detail if x['is_box_neighbour']]
    l_cells = sorted(k for k in tables['cells'] if k.startswith('l_'))
    l_cells_nb = [k for k in l_cells if z_of(lattice, tables['cells'][k]['candidate_canonical']) in set(box)]

    def _tally(rows_):
        pos = [x for x in rows_ if x['d_gross'] > 0]
        neg = [x for x in rows_ if x['d_gross'] <= 0]
        return {'n': len(rows_), 'n_positive_plan_better': len(pos),
                'n_positive_determinate': sum(1 for x in pos if x['gross_verdict'] == 'determinate'),
                'n_positive_within_bar': sum(1 for x in pos if x['gross_verdict'] != 'determinate'),
                'n_negative_neighbour_better': len(neg),
                'negative': [{k: x[k] for k in ('label', 'other_cell', 'd_gross', 'gross_threshold_or_bar',
                                                'gross_verdict')} for x in neg]}

    # the Addendum 63 "12 neighbours" = the twelve l_ cells (T1)
    l_cell_claims = [x for x in l_detail if x['other_cell'] in l_cells]
    evaluated_not_in_L = [x['label'] for x in evaluated if not x['frozen_L_claims']]
    return {
        'incumbent': r['final_incumbent']['label'], 'incumbent_z': list(inc_z),
        'certificate_record': {'path': RUNS['s53']['results'], 'field': 'termination_certificate',
                               'scope_claim': cert['scope']['claim'], 'holds': cert['holds'],
                               'poll_set_spanning': cert['spanning'],
                               'box_neighbourhood_counts': {k: cert['box_neighbourhood'][k] for k in
                                                            ('n_feasible', 'n_in_poll_set', 'n_outside_poll_set')},
                               'cached_box_neighbours_counts': cert['cached_box_neighbours']['counts_by_class']},
        'a_final_poll': {'poll_index': cert['poll_index'], 'halton_t': cert['halton_t'],
                         'n_poll_points': cert['n_poll_points'],
                         'n_evaluated_new': sum(1 for e in cert['poll_set'] if e['disposition'] == 'new_evaluation'),
                         'n_cache_hits': sum(1 for e in cert['poll_set'] if e['disposition'] == 'cache_hit'),
                         'search_classes': dict(Counter(e['comparison_vs_incumbent'] for e in cert['poll_set'])),
                         'completion_triggered': cert['completion_triggered']},
        'b_box': {'n_feasible': len(box), 'n_evaluated_in_search': len(evaluated),
                  'n_in_final_poll_set': sum(1 for x in rows if x['where'] == 'final_poll_set'),
                  'n_cached_outside_poll_set': sum(1 for x in rows if x['where'] == 'cached_outside_poll_set'),
                  'n_not_evaluated_in_search': len(not_ev),
                  'not_evaluated_but_found_in_another_m2_campaign': not_ev_but_elsewhere,
                  'scan_scope': {'n_campaign_results_files_tracked': scan['n_files'],
                                 'n_evaluation_records_on_disk': scan['n_evaluation_records_on_disk'],
                                 'n_evaluation_records_tracked': scan['n_evaluation_records_tracked'],
                                 'n_evaluation_records_on_disk_untracked':
                                     scan['n_evaluation_records_on_disk_untracked'],
                                 'n_points_with_canonical': scan['n_points_with_canonical'],
                                 'match': 'same canonical candidate on the search lattice and '
                                          'flex_price_multiplier == 2.0 (point field, else top-level, else 1.0)'}},
        'c_search_time_evaluated': dict(Counter(x['search_symmetric_class'] for x in evaluated)),
        'c_L_claims_all': _tally(l_detail),
        'c_L_claims_box_neighbours': _tally(l_nb),
        'c_L_claims_twelve_l_cells': _tally(l_cell_claims),
        'L_claims_not_box_neighbours': [x for x in l_detail if not x['is_box_neighbour']],
        'l_cells_T1': l_cells, 'l_cells_that_are_box_neighbours': l_cells_nb,
        'evaluated_box_neighbours_without_L_claim': [
            {k: x[k] for k in ('label', 'where', 'search_F_minus_F_inc_eur', 'search_resolution_eur',
                               'search_symmetric_class', 'frozen_table_rows')}
            for x in evaluated if not x['frozen_L_claims']],
        'L_claims_detail': l_detail,
        'rows': rows,
        'evaluated_not_in_L_labels': evaluated_not_in_L,
    }


def main():
    t0 = datetime.now(timezone.utc)
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MAN):
        _log(f'[W173] REFUSED: {OUT_DIR} already holds an output; not overwritten')
        sys.exit(1)
    inputs = [MAIN_TEX, FROZEN_TABLES, W2_COSTS, 'p515_s47_phase_b_record.py', 'p515_s51_f2_phase_b.py',
              'p515_s53_f2_certificate.py', 'p513_solve_profile_guard.py']
    for cfg in RUNS.values():
        inputs += [cfg['results'], cfg['spec']]
    before = {p: _sha(p) for p in inputs}
    integrity = {}
    integrity['main_tex_sha256'] = _check('main.tex sha256 = declared', before[MAIN_TEX] == MAIN_TEX_SHA,
                                          before[MAIN_TEX])
    head = subprocess.run(['git', '-C', CLONE, 'rev-parse', '--short=7', 'HEAD'], capture_output=True, text=True
                          ).stdout.strip()
    integrity['clone_head'] = _check('clone HEAD = 407f8df', head == CLONE_HEAD, head)
    integrity['frozen_tables_sha256'] = _check('frozen tables sha256', before[FROZEN_TABLES] == FROZEN_TABLES_SHA)
    specs, res = {}, {}
    for name, cfg in RUNS.items():
        _check(f'{name} spec sha256', before[cfg['spec']] == cfg['spec_sha256'], before[cfg['spec']])
        _check(f'{name} results sha256', before[cfg['results']] == cfg['results_sha256'], before[cfg['results']])
        for p in (cfg['spec'], cfg['results']):
            st = _tracked_clean(p)
            _check(f'{p} tracked and clean', st['git_tracked'] and st['git_clean'], st)
        specs[name] = _load(cfg['spec'])
        specs[name + '_sha'] = before[cfg['spec']]
        specs[name + '_res_sha'] = before[cfg['results']]
        res[name] = _load(cfg['results'])
    st = _tracked_clean(FROZEN_TABLES)
    _check('frozen tables tracked and clean', st['git_tracked'] and st['git_clean'], st)

    w2 = _load(W2_COSTS)
    costs = PB.unit_costs_from_w2(w2)
    lattice = PB.Lattice(tuple(sorted(costs)), costs)
    w2x = PB.w2_cross_check(lattice, w2)
    _check('I(x) closed form = W2 table', w2x['ok'], w2x)

    with open(MAIN_TEX, encoding='utf-8') as fh:
        tex_lines = fh.read().split('\n')
    tex = tex_locations(tex_lines)
    scope_line = tex['sentence_scope_start']['lines']
    sentence = None
    if scope_line:
        ln = tex_lines[scope_line[0] - 1]
        i = ln.index(TEX_NEEDLES['sentence_scope_start'])
        j = ln.index('No global optimality')
        sentence = ln[i:j].strip()

    tables = _load(FROZEN_TABLES)['tables']
    by_z = table_rows(tables, lattice)

    out1 = item1(res, specs, lattice)
    f2_box = lattice.neighbourhood(tuple(res['s53']['final_incumbent']['z']))
    scan = scan_all_campaigns(set(f2_box), lattice, 2.0)
    x0 = x0_certificate(res['s47'], lattice, tables, by_z)
    f2 = f2_certificate(res['s53'], lattice, tables, by_z, scan)
    # s51's final incumbent = s53's incumbent; the box s51 refused is the same set
    _check('s51 refused completion n_feasible = s53 box size',
           res['s51']['termination'].get('completion_n_feasible') == len(f2_box))

    after = {p: _sha(p) for p in inputs}
    _check('inputs unchanged during the run', after == before)

    gfail = GUARD.verify(0)
    others = []
    for mname, mod in list(sys.modules.items()):
        if not (mname.startswith('p51') or mname == '__main__'):
            continue  # project launchers only (lazy third-party modules raise on getattr)
        for attr in ('PARENT_GUARD', 'GUARD'):
            g = getattr(mod, attr, None)
            if isinstance(g, SolveProfileGuard) and g is not GUARD:
                others.append({'module': mname, 'attr': attr, 'label': g.label, 'verify_0': g.verify(0)})
    if gfail or any(o['verify_0'] for o in others) or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        _log(f'[W173 GUARD FAULT] {gfail} {others} pickle {PICKLE_COUNTS}')
        sys.exit(1)

    result = {
        'stage': 'P5.15 Addendum 69 W173 -- section 2.1 CONFIRM points: search frame (Delta_0, frame rule, N^max, '
                 'directions, snap) and the neighbour counts of the x = 0 and F2 certificates',
        'script': os.path.basename(__file__), 'script_sha256': _sha(__file__),
        'git_HEAD': _git('rev-parse', 'HEAD'), 'interpreter': sys.executable,
        'started_utc': t0.isoformat(), 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'objective_convention': 'F = I + Q, Q = certified gross_operational_cost (settlement excluded); search-time '
                                'differences use the record Q the search read; frozen-table differences are d_gross '
                                '(F(other) - F(ref), positive = the neighbour is worse) under the table rule',
        'box_definition': 'PB.Lattice.neighbourhood(inc): every bound-, rule- (P = 0 <=> E = 0; 2 h <= E/P <= 4 h; '
                          'E <= 5 MWh; year index in range) and budget- (I(x) <= 1e6 EUR) feasible canonical lattice '
                          'point z != inc with ||z - inc||_inf <= 1 in (zP5, zE5, zP7, zE7, zP9, zE9, zY); '
                          'granules 0.25 MVA, 0.5 MWh, one year index = 5 years',
        'instances': {'x0': {'label': x0['incumbent'], 'z': x0['incumbent_z'],
                             'eval_key': res['s47']['final_incumbent']['eval_key']},
                      'F2': {'label': f2['incumbent'], 'z': f2['incumbent_z'],
                             'eval_key': res['s53']['final_incumbent']['eval_key'], 'flex_price_multiplier': 2.0}},
        'item1': out1, 'item2': {'x0': x0, 'F2': f2},
        'manuscript': {'clone': CLONE, 'head': head, 'main_tex_sha256': before[MAIN_TEX], 'locations': tex,
                       'sentence_l443_scope': sentence},
        'code_locations': code_locations(),
        'I_x_cross_check_vs_W2': w2x,
        'integrity': {'failures': FAILED, 'inputs_sha256': before},
        'guards': {'own': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                   'imported_parent_guards': others, 'pickle_counts': PICKLE_COUNTS},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    man = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'inputs': after}
    with open(OUT_MAN, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'[W173] x0: final poll {x0["a_final_poll"]["n_poll_set_feasible_distinct"]} feasible points; box '
         f'{x0["b_box"]["n_feasible"]}, evaluated {x0["b_box"]["n_evaluated"]}; search classes {x0["c_search_time"]}; '
         f're-settled in frozen tables {x0["c_resettled_in_frozen_tables"]["n"]}')
    _log(f'[W173] F2: final poll {f2["a_final_poll"]["n_poll_points"]} points; box {f2["b_box"]["n_feasible"]}, '
         f'evaluated {f2["b_box"]["n_evaluated_in_search"]} ({f2["b_box"]["n_in_final_poll_set"]} poll + '
         f'{f2["b_box"]["n_cached_outside_poll_set"]} cached), not evaluated {f2["b_box"]["n_not_evaluated_in_search"]}'
         f' (found elsewhere at m = 2: {f2["b_box"]["not_evaluated_but_found_in_another_m2_campaign"]})')
    _log(f'[W173] F2 L claims (box neighbours): {f2["c_L_claims_box_neighbours"]}')
    _log(f'[W173] F2 L claims (twelve l_ cells): {f2["c_L_claims_twelve_l_cells"]}')
    for name in RUNS:
        _log(f'[W173] {name}: Delta sequence {out1[name]["delta_sequence"]}; termination '
             f'{out1[name]["termination"]["reason"]}; new evaluations {out1[name]["n_new_evaluations_total"]} of '
             f'{out1[name]["max_new_evaluations"]}')
    _log(f'[W173] guards: own {GUARD.counts}, {len(others)} imported parent guards at 0; pickle {PICKLE_COUNTS}; '
         f'integrity failures {len(FAILED)}')
    for f in FAILED:
        _log(f'[W173 INTEGRITY FAILURE] {f}')
    sys.exit(3 if FAILED else 0)


if __name__ == '__main__':
    main()
